/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#pragma once

#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"
#include "mllee/train_trainer.hpp"

#include <cstdint>

/**
 * @brief Fixed-point autograd verification: torch.autograd.gradcheck, no heap.
 *
 * The checker runs the caller's real `ForwardHook` twice per checked lane.
 * Once through the Tape (analytic side: record, backward, read the Q15 lanes
 * the shipped backward kernels produce) and twice forward-only with the one
 * int8 parameter code under test nudged by plus and minus one legal code step
 * (numeric side: a central finite difference of the Q7 loss).  Because the
 * hook is the same function a Trainer step calls, a checked graph is the
 * training graph, not a reimplementation of it.
 *
 * The scaling, worked out once here because it is the whole design:
 *   - the loss value is Q7 (one unit = 1/128 nat; see train_loss.hpp), and the
 *     gradient lane domain is dL/d(real) * 2^15 where real = code * 258/32768
 *     (the convention `mse_backward` and friends implement: a lane of value L
 *     in Q7 per unit real is exactly L * 256 lane units);
 *   - therefore the numeric slope per real unit in Q7 is
 *         h_num  = (loss(code+1) - loss(code-1)) / 2,
 *     and the analytic slope in the same Q7 domain is
 *         h_ana  = lane / 256   (round-half-away, `div_round`);
 *   - a loss difference of ONE Q7 unit (the smallest the numeric side can
 *     resolve at all) maps to 256 lane units, i.e. it is representable in the
 *     Q15 gradient domain but with only 8 distinct numeric estimates below a
 *     lane of 2048 -- which is why the checker refuses to grade a lane whose
 *     |h_num| is under the documented resolution floor `_M_min_delta_q7`
 *     rather than pretending a 1-unit quotient is a measurement.
 * Relative error is then ratio_q15(|h_ana - h_num|, max(|h_num|, 1)), compared
 * against an explicit Q15 epsilon (default 0.25 = 8192: deliberately loose
 * against a fixed-point reference, and the tests additionally pin exact
 * agreement on paths where both sides are integer-identity).
 *
 * Saturation awareness: a lane is reported `unverifiable`, never failed, when
 *   - the code sits at the clamp edge (+/-127: no symmetric window exists
 *     inside the legal [-127, 127] codes);
 *   - the analytic lane sits on a Q15 rail (the backward clamped, so there is
 *     no true gradient to compare against);
 *   - either perturbed forward lands on the Q7 log sentinel or the loss rails
 *     (the difference came from a saturating narrowing, not from the slope);
 *   - |h_num| is below the resolution floor (quantisation, not a gradient).
 * A Frozen tensor or one with no gradient buffer is skipped wholesale, matching
 * requires_grad() == false in the autograd sense.
 *
 * Cost model: exactly 1 analytic forward+backward plus 2 forwards per checked
 * lane, so full verification is O(2 * elements) forwards -- `grad_check_passes`
 * is the budget query and spot-check mode (every Nth element, or an explicit
 * index list) is the lever.  Memory: the checker allocates nothing; the only
 * buffers are the caller's results array (`grad_check_bytes`) and whatever a
 * single forward already costs the pool, which is what a Trainer step costs
 * anyway, because the pool is reset after every forward this checker runs.
 */
namespace zzs::mllee::train
{
/** @brief Default relative-error tolerance, Q15 (0.25). */
inline constexpr int32_t _S_grad_check_eps_q15 = 8192;

/**
 * @brief Default numeric resolution floor in Q7: 8 units (1/16 nat of loss
 *         difference across the +/-1 code probe).  Below this a central
 *         difference is quantisation noise, so the lane is unverifiable.
 */
inline constexpr int32_t _S_grad_check_min_delta_q7 = 8;

/**
 * @brief A |loss| at or beyond this Q7 magnitude is treated as a rail hit.
 *        q7_log_q15's p <= 0 sentinel is _S_q15_min and one cross-entropy
 *        term caps near 1153, so anything past 16384 (128 nats) can only be
 *        a saturated narrowing, never a measurable slope.
 */
inline constexpr int32_t _S_grad_check_loss_rail_q7 = 16384;

/** @brief Largest explicit spot-check index list the driver accepts. */
inline constexpr uint32_t _S_grad_check_max_indices = 256u;

/** @name Checker knobs and verdicts
 *
 * The config is a POD so a device caller can keep it in .rodata; the result is
 * a POD because it is written into a caller-owned array (the only storage the
 * checker touches) and static_assert'd so the array sizing is portable.
 * @{
 */

/** @brief What to check, how tightly, and at what cost. */
struct GradCheckConfig
{
    /** @brief Relative-error tolerance in Q15 (lane-domain 1.0 = 32768). */
    int32_t _M_epsilon_rel_q15 = _S_grad_check_eps_q15;
    /** @brief Resolution floor for the numeric quotient, Q7. */
    int32_t _M_min_delta_q7 = _S_grad_check_min_delta_q7;
    /** @brief Spot-check every Nth element; 0 = every element. */
    uint32_t _M_spot_stride = 0u;
    /** @brief Explicit index list for this parameter (overrides the stride). */
    const uint32_t *_M_index_list = nullptr;
    uint32_t _M_index_count = 0u;
    uint32_t _M_pad = 0u;
};

static_assert (sizeof (GradCheckConfig) == (sizeof (void *) == 4u ? 24u : 32u),
               "grad-check config is six words, .rodata resident");

/** @brief Per-parameter verdict; caller zeroes it, the driver fills it once. */
struct GradCheckResult
{
    /** @brief Lanes actually graded (agreement or disagreement). */
    uint32_t _M_checked = 0u;
    /** @brief Lanes skipped as unverifiable (edges, rails, noise). */
    uint32_t _M_unverifiable = 0u;
    /** @brief Lanes whose analytic gradient missed the numeric estimate by
     *        more than the tolerance -- the count that fails the check. */
    uint32_t _M_disagreements = 0u;
    /** @brief Worst relative error seen, Q15. */
    int32_t _M_max_rel_err_q15 = 0;
    /** @brief Element index of that worst lane. */
    uint32_t _M_worst_index = 0u;
    /** @brief Forward passes consumed by this parameter (1 + 2 * checked). */
    uint32_t _M_forward_passes = 0u;
    /** @brief True when the parameter was skipped (frozen / no grad). */
    uint32_t _M_skipped = 0u;
    /** @brief pass flag: zero disagreements (a fully-unverifiable tensor passes). */
    uint32_t _M_passed = 0u;
};

static_assert (sizeof (GradCheckResult) == 32u,
               "grad-check verdict is eight words per parameter");

/**@}*/

/** @brief Bytes of results storage a caller must supply for @p params tensors. */
[[nodiscard]] inline constexpr uint32_t
grad_check_bytes (const uint32_t params) noexcept
{
    const uint64_t bytes = static_cast<uint64_t> (params) * sizeof (GradCheckResult);
    return bytes > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t> (bytes);
}

/**
 * @brief Forward passes a run costs: one recorded analytic pass over the whole
 *        graph plus two numerics per checked lane (spot-checking shrinks the
 *        lane count, never the constant).
 */
[[nodiscard]] inline constexpr uint32_t
grad_check_passes (const uint32_t checked_lanes) noexcept
{
    const uint64_t passes = 1ull + 2ull * checked_lanes;
    return passes > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t> (passes);
}

/** @brief True when a Q15 lane value sits on a saturation rail. */
[[nodiscard]] inline constexpr bool
grad_check_lane_railed (const int32_t lane_q15) noexcept
{
    return lane_q15 >= _S_q15_max || lane_q15 <= _S_q15_min;
}

/**
 * @brief Grade one parameter: analytic vs central-difference, in the Q7 slope
 *        domain.  Exposed so a test or a tuned caller can drive parameters in
 *        separate budget slices; grad_check is the whole-list form.
 *
 * @param context a StepContext whose tape has ALREADY run the recorded
 *        backward, so the parameter's gradient lanes hold the analytic side.
 * @return BAD_LAYOUT for a malformed context, otherwise the forward hook's own
 *         failure Status, which aborts the sweep (a hook that cannot run
 *         unperturbed cannot be trusted perturbed).
 */
[[nodiscard]] inline Status
grad_check_param (StepContext &context, const ForwardHook forward,
                  const Trainable &param, uint32_t param_index,
                  const GradCheckConfig &config,
                  GradCheckResult &result) noexcept
{
    Pool *const pool = context._M_pool;
    Tape *const tape = context._M_tape;
    if (pool == nullptr || tape == nullptr || forward == nullptr)
        return Status::BAD_LAYOUT;
    if (tape->is_open ()) /* the numeric side runs with the tape closed: a
                              recording forward would leak nodes into the
                              next probe and change every loss after it */
        return Status::INTERNAL;
    const uint32_t count = param._M_count;
    uint32_t checked = 0u;
    uint32_t unverifiable = 0u;
    uint32_t disagreements = 0u;
    int32_t worst_rel = 0;
    uint32_t worst_index = 0u;

    const auto selected = [&config] (const uint32_t index) noexcept
    {
        if (config._M_index_list != nullptr)
            {
                for (uint32_t k = 0; k < config._M_index_count; ++k)
                    if (config._M_index_list[k] == index)
                        return true;
                return false;
            }
        return config._M_spot_stride == 0u
                   || index % config._M_spot_stride == 0u;
    };

    const int32_t eps = config._M_epsilon_rel_q15 < 0
                            ? 0
                            : (config._M_epsilon_rel_q15 > _S_q15_max
                                   ? _S_q15_max
                                   : config._M_epsilon_rel_q15);
    const int32_t floor_q7 = config._M_min_delta_q7 < 0
                                 ? 0
                                 : config._M_min_delta_q7;

    for (uint32_t i = 0; i < count; ++i)
        {
            if (!selected (i))
                continue;
            const int32_t analytic = param.grad_at (i);
            const int32_t code = param.code_at (i);
            /* Edge codes: no symmetric probe inside [-127, 127]. */
            if (code >= 127 || code <= -127)
                {
                    ++unverifiable;
                    continue;
                }
            if (grad_check_lane_railed (analytic))
                {
                    ++unverifiable;
                    continue;
                }
            const int8_t original = param._M_code[i];
            param._M_code[i] = static_cast<int8_t> (code + 1);
            const Status up_status = forward (context);
            pool->reset ();
            if (!status_ok (up_status))
                {
                    param._M_code[i] = original;
                    return up_status;
                }
            const int32_t loss_plus = context._M_loss_q7;
            param._M_code[i] = static_cast<int8_t> (code - 1);
            const Status down_status = forward (context);
            pool->reset ();
            if (!status_ok (down_status))
                {
                    param._M_code[i] = original;
                    return down_status;
                }
            const int32_t loss_minus = context._M_loss_q7;
            param._M_code[i] = original;

            const int32_t magnitude_plus = loss_plus < 0 ? -loss_plus : loss_plus;
            const int32_t magnitude_minus
                = loss_minus < 0 ? -loss_minus : loss_minus;
            if (magnitude_plus >= _S_grad_check_loss_rail_q7
                || magnitude_minus >= _S_grad_check_loss_rail_q7)
                {
                    ++unverifiable; /* a saturating narrowing produced one of
                                       the endpoints, not the slope */
                    continue;
                }
            const int64_t difference = static_cast<int64_t> (loss_plus)
                                       - static_cast<int64_t> (loss_minus);
            const int32_t clamped = difference > 0x7FFFFFFFll
                                        ? 0x7FFFFFFF
                                        : (difference < -0x7FFFFFFFll
                                               ? -0x7FFFFFFF - 1
                                               : static_cast<int32_t> (difference));
            const int32_t h_num = div_round (clamped, 2);
            const int32_t h_num_magnitude = h_num < 0 ? -h_num : h_num;
            if (h_num_magnitude < floor_q7)
                {
                    ++unverifiable; /* below the Q7 resolution floor */
                    continue;
                }
            /* The same slope, analytic side: lane -> Q7 per unit real. */
            const int32_t h_ana = div_round (analytic, 256);
            const int32_t error = q15_abs (h_ana - h_num);
            const int32_t rel
                = ratio_q15 (error, h_num_magnitude); /* h_num_magnitude >= 1 */
            if (rel > worst_rel)
                {
                    worst_rel = rel;
                    worst_index = i;
                }
            ++checked;
            if (rel > eps)
                ++disagreements;
        }

    result._M_checked = checked;
    result._M_unverifiable = unverifiable;
    result._M_disagreements = disagreements;
    result._M_max_rel_err_q15 = worst_rel;
    result._M_worst_index = worst_index;
    result._M_skipped = 0u;
    result._M_passed = disagreements == 0u ? 1u : 0u;
    result._M_forward_passes = grad_check_passes (checked + unverifiable);
    (void)param_index; /* kept for caller-side attribution and future hooks */
    return Status::OK;
}

/**
 * @brief Full sweep over the parameter list of a step context.
 *
 * Sequence: validate; zero gradients; run ONE recorded forward+backward with
 * the loss scale pinned to unity (so the analytic lanes are the unscaled
 * gradients -- q15_mul (base, 32768) is exact, which is why unity is chosen
 * instead of the run's current scale); reset the tape and pool; then grade
 * every parameter against the numeric side with the tape closed.  Frozen or
 * gradient-less tensors are skipped with _M_skipped set and a passing verdict.
 *
 * @param results_out caller-owned array of context._M_param_count results
 *        (`grad_check_bytes`); not pre-zeroed by the driver -- every checked
 *        tensor gets all eight fields written, skipped tensors get the skip
 *        fields written.
 * @return BAD_LAYOUT for a malformed context, INTERNAL if the tape or pool
 *         cannot run the recorded side, otherwise the first hook failure.
 */
[[nodiscard]] inline Status
grad_check (StepContext &context, const ForwardHook forward,
            const GradCheckConfig &config,
            GradCheckResult *const results_out) noexcept
{
    if (forward == nullptr || results_out == nullptr
        || context._M_pool == nullptr || context._M_tape == nullptr
        || context._M_params == nullptr
        || context._M_param_count == 0u
        || context._M_param_count > _S_max_trainer_params)
        return Status::BAD_LAYOUT;
    if (config._M_index_list != nullptr
        && config._M_index_count > _S_grad_check_max_indices)
        return Status::OUT_OF_RANGE;
    for (uint32_t i = 0; i < context._M_param_count; ++i)
        if (!status_ok (context._M_params[i].validate ()))
            return Status::BAD_LAYOUT;

    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    if (tape.is_open ())
        tape.reset ();
    pool.reset ();

    /* Analytic side, once, with the loss scale pinned to unity. */
    for (uint32_t i = 0; i < context._M_param_count; ++i)
        context._M_params[i].zero_grad ();
    const Status begun = tape.begin ();
    if (!status_ok (begun))
        return begun;
    tape.set_loss_scale (_S_q15_one);
    context._M_loss_scale_q15 = _S_q15_one;
    const Status recorded = forward (context);
    if (!status_ok (recorded))
        {
            tape.reset ();
            return recorded;
        }
    if (!status_ok (context._M_loss.validate ())
        || context._M_loss._M_grad == nullptr)
        {
            tape.reset ();
            return Status::BAD_LAYOUT;
        }
    const Status back = tape.backward (context._M_loss);
    tape.reset (); /* closes the tape AND resets the pool: the numeric side
                      must see is_open() == false and an empty arena */
    if (!status_ok (back))
        return back;

    for (uint32_t i = 0; i < context._M_param_count; ++i)
        {
            const Trainable &param = context._M_params[i];
            GradCheckResult &result = results_out[i];
            if (!param.requires_grad () || param._M_grad == nullptr
                || param._M_count == 0u)
                {
                    result = GradCheckResult{};
                    result._M_skipped = 1u;
                    result._M_passed = 1u;
                    continue;
                }
            const Status status
                = grad_check_param (context, forward, param, i, config, result);
            if (!status_ok (status))
                return status;
        }
    return Status::OK;
}

/** @brief True when every non-skipped parameter passed the sweep. */
[[nodiscard]] inline bool
grad_check_all_passed (const GradCheckResult *const results,
                       const uint32_t count) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        if (results[i]._M_passed == 0u)
            return false;
    return true;
}
} // namespace zzs::mllee::train
