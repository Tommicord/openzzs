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
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Training twin of the inference NORM op, plus the residual add.
 *
 * The forward is the same normalization the inference kernel performs, but
 * evaluated in Q15 lanes instead of the code-domain shortcut, because the
 * backward needs a continuous deviation and a per-lane mean:
 *
 *   - a code widens with `code_to_q15` (code 127 == 32766), so a gamma or beta
 *     tensor is an ordinary int8 parameter with the module's usual scale:
 *     gamma code 127 is 0.99994, i.e. the deployment value 128 in Q7;
 *   - LayerNorm centers the window, RMSNorm does not, and the deviation is the
 *     integer root of the mean square in Q30 (`deviation` is then already Q15
 *     because sqrt (v / 2^30) * 2^15 == sqrt (v));
 *   - the output narrows through `q15_to_code`, the inverse code step.
 *
 * Gradient lanes follow the rest of the training path: a lane is dL/d(real) in
 * Q15, real = code / 127.  The analytic LayerNorm Jacobian
 *
 *   dx_i = (g_i gamma_i - mean(g gamma) - xhat_i mean(g gamma xhat)) / dev
 *
 * is evaluated with `q15_mul` / `q15_div`, dgamma_i = g_i xhat_i and dbeta_i =
 * g_i.  A zero deviation (a constant window) selects the exact-centre branch:
 * xhat = 0 and dx = 0, matching `_S_norm_value`'s divisor-zero branch, while
 * the bias still receives its gradient.
 *
 * The residual connection is the plain elementwise add of two code tensors
 * (`residual_run`), so a transformer block records ADD -> NORM as two nodes and
 * the backward routes dL/dx into both branches.  Inference may fuse the pair
 * into NORM_RESIDUAL to save an arena row; the training graph keeps them
 * separate so the tape owns each partial derivative.
 *
 * Everything is caller-owned: a run carves one stats slab and one lane scratch
 * from the tape's pool, and `NormSpec::bytes ()` is their exact capacity bound.
 */
namespace zzs::mllee::train
{
/** @brief Saved slot holding the spec copy the backward reads back. */
inline constexpr uint32_t _S_norm_spec = 0u;

/** @brief Saved slot holding the per-window mean/deviation pairs. */
inline constexpr uint32_t _S_norm_stats = 1u;

/** @brief Saved slot holding the backward's two lane rows. */
inline constexpr uint32_t _S_norm_scratch = 2u;

/** @brief Default variance floor, Q30 (zero: the exact-centre branch is used). */
inline constexpr uint32_t _S_norm_eps_q30 = 0u;

/** @brief Fraction width of the normalized lane: |xhat| can reach sqrt(L-1). */
inline constexpr uint32_t _S_norm_xhat_bits = 12u;

/**
 * @brief One window's forward statistics.
 *
 * @p _M_deviation is the Q15 standard deviation (an integer root, never
 * negative); zero marks a constant window.  Two int32 words keep the slab
 * 4-byte aligned and the struct trivially copyable.
 */
struct NormStats
{
    int32_t _M_mean = 0;
    int32_t _M_deviation = 0;
};

static_assert (sizeof (NormStats) == 8u, "one window state is two pool words");

/**
 * @brief LayerNorm / RMSNorm geometry and deployment flags.
 *
 * A window is `_M_window` channels (0 means the whole feature row), so the same
 * spec covers a per-token transformer norm and the grouped form of the
 * inference NORM_INSTANCE path.  `features` must be a multiple of the window.
 */
struct NormSpec
{
    uint16_t _M_batch = 1u;
    uint16_t _M_features = 0u;
    /** @brief Channels per normalized window; 0 selects the whole row. */
    uint16_t _M_window = 0u;
    /** @brief Non-zero when a gamma operand participates. */
    uint8_t _M_use_gamma = 1u;
    /** @brief Non-zero when a beta operand participates. */
    uint8_t _M_use_bias = 0u;
    /** @brief Non-zero selects RMSNorm (no mean subtraction). */
    uint8_t _M_rms = 0u;
    uint8_t _M_pad = 0u;
    /** @brief Variance floor added before the root, Q30. */
    uint32_t _M_eps_q30 = _S_norm_eps_q30;
    /** @brief Op id recorded on the tape. */
    uint32_t _M_op_id = 0u;

    [[nodiscard]] constexpr uint32_t
    window () const noexcept
    {
        return this->_M_window != 0u ? this->_M_window : this->_M_features;
    }

    [[nodiscard]] constexpr uint32_t
    windows_per_row () const noexcept
    {
        const uint32_t length = this->window ();
        return length != 0u ? static_cast<uint32_t> (this->_M_features) / length
                            : 0u;
    }

    [[nodiscard]] constexpr uint32_t
    windows () const noexcept
    {
        return static_cast<uint32_t> (this->_M_batch)
               * this->windows_per_row ();
    }

    [[nodiscard]] constexpr uint32_t
    elements () const noexcept
    {
        return static_cast<uint32_t> (this->_M_batch) * this->_M_features;
    }

    [[nodiscard]] constexpr Shape
    input_shape () const noexcept
    {
        return Shape::matrix (this->_M_batch, this->_M_features);
    }

    [[nodiscard]] constexpr Shape
    gamma_shape () const noexcept
    {
        return Shape::vector (this->_M_features);
    }

    /** @brief Pool bytes one recorded layer needs. */
    [[nodiscard]] constexpr uint32_t
    stats_bytes () const noexcept
    {
        return align_up (this->windows ()
                             * static_cast<uint32_t> (sizeof (NormStats)),
                         _S_pool_alignment);
    }

    [[nodiscard]] constexpr uint32_t
    scratch_bytes () const noexcept
    {
        return align_up (2u * this->window () * 4u, _S_pool_alignment);
    }

    [[nodiscard]] constexpr uint32_t
    bytes () const noexcept
    {
        return this->stats_bytes () + this->scratch_bytes ()
               + align_up (static_cast<uint32_t> (sizeof (NormSpec)),
                           _S_pool_alignment);
    }

    /**
     * @brief Check geometry and the boolean flag bytes.
     *
     * @return BAD_LAYOUT for a zero axis, a window that does not divide the
     *         feature axis, or a flag byte outside {0, 1}; OUT_OF_RANGE for a
     *         tensor a training view cannot address.
     */
    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (this->_M_batch == 0u || this->_M_features == 0u)
            return Status::BAD_LAYOUT;
        const uint32_t length = this->window ();
        if (length == 0u || this->_M_features % length != 0u)
            return Status::BAD_LAYOUT;
        if (this->_M_use_gamma > 1u || this->_M_use_bias > 1u
            || this->_M_rms > 1u)
            return Status::BAD_LAYOUT;
        if (this->elements () > _S_max_elements)
            return Status::OUT_OF_RANGE;
        return Status::OK;
    }
};

static_assert (sizeof (NormSpec) == 20u,
               "a saved norm spec is five pool words");

/**
 * @brief Check one operand view against the shape the spec expects.
 *
 * The count must match; the rank may be one lower only for a single row, i.e.
 * the [1, C] and [C] spellings of the same vector.
 */
[[nodiscard]] inline constexpr Status
norm_check_operand (const Trainable &tensor, const Shape &want,
                    const uint32_t elements) noexcept
{
    if (!status_ok (want.validate ()))
        return Status::BAD_LAYOUT;
    if (tensor._M_code == nullptr)
        return Status::BAD_LAYOUT;
    if (tensor._M_count != elements || tensor._M_count != want.elements ())
        return Status::BAD_LAYOUT;
    if (tensor._M_shape._M_rank == want._M_rank)
        return tensor._M_shape == want ? Status::OK : Status::BAD_LAYOUT;
    if (tensor._M_shape._M_rank == 1u && want._M_rank == 2u
        && want._M_dims[0] == 1u && tensor._M_shape._M_dims[0] == want._M_dims[1])
        return Status::OK;
    return Status::BAD_LAYOUT;
}

/** @brief Round-half-away division of an int64 sum by a positive count. */
[[nodiscard]] inline constexpr int32_t
norm_mean_q15 (const int64_t sum, const uint32_t count) noexcept
{
    if (count == 0u)
        return 0;
    const int64_t half = static_cast<int64_t> (count) / 2;
    return static_cast<int32_t> (sum >= 0 ? (sum + half) / count
                                          : (sum - half) / count);
}

/**
 * @brief Compute one window's mean and deviation in Q15 / Q30.
 *
 * @p values points at the window's codes.  The square sum is int64 (a 255-code
 * swing squared is 4.3e9, so the accumulator is wide before the divide), and
 * the root sees a value clamped to the uint32 rail `_S_isqrt` accepts.
 */
inline void
norm_window_stats (const int8_t *const ZZS_TRAIN_RESTRICT values,
                   const uint32_t count, const bool rms, const uint32_t eps_q30,
                   NormStats &stats) noexcept
{
    int64_t sum = 0;
    int64_t squares = 0;
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t value = code_to_q15 (values[i]);
            sum += value;
            squares += static_cast<int64_t> (value) * value;
        }
    const int32_t mean = rms ? 0 : norm_mean_q15 (sum, count);
    int64_t variance;
    if (rms)
        variance = (squares + static_cast<int64_t> (count) / 2)
                   / static_cast<int64_t> (count);
    else
        {
            int64_t centered_squares = 0;
            for (uint32_t i = 0; i < count; ++i)
                {
                    const int32_t centered
                        = code_to_q15 (values[i]) - mean;
                    centered_squares += static_cast<int64_t> (centered) * centered;
                }
            variance = (centered_squares + static_cast<int64_t> (count) / 2)
                       / static_cast<int64_t> (count);
        }
    if (variance < 0)
        variance = 0;
    const uint64_t biased = static_cast<uint64_t> (variance)
                            + static_cast<uint64_t> (eps_q30);
    const uint32_t bounded
        = biased > 0xFFFFFFFFull ? 0xFFFFFFFFu
                                 : static_cast<uint32_t> (biased);
    stats._M_mean = mean;
    stats._M_deviation = static_cast<int32_t> (_S_isqrt (bounded));
}

/** @brief One Q15 lane of the forward normalization. */
[[nodiscard]] inline constexpr int32_t
norm_lane_q15 (const int32_t value_q15, const NormStats &stats, const bool rms,
               const int32_t gamma_q15, const int32_t beta_q15) noexcept
{
    const int32_t centered = rms ? value_q15 : value_q15 - stats._M_mean;
    const int32_t xhat
        = stats._M_deviation != 0
              ? static_cast<int32_t> (
                    (static_cast<int64_t> (centered) << _S_norm_xhat_bits)
                    / stats._M_deviation)
              : 0;
    const int32_t scaled = static_cast<int32_t> (
        _S_round_shift (static_cast<int64_t> (gamma_q15) * xhat,
                        _S_norm_xhat_bits));
    return q15_clamp (scaled + beta_q15);
}

/**
 * @brief Forward pass with no tape recording.
 *
 * @p stats receives one `NormStats` per window and is the only buffer the
 * backward reads besides the live operands.  A gamma/beta operand is required
 * exactly when its spec flag is set; the unused view may be default-constructed.
 * The output gradient lanes are zeroed here, because a forward always precedes
 * the backward walk that accumulates into them.
 */
[[nodiscard]] inline Status
norm_forward (const NormSpec &spec, const Trainable &input,
              const Trainable &gamma, const Trainable &bias, Trainable &output,
              NormStats *const stats) noexcept
{
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    Status operand
        = norm_check_operand (input, spec.input_shape (), spec.elements ());
    if (!status_ok (operand))
        return operand;
    operand = norm_check_operand (output, spec.input_shape (), spec.elements ());
    if (!status_ok (operand))
        return operand;
    if (spec._M_use_gamma != 0u)
        {
            operand = norm_check_operand (gamma, spec.gamma_shape (),
                                          spec._M_features);
            if (!status_ok (operand))
                return operand;
        }
    if (spec._M_use_bias != 0u)
        {
            operand = norm_check_operand (bias, spec.gamma_shape (),
                                          spec._M_features);
            if (!status_ok (operand))
                return operand;
        }
    if (stats == nullptr)
        return Status::BAD_LAYOUT;

    const uint32_t length = spec.window ();
    const uint32_t per_row = spec.windows_per_row ();
    const bool rms = spec._M_rms != 0u;
    const uint32_t features = spec._M_features;

    for (uint32_t row = 0; row < spec._M_batch; ++row)
        for (uint32_t window = 0; window < per_row; ++window)
            {
                const uint32_t base = row * features + window * length;
                NormStats &state = stats[row * per_row + window];
                norm_window_stats (input._M_code + base, length, rms,
                                   spec._M_eps_q30, state);
                for (uint32_t i = 0; i < length; ++i)
                    {
                        const uint32_t feature = window * length + i;
                        const int32_t gamma_q15
                            = spec._M_use_gamma != 0u
                                  ? code_to_q15 (gamma._M_code[feature])
                                  : _S_q15_one;
                        const int32_t beta_q15
                            = spec._M_use_bias != 0u
                                  ? code_to_q15 (bias._M_code[feature])
                                  : 0;
                        output._M_code[base + i] = q15_to_code (
                            norm_lane_q15 (code_to_q15 (input._M_code[base + i]),
                                           state, rms, gamma_q15, beta_q15));
                    }
            }
    output.zero_grad ();
    return Status::OK;
}

/**
 * @brief Backward of the normalization, in the `BackwardFn` signature.
 *
 * Recorded node layout: operands [input, gamma | absent, bias | absent]; saved
 * [spec, stats, scratch].  The scratch is two `window`-lane rows: the routed
 * gradient `g gamma` and `xhat`.  Both passes over a window keep their totals in
 * int64 locals, so a long feature row cannot wrap the mean or the dot term.
 */
[[nodiscard]] inline Status
norm_backward (const Trainable &output, const Trainable &grad_output,
               GradContext &context) noexcept
{
    const NormSpec *const slot = context.saved_as<NormSpec> (_S_norm_spec);
    if (slot == nullptr)
        return Status::BAD_LAYOUT;
    const NormSpec spec = *slot;
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    if (context._M_operands == nullptr || context._M_operand_count < 1u)
        return Status::BAD_LAYOUT;
    const Trainable &input = context._M_operands[0];
    const uint32_t need = 1u + (spec._M_use_gamma != 0u ? 1u : 0u)
                          + (spec._M_use_bias != 0u ? 1u : 0u);
    if (context._M_operand_count < need)
        return Status::BAD_LAYOUT;
    const Trainable *const gamma
        = spec._M_use_gamma != 0u ? context._M_operands + 1 : nullptr;
    const Trainable *const bias
        = spec._M_use_bias != 0u
              ? context._M_operands + 1 + (spec._M_use_gamma != 0u ? 1u : 0u)
              : nullptr;
    if (output._M_count != spec.elements ()
        || input._M_count != spec.elements ())
        return Status::BAD_LAYOUT;
    if (grad_output._M_grad == nullptr)
        return Status::OK;
    const NormStats *const stats = context.saved_as<NormStats> (_S_norm_stats);
    int32_t *const scratch = context.saved_as<int32_t> (_S_norm_scratch);
    if (stats == nullptr || scratch == nullptr)
        return Status::BAD_LAYOUT;

    const uint32_t length = spec.window ();
    const uint32_t per_row = spec.windows_per_row ();
    const bool rms = spec._M_rms != 0u;
    const uint32_t features = spec._M_features;
    int32_t *const routed = scratch;
    int32_t *const xhat_row = scratch + length;
    uint32_t saturated = 0u;

    for (uint32_t row = 0; row < spec._M_batch; ++row)
        for (uint32_t window = 0; window < per_row; ++window)
            {
                const uint32_t base = row * features + window * length;
                const NormStats &state = stats[row * per_row + window];
                int64_t sum_routed = 0;
                int64_t sum_routed_xhat = 0;
                for (uint32_t i = 0; i < length; ++i)
                    {
                        const uint32_t feature = window * length + i;
                        const int32_t value = code_to_q15 (input._M_code[base + i]);
                        const int32_t centered
                            = rms ? value : value - state._M_mean;
                        const int32_t xhat
                            = state._M_deviation != 0
                                  ? static_cast<int32_t> (
                                        (static_cast<int64_t> (centered)
                                         << _S_norm_xhat_bits)
                                        / state._M_deviation)
                                  : 0;
                        const int32_t gamma_q15
                            = spec._M_use_gamma != 0u
                                  ? code_to_q15 (gamma->_M_code[feature])
                                  : _S_q15_one;
                        const int32_t g = grad_output._M_grad[base + i];
                        const int32_t routed_g = q15_mul (g, gamma_q15);
                        xhat_row[i] = xhat;
                        routed[i] = routed_g;
                        sum_routed += routed_g;
                        sum_routed_xhat += _S_round_shift (
                            static_cast<int64_t> (routed_g) * xhat,
                            _S_norm_xhat_bits);
                        if (gamma != nullptr && gamma->_M_grad != nullptr
                            && gamma->add_grad (feature,
                                                q15_requant (
                                                    static_cast<int64_t> (g) * xhat,
                                                    _S_norm_xhat_bits)))
                            ++saturated;
                        if (bias != nullptr && bias->_M_grad != nullptr
                            && bias->add_grad (feature, g))
                            ++saturated;
                    }
                const int32_t mean_routed
                    = norm_mean_q15 (sum_routed, length);
                const int32_t mean_routed_xhat
                    = norm_mean_q15 (sum_routed_xhat, length);
                if (input._M_grad == nullptr)
                    continue;
                for (uint32_t i = 0; i < length; ++i)
                    {
                        const int32_t raw
                            = routed[i] - mean_routed
                              - _S_round_shift (
                                  static_cast<int64_t> (xhat_row[i])
                                      * mean_routed_xhat,
                                  _S_norm_xhat_bits);
                        const int32_t delta
                            = state._M_deviation != 0
                                  ? q15_div (raw, state._M_deviation)
                                  : 0;
                        if (input.add_grad (base + i, delta))
                            ++saturated;
                    }
            }
    context._M_saturation_events += saturated;
    return Status::OK;
}

/**
 * @brief Carve the norm state, run the forward and record the node.
 *
 * Three saved slots -- stats, scratch and the spec -- and three operands at
 * most.  `NormSpec::bytes ()` is the exact pool cost, so a caller that sizes an
 * arena from it never sees a surprise.
 */
[[nodiscard]] inline Status
norm_run (Tape &tape, const NormSpec &spec, const Trainable &input,
          const Trainable &gamma, const Trainable &bias,
          Trainable &output) noexcept
{
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    if (input._M_count != spec.elements ()
        || output._M_count != spec.elements ())
        return Status::BAD_LAYOUT;
    if (spec._M_use_gamma != 0u && gamma._M_count != spec._M_features)
        return Status::BAD_LAYOUT;
    if (spec._M_use_bias != 0u && bias._M_count != spec._M_features)
        return Status::BAD_LAYOUT;

    Saved slots[_S_max_operands];
    Trainable operands[_S_max_operands];
    uint32_t operand_count = 0u;
    NormStats *stats = nullptr;
    int32_t *scratch = nullptr;

    const Status save_stats
        = tape.save (spec.windows (), slots[_S_norm_stats], stats);
    if (!status_ok (save_stats))
        return save_stats;
    const Status save_scratch
        = tape.save (2u * spec.window (), slots[_S_norm_scratch], scratch);
    if (!status_ok (save_scratch))
        return save_scratch;
    {
        NormSpec *saved_spec = nullptr;
        const Status save = tape.save (1u, slots[_S_norm_spec], saved_spec);
        if (!status_ok (save))
            return save;
        *saved_spec = spec;
    }

    const Status forward
        = norm_forward (spec, input, gamma, bias, output, stats);
    if (!status_ok (forward))
        return forward;

    operands[operand_count++] = input;
    if (spec._M_use_gamma != 0u)
        operands[operand_count++] = gamma;
    if (spec._M_use_bias != 0u)
        operands[operand_count++] = bias;
    return tape.record (norm_backward, spec._M_op_id, output, operands,
                        operand_count, slots, _S_max_operands);
}

/** @brief Forward of `a + b` in the code domain (saturating int8). */
[[nodiscard]] inline Status
residual_forward (const Trainable &a, const Trainable &b,
                  Trainable &output) noexcept
{
    if (a._M_count == 0u || a._M_count != b._M_count
        || a._M_count != output._M_count)
        return Status::BAD_LAYOUT;
    if (!status_ok (a.validate ()) || !status_ok (b.validate ())
        || !status_ok (output.validate ()))
        return Status::BAD_LAYOUT;
    if (a._M_shape != b._M_shape || a._M_shape != output._M_shape)
        return Status::BAD_LAYOUT;
    for (uint32_t i = 0; i < a._M_count; ++i)
        output._M_code[i] = _S_i8_code (static_cast<int32_t> (a._M_code[i])
                                        + static_cast<int32_t> (b._M_code[i]));
    output.zero_grad ();
    return Status::OK;
}

/**
 * @brief Backward of the residual add: the upstream lane reaches both branches.
 *
 * Recorded node layout: operands [a, b]; no saved state.  The addition is its
 * own Jacobian (1 to each operand), so this is the one place a residual stream
 * forks; a branch that is frozen simply has no gradient buffer and the
 * saturating `add_grad` skips it.
 */
[[nodiscard]] inline Status
residual_backward (const Trainable &output, const Trainable &grad_output,
                   GradContext &context) noexcept
{
    (void)output;
    if (context._M_operands == nullptr || context._M_operand_count < 2u)
        return Status::BAD_LAYOUT;
    if (grad_output._M_grad == nullptr)
        return Status::OK;
    uint32_t saturated = 0u;
    for (uint32_t operand = 0; operand < 2u; ++operand)
        {
            const Trainable &branch = context._M_operands[operand];
            if (branch._M_grad == nullptr)
                continue;
            for (uint32_t i = 0; i < grad_output._M_count; ++i)
                if (branch.add_grad (i, grad_output._M_grad[i]))
                    ++saturated;
        }
    context._M_saturation_events += saturated;
    return Status::OK;
}

/** @brief Run and record one residual add. */
[[nodiscard]] inline Status
residual_run (Tape &tape, const Trainable &a, const Trainable &b,
              Trainable &output, const uint32_t op_id = 0u) noexcept
{
    const Status forward = residual_forward (a, b, output);
    if (!status_ok (forward))
        return forward;
    Trainable operands[2] = { a, b };
    return tape.record (residual_backward, op_id, output, operands, 2u, nullptr,
                        0u);
}
} // namespace zzs::mllee::train
