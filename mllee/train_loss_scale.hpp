/* Copyright(c) 2026 Tommicord
 *
 * Permission for use, copying and distribution is granted under the same
 * MIT-style terms as the rest of the mllee module (see mllee/core.hpp).
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND. */

#pragma once

#include "mllee/status.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Loss scaling: static and dynamic Q15 scale plus unscale.
 *
 * The tape seeds the loss node with its loss scale (Tape::backward writes
 * loss.grad[0] = q15_clamp(scale)) and every backward function multiplies its
 * incoming seed through q15_mul, so each gradient lane at rest carries
 *
 *     lane = round(true_q15 * scale_q15 / 32768)
 *
 * instead of the true value.  Because the seed itself is a Q15 lane, the
 * scale can never exceed the Q15 rails: the ceiling here is 32767, not the
 * 2^k amplifiers used by fp16 mixed precision.  The consequence is inverted
 * precision logic -- a *large* scale keeps gradient lanes precise (a scale
 * of ~1.0 is the identity), while backing off (halving) deliberately trades
 * lane resolution for headroom whenever saturating accumulation trips the
 * tape's saturation counter.  Growing back toward the ceiling restores that
 * resolution.
 *
 * The dynamic state machine (LossScaler) is driven purely by the tape's
 * per-pass TapeStats::_M_saturation_events, which GradContext increments for
 * every q15_add_sat / add_grad rail hit recorded during backward:
 *   - saturation events > 0   -> the step is rejected (the caller must skip
 *     the optimizer), the scale halves, the clean-step counter resets;
 *   - N consecutive clean steps -> the scale doubles, clamped to the ceiling;
 *   - the floor keeps the scale away from 0 so unscale never amplifies by an
 *     absurd factor and lanes never quantise everything to zero.
 *
 * unscale_grads_q15 divides each gradient lane back to true Q15 after the
 * pass, rounding half away from zero like the rest of the module and
 * saturating (never wrapping) at the rails; lanes that clamp are counted so
 * the caller can see that a scale-invariant overflow slipped through (it can
 * happen when a single grad add saturates exactly once and the counter was
 * consumed by a different node).
 *
 * Struct sizes are static_asserted.  Pool cost: zero -- every type here is
 * caller-owned POD arithmetic plus in-place lane edits; nothing is carved.
 */
namespace zzs::mllee::train
{
/** @brief Largest scale the tape will honour (q15_clamp of the seed). */
inline constexpr int32_t _S_scale_ceiling = _S_q15_max;

/** @brief Smallest scale a scaler may decay to (protects unscale division). */
inline constexpr int32_t _S_scale_floor_min = 32u;

/** @brief What a dynamic scaler decided after one backward pass. */
enum class ScaleAction : uint32_t
{
    /** @brief Clean step, scale unchanged; the optimizer may proceed. */
    Unchanged = 0u,
    /** @brief Saturation seen: reject the step, scale halved (floored). */
    BackedOff = 1u,
    /** @brief N clean steps done: scale doubled (possibly capped at ceiling). */
    Grown = 2u,
};

static_assert (sizeof (ScaleAction) == 4u, "scale action is one word");

/**
 * @brief Dynamic loss scaler: one 40-byte caller-owned state machine.
 *
 * Plain data, no heap, no statics; the training loop keeps one per run.
 * All counters are per-run and only ever advance inside dynamic_on_backward.
 */
struct LossScaler
{
    /** @brief Current seed scale in Q15, always within [floor, ceiling]. */
    int32_t _M_scale_q15 = _S_scale_ceiling;
    /** @brief Lower bound; never below _S_scale_floor_min once initialised. */
    int32_t _M_floor_q15 = _S_scale_floor_min;
    /** @brief Upper bound; at most _S_scale_ceiling (the Q15 rail). */
    int32_t _M_ceiling_q15 = _S_scale_ceiling;
    /** @brief Consecutive clean steps required before the next growth. */
    uint32_t _M_growth_interval = 2000u;
    /** @brief Clean steps accumulated since the last scale change. */
    uint32_t _M_clean_steps = 0u;
    /** @brief Total halving events. */
    uint32_t _M_backoffs = 0u;
    /** @brief Total doublings that actually changed the scale. */
    uint32_t _M_growths = 0u;
    /** @brief Steps whose gradients must be discarded (saturation seen). */
    uint32_t _M_skipped_steps = 0u;
    /** @brief High-water scale seen (starts at the initialised scale). */
    int32_t _M_peak_scale_q15 = _S_scale_ceiling;
    /** @brief Low-water scale seen. */
    int32_t _M_low_scale_q15 = _S_scale_ceiling;
};

static_assert (sizeof (LossScaler) == 40u, "scaler is ten words");

/**
 * @brief Initialise (or re-initialise) a dynamic scaler.
 *
 * The initial scale is clamped into [floor_q15, ceiling_q15]; a ceiling
 * above the Q15 rail is rejected because the tape could never honour it.
 */
[[nodiscard]] inline Status
dynamic_reset (LossScaler &scaler, const int32_t initial_q15,
               const uint32_t growth_interval, const int32_t floor_q15,
               const int32_t ceiling_q15) noexcept
{
    if (floor_q15 < _S_scale_floor_min || ceiling_q15 < floor_q15
        || ceiling_q15 > _S_scale_ceiling || growth_interval == 0u)
        return Status::BAD_LAYOUT;
    int32_t scale = initial_q15 < floor_q15 ? floor_q15 : initial_q15;
    if (scale > ceiling_q15)
        scale = ceiling_q15;
    scaler._M_scale_q15 = scale;
    scaler._M_floor_q15 = floor_q15;
    scaler._M_ceiling_q15 = ceiling_q15;
    scaler._M_growth_interval = growth_interval;
    scaler._M_clean_steps = 0u;
    scaler._M_backoffs = 0u;
    scaler._M_growths = 0u;
    scaler._M_skipped_steps = 0u;
    scaler._M_peak_scale_q15 = scale;
    scaler._M_low_scale_q15 = scale;
    return Status::OK;
}

/** @brief Push the current scale onto the tape for the next forward/backward. */
inline void
dynamic_apply (const LossScaler &scaler, Tape &tape) noexcept
{
    tape.set_loss_scale (scaler._M_scale_q15);
}

/**
 * @brief One state-machine transition driven by the tape's saturation count.
 *
 * Call after Tape::backward and before the optimizer step.  A BackedOff
 * result means this step's gradients are poisoned: the caller must skip the
 * optimizer update (and typically zero_grad) because saturated lanes are
 * unrecoverable at any scale.  Otherwise the clean counter advances and, once
 * it reaches the growth interval, the scale doubles toward the ceiling.
 */
[[nodiscard]] inline ScaleAction
dynamic_advance (LossScaler &scaler, const TapeStats &stats) noexcept
{
    if (stats._M_saturation_events != 0u)
    {
        ++scaler._M_backoffs;
        ++scaler._M_skipped_steps;
        scaler._M_clean_steps = 0u;
        const int32_t halved = scaler._M_scale_q15 >> 1;
        scaler._M_scale_q15 = halved < scaler._M_floor_q15 ? scaler._M_floor_q15
                                                           : halved;
    }
    else
    {
        ++scaler._M_clean_steps;
        if (scaler._M_clean_steps >= scaler._M_growth_interval)
        {
            scaler._M_clean_steps = 0u;
            if (scaler._M_scale_q15 < scaler._M_ceiling_q15)
            {
                const int32_t doubled = scaler._M_scale_q15 << 1;
                scaler._M_scale_q15 = doubled > scaler._M_ceiling_q15
                                          ? scaler._M_ceiling_q15
                                          : doubled;
                ++scaler._M_growths;
                if (scaler._M_scale_q15 > scaler._M_peak_scale_q15)
                    scaler._M_peak_scale_q15 = scaler._M_scale_q15;
                return ScaleAction::Grown;
            }
        }
        return ScaleAction::Unchanged;
    }
    if (scaler._M_scale_q15 < scaler._M_low_scale_q15)
        scaler._M_low_scale_q15 = scaler._M_scale_q15;
    return ScaleAction::BackedOff;
}

/**
 * @brief Convenience: advance from the tape's own stats and re-seed the tape.
 *
 * Returns the action so the caller knows whether to skip the optimizer step;
 * the tape always ends up carrying the new scale for the next pass.
 */
[[nodiscard]] inline ScaleAction
dynamic_on_backward (LossScaler &scaler, Tape &tape) noexcept
{
    const ScaleAction action = dynamic_advance (scaler, tape.stats ());
    dynamic_apply (scaler, tape);
    return action;
}

/** @brief Fixed-scale scaler: no state machine, one Q15 multiplier. */
struct StaticLossScale
{
    /** @brief Seed scale in Q15, [1, _S_scale_ceiling]. */
    int32_t _M_scale_q15 = _S_scale_ceiling;
    /** @brief Steps applied (telemetry for the run log). */
    uint32_t _M_steps_applied = 0u;
};

static_assert (sizeof (StaticLossScale) == 8u, "static scale is two words");

/** @brief Initialise a static scale; out-of-range values are rejected. */
[[nodiscard]] inline constexpr Status
static_reset (StaticLossScale &scale, const int32_t scale_q15) noexcept
{
    if (scale_q15 < 1 || scale_q15 > _S_scale_ceiling)
        return Status::BAD_LAYOUT;
    scale._M_scale_q15 = scale_q15;
    scale._M_steps_applied = 0u;
    return Status::OK;
}

/** @brief Push a static scale onto the tape and count the application. */
inline void
static_apply (StaticLossScale &scale, Tape &tape) noexcept
{
    tape.set_loss_scale (scale._M_scale_q15);
    ++scale._M_steps_applied;
}

/** @brief Outcome of one unscale pass over a parameter list. */
struct UnscaleReport
{
    /** @brief Lanes divided. */
    uint32_t _M_lanes_processed = 0u;
    /** @brief Lanes whose true value exceeded the Q15 rails and clamped. */
    uint32_t _M_overflow_clamped = 0u;
    /** @brief Parameters skipped for having no gradient buffer. */
    uint32_t _M_params_skipped = 0u;
};

static_assert (sizeof (UnscaleReport) == 12u, "unscale report is three words");

/**
 * @brief Divide every gradient lane back to true Q15: lane * 32768 / scale,
 *         rounded half away from zero, saturating (never wrapping) at rails.
 *
 * Works in place through the Trainable gradient pointers; no pool and no
 * SIMD wrapper exists for per-lane division, so the loop is explicit and
 * stride-1.  A scale of 32768 (the tape's un-clamped constructor default)
 * is the exact identity.  Zero gradients stay zero lanes without a divide.
 */
[[nodiscard]] inline Status
unscale_grads_q15 (const Trainable *const params, const uint32_t count,
                   const int32_t scale_q15, UnscaleReport &report) noexcept
{
    report = UnscaleReport{};
    if (params == nullptr || scale_q15 < 1 || scale_q15 > _S_q15_one)
        return Status::BAD_LAYOUT;
    if (count == 0u)
        return Status::OK;
    if (scale_q15 == _S_q15_one)
    {
        for (uint32_t p = 0; p < count; ++p)
            if (params[p]._M_grad != nullptr)
                report._M_lanes_processed += params[p]._M_count;
            else
                ++report._M_params_skipped;
        return Status::OK;
    }
    for (uint32_t p = 0; p < count; ++p)
    {
        const Trainable &param = params[p];
        if (param._M_grad == nullptr)
        {
            ++report._M_params_skipped;
            continue;
        }
        for (uint32_t i = 0; i < param._M_count; ++i)
        {
            const int32_t lane = param._M_grad[i];
            if (lane != 0)
            {
                /* lane << 15 fits int32 for any lane in [-32768, 32767]. */
                const int32_t wide = lane << 15;
                const int32_t true_q15 = div_round (wide, scale_q15);
                if (true_q15 > _S_q15_max || true_q15 < _S_q15_min)
                    ++report._M_overflow_clamped;
                param._M_grad[i] = q15_to_i16_sat (true_q15);
            }
            ++report._M_lanes_processed;
        }
    }
    return Status::OK;
}

/** @brief Snapshot of everything the counters know, for the run log. */
struct LossScaleStats
{
    int32_t _M_scale_q15;
    int32_t _M_floor_q15;
    int32_t _M_ceiling_q15;
    int32_t _M_peak_scale_q15;
    int32_t _M_low_scale_q15;
    uint32_t _M_clean_steps;
    uint32_t _M_backoffs;
    uint32_t _M_growths;
    uint32_t _M_skipped_steps;
    uint32_t _M_growth_interval;
};

static_assert (sizeof (LossScaleStats) == 40u, "stats mirror the scaler");

/** @brief Copy the dynamic scaler's observable state. */
[[nodiscard]] inline constexpr LossScaleStats
dynamic_stats (const LossScaler &scaler) noexcept
{
    return LossScaleStats{ scaler._M_scale_q15, scaler._M_floor_q15,
                           scaler._M_ceiling_q15, scaler._M_peak_scale_q15,
                           scaler._M_low_scale_q15, scaler._M_clean_steps,
                           scaler._M_backoffs, scaler._M_growths,
                           scaler._M_skipped_steps,
                           scaler._M_growth_interval };
}

/** @brief Copy the static scaler's observable state (floor/ceiling unused). */
[[nodiscard]] inline constexpr LossScaleStats
static_stats (const StaticLossScale &scale) noexcept
{
    return LossScaleStats{ scale._M_scale_q15, 1, _S_scale_ceiling,
                           scale._M_scale_q15, scale._M_scale_q15,
                           0u, 0u, 0u, 0u, scale._M_steps_applied };
}
} // namespace zzs::mllee::train
