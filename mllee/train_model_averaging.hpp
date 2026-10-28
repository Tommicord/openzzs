/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#pragma once

#ifndef ZZS_MLLEE_TRAIN_MODEL_AVERAGING_HPP
#define ZZS_MLLEE_TRAIN_MODEL_AVERAGING_HPP

#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_simd.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Weight averaging over int8 code streams: uniform, EMA, SWA, gated.
 *
 * Averaging trained (or snapshot) weights is the cheapest generalization win
 * available to an embedded training run: no extra data, no extra optimiser state,
 * just a second view of the same loss basin.  The constraint this module adds is
 * that the result must still be a *valid int8 code stream*, because the inference
 * runtime reads codes, not the accumulator they were averaged in.  Every path here
 * therefore ends in the same per-tensor round-to-code step (`avg_round_to_codes`,
 * built on the module's own `_S_round_shift` and `_S_i8_code`), so an averaged
 * weight lands exactly on the grid the requantizer expects and the -128 code is
 * never emitted.
 *
 * Modes:
 *
 *   - Uniform: the arithmetic mean over a window of snapshots.  The lane accumulator
 *     is int64 (a whole snapshot set fits without scaling), the division is
 *     round-half-away-from-zero, and no scratch buffer is needed: one pass writes
 *     the code stream;
 *   - Ema: `state += (target - state) * (1 - decay) / 32768` with the state held in
 *     caller-owned int16 Q15 lanes, fused into a single stride-1 pass, with the
 *     decay itself a Q15 constant so a schedule can step it without a float;
 *   - Swa: the mean over an explicit list of snapshot indices -- a snapshot may be
 *     listed twice and then weighs twice, which is how a caller expresses
 *     unequal period weighting without a second code path.
 *
 * Two guards make averaging safe rather than merely cheap:
 *
 *   - `AvgGate` accepts a snapshot only when the caller's metric improved by at
 *     least `_M_min_delta_q15`, so a diverging tail never dilutes the average;
 *   - `avg_check_divergence` counts lanes whose sign contradicts the running
 *     average, per snapshot, and fails with OUT_OF_RANGE past a caller-set limit:
 *     a sign flip against the mean is the earliest visible symptom of two models
 *     that are not in the same basin (and therefore must not be averaged).
 *
 * Device-path rules honoured here: no heap, no exceptions, no RTTI, no recursion,
 * no writable statics, no float, errors as `Status` values, one pass per buffer,
 * stride-1 over 4-byte-aligned streams.
 *
 * ### Pool cost (all caller-owned; nothing here allocates)
 *
 * Sizes are the device (32-bit pointer) layouts the `static_assert`s pin; on a
 * 64-bit host each view that holds a pointer grows by its trailing alignment only.
 *
 * | Object | Device bytes |
 * |---|---|
 * | `SnapshotSet` | 16 |
 * | `ParamSet` | 16 |
 * | `EmaState` | 24 |
 * | `AvgGate` | 24 |
 * | `DivergenceReport` | 24 |
 * | `AveragingReport` | 32 |
 * | uniform or SWA average of N lanes | 0 (one fused int64 pass, no scratch) |
 * | gated average, S offered snapshots | 4 · S (caller index scratch) |
 * | EMA state, N lanes | 2 · N |
 * | snapshot pointer table, S streams | S · sizeof(pointer) |
 *
 * Averaging is the cheapest training-side module: only the EMA form needs state,
 * and a model-parallel EMA over four tensors of 256 lanes costs 4 · 512 = 2048
 * pool bytes.
 */
namespace zzs::mllee::train
{
/** @brief Which averaging rule produced this weight. */
enum class AveragingMode : uint8_t
{
    Uniform = 0,
    Ema = 1,
    Swa = 2,
};

/** @brief Largest lane count one averaging pass addresses. */
inline constexpr uint32_t _S_avg_max_count = 65535u;

/** @brief Largest snapshot count one set addresses. */
inline constexpr uint32_t _S_avg_max_snapshots = 4096u;

/** @brief int64 accumulator ceiling, i.e. where a sum saturates instead of wrapping. */
inline constexpr int64_t _S_avg_acc_max = (1LL << 60) - 1;

/**
 * @brief Round an int64 accumulator by round-half-away-from-zero division.
 *
 * The averaging mean must not inherit `_S_round_shift`'s tie-toward-plus-infinity
 * on a *negative* sum: a mean of {-1, 0} has to come out at -1, i.e. away from
 * zero, exactly like `div_round` but over the int64 accumulator.
 */
[[nodiscard]] inline constexpr int32_t
avg_div_round (const int64_t numerator, const uint32_t denominator) noexcept
{
    if (denominator == 0u)
        return 0;
    const uint64_t divisor = denominator;
    const bool negative = numerator < 0;
    /* Negate in the unsigned domain: `-INT64_MIN` is not representable as a
       signed int64, while its two's-complement magnitude is. */
    const uint64_t magnitude = negative
                                   ? ~static_cast<uint64_t> (numerator) + 1u
                                   : static_cast<uint64_t> (numerator);
    const uint64_t quotient
        = (magnitude + divisor / 2u) / divisor;
    /* Round-half-away-from-zero can push the magnitude one step past INT32_MAX
       (e.g. numerator = INT64_MIN-ish, denominator = 1), so saturate instead of
       letting the cast wrap. */
    if (quotient > 2147483648ULL)
        return negative ? (-2147483647 - 1) : 2147483647;
    const int64_t signed_quotient = negative ? -static_cast<int64_t> (quotient)
                                             : static_cast<int64_t> (quotient);
    if (signed_quotient > 2147483647LL)
        return 2147483647;
    if (signed_quotient < (-2147483647 - 1LL))
        return -2147483647 - 1;
    return static_cast<int32_t> (signed_quotient);
}

/**
 * @brief A caller-owned set of equal-length int8 code streams: the snapshots.
 *
 * Every stream must be 4-byte aligned, because the SIMD narrowing helpers are the
 * tail of every averaging path.  The set owns nothing: it is a view over the
 * buffers the trainer or the checkpoint reader already has.
 */
struct SnapshotSet
{
    const int8_t *const *_M_streams = nullptr;
    uint32_t _M_snapshots = 0u;
    uint32_t _M_count = 0u;
    uint8_t _M_valid = 0u;
    uint8_t _M_pad[3] = { 0u, 0u, 0u };
};

static_assert (sizeof (SnapshotSet) == (sizeof (void *) == 4u ? 16u : 24u),
               "a snapshot set is one pointer table plus two counters");

/**
 * @brief Bind a snapshot set and validate every stream.
 *
 * Null streams, a zero lane count, a misaligned buffer and an oversized set are
 * all rejected here rather than at the first inner-loop dereference.
 */
[[nodiscard]] inline Status
avg_snapshots_open (SnapshotSet &set, const int8_t *const *const streams,
                    const uint32_t snapshots, const uint32_t count) noexcept
{
    set._M_valid = 0u;
    set._M_streams = nullptr;
    set._M_snapshots = 0u;
    set._M_count = 0u;
    if (streams == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (streams))
        return Status::BAD_LAYOUT;
    if (snapshots == 0u || snapshots > _S_avg_max_snapshots)
        return Status::OUT_OF_RANGE;
    if (count == 0u || count > _S_avg_max_count)
        return Status::OUT_OF_RANGE;
    for (uint32_t index = 0u; index < snapshots; ++index)
        {
            if (streams[index] == nullptr)
                return Status::BAD_LAYOUT;
            if (!simd_aligned (streams[index]))
                return Status::BAD_LAYOUT;
        }
    set._M_streams = streams;
    set._M_snapshots = snapshots;
    set._M_count = count;
    set._M_valid = 1u;
    return Status::OK;
}

/**
 * @brief The tensors an average writes into: mutable code streams and their
 *        gradient lanes.
 *
 * A validated view only -- the `Trainable` array and every buffer behind it stays
 * caller-owned.  `_M_lane_total` is summed at open time, so a report can state a
 * pass's traffic without walking the shapes again.
 */
struct ParamSet
{
    Trainable *_M_params = nullptr;
    uint32_t _M_tensors = 0u;
    uint32_t _M_lane_total = 0u;
    uint8_t _M_valid = 0u;
    uint8_t _M_pad[3] = { 0u, 0u, 0u };

    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (this->_M_params == nullptr || this->_M_valid == 0u)
            return Status::BAD_LAYOUT;
        if (this->_M_tensors == 0u)
            return Status::OUT_OF_RANGE;
        return Status::OK;
    }
};

static_assert (sizeof (ParamSet) == (sizeof (void *) == 4u ? 16u : 24u),
               "a parameter set is one pointer, two counters and a flag word");

/**
 * @brief Bind a parameter set, validating every tensor as it is measured.
 *
 * A tensor whose code stream is null or misaligned, or whose element count is out
 * of the module range, fails here with the `Trainable` status rather than half way
 * through an averaging pass.
 */
[[nodiscard]] inline Status
avg_params_open (ParamSet &set, Trainable *const params,
                 const uint32_t tensors) noexcept
{
    set._M_valid = 0u;
    set._M_params = nullptr;
    set._M_tensors = 0u;
    set._M_lane_total = 0u;
    if (params == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (params))
        return Status::BAD_LAYOUT;
    if (tensors == 0u || tensors > _S_avg_max_count)
        return Status::OUT_OF_RANGE;
    uint32_t lanes = 0u;
    for (uint32_t tensor = 0u; tensor < tensors; ++tensor)
        {
            const Status status = params[tensor].validate ();
            if (!status_ok (status))
                return status;
            if (params[tensor]._M_code == nullptr)
                return Status::BAD_LAYOUT;
            lanes += params[tensor]._M_count;
            if (lanes > 0x0FFFFFFFu)
                return Status::OUT_OF_RANGE;
        }
    set._M_params = params;
    set._M_tensors = tensors;
    set._M_lane_total = lanes;
    set._M_valid = 1u;
    return Status::OK;
}

/**
 * @brief Uniform average of snapshots [@p from, @p to) into a code stream.
 *
 * One pass, one int64 accumulator per lane, no scratch: the widened codes are
 * summed, divided by the snapshot count with rounding away from zero, and narrowed
 * back onto the code grid.  A window of one snapshot is a copy, which is what lets
 * the same call serve an EMA seed or a plain restore.
 */
[[nodiscard]] inline Status
avg_uniform (const SnapshotSet &set, const uint32_t from, const uint32_t to,
             int8_t *const destination) noexcept
{
    if (set._M_valid == 0u || destination == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (destination))
        return Status::BAD_LAYOUT;
    if (from >= to || to > set._M_snapshots)
        return Status::OUT_OF_RANGE;
    const uint32_t window = to - from;
    const uint32_t count = set._M_count;
    for (uint32_t lane = 0u; lane < count; ++lane)
        {
            int64_t accumulator = 0;
            for (uint32_t snapshot = from; snapshot < to; ++snapshot)
                {
                    accumulator += code_to_q15 (set._M_streams[snapshot][lane]);
                    if (accumulator > _S_avg_acc_max)
                        accumulator = _S_avg_acc_max;
                    else if (accumulator < -_S_avg_acc_max)
                        accumulator = -_S_avg_acc_max;
                }
            const int32_t mean = avg_div_round (accumulator, window);
            destination[lane] = q15_to_code (mean);
        }
    return Status::OK;
}

/**
 * @brief SWA: the uniform average over an explicit list of snapshot indices.
 *
 * Duplicates are allowed and weigh again -- the documented way to put more
 * emphasis on a late snapshot without a second accumulation path.  @p indices must
 * be 4-byte aligned and hold @p used entries, every one in range.
 */
[[nodiscard]] inline Status
avg_swa (const SnapshotSet &set, const uint32_t *const indices,
         const uint32_t used, int8_t *const destination) noexcept
{
    if (set._M_valid == 0u || destination == nullptr || indices == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (destination) || !simd_aligned (indices))
        return Status::BAD_LAYOUT;
    if (used == 0u || used > _S_avg_max_snapshots)
        return Status::OUT_OF_RANGE;
    for (uint32_t index = 0u; index < used; ++index)
        if (indices[index] >= set._M_snapshots)
            return Status::OUT_OF_RANGE;
    const uint32_t count = set._M_count;
    for (uint32_t lane = 0u; lane < count; ++lane)
        {
            int64_t accumulator = 0;
            for (uint32_t index = 0u; index < used; ++index)
                {
                    accumulator += code_to_q15 (
                        set._M_streams[indices[index]][lane]);
                    if (accumulator > _S_avg_acc_max)
                        accumulator = _S_avg_acc_max;
                    else if (accumulator < -_S_avg_acc_max)
                        accumulator = -_S_avg_acc_max;
                }
            destination[lane] = q15_to_code (avg_div_round (accumulator, used));
        }
    return Status::OK;
}

/**
 * @brief EMA state: one Q15 int16 lane per weight, owned by the caller.
 *
 * The state deliberately outlives the code grid: averaging in Q15 and only
 * narrowing on read is what stops a slow decay from quantizing the average to
 * zero.  `_M_updates` counts the steps folded in, so a caller can bias-correct
 * (`state / (1 - decay^n)`) with the module's own integer ratio helper.
 */
struct EmaState
{
    int16_t *_M_lanes = nullptr;
    uint32_t _M_count = 0u;
    int32_t _M_decay_q15 = 32572;
    uint32_t _M_updates = 0u;
    /** @brief Saturating count of lanes that clamped during updates. */
    uint32_t _M_saturation_events = 0u;
    uint8_t _M_valid = 0u;
    uint8_t _M_pad[3] = { 0u, 0u, 0u };

    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (this->_M_lanes == nullptr || this->_M_valid == 0u)
            return Status::BAD_LAYOUT;
        if (this->_M_count > _S_avg_max_count)
            return Status::OUT_OF_RANGE;
        if (this->_M_decay_q15 < 0 || this->_M_decay_q15 > _S_q15_max)
            return Status::OUT_OF_RANGE;
        return Status::OK;
    }
};

static_assert (sizeof (EmaState) == (sizeof (void *) == 4u ? 24u : 32u),
               "an EMA state view is a lane pointer, four counters and a flag");

/** @brief Bind an EMA state over caller-owned int16 lanes. */
[[nodiscard]] inline Status
avg_ema_open (EmaState &state, int16_t *const lanes, const uint32_t count,
              const int32_t decay_q15) noexcept
{
    state._M_valid = 0u;
    state._M_lanes = nullptr;
    state._M_count = 0u;
    state._M_updates = 0u;
    state._M_saturation_events = 0u;
    if (lanes == nullptr)
        return Status::BAD_LAYOUT;
    if (count == 0u || count > _S_avg_max_count)
        return Status::OUT_OF_RANGE;
    if (decay_q15 < 0 || decay_q15 > _S_q15_max)
        return Status::OUT_OF_RANGE;
    if (!simd_aligned (lanes))
        return Status::BAD_LAYOUT;
    state._M_lanes = lanes;
    state._M_count = count;
    state._M_decay_q15 = decay_q15;
    state._M_valid = 1u;
    return Status::OK;
}

/** @brief Seed an EMA state from a code stream: the n = 0 case. */
[[nodiscard]] inline Status
avg_ema_seed (EmaState &state, const int8_t *const codes) noexcept
{
    const Status state_status = state.validate ();
    if (!status_ok (state_status))
        return state_status;
    if (codes == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (codes))
        return Status::BAD_LAYOUT;
    train_simd::codes_to_q15 (codes, state._M_lanes, state._M_count);
    state._M_updates = 0u;
    state._M_saturation_events = 0u;
    return Status::OK;
}

/**
 * @brief Fold one snapshot into an EMA state: `s += (t - s) * (1 - d) / 2^15`.
 *
 * Written as the difference form rather than `d*s + (1-d)*t` because it needs one
 * multiply per lane instead of two, and because a lane already at its target does
 * not move at all -- which keeps a converged state bit-stable under a saturating
 * narrowing.  One fused pass over the state and the code stream.
 */
[[nodiscard]] inline Status
avg_ema_step (EmaState &state, const int8_t *const codes) noexcept
{
    const Status state_status = state.validate ();
    if (!status_ok (state_status))
        return state_status;
    if (codes == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (codes))
        return Status::BAD_LAYOUT;
    const int32_t move = _S_q15_one - 1 - state._M_decay_q15;
    uint32_t saturations = 0u;
    for (uint32_t lane = 0u; lane < state._M_count; ++lane)
        {
            const int32_t held = state._M_lanes[lane];
            const int32_t target = code_to_q15 (codes[lane]);
            /* `q15_mul` clamps its own result, so a delta that exceeds the Q15
               span after scaling can only move the lane to the rail, never past
               it; the saturating store below still guards the +/-32768 seam. */
            const int32_t step_q15 = q15_mul (target - held, move);
            const int32_t next = held + step_q15;
            if (next > _S_q15_max || next < _S_q15_min)
                {
                    state._M_lanes[lane] = q15_to_i16_sat (next);
                    ++saturations;
                }
            else
                state._M_lanes[lane] = static_cast<int16_t> (next);
        }
    state._M_saturation_events
        = state._M_saturation_events > 0xFFFFFFFFu - saturations
              ? 0xFFFFFFFFu
              : state._M_saturation_events + saturations;
    if (state._M_updates < 0xFFFFFFFFu)
        ++state._M_updates;
    return Status::OK;
}

/** @brief Set the decay of a live EMA state, e.g. from a warmup schedule. */
[[nodiscard]] inline constexpr Status
avg_ema_set_decay (EmaState &state, const int32_t decay_q15) noexcept
{
    if (state._M_valid == 0u)
        return Status::BAD_LAYOUT;
    if (decay_q15 < 0 || decay_q15 > _S_q15_max)
        return Status::OUT_OF_RANGE;
    state._M_decay_q15 = decay_q15;
    return Status::OK;
}

/**
 * @brief The per-tensor round-to-code step every averaging path ends in.
 *
 * Uses `train_simd::q15_to_codes`, so the device path and the portable path narrow
 * a Q15 lane to a code identically, and the result is always inside the [-127, 127]
 * grid the module emits.
 */
[[nodiscard]] inline Status
avg_round_to_codes (const int16_t *const lanes, int8_t *const codes,
                    const uint32_t count) noexcept
{
    if (lanes == nullptr || codes == nullptr)
        return Status::BAD_LAYOUT;
    if (count > _S_avg_max_count)
        return Status::OUT_OF_RANGE;
    if (!simd_aligned (lanes) || !simd_aligned (codes))
        return Status::BAD_LAYOUT;
    train_simd::q15_to_codes (lanes, codes, count);
    return Status::OK;
}

/** @brief Read an EMA state out as a code stream. */
[[nodiscard]] inline Status
avg_ema_read (const EmaState &state, int8_t *const codes) noexcept
{
    const Status state_status = state.validate ();
    if (!status_ok (state_status))
        return state_status;
    return avg_round_to_codes (state._M_lanes, codes, state._M_count);
}

/** @brief Bias-correct an EMA state into lanes the caller owns (decay^n form). */
[[nodiscard]] inline Status
avg_ema_bias_corrected (const EmaState &state, int16_t *const out) noexcept
{
    const Status state_status = state.validate ();
    if (!status_ok (state_status))
        return state_status;
    if (out == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (out))
        return Status::BAD_LAYOUT;
    /* 1 - decay^n, accumulated in Q15 with the module's own saturating multiply:
       the correction factor only ever grows towards 1.0, so it cannot overflow.
       n == 0 means the state is a bare seed, i.e. already unbiased. */
    if (state._M_updates == 0u)
        {
            for (uint32_t lane = 0u; lane < state._M_count; ++lane)
                out[lane] = state._M_lanes[lane];
            return Status::OK;
        }
    int32_t decay_power = state._M_decay_q15;
    int32_t one_minus = _S_q15_one - 1 - state._M_decay_q15;
    for (uint32_t update = 1u; update < state._M_updates; ++update)
        {
            decay_power = q15_mul (decay_power, state._M_decay_q15);
            one_minus = _S_q15_one - 1 - decay_power;
        }
    if (one_minus <= 0)
        {
            for (uint32_t lane = 0u; lane < state._M_count; ++lane)
                out[lane] = state._M_lanes[lane];
            return Status::OK;
        }
    const int32_t reciprocal = q15_reciprocal (one_minus);
    for (uint32_t lane = 0u; lane < state._M_count; ++lane)
        out[lane] = q15_to_i16_sat (
            q15_mul (static_cast<int32_t> (state._M_lanes[lane]), reciprocal));
    return Status::OK;
}

/**
 * @brief Metric gate: average only when the loss improved.
 *
 * Lower is the better direction by default (a loss in Q7 log units or a Q15
 * accuracy complement); `_M_min_delta_q15` sets how much improvement counts as
 * improvement, so a metric that only jitters in the last bits stops triggering
 * snapshots.  `_M_has_best` marks the first sample, which is always accepted --
 * without that, a run started mid-training would never average at all.
 */
struct AvgGate
{
    int32_t _M_best_q15 = 0;
    /** @brief Improvement required over `_M_best_q15` to accept a snapshot. */
    int32_t _M_min_delta_q15 = 1;
    uint32_t _M_accepted = 0u;
    uint32_t _M_rejected = 0u;
    uint32_t _M_compared = 0u;
    /** @brief 0 == lower metric wins (a loss); 1 == higher metric wins. */
    uint8_t _M_higher_is_better = 0u;
    uint8_t _M_has_best = 0u;
    uint16_t _M_pad = 0u;

    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        return this->_M_min_delta_q15 >= 0 ? Status::OK : Status::OUT_OF_RANGE;
    }
};

static_assert (sizeof (AvgGate) == 24u,
               "the averaging gate must stay a 24-byte POD");

/** @brief Reset a gate, e.g. after a snapshot set was consumed. */
constexpr void
avg_gate_reset (AvgGate &gate) noexcept
{
    gate._M_best_q15 = 0;
    gate._M_min_delta_q15 = gate._M_min_delta_q15 < 0 ? 1
                                                     : gate._M_min_delta_q15;
    gate._M_accepted = 0u;
    gate._M_rejected = 0u;
    gate._M_compared = 0u;
    gate._M_has_best = 0u;
    gate._M_pad = 0u;
}

/**
 * @brief Ask the gate whether this snapshot's metric justifies averaging.
 *
 * Updates the best value when it accepts, so a caller that averages does not also
 * have to track the metric trend.  The answer is deterministic: the same metric
 * sequence always yields the same accept/reject trace.
 */
[[nodiscard]] inline Status
avg_gate_consider (AvgGate &gate, const int32_t metric_q15, bool &accept) noexcept
{
    const Status gate_status = gate.validate ();
    if (!status_ok (gate_status))
        return gate_status;
    if (gate._M_compared < 0xFFFFFFFFu)
        ++gate._M_compared;
    if (gate._M_has_best == 0u)
        {
            gate._M_has_best = 1u;
            gate._M_best_q15 = metric_q15;
            ++gate._M_accepted;
            accept = true;
            return Status::OK;
        }
    const int64_t margin
        = gate._M_higher_is_better != 0u
              ? static_cast<int64_t> (metric_q15) - gate._M_best_q15
              : static_cast<int64_t> (gate._M_best_q15) - metric_q15;
    if (margin >= gate._M_min_delta_q15)
        {
            gate._M_best_q15 = metric_q15;
            ++gate._M_accepted;
            accept = true;
            return Status::OK;
        }
    ++gate._M_rejected;
    accept = false;
    return Status::OK;
}

/** @brief Per-lane sign disagreement against the average; zeros never diverge. */
[[nodiscard]] inline constexpr bool
avg_lane_diverges (const int32_t average_code, const int32_t snapshot_code) noexcept
{
    return (average_code > 0 && snapshot_code < 0)
           || (average_code < 0 && snapshot_code > 0);
}

/**
 * @brief What a divergence check found: the worst snapshot and its lane fraction.
 */
struct DivergenceReport
{
    uint32_t _M_lanes_compared = 0u;
    /** @brief Total sign-flipped lanes over every snapshot compared. */
    uint32_t _M_lanes_diverging = 0u;
    uint32_t _M_worst_snapshot = 0u;
    uint32_t _M_worst_snapshot_lanes = 0u;
    /** @brief Worst snapshot's diverging fraction, Q15. */
    int32_t _M_worst_fraction_q15 = 0;
    /** @brief Limit the caller supplied, echoed for the log. */
    int32_t _M_limit_q15 = 0;
};

static_assert (sizeof (DivergenceReport) == 24u,
               "the divergence report is six counters and must stay 24 bytes");

/**
 * @brief Detect snapshots that disagree in sign with the running average.
 *
 * A snapshot in the same loss basin as the average has, at most, a few lanes whose
 * sign flipped; a snapshot from a different basin flips whole channels.  The check
 * is therefore a per-snapshot lane count against a caller-set Q15 fraction limit,
 * and it returns OUT_OF_RANGE as soon as one snapshot crosses it -- the caller
 * still gets the full report, because averaging must not proceed silently.
 *
 * Zero lanes are exempt in both directions: a code of 0 carries no sign, so an
 * averaged-away weight cannot be reported as a divergence.
 */
[[nodiscard]] inline Status
avg_check_divergence (const int8_t *const average, const SnapshotSet &set,
                      const int32_t limit_q15,
                      DivergenceReport &report) noexcept
{
    report = DivergenceReport{};
    report._M_limit_q15 = limit_q15;
    if (average == nullptr || set._M_valid == 0u)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (average))
        return Status::BAD_LAYOUT;
    if (limit_q15 < 0 || limit_q15 > _S_q15_max)
        return Status::OUT_OF_RANGE;
    uint32_t worst_count = 0u;
    uint32_t worst_index = 0u;
    for (uint32_t snapshot = 0u; snapshot < set._M_snapshots; ++snapshot)
        {
            uint32_t diverging = 0u;
            for (uint32_t lane = 0u; lane < set._M_count; ++lane)
                {
                    if (avg_lane_diverges (average[lane],
                                           set._M_streams[snapshot][lane]))
                        ++diverging;
                }
            report._M_lanes_compared += set._M_count;
            report._M_lanes_diverging += diverging;
            if (diverging > worst_count)
                {
                    worst_count = diverging;
                    worst_index = snapshot;
                }
        }
    report._M_worst_snapshot = worst_index;
    report._M_worst_snapshot_lanes = worst_count;
    /* worst_count <= set._M_count <= 65535, so both operands fit int32 exactly. */
    report._M_worst_fraction_q15
        = ratio_q15 (static_cast<int32_t> (worst_count),
                     static_cast<int32_t> (set._M_count));
    if (report._M_worst_fraction_q15 > limit_q15)
        return Status::OUT_OF_RANGE;
    return Status::OK;
}

/** @brief Count of lanes an averaged set and its source differ in, any amount. */
[[nodiscard]] inline Status
avg_code_drift (const int8_t *const average, const int8_t *const source,
                const uint32_t count, uint32_t &drift_out,
                int32_t &max_step_out) noexcept
{
    if (average == nullptr || source == nullptr)
        return Status::BAD_LAYOUT;
    if (count > _S_avg_max_count)
        return Status::OUT_OF_RANGE;
    uint32_t drift = 0u;
    int32_t maximum = 0;
    for (uint32_t lane = 0u; lane < count; ++lane)
        {
            const int32_t difference
                = average[lane] > source[lane] ? average[lane] - source[lane]
                                               : source[lane] - average[lane];
            if (difference != 0)
                ++drift;
            if (difference > maximum)
                maximum = difference;
        }
    drift_out = drift;
    max_step_out = maximum;
    return Status::OK;
}

/**
 * @brief What one averaging run did, across a whole parameter set.
 */
struct AveragingReport
{
    uint32_t _M_tensors = 0u;
    uint32_t _M_lanes = 0u;
    /** @brief Snapshots folded in (uniform window size, or the SWA list length). */
    uint32_t _M_snapshots_used = 0u;
    uint32_t _M_gate_accepted = 0u;
    uint32_t _M_gate_rejected = 0u;
    /** @brief Largest code-step distance between the average and the newest input. */
    int32_t _M_max_code_drift = 0;
    uint32_t _M_drifted_lanes = 0u;
    uint8_t _M_mode = static_cast<uint8_t> (AveragingMode::Uniform);
    uint8_t _M_pad[3] = { 0u, 0u, 0u };
};

static_assert (sizeof (AveragingReport) == 32u,
               "the averaging report is seven counters plus a mode byte");

/**
 * @brief Turn a caller-supplied metric series into the snapshot index list to keep.
 *
 * This is the "average only when the loss improved" rule made usable: the metric
 * array is whatever the caller measures (a Q7 log loss, a Q15 error complement, a
 * validation score), one entry per snapshot in publication order, and the gate
 * decides entry by entry.  Accepted snapshot indices are written to
 * @p indices_out, which the caller owns and which must have room for @p count
 * entries (the worst case is every snapshot improving).
 *
 * The trace is deterministic: the same metric series always yields the same index
 * list and the same counters, because the gate carries no hidden state beyond its
 * own fields.
 *
 * @param count        entries in @p metrics, i.e. snapshots offered
 * @param indices_out  caller-owned index list, filled with accepted snapshot ids
 * @param used_out     how many entries were accepted
 */
[[nodiscard]] inline Status
avg_gate_record (AvgGate &gate, const int32_t *const metrics,
                 const uint32_t count, uint32_t *const indices_out,
                 uint32_t &used_out) noexcept
{
    const Status gate_status = gate.validate ();
    if (!status_ok (gate_status))
        return gate_status;
    used_out = 0u;
    if (metrics == nullptr || indices_out == nullptr)
        return Status::BAD_LAYOUT;
    if (count == 0u || count > _S_avg_max_snapshots)
        return Status::OUT_OF_RANGE;
    if (!simd_aligned (metrics) || !simd_aligned (indices_out))
        return Status::BAD_LAYOUT;
    for (uint32_t index = 0u; index < count; ++index)
        {
            bool accept = false;
            const Status consider = avg_gate_consider (gate, metrics[index], accept);
            if (!status_ok (consider))
                return consider;
            if (accept)
                {
                    indices_out[used_out] = index;
                    ++used_out;
                }
        }
    return used_out == 0u ? Status::OUT_OF_RANGE : Status::OK;
}

/**
 * @brief Average a whole parameter set from per-tensor snapshot sets.
 *
 * @p sets is a parallel array of @p destination ._M_tensors entries; every set's
 * lane count must equal the tensor it feeds, so a mismatched pair of buffers is a
 * BAD_LAYOUT rather than a partial write.
 *
 *   - Uniform averages snapshots [0, snapshots) of each set;
 *   - Swa averages exactly the @p used entries of @p indices (duplicates weigh
 *     again);
 *   - Ema keeps per-tensor int16 state, so it is refused here -- use
 *     `avg_ema_update_set`, which is the same operation once state exists.
 *
 * The report also records how far the result sits from the newest snapshot
 * (`avg_code_drift`), which is the number that tells a caller whether averaging
 * actually moved the weights or just re-emitted the last epoch.
 */
[[nodiscard]] inline Status
avg_average_set (const SnapshotSet *const sets, const ParamSet &destination,
                 const AveragingMode mode, const uint32_t *const indices,
                 const uint32_t used, AveragingReport &report) noexcept
{
    report = AveragingReport{};
    report._M_mode = static_cast<uint8_t> (mode);
    const Status destination_status = destination.validate ();
    if (!status_ok (destination_status))
        return destination_status;
    if (sets == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (sets))
        return Status::BAD_LAYOUT;
    if (mode == AveragingMode::Ema)
        return Status::UNSUPPORTED;
    if (mode == AveragingMode::Swa && (indices == nullptr || used == 0u))
        return Status::BAD_LAYOUT;
    if (mode == AveragingMode::Uniform && indices != nullptr)
        return Status::BAD_LAYOUT;
    const uint32_t tensors = destination._M_tensors;
    for (uint32_t tensor = 0u; tensor < tensors; ++tensor)
        {
            Trainable &param = destination._M_params[tensor];
            const Status parameter_status = param.validate ();
            if (!status_ok (parameter_status))
                return parameter_status;
            if (sets[tensor]._M_valid == 0u)
                return Status::BAD_LAYOUT;
            if (sets[tensor]._M_count != param._M_count)
                return Status::BAD_LAYOUT;
            const Status status
                = mode == AveragingMode::Swa
                      ? avg_swa (sets[tensor], indices, used, param._M_code)
                      : avg_uniform (sets[tensor], 0u, sets[tensor]._M_snapshots,
                                     param._M_code);
            if (!status_ok (status))
                return status;
            ++report._M_tensors;
            report._M_lanes += param._M_count;
            const uint32_t applied
                = mode == AveragingMode::Swa ? used : sets[tensor]._M_snapshots;
            if (applied > report._M_snapshots_used)
                report._M_snapshots_used = applied;
            /* Drift is measured against the newest snapshot: that is the weight set
               averaging replaced, so a zero drift means a no-op average. */
            const int8_t *const newest
                = sets[tensor]._M_streams[sets[tensor]._M_snapshots - 1u];
            uint32_t drifted = 0u;
            int32_t maximum = 0;
            const Status drift_status
                = avg_code_drift (param._M_code, newest, param._M_count, drifted,
                                  maximum);
            if (!status_ok (drift_status))
                return drift_status;
            report._M_drifted_lanes += drifted;
            if (maximum > report._M_max_code_drift)
                report._M_max_code_drift = maximum;
        }
    return Status::OK;
}

/**
 * @brief Gated average: keep only the snapshots whose metric improved, then SWA.
 *
 * The gate is driven purely by the caller's @p metrics series -- one entry per
 * snapshot, in the same order the sets publish them -- so nothing here reads a
 * gradient or a loss tensor of its own.  @p index_scratch is caller-owned storage
 * for at most @p metric_count entries; @p metric_count must not exceed the snapshot
 * count of any set, which is what keeps the index list valid for every tensor.
 *
 * When the gate accepts nothing the average is not attempted: OUT_OF_RANGE comes
 * back with the report counters filled.
 */
[[nodiscard]] inline Status
avg_gated_average (const SnapshotSet *const sets, const ParamSet &destination,
                   AvgGate &gate, const int32_t *const metrics,
                   const uint32_t metric_count, uint32_t *const index_scratch,
                   AveragingReport &report) noexcept
{
    report = AveragingReport{};
    report._M_mode = static_cast<uint8_t> (AveragingMode::Swa);
    const Status destination_status = destination.validate ();
    if (!status_ok (destination_status))
        return destination_status;
    if (sets == nullptr || metrics == nullptr || index_scratch == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (sets) || !simd_aligned (metrics)
        || !simd_aligned (index_scratch))
        return Status::BAD_LAYOUT;
    if (metric_count == 0u || metric_count > _S_avg_max_snapshots)
        return Status::OUT_OF_RANGE;
    for (uint32_t tensor = 0u; tensor < destination._M_tensors; ++tensor)
        if (sets[tensor]._M_valid == 0u || sets[tensor]._M_snapshots < metric_count)
            return Status::BAD_LAYOUT;
    uint32_t used = 0u;
    const Status record_status = avg_gate_record (gate, metrics, metric_count,
                                                 index_scratch, used);
    report._M_gate_accepted = gate._M_accepted;
    report._M_gate_rejected = gate._M_rejected;
    if (!status_ok (record_status))
        return record_status;
    const Status average_status = avg_average_set (sets, destination,
                                                  AveragingMode::Swa, index_scratch,
                                                  used, report);
    if (status_ok (average_status))
        {
            report._M_gate_accepted = gate._M_accepted;
            report._M_gate_rejected = gate._M_rejected;
        }
    return average_status;
}

/**
 * @brief Fold one snapshot into every tensor's EMA state, then read the codes back.
 *
 * @p states is parallel to @p destination ._M_tensors and each state must already
 * be open, over int16 lanes the caller owns, with a lane count equal to the tensor
 * it feeds.  This is the whole EMA loop body: step, narrow, and leave the code
 * stream ready for the next forward pass.  A caller that wants the rail-clamp count
 * reads `EmaState::_M_saturation_events` per tensor.
 */
[[nodiscard]] inline Status
avg_ema_update_set (EmaState *const states, const ParamSet &destination,
                    const int8_t *const *const snapshot,
                    AveragingReport &report) noexcept
{
    report = AveragingReport{};
    report._M_mode = static_cast<uint8_t> (AveragingMode::Ema);
    const Status destination_status = destination.validate ();
    if (!status_ok (destination_status))
        return destination_status;
    if (states == nullptr || snapshot == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (states) || !simd_aligned (snapshot))
        return Status::BAD_LAYOUT;
    const uint32_t tensors = destination._M_tensors;
    for (uint32_t tensor = 0u; tensor < tensors; ++tensor)
        {
            const Status state_status = states[tensor].validate ();
            if (!status_ok (state_status))
                return state_status;
            Trainable &param = destination._M_params[tensor];
            const Status parameter_status = param.validate ();
            if (!status_ok (parameter_status))
                return parameter_status;
            if (snapshot[tensor] == nullptr)
                return Status::BAD_LAYOUT;
            if (!simd_aligned (snapshot[tensor]))
                return Status::BAD_LAYOUT;
            if (states[tensor]._M_count != param._M_count)
                return Status::BAD_LAYOUT;
            const Status step_status = avg_ema_step (states[tensor], snapshot[tensor]);
            if (!status_ok (step_status))
                return step_status;
            const Status read_status
                = avg_ema_read (states[tensor], param._M_code);
            if (!status_ok (read_status))
                return read_status;
            ++report._M_tensors;
            report._M_lanes += states[tensor]._M_count;
            if (states[tensor]._M_updates > report._M_snapshots_used)
                report._M_snapshots_used = states[tensor]._M_updates;
        }
    return Status::OK;
}

/**
 * @brief Publish a parameter set's current codes into caller-owned snapshot buffers.
 *
 * This is the writer side of a SWA ring: the trainer sizes the ring, calls this at
 * each checkpoint interval, and later binds the ring with `avg_snapshots_open`.
 * Codes are copied stride-1, one pass per tensor; nothing here allocates and the
 * destination buffers stay the caller's.
 */
[[nodiscard]] inline Status
avg_snapshot_codes (const ParamSet &source, int8_t *const *const destination) noexcept
{
    const Status source_status = source.validate ();
    if (!status_ok (source_status))
        return source_status;
    if (destination == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (destination))
        return Status::BAD_LAYOUT;
    for (uint32_t tensor = 0u; tensor < source._M_tensors; ++tensor)
        {
            const Trainable &param = source._M_params[tensor];
            const Status parameter_status = param.validate ();
            if (!status_ok (parameter_status))
                return parameter_status;
            if (destination[tensor] == nullptr)
                return Status::BAD_LAYOUT;
            if (!simd_aligned (destination[tensor]))
                return Status::BAD_LAYOUT;
            const uint32_t count = param._M_count;
            for (uint32_t lane = 0u; lane < count; ++lane)
                destination[tensor][lane] = param._M_code[lane];
        }
    return Status::OK;
}

/** @brief Bytes of caller storage an EMA state over @p count lanes costs. */
[[nodiscard]] inline constexpr uint32_t
avg_ema_state_bytes (const uint32_t count) noexcept
{
    return align_up (count * 2u, _S_pool_alignment);
}

/** @brief Bytes of caller storage a snapshot table of @p snapshots streams costs. */
[[nodiscard]] inline constexpr uint32_t
avg_snapshot_table_bytes (const uint32_t snapshots) noexcept
{
    return align_up (snapshots * static_cast<uint32_t> (sizeof (void *)),
                     _S_pool_alignment);
}
} // namespace zzs::mllee::train

#endif // ZZS_MLLEE_TRAIN_MODEL_AVERAGING_HPP
