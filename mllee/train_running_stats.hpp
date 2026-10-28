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

#include "mllee/status.hpp"
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_random.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Running mean/variance buffers for the normalization layers.
 *
 * Batch- and group-norm train on a batch's own statistics but deploy with an
 * exponential moving average of them, so the buffers, the momentum update and
 * the eval-mode freeze live here rather than inside every normalization kind.
 *
 * @par The update, exactly PyTorch's
 *   mean <- (1 - m) mean + m batch_mean
 *   var  <- (1 - m) var  + m unbiased_batch_var
 * with `m` a Q15 momentum (3277 for the usual 0.1) and the batch variance
 * converted from the biased (divide-by-n) form the forward computes to the
 * unbiased one by Bessel's integer ratio n/(n-1).  The ratio is kept at 14
 * fraction bits because it exceeds 1.0 (2.0 for a batch of two), and n <= 1
 * leaves it at exactly 1.0 -- PyTorch's documented convention, since a
 * single-element window has no unbiased estimate.
 *
 * The increment form @ref running_stats_lerp_q15 applies, `V' = V + m (U - V)`
 * with `U` the unbiased batch variance, is the same arithmetic as `(1-m) V + m
 * U` without needing a `1 - m` lane -- 1.0 does not fit one, so the two-term
 * form would bias every update by one ulp (3e-5) of the running value.  The
 * buffers stay readable in eval mode and are frozen by @ref
 * running_stats_freeze, which is what makes the eval pass a pure read.
 *
 * @par Memory
 * Two int16 lanes per window: @ref running_stats_bytes is 4 * windows, each
 * half aligned to the pool's 4-byte requirement.  A snapshot costs the same
 * again in caller-owned storage.
 */
namespace zzs::mllee::train
{
/** @brief Largest window count a stats block addresses. */
inline constexpr uint32_t _S_max_running_windows = 8192u;

/** @brief PyTorch's default momentum 0.1 in Q15. */
inline constexpr int32_t _S_running_momentum_default_q15 = 3277;

/**
 * @brief Running statistics for one normalization layer.
 *
 * The two lane buffers are pool-carved and 2-byte aligned individually, so a
 * SIMD pass over the mean half never crosses into the variance half.
 */
struct RunningStats
{
    int16_t *_M_mean = nullptr;
    int16_t *_M_var = nullptr;
    uint32_t _M_windows = 0u;
    uint32_t _M_step = 0u;
    uint32_t _M_frozen = 0u;
    uint32_t _M_saturated = 0u;
};

static_assert (sizeof (RunningStats) == (sizeof (void *) == 4u ? 24u : 32u),
               "the running-stats view is two pointers plus four counters");

/** @brief Pool bytes of the two lane buffers for @p windows windows. */
[[nodiscard]] inline constexpr uint32_t
running_stats_bytes (const uint32_t windows) noexcept
{
    return 4u * windows;
}

/** @brief Pool bytes of a checkpoint copy of the same block. */
[[nodiscard]] inline constexpr uint32_t
running_stats_snapshot_bytes (const uint32_t windows) noexcept
{
    return 4u * windows + 4u;
}

/**
 * @brief Bessel's correction n/(n-1) at 14 fraction bits.
 *
 * A window of one or zero elements has no unbiased variance, so the ratio
 * collapses to 1.0, matching the reference implementation.
 */
[[nodiscard]] inline constexpr int32_t
bessel_ratio_q14 (const uint32_t elements) noexcept
{
    if (elements <= 1u)
        return 16384;
    const uint64_t scaled = (static_cast<uint64_t> (elements) << 14u)
                            / (elements - 1u);
    return scaled > 0xFFFFull ? 16384 : static_cast<int32_t> (scaled);
}

/** @brief Turn a biased window variance into the unbiased convention. */
[[nodiscard]] inline constexpr int32_t
running_stats_unbiased_q15 (const int32_t biased_var_q15,
                            const uint32_t elements) noexcept
{
    const int32_t ratio = bessel_ratio_q14 (elements);
    const int32_t value = q15_scale_shift (biased_var_q15, ratio, 14u);
    return q15_clamp (value);
}

/** @brief Bind already-owned lane buffers (the non-pool path, e.g. a .bss block). */
[[nodiscard]] inline Status
running_stats_bind (RunningStats &stats, int16_t *const mean,
                    int16_t *const var, const uint32_t windows) noexcept
{
    if (mean == nullptr || var == nullptr)
        return Status::BAD_LAYOUT;
    if (windows == 0u || windows > _S_max_running_windows)
        return Status::OUT_OF_RANGE;
    stats._M_mean = mean;
    stats._M_var = var;
    stats._M_windows = windows;
    stats._M_step = 0u;
    stats._M_frozen = 0u;
    stats._M_saturated = 0u;
    return Status::OK;
}

/** @brief Carve and zero the two buffers from the training pool. */
[[nodiscard]] inline Status
running_stats_alloc (Pool &pool, RunningStats &stats,
                     const uint32_t windows) noexcept
{
    if (windows == 0u || windows > _S_max_running_windows)
        return Status::OUT_OF_RANGE;
    int16_t *mean = nullptr;
    int16_t *var = nullptr;
    const Status first = pool.carve_typed (windows, mean);
    if (!status_ok (first))
        {
            stats = RunningStats{};
            return first;
        }
    const Status second = pool.carve_typed (windows, var);
    if (!status_ok (second))
        {
            stats = RunningStats{};
            return second;
        }
    return running_stats_bind (stats, mean, var, windows);
}

/** @brief Zero both buffers and the counters: the "fresh layer" state. */
inline void
running_stats_reset (RunningStats &stats) noexcept
{
    if (stats._M_mean != nullptr)
        fill_lanes (stats._M_mean, 0, stats._M_windows);
    if (stats._M_var != nullptr)
        fill_lanes (stats._M_var, 0, stats._M_windows);
    stats._M_step = 0u;
    stats._M_saturated = 0u;
}

/** @brief Eval mode: the buffers stop moving but stay readable. */
inline void
running_stats_freeze (RunningStats &stats) noexcept
{
    stats._M_frozen = 1u;
}

/** @brief Training mode again. */
inline void
running_stats_thaw (RunningStats &stats) noexcept
{
    stats._M_frozen = 0u;
}

/** @brief True when @p stats ignores updates (and the norm layer reads it). */
[[nodiscard]] inline constexpr bool
running_stats_active (const RunningStats &stats) noexcept
{
    return stats._M_frozen == 0u && stats._M_mean != nullptr
           && stats._M_var != nullptr;
}

/**
 * @brief One Q15 lerp: x + m (y - x), rounded, with the saturation reported.
 *
 * The algebraically identical (1-m) x + m y needs a 1-m lane, which is one ulp
 * short of 1.0 and biases every frozen-buffer read by that amount; the lerp
 * form is exact at both m = 0 and m = 1.
 */
[[nodiscard]] inline constexpr int32_t
running_stats_lerp_q15 (const int32_t current_q15, const int32_t target_q15,
                        const int32_t momentum_q15, bool &saturated) noexcept
{
    const int32_t delta = target_q15 - current_q15;
    const int32_t step = q15_scale_shift (delta, momentum_q15, _S_grad_frac_bits);
    const int32_t raw = current_q15 + step;
    const int32_t value = q15_clamp (raw);
    saturated = value != raw;
    return value;
}

/**
 * @brief One momentum step over every window.
 *
 * @param batch_mean per-window batch mean, Q15.
 * @param batch_var  per-window batch variance in the biased (divide-by-n) form,
 *                   Q15; converted here.
 * @param elements   window size, the Bessel ratio's argument.
 * @param momentum_q15 the mixing weight, clamped to [0, 1] as a Q15 value.
 *
 * A frozen block is a no-op that still returns OK, so the caller does not need
 * a second code path between train and eval.
 */
[[nodiscard]] inline Status
running_stats_update (RunningStats &stats, const int16_t *const batch_mean,
                      const int16_t *const batch_var, const uint32_t elements,
                      const int32_t momentum_q15) noexcept
{
    if (!running_stats_active (stats))
        return Status::OK;
    if (batch_mean == nullptr || batch_var == nullptr)
        return Status::BAD_LAYOUT;
    const int32_t momentum = momentum_q15 < 0
                                 ? 0
                                 : (momentum_q15 > _S_q15_max ? _S_q15_max
                                                              : momentum_q15);
    for (uint32_t window = 0u; window < stats._M_windows; ++window)
        {
            bool saturated = false;
            const int32_t mean_next = running_stats_lerp_q15 (
                stats._M_mean[window], batch_mean[window], momentum, saturated);
            if (saturated)
                ++stats._M_saturated;
            const int32_t unbiased = running_stats_unbiased_q15 (
                batch_var[window], elements);
            const int32_t var_next = running_stats_lerp_q15 (
                stats._M_var[window], unbiased, momentum, saturated);
            if (saturated)
                ++stats._M_saturated;
            stats._M_mean[window] = static_cast<int16_t> (mean_next);
            stats._M_var[window] = static_cast<int16_t> (var_next);
        }
    ++stats._M_step;
    return Status::OK;
}

/**
 * @brief Copy the buffers and the counter into caller-owned storage.
 *
 * @p scratch must be @ref running_stats_snapshot_bytes bytes: two int16 arrays
 * and the step counter as one int32.  Layout is mean, var, step, so a
 * checkpoint file can be read back with the same three slices.
 */
[[nodiscard]] inline Status
running_stats_snapshot (const RunningStats &stats, uint8_t *const scratch) noexcept
{
    if (scratch == nullptr || !running_stats_active (stats))
        return Status::BAD_LAYOUT;
    if (!simd_aligned (scratch))
        return Status::BAD_LAYOUT;
    int16_t *mean = reinterpret_cast<int16_t *> (scratch);
    int16_t *var = mean + stats._M_windows;
    for (uint32_t window = 0u; window < stats._M_windows; ++window)
        {
            mean[window] = stats._M_mean[window];
            var[window] = stats._M_var[window];
        }
    int32_t *step = reinterpret_cast<int32_t *> (
        scratch + 4u * stats._M_windows);
    *step = static_cast<int32_t> (stats._M_step);
    return Status::OK;
}

/** @brief The inverse of @ref running_stats_snapshot. */
[[nodiscard]] inline Status
running_stats_restore (RunningStats &stats,
                       const uint8_t *const scratch) noexcept
{
    if (scratch == nullptr || !running_stats_active (stats))
        return Status::BAD_LAYOUT;
    const int16_t *mean = reinterpret_cast<const int16_t *> (scratch);
    const int16_t *var = mean + stats._M_windows;
    for (uint32_t window = 0u; window < stats._M_windows; ++window)
        {
            stats._M_mean[window] = mean[window];
            stats._M_var[window] = var[window];
        }
    const int32_t *step = reinterpret_cast<const int32_t *> (
        scratch + 4u * stats._M_windows);
    stats._M_step = static_cast<uint32_t> (*step < 0 ? 0 : *step);
    return Status::OK;
}

/**
 * @brief The inference form of one window: the pair the core's norm op wants.
 *
 * `_S_window_stats` works in code units and `_S_norm_value` divides the
 * centered code by the deviation, so deploying a trained layer needs the mean
 * and the standard deviation on the tensor's own code grid, not in Q15.  The
 * variance is taken in the unbiased convention the buffers keep, the root is
 * the floor integer square root (the same `_S_isqrt` the op uses), and the
 * result is a pair of plain integer counts.
 */
[[nodiscard]] inline Status
running_stats_deploy_form (const RunningStats &stats, const uint32_t window,
                           const uint32_t index, int32_t &mean_code,
                           uint32_t &deviation_code) noexcept
{
    if (stats._M_mean == nullptr || stats._M_var == nullptr)
        return Status::BAD_LAYOUT;
    if (index >= stats._M_windows || window == 0u)
        return Status::OUT_OF_RANGE;
    const int32_t mean_q15 = stats._M_mean[index];
    const int32_t var_q15 = stats._M_var[index];
    mean_code = q15_to_code (mean_q15);
    /* The variance is a Q15 magnitude whose root wants 15 fraction bits, so it
       goes through the wide root: sqrt (var) * 2^15 / 2^15 = sqrt (var << 15). */
    const int32_t deviation_q15 = q15_sqrt_wide (var_q15);
    deviation_code = static_cast<uint32_t> (
        _S_round_shift (static_cast<int64_t> (deviation_q15) * 127,
                        _S_grad_frac_bits));
    return Status::OK;
}
} // namespace zzs::mllee::train
