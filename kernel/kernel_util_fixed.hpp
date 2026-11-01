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

#include "kernel/kernel_config.hpp"
#include "kernel/kernel_fixed.hpp"
#include "kernel/kernel_kernel_fixed_simd.hpp"

#include <cstdint>

/**
 * @brief Additional fixed-point helpers used by the feature stages.
 *
 * Vectorized RMS (sqrt of mean of squares), a sorting-network median for the
 * fixed 5-window DC history (O(1) compare-exchanges, no data-dependent loop)
 * plus a selection-based fallback for larger counts, and a spectral-centroid
 * numerator/denominator pair over Q14 bin magnitudes.  All integer-only; no
 * sorting of the full data set.
 * @{
 */
namespace zzs::kernel::fx
{

/** @brief Compare-exchange: after the call, `a <= b`. */
inline void
ce5 (int16_t &a, int16_t &b) noexcept
{
    if (a > b)
        {
            const int16_t t = a;
            a = b;
            b = t;
        }
}

/** @brief Vector RMS of `count` Q15 samples, returned as Q15.
 *
 * `rms = sqrt(mean(x^2))`.  The sum of squares uses `simd::sum_sq_q15` and the
 * integer square root is `_S_isqrt`.  Scale: sum_sq is Q30 (Q15*xQ15), the mean
 * divides by count, then sqrt returns Q15. */
[[nodiscard]] inline int32_t
rms_q15 (const int16_t *const in, const uint32_t count) noexcept
{
    if (count == 0u)
        return 0;
    const int64_t sum_sq = zzs::kernel::simd::sum_sq_q15 (in, count);
    /* mean = sum_sq / count (Q30).  Use the magic reciprocal. */
    const zzs::mllee::_S_Div32 div = zzs::mllee::_S_div32_make (
        count == 0u ? 1u : count);
    const uint32_t mean_q30 = zzs::mllee::_S_div32_u (
        static_cast<uint32_t> (sum_sq > 0 ? sum_sq : 0), div);
    /* sqrt(Q30) -> Q15: isqrt of mean_q30. */
    return static_cast<int32_t> (zzs::mllee::_S_isqrt (mean_q30));
}

/** @brief Vector peak magnitude of `count` Q15 samples, returned as Q15. */
[[nodiscard]] inline int32_t
peak_q15 (const int16_t *const in, const uint32_t count) noexcept
{
    if (count == 0u)
        return 0;
    return static_cast<int32_t> (zzs::kernel::simd::abs_max_q15 (in, count));
}

/** @brief Vector mean of `count` Q15 samples, returned as Q15. */
[[nodiscard]] inline int32_t
mean_q15 (const int16_t *const in, const uint32_t count) noexcept
{
    if (count == 0u)
        return 0;
    const int64_t sum = zzs::kernel::simd::sum_q15 (in, count);
    return zzs::kernel::fx::window_mean_i32 (
        sum, zzs::kernel::fx::window_divide (count, 0u));
}

/** @brief Median of `count` Q15 samples.
 *
 * For `count <= 5` (the flicker 5-window DC history) a 5-element sorting
 * network is used: exactly 10 compare-exchanges, fully unrolled, O(1) with no
 * data-dependent loop — fully predictable on a Cortex-M4.  For larger counts a
 * selection-based fallback is used.
 */
[[nodiscard]] inline int32_t
median_q15 (const int16_t *const in, const uint32_t count) noexcept
{
    if (count == 0u)
        return 0;
    if (count <= 5u)
        {
            int16_t v[5];
            for (uint32_t i = 0; i < count; ++i)
                v[i] = in[i];
            /* 5-element sorting network (10 compare-exchanges). */
            ce5 (v[0], v[1]);
            ce5 (v[3], v[4]);
            ce5 (v[2], v[4]);
            ce5 (v[2], v[3]);
            ce5 (v[1], v[4]);
            ce5 (v[0], v[3]);
            ce5 (v[0], v[2]);
            ce5 (v[1], v[3]);
            ce5 (v[1], v[2]);
            ce5 (v[3], v[4]);
            return static_cast<int32_t> (v[count / 2u]);
        }
    /* Fallback: selection-based median for larger counts (non-destructive). */
    const uint32_t k = count / 2u; /* lower median index */
    int16_t low = in[0];
    int16_t high = in[0];
    for (uint32_t i = 1; i < count; ++i)
        {
            if (in[i] < low)
                low = in[i];
            if (in[i] > high)
                high = in[i];
        }
    int16_t lo = low;
    int16_t hi = high;
    while (lo < hi)
        {
            const int16_t mid = static_cast<int16_t> (
                (static_cast<int32_t> (lo) + hi) >> 1);
            uint32_t le_count = 0;
            int16_t le_max = lo;
            for (uint32_t i = 0; i < count; ++i)
                {
                    if (in[i] <= mid)
                        {
                            ++le_count;
                            if (in[i] > le_max)
                                le_max = in[i];
                        }
                }
            if (le_count > k)
                hi = le_max;
            else
                lo = static_cast<int16_t> (mid + 1);
        }
    return static_cast<int32_t> (lo);
}

/** @brief Spectral centroid numerator and denominator over bin magnitudes.
 *
 * `centroid = sum(mag[i] * bin_i) / sum(mag[i])`, returned as a Q15 fraction of
 * the Nyquist bin.  `mags` holds `count` Q7 magnitudes; `bin_i` is the bin
 * index.  Uses `simd::dot_widen_q15` for the weighted sum and `simd::sum_q15`
 * for the magnitude sum. */
[[nodiscard]] inline int32_t
spectral_centroid_q15 (const int16_t *const mags, const uint32_t count) noexcept
{
    if (count == 0u)
        return 0;
    int64_t weighted_sum = 0;
    int64_t mag_sum = 0;
    for (uint32_t i = 0; i < count; ++i)
        {
            weighted_sum += static_cast<int64_t> (mags[i]) * static_cast<int64_t> (i);
            mag_sum += mags[i];
        }
    if (mag_sum == 0)
        return 0;
    /* Magic-reciprocal divide replaces the runtime 64-bit / by mag_sum.
       weighted_sum >= 0 (mags are magnitudes) and mag_sum > 0, so unsigned
       division gives the same floor result as the original signed /. */
    const uint32_t num = static_cast<uint32_t> (
        weighted_sum << zzs::kernel::_S_state_frac_bits);
    const zzs::mllee::_S_Div32 div = zzs::mllee::_S_div32_make (
        static_cast<uint32_t> (mag_sum));
    const uint32_t centroid = zzs::mllee::_S_div32_u (num, div);
    return static_cast<int32_t> (centroid);
}

/** @} */
} // namespace zzs::kernel::fx
