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

#include <cstdint>

/**
 * @brief Vectorized fixed-point primitives for the preprocessing bank.
 *
 * Every function here has a portable scalar fallback that computes the same
 * int32/int64 intermediate as the vector tier, so `-DZZS_KERNEL_SCALAR=1`
 * stays a meaningful differential test.  Lane count is `_S_simd_lanes` (32 at
 * AVX-512, 16 at AVX2, 8 at NEON/SSE2, 1 otherwise).
 *
 * @section Storage
 *
 * Q15 samples live as int16 in memory; products widen to int32/int64
 * accumulators so a 200-tap FIR or a 200-sample power sum never overflows.
 * No function here touches floating point.
 * @{
 */
namespace zzs::kernel::simd {

using zzs::kernel::fx::q15_abs;
using zzs::kernel::fx::q15_add_sat;
using zzs::kernel::fx::q15_mul;

/** @brief Sum of `lhs[i] * rhs[i]` over `count` Q15 samples, as int64.
 *
 * Every tier (NEON, SSE2, AVX2, AVX-512, scalar) returns the identical int64
 * for the same inputs: the widening int16x16 -> int32 product is accumulated
 * in int32 lanes then reduced to int64 at the end of each vector block, so no
 * tier's intermediate rounding differs.
 */
[[nodiscard]] inline int64_t
dot_widen_q15 (const int16_t *const ZZS_KERNEL_RESTRICT lhs, const int16_t *const ZZS_KERNEL_RESTRICT rhs,
               const uint32_t count) noexcept {
#if ZZS_KERNEL_AVX512
        int64_t total = 0;
        uint32_t i = 0;
        for (; i + 32u <= count; i += 32u) {
                __m256i acc_lo = _mm256_setzero_si256 ();
                __m256i acc_hi = _mm256_setzero_si256 ();
                for (uint32_t k = 0; k < 32u; k += 16u) {
                        const __m256i a = _mm256_loadu_si256 (reinterpret_cast<const __m256i *> (lhs + i + k));
                        const __m256i b = _mm256_loadu_si256 (reinterpret_cast<const __m256i *> (rhs + i + k));
                        const __m256i a_lo = _mm256_cvtepi16_epi32 (_mm256_extracti128_si256 (a, 0));
                        const __m256i a_hi = _mm256_cvtepi16_epi32 (_mm256_extracti128_si256 (a, 1));
                        const __m256i b_lo = _mm256_cvtepi16_epi32 (_mm256_extracti128_si256 (b, 0));
                        const __m256i b_hi = _mm256_cvtepi16_epi32 (_mm256_extracti128_si256 (b, 1));
                        acc_lo = _mm256_add_epi32 (acc_lo, _mm256_mullo_epi32 (a_lo, b_lo));
                        acc_hi = _mm256_add_epi32 (acc_hi, _mm256_mullo_epi32 (a_hi, b_hi));
                }
                __m128i sum_lo = _mm_add_epi32 (_mm256_castsi256_si128 (acc_lo), _mm256_extractf128_si256 (acc_lo, 1));
                __m128i sum_hi = _mm_add_epi32 (_mm256_castsi256_si128 (acc_hi), _mm256_extractf128_si256 (acc_hi, 1));
                sum_lo = _mm_add_epi32 (sum_lo, _mm_shuffle_epi32 (sum_lo, 0x4e));
                sum_lo = _mm_add_epi32 (sum_lo, _mm_shuffle_epi32 (sum_lo, 0xb1));
                sum_hi = _mm_add_epi32 (sum_hi, _mm_shuffle_epi32 (sum_hi, 0x4e));
                sum_hi = _mm_add_epi32 (sum_hi, _mm_shuffle_epi32 (sum_hi, 0xb1));
                total += static_cast<int64_t> (_mm_cvtsi128_si32 (sum_lo))
                         + static_cast<int64_t> (_mm_cvtsi128_si32 (sum_hi));
        }
        for (; i < count; ++i)
                total += static_cast<int64_t> (lhs[i]) * rhs[i];
        return total;
#elif ZZS_KERNEL_AVX2
        int64_t total = 0;
        uint32_t i = 0;
        for (; i + 16u <= count; i += 16u) {
                __m256i acc = _mm256_setzero_si256 ();
                for (uint32_t k = 0; k < 16u; k += 8u) {
                        const __m256i a = _mm256_loadu_si256 (reinterpret_cast<const __m256i *> (lhs + i + k));
                        const __m256i b = _mm256_loadu_si256 (reinterpret_cast<const __m256i *> (rhs + i + k));
                        const __m256i a_lo = _mm256_cvtepi16_epi32 (_mm256_extracti128_si256 (a, 0));
                        const __m256i a_hi = _mm256_cvtepi16_epi32 (_mm256_extracti128_si256 (a, 1));
                        const __m256i b_lo = _mm256_cvtepi16_epi32 (_mm256_extracti128_si256 (b, 0));
                        const __m256i b_hi = _mm256_cvtepi16_epi32 (_mm256_extracti128_si256 (b, 1));
                        acc = _mm256_add_epi32 (acc, _mm256_mullo_epi32 (a_lo, b_lo));
                        acc = _mm256_add_epi32 (acc, _mm256_mullo_epi32 (a_hi, b_hi));
                }
                __m128i sum = _mm_add_epi32 (_mm256_castsi256_si128 (acc), _mm256_extractf128_si256 (acc, 1));
                sum = _mm_add_epi32 (sum, _mm_shuffle_epi32 (sum, 0x4e));
                sum = _mm_add_epi32 (sum, _mm_shuffle_epi32 (sum, 0xb1));
                total += static_cast<int64_t> (_mm_cvtsi128_si32 (sum));
        }
        for (; i < count; ++i)
                total += static_cast<int64_t> (lhs[i]) * rhs[i];
        return total;
#elif ZZS_KERNEL_NEON
        int64_t total = 0;
        uint32_t i = 0;
        for (; i + 8u <= count; i += 8u) {
                const int16x8_t a = vld1q_s16 (lhs + i);
                const int16x8_t b = vld1q_s16 (rhs + i);
                const int32x4_t acc_lo = vmull_s16 (vget_low_s16 (a), vget_low_s16 (b));
                const int32x4_t acc_hi = vmull_s16 (vget_high_s16 (a), vget_high_s16 (b));
                total += static_cast<int64_t> (vgetq_lane_s32 (acc_lo, 0)) + vgetq_lane_s32 (acc_lo, 1)
                         + vgetq_lane_s32 (acc_lo, 2) + vgetq_lane_s32 (acc_lo, 3) + vgetq_lane_s32 (acc_hi, 0)
                         + vgetq_lane_s32 (acc_hi, 1) + vgetq_lane_s32 (acc_hi, 2) + vgetq_lane_s32 (acc_hi, 3);
        }
        for (; i < count; ++i)
                total += static_cast<int64_t> (lhs[i]) * rhs[i];
        return total;
#elif ZZS_KERNEL_SSE2
        int64_t total = 0;
        uint32_t i = 0;
        for (; i + 8u <= count; i += 8u) {
                const __m128i a = _mm_loadu_si128 (reinterpret_cast<const __m128i *> (lhs + i));
                const __m128i b = _mm_loadu_si128 (reinterpret_cast<const __m128i *> (rhs + i));
                const __m128i a_lo = _mm_cvtepi16_epi32 (_mm_unpacklo_epi16 (a, _mm_setzero_si128 ()));
                const __m128i a_hi = _mm_cvtepi16_epi32 (_mm_unpackhi_epi16 (a, _mm_setzero_si128 ()));
                const __m128i b_lo = _mm_cvtepi16_epi32 (_mm_unpacklo_epi16 (b, _mm_setzero_si128 ()));
                const __m128i b_hi = _mm_cvtepi16_epi32 (_mm_unpackhi_epi16 (b, _mm_setzero_si128 ()));
                __m128i acc = _mm_add_epi32 (_mm_mullo_epi32 (a_lo, b_lo), _mm_mullo_epi32 (a_hi, b_hi));
                acc = _mm_add_epi32 (acc, _mm_shuffle_epi32 (acc, 0x4e));
                acc = _mm_add_epi32 (acc, _mm_shuffle_epi32 (acc, 0xb1));
                total += static_cast<int64_t> (_mm_cvtsi128_si32 (acc));
        }
        for (; i < count; ++i)
                total += static_cast<int64_t> (lhs[i]) * rhs[i];
        return total;
#else
        int64_t total = 0;
        for (uint32_t i = 0; i < count; ++i)
                total += static_cast<int64_t> (lhs[i]) * rhs[i];
        return total;
#endif
}

/** @brief Sum of `in[i]^2` over `count` Q15 samples, as int64. */
[[nodiscard]] inline int64_t
sum_sq_q15 (const int16_t *const ZZS_KERNEL_RESTRICT in, const uint32_t count) noexcept {
        return dot_widen_q15 (in, in, count);
}

/** @brief Sum of `in[i]` over `count` Q15 samples, as int64. */
[[nodiscard]] inline int64_t
sum_q15 (const int16_t *const ZZS_KERNEL_RESTRICT in, const uint32_t count) noexcept {
#if ZZS_KERNEL_NEON || ZZS_KERNEL_SSE2 || ZZS_KERNEL_AVX2 || ZZS_KERNEL_AVX512
        int64_t total = 0;
        uint32_t i = 0;
#if ZZS_KERNEL_NEON
        for (; i + 8u <= count; i += 8u) {
                const int16x8_t v = vld1q_s16 (in + i);
                const int32x4_t lo = vmovl_s16 (vget_low_s16 (v));
                const int32x4_t hi = vmovl_s16 (vget_high_s16 (v));
                total += static_cast<int64_t> (vgetq_lane_s32 (lo, 0)) + vgetq_lane_s32 (lo, 1) + vgetq_lane_s32 (lo, 2)
                         + vgetq_lane_s32 (lo, 3) + vgetq_lane_s32 (hi, 0) + vgetq_lane_s32 (hi, 1)
                         + vgetq_lane_s32 (hi, 2) + vgetq_lane_s32 (hi, 3);
        }
#elif ZZS_KERNEL_SSE2 || ZZS_KERNEL_AVX2 || ZZS_KERNEL_AVX512
        for (; i + 8u <= count; i += 8u) {
                const __m128i v = _mm_loadu_si128 (reinterpret_cast<const __m128i *> (in + i));
                const __m128i lo = _mm_cvtepi16_epi32 (_mm_unpacklo_epi16 (v, _mm_setzero_si128 ()));
                const __m128i hi = _mm_cvtepi16_epi32 (_mm_unpackhi_epi16 (v, _mm_setzero_si128 ()));
                __m128i acc = _mm_add_epi32 (lo, hi);
                acc = _mm_add_epi32 (acc, _mm_shuffle_epi32 (acc, 0x4e));
                acc = _mm_add_epi32 (acc, _mm_shuffle_epi32 (acc, 0xb1));
                total += static_cast<int64_t> (_mm_cvtsi128_si32 (acc));
        }
#endif
        for (; i < count; ++i)
                total += in[i];
        return total;
#else
        int64_t total = 0;
        for (uint32_t i = 0; i < count; ++i)
                total += in[i];
        return total;
#endif
}

/** @brief Peak |in[i]| over `count` Q15 samples, returned as int16. */
[[nodiscard]] inline int16_t
abs_max_q15 (const int16_t *const ZZS_KERNEL_RESTRICT in, const uint32_t count) noexcept {
#if ZZS_KERNEL_NEON
        int16x8_t acc = vdupq_n_s16 (0);
        uint32_t i = 0;
        for (; i + 8u <= count; i += 8u) {
                const int16x8_t v = vld1q_s16 (in + i);
                acc = vmaxq_s16 (acc, vabsq_s16 (v));
        }
        int16_t best = static_cast<int16_t> (vgetq_lane_s16 (acc, 0));
        best = static_cast<int16_t> (vgetq_lane_s16 (acc, 1) > best ? vgetq_lane_s16 (acc, 1) : best);
        best = static_cast<int16_t> (vgetq_lane_s16 (acc, 2) > best ? vgetq_lane_s16 (acc, 2) : best);
        best = static_cast<int16_t> (vgetq_lane_s16 (acc, 3) > best ? vgetq_lane_s16 (acc, 3) : best);
        best = static_cast<int16_t> (vgetq_lane_s16 (acc, 4) > best ? vgetq_lane_s16 (acc, 4) : best);
        best = static_cast<int16_t> (vgetq_lane_s16 (acc, 5) > best ? vgetq_lane_s16 (acc, 5) : best);
        best = static_cast<int16_t> (vgetq_lane_s16 (acc, 6) > best ? vgetq_lane_s16 (acc, 6) : best);
        best = static_cast<int16_t> (vgetq_lane_s16 (acc, 7) > best ? vgetq_lane_s16 (acc, 7) : best);
        for (; i < count; ++i) {
                const int32_t v = q15_abs (in[i]);
                best = static_cast<int16_t> (v > q15_abs (best) ? in[i] : best);
        }
        return static_cast<int16_t> (q15_abs (best));
#else
        int32_t best = 0;
        for (uint32_t i = 0; i < count; ++i) {
                const int32_t v = q15_abs (in[i]);
                best = v > best ? v : best;
        }
        return static_cast<int16_t> (best);
#endif
}

/** @brief Q15 x Q15 -> Q15 saturating, lane by lane, `count` outputs. */
inline void
q15_mul_lanes (const int16_t *const ZZS_KERNEL_RESTRICT lhs, const int16_t *const ZZS_KERNEL_RESTRICT rhs,
               int16_t *const ZZS_KERNEL_RESTRICT out, const uint32_t count) noexcept {
#if ZZS_KERNEL_NEON
        uint32_t i = 0;
        for (; i + 8u <= count; i += 8u) {
                const int16x8_t a = vld1q_s16 (lhs + i);
                const int16x8_t b = vld1q_s16 (rhs + i);
                /* Widen to int32, multiply, add Q15 rounding term, shift,
                   saturate-narrow back to int16 per lane. */
                const int32x4_t prod_lo = vmull_s16 (vget_low_s16 (a), vget_low_s16 (b));
                const int32x4_t prod_hi = vmull_s16 (vget_high_s16 (a), vget_high_s16 (b));
                const int32x4_t rnd = vdupq_n_s32 (1 << 14);
                const int32x4_t sum_lo = vaddq_s32 (prod_lo, rnd);
                const int32x4_t sum_hi = vaddq_s32 (prod_hi, rnd);
                const int32x4_t sh_lo = vshrq_n_s32 (sum_lo, 15);
                const int32x4_t sh_hi = vshrq_n_s32 (sum_hi, 15);
                const int16x4_t res_lo = vqmovn_s32 (sh_lo);
                const int16x4_t res_hi = vqmovn_s32 (sh_hi);
                vst1q_s16 (out + i, vcombine_s16 (res_lo, res_hi));
        }
        for (; i < count; ++i)
                out[i] = static_cast<int16_t> (fx::q15_mul (lhs[i], rhs[i]));
#else
        for (uint32_t i = 0; i < count; ++i)
                out[i] = static_cast<int16_t> (fx::q15_mul (lhs[i], rhs[i]));
#endif
}

/** @brief Saturating Q15 add `lhs[i] + rhs[i]`, lane by lane, `count` outputs. */
inline void
q15_add_sat_lanes (const int16_t *const ZZS_KERNEL_RESTRICT lhs, const int16_t *const ZZS_KERNEL_RESTRICT rhs,
                   int16_t *const ZZS_KERNEL_RESTRICT out, const uint32_t count) noexcept {
#if ZZS_KERNEL_NEON
        uint32_t i = 0;
        for (; i + 8u <= count; i += 8u) {
                const int16x8_t a = vld1q_s16 (lhs + i);
                const int16x8_t b = vld1q_s16 (rhs + i);
                vst1q_s16 (out + i, vqaddq_s16 (a, b));
        }
        for (; i < count; ++i)
                out[i] = static_cast<int16_t> (fx::q15_add_sat (lhs[i], rhs[i]));
#elif ZZS_KERNEL_SSE2 || ZZS_KERNEL_AVX2 || ZZS_KERNEL_AVX512
        uint32_t i = 0;
        for (; i + 8u <= count; i += 8u) {
                const __m128i a = _mm_loadu_si128 (reinterpret_cast<const __m128i *> (lhs + i));
                const __m128i b = _mm_loadu_si128 (reinterpret_cast<const __m128i *> (rhs + i));
                /* SSE2 lacks saturating adds; clamp manually per 16-bit lane. */
                __m128i sum = _mm_add_epi16 (a, b);
                const __m128i min16 = _mm_set1_epi16 (-32768);
                const __m128i max16 = _mm_set1_epi16 (32767);
                sum = _mm_max_epi16 (sum, min16);
                sum = _mm_min_epi16 (sum, max16);
                _mm_storeu_si128 (reinterpret_cast<__m128i *> (out + i), sum);
        }
        for (; i < count; ++i)
                out[i] = static_cast<int16_t> (fx::q15_add_sat (lhs[i], rhs[i]));
#else
        for (uint32_t i = 0; i < count; ++i)
                out[i] = static_cast<int16_t> (fx::q15_add_sat (lhs[i], rhs[i]));
#endif
}

/** @brief Saturating Q15 absolute value, lane by lane, `count` outputs. */
inline void
q15_abs_lanes (const int16_t *const ZZS_KERNEL_RESTRICT in, int16_t *const ZZS_KERNEL_RESTRICT out,
               const uint32_t count) noexcept {
#if ZZS_KERNEL_NEON
        uint32_t i = 0;
        for (; i + 8u <= count; i += 8u) {
                vst1q_s16 (out + i, vabsq_s16 (vld1q_s16 (in + i)));
        }
        for (; i < count; ++i)
                out[i] = static_cast<int16_t> (fx::q15_abs (in[i]));
#elif ZZS_KERNEL_SSE2 || ZZS_KERNEL_AVX2 || ZZS_KERNEL_AVX512
        uint32_t i = 0;
        for (; i + 8u <= count; i += 8u) {
                const __m128i v = _mm_loadu_si128 (reinterpret_cast<const __m128i *> (in + i));
                _mm_storeu_si128 (reinterpret_cast<__m128i *> (out + i), _mm_abs_epi16 (v));
        }
        for (; i < count; ++i)
                out[i] = static_cast<int16_t> (fx::q15_abs (in[i]));
#else
        for (uint32_t i = 0; i < count; ++i)
                out[i] = static_cast<int16_t> (fx::q15_abs (in[i]));
#endif
}

/**
 * @brief Saturating vector clip: every lane of `in` is clamped to [lo, hi]
 *        Q15 and written to `out`.  `count` outputs.
 */
inline void
q15_clip_lanes (const int16_t *const ZZS_KERNEL_RESTRICT in, const int16_t lo_q15, const int16_t hi_q15,
                int16_t *const ZZS_KERNEL_RESTRICT out, const uint32_t count) noexcept {
#if ZZS_KERNEL_NEON
        const int16x8_t v_lo = vdupq_n_s16 (lo_q15);
        const int16x8_t v_hi = vdupq_n_s16 (hi_q15);
        uint32_t i = 0;
        for (; i + 8u <= count; i += 8u) {
                const int16x8_t v = vld1q_s16 (in + i);
                int16x8_t clipped = vmaxq_s16 (v, v_lo);
                clipped = vminq_s16 (clipped, v_hi);
                vst1q_s16 (out + i, clipped);
        }
        for (; i < count; ++i) {
                const int32_t v = in[i];
                out[i] = static_cast<int16_t> (v < lo_q15 ? lo_q15 : (v > hi_q15 ? hi_q15 : v));
        }
#elif ZZS_KERNEL_SSE2 || ZZS_KERNEL_AVX2 || ZZS_KERNEL_AVX512
        const __m128i v_lo = _mm_set1_epi16 (lo_q15);
        const __m128i v_hi = _mm_set1_epi16 (hi_q15);
        uint32_t i = 0;
        for (; i + 8u <= count; i += 8u) {
                const __m128i v = _mm_loadu_si128 (reinterpret_cast<const __m128i *> (in + i));
                __m128i clipped = _mm_max_epi16 (v, v_lo);
                clipped = _mm_min_epi16 (clipped, v_hi);
                _mm_storeu_si128 (reinterpret_cast<__m128i *> (out + i), clipped);
        }
        for (; i < count; ++i) {
                const int32_t v = in[i];
                out[i] = static_cast<int16_t> (v < lo_q15 ? lo_q15 : (v > hi_q15 ? hi_q15 : v));
        }
#else
        for (uint32_t i = 0; i < count; ++i) {
                const int32_t v = in[i];
                out[i] = static_cast<int16_t> (v < lo_q15 ? lo_q15 : (v > hi_q15 ? hi_q15 : v));
        }
#endif
}

/**
 * @brief Vector accumulate-with-wrap: `out[i] = (out[i] + in[i])` with
 *        int16 wrapping (two's complement, no saturation).  `count` outputs.
 */
inline void
q15_add_wrap_lanes (const int16_t *const ZZS_KERNEL_RESTRICT in, int16_t *const ZZS_KERNEL_RESTRICT out,
                    const uint32_t count) noexcept {
#if ZZS_KERNEL_NEON
        uint32_t i = 0;
        for (; i + 8u <= count; i += 8u) {
                const int16x8_t a = vld1q_s16 (out + i);
                const int16x8_t b = vld1q_s16 (in + i);
                vst1q_s16 (out + i, vaddq_s16 (a, b));
        }
        for (; i < count; ++i)
                out[i] = static_cast<int16_t> (static_cast<int32_t> (out[i]) + in[i]);
#elif ZZS_KERNEL_SSE2 || ZZS_KERNEL_AVX2 || ZZS_KERNEL_AVX512
        uint32_t i = 0;
        for (; i + 8u <= count; i += 8u) {
                const __m128i a = _mm_loadu_si128 (reinterpret_cast<const __m128i *> (out + i));
                const __m128i b = _mm_loadu_si128 (reinterpret_cast<const __m128i *> (in + i));
                _mm_storeu_si128 (reinterpret_cast<__m128i *> (out + i), _mm_add_epi16 (a, b));
        }
        for (; i < count; ++i)
                out[i] = static_cast<int16_t> (static_cast<int32_t> (out[i]) + in[i]);
#else
        for (uint32_t i = 0; i < count; ++i)
                out[i] = static_cast<int16_t> (static_cast<int32_t> (out[i]) + in[i]);
#endif
}

/** @} */
} // namespace zzs::kernel::simd
