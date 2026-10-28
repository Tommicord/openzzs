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

#include <cstdint>

/*
 * The assembly kernels are selected at build time, never at run time.  That
 * keeps the hot path free of CPUID probing, branch-on-CPU and hidden module
 * state, which matters for the instruction-count predictability the embedded
 * target needs.  Without a target define every primitive below is a portable
 * C++ fallback so the headers remain usable on any host.
 */
#if defined(ZZS_MLLEE_ASM_X86_64) || defined(ZZS_MLLEE_ASM_CORTEX_M4)
#define ZZS_MLLEE_HAS_ASM 1
extern "C"
{
    int32_t zzs_mllee_dot_i8 (const int8_t *lhs, const int8_t *rhs,
                                   uint32_t count);
    int32_t zzs_mllee_requant_q15 (int32_t accumulator, int32_t multiplier);
    void zzs_mllee_unpack_i4 (const uint8_t *source, int8_t *destination,
                              uint32_t bytes);
    void zzs_mllee_add_sat_i8 (const int8_t *lhs, const int8_t *rhs,
                               int8_t *destination, uint32_t count);
    void zzs_mllee_sub_sat_i8 (const int8_t *lhs, const int8_t *rhs,
                               int8_t *destination, uint32_t count);
    int32_t zzs_mllee_sum_i8 (const int8_t *values, uint32_t count);
    int8_t zzs_mllee_max_i8 (const int8_t *values, uint32_t count);
    void zzs_mllee_stats_i8 (const int8_t *values, uint32_t count, int32_t *sum,
                             int32_t *square_sum);

#if defined(ZZS_MLLEE_ASM_X86_64)
    void zzs_mllee_quantize_affine_i8 (const float *source, int8_t *destination,
                                       uint32_t count, float scale,
                                       int32_t zero_point);
    void zzs_mllee_dequantize_i8 (const int8_t *source, float *destination,
                                  uint32_t count, float scale);
#if defined(ZZS_MLLEE_ASM_SSSE3)
    void zzs_mllee_expand_codes_ssse3 (const uint8_t *source, int8_t *destination,
                                       uint32_t bytes, const int8_t *lut16);
    void zzs_mllee_gather_lut8 (const int8_t *lut256, const uint8_t *codes,
                                int8_t *destination, uint32_t count);
#endif
#if defined(ZZS_MLLEE_ASM_AVX2)
    int32_t zzs_mllee_dot_i8_avx2 (const int8_t *lhs, const int8_t *rhs,
                                        uint32_t count);
#endif
#endif
}
#endif

namespace zzs::mllee::simd
{
/** @brief Signed int8 dot product, returned as int32 (the core's accumulator width). */
[[nodiscard]] inline int32_t
dot_i8 (const int8_t *const lhs, const int8_t *const rhs,
        const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_ASM_AVX2)
    return zzs_mllee_dot_i8_avx2 (lhs, rhs, count);
#elif defined(ZZS_MLLEE_HAS_ASM)
    return zzs_mllee_dot_i8 (lhs, rhs, count);
#else
    int32_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<int32_t> (lhs[i]) * static_cast<int32_t> (rhs[i]);
    return total;
#endif
}

/** @brief Q15 requantization with round-half-up, exact for every int32 accumulator. */
[[nodiscard]] inline int32_t
requant_q15 (const int64_t accumulator, const int32_t multiplier) noexcept
{
#if defined(ZZS_MLLEE_HAS_ASM)
    if (accumulator >= -2147483648LL && accumulator <= 2147483647LL)
        return zzs_mllee_requant_q15 (static_cast<int32_t> (accumulator), multiplier);
#endif
    return _S_requant_q15 (accumulator, multiplier);
}

/**
 * @brief Expand packed nibble codes through a 16-entry table, low nibble first.
 *
 * This is the int4/float4 weight-row expansion: `lut16` is the model's own
 * flash table for a float4 layer, null for the two's-complement int4 mapping.
 * The SSSE3 kernel spends one pshufb per nibble group -- the shuffle register
 * doubles as the table -- instead of a shift, a mask and a sign test per code,
 * and it keeps its own int4 table when handed a null one.
 */
inline void
expand_codes (const uint8_t *const source, int8_t *const destination,
              const uint32_t bytes, const int8_t *const lut16) noexcept
{
#if defined(ZZS_MLLEE_ASM_SSSE3)
    if (lut16 != nullptr)
        {
            int8_t table[16];
            for (uint32_t i = 0; i < 16u; ++i)
                table[i] = lut16[i];
            zzs_mllee_expand_codes_ssse3 (source, destination, bytes, table);
            return;
        }
    zzs_mllee_expand_codes_ssse3 (source, destination, bytes, nullptr);
#else
    for (uint32_t i = 0; i < bytes; ++i)
        {
            const uint8_t low = source[i] & 0x0Fu;
            const uint8_t high = static_cast<uint8_t> ((source[i] >> 4) & 0x0Fu);
            destination[i * 2u] =
                static_cast<int8_t> (lut16 != nullptr ? lut16[low] : _S_from_int4 (low));
            destination[i * 2u + 1u] =
                static_cast<int8_t> (lut16 != nullptr ? lut16[high] : _S_from_int4 (high));
        }
#endif
}

/** @brief Expand packed int4 (two's complement, low nibble first) into int8 codes. */
inline void
unpack_i4 (const uint8_t *const source, int8_t *const destination,
           const uint32_t bytes) noexcept
{
#if defined(ZZS_MLLEE_HAS_ASM) && !defined(ZZS_MLLEE_ASM_SSSE3)
    zzs_mllee_unpack_i4 (source, destination, bytes);
#else
    expand_codes (source, destination, bytes, nullptr);
#endif
}

/** @brief Map byte-wide codes through a 256-entry flash table. */
inline void
gather_lut8 (const int8_t *const lut256, const uint8_t *const codes,
             int8_t *const destination, const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_ASM_SSSE3)
    zzs_mllee_gather_lut8 (lut256, codes, destination, count);
#else
    for (uint32_t i = 0; i < count; ++i)
        destination[i] = lut256[codes[i]];
#endif
}

/** @brief Saturating int8 addition, clamped to the core's [-127, 127] code range. */
inline void
add_sat_i8 (const int8_t *const lhs, const int8_t *const rhs,
            int8_t *const destination, const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_ASM)
    zzs_mllee_add_sat_i8 (lhs, rhs, destination, count);
#else
    for (uint32_t i = 0; i < count; ++i)
        destination[i] = _S_i8_code (static_cast<int32_t> (lhs[i]) + rhs[i]);
#endif
}

/** @brief Saturating int8 subtraction, clamped to the core's [-127, 127] code range. */
inline void
sub_sat_i8 (const int8_t *const lhs, const int8_t *const rhs,
            int8_t *const destination, const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_ASM)
    zzs_mllee_sub_sat_i8 (lhs, rhs, destination, count);
#else
    for (uint32_t i = 0; i < count; ++i)
        destination[i] = _S_i8_code (static_cast<int32_t> (lhs[i]) - rhs[i]);
#endif
}

/**
 * @brief Window statistics: the sum and the sum of squares of `count` codes.
 *
 * Both are int32; the square term peaks at count * 16129, so any layer
 * dimension the format allows (uint16) stays inside the lane.
 */
inline void
stats_i8 (const int8_t *const values, const uint32_t count, int32_t &sum,
          int32_t &square_sum) noexcept
{
#if defined(ZZS_MLLEE_HAS_ASM)
    int32_t total = 0;
    int32_t squares = 0;
    zzs_mllee_stats_i8 (values, count, &total, &squares);
    sum = total;
    square_sum = squares;
#else
    sum = 0;
    square_sum = 0;
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t value = values[i];
            sum += value;
            square_sum += value * value;
        }
#endif
}

[[nodiscard]] inline int32_t
sum_i8 (const int8_t *const values, const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_ASM)
    return zzs_mllee_sum_i8 (values, count);
#else
    int32_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += values[i];
    return total;
#endif
}

[[nodiscard]] inline int8_t
max_i8 (const int8_t *const values, const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_ASM)
    return zzs_mllee_max_i8 (values, count);
#else
    int8_t maximum = -128;
    for (uint32_t i = 0; i < count; ++i)
        if (values[i] > maximum)
            maximum = values[i];
    return maximum;
#endif
}

#ifdef ZZS_MLLEE_HOST
/** @brief Bulk PyTorch-style affine quantization (host calibration path). */
inline void
quantize_affine_i8 (const float *const source, int8_t *const destination,
                    const uint32_t count, const float scale,
                    const int32_t zero_point) noexcept
{
#if defined(ZZS_MLLEE_ASM_X86_64)
    zzs_mllee_quantize_affine_i8 (source, destination, count, scale, zero_point);
#else
    for (uint32_t i = 0; i < count; ++i)
        destination[i] = static_cast<int8_t> (
            zzs::mllee::quantize_affine_i8 (source[i], scale, zero_point));
#endif
}

/** @brief Bulk dequantization (host inspection/debug path). */
inline void
dequantize_i8 (const int8_t *const source, float *const destination,
               const uint32_t count, const float scale) noexcept
{
#if defined(ZZS_MLLEE_ASM_X86_64)
    zzs_mllee_dequantize_i8 (source, destination, count, scale);
#else
    for (uint32_t i = 0; i < count; ++i)
        destination[i] = zzs::mllee::dequantize_i8 (source[i], scale);
#endif
}
#endif // ZZS_MLLEE_HOST
} // namespace zzs::mllee::simd
