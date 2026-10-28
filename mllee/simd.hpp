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
    int32_t zzs_mllee_dot_i4_avx2 (const uint8_t *packed, const int8_t *rhs,
                                        uint32_t codes, const int8_t *lut16);
    int32_t zzs_mllee_dot_i2_avx2 (const uint8_t *packed, const int8_t *rhs,
                                        uint32_t codes, const int8_t *lut4);
    void zzs_mllee_clamp_i8 (const int8_t *src, int8_t *dst, uint32_t count,
                             int8_t low, int8_t high);
    void zzs_mllee_masked_fill_i8 (const int8_t *src, const int8_t *mask,
                                   int8_t *dst, uint32_t count, int8_t fill);
    int32_t zzs_mllee_sum_sq_i8 (const int8_t *values, uint32_t count);
#endif
#if defined(ZZS_MLLEE_ASM_AVX_VNNI)
    int32_t zzs_mllee_dot_i8_avx_vnni (const int8_t *lhs, const int8_t *rhs,
                                            uint32_t count);
#endif
#if defined(ZZS_MLLEE_ASM_AVX512)
    int32_t zzs_mllee_dot_i8_avx512 (const int8_t *lhs, const int8_t *rhs,
                                          uint32_t count);
#endif
#endif
}
#endif

namespace zzs::mllee::simd
{
/**
 * @brief Signed int8 dot product, returned as int32 (the core's accumulator
 *        width).
 *
 * Kernel precedence is VPDPBUSD > AVX2 widening > SSE2 > portable, all decided
 * at build time.  The two vector forms differ in their exactness window, which
 * is why the widening form is not simply the slower option: `vpdpbusd` folds a
 * biased accumulator (see mllee/asm-x86_64/mllee_int8_dot_avx_vnni.S) that
 * stays inside int32 up to 65793 codes, while the sign-extending forms are
 * exact up to 133143.  Every row the blob format can describe is far shorter
 * than both -- dimensions are uint16 and the staged conv window is 64 codes --
 * so either kernel is exact for every model the reader accepts.
 */
[[nodiscard]] inline int32_t
dot_i8 (const int8_t *const ZZS_MLLEE_RESTRICT lhs,
        const int8_t *const ZZS_MLLEE_RESTRICT rhs,
        const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_ASM_AVX512)
    return zzs_mllee_dot_i8_avx512 (lhs, rhs, count);
#elif defined(ZZS_MLLEE_ASM_AVX_VNNI)
    return zzs_mllee_dot_i8_avx_vnni (lhs, rhs, count);
#elif defined(ZZS_MLLEE_ASM_AVX2)
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

/**
 * @brief Dot product over packed int4/float4 codes (two codes per byte).
 *
 * `lut16` is the 16-entry float4 or caller table; null selects the
 * two's-complement int4 mapping.  The AVX2 kernel expands the nibbles with
 * pshufb and multiplies 32 codes per iteration without a scratch buffer.  The
 * scalar fallback is the same low-nibble-first layout, so every build produces
 * identical accumulators.
 */
[[nodiscard]] inline int32_t
dot_i4 (const uint8_t *const ZZS_MLLEE_RESTRICT packed,
        const int8_t *const ZZS_MLLEE_RESTRICT rhs, const uint32_t codes,
        const int8_t *const ZZS_MLLEE_RESTRICT lut16) noexcept
{
#if defined(ZZS_MLLEE_ASM_AVX2)
    return zzs_mllee_dot_i4_avx2 (packed, rhs, codes, lut16);
#else
    int32_t total = 0;
    for (uint32_t i = 0; i < codes; ++i)
        {
            const uint32_t nibble
                = (i & 1u) != 0u ? (packed[i >> 1] >> 4) : (packed[i >> 1] & 0x0Fu);
            const int32_t weight = lut16 != nullptr ? lut16[nibble]
                                                    : _S_from_int4 (static_cast<uint8_t> (nibble));
            total += weight * static_cast<int32_t> (rhs[i]);
        }
    return total;
#endif
}

/** @brief Dot product over packed int2 codes (four codes per byte). */
[[nodiscard]] inline int32_t
dot_i2 (const uint8_t *const ZZS_MLLEE_RESTRICT packed,
        const int8_t *const ZZS_MLLEE_RESTRICT rhs, const uint32_t codes,
        const int8_t *const ZZS_MLLEE_RESTRICT lut4) noexcept
{
#if defined(ZZS_MLLEE_ASM_AVX2)
    return zzs_mllee_dot_i2_avx2 (packed, rhs, codes, lut4);
#else
    int32_t total = 0;
    for (uint32_t i = 0; i < codes; ++i)
        {
            const uint8_t byte = packed[i >> 2];
            const uint32_t code = (byte >> ((i & 3u) * 2u)) & 0x03u;
            const int32_t weight = lut4 != nullptr ? lut4[code]
                                                   : _S_from_int2 (static_cast<uint8_t> (code));
            total += weight * static_cast<int32_t> (rhs[i]);
        }
    return total;
#endif
}

/**
 * @brief Dot product over bit-packed 3-bit codes (eight codes in three bytes).
 *
 * The 3-bit grid does not map onto a byte shuffle, so this path is a load and
 * eight shift-masks per 24-bit group rather than a `pshufb`.  Every group is
 * one unaligned 32-bit load (only the first three bytes are used), which is one
 * memory access per eight codes instead of one per code, and the loop body is
 * fully unrolled so no shift count is materialized in a register.  `lut8` is
 * the F3E1M1 table; null selects the two's-complement I3 mapping.
 */
[[nodiscard]] inline int32_t
dot_i3 (const uint8_t *const ZZS_MLLEE_RESTRICT packed,
        const int8_t *const ZZS_MLLEE_RESTRICT rhs, const uint32_t codes,
        const int8_t *const ZZS_MLLEE_RESTRICT lut8) noexcept
{
    int32_t total = 0;
    uint32_t i = 0;
    for (; i + 8u <= codes; i += 8u)
        {
            const uint8_t *const group = packed + (i * 3u >> 3);
            const uint32_t word = static_cast<uint32_t> (group[0])
                                  | (static_cast<uint32_t> (group[1]) << 8)
                                  | (static_cast<uint32_t> (group[2]) << 16);
            const int8_t *const r = rhs + i;
            total += (lut8 != nullptr ? lut8[word & 7u]
                                      : _S_from_int3 (static_cast<uint8_t> (word & 7u)))
                     * static_cast<int32_t> (r[0]);
            total += (lut8 != nullptr ? lut8[word >> 3 & 7u]
                                      : _S_from_int3 (static_cast<uint8_t> (word >> 3 & 7u)))
                     * static_cast<int32_t> (r[1]);
            total += (lut8 != nullptr ? lut8[word >> 6 & 7u]
                                      : _S_from_int3 (static_cast<uint8_t> (word >> 6 & 7u)))
                     * static_cast<int32_t> (r[2]);
            total += (lut8 != nullptr ? lut8[word >> 9 & 7u]
                                      : _S_from_int3 (static_cast<uint8_t> (word >> 9 & 7u)))
                     * static_cast<int32_t> (r[3]);
            total += (lut8 != nullptr ? lut8[word >> 12 & 7u]
                                      : _S_from_int3 (static_cast<uint8_t> (word >> 12 & 7u)))
                     * static_cast<int32_t> (r[4]);
            total += (lut8 != nullptr ? lut8[word >> 15 & 7u]
                                      : _S_from_int3 (static_cast<uint8_t> (word >> 15 & 7u)))
                     * static_cast<int32_t> (r[5]);
            total += (lut8 != nullptr ? lut8[word >> 18 & 7u]
                                      : _S_from_int3 (static_cast<uint8_t> (word >> 18 & 7u)))
                     * static_cast<int32_t> (r[6]);
            total += (lut8 != nullptr ? lut8[word >> 21 & 7u]
                                      : _S_from_int3 (static_cast<uint8_t> (word >> 21 & 7u)))
                     * static_cast<int32_t> (r[7]);
        }
    if (i < codes)
        {
            const uint8_t *const group = packed + (i * 3u >> 3);
            const uint32_t remaining_bits = (codes - i) * 3u;
            const uint32_t bytes = (remaining_bits + 7u) >> 3;
            uint32_t word = group[0];
            if (bytes > 1u)
                word |= static_cast<uint32_t> (group[1]) << 8;
            if (bytes > 2u)
                word |= static_cast<uint32_t> (group[2]) << 16;
            for (uint32_t j = 0; i < codes; ++i, ++j)
                {
                    const uint32_t field = (word >> (j * 3u)) & 7u;
                    const int32_t weight
                        = lut8 != nullptr ? lut8[field]
                                          : _S_from_int3 (static_cast<uint8_t> (field));
                    total += weight * static_cast<int32_t> (rhs[i]);
                }
        }
    return total;
}

/** @brief Clamp every byte of `src` to `[low, high]` in one pass. */
inline void
clamp_i8 (const int8_t *const ZZS_MLLEE_RESTRICT src,
          int8_t *const ZZS_MLLEE_RESTRICT dst, const uint32_t count, const int8_t low,
          const int8_t high) noexcept
{
#if defined(ZZS_MLLEE_ASM_AVX2)
    zzs_mllee_clamp_i8 (src, dst, count, low, high);
#else
    for (uint32_t i = 0; i < count; ++i)
        dst[i] = src[i] < low ? low : (src[i] > high ? high : src[i]);
#endif
}

/** @brief Select `fill` where `mask` is non-zero, otherwise keep `src`. */
inline void
masked_fill_i8 (const int8_t *const ZZS_MLLEE_RESTRICT src,
                const int8_t *const ZZS_MLLEE_RESTRICT mask,
                int8_t *const ZZS_MLLEE_RESTRICT dst, const uint32_t count,
                const int8_t fill) noexcept
{
#if defined(ZZS_MLLEE_ASM_AVX2)
    zzs_mllee_masked_fill_i8 (src, mask, dst, count, fill);
#else
    for (uint32_t i = 0; i < count; ++i)
        dst[i] = mask[i] != 0 ? fill : src[i];
#endif
}

/** @brief Sum of squared int8 codes; the L2-norm path never needs the mean. */
[[nodiscard]] inline int32_t
sum_sq_i8 (const int8_t *const ZZS_MLLEE_RESTRICT values, const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_ASM_AVX2)
    return zzs_mllee_sum_sq_i8 (values, count);
#else
    int32_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<int32_t> (values[i]) * values[i];
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
expand_codes (const uint8_t *const ZZS_MLLEE_RESTRICT source,
             int8_t *const ZZS_MLLEE_RESTRICT destination,
              const uint32_t bytes, const int8_t *const lut16) noexcept
{
#if defined(ZZS_MLLEE_ASM_SSSE3)
    zzs_mllee_expand_codes_ssse3 (source, destination, bytes, lut16);
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
unpack_i4 (const uint8_t *const ZZS_MLLEE_RESTRICT source,
            int8_t *const ZZS_MLLEE_RESTRICT destination,
           const uint32_t bytes) noexcept
{
#if defined(ZZS_MLLEE_HAS_ASM) && !defined(ZZS_MLLEE_ASM_SSSE3)
    zzs_mllee_unpack_i4 (source, destination, bytes);
#else
    expand_codes (source, destination, bytes, nullptr);
#endif
}

/**
 * @brief Expand 2-bit codes (four per byte, low pair first) into int8 codes.
 *
 * `lut4` is the format's four-entry table when the dtype is a 2-bit float (the
 * module ships none today, so this is reserved for a caller-provided table);
 * null selects the two's-complement int2 grid {-2, -1, 0, 1}.
 */
inline void
expand_codes2 (const uint8_t *const ZZS_MLLEE_RESTRICT source,
               int8_t *const ZZS_MLLEE_RESTRICT destination,
               const uint32_t bytes, const int8_t *const lut4) noexcept
{
    for (uint32_t i = 0; i < bytes; ++i)
        {
            const uint32_t byte = source[i];
            destination[i * 4u] = static_cast<int8_t> (
                lut4 != nullptr ? lut4[byte & 3u] : _S_from_int2 (static_cast<uint8_t> (byte & 3u)));
            destination[i * 4u + 1u] = static_cast<int8_t> (
                lut4 != nullptr ? lut4[byte >> 2 & 3u]
                                : _S_from_int2 (static_cast<uint8_t> (byte >> 2 & 3u)));
            destination[i * 4u + 2u] = static_cast<int8_t> (
                lut4 != nullptr ? lut4[byte >> 4 & 3u]
                                : _S_from_int2 (static_cast<uint8_t> (byte >> 4 & 3u)));
            destination[i * 4u + 3u] = static_cast<int8_t> (
                lut4 != nullptr ? lut4[byte >> 6 & 3u]
                                : _S_from_int2 (static_cast<uint8_t> (byte >> 6 & 3u)));
        }
}

/**
 * @brief Expand bit-packed 3-bit codes (eight per three bytes) into int8 codes.
 *
 * `codes` is rounded down to whole groups by the caller (a 3-bit row is always
 * a multiple of eight codes); the unaligned 32-bit load consumes exactly three
 * bytes per group.  `lut8` is the F3E1M1 table, null the two's-complement I3
 * grid.
 */
inline void
expand_codes3 (const uint8_t *const ZZS_MLLEE_RESTRICT source,
               int8_t *const ZZS_MLLEE_RESTRICT destination,
               const uint32_t codes, const int8_t *const lut8) noexcept
{
    uint32_t i = 0;
    for (; i + 8u <= codes; i += 8u)
        {
            const uint8_t *const group = source + (i * 3u >> 3);
            const uint32_t word = static_cast<uint32_t> (group[0])
                                  | (static_cast<uint32_t> (group[1]) << 8)
                                  | (static_cast<uint32_t> (group[2]) << 16);
            for (uint32_t j = 0; j < 8u; ++j)
                {
                    const uint32_t field = (word >> (j * 3u)) & 7u;
                    destination[i + j] = static_cast<int8_t> (
                        lut8 != nullptr ? lut8[field]
                                        : _S_from_int3 (static_cast<uint8_t> (field)));
                }
        }
    if (i < codes)
        {
            const uint8_t *const group = source + (i * 3u >> 3);
            const uint32_t remaining_bits = (codes - i) * 3u;
            const uint32_t bytes = (remaining_bits + 7u) >> 3;
            uint32_t word = group[0];
            if (bytes > 1u)
                word |= static_cast<uint32_t> (group[1]) << 8;
            if (bytes > 2u)
                word |= static_cast<uint32_t> (group[2]) << 16;
            for (uint32_t j = 0; i < codes; ++i, ++j)
                {
                    const uint32_t field = (word >> (j * 3u)) & 7u;
                    destination[i] = static_cast<int8_t> (
                        lut8 != nullptr ? lut8[field]
                                        : _S_from_int3 (static_cast<uint8_t> (field)));
                }
        }
}

/** @brief Map byte-wide codes through a 256-entry flash table. */
inline void
gather_lut8 (const int8_t *const ZZS_MLLEE_RESTRICT lut256,
             const uint8_t *const ZZS_MLLEE_RESTRICT codes,
             int8_t *const ZZS_MLLEE_RESTRICT destination, const uint32_t count) noexcept
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
add_sat_i8 (const int8_t *const ZZS_MLLEE_RESTRICT lhs,
            const int8_t *const ZZS_MLLEE_RESTRICT rhs,
            int8_t *const ZZS_MLLEE_RESTRICT destination, const uint32_t count) noexcept
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
sub_sat_i8 (const int8_t *const ZZS_MLLEE_RESTRICT lhs,
            const int8_t *const ZZS_MLLEE_RESTRICT rhs,
            int8_t *const ZZS_MLLEE_RESTRICT destination, const uint32_t count) noexcept
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
stats_i8 (const int8_t *const ZZS_MLLEE_RESTRICT values, const uint32_t count,
          int32_t &sum, int32_t &square_sum) noexcept
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
sum_i8 (const int8_t *const ZZS_MLLEE_RESTRICT values, const uint32_t count) noexcept
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
max_i8 (const int8_t *const ZZS_MLLEE_RESTRICT values, const uint32_t count) noexcept
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
quantize_affine_i8 (const float *const ZZS_MLLEE_RESTRICT source,
                    int8_t *const ZZS_MLLEE_RESTRICT destination,
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
dequantize_i8 (const int8_t *const ZZS_MLLEE_RESTRICT source,
               float *const ZZS_MLLEE_RESTRICT destination,
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
