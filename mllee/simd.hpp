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

#include "mllee/core_status.hpp"
#include "mllee/quant.hpp"

#include <cstdint>

/*
 * SIMD kernels are portable C++ implementations.  Every primitive below has a
 * portable C++ fallback so the headers remain usable on any host and any
 * compiler.  SIMD intrinsics were previously provided via hand-written
 * assembly; those assembly sources have been removed and the primitives are
 * now plain portable C++.
 */
namespace zzs::mllee::simd {
/** @brief Signed int8 dot product, returned as int32 (the core's accumulator width). */
[[nodiscard]] inline int32_t
dot_i8 (const int8_t *const ZZS_MLLEE_RESTRICT lhs, const int8_t *const ZZS_MLLEE_RESTRICT rhs,
        const uint32_t count) noexcept {
    int32_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<int32_t> (lhs[i]) * static_cast<int32_t> (rhs[i]);
    return total;
}

/** @brief Dot product over packed int4/float4 codes (two codes per byte). */
[[nodiscard]] inline int32_t
dot_i4 (const uint8_t *const ZZS_MLLEE_RESTRICT packed, const int8_t *const ZZS_MLLEE_RESTRICT rhs,
        const uint32_t codes, const int8_t *const ZZS_MLLEE_RESTRICT lut16) noexcept {
    int32_t total = 0;
    for (uint32_t i = 0; i < codes; ++i) {
        const uint32_t nibble = (i & 1u) != 0u ? (packed[i >> 1] >> 4) : (packed[i >> 1] & 0x0Fu);
        const int32_t weight = lut16 != nullptr ? lut16[nibble] : _S_from_int4 (static_cast<uint8_t> (nibble));
        total += weight * static_cast<int32_t> (rhs[i]);
    }
    return total;
}

/** @brief Dot product over packed int2 codes (four codes per byte). */
[[nodiscard]] inline int32_t
dot_i2 (const uint8_t *const ZZS_MLLEE_RESTRICT packed, const int8_t *const ZZS_MLLEE_RESTRICT rhs,
        const uint32_t codes, const int8_t *const ZZS_MLLEE_RESTRICT lut4) noexcept {
    int32_t total = 0;
    for (uint32_t i = 0; i < codes; ++i) {
        const uint8_t byte = packed[i >> 2];
        const uint32_t code = (byte >> ((i & 3u) * 2u)) & 0x03u;
        const int32_t weight = lut4 != nullptr ? lut4[code] : _S_from_int2 (static_cast<uint8_t> (code));
        total += weight * static_cast<int32_t> (rhs[i]);
    }
    return total;
}

/** @brief Dot product over bit-packed 3-bit codes (eight codes in three bytes). */
[[nodiscard]] inline int32_t
dot_i3 (const uint8_t *const ZZS_MLLEE_RESTRICT packed, const int8_t *const ZZS_MLLEE_RESTRICT rhs,
        const uint32_t codes, const int8_t *const ZZS_MLLEE_RESTRICT lut8) noexcept {
    int32_t total = 0;
    uint32_t i = 0;
    for (; i + 8u <= codes; i += 8u) {
        const uint8_t *const group = packed + (i * 3u >> 3);
        const uint32_t word = static_cast<uint32_t> (group[0]) | (static_cast<uint32_t> (group[1]) << 8)
                              | (static_cast<uint32_t> (group[2]) << 16);
        const int8_t *const r = rhs + i;
        total += (lut8 != nullptr ? lut8[word & 7u] : _S_from_int3 (static_cast<uint8_t> (word & 7u)))
                 * static_cast<int32_t> (r[0]);
        total += (lut8 != nullptr ? lut8[word >> 3 & 7u] : _S_from_int3 (static_cast<uint8_t> (word >> 3 & 7u)))
                 * static_cast<int32_t> (r[1]);
        total += (lut8 != nullptr ? lut8[word >> 6 & 7u] : _S_from_int3 (static_cast<uint8_t> (word >> 6 & 7u)))
                 * static_cast<int32_t> (r[2]);
        total += (lut8 != nullptr ? lut8[word >> 9 & 7u] : _S_from_int3 (static_cast<uint8_t> (word >> 9 & 7u)))
                 * static_cast<int32_t> (r[3]);
        total += (lut8 != nullptr ? lut8[word >> 12 & 7u] : _S_from_int3 (static_cast<uint8_t> (word >> 12 & 7u)))
                 * static_cast<int32_t> (r[4]);
        total += (lut8 != nullptr ? lut8[word >> 15 & 7u] : _S_from_int3 (static_cast<uint8_t> (word >> 15 & 7u)))
                 * static_cast<int32_t> (r[5]);
        total += (lut8 != nullptr ? lut8[word >> 18 & 7u] : _S_from_int3 (static_cast<uint8_t> (word >> 18 & 7u)))
                 * static_cast<int32_t> (r[6]);
        total += (lut8 != nullptr ? lut8[word >> 21 & 7u] : _S_from_int3 (static_cast<uint8_t> (word >> 21 & 7u)))
                 * static_cast<int32_t> (r[7]);
    }
    if (i < codes) {
        const uint8_t *const group = packed + (i * 3u >> 3);
        const uint32_t remaining_bits = (codes - i) * 3u;
        const uint32_t bytes = (remaining_bits + 7u) >> 3;
        uint32_t word = group[0];
        if (bytes > 1u)
            word |= static_cast<uint32_t> (group[1]) << 8;
        if (bytes > 2u)
            word |= static_cast<uint32_t> (group[2]) << 16;
        for (uint32_t j = 0; i < codes; ++i, ++j) {
            const uint32_t field = (word >> (j * 3u)) & 7u;
            const int32_t weight = lut8 != nullptr ? lut8[field] : _S_from_int3 (static_cast<uint8_t> (field));
            total += weight * static_cast<int32_t> (rhs[i]);
        }
    }
    return total;
}

/** @brief Clamp every byte of `src` to `[low, high]` in one pass. */
inline void
clamp_i8 (const int8_t *const ZZS_MLLEE_RESTRICT src, int8_t *const ZZS_MLLEE_RESTRICT dst, const uint32_t count,
          const int8_t low, const int8_t high) noexcept {
    for (uint32_t i = 0; i < count; ++i)
        dst[i] = src[i] < low ? low : (src[i] > high ? high : src[i]);
}

/** @brief Select `fill` where `mask` is non-zero, otherwise keep `src`. */
inline void
masked_fill_i8 (const int8_t *const ZZS_MLLEE_RESTRICT src, const int8_t *const ZZS_MLLEE_RESTRICT mask,
                int8_t *const ZZS_MLLEE_RESTRICT dst, const uint32_t count, const int8_t fill) noexcept {
    for (uint32_t i = 0; i < count; ++i)
        dst[i] = mask[i] != 0 ? fill : src[i];
}

/** @brief Sum of squared int8 codes; the L2-norm path never needs the mean. */
[[nodiscard]] inline int32_t
sum_sq_i8 (const int8_t *const ZZS_MLLEE_RESTRICT values, const uint32_t count) noexcept {
    int32_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<int32_t> (values[i]) * values[i];
    return total;
}

/** @brief Q15 requantization with round-half-up, exact for every int32 accumulator. */
[[nodiscard]] inline int32_t
requant_q15 (const int64_t accumulator, const int32_t multiplier) noexcept {
    return _S_requant_q15 (accumulator, multiplier);
}

/** @brief Expand packed nibble codes through a 16-entry table, low nibble first. */
inline void
expand_codes (const uint8_t *const ZZS_MLLEE_RESTRICT source, int8_t *const ZZS_MLLEE_RESTRICT destination,
              const uint32_t bytes, const int8_t *const lut16) noexcept {
    for (uint32_t i = 0; i < bytes; ++i) {
        const uint8_t low = source[i] & 0x0Fu;
        const uint8_t high = static_cast<uint8_t> ((source[i] >> 4) & 0x0Fu);
        destination[i * 2u] = static_cast<int8_t> (lut16 != nullptr ? lut16[low] : _S_from_int4 (low));
        destination[i * 2u + 1u] = static_cast<int8_t> (lut16 != nullptr ? lut16[high] : _S_from_int4 (high));
    }
}

/** @brief Expand packed int4 (two's complement, low nibble first) into int8 codes. */
inline void
unpack_i4 (const uint8_t *const ZZS_MLLEE_RESTRICT source, int8_t *const ZZS_MLLEE_RESTRICT destination,
           const uint32_t bytes) noexcept {
    expand_codes (source, destination, bytes, nullptr);
}

/**
 * @brief Expand 2-bit codes (four per byte, low pair first) into int8 codes.
 */
inline void
expand_codes2 (const uint8_t *const ZZS_MLLEE_RESTRICT source, int8_t *const ZZS_MLLEE_RESTRICT destination,
               const uint32_t bytes, const int8_t *const lut4) noexcept {
    for (uint32_t i = 0; i < bytes; ++i) {
        const uint32_t byte = source[i];
        destination[i * 4u] =
            static_cast<int8_t> (lut4 != nullptr ? lut4[byte & 3u] : _S_from_int2 (static_cast<uint8_t> (byte & 3u)));
        destination[i * 4u + 1u] = static_cast<int8_t> (
            lut4 != nullptr ? lut4[byte >> 2 & 3u] : _S_from_int2 (static_cast<uint8_t> (byte >> 2 & 3u)));
        destination[i * 4u + 2u] = static_cast<int8_t> (
            lut4 != nullptr ? lut4[byte >> 4 & 3u] : _S_from_int2 (static_cast<uint8_t> (byte >> 4 & 3u)));
        destination[i * 4u + 3u] = static_cast<int8_t> (
            lut4 != nullptr ? lut4[byte >> 6 & 3u] : _S_from_int2 (static_cast<uint8_t> (byte >> 6 & 3u)));
    }
}

/**
 * @brief Expand bit-packed 3-bit codes (eight per three bytes) into int8 codes.
 */
inline void
expand_codes3 (const uint8_t *const ZZS_MLLEE_RESTRICT source, int8_t *const ZZS_MLLEE_RESTRICT destination,
               const uint32_t codes, const int8_t *const lut8) noexcept {
    uint32_t i = 0;
    for (; i + 8u <= codes; i += 8u) {
        const uint8_t *const group = source + (i * 3u >> 3);
        const uint32_t word = static_cast<uint32_t> (group[0]) | (static_cast<uint32_t> (group[1]) << 8)
                              | (static_cast<uint32_t> (group[2]) << 16);
        for (uint32_t j = 0; j < 8u; ++j) {
            const uint32_t field = (word >> (j * 3u)) & 7u;
            destination[i + j] =
                static_cast<int8_t> (lut8 != nullptr ? lut8[field] : _S_from_int3 (static_cast<uint8_t> (field)));
        }
    }
    if (i < codes) {
        const uint8_t *const group = source + (i * 3u >> 3);
        const uint32_t remaining_bits = (codes - i) * 3u;
        const uint32_t bytes = (remaining_bits + 7u) >> 3;
        uint32_t word = group[0];
        if (bytes > 1u)
            word |= static_cast<uint32_t> (group[1]) << 8;
        if (bytes > 2u)
            word |= static_cast<uint32_t> (group[2]) << 16;
        for (uint32_t j = 0; i < codes; ++i, ++j) {
            const uint32_t field = (word >> (j * 3u)) & 7u;
            destination[i] =
                static_cast<int8_t> (lut8 != nullptr ? lut8[field] : _S_from_int3 (static_cast<uint8_t> (field)));
        }
    }
}

/** @brief Map byte-wide codes through a 256-entry flash table. */
inline void
gather_lut8 (const int8_t *const ZZS_MLLEE_RESTRICT lut256, const uint8_t *const ZZS_MLLEE_RESTRICT codes,
             int8_t *const ZZS_MLLEE_RESTRICT destination, const uint32_t count) noexcept {
    for (uint32_t i = 0; i < count; ++i)
        destination[i] = lut256[codes[i]];
}

/** @brief Saturating int8 addition, clamped to the core's [-127, 127] code range. */
inline void
add_sat_i8 (const int8_t *const ZZS_MLLEE_RESTRICT lhs, const int8_t *const ZZS_MLLEE_RESTRICT rhs,
            int8_t *const ZZS_MLLEE_RESTRICT destination, const uint32_t count) noexcept {
    for (uint32_t i = 0; i < count; ++i)
        destination[i] = _S_i8_code (static_cast<int32_t> (lhs[i]) + rhs[i]);
}

/** @brief Saturating int8 subtraction, clamped to the core's [-127, 127] code range. */
inline void
sub_sat_i8 (const int8_t *const ZZS_MLLEE_RESTRICT lhs, const int8_t *const ZZS_MLLEE_RESTRICT rhs,
            int8_t *const ZZS_MLLEE_RESTRICT destination, const uint32_t count) noexcept {
    for (uint32_t i = 0; i < count; ++i)
        destination[i] = _S_i8_code (static_cast<int32_t> (lhs[i]) - rhs[i]);
}

/**
 * @brief Window statistics: the sum and the sum of squares of `count` codes.
 */
inline void
stats_i8 (const int8_t *const ZZS_MLLEE_RESTRICT values, const uint32_t count, int32_t &sum,
          int32_t &square_sum) noexcept {
    sum = 0;
    square_sum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        const int32_t value = values[i];
        sum += value;
        square_sum += value * value;
    }
}

/** @brief Sum of int8 codes. */
[[nodiscard]] inline int32_t
sum_i8 (const int8_t *const ZZS_MLLEE_RESTRICT values, const uint32_t count) noexcept {
    int32_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += values[i];
    return total;
}

/** @brief Maximum of int8 codes. */
[[nodiscard]] inline int8_t
max_i8 (const int8_t *const ZZS_MLLEE_RESTRICT values, const uint32_t count) noexcept {
    int8_t maximum = -128;
    for (uint32_t i = 0; i < count; ++i)
        if (values[i] > maximum)
            maximum = values[i];
    return maximum;
}

#ifdef ZZS_MLLEE_HOST
/** @brief Bulk PyTorch-style affine quantization (host calibration path). */
inline void
quantize_affine_i8 (const float *const ZZS_MLLEE_RESTRICT source, int8_t *const ZZS_MLLEE_RESTRICT destination,
                    const uint32_t count, const float scale, const int32_t zero_point) noexcept {
    for (uint32_t i = 0; i < count; ++i)
        destination[i] = static_cast<int8_t> (zzs::mllee::quantize_affine_i8 (source[i], scale, zero_point));
}

/** @brief Bulk dequantization (host inspection/debug path). */
inline void
dequantize_i8 (const int8_t *const ZZS_MLLEE_RESTRICT source, float *const ZZS_MLLEE_RESTRICT destination,
               const uint32_t count, const float scale) noexcept {
    for (uint32_t i = 0; i < count; ++i)
        destination[i] = zzs::mllee::dequantize_i8 (source[i], scale);
}
#endif // ZZS_MLLEE_HOST
} // namespace zzs::mllee::simd
