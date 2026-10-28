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

#include <cstdint>

#ifdef ZZS_MLLEE_HOST
#include <cmath>
#include <cstring>
#include <limits>
#endif

namespace zzs::mllee
{
/**
 * @brief Quantized storage formats understood by the core.
 *
 * int8, int4, int3 and int2 are integer affine formats.  The float formats are
 * stored as their native bit patterns; the exporter resolves them into an int8
 * LUT in Flash so the runtime always multiplies int8 by int8.
 *
 * On-disk bit layouts, all little-endian and LSB-first within a byte:
 *   I8      8 bits/code, one code per byte.
 *   I4      4 bits/code, two codes per byte, low nibble first.
 *   F4E2M1  4 bits/code, same nibble order as I4 (sign in bit 3).
 *   I2      2 bits/code, four codes per byte, lowest bits first.
 *   I3      3 bits/code, bit-packed: code i occupies bits [3i, 3i+3).
 *   F3E1M1  3 bits/code, same reservoir as I3 (sign in bit 2).
 *   F8x     8 bits/code, one code per byte.
 *
 * A 3-bit row is byte-terminated only when its code count is a multiple of 8
 * (8 codes = 24 bits = 3 bytes exactly); `_S_codes_aligned` and `_S_row_bytes`
 * are the helpers every row/weight calculation must use.
 *
 * The numeric values are on-disk constants: new formats are appended so that
 * every previously exported model keeps decoding.  F2E1M1 was never shipped in
 * a blob and is replaced by F3E1M1; I3 is appended last, so the existing
 * I8..F3E1M1 values keep their numeric identity.
 */
enum class Dtype : uint8_t
{
    I8 = 0,
    I4,
    F8E4M3,
    F8E5M2,
    F4E2M1,
    I2,
    F3E1M1,
    I3,
};

/** @brief Largest legal `Dtype` value; the format validator bounds against it. */
inline constexpr uint8_t _S_dtype_max = static_cast<uint8_t> (Dtype::I3);

/** @brief Floor(log2(value)); 0 for `value <= 1`, host- and constexpr-safe. */
[[nodiscard]] constexpr uint32_t
_S_ilog2 (const uint32_t value) noexcept
{
#if defined(__has_builtin) && __has_builtin(__builtin_clz)
    if !consteval {
        if (value != 0)
            return 31u - static_cast<uint32_t> (__builtin_clz (value));
    }
#endif
    uint32_t bits = 0;
    while ((value >> bits) > 1u)
        ++bits;
    return bits;
}

/**
 * @brief Bits per packed code: 8, 4, 3 or 2.
 *
 * This is the one number the packing model is built from: codes per byte (where
 * that is an integer), the code mask and therefore every packed-row computation
 * in the reader, the ops and the format validator derive from it.
 */
[[nodiscard]] inline constexpr uint32_t
_S_code_bits (const Dtype dtype) noexcept
{
    switch (dtype)
        {
        case Dtype::I2:
            return 2u;
        case Dtype::I3:
        case Dtype::F3E1M1:
            return 3u;
        case Dtype::I4:
        case Dtype::F4E2M1:
            return 4u;
        default:
            return 8u;
        }
}

/** @brief True for integer formats (no Flash LUT needed). */
[[nodiscard]] inline constexpr bool
_S_integer_dtype (const Dtype dtype) noexcept
{
    return dtype == Dtype::I8 || dtype == Dtype::I4 || dtype == Dtype::I3 || dtype == Dtype::I2;
}

/**
 * @brief True when one byte holds a whole number of codes.
 *
 * This is the gate for the power-of-two fast paths.  3-bit formats do not
 * divide 8, so they must never be routed through `_S_codes_per_byte` or
 * `_S_code_shift`; use `_S_row_bytes` / `_S_code_at` / the reader reservoir.
 */
[[nodiscard]] inline constexpr bool
_S_byte_divisible (const Dtype dtype) noexcept
{
    return (8u % _S_code_bits (dtype)) == 0u;
}

/**
 * @brief Codes per stream byte for a byte-divisible format: 1, 2 or 4.
 *
 * Returns 0 for 3-bit formats, where the value does not exist.  Callers that
 * would divide by this must first check `_S_byte_divisible`; the returned 0 is
 * a sentinel, not a count.
 */
[[nodiscard]] inline constexpr uint32_t
_S_codes_per_byte (const Dtype dtype) noexcept
{
    const uint32_t bits = _S_code_bits (dtype);
    return (8u % bits) == 0u ? 8u / bits : 0u;
}

/** @brief log2 of the codes per stream byte for byte-divisible formats, else 0. */
[[nodiscard]] inline constexpr uint32_t
_S_code_shift (const Dtype dtype) noexcept
{
    const uint32_t codes = _S_codes_per_byte (dtype);
    if (codes == 0u)
        return 0u;
#if defined(__has_builtin) && __has_builtin(__builtin_ctz)
    if !consteval {
        return static_cast<uint32_t> (__builtin_ctz (codes));
    }
#endif
    uint32_t shift = 0;
    while ((1u << shift) < codes)
        ++shift;
    return shift;
}

/** @brief True when several codes share one stream byte. */
[[nodiscard]] inline constexpr bool
_S_packed_dtype (const Dtype dtype) noexcept
{
    return _S_code_bits (dtype) != 8u;
}

/** @brief Mask of one packed code, i.e. the field width of `dtype`. */
[[nodiscard]] inline constexpr uint32_t
_S_code_mask (const Dtype dtype) noexcept
{
    return (1u << _S_code_bits (dtype)) - 1u;
}

/**
 * @brief Bytes a row of `count` codes occupies.
 *
 * The last byte is always included, so a byte-divisible row with an odd code
 * count (I4/I2) still fits.  A 3-bit row that is not a multiple of 8 codes is
 * rejected by `_S_codes_aligned` before packing; rounding up here merely keeps
 * the sizer total for temporary buffers.
 */
[[nodiscard]] inline constexpr uint32_t
_S_row_bytes (const Dtype dtype, const uint32_t count) noexcept
{
    return (count * _S_code_bits (dtype) + 7u) / 8u;
}

/**
 * @brief True when `count` codes of `dtype` end on a byte boundary.
 *
 * 1/2/4-bit formats store a whole number of codes per byte, so the row may end
 * in a partially filled byte and is still byte-terminated.  3-bit codes span
 * bytes and only align after 8 codes (24 bits = 3 bytes); a row that does not
 * align would share its final byte with the next row, so the validator rejects
 * it instead of silently changing the layout.
 */
[[nodiscard]] inline constexpr bool
_S_codes_aligned (const Dtype dtype, const uint32_t count) noexcept
{
    return _S_byte_divisible (dtype) || (count % 8u) == 0u;
}

/** @brief Byte index of code `index` in a bit-packed row. */
[[nodiscard]] inline constexpr uint32_t
_S_code_byte (const Dtype dtype, const uint32_t index) noexcept
{
    return (index * _S_code_bits (dtype)) >> 3;
}

/** @brief Bit offset of code `index` inside `_S_code_byte (dtype, index)`. */
[[nodiscard]] inline constexpr uint32_t
_S_code_bit (const Dtype dtype, const uint32_t index) noexcept
{
    return (index * _S_code_bits (dtype)) & 7u;
}

/** @brief Code value range of an integer `dtype`: {I8: [-128, 127], I4: [-8, 7], I3: [-4, 3], I2: [-2, 1]}. */
[[nodiscard]] inline constexpr int32_t
_S_code_qmin (const Dtype dtype) noexcept
{
    return -(1 << (_S_code_bits (dtype) - 1u));
}

/**
 * @brief Largest positive code of an integer `dtype`.
 *
 * This is the symmetric-quantization bound: 127 for I8, 7 for I4, 3 for I3 and
 * 1 for I2, matching the module's `[-qmax, qmax]` weight grid.
 */
[[nodiscard]] inline constexpr int32_t
_S_code_qmax (const Dtype dtype) noexcept
{
    return (1 << (_S_code_bits (dtype) - 1u)) - 1;
}

[[nodiscard]] constexpr int32_t
_S_clamp_i8 (const int32_t value) noexcept
{
    return value < -127 ? -127 : (value > 127 ? 127 : value);
}

[[nodiscard]] constexpr int8_t
_S_i8_code (const int32_t value) noexcept
{
    return static_cast<int8_t> (_S_clamp_i8 (value));
}

/**
 * @brief Sign-extend a `bits`-wide two's-complement code.
 *
 * The reader path uses this for every packed integer format, so the field
 * width comes from `_S_code_bits` and never from a hard-coded nibble.
 */
[[nodiscard]] constexpr int32_t
_S_from_int_bits (const uint32_t code, const uint32_t bits) noexcept
{
    const uint32_t field = code & ((1u << bits) - 1u);
    const uint32_t sign = 1u << (bits - 1u);
    return (field & sign) != 0u ? static_cast<int32_t> (field) - static_cast<int32_t> (sign * 2u)
                                : static_cast<int32_t> (field);
}

[[nodiscard]] constexpr int32_t
_S_from_int4 (const uint8_t nibble) noexcept
{
    return _S_from_int_bits (nibble, 4u);
}

/** @brief Two's-complement int2 code: 0, 1 map to 0, 1 and 2, 3 to -2, -1. */
[[nodiscard]] constexpr int32_t
_S_from_int2 (const uint8_t code) noexcept
{
    return _S_from_int_bits (code, 2u);
}

/** @brief Two's-complement int3 code: 0..3 map to 0..3 and 4..7 to -4..-1. */
[[nodiscard]] constexpr int32_t
_S_from_int3 (const uint8_t code) noexcept
{
    return _S_from_int_bits (code, 3u);
}

/** @brief Pack two int4 codes, low nibble first (PyTorch/torchao order). */
[[nodiscard]] constexpr uint8_t
_S_pack_int4 (const int8_t low, const int8_t high) noexcept
{
    return static_cast<uint8_t> ((static_cast<uint8_t> (low) & 0x0F)
                                      | ((static_cast<uint8_t> (high) & 0x0F) << 4));
}

/**
 * @brief Pack four int2 codes, lowest index in the lowest two bits.
 *
 * Same low-bits-first rule as `_S_pack_int4`, so code `i` of a row sits in byte
 * `i / 4` at shift `(i % 4) * 2`.
 */
[[nodiscard]] constexpr uint8_t
_S_pack_int2 (const int8_t c0, const int8_t c1, const int8_t c2, const int8_t c3) noexcept
{
    return static_cast<uint8_t> ((static_cast<uint8_t> (c0) & 0x03)
                                      | ((static_cast<uint8_t> (c1) & 0x03) << 2)
                                      | ((static_cast<uint8_t> (c2) & 0x03) << 4)
                                      | ((static_cast<uint8_t> (c3) & 0x03) << 6));
}

/**
 * @brief One code of a bit-packed LSB-first buffer.
 *
 * Code `index` starts at bit `index * bits`; for 3-bit codes that position can
 * straddle a byte boundary, so the neighbouring byte is folded into a 16-bit
 * word before the shift.  This is the unpacker the exporter tests and the
 * reader's row path must agree with.
 */
[[nodiscard]] constexpr uint32_t
_S_code_at (const uint8_t *const data, const uint32_t index, const uint32_t bits) noexcept
{
    const uint32_t bit = index * bits;
    const uint32_t byte = bit >> 3;
    const uint32_t shift = bit & 7u;
    const uint32_t field = (1u << bits) - 1u;
    uint32_t word = static_cast<uint32_t> (data[byte]);
    if (shift + bits > 8u)
        word |= static_cast<uint32_t> (data[byte + 1u]) << 8;
    return (word >> shift) & field;
}

/**
 * @brief One code of a single packed byte.
 *
 * Only meaningful for byte-divisible formats (1/2/4/8 bits per code); the
 * 3-bit case has no per-byte form and returns 0.  Use the buffer overload above
 * for 3-bit rows.
 */
[[nodiscard]] constexpr uint32_t
_S_code_at (const uint8_t byte, const uint32_t index, const uint32_t bits) noexcept
{
    if ((8u % bits) != 0u)
        return 0u;
    const uint32_t per_byte = 8u / bits;
    return (static_cast<uint32_t> (byte) >> ((index % per_byte) * bits))
           & ((1u << bits) - 1u);
}

/**
 * @brief Write one code into a bit-packed LSB-first buffer.
 *
 * `bit` is an absolute bit offset; a 3-bit code may split across two bytes, in
 * which case the low byte keeps the low bits and `data[byte + 1]` the rest.
 * Bits outside the field are preserved, so codes can be written in any order.
 */
constexpr void
_S_pack_code_at (uint8_t *const data, const uint32_t bit, const int32_t code,
                 const uint32_t bits) noexcept
{
    const uint32_t byte = bit >> 3;
    const uint32_t shift = bit & 7u;
    const uint32_t field = (1u << bits) - 1u;
    const uint32_t value = static_cast<uint32_t> (code) & field;
    data[byte] = static_cast<uint8_t> (
        (static_cast<uint32_t> (data[byte]) & ~(field << shift))
        | ((value << shift) & 0xFFu));
    if (shift + bits > 8u)
        {
            const uint32_t high_shift = 8u - shift;
            data[byte + 1u] = static_cast<uint8_t> (
                (static_cast<uint32_t> (data[byte + 1u]) & ~(field >> high_shift))
                | (value >> high_shift));
        }
}

/**
 * @brief Pack `count` codes low-bits-first into `out`; returns bytes written.
 *
 * Works for every width, including 3-bit: the caller must provide
 * `_S_row_bytes (dtype, count)` bytes.  This is the exporter/host packing
 * helper; the reader never packs.
 */
inline uint32_t
_S_pack_codes (uint8_t *const out, const int8_t *const codes, const uint32_t count,
               const uint32_t bits) noexcept
{
    const uint32_t bytes = (count * bits + 7u) / 8u;
    for (uint32_t i = 0; i < bytes; ++i)
        out[i] = 0;
    for (uint32_t i = 0; i < count; ++i)
        _S_pack_code_at (out, i * bits, codes[i], bits);
    return bytes;
}

/** @brief Pack a row of `count` codes of `dtype` into `out`; returns bytes written. */
inline uint32_t
_S_pack_codes (const Dtype dtype, const int8_t *const codes, const uint32_t count,
               uint8_t *const out) noexcept
{
    return _S_pack_codes (out, codes, count, _S_code_bits (dtype));
}

/**
 * @brief Pack `count` codes (at most one stream byte's worth) low-bits first.
 *
 * Byte-at-a-time convenience kept for the nibble writers; `count * bits` must
 * not exceed 8.
 */
[[nodiscard]] constexpr uint8_t
_S_pack_codes (const int8_t *const codes, const uint32_t count, const uint32_t bits) noexcept
{
    uint8_t byte = 0;
    for (uint32_t i = 0; i < count; ++i)
        byte = static_cast<uint8_t> (byte
                                     | ((static_cast<uint8_t> (codes[i]) & ((1u << bits) - 1u))
                                        << (i * bits)));
    return byte;
}

/** @brief Arithmetic right shift with round-half-up (ties toward +infinity). */
[[nodiscard]] constexpr int32_t
_S_round_shift (const int64_t value, const unsigned int shift) noexcept
{
    if (shift == 0)
        return static_cast<int32_t> (value);
    const int64_t half = static_cast<int64_t> (1) << (shift - 1u);
    return static_cast<int32_t> ((value + half) >> shift);
}

/**
 * @brief Requantize an integer accumulator with a Q15 multiplier.
 *
 * The multiplier folds s_weight * s_input / s_output into one integer so the
 * runtime never touches floating point.
 */
[[nodiscard]] constexpr int32_t
_S_requant_q15 (const int64_t accumulator, const int32_t multiplier_q15) noexcept
{
    return _S_round_shift (accumulator * static_cast<int64_t> (multiplier_q15), 15u);
}

/**
 * @brief Integer square root (floor).
 *
 * The restoring method needs one iteration per two exponent bits, so the
 * leading-zero count sets the first bit pair instead of shifting down from
 * 2^30; a small value therefore costs a couple of steps, not 15.
 */
[[nodiscard]] constexpr uint32_t
_S_isqrt (const uint32_t value) noexcept
{
    if (value <= 1u)
        return value;
    uint32_t root = 0;
    uint32_t bit = static_cast<uint32_t> (1) << ((_S_ilog2 (value) / 2u) * 2u);
    uint32_t remainder = value;
    while (bit != 0)
        {
            if (remainder >= root + bit)
                {
                    remainder -= root + bit;
                    root = (root >> 1) + bit;
                }
            else
                root >>= 1;
            bit >>= 2;
        }
    return root;
}

#ifdef ZZS_MLLEE_HOST
/**
 * @brief PyTorch-style round-to-nearest-even conversion to int8.
 *
 * The clamp happens in float before the integer cast: a huge input, an
 * infinity or a NaN therefore cannot overflow `int32_t` (which would be
 * undefined behaviour) and saturates or maps to zero instead.
 */
[[nodiscard]] inline int32_t
quantize_affine_i8 (const float value, const float scale, const int32_t zero_point) noexcept
{
    if (!(scale > 0.0f))
        return 0;
    const float quotient = value / scale;
    if (!(quotient == quotient))
        return 0;
    const float shifted = std::nearbyint (quotient) + static_cast<float> (zero_point);
    if (!(shifted > -127.0f))
        return -127;
    if (shifted >= 127.0f)
        return 127;
    return static_cast<int32_t> (shifted);
}

/**
 * @brief Symmetric weight quantization on the `[-qmax, qmax]` grid.
 *
 * Used for every integer width: qmax is 127/7/3/1 for I8/I4/I3/I2.  The clamp
 * runs in float before the cast, so infinity saturates, NaN becomes zero and a
 * magnitude above `INT32_MAX * scale` cannot be converted out of range.
 */
[[nodiscard]] inline int32_t
quantize_symmetric (const float value, const float scale, const int32_t qmax) noexcept
{
    if (!(scale > 0.0f) || qmax <= 0)
        return 0;
    const float quotient = value / scale;
    if (!(quotient == quotient))
        return 0;
    const float rounded = std::nearbyint (quotient);
    if (!(rounded > -static_cast<float> (qmax)))
        return -qmax;
    if (rounded >= static_cast<float> (qmax))
        return qmax;
    return static_cast<int32_t> (rounded);
}

/** @brief Symmetric quantization driven by a width in bits (2, 3, 4 or 8). */
[[nodiscard]] inline int32_t
quantize_symmetric_bits (const float value, const float scale, const uint32_t bits) noexcept
{
    if (bits < 2u || bits > 8u)
        return 0;
    return quantize_symmetric (value, scale, (1 << (bits - 1u)) - 1);
}

/** @brief Symmetric quantization driven by an integer `Dtype` (I8/I4/I3/I2). */
[[nodiscard]] inline int32_t
quantize_symmetric_dtype (const float value, const float scale, const Dtype dtype) noexcept
{
    if (!_S_integer_dtype (dtype))
        return 0;
    return quantize_symmetric (value, scale, _S_code_qmax (dtype));
}

/** @brief Dequantize one integer code from any width: `code * scale`. */
[[nodiscard]] inline float
dequantize_code (const int32_t code, const float scale) noexcept
{
    return static_cast<float> (code) * scale;
}

[[nodiscard]] inline float
dequantize_i8 (const int32_t q, const float scale) noexcept
{
    return dequantize_code (q, scale);
}

/**
 * @brief The IEEE-754 binary32 pattern of `value`.
 *
 * `memcpy` rather than a union or `reinterpret_cast`: it is the only spelling
 * that is unconditionally well-defined, and it compiles to a single `mov`
 * because both operands are the same width.
 */
[[nodiscard]] inline uint32_t
_S_float_bits (const float value) noexcept
{
    uint32_t bits = 0u;
    std::memcpy (&bits, &value, sizeof (bits));
    return bits;
}

/**
 * @brief Right shift with round-to-nearest-even, the whole low-bit encoder in
 *        one operation.
 *
 * Every float format below is the binary32 significand narrowed to fewer
 * mantissa bits, which is exactly a shift plus a rounding decision: the round
 * bit is the top discarded bit and the sticky bits are those under it.  Ties
 * resolve to an even result, which is the same rule the exhaustive reference
 * scan applies by preferring the even code.
 */
[[nodiscard]] inline uint32_t
_S_shr_rne (const uint32_t value, const uint32_t shift) noexcept
{
    const uint32_t quotient = value >> shift;
    const uint32_t remainder = value & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1u);
    const bool round_up
        = remainder > half || (remainder == half && (quotient & 1u) != 0u);
    return quotient + (round_up ? 1u : 0u);
}

/**
 * @brief The unbiased exponent of a finite, normal, positive `magnitude`.
 *
 * Reads the binary32 exponent field directly, so it costs one shift and one
 * subtraction instead of a `std::ilogb`/`frexp` call.
 */
[[nodiscard]] inline int32_t
_S_float_exponent (const uint32_t bits) noexcept
{
    return static_cast<int32_t> (bits >> 23) - 127;
}

/** @brief e4m3fn (bias 7, max 448, no infinities, 0x7F/0xFF are NaN). */
[[nodiscard]] inline float
f8_e4m3_to_float (const uint8_t bits) noexcept
{
    const bool negative = (bits & 0x80u) != 0;
    const int exponent = (bits >> 3) & 0x0F;
    const int mantissa = bits & 0x07;
    if (exponent == 0x0F && mantissa == 0x07)
        return std::numeric_limits<float>::quiet_NaN ();
    if (exponent == 0)
        {
            const float value = std::ldexp (static_cast<float> (mantissa) / 8.0f, 1 - 7);
            return negative ? -value : value;
        }
    const float value = std::ldexp (1.0f + static_cast<float> (mantissa) / 8.0f, exponent - 7);
    return negative ? -value : value;
}

/**
 * @brief Narrow a binary32 to e4m3fn in constant time.
 *
 * The exhaustive scan this replaces is exactly round-to-nearest-even over the
 * format grid, so the grid arithmetic does the work instead: below the minimum
 * normal the representable values are a uniform 2^-9 lattice, and above it the
 * binary32 mantissa is shifted down to three bits with `_S_shr_rne` and a
 * rounding carry bumped into the next binade.  The 448 clamp bounds the result
 * to 0x7E, so neither the exponent nor the mantissa can reach the 0x7F NaN
 * pattern, and the sign is re-attached at the end.
 */
[[nodiscard]] inline uint8_t
f8_e4m3_from_float (const float value) noexcept
{
    if (std::isnan (value))
        return 0x7F;
    const bool negative = std::signbit (value);
    float magnitude = std::fabs (value);
    if (!(magnitude < 448.0f))
        magnitude = 448.0f;
    uint8_t code;
    if (magnitude < 0.015625f)
        code = static_cast<uint8_t> (
            static_cast<int32_t> (std::nearbyint (magnitude * 512.0f)));
    else
        {
            const uint32_t bits = _S_float_bits (magnitude);
            int32_t exponent = _S_float_exponent (bits) + 7;
            uint32_t mantissa = _S_shr_rne (bits & 0x7FFFFFu, 20u);
            if (mantissa > 7u)
                {
                    mantissa = 0u;
                    ++exponent;
                }
            code = static_cast<uint8_t> (
                (exponent << 3) | static_cast<int32_t> (mantissa));
        }
    return static_cast<uint8_t> (code | (negative ? 0x80u : 0u));
}

/** @brief e5m2 (bias 15, infinities and NaNs supported). */
[[nodiscard]] inline float
f8_e5m2_to_float (const uint8_t bits) noexcept
{
    const bool negative = (bits & 0x80u) != 0;
    const int exponent = (bits >> 2) & 0x1F;
    const int mantissa = bits & 0x03;
    if (exponent == 0x1F)
        {
            if (mantissa != 0)
                return std::numeric_limits<float>::quiet_NaN ();
            return negative ? -std::numeric_limits<float>::infinity ()
                            : std::numeric_limits<float>::infinity ();
        }
    if (exponent == 0)
        {
            const float value = std::ldexp (static_cast<float> (mantissa) / 4.0f, 1 - 15);
            return negative ? -value : value;
        }
    const float value = std::ldexp (1.0f + static_cast<float> (mantissa) / 4.0f, exponent - 15);
    return negative ? -value : value;
}

/**
 * @brief Narrow a binary32 to e5m2 in constant time.
 *
 * Same lattice argument as e4m3fn with the e5m2 constants: the subnormal step
 * is 2^-16, the mantissa keeps two bits, and the 57344 clamp keeps the result
 * out of the 0x7C/0x7D infinity and 0x7E/0x7F NaN patterns the scan skipped.
 */
[[nodiscard]] inline uint8_t
f8_e5m2_from_float (const float value) noexcept
{
    if (std::isnan (value))
        return 0x7F;
    const bool negative = std::signbit (value);
    float magnitude = std::fabs (value);
    if (!(magnitude < 57344.0f))
        magnitude = 57344.0f;
    uint8_t code;
    if (magnitude < 6.103515625e-05f)
        code = static_cast<uint8_t> (
            static_cast<int32_t> (std::nearbyint (magnitude * 65536.0f)));
    else
        {
            const uint32_t bits = _S_float_bits (magnitude);
            int32_t exponent = _S_float_exponent (bits) + 15;
            uint32_t mantissa = _S_shr_rne (bits & 0x7FFFFFu, 21u);
            if (mantissa > 3u)
                {
                    mantissa = 0u;
                    ++exponent;
                }
            code = static_cast<uint8_t> (
                (exponent << 2) | static_cast<int32_t> (mantissa));
        }
    return static_cast<uint8_t> (code | (negative ? 0x80u : 0u));
}

/** @brief e2m1 values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}, sign in bit 3. */
[[nodiscard]] inline float
f4_e2m1_to_float (const uint8_t bits) noexcept
{
    constexpr float values[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const float value = values[bits & 0x07];
    return (bits & 0x08u) != 0 ? -value : value;
}

/**
 * @brief Narrow a binary32 to e2m1 in constant time.
 *
 * e2m1 has one mantissa bit, so below the minimum normal (1.0) the grid is a
 * uniform 0.5 lattice and the whole code is `round (2 * magnitude)`; at or above
 * it the code is the exponent field shifted left by one plus the rounded bit.
 * Anything from 6.5 upward, infinities included, saturates to 7 (6.0) because
 * the exponent field cannot pass 3.  A NaN keeps the scan's fall-through result:
 * the sign-only code 0.
 */
[[nodiscard]] inline uint8_t
f4_e2m1_from_float (const float value) noexcept
{
    const bool negative = std::signbit (value);
    if (std::isnan (value))
        return static_cast<uint8_t> (negative ? 0x08u : 0u);
    const float magnitude = std::fabs (value);
    uint8_t code;
    if (magnitude < 1.0f)
        code = static_cast<uint8_t> (
            static_cast<int32_t> (std::nearbyint (magnitude * 2.0f)));
    else
        {
            const uint32_t bits = _S_float_bits (magnitude);
            int32_t exponent = _S_float_exponent (bits) + 1;
            uint32_t mantissa = _S_shr_rne (bits & 0x7FFFFFu, 22u);
            if (mantissa > 1u)
                {
                    mantissa = 0u;
                    ++exponent;
                }
            code = exponent > 3
                       ? 7u
                       : static_cast<uint8_t> (
                             (exponent << 1) | static_cast<int32_t> (mantissa));
        }
    return static_cast<uint8_t> (code | (negative ? 0x08u : 0u));
}

/**
 * @brief 3-bit minifloat E1M1: sign bit 2, exponent bit 1, mantissa bit 0.
 *
 * Code layout, LSB-first like every other packed format:
 *
 *     bit 2 = S, bit 1 = E, bit 0 = M
 *
 * With an exponent bias of 1 the IEEE-style value is
 *
 *     E = 0 (subnormal):  M * 2^(1 - bias - M_bits) = M * 2^(1 - 1 - 1) = 0.5 M
 *     E = 1 (normal):     (1 + M/2) * 2^(E - bias)  = 1 + M/2
 *
 * so the four positive codes are {0, 0.5, 1.0, 1.5} and the sign bit mirrors
 * them.  Normal and subnormal steps coincide (0.5), making the magnitude grid a
 * uniform 0.5 lattice; that is the smallest IEEE-style E1M1 encoding with a
 * signed zero and a subnormal, and its values are exactly the low half of
 * F4E2M1.  Code 4 is negative zero, the same convention as F4E2M1 code 8.
 */
[[nodiscard]] inline float
f3_e1m1_to_float (const uint8_t bits) noexcept
{
    constexpr float values[4] = { 0.0f, 0.5f, 1.0f, 1.5f };
    const float value = values[bits & 0x03u];
    return (bits & 0x04u) != 0 ? -value : value;
}

/**
 * @brief Narrow a binary32 to E1M1 in constant time, round-to-nearest-even.
 *
 * The E1M1 magnitude grid is uniform with step 0.5, so `nearbyint (2 * |x|)` is
 * the nearest code and its ties-to-even rule is exactly the "prefer the even
 * code" rule the exhaustive scan uses.  Magnitudes at or above 1.5 (including
 * infinity) saturate to code 3; NaN keeps the sign-only code, as F4E2M1 does.
 */
[[nodiscard]] inline uint8_t
f3_e1m1_from_float (const float value) noexcept
{
    const bool negative = std::signbit (value);
    if (std::isnan (value))
        return static_cast<uint8_t> (negative ? 0x04u : 0u);
    const float magnitude = std::fabs (value);
    uint8_t code;
    if (!(magnitude < 1.5f))
        code = 3u;
    else
        code = static_cast<uint8_t> (
            static_cast<int32_t> (std::nearbyint (magnitude * 2.0f)));
    return static_cast<uint8_t> (code | (negative ? 0x04u : 0u));
}

/** @brief Build the Flash weight LUT mapping an f8 code to a normalized int8 code. */
inline void
build_weight_lut_e4m3 (const float max_abs, int8_t *lut) noexcept
{
    const float scale = max_abs > 0.0f ? max_abs : 1.0f;
    for (int code = 0; code < 256; ++code)
        {
            const float value = f8_e4m3_to_float (static_cast<uint8_t> (code));
            lut[code] = std::isnan (value)
                            ? 0
                            : _S_i8_code (static_cast<int32_t> (
                                  std::nearbyint (value / scale * 127.0f)));
        }
}

inline void
build_weight_lut_e5m2 (const float max_abs, int8_t *lut) noexcept
{
    const float scale = max_abs > 0.0f ? max_abs : 1.0f;
    for (int code = 0; code < 256; ++code)
        {
            const float value = f8_e5m2_to_float (static_cast<uint8_t> (code));
            lut[code] = (!std::isfinite (value))
                            ? 0
                            : _S_i8_code (static_cast<int32_t> (
                                  std::nearbyint (value / scale * 127.0f)));
        }
}

inline void
build_weight_lut_e2m1 (const float max_abs, int8_t *lut) noexcept
{
    const float scale = max_abs > 0.0f ? max_abs : 1.0f;
    for (int code = 0; code < 16; ++code)
        {
            const float value = f4_e2m1_to_float (static_cast<uint8_t> (code));
            lut[code] = _S_i8_code (
                static_cast<int32_t> (std::nearbyint (value / scale * 127.0f)));
        }
}

/** @brief Build the 8-entry Flash LUT for F3E1M1 (codes {0, ±0.5, ±1, ±1.5}). */
inline void
build_weight_lut_e1m1 (const float max_abs, int8_t *lut) noexcept
{
    const float scale = max_abs > 0.0f ? max_abs : 1.0f;
    for (int code = 0; code < 8; ++code)
        {
            const float value = f3_e1m1_to_float (static_cast<uint8_t> (code));
            lut[code] = _S_i8_code (
                static_cast<int32_t> (std::nearbyint (value / scale * 127.0f)));
        }
}

/** @brief Q15 requant multiplier from a float scale ratio. */
[[nodiscard]] inline int32_t
requant_multiplier (const double weight_scale, const double input_scale, const double output_scale) noexcept
{
    if (!(output_scale > 0.0))
        return 0;
    const double ratio = weight_scale * input_scale / output_scale;
    const double scaled = ratio * 32768.0;
    if (scaled <= -2147483648.0)
        return std::numeric_limits<int32_t>::min ();
    if (scaled >= 2147483647.0)
        return std::numeric_limits<int32_t>::max ();
    return static_cast<int32_t> (std::nearbyint (scaled));
}
#endif // ZZS_MLLEE_HOST
} // namespace zzs::mllee
