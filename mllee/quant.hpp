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
#include <limits>
#endif

namespace zzs::mllee
{
/**
 * @brief Quantized storage formats understood by the core.
 *
 * int8 and int4 are integer affine formats.  The float formats are stored as
 * their native bit patterns; the exporter resolves them into an int8 LUT in
 * Flash so the runtime always multiplies int8 by int8.
 */
enum class Dtype : uint8_t
{
    I8 = 0,
    I4,
    F8E4M3,
    F8E5M2,
    F4E2M1,
};

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

[[nodiscard]] constexpr int32_t
_S_from_int4 (const uint8_t nibble) noexcept
{
    const uint8_t n = nibble & 0x0F;
    return (n & 0x08) != 0 ? static_cast<int32_t> (n) - 16
                           : static_cast<int32_t> (n);
}

/** @brief Pack two int4 codes, low nibble first (PyTorch/torchao order). */
[[nodiscard]] constexpr uint8_t
_S_pack_int4 (const int8_t low, const int8_t high) noexcept
{
    return static_cast<uint8_t> ((static_cast<uint8_t> (low) & 0x0F)
                                      | ((static_cast<uint8_t> (high) & 0x0F) << 4));
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

/** @brief Integer square root (floor). */
[[nodiscard]] constexpr uint32_t
_S_isqrt (const uint32_t value) noexcept
{
    uint32_t root = 0;
    uint32_t bit = static_cast<uint32_t> (1) << 30;
    uint32_t remainder = value;
    while (bit > remainder)
        bit >>= 2;
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
/** @brief PyTorch-style round-to-nearest-even conversion to int8. */
[[nodiscard]] inline int32_t
quantize_affine_i8 (const float value, const float scale, const int32_t zero_point) noexcept
{
    if (!(scale > 0.0f))
        return 0;
    const auto q = static_cast<int32_t> (std::nearbyint (value / scale));
    return _S_clamp_i8 (q + zero_point);
}

/** @brief Symmetric weight quantization used for int8 and int4 weights. */
[[nodiscard]] inline int32_t
quantize_symmetric (const float value, const float scale, const int32_t qmax) noexcept
{
    if (!(scale > 0.0f))
        return 0;
    const auto q = static_cast<int32_t> (std::nearbyint (value / scale));
    return q < -qmax ? -qmax : (q > qmax ? qmax : q);
}

[[nodiscard]] inline float
dequantize_i8 (const int32_t q, const float scale) noexcept
{
    return static_cast<float> (q) * scale;
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

[[nodiscard]] inline uint8_t
f8_e4m3_from_float (float value) noexcept
{
    if (std::isnan (value))
        return 0x7F;
    const bool negative = std::signbit (value);
    float magnitude = std::fabs (value);
    if (magnitude > 448.0f)
        magnitude = 448.0f;
    uint8_t best = 0;
    float best_distance = std::numeric_limits<float>::infinity ();
    for (int code = 0; code < 0x80; ++code)
        {
            const float candidate = f8_e4m3_to_float (static_cast<uint8_t> (code));
            const float distance = std::fabs (candidate - magnitude);
            if (distance < best_distance
                || (distance == best_distance && (code & 1) == 0))
                {
                    best_distance = distance;
                    best = static_cast<uint8_t> (code);
                }
        }
    return static_cast<uint8_t> (best | (negative ? 0x80u : 0u));
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

[[nodiscard]] inline uint8_t
f8_e5m2_from_float (float value) noexcept
{
    if (std::isnan (value))
        return 0x7F;
    const bool negative = std::signbit (value);
    float magnitude = std::fabs (value);
    if (magnitude > 57344.0f)
        magnitude = 57344.0f;
    uint8_t best = 0;
    float best_distance = std::numeric_limits<float>::infinity ();
    for (int code = 0; code < 0x80; ++code)
        {
            const float candidate = f8_e5m2_to_float (static_cast<uint8_t> (code));
            if (std::isinf (candidate))
                continue;
            const float distance = std::fabs (candidate - magnitude);
            if (distance < best_distance
                || (distance == best_distance && (code & 1) == 0))
                {
                    best_distance = distance;
                    best = static_cast<uint8_t> (code);
                }
        }
    return static_cast<uint8_t> (best | (negative ? 0x80u : 0u));
}

/** @brief e2m1 values: {0, 0.5, 1, 1.5, 2, 3, 4, 6}, sign in bit 3. */
[[nodiscard]] inline float
f4_e2m1_to_float (const uint8_t bits) noexcept
{
    constexpr float values[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    const float value = values[bits & 0x07];
    return (bits & 0x08u) != 0 ? -value : value;
}

[[nodiscard]] inline uint8_t
f4_e2m1_from_float (float value) noexcept
{
    const bool negative = std::signbit (value);
    const float magnitude = std::fabs (value);
    uint8_t best = 0;
    float best_distance = std::numeric_limits<float>::infinity ();
    for (int code = 0; code < 8; ++code)
        {
            const float candidate = f4_e2m1_to_float (static_cast<uint8_t> (code));
            const float distance = std::fabs (candidate - magnitude);
            if (distance < best_distance
                || (distance == best_distance && (code & 1) == 0))
                {
                    best_distance = distance;
                    best = static_cast<uint8_t> (code);
                }
        }
    return static_cast<uint8_t> (best | (negative ? 0x08u : 0u));
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
