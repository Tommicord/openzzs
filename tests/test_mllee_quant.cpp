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

#ifndef ZZS_MLLEE_HOST
#define ZZS_MLLEE_HOST
#endif

#include "mllee/mllee.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace zzs::mllee;

TEST_CASE ("int8 helpers clamp and saturate", "[mllee][quant]")
{
    REQUIRE (_S_clamp_i8 (0) == 0);
    REQUIRE (_S_clamp_i8 (127) == 127);
    REQUIRE (_S_clamp_i8 (128) == 127);
    REQUIRE (_S_clamp_i8 (-127) == -127);
    REQUIRE (_S_clamp_i8 (-128) == -127);
    REQUIRE (_S_i8_code (200) == 127);
    REQUIRE (_S_i8_code (-200) == -127);
}

TEST_CASE ("round shift rounds half up", "[mllee][quant]")
{
    REQUIRE (_S_round_shift (0, 0) == 0);
    REQUIRE (_S_round_shift (1234, 0) == 1234);
    REQUIRE (_S_round_shift (100, 1) == 50);
    REQUIRE (_S_round_shift (-100, 1) == -50);
    REQUIRE (_S_round_shift (101, 1) == 51);
    REQUIRE (_S_round_shift (-101, 1) == -50);
    REQUIRE (_S_round_shift (3, 1) == 2);
    REQUIRE (_S_round_shift (-3, 1) == -1);
    REQUIRE (_S_round_shift (1 << 20, 10) == 1024);
}

TEST_CASE ("Q15 requantization", "[mllee][quant]")
{
    REQUIRE (_S_requant_q15 (0, 32768) == 0);
    REQUIRE (_S_requant_q15 (100, 32768) == 100);
    REQUIRE (_S_requant_q15 (-100, 32768) == -100);
    REQUIRE (_S_requant_q15 (100, 16384) == 50);
    REQUIRE (_S_requant_q15 (100, 65536) == 200);
    REQUIRE (_S_requant_q15 (100000, 32768) == 100000);
}

TEST_CASE ("integer square root floors", "[mllee][quant]")
{
    REQUIRE (_S_isqrt (0) == 0);
    REQUIRE (_S_isqrt (1) == 1);
    REQUIRE (_S_isqrt (2) == 1);
    REQUIRE (_S_isqrt (3) == 1);
    REQUIRE (_S_isqrt (4) == 2);
    REQUIRE (_S_isqrt (15) == 3);
    REQUIRE (_S_isqrt (16) == 4);
    REQUIRE (_S_isqrt (255) == 15);
    REQUIRE (_S_isqrt (256) == 16);
    REQUIRE (_S_isqrt (1000000) == 1000);
    REQUIRE (_S_isqrt (999999) == 999);
}

TEST_CASE ("int4 two's complement packing", "[mllee][quant]")
{
    for (int code = -8; code <= 7; ++code)
        {
            const auto nibble = static_cast<uint8_t> (code & 0x0F);
            REQUIRE (_S_from_int4 (nibble) == code);
        }
    REQUIRE (_S_pack_int4 (-8, 7) == 0x78);
    REQUIRE (_S_pack_int4 (-1, 1) == 0x1F);
    REQUIRE (_S_pack_int4 (3, -3) == 0xD3);
    REQUIRE (_S_from_int4 (_S_pack_int4 (-4, 5) & 0x0F) == -4);
    REQUIRE (_S_from_int4 ((_S_pack_int4 (-4, 5) >> 4) & 0x0F) == 5);
}

TEST_CASE ("float8 e4m3fn reference values", "[mllee][quant]")
{
    REQUIRE (f8_e4m3_to_float (0x00) == 0.0f);
    REQUIRE (f8_e4m3_to_float (0x38) == 1.0f);
    REQUIRE (f8_e4m3_to_float (0xB8) == -1.0f);
    REQUIRE (f8_e4m3_to_float (0x3C) == 1.5f);
    REQUIRE (f8_e4m3_to_float (0x7E) == 448.0f);
    REQUIRE (f8_e4m3_to_float (0x01) == std::ldexp (1.0f, -9));
    REQUIRE (std::isnan (f8_e4m3_to_float (0x7F)));
    REQUIRE (std::isnan (f8_e4m3_to_float (0xFF)));

    REQUIRE (f8_e4m3_from_float (0.0f) == 0x00);
    REQUIRE (f8_e4m3_from_float (1.0f) == 0x38);
    REQUIRE (f8_e4m3_from_float (-1.0f) == 0xB8);
    REQUIRE (f8_e4m3_from_float (1.5f) == 0x3C);
    REQUIRE (f8_e4m3_from_float (-1.5f) == 0xBC);
    REQUIRE (f8_e4m3_from_float (500.0f) == 0x7E);
    REQUIRE (f8_e4m3_from_float (-500.0f) == 0xFE);
    REQUIRE (f8_e4m3_from_float (std::numeric_limits<float>::quiet_NaN ()) == 0x7F);

    for (int code = 0; code < 0x7F; ++code)
        {
            const float value = f8_e4m3_to_float (static_cast<uint8_t> (code));
            if (std::fabs (value) < 1e-6f)
                continue;
            REQUIRE (f8_e4m3_from_float (value) == code);
        }
}

TEST_CASE ("float8 e5m2 reference values", "[mllee][quant]")
{
    REQUIRE (f8_e5m2_to_float (0x00) == 0.0f);
    REQUIRE (f8_e5m2_to_float (0x3C) == 1.0f);
    REQUIRE (f8_e5m2_to_float (0xBC) == -1.0f);
    REQUIRE (f8_e5m2_to_float (0x01) == std::ldexp (1.0f, -16));
    REQUIRE (std::isinf (f8_e5m2_to_float (0x7C)));
    REQUIRE (std::isnan (f8_e5m2_to_float (0x7F)));
    REQUIRE (f8_e5m2_from_float (1.0f) == 0x3C);
    REQUIRE (f8_e5m2_from_float (-1.0f) == 0xBC);
    REQUIRE (f8_e5m2_from_float (60000.0f) == 0x7B);
    REQUIRE (f8_e5m2_from_float (std::numeric_limits<float>::infinity ()) == 0x7B);
    REQUIRE (f8_e5m2_from_float (std::numeric_limits<float>::quiet_NaN ()) == 0x7F);
}

TEST_CASE ("float4 e2m1 reference values", "[mllee][quant]")
{
    constexpr float expected[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
    for (int code = 0; code < 8; ++code)
        {
            REQUIRE (f4_e2m1_to_float (static_cast<uint8_t> (code)) == expected[code]);
            REQUIRE (f4_e2m1_to_float (static_cast<uint8_t> (code | 0x08))
                     == -expected[code]);
        }
    REQUIRE (f4_e2m1_from_float (0.1f) == 0x0);
    REQUIRE (f4_e2m1_from_float (0.4f) == 0x1);
    REQUIRE (f4_e2m1_from_float (1.4f) == 0x3);
    REQUIRE (f4_e2m1_from_float (5.9f) == 0x7);
    REQUIRE (f4_e2m1_from_float (-5.9f) == 0xF);
    REQUIRE (f4_e2m1_from_float (100.0f) == 0x7);
}

namespace
{
/**
 * @brief The exhaustive scan the encoders used before the constant-time form.
 *
 * Kept verbatim as the oracle: nearest representable value, ties to the even
 * code, magnitudes above the format maximum clamped first.  The decode table is
 * hoisted because the decoder is not under test here.
 */
template <int Codes>
[[nodiscard]] uint8_t
scan_encode (const float value, const float (&grid)[Codes], const float maximum,
             const bool skip_non_finite, const uint8_t sign_bit) noexcept
{
    if (std::isnan (value))
        return sign_bit != 0x80u
                   ? static_cast<uint8_t> (std::signbit (value) ? sign_bit : 0u)
                   : 0x7Fu;
    const bool negative = std::signbit (value);
    float magnitude = std::fabs (value);
    if (magnitude > maximum)
        magnitude = maximum;
    int best = 0;
    float best_distance = std::numeric_limits<float>::infinity ();
    for (int code = 0; code < Codes; ++code)
        {
            const float candidate = grid[code];
            if (skip_non_finite && !std::isfinite (candidate))
                continue;
            const float distance = std::fabs (candidate - magnitude);
            if (distance < best_distance
                || (distance == best_distance && (code & 1) == 0))
                {
                    best_distance = distance;
                    best = code;
                }
        }
    return static_cast<uint8_t> (best | (negative ? sign_bit : 0u));
}

/** @brief The first disagreement found, plus how many probes disagreed. */
struct ScanDiff
{
    int _M_mismatches = 0;
    float _M_value = 0.0f;
    int _M_expected = 0;
    int _M_actual = 0;
};

/** @brief Every probe worth comparing: grid points, midpoints, ties and ulps. */
template <int Codes, typename NewEncode>
[[nodiscard]] ScanDiff
compare_against_scan (const float (&grid)[Codes], const float maximum,
                      const bool skip_non_finite, const uint8_t sign_bit,
                      const NewEncode &encode) noexcept
{
    ScanDiff diff;
    const auto probe = [&] (const float value)
    {
        const uint8_t expected = scan_encode<Codes> (value, grid, maximum,
                                                     skip_non_finite, sign_bit);
        const uint8_t actual = encode (value);
        if (expected != actual)
            {
                if (diff._M_mismatches == 0)
                    {
                        diff._M_value = value;
                        diff._M_expected = expected;
                        diff._M_actual = actual;
                    }
                ++diff._M_mismatches;
            }
    };
    for (int code = 0; code + 1 < Codes; ++code)
        {
            const float low = grid[code];
            const float high = grid[code + 1];
            if (!std::isfinite (low) || !std::isfinite (high))
                continue;
            const float middle = (low + high) * 0.5f;
            const float span = high - low;
            // Both endpoints, the exact tie, one ulp either side of the tie and
            // the quarter points, each signed.
            const float samples[]
                = { low,     high,     middle,  std::nextafter (middle, low),
                    std::nextafter (middle, high), low + span * 0.25f,
                    low + span * 0.75f };
            for (const float sample : samples)
                {
                    probe (sample);
                    probe (-sample);
                }
        }
    // Beyond the largest finite value and at the special patterns.
    for (const float value
         : { 0.0f, -0.0f, maximum, -maximum, maximum * 2.0f, maximum * 1.001f,
             std::numeric_limits<float>::infinity (),
             -std::numeric_limits<float>::infinity (),
             std::numeric_limits<float>::quiet_NaN (),
             std::numeric_limits<float>::denorm_min (),
             std::numeric_limits<float>::min (), grid[1], grid[1] * 0.5f,
             std::ldexp (grid[1], 126) })
        {
            probe (value);
            probe (-value);
        }
    // A coarse stride through every binary32 binade checks the exponent math
    // at magnitudes the grid never reaches.
    for (uint32_t bits = 0x00000001u; bits < 0x7F800000u; bits += 0x00020000u)
        {
            float value = 0.0f;
            std::memcpy (&value, &bits, sizeof (value));
            probe (value);
            probe (-value);
        }
    return diff;
}
} // namespace

TEST_CASE ("constant-time low-bit encoders equal the exhaustive scan",
          "[mllee][quant]")
{
    float grid8[0x80];
    for (int code = 0; code < 0x80; ++code)
        grid8[code] = f8_e4m3_to_float (static_cast<uint8_t> (code));
    float grid5[0x80];
    for (int code = 0; code < 0x80; ++code)
        grid5[code] = f8_e5m2_to_float (static_cast<uint8_t> (code));
    float grid4[8];
    for (int code = 0; code < 8; ++code)
        grid4[code] = f4_e2m1_to_float (static_cast<uint8_t> (code));
    float grid3[4];
    for (int code = 0; code < 4; ++code)
        grid3[code] = f3_e1m1_to_float (static_cast<uint8_t> (code));

    // Round-trip the whole non-negative grid: the new encoder must reproduce it.
    for (int code = 0; code < 0x7F; ++code)
        {
            if (std::isnan (grid8[code]))
                continue;
            REQUIRE (f8_e4m3_from_float (grid8[code]) == code);
        }
    for (int code = 0; code < 0x7C; ++code)
        REQUIRE (f8_e5m2_from_float (grid5[code]) == code);
    for (int code = 0; code < 8; ++code)
        REQUIRE (f4_e2m1_from_float (grid4[code]) == code);
    for (int code = 0; code < 4; ++code)
        REQUIRE (f3_e1m1_from_float (grid3[code]) == code);

    const ScanDiff e4m3
        = compare_against_scan (grid8, 448.0f, false, 0x80u,
                                [] (float v) { return f8_e4m3_from_float (v); });
    INFO ("e4m3fn: value=" << e4m3._M_value << " expected="
                           << e4m3._M_expected << " actual=" << e4m3._M_actual);
    CHECK (e4m3._M_mismatches == 0);

    const ScanDiff e5m2
        = compare_against_scan (grid5, 57344.0f, true, 0x80u,
                                [] (float v) { return f8_e5m2_from_float (v); });
    INFO ("e5m2: value=" << e5m2._M_value << " expected=" << e5m2._M_expected
                         << " actual=" << e5m2._M_actual);
    CHECK (e5m2._M_mismatches == 0);

    const ScanDiff e2m1
        = compare_against_scan (grid4, 6.0f, false, 0x08u,
                                [] (float v) { return f4_e2m1_from_float (v); });
    INFO ("e2m1: value=" << e2m1._M_value << " expected=" << e2m1._M_expected
                         << " actual=" << e2m1._M_actual);
    CHECK (e2m1._M_mismatches == 0);

    const ScanDiff e1m1
        = compare_against_scan (grid3, 1.5f, false, 0x04u,
                                [] (float v) { return f3_e1m1_from_float (v); });
    INFO ("e1m1: value=" << e1m1._M_value << " expected=" << e1m1._M_expected
                         << " actual=" << e1m1._M_actual);
    CHECK (e1m1._M_mismatches == 0);
}

TEST_CASE ("float format LUTs normalize to int8", "[mllee][quant]")
{
    std::int8_t lut[256] = {};
    build_weight_lut_e4m3 (1.0f, lut);
    REQUIRE (lut[0x00] == 0);
    REQUIRE (lut[0x38] == 127);
    REQUIRE (lut[0xB8] == -127);
    REQUIRE (lut[0x7F] == 0);
    REQUIRE (lut[0xFF] == 0);
    REQUIRE (lut[0xFE] == -127);

    build_weight_lut_e5m2 (1.0f, lut);
    REQUIRE (lut[0x3C] == 127);
    REQUIRE (lut[0xBC] == -127);
    REQUIRE (lut[0x7C] == 0);
    REQUIRE (lut[0x7F] == 0);

    std::int8_t lut4[16] = {};
    build_weight_lut_e2m1 (6.0f, lut4);
    REQUIRE (lut4[0x0] == 0);
    REQUIRE (lut4[0x1] == 11);
    REQUIRE (lut4[0x7] == 127);
    REQUIRE (lut4[0xF] == -127);

    std::int8_t lut3[8] = {};
    build_weight_lut_e1m1 (1.5f, lut3);
    REQUIRE (lut3[0x0] == 0);
    REQUIRE (lut3[0x1] == 42);
    REQUIRE (lut3[0x2] == 85);
    REQUIRE (lut3[0x3] == 127);
    REQUIRE (lut3[0x4] == 0);
    REQUIRE (lut3[0x5] == -42);
    REQUIRE (lut3[0x6] == -85);
    REQUIRE (lut3[0x7] == -127);
    // A LUT scale below the largest representable magnitude clamps to int8.
    build_weight_lut_e1m1 (1.0f, lut3);
    REQUIRE (lut3[0x3] == 127);

    REQUIRE (_S_lut_bytes (Dtype::I8) == 0);
    REQUIRE (_S_lut_bytes (Dtype::I4) == 0);
    REQUIRE (_S_lut_bytes (Dtype::I3) == 0);
    REQUIRE (_S_lut_bytes (Dtype::F3E1M1) == 8);
    REQUIRE (_S_lut_bytes (Dtype::F4E2M1) == 16);
    REQUIRE (_S_lut_bytes (Dtype::F8E4M3) == 256);
    REQUIRE (_S_lut_bytes (Dtype::F8E5M2) == 256);
}

TEST_CASE ("affine and symmetric quantization", "[mllee][quant]")
{
    REQUIRE (quantize_affine_i8 (0.0f, 0.1f, 0) == 0);
    REQUIRE (quantize_affine_i8 (1.0f, 0.1f, 0) == 10);
    REQUIRE (quantize_affine_i8 (-1.0f, 0.1f, 0) == -10);
    REQUIRE (quantize_affine_i8 (1.0f, 0.1f, 3) == 13);
    REQUIRE (quantize_affine_i8 (100.0f, 0.1f, 0) == 127);
    REQUIRE (quantize_affine_i8 (1.0f, 0.0f, 0) == 0);

    REQUIRE (quantize_symmetric (0.0f, 1.0f, 7) == 0);
    REQUIRE (quantize_symmetric (3.6f, 1.0f, 7) == 4);
    REQUIRE (quantize_symmetric (7.9f, 1.0f, 7) == 7);
    REQUIRE (quantize_symmetric (-7.9f, 1.0f, 7) == -7);
    REQUIRE (quantize_symmetric (7.9f, 0.0f, 7) == 0);

    REQUIRE (dequantize_i8 (10, 0.1f) == 1.0f);
    REQUIRE (dequantize_i8 (-10, 0.1f) == -1.0f);
}

TEST_CASE ("requant multiplier is Q15", "[mllee][quant]")
{
    REQUIRE (requant_multiplier (1.0, 1.0, 1.0) == 32768);
    REQUIRE (requant_multiplier (0.5, 1.0, 1.0) == 16384);
    REQUIRE (requant_multiplier (1.0, 1.0, 2.0) == 16384);
    REQUIRE (requant_multiplier (0.1, 0.05, 0.2) == 819);
    REQUIRE (requant_multiplier (1.0, 1.0, 0.0) == 0);
}

TEST_CASE ("dtype values are append-only", "[mllee][quant]")
{
    REQUIRE (static_cast<uint8_t> (Dtype::I8) == 0);
    REQUIRE (static_cast<uint8_t> (Dtype::I4) == 1);
    REQUIRE (static_cast<uint8_t> (Dtype::F8E4M3) == 2);
    REQUIRE (static_cast<uint8_t> (Dtype::F8E5M2) == 3);
    REQUIRE (static_cast<uint8_t> (Dtype::F4E2M1) == 4);
    REQUIRE (static_cast<uint8_t> (Dtype::I2) == 5);
    REQUIRE (static_cast<uint8_t> (Dtype::F3E1M1) == 6);
    REQUIRE (static_cast<uint8_t> (Dtype::I3) == 7);
    REQUIRE (_S_dtype_max == 7);
}

TEST_CASE ("code widths and byte-aligned row math", "[mllee][quant]")
{
    REQUIRE (_S_code_bits (Dtype::I8) == 8);
    REQUIRE (_S_code_bits (Dtype::I4) == 4);
    REQUIRE (_S_code_bits (Dtype::F4E2M1) == 4);
    REQUIRE (_S_code_bits (Dtype::I2) == 2);
    REQUIRE (_S_code_bits (Dtype::I3) == 3);
    REQUIRE (_S_code_bits (Dtype::F3E1M1) == 3);

    REQUIRE (_S_codes_per_byte (Dtype::I8) == 1);
    REQUIRE (_S_codes_per_byte (Dtype::I4) == 2);
    REQUIRE (_S_codes_per_byte (Dtype::I2) == 4);
    // 3-bit codes do not divide a byte; the sentinel is 0, never a divisor.
    REQUIRE (_S_codes_per_byte (Dtype::I3) == 0);
    REQUIRE (_S_codes_per_byte (Dtype::F3E1M1) == 0);
    REQUIRE (_S_byte_divisible (Dtype::I4));
    REQUIRE_FALSE (_S_byte_divisible (Dtype::I3));

    REQUIRE (_S_row_bytes (Dtype::I8, 7) == 7);
    REQUIRE (_S_row_bytes (Dtype::I4, 7) == 4);
    REQUIRE (_S_row_bytes (Dtype::I2, 7) == 2);
    REQUIRE (_S_row_bytes (Dtype::I3, 8) == 3);
    REQUIRE (_S_row_bytes (Dtype::I3, 16) == 6);
    REQUIRE (_S_row_bytes (Dtype::F3E1M1, 24) == 9);

    REQUIRE (_S_codes_aligned (Dtype::I8, 3));
    REQUIRE (_S_codes_aligned (Dtype::I4, 3));
    REQUIRE (_S_codes_aligned (Dtype::I2, 3));
    REQUIRE (_S_codes_aligned (Dtype::I3, 0));
    REQUIRE (_S_codes_aligned (Dtype::I3, 8));
    REQUIRE (_S_codes_aligned (Dtype::I3, 16));
    REQUIRE (_S_codes_aligned (Dtype::F3E1M1, 8));
    REQUIRE_FALSE (_S_codes_aligned (Dtype::I3, 7));
    REQUIRE_FALSE (_S_codes_aligned (Dtype::F3E1M1, 9));

    REQUIRE (_S_code_byte (Dtype::I3, 0) == 0);
    REQUIRE (_S_code_byte (Dtype::I3, 3) == 1);
    REQUIRE (_S_code_byte (Dtype::I3, 7) == 2);
    REQUIRE (_S_code_bit (Dtype::I3, 0) == 0);
    REQUIRE (_S_code_bit (Dtype::I3, 2) == 6);
    REQUIRE (_S_code_bit (Dtype::I3, 3) == 1);
    REQUIRE (_S_code_bit (Dtype::I3, 5) == 7);
}

TEST_CASE ("integer code ranges include int3", "[mllee][quant]")
{
    REQUIRE (_S_code_qmin (Dtype::I8) == -128);
    REQUIRE (_S_code_qmax (Dtype::I8) == 127);
    REQUIRE (_S_code_qmin (Dtype::I4) == -8);
    REQUIRE (_S_code_qmax (Dtype::I4) == 7);
    REQUIRE (_S_code_qmin (Dtype::I3) == -4);
    REQUIRE (_S_code_qmax (Dtype::I3) == 3);
    REQUIRE (_S_code_qmin (Dtype::I2) == -2);
    REQUIRE (_S_code_qmax (Dtype::I2) == 1);

    for (int code = 0; code < 4; ++code)
        {
            constexpr int expected[4] = { 0, 1, -2, -1 };
            REQUIRE (_S_from_int2 (static_cast<uint8_t> (code)) == expected[code]);
        }
    for (int code = 0; code < 8; ++code)
        {
            constexpr int expected[8] = { 0, 1, 2, 3, -4, -3, -2, -1 };
            REQUIRE (_S_from_int3 (static_cast<uint8_t> (code)) == expected[code]);
            REQUIRE (_S_from_int_bits (static_cast<uint32_t> (code), 3u) == expected[code]);
        }
    for (int code = -8; code <= 7; ++code)
        REQUIRE (_S_from_int_bits (static_cast<uint32_t> (code), 4u) == code);
    for (int code = -128; code <= 127; ++code)
        REQUIRE (_S_from_int_bits (static_cast<uint32_t> (code), 8u) == code);
}

TEST_CASE ("int2 packing is four codes per byte, low bits first", "[mllee][quant]")
{
    const uint8_t packed = _S_pack_int2 (-2, -1, 0, 1);
    REQUIRE (packed == 0x4E);
    REQUIRE (_S_from_int2 (static_cast<uint8_t> (_S_code_at (packed, 0, 2))) == -2);
    REQUIRE (_S_from_int2 (static_cast<uint8_t> (_S_code_at (packed, 1, 2))) == -1);
    REQUIRE (_S_from_int2 (static_cast<uint8_t> (_S_code_at (packed, 2, 2))) == 0);
    REQUIRE (_S_from_int2 (static_cast<uint8_t> (_S_code_at (packed, 3, 2))) == 1);
}

TEST_CASE ("int3 packs eight codes into three bytes", "[mllee][quant]")
{
    const std::int8_t codes[8] = { 0, 1, 2, 3, -4, -3, -2, -1 };
    uint8_t bytes[3] = {};
    REQUIRE (_S_pack_codes (Dtype::I3, codes, 8, bytes) == 3);
    REQUIRE (bytes[0] == 0x88);
    REQUIRE (bytes[1] == 0xC6);
    REQUIRE (bytes[2] == 0xFA);
    for (int i = 0; i < 8; ++i)
        {
            REQUIRE (_S_code_at (bytes, static_cast<uint32_t> (i), 3u)
                     == (static_cast<uint32_t> (codes[i]) & 0x07u));
            REQUIRE (_S_from_int3 (static_cast<uint8_t> (_S_code_at (
                         bytes, static_cast<uint32_t> (i), 3u)))
                     == codes[i]);
        }
}

TEST_CASE ("packed codes round-trip at every bit offset", "[mllee][quant]")
{
    for (const uint32_t bits : { 2u, 3u, 4u, 8u })
        {
            const uint32_t field = (1u << bits) - 1u;
            std::mt19937 generator (bits * 7919u + 3u);
            for (uint32_t count : { 1u, 2u, 3u, 5u, 8u, 17u, 37u })
                {
                    std::vector<std::int8_t> codes (count);
                    for (std::int8_t &code : codes)
                        code = static_cast<std::int8_t> (generator () & field);
                    std::vector<uint8_t> packed ((count * bits + 7u) / 8u, 0);
                    REQUIRE (_S_pack_codes (packed.data (), codes.data (), count, bits)
                             == packed.size ());
                    for (uint32_t i = 0; i < count; ++i)
                        REQUIRE (_S_code_at (packed.data (), i, bits)
                                 == (static_cast<uint32_t> (codes[i]) & field));
                }
        }

    // Eight consecutive 3-bit codes visit all eight in-byte bit offsets.
    bool seen[8] = {};
    for (uint32_t i = 0; i < 8; ++i)
        seen[_S_code_bit (Dtype::I3, i)] = true;
    for (const bool offset : seen)
        REQUIRE (offset);

    // Writing a code at an arbitrary offset preserves its neighbours.
    uint8_t scratch[2] = { 0xA5u, 0x5Au };
    _S_pack_code_at (scratch, 6u, 5, 3u);
    REQUIRE (_S_code_at (scratch, 2u, 3u) == 5u);
    REQUIRE ((scratch[0] & 0x3Fu) == (0xA5u & 0x3Fu));
    REQUIRE ((scratch[1] & 0xF8u) == (0x5Au & 0xF8u));
}

TEST_CASE ("float3 e1m1 grid, ties and round trip", "[mllee][quant]")
{
    constexpr float expected[4] = { 0.0f, 0.5f, 1.0f, 1.5f };
    for (int code = 0; code < 4; ++code)
        {
            REQUIRE (f3_e1m1_to_float (static_cast<uint8_t> (code)) == expected[code]);
            REQUIRE (f3_e1m1_to_float (static_cast<uint8_t> (code | 0x04)) == -expected[code]);
        }
    // Exhaustive code -> value -> code round trip, including negative zero.
    for (int code = 0; code < 8; ++code)
        {
            const float value = f3_e1m1_to_float (static_cast<uint8_t> (code));
            REQUIRE (f3_e1m1_from_float (value) == code);
        }

    REQUIRE (f3_e1m1_from_float (0.0f) == 0x0);
    REQUIRE (f3_e1m1_from_float (-0.0f) == 0x4);
    REQUIRE (f3_e1m1_from_float (0.1f) == 0x0);
    REQUIRE (f3_e1m1_from_float (0.25f) == 0x0); // exact tie rounds to even code 0
    REQUIRE (f3_e1m1_from_float (0.3f) == 0x1);
    REQUIRE (f3_e1m1_from_float (0.4f) == 0x1);
    REQUIRE (f3_e1m1_from_float (0.75f) == 0x2); // tie rounds to even code 2
    REQUIRE (f3_e1m1_from_float (1.25f) == 0x2); // tie rounds to even code 2
    REQUIRE (f3_e1m1_from_float (1.4f) == 0x3);
    REQUIRE (f3_e1m1_from_float (1.5f) == 0x3);
    REQUIRE (f3_e1m1_from_float (5.9f) == 0x3);
    REQUIRE (f3_e1m1_from_float (-5.9f) == 0x7);
    REQUIRE (f3_e1m1_from_float (std::numeric_limits<float>::infinity ()) == 0x3);
    REQUIRE (f3_e1m1_from_float (-std::numeric_limits<float>::infinity ()) == 0x7);
    REQUIRE (f3_e1m1_from_float (std::numeric_limits<float>::quiet_NaN ()) == 0x0);
}

TEST_CASE ("width-driven quantization clamps and cannot overflow", "[mllee][quant]")
{
    for (const Dtype dtype : { Dtype::I8, Dtype::I4, Dtype::I3, Dtype::I2 })
        {
            const int32_t qmax = _S_code_qmax (dtype);
            REQUIRE (quantize_symmetric_dtype (0.0f, 0.1f, dtype) == 0);
            REQUIRE (quantize_symmetric_dtype (0.1f * qmax, 0.1f, dtype) == qmax);
            REQUIRE (quantize_symmetric_dtype (-0.1f * qmax, 0.1f, dtype) == -qmax);
            REQUIRE (quantize_symmetric_dtype (100.0f, 0.1f, dtype) == qmax);
            REQUIRE (quantize_symmetric_dtype (-100.0f, 0.1f, dtype) == -qmax);
            REQUIRE (quantize_symmetric_dtype (std::numeric_limits<float>::infinity (),
                                               1.0f, dtype)
                     == qmax);
            REQUIRE (quantize_symmetric_dtype (-std::numeric_limits<float>::infinity (),
                                               1.0f, dtype)
                     == -qmax);
            REQUIRE (quantize_symmetric_dtype (std::numeric_limits<float>::quiet_NaN (),
                                               1.0f, dtype)
                     == 0);
            REQUIRE (quantize_symmetric_dtype (1.0f, 0.0f, dtype) == 0);
        }

    REQUIRE (quantize_symmetric_bits (1.0f, 0.1f, 4) == 7);
    REQUIRE (quantize_symmetric_bits (-1.0f, 0.1f, 3) == -3);
    REQUIRE (quantize_symmetric_bits (1.0f, 0.1f, 2) == 1);
    REQUIRE (quantize_symmetric_bits (1.0f, 0.1f, 1) == 0);
    REQUIRE (quantize_symmetric_bits (1.0f, 0.1f, 9) == 0);

    // Huge quotients used to be an out-of-range float->int conversion.
    REQUIRE (quantize_symmetric (3.0e38f, 1.0e-30f, 7) == 7);
    REQUIRE (quantize_symmetric (-3.0e38f, 1.0e-30f, 7) == -7);
    REQUIRE (quantize_affine_i8 (3.0e38f, 1.0e-30f, 0) == 127);
    REQUIRE (quantize_affine_i8 (-3.0e38f, 1.0e-30f, 0) == -127);
    REQUIRE (quantize_affine_i8 (std::numeric_limits<float>::quiet_NaN (), 1.0f, 0) == 0);

    REQUIRE (dequantize_code (3, 0.25f) == 0.75f);
    REQUIRE (dequantize_code (-2, 0.5f) == -1.0f);
    REQUIRE (dequantize_i8 (10, 0.1f) == dequantize_code (10, 0.1f));
}

namespace
{
void
put_u16 (std::vector<uint8_t> &bytes, const std::size_t offset, const uint16_t value)
{
    bytes[offset] = static_cast<uint8_t> (value & 0xFFu);
    bytes[offset + 1u] = static_cast<uint8_t> (value >> 8);
}

void
put_u32 (std::vector<uint8_t> &bytes, const std::size_t offset, const uint32_t value)
{
    bytes[offset] = static_cast<uint8_t> (value & 0xFFu);
    bytes[offset + 1u] = static_cast<uint8_t> ((value >> 8) & 0xFFu);
    bytes[offset + 2u] = static_cast<uint8_t> ((value >> 16) & 0xFFu);
    bytes[offset + 3u] = static_cast<uint8_t> ((value >> 24) & 0xFFu);
}

/**
 * @brief Minimal valid layer descriptor plus its weight/LUT tail.
 *
 * `weight_codes` is the raw code count; the weight stream is laid out at offset
 * 48 and the optional LUT directly after it.
 */
std::vector<uint8_t>
make_layer (const Op op, const Dtype dtype, const uint16_t in_dim, const uint16_t out_dim,
            const uint32_t weight_codes, const uint32_t lut_bytes, const uint8_t flags = 0,
            const uint8_t head_dim = 0, const uint8_t seq = 0, const uint8_t kernel = 1)
{
    const uint32_t weight_bytes = _S_row_bytes (dtype, weight_codes);
    const uint32_t weight_off = 48u;
    const uint32_t lut_off = lut_bytes != 0u ? weight_off + weight_bytes : 0u;
    std::vector<uint8_t> model (weight_off + weight_bytes + lut_bytes, 0);
    model[0] = static_cast<uint8_t> (op);
    model[1] = static_cast<uint8_t> (Act::NONE);
    model[2] = static_cast<uint8_t> (dtype);
    model[3] = static_cast<uint8_t> (Codec::RAW);
    put_u16 (model, 4, in_dim);
    put_u16 (model, 6, out_dim);
    model[8] = kernel;
    model[9] = 1;
    model[12] = head_dim;
    model[13] = seq;
    model[14] = flags;
    put_u32 (model, 16, weight_off);
    put_u32 (model, 20, weight_bytes);
    put_u32 (model, 32, lut_off);
    put_u16 (model, 36, _S_external_input);
    put_u16 (model, 38, 0);
    put_u32 (model, 44, 32768);
    return model;
}
} // namespace

TEST_CASE ("format validator accepts aligned 3-bit layers and rejects the rest",
           "[mllee][quant]")
{
    Layer layer;

    SECTION ("LINEAR int3 row of 8 codes")
    {
        const std::vector<uint8_t> model
            = make_layer (Op::LINEAR, Dtype::I3, 8, 1, 8, 0);
        REQUIRE (parse_layer (model.data (), static_cast<uint32_t> (model.size ()), 0, layer)
                 == Status::OK);
        REQUIRE (layer._M_dtype == Dtype::I3);
        REQUIRE (layer._M_weight_len == 3);
    }

    SECTION ("LINEAR int3 row of 7 codes is unaligned")
    {
        const std::vector<uint8_t> model
            = make_layer (Op::LINEAR, Dtype::I3, 7, 1, 7, 0);
        REQUIRE (parse_layer (model.data (), static_cast<uint32_t> (model.size ()), 0, layer)
                 == Status::BAD_LAYOUT);
    }

    SECTION ("float3 needs its 8-entry LUT")
    {
        const std::vector<uint8_t> good
            = make_layer (Op::LINEAR, Dtype::F3E1M1, 8, 1, 8, 8);
        REQUIRE (parse_layer (good.data (), static_cast<uint32_t> (good.size ()), 0, layer)
                 == Status::OK);
        const std::vector<uint8_t> missing
            = make_layer (Op::LINEAR, Dtype::F3E1M1, 8, 1, 8, 0);
        REQUIRE (parse_layer (missing.data (), static_cast<uint32_t> (missing.size ()), 0, layer)
                 == Status::BAD_LAYOUT);
    }

    SECTION ("dtype above _S_dtype_max is rejected")
    {
        std::vector<uint8_t> model
            = make_layer (Op::LINEAR, Dtype::I3, 8, 1, 8, 0);
        model[2] = static_cast<uint8_t> (_S_dtype_max + 1u);
        REQUIRE (parse_layer (model.data (), static_cast<uint32_t> (model.size ()), 0, layer)
                 == Status::BAD_LAYOUT);
    }

    SECTION ("depthwise CONV1D row is one tap per channel")
    {
        const std::vector<uint8_t> aligned = make_layer (
            Op::CONV1D, Dtype::I3, 8, 8, 8, 0, CONV_DEPTHWISE, 1, 1, 8);
        REQUIRE (parse_layer (aligned.data (), static_cast<uint32_t> (aligned.size ()), 0, layer)
                 == Status::OK);
        const std::vector<uint8_t> unaligned = make_layer (
            Op::CONV1D, Dtype::I3, 8, 8, 1, 0, CONV_DEPTHWISE, 1, 1, 1);
        REQUIRE (parse_layer (unaligned.data (),
                              static_cast<uint32_t> (unaligned.size ()), 0, layer)
                 == Status::BAD_LAYOUT);
    }
}
