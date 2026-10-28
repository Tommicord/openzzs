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
#include <limits>

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
