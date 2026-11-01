/*
 * Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

/**
 * @file test_fixed_point.cpp
 * @brief Unit tests for Q11/Q15/Q31 fixed-point arithmetic and conversions.
 */

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "kernel/sensor-driver/fixed/fixed_conversions.hpp"
#include "kernel/sensor-driver/fixed/q11.hpp"
#include "kernel/sensor-driver/fixed/q15.hpp"
#include "kernel/sensor-driver/fixed/q31.hpp"

namespace
{
using namespace zzs::kernel::sensor::driver::fixed;
} // namespace

TEST_CASE ("Q11 add saturates at max/min bounds", "[sensor-driver][fixed-point][q11]")
{
    REQUIRE (q11_add (_S_q11_max, 1) == _S_q11_max);
    REQUIRE (q11_add (_S_q11_min, -1) == _S_q11_min);
    REQUIRE (q11_add (_S_q11_max, _S_q11_max) == _S_q11_max);
    REQUIRE (q11_add (_S_q11_min, _S_q11_min) == _S_q11_min);

    /* No saturation well inside range. */
    REQUIRE (q11_add (100, 200) == 300);
    REQUIRE (q11_add (-100, -200) == -300);
}

TEST_CASE ("Q11 sub saturates at max/min bounds", "[sensor-driver][fixed-point][q11]")
{
    REQUIRE (q11_sub (_S_q11_max, -1) == _S_q11_max);
    REQUIRE (q11_sub (_S_q11_min, 1) == _S_q11_min);
    REQUIRE (q11_sub (0, _S_q11_max) == -_S_q11_max);
    REQUIRE (q11_sub (0, _S_q11_min) == _S_q11_max);

    REQUIRE (q11_sub (500, 200) == 300);
    REQUIRE (q11_sub (-500, -200) == -300);
}

TEST_CASE ("Q15 add saturates at Q15_MAX/Q15_MIN", "[sensor-driver][fixed-point][q15]")
{
    REQUIRE (q15_add (_S_q15_max, 1) == _S_q15_max);
    REQUIRE (q15_add (_S_q15_min, -1) == _S_q15_min);
    REQUIRE (q15_add (_S_q15_max, _S_q15_max) == _S_q15_max);
    REQUIRE (q15_add (_S_q15_min, _S_q15_min) == _S_q15_min);

    REQUIRE (q15_add (10000, 20000) == 30000);
    REQUIRE (q15_add (-10000, -20000) == -30000);
}

TEST_CASE ("Q15 sub saturates at Q15_MAX/Q15_MIN", "[sensor-driver][fixed-point][q15]")
{
    REQUIRE (q15_sub (_S_q15_max, -1) == _S_q15_max);
    REQUIRE (q15_sub (_S_q15_min, 1) == _S_q15_min);
    REQUIRE (q15_sub (0, _S_q15_max) == -_S_q15_max);
    REQUIRE (q15_sub (0, _S_q15_min) == _S_q15_max);

    REQUIRE (q15_sub (50000, 20000) == 30000);
    REQUIRE (q15_sub (-50000, 20000) == _S_q15_min);
}

TEST_CASE ("Q15 multiply round-half-up and saturation", "[sensor-driver][fixed-point][q15]")
{
    /* 1.0 * 1.0 saturates at Q15_MAX (1.0 is not representable in Q15). */
    REQUIRE (q15_mul (_S_q15_one, _S_q15_one) == _S_q15_max);

    /* 0.5 * 0.5 == 0.25 in Q15. */
    REQUIRE (q15_mul (_S_q15_one / 2, _S_q15_one / 2) == _S_q15_one / 4);

    /* Saturation on large product. */
    REQUIRE (q15_mul (_S_q15_max, _S_q15_max) == 32766);

    /* Round-half-up: (1 << 14) * (1 << 14) >> 15 with +2^14 rounding term. */
    REQUIRE (q15_mul (1 << 14, 1 << 14) == (1 << 13));

    /* Negative operands. */
    REQUIRE (q15_mul (-_S_q15_one, _S_q15_one) == -_S_q15_one);
}

TEST_CASE ("Q15_MIN negation cannot alias its own negative", "[sensor-driver][fixed-point][q15][edge]")
{
    REQUIRE (q15_abs (_S_q15_min) == _S_q15_max);
    REQUIRE (q15_abs (_S_q15_max) == _S_q15_max);
    REQUIRE (q15_abs (0) == 0);
    REQUIRE (q15_abs (-100) == 100);
}

TEST_CASE ("Q31 accumulation without overflow (many Q15 samples)",
           "[sensor-driver][fixed-point][q31]")
{
    q31_t acc = 0;
    /* Accumulate max Q15 many times; Q31 has 31 fraction bits so this must not
     * overflow the way Q15 would. */
    for (int i = 0; i < 100000; ++i)
        acc = q31_accumulate (acc, _S_q15_max);
    REQUIRE (acc > 0);

    /* Accumulate min Q15 many times. */
    acc = 0;
    for (int i = 0; i < 100000; ++i)
        acc = q31_accumulate (acc, _S_q15_min);
    REQUIRE (acc < 0);
}

TEST_CASE ("Q31 accumulation of max values saturates", "[sensor-driver][fixed-point][q31][edge]")
{
    q31_t acc = _S_q31_max;
    REQUIRE (q31_add (acc, 1) == _S_q31_max);
    REQUIRE (q31_add (_S_q31_min, -1) == _S_q31_min);
}

TEST_CASE ("Cross-format conversions Q11<->Q15", "[sensor-driver][fixed-point][conversion]")
{
    /* Q11 -> Q15 shifts left 4 (gain 4 fraction bits). */
    REQUIRE (q11_to_q15 (_S_q11_one) == _S_q15_one);
    REQUIRE (q11_to_q15 (0) == 0);
    REQUIRE (q11_to_q15 (-_S_q11_one) == _S_q15_min);

    /* Q15 -> Q11 shifts right 4 with round-half-up. */
    REQUIRE (q15_to_q11 (_S_q15_one) == _S_q11_one);
    REQUIRE (q15_to_q11 (0) == 0);
    REQUIRE (q15_to_q11 (-_S_q15_one) == -_S_q11_one);
}

TEST_CASE ("Cross-format conversions Q15<->Q31", "[sensor-driver][fixed-point][conversion]")
{
    /* Q15 -> Q31 shifts left 16. */
    REQUIRE (q31_from_q15 (_S_q15_one) == _S_q31_one); /* 1.0 Q15 -> 1.0 Q31 */
    REQUIRE (q31_from_q15 (0) == 0);
    REQUIRE (q31_from_q15 (_S_q15_min) == _S_q31_min);

    /* Q31 -> Q15 drops 16 fraction bits with saturation. */
    REQUIRE (q31_to_q15 (_S_q31_one) == _S_q15_max);
    REQUIRE (q31_to_q15 (0) == 0);
    REQUIRE (q31_to_q15 (_S_q31_min) == _S_q15_min);
    REQUIRE (q31_to_q15 (_S_q31_max) == _S_q15_max);
}

TEST_CASE ("Cross-format conversions Q11<->Q31", "[sensor-driver][fixed-point][conversion]")
{
    /* Q11 -> Q31 shifts left 20. */
    REQUIRE (q11_to_q31 (_S_q11_one) == _S_q31_one); /* 1.0 Q11 -> 1.0 Q31 */
    REQUIRE (q11_to_q31 (0) == 0);
    REQUIRE (q11_to_q31 (-_S_q11_one) == _S_q31_min);

    /* Q31 -> Q11 drops 20 fraction bits with saturation. */
    REQUIRE (q31_to_q11 (_S_q31_one) == _S_q11_one);
    REQUIRE (q31_to_q11 (0) == 0);
    REQUIRE (q31_to_q11 (_S_q31_min) == -_S_q11_one);
    REQUIRE (q31_to_q11 (_S_q31_max) == 2047);
}

TEST_CASE ("Zero handling in all formats", "[sensor-driver][fixed-point][edge]")
{
    REQUIRE (q11_add (0, 0) == 0);
    REQUIRE (q11_sub (0, 0) == 0);
    REQUIRE (q11_mul (0, _S_q11_max) == 0);
    REQUIRE (q11_from_raw (0) == 0);
    REQUIRE (q11_to_raw (0) == 0);

    REQUIRE (q15_add (0, 0) == 0);
    REQUIRE (q15_sub (0, 0) == 0);
    REQUIRE (q15_mul (0, _S_q15_max) == 0);
    REQUIRE (q15_abs (0) == 0);
    REQUIRE (q15_from_raw (0) == 0);
    REQUIRE (q15_to_raw (0) == 0);

    REQUIRE (q31_add (0, 0) == 0);
    REQUIRE (q31_sub (0, 0) == 0);
    REQUIRE (q31_mul (0, _S_q31_max) == 0);
    REQUIRE (q31_from_raw (0) == 0);
    REQUIRE (q31_to_raw (0) == 0);
    REQUIRE (q31_accumulate (0, 0) == 0);
}

TEST_CASE ("constexpr compile-time evaluation", "[sensor-driver][fixed-point][constexpr]")
{
    constexpr q11_t a = q11_add (100, 200);
    static_assert (a == 300, "Q11 add must be constexpr");

    constexpr q15_t b = q15_mul (_S_q15_one, _S_q15_one);
    static_assert (b == _S_q15_max, "Q15 mul saturates at max");

    constexpr q31_t c = q31_from_q15 (_S_q15_one);
    static_assert (c == static_cast<q31_t> (_S_q15_one) << 16, "Q31 from Q15 must be constexpr");

    constexpr q15_t d = q11_to_q15 (_S_q11_one);
    static_assert (d == _S_q15_one, "Q11 to Q15 must be constexpr");
}
