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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

/**
 * @file test_builtins_portable.cpp
 * @brief Force the portable (non-`__builtin_*_overflow`) fallbacks in
 *        `zzs_builtins.hpp` and verify signed and unsigned boundary behaviour.
 *
 * This translation unit must include the header before anything else defines
 * the builtin-dispatch macro; `ZZS_BUILTINS_FORCE_PORTABLE` selects the
 * fallback path even on GCC/Clang so the code that MSVC actually uses is
 * exercised on every host.
 */

#define ZZS_BUILTINS_FORCE_PORTABLE 1

#include "kernel/device-driver/builtins/zzs_builtins.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

TEST_CASE ("builtins.portable mul_overflow signed boundaries", "[builtins][portable]") {
  int8_t r = 0;

  REQUIRE (ZZS_MUL_OVERFLOW (int8_t (100), int8_t (2), r) == true);
  REQUIRE (r == int8_t (-56));

  REQUIRE (ZZS_MUL_OVERFLOW (int8_t (-128), int8_t (-1), r) == true);
  REQUIRE (r == int8_t (-128));

  REQUIRE (ZZS_MUL_OVERFLOW (int8_t (-128), int8_t (1), r) == false);
  REQUIRE (r == int8_t (-128));

  REQUIRE (ZZS_MUL_OVERFLOW (int8_t (-2), int8_t (-2), r) == false);
  REQUIRE (r == int8_t (4));
}

TEST_CASE ("builtins.portable mul_overflow unsigned boundaries", "[builtins][portable]") {
  uint8_t r = 0;

  REQUIRE (ZZS_MUL_OVERFLOW (uint8_t (100), uint8_t (2), r) == false);
  REQUIRE (r == uint8_t (200));

  REQUIRE (ZZS_MUL_OVERFLOW (uint8_t (200), uint8_t (2), r) == true);
  REQUIRE (r == uint8_t (144));

  REQUIRE (ZZS_MUL_OVERFLOW (uint8_t (0), uint8_t (255), r) == false);
  REQUIRE (r == uint8_t (0));

  uint32_t w = 0;
  REQUIRE (ZZS_MUL_OVERFLOW (uint32_t (100000), uint32_t (20000), w) == false);
  REQUIRE (w == uint32_t (2000000000u));
}

TEST_CASE ("builtins.portable add_overflow signed and unsigned", "[builtins][portable]") {
  int8_t s = 0;

  REQUIRE (ZZS_ADD_OVERFLOW (int8_t (127), int8_t (1), s) == true);
  REQUIRE (s == int8_t (-128));

  REQUIRE (ZZS_ADD_OVERFLOW (int8_t (100), int8_t (27), s) == false);
  REQUIRE (s == int8_t (127));

  REQUIRE (ZZS_ADD_OVERFLOW (int8_t (-128), int8_t (-1), s) == true);
  REQUIRE (s == int8_t (127));

  uint8_t u = 0;

  REQUIRE (ZZS_ADD_OVERFLOW (uint8_t (200), uint8_t (100), u) == true);
  REQUIRE (u == uint8_t (44));

  REQUIRE (ZZS_ADD_OVERFLOW (uint8_t (200), uint8_t (55), u) == false);
  REQUIRE (u == uint8_t (255));
}

TEST_CASE ("builtins.portable sub_overflow signed and unsigned", "[builtins][portable]") {
  int8_t s = 0;

  /* 0 - (-128) = 128, not representable in int8_t; wraps to -128. */
  REQUIRE (ZZS_SUB_OVERFLOW (int8_t (0), int8_t (-128), s) == true);
  REQUIRE (s == int8_t (-128));

  REQUIRE (ZZS_SUB_OVERFLOW (int8_t (127), int8_t (-128), s) == true);
  REQUIRE (s == int8_t (-1));

  REQUIRE (ZZS_SUB_OVERFLOW (int8_t (-128), int8_t (1), s) == true);
  REQUIRE (s == int8_t (127));

  REQUIRE (ZZS_SUB_OVERFLOW (int8_t (10), int8_t (3), s) == false);
  REQUIRE (s == int8_t (7));

  uint8_t u = 0;

  REQUIRE (ZZS_SUB_OVERFLOW (uint8_t (5), uint8_t (10), u) == true);
  REQUIRE (u == uint8_t (251));

  REQUIRE (ZZS_SUB_OVERFLOW (uint8_t (10), uint8_t (5), u) == false);
  REQUIRE (u == uint8_t (5));
}

TEST_CASE ("builtins.assume_aligned_check accepts naturally aligned pointees", "[builtins][portable]") {
  int64_t *p = nullptr;
  ZZS_ASSUME_ALIGNED_CHECK (p, 8);
  ZZS_ASSUME_ALIGNED_CHECK (p, 4);
}
