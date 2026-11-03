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

/**
 * @file test_builtins.cpp
 * @brief Exercise the `ZZS_*` compiler-builtin wrappers on the host (x86_64).
 *
 * Each macro is checked for correct behaviour where it has an observable
 * result (assume_aligned, likely/unlikely, overflow, memcpy/memset).  Prefetch
 * and unreachable have no observable host effect; they are exercised for
 * compile + run safety.
 */

#include "kernel/device-driver/builtins/zzs_builtins.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <type_traits>
namespace zzs::kernel::builtins::test {

TEST_CASE ("builtins.assume_aligned returns the aligned pointer", "[builtins]") {
    alignas (64) int64_t aligned_storage[8];
    uint8_t buf[64];

    void *p = ZZS_ASSUME_ALIGNED (aligned_storage, 64);
    REQUIRE (p == aligned_storage);

    void *q = ZZS_ASSUME_ALIGNED (buf, 1);
    REQUIRE (q == buf);
}

/** @brief Compile-time alignment check macro: the pointee type's alignment
 *         must be at least the requested alignment. */
TEST_CASE ("builtins.assume_aligned_check is a valid static assertion", "[builtins]") {
    alignas (64) int64_t aligned_storage[8];
    int64_t *p = aligned_storage;
    ZZS_ASSUME_ALIGNED_CHECK (p, 8); /* alignof(int64_t) == 8 */
}

TEST_CASE ("builtins.likely_unlikely preserve the expression truth value", "[builtins]") {
    REQUIRE (ZZS_LIKELY (true) == true);
    REQUIRE (ZZS_LIKELY (false) == false);
    REQUIRE (ZZS_UNLIKELY (true) == true);
    REQUIRE (ZZS_UNLIKELY (false) == false);

    const int x = 7;
    REQUIRE (ZZS_LIKELY (x == 7) == true);
    REQUIRE (ZZS_UNLIKELY (x == 3) == false);
}

TEST_CASE ("builtins.mul_overflow detects and reports overflow", "[builtins]") {
    int32_t r = 0;

    /* No overflow: 1000 * 1000 fits in int32. */
    REQUIRE (ZZS_MUL_OVERFLOW (1000, 1000, r) == false);
    REQUIRE (r == 1000000);

    /* Overflow: 1000000 * 1000000 exceeds int32. */
    REQUIRE (ZZS_MUL_OVERFLOW (1000000, 1000000, r) == true);

    /* Zero is never overflow. */
    REQUIRE (ZZS_MUL_OVERFLOW (0, 12345, r) == false);
    REQUIRE (r == 0);

    /* Negative path: -2 * -2 = 4. */
    REQUIRE (ZZS_MUL_OVERFLOW (-2, -2, r) == false);
    REQUIRE (r == 4);
}

TEST_CASE ("builtins.add_sub_overflow detect bounds", "[builtins]") {
    int32_t r = 0;

    REQUIRE (ZZS_ADD_OVERFLOW (100, 200, r) == false);
    REQUIRE (r == 300);

    REQUIRE (ZZS_ADD_OVERFLOW (INT32_MAX, 1, r) == true); /* INT32_MAX + 1 */

    REQUIRE (ZZS_SUB_OVERFLOW (10, 3, r) == false);
    REQUIRE (r == 7);

    REQUIRE (ZZS_SUB_OVERFLOW (INT32_MIN, 1, r) == true); /* INT32_MIN - 1 */
}

TEST_CASE ("builtins.memcpy_memset copy and fill memory", "[builtins]") {
    uint8_t src[16] = {};
    uint8_t dst[16] = {};
    for (uint32_t i = 0; i < 16; ++i)
        src[i] = static_cast<uint8_t> (i + 1);

    ZZS_MEMCPY (dst, src, 16);
    REQUIRE (std::memcmp (dst, src, 16) == 0);

    ZZS_MEMSET (dst, 0xAA, 16);
    for (uint32_t i = 0; i < 16; ++i)
        REQUIRE (dst[i] == 0xAAu);
}

TEST_CASE ("builtins.prefetch_and_unreachable_run_safely", "[builtins]") {
    uint8_t buf[16] = {};

    /* Prefetch must compile and run without trapping. */
    ZZS_PREFETCH (buf, 0, 3);
    ZZS_PREFETCH (buf, 1, 0);

    /* unreachable is noreturn; exercise it only in a branch the optimizer can
     * prove unreachable so the test itself does not abort.  Calling it directly
     * would end the process, so we verify the type trait instead. */
    static_assert (std::is_same_v<decltype (::zzs::kernel::builtins::unreachable ()), void>,
                   "unreachable returns void");
    (void) buf;
}

} // namespace zzs::kernel::builtins::test
