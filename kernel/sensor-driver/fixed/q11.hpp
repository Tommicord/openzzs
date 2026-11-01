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

#pragma once

#include "kernel/kernel_fixed.hpp"

#include <cstdint>
#include <limits.h>

/**
 * @brief Q11 fixed-point arithmetic (20 integer bits, 11 fraction bits,
 *        int32 storage).
 *
 * The fixed driver submodule owns these widths so the sensor pipeline never
 * mixes scales silently.  Q11 is the narrowest working format: it holds values
 * up to ~1024 with 1/2048 resolution and is used where headroom matters more
 * than fraction precision (temperature compensation coefficients, narrow
 * accumulator windows).
 *
 * All functions are constexpr, noexcept and [[nodiscard]].  Saturating on
 * overflow; round-half-up on shifts (the _S_round_shift convention mllee and
 * kernel_fixed already share).
 * @{
 */
namespace zzs::kernel::sensor::driver::fixed {

/** @brief Overflow-checked addition, preferring compiler builtins. */
#if defined(__has_builtin) && __has_builtin(__builtin_add_overflow)
[[nodiscard]] inline constexpr bool
_S_add_overflow (int32_t a, int32_t b, int32_t &out) noexcept {
        return __builtin_add_overflow (a, b, &out);
}
#else
[[nodiscard]] inline constexpr bool
_S_add_overflow (int32_t a, int32_t b, int32_t &out) noexcept {
        if (b > 0 && a > INT32_MAX - b)
                return true;
        if (b < 0 && a < INT32_MIN - b)
                return true;
        out = a + b;
        return false;
}
#endif

/** @brief Overflow-checked subtraction, preferring compiler builtins. */
#if defined(__has_builtin) && __has_builtin(__builtin_sub_overflow)
[[nodiscard]] inline constexpr bool
_S_sub_overflow (int32_t a, int32_t b, int32_t &out) noexcept {
        return __builtin_sub_overflow (a, b, &out);
}
#else
[[nodiscard]] inline constexpr bool
_S_sub_overflow (int32_t a, int32_t b, int32_t &out) noexcept {
        if (b < 0 && a > INT32_MAX + b)
                return true;
        if (b > 0 && a < INT32_MIN + b)
                return true;
        out = a - b;
        return false;
}
#endif

/** @brief Overflow-checked multiplication, preferring compiler builtins. */
#if defined(__has_builtin) && __has_builtin(__builtin_mul_overflow)
[[nodiscard]] inline constexpr bool
_S_mul_overflow (int32_t a, int32_t b, int32_t &out) noexcept {
        return __builtin_mul_overflow (a, b, &out);
}
#else
[[nodiscard]] inline constexpr bool
_S_mul_overflow (int32_t a, int32_t b, int32_t &out) noexcept {
        if (a == 0 || b == 0) {
                out = 0;
                return false;
        }
        const int32_t abs_a = a < 0 ? (a == INT32_MIN ? INT32_MAX : -a) : a;
        const int32_t abs_b = b < 0 ? (b == INT32_MIN ? INT32_MAX : -b) : b;
        if (abs_a > INT32_MAX / abs_b)
                return true;
        out = a * b;
        return false;
}
#endif

/** @brief Overflow-checked addition for int64, preferring compiler builtins. */
#if defined(__has_builtin) && __has_builtin(__builtin_add_overflow)
[[nodiscard]] inline constexpr bool
_S_add_overflow_s64 (int64_t a, int64_t b, int64_t &out) noexcept {
        return __builtin_add_overflow (a, b, &out);
}
#else
[[nodiscard]] inline constexpr bool
_S_add_overflow_s64 (int64_t a, int64_t b, int64_t &out) noexcept {
        if (b > 0 && a > INT64_MAX - b)
                return true;
        if (b < 0 && a < INT64_MIN - b)
                return true;
        out = a + b;
        return false;
}
#endif

/** @brief Overflow-checked subtraction for int64, preferring compiler builtins. */
#if defined(__has_builtin) && __has_builtin(__builtin_sub_overflow)
[[nodiscard]] inline constexpr bool
_S_sub_overflow_s64 (int64_t a, int64_t b, int64_t &out) noexcept {
        return __builtin_sub_overflow (a, b, &out);
}
#else
[[nodiscard]] inline constexpr bool
_S_sub_overflow_s64 (int64_t a, int64_t b, int64_t &out) noexcept {
        if (b < 0 && a > INT64_MAX + b)
                return true;
        if (b > 0 && a < INT64_MIN + b)
                return true;
        out = a - b;
        return false;
}
#endif

/** @brief Overflow-checked multiplication for int64, preferring compiler builtins. */
#if defined(__has_builtin) && __has_builtin(__builtin_mul_overflow)
[[nodiscard]] inline constexpr bool
_S_mul_overflow_s64 (int64_t a, int64_t b, int64_t &out) noexcept {
        return __builtin_mul_overflow (a, b, &out);
}
#else
[[nodiscard]] inline constexpr bool
_S_mul_overflow_s64 (int64_t a, int64_t b, int64_t &out) noexcept {
        if (a == 0 || b == 0) {
                out = 0;
                return false;
        }
        const int64_t abs_a = a < 0 ? (a == INT64_MIN ? INT64_MAX : -a) : a;
        const int64_t abs_b = b < 0 ? (b == INT64_MIN ? INT64_MAX : -b) : b;
        if (abs_a > INT64_MAX / abs_b)
                return true;
        out = a * b;
        return false;
}
#endif

/** @brief Fraction bits of Q11. */
inline constexpr uint32_t _S_q11_frac_bits = 11u;

/** @brief Q11 value of 1.0. */
inline constexpr int32_t _S_q11_one = 1 << _S_q11_frac_bits;

/** @brief Largest finite Q11 value: (2^20 - 1) in Q11. */
inline constexpr int32_t _S_q11_max = (1 << 20) - 1;

/** @brief Smallest finite Q11 value: -(2^20) in Q11. */
inline constexpr int32_t _S_q11_min = -(1 << 20);

/** @brief Q11 underlying storage type. */
using q11_t = int32_t;

/** @brief Wrap a raw int32 as a Q11 value without range checking. */
[[nodiscard]] inline constexpr q11_t
q11_from_raw (const int32_t raw) noexcept {
        return raw;
}

/** @brief Return the raw int32 storage of a Q11 value. */
[[nodiscard]] inline constexpr int32_t
q11_to_raw (const q11_t value) noexcept {
        return value;
}

/** @brief Saturating Q11 add.  Uses __builtin_add_overflow when available,
 *         with a portable manual fallback otherwise, then clamps to the Q11
 *         range (the builtin only guards machine overflow, not format range). */
[[nodiscard]] inline constexpr q11_t
q11_add (const q11_t lhs, const q11_t rhs) noexcept {
        int32_t sum = 0;
        if (_S_add_overflow (lhs, rhs, sum))
                return rhs > 0 ? _S_q11_max : _S_q11_min;
        return sum > _S_q11_max ? _S_q11_max : (sum < _S_q11_min ? _S_q11_min : sum);
}

/** @brief Saturating Q11 subtract.  Uses __builtin_sub_overflow when
 *         available, with a portable manual fallback otherwise, then clamps to
 *         the Q11 range. */
[[nodiscard]] inline constexpr q11_t
q11_sub (const q11_t lhs, const q11_t rhs) noexcept {
        int32_t diff = 0;
        if (_S_sub_overflow (lhs, rhs, diff))
                return rhs < 0 ? _S_q11_max : _S_q11_min;
        return diff > _S_q11_max ? _S_q11_max : (diff < _S_q11_min ? _S_q11_min : diff);
}

/** @brief Q11 x Q11 -> Q11 with round-half-up and saturation. */
[[nodiscard]] inline constexpr q11_t
q11_mul (const q11_t lhs, const q11_t rhs) noexcept {
        const int32_t product =
            static_cast<int32_t> (zzs::mllee::_S_round_shift (static_cast<int64_t> (lhs) * rhs, _S_q11_frac_bits));
        return product > _S_q11_max ? _S_q11_max : (product < _S_q11_min ? _S_q11_min : product);
}

/** @brief Q15 -> Q11: shift right by 4 (drop 4 fraction bits). */
[[nodiscard]] inline constexpr q11_t
q11_from_q15 (const int32_t q15) noexcept {
        return static_cast<q11_t> (zzs::mllee::_S_round_shift (static_cast<int64_t> (q15), 4u));
}

/** @brief Q11 -> Q15: shift left by 4 (gain 4 fraction bits).  The shift is
 *         an int32 multiply by 16; it cannot overflow int32 for any valid
 *         Q11 operand, but the builtin check guards the narrowing anyway. */
[[nodiscard]] inline constexpr int32_t
q11_to_q15 (const q11_t value) noexcept {
        int32_t out = 0;
        if (_S_mul_overflow (value, 16, out))
                return value > 0 ? INT32_MAX : INT32_MIN;
        return out;
}

/** @} */
} // namespace zzs::kernel::sensor::driver::fixed
