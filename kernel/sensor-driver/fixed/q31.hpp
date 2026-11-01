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
#include "q15.hpp"

#include <cstdint>
#include <limits.h>

/**
 * @brief Q31 fixed-point arithmetic (0 integer bits, 31 fraction bits,
 *        int64 storage for accumulation).
 *
 * Q31 is a fractional-only format: every value lies in [-1, 1) and products
 * stay in int64 so an accumulator can sum thousands of Q15 samples without
 * losing the low fraction bits.  Used for bias accumulation and offset
 * calibration where a running sum must not saturate at the Q15 wire width.
 * Round-half-up on shifts follows the `_S_round_shift` convention.
 * @{
 */
namespace zzs::kernel::sensor::driver::fixed {

/** @brief Fraction bits of Q31. */
inline constexpr uint32_t _S_q31_frac_bits = 31u;

/** @brief Q31 value of 1.0 (one past the largest positive Q31). */
inline constexpr int64_t _S_q31_one = 1LL << _S_q31_frac_bits;

/** @brief Largest finite Q31 value: 1 - 2^-31. */
inline constexpr int64_t _S_q31_max = (1LL << _S_q31_frac_bits) - 1LL;

/** @brief Smallest finite Q31 value: -1.0. */
inline constexpr int64_t _S_q31_min = -(1LL << _S_q31_frac_bits);

/** @brief Q31 underlying storage type (int64 for accumulation headroom). */
using q31_t = int64_t;

/** @brief Wrap a raw int64 as a Q31 value without range checking. */
[[nodiscard]] inline constexpr q31_t
q31_from_raw (const int64_t raw) noexcept {
        return raw;
}

/** @brief Return the raw int64 storage of a Q31 value. */
[[nodiscard]] inline constexpr int64_t
q31_to_raw (const q31_t value) noexcept {
        return value;
}

/** @brief Saturating Q31 add.  Uses __builtin_add_overflow when available,
 *         with a portable manual fallback otherwise, then clamps to the Q31
 *         range. */
[[nodiscard]] inline constexpr q31_t
q31_add (const q31_t lhs, const q31_t rhs) noexcept {
        int64_t sum = 0;
        if (_S_add_overflow_s64 (lhs, rhs, sum))
                return rhs > 0 ? _S_q31_max : _S_q31_min;
        return sum > _S_q31_max ? _S_q31_max : (sum < _S_q31_min ? _S_q31_min : sum);
}

/** @brief Saturating Q31 subtract.  Uses __builtin_sub_overflow when
 *         available, with a portable manual fallback otherwise, then clamps to
 *         the Q31 range. */
[[nodiscard]] inline constexpr q31_t
q31_sub (const q31_t lhs, const q31_t rhs) noexcept {
        int64_t diff = 0;
        if (_S_sub_overflow_s64 (lhs, rhs, diff))
                return rhs < 0 ? _S_q31_max : _S_q31_min;
        return diff > _S_q31_max ? _S_q31_max : (diff < _S_q31_min ? _S_q31_min : diff);
}

/** @brief Q31 x Q31 -> Q31 with round-half-up.
 *
 *  The full 62-bit product is formed in int64 (exact for two Q31 operands) and
 *  rounded at the Q31 boundary.  An explicit __builtin_mul_overflow check
 *  guards the int64 intermediate before the rounding shift; on overflow the
 *  result saturates to the Q31 range. */
[[nodiscard]] inline constexpr q31_t
q31_mul (const q31_t lhs, const q31_t rhs) noexcept {
        int64_t product = 0;
        if (_S_mul_overflow_s64 (lhs, rhs, product))
                return ((lhs < 0) != (rhs < 0)) ? _S_q31_min : _S_q31_max;
        return zzs::mllee::_S_round_shift (product, _S_q31_frac_bits);
}

/** @brief Q31 x Q31 -> Q31 saturating multiply (explicit saturation helper).
 *
 *  Identical to q31_mul but named to make the saturating intent obvious at the
 *  call site. */
[[nodiscard]] inline constexpr q31_t
q31_mul_sat (const q31_t lhs, const q31_t rhs) noexcept {
        return q31_mul (lhs, rhs);
}

/** @brief Q15 -> Q31: shift left by 16 (gain 16 fraction bits).  The shift is
 *         an int64 multiply by 2^16, guarded by __builtin_mul_overflow. */
[[nodiscard]] inline constexpr q31_t
q31_from_q15 (const q15_t value) noexcept {
        int64_t out = 0;
        if (_S_mul_overflow_s64 (static_cast<int64_t> (value), 1LL << 16, out))
                return value > 0 ? _S_q31_max : _S_q31_min;
        return out;
}

/** @brief Q31 -> Q15: narrow with saturation.  Drop 16 fraction bits and
 *         clamp into the Q15 int16 range. */
[[nodiscard]] inline constexpr q15_t
q31_to_q15 (const q31_t value) noexcept {
        return zzs::kernel::sensor::driver::fixed::q15_clamp (static_cast<int32_t> (value >> 16));
}

/** @brief Add a Q15 sample to a Q31 accumulator.  The sample is promoted to
 *         Q31 (shift left 16) and added with Q31 saturation. */
[[nodiscard]] inline constexpr q31_t
q31_accumulate (const q31_t acc, const q15_t val) noexcept {
        return q31_add (acc, q31_from_q15 (val));
}

/** @} */
} // namespace zzs::kernel::sensor::driver::fixed
