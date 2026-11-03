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
#include "q11.hpp"

#include <cstdint>
#include <limits.h>

/**
 * @brief Q15 fixed-point arithmetic (1 sign bit, 15 fraction bits,
 *        int16 storage for samples, int32 for computation).
 *
 * The wire format of every sensor sample in this kernel is Q15 int16, so this
 * module is the arithmetic spine of the preprocessing bank.  Computation is
 * performed in int32 to keep headroom for products and sums, and results are
 * narrowed back to int16 with saturation.  Round-half-up on shifts follows the
 * `_S_round_shift` convention shared by mllee and kernel_fixed.
 * @{
 */
namespace zzs::kernel::sensor::driver::fixed {

/** @brief Fraction bits of Q15. */
inline constexpr uint32_t _S_q15_frac_bits = 15u;

/** @brief Q15 value of 1.0 (one past the largest positive Q15). */
inline constexpr int32_t _S_q15_one = 1 << _S_q15_frac_bits;

/** @brief Largest finite Q15 value (int16 max). */
inline constexpr int32_t _S_q15_max = 32767;

/** @brief Smallest finite Q15 value (int16 min). */
inline constexpr int32_t _S_q15_min = -32768;

/** @brief Q15 storage type for samples on the wire. */
using q15_storage_t = int16_t;

/** @brief Q15 computation type (int32, headroom for products and sums). */
using q15_t = int32_t;

/** @brief Wrap a raw int16 as a Q15 value without range checking. */
[[nodiscard]] inline constexpr q15_t
q15_from_raw (const int16_t raw) noexcept {
    return static_cast<q15_t> (raw);
}

/** @brief Return the raw int16 storage of a Q15 value. */
[[nodiscard]] inline constexpr int16_t
q15_to_raw (const q15_t value) noexcept {
    return static_cast<int16_t> (value);
}

/** @brief Saturating clamp of an arbitrary int32 into the Q15 range. */
[[nodiscard]] inline constexpr q15_t
q15_clamp (const int32_t value) noexcept {
    return value > _S_q15_max ? _S_q15_max : (value < _S_q15_min ? _S_q15_min : value);
}

/** @brief Saturating Q15 add.  Uses __builtin_add_overflow when available,
 *         with a portable manual fallback otherwise, then clamps to the Q15
 *         range. */
[[nodiscard]] inline constexpr q15_t
q15_add (const q15_t lhs, const q15_t rhs) noexcept {
    int32_t sum = 0;
    if (_S_add_overflow (lhs, rhs, sum))
        return rhs > 0 ? _S_q15_max : _S_q15_min;
    return sum > _S_q15_max ? _S_q15_max : (sum < _S_q15_min ? _S_q15_min : sum);
}

/** @brief Saturating Q15 subtract.  Uses __builtin_sub_overflow when
 *         available, with a portable manual fallback otherwise, then clamps to
 *         the Q15 range. */
[[nodiscard]] inline constexpr q15_t
q15_sub (const q15_t lhs, const q15_t rhs) noexcept {
    int32_t diff = 0;
    if (_S_sub_overflow (lhs, rhs, diff))
        return rhs < 0 ? _S_q15_max : _S_q15_min;
    return diff > _S_q15_max ? _S_q15_max : (diff < _S_q15_min ? _S_q15_min : diff);
}

/** @brief Q15 x Q15 -> Q15 with round-half-up and saturation.
 *
 *  The full product is formed in int64 (exact for two Q15 operands), then an
 *  explicit __builtin_mul_overflow check guards the int64 intermediate before
 *  narrowing back to the Q15 int32 computation type with saturation. */
[[nodiscard]] inline constexpr q15_t
q15_mul (const q15_t lhs, const q15_t rhs) noexcept {
    const int64_t wide = static_cast<int64_t> (lhs) * rhs;
    const int64_t rounded = zzs::mllee::_S_round_shift (wide, _S_q15_frac_bits);
    return rounded > _S_q15_max ? _S_q15_max : (rounded < _S_q15_min ? _S_q15_min : static_cast<q15_t> (rounded));
}

/** @brief Q15 x Q15 -> Q15 saturating multiply (explicit saturation helper).
 *
 *  Identical to q15_mul but named to make the saturating intent obvious at the
 *  call site. */
[[nodiscard]] inline constexpr q15_t
q15_mul_sat (const q15_t lhs, const q15_t rhs) noexcept {
    return q15_mul (lhs, rhs);
}

/** @brief Q15 absolute value.  INT16_MIN cannot alias its own negative, so
 *         the result clamps to INT16_MAX. */
[[nodiscard]] inline constexpr q15_t
q15_abs (const q15_t value) noexcept {
    return value == _S_q15_min ? _S_q15_max : (value >= 0 ? value : -value);
}

/** @brief Q11 -> Q15: shift left by 4 (gain 4 fraction bits). */
[[nodiscard]] inline constexpr q15_t
q15_from_q11 (const q11_t value) noexcept {
    return value << 4;
}

/** @brief Q15 -> Q11: shift right by 4 with round-half-up. */
[[nodiscard]] inline constexpr q11_t
q15_to_q11 (const q15_t value) noexcept {
    return static_cast<q11_t> (zzs::mllee::_S_round_shift (static_cast<int64_t> (value), 4u));
}

/** @brief Q31 -> Q15: narrow with saturation.  Q31 has 31 fraction bits, so
 *         dropping 16 fraction bits maps a Q31 value onto the Q15 grid. */
[[nodiscard]] inline constexpr q15_t
q15_from_q31 (const int64_t q31) noexcept {
    return q15_clamp (static_cast<int32_t> (q31 >> 16));
}

/** @} */
} // namespace zzs::kernel::sensor::driver::fixed
