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
#include "q15.hpp"
#include "q31.hpp"

#include <cstdint>
#include <limits.h>

/**
 * @brief Cross-format conversions between Q11, Q15 and Q31.
 *
 * Every conversion here is explicit at the call site: widening shifts gain
 * fraction bits, narrowing shifts drop them with round-half-up, and every
 * narrowing saturates into the target range.  The same conversions are also
 * reachable from the individual width headers; this header gathers them so a
 * stage that mixes formats has one place to read.
 *
 * `fixed_to_q11/q15/q31(double)` are constexpr only: they accept a double
 * literal so a calibration table can be written in plain decimal and folded at
 * compile time.  They must not be called with a run-time double value -- there
 * is no floating point anywhere else in this submodule.
 * @{
 */
namespace zzs::kernel::sensor::driver::fixed {

/** @brief Q11 -> Q31: shift left by 20 (gain 20 fraction bits).  The shift is
 *         an int64 multiply by 2^20, guarded by __builtin_mul_overflow. */
[[nodiscard]] inline constexpr q31_t
q11_to_q31 (const q11_t value) noexcept {
    int64_t out = 0;
    if (_S_mul_overflow_s64 (static_cast<int64_t> (value), 1LL << 20, out))
        return value > 0 ? _S_q31_max : _S_q31_min;
    return out;
}

/** @brief Q31 -> Q11: narrow with saturation.  Drop 20 fraction bits and
 *         clamp into the Q11 range. */
[[nodiscard]] inline constexpr q11_t
q31_to_q11 (const q31_t value) noexcept {
    const int32_t narrowed = static_cast<int32_t> (value >> 20);
    return narrowed > _S_q11_max
               ? static_cast<q11_t> (_S_q11_max)
               : (narrowed < _S_q11_min ? static_cast<q11_t> (_S_q11_min) : static_cast<q11_t> (narrowed));
}

/** @brief Convert a compile-time double constant to Q11.  constexpr only --
 *         do not pass a run-time double. */
[[nodiscard]] inline constexpr q11_t
fixed_to_q11 (const double value) noexcept {
    return static_cast<q11_t> (value * static_cast<double> (_S_q11_one));
}

/** @brief Convert a compile-time double constant to Q15.  constexpr only. */
[[nodiscard]] inline constexpr q15_t
fixed_to_q15 (const double value) noexcept {
    return static_cast<q15_t> (value * static_cast<double> (_S_q15_one));
}

/** @brief Convert a compile-time double constant to Q31.  constexpr only. */
[[nodiscard]] inline constexpr q31_t
fixed_to_q31 (const double value) noexcept {
    return static_cast<q31_t> (value * static_cast<double> (_S_q31_one));
}

/** @} */
} // namespace zzs::kernel::sensor::driver::fixed
