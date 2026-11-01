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

#include <cstdint>

/**
 * @brief Fixed-point arithmetic types for @mllee-native/ dataset generation.
 *
 * All types are constexpr-compatible and usable in consteval contexts.
 * No float/double anywhere.  Storage widths:
 *   - Q11: int16_t storage, 11 fractional bits (range ~-16.0 to +15.999)
 *   - Q15: int16_t storage, 15 fractional bits (range ~-1.0 to +0.99997)
 *   - Q31: int32_t storage, 31 fractional bits (range ~-1.0 to +0.9999999995)
 *
 * Arithmetic uses widening int32/int64 intermediates to avoid overflow.
 * @{
 */
namespace zzs::mllee_native::fx {

/** @brief Q11 fixed-point value stored as int16 (1 sign + 4 integer + 11 fractional). */
struct Q11 {
  int16_t _M_raw;

  [[nodiscard]] static constexpr Q11
  from_raw (int16_t raw) noexcept {
    return Q11{raw};
  }
  [[nodiscard]] static constexpr Q11
  from_int (int16_t v) noexcept {
    return Q11{static_cast<int16_t> (v << 11u)};
  }
  [[nodiscard]] static constexpr Q11
  from_frac (int32_t num, int32_t den) noexcept {
    return Q11{static_cast<int16_t> ((static_cast<int64_t> (num) << 11) / den)};
  }
  [[nodiscard]] constexpr int32_t
  to_int32 () const noexcept {
    return static_cast<int32_t> (_M_raw) >> 11;
  }
  [[nodiscard]] constexpr int64_t
  to_i64_q11 () const noexcept {
    return static_cast<int64_t> (_M_raw);
  }
};

/** @brief Q15 fixed-point value stored as int16 (1 sign + 15 fractional). */
struct Q15 {
  int16_t _M_raw;

  [[nodiscard]] static constexpr Q15
  from_raw (int16_t raw) noexcept {
    return Q15{raw};
  }
  [[nodiscard]] static constexpr Q15
  from_int (int16_t v) noexcept {
    return Q15{static_cast<int16_t> (v << 15u)};
  }
  [[nodiscard]] static constexpr Q15
  from_frac (int32_t num, int32_t den) noexcept {
    return Q15{static_cast<int16_t> ((static_cast<int64_t> (num) << 15) / den)};
  }
  [[nodiscard]] constexpr int32_t
  to_int32 () const noexcept {
    return static_cast<int32_t> (_M_raw) >> 15;
  }
  [[nodiscard]] constexpr int64_t
  to_i64_q15 () const noexcept {
    return static_cast<int64_t> (_M_raw);
  }
};

/** @brief Q31 fixed-point value stored as int32 (1 sign + 31 fractional). */
struct Q31 {
  int32_t _M_raw;

  [[nodiscard]] static constexpr Q31
  from_raw (int32_t raw) noexcept {
    return Q31{raw};
  }
  [[nodiscard]] static constexpr Q31
  from_int (int32_t v) noexcept {
    return Q31{v << 31u};
  }
  [[nodiscard]] static constexpr Q31
  from_frac (int64_t num, int64_t den) noexcept {
    return Q31{static_cast<int32_t> ((num << 31) / den)};
  }
  [[nodiscard]] constexpr int64_t
  to_i64_q31 () const noexcept {
    return static_cast<int64_t> (_M_raw);
  }
};

/** @brief Q11 x Q11 -> Q11 (saturating). */
[[nodiscard]] constexpr Q11
q11_mul (const Q11 &a, const Q11 &b) noexcept {
  const int64_t prod = a.to_i64_q11 () * b.to_i64_q11 ();
  const int64_t res = (prod + (1 << 10)) >> 11;
  const int64_t clamped = res > 32767 ? 32767 : (res < -32768 ? -32768 : res);
  return Q11{static_cast<int16_t> (clamped)};
}

/** @brief Q15 x Q15 -> Q15 (saturating). */
[[nodiscard]] constexpr Q15
q15_mul (const Q15 &a, const Q15 &b) noexcept {
  const int64_t prod = a.to_i64_q15 () * b.to_i64_q15 ();
  const int64_t res = (prod + (1 << 14)) >> 15;
  const int64_t clamped = res > 32767 ? 32767 : (res < -32768 ? -32768 : res);
  return Q15{static_cast<int16_t> (clamped)};
}

/** @brief Q31 x Q31 -> Q31 (saturating). */
[[nodiscard]] constexpr Q31
q31_mul (const Q31 &a, const Q31 &b) noexcept {
  const int64_t prod = a.to_i64_q31 () * b.to_i64_q31 ();
  const int64_t res = (prod + (1 << 30)) >> 31;
  const int64_t clamped = res > 2147483647LL ? 2147483647LL : (res < -2147483648LL ? -2147483648LL : res);
  return Q31{static_cast<int32_t> (clamped)};
}

/** @brief Q11 + Q11 (saturating). */
[[nodiscard]] constexpr Q11
q11_add (const Q11 &a, const Q11 &b) noexcept {
  const int32_t sum = static_cast<int32_t> (a._M_raw) + b._M_raw;
  const int32_t clamped = sum > 32767 ? 32767 : (sum < -32768 ? -32768 : sum);
  return Q11{static_cast<int16_t> (clamped)};
}

/** @brief Q15 + Q15 (saturating). */
[[nodiscard]] constexpr Q15
q15_add (const Q15 &a, const Q15 &b) noexcept {
  const int32_t sum = static_cast<int32_t> (a._M_raw) + b._M_raw;
  const int32_t clamped = sum > 32767 ? 32767 : (sum < -32768 ? -32768 : sum);
  return Q15{static_cast<int16_t> (clamped)};
}

/** @brief Q31 + Q31 (saturating). */
[[nodiscard]] constexpr Q31
q31_add (const Q31 &a, const Q31 &b) noexcept {
  const int64_t sum = static_cast<int64_t> (a._M_raw) + b._M_raw;
  const int64_t clamped = sum > 2147483647LL ? 2147483647LL : (sum < -2147483648LL ? -2147483648LL : sum);
  return Q31{static_cast<int32_t> (clamped)};
}

/** @brief Convert Q11 -> Q15 (widen). */
[[nodiscard]] constexpr Q15
q11_to_q15 (const Q11 &v) noexcept {
  return Q15{static_cast<int16_t> (static_cast<int32_t> (v._M_raw) << 4)};
}

/** @brief Convert Q15 -> Q11 (narrow, saturating). */
[[nodiscard]] constexpr Q11
q15_to_q11 (const Q15 &v) noexcept {
  const int32_t v32 = static_cast<int32_t> (v._M_raw) >> 4;
  const int32_t clamped = v32 > 32767 ? 32767 : (v32 < -32768 ? -32768 : v32);
  return Q11{static_cast<int16_t> (clamped)};
}

/** @brief Convert Q15 -> Q31 (widen). */
[[nodiscard]] constexpr Q31
q15_to_q31 (const Q15 &v) noexcept {
  return Q31{static_cast<int32_t> (v._M_raw) << 16};
}

/** @brief Convert Q31 -> Q15 (narrow, saturating). */
[[nodiscard]] constexpr Q15
q31_to_q15 (const Q31 &v) noexcept {
  const int64_t v64 = static_cast<int64_t> (v._M_raw) >> 16;
  const int64_t clamped = v64 > 32767 ? 32767 : (v64 < -32768 ? -32768 : v64);
  return Q15{static_cast<int16_t> (clamped)};
}

/** @brief Absolute value of Q11. */
[[nodiscard]] constexpr Q11
q11_abs (const Q11 &v) noexcept {
  return Q11{v._M_raw < 0 ? static_cast<int16_t> (-v._M_raw) : v._M_raw};
}

/** @brief Absolute value of Q15. */
[[nodiscard]] constexpr Q15
q15_abs (const Q15 &v) noexcept {
  return Q15{v._M_raw < 0 ? static_cast<int16_t> (-v._M_raw) : v._M_raw};
}

/** @brief Absolute value of Q31. */
[[nodiscard]] constexpr Q31
q31_abs (const Q31 &v) noexcept {
  return Q31{v._M_raw < 0 ? static_cast<int32_t> (-v._M_raw) : v._M_raw};
}

/** @} */
} // namespace zzs::mllee_native::fx
