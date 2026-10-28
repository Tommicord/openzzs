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

#pragma once

#include "mllee/ops.hpp"
#include "mllee/quant.hpp"
#include "mllee/status.hpp"

#include <cstdint>

/**
 * @brief Fixed-point vocabulary for the training path.
 *
 * The inference core works in int8 activation codes with Q15 requantization.
 * Training reuses exactly that value model, so a tensor produced here can be
 * serialized by the packlow codec and executed by Core without a conversion
 * step:
 *
 *   - weights, activations and outputs are int8 codes in [-127, 127]
 *     (the -128 code is never emitted; see zzs::mllee::_S_i8_code);
 *   - gradients, momenta and learning rates are Q15 values held in int16
 *     storage ([-32768, 32767], one unit = 1/32768) and widened to int32 or
 *     int64 inside a kernel;
 *   - products of two Q15 values are reduced back with a rounded shift
 *     (_S_round_shift), never with a float multiply.
 *
 * Nothing in this header allocates, throws, recurses or reads a writable
 * static: every table is constexpr and lands in .rodata, and every function is
 * either constexpr or inline on caller-owned storage.
 */
namespace zzs::mllee::train
{
/** @brief Fraction width of the training fixed-point format. */
inline constexpr unsigned _S_grad_frac_bits = 15u;

/** @brief 1.0 in Q15; also the exclusive upper bound of a saturated value. */
inline constexpr int32_t _S_q15_one = 32768;

inline constexpr int32_t _S_q15_max = 32767;
inline constexpr int32_t _S_q15_min = -32768;

/** @brief Q15 representation of e^-1 (rounded), used by decay schedules. */
inline constexpr int32_t _S_q15_inv_e = 12069;

/** @brief One code step expressed in Q15: 32768 / 127, exact enough to be a constant. */
inline constexpr int32_t _S_code_to_q15_scale = 258;

/**
 * @name Code / Q15 conversions
 *
 * A code c maps to the Q15 value c * 258, so code 127 is 32766 (just under 1.0)
 * and the round trip is lossless: q15_to_code (code_to_q15 (c)) == c for every
 * c in [-127, 127].
 * @{
 */

[[nodiscard]] inline constexpr int32_t
code_to_q15 (const int32_t code) noexcept
{
    return code * _S_code_to_q15_scale;
}

[[nodiscard]] inline constexpr int8_t
q15_to_code (const int32_t value_q15) noexcept
{
    return _S_i8_code (_S_round_shift (
        static_cast<int64_t> (value_q15) * 127, _S_grad_frac_bits));
}

/** @brief Saturating narrowing of an int32 Q15 intermediate into int16 storage. */
[[nodiscard]] inline constexpr int16_t
q15_to_i16_sat (const int32_t value) noexcept
{
    if (value > _S_q15_max)
        return static_cast<int16_t> (_S_q15_max);
    if (value < _S_q15_min)
        return static_cast<int16_t> (_S_q15_min);
    return static_cast<int16_t> (value);
}

/** @brief Saturating clamp of a Q15 int32 to the Q15 range. */
[[nodiscard]] inline constexpr int32_t
q15_clamp (const int32_t value) noexcept
{
    if (value > _S_q15_max)
        return _S_q15_max;
    if (value < _S_q15_min)
        return _S_q15_min;
    return value;
}

/** @} */

/**
 * @name Core Q15 arithmetic
 *
 * Every helper here is branch-free except where a saturation actually has to
 * pick a different result, and every one has a single defined behaviour at the
 * range edges -- the training path must not depend on undefined overflow.
 * @{
 */

/** @brief (a * b) / 2^15 with the same round-half-up remainder as _S_round_shift
 *         (ties go toward +infinity, so the bias is +0.5 LSB, not sign-symmetric). */
[[nodiscard]] inline constexpr int32_t
q15_mul (const int32_t a, const int32_t b) noexcept
{
    return q15_clamp (_S_round_shift (static_cast<int64_t> (a) * b, _S_grad_frac_bits));
}

/** @brief (a * b) / 2^15 truncated toward minus infinity (cheaper, biased). */
[[nodiscard]] inline constexpr int32_t
q15_mul_trunc (const int32_t a, const int32_t b) noexcept
{
    return q15_clamp (static_cast<int32_t> (
        (static_cast<int64_t> (a) * b) >> _S_grad_frac_bits));
}

/** @brief (a * b + round) >> shift for an int64 accumulator, saturating to Q15. */
[[nodiscard]] inline constexpr int32_t
q15_requant (const int64_t accumulator, const uint32_t shift) noexcept
{
    return q15_clamp (_S_round_shift (accumulator, shift));
}

/** @brief a / b in Q15 (a and b are both Q15); b == 0 yields 0, never a trap. */
[[nodiscard]] inline constexpr int32_t
q15_div (const int32_t a, const int32_t b) noexcept
{
    if (b == 0)
        return 0;
    const int64_t scaled = (static_cast<int64_t> (a) << _S_grad_frac_bits) / b;
    return q15_clamp (static_cast<int32_t> (
        scaled > _S_q15_max ? _S_q15_max : (scaled < _S_q15_min ? _S_q15_min : scaled)));
}

/** @brief Saturating Q15 sum of two Q15 values. */
[[nodiscard]] inline constexpr int32_t
q15_add_sat (const int32_t a, const int32_t b) noexcept
{
    return q15_clamp (a + b);
}

/** @brief Saturating Q15 difference of two Q15 values. */
[[nodiscard]] inline constexpr int32_t
q15_sub_sat (const int32_t a, const int32_t b) noexcept
{
    return q15_clamp (a - b);
}

/** @brief a * b / 2^shift with rounding, for an int32 accumulator and a small shift. */
[[nodiscard]] inline constexpr int32_t
q15_scale_shift (const int32_t a, const int32_t b, const uint32_t shift) noexcept
{
    return static_cast<int32_t> (
        (static_cast<int64_t> (a) * b + (static_cast<int64_t> (1) << (shift - 1u))) >> shift);
}

/** @brief 1024-iteration-free reciprocal: floor(2^30 / x) saturating to Q15. */
[[nodiscard]] inline constexpr int32_t
q15_reciprocal (const int32_t x_q15) noexcept
{
    if (x_q15 <= 0)
        return _S_q15_max;
    const int64_t scaled = (static_cast<int64_t> (1) << 30) / x_q15;
    return static_cast<int32_t> (scaled > _S_q15_max ? _S_q15_max : scaled);
}

/** @brief sqrt in Q15: sqrt(x * 2^15) using the core's integer square root. */
[[nodiscard]] inline constexpr int32_t
q15_sqrt (const int32_t x_q15) noexcept
{
    if (x_q15 <= 0)
        return 0;
    const uint32_t scaled = static_cast<uint32_t> (x_q15) << _S_grad_frac_bits;
    return static_cast<int32_t> (_S_isqrt (scaled));
}

/** @brief Integer absolute value of a Q15 value (no INT_MIN aliasing). */
[[nodiscard]] inline constexpr int32_t
q15_abs (const int32_t value) noexcept
{
    return value < 0 ? -value : value;
}

/** @} */

/**
 * @name Transcendental helpers
 *
 * Two fixed-point domains are in play, because one Q15 lane cannot hold a
 * magnitude of 1 or more:
 *
 *   - Q15 (15 fraction bits) for probabilities, gradients and activations;
 *   - Q7  (7 fraction bits)  for logarithm and exponent domains, where a
 *     realistic loss term spans roughly -30 .. 0 and Q15 would overflow.
 *
 * exp comes from the inference module's own table (_S_exp_table: 256 entries of
 * exp(-i/8)) with linear interpolation between entries, so the worst error is
 * the curvature of one 1/8 step (2e-4 at its steepest) and both paths agree on
 * what exp means.  tanh, sigmoid and log are built on that same exp, which
 * keeps them defined over the whole +/-8 window instead of only the [-1, 1]
 * slice the inference activation table covers.
 *
 * The table's own tail floors at 4 (its generator recurs with a half-LSB
 * rounding term, so the sequence stalls at 4 once the per-step decay drops below
 * it, at t near 9).  Nothing inside the documented +/-8 window reaches that
 * floor -- t = 8 is entry 64, still 11 -- but a caller that asks for exp(-20)
 * gets 4, not 0, and that is the useful direction: the smallest probability the
 * log path can see is 4 / 2^15, which caps one cross-entropy term at ln(8192) =
 * 9.01, i.e. 1153 Q7 units, comfortably inside the Q7 loss window above.
 * @{
 */

/** @brief Fraction width of the log/exp domain. */
inline constexpr unsigned _S_log_frac_bits = 7u;

/** @brief 1.0 in Q7 log units. */
inline constexpr int32_t _S_q7_one = 128;

/** @brief ln 2 in Q7 log units (rounded; the arithmetic uses the Q15 form). */
inline constexpr int32_t _S_ln2_q7 = 89;

/** @brief ln 2 in Q15, the form every reduction below keeps its error in. */
inline constexpr int32_t _S_ln2_q15 = 22714;

/** @brief Largest |x| the exp/tanh paths accept before clamping: 8.0 in Q15. */
inline constexpr int32_t _S_exp_clamp_q15 = 262144;

/** @brief num / den as a Q15 value; the operands are plain int32, so a ratio of
 *         two Q15 quantities that exceeds 1.0 still evaluates correctly. */
[[nodiscard]] inline constexpr int32_t
ratio_q15 (const int32_t num, const int32_t den) noexcept
{
    if (den == 0)
        return 0;
    const int64_t scaled = (static_cast<int64_t> (num) << 15) / den;
    return q15_clamp (static_cast<int32_t> (
        scaled > _S_q15_max ? _S_q15_max : (scaled < -_S_q15_one ? -_S_q15_one : scaled)));
}

/** @brief exp(-t) in Q15 for t >= 0, interpolated on the inference exp table. */
[[nodiscard]] inline constexpr int32_t
q15_exp_neg (const int32_t t_q15) noexcept
{
    if (t_q15 <= 0)
        return _S_q15_max;
    const int64_t scaled = static_cast<int64_t> (t_q15) * 8;
    const int32_t index = static_cast<int32_t> (scaled >> 15);
    if (index >= 255)
        return 0;
    const int32_t fraction = static_cast<int32_t> (scaled & 0x7FFFu);
    const int32_t low = _S_exp_table._M_value[static_cast<uint32_t> (index)];
    const int32_t high = _S_exp_table._M_value[static_cast<uint32_t> (index + 1)];
    return low + (static_cast<int32_t> (high - low) * fraction >> 15);
}

/** @brief exp(x) in Q15.  x > 0 would exceed 1.0 and clamps to the Q15 max. */
[[nodiscard]] inline constexpr int32_t
q15_exp (const int32_t x_q15) noexcept
{
    return x_q15 <= 0 ? q15_exp_neg (-x_q15) : _S_q15_max;
}

/** @brief exp(x) for an x already in Q7 log units, result in Q15. */
[[nodiscard]] inline constexpr int32_t
q15_exp_q7 (const int32_t x_q7) noexcept
{
    return q15_exp (x_q7 << 8);
}

/** @brief tanh(x) for x in Q15, over the full +/-8 domain. */
[[nodiscard]] inline constexpr int32_t
q15_tanh (const int32_t x_q15) noexcept
{
    const int32_t magnitude = x_q15 < 0 ? -x_q15 : x_q15;
    const int32_t clamped
        = magnitude > _S_exp_clamp_q15 ? _S_exp_clamp_q15 : magnitude;
    const int32_t decay = q15_exp_neg (2 * clamped);
    const int32_t value = ratio_q15 (_S_q15_one - 1 - decay, _S_q15_one + decay);
    return x_q15 < 0 ? -value : value;
}

/** @brief sigmoid(x) = 1 / (1 + exp(-x)) in Q15; saturates at the Q15 edges. */
[[nodiscard]] inline constexpr int32_t
q15_sigmoid (const int32_t x_q15) noexcept
{
    if (x_q15 <= -_S_exp_clamp_q15)
        return 0;
    if (x_q15 >= _S_exp_clamp_q15)
        return _S_q15_max;
    /* exp(-x) only represents x >= 0, so the negative half uses
       sigmoid(-a) = 1 - sigmoid(a) instead of an unrepresentable exp. */
    if (x_q15 >= 0)
        return ratio_q15 (_S_q15_one, _S_q15_one + q15_exp_neg (x_q15));
    const int32_t mirrored
        = ratio_q15 (_S_q15_one, _S_q15_one + q15_exp_neg (-x_q15));
    return _S_q15_one - mirrored;
}

/** @brief sigmoid'(x) = s * (1 - s) in Q15, without a second exp evaluation. */
[[nodiscard]] inline constexpr int32_t
q15_sigmoid_grad (const int32_t s_q15) noexcept
{
    return q15_mul (s_q15, _S_q15_one - 1 - s_q15);
}

/** @brief log(p) in Q7 for a positive Q15 p; p <= 0 maps to the -inf sentinel. */
[[nodiscard]] inline constexpr int32_t
q7_log_q15 (const int32_t p_q15) noexcept
{
    if (p_q15 <= 0)
        return _S_q15_min;
    /* Normalize p to the [0.5, 1) window: p = m * 2^-n with m in Q15. */
    uint32_t shifted = static_cast<uint32_t> (p_q15);
    int32_t n = 0;
    while (shifted < 16384u)
        {
            shifted <<= 1;
            ++n;
        }
    while (shifted >= _S_q15_one)
        {
            shifted >>= 1;
            --n;
        }
    const int32_t m = static_cast<int32_t> (shifted);
    /* ln m = 2 * atanh((m - 1) / (m + 1)), series in u^2 (|u| <= 1/3). */
    const int32_t u = ratio_q15 (m - _S_q15_one, m + _S_q15_one);
    const int32_t u2 = q15_mul (u, u);
    const int32_t inner = q15_mul (u2, 6554) + 10923;   /* u2/5 + 1/3 */
    const int32_t bracket = q15_mul (u2, inner) + _S_q15_one; /* 1 + u2/3 + u2^2/5 */
    const int32_t ln_m = 2 * q15_mul (u, bracket);
    /* ln p = ln m - n * ln 2, accumulated in Q15 so the octave count does not
       multiply the rounding error of a Q7 ln 2, then narrowed to Q7. */
    const int32_t total_q15 = ln_m - n * _S_ln2_q15;
    const int32_t result
        = _S_round_shift (static_cast<int64_t> (total_q15) * _S_q7_one, 15u);
    return result < _S_q15_min ? _S_q15_min : result;
}

/** @brief -log(p) in Q7, never negative: the cross-entropy primitive. */
[[nodiscard]] inline constexpr int32_t
q7_neg_log_q15 (const int32_t p_q15) noexcept
{
    const int32_t l = q7_log_q15 (p_q15);
    return l > 0 ? 0 : -l;
}

/** @brief Convert a Q15 magnitude to Q7 log units (only 1.0 -> 0 is meaningful). */
[[nodiscard]] inline constexpr int32_t
q7_from_q15 (const int32_t value_q15) noexcept
{
    return value_q15 >> 8;
}

/** @brief Convert Q7 log units back to Q15, saturating outside the Q15 window. */
[[nodiscard]] inline constexpr int32_t
q15_from_q7 (const int32_t value_q7) noexcept
{
    return q15_clamp (value_q7 << 8);
}

/** @brief ln(1 + exp(x)) in Q7 for x in Q15: the softplus/log-sum-exp atom. */
[[nodiscard]] inline constexpr int32_t
q7_log1p_exp (const int32_t x_q15) noexcept
{
    if (x_q15 > 8192)
        {
            /* ln(1 + e^x) = x + ln(1 + e^-x) and the correction is under 0.001. */
            const int32_t correction
                = q7_log_q15 (_S_q15_one + q15_exp_neg (x_q15));
            return q7_from_q15 (x_q15) + correction;
        }
    return q7_log_q15 (_S_q15_one + q15_exp (x_q15));
}

/** @} */

/** @brief Round-half-away-from-zero division of two int32 values (no floats). */
[[nodiscard]] inline constexpr int32_t
div_round (int32_t numerator, const int32_t denominator) noexcept
{
    if (denominator == 0)
        return 0;
    const bool negative
        = (numerator < 0) != (denominator < 0);
    if (numerator < 0)
        numerator = -numerator;
    const int32_t absolute_denominator
        = denominator < 0 ? -denominator : denominator;
    const int32_t quotient
        = (numerator + absolute_denominator / 2) / absolute_denominator;
    return negative ? -quotient : quotient;
}

/** @brief Smallest multiple of @p alignment that is >= @p value. */
[[nodiscard]] inline constexpr uint32_t
align_up (const uint32_t value, const uint32_t alignment) noexcept
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

/** @brief True when @p value is a valid Q15 gradient lane. */
[[nodiscard]] inline constexpr bool
q15_in_range (const int32_t value) noexcept
{
    return value >= _S_q15_min && value <= _S_q15_max;
}
} // namespace zzs::mllee::train
