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

#include "kernel/kernel_config.hpp"
#include "mllee/core_arith.hpp"
#include "mllee/core_ops.hpp"
#include "mllee/core_quant.hpp"

#include <cstdint>

/**
 * @brief The kernel's fixed-point vocabulary.
 *
 * Every arithmetic decision the preprocessing bank makes is here, so that the
 * question "what scale is this value in" has one answer per function instead of
 * one per stage.  The helpers deliberately sit on top of the primitives mllee
 * already ships -- `_S_round_shift` for rounding, `_S_isqrt` for the square
 * root, `_S_mulhi_u32` and `_S_div32_make` for division by an invariant,
 * `_S_mulhi_s64` for wide products -- and add only what a DSP path needs and
 * the inference core has no reason to own: the coefficient scales, the integer
 * logarithm, and the runtime trigonometry the bin banks configure themselves.
 *
 * Nothing here targets a vector unit; the lane forms are in
 * kernel_fixed_simd.hpp and are required to compute exactly what the scalar
 * function computes, which is what makes `-DZZS_KERNEL_SCALAR=1` a usable
 * differential test rather than a second implementation.
 *
 * @section Formats
 *
 * The four widths the module shares are declared below.  A function name states
 * its scales: `q15_mul` takes two Q15 and returns one, `resonator_step` takes a
 * Q13 coefficient and a Q7 state.  Mixing two formats is always explicit at the
 * call site, never hidden inside a cast.
 */
namespace zzs::kernel
{
/** @brief Fraction bits of a normalized sample on the wire (int16 storage). */
inline constexpr uint32_t _S_sample_frac_bits = 15u;
/** @brief Fraction bits of a filter state or accumulator (int32 storage). */
inline constexpr uint32_t _S_state_frac_bits = 15u;
/** @brief Fraction bits of a Goertzel resonator state (int16 storage). */
inline constexpr uint32_t _S_resonator_frac_bits = 7u;
/** @brief Fraction bits of a resonator coefficient: the cosine itself, Q15. */
inline constexpr uint32_t _S_coeff_frac_bits = 15u;

/**
 * @brief The `2 *` of `2*cos(theta)` comes from the multiply, not the constant.
 *
 * A hardware fixed-point rounding multiply -- `vpmulhrsw` on x86, `vqdmulh_s16`
 * on NEON -- computes `(a*b + 2^14) >> 15`, which is the Q15 product of two Q15
 * operands and is already a *doubling* high multiply: it returns `2*a*b` scaled
 * by 2^16.  So feeding it a Q15 cosine and a Q15-scaled state yields
 * `cos * state`, and one saturating doubling turns it into the `2*cos * state`
 * the Goertzel recursion wants.
 *
 * Storing `2*cos` directly instead would need Q15 values up to 65536, one past
 * int16, and a bank would have to track a per-bin scale factor to avoid it.
 * Carrying the cosine and doubling the product costs one extra lane instruction
 * and no bookkeeping at all, which is the trade this module takes.
 */
inline constexpr uint32_t _S_coeff_double_shift = 1u;

/** @brief Fraction bits of the `10*log10(2)` constant, independent of the coefficient format. */
inline constexpr uint32_t _S_log10_const_frac_bits = 13u;

/** @brief int16 extremes of the sample format, the clamp every stage applies on write. */
inline constexpr int32_t _S_sample_max = 32767;
inline constexpr int32_t _S_sample_min = -32768;

/** @brief Q15 value of 1.0, used as the exclusive ceiling of a saturated Q15. */
inline constexpr int32_t _S_q15_one = 32768;

/**
 * @brief Shift between the wire format and the resonator format.
 *
 * A Q15 sample loses eight fraction bits to become a Q7 resonator input, which
 * is the cost of the bound `_S_resonator_max_samples * |x| / 2 <= 2^14`: a
 * resonator's state grows linearly with the window at its own frequency, so the
 * state scale, not the sample scale, is what limits the input.  Coherent
 * integration over the window returns most of the discarded bits -- the
 * magnitude of a matched tone scales with N -- so the loss lands on the noise
 * floor and not on the feature.
 */
inline constexpr uint32_t _S_resonator_input_shift = _S_sample_frac_bits - _S_resonator_frac_bits;
} // namespace zzs::kernel

namespace zzs::kernel::fx
{
using zzs::mllee::_S_round_shift;

/** @brief Round-half-up arithmetic shift of an int32 by a runtime-independent count. */
[[nodiscard]] inline constexpr int32_t
asr_round (const int32_t value, const uint32_t shift) noexcept
{
    if (shift == 0u)
        return value;
    return static_cast<int32_t> ((static_cast<int64_t> (value)
                                  + (static_cast<int64_t> (1) << (shift - 1u)))
                                 >> shift);
}

/** @brief Truncating arithmetic shift; the lane kernels use the same floor so tiers agree. */
[[nodiscard]] inline constexpr int32_t
asr_floor (const int32_t value, const uint32_t shift) noexcept
{
    return value >> shift;
}

/** @brief Saturating clamp into the sample format. */
[[nodiscard]] inline constexpr int16_t
clamp_q15 (const int32_t value) noexcept
{
    return static_cast<int16_t> (value > _S_sample_max
                                     ? _S_sample_max
                                     : (value < _S_sample_min ? _S_sample_min : value));
}

/**
 * @brief Q15 x Q15 -> Q15 with round-half-up, saturating.
 *
 * The product is formed in int64 so the result is exact before the shift, and
 * the rounding term is the same one `_S_round_shift` applies everywhere else in
 * the product, which keeps a kernel and the inference core that consumes it
 * agreeing on what "multiply by a gain" means.
 */
[[nodiscard]] inline constexpr int32_t
q15_mul (const int32_t lhs, const int32_t rhs) noexcept
{
    return clamp_q15 (_S_round_shift (static_cast<int64_t> (lhs) * rhs, _S_state_frac_bits));
}

/** @brief Q15 x Q15 -> Q15 without the clamp, for an accumulator that keeps headroom. */
[[nodiscard]] inline constexpr int32_t
q15_mul_wide (const int32_t lhs, const int32_t rhs) noexcept
{
    return _S_round_shift (static_cast<int64_t> (lhs) * rhs, _S_state_frac_bits);
}

/** @brief Saturating Q15 add and difference of two Q15 values. */
[[nodiscard]] inline constexpr int32_t
q15_add_sat (const int32_t lhs, const int32_t rhs) noexcept
{
    const int64_t sum = static_cast<int64_t> (lhs) + rhs;
    return static_cast<int32_t> (sum > _S_sample_max ? _S_sample_max
                                                     : (sum < _S_sample_min ? _S_sample_min
                                                                             : sum));
}

[[nodiscard]] inline constexpr int32_t
q15_sub_sat (const int32_t lhs, const int32_t rhs) noexcept
{
    return q15_add_sat (lhs, -rhs);
}

/** @brief Q15 absolute value that cannot alias its own negative. */
[[nodiscard]] inline constexpr int32_t
q15_abs (const int32_t value) noexcept
{
    return value >= 0 ? value : -value;
}

/**
 * @brief `2*cos * state`, rounded: the resonator's coefficient product.
 *
 * `cosine_q15 * state_q7` is formed in int32, rounded at the Q15 boundary
 * exactly as `vpmulhrsw` / `vqdmulh_s16` round it, then doubled with the same
 * saturation those instructions' callers must apply.  Written as its own
 * function because the bound is what makes the format choice safe: the state
 * peaks at `_S_resonator_max_samples * |x| / 2` = 2^14, so the doubled product
 * peaks at 2^15 and the int32 intermediate at 2^30, and every tier -- one
 * instruction on a rounding-multiply target, three operations on the portable
 * path -- lands on the same int16 result.
 */
[[nodiscard]] inline constexpr int16_t
coeff_2cos_mul_q7 (const int16_t cosine_q15, const int16_t state_q7) noexcept
{
    const int32_t product = static_cast<int32_t> (cosine_q15) * state_q7;
    const int32_t rounded = (product + (1 << (_S_coeff_frac_bits - 1u)))
                            >> _S_coeff_frac_bits;
    const int32_t doubled = rounded << _S_coeff_double_shift;
    return static_cast<int16_t> (doubled > _S_sample_max
                                     ? _S_sample_max
                                     : (doubled < _S_sample_min ? _S_sample_min : doubled));
}

/** @brief One resonator step: s0 = 2*cos*s1 - s2 + x, all in Q7, saturating. */
[[nodiscard]] inline constexpr int16_t
resonator_step (const int16_t cosine_q15, const int16_t state_q7,
                const int16_t previous_q7, const int16_t input_q7) noexcept
{
    /* Saturating rather than wrapping: a resonator that overflows should clip
       its bin, not produce a state of the opposite sign two clocks later.  The
       lane kernels use the hardware saturating add/sub, so both paths agree. */
    const int32_t fed = static_cast<int32_t> (coeff_2cos_mul_q7 (cosine_q15, state_q7))
                        - previous_q7 + input_q7;
    return fed > _S_sample_max ? static_cast<int16_t> (_S_sample_max)
                               : (fed < _S_sample_min ? static_cast<int16_t> (_S_sample_min)
                                                      : static_cast<int16_t> (fed));
}

/** @brief Q15 sample to the Q7 scale a resonator accumulates at. */
[[nodiscard]] inline constexpr int16_t
to_resonator_q7 (const int32_t sample_q15) noexcept
{
    return static_cast<int16_t> (_S_round_shift (sample_q15, _S_resonator_input_shift));
}

/** @brief Q7 resonator value back to the wire format. */
[[nodiscard]] inline constexpr int32_t
from_resonator_q7 (const int32_t value_q7) noexcept
{
    return value_q7 << _S_resonator_input_shift;
}

/**
 * @brief Goertzel squared magnitude of one finished bin, in Q14.
 *
 * `power = s1^2 + s2^2 - 2*cos(theta) * s1 * s2` with s1 the newest state and
 * s2 the one before it.  Both squares are Q14 because the states are Q7, so the
 * coefficient term is formed without any rescale: `2*cos*s1` is a Q7 value and
 * multiplying it by the Q7 s2 is already Q14.  At the state bound above each
 * term peaks at 2^30, which keeps the whole expression inside int64 and the
 * non-negative result inside the uint32 the integer square root takes.
 */
[[nodiscard]] inline constexpr uint32_t
resonator_power (const int16_t state_q7, const int16_t previous_q7,
                 const int16_t cosine_q15) noexcept
{
    const int64_t direct = static_cast<int64_t> (state_q7) * state_q7
                           + static_cast<int64_t> (previous_q7) * previous_q7;
    const int64_t cross = static_cast<int64_t> (coeff_2cos_mul_q7 (cosine_q15, state_q7))
                          * previous_q7;
    const int64_t power = direct - cross;
    return power > 0 ? static_cast<uint32_t> (power) : 0u;
}

/** @brief Magnitude of a Q14 power as Q7, via the inference core's integer square root. */
[[nodiscard]] inline constexpr int32_t
magnitude_q7 (const uint32_t power_q14) noexcept
{
    return static_cast<int32_t> (zzs::mllee::_S_isqrt (power_q14));
}

/**
 * @brief Ratio of two int32 counts as Q15, saturating, zero denominator -> 0.
 *
 * Every normalized feature in the module is one of these; the semantics match
 * the training path's `ratio_q15` exactly so a feature computed on a device and
 * the same feature computed by a calibration script cannot disagree.
 */
[[nodiscard]] inline constexpr int32_t
ratio_q15 (const int64_t numerator, const int64_t denominator) noexcept
{
    if (denominator == 0)
        return 0;
    const int64_t scaled = (numerator << _S_state_frac_bits) / denominator;
    return static_cast<int32_t> (scaled > _S_sample_max
                                     ? _S_sample_max
                                     : (scaled < -_S_q15_one ? -_S_q15_one : scaled));
}

/**
 * @brief Integer log2 with a Q15 fractional part, by digit iteration.
 *
 * The exponent comes from a leading-zero count, the mantissa is normalized into
 * [1, 2) and then squared: each squaring decides one fractional bit, because
 * `log2(m) >= 2^-i` exactly when `m^(2^i) >= 2`.  It is the same restoring form
 * as the integer square root the inference core already uses, needs no table,
 * and is exact to the last bit it returns instead of being a fit -- which
 * matters because this logarithm is what turns a power into a dB feature, and a
 * fit's error would be indistinguishable from a real level change.
 *
 * The loop is fifteen int64 squarings, so it belongs at a reconfiguration or a
 * window boundary, not inside a per-sample recurrence.
 */
[[nodiscard]] inline constexpr int32_t
log2_q15 (const uint32_t value) noexcept
{
    if (value == 0u)
        return -32768; /* the documented floor, one full octave below any real level */
    const uint32_t exponent = zzs::mllee::_S_log2_u32 (value);
    /* m = value / 2^exponent in Q15, so m lands in [32768, 65536) either way. */
    uint32_t work = exponent >= _S_state_frac_bits
                        ? value >> (exponent - _S_state_frac_bits)
                        : value << (_S_state_frac_bits - exponent);
    int32_t fraction = 0;
    for (uint32_t bit = 15u; bit-- > 0u;)
        {
            /* work is Q15 in [1, 2): square it (Q30 -> Q15) and fold the octave. */
            const uint32_t squared = static_cast<uint32_t> (
                (static_cast<uint64_t> (work) * work) >> _S_state_frac_bits);
            if (squared >= (2u << _S_state_frac_bits))
                {
                    work = squared >> 1;
                    fraction |= static_cast<int32_t> (1) << bit;
                }
            else
                work = squared;
        }
    return (static_cast<int32_t> (exponent) << _S_state_frac_bits) + fraction;
}

/** @brief 10 * log10(2) in Q13, the only constant the dB path needs. */
inline constexpr int32_t _S_ten_log10_two_q13 = 24663;

/**
 * @brief Squared magnitude a full-scale Q15 sine leaves in a Q7 resonator.
 *
 * A matched tone accumulates coherently: the state at the end of the window is
 * `N * A / 2`, with A the input amplitude in Q7.  That is the kernel's 0 dB
 * point, and it is derived from the window length rather than stored, so
 * changing a window changes the level reference with it instead of silently
 * re-basing every dB feature the bank produces.
 */
[[nodiscard]] inline constexpr uint32_t
full_scale_power_q14 (const uint32_t window_samples) noexcept
{
    const int32_t amplitude_q7 = to_resonator_q7 (_S_sample_max);
    const int64_t state_q7 = (static_cast<int64_t> (amplitude_q7) * window_samples) / 2;
    return static_cast<uint32_t> (state_q7 * state_q7 * 2);
}

/**
 * @brief A window's dB zero point, precomputed.
 *
 * Both logarithms are the expensive half of the dB conversion and both depend
 * only on the window length, so a stage builds one of these at configuration
 * time and every level feature it later emits is two multiplies away.
 */
struct _S_LevelReference
{
    int32_t _M_log2_power_q15;
};

[[nodiscard]] inline constexpr _S_LevelReference
level_reference_of (const uint32_t window_samples) noexcept
{
    return _S_LevelReference{ log2_q15 (full_scale_power_q14 (window_samples)) };
}

/**
 * @brief Power in dB relative to a `_S_LevelReference`, returned in Q8.
 *
 * `10*log10(p / p0)` is `3.0103 * (log2 p - log2 p0)`; the product of a Q15
 * difference and the Q13 constant is Q28 and the narrowing to Q8 is the only
 * rounding, so a level read twice in the same window is the same integer.
 */
[[nodiscard]] inline constexpr int32_t
db_q8 (const uint32_t power_q14, const _S_LevelReference reference) noexcept
{
    if (power_q14 == 0u)
        return -32768;
    const int32_t delta = log2_q15 (power_q14) - reference._M_log2_power_q15;
    const int64_t scaled = static_cast<int64_t> (delta) * _S_ten_log10_two_q13;
    return static_cast<int32_t> (scaled >> (_S_state_frac_bits + _S_coeff_frac_bits - 8u));
}

/**
 * @name Runtime trigonometry
 *
 * A bin bank is configured by a frequency and a sample rate, both of which are
 * run-time values on a device that can be told about either by a packet.  The
 * angle is carried in the same units the inference core's rotary embedding uses
 * -- 1/128 of a turn -- because that is what makes one quarter-wave table serve
 * both paths, and `_S_sin_table` is the table it already ships.
 * @{
 */

/** @brief Angle units per turn, shared with mllee's `_S_sin_unit`. */
inline constexpr uint32_t _S_turn_units = 128u;

/** @brief Fraction bits below a turn unit, so a non-integer bin is representable. */
inline constexpr uint32_t _S_turn_unit_frac_bits = 8u;

/**
 * @brief sin(2*pi*unit/128) with `unit` carrying `_S_turn_unit_frac_bits` of
 *        fraction, linearly interpolated on the 33-entry quarter-wave table.
 *
 * The table steps a quarter wave in 32 samples, so the interpolation error is
 * bounded by the curvature of one step: `step^2 / 8` = 3.0e-4 absolute, or 6e-4
 * once doubled into a `2*cos` coefficient.  Fed through a resonator that is a
 * tuning error under a hundredth of a hertz at the kernel's sample rates, which
 * is the reason no CORDIC ladder or wider table is carried.
 */
[[nodiscard]] inline constexpr int32_t
sin_q15_fractional (uint32_t unit_fixed) noexcept
{
    /* One turn is 128 units and 256 sub-units, which is 2^15 -- a mask, not a
       modulo, and the reason the fraction width is eight and not ten. */
    constexpr uint32_t scale = 1u << _S_turn_unit_frac_bits;
    constexpr uint32_t turn_mask = (_S_turn_units << _S_turn_unit_frac_bits) - 1u;
    unit_fixed &= turn_mask;
    const uint32_t units = unit_fixed >> _S_turn_unit_frac_bits;
    const uint32_t remainder = unit_fixed & (scale - 1u);
    const uint32_t quadrant = units >> 5u;
    const uint32_t index = units & 31u;

    int32_t low;
    int32_t high;
    if (quadrant <= 1u)
        {
            /* First and second quadrant: read the table forward or mirrored. */
            low = quadrant == 0u
                      ? zzs::mllee::_S_sin_table[index]
                      : zzs::mllee::_S_sin_table[32u - index];
            high = quadrant == 0u
                       ? zzs::mllee::_S_sin_table[index + 1u]
                       : zzs::mllee::_S_sin_table[31u - index];
        }
    else
        {
            low = quadrant == 2u
                      ? -zzs::mllee::_S_sin_table[index]
                      : -zzs::mllee::_S_sin_table[32u - index];
            high = quadrant == 2u
                       ? -zzs::mllee::_S_sin_table[index + 1u]
                       : -zzs::mllee::_S_sin_table[31u - index];
        }
    return low + (static_cast<int32_t> ((high - low) * static_cast<int32_t> (remainder))
                  >> _S_turn_unit_frac_bits);
}

/** @brief cos from the same table: a quarter turn (32 units) ahead of the sine. */
[[nodiscard]] inline constexpr int32_t
cos_q15_fractional (const uint32_t unit_fixed) noexcept
{
    return sin_q15_fractional (unit_fixed + (32u << _S_turn_unit_frac_bits));
}

/**
 * @brief The `2*cos(theta)` resonator coefficient in Q15 for an angle in units.
 *
 * The stored value is the Q15 cosine, not twice it: the factor of two is
 * supplied by the doubling-high multiply that every vector tier already has, as
 * documented on `_S_coeff_double_shift`.  The name keeps `2cos` because that is
 * the constant the recursion is written with, and a caller that reads this
 * value into a formula written against `2*cos` is the bug the doc prevents.
 *
 * The quantization step is 2^-15 of 1.0, i.e. 3.1e-5 of the tuning constant.
 * That is finer than the table's own interpolation error, so the coefficient
 * costs nothing and the bin's accuracy is set by `sin_q15_fractional` alone.
 */
[[nodiscard]] inline constexpr int16_t
coeff_cos_q15 (const uint32_t unit_fixed) noexcept
{
    return static_cast<int16_t> (cos_q15_fractional (unit_fixed));
}

/**
 * @brief Coefficient for a bin at `frequency_hz / sample_rate_hz` of the sample rate.
 *
 * `theta = 2*pi*f/fs` and one turn is 128 units, so the unit count is
 * `f * 128 / fs`; the product is formed in int64 and divided once per bank
 * reconfiguration, never per sample, so the bank pays one wide divide for as
 * many bins as it likes.
 */
[[nodiscard]] inline constexpr uint32_t
unit_fixed_of_ratio (const uint32_t frequency_hz, const uint32_t sample_rate_hz) noexcept
{
    if (sample_rate_hz == 0u)
        return 0u;
    const uint64_t numerator
        = static_cast<uint64_t> (frequency_hz) * _S_turn_units << _S_turn_unit_frac_bits;
    return static_cast<uint32_t> (numerator / sample_rate_hz);
}

/** @brief Coefficient of the bin at `frequency_hz` for a channel at `sample_rate_hz`. */
[[nodiscard]] inline constexpr int16_t
coeff_cos_q15_of_hz (const uint32_t frequency_hz, const uint32_t sample_rate_hz) noexcept
{
    return coeff_cos_q15 (unit_fixed_of_ratio (frequency_hz, sample_rate_hz));
}
/** @} */

/**
 * @brief Divide a running sum by an invariant window length without a divide.
 *
 * Averaging is the one place a preprocessing loop meets a variable denominator,
 * and on a Cortex-M4 a 32-bit divide is a multi-cycle sequence.  The magic
 * reciprocal mllee already builds (`_S_div32_make`, `ceil(2^32/d)` with one
 * correction) is bit-exact, so a window mean costs a multiply-high, a multiply
 * and a compare -- the same argument the inference core makes for its softmax
 * denominator.
 */
struct _S_WindowDivide
{
    zzs::mllee::_S_Div32 _M_reciprocal;
    uint32_t _M_shift;
};

[[nodiscard]] inline constexpr _S_WindowDivide
window_divide (const uint32_t window, const uint32_t extra_shift) noexcept
{
    return _S_WindowDivide{ zzs::mllee::_S_div32_make (window == 0u ? 1u : window),
                            extra_shift };
}

/** @brief `(sum >> extra_shift) / window` for a reciprocal built above. */
[[nodiscard]] inline constexpr uint32_t
window_mean_u32 (const uint64_t sum, const _S_WindowDivide divide) noexcept
{
    const uint32_t scaled = static_cast<uint32_t> (sum >> divide._M_shift);
    return zzs::mllee::_S_div32_u (scaled, divide._M_reciprocal);
}

/** @brief Signed form of the same division, truncating toward zero. */
[[nodiscard]] inline constexpr int32_t
window_mean_i32 (const int64_t sum, const _S_WindowDivide divide) noexcept
{
    const int32_t scaled = static_cast<int32_t> (sum >> divide._M_shift);
    if (scaled >= 0)
        return static_cast<int32_t> (
            zzs::mllee::_S_div32_u (static_cast<uint32_t> (scaled), divide._M_reciprocal));
    return -static_cast<int32_t> (zzs::mllee::_S_div32_u (
        0u - static_cast<uint32_t> (scaled), divide._M_reciprocal));
}
} // namespace zzs::kernel::fx
