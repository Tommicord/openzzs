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

/**
 * @brief Sensor-specific fixed-point calibration.
 *
 * Converts raw Q15 sensor samples into physical units (accelerometer mg,
 * gyroscope mdps, magnetometer gauss) with Q15 sensitivity factors, applies
 * Q11 temperature compensation coefficients, and accumulates Q31 bias offsets
 * for zero-rate calibration.  Every value is integer-only; sensitivity factors
 * are Q15 multipliers so that `physical = q15_mul (raw_q15, sensitivity_q15)`.
 * @{
 */
namespace zzs::kernel::sensor::driver::fixed {

/**
 * @name Sensitivity conversions (Q15 multiplier per LSB).
 *
 * Stored as Q15 so that `q15_mul (raw_q15, sensitivity_q15)` yields the
 * physical value in the documented unit.  Values below are typical MEMS
 * full-scale figures; a platform overrides them at calibration time.
 * @{
 */

/** @brief Accelerometer sensitivity: 0.061 mg per LSB (±2 g full scale),
 *         as a Q15 multiplier: round(0.061 * 32768) = 1999.
 *
 * `accel_mg = q15_mul (raw_q15, _S_accel_sensitivity_q15_mg_per_lsb)`. */
inline constexpr q15_t _S_accel_sensitivity_q15_mg_per_lsb = 1999;

/** @brief Gyroscope sensitivity: 8.75 mdps per LSB (±250 dps full scale),
 *         as a Q15 multiplier: round(8.75 * 32768) = 286720.
 *
 * `gyro_mdps = q15_mul (raw_q15, _S_gyro_sensitivity_q15_mdps_per_lsb)`. */
inline constexpr q15_t _S_gyro_sensitivity_q15_mdps_per_lsb = 286720;

/** @brief Magnetometer sensitivity: 0.684 milli-gauss per LSB (±4 gauss
 *         full scale), as a Q15 multiplier: round(0.684 * 32768) = 22412.
 *
 * `mag_milligauss = q15_mul (raw_q15, _S_mag_sensitivity_q15_milligauss_per_lsb)`. */
inline constexpr q15_t _S_mag_sensitivity_q15_milligauss_per_lsb = 22412;

/** @} */

/**
 * @name Raw-to-physical conversions.
 *
 * Each converts a Q15 raw sample into physical units using the matching
 * Q15 sensitivity.  All saturating; results carry the documented unit.
 * @{
 */

/** @brief Convert a Q15 accelerometer sample to milli-g. */
[[nodiscard]] inline constexpr q15_t
accel_raw_to_mg (const q15_t raw_q15) noexcept {
  return q15_mul (raw_q15, _S_accel_sensitivity_q15_mg_per_lsb);
}

/** @brief Convert a Q15 gyroscope sample to milli-degrees per second. */
[[nodiscard]] inline constexpr q15_t
gyro_raw_to_mdps (const q15_t raw_q15) noexcept {
  return q15_mul (raw_q15, _S_gyro_sensitivity_q15_mdps_per_lsb);
}

/** @brief Convert a Q15 magnetometer sample to milli-gauss. */
[[nodiscard]] inline constexpr q15_t
mag_raw_to_milligauss (const q15_t raw_q15) noexcept {
  return q15_mul (raw_q15, _S_mag_sensitivity_q15_milligauss_per_lsb);
}

/** @} */

/**
 * @name Q11 temperature compensation coefficients.
 *
 * Temperature compensation is applied as a Q11 coefficient multiplied against
 * a temperature delta (itself Q11) and added to the zero-rate offset.  Values
 * are per-axis, indexed the same way the sensor channels are.
 * @{
 */

/** @brief Temperature coefficient for accelerometer zero offset, Q11 per
 *         degree C. */
inline constexpr q11_t _S_accel_temp_coeff_q11[3] = {
    4, /* X: 2 mg per deg C (round(0.002 * 2048) = 4) */
    4, /* Y */
    4, /* Z */
};

/** @brief Temperature coefficient for gyroscope zero offset, Q11 per degree C. */
inline constexpr q11_t _S_gyro_temp_coeff_q11[3] = {
    2, /* X: 1 mdps per deg C (round(0.001 * 2048) = 2) */
    2, /* Y */
    2, /* Z */
};

/** @brief Temperature coefficient for magnetometer zero offset, Q11 per degree C. */
inline constexpr q11_t _S_mag_temp_coeff_q11[3] = {
    1, /* X (round(0.0005 * 2048) = 1) */
    1, /* Y */
    1, /* Z */
};

/** @brief Apply Q11 temperature compensation to a Q15 zero-rate offset.
 *
 * `compensated_q15 = offset_q15 + q11_to_q15 (q11_mul (coeff_q11, temp_delta_q11))`. */
[[nodiscard]] inline constexpr q15_t
temperature_compensate_q15 (const q15_t offset_q15, const q11_t coeff_q11, const q11_t temp_delta_q11) noexcept {
  return q15_add (offset_q15, q11_to_q15 (q11_mul (coeff_q11, temp_delta_q11)));
}

/** @} */

/**
 * @brief Q31 bias accumulator for offset calibration.
 *
 * Holds a running Q31 sum of Q15 samples plus a sample count, so a stage can
 * accumulate a window of samples and then divide to obtain a Q31 bias without
 * ever saturating at the Q15 wire width.  The count is bounded by construction
 * (a calibration window fits in uint32), and `mean_q31` saturates on the
 * narrowing to Q15.
 * @{
 */
struct _S_BiasAccumulator {
  q31_t _M_sum_q31 = 0;
  uint32_t _M_count = 0;
};

/** @brief Add one Q15 sample to the bias accumulator (Q31 promotion, no
 *         saturation loss on the way in). */
[[nodiscard]] inline constexpr _S_BiasAccumulator
bias_accumulate (_S_BiasAccumulator acc, const q15_t sample_q15) noexcept {
  acc._M_sum_q31 = q31_add (acc._M_sum_q31, q31_from_q15 (sample_q15));
  acc._M_count += 1u;
  return acc;
}

/** @brief Reset the accumulator to an empty state. */
[[nodiscard]] inline constexpr _S_BiasAccumulator
bias_reset (void) noexcept {
  return _S_BiasAccumulator{};
}

/** @brief Return the accumulated mean bias as a Q15 value, saturating.
 *
 * The mean is computed in Q31 (`sum / count`, shift aligned) and then narrowed
 * to Q15 with saturation.  An empty accumulator yields zero. */
[[nodiscard]] inline constexpr q15_t
bias_mean_q15 (const _S_BiasAccumulator &acc) noexcept {
  if (acc._M_count == 0u)
    return 0;
  /* Mean in Q31: align sum down by the count's log2, then divide the
     remainder so no precision is lost before the Q15 narrowing. */
  const uint32_t shift = zzs::mllee::_S_log2_u32 (acc._M_count);
  const int64_t aligned = acc._M_sum_q31 >> shift;
  const int64_t remainder = acc._M_sum_q31 - (aligned << shift);
  const int64_t mean_q31 = aligned + (remainder / static_cast<int64_t> (acc._M_count));
  return q31_to_q15 (mean_q31);
}

/** @brief Return the accumulated mean bias as a Q31 value. */
[[nodiscard]] inline constexpr q31_t
bias_mean_q31 (const _S_BiasAccumulator &acc) noexcept {
  if (acc._M_count == 0u)
    return 0;
  const uint32_t shift = zzs::mllee::_S_log2_u32 (acc._M_count);
  const int64_t aligned = acc._M_sum_q31 >> shift;
  const int64_t remainder = acc._M_sum_q31 - (aligned << shift);
  return aligned + (remainder / static_cast<int64_t> (acc._M_count));
}

} // namespace zzs::kernel::sensor::driver::fixed
