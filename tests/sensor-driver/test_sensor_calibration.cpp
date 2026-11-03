/*
 * Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

/**
 * @file test_sensor_calibration.cpp
 * @brief Unit tests for sensor calibration (raw-to-physical, temperature,
 *        bias accumulation).
 */

#include "kernel/sensor-driver/fixed/fixed_conversions.hpp"
#include "kernel/sensor-driver/fixed/sensor_calibration.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace {
using namespace zzs::kernel::sensor::driver::fixed;
} // namespace

TEST_CASE ("Accelerometer raw-to-mg conversion (Q15 sensitivity)", "[sensor-driver][calibration][accel]") {
    /* 1 g LSB at ±2 g full scale: raw=32768 -> ~2000 mg (no saturation at Q15). */
    REQUIRE (accel_raw_to_mg (_S_q15_one) == 1999);

    /* Small raw value stays in range: raw=100 -> 100 * 1999 >> 15 = 6 mg. */
    REQUIRE (accel_raw_to_mg (100) == 6);

    /* Zero raw -> zero mg. */
    REQUIRE (accel_raw_to_mg (0) == 0);

    /* Negative raw -> negative mg. */
    REQUIRE (accel_raw_to_mg (-100) == -6);

    /* Max negative raw saturates. */
    REQUIRE (accel_raw_to_mg (_S_q15_min) == -1999);
}

TEST_CASE ("Gyroscope raw-to-mdps conversion (Q15 sensitivity)", "[sensor-driver][calibration][gyro]") {
    /* raw=100 -> 100 * 286720 >> 15 = 875 mdps. */
    REQUIRE (gyro_raw_to_mdps (100) == 875);

    /* Zero raw -> zero mdps. */
    REQUIRE (gyro_raw_to_mdps (0) == 0);

    /* Max raw saturates at Q15_MAX. */
    REQUIRE (gyro_raw_to_mdps (_S_q15_max) == _S_q15_max);
}

TEST_CASE ("Magnetometer raw-to-milligauss conversion (Q15 sensitivity)", "[sensor-driver][calibration][mag]") {
    /* raw=100 -> 100 * 22412 >> 15 = 68 milligauss. */
    REQUIRE (mag_raw_to_milligauss (100) == 68);

    /* Zero raw -> zero milligauss. */
    REQUIRE (mag_raw_to_milligauss (0) == 0);

    /* Mag sensitivity is moderate; max raw does not saturate. */
    REQUIRE (mag_raw_to_milligauss (_S_q15_max) == 22411);
}

TEST_CASE ("Bias accumulation and mean computation (Q31 accumulator)", "[sensor-driver][calibration][bias]") {
    _S_BiasAccumulator acc = bias_reset ();
    REQUIRE (bias_mean_q15 (acc) == 0);
    REQUIRE (bias_mean_q31 (acc) == 0);

    /* Accumulate a known window of Q15 samples (count is a power of two so the
     * shift-based mean is exact). Mean of 1000, 2000 = 1500 in Q15. */
    acc = bias_accumulate (acc, 1000);
    acc = bias_accumulate (acc, 2000);
    /* Mean of 1000, 2000 = 1500 in Q15. */
    REQUIRE (bias_mean_q15 (acc) == 1500);

    /* Q31 mean matches. */
    REQUIRE (bias_mean_q31 (acc) == static_cast<q31_t> (1500) << 16);
}

TEST_CASE ("Bias accumulation of max values does not saturate Q31", "[sensor-driver][calibration][bias][edge]") {
    /* A single max sample gives the exact max mean (count == 1). */
    _S_BiasAccumulator acc = bias_reset ();
    acc = bias_accumulate (acc, _S_q15_max);
    REQUIRE (acc._M_count == 1u);
    REQUIRE (bias_mean_q15 (acc) == _S_q15_max);
}

TEST_CASE ("Temperature compensation (Q11 coefficients)", "[sensor-driver][calibration][temp]") {
    /* offset=0, coeff=0.002 Q11, temp_delta=10 deg C Q11 -> small positive. */
    const q11_t temp_delta_q11 = static_cast<q11_t> (10) << _S_q11_frac_bits;
    const q15_t comp = temperature_compensate_q15 (0, _S_accel_temp_coeff_q11[0], temp_delta_q11);
    /* Q11 coeff for 0.002 is 4; 0.002 * 10 deg -> Q15 = 640. */
    REQUIRE (comp == 640);

    /* Zero delta leaves offset unchanged. */
    REQUIRE (temperature_compensate_q15 (1000, _S_accel_temp_coeff_q11[0], 0) == 1000);

    /* Zero offset with zero delta is zero. */
    REQUIRE (temperature_compensate_q15 (0, _S_accel_temp_coeff_q11[0], 0) == 0);
}

TEST_CASE ("Edge cases: zero raw values and max raw values", "[sensor-driver][calibration][edge]") {
    REQUIRE (accel_raw_to_mg (0) == 0);
    REQUIRE (gyro_raw_to_mdps (0) == 0);
    REQUIRE (mag_raw_to_milligauss (0) == 0);

    /* Accel sensitivity is small enough that max raw does not saturate. */
    REQUIRE (accel_raw_to_mg (_S_q15_max) == 1999);
    /* Gyro sensitivity is large enough that max raw saturates at Q15_MAX. */
    REQUIRE (gyro_raw_to_mdps (_S_q15_max) == _S_q15_max);
    /* Mag sensitivity is moderate; max raw yields 22411 milligauss. */
    REQUIRE (mag_raw_to_milligauss (_S_q15_max) == 22411);

    REQUIRE (accel_raw_to_mg (_S_q15_min) == -1999);
    REQUIRE (gyro_raw_to_mdps (_S_q15_min) == _S_q15_min);
    /* Mag sensitivity is moderate; min raw yields -22412 milligauss. */
    REQUIRE (mag_raw_to_milligauss (_S_q15_min) == -22412);
}
