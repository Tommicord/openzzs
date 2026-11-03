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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

#pragma once

#include <cstdint>

/**
 * @brief Bosch BMI160 6-axis IMU register map (gyroscope portion).
 *
 * The BMI160 is a 3D accelerometer + 3D gyroscope found in many phones.
 * Registers are integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::bmi160g {

/** @brief CHIP_ID register address. */
inline constexpr uint8_t RegChipId = 0x00;

/** @brief Expected CHIP_ID value for BMI160. */
inline constexpr uint8_t ChipIdValue = 0xD1;

/** @brief PMU_CMD: power mode / reset command. */
inline constexpr uint8_t RegPmuCmd = 0x7E;
/* PMU_CMD bit fields. */
inline constexpr uint8_t PmuCmdSoftReset = 0xB6;  /* soft reset value */
inline constexpr uint8_t PmuCmdGyroNormal = 0x11; /* gyro normal mode */

/** @brief GYRO_CONF: gyroscope ODR / bandwidth. */
inline constexpr uint8_t RegGyroConf = 0x42;
/* GYRO_CONF bit fields. */
inline constexpr uint8_t GyroConfOdrMask = 0x0F;  /* bits [3:0] ODR */
inline constexpr uint8_t GyroConfOdr100Hz = 0x07; /* 100 Hz */
inline constexpr uint8_t GyroConfOdr200Hz = 0x08; /* 200 Hz */

/** @brief GYRO_RANGE: gyroscope full-scale range. */
inline constexpr uint8_t RegGyroRange = 0x43;
/* GYRO_RANGE bit fields. */
inline constexpr uint8_t GyroRangeMask = 0x0F;    /* bits [3:0] full-scale */
inline constexpr uint8_t GyroRange2000dps = 0x00; /* ±2000 dps */
inline constexpr uint8_t GyroRange1000dps = 0x01; /* ±1000 dps */
inline constexpr uint8_t GyroRange500dps = 0x02;  /* ±500 dps */
inline constexpr uint8_t GyroRange250dps = 0x03;  /* ±250 dps */
inline constexpr uint8_t GyroRange125dps = 0x04;  /* ±125 dps */

/** @brief INT_EN_0: interrupt enable. */
inline constexpr uint8_t RegIntEn0 = 0x54;

/** @brief INT_STATUS_0: interrupt status. */
inline constexpr uint8_t RegIntStatus0 = 0x5A;

/** @brief GYRO_X: gyroscope X axis low byte (6 bytes X/Y/Z). */
inline constexpr uint8_t RegGyroX = 0x0C;

/** @brief I2C address when SDO = 0. */
inline constexpr uint8_t I2cAddrSdoLow = 0x68;

/** @brief I2C address when SDO = 1. */
inline constexpr uint8_t I2cAddrSdoHigh = 0x69;

/** @brief Gyroscope sensitivity at ±2000 dps: 16.4 LSB/(dps) → 0.061 mdps/LSB
 *         → Q15 multiplier ≈ 2000. */
inline constexpr int32_t GyroSensQ15MdpsPerLsb = 2000;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::bmi160g
