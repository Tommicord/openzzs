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
 * @brief ADXL345 3-axis accelerometer register map.
 *
 * The ADXL345 is a low-power 3-axis accelerometer found in many phones and
 * tablets.  Registers are integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::adxl345 {

/** @brief DEVID register address. */
inline constexpr uint8_t RegDevid = 0x00;

/** @brief Expected DEVID value for ADXL345. */
inline constexpr uint8_t DevidValue = 0xE5;

/** @brief BW_RATE: power mode and output data rate. */
inline constexpr uint8_t RegBwRate = 0x2C;
/* BW_RATE bit fields. */
inline constexpr uint8_t BwRateOdrMask = 0x0F;  /* bits [3:0] output data rate */
inline constexpr uint8_t BwRateOdr100Hz = 0x0A; /* 100 Hz */
inline constexpr uint8_t BwRateOdr200Hz = 0x0B; /* 200 Hz */

/** @brief POWER_CTL: power-saving features control. */
inline constexpr uint8_t RegPowerCtl = 0x2D;
/* POWER_CTL bit fields. */
inline constexpr uint8_t PowerCtlMeasure = 0x08; /* bit 3 measurement mode */

/** @brief DATA_FORMAT: data format / self-test / full resolution / range. */
inline constexpr uint8_t RegDataFormat = 0x31;
/* DATA_FORMAT bit fields. */
inline constexpr uint8_t DataFormatFsMask = 0x03;    /* bits [1:0] full-scale range */
inline constexpr uint8_t DataFormatFs2g = 0x00;      /* ±2 g */
inline constexpr uint8_t DataFormatFs4g = 0x01;      /* ±4 g */
inline constexpr uint8_t DataFormatFs8g = 0x02;      /* ±8 g */
inline constexpr uint8_t DataFormatFs16g = 0x03;     /* ±16 g */
inline constexpr uint8_t DataFormatJustify = 0x04;   /* bit 2 MSB vs LSB justified */
inline constexpr uint8_t DataFormatFullRes = 0x08;   /* bit 3 full resolution mode */
inline constexpr uint8_t DataFormatIntInvert = 0x20; /* bit 5 interrupt polarity */

/** @brief DATAX0: accelerometer X axis low byte (6 bytes X/Y/Z). */
inline constexpr uint8_t RegDataX0 = 0x32;

/** @brief I2C address when CS/ALT ADDRESS pin = 0. */
inline constexpr uint8_t I2cAddrAltLow = 0x1A;

/** @brief I2C address when CS/ALT ADDRESS pin = 1. */
inline constexpr uint8_t I2cAddrAltHigh = 0x53;

/** @brief Accelerometer sensitivity at ±2 g full resolution: 3.9 mg/LSB → Q15 =
 *         3.9 * 32768 / 1000 = 127.8 ≈ 128 mg/LSB?  Actually 3.9 mg/LSB =
 *         Q15 multiplier 128 (milligauss units already). */
inline constexpr int32_t AccelSensQ15MgPerLsb = 128;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::adxl345
