/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
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
 * @brief AK0991x (AKM) 3-axis magnetometer register map.
 *
 * The AK0991x family (AK09911, AK09912, AK09915, AK09916, AK09918) is one of
 * the most common magnetometers found in Android phones (Samsung Galaxy,
 * Google Pixel, Sony Xperia, and many others).  Registers are integer-only;
 * no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::ak0991x {

/** @brief WHO_AM_I / WIA register address. */
inline constexpr uint8_t RegWia = 0x00;

/** @brief Expected WHO_AM_I value for AK0991x family. */
inline constexpr uint8_t WiaValue = 0x48;

/** @brief ST1: data ready status. Bit 0 = DRDY. */
inline constexpr uint8_t RegSt1 = 0x10;

/** @brief Data block: HXL, HXH, HYL, HYH, HZL, HZH (6 bytes). */
inline constexpr uint8_t RegHxl = 0x11;

/** @brief ST2: data overflow / lock status. */
inline constexpr uint8_t RegSt2 = 0x18;

/** @brief CNTL2: measurement mode control. */
inline constexpr uint8_t RegCntl2 = 0x1B;
/* CNTL2 bit fields. */
inline constexpr uint8_t Cntl2ModeCont100Hz = 0x10; /* continuous mode 100 Hz */
inline constexpr uint8_t Cntl2ModeCont200Hz = 0x20; /* continuous mode 200 Hz */

/** @brief CNTL3: soft reset. Bit 0 = SRST. */
inline constexpr uint8_t RegCntl3 = 0x1C;
inline constexpr uint8_t Cntl3SoftReset = 0x01;

/** @brief I2C address (fixed 7-bit, AK0991x uses 0x0C). */
inline constexpr uint8_t I2cAddr = 0x0C;

/** @brief Number of bytes in the 3-axis data block (X/Y/Z, 2 bytes each). */
inline constexpr uint32_t DataBlockBytes = 6u;

/** @brief Magnetometer ODR: 100 Hz. */
inline constexpr uint32_t OdrHz = 100u;

/** @brief Sensitivity: AK0991x output is 16-bit, full scale ±4900 µT.
 *  Sensitivity ≈ 0.15 µT/LSB → Q15 multiplier ≈ 4915 (0.15 * 32768 / 1000
 *  scaled to milligauss: 1.5 mG/LSB → Q15 multiplier 49152). */
inline constexpr int32_t MagSensQ15MilligaussPerLsb = 49152;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::ak0991x
