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
 * @brief AKM AK8963 3-axis magnetometer register map.
 *
 * The AK8963 is an older 3-axis magnetometer found in many early Android
 * phones (e.g. MPU-9250 companion).  Registers are integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::ak8963 {

/** @brief WHO_AM_I / WIA register address. */
inline constexpr uint8_t RegWia = 0x00;

/** @brief Expected WHO_AM_I value for AK8963. */
inline constexpr uint8_t WiaValue = 0x48;

/** @brief ST1: data ready status. Bit 0 = DRDY. */
inline constexpr uint8_t RegSt1 = 0x02;

/** @brief Data block: HXL, HXH, HYL, HYH, HZL, HZH (6 bytes). */
inline constexpr uint8_t RegHxl = 0x03;

/** @brief ST2: data overflow / lock status. */
inline constexpr uint8_t RegSt2 = 0x09;

/** @brief CNTL1: mode control. */
inline constexpr uint8_t RegCntl1 = 0x0A;
/* CNTL1 bit fields. */
inline constexpr uint8_t Cntl1ModePowerDown = 0x00; /* power-down */
inline constexpr uint8_t Cntl1ModeSingle = 0x01;    /* single measurement */
inline constexpr uint8_t Cntl1ModeCont8Hz = 0x02;   /* continuous 8 Hz  */
inline constexpr uint8_t Cntl1ModeCont100Hz = 0x06; /* continuous 100 Hz */
inline constexpr uint8_t Cntl1Fs14bit = 0x00;       /* 14-bit output    */
inline constexpr uint8_t Cntl1Fs16bit = 0x10;       /* 16-bit output    */

/** @brief CNTL2: soft reset / self-test. */
inline constexpr uint8_t RegCntl2 = 0x0B;
inline constexpr uint8_t Cntl2SoftReset = 0x01; /* bit 0 soft reset */

/** @brief I2C address (fixed 7-bit, AK8963 uses 0x0C). */
inline constexpr uint8_t I2cAddr = 0x0C;

/** @brief Number of bytes in the 3-axis data block (X/Y/Z, 2 bytes each). */
inline constexpr uint32_t DataBlockBytes = 6u;

/** @brief Magnetometer ODR: 100 Hz. */
inline constexpr uint32_t OdrHz = 100u;

/** @brief Sensitivity: AK8963 full scale ±4900 µT, 16-bit → 0.15 µT/LSB →
 *         1.5 mG/LSB → Q15 multiplier ≈ 49152. */
inline constexpr int32_t MagSensQ15MilligaussPerLsb = 49152;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::ak8963
