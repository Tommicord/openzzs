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
 * @brief ST LIS2DH12 3-axis accelerometer register map.
 *
 * The LIS2DH12 is a low-power 3-axis accelerometer found in many phones.
 * Registers are integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::lis2dh12 {

/** @brief WHO_AM_I register address. */
inline constexpr uint8_t RegWhoAmI = 0x0F;

/** @brief Expected WHO_AM_I value for LIS2DH12. */
inline constexpr uint8_t WhoAmIValue = 0x33;

/** @brief CTRL_REG1: ODR + power mode. */
inline constexpr uint8_t RegCtrlReg1 = 0x20;
/* CTRL_REG1 bit fields. */
inline constexpr uint8_t CtrlReg1OdrMask = 0xF0;  /* bits [7:4] ODR */
inline constexpr uint8_t CtrlReg1Odr100Hz = 0x50; /* 100 Hz normal mode */
inline constexpr uint8_t CtrlReg1Odr200Hz = 0x60; /* 200 Hz normal mode */
inline constexpr uint8_t CtrlReg1Odr400Hz = 0x70; /* 400 Hz normal mode */
inline constexpr uint8_t CtrlReg1Lpen = 0x08;     /* bit 3 low-power enable */

/** @brief CTRL_REG4: full-scale, BDU. */
inline constexpr uint8_t RegCtrlReg4 = 0x23;
/* CTRL_REG4 bit fields. */
inline constexpr uint8_t CtrlReg4Bdu = 0x80;    /* bit 7 block data update */
inline constexpr uint8_t CtrlReg4FsMask = 0x30; /* bits [5:4] full-scale */
inline constexpr uint8_t CtrlReg4Fs2g = 0x00;   /* ±2 g */
inline constexpr uint8_t CtrlReg4Fs4g = 0x10;   /* ±4 g */
inline constexpr uint8_t CtrlReg4Fs8g = 0x20;   /* ±8 g */
inline constexpr uint8_t CtrlReg4Fs16g = 0x30;  /* ±16 g */

/** @brief CTRL_REG5: self-test enable. */
inline constexpr uint8_t RegCtrlReg5 = 0x24;
/* CTRL_REG5 bit fields. */
inline constexpr uint8_t CtrlReg5St = 0x02; /* bit 1 self-test enable */

/** @brief STATUS_REG: data ready flags. */
inline constexpr uint8_t RegStatusReg = 0x27;
inline constexpr uint8_t StatusZyxda = 0x08; /* bit 3 X/Y/Z data available */

/** @brief OUT_X_L: accelerometer X axis low byte (6 bytes X/Y/Z). */
inline constexpr uint8_t RegOutXL = 0x28;

/** @brief I2C address when SA0 = 0. */
inline constexpr uint8_t I2cAddrSa0Low = 0x18;

/** @brief I2C address when SA0 = 1. */
inline constexpr uint8_t I2cAddrSa0High = 0x19;

/** @brief Accelerometer sensitivity at ±2 g: 0.061 mg/LSB → Q15 = 1999. */
inline constexpr int32_t AccelSensQ15MgPerLsb = 1999;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::lis2dh12
