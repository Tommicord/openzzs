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
 * @brief ST L3GD20 3-axis gyroscope register map.
 *
 * The L3GD20 is a 3-axis digital gyroscope found in older phones and IMU
 * boards.  Registers are integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::l3gd20 {

/** @brief WHO_AM_I register address. */
inline constexpr uint8_t RegWhoAmI = 0x0F;

/** @brief Expected WHO_AM_I value for L3GD20. */
inline constexpr uint8_t WhoAmIValue = 0xD7;

/** @brief CTRL_REG1: ODR + power mode + axes enable. */
inline constexpr uint8_t RegCtrlReg1 = 0x20;
/* CTRL_REG1 bit fields. */
inline constexpr uint8_t CtrlReg1OdrMask = 0xF0;  /* bits [7:4] output data rate */
inline constexpr uint8_t CtrlReg1Odr95Hz = 0x00;  /* 95 Hz   */
inline constexpr uint8_t CtrlReg1Odr190Hz = 0x40; /* 190 Hz  */
inline constexpr uint8_t CtrlReg1Odr380Hz = 0x80; /* 380 Hz  */
inline constexpr uint8_t CtrlReg1Odr760Hz = 0xC0; /* 760 Hz  */
inline constexpr uint8_t CtrlReg1Xen = 0x04;      /* bit 2 X axis enable */
inline constexpr uint8_t CtrlReg1Yen = 0x02;      /* bit 1 Y axis enable */
inline constexpr uint8_t CtrlReg1Zen = 0x01;      /* bit 0 Z axis enable */

/** @brief CTRL_REG2: high-pass filter config. */
inline constexpr uint8_t RegCtrlReg2 = 0x21;

/** @brief CTRL_REG3: interrupt config. */
inline constexpr uint8_t RegCtrlReg3 = 0x22;

/** @brief CTRL_REG4: full-scale / BDU / endian. */
inline constexpr uint8_t RegCtrlReg4 = 0x23;
/* CTRL_REG4 bit fields. */
inline constexpr uint8_t CtrlReg4Bdu = 0x80;       /* bit 7 block data update */
inline constexpr uint8_t CtrlReg4FsMask = 0x30;    /* bits [5:4] full-scale */
inline constexpr uint8_t CtrlReg4Fs250dps = 0x00;  /* ±250 dps */
inline constexpr uint8_t CtrlReg4Fs500dps = 0x10;  /* ±500 dps */
inline constexpr uint8_t CtrlReg4Fs2000dps = 0x20; /* ±2000 dps */

/** @brief CTRL_REG5: reboot / high-pass filter enable. */
inline constexpr uint8_t RegCtrlReg5 = 0x24;

/** @brief STATUS_REG: data ready flags. */
inline constexpr uint8_t RegStatusReg = 0x27;
/* STATUS_REG bit fields. */
inline constexpr uint8_t StatusZyxda = 0x08; /* bit 3 X/Y/Z data available */

/** @brief OUT_X_L: gyroscope X axis low byte (6 bytes X/Y/Z). */
inline constexpr uint8_t RegOutXL = 0x28;

/** @brief I2C address when SA0 = 0. */
inline constexpr uint8_t I2cAddrSa0Low = 0x6A;

/** @brief I2C address when SA0 = 1. */
inline constexpr uint8_t I2cAddrSa0High = 0x6B;

/** @brief Gyroscope sensitivity at ±250 dps: 8.75 mdps/LSB → Q15 multiplier
 *         8.75 * 32768 / 1000 ≈ 287. */
inline constexpr int32_t GyroSensQ15MdpsPerLsb = 287;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::l3gd20
