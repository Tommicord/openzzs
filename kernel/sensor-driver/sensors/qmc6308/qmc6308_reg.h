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
 * @brief QST QMC6308 3-axis magnetometer register map.
 *
 * The QMC6308 is a common magnetometer in budget phones (Infinix, Tecno,
 * itel).  The connected Infinix X6885 uses this sensor at 0x0D.
 *
 * Register map follows the mainline Linux IIO driver
 * `drivers/iio/magnetometer/qmc6308.c` and datasheet Rev. F:
 *
 * | Register | Address | Contents                                |
 * |----------|---------|-----------------------------------------|
 * | ID       | 0x00    | 0x80                                    |
 * | X_LSB    | 0x01    | X/Y/Z little-endian, 6 bytes            |
 * | STATUS   | 0x09    | DRDY[0], OVFL[1]                        |
 * | CTRL1    | 0x0A    | MODE[1:0], ODR[3:2], OSR1[5:4], OSR2[7:6]|
 * | CTRL2    | 0x0B    | SET/RESET[1:0], RNG[3:2], ST[6], RST[7]  |
 * | CTRL3    | 0x0D    | reserved                                |
 * | CTRL4    | 0x29    | reserved                                |
 *
 * Note that CTRL2 holds the range and the soft reset, not CTRL1.  Registers
 * are integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::qmc6308 {

/** @brief Chip ID register address. */
inline constexpr uint8_t RegChipId = 0x00;

/** @brief Expected chip ID value for QMC6308. */
inline constexpr uint8_t ChipIdValue = 0x80;

/** @brief Data block: XL, XH, YL, YH, ZL, ZH (6 bytes). */
inline constexpr uint8_t RegDataXlsb = 0x01;

/** @brief Status register: DRDY bit 0, overflow bit 1. */
inline constexpr uint8_t RegStatus = 0x09;
inline constexpr uint8_t StatusDataReady = 0x01; /* bit 0 data ready */
inline constexpr uint8_t StatusOverflow = 0x02;  /* bit 1 output overflow */

/** @brief Control register 1: mode, ODR, oversampling. */
inline constexpr uint8_t RegCtrl1 = 0x0A;
/* CTRL1 bit fields. */
inline constexpr uint8_t Ctrl1ModeMask = 0x03;    /* bits [1:0] power mode */
inline constexpr uint8_t Ctrl1ModeSuspend = 0x00; /* suspend */
inline constexpr uint8_t Ctrl1ModeNormal = 0x01;  /* continuous measurement */
inline constexpr uint8_t Ctrl1OdrMask = 0x0C;     /* bits [3:2] output data rate */
inline constexpr uint8_t Ctrl1Odr10Hz = 0x00;     /* 10 Hz   */
inline constexpr uint8_t Ctrl1Odr50Hz = 0x04;     /* 50 Hz   */
inline constexpr uint8_t Ctrl1Odr100Hz = 0x08;    /* 100 Hz  */
inline constexpr uint8_t Ctrl1Odr200Hz = 0x0C;    /* 200 Hz  */
inline constexpr uint8_t Ctrl1Osr1Mask = 0x30;    /* bits [5:4] OSR1 */
inline constexpr uint8_t Ctrl1Osr1_8 = 0x00;      /* 8x oversampling */
inline constexpr uint8_t Ctrl1Osr1_4 = 0x10;      /* 4x oversampling */
inline constexpr uint8_t Ctrl1Osr1_2 = 0x20;      /* 2x oversampling */
inline constexpr uint8_t Ctrl1Osr1_1 = 0x30;      /* 1x oversampling */
inline constexpr uint8_t Ctrl1Osr2Mask = 0xC0;    /* bits [7:6] OSR2 */
inline constexpr uint8_t Ctrl1Osr2_8 = 0x00;      /* 8x oversampling */
inline constexpr uint8_t Ctrl1Osr2_4 = 0x40;      /* 4x oversampling */
inline constexpr uint8_t Ctrl1Osr2_2 = 0x80;      /* 2x oversampling */
inline constexpr uint8_t Ctrl1Osr2_1 = 0xC0;      /* 1x oversampling */

/** @brief Control register 2: SET/RESET, range, self-test, soft reset. */
inline constexpr uint8_t RegCtrl2 = 0x0B;
/* CTRL2 bit fields. */
inline constexpr uint8_t Ctrl2SetResetMask = 0x03; /* bits [1:0] SET/RESET */
inline constexpr uint8_t Ctrl2SetResetOn = 0x00;   /* SET/RESET enabled */
inline constexpr uint8_t Ctrl2RangeMask = 0x0C;    /* bits [3:2] full scale */
inline constexpr uint8_t Ctrl2Range30G = 0x00;     /* ±30 gauss */
inline constexpr uint8_t Ctrl2Range12G = 0x04;     /* ±12 gauss */
inline constexpr uint8_t Ctrl2Range8G = 0x08;      /* ±8 gauss */
inline constexpr uint8_t Ctrl2Range2G = 0x0C;      /* ±2 gauss */
inline constexpr uint8_t Ctrl2SelfTest = 0x40;     /* bit 6 self test */
inline constexpr uint8_t Ctrl2SoftReset = 0x80;    /* bit 7 soft reset */

/** @brief Control register 3 (reserved). */
inline constexpr uint8_t RegCtrl3 = 0x0D;

/** @brief Control register 4 (reserved). */
inline constexpr uint8_t RegCtrl4 = 0x29;

/** @brief Fixed 7-bit I2C address. */
inline constexpr uint8_t I2cAddr = 0x0D;

/** @brief Number of bytes in the 3-axis data block (X/Y/Z, 2 bytes each). */
inline constexpr uint32_t DataBlockBytes = 6u;

/** @brief Magnetometer ODR: 100 Hz. */
inline constexpr uint32_t OdrHz = 100u;

/** @brief Sensitivity at ±8 gauss: 3750 LSB/gauss → 0.266667 mgauss/LSB,
 *         Q15 multiplier round(0.266667 * 32768) = 8738. */
inline constexpr int32_t MagSensQ15MilligaussPerLsb = 8738;

/** @brief Sensitivity at ±2 gauss: 15000 LSB/gauss → 0.066667 mgauss/LSB. */
inline constexpr int32_t MagSensQ15MilligaussPerLsb2G = 2185;

/** @brief Sensitivity at ±12 gauss: 2500 LSB/gauss → 0.4 mgauss/LSB. */
inline constexpr int32_t MagSensQ15MilligaussPerLsb12G = 13107;

/** @brief Sensitivity at ±30 gauss: 1000 LSB/gauss → 1.0 mgauss/LSB (Q15
 *         saturates at 32767 rather than 1.0 * 32768). */
inline constexpr int32_t MagSensQ15MilligaussPerLsb30G = 32767;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::qmc6308
