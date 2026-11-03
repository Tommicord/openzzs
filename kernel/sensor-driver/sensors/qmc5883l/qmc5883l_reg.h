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
 * @brief QST QMC5883L 3-axis magnetometer register map.
 *
 * The QMC5883L is a low-cost magnetometer found in budget phones and on
 * breakout boards.  Despite the similar name it has a different register map
 * from the QMC6308.
 *
 * Register map follows the mainline Linux IIO driver
 * `drivers/iio/magnetometer/qmc5883l.c` and datasheet Rev. B:
 *
 * | Register   | Address | Contents                                   |
 * |------------|---------|--------------------------------------------|
 * | X_LSB      | 0x00    | X/Y/Z little-endian, 6 bytes               |
 * | STATUS     | 0x06    | DRDY[0], OVL[1]                            |
 * | CTRL1      | 0x09    | MODE[1:0], ODR[3:2], RNG[5:4], OSR[7:6]    |
 * | CTRL2      | 0x0A    | INT enable, soft reset bit 7               |
 * | SET/RESET  | 0x0B    | SET/RESET period, enable bit 0             |
 * | ID         | 0x0D    | 0xFF                                       |
 *
 * Registers are integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::qmc5883l {

/** @brief Chip ID register address. */
inline constexpr uint8_t RegChipId = 0x0D;

/** @brief Expected chip ID value for QMC5883L. */
inline constexpr uint8_t ChipIdValue = 0xFF;

/** @brief Data block: XL, XH, YL, YH, ZL, ZH (6 bytes). */
inline constexpr uint8_t RegDataXlsb = 0x00;

/** @brief Status register: DRDY bit 0, overflow bit 1. */
inline constexpr uint8_t RegStatus = 0x06;
inline constexpr uint8_t StatusDataReady = 0x01; /* bit 0 data ready */
inline constexpr uint8_t StatusOverflow = 0x02;  /* bit 1 output overflow */

/** @brief Control register 1: mode, ODR, range, oversampling. */
inline constexpr uint8_t RegCtrl1 = 0x09;
/* CTRL1 bit fields. */
inline constexpr uint8_t Ctrl1ModeMask = 0x03;       /* bits [1:0] mode */
inline constexpr uint8_t Ctrl1ModeStandby = 0x00;    /* standby */
inline constexpr uint8_t Ctrl1ModeContinuous = 0x01; /* continuous measurement */
inline constexpr uint8_t Ctrl1OdrMask = 0x0C;        /* bits [3:2] output data rate */
inline constexpr uint8_t Ctrl1Odr10Hz = 0x00;        /* 10 Hz  */
inline constexpr uint8_t Ctrl1Odr50Hz = 0x04;        /* 50 Hz  */
inline constexpr uint8_t Ctrl1Odr100Hz = 0x08;       /* 100 Hz */
inline constexpr uint8_t Ctrl1Odr200Hz = 0x0C;       /* 200 Hz */
inline constexpr uint8_t Ctrl1RangeMask = 0x30;      /* bits [5:4] full scale */
inline constexpr uint8_t Ctrl1Range2G = 0x00;        /* ±2 gauss  */
inline constexpr uint8_t Ctrl1Range8G = 0x10;        /* ±8 gauss  */
inline constexpr uint8_t Ctrl1OsrMask = 0xC0;        /* bits [7:6] oversampling */
inline constexpr uint8_t Ctrl1Osr512 = 0x00;         /* 512x oversampling */
inline constexpr uint8_t Ctrl1Osr256 = 0x40;         /* 256x oversampling */
inline constexpr uint8_t Ctrl1Osr128 = 0x80;         /* 128x oversampling */
inline constexpr uint8_t Ctrl1Osr64 = 0xC0;          /* 64x oversampling  */

/** @brief Control register 2: interrupt configuration / soft reset. */
inline constexpr uint8_t RegCtrl2 = 0x0A;
/* CTRL2 bit fields. */
inline constexpr uint8_t Ctrl2IntMask = 0x01;    /* bit 0 interrupt enable */
inline constexpr uint8_t Ctrl2IntEnable = 0x00;  /* interrupt pin enabled */
inline constexpr uint8_t Ctrl2IntDisable = 0x01; /* interrupt pin disabled */
inline constexpr uint8_t Ctrl2SoftReset = 0x80;  /* bit 7 soft reset */

/** @brief SET/RESET period register. */
inline constexpr uint8_t RegSetReset = 0x0B;
inline constexpr uint8_t SetResetEnable = 0x01; /* bit 0 enable SET/RESET */

/** @brief Fixed 7-bit I2C address. */
inline constexpr uint8_t I2cAddr = 0x0D;

/** @brief Number of bytes in the 3-axis data block (X/Y/Z, 2 bytes each). */
inline constexpr uint32_t DataBlockBytes = 6u;

/** @brief Magnetometer ODR: 100 Hz. */
inline constexpr uint32_t OdrHz = 100u;

/** @brief Sensitivity at ±8 gauss: 3000 LSB/gauss → 0.333333 mgauss/LSB,
 *         Q15 multiplier round(0.333333 * 32768) = 10923. */
inline constexpr int32_t MagSensQ15MilligaussPerLsb = 10923;

/** @brief Sensitivity at ±2 gauss: 12000 LSB/gauss → 0.083333 mgauss/LSB,
 *         Q15 multiplier round(0.083333 * 32768) = 2731. */
inline constexpr int32_t MagSensQ15MilligaussPerLsb2G = 2731;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::qmc5883l
