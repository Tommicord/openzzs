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
 * @brief LIS3MDL register map (adapted from STMicroelectronics STMems driver).
 *
 * Register addresses and bitfield definitions for the ST LIS3MDL 3-axis
 * magnetometer.  Integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::lis3mdl
{

/** @brief WHO_AM_I register address. */
inline constexpr uint8_t RegWhoAmI = 0x0F;

/** @brief Expected WHO_AM_I value for LIS3MDL. */
inline constexpr uint8_t WhoAmIValue = 0x3D;

/** @brief CTRL_REG1: ODR/Mode, temp enable, self-test. */
inline constexpr uint8_t RegCtrlReg1 = 0x20;
/* CTRL_REG1 bit fields. */
inline constexpr uint8_t CtrlReg1TempEn = 0x01;     /* bit 0  temperature sensor enable */
inline constexpr uint8_t CtrlReg1OmMask  = 0x7E;    /* bits [6:1] operating mode/ODR */
inline constexpr uint8_t CtrlReg1St      = 0x80;    /* bit 7  self-test enable */

/** @brief CTRL_REG2: full-scale, soft reset, reboot. */
inline constexpr uint8_t RegCtrlReg2 = 0x21;
/* CTRL_REG2 bit fields. */
inline constexpr uint8_t CtrlReg2FsMask   = 0x30;   /* bits [5:4] full-scale */
inline constexpr uint8_t CtrlReg2Fs4Gauss = 0x00;   /* ±4 gauss (default) */
inline constexpr uint8_t CtrlReg2Fs8Gauss = 0x10;   /* ±8 gauss */
inline constexpr uint8_t CtrlReg2Fs12Gauss = 0x20;  /* ±12 gauss */
inline constexpr uint8_t CtrlReg2Fs16Gauss = 0x30;  /* ±16 gauss */
inline constexpr uint8_t CtrlReg2SoftRst  = 0x08;   /* bit 3  software reset */
inline constexpr uint8_t CtrlReg2Reboot   = 0x10;   /* bit 4  reboot memory */

/** @brief CTRL_REG3: operating mode, SPI mode. */
inline constexpr uint8_t RegCtrlReg3 = 0x22;
/* CTRL_REG3 bit fields. */
inline constexpr uint8_t CtrlReg3MdMask   = 0x03;   /* bits [1:0] operating mode */
inline constexpr uint8_t CtrlReg3MdCont   = 0x00;   /* continuous conversion mode */
inline constexpr uint8_t CtrlReg3MdSingle = 0x01;   /* single conversion mode */
inline constexpr uint8_t CtrlReg3MdPwrDn  = 0x02;   /* power-down */
inline constexpr uint8_t CtrlReg3Sim      = 0x20;   /* bit 5 SPI serial interface mode */

/** @brief STATUS_REG: data ready flags. */
inline constexpr uint8_t RegStatusReg = 0x27;
/* STATUS_REG bit fields. */
inline constexpr uint8_t StatusXda = 0x01;   /* bit 0 X data available */
inline constexpr uint8_t StatusYda = 0x02;   /* bit 1 Y data available */
inline constexpr uint8_t StatusZda = 0x04;   /* bit 2 Z data available */

/** @brief OUT_X_L: magnetometer X axis low byte (6 bytes X/Y/Z). */
inline constexpr uint8_t RegOutXL = 0x28;

/** @brief I2C address when SA0 = 0. */
inline constexpr uint8_t I2cAddrSa0Low = 0x39;

/** @brief I2C address when SA0 = 1. */
inline constexpr uint8_t I2cAddrSa0High = 0x3D;

/** @brief Magnetometer sensitivity at ±4 gauss: 1/6842 gauss/LSB =
 *         0.146 milligauss/LSB → Q15 multiplier = 4789. */
inline constexpr int32_t MagSensQ15MilligaussPerLsb = 4789;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::lis3mdl
