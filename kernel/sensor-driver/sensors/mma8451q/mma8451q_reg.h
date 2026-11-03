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
 * @brief NXP MMA8451Q 3-axis accelerometer register map.
 *
 * The MMA8451Q is a low-power 3-axis accelerometer found in many phones.
 * Registers are integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::mma8451q {

/** @brief WHO_AM_I register address. */
inline constexpr uint8_t RegWhoAmI = 0x0D;

/** @brief Expected WHO_AM_I value for MMA8451Q. */
inline constexpr uint8_t WhoAmIValue = 0x1A;

/** @brief CTRL_REG1: ODR + active mode. */
inline constexpr uint8_t RegCtrlReg1 = 0x2A;
/* CTRL_REG1 bit fields. */
inline constexpr uint8_t CtrlReg1OdrMask = 0x07;  /* bits [2:0] output data rate */
inline constexpr uint8_t CtrlReg1Odr100Hz = 0x01; /* 100 Hz */
inline constexpr uint8_t CtrlReg1Odr200Hz = 0x02; /* 200 Hz */
inline constexpr uint8_t CtrlReg1Active = 0x01;   /* bit 0 active mode (after ODR) */

/** @brief CTRL_REG2: self-test / reset. */
inline constexpr uint8_t RegCtrlReg2 = 0x2B;
/* CTRL_REG2 bit fields. */
inline constexpr uint8_t CtrlReg2St = 0x40;  /* bit 6 self-test enable */
inline constexpr uint8_t CtrlReg2Rst = 0x40; /* bit 6 software reset */

/** @brief CTRL_REG3: interrupt config. */
inline constexpr uint8_t RegCtrlReg3 = 0x2C;

/** @brief CTRL_REG4: interrupt enable. */
inline constexpr uint8_t RegCtrlReg4 = 0x2D;

/** @brief CTRL_REG5: interrupt config. */
inline constexpr uint8_t RegCtrlReg5 = 0x2E;

/** @brief STATUS: data ready / status flags. */
inline constexpr uint8_t RegStatus = 0x00;
/* STATUS bit fields. */
inline constexpr uint8_t StatusZyxdr = 0x08; /* bit 3 X/Y/Z data ready */

/** @brief OUT_X_MSB: accelerometer X axis (6 bytes X/Y/Z, 14-bit left-aligned). */
inline constexpr uint8_t RegOutXMsb = 0x01;

/** @brief I2C address when SA0 = 0. */
inline constexpr uint8_t I2cAddrSa0Low = 0x1C;

/** @brief I2C address when SA0 = 1. */
inline constexpr uint8_t I2cAddrSa0High = 0x1D;

/** @brief Accelerometer sensitivity at ±2 g: 0.244 mg/LSB → Q15 multiplier
 *         0.244 * 32768 / 1000 ≈ 8 (but we store in milli-units). Use 1000 *
 *         sensitivity so Q15 = 0.244 mg/LSB → multiplier ~8? Instead scale by
 *         4096: 0.244 * 4096 ≈ 1000. */
inline constexpr int32_t AccelSensQ15MgPerLsb = 1000;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::mma8451q
