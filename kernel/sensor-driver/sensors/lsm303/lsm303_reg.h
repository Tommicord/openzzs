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
 * @brief LSM303 register map (adapted from ST LSM303AGR/DLHC).
 *
 * The LSM303 is a combined accelerometer + magnetometer on one package with two
 * I2C sub-addresses.  Register addresses and bitfield definitions for both
 * sub-devices.  Integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::lsm303 {

/** @brief Accelerometer sub-device WHO_AM_I register address. */
inline constexpr uint8_t RegWhoAmI = 0x0F;

/** @brief Expected accelerometer WHO_AM_I value (LSM303DLHC/AGR). */
inline constexpr uint8_t AccelWhoAmIValue = 0x33;

/** @brief Expected magnetometer WHO_AM_I value (LSM303AGR). */
inline constexpr uint8_t MagWhoAmIValue = 0x40;

/* ---------------------------------------------------------------
 * Accelerometer registers (I2C address 0x18/0x19, SA0 pin)
 * --------------------------------------------------------------- */

/** @brief Accel CTRL_REG1_A: ODR + power mode. */
inline constexpr uint8_t RegCtrlReg1A = 0x20;
/* CTRL_REG1_A bit fields. */
inline constexpr uint8_t CtrlReg1AODRMask = 0xF0;  /* bits [7:4] ODR */
inline constexpr uint8_t CtrlReg1AODR100Hz = 0x50; /* 100 Hz normal mode */

/** @brief Accel CTRL_REG4_A: full-scale, BDU. */
inline constexpr uint8_t RegCtrlReg4A = 0x23;
/* CTRL_REG4_A bit fields. */
inline constexpr uint8_t CtrlReg4ABdu = 0x40;    /* bit 6 block data update */
inline constexpr uint8_t CtrlReg4AFsMask = 0x30; /* bits [5:4] full-scale */
inline constexpr uint8_t CtrlReg4AFs2g = 0x00;   /* ±2 g */

/** @brief Accel CTRL_REG5_A: self-test enable. */
inline constexpr uint8_t RegCtrlReg5A = 0x24;
/* CTRL_REG5_A bit fields. */
inline constexpr uint8_t CtrlReg5AStX = 0x02; /* bit 1 accelerometer self-test X */
inline constexpr uint8_t CtrlReg5AStY = 0x04; /* bit 2 accelerometer self-test Y */
inline constexpr uint8_t CtrlReg5AStZ = 0x08; /* bit 3 accelerometer self-test Z */

/** @brief Accel status register. */
inline constexpr uint8_t RegStatusRegA = 0x27;
inline constexpr uint8_t StatusRegAZyxda = 0x08; /* bit 3 X/Y/Z data available */

/** @brief Accel OUT_X_L_A: 6 bytes X/Y/Z. */
inline constexpr uint8_t RegOutXLAccel = 0x28;

/* ---------------------------------------------------------------
 * Magnetometer registers (I2C address 0x1C/0x1D, SA1 pin)
 * --------------------------------------------------------------- */

/** @brief Mag CTRL_REG1_M: temperature + data rate. */
inline constexpr uint8_t RegCtrlReg1M = 0x20;

/** @brief Mag CTRL_REG2_M: full-scale. */
inline constexpr uint8_t RegCtrlReg2M = 0x21;
inline constexpr uint8_t CtrlReg2MFsMask = 0x60;    /* bits [6:5] full-scale */
inline constexpr uint8_t CtrlReg2MFsDefault = 0x00; /* ±1.3 gauss (default) */

/** @brief Mag CTRL_REG3_M: operating mode. */
inline constexpr uint8_t RegCtrlReg3M = 0x22;
inline constexpr uint8_t CtrlReg3MmdContinuous = 0x00; /* bits [1:0] continuous */

/** @brief Mag status register. */
inline constexpr uint8_t RegStatusRegM = 0x27;
inline constexpr uint8_t StatusRegMZyxda = 0x08; /* bit 3 X/Y/Z data available */

/** @brief Mag OUT_X_L_M: 6 bytes X/Y/Z. */
inline constexpr uint8_t RegOutXLMag = 0x68;

/* ---------------------------------------------------------------
 * I2C addresses
 * --------------------------------------------------------------- */

/** @brief Accelerometer I2C address when SA0 = 0. */
inline constexpr uint8_t I2cAddrAccelSa0Low = 0x18;

/** @brief Accelerometer I2C address when SA0 = 1. */
inline constexpr uint8_t I2cAddrAccelSa0High = 0x19;

/** @brief Magnetometer I2C address when SA1 = 0. */
inline constexpr uint8_t I2cAddrMagSa1Low = 0x1C;

/** @brief Magnetometer I2C address when SA1 = 1. */
inline constexpr uint8_t I2cAddrMagSa1High = 0x1D;

/* ---------------------------------------------------------------
 * Sensitivity (Q15 multipliers)
 * --------------------------------------------------------------- */

/** @brief Accelerometer sensitivity at ±2 g: 0.061 mg/LSB → Q15 = 1999. */
inline constexpr int32_t AccelSensQ15MgPerLsb = 1999;

/** @brief Magnetometer sensitivity: ~0.684 milligauss/LSB → Q15 = 22412. */
inline constexpr int32_t MagSensQ15MilligaussPerLsb = 22412;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::lsm303
