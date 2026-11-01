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
 * @brief LSM6DSO register map (adapted from STMicroelectronics STMems driver).
 *
 * Register addresses and bitfield definitions for the ST LSM6DSO 6-axis
 * IMU (3D accelerometer + 3D gyroscope).  Integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::lsm6dso
{

/** @brief WHO_AM_I register address. */
inline constexpr uint8_t RegWhoAmI = 0x0F;

/** @brief Expected WHO_AM_I value for LSM6DSO. */
inline constexpr uint8_t WhoAmIValue = 0x6C;

/** @brief CTRL1_XL: accelerometer ODR/FS. */
inline constexpr uint8_t RegCtrl1Xl = 0x10;
/* CTRL1_XL bit fields. */
inline constexpr uint8_t Ctrl1XlOdrMask = 0x0F;   /* bits [3:0] ODR_XL */
inline constexpr uint8_t Ctrl1XlFsMask  = 0x30;   /* bits [5:4] FS_XL */
inline constexpr uint8_t Ctrl1XlFs2g    = 0x00;   /* ±2 g full scale */
inline constexpr uint8_t Ctrl1XlFs4g    = 0x10;   /* ±4 g */
inline constexpr uint8_t Ctrl1XlFs8g    = 0x20;   /* ±8 g */
inline constexpr uint8_t Ctrl1XlFs16g   = 0x30;   /* ±16 g */

/** @brief CTRL2_G: gyroscope ODR/FS. */
inline constexpr uint8_t RegCtrl2G = 0x11;
/* CTRL2_G bit fields. */
inline constexpr uint8_t Ctrl2GOdrMask = 0x0F;   /* bits [3:0] ODR_G */
inline constexpr uint8_t Ctrl2GFsMask  = 0x70;   /* bits [6:4] FS_G */
inline constexpr uint8_t Ctrl2GFs125dps = 0x00;  /* ±125 dps */
inline constexpr uint8_t Ctrl2GFs250dps = 0x10;  /* ±250 dps */
inline constexpr uint8_t Ctrl2GFs500dps = 0x20;  /* ±500 dps */
inline constexpr uint8_t Ctrl2GFs1000dps = 0x40; /* ±1000 dps */
inline constexpr uint8_t Ctrl2GFs2000dps = 0x80; /* ±2000 dps */

/** @brief CTRL3_C: reset / boot / BDU / IF_INC. */
inline constexpr uint8_t RegCtrl3C = 0x12;
/* CTRL3_C bit fields. */
inline constexpr uint8_t Ctrl3CSwReset = 0x01;   /* bit 0  software reset */
inline constexpr uint8_t Ctrl3CIfInc   = 0x04;   /* bit 2  register address auto-increment */
inline constexpr uint8_t Ctrl3CBdu     = 0x08;   /* bit 3  block data update */
inline constexpr uint8_t Ctrl3CBoot    = 0x80;   /* bit 7  reboot memory content */

/** @brief CTRL4_C: self-test / enable. */
inline constexpr uint8_t RegCtrl4C = 0x13;
/* CTRL4_C bit fields. */
inline constexpr uint8_t Ctrl4CStXl  = 0x10;   /* bit 4 accelerometer self-test enable */
inline constexpr uint8_t Ctrl4CStG   = 0x20;   /* bit 5 gyroscope self-test enable */

/** @brief CTRL5_C: self-test enable and edge interrupt config. */
inline constexpr uint8_t RegCtrl5C = 0x14;
/* CTRL5_C bit fields. */
inline constexpr uint8_t Ctrl5CStXlMask = 0xC0;   /* bits [7:6] ST_XL sign */
inline constexpr uint8_t Ctrl5CStXlPos  = 0x40;   /* 01 accelerometer self-test positive */
inline constexpr uint8_t Ctrl5CStXlNeg  = 0x80;   /* 10 accelerometer self-test negative */
inline constexpr uint8_t Ctrl5CStGMask  = 0x30;   /* bits [5:4] ST_G sign */
inline constexpr uint8_t Ctrl5CStGPos   = 0x10;   /* 01 gyroscope self-test positive */
inline constexpr uint8_t Ctrl5CStGNeg   = 0x20;   /* 10 gyroscope self-test negative */

/** @brief STATUS_REG: data ready flags. */
inline constexpr uint8_t RegStatusReg = 0x1E;
/* STATUS_REG bit fields. */
inline constexpr uint8_t StatusXlda = 0x01;   /* bit 0 accelerometer data available */
inline constexpr uint8_t StatusGda  = 0x02;   /* bit 1 gyroscope data available */

/** @brief OUTX_L_A: accelerometer X axis low byte (6 bytes X/Y/Z). */
inline constexpr uint8_t RegOutXLAccel = 0x28;

/** @brief OUTX_L_G: gyroscope X axis low byte (6 bytes X/Y/Z). */
inline constexpr uint8_t RegOutXLGyro = 0x22;

/** @brief I2C address when SA0 = 0. */
inline constexpr uint8_t I2cAddrSa0Low = 0xD5;

/** @brief I2C address when SA0 = 1. */
inline constexpr uint8_t I2cAddrSa0High = 0xD7;

/** @brief Accelerometer sensitivity at ±2 g: 0.061 mg/LSB → Q15 = 1999. */
inline constexpr int32_t AccelSensQ15MgPerLsb = 1999;

/** @brief Gyroscope sensitivity at ±125 dps: 4.375 mdps/LSB.
 *
 *  Q15 cannot hold 4.375 * 2^15 = 143360 (exceeds Q15 max 32767 by 4.4×), so
 *  this lives in Q31 as int64_t: 4.375 * 2^31 = 9395240960.  Use q31_mul for
 *  gyro samples, q15_mul for accel/mag.
 */
inline constexpr int64_t GyroSensQ31MdpsPerLsb = 9395240960LL;

/** @brief Gyroscope sensitivity at ±125 dps: 4.375 mdps/LSB.
 *
 *  Kept under the historical `GyroSensQ15MdpsPerLsb` name for callers that
 *  reference it, but stored as an int64 Q31 value (4.375 * 2^31 = 9395240960)
 *  because 143360 does not fit in Q15.  Use the Q31/wide path for gyro
 *  conversion; do not pass it to the saturating Q15 multiply.
 */
inline constexpr int64_t GyroSensQ15MdpsPerLsb = GyroSensQ31MdpsPerLsb;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::lsm6dso
