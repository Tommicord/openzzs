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
 * @brief InvenSense ICM20608 3-axis gyroscope register map.
 *
 * The ICM20608 is a 3-axis digital gyroscope found in many phones.  Registers
 * are integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::icm20608 {

/** @brief WHO_AM_I register address. */
inline constexpr uint8_t RegWhoAmI = 0x75;

/** @brief Expected WHO_AM_I value for ICM20608. */
inline constexpr uint8_t WhoAmIValue = 0xA0;

/** @brief PWR_MGMT_1: power management / clock source / reset. */
inline constexpr uint8_t RegPwrMgmt1 = 0x6B;
/* PWR_MGMT_1 bit fields. */
inline constexpr uint8_t PwrMgmt1Reset = 0x80;    /* bit 7 device reset */
inline constexpr uint8_t PwrMgmt1Sleep = 0x40;    /* bit 6 sleep mode */
inline constexpr uint8_t PwrMgmt1ClkMask = 0x07;  /* bits [2:0] clock source */
inline constexpr uint8_t PwrMgmt1ClkXgyro = 0x01; /* PLL with X gyro reference */

/** @brief PWR_MGMT_2: enable/disable accel and gyro. */
inline constexpr uint8_t RegPwrMgmt2 = 0x6C;

/** @brief GYRO_CONFIG: gyroscope full-scale. */
inline constexpr uint8_t RegGyroConfig = 0x1B;
/* GYRO_CONFIG bit fields. */
inline constexpr uint8_t GyroConfigFsMask = 0x18;    /* bits [4:3] full-scale */
inline constexpr uint8_t GyroConfigFs250dps = 0x00;  /* ±250 dps */
inline constexpr uint8_t GyroConfigFs500dps = 0x08;  /* ±500 dps */
inline constexpr uint8_t GyroConfigFs1000dps = 0x10; /* ±1000 dps */
inline constexpr uint8_t GyroConfigFs2000dps = 0x18; /* ±2000 dps */

/** @brief SMPLRT_DIV: sample rate divider. */
inline constexpr uint8_t RegSmlrtDiv = 0x19;

/** @brief INT_STATUS: interrupt status. */
inline constexpr uint8_t RegIntStatus = 0x3A;
/* INT_STATUS bit fields. */
inline constexpr uint8_t IntStatusDataReady = 0x01; /* bit 0 data ready */

/** @brief GYRO_XOUT_H: gyroscope X axis high byte (6 bytes X/Y/Z). */
inline constexpr uint8_t RegGyroXoutH = 0x43;

/** @brief I2C address when AD0 = 0. */
inline constexpr uint8_t I2cAddrAd0Low = 0x68;

/** @brief I2C address when AD0 = 1. */
inline constexpr uint8_t I2cAddrAd0High = 0x69;

/** @brief Gyroscope sensitivity at ±250 dps: 131 LSB/(dps) → 7.63 mdps/LSB → Q15
 *         multiplier ≈ 250. */
inline constexpr int32_t GyroSensQ15MdpsPerLsb = 250;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::icm20608
