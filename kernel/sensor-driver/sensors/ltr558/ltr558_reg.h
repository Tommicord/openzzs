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
 * @brief Lite-On LTR-558 ambient light / proximity sensor register map.
 *
 * The LTR-558 is a low-power ambient light and proximity sensor found in many
 * phones.  Registers are integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::ltr558 {

/** @brief Part ID register address. */
inline constexpr uint8_t RegPartId = 0x06;

/** @brief Expected Part ID value for LTR-558. */
inline constexpr uint8_t PartIdValue = 0x05;

/** @brief ALS control register. */
inline constexpr uint8_t RegAlsCtrl = 0x00;
/* ALS_CTRL bit fields. */
inline constexpr uint8_t AlsCtrlModeStandby = 0x00; /* standby */
inline constexpr uint8_t AlsCtrlModeActive = 0x01;  /* active mode */
inline constexpr uint8_t AlsCtrlGainMask = 0x06;    /* bits [2:1] gain */
inline constexpr uint8_t AlsCtrlGain1x = 0x00;      /* 1x gain */
inline constexpr uint8_t AlsCtrlGain2x = 0x02;      /* 2x gain */
inline constexpr uint8_t AlsCtrlGain4x = 0x04;      /* 4x gain */
inline constexpr uint8_t AlsCtrlGain8x = 0x06;      /* 8x gain */
inline constexpr uint8_t AlsCtrlOdrMask = 0x18;     /* bits [4:3] ODR */
inline constexpr uint8_t AlsCtrlOdr50ms = 0x00;     /* 50 ms (20 Hz) */
inline constexpr uint8_t AlsCtrlOdr100ms = 0x08;    /* 100 ms (10 Hz) */
inline constexpr uint8_t AlsCtrlOdr200ms = 0x10;    /* 200 ms (5 Hz) */
inline constexpr uint8_t AlsCtrlOdr400ms = 0x18;    /* 400 ms (2.5 Hz) */
inline constexpr uint8_t AlsCtrlOdr1000ms = 0x00;   /* reserved */

/** @brief PS control register. */
inline constexpr uint8_t RegPsCtrl = 0x01;

/** @brief PS/LED control register. */
inline constexpr uint8_t RegPsLed = 0x02;

/** @brief Numb of pulses / measurement rate. */
inline constexpr uint8_t RegMeasRate = 0x04;

/** @brief PART_ID / chip ID. */
inline constexpr uint8_t RegChipId = 0x06;

/** @brief ALS data MSB / data block (ALS_DATA_CH1_LOW = 0x0D, CH1_HIGH = 0x0E). */
inline constexpr uint8_t RegAlsDataCh1Low = 0x0D;
inline constexpr uint8_t RegAlsDataCh1High = 0x0E;

/** @brief ALS status register. Bit 0 = ALS data valid. */
inline constexpr uint8_t RegAlsStatus = 0x0C;
/* ALS_STATUS bit fields. */
inline constexpr uint8_t AlsStatusAlsValid = 0x01;

/** @brief I2C address. */
inline constexpr uint8_t I2cAddr = 0x23;

/** @brief ALS data block size (CH1 low + high = 2 bytes, but we read 2 bytes). */
inline constexpr uint32_t AlsDataBytes = 2u;

/** @brief Light ODR: 10 Hz (100 ms). */
inline constexpr uint32_t OdrHz = 10u;

/** @brief Lux conversion: gain 1x, 100 ms → ~0.59 lux/count → Q15 multiplier
 *         0.59 * 32768 ≈ 19333 (in milli-lux: 19333 * 1000). We store milli-lux. */
inline constexpr int32_t LuxSensQ15MilliLuxPerCount = 590;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::ltr558
