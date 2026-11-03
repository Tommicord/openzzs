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
 * @brief VEML7700 register map.
 *
 * Register addresses and bitfield definitions for the Vishay VEML7700
 * ambient light sensor.  Integer-only; no float.  Lux is computed in Q11.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::veml7700 {

/** @brief I2C address (fixed). */
inline constexpr uint8_t I2cAddr = 0x10;

/** @brief ALS_CONF register: gain, integration time, shutdown, interrupt. */
inline constexpr uint8_t RegAlsConf = 0x00;
/* ALS_CONF bit fields (16-bit register, high byte first). */
inline constexpr uint16_t AlsConfGainMask = 0x7000;   /* bits [14:12] ALS gain */
inline constexpr uint16_t AlsConfGain1_6x = 0x0000;   /* 1/6x */
inline constexpr uint16_t AlsConfGain1_4x = 0x1000;   /* 1/4x (default) */
inline constexpr uint16_t AlsConfGain1x = 0x5000;     /* 1x */
inline constexpr uint16_t AlsConfGain2x = 0x7000;     /* 2x */
inline constexpr uint16_t AlsConfIntegMask = 0x0C00;  /* bits [11:10] integration time */
inline constexpr uint16_t AlsConfInteg25ms = 0x0000;  /* 25 ms */
inline constexpr uint16_t AlsConfInteg50ms = 0x0400;  /* 50 ms */
inline constexpr uint16_t AlsConfInteg100ms = 0x0800; /* 100 ms (default) */
inline constexpr uint16_t AlsConfInteg200ms = 0x0C00; /* 200 ms */
inline constexpr uint16_t AlsConfShutdown = 0x0100;   /* bit 8: 1=shutdown, 0=on */
inline constexpr uint16_t AlsConfIntEn = 0x0200;      /* bit 9: interrupt enable */

/** @brief ALS_WH: high threshold register. */
inline constexpr uint8_t RegAlsWh = 0x01;

/** @brief ALS_WL: low threshold register. */
inline constexpr uint8_t RegAlsWl = 0x02;

/** @brief ALS_DATA: ambient light data output (16-bit). */
inline constexpr uint8_t RegAlsData = 0x04;

/** @brief VEML7700 sensitivity at default settings (gain 1x, 100 ms):
 *         ~0.0576 lux per count → Q11 multiplier = 118. */
inline constexpr int32_t LuxSensQ11PerCount = 118;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::veml7700
