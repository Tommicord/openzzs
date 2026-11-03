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
 * @brief TSL2561 register map.
 *
 * Register addresses and bitfield definitions for the TAOS/AMS TSL2561
 * luminosity sensor.  Integer-only; no float.  Lux is computed in Q11.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::tsl2561 {

/** @brief I2C address when address pin = 0 (0x29 >> 1 = 0x29, 8-bit 0x52). */
inline constexpr uint8_t I2cAddrLow = 0x29;

/** @brief I2C address when address pin = floating (0x39, 8-bit 0x72). */
inline constexpr uint8_t I2cAddrFloat = 0x39;

/** @brief I2C address when address pin = 1 (0x49, 8-bit 0x92). */
inline constexpr uint8_t I2cAddrHigh = 0x49;

/** @brief CONTROL register: power on/off. */
inline constexpr uint8_t RegControl = 0x00;
inline constexpr uint8_t ControlPowerOn = 0x01; /* bit 0 */
inline constexpr uint8_t ControlPowerOff = 0x00;

/** @brief TIMING register: gain + integration time. */
inline constexpr uint8_t RegTiming = 0x01;
/* TIMING bit fields. */
inline constexpr uint8_t TimingGainMask = 0x10; /* bit 4: 0=1x, 1=16x */
inline constexpr uint8_t TimingGain1x = 0x00;
inline constexpr uint8_t TimingGain16x = 0x10;
inline constexpr uint8_t TimingIntegMask = 0x07;  /* bits [2:0] */
inline constexpr uint8_t TimingInteg13ms = 0x00;  /* 13.7 ms */
inline constexpr uint8_t TimingInteg101ms = 0x01; /* 101 ms */
inline constexpr uint8_t TimingInteg402ms = 0x02; /* 402 ms */

/** @brief CHAN0_LOW: channel 0 (broadband) data low byte. */
inline constexpr uint8_t RegChan0Low = 0x0C;

/** @brief CHAN1_LOW: channel 1 (IR) data low byte. */
inline constexpr uint8_t RegChan1Low = 0x0E;

/** @brief Number of bytes per channel read (low + high). */
inline constexpr uint32_t ChanBytes = 2u;

/** @brief Command bit: bit 7 set = command register, bit 5 = auto-increment. */
inline constexpr uint8_t CmdRegisterBit = 0x80;
inline constexpr uint8_t CmdAutoIncBit = 0x20;

/** @brief TSL2561 package type (T/FN/CL) coefficients for 1x gain, 13.7 ms.
 *
 * These are the datasheet coefficients scaled to Q15 for integer-only lux.
 * ratio = ch1 / ch0 in Q15.
 * @{
 */

/** @brief Range 1 upper bound: ratio <= 0.50 → 16384 in Q15. */
inline constexpr int32_t LuxRatioBound1 = 16384;
/** @brief Range 2 upper bound: ratio <= 0.61 → 19988 in Q15. */
inline constexpr int32_t LuxRatioBound2 = 19988;
/** @brief Range 3 upper bound: ratio <= 0.80 → 26214 in Q15. */
inline constexpr int32_t LuxRatioBound3 = 26214;
/** @brief Range 4 upper bound: ratio <= 1.30 → 42598 in Q15. */
inline constexpr int32_t LuxRatioBound4 = 42598;

/** @brief Range 2: coeffA = round(0.0224 * 32768) = 734. */
inline constexpr int32_t LuxCoeff2A = 734;
/** @brief Range 2: coeffB = round(0.031 * 32768) = 1016. */
inline constexpr int32_t LuxCoeff2B = 1016;

/** @brief Range 3: coeffA = round(0.0128 * 32768) = 419. */
inline constexpr int32_t LuxCoeff3A = 419;
/** @brief Range 3: coeffB = round(0.0153 * 32768) = 501. */
inline constexpr int32_t LuxCoeff3B = 501;

/** @brief Range 4: coeff = round(0.00314 * 32768) = 103. */
inline constexpr int32_t LuxCoeff4 = 103;

/** @brief Range 1: coeffA = round(0.0304 * 32768) = 996. */
inline constexpr int32_t LuxCoeff1A = 996;

/** @} */

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::tsl2561
