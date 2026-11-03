/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
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
 * @brief BMM150 (Bosch) 3-axis magnetometer register map.
 *
 * The BMM150 is another very common magnetometer in Android phones (Motorola,
 * some Samsung models, Xiaomi, and others).  Registers are integer-only;
 * no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::bmm150 {

/** @brief Chip ID register address. */
inline constexpr uint8_t RegChipId = 0x40;

/** @expected chip ID value for BMM150. */
inline constexpr uint8_t ChipIdValue = 0x32;

/** @brief Data X LSB (start of 8-byte data block). */
inline constexpr uint8_t RegDataXlsb = 0x42;

/** @brief Power control register. */
inline constexpr uint8_t RegPwrCntl = 0x4B;

/* PWR_CNTL bit fields. */
inline constexpr uint8_t PwrCntlOpModeMask = 0x03; /* bits [1:0] operating mode */
inline constexpr uint8_t PwrCntlSleep = 0x01;      /* sleep mode */
inline constexpr uint8_t PwrCntlForced = 0x02;     /* forced mode */
inline constexpr uint8_t PwrCntlNormal = 0x00;     /* normal mode */

/** @brief Data block size: X LSB/MSB, Y LSB/MSB, Z LSB/MSB, R LSB/MSB (8 bytes). */
inline constexpr uint32_t DataBlockBytes = 8u;

/** @brief I2C address (fixed 7-bit, BMM150 uses 0x10). */
inline constexpr uint8_t I2cAddr = 0x10;

/** @brief Magnetometer ODR: 100 Hz (typical for BMM150 normal mode). */
inline constexpr uint32_t OdrHz = 100u;

/** @brief Sensitivity: BMM150 full scale ±1300 µT, 16-bit.
 *  Sensitivity ≈ 1300 µT / 32768 ≈ 0.0397 µT/LSB → Q15 ≈ 1300 (milligauss: 0.397 mG/LSB → 13002 Q15).
 *  Actually: 1 LSB = 0.397 mG ≈ 1300 Q15 units when expressing in milligauss. */
inline constexpr int32_t MagSensQ15MilligaussPerLsb = 13002;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::bmm150
