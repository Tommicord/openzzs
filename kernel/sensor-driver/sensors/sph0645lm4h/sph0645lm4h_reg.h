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
 * @brief Knowles SPH0645LM4H PDM microphone configuration constants.
 *
 * The SPH0645LM4H is a digital PDM MEMS microphone outputting PDM over an I2S
 * interface.  There are no traditional device registers; the device is
 * configured by the I2S clock ratio.  These constants define the bit-bang
 * timing and envelope parameters.  Integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors::sph0645lm4h {

/** @brief I2S bit clock period in microseconds (e.g. 2.083 us = 480 kHz). */
inline constexpr uint32_t I2sSckPeriodUs = 2u;

/** @brief I2S word select period in microseconds (e.g. 1/16 kHz = 62.5 us). */
inline constexpr uint32_t I2sWsPeriodUs = 62u;

/** @brief Number of PDM bits per audio sample (64 for 16-bit at 480 kHz SCK). */
inline constexpr uint32_t PdmBitsPerSample = 64u;

/** @brief Audio sample rate (Hz): 16000 Hz PDM input. */
inline constexpr uint32_t AudioRateHz = 16000u;

/** @brief Sound envelope rate (Hz): 200 Hz kernel envelope rate. */
inline constexpr uint32_t EnvelopeRateHz = 200u;

/** @brief Audio samples per envelope block (16000 / 200 = 80). */
inline constexpr uint32_t EnvelopeBlock = 80u;

/** @brief Envelope smoothing coefficient (Q15), alpha = 0.1 → 3277. */
inline constexpr int32_t EnvelopeAlphaQ15 = 3277;

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors::sph0645lm4h
