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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#pragma once

#include "kernel/kernel_sensor_types.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Sound envelope channel (SOUND_ENVELOPE).
 *
 * Hardware sources: Android AudioRecord (16 kHz), iOS AVAudioEngine (16 kHz),
 * Linux ALSA/I2S MEMS, STM32/Zephyr I2S MEMS (ICS43434).  The HAL block-RMS
 * decimates 16 kHz audio to the 200 Hz envelope rate (_S_envelope_block = 80
 * samples per envelope sample) before delivering to the kernel.
 * @{
 */
namespace zzs::kernel::sensor::sound {

/** @brief Sound envelope channel id. */
inline constexpr sensor::Channel
channel (void) noexcept {
    return sensor::Channel::SOUND_ENVELOPE;
}

/** @brief Sound envelope sample rate (Hz), from the platform HAL table. */
[[nodiscard]] inline constexpr uint32_t
rate_hz (void) noexcept {
    return zzs::kernel::_S_envelope_rate_hz;
}

/** @brief Raw audio sample rate before envelope decimation (Hz). */
[[nodiscard]] inline constexpr uint32_t
audio_rate_hz (void) noexcept {
    return zzs::kernel::_S_sound_rate_hz;
}

/** @brief Audio samples per envelope block (16000 / 200 = 80). */
[[nodiscard]] inline constexpr uint32_t
envelope_block (void) noexcept {
    return zzs::kernel::_S_envelope_block;
}

/** @} */
} // namespace zzs::kernel::sensor::sound
