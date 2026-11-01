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
 * @brief Light sensor channel (LIGHT).
 *
 * Hardware sources: Android TYPE_LIGHT (1 Hz), iOS UIScreen brightness proxy,
 * Linux IIO light / GPIO, STM32/Zephyr TSL2561 / VEML7700.  Upsampled from 1 Hz
 * to the 100 Hz grid by the resampler before the flicker stage.
 * @{
 */
namespace zzs::kernel::sensor::light {

/** @brief Light channel id. */
inline constexpr sensor::Channel
channel (void) noexcept {
    return sensor::Channel::LIGHT;
}

/** @brief Light sample rate (Hz), from the platform HAL table. */
[[nodiscard]] inline constexpr uint32_t
rate_hz (void) noexcept {
    return zzs::kernel::_S_light_rate_hz;
}

/** @} */
} // namespace zzs::kernel::sensor::light
