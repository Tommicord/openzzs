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
 * @brief Battery sensor channel (BATTERY).
 *
 * Hardware sources: Android BatteryManager, iOS UIDevice, Linux sysfs
 * /sys/class/power_supply, STM32/Zephyr ADC / fuel gauge.  Sampled at 1 Hz and
 * upsampled to the 100 Hz grid by the resampler.
 * @{
 */
namespace zzs::kernel::sensor::battery {

/** @brief Battery channel id. */
inline constexpr sensor::Channel
channel (void) noexcept {
  return sensor::Channel::BATTERY;
}

/** @brief Battery sample rate (Hz), from the platform HAL table. */
[[nodiscard]] inline constexpr uint32_t
rate_hz (void) noexcept {
  return zzs::kernel::_S_battery_rate_hz;
}

/** @} */
} // namespace zzs::kernel::sensor::battery
