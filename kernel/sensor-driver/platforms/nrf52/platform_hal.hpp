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

#include "../platform_config.hpp"
#include "kernel/kernel_sensor_hal.hpp"
#include "std/maybe.hpp"

#include <cstdint>

/**
 * @brief nRF52 sensor HAL via Nordic SDK nrf_drv_twi.h / nrf_drv_spi.h.
 *
 * Reads LSM6DSO accel/gyro, LIS3MDL magnetometer, TSL2561 / VEML7700 light,
 * ICS43434 sound and battery over Nordic TWIM/SPIM.  Timestamps use the RTC
 * / TIMER counter.  Implementation lives in the platform-specific translation
 * unit.
 * @{
 */
namespace zzs::kernel::sensor::driver::platform::nrf52 {

/** @brief Build the nRF52 sensor HAL vtable. */
[[nodiscard]] zzs::base::Maybe<const zzs::kernel::hal::_S_SensorHal *, zzs::kernel::Status>
create_nrf52_sensor_hal (const PlatformConfig &config) noexcept;

/** @} */
} // namespace zzs::kernel::sensor::driver::platform::nrf52
