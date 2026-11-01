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

#include "kernel/kernel_sensor_hal.hpp"
#include "../platform_config.hpp"

#include <cstdint>

/**
 * @brief Raspberry Pi Pico (RP2040) sensor HAL via SDK i2c.h / spi.h.
 *
 * Reads LSM6DSO accel/gyro, LIS3MDL magnetometer, TSL2561 / VEML7700 light,
 * ICS43434 sound and battery over RP2040 I2C/SPI.  Timestamps use
 * to_us_since_boot().  Implementation lives in the platform-specific
 * translation unit.
 * @{
 */
namespace zzs::kernel::sensor::driver::platform::rp2040
{

/** @brief Build the RP2040 sensor HAL vtable. */
[[nodiscard]] const zzs::kernel::hal::_S_SensorHal *
create_rp2040_sensor_hal (const PlatformConfig &config) noexcept;

/** @} */
} // namespace zzs::kernel::sensor::driver::platform::rp2040
