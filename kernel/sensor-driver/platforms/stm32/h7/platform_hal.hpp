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
#include "../../platform_config.hpp"

#include <cstdint>

/**
 * @brief STM32H7 sensor HAL via STM32Cube HAL I2C/SPI.
 *
 * Reads LSM6DSO accel/gyro, LIS3MDL magnetometer, TSL2561 / VEML7700 light,
 * ICS43434 sound and ADC battery over HAL_I2C / HAL_SPI.  Timestamps use the
 * DWT cycle counter.  Implementation lives in the platform-specific
 * translation unit.
 * @{
 */
namespace zzs::kernel::sensor::driver::platform::stm32::h7
{

/** @brief Build the STM32H7 sensor HAL vtable. */
[[nodiscard]] const zzs::kernel::hal::_S_SensorHal *
create_stm32h7_sensor_hal (const PlatformConfig &config) noexcept;

/** @} */
} // namespace zzs::kernel::sensor::driver::platform::stm32::h7
