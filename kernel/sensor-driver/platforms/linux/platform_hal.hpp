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
#include "std/fmt.hpp"

#include <cstdint>

/**
 * @brief Linux x86_64/ARM sensor HAL via /dev/i2c-N and /dev/spidev-N.
 *
 * Reads LSM6DSO accel/gyro, LIS3MDL magnetometer, TSL2561 / VEML7700 light,
 * ICS43434 sound and sysfs battery over platform-native I2C/SPI device nodes.
 * Timestamps use CLOCK_MONOTONIC_RAW.  Implementation lives in the
 * platform-specific translation unit.
 * @{
 */

/* `linux` is a predefined macro (=1) on GCC/Linux; undef it so the `platform::linux`
 * namespace declaration is parsed as an identifier, not a numeric constant. */
#undef linux
namespace zzs::kernel::sensor::driver::platform::linux
{

/** @brief Build the Linux sensor HAL vtable.
 *
 * Returns the installed HAL vtable.  No configuration is required: the HAL
 * uses fixed default device nodes (/dev/i2c-1, /dev/spidev-1.0) and always
 * instantiates the full sensor set.
 */
[[nodiscard]] const zzs::kernel::hal::_S_SensorHal *
create_linux_sensor_hal () noexcept;

/** @} */
} // namespace zzs::kernel::sensor::driver::platform::linux
