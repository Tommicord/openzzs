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

#include "../platform_hal.hpp"

#include <cstdint>

/**
 * @brief ESP32 GNSS stream HAL via ESP-IDF UART driver.
 *
 * Guards all ESP-IDF includes behind `#ifdef CONFIG_IDF_TARGET_ESP32` so this
 * translation unit compiles (as a functional portable fallback) on hosts
 * without the ESP-IDF SDK.  The real UART path is enabled when building with
 * `CONFIG_IDF_TARGET_ESP32` and linking against `driver/uart.h`.  On other
 * targets the portable fallback simulates a UART stream over a fixed ring
 * buffer and parses a minimal NMEA sentence.
 * @{
 */
namespace zzs::kernel::device::driver::gnss::platform::esp32 {

/** @brief Build the ESP32 GNSS stream HAL vtable. */
[[nodiscard]] const GnssPlatformHal *
create_esp32_gnss_hal () noexcept;

/** @} */
} // namespace zzs::kernel::device::driver::gnss::platform::esp32
