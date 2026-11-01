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
 * @brief Android GNSS stream HAL via NDK UART/USB-CDC (/dev/ttyUSB*).
 *
 * Guards all NDK includes behind `#ifdef __ANDROID__` so this translation unit
 * compiles (as a functional portable fallback) on hosts without the NDK.  The
 * real UART path is enabled when building with `__ANDROID__` and linking
 * against the NDK.  On other targets the portable fallback simulates a UART
 * stream over a fixed ring buffer and parses a minimal NMEA sentence.
 * @{
 */
namespace zzs::kernel::device::driver::gnss::platform::android {

/** @brief Build the Android GNSS stream HAL vtable. */
[[nodiscard]] const GnssPlatformHal *
create_android_gnss_hal () noexcept;

/** @} */
} // namespace zzs::kernel::device::driver::gnss::platform::android
