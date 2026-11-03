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
 * @brief FreeRTOS WiFi HAL via stream driver abstraction.
 *
 * Provides `create_freertos_wifi_hal()`.  The body is guarded by
 * `#ifdef FREERTOS` so the translation unit compiles standalone without a
 * FreeRTOS port (returns UNSUPPORTED in that case).
 * @{
 */
namespace zzs::kernel::device::driver::wifi::platform::freertos {

/** @brief Build the FreeRTOS WiFi HAL vtable. */
[[nodiscard]] zzs::base::Maybe<zzs::kernel::device::driver::StreamBusInterface *, zzs::kernel::Status>
create_freertos_wifi_hal (WifiHal &hal_out) noexcept;

/** @} */

} // namespace zzs::kernel::device::driver::wifi::platform::freertos
