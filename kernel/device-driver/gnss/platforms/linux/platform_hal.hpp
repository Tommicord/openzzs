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

/* `linux` is a predefined macro (=1) on GCC/Linux; undef it so the
 * `platform::linux` namespace declaration is parsed as an identifier, not a
 * numeric constant. */
#undef linux
namespace zzs::kernel::device::driver::gnss::platform::linux {

/** @brief Build the Linux GNSS stream HAL vtable.
 *
 * Opens a UART/USB-CDC device node (e.g. /dev/ttyUSB0) and exposes it as a
 * `GnssPlatformHal`.  Returns nullptr when the device cannot be opened.
 */
[[nodiscard]] const GnssPlatformHal *
create_linux_gnss_hal () noexcept;

/** @} */
} // namespace zzs::kernel::device::driver::gnss::platform::linux
