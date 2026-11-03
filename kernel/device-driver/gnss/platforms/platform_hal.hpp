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

#include "kernel/device-driver/stream_bus.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief GNSS platform HAL abstraction.
 *
 * Provides a platform-native byte-stream bus (UART / USB-CDC / SPI-data) for
 * the GNSS receiver driver.  The concrete implementation is selected at
 * compile time via `platforms/platform_select.hpp` and instantiated by
 * `platform_hal.cpp`.  Mirrors the sensor-driver HAL pattern: a vtable of
 * function pointers plus a factory function.
 *
 * All methods noexcept, no heap, no exceptions, integer-only.
 * @{
 */
namespace zzs::kernel::device::driver::gnss::platform {

/** @brief Virtual function table for a platform GNSS byte-stream bus. */
struct GnssPlatformHal {
    /** @brief Open the underlying stream device (UART, USB-CDC, etc.). */
    zzs::kernel::Status (*open) (void) noexcept;

    /** @brief Close the underlying stream device. */
    zzs::kernel::Status (*close) (void) noexcept;

    /** @brief Read up to `max_bytes` into `buf`; `*filled` is set. Non-blocking.
     *         Returns OK when at least one byte read, STALE when none available,
     *         or a bus error. */
    zzs::kernel::Status (*read_bytes) (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept;

    /** @brief Write `len` bytes from `buf`. Returns OK when all accepted. */
    zzs::kernel::Status (*write_bytes) (const uint8_t *buf, uint16_t len) noexcept;

    /** @brief Block for at least `ms` milliseconds. */
    void (*delay_ms) (uint32_t ms) noexcept;

    /** @brief Flush any buffered receive data. */
    zzs::kernel::Status (*flush_rx) (void) noexcept;

    /** @brief Return true when the stream is open/ready. */
    bool (*is_open) (void) noexcept;
};

/** @brief Return the installed platform HAL vtable, or nullptr when none is
 *         available for this build target. */
[[nodiscard]] const GnssPlatformHal *
create_gnss_platform_hal () noexcept;

/** @} */
} // namespace zzs::kernel::device::driver::gnss::platform
