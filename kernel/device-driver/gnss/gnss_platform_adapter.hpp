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

/**
 * @file kernel/device-driver/gnss/gnss_platform_adapter.hpp
 * @brief Header-only adapter: `GnssPlatformHal` -> `StreamBusInterface`.
 *
 * Wraps a platform GNSS HAL vtable (`GnssPlatformHal`) in the
 * `StreamBusInterface` abstraction used by `GnssDriver` / `GnssStream`, so a
 * platform HAL can plug into the driver without heap allocation or a dynamic
 * bus object.  All state lives in the adapter instance (a non-owning HAL
 * pointer); no heap, no writable statics, noexcept throughout.
 * @{
 */

#include "kernel/device-driver/gnss/platforms/platform_hal.hpp"
#include "kernel/device-driver/stream_bus.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>

namespace zzs::kernel::device::driver::gnss {

namespace platform = zzs::kernel::device::driver::gnss::platform;

/** @brief Adapts a `GnssPlatformHal` vtable to `StreamBusInterface`.
 *
 * The caller owns the `GnssPlatformHal` (typically a static const vtable
 * returned by `create_gnss_platform_hal()`); this adapter merely forwards
 * calls to it.  No heap, no copy, no writable statics. */
class GnssPlatformBusAdapter : public zzs::kernel::device::driver::StreamBusInterface {
  public:
    /** @brief Construct wrapping a non-owning HAL vtable pointer. */
    explicit GnssPlatformBusAdapter (const platform::GnssPlatformHal *hal) noexcept : _M_hal (hal) {
    }

    GnssPlatformBusAdapter (const GnssPlatformBusAdapter &) = delete;
    GnssPlatformBusAdapter &
    operator= (const GnssPlatformBusAdapter &) = delete;
    GnssPlatformBusAdapter (GnssPlatformBusAdapter &&) = delete;
    GnssPlatformBusAdapter &
    operator= (GnssPlatformBusAdapter &&) = delete;
    ~GnssPlatformBusAdapter () override = default;

    // -- StreamBusInterface --

    [[nodiscard]] zzs::kernel::Status
    read_bytes (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept override {
        if (_M_hal == nullptr || _M_hal->read_bytes == nullptr)
            return zzs::kernel::Status::INTERNAL;
        return _M_hal->read_bytes (buf, max_bytes, filled);
    }

    [[nodiscard]] zzs::kernel::Status
    write_bytes (const uint8_t *buf, uint16_t len) noexcept override {
        if (_M_hal == nullptr || _M_hal->write_bytes == nullptr)
            return zzs::kernel::Status::INTERNAL;
        return _M_hal->write_bytes (buf, len);
    }

    void
    delay_ms (uint32_t ms) noexcept override {
        if (_M_hal != nullptr && _M_hal->delay_ms != nullptr)
            _M_hal->delay_ms (ms);
    }

    [[nodiscard]] zzs::kernel::Status
    flush_rx () noexcept override {
        if (_M_hal == nullptr || _M_hal->flush_rx == nullptr)
            return zzs::kernel::Status::INTERNAL;
        return _M_hal->flush_rx ();
    }

    [[nodiscard]] bool
    is_open () const noexcept override {
        if (_M_hal == nullptr || _M_hal->is_open == nullptr)
            return false;
        return _M_hal->is_open ();
    }

    /** @brief Open the underlying stream. */
    [[nodiscard]] zzs::kernel::Status
    open () noexcept {
        if (_M_hal == nullptr || _M_hal->open == nullptr)
            return zzs::kernel::Status::INTERNAL;
        return _M_hal->open ();
    }

    /** @brief Close the underlying stream. */
    [[nodiscard]] zzs::kernel::Status
    close () noexcept {
        if (_M_hal == nullptr || _M_hal->close == nullptr)
            return zzs::kernel::Status::INTERNAL;
        return _M_hal->close ();
    }

    /** @brief Return the wrapped HAL vtable (may be nullptr). */
    [[nodiscard]] const platform::GnssPlatformHal *
    hal () const noexcept {
        return _M_hal;
    }

  private:
    const platform::GnssPlatformHal *_M_hal = nullptr;
};

/** @} */
} // namespace zzs::kernel::device::driver::gnss
