/*
 * Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

#pragma once

/**
 * @file test_mock_stream_bus.hpp
 * @brief Mock StreamBusInterface for GNSS driver unit tests.
 *
 * A deterministic in-memory byte stream. Tests inject bytes that the driver
 * reads as if they arrived from a GNSS receiver over UART/USB-CDC. Writes are
 * recorded for inspection (e.g. UBX CFG commands sent during init).
 *
 * No hardware, no float, no heap.
 */

#include "kernel/device-driver/stream_bus.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace test {

/** @brief In-memory byte stream implementing StreamBusInterface. */
class MockStreamBus : public zzs::kernel::device::driver::StreamBusInterface {
  public:
    MockStreamBus () noexcept = default;

    ~MockStreamBus () override = default;
    MockStreamBus (const MockStreamBus &) = delete;
    MockStreamBus &
    operator= (const MockStreamBus &) = delete;
    MockStreamBus (MockStreamBus &&) = delete;
    MockStreamBus &
    operator= (MockStreamBus &&) = delete;

    // -- StreamBusInterface --

    [[nodiscard]] zzs::kernel::Status
    read_bytes (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept override {
        if (buf == nullptr || filled == nullptr)
            return zzs::kernel::Status::OUT_OF_RANGE;
        if (!_M_open)
            return zzs::kernel::Status::INTERNAL;
        if (_M_rx_index >= _M_rx.size ()) {
            *filled = 0u;
            return zzs::kernel::Status::STALE;
        }
        uint16_t n = 0u;
        while (n < max_bytes && _M_rx_index < _M_rx.size ()) {
            buf[n++] = _M_rx[_M_rx_index++];
        }
        *filled = n;
        return n > 0u ? zzs::kernel::Status::OK : zzs::kernel::Status::STALE;
    }

    [[nodiscard]] zzs::kernel::Status
    write_bytes (const uint8_t *buf, uint16_t len) noexcept override {
        if (buf == nullptr && len > 0u)
            return zzs::kernel::Status::OUT_OF_RANGE;
        if (!_M_open)
            return zzs::kernel::Status::INTERNAL;
        for (uint16_t i = 0u; i < len; ++i)
            _M_tx.push_back (buf[i]);
        return zzs::kernel::Status::OK;
    }

    void
    delay_ms (uint32_t ms) noexcept override {
        _M_delay_calls += 1u;
        _M_delay_total_ms += ms;
    }

    [[nodiscard]] zzs::kernel::Status
    flush_rx () noexcept override {
        _M_rx.clear ();
        _M_rx_index = 0u;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] bool
    is_open () const noexcept override {
        return _M_open;
    }

    // -- Test helpers --

    void
    set_open (bool open) noexcept {
        _M_open = open;
    }

    /** @brief Inject received bytes (as if arriving from the GNSS receiver). */
    void
    inject (const uint8_t *data, uint16_t len) noexcept {
        if (data == nullptr || len == 0u)
            return;
        for (uint16_t i = 0u; i < len; ++i)
            _M_rx.push_back (data[i]);
    }

    /** @brief Inject a null-terminated string as received bytes. */
    void
    inject_str (const char *str) noexcept {
        if (str == nullptr)
            return;
        uint16_t len = 0u;
        while (str[len] != '\0')
            ++len;
        inject (reinterpret_cast<const uint8_t *> (str), len);
    }

    void
    clear () noexcept {
        _M_rx.clear ();
        _M_rx_index = 0u;
        _M_tx.clear ();
        _M_delay_calls = 0u;
        _M_delay_total_ms = 0u;
    }

    [[nodiscard]] uint32_t
    delay_calls () const noexcept {
        return _M_delay_calls;
    }

    [[nodiscard]] uint32_t
    delay_total_ms () const noexcept {
        return _M_delay_total_ms;
    }

    [[nodiscard]] const std::vector<uint8_t> &
    tx_bytes () const noexcept {
        return _M_tx;
    }

    [[nodiscard]] uint32_t
    tx_count () const noexcept {
        return static_cast<uint32_t> (_M_tx.size ());
    }

    [[nodiscard]] uint32_t
    rx_remaining () const noexcept {
        return static_cast<uint32_t> (_M_rx.size () - _M_rx_index);
    }

  private:
    bool _M_open = false;
    std::vector<uint8_t> _M_rx;
    uint32_t _M_rx_index = 0u;
    std::vector<uint8_t> _M_tx;
    uint32_t _M_delay_calls = 0u;
    uint32_t _M_delay_total_ms = 0u;
};

} // namespace test
