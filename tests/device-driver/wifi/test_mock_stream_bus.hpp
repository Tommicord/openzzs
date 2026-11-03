/*
 * Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

#pragma once

/**
 * @file test_mock_stream_bus.hpp
 * @brief Mock StreamBusInterface for WiFi driver unit tests.
 *
 * An in-memory byte stream that can inject canned responses on read() and
 * record everything written via write_bytes().  No hardware, no float, no
 * heap.
 */

#include "kernel/device-driver/stream_bus.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace test {

using namespace zzs::kernel::device::driver::wifi;

/** @brief Maximum bytes the mock can hold in its receive queue. */
inline constexpr uint16_t MockMaxRxBytes = 256u;

/** @brief Maximum number of commands the mock records. */
inline constexpr uint16_t MockMaxCommands = 16u;

/** @brief Mock byte-stream bus implementing StreamBusInterface. */
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

  [[nodiscard]] zzs::kernel::Status
  read_bytes (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept override {
    *filled = 0u;
    if (buf == nullptr || filled == nullptr)
      return zzs::kernel::Status::OUT_OF_RANGE;
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    if (_M_read_error != zzs::kernel::Status::OK)
      return _M_read_error;
    if (_M_rx_count == 0u) {
      _M_stale_reads += 1u;
      return zzs::kernel::Status::STALE;
    }
    uint16_t to_copy = (max_bytes < _M_rx_count) ? max_bytes : _M_rx_count;
    uint16_t head = _M_rx_head;
    for (uint16_t i = 0u; i < to_copy; ++i) {
      buf[i] = _M_rx_buf[head];
      head = static_cast<uint16_t> ((head + 1u) % MockMaxRxBytes);
    }
    _M_rx_head = head;
    _M_rx_count -= to_copy;
    *filled = to_copy;
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] zzs::kernel::Status
  write_bytes (const uint8_t *buf, uint16_t len) noexcept override {
    if (buf == nullptr && len > 0u)
      return zzs::kernel::Status::OUT_OF_RANGE;
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    if (_M_write_error != zzs::kernel::Status::OK)
      return _M_write_error;
    if (_M_command_count < MockMaxCommands) {
      Command &cmd = _M_commands[_M_command_count++];
      cmd._M_len = len;
      for (uint16_t i = 0u; i < len; ++i)
        cmd._M_data[i] = buf[i];
    }
    _M_total_written += len;
    return zzs::kernel::Status::OK;
  }

  void
  delay_ms (uint32_t ms) noexcept override {
    _M_delay_calls += 1u;
    _M_delay_total_ms += ms;
  }

  [[nodiscard]] zzs::kernel::Status
  flush_rx () noexcept override {
    _M_rx_head = 0u;
    _M_rx_count = 0u;
    _M_stale_reads = 0u;
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] bool
  is_open () const noexcept override {
    return _M_open;
  }

  // -- test helpers --

  /** @brief Queue a byte sequence to be returned on subsequent read_bytes(). */
  void
  inject_response (const uint8_t *data, uint16_t len) noexcept {
    if (data == nullptr || len == 0u)
      return;
    uint16_t space = MockMaxRxBytes - _M_rx_count;
    uint16_t to_copy = (len < space) ? len : space;
    uint16_t tail = static_cast<uint16_t> ((_M_rx_head + _M_rx_count) % MockMaxRxBytes);
    for (uint16_t i = 0u; i < to_copy; ++i)
      _M_rx_buf[(tail + i) % MockMaxRxBytes] = data[i];
    _M_rx_count += to_copy;
  }

  /** @brief Queue a length-prefixed frame: [len_lo][len_hi][bytes...]. */
  void
  inject_frame (const uint8_t *bytes, uint16_t len) noexcept {
    if (len > WifiMaxFrameBytes)
      return;
    uint8_t prefix[2] = {static_cast<uint8_t> (len & 0xffu), static_cast<uint8_t> ((len >> 8) & 0xffu)};
    inject_response (prefix, 2u);
    inject_response (bytes, len);
  }

  /** @brief Set the read error returned by read_bytes(). */
  void
  set_read_error (zzs::kernel::Status err) noexcept {
    _M_read_error = err;
  }

  /** @brief Set the write error returned by write_bytes(). */
  void
  set_write_error (zzs::kernel::Status err) noexcept {
    _M_write_error = err;
  }

  /** @brief Open / close the stream. */
  void
  set_open (bool open) noexcept {
    _M_open = open;
  }

  /** @brief Number of write_bytes() calls recorded. */
  [[nodiscard]] uint16_t
  command_count () const noexcept {
    return _M_command_count;
  }

  /** @brief The nth recorded command's data. */
  [[nodiscard]] const uint8_t *
  command_data (uint16_t index) const noexcept {
    if (index < _M_command_count)
      return _M_commands[index]._M_data;
    return nullptr;
  }

  /** @brief The nth recorded command's length. */
  [[nodiscard]] uint16_t
  command_len (uint16_t index) const noexcept {
    if (index < _M_command_count)
      return _M_commands[index]._M_len;
    return 0u;
  }

  /** @brief Total bytes written. */
  [[nodiscard]] uint32_t
  total_written () const noexcept {
    return _M_total_written;
  }

  [[nodiscard]] uint32_t
  delay_calls () const noexcept {
    return _M_delay_calls;
  }

  [[nodiscard]] uint32_t
  delay_total_ms () const noexcept {
    return _M_delay_total_ms;
  }

  [[nodiscard]] uint32_t
  stale_reads () const noexcept {
    return _M_stale_reads;
  }

  /** @brief Number of bytes currently queued for reading. */
  [[nodiscard]] uint16_t
  rx_queued () const noexcept {
    return _M_rx_count;
  }

  void
  clear () noexcept {
    _M_rx_head = 0u;
    _M_rx_count = 0u;
    _M_command_count = 0u;
    _M_total_written = 0u;
    _M_delay_calls = 0u;
    _M_delay_total_ms = 0u;
    _M_stale_reads = 0u;
    _M_read_error = zzs::kernel::Status::OK;
    _M_write_error = zzs::kernel::Status::OK;
    for (uint16_t i = 0u; i < MockMaxCommands; ++i) {
      _M_commands[i]._M_len = 0u;
      for (uint16_t j = 0u; j < MockMaxRxBytes; ++j)
        _M_commands[i]._M_data[j] = 0u;
    }
    for (uint16_t i = 0u; i < MockMaxRxBytes; ++i)
      _M_rx_buf[i] = 0u;
  }

private:
  struct Command {
    uint8_t _M_data[MockMaxRxBytes] = {};
    uint16_t _M_len = 0u;
  };

  uint8_t _M_rx_buf[MockMaxRxBytes] = {};
  uint16_t _M_rx_head = 0u;
  uint16_t _M_rx_count = 0u;
  Command _M_commands[MockMaxCommands] = {};
  uint16_t _M_command_count = 0u;
  uint32_t _M_total_written = 0u;
  uint32_t _M_delay_calls = 0u;
  uint32_t _M_delay_total_ms = 0u;
  uint32_t _M_stale_reads = 0u;
  bool _M_open = true;
  zzs::kernel::Status _M_read_error = zzs::kernel::Status::OK;
  zzs::kernel::Status _M_write_error = zzs::kernel::Status::OK;
};

} // namespace test
