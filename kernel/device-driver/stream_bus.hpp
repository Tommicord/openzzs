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

#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief Stream-oriented byte bus for device drivers (GNSS receivers, WiFi
 *        modules, cellular modems) that communicate over UART/USB/ SPI data
 *        channels rather than register-mapped I2C/SPI.
 *
 * Mirrors `kernel/sensor-driver/bus/bus_interface.hpp` but for byte streams:
 * non-owning, `noexcept`, `[[nodiscard]]`, integer-only, no heap, no RTTI,
 * no exceptions.
 * @{
 */
namespace zzs::kernel::device::driver {

/** @brief Abstract byte-stream bus implemented by UART, USB-CDC, or SPI-data
 *         platform adapters. */
class StreamBusInterface {
public:
  /** @brief Default constructor. */
  StreamBusInterface () = default;

  /** @brief Virtual destructor, defaulted. */
  virtual ~StreamBusInterface () = default;

  StreamBusInterface (const StreamBusInterface &) = delete;
  StreamBusInterface &operator= (const StreamBusInterface &) = delete;
  StreamBusInterface (StreamBusInterface &&) = delete;
  StreamBusInterface &operator= (StreamBusInterface &&) = delete;

  /** @brief Read up to `max_bytes` into `buf`; `*filled` is set to the number
   *         of bytes actually read. Returns OK when at least one byte was read,
   *         STALE when none are available, or a bus error. Non-blocking. */
  [[nodiscard]] virtual Status
  read_bytes (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept = 0;

  /** @brief Write `len` bytes from `buf`. Returns OK when all bytes were
   *         accepted by the underlying driver, or a bus error. */
  [[nodiscard]] virtual Status
  write_bytes (const uint8_t *buf, uint16_t len) noexcept = 0;

  /** @brief Block for at least `ms` milliseconds. */
  virtual void
  delay_ms (uint32_t ms) noexcept = 0;

  /** @brief Flush any buffered receive data (discard pending bytes). */
  [[nodiscard]] virtual Status
  flush_rx () noexcept = 0;

  /** @brief Return true when the underlying stream is open/ready. */
  [[nodiscard]] virtual bool
  is_open () const noexcept = 0;
};

/** @} */
} // namespace zzs::kernel::device::driver
