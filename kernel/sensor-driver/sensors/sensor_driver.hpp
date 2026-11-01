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

#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_status.hpp"
#include "kernel/sensor-driver/bus/bus_interface.hpp"
#include "kernel/sensor-driver/fixed/q15.hpp"

#include <cstdint>

/**
 * @brief Common sensor driver interface.
 *
 * Every concrete driver (LSM6DSO, LIS3MDL, LSM303, TSL2561, VEML7700,
 * ICS43434) derives from SensorDriver and implements init, read_sample and the
 * channel control methods.  All operations are `noexcept`, `[[nodiscard]]`,
 * `constexpr` where possible, integer-only, and allocate nothing.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors {

/** @brief Abstract base class every sensor driver implements. */
class SensorDriver {
public:
  /** @brief Construct with a non-owning bus pointer and an optional clock
   *         source callback used to timestamp samples. */
  explicit SensorDriver (bus::BusInterface *bus, uint64_t (*clock) (void) = nullptr) noexcept
      : _M_bus (bus), _M_clock (clock) {
  }

  /** @brief Virtual destructor, defaulted. */
  virtual ~SensorDriver () = default;

  /** @brief Initialize the device, verify WHO_AM_I, configure registers. */
  [[nodiscard]] virtual zzs::kernel::Status
  init () noexcept = 0;

  /** @brief Read the next available sample into `out`. Returns STALE when no
   *         new sample is ready and NOT_READY while the device warms up. */
  [[nodiscard]] virtual zzs::kernel::Status
  read_sample (sensor::Sample &out) noexcept = 0;

  /** @brief Enable or disable a channel on this device. */
  [[nodiscard]] virtual zzs::kernel::Status
  set_channel_enabled (sensor::Channel ch, bool enabled) noexcept = 0;

  /** @brief Return the configured sample rate in Hz for `ch`. */
  [[nodiscard]] virtual uint32_t
  channel_rate_hz (sensor::Channel ch) noexcept = 0;

  /** @brief Return the device WHO_AM_I value read at init time. */
  [[nodiscard]] virtual uint8_t
  device_id () const noexcept = 0;

  /** @brief Run the built-in self-test and return true when it passes. */
  [[nodiscard]] virtual bool
  self_test () noexcept = 0;

protected:
  /** @brief Bus the driver talks to; non-owning, set by the constructor. */
  bus::BusInterface *_M_bus = nullptr;

  /** @brief Device WHO_AM_I value captured during init. */
  uint8_t _M_device_id = 0;

  /** @brief Clock source callback for sample timestamps (may be nullptr). */
  uint64_t (*_M_clock) (void) = nullptr;

  /** @brief Return the current timestamp: the per-instance clock callback when
   *         one was supplied, otherwise the global kernel clock source, or 0
   *         when neither is available. */
  [[nodiscard]] inline uint64_t
  timestamp_ns () const noexcept {
    if (_M_clock != nullptr)
      return _M_clock ();
    return zzs::kernel::clock::clock_source_now_ns ();
  }

  /** @brief Read up to `max_samples` into `out`; `*filled` is set to the
   *         number of samples written.  Returns OK when at least one sample
   *         was produced, STALE when none are ready, NOT_READY while warming
   *         up, or a bus error.  Default implementation loops `read_sample()`;
   *         concrete drivers may override to burst multiple axes from a single
   *         bus transaction (lower level, fewer transactions, less memory
   *         copying). */
  [[nodiscard]] virtual zzs::kernel::Status
  read_samples (sensor::Sample *out, uint16_t max_samples, uint16_t *filled) noexcept {
    if (out == nullptr || filled == nullptr)
      return zzs::kernel::Status::OUT_OF_RANGE;
    if (max_samples == 0u) {
      *filled = 0u;
      return zzs::kernel::Status::OK;
    }
    uint16_t n = 0u;
    while (n < max_samples) {
      sensor::Sample s;
      zzs::kernel::Status st = read_sample (s);
      if (!zzs::kernel::status_ok (st)) {
        if (st == zzs::kernel::Status::STALE || st == zzs::kernel::Status::NOT_READY)
          break;
        return st;
      }
      out[n++] = s;
    }
    *filled = n;
    return (n > 0u) ? zzs::kernel::Status::OK : zzs::kernel::Status::STALE;
  }

  /** @brief Read `len` bytes from `reg` into `buf`. */
  [[nodiscard]] inline zzs::kernel::Status
  bus_read (uint8_t reg, uint8_t *buf, uint16_t len) noexcept {
    return _M_bus == nullptr ? zzs::kernel::Status::INTERNAL : _M_bus->read_reg (reg, buf, len);
  }

  /** @brief Write `len` bytes from `buf` to `reg`. */
  [[nodiscard]] inline zzs::kernel::Status
  bus_write (uint8_t reg, const uint8_t *buf, uint16_t len) noexcept {
    return _M_bus == nullptr ? zzs::kernel::Status::INTERNAL : _M_bus->write_reg (reg, buf, len);
  }

  /** @brief Read one register value. */
  [[nodiscard]] inline zzs::kernel::Status
  bus_read8 (uint8_t reg, uint8_t &value) noexcept {
    return bus_read (reg, &value, 1u);
  }

  /** @brief Write one register value. */
  [[nodiscard]] inline zzs::kernel::Status
  bus_write8 (uint8_t reg, uint8_t value) noexcept {
    return bus_write (reg, &value, 1u);
  }

  /** @brief Read-modify-write a single register bitmask. */
  [[nodiscard]] inline zzs::kernel::Status
  bus_update8 (uint8_t reg, uint8_t set_mask, uint8_t clear_mask) noexcept {
    uint8_t value = 0;
    zzs::kernel::Status st = bus_read8 (reg, value);
    if (!zzs::kernel::status_ok (st))
      return st;
    value = static_cast<uint8_t> ((value & ~clear_mask) | set_mask);
    return bus_write8 (reg, value);
  }
};

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors
