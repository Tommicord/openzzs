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
#include "kernel/sensor-driver/sensors/sensor_driver.hpp"
#include "lis2de12_reg.h"

#include <cstdint>

/** @brief Raw 16-bit little-endian sample from a byte pair. */
[[nodiscard]] inline constexpr int16_t
lis2de12_raw16 (const uint8_t *p) noexcept {
  return static_cast<int16_t> (static_cast<uint16_t> (p[0]) | (static_cast<uint16_t> (p[1]) << 8));
}

/**
 * @brief ST LIS2DE12 3-axis accelerometer driver (ACCEL_X/Y/Z).
 *
 * Reads the 6-byte accelerometer data block and converts raw counts to Q15
 * using the Q15 sensitivity multiplier from lis2de12_reg.h.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors {

/** @brief Number of bytes in a 3-axis data block (X/Y/Z, 2 bytes each). */
inline constexpr uint32_t Lis2de12DataBlockBytes = 6u;

/** @brief LIS2DE12 accelerometer ODR: 100 Hz (CTRL_REG1 ODR = 0101). */
inline constexpr uint8_t Lis2de12Odr100Hz = 0x5;

/** @brief LIS2DE12 accelerometer driver (ACCEL_X/Y/Z). */
class Lis2de12AccelDriver : public SensorDriver {
public:
  /** @brief Construct with a non-owning bus pointer and an optional clock
   *         source callback for timestamps. */
  explicit Lis2de12AccelDriver (bus::BusInterface *bus, uint64_t (*clock) (void) = nullptr) noexcept
      : SensorDriver (bus, clock) {
  }

  [[nodiscard]] zzs::kernel::Status
  init () noexcept override {
    uint8_t who = 0;
    zzs::kernel::Status st = bus_read8 (lis2de12::RegWhoAmI, who);
    if (!zzs::kernel::status_ok (st))
      return st;
    if (who != lis2de12::WhoAmIValue)
      return zzs::kernel::Status::BAD_TYPE;
    _M_device_id = who;

    /* Enable BDU, set accel ODR 100 Hz, full scale ±2 g. */
    st = bus_write8 (lis2de12::RegCtrlReg4, static_cast<uint8_t> (lis2de12::CtrlReg4Bdu | lis2de12::CtrlReg4Fs2g));
    if (!zzs::kernel::status_ok (st))
      return st;
    st = bus_write8 (lis2de12::RegCtrlReg1, lis2de12::CtrlReg1Odr100Hz);
    if (!zzs::kernel::status_ok (st))
      return st;

    _M_cache_count = 0u;
    _M_cache_idx = 0u;
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] zzs::kernel::Status
  read_sample (sensor::Sample &out) noexcept override {
    uint16_t filled = 0u;
    return read_samples (&out, 1u, &filled) == zzs::kernel::Status::OK && filled > 0u ? zzs::kernel::Status::OK
                                                                                      : zzs::kernel::Status::STALE;
  }

  [[nodiscard]] zzs::kernel::Status
  read_samples (sensor::Sample *out, uint16_t max_samples, uint16_t *filled) noexcept override {
    if (out == nullptr || filled == nullptr)
      return zzs::kernel::Status::OUT_OF_RANGE;
    if (max_samples == 0u) {
      *filled = 0u;
      return zzs::kernel::Status::OK;
    }

    if (_M_cache_count == 0u) {
      zzs::kernel::Status st = _M_refresh_cache ();
      if (!zzs::kernel::status_ok (st))
        return st;
      if (_M_cache_count == 0u)
        return zzs::kernel::Status::STALE;
    }

    uint16_t n = 0u;
    while (n < max_samples && _M_cache_idx < _M_cache_count) {
      out[n] = _M_cache[_M_cache_idx];
      ++_M_cache_idx;
      ++n;
    }
    if (_M_cache_idx >= _M_cache_count) {
      _M_cache_idx = 0u;
      _M_cache_count = 0u;
    }
    *filled = n;
    return (n > 0u) ? zzs::kernel::Status::OK : zzs::kernel::Status::STALE;
  }

  [[nodiscard]] zzs::kernel::Status
  set_channel_enabled (sensor::Channel ch, bool enabled) noexcept override {
    const uint8_t idx = static_cast<uint8_t> (ch) - static_cast<uint8_t> (sensor::Channel::ACCEL_X);
    if (idx >= 3u)
      return zzs::kernel::Status::OUT_OF_RANGE;
    _M_enabled[idx] = enabled;
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] uint32_t
  channel_rate_hz (sensor::Channel ch) noexcept override {
    (void) ch;
    return 100u;
  }

  [[nodiscard]] uint8_t
  device_id () const noexcept override {
    return _M_device_id;
  }

  [[nodiscard]] bool
  self_test () noexcept override {
    uint8_t status = 0;
    if (!zzs::kernel::status_ok (bus_read8 (lis2de12::RegStatusReg, status)))
      return false;
    uint8_t data[Lis2de12DataBlockBytes] = {0};
    if (!zzs::kernel::status_ok (bus_read (lis2de12::RegOutXL, data, Lis2de12DataBlockBytes)))
      return false;
    const int16_t baseline = lis2de12_raw16 (&data[0]);

    uint8_t ctrl5 = 0;
    if (!zzs::kernel::status_ok (bus_read8 (lis2de12::RegCtrlReg5, ctrl5)))
      return false;
    const uint8_t ctrl5_st = static_cast<uint8_t> (ctrl5 | lis2de12::CtrlReg5St);
    zzs::kernel::Status st = bus_write8 (lis2de12::RegCtrlReg5, ctrl5_st);
    if (!zzs::kernel::status_ok (st)) {
      (void) bus_write8 (lis2de12::RegCtrlReg5, ctrl5);
      return false;
    }
    _M_bus->delay_ms (10u);

    if (!zzs::kernel::status_ok (bus_read8 (lis2de12::RegStatusReg, status))) {
      (void) bus_write8 (lis2de12::RegCtrlReg5, ctrl5);
      return false;
    }
    if (!zzs::kernel::status_ok (bus_read (lis2de12::RegOutXL, data, Lis2de12DataBlockBytes))) {
      (void) bus_write8 (lis2de12::RegCtrlReg5, ctrl5);
      return false;
    }
    const int16_t after = lis2de12_raw16 (&data[0]);

    (void) bus_write8 (lis2de12::RegCtrlReg5, ctrl5);

    const int32_t delta = static_cast<int32_t> (after) - static_cast<int32_t> (baseline);
    return delta != 0;
  }

private:
  [[nodiscard]] zzs::kernel::Status
  _M_refresh_cache () noexcept {
    uint8_t status = 0;
    zzs::kernel::Status st = bus_read8 (lis2de12::RegStatusReg, status);
    if (!zzs::kernel::status_ok (st))
      return st;
    if ((status & lis2de12::StatusZyxda) == 0u)
      return zzs::kernel::Status::STALE;

    uint8_t data[Lis2de12DataBlockBytes] = {0};
    st = bus_read (lis2de12::RegOutXL, data, Lis2de12DataBlockBytes);
    if (!zzs::kernel::status_ok (st))
      return st;

    const uint64_t ts = timestamp_ns ();
    uint8_t count = 0u;
    for (uint8_t axis = 0u; axis < 3u; ++axis) {
      if (!_M_enabled[axis])
        continue;
      const int16_t raw = lis2de12_raw16 (&data[axis * 2u]);
      const int32_t q15 = zzs::kernel::sensor::driver::fixed::q15_from_raw (raw);
      const int32_t phys_q15 = zzs::kernel::sensor::driver::fixed::q15_mul (q15, lis2de12::AccelSensQ15MgPerLsb);

      _M_cache[count]._M_channel =
          static_cast<sensor::Channel> (static_cast<uint8_t> (sensor::Channel::ACCEL_X) + axis);
      _M_cache[count]._M_sample_q15 = zzs::kernel::sensor::driver::fixed::q15_to_raw (phys_q15);
      _M_cache[count]._M_timestamp_ns = ts;
      ++count;
    }
    _M_cache_count = count;
    _M_cache_idx = 0u;
    return zzs::kernel::Status::OK;
  }

  sensor::Sample _M_cache[3] = {};
  uint8_t _M_cache_count = 0u;
  uint8_t _M_cache_idx = 0u;
  bool _M_enabled[3] = {true, true, true};
};

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors
