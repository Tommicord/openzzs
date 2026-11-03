/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

#pragma once

#include "ak0991x_reg.h"
#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_status.hpp"
#include "kernel/sensor-driver/sensors/sensor_driver.hpp"

#include <cstdint>

/** @brief Raw 16-bit little-endian sample from a byte pair. */
[[nodiscard]] inline constexpr int16_t
ak0991x_raw16 (const uint8_t *p) noexcept {
  return static_cast<int16_t> (static_cast<uint16_t> (p[0]) | (static_cast<uint16_t> (p[1]) << 8));
}

/**
 * @brief AK0991x (AKM) 3-axis magnetometer driver (MAG_X/Y/Z).
 *
 * Reads the 6-byte magnetometer data block and converts raw counts to Q15
 * milligauss using the Q15 sensitivity multiplier from ak0991x_reg.h.
 * Supports AK09911/AK09912/AK09915/AK09916/AK09918 variants.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors {

/** @brief AK0991x magnetometer driver (MAG_X/Y/Z). */
class Ak0991xMagDriver : public SensorDriver {
public:
  /** @brief Construct with a non-owning bus pointer and an optional clock
   *         source callback for timestamps. */
  explicit Ak0991xMagDriver (bus::BusInterface *bus, uint64_t (*clock) (void) = nullptr) noexcept
      : SensorDriver (bus, clock) {
  }

  [[nodiscard]] zzs::kernel::Status
  init () noexcept override {
    uint8_t who = 0;
    zzs::kernel::Status st = bus_read8 (ak0991x::RegWia, who);
    if (!zzs::kernel::status_ok (st))
      return st;
    if (who != ak0991x::WiaValue)
      return zzs::kernel::Status::BAD_TYPE;
    _M_device_id = who;

    /* Soft reset. */
    st = bus_write8 (ak0991x::RegCntl3, ak0991x::Cntl3SoftReset);
    if (!zzs::kernel::status_ok (st))
      return st;
    _M_bus->delay_ms (1u);

    /* Continuous measurement mode 100 Hz. */
    st = bus_write8 (ak0991x::RegCntl2, ak0991x::Cntl2ModeCont100Hz);
    if (!zzs::kernel::status_ok (st))
      return st;

    _M_axis = 0;
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] zzs::kernel::Status
  read_sample (sensor::Sample &out) noexcept override {
    /* Round-robin across enabled axes: X, Y, Z (per-instance state). */
    while (_M_axis < 3u && !_M_enabled[_M_axis])
      _M_axis = static_cast<uint8_t> ((_M_axis + 1u) % 3u);
    if (_M_axis >= 3u)
      return zzs::kernel::Status::STALE;

    uint8_t st1 = 0;
    zzs::kernel::Status st = bus_read8 (ak0991x::RegSt1, st1);
    if (!zzs::kernel::status_ok (st))
      return st;

    uint8_t ready = ak0991x::RegSt1; /* DRDY is bit 0 of ST1 */
    if ((st1 & 0x01u) == 0u)
      return zzs::kernel::Status::STALE;

    uint8_t data[ak0991x::DataBlockBytes] = {0};
    st = bus_read (ak0991x::RegHxl, data, ak0991x::DataBlockBytes);
    if (!zzs::kernel::status_ok (st))
      return st;

    const int16_t raw = ak0991x_raw16 (&data[_M_axis * 2u]);
    const int32_t q15 = zzs::kernel::sensor::driver::fixed::q15_from_raw (raw);
    const int32_t phys_q15 = zzs::kernel::sensor::driver::fixed::q15_mul (q15, ak0991x::MagSensQ15MilligaussPerLsb);

    out._M_channel = static_cast<sensor::Channel> (static_cast<uint8_t> (sensor::Channel::MAG_X) + _M_axis);
    out._M_sample_q15 = zzs::kernel::sensor::driver::fixed::q15_to_raw (phys_q15);
    out._M_timestamp_ns = timestamp_ns ();

    _M_axis = static_cast<uint8_t> ((_M_axis + 1u) % 3u);
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] zzs::kernel::Status
  set_channel_enabled (sensor::Channel ch, bool enabled) noexcept override {
    const uint8_t idx = static_cast<uint8_t> (ch) - static_cast<uint8_t> (sensor::Channel::MAG_X);
    if (idx >= 3u)
      return zzs::kernel::Status::OUT_OF_RANGE;
    _M_enabled[idx] = enabled;
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] uint32_t
  channel_rate_hz (sensor::Channel ch) noexcept override {
    (void) ch;
    return ak0991x::OdrHz;
  }

  [[nodiscard]] uint8_t
  device_id () const noexcept override {
    return _M_device_id;
  }

  [[nodiscard]] bool
  self_test () noexcept override {
    /* AK0991x has no dedicated self-test register; verify by reading data. */
    uint8_t st1 = 0;
    if (!zzs::kernel::status_ok (bus_read8 (ak0991x::RegSt1, st1)))
      return false;
    uint8_t data[ak0991x::DataBlockBytes] = {0};
    return zzs::kernel::status_ok (bus_read (ak0991x::RegHxl, data, ak0991x::DataBlockBytes));
  }

private:
  /** @brief Per-instance round-robin axis cursor (X/Y/Z). */
  uint8_t _M_axis = 0;
  /** @brief Whether each channel is enabled. */
  bool _M_enabled[3] = {true, true, true};
};

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors
