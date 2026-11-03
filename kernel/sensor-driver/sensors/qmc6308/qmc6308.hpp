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
#include "qmc6308_reg.h"

#include <cstdint>

/** @brief Raw 16-bit little-endian sample from a byte pair. */
[[nodiscard]] inline constexpr int16_t
qmc6308_raw16 (const uint8_t *p) noexcept {
  return static_cast<int16_t> (static_cast<uint16_t> (p[0]) | (static_cast<uint16_t> (p[1]) << 8));
}

/**
 * @brief QST QMC6308 3-axis magnetometer driver (MAG_X/Y/Z).
 *
 * The QMC6308 is a common magnetometer in budget phones (Infinix, Tecno,
 * itel).  Reads the 6-byte magnetometer data block and converts raw counts
 * to Q15 milligauss using the Q15 sensitivity multiplier from qmc6308_reg.h.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors {

/** @brief Number of bytes in a 3-axis data block (X/Y/Z, 2 bytes each). */
inline constexpr uint32_t Qmc6308DataBlockBytes = 6u;

/** @brief QMC6308 magnetometer driver (MAG_X/Y/Z). */
class Qmc6308MagDriver : public SensorDriver {
public:
  /** @brief Construct with a non-owning bus pointer and an optional clock
   *         source callback for timestamps. */
  explicit Qmc6308MagDriver (bus::BusInterface *bus, uint64_t (*clock) (void) = nullptr) noexcept
      : SensorDriver (bus, clock) {
  }

  [[nodiscard]] zzs::kernel::Status
  init () noexcept override {
    uint8_t who = 0;
    zzs::kernel::Status st = bus_read8 (qmc6308::RegChipId, who);
    if (!zzs::kernel::status_ok (st))
      return st;
    if (who != qmc6308::ChipIdValue)
      return zzs::kernel::Status::BAD_TYPE;
    _M_device_id = who;

    /* Soft reset lives in CTRL2 bit 7, then wait for the POR time (250 us). */
    st = bus_write8 (qmc6308::RegCtrl2, qmc6308::Ctrl2SoftReset);
    if (!zzs::kernel::status_ok (st))
      return st;
    _M_bus->delay_ms (1u);

    /* Continuous mode + 100 Hz ODR + 8x oversampling on both channels. */
    st = bus_write8 (qmc6308::RegCtrl1, static_cast<uint8_t> (qmc6308::Ctrl1ModeNormal | qmc6308::Ctrl1Odr100Hz
                                                              | qmc6308::Ctrl1Osr1_8 | qmc6308::Ctrl1Osr2_8));
    if (!zzs::kernel::status_ok (st))
      return st;

    /* SET/RESET enabled, ±8 gauss (3750 LSB/gauss). */
    st = bus_write8 (qmc6308::RegCtrl2, static_cast<uint8_t> (qmc6308::Ctrl2SetResetOn | qmc6308::Ctrl2Range8G));
    if (!zzs::kernel::status_ok (st))
      return st;

    _M_axis = 0;
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] zzs::kernel::Status
  read_sample (sensor::Sample &out) noexcept override {
    /* Round-robin across enabled axes: X, Y, Z (per-instance state). */
    bool found = false;
    for (uint8_t i = 0; i < 3u; ++i) {
      const uint8_t candidate = static_cast<uint8_t> ((_M_axis + i) % 3u);
      if (_M_enabled[candidate]) {
        _M_axis = candidate;
        found = true;
        break;
      }
    }
    if (!found)
      return zzs::kernel::Status::STALE;

    uint8_t status = 0;
    zzs::kernel::Status st = bus_read8 (qmc6308::RegStatus, status);
    if (!zzs::kernel::status_ok (st))
      return st;

    if ((status & qmc6308::StatusDataReady) == 0u)
      return zzs::kernel::Status::STALE;

    uint8_t data[Qmc6308DataBlockBytes] = {0};
    st = bus_read (qmc6308::RegDataXlsb, data, Qmc6308DataBlockBytes);
    if (!zzs::kernel::status_ok (st))
      return st;

    const int16_t raw = qmc6308_raw16 (&data[_M_axis * 2u]);
    const int32_t q15 = zzs::kernel::sensor::driver::fixed::q15_from_raw (raw);
    const int32_t phys_q15 = zzs::kernel::sensor::driver::fixed::q15_mul (q15, qmc6308::MagSensQ15MilligaussPerLsb);

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
    return qmc6308::OdrHz;
  }

  [[nodiscard]] uint8_t
  device_id () const noexcept override {
    return _M_device_id;
  }

  [[nodiscard]] bool
  self_test () noexcept override {
    uint8_t status = 0;
    if (!zzs::kernel::status_ok (bus_read8 (qmc6308::RegStatus, status)))
      return false;
    uint8_t data[Qmc6308DataBlockBytes] = {0};
    return zzs::kernel::status_ok (bus_read (qmc6308::RegDataXlsb, data, Qmc6308DataBlockBytes));
  }

private:
  /** @brief Per-instance round-robin axis cursor (X/Y/Z). */
  uint8_t _M_axis = 0;
  /** @brief Whether each channel is enabled. */
  bool _M_enabled[3] = {true, true, true};
};

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors
