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
#include "veml7700_reg.h"

#include <cstdint>

/**
 * @brief VEML7700 ambient light sensor driver (LIGHT channel, Q11 lux).
 *
 * Configures gain and integration time, reads the 16-bit ALS_DATA register
 * and converts the raw count to Q11 lux using an integer-only sensitivity
 * multiplier.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors {

/** @brief Number of bytes in the ALS_DATA register read. */
inline constexpr uint32_t Veml7700AlsDataBytes = 2u;

/** @brief Raw 16-bit big-endian sample from a byte pair (Vishay convention). */
[[nodiscard]] inline constexpr uint16_t
veml7700_raw16 (const uint8_t *p) noexcept {
  return static_cast<uint16_t> ((static_cast<uint16_t> (p[0]) << 8) | static_cast<uint16_t> (p[1]));
}

/** @brief VEML7700 light sensor driver (LIGHT channel). */
class Veml7700LightDriver : public SensorDriver {
public:
  /** @brief Construct with a non-owning bus pointer and an optional clock
   *         source callback for timestamps. */
  explicit Veml7700LightDriver (bus::BusInterface *bus, uint64_t (*clock) (void) = nullptr) noexcept
      : SensorDriver (bus, clock) {
  }

  [[nodiscard]] zzs::kernel::Status
  init () noexcept override {
    /* ALS_CONF: gain 1x, integration time 100 ms, shutdown=0 (on),
     * interrupt disabled. */
    const uint16_t conf = static_cast<uint16_t> (veml7700::AlsConfGain1x | veml7700::AlsConfInteg100ms);
    const uint8_t conf_hi = static_cast<uint8_t> (conf >> 8);
    const uint8_t conf_lo = static_cast<uint8_t> (conf & 0xFF);

    /* VEML7700 16-bit registers are written high byte then low byte.
     * Write the command/register address followed by the two data bytes. */
    uint8_t write_buf[3] = {static_cast<uint8_t> (veml7700::RegAlsConf), conf_hi, conf_lo};
    zzs::kernel::Status st = bus_write (veml7700::RegAlsConf, write_buf, Veml7700AlsDataBytes + 1u);
    if (!zzs::kernel::status_ok (st))
      return st;

    _M_device_id = veml7700::I2cAddr;
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] zzs::kernel::Status
  read_sample (sensor::Sample &out) noexcept override {
    uint8_t buf[Veml7700AlsDataBytes] = {0};
    zzs::kernel::Status st = bus_read (veml7700::RegAlsData, buf, Veml7700AlsDataBytes);
    if (!zzs::kernel::status_ok (st))
      return st;

    const uint16_t raw = veml7700_raw16 (buf);
    /* Lux in Q11: raw_count * sensitivity_q11 (already Q11). */
    int32_t lux_raw = 0;
    if (__builtin_mul_overflow (static_cast<int32_t> (raw), veml7700::LuxSensQ11PerCount, &lux_raw))
      lux_raw = raw > 0 ? INT32_MAX : INT32_MIN;
    const zzs::kernel::sensor::driver::fixed::q11_t lux_q11 =
        static_cast<zzs::kernel::sensor::driver::fixed::q11_t> (lux_raw);

    out._M_channel = sensor::Channel::LIGHT;
    out._M_sample_q15 = zzs::kernel::sensor::driver::fixed::q11_to_q15 (lux_q11);
    out._M_timestamp_ns = timestamp_ns ();
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] zzs::kernel::Status
  set_channel_enabled (sensor::Channel ch, bool enabled) noexcept override {
    if (ch != sensor::Channel::LIGHT)
      return zzs::kernel::Status::OUT_OF_RANGE;
    _M_enabled = enabled;
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] uint32_t
  channel_rate_hz (sensor::Channel ch) noexcept override {
    (void) ch;
    return 1u;
  }

  [[nodiscard]] uint8_t
  device_id () const noexcept override {
    return _M_device_id;
  }

  [[nodiscard]] bool
  self_test () noexcept override {
    uint8_t buf[Veml7700AlsDataBytes] = {0};
    if (!zzs::kernel::status_ok (bus_read (veml7700::RegAlsData, buf, Veml7700AlsDataBytes)))
      return false;
    const uint16_t raw = veml7700_raw16 (buf);
    return raw != 0u && raw != 0xFFFFu;
  }

private:
  /** @brief Whether the LIGHT channel is enabled. */
  bool _M_enabled = true;
};

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors
