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

#include "adxl345_reg.h"
#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_status.hpp"
#include "kernel/sensor-driver/sensors/sensor_driver.hpp"

#include <cstdint>

/** @brief Raw 16-bit little-endian sample from a byte pair. */
[[nodiscard]] inline constexpr int16_t
adxl345_raw16 (const uint8_t *p) noexcept {
    return static_cast<int16_t> (static_cast<uint16_t> (p[0]) | (static_cast<uint16_t> (p[1]) << 8));
}

/**
 * @brief ADXL345 3-axis accelerometer driver (ACCEL_X/Y/Z).
 *
 * Reads the 6-byte accelerometer data block and converts raw counts to Q15
 * using the Q15 sensitivity multiplier from adxl345_reg.h.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors {

/** @brief Number of bytes in a 3-axis data block (X/Y/Z, 2 bytes each). */
inline constexpr uint32_t Adxl345DataBlockBytes = 6u;

/** @brief ADXL345 accelerometer ODR: 100 Hz (BW_RATE rate code 0x0A). */
inline constexpr uint8_t Adxl345Odr100Hz = 0x0A;

/** @brief ADXL345 accelerometer driver (ACCEL_X/Y/Z). */
class Adxl345AccelDriver : public SensorDriver {
  public:
    /** @brief Construct with a non-owning bus pointer and an optional clock
     *         source callback for timestamps. */
    explicit Adxl345AccelDriver (bus::BusInterface *bus, uint64_t (*clock) (void) = nullptr) noexcept
        : SensorDriver (bus, clock) {
    }

    [[nodiscard]] zzs::kernel::Status
    init () noexcept override {
        uint8_t who = 0;
        zzs::kernel::Status st = bus_read8 (adxl345::RegDevid, who);
        if (!zzs::kernel::status_ok (st))
            return st;
        if (who != adxl345::DevidValue)
            return zzs::kernel::Status::BAD_TYPE;
        _M_device_id = who;

        /* Full resolution mode, ±2 g range. */
        st = bus_write8 (adxl345::RegDataFormat,
                         static_cast<uint8_t> (adxl345::DataFormatFullRes | adxl345::DataFormatFs2g));
        if (!zzs::kernel::status_ok (st))
            return st;

        /* Output data rate 100 Hz. */
        st = bus_write8 (adxl345::RegBwRate, adxl345::BwRateOdr100Hz);
        if (!zzs::kernel::status_ok (st))
            return st;

        /* Enter measurement mode. */
        st = bus_write8 (adxl345::RegPowerCtl, adxl345::PowerCtlMeasure);
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
        if (!zzs::kernel::status_ok (bus_read8 (adxl345::RegDataFormat, status)))
            return false;
        uint8_t data[Adxl345DataBlockBytes] = {0};
        if (!zzs::kernel::status_ok (bus_read (adxl345::RegDataX0, data, Adxl345DataBlockBytes)))
            return false;
        const int16_t baseline = adxl345_raw16 (&data[0]);

        /* Enable self-test bit in DATA_FORMAT. */
        uint8_t fmt = 0;
        if (!zzs::kernel::status_ok (bus_read8 (adxl345::RegDataFormat, fmt)))
            return false;
        const uint8_t fmt_st = static_cast<uint8_t> (fmt | 0x80u);
        zzs::kernel::Status st = bus_write8 (adxl345::RegDataFormat, fmt_st);
        if (!zzs::kernel::status_ok (st)) {
            (void) bus_write8 (adxl345::RegDataFormat, fmt);
            return false;
        }
        _M_bus->delay_ms (10u);

        if (!zzs::kernel::status_ok (bus_read (adxl345::RegDataX0, data, Adxl345DataBlockBytes))) {
            (void) bus_write8 (adxl345::RegDataFormat, fmt);
            return false;
        }
        const int16_t after = adxl345_raw16 (&data[0]);

        (void) bus_write8 (adxl345::RegDataFormat, fmt);

        const int32_t delta = static_cast<int32_t> (after) - static_cast<int32_t> (baseline);
        return delta != 0;
    }

  private:
    [[nodiscard]] zzs::kernel::Status
    _M_refresh_cache () noexcept {
        uint8_t data[Adxl345DataBlockBytes] = {0};
        zzs::kernel::Status st = bus_read (adxl345::RegDataX0, data, Adxl345DataBlockBytes);
        if (!zzs::kernel::status_ok (st))
            return st;

        const uint64_t ts = timestamp_ns ();
        uint8_t count = 0u;
        for (uint8_t axis = 0u; axis < 3u; ++axis) {
            if (!_M_enabled[axis])
                continue;
            const int16_t raw = adxl345_raw16 (&data[axis * 2u]);
            const int32_t q15 = zzs::kernel::sensor::driver::fixed::q15_from_raw (raw);
            const int32_t phys_q15 = zzs::kernel::sensor::driver::fixed::q15_mul (q15, adxl345::AccelSensQ15MgPerLsb);

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
