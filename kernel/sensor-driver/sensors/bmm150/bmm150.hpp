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

#include "bmm150_reg.h"
#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_status.hpp"
#include "kernel/sensor-driver/sensors/sensor_driver.hpp"

#include <cstdint>

/**
 * @brief BMM150 (Bosch) 3-axis magnetometer driver (MAG_X/Y/Z).
 *
 * Reads the 8-byte magnetometer data block (X/Y/Z + hall resistance) and
 * converts raw counts to Q15 milligauss.  BMM150 data is 13-bit signed
 * (bits [15:3] of each 16-bit axis word), so each axis value is arithmetic
 * right-shifted by 3 and sign-extended before Q15 conversion.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors {

/** @brief Number of bytes in a BMM150 data block (X/Y/Z + R, 2 bytes each). */
inline constexpr uint32_t Bmm150DataBlockBytes = 8u;

/** @brief Raw 16-bit little-endian sample from a byte pair. */
[[nodiscard]] inline constexpr int16_t
bmm150_raw16 (const uint8_t *p) noexcept {
    return static_cast<int16_t> (static_cast<uint16_t> (p[0]) | (static_cast<uint16_t> (p[1]) << 8));
}

/** @brief BMM150 magnetometer driver (MAG_X/Y/Z). */
class Bmm150MagDriver : public SensorDriver {
  public:
    /** @brief Construct with a non-owning bus pointer and an optional clock
     *         source callback for timestamps. */
    explicit Bmm150MagDriver (bus::BusInterface *bus, uint64_t (*clock) (void) = nullptr) noexcept
        : SensorDriver (bus, clock) {
    }

    [[nodiscard]] zzs::kernel::Status
    init () noexcept override {
        uint8_t who = 0;
        zzs::kernel::Status st = bus_read8 (bmm150::RegChipId, who);
        if (!zzs::kernel::status_ok (st))
            return st;
        if (who != bmm150::ChipIdValue)
            return zzs::kernel::Status::BAD_TYPE;
        _M_device_id = who;

        /* Set normal mode (continuous measurement). */
        st = bus_write8 (bmm150::RegPwrCntl, bmm150::PwrCntlNormal);
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

        uint8_t data[Bmm150DataBlockBytes] = {0};
        zzs::kernel::Status st = bus_read (bmm150::RegDataXlsb, data, Bmm150DataBlockBytes);
        if (!zzs::kernel::status_ok (st))
            return st;

        /* Each axis occupies 2 bytes; axis 0 = X (bytes 0-1), 1 = Y (2-3), 2 = Z (4-5). */
        const int16_t raw16 = bmm150_raw16 (&data[_M_axis * 2u]);

        /* BMM150 data is 13-bit signed in bits [15:3]; arithmetic shift right by 3
         * sign-extends the value. */
        const int16_t raw = static_cast<int16_t> (raw16 >> 3);

        const int32_t q15 = zzs::kernel::sensor::driver::fixed::q15_from_raw (raw);
        const int32_t phys_q15 = zzs::kernel::sensor::driver::fixed::q15_mul (q15, bmm150::MagSensQ15MilligaussPerLsb);

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
        return bmm150::OdrHz;
    }

    [[nodiscard]] uint8_t
    device_id () const noexcept override {
        return _M_device_id;
    }

    [[nodiscard]] bool
    self_test () noexcept override {
        uint8_t data[Bmm150DataBlockBytes] = {0};
        return zzs::kernel::status_ok (bus_read (bmm150::RegDataXlsb, data, Bmm150DataBlockBytes));
    }

  private:
    /** @brief Per-instance round-robin axis cursor (X/Y/Z). */
    uint8_t _M_axis = 0;
    /** @brief Whether each channel is enabled. */
    bool _M_enabled[3] = {true, true, true};
};

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors
