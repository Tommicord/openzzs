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
#include "ltr558_reg.h"

#include <cstdint>

/** @brief Raw 16-bit little-endian sample from a byte pair. */
[[nodiscard]] inline constexpr uint16_t
ltr558_raw16 (const uint8_t *p) noexcept {
    return static_cast<uint16_t> (p[0]) | (static_cast<uint16_t> (p[1]) << 8);
}

/**
 * @brief Lite-On LTR-558 ambient light sensor driver (LIGHT channel).
 *
 * Reads the 2-byte ALS data block (CH1) and converts raw counts to Q15
 * milli-lux using the sensitivity multiplier from ltr558_reg.h.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors {

/** @brief LTR-558 light sensor driver (LIGHT channel). */
class Ltr558LightDriver : public SensorDriver {
  public:
    /** @brief Construct with a non-owning bus pointer and an optional clock
     *         source callback for timestamps. */
    explicit Ltr558LightDriver (bus::BusInterface *bus, uint64_t (*clock) (void) = nullptr) noexcept
        : SensorDriver (bus, clock) {
    }

    [[nodiscard]] zzs::kernel::Status
    init () noexcept override {
        uint8_t who = 0;
        zzs::kernel::Status st = bus_read8 (ltr558::RegPartId, who);
        if (!zzs::kernel::status_ok (st))
            return st;
        if (who != ltr558::PartIdValue)
            return zzs::kernel::Status::BAD_TYPE;
        _M_device_id = who;

        /* ALS active mode, gain 1x, ODR 100 ms (10 Hz). */
        st = bus_write8 (ltr558::RegAlsCtrl, static_cast<uint8_t> (ltr558::AlsCtrlModeActive | ltr558::AlsCtrlGain1x
                                                                   | ltr558::AlsCtrlOdr100ms));
        if (!zzs::kernel::status_ok (st))
            return st;

        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    read_sample (sensor::Sample &out) noexcept override {
        uint8_t status = 0;
        zzs::kernel::Status st = bus_read8 (ltr558::RegAlsStatus, status);
        if (!zzs::kernel::status_ok (st))
            return st;
        if ((status & ltr558::AlsStatusAlsValid) == 0u)
            return zzs::kernel::Status::STALE;

        uint8_t data[ltr558::AlsDataBytes] = {0};
        st = bus_read (ltr558::RegAlsDataCh1Low, data, ltr558::AlsDataBytes);
        if (!zzs::kernel::status_ok (st))
            return st;

        const uint16_t raw = ltr558_raw16 (data);
        const int32_t q15 = zzs::kernel::sensor::driver::fixed::q15_from_raw (static_cast<int16_t> (raw));
        const int32_t phys_q15 = zzs::kernel::sensor::driver::fixed::q15_mul (q15, ltr558::LuxSensQ15MilliLuxPerCount);

        out._M_channel = sensor::Channel::LIGHT;
        out._M_sample_q15 = zzs::kernel::sensor::driver::fixed::q15_to_raw (phys_q15);
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
        return ltr558::OdrHz;
    }

    [[nodiscard]] uint8_t
    device_id () const noexcept override {
        return _M_device_id;
    }

    [[nodiscard]] bool
    self_test () noexcept override {
        uint8_t data[ltr558::AlsDataBytes] = {0};
        return zzs::kernel::status_ok (bus_read (ltr558::RegAlsDataCh1Low, data, ltr558::AlsDataBytes));
    }

  private:
    /** @brief Whether the light channel is enabled. */
    bool _M_enabled = true;
};

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors
