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
#include "tsl2561_reg.h"

#include <cstdint>

/**
 * @brief TSL2561 digital light sensor driver (LIGHT channel, Q11 lux).
 *
 * Powers the device on, configures gain and integration time, reads the two
 * channels (broadband + IR) and computes lux in Q11 using an integer-only
 * piecewise approximation of the TAOS/AMS datasheet formula.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors {

/** @brief TSL2561 PART_NUMBER register address. */
inline constexpr uint8_t Tsl2561RegPartNumber = 0x0A;

/** @brief Expected PART_NUMBER value for TSL2561. */
inline constexpr uint8_t Tsl2561PartNumberValue = 0x10;

/** @brief Raw 16-bit little-endian sample from a byte pair. */
[[nodiscard]] inline constexpr uint16_t
tsl2561_raw16 (const uint8_t *p) noexcept {
        return static_cast<uint16_t> (p[0]) | (static_cast<uint16_t> (p[1]) << 8);
}

/** @brief Compute lux in Q11 from channel 0 and channel 1 counts.
 *
 * Integer-only piecewise approximation of the TSL2561 datasheet lux formula
 * (1x gain, 13.7 ms integration time).  All intermediate arithmetic is int32
 * / int64; no float.
 */
[[nodiscard]] inline constexpr zzs::kernel::sensor::driver::fixed::q11_t
tsl2561_compute_lux_q11 (uint16_t ch0, uint16_t ch1) noexcept {
        if (ch0 == 0u)
                return 0;

        const int32_t ratio_q15 = (static_cast<int32_t> (ch1) << 15) / static_cast<int32_t> (ch0);

        int32_t lux_q15 = 0;
        if (ratio_q15 <= tsl2561::LuxRatioBound1) {
                /* Range 1: 0 < ratio <= 0.50 */
                int32_t tmp = 0;
                if (__builtin_mul_overflow (tsl2561::LuxCoeff1A, static_cast<int32_t> (ch0), &tmp))
                        tmp = tsl2561::LuxCoeff1A > 0 ? INT32_MAX : INT32_MIN;
                lux_q15 = tmp >> 15;
        } else if (ratio_q15 <= tsl2561::LuxRatioBound2) {
                /* Range 2: 0.50 < ratio <= 0.61 */
                int64_t num = 0;
                int64_t term1 = 0, term2 = 0;
                if (__builtin_mul_overflow (static_cast<int64_t> (tsl2561::LuxCoeff2A), ch0, &term1))
                        term1 = INT64_MAX;
                if (__builtin_mul_overflow (static_cast<int64_t> (tsl2561::LuxCoeff2B), ch1, &term2))
                        term2 = INT64_MAX;
                if (__builtin_sub_overflow (term1, term2, &num))
                        num = term1 > term2 ? INT64_MAX : INT64_MIN;
                lux_q15 = static_cast<int32_t> (num >> 15);
        } else if (ratio_q15 <= tsl2561::LuxRatioBound3) {
                /* Range 3: 0.61 < ratio <= 0.80 */
                int64_t num = 0;
                int64_t term1 = 0, term2 = 0;
                if (__builtin_mul_overflow (static_cast<int64_t> (tsl2561::LuxCoeff3A), ch0, &term1))
                        term1 = INT64_MAX;
                if (__builtin_mul_overflow (static_cast<int64_t> (tsl2561::LuxCoeff3B), ch1, &term2))
                        term2 = INT64_MAX;
                if (__builtin_sub_overflow (term1, term2, &num))
                        num = term1 > term2 ? INT64_MAX : INT64_MIN;
                lux_q15 = static_cast<int32_t> (num >> 15);
        } else if (ratio_q15 <= tsl2561::LuxRatioBound4) {
                /* Range 4: 0.80 < ratio <= 1.30 */
                int64_t diff = 0;
                if (__builtin_sub_overflow (static_cast<int32_t> (ch0), static_cast<int32_t> (ch1), &diff))
                        diff = ch0 > ch1 ? INT32_MAX : INT32_MIN;
                int64_t num = 0;
                if (__builtin_mul_overflow (static_cast<int64_t> (tsl2561::LuxCoeff4), diff, &num))
                        num = tsl2561::LuxCoeff4 > 0 ? INT64_MAX : INT64_MIN;
                lux_q15 = static_cast<int32_t> (num >> 15);
        } else {
                lux_q15 = 0; /* ratio > 1.30: saturated */
        }

        if (lux_q15 < 0)
                lux_q15 = 0;
        return zzs::kernel::sensor::driver::fixed::q15_to_q11 (lux_q15);
}

/** @brief TSL2561 light sensor driver (LIGHT channel). */
class Tsl2561LightDriver : public SensorDriver {
      public:
        /** @brief Construct with a non-owning bus pointer and an optional clock
         *         source callback for timestamps. */
        explicit Tsl2561LightDriver (bus::BusInterface *bus, uint64_t (*clock) (void) = nullptr) noexcept
            : SensorDriver (bus, clock) {
        }

        [[nodiscard]] zzs::kernel::Status
        init () noexcept override {
                /* Read PART_NUMBER to verify device identity. */
                uint8_t part = 0;
                zzs::kernel::Status st = bus_read8 (Tsl2561RegPartNumber | tsl2561::CmdRegisterBit, part);
                if (!zzs::kernel::status_ok (st))
                        return st;
                if (part != Tsl2561PartNumberValue)
                        return zzs::kernel::Status::BAD_TYPE;
                _M_device_id = part;

                /* Power on. */
                uint8_t ctrl = tsl2561::ControlPowerOn;
                st = bus_write8 (tsl2561::RegControl | tsl2561::CmdRegisterBit, ctrl);
                if (!zzs::kernel::status_ok (st))
                        return st;

                /* Timing: 1x gain, 402 ms integration time. */
                uint8_t timing = static_cast<uint8_t> (tsl2561::TimingGain1x | tsl2561::TimingInteg402ms);
                st = bus_write8 (tsl2561::RegTiming | tsl2561::CmdRegisterBit, timing);
                if (!zzs::kernel::status_ok (st))
                        return st;

                return zzs::kernel::Status::OK;
        }

        [[nodiscard]] zzs::kernel::Status
        read_sample (sensor::Sample &out) noexcept override {
                /* Read channel 0 (broadband) + channel 1 (IR). */
                uint8_t buf[4] = {0};
                zzs::kernel::Status st =
                    bus_read (tsl2561::RegChan0Low | tsl2561::CmdRegisterBit | tsl2561::CmdAutoIncBit, buf, 4u);
                if (!zzs::kernel::status_ok (st))
                        return st;

                const uint16_t ch0 = tsl2561_raw16 (&buf[0]);
                const uint16_t ch1 = tsl2561_raw16 (&buf[2]);

                const zzs::kernel::sensor::driver::fixed::q11_t lux_q11 = tsl2561_compute_lux_q11 (ch0, ch1);

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
                /* Basic self-test: read both channels and confirm the device responds
                 * with plausible (non-saturated, non-zero) counts. */
                uint8_t buf[4] = {0};
                if (!zzs::kernel::status_ok (
                        bus_read (tsl2561::RegChan0Low | tsl2561::CmdRegisterBit | tsl2561::CmdAutoIncBit, buf, 4u)))
                        return false;
                const uint16_t ch0 = tsl2561_raw16 (&buf[0]);
                const uint16_t ch1 = tsl2561_raw16 (&buf[2]);
                return ch0 != 0u && ch0 != 0xFFFFu && ch1 != 0xFFFFu;
        }

      private:
        /** @brief Whether the LIGHT channel is enabled. */
        bool _M_enabled = true;
};

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors
