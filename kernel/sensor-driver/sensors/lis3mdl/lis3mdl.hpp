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
#include "lis3mdl_reg.h"

#include <cstdint>

/**
 * @brief LIS3MDL 3-axis magnetometer driver (MAG_X/Y/Z).
 *
 * Reads the 6-byte magnetometer data block and converts raw counts to Q15
 * milligauss using the Q15 sensitivity multiplier from lis3mdl_reg.h.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors {

/** @brief Number of bytes in a 3-axis data block (X/Y/Z, 2 bytes each). */
inline constexpr uint32_t Lis3mdlDataBlockBytes = 6u;

/** @brief LIS3MDL ODR: 20 Hz (CTRL_REG1 OM = 101). */
inline constexpr uint8_t Lis3mdlOdr20Hz = 0x05 << 1; /* CTRL_REG1 OM bits [6:1] */

/** @brief Raw 16-bit little-endian sample from a byte pair. */
[[nodiscard]] inline constexpr int16_t
lis3mdl_raw16 (const uint8_t *p) noexcept {
    return static_cast<int16_t> (static_cast<uint16_t> (p[0]) | (static_cast<uint16_t> (p[1]) << 8));
}

/** @brief LIS3MDL magnetometer driver (MAG_X/Y/Z). */
class Lis3mdlMagDriver : public SensorDriver {
  public:
    /** @brief Construct with a non-owning bus pointer and an optional clock
     *         source callback for timestamps. */
    explicit Lis3mdlMagDriver (bus::BusInterface *bus, uint64_t (*clock) (void) = nullptr) noexcept
        : SensorDriver (bus, clock) {
    }

    [[nodiscard]] zzs::kernel::Status
    init () noexcept override {
        uint8_t who = 0;
        zzs::kernel::Status st = bus_read8 (lis3mdl::RegWhoAmI, who);
        if (!zzs::kernel::status_ok (st))
            return st;
        if (who != lis3mdl::WhoAmIValue)
            return zzs::kernel::Status::BAD_TYPE;
        _M_device_id = who;

        /* Reboot memory content, then wait. */
        st = bus_write8 (lis3mdl::RegCtrlReg2, lis3mdl::CtrlReg2Reboot);
        if (!zzs::kernel::status_ok (st))
            return st;
        _M_bus->delay_ms (1u);

        /* Continuous conversion mode, ODR 20 Hz. */
        st = bus_write8 (lis3mdl::RegCtrlReg1, static_cast<uint8_t> (lis3mdl::CtrlReg1TempEn | Lis3mdlOdr20Hz));
        if (!zzs::kernel::status_ok (st))
            return st;

        /* Full-scale ±4 gauss. */
        st = bus_write8 (lis3mdl::RegCtrlReg2, lis3mdl::CtrlReg2Fs4Gauss);
        if (!zzs::kernel::status_ok (st))
            return st;

        /* Continuous conversion mode. */
        st = bus_write8 (lis3mdl::RegCtrlReg3, lis3mdl::CtrlReg3MdCont);
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

        uint8_t status = 0;
        zzs::kernel::Status st = bus_read8 (lis3mdl::RegStatusReg, status);
        if (!zzs::kernel::status_ok (st))
            return st;

        uint8_t ready = 0u;
        switch (_M_axis) {
        case 0u:
            ready = lis3mdl::StatusXda;
            break;
        case 1u:
            ready = lis3mdl::StatusYda;
            break;
        default:
            ready = lis3mdl::StatusZda;
            break;
        }
        if ((status & ready) == 0u)
            return zzs::kernel::Status::STALE;

        uint8_t data[Lis3mdlDataBlockBytes] = {0};
        st = bus_read (lis3mdl::RegOutXL, data, Lis3mdlDataBlockBytes);
        if (!zzs::kernel::status_ok (st))
            return st;

        const int16_t raw = lis3mdl_raw16 (&data[_M_axis * 2u]);
        const int32_t q15 = zzs::kernel::sensor::driver::fixed::q15_from_raw (raw);
        /* Both raw and sensitivity are Q15, so the product is Q30; use the
         * saturating Q15 multiply (round-shift by 15) to land back in Q15. */
        const int32_t phys_q15 = zzs::kernel::sensor::driver::fixed::q15_mul (q15, lis3mdl::MagSensQ15MilligaussPerLsb);

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
        return 20u;
    }

    [[nodiscard]] uint8_t
    device_id () const noexcept override {
        return _M_device_id;
    }

    [[nodiscard]] bool
    self_test () noexcept override {
        /* Read baseline mag X output. */
        uint8_t status = 0;
        if (!zzs::kernel::status_ok (bus_read8 (lis3mdl::RegStatusReg, status)))
            return false;
        uint8_t data[Lis3mdlDataBlockBytes] = {0};
        if (!zzs::kernel::status_ok (bus_read (lis3mdl::RegOutXL, data, Lis3mdlDataBlockBytes)))
            return false;
        const int16_t baseline = lis3mdl_raw16 (&data[0]);

        /* Enable magnetometer self-test (CTRL_REG1 ST bit). */
        zzs::kernel::Status st = bus_update8 (lis3mdl::RegCtrlReg1, lis3mdl::CtrlReg1St, 0u);
        if (!zzs::kernel::status_ok (st))
            return false;
        _M_bus->delay_ms (1u);

        if (!zzs::kernel::status_ok (bus_read8 (lis3mdl::RegStatusReg, status))) {
            (void) bus_update8 (lis3mdl::RegCtrlReg1, 0u, lis3mdl::CtrlReg1St);
            return false;
        }
        if (!zzs::kernel::status_ok (bus_read (lis3mdl::RegOutXL, data, Lis3mdlDataBlockBytes))) {
            (void) bus_update8 (lis3mdl::RegCtrlReg1, 0u, lis3mdl::CtrlReg1St);
            return false;
        }
        const int16_t after = lis3mdl_raw16 (&data[0]);

        /* Disable self-test. */
        (void) bus_update8 (lis3mdl::RegCtrlReg1, 0u, lis3mdl::CtrlReg1St);

        const int32_t delta = static_cast<int32_t> (after) - static_cast<int32_t> (baseline);
        return delta != 0;
    }

  private:
    /** @brief Per-instance round-robin axis cursor (X/Y/Z). */
    uint8_t _M_axis = 0;
    /** @brief Last raw mag sample (Q15 per axis), indexed X/Y/Z. */
    int16_t _M_raw_q15[3] = {0, 0, 0};
    /** @brief Whether each channel is enabled. */
    bool _M_enabled[3] = {true, true, true};
};

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors
