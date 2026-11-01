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

#include "lsm303_reg.h"

#include "kernel/sensor-driver/sensors/sensor_driver.hpp"

#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief LSM303 combined accelerometer + magnetometer driver.
 *
 * A single driver instance exposes both the accelerometer (ACCEL_X/Y/Z) and
 * magnetometer (MAG_X/Y/Z) axes of an LSM303 package.  The two sub-devices live
 * at different I2C addresses, so the driver takes two non-owning bus pointers
 * (one per sub-device).  Each read_sample call returns the next enabled axis in
 * round-robin order across all six channels.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors
{

/** @brief Number of bytes in a 3-axis data block (X/Y/Z, 2 bytes each). */
inline constexpr uint32_t Lsm303DataBlockBytes = 6u;

/** @brief Accelerometer ODR: 100 Hz. */
inline constexpr uint8_t Lsm303AccelOdr100Hz = 0x5; /* CTRL_REG1_A ODR = 0101 */

/** @brief Raw 16-bit little-endian sample from a byte pair. */
[[nodiscard]] inline constexpr int16_t
lsm303_raw16 (const uint8_t *p) noexcept
{
    return static_cast<int16_t> (
        static_cast<uint16_t> (p[0]) | (static_cast<uint16_t> (p[1]) << 8));
}

/** @brief LSM303 combined accelerometer + magnetometer driver. */
class Lsm303Driver : public SensorDriver
{
public:
    /** @brief Construct with non-owning bus pointers for the accel and mag
     *         sub-devices and an optional clock source callback for timestamps. */
    explicit Lsm303Driver (bus::BusInterface *bus_accel,
                           bus::BusInterface *bus_mag,
                           uint64_t (*clock)(void) = nullptr) noexcept
        : SensorDriver (nullptr, clock)
    {
        _M_bus_accel = bus_accel;
        _M_bus_mag = bus_mag;
    }

    [[nodiscard]] zzs::kernel::Status
    init () noexcept override
    {
        /* ---- Accelerometer sub-device ---- */
        uint8_t who = 0;
        zzs::kernel::Status st = bus_read8_accel (lsm303::RegWhoAmI, who);
        if (!zzs::kernel::status_ok (st))
            return st;
        if (who != lsm303::AccelWhoAmIValue)
            return zzs::kernel::Status::BAD_TYPE;
        _M_device_id = who;

        /* Enable BDU, set accel ODR 100 Hz, full scale ±2 g. */
        st = bus_write8_accel (lsm303::RegCtrlReg4A,
                               static_cast<uint8_t> (lsm303::CtrlReg4ABdu |
                                                     lsm303::CtrlReg4AFs2g));
        if (!zzs::kernel::status_ok (st))
            return st;
        st = bus_write8_accel (lsm303::RegCtrlReg1A,
                               Lsm303AccelOdr100Hz);
        if (!zzs::kernel::status_ok (st))
            return st;

        /* ---- Magnetometer sub-device ---- */
        st = bus_read8_mag (lsm303::RegWhoAmI, who);
        if (!zzs::kernel::status_ok (st))
            return st;
        if (who != lsm303::MagWhoAmIValue)
            return zzs::kernel::Status::BAD_TYPE;

        /* Mag: continuous conversion mode, default full-scale. */
        st = bus_write8_mag (lsm303::RegCtrlReg3M,
                             lsm303::CtrlReg3MmdContinuous);
        if (!zzs::kernel::status_ok (st))
            return st;
        st = bus_write8_mag (lsm303::RegCtrlReg2M,
                             lsm303::CtrlReg2MFsDefault);
        if (!zzs::kernel::status_ok (st))
            return st;

        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    read_sample (sensor::Sample &out) noexcept override
    {
        /* Six channels round-robin: accel X/Y/Z then mag X/Y/Z. */
        while (axis_ < 6u && !_M_enabled[axis_])
            axis_ = static_cast<uint8_t> ((axis_ + 1u) % 6u);
        if (axis_ >= 6u)
            return zzs::kernel::Status::STALE;

        uint8_t data[Lsm303DataBlockBytes] = { 0 };
        bool is_mag = axis_ >= 3u;
        uint8_t local_axis = static_cast<uint8_t> (is_mag ? axis_ - 3u : axis_);

        zzs::kernel::Status st;
        if (is_mag)
            {
            uint8_t status = 0;
            st = bus_read8_mag (lsm303::RegStatusRegM, status);
            if (!zzs::kernel::status_ok (st))
                return st;
            if ((status & lsm303::StatusRegMZyxda) == 0u)
                return zzs::kernel::Status::STALE;
            st = bus_read_mag (lsm303::RegOutXLMag, data, Lsm303DataBlockBytes);
            }
        else
            {
            uint8_t status = 0;
            st = bus_read8_accel (lsm303::RegStatusRegA, status);
            if (!zzs::kernel::status_ok (st))
                return st;
            if ((status & lsm303::StatusRegAZyxda) == 0u)
                return zzs::kernel::Status::STALE;
            st = bus_read_accel (lsm303::RegOutXLAccel, data, Lsm303DataBlockBytes);
            }
        if (!zzs::kernel::status_ok (st))
            return st;

        const int16_t raw = lsm303_raw16 (&data[local_axis * 2u]);
        const int32_t q15 = zzs::kernel::sensor::driver::fixed::q15_from_raw (raw);
        /* Both raw and sensitivity are Q15, so the product is Q30; use the
         * saturating Q15 multiply (round-shift by 15) to land back in Q15. */
        const int32_t phys_q15 = zzs::kernel::sensor::driver::fixed::q15_mul (
            q15, is_mag ? lsm303::MagSensQ15MilligaussPerLsb
                        : lsm303::AccelSensQ15MgPerLsb);

        const uint8_t ch_base = is_mag
                                    ? static_cast<uint8_t> (sensor::Channel::MAG_X)
                                    : static_cast<uint8_t> (sensor::Channel::ACCEL_X);
        out._M_channel = static_cast<sensor::Channel> (ch_base + local_axis);
        out._M_sample_q15 = zzs::kernel::sensor::driver::fixed::q15_to_raw (
            phys_q15);
        out._M_timestamp_ns = timestamp_ns ();

        axis_ = static_cast<uint8_t> ((axis_ + 1u) % 6u);
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    set_channel_enabled (sensor::Channel ch, bool enabled) noexcept override
    {
        const uint8_t idx = static_cast<uint8_t> (ch);
        if (idx >= static_cast<uint8_t> (sensor::Channel::COUNT))
            return zzs::kernel::Status::OUT_OF_RANGE;
        _M_enabled[idx] = enabled;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] uint32_t
    channel_rate_hz (sensor::Channel ch) noexcept override
    {
        (void) ch;
        return 100u;
    }

    [[nodiscard]] uint8_t
    device_id () const noexcept override
    {
        return _M_device_id;
    }

    [[nodiscard]] bool
    self_test () noexcept override
    {
        /* Accelerometer self-test via CTRL_REG5_A ST_X bit. */
        uint8_t ctrl5 = 0;
        if (!zzs::kernel::status_ok (
                bus_read8_accel (lsm303::RegCtrlReg5A, ctrl5)))
            return false;

        /* Read baseline accel X output. */
        uint8_t data[Lsm303DataBlockBytes] = { 0 };
        if (!zzs::kernel::status_ok (
                bus_read_accel (lsm303::RegOutXLAccel, data, Lsm303DataBlockBytes)))
            return false;
        const int16_t baseline = lsm303_raw16 (&data[0]);

        /* Enable accelerometer self-test X. */
        const uint8_t ctrl5_st = static_cast<uint8_t> (ctrl5 | lsm303::CtrlReg5AStX);
        zzs::kernel::Status st = bus_write8_accel (lsm303::RegCtrlReg5A, ctrl5_st);
        if (!zzs::kernel::status_ok (st))
            {
            (void) bus_write8_accel (lsm303::RegCtrlReg5A, ctrl5);
            return false;
            }
        _M_bus_accel->delay_ms (10u);

        /* Re-read after enabling ST; output should shift. */
        if (!zzs::kernel::status_ok (
                bus_read_accel (lsm303::RegOutXLAccel, data, Lsm303DataBlockBytes)))
            {
            (void) bus_write8_accel (lsm303::RegCtrlReg5A, ctrl5);
            return false;
            }
        const int16_t after = lsm303_raw16 (&data[0]);

        /* Restore CTRL_REG5_A. */
        (void) bus_write8_accel (lsm303::RegCtrlReg5A, ctrl5);

        const int32_t delta =
            static_cast<int32_t> (after) - static_cast<int32_t> (baseline);
        return delta != 0;
    }

private:
    /** @brief Read one accel register. */
    [[nodiscard]] inline zzs::kernel::Status
    bus_read8_accel (uint8_t reg, uint8_t &value) noexcept
    {
        return _M_bus_accel == nullptr
                   ? zzs::kernel::Status::INTERNAL
                   : _M_bus_accel->read_reg (reg, &value, 1u);
    }

    /** @brief Write one accel register. */
    [[nodiscard]] inline zzs::kernel::Status
    bus_write8_accel (uint8_t reg, uint8_t value) noexcept
    {
        return _M_bus_accel == nullptr
                   ? zzs::kernel::Status::INTERNAL
                   : _M_bus_accel->write_reg (reg, &value, 1u);
    }

    /** @brief Read accel register block. */
    [[nodiscard]] inline zzs::kernel::Status
    bus_read_accel (uint8_t reg, uint8_t *buf, uint16_t len) noexcept
    {
        return _M_bus_accel == nullptr
                   ? zzs::kernel::Status::INTERNAL
                   : _M_bus_accel->read_reg (reg, buf, len);
    }

    /** @brief Write accel register block. */
    [[nodiscard]] inline zzs::kernel::Status
    bus_write_accel (uint8_t reg, const uint8_t *buf, uint16_t len) noexcept
    {
        return _M_bus_accel == nullptr
                   ? zzs::kernel::Status::INTERNAL
                   : _M_bus_accel->write_reg (reg, buf, len);
    }

    /** @brief Read one mag register. */
    [[nodiscard]] inline zzs::kernel::Status
    bus_read8_mag (uint8_t reg, uint8_t &value) noexcept
    {
        return _M_bus_mag == nullptr
                   ? zzs::kernel::Status::INTERNAL
                   : _M_bus_mag->read_reg (reg, &value, 1u);
    }

    /** @brief Write one mag register. */
    [[nodiscard]] inline zzs::kernel::Status
    bus_write8_mag (uint8_t reg, uint8_t value) noexcept
    {
        return _M_bus_mag == nullptr
                   ? zzs::kernel::Status::INTERNAL
                   : _M_bus_mag->write_reg (reg, &value, 1u);
    }

    /** @brief Read mag register block. */
    [[nodiscard]] inline zzs::kernel::Status
    bus_read_mag (uint8_t reg, uint8_t *buf, uint16_t len) noexcept
    {
        return _M_bus_mag == nullptr
                   ? zzs::kernel::Status::INTERNAL
                   : _M_bus_mag->read_reg (reg, buf, len);
    }

    /** @brief Write mag register block. */
    [[nodiscard]] inline zzs::kernel::Status
    bus_write_mag (uint8_t reg, const uint8_t *buf, uint16_t len) noexcept
    {
        return _M_bus_mag == nullptr
                   ? zzs::kernel::Status::INTERNAL
                   : _M_bus_mag->write_reg (reg, buf, len);
    }

    bus::BusInterface *_M_bus_accel = nullptr;
    bus::BusInterface *_M_bus_mag = nullptr;
    uint8_t _M_enabled[6] = { true, true, true, true, true, true };
    uint8_t axis_ = 0;
    uint8_t _M_device_id = 0;
};

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors
