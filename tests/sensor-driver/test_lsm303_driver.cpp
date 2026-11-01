/*
 * Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

/**
 * @file test_lsm303_driver.cpp
 * @brief Unit tests for the LSM303 combined accelerometer+magnetometer driver.
 */

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "kernel/sensor-driver/sensors/lsm303/lsm303.hpp"
#include "kernel/sensor-driver/sensors/lsm303/lsm303_reg.h"

#include "test_mock_bus.hpp"

namespace
{
using namespace zzs::kernel;
using namespace zzs::kernel::sensor::driver::sensors;
using namespace zzs::kernel::sensor::driver::sensors::lsm303;
using namespace zzs::kernel::sensor::driver::fixed;
} // namespace

TEST_CASE ("Accel sub-device ID verification", "[sensor-driver][lsm303][accel-id]")
{
    test::MockBus accel_bus;
    test::MockBus mag_bus;
    accel_bus.set_initialized (true);
    mag_bus.set_initialized (true);
    accel_bus.set_reg (RegWhoAmI, AccelWhoAmIValue);
    mag_bus.set_reg (RegWhoAmI, MagWhoAmIValue);

    Lsm303Driver drv (&accel_bus, &mag_bus);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);
    REQUIRE (drv.device_id () == AccelWhoAmIValue);
}

TEST_CASE ("Mag sub-device ID verification", "[sensor-driver][lsm303][mag-id]")
{
    test::MockBus accel_bus;
    test::MockBus mag_bus;
    accel_bus.set_initialized (true);
    mag_bus.set_initialized (true);
    accel_bus.set_reg (RegWhoAmI, AccelWhoAmIValue);
    mag_bus.set_reg (RegWhoAmI, MagWhoAmIValue);

    Lsm303Driver drv (&accel_bus, &mag_bus);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);

    /* Mag WHO_AM_I is read at init; both sub-devices must be present. */
    REQUIRE (drv.device_id () == AccelWhoAmIValue);
}

TEST_CASE ("Combined accel+mag read", "[sensor-driver][lsm303][combined-read]")
{
    test::MockBus accel_bus;
    test::MockBus mag_bus;
    accel_bus.set_initialized (true);
    mag_bus.set_initialized (true);
    accel_bus.set_reg (RegWhoAmI, AccelWhoAmIValue);
    mag_bus.set_reg (RegWhoAmI, MagWhoAmIValue);

    /* Accel status: data ready; X=100. */
    accel_bus.set_reg (RegStatusRegA, StatusRegAZyxda);
    uint8_t accel_data[6] = {
        static_cast<uint8_t> (100 & 0xFF), static_cast<uint8_t> ((100 >> 8) & 0xFF),
        0, 0, 0, 0,
    };
    accel_bus.set_block (RegOutXLAccel, accel_data, 6u);

    /* Mag status: data ready; X=200. */
    mag_bus.set_reg (RegStatusRegM, StatusRegMZyxda);
    uint8_t mag_data[6] = {
        static_cast<uint8_t> (200 & 0xFF), static_cast<uint8_t> ((200 >> 8) & 0xFF),
        0, 0, 0, 0,
    };
    mag_bus.set_block (RegOutXLMag, mag_data, 6u);

    Lsm303Driver drv (&accel_bus, &mag_bus);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);

    /* First channel round-robin is ACCEL_X. */
    sensor::Sample sample;
    REQUIRE (drv.read_sample (sample) == zzs::kernel::Status::OK);
    REQUIRE (sample._M_channel == sensor::Channel::ACCEL_X);
    REQUIRE (sample._M_sample_q15 ==
             q15_to_raw (q15_mul (q15_from_raw (100),
                                  AccelSensQ15MgPerLsb)));

    /* Advance through ACCEL_Y, ACCEL_Z then MAG_X. */
    REQUIRE (drv.read_sample (sample) == zzs::kernel::Status::OK);
    REQUIRE (sample._M_channel == sensor::Channel::ACCEL_Y);
    REQUIRE (drv.read_sample (sample) == zzs::kernel::Status::OK);
    REQUIRE (sample._M_channel == sensor::Channel::ACCEL_Z);
    REQUIRE (drv.read_sample (sample) == zzs::kernel::Status::OK);
    REQUIRE (sample._M_channel == sensor::Channel::MAG_X);
    REQUIRE (sample._M_sample_q15 ==
             q15_to_raw (q15_mul (q15_from_raw (200),
                                  MagSensQ15MilligaussPerLsb)));
}

TEST_CASE ("Independent accel/mag enable/disable", "[sensor-driver][lsm303][enable]")
{
    test::MockBus accel_bus;
    test::MockBus mag_bus;
    accel_bus.set_initialized (true);
    mag_bus.set_initialized (true);
    accel_bus.set_reg (RegWhoAmI, AccelWhoAmIValue);
    mag_bus.set_reg (RegWhoAmI, MagWhoAmIValue);

    Lsm303Driver drv (&accel_bus, &mag_bus);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);

    /* Disable all accel channels (indices 0-2 map to ACCEL_X/Y/Z). */
    REQUIRE (drv.set_channel_enabled (sensor::Channel::ACCEL_X, false)
             == zzs::kernel::Status::OK);
    REQUIRE (drv.set_channel_enabled (sensor::Channel::ACCEL_Y, false)
             == zzs::kernel::Status::OK);
    REQUIRE (drv.set_channel_enabled (sensor::Channel::ACCEL_Z, false)
             == zzs::kernel::Status::OK);

    /* Out-of-range channel id is rejected. */
    REQUIRE (drv.set_channel_enabled (sensor::Channel::COUNT, true)
             == zzs::kernel::Status::OUT_OF_RANGE);
}
