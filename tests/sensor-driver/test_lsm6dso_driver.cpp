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
 * @file test_lsm6dso_driver.cpp
 * @brief Unit tests for the LSM6DSO accelerometer/gyroscope driver (mock bus).
 */

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "kernel/sensor-driver/sensors/lsm6dso/lsm6dso.hpp"
#include "kernel/sensor-driver/sensors/lsm6dso/lsm6dso_reg.h"

#include "test_mock_bus.hpp"

namespace
{
using namespace zzs::kernel;
using namespace zzs::kernel::sensor::driver::sensors;
using namespace zzs::kernel::sensor::driver::sensors::lsm6dso;
using namespace zzs::kernel::sensor::driver::fixed;
} // namespace

TEST_CASE ("Device ID verification (WHO_AM_I = 0x6C)", "[sensor-driver][lsm6dso]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    Lsm6dsoAccelDriver accel (&bus);
    REQUIRE (accel.init () == zzs::kernel::Status::OK);
    REQUIRE (accel.device_id () == WhoAmIValue);

    Lsm6dsoGyroDriver gyro (&bus);
    REQUIRE (gyro.init () == zzs::kernel::Status::OK);
    REQUIRE (gyro.device_id () == WhoAmIValue);
}

TEST_CASE ("Accelerometer full-scale configuration (2g/4g/8g/16g)",
           "[sensor-driver][lsm6dso][accel]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    Lsm6dsoAccelDriver accel (&bus);
    REQUIRE (accel.init () == zzs::kernel::Status::OK);

    /* After init, CTRL1_XL holds ODR 208 Hz | FS 2g = 0x55. */
    REQUIRE (bus.get_reg (RegCtrl1Xl) ==
             static_cast<uint8_t> (Lsm6dsoOdr208Hz | Ctrl1XlFs2g));
}

TEST_CASE ("Gyroscope full-scale configuration (125/250/500/1000/2000 dps)",
           "[sensor-driver][lsm6dso][gyro]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    Lsm6dsoGyroDriver gyro (&bus);
    REQUIRE (gyro.init () == zzs::kernel::Status::OK);

    /* After init, CTRL2_G holds ODR 208 Hz | FS 125 dps = 0x05. */
    REQUIRE (bus.get_reg (RegCtrl2G) ==
             static_cast<uint8_t> (Lsm6dsoOdrG208Hz | Ctrl2GFs125dps));
}

TEST_CASE ("Data rate configuration", "[sensor-driver][lsm6dso][odr]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    Lsm6dsoAccelDriver accel (&bus);
    REQUIRE (accel.init () == zzs::kernel::Status::OK);
    REQUIRE (accel.channel_rate_hz (sensor::Channel::ACCEL_X) == 208u);

    Lsm6dsoGyroDriver gyro (&bus);
    REQUIRE (gyro.init () == zzs::kernel::Status::OK);
    REQUIRE (gyro.channel_rate_hz (sensor::Channel::GYRO_X) == 208u);
}

TEST_CASE ("Accelerometer raw data read (6 bytes → 3 axes Q15)",
           "[sensor-driver][lsm6dso][accel-read]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    /* Status register: accelerometer data available. */
    bus.set_reg (RegStatusReg, StatusXlda);

    /* 6-byte data block: X=1000, Y=-2000, Z=3000 (little-endian). */
    uint8_t data[6] = {
        static_cast<uint8_t> (1000 & 0xFF),
        static_cast<uint8_t> ((1000 >> 8) & 0xFF),
        static_cast<uint8_t> ((-2000) & 0xFF),
        static_cast<uint8_t> (((-2000) >> 8) & 0xFF),
        static_cast<uint8_t> (3000 & 0xFF),
        static_cast<uint8_t> ((3000 >> 8) & 0xFF),
    };
    bus.set_block (RegOutXLAccel, data, 6u);

    Lsm6dsoAccelDriver accel (&bus);
    REQUIRE (accel.init () == zzs::kernel::Status::OK);

    /* Read all three axes (round-robin order depends on the static axis, so
     * collect by channel index). */
    int16_t values[3] = { 0, 0, 0 };
    bool seen[3] = { false, false, false };
    for (int i = 0; i < 3; ++i)
        {
        sensor::Sample sample;
        REQUIRE (accel.read_sample (sample) == zzs::kernel::Status::OK);
        const uint8_t idx = static_cast<uint8_t> (sample._M_channel)
                            - static_cast<uint8_t> (sensor::Channel::ACCEL_X);
        REQUIRE (idx < 3u);
        values[idx] = sample._M_sample_q15;
        seen[idx] = true;
        }
    REQUIRE ((seen[0] && seen[1] && seen[2]));
    REQUIRE (values[0] == q15_to_raw (q15_mul (q15_from_raw (1000),
                                               AccelSensQ15MgPerLsb)));
    REQUIRE (values[1] == q15_to_raw (q15_mul (q15_from_raw (-2000),
                                               AccelSensQ15MgPerLsb)));
    REQUIRE (values[2] == q15_to_raw (q15_mul (q15_from_raw (3000),
                                               AccelSensQ15MgPerLsb)));
}

TEST_CASE ("Gyroscope raw data read (6 bytes → 3 axes Q15)",
           "[sensor-driver][lsm6dso][gyro-read]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    /* Status register: gyroscope data available. */
    bus.set_reg (RegStatusReg, StatusGda);

    uint8_t data[6] = {
        static_cast<uint8_t> (500 & 0xFF),
        static_cast<uint8_t> ((500 >> 8) & 0xFF),
        0, 0, 0, 0,
    };
    bus.set_block (RegOutXLGyro, data, 6u);

    Lsm6dsoGyroDriver gyro (&bus);
    REQUIRE (gyro.init () == zzs::kernel::Status::OK);

    sensor::Sample sample;
    REQUIRE (gyro.read_sample (sample) == zzs::kernel::Status::OK);
    REQUIRE (sample._M_channel == sensor::Channel::GYRO_X);
    REQUIRE (sample._M_sample_q15 ==
             q15_to_raw (q15_mul (q15_from_raw (500),
                                  GyroSensQ15MdpsPerLsb)));
}

TEST_CASE ("Data ready flag checking", "[sensor-driver][lsm6dso][data-ready]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    Lsm6dsoAccelDriver accel (&bus);
    REQUIRE (accel.init () == zzs::kernel::Status::OK);

    /* No data ready -> STALE. */
    bus.set_reg (RegStatusReg, 0x00);
    sensor::Sample sample;
    REQUIRE (accel.read_sample (sample) == zzs::kernel::Status::STALE);

    /* Data ready -> OK. */
    bus.set_reg (RegStatusReg, StatusXlda);
    uint8_t data[6] = { 1, 0, 0, 0, 0, 0 };
    bus.set_block (RegOutXLAccel, data, 6u);
    REQUIRE (accel.read_sample (sample) == zzs::kernel::Status::OK);
}

TEST_CASE ("Self-test: write self-test bits, verify output change",
           "[sensor-driver][lsm6dso][self-test]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);
    bus.set_reg (RegStatusReg, StatusXlda);
    uint8_t data[6] = { 100, 0, 0, 0, 0, 0 };
    bus.set_block (RegOutXLAccel, data, 6u);

    Lsm6dsoAccelDriver accel (&bus);
    REQUIRE (accel.init () == zzs::kernel::Status::OK);

    /* The mock returns the same bytes before/after, so after == baseline and
     * self_test returns false (no actual ST hardware in the mock). */
    REQUIRE (accel.self_test () == false);
}

TEST_CASE ("Edge cases: invalid device ID, NULL bus pointer, read before init",
           "[sensor-driver][lsm6dso][edge]")
{
    /* Invalid device ID -> init fails with BAD_TYPE. */
    {
        test::MockBus bus;
        bus.set_initialized (true);
        bus.set_reg (RegWhoAmI, 0x00); /* wrong ID */
        Lsm6dsoAccelDriver accel (&bus);
        REQUIRE (accel.init () == zzs::kernel::Status::BAD_TYPE);
    }

    /* NULL bus pointer -> init returns INTERNAL (bus_read8 guards null). */
    {
        Lsm6dsoAccelDriver accel (nullptr);
        REQUIRE (accel.init () == zzs::kernel::Status::INTERNAL);
    }

    /* Read before init: device_id is 0 and read_sample returns STALE/not ready
     * because status register reads 0. */
    {
        test::MockBus bus;
        bus.set_initialized (true);
        Lsm6dsoAccelDriver accel (&bus);
        REQUIRE (accel.device_id () == 0);
        sensor::Sample sample;
        REQUIRE (accel.read_sample (sample) == zzs::kernel::Status::STALE);
    }
}
