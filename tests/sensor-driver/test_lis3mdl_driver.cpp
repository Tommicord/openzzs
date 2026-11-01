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
 * @file test_lis3mdl_driver.cpp
 * @brief Unit tests for the LIS3MDL magnetometer driver (mock bus).
 */

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "kernel/sensor-driver/sensors/lis3mdl/lis3mdl.hpp"
#include "kernel/sensor-driver/sensors/lis3mdl/lis3mdl_reg.h"

#include "test_mock_bus.hpp"

namespace
{
using namespace zzs::kernel;
using namespace zzs::kernel::sensor::driver::sensors;
using namespace zzs::kernel::sensor::driver::sensors::lis3mdl;
using namespace zzs::kernel::sensor::driver::fixed;
} // namespace

TEST_CASE ("Device ID verification (WHO_AM_I = 0x3D)", "[sensor-driver][lis3mdl]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    Lis3mdlMagDriver mag (&bus);
    REQUIRE (mag.init () == zzs::kernel::Status::OK);
    REQUIRE (mag.device_id () == WhoAmIValue);
}

TEST_CASE ("Full-scale configuration (4/8/12/16 gauss)", "[sensor-driver][lis3mdl][fs]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    Lis3mdlMagDriver mag (&bus);
    REQUIRE (mag.init () == zzs::kernel::Status::OK);

    /* init writes FS 4 gauss = 0x00 to CTRL_REG2. */
    REQUIRE (bus.get_reg (RegCtrlReg2) == CtrlReg2Fs4Gauss);
}

TEST_CASE ("Data rate configuration", "[sensor-driver][lis3mdl][odr]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    Lis3mdlMagDriver mag (&bus);
    REQUIRE (mag.init () == zzs::kernel::Status::OK);
    REQUIRE (mag.channel_rate_hz (sensor::Channel::MAG_X) == 20u);
}

TEST_CASE ("Operating mode (continuous/single/power-down)", "[sensor-driver][lis3mdl][mode]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    Lis3mdlMagDriver mag (&bus);
    REQUIRE (mag.init () == zzs::kernel::Status::OK);

    /* init selects continuous mode = 0x00 in CTRL_REG3. */
    REQUIRE (bus.get_reg (RegCtrlReg3) == CtrlReg3MdCont);
}

TEST_CASE ("Magnetic data read (6 bytes → 3 axes Q15)", "[sensor-driver][lis3mdl][read]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    /* All axes ready; full 6-byte data block (X=700, Y=800, Z=900). */
    bus.set_reg (RegStatusReg,
                 static_cast<uint8_t> (StatusXda | StatusYda | StatusZda));
    uint8_t data[6] = {
        static_cast<uint8_t> (700 & 0xFF),
        static_cast<uint8_t> ((700 >> 8) & 0xFF),
        static_cast<uint8_t> (800 & 0xFF),
        static_cast<uint8_t> ((800 >> 8) & 0xFF),
        static_cast<uint8_t> (900 & 0xFF),
        static_cast<uint8_t> ((900 >> 8) & 0xFF),
    };
    bus.set_block (RegOutXL, data, 6u);

    Lis3mdlMagDriver mag (&bus);
    REQUIRE (mag.init () == zzs::kernel::Status::OK);

    /* Read all three axes (round-robin order depends on the static axis, so
     * collect by channel index). */
    int16_t values[3] = { 0, 0, 0 };
    bool seen[3] = { false, false, false };
    for (int i = 0; i < 3; ++i)
        {
        sensor::Sample sample;
        REQUIRE (mag.read_sample (sample) == zzs::kernel::Status::OK);
        const uint8_t idx = static_cast<uint8_t> (sample._M_channel)
                            - static_cast<uint8_t> (sensor::Channel::MAG_X);
        REQUIRE (idx < 3u);
        values[idx] = sample._M_sample_q15;
        seen[idx] = true;
        }
    REQUIRE ((seen[0] && seen[1] && seen[2]));
    const int32_t raw_vals[3] = { 700, 800, 900 };
    for (uint32_t i = 0; i < 3; ++i)
        REQUIRE (values[i] == q15_to_raw (q15_mul (q15_from_raw (raw_vals[i]),
                                                   MagSensQ15MilligaussPerLsb)));
}

TEST_CASE ("Data ready flag", "[sensor-driver][lis3mdl][data-ready]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    Lis3mdlMagDriver mag (&bus);
    REQUIRE (mag.init () == zzs::kernel::Status::OK);

    /* Nothing ready -> STALE (any axis). */
    bus.set_reg (RegStatusReg, 0x00);
    sensor::Sample sample;
    REQUIRE (mag.read_sample (sample) == zzs::kernel::Status::STALE);

    /* All axes ready -> OK. */
    bus.set_reg (RegStatusReg,
                 static_cast<uint8_t> (StatusXda | StatusYda | StatusZda));
    uint8_t data[6] = { 5, 0, 6, 0, 7, 0 };
    bus.set_block (RegOutXL, data, 6u);
    REQUIRE (mag.read_sample (sample) == zzs::kernel::Status::OK);
}

TEST_CASE ("Self-test verification", "[sensor-driver][lis3mdl][self-test]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);
    bus.set_reg (RegStatusReg, StatusXda);
    uint8_t data[6] = { 50, 0, 0, 0, 0, 0 };
    bus.set_block (RegOutXL, data, 6u);

    Lis3mdlMagDriver mag (&bus);
    REQUIRE (mag.init () == zzs::kernel::Status::OK);

    /* Mock returns identical bytes, so output does not change -> false. */
    REQUIRE (mag.self_test () == false);
}

TEST_CASE ("Edge cases: invalid device ID, NULL bus", "[sensor-driver][lis3mdl][edge]")
{
    /* Invalid device ID -> BAD_TYPE. */
    {
        test::MockBus bus;
        bus.set_initialized (true);
        bus.set_reg (RegWhoAmI, 0x00);
        Lis3mdlMagDriver mag (&bus);
        REQUIRE (mag.init () == zzs::kernel::Status::BAD_TYPE);
    }

    /* NULL bus -> INTERNAL. */
    {
        Lis3mdlMagDriver mag (nullptr);
        REQUIRE (mag.init () == zzs::kernel::Status::INTERNAL);
    }
}
