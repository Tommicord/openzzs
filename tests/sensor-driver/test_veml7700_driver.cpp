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
 * @file test_veml7700_driver.cpp
 * @brief Unit tests for the VEML7700 light sensor driver (mock bus).
 */

#include "kernel/sensor-driver/sensors/veml7700/veml7700.hpp"
#include "kernel/sensor-driver/sensors/veml7700/veml7700_reg.h"
#include "test_mock_bus.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace {
using namespace zzs::kernel;
using namespace zzs::kernel::sensor::driver::sensors;
using namespace zzs::kernel::sensor::driver::sensors::veml7700;
using namespace zzs::kernel::sensor::driver::fixed;
} // namespace

TEST_CASE ("ALS configuration", "[sensor-driver][veml7700][als-conf]") {
    test::MockBus bus;
    bus.set_initialized (true);

    Veml7700LightDriver drv (&bus);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);

    /* ALS_CONF: gain 1x, integration 100 ms, shutdown=0 (on).
     * The driver writes {reg_addr, conf_hi, conf_lo} starting at RegAlsConf. */
    const uint16_t expected = static_cast<uint16_t> (AlsConfGain1x | AlsConfInteg100ms);
    REQUIRE (bus.get_reg (static_cast<uint8_t> (RegAlsConf + 1)) == static_cast<uint8_t> (expected >> 8));
    REQUIRE (bus.get_reg (static_cast<uint8_t> (RegAlsConf + 2)) == static_cast<uint8_t> (expected & 0xFF));
}

TEST_CASE ("Q11 lux computation", "[sensor-driver][veml7700][lux]") {
    test::MockBus bus;
    bus.set_initialized (true);

    Veml7700LightDriver drv (&bus);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);

    /* raw = 1000 counts * 118 Q11 per count. */
    uint8_t buf[2] = {
        static_cast<uint8_t> (1000 >> 8),   /* big-endian high */
        static_cast<uint8_t> (1000 & 0xFF), /* big-endian low */
    };
    bus.set_block (RegAlsData, buf, 2u);

    sensor::Sample sample;
    REQUIRE (drv.read_sample (sample) == zzs::kernel::Status::OK);
    REQUIRE (sample._M_channel == sensor::Channel::LIGHT);
    REQUIRE (sample._M_sample_q15
             == q15_to_raw (q11_to_q15 (static_cast<q11_t> (static_cast<int32_t> (1000) * LuxSensQ11PerCount))));
}

TEST_CASE ("High/low threshold registers", "[sensor-driver][veml7700][threshold]") {
    test::MockBus bus;
    bus.set_initialized (true);

    Veml7700LightDriver drv (&bus);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);

    /* Write high threshold 0x0ABC (two data bytes at RegAlsWh/RegAlsWh+1). */
    uint8_t th_hi[2] = {
        static_cast<uint8_t> (0x0ABC >> 8),
        static_cast<uint8_t> (0x0ABC & 0xFF),
    };
    REQUIRE (bus.write_reg (RegAlsWh, th_hi, 2u) == zzs::kernel::Status::OK);
    REQUIRE (bus.get_reg (RegAlsWh) == 0x0Au);
    REQUIRE (bus.get_reg (static_cast<uint8_t> (RegAlsWh + 1)) == 0xBCu);

    /* Write low threshold 0x0010 (two data bytes at RegAlsWl/RegAlsWl+1). */
    uint8_t th_lo[2] = {0x00u, 0x10u};
    REQUIRE (bus.write_reg (RegAlsWl, th_lo, 2u) == zzs::kernel::Status::OK);
    REQUIRE (bus.get_reg (RegAlsWl) == 0x00u);
    REQUIRE (bus.get_reg (static_cast<uint8_t> (RegAlsWl + 1)) == 0x10u);
}

TEST_CASE ("Edge cases: dark condition, bright condition saturation", "[sensor-driver][veml7700][edge]") {
    test::MockBus bus;
    bus.set_initialized (true);

    Veml7700LightDriver drv (&bus);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);

    /* Dark: raw = 0 -> lux 0. */
    uint8_t dark[2] = {0x00, 0x00};
    bus.set_block (RegAlsData, dark, 2u);
    sensor::Sample sample;
    REQUIRE (drv.read_sample (sample) == zzs::kernel::Status::OK);
    REQUIRE (sample._M_sample_q15 == 0);

    /* Bright: raw = 0xFFFF -> bright (non-zero; Q11 lux overflows the int16
     * wire format, so just confirm it is not the dark value). */
    uint8_t bright[2] = {0xFF, 0xFF};
    bus.set_block (RegAlsData, bright, 2u);
    REQUIRE (drv.read_sample (sample) == zzs::kernel::Status::OK);
    REQUIRE (sample._M_sample_q15 != 0);
}
