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
 * @file test_tsl2561_driver.cpp
 * @brief Unit tests for the TSL2561 light sensor driver (mock bus).
 */

#include "kernel/sensor-driver/sensors/tsl2561/tsl2561.hpp"
#include "kernel/sensor-driver/sensors/tsl2561/tsl2561_reg.h"
#include "test_mock_bus.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace {
using namespace zzs::kernel;
using namespace zzs::kernel::sensor::driver::sensors;
using namespace zzs::kernel::sensor::driver::sensors::tsl2561;
using namespace zzs::kernel::sensor::driver::fixed;
} // namespace

TEST_CASE ("Power on/off", "[sensor-driver][tsl2561][power]") {
    test::MockBus bus;
    bus.set_initialized (true);

    Tsl2561LightDriver drv (&bus);

    /* Power on byte written to CONTROL during init. */
    uint8_t part = Tsl2561PartNumberValue;
    bus.set_reg (Tsl2561RegPartNumber | CmdRegisterBit, part);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);
    REQUIRE (bus.get_reg (RegControl | CmdRegisterBit) == ControlPowerOn);
}

TEST_CASE ("Gain and timing configuration", "[sensor-driver][tsl2561][config]") {
    test::MockBus bus;
    bus.set_initialized (true);

    uint8_t part = Tsl2561PartNumberValue;
    bus.set_reg (Tsl2561RegPartNumber | CmdRegisterBit, part);

    Tsl2561LightDriver drv (&bus);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);

    /* 1x gain, 402 ms integration time. */
    REQUIRE (bus.get_reg (RegTiming | CmdRegisterBit) == static_cast<uint8_t> (TimingGain1x | TimingInteg402ms));
}

TEST_CASE ("Two-channel read (CH0, CH1)", "[sensor-driver][tsl2561][read]") {
    test::MockBus bus;
    bus.set_initialized (true);

    uint8_t part = Tsl2561PartNumberValue;
    bus.set_reg (Tsl2561RegPartNumber | CmdRegisterBit, part);

    Tsl2561LightDriver drv (&bus);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);

    /* CH0=500, CH1=100 (little-endian pairs). */
    uint8_t data[4] = {
        static_cast<uint8_t> (500 & 0xFF),
        static_cast<uint8_t> ((500 >> 8) & 0xFF),
        static_cast<uint8_t> (100 & 0xFF),
        static_cast<uint8_t> ((100 >> 8) & 0xFF),
    };
    bus.set_block (RegChan0Low | CmdRegisterBit | CmdAutoIncBit, data, 4u);

    sensor::Sample sample;
    REQUIRE (drv.read_sample (sample) == zzs::kernel::Status::OK);
    REQUIRE (sample._M_channel == sensor::Channel::LIGHT);
    REQUIRE (sample._M_sample_q15 == q11_to_q15 (tsl2561_compute_lux_q11 (500, 100)));
}

TEST_CASE ("Q11 lux computation", "[sensor-driver][tsl2561][lux]") {
    /* Dark: ch0 == 0 -> lux 0. */
    REQUIRE (tsl2561_compute_lux_q11 (0, 0) == 0);

    /* Bright, low ratio -> range 1. ch0=1000, ch1=100 (ratio 0.1). */
    q11_t lux = tsl2561_compute_lux_q11 (1000, 100);
    REQUIRE (lux >= 0);

    /* Saturated: ratio > 1.30 -> lux 0. ch0=100, ch1=500 (ratio 5). */
    REQUIRE (tsl2561_compute_lux_q11 (100, 500) == 0);
}

TEST_CASE ("Edge cases: saturation (CH0 max), dark condition (CH0=0)", "[sensor-driver][tsl2561][edge]") {
    /* CH0 max (0xFFFF) with CH1=0 -> range 1, large lux clamped to Q11 max. */
    q11_t lux = tsl2561_compute_lux_q11 (0xFFFFu, 0u);
    REQUIRE (lux <= _S_q11_max);
    REQUIRE (lux >= 0);

    /* Dark condition: CH0 == 0 -> 0 lux regardless of CH1. */
    REQUIRE (tsl2561_compute_lux_q11 (0, 0xFFFFu) == 0);
}
