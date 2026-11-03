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
 * @file test_platform_hal.cpp
 * @brief Unit tests for the platform HAL mock (initialization, Q15 sample
 *        reads, channel enable/disable, channel rate query).
 */

#include "kernel/sensor-driver/fixed/q15.hpp"
#include "kernel/sensor-driver/fixed/q31.hpp"
#include "test_mock_bus.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace {
using namespace zzs::kernel::sensor::driver::fixed;
} // namespace

TEST_CASE ("HAL initialization", "[sensor-driver][platform-hal][init]") {
    test::MockHal hal;
    REQUIRE (hal.is_initialized () == false);

    REQUIRE (hal.init () == true);
    REQUIRE (hal.is_initialized () == true);
}

TEST_CASE ("Sample read returns Q15 values", "[sensor-driver][platform-hal][sample]") {
    test::MockHal hal;
    REQUIRE (hal.init () == true);

    hal.set_value (0, 1000);
    hal.set_value (1, -2000);
    hal.set_value (2, 32767);

    REQUIRE (hal.read_channel (0) == 1000);
    REQUIRE (hal.read_channel (1) == -2000);
    REQUIRE (hal.read_channel (2) == _S_q15_max);
}

TEST_CASE ("Channel enable/disable", "[sensor-driver][platform-hal][channel]") {
    test::MockHal hal;
    REQUIRE (hal.init () == true);

    REQUIRE (hal.channel_enabled (0) == false);
    hal.set_channel_enabled (0, true);
    REQUIRE (hal.channel_enabled (0) == true);
    hal.set_channel_enabled (0, false);
    REQUIRE (hal.channel_enabled (0) == false);
}

TEST_CASE ("Channel rate query", "[sensor-driver][platform-hal][rate]") {
    test::MockHal hal;
    REQUIRE (hal.init () == true);

    REQUIRE (hal.channel_rate_hz (0) == 100u);
    REQUIRE (hal.channel_rate_hz (5) == 100u);
    REQUIRE (hal.channel_rate_hz (11) == 100u);
}

TEST_CASE ("Error: NULL HAL, uninitialized HAL", "[sensor-driver][platform-hal][error]") {
    test::MockHal hal;

    /* Uninitialized HAL: reads return 0 (default value), channel not enabled. */
    REQUIRE (hal.read_channel (0) == 0);
    REQUIRE (hal.channel_enabled (0) == false);
    REQUIRE (hal.channel_rate_hz (0) == 100u);

    /* Re-initialization resets nothing but marks initialized. */
    REQUIRE (hal.init () == true);
    REQUIRE (hal.is_initialized () == true);
}
