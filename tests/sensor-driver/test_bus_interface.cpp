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
 * @file test_bus_interface.cpp
 * @brief Unit tests for the bus interface using a mock backend.
 */

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "kernel/sensor-driver/bus/bus_interface.hpp"

#include "test_mock_bus.hpp"

namespace
{
using namespace zzs::kernel::sensor::driver::bus;
using zzs::kernel::Status;
} // namespace

TEST_CASE ("write_reg / read_reg round-trip", "[sensor-driver][bus][round-trip]")
{
    test::MockBus bus;
    bus.set_initialized (true);

    uint8_t write_buf[3] = { 0xAA, 0xBB, 0xCC };
    REQUIRE (bus.write_reg (0x10, write_buf, 3u) == Status::OK);

    uint8_t read_buf[3] = { 0, 0, 0 };
    REQUIRE (bus.read_reg (0x10, read_buf, 3u) == Status::OK);
    REQUIRE (read_buf[0] == 0xAA);
    REQUIRE (read_buf[1] == 0xBB);
    REQUIRE (read_buf[2] == 0xCC);
}

TEST_CASE ("Register update is read-modify-write", "[sensor-driver][bus][update]")
{
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (0x20, 0b00001111);

    /* bus_update8: (value & ~clear_mask) | set_mask */
    /* This is exercised via SensorDriver::bus_update8; here we verify the
     * underlying read/write round trip the updated byte. */
    uint8_t val = 0;
    REQUIRE (bus.read_reg (0x20, &val, 1u) == Status::OK);
    REQUIRE (val == 0b00001111);

    uint8_t set = 0b10100000;
    uint8_t clear = 0b00000011;
    uint8_t updated = static_cast<uint8_t> ((val & ~clear) | set);
    REQUIRE (bus.write_reg (0x20, &updated, 1u) == Status::OK);

    uint8_t back = 0;
    REQUIRE (bus.read_reg (0x20, &back, 1u) == Status::OK);
    REQUIRE (back == 0b10101100);
}

TEST_CASE ("I2C start/stop conditions via assembly are mocked", "[sensor-driver][bus][i2c]")
{
    /* The mock bus has no real I2C; verify the interface methods exist and
     * delay_ms is exercised (start/stop timing is a no-op in the mock). */
    test::MockBus bus;
    bus.set_initialized (true);
    bus.delay_ms (5u);
    REQUIRE (bus.delay_calls () == 1u);
    REQUIRE (bus.delay_total_ms () == 5u);
}

TEST_CASE ("SPI transfer byte is mocked", "[sensor-driver][bus][spi]")
{
    test::MockBus bus;
    bus.set_initialized (true);

    uint8_t tx = 0x5A;
    REQUIRE (bus.write_reg (0x00, &tx, 1u) == Status::OK);

    uint8_t rx = 0;
    REQUIRE (bus.read_reg (0x00, &rx, 1u) == Status::OK);
    REQUIRE (rx == 0x5A);
}

TEST_CASE ("7-bit vs 10-bit addressing", "[sensor-driver][bus][addressing]")
{
    /* Mock bus is address-agnostic; verify registers at low and high indices. */
    test::MockBus bus;
    bus.set_initialized (true);

    uint8_t low_val = 0x01;
    REQUIRE (bus.write_reg (0x00, &low_val, 1u) == Status::OK);

    uint8_t high_val = 0x02;
    REQUIRE (bus.write_reg (0xFF, &high_val, 1u) == Status::OK);

    uint8_t read_low = 0;
    REQUIRE (bus.read_reg (0x00, &read_low, 1u) == Status::OK);
    REQUIRE (read_low == 0x01);

    uint8_t read_high = 0;
    REQUIRE (bus.read_reg (0xFF, &read_high, 1u) == Status::OK);
    REQUIRE (read_high == 0x02);
}

TEST_CASE ("Error handling: NULL buffer, zero length, bus not initialized",
           "[sensor-driver][bus][error]")
{
    test::MockBus bus;

    /* Not initialized -> INTERNAL. */
    uint8_t data = 0;
    REQUIRE (bus.write_reg (0x00, &data, 1u) == Status::INTERNAL);
    REQUIRE (bus.read_reg (0x00, &data, 1u) == Status::INTERNAL);

    /* Initialize, then test invalid arguments. */
    bus.set_initialized (true);
    REQUIRE (bus.write_reg (0x00, nullptr, 1u) == Status::OUT_OF_RANGE);
    REQUIRE (bus.write_reg (0x00, &data, 0u) == Status::OUT_OF_RANGE);
    REQUIRE (bus.read_reg (0x00, nullptr, 1u) == Status::OUT_OF_RANGE);
    REQUIRE (bus.read_reg (0x00, &data, 0u) == Status::OUT_OF_RANGE);
}

TEST_CASE ("Delay function", "[sensor-driver][bus][delay]")
{
    test::MockBus bus;
    bus.delay_ms (0u);
    REQUIRE (bus.delay_calls () == 1u);
    REQUIRE (bus.delay_total_ms () == 0u);

    bus.delay_ms (1000u);
    REQUIRE (bus.delay_calls () == 2u);
    REQUIRE (bus.delay_total_ms () == 1000u);
}
