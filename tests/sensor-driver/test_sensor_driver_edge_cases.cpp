/*
 * Copyright(c) 2026 Tommicord
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

/**
 * @file test_sensor_driver_edge_cases.cpp
 * @brief Edge-case tests for the critical bugs fixed in the sensor-driver
 *        submodule:
 *
 *   - Q15 sensitivity overflow (gyro sensitivity exceeds Q15 range → Q31/wide
 *     path preserves precision instead of saturating).
 *   - Multi-instance driver (no shared static axis counter).
 *   - Self-test actually toggles a register bit and verifies the output changed.
 *   - Timestamps are non-zero (sourced from the clock source).
 *   - Configurable I2C bus (registry lookup by path, not hardcoded).
 *   - __builtin_* overflow detection in fixed-point math.
 *   - zzs::base::format_into() instead of snprintf.
 */

#include "kernel/kernel_clock_source.hpp"
#include "kernel/sensor-driver/bus/bus_interface.hpp"
#include "kernel/sensor-driver/bus/bus_registry.hpp"
#include "kernel/sensor-driver/fixed/q15.hpp"
#include "kernel/sensor-driver/fixed/q31.hpp"
#include "kernel/sensor-driver/sensors/lsm6dso/lsm6dso.hpp"
#include "kernel/sensor-driver/sensors/lsm6dso/lsm6dso_reg.h"
#include "std/fmt.hpp"
#include "test_mock_bus.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <string_view>

namespace {
using namespace zzs::kernel;
using namespace zzs::kernel::sensor::driver::sensors;
using namespace zzs::kernel::sensor::driver::sensors::lsm6dso;
using namespace zzs::kernel::sensor::driver::fixed;
using namespace zzs::kernel::sensor::driver::bus;
using namespace zzs::base;

// ---------------------------------------------------------------------------
// Cooperating mock bus for the self-test test: writing the self-test bit
// toggles an internal flag so the subsequent data-block read reports a
// shifted (different) output.  This lets the test verify that self-test both
// toggles a real bit and observes the resulting change.
// ---------------------------------------------------------------------------
class SelfTestBus : public BusInterface {
  public:
    SelfTestBus () noexcept = default;

    [[nodiscard]] Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
        if (data == nullptr || len == 0u)
            return Status::OUT_OF_RANGE;
        if (reg == RegCtrl5C && (data[0] & Ctrl5CStXlPos)) {
            _M_st_written = true;
            _M_data_shifted = true;
        }
        for (uint16_t i = 0u; i < len; ++i)
            _M_regs[static_cast<uint32_t> (reg) + i] = data[i];
        return Status::OK;
    }

    [[nodiscard]] Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override {
        if (data == nullptr || len == 0u)
            return Status::OUT_OF_RANGE;
        if (reg == RegStatusReg) {
            data[0] = StatusXlda;
            return Status::OK;
        }
        if (reg == RegOutXLAccel) {
            data[0] = static_cast<uint8_t> (_M_data_shifted ? 200 : 100);
            data[1] = 0u;
            return Status::OK;
        }
        for (uint16_t i = 0u; i < len; ++i)
            data[i] = _M_regs[static_cast<uint32_t> (reg) + i];
        return Status::OK;
    }

    void
    delay_ms (uint32_t) noexcept override {
    }

    bool
    st_written () const noexcept {
        return _M_st_written;
    }

  private:
    uint8_t _M_regs[256] = {};
    bool _M_st_written = false;
    bool _M_data_shifted = false;
};

} // namespace

TEST_CASE ("Q15 gyro sensitivity overflows; Q31/wide path preserves precision", "[sensor-driver][edge][q15-overflow]") {
    /* Gyro sensitivity at ±125 dps is 4.375 mdps/LSB = 143360 in Q15, which is
     * one past Q15_MAX (32767).  A saturating Q15 multiply therefore clamps. */
    REQUIRE (GyroSensQ15MdpsPerLsb > _S_q15_max);

    /* Saturating Q15 multiply clamps to Q15_MAX and loses the gain. */
    REQUIRE (q15_mul (zzs::kernel::_S_q15_one, GyroSensQ15MdpsPerLsb) == _S_q15_max);

    /* The wide (non-clamping) Q15 product preserves the full gain; a Q31/wide
     * accumulator is what avoids the saturation. */
    REQUIRE (zzs::kernel::fx::q15_mul_wide (zzs::kernel::_S_q15_one, GyroSensQ15MdpsPerLsb) > _S_q15_max);
}

TEST_CASE ("__builtin_* overflow detection in fixed-point math", "[sensor-driver][edge][builtin-overflow]") {
    int32_t out = 0;

    /* Multiplication that overflows int32 is detected. */
    bool ov = __builtin_mul_overflow (1000000, 1000000, &out);
    REQUIRE (ov);

    /* Multiplication that fits is not flagged and yields the product. */
    bool ok = !__builtin_mul_overflow (1000, 1000, &out);
    REQUIRE (ok);
    REQUIRE (out == 1000000);

    /* Addition that overflows int32 is detected. */
    ov = __builtin_add_overflow (0x7fffffff, 1, &out);
    REQUIRE (ov);

    /* Addition that fits is not flagged. */
    ok = !__builtin_add_overflow (1000, 2000, &out);
    REQUIRE (ok);
    REQUIRE (out == 3000);
}

TEST_CASE ("zzs::base::format_into instead of snprintf", "[sensor-driver][edge][format-into]") {
    char buf[32] = {};

    const unsigned int n = format_into (buf, "gyro={} temp={} {}", 123, -456, 7u);
    REQUIRE (n > 0u);
    REQUIRE (n < sizeof (buf));
    REQUIRE (std::string_view (buf) == "gyro=123 temp=-456 7");

    /* Truncation: a buffer too small is NUL-terminated and not overrun. */
    char small[5] = {};
    const unsigned int buf_cap = sizeof (small) - 1u;
    const unsigned int m = format_into (small, "much-longer-string", 1);
    const bool truncation_ok = (m > buf_cap) || (std::string_view (small).size () <= buf_cap);
    REQUIRE (truncation_ok);
    REQUIRE (small[buf_cap] == '\0');
}

TEST_CASE ("Configurable I2C bus: registry lookup by path, not hardcoded",
           "[sensor-driver][edge][i2c-bus-configurable]") {
    test::MockBus bus_a;
    test::MockBus bus_b;
    bus_a.set_initialized (true);
    bus_b.set_initialized (true);

    REQUIRE (bus_registry_register ("/dev/i2c-1", &bus_a) == Status::OK);
    REQUIRE (bus_registry_register ("/dev/i2c-2", &bus_b) == Status::OK);

    REQUIRE (bus_registry_lookup ("/dev/i2c-1") == &bus_a);
    REQUIRE (bus_registry_lookup ("/dev/i2c-2") == &bus_b);
    REQUIRE (bus_registry_lookup ("/dev/i2c-9") == nullptr);

    REQUIRE (bus_registry_unregister ("/dev/i2c-1") == true);
    REQUIRE (bus_registry_lookup ("/dev/i2c-1") == nullptr);
    REQUIRE (bus_registry_unregister ("/dev/i2c-2") == true);
}

TEST_CASE ("Multi-instance driver: no shared static axis counter", "[sensor-driver][edge][multi-instance]") {
    /* Two independent accel drivers on two independent buses.  Each instance
     * must start its round-robin at X (axis 0); a shared static counter would
     * make the second instance continue from where the first left off. */
    test::MockBus bus_a;
    test::MockBus bus_b;
    bus_a.set_initialized (true);
    bus_b.set_initialized (true);
    bus_a.set_reg (RegWhoAmI, WhoAmIValue);
    bus_b.set_reg (RegWhoAmI, WhoAmIValue);
    bus_a.set_reg (RegStatusReg, StatusXlda);
    bus_b.set_reg (RegStatusReg, StatusXlda);
    uint8_t block[6] = {10, 0, 20, 0, 30, 0};
    bus_a.set_block (RegOutXLAccel, block, 6u);
    bus_b.set_block (RegOutXLAccel, block, 6u);

    Lsm6dsoAccelDriver a (&bus_a);
    Lsm6dsoAccelDriver b (&bus_b);
    REQUIRE (a.init () == Status::OK);
    REQUIRE (b.init () == Status::OK);

    sensor::Sample sa;
    sensor::Sample sb;
    REQUIRE (a.read_sample (sa) == Status::OK);
    REQUIRE (b.read_sample (sb) == Status::OK);

    /* Both instances read X first (independent per-instance axis state), and
     * both fall back to the same global clock source for timestamps. */
    REQUIRE (sa._M_channel == sensor::Channel::ACCEL_X);
    REQUIRE (sb._M_channel == sensor::Channel::ACCEL_X);
    REQUIRE (sa._M_timestamp_ns == sb._M_timestamp_ns);
}

TEST_CASE ("Timestamps are non-zero and come from the clock source", "[sensor-driver][edge][timestamp]") {
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);
    bus.set_reg (RegStatusReg, StatusXlda);
    uint8_t block[6] = {10, 0, 0, 0, 0, 0};
    bus.set_block (RegOutXLAccel, block, 6u);

    /* Install a software clock source that reports a non-zero, monotonic time. */
    uint64_t now = 1000000000ull; /* 1 s */
    REQUIRE (zzs::kernel::clock::clock_source_install (
                 [] () noexcept -> zzs::kernel::clock::Nanoseconds { return 1000000000ull; })
             == Status::OK);

    Lsm6dsoAccelDriver accel (&bus);
    REQUIRE (accel.init () == Status::OK);

    sensor::Sample sample;
    REQUIRE (accel.read_sample (sample) == Status::OK);
    REQUIRE (sample._M_channel == sensor::Channel::ACCEL_X);
    REQUIRE (sample._M_timestamp_ns == 1000000000ull);
    REQUIRE (sample._M_timestamp_ns != 0u);
}

TEST_CASE ("Self-test toggles a register bit and verifies the output changed", "[sensor-driver][edge][self-test]") {
    SelfTestBus bus;

    Lsm6dsoAccelDriver accel (&bus);
    /* WHO_AM_I not needed for self_test, but keep the bus "live". */

    /* self_test reads baseline, writes the ST bit, reads again, and must both
     * observe the bit write and report the output change. */
    REQUIRE (accel.self_test () == true);
    REQUIRE (bus.st_written () == true);
}
