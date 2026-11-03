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
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * @file test_sensor_stream.cpp
 * @brief Unit tests for the streaming sensor-driver layer (ScanRing,
 *        read_samples burst API, LSM6DSO burst reads).
 *
 * Real use cases:
 *   - fixed ring push/pop/watermark/drop-oldest (no heap)
 *   - one bus burst transaction produces all 3 axes with a shared timestamp
 *   - partial draining of the per-instance sample cache
 *   - channel enable/disable reduces the sample count
 *   - single read_sample() still works (backwards compatibility)
 *   - STALE when no data is ready
 *
 * No hardware, no float, no heap. Every driver talks to a mock bus.
 */

#include "kernel/sensor-driver/sensors/lsm6dso/lsm6dso.hpp"
#include "kernel/sensor-driver/sensors/lsm6dso/lsm6dso_reg.h"
#include "kernel/sensor-driver/stream/sensor_stream.hpp"
#include "test_mock_bus.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace {
using namespace zzs::kernel;
using namespace zzs::kernel::sensor;
using namespace zzs::kernel::sensor::driver::sensors;
using namespace zzs::kernel::sensor::driver::stream;
using namespace zzs::kernel::sensor::driver::sensors::lsm6dso;
} // namespace

TEST_CASE ("ScanRing push/pop ordering and emptiness", "[sensor-driver][stream][ring]") {
    ScanRing ring (4u);
    REQUIRE (ring.empty ());
    REQUIRE (ring.filled () == 0u);

    sensor::Sample s1{};
    s1._M_channel = sensor::Channel::ACCEL_X;
    s1._M_sample_q15 = 100;
    s1._M_timestamp_ns = 1000ull;
    REQUIRE (ring.push (s1) == zzs::kernel::Status::OK);

    sensor::Sample s2{};
    s2._M_channel = sensor::Channel::ACCEL_Y;
    s2._M_sample_q15 = -200;
    s2._M_timestamp_ns = 1001ull;
    REQUIRE (ring.push (s2) == zzs::kernel::Status::OK);

    REQUIRE (ring.filled () == 2u);
    REQUIRE (!ring.empty ());

    sensor::Sample out{};
    REQUIRE (ring.pop (out) == zzs::kernel::Status::OK);
    REQUIRE (out._M_channel == sensor::Channel::ACCEL_X);
    REQUIRE (out._M_sample_q15 == 100);
    REQUIRE (out._M_timestamp_ns == 1000ull);

    REQUIRE (ring.pop (out) == zzs::kernel::Status::OK);
    REQUIRE (out._M_channel == sensor::Channel::ACCEL_Y);
    REQUIRE (out._M_sample_q15 == -200);

    REQUIRE (ring.empty ());
    REQUIRE (ring.pop (out) == zzs::kernel::Status::STALE);
}

TEST_CASE ("ScanRing drop-oldest when full and watermark", "[sensor-driver][stream][ring][watermark]") {
    ScanRing ring (3u);
    ring.set_watermark (2u);

    sensor::Sample s{};
    s._M_sample_q15 = 0;
    s._M_timestamp_ns = 0;
    for (int ch = 0; ch < 5; ++ch) {
        s._M_channel = static_cast<sensor::Channel> (ch);
        s._M_sample_q15 = static_cast<int16_t> (ch * 10);
        REQUIRE (ring.push (s) == zzs::kernel::Status::OK);
    }
    /* Depth 3, 5 pushes -> oldest 2 dropped, 3 remain. */
    REQUIRE (ring.filled () == 3u);
    REQUIRE (ring.full ());
    REQUIRE (ring.watermark_reached ());

    sensor::Sample out{};
    REQUIRE (ring.pop (out) == zzs::kernel::Status::OK);
    /* First remaining sample is ch=2 (ch 0 and 1 were dropped). */
    REQUIRE (out._M_channel == sensor::Channel::MAG_Z);
    REQUIRE (out._M_sample_q15 == 20);
}

TEST_CASE ("LSM6DSO accel streaming burst: one transaction -> 3 axes, one ts",
           "[sensor-driver][stream][lsm6dso][accel-burst]") {
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);
    bus.set_reg (RegStatusReg, StatusXlda);

    /* 6-byte block: X=1000, Y=-2000, Z=3000 (little-endian). */
    uint8_t data[6] = {
        static_cast<uint8_t> (1000 & 0xFF),    static_cast<uint8_t> ((1000 >> 8) & 0xFF),
        static_cast<uint8_t> ((-2000) & 0xFF), static_cast<uint8_t> (((-2000) >> 8) & 0xFF),
        static_cast<uint8_t> (3000 & 0xFF),    static_cast<uint8_t> ((3000 >> 8) & 0xFF),
    };
    bus.set_block (RegOutXLAccel, data, 6u);

    const uint32_t calls_before = bus.read_reg_calls ();

    Lsm6dsoAccelDriver accel (&bus);
    REQUIRE (accel.init () == zzs::kernel::Status::OK);

    const uint32_t calls_after_init = bus.read_reg_calls ();
    /* init performs exactly one register read (WHO_AM_I). */
    REQUIRE (calls_after_init == calls_before + 1u);

    sensor::Sample out[3];
    uint16_t filled = 0u;
    REQUIRE (accel.read_samples (out, 3u, &filled) == zzs::kernel::Status::OK);
    REQUIRE (filled == 3u);

    /* read_samples performs exactly one status read + one 6-byte burst block
     * read (no per-axis transactions). */
    REQUIRE (bus.read_reg_calls () == calls_after_init + 2u);
    REQUIRE (bus.last_read_reg () == RegOutXLAccel);

    /* All three axes share the same timestamp (single scan). */
    REQUIRE (out[0]._M_channel == sensor::Channel::ACCEL_X);
    REQUIRE (out[1]._M_channel == sensor::Channel::ACCEL_Y);
    REQUIRE (out[2]._M_channel == sensor::Channel::ACCEL_Z);
    REQUIRE (out[0]._M_timestamp_ns == out[1]._M_timestamp_ns);
    REQUIRE (out[1]._M_timestamp_ns == out[2]._M_timestamp_ns);

    /* Cache drained: next read triggers a fresh burst. */
    uint16_t filled2 = 0u;
    REQUIRE (accel.read_samples (out, 3u, &filled2) == zzs::kernel::Status::OK);
    REQUIRE (filled2 == 3u);
}

TEST_CASE ("LSM6DSO accel streaming partial drain (max_samples limit)", "[sensor-driver][stream][lsm6dso][partial]") {
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);
    bus.set_reg (RegStatusReg, StatusXlda);
    uint8_t data[6] = {
        static_cast<uint8_t> (10 & 0xFF), 0, static_cast<uint8_t> (20 & 0xFF), 0, static_cast<uint8_t> (30 & 0xFF), 0,
    };
    bus.set_block (RegOutXLAccel, data, 6u);

    Lsm6dsoAccelDriver accel (&bus);
    REQUIRE (accel.init () == zzs::kernel::Status::OK);

    /* Drain one at a time. */
    sensor::Sample out{};
    uint16_t filled = 0u;
    REQUIRE (accel.read_samples (&out, 1u, &filled) == zzs::kernel::Status::OK);
    REQUIRE (filled == 1u);
    REQUIRE (out._M_channel == sensor::Channel::ACCEL_X);

    REQUIRE (accel.read_samples (&out, 1u, &filled) == zzs::kernel::Status::OK);
    REQUIRE (filled == 1u);
    REQUIRE (out._M_channel == sensor::Channel::ACCEL_Y);

    REQUIRE (accel.read_samples (&out, 1u, &filled) == zzs::kernel::Status::OK);
    REQUIRE (filled == 1u);
    REQUIRE (out._M_channel == sensor::Channel::ACCEL_Z);

    /* Cache exhausted until new data arrives; still OK because status says
     * data available and the mock re-supplies the same block. */
    REQUIRE (accel.read_samples (&out, 1u, &filled) == zzs::kernel::Status::OK);
    REQUIRE (filled == 1u);
}

TEST_CASE ("LSM6DSO accel channel enable reduces streamed samples", "[sensor-driver][stream][lsm6dso][channel]") {
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);
    bus.set_reg (RegStatusReg, StatusXlda);
    uint8_t data[6] = {
        static_cast<uint8_t> (5 & 0xFF), 0, static_cast<uint8_t> (6 & 0xFF), 0, static_cast<uint8_t> (7 & 0xFF), 0,
    };
    bus.set_block (RegOutXLAccel, data, 6u);

    Lsm6dsoAccelDriver accel (&bus);
    REQUIRE (accel.init () == zzs::kernel::Status::OK);

    /* Disable Y and Z: only X is streamed. */
    REQUIRE (accel.set_channel_enabled (sensor::Channel::ACCEL_Y, false) == zzs::kernel::Status::OK);
    REQUIRE (accel.set_channel_enabled (sensor::Channel::ACCEL_Z, false) == zzs::kernel::Status::OK);

    sensor::Sample out[3];
    uint16_t filled = 0u;
    REQUIRE (accel.read_samples (out, 3u, &filled) == zzs::kernel::Status::OK);
    REQUIRE (filled == 1u);
    REQUIRE (out[0]._M_channel == sensor::Channel::ACCEL_X);
}

TEST_CASE ("LSM6DSO accel read_sample single still works (back-compat)", "[sensor-driver][stream][lsm6dso][single]") {
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);
    bus.set_reg (RegStatusReg, StatusXlda);
    uint8_t data[6] = {42, 0, 0, 0, 0, 0};
    bus.set_block (RegOutXLAccel, data, 6u);

    Lsm6dsoAccelDriver accel (&bus);
    REQUIRE (accel.init () == zzs::kernel::Status::OK);

    sensor::Sample out{};
    REQUIRE (accel.read_sample (out) == zzs::kernel::Status::OK);
    REQUIRE (out._M_channel == sensor::Channel::ACCEL_X);
}

TEST_CASE ("LSM6DSO accel STALE when no data ready", "[sensor-driver][stream][lsm6dso][stale]") {
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);
    bus.set_reg (RegStatusReg, 0x00u); /* no data */

    Lsm6dsoAccelDriver accel (&bus);
    REQUIRE (accel.init () == zzs::kernel::Status::OK);

    sensor::Sample out{};
    uint16_t filled = 0u;
    REQUIRE (accel.read_samples (&out, 3u, &filled) == zzs::kernel::Status::STALE);
    REQUIRE (filled == 0u);
}

TEST_CASE ("LSM6DSO accel read_samples argument validation", "[sensor-driver][stream][lsm6dso][args]") {
    test::MockBus bus;
    bus.set_initialized (true);
    bus.set_reg (RegWhoAmI, WhoAmIValue);

    Lsm6dsoAccelDriver accel (&bus);
    REQUIRE (accel.init () == zzs::kernel::Status::OK);

    sensor::Sample out{};
    uint16_t filled = 3u;
    REQUIRE (accel.read_samples (nullptr, 3u, &filled) == zzs::kernel::Status::OUT_OF_RANGE);
    REQUIRE (accel.read_samples (&out, 3u, nullptr) == zzs::kernel::Status::OUT_OF_RANGE);

    filled = 3u;
    REQUIRE (accel.read_samples (&out, 0u, &filled) == zzs::kernel::Status::OK);
    REQUIRE (filled == 0u);
}
