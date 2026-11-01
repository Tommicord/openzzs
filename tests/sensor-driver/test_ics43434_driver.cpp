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
 * @file test_ics43434_driver.cpp
 * @brief Unit tests for the ICS43434 sound sensor driver (mock bus).
 */

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

#include "kernel/sensor-driver/sensors/ics43434/ics43434.hpp"
#include "kernel/sensor-driver/sensors/ics43434/ics43434_reg.h"

#include "test_mock_bus.hpp"

namespace
{
using namespace zzs::kernel;
using namespace zzs::kernel::sensor::driver::sensors;
using namespace zzs::kernel::sensor::driver::sensors::ics43434;
using namespace zzs::kernel::sensor::driver::fixed;
} // namespace

namespace
{
thread_local int16_t g_pdm_value = 0;

int16_t
pdm_read (void) noexcept
{
    return g_pdm_value;
}
} // namespace

TEST_CASE ("PDM/I2S read", "[sensor-driver][ics43434][pdm]")
{
    test::MockBus bus;
    bus.set_initialized (true);

    g_pdm_value = 100;
    Ics43434SoundDriver drv (&bus, pdm_read);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);

    sensor::Sample sample;

    /* Consume EnvelopeBlock-1 samples: each read_sample returns STALE because
     * the block is not yet full. */
    for (uint32_t i = 0; i < EnvelopeBlock - 1u; ++i)
        REQUIRE (drv.read_sample (sample) == zzs::kernel::Status::STALE);

    /* The EnvelopeBlock-th sample completes the block and yields OK. */
    REQUIRE (drv.read_sample (sample) == zzs::kernel::Status::OK);
    REQUIRE (sample._M_channel == sensor::Channel::SOUND_ENVELOPE);
    REQUIRE (sample._M_sample_q15 > 0);
}

TEST_CASE ("Q15 sound envelope computation", "[sensor-driver][ics43434][envelope]")
{
    test::MockBus bus;
    bus.set_initialized (true);

    /* Silence: every sample is 0 -> envelope stays 0. */
    g_pdm_value = 0;
    Ics43434SoundDriver drv (&bus, pdm_read);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);

    sensor::Sample sample;
    /* Pump several blocks so the exponential smoothing settles. */
    for (int round = 0; round < 10; ++round)
        for (uint32_t i = 0; i < EnvelopeBlock; ++i)
            (void) drv.read_sample (sample);

    REQUIRE (sample._M_channel == sensor::Channel::SOUND_ENVELOPE);
    REQUIRE (sample._M_sample_q15 == 0);
}

TEST_CASE ("Edge cases: silence (envelope = 0), max amplitude",
           "[sensor-driver][ics43434][edge]")
{
    test::MockBus bus;
    bus.set_initialized (true);

    /* Max amplitude sample. */
    g_pdm_value = 32767;
    Ics43434SoundDriver drv (&bus, pdm_read);
    REQUIRE (drv.init () == zzs::kernel::Status::OK);

    sensor::Sample sample;
    for (int round = 0; round < 5; ++round)
        for (uint32_t i = 0; i < EnvelopeBlock; ++i)
            (void) drv.read_sample (sample);

    /* Envelope should be positive and not saturate at Q15_MAX. */
    REQUIRE (sample._M_sample_q15 > 0);
    REQUIRE (sample._M_sample_q15 <= _S_q15_max);
}
