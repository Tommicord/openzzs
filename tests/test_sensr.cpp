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
 * @file test_sensr.cpp
 * @brief Unit tests for the portable sensr sensor abstraction.
 *
 * The tests exercise the shared backend machinery (queue, statistics, sampler
 * thread, enable/disable bookkeeping, helpers) through the deterministic mock
 * backend, which behaves identically on every host platform. Platform specific
 * backends are validated for their guaranteed behaviour on the host (Android and
 * iOS constructors return NULL off-platform).
 */

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

extern "C"
{
#include "sensr/sensr.h"
#include "sensr/sensr_android.h"
#include "sensr/sensr_ios.h"
#include "sensr/sensr_mock.h"
}

namespace
{

struct ZZS_sensr_backend_deleter
{
        void operator() (sensr_backend *pBackend) const
        {
                ZZS_sensr_backend_destroy (pBackend);
        }
};

using backend_ptr = std::unique_ptr<sensr_backend, ZZS_sensr_backend_deleter>;

sensr_backend *
make_mock (uint32_t capacity = SENSR_DEFAULT_QUEUE_CAPACITY, bool autoAdvance = false)
{
        ZZS_sensr_backend_config config;
        ZZS_sensr_backend_config_init (&config);
        config.eventQueueCapacity = capacity;

        sensr_backend *pBackend = ZZS_sensr_backend_create (ZZS_sensr_backend_MOCK, &config);
        if (pBackend && !autoAdvance)
                sensr_mock_backend_set_auto_advance (pBackend, 0);
        return pBackend;
}

sensr_event
make_event (sensr_sensor_type type, float value, uint64_t sequence)
{
        sensr_event event;
        std::memset (&event, 0, sizeof (event));
        event.source = ZZS_sensr_backend_MOCK;
        event.sequence = sequence;
        event.sample.type = type;
        event.sample.accuracy = SENSR_ACCURACY_HIGH;
        event.sample.timestampNs = sequence * 1000u;
        event.sample.valueCount = sensr_sensor_type_value_count (type);
        if (event.sample.valueCount == 0)
                event.sample.valueCount = 1;
        event.sample.values[0] = value;
        return event;
}

} // namespace

TEST_CASE ("config defaults are sane", "[sensr][config]")
{
        ZZS_sensr_backend_config config;
        std::memset (&config, 0xAB, sizeof (config));

        ZZS_sensr_backend_config_init (&config);
        REQUIRE (config.eventQueueCapacity == SENSR_DEFAULT_QUEUE_CAPACITY);
        REQUIRE (config.sampleRateHz == SENSR_DEFAULT_SAMPLE_RATE_HZ);
        REQUIRE (config.useSamplerThread == 1);
        REQUIRE (config.androidPackageName == nullptr);

        /* Passing NULL must be a harmless no-op. */
        ZZS_sensr_backend_config_init (nullptr);
}

TEST_CASE ("backend lifecycle via the mock backend", "[sensr][lifecycle]")
{
        backend_ptr backend{make_mock ()};
        REQUIRE (backend != nullptr);

        CHECK (ZZS_sensr_backend_is_initialized (backend.get ()) == 1);
        CHECK (ZZS_sensr_backend_get_kind (backend.get ()) == ZZS_sensr_backend_MOCK);
        CHECK (ZZS_sensr_backend_name (backend.get ()) != nullptr);
        CHECK (std::string (ZZS_sensr_backend_name (backend.get ())) == "mock");
        CHECK (ZZS_sensr_backend_is_running (backend.get ()) == 0);
        CHECK (ZZS_sensr_backend_available (backend.get ()) == 0);
}

TEST_CASE ("destroy and query handle NULL", "[sensr][lifecycle][null]")
{
        ZZS_sensr_backend_destroy (nullptr);

        CHECK (ZZS_sensr_backend_name (nullptr) == nullptr);
        CHECK (ZZS_sensr_backend_get_kind (nullptr) == ZZS_sensr_backend_UNKNOWN);
        CHECK (ZZS_sensr_backend_is_initialized (nullptr) == 0);
        CHECK (ZZS_sensr_backend_is_running (nullptr) == 0);
        CHECK (ZZS_sensr_backend_sensor_count (nullptr) == 0);
        CHECK (ZZS_sensr_backend_available (nullptr) == 0);
        CHECK (ZZS_sensr_backend_is_enabled (nullptr, SENSR_SENSOR_ACCELEROMETER) == 0);
        CHECK (ZZS_sensr_backend_get_rate (nullptr, SENSR_SENSOR_ACCELEROMETER) == 0);
}

TEST_CASE ("factory dispatch", "[sensr][factory]")
{
        CHECK (ZZS_sensr_backend_create (ZZS_sensr_backend_UNKNOWN, nullptr) == nullptr);

        backend_ptr mock{ZZS_sensr_backend_create (ZZS_sensr_backend_MOCK, nullptr)};
        REQUIRE (mock != nullptr);
        CHECK (ZZS_sensr_backend_get_kind (mock.get ()) == ZZS_sensr_backend_MOCK);

#if !defined(__ANDROID__)
        CHECK (ZZS_sensr_backend_create (ZZS_sensr_backend_ANDROID, nullptr) == nullptr);
        CHECK (ZZS_sensr_android_backend_create (nullptr) == nullptr);
#endif

#if !defined(__APPLE__)
        CHECK (ZZS_sensr_backend_create (ZZS_sensr_backend_IOS, nullptr) == nullptr);
        CHECK (ZZS_sensr_ios_backend_create (nullptr) == nullptr);
#endif

        backend_ptr def{ZZS_sensr_backend_create_default (nullptr)};
        REQUIRE (def != nullptr);
        CHECK (ZZS_sensr_backend_is_initialized (def.get ()) == 1);
}

TEST_CASE ("sensor enumeration", "[sensr][sensors]")
{
        backend_ptr backend{make_mock ()};
        REQUIRE (backend != nullptr);

        const size_t count = ZZS_sensr_backend_sensor_count (backend.get ());
        CHECK (count > 0);
        CHECK (count <= SENSR_MAX_SENSORS);

        for (size_t i = 0; i < count; ++i)
                {
                        sensr_sensor_info info;
                        REQUIRE (ZZS_sensr_backend_sensor_at (backend.get (), i, &info) == SENSR_OK);
                        CHECK (info.name[0] != '\0');
                        CHECK (info.type > SENSR_SENSOR_UNKNOWN);
                        CHECK (info.type < SENSR_SENSOR_COUNT);
                }

        CHECK (ZZS_sensr_backend_sensor_at (backend.get (), count, nullptr) == SENSR_ERROR_INVALID_ARGUMENT);

        sensr_sensor_info info;
        CHECK (ZZS_sensr_backend_sensor_at (backend.get (), count, &info) == SENSR_ERROR_NOT_FOUND);

        REQUIRE (ZZS_sensr_backend_find_sensor (backend.get (), SENSR_SENSOR_ACCELEROMETER, &info) == SENSR_OK);
        CHECK (info.available == 1);
        CHECK (info.valueCount == 3);
        CHECK (ZZS_sensr_backend_has_sensor (backend.get (), SENSR_SENSOR_ACCELEROMETER) == 1);

        CHECK (ZZS_sensr_backend_find_sensor (backend.get (), SENSR_SENSOR_UNKNOWN, &info) == SENSR_ERROR_INVALID_ARGUMENT);
        CHECK (ZZS_sensr_backend_find_sensor (nullptr, SENSR_SENSOR_ACCELEROMETER, &info) == SENSR_ERROR_INVALID_ARGUMENT);
}

TEST_CASE ("sensor name round trips", "[sensr][helpers]")
{
        for (int i = 1; i < SENSR_SENSOR_COUNT; ++i)
                {
                        const char *name = sensr_sensor_type_name ((sensr_sensor_type) i);
                        REQUIRE (name != nullptr);
                        CHECK (std::string (name) != "unknown");
                        CHECK (sensr_sensor_type_from_name (name) == (sensr_sensor_type) i);
                }

        CHECK (sensr_sensor_type_from_name (nullptr) == SENSR_SENSOR_UNKNOWN);
        CHECK (sensr_sensor_type_from_name ("not-a-sensor") == SENSR_SENSOR_UNKNOWN);
        CHECK (sensr_sensor_type_name (SENSR_SENSOR_UNKNOWN) != nullptr);
        CHECK (sensr_sensor_type_name ((sensr_sensor_type) -5) != nullptr);
        CHECK (sensr_sensor_type_name ((sensr_sensor_type) SENSR_SENSOR_COUNT) != nullptr);
}

TEST_CASE ("sensor value counts", "[sensr][helpers]")
{
        CHECK (sensr_sensor_type_value_count (SENSR_SENSOR_ACCELEROMETER) == 3);
        CHECK (sensr_sensor_type_value_count (SENSR_SENSOR_GYROSCOPE) == 3);
        CHECK (sensr_sensor_type_value_count (SENSR_SENSOR_LIGHT) == 1);
        CHECK (sensr_sensor_type_value_count (SENSR_SENSOR_ROTATION_VECTOR) == 5);
        CHECK (sensr_sensor_type_value_count (SENSR_SENSOR_MAGNETIC_FIELD_UNCALIBRATED) == 6);
        CHECK (sensr_sensor_type_value_count (SENSR_SENSOR_GYROSCOPE_UNCALIBRATED) == 6);
        CHECK (sensr_sensor_type_value_count (SENSR_SENSOR_UNKNOWN) == 0);
        CHECK (sensr_sensor_type_value_count ((sensr_sensor_type) SENSR_SENSOR_COUNT) == 0);
}

TEST_CASE ("enable/disable lifecycle", "[sensr][enable]")
{
        backend_ptr backend{make_mock ()};
        REQUIRE (backend != nullptr);

        CHECK (ZZS_sensr_backend_is_enabled (backend.get (), SENSR_SENSOR_ACCELEROMETER) == 0);
        CHECK (ZZS_sensr_backend_get_rate (backend.get (), SENSR_SENSOR_ACCELEROMETER) == 0);

        REQUIRE (ZZS_sensr_backend_enable (backend.get (), SENSR_SENSOR_ACCELEROMETER, 50) == SENSR_OK);
        CHECK (ZZS_sensr_backend_is_enabled (backend.get (), SENSR_SENSOR_ACCELEROMETER) == 1);
        CHECK (ZZS_sensr_backend_get_rate (backend.get (), SENSR_SENSOR_ACCELEROMETER) == 50);

        REQUIRE (ZZS_sensr_backend_set_rate (backend.get (), SENSR_SENSOR_ACCELEROMETER, 25) == SENSR_OK);
        CHECK (ZZS_sensr_backend_get_rate (backend.get (), SENSR_SENSOR_ACCELEROMETER) == 25);

        REQUIRE (ZZS_sensr_backend_disable (backend.get (), SENSR_SENSOR_ACCELEROMETER) == SENSR_OK);
        CHECK (ZZS_sensr_backend_is_enabled (backend.get (), SENSR_SENSOR_ACCELEROMETER) == 0);

        /* Disabling again is idempotent. */
        CHECK (ZZS_sensr_backend_disable (backend.get (), SENSR_SENSOR_ACCELEROMETER) == SENSR_OK);
        /* Setting the rate of a disabled sensor fails. */
        CHECK (ZZS_sensr_backend_set_rate (backend.get (), SENSR_SENSOR_ACCELEROMETER, 25) == SENSR_ERROR_NOT_ENABLED);
}

TEST_CASE ("enable rejects invalid arguments", "[sensr][enable][edge]")
{
        backend_ptr backend{make_mock ()};
        REQUIRE (backend != nullptr);

        CHECK (ZZS_sensr_backend_enable (backend.get (), SENSR_SENSOR_UNKNOWN, 50) == SENSR_ERROR_INVALID_ARGUMENT);
        CHECK (ZZS_sensr_backend_enable (backend.get (), (sensr_sensor_type) SENSR_SENSOR_COUNT, 50)
               == SENSR_ERROR_INVALID_ARGUMENT);
        CHECK (ZZS_sensr_backend_enable (nullptr, SENSR_SENSOR_ACCELEROMETER, 50) == SENSR_ERROR_INVALID_ARGUMENT);
        CHECK (ZZS_sensr_backend_disable (nullptr, SENSR_SENSOR_ACCELEROMETER) == SENSR_ERROR_INVALID_ARGUMENT);
        CHECK (ZZS_sensr_backend_disable (backend.get (), SENSR_SENSOR_UNKNOWN) == SENSR_ERROR_INVALID_ARGUMENT);
}

TEST_CASE ("rate clamping", "[sensr][enable][edge]")
{
        backend_ptr backend{make_mock ()};
        REQUIRE (backend != nullptr);

        /* Absurdly high rates are clamped to the module maximum. */
        REQUIRE (ZZS_sensr_backend_enable (backend.get (), SENSR_SENSOR_ACCELEROMETER, 100000u) == SENSR_OK);
        uint32_t rate = ZZS_sensr_backend_get_rate (backend.get (), SENSR_SENSOR_ACCELEROMETER);
        CHECK (rate >= SENSR_MIN_SAMPLE_RATE_HZ);
        CHECK (rate <= SENSR_MAX_SAMPLE_RATE_HZ);

        /* A zero rate falls back to the configured default. */
        REQUIRE (ZZS_sensr_backend_enable (backend.get (), SENSR_SENSOR_GYROSCOPE, 0) == SENSR_OK);
        CHECK (ZZS_sensr_backend_get_rate (backend.get (), SENSR_SENSOR_GYROSCOPE) == SENSR_DEFAULT_SAMPLE_RATE_HZ);

        REQUIRE (ZZS_sensr_backend_disable_all (backend.get ()) == SENSR_OK);
        CHECK (ZZS_sensr_backend_is_enabled (backend.get (), SENSR_SENSOR_ACCELEROMETER) == 0);
        CHECK (ZZS_sensr_backend_is_enabled (backend.get (), SENSR_SENSOR_GYROSCOPE) == 0);
}

TEST_CASE ("fail-enable propagates platform errors", "[sensr][enable]")
{
        backend_ptr backend{make_mock ()};
        REQUIRE (backend != nullptr);

        REQUIRE (sensr_mock_backend_set_fail_enable (backend.get (), 1) == SENSR_OK);
        CHECK (ZZS_sensr_backend_enable (backend.get (), SENSR_SENSOR_ACCELEROMETER, 50)
               == SENSR_ERROR_PLATFORM_FAILURE);
        CHECK (ZZS_sensr_backend_is_enabled (backend.get (), SENSR_SENSOR_ACCELEROMETER) == 0);

        REQUIRE (sensr_mock_backend_set_fail_enable (backend.get (), 0) == SENSR_OK);
        CHECK (ZZS_sensr_backend_enable (backend.get (), SENSR_SENSOR_ACCELEROMETER, 50) == SENSR_OK);
}

TEST_CASE ("event injection preserves FIFO order", "[sensr][queue]")
{
        backend_ptr backend{make_mock (8)};
        REQUIRE (backend != nullptr);

        for (uint64_t i = 1; i <= 5; ++i)
                {
                        sensr_event event = make_event (SENSR_SENSOR_ACCELEROMETER, (float) i, i);
                        REQUIRE (ZZS_sensr_backend_inject_event (backend.get (), &event) == SENSR_OK);
                }

        CHECK (ZZS_sensr_backend_available (backend.get ()) == 5);

        sensr_event out[8];
        size_t count = 0;
        REQUIRE (ZZS_sensr_backend_poll (backend.get (), out, 8, &count) == SENSR_OK);
        REQUIRE (count == 5);
        for (uint64_t i = 0; i < 5; ++i)
                {
                        CHECK (out[i].sequence == i + 1);
                        CHECK (out[i].sample.values[0] == (float) (i + 1));
                }
        CHECK (ZZS_sensr_backend_available (backend.get ()) == 0);
}

TEST_CASE ("ring wraparound keeps ordering", "[sensr][queue][edge]")
{
        backend_ptr backend{make_mock (4)};
        REQUIRE (backend != nullptr);

        /* Interleave inject/poll across the wrap boundary several times. */
        for (int round = 0; round < 6; ++round)
                {
                        sensr_event event = make_event (SENSR_SENSOR_GYROSCOPE, (float) round, (uint64_t) round);
                        REQUIRE (ZZS_sensr_backend_inject_event (backend.get (), &event) == SENSR_OK);

                        sensr_event out;
                        REQUIRE (ZZS_sensr_backend_read (backend.get (), &out, 0) == SENSR_OK);
                        CHECK (out.sample.values[0] == (float) round);
                }
        CHECK (ZZS_sensr_backend_available (backend.get ()) == 0);
}

TEST_CASE ("queue overflow drops events and is reported", "[sensr][queue][edge]")
{
        backend_ptr backend{make_mock (4)};
        REQUIRE (backend != nullptr);

        for (uint64_t i = 1; i <= 4; ++i)
                {
                        sensr_event event = make_event (SENSR_SENSOR_ACCELEROMETER, (float) i, i);
                        REQUIRE (ZZS_sensr_backend_inject_event (backend.get (), &event) == SENSR_OK);
                }

        sensr_event overflow = make_event (SENSR_SENSOR_ACCELEROMETER, 99.0f, 99);
        CHECK (ZZS_sensr_backend_inject_event (backend.get (), &overflow) == SENSR_ERROR_QUEUE_FULL);
        CHECK (ZZS_sensr_backend_available (backend.get ()) == 4);

        sensr_stats stats;
        REQUIRE (ZZS_sensr_backend_get_stats (backend.get (), &stats) == SENSR_OK);
        CHECK (stats.enqueued == 4);
        CHECK (stats.dropped == 1);
        CHECK (stats.produced == 5);

        /* Flushing frees the queue again. */
        REQUIRE (ZZS_sensr_backend_flush (backend.get ()) == SENSR_OK);
        CHECK (ZZS_sensr_backend_available (backend.get ()) == 0);
        CHECK (ZZS_sensr_backend_inject_event (backend.get (), &overflow) == SENSR_OK);
}

TEST_CASE ("stats can be reset", "[sensr][stats]")
{
        backend_ptr backend{make_mock (4)};
        REQUIRE (backend != nullptr);

        sensr_event event = make_event (SENSR_SENSOR_LIGHT, 100.0f, 1);
        REQUIRE (ZZS_sensr_backend_inject_event (backend.get (), &event) == SENSR_OK);

        sensr_event out;
        REQUIRE (ZZS_sensr_backend_read (backend.get (), &out, 0) == SENSR_OK);

        sensr_stats stats;
        REQUIRE (ZZS_sensr_backend_get_stats (backend.get (), &stats) == SENSR_OK);
        CHECK (stats.enqueued == 1);
        CHECK (stats.dequeued == 1);

        REQUIRE (ZZS_sensr_backend_reset_stats (backend.get ()) == SENSR_OK);
        REQUIRE (ZZS_sensr_backend_get_stats (backend.get (), &stats) == SENSR_OK);
        CHECK (stats.enqueued == 0);
        CHECK (stats.dequeued == 0);
        CHECK (stats.dropped == 0);
        CHECK (stats.produced == 0);

        CHECK (ZZS_sensr_backend_get_stats (nullptr, &stats) == SENSR_ERROR_INVALID_ARGUMENT);
        CHECK (ZZS_sensr_backend_get_stats (backend.get (), nullptr) == SENSR_ERROR_INVALID_ARGUMENT);
}

TEST_CASE ("read timeouts", "[sensr][queue][timeout]")
{
        backend_ptr backend{make_mock ()};
        REQUIRE (backend != nullptr);

        sensr_event event;
        CHECK (ZZS_sensr_backend_read (backend.get (), &event, 0) == SENSR_ERROR_WOULD_BLOCK);
        CHECK (ZZS_sensr_backend_wait (backend.get (), 0) == SENSR_ERROR_WOULD_BLOCK);

        const uint64_t start = sensr_monotonic_ns ();
        CHECK (ZZS_sensr_backend_read (backend.get (), &event, 25) == SENSR_ERROR_TIMEOUT);
        const uint64_t elapsed = sensr_monotonic_ns () - start;
        CHECK (elapsed >= 15u * 1000000u);

        CHECK (ZZS_sensr_backend_read (nullptr, &event, 0) == SENSR_ERROR_INVALID_ARGUMENT);
        CHECK (ZZS_sensr_backend_read (backend.get (), nullptr, 0) == SENSR_ERROR_INVALID_ARGUMENT);
        CHECK (ZZS_sensr_backend_wait (nullptr, 0) == SENSR_ERROR_INVALID_ARGUMENT);
}

TEST_CASE ("poll edge cases", "[sensr][queue][edge]")
{
        backend_ptr backend{make_mock ()};
        REQUIRE (backend != nullptr);

        size_t count = 123;
        CHECK (ZZS_sensr_backend_poll (backend.get (), nullptr, 0, &count) == SENSR_OK);
        CHECK (count == 0);

        sensr_event out[4];
        count = 123;
        CHECK (ZZS_sensr_backend_poll (backend.get (), out, 4, &count) == SENSR_OK);
        CHECK (count == 0);

        CHECK (ZZS_sensr_backend_poll (backend.get (), nullptr, 4, &count) == SENSR_ERROR_INVALID_ARGUMENT);
        CHECK (ZZS_sensr_backend_poll (nullptr, out, 4, &count) == SENSR_ERROR_INVALID_ARGUMENT);
}

TEST_CASE ("pump drains the mock clock", "[sensr][mock][queue]")
{
        backend_ptr backend{make_mock ()};
        REQUIRE (backend != nullptr);
        REQUIRE (sensr_mock_backend_get_auto_advance (backend.get ()) == 0);

        REQUIRE (ZZS_sensr_backend_enable (backend.get (), SENSR_SENSOR_ACCELEROMETER, 100) == SENSR_OK);

        /* The first pump produces exactly one sample. */
        CHECK (ZZS_sensr_backend_pump (backend.get (), 0) == 1);
        CHECK (ZZS_sensr_backend_available (backend.get ()) == 1);

        /* Advancing the virtual clock by 95 ms at 100 Hz yields 9 more samples
         * (one for each 10 ms period that elapsed after the first sample). */
        sensr_mock_backend_advance (backend.get (), 95u * 1000000u);
        CHECK (ZZS_sensr_backend_pump (backend.get (), 0) == 9);

        sensr_event out[16];
        size_t count = 0;
        REQUIRE (ZZS_sensr_backend_poll (backend.get (), out, 16, &count) == SENSR_OK);
        REQUIRE (count == 10);
        for (size_t i = 0; i < count; ++i)
                CHECK (out[i].sample.type == SENSR_SENSOR_ACCELEROMETER);
}

TEST_CASE ("mock generation is deterministic", "[sensr][mock]")
{
        backend_ptr a{make_mock ()};
        backend_ptr b{make_mock ()};
        REQUIRE (a != nullptr);
        REQUIRE (b != nullptr);

        REQUIRE (sensr_mock_backend_set_seed (a.get (), 0x12345678u) == SENSR_OK);
        REQUIRE (sensr_mock_backend_set_seed (b.get (), 0x12345678u) == SENSR_OK);
        REQUIRE (ZZS_sensr_backend_enable (a.get (), SENSR_SENSOR_ACCELEROMETER, 100) == SENSR_OK);
        REQUIRE (ZZS_sensr_backend_enable (b.get (), SENSR_SENSOR_ACCELEROMETER, 100) == SENSR_OK);

        sensr_event eventA;
        sensr_event eventB;
        REQUIRE (ZZS_sensr_backend_read (a.get (), &eventA, 0) == SENSR_OK);
        REQUIRE (ZZS_sensr_backend_read (b.get (), &eventB, 0) == SENSR_OK);

        CHECK (eventA.sample.type == eventB.sample.type);
        CHECK (eventA.sample.valueCount == eventB.sample.valueCount);
        for (uint32_t i = 0; i < eventA.sample.valueCount; ++i)
                CHECK (eventA.sample.values[i] == eventB.sample.values[i]);

        CHECK (sensr_mock_backend_generated (a.get ()) == sensr_mock_backend_generated (b.get ()));
}

TEST_CASE ("sampler thread produces and stops", "[sensr][thread]")
{
        backend_ptr backend{make_mock (256, true)};
        REQUIRE (backend != nullptr);

        REQUIRE (ZZS_sensr_backend_enable (backend.get (), SENSR_SENSOR_ACCELEROMETER, 200) == SENSR_OK);

        CHECK (ZZS_sensr_backend_is_running (backend.get ()) == 0);
        REQUIRE (ZZS_sensr_backend_start (backend.get ()) == SENSR_OK);
        CHECK (ZZS_sensr_backend_is_running (backend.get ()) == 1);
        CHECK (ZZS_sensr_backend_start (backend.get ()) == SENSR_ERROR_ALREADY_RUNNING);

        sensr_event event;
        REQUIRE (ZZS_sensr_backend_read (backend.get (), &event, 1000) == SENSR_OK);
        CHECK (event.sample.type == SENSR_SENSOR_ACCELEROMETER);
        CHECK (event.sample.valueCount == 3);
        CHECK (event.source == ZZS_sensr_backend_MOCK);

        REQUIRE (ZZS_sensr_backend_stop (backend.get ()) == SENSR_OK);
        CHECK (ZZS_sensr_backend_is_running (backend.get ()) == 0);
        CHECK (ZZS_sensr_backend_stop (backend.get ()) == SENSR_OK);

        /* Restarting is allowed. */
        REQUIRE (ZZS_sensr_backend_start (backend.get ()) == SENSR_OK);
        REQUIRE (ZZS_sensr_backend_stop (backend.get ()) == SENSR_OK);
}

TEST_CASE ("backends are independent", "[sensr][lifecycle]")
{
        backend_ptr a{make_mock ()};
        backend_ptr b{make_mock ()};
        REQUIRE (a != nullptr);
        REQUIRE (b != nullptr);

        REQUIRE (ZZS_sensr_backend_enable (a.get (), SENSR_SENSOR_LIGHT, 10) == SENSR_OK);
        CHECK (ZZS_sensr_backend_is_enabled (a.get (), SENSR_SENSOR_LIGHT) == 1);
        CHECK (ZZS_sensr_backend_is_enabled (b.get (), SENSR_SENSOR_LIGHT) == 0);

        sensr_event event = make_event (SENSR_SENSOR_LIGHT, 42.0f, 1);
        REQUIRE (ZZS_sensr_backend_inject_event (a.get (), &event) == SENSR_OK);
        CHECK (ZZS_sensr_backend_available (a.get ()) == 1);
        CHECK (ZZS_sensr_backend_available (b.get ()) == 0);
}

TEST_CASE ("capacity is clamped and overflow guarded", "[sensr][config][edge]")
{
        ZZS_sensr_backend_config config;
        ZZS_sensr_backend_config_init (&config);
        config.eventQueueCapacity = 100000000u;

        backend_ptr backend{ZZS_sensr_backend_create (ZZS_sensr_backend_MOCK, &config)};
        REQUIRE (backend != nullptr);
        CHECK (backend->config.eventQueueCapacity < 100000000u);
        CHECK (backend->config.eventQueueCapacity > 0u);

        ZZS_sensr_backend_config zero;
        ZZS_sensr_backend_config_init (&zero);
        zero.eventQueueCapacity = 0;
        backend_ptr defaulted{ZZS_sensr_backend_create (ZZS_sensr_backend_MOCK, &zero)};
        REQUIRE (defaulted != nullptr);
        CHECK (defaulted->config.eventQueueCapacity == SENSR_DEFAULT_QUEUE_CAPACITY);
}

TEST_CASE ("inject validates arguments", "[sensr][queue][edge]")
{
        backend_ptr backend{make_mock ()};
        REQUIRE (backend != nullptr);

        sensr_event event = make_event (SENSR_SENSOR_ACCELEROMETER, 1.0f, 1);
        CHECK (ZZS_sensr_backend_inject_event (nullptr, &event) == SENSR_ERROR_INVALID_ARGUMENT);
        CHECK (ZZS_sensr_backend_inject_event (backend.get (), nullptr) == SENSR_ERROR_INVALID_ARGUMENT);
        CHECK (ZZS_sensr_backend_flush (nullptr) == SENSR_ERROR_INVALID_ARGUMENT);
}

TEST_CASE ("helper string conversions", "[sensr][helpers]")
{
        CHECK (std::string (sensr_accuracy_name (SENSR_ACCURACY_HIGH)) == "high");
        CHECK (std::string (sensr_accuracy_name (SENSR_ACCURACY_NO_CONTACT)) == "no_contact");
        CHECK (std::string (sensr_accuracy_name (12345)) == "unknown");

        CHECK (std::string (ZZS_sensr_backend_kind_name (ZZS_sensr_backend_MOCK)) == "mock");
        CHECK (std::string (ZZS_sensr_backend_kind_name (ZZS_sensr_backend_ANDROID)) == "android");
        CHECK (std::string (ZZS_sensr_backend_kind_name (ZZS_sensr_backend_IOS)) == "ios");
        CHECK (std::string (ZZS_sensr_backend_kind_name (ZZS_sensr_backend_UNKNOWN)) == "unknown");
        CHECK (std::string (ZZS_sensr_backend_kind_name ((ZZS_sensr_backend_kind) 99)) == "unknown");

        CHECK (std::string (sensr_error_to_string (SENSR_OK)) == "Success");
        CHECK (std::string (sensr_error_to_string (SENSR_ERROR_QUEUE_FULL)) == "Event queue full");
        CHECK (std::string (sensr_error_to_string (SENSR_ERROR_TIMEOUT)) == "Timed out");
        CHECK (sensr_error_to_string (SENSR_ERROR_INVALID_ARGUMENT) != nullptr);
}

TEST_CASE ("monotonic clock is non-decreasing", "[sensr][helpers]")
{
        const uint64_t first = sensr_monotonic_ns ();
        const uint64_t second = sensr_monotonic_ns ();
        CHECK (second >= first);
}
