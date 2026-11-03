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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

/**
 * @file test_wifi_linux_hal.cpp
 * @brief Unit tests for the Linux WiFi HAL: LinuxStreamBus ctor/dtor,
 *        socketpair factory, short-write looping, null-pointer rejection,
 *        closed-fd errors.
 *
 * Uses real unix-domain socketpairs.  Hardware-independent: no physical radio.
 */

#undef linux
#include "kernel/device-driver/wifi/platforms/linux/platform_hal.hpp"
#include "kernel/kernel_status.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <unistd.h>

using namespace zzs::kernel;
using namespace zzs::kernel::device::driver::wifi::platform;

TEST_CASE ("wifi.linux_hal.socketpair_factory", "[wifi][linux-hal]") {
    int fds[2] = {-1, -1};
    REQUIRE (LinuxStreamBus::create_socketpair (fds) == Status::OK);
    REQUIRE (fds[0] >= 0);
    REQUIRE (fds[1] >= 0);
    ::close (fds[0]);
    ::close (fds[1]);
}

TEST_CASE ("wifi.linux_hal.bus_ctor_dtor", "[wifi][linux-hal]") {
    int fds[2] = {-1, -1};
    REQUIRE (LinuxStreamBus::create_socketpair (fds) == Status::OK);

    /* Scope the bus so its destructor runs and closes fds[0]. */
    {
        LinuxStreamBus bus (fds[0]);
        REQUIRE (bus.is_open () == true);
        REQUIRE (bus.fd () == fds[0]);
    }

    /* After destruction, fds[0] should be closed: reading must fail with EBADF. */
    uint8_t dummy = 0;
    ssize_t n = ::read (fds[0], &dummy, 1);
    REQUIRE (n == -1);
    REQUIRE (errno == EBADF);

    ::close (fds[1]);
}

TEST_CASE ("wifi.linux_hal.read_write_roundtrip", "[wifi][linux-hal]") {
    int fds[2] = {-1, -1};
    REQUIRE (LinuxStreamBus::create_socketpair (fds) == Status::OK);

    LinuxStreamBus bus (fds[0]);

    /* Write to fds[1], read from fds[0]. */
    uint8_t tx[8] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08};
    REQUIRE (::write (fds[1], tx, sizeof (tx)) == static_cast<ssize_t> (sizeof (tx)));

    uint8_t rx[8] = {};
    uint16_t filled = 0u;
    REQUIRE (bus.read_bytes (rx, sizeof (rx), &filled) == Status::OK);
    REQUIRE (filled == sizeof (tx));
    REQUIRE (std::memcmp (rx, tx, sizeof (tx)) == 0);

    ::close (fds[0]);
    ::close (fds[1]);
}

TEST_CASE ("wifi.linux_hal.short_write_looping", "[wifi][linux-hal]") {
    int fds[2] = {-1, -1};
    REQUIRE (LinuxStreamBus::create_socketpair (fds) == Status::OK);

    LinuxStreamBus bus (fds[0]);

    /* Write more than fits in the socket buffer; write_bytes() must loop until
     * all bytes are accepted or return a real error.  Use a large buffer to
     * exercise the short-write retry path. */
    constexpr uint16_t kBigLen = 4096;
    uint8_t big[kBigLen] = {};
    for (uint16_t i = 0u; i < kBigLen; ++i)
        big[i] = static_cast<uint8_t> (i & 0xffu);

    /* Drain fds[1] first so the buffer is empty. */
    uint8_t drain[256];
    while (::read (fds[1], drain, sizeof (drain)) > 0)
        ;

    REQUIRE (bus.write_bytes (big, kBigLen) == Status::OK);

    ::close (fds[0]);
    ::close (fds[1]);
}

TEST_CASE ("wifi.linux_hal.null_pointer_rejection", "[wifi][linux-hal]") {
    int fds[2] = {-1, -1};
    REQUIRE (LinuxStreamBus::create_socketpair (fds) == Status::OK);
    LinuxStreamBus bus (fds[0]);

    uint16_t filled = 0u;
    REQUIRE (bus.read_bytes (nullptr, 64, &filled) == Status::OUT_OF_RANGE);
    REQUIRE (bus.read_bytes (nullptr, 0, nullptr) == Status::OUT_OF_RANGE);
    REQUIRE (bus.read_bytes (nullptr, 0, &filled) == Status::OUT_OF_RANGE);
    REQUIRE (bus.write_bytes (nullptr, 4u) == Status::OUT_OF_RANGE);

    ::close (fds[0]);
    ::close (fds[1]);
}

TEST_CASE ("wifi.linux_hal.closed_fd_errors", "[wifi][linux-hal]") {
    int fds[2] = {-1, -1};
    REQUIRE (LinuxStreamBus::create_socketpair (fds) == Status::OK);
    LinuxStreamBus bus (fds[0]);

    /* Close the fd out from under the bus. */
    ::close (fds[0]);

    uint16_t filled = 0u;
    REQUIRE (bus.read_bytes (nullptr, 64, &filled) == Status::OUT_OF_RANGE);
    REQUIRE (bus.read_bytes (nullptr, 0, &filled) == Status::OUT_OF_RANGE);

    uint8_t rx[8] = {};
    REQUIRE (bus.read_bytes (rx, sizeof (rx), &filled) == Status::INTERNAL);
    REQUIRE (filled == 0u);

    uint8_t tx[8] = {1, 2, 3, 4};
    REQUIRE (bus.write_bytes (tx, sizeof (tx)) == Status::INTERNAL);

    ::close (fds[1]);
}

TEST_CASE ("wifi.linux_hal.bus_is_open_after_close", "[wifi][linux-hal]") {
    int fds[2] = {-1, -1};
    REQUIRE (LinuxStreamBus::create_socketpair (fds) == Status::OK);
    LinuxStreamBus bus (fds[0]);

    REQUIRE (bus.is_open () == true);
    /* Close the fd externally; is_open() reflects the stored fd value. */
    ::close (fds[0]);

    uint8_t rx[8] = {};
    uint16_t filled = 0u;
    REQUIRE (bus.read_bytes (rx, sizeof (rx), &filled) == Status::INTERNAL);
    REQUIRE (filled == 0u);

    ::close (fds[1]);
}

TEST_CASE ("wifi.linux_hal.create_socketpair_null_fd_array", "[wifi][linux-hal]") {
    /* create_socketpair with null array should not crash. */
    REQUIRE (LinuxStreamBus::create_socketpair (nullptr) == Status::INTERNAL);
}
