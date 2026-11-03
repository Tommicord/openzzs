/*
 * Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

/**
 * @file test_wifi_platform_hal.cpp
 * @brief Unit tests for every WiFi platform HAL factory.
 *
 * Each platform's `create_<platform>_wifi_hal()` factory is exercised directly.
 * On the host (no vendor SDK headers present) every platform takes its
 * portable functional fallback path: the radio-control vtable hooks are
 * populated with validating stubs, and the stream bus is reported as
 * `UNSUPPORTED` (the application supplies its own bus).
 *
 * Hardware-independent: no vendor SDKs, no physical radio, no float, no heap.
 */

#include "kernel/device-driver/wifi/platforms/platform_hal.hpp"
#include "kernel/kernel_status.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

/* `linux` is a predefined macro on Linux hosts; undef it so the
 * zzs::kernel::device::driver::wifi::platform::linux namespace can be named. */
#undef linux

using namespace zzs::kernel;
using namespace zzs::kernel::device::driver::wifi::platform;
namespace wp = zzs::kernel::device::driver::wifi::platform;

// ---------------------------------------------------------------------------
// Helper: populate a WifiHal and check the radio-control hooks.
// ---------------------------------------------------------------------------

namespace {

/** @brief Verify the radio-control vtable hooks are populated and behave.
 *
 * When `hooks_functional` is true (Linux host), hooks return OK.  When false
 * (all non-Linux fallbacks), hooks return UNSUPPORTED because the underlying
 * hardware is not available. */
void
check_hal_hooks (WifiHal &hal, bool hooks_functional) noexcept {
    REQUIRE (hal._M_reset != nullptr);
    REQUIRE (hal._M_set_channel != nullptr);
    REQUIRE (hal._M_set_country != nullptr);
    REQUIRE (hal._M_mac_addr != nullptr);
    REQUIRE (hal._M_delay_ms != nullptr);

    Status expected = hooks_functional ? Status::OK : Status::UNSUPPORTED;

    /* reset */
    REQUIRE (hal._M_reset () == expected);

    /* set_channel: valid channel returns the expected status. */
    REQUIRE (hal._M_set_channel (1u) == expected);
    REQUIRE (hal._M_set_channel (13u) == expected);
    REQUIRE (hal._M_set_channel (165u) == expected);
    /* Out-of-range channels: OUT_OF_RANGE for functional HAL, UNSUPPORTED for fallback. */
    REQUIRE (hal._M_set_channel (0u) == (hooks_functional ? Status::OUT_OF_RANGE : expected));
    REQUIRE (hal._M_set_channel (166u) == (hooks_functional ? Status::OUT_OF_RANGE : expected));

    /* set_country: non-null returns the expected status; nullptr is out of range
     * (functional) or unsupported (fallback). */
    char cc[2] = {'U', 'S'};
    REQUIRE (hal._M_set_country (cc) == expected);
    REQUIRE (hal._M_set_country (nullptr) == (hooks_functional ? Status::OUT_OF_RANGE : expected));

    /* mac_addr: non-null returns the expected status; nullptr is out of range
     * (functional) or unsupported (fallback). */
    uint8_t mac[6] = {0};
    REQUIRE (hal._M_mac_addr (mac) == expected);
    REQUIRE (hal._M_mac_addr (nullptr) == (hooks_functional ? Status::OUT_OF_RANGE : expected));

    /* delay_ms: returns void, no crash. */
    hal._M_delay_ms (0u);
    hal._M_delay_ms (100u);
}

/** @brief Run a factory through its paces: returns UNSUPPORTED bus, hooks set. */
template <typename Factory>
void
test_platform_factory (Factory &&factory, const char *platform_name) noexcept {
    WifiHal hal{};
    auto result = factory (hal);
    /* The portable fallback reports UNSUPPORTED for the stream bus. */
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    /* But the radio-control hooks are populated. */
    check_hal_hooks (hal, /* hooks_functional = */ false);
    (void) platform_name;
}

} // namespace

// ---------------------------------------------------------------------------
// Linux
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.linux.factory", "[wifi][platform-hal][linux]") {
    WifiHal hal{};
    auto result = create_linux_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ true);
}

// ---------------------------------------------------------------------------
// Windows
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.windows.factory", "[wifi][platform-hal][windows]") {
    WifiHal hal{};
    auto result = create_windows_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// STM32F4
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.stm32f4.factory", "[wifi][platform-hal][stm32f4]") {
    WifiHal hal{};
    auto result = create_stm32f4_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// STM32F7
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.stm32f7.factory", "[wifi][platform-hal][stm32f7]") {
    WifiHal hal{};
    auto result = create_stm32f7_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// STM32H7
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.stm32h7.factory", "[wifi][platform-hal][stm32h7]") {
    WifiHal hal{};
    auto result = create_stm32h7_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// STM32L4
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.stm32l4.factory", "[wifi][platform-hal][stm32l4]") {
    WifiHal hal{};
    auto result = create_stm32l4_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// ESP32
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.esp32.factory", "[wifi][platform-hal][esp32]") {
    WifiHal hal{};
    auto result = create_esp32_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// FreeRTOS
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.freertos.factory", "[wifi][platform-hal][freertos]") {
    WifiHal hal{};
    auto result = create_freertos_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// Android
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.android.factory", "[wifi][platform-hal][android]") {
    WifiHal hal{};
    auto result = create_android_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// iOS
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.ios.factory", "[wifi][platform-hal][ios]") {
    WifiHal hal{};
    auto result = create_ios_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// Zephyr
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.zephyr.factory", "[wifi][platform-hal][zephyr]") {
    WifiHal hal{};
    auto result = create_zephyr_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// RP2040
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.rp2040.factory", "[wifi][platform-hal][rp2040]") {
    WifiHal hal{};
    auto result = create_rp2040_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// nRF52
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.nrf52.factory", "[wifi][platform-hal][nrf52]") {
    WifiHal hal{};
    auto result = create_nrf52_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// SAMD21
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.samd21.factory", "[wifi][platform-hal][samd21]") {
    WifiHal hal{};
    auto result = create_samd21_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// BeagleBone
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.beaglebone.factory", "[wifi][platform-hal][beaglebone]") {
    WifiHal hal{};
    auto result = create_beaglebone_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// macOS
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.macos.factory", "[wifi][platform-hal][macos]") {
    WifiHal hal{};
    auto result = create_macos_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// QNX
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.qnx.factory", "[wifi][platform-hal][qnx]") {
    WifiHal hal{};
    auto result = create_qnx_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// VxWorks
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.vxworks.factory", "[wifi][platform-hal][vxworks]") {
    WifiHal hal{};
    auto result = create_vxworks_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}

// ---------------------------------------------------------------------------
// RTLinux
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.platform.rtlinux.factory", "[wifi][platform-hal][rtlinux]") {
    WifiHal hal{};
    auto result = create_rtlinux_wifi_hal (hal);
    REQUIRE (result.has_error ());
    REQUIRE (result.error () == Status::UNSUPPORTED);
    check_hal_hooks (hal, /* hooks_functional = */ false);
}
