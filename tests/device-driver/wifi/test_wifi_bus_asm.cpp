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
 * @file test_wifi_bus_asm.cpp
 * @brief Unit tests for the WiFi bus assembly primitives (x86_64).
 *
 * Calls the actual exported assembly symbols and verifies shift counts
 * 0/1/7/8/15/31 produce correct masks, plus equivalence with the C++ fallback.
 */

#include "kernel/device-driver/wifi/bus/wifi_bus_asm.hpp"
#include "kernel/kernel_status.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

#ifdef ZZS_DEVICE_DRIVER_WIFI_ASM_X86_64

TEST_CASE ("wifi.bus.asm.x86_64.gpio_toggle_delay", "[wifi][bus-asm][x86_64]") {
    for (uint32_t bit : {0u, 1u, 7u, 8u, 15u, 31u}) {
        uint32_t reg = 0x00000000u;
        const uint32_t count = 3u;
        zzs_wifi_bus_gpio_toggle_delay (&reg, bit, count);
        /* Toggling bit `bit` `count` times: if count is odd, bit is flipped;
         * if even, unchanged.  count=3 is odd, so bit should be set. */
        uint32_t expected = 1u << bit;
        REQUIRE (reg == expected);
    }

    /* Null base: no-op. */
    uint32_t reg = 0xdeadbeefu;
    zzs_wifi_bus_gpio_toggle_delay (nullptr, 5u, 100u);
    REQUIRE (reg == 0xdeadbeefu);

    /* Bit >= 32: no-op. */
    reg = 0u;
    zzs_wifi_bus_gpio_toggle_delay (&reg, 32u, 10u);
    REQUIRE (reg == 0u);

    /* Count = 0: no toggles. */
    reg = 0u;
    zzs_wifi_bus_gpio_toggle_delay (&reg, 3u, 0u);
    REQUIRE (reg == 0u);

    /* Count = 1: single toggle. */
    reg = 0u;
    zzs_wifi_bus_gpio_toggle_delay (&reg, 31u, 1u);
    REQUIRE (reg == (1u << 31u));
}

TEST_CASE ("wifi.bus.asm.x86_64.spi_cs_set", "[wifi][bus-asm][x86_64]") {
    for (uint32_t bit : {0u, 1u, 7u, 8u, 15u, 31u}) {
        uint32_t reg = 0xffffffffu;
        zzs_wifi_bus_spi_cs_set (&reg, bit, 1 /* asserted */);
        /* CS active low: bit is cleared. */
        REQUIRE (reg == (0xffffffffu & ~(1u << bit)));

        reg = 0xffffffffu;
        zzs_wifi_bus_spi_cs_set (&reg, bit, 0 /* deasserted */);
        /* CS deasserted: bit is set. */
        REQUIRE (reg == 0xffffffffu);
    }

    /* Null base: no-op. */
    uint32_t reg = 0xdeadbeefu;
    zzs_wifi_bus_spi_cs_set (nullptr, 5u, 1);
    REQUIRE (reg == 0xdeadbeefu);

    /* Bit >= 32: no-op. */
    reg = 0u;
    zzs_wifi_bus_spi_cs_set (&reg, 33u, 1);
    REQUIRE (reg == 0u);
}

TEST_CASE ("wifi.bus.asm.x86_64.equivalence_with_cpp", "[wifi][bus-asm][x86_64]") {
    /* Verify asm matches the C++ fallback for toggle_delay across shift counts. */
    for (uint32_t bit : {0u, 1u, 7u, 8u, 15u, 31u}) {
        uint32_t reg_asm = 0u;
        uint32_t reg_cpp = 0u;
        const uint32_t count = 5u; /* odd -> bit flipped */

        zzs_wifi_bus_gpio_toggle_delay (&reg_asm, bit, count);
        zzs::kernel::device::driver::wifi::bus::asm_impl::wifi_bus_gpio_toggle_delay_cpp (&reg_cpp, bit, count);

        REQUIRE (reg_asm == reg_cpp);
        REQUIRE (reg_asm == (1u << bit));
    }

    /* spi_cs_set equivalence. */
    for (uint32_t bit : {0u, 1u, 7u, 8u, 15u, 31u}) {
        uint32_t reg_asm = 0xffffffffu;
        uint32_t reg_cpp = 0xffffffffu;

        zzs_wifi_bus_spi_cs_set (&reg_asm, bit, 1);
        zzs::kernel::device::driver::wifi::bus::asm_impl::wifi_bus_spi_cs_set_cpp (&reg_cpp, bit, 1);

        REQUIRE (reg_asm == reg_cpp);
        REQUIRE (reg_asm == (0xffffffffu & ~(1u << bit)));
    }
}

#else
TEST_CASE ("wifi.bus.asm.x86_64.skip_when_no_asm", "[wifi][bus-asm]") {
    SUCCEED ("x86_64 assembly not enabled; skipping asm tests.");
}
#endif
