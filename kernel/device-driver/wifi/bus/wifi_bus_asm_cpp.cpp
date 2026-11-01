/* Copyright(c) 2026 Tommicord
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
 * @file wifi_bus_asm_cpp.cpp
 * @brief Portable C++ fallback for the WiFi bus assembly primitives.
 *
 * Provides the same extern "C" symbols as the assembly routines so the bus
 * layer links against a single implementation regardless of whether assembly
 * is available.  Compiled only when the asm target is `none`.
 */

#include "kernel/device-driver/wifi/bus/wifi_bus_asm.hpp"

#include <cstdint>

extern "C" {

void
zzs_wifi_bus_gpio_toggle_delay (volatile uint32_t *base, uint32_t bit,
                                uint32_t count) noexcept {
  zzs::kernel::device::driver::wifi::bus::asm_impl::
      wifi_bus_gpio_toggle_delay_cpp (base, bit, count);
}

void
zzs_wifi_bus_spi_cs_set (volatile uint32_t *base, uint32_t bit,
                         int asserted) noexcept {
  zzs::kernel::device::driver::wifi::bus::asm_impl::wifi_bus_spi_cs_set_cpp (
      base, bit, asserted);
}

} // extern "C"
