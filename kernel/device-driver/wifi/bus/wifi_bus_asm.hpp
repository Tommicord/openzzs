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

#pragma once

/**
 * @brief WiFi bus assembly primitives with portable C++ fallback.
 *
 * Mirrors `kernel/sensor-driver/bus/asm/`.  Hand-written assembly routines for
 * GPIO bit-bang delay and SPI chip-select toggle, selected at build time by
 * `OPENZZS_DEVICE_DRIVER_WIFI_BUS_ASM` and the asm target (x86_64 / cortex-m4 /
 * none).  When no assembly target is available, the C++ fallback in
 * `wifi_bus_asm_cpp.cpp` provides the same extern "C" symbols.
 *
 * All functions are `extern "C"`, `noexcept`, integer-only, and operate on
 * fixed register/memory addresses supplied by the caller.  No heap, no statics.
 * @{
 */
#ifdef __cplusplus
extern "C" {
#endif

/** @brief Toggle a GPIO bit `count` times as a short bus settle delay.
 *
 * `base` is the GPIO register base address; `bit` is the bit index within the
 * output data register.  Reads the ODR, toggles the bit, writes it back,
 * repeated `count` times.  This is a timing primitive only; real hardware
 * should use a proper timer.
 */
void zzs_wifi_bus_gpio_toggle_delay (volatile uint32_t *base,
                                     uint32_t bit, uint32_t count) noexcept;

/** @brief Assert (low) or deassert (high) a SPI chip-select line.
 *
 * `base` is the GPIO register base address; `bit` is the bit index.  When
 * `asserted` is non-zero the bit is cleared (CS active low); otherwise set.
 */
void zzs_wifi_bus_spi_cs_set (volatile uint32_t *base, uint32_t bit,
                              int asserted) noexcept;

/** @brief Portable C++ implementations (used when no asm target). */
#ifdef __cplusplus
} // extern "C"

namespace zzs::kernel::device::driver::wifi::bus::asm_impl {

/** @brief C++ GPIO bit-toggle delay. */
[[nodiscard]] inline void
wifi_bus_gpio_toggle_delay_cpp (volatile uint32_t *base, uint32_t bit,
                                uint32_t count) noexcept {
  if (base == nullptr || bit >= 32u)
    return;
  uint32_t mask = 1u << bit;
  for (uint32_t i = 0u; i < count; ++i) {
    uint32_t odr = *base;
    *base = odr ^ mask;
  }
}

/** @brief C++ SPI chip-select set. */
[[nodiscard]] inline void
wifi_bus_spi_cs_set_cpp (volatile uint32_t *base, uint32_t bit,
                         int asserted) noexcept {
  if (base == nullptr || bit >= 32u)
    return;
  uint32_t mask = 1u << bit;
  uint32_t odr = *base;
  if (asserted != 0)
    *base = odr & ~mask; /* CS active low */
  else
    *base = odr | mask;  /* CS deasserted */
}

} // namespace zzs::kernel::device::driver::wifi::bus::asm_impl
#endif

/** @} */
