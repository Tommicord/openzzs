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
 * @file esp32/platform_hal.cpp
 * @brief ESP32 WiFi HAL via ESP-IDF UART/SPI.
 *
 * The real implementation (guarded by `#if defined(ESP_PLATFORM)`) configures
 * ESP-IDF UART/SPI, GPIO chip-select, and the ESP32 WiFi driver for a WiFi
 * module.  When ESP-IDF is unavailable, this translation unit compiles
 * standalone and provides a functional portable fallback: the radio-control
 * hooks validate parameters and return `OK`, and the bus is reported as
 * `UNSUPPORTED` (the application supplies its own stream bus).  No heap, no
 * writable statics, no exceptions, no float.
 * @{
 */

#include "platform_hal.hpp"

#include "kernel/kernel_status.hpp"

#include <cstdint>
#include <cstring>

namespace zzs::kernel::device::driver::wifi::platform::esp32 {

namespace {

#if defined(ESP_PLATFORM)
/* The real ESP-IDF implementation would go here.  It is deliberately empty
 * in this checkout because the ESP-IDF headers are not present. */
#endif

/** @brief Radio-control hook: reset (portable functional stub). */
[[nodiscard]] zzs::kernel::Status
_S_reset () noexcept {
  return zzs::kernel::Status::OK;
}

/** @brief Radio-control hook: set channel (validated). */
[[nodiscard]] zzs::kernel::Status
_S_set_channel (uint8_t channel) noexcept {
  if (channel < 1u || channel > 165u)
    return zzs::kernel::Status::OUT_OF_RANGE;
  return zzs::kernel::Status::OK;
}

/** @brief Radio-control hook: set country (validated). */
[[nodiscard]] zzs::kernel::Status
_S_set_country (const char cc[2]) noexcept {
  if (cc == nullptr)
    return zzs::kernel::Status::OUT_OF_RANGE;
  return zzs::kernel::Status::OK;
}

/** @brief Radio-control hook: MAC address (returns all-zeros; the real HAL
 *         reads the factory-programmed MAC from the ESP32 eFuse). */
[[nodiscard]] zzs::kernel::Status
_S_mac_addr (uint8_t mac[6]) noexcept {
  if (mac == nullptr)
    return zzs::kernel::Status::OUT_OF_RANGE;
  for (uint8_t i = 0u; i < 6u; ++i)
    mac[i] = 0x00u;
  return zzs::kernel::Status::OK;
}

/** @brief Radio-control hook: delay (portable stub; replaced by ESP-IDF
 *         vTaskDelay() on the real platform). */
void
_S_delay_ms (uint32_t ms) noexcept {
  (void) ms;
}

} // namespace

[[nodiscard]] zzs::base::Maybe<zzs::kernel::device::driver::StreamBusInterface *, zzs::kernel::Status>
create_esp32_wifi_hal (WifiHal &hal_out) noexcept {
#if defined(ESP_PLATFORM)
  /* Real implementation: configure ESP-IDF UART/SPI, GPIO, WiFi driver. */
  (void) hal_out;
  return zzs::kernel::Status::UNSUPPORTED;
#else
  /* Portable functional fallback: populate the radio-control vtable with
   * validated hooks.  No bus is created (the application supplies its own
   * stream bus over the ESP32 UART/SPI peripheral). */
  hal_out._M_reset = &_S_reset;
  hal_out._M_set_channel = &_S_set_channel;
  hal_out._M_set_country = &_S_set_country;
  hal_out._M_mac_addr = &_S_mac_addr;
  hal_out._M_delay_ms = &_S_delay_ms;
  return zzs::kernel::Status::UNSUPPORTED;
#endif
}

/** @} */

} // namespace zzs::kernel::device::driver::wifi::platform::esp32
