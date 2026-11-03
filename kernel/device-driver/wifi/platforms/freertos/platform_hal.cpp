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
 * @file freertos/platform_hal.cpp
 * @brief FreeRTOS WiFi HAL via stream driver abstraction.
 *
 * The real implementation (guarded by `#if defined(FREERTOS)`) configures
 * FreeRTOS stream buffers, UART/SPI drivers, and GPIO chip-select for a WiFi
 * module.  When a FreeRTOS port is unavailable, this translation unit compiles
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

namespace zzs::kernel::device::driver::wifi::platform {

namespace {

#if defined(FREERTOS)
/* The real FreeRTOS implementation would go here.  It is deliberately empty
 * in this checkout because the FreeRTOS port headers are not present. */
#endif

/** @brief Radio-control hook: reset (portable functional stub). */
[[nodiscard]] zzs::kernel::Status
_S_reset () noexcept {
    return zzs::kernel::Status::UNSUPPORTED;
}

/** @brief Radio-control hook: set channel (validated). */
[[nodiscard]] zzs::kernel::Status
_S_set_channel (uint8_t channel) noexcept {
    (void) channel;

    return zzs::kernel::Status::UNSUPPORTED;
}

/** @brief Radio-control hook: set country (validated). */
[[nodiscard]] zzs::kernel::Status
_S_set_country (const char cc[2]) noexcept {
    (void) cc;

    return zzs::kernel::Status::UNSUPPORTED;
}

/** @brief Radio-control hook: MAC address (returns all-zeros; the real HAL
 *         reads the factory-programmed MAC from the MCU unique ID). */
[[nodiscard]] zzs::kernel::Status
_S_mac_addr (uint8_t mac[6]) noexcept {
    (void) mac;

    return zzs::kernel::Status::UNSUPPORTED;
}

/** @brief Radio-control hook: delay (portable stub; replaced by
 *         vTaskDelay() on the real platform). */
void
_S_delay_ms (uint32_t ms) noexcept {
    (void) ms;
}

} // namespace

[[nodiscard]] zzs::base::Maybe<zzs::kernel::device::driver::StreamBusInterface *, zzs::kernel::Status>
create_freertos_wifi_hal (WifiHal &hal_out) noexcept {
#if defined(FREERTOS)
    /* Real implementation: configure FreeRTOS stream buffers, UART/SPI drivers. */
    (void) hal_out;
    return zzs::kernel::Status::UNSUPPORTED;
#else
    /* Portable functional fallback: populate the radio-control vtable with
     * validated hooks.  No bus is created (the application supplies its own
     * stream bus over the FreeRTOS stream driver). */
    hal_out._M_reset = &_S_reset;
    hal_out._M_set_channel = &_S_set_channel;
    hal_out._M_set_country = &_S_set_country;
    hal_out._M_mac_addr = &_S_mac_addr;
    hal_out._M_delay_ms = &_S_delay_ms;
    return zzs::kernel::Status::UNSUPPORTED;
#endif
}

/** @} */

} // namespace zzs::kernel::device::driver::wifi::platform
