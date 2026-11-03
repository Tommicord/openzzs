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

#include "kernel/device-driver/stream_bus.hpp"
#include "kernel/device-driver/wifi/platforms/platform_select.hpp"
#include "kernel/kernel_status.hpp"
#include "std/maybe.hpp"

#include <cstdint>

/**
 * @brief WiFi platform hardware abstraction layer (HAL).
 *
 * Mirrors `kernel/sensor-driver/platforms/platform_hal.hpp`: each platform
 * provides a `create_<platform>_wifi_hal()` factory returning a stream bus
 * adapter plus radio control hooks (reset, set channel, set country).  The HAL
 * is selected at build time by `ZZS_KERNEL_PLATFORM_*` and compiled into the
 * `openzzs-device-driver-wifi` static library.  Platform files are guarded by
 * `#ifdef` so they compile standalone without vendor SDKs.
 * @{
 */
namespace zzs::kernel::device::driver::wifi::platform {

/** @brief Radio control operations exposed by a platform HAL. */
struct WifiHal {
    /** @brief Reset the WiFi radio. */
    zzs::kernel::Status (*_M_reset) (void) noexcept = nullptr;
    /** @brief Set the operating channel (1..165). */
    zzs::kernel::Status (*_M_set_channel) (uint8_t channel) noexcept = nullptr;
    /** @brief Set the regulatory country code (2 ASCII octets). */
    zzs::kernel::Status (*_M_set_country) (const char cc[2]) noexcept = nullptr;
    /** @brief Return the station MAC address (6 octets). */
    zzs::kernel::Status (*_M_mac_addr) (uint8_t mac[6]) noexcept = nullptr;
    /** @brief Platform-specific delay. */
    void (*_M_delay_ms) (uint32_t ms) noexcept = nullptr;
};

/** @brief Linux host HAL (/dev or socket stream bus). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_linux_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief Windows HAL (CreateFile/DeviceIoControl stream). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_windows_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief STM32F4 HAL (STM32Cube UART/SPI). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_stm32f4_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief STM32F7 HAL (STM32Cube UART/SPI). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_stm32f7_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief STM32H7 HAL (STM32Cube UART/SPI). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_stm32h7_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief STM32L4 HAL (STM32Cube low-power UART/SPI). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_stm32l4_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief ESP32 HAL (ESP-IDF UART/SPI). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_esp32_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief FreeRTOS HAL (stream driver abstraction). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_freertos_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief Android HAL (NDK socket / stream). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_android_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief iOS HAL (BSD socket stream). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_ios_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief Zephyr HAL (device stream). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_zephyr_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief RP2040 HAL (Pico SDK UART/SPI). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_rp2040_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief nRF52 HAL (Nordic SDK UART/SPI). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_nrf52_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief SAMD21 HAL (Atmel SERCOM UART/SPI). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_samd21_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief BeagleBone HAL (/dev/ttyS + GPIO sysfs). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_beaglebone_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief macOS HAL (BSD socket stream). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_macos_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief QNX HAL (/dev/ser stream + devctl). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_qnx_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief VxWorks HAL (usSioDrv stream). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_vxworks_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief RTLinux HAL (Xenomai real-time UART stream). */
[[nodiscard]] zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_rtlinux_wifi_hal (WifiHal &hal_out) noexcept;

/** @brief Select and build the active platform HAL at build time. */
[[nodiscard]] inline zzs::base::Maybe<StreamBusInterface *, zzs::kernel::Status>
create_platform_wifi_hal (WifiHal &hal_out) noexcept {
#if defined(ZZS_KERNEL_PLATFORM_LINUX)
    return create_linux_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_WINDOWS)
    return create_windows_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_STM32F4)
    return create_stm32f4_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_STM32F7)
    return create_stm32f7_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_STM32H7)
    return create_stm32h7_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_STM32L4)
    return create_stm32l4_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_ESP32)
    return create_esp32_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_FREERTOS)
    return create_freertos_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_ANDROID)
    return create_android_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_IOS)
    return create_ios_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_ZEPHYR)
    return create_zephyr_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_RP2040)
    return create_rp2040_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_NRF52)
    return create_nrf52_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_SAMD21)
    return create_samd21_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_BEAGLEBONE)
    return create_beaglebone_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_MACOS)
    return create_macos_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_QNX)
    return create_qnx_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_VXWORKS)
    return create_vxworks_wifi_hal (hal_out);
#elif defined(ZZS_KERNEL_PLATFORM_RTLINUX)
    return create_rtlinux_wifi_hal (hal_out);
#else
    (void) hal_out;
    return zzs::kernel::Status::UNSUPPORTED;
#endif
}

/** @} */

} // namespace zzs::kernel::device::driver::wifi::platform
