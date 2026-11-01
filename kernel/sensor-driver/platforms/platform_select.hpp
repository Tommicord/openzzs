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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#pragma once

/**
 * @brief Compile-time platform selection via `#ifdef` macros.
 *
 * Exactly one platform macro should be defined at build time.  The header maps
 * each macro to a `ZZS_KERNEL_PLATFORM_*` identifier used by the build system
 * and by `platform_registry.hpp`.  If no platform macro is defined, a default
 * of `ZZS_KERNEL_PLATFORM_LINUX` is assumed on `__linux__` hosts; otherwise
 * a compile error is emitted.
 *
 * Defined macros (one should be active):
 *   ZZS_KERNEL_PLATFORM_LINUX      Linux /dev/i2c-N /dev/spidev-N
 *   ZZS_KERNEL_PLATFORM_WINDOWS    Windows CreateFile/DeviceIoControl I2C/SPI
 *   ZZS_KERNEL_PLATFORM_STM32F4    STM32F4 HAL_I2C/HAL_SPI
 *   ZZS_KERNEL_PLATFORM_STM32F7    STM32F7 HAL_I2C/HAL_SPI
 *   ZZS_KERNEL_PLATFORM_STM32H7    STM32H7 HAL_I2C/HAL_SPI
 *   ZZS_KERNEL_PLATFORM_STM32L4    STM32L4 low-power HAL_I2C/HAL_SPI
 *   ZZS_KERNEL_PLATFORM_ANDROID    Android NDK ASensorManager / AHardwareBuffer
 *   ZZS_KERNEL_PLATFORM_IOS        iOS CoreMotion / IOKit I2C
 *   ZZS_KERNEL_PLATFORM_ZEPHYR     Zephyr RTOS i2c_dt_spec / spi_dt_spec
 *   ZZS_KERNEL_PLATFORM_FREERTOS   FreeRTOS I2C/SPI driver abstraction
 *   ZZS_KERNEL_PLATFORM_RP2040     Raspberry Pi Pico SDK i2c.h / spi.h
 *   ZZS_KERNEL_PLATFORM_ESP32      ESP-IDF i2c.h / spi_master.h
 *   ZZS_KERNEL_PLATFORM_NRF52      Nordic SDK nrf_drv_twi.h / nrf_drv_spi.h
 *   ZZS_KERNEL_PLATFORM_SAMD21     Atmel SERCOM I2C/SPI
 *   ZZS_KERNEL_PLATFORM_BEAGLEBONE BeagleBone /dev/i2c-N GPIO sysfs
 *   ZZS_KERNEL_PLATFORM_MACOS      macOS IOKit IOI2CInterface
 *   ZZS_KERNEL_PLATFORM_QNX        QNX /dev/i2c<N> devctl
 *   ZZS_KERNEL_PLATFORM_VXWORKS    VxWorks i2cDev/spiDev drivers
 *   ZZS_KERNEL_PLATFORM_RTLINUX    RTLinux (Xenomai) real-time I2C/SPI
 * @{
 */

// ---------------------------------------------------------------------------
// Platform selection macros
// ---------------------------------------------------------------------------

#if defined(ZZS_KERNEL_PLATFORM_LINUX)      + \
    defined(ZZS_KERNEL_PLATFORM_WINDOWS)    + \
    defined(ZZS_KERNEL_PLATFORM_STM32F4)    + \
    defined(ZZS_KERNEL_PLATFORM_STM32F7)    + \
    defined(ZZS_KERNEL_PLATFORM_STM32H7)    + \
    defined(ZZS_KERNEL_PLATFORM_STM32L4)    + \
    defined(ZZS_KERNEL_PLATFORM_ANDROID)    + \
    defined(ZZS_KERNEL_PLATFORM_IOS)        + \
    defined(ZZS_KERNEL_PLATFORM_ZEPHYR)     + \
    defined(ZZS_KERNEL_PLATFORM_FREERTOS)   + \
    defined(ZZS_KERNEL_PLATFORM_RP2040)     + \
    defined(ZZS_KERNEL_PLATFORM_ESP32)      + \
    defined(ZZS_KERNEL_PLATFORM_NRF52)      + \
    defined(ZZS_KERNEL_PLATFORM_SAMD21)     + \
    defined(ZZS_KERNEL_PLATFORM_BEAGLEBONE) + \
    defined(ZZS_KERNEL_PLATFORM_MACOS)      + \
    defined(ZZS_KERNEL_PLATFORM_QNX)        + \
    defined(ZZS_KERNEL_PLATFORM_VXWORKS)    + \
    defined(ZZS_KERNEL_PLATFORM_RTLINUX)
#  if (defined(ZZS_KERNEL_PLATFORM_LINUX)      + \
       defined(ZZS_KERNEL_PLATFORM_WINDOWS)    + \
       defined(ZZS_KERNEL_PLATFORM_STM32F4)    + \
       defined(ZZS_KERNEL_PLATFORM_STM32F7)    + \
       defined(ZZS_KERNEL_PLATFORM_STM32H7)    + \
       defined(ZZS_KERNEL_PLATFORM_STM32L4)    + \
       defined(ZZS_KERNEL_PLATFORM_ANDROID)    + \
       defined(ZZS_KERNEL_PLATFORM_IOS)        + \
       defined(ZZS_KERNEL_PLATFORM_ZEPHYR)     + \
       defined(ZZS_KERNEL_PLATFORM_FREERTOS)   + \
       defined(ZZS_KERNEL_PLATFORM_RP2040)     + \
       defined(ZZS_KERNEL_PLATFORM_ESP32)      + \
       defined(ZZS_KERNEL_PLATFORM_NRF52)      + \
       defined(ZZS_KERNEL_PLATFORM_SAMD21)     + \
       defined(ZZS_KERNEL_PLATFORM_BEAGLEBONE) + \
       defined(ZZS_KERNEL_PLATFORM_MACOS)      + \
       defined(ZZS_KERNEL_PLATFORM_QNX)        + \
       defined(ZZS_KERNEL_PLATFORM_VXWORKS)    + \
       defined(ZZS_KERNEL_PLATFORM_RTLINUX)) != 1
#    error "Exactly one ZZS_KERNEL_PLATFORM_* macro must be defined"
#  endif
#endif

// Default to Linux on __linux__ hosts when nothing is selected.
#if !defined(ZZS_KERNEL_PLATFORM_LINUX)     && \
    !defined(ZZS_KERNEL_PLATFORM_WINDOWS)   && \
    !defined(ZZS_KERNEL_PLATFORM_STM32F4)   && \
    !defined(ZZS_KERNEL_PLATFORM_STM32F7)   && \
    !defined(ZZS_KERNEL_PLATFORM_STM32H7)   && \
    !defined(ZZS_KERNEL_PLATFORM_STM32L4)   && \
    !defined(ZZS_KERNEL_PLATFORM_ANDROID)   && \
    !defined(ZZS_KERNEL_PLATFORM_IOS)       && \
    !defined(ZZS_KERNEL_PLATFORM_ZEPHYR)    && \
    !defined(ZZS_KERNEL_PLATFORM_FREERTOS)  && \
    !defined(ZZS_KERNEL_PLATFORM_RP2040)    && \
    !defined(ZZS_KERNEL_PLATFORM_ESP32)     && \
    !defined(ZZS_KERNEL_PLATFORM_NRF52)     && \
    !defined(ZZS_KERNEL_PLATFORM_SAMD21)    && \
    !defined(ZZS_KERNEL_PLATFORM_BEAGLEBONE)&& \
    !defined(ZZS_KERNEL_PLATFORM_MACOS)     && \
    !defined(ZZS_KERNEL_PLATFORM_QNX)       && \
    !defined(ZZS_KERNEL_PLATFORM_VXWORKS)   && \
    !defined(ZZS_KERNEL_PLATFORM_RTLINUX)
#  ifdef __linux__
#    define ZZS_KERNEL_PLATFORM_LINUX 1
#  else
#    error "No ZZS_KERNEL_PLATFORM_* macro defined; define exactly one (e.g. -DZZS_KERNEL_PLATFORM_LINUX)"
#  endif
#endif

// ---------------------------------------------------------------------------
// Map each macro to a canonical ZZS_KERNEL_PLATFORM_<ID> value
// ---------------------------------------------------------------------------

#if defined(ZZS_KERNEL_PLATFORM_LINUX)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_LINUX
#  define ZZS_KERNEL_PLATFORM_NAME "linux"
#elif defined(ZZS_KERNEL_PLATFORM_WINDOWS)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_WINDOWS
#  define ZZS_KERNEL_PLATFORM_NAME "windows"
#elif defined(ZZS_KERNEL_PLATFORM_STM32F4)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_STM32F4
#  define ZZS_KERNEL_PLATFORM_NAME "stm32/f4"
#elif defined(ZZS_KERNEL_PLATFORM_STM32F7)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_STM32F7
#  define ZZS_KERNEL_PLATFORM_NAME "stm32/f7"
#elif defined(ZZS_KERNEL_PLATFORM_STM32H7)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_STM32H7
#  define ZZS_KERNEL_PLATFORM_NAME "stm32/h7"
#elif defined(ZZS_KERNEL_PLATFORM_STM32L4)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_STM32L4
#  define ZZS_KERNEL_PLATFORM_NAME "stm32/l4"
#elif defined(ZZS_KERNEL_PLATFORM_ANDROID)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_ANDROID
#  define ZZS_KERNEL_PLATFORM_NAME "android"
#elif defined(ZZS_KERNEL_PLATFORM_IOS)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_IOS
#  define ZZS_KERNEL_PLATFORM_NAME "ios"
#elif defined(ZZS_KERNEL_PLATFORM_ZEPHYR)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_ZEPHYR
#  define ZZS_KERNEL_PLATFORM_NAME "zephyr"
#elif defined(ZZS_KERNEL_PLATFORM_FREERTOS)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_FREERTOS
#  define ZZS_KERNEL_PLATFORM_NAME "freertos"
#elif defined(ZZS_KERNEL_PLATFORM_RP2040)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_RP2040
#  define ZZS_KERNEL_PLATFORM_NAME "rp2040"
#elif defined(ZZS_KERNEL_PLATFORM_ESP32)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_ESP32
#  define ZZS_KERNEL_PLATFORM_NAME "esp32"
#elif defined(ZZS_KERNEL_PLATFORM_NRF52)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_NRF52
#  define ZZS_KERNEL_PLATFORM_NAME "nrf52"
#elif defined(ZZS_KERNEL_PLATFORM_SAMD21)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_SAMD21
#  define ZZS_KERNEL_PLATFORM_NAME "samd21"
#elif defined(ZZS_KERNEL_PLATFORM_BEAGLEBONE)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_BEAGLEBONE
#  define ZZS_KERNEL_PLATFORM_NAME "beaglebone"
#elif defined(ZZS_KERNEL_PLATFORM_MACOS)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_MACOS
#  define ZZS_KERNEL_PLATFORM_NAME "macos"
#elif defined(ZZS_KERNEL_PLATFORM_QNX)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_QNX
#  define ZZS_KERNEL_PLATFORM_NAME "qnx"
#elif defined(ZZS_KERNEL_PLATFORM_VXWORKS)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_VXWORKS
#  define ZZS_KERNEL_PLATFORM_NAME "vxworks"
#elif defined(ZZS_KERNEL_PLATFORM_RTLINUX)
#  define ZZS_KERNEL_PLATFORM_ID ZZS_KERNEL_PLATFORM_RTLINUX
#  define ZZS_KERNEL_PLATFORM_NAME "rtlinux"
#endif

/** @} */