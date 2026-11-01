# Copyright(c) 2026 Tommicord
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# * furnished to do so, subject to the following conditions:
# *
# * The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# ---------------------------------------------------------------------------
# device-driver/wifi: WiFi (802.11) antenna driver for microcontrollers and
# embedded platforms.
#
# Mirrors the design of kernel/sensor-driver/:
#   - abstract OOP driver base (wifi_driver.hpp/.cpp)
#   - 802.11 MAC frame parser (wifi_frame.hpp)
#   - encryption / 4-way handshake model (wifi_crypto.hpp, header-only)
#   - streaming byte ring + frame reassembler (stream/)
#   - per-platform HAL translation units (platforms/)
#   - optional assembly bit-bang primitives (bus/asm/)
#
# All library code: noexcept, no heap, no writable statics, no exceptions,
# no RTTI, C++23, integer-only.
# ---------------------------------------------------------------------------

option(OPENZZS_DEVICE_DRIVER_WIFI_BUS_ASM "Build WiFi bus assembly bit-bang routines" ON)

set(_zzs_wifi_asm_target "auto" CACHE STRING
  "WiFi bus assembly target: auto, x86_64, cortex-m4, none")

if (_zzs_wifi_asm_target STREQUAL "auto" AND OPENZZS_DEVICE_DRIVER_WIFI_BUS_ASM)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _zzs_wifi_arch)
  if (_zzs_wifi_arch MATCHES "x86_64|amd64")
    set(_zzs_wifi_asm_target "x86_64")
  elseif (_zzs_wifi_arch MATCHES "cortex|^arm|aarch64")
    set(_zzs_wifi_asm_target "cortex-m4")
  else ()
    set(_zzs_wifi_asm_target "none")
    message(STATUS "device-driver/wifi: no assembly for architecture '${CMAKE_SYSTEM_PROCESSOR}', using C++ fallback")
  endif ()
elseif (NOT OPENZZS_DEVICE_DRIVER_WIFI_BUS_ASM)
  set(_zzs_wifi_asm_target "none")
  message(STATUS "device-driver/wifi: assembly bit-bang disabled by option, using C++ fallback")
endif ()

set(_zzs_wifi_asm_sources "")

if (_zzs_wifi_asm_target STREQUAL "x86_64")
  enable_language(ASM)
  list(APPEND _zzs_wifi_asm_sources
    bus/asm/wifi_bus_asm_x86_64.S)
elseif (_zzs_wifi_asm_target STREQUAL "cortex-m4")
  enable_language(ASM)
  list(APPEND _zzs_wifi_asm_sources
    bus/asm/wifi_bus_asm_arm.S)
elseif (_zzs_wifi_asm_target STREQUAL "none")
  set(_zzs_wifi_asm_sources "")
else ()
  message(FATAL_ERROR "Unknown device-driver/wifi asm target '${_zzs_wifi_asm_target}'")
endif ()

# ---------------------------------------------------------------------------
# WiFi bus assembly static library (only when assembly sources exist).
# ---------------------------------------------------------------------------

if (NOT _zzs_wifi_asm_sources STREQUAL "")
  add_library(openzzs-device-driver-wifi-bus-asm STATIC ${_zzs_wifi_asm_sources})
  set_target_properties(openzzs-device-driver-wifi-bus-asm PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(openzzs-device-driver-wifi-bus-asm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

  if (_zzs_wifi_asm_target STREQUAL "x86_64")
    target_compile_definitions(openzzs-device-driver-wifi-bus-asm PUBLIC ZZS_DEVICE_DRIVER_WIFI_ASM_X86_64=1)
  elseif (_zzs_wifi_asm_target STREQUAL "cortex-m4")
    target_compile_definitions(openzzs-device-driver-wifi-bus-asm PUBLIC ZZS_DEVICE_DRIVER_WIFI_ASM_ARM=1)
    target_compile_options(openzzs-device-driver-wifi-bus-asm PRIVATE -mcpu=cortex-m4 -mthumb)
  endif ()

  target_compile_definitions(openzzs-device-driver-wifi-bus-asm PRIVATE ZZS_DEVICE_DRIVER_WIFI_BUS_ASM_ENABLED=1)
  message(STATUS "device-driver/wifi: bus assembly enabled for ${_zzs_wifi_asm_target}")
else ()
  message(STATUS "device-driver/wifi: using C++ bus fallback")
endif ()

# ---------------------------------------------------------------------------
# C++ bus fallback (only compiled when no assembly target).
# Provides the same extern "C" symbols as the assembly routines.
# ---------------------------------------------------------------------------

if (_zzs_wifi_asm_target STREQUAL "none")
  add_library(opzzs-device-driver-wifi-bus-cpp STATIC
    bus/wifi_bus_asm_cpp.cpp)
  set_target_properties(opzzs-device-driver-wifi-bus-cpp PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(opzzs-device-driver-wifi-bus-cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
  target_compile_definitions(opzzs-device-driver-wifi-bus-cpp PRIVATE ZZS_DEVICE_DRIVER_WIFI_BUS_ASM_ENABLED=1)
endif ()

# ---------------------------------------------------------------------------
# WiFi driver library sources.
# ---------------------------------------------------------------------------

set(_zzs_device_driver_wifi_sources
  wifi_driver.cpp
  wifi_frame.cpp
  stream/wifi_stream.cpp
  generic_wifi_driver.cpp)

# ---------------------------------------------------------------------------
# WiFi platform HAL selection.
#
# Optionally override the platform explicitly (e.g. for cross-compilation).
# When not set, the platform is auto-detected from CMAKE_SYSTEM_NAME below.
# ---------------------------------------------------------------------------

set(_zzs_wifi_platform "auto")

# Map CMAKE_SYSTEM_NAME to a default platform when _zzs_wifi_platform is "auto".
if (_zzs_wifi_platform STREQUAL "auto")
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(_zzs_wifi_platform "linux")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(_zzs_wifi_platform "windows")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    set(_zzs_wifi_platform "macos")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "STM32F4")
    set(_zzs_wifi_platform "stm32/f4")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "STM32F7")
    set(_zzs_wifi_platform "stm32/f7")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "STM32H7")
    set(_zzs_wifi_platform "stm32/h7")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "STM32L4")
    set(_zzs_wifi_platform "stm32/l4")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "ESP32")
    set(_zzs_wifi_platform "esp32")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "FreeRTOS")
    set(_zzs_wifi_platform "freertos")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Zephyr")
    set(_zzs_wifi_platform "zephyr")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "QNX")
    set(_zzs_wifi_platform "qnx")
  elseif (CMAKE_SYSTEM_NAME STREQUAL "VxWorks")
    set(_zzs_wifi_platform "vxworks")
  else ()
    set(_zzs_wifi_platform "linux")
    message(STATUS "device-driver/wifi: unknown system '${CMAKE_SYSTEM_NAME}', defaulting to linux platform HAL")
  endif ()
endif ()

# Platform HAL: only the translation unit matching this target compiles.
set(_zzs_wifi_platform_source "")
set(_zzs_wifi_platform_def "")
if (_zzs_wifi_platform STREQUAL "linux")
  set(_zzs_wifi_platform_source platforms/linux/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_LINUX=1)
elseif (_zzs_wifi_platform STREQUAL "windows")
  set(_zzs_wifi_platform_source platforms/windows/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_WINDOWS=1)
elseif (_zzs_wifi_platform STREQUAL "stm32/f4")
  set(_zzs_wifi_platform_source platforms/stm32/f4/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_STM32F4=1)
elseif (_zzs_wifi_platform STREQUAL "stm32/f7")
  set(_zzs_wifi_platform_source platforms/stm32/f7/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_STM32F7=1)
elseif (_zzs_wifi_platform STREQUAL "stm32/h7")
  set(_zzs_wifi_platform_source platforms/stm32/h7/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_STM32H7=1)
elseif (_zzs_wifi_platform STREQUAL "stm32/l4")
  set(_zzs_wifi_platform_source platforms/stm32/l4/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_STM32L4=1)
elseif (_zzs_wifi_platform STREQUAL "esp32")
  set(_zzs_wifi_platform_source platforms/esp32/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_ESP32=1)
elseif (_zzs_wifi_platform STREQUAL "freertos")
  set(_zzs_wifi_platform_source platforms/freertos/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_FREERTOS=1)
elseif (_zzs_wifi_platform STREQUAL "android")
  set(_zzs_wifi_platform_source platforms/android/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_ANDROID=1)
elseif (_zzs_wifi_platform STREQUAL "ios")
  set(_zzs_wifi_platform_source platforms/ios/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_IOS=1)
elseif (_zzs_wifi_platform STREQUAL "zephyr")
  set(_zzs_wifi_platform_source platforms/zephyr/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_ZEPHYR=1)
elseif (_zzs_wifi_platform STREQUAL "rp2040")
  set(_zzs_wifi_platform_source platforms/rp2040/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_RP2040=1)
elseif (_zzs_wifi_platform STREQUAL "nrf52")
  set(_zzs_wifi_platform_source platforms/nrf52/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_NRF52=1)
elseif (_zzs_wifi_platform STREQUAL "samd21")
  set(_zzs_wifi_platform_source platforms/samd21/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_SAMD21=1)
elseif (_zzs_wifi_platform STREQUAL "beaglebone")
  set(_zzs_wifi_platform_source platforms/beaglebone/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_BEAGLEBONE=1)
elseif (_zzs_wifi_platform STREQUAL "macos")
  set(_zzs_wifi_platform_source platforms/macos/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_MACOS=1)
elseif (_zzs_wifi_platform STREQUAL "qnx")
  set(_zzs_wifi_platform_source platforms/qnx/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_QNX=1)
elseif (_zzs_wifi_platform STREQUAL "vxworks")
  set(_zzs_wifi_platform_source platforms/vxworks/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_VXWORKS=1)
elseif (_zzs_wifi_platform STREQUAL "rtlinux")
  set(_zzs_wifi_platform_source platforms/rtlinux/platform_hal.cpp)
  set(_zzs_wifi_platform_def ZZS_KERNEL_PLATFORM_RTLINUX=1)
else ()
  message(FATAL_ERROR "Unknown device-driver/wifi platform '${_zzs_wifi_platform}'")
endif ()

if (NOT _zzs_wifi_platform_source STREQUAL "")
  list(APPEND _zzs_device_driver_wifi_sources ${_zzs_wifi_platform_source})
endif ()

add_library(openzzs-device-driver-wifi STATIC ${_zzs_device_driver_wifi_sources})
target_include_directories(openzzs-device-driver-wifi PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
target_link_libraries(openzzs-device-driver-wifi PRIVATE openzzs-std)

if (NOT _zzs_wifi_platform_def STREQUAL "")
  target_compile_definitions(openzzs-device-driver-wifi PRIVATE ${_zzs_wifi_platform_def})
endif ()

message(STATUS "device-driver/wifi: platform HAL = '${_zzs_wifi_platform}' (${_zzs_wifi_platform_source})")

if (TARGET openzzs-device-driver-wifi-bus-asm)
  target_link_libraries(openzzs-device-driver-wifi PRIVATE openzzs-device-driver-wifi-bus-asm)
endif ()
if (TARGET opzzs-device-driver-wifi-bus-cpp)
  target_link_libraries(openzzs-device-driver-wifi PRIVATE opzzs-device-driver-wifi-bus-cpp)
endif ()
set_target_properties(openzzs-device-driver-wifi PROPERTIES LINKER_LANGUAGE CXX)
