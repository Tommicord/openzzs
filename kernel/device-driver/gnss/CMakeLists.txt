# Copyright(c) 2026 Tommicord
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
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
# GNSS device driver module
# ---------------------------------------------------------------------------
option(OPENZZS_DEVICE_DRIVER_GNSS "Build the GNSS driver module" ON)

if (OPENZZS_DEVICE_DRIVER_GNSS)

  set(_zzs_device_driver_gnss_sources
    gnss_driver.cpp
    nmea/nmea_parser.cpp
    nmea/nmea_checksum.cpp
    ubx/ubx_parser.cpp
    stream/gnss_stream.cpp
    gnss_ublox.cpp
    platforms/platform_hal.cpp)

  add_library(openzzs-device-driver-gnss STATIC ${_zzs_device_driver_gnss_sources})
  target_include_directories(openzzs-device-driver-gnss PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
  target_link_libraries(openzzs-device-driver-gnss PRIVATE openzzs-std)
  set_target_properties(openzzs-device-driver-gnss PROPERTIES LINKER_LANGUAGE CXX)

  # Platform HAL: select exactly one translation unit by platform.
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND _zzs_device_driver_gnss_sources
      platforms/linux/platform_hal.cpp)
    target_compile_definitions(openzzs-device-driver-gnss PRIVATE
      ZZS_DEVICE_DRIVER_GNSS_PLATFORM_LINUX=1)
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Generic" AND
          CMAKE_SYSTEM_PROCESSOR STREQUAL "arm")
    list(APPEND _zzs_device_driver_gnss_sources
      platforms/stm32/f4/platform_hal.cpp)
    target_compile_definitions(openzzs-device-driver-gnss PRIVATE
      ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32F4=1)
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Windows")
    list(APPEND _zzs_device_driver_gnss_sources
      platforms/windows/platform_hal.cpp)
    target_compile_definitions(openzzs-device-driver-gnss PRIVATE
      ZZS_DEVICE_DRIVER_GNSS_PLATFORM_WINDOWS=1)
  elseif (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    list(APPEND _zzs_device_driver_gnss_sources
      platforms/macos/platform_hal.cpp)
    target_compile_definitions(openzzs-device-driver-gnss PRIVATE
      ZZS_DEVICE_DRIVER_GNSS_PLATFORM_MACOS=1)
  else ()
    # Default to Linux for any other host target.
    list(APPEND _zzs_device_driver_gnss_sources
      platforms/linux/platform_hal.cpp)
    target_compile_definitions(openzzs-device-driver-gnss PRIVATE
      ZZS_DEVICE_DRIVER_GNSS_PLATFORM_LINUX=1)
  endif ()

endif ()
