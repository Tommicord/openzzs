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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ---------------------------------------------------------------------------
# device-driver: WiFi and GNSS drivers for microcontrollers and embedded
# platforms.
#
# Mirrors the design of kernel/sensor-driver/:
#   - abstract driver base class (OOP, virtual init/read/deinit)
#   - streaming ring buffers (fixed C arrays, no heap)
#   - per-platform HAL translation units
#   - mock bus/HAL for host tests (Catch2)
#
# All library code: noexcept, no heap, no writable statics, no exceptions,
# no RTTI, C++23, integer-only.
# ---------------------------------------------------------------------------

option(OPENZZS_DEVICE_DRIVER_WIFI "Build the WiFi driver module" ON)
option(OPENZZS_DEVICE_DRIVER_GNSS "Build the GNSS driver module" ON)

# ---------------------------------------------------------------------------
# GNSS module
# ---------------------------------------------------------------------------
if (OPENZZS_DEVICE_DRIVER_GNSS)
  set(_zzs_device_driver_gnss_sources
    gnss/gnss_driver.cpp
    gnss/nmea/nmea_parser.cpp
    gnss/nmea/nmea_checksum.cpp
    gnss/ubx/ubx_parser.cpp
    gnss/stream/gnss_stream.cpp
    gnss/gnss_ublox.cpp)

  # Host platform HAL (only on Linux hosts) - must be appended BEFORE
  # add_library so the source is actually compiled into the target.
  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND _zzs_device_driver_gnss_sources
      gnss/platforms/linux/platform_hal.cpp)
  endif ()

  add_library(openzzs-device-driver-gnss STATIC ${_zzs_device_driver_gnss_sources})
  target_include_directories(openzzs-device-driver-gnss PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
  target_link_libraries(openzzs-device-driver-gnss PRIVATE openzzs-std)
  set_target_properties(openzzs-device-driver-gnss PROPERTIES LINKER_LANGUAGE CXX)
endif ()

# ---------------------------------------------------------------------------
# WiFi module
# ---------------------------------------------------------------------------
if (OPENZZS_DEVICE_DRIVER_WIFI)
  set(_zzs_device_driver_wifi_sources
    wifi/wifi_driver.cpp
    wifi/wifi_frame.cpp
    wifi/stream/wifi_stream.cpp)

  add_library(openzzs-device-driver-wifi STATIC ${_zzs_device_driver_wifi_sources})
  target_include_directories(openzzs-device-driver-wifi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
  target_link_libraries(openzzs-device-driver-wifi PRIVATE openzzs-std)
  set_target_properties(openzzs-device-driver-wifi PROPERTIES LINKER_LANGUAGE CXX)

  if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    list(APPEND _zzs_device_driver_wifi_sources
      wifi/platforms/linux/platform_hal.cpp)
  endif ()
endif ()
