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

# Bus assembly kernels (x86_64 / Cortex-M4 bit-bang routines).
add_subdirectory(bus)

# ---------------------------------------------------------------------------
# Sensor-driver library: aggregates the bus bit-bang layer and the host
# platform HAL.  Provides openzzs-sensor-driver for tests and platform HAL
# consumers.  Links against openzzs-std and openzzs-mllee plus the bus
# assembly library when present.
# ---------------------------------------------------------------------------

set(_zzs_sensor_driver_sources
  bus/i2c/i2c_bitbang.cpp
  bus/spi/spi_bitbang.cpp)

# Host platform HAL: only the translation unit matching this target compiles.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND _zzs_sensor_driver_sources
    platforms/linux/platform_hal.cpp)
endif ()

add_library(openzzs-sensor-driver STATIC ${_zzs_sensor_driver_sources})
target_include_directories(openzzs-sensor-driver PUBLIC
  ${CMAKE_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(openzzs-sensor-driver PRIVATE
  openzzs-std
  openzzs-mllee)
if (TARGET openzzs-sensor-driver-bus-asm)
  target_link_libraries(openzzs-sensor-driver PRIVATE openzzs-sensor-driver-bus-asm)
endif ()
set_target_properties(openzzs-sensor-driver PROPERTIES LINKER_LANGUAGE CXX)
