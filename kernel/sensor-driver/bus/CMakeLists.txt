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
# Sensor-driver bus bit-bang primitives.
#
# Builds either the hand-written assembly bit-bang routines (bus/asm/) or, when
# no assembly is available for the target, the portable C++ fallback
# (i2c/i2c_bitbang_cpp.cpp).  Both produce the same extern "C" symbols, so the
# bus layer (i2c_bitbang.cpp, spi_bitbang.cpp, compiled into openzzs-sensor-driver
# by the top-level CMakeLists.txt) links against this single static library
# regardless of which implementation is used.
# ---------------------------------------------------------------------------

option(OPENZZS_SENSOR_DRIVER_BUS_ASM "Build sensor-driver bus assembly bit-bang routines" ON)

set(_zzs_bus_asm_target "auto" CACHE STRING
  "sensor-driver bus assembly target: auto, x86_64, cortex-m4, none")

if (_zzs_bus_asm_target STREQUAL "auto" AND OPENZZS_SENSOR_DRIVER_BUS_ASM)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _zzs_bus_arch)
  if (_zzs_bus_arch MATCHES "x86_64|amd64")
    set(_zzs_bus_asm_target "x86_64")
  elseif (_zzs_bus_arch MATCHES "cortex|^arm|aarch64")
    set(_zzs_bus_asm_target "cortex-m4")
  else ()
    set(_zzs_bus_asm_target "none")
    message(STATUS "sensor-driver/bus: no assembly for architecture '${CMAKE_SYSTEM_PROCESSOR}', using C++ fallback")
  endif ()
elseif (NOT OPENZZS_SENSOR_DRIVER_BUS_ASM)
  set(_zzs_bus_asm_target "none")
  message(STATUS "sensor-driver/bus: assembly bit-bang disabled by option, using C++ fallback")
endif ()

set(_zzs_bus_asm_sources "")

if (_zzs_bus_asm_target STREQUAL "x86_64")
  enable_language(ASM)
  list(APPEND _zzs_bus_asm_sources
    asm/i2c_bitbang_x86_64.S
    asm/spi_bitbang_x86_64.S)
elseif (_zzs_bus_asm_target STREQUAL "cortex-m4")
  enable_language(ASM)
  list(APPEND _zzs_bus_asm_sources
    asm/i2c_bitbang_arm.S
    asm/spi_bitbang_arm.S)
elseif (_zzs_bus_asm_target STREQUAL "none")
  set(_zzs_bus_asm_sources "")
else ()
  message(FATAL_ERROR "Unknown sensor-driver/bus asm target '${_zzs_bus_asm_target}'")
endif ()

# ---------------------------------------------------------------------------
# Assembly bit-bang static library (only when assembly sources exist).
# ---------------------------------------------------------------------------

if (NOT _zzs_bus_asm_sources STREQUAL "")
  add_library(openzzs-sensor-driver-bus-asm STATIC ${_zzs_bus_asm_sources})
  set_target_properties(openzzs-sensor-driver-bus-asm PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(openzzs-sensor-driver-bus-asm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

  if (_zzs_bus_asm_target STREQUAL "x86_64")
    target_compile_definitions(openzzs-sensor-driver-bus-asm PUBLIC ZZS_SENSOR_DRIVER_ASM_X86_64=1)
  elseif (_zzs_bus_asm_target STREQUAL "cortex-m4")
    target_compile_definitions(openzzs-sensor-driver-bus-asm PUBLIC ZZS_SENSOR_DRIVER_ASM_ARM=1)
    target_compile_options(openzzs-sensor-driver-bus-asm PRIVATE -mcpu=cortex-m4 -mthumb)
  endif ()

  target_compile_definitions(openzzs-sensor-driver-bus-asm PRIVATE ZZS_SENSOR_DRIVER_BUS_ASM_ENABLED=1)
  message(STATUS "sensor-driver/bus: assembly bit-bang enabled for ${_zzs_bus_asm_target}")
else ()
  message(STATUS "sensor-driver/bus: using C++ bit-bang fallback")
endif ()

# ---------------------------------------------------------------------------
# C++ bit-bang fallback (only compiled when no assembly target).
# Provides the same extern "C" symbols as the assembly routines.
# ---------------------------------------------------------------------------

if (_zzs_bus_asm_target STREQUAL "none")
  add_library(openzzs-sensor-driver-bus-cpp STATIC
    i2c/i2c_bitbang_cpp.cpp)
  set_target_properties(openzzs-sensor-driver-bus-cpp PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(openzzs-sensor-driver-bus-cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
  target_compile_definitions(openzzs-sensor-driver-bus-cpp PRIVATE ZZS_SENSOR_DRIVER_BUS_ASM_ENABLED=1)
endif ()
