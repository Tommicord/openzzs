cmake_minimum_required(VERSION 3.20)
project(openzzs VERSION 1.0.0 LANGUAGES CXX ASM)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
  add_compile_options(/Oy-)
else ()
  add_compile_options(-fno-omit-frame-pointer)
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(
    -Oz
    -ffunction-sections
    -fdata-sections
    -ffile-prefix-map=${CMAKE_SOURCE_DIR}=.
    -fdebug-prefix-map=${CMAKE_SOURCE_DIR}=.)
  add_link_options(-Wl,--gc-sections -Wl,--strip-all -Wl,--build-id=none)
  set(CMAKE_BUILD_RPATH "")
  set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
  set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> qcD <TARGET> <LINK_FLAGS> <OBJECTS>")
  set(OPENZZS_RUNTIME_RELEASE_OPTIONS
    -fno-unwind-tables
    -fno-asynchronous-unwind-tables)
endif ()

set(_zzs_std_sources
  std/basic_allocator.hpp
  std/basic_iterator.hpp
  std/fmt_sink.hpp
  std/fmt.hpp
  std/io.hpp
  std/log.hpp
  std/maybe.hpp
  std/signal_handler.hpp
  std/secure_channel.hpp
  std/swiss_table.hpp
  std/string_view.hpp
  std/sunwind.hpp
)
add_library(openzzs-std STATIC ${_zzs_std_sources})
target_include_directories(openzzs-std PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(openzzs-std PROPERTIES LINKER_LANGUAGE CXX)

# ---------------------------------------------------------------------------
# mllee optional SIMD intrinsics kernels.
#
# The instruction-set extensions are selected at configure time (never probed
# at run time) so the hot path stays free of CPUID branches, dispatch tables
# and hidden module state.  That is what makes the instruction count of a
# layer run predictable on a Cortex-M4 and repeatable on a host.
#
# The corresponding -m flags are added to the openzzs-mllee interface so the
# intrinsics in mllee/core_simd.hpp and mllee/train_simd.hpp are compiled in.
# Every intrinsic path has a portable C++ fallback in the same header, used
# when the option is off or the compiler does not define the feature macro.
# ---------------------------------------------------------------------------
option(OPENZZS_MLLEE_SIMD "Build mllee SIMD intrinsics kernels" ON)
option(OPENZZS_MLLEE_SSSE3 "Use SSSE3 pshufb kernels on x86-64 (2006+ parts)" ON)
option(OPENZZS_MLLEE_AVX2 "Use AVX2 mllee kernels on x86-64 (Haswell 2013+)" ON)
option(OPENZZS_MLLEE_AVX_VNNI
  "Use AVX-VNNI vpdpbusd kernels on x86-64 (Ice Lake 2019+ parts)" OFF)
option(OPENZZS_MLLEE_AVX512
  "Use AVX-512 VNNI/BW kernels on x86-64 (Skylake-X 2017+ / Ice Lake+)" OFF)

set(_zzs_mllee_simd_compile_options "")
if (OPENZZS_MLLEE_SIMD)
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    set(_zzs_mllee_simd_features "sse2")
    if (OPENZZS_MLLEE_SSSE3)
      list(APPEND _zzs_mllee_simd_compile_options -mssse3)
      list(APPEND _zzs_mllee_simd_features "ssse3")
    endif ()
    if (OPENZZS_MLLEE_AVX2)
      list(APPEND _zzs_mllee_simd_compile_options -mavx2)
      list(APPEND _zzs_mllee_simd_features "avx2")
    endif ()
    if (OPENZZS_MLLEE_AVX_VNNI)
      if (NOT OPENZZS_MLLEE_AVX2)
        message(FATAL_ERROR "OPENZZS_MLLEE_AVX_VNNI requires OPENZZS_MLLEE_AVX2")
      endif ()
      list(APPEND _zzs_mllee_simd_compile_options -mavxvnni)
      list(APPEND _zzs_mllee_simd_features "avxvnni")
    endif ()
    if (OPENZZS_MLLEE_AVX512)
      if (NOT OPENZZS_MLLEE_AVX2)
        message(FATAL_ERROR "OPENZZS_MLLEE_AVX512 requires OPENZZS_MLLEE_AVX2")
      endif ()
      list(APPEND _zzs_mllee_simd_compile_options -mavx512f -mavx512bw -mavx512vnni)
      list(APPEND _zzs_mllee_simd_features "avx512vnni")
    endif ()
  elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm|aarch64" OR CMAKE_CXX_COMPILER MATCHES "arm-none-eabi")
    list(APPEND _zzs_mllee_simd_compile_options -mcpu=cortex-m4 -mthumb)
    set(_zzs_mllee_simd_features "dsp")
  else ()
    set(_zzs_mllee_simd_features "none")
  endif ()
  list(JOIN _zzs_mllee_simd_features "+" _zzs_mllee_simd_features)
  message(STATUS "mllee: SIMD intrinsics enabled (${_zzs_mllee_simd_features})")
else ()
  set(_zzs_mllee_simd_features "none")
  message(STATUS "mllee: SIMD intrinsics disabled, using the portable C++ path")
endif ()

# ---------------------------------------------------------------------------
# secure_channel x86_64 AES-NI assembly.
#
# std/secure_channel_aes_gcm_x86_64.S provides the AES-256-GCM hot-path
# primitives (key schedule, single-block encrypt, GHASH via PCLMULQDQ) for
# x86-64 hosts.  It is selected at build time (never run-time CPUID); on every
# other target the portable constant-time C++ in std/secure_channel.hpp is
# used.  The file declares `.intel_syntax noprefix` itself, so no -masm flag is
# required.  The object is folded into openzzs-std so callers link it
# transparently.
# ---------------------------------------------------------------------------
if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
  enable_language(ASM)
  add_library(openzzs-secure-channel-asm STATIC std/secure_channel_aes_gcm_x86_64.S)
  target_compile_definitions(openzzs-secure-channel-asm PUBLIC ZZS_SECURE_CHANNEL_ASM_X86_64=1)
  set_target_properties(openzzs-secure-channel-asm PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(openzzs-secure-channel-asm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(openzzs-std PRIVATE openzzs-secure-channel-asm)
  message(STATUS "secure_channel: x86_64 AES-NI assembly enabled")
else ()
  message(STATUS "secure_channel: using the portable C++ AES-256-GCM path")
endif ()

add_library(openzzs-mllee INTERFACE)
target_compile_options(openzzs-mllee INTERFACE -fno-exceptions -fno-rtti)
target_include_directories(openzzs-mllee INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(openzzs-mllee INTERFACE cxx_std_23)
target_link_libraries(openzzs-mllee INTERFACE openzzs-std)
if (OPENZZS_MLLEE_SIMD AND _zzs_mllee_simd_compile_options)
  target_compile_options(openzzs-mllee INTERFACE ${_zzs_mllee_simd_compile_options})
endif ()

# Sensor-driver submodule: defines the bus assembly primitives and the
# openzzs-sensor-driver library.  See kernel/sensor-driver/bus/CMakeLists.txt.
add_subdirectory(kernel/sensor-driver/bus)

set(_zzs_sensor_driver_sources
  kernel/sensor-driver/bus/i2c/i2c_bitbang.cpp
  kernel/sensor-driver/bus/spi/spi_bitbang.cpp)

# Host platform HAL: only the translation unit matching this target compiles.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
  list(APPEND _zzs_sensor_driver_sources
    kernel/sensor-driver/platforms/linux/platform_hal.hpp)
endif ()

add_library(openzzs-sensor-driver STATIC ${_zzs_sensor_driver_sources})
target_include_directories(openzzs-sensor-driver PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/kernel/sensor-driver)
target_link_libraries(openzzs-sensor-driver PRIVATE
  openzzs-std
  openzzs-mllee)
if (TARGET openzzs-sensor-driver-bus-asm)
  target_link_libraries(openzzs-sensor-driver PRIVATE openzzs-sensor-driver-bus-asm)
endif ()
if (TARGET openzzs-sensor-driver-bus-cpp)
  target_link_libraries(openzzs-sensor-driver PRIVATE openzzs-sensor-driver-bus-cpp)
endif ()
set_target_properties(openzzs-sensor-driver PROPERTIES LINKER_LANGUAGE CXX)

if (NOT CMAKE_CROSSCOMPILING)
  add_executable(mllee_bench tools/mllee_bench.cpp)
  target_link_libraries(mllee_bench PRIVATE openzzs-mllee)
  target_compile_definitions(mllee_bench PRIVATE ZZS_MLLEE_HOST)
endif ()

# libFuzzer target for the native ONNX / MLEE blob reader. Host only: libFuzzer
# is not available for embedded cross targets.
if (NOT CMAKE_CROSSCOMPILING)
  add_executable(fuzz_onnx tools/fuzz_onnx_reader.cpp)
  target_link_libraries(fuzz_onnx PRIVATE openzzs-mllee)
  target_compile_definitions(fuzz_onnx PRIVATE ZZS_MLLEE_HOST)
  target_compile_options(fuzz_onnx PRIVATE -fsanitize=fuzzer,address,undefined)
  target_link_options(fuzz_onnx PRIVATE -fsanitize=fuzzer,address,undefined)
endif ()

find_program(ZZS_BASH bash)
if (ZZS_BASH AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_mllee_ram.sh)
  add_custom_target(check_mllee_ram
    COMMAND ${ZZS_BASH} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_mllee_ram.sh
            ${CMAKE_CURRENT_BINARY_DIR}/mllee-ram
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking the mllee RAM/stack/Flash budget"
    VERBATIM)
endif ()

if (ZZS_BASH AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_mllee_train_ram.sh)
  add_custom_target(check_mllee_train_ram
    COMMAND ${ZZS_BASH} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_mllee_train_ram.sh
            ${CMAKE_CURRENT_BINARY_DIR}/mllee-train-ram
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking the mllee training RAM/stack/Flash budget"
    VERBATIM)
endif ()

if (ZZS_BASH AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_mllee_m4_qemu.sh)
  # The script prints SKIP and exits 0 when the Arm toolchain or QEMU is absent,
  # so this target is safe to build everywhere.
  add_custom_target(check_mllee_m4_asm
    COMMAND ${ZZS_BASH} ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_mllee_m4_qemu.sh
            ${CMAKE_CURRENT_BINARY_DIR}/mllee-m4
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running the mllee Cortex-M4 kernels on QEMU mps2-an385"
    VERBATIM)
endif ()

add_executable(openzzs bootstrap-monitor/main.cpp)
target_link_libraries(openzzs PRIVATE openzzs-std openzzs-mllee openzzs-sensor-driver)

if (OPENZZS_RUNTIME_RELEASE_OPTIONS)
  target_compile_options(openzzs-std PRIVATE ${OPENZZS_RUNTIME_RELEASE_OPTIONS})
  target_compile_options(openzzs PRIVATE ${OPENZZS_RUNTIME_RELEASE_OPTIONS})
  target_compile_options(openzzs-mllee INTERFACE ${OPENZZS_RUNTIME_RELEASE_OPTIONS})
endif ()

# ---------------------------------------------------------------------------
# Install rules for find_package(openzzs) support.
# ---------------------------------------------------------------------------
include(GNUInstallDirs)

# Public headers: all of std/ and mlee/ (there is no detail/ subdirectory).
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/std/
        DESTINATION include/openzzs/std
        FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/mllee/
        DESTINATION include/openzzs/mllee
        FILES_MATCHING PATTERN "*.hpp")

install(TARGETS openzzs-std openzzs-mllee openzzs-sensor-driver
        ARCHIVE DESTINATION lib)

# CMake config package.
configure_file(openzzsConfig.cmake.in openzzsConfig.cmake @ONLY)
configure_file(openzzs-targets.cmake.in openzzs-targets.cmake @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/openzzsConfig.cmake
              ${CMAKE_CURRENT_BINARY_DIR}/openzzs-targets.cmake
        DESTINATION lib/cmake/openzzs)

# pkgconfig file.
configure_file(openzzs.pc.in openzzs.pc @ONLY)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/openzzs.pc
        DESTINATION lib/pkgconfig)

option(OPENZZS_BUILD_TESTS "Build the unit test suite" ON)

if (OPENZZS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif ()
