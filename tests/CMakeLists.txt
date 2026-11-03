# Unit tests for the openzzs modules.
#
# Catch2 is used as the test framework. It is located with find_package() first
# and, when unavailable, downloaded via FetchContent so a fresh checkout can run
# the suite without any pre-installed dependency.

enable_language(CXX)

find_package(Catch2 3 QUIET)

if (NOT Catch2_FOUND)
  include(FetchContent)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.9.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(Catch2)
endif ()

# The sensor layer is not part of this checkout yet; keep the suite configurable
# without it instead of failing the whole build.
if (TARGET sensr)
  add_executable(test_sensr test_sensr.cpp)

  target_link_libraries(test_sensr PRIVATE sensr Catch2::Catch2)

  # The global Release flags disable exceptions and RTTI for size reasons, but
  # the test framework relies on both. Re-enable them for this target only.
  if (MSVC)
    target_compile_options(test_sensr PRIVATE /EHs /GR)
  else ()
    target_compile_options(test_sensr PRIVATE -fexceptions -frtti -funwind-tables)
  endif ()

  include(Catch)
  catch_discover_tests(test_sensr)
endif ()

# Training modules land one header at a time, and several of them can be in
# flight at once; a glob keeps each test file with its module instead of making
# every contributor edit this list.  CONFIGURE_DEPENDS re-runs the check on build
# so a newly added file is picked up without an explicit configure.
file (GLOB _zzs_mllee_training_tests CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/test_mllee_training_*.cpp)

add_executable(test_mllee
  test_mllee_quant.cpp
  test_mllee_codec.cpp
  test_mllee_ops.cpp
  test_mllee_onnx.cpp
  test_mllee_simd_packed.cpp
  test_mllee_native_inference.cpp
  ${_zzs_mllee_training_tests})

target_link_libraries(test_mllee PRIVATE openzzs-mllee Catch2::Catch2WithMain)
target_compile_definitions(test_mllee PRIVATE ZZS_MLLEE_HOST)

if (MSVC)
  target_compile_options(test_mllee PRIVATE /EHs /GR)
else ()
  target_compile_options(test_mllee PRIVATE -fexceptions -frtti -funwind-tables)
endif ()

add_test(NAME test_mllee COMMAND test_mllee)

# ---------------------------------------------------------------------------
# Builtins portable-fallback tests (host only; exercises the MSVC/portable
# overflow fallbacks in zzs_builtins.hpp even on GCC/Clang via
# ZZS_BUILTINS_FORCE_PORTABLE).
# ---------------------------------------------------------------------------
if (NOT CMAKE_CROSSCOMPILING)
  add_executable(test_builtins_portable
    device-driver/builtins/test_builtins_portable.cpp)

  target_include_directories(test_builtins_portable PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel)

  target_link_libraries(test_builtins_portable PRIVATE
    openzzs-std Catch2::Catch2WithMain)

  target_compile_definitions(test_builtins_portable PRIVATE
    ZZS_BUILTINS_FORCE_PORTABLE=1)

  if (MSVC)
    target_compile_options(test_builtins_portable PRIVATE /EHs /GR)
  else ()
    target_compile_options(test_builtins_portable PRIVATE -fexceptions -frtti -funwind-tables)
  endif ()

  include(Catch)
  catch_discover_tests(test_builtins_portable)

  add_test(NAME test_builtins_portable COMMAND test_builtins_portable)
endif ()

# ---------------------------------------------------------------------------
# GNSS device driver tests (host only; mock stream bus, no hardware).
#
# kernel/device-driver is not wired into the root CMake build, so the GNSS
# sources are compiled directly into the test executable.
# ---------------------------------------------------------------------------
if (NOT CMAKE_CROSSCOMPILING)
  # Glob the GNSS test files so new end-to-end suites are picked up without
  # editing this list (CONFIGURE_DEPENDS re-runs the check on build).
  file (GLOB _zzs_test_gnss_tests CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/device-driver/gnss/test_*.cpp)

  set(_zzs_test_gnss_sources
    ${_zzs_test_gnss_tests}
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/gnss/gnss_driver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/gnss/nmea/nmea_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/gnss/nmea/nmea_checksum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/gnss/ubx/ubx_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/gnss/stream/gnss_stream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/gnss/gnss_ublox.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/gnss/platforms/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/gnss/platforms/linux/platform_hal.cpp)

  add_executable(test_gnss_driver
    ${_zzs_test_gnss_sources})

  target_include_directories(test_gnss_driver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/device-driver/gnss
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/..)

  target_link_libraries(test_gnss_driver PRIVATE
    openzzs-std Catch2::Catch2WithMain)

  target_compile_definitions(test_gnss_driver PRIVATE
    ZZS_DEVICE_DRIVER_GNSS_PLATFORM_LINUX=1)

  # The global Release flags disable exceptions and RTTI for size reasons, but
  # the test framework relies on both. Re-enable them for this target only.
  if (MSVC)
    target_compile_options(test_gnss_driver PRIVATE /EHs /GR)
  else ()
    target_compile_options(test_gnss_driver PRIVATE -fexceptions -frtti -funwind-tables)
  endif ()

  include(Catch)
  catch_discover_tests(test_gnss_driver)

  add_test(NAME test_gnss_driver COMMAND test_gnss_driver)
endif ()

# ---------------------------------------------------------------------------
# WiFi device driver tests (host only; mock stream bus, no hardware).
#
# The root CMakeLists.txt does not add_subdirectory(kernel/device-driver/wifi),
# so the WiFi library sources are compiled directly into the test executable.
# ---------------------------------------------------------------------------
if (NOT CMAKE_CROSSCOMPILING)
  # WiFi device driver library sources (compiled directly since the root
  # CMakeLists.txt does not add_subdirectory(kernel/device-driver/wifi)).
  set(_zzs_test_wifi_driver_lib_sources
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/wifi_driver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/wifi_frame.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/stream/wifi_stream.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/generic_wifi_driver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/linux/platform_hal.cpp)

  # x86_64 bus assembly (selected at build time, matching
  # kernel/device-driver/wifi/CMakeLists.txt).
  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    list(APPEND _zzs_test_wifi_driver_lib_sources
      ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/bus/asm/wifi_bus_asm_x86_64.S)
  endif ()

  add_library(openzzs-device-driver-wifi STATIC ${_zzs_test_wifi_driver_lib_sources})
  target_include_directories(openzzs-device-driver-wifi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(openzzs-device-driver-wifi PRIVATE openzzs-std)
  set_target_properties(openzzs-device-driver-wifi PROPERTIES LINKER_LANGUAGE CXX)

  if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    target_compile_definitions(openzzs-device-driver-wifi PRIVATE ZZS_DEVICE_DRIVER_WIFI_ASM_X86_64=1)
  endif ()
  target_include_directories(openzzs-device-driver-wifi PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(openzzs-device-driver-wifi PRIVATE openzzs-std)
  set_target_properties(openzzs-device-driver-wifi PROPERTIES LINKER_LANGUAGE CXX)

  # Glob the WiFi driver tests; the platform-HAL suite is built as a separate
  # executable below (it compiles every platform translation unit).
  file (GLOB _zzs_test_wifi_driver_tests CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/device-driver/wifi/test_wifi_*.cpp)
  list (FILTER _zzs_test_wifi_driver_tests EXCLUDE REGEX "test_wifi_platform_hal\\.cpp$")

  add_executable(test_wifi_driver
    ${_zzs_test_wifi_driver_tests})

  target_include_directories(test_wifi_driver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/device-driver/wifi
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/..)

  target_link_libraries(test_wifi_driver PRIVATE
    openzzs-device-driver-wifi Catch2::Catch2WithMain)

  target_compile_definitions(test_wifi_driver PRIVATE ZZS_KERNEL_PLATFORM_LINUX=1)

  # The global Release flags disable exceptions and RTTI for size reasons, but
  # the test framework relies on both. Re-enable them for this target only.
  if (MSVC)
    target_compile_options(test_wifi_driver PRIVATE /EHs /GR)
  else ()
    target_compile_options(test_wifi_driver PRIVATE -fexceptions -frtti -funwind-tables)
  endif ()

  include(Catch)
  catch_discover_tests(test_wifi_driver)

  add_test(NAME test_wifi_driver COMMAND test_wifi_driver)
endif ()

# ---------------------------------------------------------------------------
# WiFi platform HAL tests (host only; every platform's factory is exercised
# through its portable fallback path).  Compiles every platform translation
# unit directly so all create_*_wifi_hal() factories are linked and testable.
# ---------------------------------------------------------------------------
if (NOT CMAKE_CROSSCOMPILING)
  set(_zzs_test_wifi_platform_sources
    device-driver/wifi/test_wifi_platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/linux/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/windows/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/stm32/f4/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/stm32/f7/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/stm32/h7/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/stm32/l4/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/esp32/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/freertos/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/android/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/ios/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/zephyr/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/rp2040/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/nrf52/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/samd21/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/beaglebone/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/macos/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/qnx/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/vxworks/platform_hal.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel/device-driver/wifi/platforms/rtlinux/platform_hal.cpp)

  add_executable(test_wifi_platform_hal
    ${_zzs_test_wifi_platform_sources})

  target_include_directories(test_wifi_platform_hal PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/device-driver/wifi
    ${CMAKE_CURRENT_SOURCE_DIR}/../kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/..)

  target_link_libraries(test_wifi_platform_hal PRIVATE
    openzzs-std Catch2::Catch2WithMain)

  target_compile_definitions(test_wifi_platform_hal PRIVATE ZZS_KERNEL_PLATFORM_LINUX=1)

  # The global Release flags disable exceptions and RTTI for size reasons, but
  # the test framework relies on both. Re-enable them for this target only.
  if (MSVC)
    target_compile_options(test_wifi_platform_hal PRIVATE /EHs /GR)
  else ()
    target_compile_options(test_wifi_platform_hal PRIVATE -fexceptions -frtti -funwind-tables)
  endif ()

  include(Catch)
  catch_discover_tests(test_wifi_platform_hal)

  add_test(NAME test_wifi_platform_hal COMMAND test_wifi_platform_hal)
endif ()
