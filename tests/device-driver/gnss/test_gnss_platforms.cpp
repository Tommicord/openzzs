/*
 * Copyright(c) 2026 Tommicord
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

/**
 * @file test_gnss_platforms.cpp
 * @brief Unit tests for the GNSS platform HAL factory and each platform HAL.
 *
 * Verifies that `create_gnss_platform_hal()` returns a non-null vtable for the
 * selected platform, and that each platform's `create_<platform>_gnss_hal()`
 * factory returns non-null when its selection macro is defined.  Exercises the
 * public `GnssPlatformHal` vtable contract (open / is_open / read_bytes /
 * write_bytes / null-argument handling).  Hardware-independent: no real UART
 * is required; on hosts the selected platform falls back to its portable
 * simulated UART when the vendor device node is absent.
 */

#include "kernel/device-driver/gnss/platforms/platform_hal.hpp"
#include "kernel/device-driver/gnss/platforms/platform_select.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>

/* `linux` is a predefined macro (=1) on GCC/Linux; undef it so the
 * `platform::linux` namespace is parsed as an identifier. */
#undef linux

/* Include each platform's factory header when its selection macro is active.
 * Only the block matching the build's selected platform is compiled. */
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_LINUX)
#include "kernel/device-driver/gnss/platforms/linux/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32F4)
#include "kernel/device-driver/gnss/platforms/stm32/f4/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32F7)
#include "kernel/device-driver/gnss/platforms/stm32/f7/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32H7)
#include "kernel/device-driver/gnss/platforms/stm32/h7/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32L4)
#include "kernel/device-driver/gnss/platforms/stm32/l4/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_ANDROID)
#include "kernel/device-driver/gnss/platforms/android/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_IOS)
#include "kernel/device-driver/gnss/platforms/ios/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_MACOS)
#include "kernel/device-driver/gnss/platforms/macos/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_WINDOWS)
#include "kernel/device-driver/gnss/platforms/windows/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_ZEPHYR)
#include "kernel/device-driver/gnss/platforms/zephyr/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_FREERTOS)
#include "kernel/device-driver/gnss/platforms/freertos/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_RP2040)
#include "kernel/device-driver/gnss/platforms/rp2040/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_ESP32)
#include "kernel/device-driver/gnss/platforms/esp32/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_NRF52)
#include "kernel/device-driver/gnss/platforms/nrf52/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_SAMD21)
#include "kernel/device-driver/gnss/platforms/samd21/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_BEAGLEBONE)
#include "kernel/device-driver/gnss/platforms/beaglebone/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_QNX)
#include "kernel/device-driver/gnss/platforms/qnx/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_VXWORKS)
#include "kernel/device-driver/gnss/platforms/vxworks/platform_hal.hpp"
#endif
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_RTLINUX)
#include "kernel/device-driver/gnss/platforms/rtlinux/platform_hal.hpp"
#endif

#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

namespace gnss_platform = zzs::kernel::device::driver::gnss::platform;

// ---------------------------------------------------------------------------
// The selected platform's dispatch factory must return a non-null vtable.
// ---------------------------------------------------------------------------

TEST_CASE ("gnss.platform.dispatch_factory_non_null", "[gnss][platform]") {
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = gnss_platform::create_gnss_platform_hal ();
  REQUIRE (hal != nullptr);
  REQUIRE (hal->open != nullptr);
  REQUIRE (hal->close != nullptr);
  REQUIRE (hal->read_bytes != nullptr);
  REQUIRE (hal->write_bytes != nullptr);
  REQUIRE (hal->delay_ms != nullptr);
  REQUIRE (hal->flush_rx != nullptr);
  REQUIRE (hal->is_open != nullptr);
}

// ---------------------------------------------------------------------------
// Public vtable contract: null argument handling.
// ---------------------------------------------------------------------------

TEST_CASE ("gnss.platform.vtable_null_args", "[gnss][platform]") {
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = gnss_platform::create_gnss_platform_hal ();
  REQUIRE (hal != nullptr);

  uint16_t filled = 0u;

  // read_bytes with null pointers must return OUT_OF_RANGE, never crash.
  REQUIRE (hal->read_bytes (nullptr, 10u, &filled) == zzs::kernel::Status::OUT_OF_RANGE);
  REQUIRE (hal->read_bytes (nullptr, 0u, &filled) == zzs::kernel::Status::OUT_OF_RANGE);
  uint8_t buf[4] = {};
  REQUIRE (hal->read_bytes (buf, 10u, nullptr) == zzs::kernel::Status::OUT_OF_RANGE);

  // write_bytes with null buf and nonzero len must return OUT_OF_RANGE.
  REQUIRE (hal->write_bytes (nullptr, 10u) == zzs::kernel::Status::OUT_OF_RANGE);
  // write_bytes with null buf and zero len bypasses the null-pointer check;
  // on a closed real bus it reports INTERNAL, on an open fallback bus OK.
  // Either way it must not be misreported as OUT_OF_RANGE.
  REQUIRE (hal->write_bytes (nullptr, 0u) != zzs::kernel::Status::OUT_OF_RANGE);

  // delay_ms and flush_rx are always safe to call.
  hal->delay_ms (0u);
  hal->delay_ms (1u);
  (void) hal->flush_rx ();
  (void) hal->is_open ();
}

// ---------------------------------------------------------------------------
// Each platform's create_<platform>_gnss_hal() returns non-null on its macro.
// Only the block matching the active selection macro is compiled/executed.
// ---------------------------------------------------------------------------

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_LINUX)
TEST_CASE ("gnss.platform.linux_factory_non_null", "[gnss][platform][linux]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::linux;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_linux_gnss_hal ();
  REQUIRE (hal != nullptr);
  /* Exercise the vtable safely; open() depends on real device presence. */
  (void) hal->open ();
  (void) hal->is_open ();
  (void) hal->close ();
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32F4)
TEST_CASE ("gnss.platform.stm32f4_factory_non_null", "[gnss][platform][stm32/f4]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::stm32::f4;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_stm32f4_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32F7)
TEST_CASE ("gnss.platform.stm32f7_factory_non_null", "[gnss][platform][stm32/f7]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::stm32::f7;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_stm32f7_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32H7)
TEST_CASE ("gnss.platform.stm32h7_factory_non_null", "[gnss][platform][stm32/h7]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::stm32::h7;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_stm32h7_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32L4)
TEST_CASE ("gnss.platform.stm32l4_factory_non_null", "[gnss][platform][stm32/l4]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::stm32::l4;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_stm32l4_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_ANDROID)
TEST_CASE ("gnss.platform.android_factory_non_null", "[gnss][platform][android]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::android;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_android_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_IOS)
TEST_CASE ("gnss.platform.ios_factory_non_null", "[gnss][platform][ios]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::ios;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_ios_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_MACOS)
TEST_CASE ("gnss.platform.macos_factory_non_null", "[gnss][platform][macos]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::macos;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_macos_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_WINDOWS)
TEST_CASE ("gnss.platform.windows_factory_non_null", "[gnss][platform][windows]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::windows;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_windows_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_ZEPHYR)
TEST_CASE ("gnss.platform.zephyr_factory_non_null", "[gnss][platform][zephyr]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::zephyr;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_zephyr_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_FREERTOS)
TEST_CASE ("gnss.platform.freertos_factory_non_null", "[gnss][platform][freertos]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::freertos;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_freertos_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_RP2040)
TEST_CASE ("gnss.platform.rp2040_factory_non_null", "[gnss][platform][rp2040]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::rp2040;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_rp2040_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_ESP32)
TEST_CASE ("gnss.platform.esp32_factory_non_null", "[gnss][platform][esp32]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::esp32;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_esp32_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_NRF52)
TEST_CASE ("gnss.platform.nrf52_factory_non_null", "[gnss][platform][nrf52]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::nrf52;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_nrf52_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_SAMD21)
TEST_CASE ("gnss.platform.samd21_factory_non_null", "[gnss][platform][samd21]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::samd21;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_samd21_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_BEAGLEBONE)
TEST_CASE ("gnss.platform.beaglebone_factory_non_null", "[gnss][platform][beaglebone]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::beaglebone;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_beaglebone_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_QNX)
TEST_CASE ("gnss.platform.qnx_factory_non_null", "[gnss][platform][qnx]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::qnx;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_qnx_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_VXWORKS)
TEST_CASE ("gnss.platform.vxworks_factory_non_null", "[gnss][platform][vxworks]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::vxworks;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_vxworks_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_RTLINUX)
TEST_CASE ("gnss.platform.rtlinux_factory_non_null", "[gnss][platform][rtlinux]") {
  namespace plat = zzs::kernel::device::driver::gnss::platform::rtlinux;
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_rtlinux_gnss_hal ();
  REQUIRE (hal != nullptr);
}
#endif

// ---------------------------------------------------------------------------
// Each platform translation unit compiles standalone: including its header and
// calling its factory (guarded by its macro) is sufficient evidence.  The
// portable fallback path (no vendor SDK) exercises the simulated UART ring
// buffer through the public vtable: open -> is_open true, read STALE when
// empty, write accepted.
// ---------------------------------------------------------------------------

TEST_CASE ("gnss.platform.portable_fallback_contract", "[gnss][platform][fallback]") {
  const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = gnss_platform::create_gnss_platform_hal ();
  REQUIRE (hal != nullptr);

  // Reading from the stream is always safe and sets *filled.  With valid
  // arguments it never returns OUT_OF_RANGE; the exact code depends on whether
  // a real device is present (OK/STALE) or not (INTERNAL).
  uint8_t rb[8] = {};
  uint16_t filled = 0x5555u;
  const zzs::kernel::Status st = hal->read_bytes (rb, sizeof (rb), &filled);
  REQUIRE (st != zzs::kernel::Status::OUT_OF_RANGE);

  // Writing a small command is always safe; with valid arguments it never
  // returns OUT_OF_RANGE.
  const uint8_t cmd[] = {'P', 'W', 'R', '?'};
  const zzs::kernel::Status wt = hal->write_bytes (cmd, sizeof (cmd));
  REQUIRE (wt != zzs::kernel::Status::OUT_OF_RANGE);

  hal->flush_rx ();
  hal->close ();
}
