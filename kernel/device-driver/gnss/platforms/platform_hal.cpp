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

#include "platform_hal.hpp"

#include "platform_select.hpp"

#include <cstdint>

/* `linux` is a predefined macro (=1) on GCC/Linux; undef it so the
 * `platform::linux` namespace is parsed as an identifier. */
#undef linux

// Forward declarations of the platform-specific create functions.  Each is
// guarded by its selection macro so only the selected platform's factory is
// referenced in any given build.
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_LINUX)
namespace zzs::kernel::device::driver::gnss::platform::linux {
[[nodiscard]] const GnssPlatformHal *
create_linux_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::linux
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32F4)
namespace zzs::kernel::device::driver::gnss::platform::stm32::f4 {
[[nodiscard]] const GnssPlatformHal *
create_stm32f4_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::stm32::f4
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32F7)
namespace zzs::kernel::device::driver::gnss::platform::stm32::f7 {
[[nodiscard]] const GnssPlatformHal *
create_stm32f7_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::stm32::f7
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32H7)
namespace zzs::kernel::device::driver::gnss::platform::stm32::h7 {
[[nodiscard]] const GnssPlatformHal *
create_stm32h7_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::stm32::h7
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32L4)
namespace zzs::kernel::device::driver::gnss::platform::stm32::l4 {
[[nodiscard]] const GnssPlatformHal *
create_stm32l4_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::stm32::l4
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_ANDROID)
namespace zzs::kernel::device::driver::gnss::platform::android {
[[nodiscard]] const GnssPlatformHal *
create_android_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::android
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_IOS)
namespace zzs::kernel::device::driver::gnss::platform::ios {
[[nodiscard]] const GnssPlatformHal *
create_ios_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::ios
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_MACOS)
namespace zzs::kernel::device::driver::gnss::platform::macos {
[[nodiscard]] const GnssPlatformHal *
create_macos_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::macos
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_WINDOWS)
namespace zzs::kernel::device::driver::gnss::platform::windows {
[[nodiscard]] const GnssPlatformHal *
create_windows_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::windows
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_ZEPHYR)
namespace zzs::kernel::device::driver::gnss::platform::zephyr {
[[nodiscard]] const GnssPlatformHal *
create_zephyr_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::zephyr
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_FREERTOS)
namespace zzs::kernel::device::driver::gnss::platform::freertos {
[[nodiscard]] const GnssPlatformHal *
create_freertos_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::freertos
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_RP2040)
namespace zzs::kernel::device::driver::gnss::platform::rp2040 {
[[nodiscard]] const GnssPlatformHal *
create_rp2040_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::rp2040
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_ESP32)
namespace zzs::kernel::device::driver::gnss::platform::esp32 {
[[nodiscard]] const GnssPlatformHal *
create_esp32_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::esp32
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_NRF52)
namespace zzs::kernel::device::driver::gnss::platform::nrf52 {
[[nodiscard]] const GnssPlatformHal *
create_nrf52_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::nrf52
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_SAMD21)
namespace zzs::kernel::device::driver::gnss::platform::samd21 {
[[nodiscard]] const GnssPlatformHal *
create_samd21_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::samd21
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_BEAGLEBONE)
namespace zzs::kernel::device::driver::gnss::platform::beaglebone {
[[nodiscard]] const GnssPlatformHal *
create_beaglebone_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::beaglebone
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_QNX)
namespace zzs::kernel::device::driver::gnss::platform::qnx {
[[nodiscard]] const GnssPlatformHal *
create_qnx_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::qnx
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_VXWORKS)
namespace zzs::kernel::device::driver::gnss::platform::vxworks {
[[nodiscard]] const GnssPlatformHal *
create_vxworks_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::vxworks
#endif

#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_RTLINUX)
namespace zzs::kernel::device::driver::gnss::platform::rtlinux {
[[nodiscard]] const GnssPlatformHal *
create_rtlinux_gnss_hal () noexcept;
} // namespace zzs::kernel::device::driver::gnss::platform::rtlinux
#endif

namespace zzs::kernel::device::driver::gnss::platform {

[[nodiscard]] const GnssPlatformHal *
create_gnss_platform_hal () noexcept {
#if defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_LINUX)
    return zzs::kernel::device::driver::gnss::platform::linux::create_linux_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32F4)
    return zzs::kernel::device::driver::gnss::platform::stm32::f4::create_stm32f4_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32F7)
    return zzs::kernel::device::driver::gnss::platform::stm32::f7::create_stm32f7_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32H7)
    return zzs::kernel::device::driver::gnss::platform::stm32::h7::create_stm32h7_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_STM32L4)
    return zzs::kernel::device::driver::gnss::platform::stm32::l4::create_stm32l4_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_ANDROID)
    return zzs::kernel::device::driver::gnss::platform::android::create_android_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_IOS)
    return zzs::kernel::device::driver::gnss::platform::ios::create_ios_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_MACOS)
    return zzs::kernel::device::driver::gnss::platform::macos::create_macos_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_WINDOWS)
    return zzs::kernel::device::driver::gnss::platform::windows::create_windows_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_ZEPHYR)
    return zzs::kernel::device::driver::gnss::platform::zephyr::create_zephyr_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_FREERTOS)
    return zzs::kernel::device::driver::gnss::platform::freertos::create_freertos_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_RP2040)
    return zzs::kernel::device::driver::gnss::platform::rp2040::create_rp2040_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_ESP32)
    return zzs::kernel::device::driver::gnss::platform::esp32::create_esp32_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_NRF52)
    return zzs::kernel::device::driver::gnss::platform::nrf52::create_nrf52_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_SAMD21)
    return zzs::kernel::device::driver::gnss::platform::samd21::create_samd21_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_BEAGLEBONE)
    return zzs::kernel::device::driver::gnss::platform::beaglebone::create_beaglebone_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_QNX)
    return zzs::kernel::device::driver::gnss::platform::qnx::create_qnx_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_VXWORKS)
    return zzs::kernel::device::driver::gnss::platform::vxworks::create_vxworks_gnss_hal ();
#elif defined(ZZS_DEVICE_DRIVER_GNSS_PLATFORM_RTLINUX)
    return zzs::kernel::device::driver::gnss::platform::rtlinux::create_rtlinux_gnss_hal ();
#else
    return nullptr;
#endif
}

} // namespace zzs::kernel::device::driver::gnss::platform
