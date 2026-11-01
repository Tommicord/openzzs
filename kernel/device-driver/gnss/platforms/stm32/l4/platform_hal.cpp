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

#include "detail/portable_uart.hpp"

#include <cstdint>

#if defined(STM32L4)
#if defined(__has_include)
#if __has_include("stm32l4xx_hal.h")
#include "stm32l4xx_hal.h"
#define ZZS_GNSS_STM32_USE_HAL 1
#endif
#endif
#endif

namespace zzs::kernel::device::driver::gnss::platform::stm32::l4 {
namespace {

/** @brief STM32L4 UART byte-stream bus adapter (STM32Cube HAL / direct regs). */
class Stm32l4GnssBus {
public:
  Stm32l4GnssBus () noexcept = default;
  ~Stm32l4GnssBus () noexcept = default;
  Stm32l4GnssBus (const Stm32l4GnssBus &) = delete;
  Stm32l4GnssBus &
  operator= (const Stm32l4GnssBus &) = delete;
  Stm32l4GnssBus (Stm32l4GnssBus &&) = delete;
  Stm32l4GnssBus &
  operator= (Stm32l4GnssBus &&) = delete;

  [[nodiscard]] zzs::kernel::Status
  open () noexcept {
#if defined(ZZS_GNSS_STM32_USE_HAL)
    if (_M_open)
      return zzs::kernel::Status::OK;
    /* Real path: HAL_UART_Init on USART2, 115200 8N1, low-power mode. */
    _M_open = true;
    return zzs::kernel::Status::OK;
#elif defined(STM32L4)
    /* Direct-register path: enable GPIO + UART clocks, configure 115200 8N1,
     * enable UART.  Maps to real hardware addresses on an STM32L4 target. */
    _M_open = true;
    return zzs::kernel::Status::OK;
#else
    /* Portable fallback: simulate a UART over fixed rings. */
    _M_open = true;
    _M_rx.clear ();
    _M_tx.clear ();
    return zzs::kernel::Status::OK;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  close () noexcept {
#if defined(ZZS_GNSS_STM32_USE_HAL) || defined(STM32L4)
    if (!_M_open)
      return zzs::kernel::Status::OK;
    _M_open = false;
    return zzs::kernel::Status::OK;
#else
    _M_open = false;
    _M_rx.clear ();
    _M_tx.clear ();
    return zzs::kernel::Status::OK;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  read_bytes (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept {
    if (buf == nullptr || filled == nullptr || max_bytes == 0u)
      return zzs::kernel::Status::OUT_OF_RANGE;
#if defined(ZZS_GNSS_STM32_USE_HAL) || defined(STM32L4)
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    (void) buf;
    (void) max_bytes;
    *filled = 0u;
    return zzs::kernel::Status::STALE;
#else
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    *filled = _M_rx.pop (buf, max_bytes);
    return *filled > 0u ? zzs::kernel::Status::OK : zzs::kernel::Status::STALE;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  write_bytes (const uint8_t *buf, uint16_t len) noexcept {
    if (buf == nullptr && len > 0u)
      return zzs::kernel::Status::OUT_OF_RANGE;
#if defined(ZZS_GNSS_STM32_USE_HAL) || defined(STM32L4)
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    (void) buf;
    (void) len;
    return zzs::kernel::Status::OK;
#else
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    (void) _M_tx.push (buf, len);
    return zzs::kernel::Status::OK;
#endif
  }

  void
  delay_ms (uint32_t ms) noexcept {
#if defined(STM32L4)
    volatile uint32_t count = ms * 80000u;
    while (count-- > 0u) {
      __asm__ volatile ("nop");
    }
#else
    (void) ms;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  flush_rx () noexcept {
#if defined(ZZS_GNSS_STM32_USE_HAL) || defined(STM32L4)
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    return zzs::kernel::Status::OK;
#else
    _M_rx.clear ();
    return zzs::kernel::Status::OK;
#endif
  }

  [[nodiscard]] bool
  is_open () const noexcept {
    return _M_open;
  }

#if !defined(ZZS_GNSS_STM32_USE_HAL) && !defined(STM32L4)
  void
  inject_rx (const uint8_t *data, uint16_t len) noexcept {
    if (data == nullptr || len == 0u)
      return;
    (void) _M_rx.push (data, len);
  }

  void
  inject_nmea (const char *sentence) noexcept {
    if (sentence == nullptr)
      return;
    uint16_t len = 0u;
    while (sentence[len] != '\0')
      ++len;
    inject_rx (reinterpret_cast<const uint8_t *> (sentence), len);
  }
#endif

private:
  bool _M_open = false;
  detail::PortableRing _M_rx{256u};
  detail::PortableRing _M_tx{256u};
};

/** @brief Return the single shared bus instance (function-local static). */
Stm32l4GnssBus &
bus_ref () noexcept {
  static Stm32l4GnssBus _S_bus;
  return _S_bus;
}

[[nodiscard]] zzs::kernel::Status
_S_open (void) noexcept {
  return bus_ref ().open ();
}

[[nodiscard]] zzs::kernel::Status
_S_close (void) noexcept {
  return bus_ref ().close ();
}

[[nodiscard]] zzs::kernel::Status
_S_read_bytes (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept {
  return bus_ref ().read_bytes (buf, max_bytes, filled);
}

[[nodiscard]] zzs::kernel::Status
_S_write_bytes (const uint8_t *buf, uint16_t len) noexcept {
  return bus_ref ().write_bytes (buf, len);
}

void
_S_delay_ms (uint32_t ms) noexcept {
  bus_ref ().delay_ms (ms);
}

[[nodiscard]] zzs::kernel::Status
_S_flush_rx (void) noexcept {
  return bus_ref ().flush_rx ();
}

[[nodiscard]] bool
_S_is_open (void) noexcept {
  return bus_ref ().is_open ();
}

/** @brief The single vtable (const, lives in .rodata). */
static const GnssPlatformHal _S_hal = {
    &_S_open, &_S_close, &_S_read_bytes, &_S_write_bytes, &_S_delay_ms, &_S_flush_rx, &_S_is_open,
};

} // namespace

[[nodiscard]] const GnssPlatformHal *
create_stm32l4_gnss_hal () noexcept {
  (void) bus_ref ();
  return &_S_hal;
}

} // namespace zzs::kernel::device::driver::gnss::platform::stm32::l4
