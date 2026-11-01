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

namespace zzs::kernel::device::driver::gnss::platform::freertos {
namespace {

/** @brief FreeRTOS UART byte-stream bus adapter (vendor UART driver hooks). */
class FreertosGnssBus {
public:
  explicit FreertosGnssBus (void *uart_handle) noexcept : _M_handle (uart_handle), _M_open (false) {
  }
  ~FreertosGnssBus () noexcept = default;
  FreertosGnssBus (const FreertosGnssBus &) = delete;
  FreertosGnssBus &
  operator= (const FreertosGnssBus &) = delete;
  FreertosGnssBus (FreertosGnssBus &&) = delete;
  FreertosGnssBus &
  operator= (FreertosGnssBus &&) = delete;

  [[nodiscard]] zzs::kernel::Status
  open () noexcept {
#if defined(FREERTOS)
    if (_M_open)
      return zzs::kernel::Status::OK;
    extern "C" zzs::kernel::Status freertos_gnss_uart_open (void *) noexcept;
    zzs::kernel::Status st = freertos_gnss_uart_open (_M_handle);
    if (!zzs::kernel::status_ok (st))
      return st;
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
#if defined(FREERTOS)
    if (!_M_open)
      return zzs::kernel::Status::OK;
    extern "C" zzs::kernel::Status freertos_gnss_uart_close (void *) noexcept;
    zzs::kernel::Status st = freertos_gnss_uart_close (_M_handle);
    _M_open = false;
    return st;
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
#if defined(FREERTOS)
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    extern "C" zzs::kernel::Status freertos_gnss_uart_read (void *, uint8_t *, uint16_t, uint16_t *) noexcept;
    return freertos_gnss_uart_read (_M_handle, buf, max_bytes, filled);
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
#if defined(FREERTOS)
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    extern "C" zzs::kernel::Status freertos_gnss_uart_write (void *, const uint8_t *, uint16_t) noexcept;
    return freertos_gnss_uart_write (_M_handle, buf, len);
#else
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    (void) _M_tx.push (buf, len);
    return zzs::kernel::Status::OK;
#endif
  }

  void
  delay_ms (uint32_t ms) noexcept {
#if defined(FREERTOS)
    vTaskDelay (static_cast<TickType_t> (ms) / portTICK_PERIOD_MS);
#else
    (void) ms;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  flush_rx () noexcept {
#if defined(FREERTOS)
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    extern "C" zzs::kernel::Status freertos_gnss_uart_flush_rx (void *) noexcept;
    return freertos_gnss_uart_flush_rx (_M_handle);
#else
    _M_rx.clear ();
    return zzs::kernel::Status::OK;
#endif
  }

  [[nodiscard]] bool
  is_open () const noexcept {
    return _M_open;
  }

#if !defined(FREERTOS)
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
#if defined(FREERTOS)
#include <FreeRTOS.h>
#include <task.h>
#endif
  void *_M_handle;
  bool _M_open = false;
  detail::PortableRing _M_rx{256u};
  detail::PortableRing _M_tx{256u};
};

/** @brief Return the single shared bus instance (function-local static). */
FreertosGnssBus &
bus_ref () noexcept {
  static FreertosGnssBus _S_bus (nullptr);
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
create_freertos_gnss_hal () noexcept {
  (void) bus_ref ();
  return &_S_hal;
}

} // namespace zzs::kernel::device::driver::gnss::platform::freertos
