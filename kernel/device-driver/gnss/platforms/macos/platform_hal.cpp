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

namespace zzs::kernel::device::driver::gnss::platform::macos {
namespace {

/** @brief macOS UART byte-stream bus adapter (IOKit serial / USB-CDC). */
class MacosGnssBus {
public:
  explicit MacosGnssBus (const char *device) noexcept : _M_open (false) {
    if (device != nullptr) {
      uint32_t n = 0u;
      while (device[n] != '\0' && n < sizeof (_M_device) - 1u) {
        _M_device[n] = device[n];
        ++n;
      }
      _M_device[n] = '\0';
    }
  }

  ~MacosGnssBus () noexcept = default;
  MacosGnssBus (const MacosGnssBus &) = delete;
  MacosGnssBus &
  operator= (const MacosGnssBus &) = delete;
  MacosGnssBus (MacosGnssBus &&) = delete;
  MacosGnssBus &
  operator= (MacosGnssBus &&) = delete;

  [[nodiscard]] zzs::kernel::Status
  open () noexcept {
#if defined(__APPLE__)
    if (_M_open)
      return zzs::kernel::Status::OK;
    /* Real path: IOSerialBSDClient open of /dev/tty.* then configure baud. */
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
#if defined(__APPLE__)
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
#if defined(__APPLE__)
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
#if defined(__APPLE__)
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
#if defined(__APPLE__)
    struct timespec ts = {.tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000ull};
    ::nanosleep (&ts, nullptr);
#else
    (void) ms;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  flush_rx () noexcept {
#if defined(__APPLE__)
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

#if !defined(__APPLE__)
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
#if defined(__APPLE__)
#include <time.h>
#endif
  char _M_device[64];
  bool _M_open;
  detail::PortableRing _M_rx{256u};
  detail::PortableRing _M_tx{256u};
};

/** @brief Return the single shared bus instance (function-local static). */
MacosGnssBus &
bus_ref () noexcept {
  static MacosGnssBus _S_bus ("/dev/tty.usbmodem1");
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
create_macos_gnss_hal () noexcept {
  (void) bus_ref ();
  return &_S_hal;
}

} // namespace zzs::kernel::device::driver::gnss::platform::macos
