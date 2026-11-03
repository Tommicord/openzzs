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

#include "kernel/device-driver/gnss/platforms/detail/portable_uart.hpp"

#include <cstdint>

namespace zzs::kernel::device::driver::gnss::platform::windows {
namespace {

/** @brief Windows UART byte-stream bus adapter (CreateFile/DeviceIoControl COM). */
class WindowsGnssBus {
public:
  explicit WindowsGnssBus (const wchar_t *port) noexcept : _M_open (false) {
#if defined(_WIN32)
    _M_handle = INVALID_HANDLE_VALUE;
#else
    _M_handle = nullptr;
#endif
    if (port != nullptr) {
      uint32_t n = 0u;
      while (port[n] != L'\0' && n < sizeof (_M_port) / sizeof (_M_port[0]) - 1u) {
        _M_port[n] = port[n];
        ++n;
      }
      _M_port[n] = L'\0';
    }
  }

  ~WindowsGnssBus () noexcept {
#if defined(_WIN32)
    if (_M_handle != INVALID_HANDLE_VALUE)
      ::CloseHandle (_M_handle);
#endif
  }

  WindowsGnssBus (const WindowsGnssBus &) = delete;
  WindowsGnssBus &
  operator= (const WindowsGnssBus &) = delete;
  WindowsGnssBus (WindowsGnssBus &&) = delete;
  WindowsGnssBus &
  operator= (WindowsGnssBus &&) = delete;

  [[nodiscard]] zzs::kernel::Status
  open () noexcept {
#if defined(_WIN32)
    if (_M_open)
      return zzs::kernel::Status::OK;
    _M_handle = ::CreateFileW (_M_port, GENERIC_READ | GENERIC_WRITE, 0u, nullptr, OPEN_EXISTING, 0u, nullptr);
    if (_M_handle == INVALID_HANDLE_VALUE)
      return zzs::kernel::Status::INTERNAL;
    /* Real path: SetCommState (DCB 115200 8N1), SetCommTimeouts. */
    _M_open = true;
    return zzs::kernel::Status::OK;
#else
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  close () noexcept {
#if defined(_WIN32)
    if (!_M_open)
      return zzs::kernel::Status::OK;
    if (!::CloseHandle (_M_handle)) {
      _M_handle = INVALID_HANDLE_VALUE;
      _M_open = false;
      return zzs::kernel::Status::INTERNAL;
    }
    _M_handle = INVALID_HANDLE_VALUE;
    _M_open = false;
    return zzs::kernel::Status::OK;
#else
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  read_bytes (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept {
    if (buf == nullptr || filled == nullptr || max_bytes == 0u)
      return zzs::kernel::Status::OUT_OF_RANGE;
#if defined(_WIN32)
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    DWORD read = 0;
    if (!::ReadFile (_M_handle, buf, static_cast<DWORD> (max_bytes), &read, nullptr))
      return zzs::kernel::Status::STALE;
    *filled = static_cast<uint16_t> (read);
    return read > 0 ? zzs::kernel::Status::OK : zzs::kernel::Status::STALE;
#else
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  write_bytes (const uint8_t *buf, uint16_t len) noexcept {
    if (buf == nullptr && len > 0u)
      return zzs::kernel::Status::OUT_OF_RANGE;
#if defined(_WIN32)
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    DWORD written = 0;
    if (!::WriteFile (_M_handle, buf, static_cast<DWORD> (len), &written, nullptr))
      return zzs::kernel::Status::INTERNAL;
    return written == static_cast<DWORD> (len) ? zzs::kernel::Status::OK : zzs::kernel::Status::INTERNAL;
#else
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  void
  delay_ms (uint32_t ms) noexcept {
#if defined(_WIN32)
    ::Sleep (static_cast<DWORD> (ms));
#else
    (void) ms;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  flush_rx () noexcept {
#if defined(_WIN32)
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    if (!::PurgeComm (_M_handle, PURGE_RXABORT | PURGE_RXCLEAR))
      return zzs::kernel::Status::INTERNAL;
    return zzs::kernel::Status::OK;
#else
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  [[nodiscard]] bool
  is_open () const noexcept {
    return _M_open;
  }

#if !defined(_WIN32)
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
#if defined(_WIN32)
#include <windows.h>
#endif
  void *_M_handle;
  wchar_t _M_port[64];
  bool _M_open;
  detail::PortableRing _M_rx{256u};
  detail::PortableRing _M_tx{256u};
};

/** @brief Return the single shared bus instance (function-local static). */
WindowsGnssBus &
bus_ref () noexcept {
  static WindowsGnssBus _S_bus (L"COM1");
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
create_windows_gnss_hal () noexcept {
  (void) bus_ref ();
  return &_S_hal;
}

} // namespace zzs::kernel::device::driver::gnss::platform::windows
