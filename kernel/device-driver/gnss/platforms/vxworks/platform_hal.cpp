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

namespace zzs::kernel::device::driver::gnss::platform::vxworks {
namespace {

/** @brief VxWorks UART byte-stream bus adapter (serial driver / iosLib). */
class VxworksGnssBus {
public:
  explicit VxworksGnssBus (const char *dev_name) noexcept : _M_open (false) {
#if defined(__VXWORKS__)
    _M_fd = ERROR;
#else
    _M_fd = -1;
#endif
    if (dev_name != nullptr) {
      uint32_t n = 0u;
      while (dev_name[n] != '\0' && n < sizeof (_M_dev_name) - 1u) {
        _M_dev_name[n] = dev_name[n];
        ++n;
      }
      _M_dev_name[n] = '\0';
    }
  }

  ~VxworksGnssBus () noexcept {
#if defined(__VXWORKS__)
    if (_M_fd != ERROR)
#else
    if (_M_fd >= 0)
#endif
      ::close (_M_fd);
  }

  VxworksGnssBus (const VxworksGnssBus &) = delete;
  VxworksGnssBus &
  operator= (const VxworksGnssBus &) = delete;
  VxworksGnssBus (VxworksGnssBus &&) = delete;
  VxworksGnssBus &
  operator= (VxworksGnssBus &&) = delete;

  [[nodiscard]] zzs::kernel::Status
  open () noexcept {
#if defined(__VXWORKS__)
    if (_M_open)
      return zzs::kernel::Status::OK;
    _M_fd = open (_M_dev_name, O_RDWR, 0);
    if (_M_fd == ERROR)
      return zzs::kernel::Status::INTERNAL;
    /* Real path: ioctl _M_fd with FIOBAUD to set 115200 8N1. */
    _M_open = true;
    return zzs::kernel::Status::OK;
#else
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  close () noexcept {
#if defined(__VXWORKS__)
    if (!_M_open)
      return zzs::kernel::Status::OK;
    if (::close (_M_fd) == ERROR) {
      _M_fd = ERROR;
      _M_open = false;
      return zzs::kernel::Status::INTERNAL;
    }
    _M_fd = ERROR;
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
#if defined(__VXWORKS__)
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    (void) buf;
    (void) max_bytes;
    *filled = 0u;
    return zzs::kernel::Status::STALE;
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
#if defined(__VXWORKS__)
    if (!_M_open)
      return zzs::kernel::Status::INTERNAL;
    (void) buf;
    (void) len;
    return zzs::kernel::Status::OK;
#else
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  void
  delay_ms (uint32_t ms) noexcept {
#if defined(__VXWORKS__)
    sysDelayMs (ms);
#else
    (void) ms;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  flush_rx () noexcept {
#if defined(__VXWORKS__)
    if (!_M_open)
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

#if !defined(__VXWORKS__)
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
#if defined(__VXWORKS__)
#include <iosLib.h>
#include <sysLib.h>
#include <vxworks.h>
#endif
  int _M_fd;
  char _M_dev_name[64];
  bool _M_open;
  detail::PortableRing _M_rx{256u};
  detail::PortableRing _M_tx{256u};
};

/** @brief Return the single shared bus instance (function-local static). */
VxworksGnssBus &
bus_ref () noexcept {
  static VxworksGnssBus _S_bus ("/tyCo/0");
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
create_vxworks_gnss_hal () noexcept {
  (void) bus_ref ();
  return &_S_hal;
}

} // namespace zzs::kernel::device::driver::gnss::platform::vxworks
