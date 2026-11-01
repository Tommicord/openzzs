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

#include <cerrno>
#include <cstdint>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#include <termios.h>
#include <time.h>
#include <unistd.h>
#endif

namespace zzs::kernel::device::driver::gnss::platform::linux {

namespace {

/** @brief Linux UART/USB-CDC byte-stream bus adapter (termios). */
class LinuxGnssBus {
public:
  explicit LinuxGnssBus (const char *device) noexcept : _M_fd (-1), _M_device{} {
    if (device != nullptr) {
      uint32_t n = 0u;
      while (device[n] != '\0' && n < sizeof (_M_device) - 1u) {
        _M_device[n] = device[n];
        ++n;
      }
      _M_device[n] = '\0';
    }
  }

  ~LinuxGnssBus () noexcept {
    if (_M_fd >= 0)
      ::close (_M_fd);
  }

  LinuxGnssBus (const LinuxGnssBus &) = delete;
  LinuxGnssBus &
  operator= (const LinuxGnssBus &) = delete;
  LinuxGnssBus (LinuxGnssBus &&) = delete;
  LinuxGnssBus &
  operator= (LinuxGnssBus &&) = delete;

  [[nodiscard]] zzs::kernel::Status
  open () noexcept {
#if defined(__linux__)
    if (_M_fd >= 0)
      return zzs::kernel::Status::OK;
    _M_fd = ::open (_M_device, O_RDWR | O_NOCTTY | O_NONBLOCK);
    if (_M_fd < 0)
      return zzs::kernel::Status::INTERNAL;
    struct termios tios = {};
    if (::tcgetattr (_M_fd, &tios) != 0) {
      ::close (_M_fd);
      _M_fd = -1;
      return zzs::kernel::Status::INTERNAL;
    }
    ::cfmakeraw (&tios);
    ::cfsetispeed (&tios, B9600);
    ::cfsetospeed (&tios, B9600);
    tios.c_cflag |= CLOCAL | CREAD;
    tios.c_cc[VMIN] = 0;
    tios.c_cc[VTIME] = 0;
    if (::tcsetattr (_M_fd, TCSANOW, &tios) != 0) {
      ::close (_M_fd);
      _M_fd = -1;
      return zzs::kernel::Status::INTERNAL;
    }
    ::tcflush (_M_fd, TCIFLUSH);
    return zzs::kernel::Status::OK;
#else
    (void) 0;
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  close () noexcept {
#if defined(__linux__)
    if (_M_fd < 0)
      return zzs::kernel::Status::OK;
    int rc = ::close (_M_fd);
    _M_fd = -1;
    return rc == 0 ? zzs::kernel::Status::OK : zzs::kernel::Status::INTERNAL;
#else
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  read_bytes (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept {
    if (buf == nullptr || filled == nullptr || max_bytes == 0u)
      return zzs::kernel::Status::OUT_OF_RANGE;
#if defined(__linux__)
    if (_M_fd < 0)
      return zzs::kernel::Status::INTERNAL;
    ssize_t n = ::read (_M_fd, buf, static_cast<size_t> (max_bytes));
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
        return zzs::kernel::Status::STALE;
      return zzs::kernel::Status::INTERNAL;
    }
    *filled = static_cast<uint16_t> (n);
    return n > 0 ? zzs::kernel::Status::OK : zzs::kernel::Status::STALE;
#else
    (void) buf;
    (void) max_bytes;
    *filled = 0u;
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  write_bytes (const uint8_t *buf, uint16_t len) noexcept {
    if (buf == nullptr && len > 0u)
      return zzs::kernel::Status::OUT_OF_RANGE;
#if defined(__linux__)
    if (_M_fd < 0)
      return zzs::kernel::Status::INTERNAL;
    ssize_t n = ::write (_M_fd, buf, static_cast<size_t> (len));
    return n == static_cast<ssize_t> (len) ? zzs::kernel::Status::OK : zzs::kernel::Status::INTERNAL;
#else
    (void) buf;
    (void) len;
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  void
  delay_ms (uint32_t ms) noexcept {
#if defined(__linux__)
    struct timespec ts = {.tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000ull};
    ::nanosleep (&ts, nullptr);
#else
    (void) ms;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  flush_rx () noexcept {
#if defined(__linux__)
    if (_M_fd < 0)
      return zzs::kernel::Status::INTERNAL;
    if (::tcflush (_M_fd, TCIFLUSH) != 0)
      return zzs::kernel::Status::INTERNAL;
    return zzs::kernel::Status::OK;
#else
    return zzs::kernel::Status::UNSUPPORTED;
#endif
  }

  [[nodiscard]] bool
  is_open () const noexcept {
    return _M_fd >= 0;
  }

private:
  int _M_fd;
  char _M_device[64];
};

/** @brief Return the single shared bus instance (function-local static). */
LinuxGnssBus &
bus_ref () noexcept {
  static LinuxGnssBus _S_bus ("/dev/ttyUSB0");
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
    &_S_open,  // open
    &_S_close, // close
    &_S_read_bytes, &_S_write_bytes, &_S_delay_ms, &_S_flush_rx,
    &_S_is_open, // is_open
};

} // namespace

[[nodiscard]] const GnssPlatformHal *
create_linux_gnss_hal () noexcept {
  /* Touch the bus so it is constructed before the vtable is handed out. */
  (void) bus_ref ();
  return &_S_hal;
}

} // namespace zzs::kernel::device::driver::gnss::platform::linux
