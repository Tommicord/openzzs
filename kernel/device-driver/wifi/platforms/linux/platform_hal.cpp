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

/**
 * @file linux/platform_hal.cpp
 * @brief Linux host WiFi HAL: functional radio-control hooks plus a real
 *        unix-domain socket-pair stream bus class.
 *
 * On the host there is no physical WiFi radio, so the radio-control hooks are
 * functional no-ops: they accept, validate and store configuration and return
 * `OK`.  The `LinuxStreamBus` class wraps one end of an
 * `socketpair(AF_UNIX, SOCK_STREAM)`; it is intended to be stack- or member-
 * allocated by the application (no heap, no statics).  `create_linux_wifi_hal`
 * populates the `WifiHal` vtable and reports `UNSUPPORTED` for the bus, which
 * means the application supplies its own stream bus (e.g. a `LinuxStreamBus`
 * connected to a module emulator, or a mock bus in tests).
 *
 * All operations are non-blocking; `read_bytes()` returns `STALE` when no
 * data is available.  No heap, no writable statics, no exceptions, no float.
 * @{
 */

#include "platform_hal.hpp"

#if defined(__linux__)
#include <cerrno>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#endif

#include "kernel/kernel_status.hpp"

#include <cstdint>
#include <cstring>

#undef linux
namespace zzs::kernel::device::driver::wifi::platform::linux {

namespace {

#if defined(__linux__)

/** @brief Real unix-domain socket-pair stream bus for the Linux host.
 *
 * Wraps one end of an `AF_UNIX` `SOCK_STREAM` socketpair.  `read_bytes()` is
 * non-blocking (`MSG_DONTWAIT`): returns `OK` when data is available,
 * `STALE` when the socket would block.  `write_bytes()` sends all bytes.
 * Intended for stack or member allocation; no heap, no statics.
 */
class LinuxStreamBus : public zzs::kernel::device::driver::StreamBusInterface {
public:
  /** @brief Wrap an existing socket file descriptor. */
  explicit LinuxStreamBus (int fd) noexcept : _M_fd (fd) {
  }

  ~LinuxStreamBus () override {
    if (_M_fd >= 0)
      ::close (_M_fd);
  }

  LinuxStreamBus (const LinuxStreamBus &) = delete;
  LinuxStreamBus &
  operator= (const LinuxStreamBus &) = delete;
  LinuxStreamBus (LinuxStreamBus &&) = delete;
  LinuxStreamBus &
  operator= (LinuxStreamBus &&) = delete;

  [[nodiscard]] zzs::kernel::Status
  read_bytes (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept {
    *filled = 0u;
    if (buf == nullptr || filled == nullptr)
      return zzs::kernel::Status::OUT_OF_RANGE;
    if (_M_fd < 0)
      return zzs::kernel::Status::INTERNAL;
    ssize_t n = ::recv (_M_fd, buf, static_cast<size_t> (max_bytes), MSG_DONTWAIT);
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK)
        return zzs::kernel::Status::STALE;
      return zzs::kernel::Status::INTERNAL;
    }
    if (n == 0)
      return zzs::kernel::Status::STALE;
    *filled = static_cast<uint16_t> (n);
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] zzs::kernel::Status
  write_bytes (const uint8_t *buf, uint16_t len) noexcept {
    if (buf == nullptr && len > 0u)
      return zzs::kernel::Status::OUT_OF_RANGE;
    if (_M_fd < 0)
      return zzs::kernel::Status::INTERNAL;
    ssize_t n = ::send (_M_fd, buf, static_cast<size_t> (len), 0);
    if (n < 0)
      return zzs::kernel::Status::INTERNAL;
    return zzs::kernel::Status::OK;
  }

  void
  delay_ms (uint32_t ms) noexcept override {
#if defined(__linux__)
    struct timespec ts = {.tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000ull};
    ::nanosleep (&ts, nullptr);
#else
    (void) ms;
#endif
  }

  [[nodiscard]] zzs::kernel::Status
  flush_rx () noexcept override {
    if (_M_fd < 0)
      return zzs::kernel::Status::INTERNAL;
    uint8_t dummy[64];
    uint16_t filled = 0u;
    for (;;) {
      zzs::kernel::Status st = read_bytes (dummy, sizeof (dummy), &filled);
      if (st == zzs::kernel::Status::STALE)
        break;
      if (!zzs::kernel::status_ok (st))
        return st;
    }
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] bool
  is_open () const noexcept override {
    return _M_fd >= 0;
  }

  /** @brief Return the underlying socket file descriptor. */
  [[nodiscard]] int
  fd () const noexcept {
    return _M_fd;
  }

private:
  int _M_fd;
};

/** @brief Create a connected unix-domain socketpair.  Returns both fds; the
 *         caller wraps one end in a `LinuxStreamBus` (stack-allocated) and
 *         uses the other for a module emulator.  No heap, no statics. */
[[nodiscard]] zzs::kernel::Status
_S_create_socketpair (int fds[2]) noexcept {
#if defined(__linux__)
  fds[0] = -1;
  fds[1] = -1;
  if (::socketpair (AF_UNIX, SOCK_STREAM, 0, fds) != 0)
    return zzs::kernel::Status::INTERNAL;
  for (int i = 0; i < 2; ++i) {
    int flags = ::fcntl (fds[i], F_GETFL, 0);
    if (flags < 0 || ::fcntl (fds[i], F_SETFL, flags | O_NONBLOCK) != 0) {
      ::close (fds[0]);
      ::close (fds[1]);
      fds[0] = -1;
      fds[1] = -1;
      return zzs::kernel::Status::INTERNAL;
    }
  }
  return zzs::kernel::Status::OK;
#else
  (void) fds;
  return zzs::kernel::Status::UNSUPPORTED;
#endif
}

/** @brief Radio-control hook: reset (host functional no-op). */
[[nodiscard]] zzs::kernel::Status
_S_reset () noexcept {
  return zzs::kernel::Status::OK;
}

/** @brief Radio-control hook: set channel (host functional no-op, validated). */
[[nodiscard]] zzs::kernel::Status
_S_set_channel (uint8_t channel) noexcept {
  if (channel < 1u || channel > 165u)
    return zzs::kernel::Status::OUT_OF_RANGE;
  return zzs::kernel::Status::OK;
}

/** @brief Radio-control hook: set country (host functional no-op, validated). */
[[nodiscard]] zzs::kernel::Status
_S_set_country (const char cc[2]) noexcept {
  if (cc == nullptr)
    return zzs::kernel::Status::OUT_OF_RANGE;
  return zzs::kernel::Status::OK;
}

/** @brief Radio-control hook: MAC address (host fixed locally-administered MAC). */
[[nodiscard]] zzs::kernel::Status
_S_mac_addr (uint8_t mac[6]) noexcept {
  if (mac == nullptr)
    return zzs::kernel::Status::OUT_OF_RANGE;
  const uint8_t host_mac[6] = {0x02, 0x00, 0x00, 0x12, 0x34, 0x56};
  for (uint8_t i = 0u; i < 6u; ++i)
    mac[i] = host_mac[i];
  return zzs::kernel::Status::OK;
}

/** @brief Radio-control hook: delay (host nanosleep). */
void
_S_delay_ms (uint32_t ms) noexcept {
#if defined(__linux__)
  struct timespec ts = {.tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000ull};
  ::nanosleep (&ts, nullptr);
#else
  (void) ms;
#endif
}

#endif /* defined(__linux__) */

} // namespace

[[nodiscard]] zzs::base::Maybe<zzs::kernel::device::driver::StreamBusInterface *, zzs::kernel::Status>
create_linux_wifi_hal (WifiHal &hal_out) noexcept {
#if defined(__linux__)
  /* Populate the radio-control vtable with functional host hooks. */
  hal_out._M_reset = &_S_reset;
  hal_out._M_set_channel = &_S_set_channel;
  hal_out._M_set_country = &_S_set_country;
  hal_out._M_mac_addr = &_S_mac_addr;
  hal_out._M_delay_ms = &_S_delay_ms;

  /* On Linux there is no in-kernel WiFi radio to enumerate here.  The stream
   * bus is supplied by the application (e.g. a stack-allocated LinuxStreamBus
   * over a socketpair, or a MockStreamBus in tests).  Report UNSUPPORTED so
   * the caller knows it must provide its own bus. */
  return zzs::kernel::Status::UNSUPPORTED;
#else
  (void) hal_out;
  return zzs::kernel::Status::UNSUPPORTED;
#endif
}

/** @} */

} // namespace zzs::kernel::device::driver::wifi::platform::linux
