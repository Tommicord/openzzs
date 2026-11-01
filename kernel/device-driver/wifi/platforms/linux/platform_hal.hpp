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

#pragma once

#include "../platform_hal.hpp"

#include <cstdint>

/**
 * @brief Linux host WiFi HAL via a unix-domain socket-pair stream bus.
 *
 * Provides `create_linux_wifi_hal()` plus the `LinuxStreamBus` class: a real
 * `AF_UNIX`/`SOCK_STREAM` socket-pair bus that wraps one end of a
 * `socketpair()`.  The radio-control hooks are functional no-ops on the host
 * (they validate and accept configuration, return `OK`).  Compiles standalone
 * on any host; the socket operations are guarded by `#ifdef __linux__`.
 * @{
 */
#undef linux
namespace zzs::kernel::device::driver::wifi::platform::linux {

/** @brief Build the Linux WiFi HAL vtable.  The stream bus is supplied by the
 *         caller (e.g. a stack-allocated `LinuxStreamBus` or a mock bus); the
 *         factory populates the radio-control hooks and reports `UNSUPPORTED`
 *         for the bus. */
[[nodiscard]] zzs::base::Maybe<zzs::kernel::device::driver::StreamBusInterface *, zzs::kernel::Status>
create_linux_wifi_hal (WifiHal &hal_out) noexcept;

#if defined(__linux__)

/** @brief Real unix-domain socket-pair stream bus for the Linux host.
 *
 * Wraps one end of an `AF_UNIX` `SOCK_STREAM` socketpair.  `read_bytes()` is
 * non-blocking (`MSG_DONTWAIT`): returns `OK` when data is available,
 * `STALE` when the socket would block.  Intended for stack or member
 * allocation; no heap, no statics.
 */
class LinuxStreamBus : public zzs::kernel::device::driver::StreamBusInterface {
public:
  /** @brief Wrap an existing socket file descriptor. */
  explicit LinuxStreamBus (int fd) noexcept;

  /** @brief Close the socket. */
  ~LinuxStreamBus () override;

  LinuxStreamBus (const LinuxStreamBus &) = delete;
  LinuxStreamBus &
  operator= (const LinuxStreamBus &) = delete;
  LinuxStreamBus (LinuxStreamBus &&) = delete;
  LinuxStreamBus &
  operator= (LinuxStreamBus &&) = delete;

  [[nodiscard]] zzs::kernel::Status
  read_bytes (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept override;

  [[nodiscard]] zzs::kernel::Status
  write_bytes (const uint8_t *buf, uint16_t len) noexcept override;

  void
  delay_ms (uint32_t ms) noexcept override;

  [[nodiscard]] zzs::kernel::Status
  flush_rx () noexcept override;

  [[nodiscard]] bool
  is_open () const noexcept override;

  /** @brief Return the underlying socket file descriptor. */
  [[nodiscard]] int
  fd () const noexcept {
    return _M_fd;
  }

  /** @brief Create a connected non-blocking unix-domain socketpair.
   *
   * `fds[0]` and `fds[1]` are set to the two socket descriptors.  The caller
   * wraps one end in a `LinuxStreamBus` and uses the other for a module
   * emulator or test harness.  No heap, no statics.
   */
  [[nodiscard]] static zzs::kernel::Status
  create_socketpair (int fds[2]) noexcept;

private:
  int _M_fd = -1;
};

#endif /* defined(__linux__) */

/** @} */

} // namespace zzs::kernel::device::driver::wifi::platform::linux
