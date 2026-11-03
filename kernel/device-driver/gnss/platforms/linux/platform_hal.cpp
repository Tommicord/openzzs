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

/** @brief Configuration for the Linux GNSS bus (device path + baud). */
struct LinuxGnssBusConfig {
    char _M_device[64] = "/dev/ttyUSB0";
    uint32_t _M_baud_rate = 9600u;
};

/** @brief Return the bus configuration (settable before first open). */
LinuxGnssBusConfig &
bus_config () noexcept {
    static LinuxGnssBusConfig _S_cfg;
    return _S_cfg;
}

/** @brief Linux UART/USB-CDC byte-stream bus adapter (termios). */
class LinuxGnssBus {
  public:
    explicit LinuxGnssBus (const char *device, uint32_t baud_rate) noexcept : _M_fd (-1), _M_baud_rate (baud_rate) {
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
        if (_M_device[0] == '\0')
            return zzs::kernel::Status::INTERNAL;
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
        speed_t baud = baud_to_speed (_M_baud_rate);
        if (baud == 0) {
            ::close (_M_fd);
            _M_fd = -1;
            return zzs::kernel::Status::INTERNAL;
        }
        ::cfsetispeed (&tios, baud);
        ::cfsetospeed (&tios, baud);
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
        if (buf == nullptr || filled == nullptr)
            return zzs::kernel::Status::OUT_OF_RANGE;
#if defined(__linux__)
        if (_M_fd < 0)
            return zzs::kernel::Status::INTERNAL;
        *filled = 0u;
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
        if (len == 0u)
            return zzs::kernel::Status::OK;
#if defined(__linux__)
        if (_M_fd < 0)
            return zzs::kernel::Status::INTERNAL;
        uint16_t written = 0u;
        // Retry short writes; only a real failure is an error.
        while (written < len) {
            ssize_t n = ::write (_M_fd, buf + written, static_cast<size_t> (len - written));
            if (n < 0) {
                if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)
                    continue;
                return zzs::kernel::Status::INTERNAL;
            }
            if (n == 0)
                break; // nothing accepted; avoid spinning forever
            written = static_cast<uint16_t> (written + static_cast<uint16_t> (n));
        }
        return written == len ? zzs::kernel::Status::OK : zzs::kernel::Status::INTERNAL;
#else
        (void) buf;
        (void) len;
        return zzs::kernel::Status::UNSUPPORTED;
#endif
    }

    void
    delay_ms (uint32_t ms) noexcept {
#if defined(__linux__)
        struct timespec ts = {.tv_sec = static_cast<time_t> (ms / 1000),
                              .tv_nsec = static_cast<long> ((ms % 1000) * 1000000ll)};
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
#if defined(__linux__)
    [[nodiscard]] static speed_t
    baud_to_speed (uint32_t baud) noexcept {
        switch (baud) {
        case 9600u:
            return B9600;
        case 19200u:
            return B19200;
        case 38400u:
            return B38400;
        case 57600u:
            return B57600;
        case 115200u:
            return B115200;
        case 230400u:
            return B230400;
        default:
            return 0;
        }
    }
#endif

    int _M_fd;
    char _M_device[64];
    uint32_t _M_baud_rate;
};

/** @brief Return the single shared bus instance. */
LinuxGnssBus &
bus_ref () noexcept {
    const auto &cfg = bus_config ();
    static LinuxGnssBus _S_bus (cfg._M_device, cfg._M_baud_rate);
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
    return &_S_hal;
}

/** @brief Configure the Linux GNSS bus (device path + baud) before first open. */
[[nodiscard]] zzs::kernel::Status
linux_gnss_hal_set_config (const char *device, uint32_t baud_rate) noexcept {
    auto &cfg = bus_config ();
    if (device != nullptr) {
        uint32_t n = 0u;
        while (device[n] != '\0' && n < sizeof (cfg._M_device) - 1u) {
            cfg._M_device[n] = device[n];
            ++n;
        }
        cfg._M_device[n] = '\0';
    } else {
        cfg._M_device[0] = '\0';
    }
    if (baud_rate != 0u)
        cfg._M_baud_rate = baud_rate;
    return zzs::kernel::Status::OK;
}

} // namespace zzs::kernel::device::driver::gnss::platform::linux
