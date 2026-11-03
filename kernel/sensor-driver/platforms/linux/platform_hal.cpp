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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#include "platform_hal.hpp"

#ifdef __linux__
#include <fcntl.h>
#include <linux/i2c-dev.h>
#include <linux/spi/spidev.h>
#include <sys/ioctl.h>
#include <time.h>
#include <unistd.h>
#endif

#include "kernel/kernel_clock_source.hpp"
#include "kernel/kernel_config.hpp"
#include "kernel/sensor-driver/bus/i2c/i2c_bitbang.hpp"
#include "kernel/sensor-driver/bus/spi/spi_bitbang.hpp"
#include "kernel/sensor-driver/sensors/ics43434/ics43434.hpp"
#include "kernel/sensor-driver/sensors/lis3mdl/lis3mdl.hpp"
#include "kernel/sensor-driver/sensors/lsm303/lsm303.hpp"
#include "kernel/sensor-driver/sensors/lsm6dso/lsm6dso.hpp"
#include "kernel/sensor-driver/sensors/tsl2561/tsl2561.hpp"
#include "kernel/sensor-driver/sensors/veml7700/veml7700.hpp"
#include "kernel/sensor-driver/stream/sensor_stream.hpp"
#include "std/fmt.hpp"
#include "std/maybe.hpp"

#include <cstdint>
#include <cstring>

namespace zzs::kernel::sensor::driver::platform::linux {}

namespace {

/** @brief Platform-native Linux /dev/i2c-N bus adapter. */
class LinuxI2cBus : public bus::BusInterface {
  public:
    explicit LinuxI2cBus (int bus_num, uint8_t addr) noexcept : _M_fd (-1), _M_addr (addr), _M_open_failed (false) {
#ifdef __linux__
        char path[32];
        const unsigned int n = zzs::base::format_into (path, "/dev/i2c-{}", bus_num);
        if (n > 0u && n < sizeof (path)) {
            _M_fd = ::open (path, O_RDWR);
            if (_M_fd >= 0)
                ::ioctl (_M_fd, I2C_SLAVE, _M_addr);
            else
                _M_open_failed = true;
        } else
            _M_open_failed = true;
#endif
    }

    ~LinuxI2cBus () noexcept {
#ifdef __linux__
        if (_M_fd >= 0)
            ::close (_M_fd);
#endif
    }

    LinuxI2cBus (const LinuxI2cBus &) = delete;
    LinuxI2cBus &
    operator= (const LinuxI2cBus &) = delete;
    LinuxI2cBus (LinuxI2cBus &&) = delete;
    LinuxI2cBus &
    operator= (LinuxI2cBus &&) = delete;

    [[nodiscard]] bool
    open_failed () const noexcept {
        return _M_open_failed;
    }

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
#ifdef __linux__
        if (_M_fd < 0)
            return zzs::kernel::Status::INTERNAL;
        uint8_t buf[257];
        if (len + 1u > sizeof (buf))
            return zzs::kernel::Status::OUT_OF_RANGE;
        buf[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            buf[i + 1u] = data[i];
        if (::write (_M_fd, buf, static_cast<size_t> (len + 1u)) < 0)
            return zzs::kernel::Status::INTERNAL;
        return zzs::kernel::Status::OK;
#else
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::UNSUPPORTED;
#endif
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override {
#ifdef __linux__
        if (_M_fd < 0)
            return zzs::kernel::Status::INTERNAL;
        if (::write (_M_fd, &reg, 1u) < 0)
            return zzs::kernel::Status::INTERNAL;
        if (::read (_M_fd, data, static_cast<size_t> (len)) < 0)
            return zzs::kernel::Status::INTERNAL;
        return zzs::kernel::Status::OK;
#else
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::UNSUPPORTED;
#endif
    }

    void
    delay_ms (uint32_t ms) noexcept override {
#ifdef __linux__
        struct timespec ts = {.tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000ull};
        ::nanosleep (&ts, nullptr);
#else
        (void) ms;
#endif
    }

  private:
    int _M_fd;
    uint8_t _M_addr;
    bool _M_open_failed;
};

/** @brief Platform-native Linux /dev/spidev-N bus adapter. */
class LinuxSpiBus : public bus::BusInterface {
  public:
    explicit LinuxSpiBus (int bus_num) noexcept
        : _M_fd (-1), _M_mode (0), _M_bits (8), _M_speed (1000000u), _M_open_failed (false) {
#ifdef __linux__
        char path[32];
        const unsigned int n = zzs::base::format_into (path, "/dev/spidev{}.0", bus_num);
        if (n > 0u && n < sizeof (path)) {
            _M_fd = ::open (path, O_RDWR);
            if (_M_fd < 0)
                _M_open_failed = true;
        } else
            _M_open_failed = true;
#endif
    }

    ~LinuxSpiBus () noexcept {
#ifdef __linux__
        if (_M_fd >= 0)
            ::close (_M_fd);
#endif
    }

    LinuxSpiBus (const LinuxSpiBus &) = delete;
    LinuxSpiBus &
    operator= (const LinuxSpiBus &) = delete;
    LinuxSpiBus (LinuxSpiBus &&) = delete;
    LinuxSpiBus &
    operator= (LinuxSpiBus &&) = delete;

    [[nodiscard]] bool
    open_failed () const noexcept {
        return _M_open_failed;
    }

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
#ifdef __linux__
        if (_M_fd < 0 || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t tx[257];
        tx[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            tx[i + 1u] = data[i];
        struct spi_ioc_transfer tr = {};
        tr.tx_buf = reinterpret_cast<uint64_t> (tx);
        tr.rx_buf = 0;
        tr.len = static_cast<uint32_t> (len + 1u);
        tr.speed_hz = _M_speed;
        tr.bits_per_word = _M_bits;
        if (::ioctl (_M_fd, SPI_IOC_MESSAGE (1), &tr) < 0)
            return zzs::kernel::Status::INTERNAL;
        return zzs::kernel::Status::OK;
#else
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::UNSUPPORTED;
#endif
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override {
#ifdef __linux__
        if (_M_fd < 0 || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t tx[257] = {reg};
        uint8_t rx[257] = {0};
        struct spi_ioc_transfer tr = {};
        tr.tx_buf = reinterpret_cast<uint64_t> (tx);
        tr.rx_buf = reinterpret_cast<uint64_t> (rx);
        tr.len = static_cast<uint32_t> (len + 1u);
        tr.speed_hz = _M_speed;
        tr.bits_per_word = _M_bits;
        if (::ioctl (_M_fd, SPI_IOC_MESSAGE (1), &tr) < 0)
            return zzs::kernel::Status::INTERNAL;
        for (uint16_t i = 0u; i < len; ++i)
            data[i] = rx[i + 1u];
        return zzs::kernel::Status::OK;
#else
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::UNSUPPORTED;
#endif
    }

    void
    delay_ms (uint32_t ms) noexcept override {
#ifdef __linux__
        struct timespec ts = {.tv_sec = ms / 1000, .tv_nsec = (ms % 1000) * 1000000ull};
        ::nanosleep (&ts, nullptr);
#else
        (void) ms;
#endif
    }

  private:
    int _M_fd;
    uint8_t _M_mode;
    uint8_t _M_bits;
    uint32_t _M_speed;
    bool _M_open_failed;
};

/** @brief Shared scan ring: drivers burst-fill it via read_samples(), the HAL
 *         drains it one sample per _S_hal_read_sample() call.  Fixed 16-sample
 *         depth, drop-oldest when full, no heap. */
static stream::ScanRing _S_ring (16u);

/** @brief Burst-fill the shared scan ring from one driver using read_samples().
 *         Returns OK even when the driver reports STALE (no new data). */
[[nodiscard]] zzs::kernel::Status
_S_fill_ring (sensors::SensorDriver *driver) noexcept {
    if (driver == nullptr)
        return zzs::kernel::Status::OK;
    sensor::Sample buf[8u];
    uint16_t filled = 0u;
    zzs::kernel::Status st = driver->read_samples (buf, 8u, &filled);
    if (!zzs::kernel::status_ok (st) && st != zzs::kernel::Status::STALE)
        return st;
    for (uint16_t i = 0u; i < filled; ++i)
        _S_ring.push (buf[i]);
    return zzs::kernel::Status::OK;
}

/* Static bus and driver instances (QNX-style, no config struct, no placement new). */
static LinuxI2cBus _S_i2c_accel_bus (0, 0x6Au); // LSM6DSO accel/gyro, bus 0
static LinuxI2cBus _S_i2c_mag_bus (0, 0x1Cu);   // LIS3MDL, bus 0
static LinuxI2cBus _S_i2c_light_bus (0, 0x39u); // TSL2561/VEML7700, bus 0
static LinuxSpiBus _S_spi_bus (-1);             // No SPI by default

static sensors::Lsm6dsoAccelDriver _S_accel_driver (&_S_i2c_accel_bus);
static sensors::Lsm6dsoGyroDriver _S_gyro_driver (&_S_i2c_accel_bus);
static sensors::Lis3mdlMagDriver _S_mag_driver (&_S_i2c_mag_bus);
static sensors::Tsl2561LightDriver _S_light_driver (&_S_i2c_light_bus);
static sensors::Ics43434SoundDriver _S_sound_driver (&_S_i2c_light_bus, nullptr);

static zzs::kernel::hal::_S_SensorHal _S_hal_vtable;

[[nodiscard]] zzs::kernel::Status
_S_hal_read_sample (sensor::Sample &out) noexcept {
    /* Drain from the shared scan ring first (burst-filled previously). */
    if (_S_ring.pop (out) == zzs::kernel::Status::OK)
        return zzs::kernel::Status::OK;
    /* Ring empty: refill by bursting each enabled driver in round-robin order. */
    static uint8_t idx = 0u;
    for (uint8_t tries = 0u; tries < 5u; ++tries) {
        switch (idx) {
        case 0u:
            (void) _S_fill_ring (&_S_accel_driver);
            break;
        case 1u:
            (void) _S_fill_ring (&_S_gyro_driver);
            break;
        case 2u:
            (void) _S_fill_ring (&_S_mag_driver);
            break;
        case 3u:
            (void) _S_fill_ring (&_S_light_driver);
            break;
        default:
            (void) _S_fill_ring (&_S_sound_driver);
            break;
        }
        idx = static_cast<uint8_t> ((idx + 1u) % 5u);
        if (_S_ring.filled () > 0u)
            break;
    }
    return _S_ring.pop (out);
}

[[nodiscard]] zzs::kernel::Status
_S_hal_set_channel_enabled (sensor::Channel ch, bool enabled) noexcept {
    (void) _S_accel_driver.set_channel_enabled (ch, enabled);
    (void) _S_gyro_driver.set_channel_enabled (ch, enabled);
    (void) _S_mag_driver.set_channel_enabled (ch, enabled);
    (void) _S_light_driver.set_channel_enabled (ch, enabled);
    (void) _S_sound_driver.set_channel_enabled (ch, enabled);
    return zzs::kernel::Status::OK;
}

[[nodiscard]] uint32_t
_S_hal_channel_rate_hz (sensor::Channel ch) noexcept {
    return sensor::channel_rate_hz (ch);
}

[[nodiscard]] zzs::kernel::Status
_S_hal_init (void) noexcept {
    zzs::kernel::Status st = zzs::kernel::clock::clock_source_install (&_S_linux_now_ns);
    if (!zzs::kernel::status_ok (st))
        return st;
    st = _S_accel_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    st = _S_gyro_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    st = _S_mag_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    st = _S_light_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    st = _S_sound_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    return zzs::kernel::Status::OK;
}

} // namespace

[[nodiscard]] const zzs::kernel::hal::_S_SensorHal *
create_linux_sensor_hal () noexcept {
    /* Auto-detect I2C bus for each sensor address; fall back to bus 0. */
    int accel_bus = _S_open_i2c_for_addr (0x6Au);
    int mag_bus = _S_open_i2c_for_addr (0x1Cu);
    int light_bus = _S_open_i2c_for_addr (0x39u);
    int spi_bus_num = _S_probe_spi_bus ();

    if (accel_bus >= 0)
        _S_i2c_accel_bus = LinuxI2cBus (accel_bus, 0x6Au);
    if (mag_bus >= 0)
        _S_i2c_mag_bus = LinuxI2cBus (mag_bus, 0x1Cu);
    if (light_bus >= 0)
        _S_i2c_light_bus = LinuxI2cBus (light_bus, 0x39u);
    if (spi_bus_num >= 0)
        _S_spi_bus = LinuxSpiBus (spi_bus_num);

    _S_hal_vtable._M_read_sample = &_S_hal_read_sample;
    _S_hal_vtable._M_set_channel_enabled = &_S_hal_set_channel_enabled;
    _S_hal_vtable._M_channel_rate_hz = &_S_hal_channel_rate_hz;
    _S_hal_vtable._M_init = &_S_hal_init;
    return &_S_hal_vtable;
}

} // namespace zzs::kernel::sensor::driver::platform::linux
