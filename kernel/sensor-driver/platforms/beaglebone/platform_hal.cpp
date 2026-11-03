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
#include "std/fmt.hpp"
#include "std/maybe.hpp"

#include <cstdint>

namespace zzs::kernel::sensor::driver::platform::beaglebone {

namespace {

/** @brief BeagleBone /dev/i2c-N bus adapter with GPIO sysfs chip select. */
class BeagleBoneI2cBus : public bus::BusInterface {
  public:
    BeagleBoneI2cBus (int bus_num, uint8_t addr) noexcept : _M_fd (-1), _M_addr (addr) {
#ifdef __linux__
        char path[32];
        const unsigned int n = zzs::base::format_into (path, "/dev/i2c-{}", bus_num);
        (void) n;
        _M_fd = open (path, O_RDWR);
        if (_M_fd >= 0)
            ioctl (_M_fd, I2C_SLAVE, _M_addr);
#endif
    }
    ~BeagleBoneI2cBus () noexcept {
#ifdef __linux__
        if (_M_fd >= 0)
            close (_M_fd);
#endif
    }
    BeagleBoneI2cBus (const BeagleBoneI2cBus &) = delete;
    BeagleBoneI2cBus &
    operator= (const BeagleBoneI2cBus &) = delete;
    BeagleBoneI2cBus (BeagleBoneI2cBus &&) = delete;
    BeagleBoneI2cBus &
    operator= (BeagleBoneI2cBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
#ifdef __linux__
        if (_M_fd < 0 || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t buf[257];
        buf[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            buf[i + 1u] = data[i];
        if (write (_M_fd, buf, static_cast<size_t> (len + 1u)) < 0)
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
        if (write (_M_fd, &reg, 1u) < 0)
            return zzs::kernel::Status::INTERNAL;
        if (read (_M_fd, data, static_cast<size_t> (len)) < 0)
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
        struct timespec ts = {ms / 1000, (ms % 1000) * 1000000ull};
        nanosleep (&ts, nullptr);
#else
        (void) ms;
#endif
    }

  private:
    int _M_fd;
    uint8_t _M_addr;
};

/** @brief Monotonic nanoseconds via CLOCK_MONOTONIC_RAW. */
[[nodiscard]] uint64_t
_S_beaglebone_now_ns (void) noexcept {
#ifdef __linux__
    struct timespec ts;
    if (clock_gettime (CLOCK_MONOTONIC_RAW, &ts) == 0)
        return static_cast<uint64_t> (ts.tv_sec) * 1000000000ull + static_cast<uint64_t> (ts.tv_nsec);
#endif
    return 0ull;
}

static BeagleBoneI2cBus _S_i2c_accel_bus (1, 0x6Au);
static BeagleBoneI2cBus _S_i2c_mag_bus (1, 0x1Cu);
static BeagleBoneI2cBus _S_i2c_light_bus (2, 0x39u);

static sensors::Lsm6dsoAccelDriver _S_accel_driver (&_S_i2c_accel_bus);
static sensors::Lsm6dsoGyroDriver _S_gyro_driver (&_S_i2c_accel_bus);
static sensors::Lis3mdlMagDriver _S_mag_driver (&_S_i2c_mag_bus);
static sensors::Tsl2561LightDriver _S_light_driver (&_S_i2c_light_bus);
static sensors::Ics43434SoundDriver _S_sound_driver (&_S_i2c_light_bus, nullptr);

static zzs::kernel::hal::_S_SensorHal _S_hal_vtable;

[[nodiscard]] zzs::kernel::Status
_S_hal_read_sample (sensor::Sample &out) noexcept {
    static uint8_t idx = 0u;
    zzs::kernel::Status st = zzs::kernel::Status::STALE;
    for (uint8_t tries = 0u; tries < 5u; ++tries) {
        switch (idx) {
        case 0u:
            st = _S_accel_driver.read_sample (out);
            break;
        case 1u:
            st = _S_gyro_driver.read_sample (out);
            break;
        case 2u:
            st = _S_mag_driver.read_sample (out);
            break;
        case 3u:
            st = _S_light_driver.read_sample (out);
            break;
        default:
            st = _S_sound_driver.read_sample (out);
            break;
        }
        idx = static_cast<uint8_t> ((idx + 1u) % 5u);
        if (zzs::kernel::status_ok (st))
            return zzs::kernel::Status::OK;
    }
    return st;
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
    zzs::kernel::Status st = clock_source_install (&_S_beaglebone_now_ns);
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

[[nodiscard]] zzs::base::Maybe<const zzs::kernel::hal::_S_SensorHal *, zzs::kernel::Status>
create_beaglebone_sensor_hal () noexcept {
    _S_hal_vtable._M_read_sample = &_S_hal_read_sample;
    _S_hal_vtable._M_set_channel_enabled = &_S_hal_set_channel_enabled;
    _S_hal_vtable._M_channel_rate_hz = &_S_hal_channel_rate_hz;
    _S_hal_vtable._M_init = &_S_hal_init;
    return zzs::base::Maybe<const zzs::kernel::hal::_S_SensorHal *, zzs::kernel::Status> (&_S_hal_vtable);
}

} // namespace zzs::kernel::sensor::driver::platform::beaglebone
