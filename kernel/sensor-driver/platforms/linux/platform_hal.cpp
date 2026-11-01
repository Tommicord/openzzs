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
#  include <fcntl.h>
#  include <unistd.h>
#  include <sys/ioctl.h>
#  include <linux/i2c-dev.h>
#  include <linux/spi/spidev.h>
#  include <time.h>
#  include <cstdlib>
#endif

#include "kernel/sensor-driver/bus/i2c/i2c_bitbang.hpp"
#include "kernel/sensor-driver/bus/spi/spi_bitbang.hpp"
#include "kernel/sensor-driver/sensors/lsm6dso/lsm6dso.hpp"
#include "kernel/sensor-driver/sensors/lis3mdl/lis3mdl.hpp"
#include "kernel/sensor-driver/sensors/lsm303/lsm303.hpp"
#include "kernel/sensor-driver/sensors/tsl2561/tsl2561.hpp"
#include "kernel/sensor-driver/sensors/veml7700/veml7700.hpp"
#include "kernel/sensor-driver/sensors/ics43434/ics43434.hpp"
#include "kernel/kernel_clock_source.hpp"
#include "kernel/kernel_config.hpp"

#include "std/fmt.hpp"

#include <cstdint>

/* `linux` is a predefined macro (=1) on GCC/Linux; undef it so the
 * `platform::linux` namespace declaration parses correctly. */
#undef linux
namespace zzs::kernel::sensor::driver::platform::linux
{

namespace {

/** @brief Build a /dev/i2c-N device path from a bus number using format_into. */
[[nodiscard]] bool
i2c_dev_path (int bus, char (&out)[32]) noexcept
{
    const unsigned int n = zzs::base::format_into (out, "/dev/i2c-{}", bus);
    return n > 0u && n < sizeof (out);
}

/** @brief Platform-native Linux /dev/i2c-N bus adapter. */
class LinuxI2cBus : public bus::BusInterface
{
public:
    explicit LinuxI2cBus (const char *dev_path, uint8_t addr) noexcept
        : _M_fd (-1)
        , _M_addr (addr)
    {
#ifdef __linux__
        _M_fd = ::open (dev_path, O_RDWR);
        if (_M_fd >= 0)
            ::ioctl (_M_fd, I2C_SLAVE, _M_addr);
        else
            _M_open_failed = true;
#endif
    }

    ~LinuxI2cBus () noexcept
    {
#ifdef __linux__
        if (_M_fd >= 0)
            ::close (_M_fd);
#endif
    }

    LinuxI2cBus (const LinuxI2cBus &) = delete;
    LinuxI2cBus &operator= (const LinuxI2cBus &) = delete;
    LinuxI2cBus (LinuxI2cBus &&) = delete;
    LinuxI2cBus &operator= (LinuxI2cBus &&) = delete;

    [[nodiscard]] bool
    open_failed () const noexcept
    {
        return _M_open_failed;
    }

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override
    {
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
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override
    {
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
    delay_ms (uint32_t ms) noexcept override
    {
#ifdef __linux__
        struct timespec ts = { .tv_sec = ms / 1000,
                               .tv_nsec = (ms % 1000) * 1000000ull };
        ::nanosleep (&ts, nullptr);
#else
        (void) ms;
#endif
    }

private:
    int _M_fd;
    uint8_t _M_addr;
    bool _M_open_failed = false;
};

/** @brief Platform-native Linux /dev/spidev-N bus adapter. */
class LinuxSpiBus : public bus::BusInterface
{
public:
    explicit LinuxSpiBus (const char *dev_path) noexcept
        : _M_fd (-1)
        , _M_mode (0)
        , _M_bits (8)
        , _M_speed (1000000u)
    {
#ifdef __linux__
        _M_fd = ::open (dev_path, O_RDWR);
        if (_M_fd < 0)
            _M_open_failed = true;
#endif
    }

    ~LinuxSpiBus () noexcept
    {
#ifdef __linux__
        if (_M_fd >= 0)
            ::close (_M_fd);
#endif
    }

    LinuxSpiBus (const LinuxSpiBus &) = delete;
    LinuxSpiBus &operator= (const LinuxSpiBus &) = delete;
    LinuxSpiBus (LinuxSpiBus &&) = delete;
    LinuxSpiBus &operator= (LinuxSpiBus &&) = delete;

    [[nodiscard]] bool
    open_failed () const noexcept
    {
        return _M_open_failed;
    }

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override
    {
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
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override
    {
#ifdef __linux__
        if (_M_fd < 0 || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t tx[257] = { reg };
        uint8_t rx[257] = { 0 };
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
    delay_ms (uint32_t ms) noexcept override
    {
#ifdef __linux__
        struct timespec ts = { .tv_sec = ms / 1000,
                               .tv_nsec = (ms % 1000) * 1000000ull };
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
    bool _M_open_failed = false;
};

/** @brief Monotonic nanoseconds via CLOCK_MONOTONIC_RAW. */
[[nodiscard]] uint64_t
linux_now_ns (void) noexcept
{
#ifdef __linux__
    struct timespec ts;
    if (::clock_gettime (CLOCK_MONOTONIC_RAW, &ts) == 0)
    {
        uint64_t sec = 0ull;
        uint64_t nsec = 0ull;
        if (__builtin_mul_overflow (static_cast<uint64_t> (ts.tv_sec),
                                    1000000000ull, &sec))
            return 0ull;
        nsec = static_cast<uint64_t> (ts.tv_nsec);
        if (__builtin_add_overflow (sec, nsec, &sec))
            return 0ull;
        return sec;
    }
#endif
    return 0ull;
}

/** File-scope storage for config-driven buses/drivers (placement new). */
struct HalObjects
{
    LinuxI2cBus *i2c_accel = nullptr;
    LinuxI2cBus *i2c_mag = nullptr;
    LinuxI2cBus *i2c_light = nullptr;
    LinuxSpiBus *spi = nullptr;

    sensors::Lsm6dsoAccelDriver *accel = nullptr;
    sensors::Lsm6dsoGyroDriver *gyro = nullptr;
    sensors::Lis3mdlMagDriver *mag = nullptr;
    sensors::Tsl2561LightDriver *light = nullptr;
    sensors::Ics43434SoundDriver *sound = nullptr;
};

alignas (LinuxI2cBus) static unsigned char i2c_accel_storage [sizeof (LinuxI2cBus)];
alignas (LinuxI2cBus) static unsigned char i2c_mag_storage [sizeof (LinuxI2cBus)];
alignas (LinuxI2cBus) static unsigned char i2c_light_storage [sizeof (LinuxI2cBus)];
alignas (LinuxSpiBus) static unsigned char spi_storage [sizeof (LinuxSpiBus)];

alignas (sensors::Lsm6dsoAccelDriver)
    static unsigned char accel_storage [sizeof (sensors::Lsm6dsoAccelDriver)];
alignas (sensors::Lsm6dsoGyroDriver)
    static unsigned char gyro_storage [sizeof (sensors::Lsm6dsoGyroDriver)];
alignas (sensors::Lis3mdlMagDriver)
    static unsigned char mag_storage [sizeof (sensors::Lis3mdlMagDriver)];
alignas (sensors::Tsl2561LightDriver)
    static unsigned char light_storage [sizeof (sensors::Tsl2561LightDriver)];
alignas (sensors::Ics43434SoundDriver)
    static unsigned char sound_storage [sizeof (sensors::Ics43434SoundDriver)];

static HalObjects hal_objs {};
static zzs::kernel::hal::_S_SensorHal _S_HalVtable {};
static bool _S_HalConstructed = false;

[[nodiscard]] zzs::kernel::Status
hal_read_sample (sensor::Sample &out) noexcept
{
    if (!_S_HalConstructed)
        return zzs::kernel::Status::NOT_READY;
    /* Round-robin across enabled drivers: accel, gyro, mag, light, sound. */
    static uint8_t idx = 0u;
    zzs::kernel::Status st = zzs::kernel::Status::STALE;
    for (uint8_t tries = 0u; tries < 5u; ++tries)
        {
        switch (idx)
            {
            case 0u:
                if (hal_objs.accel != nullptr)
                    st = hal_objs.accel->read_sample (out);
                break;
            case 1u:
                if (hal_objs.gyro != nullptr)
                    st = hal_objs.gyro->read_sample (out);
                break;
            case 2u:
                if (hal_objs.mag != nullptr)
                    st = hal_objs.mag->read_sample (out);
                break;
            case 3u:
                if (hal_objs.light != nullptr)
                    st = hal_objs.light->read_sample (out);
                break;
            default:
                if (hal_objs.sound != nullptr)
                    st = hal_objs.sound->read_sample (out);
                break;
            }
        idx = static_cast<uint8_t> ((idx + 1u) % 5u);
        if (zzs::kernel::status_ok (st))
            return zzs::kernel::Status::OK;
        }
    return st;
}

[[nodiscard]] zzs::kernel::Status
hal_set_channel_enabled (sensor::Channel ch, bool enabled) noexcept
{
    if (hal_objs.accel != nullptr)
        (void) hal_objs.accel->set_channel_enabled (ch, enabled);
    if (hal_objs.gyro != nullptr)
        (void) hal_objs.gyro->set_channel_enabled (ch, enabled);
    if (hal_objs.mag != nullptr)
        (void) hal_objs.mag->set_channel_enabled (ch, enabled);
    if (hal_objs.light != nullptr)
        (void) hal_objs.light->set_channel_enabled (ch, enabled);
    if (hal_objs.sound != nullptr)
        (void) hal_objs.sound->set_channel_enabled (ch, enabled);
    return zzs::kernel::Status::OK;
}

[[nodiscard]] uint32_t
hal_channel_rate_hz (sensor::Channel ch) noexcept
{
    return sensor::channel_rate_hz (ch);
}

[[nodiscard]] zzs::kernel::Status
hal_init (void) noexcept
{
    if (!_S_HalConstructed)
        return zzs::kernel::Status::NOT_READY;
    zzs::kernel::Status st = zzs::kernel::Status::OK;
    st = zzs::kernel::clock::clock_source_install (&linux_now_ns);
    if (!zzs::kernel::status_ok (st))
        return st;
    if (hal_objs.accel != nullptr)
        {
        st = hal_objs.accel->init ();
        if (!zzs::kernel::status_ok (st))
            return st;
        }
    if (hal_objs.gyro != nullptr)
        {
        st = hal_objs.gyro->init ();
        if (!zzs::kernel::status_ok (st))
            return st;
        }
    if (hal_objs.mag != nullptr)
        {
        st = hal_objs.mag->init ();
        if (!zzs::kernel::status_ok (st))
            return st;
        }
    if (hal_objs.light != nullptr)
        {
        st = hal_objs.light->init ();
        if (!zzs::kernel::status_ok (st))
            return st;
        }
    if (hal_objs.sound != nullptr)
        {
        st = hal_objs.sound->init ();
        if (!zzs::kernel::status_ok (st))
            return st;
        }
    return zzs::kernel::Status::OK;
}

} // namespace

[[nodiscard]] const zzs::kernel::hal::_S_SensorHal *
create_linux_sensor_hal (const PlatformConfig &config) noexcept
{
    if (_S_HalConstructed)
        return &_S_HalVtable;

    char accel_path[32] = {};
    char mag_path[32] = {};
    char light_path[32] = {};

    const bool use_accel_path =
        config.accel_i2c.enabled && i2c_dev_path (config.i2c_bus, accel_path);
    const bool use_mag_path =
        config.mag_i2c.enabled && i2c_dev_path (config.i2c_bus, mag_path);
    const bool use_light_path =
        config.light_i2c.enabled && i2c_dev_path (config.i2c_bus, light_path);

    /* Accel / gyro share the accel bus. Create it when the accel device is
     * configured and at least one of accel/gyro is enabled. */
    const bool accel_bus_needed =
        config.accel_i2c.enabled &&
        (has_flag (config.enabled_sensors, SensorMask::accel) ||
         has_flag (config.enabled_sensors, SensorMask::gyro));
    if (accel_bus_needed && !use_accel_path)
        return nullptr;
    if (config.mag_i2c.enabled && !use_mag_path)
        return nullptr;
    if (config.light_i2c.enabled && !use_light_path)
        return nullptr;

    /* Accel / gyro share the accel bus. */
    if (accel_bus_needed)
        {
        new (i2c_accel_storage) LinuxI2cBus (accel_path, config.accel_i2c.addr);
        hal_objs.i2c_accel =
            reinterpret_cast<LinuxI2cBus *> (i2c_accel_storage);
        }
    if (config.mag_i2c.enabled)
        {
        new (i2c_mag_storage) LinuxI2cBus (mag_path, config.mag_i2c.addr);
        hal_objs.i2c_mag = reinterpret_cast<LinuxI2cBus *> (i2c_mag_storage);
        }
    if (config.light_i2c.enabled)
        {
        new (i2c_light_storage) LinuxI2cBus (light_path, config.light_i2c.addr);
        hal_objs.i2c_light = reinterpret_cast<LinuxI2cBus *> (i2c_light_storage);
        }
    if (config.i2c_dev_path != nullptr)
        {
        new (spi_storage) LinuxSpiBus (config.i2c_dev_path);
        hal_objs.spi = reinterpret_cast<LinuxSpiBus *> (spi_storage);
        }

    bool any_sensor = false;
    if (has_flag (config.enabled_sensors, SensorMask::accel) &&
        hal_objs.i2c_accel != nullptr)
        {
        new (accel_storage)
            sensors::Lsm6dsoAccelDriver (hal_objs.i2c_accel);
        hal_objs.accel = reinterpret_cast<sensors::Lsm6dsoAccelDriver *> (
            accel_storage);
        any_sensor = true;
        }
    if (has_flag (config.enabled_sensors, SensorMask::gyro) &&
        hal_objs.i2c_accel != nullptr)
        {
        new (gyro_storage)
            sensors::Lsm6dsoGyroDriver (hal_objs.i2c_accel);
        hal_objs.gyro = reinterpret_cast<sensors::Lsm6dsoGyroDriver *> (
            gyro_storage);
        any_sensor = true;
        }
    if (has_flag (config.enabled_sensors, SensorMask::mag) &&
        hal_objs.i2c_mag != nullptr)
        {
        new (mag_storage) sensors::Lis3mdlMagDriver (hal_objs.i2c_mag);
        hal_objs.mag = reinterpret_cast<sensors::Lis3mdlMagDriver *> (mag_storage);
        any_sensor = true;
        }
    if (has_flag (config.enabled_sensors, SensorMask::light) &&
        hal_objs.i2c_light != nullptr)
        {
        new (light_storage)
            sensors::Tsl2561LightDriver (hal_objs.i2c_light);
        hal_objs.light = reinterpret_cast<sensors::Tsl2561LightDriver *> (
            light_storage);
        any_sensor = true;
        }
    if (has_flag (config.enabled_sensors, SensorMask::sound) &&
        hal_objs.i2c_light != nullptr)
        {
        new (sound_storage)
            sensors::Ics43434SoundDriver (hal_objs.i2c_light);
        hal_objs.sound = reinterpret_cast<sensors::Ics43434SoundDriver *> (
            sound_storage);
        any_sensor = true;
        }
    if (!any_sensor)
        return nullptr;
    _S_HalConstructed = true;
    _S_HalVtable._M_read_sample = &hal_read_sample;
    _S_HalVtable._M_set_channel_enabled = &hal_set_channel_enabled;
    _S_HalVtable._M_channel_rate_hz = &hal_channel_rate_hz;
    _S_HalVtable._M_init = &hal_init;
    return &_S_HalVtable;
}

} // namespace zzs::kernel::sensor::driver::platform::linux
