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

#ifdef CONFIG_ZEPHYR
#  include <zephyr/kernel.h>
#  include <zephyr/drivers/i2c.h>
#  include <zephyr/drivers/spi.h>
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

#include <cstdint>

namespace zzs::kernel::sensor::driver::platform::zephyr
{

namespace {

/** @brief Zephyr I2C device driver adapter (i2c_dt_spec). */
class ZephyrI2cBus : public bus::BusInterface
{
public:
    ZephyrI2cBus (const struct device *dev, uint16_t addr) noexcept
        : _M_dev (dev), _M_addr (addr) {}
    ~ZephyrI2cBus () noexcept = default;
    ZephyrI2cBus (const ZephyrI2cBus &) = delete;
    ZephyrI2cBus &operator= (const ZephyrI2cBus &) = delete;
    ZephyrI2cBus (ZephyrI2cBus &&) = delete;
    ZephyrI2cBus &operator= (ZephyrI2cBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override
    {
#ifdef CONFIG_ZEPHYR
        if (_M_dev == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t buf[257];
        buf[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            buf[i + 1u] = data[i];
        struct i2c_msg msgs[1] = {
            { .buf = buf, .len = static_cast<uint32_t> (len + 1u),
              .flags = I2C_WRITE | I2C_ADDR(_M_addr) }
        };
        if (i2c_transfer (_M_dev, msgs, 1, _M_addr) != 0)
            return zzs::kernel::Status::INTERNAL;
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override
    {
#ifdef CONFIG_ZEPHYR
        if (_M_dev == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t wbuf = reg;
        struct i2c_msg msgs[2] = {
            { .buf = &wbuf, .len = 1u, .flags = I2C_WRITE | I2C_ADDR(_M_addr) },
            { .buf = data, .len = static_cast<uint32_t> (len),
              .flags = I2C_READ | I2C_ADDR(_M_addr) | I2C_RESTART }
        };
        if (i2c_transfer (_M_dev, msgs, 2, _M_addr) != 0)
            return zzs::kernel::Status::INTERNAL;
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    void
    delay_ms (uint32_t ms) noexcept override
    {
#ifdef CONFIG_ZEPHYR
        k_msleep (static_cast<int32_t> (ms));
#else
        (void) ms;
#endif
    }

    void
    set_device (const struct device *dev) noexcept { _M_dev = dev; }

private:
    const struct device *_M_dev;
    uint16_t _M_addr;
};

/** @brief Zephyr SPI device driver adapter (spi_dt_spec). */
class ZephyrSpiBus : public bus::BusInterface
{
public:
    ZephyrSpiBus (const struct device *dev, uint8_t cs) noexcept
        : _M_dev (dev), _M_cs (cs) {}
    ~ZephyrSpiBus () noexcept = default;
    ZephyrSpiBus (const ZephyrSpiBus &) = delete;
    ZephyrSpiBus &operator= (const ZephyrSpiBus &) = delete;
    ZephyrSpiBus (ZephyrSpiBus &&) = delete;
    ZephyrSpiBus &operator= (ZephyrSpiBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override
    {
#ifdef CONFIG_ZEPHYR
        if (_M_dev == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t tx[257];
        tx[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            tx[i + 1u] = data[i];
        struct spi_config cfg = { .frequency = 1000000u };
        const struct spi_cs_control control = { .gpio_dev = nullptr,
                                                .gpio_pin = _M_cs,
                                                .gpio_dt_flags = 0 };
        cfg.cs = &control;
        struct spi_buf buf = { .buf = tx, .len = static_cast<size_t> (len + 1u) };
        struct spi_buf_set tx_set = { .buffers = &buf, .count = 1u };
        if (spi_transceive (_M_dev, &cfg, &tx_set, nullptr) != 0)
            return zzs::kernel::Status::INTERNAL;
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override
    {
#ifdef CONFIG_ZEPHYR
        if (_M_dev == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t tx[257] = { reg };
        uint8_t rx[257] = { 0 };
        struct spi_config cfg = { .frequency = 1000000u };
        const struct spi_cs_control control = { .gpio_dev = nullptr,
                                                .gpio_pin = _M_cs,
                                                .gpio_dt_flags = 0 };
        cfg.cs = &control;
        struct spi_buf txb = { .buf = tx, .len = static_cast<size_t> (len + 1u) };
        struct spi_buf rxb = { .buf = rx, .len = static_cast<size_t> (len + 1u) };
        struct spi_buf_set tx_set = { .buffers = &txb, .count = 1u };
        struct spi_buf_set rx_set = { .buffers = &rxb, .count = 1u };
        if (spi_transceive (_M_dev, &cfg, &tx_set, &rx_set) != 0)
            return zzs::kernel::Status::INTERNAL;
        for (uint16_t i = 0u; i < len; ++i)
            data[i] = rx[i + 1u];
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    void
    delay_ms (uint32_t ms) noexcept override
    {
#ifdef CONFIG_ZEPHYR
        k_msleep (static_cast<int32_t> (ms));
#else
        (void) ms;
#endif
    }

private:
    const struct device *_M_dev;
    uint8_t _M_cs;
};

/** @brief Monotonic nanoseconds via k_uptime_ticks(). */
[[nodiscard]] uint64_t
zephyr_now_ns (void) noexcept
{
#ifdef CONFIG_ZEPHYR
    uint64_t ticks = k_uptime_ticks ();
    return ticks * 1000000000ull / static_cast<uint64_t> (CONFIG_SYS_CLOCK_TICKS_PER_SEC);
#endif
    return 0ull;
}

static ZephyrI2cBus i2c_accel_bus (nullptr, 0x6Au);
static ZephyrI2cBus i2c_mag_bus (nullptr, 0x1Cu);
static ZephyrI2cBus i2c_light_bus (nullptr, 0x39u);

static sensors::Lsm6dsoAccelDriver accel_driver (&i2c_accel_bus);
static sensors::Lsm6dsoGyroDriver gyro_driver (&i2c_accel_bus);
static sensors::Lis3mdlMagDriver mag_driver (&i2c_mag_bus);
static sensors::Tsl2561LightDriver light_driver (&i2c_light_bus);
static sensors::Ics43434SoundDriver sound_driver (&i2c_light_bus, nullptr);

static zzs::kernel::hal::_S_SensorHal hal_vtable;

[[nodiscard]] zzs::kernel::Status
hal_read_sample (sensor::Sample &out) noexcept
{
    static uint8_t idx = 0u;
    zzs::kernel::Status st = zzs::kernel::Status::STALE;
    for (uint8_t tries = 0u; tries < 5u; ++tries)
        {
        switch (idx)
            {
            case 0u: st = accel_driver.read_sample (out); break;
            case 1u: st = gyro_driver.read_sample (out); break;
            case 2u: st = mag_driver.read_sample (out); break;
            case 3u: st = light_driver.read_sample (out); break;
            default: st = sound_driver.read_sample (out); break;
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
    (void) accel_driver.set_channel_enabled (ch, enabled);
    (void) gyro_driver.set_channel_enabled (ch, enabled);
    (void) mag_driver.set_channel_enabled (ch, enabled);
    (void) light_driver.set_channel_enabled (ch, enabled);
    (void) sound_driver.set_channel_enabled (ch, enabled);
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
    zzs::kernel::Status st = clock_source_install (&zephyr_now_ns);
    if (!zzs::kernel::status_ok (st))
        return st;
    st = accel_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    st = gyro_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    st = mag_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    st = light_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    st = sound_driver.init ();
    if (!zzs::kernel::status_ok (st))
        return st;
    return zzs::kernel::Status::OK;
}

} // namespace

[[nodiscard]] const zzs::kernel::hal::_S_SensorHal *
create_zephyr_sensor_hal (const PlatformConfig &config) noexcept
{
    (void) config;
    hal_vtable._M_read_sample = &hal_read_sample;
    hal_vtable._M_set_channel_enabled = &hal_set_channel_enabled;
    hal_vtable._M_channel_rate_hz = &hal_channel_rate_hz;
    hal_vtable._M_init = &hal_init;
    return &hal_vtable;
}

} // namespace zzs::kernel::sensor::driver::platform::zephyr
