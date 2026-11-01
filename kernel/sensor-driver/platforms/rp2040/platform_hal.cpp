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

#ifdef PICO_RP2040
#  include "pico/stdlib.h"
#  include "hardware/i2c.h"
#  include "hardware/spi.h"
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

namespace zzs::kernel::sensor::driver::platform::rp2040
{

namespace {

/** @brief RP2040 SDK I2C bus adapter (i2c_write_blocking / i2c_read_blocking). */
class Rp2040I2cBus : public bus::BusInterface
{
public:
    Rp2040I2cBus (i2c_inst_t *inst, uint8_t addr) noexcept
        : _M_inst (inst), _M_addr (addr) {}
    ~Rp2040I2cBus () noexcept = default;
    Rp2040I2cBus (const Rp2040I2cBus &) = delete;
    Rp2040I2cBus &operator= (const Rp2040I2cBus &) = delete;
    Rp2040I2cBus (Rp2040I2cBus &&) = delete;
    Rp2040I2cBus &operator= (Rp2040I2cBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override
    {
#ifdef PICO_RP2040
        if (_M_inst == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t buf[257];
        buf[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            buf[i + 1u] = data[i];
        if (i2c_write_blocking (_M_inst, _M_addr, buf, static_cast<size_t> (len + 1u),
                                false) < 0)
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
#ifdef PICO_RP2040
        if (_M_inst == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        if (i2c_write_blocking (_M_inst, _M_addr, &reg, 1u, true) < 0)
            return zzs::kernel::Status::INTERNAL;
        if (i2c_read_blocking (_M_inst, _M_addr, data, static_cast<size_t> (len),
                               false) < 0)
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
#ifdef PICO_RP2040
        sleep_ms (ms);
#else
        (void) ms;
#endif
    }

private:
    i2c_inst_t *_M_inst;
    uint8_t _M_addr;
};

/** @brief RP2040 SDK SPI bus adapter (spi_write_blocking / spi_read_blocking). */
class Rp2040SpiBus : public bus::BusInterface
{
public:
    Rp2040SpiBus (spi_inst_t *inst) noexcept : _M_inst (inst) {}
    ~Rp2040SpiBus () noexcept = default;
    Rp2040SpiBus (const Rp2040SpiBus &) = delete;
    Rp2040SpiBus &operator= (const Rp2040SpiBus &) = delete;
    Rp2040SpiBus (Rp2040SpiBus &&) = delete;
    Rp2040SpiBus &operator= (Rp2040SpiBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override
    {
#ifdef PICO_RP2040
        if (_M_inst == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t buf[257];
        buf[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            buf[i + 1u] = data[i];
        spi_write_blocking (_M_inst, buf, static_cast<size_t> (len + 1u));
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override
    {
#ifdef PICO_RP2040
        if (_M_inst == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t tx[257] = { reg };
        uint8_t rx[257] = { 0 };
        spi_write_read_blocking (_M_inst, tx, rx, static_cast<size_t> (len + 1u));
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
#ifdef PICO_RP2040
        sleep_ms (ms);
#else
        (void) ms;
#endif
    }

private:
    spi_inst_t *_M_inst;
};

/** @brief Monotonic nanoseconds via to_us_since_boot(). */
[[nodiscard]] uint64_t
rp2040_now_ns (void) noexcept
{
#ifdef PICO_RP2040
    return static_cast<uint64_t> (to_us_since_boot ()) * 1000ull;
#endif
    return 0ull;
}

static Rp2040I2cBus i2c_accel_bus (nullptr, 0x6Au);
static Rp2040I2cBus i2c_mag_bus (nullptr, 0x1Cu);
static Rp2040I2cBus i2c_light_bus (nullptr, 0x39u);

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
    zzs::kernel::Status st = clock_source_install (&rp2040_now_ns);
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
create_rp2040_sensor_hal () noexcept
{
    hal_vtable._M_read_sample = &hal_read_sample;
    hal_vtable._M_set_channel_enabled = &hal_set_channel_enabled;
    hal_vtable._M_channel_rate_hz = &hal_channel_rate_hz;
    hal_vtable._M_init = &hal_init;
    return &hal_vtable;
}

} // namespace zzs::kernel::sensor::driver::platform::rp2040
