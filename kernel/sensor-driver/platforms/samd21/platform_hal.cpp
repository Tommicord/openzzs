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

#ifdef SAMD21
#include "samd21.h"
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

#include <cstdint>

namespace zzs::kernel::sensor::driver::platform::samd21 {

namespace {

/** @brief Atmel SERCOM I2C bus adapter (SERCOM I2CM registers). */
class Samd21I2cBus : public bus::BusInterface {
  public:
    Samd21I2cBus (Sercom *sercom, uint8_t addr) noexcept : _M_sercom (sercom), _M_addr (addr) {
    }
    ~Samd21I2cBus () noexcept = default;
    Samd21I2cBus (const Samd21I2cBus &) = delete;
    Samd21I2cBus &
    operator= (const Samd21I2cBus &) = delete;
    Samd21I2cBus (Samd21I2cBus &&) = delete;
    Samd21I2cBus &
    operator= (Samd21I2cBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
#ifdef SAMD21
        if (_M_sercom == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        /* SERCOM I2CM master write: address + reg + data bytes. */
        SercomI2cm *i2c = &_M_sercom->I2CM;
        uint8_t buf[257];
        buf[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            buf[i + 1u] = data[i];
        while (i2c->STATUS.bit.SYNCBUSY) { /* wait */
        }
        i2c->ADDR.bit.ADDR = (_M_addr << 1) | 0u;
        for (uint16_t i = 0u; i < len + 1u; ++i) {
            while (!i2c->INTFLAG.bit.MB) { /* wait */
            }
            i2c->DATA.bit.DATA = buf[i];
        }
        i2c->CTRLB.bit.CMD = 0x3u; /* stop condition */
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override {
#ifdef SAMD21
        if (_M_sercom == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        SercomI2cm *i2c = &_M_sercom->I2CM;
        while (i2c->STATUS.bit.SYNCBUSY) { /* wait */
        }
        i2c->ADDR.bit.ADDR = (_M_addr << 1) | 0u;
        while (!i2c->INTFLAG.bit.MB) { /* wait */
        }
        i2c->DATA.bit.DATA = reg;
        i2c->CTRLB.bit.CMD = 0x2u; /* repeated start */
        i2c->ADDR.bit.ADDR = (_M_addr << 1) | 1u;
        for (uint16_t i = 0u; i < len; ++i) {
            while (!i2c->INTFLAG.bit.SB) { /* wait */
            }
            data[i] = static_cast<uint8_t> (i2c->DATA.bit.DATA);
        }
        i2c->CTRLB.bit.CMD = 0x3u; /* stop condition */
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    void
    delay_ms (uint32_t ms) noexcept override {
#ifdef SAMD21
        for (uint32_t i = 0u; i < ms * 1000u; ++i)
            __NOP ();
#else
        (void) ms;
#endif
    }

  private:
    Sercom *_M_sercom;
    uint8_t _M_addr;
};

/** @brief Atmel SERCOM SPI bus adapter (SERCOM SPIM registers). */
class Samd21SpiBus : public bus::BusInterface {
  public:
    Samd21SpiBus (Sercom *sercom) noexcept : _M_sercom (sercom) {
    }
    ~Samd21SpiBus () noexcept = default;
    Samd21SpiBus (const Samd21SpiBus &) = delete;
    Samd21SpiBus &
    operator= (const Samd21SpiBus &) = delete;
    Samd21SpiBus (Samd21SpiBus &&) = delete;
    Samd21SpiBus &
    operator= (Samd21SpiBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
#ifdef SAMD21
        if (_M_sercom == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        SercomSpi *spi = &_M_sercom->SPI;
        uint8_t tx[257];
        tx[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            tx[i + 1u] = data[i];
        for (uint16_t i = 0u; i < len + 1u; ++i) {
            while (!spi->INTFLAG.bit.DRE) { /* wait */
            }
            spi->DATA.bit.DATA = tx[i];
        }
        while (!spi->INTFLAG.bit.TXC) {/* wait */
        }
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override {
#ifdef SAMD21
        if (_M_sercom == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        SercomSpi *spi = &_M_sercom->SPI;
        uint8_t tx[257] = {reg};
        uint8_t rx[257] = {0};
        for (uint16_t i = 0u; i < len + 1u; ++i) {
            while (!spi->INTFLAG.bit.DRE) { /* wait */
            }
            spi->DATA.bit.DATA = tx[i];
            while (!spi->INTFLAG.bit.RXC) { /* wait */
            }
            rx[i] = static_cast<uint8_t> (spi->DATA.bit.DATA);
        }
        for (uint16_t i = 0u; i < len; ++i)
            data[i] = rx[i + 1u];
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    void
    delay_ms (uint32_t ms) noexcept override {
#ifdef SAMD21
        for (uint32_t i = 0u; i < ms * 1000u; ++i)
            __NOP ();
#else
        (void) ms;
#endif
    }

  private:
    Sercom *_M_sercom;
};

/** @brief Monotonic nanoseconds via DWT cycle counter. */
[[nodiscard]] uint64_t
samd21_now_ns (void) noexcept {
#ifdef SAMD21
    uint32_t cyccnt = CoreDebug->DEMCR & CoreDebug_DEMCR_TRCENA_Msk ? DWT->CYCCNT : 0u;
    return static_cast<uint64_t> (cyccnt) * 1000000000ull / static_cast<uint64_t> (48000000u);
#endif
    return 0ull;
}

static Samd21I2cBus i2c_accel_bus (nullptr, 0x6Au);
static Samd21I2cBus i2c_mag_bus (nullptr, 0x1Cu);
static Samd21I2cBus i2c_light_bus (nullptr, 0x39u);

static sensors::Lsm6dsoAccelDriver accel_driver (&i2c_accel_bus);
static sensors::Lsm6dsoGyroDriver gyro_driver (&i2c_accel_bus);
static sensors::Lis3mdlMagDriver mag_driver (&i2c_mag_bus);
static sensors::Tsl2561LightDriver light_driver (&i2c_light_bus);
static sensors::Ics43434SoundDriver sound_driver (&i2c_light_bus, nullptr);

static zzs::kernel::hal::_S_SensorHal hal_vtable;

[[nodiscard]] zzs::kernel::Status
hal_read_sample (sensor::Sample &out) noexcept {
    static uint8_t idx = 0u;
    zzs::kernel::Status st = zzs::kernel::Status::STALE;
    for (uint8_t tries = 0u; tries < 5u; ++tries) {
        switch (idx) {
        case 0u:
            st = accel_driver.read_sample (out);
            break;
        case 1u:
            st = gyro_driver.read_sample (out);
            break;
        case 2u:
            st = mag_driver.read_sample (out);
            break;
        case 3u:
            st = light_driver.read_sample (out);
            break;
        default:
            st = sound_driver.read_sample (out);
            break;
        }
        idx = static_cast<uint8_t> ((idx + 1u) % 5u);
        if (zzs::kernel::status_ok (st))
            return zzs::kernel::Status::OK;
    }
    return st;
}

[[nodiscard]] zzs::kernel::Status
hal_set_channel_enabled (sensor::Channel ch, bool enabled) noexcept {
    (void) accel_driver.set_channel_enabled (ch, enabled);
    (void) gyro_driver.set_channel_enabled (ch, enabled);
    (void) mag_driver.set_channel_enabled (ch, enabled);
    (void) light_driver.set_channel_enabled (ch, enabled);
    (void) sound_driver.set_channel_enabled (ch, enabled);
    return zzs::kernel::Status::OK;
}

[[nodiscard]] uint32_t
hal_channel_rate_hz (sensor::Channel ch) noexcept {
    return sensor::channel_rate_hz (ch);
}

[[nodiscard]] zzs::kernel::Status
hal_init (void) noexcept {
    zzs::kernel::Status st = clock_source_install (&samd21_now_ns);
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
create_samd21_sensor_hal () noexcept {
    hal_vtable._M_read_sample = &hal_read_sample;
    hal_vtable._M_set_channel_enabled = &hal_set_channel_enabled;
    hal_vtable._M_channel_rate_hz = &hal_channel_rate_hz;
    hal_vtable._M_init = &hal_init;
    return &hal_vtable;
}

} // namespace zzs::kernel::sensor::driver::platform::samd21
