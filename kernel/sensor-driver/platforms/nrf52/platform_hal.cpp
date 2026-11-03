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

#ifdef NRF52
#include <nrf_drv_spi.h>
#include <nrf_drv_twi.h>
#include <nrfx.h>
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
#include "std/maybe.hpp"

#include <cstdint>

namespace zzs::kernel::sensor::driver::platform::nrf52 {

namespace {

/** @brief Nordic SDK TWIM bus adapter (nrf_drv_twi_tx/rx). */
class Nrf52I2cBus : public bus::BusInterface {
  public:
    Nrf52I2cBus (const nrf_drv_twi_t *instance, uint8_t addr) noexcept : _M_instance (instance), _M_addr (addr) {
    }
    ~Nrf52I2cBus () noexcept = default;
    Nrf52I2cBus (const Nrf52I2cBus &) = delete;
    Nrf52I2cBus &
    operator= (const Nrf52I2cBus &) = delete;
    Nrf52I2cBus (Nrf52I2cBus &&) = delete;
    Nrf52I2cBus &
    operator= (Nrf52I2cBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
#ifdef NRF52
        if (_M_instance == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t buf[257];
        buf[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            buf[i + 1u] = data[i];
        if (nrf_drv_twi_tx (_M_instance, _M_addr, buf, static_cast<uint8_t> (len + 1u), false) != NRF_SUCCESS)
            return zzs::kernel::Status::INTERNAL;
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override {
#ifdef NRF52
        if (_M_instance == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        if (nrf_drv_twi_tx (_M_instance, _M_addr, &reg, 1u, true) != NRF_SUCCESS)
            return zzs::kernel::Status::INTERNAL;
        if (nrf_drv_twi_rx (_M_instance, _M_addr, data, static_cast<uint8_t> (len)) != NRF_SUCCESS)
            return zzs::kernel::Status::INTERNAL;
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    void
    delay_ms (uint32_t ms) noexcept override {
#ifdef NRF52
        nrf_delay_ms (ms);
#else
        (void) ms;
#endif
    }

  private:
    const nrf_drv_twi_t *_M_instance;
    uint8_t _M_addr;
};

/** @brief Nordic SDK SPIM bus adapter (nrf_drv_spi_transfer). */
class Nrf52SpiBus : public bus::BusInterface {
  public:
    Nrf52SpiBus (const nrf_drv_spi_t *instance) noexcept : _M_instance (instance) {
    }
    ~Nrf52SpiBus () noexcept = default;
    Nrf52SpiBus (const Nrf52SpiBus &) = delete;
    Nrf52SpiBus &
    operator= (const Nrf52SpiBus &) = delete;
    Nrf52SpiBus (Nrf52SpiBus &&) = delete;
    Nrf52SpiBus &
    operator= (Nrf52SpiBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
#ifdef NRF52
        if (_M_instance == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t tx[257];
        tx[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            tx[i + 1u] = data[i];
        nrf_drv_spi_xfer_desc_t xfer = NRF_DRV_SPI_XFER_DESC_TX (tx, static_cast<uint8_t> (len + 1u));
        if (nrf_drv_spi_transfer (_M_instance, &xfer, 0) != NRF_SUCCESS)
            return zzs::kernel::Status::INTERNAL;
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override {
#ifdef NRF52
        if (_M_instance == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        uint8_t tx[257] = {reg};
        uint8_t rx[257] = {0};
        nrf_drv_spi_xfer_desc_t xfer =
            NRF_DRV_SPI_XFER_DESC_TRX (tx, static_cast<uint8_t> (len + 1u), rx, static_cast<uint8_t> (len + 1u));
        if (nrf_drv_spi_transfer (_M_instance, &xfer, 0) != NRF_SUCCESS)
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
    delay_ms (uint32_t ms) noexcept override {
#ifdef NRF52
        nrf_delay_ms (ms);
#else
        (void) ms;
#endif
    }

  private:
    const nrf_drv_spi_t *_M_instance;
};

/** @brief Monotonic nanoseconds via RTC/TIMER counter. */
[[nodiscard]] uint64_t
_S_nrf52_now_ns (void) noexcept {
#ifdef NRF52
    uint32_t ticks = NRF_RTC1->COUNTER;
    return static_cast<uint64_t> (ticks) * 30517ull / 1000000ull;
#endif
    return 0ull;
}

static Nrf52I2cBus _S_i2c_accel_bus (nullptr, 0x6Au);
static Nrf52I2cBus _S_i2c_mag_bus (nullptr, 0x1Cu);
static Nrf52I2cBus _S_i2c_light_bus (nullptr, 0x39u);

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
    zzs::kernel::Status st = clock_source_install (&_S_nrf52_now_ns);
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
create_nrf52_sensor_hal () noexcept {
    _S_hal_vtable._M_read_sample = &_S_hal_read_sample;
    _S_hal_vtable._M_set_channel_enabled = &_S_hal_set_channel_enabled;
    _S_hal_vtable._M_channel_rate_hz = &_S_hal_channel_rate_hz;
    _S_hal_vtable._M_init = &_S_hal_init;
    return &_S_hal_vtable;
}

} // namespace zzs::kernel::sensor::driver::platform::nrf52
