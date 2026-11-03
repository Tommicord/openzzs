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

#ifdef FREERTOS
#include <FreeRTOS.h>
#include <task.h>
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

namespace zzs::kernel::sensor::driver::platform::freertos {

namespace {

/** @brief FreeRTOS I2C bus adapter (vendor I2C driver). */
class FreertosI2cBus : public bus::BusInterface {
  public:
    FreertosI2cBus (void *i2c_handle, uint8_t addr) noexcept : _M_handle (i2c_handle), _M_addr (addr) {
    }
    ~FreertosI2cBus () noexcept = default;
    FreertosI2cBus (const FreertosI2cBus &) = delete;
    FreertosI2cBus &
    operator= (const FreertosI2cBus &) = delete;
    FreertosI2cBus (FreertosI2cBus &&) = delete;
    FreertosI2cBus &
    operator= (FreertosI2cBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
#ifdef FREERTOS
        if (_M_handle == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        /* Vendor-specific I2C write hook; implemented in the BSP layer. */
        extern "C" zzs::kernel::Status freertos_i2c_write (void *handle, uint8_t addr, uint8_t reg, const uint8_t *data,
                                                           uint16_t len) noexcept;
        return freertos_i2c_write (_M_handle, _M_addr, reg, data, len);
#else
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::UNSUPPORTED;
#endif
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override {
#ifdef FREERTOS
        if (_M_handle == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        extern "C" zzs::kernel::Status freertos_i2c_read (void *handle, uint8_t addr, uint8_t reg, uint8_t *data,
                                                          uint16_t len) noexcept;
        return freertos_i2c_read (_M_handle, _M_addr, reg, data, len);
#else
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::UNSUPPORTED;
#endif
    }

    void
    delay_ms (uint32_t ms) noexcept override {
#ifdef FREERTOS
        vTaskDelay (static_cast<TickType_t> (ms) / portTICK_PERIOD_MS);
#else
        (void) ms;
#endif
    }

  private:
    void *_M_handle;
    uint8_t _M_addr;
};

/** @brief FreeRTOS SPI bus adapter (vendor SPI driver). */
class FreertosSpiBus : public bus::BusInterface {
  public:
    FreertosSpiBus (void *spi_handle) noexcept : _M_handle (spi_handle) {
    }
    ~FreertosSpiBus () noexcept = default;
    FreertosSpiBus (const FreertosSpiBus &) = delete;
    FreertosSpiBus &
    operator= (const FreertosSpiBus &) = delete;
    FreertosSpiBus (FreertosSpiBus &&) = delete;
    FreertosSpiBus &
    operator= (FreertosSpiBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
#ifdef FREERTOS
        if (_M_handle == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        extern "C" zzs::kernel::Status freertos_spi_write (void *handle, uint8_t reg, const uint8_t *data,
                                                           uint16_t len) noexcept;
        return freertos_spi_write (_M_handle, reg, data, len);
#else
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::UNSUPPORTED;
#endif
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override {
#ifdef FREERTOS
        if (_M_handle == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        extern "C" zzs::kernel::Status freertos_spi_read (void *handle, uint8_t reg, uint8_t *data,
                                                          uint16_t len) noexcept;
        return freertos_spi_read (_M_handle, reg, data, len);
#else
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::UNSUPPORTED;
#endif
    }

    void
    delay_ms (uint32_t ms) noexcept override {
#ifdef FREERTOS
        vTaskDelay (static_cast<TickType_t> (ms) / portTICK_PERIOD_MS);
#else
        (void) ms;
#endif
    }

  private:
    void *_M_handle;
};

/** @brief Monotonic nanoseconds via xTaskGetTickCountFromISR(). */
[[nodiscard]] uint64_t
_S_freertos_now_ns (void) noexcept {
#ifdef FREERTOS
    TickType_t ticks = xTaskGetTickCountFromISR ();
    return static_cast<uint64_t> (ticks) * 1000000000ull / static_cast<uint64_t> (configTICK_RATE_HZ);
#endif
    return 0ull;
}

static FreertosI2cBus _S_i2c_accel_bus (nullptr, 0x6Au);
static FreertosI2cBus _S_i2c_mag_bus (nullptr, 0x1Cu);
static FreertosI2cBus _S_i2c_light_bus (nullptr, 0x39u);

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
    zzs::kernel::Status st = clock_source_install (&_S_freertos_now_ns);
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
create_freertos_sensor_hal () noexcept {
    _S_hal_vtable._M_read_sample = &_S_hal_read_sample;
    _S_hal_vtable._M_set_channel_enabled = &_S_hal_set_channel_enabled;
    _S_hal_vtable._M_channel_rate_hz = &_S_hal_channel_rate_hz;
    _S_hal_vtable._M_init = &_S_hal_init;
    return zzs::base::Maybe<const zzs::kernel::hal::_S_SensorHal *, zzs::kernel::Status> (&_S_hal_vtable);
}

} // namespace zzs::kernel::sensor::driver::platform::freertos
