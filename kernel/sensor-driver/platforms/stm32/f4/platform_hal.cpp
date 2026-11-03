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

#ifdef STM32F4
#include "stm32f4xx_hal.h"
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

#include <cstdint>

namespace zzs::kernel::sensor::driver::platform::stm32::f4 {

namespace {

/** @brief STM32Cube HAL I2C bus adapter (HAL_I2C_Mem_Write/Read). */
class Stm32I2cBus : public bus::BusInterface {
  public:
    Stm32I2cBus (I2C_HandleTypeDef *hi2c, uint16_t dev_addr) noexcept
        : _M_hi2c (hi2c), _M_dev_addr (dev_addr), _M_timeout (100u) {
    }

    ~Stm32I2cBus () noexcept = default;

    Stm32I2cBus (const Stm32I2cBus &) = delete;
    Stm32I2cBus &
    operator= (const Stm32I2cBus &) = delete;
    Stm32I2cBus (Stm32I2cBus &&) = delete;
    Stm32I2cBus &
    operator= (Stm32I2cBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
#ifdef STM32F4
        if (_M_hi2c == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        if (HAL_I2C_Mem_Write (_M_hi2c, _M_dev_addr, reg, I2C_MEMADD_SIZE_8BIT, const_cast<uint8_t *> (data), len,
                               _M_timeout)
            != HAL_OK)
            return zzs::kernel::Status::INTERNAL;
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override {
#ifdef STM32F4
        if (_M_hi2c == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        if (HAL_I2C_Mem_Read (_M_hi2c, _M_dev_addr, reg, I2C_MEMADD_SIZE_8BIT, data, len, _M_timeout) != HAL_OK)
            return zzs::kernel::Status::INTERNAL;
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    void
    delay_ms (uint32_t ms) noexcept override {
#ifdef STM32F4
        HAL_Delay (ms);
#else
        (void) ms;
#endif
    }

    void
    set_handle (I2C_HandleTypeDef *hi2c) noexcept {
        _M_hi2c = hi2c;
    }

  private:
    I2C_HandleTypeDef *_M_hi2c;
    uint16_t _M_dev_addr;
    uint32_t _M_timeout;
};

/** @brief STM32Cube HAL SPI bus adapter (HAL_SPI_TransmitReceive). */
class Stm32SpiBus : public bus::BusInterface {
  public:
    Stm32SpiBus (SPI_HandleTypeDef *hspi, GPIO_TypeDef *cs_port, uint16_t cs_pin) noexcept
        : _M_hspi (hspi), _M_cs_port (cs_port), _M_cs_pin (cs_pin), _M_timeout (100u) {
    }

    ~Stm32SpiBus () noexcept = default;

    Stm32SpiBus (const Stm32SpiBus &) = delete;
    Stm32SpiBus &
    operator= (const Stm32SpiBus &) = delete;
    Stm32SpiBus (Stm32SpiBus &&) = delete;
    Stm32SpiBus &
    operator= (Stm32SpiBus &&) = delete;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override {
#ifdef STM32F4
        if (_M_hspi == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        HAL_GPIO_WritePin (_M_cs_port, _M_cs_pin, GPIO_PIN_RESET);
        uint8_t tx[257];
        tx[0] = reg;
        for (uint16_t i = 0u; i < len; ++i)
            tx[i + 1u] = data[i];
        if (HAL_SPI_Transmit (_M_hspi, tx, len + 1u, _M_timeout) != HAL_OK) {
            HAL_GPIO_WritePin (_M_cs_port, _M_cs_pin, GPIO_PIN_SET);
            return zzs::kernel::Status::INTERNAL;
        }
        HAL_GPIO_WritePin (_M_cs_port, _M_cs_pin, GPIO_PIN_SET);
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override {
#ifdef STM32F4
        if (_M_hspi == nullptr || data == nullptr)
            return zzs::kernel::Status::INTERNAL;
        HAL_GPIO_WritePin (_M_cs_port, _M_cs_pin, GPIO_PIN_RESET);
        uint8_t tx[257] = {reg};
        uint8_t rx[257] = {0};
        if (HAL_SPI_TransmitReceive (_M_hspi, tx, rx, len + 1u, _M_timeout) != HAL_OK) {
            HAL_GPIO_WritePin (_M_cs_port, _M_cs_pin, GPIO_PIN_SET);
            return zzs::kernel::Status::INTERNAL;
        }
        for (uint16_t i = 0u; i < len; ++i)
            data[i] = rx[i + 1u];
        HAL_GPIO_WritePin (_M_cs_port, _M_cs_pin, GPIO_PIN_SET);
#endif
        (void) reg;
        (void) data;
        (void) len;
        return zzs::kernel::Status::OK;
    }

    void
    delay_ms (uint32_t ms) noexcept override {
#ifdef STM32F4
        HAL_Delay (ms);
#else
        (void) ms;
#endif
    }

  private:
    SPI_HandleTypeDef *_M_hspi;
    GPIO_TypeDef *_M_cs_port;
    uint16_t _M_cs_pin;
    uint32_t _M_timeout;
};

/** @brief Monotonic nanoseconds via DWT cycle counter. */
[[nodiscard]] uint64_t
stm32f4_now_ns (void) noexcept {
#ifdef STM32F4
    /* DWT_CYCCNT is a 32-bit free-running counter; combine with a software
     * overflow latch for 64-bit nanoseconds. */
    uint32_t cyccnt = DWT->CYCCNT;
    static uint32_t s_last = 0u;
    static uint64_t s_overflows = 0ull;
    if (cyccnt < s_last)
        ++s_overflows;
    s_last = cyccnt;
    const uint64_t cycles = (s_overflows << 32) | static_cast<uint64_t> (cyccnt);
    /* Approximate: SystemCoreClock / 1e9 ns per cycle. */
    return cycles * 1000000000ull / static_cast<uint64_t> (SystemCoreClock);
#endif
    return 0ull;
}

static Stm32I2cBus i2c_accel_bus (nullptr, 0x6Au); // LSM6DSO
static Stm32I2cBus i2c_mag_bus (nullptr, 0x1Cu);   // LIS3MDL
static Stm32I2cBus i2c_light_bus (nullptr, 0x39u); // TSL2561 / VEML7700

static sensors::Lsm6dsoAccelDriver accel_driver (&i2c_accel_bus);
static sensors::Lsm6dsoGyroDriver gyro_driver (&i2c_accel_bus);
static sensors::Lis3mdlMagDriver mag_driver (&i2c_mag_bus);
static sensors::Tsl2561LightDriver light_driver (&i2c_light_bus);
static sensors::Ics43434SoundDriver sound_driver (&i2c_light_bus, nullptr);

static zzs::kernel::hal::_S_SensorHal hal_vtable;

/** @brief Shared scan ring: drivers burst-fill it via read_samples(), the HAL
 *         drains it one sample per hal_read_sample() call.  Fixed 16-sample
 *         depth, drop-oldest when full, no heap. */
static stream::ScanRing hal_ring (16u);

/** @brief Burst-fill the shared scan ring from one driver using read_samples().
 *         Returns OK even when the driver reports STALE (no new data). */
[[nodiscard]] zzs::kernel::Status
hal_fill_ring (sensors::SensorDriver *driver) noexcept {
    if (driver == nullptr)
        return zzs::kernel::Status::OK;
    sensor::Sample buf[8u];
    uint16_t filled = 0u;
    zzs::kernel::Status st = driver->read_samples (buf, 8u, &filled);
    if (!zzs::kernel::status_ok (st) && st != zzs::kernel::Status::STALE)
        return st;
    for (uint16_t i = 0u; i < filled; ++i)
        hal_ring.push (buf[i]);
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
hal_read_sample (sensor::Sample &out) noexcept {
    /* Drain from the shared scan ring first (burst-filled previously). */
    if (hal_ring.pop (out) == zzs::kernel::Status::OK)
        return zzs::kernel::Status::OK;
    /* Ring empty: refill by bursting each enabled driver in round-robin order. */
    static uint8_t idx = 0u;
    for (uint8_t tries = 0u; tries < 5u; ++tries) {
        switch (idx) {
        case 0u:
            (void) hal_fill_ring (&accel_driver);
            break;
        case 1u:
            (void) hal_fill_ring (&gyro_driver);
            break;
        case 2u:
            (void) hal_fill_ring (&mag_driver);
            break;
        case 3u:
            (void) hal_fill_ring (&light_driver);
            break;
        default:
            (void) hal_fill_ring (&sound_driver);
            break;
        }
        idx = static_cast<uint8_t> ((idx + 1u) % 5u);
        if (hal_ring.filled () > 0u)
            break;
    }
    return hal_ring.pop (out);
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
    zzs::kernel::Status st = clock_source_install (&stm32f4_now_ns);
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
create_stm32f4_sensor_hal () noexcept {
    hal_vtable._M_read_sample = &hal_read_sample;
    hal_vtable._M_set_channel_enabled = &hal_set_channel_enabled;
    hal_vtable._M_channel_rate_hz = &hal_channel_rate_hz;
    hal_vtable._M_init = &hal_init;
    return &hal_vtable;
}

} // namespace zzs::kernel::sensor::driver::platform::stm32::f4
