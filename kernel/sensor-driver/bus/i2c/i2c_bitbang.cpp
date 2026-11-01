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

#include "kernel/sensor-driver/bus/i2c/i2c_bitbang.hpp"

#include <cstdint>

/**
 * @brief I2C bit-bang primitives.
 *
 * When the assembly library is built (ZZS_SENSOR_DRIVER_BUS_ASM_ENABLED),
 * these are declared as extern "C" and linked from bus/asm/.  Otherwise the
 * portable C++ fallback in i2c_bitbang_cpp.cpp provides the same symbols.
 */
#if defined(ZZS_SENSOR_DRIVER_BUS_ASM_ENABLED)
#if defined(__x86_64__) || defined(__i386__)
extern "C" {
    void bus_gpio_set (uint32_t gpio_base, uint32_t pin, uint8_t value) noexcept;
    uint8_t bus_gpio_get (uint32_t gpio_base, uint32_t pin) noexcept;
    void bus_i2c_delay_us (uint32_t us) noexcept;
    void bus_i2c_start (uint32_t gpio_base, uint32_t sda_pin, uint32_t scl_pin) noexcept;
    void bus_i2c_stop (uint32_t gpio_base, uint32_t sda_pin, uint32_t scl_pin) noexcept;
    uint8_t bus_i2c_write_bit (uint32_t gpio_base, uint32_t sda_pin,
                               uint32_t scl_pin, uint8_t bit) noexcept;
    uint8_t bus_i2c_read_bit (uint32_t gpio_base, uint32_t sda_pin,
                              uint32_t scl_pin) noexcept;
    uint8_t bus_i2c_write_byte (uint32_t gpio_base, uint32_t sda_pin,
                                uint32_t scl_pin, uint8_t byte) noexcept;
    uint8_t bus_i2c_read_byte (uint32_t gpio_base, uint32_t sda_pin,
                               uint32_t scl_pin, uint8_t ack) noexcept;
}
#elif defined(__arm__) || defined(__aarch64__)
extern "C" {
    void bus_gpio_set (uint32_t gpio_base, uint32_t pin, uint8_t value) noexcept;
    uint8_t bus_gpio_get (uint32_t gpio_base, uint32_t pin) noexcept;
    void bus_i2c_delay_us (uint32_t us) noexcept;
    void bus_i2c_start (uint32_t gpio_base, uint32_t sda_pin, uint32_t scl_pin) noexcept;
    void bus_i2c_stop (uint32_t gpio_base, uint32_t sda_pin, uint32_t scl_pin) noexcept;
    uint8_t bus_i2c_write_bit (uint32_t gpio_base, uint32_t sda_pin,
                               uint32_t scl_pin, uint8_t bit) noexcept;
    uint8_t bus_i2c_read_bit (uint32_t gpio_base, uint32_t sda_pin,
                              uint32_t scl_pin) noexcept;
    uint8_t bus_i2c_write_byte (uint32_t gpio_base, uint32_t sda_pin,
                                uint32_t scl_pin, uint8_t byte) noexcept;
    uint8_t bus_i2c_read_byte (uint32_t gpio_base, uint32_t sda_pin,
                               uint32_t scl_pin, uint8_t ack) noexcept;
}
#else
#error "ZZS_SENSOR_DRIVER_BUS_ASM_ENABLED but no assembly routines for this target"
#endif
#else
/* C++ fallback: the symbols are defined in i2c_bitbang_cpp.cpp. */
extern "C" {
    void bus_gpio_set (uint32_t gpio_base, uint32_t pin, uint8_t value) noexcept;
    uint8_t bus_gpio_get (uint32_t gpio_base, uint32_t pin) noexcept;
    void bus_i2c_delay_us (uint32_t us) noexcept;
    void bus_i2c_start (uint32_t gpio_base, uint32_t sda_pin, uint32_t scl_pin) noexcept;
    void bus_i2c_stop (uint32_t gpio_base, uint32_t sda_pin, uint32_t scl_pin) noexcept;
    uint8_t bus_i2c_write_bit (uint32_t gpio_base, uint32_t sda_pin,
                               uint32_t scl_pin, uint8_t bit) noexcept;
    uint8_t bus_i2c_read_bit (uint32_t gpio_base, uint32_t sda_pin,
                              uint32_t scl_pin) noexcept;
    uint8_t bus_i2c_write_byte (uint32_t gpio_base, uint32_t sda_pin,
                                uint32_t scl_pin, uint8_t byte) noexcept;
    uint8_t bus_i2c_read_byte (uint32_t gpio_base, uint32_t sda_pin,
                               uint32_t scl_pin, uint8_t ack) noexcept;
}
#endif

static_assert (sizeof (zzs::kernel::sensor::driver::bus::I2cConfig) <= 64u,
               "I2cConfig must fit within 64 bytes");
static_assert (alignof (zzs::kernel::sensor::driver::bus::I2cConfig) == 4u,
               "I2cConfig alignment must be 4 bytes");

namespace zzs::kernel::sensor::driver::bus
{

I2cBitBang::I2cBitBang (const I2cConfig &config) noexcept
    : _M_config (config)
{
}

[[nodiscard]] Status
I2cBitBang::write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept
{
    if (data == nullptr && len > 0u)
        return Status::OUT_OF_RANGE;
    Status st = start ();
    if (st != Status::OK)
        return st;
    st = write_addr7 (_M_config.device_addr, false);
    if (st != Status::OK)
        { stop (); return st; }
    st = write_byte (reg);
    if (st != Status::OK)
        { stop (); return st; }
    for (uint16_t i = 0u; i < len; ++i)
        {
        st = write_byte (data[i]);
        if (st != Status::OK)
            { stop (); return st; }
        }
    stop ();
    return Status::OK;
}

[[nodiscard]] Status
I2cBitBang::read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept
{
    if (data == nullptr && len > 0u)
        return Status::OUT_OF_RANGE;
    /* Write register address, then repeated-start + read. */
    Status st = start ();
    if (st != Status::OK)
        return st;
    st = write_addr7 (_M_config.device_addr, false);
    if (st != Status::OK)
        { stop (); return st; }
    st = write_byte (reg);
    if (st != Status::OK)
        { stop (); return st; }
    /* Repeated start, read direction */
    st = start ();
    if (st != Status::OK)
        { stop (); return st; }
    st = write_addr7 (_M_config.device_addr, true);
    if (st != Status::OK)
        { stop (); return st; }
    for (uint16_t i = 0u; i < len; ++i)
        {
        bool last = (i == len - 1u);
        st = read_byte (&data[i], last);
        if (st != Status::OK)
            { stop (); return st; }
        }
    stop ();
    return Status::OK;
}

void
I2cBitBang::delay_ms (uint32_t ms) noexcept
{
    while (ms-- > 0u)
        bus_i2c_delay_us (1000u);
}

[[nodiscard]] Status
I2cBitBang::start () noexcept
{
    bus_i2c_start (_M_config.gpio_base, _M_config.sda_pin, _M_config.scl_pin);
    return Status::OK;
}

[[nodiscard]] Status
I2cBitBang::stop () noexcept
{
    bus_i2c_stop (_M_config.gpio_base, _M_config.sda_pin, _M_config.scl_pin);
    return Status::OK;
}

[[nodiscard]] Status
I2cBitBang::write (uint8_t addr, const uint8_t *data, uint16_t len,
                   bool repeated_start) noexcept
{
    if (data == nullptr && len > 0u)
        return Status::OUT_OF_RANGE;
    if (!repeated_start)
        {
        Status st = start ();
        if (st != Status::OK)
            return st;
        }
    Status st = write_addr7 (addr, false);
    if (st != Status::OK)
        { stop (); return st; }
    for (uint16_t i = 0u; i < len; ++i)
        {
        st = write_byte (data[i]);
        if (st != Status::OK)
            { stop (); return st; }
        }
    if (!repeated_start)
        stop ();
    return Status::OK;
}

[[nodiscard]] Status
I2cBitBang::read (uint8_t addr, uint8_t *data, uint16_t len,
                  bool repeated_start) noexcept
{
    if (data == nullptr && len > 0u)
        return Status::OUT_OF_RANGE;
    if (!repeated_start)
        {
        Status st = start ();
        if (st != Status::OK)
            return st;
        }
    Status st = write_addr7 (addr, true);
    if (st != Status::OK)
        { stop (); return st; }
    for (uint16_t i = 0u; i < len; ++i)
        {
        bool last = (i == len - 1u);
        st = read_byte (&data[i], last);
        if (st != Status::OK)
            { stop (); return st; }
        }
    if (!repeated_start)
        stop ();
    return Status::OK;
}

[[nodiscard]] Status
I2cBitBang::write_byte (uint8_t byte) noexcept
{
    uint8_t ack = bus_i2c_write_byte (_M_config.gpio_base,
                                      _M_config.sda_pin,
                                      _M_config.scl_pin,
                                      byte);
    if (ack != 0u)
        return Status::INTERNAL;
    bit_delay ();
    return Status::OK;
}

[[nodiscard]] Status
I2cBitBang::read_byte (uint8_t *byte, bool last) noexcept
{
    uint8_t ack = last ? 1u : 0u;   /* NACK on last byte, ACK otherwise */
    uint8_t val = bus_i2c_read_byte (_M_config.gpio_base,
                                     _M_config.sda_pin,
                                     _M_config.scl_pin,
                                     ack);
    *byte = val;
    bit_delay ();
    return Status::OK;
}

[[nodiscard]] Status
I2cBitBang::write_addr7 (uint8_t addr, bool write) noexcept
{
    /* 7-bit address: (addr << 1) | (write ? 0 : 1) */
    uint8_t byte = static_cast<uint8_t> ((addr << 1u) | (write ? 0u : 1u));
    return write_byte (byte);
}

[[nodiscard]] Status
I2cBitBang::write_addr10 (uint8_t addr_high, uint8_t addr_low, bool write) noexcept
{
    /* 10-bit addressing: 11110xx0 (write) then addr_low */
    uint8_t first = static_cast<uint8_t> (0xF0u | ((addr_high & 0x03u) << 1u) | (write ? 0u : 1u));
    Status st = write_byte (first);
    if (st != Status::OK)
        return st;
    return write_byte (addr_low);
}

void
I2cBitBang::bit_delay () noexcept
{
    /* Half-clock-period delay between bits based on configured clock speed. */
    uint32_t us = 0u;
    switch (_M_config.clock)
        {
        case I2cClockHz::k100kHz:
            us = 5u;   /* 100 kHz -> 10 us period, ~5 us half-period */
            break;
        case I2cClockHz::k400kHz:
            us = 2u;   /* 400 kHz -> 2.5 us period */
            break;
        case I2cClockHz::k1MHz:
            us = 1u;   /* 1 MHz -> 1 us period */
            break;
        }
    bus_i2c_delay_us (us);
}

} // namespace zzs::kernel::sensor::driver::bus
