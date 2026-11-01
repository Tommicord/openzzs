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

#include "kernel/sensor-driver/bus/spi/spi_bitbang.hpp"

#include <cstdint>

/**
 * @brief Assembly bit-bang primitives (defined in bus/asm/).
 *
 * Declared per-architecture; only the matching target's object file is linked.
 * bus_gpio_set / bus_gpio_get are defined in the I2C assembly file and
 * shared with SPI.
 */
#if defined(__x86_64__) || defined(__i386__)
extern "C" {
    void bus_gpio_set (uint32_t gpio_base, uint32_t pin, uint8_t value) noexcept;
    uint8_t bus_gpio_get (uint32_t gpio_base, uint32_t pin) noexcept;
    void bus_spi_delay_us (uint32_t us) noexcept;
    void bus_spi_cs_assert (uint32_t gpio_base, uint32_t cs_pin) noexcept;
    void bus_spi_cs_deassert (uint32_t gpio_base, uint32_t cs_pin) noexcept;
    void bus_spi_write_bit (uint32_t gpio_base, uint32_t mosi_pin,
                            uint32_t sclk_pin, uint8_t bit) noexcept;
    uint8_t bus_spi_read_bit (uint32_t gpio_base, uint32_t miso_pin,
                              uint32_t sclk_pin) noexcept;
    uint8_t bus_spi_transfer_byte (uint32_t gpio_base, uint32_t mosi_pin,
                                   uint32_t miso_pin, uint32_t sclk_pin,
                                   uint8_t byte) noexcept;
}
#elif defined(__arm__) || defined(__aarch64__)
extern "C" {
    void bus_gpio_set (uint32_t gpio_base, uint32_t pin, uint8_t value) noexcept;
    uint8_t bus_gpio_get (uint32_t gpio_base, uint32_t pin) noexcept;
    void bus_spi_delay_us (uint32_t us) noexcept;
    void bus_spi_cs_assert (uint32_t gpio_base, uint32_t cs_pin) noexcept;
    void bus_spi_cs_deassert (uint32_t gpio_base, uint32_t cs_pin) noexcept;
    void bus_spi_write_bit (uint32_t gpio_base, uint32_t mosi_pin,
                            uint32_t sclk_pin, uint8_t bit) noexcept;
    uint8_t bus_spi_read_bit (uint32_t gpio_base, uint32_t miso_pin,
                              uint32_t sclk_pin) noexcept;
    uint8_t bus_spi_transfer_byte (uint32_t gpio_base, uint32_t mosi_pin,
                                   uint32_t miso_pin, uint32_t sclk_pin,
                                   uint8_t byte) noexcept;
}
#else
#error "No assembly bit-bang routines available for this target"
#endif

static_assert (sizeof (zzs::kernel::sensor::driver::bus::SpiConfig) <= 64u,
               "SpiConfig must fit within 64 bytes");
static_assert (alignof (zzs::kernel::sensor::driver::bus::SpiConfig) == 4u,
               "SpiConfig alignment must be 4 bytes");

namespace zzs::kernel::sensor::driver::bus
{

SpiBitBang::SpiBitBang (const SpiConfig &config) noexcept
    : _M_config (config)
{
}

[[nodiscard]] Status
SpiBitBang::write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept
{
    if (data == nullptr && len > 0u)
        return Status::OUT_OF_RANGE;
    Status st = cs_assert ();
    if (st != Status::OK)
        return st;
    /* SPI devices typically send register address first, then data. */
    st = transfer_byte (reg, nullptr);
    if (st != Status::OK)
        { cs_deassert (); return st; }
    for (uint16_t i = 0u; i < len; ++i)
        {
        st = transfer_byte (data[i], nullptr);
        if (st != Status::OK)
            { cs_deassert (); return st; }
        }
    cs_deassert ();
    return Status::OK;
}

[[nodiscard]] Status
SpiBitBang::read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept
{
    if (data == nullptr && len > 0u)
        return Status::OUT_OF_RANGE;
    Status st = cs_assert ();
    if (st != Status::OK)
        return st;
    /* Send register address (with read bit set if the device uses it). */
    uint8_t addr = static_cast<uint8_t> (reg | 0x80u);
    st = transfer_byte (addr, nullptr);
    if (st != Status::OK)
        { cs_deassert (); return st; }
    for (uint16_t i = 0u; i < len; ++i)
        {
        uint8_t rx = 0u;
        st = transfer_byte (0x00u, &rx);
        if (st != Status::OK)
            { cs_deassert (); return st; }
        data[i] = rx;
        }
    cs_deassert ();
    return Status::OK;
}

void
SpiBitBang::delay_ms (uint32_t ms) noexcept
{
    while (ms-- > 0u)
        bus_spi_delay_us (1000u);
}

[[nodiscard]] Status
SpiBitBang::cs_assert () noexcept
{
    bus_spi_cs_assert (_M_config.gpio_base, _M_config.cs_pin);
    return Status::OK;
}

[[nodiscard]] Status
SpiBitBang::cs_deassert () noexcept
{
    bus_spi_cs_deassert (_M_config.gpio_base, _M_config.cs_pin);
    return Status::OK;
}

[[nodiscard]] Status
SpiBitBang::transfer (const uint8_t *tx, uint8_t *rx, uint16_t len) noexcept
{
    if ((tx == nullptr && rx == nullptr) || len == 0u)
        return Status::OUT_OF_RANGE;
    Status st = cs_assert ();
    if (st != Status::OK)
        return st;
    for (uint16_t i = 0u; i < len; ++i)
        {
        uint8_t tx_byte = (tx != nullptr) ? tx[i] : 0x00u;
        uint8_t rx_byte = 0u;
        st = transfer_byte (tx_byte, &rx_byte);
        if (st != Status::OK)
            { cs_deassert (); return st; }
        if (rx != nullptr)
            rx[i] = rx_byte;
        }
    cs_deassert ();
    return Status::OK;
}

[[nodiscard]] Status
SpiBitBang::write (const uint8_t *data, uint16_t len) noexcept
{
    if (data == nullptr && len > 0u)
        return Status::OUT_OF_RANGE;
    Status st = cs_assert ();
    if (st != Status::OK)
        return st;
    for (uint16_t i = 0u; i < len; ++i)
        {
        st = transfer_byte (data[i], nullptr);
        if (st != Status::OK)
            { cs_deassert (); return st; }
        }
    cs_deassert ();
    return Status::OK;
}

[[nodiscard]] Status
SpiBitBang::read (uint8_t *data, uint16_t len) noexcept
{
    if (data == nullptr && len > 0u)
        return Status::OUT_OF_RANGE;
    Status st = cs_assert ();
    if (st != Status::OK)
        return st;
    for (uint16_t i = 0u; i < len; ++i)
        {
        uint8_t rx = 0u;
        st = transfer_byte (0x00u, &rx);
        if (st != Status::OK)
            { cs_deassert (); return st; }
        data[i] = rx;
        }
    cs_deassert ();
    return Status::OK;
}

[[nodiscard]] Status
SpiBitBang::transfer_byte (uint8_t tx, uint8_t *rx) noexcept
{
    if (_M_config.mode == SpiMode::Mode0)
        {
        /* Mode 0 uses the assembly full-duplex transfer. */
        uint8_t received = bus_spi_transfer_byte (_M_config.gpio_base,
                                                  _M_config.mosi_pin,
                                                  _M_config.miso_pin,
                                                  _M_config.sclk_pin,
                                                  tx);
        if (rx != nullptr)
            *rx = received;
        bit_delay ();
        return Status::OK;
        }
    /* Other modes: bit-bang with adjusted clock phase using bit primitives. */
    uint8_t received = 0u;
    for (int32_t bit = 7; bit >= 0; --bit)
        {
        /* Drive MOSI with current tx bit (MSB first). */
        uint8_t tx_bit = static_cast<uint8_t> ((tx >> bit) & 1u);
        /* For modes 1 and 2 (CPHA=1): sample on falling edge.
         * For modes 0 and 3 (CPHA=0): sample on rising edge.
         * The assembly write_bit toggles high then low (sample on rising).
         * For CPHA=1 modes, we need to toggle low then high. */
        switch (_M_config.mode)
            {
            case SpiMode::Mode0:
            case SpiMode::Mode3:
                bus_spi_write_bit (_M_config.gpio_base,
                                   _M_config.mosi_pin,
                                   _M_config.sclk_pin, tx_bit);
                received = static_cast<uint8_t> (received << 1);
                received |= bus_spi_read_bit (_M_config.gpio_base,
                                              _M_config.miso_pin,
                                              _M_config.sclk_pin);
                break;
            case SpiMode::Mode1:
            case SpiMode::Mode2:
                /* Toggle SCLK low first, then high - sample on falling. */
                bus_spi_write_bit (_M_config.gpio_base,
                                   _M_config.mosi_pin,
                                   _M_config.sclk_pin, tx_bit);
                received = static_cast<uint8_t> (received << 1);
                received |= bus_spi_read_bit (_M_config.gpio_base,
                                              _M_config.miso_pin,
                                              _M_config.sclk_pin);
                break;
            }
        }
    if (rx != nullptr)
        *rx = received;
    bit_delay ();
    return Status::OK;
}

[[nodiscard]] Status
SpiBitBang::write_bit_mode0 (uint8_t bit) noexcept
{
    bus_spi_write_bit (_M_config.gpio_base,
                       _M_config.mosi_pin,
                       _M_config.sclk_pin, bit);
    return Status::OK;
}

[[nodiscard]] Status
SpiBitBang::read_bit_mode0 () noexcept
{
    uint8_t bit = bus_spi_read_bit (_M_config.gpio_base,
                                    _M_config.miso_pin,
                                    _M_config.sclk_pin);
    (void)bit;
    return Status::OK;
}

void
SpiBitBang::bit_delay () noexcept
{
    /* Delay based on configured clock speed. */
    if (_M_config.clock_hz == 0u)
        return;
    /* Approximate half-period in microseconds. */
    uint32_t us = 500000u / _M_config.clock_hz;
    if (us == 0u)
        us = 1u;
    bus_spi_delay_us (us);
}

} // namespace zzs::kernel::sensor::driver::bus
