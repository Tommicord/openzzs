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

#pragma once

#include "kernel/sensor-driver/bus/bus_interface.hpp"

#include <cstdint>

/**
 * @brief Bit-banged SPI bus using assembly routines for GPIO toggling.
 *
 * Implements SPI over software-controlled GPIO pins using the hand-written
 * assembly bit-bang routines in `bus/asm/`.  Supports all four clock
 * polarity/phase modes (0-3), MSB-first transfer, and full-duplex byte
 * transfer.
 *
 * No heap, no float, no RTTI, no exceptions.
 * @{
 */
namespace zzs::kernel::sensor::driver::bus {

/** @brief SPI clock polarity / phase mode. */
enum class SpiMode : uint8_t {
    Mode0 = 0u, /**< CPOL=0, CPHA=0: sample on rising, setup on falling */
    Mode1 = 1u, /**< CPOL=0, CPHA=1: sample on falling, setup on rising */
    Mode2 = 2u, /**< CPOL=1, CPHA=0: sample on falling, setup on rising */
    Mode3 = 3u, /**< CPOL=1, CPHA=1: sample on rising, setup on falling */
};

/** @brief SpiBitBang configuration. */
struct SpiConfig {
    uint32_t gpio_base = 0u; /**< Memory-mapped GPIO base address */
    uint32_t mosi_pin = 0u;  /**< MOSI GPIO pin number */
    uint32_t miso_pin = 0u;  /**< MISO GPIO pin number */
    uint32_t sclk_pin = 0u;  /**< SCLK GPIO pin number */
    uint32_t cs_pin = 0u;    /**< Chip select GPIO pin number */
    SpiMode mode = SpiMode::Mode0;
    uint32_t clock_hz = 1000000u; /**< SPI clock speed in Hz */
};

/** @brief Bit-banged SPI bus implementation. */
class SpiBitBang : public BusInterface {
  public:
    /** @brief Construct with configuration. */
    explicit SpiBitBang (const SpiConfig &config) noexcept;

    /** @brief Destructor. */
    ~SpiBitBang () override = default;

    /** @brief Disable copy/move. */
    SpiBitBang (const SpiBitBang &) = delete;
    SpiBitBang &
    operator= (const SpiBitBang &) = delete;
    SpiBitBang (SpiBitBang &&) = delete;
    SpiBitBang &
    operator= (SpiBitBang &&) = delete;

    // -- BusInterface --
    [[nodiscard]] Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override;

    [[nodiscard]] Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override;

    void
    delay_ms (uint32_t ms) noexcept override;

    // -- SPI-specific operations --
    /**
     * @brief Full-duplex transfer of `len` bytes.
     * @param tx  transmit buffer (may be nullptr for read-only)
     * @param rx  receive buffer (may be nullptr for write-only)
     */
    [[nodiscard]] Status
    transfer (const uint8_t *tx, uint8_t *rx, uint16_t len) noexcept;

    /** @brief Write `len` bytes after asserting CS. */
    [[nodiscard]] Status
    write (const uint8_t *data, uint16_t len) noexcept;

    /** @brief Read `len` bytes after asserting CS. */
    [[nodiscard]] Status
    read (uint8_t *data, uint16_t len) noexcept;

    /** @brief Assert chip select. */
    [[nodiscard]] Status
    cs_assert () noexcept;

    /** @brief Deassert chip select. */
    [[nodiscard]] Status
    cs_deassert () noexcept;

  private:
    [[nodiscard]] Status
    transfer_byte (uint8_t tx, uint8_t *rx) noexcept;

    [[nodiscard]] Status
    write_bit_mode0 (uint8_t bit) noexcept;

    [[nodiscard]] Status
    read_bit_mode0 () noexcept;

    void
    bit_delay () noexcept;

    SpiConfig _M_config;
};

/** @} */
} // namespace zzs::kernel::sensor::driver::bus
