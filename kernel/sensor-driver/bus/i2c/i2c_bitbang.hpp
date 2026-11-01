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
 * @brief Bit-banged I2C bus using assembly routines for GPIO toggling.
 *
 * Implements I2C over software-controlled GPIO pins using the hand-written
 * assembly bit-bang routines in `bus/asm/`.  Supports 7-bit and 10-bit
 * addressing, repeated starts, and three standard clock speeds
 * (100 kHz, 400 kHz, 1 MHz).
 *
 * No heap, no float, no RTTI, no exceptions.
 * @{
 */
namespace zzs::kernel::sensor::driver::bus {

/** @brief I2C clock speed options. */
enum class I2cClockHz : uint32_t {
        k100kHz = 100000u,
        k400kHz = 400000u,
        k1MHz = 1000000u,
};

/** @brief I2cBitBang configuration. */
struct I2cConfig {
        uint32_t gpio_base = 0u;  /**< Memory-mapped GPIO base address */
        uint32_t sda_pin = 0u;    /**< SDA GPIO pin number */
        uint32_t scl_pin = 0u;    /**< SCL GPIO pin number */
        uint8_t device_addr = 0u; /**< 7-bit I2C device address */
        I2cClockHz clock = I2cClockHz::k100kHz;
};

/** @brief Bit-banged I2C bus implementation. */
class I2cBitBang : public BusInterface {
      public:
        /** @brief Construct with configuration. */
        explicit I2cBitBang (const I2cConfig &config) noexcept;

        /** @brief Destructor. */
        ~I2cBitBang () override = default;

        /** @brief Disable copy/move. */
        I2cBitBang (const I2cBitBang &) = delete;
        I2cBitBang &
        operator= (const I2cBitBang &) = delete;
        I2cBitBang (I2cBitBang &&) = delete;
        I2cBitBang &
        operator= (I2cBitBang &&) = delete;

        // -- BusInterface --
        [[nodiscard]] Status
        write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override;

        [[nodiscard]] Status
        read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override;

        void
        delay_ms (uint32_t ms) noexcept override;

        // -- I2C-specific operations --
        /**
         * @brief Write `len` bytes to device `addr` (7-bit).
         * @param repeated_start  if true, do not send STOP before the write
         *                        (used for combined write-then-read sequences).
         */
        [[nodiscard]] Status
        write (uint8_t addr, const uint8_t *data, uint16_t len, bool repeated_start = false) noexcept;

        /**
         * @brief Read `len` bytes from device `addr` (7-bit).
         * @param repeated_start  if true, issue Repeated Start instead of
         *                        Stop+Start.
         */
        [[nodiscard]] Status
        read (uint8_t addr, uint8_t *data, uint16_t len, bool repeated_start = false) noexcept;

        /** @brief Send a start condition. */
        [[nodiscard]] Status
        start () noexcept;

        /** @brief Send a stop condition. */
        [[nodiscard]] Status
        stop () noexcept;

      private:
        [[nodiscard]] Status
        write_byte (uint8_t byte) noexcept;

        [[nodiscard]] Status
        read_byte (uint8_t *byte, bool last) noexcept;

        [[nodiscard]] Status
        write_addr7 (uint8_t addr, bool write) noexcept;

        [[nodiscard]] Status
        write_addr10 (uint8_t addr_high, uint8_t addr_low, bool write) noexcept;

        void
        bit_delay () noexcept;

        I2cConfig _M_config;
};

/** @} */
} // namespace zzs::kernel::sensor::driver::bus
