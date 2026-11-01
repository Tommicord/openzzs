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

#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief Sensor-driver bus abstraction layer.
 *
 * Provides a platform-independent register-level bus interface for sensor
 * drivers (I2C, SPI, etc.).  Concrete implementations (bit-banged I2C,
 * bit-banged SPI, platform native controllers) inherit from this interface.
 *
 * All operations are `noexcept` and return `zzs::kernel::Status`.  No heap,
 * no float, no RTTI, no exceptions.
 * @{
 */
namespace zzs::kernel::sensor::driver::bus {

/** @brief Abstract bus interface implemented by I2C/SPI bit-bang and native controllers. */
class BusInterface {
public:
  /** @brief Virtual destructor, defaulted. */
  virtual ~BusInterface () = default;

  /** @brief Write `len` bytes to register `reg` on the bus device. */
  [[nodiscard]] virtual Status
  write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept = 0;

  /** @brief Read `len` bytes from register `reg` on the bus device. */
  [[nodiscard]] virtual Status
  read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept = 0;

  /** @brief Block for at least `ms` milliseconds. */
  virtual void
  delay_ms (uint32_t ms) noexcept = 0;
};

/** @} */
} // namespace zzs::kernel::sensor::driver::bus
