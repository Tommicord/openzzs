/*
 * Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

#pragma once

/**
 * @file test_mock_bus.hpp
 * @brief Mock bus and HAL implementations for sensor-driver unit tests.
 *
 * A deterministic in-memory register model plus a HAL that yields Q15 sample
 * values on demand. No hardware, no float, no heap.
 */

#include "kernel/sensor-driver/bus/bus_interface.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>

namespace test
{

/** @brief In-memory register bank implementing BusInterface. */
class MockBus : public zzs::kernel::sensor::driver::bus::BusInterface
{
public:
    MockBus () noexcept = default;

    [[nodiscard]] zzs::kernel::Status
    write_reg (uint8_t reg, const uint8_t *data, uint16_t len) noexcept override
    {
        if (data == nullptr || len == 0u)
            return zzs::kernel::Status::OUT_OF_RANGE;
        if (!_M_initialized)
            return zzs::kernel::Status::INTERNAL;
        for (uint16_t i = 0u; i < len; ++i)
            _M_regs[static_cast<uint32_t> (reg) + i] = data[i];
        _M_last_written_reg = reg;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    read_reg (uint8_t reg, uint8_t *data, uint16_t len) noexcept override
    {
        if (data == nullptr || len == 0u)
            return zzs::kernel::Status::OUT_OF_RANGE;
        if (!_M_initialized)
            return zzs::kernel::Status::INTERNAL;
        _M_read_reg_calls += 1u;
        for (uint16_t i = 0u; i < len; ++i)
            data[i] = _M_regs[static_cast<uint32_t> (reg) + i];
        _M_last_read_reg = reg;
        return zzs::kernel::Status::OK;
    }

    void
    delay_ms (uint32_t ms) noexcept override
    {
        _M_delay_calls += 1u;
        _M_delay_total_ms += ms;
    }

    /** @brief Inject a register value so a driver read observes it. */
    void
    set_reg (uint8_t reg, uint8_t value) noexcept
    {
        _M_regs[static_cast<uint32_t> (reg)] = value;
    }

    /** @brief Inject a multi-byte block at a starting register. */
    void
    set_block (uint8_t reg, const uint8_t *data, uint16_t len) noexcept
    {
        if (data == nullptr || len == 0u)
            return;
        for (uint16_t i = 0u; i < len; ++i)
            _M_regs[static_cast<uint32_t> (reg) + i] = data[i];
    }

    uint8_t
    get_reg (uint8_t reg) const noexcept
    {
        return _M_regs[static_cast<uint32_t> (reg)];
    }

    void
    set_initialized (bool initialized) noexcept
    {
        _M_initialized = initialized;
    }

    bool
    is_initialized () const noexcept
    {
        return _M_initialized;
    }

    uint32_t
    delay_calls () const noexcept
    {
        return _M_delay_calls;
    }

    uint32_t
    delay_total_ms () const noexcept
    {
        return _M_delay_total_ms;
    }

    uint32_t
    read_reg_calls () const noexcept
    {
        return _M_read_reg_calls;
    }

    uint8_t
    last_written_reg () const noexcept
    {
        return _M_last_written_reg;
    }

    uint8_t
    last_read_reg () const noexcept
    {
        return _M_last_read_reg;
    }

    void
    clear () noexcept
    {
        _M_regs.fill (0);
        _M_last_written_reg = 0;
        _M_last_read_reg = 0;
        _M_delay_calls = 0;
        _M_delay_total_ms = 0;
        _M_read_reg_calls = 0;
    }

private:
    std::array<uint8_t, 256u> _M_regs{};
    bool _M_initialized = false;
    uint32_t _M_delay_calls = 0u;
    uint32_t _M_delay_total_ms = 0u;
    uint32_t _M_read_reg_calls = 0u;
    uint8_t _M_last_written_reg = 0u;
    uint8_t _M_last_read_reg = 0u;
};

/** @brief Minimal HAL producing Q15 sample values on demand. */
class MockHal
{
public:
    MockHal () noexcept = default;

    [[nodiscard]] bool
    init () noexcept
    {
        _M_initialized = true;
        return true;
    }

    [[nodiscard]] bool
    is_initialized () const noexcept
    {
        return _M_initialized;
    }

    [[nodiscard]] int16_t
    read_channel (uint32_t ch) noexcept
    {
        return _M_values[ch % _M_values.size ()];
    }

    void
    set_value (uint32_t ch, int16_t q15) noexcept
    {
        if (ch < _M_values.size ())
            _M_values[ch] = q15;
    }

    [[nodiscard]] bool
    channel_enabled (uint32_t ch) const noexcept
    {
        return ch < _M_values.size () && _M_enabled[ch];
    }

    void
    set_channel_enabled (uint32_t ch, bool enabled) noexcept
    {
        if (ch < _M_values.size ())
            _M_enabled[ch] = enabled;
    }

    [[nodiscard]] uint32_t
    channel_rate_hz (uint32_t ch) const noexcept
    {
        return 100u;
    }

    void
    set_initialized (bool initialized) noexcept
    {
        _M_initialized = initialized;
    }

    void
    reset () noexcept
    {
        _M_initialized = false;
        _M_values.fill (0);
        _M_enabled.fill (false);
    }

private:
    bool _M_initialized = false;
    std::array<int16_t, 12u> _M_values{};
    std::array<bool, 12u> _M_enabled{};
};

} // namespace test
