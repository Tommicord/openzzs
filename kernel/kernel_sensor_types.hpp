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

#pragma once

#include "kernel/kernel_clock_source.hpp"
#include "kernel/kernel_config.hpp"
#include "kernel/kernel_status.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Sensor sample types for the preprocessing bank.
 *
 * Every sensor event is a Q15 int16 sample tagged with a channel id and a
 * monotonic nanosecond timestamp.  The kernel is axis-agnostic: a 3-axis
 * magnetometer delivers 3 samples (one per axis) on consecutive calls, and a
 * single-axis light sensor delivers one.  The caller (HAL) maps hardware channels
 * to kernel channel ids.
 *
 * No heap, no std::string, no float.
 * @{
 */
namespace zzs::kernel::sensor {

using zzs::kernel::clock::Nanoseconds;

/** @brief Kernel channel ids; the HAL maps hardware sensors to these. */
enum class Channel : uint8_t {
    MAG_X = 0,
    MAG_Y = 1,
    MAG_Z = 2,
    ACCEL_X = 3,
    ACCEL_Y = 4,
    ACCEL_Z = 5,
    GYRO_X = 6,
    GYRO_Y = 7,
    GYRO_Z = 8,
    LIGHT = 9,
    SOUND_ENVELOPE = 10,
    BATTERY = 11,
    COUNT = 12,
};

/** @brief One Q15 sensor sample. */
struct Sample {
    Channel _M_channel;
    int16_t _M_sample_q15;
    Nanoseconds _M_timestamp_ns;
};

/** @brief Number of kernel channels. */
[[nodiscard]] inline constexpr uint32_t
channel_count (void) noexcept {
    return static_cast<uint32_t> (Channel::COUNT);
}

/** @brief Expected sample rate (Hz) of a channel, from the platform HAL table. */
[[nodiscard]] inline constexpr uint32_t
channel_rate_hz (const Channel channel) noexcept {
    switch (channel) {
    case Channel::MAG_X:
    case Channel::MAG_Y:
    case Channel::MAG_Z:
        return zzs::kernel::_S_mag_rate_hz;
    case Channel::ACCEL_X:
    case Channel::ACCEL_Y:
    case Channel::ACCEL_Z:
        return zzs::kernel::_S_accel_rate_hz;
    case Channel::GYRO_X:
    case Channel::GYRO_Y:
    case Channel::GYRO_Z:
        return zzs::kernel::_S_gyro_rate_hz;
    case Channel::LIGHT:
    case Channel::BATTERY:
        return zzs::kernel::_S_light_rate_hz;
    case Channel::SOUND_ENVELOPE:
        return zzs::kernel::_S_envelope_rate_hz;
    default:
        return zzs::kernel::_S_grid_rate_hz;
    }
}

/** @} */
} // namespace zzs::kernel::sensor
