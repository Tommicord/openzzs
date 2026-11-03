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

#include <cstdint>

/**
 * @brief Shared types used across the preprocessing bank.
 *
 * Centralizes the sample, window, grid-tick and feature-row types so every
 * stage agrees on the data layout without duplicating declarations.
 *
 * No heap, no float, no std::string/vector/array.
 * @{
 */
namespace zzs::kernel::types {

using zzs::kernel::clock::Nanoseconds;

/** @brief A normalized Q15 sensor sample on the wire (int16 storage). */
using SampleQ15 = int16_t;

/** @brief A filter state or accumulator (Q15 int32 storage). */
using StateQ15 = int32_t;

/** @brief A Goertzel resonator state (Q7 int16 storage). */
using ResonatorQ7 = int16_t;

/** @brief A filter/resonator coefficient (Q13/Q14 int16 storage). */
using CoeffQ15 = int16_t;

/** @brief A power value (Q14 uint32 storage). */
using PowerQ14 = uint32_t;

/** @brief A level in dB (Q8 int32 storage). */
using LevelQ8 = int32_t;

/** @brief A grid tick index, monotonic, wraps at 2^32. */
using GridTick = uint32_t;

/** @brief A feature value in Q15 before int8 coding. */
using FeatureQ15 = int32_t;

/** @brief One packed feature row: _S_feature_count int8 codes. */
struct FeatureRow {
    int8_t _M_code[zzs::kernel::_S_feature_count];
};

/** @brief One grid-tick sample bundle: all channels resampled to the grid. */
struct GridSample {
    GridTick _M_tick;
    Nanoseconds _M_timestamp_ns;
    SampleQ15 _M_mag_x;
    SampleQ15 _M_mag_y;
    SampleQ15 _M_mag_z;
    SampleQ15 _M_accel_x;
    SampleQ15 _M_accel_y;
    SampleQ15 _M_accel_z;
    SampleQ15 _M_gyro_x;
    SampleQ15 _M_gyro_y;
    SampleQ15 _M_gyro_z;
    SampleQ15 _M_light;
    SampleQ15 _M_sound_envelope;
    SampleQ15 _M_battery;
};

/** @brief Clear a grid sample to zero. */
inline void
grid_sample_clear (GridSample &gs) noexcept {
    gs._M_tick = 0;
    gs._M_timestamp_ns = 0;
    gs._M_mag_x = 0;
    gs._M_mag_y = 0;
    gs._M_mag_z = 0;
    gs._M_accel_x = 0;
    gs._M_accel_y = 0;
    gs._M_accel_z = 0;
    gs._M_gyro_x = 0;
    gs._M_gyro_y = 0;
    gs._M_gyro_z = 0;
    gs._M_light = 0;
    gs._M_sound_envelope = 0;
    gs._M_battery = 0;
}

/** @} */
} // namespace zzs::kernel::types
