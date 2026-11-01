/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person arriving slower is
 * interpolated up, a channel that arrives faster is decimated down, and both
 * cases are the resampler's job rather than the stage's.
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
#include "kernel/kernel_fixed.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Common-grid time synchronization.
 *
 * All sensors are aligned to the `_S_grid_rate_hz` (100 Hz) tick.  A grid tick
 * is `_S_grid_period_ns` nanoseconds.  The tracker keeps a phase accumulator
 * (nanoseconds into the current tick) and reports, on every tick boundary, how
 * many samples each channel should contribute after resampling.
 *
 * Drift correction: the gyro/accel correlation peak between consecutive windows
 * is compared against the grid's expectation; a sustained offset shifts the
 * phase accumulator by a fraction of a tick, which re-syncs the grid without
 * dropping or duplicating a whole sample.
 *
 * No heap, no float, no sorting.  All thresholds are computed dynamically from
 * the configured grid period.
 * @{
 */
namespace zzs::kernel::clock {

/** @brief Nanoseconds per grid tick. */
inline constexpr Nanoseconds _S_grid_period_ns = zzs::kernel::_S_grid_period_ns;

/** @brief Maximum tolerable drift before a re-sync, as a fraction of a tick (Q15). */
inline constexpr int32_t _S_drift_resync_frac_q15 = 16384; /* 0.5 tick */

/** @brief Grid phase tracker state. */
struct _S_GridTracker {
        Nanoseconds _M_last_tick_ns; /* ns timestamp of last grid tick boundary */
        int64_t _M_phase_ns;         /* phase into the current tick, can be negative */
        uint32_t _M_tick_count;      /* total ticks emitted */
        int32_t _M_drift_accum_q15;  /* accumulated drift correction, Q15 fraction */
};

/** @brief Initialize the tracker from the current clock source. */
[[nodiscard]] inline zzs::kernel::Status
grid_tracker_init (_S_GridTracker &tracker, const Nanoseconds now_ns) noexcept {
        tracker._M_last_tick_ns = now_ns - (now_ns % _S_grid_period_ns);
        tracker._M_phase_ns = 0;
        tracker._M_tick_count = 0;
        tracker._M_drift_accum_q15 = 0;
        zzs::base::log_info ("grid tracker initialized");
        return zzs::kernel::Status::OK;
}

/** @brief Advance the tracker to `now_ns`, returning how many grid ticks elapsed. */
[[nodiscard]] inline uint32_t
grid_tracker_advance (_S_GridTracker &tracker, const Nanoseconds now_ns) noexcept {
        if (now_ns < tracker._M_last_tick_ns)
                return 0u; /* clock went backwards: ignore, stay put */

        const int64_t elapsed_ns = static_cast<int64_t> (now_ns) - static_cast<int64_t> (tracker._M_last_tick_ns);
        const int64_t total_phase_ns = tracker._M_phase_ns + elapsed_ns;

        /* A negative total phase (clock jumped) cannot produce a valid tick count:
           clamp to zero ticks and reset phase rather than wrapping uint32_t. */
        int64_t ticks = 0;
        int64_t remainder_ns = total_phase_ns;
        if (total_phase_ns >= 0) {
                ticks = total_phase_ns / static_cast<int64_t> (_S_grid_period_ns);
                remainder_ns = total_phase_ns % static_cast<int64_t> (_S_grid_period_ns);
        } else {
                tracker._M_phase_ns = 0;
        }

        if (ticks > 0) {
                tracker._M_last_tick_ns += static_cast<Nanoseconds> (ticks) * _S_grid_period_ns;
                tracker._M_phase_ns = remainder_ns;
                tracker._M_tick_count += static_cast<uint32_t> (ticks);
        } else if (total_phase_ns >= 0) {
                tracker._M_phase_ns = total_phase_ns;
        }
        return static_cast<uint32_t> (ticks);
}

/**
 * @brief Record a measured drift (gyro/accel correlation peak offset) as a Q15
 *        fraction of a tick and fold it into the phase accumulator.
 *
 * `drift_frac_q15` is positive when the physical source runs ahead of the grid.
 * The correction is clamped to _S_drift_resync_frac_q15 so a single spike cannot
 * jump the grid by more than half a tick.
 */
inline void
grid_tracker_apply_drift (_S_GridTracker &tracker, const int32_t drift_frac_q15) noexcept {
        const int32_t clamped =
            drift_frac_q15 > _S_drift_resync_frac_q15
                ? _S_drift_resync_frac_q15
                : (drift_frac_q15 < -_S_drift_resync_frac_q15 ? -_S_drift_resync_frac_q15 : drift_frac_q15);
        tracker._M_drift_accum_q15 = zzs::kernel::fx::q15_add_sat (tracker._M_drift_accum_q15, clamped);
        const int64_t drift_ns =
            (static_cast<int64_t> (tracker._M_drift_accum_q15) * static_cast<int64_t> (_S_grid_period_ns))
            >> zzs::kernel::_S_state_frac_bits;
        tracker._M_phase_ns += drift_ns;
        /* Normalize phase into [0, period) so a negative or over-tick phase cannot
           produce a negative tick count on the next advance. */
        if (tracker._M_phase_ns < 0)
                tracker._M_phase_ns = 0;
        else if (tracker._M_phase_ns >= static_cast<int64_t> (_S_grid_period_ns))
                tracker._M_phase_ns = static_cast<int64_t> (_S_grid_period_ns) - 1;
        tracker._M_drift_accum_q15 = 0; /* applied once */
}

/** @brief Current tick index (monotonic, wraps only at 2^32). */
[[nodiscard]] inline uint32_t
grid_tracker_tick_count (const _S_GridTracker &tracker) noexcept {
        return tracker._M_tick_count;
}

/** @} */
} // namespace zzs::kernel::clock
