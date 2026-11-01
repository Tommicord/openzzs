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

#include "kernel/kernel_config.hpp"
#include "kernel/kernel_dsp_goertzel.hpp"
#include "kernel/kernel_feat_pack.hpp"
#include "kernel/kernel_kernel_stage.hpp"
#include "kernel/kernel_meta_ids.hpp"
#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_util_fixed.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Flicker feature stage.
 *
 * Input: a 100-sample window @ 100 Hz of light (upsampled from 1 Hz by the
 * resampler).
 *   - light_dc_level: mean lux (Q15).
 *   - light_100hz_power: 100/120 Hz spectral power (Goertzel at mains*2).
 *   - light_flicker_index: (max-min)/(max+min) over the cycle (Q15).
 *   - light_flicker_drop: DC delta vs. a 5-window median (Q15).
 *
 * Output: 4 Q15 features into slots [flicker_start .. flicker_start+3].
 *
 * No float, no sorting of the full window (median via selection pass), no
 * hardcoded thresholds.
 * @{
 */
namespace zzs::kernel::feat {

using zzs::kernel::dsp::_S_GoertzelBin;
using zzs::kernel::dsp::goertzel_bin_configure;
using zzs::kernel::dsp::goertzel_bin_power_q14;
using zzs::kernel::dsp::goertzel_bin_reset;
using zzs::kernel::dsp::goertzel_bin_step;
using zzs::kernel::fx::magnitude_q7;

/** @brief Flicker stage state. */
struct _S_FlickerState {
        uint32_t _M_sample_rate_hz;
        uint32_t _M_window_samples;
        uint32_t _M_tick;
        int16_t _M_window_buf[_S_flicker_samples];

        /* 100/120 Hz hum resonator, stepped incrementally in feed and read in
           produce (no window re-feed). */
        _S_GoertzelBin _M_hum_bin;

        /* 5-window DC history for the flicker drop median. */
        int16_t _M_dc_history[5];
        uint32_t _M_dc_index;
        bool _M_window_ready; /* true once a full window has been accumulated */
        bool _M_configured;
};

/** @brief Configure flicker from the light sample rate. */
[[nodiscard]] inline zzs::kernel::Status
flicker_configure (void *const state) noexcept {
        _S_FlickerState *const s = static_cast<_S_FlickerState *> (state);
        if (s == nullptr)
                return zzs::kernel::Status::OUT_OF_RANGE;
        if (s->_M_sample_rate_hz == 0u)
                s->_M_sample_rate_hz = zzs::kernel::_S_light_rate_hz;
        if (s->_M_window_samples == 0u)
                s->_M_window_samples = zzs::kernel::_S_flicker_samples;
        /* Hum resonator at 2*mains (100/120 Hz). */
        const zzs::kernel::Status st = zzs::kernel::dsp::goertzel_bin_configure (
            s->_M_hum_bin, zzs::kernel::_S_mains_low_hz * 2u, s->_M_sample_rate_hz);
        if (st != zzs::kernel::Status::OK) {
                zzs::base::log_error ("flicker_configure: goertzel_bin_configure={}", zzs::kernel::status_name (st));
                return st;
        }
        s->_M_tick = 0;
        s->_M_dc_index = 0;
        for (uint32_t i = 0; i < 5u; ++i)
                s->_M_dc_history[i] = 0;
        s->_M_window_ready = false;
        s->_M_configured = true;
        zzs::base::log_debug ("flicker_configure: rate={} window={}", s->_M_sample_rate_hz, s->_M_window_samples);
        return zzs::kernel::Status::OK;
}

/** @brief Reset flicker state. */
inline void
flicker_reset (void *const state) noexcept {
        _S_FlickerState *const s = static_cast<_S_FlickerState *> (state);
        if (s == nullptr)
                return;
        for (uint32_t i = 0; i < s->_M_window_samples; ++i)
                s->_M_window_buf[i] = 0;
        zzs::kernel::dsp::goertzel_bin_reset (s->_M_hum_bin);
        s->_M_tick = 0;
        for (uint32_t i = 0; i < 5u; ++i)
                s->_M_dc_history[i] = 0;
        s->_M_dc_index = 0;
        s->_M_window_ready = false;
}

/** @brief Feed one resampled Q15 light sample. */
[[nodiscard]] inline zzs::kernel::Status
flicker_feed (void *const state, const sensor::Channel channel, const int16_t sample_q15) noexcept {
        _S_FlickerState *const s = static_cast<_S_FlickerState *> (state);
        if (s == nullptr || !s->_M_configured)
                return zzs::kernel::Status::NOT_READY;
        if (channel != sensor::Channel::LIGHT)
                return zzs::kernel::Status::OK;
        if (s->_M_tick < s->_M_window_samples)
                s->_M_window_buf[s->_M_tick] = sample_q15;

        /* Step the hum resonator incrementally per sample. */
        zzs::kernel::dsp::goertzel_bin_step (s->_M_hum_bin, sample_q15);

        ++s->_M_tick;
        if (s->_M_tick >= s->_M_window_samples) {
                s->_M_tick = 0;
                s->_M_window_ready = true;
        }
        return zzs::kernel::Status::OK;
}

/** @brief Produce the 4 flicker Q15 features into `slots`. */
[[nodiscard]] inline zzs::kernel::Status
flicker_produce (void *const state, int32_t *const slots, const uint32_t slot_count) noexcept {
        _S_FlickerState *const s = static_cast<_S_FlickerState *> (state);
        if (s == nullptr || !s->_M_configured || slots == nullptr)
                return zzs::kernel::Status::NOT_READY;
        if (slot_count < meta::_S_feat_flicker_start + meta::_S_feat_flicker_count) {
                zzs::base::log_error ("flicker_produce: budget_exceeded slots={} need={}", slot_count,
                                      meta::_S_feat_flicker_start + meta::_S_feat_flicker_count);
                return zzs::kernel::Status::BUDGET_EXCEEDED;
        }

        const uint32_t n = s->_M_window_samples;
        if (n == 0u)
                return zzs::kernel::Status::NOT_READY;

        /* light_dc_level: mean. */
        const int32_t dc = zzs::kernel::fx::mean_q15 (s->_M_window_buf, n);
        slots[meta::_S_feat_flicker_start + meta::flicker::dc_level] = dc;

        /* Store DC history for flicker_drop (median of last 5 windows). */
        s->_M_dc_history[s->_M_dc_index % 5u] = dc;
        ++s->_M_dc_index;

        if (!s->_M_window_ready)
                return zzs::kernel::Status::NOT_READY;

        /* light_100hz_power: O(1) read of the incrementally-stepped hum resonator. */
        const uint32_t hum_power_q14 = zzs::kernel::dsp::goertzel_bin_power_q14 (s->_M_hum_bin);
        const zzs::kernel::fx::_S_LevelReference ref = zzs::kernel::fx::level_reference_of (n);
        const int32_t hum_db_q8 = zzs::kernel::fx::db_q8 (hum_power_q14, ref);
        slots[meta::_S_feat_flicker_start + meta::flicker::power_100hz] = hum_db_q8 << 7u;

        /* light_flicker_index: (max-min)/(max+min) as Q15. */
        int32_t max_v = s->_M_window_buf[0];
        int32_t min_v = s->_M_window_buf[0];
        for (uint32_t i = 1; i < n; ++i) {
                if (s->_M_window_buf[i] > max_v)
                        max_v = s->_M_window_buf[i];
                if (s->_M_window_buf[i] < min_v)
                        min_v = s->_M_window_buf[i];
        }
        const int64_t denom = static_cast<int64_t> (max_v) + min_v;
        int32_t flicker_index_q15 = 0;
        if (denom != 0) {
                const int64_t numer = static_cast<int64_t> (max_v) - min_v;
                flicker_index_q15 = static_cast<int32_t> ((numer << zzs::kernel::_S_state_frac_bits) / denom);
        }
        slots[meta::_S_feat_flicker_start + meta::flicker::flicker_index] = flicker_index_q15;

        /* light_flicker_drop: DC delta vs. 5-window median (Q15). */
        const int32_t median_dc = zzs::kernel::fx::median_q15 (s->_M_dc_history, 5u);
        const int32_t drop_q15 = dc - median_dc;
        slots[meta::_S_feat_flicker_start + meta::flicker::flicker_drop] = drop_q15;

        /* Window consumed: reset the hum resonator for the next window. */
        zzs::kernel::dsp::goertzel_bin_reset (s->_M_hum_bin);
        s->_M_window_ready = false;
        zzs::base::log_debug ("flicker_produce: window complete");
        return zzs::kernel::Status::OK;
}

/** @brief Flicker stage vtable. */
[[nodiscard]] inline const _S_StageVtable &
flicker_vtable (void) noexcept {
        static const _S_StageVtable vt = {flicker_configure, flicker_reset, flicker_feed, flicker_produce};
        return vt;
}

/** @} */
} // namespace zzs::kernel::feat
