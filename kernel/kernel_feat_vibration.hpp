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
#include "kernel/kernel_dsp_biquad.hpp"
#include "kernel/kernel_dsp_goertzel.hpp"
#include "kernel/kernel_feat_pack.hpp"
#include "kernel/kernel_kernel_stage.hpp"
#include "kernel/kernel_meta_ids.hpp"
#include "kernel/kernel_sensor_accel.hpp"
#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_status.hpp"
#include "kernel/kernel_util_fixed.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Vibration feature stage.
 *
 * Input: a 200-sample window @ 100 Hz per accelerometer axis (X/Y/Z).
 *   - accel_rms_xyz: vector RMS over the 10–100 Hz bandpassed magnitude.
 *   - accel_peak_xyz: peak vector magnitude.
 *   - accel_spectral_centroid: centroid of the 10–100 Hz spectrum (Goertzel
 *     bins at 10 Hz steps, weighted by magnitude).
 *   - accel_50hz_power: power at 50/60 Hz (transformer vibration).
 *
 * Output: 4 Q15 features into slots [vib_start .. vib_start+3].
 *
 * The bandpass is 10–100 Hz (biquad).  The 50/60 Hz power uses a Goertzel bin
 * at the configured mains fundamental.  No float, no hardcoded thresholds.
 * @{
 */
namespace zzs::kernel::feat {

namespace ac = zzs::kernel::sensor::accel;
using zzs::kernel::dsp::_S_BiquadCoeffs;
using zzs::kernel::dsp::_S_BiquadState;
using zzs::kernel::dsp::_S_GoertzelBin;
using zzs::kernel::dsp::bandpass_coeffs;
using zzs::kernel::dsp::biquad_reset;
using zzs::kernel::dsp::biquad_step;
using zzs::kernel::dsp::goertzel_bin_configure;
using zzs::kernel::dsp::goertzel_bin_reset;

/** @brief Vibration stage state. */
struct _S_VibrationState {
    /* Bandpass 10-100 Hz per axis. */
    _S_BiquadCoeffs _M_bp_coeffs[ac::COUNT];
    _S_BiquadState _M_bp_state[ac::COUNT];

    /* 50/60 Hz Goertzel bin per axis (stepped incrementally in feed). */
    _S_GoertzelBin _M_hum_bin[ac::COUNT];

    /* Spectral-centroid Goertzel bins at 10,20,...,100 Hz (stepped
       incrementally in feed on the X axis, read in produce). */
    _S_GoertzelBin _M_centroid_bins[10];

    uint32_t _M_sample_rate_hz;
    uint32_t _M_window_samples;
    uint32_t _M_tick;
    int16_t _M_window_buf[ac::COUNT][_S_vibration_samples];
    bool _M_window_ready; /* true once a full window has been accumulated */
    bool _M_configured;
    uint32_t _M_mains_freq_hz; /* detected mains frequency (Hz) */
};

/** @brief Configure vibration from the accelerometer sample rate. */
[[nodiscard]] inline zzs::kernel::Status
vibration_configure (void *const state) noexcept {
    _S_VibrationState *const s = static_cast<_S_VibrationState *> (state);
    if (s == nullptr)
        return zzs::kernel::Status::OUT_OF_RANGE;
    if (s->_M_sample_rate_hz == 0u)
        s->_M_sample_rate_hz = zzs::kernel::_S_accel_rate_hz;
    if (s->_M_window_samples == 0u)
        s->_M_window_samples = zzs::kernel::_S_vibration_samples;

    /* Bandpass 10-100 Hz, Q=1. */
    const int32_t q_q15 = static_cast<int32_t> (zzs::kernel::_S_q15_one);
    for (uint32_t a = 0; a < ac::COUNT; ++a) {
        zzs::kernel::dsp::biquad_reset (s->_M_bp_state[a]);
        zzs::kernel::Status st = zzs::kernel::dsp::bandpass_coeffs (zzs::kernel::_S_vibration_band_low_hz,
                                                                    s->_M_sample_rate_hz, q_q15, s->_M_bp_coeffs[a]);
        if (st != zzs::kernel::Status::OK) {
            zzs::base::log_error ("vibration_configure: bandpass_coeffs={}", zzs::kernel::status_name (st));
            return st;
        }
        st = zzs::kernel::dsp::bandpass_coeffs (zzs::kernel::_S_vibration_band_high_hz, s->_M_sample_rate_hz, q_q15,
                                                s->_M_bp_coeffs[a]);
        (void) st;
    }

    /* Hum bin at the mains fundamental (50 Hz by default; the bank reports the
       stronger of 50/60 elsewhere, here we use the configured low grid). */
    for (uint32_t a = 0; a < ac::COUNT; ++a) {
        const zzs::kernel::Status st =
            zzs::kernel::dsp::goertzel_bin_configure (s->_M_hum_bin[a], s->_M_mains_freq_hz, s->_M_sample_rate_hz);
        (void) st;
    }

    /* Spectral-centroid bins at 10 Hz steps across the vibration band. */
    for (uint32_t b = 0; b < 10u; ++b) {
        const uint32_t f = zzs::kernel::_S_vibration_band_low_hz + b * 10u;
        const zzs::kernel::Status st =
            zzs::kernel::dsp::goertzel_bin_configure (s->_M_centroid_bins[b], f, s->_M_sample_rate_hz);
        (void) st;
    }

    s->_M_tick = 0;
    s->_M_window_ready = false;
    s->_M_mains_freq_hz = 60u;
    zzs::base::log_debug ("vibration_configure: rate={} window={}", s->_M_sample_rate_hz, s->_M_window_samples);
    return zzs::kernel::Status::OK;
}

/** @brief Reset vibration state. */
inline void
vibration_reset (void *const state) noexcept {
    _S_VibrationState *const s = static_cast<_S_VibrationState *> (state);
    if (s == nullptr)
        return;
    for (uint32_t a = 0; a < ac::COUNT; ++a) {
        zzs::kernel::dsp::biquad_reset (s->_M_bp_state[a]);
        zzs::kernel::dsp::goertzel_bin_reset (s->_M_hum_bin[a]);
        for (uint32_t i = 0; i < s->_M_window_samples; ++i)
            s->_M_window_buf[a][i] = 0;
    }
    for (uint32_t b = 0; b < 10u; ++b)
        zzs::kernel::dsp::goertzel_bin_reset (s->_M_centroid_bins[b]);
    s->_M_tick = 0;
    s->_M_window_ready = false;
    s->_M_mains_freq_hz = 60u;
}

/** @brief Feed one resampled Q15 accelerometer sample. */
[[nodiscard]] inline zzs::kernel::Status
vibration_feed (void *const state, const sensor::Channel channel, const int16_t sample_q15) noexcept {
    _S_VibrationState *const s = static_cast<_S_VibrationState *> (state);
    if (s == nullptr || !s->_M_configured)
        return zzs::kernel::Status::NOT_READY;

    const uint32_t axis = static_cast<uint32_t> (channel) - static_cast<uint32_t> (sensor::Channel::ACCEL_X);
    if (axis >= ac::COUNT)
        return zzs::kernel::Status::OK;

    const int16_t filtered = zzs::kernel::dsp::biquad_step (s->_M_bp_coeffs[axis], s->_M_bp_state[axis], sample_q15);
    if (s->_M_tick < s->_M_window_samples)
        s->_M_window_buf[axis][s->_M_tick] = filtered;

    /* Step the hum resonator for this axis, incrementally. */
    static_cast<void> (zzs::kernel::dsp::goertzel_bin_step (s->_M_hum_bin[axis], filtered));

    /* Step the spectral-centroid resonators on the X axis only (centroid uses
       the X-axis window). */
    if (axis == static_cast<uint32_t> (sensor::Channel::ACCEL_X))
        for (uint32_t b = 0; b < 10u; ++b)
            static_cast<void> (zzs::kernel::dsp::goertzel_bin_step (s->_M_centroid_bins[b], filtered));

    /* Advance the shared window tick once all three axes have a sample at this
       grid tick. */
    if (axis == static_cast<uint32_t> (sensor::Channel::ACCEL_Z)) {
        ++s->_M_tick;
        if (s->_M_tick >= s->_M_window_samples) {
            s->_M_tick = 0;
            s->_M_window_ready = true;
        }
    }
    return zzs::kernel::Status::OK;
}

/** @brief Produce the 4 vibration Q15 features into `slots`. */
[[nodiscard]] inline zzs::kernel::Status
vibration_produce (void *const state, int32_t *const slots, const uint32_t slot_count) noexcept {
    _S_VibrationState *const s = static_cast<_S_VibrationState *> (state);
    if (s == nullptr || !s->_M_configured || slots == nullptr)
        return zzs::kernel::Status::NOT_READY;
    if (slot_count < meta::_S_feat_vib_start + meta::_S_feat_vib_count) {
        zzs::base::log_error ("vibration_produce: budget_exceeded slots={} need={}", slot_count,
                              meta::_S_feat_vib_start + meta::_S_feat_vib_count);
        return zzs::kernel::Status::BUDGET_EXCEEDED;
    }

    /* Build a per-sample vector magnitude array for RMS/peak/centroid. */
    int16_t mag[_S_vibration_samples];
    for (uint32_t i = 0; i < s->_M_window_samples; ++i) {
        int32_t vx = s->_M_window_buf[ac::X][i];
        int32_t vy = s->_M_window_buf[ac::Y][i];
        int32_t vz = s->_M_window_buf[ac::Z][i];
        /* Vector magnitude = sqrt(vx^2+vy^2+vz^2) as Q15 via isqrt. */
        const int64_t sq =
            static_cast<int64_t> (vx) * vx + static_cast<int64_t> (vy) * vy + static_cast<int64_t> (vz) * vz;
        mag[i] = static_cast<int16_t> (zzs::mllee::_S_isqrt (static_cast<uint32_t> (sq > 0 ? sq : 0)));
    }

    /* accel_rms_xyz */
    const int32_t rms_xyz = zzs::kernel::fx::rms_q15 (mag, s->_M_window_samples);
    slots[meta::_S_feat_vib_start + meta::vib::rms_xyz] = rms_xyz;

    /* accel_peak_xyz */
    const int32_t peak_xyz = zzs::kernel::fx::peak_q15 (mag, s->_M_window_samples);
    slots[meta::_S_feat_vib_start + meta::vib::peak_xyz] = peak_xyz;

    if (!s->_M_window_ready)
        return zzs::kernel::Status::NOT_READY;

    /* accel_spectral_centroid: centroid of the incrementally-stepped centroid
       bin magnitudes (O(10) reads, no window re-feed). */
    int16_t bins[10];
    for (uint32_t b = 0; b < 10u; ++b)
        bins[b] = static_cast<int16_t> (
            zzs::kernel::fx::magnitude_q7 (zzs::kernel::dsp::goertzel_bin_power_q14 (s->_M_centroid_bins[b])));
    const int32_t centroid = zzs::kernel::fx::spectral_centroid_q15 (bins, 10u);
    slots[meta::_S_feat_vib_start + meta::vib::spectral_centroid] = centroid;

    /* accel_50hz_power: read the incrementally-stepped hum bin power per axis. */
    int32_t best_power_q14 = 0;
    for (uint32_t a = 0; a < ac::COUNT; ++a) {
        const uint32_t p = zzs::kernel::dsp::goertzel_bin_power_q14 (s->_M_hum_bin[a]);
        if (static_cast<int32_t> (p) > best_power_q14)
            best_power_q14 = static_cast<int32_t> (p);
    }
    const zzs::kernel::fx::_S_LevelReference ref = zzs::kernel::fx::level_reference_of (s->_M_window_samples);
    const int32_t hum_db_q8 = zzs::kernel::fx::db_q8 (static_cast<uint32_t> (best_power_q14), ref);
    slots[meta::_S_feat_vib_start + meta::vib::power_50hz] = hum_db_q8 << 7u;

    /* Window consumed: reset all resonators for the next window. */
    for (uint32_t a = 0; a < ac::COUNT; ++a)
        zzs::kernel::dsp::goertzel_bin_reset (s->_M_hum_bin[a]);
    for (uint32_t b = 0; b < 10u; ++b)
        zzs::kernel::dsp::goertzel_bin_reset (s->_M_centroid_bins[b]);
    s->_M_window_ready = false;
    s->_M_mains_freq_hz = 60u;
    zzs::base::log_debug ("vibration_produce: window complete");
    return zzs::kernel::Status::OK;
}

/** @brief Vibration stage vtable. */
[[nodiscard]] inline const _S_StageVtable &
vibration_vtable (void) noexcept {
    static const _S_StageVtable vt = {vibration_configure, vibration_reset, vibration_feed, vibration_produce};
    return vt;
}

/** @} */
} // namespace zzs::kernel::feat
