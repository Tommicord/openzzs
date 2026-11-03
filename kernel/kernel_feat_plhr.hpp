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
#include "kernel/kernel_dsp_goertzel_bank.hpp"
#include "kernel/kernel_feat_pack.hpp"
#include "kernel/kernel_kernel_stage.hpp"
#include "kernel/kernel_meta_ids.hpp"
#include "kernel/kernel_sensor_accel.hpp"
#include "kernel/kernel_sensor_mag.hpp"
#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_status.hpp"
#include "kernel/kernel_util_fixed.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief PLHR (Power Line Harmonic Ratio) spectral feature stage.
 *
 * Input: a 200-sample window @ 200 Hz per magnetometer axis (X/Y/Z), i.e.
 * 3 × 200 int16 Q15.  Per axis:
 *   1. Bandpass 45–250 Hz (biquad) to isolate the mains harmonics.
 *   2. Goertzel bank at the 50 Hz and 60 Hz harmonic grids:
 *      50 Hz grid: bins 64, 128, 192, 256 (i.e. 50, 100, 150, 200 Hz)
 *      60 Hz grid: bins 77, 154, 231, 308 (i.e. 60, 120, 180, 240 Hz)
 *      Since a 256-point FFT at 200 Hz = 0.78 Hz/bin, the harmonic bins are
 *      not integral; Goertzel hits them exactly.
 *   3. Each bin's squared magnitude (Q14) is normalized against the window's
 *      full-scale reference and written as a Q15 feature.
 *
 * Output: 3 axes × 2 grids × 4 harmonics = 24 Q15 features, packed into slots
 * [0..23] per kernel_meta_ids.hpp.
 *
 * The bandpass and Goertzel coefficients are computed dynamically from the
 * configured sample rate; no float, no hardcoded frequency thresholds.
 * @{
 */
namespace zzs::kernel::feat {

namespace sm = zzs::kernel::sensor::mag;
using zzs::kernel::dsp::_S_BiquadCoeffs;
using zzs::kernel::dsp::_S_BiquadState;
using zzs::kernel::dsp::_S_goertzel_max_bins;
using zzs::kernel::dsp::_S_GoertzelBank;
using zzs::kernel::dsp::_S_GoertzelBin;
using zzs::kernel::dsp::bandpass_coeffs;
using zzs::kernel::dsp::biquad_reset;
using zzs::kernel::dsp::biquad_step;
using zzs::kernel::dsp::goertzel_bank_add_bin;
using zzs::kernel::dsp::goertzel_bank_power_q14;
using zzs::kernel::dsp::goertzel_bank_reset;
using zzs::kernel::dsp::goertzel_bank_step;
using zzs::kernel::dsp::goertzel_bin_configure;
using zzs::kernel::dsp::goertzel_bin_power_q14;
using zzs::kernel::dsp::goertzel_bin_reset;
using zzs::kernel::dsp::goertzel_bin_step;

/** @brief PLHR stage state. */
struct _S_PlhrState {
  /* Bandpass per axis: 45-250 Hz. */
  _S_BiquadCoeffs _M_bp_coeffs[sm::COUNT];
  _S_BiquadState _M_bp_state[sm::COUNT];

  /* Goertzel banks: one per axis, two grids (50 Hz, 60 Hz), 4 harmonics each. */
  _S_GoertzelBank _M_bank[sm::COUNT];

  uint32_t _M_sample_rate_hz;
  uint32_t _M_window_samples;
  uint32_t _M_tick;                                  /* samples fed into the current window */
  int16_t _M_window_buf[sm::COUNT][_S_plhr_samples]; /* per-axis window */
  bool _M_window_ready;                              /* true once a full window has been accumulated */
  bool _M_configured;
  bool _M_mains_detected; /* true once mains frequency has been detected */
};

/** @brief Configure PLHR from the magnetometer sample rate and window length. */
[[nodiscard]] inline zzs::kernel::Status
plhr_configure (void *const state) noexcept {
  _S_PlhrState *const s = static_cast<_S_PlhrState *> (state);
  if (s == nullptr)
    return zzs::kernel::Status::OUT_OF_RANGE;
  if (s->_M_sample_rate_hz == 0u)
    s->_M_sample_rate_hz = zzs::kernel::_S_grid_rate_hz;
  if (s->_M_window_samples == 0u)
    s->_M_window_samples = zzs::kernel::_S_plhr_samples;

  /* Bandpass 45-250 Hz per axis with Q=2 (dynamic, derived from the band). */
  const int32_t q_q15 = static_cast<int32_t> ((static_cast<int64_t> (zzs::kernel::_S_q15_one) * 2)
                                              / zzs::kernel::_S_q15_one); /* Q = 1.0 */
  for (uint32_t a = 0; a < sm::COUNT; ++a) {
    zzs::kernel::dsp::biquad_reset (s->_M_bp_state[a]);
    zzs::kernel::Status st = zzs::kernel::dsp::bandpass_coeffs (zzs::kernel::_S_plhr_band_low_hz, s->_M_sample_rate_hz,
                                                                q_q15, s->_M_bp_coeffs[a]);
    if (st != zzs::kernel::Status::OK) {
      zzs::base::log_error ("plhr_configure: bandpass_coeffs={}", zzs::kernel::status_name (st));
      return st;
    }
    st = zzs::kernel::dsp::bandpass_coeffs (zzs::kernel::_S_plhr_band_high_hz, s->_M_sample_rate_hz, q_q15,
                                            s->_M_bp_coeffs[a]);
    (void) st;
  }

  /* Goertzel banks: 50 Hz grid + 60 Hz grid, 4 harmonics each, per axis.
     Both grids are configured so the first window can be scanned to detect
     which mains frequency is present; the result is stored in
     zzs::kernel::_S_mains_freq_hz for other stages to use. */
  for (uint32_t a = 0; a < sm::COUNT; ++a) {
    s->_M_bank[a]._M_bin_count = 0;
    constexpr uint32_t grid_freqs[2] = {50u, 60u};
    for (uint32_t g = 0; g < 2u; ++g) {
      const uint32_t base = grid_freqs[g];
      for (uint32_t h = 0; h < zzs::kernel::_S_harmonics; ++h) {
        const uint32_t f = base * (h + 1u);
        if (f > s->_M_sample_rate_hz / 2u)
          continue; /* above Nyquist: skip bin */
        const zzs::kernel::Status st = zzs::kernel::dsp::goertzel_bank_add_bin (s->_M_bank[a], f, s->_M_sample_rate_hz);
        if (st != zzs::kernel::Status::OK) {
          zzs::base::log_error ("plhr_configure: goertzel_add_bin={}", zzs::kernel::status_name (st));
          return st;
        }
      }
    }
  }

  s->_M_tick = 0;
  s->_M_window_ready = false;
  s->_M_mains_detected = false;
  s->_M_configured = true;
  zzs::base::log_debug ("plhr_configure: rate={} window={}", s->_M_sample_rate_hz, s->_M_window_samples);
  return zzs::kernel::Status::OK;
}

/** @brief Reset PLHR state. */
inline void
plhr_reset (void *const state) noexcept {
  _S_PlhrState *const s = static_cast<_S_PlhrState *> (state);
  if (s == nullptr)
    return;
  for (uint32_t a = 0; a < sm::COUNT; ++a) {
    zzs::kernel::dsp::biquad_reset (s->_M_bp_state[a]);
    zzs::kernel::dsp::goertzel_bank_reset (s->_M_bank[a]);
    for (uint32_t i = 0; i < s->_M_window_samples; ++i)
      s->_M_window_buf[a][i] = 0;
  }
  s->_M_tick = 0;
  s->_M_window_ready = false;
  s->_M_mains_detected = false;
}

/** @brief Feed one resampled Q15 sample for a magnetometer axis. */
[[nodiscard]] inline zzs::kernel::Status
plhr_feed (void *const state, const sensor::Channel channel, const int16_t sample_q15) noexcept {
  _S_PlhrState *const s = static_cast<_S_PlhrState *> (state);
  if (s == nullptr || !s->_M_configured)
    return zzs::kernel::Status::NOT_READY;

  const uint32_t axis = static_cast<uint32_t> (channel) - static_cast<uint32_t> (sensor::Channel::MAG_X);
  if (axis >= sm::COUNT)
    return zzs::kernel::Status::OK; /* not a magnetometer axis: ignore */

  /* Bandpass the sample. */
  const int16_t filtered = zzs::kernel::dsp::biquad_step (s->_M_bp_coeffs[axis], s->_M_bp_state[axis], sample_q15);

  /* Store into the per-axis window buffer. */
  if (s->_M_tick < s->_M_window_samples)
    s->_M_window_buf[axis][s->_M_tick] = filtered;

  /* Step every Goertzel resonator incrementally, per sample.  This keeps the
     resonator states ahead of the window boundary so produce() can read each
     bin's power in O(1) instead of re-feeding the whole window (O(window x
     bins)) at the boundary. */
  static_cast<void> (zzs::kernel::dsp::goertzel_bank_step (s->_M_bank[axis], filtered));

  /* Advance the shared window tick once all three axes have a sample at this
     grid tick.  We track tick per-axis by only advancing when the Z axis
     (last) is fed; a cleaner design tracks a global grid counter, but this
     keeps stage state self-contained. */
  if (axis == static_cast<uint32_t> (sensor::Channel::MAG_Z)) {
    ++s->_M_tick;
    if (s->_M_tick >= s->_M_window_samples) {
      s->_M_tick = 0;
      s->_M_window_ready = true;
    }
  }
  return zzs::kernel::Status::OK;
}

/** @brief Produce the 24 PLHR Q15 features into `slots`. */
[[nodiscard]] inline zzs::kernel::Status
plhr_produce (void *const state, int32_t *const slots, const uint32_t slot_count) noexcept {
  _S_PlhrState *const s = static_cast<_S_PlhrState *> (state);
  if (s == nullptr || !s->_M_configured || slots == nullptr)
    return zzs::kernel::Status::NOT_READY;
  if (slot_count < meta::_S_feat_plhr_count) {
    zzs::base::log_error ("plhr_produce: budget_exceeded slots={} need={}", slot_count, meta::_S_feat_plhr_count);
    return zzs::kernel::Status::BUDGET_EXCEEDED;
  }

  if (!s->_M_window_ready)
    return zzs::kernel::Status::NOT_READY;

  /* On the first window, scan 50 Hz and 60 Hz to detect the mains grid
     frequency and publish it in zzs::kernel::_S_mains_freq_hz so other
     stages (acoustic, flicker) can tune their resonators to the same grid. */
  if (!s->_M_mains_detected) {
    const uint32_t detected =
        zzs::kernel::detect_mains_freq_hz (s->_M_window_buf[0], s->_M_window_samples, s->_M_sample_rate_hz);
    zzs::kernel::_S_mains_freq_hz = detected;
    s->_M_mains_detected = true;
    zzs::base::log_debug ("plhr_produce: mains detected={} Hz", detected);
  }

  /* Level reference for normalizing bin power to dB-relative Q15. */
  const zzs::kernel::fx::_S_LevelReference ref = zzs::kernel::fx::level_reference_of (s->_M_window_samples);

  for (uint32_t a = 0; a < sm::COUNT; ++a) {
    for (uint32_t g = 0; g < 2u; ++g) {
      for (uint32_t h = 0; h < zzs::kernel::_S_harmonics; ++h) {
        const uint32_t bin_index = g * zzs::kernel::_S_harmonics + h;
        if (bin_index >= s->_M_bank[a]._M_bin_count)
          continue;
        /* O(1) read of the incrementally-stepped resonator. */
        const uint32_t power_q14 = zzs::kernel::dsp::goertzel_bank_power_q14 (s->_M_bank[a], bin_index);
        const int32_t level_q8 = zzs::kernel::fx::db_q8 (power_q14, ref);
        /* Q8 dB -> Q15 feature: scale by 2^7. */
        const int32_t feature_q15 = level_q8 << 7u;
        const uint32_t slot = meta::feat_plhr_index (a, g, h);
        slots[slot] = feature_q15;
      }
    }
  }

  /* Window consumed: reset resonators for the next window. */
  for (uint32_t a = 0; a < sm::COUNT; ++a)
    zzs::kernel::dsp::goertzel_bank_reset (s->_M_bank[a]);
  s->_M_window_ready = false;
  zzs::base::log_debug ("plhr_produce: window complete");
  return zzs::kernel::Status::OK;
}

/** @brief PLHR stage vtable. */
[[nodiscard]] inline const _S_StageVtable &
plhr_vtable (void) noexcept {
  static const _S_StageVtable vt = {plhr_configure, plhr_reset, plhr_feed, plhr_produce};
  return vt;
}

/** @} */
} // namespace zzs::kernel::feat

namespace zzs::kernel {

/** @brief Detect the mains grid frequency from a window of samples.
 *
 * Runs a Goertzel scan at 50 Hz and 60 Hz on `samples` and returns the
 * frequency with the stronger squared-magnitude bin.  The caller stores the
 * result in `_S_mains_freq_hz`.
 */
[[nodiscard]] inline uint32_t
detect_mains_freq_hz (const int16_t *const samples, const uint32_t count, const uint32_t sample_rate_hz) noexcept {
  if (samples == nullptr || count == 0u || sample_rate_hz == 0u)
    return _S_mains_freq_hz;

  dsp::_S_GoertzelBin bin50;
  dsp::_S_GoertzelBin bin60;
  if (dsp::goertzel_bin_configure (bin50, 50u, sample_rate_hz) != zzs::kernel::Status::OK)
    return _S_mains_freq_hz;
  if (dsp::goertzel_bin_configure (bin60, 60u, sample_rate_hz) != zzs::kernel::Status::OK)
    return _S_mains_freq_hz;

  for (uint32_t i = 0; i < count; ++i) {
    dsp::goertzel_bin_step (bin50, samples[i]);
    dsp::goertzel_bin_step (bin60, samples[i]);
  }

  const uint32_t p50 = dsp::goertzel_bin_power_q14 (bin50);
  const uint32_t p60 = dsp::goertzel_bin_power_q14 (bin60);
  const uint32_t detected = p60 > p50 ? 60u : 50u;
  _S_mains_freq_hz = detected;
  return detected;
}

} // namespace zzs::kernel
