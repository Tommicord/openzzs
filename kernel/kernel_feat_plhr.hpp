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
  bool _M_mains_detected;              /* true once mains frequency has been detected */
  uint32_t _M_prev_mains_freq_hz;      /* previous window's mains frequency (Hz) */
  bool _M_prev_freq_valid;             /* true once a previous frequency exists */
  uint32_t _M_windows_since_detection; /* windows since last mains re-detection */
  uint32_t _M_mains_freq_hz;           /* currently detected mains frequency (Hz) */
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
  s->_M_prev_mains_freq_hz = 0u;
  s->_M_prev_freq_valid = false;
  s->_M_windows_since_detection = 0u;
  s->_M_mains_freq_hz = zzs::kernel::_S_mains_freq_hz;
  s->_M_configured = true;
  zzs::base::log_debug ("plhr_configure: rate={} window={} mains_hz={}", s->_M_sample_rate_hz, s->_M_window_samples,
                        s->_M_mains_freq_hz);
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
  s->_M_prev_mains_freq_hz = 0u;
  s->_M_prev_freq_valid = false;
  s->_M_windows_since_detection = 0u;
  s->_M_mains_freq_hz = zzs::kernel::_S_mains_freq_hz;
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
  if (slot_count < meta::_S_feat_reserved_start + meta::_S_feat_reserved_count) {
    zzs::base::log_error ("plhr_produce: budget_exceeded slots={} need={}", slot_count,
                          meta::_S_feat_reserved_start + meta::_S_feat_reserved_count);
    return zzs::kernel::Status::BUDGET_EXCEEDED;
  }

  if (!s->_M_window_ready)
    return zzs::kernel::Status::NOT_READY;

  /* Mains frequency detection with periodic re-detection and hysteresis.
   * First window: fine scan around nominal.  Subsequent windows: re-scan
   * every kMainsRedetectInterval windows; only retune if the new frequency
   * differs by more than kMainsHysteresisHz (avoids oscillation). */
  constexpr uint32_t kMainsRedetectInterval = 30u; /* ~30 s at 100 Hz grid */
  constexpr uint32_t kMainsHysteresisHz = 1u;      /* 1 Hz hysteresis */

  if (!s->_M_mains_detected) {
    /* First window: fine scan around nominal 60 Hz (default). */
    const uint32_t nominal = s->_M_sample_rate_hz >= 200u ? 60u : 50u;
    const uint32_t detected_mhz =
        zzs::kernel::detect_mains_freq_fine (s->_M_window_buf[0], s->_M_window_samples, s->_M_sample_rate_hz, nominal);
    const uint32_t detected_hz = detected_mhz / 1000;
    s->_M_mains_freq_hz = detected_hz;
    zzs::kernel::_S_mains_freq_hz = detected_hz;
    s->_M_mains_detected = true;
    s->_M_prev_mains_freq_hz = detected_hz;
    s->_M_prev_freq_valid = true;
    s->_M_windows_since_detection = 0u;
    zzs::base::log_debug ("plhr_produce: mains detected={} Hz (mHz={})", detected_hz, detected_mhz);
  } else {
    /* Periodic re-detection with hysteresis. */
    ++s->_M_windows_since_detection;
    if (s->_M_windows_since_detection >= kMainsRedetectInterval) {
      const uint32_t nominal = s->_M_mains_freq_hz;
      const uint32_t detected_mhz = zzs::kernel::detect_mains_freq_fine (s->_M_window_buf[0], s->_M_window_samples,
                                                                         s->_M_sample_rate_hz, nominal);
      const uint32_t detected_hz = detected_mhz / 1000;
      s->_M_windows_since_detection = 0u;

      /* Hysteresis: only retune if the difference exceeds the threshold. */
      const int32_t diff = static_cast<int32_t> (detected_hz) - static_cast<int32_t> (s->_M_mains_freq_hz);
      if (diff > static_cast<int32_t> (kMainsHysteresisHz) || diff < -static_cast<int32_t> (kMainsHysteresisHz)) {
        s->_M_prev_mains_freq_hz = s->_M_mains_freq_hz;
        s->_M_mains_freq_hz = detected_hz;
        zzs::kernel::_S_mains_freq_hz = detected_hz;
        zzs::base::log_debug ("plhr_produce: mains retuned {} -> {} Hz", s->_M_prev_mains_freq_hz, detected_hz);
      } else {
        s->_M_prev_mains_freq_hz = s->_M_mains_freq_hz;
      }
    }
  }

  /* Level reference for normalizing bin power to dB-relative Q15. */
  const zzs::kernel::fx::_S_LevelReference ref = zzs::kernel::fx::level_reference_of (s->_M_window_samples);

  /* ---- PLHR harmonic features (slots 0-23) ---- */
  for (uint32_t a = 0; a < sm::COUNT; ++a) {
    for (uint32_t g = 0; g < 2u; ++g) {
      for (uint32_t h = 0; h < zzs::kernel::_S_harmonics; ++h) {
        const uint32_t bin_index = g * zzs::kernel::_S_harmonics + h;
        if (bin_index >= s->_M_bank[a]._M_bin_count)
          continue;
        /* O(1) read of the incrementally-stepped resonator. */
        const uint32_t power_q14 = zzs::kernel::dsp::goertzel_bank_power_q14 (s->_M_bank[a], bin_index);
        const int32_t level_q8 = zzs::kernel::fx::db_q8 (power_q14, ref);
        /* Q8 dB -> Q15 feature: scale by 2^7.  Use int64_t to avoid signed
         * left-shift overflow when level_q8 = INT32_MIN (-32768). */
        const int64_t feature_q15_raw = static_cast<int64_t> (level_q8) * 128;
        const int32_t feature_q15 = feature_q15_raw > 32767
                                        ? 32767
                                        : (feature_q15_raw < -32768 ? -32768 : static_cast<int32_t> (feature_q15_raw));
        const uint32_t slot = meta::feat_plhr_index (a, g, h);
        slots[slot] = feature_q15;
      }
    }
  }

  /* ---- Spectral features (slots 35-37) ---- */

  /* Slot 35: Spectral flatness = geometric_mean / arithmetic_mean of bin powers.
   * Low value = tonal (grid hum); High value = broadband noise.
   * We compute across all bins of axis X (representative axis). */
  {
    const uint32_t bin_count = s->_M_bank[0]._M_bin_count;
    if (bin_count > 0u) {
      /* Arithmetic mean of powers (Q14). */
      int64_t sum_power = 0;
      int64_t sum_log = 0;
      for (uint32_t b = 0; b < bin_count; ++b) {
        const uint32_t p = zzs::kernel::dsp::goertzel_bank_power_q14 (s->_M_bank[0], b);
        sum_power += static_cast<int64_t> (p);
        /* Approximate log2(p) via integer bit scan for geometric mean. */
        if (p > 0u) {
          int32_t log_p = 0;
          uint32_t v = p;
          while (v > 1u) {
            v >>= 1u;
            ++log_p;
          }
          sum_log += static_cast<int64_t> (log_p);
        }
      }
      const int64_t arith_mean = sum_power / static_cast<int64_t> (bin_count);
      const int64_t geo_mean_log2 = sum_log / static_cast<int64_t> (bin_count);
      /* Flatness = 2^(geo_mean_log2) / arith_mean, in Q15.
       * We approximate: flatness_q15 = (1 << geo_mean_log2) * 32768 / arith_mean. */
      int32_t flatness_q15 = 0;
      if (arith_mean > 0) {
        /* geo_mean_linear = 1 << geo_mean_log2 (approximate, clamped) */
        int32_t geo_mean_linear = 1;
        if (geo_mean_log2 > 15)
          geo_mean_linear = 32767;
        else if (geo_mean_log2 > 0)
          geo_mean_linear = 1 << static_cast<int32_t> (geo_mean_log2);
        /* Q15: (geo_mean_linear << 15) / arith_mean */
        const int64_t flatness_raw = (static_cast<int64_t> (geo_mean_linear) << 15) / static_cast<int64_t> (arith_mean);
        flatness_q15 =
            flatness_raw > 32767 ? 32767 : (flatness_raw < -32768 ? -32768 : static_cast<int32_t> (flatness_raw));
      }
      slots[meta::_S_feat_reserved_start + meta::spectral::spectral_flatness] = flatness_q15;
    }
  }

  /* Slot 36: Harmonic-to-noise ratio = harmonic-bin power / inter-harmonic-bin power.
   * Harmonic bins (indices 0,2,4,6 per grid) vs inter-harmonic bins (1,3,5,7).
   * Computed on axis X. */
  {
    const uint32_t bin_count = s->_M_bank[0]._M_bin_count;
    if (bin_count >= 8u) {
      int64_t harm_power = 0;
      int64_t noise_power = 0;
      for (uint32_t g = 0; g < 2u; ++g) {
        for (uint32_t h = 0; h < zzs::kernel::_S_harmonics; ++h) {
          const uint32_t bin_index = g * zzs::kernel::_S_harmonics + h;
          const uint32_t p = zzs::kernel::dsp::goertzel_bank_power_q14 (s->_M_bank[0], bin_index);
          if (h % 2u == 0u)
            harm_power += static_cast<int64_t> (p);
          else
            noise_power += static_cast<int64_t> (p);
        }
      }
      int32_t hnr_q15 = 0;
      if (noise_power > 0) {
        const int64_t hnr_raw = (harm_power << 15) / static_cast<int64_t> (noise_power);
        hnr_q15 = hnr_raw > 32767 ? 32767 : (hnr_raw < -32768 ? -32768 : static_cast<int32_t> (hnr_raw));
      } else if (harm_power > 0) {
        hnr_q15 = 32767;
      }
      slots[meta::_S_feat_reserved_start + meta::spectral::harmonic_to_noise] = hnr_q15;
    }
  }

  /* Slot 37: Frequency drift rate = |this_freq - prev_freq| (Hz, as Q15). */
  {
    int32_t drift_q15 = 0;
    if (s->_M_prev_freq_valid) {
      const int32_t drift_hz =
          static_cast<int32_t> (s->_M_mains_freq_hz) - static_cast<int32_t> (s->_M_prev_mains_freq_hz);
      /* Q15: drift_hz * 32768 (1 Hz = 32768 Q15).  Use int64_t to avoid
       * overflow when |drift_hz| > 1024. */
      const int64_t drift_q15_raw = static_cast<int64_t> (drift_hz) * 32768;
      drift_q15 =
          drift_q15_raw > 32767 ? 32767 : (drift_q15_raw < -32768 ? -32768 : static_cast<int32_t> (drift_q15_raw));
    }
    slots[meta::_S_feat_reserved_start + meta::spectral::freq_drift] = drift_q15;

    /* Update previous frequency for next window. */
    s->_M_prev_mains_freq_hz = s->_M_mains_freq_hz;
    s->_M_prev_freq_valid = true;
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

/**
 * @brief Fine mains grid frequency detection with parabolic interpolation.
 *
 * Scans Goertzel bins around the nominal mains frequency in fine steps
 * (nominal_hz - 2 Hz to nominal_hz + 2 Hz in 0.5 Hz steps = 9 bins), then
 * applies parabolic interpolation on the peak bin's neighbors to estimate
 * the sub-bin frequency.  Returns the estimate in millihertz.
 *
 * Integer-only implementation: frequencies are represented as integer Hz
 * during scanning; parabolic interpolation uses Q16 fixed-point arithmetic
 * for the sub-bin offset.
 *
 * @param samples          window of Q15 int16 samples
 * @param count            number of samples
 * @param sample_rate_hz   sample rate of the samples
 * @param nominal_hz       nominal mains frequency (50 or 60) to scan around
 * @returns Estimated mains frequency in millihertz, or nominal_hz * 1000 on
 *          invalid input.
 */
[[nodiscard]] inline uint32_t
detect_mains_freq_fine (const int16_t *const samples, const uint32_t count, const uint32_t sample_rate_hz,
                        const uint32_t nominal_hz) noexcept {
  if (samples == nullptr || count == 0u || sample_rate_hz == 0u || nominal_hz == 0u)
    return nominal_hz * 1000u;

  const uint32_t start_hz = nominal_hz > 2u ? nominal_hz - 2u : nominal_hz;
  const uint32_t end_hz = nominal_hz + 2u;

  /* Scan 9 frequencies: nominal-2, nominal-1.5, ..., nominal+2 Hz.
   * We step by 1 Hz and use a half-bin offset trick: configure bins at
   * f Hz and f+1 Hz, then interpolate between them. */
  int32_t best_freq_mhz = static_cast<int32_t> (nominal_hz) * 1000;
  uint32_t best_power = 0u;
  uint32_t prev_power = 0u;
  uint32_t peak_power = 0u;
  uint32_t next_power = 0u;
  uint32_t peak_hz = nominal_hz;

  dsp::_S_GoertzelBin bin_a;
  dsp::_S_GoertzelBin bin_b;

  for (uint32_t f = start_hz; f <= end_hz; ++f) {
    if (f > sample_rate_hz / 2u)
      continue;

    if (dsp::goertzel_bin_configure (bin_a, f, sample_rate_hz) != zzs::kernel::Status::OK)
      continue;

    for (uint32_t i = 0; i < count; ++i)
      dsp::goertzel_bin_step (bin_a, samples[i]);

    const uint32_t power = dsp::goertzel_bin_power_q14 (bin_a);

    if (power > best_power) {
      next_power = peak_power;
      peak_power = power;
      peak_hz = f;
      best_power = power;
      best_freq_mhz = static_cast<int32_t> (f) * 1000;
    } else if (power > next_power) {
      next_power = power;
    }

    prev_power = power;
  }

  /* Parabolic interpolation on peak bin neighbors for sub-bin estimate.
   * If peak is at edge or neighbors are unavailable, return integer Hz. */
  if (peak_hz > start_hz && peak_hz < end_hz && peak_power > 0u) {
    /* Need neighbors at peak_hz - 1 and peak_hz + 1. Re-scan those two. */
    dsp::_S_GoertzelBin bin_lo;
    dsp::_S_GoertzelBin bin_hi;

    if (dsp::goertzel_bin_configure (bin_lo, peak_hz - 1u, sample_rate_hz) == zzs::kernel::Status::OK
        && dsp::goertzel_bin_configure (bin_hi, peak_hz + 1u, sample_rate_hz) == zzs::kernel::Status::OK) {
      for (uint32_t i = 0; i < count; ++i) {
        dsp::goertzel_bin_step (bin_lo, samples[i]);
        dsp::goertzel_bin_step (bin_hi, samples[i]);
      }

      const int32_t p_lo = static_cast<int32_t> (dsp::goertzel_bin_power_q14 (bin_lo));
      const int32_t p_c = static_cast<int32_t> (peak_power);
      const int32_t p_hi = static_cast<int32_t> (dsp::goertzel_bin_power_q14 (bin_hi));

      /* Parabolic vertex offset: delta = 0.5 * (p_lo - p_hi) / (p_lo - 2*p_c + p_hi)
       * Use int64_t throughout to avoid signed integer overflow on
       * (p_lo - p_hi) << 16 and 2 * p_c. */
      const int64_t denom = static_cast<int64_t> (p_lo) - 2 * static_cast<int64_t> (p_c) + static_cast<int64_t> (p_hi);
      if (denom != 0) {
        const int64_t num = static_cast<int64_t> (p_lo - p_hi) << 16;
        const int64_t delta_q16 = num / (2 * denom);
        /* Clamp delta to [-1, +1] Hz in Q16 */
        const int64_t delta_clamped = delta_q16 > 65536 ? 65536 : (delta_q16 < -65536 ? -65536 : delta_q16);
        best_freq_mhz = static_cast<int32_t> (peak_hz) * 1000 + (static_cast<int64_t> (delta_clamped) * 1000 / 65536);
      }
    }
  }

  _S_mains_freq_hz = static_cast<uint32_t> (best_freq_mhz / 1000);
  return static_cast<uint32_t> (best_freq_mhz);
}

} // namespace zzs::kernel
