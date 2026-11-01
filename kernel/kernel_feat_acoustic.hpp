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
 * @brief Acoustic feature stage.
 *
 * Input: the sound envelope @ 200 Hz (from 16 kHz block-RMS).
 *   - sound_100hz_power: 100/120 Hz hum power (Goertzel).
 *   - sound_thd: hum THD = sqrt(sum of harmonic powers) / fundamental power.
 *   - sound_level_dB: broadband SPL (RMS relative to full-scale reference).
 *
 * Output: 3 Q15 features into slots [acoustic_start .. acoustic_start+2].
 *
 * No float, no hardcoded thresholds; the hum and harmonic frequencies are
 * derived from the configured mains grid.
 * @{
 */
namespace zzs::kernel::feat {

using zzs::kernel::dsp::_S_GoertzelBin;
using zzs::kernel::dsp::goertzel_bin_configure;
using zzs::kernel::dsp::goertzel_bin_power_q14;
using zzs::kernel::dsp::goertzel_bin_reset;
using zzs::kernel::dsp::goertzel_bin_step;
using zzs::kernel::fx::magnitude_q7;

/** @brief Acoustic stage state. */
struct _S_AcousticState {
  uint32_t _M_sample_rate_hz;
  uint32_t _M_window_samples;
  uint32_t _M_tick;
  int16_t _M_window_buf[_S_acoustic_samples];

  /* Hum (2*mains) and harmonic (h1/h2/h3) resonators, stepped incrementally
     in feed and read in produce. */
  _S_GoertzelBin _M_hum_bin;
  _S_GoertzelBin _M_h1_bin;
  _S_GoertzelBin _M_h2_bin;
  _S_GoertzelBin _M_h3_bin;

  bool _M_window_ready; /* true once a full window has been accumulated */
  bool _M_configured;
};

/** @brief Configure acoustic from the envelope sample rate. */
[[nodiscard]] inline zzs::kernel::Status
acoustic_configure (void *const state) noexcept {
  _S_AcousticState *const s = static_cast<_S_AcousticState *> (state);
  if (s == nullptr)
    return zzs::kernel::Status::OUT_OF_RANGE;
  if (s->_M_sample_rate_hz == 0u)
    s->_M_sample_rate_hz = zzs::kernel::_S_envelope_rate_hz;
  if (s->_M_window_samples == 0u)
    s->_M_window_samples = zzs::kernel::_S_acoustic_samples;

  /* Hum at 2*mains and the fundamental + 2nd/3rd harmonics for THD. */
  const zzs::kernel::Status st0 =
      zzs::kernel::dsp::goertzel_bin_configure (s->_M_hum_bin, zzs::kernel::_S_mains_low_hz * 2u, s->_M_sample_rate_hz);
  const zzs::kernel::Status st1 =
      zzs::kernel::dsp::goertzel_bin_configure (s->_M_h1_bin, zzs::kernel::_S_mains_low_hz, s->_M_sample_rate_hz);
  const zzs::kernel::Status st2 =
      zzs::kernel::dsp::goertzel_bin_configure (s->_M_h2_bin, zzs::kernel::_S_mains_low_hz * 2u, s->_M_sample_rate_hz);
  const zzs::kernel::Status st3 =
      zzs::kernel::dsp::goertzel_bin_configure (s->_M_h3_bin, zzs::kernel::_S_mains_low_hz * 3u, s->_M_sample_rate_hz);
  if (st0 != zzs::kernel::Status::OK) {
    zzs::base::log_error ("acoustic_configure: goertzel_bin_configure={}", zzs::kernel::status_name (st0));
    return st0;
  }
  (void) st1;
  (void) st2;
  (void) st3;

  s->_M_tick = 0;
  s->_M_window_ready = false;
  s->_M_configured = true;
  zzs::base::log_debug ("acoustic_configure: rate={} window={}", s->_M_sample_rate_hz, s->_M_window_samples);
  return zzs::kernel::Status::OK;
}

/** @brief Reset acoustic state. */
inline void
acoustic_reset (void *const state) noexcept {
  _S_AcousticState *const s = static_cast<_S_AcousticState *> (state);
  if (s == nullptr)
    return;
  for (uint32_t i = 0; i < s->_M_window_samples; ++i)
    s->_M_window_buf[i] = 0;
  zzs::kernel::dsp::goertzel_bin_reset (s->_M_hum_bin);
  zzs::kernel::dsp::goertzel_bin_reset (s->_M_h1_bin);
  zzs::kernel::dsp::goertzel_bin_reset (s->_M_h2_bin);
  zzs::kernel::dsp::goertzel_bin_reset (s->_M_h3_bin);
  s->_M_tick = 0;
  s->_M_window_ready = false;
}

/** @brief Feed one resampled Q15 sound-envelope sample. */
[[nodiscard]] inline zzs::kernel::Status
acoustic_feed (void *const state, const sensor::Channel channel, const int16_t sample_q15) noexcept {
  _S_AcousticState *const s = static_cast<_S_AcousticState *> (state);
  if (s == nullptr || !s->_M_configured)
    return zzs::kernel::Status::NOT_READY;
  if (channel != sensor::Channel::SOUND_ENVELOPE)
    return zzs::kernel::Status::OK;
  if (s->_M_tick < s->_M_window_samples)
    s->_M_window_buf[s->_M_tick] = sample_q15;

  /* Step all four resonators incrementally per sample. */
  zzs::kernel::dsp::goertzel_bin_step (s->_M_hum_bin, sample_q15);
  zzs::kernel::dsp::goertzel_bin_step (s->_M_h1_bin, sample_q15);
  zzs::kernel::dsp::goertzel_bin_step (s->_M_h2_bin, sample_q15);
  zzs::kernel::dsp::goertzel_bin_step (s->_M_h3_bin, sample_q15);

  ++s->_M_tick;
  if (s->_M_tick >= s->_M_window_samples) {
    s->_M_tick = 0;
    s->_M_window_ready = true;
  }
  return zzs::kernel::Status::OK;
}

/** @brief Produce the 3 acoustic Q15 features into `slots`. */
[[nodiscard]] inline zzs::kernel::Status
acoustic_produce (void *const state, int32_t *const slots, const uint32_t slot_count) noexcept {
  _S_AcousticState *const s = static_cast<_S_AcousticState *> (state);
  if (s == nullptr || !s->_M_configured || slots == nullptr)
    return zzs::kernel::Status::NOT_READY;
  if (slot_count < meta::_S_feat_acoustic_start + meta::_S_feat_acoustic_count) {
    zzs::base::log_error ("acoustic_produce: budget_exceeded slots={} need={}", slot_count,
                          meta::_S_feat_acoustic_start + meta::_S_feat_acoustic_count);
    return zzs::kernel::Status::BUDGET_EXCEEDED;
  }

  if (!s->_M_window_ready)
    return zzs::kernel::Status::NOT_READY;

  const uint32_t n = s->_M_window_samples;
  if (n == 0u)
    return zzs::kernel::Status::NOT_READY;

  /* sound_level_dB: broadband RMS of the envelope, relative to full scale. */
  const int32_t rms = zzs::kernel::fx::rms_q15 (s->_M_window_buf, n);
  const zzs::kernel::fx::_S_LevelReference ref = zzs::kernel::fx::level_reference_of (n);
  /* RMS^2 as Q14 power. */
  const int64_t rms_sq = static_cast<int64_t> (rms) * rms;
  const uint32_t rms_power_q14 = static_cast<uint32_t> (rms_sq > 0 ? rms_sq : 0);
  const int32_t level_db_q8 = zzs::kernel::fx::db_q8 (rms_power_q14, ref);
  slots[meta::_S_feat_acoustic_start + meta::acoustic::level_db] = level_db_q8 << 7u;

  /* sound_100hz_power: O(1) read of the incrementally-stepped hum resonator. */
  const uint32_t hum_power_q14 = zzs::kernel::dsp::goertzel_bin_power_q14 (s->_M_hum_bin);
  const int32_t hum_db_q8 = zzs::kernel::fx::db_q8 (hum_power_q14, ref);
  slots[meta::_S_feat_acoustic_start + meta::acoustic::power_100hz] = hum_db_q8 << 7u;

  /* sound_thd: THD = sqrt(h2^2 + h3^2) / h1.  O(1) reads plus a
     magic-reciprocal divide (replaces the runtime 64-bit / by fund_root). */
  const uint32_t p1 = zzs::kernel::dsp::goertzel_bin_power_q14 (s->_M_h1_bin);
  const uint32_t p2 = zzs::kernel::dsp::goertzel_bin_power_q14 (s->_M_h2_bin);
  const uint32_t p3 = zzs::kernel::dsp::goertzel_bin_power_q14 (s->_M_h3_bin);
  int32_t thd_q15 = 0;
  if (p1 != 0u) {
    const uint32_t harm_root = zzs::mllee::_S_isqrt (p2 + p3);
    const uint32_t fund_root = zzs::mllee::_S_isqrt (p1);
    if (fund_root != 0u) {
      const uint32_t num = static_cast<uint32_t> (static_cast<int64_t> (harm_root) << zzs::kernel::_S_state_frac_bits);
      const zzs::mllee::_S_Div32 div = zzs::mllee::_S_div32_make (fund_root);
      thd_q15 = static_cast<int32_t> (zzs::mllee::_S_div32_u (num, div));
    }
  }
  slots[meta::_S_feat_acoustic_start + meta::acoustic::thd] = thd_q15;

  /* Window consumed: reset resonators for the next window. */
  zzs::kernel::dsp::goertzel_bin_reset (s->_M_hum_bin);
  zzs::kernel::dsp::goertzel_bin_reset (s->_M_h1_bin);
  zzs::kernel::dsp::goertzel_bin_reset (s->_M_h2_bin);
  zzs::kernel::dsp::goertzel_bin_reset (s->_M_h3_bin);
  s->_M_window_ready = false;
  zzs::base::log_debug ("acoustic_produce: window complete");
  return zzs::kernel::Status::OK;
}

/** @brief Acoustic stage vtable. */
[[nodiscard]] inline const _S_StageVtable &
acoustic_vtable (void) noexcept {
  static constexpr _S_StageVtable vt = {acoustic_configure, acoustic_reset, acoustic_feed, acoustic_produce};
  return vt;
}

/** @} */
} // namespace zzs::kernel::feat
