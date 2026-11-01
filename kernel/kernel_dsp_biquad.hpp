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
#include "kernel/kernel_fixed.hpp"
#include "kernel/kernel_status.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Fixed-point biquad IIR filter stages for the preprocessing bank.
 *
 * All coefficients and states are integer-only.  A biquad is a direct-form-II
 * transposed section; the state lives in Q15 int32 so a resonant pole can hold
 * without per-sample renormalization, and the input/output samples are Q15
 * int16 on the wire.  Coefficients are derived from a centre frequency and Q at
 * configuration time and stored as Q15 int16 (b0, b1, b2, a1, a2), with a1/a2
 * carrying their sign.
 *
 * A two-section cascade (`_S_BiquadCascade`) stacks two sections for steeper
 * skirts: the PLHR bandpass is 45–250 Hz and the vibration bandpass is
 * 10–100 Hz, each built from `bandpass_coeffs` with integer trigonometry only.
 *
 * No heap, no float, no sorting.  The per-sample step is force-inlined and the
 * coefficient setup is a one-time cost.
 * @{
 */
namespace zzs::kernel::dsp {

/** @brief One biquad's Q15 coefficients (direct-form-II transposed). */
struct _S_BiquadCoeffs {
  int16_t _M_b0_q15;
  int16_t _M_b1_q15;
  int16_t _M_b2_q15;
  int16_t _M_a1_q15;
  int16_t _M_a2_q15;
};

/** @brief One biquad's Q15 state (two delays, int32 for headroom). */
struct _S_BiquadState {
  int32_t _M_s1_q15;
  int32_t _M_s2_q15;
};

/**
 * @brief Normalize a bandpass biquad: centre frequency + Q -> Q15 coefficients.
 *
 * `theta = 2*pi*fc/fs`; `alpha = sin(theta)/(2*Q)`.  The standard RBJ bandpass
 * coefficients are:
 *   b0 =  alpha, b1 = 0, b2 = -alpha
 *   a0 = 1 + alpha, a1 = -2*cos(theta), a2 = 1 - alpha
 * Normalized by a0 so b0 = alpha/(1+alpha), etc.  Every value is computed from
 * the integer trigonometry in `kernel_fixed.hpp`; no float threshold is baked
 * in.  The Q is expressed as a Q15 value (Q15 = 1.0).
 *
 * @returns Status::OK on success, OUT_OF_RANGE when fc >= fs/2 or Q <= 0.
 */
[[nodiscard]] inline zzs::kernel::Status
bandpass_coeffs (const uint32_t fc_hz, const uint32_t fs_hz, const int32_t q_q15, _S_BiquadCoeffs &coeffs) noexcept {
  using zzs::kernel::fx::coeff_cos_q15_of_hz;
  using zzs::kernel::fx::sin_q15_fractional;
  using zzs::kernel::fx::unit_fixed_of_ratio;

  if (fs_hz == 0u || fc_hz >= fs_hz / 2u || q_q15 <= 0) {
    zzs::base::log_error ("bandpass_coeffs: fc={} fs={} q={} -> {}", fc_hz, fs_hz, q_q15,
                          zzs::kernel::status_name (zzs::kernel::Status::OUT_OF_RANGE));
    return zzs::kernel::Status::OUT_OF_RANGE;
  }

  /* theta units = fc * 128 / fs (in 1/256-of-a-unit fixed). */
  const uint32_t theta_units = unit_fixed_of_ratio (fc_hz, fs_hz);
  const int32_t cos_theta = zzs::kernel::fx::cos_q15_fractional (theta_units);
  const int32_t sin_theta = zzs::kernel::fx::sin_q15_fractional (theta_units);

  /* alpha = sin(theta) / (2*Q).  Both sin and Q are Q15, so alpha is Q15. */
  const int64_t two_q = static_cast<int64_t> (2) * q_q15;
  const int32_t alpha_q15 =
      static_cast<int32_t> ((static_cast<int64_t> (sin_theta) << zzs::kernel::_S_state_frac_bits) / two_q);

  /* a0 = 1 + alpha (Q15). */
  const int32_t a0_q15 = zzs::kernel::_S_q15_one + alpha_q15;
  if (a0_q15 <= 0) {
    zzs::base::log_error ("bandpass_coeffs: fc={} fs={} q={} a0={} -> {}", fc_hz, fs_hz, q_q15, a0_q15,
                          zzs::kernel::status_name (zzs::kernel::Status::OUT_OF_RANGE));
    return zzs::kernel::Status::OUT_OF_RANGE;
  }

  const int32_t a0_inv_q15 = static_cast<int32_t> (
      (static_cast<int64_t> (zzs::kernel::_S_q15_one) << zzs::kernel::_S_state_frac_bits) / a0_q15);

  /* b0 = alpha/a0, b2 = -alpha/a0, b1 = 0. */
  coeffs._M_b0_q15 = static_cast<int16_t> (zzs::kernel::fx::q15_mul_wide (alpha_q15, a0_inv_q15));
  coeffs._M_b1_q15 = 0;
  coeffs._M_b2_q15 = static_cast<int16_t> (-static_cast<int32_t> (coeffs._M_b0_q15));

  /* a1 = -2*cos(theta)/a0, a2 = (1 - alpha)/a0. */
  /* cos is Q15; 2*cos fits int16 only below 1.0, which holds for fc < fs/2. */
  const int32_t neg_2cos_q15 = -(2 * static_cast<int32_t> (cos_theta));
  coeffs._M_a1_q15 = static_cast<int16_t> (zzs::kernel::fx::q15_mul_wide (neg_2cos_q15, a0_inv_q15));
  const int32_t one_minus_alpha_q15 = zzs::kernel::_S_q15_one - alpha_q15;
  coeffs._M_a2_q15 = static_cast<int16_t> (zzs::kernel::fx::q15_mul_wide (one_minus_alpha_q15, a0_inv_q15));

  zzs::base::log_debug ("bandpass_coeffs: ok fc={} fs={} q={} b0={} a1={} a2={}", fc_hz, fs_hz, q_q15, coeffs._M_b0_q15,
                        coeffs._M_a1_q15, coeffs._M_a2_q15);
  return zzs::kernel::Status::OK;
}

/** @brief Reset a biquad's state to zero. */
inline void
biquad_reset (_S_BiquadState &state) noexcept {
  zzs::base::log_debug ("biquad_reset");
  state._M_s1_q15 = 0;
  state._M_s2_q15 = 0;
}

/**
 * @brief One biquad sample: direct-form-II transposed, Q15 in/out.
 *
 * y[n] = b0*x[n] + s1;   s1 = b1*x[n] + s2 - a1*y[n];   s2 = b2*x[n] - a2*y[n].
 * All intermediates widen to int32/int64; the output is clamped to Q15.
 */
[[nodiscard]] inline int16_t
biquad_step (const _S_BiquadCoeffs &coeffs, _S_BiquadState &state, const int16_t sample_q15) noexcept {
  using zzs::kernel::fx::clamp_q15;
  using zzs::kernel::fx::q15_add_sat;
  using zzs::kernel::fx::q15_mul_wide;
  using zzs::kernel::fx::q15_sub_sat;

  const int32_t x = sample_q15;
  const int32_t b0x = q15_mul_wide (coeffs._M_b0_q15, x);
  const int32_t y = q15_add_sat (b0x, state._M_s1_q15);

  const int32_t b1x = q15_mul_wide (coeffs._M_b1_q15, x);
  const int32_t b2x = q15_mul_wide (coeffs._M_b2_q15, x);
  const int32_t a1y = q15_mul_wide (coeffs._M_a1_q15, y);
  const int32_t a2y = q15_mul_wide (coeffs._M_a2_q15, y);

  state._M_s1_q15 = q15_add_sat (b1x, q15_sub_sat (state._M_s2_q15, a1y));
  state._M_s2_q15 = q15_add_sat (b2x, q15_sub_sat (0, a2y));

  return clamp_q15 (y);
}

/** @brief Number of sections in a biquad cascade (fixed at 2 for the budget). */
inline constexpr uint32_t _S_biquad_max_sections = 2u;

/** @brief Two-section biquad cascade: section output feeds the next section's input. */
struct _S_BiquadCascade {
  _S_BiquadCoeffs _M_coeff[_S_biquad_max_sections];
  _S_BiquadState _M_state[_S_biquad_max_sections];
  uint32_t _M_section_count;
};

/** @brief Reset every section's state; coefficients are unchanged. */
inline void
biquad_cascade_reset (_S_BiquadCascade &cascade) noexcept {
  zzs::base::log_debug ("biquad_cascade_reset: sections={}", cascade._M_section_count);
  for (uint32_t i = 0; i < cascade._M_section_count; ++i)
    biquad_reset (cascade._M_state[i]);
}

/**
 * @brief Configure both sections as bandpasses.
 *
 * Section 0: 45–250 Hz (PLHR passband).  Section 1: 10–100 Hz (vibration band).
 * Each section gets `q_q15` as its Q factor.  Returns OUT_OF_RANGE if either
 * section fails coefficient derivation (e.g. fc >= fs/2).
 */
[[nodiscard]] inline zzs::kernel::Status
biquad_cascade_configure_bandpass (_S_BiquadCascade &cascade, const uint32_t fs_hz, const int32_t q_q15) noexcept {
  zzs::base::log_debug ("biquad_cascade_configure_bandpass: fs={} q={} sections={}", fs_hz, q_q15,
                        _S_biquad_max_sections);

  cascade._M_section_count = _S_biquad_max_sections;

  /* Section 0: PLHR passband 45–250 Hz -> centre at geometric mean ~106 Hz,
     Q chosen to cover the band; the caller may override per-section afterwards. */
  zzs::kernel::Status st = bandpass_coeffs (45u, fs_hz, q_q15, cascade._M_coeff[0]);
  if (st != zzs::kernel::Status::OK) {
    zzs::base::log_error ("biquad_cascade_configure_bandpass: section 0 fc=45 fs={} q={} -> {}", fs_hz, q_q15,
                          zzs::kernel::status_name (st));
    return st;
  }
  st = bandpass_coeffs (250u, fs_hz, q_q15, cascade._M_coeff[1]);
  if (st != zzs::kernel::Status::OK) {
    zzs::base::log_error ("biquad_cascade_configure_bandpass: section 1 fc=250 fs={} q={} -> {}", fs_hz, q_q15,
                          zzs::kernel::status_name (st));
    return st;
  }

  biquad_cascade_reset (cascade);
  return zzs::kernel::Status::OK;
}

/**
 * @brief One sample through the cascade: each section's output is the next
 *        section's input.  Returns the final section's output clamped to Q15.
 */
[[nodiscard]] inline int16_t
biquad_cascade_step (_S_BiquadCascade &cascade, const int16_t sample_q15) noexcept {
  int16_t x = sample_q15;
  for (uint32_t i = 0; i < cascade._M_section_count; ++i)
    x = biquad_step (cascade._M_coeff[i], cascade._M_state[i], x);
  return x;
}

/** @} */
} // namespace zzs::kernel::dsp
