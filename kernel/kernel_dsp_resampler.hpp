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
#include "kernel/kernel_dsp_polyphase.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Rational p/q resampler built on the polyphase FIR engine.
 *
 * Converts a stream at `in_rate_hz` to `out_rate_hz` by reducing the ratio to
 * p/q (p = out_rate/gcd, q = in_rate/gcd) and selecting one of `q` polyphase
 * branches per output sample according to the accumulated phase.  Each branch
 * holds up to 8 taps, so one output sample costs exactly one 8-tap dot product.
 *
 * Common configurations from the plan:
 *   - 1 Hz -> 100 Hz (light, battery): p = 100, q = 1, one branch (identity-ish
 *     with a hold/interpolate branch set at configuration time).
 *   - 200 Hz -> 100 Hz (mag/accel down to grid): p = 1, q = 2, decimation by 2.
 *   - 200 Hz -> 200 Hz (sound envelope): p = 1, q = 1, passthrough branch.
 *
 * Output-driven model: the phase advances by `q` per output sample and wraps mod
 * `p`; whenever it wraps a new input sample is consumed.  This is the standard
 * rational resampler and needs no floating-point accumulator.
 *
 * No heap, no float, no sorting.  Coefficients are configured once; the resampler
 * state is a small POD.
 * @{
 */
namespace zzs::kernel::dsp {

/** @brief Rational resampler state. */
struct _S_Resampler {
  _S_PolyphaseFir _M_fir;
  uint32_t _M_p;              /* output rate / gcd (phase modulus) */
  uint32_t _M_q;              /* input rate / gcd (phase step per output) */
  uint32_t _M_phase;          /* 0 .. p-1, advances by q per output sample */
  uint32_t _M_input_consumed; /* number of input samples fed so far */
  bool _M_configured;
  bool _M_linear_mode;       /* linear-interpolation branch mode */
  int16_t _M_prev_input_q15; /* input consumed at previous wrap */
  int16_t _M_curr_input_q15; /* input consumed at current/most recent wrap */
};

/** @brief Greatest common divisor, iterative, no recursion. */
[[nodiscard]] inline constexpr uint32_t
_S_gcd_u32 (uint32_t a, uint32_t b) noexcept {
  while (b != 0u) {
    const uint32_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

/** @brief Reset phase and delay line; keep coefficients. */
inline void
resampler_reset (_S_Resampler &res) noexcept {
  zzs::base::log_debug ("resampler_reset: p={} q={}", res._M_p, res._M_q);
  res._M_phase = 0;
  res._M_input_consumed = 0;
  res._M_prev_input_q15 = 0;
  res._M_curr_input_q15 = 0;
  zzs::kernel::dsp::polyphase_reset (res._M_fir);
}

/**
 * @brief Configure the resampler for in_rate_hz -> out_rate_hz using a single
 *        polyphase branch set to a flat (nearest-neighbor / hold) response.
 *
 * A flat branch with a single tap of gain 1.0 (Q15 one) reproduces the input
 * sample at the selected phase, which is the correct behavior for
 * 1 Hz -> 100 Hz (each input sample is held across 100 grid ticks) and for
 * passthrough.  Decimation (200 Hz -> 100 Hz) uses q = 2 with both branches set
 * to gain 1/2 (averaging the two input samples per output).
 *
 * Callers that want a sharper prototype FIR overwrite `_M_fir._M_branch`
 * coefficients directly after calling this.
 *
 * @returns Status::OK, UNSUPPORTED when the branch count would exceed 16.
 */
[[nodiscard]] inline zzs::kernel::Status
resampler_configure (_S_Resampler &res, const uint32_t in_rate_hz, const uint32_t out_rate_hz) noexcept {
  if (in_rate_hz == 0u || out_rate_hz == 0u) {
    zzs::base::log_error ("resampler_configure: in={} out={} -> {}", in_rate_hz, out_rate_hz,
                          zzs::kernel::status_name (zzs::kernel::Status::OUT_OF_RANGE));
    return zzs::kernel::Status::OUT_OF_RANGE;
  }
  const uint32_t g = _S_gcd_u32 (in_rate_hz, out_rate_hz);
  res._M_p = out_rate_hz / g;
  res._M_q = in_rate_hz / g;
  if (res._M_q > _S_poly_max_branches) {
    zzs::base::log_error ("resampler_configure: in={} out={} q={} -> {}", in_rate_hz, out_rate_hz, res._M_q,
                          zzs::kernel::status_name (zzs::kernel::Status::UNSUPPORTED));
    return zzs::kernel::Status::UNSUPPORTED;
  }

  res._M_fir._M_branch_count = res._M_q;
  res._M_fir._M_delay_len = _S_poly_max_taps;

  for (uint32_t b = 0; b < res._M_q; ++b) {
    _S_PolyBranch &br = res._M_fir._M_branch[b];
    br._M_tap_count = 1u;
    br._M_coeff_q15[0] = zzs::kernel::_S_q15_one_i16;
    for (uint32_t t = 1; t < _S_poly_max_taps; ++t)
      br._M_coeff_q15[t] = 0;
    br._M_gain_q15 = zzs::kernel::_S_q15_one_i16;
  }

  res._M_linear_mode = false;
  res._M_prev_input_q15 = 0;
  res._M_curr_input_q15 = 0;

  resampler_reset (res);
  res._M_configured = true;
  zzs::base::log_debug ("resampler_configure: ok in={} out={} p={} q={} branches={}", in_rate_hz, out_rate_hz, res._M_p,
                        res._M_q, res._M_q);
  return zzs::kernel::Status::OK;
}

/**
 * @brief Enable or disable the linear-interpolation branch mode.
 *
 * In linear mode, `resampler_pull` produces each output by linearly
 * interpolating between the previous and current input sample at the
 * fractional phase `phase/p`, instead of evaluating a polyphase branch.
 * This is the low-cost path for 1 Hz -> 100 Hz light/battery channels where a
 * full FIR is unnecessary.
 */
inline void
resampler_set_linear_mode (_S_Resampler &res, const bool enable) noexcept {
  res._M_linear_mode = enable;
}

/**
 * @brief Set one branch's tap coefficients (caller supplies `tap_count` Q15
 *        values, <= 8).  Used after configuration to install a real prototype
 *        FIR instead of the flat hold response.
 */
[[nodiscard]] inline zzs::kernel::Status
resampler_set_branch (_S_Resampler &res, const uint32_t branch, const int16_t *const ZZS_KERNEL_RESTRICT coeff_q15,
                      const uint32_t tap_count) noexcept {
  if (!res._M_configured || branch >= res._M_q || tap_count == 0u || tap_count > _S_poly_max_taps) {
    zzs::base::log_error ("resampler_set_branch: branch={} taps={} -> {}", branch, tap_count,
                          zzs::kernel::status_name (zzs::kernel::Status::OUT_OF_RANGE));
    return zzs::kernel::Status::OUT_OF_RANGE;
  }
  _S_PolyBranch &br = res._M_fir._M_branch[branch];
  br._M_tap_count = tap_count;
  for (uint32_t t = 0; t < tap_count; ++t)
    br._M_coeff_q15[t] = coeff_q15[t];
  for (uint32_t t = tap_count; t < _S_poly_max_taps; ++t)
    br._M_coeff_q15[t] = 0;
  zzs::base::log_debug ("resampler_set_branch: ok branch={} taps={}", branch, tap_count);
  return zzs::kernel::Status::OK;
}

/**
 * @brief Feed one input sample (input-driven).  Returns Status::OK and sets
 *        *produced when an output sample is available on the current phase;
 *        otherwise *produced = false and no output exists yet.
 *
 * The phase advances by `p` mod `q` per call; when it wraps to 0 the input is
 * consumed and the branch at phase 0 is evaluated.  A caller loops:
 *   for each input sample: resampler_push(res, s, &produced, &out);
 *     if (produced) consume(out);
 */
[[nodiscard]] inline zzs::kernel::Status
resampler_push (_S_Resampler &res, const int16_t sample_q15, bool *const produced, int16_t *const out) noexcept {
  if (!res._M_configured)
    return zzs::kernel::Status::OUT_OF_RANGE;
  *produced = false;
  *out = 0;

  _S_PolyphaseFir &fir = res._M_fir;
  _S_PolyBranch &branch = fir._M_branch[res._M_phase];

  /* Push the sample into the delay line before evaluating this phase. */
  static_cast<void> (polyphase_push (fir, sample_q15));

  /* Advance phase; a wrap to 0 means this input sample is now "current" and
     the branch at phase 0 produces an output. */
  const uint32_t next_phase = (res._M_phase + res._M_p) % res._M_q;
  const bool wrapped = (next_phase == 0u);
  res._M_phase = next_phase;
  if (wrapped)
    ++res._M_input_consumed;

  if (wrapped) {
    *out = polyphase_step_branch (fir, 0u);
    *produced = true;
  }
  return zzs::kernel::Status::OK;
}

/**
 * @brief Produce one output sample (output-driven).  The phase advances by `q`
 *        per output sample and wraps mod `p`; whenever it wraps a new input
 *        sample is consumed from `next_input` (when `have_input` is true).
 *        Sets *produced when an output sample is available.
 *
 * This is the natural loop for a downstream that requests output at a fixed
 * rate:
 *   for each output sample needed:
 *     resampler_pull(res, next_input, &have_input, &produced, &out);
 *       if (produced) consume(out);
 *
 * In linear-interpolation mode the output is the linear blend of the previous
 * and current input at fractional phase `phase/p`, in Q15:
 *   out = prev + q15_mul((phase << 15) / p, curr - prev)
 * No float is used; the division by p is exact since p <= 65535 in practice.
 */
[[nodiscard]] inline zzs::kernel::Status
resampler_pull (_S_Resampler &res, const int16_t next_input, bool *const have_input, bool *const produced,
                int16_t *const out) noexcept {
  if (!res._M_configured)
    return zzs::kernel::Status::OUT_OF_RANGE;
  *produced = false;
  *out = 0;

  /* Advance phase by q per output sample; count how many inputs we consume. */
  const uint64_t phase_new = static_cast<uint64_t> (res._M_phase) + res._M_q;
  const uint32_t wrapped_count = static_cast<uint32_t> (phase_new / res._M_p);
  res._M_phase = static_cast<uint32_t> (phase_new % res._M_p);

  if (wrapped_count > 0u && *have_input) {
    res._M_prev_input_q15 = res._M_curr_input_q15;
    res._M_curr_input_q15 = next_input;
    res._M_input_consumed += wrapped_count;
  }

  const int32_t curr = *have_input ? res._M_curr_input_q15 : res._M_prev_input_q15;
  const int32_t prev = res._M_prev_input_q15;

  if (res._M_linear_mode) {
    /* frac = phase / p in Q15: (phase << 15) / p.
       out = prev + q15_mul(frac_q15, curr - prev). */
    const int32_t frac_q15 =
        static_cast<int32_t> ((static_cast<int64_t> (res._M_phase) << zzs::kernel::_S_state_frac_bits) / res._M_p);
    const int32_t diff = curr - prev;
    const int32_t interp = prev + zzs::kernel::fx::q15_mul (frac_q15, diff);
    *out = static_cast<int16_t> (zzs::kernel::fx::clamp_q15 (interp));
  } else {
    if (*have_input) {
      static_cast<void> (polyphase_push (res._M_fir, next_input));
      res._M_curr_input_q15 = next_input;
    }
    /* Select the branch for the current fractional phase.  The phase
       is in [0, p); map it to a branch index in [0, q). */
    const uint32_t branch = (res._M_phase * res._M_q) / res._M_p;
    *out = polyphase_step_branch (res._M_fir, branch);
  }

  if (*have_input)
    *produced = true;

  return zzs::kernel::Status::OK;
}

/** @} */
} // namespace zzs::kernel::dsp
