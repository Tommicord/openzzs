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
#include "kernel/kernel_kernel_fixed_simd.hpp"
#include "kernel/kernel_status.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Polyphase FIR filter engine: 16 branches x 8 taps max, Q15 coefficients.
 *
 * A rational resampler p/q decomposes a prototype low-pass FIR into `q` polyphase
 * branches, each holding `ceil(taps/q)` taps.  This module caps the design at
 * 16 branches and 8 taps per branch (128 Q15 coefficients total per resampler),
 * which matches the plan's budget.  The branch to use is selected by the
 * fractional phase of the input/output sample grid, so every output sample costs
 * exactly one dot product of 8 int16 samples against 8 int16 coefficients.
 *
 * The dot product is vectorized (`simd::dot_widen_q15`) on NEON/SSE2/AVX2/AVX-512
 * with a portable scalar fallback; every tier returns the same int64 accumulator.
 *
 * No heap, no float, no sorting.  Coefficients and the delay line live in caller
 * buffers; the engine itself is a small POD.
 * @{
 */
namespace zzs::kernel::dsp {

/** @brief Maximum polyphase branches (must be a power of two for the phase mask). */
inline constexpr uint32_t _S_poly_max_branches = 16u;
/** @brief Maximum taps per polyphase branch. */
inline constexpr uint32_t _S_poly_max_taps = 8u;
/** @brief Total Q15 coefficients per resampler (branches x taps). */
inline constexpr uint32_t _S_poly_max_coeffs = _S_poly_max_branches * _S_poly_max_taps;

/** @brief One polyphase branch: `tap_count` Q15 coefficients. */
struct _S_PolyBranch {
    int16_t _M_coeff_q15[_S_poly_max_taps];
    uint32_t _M_tap_count;
    int16_t _M_gain_q15; /* per-branch gain applied after the dot product (Q15) */
};

/** @brief Polyphase FIR state: branches plus a circular delay line.
 *
 * `_M_delay` is a ring of `_M_delay_len` samples; `_M_delay_idx` is the next
 * write slot, so the newest valid sample sits at `(_M_delay_idx - 1) &
 * (_M_delay_len - 1)`.  `_M_delay_len` must be a power of two so the wrap mask
 * is free (it is always configured to `_S_poly_max_taps = 8`).  Push and step
 * are both O(1): no shift-register loop.
 */
struct _S_PolyphaseFir {
    _S_PolyBranch _M_branch[_S_poly_max_branches];
    uint32_t _M_branch_count;
    int16_t _M_delay[_S_poly_max_taps]; /* circular delay line */
    uint32_t _M_delay_len;              /* power of two */
    uint32_t _M_delay_idx;              /* next write index, 0 .. delay_len-1 */
};

/** @brief Reset the delay line; branches keep their coefficients and gains. */
inline void
polyphase_reset (_S_PolyphaseFir &fir) noexcept {
    zzs::base::log_debug ("polyphase_reset: delay_len={}", fir._M_delay_len);
    for (uint32_t i = 0; i < fir._M_delay_len; ++i)
        fir._M_delay[i] = 0;
    fir._M_delay_idx = 0u;
}

/**
 * @brief Push one input sample into the circular delay line.
 *
 * The newest sample is written at `_M_delay_idx`, which then advances by one
 * (mod `_M_delay_len`).  The newest valid sample always sits at
 * `(_M_delay_idx - 1) & (_M_delay_len - 1)`, so `polyphase_step_branch` reads
 * taps with the same mask.  O(1): no shift-register loop.
 * @returns Status::OK, or OUT_OF_RANGE if the delay line is not configured.
 */
[[nodiscard]] inline zzs::kernel::Status
polyphase_push (_S_PolyphaseFir &fir, const int16_t sample_q15) noexcept {
    if (fir._M_delay_len == 0u || fir._M_delay_len > _S_poly_max_taps
        || (fir._M_delay_len & (fir._M_delay_len - 1u)) != 0u)
        return zzs::kernel::Status::OUT_OF_RANGE;
    fir._M_delay[fir._M_delay_idx] = sample_q15;
    fir._M_delay_idx = (fir._M_delay_idx + 1u) & (fir._M_delay_len - 1u);
    return zzs::kernel::Status::OK;
}

/**
 * @brief Load `tap_count` Q15 coefficients into branch `branch`, zeroing the
 *        remaining taps and setting the branch gain to unity.
 * @returns Status::OK, OUT_OF_RANGE on a bad branch index or tap count.
 */
[[nodiscard]] inline zzs::kernel::Status
polyphase_load_branch (_S_PolyphaseFir &fir, const uint32_t branch, const int16_t *const ZZS_KERNEL_RESTRICT coeff_q15,
                       const uint32_t tap_count) noexcept {
    if (branch >= fir._M_branch_count || coeff_q15 == nullptr || tap_count == 0u || tap_count > _S_poly_max_taps) {
        zzs::base::log_error ("polyphase_load_branch: branch={} taps={} -> {}", branch, tap_count,
                              zzs::kernel::status_name (zzs::kernel::Status::OUT_OF_RANGE));
        return zzs::kernel::Status::OUT_OF_RANGE;
    }
    _S_PolyBranch &br = fir._M_branch[branch];
    br._M_tap_count = tap_count;
    for (uint32_t t = 0; t < tap_count; ++t)
        br._M_coeff_q15[t] = coeff_q15[t];
    for (uint32_t t = tap_count; t < _S_poly_max_taps; ++t)
        br._M_coeff_q15[t] = 0;
    br._M_gain_q15 = zzs::kernel::_S_q15_one_i16;
    zzs::base::log_debug ("polyphase_load_branch: ok branch={} taps={}", branch, tap_count);
    return zzs::kernel::Status::OK;
}

/**
 * @brief Set the per-branch gain (Q15).  Applied after the dot product in
 *        `polyphase_step_branch` so a branch can be attenuated without
 *        re-loading its taps.
 */
[[nodiscard]] inline zzs::kernel::Status
polyphase_set_branch_gain (_S_PolyphaseFir &fir, const uint32_t branch, const int16_t gain_q15) noexcept {
    if (branch >= fir._M_branch_count)
        return zzs::kernel::Status::OUT_OF_RANGE;
    fir._M_branch[branch]._M_gain_q15 = gain_q15;
    return zzs::kernel::Status::OK;
}

/**
 * @brief Compute one output sample on branch `branch` (0 <= branch < branch_count).
 *
 * Output = dot(delay[0..tap_count-1], branch.coeff[0..tap_count-1]) in Q15,
 * saturating, then multiplied by the branch gain (Q15 x Q15 -> Q15 saturating).
 * The delay line holds the newest sample first, matching the coefficient order,
 * so no reversal is needed.
 */
[[nodiscard]] inline int16_t
polyphase_step_branch (const _S_PolyphaseFir &fir, const uint32_t branch) noexcept {
    const _S_PolyBranch &br = fir._M_branch[branch];
    if (br._M_tap_count == 0u || br._M_tap_count > fir._M_delay_len)
        return 0;
    /* Gather the tap_count newest samples into logical order (newest first).
       _S_poly_max_taps is 8, so this fixed-size gather is O(1) and feeds the
       vectorized dot product a contiguous buffer. */
    const uint32_t mask = fir._M_delay_len - 1u;
    const uint32_t newest = (fir._M_delay_idx + mask) & mask; /* (idx - 1) & mask */
    int16_t linear[_S_poly_max_taps];
    for (uint32_t t = 0; t < br._M_tap_count; ++t)
        linear[t] = fir._M_delay[(newest + fir._M_delay_len - t) & mask];
    const int64_t acc = zzs::kernel::simd::dot_widen_q15 (linear, br._M_coeff_q15, br._M_tap_count);
    /* dot_widen_q15 returns Q30 (sum of Q15xQ15 products).  Round-half-up to Q15
       before applying the branch gain; plain `>> 15` truncates toward -inf and
       biases positives down by 1 LSB, which breaks exact passthrough. */
    const int32_t out_q15 = zzs::kernel::fx::_S_round_shift (acc, zzs::kernel::_S_state_frac_bits);
    int32_t out = zzs::kernel::fx::q15_mul (out_q15, static_cast<int32_t> (br._M_gain_q15));
    return static_cast<int16_t> (out);
}

/** @} */
} // namespace zzs::kernel::dsp
