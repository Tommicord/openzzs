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
 * @brief Goertzel resonator for exact-frequency bin detection.
 *
 * Computes the squared magnitude at a single frequency bin over a window of
 * `window_samples` Q15 int16 samples using the resonator recursion
 *   s0 = 2*cos(theta)*s1 - s2 + x
 * with all state in Q7 int16.  The resonator is the preferred spectral path
 * because it works for any window length and any frequency (the 50/60 Hz
 * harmonic bins at 200 Hz do not land on integer FFT bins, so a 256-point FFT
 * would need interpolation; Goertzel hits them exactly).
 *
 * Per-sample cost is one saturating multiply-add; the window's magnitude is
 * extracted once at the end via `resonator_power` (Q14) then `magnitude_q7`.
 * No heap, no float, no sorting.
 * @{
 */
namespace zzs::kernel::dsp {

/** @brief One Goertzel bin resonator: coefficient plus two Q7 state delays. */
struct _S_GoertzelBin {
        int16_t _M_cosine_q15; /* cos(theta), Q15 (2*cos supplied by the doubling MAC) */
        int16_t _M_s1_q7;      /* newest resonator state, Q7 */
        int16_t _M_s2_q7;      /* previous resonator state, Q7 */
};

/** @brief Configure a bin for `frequency_hz` at `sample_rate_hz`. */
[[nodiscard]] inline zzs::kernel::Status
goertzel_bin_configure (_S_GoertzelBin &bin, const uint32_t frequency_hz, const uint32_t sample_rate_hz) noexcept {
        if (sample_rate_hz == 0u) {
                zzs::base::log_error ("goertzel_bin_configure: freq={} rate={} -> {}", frequency_hz, sample_rate_hz,
                                      zzs::kernel::status_name (zzs::kernel::Status::OUT_OF_RANGE));
                return zzs::kernel::Status::OUT_OF_RANGE;
        }
        zzs::base::log_debug ("goertzel_bin_configure: freq={} rate={}", frequency_hz, sample_rate_hz);
        bin._M_cosine_q15 = zzs::kernel::fx::coeff_cos_q15_of_hz (frequency_hz, sample_rate_hz);
        bin._M_s1_q7 = 0;
        bin._M_s2_q7 = 0;
        return zzs::kernel::Status::OK;
}

/** @brief Reset resonator state without changing the coefficient. */
inline void
goertzel_bin_reset (_S_GoertzelBin &bin) noexcept {
        zzs::base::log_debug ("goertzel_bin_reset");
        bin._M_s1_q7 = 0;
        bin._M_s2_q7 = 0;
}

/**
 * @brief Feed one Q15 sample into the resonator.
 *
 * The sample is converted Q15 -> Q7 and the recursion advances.  Force-inlined;
 * this is the hot path called once per sample per bin.
 */
ZZS_KERNEL_INLINE void
goertzel_bin_step (_S_GoertzelBin &bin, const int16_t sample_q15) noexcept {
        const int16_t input_q7 = zzs::kernel::fx::to_resonator_q7 (sample_q15);
        const int16_t s0 = zzs::kernel::fx::resonator_step (bin._M_cosine_q15, bin._M_s1_q7, bin._M_s2_q7, input_q7);
        bin._M_s2_q7 = bin._M_s1_q7;
        bin._M_s1_q7 = s0;
}

/**
 * @brief Finished-bin squared magnitude in Q14 (after the window is exhausted).
 *
 * Call once after feeding `window_samples` samples; then `magnitude_q7` gives
 * the Q7 magnitude if desired.
 */
[[nodiscard]] inline uint32_t
goertzel_bin_power_q14 (const _S_GoertzelBin &bin) noexcept {
        return zzs::kernel::fx::resonator_power (bin._M_s1_q7, bin._M_s2_q7, bin._M_cosine_q15);
}

/** @brief Finished-bin magnitude in Q7. */
[[nodiscard]] inline int32_t
goertzel_bin_magnitude_q7 (const _S_GoertzelBin &bin) noexcept {
        return zzs::kernel::fx::magnitude_q7 (goertzel_bin_power_q14 (bin));
}

/** @} */
} // namespace zzs::kernel::dsp
