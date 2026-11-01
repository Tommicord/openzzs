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
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Goertzel bin bank: one resonator per target frequency bin.
 *
 * The bank is configured with a list of (frequency_hz, sample_rate_hz) pairs and
 * feeds every input sample into every resonator.  After a window is exhausted,
 * each bin's squared magnitude (Q14) and magnitude (Q7) are available.  This is
 * what the PLHR stage uses for the 50 Hz and 60 Hz harmonic grids, and what the
 * acoustic stage uses for the 100/120 Hz hum.
 *
 * Maximum bin count is fixed at compile time (_S_goertzel_max_bins) so the bank
 * is a stack-sized POD with no heap.  No float, no sorting.
 * @{
 */
namespace zzs::kernel::dsp {

/** @brief Maximum simultaneous Goertzel bins (keeps the bank stack-sized). */
inline constexpr uint32_t _S_goertzel_max_bins = 16u;

/** @brief One configured bin: descriptor plus resonator state. */
struct _S_GoertzelBinEntry {
  uint32_t _M_frequency_hz;
  uint32_t _M_sample_rate_hz;
  _S_GoertzelBin _M_resonator;
};

/** @brief The bank: bins plus the number configured. */
struct _S_GoertzelBank {
  _S_GoertzelBinEntry _M_bin[_S_goertzel_max_bins];
  uint32_t _M_bin_count;
};

/** @brief Reset all resonator states; coefficients are unchanged. */
inline void
goertzel_bank_reset (_S_GoertzelBank &bank) noexcept {
  zzs::base::log_debug ("goertzel_bank_reset: bins={}", bank._M_bin_count);
  for (uint32_t i = 0; i < bank._M_bin_count; ++i)
    zzs::kernel::dsp::goertzel_bin_reset (bank._M_bin[i]._M_resonator);
}

/**
 * @brief Add a bin at `frequency_hz` for a channel at `sample_rate_hz`.
 * @returns Status::OK, OUT_OF_RANGE when the bank is full or the rate is 0.
 */
[[nodiscard]] inline zzs::kernel::Status
goertzel_bank_add_bin (_S_GoertzelBank &bank, const uint32_t frequency_hz, const uint32_t sample_rate_hz) noexcept {
  if (sample_rate_hz == 0u || bank._M_bin_count >= _S_goertzel_max_bins) {
    zzs::base::log_error ("goertzel_bank_add_bin: freq={} rate={} bins={} -> {}", frequency_hz, sample_rate_hz,
                          bank._M_bin_count, zzs::kernel::status_name (zzs::kernel::Status::OUT_OF_RANGE));
    return zzs::kernel::Status::OUT_OF_RANGE;
  }
  _S_GoertzelBinEntry &entry = bank._M_bin[bank._M_bin_count];
  entry._M_frequency_hz = frequency_hz;
  entry._M_sample_rate_hz = sample_rate_hz;
  const zzs::kernel::Status st =
      zzs::kernel::dsp::goertzel_bin_configure (entry._M_resonator, frequency_hz, sample_rate_hz);
  if (st != zzs::kernel::Status::OK) {
    zzs::base::log_error ("goertzel_bank_add_bin: freq={} rate={} bins={} -> {}", frequency_hz, sample_rate_hz,
                          bank._M_bin_count, zzs::kernel::status_name (st));
    return st;
  }
  ++bank._M_bin_count;
  zzs::base::log_debug ("goertzel_bank_add_bin: ok freq={} rate={} bins={}", frequency_hz, sample_rate_hz,
                        bank._M_bin_count);
  return zzs::kernel::Status::OK;
}

/** @brief Feed one Q15 sample into every resonator in the bank. */
[[nodiscard]] inline zzs::kernel::Status
goertzel_bank_step (_S_GoertzelBank &bank, const int16_t sample_q15) noexcept {
  for (uint32_t i = 0; i < bank._M_bin_count; ++i)
    zzs::kernel::dsp::goertzel_bin_step (bank._M_bin[i]._M_resonator, sample_q15);
  return zzs::kernel::Status::OK;
}

/** @brief Squared magnitude (Q14) of bin `index` after its window is done. */
[[nodiscard]] inline uint32_t
goertzel_bank_power_q14 (const _S_GoertzelBank &bank, const uint32_t index) noexcept {
  if (index >= bank._M_bin_count)
    return 0u;
  return zzs::kernel::dsp::goertzel_bin_power_q14 (bank._M_bin[index]._M_resonator);
}

/** @brief Magnitude (Q7) of bin `index` after its window is done. */
[[nodiscard]] inline int32_t
goertzel_bank_magnitude_q7 (const _S_GoertzelBank &bank, const uint32_t index) noexcept {
  if (index >= bank._M_bin_count)
    return 0;
  return zzs::kernel::dsp::goertzel_bin_magnitude_q7 (bank._M_bin[index]._M_resonator);
}

/** @brief Frequency of bin `index`. */
[[nodiscard]] inline uint32_t
goertzel_bank_frequency_hz (const _S_GoertzelBank &bank, const uint32_t index) noexcept {
  if (index >= bank._M_bin_count)
    return 0u;
  return bank._M_bin[index]._M_frequency_hz;
}

/** @} */
} // namespace zzs::kernel::dsp
