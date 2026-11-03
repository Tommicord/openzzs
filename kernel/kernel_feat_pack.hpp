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
#include "kernel/kernel_meta_ids.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Feature vector packing: Q15 features -> int8-coded model row.
 *
 * Each feature stage writes its Q15 results into a feature slot; this module
 * clamps and converts each slot to int8 (_S_i8_code) and lays the row out per
 * `kernel_meta_ids.hpp`.  The row is the exact layout the mllee model input
 * expects: 38 int8 codes, one per feature, ordered by the feat_ids table. The
 * caller owns the row buffer (_S_feature_count bytes).
 * @{
 */
namespace zzs::kernel::feat {

/** @brief Q15-to-int8 code step: 32768 / 127 ≈ 258.
 *
 * Matches `mllee_native::detail::q15_raw_to_code` used by the training
 * pipeline, so inference input codes match training input codes exactly.
 */
[[nodiscard]] inline int8_t
q15_to_i8_code (const int32_t value_q15) noexcept {
  /* Divide by the code step (258), round toward zero, clamp to [-48, 48].
   * Matches `mllee_native::detail::q15_raw_to_code` used by the training
   * pipeline.  Training data codes fall within [-86, 29]; clamping at 48 keeps
   * saturated device features (e.g. flicker 100Hz hitting 126) from dominating
   * the model and producing degenerate one-hot [127, 0, 0] outputs, while
   * preserving the device-relevant features (flicker_100hz max=31,
   * acoustic_100hz max=42) that the model was trained on. */
  const int32_t code = value_q15 / 258;
  return static_cast<int8_t> (code < -48 ? -48 : (code > 48 ? 48 : code));
}

/** @brief Pack one Q15 feature value into row slot `index` as an int8 code. */
inline void
feat_pack_slot (int8_t *const row, const uint32_t index, const int32_t value_q15) noexcept {
  if (row == nullptr || index >= zzs::kernel::_S_feature_count)
    return;
  row[index] = q15_to_i8_code (value_q15);
}

/** @brief Pack a full feature row: write `value_q15` at every slot given. */
inline void
feat_pack_row (int8_t *const row, const int32_t *const values_q15, const uint32_t count) noexcept {
  if (row == nullptr || values_q15 == nullptr)
    return;
  const uint32_t n = count < zzs::kernel::_S_feature_count ? count : zzs::kernel::_S_feature_count;
  for (uint32_t i = 0; i < n; ++i)
    row[i] = q15_to_i8_code (values_q15[i]);
  zzs::base::log_debug ("feat_pack_row: emitted row count={}", n);
}

/** @brief Zero a feature row. */
inline void
feat_row_clear (int8_t *const row) noexcept {
  if (row == nullptr)
    return;
  for (uint32_t i = 0; i < zzs::kernel::_S_feature_count; ++i)
    row[i] = 0;
  zzs::base::log_debug ("feat_row_clear: row cleared");
}

/** @brief Number of bytes in one packed feature row. */
[[nodiscard]] inline constexpr uint32_t
feat_row_bytes (void) noexcept {
  return zzs::kernel::_S_feature_row_bytes;
}

/** @} */
} // namespace zzs::kernel::feat
