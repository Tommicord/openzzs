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
#include "kernel/kernel_win_seq.hpp"

#include <cstdint>

/**
 * @brief Feature frame assembly: one mllee input frame [seq_len=5, 38].
 *
 * A frame is the model's input: `seq_len` packed int8 feature rows laid out
 * contiguously.  The frame assembler accepts completed rows from the bank and
 * reports Status::OK when a full frame is ready, NOT_READY while rows are still
 * being collected.  The caller owns the frame buffer (seq_len * feature_count
 * bytes) and may pass it directly to mllee's inference entry.
 * @{
 */
namespace zzs::kernel::frames {

/** @brief One feature frame: seq_len rows x feature_count int8 codes. */
struct FeatureFrame {
  int8_t _M_row[zzs::kernel::_S_seq_len][zzs::kernel::_S_feature_count];
  uint32_t _M_row_count;
  bool _M_complete;
};

/** @brief Initialize a frame. */
inline void
frame_init (FeatureFrame &frame) noexcept {
  for (uint32_t r = 0; r < zzs::kernel::_S_seq_len; ++r)
    for (uint32_t c = 0; c < zzs::kernel::_S_feature_count; ++c)
      frame._M_row[r][c] = 0;
  frame._M_row_count = 0;
  frame._M_complete = false;
}

/** @brief Append one packed feature row. Returns true when the frame completes. */
[[nodiscard]] inline bool
frame_append_row (FeatureFrame &frame, const int8_t *const row) noexcept {
  if (row == nullptr || frame._M_complete || frame._M_row_count >= zzs::kernel::_S_seq_len)
    return frame._M_complete;
  for (uint32_t c = 0; c < zzs::kernel::_S_feature_count; ++c)
    frame._M_row[frame._M_row_count][c] = row[c];
  ++frame._M_row_count;
  if (frame._M_row_count >= zzs::kernel::_S_seq_len)
    frame._M_complete = true;
  return frame._M_complete;
}

/** @brief Apply one packed feature row (alias for frame_append_row). */
[[nodiscard]] inline bool
apply_row (FeatureFrame &frame, const int8_t *const row) noexcept {
  return frame_append_row (frame, row);
}

/** @brief True when a full frame is available. */
[[nodiscard]] inline bool
frame_complete (const FeatureFrame &frame) noexcept {
  return frame._M_complete;
}

/** @brief Bytes of one frame. */
[[nodiscard]] inline constexpr uint32_t
frame_bytes (void) noexcept {
  return zzs::kernel::_S_sequence_bytes;
}

/** @brief Reset the frame for the next inference. */
inline void
frame_reset (FeatureFrame &frame) noexcept {
  frame_init (frame);
}

/** @} */
} // namespace zzs::kernel::frames
