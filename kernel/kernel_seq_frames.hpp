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

#include "kernel/device-driver/builtins/zzs_builtins.hpp"
#include "kernel/kernel_config.hpp"

#include <cstdint>

/**
 * @brief Feature frame assembly: one mllee input frame [seq_len=5, 38].
 *
 * A frame is the model's input: `seq_len` packed int8 feature rows laid out
 * contiguously.  The sequence builder accepts completed rows from the bank and
 * reports Status::OK when a full frame is ready, NOT_READY while rows are still
 * being collected.  The caller owns the frame buffer (seq_len * feature_count
 * bytes) and may pass it directly to mllee's inference entry.
 *
 * The sequence owns its frame buffer as a fixed member array (RAII) and is
 * configured at construction time; there is no two-phase `init()`.  Copy/move
 * are deleted because the write position is meaningful state that must not be
 * duplicated.  Row copies use `ZZS_MEMCPY` instead of hand-rolled byte loops.
 * @{
 */
namespace zzs::kernel::frames {

/** @brief One feature frame: seq_len rows x feature_count int8 codes. */
class Sequence {
  int8_t _M_row[zzs::kernel::_S_seq_len][zzs::kernel::_S_feature_count]{};
  uint32_t _M_row_count = 0u;
  bool _M_complete = false;

public:
  Sequence () = default;
  Sequence (const Sequence &) = delete;
  Sequence &
  operator= (const Sequence &) = delete;
  Sequence (Sequence &&) = delete;
  Sequence &
  operator= (Sequence &&) = delete;

  /** @brief Append one packed feature row. Returns true when the frame completes. */
  [[nodiscard]] inline bool
  append_row (const int8_t *const row) noexcept {
    if (ZZS_UNLIKELY (row == nullptr || _M_complete || _M_row_count >= zzs::kernel::_S_seq_len))
      return _M_complete;
    ZZS_MEMCPY (_M_row[_M_row_count], row, zzs::kernel::_S_feature_count * sizeof (int8_t));
    ++_M_row_count;
    if (_M_row_count >= zzs::kernel::_S_seq_len)
      _M_complete = true;
    return _M_complete;
  }

  /** @brief Apply one packed feature row (alias for append_row). */
  [[nodiscard]] inline bool
  apply_row (const int8_t *const row) noexcept {
    return append_row (row);
  }

  /** @brief True when a full frame is available. */
  [[nodiscard]] inline bool
  complete () const noexcept {
    return _M_complete;
  }

  /** @brief Rows appended so far. */
  [[nodiscard]] inline uint32_t
  row_count () const noexcept {
    return _M_row_count;
  }

  /** @brief Pointer to the contiguous frame buffer (seq_len * feature_count int8). */
  [[nodiscard]] inline int8_t *
  data () noexcept {
    return &_M_row[0][0];
  }

  /** @brief Bytes of one frame. */
  [[nodiscard]] inline static constexpr uint32_t
  frame_bytes () noexcept {
    return zzs::kernel::_S_sequence_bytes;
  }

  /** @brief Reset the frame for the next inference. */
  inline void
  reset () noexcept {
    ZZS_MEMSET (_M_row, 0, sizeof (_M_row));
    _M_row_count = 0u;
    _M_complete = false;
  }
};

/** @} */
} // namespace zzs::kernel::frames
