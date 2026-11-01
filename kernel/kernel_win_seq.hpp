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
#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief Sequence builder: assembles `seq_len` overlapping feature rows into the
 *        model's input frame [seq_len=5, feature_count=38].
 *
 * The model consumes one sequence per inference: 5 overlapping 1-second windows
 * (10 s context).  Each window produces one packed int8 feature row via the
 * feature stage; the sequence builder stores rows in a small ring and hands the
 * latest `seq_len` rows to the caller as a contiguous `int8` buffer.  Not enough
 * rows yet → NOT_READY.
 *
 * The caller owns the row storage (seq_len rows) and the output frame buffer
 * (seq_len * feature_count bytes).
 * @{
 */
namespace zzs::kernel::seq {

/** @brief Sequence state. */
struct _S_Sequence {
  int8_t *_M_rows;       /* caller storage: seq_len * feature_count bytes */
  uint32_t _M_row_bytes; /* == feature_count */
  uint32_t _M_write;     /* next row write index */
  uint32_t _M_count;     /* rows stored */
  bool _M_full;
};

/** @brief Initialize over caller-owned row storage. */
[[nodiscard]] inline zzs::kernel::Status
sequence_init (_S_Sequence &seq, int8_t *const rows, const uint32_t row_bytes) noexcept {
  if (rows == nullptr || row_bytes == 0u)
    return zzs::kernel::Status::OUT_OF_RANGE;
  seq._M_rows = rows;
  seq._M_row_bytes = row_bytes;
  seq._M_write = 0;
  seq._M_count = 0;
  seq._M_full = false;
  return zzs::kernel::Status::OK;
}

/** @brief Append one packed feature row. Overwrites oldest when full. */
inline void
sequence_push_row (_S_Sequence &seq, const int8_t *const row) noexcept {
  if (row == nullptr)
    return;
  int8_t *const dst = seq._M_rows + seq._M_write * seq._M_row_bytes;
  for (uint32_t i = 0; i < seq._M_row_bytes; ++i)
    dst[i] = row[i];
  if (seq._M_count < zzs::kernel::_S_seq_len)
    ++seq._M_count;
  else
    seq._M_full = true;
  /* seq_len (5) is not a power of two; avoid the modulo by branching. */
  seq._M_write = (seq._M_write + 1u) >= zzs::kernel::_S_seq_len ? 0u : seq._M_write + 1u;
}

/** @brief Rows currently stored. */
[[nodiscard]] inline uint32_t
sequence_count (const _S_Sequence &seq) noexcept {
  return seq._M_count;
}

/** @brief True when a full seq_len sequence is available. */
[[nodiscard]] inline bool
sequence_ready (const _S_Sequence &seq) noexcept {
  return seq._M_count >= zzs::kernel::_S_seq_len;
}

/**
 * @brief Copy the latest `seq_len` rows into `out` oldest-first.
 *
 * `out` must hold `seq_len * row_bytes` bytes.  Returns Status::OK when a full
 * sequence exists, NOT_READY otherwise (and leaves `out` untouched).
 */
[[nodiscard]] inline zzs::kernel::Status
sequence_snapshot (const _S_Sequence &seq, int8_t *const out) noexcept {
  if (!sequence_ready (seq) || out == nullptr)
    return zzs::kernel::Status::NOT_READY;
  /* Oldest row is at (write - count) mod seq_len; rows are contiguous in time.
     seq_len (5) is not a power of two, so wrap by conditional subtract
     instead of a modulo (constant time on Cortex-M4). */
  const uint32_t start_raw = seq._M_write + zzs::kernel::_S_seq_len - seq._M_count;
  const uint32_t start = start_raw >= zzs::kernel::_S_seq_len ? start_raw - zzs::kernel::_S_seq_len : start_raw;
  for (uint32_t r = 0; r < zzs::kernel::_S_seq_len; ++r) {
    uint32_t src_index = start + r;
    if (src_index >= zzs::kernel::_S_seq_len)
      src_index -= zzs::kernel::_S_seq_len;
    const int8_t *const src = seq._M_rows + src_index * seq._M_row_bytes;
    int8_t *const dst = out + r * seq._M_row_bytes;
    for (uint32_t i = 0; i < seq._M_row_bytes; ++i)
      dst[i] = src[i];
  }
  return zzs::kernel::Status::OK;
}

/** @brief Bytes of one sequence frame (seq_len * feature_count). */
[[nodiscard]] inline constexpr uint32_t
sequence_frame_bytes (void) noexcept {
  return zzs::kernel::_S_sequence_bytes;
}

/** @} */
} // namespace zzs::kernel::seq
