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

#include "kernel/kernel_clock_grid.hpp"
#include "kernel/kernel_config.hpp"
#include "kernel/kernel_feat_pack.hpp"
#include "kernel/kernel_kernel_stage.hpp"
#include "kernel/kernel_meta_ids.hpp"
#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_status.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief PreprocessingBank: the orchestrator between the sensor HAL and the
 *        mllee inference core.
 *
 * The bank owns all stage state and the per-channel resamplers/ring buffers. It
 * is fed one raw sensor sample at a time (via `bank_feed_sample`), resamples
 * each channel to the 100 Hz grid, feeds every stage, and produces one packed
 * int8 feature row per window via `bank_produce_row`.  Rows are handed to the
 * sequence builder (seq_len=5) for the model.
 *
 * The entire bank plus its deepest call chain fits inside
 * `_S_bank_budget_bytes` of RAM; the static_assert below enforces the object
 * share.  Nothing allocates; stage state and scratch are pre-sized PODs.
 *
 * The caller supplies all backing storage (stage state, resamplers, rings, row
 * buffer) so the bank itself is just a small POD of pointers and counts.
 * @{
 */
namespace zzs::kernel {

/** @brief Maximum registered stages. */
inline constexpr uint32_t _S_bank_max_stages = 8u;

/** @brief Maximum resampled channels tracked simultaneously. */
inline constexpr uint32_t _S_bank_max_channels = 12u;

/** @brief Bank configuration: stage list and grid tracker. */
struct _S_BankConfig {
  _S_Stage _M_stages[_S_bank_max_stages];
  uint32_t _M_stage_count;
  clock::_S_GridTracker _M_grid;
};

/** @brief Bank runtime state. */
struct _S_PreprocessingBank {
  _S_BankConfig *_M_config;
  bool _M_initialized;
  /** @brief Persistent feature slots: retain values from stages that produced
   *         in earlier calls, so a row contains the latest features from every
   *         stage even when stages complete at different rates. */
  int32_t _M_slots[_S_feature_count] = {};
};

/** @brief Static RAM-share check: the bank object plus pointer overhead must
 *        fit within the bank budget. */
static_assert (sizeof (_S_PreprocessingBank) <= _S_bank_budget_bytes,
               "PreprocessingBank object exceeds the RAM budget share");

/** @brief Initialize the bank from a config. */
[[nodiscard]] inline Status
bank_init (_S_PreprocessingBank &bank, _S_BankConfig &config) noexcept {
  bank._M_config = &config;
  bank._M_initialized = true;
  for (uint32_t i = 0; i < _S_feature_count; ++i)
    bank._M_slots[i] = 0;
  for (uint32_t i = 0; i < config._M_stage_count; ++i) {
    const Status st = config._M_stages[i]._M_vtable->_M_configure (config._M_stages[i]._M_state);
    if (st != Status::OK) {
      zzs::base::log_error ("stage {} configure failed ({})", i, status_name (st));
      return st;
    }
    zzs::base::log_debug ("stage {} configured", i);
    config._M_stages[i]._M_vtable->_M_reset (config._M_stages[i]._M_state);
    zzs::base::log_debug ("stage {} reset", i);
  }
  const Status st = clock::grid_tracker_init (config._M_grid, 0ull);
  if (st != Status::OK) {
    zzs::base::log_error ("grid tracker init failed ({})", status_name (st));
    return st;
  }
  zzs::base::log_info ("bank initialized, {} stages", config._M_stage_count);
  return Status::OK;
}

/** @brief Reset every stage and the grid tracker. */
inline void
bank_reset (_S_PreprocessingBank &bank) noexcept {
  if (bank._M_config == nullptr)
    return;
  zzs::base::log_debug ("bank reset, {} stages", bank._M_config->_M_stage_count);
  for (uint32_t i = 0; i < bank._M_config->_M_stage_count; ++i)
    bank._M_config->_M_stages[i]._M_vtable->_M_reset (bank._M_config->_M_stages[i]._M_state);
}

/** @brief Feed one raw sensor sample into the bank.
 *
 * Advances the grid tracker with the sample's timestamp (so the grid actually
 * progresses), resamples the channel to the grid, and feeds every stage.
 * Returns Status::OK, NOT_READY when the grid has not ticked, or an error.
 */
[[nodiscard]] inline Status
bank_feed_sample (_S_PreprocessingBank &bank, const sensor::Sample &sample) noexcept {
  if (!bank._M_initialized || bank._M_config == nullptr)
    return Status::OUT_OF_RANGE;
  if (bank._M_config->_M_stage_count > _S_bank_max_stages)
    return Status::OUT_OF_RANGE;
  for (uint32_t i = 0; i < bank._M_config->_M_stage_count; ++i) {
    if (bank._M_config->_M_stages[i]._M_vtable == nullptr)
      return Status::OUT_OF_RANGE;
    if (bank._M_config->_M_stages[i]._M_state == nullptr)
      return Status::OUT_OF_RANGE;
  }

  /* Advance the common grid with this sample's timestamp so the grid
     actually progresses (was previously dead: never advanced). */
  (void) clock::grid_tracker_advance (bank._M_config->_M_grid, sample._M_timestamp_ns);

  const int16_t sample_q15 = sample._M_sample_q15;
  (void) stage_feed_all (bank._M_config->_M_stages, bank._M_config->_M_stage_count, sample._M_channel, sample_q15);
  return Status::OK;
}

/** @brief Produce features from every stage into a packed int8 feature row.
 *
 * Returns NOT_READY when no stage produced a finished window this tick.
 * The row is laid out per kernel_meta_ids.hpp and ready for mllee input.
 */
[[nodiscard]] inline Status
bank_produce_row (_S_PreprocessingBank &bank, int8_t *const row) noexcept {
  if (!bank._M_initialized || bank._M_config == nullptr)
    return Status::OUT_OF_RANGE;
  if (row == nullptr)
    return Status::OUT_OF_RANGE;
  /* Slots persist across calls: stages that have not completed since the last
     row retain their previous feature values.  Only stages that complete now
     overwrite their slots. */
  const Status st =
      stage_produce_all (bank._M_config->_M_stages, bank._M_config->_M_stage_count, bank._M_slots, _S_feature_count);
  if (st == Status::NOT_READY)
    return Status::NOT_READY;
  if (st != Status::OK)
    zzs::base::log_error ("stage produce failed ({})", status_name (st));
  else
    zzs::base::log_debug ("feature row produced");
  feat::feat_pack_row (row, bank._M_slots, _S_feature_count);
  return Status::OK;
}

/** @} */
} // namespace zzs::kernel
