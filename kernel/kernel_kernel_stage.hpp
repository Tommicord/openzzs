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
#include "kernel/kernel_feat_pack.hpp"
#include "kernel/kernel_fixed.hpp"
#include "kernel/kernel_meta_ids.hpp"
#include "kernel/kernel_sensor_types.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Preprocessing stage interface.
 *
 * Each feature stage (PLHR, vibration, flicker, acoustic) implements this
 * vtable: `configure` sets up filters/resonators from runtime parameters,
 * `reset` clears state, `feed` consumes one resampled grid-tick sample for a
 * channel, and `produce` writes its Q15 feature values into a caller-provided
 * slot buffer when a window is complete.  Stages are pull-based: the bank calls
 * `feed` per sample and `produce` per window.
 *
 * No heap, no float, no RTTI, no exceptions.  Stage state is a small POD owned
 * by the bank; scratch buffers are caller-provided to stay inside the RAM
 * budget.
 * @{
 */
namespace zzs::kernel {

/** @brief Opaque per-stage state; each stage defines its own concrete struct. */
struct _S_StageState;

/** @brief Stage vtable. */
struct _S_StageVtable {
    /** @brief Configure from runtime parameters (rates, windows, gains). */
    Status (*_M_configure) (void *const state) noexcept;

    /** @brief Reset all stage state to its post-configure defaults. */
    void (*_M_reset) (void *const state) noexcept;

    /** @brief Feed one resampled Q15 sample for a channel. */
    Status (*_M_feed) (void *const state, const sensor::Channel channel, const int16_t sample_q15) noexcept;

    /** @brief Produce Q15 feature values into `slots` (count slots) when a
     *        window is complete; returns NOT_READY when no window finished. */
    Status (*_M_produce) (void *const state, int32_t *const slots, const uint32_t slot_count) noexcept;
};

/** @brief A registered stage: vtable plus its state object. */
struct _S_Stage {
    const _S_StageVtable *_M_vtable;
    void *_M_state;
};

/** @brief Feed one sample into every registered stage. */
[[nodiscard]] inline Status
stage_feed_all (const _S_Stage *const stages, const uint32_t stage_count, const sensor::Channel channel,
                const int16_t sample_q15) noexcept {
    Status last = Status::OK;
    for (uint32_t i = 0; i < stage_count; ++i) {
        const Status st = stages[i]._M_vtable->_M_feed (stages[i]._M_state, channel, sample_q15);
        if (st != Status::OK)
            last = st;
    }
    return last;
}

/** @brief Produce features from every stage into a single flat slot buffer. */
[[nodiscard]] inline Status
stage_produce_all (const _S_Stage *const stages, const uint32_t stage_count, int32_t *const slots,
                   const uint32_t slot_count) noexcept {
    if (slots == nullptr)
        return Status::OUT_OF_RANGE;
    if (slot_count < zzs::kernel::_S_feature_count)
        return Status::OUT_OF_RANGE;
    Status last = Status::NOT_READY;
    for (uint32_t i = 0; i < stage_count; ++i) {
        if (stages[i]._M_vtable == nullptr)
            return Status::OUT_OF_RANGE;
        if (stages[i]._M_state == nullptr)
            return Status::OUT_OF_RANGE;
        const Status st = stages[i]._M_vtable->_M_produce (stages[i]._M_state, slots, slot_count);
        if (st == Status::OK)
            last = Status::OK;
        else if (st != Status::NOT_READY) {
            zzs::base::log_error ("stage {} produce failed ({})", i, status_name (st));
            return st; // Real error
        }
        zzs::base::log_debug ("stage {} produced window", i);
    }
    // Return OK if at least one stage produced.  Slots persist in the bank, so
    // a row contains the latest features from every stage even when stages
    // complete at different rates (PLHR/vibration ~1s, flicker ~100s at 1Hz).
    return last;
}

/** @} */
} // namespace zzs::kernel
