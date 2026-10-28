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

#ifndef ZZS_MLLEE_TRAIN_PRUNE_HPP
#define ZZS_MLLEE_TRAIN_PRUNE_HPP

#include "mllee/format.hpp"
#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_simd.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Magnitude pruning over int8 code weights: masks, blocks, schedules.
 *
 * Pruning here means "a lane is permanently zero and stays zero", which in this
 * module is a statement about int8 codes: the -127..127 grid already carries the
 * weight, so a magnitude ranking is an integer ranking.  Ranking by `|code|` is
 * exactly ranking by the lane's Q15 magnitude, because `code_to_q15` multiplies
 * every code by the same positive step (258) -- so no second numeric convention
 * is introduced, and the selection costs no float, no sort and no
 * element-proportional buffer.
 *
 * Unstructured selection is a two-pass sweep over a caller-owned 128-bin
 * magnitude histogram (`PruneHistogram`, 516 bytes whatever the tensor size):
 *
 *   1. bin `|code|` for every lane -- one pass, stride-1;
 *   2. walk the bins from the largest magnitude down to the threshold bin `t`
 *      where `count(|m| > t) <= keep <= count(|m| >= t)`, which fixes how many
 *      lanes of the tie group `t` survive: `tie_keep = keep - count(|m| > t)`;
 *   3. one pass over the lanes: a lane survives iff `|code| > t`, or iff
 *      `|code| == t` and it is among the first `tie_keep` lanes of that magnitude
 *      in ascending index order.
 *
 * The tie rule is part of the contract: ties break toward the *lowest index*, so a
 * rerun over the same weights reproduces the same mask byte-for-byte -- the
 * determinism mllee/train_checkpoint.hpp relies on.
 *
 * Patterns:
 *
 *   - Unstructured: any lane may go; the target sparsity is met exactly, to the
 *     granularity of one lane, and `PruneReport::_M_achieved_sparsity_q15` states
 *     what was actually reached;
 *   - NinM (`2:4`, `4:8`): each aligned block of M contiguous lanes keeps its N
 *     largest magnitudes -- the structured zero pattern sparse multiply hardware
 *     wants, computed lane-locally with no histogram and a bounded 8-entry order
 *     window in registers;
 *   - StructuredRows: whole channel rows are dropped by row L1 magnitude through
 *     the same threshold/tie machinery, and the resulting stream is compacted and
 *     the `Layer` descriptor's in/out dimensions rewritten, so a removed channel
 *     costs the bytes it no longer occupies instead of a mask.
 *
 * Both passes honour a `PruneMask`: forward zeroes the pruned codes, backward
 * zeroes those lanes' gradient, so a pruned weight receives no gradient update and
 * cannot come back to life.  The mask is one byte per lane -- exactly the operand
 * type of `route_by_mask` in mllee/train_backward_kernels.hpp -- plus a packed
 * one-bit-per-lane form for storage.
 *
 * Gradual sparsity is a pure function of the step index
 * (`prune_sparsity_at_q15`: constant, linear, plateau/ramp/plateau), so a run
 * resumed from a checkpoint recomputes the same schedule instead of trusting a
 * counter that may or may not have been serialized.
 *
 * ### Pool cost (all caller-owned; nothing here allocates)
 *
 * | Object | Bytes |
 * |---|---|
 * | `PruneConfig` / `PruneReport` / `PruneRunReport` | 32 / 40 / 36 |
 * | `PruneMask` / `PruneHistogram` | 24 / 516 |
 * | `PruneRowGeom` / `StructuredPlan` / `DimRewrite` | 24 / 40 / 12 |
 * | byte mask over N lanes | N (aligned to 4) |
 * | packed mask over N lanes | (N + 7) / 8 |
 * | structured row scores + flags | 4 · R + R |
 * | **one unstructured prune of N lanes** | 516 + N |
 * | **one N:M prune of N lanes** | N |
 * | **one structured prune of R rows of L lanes** | 516 + 5 · R + N |
 *
 * A 1024-lane tensor costs 516 + 1024 = 1540 bytes for an unstructured pass; the
 * histogram is reused across tensors, so the peak is 516 + the largest mask rather
 * than the sum over layers.
 */
namespace zzs::mllee::train
{
/** @brief Which zero pattern a prune produces. */
enum class PrunePattern : uint8_t
{
    Unstructured = 0,
    /** @brief N of every M contiguous lanes survive (2:4, 4:8). */
    NinM = 1,
    /** @brief Whole channel rows are removed and dimensions rewritten. */
    StructuredRows = 2,
};

/** @brief How the requested sparsity moves with the training step. */
enum class SparsitySchedule : uint8_t
{
    /** @brief Always the configured target. */
    Constant = 0,
    /** @brief Linear from initial to final over [0, ramp_end). */
    Linear = 1,
    /** @brief Plateau at initial, linear ramp, plateau at final. */
    ThreeStage = 2,
};

/** @brief Largest block width the N:M path supports (4:8 is the widest). */
inline constexpr uint32_t _S_prune_max_block = 8u;

/** @brief Magnitude classes: |code| in [0, 127], i.e. the whole code grid. */
inline constexpr uint32_t _S_prune_mag_bins = 128u;

/** @brief Q15 sparsity ceiling: 32767/32768, i.e. all but one lane. */
inline constexpr int32_t _S_prune_max_sparsity_q15 = 32767;

/** @brief Largest lane count one prune addresses. */
inline constexpr uint32_t _S_prune_max_count = 65535u;

/** @brief int32 span used to saturate a row score. */
inline constexpr int32_t _S_prune_score_max = 2147483647;

/**
 * @brief Prune request: how much to remove, in which pattern, on which schedule.
 */
struct PruneConfig
{
    /** @brief Desired fraction of lanes zeroed, Q15 (0 == no pruning). */
    int32_t _M_target_sparsity_q15 = 0;
    /** @brief Sparsity before the ramp (gradual schedules), Q15. */
    int32_t _M_initial_sparsity_q15 = 0;
    /** @brief Sparsity after the ramp (gradual schedules), Q15. */
    int32_t _M_final_sparsity_q15 = 0;
    /** @brief First ramp step, read by ThreeStage only. */
    uint32_t _M_ramp_start_step = 0u;
    /** @brief Step at which the ramp has reached final. */
    uint32_t _M_ramp_end_step = 0u;
    uint8_t _M_pattern = static_cast<uint8_t> (PrunePattern::Unstructured);
    /** @brief Lanes kept per block; read by the N:M pattern only. */
    uint8_t _M_block_n = 2u;
    /** @brief Block width in lanes; read by the N:M pattern only. */
    uint8_t _M_block_m = 4u;
    /** @brief Row axis for StructuredRows (the channel axis of the tensor). */
    uint8_t _M_axis = 0u;
    uint8_t _M_schedule = static_cast<uint8_t> (SparsitySchedule::Constant);
    uint8_t _M_reserved0 = 0u;
    uint8_t _M_reserved1 = 0u;
    uint16_t _M_pad = 0u;

    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (this->_M_pattern > static_cast<uint8_t> (PrunePattern::StructuredRows))
            return Status::BAD_LAYOUT;
        if (this->_M_schedule > static_cast<uint8_t> (SparsitySchedule::ThreeStage))
            return Status::BAD_LAYOUT;
        if (this->_M_target_sparsity_q15 < 0
            || this->_M_target_sparsity_q15 > _S_prune_max_sparsity_q15)
            return Status::OUT_OF_RANGE;
        if (this->_M_initial_sparsity_q15 < 0
            || this->_M_initial_sparsity_q15 > _S_prune_max_sparsity_q15)
            return Status::OUT_OF_RANGE;
        if (this->_M_final_sparsity_q15 < 0
            || this->_M_final_sparsity_q15 > _S_prune_max_sparsity_q15)
            return Status::OUT_OF_RANGE;
        if (this->_M_axis >= _S_max_rank)
            return Status::BAD_LAYOUT;
        if (this->_M_pattern == static_cast<uint8_t> (PrunePattern::NinM))
            {
                /* Only the two patterns sparse multiply units actually exploit. */
                if (!(this->_M_block_n == 2u && this->_M_block_m == 4u)
                    && !(this->_M_block_n == 4u && this->_M_block_m == 8u))
                    return Status::UNSUPPORTED;
            }
        if (this->_M_schedule
                != static_cast<uint8_t> (SparsitySchedule::Constant)
            && this->_M_ramp_end_step == 0u)
            return Status::BAD_LAYOUT;
        if (this->_M_schedule == static_cast<uint8_t> (SparsitySchedule::ThreeStage)
            && this->_M_ramp_start_step > this->_M_ramp_end_step)
            return Status::BAD_LAYOUT;
        return Status::OK;
    }
};

static_assert (sizeof (PruneConfig) == 32u,
               "prune config must stay a 32-byte POD");

/**
 * @brief Caller-owned magnitude histogram: the only prune scratch wider than 1.
 *
 * Bins are the |code| classes 0..127, which is also the Q15 magnitude ranking
 * because `code_to_q15` scales every code by the same positive step.
 */
struct PruneHistogram
{
    uint32_t _M_bins[_S_prune_mag_bins];
    uint32_t _M_total;

    constexpr void
    reset () noexcept
    {
        for (uint32_t bin = 0; bin < _S_prune_mag_bins; ++bin)
            this->_M_bins[bin] = 0u;
        this->_M_total = 0u;
    }

    /** @brief Saturating count of one lane's magnitude class. */
    constexpr void
    add (const uint32_t magnitude) noexcept
    {
        const uint32_t bin = magnitude < 127u ? magnitude : 127u;
        if (this->_M_bins[bin] < 0xFFFFFFFFu)
            ++this->_M_bins[bin];
        if (this->_M_total < 0xFFFFFFFFu)
            ++this->_M_total;
    }
};

static_assert (sizeof (PruneHistogram) == 516u,
               "the prune histogram is a measured 516-byte slab");

/**
 * @brief The mask both passes honour: one byte per lane, 1 == survives.
 *
 * The byte form is deliberate: it is the operand type of `route_by_mask`, so a
 * masked gradient accumulation uses the same vectorized atom dropout and max-pool
 * backward already use, and it stays stride-1 in both passes.
 */
struct PruneMask
{
    uint8_t *_M_bits = nullptr;
    uint32_t _M_count = 0u;
    /** @brief Lanes with a zero bit, maintained by the prune that wrote it. */
    uint32_t _M_pruned = 0u;
    uint8_t _M_pattern = 0u;
    /** @brief Non-zero once a prune has filled this mask. */
    uint8_t _M_valid = 0u;
    uint16_t _M_pad = 0u;

    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (this->_M_bits == nullptr || this->_M_valid == 0u)
            return Status::BAD_LAYOUT;
        if (this->_M_count > _S_prune_max_count)
            return Status::OUT_OF_RANGE;
        return this->_M_pruned <= this->_M_count ? Status::OK
                                                : Status::CORRUPT_STREAM;
    }

    [[nodiscard]] constexpr bool
    keeps (const uint32_t index) const noexcept
    {
        return this->_M_bits[index] != 0u;
    }

    /** @brief Achieved sparsity of this mask, Q15. */
    [[nodiscard]] constexpr int32_t
    sparsity_q15 () const noexcept
    {
        return ratio_q15 (sat_i32 (this->_M_pruned),
                          sat_i32 (this->_M_count == 0u ? 1u
                                                       : this->_M_count));
    }

    [[nodiscard]] static constexpr int32_t
    sat_i32 (const uint64_t value) noexcept
    {
        return value > static_cast<uint64_t> (_S_prune_score_max)
                   ? _S_prune_score_max
                   : static_cast<int32_t> (value);
    }
};

static_assert (sizeof (PruneMask) == (sizeof (void *) == 4u ? 16u : 24u),
               "a prune mask view must stay a single-word-pointer-sized POD");

/** @brief What one prune did: exact counts, the threshold and the tie group. */
struct PruneReport
{
    uint32_t _M_elements = 0u;
    /** @brief Lanes zeroed by this pass. */
    uint32_t _M_pruned = 0u;
    /** @brief Lanes the requested sparsity asked to remove. */
    uint32_t _M_target_pruned = 0u;
    /** @brief Achieved fraction, Q15: the number the ops gate reads. */
    int32_t _M_achieved_sparsity_q15 = 0;
    /** @brief Fraction the schedule asked for at this step, Q15. */
    int32_t _M_requested_sparsity_q15 = 0;
    /** @brief Magnitude class the survival decision turned on. */
    int32_t _M_threshold_magnitude = 0;
    /** @brief Lanes sharing the threshold class (the tie group). */
    uint32_t _M_tie_group = 0u;
    /** @brief Of the tie group, how many survived: the tie rule applied. */
    uint32_t _M_tie_kept = 0u;
    /** @brief N:M blocks touched. */
    uint32_t _M_blocks_total = 0u;
    /** @brief Short tail blocks, which hold fewer than M lanes by definition. */
    uint32_t _M_blocks_partial = 0u;
};

static_assert (sizeof (PruneReport) == 40u,
               "the per-tensor prune report must stay a 40-byte POD");

/** @brief Aggregated report across a set of tensors and one structured rewrite. */
struct PruneRunReport
{
    uint32_t _M_tensors = 0u;
    uint32_t _M_elements = 0u;
    uint32_t _M_pruned = 0u;
    /** @brief Sparsity weighted over every tensor in the run, Q15. */
    int32_t _M_achieved_sparsity_q15 = 0;
    /** @brief Sparsity the schedule asked for at the step this run used, Q15. */
    int32_t _M_step_sparsity_q15 = 0;
    uint32_t _M_rows_before = 0u;
    uint32_t _M_rows_after = 0u;
    uint32_t _M_blocks_total = 0u;
    uint32_t _M_blocks_partial = 0u;
};

static_assert (sizeof (PruneRunReport) == 36u,
               "the run report must stay a 36-byte POD");

/** @brief Saturating int32 view of a uint64 counter. */
[[nodiscard]] inline constexpr int32_t
prune_sat_i32 (const uint64_t value) noexcept
{
    return value > static_cast<uint64_t> (_S_prune_score_max)
               ? _S_prune_score_max
               : static_cast<int32_t> (value);
}

/**
 * @brief Sparsity requested at a step: a pure function of the step index.
 *
 * - Constant:   `target` at every step.
 * - Linear:     `initial` at step 0 to `final` at `ramp_end`, clamped outside.
 * - ThreeStage: `initial` up to `ramp_start`, the same linear interpolation across
 *               [ramp_start, ramp_end], `final` after.
 *
 * The interpolation is `initial + (final - initial) * (step - start)
 * / (end - start)` with round-half-up on the remainder in int64, so a resumed run
 * recomputes the identical value; nothing is stored between steps.
 */
[[nodiscard]] inline constexpr int32_t
prune_sparsity_at_q15 (const PruneConfig &config, const uint32_t step) noexcept
{
    if (config._M_schedule == static_cast<uint8_t> (SparsitySchedule::Constant))
        return config._M_target_sparsity_q15;
    const uint32_t start
        = config._M_schedule == static_cast<uint8_t> (SparsitySchedule::ThreeStage)
              ? config._M_ramp_start_step
              : 0u;
    const uint32_t end
        = config._M_ramp_end_step == 0u ? 1u : config._M_ramp_end_step;
    const int32_t from = config._M_initial_sparsity_q15;
    const int32_t to = config._M_final_sparsity_q15;
    if (step <= start)
        return from;
    if (step >= end)
        return to;
    const int64_t span = static_cast<int64_t> (end) - static_cast<int64_t> (start);
    const int64_t numerator = (static_cast<int64_t> (to) - from)
                              * (static_cast<int64_t> (step) - start);
    const int64_t magnitude = numerator < 0 ? -numerator : numerator;
    const int64_t quotient = (magnitude + span / 2) / span;
    return q15_clamp (static_cast<int32_t> (
        from + (numerator < 0 ? -quotient : quotient)));
}

/** @brief Lanes the request asks to remove from @p count lanes (floored). */
[[nodiscard]] inline constexpr uint32_t
prune_target_pruned (const uint32_t count, const int32_t sparsity_q15) noexcept
{
    if (sparsity_q15 <= 0 || count == 0u)
        return 0u;
    const uint64_t product
        = static_cast<uint64_t> (count) * static_cast<uint32_t> (sparsity_q15);
    const uint64_t pruned = product / static_cast<uint64_t> (_S_q15_one);
    return pruned > static_cast<uint64_t> (count) ? count
                                                 : static_cast<uint32_t> (pruned);
}

/** @brief Absolute value of a code as a magnitude class in [0, 127]. */
[[nodiscard]] inline constexpr uint32_t
prune_magnitude (const int32_t code) noexcept
{
    const uint32_t magnitude = static_cast<uint32_t> (code < 0 ? -code : code);
    return magnitude > 127u ? 127u : magnitude;
}

/** @brief Fill a magnitude histogram from a code stream: one pass, stride-1. */
[[nodiscard]] inline Status
prune_histogram_of_codes (PruneHistogram &histogram, const int8_t *const codes,
                          const uint32_t count) noexcept
{
    if (codes == nullptr)
        return Status::BAD_LAYOUT;
    if (count > _S_prune_max_count)
        return Status::OUT_OF_RANGE;
    histogram.reset ();
    for (uint32_t i = 0; i < count; ++i)
        histogram.add (prune_magnitude (codes[i]));
    return Status::OK;
}

/**
 * @brief Threshold and tie budget for a survivor count over a histogram.
 *
 * Walks the magnitude classes from the top down; the first class `t` whose
 * strict-above count still fits the budget and whose own count can cover the
 * remainder is the threshold, and `tie_keep` is how many of its lanes survive.
 */
[[nodiscard]] inline constexpr Status
prune_select_threshold (const PruneHistogram &histogram, const uint32_t keep,
                        uint32_t &threshold_out, uint32_t &tie_keep_out) noexcept
{
    if (keep > histogram._M_total)
        return Status::OUT_OF_RANGE;
    uint32_t above = 0u;
    for (uint32_t bin = _S_prune_mag_bins; bin-- > 0u;)
        {
            const uint32_t here = histogram._M_bins[bin];
            if (above <= keep
                && static_cast<uint64_t> (above) + here >= keep)
                {
                    threshold_out = bin;
                    tie_keep_out = keep - above;
                    return Status::OK;
                }
            above += here;
        }
    threshold_out = 0u;
    tie_keep_out = keep;
    return Status::OK;
}

/** @brief Write an unstructured mask from a code stream and a survivor budget. */
[[nodiscard]] inline Status
prune_mask_unstructured (const int8_t *const codes, const uint32_t count,
                         const uint32_t keep, PruneHistogram &scratch,
                         PruneMask &mask, uint32_t &threshold_out,
                         uint32_t &tie_group_out,
                         uint32_t &tie_kept_out) noexcept
{
    if (codes == nullptr || mask._M_bits == nullptr)
        return Status::BAD_LAYOUT;
    if (count > _S_prune_max_count || mask._M_count != count)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (mask._M_bits))
        return Status::BAD_LAYOUT;
    uint32_t threshold = 0u;
    uint32_t tie_keep = 0u;
    const Status histogram_status
        = prune_histogram_of_codes (scratch, codes, count);
    if (!status_ok (histogram_status))
        return histogram_status;
    const Status select_status
        = prune_select_threshold (scratch, keep, threshold, tie_keep);
    if (!status_ok (select_status))
        return select_status;
    uint32_t used_ties = 0u;
    uint32_t pruned = 0u;
    for (uint32_t i = 0; i < count; ++i)
        {
            const uint32_t magnitude = prune_magnitude (codes[i]);
            uint8_t survives = 0u;
            if (magnitude > threshold)
                survives = 1u;
            else if (magnitude == threshold && used_ties < tie_keep)
                {
                    /* Ascending index wins the tie group, so the mask is a
                       deterministic function of the weights alone. */
                    survives = 1u;
                    ++used_ties;
                }
            mask._M_bits[i] = survives;
            if (survives == 0u)
                ++pruned;
        }
    mask._M_pruned = pruned;
    mask._M_valid = 1u;
    mask._M_pattern = static_cast<uint8_t> (PrunePattern::Unstructured);
    threshold_out = threshold;
    tie_group_out = scratch._M_bins[threshold];
    tie_kept_out = tie_keep;
    return Status::OK;
}

/**
 * @brief N:M block selection: every aligned block keeps its N largest magnitudes.
 *
 * Blocks are contiguous and aligned to the buffer start, which is what the 2:4 and
 * 4:8 sparse multiply layouts require.  Inside a block the ranking is a bounded
 * stable insertion order over at most eight lanes -- no buffer proportional to the
 * tensor is touched and no element is looked up outside the block.  Ties break to
 * the lower lane index; a short tail block keeps its own N largest (all lanes when
 * it holds fewer than N).
 */
[[nodiscard]] inline Status
prune_mask_ninm (const int8_t *const codes, const uint32_t count,
                 const uint32_t block_n, const uint32_t block_m,
                 PruneMask &mask, uint32_t &blocks_total_out,
                 uint32_t &blocks_partial_out) noexcept
{
    if (codes == nullptr || mask._M_bits == nullptr)
        return Status::BAD_LAYOUT;
    if (count > _S_prune_max_count || mask._M_count != count)
        return Status::BAD_LAYOUT;
    if (block_m == 0u || block_m > _S_prune_max_block || block_n == 0u
        || block_n >= block_m)
        return Status::UNSUPPORTED;
    if (!simd_aligned (mask._M_bits))
        return Status::BAD_LAYOUT;
    uint32_t index = 0u;
    uint32_t blocks = 0u;
    uint32_t partial = 0u;
    uint32_t pruned = 0u;
    while (index < count)
        {
            const uint32_t width
                = count - index < block_m ? count - index : block_m;
            if (width < block_m)
                ++partial;
            ++blocks;
            /* Stable descending insertion: strictly larger magnitudes pull ahead,
               equal magnitudes keep the earlier lane, so ties go to the lower
               index inside the block. */
            uint8_t order[_S_prune_max_block] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
            uint8_t survives[_S_prune_max_block] = { 0u, 0u, 0u, 0u, 0u, 0u, 0u,
                                                     0u };
            uint32_t held = 0u;
            for (uint32_t lane = 0u; lane < width; ++lane)
                {
                    const uint32_t magnitude = prune_magnitude (
                        codes[index + lane]);
                    uint32_t slot = held;
                    while (slot > 0u
                           && prune_magnitude (
                                 codes[index + order[slot - 1u]]) < magnitude)
                        {
                            order[slot] = order[slot - 1u];
                            --slot;
                        }
                    while (slot < held
                           && prune_magnitude (
                                 codes[index + order[slot]]) == magnitude)
                        ++slot;
                    order[slot] = static_cast<uint8_t> (lane);
                    ++held;
                }
            const uint32_t survivors = held < block_n ? held : block_n;
            for (uint32_t k = 0u; k < survivors; ++k)
                survives[order[k]] = 1u;
            for (uint32_t lane = 0u; lane < width; ++lane)
                {
                    mask._M_bits[index + lane] = survives[lane];
                    if (survives[lane] == 0u)
                        ++pruned;
                }
            index += width;
        }
    mask._M_pruned = pruned;
    mask._M_valid = 1u;
    mask._M_pattern = static_cast<uint8_t> (PrunePattern::NinM);
    blocks_total_out = blocks;
    blocks_partial_out = partial;
    return Status::OK;
}

/**
 * @brief Row geometry of a structured prune: one channel slice, repeated.
 *
 * A row is `stride(axis)` contiguous lanes, `dim(axis)` rows make one period, and
 * `count / period` periods tile the buffer -- which covers both the leading axis of
 * a rank-2 weight (one period, one row per output channel) and a channel axis in
 * the middle of a rank-4 tensor (the same row reappearing in every outer block).
 */
struct PruneRowGeom
{
    uint32_t _M_row_len = 0u;
    uint32_t _M_rows = 0u;
    uint32_t _M_period = 0u;
    uint32_t _M_blocks = 0u;
    uint32_t _M_count = 0u;
    uint8_t _M_axis = 0u;
    uint8_t _M_valid = 0u;
    uint16_t _M_pad = 0u;
};

static_assert (sizeof (PruneRowGeom) == 24u,
               "row geometry must stay a 24-byte POD");

/** @brief Derive the row geometry of @p shape along its channel @p axis. */
[[nodiscard]] inline constexpr Status
prune_row_geom (const Shape &shape, const uint8_t axis,
                PruneRowGeom &out) noexcept
{
    const Status shape_status = shape.validate ();
    if (!status_ok (shape_status))
        return shape_status;
    out._M_valid = 0u;
    if (axis >= shape._M_rank)
        return Status::BAD_LAYOUT;
    const uint32_t count = shape.elements ();
    const uint32_t rows = shape._M_dims[axis];
    const uint32_t row_len = shape.stride (axis);
    if (rows == 0u || row_len == 0u || rows > _S_prune_max_count)
        return Status::OUT_OF_RANGE;
    const uint32_t period = rows * row_len;
    if (period == 0u || count % period != 0u)
        return Status::BAD_LAYOUT;
    out._M_row_len = row_len;
    out._M_rows = rows;
    out._M_period = period;
    out._M_blocks = count / period;
    out._M_count = count;
    out._M_axis = axis;
    out._M_valid = 1u;
    return Status::OK;
}

/**
 * @brief Rows a structured prune keeps, plus the compacted stream's geometry.
 */
struct StructuredPlan
{
    PruneRowGeom _M_geom;
    /** @brief One byte per row: 1 keeps the row. */
    uint8_t *_M_keep = nullptr;
    uint32_t _M_rows_kept = 0u;
    /** @brief Lanes in the compacted stream: blocks * rows_kept * row_len. */
    uint32_t _M_kept_count = 0u;
};

static_assert (sizeof (StructuredPlan) == (sizeof (void *) == 4u ? 36u : 40u),
               "the structured plan is its geometry plus one keep-pointer");

/** @brief In/out dimension rewrite a structured prune produces. */
struct DimRewrite
{
    uint16_t _M_in_from = 0u;
    uint16_t _M_in_to = 0u;
    uint16_t _M_out_from = 0u;
    uint16_t _M_out_to = 0u;
    uint8_t _M_axis = 0u;
    uint8_t _M_valid = 0u;
    uint16_t _M_pad = 0u;
};

static_assert (sizeof (DimRewrite) == 12u,
               "a dimension rewrite must stay a 12-byte POD");

/**
 * @brief Choose which whole rows survive, by row L1 magnitude.
 *
 * A row's score is the saturating L1 sum of |code| over every one of its slices
 * (so a channel that is small in every outer block ranks low, which is what a
 * channel prune must measure); scores are folded into the shared 128-bin histogram
 * so the threshold and tie walk is identical to the unstructured path.
 *
 * @p scores is caller-owned `geom._M_rows` int32 lanes and @p keep caller-owned
 * `geom._M_rows` bytes; both are validated.  Cost: one pass over the codes for the
 * scores, one short pass over the scores for the histogram, one pass over the
 * codes to zero the dropped rows.
 */
[[nodiscard]] inline Status
prune_select_rows (const int8_t *const codes, const PruneRowGeom &geom,
                   const uint32_t keep, int32_t *const scores,
                   uint8_t *const keep_rows, PruneHistogram &scratch,
                   StructuredPlan &plan, uint32_t &threshold_out,
                   uint32_t &tie_group_out, uint32_t &tie_kept_out) noexcept
{
    if (codes == nullptr || scores == nullptr || keep_rows == nullptr)
        return Status::BAD_LAYOUT;
    if (geom._M_valid == 0u)
        return Status::INTERNAL;
    if (!simd_aligned (keep_rows) || !simd_aligned (scores))
        return Status::BAD_LAYOUT;
    const uint32_t rows = geom._M_rows;
    const uint32_t row_len = geom._M_row_len;
    for (uint32_t row = 0u; row < rows; ++row)
        scores[row] = 0;
    /* Pass 1: row L1 over every outer block, stride-1 within a row slice. */
    uint32_t position = 0u;
    for (uint32_t block = 0u; block < geom._M_blocks; ++block)
        {
            for (uint32_t row = 0u; row < rows; ++row)
                {
                    int64_t accumulator = scores[row];
                    for (uint32_t lane = 0u; lane < row_len; ++lane)
                        {
                            const int32_t code = codes[position];
                            ++position;
                            accumulator += code < 0 ? -static_cast<int64_t> (code)
                                                    : static_cast<int64_t> (code);
                        }
                    scores[row] = accumulator > static_cast<int64_t> (_S_prune_score_max)
                                      ? _S_prune_score_max
                                      : static_cast<int32_t> (accumulator);
                }
        }
    if (position != geom._M_count)
        return Status::BAD_LAYOUT;
    /* Largest score sets the rescale, so the row classes share one histogram. */
    int32_t max_score = 0;
    for (uint32_t row = 0u; row < rows; ++row)
        if (scores[row] > max_score)
            max_score = scores[row];
    scratch.reset ();
    for (uint32_t row = 0u; row < rows; ++row)
        {
            const uint32_t magnitude = max_score > 0
                                           ? static_cast<uint32_t> (
                                                 (static_cast<int64_t> (scores[row])
                                                  * 127) / max_score)
                                           : 0u;
            scratch.add (magnitude);
        }
    uint32_t threshold = 0u;
    uint32_t tie_keep = 0u;
    const Status select_status
        = prune_select_threshold (scratch, keep, threshold, tie_keep);
    if (!status_ok (select_status))
        return select_status;
    /* Pass 2 over the scores: the same ascending-index tie rule as the lanes. */
    uint32_t used_ties = 0u;
    uint32_t rows_kept = 0u;
    for (uint32_t row = 0u; row < rows; ++row)
        {
            const uint32_t magnitude = max_score > 0
                                           ? static_cast<uint32_t> (
                                                 (static_cast<int64_t> (scores[row])
                                                  * 127) / max_score)
                                           : 0u;
            uint8_t survives = 0u;
            if (magnitude > threshold)
                survives = 1u;
            else if (magnitude == threshold && used_ties < tie_keep)
                {
                    survives = 1u;
                    ++used_ties;
                }
            keep_rows[row] = survives;
            if (survives != 0u)
                ++rows_kept;
        }
    plan._M_geom = geom;
    plan._M_keep = keep_rows;
    plan._M_rows_kept = rows_kept;
    plan._M_kept_count = geom._M_blocks * rows_kept * row_len;
    threshold_out = threshold;
    tie_group_out = scratch._M_bins[threshold];
    tie_kept_out = tie_keep;
    return Status::OK;
}

/**
 * @brief Compact a code stream to the kept rows: blocks of kept slices, packed.
 *
 * One pass over @p source, stride-1 copies of `row_len` bytes at a time, so a
 * removed channel costs its bytes instead of a mask.  @p destination must hold
 * `plan._M_kept_count` bytes.
 */
[[nodiscard]] inline Status
prune_compact_codes (const int8_t *const source, const StructuredPlan &plan,
                     int8_t *const destination,
                     const uint32_t capacity) noexcept
{
    if (source == nullptr || destination == nullptr || plan._M_keep == nullptr)
        return Status::BAD_LAYOUT;
    if (capacity < plan._M_kept_count)
        return Status::TRUNCATED;
    const uint32_t row_len = plan._M_geom._M_row_len;
    const uint32_t rows = plan._M_geom._M_rows;
    uint32_t read = 0u;
    uint32_t written = 0u;
    for (uint32_t block = 0u; block < plan._M_geom._M_blocks; ++block)
        {
            for (uint32_t row = 0u; row < rows; ++row)
                {
                    if (plan._M_keep[row] != 0u)
                        {
                            for (uint32_t lane = 0u; lane < row_len; ++lane)
                                destination[written + lane] = source[read + lane];
                            written += row_len;
                        }
                    read += row_len;
                }
        }
    return written == plan._M_kept_count ? Status::OK : Status::INTERNAL;
}

/** @brief Same compaction for a Q15 gradient or bias lane stream. */
[[nodiscard]] inline Status
prune_compact_lanes (const int16_t *const source, const StructuredPlan &plan,
                     int16_t *const destination,
                     const uint32_t capacity) noexcept
{
    if (source == nullptr || destination == nullptr || plan._M_keep == nullptr)
        return Status::BAD_LAYOUT;
    if (capacity < plan._M_kept_count)
        return Status::TRUNCATED;
    const uint32_t row_len = plan._M_geom._M_row_len;
    const uint32_t rows = plan._M_geom._M_rows;
    uint32_t read = 0u;
    uint32_t written = 0u;
    for (uint32_t block = 0u; block < plan._M_geom._M_blocks; ++block)
        {
            for (uint32_t row = 0u; row < rows; ++row)
                {
                    if (plan._M_keep[row] != 0u)
                        {
                            for (uint32_t lane = 0u; lane < row_len; ++lane)
                                {
                                    destination[written + lane]
                                        = source[read + lane];
                                }
                            written += row_len;
                        }
                    read += row_len;
                }
        }
    return written == plan._M_kept_count ? Status::OK : Status::INTERNAL;
}

/** @brief Grow @p shape to the compacted row count along its channel axis. */
[[nodiscard]] inline constexpr Status
prune_rewrite_shape (const StructuredPlan &plan, Shape &shape) noexcept
{
    if (plan._M_keep == nullptr || plan._M_geom._M_valid == 0u)
        return Status::BAD_LAYOUT;
    const uint8_t axis = plan._M_geom._M_axis;
    if (axis >= shape._M_rank)
        return Status::BAD_LAYOUT;
    if (shape._M_dims[axis] != plan._M_geom._M_rows)
        return Status::BAD_LAYOUT;
    if (plan._M_rows_kept == 0u || plan._M_rows_kept > _S_max_elements)
        return Status::OUT_OF_RANGE;
    shape._M_dims[axis] = static_cast<uint16_t> (plan._M_rows_kept);
    return status_ok (shape.validate ()) ? Status::OK : Status::OUT_OF_RANGE;
}

/**
 * @brief Dimension rewrite implied by a structured prune.
 *
 * A removed output channel shrinks `out_dim`; a removed input channel shrinks
 * `in_dim`; the axis not being pruned is reported unchanged, so the caller sees
 * both ends of the rewrite in one record.
 */
[[nodiscard]] inline constexpr Status
prune_dims_rewrite (const StructuredPlan &plan, const uint32_t in_dim,
                    const uint32_t out_dim, DimRewrite &out) noexcept
{
    if (plan._M_keep == nullptr || plan._M_geom._M_valid == 0u)
        return Status::BAD_LAYOUT;
    if (in_dim == 0u || out_dim == 0u || in_dim > 0xFFFFu || out_dim > 0xFFFFu)
        return Status::OUT_OF_RANGE;
    const uint32_t kept = plan._M_rows_kept;
    const uint32_t dropped = plan._M_geom._M_rows - kept;
    out._M_in_from = static_cast<uint16_t> (in_dim);
    out._M_out_from = static_cast<uint16_t> (out_dim);
    if (plan._M_geom._M_axis == 0u)
        {
            if (dropped > out_dim)
                return Status::OUT_OF_RANGE;
            out._M_out_to = static_cast<uint16_t> (out_dim - dropped);
            out._M_in_to = static_cast<uint16_t> (in_dim);
        }
    else
        {
            /* An input-side prune drops `dropped * row_len` input lanes per group;
               row_len is the element span of one row, so the descriptor's in_dim
               shrinks by that product, which the caller has sized as in_dim. */
            const uint64_t removed
                = static_cast<uint64_t> (dropped) * plan._M_geom._M_row_len;
            if (removed > in_dim)
                return Status::OUT_OF_RANGE;
            out._M_in_to = static_cast<uint16_t> (in_dim - removed);
            out._M_out_to = static_cast<uint16_t> (out_dim);
        }
    out._M_axis = plan._M_geom._M_axis;
    out._M_valid = 1u;
    out._M_pad = 0u;
    return Status::OK;
}

/**
 * @brief Fold a structured rewrite into a `Layer` descriptor.
 *
 * Rewrites in/out dims, the weight stream length (the surviving rows times the
 * surviving input width) and the arena in/out lengths, then leaves validation to
 * mllee/format.hpp's own `_S_validate_shape` -- which is the interop gate
 * tests/test_mllee_training_quantize.cpp runs on the rewritten descriptor.
 */
[[nodiscard]] inline Status
prune_apply_rewrite (const DimRewrite &rewrite, Layer &layer) noexcept
{
    if (rewrite._M_valid == 0u)
        return Status::BAD_LAYOUT;
    if (layer._M_in_dim != rewrite._M_in_from
        || layer._M_out_dim != rewrite._M_out_from)
        return Status::BAD_LAYOUT;
    if (rewrite._M_in_to == 0u || rewrite._M_out_to == 0u)
        return Status::OUT_OF_RANGE;
    layer._M_in_dim = rewrite._M_in_to;
    layer._M_out_dim = rewrite._M_out_to;
    const uint64_t weight_bytes
        = static_cast<uint64_t> (layer._M_out_dim) * layer._M_in_dim;
    if (weight_bytes > 0xFFFFFFFFull)
        return Status::OUT_OF_RANGE;
    if (layer._M_weight_len != 0u
        && weight_bytes <= static_cast<uint64_t> (_S_max_elements))
        layer._M_weight_len = static_cast<uint32_t> (weight_bytes);
    layer._M_in_len = static_cast<uint16_t> (layer._M_in_dim);
    layer._M_out_len = static_cast<uint16_t> (layer._M_out_dim);
    return Status::OK;
}

/** @brief Zero every lane whose mask bit is 0: the forward half of the contract. */
[[nodiscard]] inline Status
prune_apply_forward (const PruneMask &mask, int8_t *const codes) noexcept
{
    const Status mask_status = mask.validate ();
    if (!status_ok (mask_status))
        return mask_status;
    if (codes == nullptr)
        return Status::BAD_LAYOUT;
    for (uint32_t i = 0; i < mask._M_count; ++i)
        codes[i] = mask._M_bits[i] != 0u ? codes[i] : 0;
    return Status::OK;
}

/** @brief Zero every gradient lane whose mask bit is 0: the backward half. */
[[nodiscard]] inline Status
prune_apply_backward (const PruneMask &mask, const Trainable &weight) noexcept
{
    const Status mask_status = mask.validate ();
    if (!status_ok (mask_status))
        return mask_status;
    if (weight._M_grad == nullptr)
        return Status::BAD_LAYOUT;
    if (weight._M_count != mask._M_count)
        return Status::BAD_LAYOUT;
    for (uint32_t i = 0; i < mask._M_count; ++i)
        weight._M_grad[i] = mask._M_bits[i] != 0u ? weight._M_grad[i] : 0;
    return Status::OK;
}

/**
 * @brief Accumulate a masked gradient into a weight: pruned lanes get nothing.
 *
 * The fused single pass is the scalar form of `route_by_mask` followed by
 * `train_simd::add_grad`; with an aligned scratch available,
 * `prune_route_and_add` below is the vectorized pair.
 *
 * @param saturated_out lanes whose saturating sum clamped
 */
[[nodiscard]] inline Status
prune_accumulate_masked (const Trainable &weight, const uint8_t *const mask,
                         const int16_t *const grad, const uint32_t count,
                         uint32_t &saturated_out) noexcept
{
    if (mask == nullptr || grad == nullptr || weight._M_grad == nullptr)
        return Status::BAD_LAYOUT;
    if (count > _S_prune_max_count || weight._M_count != count)
        return Status::BAD_LAYOUT;
    saturated_out = 0u;
    for (uint32_t i = 0; i < count; ++i)
        if (mask[i] != 0u && weight.add_grad (i, grad[i]))
            ++saturated_out;
    return Status::OK;
}

/**
 * @brief Vectorized masked accumulation through the shared backward atoms.
 *
 * Needs a caller-owned @p scratch of @p count int16 lanes: `route_by_mask` writes
 * the gated gradient there and `train_simd::add_grad` folds it into the weight's
 * lanes -- two stride-1 passes over aligned 4-byte buffers, which is what the SSE2
 * and SSSE3 paths want.
 */
[[nodiscard]] inline Status
prune_route_and_add (const Trainable &weight, const uint8_t *const mask,
                     const int16_t *const grad, int16_t *const scratch,
                     const uint32_t count, uint32_t &saturated_out) noexcept
{
    if (mask == nullptr || grad == nullptr || scratch == nullptr)
        return Status::BAD_LAYOUT;
    if (weight._M_grad == nullptr)
        return Status::BAD_LAYOUT;
    if (count > _S_prune_max_count || weight._M_count != count)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (scratch) || !simd_aligned (weight._M_grad))
        return Status::BAD_LAYOUT;
    route_by_mask (grad, mask, scratch, count);
    saturated_out = train_simd::add_grad (weight._M_grad, scratch, count);
    return Status::OK;
}

/** @brief Enforce a mask on a tensor: codes and gradient lanes both zeroed. */
[[nodiscard]] inline Status
prune_enforce (const PruneMask &mask, const Trainable &weight) noexcept
{
    const Status forward_status = prune_apply_forward (mask, weight._M_code);
    if (!status_ok (forward_status))
        return forward_status;
    return prune_apply_backward (mask, weight);
}

/** @brief Pack a byte mask to one bit per lane, for storage or a checkpoint. */
[[nodiscard]] inline Status
prune_mask_pack (const PruneMask &mask, uint8_t *const packed,
                 const uint32_t capacity) noexcept
{
    const Status mask_status = mask.validate ();
    if (!status_ok (mask_status))
        return mask_status;
    const uint32_t needed = (mask._M_count + 7u) / 8u;
    if (packed == nullptr || capacity < needed)
        return Status::TRUNCATED;
    for (uint32_t byte = 0u; byte < needed; ++byte)
        packed[byte] = 0u;
    for (uint32_t i = 0; i < mask._M_count; ++i)
        {
            if (mask._M_bits[i] == 0u)
                continue;
            packed[i >> 3] = static_cast<uint8_t> (packed[i >> 3]
                                                  | (1u << (i & 7u)));
        }
    return Status::OK;
}

/** @brief Read one lane of a packed mask. */
[[nodiscard]] inline constexpr bool
prune_mask_packed_keeps (const uint8_t *const packed,
                         const uint32_t index) noexcept
{
    return packed != nullptr
           && (packed[index >> 3] & static_cast<uint8_t> (1u << (index & 7u)))
                  != 0u;
}

/** @brief Invert a packed mask: pruned lanes become set, for a sparse index. */
[[nodiscard]] inline Status
prune_mask_pack_inverted (const uint8_t *const packed, const uint32_t count,
                          uint8_t *const out, const uint32_t capacity) noexcept
{
    const uint32_t needed = (count + 7u) / 8u;
    if (packed == nullptr || out == nullptr || capacity < needed)
        return Status::BAD_LAYOUT;
    for (uint32_t byte = 0u; byte < needed; ++byte)
        out[byte] = static_cast<uint8_t> (~packed[byte]);
    const uint32_t spare = needed * 8u - count;
    if (spare != 0u)
        out[needed - 1u] = static_cast<uint8_t> (
            out[needed - 1u] & ((1u << (8u - spare)) - 1u));
    return Status::OK;
}

/** @brief Expand a packed mask back to the byte form the kernels want. */
[[nodiscard]] inline Status
prune_mask_unpack (const uint8_t *const packed, const uint32_t count,
                   uint8_t *const out, const uint32_t capacity,
                   PruneMask &mask) noexcept
{
    if (packed == nullptr || out == nullptr || capacity < count)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (out))
        return Status::BAD_LAYOUT;
    uint32_t pruned = 0u;
    for (uint32_t i = 0; i < count; ++i)
        {
            const bool keeps = prune_mask_packed_keeps (packed, i);
            out[i] = keeps ? 1u : 0u;
            if (!keeps)
                ++pruned;
        }
    mask._M_bits = out;
    mask._M_count = count;
    mask._M_pruned = pruned;
    mask._M_valid = 1u;
    return Status::OK;
}

/** @brief Check a mask against a tensor and recompute its pruned count. */
[[nodiscard]] inline Status
prune_validate_mask (const PruneMask &mask, const Trainable &weight,
                     bool recompute) noexcept
{
    const Status mask_status = mask.validate ();
    if (!status_ok (mask_status))
        return mask_status;
    const Status weight_status = weight.validate ();
    if (!status_ok (weight_status))
        return weight_status;
    if (mask._M_count != weight._M_count)
        return Status::BAD_LAYOUT;
    if (!recompute)
        return Status::OK;
    uint32_t pruned = 0u;
    for (uint32_t i = 0; i < mask._M_count; ++i)
        if (mask._M_bits[i] == 0u)
            ++pruned;
    if (pruned != mask._M_pruned)
        return Status::CORRUPT_STREAM;
    /* A mask that forbids a lane whose code is non-zero is inconsistent with a
       forward pass that already honoured it. */
    for (uint32_t i = 0; i < mask._M_count; ++i)
        if (mask._M_bits[i] == 0u && weight._M_code[i] != 0)
            return Status::CORRUPT_STREAM;
    return Status::OK;
}

/** @brief Count the zero lanes of a code stream: the mask-free sparsity check. */
[[nodiscard]] inline Status
prune_count_zeros (const int8_t *const codes, const uint32_t count,
                   uint32_t &zeros_out, int32_t &sparsity_q15_out) noexcept
{
    if (codes == nullptr)
        return Status::BAD_LAYOUT;
    if (count > _S_prune_max_count)
        return Status::OUT_OF_RANGE;
    uint32_t zeros = 0u;
    for (uint32_t i = 0; i < count; ++i)
        if (codes[i] == 0)
            ++zeros;
    zeros_out = zeros;
    sparsity_q15_out = ratio_q15 (prune_sat_i32 (zeros),
                                  prune_sat_i32 (count == 0u ? 1u : count));
    return Status::OK;
}

/**
 * @brief Prune one tensor, fill its mask, and zero the removed lanes now.
 *
 * The structured pattern is not served here: it rewrites dimensions and compacts
 * streams instead of writing a lane mask, so it has its own entry point
 * (`prune_apply_structured`).  @p histogram is the caller's 516-byte scratch,
 * unused by the N:M path.
 */
[[nodiscard]] inline Status
prune_apply (const PruneConfig &config, const uint32_t step,
             const Trainable &weight, PruneHistogram &histogram, PruneMask &mask,
             PruneReport &report) noexcept
{
    const Status config_status = config.validate ();
    if (!status_ok (config_status))
        return config_status;
    const Status weight_status = weight.validate ();
    if (!status_ok (weight_status))
        return weight_status;
    const PrunePattern pattern = static_cast<PrunePattern> (config._M_pattern);
    if (pattern == PrunePattern::StructuredRows)
        return Status::UNSUPPORTED;
    if (mask._M_bits == nullptr)
        return Status::BAD_LAYOUT;
    const uint32_t count = weight._M_count;
    if (count == 0u)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (mask._M_bits))
        return Status::BAD_LAYOUT;
    const int32_t sparsity = prune_sparsity_at_q15 (config, step);
    report = PruneReport{};
    report._M_elements = count;
    report._M_requested_sparsity_q15 = sparsity;
    mask._M_count = count;
    mask._M_pattern = config._M_pattern;
    mask._M_valid = 0u;
    mask._M_pruned = 0u;
    if (sparsity <= 0)
        {
            /* Nothing requested, but the mask still has to exist as all-keep: the
               backward hook reads it on every step. */
            for (uint32_t i = 0; i < count; ++i)
                mask._M_bits[i] = 1u;
            mask._M_valid = 1u;
            report._M_achieved_sparsity_q15 = 0;
            return Status::OK;
        }
    const uint32_t target_pruned = prune_target_pruned (count, sparsity);
    report._M_target_pruned = target_pruned;
    const uint32_t keep = count - target_pruned;
    if (pattern == PrunePattern::NinM)
        {
            const Status status = prune_mask_ninm (
                weight._M_code, count, config._M_block_n, config._M_block_m, mask,
                report._M_blocks_total, report._M_blocks_partial);
            if (!status_ok (status))
                return status;
        }
    else
        {
            uint32_t threshold = 0u;
            uint32_t tie_group = 0u;
            uint32_t tie_kept = 0u;
            const Status status
                = prune_mask_unstructured (weight._M_code, count, keep, histogram,
                                           mask, threshold, tie_group, tie_kept);
            if (!status_ok (status))
                return status;
            report._M_threshold_magnitude = static_cast<int32_t> (threshold);
            report._M_tie_group = tie_group;
            report._M_tie_kept = tie_kept;
        }
    report._M_pruned = mask._M_pruned;
    report._M_achieved_sparsity_q15 = mask.sparsity_q15 ();
    return prune_apply_forward (mask, weight._M_code);
}

/**
 * @brief Structured variant: drop whole channel rows and compact the stream.
 *
 * Writes the kept rows into @p compacted (which the caller sizes with
 * `plan._M_kept_count` after the selection, so the sizing needs no guess) and
 * zeroes the dropped rows in place in the source tensor, so a forward pass run
 * before the compaction still sees the pruned weights.  The dimension rewrite and
 * the `Layer` fold are separate calls so the caller stays in control of the blob.
 */
[[nodiscard]] inline Status
prune_apply_structured (const PruneConfig &config, const uint32_t step,
                        const Trainable &weight, int32_t *const scores,
                        uint8_t *const keep_rows, PruneHistogram &histogram,
                        int8_t *const compacted, const uint32_t capacity,
                        StructuredPlan &plan, PruneReport &report) noexcept
{
    const Status config_status = config.validate ();
    if (!status_ok (config_status))
        return config_status;
    if (static_cast<PrunePattern> (config._M_pattern)
        != PrunePattern::StructuredRows)
        return Status::BAD_LAYOUT;
    const Status weight_status = weight.validate ();
    if (!status_ok (weight_status))
        return weight_status;
    PruneRowGeom geom;
    const Status geom_status = prune_row_geom (weight._M_shape, config._M_axis,
                                              geom);
    if (!status_ok (geom_status))
        return geom_status;
    const int32_t sparsity = prune_sparsity_at_q15 (config, step);
    report = PruneReport{};
    report._M_elements = weight._M_count;
    report._M_requested_sparsity_q15 = sparsity;
    const uint32_t target_pruned = prune_target_pruned (geom._M_rows, sparsity);
    report._M_target_pruned = target_pruned * geom._M_row_len;
    const uint32_t keep_rows_wanted = geom._M_rows - target_pruned;
    uint32_t threshold = 0u;
    uint32_t tie_group = 0u;
    uint32_t tie_kept = 0u;
    const Status select_status = prune_select_rows (
        weight._M_code, geom, keep_rows_wanted, scores, keep_rows, histogram,
        plan, threshold, tie_group, tie_kept);
    if (!status_ok (select_status))
        return select_status;
    if (capacity < plan._M_kept_count)
        return Status::TRUNCATED;
    const Status compact_status
        = prune_compact_codes (weight._M_code, plan, compacted, capacity);
    if (!status_ok (compact_status))
        return compact_status;
    report._M_threshold_magnitude = static_cast<int32_t> (threshold);
    report._M_tie_group = tie_group;
    report._M_tie_kept = tie_kept;
    report._M_pruned = weight._M_count - plan._M_kept_count;
    report._M_achieved_sparsity_q15
        = ratio_q15 (prune_sat_i32 (report._M_pruned),
                     prune_sat_i32 (weight._M_count));
    return Status::OK;
}

/** @brief Prune a set of tensors with one config each, aggregating the report. */
[[nodiscard]] inline Status
prune_apply_set (const Trainable *const weights, const PruneConfig *const configs,
                 const uint32_t count, const uint32_t step,
                 PruneHistogram &histogram, PruneMask *const masks,
                 PruneReport *const reports, PruneRunReport &run) noexcept
{
    if (weights == nullptr || configs == nullptr || masks == nullptr
        || reports == nullptr)
        return Status::BAD_LAYOUT;
    if (count == 0u || count > _S_prune_max_count)
        return Status::OUT_OF_RANGE;
    run = PruneRunReport{};
    for (uint32_t tensor = 0u; tensor < count; ++tensor)
        {
            const Status status = prune_apply (configs[tensor], step,
                                               weights[tensor], histogram,
                                               masks[tensor], reports[tensor]);
            if (!status_ok (status))
                return status;
            ++run._M_tensors;
            run._M_elements += reports[tensor]._M_elements;
            run._M_pruned += reports[tensor]._M_pruned;
            run._M_blocks_total += reports[tensor]._M_blocks_total;
            run._M_blocks_partial += reports[tensor]._M_blocks_partial;
        }
    run._M_achieved_sparsity_q15 = ratio_q15 (prune_sat_i32 (run._M_pruned),
                                             prune_sat_i32 (run._M_elements));
    run._M_step_sparsity_q15 = prune_sparsity_at_q15 (configs[0], step);
    return Status::OK;
}

/**
 * @brief Backward hook: the one call an op's weight-gradient path makes.
 *
 * A pruned lane must not accumulate *and* must not keep a stale value, so the hook
 * both gates the accumulation and clears the lane: "no gradient, no motion".
 */
[[nodiscard]] inline Status
prune_backward_hook (const Trainable &weight, const PruneMask &mask,
                     const Trainable &grad_source,
                     uint32_t &saturated_out) noexcept
{
    const Status mask_status = mask.validate ();
    if (!status_ok (mask_status))
        return mask_status;
    if (weight._M_grad == nullptr || grad_source._M_grad == nullptr)
        return Status::BAD_LAYOUT;
    if (weight._M_count != mask._M_count
        || grad_source._M_count != weight._M_count)
        return Status::BAD_LAYOUT;
    const Status status = prune_accumulate_masked (weight, mask._M_bits,
                                                  grad_source._M_grad,
                                                  weight._M_count, saturated_out);
    if (!status_ok (status))
        return status;
    return prune_apply_backward (mask, weight);
}

/** @brief Bytes a byte mask over @p count lanes costs. */
[[nodiscard]] inline constexpr uint32_t
prune_mask_bytes (const uint32_t count) noexcept
{
    return align_up (count, _S_pool_alignment);
}

/** @brief Bytes a packed mask over @p count lanes costs. */
[[nodiscard]] inline constexpr uint32_t
prune_mask_packed_bytes (const uint32_t count) noexcept
{
    return align_up ((count + 7u) / 8u, _S_pool_alignment);
}

/** @brief Bytes the structured path's row scores and flags cost. */
[[nodiscard]] inline constexpr uint32_t
prune_row_scratch_bytes (const uint32_t rows) noexcept
{
    return align_up (rows * 4u, _S_pool_alignment) + align_up (rows, _S_pool_alignment);
}
} // namespace zzs::mllee::train

#endif // ZZS_MLLEE_TRAIN_PRUNE_HPP
