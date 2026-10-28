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

#ifndef ZZS_MLLEE_TRAIN_CALIBRATION_HPP
#define ZZS_MLLEE_TRAIN_CALIBRATION_HPP

#include "mllee/format.hpp"
#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Post-training calibration: float-free observers and grid selection.
 *
 * Calibration is the offline half of quantization: run the trained graph over a
 * handful of representative batches, watch the distribution of each tensor, and
 * pick the affine grid that minimizes the error that grid induces.  Everything
 * here stays inside the module's existing numeric conventions -- no second float
 * path and no new fixed-point width:
 *
 *   - an observed value is a Q15 int16 lane (the same lane mllee/train_simd.hpp
 *     moves), or an int8 code widened through `code_to_q15` on the identity grid
 *     (step 258), which is exactly what a code means in this module;
 *   - the observer's support is the whole signed Q15 domain cut into 256 equal
 *     bins of 256 units, so the binning is exact, needs no data-dependent window
 *     and therefore no rescaling between calibration batches;
 *   - a chosen grid is reported as `scale_q15` (one code step, the number
 *     `code_to_q15` uses) and its reciprocal `multiplier_q15`
 *     (= round(32768 / scale_q15)), the form mllee/format.hpp reads back at
 *     `Layer::_M_scale_off` through `_S_i32`;
 *   - every error figure is an integer sum (squared error, or a Q7 divergence
 *     built from the module's own `q7_log_q15`), never a float mean.
 *
 * Selection rules, each in a symmetric and an asymmetric form:
 *
 *   - MinMax     : the exact observed extreme, `scale = ceil(abs_max / 127)`;
 *   - Histogram  : the bound of the top *occupied bin* (edge-based, so it never
 *                  clips, and deliberately sub-bin coarse);
 *   - Percentile : an outlier-aware bound that drops a caller-sized tail -- at
 *                  least `_M_percentile_q15` / 32768 of the samples must survive
 *                  -- by removing equal-mass pairs from both ends;
 *   - Mse        : the candidate grid minimizing the integer sum of squared
 *                  quantization errors over the histogram;
 *   - Kl         : the candidate grid minimizing KL(P||Q), P the observed bin
 *                  distribution and Q the same mass pushed through the candidate
 *                  quantizer.
 *
 * Mse and Kl search the same candidate axis: how much of the observed range the
 * grid is allowed to represent (code counts 127 down to `_M_min_codes` of the
 * rule's bound).  Clipping a rare spike is usually the smaller total error, which
 * is exactly the trade these two objectives make -- and why a percentile bound is
 * the outlier-aware alternative to MinMax.
 *
 * ### Pool cost (all caller-owned; nothing here allocates)
 *
 * | Object | Bytes |
 * |---|---|
 * | `CalibHistogram` (256 bins + exact extremes) | 1036 |
 * | `CalibLayout` / `CalibConfig` | 16 / 16 |
 * | `CalibChoice` | 28 |
 * | `LayerCalibration` (one report row) | 28 |
 * | `CalibrationReport` (descriptor, not the rows) | 48 |
 * | KL scratch (`calib_kl_scratch_bytes`) | 1024 |
 * | inference scale-table emission | 4 · channels |
 * | **one observed tensor, one grid** | 1036 |
 * | **one observed tensor, C channels** | 1036 · C |
 * | report rows for L layers, C channels | 28 · L · C |
 *
 * The intended use is one tensor at a time: observe into one histogram, select,
 * append the row, reset and reuse the same 1036 bytes -- so the calibration
 * working set is one histogram plus the 1 KiB KL scratch, independent of model
 * depth.
 */
namespace zzs::mllee::train
{
/** @brief The five grid-selection rules a calibration run can use. */
enum class CalibMethod : uint8_t
{
    MinMax = 0,
    Histogram = 1,
    Percentile = 2,
    Mse = 3,
    Kl = 4,
};

/** @brief Axis selector meaning "one grid for the whole tensor". */
inline constexpr uint16_t _S_calib_per_tensor = 0xFFFFu;

/** @brief Observer support: 256 signed bins over the whole Q15 domain. */
inline constexpr uint32_t _S_calib_bins = 256u;

/** @brief Bin index is `(value + 32768) >> 8`, i.e. one bin is 256 Q15 units. */
inline constexpr uint32_t _S_calib_bin_shift = 8u;
inline constexpr int32_t _S_calib_bin_width_q15 = 1 << _S_calib_bin_shift;

/** @brief Floor on a grid step, so a degenerate observation cannot give 0. */
inline constexpr int32_t _S_calib_min_scale_q15 = 1;

/** @brief Candidate code counts: 127 (full range) down to 32 (clipped hard). */
inline constexpr int32_t _S_calib_min_codes = 32;
inline constexpr uint32_t _S_calib_candidate_count = 96u;

/** @brief Q15 probability floor that keeps the KL objective finite. */
inline constexpr int32_t _S_calib_prob_floor_q15 = 1;

/** @brief int32 span used to saturate accumulated calibration sums. */
inline constexpr int64_t _S_calib_i32_max = 2147483647LL;

/** @brief Cap of the MSE objective: saturation, not wraparound. */
inline constexpr int64_t _S_calib_mse_cap = 1LL << 60;

/**
 * @brief One observed tensor (or channel): its distribution plus its extremes.
 *
 * The histogram needs no rescale between batches because its support is the whole
 * Q15 domain rather than a data-dependent window; that is what makes an
 * incremental, heap-free observer over many calibration batches exact.  Counts
 * saturate instead of wrapping, because a wrapped count would silently invert a
 * tail and select a *smaller* grid than the data supports.
 */
struct CalibHistogram
{
    uint32_t _M_counts[_S_calib_bins];
    /** @brief Saturating sample count across every observed batch. */
    uint32_t _M_total;
    /** @brief Exact observed extremes, kept so MinMax is sub-bin accurate. */
    int16_t _M_min_q15;
    int16_t _M_max_q15;
    /** @brief Batches folded in, for the report and for coverage checks. */
    uint16_t _M_batches;
    uint16_t _M_pad;

    /** @brief Bin index of a Q15 value over the signed support. */
    [[nodiscard]] static constexpr uint32_t
    bin_of (const int32_t value_q15) noexcept
    {
        const uint32_t shifted
            = static_cast<uint32_t> (q15_clamp (value_q15) + _S_q15_one);
        return shifted >> _S_calib_bin_shift;
    }

    /** @brief Center of a bin in Q15: the value every objective is evaluated at. */
    [[nodiscard]] static constexpr int32_t
    bin_center (const uint32_t bin) noexcept
    {
        return (static_cast<int32_t> (bin) << _S_calib_bin_shift)
               + _S_calib_bin_width_q15 / 2 - _S_q15_one;
    }

    /** @brief Upper edge of a bin in Q15 (32768 for the top bin). */
    [[nodiscard]] static constexpr int32_t
    bin_upper (const uint32_t bin) noexcept
    {
        return (static_cast<int32_t> (bin) + 1) * _S_calib_bin_width_q15
               - _S_q15_one;
    }

    /** @brief Lower edge of a bin in Q15 (-32768 for the bottom bin). */
    [[nodiscard]] static constexpr int32_t
    bin_lower (const uint32_t bin) noexcept
    {
        return static_cast<int32_t> (bin) * _S_calib_bin_width_q15 - _S_q15_one;
    }

    /** @brief Largest Q15 magnitude observed, exactly. */
    [[nodiscard]] constexpr int32_t
    abs_max () const noexcept
    {
        const int32_t negative = this->_M_min_q15 < 0 ? -this->_M_min_q15 : 0;
        const int32_t positive
            = this->_M_max_q15 > 0 ? this->_M_max_q15 : 0;
        return positive > negative ? positive : negative;
    }

    /** @brief Largest magnitude any occupied bin's edge represents. */
    [[nodiscard]] constexpr int32_t
    bin_magnitude (const uint32_t bin) const noexcept
    {
        return CalibHistogram::bin_center (bin) < 0
                   ? -CalibHistogram::bin_lower (bin)
                   : CalibHistogram::bin_upper (bin);
    }
};

static_assert (sizeof (CalibHistogram) == 1036u,
               "the calibration observer is a measured 1036-byte slab");

/** @brief Caller bytes a run of @p channels observers costs. */
[[nodiscard]] inline constexpr uint32_t
calib_histogram_bytes (const uint32_t channels) noexcept
{
    return channels * static_cast<uint32_t> (sizeof (CalibHistogram));
}

/** @brief Caller bytes the KL objective's scratch costs (one observer's bins). */
[[nodiscard]] inline constexpr uint32_t
calib_kl_scratch_bytes () noexcept
{
    return _S_calib_bins * 4u;
}

/** @brief Saturating +1 of an observer count. */
inline constexpr void
calib_count_saturate (uint32_t &field) noexcept
{
    if (field < 0xFFFFFFFFu)
        ++field;
}

/** @brief Reset an observer between tensors; batches are folded, not reset. */
constexpr void
calib_reset (CalibHistogram &histogram) noexcept
{
    for (uint32_t bin = 0; bin < _S_calib_bins; ++bin)
        histogram._M_counts[bin] = 0u;
    histogram._M_total = 0u;
    histogram._M_min_q15 = 0;
    histogram._M_max_q15 = 0;
    histogram._M_batches = 0u;
    histogram._M_pad = 0u;
}

/** @brief Fold a single Q15 value: one bin touch, saturating extremes. */
inline constexpr void
calib_fold_q15 (CalibHistogram &histogram, const int32_t value_q15) noexcept
{
    const int32_t value = q15_clamp (value_q15);
    calib_count_saturate (histogram._M_counts[CalibHistogram::bin_of (value)]);
    if (histogram._M_total == 0u)
        {
            histogram._M_min_q15 = static_cast<int16_t> (value);
            histogram._M_max_q15 = static_cast<int16_t> (value);
        }
    else
        {
            if (value < histogram._M_min_q15)
                histogram._M_min_q15 = static_cast<int16_t> (value);
            if (value > histogram._M_max_q15)
                histogram._M_max_q15 = static_cast<int16_t> (value);
        }
    calib_count_saturate (histogram._M_total);
}

/**
 * @brief Fold one calibration batch of Q15 lanes into an observer.
 *
 * One pass, stride-1 over the caller's int16 stream; the widening happens in
 * registers, so no temporary buffer is carved and nothing is looked up per
 * element.
 */
[[nodiscard]] inline Status
calib_observe_lanes (CalibHistogram &histogram, const int16_t *const values,
                     const uint32_t count) noexcept
{
    if (values == nullptr)
        return Status::BAD_LAYOUT;
    if (count > _S_max_elements)
        return Status::OUT_OF_RANGE;
    for (uint32_t i = 0; i < count; ++i)
        calib_fold_q15 (histogram, values[i]);
    if (count != 0u && histogram._M_batches < 0xFFFFu)
        ++histogram._M_batches;
    return Status::OK;
}

/**
 * @brief Fold one calibration batch of int8 codes in on the identity grid.
 *
 * A code's value is `code * 258`, so calibrating from already-quantized
 * activations -- the normal case for a frozen inference graph -- uses the same
 * support and the same selection code as the lane path.
 */
[[nodiscard]] inline Status
calib_observe_codes (CalibHistogram &histogram, const int8_t *const codes,
                     const uint32_t count) noexcept
{
    if (codes == nullptr)
        return Status::BAD_LAYOUT;
    if (count > _S_max_elements)
        return Status::OUT_OF_RANGE;
    for (uint32_t i = 0; i < count; ++i)
        calib_fold_q15 (histogram, code_to_q15 (codes[i]));
    if (count != 0u && histogram._M_batches < 0xFFFFu)
        ++histogram._M_batches;
    return Status::OK;
}

/**
 * @brief Channel geometry of an axis-wise observer: how a buffer splits into runs.
 *
 * For an axis-wise grid channel @p c occupies the contiguous runs of
 * `stride(axis)` elements repeating every `dim(axis) * stride(axis)`, so the inner
 * loop stays stride-1 over a 4-byte-aligned buffer and the histogram pointer is
 * hoisted per run.
 */
struct CalibLayout
{
    uint32_t _M_count = 0u;
    /** @brief Elements between two channel runs of the same channel. */
    uint32_t _M_period = 0u;
    /** @brief Run length in elements. */
    uint32_t _M_run = 0u;
    uint16_t _M_channels = 1u;
    uint8_t _M_axis = 0u;
    /** @brief Set by a successful calib_layout() so a stale layout cannot be used. */
    uint8_t _M_valid = 0u;
};

static_assert (sizeof (CalibLayout) == 16u,
               "calibration layout must stay a 16-byte descriptor");

/** @brief Derive the channel runs of @p shape along @p axis. */
[[nodiscard]] inline constexpr Status
calib_layout (const Shape &shape, const uint16_t axis, CalibLayout &out) noexcept
{
    const Status shape_status = shape.validate ();
    if (!status_ok (shape_status))
        return shape_status;
    out._M_valid = 0u;
    const uint32_t count = shape.elements ();
    if (axis == _S_calib_per_tensor)
        {
            out._M_count = count;
            out._M_run = count;
            out._M_period = count;
            out._M_channels = 1u;
            out._M_axis = 0u;
            out._M_valid = 1u;
            return Status::OK;
        }
    if (axis >= shape._M_rank)
        return Status::BAD_LAYOUT;
    const uint32_t channels = shape._M_dims[axis];
    if (channels == 0u || channels > _S_max_elements)
        return Status::BAD_LAYOUT;
    const uint32_t run = shape.stride (static_cast<uint8_t> (axis));
    out._M_count = count;
    out._M_run = run;
    out._M_period = run * channels;
    out._M_channels = static_cast<uint16_t> (channels);
    out._M_axis = static_cast<uint8_t> (axis);
    out._M_valid = 1u;
    return Status::OK;
}

/** @brief Fold one batch into an axis-wise observer array: one pass per run. */
[[nodiscard]] inline Status
calib_observe_axiswise (CalibHistogram *const histograms,
                        const CalibLayout &layout, const int16_t *const values,
                        const int8_t *const codes) noexcept
{
    if (histograms == nullptr || layout._M_valid == 0u)
        return Status::BAD_LAYOUT;
    if ((values == nullptr) == (codes == nullptr))
        return Status::BAD_LAYOUT;
    if (layout._M_run == 0u || layout._M_period == 0u
        || layout._M_channels == 0u)
        return Status::BAD_LAYOUT;
    uint32_t start = 0u;
    while (start < layout._M_count)
        {
            const uint32_t channel
                = (start % layout._M_period) / layout._M_run;
            if (channel >= layout._M_channels)
                return Status::BAD_LAYOUT;
            const uint32_t available = layout._M_count - start;
            const uint32_t run
                = layout._M_run < available ? layout._M_run : available;
            const Status status
                = values != nullptr
                      ? calib_observe_lanes (histograms[channel],
                                             values + start, run)
                      : calib_observe_codes (histograms[channel], codes + start,
                                             run);
            if (!status_ok (status))
                return status;
            start += run;
        }
    return Status::OK;
}

/** @brief Reset an observer array. */
inline void
calib_reset_axiswise (CalibHistogram *const histograms,
                      const uint32_t channels) noexcept
{
    if (histograms == nullptr)
        return;
    for (uint32_t channel = 0; channel < channels; ++channel)
        calib_reset (histograms[channel]);
}

/**
 * @name Grid arithmetic
 *
 * `scale_q15` is one code step in Q15 units (258 is the identity grid, code 127 =
 * 32766) and `multiplier_q15` is its reciprocal against 2^15, i.e.
 * `code = round(value_q15 * multiplier_q15 / 32768)`.  Both are the numbers
 * mllee/format.hpp carries in a layer's scale table, so nothing converts later.
 * @{
 */

/** @brief Smallest code step that still represents @p abs_max_q15 at code 127. */
[[nodiscard]] inline constexpr int32_t
calib_scale_from_abs_max (const int32_t abs_max_q15,
                          const int32_t min_scale_q15) noexcept
{
    const int32_t floor_scale
        = min_scale_q15 > _S_calib_min_scale_q15 ? min_scale_q15
                                                : _S_calib_min_scale_q15;
    if (abs_max_q15 <= 0)
        return floor_scale;
    const int32_t scale = (abs_max_q15 + 126) / 127; /* ceil, unsigned domain */
    return scale > floor_scale ? scale : floor_scale;
}

/** @brief round(32768 / scale_q15): the Q15 -> code multiplier. */
[[nodiscard]] inline constexpr int32_t
calib_multiplier_from_scale (const int32_t scale_q15) noexcept
{
    if (scale_q15 <= 0)
        return 0;
    /* round-half-up division of 32768 by the step: (2*32768 + d) / (2*d). */
    const uint32_t denominator = static_cast<uint32_t> (scale_q15) * 2u;
    const uint32_t quotient = (65536u + denominator) / denominator;
    return quotient == 0u ? 1 : static_cast<int32_t> (quotient);
}

/** @brief Value one code represents under the grid, clamped to Q15. */
[[nodiscard]] inline constexpr int32_t
calib_dequantize_q15 (const int32_t code, const int32_t scale_q15,
                      const int32_t zero_point) noexcept
{
    const int64_t value
        = static_cast<int64_t> (code - zero_point) * scale_q15;
    return q15_clamp (value > _S_calib_i32_max ? static_cast<int32_t> (_S_calib_i32_max)
                                               : static_cast<int32_t> (value));
}

/**
 * @brief Code a Q15 value lands on, clipped to the grid.
 *
 * Uses the module's own `_S_round_shift`, so a calibration estimate and a
 * fake-quant forward pass agree on every value including the ties, and the
 * -128 code is never produced (`_S_clamp_i8` bounds to [-127, 127]).
 */
[[nodiscard]] inline constexpr int32_t
calib_quantize_code (const int32_t value_q15, const int32_t multiplier_q15,
                     const int32_t zero_point) noexcept
{
    const int32_t raw = _S_round_shift (
        static_cast<int64_t> (q15_clamp (value_q15)) * multiplier_q15, 15u);
    return _S_clamp_i8 (raw + zero_point);
}

/** @brief Round trip of one value under one grid: quantize then dequantize. */
[[nodiscard]] inline constexpr int32_t
calib_round_trip_q15 (const int32_t value_q15, const int32_t scale_q15,
                      const int32_t multiplier_q15,
                      const int32_t zero_point) noexcept
{
    return calib_dequantize_q15 (
        calib_quantize_code (value_q15, multiplier_q15, zero_point), scale_q15,
        zero_point);
}

/**
 * @brief Zero point of an affine grid over [lo, hi]: the code that means 0.
 *
 * `scale = ceil((hi - lo) / 254)` gives the 254 codes the module can emit room for
 * the whole range, and the zero point follows from the lower end so `lo` lands on
 * code 0.  Both clamp to the code range, so a degenerate range still yields a
 * legal grid.
 */
[[nodiscard]] inline constexpr Status
calib_asymmetric_grid (const int32_t lo_q15, const int32_t hi_q15,
                       const int32_t min_scale_q15, int32_t &scale_q15,
                       int32_t &zero_point) noexcept
{
    if (hi_q15 < lo_q15)
        return Status::BAD_LAYOUT;
    const int64_t span = static_cast<int64_t> (hi_q15) - lo_q15;
    const int32_t span_i32
        = span > _S_calib_i32_max ? static_cast<int32_t> (_S_calib_i32_max)
                                  : static_cast<int32_t> (span);
    int32_t scale = (span_i32 + 253) / 254;
    if (scale < min_scale_q15)
        scale = min_scale_q15 > _S_calib_min_scale_q15 ? min_scale_q15
                                                      : _S_calib_min_scale_q15;
    const int32_t multiplier = calib_multiplier_from_scale (scale);
    const int32_t point
        = _S_clamp_i8 (_S_round_shift (static_cast<int64_t> (-lo_q15) * multiplier,
                                       15u));
    scale_q15 = scale;
    zero_point = point;
    return Status::OK;
}

/** @} */

/**
 * @brief Outlier-aware percentile bound.
 *
 * At least `_M_percentile_q15` / 32768 of the samples must survive, so the
 * droppable budget is `(32768 - percentile) * total / 32768` samples.  The tail is
 * removed in equal-mass *pairs* -- the lowest occupied bin and the highest
 * occupied bin are dropped together -- because a symmetric grid has to bound both
 * tails; a pair is dropped only while the whole pair still fits the budget, which
 * is also the tie rule when one side of a pair is empty.  The bound is read at bin
 * *edges*, never at sub-bin centres, and is then capped by the exact observed
 * maximum, so it never invents headroom and never clips a surviving sample.
 */
[[nodiscard]] inline Status
calib_percentile_bound (const CalibHistogram &histogram,
                        const int32_t percentile_q15, int32_t &out) noexcept
{
    out = 0;
    if (percentile_q15 < 0 || percentile_q15 > _S_q15_one)
        return Status::OUT_OF_RANGE;
    if (histogram._M_total == 0u)
        {
            out = _S_calib_bin_width_q15;
            return Status::OK;
        }
    const int32_t allowed = _S_q15_one - percentile_q15;
    const int64_t budget
        = (static_cast<int64_t> (histogram._M_total) * allowed) / _S_q15_one;
    uint32_t low = 0u;
    uint32_t high = _S_calib_bins - 1u;
    int64_t dropped = 0;
    while (low < high)
        {
            const int64_t pair
                = static_cast<int64_t> (histogram._M_counts[high])
                  + static_cast<int64_t> (histogram._M_counts[low]);
            if (dropped + pair > budget)
                break;
            dropped += pair;
            ++low;
            --high;
        }
    int32_t bound = histogram.bin_magnitude (high);
    const int32_t mirrored = histogram.bin_magnitude (low);
    if (mirrored > bound)
        bound = mirrored;
    const int32_t exact = histogram.abs_max ();
    if (bound > exact && exact >= _S_calib_bin_width_q15)
        bound = exact;
    if (bound < _S_calib_bin_width_q15)
        bound = _S_calib_bin_width_q15;
    out = bound;
    return Status::OK;
}

/** @brief Edge bound of the top occupied bin: the histogram rule. */
[[nodiscard]] inline constexpr int32_t
calib_histogram_bound (const CalibHistogram &histogram) noexcept
{
    int32_t bound = 0;
    for (uint32_t bin = 0; bin < _S_calib_bins; ++bin)
        {
            if (histogram._M_counts[bin] == 0u)
                continue;
            const int32_t magnitude = histogram.bin_magnitude (bin);
            if (magnitude > bound)
                bound = magnitude;
        }
    return bound < _S_calib_bin_width_q15 ? _S_calib_bin_width_q15 : bound;
}

/** @brief The range a rule reads out of one histogram: see `calib_bound`. */

/**
 * @brief One calibrated grid plus the error it costs.
 *
 * `_M_worst_error_q15` is the largest absolute round-trip error over the observed
 * histogram with clip error included: a grid that clips a spike reports that
 * spike's full distance here, which is what makes the report actionable.
 * `_M_objective_q15` is the winning objective, int32-saturated (an MSE sum for
 * Mse, a Q7 divergence sum for Kl) and only comparable within its own method.
 */
struct CalibChoice
{
    int32_t _M_scale_q15 = _S_code_to_q15_scale;
    int32_t _M_multiplier_q15 = 127;
    /** @brief Code that represents value 0; always 0 for the symmetric form. */
    int32_t _M_zero_point = 0;
    /** @brief Bound the grid was built from (Q15), i.e. the represented range. */
    int32_t _M_bound_q15 = 0;
    int32_t _M_worst_error_q15 = 0;
    int32_t _M_objective_q15 = 0;
    uint8_t _M_method = 0u;
    uint8_t _M_asymmetric = 0u;
    /** @brief Non-zero when the observer was empty, so this is a default grid. */
    uint8_t _M_degenerate = 0u;
    uint8_t _M_codes = 127u;

    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (this->_M_scale_q15 <= 0 || this->_M_multiplier_q15 <= 0)
            return Status::OUT_OF_RANGE;
        if (this->_M_zero_point < -127 || this->_M_zero_point > 127)
            return Status::OUT_OF_RANGE;
        if (this->_M_method > static_cast<uint8_t> (CalibMethod::Kl))
            return Status::UNSUPPORTED;
        return Status::OK;
    }
};

static_assert (sizeof (CalibChoice) == 28u,
               "a calibration choice must stay a 28-byte POD");

/**
 * @brief Observer configuration: the rule, the grid form, the channel axis.
 */
struct CalibConfig
{
    uint8_t _M_method = static_cast<uint8_t> (CalibMethod::Mse);
    /** @brief 1 selects a zero-point grid; 0 keeps the symmetric form. */
    uint8_t _M_asymmetric = 0u;
    /** @brief Channel axis, or _S_calib_per_tensor for one grid per tensor. */
    uint16_t _M_axis = _S_calib_per_tensor;
    /** @brief Percentile target in Q15 (32768 == keep every sample). */
    int32_t _M_percentile_q15 = 32409;
    /** @brief Grid-step floor, so one quiet batch cannot collapse the scale. */
    int32_t _M_min_scale_q15 = _S_calib_bin_width_q15;
    /** @brief Candidate code counts stop here (1..127, 32 by default). */
    int32_t _M_min_codes = _S_calib_min_codes;

    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (this->_M_method > static_cast<uint8_t> (CalibMethod::Kl))
            return Status::UNSUPPORTED;
        if (this->_M_percentile_q15 < 0
            || this->_M_percentile_q15 > _S_q15_one)
            return Status::OUT_OF_RANGE;
        if (this->_M_min_scale_q15 < _S_calib_min_scale_q15)
            return Status::OUT_OF_RANGE;
        if (this->_M_min_codes < 1 || this->_M_min_codes > 127)
            return Status::OUT_OF_RANGE;
        if (this->_M_axis != _S_calib_per_tensor && this->_M_axis >= _S_max_rank)
            return Status::BAD_LAYOUT;
        return Status::OK;
    }
};

static_assert (sizeof (CalibConfig) == 16u,
               "calibration config must stay a 16-byte POD");

[[nodiscard]] inline Status
calib_bound (const CalibHistogram &histogram, const CalibConfig &config,
             int32_t &out) noexcept
{
    switch (static_cast<CalibMethod> (config._M_method))
        {
        case CalibMethod::MinMax:
        case CalibMethod::Mse:
        case CalibMethod::Kl:
            out = histogram.abs_max ();
            return Status::OK;
        case CalibMethod::Histogram:
            out = calib_histogram_bound (histogram);
            return Status::OK;
        case CalibMethod::Percentile:
            return calib_percentile_bound (histogram, config._M_percentile_q15,
                                           out);
        }
    return Status::UNSUPPORTED;
}

/** @brief Sweep the histogram: largest round-trip error, clip error included. */
[[nodiscard]] inline constexpr int32_t
calib_worst_error (const CalibHistogram &histogram, const int32_t scale_q15,
                   const int32_t multiplier_q15,
                   const int32_t zero_point) noexcept
{
    int32_t worst = 0;
    for (uint32_t bin = 0; bin < _S_calib_bins; ++bin)
        {
            if (histogram._M_counts[bin] == 0u)
                continue;
            const int32_t value = CalibHistogram::bin_center (bin);
            const int32_t back = calib_round_trip_q15 (value, scale_q15,
                                                      multiplier_q15,
                                                      zero_point);
            const int32_t error = back > value ? back - value : value - back;
            if (error > worst)
                worst = error;
        }
    return worst;
}

/**
 * @brief Sum of squared round-trip errors over the histogram, weighted by counts.
 *
 * Every candidate is scored on the same support, so the objective is compared as
 * an integer sum with no division: it is monotone in the mean square error, which
 * is what the selection needs, and it saturates rather than wrapping.
 */
[[nodiscard]] inline int64_t
calib_mse_objective (const CalibHistogram &histogram, const int32_t scale_q15,
                     const int32_t multiplier_q15,
                     const int32_t zero_point) noexcept
{
    int64_t total = 0;
    for (uint32_t bin = 0; bin < _S_calib_bins; ++bin)
        {
            const uint32_t count = histogram._M_counts[bin];
            if (count == 0u)
                continue;
            const int32_t value = CalibHistogram::bin_center (bin);
            const int32_t back = calib_round_trip_q15 (value, scale_q15,
                                                      multiplier_q15,
                                                      zero_point);
            const int64_t delta = back - value;
            total += delta * delta * static_cast<int64_t> (count);
            if (total > _S_calib_mse_cap)
                return _S_calib_mse_cap;
        }
    return total;
}

/**
 * @brief KL(P||Q) over the observer support, in Q7 units, caller-scratch.
 *
 * P is the observed bin distribution.  Q is the same mass pushed through the
 * candidate quantizer: bin i's count moves to the bin holding the reconstructed
 * centre of bin i, so Q keeps P's total and only redistributes it -- and a grid
 * that vacates a bin entirely is penalized through the Q15 probability floor.
 * Bins with no observed mass contribute nothing (the 0 log 0 convention).
 *
 * @p scratch must be 4-byte aligned and hold `calib_kl_scratch_bytes ()` bytes;
 * it is used for the pushed distribution only, never as hidden state.
 */
[[nodiscard]] inline Status
calib_kl_objective (const CalibHistogram &histogram, const int32_t scale_q15,
                    const int32_t multiplier_q15, const int32_t zero_point,
                    uint32_t *const scratch, int32_t &out) noexcept
{
    if (scratch == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (scratch))
        return Status::BAD_LAYOUT;
    if (histogram._M_total == 0u)
        {
            out = 0;
            return Status::OK;
        }
    for (uint32_t bin = 0; bin < _S_calib_bins; ++bin)
        scratch[bin] = 0u;
    for (uint32_t bin = 0; bin < _S_calib_bins; ++bin)
        {
            const uint32_t count = histogram._M_counts[bin];
            if (count == 0u)
                continue;
            const int32_t value = CalibHistogram::bin_center (bin);
            const int32_t back = calib_round_trip_q15 (value, scale_q15,
                                                      multiplier_q15,
                                                      zero_point);
            calib_count_saturate (scratch[CalibHistogram::bin_of (back)]);
        }
    const int32_t total = static_cast<int32_t> (
        histogram._M_total > static_cast<uint32_t> (_S_calib_i32_max)
            ? static_cast<int32_t> (_S_calib_i32_max)
            : static_cast<int32_t> (histogram._M_total));
    int32_t divergence = 0;
    for (uint32_t bin = 0; bin < _S_calib_bins; ++bin)
        {
            const uint32_t observed = histogram._M_counts[bin];
            if (observed == 0u)
                continue;
            const int32_t p_q15 = ratio_q15 (static_cast<int32_t> (observed),
                                             total);
            const uint32_t pushed
                = scratch[bin] == 0u ? _S_calib_prob_floor_q15 : scratch[bin];
            const int32_t q_q15 = ratio_q15 (
                static_cast<int32_t> (pushed > static_cast<uint32_t> (_S_calib_i32_max)
                                          ? static_cast<int32_t> (_S_calib_i32_max)
                                          : static_cast<int32_t> (pushed)),
                total);
            const int32_t log_ratio
                = q7_log_q15 (p_q15 < _S_calib_prob_floor_q15
                                  ? _S_calib_prob_floor_q15
                                  : p_q15)
                  - q7_log_q15 (q_q15 < _S_calib_prob_floor_q15
                                    ? _S_calib_prob_floor_q15
                                    : q_q15);
            divergence = q15_add_sat (divergence, q15_mul (p_q15, log_ratio));
        }
    out = divergence;
    return Status::OK;
}

/**
 * @brief Build the grid a candidate range produces.
 *
 * The candidate axis for Mse and Kl is *how much of the observed range the grid
 * represents*: @p bound_q15 below the observed maximum means the rest is clipped.
 * The symmetric form puts that bound on code 127; the asymmetric form clips the
 * exact [min, max] range to [-bound, bound] first, so a lopsided distribution
 * still gets its zero point from the surviving lower end.
 */
[[nodiscard]] inline Status
calib_grid_for_bound (const CalibHistogram &histogram, const CalibConfig &config,
                      const int32_t bound_q15, int32_t &scale_q15,
                      int32_t &multiplier_q15, int32_t &zero_point) noexcept
{
    if (bound_q15 <= 0)
        return Status::BAD_LAYOUT;
    if (config._M_asymmetric != 0u)
        {
            int32_t lo = histogram._M_min_q15;
            int32_t hi = histogram._M_max_q15;
            if (hi > bound_q15)
                hi = bound_q15;
            if (lo < -bound_q15)
                lo = -bound_q15;
            const Status grid_status = calib_asymmetric_grid (
                lo, hi, config._M_min_scale_q15, scale_q15, zero_point);
            if (!status_ok (grid_status))
                return grid_status;
        }
    else
        {
            scale_q15
                = (bound_q15 + 126) / 127 > config._M_min_scale_q15
                      ? (bound_q15 + 126) / 127
                      : config._M_min_scale_q15;
            if (scale_q15 < _S_calib_min_scale_q15)
                scale_q15 = _S_calib_min_scale_q15;
            zero_point = 0;
        }
    multiplier_q15 = calib_multiplier_from_scale (scale_q15);
    if (multiplier_q15 <= 0)
        return Status::OUT_OF_RANGE;
    return Status::OK;
}

/** @brief Choice for the rules that read a bound and stop there. */
[[nodiscard]] inline Status
calib_choice_from_bound (const CalibHistogram &histogram,
                         const CalibConfig &config, const int32_t bound_q15,
                         CalibChoice &out) noexcept
{
    int32_t scale = 0;
    int32_t multiplier = 0;
    int32_t zero_point = 0;
    const Status grid_status = calib_grid_for_bound (histogram, config, bound_q15,
                                                   scale, multiplier,
                                                   zero_point);
    if (!status_ok (grid_status))
        return grid_status;
    out._M_scale_q15 = scale;
    out._M_multiplier_q15 = multiplier;
    out._M_zero_point = zero_point;
    out._M_bound_q15 = bound_q15;
    out._M_method = config._M_method;
    out._M_asymmetric = config._M_asymmetric;
    out._M_degenerate = histogram._M_total == 0u ? 1u : 0u;
    out._M_codes = static_cast<uint8_t> (
        bound_q15 / scale > 127 ? 127 : bound_q15 / scale);
    out._M_worst_error_q15 = calib_worst_error (histogram, scale, multiplier,
                                                zero_point);
    out._M_objective_q15 = 0;
    return Status::OK;
}

/**
 * @brief Search the candidate grids and keep the best by @p is_kl.
 *
 * Candidates are the code counts 127 down to `config._M_min_codes`, i.e. the
 * represented range from the full observed bound down to a hard clip; the grid for
 * each is `calib_grid_for_bound`.  Scanning from the widest range and keeping the
 * first *strict* minimum is the documented tie rule: on a plateau the wider range
 * wins, because it never clips more than the narrower one.
 */
[[nodiscard]] inline Status
calib_search (const CalibHistogram &histogram, const CalibConfig &config,
              const int32_t bound_q15, const bool is_kl,
              uint32_t *const scratch, CalibChoice &out) noexcept
{
    if (bound_q15 <= 0)
        return Status::BAD_LAYOUT;
    if (is_kl && scratch == nullptr)
        return Status::BAD_LAYOUT;
    int64_t best_objective = 0;
    int32_t best_scale = 0;
    int32_t best_multiplier = 0;
    int32_t best_zero_point = 0;
    int32_t best_codes = 0;
    for (int32_t codes = 127; codes >= config._M_min_codes; --codes)
        {
            /* bound * codes / 127, rounded up: the represented range of this
               candidate, so codes == 127 is the unclipped bound.  bound_q15 is
               only checked for being positive above, so an out-of-domain value
               saturates at the lane instead of wrapping into a negative bound
               that the candidate loop would silently skip. */
            const int64_t candidate_wide
                = (static_cast<int64_t> (bound_q15) * codes + 126) / 127;
            if (candidate_wide <= 0)
                continue;
            const int32_t candidate = static_cast<int32_t> (
                candidate_wide > _S_q15_max ? _S_q15_max : candidate_wide);
            int32_t scale = 0;
            int32_t multiplier = 0;
            int32_t zero_point = 0;
            const Status grid_status = calib_grid_for_bound (
                histogram, config, candidate, scale, multiplier, zero_point);
            if (!status_ok (grid_status))
                return grid_status;
            int64_t objective = 0;
            if (is_kl)
                {
                    int32_t divergence = 0;
                    const Status kl_status = calib_kl_objective (
                        histogram, scale, multiplier, zero_point, scratch,
                        divergence);
                    if (!status_ok (kl_status))
                        return kl_status;
                    objective = divergence;
                }
            else
                objective = calib_mse_objective (histogram, scale, multiplier,
                                                 zero_point);
            if (best_codes == 0 || objective < best_objective)
                {
                    best_codes = codes;
                    best_objective = objective;
                    best_scale = scale;
                    best_multiplier = multiplier;
                    best_zero_point = zero_point;
                }
        }
    if (best_codes == 0)
        return Status::INTERNAL;
    out._M_scale_q15 = best_scale;
    out._M_multiplier_q15 = best_multiplier;
    out._M_zero_point = best_zero_point;
    out._M_bound_q15 = bound_q15;
    out._M_method = config._M_method;
    out._M_asymmetric = config._M_asymmetric;
    out._M_degenerate = histogram._M_total == 0u ? 1u : 0u;
    out._M_codes = static_cast<uint8_t> (best_codes);
    out._M_objective_q15 = best_objective > _S_calib_i32_max
                               ? static_cast<int32_t> (_S_calib_i32_max)
                               : (best_objective < -_S_calib_i32_max
                                  ? static_cast<int32_t> (-_S_calib_i32_max)
                                      : static_cast<int32_t> (best_objective));
    out._M_worst_error_q15 = calib_worst_error (histogram, best_scale,
                                                best_multiplier,
                                                best_zero_point);
    return Status::OK;
}

/** @brief Select the grid for one observed tensor (or one observed channel). */
[[nodiscard]] inline Status
calib_select (const CalibHistogram &histogram, const CalibConfig &config,
              CalibChoice &out, uint32_t *const kl_scratch = nullptr) noexcept
{
    const Status config_status = config.validate ();
    if (!status_ok (config_status))
        return config_status;
    out = CalibChoice{};
    if (histogram._M_total == 0u)
        {
            /* Nothing observed: emit the identity grid and flag the row, rather
               than inventing a step of 1 that would clip every later value. */
            out._M_scale_q15 = _S_code_to_q15_scale;
            out._M_multiplier_q15
                = calib_multiplier_from_scale (_S_code_to_q15_scale);
            out._M_method = config._M_method;
            out._M_asymmetric = 0u;
            out._M_degenerate = 1u;
            out._M_codes = 127u;
            return Status::OK;
        }
    int32_t bound = 0;
    const Status bound_status = calib_bound (histogram, config, bound);
    if (!status_ok (bound_status))
        return bound_status;
    if (bound < _S_calib_bin_width_q15)
        bound = _S_calib_bin_width_q15;
    const CalibMethod method = static_cast<CalibMethod> (config._M_method);
    if (method == CalibMethod::Mse || method == CalibMethod::Kl)
        return calib_search (histogram, config, bound,
                             method == CalibMethod::Kl, kl_scratch, out);
    return calib_choice_from_bound (histogram, config, bound, out);
}

/** @brief Select one grid per observed channel. */
[[nodiscard]] inline Status
calib_select_axiswise (const CalibHistogram *const histograms,
                       const uint32_t channels, const CalibConfig &config,
                       CalibChoice *const out,
                       uint32_t *const kl_scratch = nullptr) noexcept
{
    if (histograms == nullptr || out == nullptr)
        return Status::BAD_LAYOUT;
    if (channels == 0u || channels > _S_max_elements)
        return Status::OUT_OF_RANGE;
    for (uint32_t channel = 0; channel < channels; ++channel)
        {
            const Status status = calib_select (histograms[channel], config,
                                                out[channel], kl_scratch);
            if (!status_ok (status))
                return status;
        }
    return Status::OK;
}

/** @brief One report row: a layer channel with its grid and its worst error. */
struct LayerCalibration
{
    uint16_t _M_layer = 0u;
    uint16_t _M_channel = 0u;
    int32_t _M_scale_q15 = 0;
    int32_t _M_multiplier_q15 = 0;
    int32_t _M_zero_point = 0;
    /** @brief Largest round-trip error the grid produced, Q15. */
    int32_t _M_worst_error_q15 = 0;
    /** @brief Observed bound the grid was built from, Q15. */
    int32_t _M_bound_q15 = 0;
    uint8_t _M_method = 0u;
    uint8_t _M_flags = 0u;
    /** @brief Saturating low half of the observer's sample count. */
    uint16_t _M_samples = 0u;
};

static_assert (sizeof (LayerCalibration) == 28u,
               "one calibration report row is 28 bytes by design");

/** @brief Flag bit: the row came from an asymmetric (zero-point) grid. */
inline constexpr uint8_t _S_calib_flag_asymmetric = 1u << 0;
/** @brief Flag bit: the observer was empty, so the grid is a default. */
inline constexpr uint8_t _S_calib_flag_degenerate = 1u << 1;

/**
 * @brief Per-layer calibration report over caller-owned rows.
 *
 * The rows are the artifact a quantizer consumes: one chosen grid plus the worst
 * error it caused, so a regression shows up per layer instead of as one global
 * number.  Rows are appended through `calib_report_add`, which fails with
 * ARENA_OVERFLOW when the caller sized the array too short -- nothing is
 * reallocated here.
 */
struct CalibrationReport
{
    CalibConfig _M_config;
    LayerCalibration *_M_entries = nullptr;
    uint32_t _M_capacity = 0u;
    uint32_t _M_count = 0u;
    uint32_t _M_batches = 0u;
    /** @brief Saturating total of observed samples across all rows. */
    uint32_t _M_elements = 0u;
    int32_t _M_worst_error_q15 = 0;
    uint16_t _M_worst_entry = 0u;
    uint16_t _M_pad = 0u;
};

static_assert (sizeof (CalibrationReport) == (sizeof (void *) == 4u ? 44u : 48u),
               "the report descriptor is config plus a pointer and seven words");

/** @brief Bind a report to caller-owned row storage. */
[[nodiscard]] inline Status
calib_report_open (CalibrationReport &report, const CalibConfig &config,
                   LayerCalibration *const entries,
                   const uint32_t capacity) noexcept
{
    const Status config_status = config.validate ();
    if (!status_ok (config_status))
        return config_status;
    if (entries == nullptr && capacity != 0u)
        return Status::BAD_LAYOUT;
    if (capacity != 0u && !simd_aligned (entries))
        return Status::BAD_LAYOUT;
    report._M_config = config;
    report._M_entries = entries;
    report._M_capacity = capacity;
    report._M_count = 0u;
    report._M_batches = 0u;
    report._M_elements = 0u;
    report._M_worst_error_q15 = 0;
    report._M_worst_entry = 0u;
    report._M_pad = 0u;
    return Status::OK;
}

/** @brief Append one selected grid; keeps the running worst layer and error. */
[[nodiscard]] inline Status
calib_report_add (CalibrationReport &report, const uint32_t layer_index,
                  const uint32_t channel, const CalibChoice &choice,
                  const uint32_t samples) noexcept
{
    if (report._M_entries == nullptr)
        return Status::BAD_LAYOUT;
    const Status choice_status = choice.validate ();
    if (!status_ok (choice_status))
        return choice_status;
    if (report._M_count >= report._M_capacity)
        return Status::ARENA_OVERFLOW;
    if (layer_index > 0xFFFFu || channel > 0xFFFFu)
        return Status::OUT_OF_RANGE;
    LayerCalibration &row = report._M_entries[report._M_count];
    row._M_layer = static_cast<uint16_t> (layer_index);
    row._M_channel = static_cast<uint16_t> (channel);
    row._M_scale_q15 = choice._M_scale_q15;
    row._M_multiplier_q15 = choice._M_multiplier_q15;
    row._M_zero_point = choice._M_zero_point;
    row._M_worst_error_q15 = choice._M_worst_error_q15;
    row._M_bound_q15 = choice._M_bound_q15;
    row._M_method = choice._M_method;
    row._M_flags = (choice._M_asymmetric != 0u ? _S_calib_flag_asymmetric : 0u)
                   | (choice._M_degenerate != 0u ? _S_calib_flag_degenerate : 0u);
    row._M_samples = samples > 0xFFFFu ? 0xFFFFu : static_cast<uint16_t> (samples);
    ++report._M_count;
    if (choice._M_worst_error_q15 > report._M_worst_error_q15)
        {
            report._M_worst_error_q15 = choice._M_worst_error_q15;
            report._M_worst_entry = static_cast<uint16_t> (report._M_count - 1u);
        }
    report._M_elements
        = report._M_elements > 0xFFFFFFFFu - samples
              ? 0xFFFFFFFFu
              : report._M_elements + samples;
    calib_count_saturate (report._M_batches);
    return Status::OK;
}

/** @brief Validate the whole report: legal grids, no half-written row. */
[[nodiscard]] inline Status
calib_report_finish (const CalibrationReport &report) noexcept
{
    if (report._M_entries == nullptr && report._M_capacity != 0u)
        return Status::BAD_LAYOUT;
    if (report._M_count > report._M_capacity)
        return Status::CORRUPT_STREAM;
    for (uint32_t index = 0; index < report._M_count; ++index)
        {
            const LayerCalibration &row = report._M_entries[index];
            if (row._M_scale_q15 <= 0 || row._M_multiplier_q15 <= 0)
                return Status::CORRUPT_STREAM;
            if (row._M_zero_point < -127 || row._M_zero_point > 127)
                return Status::CORRUPT_STREAM;
            if (row._M_method > static_cast<uint8_t> (CalibMethod::Kl))
                return Status::CORRUPT_STREAM;
        }
    return Status::OK;
}

/** @brief Rows a report of @p layers layers and @p channels channels needs. */
[[nodiscard]] inline constexpr uint32_t
calib_report_rows (const uint32_t layers, const uint32_t channels) noexcept
{
    return layers * channels;
}

/** @brief Bytes @p layers layers and @p channels channels of report rows cost. */
[[nodiscard]] inline constexpr uint32_t
calib_report_bytes (const uint32_t layers, const uint32_t channels) noexcept
{
    return calib_report_rows (layers, channels)
           * static_cast<uint32_t> (sizeof (LayerCalibration));
}

/**
 * @name Inference-format interop
 *
 * mllee/format.hpp reads a layer's per-row / per-group scales as int32
 * little-endian Q15 multipliers at `Layer::_M_scale_off`
 * (`_S_i32 (model + off + index * 4)`) and the per-tensor fallback from
 * `Layer::_M_requant_mult`.  These helpers emit exactly that, so a calibration
 * run drops into a model blob with no conversion step;
 * tests/test_mllee_training_quantize.cpp reads the emitted table back with the
 * inference header's own `_S_i32` accessor.
 * @{
 */

/** @brief Little-endian int32 store, the mirror of format.hpp's `_S_i32`. */
inline constexpr void
calib_put_i32 (uint8_t *const out, const int32_t value) noexcept
{
    const uint32_t bits = static_cast<uint32_t> (value);
    out[0] = static_cast<uint8_t> (bits & 0xFFu);
    out[1] = static_cast<uint8_t> ((bits >> 8) & 0xFFu);
    out[2] = static_cast<uint8_t> ((bits >> 16) & 0xFFu);
    out[3] = static_cast<uint8_t> ((bits >> 24) & 0xFFu);
}

/** @brief Bytes the scale table of @p rows entries costs. */
[[nodiscard]] inline constexpr uint32_t
calib_scale_table_bytes (const uint32_t rows) noexcept
{
    return rows * 4u;
}

/**
 * @brief Write report rows into an inference scale table, one int32 per row.
 *
 * @p out must be 4-byte aligned and hold `calib_scale_table_bytes (count)` bytes;
 * the caller keeps `count` within the `rows x groups` slots its layer descriptor
 * declares.
 */
[[nodiscard]] inline Status
calib_write_scale_table (const CalibrationReport &report, const uint32_t count,
                         uint8_t *const out) noexcept
{
    if (out == nullptr)
        return Status::BAD_LAYOUT;
    if (count == 0u || count > report._M_count)
        return Status::OUT_OF_RANGE;
    if (!simd_aligned (out))
        return Status::BAD_LAYOUT;
    for (uint32_t index = 0; index < count; ++index)
        {
            const LayerCalibration &row = report._M_entries[index];
            if (row._M_multiplier_q15 <= 0)
                return Status::CORRUPT_STREAM;
            calib_put_i32 (out + index * 4u, row._M_multiplier_q15);
        }
    return Status::OK;
}

/**
 * @brief Fold one report row into a layer descriptor and its scale table slot.
 *
 * `_M_requant_mult` always gets the row's per-tensor multiplier; when a table is
 * supplied, @p table_bytes (a multiple of 4, starting at @p table_off inside the
 * caller's model blob) is filled with the same Q15 multiplier per row slot, which
 * is what `_S_scale_multiplier` then reads.  The descriptor is not re-validated
 * here -- the caller runs the inference header's own validator afterwards, and
 * that is the interop gate.
 */
[[nodiscard]] inline Status
calib_apply_to_layer (const CalibrationReport &report, const uint32_t row_index,
                      Layer &layer, uint8_t *const scale_table,
                      const uint32_t table_off,
                      const uint32_t table_bytes) noexcept
{
    if (row_index >= report._M_count || row_index >= report._M_capacity)
        return Status::OUT_OF_RANGE;
    const LayerCalibration &row = report._M_entries[row_index];
    const CalibChoice choice{ row._M_scale_q15, row._M_multiplier_q15,
                              row._M_zero_point, row._M_bound_q15,
                              row._M_worst_error_q15, 0, row._M_method,
                              row._M_flags, 0u, 127u };
    const Status choice_status = choice.validate ();
    if (!status_ok (choice_status))
        return choice_status;
    if (scale_table != nullptr)
        {
            if (table_bytes == 0u || table_bytes % 4u != 0u)
                return Status::BAD_LAYOUT;
            if (!simd_aligned (scale_table))
                return Status::BAD_LAYOUT;
            if (table_off > 0xFFFFFFFFu - table_bytes)
                return Status::OUT_OF_RANGE;
            const uint32_t slots = table_bytes / 4u;
            for (uint32_t slot = 0; slot < slots; ++slot)
                calib_put_i32 (scale_table + slot * 4u, row._M_multiplier_q15);
            layer._M_scale_off = table_off;
        }
    layer._M_requant_mult = row._M_multiplier_q15;
    return Status::OK;
}

/**
 * @brief The Q15 requant multiplier of three calibrated grid steps.
 *
 * This is `s_weight * s_input / s_output` folded into one Q15 number exactly as
 * the runtime wants it, computed with the module's own rounding shift -- the
 * float-free counterpart of the host helper `zzs::mllee::requant_multiplier`.
 */
[[nodiscard]] inline constexpr int32_t
calib_requant_multiplier (const int32_t weight_scale_q15,
                          const int32_t input_scale_q15,
                          const int32_t output_scale_q15) noexcept
{
    if (weight_scale_q15 <= 0 || input_scale_q15 <= 0 || output_scale_q15 <= 0)
        return 0;
    const int64_t numerator
        = static_cast<int64_t> (weight_scale_q15) * input_scale_q15;
    /* ratio * 2^15 in one int64 division, then the rounding shift back: the Q15
       steps cancel the extra 2^15 the ratio would otherwise carry. */
    const int64_t scaled
        = (numerator << _S_grad_frac_bits) / (static_cast<int64_t> (output_scale_q15)
                                              * _S_q15_one);
    if (scaled > _S_calib_i32_max)
        return static_cast<int32_t> (_S_calib_i32_max);
    if (scaled < -_S_calib_i32_max)
        return -static_cast<int32_t> (_S_calib_i32_max);
    return _S_round_shift (scaled, 15u);
}

/**
 * @brief Emit a report as the (scale, multiplier, zero point) tables a fake-quant
 *        op reads.
 *
 * Plain arrays on purpose: this header stays independent of any other training
 * module's descriptors, so a calibrated model can be handed to the
 * quantization-aware path or to inference without either side owning the other.
 */
[[nodiscard]] inline Status
calib_emit_grid (const CalibrationReport &report, const uint32_t count,
                 int32_t *const scales, int32_t *const multipliers,
                 int8_t *const zero_points) noexcept
{
    if (report._M_entries == nullptr)
        return Status::BAD_LAYOUT;
    if (scales == nullptr || multipliers == nullptr)
        return Status::BAD_LAYOUT;
    if (count == 0u || count > report._M_count)
        return Status::OUT_OF_RANGE;
    if (!simd_aligned (scales) || !simd_aligned (multipliers))
        return Status::BAD_LAYOUT;
    for (uint32_t index = 0; index < count; ++index)
        {
            const LayerCalibration &row = report._M_entries[index];
            scales[index] = row._M_scale_q15;
            multipliers[index] = row._M_multiplier_q15;
            if (zero_points != nullptr)
                zero_points[index] = _S_i8_code (row._M_zero_point);
        }
    return Status::OK;
}

/** @} */

/**
 * @brief Observe-then-select for one batch of channel histograms: the shape a
 *        calibration tool calls per layer.
 *
 * Each histogram is read a bounded number of times (256 bins per candidate, at
 * most 96 candidates for Mse/Kl) and no buffer is copied.
 */
[[nodiscard]] inline Status
calib_run_batch (CalibrationReport &report, const uint32_t layer_index,
                 const CalibHistogram *const histograms, const uint32_t channels,
                 const CalibConfig &config,
                 uint32_t *const kl_scratch = nullptr) noexcept
{
    if (histograms == nullptr)
        return Status::BAD_LAYOUT;
    if (channels == 0u || channels > _S_max_elements)
        return Status::OUT_OF_RANGE;
    for (uint32_t channel = 0; channel < channels; ++channel)
        {
            CalibChoice choice;
            const Status status = calib_select (histograms[channel], config,
                                                choice, kl_scratch);
            if (!status_ok (status))
                return status;
            const Status added = calib_report_add (report, layer_index, channel,
                                                   choice,
                                                   histograms[channel]._M_total);
            if (!status_ok (added))
                return added;
        }
    return Status::OK;
}

/**
 * @brief One-shot observe + select over a lane buffer: the convenience entry.
 *
 * Carves nothing: the histogram and the choice live where the caller put them.
 */
[[nodiscard]] inline Status
calib_observe_and_select (CalibHistogram &histogram, const int16_t *const values,
                          const uint32_t count, const CalibConfig &config,
                          CalibChoice &out,
                          uint32_t *const kl_scratch = nullptr) noexcept
{
    calib_reset (histogram);
    const Status observed = calib_observe_lanes (histogram, values, count);
    if (!status_ok (observed))
        return observed;
    return calib_select (histogram, config, out, kl_scratch);
}
} // namespace zzs::mllee::train

#endif // ZZS_MLLEE_TRAIN_CALIBRATION_HPP
