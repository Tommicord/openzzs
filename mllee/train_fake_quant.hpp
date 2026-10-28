/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#pragma once

/* Include guard beyond #pragma once: this header is a quantization *contract*
 * and must never be silently skipped or duplicated by a sibling module. */
#ifndef ZZS_MLLEE_TRAIN_FAKE_QUANT_HPP
#define ZZS_MLLEE_TRAIN_FAKE_QUANT_HPP

#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_simd.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Quantization-aware training (QAT): fake quantization plus the exact
 *        straight-through estimator.
 *
 * The value model is the one the inference core already pays for: an activation
 * or a weight is an int8 code in [-127, 127] whose real value is
 * `code * scale_q15 / 32768`, and every product is requantized with a single Q15
 * multiplier (mllee/README.md, "Quantization").  This header adds no third
 * numeric convention:
 *
 *   - `_M_scale_q15`      : one code step in Q15 units.  258 is the identity grid
 *                           (code 127 = 32766 = 1.0 - eps), i.e. `code_to_q15`;
 *   - `_M_multiplier_q15` : the reciprocal `round(2^15 / scale_q15)`, so a Q15
 *                           value becomes a code through `_S_round_shift`, and it
 *                           is the same kind of number as the inference header's
 *                           `_M_requant_mult`;
 *   - `_M_zero_point`     : an int8 code offset, used only by the asymmetric form.
 *
 * Forward: `code = clamp(round(v * multiplier / 2^15) + zero_point, qmin, qmax)`,
 * and the tensor handed to the next op is the *dequantized* value
 * `(code - zero_point) * scale_q15` -- quantize-dequantize, so training sees
 * exactly the error inference will make.
 *
 * Backward (straight-through): the gradient passes unchanged except that a lane
 * whose forward value fell outside the clamp range receives **zero** gradient.
 * That predicate is computed exactly, not approximately: a lane is clipped iff
 * `round(v * m / 2^15) + zp` lies outside `[qmin, qmax]`, and because
 * `_S_round_shift` is a floor of `(v*m + 2^14) / 2^15`, that is equivalent to
 * `v` lying outside a *closed integer interval* `[lo, hi]` that `ClipWindow`
 * derives once per channel with int64 arithmetic (`fq_ceil_div` /
 * `fq_floor_div`).  No epsilon, no float, no off-by-one at the boundary --
 * `tests/test_mllee_training_quantize.cpp` pins both edge values.
 *
 * Channel structure: with an axis-wise (per-channel) grid the elements of one
 * channel form contiguous runs -- a run of `stride(axis)` elements repeating
 * every `dim(axis) * stride(axis)` -- so the multiplier, the zero point and the
 * two clamp bounds are hoisted out of the inner loop, both streams stay stride-1
 * and 4-byte aligned, and every buffer is touched exactly once.  Nothing looks
 * anything up in a pool inside an element loop.
 *
 * Device-path rules honoured here: no heap, no exceptions, no RTTI, no
 * recursion, no writable statics, no float, errors as `Status` values, gradient
 * accumulation through `Trainable::add_grad` (the tape contract).
 *
 * ### Pool cost (all caller-owned; nothing here allocates)
 *
 * | Object | Bytes |
 * |---|---|
 * | `FakeQuantParams` (POD descriptor) | 24 on a 32-bit target, 32 on a host |
 * | `FakeQuantLayout` / `ClipWindow` / `FakeQuantNodeState` | 20 / 20 / 24 |
 * | per-channel scale + multiplier tables | 4 · C + 4 · C |
 * | per-channel zero points (asymmetric) | 1 · C |
 * | clip windows computed on demand | 20 · C |
 * | `MinMaxAccumulator` observed per channel | 20 · C |
 * | STE accumulate scratch (`fake_quant_backward_ste_accumulate`) | 2 · N |
 * | **recorded fake-quant node** (`record_fake_quant`) | 2 · N + 24 · C + 24 |
 *
 * With C = 32 channels and N = 256 elements a recorded node costs 24 · 32 + 512
 * + 24 = 1328 pool bytes per pass; `Tape::stats()._M_saved_bytes` reports it, and
 * the trainer holds one node per quantized tensor.
 */
namespace zzs::mllee::train
{
/** @brief The two affine forms the inference header supports. */
enum class FakeQuantMode : uint8_t
{
    /** @brief zero point 0, symmetric grid, `qmin == -qmax`. */
    Symmetric = 0,
    /** @brief explicit zero point, `qmin = -127`, `qmax = 127` by default. */
    Asymmetric = 1,
};

/** @brief Axis selector for "one grid for the whole tensor". */
inline constexpr uint8_t _S_fq_per_tensor = 0xFFu;

/** @brief Widest per-channel table a single fake-quant op addresses. */
inline constexpr uint32_t _S_fq_max_channels = 256u;

/** @brief int32 span used to saturate an exact clip bound. */
inline constexpr int32_t _S_fq_int32_min = -2147483647 - 1;
inline constexpr int32_t _S_fq_int32_max = 2147483647;

/**
 * @brief One fake-quantizer's parameters: per-tensor or per-channel (axis-wise).
 *
 * A POD descriptor -- no vtable, no ownership -- so a tape node can copy it and
 * a backward function can read it through `GradContext::saved_as`.
 * `_M_axis == _S_fq_per_tensor` means one table entry; otherwise the tensor is
 * read as `dim(_M_axis)` channel runs and every table has `_M_channels` entries.
 */
struct FakeQuantParams
{
    /** @brief Code -> Q15 step, one entry per channel.  Must be > 0. */
    const int32_t *_M_scale_q15 = nullptr;
    /** @brief Q15 -> code multiplier, one entry per channel.  Must be > 0. */
    const int32_t *_M_multiplier_q15 = nullptr;
    /** @brief Zero points; null for the symmetric form. */
    const int8_t *_M_zero_point = nullptr;
    int8_t _M_qmin = -127;
    int8_t _M_qmax = 127;
    uint8_t _M_mode = static_cast<uint8_t> (FakeQuantMode::Symmetric);
    uint8_t _M_axis = _S_fq_per_tensor;
    uint16_t _M_channels = 1u;

    [[nodiscard]] static constexpr FakeQuantParams
    symmetric (const int32_t *const scale_q15,
               const int32_t *const multiplier_q15) noexcept
    {
        FakeQuantParams params;
        params._M_scale_q15 = scale_q15;
        params._M_multiplier_q15 = multiplier_q15;
        return params;
    }

    [[nodiscard]] static constexpr FakeQuantParams
    asymmetric (const int32_t *const scale_q15,
                const int32_t *const multiplier_q15,
                const int8_t *const zero_point) noexcept
    {
        FakeQuantParams params;
        params._M_scale_q15 = scale_q15;
        params._M_multiplier_q15 = multiplier_q15;
        params._M_zero_point = zero_point;
        params._M_mode = static_cast<uint8_t> (FakeQuantMode::Asymmetric);
        return params;
    }

    /** @brief Turn this descriptor into an axis-wise (per-channel) one. */
    [[nodiscard]] constexpr FakeQuantParams
    on_axis (const uint8_t axis, const uint16_t channels) const noexcept
    {
        FakeQuantParams view = *this;
        view._M_axis = axis;
        view._M_channels = channels;
        return view;
    }

    [[nodiscard]] constexpr bool
    per_channel () const noexcept
    {
        return this->_M_axis != _S_fq_per_tensor;
    }

    [[nodiscard]] constexpr uint32_t
    table_index (const uint32_t channel) const noexcept
    {
        return this->per_channel () ? channel : 0u;
    }

    [[nodiscard]] constexpr int32_t
    scale_at (const uint32_t channel) const noexcept
    {
        return this->_M_scale_q15[this->table_index (channel)];
    }

    [[nodiscard]] constexpr int32_t
    multiplier_at (const uint32_t channel) const noexcept
    {
        return this->_M_multiplier_q15[this->table_index (channel)];
    }

    [[nodiscard]] constexpr int32_t
    zero_point_at (const uint32_t channel) const noexcept
    {
        if (this->_M_mode != static_cast<uint8_t> (FakeQuantMode::Asymmetric)
            || this->_M_zero_point == nullptr)
            return 0;
        return this->_M_zero_point[this->table_index (channel)];
    }

    /** @brief Validate tables, channel count and the clamp range. */
    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (this->_M_scale_q15 == nullptr || this->_M_multiplier_q15 == nullptr)
            return Status::BAD_LAYOUT;
        if (this->_M_channels == 0u || this->_M_channels > _S_fq_max_channels)
            return Status::OUT_OF_RANGE;
        if (this->_M_qmin > this->_M_qmax)
            return Status::BAD_LAYOUT;
        if (this->_M_mode > static_cast<uint8_t> (FakeQuantMode::Asymmetric))
            return Status::BAD_LAYOUT;
        if (this->_M_axis != _S_fq_per_tensor && this->_M_axis >= _S_max_rank)
            return Status::BAD_LAYOUT;
        /* int8 cannot exceed 127, so the only out-of-range code here is -128,
           which the module never emits (see _S_i8_code). */
        if (this->_M_qmin < -127)
            return Status::OUT_OF_RANGE;
        if (this->_M_mode == static_cast<uint8_t> (FakeQuantMode::Symmetric)
            && (this->_M_qmax <= 0 || this->_M_qmin != -this->_M_qmax))
            return Status::BAD_LAYOUT;
        for (uint32_t channel = 0; channel < this->_M_channels; ++channel)
            {
                if (this->_M_scale_q15[channel] <= 0
                    || this->_M_multiplier_q15[channel] <= 0)
                    return Status::OUT_OF_RANGE;
                /* A signed byte tops out at 127, so the only reachable violation
                   is the -128 code the module never emits. */
                if (this->_M_mode == static_cast<uint8_t> (FakeQuantMode::Asymmetric)
                    && this->_M_zero_point != nullptr
                    && static_cast<int> (this->_M_zero_point[channel]) < -127)
                    return Status::OUT_OF_RANGE;
            }
        return Status::OK;
    }
};

static_assert (sizeof (FakeQuantParams) <= (sizeof (void *) == 4u ? 24u : 40u),
               "fake-quant descriptor must stay a copyable leaf");

/**
 * @brief The contiguous-run layout an axis-wise quantization walks.
 *
 * For a row-major tensor, fixing one coordinate of axis `a` selects a contiguous
 * run of `_M_run` elements repeating every `_M_group` elements, `_M_outer` times
 * over the leading axes; the total element count is `_M_count`.  A per-tensor
 * grid is the degenerate `run == group == count, outer == 1` case, so every
 * kernel uses one base formula
 * `base = block * _M_group + channel * _M_run` and needs no per-element branch.
 */
struct FakeQuantLayout
{
    uint32_t _M_run = 0u;
    uint32_t _M_group = 0u;
    uint32_t _M_outer = 1u;
    uint32_t _M_count = 0u;
    uint16_t _M_channels = 1u;
    uint8_t _M_axis = _S_fq_per_tensor;
    uint8_t _M_valid = 0u;

    [[nodiscard]] constexpr bool
    per_channel () const noexcept
    {
        return this->_M_axis != _S_fq_per_tensor;
    }

    [[nodiscard]] constexpr uint32_t
    base (const uint32_t channel, const uint32_t block) const noexcept
    {
        return block * this->_M_group + channel * this->_M_run;
    }
};

static_assert (sizeof (FakeQuantLayout) == 20u,
               "fake-quant layout must stay 20 bytes on both targets");

/** @brief Derive the run layout of @p params from a validated shape. */
[[nodiscard]] inline constexpr Status
fake_quant_layout (const Shape &shape, const FakeQuantParams &params,
                   FakeQuantLayout &out) noexcept
{
    const Status shape_status = shape.validate ();
    if (!status_ok (shape_status))
        return shape_status;
    FakeQuantParams checked = params;
    const Status params_status = checked.validate ();
    if (!status_ok (params_status))
        return params_status;

    FakeQuantLayout layout;
    layout._M_axis = params._M_axis;
    layout._M_channels = params._M_channels;
    layout._M_count = shape.elements ();
    if (!params.per_channel ())
        {
            layout._M_run = layout._M_count;
            layout._M_group = layout._M_count;
            layout._M_outer = 1u;
        }
    else
        {
            if (params._M_axis >= shape._M_rank)
                return Status::BAD_LAYOUT;
            if (shape._M_dims[params._M_axis] != params._M_channels)
                return Status::BAD_LAYOUT;
            layout._M_run = shape.stride (params._M_axis);
            layout._M_group = layout._M_run * params._M_channels;
            if (layout._M_group == 0u || layout._M_count % layout._M_group != 0u)
                return Status::BAD_LAYOUT;
            layout._M_outer = layout._M_count / layout._M_group;
        }
    if (layout._M_run == 0u)
        return Status::BAD_LAYOUT;
    layout._M_valid = 1u;
    out = layout;
    return Status::OK;
}

/** @brief Layout of a flat element count with a per-channel split (no Shape). */
[[nodiscard]] inline constexpr Status
fake_quant_layout_flat (const uint32_t count, const uint32_t channels,
                        FakeQuantLayout &out) noexcept
{
    if (count == 0u || channels == 0u || channels > _S_fq_max_channels
        || count % channels != 0u)
        return Status::BAD_LAYOUT;
    FakeQuantLayout layout;
    layout._M_run = count / channels;
    layout._M_group = count;
    layout._M_outer = 1u;
    layout._M_count = count;
    layout._M_channels = static_cast<uint16_t> (channels);
    /* A flat split is channel-major: channel c owns the contiguous run
       [c * run, (c + 1) * run), which the shared base formula reproduces with
       one outer block. */
    layout._M_axis = 0u;
    layout._M_valid = 1u;
    out = layout;
    return Status::OK;
}

/** @brief Smallest positive scale whose grid reaches @p abs_max_q15 unclipped. */
[[nodiscard]] inline constexpr int32_t
fq_symmetric_scale (const int32_t abs_max_q15) noexcept
{
    if (abs_max_q15 <= 0)
        return _S_code_to_q15_scale;
    const int64_t widened = (static_cast<int64_t> (abs_max_q15) + 126) / 127;
    if (widened > _S_q15_max)
        return _S_q15_max;
    return widened < 1 ? 1 : static_cast<int32_t> (widened);
}

/** @brief The reciprocal Q15 multiplier of a code scale: round(2^15 / scale). */
[[nodiscard]] inline constexpr int32_t
fq_multiplier_from_scale (const int32_t scale_q15) noexcept
{
    if (scale_q15 <= 0)
        return 0;
    const int64_t numerator = static_cast<int64_t> (_S_q15_one) * _S_q15_one;
    const int64_t rounded = (numerator + scale_q15 / 2) / scale_q15;
    return static_cast<int32_t> (rounded > _S_fq_int32_max ? _S_fq_int32_max
                                                           : rounded);
}

/** @brief Floor division by a positive divisor (never implementation-defined). */
[[nodiscard]] inline constexpr int64_t
fq_floor_div (const int64_t numerator, const int64_t positive_denominator) noexcept
{
    const int64_t quotient = numerator / positive_denominator;
    if (numerator % positive_denominator != 0 && numerator < 0)
        return quotient - 1;
    return quotient;
}

/** @brief Ceil division by a positive divisor. */
[[nodiscard]] inline constexpr int64_t
fq_ceil_div (const int64_t numerator, const int64_t positive_denominator) noexcept
{
    const int64_t quotient = numerator / positive_denominator;
    if (numerator % positive_denominator != 0 && numerator > 0)
        return quotient + 1;
    return quotient;
}

/** @brief Saturate an int64 to the int32 span, so a bound comparison is exact. */
[[nodiscard]] inline constexpr int32_t
fq_narrow_i32_sat (const int64_t value) noexcept
{
    if (value > _S_fq_int32_max)
        return _S_fq_int32_max;
    if (value < _S_fq_int32_min)
        return _S_fq_int32_min;
    return static_cast<int32_t> (value);
}

/**
 * @brief The exact "not clipped" interval of a quantizer, in Q15 value units.
 *
 * `round_half_up (v * m / 2^15) + zp` lands inside `[qmin, qmax]` iff
 * `_M_low <= v <= _M_high`, where both bounds come straight out of the rounding
 * rule `_S_round_shift` implements (`floor((v*m + 2^14) / 2^15)`):
 *
 *   low  = ceil ((qmin - zp) * 2^15 - 2^14) / m
 *   high = floor ((qmax - zp + 1) * 2^15 - 2^14 - 1) / m
 *
 * which is why the STE can decide the clamp from the *value* alone and needs no
 * per-element clip mask.
 */
struct ClipWindow
{
    int32_t _M_low = _S_fq_int32_min;
    int32_t _M_high = _S_fq_int32_max;
    int32_t _M_code_low = 0;
    int32_t _M_code_high = 0;
    int8_t _M_qmin = -127;
    int8_t _M_qmax = 127;
    uint8_t _M_valid = 0u;
    uint8_t _M_pad = 0u;
};

static_assert (sizeof (ClipWindow) == 20u,
               "clip window must stay 20 bytes on both targets");

/** @brief Build the exact clip window of one channel. */
[[nodiscard]] inline constexpr ClipWindow
clip_window_of (const int32_t multiplier_q15, const int32_t zero_point,
                const int8_t qmin, const int8_t qmax) noexcept
{
    ClipWindow window;
    window._M_qmin = qmin;
    window._M_qmax = qmax;
    if (multiplier_q15 <= 0)
        return window;
    const int64_t half = static_cast<int64_t> (_S_q15_one) >> 1;
    const int64_t low_level = static_cast<int64_t> (qmin) - zero_point;
    const int64_t high_level = static_cast<int64_t> (qmax) - zero_point;
    const int64_t low = fq_ceil_div (low_level * _S_q15_one - half, multiplier_q15);
    const int64_t high
        = fq_floor_div ((high_level + 1) * _S_q15_one - half - 1, multiplier_q15);
    window._M_low = fq_narrow_i32_sat (low);
    window._M_high = fq_narrow_i32_sat (high);
    window._M_code_low = static_cast<int32_t> (low_level);
    window._M_code_high = static_cast<int32_t> (high_level);
    window._M_valid = 1u;
    return window;
}

/** @brief Clip windows for every channel into caller-owned storage. */
[[nodiscard]] inline constexpr Status
fake_quant_clip_windows (const FakeQuantParams &params, ClipWindow *const out,
                         const uint32_t capacity) noexcept
{
    const Status status = params.validate ();
    if (!status_ok (status))
        return status;
    if (out == nullptr)
        return Status::BAD_LAYOUT;
    if (capacity < params._M_channels)
        return Status::ARENA_OVERFLOW;
    for (uint32_t channel = 0; channel < params._M_channels; ++channel)
        out[channel] = clip_window_of (params.multiplier_at (channel),
                                       params.zero_point_at (channel),
                                       params._M_qmin, params._M_qmax);
    return Status::OK;
}

/** @brief The code a Q15 value maps to *before* the clamp (may exceed int8). */
[[nodiscard]] inline constexpr int32_t
fq_pre_clamp_code (const int32_t value_q15, const int32_t multiplier_q15,
                   const int32_t zero_point) noexcept
{
    return _S_round_shift (static_cast<int64_t> (value_q15) * multiplier_q15,
                           _S_grad_frac_bits)
           + zero_point;
}

/** @brief The dequantized Q15 value of a code on a channel's grid. */
[[nodiscard]] inline constexpr int32_t
fq_dequantize_q15 (const int32_t code, const int32_t scale_q15,
                   const int32_t zero_point) noexcept
{
    return q15_clamp ((code - zero_point) * scale_q15);
}

/**
 * @brief One quantize-dequantize step on a single value.
 * @param clipped true when the forward code hit the clamp range -- exactly the
 *        condition the straight-through estimator zeroes on.
 */
[[nodiscard]] inline constexpr int32_t
fq_dequantized_value (const int32_t value_q15, const int32_t scale_q15,
                      const int32_t multiplier_q15, const int32_t zero_point,
                      const int8_t qmin, const int8_t qmax, bool &clipped) noexcept
{
    const int32_t raw = fq_pre_clamp_code (value_q15, multiplier_q15, zero_point);
    clipped = raw < static_cast<int32_t> (qmin) || raw > static_cast<int32_t> (qmax);
    return fq_dequantize_q15 (_S_clamp_i8 (raw), scale_q15, zero_point);
}

/**
 * @brief Forward: quantize a Q15 lane stream to int8 codes and dequantize it
 *        again into the lane stream the next op consumes.
 *
 * One pass per channel run; either output may be null.  The layout must come
 * from `fake_quant_layout` / `fake_quant_layout_flat` for the same element count.
 */
[[nodiscard]] inline Status
fake_quant_forward (const int16_t *const values_q15, int8_t *const codes,
                    int16_t *const dequantized, const FakeQuantParams &params,
                    const FakeQuantLayout &layout, uint32_t &clipped_lanes) noexcept
{
    const Status status = params.validate ();
    if (!status_ok (status))
        return status;
    if (values_q15 == nullptr || (codes == nullptr && dequantized == nullptr))
        return Status::BAD_LAYOUT;
    if (layout._M_valid == 0u || layout._M_channels != params._M_channels)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (values_q15) || (codes != nullptr && !simd_aligned (codes))
        || (dequantized != nullptr && !simd_aligned (dequantized)))
        return Status::BAD_LAYOUT;

    for (uint32_t channel = 0; channel < params._M_channels; ++channel)
        {
            /* Hoisted: the grid and both clamp bounds for this channel. */
            const int32_t multiplier = params.multiplier_at (channel);
            const int32_t scale = params.scale_at (channel);
            const int32_t zero_point = params.zero_point_at (channel);
            const int32_t low_level = static_cast<int32_t> (params._M_qmin) - zero_point;
            const int32_t high_level = static_cast<int32_t> (params._M_qmax) - zero_point;
            for (uint32_t block = 0; block < layout._M_outer; ++block)
                {
                    const uint32_t base = layout.base (channel, block);
                    for (uint32_t i = 0; i < layout._M_run; ++i)
                        {
                            const uint32_t index = base + i;
                            const int32_t raw = fq_pre_clamp_code (
                                values_q15[index], multiplier, zero_point);
                            if (raw < low_level || raw > high_level)
                                ++clipped_lanes;
                            const int8_t code = _S_i8_code (raw);
                            if (codes != nullptr)
                                codes[index] = code;
                            if (dequantized != nullptr)
                                dequantized[index] = q15_to_i16_sat (
                                    fq_dequantize_q15 (code, scale, zero_point));
                        }
                }
        }
    return Status::OK;
}

/**
 * @brief Straight-through backward, exact at the clamp edges.
 *
 * `grad_in = (low <= v <= high) ? grad_out : 0` with the closed integer bounds
 * of `ClipWindow`, i.e. bit-for-bit the forward clip test: a lane the forward
 * pass could not represent contributes nothing, while a lane that merely rounded
 * onto the extreme code keeps its full gradient.
 *
 * @return the number of lanes zeroed because they were clipped.
 */
[[nodiscard]] inline uint32_t
fake_quant_backward_ste (const int16_t *const values_q15, const int16_t *const grad_out,
                         int16_t *const grad_in, const ClipWindow *const windows,
                         const FakeQuantParams &params,
                         const FakeQuantLayout &layout) noexcept
{
    if (values_q15 == nullptr || grad_out == nullptr || grad_in == nullptr
        || windows == nullptr || layout._M_valid == 0u)
        return 0u;
    if (layout._M_channels != params._M_channels)
        return 0u;
    uint32_t zeroed = 0u;
    for (uint32_t channel = 0; channel < layout._M_channels; ++channel)
        {
            const int32_t low = windows[channel]._M_low;
            const int32_t high = windows[channel]._M_high;
            for (uint32_t block = 0; block < layout._M_outer; ++block)
                {
                    const uint32_t base = layout.base (channel, block);
                    for (uint32_t i = 0; i < layout._M_run; ++i)
                        {
                            const uint32_t index = base + i;
                            const int32_t value = values_q15[index];
                            const bool keep = value >= low && value <= high;
                            grad_in[index] = keep ? grad_out[index] : 0;
                            if (!keep)
                                ++zeroed;
                        }
                }
        }
    return zeroed;
}

/**
 * @brief Accumulating form of the STE: the gated copy goes through the SIMD lane
 *        add of `train_simd::add_grad` instead of overwriting.
 *
 * Costs 2 * N bytes of caller-owned scratch, which buys a vectorized accumulate
 * and keeps the gradient saturating exactly like every other backward path.
 *
 * @return the number of lanes zeroed because they were clipped.
 */
[[nodiscard]] inline Status
fake_quant_backward_ste_accumulate (const int16_t *const values_q15,
                                    const int16_t *const grad_out,
                                    int16_t *const grad_in, int16_t *const scratch,
                                    const uint32_t scratch_count,
                                    const ClipWindow *const windows,
                                    const FakeQuantParams &params,
                                    const FakeQuantLayout &layout,
                                    uint32_t &zeroed, uint32_t &saturated) noexcept
{
    if (values_q15 == nullptr || grad_out == nullptr || grad_in == nullptr
        || scratch == nullptr || windows == nullptr || layout._M_valid == 0u)
        return Status::BAD_LAYOUT;
    if (scratch_count < layout._M_count || layout._M_channels != params._M_channels)
        return Status::ARENA_OVERFLOW;
    if (!simd_aligned (values_q15) || !simd_aligned (grad_out)
        || !simd_aligned (grad_in) || !simd_aligned (scratch))
        return Status::BAD_LAYOUT;
    zeroed = 0u;
    saturated = 0u;
    for (uint32_t channel = 0; channel < layout._M_channels; ++channel)
        {
            const int32_t low = windows[channel]._M_low;
            const int32_t high = windows[channel]._M_high;
            for (uint32_t block = 0; block < layout._M_outer; ++block)
                {
                    const uint32_t base = layout.base (channel, block);
                    for (uint32_t i = 0; i < layout._M_run; ++i)
                        {
                            const uint32_t index = base + i;
                            const bool keep
                                = values_q15[index] >= low && values_q15[index] <= high;
                            scratch[base + i] = keep ? grad_out[index] : 0;
                            if (!keep)
                                ++zeroed;
                        }
                }
        }
    saturated = add_lanes (grad_in, scratch, layout._M_count);
    return Status::OK;
}

/**
 * @brief Observer-style min / max tracker with saturating accumulators.
 *
 * `_M_min_q15` / `_M_max_q15` / `_M_abs_max_q15` seed on the first observation
 * (so an all-negative stream is not silently clipped to 0); `_M_sum_q15` is the
 * saturating L1 accumulator every narrowing step reads, widened through int64 so
 * a long observation cannot wrap before it clamps.
 */
struct MinMaxAccumulator
{
    int32_t _M_min_q15 = 0;
    int32_t _M_max_q15 = 0;
    int32_t _M_abs_max_q15 = 0;
    int32_t _M_sum_q15 = 0;
    uint16_t _M_count = 0u;
    uint8_t _M_seeded = 0u;
    uint8_t _M_pad = 0u;
};

static_assert (sizeof (MinMaxAccumulator) == 20u,
               "min/max accumulator must stay 20 bytes on both targets");

/** @brief Reset an accumulator array (one entry per channel). */
inline constexpr void
min_max_reset (MinMaxAccumulator *const accumulators, const uint32_t count) noexcept
{
    if (accumulators == nullptr)
        return;
    for (uint32_t i = 0; i < count; ++i)
        accumulators[i] = MinMaxAccumulator{};
}

/** @brief Saturating accumulate of one int64 total into a Q15 field. */
constexpr void
accumulate_saturating (int32_t &field, const int64_t delta) noexcept
{
    field = clamp_sum_q15 (static_cast<int64_t> (field) + delta);
}

/** @brief Observe one Q15 lane stream into a channel's accumulator. */
inline void
observe_lanes (MinMaxAccumulator &accumulator, const int16_t *const values_q15,
               const uint32_t count) noexcept
{
    if (values_q15 == nullptr || count == 0u)
        return;
    int32_t low = values_q15[0];
    int32_t high = values_q15[0];
    int64_t abs_total = 0;
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t value = values_q15[i];
            if (value < low)
                low = value;
            if (value > high)
                high = value;
            abs_total += value < 0 ? -static_cast<int64_t> (value) : value;
        }
    if (accumulator._M_seeded == 0u)
        {
            accumulator._M_min_q15 = low;
            accumulator._M_max_q15 = high;
            accumulator._M_seeded = 1u;
        }
    else
        {
            if (low < accumulator._M_min_q15)
                accumulator._M_min_q15 = low;
            if (high > accumulator._M_max_q15)
                accumulator._M_max_q15 = high;
        }
    /* The extreme of a batch is always one of the two ends of its range, so the
       saturating L1 accumulator above is the only second pass this needs. */
    const int32_t batch_abs = high > -low ? high : -low;
    if (batch_abs > accumulator._M_abs_max_q15)
        accumulator._M_abs_max_q15 = batch_abs;
    accumulate_saturating (accumulator._M_sum_q15, abs_total);
    accumulator._M_count = static_cast<uint16_t> (
        accumulator._M_count > 0xFFFFu - count ? 0xFFFFu : accumulator._M_count + count);
}

/** @brief Observe an int8 code stream (identity grid, `code_to_q15`). */
inline void
observe_codes (MinMaxAccumulator &accumulator, const int8_t *const codes,
               const uint32_t count) noexcept
{
    if (codes == nullptr || count == 0u)
        return;
    int32_t low = code_to_q15 (codes[0]);
    int32_t high = code_to_q15 (codes[0]);
    int64_t abs_total = 0;
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t value = code_to_q15 (codes[i]);
            if (value < low)
                low = value;
            if (value > high)
                high = value;
            abs_total += value < 0 ? -static_cast<int64_t> (value) : value;
        }
    if (accumulator._M_seeded == 0u)
        {
            accumulator._M_min_q15 = low;
            accumulator._M_max_q15 = high;
            accumulator._M_seeded = 1u;
        }
    else
        {
            if (low < accumulator._M_min_q15)
                accumulator._M_min_q15 = low;
            if (high > accumulator._M_max_q15)
                accumulator._M_max_q15 = high;
        }
    const int32_t batch_abs = high > -low ? high : -low;
    if (batch_abs > accumulator._M_abs_max_q15)
        accumulator._M_abs_max_q15 = batch_abs;
    accumulate_saturating (accumulator._M_sum_q15, abs_total);
    accumulator._M_count = static_cast<uint16_t> (
        accumulator._M_count > 0xFFFFu - count ? 0xFFFFu : accumulator._M_count + count);
}

/** @brief Observe a whole tensor channel by channel through the run layout. */
[[nodiscard]] inline Status
observe_lanes_axiswise (MinMaxAccumulator *const accumulators,
                        const int16_t *const values_q15,
                        const FakeQuantLayout &layout) noexcept
{
    if (accumulators == nullptr || values_q15 == nullptr || layout._M_valid == 0u)
        return Status::BAD_LAYOUT;
    if (layout._M_channels > _S_fq_max_channels)
        return Status::OUT_OF_RANGE;
    for (uint32_t channel = 0; channel < layout._M_channels; ++channel)
        for (uint32_t block = 0; block < layout._M_outer; ++block)
            observe_lanes (accumulators[channel],
                           values_q15 + layout.base (channel, block), layout._M_run);
    return Status::OK;
}

/** @brief Observe a whole tensor of codes channel by channel. */
[[nodiscard]] inline Status
observe_codes_axiswise (MinMaxAccumulator *const accumulators, const int8_t *const codes,
                        const FakeQuantLayout &layout) noexcept
{
    if (accumulators == nullptr || codes == nullptr || layout._M_valid == 0u)
        return Status::BAD_LAYOUT;
    if (layout._M_channels > _S_fq_max_channels)
        return Status::OUT_OF_RANGE;
    for (uint32_t channel = 0; channel < layout._M_channels; ++channel)
        for (uint32_t block = 0; block < layout._M_outer; ++block)
            observe_codes (accumulators[channel],
                           codes + layout.base (channel, block), layout._M_run);
    return Status::OK;
}

/** @brief Symmetric scale that covers an observed range (nothing clips). */
[[nodiscard]] inline constexpr int32_t
scale_from_observed (const MinMaxAccumulator &observed) noexcept
{
    return fq_symmetric_scale (observed._M_abs_max_q15);
}

/**
 * @brief Asymmetric scale / zero point for an observed range.
 *
 * Exactly the README affine rule with `qmin = -127`, `qmax = 127` (254 steps):
 * `scale = (max - min) / 254` rounded up so the extremes stay representable, and
 * `zero_point = clamp(-127 - round(min / scale), -127, 127)`.
 */
[[nodiscard]] inline constexpr Status
scale_zero_from_observed (const MinMaxAccumulator &observed, int32_t &scale_q15,
                          int8_t &zero_point) noexcept
{
    const int64_t span = static_cast<int64_t> (observed._M_max_q15)
                         - static_cast<int64_t> (observed._M_min_q15);
    if (span <= 0)
        {
            scale_q15 = _S_code_to_q15_scale;
            zero_point = 0;
            return Status::OK;
        }
    const int64_t scaled = fq_ceil_div (span, 254);
    if (scaled < 1 || scaled > _S_q15_max)
        return Status::OUT_OF_RANGE;
    scale_q15 = static_cast<int32_t> (scaled);
    const int32_t multiplier = fq_multiplier_from_scale (scale_q15);
    const int32_t raw = -127 - _S_round_shift (
        static_cast<int64_t> (observed._M_min_q15) * multiplier, _S_grad_frac_bits);
    zero_point = static_cast<int8_t> (raw < -127 ? -127 : (raw > 127 ? 127 : raw));
    return Status::OK;
}

/**
 * @brief Clipping-range narrowing: shrink the quantizer's grid to the observed range.
 *
 * A clamp range far wider than the data wastes codes *and* permanently zeroes the
 * gradient of every lane outside it, because the STE only passes where the
 * forward pass could represent the value.  `narrow_clip_range` recomputes the
 * grid so the extreme codes line up with the observed range, optionally keeping
 * only a percentile fraction of it:
 *
 *   - the returned scale is never larger than the input scale (narrowing only
 *     tightens the grid, it can never silently re-expand it);
 *   - for the symmetric form the returned clamp codes shrink to the code the
 *     observed extreme actually reaches, so `[-127, 127]` becomes e.g.
 *     `[-96, 96]` for a distribution that never fills the grid;
 *   - the asymmetric form keeps the full code range and narrows the covered
 *     value span instead, which is what its zero point is for.
 */
[[nodiscard]] inline Status
narrow_clip_range (const FakeQuantParams &params,
                   const MinMaxAccumulator *const observed,
                   const uint32_t observed_capacity, const int32_t keep_fraction_bp,
                   int32_t *const out_scale_q15, int32_t *const out_multiplier_q15,
                   int8_t *const out_qmin, int8_t *const out_qmax) noexcept
{
    const Status status = params.validate ();
    if (!status_ok (status))
        return status;
    if (observed == nullptr || out_scale_q15 == nullptr || out_multiplier_q15 == nullptr
        || out_qmin == nullptr || out_qmax == nullptr)
        return Status::BAD_LAYOUT;
    if (observed_capacity < params._M_channels)
        return Status::ARENA_OVERFLOW;
    if (keep_fraction_bp < 1 || keep_fraction_bp > 10000)
        return Status::OUT_OF_RANGE;
    const bool symmetric = params._M_mode == static_cast<uint8_t> (FakeQuantMode::Symmetric);
    for (uint32_t channel = 0; channel < params._M_channels; ++channel)
        {
            const MinMaxAccumulator &stats = observed[channel];
            const int32_t current_scale = params.scale_at (channel);
            int32_t wanted = current_scale;
            if (symmetric)
                {
                    const int64_t bounded = fq_floor_div (
                        static_cast<int64_t> (stats._M_abs_max_q15) * keep_fraction_bp,
                        10000);
                    wanted = fq_symmetric_scale (fq_narrow_i32_sat (bounded));
                }
            else
                {
                    const int64_t span = static_cast<int64_t> (stats._M_max_q15)
                                         - static_cast<int64_t> (stats._M_min_q15);
                    if (span > 0)
                        wanted = static_cast<int32_t> (fq_ceil_div (
                            span * keep_fraction_bp, 254LL * 10000));
                }
            if (wanted < 1)
                wanted = 1;
            const int32_t scale = wanted > current_scale ? current_scale : wanted;
            out_scale_q15[channel] = scale;
            out_multiplier_q15[channel] = fq_multiplier_from_scale (scale);
            if (out_multiplier_q15[channel] <= 0)
                return Status::OUT_OF_RANGE;
            if (symmetric)
                {
                    const int32_t reached = _S_round_shift (
                        static_cast<int64_t> (stats._M_abs_max_q15)
                            * out_multiplier_q15[channel],
                        _S_grad_frac_bits);
                    const int8_t bound = static_cast<int8_t> (
                        reached < 1 ? 1 : (reached > 127 ? 127 : reached));
                    out_qmin[channel] = static_cast<int8_t> (-bound);
                    out_qmax[channel] = bound;
                }
            else
                {
                    out_qmin[channel] = -127;
                    out_qmax[channel] = 127;
                }
        }
    return Status::OK;
}

/** @brief Recompute the reciprocal multiplier table of a scale table. */
[[nodiscard]] inline constexpr Status
refresh_multiplier_table (const int32_t *const scale_q15, int32_t *const multiplier_q15,
                          const uint32_t channels) noexcept
{
    if (scale_q15 == nullptr || multiplier_q15 == nullptr)
        return Status::BAD_LAYOUT;
    if (channels == 0u || channels > _S_fq_max_channels)
        return Status::OUT_OF_RANGE;
    for (uint32_t channel = 0; channel < channels; ++channel)
        {
            multiplier_q15[channel] = fq_multiplier_from_scale (scale_q15[channel]);
            if (multiplier_q15[channel] <= 0)
                return Status::OUT_OF_RANGE;
        }
    return Status::OK;
}

/**
 * @brief Learnable-scale variant: the gradient of the quantizer's own scale.
 *
 * With the surrogate `y = scale * code + (-zero_point * scale)`, the residual a
 * lane still carries is `value - scale * code` (the quantization error the scale
 * has not absorbed), and the scale gradient is the projection of the incoming
 * gradient onto that residual, summed per channel:
 *
 * `grad_scale[c] = sum_{i in c} grad_i * (value_i - scale_q15[c] * code_i)`
 *
 * accumulated in int64 (a Q30 product) and narrowed once with `narrow_sum_q15`,
 * so the only rounding is the final one.  One pass over three stride-1 streams,
 * no scratch and no pool lookup.
 */
[[nodiscard]] inline Status
learnable_scale_backward (const int16_t *const values_q15, const int8_t *const codes,
                          const int16_t *const grad_out, int16_t *const grad_scale_q15,
                          const FakeQuantParams &params, const FakeQuantLayout &layout) noexcept
{
    const Status status = params.validate ();
    if (!status_ok (status))
        return status;
    if (values_q15 == nullptr || codes == nullptr || grad_out == nullptr
        || grad_scale_q15 == nullptr || layout._M_valid == 0u)
        return Status::BAD_LAYOUT;
    if (layout._M_channels != params._M_channels)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (values_q15) || !simd_aligned (grad_out)
        || !simd_aligned (grad_scale_q15) || !simd_aligned (codes))
        return Status::BAD_LAYOUT;

    for (uint32_t channel = 0; channel < params._M_channels; ++channel)
        {
            const int32_t scale = params.scale_at (channel);
            const int32_t zero_point = params.zero_point_at (channel);
            int64_t total_q30 = 0;
            for (uint32_t block = 0; block < layout._M_outer; ++block)
                {
                    const uint32_t base = layout.base (channel, block);
                    for (uint32_t i = 0; i < layout._M_run; ++i)
                        {
                            const uint32_t index = base + i;
                            const int32_t dequantized
                                = fq_dequantize_q15 (codes[index], scale, zero_point);
                            total_q30 += static_cast<int64_t> (grad_out[index])
                                         * (values_q15[index] - dequantized);
                        }
                }
            grad_scale_q15[channel] = q15_to_i16_sat (narrow_sum_q15 (total_q30));
        }
    return Status::OK;
}

/**
 * @brief Same gradient through the shared lane kernels, for a channel-major grid.
 *
 * The residual stream is materialized once (2 bytes per element of caller-owned
 * scratch) and reduced with `dot_lanes`, i.e. the SSE2 / SMLAD path.  The code is
 * widened by `train_simd::codes_to_q15` and put on this quantizer's grid by
 * `scale_lanes` with `k = round(2^15 * scale / 258)`, so a lane passes through two
 * roundings instead of one: this form agrees with `learnable_scale_backward` to
 * within one Q15 unit per lane, which is why the fused form above is the one the
 * test oracle pins.
 */
[[nodiscard]] inline Status
learnable_scale_backward_dot (const int16_t *const values_q15, const int8_t *const codes,
                              const int16_t *const grad_out, int16_t *const grad_scale_q15,
                              const FakeQuantParams &params, const FakeQuantLayout &layout,
                              int16_t *const scratch, const uint32_t scratch_capacity) noexcept
{
    const Status status = params.validate ();
    if (!status_ok (status))
        return status;
    if (values_q15 == nullptr || codes == nullptr || grad_out == nullptr
        || grad_scale_q15 == nullptr || scratch == nullptr || layout._M_valid == 0u)
        return Status::BAD_LAYOUT;
    if (scratch_capacity < layout._M_count)
        return Status::ARENA_OVERFLOW;
    if (!simd_aligned (scratch) || !simd_aligned (grad_out) || !simd_aligned (values_q15))
        return Status::BAD_LAYOUT;
    for (uint32_t channel = 0; channel < params._M_channels; ++channel)
        {
            const int32_t scale = params.scale_at (channel);
            const int32_t zero_point = params.zero_point_at (channel);
            /* code * scale == (code * 258) * (2^15 * scale / 258) / 2^15 */
            const int32_t grid_step = fq_narrow_i32_sat (
                fq_ceil_div (static_cast<int64_t> (_S_q15_one) * scale,
                             _S_code_to_q15_scale));
            const int32_t offset = zero_point * scale;
            for (uint32_t block = 0; block < layout._M_outer; ++block)
                {
                    const uint32_t base = layout.base (channel, block);
                    train_simd::codes_to_q15 (codes + base, scratch + base, layout._M_run);
                    scale_lanes (scratch + base, grid_step, layout._M_run);
                    for (uint32_t i = 0; i < layout._M_run; ++i)
                        scratch[base + i] = q15_to_i16_sat (
                            values_q15[base + i] - scratch[base + i] + offset);
                }
        }
    for (uint32_t channel = 0; channel < params._M_channels; ++channel)
        {
            int64_t total_q30 = 0;
            for (uint32_t block = 0; block < layout._M_outer; ++block)
                {
                    const uint32_t base = layout.base (channel, block);
                    /* `dot_lanes` returns the Q15-narrowed sum of this run (a run
                       of 32767-unit lanes still fits int32), so re-widening it and
                       narrowing the merged total keeps the result a single
                       saturating step away from the fused form. */
                    total_q30 += static_cast<int64_t> (
                                     dot_lanes (grad_out + base, scratch + base,
                                                layout._M_run))
                                 << _S_grad_frac_bits;
                }
            grad_scale_q15[channel] = q15_to_i16_sat (narrow_sum_q15 (total_q30));
        }
    return Status::OK;
}

/** @brief Apply one scale gradient: a saturating Q15 step on every channel grid. */
[[nodiscard]] inline Status
learnable_scale_step (int32_t *const scale_q15, const int16_t *const grad_scale_q15,
                      const uint32_t channels, const int32_t lr_q15) noexcept
{
    if (scale_q15 == nullptr || grad_scale_q15 == nullptr)
        return Status::BAD_LAYOUT;
    if (channels == 0u || channels > _S_fq_max_channels)
        return Status::OUT_OF_RANGE;
    for (uint32_t channel = 0; channel < channels; ++channel)
        {
            const int32_t delta = q15_mul (-grad_scale_q15[channel], lr_q15);
            const int64_t next = static_cast<int64_t> (scale_q15[channel]) + delta;
            scale_q15[channel] = static_cast<int32_t> (
                next < 1 ? 1 : (next > _S_q15_max ? _S_q15_max : next));
        }
    return Status::OK;
}

/**
 * @brief Pool-safe clip window: six int32 words.
 *
 * `ClipWindow` mixes 32-bit and 8-bit fields, which a pool slab may not be read
 * through as an array of itself; this is the same six quantities with the two
 * clamp codes packed into one word, so `saved_as<ClipWindowLane>` stays a
 * strictly aligned int32 stream and `as_window` unpacks without a type pun.
 */
struct alignas (4) ClipWindowLane
{
    int32_t _M_words[6];

    [[nodiscard]] ClipWindow
    as_window () const noexcept
    {
        ClipWindow window;
        window._M_low = this->_M_words[0];
        window._M_high = this->_M_words[1];
        window._M_code_low = this->_M_words[2];
        window._M_code_high = this->_M_words[3];
        window._M_qmin = this->qmin ();
        window._M_qmax = this->qmax ();
        window._M_valid = this->_M_words[5] != 0 ? 1u : 0u;
        window._M_pad = 0u;
        return window;
    }

    void
    set_window (const ClipWindow &window) noexcept
    {
        this->_M_words[0] = window._M_low;
        this->_M_words[1] = window._M_high;
        this->_M_words[2] = window._M_code_low;
        this->_M_words[3] = window._M_code_high;
        this->_M_words[4] = static_cast<int32_t> (static_cast<uint8_t> (window._M_qmin))
                            | (static_cast<int32_t> (static_cast<uint8_t> (window._M_qmax)) << 8);
        this->_M_words[5] = window._M_valid;
    }

    [[nodiscard]] int8_t
    qmin () const noexcept
    {
        return static_cast<int8_t> (this->_M_words[4] & 0xFF);
    }

    [[nodiscard]] int8_t
    qmax () const noexcept
    {
        return static_cast<int8_t> ((this->_M_words[4] >> 8) & 0xFF);
    }

    [[nodiscard]] bool
    valid () const noexcept
    {
        return this->_M_words[5] != 0;
    }
};

static_assert (sizeof (ClipWindowLane) == 24u,
               "clip window lane must stay 24 bytes on both targets");

/** @brief Saved layout of a recorded fake-quant node. */
struct alignas (4) FakeQuantNodeState
{
    uint32_t _M_run = 0u;
    uint32_t _M_group = 0u;
    uint32_t _M_outer = 0u;
    uint32_t _M_count = 0u;
    uint32_t _M_channels = 0u;
    uint32_t _M_pad = 0u;
};

static_assert (sizeof (FakeQuantNodeState) == 24u,
               "fake-quant node state must stay 24 bytes");

/** @brief Fill a node state from a layout. */
inline constexpr void
fake_quant_node_state_set (FakeQuantNodeState &state, const FakeQuantLayout &layout) noexcept
{
    state._M_run = layout._M_run;
    state._M_group = layout._M_group;
    state._M_outer = layout._M_outer;
    state._M_count = layout._M_count;
    state._M_channels = layout._M_channels;
    state._M_pad = 0u;
}

/** @brief Rebuild the run layout of a recorded node from its saved state. */
[[nodiscard]] inline constexpr Status
fake_quant_node_layout (const FakeQuantNodeState &state, FakeQuantLayout &out) noexcept
{
    FakeQuantLayout layout;
    layout._M_run = state._M_run;
    layout._M_group = state._M_group;
    layout._M_outer = state._M_outer;
    layout._M_count = state._M_count;
    if (state._M_channels > 0xFFFFu || state._M_run == 0u || state._M_count == 0u)
        return Status::BAD_LAYOUT;
    layout._M_channels = static_cast<uint16_t> (state._M_channels);
    layout._M_axis = state._M_channels == 1u ? _S_fq_per_tensor : 0u;
    layout._M_valid = 1u;
    out = layout;
    return Status::OK;
}

/**
 * @brief Tape backward of a recorded fake-quant node (the tape contract).
 *
 * Reads saved[0] = the forward Q15 lane stream, saved[1] = the per-channel clip
 * windows, saved[2] = the layout, and routes the incoming gradient straight
 * through, zeroing exactly the lanes the forward pass clipped.  Accumulation
 * goes through `Trainable::add_grad` and saturation is reported through
 * `GradContext::_M_saturation_events`, as `mllee/train_graph.hpp` requires.
 */
[[nodiscard]] inline Status
fake_quant_backward (const Trainable &output, const Trainable &grad_output,
                     GradContext &context) noexcept
{
    if (context._M_operand_count < 1u || context._M_saved_count < 3u)
        return Status::BAD_LAYOUT;
    if (output._M_grad == nullptr || grad_output._M_grad == nullptr)
        return Status::BAD_LAYOUT;
    const Trainable &input = context._M_operands[0];
    if (input._M_grad == nullptr)
        return Status::OK;
    if (input._M_count != output._M_count)
        return Status::BAD_LAYOUT;

    const int16_t *const values = context.saved_as<int16_t> (0);
    const ClipWindowLane *const windows = context.saved_as<ClipWindowLane> (1);
    const FakeQuantNodeState *const state = context.saved_as<FakeQuantNodeState> (2);
    if (values == nullptr || windows == nullptr || state == nullptr)
        return Status::BAD_LAYOUT;
    FakeQuantLayout layout;
    const Status layout_status = fake_quant_node_layout (*state, layout);
    if (!status_ok (layout_status))
        return layout_status;
    if (layout._M_count != output._M_count)
        return Status::BAD_LAYOUT;

    const int16_t *const grad_out = grad_output._M_grad;
    for (uint32_t channel = 0; channel < layout._M_channels; ++channel)
        {
            const int32_t low = windows[channel]._M_words[0];
            const int32_t high = windows[channel]._M_words[1];
            if (windows[channel]._M_words[5] == 0)
                return Status::BAD_LAYOUT;
            for (uint32_t block = 0; block < layout._M_outer; ++block)
                {
                    const uint32_t base = layout.base (channel, block);
                    for (uint32_t i = 0; i < layout._M_run; ++i)
                        {
                            const uint32_t index = base + i;
                            const int32_t value = values[index];
                            /* Straight through, clipped lanes excluded. */
                            if (value < low || value > high)
                                continue;
                            if (input.add_grad (index, grad_out[index]))
                                ++context._M_saturation_events;
                        }
                }
        }
    return Status::OK;
}

/**
 * @brief Record a fake-quant op on a tape.
 *
 * @p forward_values the Q15 stream that was quantized this pass (the
 *        pre-activation of the quantizer, not the codes): it is what the exact
 *        clip predicate is evaluated on during the backward walk.
 *
 * Costs `2 * N + 24 * C + 24` pool bytes for the saved streams and is the only
 * way to get the exact STE into a backward pass: the clip windows travel with the
 * node instead of a writable static.
 */
[[nodiscard]] inline Status
record_fake_quant (Tape &tape, const Trainable &input, const Trainable &output,
                   const int16_t *const forward_values, const FakeQuantParams &params,
                   const FakeQuantLayout &layout, const uint32_t op_id = 0u) noexcept
{
    const Status params_status = params.validate ();
    if (!status_ok (params_status))
        return params_status;
    const Status input_status = input.validate ();
    const Status output_status = output.validate ();
    if (!status_ok (input_status) || !status_ok (output_status))
        return Status::BAD_LAYOUT;
    if (forward_values == nullptr || output._M_grad == nullptr)
        return Status::BAD_LAYOUT;
    if (input._M_count != output._M_count)
        return Status::BAD_LAYOUT;
    if (layout._M_valid == 0u || layout._M_channels != params._M_channels
        || layout._M_count != output._M_count)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (forward_values))
        return Status::BAD_LAYOUT;

    Saved values_handle;
    const Status values_status
        = tape.save_copy (forward_values, output._M_count, values_handle);
    if (!status_ok (values_status))
        return values_status;

    Saved window_handle;
    ClipWindowLane *windows = nullptr;
    const Status windows_status
        = tape.save (params._M_channels, window_handle, windows);
    if (!status_ok (windows_status))
        return windows_status;
    for (uint32_t channel = 0; channel < params._M_channels; ++channel)
        windows[channel].set_window (clip_window_of (
            params.multiplier_at (channel), params.zero_point_at (channel),
            params._M_qmin, params._M_qmax));

    Saved state_handle;
    FakeQuantNodeState *state = nullptr;
    const Status state_status = tape.save (1u, state_handle, state);
    if (!status_ok (state_status))
        return state_status;
    fake_quant_node_state_set (*state, layout);

    Trainable operand_views[1] = { input };
    const Saved saved[3] = { values_handle, window_handle, state_handle };
    return tape.record (fake_quant_backward, op_id, output, operand_views, 1u,
                        saved, 3u);
}
} // namespace zzs::mllee::train

#endif /* ZZS_MLLEE_TRAIN_FAKE_QUANT_HPP */
