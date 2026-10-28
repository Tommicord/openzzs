/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#pragma once

#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Max, average, adaptive-average, L2 and global pooling, 1-D and 2-D.
 *
 * @section pool_kinds Kinds and their saved state
 *
 * Every kind reads one contiguous-or-strided spatial plane per (batch item,
 * channel) and reduces a window:
 *
 *   - `PoolKind::MAX` keeps the largest code; the forward saves the winner's
 *     plane-normalized coordinate per output element, which is the only form the
 *     backward needs (a tie goes to the first tap in window order, and the saved
 *     coordinate already encodes that choice);
 *   - `PoolKind::AVERAGE` divides the window sum by its area, either the nominal
 *     `k0 * k1` (`_M_count_include_pad`) or the number of taps that landed inside
 *     the real border.  The forward saves that per-output area, and the backward
 *     divides by the same number, so the padding rule cannot drift between the
 *     two directions;
 *   - `PoolKind::L2` is `isqrt (round (sum of squares / area))`, the code-domain
 *     L2 pooling of the inference op.  Its backward factor needs the input value
 *     at each tap, so an L2 layer also saves the input codes;
 *   - `PoolKind::ADAPTIVE_AVERAGE` takes the output length from the spec and
 *     derives window `start = floor (i in / out)`,
 *     `end = ceil ((i + 1) in / out)` per axis, so windows overlap by at most one
 *     position; padding is not part of the adaptive rule (BAD_LAYOUT);
 *   - the `GLOBAL_*` kinds are one window over the whole plane with an output of
 *     one position per channel; the window is walked as a plane loop instead of a
 *     bounded stack window, so a global layer has no tap-count limit beyond the
 *     element cap.
 *
 * @section pool_backward Scattered or broadcast
 *
 * Max pooling is a scatter -- one destination per output element -- so its
 * backward feeds `scatter_by_index` the plane's own (staged, therefore
 * contiguous whichever layout is in use) gradients, an index stream that maps
 * each output element to its saved winner plane index, and a plane-sized lane
 * buffer as the destination; the tail of the index beyond the output plane is
 * set to the destination count, which the primitive documents as a skip, so it
 * never reads past the staged slice.  Two outputs that picked the same input --
 * overlapping windows -- both land on it, because the primitive adds.  Average
 * and L2 are a broadcast -- one destination per window tap -- so their backward
 * builds a window lane buffer, applies the in-border mask with `route_by_mask`
 * and adds each lane in; a global window covers the whole plane and every tap is
 * inside, so it skips the frame buffer and the mask entirely.  Both paths
 * accumulate (they never overwrite the operand), and both count the lanes that
 * clamped.
 *
 * @section pool_layouts Layouts
 *
 * `[batch, C, H, W]` (`PoolLayout::CHANNEL_MAJOR`) and `[batch, H, W, C]`
 * (`PoolLayout::SPATIAL_MAJOR`) are both supported through one set of strides;
 * rank 1 pins the second spatial axis to 1, so the same loop nest covers 1-D and
 * 2-D.  Output length is
 * `min (in, floor ((in + low + high - kernel) / stride) + 1)` for the fixed-window
 * kinds -- the trailing position that would read only padding is dropped, exactly
 * like the inference ops, and a length below 1 is BAD_LAYOUT.
 *
 * @section pool_pool Pool traffic
 *
 * With `total = batch * channels * out0 * out1`, `plane = in0 * in1` and
 * `n = batch * channels * plane`:
 *
 *   - saved window state : 2 total bytes (argmax coordinates or areas);
 *   - saved input codes  : n bytes, only for the L2 kinds (their backward factor
 *                          needs the tapped value) and only when the input
 *                          collects a gradient;
 *   - backward scratch   : 4 plane + 2 out-plane bytes (the lane plane, the
 *                          padded index plane and the staged output lanes),
 *                          only for the max kinds;
 *   - saved spec         : sizeof (PoolSpec) rounded up to the pool alignment.
 *
 * The window loop itself lives in `_S_pool_window_chunk` stack slots (int16 lanes
 * plus a byte mask) inside `pool_backward_window`, refilled once per chunk of
 * taps, because a bounded frame is what keeps the pooling backward allocatable by
 * a caller that cannot afford a plane.  The max path takes a separate noinline
 * child, so those slots never sit on the frame of a scatter that cannot use them.
 */
namespace zzs::mllee::train
{
/** @brief Taps one fixed-window backward keeps on the frame. */
inline constexpr uint32_t _S_pool_window_max = 64u;

/**
 * @brief Taps one fixed-window backward pass routes, a SIMD lane group.
 *
 * `_S_pool_window_max` is the widest window a layer may ask for; the backward
 * does not reserve that many frame slots, it walks the window in chunks of this
 * size.  Sixteen int16 lanes plus their mask is one 32-byte group, which is what
 * the routing primitive wants, and it bounds the pooling backward's frame by a
 * constant that does not grow with the layer.
 */
inline constexpr uint32_t _S_pool_window_chunk = 16u;

/** @brief Saved slot holding the spec copy. */
inline constexpr uint32_t _S_pool_spec = 0u;

/** @brief Saved slot holding the argmax coordinates or the window areas. */
inline constexpr uint32_t _S_pool_window = 1u;

/** @brief Saved slot holding the backward scratch (max kinds only). */
inline constexpr uint32_t _S_pool_scratch = 2u;

/** @brief Saved slot holding the input codes (L2 kinds only). */
inline constexpr uint32_t _S_pool_input = 3u;

/** @brief Pooling reduction. */
enum class PoolKind : uint8_t
{
    MAX = 0,
    AVERAGE,
    L2,
    ADAPTIVE_AVERAGE,
    GLOBAL_MAX,
    GLOBAL_AVERAGE,
    GLOBAL_L2,
};

/** @brief Element order of the pooled spatial axes. */
enum class PoolLayout : uint8_t
{
    CHANNEL_MAJOR = 0,
    SPATIAL_MAJOR = 1,
};

/** @brief True when @p kind is one of the max reductions. */
[[nodiscard]] inline constexpr bool
pool_is_max (const PoolKind kind) noexcept
{
    return kind == PoolKind::MAX || kind == PoolKind::GLOBAL_MAX;
}

/** @brief True when @p kind averages or maximizes over the whole plane. */
[[nodiscard]] inline constexpr bool
pool_is_global (const PoolKind kind) noexcept
{
    return kind == PoolKind::GLOBAL_MAX || kind == PoolKind::GLOBAL_AVERAGE
           || kind == PoolKind::GLOBAL_L2;
}

/** @brief True when the backward needs the tapped input values. */
[[nodiscard]] inline constexpr bool
pool_needs_input (const PoolKind kind) noexcept
{
    return kind == PoolKind::L2 || kind == PoolKind::GLOBAL_L2;
}

/** @brief True when @p kind is one of the L2 reductions. */
[[nodiscard]] inline constexpr bool
pool_is_l2 (const PoolKind kind) noexcept
{
    return kind == PoolKind::L2 || kind == PoolKind::GLOBAL_L2;
}

/** @brief True when @p kind is one of the supported reductions. */
[[nodiscard]] inline constexpr bool
pool_kind_valid (const PoolKind kind) noexcept
{
    return kind >= PoolKind::MAX && kind <= PoolKind::GLOBAL_L2;
}

/**
 * @brief One pooling layer.
 *
 * `ADAPTIVE_AVERAGE` reads `_M_out` as its target length and ignores kernel and
 * stride; the global kinds ignore everything but the input extent; the fixed
 * kinds derive `_M_out` from kernel and stride, and an `_M_out` of 0 asks for that
 * derivation.
 */
struct PoolSpec
{
    uint16_t _M_batch = 1u;
    uint16_t _M_channels = 1u;
    uint16_t _M_in[2] = { 1u, 1u };
    uint16_t _M_out[2] = { 0u, 0u };
    uint8_t _M_kernel[2] = { 1u, 1u };
    uint8_t _M_stride[2] = { 1u, 1u };
    int8_t _M_pad[4] = { 0, 0, 0, 0 };
    PoolKind _M_kind = PoolKind::MAX;
    PoolLayout _M_layout = PoolLayout::CHANNEL_MAJOR;
    uint8_t _M_rank = 1u;
    uint8_t _M_count_include_pad = 1u;
    uint32_t _M_op_id = 0u;

    /** @brief Axis count: 2 for rank 2, 1 for rank 1 (axis 1 stays pinned). */
    [[nodiscard]] constexpr uint32_t
    axes () const noexcept
    {
        return this->_M_rank == 2u ? 2u : 1u;
    }

    /** @brief Output length along @p axis, derived for the fixed-window kinds. */
    [[nodiscard]] constexpr uint32_t
    out_length (const uint32_t axis) const noexcept
    {
        if (axis >= this->axes ())
            return 1u;
        if (this->_M_out[axis] != 0u)
            return this->_M_out[axis];
        if (pool_is_global (this->_M_kind))
            return 1u;
        /* A zero stride is a shape error `validate ()` reports, but the length
           helpers run before it, so pin the divide here: 0 means "no output
           positions", which validate already rejects. */
        if (this->_M_stride[axis] == 0u)
            return 0u;
        const int32_t span = static_cast<int32_t> (this->_M_in[axis])
                             + this->_M_pad[axis * 2u]
                             + this->_M_pad[axis * 2u + 1u]
                             - this->_M_kernel[axis];
        if (span < 0)
            return 0u;
        const int32_t placed = span / this->_M_stride[axis] + 1;
        const int32_t limit = static_cast<int32_t> (this->_M_in[axis]);
        return static_cast<uint32_t> (placed < limit ? placed : limit);
    }

    /** @brief Output positions per channel. */
    [[nodiscard]] constexpr uint32_t
    out_plane () const noexcept
    {
        return this->out_length (0u) * this->out_length (1u);
    }

    /** @brief Input positions per channel. */
    [[nodiscard]] constexpr uint32_t
    in_plane () const noexcept
    {
        return static_cast<uint32_t> (this->_M_in[0]) * this->_M_in[1];
    }

    /** @brief Output element count. */
    [[nodiscard]] constexpr uint32_t
    output_elements () const noexcept
    {
        return static_cast<uint32_t> (this->_M_batch) * this->_M_channels
               * this->out_plane ();
    }

    /** @brief Input element count. */
    [[nodiscard]] constexpr uint32_t
    input_elements () const noexcept
    {
        return static_cast<uint32_t> (this->_M_batch) * this->_M_channels
               * this->in_plane ();
    }

    /** @brief Largest number of taps one output position can span. */
    [[nodiscard]] constexpr uint32_t
    max_window () const noexcept
    {
        uint32_t bound = 1u;
        for (uint32_t axis = 0; axis < this->axes (); ++axis)
            {
                const uint32_t in = this->_M_in[axis];
                const uint32_t out = this->out_length (axis);
                uint32_t length = 0u;
                if (this->_M_kind == PoolKind::ADAPTIVE_AVERAGE)
                    {
                        if (out == 0u)
                            return 0u;
                        const uint32_t first
                            = static_cast<uint32_t> (in * (out - 1u)) / out + 1u;
                        const uint32_t last
                            = (in * out + in + out - 1u) / out;
                        length = last > first ? last - first : 1u;
                    }
                else if (pool_is_global (this->_M_kind))
                    {
                        return in;
                    }
                else
                    length = this->_M_kernel[axis];
                bound *= length;
            }
        return bound;
    }

    /** @brief Bytes of the backward scratch the max kinds need. */
    [[nodiscard]] constexpr uint32_t
    scratch_bytes () const noexcept
    {
        if (!pool_is_max (this->_M_kind))
            return 0u;
        const uint32_t plane
            = align_up (2u * this->in_plane (), _S_pool_alignment);
        /* lane plane + index plane + one staged output plane (a spatial-major
           channel is not contiguous, so its lanes are packed before the
           scatter).  out_plane never exceeds in_plane (see validate). */
        return 2u * plane
               + align_up (2u * this->out_plane (), _S_pool_alignment);
    }

    /** @brief Pool bytes one recorded layer needs at its worst case. */
    [[nodiscard]] constexpr uint32_t
    bytes () const noexcept
    {
        const uint32_t window
            = align_up (2u * this->output_elements (), _S_pool_alignment);
        const uint32_t input
            = pool_needs_input (this->_M_kind)
                  ? align_up (this->input_elements (), _S_pool_alignment)
                  : 0u;
        return window + input + this->scratch_bytes ()
               + align_up (static_cast<uint32_t> (sizeof (PoolSpec)),
                           _S_pool_alignment);
    }

    [[nodiscard]] constexpr Shape
    input_shape () const noexcept
    {
        if (this->_M_rank == 1u)
            return Shape::tensor3 (this->_M_batch, this->_M_channels,
                                   this->_M_in[0]);
        return Shape::tensor4 (this->_M_batch, this->_M_channels, this->_M_in[0],
                               this->_M_in[1]);
    }

    [[nodiscard]] constexpr Shape
    output_shape () const noexcept
    {
        if (this->_M_rank == 1u)
            return Shape::tensor3 (this->_M_batch, this->_M_channels,
                                   static_cast<uint16_t> (this->out_length (0u)));
        return Shape::tensor4 (this->_M_batch, this->_M_channels,
                               static_cast<uint16_t> (this->out_length (0u)),
                               static_cast<uint16_t> (this->out_length (1u)));
    }

    /**
     * @brief Check the window and output geometry.
     *
     * Every output axis is validated against `in_plane ()` as well: an output
     * never extends past the input, which is the invariant the max backward
     * relies on when it pads its index buffer with a skip sentinel.
     *
     * @return BAD_LAYOUT for an unsupported kind, a zero axis, a stride or kernel
     *         of 0, padding (or a kernel wider than the input) on an adaptive
     *         layer, an output length below 1 or a caller-supplied output that
     *         exceeds the input; OUT_OF_RANGE for an element count a training
     *         tensor cannot address or a window wider than
     *         `_S_pool_window_max` taps.  A window whose taps all fall outside
     *         the real border is not a shape error and is reported by the
     *         forward, which is the only pass that walks the taps.
     */
    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (!pool_kind_valid (this->_M_kind))
            return Status::BAD_LAYOUT;
        if (this->_M_rank == 0u || this->_M_rank > 2u)
            return Status::BAD_LAYOUT;
        if (this->_M_batch == 0u || this->_M_channels == 0u)
            return Status::BAD_LAYOUT;
        if (this->_M_in[0] == 0u || this->_M_in[1] == 0u)
            return Status::BAD_LAYOUT;
        if (this->_M_layout != PoolLayout::CHANNEL_MAJOR
            && this->_M_layout != PoolLayout::SPATIAL_MAJOR)
            return Status::BAD_LAYOUT;
        if (this->_M_rank == 1u
            && (this->_M_in[1] != 1u || this->_M_kernel[1] != 1u
                || this->_M_stride[1] != 1u || this->_M_out[1] > 1u
                || this->_M_pad[2] != 0 || this->_M_pad[3] != 0))
            return Status::BAD_LAYOUT;
        const uint32_t axes = this->axes ();
        const bool adaptive = this->_M_kind == PoolKind::ADAPTIVE_AVERAGE;
        for (uint32_t axis = 0; axis < axes; ++axis)
            {
                const uint32_t out = this->out_length (axis);
                if (out == 0u || out > this->_M_in[axis])
                    return Status::BAD_LAYOUT;
                if (pool_is_global (this->_M_kind))
                    {
                        if (out != 1u)
                            return Status::BAD_LAYOUT;
                        continue;
                    }
                if (this->_M_kernel[axis] == 0u || this->_M_stride[axis] == 0u)
                    return Status::BAD_LAYOUT;
                if (adaptive)
                    {
                        if (this->_M_kernel[axis] > this->_M_in[axis])
                            return Status::BAD_LAYOUT;
                        if (this->_M_pad[axis * 2u] != 0
                            || this->_M_pad[axis * 2u + 1u] != 0)
                            return Status::BAD_LAYOUT;
                        continue;
                    }
                const int32_t span = static_cast<int32_t> (this->_M_in[axis])
                                     + this->_M_pad[axis * 2u]
                                     + this->_M_pad[axis * 2u + 1u]
                                     - this->_M_kernel[axis];
                if (span < 0)
                    return Status::BAD_LAYOUT;
            }
        if (this->out_plane () == 0u)
            return Status::BAD_LAYOUT;
        if (this->input_elements () > _S_max_elements
            || this->output_elements () > _S_max_elements)
            return Status::OUT_OF_RANGE;
        if (this->in_plane () > _S_max_elements)
            return Status::OUT_OF_RANGE;
        const uint32_t bound = this->max_window ();
        if (bound == 0u)
            return Status::BAD_LAYOUT;
        if (!pool_is_global (this->_M_kind) && bound > _S_pool_window_max)
            return Status::OUT_OF_RANGE;
        return Status::OK;
    }
};

static_assert (sizeof (PoolSpec) == 28u, "pool spec is seven pool words");
static_assert (sizeof (PoolKind) == 1u && sizeof (PoolLayout) == 1u,
               "the pooling tags are single bytes");

/** @brief Element strides of one pooled tensor, in elements. */
struct PoolStrides
{
    uint32_t _M_batch = 0u;
    uint32_t _M_channel = 0u;
    uint32_t _M_axis0 = 0u;
    uint32_t _M_axis1 = 0u;
};

static_assert (sizeof (PoolStrides) == 16u, "four stride words");

/** @brief Strides of the input tensor. */
[[nodiscard]] inline constexpr PoolStrides
pool_input_strides (const PoolSpec &spec) noexcept
{
    if (spec._M_layout == PoolLayout::CHANNEL_MAJOR)
        return PoolStrides{ static_cast<uint32_t> (spec._M_channels)
                                * spec.in_plane (),
                            spec.in_plane (), spec._M_in[1], 1u };
    return PoolStrides{ spec.in_plane () * spec._M_channels, 1u,
                        static_cast<uint32_t> (spec._M_in[1])
                            * spec._M_channels,
                        spec._M_channels };
}

/** @brief Strides of the output tensor. */
[[nodiscard]] inline constexpr PoolStrides
pool_output_strides (const PoolSpec &spec) noexcept
{
    const uint32_t plane = spec.out_plane ();
    if (spec._M_layout == PoolLayout::CHANNEL_MAJOR)
        return PoolStrides{ static_cast<uint32_t> (spec._M_channels) * plane,
                            plane, spec.out_length (1u), 1u };
    return PoolStrides{ plane * spec._M_channels, 1u,
                        static_cast<uint32_t> (spec.out_length (1u))
                            * spec._M_channels,
                        spec._M_channels };
}

/** @brief Flat element at (@p outer, @p channel, @p a0, @p a1). */
[[nodiscard]] inline constexpr uint32_t
pool_element (const PoolStrides &strides, const uint32_t outer,
              const uint32_t channel, const uint32_t a0,
              const uint32_t a1) noexcept
{
    return outer * strides._M_batch + channel * strides._M_channel
           + a0 * strides._M_axis0 + a1 * strides._M_axis1;
}

/** @brief Window of one output position: first coordinate and tap count. */
struct PoolWindow
{
    int32_t _M_start[2] = { 0, 0 };
    uint32_t _M_length[2] = { 1u, 1u };
};

static_assert (sizeof (PoolWindow) == 16u, "two starts and two lengths");

/** @brief The window that output position @p flat reads, per axis. */
[[nodiscard]] inline constexpr PoolWindow
pool_window (const PoolSpec &spec, const uint32_t flat) noexcept
{
    PoolWindow window;
    const uint32_t out1 = spec.out_length (1u);
    const uint32_t a0 = out1 != 0u ? flat / out1 : 0u;
    const uint32_t a1 = out1 != 0u ? flat % out1 : flat;
    for (uint32_t axis = 0; axis < 2u; ++axis)
        {
            if (axis >= spec.axes ())
                {
                    window._M_start[axis] = 0;
                    window._M_length[axis] = 1u;
                    continue;
                }
            const uint32_t in = spec._M_in[axis];
            const uint32_t out = spec.out_length (axis);
            const uint32_t coordinate = axis == 0u ? a0 : a1;
            if (pool_is_global (spec._M_kind))
                {
                    window._M_start[axis] = 0;
                    window._M_length[axis] = in;
                }
                else if (spec._M_kind == PoolKind::ADAPTIVE_AVERAGE)
                {
                    const uint32_t start = out != 0u ? in * coordinate / out : 0u;
                    uint32_t end = out != 0u
                                       ? (in * (coordinate + 1u) + out - 1u) / out
                                       : in;
                    if (end <= start)
                        end = start + 1u;
                    window._M_start[axis] = static_cast<int32_t> (start);
                    window._M_length[axis] = end - start;
                }
                else
                {
                    window._M_start[axis] = static_cast<int32_t> (
                        coordinate * spec._M_stride[axis])
                                            - spec._M_pad[axis * 2u];
                    window._M_length[axis] = spec._M_kernel[axis];
                }
        }
    return window;
}

/**
 * @brief One window tap: its source coordinate per axis, whether it is inside the
 *        real border, and the flat input element it reads.
 */
struct PoolTap
{
    uint32_t _M_element = 0u;
    uint32_t _M_plane_index = 0u;
    bool _M_inside = false;
};

static_assert (sizeof (PoolTap) == 12u, "two ids and a flag");

/**
 * @brief Resolve tap (@p t0, @p t1) of @p window for position @p flat.
 *
 * The coordinate test is the only padding mechanism: a tap outside `[0, in)` is
 * reported as outside and never addressed, so no read or write ever leaves the
 * tensor.
 */
[[nodiscard]] inline constexpr PoolTap
pool_tap (const PoolSpec &spec, const PoolStrides &strides,
          const PoolWindow &window, const uint32_t item, const uint32_t channel,
          const uint32_t t0, const uint32_t t1) noexcept
{
    PoolTap tap;
    if (window._M_start[0] + static_cast<int32_t> (t0) < 0)
        return tap;
    const uint32_t source0 = static_cast<uint32_t> (window._M_start[0]
                                                    + static_cast<int32_t> (t0));
    const uint32_t source1 = static_cast<uint32_t> (window._M_start[1]
                                                    + static_cast<int32_t> (t1));
    const bool inside0 = source0 < spec._M_in[0];
    const bool inside1 = source1 < spec._M_in[1];
    tap._M_inside = inside0 && inside1;
    if (!tap._M_inside)
        return tap;
    tap._M_plane_index = source0 * spec._M_in[1] + source1;
    tap._M_element = pool_element (strides, item, channel, source0, source1);
    return tap;
}

/**
 * @brief Tap @p tap_index of a @p length1-wide window, counting in row order.
 *
 * The backward walks a window in chunks, so the tap loops are flattened; this is
 * the row-major decomposition that keeps tap `q` the same tap the nested
 * `t0 * length1 + t1` loop would have visited, which is what makes the chunked
 * gradient -- and the order the lanes are added in -- identical.
 */
[[nodiscard]] inline constexpr PoolTap
pool_window_tap (const PoolSpec &spec, const PoolStrides &strides,
                 const PoolWindow &window, const uint32_t item,
                 const uint32_t channel, const uint32_t tap_index,
                 const uint32_t length1) noexcept
{
    const uint32_t t0 = length1 != 0u ? tap_index / length1 : 0u;
    const uint32_t t1 = length1 != 0u ? tap_index % length1 : tap_index;
    return pool_tap (spec, strides, window, item, channel, t0, t1);
}

/** @brief Bound a tensor against the count and rank @p spec expects. */
[[nodiscard]] inline constexpr Status
pool_check_operand (const Trainable &tensor, const uint32_t elements,
                    const uint8_t rank) noexcept
{
    if (tensor._M_code == nullptr)
        return Status::BAD_LAYOUT;
    if (tensor._M_count != elements)
        return Status::BAD_LAYOUT;
    if (tensor._M_shape._M_rank != 0u && tensor._M_shape._M_rank != rank)
        return Status::BAD_LAYOUT;
    return Status::OK;
}

/**
 * @brief Flat element of the windowed output position @p flat of one
 *        (batch item, channel) plane.
 *
 * The cursor order both directions walk is (item, channel, flat), which is only
 * the memory order of a channel-major tensor; a spatial-major plane has to be
 * addressed through its strides, so one helper resolves positions for the
 * forward, the saved state and the backward and they cannot disagree.
 */
[[nodiscard]] inline constexpr uint32_t
pool_output_element (const PoolSpec &spec, const PoolStrides &strides,
                     const uint32_t item, const uint32_t channel,
                     const uint32_t flat) noexcept
{
    const uint32_t out1 = spec.out_length (1u);
    return pool_element (strides, item, channel,
                         out1 != 0u ? flat / out1 : 0u,
                         out1 != 0u ? flat % out1 : flat);
}

/**
 * @brief The gradient one window tap receives from the output lane @p grad.
 *
 * Average and adaptive average split the lane evenly over the window (`1 / area`
 * per tap); L2 pools `y = isqrt (sum of squares / area)`, so its derivative is
 * `x_t / (area * y)` and vanishes when the pooled value is zero, where the square
 * root has no slope.  @p value is the tapped code (ignored unless the kind is L2)
 * and @p y the pooled output code.
 */
[[nodiscard]] inline constexpr int32_t
pool_window_lane (const PoolSpec &spec, const int32_t grad,
                  const uint32_t area, const int32_t y,
                  const int32_t value) noexcept
{
    if (area == 0u)
        return 0;
    if (pool_is_l2 (spec._M_kind))
        /* The product `area * y` is a denominator, so it is bounded to keep the
           int32 form exact for every window a valid spec can describe. */
        return y > 0
                   ? div_round (grad * value,
                                static_cast<int32_t> (area > 0x7FFFFu ? 0x7FFFFu
                                                                      : area)
                                    * y)
                   : 0;
    return div_round (grad, static_cast<int32_t> (area));
}

/**
 * @brief Forward pass with no tape recording.
 *
 * @param window_state  `output_elements ()` slots: the winner coordinate per
 *                      output element for the max kinds, the tap area for the
 *                      others.  It is mandatory -- the backward reads it, and a
 *                      forward without it could not be differentiated.
 */
[[nodiscard]] inline Status
pool_forward (const PoolSpec &spec, const Trainable &input, Trainable &output,
              uint16_t *const window_state) noexcept
{
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    const uint8_t tensor_rank = spec._M_rank == 1u ? 3u : 4u;
    if (!status_ok (pool_check_operand (input, spec.input_elements (),
                                        tensor_rank)))
        return Status::BAD_LAYOUT;
    if (!status_ok (pool_check_operand (output, spec.output_elements (),
                                        tensor_rank)))
        return Status::BAD_LAYOUT;
    if (window_state == nullptr)
        return Status::BAD_LAYOUT;

    const PoolStrides in_stride = pool_input_strides (spec);
    const PoolStrides out_stride = pool_output_strides (spec);
    const uint32_t plane = spec.out_plane ();
    const bool global = pool_is_global (spec._M_kind);
    const uint32_t window_bound = spec.max_window ();
    if (!global && window_bound > _S_pool_window_max)
        return Status::OUT_OF_RANGE;

    uint32_t cursor = 0u;
    for (uint32_t item = 0; item < spec._M_batch; ++item)
        for (uint32_t channel = 0; channel < spec._M_channels; ++channel)
            for (uint32_t flat = 0; flat < plane; ++flat, ++cursor)
                {
                    const PoolWindow window = pool_window (spec, flat);
                    int32_t sum = 0;
                    int32_t square = 0;
                    int32_t best = -128;
                    uint32_t best_plane = 0u;
                    uint32_t inside = 0u;
                    const uint32_t length1
                        = global ? spec._M_in[1] : window._M_length[1];
                    const uint32_t length0
                        = global ? spec._M_in[0] : window._M_length[0];
                    for (uint32_t t0 = 0; t0 < length0; ++t0)
                        for (uint32_t t1 = 0; t1 < length1; ++t1)
                            {
                                const PoolTap tap = pool_tap (spec, in_stride,
                                                              window, item,
                                                              channel, t0, t1);
                                if (!tap._M_inside)
                                    continue;
                                const int32_t value = input._M_code[tap._M_element];
                                ++inside;
                                if (pool_is_max (spec._M_kind))
                                    {
                                        if (value > best)
                                            {
                                                best = value;
                                                best_plane = tap._M_plane_index;
                                            }
                                        continue;
                                    }
                                sum += value;
                                square += value * value;
                            }
                    /* A window whose taps all fall in the padding carries no
                       input at all: it has no argmax and its mean is undefined,
                       so the layer is rejected rather than given a magic value. */
                    if (inside == 0u)
                        return Status::BAD_LAYOUT;
                    const uint32_t nominal = length0 * length1;
                    const uint32_t area = spec._M_count_include_pad != 0u
                                              ? (nominal != 0u ? nominal : 1u)
                                              : (inside != 0u ? inside : 1u);
                    if (area > 0xFFFFu)
                        return Status::OUT_OF_RANGE;
                    window_state[cursor] = pool_is_max (spec._M_kind)
                                               ? static_cast<uint16_t> (best_plane)
                                               : static_cast<uint16_t> (area);
                    int32_t code = 0;
                    switch (spec._M_kind)
                        {
                        case PoolKind::MAX:
                        case PoolKind::GLOBAL_MAX:
                            code = best;
                            break;
                        case PoolKind::AVERAGE:
                        case PoolKind::ADAPTIVE_AVERAGE:
                        case PoolKind::GLOBAL_AVERAGE:
                            code = div_round (sum, static_cast<int32_t> (area));
                            break;
                        case PoolKind::L2:
                        case PoolKind::GLOBAL_L2:
                            {
                                const uint32_t mean
                                    = static_cast<uint32_t> (div_round (
                                        square, static_cast<int32_t> (area)));
                                code = static_cast<int32_t> (_S_isqrt (mean));
                            }
                            break;
                        }
                    output._M_code[pool_output_element (spec, out_stride, item,
                                                         channel, flat)]
                        = _S_i8_code (code);
                }
    output.zero_grad ();
    return Status::OK;
}

/**
 * @brief The max reductions: one plane-sized scatter per channel.
 *
 * The index stream is read by the primitive as "output flat -> winner plane",
 * so it is `in_plane` long with its tail beyond `plane` set to the skip
 * sentinel: two outputs that pick the same input (overlapping windows) then both
 * land on it, which an inverse map could not express.  The staged lanes are the
 * plane's own gradients, folded into the operand with saturating adds.  The
 * three scratch slices come from the recorded slab, so this frame carries
 * pointers only.
 */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
pool_backward_max (const Trainable &grad_output, const Trainable &input,
                   const PoolSpec &spec, const uint16_t *window_state,
                   int8_t *const scratch, uint32_t &saturated) noexcept
{
    if (scratch == nullptr || !simd_aligned (scratch))
        return Status::BAD_LAYOUT;
    const uint32_t plane_bytes = align_up (2u * spec.in_plane (),
                                           _S_pool_alignment);
    int16_t *const lane_scratch = reinterpret_cast<int16_t *> (scratch);
    uint16_t *const index_scratch
        = reinterpret_cast<uint16_t *> (scratch + plane_bytes);
    int16_t *const grad_stage
        = reinterpret_cast<int16_t *> (scratch + 2u * plane_bytes);

    const PoolStrides in_stride = pool_input_strides (spec);
    const PoolStrides out_stride = pool_output_strides (spec);
    const uint32_t plane = spec.out_plane ();
    const uint32_t in_plane = spec.in_plane ();
    const uint32_t in1 = spec._M_in[1];
    const uint16_t sentinel = static_cast<uint16_t> (in_plane);

    uint32_t cursor = 0u;
    for (uint32_t item = 0; item < spec._M_batch; ++item)
        for (uint32_t channel = 0; channel < spec._M_channels; ++channel)
            {
                for (uint32_t flat = 0; flat < plane; ++flat, ++cursor)
                    {
                        grad_stage[flat]
                            = grad_output._M_grad[pool_output_element (
                                  spec, out_stride, item, channel, flat)];
                        const uint32_t q = window_state[cursor];
                        if (q >= in_plane)
                            return Status::BAD_LAYOUT;
                        index_scratch[flat] = static_cast<uint16_t> (q);
                    }
                for (uint32_t q = plane; q < in_plane; ++q)
                    index_scratch[q] = sentinel;
                scatter_by_index (grad_stage, index_scratch, lane_scratch,
                                  in_plane, saturated);
                for (uint32_t q = 0u; q < in_plane; ++q)
                    {
                        if (lane_scratch[q] == 0)
                            continue;
                        const uint32_t a0 = in1 != 0u ? q / in1 : 0u;
                        const uint32_t a1 = in1 != 0u ? q % in1 : q;
                        if (input.add_grad (
                                pool_element (in_stride, item, channel, a0, a1),
                                lane_scratch[q]))
                            ++saturated;
                    }
            }
    return Status::OK;
}

/**
 * @brief The average, adaptive and L2 reductions: one window broadcast per
 *        output position.
 *
 * A fixed window builds its lane buffer and its in-border mask on the frame,
 * routes them through `route_by_mask` and adds each surviving lane into the
 * operand -- `_S_pool_window_chunk` taps at a time, so the frame is a constant
 * however wide the layer is (`validate ()` still caps a window at
 * `_S_pool_window_max` taps).  A global window covers the whole plane, every tap
 * is inside by construction, so it takes the same route one tap at a time with no
 * frame buffer at all.
 */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
pool_backward_window (const Trainable &output, const Trainable &grad_output,
                      const Trainable &input, const PoolSpec &spec,
                      const uint16_t *window_state, const int8_t *input_codes,
                      uint32_t &saturated) noexcept
{
    const PoolStrides in_stride = pool_input_strides (spec);
    const PoolStrides out_stride = pool_output_strides (spec);
    const uint32_t plane = spec.out_plane ();
    const bool global = pool_is_global (spec._M_kind);
    int16_t window_lanes[_S_pool_window_chunk];
    uint8_t window_mask[_S_pool_window_chunk];

    uint32_t cursor = 0u;
    for (uint32_t item = 0; item < spec._M_batch; ++item)
        for (uint32_t channel = 0; channel < spec._M_channels; ++channel)
            for (uint32_t flat = 0; flat < plane; ++flat, ++cursor)
                {
                    const uint32_t out_element = pool_output_element (
                        spec, out_stride, item, channel, flat);
                    const int32_t grad = grad_output._M_grad[out_element];
                    const PoolWindow window = pool_window (spec, flat);
                    const uint32_t length0
                        = global ? spec._M_in[0] : window._M_length[0];
                    const uint32_t length1
                        = global ? spec._M_in[1] : window._M_length[1];
                    const uint32_t count = length0 * length1;
                    /* validate () caps a fixed window at
                       _S_pool_window_max taps, so a wider window is always a
                       global one: it covers the whole plane, every tap is
                       inside the border by construction, and no frame buffer
                       or mask is needed. */
                    if (count > _S_pool_window_max && !global)
                        return Status::OUT_OF_RANGE;
                    const uint32_t area = window_state[cursor];
                    const int32_t y = output._M_code[out_element];
                    if (global)
                        {
                            for (uint32_t t0 = 0; t0 < length0; ++t0)
                                for (uint32_t t1 = 0; t1 < length1; ++t1)
                                    {
                                        const PoolTap tap = pool_tap (
                                            spec, in_stride, window, item,
                                            channel, t0, t1);
                                        if (!tap._M_inside)
                                            continue;
                                        const int32_t value
                                            = pool_is_l2 (spec._M_kind)
                                                  ? input_codes[tap._M_element]
                                                  : 0;
                                        const int32_t lane = pool_window_lane (
                                            spec, grad, area, y, value);
                                        if (lane == 0)
                                            continue;
                                        if (input.add_grad (tap._M_element,
                                                            lane))
                                            ++saturated;
                                    }
                            continue;
                        }
                    /* The frame carries one chunk of taps, not the whole window:
                       a 64-tap layer is four passes of `_S_pool_window_chunk`
                       taps, so a wide window costs the same stack as a narrow
                       one.  Row order is preserved across the chunks, so the
                       lanes reach `add_grad` in exactly the sequence the whole
                       window would have used. */
                    for (uint32_t base = 0u; base < count;
                         base += _S_pool_window_chunk)
                        {
                            const uint32_t length
                                = count - base < _S_pool_window_chunk
                                      ? count - base
                                      : _S_pool_window_chunk;
                            for (uint32_t k = 0u; k < length; ++k)
                                {
                                    const PoolTap tap = pool_window_tap (
                                        spec, in_stride, window, item, channel,
                                        base + k, length1);
                                    const int32_t value
                                        = tap._M_inside
                                              ? input_codes[tap._M_element]
                                              : 0;
                                    window_lanes[k] = q15_to_i16_sat (
                                        pool_window_lane (spec, grad, area, y,
                                                          value));
                                    window_mask[k] = tap._M_inside ? 1u : 0u;
                                }
                            route_by_mask (window_lanes, window_mask,
                                           window_lanes, length);
                            for (uint32_t k = 0u; k < length; ++k)
                                {
                                    if (window_lanes[k] == 0
                                        || !window_mask[k])
                                        continue;
                                    const PoolTap tap = pool_window_tap (
                                        spec, in_stride, window, item, channel,
                                        base + k, length1);
                                    if (input.add_grad (tap._M_element,
                                                        window_lanes[k]))
                                        ++saturated;
                                }
                        }
                }
    return Status::OK;
}

/**
 * @brief Backward pass in the `BackwardFn` signature.
 *
 * Recorded node layout: operands [input] (a second operand slot is kept empty so
 * a pooling node never looks like a two-tensor op); saved
 * [spec, window state, scratch | null, input copy | null].
 *
 * The dispatch is deliberately thin.  A max node stages whole planes in recorded
 * scratch and a windowed node keeps a chunk of taps on the frame, so one shared
 * body would size the pooling backward for the sum of two paths that never run
 * together; each child owns only its own buffers and this frame carries the
 * validation both of them need.  Max scatters through
 * `scatter_by_index`, average, adaptive and L2 broadcast through
 * `route_by_mask`, and the global kinds do the same over the whole plane without
 * a window bound.  Both paths accumulate -- they never overwrite the operand --
 * and both count the lanes that clamped.
 */
[[nodiscard]] inline Status
pool_backward (const Trainable &output, const Trainable &grad_output,
               GradContext &context) noexcept
{
    const PoolSpec *const slot = context.saved_as<PoolSpec> (_S_pool_spec);
    if (slot == nullptr)
        return Status::BAD_LAYOUT;
    const Status status = slot->validate ();
    if (!status_ok (status))
        return status;
    if (context._M_operands == nullptr || context._M_operand_count < 1u)
        return Status::BAD_LAYOUT;
    const Trainable &input = context._M_operands[0];
    if (output._M_count != slot->output_elements ()
        || input._M_count != slot->input_elements ())
        return Status::BAD_LAYOUT;
    if (grad_output._M_grad == nullptr)
        return Status::BAD_LAYOUT;
    if (input._M_grad == nullptr)
        return Status::OK;
    const uint16_t *const window_state
        = context.saved_as<uint16_t> (_S_pool_window);
    if (window_state == nullptr)
        return Status::BAD_LAYOUT;

    uint32_t saturated = 0u;
    Status done;
    if (pool_is_max (slot->_M_kind))
        done = pool_backward_max (grad_output, input, *slot, window_state,
                                  context.saved_as<int8_t> (_S_pool_scratch),
                                  saturated);
    else
        {
            const int8_t *const saved_input
                = context.saved_as<int8_t> (_S_pool_input);
            done = pool_backward_window (
                output, grad_output, input, *slot, window_state,
                saved_input != nullptr ? saved_input : input._M_code, saturated);
        }
    if (!status_ok (done))
        return done;
    context._M_saturation_events += saturated;
    return Status::OK;
}

/**
 * @brief Save the pooling state, run the forward and record the node.
 *
 * All four tensors come from one carve of the tape's own pool, so `bytes ()` is
 * the whole cost of a pooling node and a backward function never allocates.
 *
 * @return BAD_LAYOUT for a shape mismatch, ARENA_OVERFLOW when the pool cannot
 *         supply the state, INTERNAL when the tape is not open, OK otherwise.
 */
[[nodiscard]] inline Status
pool_run (Tape &tape, const PoolSpec &spec, const Trainable &input,
          Trainable &output) noexcept
{
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    if (input._M_count != spec.input_elements ()
        || output._M_count != spec.output_elements ())
        return Status::BAD_LAYOUT;

    Pool &pool = tape.pool ();
    Saved slots[_S_max_operands];
    uint8_t *raw = nullptr;
    Slab slab;
    const Status carve = pool.carve (spec.bytes (), 1u, slab);
    if (!status_ok (carve))
        return carve;
    raw = pool.at (slab._M_offset);
    if (raw == nullptr)
        return Status::BAD_LAYOUT;

    uint16_t *window_state = nullptr;
    int8_t *saved_input = nullptr;
    PoolSpec *saved_spec = nullptr;

    // Lay the tensors inside one carve so the accounting is one number.  The max
    // scratch regions are only stepped over here: `pool_backward` unpacks them
    // from the `_S_pool_scratch` handle at the offsets its header comment lists.
    uint32_t offset = 0u;
    constexpr uint32_t spec_bytes
        = align_up (static_cast<uint32_t> (sizeof (PoolSpec)),
                    _S_pool_alignment);
    {
        saved_spec = reinterpret_cast<PoolSpec *> (raw + offset);
        offset += spec_bytes;
    }
    {
        const uint32_t elements = spec.output_elements ();
        window_state = reinterpret_cast<uint16_t *> (raw + offset);
        offset += align_up (2u * elements, _S_pool_alignment);
    }
    if (pool_is_max (spec._M_kind))
        offset += spec.scratch_bytes ();
    if (pool_needs_input (spec._M_kind) && input._M_grad != nullptr)
        saved_input = reinterpret_cast<int8_t *> (raw + offset);

    *saved_spec = spec;
    slots[_S_pool_spec]._M_offset = slab._M_offset;
    slots[_S_pool_spec]._M_bytes = spec_bytes;
    slots[_S_pool_spec]._M_count = 1u;
    constexpr uint32_t window_offset = spec_bytes;
    slots[_S_pool_window]._M_offset = slab._M_offset + window_offset;
    slots[_S_pool_window]._M_bytes
        = align_up (2u * spec.output_elements (), _S_pool_alignment);
    slots[_S_pool_window]._M_count = spec.output_elements ();
    if (pool_is_max (spec._M_kind))
        {
            const uint32_t scratch_offset = window_offset
                                             + slots[_S_pool_window]._M_bytes;
            slots[_S_pool_scratch]._M_offset = slab._M_offset + scratch_offset;
            slots[_S_pool_scratch]._M_bytes = spec.scratch_bytes ();
            slots[_S_pool_scratch]._M_count = spec.scratch_bytes ();
        }
    if (saved_input != nullptr)
        {
            const uint32_t input_offset
                = spec.scratch_bytes () + window_offset
                  + slots[_S_pool_window]._M_bytes;
            slots[_S_pool_input]._M_offset = slab._M_offset + input_offset;
            slots[_S_pool_input]._M_bytes
                = align_up (spec.input_elements (), _S_pool_alignment);
            slots[_S_pool_input]._M_count = spec.input_elements ();
        }

    const Status forward
        = pool_forward (spec, input, output, window_state);
    if (!status_ok (forward))
        return forward;
    if (saved_input != nullptr)
        for (uint32_t i = 0; i < spec.input_elements (); ++i)
            saved_input[i] = input._M_code[i];

    // The node has one operand, so the tape reads the caller's `Trainable`
    // directly; a copy into a `_S_max_operands` array here would only enlarge the
    // frame the forward pass nests inside.
    return tape.record (pool_backward, spec._M_op_id, output, &input, 1u,
                        slots, _S_max_operands);
}
} // namespace zzs::mllee::train
