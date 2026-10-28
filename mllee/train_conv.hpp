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

#include "mllee/format.hpp"
#include "mllee/ops.hpp"
#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_activations.hpp"
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief 1-D and 2-D convolution with stride, padding, dilation and groups.
 *
 * @section conv_im2col Why an im2col buffer rather than a direct convolution
 *
 * This module collects each output window's taps into one caller-owned buffer
 * and runs the product through the shared matrix kernels, instead of nesting a
 * five-deep direct loop.  Three reasons, in the order they bite:
 *
 *   1. one kernel does both directions.  The collection is a matrix whose rows
 *      are output positions and whose columns are (channel, tap0, tap1) triples,
 *      so a conv block *is* a batched matmul: the forward is
 *      `collection x W^T`, and the backward is one `matmul_backward_q15` call
 *      that yields dPatch (the col2im source) and dW together.  The geometry
 *      therefore lives in two index loops that cannot drift apart from the
 *      weight layout, which is the classic source of wrong conv gradients;
 *   2. the arithmetic is the vectorized rank-1 form of `gemm_codes` (contiguous
 *      inner loop, int32 accumulator over codes) and the int64-local form of
 *      `matmul_backward_q15`, rather than a bespoke per-tap accumulator every
 *      assembly target would have to duplicate;
 *   3. the cost is a number the caller can see.  `ConvSpec::bytes()` reports it,
 *      and a model that needs more than the arena gets ARENA_OVERFLOW at the
 *      carve instead of a stack overflow at the first forward.
 *
 * The buffer is per batch item and per group (`rows = out0 * out1` positions,
 * `taps = in_per_group * k0 * k1` columns), never for the whole batch, which
 * keeps it `rows * taps` bytes instead of `batch * rows * taps`.
 *
 * @section conv_layouts Layouts and weight order
 *
 * Input and output are `[batch, C, H, W]` (`ConvLayout::CHANNEL_MAJOR`, the
 * inference order) or `[batch, H, W, C]` (`ConvLayout::SPATIAL_MAJOR`).  Rank 1
 * uses the same flag for `[batch, C, L]` and `[batch, L, C]` and pins the second
 * spatial axis to length 1, kernel 1, stride 1, dilation 1 and zero padding, so
 * one loop nest serves both ranks.  The weight tensor is
 * `[out, in_per_group, k0, k1]` flattened, i.e. the column order inside a group
 * is `(c_local * k0 + tap0) * k1 + tap1` -- the row-per-output-unit order the
 * inference conv ops read, so a trained conv serializes without a shuffle.
 *
 * @section conv_padding Padding without a border read
 *
 * A tap whose source coordinate falls outside `[0, in)` is written as code 0 in
 * the collection and skipped again by col2im, so no index ever leaves the real
 * tensor: the padded border is expressed entirely by the coordinate test, which
 * is why each axis carries two independent signed paddings.  The output length is
 * `floor((in + low + high - dilation * (kernel - 1) - 1) / stride) + 1`; a value
 * below 1 is BAD_LAYOUT, not a clamp.
 *
 * @section conv_numbers Domains
 *
 * Forward: the code domain, like the inference op.  The accumulator holds
 * `sum x w` over codes; `round_shift (acc, 7)` with the default multiplier 32768
 * is `round (acc / 127)`, the identity requantization of two unit-normalized
 * int8 operands, then an optional Q15 multiplier, then the bias code, then the
 * fused activation through `_S_apply_act`.
 * Backward: the Q15 lane domain, like every other module here.  Because a group
 * block is a matmul over `rows`, `in_features = taps` and
 * `out_features = out_per_group`, `matmul_backward_q15` widens the weight codes
 * itself, so dPatch and dW come out as `dL/d(value / 127)` lanes with the tape's
 * loss scale already inside them -- a layer backward must not rescale again.
 *
 * @section conv_pool Pool traffic
 *
 * With `rows = out0 * out1`, `taps = in_per_group * k0 * k1`,
 * `opg = out_per_group`, `total = batch * out_channels * rows` and
 * `n = batch * in_channels * in0 * in1`, the layer carves ONE scratch slab of
 *
 *   align (4 rows*opg)   block accumulator (forward GEMM only)
 * + align (rows*taps)    collection (im2col codes, both directions)
 * + align (2 rows*taps)  dPatch lanes (the dX half of the backward matmul)
 * + align (2 rows*opg)   dY lanes, gathered into the packed [rows, opg] block
 * + align (2 total)      activation routing (unused without a fused act)
 * + align (2 opg)        per-channel bias sums
 *
 * plus the saved input copy (n bytes, only while the weight collects a
 * gradient), the saved pre-activation (total bytes, only with a fused act) and
 * the saved spec.  The collection and the dPatch lanes stay separate regions on
 * purpose: `matmul_backward_q15` reads the codes while it writes the lanes, so
 * aliasing them would let dX overwrite the multiplicand of dW.
 * Every region starts on a 4-byte boundary and is walked stride-1 inside its own
 * loop; the layout gathers and the col2im scatter are the only strided accesses
 * and they are scalar by construction.
 */
namespace zzs::mllee::train
{
/** @brief Largest spatial rank this module supports. */
inline constexpr uint8_t _S_conv_max_rank = 2u;

/** @brief Saved slot holding the spec copy the backward reads back. */
inline constexpr uint32_t _S_conv_spec = 0u;

/** @brief Saved slot holding the input copy. */
inline constexpr uint32_t _S_conv_input = 1u;

/** @brief Saved slot holding the pre-activation copy. */
inline constexpr uint32_t _S_conv_pre = 2u;

/** @brief Saved slot holding the single combined scratch slab. */
inline constexpr uint32_t _S_conv_scratch = 3u;

/** @brief Element order of the spatial axes. */
enum class ConvLayout : uint8_t
{
    /** @brief [batch, channels, h, w] -- the inference order. */
    CHANNEL_MAJOR = 0,
    /** @brief [batch, h, w, channels]. */
    SPATIAL_MAJOR = 1,
};

/**
 * @brief Output length of one axis.
 *
 * @return the length, or 0 when the window cannot be placed even once (a
 *         negative numerator is reported as 0, never wrapped).
 */
[[nodiscard]] inline constexpr int32_t
conv_axis_length (const uint32_t input_length, const uint32_t kernel,
                  const uint32_t stride, const uint32_t dilation,
                  const int32_t pad_low, const int32_t pad_high) noexcept
{
    const int32_t span = static_cast<int32_t> (input_length) + pad_low + pad_high
                         - static_cast<int32_t> (dilation)
                               * (static_cast<int32_t> (kernel) - 1) - 1;
    if (span < 0)
        return 0;
    return span / static_cast<int32_t> (stride) + 1;
}

/**
 * @brief Axis geometry of one convolution.
 *
 * A plain POD: `conv_run` saves it into the pool and the backward reads it back,
 * so the configuration never needs a capture or a writable static.
 */
struct ConvSpec
{
    uint16_t _M_batch = 1u;
    uint16_t _M_in_channels = 0u;
    uint16_t _M_out_channels = 0u;
    /** @brief Spatial extent per axis; axis 1 is 1 for a rank-1 layer. */
    uint16_t _M_in[2] = { 1u, 1u };
    uint8_t _M_groups = 1u;
    uint8_t _M_kernel[2] = { 1u, 1u };
    uint8_t _M_stride[2] = { 1u, 1u };
    uint8_t _M_dilation[2] = { 1u, 1u };
    /** @brief pad[0] / pad[1] bound axis 0, pad[2] / pad[3] bound axis 1. */
    int8_t _M_pad[4] = { 0, 0, 0, 0 };
    uint8_t _M_rank = 1u;
    ConvLayout _M_layout = ConvLayout::CHANNEL_MAJOR;
    Act _M_act = Act::NONE;
    int8_t _M_aux_code = 0;
    uint8_t _M_requant_shift = 7u;
    uint8_t _M_use_bias = 1u;
    /** @brief Q15 multiplier applied after the shift; 32768 is unity. */
    int32_t _M_requant_mult_q15 = _S_q15_one;
    /** @brief Op id recorded on the tape. */
    uint32_t _M_op_id = 0u;

    [[nodiscard]] constexpr bool
    has_bias () const noexcept
    {
        return this->_M_use_bias != 0u;
    }

    [[nodiscard]] constexpr bool
    has_activation () const noexcept
    {
        return this->_M_act != Act::NONE;
    }

    /** @brief Input channels feeding one group. */
    [[nodiscard]] constexpr uint32_t
    in_per_group () const noexcept
    {
        return this->_M_groups != 0u
                   ? static_cast<uint32_t> (this->_M_in_channels) / this->_M_groups
                   : 0u;
    }

    /** @brief Output channels produced by one group. */
    [[nodiscard]] constexpr uint32_t
    out_per_group () const noexcept
    {
        return this->_M_groups != 0u
                   ? static_cast<uint32_t> (this->_M_out_channels) / this->_M_groups
                   : 0u;
    }

    /** @brief Output positions along @p axis (1 for the pinned rank-1 axis). */
    [[nodiscard]] constexpr uint32_t
    out_length (const uint32_t axis) const noexcept
    {
        if (axis >= this->_M_rank)
            return 1u;
        return static_cast<uint32_t> (
            conv_axis_length (this->_M_in[axis], this->_M_kernel[axis],
                              this->_M_stride[axis], this->_M_dilation[axis],
                              this->_M_pad[axis * 2u],
                              this->_M_pad[axis * 2u + 1u]));
    }

    /** @brief Output positions per batch item and group (the collection rows). */
    [[nodiscard]] constexpr uint32_t
    rows () const noexcept
    {
        return this->out_length (0u) * this->out_length (1u);
    }

    /** @brief Collection columns: input channels per group times kernel taps. */
    [[nodiscard]] constexpr uint32_t
    taps () const noexcept
    {
        return this->in_per_group () * this->_M_kernel[0] * this->_M_kernel[1];
    }

    [[nodiscard]] constexpr uint32_t
    output_elements () const noexcept
    {
        return static_cast<uint32_t> (this->_M_batch)
               * this->_M_out_channels * this->rows ();
    }

    [[nodiscard]] constexpr uint32_t
    input_elements () const noexcept
    {
        return static_cast<uint32_t> (this->_M_batch) * this->_M_in_channels
               * this->_M_in[0] * this->_M_in[1];
    }

    [[nodiscard]] constexpr uint32_t
    weight_elements () const noexcept
    {
        return static_cast<uint32_t> (this->_M_out_channels) * this->taps ();
    }

    /** @brief Scratch slab bytes; defined next to `conv_regions` so both agree. */
    [[nodiscard]] constexpr uint32_t
    scratch_bytes () const noexcept;

    /** @brief Pool bytes one recorded layer needs at its worst case. */
    [[nodiscard]] constexpr uint32_t
    bytes () const noexcept;

    [[nodiscard]] constexpr Shape
    input_shape () const noexcept
    {
        if (this->_M_rank == 1u)
            return Shape::tensor3 (this->_M_batch, this->_M_in_channels,
                                   this->_M_in[0]);
        return Shape::tensor4 (this->_M_batch, this->_M_in_channels,
                               this->_M_in[0], this->_M_in[1]);
    }

    [[nodiscard]] constexpr Shape
    weight_shape () const noexcept
    {
        if (this->_M_rank == 1u)
            return Shape::tensor3 (this->_M_out_channels,
                                   static_cast<uint16_t> (this->in_per_group ()),
                                   this->_M_kernel[0]);
        return Shape::tensor4 (this->_M_out_channels,
                               static_cast<uint16_t> (this->in_per_group ()),
                               this->_M_kernel[0], this->_M_kernel[1]);
    }

    [[nodiscard]] constexpr Shape
    output_shape () const noexcept
    {
        if (this->_M_rank == 1u)
            return Shape::tensor3 (this->_M_batch, this->_M_out_channels,
                                   static_cast<uint16_t> (this->out_length (0u)));
        return Shape::tensor4 (this->_M_batch, this->_M_out_channels,
                               static_cast<uint16_t> (this->out_length (0u)),
                               static_cast<uint16_t> (this->out_length (1u)));
    }

    /**
     * @brief Check the geometry and every derived count.
     *
     * @return BAD_LAYOUT for a rank, group, kernel, stride, dilation or padding
     *         combination that cannot describe a convolution -- including one
     *         whose output would be shorter than a single position -- and
     *         OUT_OF_RANGE for a count the pool or a tensor view cannot address.
     */
    [[nodiscard]] Status
    validate () const noexcept
    {
        if (this->_M_rank == 0u || this->_M_rank > _S_conv_max_rank)
            return Status::BAD_LAYOUT;
        if (this->_M_batch == 0u || this->_M_in_channels == 0u
            || this->_M_out_channels == 0u || this->_M_groups == 0u)
            return Status::BAD_LAYOUT;
        if (this->_M_in[0] == 0u || this->_M_in[1] == 0u)
            return Status::BAD_LAYOUT;
        if (this->_M_requant_shift > 31u)
            return Status::BAD_LAYOUT;
        if (this->_M_requant_mult_q15 < 0 || this->_M_requant_mult_q15 > 1048576)
            return Status::OUT_OF_RANGE;
        if (this->_M_in_channels % this->_M_groups != 0u
            || this->_M_out_channels % this->_M_groups != 0u)
            return Status::BAD_LAYOUT;
        if (this->_M_layout != ConvLayout::CHANNEL_MAJOR
            && this->_M_layout != ConvLayout::SPATIAL_MAJOR)
            return Status::BAD_LAYOUT;
        for (uint32_t axis = 0; axis < this->_M_rank; ++axis)
            {
                if (this->_M_kernel[axis] == 0u || this->_M_stride[axis] == 0u
                    || this->_M_dilation[axis] == 0u)
                    return Status::BAD_LAYOUT;
                if (this->out_length (axis) == 0u)
                    return Status::BAD_LAYOUT;
            }
        if (this->_M_rank == 1u
            && (this->_M_in[1] != 1u || this->_M_kernel[1] != 1u
                || this->_M_stride[1] != 1u || this->_M_dilation[1] != 1u
                || this->_M_pad[2] != 0 || this->_M_pad[3] != 0))
            return Status::BAD_LAYOUT;
        const uint64_t rows = static_cast<uint64_t> (this->rows ());
        const uint64_t taps = static_cast<uint64_t> (this->taps ());
        const uint64_t input = static_cast<uint64_t> (this->input_elements ());
        const uint64_t output = static_cast<uint64_t> (this->output_elements ());
        const uint64_t weight = static_cast<uint64_t> (this->weight_elements ());
        if (input > _S_max_elements || output > _S_max_elements
            || weight > _S_max_elements)
            return Status::OUT_OF_RANGE;
        if (rows * taps > _S_max_elements
            || rows * this->_M_out_channels > _S_max_elements)
            return Status::OUT_OF_RANGE;
        return Status::OK;
    }
};

static_assert (sizeof (ConvSpec) == 36u, "conv spec is a compact POD");
static_assert (sizeof (ConvLayout) == 1u, "the layout tag is one byte");

/**
 * @brief The sub-buffers of one scratch slab, as byte offsets from its base.
 *
 * Both directions lay the slab out with this one function, so the forward and
 * the backward cannot disagree about where a region starts.
 */
struct ConvRegions
{
    uint32_t _M_block = 0u;
    uint32_t _M_collection = 0u;
    uint32_t _M_patch = 0u;
    uint32_t _M_grad = 0u;
    uint32_t _M_routed = 0u;
    uint32_t _M_bias = 0u;
    uint32_t _M_bytes = 0u;
};

static_assert (sizeof (ConvRegions) == 28u, "six offsets and a length");

/** @brief Lay the scratch slab out for @p spec. */
[[nodiscard]] inline constexpr ConvRegions
conv_regions (const ConvSpec &spec) noexcept
{
    const uint32_t rows = spec.rows ();
    const uint32_t taps = spec.taps ();
    const uint32_t opg = spec.out_per_group ();
    ConvRegions region;
    region._M_block = 0u;
    region._M_collection
        = region._M_block + align_up (4u * rows * opg, _S_pool_alignment);
    region._M_patch
        = region._M_collection + align_up (rows * taps, _S_pool_alignment);
    region._M_grad = region._M_patch
                     + align_up (2u * rows * taps, _S_pool_alignment);
    region._M_routed
        = region._M_grad + align_up (2u * rows * opg, _S_pool_alignment);
    region._M_bias
        = region._M_routed
          + align_up (2u * spec.output_elements (), _S_pool_alignment);
    region._M_bytes
        = region._M_bias + align_up (2u * opg, _S_pool_alignment);
    return region;
}

[[nodiscard]] inline constexpr uint32_t
ConvSpec::scratch_bytes () const noexcept
{
    return conv_regions (*this)._M_bytes;
}

[[nodiscard]] inline constexpr uint32_t
ConvSpec::bytes () const noexcept
{
    return this->scratch_bytes ()
           + align_up (this->input_elements (), _S_pool_alignment)
           + align_up (this->output_elements (), _S_pool_alignment)
           + align_up (static_cast<uint32_t> (sizeof (ConvSpec)),
                       _S_pool_alignment);
}

/** @brief Element strides of one tensor under a layout, in elements. */
struct ConvStrides
{
    uint32_t _M_batch = 0u;
    uint32_t _M_channel = 0u;
    uint32_t _M_axis0 = 0u;
    uint32_t _M_axis1 = 0u;
};

static_assert (sizeof (ConvStrides) == 16u, "four stride words");

/** @brief Strides of the input tensor. */
[[nodiscard]] inline constexpr ConvStrides
conv_input_strides (const ConvSpec &spec) noexcept
{
    if (spec._M_layout == ConvLayout::CHANNEL_MAJOR)
        return ConvStrides{ static_cast<uint32_t> (spec._M_in_channels)
                                * spec._M_in[0] * spec._M_in[1],
                            static_cast<uint32_t> (spec._M_in[0]) * spec._M_in[1],
                            spec._M_in[1], 1u };
    return ConvStrides{ static_cast<uint32_t> (spec._M_in[0]) * spec._M_in[1]
                            * spec._M_in_channels,
                        1u,
                        static_cast<uint32_t> (spec._M_in[1])
                            * spec._M_in_channels,
                        spec._M_in_channels };
}

/** @brief Strides of the output tensor. */
[[nodiscard]] inline constexpr ConvStrides
conv_output_strides (const ConvSpec &spec) noexcept
{
    const uint32_t rows = spec.rows ();
    if (spec._M_layout == ConvLayout::CHANNEL_MAJOR)
        return ConvStrides{ static_cast<uint32_t> (spec._M_out_channels) * rows,
                            rows, spec.out_length (1u), 1u };
    return ConvStrides{ rows * spec._M_out_channels, 1u,
                        static_cast<uint32_t> (spec.out_length (1u))
                            * spec._M_out_channels,
                        spec._M_out_channels };
}

/** @brief Flat element of @p strides at (@p outer, @p channel, @p a0, @p a1). */
[[nodiscard]] inline constexpr uint32_t
conv_element (const ConvStrides &strides, const uint32_t outer,
              const uint32_t channel, const uint32_t a0,
              const uint32_t a1) noexcept
{
    return outer * strides._M_batch + channel * strides._M_channel
           + a0 * strides._M_axis0 + a1 * strides._M_axis1;
}

/**
 * @brief Bound a tensor against the count and rank @p spec expects.
 *
 * A view whose shape was never filled in (rank 0) is accepted: kernels here read
 * flat counts and the layout strides, never the operand shape.
 */
[[nodiscard]] inline constexpr Status
conv_check_operand (const Trainable &tensor, const uint32_t elements,
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
 * @brief Forward pass with no tape recording.
 *
 * @p scratch must hold `ConvSpec::scratch_bytes ()` bytes on a 4-byte boundary;
 * a caller without a tape can carve it straight from a Pool, and `conv_run`
 * saves it as the node's single scratch handle so the backward can reuse it.
 *
 * @param preactivation  null, or `output_elements ()` codes for the
 *                       pre-activation (required when the act is not NONE)
 */
[[nodiscard]] inline Status
conv_forward (const ConvSpec &spec, const Trainable &input,
              const Trainable &weight, const Trainable &bias, Trainable &output,
              int8_t *const preactivation, int8_t *const scratch) noexcept
{
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    const uint8_t tensor_rank = spec._M_rank == 1u ? 3u : 4u;
    if (!status_ok (conv_check_operand (input, spec.input_elements (),
                                        tensor_rank)))
        return Status::BAD_LAYOUT;
    if (!status_ok (conv_check_operand (weight, spec.weight_elements (),
                                        tensor_rank)))
        return Status::BAD_LAYOUT;
    if (!status_ok (conv_check_operand (output, spec.output_elements (),
                                        tensor_rank)))
        return Status::BAD_LAYOUT;
    if (spec.has_bias ()
        && !status_ok (conv_check_operand (bias, spec._M_out_channels, 1u)))
        return Status::BAD_LAYOUT;
    if (spec.has_activation () && preactivation == nullptr)
        return Status::BAD_LAYOUT;
    if (scratch == nullptr || !simd_aligned (scratch))
        return Status::BAD_LAYOUT;

    const ConvRegions region = conv_regions (spec);
    const uint32_t rows = spec.rows ();
    const uint32_t taps = spec.taps ();
    const uint32_t opg = spec.out_per_group ();
    const uint32_t in_pg = spec.in_per_group ();
    const uint32_t out1 = spec.out_length (1u);
    int32_t *const block = reinterpret_cast<int32_t *> (scratch + region._M_block);
    int8_t *const collection = scratch + region._M_collection;
    const ConvStrides in_stride = conv_input_strides (spec);
    const ConvStrides out_stride = conv_output_strides (spec);
    const bool channel_major = spec._M_layout == ConvLayout::CHANNEL_MAJOR;

    for (uint32_t item = 0; item < spec._M_batch; ++item)
        for (uint32_t group = 0; group < spec._M_groups; ++group)
            {
                // im2col, one row per output position and one column per
                // (channel, tap0, tap1) triple in the weight's own order.  A tap
                // outside the real border writes code 0 and is never read.
                for (uint32_t row = 0; row < rows; ++row)
                    {
                        const uint32_t a0 = out1 != 0u ? row / out1 : 0u;
                        const uint32_t a1 = row % out1;
                        int8_t *const destination = collection + row * taps;
                        uint32_t column = 0u;
                        for (uint32_t channel = 0; channel < in_pg; ++channel)
                            for (uint32_t tap0 = 0; tap0 < spec._M_kernel[0];
                                 ++tap0)
                                {
                                    const int32_t source0 = static_cast<int32_t> (
                                                a0 * spec._M_stride[0])
                                            + static_cast<int32_t> (
                                                tap0 * spec._M_dilation[0])
                                            - spec._M_pad[0];
                                    const bool inside0
                                        = source0 >= 0
                                          && static_cast<uint32_t> (source0)
                                                 < spec._M_in[0];
                                    for (uint32_t tap1 = 0;
                                         tap1 < spec._M_kernel[1]; ++tap1)
                                        {
                                            const int32_t source1 = static_cast<int32_t> (
                                                        a1 * spec._M_stride[1])
                                                    + static_cast<int32_t> (
                                                          tap1 * spec._M_dilation[1])
                                                    - spec._M_pad[2];
                                            const bool inside1
                                                = source1 >= 0
                                                  && static_cast<uint32_t> (source1)
                                                         < spec._M_in[1];
                                            destination[column]
                                                = inside0 && inside1
                                                      ? input._M_code[conv_element (
                                                            in_stride, item,
                                                            group * in_pg + channel,
                                                            static_cast<uint32_t> (
                                                                source0),
                                                            static_cast<uint32_t> (
                                                                source1))]
                                                      : 0;
                                            ++column;
                                        }
                                }
                    }

                const int8_t *const weight_block
                    = weight._M_code + group * opg * taps;
                const Accum accumulator{ block, rows * opg };
                accumulator.zero ();
                const Status gemm = channel_major
                                        ? gemm_codes (
                                              matrix_codes (weight_block, opg,
                                                            taps, false),
                                              matrix_codes (collection, rows, taps,
                                                            true),
                                              accumulator)
                                        : gemm_codes (
                                              matrix_codes (collection, rows, taps,
                                                            false),
                                              matrix_codes (weight_block, opg, taps,
                                                            true),
                                              accumulator);
                if (!status_ok (gemm))
                    return gemm;

                for (uint32_t row = 0; row < rows; ++row)
                    {
                        const uint32_t a0 = out1 != 0u ? row / out1 : 0u;
                        const uint32_t a1 = row % out1;
                        for (uint32_t local = 0; local < opg; ++local)
                            {
                                const uint32_t channel = group * opg + local;
                                const uint32_t source
                                    = channel_major ? local * rows + row
                                                    : row * opg + local;
                                int32_t value = _S_round_shift (
                                    static_cast<int64_t> (block[source]),
                                    spec._M_requant_shift);
                                if (spec._M_requant_mult_q15 != _S_q15_one)
                                    value = _S_round_shift (
                                        static_cast<int64_t> (value)
                                            * spec._M_requant_mult_q15,
                                        _S_grad_frac_bits);
                                if (spec.has_bias ())
                                    value += bias._M_code[channel];
                                const uint32_t target
                                    = conv_element (out_stride, item, channel, a0,
                                                    a1);
                                const int32_t code = _S_i8_code (value);
                                if (!spec.has_activation ())
                                    {
                                        output._M_code[target]
                                            = static_cast<int8_t> (code);
                                        continue;
                                    }
                                preactivation[target] = static_cast<int8_t> (code);
                                output._M_code[target] = static_cast<int8_t> (
                                    act_code (spec._M_act, code, spec._M_aux_code));
                            }
                    }
            }
    output.zero_grad ();
    return Status::OK;
}

/**
 * @brief Backward pass: dX by col2im, dW by the transposed collection and db by
 *        a channel sum, in the `BackwardFn` signature.
 *
 * Recorded node layout: operands [input, weight, bias]; saved
 * [spec, input copy | null, pre-activation | null, scratch].  Each (item, group)
 * block is differentiated as the matmul it is -- `rows` by `taps` against
 * `out_per_group` -- so one `matmul_backward_q15` call produces both dPatch (into
 * the collection region, widened to lanes) and dW (straight into the weight's own
 * gradient lanes at the group offset).  A frozen operand simply has no lane
 * buffer, and the kernel skips that half.
 */
[[nodiscard]] inline Status
conv_backward (const Trainable &output, const Trainable &grad_output,
               GradContext &context) noexcept
{
    const ConvSpec *const slot = context.saved_as<ConvSpec> (_S_conv_spec);
    if (slot == nullptr)
        return Status::BAD_LAYOUT;
    const ConvSpec spec = *slot;
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    if (context._M_operands == nullptr || context._M_operand_count < 2u)
        return Status::BAD_LAYOUT;
    const Trainable &input = context._M_operands[0];
    const Trainable &weight = context._M_operands[1];
    const Trainable *const bias
        = spec.has_bias () && context._M_operand_count > 2u
              ? context._M_operands + 2
              : nullptr;
    if (output._M_count != spec.output_elements ()
        || input._M_count != spec.input_elements ()
        || weight._M_count != spec.weight_elements ())
        return Status::BAD_LAYOUT;
    if (grad_output._M_grad == nullptr)
        return Status::BAD_LAYOUT;

    int8_t *const scratch = context.saved_as<int8_t> (_S_conv_scratch);
    if (scratch == nullptr || !simd_aligned (scratch))
        return Status::BAD_LAYOUT;

    const ConvRegions region = conv_regions (spec);
    const uint32_t rows = spec.rows ();
    const uint32_t taps = spec.taps ();
    const uint32_t opg = spec.out_per_group ();
    const uint32_t in_pg = spec.in_per_group ();
    const uint32_t total = spec.output_elements ();
    const uint32_t out1 = spec.out_length (1u);
    int8_t *const collection = scratch + region._M_collection;
    int16_t *const patch_lanes
        = reinterpret_cast<int16_t *> (scratch + region._M_patch);
    int16_t *const grad_lanes
        = reinterpret_cast<int16_t *> (scratch + region._M_grad);
    int16_t *const routed
        = reinterpret_cast<int16_t *> (scratch + region._M_routed);
    int16_t *const bias_sum
        = reinterpret_cast<int16_t *> (scratch + region._M_bias);
    uint32_t saturated = 0u;

    // Route dY through the fused activation first, flat over the output tensor,
    // so the gather below reads one lane stream whichever layout is in use.
    const int16_t *source_grad = grad_output._M_grad;
    if (spec.has_activation ())
        {
            int8_t *const pre = context.saved_as<int8_t> (_S_conv_pre);
            if (pre == nullptr)
                return Status::BAD_LAYOUT;
            saturated += act_backward_codes (spec._M_act, pre, source_grad,
                                            routed, total, spec._M_aux_code);
            source_grad = routed;
        }

    const ConvStrides in_stride = conv_input_strides (spec);
    const ConvStrides out_stride = conv_output_strides (spec);
    int8_t *const saved_input = context.saved_as<int8_t> (_S_conv_input);
    const int8_t *const input_codes
        = saved_input != nullptr ? saved_input : input._M_code;
    const bool collect_weight_grad = weight._M_grad != nullptr;
    const bool collect_input_grad = input._M_grad != nullptr;

    // The block view of dY: a [rows, opg] lane matrix, exactly the grad_y the
    // matmul kernel expects for batch = rows, out_features = opg.
    Trainable grad_y;
    grad_y._M_code = output._M_code;
    grad_y._M_grad = grad_lanes;
    grad_y._M_shape = Shape::matrix (static_cast<uint16_t> (rows),
                                     static_cast<uint16_t> (opg));
    grad_y._M_count = rows * opg;

    for (uint32_t item = 0; item < spec._M_batch; ++item)
        for (uint32_t group = 0; group < spec._M_groups; ++group)
            {
                for (uint32_t row = 0; row < rows; ++row)
                    {
                        const uint32_t a0 = out1 != 0u ? row / out1 : 0u;
                        const uint32_t a1 = row % out1;
                        for (uint32_t local = 0; local < opg; ++local)
                            grad_lanes[row * opg + local]
                                = source_grad[conv_element (
                                      out_stride, item, group * opg + local, a0,
                                      a1)];
                    }

                // Rebuild the collection from the saved codes in the weight's own
                // column order; `matmul_backward_q15` widens it for dW and writes
                // dPatch as lanes in the separate region next to it.
                uint32_t column = 0u;
                for (uint32_t channel = 0; channel < in_pg; ++channel)
                    for (uint32_t tap0 = 0; tap0 < spec._M_kernel[0]; ++tap0)
                        {
                            const int32_t base0 = static_cast<int32_t> (
                                                      tap0 * spec._M_dilation[0])
                                                  - spec._M_pad[0];
                            for (uint32_t tap1 = 0; tap1 < spec._M_kernel[1];
                                 ++tap1)
                                {
                                    const int32_t base1 = static_cast<int32_t> (
                                                      tap1 * spec._M_dilation[1])
                                                          - spec._M_pad[2];
                                    for (uint32_t row = 0; row < rows; ++row)
                                        {
                                            const uint32_t a0
                                                = out1 != 0u ? row / out1 : 0u;
                                            const uint32_t a1 = row % out1;
                                            const int32_t source0 = static_cast<int32_t> (
                                                        a0 * spec._M_stride[0])
                                                    + base0;
                                            const int32_t source1 = static_cast<int32_t> (
                                                        a1 * spec._M_stride[1])
                                                    + base1;
                                            const bool inside
                                                = source0 >= 0
                                                  && static_cast<uint32_t> (source0)
                                                         < spec._M_in[0]
                                                  && source1 >= 0
                                                  && static_cast<uint32_t> (source1)
                                                         < spec._M_in[1];
                                            collection[row * taps + column]
                                                = inside
                                                      ? input_codes[conv_element (
                                                            in_stride, item,
                                                            group * in_pg + channel,
                                                            static_cast<uint32_t> (
                                                                source0),
                                                            static_cast<uint32_t> (
                                                                source1))]
                                                      : 0;
                                        }
                                    ++column;
                                }
                        }

                Trainable x_block;
                x_block._M_code = collection;
                x_block._M_grad = collect_input_grad ? patch_lanes : nullptr;
                x_block._M_shape = Shape::matrix (
                    static_cast<uint16_t> (rows),
                    static_cast<uint16_t> (taps));
                x_block._M_count = rows * taps;

                int8_t *const weight_block
                    = weight._M_code + group * opg * taps;
                Trainable w_block;
                w_block._M_code = weight_block;
                w_block._M_grad = collect_weight_grad
                                      ? weight._M_grad + group * opg * taps
                                      : nullptr;
                w_block._M_shape = Shape::matrix (static_cast<uint16_t> (opg),
                                                  static_cast<uint16_t> (taps));
                w_block._M_count = opg * taps;

                if (collect_input_grad)
                    fill_lanes (patch_lanes, 0, rows * taps);
                const Status matmul = matmul_backward_q15 (
                    x_block, w_block, grad_y, rows, taps, opg, saturated);
                if (!status_ok (matmul))
                    return matmul;

                // col2im: add each patch gradient back into the element it read.
                if (collect_input_grad)
                    {
                        column = 0u;
                        for (uint32_t channel = 0; channel < in_pg; ++channel)
                            for (uint32_t tap0 = 0; tap0 < spec._M_kernel[0];
                                 ++tap0)
                                {
                                    const int32_t base0 = static_cast<int32_t> (
                                                  tap0 * spec._M_dilation[0])
                                              - spec._M_pad[0];
                                    for (uint32_t tap1 = 0;
                                         tap1 < spec._M_kernel[1]; ++tap1)
                                        {
                                            const int32_t base1 = static_cast<int32_t> (
                                                        tap1 * spec._M_dilation[1])
                                                    - spec._M_pad[2];
                                            for (uint32_t row = 0; row < rows;
                                                 ++row)
                                                {
                                                    const uint32_t a0 = out1 != 0u
                                                                            ? row / out1
                                                                            : 0u;
                                                    const uint32_t a1 = row % out1;
                                                    const int32_t source0 = static_cast<int32_t> (
                                                                a0
                                                                * spec._M_stride[0])
                                                            + base0;
                                                    const int32_t source1 = static_cast<int32_t> (
                                                                a1
                                                                * spec._M_stride[1])
                                                            + base1;
                                                    if (source0 >= 0
                                                        && static_cast<uint32_t> (
                                                               source0)
                                                               < spec._M_in[0]
                                                        && source1 >= 0
                                                        && static_cast<uint32_t> (
                                                               source1)
                                                               < spec._M_in[1])
                                                        {
                                                            const uint32_t target = conv_element (
                                                                in_stride, item,
                                                                group * in_pg
                                                                    + channel,
                                                                static_cast<uint32_t> (
                                                                    source0),
                                                                static_cast<uint32_t> (
                                                                    source1));
                                                            if (input.add_grad (
                                                                    target,
                                                                    patch_lanes[row
                                                                                * taps
                                                                            + column]))
                                                                ++saturated;
                                                        }
                                                }
                                            ++column;
                                        }
                                }
                    }

                // db = sum of dY over every output position of this group.
                if (bias != nullptr && bias->_M_grad != nullptr)
                    {
                        channel_backward (grad_lanes, rows, opg, 1u, bias_sum,
                                          saturated);
                        for (uint32_t local = 0; local < opg; ++local)
                            if (bias->add_grad (group * opg + local,
                                                bias_sum[local]))
                                ++saturated;
                    }
            }

    context._M_saturation_events += saturated;
    return Status::OK;
}

/**
 * @brief Save the layer's four tensors, run the forward and record the node.
 *
 * The scratch is carved by the forward, because a `BackwardFn` may not allocate;
 * the input copy exists only while the weight collects a gradient and the
 * pre-activation copy only with a fused activation, so a frozen or
 * activation-less layer pays no activation RAM.
 *
 * @return BAD_LAYOUT for a shape mismatch, ARENA_OVERFLOW when the pool cannot
 *         supply the state, INTERNAL when the tape is not open, OK otherwise.
 */
[[nodiscard]] inline Status
conv_run (Tape &tape, const ConvSpec &spec, const Trainable &input,
          const Trainable &weight, const Trainable &bias,
          Trainable &output) noexcept
{
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    if (input._M_count != spec.input_elements ()
        || weight._M_count != spec.weight_elements ()
        || output._M_count != spec.output_elements ())
        return Status::BAD_LAYOUT;
    if (spec.has_bias () && bias._M_count != spec._M_out_channels)
        return Status::BAD_LAYOUT;

    Saved slots[_S_max_operands];
    Trainable operands[_S_max_operands];
    uint32_t operand_count = 0u;

    {
        ConvSpec *saved_spec = nullptr;
        const Status save = tape.save (1u, slots[_S_conv_spec], saved_spec);
        if (!status_ok (save))
            return save;
        *saved_spec = spec;
    }
    if (weight._M_grad != nullptr)
        {
            const Status save = tape.save_copy (input._M_code,
                                                spec.input_elements (),
                                                slots[_S_conv_input]);
            if (!status_ok (save))
                return save;
        }
    int8_t *preactivation = nullptr;
    if (spec.has_activation ())
        {
            const Status save = tape.save (spec.output_elements (),
                                           slots[_S_conv_pre], preactivation);
            if (!status_ok (save))
                return save;
        }
    int8_t *scratch = nullptr;
    {
        const Status save = tape.save (spec.scratch_bytes (),
                                       slots[_S_conv_scratch], scratch);
        if (!status_ok (save))
            return save;
    }

    const Status forward = conv_forward (spec, input, weight, bias, output,
                                         preactivation, scratch);
    if (!status_ok (forward))
        return forward;

    operands[operand_count++] = input;
    operands[operand_count++] = weight;
    if (spec.has_bias ())
        operands[operand_count++] = bias;
    for (uint32_t index = operand_count; index < _S_max_operands; ++index)
        slots[index] = Saved{};
    return tape.record (conv_backward, spec._M_op_id, output, operands,
                        operand_count, slots, _S_max_operands);
}
} // namespace zzs::mllee::train
