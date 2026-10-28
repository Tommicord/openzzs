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
 * @brief Affine (dense) layer: Y = act( requant( X W^T ) + b ).
 *
 * The layer is the training twin of the inference LINEAR op and keeps the two
 * properties that make a trained tensor deployable:
 *
 *   - the weight layout is the inference one, one row per output unit, i.e.
 *     @c W[out * in + i], which is exactly what `matrix_codes (w, in, out, true)`
 *     reads as the logical [in, out] multiplicand;
 *   - the forward runs in the int8 code domain through `gemm_codes` and the
 *     fused activation through `_S_apply_act`, so the codes written here are
 *     byte-for-byte the codes Core writes at run time.
 *
 * Shapes are the batched pair: input [batch, in], weight [out, in], bias [out],
 * output [batch, out].  A rank-1 input or output is accepted as an implicit
 * batch of one, because every kernel here reads a flat stride-1 stream.
 *
 * Requantization is a rounded shift plus an optional per-layer Q15 multiplier,
 * mirroring `Layer::_M_requant_mult`:
 *
 *   code = _S_i8_code (round_shift (round_shift (acc, shift), 15) * mult) + bias
 *
 * With `shift = 7` and `mult = 32768` (the defaults) the first term is
 * `round (acc / 127)`, the identity requantization for two unit-normalized int8
 * operands: acc = sum_i x_i w_i over codes, so acc / 127 is the output code of
 * (x/127) . (w/127).  A bias code added afterwards is already in output code
 * units, matching `_S_bias_value` in the inference op.
 *
 * Gradients live in the normalized Q15 domain that `matmul_backward_q15` and the
 * activation kernels already use: a lane is dL/d(value / 127).  The tape seeds
 * the loss lane with its loss scale, so a layer backward reads the gradient it
 * is handed and never rescales it.
 *
 * Pool traffic, all of it measured by `LinearSpec::bytes()`.  With
 * R = batch * out and N = batch * in:
 *
 *   - forward accumulator   : 4 R bytes (int32, carved by the forward);
 *   - saved pre-activation  :   R bytes when the act is not NONE.  The rectifier
 *                               family needs the sign of the pre-activation and
 *                               a post-ReLU zero cannot be told apart from a
 *                               clamped negative, so the rule is broader than
 *                               `act_needs_preactivation` on purpose;
 *   - saved input           :   N bytes when the weight collects a gradient, so
 *                               a later layer may reuse the live buffer;
 *   - backward scratch      : 2 (R + out) bytes (routed lanes plus the channel
 *                               sums), carved by the forward because a backward
 *                               function may not allocate;
 *   - saved spec            :   4 words.
 *
 * A Pool carves at `_S_pool_alignment`, so each of those buffers is 4-byte
 * aligned and stride-1 inside its own region; the only strided walk is the
 * batch-major read of dY by `channel_backward`.
 */
namespace zzs::mllee::train
{
/** @brief Largest multiplier accepted by a linear layer (32.0 in Q15). */
inline constexpr int32_t _S_linear_mult_max_q15 = 1048576;

/** @brief Saved slot holding the input copy. */
inline constexpr uint32_t _S_linear_input = 0u;

/** @brief Saved slot holding the pre-activation copy. */
inline constexpr uint32_t _S_linear_pre = 1u;

/** @brief Saved slot holding the backward scratch lanes. */
inline constexpr uint32_t _S_linear_scratch = 2u;

/** @brief Saved slot holding the spec copy the backward reads back. */
inline constexpr uint32_t _S_linear_spec = 3u;

/**
 * @brief One affine layer: geometry, quantization grid and fused activation.
 *
 * Trivially copyable and saved in the pool, so the backward function recovers
 * the whole configuration from the tape: no capture, no writable static.
 */
struct LinearSpec
{
    uint16_t _M_batch = 1u;
    uint16_t _M_in_features = 0u;
    uint16_t _M_out_features = 0u;
    /** @brief Non-zero when a bias operand participates. */
    uint8_t _M_use_bias = 1u;
    /** @brief Rounded shift applied to the code-domain accumulator. */
    uint8_t _M_requant_shift = 7u;
    Act _M_act = Act::NONE;
    int8_t _M_aux_code = 0;
    int16_t _M_pad = 0;
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

    /** @brief Output element count, R. */
    [[nodiscard]] constexpr uint32_t
    output_elements () const noexcept
    {
        return static_cast<uint32_t> (this->_M_batch)
               * this->_M_out_features;
    }

    /** @brief Input element count, N. */
    [[nodiscard]] constexpr uint32_t
    input_elements () const noexcept
    {
        return static_cast<uint32_t> (this->_M_batch) * this->_M_in_features;
    }

    /** @brief Weight element count. */
    [[nodiscard]] constexpr uint32_t
    weight_elements () const noexcept
    {
        return static_cast<uint32_t> (this->_M_out_features)
               * this->_M_in_features;
    }

    /**
     * @brief Pool bytes one recorded layer needs, per the file header table.
     *
     * This is a capacity bound, not the exact usage of one call: the saved-input
     * term is charged even for a frozen weight (where `linear_run` skips it), so
     * a caller that sizes an arena from `bytes ()` never has to know which
     * layers happen to train.
     */
    [[nodiscard]] constexpr uint32_t
    bytes () const noexcept
    {
        const uint32_t rows = this->output_elements ();
        const uint32_t channels
            = align_up (this->_M_out_features, _S_pool_alignment);
        const uint32_t accumulator = align_up (4u * rows, _S_pool_alignment);
        const uint32_t saved_pre
            = this->has_activation ()
                  ? align_up (rows, _S_pool_alignment)
                  : 0u;
        const uint32_t saved_input
            = align_up (this->input_elements (), _S_pool_alignment);
        const uint32_t scratch
            = align_up (2u * (rows + channels), _S_pool_alignment);
        const uint32_t spec = align_up (static_cast<uint32_t> (sizeof (LinearSpec)),
                                        _S_pool_alignment);
        return accumulator + saved_pre + saved_input + scratch + spec;
    }

    /**
     * @brief Check geometry and quantization fields.
     *
     * @return BAD_LAYOUT for a zero axis or an impossible shift, OUT_OF_RANGE
     *         for a count a training tensor cannot address or a multiplier above
     *         32.0.
     */
    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (this->_M_batch == 0u || this->_M_in_features == 0u
            || this->_M_out_features == 0u)
            return Status::BAD_LAYOUT;
        if (this->_M_requant_shift > 31u)
            return Status::BAD_LAYOUT;
        if (this->_M_requant_mult_q15 < 0
            || this->_M_requant_mult_q15 > _S_linear_mult_max_q15)
            return Status::OUT_OF_RANGE;
        const uint64_t inputs = static_cast<uint64_t> (this->_M_batch)
                                * this->_M_in_features;
        const uint64_t outputs = static_cast<uint64_t> (this->_M_batch)
                                 * this->_M_out_features;
        const uint64_t weights = static_cast<uint64_t> (this->_M_out_features)
                                 * this->_M_in_features;
        if (inputs > _S_max_elements || outputs > _S_max_elements
            || weights > _S_max_elements)
            return Status::OUT_OF_RANGE;
        return Status::OK;
    }

    [[nodiscard]] constexpr Shape
    input_shape () const noexcept
    {
        return Shape::matrix (this->_M_batch, this->_M_in_features);
    }

    [[nodiscard]] constexpr Shape
    weight_shape () const noexcept
    {
        return Shape::matrix (this->_M_out_features, this->_M_in_features);
    }

    [[nodiscard]] constexpr Shape
    bias_shape () const noexcept
    {
        return Shape::vector (this->_M_out_features);
    }

    [[nodiscard]] constexpr Shape
    output_shape () const noexcept
    {
        return Shape::matrix (this->_M_batch, this->_M_out_features);
    }
};

static_assert (sizeof (LinearSpec) == 20u,
               "a saved linear spec is five pool words");

/**
 * @brief Check one operand view against the shape the spec expects.
 *
 * The count must match; the rank may be one lower only when the expected shape
 * is a single row, i.e. the [1, K] and [K] spellings of the same vector.
 */
[[nodiscard]] inline constexpr Status
linear_check_operand (const Trainable &tensor, const Shape &want,
                      const uint32_t elements) noexcept
{
    if (!status_ok (want.validate ()))
        return Status::BAD_LAYOUT;
    if (tensor._M_code == nullptr)
        return Status::BAD_LAYOUT;
    if (tensor._M_count != elements || tensor._M_count != want.elements ())
        return Status::BAD_LAYOUT;
    const uint8_t rank = tensor._M_shape._M_rank;
    if (rank == want._M_rank)
        return tensor._M_shape == want ? Status::OK : Status::BAD_LAYOUT;
    if (rank == 1u && want._M_rank == 2u && want._M_dims[0] == 1u
        && tensor._M_shape._M_dims[0] == want._M_dims[1])
        return Status::OK;
    return Status::BAD_LAYOUT;
}

/**
 * @brief Forward pass with no tape recording.
 *
 * The int32 accumulator is carved from @p pool (the caller resets the pool
 * between passes) and the output gradient lanes are zeroed here, because a
 * forward always precedes the whole backward walk that accumulates into them.
 *
 * @param pool           arena the accumulator is carved from
 * @param spec           layer configuration
 * @param input          [batch, in] code tensor
 * @param weight         [out, in] code tensor, inference row order
 * @param bias           [out] code tensor, unused when the spec has no bias
 * @param output         [batch, out] code tensor, written here
 * @param preactivation  [batch, out] buffer for the pre-activation, required
 *                       when the spec fuses an activation
 */
[[nodiscard]] inline Status
linear_forward (Pool &pool, const LinearSpec &spec, const Trainable &input,
                const Trainable &weight, const Trainable &bias,
                Trainable &output, int8_t *const preactivation) noexcept
{
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    const Status in_status = linear_check_operand (
        input, spec.input_shape (), spec.input_elements ());
    if (!status_ok (in_status))
        return in_status;
    const Status w_status = linear_check_operand (
        weight, spec.weight_shape (), spec.weight_elements ());
    if (!status_ok (w_status))
        return w_status;
    const Status out_status = linear_check_operand (
        output, spec.output_shape (), spec.output_elements ());
    if (!status_ok (out_status))
        return out_status;
    if (spec.has_bias ())
        {
            const Status b_status = linear_check_operand (
                bias, spec.bias_shape (), spec._M_out_features);
            if (!status_ok (b_status))
                return b_status;
        }
    if (spec.has_activation () && preactivation == nullptr)
        return Status::BAD_LAYOUT;

    const uint32_t rows = spec.output_elements ();
    int32_t *accumulator_data = nullptr;
    const Status carve = pool.carve_typed (rows, accumulator_data);
    if (!status_ok (carve))
        return carve;
    const Accum accumulator{ accumulator_data, rows };
    accumulator.zero ();

    // A = X [batch, in]; B = W read as the logical [in, out], i.e. a transposed
    // view with row stride 1 and column stride `in`.
    const Matrix a = matrix_codes (input._M_code, spec._M_batch,
                                   spec._M_in_features, false);
    const Matrix b = matrix_codes (weight._M_code, spec._M_in_features,
                                   spec._M_out_features, true);
    const Status gemm = gemm_codes (a, b, accumulator);
    if (!status_ok (gemm))
        return gemm;

    const uint32_t out_count = spec._M_out_features;
    for (uint32_t row = 0; row < spec._M_batch; ++row)
        for (uint32_t column = 0; column < out_count; ++column)
            {
                const uint32_t index = row * out_count + column;
                int32_t value = _S_round_shift (
                    static_cast<int64_t> (accumulator_data[index]),
                    spec._M_requant_shift);
                if (spec._M_requant_mult_q15 != _S_q15_one)
                    value = _S_round_shift (
                        static_cast<int64_t> (value)
                            * spec._M_requant_mult_q15,
                        _S_grad_frac_bits);
                if (spec.has_bias ())
                    value += bias._M_code[column];
                const int32_t code = _S_i8_code (value);
                if (!spec.has_activation ())
                    {
                        output._M_code[index] = static_cast<int8_t> (code);
                        continue;
                    }
                preactivation[index] = static_cast<int8_t> (code);
                output._M_code[index] = static_cast<int8_t> (
                    act_code (spec._M_act, code, spec._M_aux_code));
            }
    output.zero_grad ();
    return Status::OK;
}

/**
 * @brief Backward of the affine layer, in the `BackwardFn` signature.
 *
 * @p grad_output is the layer output tensor: its gradient lanes carry the
 * upstream signal.  dX reaches the live input operand, dW the weight operand and
 * db the bias operand, all through `Trainable::add_grad`, so a tensor feeding
 * several nodes collects every contribution.
 *
 * Recorded node layout: operands [input, weight, bias]; saved
 * [input copy | null, pre-activation | null, scratch lanes, spec].  A null
 * input-copy handle means the live input buffer is the multiplicand, and the
 * scratch is split into @c R routed lanes followed by the per-channel sums (the
 * second part is 2-byte aligned only, and `channel_backward` writes it
 * elementwise, so no SIMD stream ever touches it).
 */
[[nodiscard]] inline Status
linear_backward (const Trainable &output, const Trainable &grad_output,
                 GradContext &context) noexcept
{
    const LinearSpec *const slot
        = context.saved_as<LinearSpec> (_S_linear_spec);
    if (slot == nullptr)
        return Status::BAD_LAYOUT;
    const LinearSpec spec = *slot;
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    if (context._M_operands == nullptr
        || context._M_operand_count < 2u + (spec.has_bias () ? 1u : 0u))
        return Status::BAD_LAYOUT;
    const Trainable &input = context._M_operands[0];
    const Trainable &weight = context._M_operands[1];
    const Trainable *const bias
        = spec.has_bias () ? context._M_operands + 2 : nullptr;
    if (output._M_count != spec.output_elements ()
        || output._M_shape != spec.output_shape ()
        || input._M_count != spec.input_elements ()
        || weight._M_count != spec.weight_elements ())
        return Status::BAD_LAYOUT;
    if (grad_output._M_grad == nullptr)
        return Status::BAD_LAYOUT;

    const uint32_t rows = spec.output_elements ();
    const uint32_t out_count = spec._M_out_features;
    uint32_t saturated = 0u;
    const int16_t *grad_lanes = grad_output._M_grad;

    // Route dY back through the fused activation before the matrix step.
    if (spec.has_activation ())
        {
            int16_t *const scratch = context.saved_as<int16_t> (_S_linear_scratch);
            int8_t *const pre = context.saved_as<int8_t> (_S_linear_pre);
            if (scratch == nullptr || pre == nullptr)
                return Status::BAD_LAYOUT;
            saturated += act_backward_codes (spec._M_act, pre, grad_lanes,
                                             scratch, rows, spec._M_aux_code);
            grad_lanes = scratch;
        }

    // dX and dW in one pass.  The saved input copy is the multiplicand for dW
    // and the live gradient lanes stay the destination for dX, so a reused or
    // frozen live buffer cannot change either result.
    Trainable x_view = input;
    int8_t *const saved_input
        = context.saved_as<int8_t> (_S_linear_input);
    if (saved_input != nullptr && x_view._M_count == spec.input_elements ())
        x_view._M_code = saved_input;
    Trainable grad_y;
    grad_y._M_code = output._M_code;
    grad_y._M_grad = const_cast<int16_t *> (grad_lanes);
    grad_y._M_shape = spec.output_shape ();
    grad_y._M_count = rows;
    const Status matmul
        = matmul_backward_q15 (x_view, weight, grad_y, spec._M_batch,
                               spec._M_in_features, out_count, saturated);
    if (!status_ok (matmul))
        return matmul;

    // db = sum over the batch axis of dY, accumulated into the bias operand.
    if (bias != nullptr && bias->_M_grad != nullptr)
        {
            int16_t *const scratch = context.saved_as<int16_t> (_S_linear_scratch);
            int16_t *const channel_sum = scratch != nullptr ? scratch + rows
                                                             : nullptr;
            if (channel_sum != nullptr)
                {
                    channel_backward (grad_lanes, spec._M_batch, out_count, 1u,
                                      channel_sum, saturated);
                    for (uint32_t column = 0; column < out_count; ++column)
                        if (bias->add_grad (column, channel_sum[column]))
                            ++saturated;
                }
            else
                {
                    // No scratch room (an activation-less layer with a null
                    // scratch handle): reduce directly, same rounding.
                    for (uint32_t column = 0; column < out_count; ++column)
                        {
                            int64_t total = 0;
                            for (uint32_t row = 0; row < spec._M_batch; ++row)
                                total += grad_lanes[row * out_count + column];
                            const int32_t lane = clamp_sum_q15 (total);
                            if (static_cast<int64_t> (lane) != total)
                                ++saturated;
                            if (bias->add_grad (column, lane))
                                ++saturated;
                        }
                }
        }

    context._M_saturation_events += saturated;
    return Status::OK;
}

/**
 * @brief Allocate the layer's tape state, run the forward and record the node.
 *
 * Four saved tensors -- the maximum a node holds -- and three operands.  The
 * input copy is skipped for a frozen weight and the pre-activation copy for an
 * activation-less layer, so the tape only pays for what the backward reads.  The
 * forward's accumulator is carved from the tape's own pool, so one arena holds
 * the whole layer and `Tape::reset` frees it in one step.
 *
 * @return BAD_LAYOUT for a shape mismatch, ARENA_OVERFLOW when the pool cannot
 *         supply the state, INTERNAL when the tape is not open, OK otherwise.
 */
[[nodiscard]] inline Status
linear_run (Tape &tape, const LinearSpec &spec, const Trainable &input,
            const Trainable &weight, const Trainable &bias,
            Trainable &output) noexcept
{
    Pool &pool = tape.pool ();
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    if (input._M_count != spec.input_elements ()
        || weight._M_count != spec.weight_elements ()
        || output._M_count != spec.output_elements ())
        return Status::BAD_LAYOUT;
    if (spec.has_bias () && bias._M_count != spec._M_out_features)
        return Status::BAD_LAYOUT;

    const uint32_t rows = spec.output_elements ();
    const uint32_t out_count = spec._M_out_features;
    Saved slots[_S_max_operands];
    Trainable operands[_S_max_operands];
    uint32_t operand_count = 0u;
    int8_t *preactivation = nullptr;

    if (weight._M_grad != nullptr)
        {
            const Status save
                = tape.save_copy (input._M_code, spec.input_elements (),
                                  slots[_S_linear_input]);
            if (!status_ok (save))
                return save;
        }
    if (spec.has_activation ())
        {
            const Status save
                = tape.save (rows, slots[_S_linear_pre], preactivation);
            if (!status_ok (save))
                return save;
        }
    {
        // Backward scratch: R routed lanes plus the per-channel sums.
        int16_t *scratch = nullptr;
        const Status save
            = tape.save (rows + out_count, slots[_S_linear_scratch], scratch);
        if (!status_ok (save))
            return save;
    }
    {
        LinearSpec *saved_spec = nullptr;
        const Status save
            = tape.save (1u, slots[_S_linear_spec], saved_spec);
        if (!status_ok (save))
            return save;
        *saved_spec = spec;
    }

    const Status forward
        = linear_forward (pool, spec, input, weight, bias, output,
                          preactivation);
    if (!status_ok (forward))
        return forward;

    operands[operand_count++] = input;
    operands[operand_count++] = weight;
    if (spec.has_bias ())
        operands[operand_count++] = bias;
    for (uint32_t index = operand_count; index < _S_max_operands; ++index)
        slots[index] = Saved{};
    return tape.record (linear_backward, spec._M_op_id, output, operands,
                        operand_count, slots, _S_max_operands);
}
} // namespace zzs::mllee::train
