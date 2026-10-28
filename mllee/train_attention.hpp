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
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Training twin of the inference ATTENTION op: multi-head scaled
 *        dot-product attention with an optional causal mask and RoPE.
 *
 * Query, key and value are three separate [seq, heads * head_dim] tensors, not
 * the packed QKV buffer the inference op reads.  Deployment is a plain
 * three-block concatenation (Q block, K block, V block), which is exactly what
 * Core's `_S_attention` addresses; keeping them separate in the graph lets the
 * three projection layers own their own gradients.
 *
 * The forward is byte-identical to inference on the same input.  It reuses the
 * inference score helper (`_S_attention_score`), the same 33-entry quarter-sine
 * table and Q16 step table for RoPE, the same `_S_Div32` reciprocal for the
 * score scale, and the same 256-entry exp table.  The output is therefore the
 * integer division `(sum weight * value + denominator / 2) / denominator`,
 * narrowed once.  Exact parity is worth the extra table lookup: it removes the
 * train/deploy skew a softmax re-implementation would introduce.
 *
 * The backward cannot use the table as a derivative -- it is a step function --
 * so it takes the continuous surrogate the softmax module already owns.  For
 * one query row, with `p_j` the normalized table weight and `dO` the upstream
 * lane of the output row:
 *
 *   dV_j     += p_j * dO
 *   dP_j      = dO . V_j
 *   ds_j      = p_j * (dP_j - sum_k p_k dP_k)          (softmax Jacobian)
 *   dscore_j  = ds_j / sqrt(head_dim)
 *   dQ       += sum_j dscore_j * K_j
 *   dK_j     += dscore_j * Q
 *
 * `p_j` is recomputed from the forward's own integer weights (two passes over
 * the key axis, exactly like the inference kernel), so the backward needs only
 * two `seq`-lane scratch rows -- never a seq x seq score matrix.  Under RoPE the
 * score dot is rotated, and the query/key gradients are rotated back by the
 * transpose of the same rotation before they reach the operand.
 *
 * Gradient lanes are dL/d(real) in Q15 (real = code / 127), matching every
 * other module.  `AttentionSpec::bytes ()` is the exact pool cost: two lane
 * rows and the saved spec.
 */
namespace zzs::mllee::train
{
/** @brief Saved slot holding the spec copy the backward reads back. */
inline constexpr uint32_t _S_attention_spec = 0u;

/** @brief Saved slot holding the two backward lane rows. */
inline constexpr uint32_t _S_attention_scratch = 1u;

/** @brief A rotated row is at most eight int16 lanes in this opset. */
inline constexpr uint32_t _S_attention_head_limit = 8u;

/** @brief Fraction width of the head-dimension square root used by backward. */
inline constexpr uint32_t _S_attention_scale_bits = 12u;

/**
 * @brief Multi-head attention geometry and deployment flags.
 *
 * `heads * head_dim` is one token's width, so the token shape is
 * `[seq, heads * head_dim]` for every operand.  The head axis is the minor one,
 * matching the inference packing.
 */
struct AttentionSpec
{
    uint16_t _M_seq = 0u;
    uint16_t _M_heads = 1u;
    uint8_t _M_head_dim = 0u;
    /** @brief Non-zero limits key j <= query i. */
    uint8_t _M_causal = 0u;
    /** @brief Non-zero rotates query and key rows with RoPE. */
    uint8_t _M_rope = 0u;
    uint8_t _M_pad = 0u;
    /** @brief Op id recorded on the tape. */
    uint32_t _M_op_id = 0u;

    [[nodiscard]] constexpr uint32_t
    stride () const noexcept
    {
        return static_cast<uint32_t> (this->_M_heads) * this->_M_head_dim;
    }

    [[nodiscard]] constexpr uint32_t
    elements () const noexcept
    {
        return static_cast<uint32_t> (this->_M_seq) * this->stride ();
    }

    [[nodiscard]] constexpr Shape
    token_shape () const noexcept
    {
        return Shape::matrix (this->_M_seq,
                              static_cast<uint16_t> (this->stride ()));
    }

    [[nodiscard]] constexpr bool
    has_rope () const noexcept
    {
        return this->_M_rope != 0u;
    }

    [[nodiscard]] constexpr bool
    has_causal () const noexcept
    {
        return this->_M_causal != 0u;
    }

    /** @brief Pool bytes one recorded layer needs. */
    [[nodiscard]] constexpr uint32_t
    bytes () const noexcept
    {
        return align_up (4u * static_cast<uint32_t> (this->_M_seq),
                         _S_pool_alignment)
               + align_up (static_cast<uint32_t> (sizeof (AttentionSpec)),
                           _S_pool_alignment);
    }

    /**
     * @brief Check the geometry and the flags.
     *
     * @return BAD_LAYOUT for a zero axis, a head dimension above eight, an odd
     *         RoPE head dimension or a flag byte outside {0, 1}; OUT_OF_RANGE
     *         when the token grid exceeds a training view.
     */
    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (this->_M_seq == 0u || this->_M_heads == 0u || this->_M_head_dim == 0u)
            return Status::BAD_LAYOUT;
        if (this->_M_head_dim > _S_attention_head_limit)
            return Status::BAD_LAYOUT;
        if (this->_M_causal > 1u || this->_M_rope > 1u)
            return Status::BAD_LAYOUT;
        if (this->has_rope () && (this->_M_head_dim & 1u) != 0u)
            return Status::BAD_LAYOUT;
        if (static_cast<uint64_t> (this->_M_seq) * this->stride ()
            > _S_max_elements)
            return Status::OUT_OF_RANGE;
        return Status::OK;
    }
};

static_assert (sizeof (AttentionSpec) == 12u,
               "a saved attention spec is three pool words");

/** @brief Check a token tensor against the spec's shape. */
[[nodiscard]] inline constexpr Status
attention_check_operand (const Trainable &tensor,
                         const AttentionSpec &spec) noexcept
{
    if (tensor._M_code == nullptr)
        return Status::BAD_LAYOUT;
    if (tensor._M_count != spec.elements ()
        || tensor._M_shape != spec.token_shape ())
        return Status::BAD_LAYOUT;
    return Status::OK;
}

/** @brief sqrt(head_dim) in Q12: 4096 at head_dim 1, 11585 at head_dim 8. */
[[nodiscard]] inline constexpr int32_t
attention_scale_q12 (const uint32_t head_dim) noexcept
{
    return static_cast<int32_t> (
        _S_isqrt (head_dim << (2u * _S_attention_scale_bits)));
}

/** @brief Divide a Q15 lane by sqrt(head_dim) using the Q12 root. */
[[nodiscard]] inline constexpr int32_t
attention_div_scale (const int32_t value_q15,
                     const int32_t scale_q12) noexcept
{
    if (scale_q12 <= 0)
        return 0;
    const int64_t scaled
        = (static_cast<int64_t> (value_q15) << _S_attention_scale_bits)
          / scale_q12;
    return q15_clamp (static_cast<int32_t> (
        scaled > _S_q15_max ? _S_q15_max
                            : (scaled < _S_q15_min ? _S_q15_min : scaled)));
}

/**
 * @brief Rotate a Q15 gradient row back through the RoPE transpose.
 *
 * The forward pair is `(x c - y s, x s + y c)`, so the transpose is
 * `(gx c + gy s, -gx s + gy c)`.  @p rotated holds Q15 lanes, the output is
 * int32 Q15 (the sum of two rotated components can exceed 1.0 before it is
 * finally saturated by `Trainable::add_grad`).
 */
inline void
attention_unrotate (const int32_t *const ZZS_TRAIN_RESTRICT rotated,
                    const uint32_t position, const uint32_t pairs,
                    int32_t *const ZZS_TRAIN_RESTRICT plain) noexcept
{
    const uint32_t row = (pairs * (pairs - 1u)) >> 1;
    for (uint32_t pair = 0; pair < pairs; ++pair)
        {
            const int32_t step = _S_rope_step_q16[row + pair];
            const uint32_t angle
                = (static_cast<uint32_t> (
                       static_cast<int32_t> (position) * step)
                   >> 16)
                  & 127u;
            const int32_t cosine = _S_sin_unit (angle + 32u);
            const int32_t sine = _S_sin_unit (angle);
            const int64_t x = rotated[2u * pair];
            const int64_t y = rotated[2u * pair + 1u];
            plain[2u * pair] = _S_round_shift (x * cosine + y * sine, 15u);
            plain[2u * pair + 1u]
                = _S_round_shift (-x * sine + y * cosine, 15u);
        }
}

/** @brief One Q/K row in Q15: the RoPE rotation of the codes, or the codes. */
inline void
attention_row_q15 (const int8_t *const ZZS_TRAIN_RESTRICT row,
                   const uint32_t position, const uint32_t head_dim,
                   const bool rope,
                   int32_t *const ZZS_TRAIN_RESTRICT lanes) noexcept
{
    if (!rope)
        {
            for (uint32_t d = 0; d < head_dim; ++d)
                lanes[d] = code_to_q15 (row[d]);
            return;
        }
    int16_t rotated[_S_attention_head_limit];
    _S_rope_rotate_row (row, position, rotated, head_dim / 2u);
    for (uint32_t d = 0; d < head_dim; ++d)
        lanes[d] = code_to_q15 (rotated[d]);
}

/**
 * @brief Forward pass with no tape recording.
 *
 * The output gradient lanes are zeroed here, because a forward always precedes
 * the backward walk that accumulates into them.
 *
 * @return BAD_LAYOUT for a shape mismatch, OK otherwise.
 */
[[nodiscard]] inline Status
attention_forward (const AttentionSpec &spec, const Trainable &query,
                   const Trainable &key, const Trainable &value,
                   Trainable &output) noexcept
{
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    Status operand = attention_check_operand (query, spec);
    if (!status_ok (operand))
        return operand;
    operand = attention_check_operand (key, spec);
    if (!status_ok (operand))
        return operand;
    operand = attention_check_operand (value, spec);
    if (!status_ok (operand))
        return operand;
    operand = attention_check_operand (output, spec);
    if (!status_ok (operand))
        return operand;

    const uint32_t sequence = spec._M_seq;
    const uint32_t head_dim = spec._M_head_dim;
    const uint32_t heads = spec._M_heads;
    const uint32_t stride = spec.stride ();
    const bool causal = spec.has_causal ();
    const bool rope = spec.has_rope ();
    const uint32_t pairs = head_dim / 2u;
    const uint32_t score_scale = _S_isqrt (head_dim << 16);
    const _S_Div32 score_div = _S_div32_make (score_scale);

    for (uint32_t head = 0; head < heads; ++head)
        {
            const uint32_t offset = head * head_dim;
            for (uint32_t query_pos = 0; query_pos < sequence; ++query_pos)
                {
                    const int8_t *const q_row
                        = query._M_code + query_pos * stride + offset;
                    int16_t rotated_query[_S_attention_head_limit] = {
                        0, 0, 0, 0, 0, 0, 0, 0
                    };
                    if (rope)
                        _S_rope_rotate_row (q_row, query_pos, rotated_query,
                                            pairs);
                    const uint32_t limit
                        = causal ? query_pos + 1u : sequence;
                    int32_t maximum = -2147483647;
                    for (uint32_t key_pos = 0; key_pos < limit; ++key_pos)
                        {
                            const int32_t score = _S_attention_score (
                                q_row, rotated_query,
                                key._M_code + key_pos * stride + offset,
                                key_pos, head_dim, rope);
                            if (score > maximum)
                                maximum = score;
                        }

                    int64_t accumulator[_S_attention_head_limit]
                        = { 0, 0, 0, 0, 0, 0, 0, 0 };
                    uint32_t denominator = 0;
                    for (uint32_t key_pos = 0; key_pos < limit; ++key_pos)
                        {
                            const int32_t score = _S_attention_score (
                                q_row, rotated_query,
                                key._M_code + key_pos * stride + offset,
                                key_pos, head_dim, rope);
                            uint32_t distance
                                = static_cast<uint32_t> (maximum - score);
                            distance = _S_div32_u (distance << 11, score_div);
                            if (distance > 255u)
                                distance = 255u;
                            const int32_t weight
                                = _S_exp_table._M_value[distance];
                            denominator += static_cast<uint32_t> (weight);
                            const int8_t *const v_row
                                = value._M_code + key_pos * stride + offset;
                            for (uint32_t d = 0; d < head_dim; ++d)
                                accumulator[d] += weight
                                                  * static_cast<int32_t> (
                                                      v_row[d]);
                        }
                    if (denominator == 0u)
                        return Status::INTERNAL;
                    const uint32_t half = denominator >> 1;
                    for (uint32_t d = 0; d < head_dim; ++d)
                        {
                            const int64_t mixed
                                = (accumulator[d] + half)
                                  / static_cast<int64_t> (denominator);
                            output._M_code[query_pos * stride + offset + d]
                                = static_cast<int8_t> (
                                    _S_i8_code (static_cast<int32_t> (mixed)));
                        }
                }
        }
    output.zero_grad ();
    return Status::OK;
}

/**
 * @brief Backward of attention, in the `BackwardFn` signature.
 *
 * Recorded node layout: operands [query, key, value]; saved [spec, scratch].
 * The scratch holds a `seq`-lane probability row followed by a `seq`-lane
 * `dP` row.  Every reduction is an int64 local narrowed once, so a long
 * sequence cannot wrap either the softmax contraction or the dQ/dK sums.
 */
[[nodiscard]] inline Status
attention_backward (const Trainable &output, const Trainable &grad_output,
                    GradContext &context) noexcept
{
    const AttentionSpec *const slot
        = context.saved_as<AttentionSpec> (_S_attention_spec);
    if (slot == nullptr)
        return Status::BAD_LAYOUT;
    const AttentionSpec spec = *slot;
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    if (context._M_operands == nullptr || context._M_operand_count < 3u)
        return Status::BAD_LAYOUT;
    const Trainable &query = context._M_operands[0];
    const Trainable &key = context._M_operands[1];
    const Trainable &value = context._M_operands[2];
    if (output._M_count != spec.elements ()
        || query._M_count != spec.elements ()
        || key._M_count != spec.elements ()
        || value._M_count != spec.elements ())
        return Status::BAD_LAYOUT;
    if (grad_output._M_grad == nullptr)
        return Status::OK;
    int16_t *const scratch = context.saved_as<int16_t> (_S_attention_scratch);
    if (scratch == nullptr)
        return Status::BAD_LAYOUT;

    const uint32_t sequence = spec._M_seq;
    const uint32_t head_dim = spec._M_head_dim;
    const uint32_t heads = spec._M_heads;
    const uint32_t stride = spec.stride ();
    const bool causal = spec.has_causal ();
    const bool rope = spec.has_rope ();
    const uint32_t pairs = head_dim / 2u;
    const uint32_t score_scale = _S_isqrt (head_dim << 16);
    const _S_Div32 score_div = _S_div32_make (score_scale);
    const int32_t scale_q12 = attention_scale_q12 (head_dim);
    int16_t *const prob = scratch;
    int16_t *const dprod = scratch + sequence;
    uint32_t saturated = 0u;

    for (uint32_t head = 0; head < heads; ++head)
        {
            const uint32_t offset = head * head_dim;
            for (uint32_t query_pos = 0; query_pos < sequence; ++query_pos)
                {
                    const int8_t *const q_row
                        = query._M_code + query_pos * stride + offset;
                    int16_t rotated_query[_S_attention_head_limit] = {
                        0, 0, 0, 0, 0, 0, 0, 0
                    };
                    if (rope)
                        _S_rope_rotate_row (q_row, query_pos, rotated_query,
                                            pairs);
                    int32_t q_lanes[_S_attention_head_limit];
                    attention_row_q15 (q_row, query_pos, head_dim, rope,
                                       q_lanes);
                    const uint32_t limit
                        = causal ? query_pos + 1u : sequence;
                    int32_t maximum = -2147483647;
                    for (uint32_t key_pos = 0; key_pos < limit; ++key_pos)
                        {
                            const int32_t score = _S_attention_score (
                                q_row, rotated_query,
                                key._M_code + key_pos * stride + offset,
                                key_pos, head_dim, rope);
                            if (score > maximum)
                                maximum = score;
                        }

                    uint32_t denominator = 0;
                    for (uint32_t key_pos = 0; key_pos < limit; ++key_pos)
                        {
                            const int32_t score = _S_attention_score (
                                q_row, rotated_query,
                                key._M_code + key_pos * stride + offset,
                                key_pos, head_dim, rope);
                            uint32_t distance
                                = static_cast<uint32_t> (maximum - score);
                            distance = _S_div32_u (distance << 11, score_div);
                            if (distance > 255u)
                                distance = 255u;
                            const int32_t weight
                                = _S_exp_table._M_value[distance];
                            denominator += static_cast<uint32_t> (weight);
                            prob[key_pos] = static_cast<int16_t> (weight);

                            /* dP_j = dO . V_j, with the value code widened to
                               Q15 so the product is a lane-domain dot. */
                            int64_t dot = 0;
                            const int8_t *const v_row
                                = value._M_code + key_pos * stride + offset;
                            for (uint32_t d = 0; d < head_dim; ++d)
                                dot += static_cast<int64_t> (
                                           grad_output._M_grad[query_pos * stride
                                                               + offset + d])
                                       * code_to_q15 (v_row[d]);
                            dprod[key_pos] = q15_to_i16_sat (
                                narrow_sum_q15 (dot));
                        }
                    if (denominator == 0u)
                        return Status::INTERNAL;

                    int64_t sum_prob_dprod = 0;
                    for (uint32_t key_pos = 0; key_pos < limit; ++key_pos)
                        {
                            const int32_t p
                                = ratio_q15 (prob[key_pos],
                                             static_cast<int32_t> (denominator));
                            prob[key_pos] = q15_to_i16_sat (p);
                            sum_prob_dprod += q15_mul (p, dprod[key_pos]);
                        }
                    const int32_t contraction
                        = clamp_sum_q15 (sum_prob_dprod);

                    int64_t dq_rot[_S_attention_head_limit]
                        = { 0, 0, 0, 0, 0, 0, 0, 0 };
                    for (uint32_t key_pos = 0; key_pos < limit; ++key_pos)
                        {
                            const int32_t p = prob[key_pos];
                            const int32_t ds = q15_mul (
                                p, dprod[key_pos] - contraction);
                            const int32_t dscore
                                = attention_div_scale (ds, scale_q12);

                            int32_t k_lanes[_S_attention_head_limit];
                            attention_row_q15 (key._M_code + key_pos * stride
                                                   + offset,
                                               key_pos, head_dim, rope,
                                               k_lanes);
                            int32_t dk_rot[_S_attention_head_limit]
                                = { 0, 0, 0, 0, 0, 0, 0, 0 };
                            for (uint32_t d = 0; d < head_dim; ++d)
                                {
                                    dq_rot[d] += q15_mul (dscore, k_lanes[d]);
                                    dk_rot[d] = q15_mul (dscore, q_lanes[d]);
                                    if (value._M_grad != nullptr
                                        && value.add_grad (
                                            key_pos * stride + offset + d,
                                            q15_mul (
                                                p, grad_output._M_grad
                                                       [query_pos * stride
                                                        + offset + d])))
                                        ++saturated;
                                }

                            if (key._M_grad != nullptr)
                                {
                                    int32_t plain[_S_attention_head_limit]
                                        = { 0, 0, 0, 0, 0, 0, 0, 0 };
                                    if (rope)
                                        attention_unrotate (dk_rot, key_pos,
                                                            pairs, plain);
                                    else
                                        for (uint32_t d = 0; d < head_dim; ++d)
                                            plain[d] = dk_rot[d];
                                    for (uint32_t d = 0; d < head_dim; ++d)
                                        if (key.add_grad (
                                                key_pos * stride + offset + d,
                                                plain[d]))
                                            ++saturated;
                                }
                        }

                    if (query._M_grad != nullptr)
                        {
                            int32_t rotated_grad[_S_attention_head_limit]
                                = { 0, 0, 0, 0, 0, 0, 0, 0 };
                            for (uint32_t d = 0; d < head_dim; ++d)
                                rotated_grad[d] = clamp_sum_q15 (dq_rot[d]);
                            int32_t plain[_S_attention_head_limit]
                                = { 0, 0, 0, 0, 0, 0, 0, 0 };
                            if (rope)
                                attention_unrotate (rotated_grad, query_pos,
                                                    pairs, plain);
                            else
                                for (uint32_t d = 0; d < head_dim; ++d)
                                    plain[d] = rotated_grad[d];
                            for (uint32_t d = 0; d < head_dim; ++d)
                                if (query.add_grad (query_pos * stride + offset
                                                        + d,
                                                    plain[d]))
                                    ++saturated;
                        }
                }
        }
    context._M_saturation_events += saturated;
    return Status::OK;
}

/**
 * @brief Carve the attention state, run the forward and record the node.
 *
 * Two saved slots -- the spec and a `2 * seq` lane scratch -- and three
 * operands.  The forward runs in constant stack beyond the eight-lane rows, so
 * the pool cost is a function of the sequence only.
 */
[[nodiscard]] inline Status
attention_run (Tape &tape, const AttentionSpec &spec, const Trainable &query,
               const Trainable &key, const Trainable &value,
               Trainable &output) noexcept
{
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    if (query._M_count != spec.elements ()
        || key._M_count != spec.elements ()
        || value._M_count != spec.elements ()
        || output._M_count != spec.elements ())
        return Status::BAD_LAYOUT;

    Saved slots[_S_max_operands];
    int16_t *scratch = nullptr;
    const Status save_scratch
        = tape.save (2u * spec._M_seq, slots[_S_attention_scratch], scratch);
    if (!status_ok (save_scratch))
        return save_scratch;
    {
        AttentionSpec *saved_spec = nullptr;
        const Status save = tape.save (1u, slots[_S_attention_spec], saved_spec);
        if (!status_ok (save))
            return save;
        *saved_spec = spec;
    }

    const Status forward
        = attention_forward (spec, query, key, value, output);
    if (!status_ok (forward))
        return forward;

    Trainable operands[3] = { query, key, value };
    return tape.record (attention_backward, spec._M_op_id, output, operands,
                        3u, slots, _S_max_operands);
}
} // namespace zzs::mllee::train
