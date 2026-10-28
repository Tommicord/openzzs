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
#include "mllee/train_random.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Inverted dropout: mask, forward, saved state, backward.
 *
 * The gain is applied in the forward, so the deployed model needs no rescale --
 * that is the whole point of inverted dropout and it is also what the inference
 * core's DROPOUT op does (`value * 127 / (127 - drop)` in the code domain, see
 * @ref dropout_inference_code, which this module reproduces exactly).
 *
 * @par Where the gain lives
 * A gain of 1/(1-p) is never a Q15 lane, so it is kept as a magnitude at 14
 * fraction bits: @ref dropout_gain_q14 returns 2^29 / keep_q15, which is 16384
 * for keep = 1 and 32768 for keep = 0.5.  Multiplying a lane by it is
 * @ref q15_scale_shift with shift 14 -- an exact requantization that rounds and
 * does not clamp, so the caller decides what saturation means and counts it.
 *
 * @par The three kinds
 *   - @ref DropoutKind::Elementwise: one Bernoulli draw per element, the usual
 *     dense-layer form;
 *   - @ref DropoutKind::Spatial: one draw per (sample, channel) replicated over
 *     the spatial axis -- Dropout2d.  It costs `units` draws instead of
 *     `units * inner` and keeps the written stream stride-1 (@ref
 *     fill_mask_blocked), so the backward path is bit-identical to the
 *     elementwise one;
 *   - @ref DropoutKind::Alpha: AlphaDropout, i.e. the SELU variant.  Dropped
 *     lanes are filled with the moment-matching value
 *       fill = -alpha * p / (1 - p),
 *     which is what makes the first and second moments of the output equal the
 *     input's while the surviving lanes are scaled by 1/(1-p): the two
 *     conditions reduce to p (alpha - 1) gain + (1 - p) fill = 0 for the mean
 *     and, using gain = 1/(1-p), to the closed form above for the fill.
 *     The published derivation assumes E[x] = 0 and E[x^2] = 1 -- true for a
 *     self-normalizing SELU net -- so on an arbitrary int8 tensor the preserved
 *     moments are approximate: the variance term is off by the input's own
 *     variance and the fill is a lane-domain constant that saturates once
 *     |fill| exceeds 1.0 (p above 0.5 with alpha 1.67).  The variant is
 *     therefore documented as a *moment-matched fill*, not an exact SELU
 *     identity, and its backward is the same masked routing as the other kinds
 *     because the fill contributes no gradient.
 *
 * @par Reproducibility
 * The mask stream is a @ref XorShift64 seeded from @ref dropout_step_seed, so
 * "epoch 3, step 7" is the same mask on every target, and @ref DropoutState
 * carries the engine state for a resume.
 *
 * @par Pool cost of one recorded layer (elements lanes)
 *   - mask:            align_up (elements, 4) bytes
 *   - gain slot:       4 bytes
 *   - backward scratch: align_up (2 * elements, 4) bytes
 * Total 3 * elements + 8 rounded to the pool alignment; @ref dropout_run
 * reports the exact figure through the tape's saved-byte counter.
 */
namespace zzs::mllee::train
{
/** @brief Which dropout pattern a layer applies. */
enum class DropoutKind : uint8_t
{
    Elementwise = 0u,
    Spatial = 1u,
    Alpha = 2u,
};

/** @brief Saved-slot index of the mask. */
inline constexpr uint32_t _S_dropout_mask = 0u;
/** @brief Saved-slot index of the Q14 gain. */
inline constexpr uint32_t _S_dropout_gain = 1u;
/** @brief Saved-slot index of the backward routing scratch. */
inline constexpr uint32_t _S_dropout_scratch = 2u;
/** @brief Op id recorded on the tape for an elementwise dropout. */
inline constexpr uint32_t _S_dropout_op_id = 0x0D01u;

/**
 * @brief Dropout parameters, plain data so it can be carved or embedded.
 *
 * @c _M_units times @c _M_inner equals @c _M_elements for the spatial kind and
 * is unused (1 times elements) for the other two.
 */
struct DropoutConfig
{
    uint64_t _M_seed = 0u;
    int32_t _M_drop_q15 = 0;
    int32_t _M_keep_q15 = _S_q15_one;
    int32_t _M_gain_q14 = 16384;
    int32_t _M_alpha_wide_q15 = 0;
    int32_t _M_fill_wide_q15 = 0;
    uint32_t _M_elements = 0u;
    uint32_t _M_units = 0u;
    uint32_t _M_inner = 0u;
    uint8_t _M_kind = 0u;
    uint8_t _M_train = 0u;
    uint16_t _M_reserved = 0u;
};

static_assert (sizeof (DropoutConfig) == 48u,
               "the dropout parameters must stay one aligned 48-byte block");

/**
 * @brief Runtime dropout state: what was drawn and what saturated.
 *
 * Sized for pool carving (24 bytes, a multiple of the pool alignment) and holds
 * the engine state so a resumed step continues the same stream.
 */
struct DropoutState
{
    uint64_t _M_engine_state = 0u;
    uint32_t _M_kept = 0u;
    uint32_t _M_draws = 0u;
    uint32_t _M_saturated = 0u;
    uint32_t _M_drop_code = 0u;
};

static_assert (sizeof (DropoutState) == 24u,
               "the dropout state is one 64-bit word plus four counts");

/** @brief Pool bytes one recorded dropout layer costs, mask and scratch both in. */
[[nodiscard]] inline constexpr uint32_t
dropout_bytes (const uint32_t elements) noexcept
{
    return align_up (elements, _S_pool_alignment)
           + align_up (2u * elements, _S_pool_alignment) + 4u;
}

/**
 * @brief Inverted-dropout gain 1/(1-p) at 14 fraction bits.
 *
 * @param keep_q15 survival probability in Q15; 0 (everything dropped) yields 0
 *        so the caller writes a zero tensor instead of dividing by zero.
 */
[[nodiscard]] inline constexpr int32_t
dropout_gain_q14 (const int32_t keep_q15) noexcept
{
    if (keep_q15 <= 0)
        return 0;
    const int64_t gain = (static_cast<int64_t> (1) << 29) / keep_q15;
    return gain > 0x40000000ll ? 0x40000000 : static_cast<int32_t> (gain);
}

/**
 * @brief AlphaDropout fill -alpha p/(1-p) in 15 fraction bits.
 *
 * The intermediate is a ratio of two Q15 quantities, hence a wide magnitude
 * that can exceed the lane range; the caller narrows it with the usual code
 * saturation and counts it.
 */
[[nodiscard]] inline constexpr int32_t
dropout_alpha_fill_q15 (const int32_t alpha_wide_q15, const int32_t drop_q15,
                        const int32_t keep_q15) noexcept
{
    if (keep_q15 <= 0 || alpha_wide_q15 == 0)
        return 0;
    const int64_t fill = -(static_cast<int64_t> (alpha_wide_q15) * drop_q15)
                         / keep_q15;
    return fill > 0x40000000ll
               ? 0x40000000
               : (fill < -0x40000000ll ? -0x40000000 : static_cast<int32_t> (fill));
}

/** @brief Per-step mask seed: a distinct stream for every (epoch, step) pair. */
[[nodiscard]] inline constexpr uint64_t
dropout_step_seed (const uint64_t base, const uint32_t epoch,
                   const uint32_t step) noexcept
{
    return rng_mix_seed (rng_derive_seed (base, epoch), step);
}

/**
 * @brief Fill in the parameter block; every derived number is computed here so
 *        the forward, the backward and the deploy form cannot disagree.
 *
 * @param units inner blocks of a spatial mask (N*C), @p inner elements per
 *        block (H*W); both must be 1 times elements for the other kinds.
 */
[[nodiscard]] inline Status
dropout_config_init (DropoutConfig &out, const DropoutKind kind,
                     const bool train, const int32_t drop_q15,
                     const uint64_t seed, const uint32_t elements,
                     const uint32_t units, const uint32_t inner,
                     const int32_t alpha_wide_q15) noexcept
{
    if (elements == 0u || elements > _S_max_elements)
        return Status::OUT_OF_RANGE;
    if (drop_q15 < 0 || drop_q15 > _S_q15_max)
        return Status::OUT_OF_RANGE;
    const uint32_t want_units = kind == DropoutKind::Spatial ? units : 1u;
    const uint32_t want_inner = kind == DropoutKind::Spatial ? inner : elements;
    if (want_units == 0u || want_inner == 0u)
        return Status::BAD_LAYOUT;
    if (static_cast<uint64_t> (want_units) * want_inner != elements)
        return Status::BAD_LAYOUT;
    out._M_seed = seed;
    out._M_kind = static_cast<uint8_t> (kind);
    out._M_train = train ? 1u : 0u;
    out._M_drop_q15 = drop_q15;
    out._M_keep_q15 = _S_q15_one - drop_q15;
    out._M_gain_q14 = dropout_gain_q14 (out._M_keep_q15);
    out._M_alpha_wide_q15 = kind == DropoutKind::Alpha ? alpha_wide_q15 : 0;
    out._M_fill_wide_q15 = kind == DropoutKind::Alpha
                               ? dropout_alpha_fill_q15 (alpha_wide_q15,
                                                         drop_q15,
                                                         out._M_keep_q15)
                               : 0;
    out._M_elements = elements;
    out._M_units = want_units;
    out._M_inner = want_inner;
    return Status::OK;
}

/**
 * @brief Draw the keep-mask.
 *
 * @param engine stream seeded from @ref dropout_step_seed by the caller.
 * @param kept   lanes that survive, in elements (a spatial unit counts its whole
 *               block); @p draws counts the Bernoulli trials actually spent.
 */
template <typename Engine>
[[nodiscard]] inline Status
dropout_generate_mask (const DropoutConfig &config, Engine &engine,
                       uint8_t *const mask, uint32_t &kept,
                       uint32_t &draws) noexcept
{
    if (mask == nullptr)
        return Status::BAD_LAYOUT;
    if (config._M_kind == static_cast<uint8_t> (DropoutKind::Spatial))
        {
            kept = fill_mask_blocked (engine, config._M_keep_q15,
                                      config._M_units, config._M_inner, mask);
            draws = config._M_units;
            return Status::OK;
        }
    kept = fill_mask (engine, config._M_keep_q15, mask, config._M_elements);
    draws = config._M_elements;
    return Status::OK;
}

/**
 * @brief Forward on the code stream: zero the dropped lanes, gain the rest.
 *
 * Eval mode is the identity copy, which is what makes the same body usable for
 * the validation pass.  The gain is applied in the code domain, so a lane near
 * full scale saturates onto 127 exactly like the deployed op does, and @p
 * saturated counts those lanes.
 */
[[nodiscard]] inline Status
dropout_forward_codes (const DropoutConfig &config, const int8_t *const in,
                       const uint8_t *const mask, int8_t *const out,
                       uint32_t &saturated) noexcept
{
    if (in == nullptr || out == nullptr)
        return Status::BAD_LAYOUT;
    if (config._M_train == 0u)
        {
            for (uint32_t i = 0; i < config._M_elements; ++i)
                out[i] = in[i];
            return Status::OK;
        }
    if (mask == nullptr)
        return Status::BAD_LAYOUT;
    const int8_t fill = q15_to_code (config._M_fill_wide_q15);
    for (uint32_t i = 0; i < config._M_elements; ++i)
        {
            if (mask[i] != 0u)
                {
                    const int32_t scaled
                        = q15_scale_shift (in[i], config._M_gain_q14, 14u);
                    if (scaled > 127 || scaled < -127)
                        ++saturated;
                    out[i] = _S_i8_code (scaled);
                    continue;
                }
            out[i] = fill;
        }
    return Status::OK;
}

/** @brief Backward lanes: grad_in = mask ? gain * grad_out : 0. */
[[nodiscard]] inline Status
dropout_route_grad_q15 (const DropoutConfig &config,
                        const int16_t *const grad_out,
                        const uint8_t *const mask, int16_t *const grad_in,
                        uint32_t &saturated) noexcept
{
    if (grad_out == nullptr || grad_in == nullptr || mask == nullptr)
        return Status::BAD_LAYOUT;
    /* Foundation primitive first: the mask is a pure route, then the gain. */
    route_by_mask (grad_out, mask, grad_in, config._M_elements);
    if (config._M_gain_q14 == 16384)
        return Status::OK; /* p == 0: the gain is the identity */
    for (uint32_t i = 0; i < config._M_elements; ++i)
        {
            const int32_t scaled = q15_scale_shift (grad_in[i],
                                                    config._M_gain_q14, 14u);
            if (scaled > _S_q15_max || scaled < _S_q15_min)
                ++saturated;
            grad_in[i] = q15_to_i16_sat (scaled);
        }
    return Status::OK;
}

/** @brief The inference op's drop parameter: p as a code-domain count in 1/127. */
[[nodiscard]] inline constexpr int32_t
dropout_aux_code (const int32_t drop_q15) noexcept
{
    return _S_round_shift (static_cast<int64_t> (drop_q15) * 127,
                            _S_grad_frac_bits);
}

/**
 * @brief The deployed DROPOUT op, one lane, bit-for-bit.
 *
 * The core computes `keep = 127 - drop` and rounds `value * 127 / keep`; using
 * it means the training forward and the inference forward differ only by the
 * 1/127 quantization of p itself (below 0.4 % of the gain for p <= 0.5).
 */
[[nodiscard]] inline constexpr int32_t
dropout_inference_code (const int32_t code, const int32_t drop_code) noexcept
{
    if (drop_code <= 0)
        return code;
    const int32_t keep = 127 - drop_code;
    if (keep <= 0)
        return 0;
    const int32_t scaled = code * 127;
    return _S_i8_code ((scaled + (scaled >= 0 ? keep / 2 : -keep / 2)) / keep);
}

/**
 * @brief Recorded forward: draw the mask, write the output, save the state.
 *
 * Three saved tensors (mask, gain, scratch) and one operand (the input).  The
 * scratch is carved here because a backward function may not allocate; it is
 * what lets the backward call @ref route_by_mask without writing through the
 * input's gradient lanes, which would break the accumulate-only rule.
 */
[[nodiscard]] inline Status
dropout_backward (const Trainable &output, const Trainable &grad_output,
                  GradContext &context) noexcept;

[[nodiscard]] inline Status
dropout_run (Tape &tape, const DropoutConfig &config,
             const Trainable &input, Trainable &output,
             DropoutState *const state = nullptr) noexcept
{
    if (!status_ok (input.validate ()) || !status_ok (output.validate ()))
        return Status::BAD_LAYOUT;
    if (input._M_count != config._M_elements
        || output._M_count != config._M_elements)
        return Status::BAD_LAYOUT;
    if (input._M_shape != output._M_shape)
        return Status::BAD_LAYOUT;

    XorShift64 engine = XorShift64::seeded (config._M_seed);
    uint8_t *mask = nullptr;
    Saved slots[_S_max_operands];
    int32_t *gain = nullptr;
    int16_t *scratch = nullptr;
    uint32_t kept = 0u;
    uint32_t draws = 0u;
    Status status = Status::OK;

    for (uint32_t index = 0u; index < _S_max_operands; ++index)
        slots[index] = Saved{};

    if (config._M_train != 0u)
        {
            if (!tape.is_open ())
                return Status::INTERNAL;
            status = tape.save (config._M_elements, slots[_S_dropout_mask],
                                mask);
            if (!status_ok (status))
                return status;
            status = tape.save (1u, slots[_S_dropout_gain], gain);
            if (!status_ok (status))
                return status;
            status = tape.save (config._M_elements, slots[_S_dropout_scratch],
                                scratch);
            if (!status_ok (status))
                return status;
            *gain = config._M_gain_q14;
            status = dropout_generate_mask (config, engine, mask, kept, draws);
            if (!status_ok (status))
                return status;
        }
    else
        {
            for (uint32_t index = 0u; index < _S_max_operands; ++index)
                slots[index] = Saved{};
        }

    uint32_t saturated = 0u;
    const Status forward
        = dropout_forward_codes (config, input._M_code, mask, output._M_code,
                                 saturated);
    if (!status_ok (forward))
        return forward;
    if (state != nullptr)
        {
            state->_M_engine_state = engine._M_state;
            state->_M_kept = kept;
            state->_M_draws = draws;
            state->_M_saturated = saturated;
            state->_M_drop_code = static_cast<uint32_t> (
                dropout_aux_code (config._M_drop_q15));
        }
    if (config._M_train == 0u || !tape.is_open ())
        return Status::OK;

    Trainable operands[_S_max_operands];
    operands[0] = input;
    return tape.record (dropout_backward, _S_dropout_op_id, output, operands,
                        1u, slots, 3u);
}

/**
 * @brief d(dropout)/dx: route the incoming gradient by the saved mask, times
 *        the inverted gain, accumulated into the input's lanes.
 *
 * Reads its own gradient through @p grad_output (the tape's convention), so the
 * only buffer it touches is the saved scratch; @ref route_by_mask does the
 * mask step and the gain multiply counts its clamps into @c
 * context._M_saturation_events.
 */
[[nodiscard]] inline Status
dropout_backward (const Trainable &output, const Trainable &grad_output,
                  GradContext &context) noexcept
{
    if (context._M_operands == nullptr || context._M_operand_count < 1u)
        return Status::BAD_LAYOUT;
    const uint8_t *const mask
        = context.saved_as<uint8_t> (_S_dropout_mask);
    const int32_t *const gain
        = context.saved_as<int32_t> (_S_dropout_gain);
    int16_t *const scratch
        = context.saved_as<int16_t> (_S_dropout_scratch);
    if (mask == nullptr || gain == nullptr || scratch == nullptr)
        return Status::BAD_LAYOUT;
    if (grad_output._M_grad == nullptr || output._M_count == 0u)
        return Status::BAD_LAYOUT;
    const Trainable &input = context._M_operands[0];
    if (input._M_count != output._M_count)
        return Status::BAD_LAYOUT;
    if (input._M_grad == nullptr)
        return Status::OK; /* a frozen branch: nothing to accumulate */

    route_by_mask (grad_output._M_grad, mask, scratch, output._M_count);
    uint32_t saturated = 0u;
    const int32_t gain_q14 = *gain;
    for (uint32_t i = 0; i < output._M_count; ++i)
        {
            const int32_t scaled = gain_q14 == 16384
                                       ? scratch[i]
                                       : q15_scale_shift (scratch[i], gain_q14,
                                                          14u);
            const int32_t lane = q15_clamp (scaled);
            if (lane != scaled)
                ++saturated;
            if (input.add_grad (i, lane))
                ++saturated;
        }
    context._M_saturation_events += saturated;
    return Status::OK;
}

/**
 * @brief Eval-mode helper: the inference op's whole stream, mask and all.
 *
 * Reproduces `_S_dropout` lane for lane from the same drop code, including the
 * drop decision (a 7-bit xorshift32 field compared against the code-domain p),
 * so a caller who wants the deployed stochastic behaviour can check its effect
 * on the trained codes without a model file.
 */
[[nodiscard]] inline Status
dropout_inference_forward (const int32_t drop_code, const uint32_t stream_seed,
                           const int8_t *const in, int8_t *const out,
                           const uint32_t count) noexcept
{
    if (in == nullptr || out == nullptr)
        return Status::BAD_LAYOUT;
    if (drop_code < 0 || drop_code > 127)
        return Status::OUT_OF_RANGE;
    if (count > _S_max_elements)
        return Status::OUT_OF_RANGE;
    const int32_t keep = 127 - drop_code;
    uint32_t state = stream_seed | 1u;
    for (uint32_t i = 0; i < count; ++i)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if (static_cast<int32_t> ((state >> 17) & 0x7Fu) < drop_code
                || keep == 0)
                {
                    out[i] = 0;
                    continue;
                }
            out[i] = static_cast<int8_t> (
                dropout_inference_code (in[i], drop_code));
        }
    return Status::OK;
}
} // namespace zzs::mllee::train
