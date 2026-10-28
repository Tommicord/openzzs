/* Copyright(c) 2026 Tommicord
 *
 * Permission for use, copying and distribution is granted under the same
 * MIT-style terms as the rest of the mllee module (see mllee/core.hpp).
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND. */

#pragma once

#include "mllee/ops.hpp"
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
 * @brief Loss ops: forward value, scalar node, and a tape-registered backward.
 *
 * Every loss here follows the same shape, set by the op-module contract in
 * mllee/train_graph.hpp:
 *
 *   - `*_forward (pool, tape, operands..., result)` computes the loss, carves a
 *     one-element scalar node (int8 code + one Q15 gradient lane) from the
 *     caller's pool, and -- when @p tape is non-null -- records a BackwardFn
 *     plus the small saved tensors (target indices, class counts, config words)
 *     that backward needs.  Backward allocates nothing and records nothing.
 *   - The backward reads the seed from the scalar node's gradient lane (the
 *     tape seeds it with the loss scale) and accumulates seed-scaled dL/dx
 *     lanes into the operand with Trainable::add_grad, counting the lanes that
 *     clamped into GradContext::_M_saturation_events.
 *
 * Why the loss VALUE lives in Q7 and the gradient lanes in Q15: a Q15 lane
 * spans (-1, 1), but a single MSE term (p - t)^2 with codes at the rails is 4,
 * a Huber term is 2*delta, a hinge term is 2, and a cross-entropy sum over C
 * classes reaches C * |log p|.  None of those fit one Q15 lane, so forward
 * accumulates in int64 Q7 (one unit = 1/128) and reports a Q7 result; only the
 * derivative -- which is per-element and bounded by its own chain rule -- is
 * narrowed into Q15 lanes, saturating at the rails.
 *
 * Pool cost of this module, per loss evaluation (n elements, C classes):
 *   mse / mae / huber / quantile / hinge : 8 bytes (scalar node) + 0 saved.
 *   bce                                   : 8 bytes (scalar node).
 *   cross entropy                         : 8 + 4 (target index / meta slab).
 *   nll                                   : 8 + 4 (target index) + 4 (meta).
 *   cosine                                : 8 bytes (scalar node).
 *   multiclass hinge                      : 8 + 4 (target index) + 4 (meta).
 * The scalar node is two 4-byte-aligned carves (an int8 and an int16 slab), so
 * a loss op costs at least 8 bytes of the 8192-byte training pool.
 */
namespace zzs::mllee::train
{
/** @name Loss op identifiers
 * The tape records an op id per node so a profile pass can attribute nodes
 * without stringly-typed names; the loss family owns 0x4C01..0x4C0A. @{ */
inline constexpr uint32_t _S_op_mse = 0x4C01u;
inline constexpr uint32_t _S_op_mae = 0x4C02u;
inline constexpr uint32_t _S_op_huber = 0x4C03u;
inline constexpr uint32_t _S_op_quantile = 0x4C04u;
inline constexpr uint32_t _S_op_bce = 0x4C05u;
inline constexpr uint32_t _S_op_cross_entropy = 0x4C06u;
inline constexpr uint32_t _S_op_nll = 0x4C07u;
inline constexpr uint32_t _S_op_cosine = 0x4C08u;
inline constexpr uint32_t _S_op_hinge = 0x4C09u;
inline constexpr uint32_t _S_op_hinge_multiclass = 0x4C0Au;
/** @} */

/** @brief Q7 log units represented by one log-prob code step (1/32 of a nat). */
inline constexpr int32_t _S_log_step_q7 = 4;

/**
 * @brief The result of a loss forward: the Q7 value, a lane-domain copy for
 *        logging, and the one-element scalar node the tape seeds.
 *
 * The node's int8 code stores the saturating Q15 copy (so a node can feed a
 * second graph op), while _M_loss_q7 keeps the exact accumulator result -- a
 * caller that needs loss magnitudes above 1.0 must read the Q7 field.
 */
struct LossResult
{
    /** @brief Loss value in Q7 (one unit = 1/128 nats/squares). */
    int32_t _M_loss_q7 = 0;
    /** @brief Same value saturated to Q15, for lane-domain logging only. */
    int32_t _M_loss_q15 = 0;
    /** @brief The scalar node (one code, one gradient lane). */
    Trainable _M_node;
};

static_assert (sizeof (LossResult) == sizeof (Trainable) + 8u,
               "loss result is two words plus one Trainable view");

/** @brief Huber transition point, in Q15 on the code-normalized diff domain. */
struct HuberParams
{
    int32_t _M_delta_q15 = _S_q15_max;
};

static_assert (sizeof (HuberParams) == 4u, "huber params is one Q15 word");

/** @brief Quantile target tau, in Q15 (0.5 = symmetric, 0.9 = penalize underprediction). */
struct QuantileParams
{
    int32_t _M_tau_q15 = 16384;
};

static_assert (sizeof (QuantileParams) == 4u, "quantile params is one Q15 word");

/** @brief Cosine-loss guard: added to |p| * |t| so a zero vector yields 0 gradient. */
struct CosineParams
{
    int32_t _M_eps_q15 = 32;
};

static_assert (sizeof (CosineParams) == 4u, "cosine params is one Q15 word");

/** @brief Hinge margin in Q7 (128 = the conventional 1.0 margin). */
struct HingeParams
{
    int32_t _M_margin_q7 = _S_q7_one;
};

static_assert (sizeof (HingeParams) == 4u, "hinge params is one Q7 word");

/** @brief Class count plus target slot, saved for the index-domain losses. */
struct TargetMeta
{
    uint16_t _M_classes = 0u;
    uint16_t _M_target = 0u;
    int32_t _M_margin_q7 = 0;
};

static_assert (sizeof (TargetMeta) == 8u, "target meta is two halfwords plus a word");

/**
 * @brief Convert a Q30 squared-diff sum (n terms) into the mean's Q7 value.
 *
 * d in Q15 makes d^2 Q30; the mean divides by n and Q7 re-bases by 2^23.  The
 * rounding term (n << 22) adds a half ulp before the division so the result is
 * round-half-away for the always-nonnegative square sum.
 */
[[nodiscard]] inline constexpr int32_t
q30_sumsq_mean_to_q7 (const int64_t total_q30, const uint32_t count) noexcept
{
    if (count == 0u)
        return 0;
    const int64_t numerator
        = total_q30 + (static_cast<int64_t> (count) << 22);
    return static_cast<int32_t> (
        numerator / (static_cast<int64_t> (count) << 23));
}

/** @brief Convert a Q15 magnitude sum into the mean's Q7 value. */
[[nodiscard]] inline constexpr int32_t
q15_sum_mean_to_q7 (const int64_t total_q15, const uint32_t count) noexcept
{
    if (count == 0u)
        return 0;
    const int64_t scaled
        = total_q15 + (static_cast<int64_t> (count) << 7);
    return static_cast<int32_t> (scaled / (static_cast<int64_t> (count) << 8));
}

/**
 * @brief Carve the one-element scalar node for a loss result and publish @p out.
 * @param loss_q7 The accumulator-domain loss value (Q7).
 */
[[nodiscard]] inline Status
finish_loss (Pool &pool, const int32_t loss_q7, LossResult &out) noexcept
{
    out._M_loss_q7 = loss_q7;
    out._M_loss_q15 = q15_clamp (loss_q7 << 8);
    int8_t *code = nullptr;
    int16_t *grad = nullptr;
    Status status = pool.carve_typed<int8_t> (1u, code);
    if (!status_ok (status))
        return status;
    status = pool.carve_typed<int16_t> (1u, grad);
    if (!status_ok (status))
        return status;
    *code = q15_to_code (out._M_loss_q15);
    *grad = 0;
    out._M_node = Trainable::make (code, grad, Shape::scalar (),
                                   Flag::RequiresGrad);
    return Status::OK;
}

/** @brief Validate the shared elementwise-loss operand pair. */
[[nodiscard]] inline Status
loss_pair_ok (const Trainable &pred, const Trainable &target) noexcept
{
    Status status = pred.validate ();
    if (!status_ok (status))
        return status;
    status = target.validate ();
    if (!status_ok (status))
        return status;
    if (pred._M_count != target._M_count || pred._M_count == 0u)
        return Status::BAD_LAYOUT;
    return Status::OK;
}

/** @brief Record a loss op when a tape was supplied; a null tape is a no-op. */
[[nodiscard]] inline Status
record_loss (Tape *const tape, const BackwardFn backward, const uint32_t op_id,
             const LossResult &result, const Trainable *const operands,
             const uint32_t operand_count, const Saved *const saved,
             const uint32_t saved_count) noexcept
{
    if (tape == nullptr)
        return Status::OK;
    return tape->record (backward, op_id, result._M_node, operands,
                         operand_count, saved, saved_count);
}

/* ======================================================================== */
/* MSE                                                                     */
/* ======================================================================== */

/** @brief dL/dpred = seed * 2 * (p - t) / n, accumulated with saturation. */
[[nodiscard]] inline Status
mse_backward (const Trainable &output, const Trainable &grad_output,
              GradContext &context) noexcept
{
    if (context._M_operand_count < 2u || output._M_count == 0u)
        return Status::BAD_LAYOUT;
    const Trainable &pred = context._M_operands[0];
    const Trainable &target = context._M_operands[1];
    const int32_t seed = grad_output.grad_at (0);
    const uint32_t count = pred._M_count;
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t diff = pred.code_at (i) - target.code_at (i);
        const int32_t d_q15 = diff * _S_code_to_q15_scale;
        const int32_t base = div_round (2 * d_q15, static_cast<int32_t> (count));
        if (pred.add_grad (i, q15_mul (base, seed)))
            ++context._M_saturation_events;
    }
    return Status::OK;
}

/**
 * @brief Mean squared error over int8-code operands: L = mean((p - t)^2).
 *
 * Loss accumulates in int64 Q30 and is reported in Q7 (magnitudes up to 4.0
 * cannot hold a Q15 lane).  Pool cost: the 8-byte scalar node, nothing saved
 * (backward recomputes the diffs from the operand code streams, stride-1).
 */
[[nodiscard]] inline Status
mse_forward (Pool &pool, Tape *const tape, const Trainable &pred,
             const Trainable &target, LossResult &out) noexcept
{
    const Status pair = loss_pair_ok (pred, target);
    if (!status_ok (pair))
        return pair;
    int64_t total_q30 = 0;
    for (uint32_t i = 0; i < pred._M_count; ++i)
    {
        const int32_t diff = pred.code_at (i) - target.code_at (i);
        const int32_t d_q15 = diff * _S_code_to_q15_scale;
        total_q30 += static_cast<int64_t> (d_q15) * d_q15;
    }
    const Status finish = finish_loss (
        pool, q30_sumsq_mean_to_q7 (total_q30, pred._M_count), out);
    if (!status_ok (finish))
        return finish;
    const Trainable operands[2] = { pred, target };
    return record_loss (tape, &mse_backward, _S_op_mse, out, operands, 2u,
                        nullptr, 0u);
}

/* ======================================================================== */
/* MAE and Huber                                                            */
/* ======================================================================== */

/** @brief dL/dpred = seed * sign (p - t) / n for the absolute loss. */
[[nodiscard]] inline Status
mae_backward (const Trainable &output, const Trainable &grad_output,
              GradContext &context) noexcept
{
    if (context._M_operand_count < 2u || output._M_count == 0u)
        return Status::BAD_LAYOUT;
    const Trainable &pred = context._M_operands[0];
    const Trainable &target = context._M_operands[1];
    const int32_t seed = grad_output.grad_at (0);
    const int32_t unit
        = div_round (_S_q15_one, static_cast<int32_t> (pred._M_count));
    for (uint32_t i = 0; i < pred._M_count; ++i)
    {
        const int32_t diff = pred.code_at (i) - target.code_at (i);
        const int32_t sign = diff > 0 ? unit : (diff < 0 ? -unit : 0);
        if (pred.add_grad (i, q15_mul (sign, seed)))
            ++context._M_saturation_events;
    }
    return Status::OK;
}

/** @brief Mean absolute error, Q7 loss = mean(|p - t|); lane domain is exact. */
[[nodiscard]] inline Status
mae_forward (Pool &pool, Tape *const tape, const Trainable &pred,
             const Trainable &target, LossResult &out) noexcept
{
    const Status pair = loss_pair_ok (pred, target);
    if (!status_ok (pair))
        return pair;
    int64_t total_q15 = 0;
    for (uint32_t i = 0; i < pred._M_count; ++i)
    {
        const int32_t diff = pred.code_at (i) - target.code_at (i);
        const int32_t d_q15 = diff * _S_code_to_q15_scale;
        total_q15 += d_q15 < 0 ? -static_cast<int64_t> (d_q15) : d_q15;
    }
    const Status finish
        = finish_loss (pool, q15_sum_mean_to_q7 (total_q15, pred._M_count), out);
    if (!status_ok (finish))
        return finish;
    const Trainable operands[2] = { pred, target };
    return record_loss (tape, &mae_backward, _S_op_mae, out, operands, 2u,
                        nullptr, 0u);
}

/** @brief Huber transition: delta inside the quadratic arm, sign outside. */
[[nodiscard]] inline Status
huber_backward (const Trainable &output, const Trainable &grad_output,
                GradContext &context) noexcept
{
    if (context._M_operand_count < 2u || context._M_saved_count < 1u
        || output._M_count == 0u)
        return Status::BAD_LAYOUT;
    const Trainable &pred = context._M_operands[0];
    const Trainable &target = context._M_operands[1];
    const int32_t *const meta = context.saved_as<int32_t> (0);
    if (meta == nullptr)
        return Status::BAD_LAYOUT;
    const int32_t delta = meta[0];
    const int32_t seed = grad_output.grad_at (0);
    for (uint32_t i = 0; i < pred._M_count; ++i)
    {
        const int32_t diff = pred.code_at (i) - target.code_at (i);
        const int32_t d_q15 = diff * _S_code_to_q15_scale;
        const int32_t magnitude = d_q15 < 0 ? -d_q15 : d_q15;
        const int32_t raw
            = magnitude <= delta ? d_q15
                                 : (d_q15 > 0 ? delta : -delta);
        const int32_t base
            = div_round (raw, static_cast<int32_t> (pred._M_count));
        if (pred.add_grad (i, q15_mul (base, seed)))
            ++context._M_saturation_events;
    }
    return Status::OK;
}

/**
 * @brief Huber loss: 0.5 d^2 while |d| <= delta, else delta (|d| - delta/2).
 *
 * The arms are accumulated in one int64 Q30 total (delta * (2|d| - delta) / 2
 * for the linear arm keeps it exact), so the report is a single Q7 mean.
 * Pool cost: 8-byte scalar node + a 4-byte delta save.
 */
[[nodiscard]] inline Status
huber_forward (Pool &pool, Tape *const tape, const Trainable &pred,
               const Trainable &target, const HuberParams &params,
               LossResult &out) noexcept
{
    const Status pair = loss_pair_ok (pred, target);
    if (!status_ok (pair))
        return pair;
    const int32_t delta = q15_clamp (params._M_delta_q15);
    int64_t total_q30 = 0;
    for (uint32_t i = 0; i < pred._M_count; ++i)
    {
        const int32_t diff = pred.code_at (i) - target.code_at (i);
        const int32_t d_q15 = diff * _S_code_to_q15_scale;
        const int32_t magnitude = d_q15 < 0 ? -d_q15 : d_q15;
        if (magnitude <= delta)
            total_q30 += (static_cast<int64_t> (d_q15) * d_q15) >> 1;
        else
            total_q30
                += (static_cast<int64_t> (delta) * (2 * magnitude - delta)) >> 1;
    }
    const Status finish
        = finish_loss (pool, q30_sumsq_mean_to_q7 (total_q30, pred._M_count),
                       out);
    if (!status_ok (finish))
        return finish;
    Saved saved;
    int32_t *delta_slot = nullptr;
    if (tape != nullptr)
    {
        const Status save = tape->save<int32_t> (1u, saved, delta_slot);
        if (!status_ok (save))
            return save;
        delta_slot[0] = delta;
    }
    const Trainable operands[2] = { pred, target };
    return record_loss (tape, &huber_backward, _S_op_huber, out, operands, 2u,
                        tape != nullptr ? &saved : nullptr,
                        tape != nullptr ? 1u : 0u);
}

/* ======================================================================== */
/* Quantile loss                                                            */
/* ======================================================================== */

/** @brief dL/dpred = seed * tau / n above the pinball knee, (tau - 1) / n below. */
[[nodiscard]] inline Status
quantile_backward (const Trainable &output, const Trainable &grad_output,
                   GradContext &context) noexcept
{
    if (context._M_operand_count < 2u || context._M_saved_count < 1u
        || output._M_count == 0u)
        return Status::BAD_LAYOUT;
    const Trainable &pred = context._M_operands[0];
    const Trainable &target = context._M_operands[1];
    const int32_t *const meta = context.saved_as<int32_t> (0);
    if (meta == nullptr)
        return Status::BAD_LAYOUT;
    const int32_t tau = meta[0];
    const int32_t seed = grad_output.grad_at (0);
    const int32_t above = div_round (tau, static_cast<int32_t> (pred._M_count));
    const int32_t below = div_round (tau - _S_q15_one,
                                     static_cast<int32_t> (pred._M_count));
    for (uint32_t i = 0; i < pred._M_count; ++i)
    {
        const int32_t diff = pred.code_at (i) - target.code_at (i);
        const int32_t slope = diff > 0 ? above : (diff < 0 ? below : 0);
        if (pred.add_grad (i, q15_mul (slope, seed)))
            ++context._M_saturation_events;
    }
    return Status::OK;
}

/** @brief Pinball loss mean(max(tau d, (tau-1) d)) with d = p - t, tau in Q15. */
[[nodiscard]] inline Status
quantile_forward (Pool &pool, Tape *const tape, const Trainable &pred,
                  const Trainable &target, const QuantileParams &params,
                  LossResult &out) noexcept
{
    const Status pair = loss_pair_ok (pred, target);
    if (!status_ok (pair))
        return pair;
    const int32_t tau = q15_clamp (params._M_tau_q15);
    const int32_t tau_minus_one = tau - _S_q15_one;
    int64_t total_q15 = 0;
    for (uint32_t i = 0; i < pred._M_count; ++i)
    {
        const int32_t diff = pred.code_at (i) - target.code_at (i);
        const int64_t d_q15 = static_cast<int64_t> (diff) * _S_code_to_q15_scale;
        const int64_t up = d_q15 * tau >> 15;
        const int64_t down = d_q15 * tau_minus_one >> 15;
        total_q15 += up > down ? up : down;
    }
    const Status finish
        = finish_loss (pool, q15_sum_mean_to_q7 (total_q15, pred._M_count), out);
    if (!status_ok (finish))
        return finish;
    Saved saved;
    int32_t *tau_slot = nullptr;
    if (tape != nullptr)
    {
        const Status save = tape->save<int32_t> (1u, saved, tau_slot);
        if (!status_ok (save))
            return save;
        tau_slot[0] = tau;
    }
    const Trainable operands[2] = { pred, target };
    return record_loss (tape, &quantile_backward, _S_op_quantile, out, operands,
                        2u, tape != nullptr ? &saved : nullptr,
                        tape != nullptr ? 1u : 0u);
}

/* ======================================================================== */
/* Binary cross-entropy, with-logits form                                   */
/* ======================================================================== */

/**
 * @brief dL/dx = seed * (sigmoid(x) - t) / n, recomputed from the logits.
 *
 * The stable with-logits backward never evaluates a log of a stored
 * probability; it re-runs q15_sigmoid on the operand's own codes, so nothing
 * but the two code streams (both operands) is needed and nothing is saved.
 */
[[nodiscard]] inline Status
bce_backward (const Trainable &output, const Trainable &grad_output,
              GradContext &context) noexcept
{
    if (context._M_operand_count < 2u || output._M_count == 0u)
        return Status::BAD_LAYOUT;
    const Trainable &logits = context._M_operands[0];
    const Trainable &target = context._M_operands[1];
    const int32_t seed = grad_output.grad_at (0);
    for (uint32_t i = 0; i < logits._M_count; ++i)
    {
        const int32_t x = logits.q15_at (i);
        const int32_t t = target.q15_at (i);
        const int32_t base = div_round (q15_sigmoid (x) - t,
                                       static_cast<int32_t> (logits._M_count));
        if (logits.add_grad (i, q15_mul (base, seed)))
            ++context._M_saturation_events;
    }
    return Status::OK;
}

/**
 * @brief mean((1 - t) softplus(x) + t softplus(-x)) with softplus in Q7.
 *
 * q7_log1p_exp keeps the arms stable for |x| up to the exp clamp, and the
 * target weighting happens after each arm has reached Q7, so no term divides
 * and no log is ever taken of a stored zero.
 */
[[nodiscard]] inline Status
bce_with_logits_forward (Pool &pool, Tape *const tape, const Trainable &logits,
                         const Trainable &target, LossResult &out) noexcept
{
    const Status pair = loss_pair_ok (logits, target);
    if (!status_ok (pair))
        return pair;
    int64_t total_q7 = 0;
    for (uint32_t i = 0; i < logits._M_count; ++i)
    {
        const int32_t x = logits.q15_at (i);
        const int32_t t = target.q15_at (i);
        const int32_t arm_pos = q7_log1p_exp (x);
        const int32_t arm_neg = q7_log1p_exp (-x);
        const int64_t weighted = static_cast<int64_t> (arm_pos) * (_S_q15_one - t)
                                 + static_cast<int64_t> (arm_neg) * t;
        total_q7 += weighted >> 15;
    }
    const Status finish
        = finish_loss (pool, div_round (static_cast<int32_t> (total_q7),
                                        static_cast<int32_t> (logits._M_count)),
                       out);
    if (!status_ok (finish))
        return finish;
    const Trainable operands[2] = { logits, target };
    return record_loss (tape, &bce_backward, _S_op_bce, out, operands, 2u,
                        nullptr, 0u);
}

/* ======================================================================== */
/* Cross-entropy over a probability buffer                                  */
/* ======================================================================== */

/** @brief Index-domain CE backward: dL/dp_y = seed * (-1 / p_y), saturated. */
[[nodiscard]] inline Status
cross_entropy_index_backward (const Trainable & /*output*/,
                              const Trainable &grad_output,
                              GradContext &context) noexcept
{
    if (context._M_operand_count < 1u || context._M_saved_count < 1u)
        return Status::BAD_LAYOUT;
    const Trainable &probs = context._M_operands[0];
    const TargetMeta *const meta = context.saved_as<TargetMeta> (0);
    if (meta == nullptr)
        return Status::BAD_LAYOUT;
    const int32_t seed = grad_output.grad_at (0);
    const uint16_t target = meta->_M_target;
    for (uint32_t i = 0; i < probs._M_count; ++i)
    {
        const int32_t grad = i == target ? q15_mul (-_S_q15_max, seed) : 0;
        if (probs.add_grad (i, grad))
            ++context._M_saturation_events;
    }
    return Status::OK;
}

/** @brief Dense-target CE backward: dL/dp_c = seed * (-t_c / p_c), saturated. */
[[nodiscard]] inline Status
cross_entropy_dense_backward (const Trainable & /*output*/,
                              const Trainable &grad_output,
                              GradContext &context) noexcept
{
    if (context._M_operand_count < 2u)
        return Status::BAD_LAYOUT;
    const Trainable &probs = context._M_operands[0];
    const Trainable &target = context._M_operands[1];
    const int32_t seed = grad_output.grad_at (0);
    for (uint32_t i = 0; i < probs._M_count; ++i)
    {
        const int32_t p = probs.q15_at (i);
        const int32_t t = target.q15_at (i);
        const int32_t ratio = p > 0 ? q15_div (-t, p) : -_S_q15_max;
        if (probs.add_grad (i, q15_mul (ratio, seed)))
            ++context._M_saturation_events;
    }
    return Status::OK;
}

/** @brief Shared loss value for both cross-entropy forms: sum -t_c log p_c. */
[[nodiscard]] inline Status
cross_entropy_value (const Trainable &probs, const Trainable &target,
                     int32_t &loss_q7) noexcept
{
    int64_t total_q7 = 0;
    for (uint32_t i = 0; i < probs._M_count; ++i)
    {
        const int32_t t = target.q15_at (i);
        if (t == 0)
            continue;
        const int32_t neg_log = q7_neg_log_q15 (probs.q15_at (i));
        total_q7 += static_cast<int64_t> (neg_log) * t >> 15;
    }
    loss_q7 = static_cast<int32_t> (total_q7);
    return Status::OK;
}

/**
 * @brief Cross-entropy of a probability buffer against one class.
 *
 * p is the softmax output (codes in [0, 127]); the log runs through
 * q7_log_q15, whose p <= 0 rail is finite, so a zero class probability yields
 * the sentinel -log cost (Q7 rail) instead of a trap.  The gradient -1/p_y is
 * unbounded by nature and saturates at the Q15 rails; document-and-saturate is
 * the only defined behaviour a lane can carry.
 * Pool cost: 8-byte node + 8-byte TargetMeta save.
 */
[[nodiscard]] inline Status
cross_entropy_forward (Pool &pool, Tape *const tape, const Trainable &probs,
                       const uint16_t target, LossResult &out) noexcept
{
    if (!status_ok (probs.validate ()))
        return Status::BAD_LAYOUT;
    if (probs._M_count == 0u || target >= probs._M_count)
        return Status::OUT_OF_RANGE;
    /* Index form: only the target slot contributes, -log p_y in Q7. */
    const int32_t loss_q7 = q7_neg_log_q15 (probs.q15_at (target));
    const Status finish = finish_loss (pool, loss_q7, out);
    if (!status_ok (finish))
        return finish;
    Saved saved;
    TargetMeta *meta = nullptr;
    if (tape != nullptr)
    {
        const Status save = tape->save<TargetMeta> (1u, saved, meta);
        if (!status_ok (save))
            return save;
        meta->_M_classes = static_cast<uint16_t> (probs._M_count);
        meta->_M_target = target;
        meta->_M_margin_q7 = 0;
    }
    const Trainable operands[1] = { probs };
    return record_loss (tape, &cross_entropy_index_backward,
                        _S_op_cross_entropy, out, operands, 1u,
                        tape != nullptr ? &saved : nullptr,
                        tape != nullptr ? 1u : 0u);
}

/**
 * @brief Cross-entropy against a dense probability target (soft labels).
 *
 * sum -t_c log p_c without normalization: for a one-hot target this equals the
 * index form, and for a soft target it equals the label-smoothed loss with
 * eps = 0.  Pool cost: 8-byte node, no saves (both streams are operands).
 */
[[nodiscard]] inline Status
soft_cross_entropy_forward (Pool &pool, Tape *const tape,
                            const Trainable &probs, const Trainable &target,
                            LossResult &out) noexcept
{
    const Status pair = loss_pair_ok (probs, target);
    if (!status_ok (pair))
        return pair;
    int32_t loss_q7 = 0;
    const Status value = cross_entropy_value (probs, target, loss_q7);
    if (!status_ok (value))
        return value;
    const Status finish = finish_loss (pool, loss_q7, out);
    if (!status_ok (finish))
        return finish;
    const Trainable operands[2] = { probs, target };
    return record_loss (tape, &cross_entropy_dense_backward,
                        _S_op_cross_entropy, out, operands, 2u, nullptr, 0u);
}

/* ======================================================================== */
/* NLL over a log-probability buffer                                        */
/* ======================================================================== */

/** @brief Encode a Q7 log-probability into a log-prob code (1/32 nat step). */
[[nodiscard]] inline constexpr int8_t
logprob_code_from_q7 (const int32_t log_q7) noexcept
{
    return _S_i8_code (div_round (log_q7, _S_log_step_q7));
}

/** @brief Decode a log-prob code into the exact Q7 log-probability. */
[[nodiscard]] inline constexpr int32_t
logprob_q7_at (const Trainable &log_probs, const uint32_t index) noexcept
{
    return log_probs.code_at (index) * _S_log_step_q7;
}

/** @brief NLL backward: dL/dlogit_c = seed * (-1) at the target class. */
[[nodiscard]] inline Status
nll_backward (const Trainable & /*output*/, const Trainable &grad_output,
              GradContext &context) noexcept
{
    if (context._M_operand_count < 1u || context._M_saved_count < 1u)
        return Status::BAD_LAYOUT;
    const Trainable &log_probs = context._M_operands[0];
    const TargetMeta *const meta = context.saved_as<TargetMeta> (0);
    if (meta == nullptr)
        return Status::BAD_LAYOUT;
    const int32_t seed = grad_output.grad_at (0);
    const uint16_t target = meta->_M_target;
    for (uint32_t i = 0; i < log_probs._M_count; ++i)
    {
        const int32_t grad
            = i == target ? q15_mul (-_S_q15_max, seed) : 0;
        if (log_probs.add_grad (i, grad))
            ++context._M_saturation_events;
    }
    return Status::OK;
}

/**
 * @brief Negative log-likelihood over a log-probability buffer.
 *
 * The buffer is a Trainable whose codes are log-probabilities in units of
 * 1/32 nat (_S_log_step_q7 Q7 per code): the Q15 lane domain cannot hold a
 * magnitude of one nat, let alone ten, while the code domain reaches
 * 127/32 = ~4 nats per side and the Q7 product recovers the full range.
 * L = -logprob_y exactly in Q7 -- no log is evaluated at run time, which is
 * the reason a log-softmax producer and this consumer agree bit for bit.
 * Pool cost: 8-byte node + 8-byte TargetMeta save.
 */
[[nodiscard]] inline Status
nll_forward (Pool &pool, Tape *const tape, const Trainable &log_probs,
             const uint16_t target, LossResult &out) noexcept
{
    if (!status_ok (log_probs.validate ()))
        return Status::BAD_LAYOUT;
    if (log_probs._M_count == 0u || target >= log_probs._M_count)
        return Status::OUT_OF_RANGE;
    const int32_t loss_q7 = -logprob_q7_at (log_probs, target);
    const Status finish = finish_loss (pool, loss_q7, out);
    if (!status_ok (finish))
        return finish;
    Saved saved;
    TargetMeta *meta = nullptr;
    if (tape != nullptr)
    {
        const Status save = tape->save<TargetMeta> (1u, saved, meta);
        if (!status_ok (save))
            return save;
        meta->_M_classes = static_cast<uint16_t> (log_probs._M_count);
        meta->_M_target = target;
        meta->_M_margin_q7 = 0;
    }
    const Trainable operands[1] = { log_probs };
    return record_loss (tape, &nll_backward, _S_op_nll, out, operands, 1u,
                        tape != nullptr ? &saved : nullptr,
                        tape != nullptr ? 1u : 0u);
}

/* ======================================================================== */
/* Cosine loss                                                              */
/* ======================================================================== */

/**
 * @brief dL/dpred in the module gradient convention (lane = dL/dp_real * 32768).
 *
 * With kappa = 258/32768 (code -> real),
 *   d(1 - cos)/dp_real_i = cos * p_real_i / Np_real^3 - t_real_i / (Np * Nt),
 * and every real quantity is code * kappa.  Folding 32768 * kappa^-1 and
 * 32768 * kappa^-2 into two integer gains keeps the whole pass in int64.
 */
[[nodiscard]] inline Status
cosine_backward (const Trainable &output, const Trainable &grad_output,
                 GradContext &context) noexcept
{
    if (context._M_operand_count < 2u || output._M_count == 0u)
        return Status::BAD_LAYOUT;
    const Trainable &pred = context._M_operands[0];
    const Trainable &target = context._M_operands[1];
    const int32_t seed = grad_output.grad_at (0);
    const uint32_t count = pred._M_count;
    int32_t dot_code = 0;
    int32_t sq_pred = 0;
    int32_t sq_target = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t p = pred.code_at (i);
        const int32_t t = target.code_at (i);
        dot_code += p * t;
        sq_pred += p * p;
        sq_target += t * t;
    }
    if (sq_pred == 0 || sq_target == 0)
        return Status::OK; /* a zero vector has no cosine gradient to route */
    const int64_t norm_pred = _S_isqrt (static_cast<uint32_t> (sq_pred));
    const int64_t norm_target = _S_isqrt (static_cast<uint32_t> (sq_target));
    const int64_t cos_q15 = q15_clamp (static_cast<int32_t> (
        (static_cast<int64_t> (dot_code) << 15) / (norm_pred * norm_target)));
    /* gain1 = 32768 / kappa, gain2 = 32768 / kappa^2, both as int64. */
    constexpr int64_t gain1 = 4161790;
    const int64_t gain2 = (gain1 * gain1) >> 15;
    for (uint32_t i = 0; i < count; ++i)
    {
        const int64_t p = pred.code_at (i);
        const int64_t t = target.code_at (i);
        const int64_t term_align = gain1 * t / (norm_pred * norm_target);
        const int64_t term_norm
            = gain2 * cos_q15 * p / (32768 * norm_pred * norm_pred * norm_pred);
        const int64_t summed = term_norm - term_align;
        const int32_t narrowed = q15_clamp (static_cast<int32_t> (
            summed > _S_q15_max
                ? _S_q15_max
                : (summed < _S_q15_min ? _S_q15_min : summed)));
        if (pred.add_grad (i, q15_mul (narrowed, seed)))
            ++context._M_saturation_events;
    }
    return Status::OK;
}

/**
 * @brief Cosine embedding loss L = 1 - cos(pred, target), summed over the
 *        element count (the two vectors are one sample, not a batch).
 *
 * |p|, |t| and cos are computed in the code domain where every magnitude is
 * an integer (127^2 * n fits int32), and only the final dimensionless ratio
 * is widened -- so no intermediate norm exceeds a Q15 lane.  A zero-norm
 * operand yields zero gradient (guarded before any division), which is the
 * params.eps convention expressed without a float epsilon.
 * Pool cost: 8-byte scalar node.
 */
[[nodiscard]] inline Status
cosine_similarity_forward (Pool &pool, Tape *const tape,
                           const Trainable &pred, const Trainable &target,
                           const CosineParams &params, LossResult &out) noexcept
{
    const Status pair = loss_pair_ok (pred, target);
    if (!status_ok (pair))
        return pair;
    (void)params; /* the guard is structural: a zero sum is checked, not offset */
    int32_t dot_code = 0;
    int32_t sq_pred = 0;
    int32_t sq_target = 0;
    for (uint32_t i = 0; i < pred._M_count; ++i)
    {
        const int32_t p = pred.code_at (i);
        const int32_t t = target.code_at (i);
        dot_code += p * t;
        sq_pred += p * p;
        sq_target += t * t;
    }
    int32_t cos_q15 = 0;
    if (sq_pred != 0 && sq_target != 0)
    {
        const int64_t denominator = static_cast<int64_t> (
            static_cast<int32_t> (_S_isqrt (static_cast<uint32_t> (sq_pred))))
                                    * static_cast<int32_t> (
                                        _S_isqrt (static_cast<uint32_t> (sq_target)));
        const int64_t raw = (static_cast<int64_t> (dot_code) << 15) / denominator;
        cos_q15 = q15_clamp (static_cast<int32_t> (raw));
    }
    const int32_t loss_q7 = _S_q7_one - (cos_q15 >> 8);
    const Status finish = finish_loss (pool, loss_q7, out);
    if (!status_ok (finish))
        return finish;
    const Trainable operands[2] = { pred, target };
    return record_loss (tape, &cosine_backward, _S_op_cosine, out, operands, 2u,
                        nullptr, 0u);
}

/* ======================================================================== */
/* Hinge and multi-class hinge                                              */
/* ======================================================================== */

/** @brief Binary hinge backward: dL/dscore = seed * (-sign(target)) / n. */
[[nodiscard]] inline Status
hinge_backward (const Trainable &output, const Trainable &grad_output,
                GradContext &context) noexcept
{
    if (context._M_operand_count < 2u || context._M_saved_count < 1u
        || output._M_count == 0u)
        return Status::BAD_LAYOUT;
    const Trainable &scores = context._M_operands[0];
    const Trainable &labels = context._M_operands[1];
    const int32_t *const margin = context.saved_as<int32_t> (0);
    if (margin == nullptr)
        return Status::BAD_LAYOUT;
    const int32_t seed = grad_output.grad_at (0);
    const int32_t unit
        = div_round (_S_q15_one, static_cast<int32_t> (scores._M_count));
    for (uint32_t i = 0; i < scores._M_count; ++i)
    {
        const int32_t y = labels.code_at (i) >= 0 ? _S_q15_max : -_S_q15_max;
        const int32_t product_q7
            = q15_mul (y, scores.q15_at (i)) >> 8;
        const int32_t grad
            = product_q7 < *margin ? q15_mul (-y, unit) : 0;
        if (scores.add_grad (i, q15_mul (grad, seed)))
            ++context._M_saturation_events;
    }
    return Status::OK;
}

/**
 * @brief Binary hinge: mean(max(0, margin - y * s)) with y = sign(label).
 *
 * The margin is a Q7 word because a 1.0 margin is not a Q15 lane; the
 * per-element product y*s is narrowed to Q7 before the comparison so both
 * sides live in one domain.  Pool cost: 8-byte node + 4-byte margin save.
 */
[[nodiscard]] inline Status
hinge_loss_forward (Pool &pool, Tape *const tape, const Trainable &scores,
                    const Trainable &labels, const HingeParams &params,
                    LossResult &out) noexcept
{
    const Status pair = loss_pair_ok (scores, labels);
    if (!status_ok (pair))
        return pair;
    const int32_t margin = params._M_margin_q7;
    int64_t total_q7 = 0;
    uint32_t violations = 0u;
    for (uint32_t i = 0; i < scores._M_count; ++i)
    {
        const int32_t y = labels.code_at (i) >= 0 ? _S_q15_max : -_S_q15_max;
        const int32_t product_q7 = q15_mul (y, scores.q15_at (i)) >> 8;
        if (product_q7 < margin)
        {
            total_q7 += margin - product_q7;
            ++violations;
        }
    }
    const int32_t loss_q7 = violations == 0u
                                ? 0
                                : div_round (static_cast<int32_t> (total_q7),
                                             static_cast<int32_t> (
                                                 scores._M_count));
    const Status finish = finish_loss (pool, loss_q7, out);
    if (!status_ok (finish))
        return finish;
    Saved saved;
    int32_t *margin_slot = nullptr;
    if (tape != nullptr)
    {
        const Status save = tape->save<int32_t> (1u, saved, margin_slot);
        if (!status_ok (save))
            return save;
        margin_slot[0] = margin;
    }
    const Trainable operands[2] = { scores, labels };
    return record_loss (tape, &hinge_backward, _S_op_hinge, out, operands, 2u,
                        tape != nullptr ? &saved : nullptr,
                        tape != nullptr ? 1u : 0u);
}

/** @brief Multi-class hinge backward: +unit per violator, -k*unit at the gold. */
[[nodiscard]] inline Status
multiclass_hinge_backward (const Trainable & /*output*/,
                           const Trainable &grad_output,
                           GradContext &context) noexcept
{
    if (context._M_operand_count < 1u || context._M_saved_count < 1u)
        return Status::BAD_LAYOUT;
    const Trainable &scores = context._M_operands[0];
    const TargetMeta *const meta = context.saved_as<TargetMeta> (0);
    if (meta == nullptr)
        return Status::BAD_LAYOUT;
    const uint16_t target = meta->_M_target;
    if (target >= scores._M_count)
        return Status::BAD_LAYOUT;
    const int32_t margin = meta->_M_margin_q7;
    const int32_t seed = grad_output.grad_at (0);
    const int32_t unit
        = div_round (_S_q15_one, static_cast<int32_t> (scores._M_count));
    int32_t violations = 0;
    for (uint32_t i = 0; i < scores._M_count; ++i)
    {
        if (i == target)
            continue;
        const int32_t s_q7
            = q15_mul (scores.q15_at (i) - scores.q15_at (target), _S_q15_max)
              >> 8;
        const bool violating = margin + s_q7 > 0;
        if (violating)
        {
            ++violations;
            if (scores.add_grad (i, q15_mul (unit, seed)))
                ++context._M_saturation_events;
        }
    }
    if (violations > 0)
    {
        const int32_t gold = q15_mul (-violations * unit, seed);
        if (scores.add_grad (target, gold))
            ++context._M_saturation_events;
    }
    return Status::OK;
}

/**
 * @brief Crammer-Singer style multi-class hinge over one score vector:
 *        sum_{c != y} max(0, margin + s_c - s_y).
 *
 * Margins and score differences are Q7 so the unit margin fits; the gradient
 * is normalized by the class count (each violator gets +1/C at the Q15 rail,
 * the gold class collects -k/C) because the raw +/-1 gradient saturates a
 * lane and hides the violation count.
 * Pool cost: 8-byte node + 8-byte TargetMeta save.
 */
[[nodiscard]] inline Status
multiclass_hinge_forward (Pool &pool, Tape *const tape,
                          const Trainable &scores, const uint16_t target,
                          const HingeParams &params, LossResult &out) noexcept
{
    if (!status_ok (scores.validate ()))
        return Status::BAD_LAYOUT;
    if (scores._M_count < 2u || target >= scores._M_count)
        return Status::OUT_OF_RANGE;
    const int32_t margin = params._M_margin_q7;
    const int32_t gold_q15 = scores.q15_at (target);
    int64_t total_q7 = 0;
    for (uint32_t i = 0; i < scores._M_count; ++i)
    {
        if (i == target)
            continue;
        const int32_t diff_q7
            = q15_mul (scores.q15_at (i) - gold_q15, _S_q15_max) >> 8;
        const int32_t term = margin + diff_q7;
        if (term > 0)
            total_q7 += term;
    }
    const Status finish = finish_loss (pool, static_cast<int32_t> (total_q7),
                                       out);
    if (!status_ok (finish))
        return finish;
    Saved saved;
    TargetMeta *meta = nullptr;
    if (tape != nullptr)
    {
        const Status save = tape->save<TargetMeta> (1u, saved, meta);
        if (!status_ok (save))
            return save;
        meta->_M_classes = static_cast<uint16_t> (scores._M_count);
        meta->_M_target = target;
        meta->_M_margin_q7 = margin;
    }
    const Trainable operands[1] = { scores };
    return record_loss (tape, &multiclass_hinge_backward,
                        _S_op_hinge_multiclass, out, operands, 1u,
                        tape != nullptr ? &saved : nullptr,
                        tape != nullptr ? 1u : 0u);
}
} // namespace zzs::mllee::train
