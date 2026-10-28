/* Copyright(c) 2026 Tommicord
 *
 * Permission for use, copying and distribution is granted under the same
 * MIT-style terms as the rest of the mllee module (see mllee/core.hpp).
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND. */

#pragma once

#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_loss.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_simd.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Knowledge distillation: temperature softening, KL, alpha blend.
 *
 * A teacher logits buffer is softened by the reciprocal temperature.  The
 * temperature itself (T >= 1) is not a Q15 lane, so every API here takes
 * inv_temperature = 1/T in Q15; where the convention needs a T or T^2
 * multiply, it is applied as a division by inv_temperature (saturating) or
 * as an explicit power-of-two shift (kl_shift), never as an unrepresentable
 * lane scalar.
 *
 * The comparison runs on log-probabilities in Q7:
 *
 *     KL = sum_c target_c * (log target_c - log student_c)
 *
 * and the zero-probability guard is structural, not a division check: terms
 * with target_c == 0 are skipped (their limit is zero), and a zero student
 * probability reaches q7_log_q15's finite -inf sentinel so the term
 * saturates at its rail.  Nothing in this file divides by a probability.
 *
 * Backward routing follows the tape contract (mllee/train_graph.hpp): the
 * student operand receives seed * invT * (p_s - p_t) (sum-form KL, matching
 * the forward's summed Q7 value) and, for the combined op, the alpha-blended
 * sum of the CE and KL arms.  When a trainable teacher is passed with the
 * route flag set, the teacher receives the analytic softmax derivative
 * invT * p_t * (delta_c - KL), also saturating.
 *
 * Pool cost: the pure helpers (soften_teacher, log_softmax_q7) carve nothing
 * -- caller-owned lane buffers.  kl_divergence_forward and distill_forward
 * each cost the shared 8-byte loss scalar node plus one 16-byte DistillMeta
 * save.  Backward recomputes both softmaxes from the operand code streams
 * (two stride-1 passes over C <= 255 classes), which trades one extra exp
 * sweep for 4*C bytes of pool the saved probabilities would need.
 */
namespace zzs::mllee::train
{
/** @brief Distillation op ids (continuing the loss family numbering). */
inline constexpr uint32_t _S_op_kl = 0x4C20u;
inline constexpr uint32_t _S_op_distill = 0x4C21u;

/** @brief Largest class count the Q7 log-prob lane buffer can span (-256..0). */
inline constexpr uint32_t _S_distill_max_classes = 255u;

/**
 * @brief Saved config for the KL and combined distill backwards.
 *
 * kl_shift applies the conventional T^2 gradient scale as a power of two:
 * T = 2 -> shift 2, T = 4 -> shift 4, keeping the multiplier representable
 * where a Q15 T^2 lane would not be.
 */
struct DistillMeta
{
    int32_t _M_inv_temperature_q15 = _S_q15_max;
    int32_t _M_alpha_q15 = 16384;
    uint16_t _M_label = 0u;
    uint8_t _M_classes = 0u;
    uint8_t _M_route_teacher = 0u;
    uint8_t _M_kl_shift = 0u;
    uint8_t _M_pad = 0u;
};

static_assert (sizeof (DistillMeta) == 16u, "distill meta is two words, a halfword and four bytes padded to alignment");

/** @brief A combined distillation evaluation: total, soft and hard arms. */
struct DistillResult
{
    LossResult _M_total;
    int32_t _M_kl_q7 = 0;
    int32_t _M_hard_q7 = 0;
};

static_assert (sizeof (DistillResult) == sizeof (LossResult) + 8u,
               "distill result is LossResult plus two words");

/** @brief Running log-sum-exp state for one softmax sweep (Q15 max, Q7 lse). */
struct SoftmaxState
{
    int32_t _M_max_q15 = 0;
    int32_t _M_lse_q7 = 0;
};

static_assert (sizeof (SoftmaxState) == 8u, "softmax state is two domain words");

/**
 * @brief Natural log of a positive integer in Q7, via mantissa normalization.
 *
 * The mantissa loop narrows value into the [0.5, 1) Q15 window the log atom
 * covers, and the exponent count contributes exact whole octaves of ln 2 in
 * Q7.  Iterative, no recursion, defined for every value >= 1.
 */
[[nodiscard]] inline constexpr int32_t
q7_log_pos_int (uint32_t value) noexcept
{
    if (value == 0u)
        return _S_q15_min;
    uint64_t mantissa = static_cast<uint64_t> (value) << 15;
    int32_t exponent = 0;
    while (mantissa >= _S_q15_one)
        {
            mantissa >>= 1;
            ++exponent;
        }
    while (mantissa < 16384u)
        {
            mantissa <<= 1;
            --exponent;
        }
    return q7_log_q15 (static_cast<int32_t> (mantissa))
           + exponent * _S_ln2_q7;
}

/**
 * @brief Widen teacher codes and scale by the reciprocal temperature.
 *
 * dest = round(teacher_code * 258 * invT / 32768): one codes_to_q15 widening
 * and one scale_lanes pass, both stride-1 and vectorized.
 */
inline Status
soften_teacher (const Trainable &teacher, const int32_t inv_temperature_q15,
                int16_t *const dest) noexcept
{
    if (!status_ok (teacher.validate ()))
        return Status::BAD_LAYOUT;
    if (dest == nullptr)
        return Status::BAD_LAYOUT;
    if (inv_temperature_q15 <= 0 || inv_temperature_q15 > _S_q15_one)
        return Status::OUT_OF_RANGE;
    train_simd::codes_to_q15 (teacher._M_code, dest, teacher._M_count);
    scale_lanes (dest, inv_temperature_q15, teacher._M_count);
    return Status::OK;
}

/** @brief Max + log-sum-exp of a Q15 lane vector; lse reported in Q7. */
[[nodiscard]] inline SoftmaxState
log_sum_exp_lanes (const int16_t *const logits_q15,
                   const uint32_t count) noexcept
{
    SoftmaxState state;
    if (count == 0u || logits_q15 == nullptr)
        return state;
    int32_t maximum = logits_q15[0];
    for (uint32_t i = 1; i < count; ++i)
        if (logits_q15[i] > maximum)
            maximum = logits_q15[i];
    uint64_t total = 0; /* sum of Q15 exp terms, each <= 32767, max term ~= 1 */
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<uint32_t> (
            q15_exp (static_cast<int32_t> (
                static_cast<int64_t> (logits_q15[i]) - maximum)));
    /* ln(sum exp(z - max)) = ln(total) - 15 ln 2 in Q7, then re-base by max. */
    state._M_max_q15 = maximum;
    state._M_lse_q7
        = (maximum >> 8) + q7_log_pos_int (static_cast<uint32_t> (total))
          - 15 * _S_ln2_q7;
    return state;
}

/**
 * @brief Log-softmax of a Q15 lane buffer into a Q7 lane buffer.
 *
 * dest[c] = (logit[c] >> 8) - lse_q7.  The saved lse is the full
 * max + ln sum exp(z - max), so the subtraction runs on the un-centered
 * logit (centering here would charge the max twice); centering stays inside
 * the lse itself, where it keeps every exp argument <= 0 (the representable
 * half of the exp path).  The Q7 result spans the full -log p range a class
 * probability can reach (down to -32768/128 = -256 nats at the rail).
 */
inline Status
log_softmax_q7 (const int16_t *const logits_q15, const uint32_t count,
                int16_t *const dest_q7, SoftmaxState &lse_out) noexcept
{
    if (logits_q15 == nullptr || dest_q7 == nullptr || count == 0u)
        return Status::BAD_LAYOUT;
    const SoftmaxState state = log_sum_exp_lanes (logits_q15, count);
    lse_out = state;
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t logit_q7 = static_cast<int32_t> (
            static_cast<int64_t> (logits_q15[i]) >> 8);
        dest_q7[i] = q15_to_i16_sat (logit_q7 - state._M_lse_q7);
    }
    return Status::OK;
}

/** @brief Softened (z * invT) Q15 logit of one code element. */
[[nodiscard]] inline constexpr int32_t
softened_logit_q15 (const Trainable &t, const uint32_t index,
                    const int32_t inv_temperature_q15) noexcept
{
    return q15_mul (t.q15_at (index), inv_temperature_q15);
}

/** @brief Softmax state of a code operand under the softened temperature. */
[[nodiscard]] inline SoftmaxState
softmax_state_from_codes (const Trainable &t,
                          const int32_t inv_temperature_q15) noexcept
{
    SoftmaxState state;
    if (t._M_count == 0u)
        return state;
    int32_t maximum = softened_logit_q15 (t, 0u, inv_temperature_q15);
    for (uint32_t i = 1; i < t._M_count; ++i)
    {
        const int32_t z = softened_logit_q15 (t, i, inv_temperature_q15);
        if (z > maximum)
            maximum = z;
    }
    uint64_t total = 0;
    for (uint32_t i = 0; i < t._M_count; ++i)
        total += static_cast<uint32_t> (
            q15_exp (static_cast<int32_t> (
                static_cast<int64_t> (
                    softened_logit_q15 (t, i, inv_temperature_q15))
                - maximum)));
    state._M_max_q15 = maximum;
    state._M_lse_q7
        = (maximum >> 8) + q7_log_pos_int (static_cast<uint32_t> (total))
          - 15 * _S_ln2_q7;
    return state;
}

/** @brief Q7 log-probability of one code operand element under @p state.
 *
 * The state's lse already carries the max (log_sum_exp form), so this reads
 * the un-centered softened logit; centering again would double-subtract it.
 */
[[nodiscard]] inline constexpr int32_t
log_prob_q7 (const Trainable &t, const uint32_t index,
             const int32_t inv_temperature_q15,
             const SoftmaxState &state) noexcept
{
    const int32_t logit_q7 = static_cast<int32_t> (
        static_cast<int64_t> (softened_logit_q15 (t, index, inv_temperature_q15))
        >> 8);
    return logit_q7 - state._M_lse_q7;
}

/** @brief Q15 softmax probability of one code operand element under @p state. */
[[nodiscard]] inline int32_t
prob_q15 (const Trainable &t, const uint32_t index,
          const int32_t inv_temperature_q15,
          const SoftmaxState &state) noexcept
{
    const int32_t lambda = log_prob_q7 (t, index, inv_temperature_q15, state);
    return q15_exp_q7 (lambda > 0 ? 0 : lambda);
}

/** @brief One summed KL sweep; skips zero targets and saturates zero students. */
[[nodiscard]] inline int32_t
kl_sum_q7 (const Trainable &student, const Trainable &teacher,
           const int32_t inv_temperature_q15, const SoftmaxState &student_state,
           const SoftmaxState &teacher_state) noexcept
{
    int64_t total_q7 = 0;
    for (uint32_t i = 0; i < teacher._M_count; ++i)
    {
        const int32_t t_q15 = prob_q15 (teacher, i, inv_temperature_q15,
                                        teacher_state);
        if (t_q15 <= 0)
            continue; /* zero target: the x*log(x) limit is the zero term */
        const int32_t delta
            = log_prob_q7 (teacher, i, inv_temperature_q15, teacher_state)
              - log_prob_q7 (student, i, inv_temperature_q15, student_state);
        total_q7 += (static_cast<int64_t> (delta) * t_q15) >> 15;
    }
    return static_cast<int32_t> (total_q7);
}

/** @brief KL backward: student gets seed * invT * (p_s - p_t), teacher optional. */
[[nodiscard]] inline Status
kl_backward (const Trainable & /*output*/, const Trainable &grad_output,
             GradContext &context) noexcept
{
    if (context._M_operand_count < 2u || context._M_saved_count < 1u)
        return Status::BAD_LAYOUT;
    const Trainable &student = context._M_operands[0];
    const Trainable &teacher = context._M_operands[1];
    const DistillMeta *const meta = context.saved_as<DistillMeta> (0);
    if (meta == nullptr)
        return Status::BAD_LAYOUT;
    const uint32_t count = student._M_count;
    if (teacher._M_count != count || count == 0u
        || count > _S_distill_max_classes)
        return Status::BAD_LAYOUT;
    const int32_t seed = grad_output.grad_at (0);
    const int32_t inv_t = meta->_M_inv_temperature_q15;
    const SoftmaxState student_state
        = softmax_state_from_codes (student, inv_t);
    const SoftmaxState teacher_state
        = softmax_state_from_codes (teacher, inv_t);
    const int32_t kl_q7
        = kl_sum_q7 (student, teacher, inv_t, student_state, teacher_state);
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t p_s
            = prob_q15 (student, i, inv_t, student_state);
        const int32_t p_t
            = prob_q15 (teacher, i, inv_t, teacher_state);
        const int32_t student_grad
            = q15_mul (q15_clamp (p_s - p_t), inv_t);
        if (student.add_grad (i, q15_mul (student_grad, seed)))
            ++context._M_saturation_events;
        if (meta->_M_route_teacher != 0u && teacher.has_grad ())
        {
            const int32_t delta
                = log_prob_q7 (teacher, i, inv_t, teacher_state)
                  - log_prob_q7 (student, i, inv_t, student_state);
            /* (delta - KL) is a Q7 real difference; widen to a lane (x256),
               saturating above 1.0 -- the documented lane limit. */
            const int32_t outer
                = q15_clamp (static_cast<int32_t> (
                    static_cast<int64_t> (delta - kl_q7) << 8));
            const int32_t teacher_grad
                = q15_mul (q15_mul (p_t, inv_t), outer);
            if (teacher.add_grad (i, q15_mul (teacher_grad, seed)))
                ++context._M_saturation_events;
        }
    }
    return Status::OK;
}

/**
 * @brief KL divergence between student and teacher logits buffers, Q7 value.
 *
 * sum_c p_t (log p_t - log p_s) with both sides softened by invT; the value
 * is NOT averaged over classes because C * 0.5 nats routinely exceeds one
 * Q15 lane -- the Q7 accumulator is what holds it (see train_loss.hpp).
 * Pool cost: 8-byte node + 16-byte DistillMeta save.
 */
[[nodiscard]] inline Status
kl_divergence_forward (Pool &pool, Tape *const tape, const Trainable &student,
                       const Trainable &teacher,
                       const int32_t inv_temperature_q15, LossResult &out,
                       const bool route_teacher = false) noexcept
{
    const Status pair = loss_pair_ok (student, teacher);
    if (!status_ok (pair))
        return pair;
    if (student._M_count > _S_distill_max_classes)
        return Status::OUT_OF_RANGE;
    if (inv_temperature_q15 <= 0 || inv_temperature_q15 > _S_q15_one)
        return Status::OUT_OF_RANGE;
    const SoftmaxState student_state
        = softmax_state_from_codes (student, inv_temperature_q15);
    const SoftmaxState teacher_state
        = softmax_state_from_codes (teacher, inv_temperature_q15);
    const int32_t loss_q7 = kl_sum_q7 (student, teacher, inv_temperature_q15,
                                       student_state, teacher_state);
    const Status finish = finish_loss (pool, loss_q7, out);
    if (!status_ok (finish))
        return finish;
    Saved saved;
    DistillMeta *meta = nullptr;
    if (tape != nullptr)
    {
        const Status save = tape->save<DistillMeta> (1u, saved, meta);
        if (!status_ok (save))
            return save;
        meta->_M_inv_temperature_q15 = inv_temperature_q15;
        meta->_M_alpha_q15 = 0;
        meta->_M_label = 0u;
        meta->_M_classes = static_cast<uint8_t> (student._M_count);
        meta->_M_route_teacher = route_teacher ? 1u : 0u;
        meta->_M_kl_shift = 0u;
        meta->_M_pad = 0u;
    }
    const Trainable operands[2] = { student, teacher };
    return record_loss (tape, &kl_backward, _S_op_kl, out, operands, 2u,
                        tape != nullptr ? &saved : nullptr,
                        tape != nullptr ? 1u : 0u);
}

/** @brief Combined backward: alpha * CE grad + (1 - alpha) * 2^shift * KL grad. */
[[nodiscard]] inline Status
distill_backward (const Trainable & /*output*/,
                  const Trainable &grad_output, GradContext &context) noexcept
{
    if (context._M_operand_count < 2u || context._M_saved_count < 1u)
        return Status::BAD_LAYOUT;
    const Trainable &student = context._M_operands[0];
    const Trainable &teacher = context._M_operands[1];
    const DistillMeta *const meta = context.saved_as<DistillMeta> (0);
    if (meta == nullptr)
        return Status::BAD_LAYOUT;
    const uint32_t count = student._M_count;
    if (teacher._M_count != count || count == 0u
        || count > _S_distill_max_classes || meta->_M_label >= count)
        return Status::BAD_LAYOUT;
    const int32_t seed = grad_output.grad_at (0);
    const int32_t inv_t = meta->_M_inv_temperature_q15;
    const int32_t alpha = meta->_M_alpha_q15;
    const int32_t one_minus = _S_q15_one - 1 - alpha;
    const SoftmaxState student_state
        = softmax_state_from_codes (student, inv_t);
    const SoftmaxState teacher_state
        = softmax_state_from_codes (teacher, inv_t);
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t p_s = prob_q15 (student, i, inv_t, student_state);
        const int32_t p_t = prob_q15 (teacher, i, inv_t, teacher_state);
        const int32_t one_hot
            = i == meta->_M_label ? _S_q15_max : 0;
        /* Three Q15 factors at once in int64: (p_s - one_hot) * invT * alpha
           and (p_s - p_t) * invT * (1 - alpha), each narrowed by 2^30 with
           rounding, then the KL arm gets its 2^kl_shift scale before the
           saturating sum -- the one place a T > 1 gain is applied. */
        const int64_t hard_arm
            = (static_cast<int64_t> (q15_clamp (p_s - one_hot)) * inv_t * alpha)
              >> 30;
        const int64_t soft_raw
            = (static_cast<int64_t> (q15_clamp (p_s - p_t)) * inv_t * one_minus)
              >> 30;
        const int64_t soft_arm = soft_raw << meta->_M_kl_shift;
        const int64_t summed = hard_arm + soft_arm;
        const int32_t combined = q15_clamp (static_cast<int32_t> (
            summed > _S_q15_max ? _S_q15_max
                                : (summed < _S_q15_min ? _S_q15_min : summed)));
        if (student.add_grad (i, q15_mul (combined, seed)))
            ++context._M_saturation_events;
        if (meta->_M_route_teacher != 0u && teacher.has_grad ())
        {
            const int32_t delta
                = log_prob_q7 (teacher, i, inv_t, teacher_state)
                  - log_prob_q7 (student, i, inv_t, student_state);
            const int32_t kl_here = kl_sum_q7 (student, teacher, inv_t,
                                               student_state, teacher_state);
            const int32_t outer
                = q15_clamp (static_cast<int32_t> (
                    static_cast<int64_t> (delta - kl_here) << 8));
            const int32_t grad_raw
                = q15_mul (q15_mul (q15_mul (p_t, inv_t), outer), one_minus);
            const int64_t grad_wide
                = (static_cast<int64_t> (grad_raw) << meta->_M_kl_shift);
            const int32_t grad = q15_clamp (static_cast<int32_t> (
                grad_wide > _S_q15_max
                    ? _S_q15_max
                    : (grad_wide < _S_q15_min ? _S_q15_min : grad_wide)));
            if (teacher.add_grad (i, q15_mul (q15_clamp (grad), seed)))
                ++context._M_saturation_events;
        }
    }
    return Status::OK;
}

/**
 * @brief Combined distillation loss: alpha * CE + (1 - alpha) * 2^kl_shift * KL.
 *
 * CE is the softened log-softmax NLL of the label (exact: -log p_y from the
 * same Q7 sweep the KL uses, so both arms see the same student
 * probabilities); KL is the summed form above.  The two arms are blended by
 * alpha (Q15) with the KL gain folded into kl_shift, and the recorded
 * backward routes the blended gradient to the student and -- when
 * route_teacher is set -- the KL arm's analytic derivative to a trainable
 * teacher.
 * Pool cost: 8-byte node + 16-byte DistillMeta save.
 */
[[nodiscard]] inline Status
distill_forward (Pool &pool, Tape *const tape, const Trainable &student,
                 const Trainable &teacher, const uint16_t label,
                 const int32_t inv_temperature_q15, const int32_t alpha_q15,
                 const uint32_t kl_shift, DistillResult &out,
                 const bool route_teacher = false) noexcept
{
    const Status pair = loss_pair_ok (student, teacher);
    if (!status_ok (pair))
        return pair;
    if (student._M_count > _S_distill_max_classes || label >= student._M_count)
        return Status::OUT_OF_RANGE;
    if (inv_temperature_q15 <= 0 || inv_temperature_q15 > _S_q15_one)
        return Status::OUT_OF_RANGE;
    if (kl_shift > 4u)
        return Status::OUT_OF_RANGE;
    const int32_t alpha = q15_clamp (alpha_q15);
    const int32_t one_minus = _S_q15_one - 1 - alpha;
    const SoftmaxState student_state
        = softmax_state_from_codes (student, inv_temperature_q15);
    const SoftmaxState teacher_state
        = softmax_state_from_codes (teacher, inv_temperature_q15);
    const int32_t kl_q7 = kl_sum_q7 (student, teacher, inv_temperature_q15,
                                     student_state, teacher_state);
    const int32_t hard_q7
        = -log_prob_q7 (student, label, inv_temperature_q15, student_state);
    const int64_t blended = ((static_cast<int64_t> (hard_q7) * alpha) >> 15)
                            + (((static_cast<int64_t> (kl_q7) * one_minus)
                                >> 15)
                               << kl_shift);
    const Status finish
        = finish_loss (pool, static_cast<int32_t> (blended), out._M_total);
    if (!status_ok (finish))
        return finish;
    out._M_kl_q7 = kl_q7;
    out._M_hard_q7 = hard_q7;
    Saved saved;
    DistillMeta *meta = nullptr;
    if (tape != nullptr)
    {
        const Status save = tape->save<DistillMeta> (1u, saved, meta);
        if (!status_ok (save))
            return save;
        meta->_M_inv_temperature_q15 = inv_temperature_q15;
        meta->_M_alpha_q15 = alpha;
        meta->_M_label = label;
        meta->_M_classes = static_cast<uint8_t> (student._M_count);
        meta->_M_route_teacher = route_teacher ? 1u : 0u;
        meta->_M_kl_shift = static_cast<uint8_t> (kl_shift);
        meta->_M_pad = 0u;
    }
    const Trainable operands[2] = { student, teacher };
    return record_loss (tape, &distill_backward, _S_op_distill, out._M_total,
                        operands, 2u, tape != nullptr ? &saved : nullptr,
                        tape != nullptr ? 1u : 0u);
}
} // namespace zzs::mllee::train
