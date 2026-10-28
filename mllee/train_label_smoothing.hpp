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
 * @brief Label smoothing: soft targets, entropy, and the combined gradient.
 *
 * A smoothed target is the convex blend
 *
 *     t_smooth_c = (1 - eps) * t_c + eps / C
 *
 * with t either one-hot (a class index) or a dense Q15 probability vector.
 * Everything the blend touches is a lane, so eps is a Q15 fraction, the
 * uniform mass eps/C is a Q15 constant per class, and the mixing itself runs
 * through the train_simd primitives: widen, scale by (1 - eps), then
 * accumulate the uniform fill.
 *
 * The gradient identity this module implements is
 *
 *     dCE(t_smooth)/dp_c = (1 - eps) * grad_onehot_c + eps * uniform_c / p_c
 *
 * so a smoothed cross-entropy backward is the one-hot CE gradient scaled by
 * (1 - eps) plus the uniform (eps/C) / p_c term -- both saturating at the Q15
 * rails because 1/p_c is unbounded, exactly as documented for the raw CE in
 * mllee/train_loss.hpp.
 *
 * Pool cost per smoothed_cross_entropy_forward: the shared 8-byte loss
 * scalar node plus one SmoothedCeMeta save (12 bytes aligned).  The pure
 * helpers (smooth_onehot_q15, smooth_dense_q15, entropy_q7, blend) take
 * caller-owned buffers and carve nothing.
 */
namespace zzs::mllee::train
{
/** @brief Label-smoothing op ids (the loss family continues at 0x4C10). */
inline constexpr uint32_t _S_op_smoothed_ce = 0x4C10u;
inline constexpr uint32_t _S_op_soft_ce = 0x4C11u;

/** @brief One smoothing evaluation: eps, the uniform mass and the class count. */
struct SmoothingParams
{
    /** @brief Smoothing mass eps in Q15 (3277 = 0.1). */
    int32_t _M_eps_q15 = 3277;
    /** @brief Derived uniform per-class mass eps / C, filled by the helpers. */
    int32_t _M_uniform_q15 = 0;

    [[nodiscard]] static constexpr SmoothingParams
    make (const int32_t eps_q15, const uint32_t classes) noexcept
    {
        SmoothingParams params;
        params._M_eps_q15 = q15_clamp (eps_q15 < 0 ? 0 : eps_q15);
        /* eps is already Q15, so the uniform mass is a plain integer divide
           (ratio_q15 would re-scale by 2^15 and clamp -- wrong here). */
        params._M_uniform_q15 = classes == 0u
                                    ? 0
                                    : div_round (params._M_eps_q15,
                                                 static_cast<int32_t> (classes));
        return params;
    }

    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        return _S_q15_min <= this->_M_eps_q15
                       && this->_M_eps_q15 <= _S_q15_max
                   && this->_M_uniform_q15 >= 0
                       ? Status::OK
                       : Status::OUT_OF_RANGE;
    }
};

static_assert (sizeof (SmoothingParams) == 8u, "smoothing params is two Q15 words");

/** @brief Saved config for the smoothed CE backward (one tape slot). */
struct SmoothedCeMeta
{
    int32_t _M_eps_q15 = 0;
    int32_t _M_uniform_q15 = 0;
    uint16_t _M_target = 0u;
    uint16_t _M_classes = 0u;
};

static_assert (sizeof (SmoothedCeMeta) == 12u, "smoothed CE meta is two words plus two halfwords");

/**
 * @brief Smooth a one-hot target into Q15 class probabilities.
 *
 * dest[c] = eps/C everywhere, plus (1 - eps) at the target class.  The caller
 * owns @p dest (classes lanes, 4-byte aligned); the fill runs through
 * fill_lanes and the gold bump is a saturating add, so the whole write is the
 * two primitives the backward kernels already vectorize.
 */
inline Status
smooth_onehot_q15 (const uint16_t target, const uint32_t classes,
                   const SmoothingParams &params, int16_t *const dest) noexcept
{
    if (dest == nullptr || classes == 0u || target >= classes)
        return Status::BAD_LAYOUT;
    const Status valid = params.validate ();
    if (!status_ok (valid))
        return valid;
    fill_lanes (dest, params._M_uniform_q15, classes);
    const int32_t remainder = _S_q15_one - 1 - params._M_eps_q15;
    dest[target] = q15_to_i16_sat (dest[target] + remainder);
    return Status::OK;
}

/**
 * @brief Smooth a dense probability target (its codes widened to Q15 lanes).
 *
 * dest = (1 - eps) * t + eps/C.  The widening, the scaling and the final
 * saturating constant add are three stride-1 passes over dest -- the first
 * two are the codes_to_q15 / scale_lanes SIMD primitives, the last is the
 * constant-add no lane primitive covers.  @p dense must carry class-count
 * lanes of probability codes.
 */
inline Status
smooth_dense_q15 (const Trainable &dense, const SmoothingParams &params,
                  int16_t *const dest) noexcept
{
    if (!status_ok (dense.validate ()))
        return Status::BAD_LAYOUT;
    if (dest == nullptr)
        return Status::BAD_LAYOUT;
    const Status valid = params.validate ();
    if (!status_ok (valid))
        return valid;
    train_simd::codes_to_q15 (dense._M_code, dest, dense._M_count);
    const int32_t keep = _S_q15_one - 1 - params._M_eps_q15;
    scale_lanes (dest, keep, dense._M_count);
    for (uint32_t i = 0; i < dense._M_count; ++i)
        dest[i] = q15_to_i16_sat (dest[i] + params._M_uniform_q15);
    return Status::OK;
}

/**
 * @brief Q7 entropy H = -sum p log p of a Q15 lane probability buffer.
 *
 * Each p log p product lands in Q7 (the log is Q7, the probability is Q15,
 * the narrowed product keeps the Q7 domain), and the sum of C terms can pass
 * 1.0 -- log C grows without a lane -- so the result is reported in Q7 like
 * every other magnitude in this module family.  The accumulator sums
 * p * (-log p), which is the positive entropy directly.
 */
[[nodiscard]] inline int32_t
entropy_q7 (const int16_t *const probs_q15, const uint32_t count) noexcept
{
    int64_t total_q7 = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t p = probs_q15[i];
        if (p <= 0)
            continue; /* 0 * log 0 is defined as the zero contribution */
        total_q7 += (static_cast<int64_t> (q7_neg_log_q15 (p)) * p) >> 15;
    }
    return static_cast<int32_t> (total_q7);
}

/** @brief Entropy of an int8 probability operand (the code-domain convenience). */
[[nodiscard]] inline int32_t
entropy_q7 (const Trainable &probs) noexcept
{
    int64_t total_q7 = 0;
    for (uint32_t i = 0; i < probs._M_count; ++i)
    {
        const int32_t p = probs.q15_at (i);
        if (p <= 0)
            continue;
        total_q7 += (static_cast<int64_t> (q7_neg_log_q15 (p)) * p) >> 15;
    }
    return static_cast<int32_t> (total_q7);
}

/**
 * @brief Gradient of a uniform soft target: uniform_grad[c] = -(1/C) / p_c.
 *
 * This is exactly the CE gradient evaluated at the uniform distribution, and
 * it is the second term of the smoothing blend; like the one-hot gradient it
 * saturates at the Q15 rails when p_c is small.
 */
inline Status
uniform_target_gradient (const Trainable &probs, int16_t *const dest,
                         const uint32_t count) noexcept
{
    if (dest == nullptr || probs._M_count != count || count == 0u)
        return Status::BAD_LAYOUT;
    /* 1/C in Q15: a plain integer divide of the unit, not ratio_q15 of a
       Q15 numerator (that would scale by 2^15 again and clamp). */
    const int32_t unit = div_round (_S_q15_one, static_cast<int32_t> (count));
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t p = probs.q15_at (i);
        dest[i] = p > 0 ? q15_to_i16_sat (q15_div (-unit, p))
                        : static_cast<int16_t> (_S_q15_min);
    }
    return Status::OK;
}

/**
 * @brief Gradient blend: dest = (1 - eps) * grad_onehot + eps * grad_uniform.
 *
 * This is the module's headline identity: a caller that already has the raw
 * one-hot CE gradient lanes can turn them into the smoothed CE gradient by
 * blending in the uniform-target gradient (uniform_target_gradient above),
 * because the smoothed target is the same convex combination.  Both passes
 * are axpy_lanes, so the blend is a SIMD kernel over stride-1 lanes.
 */
inline Status
smoothed_gradient_blend (const int16_t *const grad_onehot,
                         const int16_t *const grad_uniform,
                         const SmoothingParams &params, int16_t *const dest,
                         const uint32_t count) noexcept
{
    if (grad_onehot == nullptr || grad_uniform == nullptr || dest == nullptr)
        return Status::BAD_LAYOUT;
    const Status valid = params.validate ();
    if (!status_ok (valid))
        return valid;
    const int32_t keep = _S_q15_one - 1 - params._M_eps_q15;
    fill_lanes (dest, 0, count);
    axpy_lanes (dest, grad_onehot, keep, count);
    axpy_lanes (dest, grad_uniform, params._M_eps_q15, count);
    return Status::OK;
}

/** @brief Smoothed-CE backward: (1 - eps) * grad_onehot + eps * uniform gradient. */
[[nodiscard]] inline Status
smoothed_ce_backward (const Trainable & /*output*/,
                      const Trainable &grad_output, GradContext &context) noexcept
{
    if (context._M_operand_count < 1u || context._M_saved_count < 1u)
        return Status::BAD_LAYOUT;
    const Trainable &probs = context._M_operands[0];
    const SmoothedCeMeta *const meta = context.saved_as<SmoothedCeMeta> (0);
    if (meta == nullptr)
        return Status::BAD_LAYOUT;
    const int32_t seed = grad_output.grad_at (0);
    const SmoothingParams params
        = SmoothingParams::make (meta->_M_eps_q15, meta->_M_classes);
    const int32_t keep = _S_q15_one - 1 - params._M_eps_q15;
    for (uint32_t i = 0; i < probs._M_count; ++i)
    {
        const int32_t p = probs.q15_at (i);
        const int32_t t = i == meta->_M_target ? _S_q15_max : 0;
        /* grad = -( (1 - eps) * t + eps / C ) / p, each rail saturating. */
        const int32_t weight = q15_add_sat (q15_mul (keep, t),
                                            params._M_uniform_q15);
        const int32_t ratio = p > 0 ? q15_div (-weight, p) : -_S_q15_max;
        if (probs.add_grad (i, q15_mul (ratio, seed)))
            ++context._M_saturation_events;
    }
    return Status::OK;
}

/**
 * @brief Cross-entropy of a probability buffer against a smoothed one-hot.
 *
 * L = (1 - eps) * (-log p_y) + (eps / C) * sum_c (-log p_c), accumulated in
 * Q7 (the uniform arm alone reaches -log p * eps/C * C for tiny p, so a Q15
 * lane could not hold the value even for one class).
 * Pool cost: 8-byte scalar node + 12-byte SmoothedCeMeta save.
 */
[[nodiscard]] inline Status
smoothed_cross_entropy_forward (Pool &pool, Tape *const tape,
                                const Trainable &probs, const uint16_t target,
                                const SmoothingParams &params,
                                LossResult &out) noexcept
{
    if (!status_ok (probs.validate ()))
        return Status::BAD_LAYOUT;
    if (probs._M_count == 0u || target >= probs._M_count)
        return Status::OUT_OF_RANGE;
    const Status valid = params.validate ();
    if (!status_ok (valid))
        return valid;
    const int32_t keep = _S_q15_one - 1 - params._M_eps_q15;
    int64_t total_q7 = 0;
    for (uint32_t i = 0; i < probs._M_count; ++i)
    {
        const int32_t neg_log = q7_neg_log_q15 (probs.q15_at (i));
        const int32_t t = i == target ? keep : 0;
        total_q7 += (static_cast<int64_t> (neg_log) * (t + params._M_uniform_q15))
                    >> 15;
    }
    const Status finish = finish_loss (pool, static_cast<int32_t> (total_q7),
                                       out);
    if (!status_ok (finish))
        return finish;
    Saved saved;
    SmoothedCeMeta *meta = nullptr;
    if (tape != nullptr)
    {
        const Status save
            = tape->save<SmoothedCeMeta> (1u, saved, meta);
        if (!status_ok (save))
            return save;
        meta->_M_eps_q15 = params._M_eps_q15;
        meta->_M_uniform_q15 = params._M_uniform_q15;
        meta->_M_target = target;
        meta->_M_classes = static_cast<uint16_t> (probs._M_count);
    }
    const Trainable operands[1] = { probs };
    return record_loss (tape, &smoothed_ce_backward, _S_op_smoothed_ce, out,
                        operands, 1u, tape != nullptr ? &saved : nullptr,
                        tape != nullptr ? 1u : 0u);
}

/** @brief Soft-target CE backward with an explicit eps blend of the gradients. */
[[nodiscard]] inline Status
soft_target_ce_backward (const Trainable & /*output*/,
                         const Trainable &grad_output,
                         GradContext &context) noexcept
{
    if (context._M_operand_count < 2u || context._M_saved_count < 1u)
        return Status::BAD_LAYOUT;
    const Trainable &probs = context._M_operands[0];
    const Trainable &target = context._M_operands[1];
    const SmoothedCeMeta *const meta = context.saved_as<SmoothedCeMeta> (0);
    if (meta == nullptr)
        return Status::BAD_LAYOUT;
    const int32_t seed = grad_output.grad_at (0);
    const int32_t keep = _S_q15_one - 1 - meta->_M_eps_q15;
    for (uint32_t i = 0; i < probs._M_count; ++i)
    {
        const int32_t p = probs.q15_at (i);
        const int32_t t = target.q15_at (i);
        const int32_t weight
            = q15_add_sat (q15_mul (keep, t), meta->_M_uniform_q15);
        const int32_t ratio = p > 0 ? q15_div (-weight, p) : -_S_q15_max;
        if (probs.add_grad (i, q15_mul (ratio, seed)))
            ++context._M_saturation_events;
    }
    return Status::OK;
}

/**
 * @brief Entropy/soft-target variant: CE against a dense smoothed target.
 *
 * The target operand supplies t_c, eps/C (meta->_M_uniform_q15) adds the
 * uniform mass to both the loss and the gradient, and eps = 0 degenerates to
 * the plain soft CE from mllee/train_loss.hpp with identical lanes.
 * Pool cost: 8-byte node + 12-byte meta save.
 */
[[nodiscard]] inline Status
soft_target_cross_entropy_forward (Pool &pool, Tape *const tape,
                                   const Trainable &probs,
                                   const Trainable &target,
                                   const SmoothingParams &params,
                                   LossResult &out) noexcept
{
    const Status pair = loss_pair_ok (probs, target);
    if (!status_ok (pair))
        return pair;
    const Status valid = params.validate ();
    if (!status_ok (valid))
        return valid;
    const int32_t keep = _S_q15_one - 1 - params._M_eps_q15;
    int64_t total_q7 = 0;
    for (uint32_t i = 0; i < probs._M_count; ++i)
    {
        const int32_t neg_log = q7_neg_log_q15 (probs.q15_at (i));
        const int32_t weight
            = q15_add_sat (q15_mul (keep, target.q15_at (i)),
                           params._M_uniform_q15);
        total_q7 += (static_cast<int64_t> (neg_log) * weight) >> 15;
    }
    const Status finish = finish_loss (pool, static_cast<int32_t> (total_q7),
                                       out);
    if (!status_ok (finish))
        return finish;
    Saved saved;
    SmoothedCeMeta *meta = nullptr;
    if (tape != nullptr)
    {
        const Status save
            = tape->save<SmoothedCeMeta> (1u, saved, meta);
        if (!status_ok (save))
            return save;
        meta->_M_eps_q15 = params._M_eps_q15;
        meta->_M_uniform_q15 = params._M_uniform_q15;
        meta->_M_target = 0u;
        meta->_M_classes = static_cast<uint16_t> (probs._M_count);
    }
    const Trainable operands[2] = { probs, target };
    return record_loss (tape, &soft_target_ce_backward, _S_op_soft_ce, out,
                        operands, 2u, tape != nullptr ? &saved : nullptr,
                        tape != nullptr ? 1u : 0u);
}
} // namespace zzs::mllee::train
