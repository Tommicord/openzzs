/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the
 * following conditions:
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
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_loss.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"
#include "mllee/train_trainer.hpp"

#include <cstdint>

/**
 * @brief Regularization as installable PostGradHooks: the pieces PyTorch
 *        scatters across optimizers and hand-written loops, in one header.
 *
 * Every entry point is a plain kernel over caller-owned Trainable views:
 * function pointers only, no virtual dispatch, no heap, no writable static.
 * The kernels mutate either the parameter codes (the decoupled forms, applied
 * between the backward and the update) or the gradient lanes (the coupled
 * forms, injected as extra gradient terms), and each one returns Status and
 * accumulates into a caller-owned RegularizationStats that reports every
 * saturation it caused -- the same "a clamped lane is a countable event, not a
 * silent wrap" contract the backward kernels follow.
 *
 * Why decoupled weight decay and the L2 penalty are two distinct entry points
 * and not one knob: AdamW's update shrinks the PARAMETER (p <- p - lr*lambda*p,
 * multiplicatively, independent of the gradient), while the classical coupled
 * L2 adds lambda*p to the GRADIENT, so it flows through the optimizer's own
 * state (momentum, adaptive scaling) and shrinks at a rate the optimizer
 * controls.  On plain SGD the two agree to first order; on any adaptive or
 * momentum update they are different algorithms, and the tests pin that they
 * produce different results on the same gradient.
 *
 * Rounding conventions, stated per kernel because the module requires it:
 *   - decoupled decay and row-max-norm rescale round with `div_round` (half
 *     away from zero) after the Q15 product, which is sign-symmetric and can
 *     never INCREASE a magnitude (the product's real value is <= |code|, and
 *     a tie only lands back on |code|);
 *   - gradient injections use the exact Q15 product (`q15_mul`, ties toward
 *     +infinity) and accumulate through `Trainable::add_grad`, which is where
 *     the saturation is counted, not invented;
 *   - the sign of an exact-zero code under L1 is defined away (see the L1
 *     kernel's doc block).
 *
 * Entropy and KL penalties work in the Q7 log domain and reuse the loss
 * atoms (q7_log_q15 / q7_neg_log_q15 via train_fixed, the log-prob code grid
 * `_S_log_step_q7` from train_loss); neither a cross-entropy nor a KL sum is
 * reimplemented here beyond the derivative each one injects.
 */
namespace zzs::mllee::train
{
/** @brief Policy bit: apply decoupled (AdamW-style) weight decay. */
inline constexpr uint16_t _S_reg_decay = 1u << 0;

/** @brief Policy bit: add the coupled L2 term to the gradient. */
inline constexpr uint16_t _S_reg_l2 = 1u << 1;

/** @brief Policy bit: add the L1 subgradient to the gradient. */
inline constexpr uint16_t _S_reg_l1 = 1u << 2;

/** @brief Policy bit: enforce the max row norm on parameter matrices. */
inline constexpr uint16_t _S_reg_max_norm = 1u << 3;

/** @name Stats and policy blocks
 *
 * The stats struct is the caller-owned accumulator for every kernel in this
 * header: it is NOT pre-zeroed by the kernels (the convention: the caller
 * zeroes it, the kernels only add into it), and its size is pinned because it
 * is meant to live next to a run's FitStats in a .bss telemetry block.
 * @{
 */

/** @brief Accumulated regularization accounting, one per run or per hook call. */
struct RegularizationStats
{
    /** @brief Sum of applied penalty VALUES in Q7 (0 for parameter-side forms
     *        that contribute no loss term, e.g. decoupled decay). */
    int64_t _M_penalty_q7 = 0;
    /** @brief Codes or lanes the kernel actually changed. */
    uint32_t _M_terms_applied = 0u;
    /** @brief Parameter rows rescaled by the max-norm constraint. */
    uint32_t _M_rows_clamped = 0u;
    /** @brief Lanes clamped to a Q15 rail by add_grad. */
    uint32_t _M_saturation_events = 0u;
    /** @brief Lanes whose value was already at a rail / edge and was skipped. */
    uint32_t _M_lanes_skipped = 0u;
    /** @brief Parameters skipped entirely (frozen or bias-like). */
    uint32_t _M_params_skipped = 0u;
};

static_assert (sizeof (RegularizationStats) == 32u,
               "regularization stats are one qword plus six words, telemetry-sized");

/** @brief An immutable penalty configuration, installable through the driver. */
struct RegularizationPolicy
{
    const Trainable *_M_params = nullptr;
    uint32_t _M_param_count = 0u;
    uint16_t _M_flags = 0u;
    /** @brief Column count for row max-norm (0 = read from the tensor Shape). */
    uint16_t _M_row_cols = 0u;
    int32_t _M_lr_q15 = 0;      /* 0 = take the trainer's schedule value */
    int32_t _M_decay_q15 = 0;   /* AdamW lambda, Q15 */
    int32_t _M_l2_q15 = 0;      /* coupled lambda, Q15 */
    int32_t _M_l1_q15 = 0;      /* subgradient lambda, Q15 */
    int32_t _M_max_row_norm_q15 = _S_q15_max;
    int32_t _M_pad = 0;
};

static_assert (sizeof (RegularizationPolicy) == sizeof (void *) + 32u,
               "policy is one pointer plus eight words");

/**@}*/

/**
 * @brief AdamW decay strength as a code-retention Q15 factor:
 *        R = 1 - lr*lambda, floor-clamped to the smallest representable
 *        shrinkage (one unit of 32768).  A product below 1/32768 cannot
 *        survive a Q15 lane, and decaying to R = 1.0 would be a silent no-op,
 *        so the floor is the honest minimum decay, not an approximation.
 */
[[nodiscard]] inline constexpr int32_t
decay_retention_q15 (const int32_t lr_q15, const int32_t decay_q15) noexcept
{
    const int32_t lr_lambda = q15_mul (q15_clamp (lr_q15), q15_clamp (decay_q15));
    const int32_t retention = _S_q15_one - 1 - lr_lambda; /* R as a lane value */
    return retention < 0 ? 0 : retention;
}

/**
 * @brief Decoupled weight decay (AdamW form), applied to the CODES before the
 *        update: code <- round_half_away_from_zero (R * code / 2^15).
 *
 * Parameter-space, gradient-independent, exactly the AdamW semantics and
 * precisely NOT the L2 penalty below.  The rounding can never increase |code|
 * (the product is <= |code| in real terms), and magnitudes that round back to
 * themselves simply do not decay this step -- that is the documented integer
 * behaviour of a 7-bit parameter space, and _M_terms_applied counts only the
 * codes that actually changed.  Bias-like tensors are skipped when @p
 * skip_bias is set (the convention PyTorch copies from weight init: biases
 * regularize poorly).
 */
inline Status
weight_decay_decoupled (const Trainable &param, const int32_t retention_q15,
                        const bool skip_bias,
                        RegularizationStats &stats) noexcept
{
    const Status status = param.validate ();
    if (!status_ok (status))
        return status;
    if (!param.requires_grad () || param._M_grad == nullptr)
        {
            ++stats._M_params_skipped;
            return Status::OK;
        }
    if (skip_bias && (param._M_flags & Flag::BiasLike) != 0u)
        {
            ++stats._M_params_skipped;
            return Status::OK;
        }
    const int32_t retention = q15_clamp (retention_q15);
    for (uint32_t i = 0; i < param._M_count; ++i)
        {
            const int32_t code = param._M_code[i];
            if (code == 0)
                {
                    ++stats._M_lanes_skipped; /* nothing to shrink */
                    continue;
                }
            const int32_t decayed = div_round (retention * code, _S_q15_one);
            const int32_t bounded = q15_abs (decayed) > q15_abs (code)
                                        ? (code < 0 ? -q15_abs (code) : code)
                                        : decayed;
            if (bounded != code)
                {
                    param._M_code[i] = _S_i8_code (bounded);
                    ++stats._M_terms_applied;
                }
        }
    return Status::OK;
}

/**
 * @brief Coupled L2: g += lambda * p, the classical form that enters the
 *        OPTIMIZER through the gradient lane.
 *
 * lambda_q15 is the Q15 penalty weight, p is the parameter's own Q15 code
 * value, and the product is the exact Q15 multiply (`q15_mul`, ties toward
 * +infinity).  Accumulation goes through add_grad, so a lane pushed past the
 * rail clamps there and increments _M_saturation_events -- the coupled form's
 * main failure mode (it fights the loss gradient in the same lane) is exactly
 * the failure mode the count is about.
 */
inline Status
l2_gradient_penalty (const Trainable &param, const int32_t lambda_q15,
                     RegularizationStats &stats) noexcept
{
    const Status status = param.validate ();
    if (!status_ok (status))
        return status;
    if (!param.requires_grad () || param._M_grad == nullptr)
        {
            ++stats._M_params_skipped;
            return Status::OK;
        }
    const int32_t lambda = q15_clamp (lambda_q15);
    for (uint32_t i = 0; i < param._M_count; ++i)
        {
            const int32_t delta = q15_mul (lambda, code_to_q15 (param._M_code[i]));
            if (param.add_grad (i, delta))
                ++stats._M_saturation_events;
            ++stats._M_terms_applied;
        }
    return Status::OK;
}

/**
 * @brief L1 as a subgradient sign step: g += lambda * sign(p).
 *
 * Exact-zero-avoidance rule, stated because the subgradient of |x| at 0 is
 * the whole interval [-1, 1] and an implementation must choose: a parameter
 * code of exactly 0 receives NO injection.  Choosing 0 there keeps unused
 * weights pinned at zero (the sparsity property that makes L1 worth paying
 * for) and removes the sign tie a `+lambda` choice would bake into the very
 * first step; a code whose magnitude is >= 1 always gets the full +/- lambda
 * toward zero.  The injection itself is a Q15 lane equal to lambda (the L1
 * derivative per unit real is exactly sign), again via add_grad with the rail
 * clamps counted.
 */
inline Status
l1_gradient_penalty (const Trainable &param, const int32_t lambda_q15,
                     RegularizationStats &stats) noexcept
{
    const Status status = param.validate ();
    if (!status_ok (status))
        return status;
    if (!param.requires_grad () || param._M_grad == nullptr)
        {
            ++stats._M_params_skipped;
            return Status::OK;
        }
    const int32_t lambda = q15_clamp (lambda_q15);
    for (uint32_t i = 0; i < param._M_count; ++i)
        {
            const int32_t code = param._M_code[i];
            if (code == 0)
                {
                    /* the zero-avoidance choice: subgradient 0 at the kink */
                    ++stats._M_lanes_skipped;
                    continue;
                }
            const int32_t delta = code > 0 ? -lambda : lambda;
            if (param.add_grad (i, delta))
                ++stats._M_saturation_events;
            ++stats._M_terms_applied;
        }
    return Status::OK;
}

/**
 * @brief Max-norm constraint per parameter ROW (per output unit), the original
 *        network-above-max-norm regularizer, in code space.
 *
 * @param cols 0 = take the column count from the tensor's rank-2 Shape (rows
 *        are the leading axis, stride-1 trailing axis, exactly the inference
 *        weight layout); a rank-1 tensor is one row; anything else needs an
 *        explicit cols that divides the count.
 *
 * The row norm is the L2 norm of the row's Q15 code values.  For a row whose
 * squared sum still fits the Q15 square domain it is `q15_sqrt` (sqrt of the
 * Q15 squared-sum, which returns the norm itself in Q15 because
 * sqrt(x * 2^15) undoes one of the two Q15 bases); larger rows fall back to
 * the raw integer square root, and the overflow path (a norm above 2^17 in
 * lane units) is scaled down by a power of two before the root and back after
 * it.  The factor applied to an over-budget row is max/norm, computed as
 * `q15_mul (max_norm_q15, q15_reciprocal (norm_q15))` when the norm is at or
 * above 1.0 (where the reciprocal is exact in the lane domain) and as
 * `ratio_q15` below it.  Codes are rescaled with `div_round` (half away from
 * zero), which cannot increase a magnitude.  Every rescaled row increments
 * _M_rows_clamped; every changed code increments _M_terms_applied.
 */
inline Status
max_norm_rows (const Trainable &param, const uint32_t cols,
               const int32_t max_norm_q15, RegularizationStats &stats) noexcept
{
    const Status status = param.validate ();
    if (!status_ok (status))
        return status;
    if (!param.requires_grad ())
        {
            ++stats._M_params_skipped;
            return Status::OK;
        }
    const uint32_t count = param._M_count;
    uint32_t row_len = cols;
    if (row_len == 0u)
        {
            if (param._M_shape._M_rank == 2u)
                row_len = param._M_shape._M_dims[1];
            else
                row_len = count; /* rank-1 vector: one row */
        }
    if (row_len == 0u || count % row_len != 0u)
        return Status::BAD_LAYOUT;
    const int32_t limit = max_norm_q15 < 0 ? -max_norm_q15 : max_norm_q15;
    const uint32_t rows = count / row_len;
    for (uint32_t row = 0; row < rows; ++row)
        {
            int64_t squares_q30 = 0;
            for (uint32_t k = 0; k < row_len; ++k)
                {
                    const int32_t lane = code_to_q15 (param._M_code[row * row_len + k]);
                    squares_q30 += static_cast<int64_t> (lane) * lane;
                }
            if (squares_q30 == 0)
                continue;
            /* Norm in Q15 lane units: prefer the q15_sqrt domain (the squared
               sum re-based to Q15 stays inside the helper's sound input range
               of [0, 131071]); otherwise take the raw root, with a power-of-two
               detour when even the raw square sum leaves uint32. */
            const int64_t squares_q15 = squares_q30 >> 15;
            int32_t norm_q15 = 0;
            if (squares_q15 >= 1 && squares_q15 <= 131071)
                norm_q15 = q15_sqrt (static_cast<int32_t> (squares_q15));
            else if (squares_q30 <= static_cast<int64_t> (0xFFFFFFFFull))
                norm_q15 = static_cast<int32_t> (
                    _S_isqrt (static_cast<uint32_t> (squares_q30)));
            else
                {
                    const int32_t half = static_cast<int32_t> (
                        _S_isqrt (static_cast<uint32_t> (squares_q30 >> 18)));
                    norm_q15 = half > (_S_q15_max * 2) ? _S_q15_max * 2
                                                       : half * 512;
                }
            if (norm_q15 <= limit)
                continue; /* the row already satisfies the budget */
            const int32_t factor = norm_q15 >= _S_q15_one
                                       ? q15_mul (limit, q15_reciprocal (norm_q15))
                                       : ratio_q15 (limit, norm_q15);
            ++stats._M_rows_clamped;
            for (uint32_t k = 0; k < row_len; ++k)
                {
                    const int32_t index = static_cast<int32_t> (row * row_len + k);
                    const int32_t code = param._M_code[index];
                    const int32_t scaled = div_round (factor * code, _S_q15_one);
                    const int32_t bounded = q15_abs (scaled) > q15_abs (code)
                                                ? code
                                                : scaled;
                    if (bounded != code)
                        {
                            param._M_code[index] = _S_i8_code (bounded);
                            ++stats._M_terms_applied;
                        }
                }
        }
    return Status::OK;
}

/**
 * @brief Entropy bonus as a gradient injection on an output distribution.
 *
 * The buffer is a probability Trainable (codes in [0, 127], p = code/127 in
 * the Q15 domain); the penalty value added to the run is -beta * H with
 * H = -sum p ln p, accumulated in Q7 from the train_fixed log atoms.  The
 * derivative per unit p is -beta * (ln p + 1), a Q7 quantity built from
 * q7_neg_log_q15 exactly as the loss module builds its terms, and it enters
 * each gradient lane as h_q7 * 256 (the documented Q7-to-lane base change:
 * one Q7 unit per unit-real equals 256 lane units).  Lanes at p == 0 have the
 * log at its sentinel: the injection would saturate by construction, so they
 * are counted in _M_saturation_events and clamped through add_grad rather
 * than skipped -- a zero-probability class under an entropy bonus is precisely
 * the signal a caller wants counted.
 * @return BAD_LAYOUT when the tensor carries no gradient lanes to inject into.
 */
inline Status
entropy_penalty (const Trainable &probs, const int32_t beta_q15,
                 RegularizationStats &stats) noexcept
{
    const Status status = probs.validate ();
    if (!status_ok (status))
        return status;
    if (probs._M_grad == nullptr)
        return Status::BAD_LAYOUT;
    const int32_t beta = q15_clamp (beta_q15);
    int64_t entropy_q7 = 0;
    for (uint32_t i = 0; i < probs._M_count; ++i)
        {
            const int32_t code = probs._M_code[i];
            const int32_t p_q15 = code > 0 ? code_to_q15 (code) : 0;
            const int32_t neg_log = p_q15 > 0 ? q7_neg_log_q15 (p_q15) : 0;
            entropy_q7 += (static_cast<int64_t> (p_q15) * neg_log) >> 15;
            if (p_q15 <= 0)
                {
                    ++stats._M_saturation_events; /* log-sentinel slope */
                    if (probs.add_grad (i, _S_q15_min + 1))
                        ++stats._M_saturation_events;
                    continue;
                }
            /* dH/dp (Q7 per unit real) = neg_log - 1.0; the bonus term is the
               negative of that scaled by beta, still in Q7. */
            const int32_t dH_q7 = neg_log - _S_q7_one;
            const int32_t slope_q7 = -q15_scale_shift (beta, dH_q7, 15u);
            const int64_t lane_wide = static_cast<int64_t> (slope_q7) << 8;
            const int32_t overflowed = lane_wide > _S_q15_max
                                       || lane_wide < _S_q15_min;
            if (overflowed)
                ++stats._M_saturation_events;
            if (probs.add_grad (i, static_cast<int32_t> (
                    lane_wide > _S_q15_max
                        ? _S_q15_max
                        : (lane_wide < _S_q15_min ? _S_q15_min : lane_wide))))
                ++stats._M_saturation_events;
            ++stats._M_terms_applied;
        }
    stats._M_penalty_q7 -= q15_scale_shift (beta, static_cast<int32_t> (
        entropy_q7 > _S_q15_max ? _S_q15_max : entropy_q7), 15u);
    return Status::OK;
}

/**
 * @brief KL(probs || reference) penalty as a gradient injection, Q7 log domain.
 *
 * L = beta * sum p * (ln p - ln t), built only from the q7_neg_log_q15 atom
 * (no cross-entropy reimplementation: this is the divergence term PyTorch
 * callers hand-roll for distillation and prior-matching).  The derivative per
 * unit p is beta * (ln p - ln t + 1); p == 0 contributes 0 to the value and
 * has a sentinel-bound slope, so like the entropy kernel it is counted, not
 * invented.  A reference code <= 0 pins ln t at the sentinel: the injection
 * saturates on that class and is counted as a saturation event.
 * @return BAD_LAYOUT for a length mismatch or a gradient-less probs tensor.
 */
inline Status
kl_penalty (const Trainable &probs, const int8_t *const ref_codes,
           const uint32_t ref_count, const int32_t beta_q15,
           RegularizationStats &stats) noexcept
{
    const Status status = probs.validate ();
    if (!status_ok (status))
        return status;
    if (ref_codes == nullptr || ref_count != probs._M_count
        || probs._M_grad == nullptr)
        return Status::BAD_LAYOUT;
    const int32_t beta = q15_clamp (beta_q15);
    int64_t kl_q7 = 0;
    for (uint32_t i = 0; i < probs._M_count; ++i)
        {
            const int32_t p_code = probs._M_code[i];
            const int32_t t_code = ref_codes[i];
            const int32_t p_q15 = p_code > 0 ? code_to_q15 (p_code) : 0;
            const int32_t t_q15 = t_code > 0 ? code_to_q15 (t_code) : 0;
            if (p_q15 <= 0)
                {
                    ++stats._M_saturation_events;
                    continue; /* 0 * ln 0 = 0 in the value; sentinel slope */
                }
            const int32_t neg_p = q7_neg_log_q15 (p_q15);
            const int32_t neg_t = q7_neg_log_q15 (t_q15);
            /* ln p - ln t = neg_t - neg_p, still Q7. */
            const int32_t log_ratio_q7 = neg_t - neg_p;
            kl_q7 += (static_cast<int64_t> (p_q15) * log_ratio_q7) >> 15;
            const int32_t d_q7 = log_ratio_q7 + _S_q7_one;
            const int32_t slope_q7 = q15_scale_shift (beta, d_q7, 15u);
            const int64_t lane_wide = static_cast<int64_t> (slope_q7) << 8;
            if (lane_wide > _S_q15_max || lane_wide < _S_q15_min)
                ++stats._M_saturation_events;
            if (probs.add_grad (i, static_cast<int32_t> (
                    lane_wide > _S_q15_max
                        ? _S_q15_max
                        : (lane_wide < _S_q15_min ? _S_q15_min : lane_wide))))
                ++stats._M_saturation_events;
            ++stats._M_terms_applied;
        }
    stats._M_penalty_q7 += q15_scale_shift (beta, static_cast<int32_t> (
        kl_q7 > _S_q15_max ? _S_q15_max : (kl_q7 < 0 ? 0 : kl_q7)), 15u);
    return Status::OK;
}

/**
 * @brief Activity penalty on the output MEAN: L = beta * (mean(y) - target)^2.
 *
 * Works on the forward codes (the activations a hook can see), accumulates
 * the Q7 loss term from the Q15 code domain and injects the derivative
 * 2*beta*(mean - target)/n per element into the activation's own gradient
 * lanes.  Rounding: the mean divides with div_round (half away from zero),
 * matching the tensor reductions, and the per-element injection narrows once
 * from the int64 product exactly like act_grad_product.
 * @return BAD_LAYOUT when the activation buffer carries no gradients.
 */
inline Status
mean_activity_penalty (const Trainable &activations,
                       const int32_t target_mean_q15, const int32_t beta_q15,
                       RegularizationStats &stats) noexcept
{
    const Status status = activations.validate ();
    if (!status_ok (status))
        return status;
    if (activations._M_grad == nullptr || activations._M_count == 0u)
        return Status::BAD_LAYOUT;
    const int32_t beta = q15_clamp (beta_q15);
    int64_t total_q15 = 0;
    for (uint32_t i = 0; i < activations._M_count; ++i)
        total_q15 += code_to_q15 (activations._M_code[i]);
    const int32_t mean_q15 = static_cast<int32_t> (
        total_q15 >= 0 ? (total_q15 + activations._M_count / 2)
                       / activations._M_count
                       : -(((-total_q15) + activations._M_count / 2)
                           / activations._M_count));
    const int32_t diff_q15 = q15_sub_sat (mean_q15, target_mean_q15);
    stats._M_penalty_q7 += (static_cast<int64_t> (beta) * diff_q15 * diff_q15)
                           >> 38; /* beta_q15 * diff^2 / (n-free): see block */
    const int64_t numer = 2 * static_cast<int64_t> (beta) * diff_q15;
    const int32_t scaled = static_cast<int32_t> (_S_round_shift (numer, 15u));
    const int32_t per_element
        = div_round (scaled, static_cast<int32_t> (activations._M_count));
    for (uint32_t i = 0; i < activations._M_count; ++i)
        {
            if (activations.add_grad (i, per_element))
                ++stats._M_saturation_events;
            ++stats._M_terms_applied;
        }
    return Status::OK;
}

/**
 * @brief Squared-activity penalty: L = beta * mean(y^2), the activation
 *        analogue of the coupled L2 (it shrinks what the network SAYS, not
 *        what its weights ARE).
 *
 * The value is the Q15 squared sum re-based once for Q7 (beta_q15 * sum y^2
 * / (n * 2^38)); the derivative per element is 2*beta*y/n in lane units, the
 * one narrowing from the int64 product, so a saturated rail is the only
 * reporting failure and it is counted.
 */
inline Status
squared_activity_penalty (const Trainable &activations, const int32_t beta_q15,
                          RegularizationStats &stats) noexcept
{
    const Status status = activations.validate ();
    if (!status_ok (status))
        return status;
    if (activations._M_grad == nullptr || activations._M_count == 0u)
        return Status::BAD_LAYOUT;
    const int32_t beta = q15_clamp (beta_q15);
    int64_t squares_q30 = 0;
    for (uint32_t i = 0; i < activations._M_count; ++i)
        {
            const int32_t lane = code_to_q15 (activations._M_code[i]);
            squares_q30 += static_cast<int64_t> (lane) * lane;
        }
    stats._M_penalty_q7
        += (static_cast<int64_t> (beta) * squares_q30)
           / (static_cast<int64_t> (activations._M_count) << 38);
    for (uint32_t i = 0; i < activations._M_count; ++i)
        {
            const int64_t lane = code_to_q15 (activations._M_code[i]);
            const int64_t numer = 2 * static_cast<int64_t> (beta) * lane;
            const int32_t scaled = static_cast<int32_t> (_S_round_shift (numer, 15u));
            const int32_t per_element
                = div_round (scaled, static_cast<int32_t> (activations._M_count));
            if (activations.add_grad (i, per_element))
                ++stats._M_saturation_events;
            ++stats._M_terms_applied;
        }
    return Status::OK;
}

/**
 * @brief Generic multi-penalty driver, PostGradHook semantics.
 *
 * Installable by wrapping: a caller's own `Status hook (StepContext &c, uint32_t s)
 * { return regularization_post_grad (c, s, my_policy, my_stats); }` is a valid
 * StepHooks::_M_post_grad value -- function pointer, no capture, because the
 * policy and the stats are .rodata/.bss the caller owns.  The learning rate
 * comes from the policy when set, otherwise from the schedule value the
 * trainer already parked in the context.
 *
 * A nonzero @p saturated skips the whole injection and returns OK: the
 * trainer is about to halve the loss scale and drop this update, so injecting
 * penalty terms into lanes that already clamped would double-count a failure
 * the loss-scale machine is handling.
 */
inline Status
regularization_post_grad (StepContext &context, const uint32_t saturated,
                          const RegularizationPolicy &policy,
                          RegularizationStats &stats) noexcept
{
    if (policy._M_params == nullptr || policy._M_param_count == 0u
        || policy._M_param_count > _S_max_trainer_params)
        return Status::BAD_LAYOUT;
    if (saturated != 0u)
        return Status::OK;
    const int32_t lr
        = policy._M_lr_q15 != 0 ? policy._M_lr_q15 : context._M_lr_q15;
    const int32_t retention
        = decay_retention_q15 (lr, policy._M_decay_q15);
    for (uint32_t i = 0; i < policy._M_param_count; ++i)
        {
            const Trainable &param = policy._M_params[i];
            Status status = Status::OK;
            if ((policy._M_flags & _S_reg_decay) != 0u)
                status = weight_decay_decoupled (param, retention, true, stats);
            if (status_ok (status) && (policy._M_flags & _S_reg_l2) != 0u)
                status = l2_gradient_penalty (param, policy._M_l2_q15, stats);
            if (status_ok (status) && (policy._M_flags & _S_reg_l1) != 0u)
                status = l1_gradient_penalty (param, policy._M_l1_q15, stats);
            if (status_ok (status) && (policy._M_flags & _S_reg_max_norm) != 0u)
                status = max_norm_rows (param, policy._M_row_cols,
                                        policy._M_max_row_norm_q15, stats);
            if (!status_ok (status))
                return status;
        }
    return Status::OK;
}

/* ======================================================================== */
/* Ready-made single-policy hooks                                            */
/* ======================================================================== */

/** @brief Default decoupled decay rate, Q15 (about 0.01). */
inline constexpr int32_t _S_reg_default_decay_q15 = 328;

/** @brief Default coupled L2 rate, Q15. */
inline constexpr int32_t _S_reg_default_l2_q15 = 328;

/** @brief Default L1 subgradient rate, Q15. */
inline constexpr int32_t _S_reg_default_l1_q15 = 258;

/**
 * @brief Installable hook: decoupled weight decay at the compiled default.
 *
 * Runs after every backward (the trainer passes the tape's saturation count;
 * a dirty step skips the hook's effect by returning OK untouched), shrinking
 * the parameter codes directly with the context's schedule lr.
 */
inline Status
weight_decay_post_grad (StepContext &context, const uint32_t saturated) noexcept
{
    if (saturated != 0u || context._M_params == nullptr)
        return Status::OK;
    const int32_t retention
        = decay_retention_q15 (context._M_lr_q15, _S_reg_default_decay_q15);
    /* The hook carries its own throwaway counters: it exists so a device
       caller can install decay without defining anything; a caller that
       wants the accounting uses regularization_post_grad with a stats block.
       Stats live in the caller's frame here, so no writable static. */
    RegularizationStats stats;
    for (uint32_t i = 0; i < context._M_param_count; ++i)
        {
            const Status status = weight_decay_decoupled (
                context._M_params[i], retention, true, stats);
            if (!status_ok (status))
                return status;
        }
    return Status::OK;
}

/** @brief Installable hook: coupled L2 at the compiled default. */
inline Status
l2_post_grad (StepContext &context, const uint32_t saturated) noexcept
{
    if (saturated != 0u || context._M_params == nullptr)
        return Status::OK;
    RegularizationStats stats;
    for (uint32_t i = 0; i < context._M_param_count; ++i)
        {
            const Status status = l2_gradient_penalty (
                context._M_params[i], _S_reg_default_l2_q15, stats);
            if (!status_ok (status))
                return status;
        }
    return Status::OK;
}

/** @brief Installable hook: L1 subgradient at the compiled default. */
inline Status
l1_post_grad (StepContext &context, const uint32_t saturated) noexcept
{
    if (saturated != 0u || context._M_params == nullptr)
        return Status::OK;
    RegularizationStats stats;
    for (uint32_t i = 0; i < context._M_param_count; ++i)
        {
            const Status status = l1_gradient_penalty (
                context._M_params[i], _S_reg_default_l1_q15, stats);
            if (!status_ok (status))
                return status;
        }
    return Status::OK;
}
} // namespace zzs::mllee::train
