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
#include "mllee/train_simd.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Gradient clipping: value, global-norm, per-group, L1, finite-fraction.
 *
 * Every entry point returns the PRE-clip norm in a ClipResult so the trainer
 * can log it (and so the loss-scale module can spot rail saturation after
 * the fact), and every write path saturates instead of wrapping: lanes are
 * clamped with q15_to_i16_sat / scale_lanes, which pin at the rails, and the
 * norm reductions use the int64 accumulators from the foundation so a long
 * parameter list cannot overflow the reported norm.
 *
 * The norms are lane-domain integers (one unit = 1/32768 of the code scale),
 * which is what sum_sq / sum_abs report before any narrowing.  Global norms
 * therefore exceed 1.0 routinely: _M_norm_q15 carries the saturating Q15
 * copy for logging, while the clipping decision uses the raw lane sum, so a
 * norm of 4.0 does not alias into a norm of 0.99997.
 *
 * Pool cost: zero -- every function mutates gradient lanes in place and
 * writes the result struct on the caller's stack.
 */
namespace zzs::mllee::train
{
/** @brief One clipping evaluation: norms before, lanes touched, scale used. */
struct ClipResult
{
    /** @brief Pre-clip norm in raw lane units (saturating int32). */
    int32_t _M_norm_raw = 0;
    /** @brief The same norm narrowed to Q15 for telemetry (rails at 1.0). */
    int32_t _M_norm_q15 = 0;
    /** @brief Lanes that were clamped or scaled. */
    uint32_t _M_lanes_clipped = 0u;
    /** @brief True when the clipping actually changed any lane. */
    uint32_t _M_was_scaled = 0u;
    /** @brief The factor applied to unscaled lanes, Q15 (0 = untouched). */
    int32_t _M_factor_q15 = 0;
};

static_assert (sizeof (ClipResult) == 20u, "clip result is five words");

/** @brief A named set of parameters clipped under one norm budget. */
struct ParamGroup
{
    const Trainable *_M_params = nullptr;
    uint32_t _M_count = 0u;
    /** @brief Budget in raw lane units (same domain as ClipResult::_M_norm_raw). */
    int32_t _M_max_norm_raw = _S_q15_max;
};

static_assert (sizeof (ParamGroup) == (sizeof (void *) == 4u ? 12u : 16u),
               "param group is a pointer plus two words");

/** @brief Saturating raw-lane copy of a norm, for the Q15 telemetry field. */
[[nodiscard]] inline constexpr int32_t
norm_to_q15_log (const int64_t norm_raw) noexcept
{
    return q15_clamp (static_cast<int32_t> (
        norm_raw > _S_q15_max ? _S_q15_max
                              : (norm_raw < 0 ? 0 : norm_raw)));
}

/** @brief Exact integer L2 norm of one gradient buffer in lane units. */
[[nodiscard]] inline int32_t
grad_norm_raw_q15 (const int16_t *const lanes, const uint32_t count) noexcept
{
    const int64_t squares = sum_squares_q15 (lanes, count);
    if (squares <= 0)
        return 0;
    /* int64 sqrt by integer narrowing: values above uint32 would exceed the
       _S_isqrt input width, so cap first; the cap only affects logging of
       absurd (post-saturation) gradients. */
    const uint64_t bounded = squares
                                 > static_cast<int64_t> (0xFFFFFFFFull)
                             ? 0xFFFFFFFFull
                             : static_cast<uint64_t> (squares);
    return static_cast<int32_t> (_S_isqrt (static_cast<uint32_t> (bounded)));
}

/**
 * @brief Value clamp: every gradient lane forced into [-limit, +limit].
 *
 * Reports the pre-clip L-infinity norm (the largest |lane|), which is the
 * only relevant "norm" for a value clamp and is already computed by the
 * tensor reductions.
 */
[[nodiscard]] inline ClipResult
clip_grad_value_q15 (const Trainable &param, const int32_t limit_q15) noexcept
{
    ClipResult result;
    if (param._M_grad == nullptr || param._M_count == 0u)
        return result;
    const int32_t limit = limit_q15 < 0 ? -limit_q15 : limit_q15;
    result._M_norm_raw = max_abs_q15 (param._M_grad, param._M_count);
    result._M_norm_q15 = norm_to_q15_log (result._M_norm_raw);
    for (uint32_t i = 0; i < param._M_count; ++i)
    {
        const int32_t value = param._M_grad[i];
        if (value > limit || value < -limit)
        {
            param._M_grad[i] = q15_to_i16_sat (value > 0 ? limit : -limit);
            ++result._M_lanes_clipped;
        }
    }
    result._M_was_scaled = result._M_lanes_clipped != 0u ? 1u : 0u;
    result._M_factor_q15 = 0;
    return result;
}

/**
 * @brief Global L2 norm clip over a parameter list: all lanes scaled by
 *        max / norm when the norm exceeds max (the standard total-norm form).
 *
 * The norm is summed across every parameter in int64 (the per-buffer squares
 * are exact via the foundation reduction), the factor is one ratio_q15, and
 * the scaling pass is scale_lanes -- a stride-1 SIMD kernel per buffer, no
 * pool access.  max_norm_q15 spans up to ~4.0 because the threshold is a
 * raw lane count, not a saturated lane: the comparison happens before any
 * narrowing.
 */
[[nodiscard]] inline ClipResult
clip_grad_norm_q15 (const Trainable *const params, const uint32_t count,
                    const int32_t max_norm_raw) noexcept
{
    ClipResult result;
    int64_t total_squares = 0;
    for (uint32_t p = 0; p < count; ++p)
    {
        const Trainable &param = params[p];
        if (param._M_grad == nullptr)
            continue;
        total_squares += sum_squares_q15 (param._M_grad, param._M_count);
    }
    /* Exact int64 norm: digit-pair isqrt over the wide sum. */
    int64_t wide_norm = 0;
    {
        int64_t bit = 1;
        while (bit * 4 <= total_squares)
            bit *= 4;
        int64_t remainder = total_squares;
        int64_t root = 0;
        while (bit != 0)
        {
            if (remainder >= root + bit)
            {
                remainder -= root + bit;
                root = (root >> 1) + bit;
            }
            else
                root >>= 1;
            bit >>= 2;
        }
        wide_norm = root;
    }
    result._M_norm_raw = static_cast<int32_t> (
        wide_norm > 0x7FFFFFFFll ? 0x7FFFFFFFll : wide_norm);
    result._M_norm_q15 = norm_to_q15_log (wide_norm);
    if (wide_norm == 0 || max_norm_raw <= 0 || wide_norm <= max_norm_raw)
        return result;
    const int32_t factor = ratio_q15 (max_norm_raw, result._M_norm_raw);
    result._M_factor_q15 = factor;
    result._M_was_scaled = 1u;
    for (uint32_t p = 0; p < count; ++p)
    {
        const Trainable &param = params[p];
        if (param._M_grad == nullptr)
            continue;
        scale_lanes (param._M_grad, factor, param._M_count);
        result._M_lanes_clipped += param._M_count;
    }
    return result;
}

/** @brief Per-group global-norm clip; one budget per parameter group. */
[[nodiscard]] inline Status
clip_grad_group_norms (const ParamGroup *const groups, const uint32_t group_count,
                       ClipResult *const results_out) noexcept
{
    if (groups == nullptr || results_out == nullptr)
        return Status::BAD_LAYOUT;
    for (uint32_t g = 0; g < group_count; ++g)
    {
        if (group_count > 1u && groups[g]._M_params == nullptr)
            return Status::BAD_LAYOUT;
        results_out[g] = clip_grad_norm_q15 (groups[g]._M_params,
                                             groups[g]._M_count,
                                             groups[g]._M_max_norm_raw);
    }
    return Status::OK;
}

/** @brief L1 clip: scale every lane by max / sum|g| when the L1 norm exceeds max. */
[[nodiscard]] inline ClipResult
clip_grad_l1_q15 (const Trainable *const params, const uint32_t count,
                  const int32_t max_l1_raw) noexcept
{
    ClipResult result;
    int64_t total_abs = 0;
    for (uint32_t p = 0; p < count; ++p)
    {
        const Trainable &param = params[p];
        if (param._M_grad == nullptr)
            continue;
        total_abs += l1_norm_q15 (param._M_grad, param._M_count);
    }
    result._M_norm_raw = static_cast<int32_t> (
        total_abs > 0x7FFFFFFFll ? 0x7FFFFFFFll : total_abs);
    result._M_norm_q15 = norm_to_q15_log (total_abs);
    if (total_abs == 0 || max_l1_raw <= 0 || total_abs <= max_l1_raw)
        return result;
    const int32_t factor
        = ratio_q15 (max_l1_raw, result._M_norm_raw);
    result._M_factor_q15 = factor;
    result._M_was_scaled = 1u;
    for (uint32_t p = 0; p < count; ++p)
    {
        const Trainable &param = params[p];
        if (param._M_grad == nullptr)
            continue;
        scale_lanes (param._M_grad, factor, param._M_count);
        result._M_lanes_clipped += param._M_count;
    }
    return result;
}

/**
 * @brief Clip by fraction of finite lanes.
 *
 * A lane is "non-finite" when it sits exactly at a Q15 rail (-32768 or
 * 32767) -- the signature of a saturating add upstream.  If more than
 * max_fraction_q15 (Q15 share) of the lanes are rails, the whole step is
 * rejected by zeroing the buffer (the trainer treats it like an
 * inf-found skip); otherwise every rail lane is clamped to limit_q15.
 * Reports the pre-clip L-infinity norm, which for this mode is always a
 * rail when the reject path is taken.
 */
[[nodiscard]] inline ClipResult
clip_grad_finite_fraction_q15 (const Trainable &param,
                               const int32_t max_fraction_q15,
                               const int32_t limit_q15) noexcept
{
    ClipResult result;
    if (param._M_grad == nullptr || param._M_count == 0u)
        return result;
    uint32_t rails = 0u;
    for (uint32_t i = 0; i < param._M_count; ++i)
        if (param._M_grad[i] == _S_q15_min || param._M_grad[i] == _S_q15_max)
            ++rails;
    result._M_norm_raw = max_abs_q15 (param._M_grad, param._M_count);
    result._M_norm_q15 = norm_to_q15_log (result._M_norm_raw);
    const uint32_t allowed = static_cast<uint32_t> (
        (static_cast<int64_t> (param._M_count) * max_fraction_q15) >> 15);
    if (rails > allowed)
    {
        fill_lanes (param._M_grad, 0, param._M_count);
        result._M_lanes_clipped = param._M_count;
        result._M_was_scaled = 1u;
        return result;
    }
    for (uint32_t i = 0; i < param._M_count; ++i)
    {
        if (param._M_grad[i] == _S_q15_min || param._M_grad[i] == _S_q15_max)
        {
            param._M_grad[i] = q15_to_i16_sat (param._M_grad[i] > 0
                                                   ? limit_q15
                                                   : -limit_q15);
            ++result._M_lanes_clipped;
        }
    }
    result._M_was_scaled = result._M_lanes_clipped != 0u ? 1u : 0u;
    return result;
}
} // namespace zzs::mllee::train
