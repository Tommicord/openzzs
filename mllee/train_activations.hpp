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
#include "mllee/train_fixed.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Differentiable activations for the training path.
 *
 * Two domains are covered on purpose, and which one a caller uses is not a
 * style choice:
 *
 *   - the code domain (`act_code`, `act_forward_codes`) is the inference op's
 *     own evaluation, byte for byte -- a model trained through this entry
 *     produces exactly the codes Core will produce at run time, so there is no
 *     train/deploy skew to budget for;
 *   - the Q15 lane domain (`act_q15`) is the wide, continuous form the gradient
 *     math needs, because a code is 1/127 of a unit and a derivative computed
 *     from it would be a step function.
 *
 * The derivative is therefore analytic (Q15, from the pre-activation) while the
 * saved forward output may be the quantized one.  That is the same trade
 * fake-quant training makes: quantize the value, differentiate through the real
 * function.
 *
 * Nothing here adds a table: every transcendental goes through the exp table the
 * inference module already pays for, so this header costs no extra .rodata.
 */
namespace zzs::mllee::train
{
/**
 * @brief GELU's sigmoid surrogate slope, 1.702, split into a Q15-representable
 *        form (1 + 0.702) because 1.702 does not fit a Q15 lane.
 */
inline constexpr int32_t _S_gelu_tail_q15 = 22997;

/** @brief One half in Q15: the slope of the hard-sigmoid family. */
inline constexpr int32_t _S_q15_half = 16384;

/** @brief GELU's surrogate slope 1.702 as an exact rational, because the gain
 *        has to be applied before the result is narrowed into a lane. */
inline constexpr int64_t _S_gelu_gain_num = 1702;
inline constexpr int64_t _S_gelu_gain_den = 1000;

/**
 * @brief d/dx [ x * sigmoid(1.702 x) ] in Q15 units, deliberately un-narrowed.
 *
 * The product rule gives sigmoid(u) + 1.702 * x * sigmoid'(u), and the second
 * term is where two traps live: dropping the 1.702 makes the ramp too flat (the
 * surrogate's own slope, not the true GELU's), and applying it after a Q15
 * multiply clamps x * sigmoid'(x) before the gain ever reaches it.  So the whole
 * term is one int64 product divided once, and the result -- which peaks near 1.1
 * and so cannot be a lane -- stays wide for act_grad_product to consume.
 */
[[nodiscard]] inline constexpr int64_t
gelu_derivative_wide_q15 (const int32_t x_q15) noexcept
{
    const int32_t s = q15_sigmoid (x_q15 + q15_mul (x_q15, _S_gelu_tail_q15));
    const int64_t scaled = static_cast<int64_t> (x_q15)
                           * q15_sigmoid_grad (s) * _S_gelu_gain_num;
    /* A lane-wide sigmoid derivative means u is near zero, so |x| is too: the
       product is self-limiting and the divisor only ever shrinks it. */
    const int64_t denominator = _S_gelu_gain_den << _S_grad_frac_bits;
    return static_cast<int64_t> (s) + (scaled + (denominator >> 1)) / denominator;
}

/** @brief Hard sigmoid clamp((x + 1) / 2, 0, 1) on the normalized code domain. */
[[nodiscard]] inline constexpr int32_t
hard_sigmoid_q15 (const int32_t x_q15) noexcept
{
    const int32_t shifted = (x_q15 + _S_q15_one) >> 1;
    return shifted < 0 ? 0 : (shifted > _S_q15_max ? _S_q15_max : shifted);
}

/**
 * @brief Forward activation in Q15, over the full +/-8 domain.
 *
 * @param aux_code the PReLU-family slope or the ELU alpha as an int8 code, so
 *        one argument serves both and the layer descriptor's aux byte needs no
 *        second interpretation.
 */
[[nodiscard]] inline constexpr int32_t
act_q15 (const Act act, const int32_t x_q15, const int32_t aux_code) noexcept
{
    switch (act)
        {
        case Act::RELU:
            return x_q15 < 0 ? 0 : (x_q15 > _S_q15_max ? _S_q15_max : x_q15);
        case Act::RELU6:
            /* The normalized domain has no 6: code 127 is 1.0, so the upper
               clamp is the representable maximum. */
            return x_q15 < 0 ? 0 : (x_q15 > _S_q15_max ? _S_q15_max : x_q15);
        case Act::SIGMOID:
            return q15_sigmoid (x_q15);
        case Act::TANH:
            return q15_tanh (x_q15);
        case Act::GELU:
            return q15_mul (x_q15,
                            q15_sigmoid (x_q15 + q15_mul (x_q15, _S_gelu_tail_q15)));
        case Act::LEAKY_RELU:
            return x_q15 < 0 ? q15_mul (x_q15, code_to_q15 (aux_code)) : x_q15;
        case Act::ELU:
            return x_q15 < 0
                       ? q15_mul (code_to_q15 (aux_code), q15_exp (x_q15) - _S_q15_one)
                       : x_q15;
        case Act::SILU:
            return q15_mul (x_q15, q15_sigmoid (x_q15));
        case Act::HARDSWISH:
            return q15_mul (x_q15, hard_sigmoid_q15 (x_q15));
        case Act::HARDSIGMOID:
            return hard_sigmoid_q15 (x_q15);
        case Act::NONE:
        default:
            return x_q15 < _S_q15_min ? _S_q15_min
                                      : (x_q15 > _S_q15_max ? _S_q15_max : x_q15);
        }
}

/**
 * @brief d(act)/dx in Q15, from the pre-activation and the forward result.
 *
 * The result is a lane, so a gain above 1.0 clips: hard-swish's ramp peaks at
 * 1.5 and this accessor reports 0.99996 there.  act_route_grad_q15 is the form
 * that keeps the gain, because it multiplies before it narrows.
 *
 * @param y_q15 the forward result, used where it is cheaper and more exact than
 *              re-deriving it (sigmoid, tanh and the clamped rectifiers).
 */
[[nodiscard]] inline constexpr int32_t
act_derivative_q15 (const Act act, const int32_t x_q15, const int32_t y_q15,
                    const int32_t aux_code) noexcept
{
    switch (act)
        {
        case Act::RELU:
            return y_q15 > 0 ? _S_q15_max : 0;
        case Act::RELU6:
            return y_q15 > 0 && y_q15 < _S_q15_max ? _S_q15_max : 0;
        case Act::SIGMOID:
            return q15_sigmoid_grad (y_q15);
        case Act::TANH:
            return q15_clamp (_S_q15_one - 1 - q15_mul (y_q15, y_q15));
        case Act::GELU:
            return q15_clamp (static_cast<int32_t> (gelu_derivative_wide_q15 (x_q15)));
        case Act::LEAKY_RELU:
            return x_q15 < 0 ? code_to_q15 (aux_code) : _S_q15_max;
        case Act::ELU:
            return x_q15 < 0 ? q15_mul (code_to_q15 (aux_code), q15_exp (x_q15))
                             : _S_q15_max;
        case Act::SILU:
            {
                const int32_t s = q15_sigmoid (x_q15);
                return q15_add_sat (s, q15_mul (x_q15, q15_sigmoid_grad (s)));
            }
        case Act::HARDSWISH:
            if (x_q15 <= -_S_q15_one)
                return 0;
            if (x_q15 >= _S_q15_one)
                return _S_q15_max;
            return q15_clamp (x_q15 + _S_q15_half);
        case Act::HARDSIGMOID:
            return y_q15 > 0 && y_q15 < _S_q15_max ? _S_q15_half : 0;
        case Act::NONE:
        default:
            return _S_q15_max;
        }
}

/** @brief d(act)/dx taken straight from a pre-activation code. */
[[nodiscard]] inline constexpr int32_t
act_derivative_from_code (const Act act, const int32_t code,
                          const int32_t aux_code) noexcept
{
    const int32_t x_q15 = code_to_q15 (code);
    return act_derivative_q15 (act, x_q15, act_q15 (act, x_q15, aux_code),
                               aux_code);
}

/**
 * @brief The un-narrowed gradient product grad * act'(x), in Q30.
 *
 * int64 on purpose: hard-swish's ramp has derivative x + 1/2, which reaches 1.5
 * and does not fit a lane, so the gain has to survive until after the multiply.
 * GELU's surrogate enters through the same wide form because its derivative
 * peaks near 1.1, which a lane would silently report as 0.99996.  The widest
 * product here is 32767 * 49151, comfortably inside int64.
 */
[[nodiscard]] inline constexpr int64_t
act_grad_product (const Act act, const int32_t x_q15, const int32_t grad_q15,
                  const int32_t aux_code) noexcept
{
    if (act == Act::HARDSWISH)
        {
            if (x_q15 >= _S_q15_one)
                return static_cast<int64_t> (grad_q15) << _S_grad_frac_bits;
            if (x_q15 > -_S_q15_one)
                return static_cast<int64_t> (grad_q15) * (x_q15 + _S_q15_half);
            return 0;
        }
    if (act == Act::GELU)
        return static_cast<int64_t> (grad_q15) * gelu_derivative_wide_q15 (x_q15);
    return static_cast<int64_t> (grad_q15)
           * act_derivative_q15 (act, x_q15, act_q15 (act, x_q15, aux_code),
                                 aux_code);
}

/**
 * @brief Route a gradient through an activation, reporting the saturation.
 *
 * @param clamped set true when the narrowed result had to saturate, which the
 *        caller adds to the tape's saturation count.
 */
[[nodiscard]] inline constexpr int32_t
act_route_grad_q15 (const Act act, const int32_t x_q15, const int32_t grad_q15,
                    const int32_t aux_code, bool &clamped) noexcept
{
    const int32_t narrowed = _S_round_shift (
        act_grad_product (act, x_q15, grad_q15, aux_code), _S_grad_frac_bits);
    clamped = narrowed > _S_q15_max || narrowed < _S_q15_min;
    return q15_clamp (narrowed);
}

/** @brief Route a gradient, ignoring whether the lane clamped. */
[[nodiscard]] inline constexpr int32_t
act_route_grad_q15 (const Act act, const int32_t x_q15, const int32_t grad_q15,
                    const int32_t aux_code) noexcept
{
    bool ignored = false;
    return act_route_grad_q15 (act, x_q15, grad_q15, aux_code, ignored);
}

/** @brief True when the op is a pure mask or identity, so a layer can fuse its
 *        backward into the previous one. */
[[nodiscard]] inline constexpr bool
act_is_linear (const Act act) noexcept
{
    return act == Act::NONE || act == Act::RELU;
}

/** @brief True when the backward needs the pre-activation, not only the output. */
[[nodiscard]] inline constexpr bool
act_needs_preactivation (const Act act) noexcept
{
    return act == Act::GELU || act == Act::SILU || act == Act::ELU
           || act == Act::HARDSWISH || act == Act::LEAKY_RELU
           || act == Act::HARDSIGMOID;
}

/**
 * @brief Forward in int8 code units -- the inference op, unchanged.
 *
 * Kept separate from act_q15 plus narrowing so a trained tensor and an
 * inference tensor can never disagree by a code.  The narrowing is part of that
 * equality, not a courtesy: every inference call site writes
 * `_S_i8_code (_S_apply_act (...))`, so dropping it here would let a -128 input
 * pass straight through and break the symmetric-code invariant the format checks.
 */
[[nodiscard]] inline constexpr int32_t
act_code (const Act act, const int32_t code, const int32_t aux_code) noexcept
{
    return _S_i8_code (_S_apply_act (act, code, aux_code));
}

/** @brief Elementwise code-domain forward over two stride-1, 4-byte aligned streams. */
inline void
act_forward_codes (const Act act, const int8_t *const in, int8_t *const out,
                   const uint32_t count, const int32_t aux_code) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        out[i] = static_cast<int8_t> (act_code (act, in[i], aux_code));
}

/** @brief Elementwise Q15 forward over two stride-1, 4-byte aligned lane streams. */
inline void
act_forward_q15 (const Act act, const int16_t *const in, int16_t *const out,
                 const uint32_t count, const int32_t aux_code) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        out[i] = q15_to_i16_sat (act_q15 (act, in[i], aux_code));
}

/**
 * @brief Backward over saved Q15 pre-activations: grad_in = grad_out * act'(pre).
 *
 * @return the number of lanes that clamped, which the caller adds to the tape's
 *         saturation count.
 */
inline uint32_t
act_backward_q15 (const Act act, const int16_t *const pre,
                  const int16_t *const grad_out, int16_t *const grad_in,
                  const uint32_t count, const int32_t aux_code) noexcept
{
    uint32_t saturated = 0u;
    if (act_is_linear (act))
        {
            /* RELU and identity are a mask or a copy: no transcendental, and the
               only saturation possible is the incoming lane's own. */
            for (uint32_t i = 0; i < count; ++i)
                {
                    const int32_t keep
                        = (act == Act::NONE || pre[i] >= 0) ? grad_out[i] : 0;
                    if (!q15_in_range (keep))
                        ++saturated;
                    grad_in[i] = q15_to_i16_sat (keep);
                }
            return saturated;
        }
    for (uint32_t i = 0; i < count; ++i)
        {
            bool clamped = false;
            grad_in[i] = q15_to_i16_sat (
                act_route_grad_q15 (act, pre[i], grad_out[i], aux_code, clamped));
            if (clamped)
                ++saturated;
        }
    return saturated;
}

/** @brief Backward against saved int8 pre-activations (the quantized forward path). */
inline uint32_t
act_backward_codes (const Act act, const int8_t *const pre,
                    const int16_t *const grad_out, int16_t *const grad_in,
                    const uint32_t count, const int32_t aux_code) noexcept
{
    uint32_t saturated = 0u;
    if (act_is_linear (act))
        {
            for (uint32_t i = 0; i < count; ++i)
                {
                    const int32_t keep
                        = (act == Act::NONE || pre[i] >= 0) ? grad_out[i] : 0;
                    if (!q15_in_range (keep))
                        ++saturated;
                    grad_in[i] = q15_to_i16_sat (keep);
                }
            return saturated;
        }
    for (uint32_t i = 0; i < count; ++i)
        {
            bool clamped = false;
            grad_in[i] = q15_to_i16_sat (
                act_route_grad_q15 (act, code_to_q15 (pre[i]), grad_out[i],
                                    aux_code, clamped));
            if (clamped)
                ++saturated;
        }
    return saturated;
}
} // namespace zzs::mllee::train
