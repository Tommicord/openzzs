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
#include "mllee/train_pool.hpp"
#include "mllee/train_simd.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Parameter optimizers: SGD, Adam, AdamW, Adagrad, RMSprop, Adadelta, Lion.
 *
 * One shape for every optimizer: a hyper-parameter struct (sized and
 * static_assert-ed), per-parameter state carved from the caller's pool, and a
 * step function that consumes the parameter's Q15 gradient lanes and writes
 * the update through the train_simd primitives -- sgd_step_i8 for the final
 * code-domain move, axpy_q15 / scale_q15 for the moment recursions -- so all
 * seven inherit the vectorized path for free.
 *
 * Design rules the whole module obeys:
 *
 *   - Dispatch is a constexpr function-pointer table indexed by OptimKind.
 *     No virtual, no RTTI, no writable static; the table is .rodata.
 *   - Bias correction is integer and O(1) per step: the running powers
 *     beta1^t and beta2^t are Q15 words in OptimState, multiplied once per
 *     step, and the correction is one q15_div each.  m-hat and v-hat can
 *     exceed 1.0 early in training and saturate at the rails -- the same
 *     documented behaviour as every other lane in the module.
 *   - epsilon is a Q15 constant (default _S_adam_eps_q15 = 32 ~ 0.001), added
 *     after the sqrt where sqrt(0) would otherwise divide.
 *   - L2 weight decay (SGD, Adagrad, RMSprop) enters the gradient lanes in
 *     place; decoupled decay (AdamW, Lion) is applied to the parameter codes
 *     after the step.  BiasLike parameters never decay.
 *   - The step consumes the gradient buffer: momenta are written back into
 *     the operand's own lanes where a consumer buffer is needed, so no
 *     temporary lane buffer is allocated mid-update (pool access is once, at
 *     state init).
 *
 * Per-parameter pool bytes of optimizer state (n = element count):
 *   SGD (momentum)   2n   | Adam / AdamW   4n   | Adagrad    2n
 *   RMSprop          4n   | Adadelta       4n   | Lion       2n
 * plus 32 bytes of OptimState metadata per parameter.  RMSprop carves two
 * buffers because its optional momentum lane lives in the secondary slot.
 */
namespace zzs::mllee::train
{
/** @brief Adam-family default epsilon, Q15 (~0.000976). */
inline constexpr int32_t _S_adam_eps_q15 = 32;

/** @brief Adadelta epsilon: one code quantum (258), the floor under which the
 *         ratio-driven update can never advance a code step.  A smaller eps
 *         (e.g. the 128 a float port would carry) leaves the steady-state
 *         update at ~eps, which rounds to zero code movement in sgd_step_i8,
 *         so the floor must sit at the quantum the parameters move in. */
inline constexpr int32_t _S_adadelta_eps_q15 = 258;

/** @brief Optimizer kinds, in dispatch-table order. */
enum class OptimKind : uint8_t
{
    SGD = 0,
    ADAM,
    ADAMW,
    ADAGRAD,
    RMSPROP,
    ADELTA,
    LION,
    COUNT,
};

/** @brief SGD: momentum, nesterov, dampening and L2 decay. */
struct SgdParams
{
    int32_t _M_lr_q15 = 3277;
    int32_t _M_momentum_q15 = 0;
    int32_t _M_dampening_q15 = 0;
    int32_t _M_weight_decay_q15 = 0;
    uint8_t _M_nesterov = 0u;
    uint8_t _M_pad[3] = { 0u, 0u, 0u };
};

static_assert (sizeof (SgdParams) == 20u, "sgd params is four words plus a padded flag byte");

/** @brief Adam with integer bias correction. */
struct AdamParams
{
    int32_t _M_lr_q15 = 131;
    int32_t _M_beta1_q15 = 29491;
    int32_t _M_beta2_q15 = 32735;
    int32_t _M_eps_q15 = _S_adam_eps_q15;
};

static_assert (sizeof (AdamParams) == 16u, "adam params is four Q15 words");

/** @brief AdamW: decoupled weight decay on top of Adam. */
struct AdamWParams
{
    int32_t _M_lr_q15 = 131;
    int32_t _M_beta1_q15 = 29491;
    int32_t _M_beta2_q15 = 32735;
    int32_t _M_eps_q15 = _S_adam_eps_q15;
    int32_t _M_weight_decay_q15 = 328;
};

static_assert (sizeof (AdamWParams) == 20u, "adamw params is five Q15 words");

/** @brief Adagrad: accumulating squared-gradient denominators. */
struct AdagradParams
{
    int32_t _M_lr_q15 = 1638;
    int32_t _M_eps_q15 = _S_adam_eps_q15;
    int32_t _M_weight_decay_q15 = 0;
};

static_assert (sizeof (AdagradParams) == 12u, "adagrad params is three Q15 words");

/** @brief RMSprop: exponential squared-gradient average, optional momentum. */
struct RmspropParams
{
    int32_t _M_lr_q15 = 328;
    int32_t _M_alpha_q15 = 29491;
    int32_t _M_momentum_q15 = 0;
    int32_t _M_eps_q15 = _S_adam_eps_q15;
    int32_t _M_weight_decay_q15 = 0;
};

static_assert (sizeof (RmspropParams) == 20u, "rmsprop params is five Q15 words");

/** @brief Adadelta: update-history over gradient-history, no global lr needed. */
struct AdadeltaParams
{
    int32_t _M_lr_q15 = _S_q15_max;
    int32_t _M_rho_q15 = 29491;
    int32_t _M_eps_q15 = _S_adadelta_eps_q15;
};

static_assert (sizeof (AdadeltaParams) == 12u, "adadelta params is three Q15 words");

/** @brief Lion: sign-of-interpolated-momentum update, decoupled decay. */
struct LionParams
{
    int32_t _M_lr_q15 = 33;
    int32_t _M_beta1_q15 = 32112;
    int32_t _M_beta2_q15 = 29194;
    int32_t _M_weight_decay_q15 = 0;
};

static_assert (sizeof (LionParams) == 16u, "lion params is four Q15 words");

/** @brief One union so dispatch has a single parameter ABI. */
union OptimParamsUnion
{
    constexpr OptimParamsUnion () noexcept : _M_sgd ()
    {
    }

    SgdParams _M_sgd;
    AdamParams _M_adam;
    AdamWParams _M_adamw;
    AdagradParams _M_adagrad;
    RmspropParams _M_rmsprop;
    AdadeltaParams _M_adadelta;
    LionParams _M_lion;
};

static_assert (sizeof (OptimParamsUnion) == 20u, "the widest params struct sizes the union");

/**
 * @brief Per-parameter state: moment buffers plus bias-correction powers.
 *
 * The powers start at 1.0 (Q15) so the first correction divides by
 * (1 - beta), exactly like the float reference; step counts every applied
 * update and is owned by the caller's loop, not a static.
 */
struct OptimState
{
    int16_t *_M_primary = nullptr;
    int16_t *_M_secondary = nullptr;
    int32_t _M_beta1_pow_q15 = _S_q15_one;
    int32_t _M_beta2_pow_q15 = _S_q15_one;
    uint32_t _M_step = 0u;
};

static_assert (sizeof (OptimState) == (sizeof (void *) == 4u ? 20u : 32u),
               "optim state is two lane pointers plus three words, padded to pointer alignment");

/** @brief Lane bytes one optimizer needs for @p count parameters. */
[[nodiscard]] inline constexpr uint32_t
optim_state_bytes (const OptimKind kind, const uint32_t count) noexcept
{
    switch (kind)
    {
    case OptimKind::SGD:
    case OptimKind::ADAGRAD:
    case OptimKind::LION:
        return 2u * count;
    case OptimKind::ADAM:
    case OptimKind::ADAMW:
    case OptimKind::RMSPROP:
    case OptimKind::ADELTA:
        return 4u * count;
    case OptimKind::COUNT:
        break;
    }
    return 0u;
}

/** @brief Carve the buffers and reset the bias powers for one parameter. */
[[nodiscard]] inline Status
optim_state_init (Pool &pool, const OptimKind kind, const Trainable &param,
                  OptimState &state) noexcept
{
    if (!status_ok (param.validate ()))
        return Status::BAD_LAYOUT;
    state = OptimState{};
    const uint32_t count = param._M_count;
    switch (kind)
    {
    case OptimKind::SGD:
    case OptimKind::ADAGRAD:
    case OptimKind::LION:
        return pool.carve_typed<int16_t> (count, state._M_primary);
    case OptimKind::ADAM:
    case OptimKind::ADAMW:
    case OptimKind::RMSPROP:
    case OptimKind::ADELTA:
        {
            const Status first
                = pool.carve_typed<int16_t> (count, state._M_primary);
            if (!status_ok (first))
                return first;
            return pool.carve_typed<int16_t> (count, state._M_secondary);
        }
    case OptimKind::COUNT:
        break;
    }
    return Status::UNSUPPORTED;
}

/** @brief L2 decay folded into the gradient lanes: g += wd * theta. */
inline void
apply_l2_decay (const Trainable &param, const int32_t weight_decay_q15) noexcept
{
    if (weight_decay_q15 == 0 || (param._M_flags & Flag::BiasLike) != 0u)
        return;
    for (uint32_t i = 0; i < param._M_count; ++i)
    {
        const int32_t term
            = q15_mul (weight_decay_q15, param.q15_at (i));
        param._M_grad[i] = q15_to_i16_sat (param.grad_at (i) + term);
    }
}

/** @brief Decoupled decay on the code stream: theta -= round(lr * wd * theta). */
inline void
apply_decoupled_decay (const Trainable &param, const int32_t lr_q15,
                       const int32_t weight_decay_q15) noexcept
{
    if (weight_decay_q15 == 0 || (param._M_flags & Flag::BiasLike) != 0u)
        return;
    const int32_t combined = q15_mul (lr_q15, weight_decay_q15);
    for (uint32_t i = 0; i < param._M_count; ++i)
    {
        /* The product is a Q15 value; one code step is 258 Q15 units, so the
           decay advances in whole code steps with round-half-away. */
        const int32_t decay_q15 = q15_mul (combined, param.q15_at (i));
        const int32_t code_steps
            = (decay_q15 >= 0 ? decay_q15 + 129 : decay_q15 - 129) / 258;
        param.set_code (i, param.code_at (i) - code_steps);
    }
}

/** @brief SGD with momentum / nesterov / dampening and optional L2 decay. */
[[nodiscard]] inline Status
sgd_step (const Trainable &param, OptimState &state,
          const OptimParamsUnion &params) noexcept
{
    if (param._M_grad == nullptr)
        return Status::BAD_LAYOUT;
    const SgdParams &cfg = params._M_sgd;
    /* Momentum on a state whose buffers were never carved is a wiring bug:
       falling through to the plain lr step would silently drop the requested
       recurrence and still move the parameter, so refuse before any write,
       exactly like the Adam-family steps do. */
    if (cfg._M_momentum_q15 > 0 && state._M_primary == nullptr)
        return Status::BAD_LAYOUT;
    apply_l2_decay (param, cfg._M_weight_decay_q15);
    if (cfg._M_momentum_q15 > 0)
    {
        const uint32_t count = param._M_count;
        scale_lanes (state._M_primary, cfg._M_momentum_q15, count);
        const int32_t keep = _S_q15_one - 1 - cfg._M_dampening_q15;
        axpy_lanes (state._M_primary, param._M_grad, keep, count);
        if (cfg._M_nesterov != 0u)
        {
            for (uint32_t i = 0; i < count; ++i)
                param._M_grad[i] = q15_to_i16_sat (
                    param._M_grad[i]
                    + q15_mul (cfg._M_momentum_q15, state._M_primary[i]));
            train_simd::sgd_step_i8 (param._M_code, param._M_grad,
                                     cfg._M_lr_q15, count);
        }
        else
            train_simd::sgd_step_i8 (param._M_code, state._M_primary,
                                     cfg._M_lr_q15, count);
    }
    else
        train_simd::sgd_step_i8 (param._M_code, param._M_grad, cfg._M_lr_q15,
                                 param._M_count);
    ++state._M_step;
    return Status::OK;
}

/** @brief Shared Adam / AdamW recursion; @p decoupled selects AdamW decay. */
[[nodiscard]] inline Status
adam_step_impl (const Trainable &param, OptimState &state,
                const AdamParams &cfg, const bool decoupled,
                const int32_t weight_decay_q15) noexcept
{
    if (param._M_grad == nullptr || state._M_primary == nullptr
        || state._M_secondary == nullptr)
        return Status::BAD_LAYOUT;
    const uint32_t count = param._M_count;
    if (!decoupled)
        apply_l2_decay (param, weight_decay_q15);
    /* m_t = beta1 m + (1 - beta1) g and v_t = beta2 v + (1 - beta2) g^2:
       the first recursion is the axpy primitive, the second needs the
       per-element square so it runs one fused stride-1 pass. */
    scale_lanes (state._M_primary, cfg._M_beta1_q15, count);
    axpy_lanes (state._M_primary, param._M_grad,
                _S_q15_one - 1 - cfg._M_beta1_q15, count);
    const int32_t beta2 = cfg._M_beta2_q15;
    const int32_t one_minus_beta2 = _S_q15_one - 1 - beta2;
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t g = param._M_grad[i];
        const int32_t g_sq = q15_mul (g, g);
        state._M_secondary[i] = q15_to_i16_sat (
            q15_mul (state._M_secondary[i], beta2)
            + q15_mul (one_minus_beta2, g_sq));
    }
    /* Integer bias correction: the powers advance one multiply per step and
       the corrections are two q15_divs -- no float pow, no stored step table. */
    ++state._M_step;
    state._M_beta1_pow_q15 = q15_mul (state._M_beta1_pow_q15, cfg._M_beta1_q15);
    state._M_beta2_pow_q15 = q15_mul (state._M_beta2_pow_q15, beta2);
    const int32_t corr1 = _S_q15_one - 1 - state._M_beta1_pow_q15;
    const int32_t corr2 = _S_q15_one - 1 - state._M_beta2_pow_q15;
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t m_hat = q15_div (state._M_primary[i], corr1);
        const int32_t v_hat = q15_div (state._M_secondary[i], corr2);
        const int32_t denom = q15_clamp (q15_sqrt (v_hat) + cfg._M_eps_q15);
        param._M_grad[i] = q15_to_i16_sat (q15_div (m_hat, denom));
    }
    train_simd::sgd_step_i8 (param._M_code, param._M_grad, cfg._M_lr_q15,
                             count);
    if (decoupled)
        apply_decoupled_decay (param, cfg._M_lr_q15, weight_decay_q15);
    return Status::OK;
}

/** @brief Adam. */
[[nodiscard]] inline Status
adam_step (const Trainable &param, OptimState &state,
           const OptimParamsUnion &params) noexcept
{
    return adam_step_impl (param, state, params._M_adam, false, 0);
}

/** @brief AdamW (decoupled weight decay). */
[[nodiscard]] inline Status
adamw_step (const Trainable &param, OptimState &state,
            const OptimParamsUnion &params) noexcept
{
    const AdamWParams &cfg = params._M_adamw;
    AdamParams shared;
    shared._M_lr_q15 = cfg._M_lr_q15;
    shared._M_beta1_q15 = cfg._M_beta1_q15;
    shared._M_beta2_q15 = cfg._M_beta2_q15;
    shared._M_eps_q15 = cfg._M_eps_q15;
    return adam_step_impl (param, state, shared, true, cfg._M_weight_decay_q15);
}

/** @brief Adagrad: G_t = G_{t-1} + g^2, theta -= lr * g / (sqrt(G) + eps). */
[[nodiscard]] inline Status
adagrad_step (const Trainable &param, OptimState &state,
              const OptimParamsUnion &params) noexcept
{
    if (param._M_grad == nullptr || state._M_primary == nullptr)
        return Status::BAD_LAYOUT;
    const AdagradParams &cfg = params._M_adagrad;
    apply_l2_decay (param, cfg._M_weight_decay_q15);
    const uint32_t count = param._M_count;
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t g = param._M_grad[i];
        state._M_primary[i] = q15_to_i16_sat (
            state._M_primary[i] + q15_mul (g, g));
    }
    /* The accumulator saturates at 1.0 by construction, so the denominator
       is bounded and only the division can rail -- documented, expected. */
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t denom
            = q15_clamp (q15_sqrt (state._M_primary[i]) + cfg._M_eps_q15);
        param._M_grad[i] = q15_to_i16_sat (q15_div (param._M_grad[i], denom));
    }
    train_simd::sgd_step_i8 (param._M_code, param._M_grad, cfg._M_lr_q15,
                             count);
    ++state._M_step;
    return Status::OK;
}

/** @brief RMSprop with optional momentum: v = a v + (1-a) g^2, then SGD-like. */
[[nodiscard]] inline Status
rmsprop_step (const Trainable &param, OptimState &state,
              const OptimParamsUnion &params) noexcept
{
    if (param._M_grad == nullptr || state._M_primary == nullptr)
        return Status::BAD_LAYOUT;
    const RmspropParams &cfg = params._M_rmsprop;
    /* A requested momentum lane with an uncarved secondary slot is the same
       wiring bug as in sgd_step: the un-momentum branch would silently change
       the update, so refuse before any write. */
    if (cfg._M_momentum_q15 > 0 && state._M_secondary == nullptr)
        return Status::BAD_LAYOUT;
    apply_l2_decay (param, cfg._M_weight_decay_q15);
    const uint32_t count = param._M_count;
    scale_lanes (state._M_primary, cfg._M_alpha_q15, count);
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t g = param._M_grad[i];
        state._M_primary[i] = q15_to_i16_sat (
            state._M_primary[i]
            + q15_mul (_S_q15_one - 1 - cfg._M_alpha_q15, q15_mul (g, g)));
    }
    if (cfg._M_momentum_q15 > 0)
    {
        scale_lanes (state._M_secondary, cfg._M_momentum_q15, count);
        for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t denom = q15_clamp (
                q15_sqrt (state._M_primary[i]) + cfg._M_eps_q15);
            const int32_t normed = q15_div (param._M_grad[i], denom);
            state._M_secondary[i] = q15_to_i16_sat (
                state._M_secondary[i] + normed);
        }
        train_simd::sgd_step_i8 (param._M_code, state._M_secondary,
                                 cfg._M_lr_q15, count);
    }
    else
    {
        for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t denom = q15_clamp (
                q15_sqrt (state._M_primary[i]) + cfg._M_eps_q15);
            param._M_grad[i]
                = q15_to_i16_sat (q15_div (param._M_grad[i], denom));
        }
        train_simd::sgd_step_i8 (param._M_code, param._M_grad, cfg._M_lr_q15,
                                 count);
    }
    ++state._M_step;
    return Status::OK;
}

/** @brief Adadelta: rms(update)/rms(grad) recursion, update history in secondary. */
[[nodiscard]] inline Status
adadelta_step (const Trainable &param, OptimState &state,
               const OptimParamsUnion &params) noexcept
{
    if (param._M_grad == nullptr || state._M_primary == nullptr
        || state._M_secondary == nullptr)
        return Status::BAD_LAYOUT;
    const AdadeltaParams &cfg = params._M_adadelta;
    const uint32_t count = param._M_count;
    scale_lanes (state._M_primary, cfg._M_rho_q15, count);
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t g = param._M_grad[i];
        state._M_primary[i] = q15_to_i16_sat (
            state._M_primary[i]
            + q15_mul (_S_q15_one - 1 - cfg._M_rho_q15, q15_mul (g, g)));
        const int32_t rms_grad
            = q15_clamp (q15_sqrt (state._M_primary[i]) + cfg._M_eps_q15);
        const int32_t rms_update = q15_clamp (
            q15_sqrt (state._M_secondary[i]) + cfg._M_eps_q15);
        const int32_t ratio = q15_div (rms_update, rms_grad);
        /* q15_div returns a clamped lane and g is one, so the shifted product is
           already inside the int32 range the clamp needs -- the cast is explicit
           to keep the narrowing visible rather than implicit. */
        const int32_t update = q15_clamp (static_cast<int32_t> (
            (static_cast<int64_t> (g) * ratio) >> 15));
        param._M_grad[i] = q15_to_i16_sat (update);
        state._M_secondary[i] = q15_to_i16_sat (
            q15_mul (state._M_secondary[i], cfg._M_rho_q15)
            + q15_mul (_S_q15_one - 1 - cfg._M_rho_q15,
                       q15_mul (update, update)));
    }
    train_simd::sgd_step_i8 (param._M_code, param._M_grad, cfg._M_lr_q15,
                             count);
    ++state._M_step;
    return Status::OK;
}

/** @brief sign() narrowed to a saturating Q15 rail, zero stays zero. */
[[nodiscard]] inline constexpr int32_t
sign_to_q15 (const int32_t value) noexcept
{
    return value > 0 ? _S_q15_max : (value < 0 ? _S_q15_min : 0);
}

/** @brief Lion: sign(beta1 m + (1-beta1) g), momentum recurs with beta2. */
[[nodiscard]] inline Status
lion_step (const Trainable &param, OptimState &state,
           const OptimParamsUnion &params) noexcept
{
    if (param._M_grad == nullptr || state._M_primary == nullptr)
        return Status::BAD_LAYOUT;
    const LionParams &cfg = params._M_lion;
    const uint32_t count = param._M_count;
    for (uint32_t i = 0; i < count; ++i)
    {
        const int32_t g = param._M_grad[i];
        const int32_t interpolate = q15_clamp (
            q15_mul (cfg._M_beta1_q15, state._M_primary[i])
            + q15_mul (_S_q15_one - 1 - cfg._M_beta1_q15, g));
        const int32_t update = sign_to_q15 (interpolate);
        state._M_primary[i] = q15_to_i16_sat (
            q15_mul (state._M_primary[i], cfg._M_beta2_q15)
            + q15_mul (_S_q15_one - 1 - cfg._M_beta2_q15, g));
        param._M_grad[i] = q15_to_i16_sat (update);
    }
    train_simd::sgd_step_i8 (param._M_code, param._M_grad, cfg._M_lr_q15,
                             count);
    apply_decoupled_decay (param, cfg._M_lr_q15, cfg._M_weight_decay_q15);
    ++state._M_step;
    return Status::OK;
}

/** @brief Function-pointer step, one per kind; the constexpr table is .rodata. */
using OptimStepFn = Status (*)(const Trainable &, OptimState &,
                               const OptimParamsUnion &);

/**
 * @brief Shared optimizer: a kind tag, one params union, no vtable.
 *
 * Dispatch is `for_kind (kind)` -- a table index -- so a trainer holds one
 * object per parameter group with zero pointer indirection beyond the function
 * pointer itself, and the table survives as constexpr .rodata (no writable
 * static).
 */
class Optimizer
{
  public:
    [[nodiscard]] static OptimStepFn
    for_kind (const OptimKind kind) noexcept
    {
        static constexpr OptimStepFn table[static_cast<uint32_t> (
            OptimKind::COUNT)]
            = { &sgd_step,  &adam_step,  &adamw_step, &adagrad_step,
                &rmsprop_step, &adadelta_step, &lion_step };
        const uint32_t index = static_cast<uint32_t> (kind);
        return index < static_cast<uint32_t> (OptimKind::COUNT)
                   ? table[index]
                   : nullptr;
    }

    [[nodiscard]] Status
    step (const Trainable &param, OptimState &state) const noexcept
    {
        const OptimStepFn fn = for_kind (this->_M_kind);
        if (fn == nullptr)
            return Status::UNSUPPORTED;
        return fn (param, state, this->_M_params);
    }

    /** @brief Overwrite the stored lr (from a schedule) then step. */
    [[nodiscard]] Status
    step_with_lr (const Trainable &param, OptimState &state,
                  const int32_t lr_q15) noexcept
    {
        if (!status_ok (this->set_lr (lr_q15)))
            return Status::OUT_OF_RANGE;
        return this->step (param, state);
    }

    [[nodiscard]] uint32_t
    state_bytes (const uint32_t count) const noexcept
    {
        return optim_state_bytes (this->_M_kind, count);
    }

    /** @brief Set the learning rate in whichever params arm the kind uses. */
    [[nodiscard]] Status
    set_lr (const int32_t lr_q15) noexcept
    {
        if (lr_q15 < 0 || lr_q15 > _S_q15_one)
            return Status::OUT_OF_RANGE;
        switch (this->_M_kind)
        {
        case OptimKind::SGD:
            this->_M_params._M_sgd._M_lr_q15 = lr_q15;
            break;
        case OptimKind::ADAM:
            this->_M_params._M_adam._M_lr_q15 = lr_q15;
            break;
        case OptimKind::ADAMW:
            this->_M_params._M_adamw._M_lr_q15 = lr_q15;
            break;
        case OptimKind::ADAGRAD:
            this->_M_params._M_adagrad._M_lr_q15 = lr_q15;
            break;
        case OptimKind::RMSPROP:
            this->_M_params._M_rmsprop._M_lr_q15 = lr_q15;
            break;
        case OptimKind::ADELTA:
            this->_M_params._M_adadelta._M_lr_q15 = lr_q15;
            break;
        case OptimKind::LION:
            this->_M_params._M_lion._M_lr_q15 = lr_q15;
            break;
        case OptimKind::COUNT:
            return Status::UNSUPPORTED;
        }
        return Status::OK;
    }

    [[nodiscard]] int32_t
    lr_q15 () const noexcept
    {
        switch (this->_M_kind)
        {
        case OptimKind::SGD:
            return this->_M_params._M_sgd._M_lr_q15;
        case OptimKind::ADAM:
            return this->_M_params._M_adam._M_lr_q15;
        case OptimKind::ADAMW:
            return this->_M_params._M_adamw._M_lr_q15;
        case OptimKind::ADAGRAD:
            return this->_M_params._M_adagrad._M_lr_q15;
        case OptimKind::RMSPROP:
            return this->_M_params._M_rmsprop._M_lr_q15;
        case OptimKind::ADELTA:
            return this->_M_params._M_adadelta._M_lr_q15;
        case OptimKind::LION:
            return this->_M_params._M_lion._M_lr_q15;
        case OptimKind::COUNT:
            break;
        }
        return 0;
    }

    OptimKind _M_kind = OptimKind::SGD;
    OptimParamsUnion _M_params = {};
};

static_assert (sizeof (Optimizer) == 24u, "optimizer is a kind byte, padding and one union");
} // namespace zzs::mllee::train
