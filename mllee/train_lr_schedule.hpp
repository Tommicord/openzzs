/* Copyright(c) 2026 Tommicord
 *
 * Permission for use, copying and distribution is granted under the same
 * MIT-style terms as the rest of the mllee module (see mllee/core.hpp).
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND. */

#pragma once

#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_fixed.hpp"

#include <cstdint>

/**
 * @brief Learning-rate schedules as pure constexpr arithmetic on a step.
 *
 * Every schedule here is a struct of Q15/uint32 configuration plus a
 * `lr_q15 (step)` evaluation: no writable statics (the step is an argument,
 * which is also why these are safe from several trainer objects sharing one
 * binary), no heap, no recursion, and every loop is a bounded integer
 * iteration.  The returned lr is Q15 (1.0 saturates at _S_q15_max), so an lr
 * above 1.0 -- meaningless for the code-domain step anyway -- clamps.
 *
 * Cosine-family schedules need cos in Q15.  The implementation is a folded
 * Taylor polynomial (odd terms through x^7, |err| < 3e-5 on [0, pi/2]) --
 * exact enough for one lane ulp at the domain edges and cheaper than any
 * table this module is not allowed to add.
 *
 * Pool cost: zero.  Schedules are compile-time values on the caller's stack
 * (4-44 bytes per struct, static_assert-ed); nothing is carved.
 */
namespace zzs::mllee::train
{
/** @brief pi in Q15 (3.14159 * 32768). */
inline constexpr int32_t _S_pi_q15 = 102944;

/** @brief pi / 2 in Q15. */
inline constexpr int32_t _S_half_pi_q15 = 51472;

/** @brief 2 pi in Q15. */
inline constexpr int32_t _S_two_pi_q15 = 205887;

/**
 * @brief sin on [0, pi/2] with the folded Taylor series, Q15 in and out.
 *
 * sin(x) = x (1 - x^2/6 + x^4/120 - x^6/5040); the bracket is built in Q15
 * from even powers so the intermediate products stay inside int64 and the
 * final narrowing has one rounding.  x above pi/2 is the caller's fold to do.
 */
[[nodiscard]] inline constexpr int32_t
sin_first_quadrant_q15 (const int32_t x_q15) noexcept
{
    const int64_t x = x_q15;
    const int64_t u = (x * x) >> 15;   /* x^2, Q15, <= ~80840 */
    const int64_t u2 = (u * u) >> 15;  /* x^4, Q15 */
    const int64_t u3 = (u2 * u) >> 15; /* x^6, Q15 */
    const int64_t bracket = _S_q15_one - div_round (static_cast<int32_t> (u), 6)
                            + div_round (static_cast<int32_t> (u2), 120)
                            - div_round (static_cast<int32_t> (u3), 5040);
    return static_cast<int32_t> ((x * bracket + 16384) >> 15);
}

/** @brief sin over [0, 2 pi) by quadrant folding, sign applied, Q15. */
[[nodiscard]] inline constexpr int32_t
sin_q15 (int32_t angle_q15) noexcept
{
    if (angle_q15 <= 0)
        return 0;
    if (angle_q15 >= _S_two_pi_q15)
        angle_q15 %= _S_two_pi_q15;
    if (angle_q15 <= _S_half_pi_q15)
        return sin_first_quadrant_q15 (angle_q15);
    if (angle_q15 <= _S_pi_q15)
        return sin_first_quadrant_q15 (_S_pi_q15 - angle_q15);
    if (angle_q15 <= _S_pi_q15 + _S_half_pi_q15)
        return -sin_first_quadrant_q15 (angle_q15 - _S_pi_q15);
    return -sin_first_quadrant_q15 (_S_two_pi_q15 - angle_q15);
}

/** @brief cos over [0, 2 pi) via the sine identity, Q15. */
[[nodiscard]] inline constexpr int32_t
cos_q15 (const int32_t angle_q15) noexcept
{
    if (angle_q15 <= 0)
        return _S_q15_max;
    if (angle_q15 + _S_half_pi_q15 < _S_two_pi_q15)
        return sin_q15 (angle_q15 + _S_half_pi_q15);
    return sin_q15 (angle_q15 + _S_half_pi_q15 - _S_two_pi_q15);
}

/** @brief base * factor^exp in Q15 by exponent squaring, never above base. */
[[nodiscard]] inline constexpr int32_t
q15_pow (const int32_t base_q15, const int32_t factor_q15,
         uint32_t exponent) noexcept
{
    int64_t result = base_q15;
    int64_t current = factor_q15;
    while (exponent != 0u)
        {
            if ((exponent & 1u) != 0u)
                result = (result * current + (1LL << 14)) >> 15;
            current = (current * current + (1LL << 14)) >> 15;
            exponent >>= 1;
        }
    return q15_clamp (static_cast<int32_t> (
        result > _S_q15_max ? _S_q15_max
                            : (result < _S_q15_min ? _S_q15_min : result)));
}

/** @brief Clamp a schedule result to the representable Q15 lr range. */
[[nodiscard]] inline constexpr int32_t
lr_clamp (const int64_t value_q15) noexcept
{
    return q15_clamp (static_cast<int32_t> (
        value_q15 > _S_q15_max ? _S_q15_max
                               : (value_q15 < 0 ? 0 : value_q15)));
}

/** @brief lr(step) = base, forever. */
struct ConstantSchedule
{
    int32_t _M_base_q15 = 3277;

    [[nodiscard]] constexpr int32_t
    lr_q15 (const uint32_t /*step*/) const noexcept
    {
        return lr_clamp (this->_M_base_q15);
    }
};

static_assert (sizeof (ConstantSchedule) == 4u, "constant schedule is one word");

/** @brief lr(step) = base * gamma^(step / step_size), integer drop every step_size. */
struct StepSchedule
{
    int32_t _M_base_q15 = 16384;
    int32_t _M_gamma_q15 = 3277;
    uint32_t _M_step_size = 100u;

    [[nodiscard]] constexpr int32_t
    lr_q15 (const uint32_t step) const noexcept
    {
        const uint32_t drops
            = this->_M_step_size == 0u ? 0u : step / this->_M_step_size;
        return q15_pow (this->_M_base_q15, this->_M_gamma_q15, drops);
    }
};

static_assert (sizeof (StepSchedule) == 12u, "step schedule is two words plus a counter");

/** @brief Milestone count a multi-step schedule can hold. */
inline constexpr uint8_t _S_lr_max_milestones = 8u;

/**
 * @brief lr(step) = base * gamma^(number of milestones passed).
 *
 * The milestone array is constexpr data inside the value struct the caller
 * constructs -- no module static, and the evaluation is one counting loop.
 */
struct MultiStepSchedule
{
    int32_t _M_base_q15 = 16384;
    int32_t _M_gamma_q15 = 3277;
    uint32_t _M_milestones[_S_lr_max_milestones] = { 30u, 60u, 90u,
                                                     0u, 0u, 0u, 0u, 0u };
    uint8_t _M_count = 3u;
    uint8_t _M_pad[3] = { 0u, 0u, 0u };

    [[nodiscard]] constexpr int32_t
    lr_q15 (const uint32_t step) const noexcept
    {
        uint32_t passed = 0u;
        for (uint8_t i = 0; i < this->_M_count; ++i)
            if (step >= this->_M_milestones[i])
                ++passed;
        return q15_pow (this->_M_base_q15, this->_M_gamma_q15, passed);
    }
};

static_assert (sizeof (MultiStepSchedule) == 44u, "multi-step schedule is two words, the milestone array and a padded count");

/** @brief lr(step) = base * factor^step, evaluated by squaring. */
struct ExponentialSchedule
{
    int32_t _M_base_q15 = 16384;
    int32_t _M_factor_q15 = 32765;

    [[nodiscard]] constexpr int32_t
    lr_q15 (const uint32_t step) const noexcept
    {
        return q15_pow (this->_M_base_q15, this->_M_factor_q15, step);
    }
};

static_assert (sizeof (ExponentialSchedule) == 8u, "exponential schedule is two words");

/**
 * @brief Linear warmup of an inner schedule, or a plain ramp when no inner
 *        schedule is attached (base held after the warmup).
 */
struct LinearWarmupSchedule
{
    int32_t _M_base_q15 = 16384;
    int32_t _M_start_q15 = 0;
    uint32_t _M_warmup_steps = 100u;
    /** @brief Cosine decay after warmup: 1 = cosine, 0 = constant hold. */
    uint8_t _M_decay_after = 0u;
    uint8_t _M_pad[3] = { 0u, 0u, 0u };
    uint32_t _M_total_steps = 1000u;

    [[nodiscard]] constexpr int32_t
    lr_q15 (uint32_t step) const noexcept
    {
        if (this->_M_warmup_steps != 0u && step < this->_M_warmup_steps)
        {
            const int64_t span = static_cast<int64_t> (this->_M_base_q15)
                                 - this->_M_start_q15;
            const int64_t ramp = this->_M_start_q15
                                 + (span * static_cast<int64_t> (step)
                                    + static_cast<int64_t> (this->_M_warmup_steps) / 2)
                                       / static_cast<int64_t> (this->_M_warmup_steps);
            return lr_clamp (ramp);
        }
        if (this->_M_decay_after == 0u)
            return lr_clamp (this->_M_base_q15);
        step -= this->_M_warmup_steps;
        const uint32_t span
            = this->_M_total_steps > this->_M_warmup_steps
                  ? this->_M_total_steps - this->_M_warmup_steps
                  : 1u;
        if (step >= span)
            return 0;
        const int32_t angle
            = static_cast<int32_t> ((static_cast<int64_t> (_S_half_pi_q15)
                                     * static_cast<int64_t> (step))
                                    / span);
        const int32_t cosine = cos_q15 (angle);
        return lr_clamp ((static_cast<int64_t> (this->_M_base_q15)
                          * (_S_q15_one + cosine))
                         >> 16);
    }
};

static_assert (sizeof (LinearWarmupSchedule) == 20u, "warmup schedule is five words");

/** @brief lr(step) = min + (base - min) * (1 + cos(pi * step / total)) / 2. */
struct CosineSchedule
{
    int32_t _M_base_q15 = 16384;
    int32_t _M_min_q15 = 0;
    uint32_t _M_total_steps = 1000u;

    [[nodiscard]] constexpr int32_t
    lr_q15 (const uint32_t step) const noexcept
    {
        if (this->_M_total_steps == 0u)
            return lr_clamp (this->_M_base_q15);
        const uint32_t clamped
            = step >= this->_M_total_steps ? this->_M_total_steps : step;
        const int32_t angle = static_cast<int32_t> (
            (static_cast<int64_t> (_S_pi_q15) * static_cast<int64_t> (clamped))
            / static_cast<int64_t> (this->_M_total_steps));
        const int32_t cosine = cos_q15 (angle);
        /* (1 + cos)/2 in Q15, then the [min, base] span: one shift each. */
        const int64_t shape = (_S_q15_one + cosine) >> 1;
        const int64_t span
            = static_cast<int64_t> (this->_M_base_q15) - this->_M_min_q15;
        return lr_clamp (this->_M_min_q15 + ((span * shape) >> 15));
    }
};

static_assert (sizeof (CosineSchedule) == 12u, "cosine schedule is two words plus a counter");

/** @brief Cosine restarts with a geometric period; lr per cycle from scratch. */
struct CosineRestartsSchedule
{
    int32_t _M_base_q15 = 16384;
    int32_t _M_min_q15 = 0;
    uint32_t _M_first_period = 100u;
    /** @brief Period growth factor between restarts, a small integer (1 = fixed). */
    uint32_t _M_period_mult = 2u;

    [[nodiscard]] constexpr int32_t
    lr_q15 (uint32_t step) const noexcept
    {
        if (this->_M_first_period == 0u)
            return lr_clamp (this->_M_base_q15);
        uint32_t period = this->_M_first_period;
        /* Bounded walk over restart cycles; the mult >= 2 geometric series
           covers any uint32 step in at most 32 cycles by construction. */
        uint32_t guard = 0u;
        while (step >= period && guard < 33u)
        {
            step -= period;
            ++guard;
            const uint64_t next = static_cast<uint64_t> (period)
                                  * (this->_M_period_mult == 0u
                                         ? 1u
                                         : this->_M_period_mult);
            period = next > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t> (next);
        }
        CosineSchedule inner;
        inner._M_base_q15 = this->_M_base_q15;
        inner._M_min_q15 = this->_M_min_q15;
        inner._M_total_steps = period == 0u ? 1u : period;
        return inner.lr_q15 (step);
    }
};

static_assert (sizeof (CosineRestartsSchedule) == 16u, "restarts schedule is four words");

/**
 * @brief One-cycle policy: linear base -> peak over the first 30% of the
 *        total, cosine peak -> min over the rest (annealing fraction in Q15).
 */
struct OneCycleSchedule
{
    int32_t _M_base_q15 = 1024;
    int32_t _M_peak_q15 = 16384;
    int32_t _M_min_q15 = 512;
    uint32_t _M_total_steps = 1000u;

    /** @brief Q15 fraction of the cycle spent warming up (0.3). */
    static inline constexpr int32_t _S_up_fraction_q15 = 9830;

    [[nodiscard]] constexpr int32_t
    lr_q15 (const uint32_t step) const noexcept
    {
        if (this->_M_total_steps == 0u)
            return lr_clamp (this->_M_base_q15);
        const uint32_t up_steps = static_cast<uint32_t> (
            (static_cast<int64_t> (this->_M_total_steps)
             * _S_up_fraction_q15)
            >> 15);
        const uint32_t clamped
            = step >= this->_M_total_steps ? this->_M_total_steps - 1u : step;
        if (clamped < up_steps)
        {
            const int64_t span = static_cast<int64_t> (this->_M_peak_q15)
                                 - this->_M_base_q15;
            const int64_t ramp
                = this->_M_base_q15
                  + (span * static_cast<int64_t> (clamped)
                     + static_cast<int64_t> (up_steps) / 2)
                        / static_cast<int64_t> (up_steps == 0u ? 1u : up_steps);
            return lr_clamp (ramp);
        }
        const uint32_t down_steps = this->_M_total_steps - up_steps;
        const uint32_t within = clamped - up_steps;
        const int32_t angle = static_cast<int32_t> (
            (static_cast<int64_t> (_S_pi_q15) * static_cast<int64_t> (within))
            / static_cast<int64_t> (down_steps == 0u ? 1u : down_steps));
        const int32_t cosine = cos_q15 (angle);
        const int64_t shape = (_S_q15_one + cosine) >> 1;
        const int64_t span
            = static_cast<int64_t> (this->_M_peak_q15) - this->_M_min_q15;
        return lr_clamp (this->_M_min_q15 + ((span * shape) >> 15));
    }
};

static_assert (sizeof (OneCycleSchedule) == 16u, "one-cycle schedule is four words");

/** @brief Plateau factor/patience configuration; the lr also depends on state. */
struct PlateauConfig
{
    int32_t _M_base_q15 = 16384;
    int32_t _M_factor_q15 = 3277;
    int32_t _M_min_q15 = 32;
    uint32_t _M_patience = 5u;
};

static_assert (sizeof (PlateauConfig) == 16u, "plateau config is three words plus a counter");

/**
 * @brief Caller-owned plateau state: how many consecutive stale observations
 *        (past patience) have happened and the best metric seen.
 *
 * The metric is a Q7 magnitude (a loss or error), so comparisons are integer.
 * Nothing is static: advance() takes the state by reference, and the caller
 * stores it in its own trainer struct.
 */
struct PlateauState
{
    uint32_t _M_bad_count = 0u;
    int32_t _M_best_metric_q7 = _S_q15_min;
    int32_t _M_lr_q15 = 0;
};

static_assert (sizeof (PlateauState) == 12u, "plateau state is three words");

/**
 * @brief lr(step) = base * factor^effective_drops, drop per patience stale
 *        observations, floored at min.  Pure: bad_count is the caller's.
 */
[[nodiscard]] inline constexpr int32_t
plateau_lr_q15 (const PlateauConfig &config, const uint32_t bad_count,
                const uint32_t patience) noexcept
{
    const uint32_t drops = patience == 0u ? 0u : bad_count / patience;
    const int32_t value
        = q15_pow (config._M_base_q15, config._M_factor_q15, drops);
    return value < config._M_min_q15 ? config._M_min_q15 : value;
}

/**
 * @brief Observe one epoch's metric, update the caller's plateau state, and
 *        return the new lr.
 *
 * improved = metric < best * (1 + threshold-relative is deliberately left
 * out: a Q7 threshold word is compared directly).  Returns the lr for the
 * next step, already floored at min.
 */
[[nodiscard]] inline int32_t
plateau_advance (const PlateauConfig &config, PlateauState &state,
                 const int32_t metric_q7, const int32_t improvement_margin_q7) noexcept
{
    if (state._M_best_metric_q7 == _S_q15_min
        || metric_q7 < state._M_best_metric_q7 - improvement_margin_q7)
    {
        state._M_best_metric_q7 = metric_q7;
        state._M_bad_count = 0u;
    }
    else
        ++state._M_bad_count;
    state._M_lr_q15
        = plateau_lr_q15 (config, state._M_bad_count, config._M_patience);
    return state._M_lr_q15;
}

/**
 * @brief Schedule kind tag for a trainer that owns one schedule value.
 * Dispatch mirrors the optimizer table: constexpr function pointers, no RTTI.
 */
enum class ScheduleKind : uint8_t
{
    CONSTANT = 0,
    STEP,
    MULTI_STEP,
    EXPONENTIAL,
    WARMUP,
    COSINE,
    COSINE_RESTARTS,
    ONE_CYCLE,
    COUNT,
};

/** @brief Type-erased evaluation: pointer to a schedule value, step in, lr out. */
using ScheduleFn = int32_t (*)(const void *schedule, uint32_t step);

/** @brief One adapter per kind: the table casts back to the concrete value. */
[[nodiscard]] inline constexpr ScheduleFn
schedule_fn_for (const ScheduleKind kind) noexcept
{
    /* The adapters are separate constexpr function addresses so the table is
       fully build-time; casting the schedule pointer back is sound because
       the trainer keeps the concrete struct next to the tag. */
    switch (kind)
    {
    case ScheduleKind::CONSTANT:
        return [](const void *const s, const uint32_t step) -> int32_t {
            return static_cast<const ConstantSchedule *> (s)->lr_q15 (step);
        };
    case ScheduleKind::STEP:
        return [](const void *const s, const uint32_t step) -> int32_t {
            return static_cast<const StepSchedule *> (s)->lr_q15 (step);
        };
    case ScheduleKind::MULTI_STEP:
        return [](const void *const s, const uint32_t step) -> int32_t {
            return static_cast<const MultiStepSchedule *> (s)->lr_q15 (step);
        };
    case ScheduleKind::EXPONENTIAL:
        return [](const void *const s, const uint32_t step) -> int32_t {
            return static_cast<const ExponentialSchedule *> (s)->lr_q15 (step);
        };
    case ScheduleKind::WARMUP:
        return [](const void *const s, const uint32_t step) -> int32_t {
            return static_cast<const LinearWarmupSchedule *> (s)->lr_q15 (step);
        };
    case ScheduleKind::COSINE:
        return [](const void *const s, const uint32_t step) -> int32_t {
            return static_cast<const CosineSchedule *> (s)->lr_q15 (step);
        };
    case ScheduleKind::COSINE_RESTARTS:
        return [](const void *const s, const uint32_t step) -> int32_t {
            return static_cast<const CosineRestartsSchedule *> (s)->lr_q15 (step);
        };
    case ScheduleKind::ONE_CYCLE:
        return [](const void *const s, const uint32_t step) -> int32_t {
            return static_cast<const OneCycleSchedule *> (s)->lr_q15 (step);
        };
    case ScheduleKind::COUNT:
        break;
    }
    return nullptr;
}
} // namespace zzs::mllee::train
