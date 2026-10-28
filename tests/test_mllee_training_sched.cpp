/* Tests for the training control layer: mllee/train_lr_schedule.hpp,
 * mllee/train_loss_scale.hpp and mllee/train_grad_clip.hpp.
 *
 * Every expectation below is an exact integer transcribed from the header's own
 * recurrence -- the Q15 squaring ladder (a * b + 2^14) >> 15 with C++23 floor
 * shift semantics, the truncating ramp and angle divisions, the half-away
 * div_round, the saturating int16 narrowing, the digit-pair integer square root
 * -- never a floating model with a margin.  Where a header comment claims a
 * bound the arithmetic does not deliver, the value the code actually produces
 * is pinned and the gap is reported rather than papered over.
 *
 * Two helpers are deliberately independent implementations rather than copies:
 * exact_isqrt() is a division-only binary-search floor root, so its agreement
 * with grad_norm_raw_q15 / clip_grad_norm_q15 is evidence instead of a
 * tautology, and ref_div_round() is the (2*|n| + |d|) / (2*|d|) form of the
 * module's half-away-from-zero division.
 *
 * All three headers claim "Pool cost: zero", and the module never emits the
 * -128 code, so pool counters and the code rails are checked after the fact
 * rather than assumed. */

#include "mllee/train_grad_clip.hpp"
#include "mllee/train_loss_scale.hpp"
#include "mllee/train_lr_schedule.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_simd.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace zzs::mllee;
using namespace zzs::mllee::train;

namespace
{
constexpr uint32_t _cap = 64u;   /* widest lane stream a Param fixture holds */
constexpr uint32_t _guard = 8u;  /* sentinel lanes on each side of a stream   */
constexpr int32_t _one = 32768;  /* _S_q15_one                               */
constexpr int32_t _hi = 32767;   /* _S_q15_max                               */
constexpr int32_t _lo = -32768;  /* _S_q15_min                               */
constexpr int16_t _sentinel = static_cast<int16_t> (0x5A5A);
constexpr int8_t _sentinel8 = static_cast<int8_t> (0x5A);

/** @brief The Q15 squaring ladder: (a * b + 2^14) >> 15, floor shift. */
constexpr int64_t
ladder (const int64_t a, const int64_t b) noexcept
{
    return (a * b + (1LL << 14)) >> 15;
}

/** @brief (x + 2^14) >> 15 on an int64 product: ties toward +infinity. */
constexpr int32_t
round_q15 (const int64_t x) noexcept
{
    return static_cast<int32_t> ((x + 16384) >> 15);
}

/** @brief Saturation into the int16 lane rails. */
constexpr int64_t
clamp_lane (const int64_t x) noexcept
{
    return x > _hi ? _hi : (x < _lo ? _lo : x);
}

/** @brief The [-127, 127] code rail: -128 is never emitted. */
constexpr int64_t
i8_rail (const int64_t x) noexcept
{
    return x > 127 ? 127 : (x < -127 ? -127 : x);
}

/** @brief Exact floor square root by binary search on division only. */
int64_t
exact_isqrt (const int64_t value) noexcept
{
    if (value <= 0)
        return 0;
    int64_t low = 0;
    int64_t high = value;
    while (low < high)
    {
        const int64_t mid = low + (high - low + 1) / 2;
        if (mid > value / mid)
            high = mid - 1;
        else
            low = mid;
    }
    return low;
}

/** @brief Half-away-from-zero division, written in the (2n + d) / (2d) form
 *         so it agrees with div_round() only if both really are round-half
 *         away from zero. */
constexpr int32_t
ref_div_round (int32_t numerator, const int32_t denominator) noexcept
{
    if (denominator == 0)
        return 0;
    const bool negative = (numerator < 0) != (denominator < 0);
    if (numerator < 0)
        numerator = static_cast<int32_t> (-static_cast<int64_t> (numerator));
    const int64_t wide_denominator
        = denominator < 0 ? -static_cast<int64_t> (denominator)
                          : static_cast<int64_t> (denominator);
    const int64_t quotient
        = (2 * static_cast<int64_t> (numerator) + wide_denominator)
          / (2 * wide_denominator);
    return static_cast<int32_t> (negative ? -quotient : quotient);
}

/** @brief int64 sum of squares, the accumulator clip_grad_norm_q15 uses. */
int64_t
sum_sq (const int16_t *const lanes, const uint32_t count) noexcept
{
    int64_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<int64_t> (lanes[i]) * lanes[i];
    return total;
}

/** @brief First lane whose value differs from the reference, else -1. */
int
first_diff (const int16_t *const got, const int32_t *const want,
            const uint32_t n) noexcept
{
    for (uint32_t i = 0; i < n; ++i)
        if (got[i] != want[i])
            return static_cast<int> (i);
    return -1;
}

/** @brief First lane whose code is off the [-127, 127] rail, is -128, or does
 *         not match the documented code narrowing, else -1. */
int
first_code_rail_violation (const int16_t *const lanes, const int8_t *const codes,
                           const uint32_t n) noexcept
{
    for (uint32_t i = 0; i < n; ++i)
    {
        const int8_t want = q15_to_code (lanes[i]);
        if (codes[i] != want || codes[i] < -127 || codes[i] > 127
            || codes[i] == static_cast<int8_t> (-128))
            return static_cast<int> (i);
    }
    return -1;
}

/** @brief A guarded code + lane stream, so an over-write by a kernel is
 *         visible and the int8 side can be rail-checked. */
struct Param
{
    alignas (8) int16_t _lane_raw[_guard + _cap + _guard];
    alignas (8) int8_t _code_raw[_guard + _cap + _guard];
    uint32_t _count = 0u;

    int16_t *
    lanes () noexcept
    {
        return _lane_raw + _guard;
    }

    const int16_t *
    lanes () const noexcept
    {
        return _lane_raw + _guard;
    }

    int8_t *
    codes () noexcept
    {
        return _code_raw + _guard;
    }

    const int8_t *
    codes () const noexcept
    {
        return _code_raw + _guard;
    }

    /** @brief Sentinels everywhere, then pattern[0..count) into the lanes. */
    void
    prime (const int32_t *const pattern, const uint32_t count) noexcept
    {
        _count = count;
        for (uint32_t i = 0; i < _guard; ++i)
            {
                _lane_raw[i] = _sentinel;
                _lane_raw[_guard + count + i] = _sentinel;
                _code_raw[i] = _sentinel8;
                _code_raw[_guard + count + i] = _sentinel8;
            }
        for (uint32_t i = 0; i < _cap; ++i)
            {
                lanes ()[i] = i < count ? static_cast<int16_t> (pattern[i]) : 0;
                codes ()[i] = _sentinel8;
            }
    }

    /** @brief True when neither moat was written through. */
    bool
    guards_ok () const noexcept
    {
        for (uint32_t i = 0; i < _guard; ++i)
            if (_lane_raw[i] != _sentinel || _lane_raw[_guard + _count + i] != _sentinel
                || _code_raw[i] != _sentinel8
                || _code_raw[_guard + _count + i] != _sentinel8)
                return false;
        return true;
    }

    Trainable
    view (const uint32_t flags
              = static_cast<uint32_t> (Flag::RequiresGrad)) noexcept
    {
        return Trainable::make (codes (), lanes (),
                                Shape::vector (static_cast<uint16_t> (_count)),
                                flags);
    }

    Trainable
    no_grad () noexcept
    {
        return Trainable::make (codes (), nullptr,
                                Shape::vector (static_cast<uint16_t> (_count)),
                                static_cast<uint32_t> (Flag::RequiresGrad));
    }
};

/* One recorded node: every operand lane receives the seeded loss-scale gradient
   through Trainable::add_grad, so a pre-poisoned lane clamps and the resulting
   saturation count is visible on the tape. */
int32_t g_seen_scale_q15 = 0;
uint32_t g_backward_calls = 0u;

Status
seed_backward (const Trainable &, const Trainable &grad_output,
               GradContext &context) noexcept
{
    g_seen_scale_q15 = context._M_loss_scale_q15;
    ++g_backward_calls;
    const int32_t seed = grad_output._M_grad[0];
    const Trainable &operand = context._M_operands[0];
    for (uint32_t i = 0; i < operand._M_count; ++i)
        if (operand.add_grad (i, seed))
            ++context._M_saturation_events;
    return Status::OK;
}
} // namespace

TEST_CASE ("folded Taylor sine matches exact integer pins",
           "[mllee][train][sched]")
{
    STATIC_REQUIRE (_S_pi_q15 == 102944);
    STATIC_REQUIRE (_S_half_pi_q15 == 51472);
    STATIC_REQUIRE (_S_two_pi_q15 == 205887);

    /* bracket = 32768 - u/6 + u2/120 - u3/5040 with u = (x*x) >> 15, then
       (x * bracket + 16384) >> 15.  x = 4096: u = 512, u2 = 8, u3 = 0,
       bracket = 32768 - 85 + 0 - 0 = 32683, 4096 * 32683 = 133867008,
       (133867008 + 16384) >> 15 = 4085. */
    CHECK (sin_first_quadrant_q15 (0) == 0);
    CHECK (sin_first_quadrant_q15 (1) == 1);
    CHECK (sin_first_quadrant_q15 (2) == 2);
    CHECK (sin_first_quadrant_q15 (3) == 3);
    CHECK (sin_first_quadrant_q15 (512) == 512);
    CHECK (sin_first_quadrant_q15 (4096) == 4085);
    CHECK (sin_first_quadrant_q15 (8192) == 8107);
    CHECK (sin_first_quadrant_q15 (16384) == 15710);
    CHECK (sin_first_quadrant_q15 (32768) == 27573);
    CHECK (sin_first_quadrant_q15 (40000) == 30779);
    CHECK (sin_first_quadrant_q15 (_S_half_pi_q15) == 32762);

    /* The polynomial is odd, so a negative argument evaluates straight through:
       x = -1000 gives u = 30, u2 = 0, bracket = 32763,
       (-32763000 + 16384) >> 15 = -1000 (floor). */
    CHECK (sin_first_quadrant_q15 (-1) == -1);
    CHECK (sin_first_quadrant_q15 (-1000) == -1000);

    /* Quadrant peak and the documented error budget: the header claims
       |err| < 3e-5 on [0, pi/2], but the largest value the fold reaches is
       32765 at x = 51365 -- above both the pi/2 endpoint value (32762) and
       1.0 rounded into the rail -- and the sequence decreases in places, so
       the series is not monotone and the claimed bound is not met. */
    CHECK (sin_first_quadrant_q15 (51365) == 32765);
    CHECK (sin_first_quadrant_q15 (51366) == 32764);
    CHECK (sin_first_quadrant_q15 (51471) == 32762);
    CHECK (sin_first_quadrant_q15 (51472) == 32762);
    int first_decrease = -1;
    int32_t worst_step = 0;
    for (int32_t x = 0; x < _S_half_pi_q15; ++x)
        {
            const int32_t delta
                = sin_first_quadrant_q15 (x + 1) - sin_first_quadrant_q15 (x);
            if (delta < 0 && first_decrease < 0)
                first_decrease = x;
            if (delta < worst_step)
                worst_step = delta;
        }
    CHECK (first_decrease == 21368);
    CHECK (worst_step == -3);
    CHECK (sin_first_quadrant_q15 (first_decrease) == 19886);
    CHECK (sin_first_quadrant_q15 (first_decrease + 1) == 19885);
}

TEST_CASE ("sine and cosine folding cover the whole circle exactly",
           "[mllee][train][sched]")
{
    /* sin: 0 and everything at or below it is the documented 0. */
    CHECK (sin_q15 (0) == 0);
    CHECK (sin_q15 (-1) == 0);
    CHECK (sin_q15 (_lo) == 0);
    CHECK (sin_q15 (_S_half_pi_q15) == 32762);
    CHECK (sin_q15 (_S_pi_q15) == 0);      /* sin1(pi - pi) = sin1(0) */
    CHECK (sin_q15 (154416) == -32762);    /* 3 pi / 2: -sin1(51472)  */
    CHECK (sin_q15 (205886) == -1);        /* -sin1(1)                */
    CHECK (sin_q15 (205887) == 0);         /* %= 2 pi -> 0 -> 0       */
    CHECK (sin_q15 (205894) == 7);         /* %= 2 pi -> 7            */
    CHECK (sin_q15 (618895) == 1234);      /* %= 2 pi -> 1234         */
    CHECK (sin_q15 (205887 * 4) == 0);

    /* cos: the <= 0 rail is _S_q15_max, not 1.0, and the shift identity is
       cos(x) = sin(x + pi/2) up to the single wrap the header performs. */
    CHECK (cos_q15 (0) == _hi);
    CHECK (cos_q15 (-1) == _hi);
    CHECK (cos_q15 (1) == 32762);
    CHECK (cos_q15 (4096) == 32510);
    CHECK (cos_q15 (25736) == 23170);
    CHECK (cos_q15 (_S_half_pi_q15) == 0);
    CHECK (cos_q15 (_S_pi_q15) == -32762);
    CHECK (cos_q15 (154416) == 1);        /* 3 pi / 2 + pi / 2 = 2 pi -> sin1(1) */
    CHECK (cos_q15 (205886) == 32762);
    CHECK (cos_q15 (205887) == 32762);

    /* The wrap boundary is a one-sided step: 154415 + 51472 == 2 pi exactly,
       so it takes the subtract branch and lands on the 0 -> rail edge, while
       the mirror identity therefore fails only at the single point x = 2 pi
       (cos(2 pi) = 32762 against cos(0) = 32767). */
    CHECK (cos_q15 (154415) == 0);
    CHECK (cos_q15 (154416) == 1);
    CHECK (cos_q15 (_S_two_pi_q15) != cos_q15 (0));

    int32_t sin_max = _lo;
    int32_t sin_min = _hi;
    int32_t cos_max = _lo;
    int32_t cos_min = _hi;
    uint32_t symmetry_breaks = 0u;
    uint32_t shift_breaks = 0u;
    uint32_t mirror_breaks = 0u;
    uint32_t period_breaks = 0u;
    uint32_t cos_period_breaks = 0u;
    uint32_t fold_breaks = 0u;
    for (int32_t x = 0; x <= _S_two_pi_q15; ++x)
        {
            const int32_t s = sin_q15 (x);
            const int32_t c = cos_q15 (x);
            if (s > sin_max)
                sin_max = s;
            if (s < sin_min)
                sin_min = s;
            if (c > cos_max)
                cos_max = c;
            if (c < cos_min)
                cos_min = c;
            /* pi reflection and the two negative-quadrant folds. */
            if (x <= _S_pi_q15 && s != sin_q15 (_S_pi_q15 - x))
                ++symmetry_breaks;
            if (x >= _S_pi_q15 && x <= _S_pi_q15 + _S_half_pi_q15
                && s != -sin_first_quadrant_q15 (x - _S_pi_q15))
                ++fold_breaks;
            if (x > _S_pi_q15 + _S_half_pi_q15
                && s != -sin_first_quadrant_q15 (_S_two_pi_q15 - x))
                ++fold_breaks;
            if (x >= 1 && x + _S_half_pi_q15 < _S_two_pi_q15
                && c != sin_q15 (x + _S_half_pi_q15))
                ++shift_breaks;
            if (x >= 1 && x < _S_two_pi_q15 && c != cos_q15 (_S_two_pi_q15 - x))
                ++mirror_breaks;
            if (x >= 1 && s != sin_q15 (x + _S_two_pi_q15))
                ++period_breaks;
            if (x >= 1 && c != cos_q15 (x + _S_two_pi_q15))
                ++cos_period_breaks;
        }
    CHECK (sin_max == 32765);
    CHECK (sin_min == -32765);
    CHECK (cos_max == _hi);
    CHECK (cos_min == -32765);
    CHECK (symmetry_breaks == 0u);
    CHECK (fold_breaks == 0u);
    CHECK (shift_breaks == 0u);
    CHECK (mirror_breaks == 0u);
    CHECK (period_breaks == 0u);
    CHECK (cos_period_breaks == 0u);
}

TEST_CASE ("q15_pow ladder and lr_clamp rails", "[mllee][train][sched]")
{
    /* gamma = 3277 (0.100033) applied to base 16384 (0.5): every rung is one
       ladder step, so 16384 -> (16384*3277 + 16384) >> 15 = 1639 -> 164 -> 16
       -> 2 -> 0.  Once the running value is 0 it cannot come back. */
    CHECK (q15_pow (16384, 3277, 0u) == 16384);
    CHECK (q15_pow (16384, 3277, 1u) == 1639);
    CHECK (q15_pow (16384, 3277, 2u) == 164);
    CHECK (q15_pow (16384, 3277, 3u) == 16);
    CHECK (q15_pow (16384, 3277, 4u) == 2);
    CHECK (q15_pow (16384, 3277, 5u) == 0);
    CHECK (q15_pow (16384, 3277, 40u) == 0);

    /* factor 32768 is exactly 1.0 in the ladder: (r * 32768 + 16384) >> 15
       = r + 0 -> floor, so the base is fixed for every exponent. */
    CHECK (q15_pow (16384, 32768, 7u) == 16384);
    CHECK (q15_pow (3277, 32768, 1000u) == 3277);
    CHECK (q15_pow (40000, 32768, 0u) == _hi);   /* the base itself clamps */
    CHECK (q15_pow (_lo, 32768, 0u) == _lo);     /* below, not to 0        */

    /* A factor one ulp under 1.0 still rounds back to the base at 16384:
       (16384 * 32767 + 16384) >> 15 = 536870912 >> 15 = 16384. */
    CHECK (q15_pow (16384, _hi, 1u) == 16384);
    CHECK (q15_pow (16384, _hi, 2u) == 16383);
    CHECK (q15_pow (16384, _hi, 1000u) == 15891);

    /* Exact halving: factor 16384 is 0.5, so the ladder is a plain >> 1. */
    const int32_t halved[] = { 16384, 8192, 4096, 2048, 1024, 512, 256, 128 };
    for (uint32_t k = 0; k < 8u; ++k)
        CHECK (q15_pow (16384, 16384, k) == halved[k]);

    /* Degenerate factors. */
    CHECK (q15_pow (16384, 0, 0u) == 16384);
    CHECK (q15_pow (16384, 0, 1u) == 0);
    CHECK (q15_pow (16384, 0, 2u) == 0);
    CHECK (q15_pow (16384, 1000, 12u) == 0);
    CHECK (q15_pow (100, 32000, 10u) == 79);
    CHECK (q15_pow (3277, 3277, 3u) == 3);
    CHECK (q15_pow (32767, 3277, 2u) == 328);

    /* A negative factor alternates the sign: -1.0 squared is +1.0 in the
       ladder, (16384 * 32768 + 16384) >> 15 = 16384. */
    CHECK (q15_pow (16384, _lo, 1u) == -16384);
    CHECK (q15_pow (16384, _lo, 2u) == 16384);
    CHECK (q15_pow (16384, _lo, 3u) == -16384);

    /* Invariants over the whole exponent range that matters: a non-negative
       factor at or below 1.0 can never raise the magnitude, and the result
       always sits inside the Q15 rails. */
    for (const int32_t factor : {0, 1, 3277, 16384, 32765, 32767, 32768})
        {
            int32_t previous = q15_pow (16384, factor, 0u);
            CHECK (previous == 16384);
            for (uint32_t e = 1u; e <= 400u; ++e)
                {
                    const int32_t value = q15_pow (16384, factor, e);
                    CHECK (value <= previous);
                    CHECK (value >= 0);
                    CHECK (value <= _hi);
                    previous = value;
                }
        }
    for (const int32_t base : {_lo, -1, 0, 1, 1024, 16384, _hi, 40000})
        for (uint32_t e = 0u; e <= 40u; ++e)
            CHECK (q15_pow (base, 32768, e) == q15_clamp (base));
}

TEST_CASE ("lr_clamp keeps a schedule inside the representable lr range",
           "[mllee][train][sched]")
{
    CHECK (lr_clamp (0) == 0);
    CHECK (lr_clamp (1) == 1);
    CHECK (lr_clamp (_hi) == _hi);
    CHECK (lr_clamp (_one) == _hi);      /* > rail -> rail, not wrap        */
    CHECK (lr_clamp (1000000) == _hi);
    CHECK (lr_clamp (-1) == 0);          /* negative clamps to 0, not _lo   */
    CHECK (lr_clamp (_lo) == 0);
    CHECK (lr_clamp (-1000000) == 0);
    CHECK (lr_clamp (static_cast<int64_t> (1) << 40) == _hi);
}

TEST_CASE ("constant, step, multi-step and exponential schedules sit on ladder "
           "rungs",
           "[mllee][train][sched]")
{
    STATIC_REQUIRE (sizeof (ConstantSchedule) == 4u);
    STATIC_REQUIRE (sizeof (StepSchedule) == 12u);
    STATIC_REQUIRE (sizeof (MultiStepSchedule) == 44u);
    STATIC_REQUIRE (sizeof (ExponentialSchedule) == 8u);
    STATIC_REQUIRE (_S_lr_max_milestones == 8u);

    SECTION ("constant ignores the step")
    {
        ConstantSchedule constant;
        CHECK (constant._M_base_q15 == 3277);
        CHECK (constant.lr_q15 (0u) == 3277);
        CHECK (constant.lr_q15 (1u) == 3277);
        CHECK (constant.lr_q15 (0xFFFFFFFFu) == 3277);

        constant._M_base_q15 = _one;
        CHECK (constant.lr_q15 (7u) == _hi);   /* 1.0 is not representable */
        constant._M_base_q15 = -1;
        CHECK (constant.lr_q15 (7u) == 0);     /* lr_clamp floors at 0     */
        constant._M_base_q15 = 0;
        CHECK (constant.lr_q15 (7u) == 0);
    }

    SECTION ("step drops once per whole step_size")
    {
        StepSchedule step;
        CHECK (step._M_base_q15 == 16384);
        CHECK (step._M_gamma_q15 == 3277);
        CHECK (step._M_step_size == 100u);

        /* drops = step / 100, then the q15_pow ladder rung for that count. */
        const uint32_t steps[] = { 0u,  99u,  100u, 199u, 200u,  299u,
                                   300u, 399u, 400u, 499u, 500u, 1000u };
        const int32_t want[] = { 16384, 16384, 1639, 1639, 164, 164,
                                 16,    16,    2,    2,    0,  0 };
        for (uint32_t i = 0; i < 12u; ++i)
            CHECK (step.lr_q15 (steps[i]) == want[i]);
        CHECK (step.lr_q15 (5000u) == 0);
        CHECK (step.lr_q15 (0xFFFFFFFFu) == 0);

        /* A zero step_size disables the division: the base is held forever. */
        step._M_step_size = 0u;
        CHECK (step.lr_q15 (0u) == 16384);
        CHECK (step.lr_q15 (1u) == 16384);
        CHECK (step.lr_q15 (0xFFFFFFFFu) == 16384);

        /* step_size 1 decays every single step, and the sequence is
           non-increasing and rail-bounded across the whole range. */
        step._M_step_size = 1u;
        int32_t previous = step.lr_q15 (0u);
        CHECK (previous == 16384);
        for (uint32_t s = 1u; s <= 2000u; ++s)
            {
                const int32_t lr = step.lr_q15 (s);
                CHECK (lr <= previous);
                CHECK (lr >= 0);
                CHECK (lr <= _hi);
                previous = lr;
            }
        CHECK (previous == 0);

        /* gamma exactly 1.0 pins the base at every drop count. */
        step._M_gamma_q15 = _one;
        for (const uint32_t s : {0u, 1u, 99u, 100u, 12345u, 0xFFFFFFFFu})
            CHECK (step.lr_q15 (s) == 16384);

        /* An out-of-rail base clamps before the ladder, so even many drops
           of an identity factor stay on the rail. */
        step._M_base_q15 = 40000;
        step._M_step_size = 1u;
        CHECK (step.lr_q15 (0u) == _hi);
        step._M_gamma_q15 = 3277;
        /* (40000 * 3277 + 16384) >> 15 = 131096384 >> 15 = 4000: the clamp is
           applied to the ladder result, not to the ladder input. */
        CHECK (step.lr_q15 (1u) == 4000);
        /* (40000 * 328 + 16384) >> 15 = 13136384 >> 15 = 400 (3277^2 rung). */
        CHECK (step.lr_q15 (2u) == 400);
        CHECK (step.lr_q15 (3u) == 40);
    }

    SECTION ("multi-step counts the milestones reached")
    {
        MultiStepSchedule multi;
        CHECK (multi._M_base_q15 == 16384);
        CHECK (multi._M_count == 3u);
        CHECK (multi._M_milestones[0] == 30u);
        CHECK (multi._M_milestones[2] == 90u);

        const uint32_t steps[] = { 0u, 29u, 30u, 59u, 60u, 89u, 90u, 1000u };
        const int32_t want[] = { 16384, 16384, 1639, 1639, 164, 164, 16, 16 };
        for (uint32_t i = 0; i < 8u; ++i)
            CHECK (multi.lr_q15 (steps[i]) == want[i]);

        /* >= means a milestone fires on the step itself, and a zero milestone
           is a real milestone: all three fire at step 0. */
        for (uint8_t i = 0; i < _S_lr_max_milestones; ++i)
            multi._M_milestones[i] = 0u;
        CHECK (multi.lr_q15 (0u) == 16);
        CHECK (multi.lr_q15 (0xFFFFFFFFu) == 16);

        multi._M_count = 0u;
        CHECK (multi.lr_q15 (0u) == 16384);
        CHECK (multi.lr_q15 (1234u) == 16384);

        /* The full eight-slot array at the default gamma: the 0.1 ladder
           collapses to zero from the fifth drop, so only the first four
           milestones change the answer (16384, 1639, 164, 16, 2, 0, 0, 0). */
        multi._M_count = 8u;
        for (uint8_t i = 0; i < 8u; ++i)
            multi._M_milestones[i] = 10u + 10u * i;
        CHECK (multi.lr_q15 (39u) == 16);    /* three milestones passed  */
        CHECK (multi.lr_q15 (40u) == 2);     /* four passed              */
        CHECK (multi.lr_q15 (50u) == 0);     /* five passed              */
        CHECK (multi.lr_q15 (79u) == 0);     /* seven passed             */
        CHECK (multi.lr_q15 (80u) == 0);
        CHECK (multi.lr_q15 (0xFFFFFFFFu) == 0);
    }

    SECTION ("exponential pow of the step")
    {
        ExponentialSchedule expo;
        CHECK (expo._M_base_q15 == 16384);
        CHECK (expo._M_factor_q15 == 32765);
        CHECK (expo.lr_q15 (0u) == 16384);
        CHECK (expo.lr_q15 (1u) == 16383);
        CHECK (expo.lr_q15 (2u) == 16381);
        CHECK (expo.lr_q15 (10u) == 16369);
        CHECK (expo.lr_q15 (100u) == 16234);
        CHECK (expo.lr_q15 (1000u) == 14945);
        CHECK (expo.lr_q15 (10000u) == 6536);

        expo._M_factor_q15 = _one;
        for (const uint32_t s : {0u, 1u, 7u, 100000u, 0xFFFFFFFFu})
            CHECK (expo.lr_q15 (s) == 16384);

        expo._M_factor_q15 = 3277;
        CHECK (expo.lr_q15 (3u) == 16);   /* same rungs as the step schedule */

        int32_t previous = -1;
        for (uint32_t s = 0u; s <= 1200u; ++s)
            {
                const int32_t lr = expo.lr_q15 (s);
                if (previous >= 0)
                    CHECK (lr <= previous);
                CHECK (lr >= 0);
                CHECK (lr <= _hi);
                previous = lr;
            }
    }
}

TEST_CASE ("linear warmup ramps, holds and decays with truncating arithmetic",
           "[mllee][train][sched]")
{
    STATIC_REQUIRE (sizeof (LinearWarmupSchedule) == 20u);

    /* start + (span * step + warmup / 2) / warmup with truncating division:
       step 4 is (16384 * 4 + 50) / 100 = 65586 / 100 = 655. */
    LinearWarmupSchedule warm;
    CHECK (warm._M_base_q15 == 16384);
    CHECK (warm._M_start_q15 == 0);
    CHECK (warm._M_warmup_steps == 100u);
    CHECK (warm._M_decay_after == 0u);
    CHECK (warm._M_total_steps == 1000u);

    const uint32_t ramp_steps[] = { 0u, 1u,  2u,  3u,  4u,  5u,
                                    6u, 7u,  50u, 99u, 100u, 1000u };
    const int32_t ramp_want[]
        = { 0, 164, 328, 492, 655, 819, 983, 1147, 8192, 16220, 16384, 16384 };
    for (uint32_t i = 0; i < 12u; ++i)
        CHECK (warm.lr_q15 (ramp_steps[i]) == ramp_want[i]);
    CHECK (warm.lr_q15 (0xFFFFFFFFu) == 16384);

    /* The ramp is non-decreasing when base >= start, and held after warmup. */
    for (uint32_t s = 0u; s < 99u; ++s)
        CHECK (warm.lr_q15 (s + 1u) >= warm.lr_q15 (s));

    /* A raised start shifts the whole ramp but keeps the same slope:
       4096 + (12288 * 50 + 50) / 100 = 4096 + 6144 = 10240. */
    warm._M_start_q15 = 4096;
    CHECK (warm.lr_q15 (0u) == 4096);
    CHECK (warm.lr_q15 (50u) == 10240);
    CHECK (warm.lr_q15 (99u) == 16261);
    CHECK (warm.lr_q15 (100u) == 16384);

    /* A start above the base inverts the ramp, and the truncating division
       rounds the negative increments toward zero:
       4000 + (100 - 3000) / 100 = 4000 - 29 = 3971. */
    warm._M_start_q15 = 4000;
    warm._M_base_q15 = 1000;
    CHECK (warm.lr_q15 (0u) == 4000);
    CHECK (warm.lr_q15 (1u) == 3971);
    CHECK (warm.lr_q15 (50u) == 2501);
    CHECK (warm.lr_q15 (99u) == 1031);
    CHECK (warm.lr_q15 (100u) == 1000);
    for (uint32_t s = 0u; s < 99u; ++s)
        CHECK (warm.lr_q15 (s + 1u) <= warm.lr_q15 (s));

    /* A negative start is floored to 0 for the first steps of the ramp:
       step 5 is -1000 + (17384 * 5 + 50) / 100 = -1000 + 874 = -126 -> 0,
       step 10 is -1000 + 1738 = 738. */
    warm._M_base_q15 = 16384;
    warm._M_start_q15 = -1000;
    CHECK (warm.lr_q15 (0u) == 0);
    CHECK (warm.lr_q15 (5u) == 0);
    CHECK (warm.lr_q15 (10u) == 738);
    CHECK (warm.lr_q15 (20u) == 2477);

    /* No warmup at all means the guard skips the ramp entirely. */
    warm._M_start_q15 = 0;
    warm._M_warmup_steps = 0u;
    CHECK (warm.lr_q15 (0u) == 16384);
    CHECK (warm.lr_q15 (5u) == 16384);

    /* An out-of-rail base clamps only after the ramp arithmetic:
       (40000 * 50 + 50) / 100 = 20000, then the rail from step 99 on. */
    warm._M_base_q15 = 40000;
    warm._M_warmup_steps = 100u;
    CHECK (warm.lr_q15 (50u) == 20000);
    CHECK (warm.lr_q15 (99u) == _hi);
    CHECK (warm.lr_q15 (100u) == _hi);

    warm._M_base_q15 = 16384;
    warm._M_start_q15 = 0;
    warm._M_total_steps = 1000u;
    warm._M_decay_after = 1u;
    /* After warmup the angle is (pi/2 * (step - warmup)) / (total - warmup)
       and the shape is base * (1 + cos) / 2 with one >> 16. */
    CHECK (warm.lr_q15 (99u) == 16220);   /* still inside the ramp */
    CHECK (warm.lr_q15 (100u) == 16383);  /* angle 0 -> cos rail   */
    CHECK (warm.lr_q15 (150u) == 16352);
    CHECK (warm.lr_q15 (200u) == 16259);
    CHECK (warm.lr_q15 (300u) == 15889);
    CHECK (warm.lr_q15 (550u) == 13984);
    CHECK (warm.lr_q15 (800u) == 10994);
    CHECK (warm.lr_q15 (999u) == 8206);
    /* step == span is the documented zero, and it stays zero. */
    CHECK (warm.lr_q15 (1000u) == 0);
    CHECK (warm.lr_q15 (5000u) == 0);
    CHECK (warm.lr_q15 (0xFFFFFFFFu) == 0);
    for (uint32_t s = 100u; s < 999u; ++s)
        CHECK (warm.lr_q15 (s + 1u) <= warm.lr_q15 (s));

    /* total <= warmup collapses the span to 1: the decayed value at the
       warmup step is the cos rail shape and one step later is zero. */
    warm._M_total_steps = 50u;
    CHECK (warm.lr_q15 (100u) == 16383);
    CHECK (warm.lr_q15 (101u) == 0);

    /* All rails: nothing a schedule returns can leave [0, 32767]. */
    LinearWarmupSchedule probe;
    for (const int32_t base : {0, 1, -5, 16384, _hi, 40000})
        for (const int32_t start : {-32768, -1, 0, 4096, _hi})
            for (const uint8_t decay : {0u, 1u})
                {
                    probe._M_base_q15 = base;
                    probe._M_start_q15 = start;
                    probe._M_decay_after = decay;
                    probe._M_warmup_steps = 7u;
                    probe._M_total_steps = 23u;
                    for (uint32_t s = 0u; s <= 30u; ++s)
                        {
                            const int32_t lr = probe.lr_q15 (s);
                            CHECK (lr >= 0);
                            CHECK (lr <= _hi);
                        }
                }
}

TEST_CASE ("cosine anneal shape, start rail and end rail",
           "[mllee][train][sched]")
{
    STATIC_REQUIRE (sizeof (CosineSchedule) == 12u);

    CosineSchedule cosine;
    CHECK (cosine._M_base_q15 == 16384);
    CHECK (cosine._M_min_q15 == 0);
    CHECK (cosine._M_total_steps == 1000u);

    /* angle = (pi * step) / total, shape = (1 + cos) / 2, lr = min +
       ((base - min) * shape) >> 15.  step 0: cos = 32767, shape = 32767,
       (16384 * 32767) >> 15 = 16383 -- one ulp under the base. */
    const uint32_t steps[] = { 0u,   1u,   2u,   250u, 500u,  749u,
                               750u, 999u, 1000u, 1001u, 9999u };
    const int32_t want[] = { 16383, 16382, 16382, 13984, 8192, 2417,
                             2399,  1,     1,     1,     1 };
    for (uint32_t i = 0; i < 11u; ++i)
        CHECK (cosine.lr_q15 (steps[i]) == want[i]);
    CHECK (cosine.lr_q15 (0xFFFFFFFFu) == 1);

    /* step 500 is exactly pi / 2 where the fold is 0, so the shape is the
       half-way 16384 and the lr is exactly half the base. */
    CHECK (cosine.lr_q15 (500u) == 8192);

    /* The end rail is min + 1, never min: cos(pi) evaluates to -32762, not
       -32768, so shape = (32768 - 32762) >> 1 = 3 and 16384 * 3 >> 15 = 1. */
    CHECK (cosine.lr_q15 (999u) == 1);
    CHECK (cosine.lr_q15 (1000u) == 1);

    cosine._M_min_q15 = 1024;
    CHECK (cosine.lr_q15 (0u) == 16383);
    CHECK (cosine.lr_q15 (500u) == 8704);   /* 1024 + (15360 * 16384) >> 15 */
    CHECK (cosine.lr_q15 (1000u) == 1025);  /* the floor is min + 1         */

    /* Degenerate configurations. */
    cosine._M_total_steps = 0u;
    CHECK (cosine.lr_q15 (0u) == 16384);    /* early return, no shape loss  */
    CHECK (cosine.lr_q15 (777u) == 16384);
    cosine._M_base_q15 = _hi;
    cosine._M_min_q15 = _hi;
    cosine._M_total_steps = 1000u;
    for (const uint32_t s : {0u, 500u, 1000u, 4000u})
        CHECK (cosine.lr_q15 (s) == _hi);

    /* A negative span floors down through the arithmetic shift, so the
       "minimum" of an inverted configuration is reached as min - 1. */
    cosine._M_base_q15 = 1000;
    cosine._M_min_q15 = 4000;
    cosine._M_total_steps = 100u;
    CHECK (cosine.lr_q15 (0u) == 1000);
    CHECK (cosine.lr_q15 (50u) == 2500);
    CHECK (cosine.lr_q15 (100u) == 3999);
    CHECK (cosine.lr_q15 (120u) == 3999);

    /* Short cycles are exactly the quarter/half cosine grid: total 3 gives
       16383, 12288, 4096, then the 1 rail. */
    cosine._M_base_q15 = 16384;
    cosine._M_min_q15 = 0;
    cosine._M_total_steps = 3u;
    const int32_t short_want[] = { 16383, 12288, 4096, 1, 1 };
    for (uint32_t s = 0u; s < 5u; ++s)
        CHECK (cosine.lr_q15 (s) == short_want[s]);

    cosine._M_total_steps = 7u;
    const int32_t seven_want[] = { 16383, 15572, 13299, 10015, 6369, 3084, 811,
                                   1, 1 };
    for (uint32_t s = 0u; s < 9u; ++s)
        CHECK (cosine.lr_q15 (s) == seven_want[s]);

    cosine._M_total_steps = 10u;
    cosine._M_base_q15 = _hi;
    CHECK (cosine.lr_q15 (0u) == 32766);   /* (32767 * 32767) >> 15 */
    CHECK (cosine.lr_q15 (5u) == 16383);
    CHECK (cosine.lr_q15 (10u) == 2);      /* (32767 * 3) >> 15     */

    /* Monotonicity holds for the default 1000-step cycle across the clamped
       range, but it is a property of this grid, not of the fold: a 4096-step
       cycle rises at seven steps and a 200000-step cycle at 2194 of them,
       because cos_q15 is not monotone on [0, pi]. */
    CosineSchedule mono;
    int32_t previous = mono.lr_q15 (0u);
    for (uint32_t s = 1u; s <= 1300u; ++s)
        {
            const int32_t lr = mono.lr_q15 (s);
            CHECK (lr <= previous);
            CHECK (lr >= 0);
            previous = lr;
        }

    mono._M_total_steps = 4096u;
    uint32_t breaks = 0u;
    int first_up = -1;
    for (uint32_t s = 0u; s < 4095u; ++s)
        if (mono.lr_q15 (s + 1u) > mono.lr_q15 (s))
            {
                ++breaks;
                if (first_up < 0)
                    first_up = static_cast<int> (s);
            }
    CHECK (breaks == 7u);
    CHECK (first_up == 7);
    CHECK (mono.lr_q15 (7u) == 16382);
    CHECK (mono.lr_q15 (8u) == 16383);

    mono._M_total_steps = 200000u;
    breaks = 0u;
    first_up = -1;
    for (uint32_t s = 0u; s < 199999u; ++s)
        if (mono.lr_q15 (s + 1u) > mono.lr_q15 (s))
            {
                ++breaks;
                if (first_up < 0)
                    first_up = static_cast<int> (s);
            }
    CHECK (breaks == 2194u);
    CHECK (first_up == 29);
    CHECK (mono.lr_q15 (29u) == 16382);
    CHECK (mono.lr_q15 (30u) == 16383);
    CHECK (mono.lr_q15 (100000u) == 8192);   /* half cycle */
    CHECK (mono.lr_q15 (199999u) == 1);
}

TEST_CASE ("cosine restarts walk the geometric period",
           "[mllee][train][sched]")
{
    STATIC_REQUIRE (sizeof (CosineRestartsSchedule) == 16u);

    CosineRestartsSchedule restarts;
    CHECK (restarts._M_base_q15 == 16384);
    CHECK (restarts._M_first_period == 100u);
    CHECK (restarts._M_period_mult == 2u);

    /* Periods 100, 200, 400, ... so the cycle boundaries are 100, 300, 700,
       1500, 3100 and each boundary re-evaluates the inner cosine at step 0,
       which is the 32767 shape rail rather than the base. */
    const uint32_t bounds[] = { 0u, 100u, 300u, 700u, 1500u, 3100u, 6300u };
    for (const uint32_t s : bounds)
        CHECK (restarts.lr_q15 (s) == 16383);

    const uint32_t steps[] = { 50u, 99u,  150u,  249u,  250u,  450u,
                               1000u, 1300u, 2000u, 2349u, 2350u, 4000u };
    const int32_t want[] = { 8192, 5, 13984, 2491, 2399, 11327,
                             11327, 2399, 12743, 7405, 7389, 13389 };
    for (uint32_t i = 0; i < 12u; ++i)
        CHECK (restarts.lr_q15 (steps[i]) == want[i]);

    /* The last cycle inside 1000 is the 400-period at within 150, and the
       800-period at within 300 is the same cosine phase: 11327 both times. */
    CHECK (restarts.lr_q15 (450u) == restarts.lr_q15 (1000u));

    /* mult 0 is documented as 1, and 1 is a fixed period. */
    for (const uint32_t mult : {0u, 1u})
        {
            restarts._M_period_mult = mult;
            CHECK (restarts.lr_q15 (0u) == 16383);
            CHECK (restarts.lr_q15 (100u) == 16383);
            CHECK (restarts.lr_q15 (250u) == 8192);
        }

    /* A fixed period only walks 33 cycles, so beyond 3300 steps the leftover
       step is no longer reduced: at 3300 the guard leaves within 0 (a restart
       value), at 3299 it leaves within 99 (the tail of the cycle). */
    restarts._M_period_mult = 1u;
    CHECK (restarts.lr_q15 (3299u) == 5);
    CHECK (restarts.lr_q15 (3300u) == 16383);
    /* Past the guard the leftover step is no longer reduced, so the inner
       cosine sees step 1 of a 100 period -- a plain cycle value, not a clamp. */
    CHECK (restarts.lr_q15 (3301u) == 16378);
    CHECK (restarts.lr_q15 (3400u) == 1);       /* within 100 -> pi rail  */
    CHECK (restarts.lr_q15 (4000u) == 1);       /* within 700 -> pi rail  */

    /* A zero first period is the documented constant. */
    restarts._M_period_mult = 2u;
    restarts._M_first_period = 0u;
    for (const uint32_t s : {0u, 1u, 100u, 0xFFFFFFFFu})
        CHECK (restarts.lr_q15 (s) == 16384);

    /* The geometric walk must cover any uint32 step: mult 2 reaches a
       3355443200 period in 25 steps of the walk; mult 3 hits the uint32
       period cap first. */
    restarts._M_first_period = 100u;
    restarts._M_period_mult = 2u;
    CHECK (restarts.lr_q15 (0xFFFFFFFFu) == 13413);
    restarts._M_period_mult = 3u;
    CHECK (restarts.lr_q15 (0xFFFFFFFFu) == 8221);
    restarts._M_first_period = 3000000000u;
    restarts._M_period_mult = 2u;
    CHECK (restarts.lr_q15 (0xFFFFFFFFu) == 12975);
    CHECK (restarts.lr_q15 (3000000000u) == 16383);

    /* min == base collapses the span, so every step is the same value. */
    CosineRestartsSchedule flat;
    flat._M_base_q15 = 16384;
    flat._M_min_q15 = 16384;
    for (const uint32_t s : {0u, 1u, 99u, 100u, 250u, 100000u, 0xFFFFFFFFu})
        CHECK (flat.lr_q15 (s) == 16384);

    /* Rail sweep: no configuration of the restart walk leaves [0, 32767]. */
    CosineRestartsSchedule probe;
    for (const int32_t base : {0, 1, 16384, _hi})
        for (const uint32_t mult : {0u, 1u, 2u, 5u})
            {
                probe._M_base_q15 = base;
                probe._M_min_q15 = 0;
                probe._M_first_period = 3u;
                probe._M_period_mult = mult;
                for (uint32_t s = 0u; s <= 60u; ++s)
                    {
                        const int32_t lr = probe.lr_q15 (s);
                        CHECK (lr >= 0);
                        CHECK (lr <= _hi);
                    }
            }
}

TEST_CASE ("one-cycle ramps linearly then anneals on a cosine",
           "[mllee][train][sched]")
{
    STATIC_REQUIRE (sizeof (OneCycleSchedule) == 16u);
    STATIC_REQUIRE (OneCycleSchedule::_S_up_fraction_q15 == 9830);

    OneCycleSchedule cycle;
    CHECK (cycle._M_base_q15 == 1024);
    CHECK (cycle._M_peak_q15 == 16384);
    CHECK (cycle._M_min_q15 == 512);
    CHECK (cycle._M_total_steps == 1000u);

    /* up_steps = (1000 * 9830) >> 15 = 9830000 >> 15 = 299: the ramp owns
       steps 0..298 and the cosine owns 299..999. */
    const uint32_t steps[] = { 0u,   1u,    2u,    149u,  297u, 298u,
                               299u, 300u,  301u,  600u,  998u, 999u,
                               1000u, 5000u };
    const int32_t want[] = { 1024, 1075, 1127, 8678, 16281, 16333,
                             16383, 16382, 16382, 10194, 513, 512,
                             512, 512 };
    for (uint32_t i = 0; i < 14u; ++i)
        CHECK (cycle.lr_q15 (steps[i]) == want[i]);

    /* step 1 of the ramp: 1024 + (15360 * 1 + 299 / 2) / 299 = 1024 + 51. */
    CHECK (cycle.lr_q15 (1u) == 1075);

    /* The ramp never reaches the peak (16333 at 298) and the cosine phase
       starts a full 50 units higher at 16383, so the two halves of the
       documented "base -> peak -> min" cycle are discontinuous. */
    CHECK (cycle.lr_q15 (298u) == 16333);
    CHECK (cycle.lr_q15 (299u) == 16383);

    /* The anneal lands exactly on min for the default cycle: the final angle
       is one ulp short of pi, cos = -32765, shape = 1, span * 1 >> 15 = 0. */
    CHECK (cycle.lr_q15 (999u) == 512);
    CHECK (cycle.lr_q15 (1000u) == 512);

    /* Phase monotonicity for the default configuration. */
    for (uint32_t s = 0u; s < 298u; ++s)
        CHECK (cycle.lr_q15 (s + 1u) >= cycle.lr_q15 (s));
    for (uint32_t s = 299u; s < 1000u; ++s)
        CHECK (cycle.lr_q15 (s + 1u) <= cycle.lr_q15 (s));

    /* Short totals: up_steps truncates to 0, so the whole cycle is cosine. */
    cycle._M_total_steps = 0u;
    CHECK (cycle.lr_q15 (0u) == 1024);   /* early return: the base, unshaped */
    CHECK (cycle.lr_q15 (9u) == 1024);

    cycle._M_total_steps = 1u;
    CHECK (cycle.lr_q15 (0u) == 16383);
    CHECK (cycle.lr_q15 (1u) == 16383);

    cycle._M_total_steps = 2u;
    CHECK (cycle.lr_q15 (0u) == 16383);
    CHECK (cycle.lr_q15 (1u) == 8448);   /* angle pi / 2, shape 16384 */
    CHECK (cycle.lr_q15 (2u) == 8448);

    cycle._M_total_steps = 3u;
    const int32_t three[] = { 16383, 12416, 4480, 4480 };
    for (uint32_t s = 0u; s < 4u; ++s)
        CHECK (cycle.lr_q15 (s) == three[s]);

    /* total 7 puts up_steps at 2, so the first two steps ramp:
       1024 + (15360 * 1 + 1) / 2 = 8704. */
    cycle._M_total_steps = 7u;
    const int32_t seven[] = { 1024, 8704, 16383, 14868, 10900, 5995, 2027,
                              2027 };
    for (uint32_t s = 0u; s < 8u; ++s)
        CHECK (cycle.lr_q15 (s) == seven[s]);

    cycle._M_total_steps = 1000u;

    /* An inverted base/peak inverts the ramp instead of refusing it, and the
       cosine half still anneals toward min. */
    OneCycleSchedule inverted;
    inverted._M_base_q15 = 16384;
    inverted._M_peak_q15 = 1024;
    inverted._M_min_q15 = 512;
    CHECK (inverted.lr_q15 (0u) == 16384);
    CHECK (inverted.lr_q15 (100u) == 11248);
    CHECK (inverted.lr_q15 (298u) == 1076);
    CHECK (inverted.lr_q15 (299u) == 1023);
    CHECK (inverted.lr_q15 (500u) == 926);
    CHECK (inverted.lr_q15 (999u) == 512);

    /* A long total keeps the anneal monotone only in the default 1000 grid:
       at 100000 steps the cosine phase rises 898 times. */
    OneCycleSchedule long_cycle;
    long_cycle._M_total_steps = 100000u;
    CHECK (long_cycle.lr_q15 (0u) == 1024);
    CHECK (long_cycle.lr_q15 (1u) == 1025);
    CHECK (long_cycle.lr_q15 (29997u) == 16383);
    CHECK (long_cycle.lr_q15 (29998u) == 16383);
    CHECK (long_cycle.lr_q15 (29999u) == 16382);
    CHECK (long_cycle.lr_q15 (50000u) == 13395);
    CHECK (long_cycle.lr_q15 (99999u) == 513);   /* min + 1, not min */
    uint32_t up_breaks = 0u;
    int first_up = -1;
    for (uint32_t s = 29998u; s < 99999u; ++s)
        if (long_cycle.lr_q15 (s + 1u) > long_cycle.lr_q15 (s))
            {
                ++up_breaks;
                if (first_up < 0)
                    first_up = static_cast<int> (s);
            }
    CHECK (up_breaks == 898u);
    CHECK (first_up == 30009);

    /* peak == base == min is a constant. */
    OneCycleSchedule flat;
    flat._M_base_q15 = 4096;
    flat._M_peak_q15 = 4096;
    flat._M_min_q15 = 4096;
    for (uint32_t s = 0u; s <= 100u; s += 7u)
        CHECK (flat.lr_q15 (s) == 4096);

    OneCycleSchedule probe;
    for (const int32_t base : {0, 1, 40000})
        for (const uint32_t total : {1u, 2u, 5u, 100u})
            {
                probe._M_base_q15 = base;
                probe._M_peak_q15 = base < 0 ? 0 : _hi;
                probe._M_min_q15 = 0;
                probe._M_total_steps = total;
                for (uint32_t s = 0u; s <= total + 5u; ++s)
                    {
                        const int32_t lr = probe.lr_q15 (s);
                        CHECK (lr >= 0);
                        CHECK (lr <= _hi);
                    }
            }
}

TEST_CASE ("plateau lr floors at min and the advance state machine counts "
           "stale observations",
           "[mllee][train][sched]")
{
    STATIC_REQUIRE (sizeof (PlateauConfig) == 16u);
    STATIC_REQUIRE (sizeof (PlateauState) == 12u);

    PlateauConfig config;
    CHECK (config._M_base_q15 == 16384);
    CHECK (config._M_factor_q15 == 3277);
    CHECK (config._M_min_q15 == 32);
    CHECK (config._M_patience == 5u);

    /* drops = bad_count / patience, so the lr is constant for the whole
       patience window and steps down the q15_pow ladder at each multiple. */
    const uint32_t bad_counts[] = { 0u,  4u,  5u,  9u,  10u, 14u,
                                    15u, 19u, 20u, 24u, 25u, 500u };
    const int32_t want[] = { 16384, 16384, 1639, 1639, 164, 164,
                             32,    32,    32,   32,   32,  32 };
    for (uint32_t i = 0; i < 12u; ++i)
        CHECK (plateau_lr_q15 (config, bad_counts[i], config._M_patience)
               == want[i]);

    /* A zero patience disables the division entirely. */
    for (const uint32_t bad : {0u, 7u, 1000u})
        CHECK (plateau_lr_q15 (config, bad, 0u) == 16384);

    /* The floor is a plain max, so a min above the base raises every value. */
    config._M_min_q15 = 20000;
    CHECK (plateau_lr_q15 (config, 0u, 5u) == 20000);
    CHECK (plateau_lr_q15 (config, 100u, 5u) == 20000);
    config._M_min_q15 = 32;

    /* A negative min cannot floor the collapsed ladder: the max test is
       "value < min", and 0 is not below -1. */
    config._M_min_q15 = -1;
    CHECK (plateau_lr_q15 (config, 100u, 5u) == 0);
    CHECK (plateau_lr_q15 (config, 0u, 5u) == 16384);
    config._M_min_q15 = 32;

    PlateauState state;
    CHECK (state._M_bad_count == 0u);
    CHECK (state._M_best_metric_q7 == _lo);   /* the "no observation" sentinel */
    CHECK (state._M_lr_q15 == 0);

    /* The first observation always improves because of the sentinel. */
    CHECK (plateau_advance (config, state, 500, 0) == 16384);
    CHECK (state._M_best_metric_q7 == 500);
    CHECK (state._M_bad_count == 0u);
    CHECK (state._M_lr_q15 == 16384);

    /* Ties are stale, not improvements: the comparison is strict. */
    CHECK (plateau_advance (config, state, 500, 0) == 16384);
    CHECK (state._M_bad_count == 1u);
    CHECK (plateau_advance (config, state, 501, 0) == 16384);
    CHECK (state._M_bad_count == 2u);
    CHECK (plateau_advance (config, state, 499, 0) == 16384);
    CHECK (state._M_bad_count == 0u);         /* 499 < 500: improvement */
    CHECK (state._M_best_metric_q7 == 499);

    /* Five stale observations trip the first drop. */
    for (uint32_t i = 0; i < 4u; ++i)
        {
            CHECK (plateau_advance (config, state, 600, 0) == 16384);
            CHECK (state._M_bad_count == i + 1u);
        }
    CHECK (plateau_advance (config, state, 600, 0) == 1639);
    CHECK (state._M_bad_count == 5u);
    for (uint32_t i = 0; i < 4u; ++i)
        CHECK (plateau_advance (config, state, 600, 0) == 1639);
    CHECK (state._M_bad_count == 9u);
    CHECK (plateau_advance (config, state, 600, 0) == 164);
    CHECK (state._M_bad_count == 10u);
    CHECK (plateau_advance (config, state, 600, 0) == 164);
    CHECK (state._M_bad_count == 11u);
    CHECK (state._M_lr_q15 == 164);           /* the state mirrors the return */

    /* The margin is subtracted from the best metric: an improvement needs
       metric < best - margin, so 493 beats 499 - 5 but 489 does not beat
       493 - 5 = 488. */
    CHECK (plateau_advance (config, state, 493, 5) == 16384);
    CHECK (state._M_best_metric_q7 == 493);
    CHECK (state._M_bad_count == 0u);
    CHECK (plateau_advance (config, state, 489, 5) == 16384);
    CHECK (state._M_bad_count == 1u);
    CHECK (state._M_best_metric_q7 == 493);
    CHECK (plateau_advance (config, state, 487, 5) == 16384);
    CHECK (state._M_best_metric_q7 == 487);
    CHECK (state._M_bad_count == 0u);

    /* A negative margin makes the test easier than the raw comparison:
       500 < 487 + 20 = 507 counts as an improvement. */
    CHECK (plateau_advance (config, state, 500, -20) == 16384);
    CHECK (state._M_best_metric_q7 == 500);
    CHECK (state._M_bad_count == 0u);

    /* The sentinel is also a legal metric, and then every observation looks
       like an improvement: bad_count never advances and the lr never drops. */
    PlateauState locked;
    for (uint32_t i = 0; i < 40u; ++i)
        {
            CHECK (plateau_advance (config, locked, _lo, 0) == 16384);
            CHECK (locked._M_bad_count == 0u);
            CHECK (locked._M_best_metric_q7 == _lo);
        }
}

TEST_CASE ("schedule_fn_for dispatches to the concrete evaluators",
           "[mllee][train][sched]")
{
    STATIC_REQUIRE (static_cast<uint32_t> (ScheduleKind::CONSTANT) == 0u);
    STATIC_REQUIRE (static_cast<uint32_t> (ScheduleKind::ONE_CYCLE) == 7u);
    STATIC_REQUIRE (static_cast<uint32_t> (ScheduleKind::COUNT) == 8u);

    CHECK (schedule_fn_for (ScheduleKind::COUNT) == nullptr);

    ConstantSchedule constant;
    StepSchedule step;
    MultiStepSchedule multi;
    ExponentialSchedule expo;
    LinearWarmupSchedule warm;
    CosineSchedule cosine;
    CosineRestartsSchedule restarts;
    OneCycleSchedule cycle;
    warm._M_decay_after = 1u;

    struct Entry
    {
        ScheduleKind _M_kind;
        void *_M_schedule;
        int32_t (*_M_direct) (const void *, uint32_t);
    };

    const Entry table[] = {
        { ScheduleKind::CONSTANT, &constant,
          [](const void *const s, const uint32_t step_in) {
              return static_cast<const ConstantSchedule *> (s)->lr_q15 (step_in);
          } },
        { ScheduleKind::STEP, &step,
          [](const void *const s, const uint32_t step_in) {
              return static_cast<const StepSchedule *> (s)->lr_q15 (step_in);
          } },
        { ScheduleKind::MULTI_STEP, &multi,
          [](const void *const s, const uint32_t step_in) {
              return static_cast<const MultiStepSchedule *> (s)->lr_q15 (step_in);
          } },
        { ScheduleKind::EXPONENTIAL, &expo,
          [](const void *const s, const uint32_t step_in) {
              return static_cast<const ExponentialSchedule *> (s)->lr_q15 (step_in);
          } },
        { ScheduleKind::WARMUP, &warm,
          [](const void *const s, const uint32_t step_in) {
              return static_cast<const LinearWarmupSchedule *> (s)->lr_q15 (step_in);
          } },
        { ScheduleKind::COSINE, &cosine,
          [](const void *const s, const uint32_t step_in) {
              return static_cast<const CosineSchedule *> (s)->lr_q15 (step_in);
          } },
        { ScheduleKind::COSINE_RESTARTS, &restarts,
          [](const void *const s, const uint32_t step_in) {
              return static_cast<const CosineRestartsSchedule *> (s)->lr_q15 (step_in);
          } },
        { ScheduleKind::ONE_CYCLE, &cycle,
          [](const void *const s, const uint32_t step_in) {
              return static_cast<const OneCycleSchedule *> (s)->lr_q15 (step_in);
          } },
    };

    const uint32_t steps[]
        = { 0u, 1u, 2u, 7u, 33u, 99u, 100u, 250u, 999u, 1000u, 5000u,
            100000u, 0xFFFFFFFFu };
    for (const Entry &entry : table)
        {
            const ScheduleFn fn = schedule_fn_for (entry._M_kind);
            REQUIRE (fn != nullptr);
            for (const uint32_t s : steps)
                CHECK (fn (entry._M_schedule, s) == entry._M_direct (entry._M_schedule, s));
        }
}
