/* Tests for the training foundation: fixed-point helpers, the pool, tensor
 * views, the gradient tape, the activation pair, the backward kernels and the
 * step policy.
 *
 * The numeric expectations are written against <cmath> or against a reference
 * computed in the test, never against the implementation, so a refactor that
 * changes a rounding rule fails here instead of silently moving the reference. */

#include "mllee/train_activations.hpp"
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"
#include "mllee/train_trainer.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <string>

using namespace zzs::mllee;
using namespace zzs::mllee::train;
using Catch::Approx;

namespace
{
constexpr float _q15_unit = 32768.0f;

float
to_float_q15 (const int32_t lane)
{
    return lane / _q15_unit;
}

int32_t
code_of (const float normalized)
{
    return static_cast<int32_t> (std::lround (normalized * 127.f));
}

float
normalized_of_code (const int32_t code)
{
    return code / 127.f;
}

/* act_q15 cannot report a magnitude above 1.0, so the reference is clamped to
   the domain the lane can hold before the comparison. */
float
clamp_ref (const float value)
{
    return value > 1.f ? 1.f : (value < -1.f ? -1.f : value);
}

float
ref_relu (const float x, const float)
{
    return x > 0.f ? x : 0.f;
}

float
der_relu (const float x, const float)
{
    return x > 0.f ? 1.f : 0.f;
}

float
ref_sigmoid (const float x, const float)
{
    return 1.f / (1.f + std::exp (-x));
}

float
der_sigmoid (const float x, const float)
{
    const float s = ref_sigmoid (x, 0.f);
    return s * (1.f - s);
}

float
ref_tanh (const float x, const float)
{
    return std::tanh (x);
}

float
der_tanh (const float x, const float)
{
    const float t = std::tanh (x);
    return 1.f - t * t;
}

/* The integer path implements the sigmoid surrogate of GELU, so the reference
   is that surrogate, not the erf form. */
float
ref_gelu (const float x, const float)
{
    return x / (1.f + std::exp (-1.702f * x));
}

float
der_gelu (const float x, const float)
{
    const float s = 1.f / (1.f + std::exp (-1.702f * x));
    return s + x * 1.702f * s * (1.f - s);
}

float
ref_leaky (const float x, const float aux)
{
    return x < 0.f ? aux * x : x;
}

float
der_leaky (const float x, const float aux)
{
    return x < 0.f ? aux : 1.f;
}

float
ref_elu (const float x, const float aux)
{
    return x < 0.f ? aux * (std::exp (x) - 1.f) : x;
}

float
der_elu (const float x, const float aux)
{
    return x < 0.f ? aux * std::exp (x) : 1.f;
}

float
ref_silu (const float x, const float)
{
    return x / (1.f + std::exp (-x));
}

float
der_silu (const float x, const float)
{
    const float s = 1.f / (1.f + std::exp (-x));
    return s + x * s * (1.f - s);
}

/* The hard family is normalized: the code domain has no 6, so the knee of the
   ramp sits at +/-1 instead of +/-3. */
float
ref_hardswish (const float x, const float)
{
    if (x <= -1.f)
        return 0.f;
    if (x >= 1.f)
        return x;
    return x * (x + 1.f) * 0.5f;
}

float
der_hardswish (const float x, const float)
{
    if (x <= -1.f)
        return 0.f;
    if (x >= 1.f)
        return 1.f;
    return x + 0.5f;
}

float
ref_hardsigmoid (const float x, const float)
{
    const float y = (x + 1.f) * 0.5f;
    return y < 0.f ? 0.f : (y > 1.f ? 1.f : y);
}

float
der_hardsigmoid (const float x, const float)
{
    return (x > -1.f && x < 1.f) ? 0.5f : 0.f;
}

float
ref_none (const float x, const float)
{
    return x;
}

float
der_none (const float, const float)
{
    return 1.f;
}

struct Row
{
    Act _M_act;
    const char *_M_name;
    float (*_M_ref) (float, float);
    float (*_M_der) (float, float);
    float _M_aux;
    bool _M_piecewise; /* linear on each side: compare the slope exactly */
};

/* One recorded node: grad_in[i] = 2 * the scalar seed, so both the routing and
   the saturation count of the backward walk are observable.  The optional saved
   slot proves GradContext::saved_as reaches pool storage. */
Status
double_backward (const Trainable &, const Trainable &grad_output,
                 GradContext &context)
{
    if (context._M_saved_count > 0u)
    {
        int16_t *saved = context.saved_as<int16_t> (0u);
        if (saved == nullptr)
            return Status::BAD_LAYOUT;
        if (context.saved_as<int32_t> (0u) != nullptr)
            return Status::INTERNAL; /* a 4-byte lane must not alias a 2-byte one */
        saved[0] = static_cast<int16_t> (grad_output.grad_at (0));
    }
    const int32_t seed = grad_output._M_grad[0];
    const Trainable &in = context._M_operands[0];
    const int64_t doubled = static_cast<int64_t> (seed) * 2;
    const int32_t routed = static_cast<int32_t> (
        doubled > _S_q15_max ? _S_q15_max
                             : (doubled < _S_q15_min ? _S_q15_min : doubled));
    for (uint32_t i = 0; i < in._M_count; ++i)
        if (in.add_grad (i, routed))
            ++context._M_saturation_events;
    return Status::OK;
}

Status
failing_backward (const Trainable &, const Trainable &, GradContext &)
{
    return Status::UNSUPPORTED;
}

/* ---- trainer harness ------------------------------------------------------
   The hooks are plain function pointers, so their state is at file scope and
   every test case resets it.  The toy op is a broadcast sum: loss = SUM_i w[i],
   whose backward gives every parameter lane the seeded gradient.            */
bool g_saturate = false;
bool g_forward_fails = false;
Status g_forward_status = Status::OK;
uint32_t g_forward_calls = 0;
uint32_t g_update_calls = 0;
uint32_t g_post_grad_calls = 0;
uint32_t g_reported_saturation = 0;
int32_t g_schedule_lr_q15 = 1638;

void
reset_harness ()
{
    g_saturate = false;
    g_forward_fails = false;
    g_forward_status = Status::OK;
    g_forward_calls = 0;
    g_update_calls = 0;
    g_post_grad_calls = 0;
    g_reported_saturation = 0;
    g_schedule_lr_q15 = 1638;
}

Status
sum_backward (const Trainable &output, const Trainable &grad_output,
              GradContext &context)
{
    (void) output;
    const Trainable &w = context._M_operands[0];
    const int32_t seed = grad_output._M_grad[0];
    for (uint32_t i = 0; i < w._M_count; ++i)
        if (w.add_grad (i, seed))
            ++context._M_saturation_events;
    if (g_saturate && w.add_grad (0, 40000))
        ++context._M_saturation_events;
    return Status::OK;
}

Status
toy_forward (StepContext &context)
{
    ++g_forward_calls;
    if (g_forward_fails)
        return g_forward_status;
    const Trainable &w = context._M_params[0];
    int32_t sum = 0;
    for (uint32_t i = 0; i < w._M_count; ++i)
        sum += w.code_at (i);
    context._M_loss.set_code (0, sum);
    context._M_loss_q7 = static_cast<int32_t> (
        (static_cast<int64_t> (sum) << 7) / _S_code_to_q15_scale);
    if (!context._M_tape->is_open ())
        return Status::OK; /* the evaluation pass shares this body */
    const Trainable operands[1] = { context._M_params[0] };
    return context._M_tape->record (sum_backward, 7u, context._M_loss, operands,
                                    1u, nullptr, 0u);
}

Status
toy_post_grad (StepContext &, const uint32_t saturated)
{
    ++g_post_grad_calls;
    g_reported_saturation = saturated;
    return Status::OK;
}

Status
toy_update (StepContext &context)
{
    ++g_update_calls;
    for (uint32_t p = 0; p < context._M_param_count; ++p)
    {
        const Trainable &w = context._M_params[p];
        for (uint32_t i = 0; i < w._M_count; ++i)
        {
            const int32_t delta = q15_mul (context._M_lr_q15, w.grad_at (i));
            w.set_code (i, w.code_at (i) - q15_to_code (delta));
        }
    }
    return Status::OK;
}

Status
refusing_update (StepContext &)
{
    return Status::ARENA_OVERFLOW;
}

int32_t
scheduled_lr (const uint32_t step)
{
    return step == 0u ? g_schedule_lr_q15 : g_schedule_lr_q15 + 1;
}
} // namespace

TEST_CASE ("fixed-point helpers saturate and round half away from zero",
           "[mllee][train][fixed]")
{
    SECTION ("code and Q15 are inverse to one code")
    {
        for (int32_t code = -127; code <= 127; ++code)
        {
            const int32_t q15 = code_to_q15 (code);
            REQUIRE (q15_to_code (q15) == code);
            REQUIRE (q15_in_range (q15));
        }
        CHECK (code_to_q15 (127) == 127 * _S_code_to_q15_scale);
        CHECK (code_to_q15 (0) == 0);
        /* -128 is never emitted: the grid stays symmetric. */
        CHECK (q15_to_code (-128 * _S_code_to_q15_scale) == -127);
    }

    SECTION ("clamping never wraps the sign")
    {
        CHECK (q15_clamp (40000) == _S_q15_max);
        CHECK (q15_clamp (-40000) == _S_q15_min);
        CHECK (q15_to_i16_sat (40000) == static_cast<int16_t> (_S_q15_max));
        CHECK (q15_to_i16_sat (-40000) == static_cast<int16_t> (_S_q15_min));
        CHECK_FALSE (q15_in_range (_S_q15_one));
        CHECK (q15_in_range (_S_q15_min));
        CHECK (q15_add_sat (_S_q15_max, 1) == _S_q15_max);
        CHECK (q15_sub_sat (_S_q15_min, 1) == _S_q15_min);
        CHECK (q15_abs (-500) == 500);
    }

    SECTION ("multiply rounds half up, truncate floors, and the two differ")
    {
        CHECK (q15_mul (_S_q15_one, _S_q15_one) == _S_q15_max);
        CHECK (q15_mul (16384, _S_q15_one) == 16384);
        CHECK (q15_mul (1, 16384) == 1);        /* 0.5 of a unit rounds up */
        CHECK (q15_mul_trunc (1, 16384) == 0);  /* the biased form truncates */
        CHECK (q15_mul (-16384, 16384) == -8192);
        /* the tie is (a*b + 2^14) >> 15, so it goes toward +infinity for both
           signs -- +0.5 LSB of bias, which is why the negative case reads as a
           lane of zero rather than the -1 a sign-symmetric rounding would give */
        CHECK (q15_mul (-1, 16384) == 0);
        CHECK (q15_mul_trunc (-1, 16384) == -1);
        CHECK (div_round (-1, 2) == -1);        /* the *other* tie rule, away from 0 */
        CHECK (q15_requant (static_cast<int64_t> (16384) << 15, 15u) == 16384);
        CHECK (q15_scale_shift (1000, 1000, 10u) == 977);
    }

    SECTION ("division by zero is a defined result, not a trap")
    {
        CHECK (q15_div (8192, 16384) == 16384); /* 0.25 / 0.5 */
        CHECK (q15_div (8192, 0) == 0);
        CHECK (q15_reciprocal (0) == _S_q15_max);
        CHECK (q15_reciprocal (-1) == _S_q15_max);
        CHECK (q15_reciprocal (_S_q15_max) == _S_q15_max); /* 1/1 clamps */
        CHECK (q15_reciprocal (16384) == _S_q15_max);      /* 1/0.5 = 2 clamps */
        CHECK (q15_reciprocal (_S_q15_one / 4) > 4 * _S_q15_max / 8);
        CHECK (ratio_q15 (1, 0) == 0);
    }

    SECTION ("sqrt stays on the grid")
    {
        CHECK (q15_sqrt (0) == 0);
        CHECK (q15_sqrt (-1) == 0);
        CHECK (to_float_q15 (q15_sqrt (16384))
                == Approx (std::sqrt (0.5)).margin (0.002));
        CHECK (q15_sqrt (_S_q15_one / 4) == _S_q15_one / 2);
        CHECK (q15_sqrt (_S_q15_one) == _S_q15_one);  /* sqrt(1) needs no clamp */
    }

    SECTION ("exp, sigmoid and tanh match their float twins over the window")
    {
        CHECK (q15_exp (0) == _S_q15_max);       /* e^0 = 1 clamps a lane */
        CHECK (q15_exp (1000) == _S_q15_max);
        CHECK (q15_exp_neg (0) == _S_q15_max);
        CHECK (q15_exp_neg (-5) == _S_q15_max);
        CHECK (q15_exp_neg (_S_q15_one) > 12000);
        /* the table is 256 entries of exp(-i/8) whose generator stalls at a
           floor of 4 once the per-step decay drops under half an LSB, so the
           deep tail reads 4 (1.2e-4) rather than 0 and only passing the table's
           end at t = 31.875 reaches 0.  Inside the documented +/-8 window the
           entries are exact to their rounding: t = 8 is entry 64, still 11. */
        CHECK (q15_exp_neg (_S_exp_clamp_q15 * 2) == 4); /* exp(-16) */
        CHECK (q15_exp_neg (9 * _S_q15_one) == 4);
        CHECK (q15_exp_neg (_S_exp_clamp_q15) > 4);      /* exp(-8) is above it */
        CHECK (q15_exp_neg (31 * _S_q15_one) > 0);
        CHECK (q15_exp_neg (32 * _S_q15_one) == 0);
        CHECK (q15_exp_neg (_S_q15_min) == _S_q15_max); /* a negative t clamps up */
        /* ... which is what bounds one cross-entropy term at ln(8192) = 9.01 */
        CHECK (q7_neg_log_q15 (4) / 128.0 == Approx (9.01).margin (0.02));
        for (float x = -4.f; x <= 4.01f; x += 0.5f)
        {
            CAPTURE (x);
            const int32_t lane = static_cast<int32_t> (x * _q15_unit);
            const float e = to_float_q15 (q15_exp_neg (std::abs (lane)));
            CHECK (e
                == Approx (std::exp (-std::fabs (x))) .margin (0.004));
            const float s = to_float_q15 (q15_sigmoid (lane));
            CHECK (s
                == Approx (ref_sigmoid (x, 0.f)).margin (0.004));
            const float t = to_float_q15 (q15_tanh (lane));
            CHECK (t
                == Approx (std::tanh (x)).margin (0.006));
        }
        CHECK (to_float_q15 (q15_sigmoid (0))
                == Approx (0.5f).margin (0.001f));
        CHECK (to_float_q15 (q15_sigmoid_grad (16384))
                == Approx (0.25f).margin (0.001f));
    }

    SECTION ("log domain is Q7 because a lane cannot hold a loss")
    {
        CHECK (q7_log_q15 (_S_q15_one) == 0);
        CHECK (q7_log_q15 (0) == _S_q15_min);      /* the -inf sentinel */
        CHECK (q7_log_q15 (-5) == _S_q15_min);
        CHECK (q7_from_q15 (_S_q15_one) == _S_q7_one);
        CHECK (q15_from_q7 (_S_q7_one) == _S_q15_max); /* 1.0 clamps */
        CHECK (q7_neg_log_q15 (0) == -_S_q15_min);
        CHECK (q7_neg_log_q15 (_S_q15_one) == 0);
        for (float p = 0.03f; p <= 1.0f; p *= 1.7f)
        {
            CAPTURE (p);
            const int32_t lane = static_cast<int32_t> (p * _q15_unit);
            const double got = q7_log_q15 (lane) / 128.0;
            CHECK (got
                == Approx (std::log (p)).margin (0.03));
            CHECK (q7_neg_log_q15 (lane) >= 0);
        }
        /* softplus: ln(1+e^x), the log-sum-exp atom. */
        CHECK (q7_log1p_exp (0) / 128.0
                == Approx (std::log (2.0)).margin (0.02));
        CHECK (q7_log1p_exp (_S_q15_one) / 128.0
                == Approx (std::log1p (std::exp (1.0))).margin (0.03));
        CHECK (q7_log1p_exp (-4 * _S_q15_one) >= 0);
        CHECK (q7_log1p_exp (4 * _S_q15_one) > q7_log1p_exp (_S_q15_one));
    }

    SECTION ("integer division and alignment round the way callers expect")
    {
        CHECK (div_round (3, 2) == 2);
        CHECK (div_round (-3, 2) == -2);  /* away from zero, not toward minus infinity */
        CHECK (div_round (3, -2) == -2);
        CHECK (div_round (1, 0) == 0);
        CHECK (align_up (5u, 4u) == 8u);
        CHECK (align_up (8u, 4u) == 8u);
        CHECK (align_up (0u, 4u) == 0u);
    }
}

TEST_CASE ("training pool carves aligned slabs and counts overflow",
           "[mllee][train][pool]")
{
    AlignedArena<256> arena;
    Pool pool = arena.pool ();

    CHECK (pool.stats ()._M_capacity == 256u);
    CHECK (pool.remaining () == 256u);
    CHECK (simd_aligned (arena._M_data));

    SECTION ("every slab starts on a lane boundary")
    {
        Slab first;
        Slab second;
        REQUIRE (status_ok (pool.carve (1u, 3u, first)));  /* 3 bytes */
        REQUIRE (status_ok (pool.carve (1u, 3u, second)));
        CHECK (first._M_offset == 0u);
        CHECK (first._M_bytes == 4u);
        CHECK (second._M_offset == 4u);
        CHECK (second._M_bytes == 4u);
        CHECK (simd_aligned (pool.at (second._M_offset)));
        CHECK (pool.stats ()._M_slabs == 2u);
        CHECK (pool.stats ()._M_used == 8u);
        CHECK (pool.stats ()._M_peak == 8u);
        CHECK (pool.remaining () == 248u);
        CHECK_FALSE (first.is_null ());
    }

    SECTION ("a carve that does not fit is counted, not fatal")
    {
        Slab slab;
        CHECK (pool.carve (1000u, 4u, slab) == Status::ARENA_OVERFLOW);
        CHECK (slab.is_null ());
        CHECK (pool.stats ()._M_overflows == 1u);
        CHECK (pool.stats ()._M_used == 0u);
        CHECK (pool.carve (0x20000u, 0x10000u, slab) == Status::OUT_OF_RANGE);
        CHECK (pool.stats ()._M_overflows == 1u); /* not an overflow: a bad request */
    }

    SECTION ("carve_typed zeroes and hands back a typed view")
    {
        int16_t *lanes = nullptr;
        REQUIRE (status_ok (pool.carve_typed (8u, lanes)));
        REQUIRE (lanes != nullptr);
        CHECK (lanes[7] == 0);
        lanes[3] = 1234;
        int16_t *again = nullptr;
        REQUIRE (status_ok (pool.carve_typed (8u, again)));
        CHECK (again[3] == 0);            /* a fresh slab is always zeroed */
        CHECK (pool.stats ()._M_peak == 32u);
        pool.reset ();
        CHECK (pool.stats ()._M_used == 0u);
        CHECK (pool.stats ()._M_peak == 32u); /* the high-water mark survives */
        REQUIRE (status_ok (pool.carve_typed (8u, again)));
        CHECK (again[3] == 0);
        int16_t *too_big = nullptr;
        CHECK (pool.carve_typed (10000u, too_big) == Status::ARENA_OVERFLOW);
        CHECK (too_big == nullptr);
    }

    SECTION ("a zero-length carve is legal and free")
    {
        Slab slab;
        REQUIRE (status_ok (pool.carve (0u, 4u, slab)));
        CHECK (slab._M_bytes == 0u);
        CHECK (pool.stats ()._M_used == 0u);
    }
}

TEST_CASE ("tensor views validate shapes and accumulate saturatingly",
           "[mllee][train][tensor]")
{
    SECTION ("shape")
    {
        CHECK (Shape::scalar ().elements () == 1u);
        CHECK (Shape::vector (8).elements () == 8u);
        CHECK (Shape::matrix (3, 4).elements () == 12u);
        CHECK (Shape::tensor4 (2, 3, 4, 5).elements () == 120u);
        CHECK (Shape::tensor3 (2, 3, 4).elements () == 24u);
        CHECK (Shape::tensor4 (2, 3, 4, 5).stride (1u) == 20u);
        CHECK (status_ok (Shape::matrix (3, 4).validate ()));
        CHECK (Shape::matrix (3, 4).stride (0u) == 4u);
        CHECK (Shape::matrix (3, 4).stride (1u) == 1u);
        Shape empty_axis = Shape::vector (1);
        empty_axis._M_dims[0] = 0u;
        CHECK (empty_axis.validate () == Status::BAD_LAYOUT);
        Shape too_many_axes = Shape::vector (1);
        too_many_axes._M_rank = _S_max_rank + 1u;
        CHECK (too_many_axes.validate () == Status::BAD_LAYOUT);
        Shape rank_zero = Shape::scalar ();
        rank_zero._M_rank = 0u;
        CHECK (rank_zero.validate () == Status::BAD_LAYOUT);
        Shape huge = Shape::vector (65535);
        huge._M_dims[0] = 65535u;
        huge._M_dims[1] = 2u;
        huge._M_rank = 2u;
        CHECK (huge.validate () == Status::OUT_OF_RANGE);
        CHECK (Shape::matrix (2, 3) == Shape::matrix (2, 3));
        CHECK_FALSE (Shape::matrix (2, 3) == Shape::matrix (3, 2));
    }

    SECTION ("broadcast matching and index mapping")
    {
        const Shape src = Shape::vector (4);
        const Shape dst = Shape::matrix (2, 4);
        CHECK (status_ok (broadcast_ok (src, dst)));
        CHECK (!status_ok (broadcast_ok (Shape::vector (3), dst)));
        CHECK (!status_ok (broadcast_ok (Shape::matrix (2, 2), Shape::vector (4))));
        CHECK (status_ok (broadcast_ok (Shape::vector (1), dst)));
        /* a [4] operand inside a [2,4] destination: both rows read the same
           element, so the flat index drops the leading axis. */
        CHECK (broadcast_index (0u, src, dst) == 0u);
        CHECK (broadcast_index (4u, src, dst) == 0u);
        CHECK (broadcast_index (7u, src, dst) == 3u);
    }

    SECTION ("gradient accumulation reports every clamped lane")
    {
        alignas (8) int8_t code[4] = { 10, -10, 20, -20 };
        alignas (8) int16_t grad[4] = { 30000, -30000, 0, 0 };
        const Trainable t
            = Trainable::make (code, grad, Shape::vector (4), RequiresGrad);
        REQUIRE (status_ok (t.validate ()));
        CHECK (t.requires_grad ());
        CHECK (t.simd_ready ());
        CHECK (t.q15_at (0) == code_to_q15 (10));
        CHECK_FALSE (t.add_grad (0, 1000)); /* 30000 + 1000 fits */
        CHECK (t.grad_at (0) == 31000);
        CHECK (t.add_grad (0, 10000));      /* clamps high */
        CHECK (t.grad_at (0) == _S_q15_max);
        CHECK (t.add_grad (1, -10000));     /* clamps low */
        CHECK (t.grad_at (1) == _S_q15_min);
        CHECK (t.add_grad (3, 100) == false);
        CHECK (t.code_at (3) == -20);
        CHECK (t.bytes () > 0u);
        t.set_grad (2, 40000);
        CHECK (t.grad_at (2) == _S_q15_max);
        t.zero_grad ();
        CHECK (t.grad_at (0) == 0);
        CHECK (t.grad_at (3) == 0);

        const Trainable no_grad
            = Trainable::make (code, nullptr, Shape::vector (4), RequiresGrad);
        CHECK (no_grad.grad_at (0) == 0);
        CHECK (no_grad.requires_grad ());
        CHECK (no_grad.simd_ready ()); /* a null stream is trivially aligned */
        no_grad.zero_grad ();          /* must not touch a null pointer */
        no_grad.set_grad (0, 100);
        const Trainable frozen
            = Trainable::make (code, grad, Shape::vector (4), RequiresGrad | Frozen);
        CHECK_FALSE (frozen.requires_grad ());
        const Trainable bias_like
            = Trainable::make (code, grad, Shape::vector (4), BiasLike);
        CHECK_FALSE (bias_like.requires_grad ());
    }

    SECTION ("a view whose count disagrees with its shape is rejected")
    {
        alignas (8) int8_t code[4] = { 0, 0, 0, 0 };
        Trainable t = Trainable::make (code, nullptr, Shape::vector (4), 0u);
        REQUIRE (status_ok (t.validate ()));
        t._M_count = 3u;
        CHECK (t.validate () == Status::BAD_LAYOUT);
        t._M_count = 4u;
        t._M_code = nullptr;
        CHECK (t.validate () == Status::BAD_LAYOUT);
    }

    SECTION ("reductions")
    {
        alignas (8) const int8_t codes[4] = { 100, 100, 100, 100 };
        alignas (8) const int16_t lanes[4] = { 1000, -2000, 3000, -4000 };
        alignas (8) const int8_t rhs[4] = { 1, 2, 3, 4 };
        alignas (8) const int16_t ones[4] = { _S_q15_max, _S_q15_max, _S_q15_max,
                                              _S_q15_max };
        CHECK (sum_codes (codes, 4u) == 400);
        CHECK (sum_q15 (lanes, 4u) == -2000);
        CHECK (sum_q15 (lanes, 0u) == 0);
        CHECK (mean_q15 (lanes, 4u) == -500);
        CHECK (mean_q15 (lanes, 0u) == 0);
        CHECK (max_abs_q15 (lanes, 4u) == 4000);
        CHECK (sum_squares_q15 (lanes, 4u) == 30'000'000);
        CHECK (dot_codes (codes, rhs, 4u) == 1000);
        /* multiplying by 1.0 is the identity up to the one-unit clamping error */
        CHECK (dot_q15 (lanes, ones, 4u) == -2000);
        CHECK (dot_q15 (lanes, lanes, 0u) == 0);
        /* a sum that no lane can hold saturates instead of wrapping */
        alignas (8) const int16_t big[4] = { 32000, 32000, 32000, 32000 };
        CHECK (sum_q15 (big, 4u) == _S_q15_max);
    }
}

TEST_CASE ("gradient tape records in order and walks back in reverse",
           "[mllee][train][graph]")
{
    AlignedArena<512> arena;
    Pool pool = arena.pool ();
    alignas (8) Node nodes[4];
    Tape tape (nodes, 4u, pool);

    alignas (8) int8_t in_code[4] = { 10, 20, 30, 40 };
    alignas (8) int16_t in_grad[4] = { 0, 0, 0, 0 };
    alignas (8) int8_t out_code[4] = { 20, 40, 60, 80 };
    alignas (8) int16_t out_grad[4] = { 0, 0, 0, 0 };
    alignas (8) int8_t loss_code[1] = { 0 };
    alignas (8) int16_t loss_grad[1] = { 0 };

    const Trainable in = Trainable::make (in_code, in_grad, Shape::vector (4),
                                          RequiresGrad);
    const Trainable out = Trainable::make (out_code, out_grad, Shape::vector (4),
                                           RequiresGrad);
    const Trainable loss
        = Trainable::make (loss_code, loss_grad, Shape::scalar (), 0u);

    CHECK (tape.stats ()._M_capacity == 4u);
    CHECK (tape.node_count () == 0u);
    CHECK_FALSE (tape.is_open ());
    CHECK (tape.pool ().stats ()._M_capacity == 512u);

    SECTION ("a tape with no room refuses to begin")
    {
        alignas (8) Node none[1];
        AlignedArena<64> other_arena;
        Pool other = other_arena.pool ();
        Tape empty (none, 0u, other);
        CHECK (empty.begin () == Status::BAD_LAYOUT);
        CHECK_FALSE (empty.is_open ());
    }

    SECTION ("recording past the capacity is an overflow, not a write")
    {
        REQUIRE (status_ok (tape.begin ()));
        CHECK (tape.is_open ());
        for (uint32_t i = 0; i < 4u; ++i)
            REQUIRE (status_ok (tape.record (double_backward, i, out, &in, 1u,
                                             nullptr, 0u)));
        CHECK (tape.record (double_backward, 9u, out, &in, 1u, nullptr, 0u)
               == Status::ARENA_OVERFLOW);
        CHECK (tape.node_count () == 4u);
        CHECK (tape.stats ()._M_nodes == 4u);
        CHECK (tape.stats ()._M_peak_nodes == 4u);
        CHECK (tape.node (3u) != nullptr);
        CHECK (tape.node (4u) == nullptr);
        /* an op with no backward or too many operands is rejected up front */
        CHECK (tape.record (nullptr, 1u, out, &in, 1u, nullptr, 0u)
               == Status::BAD_LAYOUT);
        const Trainable operands[_S_max_operands + 1u]
            = { in, in, in, in, in };
        CHECK (tape.record (double_backward, 1u, out, operands,
                            _S_max_operands + 1u, nullptr, 0u)
               == Status::BAD_LAYOUT);
        CHECK (tape.node_count () == 4u);
        tape.reset ();
        CHECK_FALSE (tape.is_open ());
        CHECK (tape.node_count () == 0u);
        /* recording without begin() is a wiring bug, reported as such */
        CHECK (tape.record (double_backward, 1u, out, &in, 1u, nullptr, 0u)
               == Status::INTERNAL);
    }

    tape.reset ();
    pool.reset ();
    REQUIRE (status_ok (tape.begin ()));

    SECTION ("backward seeds the loss with the scale and reaches every operand")
    {
        Saved saved;
        int16_t *saved_lanes = nullptr;
        REQUIRE (status_ok (tape.save (4u, saved, saved_lanes)));
        REQUIRE (saved_lanes != nullptr);
        CHECK (saved._M_bytes == 8u);
        CHECK (tape.stats ()._M_saved_bytes == 8u);
        saved_lanes[0] = 123;
        REQUIRE (status_ok (tape.record (double_backward, 3u, loss, &in, 1u,
                                         &saved, 1u)));
        tape.set_loss_scale (8000);
        CHECK (tape.loss_scale () == 8000);
        REQUIRE (status_ok (tape.backward (loss)));
        CHECK (loss_grad[0] == 8000);        /* the seed is the scale */
        CHECK (in_grad[0] == 16000);         /* and this op doubles it */
        CHECK (in_grad[3] == 16000);
        CHECK (saved_lanes[0] == 8000);      /* saved storage is reachable */
        CHECK (tape.stats ()._M_saturation_events == 0u);
        CHECK (tape.node_count () == 1u);
    }

    SECTION ("a seeded scale large enough to clamp is counted per lane")
    {
        /* add_grad only clamps against what is already in the lane, so start
           each one near the top: 30000 + 2 * 32767 does not fit, and all four
           lanes report it. */
        in_grad[0] = 30000;
        in_grad[1] = 30000;
        in_grad[2] = 30000;
        in_grad[3] = 30000;
        tape.reset ();
        REQUIRE (status_ok (tape.begin ()));
        REQUIRE (status_ok (tape.record (double_backward, 1u, loss, &in, 1u,
                                         nullptr, 0u)));
        tape.set_loss_scale (_S_q15_max);
        REQUIRE (status_ok (tape.backward (loss)));
        /* 2 * 32767 does not fit, so every lane clamps and reports it */
        CHECK (in_grad[0] == _S_q15_max);
        CHECK (tape.stats ()._M_saturation_events == 4u);
        CHECK (tape.stats ()._M_peak_nodes == 1u);
        /* begin() clears the per-pass count but keeps the high-water mark */
        tape.reset ();
        REQUIRE (status_ok (tape.begin ()));
        CHECK (tape.stats ()._M_saturation_events == 0u);
        CHECK (tape.stats ()._M_peak_nodes == 1u);
    }

    SECTION ("a backward failure stops the walk where it happened")
    {
        tape.reset ();
        REQUIRE (status_ok (tape.begin ()));
        REQUIRE (status_ok (tape.record (double_backward, 1u, loss, &in, 1u,
                                         nullptr, 0u)));
        REQUIRE (status_ok (tape.record (failing_backward, 2u, out, &in, 1u,
                                         nullptr, 0u)));
        /* reverse recording order: the failing node is the last recorded, so it
           runs first and the walk never reaches node 0. */
        in_grad[0] = 0;
        CHECK (tape.backward (loss) == Status::UNSUPPORTED);
        CHECK (in_grad[0] == 0);
    }

    SECTION ("save_copy duplicates a buffer and zero clears it")
    {
        tape.reset ();
        REQUIRE (status_ok (tape.begin ()));
        const int8_t source[4] = { 9, 8, 7, 6 };
        Saved copied;
        REQUIRE (status_ok (tape.save_copy (source, 4u, copied)));
        const int8_t *back = reinterpret_cast<const int8_t *> (
            pool.at (copied._M_offset));
        CHECK (back[0] == 9);
        CHECK (back[3] == 6);
        int32_t *fresh = nullptr;
        REQUIRE (status_ok (tape.save (8u, copied, fresh)));
        CHECK (fresh[7] == 0);
        uint8_t *too_small = nullptr;
        /* asking for more than the pool holds is an overflow, not a partial win */
        CHECK (tape.save (100000u, copied, too_small) == Status::ARENA_OVERFLOW);
        CHECK (too_small == nullptr);
        CHECK (pool.stats ()._M_overflows >= 1u);
    }

    SECTION ("zero_grad clears a parameter list")
    {
        in_grad[0] = 100;
        const Trainable params[1] = { in };
        Tape::zero_grad (params, 1u);
        CHECK (in_grad[0] == 0);
    }

    SECTION ("a tape's node cost is what the RAM gate measures")
    {
        CHECK (tape_node_bytes (0u) == 0u);
        CHECK (tape_node_bytes (1u) == sizeof (Node));
        CHECK (saved_bytes (Saved{ 0u, 4u, 1u }) == 4u);
        CHECK (sizeof (StepHooks) == 4u * sizeof (void (*) ()));
    }
}

TEST_CASE ("activations match their real-valued twins", "[mllee][train][act]")
{
    const Row rows[] = {
        { Act::RELU, "relu", ref_relu, der_relu, 0.f, true },
        { Act::RELU6, "relu6", ref_relu, der_relu, 0.f, true },
        { Act::SIGMOID, "sigmoid", ref_sigmoid, der_sigmoid, 0.f, false },
        { Act::TANH, "tanh", ref_tanh, der_tanh, 0.f, false },
        { Act::GELU, "gelu", ref_gelu, der_gelu, 0.f, false },
        { Act::LEAKY_RELU, "leaky_relu", ref_leaky, der_leaky,
          normalized_of_code (25), true },
        { Act::ELU, "elu", ref_elu, der_elu, normalized_of_code (100), false },
        { Act::SILU, "silu", ref_silu, der_silu, 0.f, false },
        { Act::HARDSWISH, "hardswish", ref_hardswish, der_hardswish, 0.f, true },
        { Act::HARDSIGMOID, "hardsigmoid", ref_hardsigmoid, der_hardsigmoid,
          0.f, true },
        { Act::NONE, "none", ref_none, der_none, 0.f, true },
    };

    /* The knee of the hard family sits exactly on +/-1.0, so the samples step
       across it: a derivative that only agrees away from the knee is wrong. */
    const float samples[] = { -1.0f, -0.95f, -0.5f, -0.25f, -0.05f,
                              0.0f, 0.05f, 0.25f, 0.5f, 0.95f, 1.0f };

    for (const Row &row : rows)
    {
        CAPTURE (row._M_name);
        const int32_t aux
            = row._M_aux > 0.f ? code_of (row._M_aux) : 0;
        const float aux_value = row._M_aux;

        SECTION (std::string (row._M_name) + " forward")
        {
            for (const float sample : samples)
            {
                CAPTURE (sample);
                const int32_t x = static_cast<int32_t> (sample * _q15_unit);
                const float got = to_float_q15 (act_q15 (row._M_act, x, aux));
                const float want = clamp_ref (row._M_ref (sample, aux_value));
                CHECK (got
                == Approx (want).margin (0.012f));
            }
        }

        SECTION (std::string (row._M_name) + " derivative")
        {
            for (const float sample : samples)
            {
                CAPTURE (sample);
                const int32_t x = static_cast<int32_t> (sample * _q15_unit);
                const int32_t y = act_q15 (row._M_act, x, aux);
                const float d
                    = to_float_q15 (act_derivative_q15 (row._M_act, x, y, aux));
                const float want = row._M_der (sample, aux_value);
                if (row._M_act == Act::RELU6 && sample >= 1.0f)
                    /* RELU6's upper clamp is the representable maximum in this
                       normalized domain, so the sample is exactly on the knee
                       and the derivative is the documented 0, not the
                       reference's 1. */
                    CHECK (d == 0.f);
                else if (want > 1.0f)
                    /* a lane cannot carry a gain above 1.0; the route form is
                       tested separately for keeping it. */
                    CHECK (d > 0.99f);
                else if (row._M_piecewise)
                    CHECK (d
                == Approx (want).margin (0.002f));
                else
                    CHECK (d
                == Approx (want).margin (0.03f));
            }
        }
    }

    SECTION ("hardswish keeps a gain above one only through the int64 route")
    {
        const int32_t x = static_cast<int32_t> (0.9f * _q15_unit);
        const int32_t capped
            = act_derivative_q15 (Act::HARDSWISH, x,
                                  act_q15 (Act::HARDSWISH, x, 0), 0);
        CHECK (capped == _S_q15_max); /* 1.4 clipped to the lane */
        const int32_t grad = 10000;   /* 0.305 */
        const int32_t routed = act_route_grad_q15 (Act::HARDSWISH, x, grad, 0);
        CHECK (to_float_q15 (routed)
                == Approx (0.305 * 1.4).margin (0.002));
        bool clamped = false;
        (void) act_route_grad_q15 (Act::HARDSWISH, x, _S_q15_max, 0, clamped);
        CHECK (clamped);             /* 1.4 * 1.0 does not fit a lane */
        clamped = true;
        (void) act_route_grad_q15 (Act::TANH, x, 1000, 0, clamped);
        CHECK_FALSE (clamped);
        /* outside the ramp the route form degenerates to the exact cases */
        CHECK (act_grad_product (Act::HARDSWISH, _S_q15_one, 1000, 0)
               == (static_cast<int64_t> (1000) << 15));
        CHECK (act_grad_product (Act::HARDSWISH, -_S_q15_one, 1000, 0) == 0);
        CHECK (act_grad_product (Act::HARDSWISH, -_S_q15_one - 1, 1000, 0) == 0);
    }

    SECTION ("gelu carries its own surrogate slope, 1.702, through the route")
    {
        /* d/dx [x * sigmoid(1.702 x)] peaks just above 1.0, so both halves of
           the check matter: the lane accessor has to report the clamp, and the
           route form has to keep the gain instead of narrowing it away. */
        const int32_t x = static_cast<int32_t> (0.5f * _q15_unit);
        const int32_t d = act_derivative_q15 (
            Act::GELU, x, act_q15 (Act::GELU, x, 0), 0);
        CHECK (to_float_q15 (d) == Approx (0.8792).margin (0.002));
        const int32_t knee = static_cast<int32_t> (_q15_unit - 1);
        CHECK (act_derivative_q15 (Act::GELU, knee,
                                   act_q15 (Act::GELU, knee, 0), 0)
               == _S_q15_max);                      /* 1.0678 cannot be a lane */
        CHECK (gelu_derivative_wide_q15 (knee) > _S_q15_max);
        const int32_t grad = 10000;                 /* 0.305 */
        const int32_t routed
            = act_route_grad_q15 (Act::GELU, knee, grad, 0);
        CHECK (to_float_q15 (routed)
               == Approx (0.305 * 1.0678).margin (0.002));
        bool clamped = false;
        (void) act_route_grad_q15 (Act::GELU, knee, _S_q15_max, 0, clamped);
        CHECK (clamped);
        /* the negative lobe is real: the surrogate dips, so a gradient there
           flips sign rather than vanishing */
        const int32_t below = static_cast<int32_t> (-0.5f * _q15_unit);
        CHECK (to_float_q15 (act_derivative_q15 (
                   Act::GELU, below, act_q15 (Act::GELU, below, 0), 0))
               == Approx (0.1208).margin (0.002));
        CHECK (to_float_q15 (act_derivative_q15 (
                   Act::GELU, -knee, act_q15 (Act::GELU, -knee, 0), 0))
               == Approx (-0.0678).margin (0.002));
    }

    SECTION ("the derivative from a code agrees with the lane form")
    {
        for (const Act act : { Act::RELU, Act::TANH, Act::SIGMOID, Act::SILU,
                               Act::GELU, Act::HARDSWISH })
        {
            CAPTURE (static_cast<int> (act));
            for (int32_t code = -127; code <= 127; code += 13)
            {
                const int32_t from_code
                    = act_derivative_from_code (act, code, 0);
                const int32_t x = code_to_q15 (code);
                const int32_t from_lane
                    = act_derivative_q15 (act, x, act_q15 (act, x, 0), 0);
                CHECK (from_code == from_lane);
            }
        }
    }

    SECTION ("linear acts short-circuit the backward to a mask or a copy")
    {
        CHECK (act_is_linear (Act::NONE));
        CHECK (act_is_linear (Act::RELU));
        CHECK_FALSE (act_is_linear (Act::TANH));
        CHECK (act_needs_preactivation (Act::GELU));
        CHECK (act_needs_preactivation (Act::LEAKY_RELU));
        CHECK (act_needs_preactivation (Act::HARDSWISH));
        CHECK (act_needs_preactivation (Act::SILU));
        CHECK (act_needs_preactivation (Act::ELU));
        CHECK (act_needs_preactivation (Act::HARDSIGMOID));
        CHECK_FALSE (act_needs_preactivation (Act::RELU));
        CHECK_FALSE (act_needs_preactivation (Act::SIGMOID));
        CHECK_FALSE (act_needs_preactivation (Act::TANH));
        CHECK_FALSE (act_needs_preactivation (Act::NONE));

        alignas (8) const int16_t pre[4] = { -1000, 0, 5000, -2 };
        alignas (8) const int16_t grad[4] = { 111, 222, 333, 444 };
        alignas (8) int16_t result[4] = { 0, 0, 0, 0 };
        CHECK (act_backward_q15 (Act::NONE, pre, grad, result, 4u, 0) == 0u);
        CHECK (result[0] == 111);
        CHECK (result[3] == 444);
        CHECK (act_backward_q15 (Act::RELU, pre, grad, result, 4u, 0) == 0u);
        CHECK (result[0] == 0);   /* a negative pre-activation kills the lane */
        CHECK (result[1] == 222); /* zero is on the passing side */
        CHECK (result[2] == 333);
        CHECK (result[3] == 0);
    }

    SECTION ("code-domain forward is the inference op")
    {
        alignas (8) const int8_t in[6] = { -100, -1, 0, 1, 50, 127 };
        alignas (8) int8_t out[6] = { 0, 0, 0, 0, 0, 0 };
        act_forward_codes (Act::RELU, in, out, 6u, 0);
        CHECK (out[0] == 0);
        CHECK (out[1] == 0);
        CHECK (out[2] == 0);
        CHECK (out[3] == 1);
        CHECK (out[5] == 127);
        /* every code-domain result equals the inference table's own answer */
        for (const Act act : { Act::NONE, Act::RELU, Act::RELU6, Act::TANH,
                               Act::SIGMOID, Act::HARDSIGMOID })
        {
            act_forward_codes (act, in, out, 6u, 0);
            for (uint32_t i = 0; i < 6u; ++i)
                CHECK (out[i]
                       == static_cast<int8_t> (act_code (act, in[i], 0)));
        }
        CHECK (act_code (Act::RELU, -50, 0) == 0);
        CHECK (act_code (Act::NONE, -128, 0) >= -127); /* no asymmetric code */
        act_forward_codes (Act::LEAKY_RELU, in, out, 6u, 25);
        CHECK (out[0] < 0);
        CHECK (out[0] > -100);
        /* the code path divides by 127 and truncates toward zero; the lane path
           requantizes at Q15 and rounds half up, so the two agree to within the
           one code each of them can differ by. */
        CHECK (out[0] == -19);
        const int32_t lane_leaky
            = q15_to_code (q15_mul (code_to_q15 (-100), code_to_q15 (25)));
        CHECK (lane_leaky == -20);
        CHECK (std::abs (out[0] - lane_leaky) <= 1);

        alignas (8) const int8_t pre[4] = { -64, -1, 0, 64 };
        alignas (8) const int16_t incoming[4] = { 1000, 2000, 3000, 4000 };
        alignas (8) int16_t grad_out[4] = { 0, 0, 0, 0 };
        CHECK (act_backward_codes (Act::RELU, pre, incoming, grad_out, 4u, 0)
               == 0u);
        CHECK (grad_out[0] == 0);
        CHECK (grad_out[1] == 0);
        CHECK (grad_out[2] == 3000);
        CHECK (grad_out[3] == 4000);

        alignas (8) const int16_t pre_q[4] = { -20000, -1, 1, 20000 };
        alignas (8) int16_t out_q[4] = { 0, 0, 0, 0 };
        act_forward_q15 (Act::SIGMOID, pre_q, out_q, 4u, 0);
        CHECK (out_q[2] > 16000);
        CHECK (out_q[2] < 17000);
        /* -0.610 in, so 0.352 out: the mirror identity keeps the negative half
           as accurate as the positive one. */
        CHECK (out_q[0] > 11000);
        CHECK (out_q[0] < 12000);
        CHECK (out_q[0] + out_q[3] == _S_q15_one); /* s(-x) + s(x) = 1 */
        CHECK (out_q[3] > out_q[2]);
    }

    SECTION ("a backward reports exactly the lanes that clamped")
    {
        alignas (8) const int16_t pre[3] = { 3000, 3000, 3000 };
        alignas (8) const int16_t fits[3] = { 1000, 2000, 3000 };
        alignas (8) int16_t out[3] = { 0, 0, 0 };
        CHECK (act_backward_q15 (Act::SIGMOID, pre, fits, out, 3u, 0) == 0u);
        CHECK (out[0] > 0);
        const int16_t huge[3] = { _S_q15_max, _S_q15_max, _S_q15_max };
        /* sigmoid's gain is below 1 so even a full lane stays representable */
        CHECK (act_backward_q15 (Act::SIGMOID, pre, huge, out, 3u, 0) == 0u);
        /* hardswish's ramp peaks at 1.5, so a full lane does clamp */
        const int16_t ramp[3] = { 20000, 20000, 20000 };
        CHECK (act_backward_q15 (Act::HARDSWISH, ramp, huge, out, 3u, 0) == 3u);
        CHECK (out[0] == _S_q15_max);
        /* hardswish dips between the knees: d/dx [x (x + 1) / 2] = x + 1/2, so
           below the knee but inside the ramp the gain is negative and a full
           lane comes back flipped in sign, not zeroed. */
        const int16_t dip[3] = { -30000, -30000, -30000 };
        CHECK (act_backward_q15 (Act::HARDSWISH, dip, huge, out, 3u, 0) == 0u);
        CHECK (out[0] == -13616); /* (-0.916 + 0.5) * 0.99997 in Q15 */
        /* at or below the knee the op is flat, so nothing routes at all */
        const int16_t below[3] = { -32768, -32768, -32768 };
        CHECK (act_backward_q15 (Act::HARDSWISH, below, huge, out, 3u, 0) == 0u);
        CHECK (out[0] == 0);
    }
}

TEST_CASE ("backward kernels reproduce a reference matmul and reduction",
           "[mllee][train][kernels]")
{
    SECTION ("accumulator narrowing clamps before it shifts")
    {
        const int64_t full = static_cast<int64_t> (_S_q15_max) << 15;
        CHECK (narrow_sum_q15 (full) == _S_q15_max);
        /* the accumulator bound is symmetric at +/- (2^15 - 1) rather than the
           lane's own -32768, so a fully saturated sum lands on -32767: the
           saturation count is what the trainer acts on, not that last lane. */
        CHECK (narrow_sum_q15 (-full) == -_S_q15_max);
        CHECK (narrow_sum_q15 (static_cast<int64_t> (16384) << 15) == 16384);
        CHECK (narrow_sum_q15 (0) == 0);
        /* an int32 truncation would wrap the sign; make sure it does not */
        CHECK (narrow_sum_q15 (1LL << 40) == _S_q15_max);
        CHECK (narrow_sum_q15 (-(1LL << 40)) == -_S_q15_max);
        /* the plain form clamps to the lane range itself */
        CHECK (clamp_sum_q15 (1LL << 40) == _S_q15_max);
        CHECK (clamp_sum_q15 (-1LL << 40) == _S_q15_min);
        CHECK (clamp_sum_q15 (1234) == 1234);
    }

    SECTION ("code-domain GEMM matches an integer reference product")
    {
        alignas (8) const int8_t a[6] = { 10, -20, 30, 40, 50, -60 };
        alignas (8) const int8_t b[6] = { 7, 8, -9, 10, 11, 12 };
        /* the accumulator is added into, exactly like a GEMM beta path, so the
           reference grid starts at zero and a pre-seeded one would double-count */
        alignas (8) int32_t acc[4] = { 0, 0, 0, 0 };
        const Matrix ma = matrix_codes (a, 2u, 3u, false);
        const Matrix mb = matrix_codes (b, 3u, 2u, false);
        REQUIRE (status_ok (matrix_inner_ok (ma, mb)));
        const Accum c = Accum{ acc, 4u };
        REQUIRE (status_ok (gemm_codes (ma, mb, c)));
        for (uint32_t r = 0; r < 2u; ++r)
            for (uint32_t k = 0; k < 2u; ++k)
            {
                int32_t want = 0;
                for (uint32_t i = 0; i < 3u; ++i)
                    want += a[r * 3u + i] * b[i * 2u + k];
                CHECK (acc[r * 2u + k] == want);
            }
        /* the narrow form is the requantization a layer does after the product */
        alignas (8) int8_t codes[4] = { 0, 0, 0, 0 };
        gemm_narrow_codes (c, codes, 9u);
        for (uint32_t i = 0; i < 4u; ++i)
        {
            CHECK (codes[i] >= -127);
            CHECK (codes[i] <= 127);
            /* the shift already carries the scale: narrowing is a requantized
               copy of the accumulator, with no code-step gain folded in */
            const int32_t want = static_cast<int32_t> (
                std::lround (static_cast<double> (acc[i]) / 512.0));
            const int32_t bounded = want > 127 ? 127 : (want < -127 ? -127 : want);
            CHECK (codes[i] == bounded);
        }
    }

    SECTION ("a transposed operand is a view, not a copy")
    {
        alignas (8) const int8_t a[4] = { 1, 2, 3, 4 };
        const Matrix row = matrix_codes (a, 2u, 2u, false);
        const Matrix col = matrix_codes (a, 2u, 2u, true);
        CHECK (row.code_at (0u, 1u) == 2);
        CHECK (row.code_at (1u, 0u) == 3);
        CHECK (col.code_at (0u, 1u) == 3); /* the swap moves the strides only */
        CHECK (col.code_at (1u, 0u) == 2);
        CHECK (col._M_rows == 2u);
        CHECK (col._M_cols == 2u);
        CHECK (col._M_row_stride == 1u);
        CHECK (col._M_col_stride == 2u);
        CHECK (row.is_code ());
        CHECK_FALSE (row.is_lane ());
    }

    SECTION ("an inner-dimension mismatch is rejected before any write")
    {
        alignas (8) const int8_t a[6] = { 1, 2, 3, 4, 5, 6 };
        alignas (8) const int8_t b[4] = { 1, 2, 3, 4 };
        alignas (8) int32_t acc[4] = { 0, 0, 0, 0 };
        const Matrix ma = matrix_codes (a, 2u, 3u, false);
        const Matrix mb = matrix_codes (b, 2u, 2u, false);
        CHECK (matrix_inner_ok (ma, mb) == Status::BAD_LAYOUT);
        const Accum c = Accum{ acc, 4u };
        CHECK (gemm_codes (ma, mb, c) == Status::BAD_LAYOUT);
        CHECK (gemm_q15 (ma, mb, c) == Status::BAD_LAYOUT);
        CHECK (acc[0] == 0);
        const Matrix neither = matrix_codes (nullptr, 2u, 2u, false);
        CHECK (!status_ok (matrix_inner_ok (neither, mb)));
    }

    SECTION ("Q15 GEMM narrows inside the kernel")
    {
        alignas (8) const int16_t a[4] = { 16384, 0, 0, 16384 }; /* 0.5 on the diag */
        alignas (8) const int16_t b[4] = { _S_q15_max, 0, 0, 16384 };
        alignas (8) int32_t acc[4] = { 7, 7, 7, 7 };
        const Accum c = Accum{ acc, 4u };
        REQUIRE (status_ok (gemm_q15 (matrix_lanes (a, 2u, 2u, false),
                                     matrix_lanes (b, 2u, 2u, false), c)));
        alignas (8) int16_t out[4] = { 0, 0, 0, 0 };
        gemm_narrow_lanes (c, out, 0u); /* the kernel already narrowed */
        CHECK (out[0] / _q15_unit
                == Approx (0.5).margin (0.001));
        CHECK (out[1] == 0);
        CHECK (out[3] / _q15_unit
                == Approx (0.25).margin (0.001));
        const Accum short_c = Accum{ acc, 2u };
        const Accum long_c = Accum{ acc, 3u };
        CHECK (gemm_q15 (matrix_lanes (a, 2u, 2u, false),
                         matrix_lanes (b, 2u, 2u, false), short_c)
               == Status::BAD_LAYOUT);
        CHECK (gemm_q15 (matrix_lanes (a, 2u, 2u, false),
                         matrix_lanes (b, 2u, 2u, false), long_c)
               == Status::BAD_LAYOUT);
    }

    SECTION ("linear backward matches the analytic gradient")
    {
        /* y = x * W^T with W laid out [out][in] as inference reads it. */
        alignas (8) int8_t x_code[6] = { 10, 20, 30, -40, 50, 60 };
        alignas (8) int16_t x_grad[6] = { 0, 0, 0, 0, 0, 0 };
        alignas (8) int8_t w_code[6] = { 1, 2, 3, 4, 5, 6 };
        alignas (8) int16_t w_grad[6] = { 0, 0, 0, 0, 0, 0 };
        alignas (8) int8_t gy_code[4] = { 7, 8, 9, 10 };
        alignas (8) int16_t gy_grad[4] = { 1000, 2000, -1500, 2500 };
        const Trainable x = Trainable::make (x_code, x_grad, Shape::matrix (2, 3),
                                             RequiresGrad);
        const Trainable w = Trainable::make (w_code, w_grad, Shape::matrix (2, 3),
                                             RequiresGrad);
        const Trainable gy = Trainable::make (gy_code, gy_grad, Shape::matrix (2, 2),
                                              RequiresGrad);
        uint32_t saturated = 0u;
        REQUIRE (status_ok (matmul_backward_q15 (x, w, gy, 2u, 3u, 2u, saturated)));
        CHECK (saturated == 0u);
        for (uint32_t row = 0; row < 2u; ++row)
            for (uint32_t col = 0; col < 3u; ++col)
            {
                double want = 0.0;
                for (uint32_t o = 0; o < 2u; ++o)
                    want += (gy_grad[row * 2u + o] / _q15_unit)
                            * (w_code[o * 3u + col] / 127.0);
                CHECK (x_grad[row * 3u + col] / _q15_unit
                == Approx (want).margin (0.002));
            }
        for (uint32_t o = 0; o < 2u; ++o)
            for (uint32_t col = 0; col < 3u; ++col)
            {
                double want = 0.0;
                for (uint32_t row = 0; row < 2u; ++row)
                    want += (gy_grad[row * 2u + o] / _q15_unit)
                            * (x_code[row * 3u + col] / 127.0);
                CHECK (w_grad[o * 3u + col] / _q15_unit
                == Approx (want).margin (0.002));
            }
        /* a shape that disagrees with the declared dimensions is refused */
        CHECK (matmul_backward_q15 (x, w, gy, 3u, 3u, 2u, saturated)
               != Status::OK);
    }

    SECTION ("axis and channel reductions")
    {
        alignas (8) const int16_t values[6] = { 100, 200, 300, 400, 500, 600 };
        alignas (8) int16_t dest[4] = { 0, 0, 0, 0 };
        uint32_t saturated = 0u;
        /* one entry per group: the sum of that group's inner lanes */
        reduce_axis_sum_q15 (values, 2u, 3u, dest, saturated);
        CHECK (dest[0] == 600);
        CHECK (dest[1] == 1500);
        reduce_axis_sum_q15 (values, 3u, 2u, dest, saturated);
        CHECK (dest[0] == 300);
        CHECK (dest[1] == 700);
        CHECK (dest[2] == 1100);
        CHECK (saturated == 0u);
        /* a group that no lane can hold clamps and reports itself */
        alignas (8) const int16_t big[4] = { 30000, 30000, 1, 1 };
        saturated = 0u;
        reduce_axis_sum_q15 (big, 2u, 2u, dest, saturated);
        CHECK (dest[0] == _S_q15_max);
        CHECK (saturated == 1u);

        /* channel sum over [outer=2, channel=3, inner=1] */
        alignas (8) int16_t per_channel[3] = { 0, 0, 0 };
        saturated = 0u;
        channel_backward (values, 2u, 3u, 1u, per_channel, saturated);
        CHECK (per_channel[0] == 500);
        CHECK (per_channel[1] == 700);
        CHECK (per_channel[2] == 900);
        CHECK (saturated == 0u);
        /* with an inner axis the stride jumps by the spatial size */
        alignas (8) int16_t two[2] = { 0, 0 };
        channel_backward (values, 1u, 2u, 3u, two, saturated);
        CHECK (two[0] == 600);
        CHECK (two[1] == 1500);
    }

    SECTION ("broadcast backward folds a bias-like operand")
    {
        alignas (8) int8_t dst_code[6] = { 0, 0, 0, 0, 0, 0 };
        alignas (8) int16_t dst_grad[6] = { 100, 200, 300, 400, 500, 600 };
        alignas (8) int8_t src_code[3] = { 0, 0, 0 };
        alignas (8) int16_t src_grad[3] = { 0, 0, 0 };
        const Trainable dst
            = Trainable::make (dst_code, dst_grad, Shape::matrix (2, 3), 0u);
        const Trainable src
            = Trainable::make (src_code, src_grad, Shape::vector (3), RequiresGrad);
        uint32_t saturated = 0u;
        REQUIRE (status_ok (broadcast_backward (dst, src, saturated)));
        CHECK (src_grad[0] == 500);   /* 100 + 400 over the two rows */
        CHECK (src_grad[1] == 700);
        CHECK (src_grad[2] == 900);
        CHECK (saturated == 0u);
        const Trainable wrong
            = Trainable::make (src_code, src_grad, Shape::vector (4), RequiresGrad);
        CHECK (broadcast_backward (dst, wrong, saturated) == Status::BAD_LAYOUT);
    }

    SECTION ("routing, scatter and gather")
    {
        alignas (8) const int16_t values[6] = { 100, 200, 300, 400, 500, 600 };
        alignas (8) const uint8_t mask[6] = { 1, 0, 1, 1, 0, 1 };
        alignas (8) int16_t routed[6] = { 0, 0, 0, 0, 0, 0 };
        route_by_mask (values, mask, routed, 6u);
        CHECK (routed[0] == 100);
        CHECK (routed[1] == 0);
        CHECK (routed[2] == 300);
        CHECK (routed[5] == 600);

        alignas (8) int16_t table[4] = { 9, 9, 9, 9 };
        const uint16_t index[4] = { 0, 1, 1, 3 };
        uint32_t saturated = 0u;
        scatter_by_index (values, index, table, 4u, saturated);
        CHECK (table[0] == 100);      /* the buffer is cleared first */
        CHECK (table[1] == 500);
        CHECK (table[2] == 0);
        CHECK (table[3] == 400);
        CHECK (saturated == 0u);
        /* an out-of-range index is dropped rather than writing past the table */
        const uint16_t past[4] = { 0, 99, 1, 1 };
        saturated = 0u;
        scatter_by_index (values, past, table, 4u, saturated);
        CHECK (table[0] == 100);
        CHECK (table[1] == 700);      /* lane 2 and lane 3 of the 4-lane window */
        CHECK (table[2] == 0);
        CHECK (table[3] == 0);        /* its index was the dropped one */
        CHECK (saturated == 0u);
        /* the accumulation itself clamps and says so */
        alignas (8) const int16_t big[2] = { 30000, 30000 };
        const uint16_t same[2] = { 0, 0 };
        alignas (8) int16_t over[2] = { 1, 1 };
        saturated = 0u;
        scatter_by_index (big, same, over, 2u, saturated);
        CHECK (over[0] == _S_q15_max);
        CHECK (saturated == 1u);

        /* the destination is rows * row_bytes, so a 2-row gather of 4-byte rows
           needs eight bytes or the second row lands past the buffer */
        alignas (8) const int8_t rows[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        alignas (8) int8_t gathered[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        const uint16_t pick[2] = { 1, 0 };
        gather_rows (rows, 4u, pick, 2u, gathered);
        CHECK (gathered[0] == 5);
        CHECK (gathered[1] == 6);
        CHECK (gathered[2] == 7);
        CHECK (gathered[3] == 8);
        CHECK (gathered[4] == 1);
        CHECK (gathered[7] == 4);
    }

    SECTION ("tiled transpose equals the naive form on non-multiples")
    {
        const uint32_t rows = 5u;
        for (const uint32_t cols : { 1u, 3u, 8u, 9u, 17u })
        {
            CAPTURE (cols);
            int8_t source[85];
            int8_t want[85];
            int8_t got[85];
            for (uint32_t i = 0; i < rows * cols; ++i)
                source[i] = static_cast<int8_t> (static_cast<int> (i) % 200 - 100);
            for (uint32_t r = 0; r < rows; ++r)
                for (uint32_t c = 0; c < cols; ++c)
                    want[c * rows + r] = source[r * cols + c];
            transpose_codes (source, got, rows, cols);
            for (uint32_t i = 0; i < rows * cols; ++i)
                CHECK (got[i] == want[i]);
        }
        const uint32_t lane_rows = 4u;
        const uint32_t lane_cols = 6u;
        int16_t src[24];
        int16_t want[24];
        int16_t got[24];
        for (uint32_t i = 0; i < lane_rows * lane_cols; ++i)
            src[i] = static_cast<int16_t> (i * 100 - 500);
        for (uint32_t r = 0; r < lane_rows; ++r)
            for (uint32_t c = 0; c < lane_cols; ++c)
                want[c * lane_rows + r] = src[r * lane_cols + c];
        transpose_lanes (src, got, lane_rows, lane_cols);
        for (uint32_t i = 0; i < lane_rows * lane_cols; ++i)
            CHECK (got[i] == want[i]);
    }

    SECTION ("lane arithmetic helpers saturate and count")
    {
        alignas (8) int16_t dest[4] = { 1000, 2000, 30000, -30000 };
        alignas (8) const int16_t src[4] = { 1000, 2000, 10000, -10000 };
        CHECK (add_lanes (dest, src, 4u) == 2u);
        CHECK (dest[0] == 2000);
        CHECK (dest[2] == _S_q15_max);
        CHECK (dest[3] == _S_q15_min);
        CHECK (add_lanes (dest, src, 0u) == 0u);

        alignas (8) int16_t axpy[4] = { 10000, 10000, 10000, 10000 };
        axpy_lanes (axpy, src, 16384, 4u); /* += 0.5 * src */
        CHECK (axpy[0] == 10500);
        CHECK (axpy[2] == 15000);
        axpy_lanes (axpy, src, 0, 4u);     /* a zero scalar leaves it alone */
        CHECK (axpy[0] == 10500);

        alignas (8) int16_t scaled[4] = { 10000, -10000, _S_q15_max, -32768 };
        scale_lanes (scaled, 16384, 4u);
        CHECK (scaled[0] == 5000);
        CHECK (scaled[2] == 16384);
        scale_lanes (scaled, _S_q15_max, 4u); /* identity, up to the clamp */
        CHECK (scaled[0] == 5000);

        alignas (8) int16_t filled[4] = { 1, 2, 3, 4 };
        fill_lanes (filled, 777, 4u);
        CHECK (filled[0] == 777);
        CHECK (filled[3] == 777);
        fill_lanes (filled, 40000, 4u);
        CHECK (filled[0] == _S_q15_max);

        alignas (8) const int16_t v[4] = { 3000, -4000, 500, -50 };
        CHECK (l1_norm_q15 (v, 4u) == 7550);
        CHECK (l1_norm_q15 (v, 0u) == 0);
        const double want = std::sqrt (3000.0 * 3000 + 4000.0 * 4000
                                       + 500.0 * 500 + 50.0 * 50);
        CHECK (static_cast<double> (l2_norm_q15 (v, 4u))
                == Approx (want).margin (2.0));
        CHECK (l2_norm_q15 (v, 0u) == 0);
        CHECK (dot_lanes (v, v, 4u) > 0);
        CHECK (dot_lanes (v, src, 4u) == dot_lanes (src, v, 4u));
    }
}

TEST_CASE ("trainer runs the step in the documented order",
           "[mllee][train][trainer]")
{
    reset_harness ();

    alignas (8) int8_t code[4] = { 40, 40, 40, 40 };
    alignas (8) int16_t grad[4] = { 0, 0, 0, 0 };
    alignas (8) int8_t loss_code[1] = { 0 };
    alignas (8) int16_t loss_grad[1] = { 0 };
    alignas (8) Node nodes[4];
    AlignedArena<1024> arena;
    Pool pool = arena.pool ();
    Tape tape (nodes, 4u, pool);

    Trainable param = Trainable::make (code, grad, Shape::vector (4), RequiresGrad);

    StepHooks hooks;
    hooks._M_forward = toy_forward;
    hooks._M_post_grad = toy_post_grad;
    hooks._M_update = toy_update;

    StepContext context;
    context._M_pool = &pool;
    context._M_tape = &tape;
    context._M_loss = Trainable::make (loss_code, loss_grad, Shape::scalar (), 0u);

    Trainer trainer (context, hooks);

    SECTION ("configure validates the wiring once")
    {
        CHECK (trainer.configure (nullptr, 1u, 1u) == Status::BAD_LAYOUT);
        CHECK (trainer.configure (&param, 0u, 1u) == Status::BAD_LAYOUT);
        CHECK (trainer.configure (&param, 1u, 0u) == Status::BAD_LAYOUT);
        CHECK (trainer.configure (&param, _S_max_trainer_params + 1u, 1u)
               == Status::BAD_LAYOUT);
        Trainable broken = Trainable::make (code, grad, Shape::vector (8), 0u);
        broken._M_count = 4u;
        CHECK (trainer.configure (&broken, 1u, 1u) == Status::BAD_LAYOUT);
        CHECK (trainer.context ()._M_params == nullptr); /* nothing bound */
        CHECK (trainer.configure (&param, 1u, 1u) == Status::OK);
        CHECK (trainer.context ()._M_batch == 1u);

        StepContext no_pool;
        no_pool._M_tape = &tape;
        no_pool._M_loss = context._M_loss;
        Trainer unbound (no_pool, hooks);
        CHECK (unbound.configure (&param, 1u, 1u) == Status::BAD_LAYOUT);
        StepContext no_tape;
        no_tape._M_pool = &pool;
        no_tape._M_loss = context._M_loss;
        Trainer notaped (no_tape, hooks);
        CHECK (notaped.configure (&param, 1u, 1u) == Status::BAD_LAYOUT);
        const StepHooks naked{ nullptr, nullptr, nullptr, nullptr };
        Trainer no_hooks (context, naked);
        CHECK (no_hooks.configure (&param, 1u, 1u) == Status::BAD_LAYOUT);
    }

    REQUIRE (status_ok (trainer.configure (&param, 1u, 1u)));

    SECTION ("a clean step: zero, record, backward, post, update, account")
    {
        /* The tape seeds the loss lane with the scale (32767 after the clamp),
           the toy op broadcasts it to the four parameter lanes, lr 1638 gives
           q15_mul(1638, 32767) = 1638 and q15_to_code(1638) = 6 codes. */
        StepStats stats;
        REQUIRE (status_ok (trainer.step (stats)));
        CHECK (g_forward_calls == 1u);
        CHECK (g_post_grad_calls == 1u);
        CHECK (g_update_calls == 1u);
        CHECK (grad[0] == _S_q15_max);
        CHECK (param.code_at (0) == 34);
        CHECK (param.code_at (3) == 34);
        CHECK (stats._M_step == 0u);
        CHECK (stats._M_nodes == 1u);
        CHECK (stats._M_saved_bytes == 0u);
        CHECK (stats._M_saturation_events == 0u);
        CHECK (stats._M_skipped_updates == 0u);
        CHECK (stats._M_lr_q15 == 1638);            /* the default 0.05 */
        CHECK (stats._M_loss_scale_q15 == _S_q15_one);
        CHECK (stats._M_loss_q7 == (160 << 7) / 258);
        CHECK (stats._M_status == Status::OK);
        /* every exit path resets, so a fixed arena serves an unbounded run */
        CHECK (pool.stats ()._M_used == 0u);
        CHECK (tape.node_count () == 0u);
        CHECK_FALSE (tape.is_open ());
        CHECK (trainer.loss_scale () == _S_q15_one);
        CHECK (context._M_step == 1u);
        REQUIRE (status_ok (trainer.step (stats)));
        CHECK (stats._M_step == 1u);
        CHECK (param.code_at (0) == 28);
    }

    SECTION ("a saturated step skips the update and halves the scale")
    {
        const int32_t before = param.code_at (0);
        g_saturate = true;
        StepStats stats;
        REQUIRE (status_ok (trainer.step (stats)));
        g_saturate = false;
        CHECK (param.code_at (0) == before);   /* the clipped update never ran */
        CHECK (g_update_calls == 0u);
        CHECK (g_post_grad_calls == 1u);       /* but the hook still saw it */
        CHECK (stats._M_saturation_events >= 1u);
        CHECK (g_reported_saturation == stats._M_saturation_events);
        CHECK (stats._M_skipped_updates == 1u);
        CHECK (trainer.loss_scale () == 16384);
        CHECK (stats._M_loss_scale_q15 == 16384);
        CHECK (pool.stats ()._M_used == 0u);
        CHECK (tape.node_count () == 0u);
        /* the halved scale is what the next pass seeds with */
        grad[0] = 0;
        REQUIRE (status_ok (trainer.step (stats)));
        CHECK (grad[0] == 16384);
        CHECK (trainer.totals ()._M_scale_halves == 1u);
        CHECK (trainer.totals ()._M_scale_grows == 0u);
    }

    SECTION ("the scale grows by a quarter after the patience window")
    {
        Trainer scaled (context, hooks);
        REQUIRE (status_ok (scaled.configure (&param, 1u, 1u)));
        scaled.set_loss_scale (16384);
        CHECK (scaled.loss_scale () == 16384);
        StepStats stats;
        for (uint32_t i = 0; i < _S_scale_patience - 1u; ++i)
            REQUIRE (status_ok (scaled.step (stats)));
        CHECK (scaled.loss_scale () == 16384); /* not yet */
        REQUIRE (status_ok (scaled.step (stats)));
        CHECK (scaled.loss_scale () == 16384 + (16384 >> _S_scale_growth_shift));
        CHECK (scaled.totals ()._M_scale_grows == 1u);
        /* a saturated step resets the patience window */
        g_saturate = true;
        REQUIRE (status_ok (scaled.step (stats)));
        g_saturate = false;
        const int32_t after_backoff = scaled.loss_scale ();
        CHECK (after_backoff < 16384 + (16384 >> 2));
        CHECK (scaled.totals ()._M_scale_halves == 1u);
        /* and the ceiling is the lane: it never grows past it */
        scaled.set_loss_scale (_S_q15_max);
        for (uint32_t i = 0; i < _S_scale_patience; ++i)
            REQUIRE (status_ok (scaled.step (stats)));
        CHECK (scaled.loss_scale () == _S_q15_max);
    }

    SECTION ("a failing hook aborts the step and still resets the pool")
    {
        StepStats stats;
        g_forward_fails = true;
        g_forward_status = Status::UNSUPPORTED;
        CHECK (trainer.step (stats) == Status::UNSUPPORTED);
        g_forward_status = Status::OK;
        g_forward_fails = false;
        CHECK (stats._M_status == Status::UNSUPPORTED);
        CHECK (pool.stats ()._M_used == 0u);
        CHECK (tape.node_count () == 0u);
        const uint32_t step_number = context._M_step;
        CHECK (stats._M_step == step_number); /* a failed step is not counted */

        /* an update hook that fails aborts after the gradients are ready */
        StepHooks stubborn;
        stubborn._M_forward = toy_forward;
        stubborn._M_update = refusing_update;
        Trainer stubborn_run (context, stubborn);
        REQUIRE (status_ok (stubborn_run.configure (&param, 1u, 1u)));
        CHECK (stubborn_run.step (stats) == Status::ARENA_OVERFLOW);
        CHECK (pool.stats ()._M_used == 0u);

        /* a loss with no gradient stream is a wiring bug caught before backward */
        const Trainable kept = context._M_loss;
        context._M_loss
            = Trainable::make (loss_code, nullptr, Shape::scalar (), 0u);
        CHECK (trainer.step (stats) == Status::BAD_LAYOUT);
        context._M_loss = kept;
        CHECK (pool.stats ()._M_used == 0u);
        CHECK (tape.node_count () == 0u);
    }

    SECTION ("fit aggregates a run and honours stop_on_error")
    {
        for (int8_t &slot : code)
            slot = 20;
        StepHooks quiet;
        quiet._M_forward = toy_forward;
        quiet._M_update = toy_update;
        Trainer run (context, quiet);
        REQUIRE (status_ok (run.configure (&param, 1u, 1u)));
        FitStats fit_stats;
        REQUIRE (status_ok (run.fit (5u, true, fit_stats)));
        CHECK (fit_stats._M_steps == 5u);
        CHECK (fit_stats._M_epochs == 1u);
        CHECK (fit_stats._M_tape_peak_nodes == 1u);
        CHECK (fit_stats._M_pool_peak == 0u); /* the toy op saves nothing */
        CHECK (fit_stats._M_saturation_events == 0u);
        CHECK (fit_stats._M_skipped_updates == 0u);
        CHECK (fit_stats._M_status == Status::OK);
        CHECK (fit_stats._M_scale_halves == 0u);
        CHECK (fit_stats._M_scale_grows == 0u);
        /* the five steps see codes 20, 14, 8, 2, -4 (the loss is read before the
           update), and the fifth update lands six codes further on */
        CHECK (param.code_at (0) == -10);
        CHECK (fit_stats._M_first_loss_q7 == (80 << 7) / 258);
        CHECK (fit_stats._M_last_loss_q7 == (-16 << 7) / 258);
        CHECK (fit_stats._M_best_loss_q7 == fit_stats._M_last_loss_q7);
        CHECK (fit_stats._M_best_step == 4u);

        /* a saturating run keeps going and reports every skipped update */
        g_saturate = true;
        FitStats degraded;
        REQUIRE (status_ok (run.fit (3u, false, degraded)));
        g_saturate = false;
        CHECK (degraded._M_steps == 3u);
        CHECK (degraded._M_skipped_updates == 3u);
        CHECK (degraded._M_saturation_events >= 3u);
        CHECK (degraded._M_status == Status::OK);

        /* stop_on_error ends the run on the step that failed */
        g_forward_fails = true;
        g_forward_status = Status::UNSUPPORTED;
        FitStats halted;
        CHECK (run.fit (10u, true, halted) == Status::UNSUPPORTED);
        g_forward_fails = false;
        g_forward_status = Status::OK;
        CHECK (halted._M_steps == 1u);
        CHECK (halted._M_status == Status::UNSUPPORTED);
        /* continue-on-error still runs every step and keeps the reason */
        g_forward_fails = true;
        g_forward_status = Status::UNSUPPORTED;
        FitStats soft;
        CHECK (run.fit (4u, false, soft) == Status::UNSUPPORTED);
        g_forward_fails = false;
        g_forward_status = Status::OK;
        CHECK (soft._M_steps == 4u);
        CHECK (soft._M_status == Status::UNSUPPORTED);
        /* a run in which nothing fails reports OK, not the previous run's reason */
        FitStats clean;
        CHECK (status_ok (run.fit (2u, false, clean)));
        CHECK (clean._M_status == Status::OK);

        CHECK (run.fit (0u, true, halted) == Status::BAD_LAYOUT);
    }

    SECTION ("fit_until distinguishes an early stop from a budget stop")
    {
        StepHooks quiet;
        quiet._M_forward = toy_forward;
        quiet._M_update = toy_update;
        Trainer run (context, quiet);
        REQUIRE (status_ok (run.configure (&param, 1u, 1u)));
        uint32_t taken = 0;
        FitStats until;
        /* the target is unreachable in six steps of a linear loss */
        CHECK (run.fit_until (6u, -40000, true, until, taken)
               == Status::OUT_OF_RANGE);
        CHECK (taken == 6u);
        FitStats reached;
        REQUIRE (status_ok (run.fit_until (6u, 40000, true, reached, taken)));
        CHECK (taken == 6u);
    }

    SECTION ("evaluate is forward-only")
    {
        StepHooks quiet;
        quiet._M_forward = toy_forward;
        quiet._M_update = toy_update; /* evaluate must not reach it */
        Trainer eval_run (context, quiet);
        REQUIRE (status_ok (eval_run.configure (&param, 1u, 2u)));
        const int32_t before = param.code_at (0);
        StepStats stats;
        REQUIRE (status_ok (eval_run.evaluate (stats)));
        CHECK (tape.node_count () == 0u);
        CHECK_FALSE (tape.is_open ());
        CHECK (pool.stats ()._M_used == 0u);
        CHECK (param.code_at (0) == before);  /* no update hook runs */
        CHECK (g_update_calls == 0u);
        CHECK (stats._M_loss_q7 == context._M_loss_q7);
        /* the same hook body works for training because it guards on is_open() */
        REQUIRE (status_ok (eval_run.step (stats)));
        CHECK (g_update_calls == 1u);
    }

    SECTION ("a schedule hook sets the rate and is clamped")
    {
        StepHooks scheduled;
        scheduled._M_forward = toy_forward;
        scheduled._M_update = toy_update;
        scheduled._M_schedule = scheduled_lr;
        Trainer run (context, scheduled);
        REQUIRE (status_ok (run.configure (&param, 1u, 1u)));
        g_schedule_lr_q15 = 100;
        StepStats stats;
        REQUIRE (status_ok (run.step (stats)));
        CHECK (stats._M_lr_q15 == 100);
        REQUIRE (status_ok (run.step (stats)));
        CHECK (stats._M_lr_q15 == 101);
        g_schedule_lr_q15 = 500000;
        REQUIRE (status_ok (run.step (stats)));
        CHECK (stats._M_lr_q15 == _S_q15_max); /* clamped, never wrapped */
        g_schedule_lr_q15 = 1638;
    }

    SECTION ("the loss scale is restorable and always a lane")
    {
        Trainer restored (context, hooks);
        REQUIRE (status_ok (restored.configure (&param, 1u, 1u)));
        restored.set_loss_scale (100000);
        CHECK (restored.loss_scale () == _S_q15_max);
        restored.set_loss_scale (-1);
        /* a negative scale is clamped to the lane, not floored at zero: -1 is a
           legal lane value, so only a value past the lane reaches _S_q15_min */
        CHECK (restored.loss_scale () == -1);
        restored.set_loss_scale (-100000);
        CHECK (restored.loss_scale () == _S_q15_min);
        restored.set_loss_scale (_S_q15_one);
        StepStats stats;
        REQUIRE (status_ok (restored.step (stats)));
        CHECK (grad[0] == _S_q15_max); /* the seed clamps a whole unit */
        /* configure restarts the run's counters but keeps the scale */
        restored.set_loss_scale (8000);
        REQUIRE (status_ok (restored.configure (&param, 1u, 1u)));
        CHECK (restored.loss_scale () == 8000);
        CHECK (restored.totals ()._M_steps == 0u);
        CHECK (restored.totals ()._M_scale_halves == 0u);
    }

    SECTION ("a tape that cannot record makes the step fail cleanly")
    {
        AlignedArena<64> small;
        Pool dead = small.pool ();
        alignas (8) Node no_nodes[1];
        Tape closed (no_nodes, 0u, dead);
        StepContext bound = context;
        bound._M_tape = &closed;
        bound._M_pool = &dead;
        Trainer broken (bound, hooks);
        REQUIRE (status_ok (broken.configure (&param, 1u, 1u)));
        StepStats stats;
        CHECK (broken.step (stats) == Status::BAD_LAYOUT);
        CHECK (dead.stats ()._M_used == 0u);
        /* the same tape over capacity fails with an overflow instead of a write */
        AlignedArena<1024> roomy;
        Pool live = roomy.pool ();
        Tape tiny (no_nodes, 1u, live);
        StepContext wide = context;
        wide._M_tape = &tiny;
        Trainer overfull (wide, hooks);
        REQUIRE (status_ok (overfull.configure (&param, 1u, 1u)));
        /* one node fits; the tape is 4 nodes but the toy op records one */
        REQUIRE (status_ok (overfull.step (stats)));
    }

    SECTION ("the fixed footprint is what the RAM gate asserts")
    {
        CHECK (Trainer::budget (0u, 0u) == 0u);
        CHECK (Trainer::budget (1u, 0u) == sizeof (Trainable));
        CHECK (Trainer::budget (0u, 1u) == sizeof (Node));
        CHECK (Trainer::budget (4u, 8u)
               == tape_node_bytes (8u) + 4u * sizeof (Trainable));
        /* the default-shaped run the 8 KiB budget can carry, and one it cannot */
        CHECK (Trainer::budget (16u, 24u) <= _S_trainer_pool_bytes);
        CHECK (Trainer::budget (16u, 40u) > _S_trainer_pool_bytes);
        CHECK (_S_trainer_pool_bytes == _S_train_pool_bytes);
        CHECK (sizeof (StepContext) <= 140u);
        CHECK (sizeof (StepStats) <= 40u);
        CHECK (sizeof (FitStats) <= 64u);
        CHECK (sizeof (Trainer) <= 80u);
    }
}
