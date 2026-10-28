/* Host unit tests for three mllee training modules that had no coverage:
 * mllee/train_linear.hpp (LinearSpec affine layer), mllee/train_embedding.hpp
 * (EmbedSpec lookup / bag / tied views) and mllee/train_running_stats.hpp
 * (RunningStats EMA buffers).
 *
 * WHY THE REFERENCE IS INDEPENDENT
 * ---------------------------------
 * Every numeric expectation below is recomputed inside this file from the
 * integer rules the headers document -- round-half-up shifts, the 258-unit Q15
 * code step, the [-127, 127] symmetric int8 rails and the Bessel n/(n-1) ratio
 * -- using our own `ref_*` helpers.  We never call a library kernel and echo
 * its result back as the "expected" value.  The forward matrix-vector product
 * is written as a plain reference mat-vec over int8 codes; the embedding gather
 * and bag mean are written as explicit loops; the running-stats update is a
 * hand-derived EMA.  If a library refactor moves a rounding rule, these
 * assertions fail here instead of silently re-pointing the reference at the new
 * behavior. */

#include "mllee/quant.hpp"
#include "mllee/train_activations.hpp"
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_embedding.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_linear.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_random.hpp"
#include "mllee/train_running_stats.hpp"
#include "mllee/train_tensor.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

using namespace zzs::mllee;
using namespace zzs::mllee::train;

namespace
{
/* ------------------------------------------------------------------ */
/* Independent integer reference primitives.                           */
/* ------------------------------------------------------------------ */

/* round-half-up right shift, the documented `_S_round_shift` rule. */
int32_t
ref_round_shift (const int64_t value, const unsigned shift)
{
    if (shift == 0u)
        return static_cast<int32_t> (value);
    const int64_t half = static_cast<int64_t> (1) << (shift - 1u);
    return static_cast<int32_t> ((value + half) >> shift);
}

/* symmetric int8 rail: -128 is never emitted. */
int32_t
ref_i8 (const int32_t value)
{
    return value < -127 ? -127 : (value > 127 ? 127 : value);
}

/* the Q15 code step is 258 units. */
int32_t
ref_code_q15 (const int32_t code)
{
    return code * 258;
}

/* narrow an int64 dot-product sum into a Q15 lane (train_backward_kernels). */
int32_t
ref_narrow_sum (const int64_t total)
{
    const int64_t limit = static_cast<int64_t> (32767) << 15;
    const int64_t bounded
        = total > limit ? limit : (total < -limit ? -limit : total);
    const int32_t shifted = ref_round_shift (bounded, 15u);
    return shifted > 32767 ? 32767 : (shifted < -32768 ? -32768 : shifted);
}

/* saturate a plain Q15 lane sum to [-32768, 32767]. */
int32_t
ref_clamp_sum (const int64_t total)
{
    return total > 32767 ? 32767 : (total < -32768 ? -32768
                                                   : static_cast<int32_t> (total));
}

int32_t
ref_clamp_q15 (const int32_t value)
{
    return value > 32767 ? 32767 : (value < -32768 ? -32768 : value);
}

/* round half away from zero, exactly the header's div_round. */
int32_t
ref_div_round (int32_t numerator, const int32_t denominator)
{
    if (denominator == 0)
        return 0;
    const bool negative = (numerator < 0) != (denominator < 0);
    if (numerator < 0)
        numerator = -numerator;
    const int32_t den = denominator < 0 ? -denominator : denominator;
    const int32_t quotient = (numerator + den / 2) / den;
    return negative ? -quotient : quotient;
}

/* floor integer square root of a non-negative integer. */
uint64_t
ref_isqrt (uint64_t n)
{
    if (n == 0u)
        return 0u;
    uint64_t x = n;
    uint64_t y = (x + 1u) / 2u;
    while (y < x)
        {
            x = y;
            y = (x + n / x) / 2u;
        }
    return x;
}

/* ------------------------------------------------------------------ */
/* Tensor builder: owns a code buffer plus optional gradient lanes.    */
/* ------------------------------------------------------------------ */
struct Tensor
{
    std::vector<int8_t> codes;
    std::vector<int16_t> grads;
};

Trainable
make_view (Tensor &t, const Shape shape, const bool with_grad)
{
    Trainable view
        = Trainable::make (t.codes.data (), with_grad ? t.grads.data () : nullptr,
                           shape, with_grad ? Flag::RequiresGrad : 0u);
    return view;
}

void
fill_codes (Tensor &t, const std::vector<int32_t> &values)
{
    t.codes.assign (values.begin (), values.end ());
    for (int8_t &c : t.codes)
        c = static_cast<int8_t> (ref_i8 (c));
    t.grads.assign (t.codes.size (), 0);
}

/* ------------------------------------------------------------------ */
/* Linear forward reference: an explicit matrix-vector product.        */
/* ------------------------------------------------------------------ */

/* acc = sum_i X[b][i] * W[o][i]; W is [out, in] in inference row order. */
int32_t
ref_accum (const Tensor &x, const Tensor &w, const uint32_t in,
           const uint32_t b, const uint32_t o)
{
    int32_t sum = 0;
    for (uint32_t i = 0; i < in; ++i)
        sum += static_cast<int32_t> (x.codes[b * in + i])
               * static_cast<int32_t> (w.codes[o * in + i]);
    return sum;
}

int32_t
ref_linear_code (const int32_t acc, const uint8_t shift, const int32_t mult,
                 const int32_t bias_code, const bool use_bias)
{
    int32_t value = ref_round_shift (acc, shift);
    if (mult != 32768)
        value = ref_round_shift (static_cast<int64_t> (value) * mult, 15u);
    if (use_bias)
        value += bias_code;
    return ref_i8 (value);
}

int32_t
ref_relu_code (const int32_t code)
{
    return code < 0 ? 0 : code;
}

/* ------------------------------------------------------------------ */
/* Running-stats references.                                           */
/* ------------------------------------------------------------------ */

int32_t
ref_lerp (const int32_t current, const int32_t target, const int32_t momentum,
          bool &saturated)
{
    const int32_t delta = target - current;
    const int32_t step = static_cast<int32_t> (
        (static_cast<int64_t> (delta) * momentum + (static_cast<int64_t> (1) << 14))
        >> 15);
    const int32_t raw = current + step;
    const int32_t value = ref_clamp_q15 (raw);
    saturated = value != raw;
    return value;
}

int32_t
ref_bessel (const uint32_t elements)
{
    if (elements <= 1u)
        return 16384;
    const uint64_t scaled
        = (static_cast<uint64_t> (elements) << 14u) / (elements - 1u);
    return scaled > 0xFFFFull ? 16384 : static_cast<int32_t> (scaled);
}

int32_t
ref_unbiased (const int32_t biased_q15, const uint32_t elements)
{
    const int32_t ratio = ref_bessel (elements);
    const int32_t value = static_cast<int32_t> (
        ((static_cast<int64_t> (biased_q15) * ratio + 8192) >> 14));
    return ref_clamp_q15 (value);
}

/* Round a byte count up to the pool's alignment, mirroring the private helper
   the Pool uses for every slab. */
constexpr uint32_t
ref_align_up (const uint32_t value)
{
    return (value + _S_pool_alignment - 1u) & ~(_S_pool_alignment - 1u);
}

/* ------------------------------------------------------------------ */
/* A do-nothing backward for the raw Tape wiring tests.                */
/* ------------------------------------------------------------------ */
Status
dummy_backward (const Trainable &, const Trainable &, GradContext &)
{
    return Status::OK;
}
} // namespace

/* ================================================================== */
/* train_linear.hpp                                                   */
/* ================================================================== */

TEST_CASE ("LinearSpec::bytes bounds the pool traffic of a recorded layer")
{
    constexpr auto spec = []
    {
        LinearSpec s;
        s._M_batch = 4u;
        s._M_in_features = 6u;
        s._M_out_features = 5u;
        s._M_requant_shift = 7u;
        s._M_act = Act::RELU;
        return s;
    } ();
    static_assert (spec.output_elements () == 20u, "the geometry is fixed here");
    static_assert (sizeof (LinearSpec) == 20u, "five pool words");

    /* the slab the run carves: input copy, pre-activation, scratch, spec, plus
       the forward's int32 accumulator, each rounded to the pool alignment */
    constexpr uint32_t rows = spec.output_elements ();
    constexpr uint32_t expected
        = ref_align_up (spec.input_elements ())               /* saved input */
          + ref_align_up (rows)                               /* pre-activation */
          + ref_align_up (2u * (rows + spec._M_out_features)) /* scratch lanes */
          + ref_align_up (static_cast<uint32_t> (sizeof (LinearSpec)))
          + ref_align_up (4u * rows);                         /* accumulator */
    static_assert (expected == 196u, "four words short of the capacity bound");
    /* `bytes ()` is a documented capacity bound, so it may round a term up but
       must never understate a real run */
    static_assert (spec.bytes () == 200u, "the bound rounds the channel term");
    REQUIRE (expected <= spec.bytes ());

    Tensor x, w, bias, out;
    fill_codes (x, { 1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6,
                     7, 7, 7, 7, 7, 7, -7, -7, -7, -7, -7, -7 });
    fill_codes (w, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                     16, 17, 18, 19, 20, -1, -2, -3, -4, -5, -6, -7, 8, 9, 4 });
    fill_codes (bias, { 0, 1, -1, 2, -2 });
    fill_codes (out, { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });

    SECTION ("an arena sized exactly bytes () holds the layer")
    {
        AlignedArena<spec.bytes ()> arena;
        Pool pool = arena.pool ();
        Node nodes[2];
        Tape tape (nodes, 2u, pool);
        REQUIRE (tape.begin () == Status::OK);
        Trainable xv = make_view (x, spec.input_shape (), true);
        Trainable wv = make_view (w, spec.weight_shape (), true);
        Trainable bv = make_view (bias, spec.bias_shape (), true);
        Trainable ov = make_view (out, spec.output_shape (), true);
        REQUIRE (linear_run (tape, spec, xv, wv, bv, ov) == Status::OK);
        REQUIRE (pool.stats()._M_used == expected);
        REQUIRE (pool.remaining () == spec.bytes () - expected);
        /* every slab is 4-byte aligned, as the kernels assume */
        REQUIRE (reinterpret_cast<uintptr_t> (pool.at (0u)) % _S_pool_alignment == 0u);
    }
    SECTION ("one slab short is ARENA_OVERFLOW, not a crash")
    {
        AlignedArena<ref_align_up (expected) - 4u> arena;
        Pool pool = arena.pool ();
        Node nodes[2];
        Tape tape (nodes, 2u, pool);
        REQUIRE (tape.begin () == Status::OK);
        Trainable xv = make_view (x, spec.input_shape (), true);
        Trainable wv = make_view (w, spec.weight_shape (), true);
        Trainable bv = make_view (bias, spec.bias_shape (), true);
        Trainable ov = make_view (out, spec.output_shape (), true);
        REQUIRE (linear_run (tape, spec, xv, wv, bv, ov) == Status::ARENA_OVERFLOW);
        REQUIRE (tape.node_count () == 0u);
    }
    SECTION ("the module pool constant is the documented 8 KiB")
    {
        REQUIRE (_S_train_pool_bytes == 8192u);
        REQUIRE (_S_pool_alignment == 4u);
        REQUIRE (spec.bytes () < _S_train_pool_bytes);
    }
}

TEST_CASE ("LinearSpec validate accepts the documented grid")
{
    LinearSpec spec;
    spec._M_batch = 4u;
    spec._M_in_features = 8u;
    spec._M_out_features = 3u;
    REQUIRE (spec.validate () == Status::OK);
    REQUIRE (spec.output_elements () == 12u);
    REQUIRE (spec.input_elements () == 32u);
    REQUIRE (spec.weight_elements () == 24u);
    REQUIRE (spec.has_bias ());
    REQUIRE_FALSE (spec.has_activation ());

    spec._M_act = Act::RELU;
    REQUIRE (spec.has_activation ());
    spec._M_act = Act::NONE;
    spec._M_use_bias = 0u;
    REQUIRE_FALSE (spec.has_bias ());
}

TEST_CASE ("LinearSpec validate rejects bad geometry")
{
    SECTION ("zero axis is BAD_LAYOUT")
    {
        LinearSpec spec;
        spec._M_batch = 0u;
        REQUIRE (spec.validate () == Status::BAD_LAYOUT);
        spec._M_batch = 1u;
        spec._M_in_features = 0u;
        REQUIRE (spec.validate () == Status::BAD_LAYOUT);
        spec._M_in_features = 1u;
        spec._M_out_features = 0u;
        REQUIRE (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("impossible shift is BAD_LAYOUT")
    {
        /* the default spec has zero feature axes, which the zero-axis check
           rejects before the shift check; give the geometry life first. */
        LinearSpec spec;
        spec._M_in_features = 1u;
        spec._M_out_features = 1u;
        spec._M_requant_shift = 32u;
        REQUIRE (spec.validate () == Status::BAD_LAYOUT);
        spec._M_requant_shift = 31u;
        REQUIRE (spec.validate () == Status::OK);
    }
    SECTION ("multiplier outside [0, 32.0] is OUT_OF_RANGE")
    {
        LinearSpec spec;
        spec._M_in_features = 1u;
        spec._M_out_features = 1u;
        spec._M_requant_mult_q15 = -1;
        REQUIRE (spec.validate () == Status::OUT_OF_RANGE);
        spec._M_requant_mult_q15 = _S_linear_mult_max_q15;
        REQUIRE (spec.validate () == Status::OK);
        spec._M_requant_mult_q15 = _S_linear_mult_max_q15 + 1;
        REQUIRE (spec.validate () == Status::OUT_OF_RANGE);
    }
    SECTION ("an unaddressable count is OUT_OF_RANGE")
    {
        LinearSpec spec;
        spec._M_batch = 256u;
        spec._M_in_features = 256u;
        spec._M_out_features = 1u;
        REQUIRE (spec.validate () == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("LinearSpec shapes and bytes are consistent")
{
    LinearSpec spec;
    spec._M_batch = 3u;
    spec._M_in_features = 5u;
    spec._M_out_features = 2u;
    REQUIRE (spec.input_shape () == Shape::matrix (3u, 5u));
    REQUIRE (spec.weight_shape () == Shape::matrix (2u, 5u));
    REQUIRE (spec.bias_shape () == Shape::vector (2u));
    REQUIRE (spec.output_shape () == Shape::matrix (3u, 2u));

    const uint32_t plain = spec.bytes ();
    spec._M_act = Act::RELU;
    /* an activation layer must reserve the extra pre-activation slab */
    REQUIRE (spec.bytes () > plain);
}

TEST_CASE ("linear_forward matches an independent matrix-vector reference")
{
    /* batch=2, in=3, out=4 -- W is [out, in] row order. */
    Tensor x, w, bias, out;
    fill_codes (x, { 10, -20, 30, 7, 100, -50 });
    fill_codes (w, { 1, 2, 3, -4, 5, -6, 7, -8, 9, 10, 11, 12 });
    fill_codes (bias, { 3, -4, 5, -6 });
    fill_codes (out, { 0, 0, 0, 0, 0, 0, 0, 0 });

    LinearSpec spec;
    spec._M_batch = 2u;
    spec._M_in_features = 3u;
    spec._M_out_features = 4u;
    spec._M_requant_shift = 7u;
    spec._M_requant_mult_q15 = 32768;

    AlignedArena<1024> arena;
    Pool pool = arena.pool ();
    Trainable xv = make_view (x, Shape::matrix (2u, 3u), false);
    Trainable wv = make_view (w, Shape::matrix (4u, 3u), false);
    Trainable bv = make_view (bias, Shape::vector (4u), false);
    Trainable ov = make_view (out, Shape::matrix (2u, 4u), true);

    REQUIRE (linear_forward (pool, spec, xv, wv, bv, ov, nullptr) == Status::OK);

    for (uint32_t b = 0; b < 2u; ++b)
        for (uint32_t o = 0; o < 4u; ++o)
            {
                const int32_t acc = ref_accum (x, w, 3u, b, o);
                const int32_t want = ref_linear_code (acc, 7u, 32768,
                                                      bias.codes[o], true);
                REQUIRE (static_cast<int32_t> (out.codes[b * 4u + o]) == want);
                /* the forward always zeroes the output gradient lanes */
                REQUIRE (out.grads[b * 4u + o] == 0);
            }
    /* sanity: the reference acc for row 0, col 0 is 10*1 + -20*2 + 30*3 = 60 */
    REQUIRE (ref_accum (x, w, 3u, 0u, 0u) == 60);
}

TEST_CASE ("linear_forward without a bias ignores the bias operand")
{
    Tensor x, w, bias, out;
    fill_codes (x, { 5, 5, 5, 5, 5, 5 });
    fill_codes (w, { 4, 4, 4, 4, 4, 4 });
    fill_codes (bias, { 50, -50 });
    fill_codes (out, { 0, 0, 0, 0 });

    LinearSpec spec;
    spec._M_batch = 2u;
    spec._M_in_features = 3u;
    spec._M_out_features = 2u;
    spec._M_use_bias = 0u;

    AlignedArena<1024> arena;
    Pool pool = arena.pool ();
    Trainable xv = make_view (x, Shape::matrix (2u, 3u), false);
    Trainable wv = make_view (w, Shape::matrix (2u, 3u), false);
    Trainable bv = make_view (bias, Shape::vector (2u), false);
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);

    REQUIRE (linear_forward (pool, spec, xv, wv, bv, ov, nullptr) == Status::OK);
    /* acc per element = 5*4*3 = 60, round(60/127) = 0, no bias added */
    for (uint32_t i = 0; i < 4u; ++i)
        REQUIRE (static_cast<int32_t> (out.codes[i]) == ref_linear_code (60, 7u, 32768, 0, false));
}

TEST_CASE ("linear_forward requant multiplier path")
{
    Tensor x, w, bias, out;
    fill_codes (x, { 127, -127, 0 });
    fill_codes (w, { 100, 100, 100 });
    fill_codes (bias, { 0 });
    fill_codes (out, { 0 });

    LinearSpec spec;
    spec._M_batch = 1u;
    spec._M_in_features = 3u;
    spec._M_out_features = 1u;
    spec._M_requant_shift = 7u;
    spec._M_requant_mult_q15 = 16384; /* 0.5 in Q15 */

    AlignedArena<1024> arena;
    Pool pool = arena.pool ();
    Trainable xv = make_view (x, Shape::matrix (1u, 3u), false);
    Trainable wv = make_view (w, Shape::matrix (1u, 3u), false);
    Trainable bv = make_view (bias, Shape::vector (1u), false);
    Trainable ov = make_view (out, Shape::matrix (1u, 1u), true);

    REQUIRE (linear_forward (pool, spec, xv, wv, bv, ov, nullptr) == Status::OK);
    const int32_t acc = ref_accum (x, w, 3u, 0u, 0u);
    REQUIRE (acc == 0); /* 127*100 + -127*100 + 0 */
    REQUIRE (static_cast<int32_t> (out.codes[0])
             == ref_linear_code (acc, 7u, 16384, 0, true));
}

TEST_CASE ("linear_forward shifts table")
{
    struct Row
    {
        int32_t acc;
        uint8_t shift;
        int32_t mult;
    };
    const Row rows[] = {
        { 0, 0u, 32768 }, { 1, 1u, 32768 }, { -1, 1u, 32768 },
        { 127, 7u, 32768 }, { 64, 7u, 32768 }, { -64, 7u, 32768 },
        { 16384, 7u, 16384 }, { -32768, 7u, 32768 }, { 200000, 4u, 32768 },
    };

    Tensor x, w, bias, out;
    /* one element per row: force a chosen accumulator via x[0]=acc? codes are
       int8, so instead we exercise the requant stage directly with an acc
       helper; here we only check the code rail mapping of `ref_linear_code` */
    for (const Row &row : rows)
        {
            const int32_t code
                = ref_linear_code (row.acc, row.shift, row.mult, 0, false);
            REQUIRE (code >= -127);
            REQUIRE (code <= 127);
            /* the same grid, produced by the library primitives: a rounding
               rule that drifts here fails against an independent formula */
            REQUIRE (_S_i8_code (ref_round_shift (row.acc, row.shift))
                     == ref_i8 (ref_round_shift (row.acc, row.shift)));
            const int32_t requantized
                = _S_round_shift (static_cast<int64_t> (row.acc), row.shift);
            REQUIRE (requantized == ref_round_shift (row.acc, row.shift));
            if (row.mult != 32768)
                REQUIRE (q15_scale_shift (requantized, row.mult, _S_grad_frac_bits)
                         == ref_round_shift (
                                static_cast<int64_t> (requantized) * row.mult,
                                _S_grad_frac_bits));
        }
    REQUIRE (ref_round_shift (64, 7u) == 1);  /* round half up: (64+64)>>7 = 1 */
    REQUIRE (ref_round_shift (-64, 7u) == 0); /* (-64+64)>>7 = 0 */
    REQUIRE (ref_round_shift (1, 1u) == 1);   /* (1+1)>>1 = 1 */
    REQUIRE (ref_round_shift (-1, 1u) == 0);  /* (-1+1)>>1 = 0 */
    REQUIRE (_S_round_shift (64, 7u) == 1);
    REQUIRE (_S_round_shift (-64, 7u) == 0);
    REQUIRE (_S_round_shift (1, 1u) == 1);
    REQUIRE (_S_round_shift (-1, 1u) == 0);
    /* shift 0 is a plain cast, and -128 is never emitted by the rail */
    REQUIRE (_S_round_shift (-12345, 0u) == -12345);
    REQUIRE (_S_i8_code (-200) == -127);
    REQUIRE (_S_i8_code (200) == 127);
    REQUIRE (_S_i8_code (-128) == -127);
}

TEST_CASE ("linear_forward clamps to the symmetric int8 rails")
{
    Tensor x, w, bias, out;
    /* a saturating dot product: 127*127 * many inputs would overflow the code */
    fill_codes (x, { 127, 127 });
    fill_codes (w, { 127, 127 });
    fill_codes (bias, { 0 });
    fill_codes (out, { 0 });

    LinearSpec spec;
    spec._M_batch = 1u;
    spec._M_in_features = 2u;
    spec._M_out_features = 1u;
    spec._M_requant_shift = 0u; /* keep the raw accumulator so it saturates */

    AlignedArena<1024> arena;
    Pool pool = arena.pool ();
    Trainable xv = make_view (x, Shape::matrix (1u, 2u), false);
    Trainable wv = make_view (w, Shape::matrix (1u, 2u), false);
    Trainable bv = make_view (bias, Shape::vector (1u), false);
    Trainable ov = make_view (out, Shape::matrix (1u, 1u), true);

    REQUIRE (linear_forward (pool, spec, xv, wv, bv, ov, nullptr) == Status::OK);
    const int32_t acc = 127 * 127 + 127 * 127; /* 32258 */
    REQUIRE (acc == 32258);
    REQUIRE (static_cast<int32_t> (out.codes[0]) == 127);
    REQUIRE (static_cast<int32_t> (out.codes[0]) != -128);
}

TEST_CASE ("linear_forward fuses ReLU and keeps the pre-activation")
{
    Tensor x, w, bias, out;
    fill_codes (x, { 30, -30, 10, 10 });
    fill_codes (w, { 50, 0, 0, 50 }); /* row0=50,0 ; row1=0,50 */
    fill_codes (bias, { 0, 0 });
    fill_codes (out, { 0, 0, 0, 0 });

    LinearSpec spec;
    spec._M_batch = 2u;
    spec._M_in_features = 2u;
    spec._M_out_features = 2u;
    spec._M_requant_shift = 7u;
    spec._M_act = Act::RELU;

    int8_t pre[4] = { 9, 9, 9, 9 };
    AlignedArena<1024> arena;
    Pool pool = arena.pool ();
    Trainable xv = make_view (x, Shape::matrix (2u, 2u), false);
    Trainable wv = make_view (w, Shape::matrix (2u, 2u), false);
    Trainable bv = make_view (bias, Shape::vector (2u), false);
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);

    REQUIRE (linear_forward (pool, spec, xv, wv, bv, ov, pre) == Status::OK);

    for (uint32_t b = 0; b < 2u; ++b)
        for (uint32_t o = 0; o < 2u; ++o)
            {
                const int32_t code = ref_linear_code (
                    ref_accum (x, w, 2u, b, o), 7u, 32768, 0, true);
                REQUIRE (static_cast<int32_t> (pre[b * 2u + o]) == code);
                REQUIRE (static_cast<int32_t> (out.codes[b * 2u + o])
                         == ref_relu_code (code));
            }
}

TEST_CASE ("linear_forward requires a pre-activation buffer when act is set")
{
    Tensor x, w, bias, out;
    fill_codes (x, { 1, 2 });
    fill_codes (w, { 1, 2 });
    fill_codes (bias, { 0 });
    fill_codes (out, { 0 });

    LinearSpec spec;
    spec._M_batch = 1u;
    spec._M_in_features = 2u;
    spec._M_out_features = 1u;
    spec._M_act = Act::RELU;

    AlignedArena<1024> arena;
    Pool pool = arena.pool ();
    Trainable xv = make_view (x, Shape::matrix (1u, 2u), false);
    Trainable wv = make_view (w, Shape::matrix (1u, 2u), false);
    Trainable bv = make_view (bias, Shape::vector (1u), false);
    Trainable ov = make_view (out, Shape::matrix (1u, 1u), true);

    REQUIRE (linear_forward (pool, spec, xv, wv, bv, ov, nullptr)
             == Status::BAD_LAYOUT);
}

TEST_CASE ("linear_forward operand validation surfaces Status codes")
{
    Tensor x, w, bias, out;
    fill_codes (x, { 1, 2, 3, 4 });
    fill_codes (w, { 1, 2, 3, 4, 5, 6 });
    fill_codes (bias, { 1, 2 });
    fill_codes (out, { 0, 0 });

    LinearSpec spec;
    spec._M_batch = 2u;
    spec._M_in_features = 2u;
    spec._M_out_features = 2u;

    AlignedArena<1024> arena;
    Pool pool = arena.pool ();

    SECTION ("null code pointer")
    {
        Trainable xv;
        xv._M_code = nullptr;
        xv._M_count = 4u;
        Trainable wv = make_view (w, Shape::matrix (2u, 2u), false);
        Trainable bv = make_view (bias, Shape::vector (2u), false);
        Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
        REQUIRE (linear_forward (pool, spec, xv, wv, bv, ov, nullptr)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("wrong element count")
    {
        Tensor bad_x;
        fill_codes (bad_x, { 1, 2, 3 });
        Trainable xv = make_view (bad_x, Shape::vector (3u), false);
        Trainable wv = make_view (w, Shape::matrix (2u, 2u), false);
        Trainable bv = make_view (bias, Shape::vector (2u), false);
        Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
        REQUIRE (linear_forward (pool, spec, xv, wv, bv, ov, nullptr)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("rank-1 accepted for a [1, K] expected shape")
    {
        LinearSpec s1;
        s1._M_batch = 1u;
        s1._M_in_features = 2u;
        s1._M_out_features = 2u;
        Tensor x1, w1, b1, o1;
        fill_codes (x1, { 3, 4 });
        fill_codes (w1, { 1, 2, 3, 4 });
        fill_codes (b1, { 0, 0 });
        fill_codes (o1, { 0, 0 });
        Trainable xv = make_view (x1, Shape::vector (2u), false);
        Trainable wv = make_view (w1, Shape::matrix (2u, 2u), false);
        Trainable bv = make_view (b1, Shape::vector (2u), false);
        Trainable ov = make_view (o1, Shape::vector (2u), true);
        REQUIRE (linear_forward (pool, s1, xv, wv, bv, ov, nullptr) == Status::OK);
        for (uint32_t o = 0; o < 2u; ++o)
            REQUIRE (static_cast<int32_t> (o1.codes[o])
                     == ref_linear_code (ref_accum (x1, w1, 2u, 0u, o), 7u, 32768,
                                         0, true));
    }
    SECTION ("invalid spec short-circuits before touching operands")
    {
        LinearSpec bad;
        bad._M_batch = 0u;
        Trainable xv = make_view (x, Shape::matrix (2u, 2u), false);
        Trainable wv = make_view (w, Shape::matrix (2u, 2u), false);
        Trainable bv = make_view (bias, Shape::vector (2u), false);
        Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
        REQUIRE (linear_forward (pool, bad, xv, wv, bv, ov, nullptr)
                 == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("linear_check_operand accepts documented shapes")
{
    Tensor v;
    fill_codes (v, { 1, 2, 3 });
    Trainable rank1 = make_view (v, Shape::vector (3u), false);
    REQUIRE (linear_check_operand (rank1, Shape::matrix (1u, 3u), 3u) == Status::OK);
    REQUIRE (linear_check_operand (rank1, Shape::vector (3u), 3u) == Status::OK);

    Trainable nullv;
    nullv._M_count = 3u;
    REQUIRE (linear_check_operand (nullv, Shape::vector (3u), 3u) == Status::BAD_LAYOUT);

    fill_codes (v, { 1, 2, 3, 4 });
    Trainable wrong = make_view (v, Shape::vector (4u), false);
    REQUIRE (linear_check_operand (wrong, Shape::vector (3u), 3u) == Status::BAD_LAYOUT);
}

TEST_CASE ("linear_backward produces input and weight gradients (no act)")
{
    Tensor x, w, bias, out;
    fill_codes (x, { 10, -20, 30, 7 }); /* batch 2 x in 2 */
    fill_codes (w, { 1, 2, 3, -4 });    /* out 2 x in 2 */
    fill_codes (bias, { 0, 0 });
    fill_codes (out, { 0, 0, 0, 0 });

    LinearSpec spec;
    spec._M_batch = 2u;
    spec._M_in_features = 2u;
    spec._M_out_features = 2u;

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);

    Trainable xv = make_view (x, Shape::matrix (2u, 2u), true);
    Trainable wv = make_view (w, Shape::matrix (2u, 2u), true);
    Trainable bv = make_view (bias, Shape::vector (2u), true);
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
    REQUIRE (linear_run (tape, spec, xv, wv, bv, ov) == Status::OK);

    /* hand-authored dY lanes (Q15); the forward zeroed them, set them now */
    const int16_t dy[4] = { 500, -700, 300, 1200 };
    for (uint32_t i = 0; i < 4u; ++i)
        ov.set_grad (i, dy[i]);
    xv.zero_grad ();
    wv.zero_grad ();
    bv.zero_grad ();

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;

    REQUIRE (linear_backward (ov, ov, ctx) == Status::OK);

    /* dX[b][i] = narrow(sum_o dY[b][o] * q15(W[o][i])) */
    for (uint32_t b = 0; b < 2u; ++b)
        for (uint32_t i = 0; i < 2u; ++i)
            {
                int64_t sum = 0;
                for (uint32_t o = 0; o < 2u; ++o)
                    sum += static_cast<int64_t> (dy[b * 2u + o])
                           * ref_code_q15 (static_cast<int32_t> (w.codes[o * 2u + i]));
                REQUIRE (static_cast<int32_t> (x.grads[b * 2u + i])
                         == ref_narrow_sum (sum));
            }
    /* dW[o][i] = narrow(sum_b dY[b][o] * q15(X[b][i])) */
    for (uint32_t o = 0; o < 2u; ++o)
        for (uint32_t i = 0; i < 2u; ++i)
            {
                int64_t sum = 0;
                for (uint32_t b = 0; b < 2u; ++b)
                    sum += static_cast<int64_t> (dy[b * 2u + o])
                           * ref_code_q15 (static_cast<int32_t> (x.codes[b * 2u + i]));
                REQUIRE (static_cast<int32_t> (w.grads[o * 2u + i])
                         == ref_narrow_sum (sum));
            }
    /* db[o] = clamp_sum(sum_b dY[b][o]) */
    for (uint32_t o = 0; o < 2u; ++o)
        {
            int64_t sum = 0;
            for (uint32_t b = 0; b < 2u; ++b)
                sum += dy[b * 2u + o];
            REQUIRE (static_cast<int32_t> (bias.grads[o]) == ref_clamp_sum (sum));
        }
}

TEST_CASE ("linear_backward with a frozen weight skips the input copy but keeps dW")
{
    Tensor x, w, bias, out;
    fill_codes (x, { 6, -6, 12, 3 });
    fill_codes (w, { 2, 4, -1, 5 });
    fill_codes (bias, { 0, 0 });
    fill_codes (out, { 0, 0, 0, 0 });

    LinearSpec spec;
    spec._M_batch = 2u;
    spec._M_in_features = 2u;
    spec._M_out_features = 2u;

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);

    Trainable xv = make_view (x, Shape::matrix (2u, 2u), true);
    Trainable wv = make_view (w, Shape::matrix (2u, 2u), false); /* frozen */
    Trainable bv = make_view (bias, Shape::vector (2u), false);
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
    REQUIRE (linear_run (tape, spec, xv, wv, bv, ov) == Status::OK);

    /* the frozen-weight node must not have carved the input copy */
    REQUIRE (nodes[0]._M_saved[_S_linear_input].is_null ());

    const int16_t dy[4] = { 900, -400, 200, 600 };
    for (uint32_t i = 0; i < 4u; ++i)
        ov.set_grad (i, dy[i]);
    xv.zero_grad ();

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;
    REQUIRE (linear_backward (ov, ov, ctx) == Status::OK);

    /* dX still produced from the live input; dW is null because weight frozen */
    for (uint32_t b = 0; b < 2u; ++b)
        for (uint32_t i = 0; i < 2u; ++i)
            {
                int64_t sum = 0;
                for (uint32_t o = 0; o < 2u; ++o)
                    sum += static_cast<int64_t> (dy[b * 2u + o])
                           * ref_code_q15 (static_cast<int32_t> (w.codes[o * 2u + i]));
                REQUIRE (static_cast<int32_t> (x.grads[b * 2u + i])
                         == ref_narrow_sum (sum));
            }
    for (int16_t g : w.grads)
        REQUIRE (g == 0);
}

TEST_CASE ("linear_backward routes through ReLU before the matrix step")
{
    Tensor x, w, bias, out;
    fill_codes (x, { 40, -40 });
    fill_codes (w, { 60, 60 }); /* single output unit */
    fill_codes (bias, { 0 });
    fill_codes (out, { 0 });

    LinearSpec spec;
    spec._M_batch = 1u;
    spec._M_in_features = 2u;
    spec._M_out_features = 1u;
    spec._M_requant_shift = 7u;
    spec._M_act = Act::RELU;

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);

    Trainable xv = make_view (x, Shape::matrix (1u, 2u), true);
    Trainable wv = make_view (w, Shape::matrix (1u, 2u), true);
    Trainable bv = make_view (bias, Shape::vector (1u), true);
    Trainable ov = make_view (out, Shape::matrix (1u, 1u), true);
    REQUIRE (linear_run (tape, spec, xv, wv, bv, ov) == Status::OK);

    /* acc = 40*60 + -40*60 = 0 -> code 0, relu keeps it (>=0) */
    const int8_t saved_pre = (reinterpret_cast<int8_t *> (
        pool.at (nodes[0]._M_saved[_S_linear_pre]._M_offset)))[0];
    REQUIRE (static_cast<int32_t> (saved_pre) == 0);

    const int16_t dy[1] = { 800 };
    ov.set_grad (0, dy[0]);
    xv.zero_grad ();
    wv.zero_grad ();
    bv.zero_grad ();

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;
    REQUIRE (linear_backward (ov, ov, ctx) == Status::OK);

    /* pre==0 >=0 keeps the full gradient: routed dY == 800 */
    for (uint32_t i = 0; i < 2u; ++i)
        {
            int64_t sum = static_cast<int64_t> (800)
                          * ref_code_q15 (static_cast<int32_t> (w.codes[i]));
            REQUIRE (static_cast<int32_t> (x.grads[i]) == ref_narrow_sum (sum));
        }
    REQUIRE (static_cast<int32_t> (bias.grads[0]) == 800);
}

TEST_CASE ("linear_backward saturates a bias row sum beyond a Q15 lane")
{
    /* out=1, batch=6, dY lanes all large: sum over batch exceeds 32767 */
    Tensor x, w, bias, out;
    fill_codes (x, { 1, 1, 1, 1, 1, 1 });
    fill_codes (w, { 1 });
    fill_codes (bias, { 0 });
    fill_codes (out, { 0, 0, 0, 0, 0, 0 });

    LinearSpec spec;
    spec._M_batch = 6u;
    spec._M_in_features = 1u;
    spec._M_out_features = 1u;
    spec._M_act = Act::NONE;

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);

    Trainable xv = make_view (x, Shape::matrix (6u, 1u), false);
    Trainable wv = make_view (w, Shape::matrix (1u, 1u), true);
    Trainable bv = make_view (bias, Shape::vector (1u), true);
    Trainable ov = make_view (out, Shape::matrix (6u, 1u), true);
    REQUIRE (linear_run (tape, spec, xv, wv, bv, ov) == Status::OK);

    const int32_t per_lane = 20000;
    for (uint32_t b = 0; b < 6u; ++b)
        ov.set_grad (b, per_lane);
    bv.zero_grad ();

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;
    REQUIRE (linear_backward (ov, ov, ctx) == Status::OK);

    /* channel sum = clamp(6*20000) = 32767, bias lane then add_grad(32767) */
    REQUIRE (static_cast<int32_t> (bias.grads[0]) == 32767);
    REQUIRE (ctx._M_saturation_events > 0u);
    REQUIRE (static_cast<int64_t> (per_lane) * 6 > 32767);
}

TEST_CASE ("linear_backward without a bias carries two operands and writes no db")
{
    Tensor x, w, bias, out;
    fill_codes (x, { 4, -2, 5, 1, -3, 6 });  /* batch 3 x in 2 */
    fill_codes (w, { 3, -1, 2, 5 });          /* out 2 x in 2 */
    fill_codes (bias, { 7, 7 });              /* live but not an operand */
    fill_codes (out, { 0, 0, 0, 0, 0, 0 });

    LinearSpec spec;
    spec._M_batch = 3u;
    spec._M_in_features = 2u;
    spec._M_out_features = 2u;
    spec._M_use_bias = 0u;

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);

    Trainable xv = make_view (x, Shape::matrix (3u, 2u), true);
    Trainable wv = make_view (w, Shape::matrix (2u, 2u), true);
    Trainable bv = make_view (bias, Shape::vector (2u), true);
    Trainable ov = make_view (out, Shape::matrix (3u, 2u), true);
    REQUIRE (linear_run (tape, spec, xv, wv, bv, ov) == Status::OK);

    Node *const node = tape.node (0u);
    REQUIRE (node != nullptr);
    /* the bias is not recorded at all, so a backward cannot read a third operand */
    REQUIRE (node->_M_operand_count == 2u);
    REQUIRE (node->_M_saved_count == _S_max_operands);

    const int16_t dy[6] = { 900, -1500, 400, 800, -200, 1300 };
    for (uint32_t i = 0; i < 6u; ++i)
        ov.set_grad (i, dy[i]);
    xv.zero_grad ();
    wv.zero_grad ();
    bv.zero_grad ();

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = node->_M_operands;
    ctx._M_operand_count = node->_M_operand_count;
    ctx._M_saved = node->_M_saved;
    ctx._M_saved_count = node->_M_saved_count;
    REQUIRE (linear_backward (ov, ov, ctx) == Status::OK);

    for (uint32_t b = 0; b < 3u; ++b)
        for (uint32_t i = 0; i < 2u; ++i)
            {
                int64_t sum = 0;
                for (uint32_t o = 0; o < 2u; ++o)
                    sum += static_cast<int64_t> (dy[b * 2u + o])
                           * ref_code_q15 (
                                 static_cast<int32_t> (w.codes[o * 2u + i]));
                REQUIRE (static_cast<int32_t> (x.grads[b * 2u + i])
                         == ref_narrow_sum (sum));
            }
    for (uint32_t o = 0; o < 2u; ++o)
        for (uint32_t i = 0; i < 2u; ++i)
            {
                int64_t sum = 0;
                for (uint32_t b = 0; b < 3u; ++b)
                    sum += static_cast<int64_t> (dy[b * 2u + o])
                           * ref_code_q15 (
                                 static_cast<int32_t> (x.codes[b * 2u + i]));
                REQUIRE (static_cast<int32_t> (w.grads[o * 2u + i])
                         == ref_narrow_sum (sum));
            }
    /* no bias operand means no db: the live buffer keeps the forward's zeroes */
    for (int16_t g : bias.grads)
        REQUIRE (g == 0);
}

TEST_CASE ("linear_run saves the spec the backward reads back")
{
    Tensor x, w, bias, out;
    fill_codes (x, { 1, 2, 3, 4 });
    fill_codes (w, { 1, 2, 3, 4, 5, 6 });
    fill_codes (bias, { 0, 0, 0 });
    fill_codes (out, { 0, 0, 0, 0, 0, 0 });

    LinearSpec spec;
    spec._M_batch = 2u;
    spec._M_in_features = 2u;
    spec._M_out_features = 3u;
    spec._M_requant_shift = 3u;
    spec._M_requant_mult_q15 = 20000;
    spec._M_op_id = 77u;

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);

    Trainable xv = make_view (x, Shape::matrix (2u, 2u), true);
    Trainable wv = make_view (w, Shape::matrix (3u, 2u), true);
    Trainable bv = make_view (bias, Shape::vector (3u), true);
    Trainable ov = make_view (out, Shape::matrix (2u, 3u), true);
    REQUIRE (linear_run (tape, spec, xv, wv, bv, ov) == Status::OK);

    Node *const node = tape.node (0u);
    REQUIRE (node != nullptr);
    REQUIRE (node->_M_op_id == 77u);
    REQUIRE (node->_M_operand_count == 3u);

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = node->_M_operands;
    ctx._M_operand_count = node->_M_operand_count;
    ctx._M_saved = node->_M_saved;
    ctx._M_saved_count = node->_M_saved_count;

    /* slot order is the documented one: input copy, pre-activation, scratch,
       spec -- and a slot the call did not carve stays a null handle. */
    REQUIRE_FALSE (node->_M_saved[_S_linear_input].is_null ());
    REQUIRE (node->_M_saved[_S_linear_pre].is_null ());
    REQUIRE_FALSE (node->_M_saved[_S_linear_scratch].is_null ());
    REQUIRE_FALSE (node->_M_saved[_S_linear_spec].is_null ());

    const LinearSpec *const restored = ctx.saved_as<LinearSpec> (_S_linear_spec);
    REQUIRE (restored != nullptr);
    REQUIRE (restored->_M_batch == spec._M_batch);
    REQUIRE (restored->_M_in_features == spec._M_in_features);
    REQUIRE (restored->_M_out_features == spec._M_out_features);
    REQUIRE (restored->_M_use_bias == 1u);
    REQUIRE (restored->_M_requant_shift == 3u);
    REQUIRE (restored->_M_requant_mult_q15 == 20000);
    REQUIRE (restored->_M_op_id == 77u);
    REQUIRE (restored->_M_act == Act::NONE);

    /* the saved input copy is the pre-forward code stream, so a caller that
       recycles the live buffer cannot corrupt dW */
    const int8_t *const saved_input
        = ctx.saved_as<int8_t> (_S_linear_input);
    REQUIRE (saved_input != nullptr);
    for (uint32_t i = 0; i < spec.input_elements (); ++i)
        REQUIRE (saved_input[i] == x.codes[i]);

    /* an activation-less layer still reserves the whole scratch slab, and a
       Pool slab is rounded up to the pool's 4-byte alignment */
    const int16_t *const scratch = ctx.saved_as<int16_t> (_S_linear_scratch);
    REQUIRE (scratch != nullptr);
    const uint32_t scratch_elements = spec.output_elements () + spec._M_out_features;
    const uint32_t scratch_bytes
        = ((2u * scratch_elements + _S_pool_alignment - 1u) / _S_pool_alignment)
          * _S_pool_alignment;
    REQUIRE (node->_M_saved[_S_linear_scratch]._M_bytes == scratch_bytes);
    REQUIRE (scratch_bytes >= 2u * scratch_elements);

    /* and the layer is usable: a backward through this context succeeds */
    for (uint32_t i = 0; i < 6u; ++i)
        ov.set_grad (i, static_cast<int16_t> (100 + i));
    xv.zero_grad ();
    wv.zero_grad ();
    bv.zero_grad ();
    REQUIRE (linear_backward (ov, ov, ctx) == Status::OK);
    /* db[o] sums the channel over the batch: (100 + o) + (103 + o) */
    for (uint32_t o = 0; o < spec._M_out_features; ++o)
        REQUIRE (static_cast<int32_t> (bias.grads[o])
                 == ref_clamp_sum (static_cast<int64_t> (100 + o) + (103 + o)));
}

TEST_CASE ("matrix_codes reads a transposed weight without a copy")
{
    /* W is stored [out, in] row major; the forward needs the logical [in, out]
       view, which the header builds by swapping strides rather than copying. */
    const int8_t weight[6] = { 1, 2, 3, -4, 5, -6 }; /* out 2 x in 3 */
    const uint32_t out_count = 2u;
    const uint32_t in_count = 3u;

    const Matrix row_major = matrix_codes (weight, out_count, in_count, false);
    const Matrix transposed = matrix_codes (weight, in_count, out_count, true);

    for (uint32_t o = 0; o < out_count; ++o)
        for (uint32_t i = 0; i < in_count; ++i)
            {
                REQUIRE (row_major.code_at (o, i)
                         == static_cast<int32_t> (weight[o * in_count + i]));
                /* the same element, named by the transposed logical shape */
                REQUIRE (transposed.code_at (i, o)
                         == static_cast<int32_t> (weight[o * in_count + i]));
            }
    REQUIRE (row_major._M_row_stride == in_count);
    REQUIRE (row_major._M_col_stride == 1u);
    /* a transposed view keeps the element size and swaps the strides: row
       stride 1, column stride equal to the stored row length */
    REQUIRE (transposed._M_row_stride == 1u);
    REQUIRE (transposed._M_col_stride == in_count);
    REQUIRE (row_major.is_code ());
    REQUIRE_FALSE (row_major.is_lane ());

    /* C = X * W^T with the transposed view is the affine forward's dot pattern */
    Tensor x, w, bias, out;
    fill_codes (x, { 7, -3, 2, 5, -1, 4 }); /* batch 2 x in 3 */
    fill_codes (w, { 1, 2, 3, -4, 5, -6 }); /* out 2 x in 3 */
    fill_codes (bias, { 0, 0 });
    fill_codes (out, { 0, 0, 0, 0 });

    AlignedArena<64> arena;
    Pool pool = arena.pool ();
    const uint32_t batch = 2u;
    int32_t *grid = nullptr;
    REQUIRE (pool.carve_typed (batch * out_count, grid) == Status::OK);
    const Accum accumulator{ grid, batch * out_count };
    accumulator.zero ();
    const Matrix a = matrix_codes (x.codes.data (), batch, in_count, false);
    REQUIRE (matrix_inner_ok (a, transposed) == Status::OK);
    REQUIRE (gemm_codes (a, transposed, accumulator) == Status::OK);
    for (uint32_t b = 0; b < batch; ++b)
        for (uint32_t o = 0; o < out_count; ++o)
            REQUIRE (grid[b * out_count + o]
                     == ref_accum (x, w, in_count, b, o));

    /* a mismatched inner dimension is a shape error, not a partial product */
    const Matrix short_a = matrix_codes (x.codes.data (), batch, 2u, false);
    REQUIRE (gemm_codes (short_a, transposed, accumulator) == Status::BAD_LAYOUT);
    /* an un-zeroed destination of the wrong size is rejected too */
    const Accum tiny{ grid, 1u };
    REQUIRE (gemm_codes (a, transposed, tiny) == Status::BAD_LAYOUT);
}

/* ================================================================== */
/* train_linear.hpp: the tape-driven walk                              */
/* ================================================================== */

TEST_CASE ("Tape::backward propagates through two chained linear nodes")
{
    Tensor x, w1, b1, h, w2, b2, y, loss;
    fill_codes (x, { 3, -5, 2, 4 });
    fill_codes (w1, { 1, 2, -3, 4, 5, -6 });           /* 3 out x 2 in */
    fill_codes (b1, { 0, 0, 0 });
    fill_codes (h, { 0, 0, 0, 0, 0, 0 });               /* batch 2 x 3 */
    fill_codes (w2, { 2, -1, 3, 0, 4, -2, 1, 5, -3 }); /* 3 out x 3 in */
    fill_codes (b2, { 0, 0, 0 });
    fill_codes (y, { 0, 0, 0, 0, 0, 0 });
    fill_codes (loss, { 0 });

    LinearSpec first;
    first._M_batch = 2u;
    first._M_in_features = 2u;
    first._M_out_features = 3u;
    first._M_requant_shift = 0u;

    LinearSpec second;
    second._M_batch = 2u;
    second._M_in_features = 3u;
    second._M_out_features = 3u;
    second._M_requant_shift = 0u;

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[4];
    Tape tape (nodes, 4u, pool);
    REQUIRE (tape.begin () == Status::OK);

    Trainable xv = make_view (x, Shape::matrix (2u, 2u), true);
    Trainable w1v = make_view (w1, Shape::matrix (3u, 2u), true);
    Trainable b1v = make_view (b1, Shape::vector (3u), true);
    Trainable hv = make_view (h, Shape::matrix (2u, 3u), true);
    Trainable w2v = make_view (w2, Shape::matrix (3u, 3u), true);
    Trainable b2v = make_view (b2, Shape::vector (3u), true);
    Trainable yv = make_view (y, Shape::matrix (2u, 3u), true);

    REQUIRE (linear_run (tape, first, xv, w1v, b1v, hv) == Status::OK);
    REQUIRE (linear_run (tape, second, hv, w2v, b2v, yv) == Status::OK);
    REQUIRE (tape.node_count () == 2u);

    const int16_t dy[6] = { 1000, -2000, 500, 700, -300, 900 };
    for (uint32_t i = 0; i < 6u; ++i)
        yv.set_grad (i, dy[i]);
    xv.zero_grad ();
    w1v.zero_grad ();
    b1v.zero_grad ();
    hv.zero_grad ();
    w2v.zero_grad ();
    b2v.zero_grad ();

    Trainable lv = make_view (loss, Shape::scalar (), true);
    REQUIRE (tape.backward (lv) == Status::OK);
    /* the seed is the loss scale clamped into a Q15 lane */
    REQUIRE (static_cast<int32_t> (loss.grads[0]) == 32767);

    /* layer 2 first: dh[b][i] = narrow(sum_o dy[b][o] * q15(W2[o][i])) */
    int32_t dh[6];
    for (uint32_t b = 0; b < 2u; ++b)
        for (uint32_t i = 0; i < 3u; ++i)
            {
                int64_t sum = 0;
                for (uint32_t o = 0; o < 3u; ++o)
                    sum += static_cast<int64_t> (dy[b * 3u + o])
                           * ref_code_q15 (
                                 static_cast<int32_t> (w2.codes[o * 3u + i]));
                dh[b * 3u + i] = ref_narrow_sum (sum);
                REQUIRE (static_cast<int32_t> (h.grads[b * 3u + i])
                         == dh[b * 3u + i]);
            }
    for (uint32_t o = 0; o < 3u; ++o)
        {
            int64_t total = 0;
            for (uint32_t b = 0; b < 2u; ++b)
                total += dy[b * 3u + o];
            REQUIRE (static_cast<int32_t> (b2.grads[o]) == ref_clamp_sum (total));
        }

    /* then layer 1, reading the chained lanes the walk just wrote */
    for (uint32_t b = 0; b < 2u; ++b)
        for (uint32_t i = 0; i < 2u; ++i)
            {
                int64_t sum = 0;
                for (uint32_t o = 0; o < 3u; ++o)
                    sum += static_cast<int64_t> (dh[b * 3u + o])
                           * ref_code_q15 (
                                 static_cast<int32_t> (w1.codes[o * 2u + i]));
                REQUIRE (static_cast<int32_t> (x.grads[b * 2u + i])
                         == ref_narrow_sum (sum));
            }
    for (uint32_t o = 0; o < 3u; ++o)
        for (uint32_t i = 0; i < 2u; ++i)
            {
                int64_t sum = 0;
                for (uint32_t b = 0; b < 2u; ++b)
                    sum += static_cast<int64_t> (dh[b * 3u + o])
                           * ref_code_q15 (
                                 static_cast<int32_t> (x.codes[b * 2u + i]));
                REQUIRE (static_cast<int32_t> (w1.grads[o * 2u + i])
                         == ref_narrow_sum (sum));
            }
    for (uint32_t o = 0; o < 3u; ++o)
        {
            int64_t total = 0;
            for (uint32_t b = 0; b < 2u; ++b)
                total += dh[b * 3u + o];
            REQUIRE (static_cast<int32_t> (b1.grads[o]) == ref_clamp_sum (total));
        }

    const TapeStats stats = tape.stats ();
    REQUIRE (stats._M_nodes == 2u);
    REQUIRE (stats._M_capacity == 4u);
    REQUIRE (stats._M_saved_bytes > 0u);
    REQUIRE (stats._M_saved_bytes <= _S_train_pool_bytes);
}

TEST_CASE ("matmul_backward_q15 fills only the operand gradients that exist")
{
    Tensor x, w, dy;
    fill_codes (x, { 3, -2, 5, 1, -4, 6 });  /* batch 3 x in 2 */
    fill_codes (w, { 2, -1, 4, 0, 3, -5 });   /* out 3 x in 2, inference order */
    fill_codes (dy, { 0, 0, 0, 0, 0, 0, 0, 0, 0 });

    const uint32_t batch = 3u;
    const uint32_t in_features = 2u;
    const uint32_t out_features = 3u;
    const int16_t lanes[9] = { 900, -1200, 300, 500, 700, -800, -100, 400, 1100 };

    Trainable xv = make_view (x, Shape::matrix (batch, in_features), true);
    Trainable wv = make_view (w, Shape::matrix (out_features, in_features), true);
    Trainable dyv = make_view (dy, Shape::matrix (batch, out_features), true);
    for (uint32_t i = 0u; i < 9u; ++i)
        dyv.set_grad (i, lanes[i]);

    /* dX and dW, each written the way the header documents */
    int32_t want_dx[6];
    int32_t want_dw[6];
    for (uint32_t b = 0; b < batch; ++b)
        for (uint32_t i = 0; i < in_features; ++i)
            {
                int64_t sum = 0;
                for (uint32_t o = 0; o < out_features; ++o)
                    sum += static_cast<int64_t> (lanes[b * out_features + o])
                           * ref_code_q15 (
                                 static_cast<int32_t> (w.codes[o * in_features + i]));
                want_dx[b * in_features + i] = ref_narrow_sum (sum);
            }
    for (uint32_t o = 0; o < out_features; ++o)
        for (uint32_t i = 0; i < in_features; ++i)
            {
                int64_t sum = 0;
                for (uint32_t b = 0; b < batch; ++b)
                    sum += static_cast<int64_t> (lanes[b * out_features + o])
                           * ref_code_q15 (
                                 static_cast<int32_t> (x.codes[b * in_features + i]));
                want_dw[o * in_features + i] = ref_narrow_sum (sum);
            }

    SECTION ("both trainable operands collect their gradient")
    {
        xv.zero_grad ();
        wv.zero_grad ();
        uint32_t saturated = 0u;
        REQUIRE (matmul_backward_q15 (xv, wv, dyv, batch, in_features,
                                      out_features, saturated)
                 == Status::OK);
        for (uint32_t i = 0; i < 6u; ++i)
            {
                REQUIRE (static_cast<int32_t> (x.grads[i]) == want_dx[i]);
                REQUIRE (static_cast<int32_t> (w.grads[i]) == want_dw[i]);
            }
        /* accumulating into the same lanes doubles them, still unsaturated */
        saturated = 0u;
        REQUIRE (matmul_backward_q15 (xv, wv, dyv, batch, in_features,
                                      out_features, saturated)
                 == Status::OK);
        for (uint32_t i = 0; i < 6u; ++i)
            {
                REQUIRE (static_cast<int32_t> (x.grads[i]) == 2 * want_dx[i]);
                REQUIRE (static_cast<int32_t> (w.grads[i]) == 2 * want_dw[i]);
            }
        REQUIRE (saturated == 0u);
    }
    SECTION ("a frozen weight skips only the dW loop")
    {
        xv.zero_grad ();
        wv.zero_grad ();
        Trainable frozen_w = make_view (w, Shape::matrix (out_features, in_features),
                                        false);
        uint32_t saturated = 0u;
        REQUIRE (matmul_backward_q15 (xv, frozen_w, dyv, batch, in_features,
                                      out_features, saturated)
                 == Status::OK);
        for (uint32_t i = 0; i < 6u; ++i)
            {
                REQUIRE (static_cast<int32_t> (x.grads[i]) == want_dx[i]);
                REQUIRE (static_cast<int32_t> (w.grads[i]) == 0);
            }
    }
    SECTION ("a frozen input skips only the dX loop")
    {
        xv.zero_grad ();
        wv.zero_grad ();
        Trainable frozen_x = make_view (x, Shape::matrix (batch, in_features), false);
        uint32_t saturated = 0u;
        REQUIRE (matmul_backward_q15 (frozen_x, wv, dyv, batch, in_features,
                                      out_features, saturated)
                 == Status::OK);
        for (uint32_t i = 0; i < 6u; ++i)
            {
                REQUIRE (static_cast<int32_t> (x.grads[i]) == 0);
                REQUIRE (static_cast<int32_t> (w.grads[i]) == want_dw[i]);
            }
    }
    SECTION ("shape and gradient-buffer mismatches are BAD_LAYOUT")
    {
        uint32_t saturated = 0u;
        Trainable short_x = make_view (x, Shape::vector (5u), true);
        REQUIRE (matmul_backward_q15 (short_x, wv, dyv, batch, in_features,
                                      out_features, saturated)
                 == Status::BAD_LAYOUT);
        Trainable plain_dy = make_view (dy, Shape::matrix (batch, out_features),
                                        false);
        REQUIRE (matmul_backward_q15 (xv, wv, plain_dy, batch, in_features,
                                      out_features, saturated)
                 == Status::BAD_LAYOUT);
        /* the wrong batch for the same buffers is a shape error too */
        REQUIRE (matmul_backward_q15 (xv, wv, dyv, 2u, in_features, out_features,
                                      saturated)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("a lane that cannot hold the sum clamps and is counted")
    {
        /* drive every dY lane near the rail, then start two lanes already at
           32000: those two accumulations clamp and the counter says so */
        for (uint32_t i = 0; i < 9u; ++i)
            dyv.set_grad (i, 30000);
        xv.zero_grad ();
        wv.zero_grad ();
        xv.set_grad (0, 32000);
        wv.set_grad (0, 32000);

        int64_t dx_total = 0;
        for (uint32_t o = 0; o < out_features; ++o)
            dx_total += 30000LL
                        * ref_code_q15 (
                              static_cast<int32_t> (w.codes[o * in_features]));
        int64_t dw_total = 0;
        for (uint32_t b = 0; b < batch; ++b)
            dw_total += 30000LL
                        * ref_code_q15 (
                              static_cast<int32_t> (x.codes[b * in_features]));
        REQUIRE (32000 + ref_narrow_sum (dx_total) > 32767);
        REQUIRE (32000 + ref_narrow_sum (dw_total) > 32767);

        uint32_t saturated = 0u;
        REQUIRE (matmul_backward_q15 (xv, wv, dyv, batch, in_features,
                                      out_features, saturated)
                 == Status::OK);
        REQUIRE (static_cast<int32_t> (x.grads[0])
                 == ref_clamp_q15 (32000 + ref_narrow_sum (dx_total)));
        REQUIRE (static_cast<int32_t> (w.grads[0])
                 == ref_clamp_q15 (32000 + ref_narrow_sum (dw_total)));
        REQUIRE (x.grads[0] == 32767);
        REQUIRE (w.grads[0] == 32767);
        REQUIRE (saturated == 2u);
    }
    SECTION ("narrow_sum_q15 rails an int64 accumulator into one Q15 lane")
    {
        const int64_t totals[]
            = { 0, 1, -1, static_cast<int64_t> (32767) << 15,
                (static_cast<int64_t> (32767) << 15) + (1 << 15),
                -(static_cast<int64_t> (32768) << 15),
                -(static_cast<int64_t> (32768) << 15) - (1 << 15),
                123456789012LL, -123456789012LL };
        for (const int64_t total : totals)
            {
                REQUIRE (narrow_sum_q15 (total) == ref_narrow_sum (total));
                const int32_t lane = narrow_sum_q15 (total);
                REQUIRE (lane <= 32767);
                REQUIRE (lane >= -32768);
            }
        /* a plain lane sum clamps without any shift */
        REQUIRE (clamp_sum_q15 (40000) == 32767);
        REQUIRE (clamp_sum_q15 (-40000) == -32768);
        REQUIRE (clamp_sum_q15 (500) == 500);
    }
}

TEST_CASE ("channel_backward reduces the batch axis for a bias gradient")
{
    /* the shape a linear layer hands it: outer = batch, channels = out,
       inner = 1, so the stride jumps a whole row */
    const uint32_t batch = 4u;
    const uint32_t channels = 3u;
    int16_t grad[12];
    for (uint32_t i = 0; i < 12u; ++i)
        grad[i] = static_cast<int16_t> (static_cast<int32_t> (i) * 311 - 600);
    int16_t destination[3] = { 0, 0, 0 };
    uint32_t saturated = 0u;

    channel_backward (grad, batch, channels, 1u, destination, saturated);
    for (uint32_t channel = 0; channel < channels; ++channel)
        {
            int64_t total = 0;
            for (uint32_t b = 0; b < batch; ++b)
                total += grad[b * channels + channel];
            REQUIRE (static_cast<int32_t> (destination[channel])
                     == ref_clamp_sum (total));
        }
    REQUIRE (saturated == 0u);

    /* a channel column whose sum leaves the rail clamps and reports itself */
    int16_t hot[6] = { 20000, -20000, 20000, -20000, 20000, -20000 };
    int16_t out2[2] = { 0, 0 };
    saturated = 0u;
    channel_backward (hot, 3u, 2u, 1u, out2, saturated);
    REQUIRE (static_cast<int32_t> (out2[0]) == ref_clamp_sum (60000));
    REQUIRE (static_cast<int32_t> (out2[1]) == ref_clamp_sum (-60000));
    REQUIRE (out2[0] == 32767);
    REQUIRE (out2[1] == -32768);
    REQUIRE (saturated == 2u);
}

TEST_CASE ("linear_backward validates the output shape it is handed")
{
    Tensor x, w, bias, out;
    fill_codes (x, { 1, 2, 3, 4 });
    fill_codes (w, { 1, 2, 3, 4 });
    fill_codes (bias, { 0, 0 });
    fill_codes (out, { 0, 0, 0, 0 });

    LinearSpec spec;
    spec._M_batch = 2u;
    spec._M_in_features = 2u;
    spec._M_out_features = 2u;

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);
    Trainable xv = make_view (x, Shape::matrix (2u, 2u), true);
    Trainable wv = make_view (w, Shape::matrix (2u, 2u), true);
    Trainable bv = make_view (bias, Shape::vector (2u), true);
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
    REQUIRE (linear_run (tape, spec, xv, wv, bv, ov) == Status::OK);

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;

    SECTION ("the same count spelled as a vector is a layout error")
    {
        /* the count matches, the rank does not: linear_backward compares the
           shape as well, so a [4] output cannot stand in for [2, 2] */
        Tensor flat;
        fill_codes (flat, { 0, 0, 0, 0 });
        Trainable flatv = make_view (flat, Shape::vector (4u), true);
        REQUIRE (linear_backward (flatv, flatv, ctx) == Status::BAD_LAYOUT);
    }
    SECTION ("a missing third operand is a layout error")
    {
        GradContext short_ctx = ctx;
        short_ctx._M_operand_count = 2u; /* bias is demanded by the spec */
        REQUIRE (linear_backward (ov, ov, short_ctx) == Status::BAD_LAYOUT);
    }
    SECTION ("an output with no gradient lanes is a layout error")
    {
        Trainable frozen = make_view (out, Shape::matrix (2u, 2u), false);
        REQUIRE (linear_backward (frozen, frozen, ctx) == Status::BAD_LAYOUT);
    }
    SECTION ("a null operand table is a layout error")
    {
        GradContext broken = ctx;
        broken._M_operands = nullptr;
        REQUIRE (linear_backward (ov, ov, broken) == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("linear_run reports arena overflow and closed-tape as Status")
{
    Tensor x, w, bias, out;
    fill_codes (x, { 1, 2, 3, 4 });
    fill_codes (w, { 1, 2, 3, 4, 5, 6, 7, 8 });
    fill_codes (bias, { 0, 0 });
    fill_codes (out, { 0, 0 });

    LinearSpec spec;
    spec._M_batch = 2u;
    spec._M_in_features = 2u;
    spec._M_out_features = 2u;

    SECTION ("pool too small is ARENA_OVERFLOW")
    {
        AlignedArena<16> tiny;
        Pool pool = tiny.pool ();
        Node nodes[2];
        Tape tape (nodes, 2u, pool);
        REQUIRE (tape.begin () == Status::OK);
        Trainable xv = make_view (x, Shape::matrix (2u, 2u), true);
        Trainable wv = make_view (w, Shape::matrix (2u, 2u), true);
        Trainable bv = make_view (bias, Shape::vector (2u), true);
        Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
        REQUIRE (linear_run (tape, spec, xv, wv, bv, ov) == Status::ARENA_OVERFLOW);
    }
    SECTION ("tape not open is INTERNAL")
    {
        AlignedArena<_S_train_pool_bytes> arena;
        Pool pool = arena.pool ();
        Node nodes[2];
        Tape tape (nodes, 2u, pool);
        Trainable xv = make_view (x, Shape::matrix (2u, 2u), true);
        Trainable wv = make_view (w, Shape::matrix (2u, 2u), true);
        Trainable bv = make_view (bias, Shape::vector (2u), true);
        Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
        REQUIRE (linear_run (tape, spec, xv, wv, bv, ov) == Status::INTERNAL);
    }
    SECTION ("count mismatch is BAD_LAYOUT")
    {
        AlignedArena<_S_train_pool_bytes> arena;
        Pool pool = arena.pool ();
        Node nodes[2];
        Tape tape (nodes, 2u, pool);
        REQUIRE (tape.begin () == Status::OK);
        Trainable xv = make_view (x, Shape::matrix (2u, 2u), true);
        Trainable wv = make_view (w, Shape::matrix (2u, 2u), true);
        Trainable bv = make_view (bias, Shape::vector (2u), true);
        Tensor small;
        fill_codes (small, { 0 });
        Trainable ov = make_view (small, Shape::vector (1u), true);
        REQUIRE (linear_run (tape, spec, xv, wv, bv, ov) == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("Tape record enforces the four-operand cap and capacity")
{
    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();

    Tensor dummy;
    fill_codes (dummy, { 0 });
    Trainable t = make_view (dummy, Shape::vector (1u), true);
    Saved slots[_S_max_operands];
    Trainable operands[_S_max_operands + 1u];

    SECTION ("operand_count above the cap is BAD_LAYOUT")
    {
        Node nodes[4];
        Tape tape (nodes, 4u, pool);
        REQUIRE (tape.begin () == Status::OK);
        REQUIRE (tape.record (dummy_backward, 0u, t, operands, _S_max_operands + 1u,
                              slots, 1u)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("saved_count above the cap is BAD_LAYOUT")
    {
        Node nodes[4];
        Tape tape (nodes, 4u, pool);
        REQUIRE (tape.begin () == Status::OK);
        REQUIRE (tape.record (dummy_backward, 0u, t, operands, 1u, slots,
                              _S_max_operands + 1u)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("null backward is BAD_LAYOUT")
    {
        Node nodes[4];
        Tape tape (nodes, 4u, pool);
        REQUIRE (tape.begin () == Status::OK);
        REQUIRE (tape.record (nullptr, 0u, t, operands, 1u, slots, 1u)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("record before begin is INTERNAL")
    {
        Node nodes[4];
        Tape tape (nodes, 4u, pool);
        REQUIRE (tape.record (dummy_backward, 0u, t, operands, 1u, slots, 1u)
                 == Status::INTERNAL);
    }
    SECTION ("capacity overflow is ARENA_OVERFLOW")
    {
        Node nodes[1];
        Tape tape (nodes, 1u, pool);
        REQUIRE (tape.begin () == Status::OK);
        REQUIRE (tape.record (dummy_backward, 0u, t, operands, 1u, slots, 1u)
                 == Status::OK);
        REQUIRE (tape.record (dummy_backward, 0u, t, operands, 1u, slots, 1u)
                 == Status::ARENA_OVERFLOW);
        REQUIRE (tape.node_count () == 1u);
    }
}

TEST_CASE ("GradContext::saved_as respects slot size and bounds")
{
    Tensor x, w, bias, out;
    fill_codes (x, { 1, 2, 3, 4 });
    fill_codes (w, { 1, 2, 3, 4 });
    fill_codes (bias, { 0, 0 });
    fill_codes (out, { 0, 0, 0, 0 });

    LinearSpec spec;
    spec._M_batch = 2u;
    spec._M_in_features = 2u;
    spec._M_out_features = 2u;

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);
    Trainable xv = make_view (x, Shape::matrix (2u, 2u), true);
    Trainable wv = make_view (w, Shape::matrix (2u, 2u), true);
    Trainable bv = make_view (bias, Shape::vector (2u), true);
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
    REQUIRE (linear_run (tape, spec, xv, wv, bv, ov) == Status::OK);

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;

    /* the spec slot reads back as LinearSpec and round-trips the config */
    const LinearSpec *read = ctx.saved_as<LinearSpec> (_S_linear_spec);
    REQUIRE (read != nullptr);
    REQUIRE (read->_M_batch == 2u);
    REQUIRE (read->_M_in_features == 2u);
    REQUIRE (read->_M_out_features == 2u);

    /* a slot out of range returns null */
    REQUIRE (ctx.saved_as<LinearSpec> (_S_max_operands) == nullptr);
    REQUIRE (ctx.saved_as<int8_t> (99u) == nullptr);

    /* the input slot is a byte stream, so asking for int32 lanes fails the size
       check and yields null (bytes < count * sizeof(int32)). */
    REQUIRE (ctx.saved_as<int32_t> (_S_linear_input) == nullptr);
    REQUIRE (ctx.saved_as<int8_t> (_S_linear_input) != nullptr);

    /* the copy holds the pre-forward input codes */
    const int8_t *const saved_input
        = ctx.saved_as<int8_t> (_S_linear_input);
    for (uint32_t i = 0; i < spec.input_elements (); ++i)
        REQUIRE (saved_input[i] == x.codes[i]);

    /* an activation-less layer never carves the pre-activation slot, and a null
       handle must read as absent -- not as the pool's first slab */
    REQUIRE (nodes[0]._M_saved[_S_linear_pre].is_null ());
    REQUIRE (ctx.saved_as<int8_t> (_S_linear_pre) == nullptr);

    /* all three operands are recorded for a bias layer */
    REQUIRE (nodes[0]._M_operand_count == 3u);
    REQUIRE (nodes[0]._M_operands[0]._M_code == x.codes.data ());
    REQUIRE (nodes[0]._M_operands[1]._M_code == w.codes.data ());
    REQUIRE (nodes[0]._M_operands[2]._M_code == bias.codes.data ());
    REQUIRE (nodes[0]._M_op_id == spec._M_op_id);

    /* the size test is per element: the six-lane int16 scratch slab can be read
       as int16 lanes but never as six int32 lanes */
    REQUIRE (ctx.saved_as<int16_t> (_S_linear_scratch) != nullptr);
    REQUIRE (ctx.saved_as<int32_t> (_S_linear_scratch) == nullptr);
}

/* ================================================================== */
/* train_embedding.hpp                                                */
/* ================================================================== */

TEST_CASE ("EmbedSpec validate rejects bad geometry and padding")
{
    EmbedSpec spec;
    spec._M_rows = 3u;
    spec._M_tokens = 3u;
    spec._M_vocab = 4u;
    spec._M_dim = 2u;
    REQUIRE (spec.validate () == Status::OK);
    REQUIRE (spec.index_count () == 3u);
    REQUIRE (spec.table_elements () == 8u);
    REQUIRE (spec.output_elements () == 6u);
    REQUIRE_FALSE (spec.is_bag ());
    REQUIRE_FALSE (spec.has_padding ());

    SECTION ("zero rows/vocab/dim is BAD_LAYOUT")
    {
        EmbedSpec s = spec;
        s._M_rows = 0u;
        REQUIRE (s.validate () == Status::BAD_LAYOUT);
        s._M_rows = 3u;
        s._M_vocab = 0u;
        REQUIRE (s.validate () == Status::BAD_LAYOUT);
        s._M_vocab = 4u;
        s._M_dim = 0u;
        REQUIRE (s.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("lookup with tokens != rows is BAD_LAYOUT")
    {
        EmbedSpec s = spec;
        s._M_tokens = 5u;
        REQUIRE (s.validate () == Status::BAD_LAYOUT);
        /* a bag is the one form allowed to disagree: rows counts bags, tokens
           counts ids */
        s._M_bag = 1u;
        REQUIRE (s.validate () == Status::OK);
        REQUIRE (s.index_count () == 5u);
        s._M_bag = 0u;
        s._M_tokens = 3u;
        REQUIRE (s.index_count () == 3u);
    }
    SECTION ("padding id at or above vocab is BAD_LAYOUT")
    {
        EmbedSpec s = spec;
        s._M_padding_index = 4;
        REQUIRE (s.validate () == Status::BAD_LAYOUT);
        s._M_padding_index = 3;
        REQUIRE (s.validate () == Status::OK);
    }
    SECTION ("unaddressable table is OUT_OF_RANGE")
    {
        EmbedSpec s = spec;
        s._M_vocab = 256u;
        s._M_dim = 256u;
        REQUIRE (s.validate () == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("EmbedSpec bytes covers index and offset streams")
{
    EmbedSpec lookup;
    lookup._M_rows = 4u;
    lookup._M_tokens = 4u;
    lookup._M_vocab = 8u;
    lookup._M_dim = 3u;
    const uint32_t lb = lookup.bytes ();
    /* index (2*4=8, aligned 8) + spec (16) = 24 */
    REQUIRE (lb == 24u);

    EmbedSpec bag = lookup;
    bag._M_bag = 1u;
    bag._M_tokens = 10u;
    /* index (2*10=20) + offsets (2*(4+1)=10 -> aligned 12) + spec (16) = 48 */
    REQUIRE (bag.bytes () == 48u);
    REQUIRE (bag.index_count () == 10u);
}

TEST_CASE ("embedding_forward gathers rows exactly")
{
    /* vocab 4, dim 3 */
    Tensor table, out;
    fill_codes (table, { 1, 2, 3, -4, -5, -6, 7, 8, 9, 10, 11, 12 });
    fill_codes (out, { 0, 0, 0, 0, 0, 0 });

    EmbedSpec spec;
    spec._M_rows = 2u;
    spec._M_tokens = 2u;
    spec._M_vocab = 4u;
    spec._M_dim = 3u;
    const uint16_t index[] = { 2u, 0u };

    Trainable tv = make_view (table, Shape::matrix (4u, 3u), true);
    Trainable ov = make_view (out, Shape::matrix (2u, 3u), true);
    REQUIRE (embedding_forward (spec, index, nullptr, tv, ov) == Status::OK);

    /* output row 0 == table row 2, output row 1 == table row 0 */
    const int32_t expected[6] = { 7, 8, 9, 1, 2, 3 };
    for (uint32_t i = 0; i < 6u; ++i)
        {
            REQUIRE (static_cast<int32_t> (out.codes[i]) == expected[i]);
            REQUIRE (out.grads[i] == 0); /* forward zeroes lanes */
        }
}

TEST_CASE ("embedding_forward zeroes the padding row")
{
    Tensor table, out;
    fill_codes (table, { 5, 6, 7, 8, 9, 10, 11, 12, 13 });
    /* padding row (index 1) is intentionally dirty in the table */
    fill_codes (out, { 0, 0, 0, 0, 0, 0 });

    EmbedSpec spec;
    spec._M_rows = 2u;
    spec._M_tokens = 2u;
    spec._M_vocab = 3u;
    spec._M_dim = 3u;
    spec._M_padding_index = 1;
    const uint16_t index[] = { 0u, 1u };

    Trainable tv = make_view (table, Shape::matrix (3u, 3u), true);
    Trainable ov = make_view (out, Shape::matrix (2u, 3u), true);
    REQUIRE (embedding_forward (spec, index, nullptr, tv, ov) == Status::OK);

    REQUIRE (static_cast<int32_t> (out.codes[0]) == 5);
    REQUIRE (static_cast<int32_t> (out.codes[1]) == 6);
    REQUIRE (static_cast<int32_t> (out.codes[2]) == 7);
    for (uint32_t c = 0; c < 3u; ++c)
        REQUIRE (static_cast<int32_t> (out.codes[3 + c]) == 0);
}

TEST_CASE ("embedding_forward rejects an out-of-range id and empty layer")
{
    Tensor table, out;
    fill_codes (table, { 1, 2, 3, 4 });
    fill_codes (out, { 0, 0, 0, 0 });

    EmbedSpec spec;
    spec._M_rows = 2u;
    spec._M_tokens = 2u;
    spec._M_vocab = 2u;
    spec._M_dim = 2u;

    Trainable tv = make_view (table, Shape::matrix (2u, 2u), true);
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);

    SECTION ("id at or above vocab is OUT_OF_RANGE")
    {
        const uint16_t index[] = { 0u, 5u };
        REQUIRE (embedding_forward (spec, index, nullptr, tv, ov)
                 == Status::OUT_OF_RANGE);
    }
    SECTION ("null index is BAD_LAYOUT")
    {
        REQUIRE (embedding_forward (spec, nullptr, nullptr, tv, ov)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("zero rows is BAD_LAYOUT")
    {
        EmbedSpec empty;
        empty._M_rows = 0u;
        empty._M_vocab = 2u;
        empty._M_dim = 2u;
        const uint16_t index[] = { 0u };
        REQUIRE (embedding_forward (empty, index, nullptr, tv, ov)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("count mismatch is BAD_LAYOUT")
    {
        Tensor small;
        fill_codes (small, { 0 });
        Trainable ov2 = make_view (small, Shape::vector (1u), true);
        const uint16_t index[] = { 0u, 1u };
        REQUIRE (embedding_forward (spec, index, nullptr, tv, ov2)
                 == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("embedding_backward scatters; repeated ids sum into one row")
{
    /* vocab 3, dim 2, rows 3, ids reference rows 0,0,2 */
    Tensor table, out;
    fill_codes (table, { 9, 9, 9, 9, 9, 9 });
    fill_codes (out, { 0, 0, 0, 0, 0, 0 });

    EmbedSpec spec;
    spec._M_rows = 3u;
    spec._M_tokens = 3u;
    spec._M_vocab = 3u;
    spec._M_dim = 2u;
    const uint16_t index[] = { 0u, 0u, 2u };

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);

    Trainable tv = make_view (table, Shape::matrix (3u, 2u), true);
    Trainable ov = make_view (out, Shape::matrix (3u, 2u), true);
    REQUIRE (embedding_run (tape, spec, index, nullptr, tv, ov) == Status::OK);

    /* dY lanes: row0, row1 both go to table row 0; row2 goes to table row 2 */
    out.grads = { 100, 200, 300, 400, 500, 600 };
    for (uint32_t i = 0; i < 6u; ++i)
        ov.set_grad (i, out.grads[i]);
    tv.zero_grad ();

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;
    REQUIRE (embedding_backward (ov, ov, ctx) == Status::OK);

    /* table row 0 collects rows 0 + 1; table row 2 collects row 2 */
    REQUIRE (static_cast<int32_t> (table.grads[0]) == 100 + 300);
    REQUIRE (static_cast<int32_t> (table.grads[1]) == 200 + 400);
    REQUIRE (static_cast<int32_t> (table.grads[2]) == 0);
    REQUIRE (static_cast<int32_t> (table.grads[3]) == 0);
    REQUIRE (static_cast<int32_t> (table.grads[4]) == 500);
    REQUIRE (static_cast<int32_t> (table.grads[5]) == 600);
}

TEST_CASE ("embedding_backward skips the padding id")
{
    Tensor table, out;
    fill_codes (table, { 1, 1, 1, 1, 1, 1 });
    fill_codes (out, { 0, 0, 0, 0 });

    EmbedSpec spec;
    spec._M_rows = 2u;
    spec._M_tokens = 2u;
    spec._M_vocab = 3u;
    spec._M_dim = 2u;
    spec._M_padding_index = 1;
    const uint16_t index[] = { 1u, 2u };

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);

    Trainable tv = make_view (table, Shape::matrix (3u, 2u), true);
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
    REQUIRE (embedding_run (tape, spec, index, nullptr, tv, ov) == Status::OK);

    out.grads = { 700, 800, 900, 1000 };
    for (uint32_t i = 0; i < 4u; ++i)
        ov.set_grad (i, out.grads[i]);
    tv.zero_grad ();

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;
    REQUIRE (embedding_backward (ov, ov, ctx) == Status::OK);

    /* padding row 1 receives nothing; row 2 receives the second output row */
    REQUIRE (static_cast<int32_t> (table.grads[0]) == 0);
    REQUIRE (static_cast<int32_t> (table.grads[1]) == 0);
    REQUIRE (static_cast<int32_t> (table.grads[2]) == 0);
    REQUIRE (static_cast<int32_t> (table.grads[3]) == 0);
    REQUIRE (static_cast<int32_t> (table.grads[4]) == 900);
    REQUIRE (static_cast<int32_t> (table.grads[5]) == 1000);
}

TEST_CASE ("embedding bag forward averages listed rows, rounding half away")
{
    /* vocab 3 dim 2; two bags: [0,1] and [2,2,2]; one empty handled separately */
    Tensor table, out;
    fill_codes (table, { 10, 20, -30, 40, 50, 60 });
    fill_codes (out, { 0, 0, 0, 0 });

    EmbedSpec spec;
    spec._M_rows = 2u;
    spec._M_tokens = 5u;
    spec._M_vocab = 3u;
    spec._M_dim = 2u;
    spec._M_bag = 1u;
    const uint16_t index[] = { 0u, 1u, 2u, 2u, 2u };
    const uint16_t offsets[] = { 0u, 2u, 5u };

    Trainable tv = make_view (table, Shape::matrix (3u, 2u), true);
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
    REQUIRE (embedding_forward (spec, index, offsets, tv, ov) == Status::OK);

    /* bag0 mean of row0(10,20) and row1(-30,40): round(sum/2) */
    REQUIRE (static_cast<int32_t> (out.codes[0])
             == ref_i8 (ref_div_round (10 + -30, 2)));
    REQUIRE (static_cast<int32_t> (out.codes[1])
             == ref_i8 (ref_div_round (20 + 40, 2)));
    /* bag1 mean of row2(50,60) three times */
    REQUIRE (static_cast<int32_t> (out.codes[2])
             == ref_i8 (ref_div_round (50 * 3, 3)));
    REQUIRE (static_cast<int32_t> (out.codes[3])
             == ref_i8 (ref_div_round (60 * 3, 3)));
}

TEST_CASE ("embedding bag forward yields a zero row for an empty list")
{
    Tensor table, out;
    fill_codes (table, { 7, 8, 9, 10 });
    fill_codes (out, { 0, 0, 0, 0 });

    EmbedSpec spec;
    spec._M_rows = 2u;
    spec._M_tokens = 1u;
    spec._M_vocab = 2u;
    spec._M_dim = 2u;
    spec._M_bag = 1u;
    const uint16_t index[] = { 1u };
    /* bag0 is [0,1) (one token), bag1 is [1,1) (empty) */
    const uint16_t offsets[] = { 0u, 1u, 1u };

    Trainable tv = make_view (table, Shape::matrix (2u, 2u), true);
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
    REQUIRE (embedding_forward (spec, index, offsets, tv, ov) == Status::OK);

    REQUIRE (static_cast<int32_t> (out.codes[0]) == 9);
    REQUIRE (static_cast<int32_t> (out.codes[1]) == 10);
    REQUIRE (static_cast<int32_t> (out.codes[2]) == 0);
    REQUIRE (static_cast<int32_t> (out.codes[3]) == 0);
}

TEST_CASE ("embedding bag checks the offset stream")
{
    Tensor table, out;
    fill_codes (table, { 1, 2, 3, 4 });

    EmbedSpec spec;
    spec._M_rows = 2u;
    spec._M_tokens = 2u;
    spec._M_vocab = 2u;
    spec._M_dim = 2u;
    spec._M_bag = 1u;
    const uint16_t index[] = { 0u, 1u };

    Trainable tv = make_view (table, Shape::matrix (2u, 2u), true);
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);

    SECTION ("offsets must start at zero")
    {
        const uint16_t bad[] = { 1u, 1u, 2u };
        REQUIRE (embedding_forward (spec, index, bad, tv, ov) == Status::BAD_LAYOUT);
    }
    SECTION ("offsets must not run backwards")
    {
        const uint16_t bad[] = { 0u, 2u, 1u };
        REQUIRE (embedding_forward (spec, index, bad, tv, ov) == Status::BAD_LAYOUT);
    }
    SECTION ("offsets must not exceed the token count")
    {
        const uint16_t bad[] = { 0u, 1u, 5u };
        REQUIRE (embedding_forward (spec, index, bad, tv, ov) == Status::BAD_LAYOUT);
    }
    SECTION ("bag requires a non-null offset stream")
    {
        REQUIRE (embedding_forward (spec, index, nullptr, tv, ov)
                 == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("embedding bag backward divides each token by the length")
{
    Tensor table, out;
    fill_codes (table, { 1, 1, 1, 1, 1, 1 });
    fill_codes (out, { 0, 0, 0, 0 });

    EmbedSpec spec;
    spec._M_rows = 2u;
    spec._M_tokens = 4u;
    spec._M_vocab = 3u;
    spec._M_dim = 2u;
    spec._M_bag = 1u;
    spec._M_scale_grads = 1u;
    const uint16_t index[] = { 0u, 1u, 2u, 2u };
    const uint16_t offsets[] = { 0u, 2u, 4u };

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);
    Trainable tv = make_view (table, Shape::matrix (3u, 2u), true);
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
    REQUIRE (embedding_run (tape, spec, index, offsets, tv, ov) == Status::OK);

    out.grads = { 1000, 1001, 2000, 2001 };
    for (uint32_t i = 0; i < 4u; ++i)
        ov.set_grad (i, out.grads[i]);
    tv.zero_grad ();

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;
    REQUIRE (embedding_backward (ov, ov, ctx) == Status::OK);

    /* Adjoint of the bag mean: token t of bag r carries
       div_round (dY[r][col], length_r), and a table row named by several tokens
       collects every contribution.  Rebuild that accumulation from the id
       stream instead of hand-listing lanes, so the reference cannot drift from
       the offsets. */
    for (uint32_t id = 0u; id < spec._M_vocab; ++id)
        for (uint32_t column = 0u; column < spec._M_dim; ++column)
            {
                int32_t expect = 0;
                for (uint32_t bag = 0u; bag < spec._M_rows; ++bag)
                    {
                        const uint32_t begin = offsets[bag];
                        const uint32_t end = offsets[bag + 1u];
                        const int32_t length = static_cast<int32_t> (end - begin);
                        for (uint32_t token = begin; token < end; ++token)
                            if (index[token] == id)
                                expect += ref_div_round (
                                    out.grads[bag * spec._M_dim + column], length);
                    }
                REQUIRE (static_cast<int32_t> (
                             table.grads[id * spec._M_dim + column])
                         == expect);
            }
    /* id 2 names both slots of bag 1, so its lanes are exactly twice
       round(2000 / 2) and round(2001 / 2); ids 0 and 1 split bag 0 once each. */
    REQUIRE (static_cast<int32_t> (table.grads[4]) == 2 * ref_div_round (2000, 2));
    REQUIRE (static_cast<int32_t> (table.grads[5]) == 2 * ref_div_round (2001, 2));
    REQUIRE (static_cast<int32_t> (table.grads[0]) == ref_div_round (1000, 2));
    REQUIRE (static_cast<int32_t> (table.grads[2]) == ref_div_round (1000, 2));
}

TEST_CASE ("embedding bag backward without scaling routes the full gradient")
{
    Tensor table, out;
    fill_codes (table, { 1, 1, 1, 1 });
    fill_codes (out, { 0, 0 });

    EmbedSpec spec;
    spec._M_rows = 1u;
    spec._M_tokens = 3u;
    spec._M_vocab = 2u;
    spec._M_dim = 2u;
    spec._M_bag = 1u;
    spec._M_scale_grads = 0u;
    const uint16_t index[] = { 0u, 0u, 1u };
    const uint16_t offsets[] = { 0u, 3u };

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);
    Trainable tv = make_view (table, Shape::matrix (2u, 2u), true);
    Trainable ov = make_view (out, Shape::matrix (1u, 2u), true);
    REQUIRE (embedding_run (tape, spec, index, offsets, tv, ov) == Status::OK);

    out.grads = { 300, 400 };
    ov.set_grad (0, 300);
    ov.set_grad (1, 400);
    tv.zero_grad ();

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;
    REQUIRE (embedding_backward (ov, ov, ctx) == Status::OK);

    /* id 0 twice with full gradient, id 1 once */
    REQUIRE (static_cast<int32_t> (table.grads[0]) == 600);
    REQUIRE (static_cast<int32_t> (table.grads[1]) == 800);
    REQUIRE (static_cast<int32_t> (table.grads[2]) == 300);
    REQUIRE (static_cast<int32_t> (table.grads[3]) == 400);
}

TEST_CASE ("embedding_backward counts the lanes that saturate the Q15 rail")
{
    Tensor table, out;
    fill_codes (table, { 1, 2, 3, 4 });
    fill_codes (out, { 0, 0, 0, 0 });

    EmbedSpec spec;
    spec._M_rows = 2u;
    spec._M_tokens = 2u;
    spec._M_vocab = 2u;
    spec._M_dim = 2u;
    const uint16_t index[] = { 0u, 1u };

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);

    Trainable tv = make_view (table, Shape::matrix (2u, 2u), true);
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
    REQUIRE (embedding_run (tape, spec, index, nullptr, tv, ov) == Status::OK);

    /* the scatter accumulates, so start from lanes already near a rail */
    const int16_t poisoned[4] = { 30000, -30000, 20000, -20000 };
    for (uint32_t i = 0; i < 4u; ++i)
        tv.set_grad (i, poisoned[i]);
    const int16_t dy[4] = { 20000, -20000, 20000, 15000 };
    for (uint32_t i = 0; i < 4u; ++i)
        ov.set_grad (i, dy[i]);

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;
    ctx._M_saturation_events = 0u;
    REQUIRE (embedding_backward (ov, ov, ctx) == Status::OK);

    /* three lane sums leave the [-32768, 32767] rails and are clamped there;
       the fourth, -20000 + 15000, fits */
    REQUIRE (static_cast<int32_t> (table.grads[0]) == 32767);
    REQUIRE (static_cast<int32_t> (table.grads[1]) == -32768);
    REQUIRE (static_cast<int32_t> (table.grads[2]) == 32767);
    REQUIRE (static_cast<int32_t> (table.grads[3]) == -5000);
    REQUIRE (ctx._M_saturation_events == 3u);
}

TEST_CASE ("embedding_backward on a frozen table is a successful no-op")
{
    Tensor table, out;
    fill_codes (table, { 5, 6, 7, 8 });
    fill_codes (out, { 0, 0, 0, 0 });

    EmbedSpec spec;
    spec._M_rows = 2u;
    spec._M_tokens = 2u;
    spec._M_vocab = 2u;
    spec._M_dim = 2u;
    const uint16_t index[] = { 1u, 0u };

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);

    Trainable tv = make_view (table, Shape::matrix (2u, 2u), false); /* frozen */
    Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
    REQUIRE (embedding_run (tape, spec, index, nullptr, tv, ov) == Status::OK);
    /* the gather is still the forward: row 1 then row 0 */
    REQUIRE (static_cast<int32_t> (out.codes[0]) == 7);
    REQUIRE (static_cast<int32_t> (out.codes[1]) == 8);
    REQUIRE (static_cast<int32_t> (out.codes[2]) == 5);
    REQUIRE (static_cast<int32_t> (out.codes[3]) == 6);

    const int16_t dy[4] = { 1000, 2000, 3000, 4000 };
    for (uint32_t i = 0; i < 4u; ++i)
        ov.set_grad (i, dy[i]);

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;
    REQUIRE (embedding_backward (ov, ov, ctx) == Status::OK);
    /* no gradient buffer means nothing to write, but not an error either */
    REQUIRE (table.grads == std::vector<int16_t> (4, 0));
    REQUIRE (ctx._M_saturation_events == 0u);

    /* an output with no gradient lanes at all is a layout error */
    Trainable frozen_out = make_view (out, Shape::matrix (2u, 2u), false);
    GradContext null_ctx = ctx;
    null_ctx._M_pool = &pool;
    REQUIRE (embedding_backward (frozen_out, frozen_out, null_ctx)
             == Status::BAD_LAYOUT);
}

TEST_CASE ("embedding bag backward skips a padding id inside a bag")
{
    Tensor table, out;
    fill_codes (table, { 100, 100, 2, 4, 6, 8 }); /* vocab 3, dim 2 */
    fill_codes (out, { 0, 0 });

    EmbedSpec spec;
    spec._M_rows = 1u;
    spec._M_tokens = 3u;
    spec._M_vocab = 3u;
    spec._M_dim = 2u;
    spec._M_bag = 1u;
    spec._M_padding_index = 1;
    const uint16_t index[] = { 1u, 2u, 2u };
    const uint16_t offsets[] = { 0u, 3u };

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);

    Trainable tv = make_view (table, Shape::matrix (3u, 2u), true);
    Trainable ov = make_view (out, Shape::matrix (1u, 2u), true);
    REQUIRE (embedding_run (tape, spec, index, offsets, tv, ov) == Status::OK);

    /* the mean divides by the full length (3) but sums only the non-padding
       entries: (6+6)/3 and (8+8)/3, rounded half away from zero */
    REQUIRE (static_cast<int32_t> (out.codes[0]) == ref_div_round (12, 3));
    REQUIRE (static_cast<int32_t> (out.codes[1]) == ref_div_round (16, 3));

    const int16_t dy[2] = { 300, 400 };
    ov.set_grad (0, dy[0]);
    ov.set_grad (1, dy[1]);
    tv.zero_grad ();

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;
    REQUIRE (embedding_backward (ov, ov, ctx) == Status::OK);

    /* each of the two live tokens carries round(dY / 3); the padding row of the
       table collects nothing even though it appeared in the list */
    const int32_t per_token[2] = { ref_div_round (300, 3), ref_div_round (400, 3) };
    for (uint32_t column = 0; column < 2u; ++column)
        {
            REQUIRE (static_cast<int32_t> (table.grads[column]) == 0);
            REQUIRE (static_cast<int32_t> (table.grads[2 + column]) == 0);
            REQUIRE (static_cast<int32_t> (table.grads[4 + column])
                     == 2 * per_token[column]);
        }
}

TEST_CASE ("embedding bag backward drops lanes whose scaled value is zero")
{
    Tensor table, out;
    fill_codes (table, { 1, 1, 1, 1 });
    fill_codes (out, { 0, 0 });

    EmbedSpec spec;
    spec._M_rows = 1u;
    spec._M_tokens = 4u;
    spec._M_vocab = 2u;
    spec._M_dim = 2u;
    spec._M_bag = 1u;
    spec._M_scale_grads = 1u;
    const uint16_t index[] = { 0u, 0u, 0u, 0u };
    const uint16_t offsets[] = { 0u, 4u };

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);

    Trainable tv = make_view (table, Shape::matrix (2u, 2u), true);
    Trainable ov = make_view (out, Shape::matrix (1u, 2u), true);
    REQUIRE (embedding_run (tape, spec, index, offsets, tv, ov) == Status::OK);

    /* a unit gradient over a four-token bag rounds to zero, and a zero lane is
       never routed: the table must stay exactly where the forward left it */
    ov.set_grad (0, 1);
    ov.set_grad (1, 1);
    tv.zero_grad ();

    GradContext ctx;
    ctx._M_pool = &pool;
    ctx._M_operands = nodes[0]._M_operands;
    ctx._M_operand_count = nodes[0]._M_operand_count;
    ctx._M_saved = nodes[0]._M_saved;
    ctx._M_saved_count = nodes[0]._M_saved_count;
    REQUIRE (embedding_backward (ov, ov, ctx) == Status::OK);
    REQUIRE (ref_div_round (1, 4) == 0);
    for (int16_t g : table.grads)
        REQUIRE (g == 0);
    REQUIRE (ctx._M_saturation_events == 0u);

    /* one more unit of signal is enough to move a lane */
    ov.set_grad (0, 2);
    tv.zero_grad ();
    REQUIRE (embedding_backward (ov, ov, ctx) == Status::OK);
    REQUIRE (static_cast<int32_t> (table.grads[0])
             == 4 * ref_div_round (2, 4));
    REQUIRE (static_cast<int32_t> (table.grads[1]) == 0);
}

TEST_CASE ("embedding_run surfaces Status codes")
{
    Tensor table, out;
    fill_codes (table, { 1, 2, 3, 4 });

    EmbedSpec spec;
    spec._M_rows = 2u;
    spec._M_tokens = 2u;
    spec._M_vocab = 2u;
    spec._M_dim = 2u;

    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);

    SECTION ("closed tape is INTERNAL")
    {
        REQUIRE_FALSE (tape.is_open ());
        const uint16_t index[] = { 0u, 1u };
        Trainable tv = make_view (table, Shape::matrix (2u, 2u), true);
        Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
        REQUIRE (embedding_run (tape, spec, index, nullptr, tv, ov)
                 == Status::INTERNAL);
    }
    REQUIRE (tape.begin () == Status::OK);

    SECTION ("out-of-range id is OUT_OF_RANGE")
    {
        const uint16_t index[] = { 0u, 9u };
        Trainable tv = make_view (table, Shape::matrix (2u, 2u), true);
        Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
        REQUIRE (embedding_run (tape, spec, index, nullptr, tv, ov)
                 == Status::OUT_OF_RANGE);
    }
    SECTION ("null index is BAD_LAYOUT")
    {
        Trainable tv = make_view (table, Shape::matrix (2u, 2u), true);
        Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
        REQUIRE (embedding_run (tape, spec, nullptr, nullptr, tv, ov)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("bag missing offsets is BAD_LAYOUT")
    {
        EmbedSpec bag = spec;
        bag._M_bag = 1u;
        bag._M_tokens = 2u;
        const uint16_t index[] = { 0u, 1u };
        Trainable tv = make_view (table, Shape::matrix (2u, 2u), true);
        Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
        REQUIRE (embedding_run (tape, bag, index, nullptr, tv, ov)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("a lookup given offsets is BAD_LAYOUT")
    {
        /* the two forms are exclusive in both directions: a lookup must not
           hand over an offset stream either */
        fill_codes (out, { 0, 0, 0, 0 });
        const uint16_t index[] = { 0u, 1u };
        const uint16_t offsets[] = { 0u, 1u, 2u };
        Trainable tv = make_view (table, Shape::matrix (2u, 2u), true);
        Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
        REQUIRE (embedding_run (tape, spec, index, offsets, tv, ov)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("both documented forms record one node")
    {
        fill_codes (out, { 0, 0, 0, 0 });
        const uint16_t lookup[] = { 1u, 0u };
        Trainable tv = make_view (table, Shape::matrix (2u, 2u), true);
        Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
        REQUIRE (embedding_run (tape, spec, lookup, nullptr, tv, ov)
                 == Status::OK);
        REQUIRE (tape.node_count () == 1u);
        Node *const node = tape.node (0u);
        REQUIRE (node != nullptr);
        REQUIRE (node->_M_operand_count == 1u);
        REQUIRE_FALSE (node->_M_saved[_S_embed_spec].is_null ());
        REQUIRE_FALSE (node->_M_saved[_S_embed_index].is_null ());
        /* a lookup has no offset stream, so that slot stays a null handle */
        REQUIRE (node->_M_saved[_S_embed_offsets].is_null ());
        REQUIRE (node->_M_saved[_S_embed_unused].is_null ());

        EmbedSpec bag = spec;
        bag._M_bag = 1u;
        bag._M_tokens = 2u;
        const uint16_t ids[] = { 0u, 1u };
        const uint16_t offsets[] = { 0u, 1u, 2u };
        tape.reset ();
        REQUIRE (tape.begin () == Status::OK);
        REQUIRE (embedding_run (tape, bag, ids, offsets, tv, ov) == Status::OK);
        Node *const bag_node = tape.node (0u);
        REQUIRE_FALSE (bag_node->_M_saved[_S_embed_offsets].is_null ());
        /* the ids and offsets are copied into the pool, so the caller may free
           its own streams right after the record */
        GradContext ctx;
        ctx._M_pool = &pool;
        ctx._M_operands = bag_node->_M_operands;
        ctx._M_operand_count = bag_node->_M_operand_count;
        ctx._M_saved = bag_node->_M_saved;
        ctx._M_saved_count = bag_node->_M_saved_count;
        const uint16_t *const saved_ids
            = ctx.saved_as<uint16_t> (_S_embed_index);
        const uint16_t *const saved_offsets
            = ctx.saved_as<uint16_t> (_S_embed_offsets);
        REQUIRE (saved_ids != nullptr);
        REQUIRE (saved_offsets != nullptr);
        for (uint32_t token = 0u; token < bag.index_count (); ++token)
            REQUIRE (saved_ids[token] == ids[token]);
        for (uint32_t bag_row = 0u; bag_row <= bag._M_rows; ++bag_row)
            REQUIRE (saved_offsets[bag_row] == offsets[bag_row]);
        const EmbedSpec *const saved_spec
            = ctx.saved_as<EmbedSpec> (_S_embed_spec);
        REQUIRE (saved_spec != nullptr);
        REQUIRE (saved_spec->_M_bag == 1u);
        REQUIRE (saved_spec->_M_rows == bag._M_rows);
    }
    SECTION ("a pool too small for the streams is ARENA_OVERFLOW")
    {
        EmbedSpec bag = spec;
        bag._M_bag = 1u;
        bag._M_tokens = 2u;
        const uint16_t ids[] = { 0u, 1u };
        const uint16_t offsets[] = { 0u, 1u, 2u };
        fill_codes (out, { 0, 0, 0, 0 });
        AlignedArena<16> tiny;
        Pool small_pool = tiny.pool ();
        Node nodes[2];
        Tape small_tape (nodes, 2u, small_pool);
        REQUIRE (small_tape.begin () == Status::OK);
        Trainable tv = make_view (table, Shape::matrix (2u, 2u), true);
        Trainable ov = make_view (out, Shape::matrix (2u, 2u), true);
        REQUIRE (embedding_run (small_tape, bag, ids, offsets, tv, ov)
                 == Status::ARENA_OVERFLOW);
        /* the failed record leaves the tape empty */
        REQUIRE (small_tape.node_count () == 0u);
    }
}

TEST_CASE ("tied views share the code and gradient buffers")
{
    Tensor owner;
    fill_codes (owner, { 1, 2, 3, 4, 5, 6 });
    Trainable own = make_view (owner, Shape::matrix (3u, 2u), true);
    Trainable alias;

    SECTION ("valid tie")
    {
        REQUIRE (tied (own, Shape::matrix (2u, 3u), alias) == Status::OK);
        REQUIRE (alias._M_code == own._M_code);
        REQUIRE (alias._M_grad == own._M_grad);
        REQUIRE (alias._M_shape == Shape::matrix (2u, 3u));
        REQUIRE (alias.requires_grad ());
        REQUIRE (grad_shared (own, alias));
        REQUIRE (fully_tied (own, alias));
        /* writing through one view is visible through the other */
        alias.set_grad (0, 123);
        REQUIRE (own.grad_at (0) == 123);
        REQUIRE (owner.grads[0] == 123);
    }
    SECTION ("element count mismatch is BAD_LAYOUT")
    {
        REQUIRE (tied (own, Shape::matrix (2u, 2u), alias) == Status::BAD_LAYOUT);
    }
    SECTION ("forward-only owner cannot be tied")
    {
        Trainable no_grad = make_view (owner, Shape::matrix (3u, 2u), false);
        REQUIRE (tied (no_grad, Shape::matrix (2u, 3u), alias) == Status::BAD_LAYOUT);
    }
    SECTION ("grad_shared / fully_tied distinguish a code-only alias")
    {
        Trainable a = make_view (owner, Shape::matrix (3u, 2u), true);
        Tensor other;
        fill_codes (other, { 1, 2, 3, 4, 5, 6 });
        Trainable b = make_view (other, Shape::matrix (3u, 2u), true);
        REQUIRE_FALSE (grad_shared (a, b));
        REQUIRE (grad_shared (a, a));
        REQUIRE (fully_tied (a, a));
    }
}

/* ================================================================== */
/* train_running_stats.hpp                                            */
/* ================================================================== */

TEST_CASE ("running_stats_bind validates its inputs")
{
    int16_t mean[4] = { 0, 0, 0, 0 };
    int16_t var[4] = { 0, 0, 0, 0 };
    RunningStats stats;

    SECTION ("null buffers are BAD_LAYOUT")
    {
        REQUIRE (running_stats_bind (stats, nullptr, var, 4u) == Status::BAD_LAYOUT);
        REQUIRE (running_stats_bind (stats, mean, nullptr, 4u) == Status::BAD_LAYOUT);
    }
    SECTION ("window count bounds")
    {
        REQUIRE (running_stats_bind (stats, mean, var, 0u) == Status::OUT_OF_RANGE);
        REQUIRE (running_stats_bind (stats, mean, var, _S_max_running_windows + 1u)
                 == Status::OUT_OF_RANGE);
        REQUIRE (running_stats_bind (stats, mean, var, 4u) == Status::OK);
        REQUIRE (stats._M_windows == 4u);
        REQUIRE (stats._M_step == 0u);
        REQUIRE (stats._M_frozen == 0u);
        REQUIRE (stats._M_saturated == 0u);
    }
}

TEST_CASE ("running_stats_alloc carves from the pool")
{
    AlignedArena<_S_train_pool_bytes> arena;
    Pool pool = arena.pool ();
    RunningStats stats;

    SECTION ("window bounds before carving")
    {
        REQUIRE (running_stats_alloc (pool, stats, 0u) == Status::OUT_OF_RANGE);
        REQUIRE (running_stats_alloc (pool, stats, _S_max_running_windows + 1u)
                 == Status::OUT_OF_RANGE);
    }
    SECTION ("valid carve")
    {
        REQUIRE (running_stats_alloc (pool, stats, 8u) == Status::OK);
        REQUIRE (stats._M_mean != nullptr);
        REQUIRE (stats._M_var != nullptr);
        REQUIRE (stats._M_windows == 8u);
        for (uint32_t w = 0; w < 8u; ++w)
            {
                REQUIRE (stats._M_mean[w] == 0);
                REQUIRE (stats._M_var[w] == 0);
            }
        /* the carve is bump-pointer arithmetic inside the caller's arena:
           mean first, then var, exactly 4 bytes per window. */
        REQUIRE (stats._M_var == stats._M_mean + 8);
        const uint8_t *const base = arena._M_data;
        REQUIRE (reinterpret_cast<const uint8_t *> (stats._M_mean) >= base);
        REQUIRE (reinterpret_cast<const uint8_t *> (stats._M_var) > base);
        REQUIRE (pool.remaining () == _S_train_pool_bytes - running_stats_bytes (8u));
        REQUIRE (pool.stats()._M_used == running_stats_bytes (8u));
        REQUIRE (pool.stats()._M_slabs == 2u);
    }
    SECTION ("pool exhaustion leaves stats empty")
    {
        /* two int16 buffers of eight windows are exactly four bytes per
           window, so a 32-byte arena is the tightest that still fits and one
           window more cannot be carved at all. */
        AlignedArena<running_stats_bytes (8u)> snug;
        Pool exact = snug.pool ();
        RunningStats s1;
        REQUIRE (running_stats_alloc (exact, s1, 8u) == Status::OK);
        REQUIRE (exact.remaining () == 0u);

        AlignedArena<32> tiny;
        Pool small = tiny.pool ();
        RunningStats s2;
        REQUIRE (running_stats_alloc (small, s2, 9u) == Status::ARENA_OVERFLOW);
        REQUIRE (s2._M_mean == nullptr);
        REQUIRE (s2._M_var == nullptr);
        REQUIRE (s2._M_windows == 0u);
    }
    REQUIRE (running_stats_bytes (8u) == 32u);
    REQUIRE (running_stats_snapshot_bytes (8u) == 36u);
}

TEST_CASE ("running_stats_active tracks the freeze switch")
{
    int16_t mean[2] = { 0, 0 };
    int16_t var[2] = { 0, 0 };
    RunningStats stats;
    REQUIRE (running_stats_bind (stats, mean, var, 2u) == Status::OK);
    REQUIRE (running_stats_active (stats));
    running_stats_freeze (stats);
    REQUIRE_FALSE (running_stats_active (stats));
    running_stats_thaw (stats);
    REQUIRE (running_stats_active (stats));
    RunningStats fresh;
    REQUIRE_FALSE (running_stats_active (fresh));
}

TEST_CASE ("bessel_ratio_q14 follows the documented rule")
{
    struct Row
    {
        uint32_t n;
        int32_t ratio;
    };
    const Row rows[] = { { 0u, 16384 }, { 1u, 16384 }, { 2u, 32768 },
                         { 3u, 24576 }, { 4u, 21845 }, { 5u, 20480 },
                         { 8u, 18724 } }; /* 8/7 at 14 fraction bits */
    for (const Row &r : rows)
        {
            REQUIRE (bessel_ratio_q14 (r.n) == r.ratio);
            REQUIRE (bessel_ratio_q14 (r.n) == ref_bessel (r.n));
        }
}

TEST_CASE ("running_stats_unbiased converts a biased variance with Bessel")
{
    struct Row
    {
        int32_t biased;
        uint32_t elements;
    };
    const Row rows[] = { { 1000, 1u }, { 1000, 2u }, { 400, 3u },
                         { 20000, 2u }, { -1234, 4u }, { 0, 8u } };
    for (const Row &r : rows)
        REQUIRE (running_stats_unbiased_q15 (r.biased, r.elements)
                 == ref_unbiased (r.biased, r.elements));

    /* a biased variance of ~0.61 that doubles exceeds the rail and clamps */
    REQUIRE (running_stats_unbiased_q15 (20000, 2u) == 32767);
    REQUIRE (ref_unbiased (20000, 2u) == 32767);
}

TEST_CASE ("running_stats_lerp_q15 is exact at both rails")
{
    bool sat = false;

    /* momentum 0 keeps the current value */
    REQUIRE (running_stats_lerp_q15 (1234, 5678, 0, sat) == 1234);
    REQUIRE_FALSE (sat);

    /* momentum Q15-one moves fully to the target (delta exact) */
    REQUIRE (running_stats_lerp_q15 (1000, 2000, 32768, sat) == 2000);
    REQUIRE_FALSE (sat);

    /* default momentum 3277 blends ~10% */
    const int32_t got = running_stats_lerp_q15 (0, 10000, 3277, sat);
    REQUIRE (got == ref_lerp (0, 10000, 3277, sat));

    /* a target beyond the rail saturates and reports it */
    bool big_sat = false;
    const int32_t clamped
        = running_stats_lerp_q15 (30000, 40000, 32768, big_sat);
    REQUIRE (clamped == 32767);
    REQUIRE (big_sat);
}

TEST_CASE ("running_stats_update performs the EMA over every window")
{
    int16_t mean[3] = { 0, 0, 0 };
    int16_t var[3] = { 0, 0, 0 };
    RunningStats stats;
    REQUIRE (running_stats_bind (stats, mean, var, 3u) == Status::OK);

    const int16_t bm[3] = { 4000, -8000, 100 };
    const int16_t bv[3] = { 2000, 9000, -500 };
    const uint32_t elements = 4u;
    const int32_t momentum = _S_running_momentum_default_q15; /* 3277 */

    REQUIRE (running_stats_update (stats, bm, bv, elements, momentum) == Status::OK);
    REQUIRE (stats._M_step == 1u);

    for (uint32_t w = 0; w < 3u; ++w)
        {
            bool s1 = false, s2 = false;
            const int32_t want_mean
                = ref_lerp (0, static_cast<int32_t> (bm[w]), momentum, s1);
            const int32_t want_var
                = ref_lerp (0, ref_unbiased (static_cast<int32_t> (bv[w]), elements),
                            momentum, s2);
            REQUIRE (static_cast<int32_t> (stats._M_mean[w]) == want_mean);
            REQUIRE (static_cast<int32_t> (stats._M_var[w]) == want_var);
        }
}

TEST_CASE ("running_stats_update clamps momentum and is a no-op when frozen")
{
    int16_t mean[2] = { 500, -500 };
    int16_t var[2] = { 1000, -1000 };
    RunningStats stats;
    REQUIRE (running_stats_bind (stats, mean, var, 2u) == Status::OK);

    const int16_t bm[2] = { 6000, 6000 };
    const int16_t bv[2] = { 7000, 7000 };

    SECTION ("negative momentum clamps to zero: no movement")
    {
        RunningStats s = stats;
        REQUIRE (running_stats_update (s, bm, bv, 4u, -123) == Status::OK);
        REQUIRE (s._M_mean[0] == 500);
        REQUIRE (s._M_mean[1] == -500);
        REQUIRE (s._M_step == 1u);
    }
    SECTION ("momentum above Q15 clamps to the max")
    {
        RunningStats s = stats;
        REQUIRE (running_stats_update (s, bm, bv, 4u, 99999) == Status::OK);
        /* value equals a Q15-max momentum lerp toward the batch mean */
        bool sat = false;
        REQUIRE (static_cast<int32_t> (s._M_mean[0])
                 == ref_lerp (500, 6000, 32767, sat));
    }
    SECTION ("frozen block ignores the update")
    {
        RunningStats s = stats;
        running_stats_freeze (s);
        REQUIRE (running_stats_update (s, bm, bv, 4u, 3277) == Status::OK);
        REQUIRE (s._M_mean[0] == 500);
        REQUIRE (s._M_step == 0u);
    }
    SECTION ("active block with null batch is BAD_LAYOUT")
    {
        RunningStats s = stats;
        REQUIRE (running_stats_update (s, nullptr, bv, 4u, 3277)
                 == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("running_stats_update is deterministic across identical runs")
{
    int16_t meanA[4] = { 0, 0, 0, 0 };
    int16_t varA[4] = { 0, 0, 0, 0 };
    int16_t meanB[4] = { 0, 0, 0, 0 };
    int16_t varB[4] = { 0, 0, 0, 0 };
    RunningStats A, B;
    REQUIRE (running_stats_bind (A, meanA, varA, 4u) == Status::OK);
    REQUIRE (running_stats_bind (B, meanB, varB, 4u) == Status::OK);

    const int16_t bm[4] = { 3000, -4000, 1500, 6000 };
    const int16_t bv[4] = { 900, 8000, -300, 12000 };

    /* ten identical steps on both objects must agree lane-for-lane */
    for (int k = 0; k < 10; ++k)
        {
            REQUIRE (running_stats_update (A, bm, bv, 5u, 3277) == Status::OK);
            REQUIRE (running_stats_update (B, bm, bv, 5u, 3277) == Status::OK);
        }
    REQUIRE (A._M_step == 10u);
    REQUIRE (B._M_step == 10u);
    for (uint32_t w = 0; w < 4u; ++w)
        {
            REQUIRE (A._M_mean[w] == B._M_mean[w]);
            REQUIRE (A._M_var[w] == B._M_var[w]);
        }
}

TEST_CASE ("running_stats_reset clears buffers and counters")
{
    int16_t mean[2] = { 0, 0 };
    int16_t var[2] = { 0, 0 };
    RunningStats stats;
    REQUIRE (running_stats_bind (stats, mean, var, 2u) == Status::OK);
    const int16_t bm[2] = { 4000, 4000 };
    const int16_t bv[2] = { 4000, 4000 };
    REQUIRE (running_stats_update (stats, bm, bv, 4u, 3277) == Status::OK);
    REQUIRE (stats._M_step == 1u);
    /* force the saturated counter to a known non-zero state via lerp directly */
    stats._M_saturated = 5u;

    running_stats_reset (stats);
    REQUIRE (stats._M_step == 0u);
    REQUIRE (stats._M_saturated == 0u);
    REQUIRE (stats._M_mean[0] == 0);
    REQUIRE (stats._M_mean[1] == 0);
    REQUIRE (stats._M_var[0] == 0);
    REQUIRE (stats._M_var[1] == 0);
}

TEST_CASE ("running_stats snapshot and restore round-trip")
{
    int16_t mean[3] = { 111, -222, 333 };
    int16_t var[3] = { 444, 555, -666 };
    RunningStats stats;
    REQUIRE (running_stats_bind (stats, mean, var, 3u) == Status::OK);
    stats._M_step = 7u;

    alignas (8) uint8_t scratch[running_stats_snapshot_bytes (3u)];
    REQUIRE (running_stats_snapshot (stats, scratch) == Status::OK);
    REQUIRE (running_stats_snapshot (stats, nullptr) == Status::BAD_LAYOUT);

    /* mutate then restore from the snapshot */
    for (uint32_t w = 0; w < 3u; ++w)
        {
            stats._M_mean[w] = 0;
            stats._M_var[w] = 0;
        }
    stats._M_step = 0u;
    REQUIRE (running_stats_restore (stats, scratch) == Status::OK);
    REQUIRE (static_cast<int32_t> (stats._M_mean[0]) == 111);
    REQUIRE (static_cast<int32_t> (stats._M_mean[1]) == -222);
    REQUIRE (static_cast<int32_t> (stats._M_mean[2]) == 333);
    REQUIRE (static_cast<int32_t> (stats._M_var[2]) == -666);
    REQUIRE (stats._M_step == 7u);

    /* a negative step in the scratch decays to zero on restore */
    int32_t *step = reinterpret_cast<int32_t *> (scratch + 4u * 3u);
    *step = -5;
    REQUIRE (running_stats_restore (stats, scratch) == Status::OK);
    REQUIRE (stats._M_step == 0u);
}

TEST_CASE ("running_stats snapshot requires an active block")
{
    int16_t mean[2] = { 1, 2 };
    int16_t var[2] = { 3, 4 };
    RunningStats stats;
    REQUIRE (running_stats_bind (stats, mean, var, 2u) == Status::OK);
    running_stats_freeze (stats);
    alignas (8) uint8_t scratch[running_stats_snapshot_bytes (2u)];
    REQUIRE (running_stats_snapshot (stats, scratch) == Status::BAD_LAYOUT);
    REQUIRE (running_stats_restore (stats, scratch) == Status::BAD_LAYOUT);
}

TEST_CASE ("running_stats_deploy_form converts mean and deviation")
{
    int16_t mean[2] = { 16384, -8192 }; /* 0.5 and -0.25 in Q15 */
    int16_t var[2] = { 8192, 0 };       /* 0.25 and 0 in Q15 */
    RunningStats stats;
    REQUIRE (running_stats_bind (stats, mean, var, 2u) == Status::OK);

    int32_t mean_code = 0;
    uint32_t dev_code = 0;
    REQUIRE (running_stats_deploy_form (stats, 8u, 0u, mean_code, dev_code)
             == Status::OK);
    /* mean_code = i8_code(round_shift(mean*127, 15)) */
    REQUIRE (mean_code == ref_i8 (ref_round_shift (16384LL * 127, 15u)));

    /* deviation = sqrt(var) on the Q15 grid, then *127 rounded */
    const uint64_t root = ref_isqrt (static_cast<uint64_t> (8192) << 15u);
    REQUIRE (dev_code
             == static_cast<uint32_t> (
                 ref_round_shift (static_cast<int64_t> (root) * 127, 15u)));

    /* a zero variance yields a zero deviation */
    REQUIRE (running_stats_deploy_form (stats, 4u, 1u, mean_code, dev_code)
             == Status::OK);
    REQUIRE (dev_code == 0u);
}

TEST_CASE ("running_stats_deploy_form bounds")
{
    int16_t mean[1] = { 100 };
    int16_t var[1] = { 100 };
    RunningStats stats;
    REQUIRE (running_stats_bind (stats, mean, var, 1u) == Status::OK);
    int32_t mc = 0;
    uint32_t dc = 0;
    REQUIRE (running_stats_deploy_form (stats, 0u, 0u, mc, dc)
             == Status::OUT_OF_RANGE); /* window == 0 */
    REQUIRE (running_stats_deploy_form (stats, 4u, 1u, mc, dc)
             == Status::OUT_OF_RANGE); /* index >= windows */

    RunningStats fresh;
    REQUIRE (running_stats_deploy_form (fresh, 4u, 0u, mc, dc) == Status::BAD_LAYOUT);
}

TEST_CASE ("running stats normalization round-trip stays within one code step")
{
    /* q15 -> deploy mean code -> back to q15 must be within one code step (258) */
    for (int32_t q = -32768; q <= 32767; q += 733)
        {
            const int32_t code = ref_i8 (ref_round_shift (static_cast<int64_t> (q) * 127, 15u));
            const int32_t back = ref_code_q15 (code);
            const int32_t error = back - q;
            REQUIRE (error <= 258);
            REQUIRE (error >= -258);
            /* the library converters land on the same code */
            REQUIRE (q15_to_code (q) == code);
            REQUIRE (code_to_q15 (q15_to_code (q)) == back);
        }
    /* the header promises the pair is lossless on the code grid itself */
    for (int32_t c = -127; c <= 127; ++c)
        REQUIRE (q15_to_code (code_to_q15 (c)) == c);
}

TEST_CASE ("running_stats_deploy_form maps every Q15 lane through the code grid")
{
    int16_t mean[1] = { 0 };
    int16_t var[1] = { 0 };
    RunningStats stats;
    REQUIRE (running_stats_bind (stats, mean, var, 1u) == Status::OK);

    int32_t mean_code = 0;
    uint32_t dev_code = 0;
    for (int32_t q = -32768; q <= 32767; q += 257)
        {
            mean[0] = static_cast<int16_t> (q);
            var[0] = 0;
            REQUIRE (running_stats_deploy_form (stats, 4u, 0u, mean_code, dev_code)
                     == Status::OK);
            REQUIRE (mean_code
                     == ref_i8 (ref_round_shift (static_cast<int64_t> (q) * 127, 15u)));
            /* the symmetric int8 rails: -128 is never emitted */
            REQUIRE (mean_code >= -127);
            REQUIRE (mean_code <= 127);
            REQUIRE (mean_code != -128);
            REQUIRE (dev_code == 0u);
            const int32_t error = code_to_q15 (mean_code) - q;
            REQUIRE (error <= _S_code_to_q15_scale);
            REQUIRE (error >= -_S_code_to_q15_scale);
        }
}

TEST_CASE ("deploy deviation is the floor integer root of the variance")
{
    int16_t mean[1] = { 0 };
    int16_t var[1] = { 0 };
    RunningStats stats;
    REQUIRE (running_stats_bind (stats, mean, var, 1u) == Status::OK);

    /* the wide root folds its argument, so check the whole documented span
       against an independent Newton square root rather than echoing the kernel */
    const uint32_t grid[] = { 0u,    1u,    2u,    3u,    10u,   100u,  512u,
                              1000u, 4096u, 8192u, 12345u, 16000u, 16384u,
                              20000u, 25000u, 32767u };
    int32_t mean_code = 0;
    uint32_t dev_code = 0;
    for (const uint32_t v : grid)
        {
            var[0] = static_cast<int16_t> (v);
            REQUIRE (running_stats_deploy_form (stats, 8u, 0u, mean_code, dev_code)
                     == Status::OK);
            const uint64_t root = ref_isqrt (static_cast<uint64_t> (v) << 15u);
            REQUIRE (static_cast<uint64_t> (q15_sqrt_wide (static_cast<int32_t> (v)))
                     == root);
            REQUIRE (dev_code
                     == static_cast<uint32_t> (
                             ref_round_shift (static_cast<int64_t> (root) * 127, 15u)));
            /* root^2 never exceeds the variance it came from */
            REQUIRE (root * root <= static_cast<uint64_t> (v) << 15u);
        }
    /* a negative variance lane has no root: the deviation collapses to zero */
    var[0] = -1000;
    REQUIRE (running_stats_deploy_form (stats, 8u, 0u, mean_code, dev_code)
             == Status::OK);
    REQUIRE (dev_code == 0u);
}

TEST_CASE ("running stats variance saturates toward the positive rail")
{
    int16_t mean[1] = { 0 };
    int16_t var[1] = { 0 };
    RunningStats stats;
    REQUIRE (running_stats_bind (stats, mean, var, 1u) == Status::OK);

    /* a biased variance whose unbiased doubling clamps to the rail */
    const int16_t bv[1] = { 32000 };
    const int16_t bm[1] = { 0 };
    REQUIRE (running_stats_update (stats, bm, bv, 2u, 32767) == Status::OK);
    /* unbiased(32000, 2) = clamp(2*32000) = 32767; lerp toward it reaches it */
    bool sat = false;
    REQUIRE (static_cast<int32_t> (stats._M_var[0])
             == ref_lerp (0, 32767, 32767, sat));
    REQUIRE (stats._M_var[0] <= 32767);
    REQUIRE (ref_unbiased (32000, 2u) == 32767);
}
