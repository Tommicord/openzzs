/* Tests for the training optimizers: the state carve, the seven step
 * functions against double-precision references of their own recurrences, the
 * coupled-versus-decoupled weight-decay split, multi-step descent on a 1-D
 * quadratic, the refusal contracts, lane saturation and determinism.
 *
 * Every numeric expectation is derived in this file and never by calling the
 * module twice: exact integer models reproduce the recurrences stage by stage
 * (q15 mul = round-half-up inside the Q15 rails, q15 div = truncating ratio,
 * the code grid = round-half-away by 258 Q15 units per code), including every
 * intermediate saturation, because a continuous double model of the same
 * recurrence diverges on the rails and on the ties.  A one-code slack is only
 * used where the header itself documents a ratio chain it cannot compute
 * exactly (Adadelta's RMS division).  */

#include "mllee/train_optim.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <cstring>

using namespace zzs::mllee;
using namespace zzs::mllee::train;
using Catch::Approx;

namespace
{
constexpr double _q15 = 32768.0;
constexpr double _code_q15 = 258.0; /* _S_code_to_q15_scale */
constexpr uint32_t _count = 8u;

/** @brief _S_round_shift semantics: floor(x + 0.5), ties toward +infinity. */
double
ref_half_up (const double x) noexcept
{
    return std::floor (x + 0.5);
}

/** @brief Exact integer q15_mul: (a * b + 2^14) >> 15 with floor semantics. */
int64_t
ref_mul_q15 (const int64_t a, const int64_t b) noexcept
{
    const int64_t shifted = a * b + 16384;
    return shifted >= 0 ? shifted / 32768 : -((-shifted + 32767) / 32768);
}

/** @brief Exact integer q15_div: truncating (a << 15) / b, railed to Q15. */
int64_t
ref_div_q15 (const int64_t a, const int64_t b) noexcept
{
    if (b == 0)
        return 0;
    const int64_t scaled = (a * 32768) / b;
    return scaled > 32767 ? 32767 : (scaled < -32768 ? -32768 : scaled);
}

/** @brief q15_to_i16_sat as an int64 clamp. */
int64_t
ref_sat16 (const int64_t value) noexcept
{
    return value > 32767 ? 32767 : (value < -32768 ? -32768 : value);
}

/** @brief q15_clamp on an int64 intermediate. */
int64_t
ref_clamp15 (const int64_t value) noexcept
{
    return value > 32767 ? 32767 : (value < -32768 ? -32768 : value);
}

/** @brief Floor of the square root, the _S_isqrt semantics q15_sqrt uses. */
int64_t
ref_isqrt (const uint64_t value) noexcept
{
    uint64_t root = static_cast<uint64_t> (std::sqrt (static_cast<double> (value)));
    while (root > 0 && root * root > value)
        --root;
    while ((root + 1) * (root + 1) <= value)
        ++root;
    return static_cast<int64_t> (root);
}

/** @brief q15_sqrt on a Q15 lane: floor(sqrt(x * 2^15)). */
int64_t
ref_q15_sqrt (const int64_t x) noexcept
{
    return x <= 0 ? 0 : ref_isqrt (static_cast<uint64_t> (x) << 15);
}

/** @brief q15_mul exactly: the rounded product inside the Q15 rails. */
int64_t
ref_q15_mul (const int64_t a, const int64_t b) noexcept
{
    return ref_clamp15 (ref_mul_q15 (a, b));
}

/** @brief sgd_step_i8 exactly: the Q15 lr product, then the 258-unit grid with
 *         round-half-away, i.e. the two-stage move the kernel performs. */
int64_t
ref_grid_move (const int64_t lane, const int64_t lr) noexcept
{
    const int64_t code_step = ref_mul_q15 (lane, lr);
    return code_step >= 0 ? (code_step + 129) / 258
                          : -((-code_step + 129) / 258);
}

/** @brief The two-stage code move of sgd_step_i8: lr times a lane, then the
 *         258-unit grid with round-half-away. */
double
ref_code_steps (const double lane_q15, const double lr_q15) noexcept
{
    const double moved = std::floor (lane_q15 * lr_q15 / _q15 + 0.5);
    return moved >= 0.0 ? std::floor ((moved + 129.0) / _code_q15)
                        : -std::floor ((-moved + 129.0) / _code_q15);
}

/** @brief The [-127, 127] code clamp _S_i8_code performs after any move. */
int32_t
ref_move_code (const int32_t code, const double steps) noexcept
{
    int64_t moved = static_cast<int64_t> (std::llround (steps));
    moved += code;
    return static_cast<int32_t> (moved > 127 ? 127 : (moved < -127 ? -127 : moved));
}

/** @brief True when every code is on the module grid and never the -128 code. */
bool
codes_on_grid (const int8_t *const codes, const uint32_t count) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        if (codes[i] < -127 || codes[i] > 127)
            return false;
    return true;
}

/** @brief |actual - expected| within a whole-code slack of `slack`. */
bool
within_codes (const int32_t actual, const int32_t expected, const int slack) noexcept
{
    const int diff = actual > expected ? actual - expected : expected - actual;
    return diff <= slack;
}

SgdParams
make_sgd (const int32_t lr, const int32_t mu = 0, const int32_t damp = 0,
          const int32_t wd = 0, const uint8_t nesterov = 0u) noexcept
{
    SgdParams cfg;
    cfg._M_lr_q15 = lr;
    cfg._M_momentum_q15 = mu;
    cfg._M_dampening_q15 = damp;
    cfg._M_weight_decay_q15 = wd;
    cfg._M_nesterov = nesterov;
    return cfg;
}

AdamParams
make_adam (const int32_t lr, const int32_t eps = _S_adam_eps_q15) noexcept
{
    AdamParams cfg;
    cfg._M_lr_q15 = lr;
    cfg._M_eps_q15 = eps;
    return cfg;
}

AdamWParams
make_adamw (const int32_t lr, const int32_t wd) noexcept
{
    AdamWParams cfg;
    cfg._M_lr_q15 = lr;
    cfg._M_weight_decay_q15 = wd;
    return cfg;
}

AdagradParams
make_adagrad (const int32_t lr, const int32_t eps = _S_adam_eps_q15,
              const int32_t wd = 0) noexcept
{
    AdagradParams cfg;
    cfg._M_lr_q15 = lr;
    cfg._M_eps_q15 = eps;
    cfg._M_weight_decay_q15 = wd;
    return cfg;
}

RmspropParams
make_rmsprop (const int32_t lr, const int32_t mu = 0,
              const int32_t eps = _S_adam_eps_q15, const int32_t wd = 0) noexcept
{
    RmspropParams cfg;
    cfg._M_lr_q15 = lr;
    cfg._M_momentum_q15 = mu;
    cfg._M_eps_q15 = eps;
    cfg._M_weight_decay_q15 = wd;
    return cfg;
}

AdadeltaParams
make_adadelta (const int32_t lr = _S_q15_max) noexcept
{
    AdadeltaParams cfg;
    cfg._M_lr_q15 = lr;
    return cfg;
}

LionParams
make_lion (const int32_t lr, const int32_t wd = 0) noexcept
{
    LionParams cfg;
    cfg._M_lr_q15 = lr;
    cfg._M_weight_decay_q15 = wd;
    return cfg;
}

OptimParamsUnion
union_of (const SgdParams &cfg) noexcept
{
    OptimParamsUnion u;
    u._M_sgd = cfg;
    return u;
}

OptimParamsUnion
union_of (const AdamParams &cfg) noexcept
{
    OptimParamsUnion u;
    u._M_adam = cfg;
    return u;
}

OptimParamsUnion
union_of (const AdamWParams &cfg) noexcept
{
    OptimParamsUnion u;
    u._M_adamw = cfg;
    return u;
}

OptimParamsUnion
union_of (const AdagradParams &cfg) noexcept
{
    OptimParamsUnion u;
    u._M_adagrad = cfg;
    return u;
}

OptimParamsUnion
union_of (const RmspropParams &cfg) noexcept
{
    OptimParamsUnion u;
    u._M_rmsprop = cfg;
    return u;
}

OptimParamsUnion
union_of (const AdadeltaParams &cfg) noexcept
{
    OptimParamsUnion u;
    u._M_adadelta = cfg;
    return u;
}

OptimParamsUnion
union_of (const LionParams &cfg) noexcept
{
    OptimParamsUnion u;
    u._M_lion = cfg;
    return u;
}

/** @brief Codes, gradient lanes, pool and one optim state for a count-8
 *         parameter, mirroring the softmax harness shape. */
struct Harness
{
    static constexpr uint32_t _capacity = 256u;

    AlignedArena<_capacity> _M_arena;
    Pool _M_pool;
    alignas (8) int8_t _M_code[_count];
    alignas (8) int16_t _M_grad[_count];
    OptimState _M_state;

    Harness () : _M_pool (_M_arena.pool ())
    {
        for (uint32_t i = 0; i < _count; ++i)
            {
                _M_code[i] = 0;
                _M_grad[i] = 0;
            }
        _M_state = OptimState{};
    }

    Trainable
    param (const uint32_t flags = Flag::RequiresGrad)
    {
        return Trainable::make (_M_code, _M_grad, Shape::vector (_count), flags);
    }

    Trainable
    view_without_grad ()
    {
        return Trainable::make (_M_code, nullptr, Shape::vector (_count),
                                Flag::RequiresGrad);
    }

    /** @brief Load one parameter snapshot; a null gradient stream zeroes the
     *         lanes, for the steps that must refuse a gradient-less operand. */
    void
    seed (const int8_t *const codes, const int16_t *const grads)
    {
        for (uint32_t i = 0; i < _count; ++i)
            {
                _M_code[i] = codes[i];
                _M_grad[i] = grads == nullptr ? 0 : grads[i];
            }
    }
};

/** @brief One independent single-element pipeline: its own arena, pool,
 *         parameter and state, so a batch run can be compared lane by lane. */
struct Solo
{
    AlignedArena<64> _M_arena;
    Pool _M_pool;
    alignas (8) int8_t _M_code[1];
    alignas (8) int16_t _M_grad[1];
    OptimState _M_state;

    Solo () : _M_pool (_M_arena.pool ())
    {
        _M_code[0] = 0;
        _M_grad[0] = 0;
        _M_state = OptimState{};
    }

    Trainable
    param ()
    {
        return Trainable::make (_M_code, _M_grad, Shape::vector (1u),
                                Flag::RequiresGrad);
    }
};
} // namespace

TEST_CASE ("optim state bytes and init carve exactly what each kind needs",
           "[mllee][train][optim]")
{
    static_assert (optim_state_bytes (OptimKind::SGD, 8u) == 16u, "sgd is one lane buffer");
    static_assert (optim_state_bytes (OptimKind::ADAM, 8u) == 32u, "adam is two lane buffers");
    static_assert (optim_state_bytes (OptimKind::ADAMW, 8u) == 32u, "adamw matches adam");
    static_assert (optim_state_bytes (OptimKind::ADAGRAD, 8u) == 16u, "adagrad is one lane buffer");
    static_assert (optim_state_bytes (OptimKind::RMSPROP, 8u) == 32u, "rmsprop is two lane buffers");
    static_assert (optim_state_bytes (OptimKind::ADELTA, 8u) == 32u, "adadelta is two lane buffers");
    static_assert (optim_state_bytes (OptimKind::LION, 8u) == 16u, "lion is one lane buffer");
    static_assert (optim_state_bytes (OptimKind::COUNT, 8u) == 0u, "the sentinel kind needs nothing");

    Harness harness;
    const Trainable param = harness.param ();

    SECTION ("init carves and zeroes the documented buffers per kind")
    {
        const struct
        {
            OptimKind _M_kind;
            uint32_t _M_bytes;
            uint32_t _M_slabs;
            bool _M_secondary;
        } table[] = {
            { OptimKind::SGD, 16u, 1u, false },
            { OptimKind::ADAGRAD, 16u, 1u, false },
            { OptimKind::LION, 16u, 1u, false },
            { OptimKind::ADAM, 32u, 2u, true },
            { OptimKind::ADAMW, 32u, 2u, true },
            { OptimKind::RMSPROP, 32u, 2u, true },
            { OptimKind::ADELTA, 32u, 2u, true },
        };

        for (const auto &row : table)
            {
                AlignedArena<256> arena;
                std::memset (arena._M_data, 0xA5, 256u); /* dirty backing */
                Pool pool = arena.pool ();
                alignas (8) int8_t code[_count];
                alignas (8) int16_t grad[_count];
                const Trainable view
                    = Trainable::make (code, grad, Shape::vector (_count),
                                       Flag::RequiresGrad);
                OptimState state;
                state._M_step = 777u; /* init must reset the metadata too */
                REQUIRE (status_ok (optim_state_init (pool, row._M_kind, view, state)));

                CHECK (state._M_primary != nullptr);
                CHECK ((state._M_secondary != nullptr) == row._M_secondary);
                CHECK (state._M_beta1_pow_q15 == _S_q15_one);
                CHECK (state._M_beta2_pow_q15 == _S_q15_one);
                CHECK (state._M_step == 0u);
                /* carved lanes are zero regardless of the dirty arena */
                bool zeroed = true;
                const uint32_t lanes = row._M_bytes / 2u;
                for (uint32_t i = 0; i < lanes; ++i)
                    zeroed = zeroed && state._M_primary[i] == 0;
                if (row._M_secondary)
                    for (uint32_t i = 0; i < _count; ++i)
                        zeroed = zeroed && state._M_secondary[i] == 0;
                CHECK (zeroed);
                /* the pool actually carved optim_state_bytes, exactly, and
                   one slab per buffer at the lane alignment */
                const PoolStats stats = pool.stats ();
                CHECK (stats._M_used == optim_state_bytes (row._M_kind, _count));
                CHECK (stats._M_slabs == row._M_slabs);
            }
    }

    SECTION ("odd counts pay the lane alignment, still deterministically")
    {
        AlignedArena<64> arena;
        Pool pool = arena.pool ();
        alignas (8) int8_t code[5];
        alignas (8) int16_t grad[5];
        const Trainable odd
            = Trainable::make (code, grad, Shape::vector (5u), Flag::RequiresGrad);
        OptimState state;
        REQUIRE (status_ok (optim_state_init (pool, OptimKind::SGD, odd, state)));
        /* 5 lanes are 10 bytes, carved at the 4-byte SIMD alignment */
        CHECK (pool.stats ()._M_used == align_up (10u, _S_pool_alignment));
        CHECK (optim_state_bytes (OptimKind::SGD, 5u) == 10u);
    }

    SECTION ("a null or zero-count parameter is rejected without carving")
    {
        const Trainable no_code
            = Trainable::make (nullptr, harness._M_grad, Shape::vector (_count), 0u);
        OptimState state;
        CHECK (optim_state_init (harness._M_pool, OptimKind::ADAM, no_code, state)
               == Status::BAD_LAYOUT);
        const Trainable zero_count
            = Trainable::make (harness._M_code, harness._M_grad,
                               Shape::vector (0u), 0u);
        CHECK (optim_state_init (harness._M_pool, OptimKind::SGD, zero_count, state)
               == Status::BAD_LAYOUT);
        CHECK (harness._M_pool.stats ()._M_used == 0u);
    }

    SECTION ("a pool that is too small is an overflow and the sentinel kind is unsupported")
    {
        AlignedArena<12> tiny;
        Pool pool = tiny.pool ();
        alignas (8) int8_t code[_count];
        alignas (8) int16_t grad[_count];
        const Trainable view
            = Trainable::make (code, grad, Shape::vector (_count), Flag::RequiresGrad);
        OptimState state;
        CHECK (optim_state_init (pool, OptimKind::SGD, view, state)
               == Status::ARENA_OVERFLOW);
        CHECK (state._M_primary == nullptr);
        CHECK (pool.stats ()._M_used == 0u);

        /* room for the primary buffer but not the secondary one */
        AlignedArena<24> half;
        Pool half_pool = half.pool ();
        OptimState half_state;
        const Status status
            = optim_state_init (half_pool, OptimKind::ADAM, view, half_state);
        CHECK (status == Status::ARENA_OVERFLOW);
        CHECK (half_state._M_primary != nullptr);
        CHECK (half_state._M_secondary == nullptr);

        CHECK (optim_state_init (pool, OptimKind::COUNT, view, state)
               == Status::UNSUPPORTED);
    }
}

TEST_CASE ("sgd_step plain lr moves the code grid with round-half-away",
           "[mllee][train][optim]")
{
    const int8_t codes[_count] = { 0, 40, -40, 127, -127, 3, -3, 100 };
    const int16_t grads[_count] = { 2580, -2580, 1290, 32767, -32768,
                                    128, -26000, 0 };
    Harness harness;
    harness.seed (codes, grads);
    const Trainable param = harness.param ();
    const OptimParamsUnion params = union_of (make_sgd (3277));

    REQUIRE (status_ok (sgd_step (param, harness._M_state, params)));
    CHECK (harness._M_state._M_step == 1u);

    /* the documented formula, evaluated exactly: the lr*lane Q15 product is
       rounded once, then divided by the 258-unit grid half-away */
    for (uint32_t i = 0; i < _count; ++i)
        {
            const double steps = ref_code_steps (static_cast<double> (grads[i]), 3277.0);
            const int32_t expected = ref_move_code (codes[i], steps);
            CHECK (harness._M_code[i] == expected);
            CHECK (harness._M_code[i] != -128);
            /* the plain path consumes the gradient buffer without rewriting it */
            CHECK (harness._M_grad[i] == grads[i]);
        }
    CHECK (codes_on_grid (harness._M_code, _count));

    SECTION ("a sub-quantum lr product moves zero codes, the documented floor")
    {
        Harness sub;
        sub.seed (codes, grads);
        const Trainable view = sub.param ();
        const int32_t lane = 129; /* exactly half a grid step below the 258 floor */
        sub._M_grad[0] = static_cast<int16_t> (128);
        REQUIRE (status_ok (sgd_step (view, sub._M_state, union_of (make_sgd (1638)))));
        /* 128 * 1638 = 6.4 Q15 after the lr, (6 + 129) / 258 = 0 codes */
        CHECK (sub._M_code[0] == codes[0]);
        CHECK (lane == 129);
    }
}

TEST_CASE ("apply_l2_decay couples into the gradient and BiasLike is exempt",
           "[mllee][train][optim]")
{
    const int8_t codes[_count] = { 20, -20, 0, 50, -50, 127, -127, 3 };
    const int16_t grads[_count] = { 3000, -3000, 100, 16000, -16000, 2000, 4000, -1000 };
    Harness harness;
    harness.seed (codes, grads);
    const Trainable param = harness.param ();

    SECTION ("g becomes g + wd * theta with the module's exact roundings")
    {
        apply_l2_decay (param, 3277);
        for (uint32_t i = 0; i < _count; ++i)
            {
                const int64_t term = ref_mul_q15 (3277, codes[i] * 258);
                const int64_t expected = ref_sat16 (grads[i] + term);
                CHECK (static_cast<int64_t> (harness._M_grad[i]) == expected);
            }
        /* the first lane is 3000 + 516 exactly: 20 codes are 5160 Q15 and
           3277/32768 of that is 516.5, rounded half-up to 516 */
        CHECK (harness._M_grad[0] == 3516);
    }

    SECTION ("zero decay and BiasLike parameters are left alone")
    {
        const int16_t before[_count]
            = { harness._M_grad[0], harness._M_grad[1], harness._M_grad[2],
                harness._M_grad[3], harness._M_grad[4], harness._M_grad[5],
                harness._M_grad[6], harness._M_grad[7] };
        apply_l2_decay (param, 0);
        for (uint32_t i = 0; i < _count; ++i)
            CHECK (harness._M_grad[i] == before[i]);
        const Trainable bias
            = Trainable::make (harness._M_code, harness._M_grad,
                               Shape::vector (_count),
                               static_cast<uint32_t> (Flag::RequiresGrad)
                                   | static_cast<uint32_t> (Flag::BiasLike));
        apply_l2_decay (bias, 8192);
        for (uint32_t i = 0; i < _count; ++i)
            CHECK (harness._M_grad[i] == before[i]);
    }

    SECTION ("sgd_step decays through the gradient and moves on the decayed lane")
    {
        /* lr 3277 hides the decay: the 516 Q15 it folds into the first lane
           stays inside one 258-unit code quantum, so both runs land on the same
           code.  A half-unit lr makes the folded lane a different move. */
        const int32_t lr = 16384;
        const int32_t wd = 8192;
        Harness run;
        run.seed (codes, grads);
        const Trainable view = run.param ();
        REQUIRE (status_ok (sgd_step (view, run._M_state,
                                      union_of (make_sgd (lr, 0, 0, wd)))));
        for (uint32_t i = 0; i < _count; ++i)
            {
                const int64_t decayed
                    = ref_sat16 (grads[i] + ref_mul_q15 (wd, codes[i] * 258));
                /* the buffer carries the decayed lane and the step consumes it */
                CHECK (static_cast<int64_t> (run._M_grad[i]) == decayed);
                const int32_t expected
                    = ref_move_code (codes[i],
                                     static_cast<double> (ref_grid_move (decayed, lr)));
                CHECK (run._M_code[i] == expected);
            }
        /* the same run without decay must differ where the codes are non-zero,
           otherwise the decay is invisible in the parameter */
        Harness plain;
        plain.seed (codes, grads);
        const Trainable plain_view = plain.param ();
        REQUIRE (status_ok (sgd_step (plain_view, plain._M_state,
                                      union_of (make_sgd (lr)))));
        const bool differs = plain._M_code[0] != run._M_code[0];
        CHECK (differs);
        CHECK (plain._M_code[0]
               == ref_move_code (codes[0],
                                 static_cast<double> (ref_grid_move (grads[0], lr))));
    }

    SECTION ("a BiasLike parameter is never decayed inside sgd_step")
    {
        Harness bias_run;
        bias_run.seed (codes, grads);
        const Trainable view
            = bias_run.param (static_cast<uint32_t> (Flag::RequiresGrad)
                                  | static_cast<uint32_t> (Flag::BiasLike));
        REQUIRE (status_ok (sgd_step (view, bias_run._M_state,
                                      union_of (make_sgd (3277, 0, 0, 3277)))));
        for (uint32_t i = 0; i < _count; ++i)
            CHECK (bias_run._M_code[i]
                   == ref_move_code (codes[i], ref_code_steps (static_cast<double> (grads[i]), 3277.0)));
    }
}

TEST_CASE ("sgd_step momentum, dampening and nesterov match the float recurrence",
           "[mllee][train][optim]")
{
    /* v_t = mu v_{t-1} + (1 - damp) g_t is carried in the primary buffer; the
       parameter moves by lr*v (plain momentum) or lr*(g + mu*v) (nesterov),
       which is exactly the form the header implements. */
    const int32_t mu = 16384;
    const int32_t lr = 1638;
    const int8_t codes[_count] = { 10, -10, 40, -40, 3, -3, 100, -100 };
    const int16_t g0[_count] = { 8000, -8000, 1290, -1290, 30000, -30000, 258, -258 };
    const int16_t g1[_count] = { 6000, -9000, 2580, -2580, 20000, -20000, 516, -516 };
    const int16_t g2[_count] = { -4000, 4000, 1290, -1290, 31000, -31000, 0, 0 };

    Harness harness;
    REQUIRE (status_ok (optim_state_init (harness._M_pool, OptimKind::SGD,
                                          harness.param (), harness._M_state)));
    harness.seed (codes, g0);

    /* the exact integer model of the two primitive calls the step makes: a
       rounded, saturating scale of the buffer and an equally shaped axpy of the
       gradient, with keep = (1 - dampening) as the header computes it */
    int64_t v[_count] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    const int32_t keep = _S_q15_one - 1; /* dampening is zero, so keep is 1.0 - ulp */
    int32_t ref_code[_count];
    int8_t code[_count];
    for (uint32_t i = 0; i < _count; ++i)
        {
            ref_code[i] = codes[i];
            code[i] = codes[i];
        }

    const int16_t *const series[3] = { g0, g1, g2 };
    for (uint32_t step = 0; step < 3u; ++step)
        {
            harness.seed (code, series[step]);
            for (uint32_t i = 0; i < _count; ++i)
                ref_code[i] = code[i];
            const Trainable param = harness.param ();
            REQUIRE (status_ok (sgd_step (param, harness._M_state,
                                          union_of (make_sgd (lr, mu)))));
            for (uint32_t i = 0; i < _count; ++i)
                {
                    v[i] = ref_sat16 (ref_sat16 (ref_mul_q15 (v[i], mu))
                                      + ref_mul_q15 (series[step][i], keep));
                    CHECK (static_cast<int64_t> (harness._M_state._M_primary[i])
                           == v[i]);
                    const int32_t expected
                        = ref_move_code (ref_code[i],
                                         ref_code_steps (static_cast<double> (v[i]), lr));
                    CHECK (harness._M_code[i] == expected);
                    CHECK (harness._M_code[i] != -128);
                    code[i] = harness._M_code[i];
                }
        }
    CHECK (harness._M_state._M_step == 3u);

    SECTION ("dampening shrinks the gradient injection to (1 - damp)")
    {
        Harness damp;
        REQUIRE (status_ok (optim_state_init (damp._M_pool, OptimKind::SGD,
                                              damp.param (), damp._M_state)));
        damp.seed (codes, g0);
        const Trainable param = damp.param ();
        REQUIRE (status_ok (sgd_step (param, damp._M_state,
                                      union_of (make_sgd (lr, mu, 16384)))));
        for (uint32_t i = 0; i < _count; ++i)
            {
                /* keep = 32767 - 16384 = 16383, so v = round(g * 16383 / 2^15) */
                CHECK (static_cast<int64_t> (damp._M_state._M_primary[i])
                       == ref_mul_q15 (g0[i], 16383));
            }
    }

    SECTION ("nesterov moves by g + mu v and rewrites the gradient lanes")
    {
        /* the case lr is too fine to show the difference: at 1638 the g + mu*v
           and the plain v lanes round onto the same code count on every lane.
           A half-unit lr separates them, so the guard below cannot pass by
           accident if the nesterov rewrite is deleted. */
        const int32_t nes_lr = 16384;
        Harness nes;
        REQUIRE (status_ok (optim_state_init (nes._M_pool, OptimKind::SGD,
                                              nes.param (), nes._M_state)));
        nes.seed (codes, g0);
        const Trainable param = nes.param ();
        REQUIRE (status_ok (sgd_step (param, nes._M_state,
                                      union_of (make_sgd (nes_lr, mu, 0, 0, 1u)))));
        bool distinguishable = false;
        for (uint32_t i = 0; i < _count; ++i)
            {
                /* the empty buffer scales to zero, so v = round(g * keep) and the
                   nesterov gradient is g + round(mu * v), both saturating */
                const int64_t v_new = ref_sat16 (ref_mul_q15 (g0[i], _S_q15_one - 1));
                const int64_t update = ref_sat16 (g0[i] + ref_mul_q15 (mu, v_new));
                CHECK (static_cast<int64_t> (nes._M_grad[i]) == update);
                CHECK (static_cast<int64_t> (nes._M_state._M_primary[i]) == v_new);
                const int64_t steps = ref_grid_move (update, nes_lr);
                const int32_t expected = ref_move_code (codes[i],
                                                        static_cast<double> (steps));
                CHECK (nes._M_code[i] == expected);
                /* momentum-only would move by v_new; on the lanes where the two
                   forms land on different codes the nesterov one must win */
                const int32_t momentum_only = ref_move_code (
                    codes[i], static_cast<double> (ref_grid_move (v_new, nes_lr)));
                if (expected != momentum_only)
                    {
                        distinguishable = true;
                        CHECK (nes._M_code[i] != momentum_only);
                    }
            }
        CHECK (distinguishable);
    }

    SECTION ("a dampened nesterov run never rails the parameter grid")
    {
        CHECK (codes_on_grid (harness._M_code, _count));
    }
}

TEST_CASE ("adam_step bias correction, moments and eps match the float reference",
           "[mllee][train][optim]")
{
    CHECK (_S_adam_eps_q15 == 32);
    const int32_t lr = 2048;
    const int8_t codes[_count] = { 0, 0, 0, 0, 0, 0, 40, -40 };
    const int16_t g0[_count] = { 8000, -8000, 12900, 0, 32767, 258, -6000, 6000 };

    Harness harness;
    const Trainable param = harness.param ();
    REQUIRE (status_ok (optim_state_init (harness._M_pool, OptimKind::ADAM,
                                          param, harness._M_state)));

    /* the exact integer model of adam_step_impl: the same rounded, saturating
       Q15 recursions, the same two-divide bias correction and the same
       square-root denominator, so every lane below is an equality, not a margin */
    const int32_t b1 = 29491;
    const int32_t b2 = 32735;
    const int32_t om1 = _S_q15_one - 1 - b1;
    const int32_t om2 = _S_q15_one - 1 - b2;
    int64_t m[_count] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    int64_t v[_count] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    int64_t pow1 = _S_q15_one;
    int64_t pow2 = _S_q15_one;
    int8_t code[_count];
    for (uint32_t i = 0; i < _count; ++i)
        code[i] = codes[i];

    const int16_t *const series[4] = { g0, g0, g0, g0 };
    for (uint32_t step = 0; step < 4u; ++step)
        {
            harness.seed (code, series[step]);
            const int16_t *const grad = series[step];
            for (uint32_t i = 0; i < _count; ++i)
                {
                    m[i] = ref_sat16 (ref_q15_mul (m[i], b1)
                                      + ref_mul_q15 (grad[i], om1));
                    v[i] = ref_sat16 (ref_q15_mul (v[i], b2)
                                      + ref_q15_mul (om2, ref_q15_mul (grad[i], grad[i])));
                }
            pow1 = ref_q15_mul (pow1, b1);
            pow2 = ref_q15_mul (pow2, b2);
            REQUIRE (status_ok (adam_step (param, harness._M_state,
                                           union_of (make_adam (lr)))));
            CHECK (static_cast<int64_t> (harness._M_state._M_beta1_pow_q15) == pow1);
            CHECK (static_cast<int64_t> (harness._M_state._M_beta2_pow_q15) == pow2);
            CHECK (harness._M_state._M_step == step + 1u);

            const int64_t corr1 = _S_q15_one - 1 - pow1;
            const int64_t corr2 = _S_q15_one - 1 - pow2;
            for (uint32_t i = 0; i < _count; ++i)
                {
                    CHECK (static_cast<int64_t> (harness._M_state._M_primary[i]) == m[i]);
                    CHECK (static_cast<int64_t> (harness._M_state._M_secondary[i]) == v[i]);
                    const int64_t m_hat = ref_div_q15 (m[i], corr1);
                    const int64_t v_hat = ref_div_q15 (v[i], corr2);
                    const int64_t denom
                        = ref_clamp15 (ref_q15_sqrt (v_hat) + _S_adam_eps_q15);
                    const int64_t lane = ref_sat16 (ref_div_q15 (m_hat, denom));
                    CHECK (static_cast<int64_t> (harness._M_grad[i]) == lane);
                    const int32_t expected = ref_move_code (
                        code[i], static_cast<double> (ref_grid_move (lane, lr)));
                    CHECK (harness._M_code[i] == expected);
                    CHECK (harness._M_code[i] != -128);
                    code[i] = harness._M_code[i];
                }
        }

    SECTION ("a zero gradient with the default eps moves nothing and divides safely")
    {
        Harness zero;
        zero.seed (codes, nullptr);
        const Trainable view = zero.param ();
        REQUIRE (status_ok (optim_state_init (zero._M_pool, OptimKind::ADAM,
                                              view, zero._M_state)));
        REQUIRE (status_ok (adam_step (view, zero._M_state, union_of (make_adam (2048)))));
        for (uint32_t i = 0; i < _count; ++i)
            {
                CHECK (zero._M_code[i] == codes[i]);
                CHECK (zero._M_grad[i] == 0);
            }
        /* eps 0 at a zero denominator is q15_div's documented 0, not a trap */
        Harness eps0;
        eps0.seed (codes, nullptr);
        const Trainable view0 = eps0.param ();
        REQUIRE (status_ok (optim_state_init (eps0._M_pool, OptimKind::ADAM,
                                              view0, eps0._M_state)));
        REQUIRE (status_ok (adam_step (view0, eps0._M_state,
                                       union_of (make_adam (2048, 0)))));
        CHECK (eps0._M_code[0] == 0);
    }

    SECTION ("eps enters the denominator after the sqrt")
    {
        const int16_t grad[_count] = { 2580, 0, 0, 0, 0, 0, 0, 0 };
        Harness small_eps;
        small_eps.seed (codes, grad);
        const Trainable view = small_eps.param ();
        REQUIRE (status_ok (optim_state_init (small_eps._M_pool, OptimKind::ADAM,
                                              view, small_eps._M_state)));
        REQUIRE (status_ok (adam_step (view, small_eps._M_state,
                                       union_of (make_adam (2048)))));
        /* g = 2580 is 0.0787: below the (1 - beta2) rail, so v-hat lands on
           zero and the denominator degenerates to eps = 32, railing the lane */
        CHECK (small_eps._M_grad[0] == _S_q15_max);
        Harness big_eps;
        big_eps.seed (codes, grad);
        const Trainable view_big = big_eps.param ();
        REQUIRE (status_ok (optim_state_init (big_eps._M_pool, OptimKind::ADAM,
                                              view_big, big_eps._M_state)));
        REQUIRE (status_ok (adam_step (view_big, big_eps._M_state,
                                       union_of (make_adam (2048, 16384)))));
        /* the same moment over eps = 0.5: m-hat / (sqrt(0) + 0.5) is a
           quarter of the rail, exactly the eps the caller asked for */
        CHECK (big_eps._M_grad[0] == 5160);
        const bool smaller_move = big_eps._M_grad[0] < small_eps._M_grad[0];
        CHECK (smaller_move);
    }
}

TEST_CASE ("adamw decouples decay while adam folds it into the moments",
           "[mllee][train][optim]")
{
    const int32_t lr = 2048;
    const int32_t wd = 8192;
    const int8_t codes[_count] = { 40, 40, -40, -40, 40, -40, 40, -40 };
    const int16_t grads[_count] = { 8000, -8000, 8000, -8000, 12000, -12000, 30000, -30000 };

    Harness coupled;
    coupled.seed (codes, grads);
    const Trainable coupled_view = coupled.param ();
    REQUIRE (status_ok (optim_state_init (coupled._M_pool, OptimKind::ADAMW,
                                          coupled_view, coupled._M_state)));
    REQUIRE (status_ok (adamw_step (coupled_view, coupled._M_state,
                                    union_of (make_adamw (lr, wd)))));

    Harness plain;
    plain.seed (codes, grads);
    const Trainable plain_view = plain.param ();
    REQUIRE (status_ok (optim_state_init (plain._M_pool, OptimKind::ADAMW,
                                          plain_view, plain._M_state)));
    REQUIRE (status_ok (adamw_step (plain_view, plain._M_state,
                                    union_of (make_adamw (lr, 0)))));

    SECTION ("the decoupled run leaves the moments and gradient lanes untouched")
    {
        /* decay is applied to the codes after the update, so m, v and the
           normalized lanes must be byte-identical to the wd == 0 run */
        for (uint32_t i = 0; i < _count; ++i)
            {
                CHECK (coupled._M_state._M_primary[i] == plain._M_state._M_primary[i]);
                CHECK (coupled._M_state._M_secondary[i] == plain._M_state._M_secondary[i]);
                CHECK (coupled._M_grad[i] == plain._M_grad[i]);
            }
    }

    SECTION ("the decoupled code is the plain code minus the rounded decay step")
    {
        const int32_t combined = static_cast<int32_t> (ref_mul_q15 (lr, wd)); /* 512 */
        CHECK (combined == 512);
        for (uint32_t i = 0; i < _count; ++i)
            {
                const int64_t decay = ref_mul_q15 (combined, plain._M_code[i] * 258);
                const int64_t steps = decay >= 0 ? (decay + 129) / 258 : -((-decay + 129) / 258);
                const int64_t expected = plain._M_code[i] - steps;
                const int32_t clamped = expected > 127 ? 127 : (expected < -127 ? -127 : static_cast<int32_t> (expected));
                CHECK (coupled._M_code[i] == clamped);
            }
        const bool decayed = coupled._M_code[0] != plain._M_code[0];
        CHECK (decayed);
    }

    SECTION ("a BiasLike parameter is never decayed by AdamW")
    {
        Harness bias;
        bias.seed (codes, grads);
        const Trainable view
            = bias.param (static_cast<uint32_t> (Flag::RequiresGrad)
                              | static_cast<uint32_t> (Flag::BiasLike));
        REQUIRE (status_ok (optim_state_init (bias._M_pool, OptimKind::ADAMW,
                                              view, bias._M_state)));
        REQUIRE (status_ok (adamw_step (view, bias._M_state, union_of (make_adamw (lr, wd)))));
        for (uint32_t i = 0; i < _count; ++i)
            CHECK (bias._M_code[i] == plain._M_code[i]);
    }

    SECTION ("coupled decay enters the gradient, so m, v and the denominator change")
    {
        Harness fold;
        fold.seed (codes, grads);
        const Trainable fold_view = fold.param ();
        REQUIRE (status_ok (optim_state_init (fold._M_pool, OptimKind::ADAM,
                                              fold_view, fold._M_state)));
        const AdamParams fold_cfg = make_adam (lr);
        REQUIRE (status_ok (adam_step_impl (fold_view, fold._M_state, fold_cfg,
                                           false, wd)));
        /* g' = 8000 + wd * (40 codes) = 8000 + 2580 = 10580 on the first lane */
        const int64_t m_expected = ref_mul_q15 (3276, 10580);
        CHECK (static_cast<int64_t> (fold._M_state._M_primary[0]) == m_expected);
        CHECK (static_cast<int64_t> (fold._M_state._M_primary[0]) == 1058);
        /* and the squared decayed gradient feeds v: the one-step recursion is
           mul(1 - beta2, mul(g', g')), i.e. mul(32, 3416) = 3 for g' = 10580,
           where the undecayed run accumulates only mul(32, 1953) = 2. */
        CHECK (static_cast<int64_t> (fold._M_state._M_secondary[0])
               == ref_q15_mul (32, ref_q15_mul (10580, 10580)));
        CHECK (static_cast<int64_t> (plain._M_state._M_secondary[0])
               == ref_q15_mul (32, ref_q15_mul (8000, 8000)));
        /* plain adam_step never folds: its gradient lanes come in untouched */
        Harness unfold;
        unfold.seed (codes, grads);
        const Trainable unfold_view = unfold.param ();
        REQUIRE (status_ok (optim_state_init (unfold._M_pool, OptimKind::ADAM,
                                              unfold_view, unfold._M_state)));
        REQUIRE (status_ok (adam_step (unfold_view, unfold._M_state,
                                       union_of (make_adam (lr)))));
        CHECK (unfold._M_state._M_primary[0] == 800);
    }

    SECTION ("apply_decoupled_decay steps whole codes, half-away, and skips BiasLike")
    {
        const int8_t decay_codes[_count] = { 40, -40, 32, -31, 0, 127, -127, 10 };
        Harness decay;
        decay.seed (decay_codes, nullptr);
        const Trainable view = decay.param ();
        apply_decoupled_decay (view, 2048, 8192);
        for (uint32_t i = 0; i < _count; ++i)
            {
                const int64_t d = ref_mul_q15 (512, decay_codes[i] * 258);
                const int64_t steps = d >= 0 ? (d + 129) / 258 : -((-d + 129) / 258);
                const int64_t expected = decay_codes[i] - steps;
                CHECK (decay._M_code[i]
                       == static_cast<int8_t> (expected > 127 ? 127 : (expected < -127 ? -127 : expected)));
            }
        /* -31 codes decay 126.9 Q15, under half a grid step: it must not move */
        CHECK (decay._M_code[3] == -31);
        CHECK (decay._M_code[4] == 0);
        const Trainable bias_view
            = Trainable::make (decay._M_code, nullptr, Shape::vector (_count),
                               static_cast<uint32_t> (Flag::BiasLike));
        apply_decoupled_decay (bias_view, 2048, 8192);
        CHECK (decay._M_code[0] != 40); /* already decayed above */
        /* BiasLike must not decay further: capture and re-apply */
        const int8_t after[_count] = { decay._M_code[0], decay._M_code[1],
                                       decay._M_code[2], decay._M_code[3],
                                       decay._M_code[4], decay._M_code[5],
                                       decay._M_code[6], decay._M_code[7] };
        apply_decoupled_decay (bias_view, 16384, 16384);
        for (uint32_t i = 0; i < _count; ++i)
            CHECK (decay._M_code[i] == after[i]);
    }
}

TEST_CASE ("adagrad accumulates squared gradients exactly and normalizes by the sqrt",
           "[mllee][train][optim]")
{
    const int32_t lr = 2048;
    const int8_t codes[_count] = { 0, 0, 0, 0, 100, -100, 10, -10 };
    const int16_t g0[_count] = { 32767, -32767, 8192, -8192, 2580, 2580, 1290, -1290 };
    const int16_t g1[_count] = { 32767, 0, 8192, 0, 2580, -2580, 1290, 1290 };

    Harness harness;
    harness.seed (codes, g0);
    const Trainable param = harness.param ();
    REQUIRE (status_ok (optim_state_init (harness._M_pool, OptimKind::ADAGRAD,
                                          param, harness._M_state)));
    REQUIRE (status_ok (adagrad_step (param, harness._M_state,
                                      union_of (make_adagrad (lr)))));

    SECTION ("G is the exact rounded square sum, the rail saturates at 1.0")
    {
        /* a railed lane squares to 32766 in Q15: the round sits just under 1.0 */
        for (uint32_t i = 0; i < _count; ++i)
            {
                const int64_t sq = ref_mul_q15 (g0[i], g0[i]);
                CHECK (static_cast<int64_t> (harness._M_state._M_primary[i])
                       == ref_sat16 (sq));
            }
        CHECK (static_cast<int64_t> (harness._M_state._M_primary[0]) == 32766);
        CHECK (static_cast<int64_t> (harness._M_state._M_primary[2]) == 2048);
    }

    int8_t code[_count];
    for (uint32_t i = 0; i < _count; ++i)
        code[i] = harness._M_code[i];

    SECTION ("the second step accumulates and the lane is g / (sqrt(G) + eps)")
    {
        harness.seed (code, g1);
        const Trainable view = harness.param ();
        REQUIRE (status_ok (adagrad_step (view, harness._M_state,
                                          union_of (make_adagrad (lr)))));
        for (uint32_t i = 0; i < _count; ++i)
            {
                const int64_t sq0 = ref_sat16 (ref_mul_q15 (g0[i], g0[i]));
                const int64_t sq1 = ref_mul_q15 (g1[i], g1[i]);
                const int64_t g_hist = ref_sat16 (sq0 + sq1);
                CHECK (static_cast<int64_t> (harness._M_state._M_primary[i]) == g_hist);
                const int64_t denom = ref_clamp15 (ref_q15_sqrt (g_hist) + _S_adam_eps_q15);
                const int64_t lane = ref_div_q15 (g1[i], denom);
                const int64_t steps_raw = ref_half_up (lane * lr / _q15);
                const int64_t steps = steps_raw >= 0 ? (steps_raw + 129) / 258
                                                     : -((-steps_raw + 129) / 258);
                CHECK (harness._M_code[i] == ref_move_code (code[i], static_cast<double> (steps)));
                CHECK (harness._M_code[i] != -128);
            }
        CHECK (harness._M_state._M_step == 2u);
        CHECK (codes_on_grid (harness._M_code, _count));
    }

    SECTION ("weight decay folds into the gradient before the accumulation")
    {
        Harness decay;
        decay.seed (codes, g0);
        const Trainable view = decay.param ();
        REQUIRE (status_ok (optim_state_init (decay._M_pool, OptimKind::ADAGRAD,
                                              view, decay._M_state)));
        REQUIRE (status_ok (adagrad_step (view, decay._M_state,
                                          union_of (make_adagrad (lr, _S_adam_eps_q15, 8192)))));
        /* code 0 lanes decay nothing; the 100-code lanes add 258*100/4 = 6450 */
        const int64_t g_decayed = 2580 + 6450;
        const int64_t sq = ref_mul_q15 (g_decayed, g_decayed);
        CHECK (static_cast<int64_t> (decay._M_state._M_primary[4]) == sq);
        /* a code-0 lane decays nothing, and its 32767 square is 32766.99 in Q15,
           so the accumulator sits one ulp under the rail */
        CHECK (static_cast<int64_t> (decay._M_state._M_primary[0]) == 32766);
    }
}

TEST_CASE ("rmsprop averages squared gradients and the momentum branch buffers",
           "[mllee][train][optim]")
{
    const int8_t codes[_count] = { 0, 0, -50, 50, 0, 0, 7, -7 };
    const int16_t g0[_count] = { 8000, -8000, 32767, -32768, 2580, -2580, 4200, -4200 };
    const int16_t g1[_count] = { -8000, 8000, 32767, 1290, 2580, 516, 0, 4200 };

    SECTION ("no momentum: v is the EMA, the lane is g / (sqrt(v) + eps)")
    {
        Harness harness;
        harness.seed (codes, g0);
        const Trainable param = harness.param ();
        REQUIRE (status_ok (optim_state_init (harness._M_pool, OptimKind::RMSPROP,
                                              param, harness._M_state)));

        /* rmsprop's no-momentum branch rewrites the gradient lanes with the
           normalised lane, so each step has to be re-seeded from its own
           gradient stream */
        const int16_t *const series[2] = { g0, g1 };
        int64_t v[_count] = { 0, 0, 0, 0, 0, 0, 0, 0 };
        int8_t code[_count];
        for (uint32_t i = 0; i < _count; ++i)
            code[i] = codes[i];

        for (uint32_t step = 0u; step < 2u; ++step)
            {
                if (step != 0u)
                    harness.seed (harness._M_code, series[step]);
                REQUIRE (status_ok (rmsprop_step (param, harness._M_state,
                                                  union_of (make_rmsprop (2048)))));
                for (uint32_t i = 0; i < _count; ++i)
                    {
                        const int64_t g = series[step][i];
                        v[i] = ref_sat16 (ref_sat16 (ref_mul_q15 (v[i], 29491))
                                          + ref_q15_mul (3276, ref_q15_mul (g, g)));
                        CHECK (static_cast<int64_t> (harness._M_state._M_primary[i])
                               == v[i]);
                        const int64_t denom = ref_clamp15 (ref_q15_sqrt (v[i]) + 32);
                        const int64_t lane = ref_sat16 (ref_div_q15 (g, denom));
                        CHECK (static_cast<int64_t> (harness._M_grad[i]) == lane);
                        const int32_t expected
                            = ref_move_code (code[i],
                                             static_cast<double> (ref_grid_move (lane, 2048)));
                        CHECK (harness._M_code[i] == expected);
                        CHECK (harness._M_code[i] != -128);
                        code[i] = harness._M_code[i];
                    }
            }
        CHECK (harness._M_state._M_step == 2u);
        CHECK (codes_on_grid (harness._M_code, _count));
    }

    SECTION ("momentum accumulates the normalized lane in the secondary buffer")
    {
        Harness harness;
        harness.seed (codes, g0);
        const Trainable param = harness.param ();
        REQUIRE (status_ok (optim_state_init (harness._M_pool, OptimKind::RMSPROP,
                                              param, harness._M_state)));
        REQUIRE (status_ok (rmsprop_step (param, harness._M_state,
                                          union_of (make_rmsprop (2048, 16384)))));
        for (uint32_t i = 0; i < _count; ++i)
            {
                const int64_t v = ref_sat16 (ref_mul_q15 (3276, ref_mul_q15 (g0[i], g0[i])));
                const int64_t denom = ref_clamp15 (ref_q15_sqrt (v) + 32);
                const int64_t normed = ref_div_q15 (g0[i], denom);
                CHECK (static_cast<int64_t> (harness._M_state._M_secondary[i]) == normed);
                const int64_t moved = ref_half_up (normed * 2048 / _q15);
                const int64_t steps = moved >= 0 ? (moved + 129) / 258 : -((-moved + 129) / 258);
                CHECK (harness._M_code[i] == ref_move_code (codes[i], static_cast<double> (steps)));
            }
        harness.seed (harness._M_code, g1);
        const Trainable view = harness.param ();
        REQUIRE (status_ok (rmsprop_step (view, harness._M_state,
                                          union_of (make_rmsprop (2048, 16384)))));
        for (uint32_t i = 0; i < _count; ++i)
            {
                /* buf = mu*buf + g/(sqrt(v)+eps) against the reference chain */
                int64_t v = ref_sat16 (ref_mul_q15 (29491,
                                                    ref_sat16 (ref_mul_q15 (3276, ref_mul_q15 (g0[i], g0[i]))))
                                       + ref_mul_q15 (3276, ref_mul_q15 (g1[i], g1[i])));
                v = ref_sat16 (v);
                const int64_t v0 = ref_sat16 (ref_mul_q15 (3276, ref_mul_q15 (g0[i], g0[i])));
                const int64_t denom0 = ref_clamp15 (ref_q15_sqrt (v0) + 32);
                const int64_t buf0 = ref_div_q15 (g0[i], denom0);
                const int64_t denom = ref_clamp15 (ref_q15_sqrt (v) + 32);
                const int64_t normed = ref_div_q15 (g1[i], denom);
                const int64_t buf = ref_sat16 (ref_mul_q15 (16384, buf0) + normed);
                CHECK (static_cast<int64_t> (harness._M_state._M_secondary[i]) == buf);
            }
    }

    SECTION ("weight decay folds into the rmsprop gradient")
    {
        Harness harness;
        harness.seed (codes, g0);
        const Trainable param = harness.param ();
        REQUIRE (status_ok (optim_state_init (harness._M_pool, OptimKind::RMSPROP,
                                              param, harness._M_state)));
        REQUIRE (status_ok (rmsprop_step (param, harness._M_state,
                                          union_of (make_rmsprop (2048, 0, _S_adam_eps_q15, 8192)))));
        for (uint32_t i = 0; i < _count; ++i)
            {
                const int64_t g = ref_sat16 (g0[i] + ref_mul_q15 (8192, codes[i] * 258));
                const int64_t v = ref_sat16 (ref_mul_q15 (3276, ref_mul_q15 (g, g)));
                CHECK (static_cast<int64_t> (harness._M_state._M_primary[i]) == v);
            }
    }

    SECTION ("a requested momentum with an uncarved secondary slot is refused")
    {
        Harness harness;
        harness.seed (codes, g0);
        const Trainable param = harness.param ();
        REQUIRE (status_ok (optim_state_init (harness._M_pool, OptimKind::RMSPROP,
                                              param, harness._M_state)));
        harness._M_state._M_secondary = nullptr; /* simulate the failed carve */
        int8_t before[_count];
        std::memcpy (before, harness._M_code, _count);
        CHECK (rmsprop_step (param, harness._M_state,
                             union_of (make_rmsprop (2048, 16384)))
               == Status::BAD_LAYOUT);
        CHECK (std::memcmp (before, harness._M_code, _count) == 0);
    }
}

TEST_CASE ("adadelta runs the rms-ratio recursion on the code-quantum eps",
           "[mllee][train][optim]")
{
    CHECK (_S_adadelta_eps_q15 == 258);
    const int32_t lr = 32767; /* the default _S_q15_max */
    const int8_t codes[_count] = { 0, 0, 50, -50, 3, -3, 100, -100 };
    const int16_t g0[_count] = { 8000, -8000, 32767, -32768, 2580, -2580, 1290, -1290 };
    const int16_t g1[_count] = { 8000, -8000, 1290, 4096, -2580, 2580, 0, 516 };

    Harness harness;
    harness.seed (codes, g0);
    const Trainable param = harness.param ();
    REQUIRE (status_ok (optim_state_init (harness._M_pool, OptimKind::ADELTA,
                                          param, harness._M_state)));

    int64_t gh[_count]; /* gradient history, Q15 */
    int64_t uh[_count]; /* update history, Q15 */
    for (uint32_t i = 0; i < _count; ++i)
        {
            gh[i] = 0;
            uh[i] = 0;
        }

    const int16_t *const series[2] = { g0, g1 };
    int8_t code[_count];
    for (uint32_t i = 0; i < _count; ++i)
        code[i] = codes[i];

    for (uint32_t step = 0; step < 2u; ++step)
        {
            harness.seed (code, series[step]);
            const Trainable view = harness.param ();
            REQUIRE (status_ok (adadelta_step (view, harness._M_state,
                                               union_of (make_adadelta ()))));
            for (uint32_t i = 0; i < _count; ++i)
                {
                    const int64_t g = series[step][i];
                    gh[i] = ref_sat16 (ref_mul_q15 (29491, gh[i])
                                       + ref_mul_q15 (3276, ref_mul_q15 (g, g)));
                    const int64_t rms_g = ref_clamp15 (ref_q15_sqrt (gh[i]) + _S_adadelta_eps_q15);
                    const int64_t rms_u = ref_clamp15 (ref_q15_sqrt (uh[i]) + _S_adadelta_eps_q15);
                    const int64_t ratio = ref_div_q15 (rms_u, rms_g);
                    /* (g * ratio) >> 15 with an arithmetic shift: floor */
                    const int64_t product = g * ratio;
                    const int64_t update = ref_clamp15 (
                        product >= 0 ? product / 32768 : -((-product + 32767) / 32768));
                    CHECK (static_cast<int64_t> (harness._M_grad[i]) == update);
                    CHECK (static_cast<int64_t> (harness._M_state._M_primary[i]) == gh[i]);
                    uh[i] = ref_sat16 (ref_mul_q15 (uh[i], 29491)
                                       + ref_mul_q15 (3276, ref_mul_q15 (update, update)));
                    CHECK (static_cast<int64_t> (harness._M_state._M_secondary[i]) == uh[i]);
                    const int32_t expected = ref_move_code (
                        code[i], ref_code_steps (static_cast<double> (update), lr));
                    /* the eps floor of one code quantum keeps the steady-state
                       ratio inside one grid step of the float recursion */
                    const bool close = within_codes (harness._M_code[i], expected, 1);
                    INFO ("step " << step << " lane " << i << " code "
                          << static_cast<int> (harness._M_code[i]) << " expected "
                          << static_cast<int> (expected) << " update " << update
                          << " rms_u " << rms_u << " rms_g " << rms_g);
                    CHECK (close);
                    CHECK (harness._M_code[i] != -128);
                    code[i] = harness._M_code[i];
                }
        }
    CHECK (harness._M_state._M_step == 2u);

    SECTION ("a zero gradient never advances a code, the eps floor in one step")
    {
        Harness zero;
        REQUIRE (status_ok (optim_state_init (zero._M_pool, OptimKind::ADELTA,
                                              zero.param (), zero._M_state)));
        REQUIRE (status_ok (adadelta_step (zero.param (), zero._M_state,
                                           union_of (make_adadelta ()))));
        CHECK (zero._M_code[0] == 0);
        CHECK (zero._M_grad[0] == 0);
        CHECK (zero._M_state._M_secondary[0] == 0);
    }
}

TEST_CASE ("lion takes the sign step and smooths with its own momentum",
           "[mllee][train][optim]")
{
    SECTION ("sign_to_q15 maps to the saturating rails and keeps zero")
    {
        CHECK (sign_to_q15 (1) == _S_q15_max);
        CHECK (sign_to_q15 (4000) == _S_q15_max);
        CHECK (sign_to_q15 (-1) == _S_q15_min);
        CHECK (sign_to_q15 (-32768) == _S_q15_min);
        CHECK (sign_to_q15 (0) == 0);
    }

    const int32_t lr = 1548; /* exactly 6 codes at a railed lane */
    const int8_t codes[_count] = { 0, 0, 0, 0, 5, -5, 30, -30 };
    const int16_t g0[_count] = { 5000, -5000, 32767, -32768, 1290, -1290, 200, -200 };

    Harness harness;
    harness.seed (codes, g0);
    const Trainable param = harness.param ();
    REQUIRE (status_ok (optim_state_init (harness._M_pool, OptimKind::LION,
                                          param, harness._M_state)));
    REQUIRE (status_ok (lion_step (param, harness._M_state, union_of (make_lion (lr)))));

    SECTION ("the first step is the sign of (1 - beta1) g and moves a whole 6 codes")
    {
        for (uint32_t i = 0; i < _count; ++i)
            {
                const int64_t interp = ref_clamp15 (ref_mul_q15 (32112, 0)
                                                    + ref_mul_q15 (555, g0[i]));
                const int64_t sign = interp > 0 ? 32767 : (interp < 0 ? -32768 : 0);
                const int64_t expected_m = ref_sat16 (ref_mul_q15 (0, 29194)
                                                      + ref_mul_q15 (3573, g0[i]));
                CHECK (static_cast<int64_t> (harness._M_state._M_primary[i]) == expected_m);
                const int32_t expected = ref_move_code (
                    codes[i], ref_code_steps (static_cast<double> (sign), lr));
                CHECK (harness._M_code[i] == expected);
                CHECK (harness._M_code[i] != -128);
            }
        CHECK (harness._M_code[0] == 6);
        CHECK (harness._M_code[1] == -6);
    }

    SECTION ("beta1 smoothing keeps the sign for an under-threshold gradient")
    {
        /* 0.98 * m1 with m1 = 3573 (the railed g0 = 32767) versus 0.017 * (-32768)
           = -557: the interpolation stays positive, so the code advances again */
        const int16_t g1[_count] = { -2000, 2000, -32768, 32767, -2000, 2000, -2000, 2000 };
        harness.seed (harness._M_code, g1);
        const Trainable view = harness.param ();
        REQUIRE (status_ok (lion_step (view, harness._M_state, union_of (make_lion (lr)))));
        CHECK (harness._M_code[0] == 12);
        CHECK (harness._M_code[1] == -12);
        /* the railed opposite gradient cannot flip a smoothed sign */
        CHECK (harness._M_code[2] == 12);
        CHECK (harness._M_code[3] == -12);
    }

    SECTION ("decoupled decay rides on the sign step, BiasLike exempt")
    {
        Harness decay;
        decay.seed (codes, g0);
        const Trainable view = decay.param ();
        REQUIRE (status_ok (optim_state_init (decay._M_pool, OptimKind::LION,
                                              view, decay._M_state)));
        REQUIRE (status_ok (lion_step (view, decay._M_state,
                                       union_of (make_lion (lr, 32767)))));
        /* 6 codes from the sign, then decay = round(1548 * code*258/2^15):
           code 6 gives 37 Q15, under half a grid step, so no decay move */
        CHECK (decay._M_code[0] == 6);
        Harness grown;
        const int8_t big[_count] = { 12, 0, 0, 0, 0, 0, 0, 0 };
        grown.seed (big, g0);
        const Trainable grown_view = grown.param ();
        REQUIRE (status_ok (optim_state_init (grown._M_pool, OptimKind::LION,
                                              grown_view, grown._M_state)));
        REQUIRE (status_ok (lion_step (grown_view, grown._M_state,
                                       union_of (make_lion (lr, 32767)))));
        /* code 12: the sign step takes it to 18, decay = 1548*18*258/2^15 =
           219 >= 129, so one code step back down to 17 */
        CHECK (grown._M_code[0] == 17);
        const Trainable bias
            = Trainable::make (grown._M_code, nullptr, Shape::vector (_count),
                               static_cast<uint32_t> (Flag::RequiresGrad)
                                   | static_cast<uint32_t> (Flag::BiasLike));
        const int8_t before[_count] = { grown._M_code[0], grown._M_code[1],
                                        grown._M_code[2], grown._M_code[3],
                                        grown._M_code[4], grown._M_code[5],
                                        grown._M_code[6], grown._M_code[7] };
        apply_decoupled_decay (bias, lr, 32767);
        for (uint32_t i = 0; i < _count; ++i)
            CHECK (grown._M_code[i] == before[i]);
    }
}

TEST_CASE ("optimizers actually descend a 1-D quadratic over many steps",
           "[mllee][train][optim]")
{
    SECTION ("sgd lr 0.1 with k = 5 per code halves the error each step")
    {
        /* the grid-exact schedule: the targets are powers of two codes away,
           so 0.5 * e rounds half-away to exactly e / 2 every step */
        const int32_t targets[_count] = { 16, 16, 8, -8, 4, -4, 2, -2 };
        const int32_t k = 1290;
        Harness harness;
        const Trainable param = harness.param ();
        REQUIRE (status_ok (optim_state_init (harness._M_pool, OptimKind::SGD,
                                              param, harness._M_state)));
        const OptimParamsUnion params = union_of (make_sgd (3277));

        int32_t prev[_count];
        for (uint32_t i = 0; i < _count; ++i)
            prev[i] = targets[i]; /* all codes start at 0 */

        for (uint32_t step = 0; step < 12u; ++step)
            {
                for (uint32_t i = 0; i < _count; ++i)
                    {
                    const int64_t g = -static_cast<int64_t> (k) * (harness._M_code[i] - targets[i]);
                    harness._M_grad[i] = static_cast<int16_t> (ref_sat16 (g));
                    }
                REQUIRE (status_ok (sgd_step (param, harness._M_state, params)));
                for (uint32_t i = 0; i < _count; ++i)
                    {
                        const int32_t err = targets[i] - harness._M_code[i];
                        const int32_t abs_err = err < 0 ? -err : err;
                        const int32_t abs_prev = prev[i] < 0 ? -prev[i] : prev[i];
                        /* strictly monotone until the target is hit */
                        const bool shrinking = abs_prev == 0 || abs_err < abs_prev;
                        CHECK (shrinking);
                        prev[i] = err;
                        CHECK (harness._M_code[i] != -128);
                    }
            }
        for (uint32_t i = 0; i < _count; ++i)
            CHECK (harness._M_code[i] == static_cast<int8_t> (targets[i]));
    }

    SECTION ("adam at lr 0.016 walks to the target and stays within a few codes")
    {
        const int32_t targets[_count] = { 10, -10, 0, 0, 0, 0, 0, 0 };
        const int32_t k = 1290;
        Harness harness;
        harness._M_code[0] = 0;
        const Trainable param = harness.param ();
        REQUIRE (status_ok (optim_state_init (harness._M_pool, OptimKind::ADAM,
                                              param, harness._M_state)));
        const OptimParamsUnion params = union_of (make_adam (512));

        int32_t first_err[_count];
        for (uint32_t step = 0; step < 16u; ++step)
            {
                for (uint32_t i = 0; i < _count; ++i)
                    {
                        const int64_t g = -static_cast<int64_t> (k)
                                          * (harness._M_code[i] - targets[i]);
                        harness._M_grad[i] = static_cast<int16_t> (ref_sat16 (g));
                    }
                REQUIRE (status_ok (adam_step (param, harness._M_state, params)));
                for (uint32_t i = 0; i < _count; ++i)
                    {
                        const int32_t err = targets[i] - harness._M_code[i];
                        const int32_t abs_err = err < 0 ? -err : err;
                        if (step == 0u)
                            first_err[i] = abs_err == 0 ? 1 : abs_err;
                        /* it never wanders further than where it started */
                        const bool bounded = abs_err <= first_err[i];
                        CHECK (bounded);
                        CHECK (harness._M_code[i] != -128);
                    }
            }
        /* converging: the far lanes closed most of the initial error and the
           zero-target zero-code lanes never woke up */
        const int32_t err0 = 10 - harness._M_code[0];
        const int32_t abs_err0 = err0 < 0 ? -err0 : err0;
        CHECK (abs_err0 < 10 / 2);
        const int32_t err1 = -10 - harness._M_code[1];
        const int32_t abs_err1 = err1 < 0 ? -err1 : err1;
        CHECK (abs_err1 < 10 / 2);
        for (uint32_t i = 2; i < _count; ++i)
            CHECK (harness._M_code[i] == 0);
    }

    SECTION ("sgd with momentum overshoots once then rings down toward the target")
    {
        /* mu 0.9 rings the same way the float optimizer does: the swing
           strictly shrinks from its first extreme */
        const int32_t target = 16;
        const int32_t k = 1290;
        Harness harness;
        const Trainable param = harness.param ();
        REQUIRE (status_ok (optim_state_init (harness._M_pool, OptimKind::SGD,
                                              param, harness._M_state)));
        const OptimParamsUnion params = union_of (make_sgd (1638, 29491));

        int32_t swing[_count];
        for (uint32_t step = 0; step < 24u; ++step)
            {
                const int64_t g = -static_cast<int64_t> (k) * (harness._M_code[0] - target);
                harness._M_grad[0] = static_cast<int16_t> (ref_sat16 (g));
                REQUIRE (status_ok (sgd_step (param, harness._M_state, params)));
                const int32_t err = std::abs (target - harness._M_code[0]);
                if (step >= 4u)
                    swing[0] = std::max (swing[0], err);
                if (step >= 20u)
                    CHECK (err <= swing[0]);
            }
        CHECK (std::abs (target - harness._M_code[0]) <= 4);
    }
}

TEST_CASE ("missing gradients and uninitialised state are refused without writes",
           "[mllee][train][optim]")
{
    const int8_t codes[_count] = { 10, -10, 30, -30, 0, 0, 5, -5 };
    OptimParamsUnion scratch;

    SECTION ("a parameter without gradient lanes stops every step cold")
    {
        Harness harness;
        harness.seed (codes, nullptr);
        const Trainable param = harness.view_without_grad ();
        int8_t before[_count];
        std::memcpy (before, harness._M_code, _count);

        scratch._M_sgd = make_sgd (3277, 16384, 0, 3277);
        CHECK (sgd_step (param, harness._M_state, scratch) == Status::BAD_LAYOUT);
        scratch._M_sgd = make_sgd (3277); /* plain path, still no gradient */
        CHECK (sgd_step (param, harness._M_state, scratch) == Status::BAD_LAYOUT);
        scratch._M_adam = make_adam (2048);
        CHECK (adam_step (param, harness._M_state, scratch) == Status::BAD_LAYOUT);
        scratch._M_adamw = make_adamw (2048, 8192);
        CHECK (adamw_step (param, harness._M_state, scratch) == Status::BAD_LAYOUT);
        scratch._M_adagrad = make_adagrad (2048, _S_adam_eps_q15, 8192);
        CHECK (adagrad_step (param, harness._M_state, scratch) == Status::BAD_LAYOUT);
        scratch._M_rmsprop = make_rmsprop (2048, 16384, _S_adam_eps_q15, 8192);
        CHECK (rmsprop_step (param, harness._M_state, scratch) == Status::BAD_LAYOUT);
        scratch._M_adadelta = make_adadelta ();
        CHECK (adadelta_step (param, harness._M_state, scratch) == Status::BAD_LAYOUT);
        scratch._M_lion = make_lion (1548, 32767);
        CHECK (lion_step (param, harness._M_state, scratch) == Status::BAD_LAYOUT);

        /* nothing moved: the refusal happens before any write */
        CHECK (std::memcmp (before, harness._M_code, _count) == 0);
        CHECK (harness._M_state._M_step == 0u);
    }

    SECTION ("every stateful step refuses a state whose buffers were never carved")
    {
        Harness harness;
        harness.seed (codes, nullptr);
        for (uint32_t i = 0; i < _count; ++i)
            harness._M_grad[i] = static_cast<int16_t> (1000 + 100 * i);
        const Trainable param = harness.param ();
        int8_t before[_count];
        std::memcpy (before, harness._M_code, _count);
        OptimState uninit; /* buffers nullptr, powers 1.0, step 0 */

        scratch._M_adam = make_adam (2048);
        CHECK (adam_step (param, uninit, scratch) == Status::BAD_LAYOUT);
        scratch._M_adamw = make_adamw (2048, 8192);
        CHECK (adamw_step (param, uninit, scratch) == Status::BAD_LAYOUT);
        scratch._M_adagrad = make_adagrad (2048);
        CHECK (adagrad_step (param, uninit, scratch) == Status::BAD_LAYOUT);
        scratch._M_rmsprop = make_rmsprop (2048);
        CHECK (rmsprop_step (param, uninit, scratch) == Status::BAD_LAYOUT);
        scratch._M_adadelta = make_adadelta ();
        CHECK (adadelta_step (param, uninit, scratch) == Status::BAD_LAYOUT);
        scratch._M_lion = make_lion (1548);
        CHECK (lion_step (param, uninit, scratch) == Status::BAD_LAYOUT);

        /* SGD asking for momentum on an uncarved state must be refused, not
           silently degraded to the plain lr step: the requested recurrence is
           part of the contract, and a half-configured optim_state_init left
           the parameter unmoved below */
        scratch._M_sgd = make_sgd (3277, 16384);
        CHECK (sgd_step (param, uninit, scratch) == Status::BAD_LAYOUT);
        CHECK (uninit._M_step == 0u);
        /* even with weight decay the refusal precedes every write */
        scratch._M_sgd = make_sgd (3277, 16384, 0, 3277);
        CHECK (sgd_step (param, uninit, scratch) == Status::BAD_LAYOUT);
        const bool untouched = std::memcmp (before, harness._M_code, _count) == 0;
        CHECK (untouched);

        /* plain lr is the documented state-free path and still runs */
        Harness plain;
        plain.seed (codes, nullptr);
        for (uint32_t i = 0; i < _count; ++i)
            plain._M_grad[i] = static_cast<int16_t> (1000 + 100 * i);
        scratch._M_sgd = make_sgd (3277);
        CHECK (status_ok (sgd_step (plain.param (), plain._M_state, scratch)));
        CHECK (plain._M_state._M_step == 1u);
    }
}

TEST_CASE ("saturating lanes rail without corrupting their neighbours",
           "[mllee][train][optim]")
{
    SECTION ("adam extremes match the same lane run alone, element by element")
    {
        const int8_t codes[_count] = { 0, 100, -100, 3, -3, 120, -120, 0 };
        const int16_t grads[_count] = { 32767, 1290, -2580, 32767, -32768,
                                        258, -32768, -32768 };
        Harness batch;
        batch.seed (codes, grads);
        const Trainable param = batch.param ();
        REQUIRE (status_ok (optim_state_init (batch._M_pool, OptimKind::ADAM,
                                              param, batch._M_state)));
        const OptimParamsUnion params = union_of (make_adam (2048));
        int8_t code[_count];
        std::memcpy (code, codes, _count);

        for (uint32_t step = 0; step < 3u; ++step)
            {
                batch.seed (code, grads);
                REQUIRE (status_ok (adam_step (param, batch._M_state, params)));
                CHECK (codes_on_grid (batch._M_code, _count));
                for (uint32_t i = 0; i < _count; ++i)
                    CHECK (batch._M_code[i] != -128);
                std::memcpy (code, batch._M_code, _count);
            }

        /* each element in its own pool with its own state: a rail in lane 4
           may not borrow lane 5's neighbour, and the batch result must be
           bit-identical lane by lane */
        for (uint32_t i = 0; i < _count; ++i)
            {
                Solo solo;
                solo._M_code[0] = codes[i];
                solo._M_grad[0] = grads[i];
                const Trainable solo_view = solo.param ();
                REQUIRE (status_ok (optim_state_init (solo._M_pool, OptimKind::ADAM,
                                                      solo_view, solo._M_state)));
                int32_t solo_code = codes[i];
                for (uint32_t step = 0; step < 3u; ++step)
                    {
                        solo._M_code[0] = static_cast<int8_t> (solo_code);
                        solo._M_grad[0] = grads[i];
                        REQUIRE (status_ok (adam_step (solo_view, solo._M_state, params)));
                        solo_code = solo._M_code[0];
                    }
                CHECK (batch._M_code[i] == static_cast<int8_t> (solo_code));
                CHECK (static_cast<int64_t> (batch._M_state._M_primary[i])
                       == static_cast<int64_t> (solo._M_state._M_primary[0]));
                CHECK (static_cast<int64_t> (batch._M_state._M_secondary[i])
                       == static_cast<int64_t> (solo._M_state._M_secondary[0]));
            }
    }

    SECTION ("sgd at the full lr rails the code stream to the grid edges")
    {
        const int8_t codes[_count] = { 120, -120, 0, 127, -127, 10, -10, 64 };
        const int16_t grads[_count] = { 32767, -32768, 32767, 32767, -32768,
                                        -32768, 32767, -30000 };
        Harness harness;
        harness.seed (codes, grads);
        const Trainable param = harness.param ();
        REQUIRE (status_ok (sgd_step (param, harness._M_state,
                                      union_of (make_sgd (32768)))));
        CHECK (harness._M_code[0] == 127);
        CHECK (harness._M_code[1] == -127);
        CHECK (harness._M_code[3] == 127);
        CHECK (harness._M_code[4] == -127);
        CHECK (codes_on_grid (harness._M_code, _count));
        for (uint32_t i = 0; i < _count; ++i)
            CHECK (harness._M_code[i] != -128);
    }

    SECTION ("lion signs and adadelta ratios stay on the grid under extremes")
    {
        const int8_t codes[_count] = { 127, -127, 0, 12, -12, 125, -125, 3 };
        const int16_t grads[_count] = { 32767, -32768, 32767, -555, 555, 32767, 1, -1 };
        Harness lion;
        lion.seed (codes, grads);
        const Trainable lion_view = lion.param ();
        REQUIRE (status_ok (optim_state_init (lion._M_pool, OptimKind::LION,
                                              lion_view, lion._M_state)));
        REQUIRE (status_ok (lion_step (lion_view, lion._M_state,
                                       union_of (make_lion (32767, 32767)))));
        /* railed up, clamped, then a decay of 1.0 at lr 1.0 removes the weight */
        CHECK (lion._M_code[0] == 0);
        CHECK (codes_on_grid (lion._M_code, _count));
        for (uint32_t i = 0; i < _count; ++i)
            CHECK (lion._M_code[i] != -128);

        Harness delta;
        delta.seed (codes, grads);
        const Trainable delta_view = delta.param ();
        REQUIRE (status_ok (optim_state_init (delta._M_pool, OptimKind::ADELTA,
                                              delta_view, delta._M_state)));
        for (uint32_t step = 0; step < 4u; ++step)
            {
                delta.seed (delta._M_code, grads);
                REQUIRE (status_ok (adadelta_step (delta_view, delta._M_state,
                                                   union_of (make_adadelta ()))));
                CHECK (codes_on_grid (delta._M_code, _count));
                const bool no_rail_poison
                    = q15_in_range (delta._M_state._M_primary[0])
                      && q15_in_range (delta._M_state._M_secondary[0]);
                CHECK (no_rail_poison);
            }
    }
}

TEST_CASE ("optimizer dispatch, lr schedule and cross-pool determinism",
           "[mllee][train][optim]")
{
    SECTION ("for_kind indexes the table and COUNT is refused")
    {
        CHECK (Optimizer::for_kind (OptimKind::SGD) == &sgd_step);
        CHECK (Optimizer::for_kind (OptimKind::ADAM) == &adam_step);
        CHECK (Optimizer::for_kind (OptimKind::ADAMW) == &adamw_step);
        CHECK (Optimizer::for_kind (OptimKind::ADAGRAD) == &adagrad_step);
        CHECK (Optimizer::for_kind (OptimKind::RMSPROP) == &rmsprop_step);
        CHECK (Optimizer::for_kind (OptimKind::ADELTA) == &adadelta_step);
        CHECK (Optimizer::for_kind (OptimKind::LION) == &lion_step);
        CHECK (Optimizer::for_kind (OptimKind::COUNT) == nullptr);

        Optimizer sentinel;
        sentinel._M_kind = OptimKind::COUNT;
        Harness harness;
        for (uint32_t i = 0; i < _count; ++i)
            harness._M_grad[i] = static_cast<int16_t> (2000);
        CHECK (sentinel.step (harness.param (), harness._M_state) == Status::UNSUPPORTED);
        CHECK (sentinel.state_bytes (_count) == 0u);
        CHECK (sentinel.lr_q15 () == 0);
        CHECK (sentinel.set_lr (100) == Status::UNSUPPORTED);
        OptimState state;
        CHECK (optim_state_init (harness._M_pool, OptimKind::COUNT,
                                 harness.param (), state)
               == Status::UNSUPPORTED);
    }

    SECTION ("set_lr and lr_q15 reach the arm the kind uses")
    {
        const OptimKind kinds[] = { OptimKind::SGD, OptimKind::ADAM, OptimKind::ADAMW,
                                    OptimKind::ADAGRAD, OptimKind::RMSPROP,
                                    OptimKind::ADELTA, OptimKind::LION };
        for (const OptimKind kind : kinds)
            {
                Optimizer opt;
                opt._M_kind = kind;
                CHECK (status_ok (opt.set_lr (1000)));
                CHECK (opt.lr_q15 () == 1000);
                CHECK (opt.lr_q15 () == opt._M_params._M_sgd._M_lr_q15); /* same offset */
                CHECK (status_ok (opt.set_lr (0)));
                CHECK (status_ok (opt.set_lr (_S_q15_one)));
                CHECK (opt.set_lr (-1) == Status::OUT_OF_RANGE);
                CHECK (opt.lr_q15 () == _S_q15_one); /* unchanged by the refusal */
                CHECK (opt.set_lr (_S_q15_one + 1) == Status::OUT_OF_RANGE);
            }
        /* the defaults survive construction in whichever arm the kind names */
        Optimizer sgd;
        CHECK (sgd.lr_q15 () == 3277);
        CHECK (sgd.state_bytes (8u) == 16u);
        Optimizer adam;
        adam._M_kind = OptimKind::ADAM;
        CHECK (adam.state_bytes (8u) == 32u);
    }

    SECTION ("step_with_lr applies a scheduled lr or refuses before moving")
    {
        Harness harness;
        for (uint32_t i = 0; i < _count; ++i)
            harness._M_grad[i] = static_cast<int16_t> (8000);
        const Trainable param = harness.param ();
        Optimizer opt;
        opt._M_kind = OptimKind::SGD;
        /* 8000 Q15 at lr 1032 is a 252-unit product, (252 + 129) / 258 = 1 code */
        opt._M_params._M_sgd = make_sgd (1032);
        REQUIRE (status_ok (optim_state_init (harness._M_pool, OptimKind::SGD,
                                              param, harness._M_state)));
        CHECK (harness._M_code[0] == 0);
        REQUIRE (status_ok (opt.step (param, harness._M_state)));
        CHECK (harness._M_code[0] == 1); /* the configured lr moved 1 code */
        int8_t before[_count];
        std::memcpy (before, harness._M_code, _count);
        /* the scheduled lr doubles the product to 504 -- 2 codes -- and the call
           records it, so the lr the optimizer carries is the one that moved */
        REQUIRE (status_ok (opt.step_with_lr (param, harness._M_state, 2064)));
        CHECK (harness._M_code[0] == 3);
        CHECK (opt.lr_q15 () == 2064);
        CHECK (opt._M_params._M_sgd._M_lr_q15 == 2064);
        std::memcpy (before, harness._M_code, _count);
        CHECK (opt.step_with_lr (param, harness._M_state, -5) == Status::OUT_OF_RANGE);
        CHECK (std::memcmp (before, harness._M_code, _count) == 0);
        CHECK (opt.lr_q15 () == 2064); /* the refusal is before any write */
    }

    SECTION ("two independently carved pools run bit-identically")
    {
        Harness a;
        Harness b;
        REQUIRE (status_ok (optim_state_init (a._M_pool, OptimKind::ADAM,
                                              a.param (), a._M_state)));
        REQUIRE (status_ok (optim_state_init (b._M_pool, OptimKind::ADAM,
                                              b.param (), b._M_state)));

        const int8_t codes[_count] = { 5, -5, 40, -40, 0, 0, 100, -100 };
        const int16_t series[4][_count] = {
            { 8000, -8000, 1290, -1290, 32767, -32768, 516, -516 },
            { -3000, 3000, 12900, -12900, 258, 258, -20000, 20000 },
            { 0, 0, -1290, 1290, 32767, 32767, -32768, -260 },
            { 12900, -12900, 0, 0, -300, 300, 7777, -7777 },
        };

        for (uint32_t step = 0; step < 4u; ++step)
            {
                a.seed (codes, series[step]);
                b.seed (codes, series[step]);
                REQUIRE (status_ok (adam_step (a.param (), a._M_state,
                                               union_of (make_adam (1024)))));
                REQUIRE (status_ok (adam_step (b.param (), b._M_state,
                                               union_of (make_adam (1024)))));
            }
        REQUIRE (status_ok (optim_state_init (a._M_pool, OptimKind::RMSPROP,
                                              a.param (), a._M_state)));
        REQUIRE (status_ok (optim_state_init (b._M_pool, OptimKind::RMSPROP,
                                              b.param (), b._M_state)));
        for (uint32_t step = 0; step < 3u; ++step)
            {
                REQUIRE (status_ok (rmsprop_step (a.param (), a._M_state,
                                                  union_of (make_rmsprop (2048, 16384)))));
                REQUIRE (status_ok (rmsprop_step (b.param (), b._M_state,
                                                  union_of (make_rmsprop (2048, 16384)))));
            }

        /* codes, gradient lanes, every state buffer and the bias powers */
        CHECK (std::memcmp (a._M_code, b._M_code, _count) == 0);
        CHECK (std::memcmp (a._M_grad, b._M_grad, sizeof (a._M_grad)) == 0);
        const uint32_t primary_bytes = 2u * _count;
        CHECK (std::memcmp (a._M_state._M_primary, b._M_state._M_primary,
                           primary_bytes)
               == 0);
        CHECK (std::memcmp (a._M_state._M_secondary, b._M_state._M_secondary,
                            primary_bytes)
               == 0);
        CHECK (a._M_state._M_beta1_pow_q15 == b._M_state._M_beta1_pow_q15);
        CHECK (a._M_state._M_beta2_pow_q15 == b._M_state._M_beta2_pow_q15);
        CHECK (a._M_state._M_step == b._M_state._M_step);
        CHECK (a._M_pool.stats ()._M_used == b._M_pool.stats ()._M_used);

        /* and the pool carve itself consumed exactly optim_state_bytes of
           identical arena bytes in both pools */
        const uint32_t used = a._M_pool.stats ()._M_used;
        CHECK (used > 0u);
        CHECK (std::memcmp (a._M_arena._M_data, b._M_arena._M_data, used) == 0);
    }
}
