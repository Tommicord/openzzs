/* Tests for the training random engines, the weight initializers and the
 * inverted dropout op.
 *
 * Every numeric expectation below is computed from an independent reference
 * written inside this file: the splitmix64 and xorshift64* streams are
 * re-derived from the published algorithms, the fixed-point helpers are
 * re-derived from the rounding rules documented in the headers (round half
 * up toward +infinity, half-away-from-zero code rounding, the /258 code step,
 * the saturating int8 rails), and the mask / forward / backward streams are
 * replayed from a reference engine whose state the test owns.  No expectation
 * is ever read back from the library implementation. */

#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_dropout.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_init.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_random.hpp"
#include "mllee/train_tensor.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>
#include <cstdint>

using namespace zzs::mllee;
using namespace zzs::mllee::train;
using Catch::Approx;

namespace
{
/* ------------------------------------------------------------------ *
 * Independent references: published algorithms, retyped here.         *
 * ------------------------------------------------------------------ */

constexpr uint64_t K_increment = 0x9E3779B97F4A7C15ull;
constexpr uint64_t K_mixer_a = 0xBF58476D1CE4E5B9ull;
constexpr uint64_t K_mixer_b = 0x94D049BB133111EBull;
constexpr uint64_t K_xorshift_mult = 0x2545F4914F6CDD1Dull;

/* The published splitmix64. */
uint64_t
ref_splitmix64 (uint64_t &state)
{
    state += K_increment;
    uint64_t z = state;
    z = (z ^ (z >> 30u)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27u)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31u);
}

/* The published xorshift64* (Vigna). Zero is replaced on entry exactly as
   the header documents, so the reference and the library start from the same
   state on a zero seed; the replacement constant is documented on XorShift64. */
uint64_t
ref_xorshift64star (uint64_t &state)
{
    if (state == 0ull)
        state = K_increment;
    state ^= state >> 12u;
    state ^= state << 25u;
    state ^= state >> 27u;
    return state * K_xorshift_mult;
}

struct RefMix
{
    uint64_t _M_state;

    uint64_t
    next_u64 ()
    {
        return ref_splitmix64 (_M_state);
    }

    uint32_t
    next_u32 ()
    {
        return static_cast<uint32_t> (next_u64 () >> 32u);
    }
};

struct RefXor
{
    uint64_t _M_state;

    uint64_t
    next_u64 ()
    {
        return ref_xorshift64star (_M_state);
    }

    uint32_t
    next_u32 ()
    {
        return static_cast<uint32_t> (next_u64 () >> 32u);
    }
};

/* One splitmix64 round over base ^ (index * increment), per the documented
   derivation. */
uint64_t
ref_derive_seed (const uint64_t base, const uint32_t stream)
{
    uint64_t state = base ^ (static_cast<uint64_t> (stream) * K_increment);
    return ref_splitmix64 (state);
}

/* The published multiply-shift (Lemire fastrange): floor (u32 * bound / 2^32),
   constant time, exclusive upper bound. */
template <typename RefEngine>
uint32_t
ref_bounded_uniform (RefEngine &engine, const uint32_t bound)
{
    if (bound == 0u)
        return 0u;
    const uint64_t product
        = static_cast<uint64_t> (engine.next_u32 ()) * static_cast<uint64_t> (bound);
    return static_cast<uint32_t> (product >> 32u);
}

/* The high 16 bits read whole as an int16 lane: exact over [-32768, 32767]. */
template <typename RefEngine>
int32_t
ref_uniform_q15 (RefEngine &engine)
{
    const uint64_t raw = engine.next_u64 ();
    return static_cast<int32_t> (static_cast<int16_t> (
        static_cast<uint16_t> (raw >> 48u)));
}

template <typename RefEngine>
int32_t
ref_uniform_positive_q15 (RefEngine &engine)
{
    const uint64_t raw = engine.next_u64 ();
    return static_cast<int32_t> (static_cast<uint32_t> (raw >> 48u) & 0x7FFFu);
}

/* Fifteen-bit draw in [0, 32767], compared against p: kept iff draw < p, so
   p is an exclusive threshold. p <= 0 and p >= 2^15 consume nothing. */
template <typename RefEngine>
bool
ref_bernoulli_q15 (RefEngine &engine, const int32_t p_q15)
{
    if (p_q15 <= 0)
        return false;
    if (p_q15 >= 32768)
        return true;
    const uint64_t raw = engine.next_u64 ();
    const int32_t draw = static_cast<int32_t> (static_cast<uint32_t> (raw >> 49u));
    return draw < p_q15;
}

template <typename RefEngine>
uint32_t
ref_fill_mask (RefEngine &engine, const int32_t keep_q15, uint8_t *mask,
               const uint32_t count)
{
    uint32_t kept = 0u;
    for (uint32_t i = 0u; i < count; ++i)
        {
            const uint8_t bit = ref_bernoulli_q15 (engine, keep_q15) ? 1u : 0u;
            mask[i] = bit;
            kept += bit;
        }
    return kept;
}

template <typename RefEngine>
uint32_t
ref_fill_mask_blocked (RefEngine &engine, const int32_t keep_q15,
                       const uint32_t units, const uint32_t inner,
                       uint8_t *mask)
{
    uint32_t kept = 0u;
    for (uint32_t unit = 0u; unit < units; ++unit)
        {
            const uint8_t bit = ref_bernoulli_q15 (engine, keep_q15) ? 1u : 0u;
            for (uint32_t lane = 0u; lane < inner; ++lane)
                mask[unit * inner + lane] = bit;
            kept += bit != 0u ? inner : 0u;
        }
    return kept;
}

void
ref_fisher_yates (RefMix &engine, uint16_t *indices, const uint32_t count)
{
    for (uint32_t tail = count; tail-- > 1u;)
        {
            const uint32_t pick = ref_bounded_uniform (engine, tail + 1u);
            const uint16_t swap = indices[pick];
            indices[pick] = indices[tail];
            indices[tail] = swap;
        }
}

uint32_t
ref_sample_without_replacement (RefMix &engine, const uint32_t population,
                                const uint32_t count, uint16_t *out,
                                uint16_t *scratch)
{
    if (population == 0u)
        return 0u;
    const uint32_t draws = count < population ? count : population;
    for (uint32_t i = 0u; i < population; ++i)
        scratch[i] = static_cast<uint16_t> (i);
    for (uint32_t i = 0u; i < draws; ++i)
        {
            const uint32_t pick
                = i + ref_bounded_uniform (engine, population - i);
            const uint16_t swap = scratch[pick];
            scratch[pick] = scratch[i];
            scratch[i] = swap;
            out[i] = scratch[i];
        }
    return draws;
}

/* ------------------------------------------------------------------ *
 * Fixed-point references, written from the documented rules.          *
 * ------------------------------------------------------------------ */

/* (value + 2^(shift-1)) >> shift: round half up (ties toward +infinity). */
int32_t
ref_round_shift (const int64_t value, const uint32_t shift)
{
    const int64_t half = static_cast<int64_t> (1) << (shift - 1u);
    return static_cast<int32_t> ((value + half) >> shift);
}

int32_t
ref_scale_shift (const int32_t a, const int32_t b, const uint32_t shift)
{
    return ref_round_shift (static_cast<int64_t> (a) * b, shift);
}

/* floor (sqrt (n)) by digits, independent of the library helper. */
uint64_t
ref_isqrt (const uint64_t n)
{
    uint64_t root = 0u;
    for (uint32_t bit = 32u; bit-- > 0u;)
        {
            const uint64_t trial = root | (static_cast<uint64_t> (1u) << bit);
            /* trial <= 2^33, trial*triple fits in uint64 for the domains used
               here (radicands below 2^46). */
            if (trial <= 0xFFFFFFFFull && trial * trial <= n)
                root = trial;
        }
    return root;
}

/* sqrt of a wide Q15 magnitude per the documented fold-and-truncate rule. */
int32_t
ref_sqrt_wide (const int32_t value_q15)
{
    if (value_q15 <= 0)
        return 0;
    uint32_t argument = static_cast<uint32_t> (value_q15);
    uint32_t folds = 0u;
    while (argument > 65535u)
        {
            argument >>= 2u;
            ++folds;
        }
    uint64_t root = ref_isqrt (static_cast<uint64_t> (argument) << 15u);
    root <<= folds;
    return root > 0x7FFFFFFFull ? 0x7FFFFFFF : static_cast<int32_t> (root);
}

int32_t
ref_sqrt_ratio (const uint32_t num, const uint32_t den)
{
    if (den == 0u || num == 0u)
        return 0;
    uint64_t radicand = (static_cast<uint64_t> (num) << 30u) / den;
    uint32_t folds = 0u;
    while (radicand > 0xFFFFFFFFull)
        {
            radicand >>= 2u;
            ++folds;
        }
    uint64_t root = ref_isqrt (radicand);
    root <<= folds;
    return root > 0x7FFFFFFFull ? 0x7FFFFFFF : static_cast<int32_t> (root);
}

int32_t
ref_ratio_wide (const uint32_t num, const uint32_t den)
{
    if (den == 0u)
        return 0;
    const uint64_t scaled = (static_cast<uint64_t> (num) << 15u) / den;
    return scaled > 0x7FFFFFFFull ? 0x7FFFFFFF : static_cast<int32_t> (scaled);
}

/* Independent floating-point reference for the inverse standard normal CDF:
   the Abramowitz & Stegun 26.2.22 rational form, evaluated in exact double
   arithmetic instead of the header's 14/15-bit coefficients.  The library holds
   c0 and d1 at 14 fraction bits and d3 at 43 ulps, and it subtracts the
   numerator/denominator ratio from t, so the cancellation near p = 1/2 costs a
   few 1e-4 of sigma while the deep tail carries the relative coefficient error.
   The band the tests use (0.004 sigma, plus 0.002 of the quantile itself) is
   derived from that documented coefficient rounding, never from the library's
   own output. */
double
ref_probit_upper_sigma (const double tail)
{
    const double t = std::sqrt (-2.0 * std::log (tail));
    const double num = 0.015320 * t * t + 0.802853 * t + 2.515517;
    const double den = ((0.001308 * t + 0.189269) * t + 1.432788) * t + 1.0;
    return t - num / den;
}

/* Round a wide Q15 value onto the code grid, half away from zero, unclamped. */
int32_t
ref_wide_to_code (const int32_t value_wide_q15, const int32_t scale_wide_q15)
{
    if (scale_wide_q15 <= 0)
        return 0;
    const int64_t numerator = static_cast<int64_t> (value_wide_q15) * 127;
    const int64_t half = scale_wide_q15 / 2;
    const int64_t rounded
        = numerator >= 0 ? numerator + half : numerator - half;
    return static_cast<int32_t> (rounded / scale_wide_q15);
}

int8_t
ref_clamp_i8 (const int32_t value)
{
    return static_cast<int8_t> (value < -127 ? -127 : (value > 127 ? 127 : value));
}

int8_t
ref_q15_to_code (const int32_t value_q15)
{
    return ref_clamp_i8 (ref_round_shift (static_cast<int64_t> (value_q15) * 127,
                                          15u));
}

/* Mean of two uniforms divided by two with truncation toward zero, then a
   clamped round-half-up Q15 product, per the header. */
template <typename RefEngine>
int32_t
ref_triangular_noise (RefEngine &engine, const int32_t scale_q15)
{
    const int32_t first = ref_uniform_q15 (engine);
    const int32_t second = ref_uniform_q15 (engine);
    const int32_t mean = (first + second) / 2;
    const int64_t product = static_cast<int64_t> (mean) * scale_q15;
    int32_t value = ref_round_shift (product, 15u);
    if (value > 32767)
        value = 32767;
    if (value < -32768)
        value = -32768;
    return value;
}

/* The documented expected-value guard: at p = 1/2 the Q14 gain must be
   exactly 2.0 in the lane domain. */
constexpr int32_t K_q15_half = 16384;
} // namespace

namespace
{
/* ------------------------------------------------------------------ *
 * Dropout references, written from the documented formulas.           *
 * ------------------------------------------------------------------ */

/* gain = 2^29 / keep (truncated integer division), zero for keep <= 0,
   capped at 2^30 per the header. */
int32_t
ref_gain_q14 (const int32_t keep_q15)
{
    if (keep_q15 <= 0)
        return 0;
    const int64_t gain = (static_cast<int64_t> (1) << 29) / keep_q15;
    return gain > 0x40000000ll ? 0x40000000 : static_cast<int32_t> (gain);
}

/* fill = -alpha * p / (1 - p), truncated toward zero, capped at 2^30. */
int32_t
ref_alpha_fill (const int32_t alpha_q15, const int32_t drop_q15,
               const int32_t keep_q15)
{
    if (keep_q15 <= 0 || alpha_q15 == 0)
        return 0;
    const int64_t fill
        = -(static_cast<int64_t> (alpha_q15) * drop_q15) / keep_q15;
    if (fill > 0x40000000ll)
        return 0x40000000;
    if (fill < -0x40000000ll)
        return -0x40000000;
    return static_cast<int32_t> (fill);
}

/* One forward lane: kept lanes get the Q14 gain with round half up and the
   int8 rail saturation; dropped lanes get the fill narrowed to a code. */
int8_t
ref_forward_lane (const int8_t in, const bool kept, const int32_t gain_q14,
                  const int32_t fill_wide_q15, uint32_t &saturated)
{
    if (!kept)
        return ref_q15_to_code (fill_wide_q15);
    const int32_t scaled
        = ref_round_shift (static_cast<int64_t> (in) * gain_q14, 14u);
    if (scaled > 127 || scaled < -127)
        ++saturated;
    return ref_clamp_i8 (scaled);
}

/* The deployed op: keep = 127 - drop, value * 127 / keep with the sign-
   dependent half added before truncating division. */
int8_t
ref_inference_code (const int32_t code, const int32_t drop_code)
{
    if (drop_code <= 0)
        return ref_clamp_i8 (code);
    const int32_t keep = 127 - drop_code;
    if (keep <= 0)
        return 0;
    const int32_t scaled = code * 127;
    const int32_t rounded
        = (scaled + (scaled >= 0 ? keep / 2 : -keep / 2)) / keep;
    return ref_clamp_i8 (rounded);
}

/* The core's 7-bit xorshift32 drop decision, per the header comment. */
uint32_t
ref_inference_field (uint32_t &state)
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return (state >> 17u) & 0x7Fu;
}

int32_t
ref_aux_code (const int32_t drop_q15)
{
    return ref_round_shift (static_cast<int64_t> (drop_q15) * 127, 15u);
}

/* One backward lane: route by mask, apply the stored gain with round half up
   and the Q15 clamp, then saturating-accumulate into the input lane.  Every
   clamp—gain or accumulate—counts one saturation event. */
void
ref_backward_lane (const int16_t grad_out, const uint8_t keep,
                   const int32_t gain_q14, int32_t running, int32_t &lane,
                   uint32_t &saturated)
{
    const int32_t routed = keep != 0u ? grad_out : 0;
    const int32_t scaled = gain_q14 == 16384
                               ? routed
                               : ref_round_shift (static_cast<int64_t> (routed)
                                                      * gain_q14,
                                                  14u);
    const int32_t clamped
        = scaled > 32767 ? 32767 : (scaled < -32768 ? -32768 : scaled);
    if (clamped != scaled)
        ++saturated;
    const int32_t sum = running + clamped;
    if (sum > 32767 || sum < -32768)
        {
            ++saturated;
            lane = sum > 32767 ? 32767 : -32768;
        }
    else
        lane = sum;
}

uint32_t
ref_align_up (const uint32_t value)
{
    return (value + 3u) & ~3u;
}

/* Fixture: a caller-owned 8 KiB training arena, a tape over caller nodes and
   input/output/code/grad buffers, all reusable and 4-byte aligned. */
struct Fixture
{
    static constexpr uint32_t _M_capacity = 16u;
    static constexpr uint32_t _M_max_elements = 4096u;

    alignas (4) uint8_t _M_backing[_S_train_pool_bytes];
    Pool _M_pool { _M_backing, _S_train_pool_bytes };
    Node _M_nodes[_M_capacity];
    Tape _M_tape { _M_nodes, _M_capacity, _M_pool };
    alignas (4) int8_t _M_in_codes[_M_max_elements];
    alignas (4) int8_t _M_out_codes[_M_max_elements];
    alignas (4) int16_t _M_in_grad[_M_max_elements];
    alignas (4) int16_t _M_out_grad[_M_max_elements];
};

/* The reference keep-mask for the exact seeding scheme of dropout_run. */
uint32_t
ref_draw_masks (const uint64_t seed, const int32_t keep_q15,
                const uint32_t units, const uint32_t inner, uint8_t *mask,
                uint64_t *state_after)
{
    RefXor engine { seed == 0ull ? K_increment : seed };
    const uint32_t kept = ref_fill_mask_blocked (engine, keep_q15, units, inner,
                                                 mask);
    if (state_after != nullptr)
        *state_after = engine._M_state;
    return kept;
}

/* Dense kinds: the library spends one Bernoulli draw per element. */
uint32_t
ref_draw_masks_flat (const uint64_t seed, const int32_t keep_q15,
                     const uint32_t elements, uint8_t *mask,
                     uint64_t *state_after)
{
    RefXor engine { seed == 0ull ? K_increment : seed };
    const uint32_t kept = ref_fill_mask (engine, keep_q15, mask, elements);
    if (state_after != nullptr)
        *state_after = engine._M_state;
    return kept;
}

} // namespace

/* ==================================================================== *
 * train_random.hpp: golden streams, bounds, determinism.                *
 * ==================================================================== */

TEST_CASE ("SplitMix64 golden stream equals the published reimplementation")
{
    const uint64_t seeds[]
        = { 0ull,
            1ull,
            42ull,
            0x0123456789ABCDEFull,
            0xFFFFFFFFFFFFFFFFull,
            K_increment };
    for (const uint64_t seed : seeds)
        {
            SplitMix64 engine = SplitMix64::seeded (seed);
            uint64_t ref_state = seed;
            for (uint32_t k = 0u; k < 16u; ++k)
                {
                    const uint64_t want = ref_splitmix64 (ref_state);
                    CHECK (engine.next_u64 () == want);
                }
        }
}

TEST_CASE ("SplitMix64 state advances exactly one golden-ratio step per draw")
{
    SplitMix64 engine = SplitMix64::seeded (7ull);
    uint64_t state = 7ull;
    for (uint32_t k = 0u; k < 40u; ++k)
        {
            state += K_increment;
            engine.next_u64 ();
            CHECK (engine._M_state == state);
        }
    /* The 64-bit wrap edge: from the top of the range the counter folds. */
    engine._M_state = 0xFFFFFFFFFFFFFFFFull;
    state = 0xFFFFFFFFFFFFFFFFull;
    engine.next_u64 ();
    state += K_increment;
    CHECK (engine._M_state == state);
    CHECK (engine._M_state == K_increment - 1ull);
}

TEST_CASE ("XorShift64 golden stream equals the published xorshift64*")
{
    const uint64_t seeds[]
        = { 1ull,
            2ull,
            0x8000000000000000ull,
            0x100000000ull,
            0xFFFFFFFFFFFFFFFFull,
            0xDEADBEEFCAFEBABEull };
    for (const uint64_t seed : seeds)
        {
            XorShift64 engine = XorShift64::seeded (seed);
            uint64_t ref_state = seed;
            for (uint32_t k = 0u; k < 16u; ++k)
                {
                    const uint64_t want = ref_xorshift64star (ref_state);
                    CHECK (engine.next_u64 () == want);
                    CHECK (engine._M_state != 0ull);
                }
        }
}

TEST_CASE ("XorShift64 replaces the zero state with the documented constant")
{
    XorShift64 engine = XorShift64::seeded (0ull);
    CHECK (engine._M_state == K_increment);
    uint64_t ref_state = K_increment;
    CHECK (engine.next_u64 () == ref_xorshift64star (ref_state));

    uint64_t raw_state = 0ull;
    uint64_t other_state = 0ull;
    const uint64_t via_function = rng_xorshift64 (raw_state);
    const uint64_t via_reference = ref_xorshift64star (other_state);
    CHECK (via_function == via_reference);
    CHECK (raw_state == other_state);
    CHECK (raw_state != 0ull);
}

TEST_CASE ("xorshift64 64-bit wrap edges keep the stream identical")
{
    /* A left shift at the very top of the range must fold, not trap. */
    XorShift64 engine = XorShift64::seeded (0xFFFFFFFFFFFFFFFFull);
    uint64_t ref_state = 0xFFFFFFFFFFFFFFFFull;
    for (uint32_t k = 0u; k < 8u; ++k)
        CHECK (engine.next_u64 () == ref_xorshift64star (ref_state));

    /* The state that loses every high bit after one shift/xor stage. */
    engine = XorShift64::seeded (0x0000000100000000ull);
    ref_state = 0x0000000100000000ull;
    for (uint32_t k = 0u; k < 8u; ++k)
        CHECK (engine.next_u64 () == ref_xorshift64star (ref_state));

    SplitMix64 mix = SplitMix64::seeded (0xFFFFFFFFFFFFFFF0ull);
    ref_state = 0xFFFFFFFFFFFFFFF0ull;
    for (uint32_t k = 0u; k < 8u; ++k)
        CHECK (mix.next_u64 () == ref_splitmix64 (ref_state));
}

TEST_CASE ("engines are reproducible per seed and distinct across seeds")
{
    const uint32_t length = 48u;
    uint64_t a[3][length];
    uint64_t b[3][length];
    const uint64_t seeds[3] = { 11ull, 22ull, 33ull };
    for (uint32_t s = 0u; s < 3u; ++s)
        {
            SplitMix64 e1 = SplitMix64::seeded (seeds[s]);
            SplitMix64 e2 = SplitMix64::seeded (seeds[s]);
            XorShift64 x1 = XorShift64::seeded (seeds[s]);
            XorShift64 x2 = XorShift64::seeded (seeds[s]);
            for (uint32_t k = 0u; k < length; ++k)
                {
                    a[0][k] = e1.next_u64 ();
                    b[0][k] = e2.next_u64 ();
                    a[1][k] = x1.next_u64 ();
                    b[1][k] = x2.next_u64 ();
                }
            for (uint32_t k = 0u; k < length; ++k)
                {
                    CHECK (a[0][k] == b[0][k]);
                    CHECK (a[1][k] == b[1][k]);
                }
        }
    /* Different seeds diverge: no overlap between seed 11 and seed 22. */
    SplitMix64 e1 = SplitMix64::seeded (11ull);
    SplitMix64 e2 = SplitMix64::seeded (22ull);
    uint32_t shared_mix = 0u;
    for (uint32_t k = 0u; k < length; ++k)
        if (e1.next_u64 () == e2.next_u64 ())
            ++shared_mix;
    CHECK (shared_mix == 0u);

    XorShift64 x1 = XorShift64::seeded (11ull);
    XorShift64 x2 = XorShift64::seeded (22ull);
    uint32_t shared_xor = 0u;
    for (uint32_t k = 0u; k < length; ++k)
        if (x1.next_u64 () == x2.next_u64 ())
            ++shared_xor;
    CHECK (shared_xor == 0u);

    SECTION ("engine layout is one plain 64-bit state")
    {
        CHECK (sizeof (SplitMix64) == 8u);
        CHECK (sizeof (XorShift64) == 8u);
        CHECK (SplitMix64::seeded (0ull)._M_state == 0ull);
        CHECK (XorShift64::seeded (5ull)._M_state == 5ull);
        /* The published constants, asserted against the literature values. */
        CHECK (_S_splitmix_increment == K_increment);
        CHECK (_S_splitmix_mixer_a == K_mixer_a);
        CHECK (_S_splitmix_mixer_b == K_mixer_b);
        CHECK (_S_xorshift_multiplier == K_xorshift_mult);
    }
}

TEST_CASE ("next_u32 is the high half of next_u64")
{
    SplitMix64 mix = SplitMix64::seeded (1234ull);
    XorShift64 xorish = XorShift64::seeded (1234ull);
    uint64_t ref_mix = 1234ull;
    uint64_t ref_xor = 1234ull;
    for (uint32_t k = 0u; k < 32u; ++k)
        {
            const uint64_t full_mix = ref_splitmix64 (ref_mix);
            CHECK (mix.next_u32 ()
                   == static_cast<uint32_t> (full_mix >> 32u));
            const uint64_t full_xor = ref_xorshift64star (ref_xor);
            CHECK (xorish.next_u32 () == static_cast<uint32_t> (full_xor >> 32u));
        }
}

TEST_CASE ("rng_derive_seed golden table over base and stream index")
{
    const uint64_t bases[] = { 0ull, 1ull, 0xCAFEBABEull,
                               0xFFFFFFFFFFFFFFFFull };
    for (const uint64_t base : bases)
        {
            uint64_t seen[16];
            for (uint32_t stream = 0u; stream < 16u; ++stream)
                {
                    const uint64_t want = ref_derive_seed (base, stream);
                    CHECK (rng_derive_seed (base, stream) == want);
                    seen[stream] = want;
                }
            /* Distinct streams from one base never collide here. */
            for (uint32_t a = 0u; a < 16u; ++a)
                for (uint32_t b = a + 1u; b < 16u; ++b)
                    CHECK (seen[a] != seen[b]);
        }
}

TEST_CASE ("rng_mix_seed and dropout_step_seed are deterministic and distinct")
{
    const uint64_t a = 0x0102030405060708ull;
    const uint64_t b = 0x0807060504030201ull;
    CHECK (rng_mix_seed (a, b) == rng_mix_seed (a, b));
    CHECK (rng_mix_seed (a, b) != rng_mix_seed (b, a));
    CHECK (rng_mix_seed (a, 0ull) != rng_mix_seed (a, 1ull));

    /* Distinct for every (epoch, step) pair in a small grid. */
    uint64_t grid[4][8];
    for (uint32_t epoch = 0u; epoch < 4u; ++epoch)
        for (uint32_t step = 0u; step < 8u; ++step)
            grid[epoch][step] = dropout_step_seed (0xABCDull, epoch, step);
    for (uint32_t e1 = 0u; e1 < 4u; ++e1)
        for (uint32_t s1 = 0u; s1 < 8u; ++s1)
            for (uint32_t e2 = 0u; e2 < 4u; ++e2)
                for (uint32_t s2 = 0u; s2 < 8u; ++s2)
                    if (e1 != e2 || s1 != s2)
                        CHECK (grid[e1][s1] != grid[e2][s2]);
}

TEST_CASE ("uniform_q15 covers the full signed lane from the top 16 bits")
{
    SplitMix64 mix = SplitMix64::seeded (0x5EEDull);
    RefMix ref_mix { 0x5EEDull };
    XorShift64 xorish = XorShift64::seeded (0x5EEDull);
    RefXor ref_xor { 0x5EEDull };
    int32_t minimum = 32767;
    int32_t maximum = -32768;
    for (uint32_t k = 0u; k < 256u; ++k)
        {
            const int32_t want_mix = ref_uniform_q15 (ref_mix);
            const int32_t got_mix = uniform_q15 (mix);
            CHECK (got_mix == want_mix);
            CHECK (got_mix >= -32768);
            CHECK (got_mix <= 32767);
            const int32_t want_xor = ref_uniform_q15 (ref_xor);
            CHECK (uniform_q15 (xorish) == want_xor);
            minimum = got_mix < minimum ? got_mix : minimum;
            maximum = got_mix > maximum ? got_mix : maximum;
        }
    CHECK (minimum <= -20000);
    CHECK (maximum >= 20000);
}

TEST_CASE ("uniform_positive_q15 stays in [0, 32767]")
{
    SplitMix64 engine = SplitMix64::seeded (0xF00Dull);
    RefMix ref_engine { 0xF00Dull };
    for (uint32_t k = 0u; k < 200u; ++k)
        {
            const int32_t value = uniform_positive_q15 (engine);
            CHECK (value == ref_uniform_positive_q15 (ref_engine));
            CHECK (value >= 0);
            CHECK (value <= 32767);
        }
}

TEST_CASE ("bounded_uniform honours the exclusive bound and spends one draw")
{
    const uint32_t bounds[] = { 1u, 2u, 3u, 7u, 100u, 255u, 1000u, 65535u };
    for (const uint32_t bound : bounds)
        {
            SplitMix64 engine = SplitMix64::seeded (bound);
            RefMix ref_engine { bound };
            for (uint32_t k = 0u; k < 40u; ++k)
                {
                    const uint32_t want
                        = ref_bounded_uniform (ref_engine, bound);
                    const uint32_t got = bounded_uniform (engine, bound);
                    CHECK (got == want);
                    CHECK (got < bound);
                }
        }
    /* bound 1 is always 0; bound 0 is always 0 and consumes nothing. */
    SplitMix64 engine = SplitMix64::seeded (99ull);
    for (uint32_t k = 0u; k < 10u; ++k)
        CHECK (bounded_uniform (engine, 1u) == 0u);
    const uint64_t state_before = engine._M_state;
    for (uint32_t k = 0u; k < 10u; ++k)
        CHECK (bounded_uniform (engine, 0u) == 0u);
    CHECK (engine._M_state == state_before);
}

TEST_CASE ("bernoulli_q15 threshold is exclusive and consumes only in range")
{
    const int32_t probabilities[]
        = { -1, 0, 1, 16383, 16384, 16385, 32766, 32767, 32768, 40000 };
    for (const int32_t p : probabilities)
        {
            XorShift64 engine = XorShift64::seeded (0xBEEFull);
            RefXor ref_engine { 0xBEEFull };
            uint32_t kept = 0u;
            for (uint32_t k = 0u; k < 64u; ++k)
                {
                    const bool want = ref_bernoulli_q15 (ref_engine, p);
                    const bool got = bernoulli_q15 (engine, p);
                    CHECK (got == want);
                    kept += got ? 1u : 0u;
                }
            if (p <= 0)
                CHECK (kept == 0u);
            if (p >= 32768)
                CHECK (kept == 64u);
            if (p <= 0 || p >= 32768)
                CHECK (engine._M_state == 0xBEEFull);
        }
    /* p = 1 keeps exactly when the 15-bit draw is 0. */
    XorShift64 engine = XorShift64::seeded (3ull);
    RefXor ref_engine { 3ull };
    for (uint32_t k = 0u; k < 200u; ++k)
        {
            const uint64_t raw = ref_engine.next_u64 ();
            const int32_t draw = static_cast<int32_t> (
                static_cast<uint32_t> (raw >> 49u));
            const bool want = draw < 1;
            CHECK (bernoulli_q15 (engine, 1) == want);
            CHECK (want == (draw == 0));
        }
}

TEST_CASE ("fill_mask reproduces the reference stream exactly")
{
    const int32_t keep_values[] = { 1, 8192, K_q15_half, 24576, 32767, 32768, 0 };
    uint8_t mask[300];
    uint8_t ref_mask[300];
    for (const int32_t keep : keep_values)
        {
            XorShift64 engine = XorShift64::seeded (0x1234ull);
            RefXor ref_engine { 0x1234ull };
            for (uint32_t i = 0u; i < 300u; ++i)
                ref_mask[i] = 0xEEu;
            const uint32_t kept
                = fill_mask (engine, keep, mask, 256u);
            const uint32_t ref_kept
                = ref_fill_mask (ref_engine, keep, ref_mask, 256u);
            CHECK (kept == ref_kept);
            uint32_t ones = 0u;
            for (uint32_t i = 0u; i < 256u; ++i)
                {
                    CHECK (mask[i] == ref_mask[i]);
                    CHECK ((mask[i] == 0u || mask[i] == 1u));
                    ones += mask[i];
                }
            CHECK (ones == kept);
            if (keep >= 32768)
                CHECK (kept == 256u);
            if (keep <= 0)
                CHECK (kept == 0u);
        }
    /* Empty count is a no-op. */
    XorShift64 engine = XorShift64::seeded (5ull);
    CHECK (fill_mask (engine, 16384, mask, 0u) == 0u);
    CHECK (engine._M_state == 5ull);
}

TEST_CASE ("fill_mask_blocked replicates one draw over the inner stride")
{
    struct Row
    {
        uint32_t _M_units;
        uint32_t _M_inner;
    };
    const Row rows[] = { { 1u, 60u }, { 60u, 1u }, { 6u, 10u }, { 3u, 21u },
                         { 12u, 5u } };
    const int32_t keep_values[] = { 1, K_q15_half, 32767 };
    uint8_t mask[128];
    uint8_t ref_mask[128];
    for (const Row &row : rows)
        for (const int32_t keep : keep_values)
            {
                XorShift64 engine = XorShift64::seeded (0x777ull);
                RefXor ref_engine { 0x777ull };
                const uint32_t kept = fill_mask_blocked (
                    engine, keep, row._M_units, row._M_inner, mask);
                const uint32_t ref_kept = ref_fill_mask_blocked (
                    ref_engine, keep, row._M_units, row._M_inner, ref_mask);
                CHECK (kept == ref_kept);
                CHECK (kept % row._M_inner == 0u);
                for (uint32_t unit = 0u; unit < row._M_units; ++unit)
                    {
                        const uint8_t head = ref_mask[unit * row._M_inner];
                        for (uint32_t lane = 0u; lane < row._M_inner; ++lane)
                            {
                                const uint32_t i = unit * row._M_inner + lane;
                                CHECK (mask[i] == head);
                                CHECK (mask[i] == ref_mask[i]);
                            }
                    }
            }
}

TEST_CASE ("fill_lanes_uniform writes the signed lane grid")
{
    SplitMix64 engine = SplitMix64::seeded (0x10Eull);
    RefMix ref_engine { 0x10Eull };
    int16_t lanes[96];
    fill_lanes_uniform (engine, lanes, 96u);
    for (uint32_t i = 0u; i < 96u; ++i)
        CHECK (lanes[i] == static_cast<int16_t> (ref_uniform_q15 (ref_engine)));
}

TEST_CASE ("triangular_noise_q15 is the documented mean of two uniforms")
{
    const int32_t scales[] = { 0, 1, 8192, 32767, 32768 };
    for (const int32_t scale : scales)
        {
            SplitMix64 engine = SplitMix64::seeded (0x7A5u + scale);
            RefMix ref_engine { 0x7A5u + scale };
            for (uint32_t k = 0u; k < 48u; ++k)
                {
                    const int32_t got = triangular_noise_q15 (engine, scale);
                    CHECK (got == ref_triangular_noise (ref_engine, scale));
                    if (scale >= 0 && scale <= 32767)
                        {
                            CHECK (got <= scale);
                            CHECK (got >= -scale);
                        }
                }
        }
}

TEST_CASE ("normal_wide_q15 respects the documented support and quantile grid")
{
    SplitMix64 engine = SplitMix64::seeded (0xC0FFEEull);
    int32_t minimum = 0;
    int32_t maximum = 0;
    for (uint32_t k = 0u; k < 400u; ++k)
        {
            const int32_t z = normal_wide_q15 (engine);
            CHECK (z <= _S_normal_t_limit_q15);
            CHECK (z >= -_S_normal_t_limit_q15);
            minimum = z < minimum ? z : minimum;
            maximum = z > maximum ? z : maximum;
        }
    CHECK (minimum < 0);
    CHECK (maximum > 0);

    /* The exposed quantile branch: monotone in the tail probability, graded
       against the double-precision A&S reference, and the deepest supported
       tail sits near 4.0 sigma (the true quantile of 1 ulp of Q15, 3.05e-5,
       is 3.997). */
    int32_t previous = _S_normal_t_limit_q15 + 1;
    for (int32_t tail = 1; tail <= K_q15_half; tail += 37)
        {
            const int32_t z = normal_quantile_q15 (tail);
            CHECK (z >= 0);
            CHECK (z <= _S_normal_t_limit_q15);
            CHECK (z <= previous);
            previous = z;
            const double want = ref_probit_upper_sigma (tail / 32768.0);
            const double got = z / 32768.0;
            CHECK (got == Approx (want).margin (0.004 + 0.002 * std::fabs (want)));
        }
    /* p = 1/2 is the documented zero crossing; the fixed-point cancellation
       leaves a sub-milli-sigma residual, so the band is the one above. */
    CHECK (normal_quantile_q15 (K_q15_half)
           == Approx (ref_probit_upper_sigma (0.5) * 32768.0).margin (164.0));
    CHECK (std::fabs (normal_quantile_q15 (K_q15_half)) < 328);
    CHECK (normal_quantile_q15 (0) == normal_quantile_q15 (1));
    CHECK (normal_quantile_q15 (K_q15_half + 100)
           == normal_quantile_q15 (K_q15_half));
    const int32_t deepest = normal_quantile_q15 (1);
    CHECK (deepest > 125000);
    CHECK (deepest < _S_normal_t_limit_q15);
    CHECK (deepest == Approx (ref_probit_upper_sigma (1.0 / 32768.0)
                              * 32768.0).margin (0.004 * 32768.0
                                                 + 0.002 * deepest));

    /* normal_q15 with a tiny sigma never clamps; the reporting counter and
       the ignoring overload agree. */
    SplitMix64 a = SplitMix64::seeded (0x9ECull);
    SplitMix64 b = SplitMix64::seeded (0x9ECull);
    uint32_t clamped = 0u;
    for (uint32_t k = 0u; k < 200u; ++k)
        {
            const int32_t reported = normal_q15 (a, 64, clamped);
            const int32_t plain = normal_q15 (b, 64);
            CHECK (reported == plain);
            CHECK (reported <= 32767);
            CHECK (reported >= -32768);
        }
    CHECK (clamped == 0u);
}

TEST_CASE ("fixed point square roots follow the documented fold rules")
{
    SECTION ("q15_sqrt_wide")
    {
        const int32_t inputs[] = { -1, 0, 1, 32768, 65535, 65536, 131072,
                                   295176, 1048576, 0x7FFFFFFF };
        for (const int32_t value : inputs)
            CHECK (q15_sqrt_wide (value) == ref_sqrt_wide (value));
        /* 1.0 maps to 1.0, 4.0 (wide) maps to 2.0 (wide). */
        CHECK (q15_sqrt_wide (32768) == 32768);
        CHECK (q15_sqrt_wide (131072) == 65536);
    }
    SECTION ("sqrt_ratio_q15")
    {
        CHECK (sqrt_ratio_q15 (0u, 5u) == 0);
        CHECK (sqrt_ratio_q15 (5u, 0u) == 0);
        CHECK (sqrt_ratio_q15 (4u, 1u) == 65536);
        CHECK (sqrt_ratio_q15 (1u, 4u) == 16384);
        CHECK (sqrt_ratio_q15 (2u, 2u) == 32768);
        const uint32_t pairs[][2] = { { 6u, 12u }, { 3u, 1u }, { 1u, 8u },
                                      { 2u, 16u }, { 25165824u, 3u },
                                      { 1u, 65535u }, { 65535u, 1u } };
        for (const auto &pair : pairs)
            CHECK (sqrt_ratio_q15 (pair[0], pair[1])
                   == ref_sqrt_ratio (pair[0], pair[1]));
    }
    SECTION ("ratio_wide_q15")
    {
        CHECK (ratio_wide_q15 (1u, 0u) == 0);
        CHECK (ratio_wide_q15 (0u, 7u) == 0);
        CHECK (ratio_wide_q15 (1u, 2u) == 16384);
        CHECK (ratio_wide_q15 (3u, 1u) == 98304);
        CHECK (ratio_wide_q15 (1u, 3u) == 10922);
        CHECK (ratio_wide_q15 (0x7FFFFFFFu, 1u) == 0x7FFFFFFF);
        const uint32_t pairs[][2]
            = { { 1u, 1u },      { 3u, 7u },       { 65535u, 3u },
                { 1u, 65535u },  { 1234567u, 99u },
                { 0x7FFFFFFFu, 2u },     { 0x7FFFFFFFu, 65535u },
                { 0xFFFFFFFFu, 1u },     { 0xFFFFFFFFu, 3u },
                { 25165824u, 15u },      { 7u, 3u } };
        for (const auto &pair : pairs)
            {
                const int32_t got = ratio_wide_q15 (pair[0], pair[1]);
                CHECK (got == ref_ratio_wide (pair[0], pair[1]));
                CHECK (got >= 0);
                CHECK (static_cast<uint64_t> (got) <= 0x7FFFFFFFull);
            }
    }
}

TEST_CASE ("permutation helpers reproduce the reference Fisher-Yates")
{
    SECTION ("make_permutation")
    {
        for (const uint32_t count : { 1u, 2u, 3u, 8u, 64u, 300u })
            {
                uint16_t indices[300];
                uint16_t ref_indices[300];
                SplitMix64 engine = SplitMix64::seeded (count + 100ull);
                RefMix ref_engine { count + 100ull };
                CHECK (make_permutation (engine, indices, count) == Status::OK);
                for (uint32_t i = 0u; i < count; ++i)
                    ref_indices[i] = static_cast<uint16_t> (i);
                ref_fisher_yates (ref_engine, ref_indices, count);
                for (uint32_t i = 0u; i < count; ++i)
                    CHECK (indices[i] == ref_indices[i]);
                /* Valid permutation: every index appears exactly once. */
                bool seen[300] = { false };
                for (uint32_t i = 0u; i < count; ++i)
                    {
                        CHECK (indices[i] < count);
                        CHECK_FALSE (seen[indices[i]]);
                        seen[indices[i]] = true;
                    }
            }
        SplitMix64 empty_engine = SplitMix64::seeded (1ull);
        CHECK (make_permutation (empty_engine, nullptr, 0u) == Status::OK);
        uint16_t dummy[4];
        SplitMix64 null_engine = SplitMix64::seeded (1ull);
        SplitMix64 range_engine = SplitMix64::seeded (1ull);
        CHECK (make_permutation (null_engine, nullptr, 3u)
               == Status::BAD_LAYOUT);
        CHECK (make_permutation (range_engine, dummy, 65536u)
               == Status::OUT_OF_RANGE);
    }
    SECTION ("init_indices and permute_indices")
    {
        uint16_t indices[40];
        init_indices (indices, 40u);
        for (uint32_t i = 0u; i < 40u; ++i)
            CHECK (indices[i] == i);
        /* A one-element permutation is the identity. */
        SplitMix64 one_engine = SplitMix64::seeded (7ull);
        CHECK (make_permutation (one_engine, indices, 1u) == Status::OK);
        CHECK (indices[0] == 0u);
        /* permute_indices shuffles existing indices exactly like the
           reference Fisher-Yates on the same engine state. */
        uint16_t ref_indices[24];
        for (uint32_t i = 0u; i < 24u; ++i)
            indices[i] = ref_indices[i] = static_cast<uint16_t> (10u + i);
        SplitMix64 engine = SplitMix64::seeded (0x55ull);
        RefMix ref_engine { 0x55ull };
        permute_indices (engine, indices, 24u);
        ref_fisher_yates (ref_engine, ref_indices, 24u);
        for (uint32_t i = 0u; i < 24u; ++i)
            CHECK (indices[i] == ref_indices[i]);
    }
    SECTION ("sample_without_replacement")
    {
        uint16_t out[64];
        uint16_t scratch[64];
        for (uint32_t population : { 1u, 5u, 17u, 64u })
            for (uint32_t count : { 0u, 1u, 3u, 64u })
                {
                    SplitMix64 engine = SplitMix64::seeded (population);
                    RefMix ref_engine { population };
                    const uint32_t drawn = sample_without_replacement (
                        engine, population, count, out, scratch);
                    CHECK (drawn == (count < population ? count : population));
                    uint16_t ref_out[64];
                    uint16_t ref_scratch[64];
                    const uint32_t ref_drawn = ref_sample_without_replacement (
                        ref_engine, population, count, ref_out, ref_scratch);
                    CHECK (ref_drawn == drawn);
                    bool seen[64] = { false };
                    for (uint32_t i = 0u; i < drawn; ++i)
                        {
                            CHECK (out[i] == ref_out[i]);
                            CHECK (out[i] < population);
                            CHECK_FALSE (seen[out[i]]);
                            seen[out[i]] = true;
                        }
                }
        SplitMix64 zero_population (SplitMix64::seeded (1ull));
        SplitMix64 null_out (SplitMix64::seeded (1ull));
        SplitMix64 null_scratch (SplitMix64::seeded (1ull));
        SplitMix64 big_population (SplitMix64::seeded (1ull));
        CHECK (sample_without_replacement (zero_population, 0u, 4u, out,
                                           scratch)
               == 0u);
        CHECK (sample_without_replacement (null_out, 8u, 4u, nullptr, scratch)
               == 0u);
        CHECK (sample_without_replacement (null_scratch, 8u, 4u, out, nullptr)
               == 0u);
        CHECK (sample_without_replacement (big_population, 65536u, 4u, out,
                                           scratch)
               == 0u);
    }
}
/* ==================================================================== *
 * train_init.hpp: fans, code rails, documented limits, dispatch.       *
 * ==================================================================== */

TEST_CASE ("fans_for_shape follows the PyTorch convention for every rank")
{
    struct Row
    {
        Shape _M_shape;
        uint32_t _M_fan_in;
        uint32_t _M_fan_out;
    };
    const Row rows[] = {
        { Shape::matrix (4u, 3u), 3u, 4u },
        { Shape::matrix (1u, 1u), 1u, 1u },
        { Shape::matrix (128u, 64u), 64u, 128u },
        { Shape::tensor3 (8u, 4u, 5u), 20u, 40u },
        { Shape::tensor3 (1u, 7u, 3u), 21u, 3u },
        { Shape::tensor4 (16u, 3u, 3u, 3u), 27u, 144u },
        { Shape::tensor4 (2u, 8u, 5u, 5u), 200u, 50u },
        { Shape::tensor4 (1u, 1u, 1u, 1u), 1u, 1u },
    };
    for (const Row &row : rows)
        {
            InitFans fans;
            REQUIRE (fans_for_shape (row._M_shape, fans) == Status::OK);
            /* Reference: rank 2 gives (dims[1], dims[0]); higher ranks fold
               the receptive field (the product of the trailing axes) into
               both sides. */
            uint32_t receptive = 1u;
            for (uint8_t axis = 2u; axis < row._M_shape._M_rank; ++axis)
                receptive *= row._M_shape._M_dims[axis];
            CHECK (fans._M_fan_in
                   == static_cast<uint32_t> (row._M_shape._M_dims[1])
                          * receptive);
            CHECK (fans._M_fan_out
                   == static_cast<uint32_t> (row._M_shape._M_dims[0])
                          * receptive);
            CHECK (fans._M_fan_in == row._M_fan_in);
            CHECK (fans._M_fan_out == row._M_fan_out);
        }

    SECTION ("a degenerate rank is rejected with BAD_LAYOUT")
    {
        InitFans fans;
        CHECK (fans_for_shape (Shape::scalar (), fans) == Status::BAD_LAYOUT);
        CHECK (fans_for_shape (Shape::vector (8u), fans) == Status::BAD_LAYOUT);
        Shape rank0;
        CHECK (fans_for_shape (rank0, fans) == Status::BAD_LAYOUT);
        Shape rank5 = Shape::tensor4 (2u, 2u, 2u, 2u);
        rank5._M_rank = 5u;
        CHECK (fans_for_shape (rank5, fans) == Status::BAD_LAYOUT);
    }
    SECTION ("a zero fan side is rejected with BAD_LAYOUT")
    {
        InitFans fans;
        CHECK (fans_for_shape (Shape::matrix (4u, 0u), fans)
               == Status::BAD_LAYOUT);
        CHECK (fans_for_shape (Shape::matrix (0u, 4u), fans)
               == Status::BAD_LAYOUT);
        CHECK (fans_for_shape (Shape::tensor3 (2u, 3u, 0u), fans)
               == Status::BAD_LAYOUT);
    }
    SECTION ("the fan pair is two plain counts")
    {
        CHECK (sizeof (InitFans) == 8u);
        InitFans fresh;
        CHECK (fresh._M_fan_in == 0u);
        CHECK (fresh._M_fan_out == 0u);
    }
}

TEST_CASE ("InitResult defaults match the documented unit scale")
{
    CHECK (sizeof (InitResult) == 16u);
    InitResult result;
    CHECK (result._M_scale_q15 == _S_q15_one);
    CHECK (result._M_sigma_q15 == 0);
    CHECK (result._M_clamped == 0u);
    CHECK (result._M_elements == 0u);
}

TEST_CASE ("wide_to_code rounds half away from zero on the documented grid")
{
    struct Row
    {
        int32_t _M_value;
        int32_t _M_scale;
        int32_t _M_code;
    };
    const Row rows[] = {
        { 0, 32768, 0 },
        { 32768, 32768, 127 },       /* code-127 magnitude maps to 127 */
        { 16384, 32768, 64 },        /* 63.5 rounds away to 64 */
        { -16384, 32768, -64 },
        { 16383, 32768, 63 },
        { -16383, 32768, -63 },
        { 65536, 32768, 254 },       /* beyond the rails: reported, not wrapped */
        { -65536, 32768, -254 },
        { 129, 129, 127 },
        { 1, 32768, 0 },             /* sub-half-code rounds to 0 */
        { -1, 32768, 0 },
        { 49152, 49152, 127 },
        { 5, 10, 64 },               /* (5*127 + 5) / 10 = 64 exactly */
        { 32768, 0, 0 },             /* non-positive scale yields 0 */
        { 32768, -5, 0 },
    };
    for (const Row &row : rows)
        {
            CHECK (wide_to_code (row._M_value, row._M_scale) == row._M_code);
            CHECK (wide_to_code (row._M_value, row._M_scale)
                   == ref_wide_to_code (row._M_value, row._M_scale));
        }
    /* Rail behaviour: values outside [-scale, scale] give codes beyond
       [-127, 127], so callers must saturate, never wrap. */
    CHECK (wide_to_code (32768, 8192) == 508);
    CHECK (wide_to_code (-32768, 8192) == -508);
    CHECK (wide_to_code (30000, 1000) == 3810);
}

TEST_CASE ("init_uniform draws the whole [-127, 127] code range without -128")
{
    int8_t codes[300];
    SplitMix64 engine = SplitMix64::seeded (0x111ull);
    RefMix ref_engine { 0x111ull };
    InitResult result;
    REQUIRE (init_uniform (engine, 20000, codes, 300u, result) == Status::OK);
    for (uint32_t i = 0u; i < 300u; ++i)
        {
            const int8_t want = ref_clamp_i8 (
                static_cast<int32_t> (ref_bounded_uniform (ref_engine, 255u))
                - 127);
            CHECK (codes[i] == want);
            CHECK (codes[i] >= -127);
            CHECK (codes[i] <= 127);
        }
    CHECK (result._M_scale_q15 == 20000);
    CHECK (result._M_clamped == 0u);
    CHECK (result._M_elements == 300u);
    /* The reported sigma is limit / sqrt(3) via the documented Q15 constant:
       (limit * 18919 + 2^14) >> 15. */
    CHECK (result._M_sigma_q15
           == ref_scale_shift (20000, _S_uniform_sigma_q15, 15u));

    SECTION ("a bad configuration reports the documented Status")
    {
        InitResult other;
        CHECK (init_uniform (engine, 20000, nullptr, 8u, other)
               == Status::BAD_LAYOUT);
        CHECK (init_uniform (engine, 20000, codes, 0u, other)
               == Status::BAD_LAYOUT);
        CHECK (init_uniform (engine, 0, codes, 8u, other)
               == Status::OUT_OF_RANGE);
        CHECK (init_uniform (engine, -1, codes, 8u, other)
               == Status::OUT_OF_RANGE);
        CHECK (init_uniform (engine, 20000, codes, 65536u, other)
               == Status::OUT_OF_RANGE);
    }
    SECTION ("the end bins are half-wide as documented")
    {
        /* Walk enough draws to see both end codes and no -128 anywhere. */
        SplitMix64 walker = SplitMix64::seeded (0xB16ull);
        int8_t stream[4096];
        InitResult walk;
        REQUIRE (init_uniform (walker, 32767, stream, 4096u, walk)
                 == Status::OK);
        bool seen_min = false;
        bool seen_max = false;
        for (uint32_t i = 0u; i < 4096u; ++i)
            {
                CHECK (stream[i] != -128);
                seen_min = seen_min || stream[i] == -127;
                seen_max = seen_max || stream[i] == 127;
            }
        CHECK (seen_min);
        CHECK (seen_max);
    }
}

TEST_CASE ("init_normal reports the 3-sigma grid scale and stays on the rails")
{
    const int32_t sigmas[] = { 1, 4096, 8192, 32767, 100000 };
    for (const int32_t sigma : sigmas)
        {
            int8_t codes[200];
            InitResult result;
            SplitMix64 engine = SplitMix64::seeded (sigma + 0x250ull);
            REQUIRE (init_normal (engine, sigma, codes, 200u, result)
                     == Status::OK);
            /* scale = 3 * sigma, documented as (sigma * 98304 + 2^13) >> 14. */
            CHECK (result._M_scale_q15
                   == ref_scale_shift (sigma, _S_normal_truncation_q14, 14u));
            CHECK (result._M_sigma_q15 == sigma);
            CHECK (result._M_elements == 200u);
            uint32_t extremes = 0u;
            for (uint32_t i = 0u; i < 200u; ++i)
                {
                    CHECK (codes[i] >= -127);
                    CHECK (codes[i] <= 127);
                    extremes += codes[i] == 127 || codes[i] == -127 ? 1u : 0u;
                }
            /* The clamped counter must agree with what the rails show. */
            CHECK (result._M_clamped <= extremes);
        }

    int8_t codes[8];
    InitResult result;
    SplitMix64 engine = SplitMix64::seeded (1ull);
    CHECK (init_normal (engine, 8192, nullptr, 4u, result)
           == Status::BAD_LAYOUT);
    CHECK (init_normal (engine, 8192, codes, 0u, result)
           == Status::BAD_LAYOUT);
    CHECK (init_normal (engine, 0, codes, 4u, result)
           == Status::OUT_OF_RANGE);
    CHECK (init_normal (engine, -3, codes, 4u, result)
           == Status::OUT_OF_RANGE);
    CHECK (init_normal (engine, 8192, codes, 65536u, result)
           == Status::OUT_OF_RANGE);

    SECTION ("a fixed seed reproduces the identical stream")
    {
        int8_t first[150];
        int8_t second[150];
        InitResult r1;
        InitResult r2;
        SplitMix64 e1 = SplitMix64::seeded (0x5EEDull);
        SplitMix64 e2 = SplitMix64::seeded (0x5EEDull);
        REQUIRE (init_normal (e1, 12000, first, 150u, r1) == Status::OK);
        REQUIRE (init_normal (e2, 12000, second, 150u, r2) == Status::OK);
        for (uint32_t i = 0u; i < 150u; ++i)
            CHECK (first[i] == second[i]);
        CHECK (r1._M_clamped == r2._M_clamped);
    }
}

TEST_CASE ("init_normal_clamped saturates into the rails around a mean")
{
    int8_t codes[400];
    InitResult result;
    SplitMix64 engine = SplitMix64::seeded (0xCAFEull);
    REQUIRE (init_normal_clamped (engine, 20000, 8000, 12000, codes, 400u,
                                 result)
             == Status::OK);
    CHECK (result._M_scale_q15 == 12000);
    CHECK (result._M_sigma_q15 == 8000);
    CHECK (result._M_elements == 400u);
    for (uint32_t i = 0u; i < 400u; ++i)
        CHECK (codes[i] != -128);
    /* A mean far off zero on a tight grid saturates: the counter is non-zero
       and every saturation is visible on the rails (a rail code can also be
       reached without clamping, so rails bound the counter from above). */
    CHECK (result._M_clamped > 0u);
    uint32_t rails = 0u;
    for (uint32_t i = 0u; i < 400u; ++i)
        rails += codes[i] == 127 || codes[i] == -127 ? 1u : 0u;
    CHECK (result._M_clamped <= rails);

    CHECK (init_normal_clamped (engine, 0, 0, 100, codes, 8u, result)
           == Status::OUT_OF_RANGE);
    CHECK (init_normal_clamped (engine, 0, 100, 0, codes, 8u, result)
           == Status::OUT_OF_RANGE);
    CHECK (init_normal_clamped (engine, 0, 100, 100, nullptr, 8u, result)
           == Status::BAD_LAYOUT);
    CHECK (init_normal_clamped (engine, 0, 100, 100, codes, 0u, result)
           == Status::BAD_LAYOUT);
}

TEST_CASE ("init_constant places values on the grid with the documented rounding")
{
    struct Row
    {
        int32_t _M_value;
        int32_t _M_scale;
        int8_t _M_code;
        uint32_t _M_clamped;
    };
    const Row rows[] = {
        { 32768, 32768, 127, 0u },   /* gamma 1.0 */
        { 0, 32768, 0, 0u },         /* beta 0.0 */
        { 16384, 32768, 64, 0u },    /* half away from zero */
        { -16384, 32768, -64, 0u },
        { 32768, 8192, 127, 12u },   /* beyond one scale: every lane saturates */
        { -32768, 8192, -127, 12u },
        { 4096, 32768, 16, 0u },
    };
    for (const Row &row : rows)
        {
            int8_t codes[12];
            InitResult result;
            REQUIRE (init_constant (row._M_value, row._M_scale, codes, 12u,
                                    result)
                     == Status::OK);
            const int32_t code
                = ref_wide_to_code (row._M_value, row._M_scale);
            CHECK (wide_to_code (row._M_value, row._M_scale) == code);
            CHECK (ref_clamp_i8 (code) == row._M_code);
            for (uint32_t i = 0u; i < 12u; ++i)
                CHECK (codes[i] == ref_clamp_i8 (code));
            CHECK (result._M_scale_q15 == row._M_scale);
            CHECK (result._M_sigma_q15 == 0);
            CHECK (result._M_elements == 12u);
            CHECK (result._M_clamped
                   == (code > 127 || code < -127 ? 12u : 0u));
            CHECK (result._M_clamped == row._M_clamped);
        }
    int8_t codes[4];
    InitResult result;
    CHECK (init_constant (0, 0, codes, 4u, result) == Status::OUT_OF_RANGE);
    CHECK (init_constant (0, -1, codes, 4u, result) == Status::OUT_OF_RANGE);
    CHECK (init_constant (0, 32768, nullptr, 4u, result)
           == Status::BAD_LAYOUT);
    CHECK (init_constant (0, 32768, codes, 65536u, result)
           == Status::OUT_OF_RANGE);

    SECTION ("unit gain and zero bias are the two named constants")
    {
        int8_t gain[9];
        int8_t bias[9];
        InitResult gain_result;
        InitResult bias_result;
        REQUIRE (init_unit_gain (gain, 9u, gain_result) == Status::OK);
        REQUIRE (init_zero_bias (bias, 9u, bias_result) == Status::OK);
        for (uint32_t i = 0u; i < 9u; ++i)
            {
                CHECK (gain[i] == 127);
                CHECK (bias[i] == 0);
            }
        CHECK (gain_result._M_scale_q15 == _S_q15_one);
        CHECK (bias_result._M_scale_q15 == _S_q15_one);
        CHECK (gain_result._M_clamped == 0u);
        CHECK (bias_result._M_clamped == 0u);
    }
}
TEST_CASE ("xavier uniform limit is sqrt (6 / (fan_in + fan_out)) in wide Q15")
{
    struct Row
    {
        uint32_t _M_fan_in;
        uint32_t _M_fan_out;
    };
    const Row rows[] = { { 1u, 1u },   { 3u, 4u },  { 64u, 32u },
                         { 128u, 128u }, { 512u, 8u }, { 1u, 1024u } };
    for (const Row &row : rows)
        {
            const InitFans fans { row._M_fan_in, row._M_fan_out };
            const uint32_t sum = row._M_fan_in + row._M_fan_out;
            int8_t codes[96];
            InitResult result;
            SplitMix64 engine = SplitMix64::seeded (sum + 0x4A00ull);
            REQUIRE (xavier_uniform (engine, fans, codes, 96u, result)
                     == Status::OK);
            /* Reference: the documented half range sqrt(6/sum), computed from
               the published fold-and-isqrt rule, not from the library. */
            const int32_t want = ref_sqrt_ratio (6u, sum);
            CHECK (result._M_scale_q15 == want);
            CHECK (want == ref_sqrt_ratio (6u, sum));
            CHECK (result._M_elements == 96u);
            for (uint32_t i = 0u; i < 96u; ++i)
                CHECK ((codes[i] >= -127 && codes[i] <= 127));
        }
    InitFans tiny { 1u, 0u };
    int8_t codes[8];
    InitResult result;
    SplitMix64 engine = SplitMix64::seeded (9ull);
    /* sum < 2 is rejected before touching the buffer. */
    CHECK (xavier_uniform (engine, tiny, codes, 8u, result)
           == Status::OUT_OF_RANGE);
    InitFans huge { 200000u, 200000u };
    CHECK (xavier_uniform (engine, huge, codes, 8u, result)
           == Status::OUT_OF_RANGE);
    /* A zero fan sum is also below the bound of 2. */
    InitFans zero_fans { 0u, 0u };
    CHECK (xavier_uniform (engine, zero_fans, codes, 8u, result)
           == Status::OUT_OF_RANGE);
}

TEST_CASE ("xavier normal sigma is sqrt (2 / (fan_in + fan_out)) on a 3-sigma grid")
{
    const InitFans fans { 40u, 80u };
    int8_t codes[64];
    InitResult result;
    SplitMix64 engine = SplitMix64::seeded (0x2468ull);
    REQUIRE (xavier_normal (engine, fans, codes, 64u, result) == Status::OK);
    const int32_t sigma = ref_sqrt_ratio (2u, 120u);
    CHECK (result._M_sigma_q15 == sigma);
    CHECK (result._M_scale_q15 == ref_scale_shift (sigma,
                                                   _S_normal_truncation_q14,
                                                   14u));
    for (uint32_t i = 0u; i < 64u; ++i)
        CHECK ((codes[i] >= -127 && codes[i] <= 127));
    SplitMix64 other = SplitMix64::seeded (0x2468ull);
    int8_t again[64];
    InitResult second;
    REQUIRE (xavier_normal (other, fans, again, 64u, second) == Status::OK);
    for (uint32_t i = 0u; i < 64u; ++i)
        CHECK (again[i] == codes[i]);
    CHECK (second._M_clamped == result._M_clamped);
}

TEST_CASE ("kaiming and he limits follow gain * sqrt (3 / fan_in)")
{
    struct Row
    {
        uint32_t _M_fan_in;
        int32_t _M_gain;
    };
    const Row rows[] = { { 1u, 32768 },    { 3u, 32768 },   { 64u, 32768 },
                         { 255u, 32768 },  { 4u, 46341 },   { 128u, 46341 },
                         { 1u, 46341 },    { 1024u, 46341 } };
    for (const Row &row : rows)
        {
            int8_t codes[80];
            InitResult result;
            SplitMix64 engine = SplitMix64::seeded (row._M_fan_in);
            REQUIRE (kaiming_uniform (engine, row._M_fan_in, row._M_gain,
                                      codes, 80u, result)
                     == Status::OK);
            const int32_t limit = ref_scale_shift (
                ref_sqrt_ratio (3u, row._M_fan_in), row._M_gain, 15u);
            CHECK (result._M_scale_q15 == limit);
            for (uint32_t i = 0u; i < 80u; ++i)
                CHECK ((codes[i] >= -127 && codes[i] <= 127));
        }
    /* He uniform is exactly the Kaiming form at the rectifier gain sqrt(2). */
    {
        int8_t he[80];
        int8_t kaiming[80];
        InitResult he_result;
        InitResult kaiming_result;
        SplitMix64 e1 = SplitMix64::seeded (0x3C3Cull);
        SplitMix64 e2 = SplitMix64::seeded (0x3C3Cull);
        REQUIRE (he_uniform (e1, 16u, he, 80u, he_result) == Status::OK);
        REQUIRE (kaiming_uniform (e2, 16u, _S_he_gain_wide_q15, kaiming, 80u,
                                  kaiming_result)
                 == Status::OK);
        for (uint32_t i = 0u; i < 80u; ++i)
            CHECK (he[i] == kaiming[i]);
        CHECK (he_result._M_scale_q15 == kaiming_result._M_scale_q15);
    }
    /* Kaiming normal and He normal: sigma = gain / sqrt (fan_in). */
    {
        int8_t codes[80];
        InitResult result;
        SplitMix64 engine = SplitMix64::seeded (0x1717ull);
        REQUIRE (kaiming_normal (engine, 9u, 32768, codes, 80u, result)
                 == Status::OK);
        CHECK (result._M_sigma_q15
               == ref_scale_shift (ref_sqrt_ratio (1u, 9u), 32768, 15u));
        CHECK (result._M_scale_q15
               == ref_scale_shift (result._M_sigma_q15,
                                   _S_normal_truncation_q14, 14u));
        int8_t he_codes[80];
        InitResult he_result;
        SplitMix64 he_engine = SplitMix64::seeded (0x1717ull);
        REQUIRE (he_normal (he_engine, 9u, he_codes, 80u, he_result)
                 == Status::OK);
        CHECK (he_result._M_sigma_q15
               == ref_scale_shift (ref_sqrt_ratio (1u, 9u),
                                   _S_he_gain_wide_q15, 15u));
    }
    SECTION ("bad configurations report the documented Status")
    {
        int8_t codes[8];
        InitResult result;
        SplitMix64 engine = SplitMix64::seeded (1ull);
        CHECK (kaiming_uniform (engine, 0u, 32768, codes, 8u, result)
               == Status::OUT_OF_RANGE);
        CHECK (kaiming_uniform (engine, 65536u, 32768, codes, 8u, result)
               == Status::OUT_OF_RANGE);
        CHECK (kaiming_uniform (engine, 8u, 0, codes, 8u, result)
               == Status::BAD_LAYOUT);
        CHECK (kaiming_uniform (engine, 8u, -5, codes, 8u, result)
               == Status::BAD_LAYOUT);
        CHECK (kaiming_normal (engine, 0u, 32768, codes, 8u, result)
               == Status::OUT_OF_RANGE);
        CHECK (kaiming_normal (engine, 8u, 0, codes, 8u, result)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("fan_in_uniform limit is 1 / sqrt (fan_in)")
{
    const uint32_t fan_ins[] = { 1u, 2u, 16u, 255u, 65535u };
    for (const uint32_t fan_in : fan_ins)
        {
            int8_t codes[48];
            InitResult result;
            SplitMix64 engine = SplitMix64::seeded (fan_in + 0x90ull);
            REQUIRE (fan_in_uniform (engine, fan_in, codes, 48u, result)
                     == Status::OK);
            CHECK (result._M_scale_q15 == ref_sqrt_ratio (1u, fan_in));
        }
    int8_t codes[4];
    InitResult result;
    SplitMix64 engine = SplitMix64::seeded (1ull);
    CHECK (fan_in_uniform (engine, 0u, codes, 4u, result)
           == Status::OUT_OF_RANGE);
    CHECK (fan_in_uniform (engine, 65536u, codes, 4u, result)
           == Status::OUT_OF_RANGE);
}

TEST_CASE ("init_result_apply publishes only consistent scales")
{
    alignas (4) int8_t code_store[8];
    Trainable tensor
        = Trainable::make (code_store, nullptr, Shape::vector (8u), 0u);
    InitResult result;
    result._M_scale_q15 = 24000;
    result._M_elements = 8u;
    REQUIRE (init_result_apply (result, tensor) == Status::OK);
    CHECK (tensor._M_scale_q15 == 24000);

    result._M_elements = 7u;
    CHECK (init_result_apply (result, tensor) == Status::BAD_LAYOUT);
    CHECK (tensor._M_scale_q15 == 24000);

    result._M_elements = 8u;
    result._M_scale_q15 = 0;
    CHECK (init_result_apply (result, tensor) == Status::OUT_OF_RANGE);
    result._M_scale_q15 = -3;
    CHECK (init_result_apply (result, tensor) == Status::OUT_OF_RANGE);
    CHECK (tensor._M_scale_q15 == 24000);
}

TEST_CASE ("orthogonal shape bounds match the documented sides")
{
    CHECK (orthogonal_shape_ok (1u, 1u) == Status::OK);
    CHECK (orthogonal_shape_ok (32u, 512u) == Status::OK);
    CHECK (orthogonal_shape_ok (512u, 32u) == Status::OK);
    CHECK (orthogonal_shape_ok (0u, 4u) == Status::BAD_LAYOUT);
    CHECK (orthogonal_shape_ok (4u, 0u) == Status::BAD_LAYOUT);
    /* The short side is the smaller of the two, so 33 x 8 only trips the
       short-side bound when both sides exceed it. */
    CHECK (orthogonal_shape_ok (33u, 33u) == Status::OUT_OF_RANGE);
    CHECK (orthogonal_shape_ok (40u, 40u) == Status::OUT_OF_RANGE);
    CHECK (orthogonal_shape_ok (8u, 513u) == Status::OUT_OF_RANGE);
    CHECK (orthogonal_shape_ok (513u, 8u) == Status::OUT_OF_RANGE);
    /* 32 x 512 = 16384 elements fits _S_max_elements. */
    CHECK (_S_max_orthogonal_short_side == 32u);
    CHECK (_S_max_orthogonal_long_side == 512u);
    CHECK (orthogonal_shape_ok (32u, 32u) == Status::OK);
}

TEST_CASE ("orthogonal_init builds a deterministic semi-orthogonal code matrix")
{
    const uint32_t rows = 8u;
    const uint32_t cols = 8u;
    int16_t scratch[rows * cols];
    int8_t codes[rows * cols];
    InitResult result;
    SplitMix64 engine = SplitMix64::seeded (0xD0D0ull + rows);
    REQUIRE (orthogonal_init (engine, rows, cols, _S_q15_one, scratch, codes,
                              result)
             == Status::OK);
    CHECK (result._M_scale_q15 == _S_q15_one);
    CHECK (result._M_elements == rows * cols);
    for (uint32_t i = 0u; i < rows * cols; ++i)
        CHECK ((codes[i] >= -127 && codes[i] <= 127));

    uint32_t norm_error = 0u;
    uint32_t cross_dot = 0u;
    REQUIRE (orthogonal_defect (codes, rows, cols, norm_error, cross_dot)
             == Status::OK);
    /* The documented budget: int8 narrowing perturbs orthogonality by about
       one code step per entry, so both defects stay far below the unit 127^2. */
    CHECK (norm_error < 16129u / 2u);
    CHECK (cross_dot < 16129u / 2u);

    SECTION ("same seed reproduces, another seed does not")
    {
        int16_t again_scratch[rows * cols];
        int8_t again_codes[rows * cols];
        InitResult again;
        SplitMix64 second = SplitMix64::seeded (0xD0D0ull + rows);
        REQUIRE (orthogonal_init (second, rows, cols, _S_q15_one,
                                  again_scratch, again_codes, again)
                 == Status::OK);
        for (uint32_t i = 0u; i < rows * cols; ++i)
            CHECK (again_codes[i] == codes[i]);
        CHECK (again._M_clamped == result._M_clamped);

        int8_t other_codes[rows * cols];
        InitResult other;
        SplitMix64 third = SplitMix64::seeded (0xD1D1ull + rows);
        REQUIRE (orthogonal_init (third, rows, cols, _S_q15_one,
                                  again_scratch, other_codes, other)
                 == Status::OK);
        bool any_different = false;
        for (uint32_t i = 0u; i < rows * cols; ++i)
            any_different = any_different || other_codes[i] != codes[i];
        CHECK (any_different);
    }
    SECTION ("rectangular shapes and rejection paths")
    {
        int16_t wide_scratch[4 * 16];
        int8_t wide_codes[4 * 16];
        InitResult wide;
        SplitMix64 e = SplitMix64::seeded (0xBEE5ull);
        REQUIRE (orthogonal_init (e, 4u, 16u, 32768, wide_scratch, wide_codes,
                                  wide)
                 == Status::OK);
        CHECK (wide._M_elements == 64u);
        uint32_t norm_error = 0u;
        uint32_t cross_dot = 0u;
        REQUIRE (orthogonal_defect (wide_codes, 4u, 16u, norm_error,
                                    cross_dot)
                 == Status::OK);
        CHECK (norm_error < 16129u / 2u);

        int16_t tall_scratch[16 * 4];
        int8_t tall_codes[16 * 4];
        InitResult tall;
        SplitMix64 t = SplitMix64::seeded (0xBEE6ull);
        REQUIRE (orthogonal_init (t, 16u, 4u, 32768, tall_scratch, tall_codes,
                                  tall)
                 == Status::OK);
        REQUIRE (orthogonal_defect (tall_codes, 16u, 4u, norm_error,
                                    cross_dot)
                 == Status::OK);
        CHECK (norm_error < 16129u / 2u);

        /* Gain below one is a layout error; the shape check runs first. */
        SplitMix64 g = SplitMix64::seeded (1ull);
        CHECK (orthogonal_init (g, 4u, 4u, 0, wide_scratch, wide_codes, wide)
               == Status::BAD_LAYOUT);
        CHECK (orthogonal_init (g, 0u, 4u, 32768, wide_scratch, wide_codes,
                                wide)
               == Status::BAD_LAYOUT);
        CHECK (orthogonal_init (g, 33u, 33u, 32768, wide_scratch, wide_codes,
                                wide)
               == Status::OUT_OF_RANGE);
        CHECK (orthogonal_init (g, 4u, 4u, 32768, nullptr, wide_codes, wide)
               == Status::BAD_LAYOUT);
        CHECK (orthogonal_defect (nullptr, 4u, 4u, norm_error, cross_dot)
               == Status::BAD_LAYOUT);
    }
}
TEST_CASE ("init_weights dispatches every documented scheme onto the tensor")
{
    const InitKind kinds[] = {
        InitKind::Uniform,        InitKind::Normal,
        InitKind::XavierUniform,  InitKind::XavierNormal,
        InitKind::KaimingUniform, InitKind::KaimingNormal,
        InitKind::HeUniform,      InitKind::HeNormal,
        InitKind::FanInUniform,   InitKind::NormalClamped,
        InitKind::Constant,
    };
    const uint32_t count = 64u;
    alignas (4) int8_t code_store[count];
    alignas (4) int16_t grad_store[count];
    Trainable tensor = Trainable::make (code_store, grad_store,
                                        Shape::vector (count),
                                        Flag::RequiresGrad);
    const InitFans fans { 8u, 8u };
    for (const InitKind kind : kinds)
        {
            InitResult result;
            SplitMix64 engine = SplitMix64::seeded (0x1150ull);
            REQUIRE (init_weights (kind, engine, fans, 0u, 0u, nullptr,
                                   tensor, result)
                     == Status::OK);
            CHECK (result._M_elements == count);
            CHECK (result._M_scale_q15 > 0);
            REQUIRE (init_result_apply (result, tensor) == Status::OK);
            CHECK (tensor._M_scale_q15 == result._M_scale_q15);
            for (uint32_t i = 0u; i < count; ++i)
                CHECK ((code_store[i] >= -127 && code_store[i] <= 127));

            /* Determinism from the same seed for the same kind. */
            int8_t saved_codes[count];
            for (uint32_t i = 0u; i < count; ++i)
                saved_codes[i] = code_store[i];
            InitResult second;
            SplitMix64 replay = SplitMix64::seeded (0x1150ull);
            REQUIRE (init_weights (kind, replay, fans, 0u, 0u, nullptr,
                                   tensor, second)
                     == Status::OK);
            for (uint32_t i = 0u; i < count; ++i)
                CHECK (code_store[i] == saved_codes[i]);
            CHECK (second._M_scale_q15 == result._M_scale_q15);
            CHECK (second._M_clamped == result._M_clamped);
        }

    SECTION ("the scales match the closed forms of the named schemes")
    {
        InitResult result;
        SplitMix64 engine = SplitMix64::seeded (7ull);
        REQUIRE (init_weights (InitKind::XavierUniform, engine, fans, 0u, 0u,
                               nullptr, tensor, result)
                 == Status::OK);
        CHECK (result._M_scale_q15 == ref_sqrt_ratio (6u, 16u));

        REQUIRE (init_weights (InitKind::KaimingUniform, engine, fans, 0u, 0u,
                               nullptr, tensor, result)
                 == Status::OK);
        /* The dispatch uses the linear gain 1.0 for Kaiming. */
        CHECK (result._M_scale_q15
               == ref_scale_shift (ref_sqrt_ratio (3u, 8u), 32768, 15u));

        REQUIRE (init_weights (InitKind::HeNormal, engine, fans, 0u, 0u,
                               nullptr, tensor, result)
                 == Status::OK);
        CHECK (result._M_sigma_q15
               == ref_scale_shift (ref_sqrt_ratio (1u, 8u),
                                   _S_he_gain_wide_q15, 15u));

        REQUIRE (init_weights (InitKind::Normal, engine, fans, 0u, 0u,
                               nullptr, tensor, result)
                 == Status::OK);
        CHECK (result._M_sigma_q15 == _S_q15_one / 2);

        REQUIRE (init_weights (InitKind::Constant, engine, fans, 0u, 0u,
                               nullptr, tensor, result)
                 == Status::OK);
        CHECK (result._M_scale_q15 == _S_q15_one);
        for (uint32_t i = 0u; i < count; ++i)
            CHECK (tensor._M_code[i] == 127);
    }

    SECTION ("orthogonal is routed through the rows and cols arguments")
    {
        alignas (4) int16_t scratch[8 * 8];
        alignas (4) int8_t matrix_codes[8 * 8];
        Trainable matrix_tensor
            = Trainable::make (matrix_codes, nullptr, Shape::matrix (8u, 8u),
                               Flag::RequiresGrad);
        InitResult result;
        SplitMix64 engine = SplitMix64::seeded (0x0A0Aull);
        REQUIRE (init_weights (InitKind::Orthogonal, engine, fans, 8u, 8u,
                               scratch, matrix_tensor, result)
                 == Status::OK);
        CHECK (result._M_elements == 64u);
        CHECK (result._M_scale_q15 == _S_q15_one);
        for (uint32_t i = 0u; i < 64u; ++i)
            CHECK ((matrix_codes[i] >= -127 && matrix_codes[i] <= 127));
    }

    SECTION ("an invalid tensor view is rejected before any write")
    {
        Trainable bad;
        InitResult result;
        SplitMix64 engine = SplitMix64::seeded (1ull);
        CHECK (init_weights (InitKind::Uniform, engine, fans, 0u, 0u, nullptr,
                             bad, result)
               == Status::BAD_LAYOUT);
        alignas (4) int8_t store[4];
        Trainable mismatched
            = Trainable::make (store, nullptr, Shape::vector (4u), 0u);
        mismatched._M_count = 3u;
        CHECK (init_weights (InitKind::Uniform, engine, fans, 0u, 0u, nullptr,
                             mismatched, result)
               == Status::BAD_LAYOUT);
    }
}

/* ==================================================================== *
 * train_dropout.hpp: configuration, gain, mask, forward, backward.     *
 * ==================================================================== */

TEST_CASE ("dropout layout constants are fixed by the pool contract")
{
    CHECK (sizeof (DropoutConfig) == 48u);
    CHECK (sizeof (DropoutState) == 24u);
    CHECK (_S_dropout_mask == 0u);
    CHECK (_S_dropout_gain == 1u);
    CHECK (_S_dropout_scratch == 2u);
    CHECK (_S_max_operands == 4u);
    CHECK (_S_dropout_op_id == 0x0D01u);
    /* Pool cost of one recorded layer: mask, gain slot, scratch. */
    const uint32_t sizes[] = { 1u, 3u, 4u, 60u, 61u, 200u, 1023u };
    for (const uint32_t elements : sizes)
        CHECK (dropout_bytes (elements)
               == ref_align_up (elements) + ref_align_up (2u * elements) + 4u);
}

TEST_CASE ("dropout_gain_q14 is the documented 2^29 over keep")
{
    /* Spot values straight from the header: 16384 at keep 1.0, 32768 at
       keep 0.5, 0 at keep 0. */
    CHECK (dropout_gain_q14 (_S_q15_one) == 16384);
    CHECK (dropout_gain_q14 (K_q15_half) == 32768);
    CHECK (dropout_gain_q14 (0) == 0);
    CHECK (dropout_gain_q14 (-7) == 0);
    CHECK (dropout_gain_q14 (1) == 536870912);
    CHECK (dropout_gain_q14 (24576) == 21845);
    for (int32_t keep = -2; keep <= 32768; keep += 517)
        CHECK (dropout_gain_q14 (keep) == ref_gain_q14 (keep));
    /* The expected-value identity at p = 1/2: the gain is exactly 2.0, so a
       kept lane doubles and E[out] = E[in] in the Q15 domain. */
    CHECK (dropout_gain_q14 (K_q15_half) == 2 * 16384);
}

TEST_CASE ("dropout_alpha_fill_q15 matches -alpha p / (1 - p)")
{
    struct Row
    {
        int32_t _M_alpha;
        int32_t _M_drop;
        int32_t _M_keep;
        int32_t _M_fill;
    };
    const Row rows[] = {
        { 32768, 16384, 16384, -32768 },
        { 54732, 8192, 24576, -18244 },   /* 448364544 / 24576 is exact */
        { 0, 8192, 24576, 0 },
        { 32768, 0, 32768, 0 },
        /* keep one ulp above zero: the ratio is capped by the 2^30 rails. */
        { 32768, 32767, 1, -1073709056 },
        { 1000000, 16384, 16384, -1000000 },
        { -32768, 16384, 16384, 32768 },
    };
    for (const Row &row : rows)
        {
            CHECK (dropout_alpha_fill_q15 (row._M_alpha, row._M_drop,
                                           row._M_keep)
                   == row._M_fill);
            CHECK (dropout_alpha_fill_q15 (row._M_alpha, row._M_drop,
                                           row._M_keep)
                   == ref_alpha_fill (row._M_alpha, row._M_drop,
                                      row._M_keep));
        }
    /* Below the one-ulp keep the fill is 0 so the caller writes plain zeros. */
    CHECK (dropout_alpha_fill_q15 (32768, 32767, 0) == 0);
    CHECK (dropout_alpha_fill_q15 (32768, 32767, -1) == 0);
}

TEST_CASE ("dropout_config_init derives keep, gain and fill for every kind")
{
    const struct
    {
        DropoutKind _M_kind;
        bool _M_train;
        int32_t _M_drop;
        uint32_t _M_units;
        uint32_t _M_inner;
    } combos[]
        = {
            { DropoutKind::Elementwise, true, 0, 1u, 60u },
            { DropoutKind::Elementwise, true, 8192, 1u, 60u },
            { DropoutKind::Elementwise, false, 16384, 7u, 9u },
            { DropoutKind::Alpha, true, 16384, 3u, 20u },
            { DropoutKind::Alpha, true, 4000, 1u, 60u },
            { DropoutKind::Spatial, true, 8000, 6u, 10u },
            { DropoutKind::Spatial, false, 32767, 12u, 5u },
        };
    for (const auto &combo : combos)
        {
            const uint32_t elements = 60u;
            DropoutConfig config;
            REQUIRE (dropout_config_init (
                         config, combo._M_kind, combo._M_train, combo._M_drop,
                         0xA5A5ull, elements, combo._M_units, combo._M_inner,
                         32768)
                     == Status::OK);
            CHECK (config._M_seed == 0xA5A5ull);
            CHECK (config._M_train == (combo._M_train ? 1u : 0u));
            CHECK (config._M_kind == static_cast<uint8_t> (combo._M_kind));
            CHECK (config._M_elements == elements);
            CHECK (config._M_drop_q15 == combo._M_drop);
            CHECK (config._M_keep_q15 == _S_q15_one - combo._M_drop);
            CHECK (config._M_gain_q14 == ref_gain_q14 (config._M_keep_q15));
            if (combo._M_kind == DropoutKind::Spatial)
                {
                    CHECK (config._M_units == combo._M_units);
                    CHECK (config._M_inner == combo._M_inner);
                }
            else
                {
                    /* The factorisation arguments are ignored: a dense mask
                       is one unit over all elements. */
                    CHECK (config._M_units == 1u);
                    CHECK (config._M_inner == elements);
                }
            if (combo._M_kind == DropoutKind::Alpha)
                {
                    CHECK (config._M_alpha_wide_q15 == 32768);
                    CHECK (config._M_fill_wide_q15
                           == ref_alpha_fill (32768, combo._M_drop,
                                              config._M_keep_q15));
                }
            else
                {
                    /* Only AlphaDropout carries an alpha and its fill term. */
                    CHECK (config._M_alpha_wide_q15 == 0);
                    CHECK (config._M_fill_wide_q15 == 0);
                }
        }
}

TEST_CASE ("dropout_config_init rejects the documented bad configurations")
{
    DropoutConfig config;
    /* Elements below one and above the arena cap. */
    CHECK (dropout_config_init (config, DropoutKind::Elementwise, true, 100,
                                1ull, 0u, 1u, 1u, 0)
           == Status::OUT_OF_RANGE);
    CHECK (dropout_config_init (config, DropoutKind::Elementwise, true, 100,
                                1ull, _S_max_elements + 1u, 1u, 1u, 0)
           == Status::OUT_OF_RANGE);
    CHECK (dropout_config_init (config, DropoutKind::Elementwise, true, 100,
                                1ull, _S_max_elements, 1u, _S_max_elements, 0)
           == Status::OK);
    /* A drop rate outside [0, _S_q15_max]. */
    CHECK (dropout_config_init (config, DropoutKind::Elementwise, true, -1,
                                1ull, 8u, 1u, 8u, 0)
           == Status::OUT_OF_RANGE);
    CHECK (dropout_config_init (config, DropoutKind::Elementwise, true,
                                _S_q15_max + 1, 1ull, 8u, 1u, 8u, 0)
           == Status::OUT_OF_RANGE);
    CHECK (dropout_config_init (config, DropoutKind::Elementwise, true,
                                _S_q15_max, 1ull, 8u, 1u, 8u, 0)
           == Status::OK);
    /* Spatial factorisation mismatches. */
    CHECK (dropout_config_init (config, DropoutKind::Spatial, true, 100, 1ull,
                                60u, 4u, 10u, 0)
           == Status::BAD_LAYOUT);
    CHECK (dropout_config_init (config, DropoutKind::Spatial, true, 100, 1ull,
                                60u, 0u, 0u, 0)
           == Status::BAD_LAYOUT);
    CHECK (dropout_config_init (config, DropoutKind::Spatial, true, 100, 1ull,
                                60u, 60u, 0u, 0)
           == Status::BAD_LAYOUT);
    /* A zero unit count for a non-spatial kind is fine: units are forced
       to the one-by-elements factorisation. */
    CHECK (dropout_config_init (config, DropoutKind::Elementwise, true, 100,
                                1ull, 60u, 0u, 0u, 0)
           == Status::OK);
    CHECK (config._M_units == 1u);
    CHECK (config._M_inner == 60u);
}
TEST_CASE ("dropout_generate_mask reproduces the reference stream per kind")
{
    uint8_t mask[300];
    uint8_t ref_mask[300];

    SECTION ("elementwise spends one draw per element")
    {
        const int32_t drops[] = { 0, 4000, 16384, 24576, 32767 };
        for (const int32_t drop : drops)
            {
                DropoutConfig config;
                REQUIRE (dropout_config_init (
                             config, DropoutKind::Elementwise, true, drop,
                             0x9E9Eull, 256u, 1u, 256u, 0)
                         == Status::OK);
                XorShift64 engine = XorShift64::seeded (config._M_seed);
                RefXor ref_engine { config._M_seed };
                uint32_t kept = 999u;
                uint32_t draws = 999u;
                REQUIRE (dropout_generate_mask (config, engine, mask, kept,
                                                draws)
                         == Status::OK);
                const uint32_t ref_kept = ref_fill_mask (
                    ref_engine, config._M_keep_q15, ref_mask, 256u);
                CHECK (kept == ref_kept);
                CHECK (draws == 256u);
                for (uint32_t i = 0u; i < 256u; ++i)
                    CHECK (mask[i] == ref_mask[i]);
                CHECK (engine._M_state == ref_engine._M_state);
            }
        XorShift64 engine = XorShift64::seeded (1ull);
        uint32_t kept = 0u;
        uint32_t draws = 0u;
        DropoutConfig null_config;
        REQUIRE (dropout_config_init (null_config, DropoutKind::Elementwise,
                                      true, 100, 1ull, 8u, 1u, 8u, 0)
                 == Status::OK);
        CHECK (dropout_generate_mask (null_config, engine, nullptr, kept,
                                      draws)
               == Status::BAD_LAYOUT);
    }
    SECTION ("spatial spends one draw per unit and replicates inner")
    {
        struct Row
        {
            uint32_t _M_units;
            uint32_t _M_inner;
        };
        const Row rows[] = { { 1u, 60u }, { 60u, 1u }, { 6u, 10u },
                             { 12u, 5u } };
        for (const Row &row : rows)
            {
                DropoutConfig config;
                REQUIRE (dropout_config_init (
                             config, DropoutKind::Spatial, true, 12000,
                             0x5A5Aull, row._M_units * row._M_inner,
                             row._M_units, row._M_inner, 0)
                         == Status::OK);
                XorShift64 engine = XorShift64::seeded (config._M_seed);
                RefXor ref_engine { config._M_seed };
                uint32_t kept = 999u;
                uint32_t draws = 999u;
                REQUIRE (dropout_generate_mask (config, engine, mask, kept,
                                                draws)
                         == Status::OK);
                CHECK (draws == row._M_units);
                const uint32_t ref_kept = ref_fill_mask_blocked (
                    ref_engine, config._M_keep_q15, row._M_units,
                    row._M_inner, ref_mask);
                CHECK (kept == ref_kept);
                for (uint32_t unit = 0u; unit < row._M_units; ++unit)
                    for (uint32_t lane = 0u; lane < row._M_inner; ++lane)
                        {
                            const uint32_t i = unit * row._M_inner + lane;
                            CHECK (mask[i] == ref_mask[i]);
                            CHECK (mask[i] == mask[unit * row._M_inner]);
                        }
                CHECK (kept % row._M_inner == 0u);
            }
    }
}

TEST_CASE ("dropout_forward_codes: eval identity, rate extremes, gain rails")
{
    const uint32_t n = 64u;
    int8_t in[64];
    int8_t out[64];
    uint8_t all_keep[64];
    uint8_t none_kept[64];
    for (uint32_t i = 0u; i < n; ++i)
        {
            in[i] = static_cast<int8_t> ((i * 7u) % 255u) - 127;
            all_keep[i] = 1u;
            none_kept[i] = 0u;
        }
    uint32_t saturated = 0u;

    SECTION ("eval mode is the identity even with no mask")
    {
        DropoutConfig config;
        REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, false,
                                      12000, 3ull, n, 1u, n, 0)
                 == Status::OK);
        saturated = 77u;
        REQUIRE (dropout_forward_codes (config, in, nullptr, out, saturated)
                 == Status::OK);
        for (uint32_t i = 0u; i < n; ++i)
            CHECK (out[i] == in[i]);
        CHECK (saturated == 77u); /* untouched: eval counts nothing */
    }
    SECTION ("rate 0 with gain 16384 is a plain copy in train mode")
    {
        DropoutConfig config;
        REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true,
                                      0, 3ull, n, 1u, n, 0)
                 == Status::OK);
        CHECK (config._M_gain_q14 == 16384);
        saturated = 0u;
        REQUIRE (dropout_forward_codes (config, in, all_keep, out, saturated)
                 == Status::OK);
        for (uint32_t i = 0u; i < n; ++i)
            CHECK (out[i] == in[i]);
        CHECK (saturated == 0u);
    }
    SECTION ("an all-dropped mask writes the fill, zero for non-alpha")
    {
        DropoutConfig config;
        REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true,
                                      _S_q15_max, 3ull, n, 1u, n, 0)
                 == Status::OK);
        saturated = 0u;
        REQUIRE (dropout_forward_codes (config, in, none_kept, out, saturated)
                 == Status::OK);
        for (uint32_t i = 0u; i < n; ++i)
            CHECK (out[i] == 0);
        CHECK (saturated == 0u);
    }
    SECTION ("kept lanes are multiplied by the Q14 gain and saturate")
    {
        /* gain(keep = 1) = 2^29: every non-zero lane saturates onto the rails
           and is counted. */
        DropoutConfig rail;
        REQUIRE (dropout_config_init (rail, DropoutKind::Elementwise, true,
                                      _S_q15_max, 3ull, n, 1u, n, 0)
                 == Status::OK);
        saturated = 0u;
        REQUIRE (dropout_forward_codes (rail, in, all_keep, out, saturated)
                 == Status::OK);
        uint32_t ref_saturated = 0u;
        for (uint32_t i = 0u; i < n; ++i)
            {
                const int8_t want = ref_forward_lane (in[i], true,
                                                      rail._M_gain_q14, 0,
                                                      ref_saturated);
                CHECK (out[i] == want);
            }
        CHECK (saturated == ref_saturated);
        CHECK (saturated > 0u);

        /* keep = 0.5: the gain is exactly 2.0 on the code grid. */
        DropoutConfig half;
        REQUIRE (dropout_config_init (half, DropoutKind::Elementwise, true,
                                      K_q15_half, 3ull, n, 1u, n, 0)
                 == Status::OK);
        CHECK (half._M_gain_q14 == 32768);
        int8_t small[64];
        for (uint32_t i = 0u; i < n; ++i)
            small[i] = static_cast<int8_t> ((i % 60u) - 30);
        saturated = 0u;
        REQUIRE (dropout_forward_codes (half, small, all_keep, out, saturated)
                 == Status::OK);
        uint32_t ref_sat = 0u;
        for (uint32_t i = 0u; i < n; ++i)
            {
                const int8_t want = ref_forward_lane (small[i], true, 32768,
                                                      0, ref_sat);
                CHECK (out[i] == want);
            }
        CHECK (saturated == ref_sat);
        CHECK (saturated == 0u);
    }
    SECTION ("the inverted gain preserves the expected value in Q15")
    {
        /* All lanes at code 40 with p = 1/2: every kept lane carries 80
           (below the rails), dropped lanes carry 0, so the output sum is
           kept * 80 and the reference kept count fixes it exactly. */
        DropoutConfig config;
        REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true,
                                      K_q15_half, 0x1234ull, n, 1u, n, 0)
                 == Status::OK);
        for (uint32_t i = 0u; i < n; ++i)
            in[i] = 40;
        uint8_t mask[64];
        uint8_t ref_mask[64];
        RefXor ref_engine { 0x1234ull };
        const uint32_t kept = ref_fill_mask (ref_engine, config._M_keep_q15,
                                             ref_mask, n);
        saturated = 0u;
        XorShift64 engine = XorShift64::seeded (config._M_seed);
        uint32_t lib_kept = 0u;
        uint32_t draws = 0u;
        REQUIRE (dropout_generate_mask (config, engine, mask, lib_kept, draws)
                 == Status::OK);
        REQUIRE (dropout_forward_codes (config, in, mask, out, saturated)
                 == Status::OK);
        int32_t sum = 0;
        for (uint32_t i = 0u; i < n; ++i)
            {
                CHECK (mask[i] == ref_mask[i]);
                sum += out[i];
            }
        CHECK (sum == static_cast<int32_t> (lib_kept) * 80);
        CHECK (lib_kept == kept);
        /* The binomial spread of the kept count bounds how far the
           empirical mean may drift from the input mean on this seed. */
        const int32_t deviation = static_cast<int32_t> (lib_kept) - 32;
        CHECK (deviation <= 20);
        CHECK (deviation >= -20);
        CHECK (sum - static_cast<int32_t> (n) * 40 == deviation * 80);
    }
    SECTION ("null buffers are refused")
    {
        DropoutConfig config;
        REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true,
                                      100, 3ull, n, 1u, n, 0)
                 == Status::OK);
        CHECK (dropout_forward_codes (config, nullptr, all_keep, out,
                                      saturated)
               == Status::BAD_LAYOUT);
        CHECK (dropout_forward_codes (config, in, all_keep, nullptr, saturated)
               == Status::BAD_LAYOUT);
        CHECK (dropout_forward_codes (config, in, nullptr, out, saturated)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("alpha dropout fills dropped lanes with the moment-matched value")
{
    const uint32_t n = 100u;
    int8_t in[100];
    int8_t out[100];
    uint8_t mask[100];
    for (uint32_t i = 0u; i < n; ++i)
        {
            in[i] = static_cast<int8_t> (i % 2u ? 50 : -50);
            mask[i] = static_cast<uint8_t> (i % 3u == 0u ? 0u : 1u);
        }
    DropoutConfig config;
    REQUIRE (dropout_config_init (config, DropoutKind::Alpha, true, K_q15_half,
                                  0xB1B1ull, n, 1u, n, 32768)
             == Status::OK);
    /* alpha = 1.0, p = 1/2, keep = 1/2: fill = -1.0 in wide Q15 = -32768,
       which narrows to code -127 on the documented grid. */
    CHECK (config._M_fill_wide_q15 == -32768);
    const int8_t fill_code = ref_q15_to_code (config._M_fill_wide_q15);
    CHECK (fill_code == -127);
    uint32_t saturated = 0u;
    REQUIRE (dropout_forward_codes (config, in, mask, out, saturated)
             == Status::OK);
    uint32_t ref_saturated = 0u;
    for (uint32_t i = 0u; i < n; ++i)
        {
            const int8_t want = ref_forward_lane (in[i], mask[i] != 0u,
                                                  config._M_gain_q14,
                                                  config._M_fill_wide_q15,
                                                  ref_saturated);
            CHECK (out[i] == want);
            if (mask[i] == 0u)
                CHECK (out[i] == fill_code);
        }
    CHECK (saturated == ref_saturated);
    /* The non-alpha kinds keep the fill at zero regardless of alpha. */
    DropoutConfig dense;
    REQUIRE (dropout_config_init (dense, DropoutKind::Elementwise, true,
                                  K_q15_half, 0xB1B1ull, n, 1u, n, 32768)
             == Status::OK);
    CHECK (dense._M_fill_wide_q15 == 0);
    saturated = 0u;
    REQUIRE (dropout_forward_codes (dense, in, mask, out, saturated)
             == Status::OK);
    for (uint32_t i = 0u; i < n; ++i)
        if (mask[i] == 0u)
            CHECK (out[i] == 0);
}
TEST_CASE ("dropout_run records mask, gain and scratch and matches the golden forward")
{
    Fixture fix;
    const uint32_t elements = 200u;
    const int32_t drop = 8000;
    const uint64_t seed = 0x31337ull;

    DropoutConfig config;
    REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true, drop,
                                  seed, elements, 1u, elements, 0)
             == Status::OK);
    for (uint32_t i = 0u; i < elements; ++i)
        {
            fix._M_in_codes[i] = static_cast<int8_t> ((i * 37u) % 255u) - 127;
            fix._M_out_codes[i] = 0;
            fix._M_in_grad[i] = 0;
            fix._M_out_grad[i] = 0;
        }
    Trainable input = Trainable::make (fix._M_in_codes, fix._M_in_grad,
                                       Shape::vector (elements),
                                       Flag::RequiresGrad);
    Trainable output = Trainable::make (fix._M_out_codes, fix._M_out_grad,
                                        Shape::vector (elements),
                                        Flag::RequiresGrad);
    REQUIRE (fix._M_tape.begin () == Status::OK);

    DropoutState state;
    REQUIRE (dropout_run (fix._M_tape, config, input, output, &state)
             == Status::OK);

    /* One node, one operand, three saved slots at the documented indices. */
    REQUIRE (fix._M_tape.node_count () == 1u);
    Node *node = fix._M_tape.node (0u);
    REQUIRE (node != nullptr);
    CHECK (node->_M_op_id == _S_dropout_op_id);
    CHECK (node->_M_operand_count == 1u);
    CHECK (node->_M_saved_count == 3u);
    CHECK (node->_M_operands[0]._M_code == fix._M_in_codes);
    CHECK (node->_M_saved[_S_dropout_mask]._M_bytes == ref_align_up (elements));
    CHECK (node->_M_saved[_S_dropout_gain]._M_bytes == 4u);
    CHECK (node->_M_saved[_S_dropout_scratch]._M_bytes
           == ref_align_up (2u * elements));
    CHECK (fix._M_tape.stats()._M_saved_bytes == dropout_bytes (elements));

    /* The golden mask and forward. */
    uint8_t ref_mask[200];
    uint64_t ref_state_after = 0ull;
    const uint32_t ref_kept = ref_draw_masks_flat (seed, _S_q15_one - drop,
                                                   elements, ref_mask,
                                                   &ref_state_after);
    const uint8_t *const pool_mask = reinterpret_cast<const uint8_t *> (
        fix._M_pool.at (node->_M_saved[_S_dropout_mask]._M_offset));
    uint32_t ref_saturated = 0u;
    for (uint32_t i = 0u; i < elements; ++i)
        {
            CHECK (pool_mask[i] == ref_mask[i]);
            const int8_t want = ref_forward_lane (
                fix._M_in_codes[i], ref_mask[i] != 0u, config._M_gain_q14,
                config._M_fill_wide_q15, ref_saturated);
            CHECK (fix._M_out_codes[i] == want);
        }

    /* The stored gain is the config's Q14 gain, readable from the pool. */
    const int32_t *const pool_gain = reinterpret_cast<const int32_t *> (
        fix._M_pool.at (node->_M_saved[_S_dropout_gain]._M_offset));
    CHECK (*pool_gain == config._M_gain_q14);
    CHECK (*pool_gain == ref_gain_q14 (_S_q15_one - drop));

    /* The reported state: engine continuation, kept, draws, saturation,
       and the code-domain drop parameter. */
    CHECK (state._M_engine_state == ref_state_after);
    CHECK (state._M_kept == ref_kept);
    CHECK (state._M_draws == elements);
    CHECK (state._M_saturated == ref_saturated);
    CHECK (state._M_drop_code
           == static_cast<uint32_t> (ref_aux_code (drop)));

    SECTION ("spatial factorisation through the same tape path")
    {
        Fixture spatial;
        const uint32_t units = 12u;
        const uint32_t inner = 10u;
        const uint32_t total = units * inner;
        DropoutConfig spatial_config;
        REQUIRE (dropout_config_init (spatial_config, DropoutKind::Spatial,
                                      true, 12000, 0x2020ull, total, units,
                                      inner, 0)
                 == Status::OK);
        for (uint32_t i = 0u; i < total; ++i)
            spatial._M_in_codes[i] = static_cast<int8_t> ((i % 2u) ? 60 : -60);
        Trainable s_in = Trainable::make (spatial._M_in_codes,
                                          spatial._M_in_grad,
                                          Shape::vector (total),
                                          Flag::RequiresGrad);
        Trainable s_out = Trainable::make (spatial._M_out_codes,
                                           spatial._M_out_grad,
                                           Shape::vector (total),
                                           Flag::RequiresGrad);
        REQUIRE (spatial._M_tape.begin () == Status::OK);
        DropoutState s_state;
        REQUIRE (dropout_run (spatial._M_tape, spatial_config, s_in, s_out,
                              &s_state)
                 == Status::OK);
        CHECK (s_state._M_draws == units);
        uint8_t ref_blocks[120];
        uint64_t after = 0ull;
        const uint32_t ref_kept_blocks = ref_draw_masks (
            0x2020ull, spatial_config._M_keep_q15, units, inner,
            ref_blocks, &after);
        CHECK (s_state._M_kept == ref_kept_blocks);
        CHECK (s_state._M_engine_state == after);
        Node *s_node = spatial._M_tape.node (0u);
        REQUIRE (s_node != nullptr);
        const uint8_t *const blocks = reinterpret_cast<const uint8_t *> (
            spatial._M_pool.at (s_node->_M_saved[_S_dropout_mask]._M_offset));
        uint32_t ref_sat = 0u;
        for (uint32_t unit = 0u; unit < units; ++unit)
            for (uint32_t lane = 0u; lane < inner; ++lane)
                {
                    const uint32_t i = unit * inner + lane;
                    CHECK (blocks[i] == ref_blocks[i]);
                    CHECK (blocks[i] == blocks[unit * inner]);
                    CHECK (spatial._M_out_codes[i]
                           == ref_forward_lane (spatial._M_in_codes[i],
                                                blocks[i] != 0u,
                                                spatial_config._M_gain_q14, 0,
                                                ref_sat));
                }
        CHECK (s_state._M_saturated == ref_sat);
    }
}

TEST_CASE ("dropout_run eval mode copies without touching the tape")
{
    Fixture fix;
    const uint32_t elements = 64u;
    DropoutConfig config;
    REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, false,
                                  12000, 0x777ull, elements, 1u, elements, 0)
             == Status::OK);
    for (uint32_t i = 0u; i < elements; ++i)
        {
            fix._M_in_codes[i] = static_cast<int8_t> (static_cast<int> (i) - 32);
            fix._M_out_codes[i] = 5;
        }
    Trainable input = Trainable::make (fix._M_in_codes, nullptr,
                                       Shape::vector (elements), 0u);
    Trainable output = Trainable::make (fix._M_out_codes, nullptr,
                                        Shape::vector (elements), 0u);
    REQUIRE (fix._M_tape.begin () == Status::OK);
    DropoutState state;
    REQUIRE (dropout_run (fix._M_tape, config, input, output, &state)
             == Status::OK);
    for (uint32_t i = 0u; i < elements; ++i)
        CHECK (fix._M_out_codes[i] == fix._M_in_codes[i]);
    CHECK (fix._M_tape.node_count () == 0u);
    CHECK (fix._M_tape.stats()._M_saved_bytes == 0u);
    /* No draws: the engine state is the seed replacement, nothing counted. */
    CHECK (state._M_engine_state == 0x777ull);
    CHECK (state._M_kept == 0u);
    CHECK (state._M_draws == 0u);
    CHECK (state._M_saturated == 0u);
    /* With no state pointer the eval path still works. */
    REQUIRE (dropout_run (fix._M_tape, config, input, output, nullptr)
             == Status::OK);
}

TEST_CASE ("dropout_run refuses a training op on a closed tape with INTERNAL")
{
    Fixture fix;
    const uint32_t elements = 16u;
    DropoutConfig config;
    REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true, 100,
                                  5ull, elements, 1u, elements, 0)
             == Status::OK);
    Trainable input = Trainable::make (fix._M_in_codes, nullptr,
                                       Shape::vector (elements), 0u);
    Trainable output = Trainable::make (fix._M_out_codes, nullptr,
                                        Shape::vector (elements), 0u);
    /* Never begun: is_open() is false from construction. */
    CHECK (dropout_run (fix._M_tape, config, input, output)
           == Status::INTERNAL);
    REQUIRE (fix._M_tape.begin () == Status::OK);
    fix._M_tape.reset ();
    /* Reset closes the pass again. */
    CHECK (dropout_run (fix._M_tape, config, input, output)
           == Status::INTERNAL);
    CHECK (fix._M_tape.node_count () == 0u);
}

TEST_CASE ("dropout_run rejects layout mismatches with BAD_LAYOUT")
{
    Fixture fix;
    DropoutConfig config;
    REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true, 100,
                                  9ull, 8u, 1u, 8u, 0)
             == Status::OK);
    REQUIRE (fix._M_tape.begin () == Status::OK);
    Trainable right = Trainable::make (fix._M_in_codes, nullptr,
                                       Shape::vector (8u), 0u);
    Trainable wrong_count = Trainable::make (fix._M_out_codes, nullptr,
                                             Shape::vector (6u), 0u);
    CHECK (dropout_run (fix._M_tape, config, right, wrong_count)
           == Status::BAD_LAYOUT);
    CHECK (dropout_run (fix._M_tape, config, wrong_count, right)
           == Status::BAD_LAYOUT);
    /* Same element count, different shape: the views must agree. */
    Trainable reshaped = Trainable::make (fix._M_out_codes, nullptr,
                                          Shape::matrix (2u, 4u), 0u);
    CHECK (dropout_run (fix._M_tape, config, right, reshaped)
           == Status::BAD_LAYOUT);
    /* A tensor with no code buffer fails validation before any draw. */
    Trainable null_code
        = Trainable::make (nullptr, nullptr, Shape::vector (8u), 0u);
    CHECK (dropout_run (fix._M_tape, config, null_code, right)
           == Status::BAD_LAYOUT);
    CHECK (fix._M_tape.node_count () == 0u);
}

TEST_CASE ("dropout_run surfaces arena overflow as a Status, never a crash")
{
    SECTION ("the backward scratch no longer fits the 8 KiB training arena")
    {
        Fixture fix;
        const uint32_t elements = 3000u; /* 3000 + 4 + 6000 > 8192 */
        DropoutConfig config;
        REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true,
                                      4000, 3ull, elements, 1u, elements, 0)
                 == Status::OK);
        Trainable input = Trainable::make (fix._M_in_codes, fix._M_in_grad,
                                           Shape::vector (elements), 0u);
        Trainable output = Trainable::make (fix._M_out_codes,
                                            fix._M_out_grad,
                                            Shape::vector (elements), 0u);
        for (uint32_t i = 0u; i < elements; ++i)
            {
                fix._M_in_codes[i] = 1;
                fix._M_out_codes[i] = 0;
            }
        REQUIRE (fix._M_tape.begin () == Status::OK);
        CHECK (dropout_run (fix._M_tape, config, input, output)
               == Status::ARENA_OVERFLOW);
        CHECK (fix._M_tape.node_count () == 0u);
        CHECK (fix._M_pool.stats()._M_overflows >= 1u);
        /* The output was never written: the forward is skipped on overflow. */
        for (uint32_t i = 0u; i < elements; ++i)
            CHECK (fix._M_out_codes[i] == 0);
    }
    SECTION ("even the mask slab does not fit a tiny pool")
    {
        alignas (4) uint8_t small_backing[512];
        Pool small_pool (small_backing, sizeof (small_backing));
        Node nodes[2];
        Tape small_tape (nodes, 2u, small_pool);
        REQUIRE (small_tape.begin () == Status::OK);
        const uint32_t elements = 600u;
        DropoutConfig config;
        REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true,
                                      4000, 3ull, elements, 1u, elements, 0)
                 == Status::OK);
        alignas (4) static int8_t in_codes[600];
        alignas (4) static int8_t out_codes[600];
        Trainable input
            = Trainable::make (in_codes, nullptr, Shape::vector (elements), 0u);
        Trainable output = Trainable::make (out_codes, nullptr,
                                            Shape::vector (elements), 0u);
        CHECK (dropout_run (small_tape, config, input, output)
               == Status::ARENA_OVERFLOW);
        CHECK (small_pool.stats()._M_overflows >= 1u);
    }
}

TEST_CASE ("dropout_backward replays the recorded mask, not a fresh draw")
{
    Fixture fix;
    const uint32_t elements = 120u;
    const int32_t drop = 12000;
    const uint64_t seed = 0xF100ull;
    DropoutConfig config;
    REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true, drop,
                                  seed, elements, 1u, elements, 0)
             == Status::OK);
    for (uint32_t i = 0u; i < elements; ++i)
        {
            fix._M_in_codes[i] = static_cast<int8_t> (i % 5u);
            fix._M_in_grad[i] = 0;
            fix._M_out_grad[i] = static_cast<int16_t> (200 - (i % 40) * 10);
        }
    Trainable input = Trainable::make (fix._M_in_codes, fix._M_in_grad,
                                       Shape::vector (elements),
                                       Flag::RequiresGrad);
    Trainable output = Trainable::make (fix._M_out_codes, fix._M_out_grad,
                                        Shape::vector (elements),
                                        Flag::RequiresGrad);
    REQUIRE (fix._M_tape.begin () == Status::OK);
    REQUIRE (dropout_run (fix._M_tape, config, input, output) == Status::OK);
    Node *node = fix._M_tape.node (0u);
    REQUIRE (node != nullptr);

    /* The forward mask, re-derived from the seed: the backward must route by
       the tape's saved copy of this very mask.  A backward that regenerated
       the stream would see a different routing because the forward already
       consumed the draws it would reuse. */
    uint8_t ref_mask[120];
    ref_draw_masks_flat (seed, config._M_keep_q15, elements, ref_mask, nullptr);
    GradContext ctx;
    ctx._M_pool = &fix._M_pool;
    ctx._M_operands = node->_M_operands;
    ctx._M_operand_count = node->_M_operand_count;
    ctx._M_saved = node->_M_saved;
    ctx._M_saved_count = node->_M_saved_count;
    const uint8_t *const saved_mask = ctx.saved_as<uint8_t> (_S_dropout_mask);
    REQUIRE (saved_mask != nullptr);
    for (uint32_t i = 0u; i < elements; ++i)
        CHECK (saved_mask[i] == ref_mask[i]);

    uint32_t ref_saturated = 0u;
    REQUIRE (dropout_backward (output, output, ctx) == Status::OK);
    for (uint32_t i = 0u; i < elements; ++i)
        {
            int32_t lane = 0;
            ref_backward_lane (fix._M_out_grad[i], ref_mask[i],
                               config._M_gain_q14, 0, lane, ref_saturated);
            CHECK (fix._M_in_grad[i] == static_cast<int16_t> (lane));
        }
    CHECK (ctx._M_saturation_events == ref_saturated);

    /* Accumulate-only contract: a second replay adds to the first result. */
    int16_t first_round[120];
    for (uint32_t i = 0u; i < elements; ++i)
        first_round[i] = fix._M_in_grad[i];
    ctx._M_saturation_events = 0u;
    uint32_t second_saturated = 0u;
    REQUIRE (dropout_backward (output, output, ctx) == Status::OK);
    for (uint32_t i = 0u; i < elements; ++i)
        {
            int32_t lane = 0;
            ref_backward_lane (fix._M_out_grad[i], ref_mask[i],
                               config._M_gain_q14, first_round[i], lane,
                               second_saturated);
            CHECK (fix._M_in_grad[i] == static_cast<int16_t> (lane));
        }
    CHECK (ctx._M_saturation_events == second_saturated);

    SECTION ("p = 0 backward is the plain route (gain short-circuit)")
    {
        Fixture zero;
        DropoutConfig zero_config;
        REQUIRE (dropout_config_init (zero_config, DropoutKind::Elementwise,
                                      true, 0, 0x1ull, 32u, 1u, 32u, 0)
                 == Status::OK);
        CHECK (zero_config._M_gain_q14 == 16384);
        for (uint32_t i = 0u; i < 32u; ++i)
            {
                zero._M_in_grad[i] = 0;
                zero._M_out_grad[i] = static_cast<int16_t> (i * 13 - 200);
            }
        Trainable z_in = Trainable::make (zero._M_in_codes, zero._M_in_grad,
                                          Shape::vector (32u),
                                          Flag::RequiresGrad);
        Trainable z_out = Trainable::make (zero._M_out_codes,
                                           zero._M_out_grad,
                                           Shape::vector (32u), 0u);
        REQUIRE (zero._M_tape.begin () == Status::OK);
        REQUIRE (dropout_run (zero._M_tape, zero_config, z_in, z_out)
                 == Status::OK);
        Node *z_node = zero._M_tape.node (0u);
        GradContext z_ctx;
        z_ctx._M_pool = &zero._M_pool;
        z_ctx._M_operands = z_node->_M_operands;
        z_ctx._M_operand_count = z_node->_M_operand_count;
        z_ctx._M_saved = z_node->_M_saved;
        z_ctx._M_saved_count = z_node->_M_saved_count;
        /* keep = 1.0: the mask is all ones, so grad_in == grad_out exactly. */
        uint8_t ones[32];
        ref_draw_masks_flat (0x1ull, 32768, 32u, ones, nullptr);
        for (uint32_t i = 0u; i < 32u; ++i)
            CHECK (ones[i] == 1u);
        REQUIRE (dropout_backward (z_out, z_out, z_ctx) == Status::OK);
        uint32_t sat = 0u;
        for (uint32_t i = 0u; i < 32u; ++i)
            {
                int32_t lane = 0;
                ref_backward_lane (zero._M_out_grad[i], 1u, 16384, 0, lane,
                                   sat);
                CHECK (zero._M_in_grad[i] == static_cast<int16_t> (lane));
                CHECK (zero._M_in_grad[i] == zero._M_out_grad[i]);
            }
        CHECK (z_ctx._M_saturation_events == 0u);
    }
    SECTION ("the backward refuses incomplete contexts")
    {
        GradContext empty_ctx;
        CHECK (dropout_backward (output, output, empty_ctx)
               == Status::BAD_LAYOUT);
        GradContext no_saved;
        no_saved._M_pool = &fix._M_pool;
        no_saved._M_operands = node->_M_operands;
        no_saved._M_operand_count = node->_M_operand_count;
        Saved null_slots[3];
        no_saved._M_saved = null_slots;
        no_saved._M_saved_count = 3u;
        CHECK (dropout_backward (output, output, no_saved)
               == Status::BAD_LAYOUT);
        /* A frozen input (no gradient lanes) is a documented no-op. */
        node->_M_operands[0]._M_grad = nullptr;
        CHECK (dropout_backward (output, output, ctx) == Status::OK);
    }
}

TEST_CASE ("tape.backward chains the dropout gradient and counts saturation")
{
    Fixture fix;
    const uint32_t elements = 64u;
    DropoutConfig config;
    REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true,
                                  K_q15_half, 0xABCDull, elements, 1u,
                                  elements, 0)
             == Status::OK);
    for (uint32_t i = 0u; i < elements; ++i)
        {
            fix._M_in_grad[i] = 0;
            fix._M_out_grad[i] = static_cast<int16_t> (i % 2u ? 30000 : -30000);
        }
    Trainable input = Trainable::make (fix._M_in_codes, fix._M_in_grad,
                                       Shape::vector (elements),
                                       Flag::RequiresGrad);
    Trainable output = Trainable::make (fix._M_out_codes, fix._M_out_grad,
                                        Shape::vector (elements), 0u);
    REQUIRE (fix._M_tape.begin () == Status::OK);
    REQUIRE (dropout_run (fix._M_tape, config, input, output) == Status::OK);

    /* The single-node graph: the tape seeds lane 0 of the output gradient
       with the loss scale (q15_clamp(32768) = 32767) and routes the rest. */
    REQUIRE (fix._M_tape.backward (output) == Status::OK);
    uint8_t ref_mask[64];
    ref_draw_masks_flat (0xABCDull, config._M_keep_q15, elements, ref_mask,
                         nullptr);
    uint32_t ref_saturated = 0u;
    for (uint32_t i = 0u; i < elements; ++i)
        {
            const int16_t seeded
                = i == 0u ? static_cast<int16_t> (32767) : fix._M_out_grad[i];
            int32_t lane = 0;
            ref_backward_lane (seeded, ref_mask[i], config._M_gain_q14, 0,
                               lane, ref_saturated);
            CHECK (fix._M_in_grad[i] == static_cast<int16_t> (lane));
        }
    CHECK (fix._M_tape.stats()._M_saturation_events == ref_saturated);
    /* With the doubling gain every kept lane of magnitude 30000 saturates. */
    CHECK (ref_saturated > 0u);
}

TEST_CASE ("dropout_route_grad_q15 applies the mask and the Q14 gain")
{
    const uint32_t elements = 48u;
    int16_t grad_out[48];
    int16_t grad_in[48];
    uint8_t mask[48];
    for (uint32_t i = 0u; i < elements; ++i)
        {
            grad_out[i] = static_cast<int16_t> (static_cast<int> (i) * 260 - 6000);
            mask[i] = static_cast<uint8_t> (i % 3u == 0u ? 0u : 1u);
        }
    SECTION ("identity gain short-circuits after the route")
    {
        DropoutConfig config;
        REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true,
                                      0, 1ull, elements, 1u, elements, 0)
                 == Status::OK);
        uint32_t saturated = 0u;
        REQUIRE (dropout_route_grad_q15 (config, grad_out, mask, grad_in,
                                         saturated)
                 == Status::OK);
        for (uint32_t i = 0u; i < elements; ++i)
            CHECK (grad_in[i] == (mask[i] != 0u ? grad_out[i] : 0));
        CHECK (saturated == 0u);
    }
    SECTION ("doubling gain saturates into the Q15 rails and counts")
    {
        DropoutConfig config;
        REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true,
                                      K_q15_half, 1ull, elements, 1u,
                                      elements, 0)
                 == Status::OK);
        CHECK (config._M_gain_q14 == 32768);
        grad_out[4] = 30000;
        grad_out[7] = -30000;
        uint32_t saturated = 0u;
        REQUIRE (dropout_route_grad_q15 (config, grad_out, mask, grad_in,
                                         saturated)
                 == Status::OK);
        uint32_t ref_saturated = 0u;
        for (uint32_t i = 0u; i < elements; ++i)
            {
                const int32_t routed = mask[i] != 0u ? grad_out[i] : 0;
                const int32_t scaled = ref_round_shift (
                    static_cast<int64_t> (routed) * 32768, 14u);
                const int32_t clamped = scaled > 32767
                                            ? 32767
                                            : (scaled < -32768 ? -32768 : scaled);
                if (clamped != scaled)
                    ++ref_saturated;
                CHECK (grad_in[i] == static_cast<int16_t> (clamped));
            }
        CHECK (saturated == ref_saturated);
        CHECK (saturated >= 2u);
    }
    SECTION ("null buffers are refused")
    {
        DropoutConfig config;
        REQUIRE (dropout_config_init (config, DropoutKind::Elementwise, true,
                                      K_q15_half, 1ull, elements, 1u,
                                      elements, 0)
                 == Status::OK);
        uint32_t saturated = 0u;
        CHECK (dropout_route_grad_q15 (config, nullptr, mask, grad_in,
                                       saturated)
               == Status::BAD_LAYOUT);
        CHECK (dropout_route_grad_q15 (config, grad_out, nullptr, grad_in,
                                       saturated)
               == Status::BAD_LAYOUT);
        CHECK (dropout_route_grad_q15 (config, grad_out, mask, nullptr,
                                       saturated)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("the deploy-domain dropout code and aux parameter are exact")
{
    SECTION ("dropout_aux_code rounds p onto the 1/127 code grid")
    {
        const int32_t drops[] = { 0, 1, 4096, 8000, 16384, 24576, 32767 };
        for (const int32_t drop : drops)
            CHECK (dropout_aux_code (drop) == ref_aux_code (drop));
        CHECK (dropout_aux_code (0) == 0);
        CHECK (dropout_aux_code (_S_q15_max) == 127);
    }
    SECTION ("dropout_inference_code reproduces value * 127 / keep")
    {
        const int32_t codes[] = { -127, -100, -64, -1, 0, 1, 64, 100, 114,
                                  127 };
        const int32_t drop_codes[] = { 0, 1, 32, 64, 100, 126, 127, 200 };
        for (const int32_t code : codes)
            for (const int32_t drop : drop_codes)
                CHECK (dropout_inference_code (code, drop)
                       == ref_inference_code (code, drop));
        CHECK (dropout_inference_code (50, 0) == 50);
        CHECK (dropout_inference_code (50, -3) == 50);
        CHECK (dropout_inference_code (50, 127) == 0);
        CHECK (dropout_inference_code (127, 64) == 127);
        CHECK (dropout_inference_code (-127, 64) == -127);
    }
    SECTION ("dropout_inference_forward replays the xorshift32 stream")
    {
        const uint32_t count = 128u;
        int8_t in[128];
        int8_t out[128];
        for (uint32_t i = 0u; i < count; ++i)
            in[i] = static_cast<int8_t> ((i * 51u) % 255u) - 127;

        for (const int32_t drop : { 1, 32, 64, 96, 126 })
            {
                const uint32_t seed = 0x9ACD0u + drop;
                REQUIRE (dropout_inference_forward (drop, seed, in, out,
                                                   count)
                         == Status::OK);
                uint32_t state = seed | 1u;
                for (uint32_t i = 0u; i < count; ++i)
                    {
                        const uint32_t field = ref_inference_field (state);
                        const bool dropped = static_cast<int32_t> (field) < drop;
                        const int8_t want = dropped
                                                ? 0
                                                : ref_inference_code (in[i],
                                                                      drop);
                        CHECK (out[i] == want);
                    }
            }
        /* drop 0 is the identity, drop 127 zeros everything. */
        REQUIRE (dropout_inference_forward (0, 0x1234u, in, out, count)
                 == Status::OK);
        for (uint32_t i = 0u; i < count; ++i)
            CHECK (out[i] == in[i]);
        REQUIRE (dropout_inference_forward (127, 0x1234u, in, out, count)
                 == Status::OK);
        for (uint32_t i = 0u; i < count; ++i)
            CHECK (out[i] == 0);
        /* A zero seed still enters the stream via the | 1 forcing. */
        int8_t zero_seed_out[8];
        REQUIRE (dropout_inference_forward (64, 0u, in, zero_seed_out, 8u)
                 == Status::OK);
        uint32_t state = 1u;
        for (uint32_t i = 0u; i < 8u; ++i)
            {
                const uint32_t field = ref_inference_field (state);
                const bool dropped = static_cast<int32_t> (field) < 64;
                CHECK (zero_seed_out[i]
                       == (dropped ? 0 : ref_inference_code (in[i], 64)));
            }
        /* Bounds and null buffers. */
        CHECK (dropout_inference_forward (64, 1u, nullptr, out, count)
               == Status::BAD_LAYOUT);
        CHECK (dropout_inference_forward (64, 1u, in, nullptr, count)
               == Status::BAD_LAYOUT);
        CHECK (dropout_inference_forward (-1, 1u, in, out, count)
               == Status::OUT_OF_RANGE);
        CHECK (dropout_inference_forward (128, 1u, in, out, count)
               == Status::OUT_OF_RANGE);
        CHECK (dropout_inference_forward (64, 1u, in, out, 65536u)
               == Status::OUT_OF_RANGE);
        CHECK (dropout_inference_forward (64, 1u, in, out, 0u)
               == Status::OK);
    }
}
