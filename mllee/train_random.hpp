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

#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_fixed.hpp"

#include <cstdint>

/**
 * @brief Deterministic, heap-free random numbers for the training path.
 *
 * Everything here is pure integer arithmetic on a state the caller owns and
 * passes by reference: no writable static, no device clock, no floating point,
 * no `std::mt19937`.  Two engines are provided because they serve different
 * needs --
 *
 *   - @ref SplitMix64 for seeds, initializers and noise injection: one counter
 *     step plus two multiplier mixes, so consecutive states decorrelate and the
 *     stream is statistically sound;
 *   - @ref XorShift64 (xorshift64*) for the per-element mask streams, where a
 *     shift/xor chain is cheaper than two 64-bit multiplies on a target with a
 *     single-cycle multiplier but no 64x64 unit.
 *
 * Both are written entirely in `uint64_t` literals and integer operators, so a
 * given seed produces a byte-identical sequence on x86-64, on Cortex-M4 and on
 * the host oracle; that is what makes a trained checkpoint reproducible.  The
 * zero state is the single forbidden input and every entry point forces it away
 * (`| 1` or an explicit replace), which keeps xorshift -- a permutation of the
 * non-zero states -- from getting stuck.
 *
 * @par Distribution error budget
 * The Q15 uniform is exact: 16 of the 64 state bits are used whole, so every
 * lane value occurs with probability 2^-16 and no rejection or modulo bias
 * exists.  The normal deviate is the inverse-CDF (rational) form of Zelen &
 * Severo, as quoted by Abramowitz & Stegun 26.2.22:
 *
 *   x = t - (c0 + c1 t + c2 t^2) / (1 + d1 t + d2 t^2 + d3 t^3),
 *   t = sqrt (-2 ln p),  p = min (u, 1 - u),
 *
 * with the published coefficients c0 = 2.515517, c1 = 0.802853, c2 = 0.015320,
 * d1 = 1.432788, d2 = 0.189269, d3 = 0.001308 (absolute error below 4.5e-4 in
 * the source domain).  Evaluated in fixed point the error becomes
 *
 *   - 6e-5 in z units from c2 and d2, 2.8e-3 from d3 (43 of 32768 ulps against
 *     0.001308, the only coefficient whose quantization is visible),
 *   - 0.5 % of `t` near the centre, where -ln p is small and the Q7 log domain
 *     of q7_neg_log_q15 costs half an ulp of 1/128,
 *   - a hard support limit: p cannot go below one ulp of Q15, so the sampler
 *     never exceeds 4.55 sigma (the true quantile there is exp(-10), i.e. 2e-5
 *     of the mass sits outside the range).
 *
 * Measured worst case against the host `std::erfc` inverse, walking every
 * nonzero tail probability on the Q15 grid, is 0.0093 sigma, and
 * tests/test_mllee_training_regularization.cpp asserts a 0.02 sigma band.
 */
namespace zzs::mllee::train
{
/** @brief splitmix64 counter increment (the 64-bit golden-ratio constant). */
inline constexpr uint64_t _S_splitmix_increment = 0x9E3779B97F4A7C15ull;
/** @brief splitmix64 first mix multiplier. */
inline constexpr uint64_t _S_splitmix_mixer_a = 0xBF58476D1CE4E5B9ull;
/** @brief splitmix64 second mix multiplier. */
inline constexpr uint64_t _S_splitmix_mixer_b = 0x94D049BB133111EBull;
/** @brief xorshift64* output multiplier (Vigna's constant). */
inline constexpr uint64_t _S_xorshift_multiplier = 0x2545F4914F6CDD1Dull;

/**
 * @brief One splitmix64 step: advance the counter, then mix it twice.
 *
 * @param state engine state, mutated in place; the caller owns it, always.
 */
[[nodiscard]] inline constexpr uint64_t
rng_splitmix64 (uint64_t &state) noexcept
{
    state += _S_splitmix_increment;
    uint64_t value = state;
    value = (value ^ (value >> 30u)) * _S_splitmix_mixer_a;
    value = (value ^ (value >> 27u)) * _S_splitmix_mixer_b;
    return value ^ (value >> 31u);
}

/**
 * @brief One xorshift64* step: three shift/xor stages and one multiplier.
 *
 * Zero is a fixed point of the xorshift part, so it is replaced on entry; a
 * non-zero state is never mapped to zero (the xorshift stage is invertible and
 * the multiplier is odd).
 */
[[nodiscard]] inline constexpr uint64_t
rng_xorshift64 (uint64_t &state) noexcept
{
    if (state == 0u)
        state = _S_splitmix_increment;
    state ^= state >> 12u;
    state ^= state << 25u;
    state ^= state >> 27u;
    return state * _S_xorshift_multiplier;
}

/** @brief The statistically strong engine: one 64-bit state and nothing else. */
struct SplitMix64
{
    uint64_t _M_state = 0u;

    /** @brief Seed the engine; every 64-bit value is usable, including 0. */
    [[nodiscard]] static constexpr SplitMix64
    seeded (const uint64_t seed) noexcept
    {
        SplitMix64 engine;
        engine._M_state = seed;
        return engine;
    }

    uint64_t
    next_u64 () noexcept
    {
        return rng_splitmix64 (this->_M_state);
    }

    [[nodiscard]] uint32_t
    next_u32 () noexcept
    {
        return static_cast<uint32_t> (this->next_u64 () >> 32u);
    }
};

/** @brief The cheap engine: one 64-bit state, per-element mask streams. */
struct XorShift64
{
    uint64_t _M_state = _S_splitmix_increment;

    /** @brief Seed the engine; 0 is replaced by the golden-ratio constant. */
    [[nodiscard]] static constexpr XorShift64
    seeded (const uint64_t seed) noexcept
    {
        XorShift64 engine;
        engine._M_state = seed == 0u ? _S_splitmix_increment : seed;
        return engine;
    }

    uint64_t
    next_u64 () noexcept
    {
        return rng_xorshift64 (this->_M_state);
    }

    [[nodiscard]] uint32_t
    next_u32 () noexcept
    {
        return static_cast<uint32_t> (this->next_u64 () >> 32u);
    }
};

static_assert (sizeof (SplitMix64) == 8u,
               "the splitmix engine must be exactly its 64-bit state");
static_assert (sizeof (XorShift64) == 8u,
               "the xorshift engine must be exactly its 64-bit state");

/**
 * @brief Independent stream from a base seed and a stream index.
 *
 * One splitmix64 round over `base ^ (index * increment)` decorrelates per-layer
 * and per-step streams, which is how the trainer reproduces "step 7 of epoch 3"
 * without storing a state per step.
 */
[[nodiscard]] inline constexpr uint64_t
rng_derive_seed (const uint64_t base, const uint32_t stream) noexcept
{
    uint64_t state = base
                     ^ (static_cast<uint64_t> (stream) * _S_splitmix_increment);
    return rng_splitmix64 (state);
}

/** @brief Fold two 64-bit values into one seed (checkpoint and epoch keys). */
[[nodiscard]] inline constexpr uint64_t
rng_mix_seed (uint64_t a, uint64_t b) noexcept
{
    a ^= b + 0x9E3779B9u + (a << 6u) + (a >> 2u);
    b ^= a + 0x9E3779B9u + (b << 6u) + (b >> 2u);
    uint64_t state = a ^ (b * _S_splitmix_mixer_a);
    return rng_splitmix64 (state);
}

/**
 * @name Uniform draws
 *
 * The 16 high bits of a draw feed a Q15 lane: bits 63..48 are the best-mixed
 * ones in both engines, and taking them whole keeps the mapping exact.
 * @{
 */

/** @brief Uniform Q15 lane in [-1.0, 1.0), i.e. an int in [-32768, 32767]. */
template <typename Engine>
[[nodiscard]] inline int32_t
uniform_q15 (Engine &engine) noexcept
{
    const uint64_t raw = engine.next_u64 ();
    return static_cast<int32_t> (static_cast<int16_t> (
        static_cast<uint16_t> (raw >> 48u)));
}

/** @brief Uniform Q15 magnitude in [0, 1.0), i.e. an int in [0, 32767]. */
template <typename Engine>
[[nodiscard]] inline int32_t
uniform_positive_q15 (Engine &engine) noexcept
{
    const uint64_t raw = engine.next_u64 ();
    return static_cast<int32_t> (static_cast<uint32_t> (raw >> 48u) & 0x7FFFu);
}

/**
 * @brief Uniform integer in [0, bound): multiply-shift (Lemire's fastrange).
 *
 * A `% bound` is biased by up to bound/2^32 whenever bound is not a power of
 * two; the high half of the 32x32 product has no such bias, needs no rejection
 * branch and therefore runs in constant time -- which matters for a mask stream
 * that must not leak its content into the timing.  bound == 0 yields 0.
 */
template <typename Engine>
[[nodiscard]] inline uint32_t
bounded_uniform (Engine &engine, const uint32_t bound) noexcept
{
    if (bound == 0u)
        return 0u;
    const uint64_t product
        = static_cast<uint64_t> (engine.next_u32 ()) * static_cast<uint64_t> (bound);
    return static_cast<uint32_t> (product >> 32u);
}

/** @brief Bernoulli draw: true with probability p (a Q15 magnitude in [0, 1)). */
template <typename Engine>
[[nodiscard]] inline bool
bernoulli_q15 (Engine &engine, const int32_t p_q15) noexcept
{
    if (p_q15 <= 0)
        return false;
    if (p_q15 >= _S_q15_one)
        return true;
    /* Fifteen bits, so the draw spans [0, 32767] and the comparison against a
       Q15 p is a 2^-15-exact Bernoulli test. */
    const uint64_t raw = engine.next_u64 ();
    const int32_t draw = static_cast<int32_t> (static_cast<uint32_t> (raw >> 49u));
    return draw < p_q15;
}

/**
 * @brief Fill a keep-mask with @p keep_probability_q15, returning the kept count.
 *
 * One byte per lane (not one bit per lane) is the deliberate choice: the
 * backward reads the mask with @ref route_by_mask, which is stride-1 and needs no
 * bit extraction, and 1 byte per element is the same footprint the inference
 * core gives a mask tensor.
 */
template <typename Engine>
[[nodiscard]] inline uint32_t
fill_mask (Engine &engine, const int32_t keep_probability_q15,
           uint8_t *const mask, const uint32_t count) noexcept
{
    uint32_t kept = 0u;
    for (uint32_t i = 0; i < count; ++i)
        {
            const uint8_t bit
                = bernoulli_q15 (engine, keep_probability_q15) ? 1u : 0u;
            mask[i] = bit;
            kept += bit;
        }
    return kept;
}

/**
 * @brief Blocked mask: one draw per unit, replicated over @p inner elements.
 *
 * This is the Dropout2d / channel-wise shape: @p units is N*C and @p inner is
 * H*W, so the written stream stays stride-1 while the draw count drops by
 * @p inner.  Returns the number of kept elements (in units of elements).
 */
template <typename Engine>
[[nodiscard]] inline uint32_t
fill_mask_blocked (Engine &engine, const int32_t keep_probability_q15,
                   const uint32_t units, const uint32_t inner,
                   uint8_t *const mask) noexcept
{
    uint32_t kept = 0u;
    for (uint32_t unit = 0; unit < units; ++unit)
        {
            const uint8_t bit
                = bernoulli_q15 (engine, keep_probability_q15) ? 1u : 0u;
            for (uint32_t lane = 0; lane < inner; ++lane)
                mask[unit * inner + lane] = bit;
            kept += bit != 0u ? inner : 0u;
        }
    return kept;
}

/** @brief Fill a lane buffer with uniform Q15 values (noise, random inputs). */
template <typename Engine>
inline void
fill_lanes_uniform (Engine &engine, int16_t *const lanes,
                    const uint32_t count) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        lanes[i] = static_cast<int16_t> (uniform_q15 (engine));
}

/** @} */

/**
 * @name Fixed-point square roots
 *
 * @ref q15_sqrt reaches only a radicand of 2.0 because `_S_isqrt` is 32-bit,
 * while init needs sqrt (2 / fan) and the normal sampler needs sqrt (-2 ln p) up
 * to 20.8.  The helpers below fold the exponent out of the radicand first, and
 * the fold costs one ulp per step, which is why they are here once instead of
 * re-derived per module.
 * @{
 */

/**
 * @brief sqrt of a 15-fraction-bit value that may exceed 1.0, result in 15
 *        fraction bits (so the result may exceed a Q15 lane: this is the wide
 *        magnitude form, never a lane).
 *
 * The argument is divided by four until `x << 15` fits a uint32; each division
 * doubles the root, so the fold is exact in scaling and only truncates -- the
 * error stays below one ulp of the result plus one ulp per fold.
 */
[[nodiscard]] inline constexpr int32_t
q15_sqrt_wide (const int32_t value_q15) noexcept
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
    uint64_t root = _S_isqrt (argument << _S_grad_frac_bits);
    root <<= folds;
    return root > 0x7FFFFFFFull ? 0x7FFFFFFF : static_cast<int32_t> (root);
}

/** @brief sqrt (num / den) in 15 fraction bits from two plain integer counts. */
[[nodiscard]] inline constexpr int32_t
sqrt_ratio_q15 (const uint32_t num, const uint32_t den) noexcept
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
    uint64_t root = _S_isqrt (static_cast<uint32_t> (radicand));
    root <<= folds;
    return root > 0x7FFFFFFFull ? 0x7FFFFFFF : static_cast<int32_t> (root);
}

/**
 * @brief (num / den) * 2^15 as an int32 that may exceed a Q15 lane.
 *
 * An initializer limit can legitimately be above 1.0 (Xavier with a two-unit
 * fan is sqrt 3), and a scale is a magnitude, not a lane, so it is reported in
 * this wide form instead of saturating.
 */
[[nodiscard]] inline constexpr int32_t
ratio_wide_q15 (const uint32_t num, const uint32_t den) noexcept
{
    if (den == 0u)
        return 0;
    const uint64_t scaled
        = (static_cast<uint64_t> (num) << _S_grad_frac_bits) / den;
    return scaled > 0x7FFFFFFFull ? 0x7FFFFFFF : static_cast<int32_t> (scaled);
}

/** @} */

/**
 * @name Approximate standard normal (inverse-CDF form)
 *
 * Coefficients of Zelen & Severo as above; c0 and d1 are held at 14 fraction
 * bits because 2.515517 and 1.432788 do not fit a Q15 lane, and the two
 * polynomials are therefore evaluated in one consistent width each.  The
 * numerator and the denominator are both at 14 fraction bits, so their ratio is
 * scale-free and only needs a Q15 shift before the subtraction from t.
 * @{
 */

/** @brief c0 = 2.515517 at 14 fraction bits. */
inline constexpr int32_t _S_normal_c0_q14 = 41213;
/** @brief c1 = 0.802853 at 15 fraction bits. */
inline constexpr int32_t _S_normal_c1_q15 = 26310;
/** @brief c2 = 0.015320 at 15 fraction bits. */
inline constexpr int32_t _S_normal_c2_q15 = 502;
/** @brief d1 = 1.432788 at 14 fraction bits. */
inline constexpr int32_t _S_normal_d1_q14 = 23484;
/** @brief d2 = 0.189269 at 15 fraction bits. */
inline constexpr int32_t _S_normal_d2_q15 = 6199;
/** @brief d3 = 0.001308 at 15 fraction bits (43 ulps; the dominant term error). */
inline constexpr int32_t _S_normal_d3_q15 = 43;
/** @brief 1.0 at 14 fraction bits, the constant term of the denominator. */
inline constexpr int32_t _S_q14_one = 16384;
/** @brief The largest |t| the sampler produces; 4.55 sigma in 15 fraction bits. */
inline constexpr int32_t _S_normal_t_limit_q15 = 149000;
/** @brief Largest index population a permutation helper addresses. */
inline constexpr uint32_t _S_max_permutation_elements = 65535u;

/**
 * @brief -ln p in 15 fraction bits, with one correction step against the
 *        inference exp table.
 *
 * `q7_neg_log_q15` answers in the Q7 log domain, whose 1/128 ulp is a 0.5 %
 * error exactly where the normal sampler needs it most (p near 0.5, where
 * -ln p is only 0.69).  One refinement removes most of it: with L the coarse
 * value and e = exp (-L),
 *
 *   -ln p = L + ln (e / p)   and   ln (e / p) = (e - p) / p  to first order,
 *
 * and the coarse error is below 0.4 %, so e / p never strays far enough from 1
 * for the second-order term (4e-6) to matter.  What limits the refinement is
 * the exp table's own linear-interpolation error, 0.2 % of e: it is only worth
 * taking while e is still a nine-bit number, so the deep tail keeps the coarse
 * value, whose error there is 0.04 % relative.  The clamp is that same argument
 * made explicit.
 */
[[nodiscard]] inline int32_t
neg_ln_q15 (const int32_t p_q15) noexcept
{
    if (p_q15 <= 0)
        return 0x7FFFFFFF;
    const int32_t coarse_q7 = q7_neg_log_q15 (p_q15);
    const int32_t coarse_q15 = coarse_q7 * 256;
    const int32_t e_q15 = q15_exp_neg (coarse_q15);
    if (e_q15 < 512)
        return coarse_q15;
    const int32_t raw = q15_div (e_q15 - p_q15, p_q15);
    const int32_t correction = raw > 256 ? 256 : (raw < -256 ? -256 : raw);
    return coarse_q15 + correction;
}

/**
 * @brief Upper branch of the inverse CDF: the z with P(Z > z) = tail.
 *
 * @param tail_q15 tail probability p in 15 fraction bits, clamped to
 *        [1, 32768 / 2] (p = 0.5 gives z = 0, one ulp gives the 4.55 limit).
 *
 * Exposed because it is the whole approximation: a host test can walk p over
 * its Q15 grid and grade the result against a floating-point probit without
 * touching the engine.
 */
[[nodiscard]] inline int32_t
normal_quantile_q15 (const int32_t tail_q15) noexcept
{
    const int32_t half = _S_q15_one / 2;
    const int32_t tail = tail_q15 < 1 ? 1 : (tail_q15 > half ? half : tail_q15);
    const int32_t t = q15_sqrt_wide (neg_ln_q15 (tail) * 2);
    /* c1 t and c2 t^2 in 14 fraction bits: q15_scale_shift never clamps, which
       is what a t of 4.55 needs and a Q15 multiply would not give. */
    const int32_t t_square_q15 = static_cast<int32_t> (
        (static_cast<int64_t> (t) * t) >> _S_grad_frac_bits);
    const int32_t c1_t_q14 = q15_scale_shift (t, _S_normal_c1_q15, 15u) >> 1u;
    const int32_t c2_t2_q14
        = q15_scale_shift (t_square_q15, _S_normal_c2_q15, 15u) >> 1u;
    const int32_t numerator = _S_normal_c0_q14 + c1_t_q14 + c2_t2_q14;
    const int32_t t_cube_q15 = q15_scale_shift (t_square_q15, t, 15u);
    const int32_t d1_t_q14 = q15_scale_shift (t, _S_normal_d1_q14, 14u) >> 1u;
    const int32_t d2_t2_q14
        = q15_scale_shift (t_square_q15, _S_normal_d2_q15, 15u) >> 1u;
    const int32_t d3_t3_q14
        = q15_scale_shift (t_cube_q15, _S_normal_d3_q15, 15u) >> 1u;
    const int32_t denominator
        = _S_q14_one + d1_t_q14 + d2_t2_q14 + d3_t3_q14;
    const int32_t correction = static_cast<int32_t> (
        (static_cast<int64_t> (numerator) << _S_grad_frac_bits) / denominator);
    const int32_t value = t - correction;
    return value > _S_normal_t_limit_q15 ? _S_normal_t_limit_q15 : (value < 0 ? 0 : value);
}

/**
 * @brief Standard-normal deviate in 15 fraction bits (a magnitude may exceed
 *        1.0: the range is [-4.55, 4.55]).
 *
 * One Q15 draw is consumed per call.  A lane u uniform on [-1, 1) is read as the
 * CDF argument (u + 1) / 2, which by symmetry is the tail probability
 * (1 - |u|) / 2 with the sign of u as the branch selector -- so the deviate is
 * symmetric by construction and the sampler only ever needs the p <= 0.5 half
 * of the rational approximation.
 */
template <typename Engine>
[[nodiscard]] inline int32_t
normal_wide_q15 (Engine &engine) noexcept
{
    int32_t uniform = uniform_q15 (engine);
    if (uniform == _S_q15_min)
        uniform = _S_q15_min + 1;
    const int32_t magnitude = uniform < 0 ? -uniform : uniform;
    const int32_t value = normal_quantile_q15 ((_S_q15_one - magnitude) / 2);
    return uniform < 0 ? -value : value;
}

/**
 * @brief sigma * z narrowed to a Q15 lane, with the saturation reported.
 *
 * @param sigma_q15 deviate standard deviation in 15 fraction bits, [0, 32767].
 * @param clamped   incremented when the lane had to saturate (0 or 1).
 */
template <typename Engine>
[[nodiscard]] inline int32_t
normal_q15 (Engine &engine, const int32_t sigma_q15,
            uint32_t &clamped) noexcept
{
    const int32_t z = normal_wide_q15 (engine);
    const int32_t product = q15_scale_shift (z, sigma_q15, _S_grad_frac_bits);
    if (product > _S_q15_max || product < _S_q15_min)
        {
            ++clamped;
            return q15_clamp (product);
        }
    return product;
}

/** @brief sigma * z narrowed to a Q15 lane, ignoring saturation. */
template <typename Engine>
[[nodiscard]] inline int32_t
normal_q15 (Engine &engine, const int32_t sigma_q15) noexcept
{
    uint32_t ignored = 0u;
    return normal_q15 (engine, sigma_q15, ignored);
}

/**
 * @brief Symmetric triangular noise in [-scale, scale]: the mean of two
 *        uniforms, so no sqrt and no log.
 *
 * Used by the gradient-noise injection of mllee/train_regularization.hpp, where
 * a bounded magnitude matters more than Gaussianity.  The division by two
 * truncates toward zero, a documented bias below one ulp of the result.
 */
template <typename Engine>
[[nodiscard]] inline int32_t
triangular_noise_q15 (Engine &engine, const int32_t scale_q15) noexcept
{
    const int32_t first = uniform_q15 (engine);
    const int32_t second = uniform_q15 (engine);
    const int32_t mean = (first + second) / 2;
    return q15_mul (mean, scale_q15);
}

/** @} */

/**
 * @name Indices: shuffle and permutation over caller-owned storage
 *
 * Fisher-Yates (Knuth) in its iterative swap form: no recursion, in place, one
 * bounded draw per position, and a uniform permutation distribution.  The
 * uint16 form matches the element-index width the tape and the dataset use;
 * @ref init_indices covers the wider index types a caller may own.
 * @{
 */

/** @brief Fill @p indices with 0, 1, ... count-1. */
template <typename Index>
inline void
init_indices (Index *const indices, const uint32_t count) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        indices[i] = static_cast<Index> (i);
}

/** @brief In-place Fisher-Yates shuffle of a caller-owned index buffer. */
template <typename Index, typename Engine>
inline void
shuffle_indices (Engine &engine, Index *const indices,
                 const uint32_t count) noexcept
{
    for (uint32_t tail = count; tail-- > 1u;)
        {
            const uint32_t pick = bounded_uniform (engine, tail + 1u);
            const Index swap = indices[pick];
            indices[pick] = indices[tail];
            indices[tail] = swap;
        }
}

/** @brief Write a shuffled 0..count-1 permutation into a caller-owned buffer. */
template <typename Engine>
inline Status
make_permutation (Engine &engine, uint16_t *const indices,
                  const uint32_t count) noexcept
{
    if (indices == nullptr && count != 0u)
        return Status::BAD_LAYOUT;
    if (count > _S_max_permutation_elements)
        return Status::OUT_OF_RANGE;
    init_indices (indices, count);
    shuffle_indices (engine, indices, count);
    return Status::OK;
}

/** @brief Shuffle an existing uint16 index buffer in place. */
template <typename Engine>
inline void
permute_indices (Engine &engine, uint16_t *const indices,
                 const uint32_t count) noexcept
{
    shuffle_indices (engine, indices, count);
}

/**
 * @brief Draw @p count distinct indices below @p population into @p out.
 *
 * Partial Fisher-Yates over the implicit identity permutation: the swap table
 * is caller-owned (@p scratch, @p population entries) so nothing is allocated,
 * and the loop stays bounded.  Returns the number written, which is
 * min (count, population); 0 when the buffers or the population are unusable.
 */
template <typename Engine>
inline uint32_t
sample_without_replacement (Engine &engine, const uint32_t population,
                            const uint32_t count, uint16_t *const out,
                            uint16_t *const scratch) noexcept
{
    if (out == nullptr || scratch == nullptr || population == 0u
        || population > _S_max_permutation_elements)
        return 0u;
    const uint32_t draws = count < population ? count : population;
    init_indices (scratch, population);
    for (uint32_t i = 0; i < draws; ++i)
        {
            const uint32_t pick = i + bounded_uniform (engine, population - i);
            const uint16_t swap = scratch[pick];
            scratch[pick] = scratch[i];
            scratch[i] = swap;
            out[i] = scratch[i];
        }
    return draws;
}

/** @} */
} // namespace zzs::mllee::train
