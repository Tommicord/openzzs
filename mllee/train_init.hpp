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
#include "mllee/train_random.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief int8-code weight initializers that report their requantization scale.
 *
 * An initializer here does not produce real-valued weights: it produces int8
 * codes plus the one number that makes those codes meaningful, the
 * @c _M_scale_q15 of the tensor (the real magnitude carried by code 127, in 15
 * fraction bits).  Every entry point therefore returns a @ref InitResult, and
 * feeding it to @ref init_result_apply is what puts a layer into the same
 * value model the inference core reads.  A scale is a magnitude, not a lane, so
 * it is kept in the wide 15-fraction-bit form of mllee/train_random.hpp and may
 * legitimately exceed 1.0 (a one-input fan gives sqrt 6 = 2.449).
 *
 * @par The two sampling domains
 * Codes are drawn in the code domain and the scale is chosen so that the
 * intended real distribution maps onto them exactly:
 *
 *   - uniform with half range L: codes are drawn uniformly over [-127, 127] and
 *     the scale is L, so the real weights are uniform over [-L, L] up to the
 *     half-bin width at the two ends (each interior code carries probability
 *     1/255, the end codes 1/510);
 *   - normal with standard deviation sigma: the scale is 3 sigma, i.e. the code
 *     grid covers +/-3 sigma, and a deviate beyond that (0.27 % of the mass)
 *     saturates onto +/-127 and is counted in @c _M_clamped.  Three sigma is
 *     the usual QAT truncation: it keeps the grid 1.5x finer than the +/-4.55
 *     hard support of @ref normal_wide_q15 would require.
 *
 * @par Orthogonality without a QR
 * @ref orthogonal_init builds a semi-orthogonal matrix as (random signed
 * permutation) followed by a few random Householder reflections, Q <- (I -
 * 2uu^T/u^Tu) Q.  A reflection is exactly orthogonal for every non-zero u -- no
 * Gram-Schmidt, no Givens angles, no cosine table -- and only needs a dot
 * product and one division per column, so the whole construction is integer
 * arithmetic in a caller-owned Q15 scratch of rows*cols lanes.  Two documented
 * approximations remain: the int8 narrowing perturbs orthogonality by about one
 * code step per entry, and the reachable set is (2^k * k! * reflection count)
 * rather than the Haar measure of the full orthogonal group.  @ref
 * orthogonal_defect measures both in code units so a caller can assert them.
 *
 * @par Memory
 * Codes: @c count bytes, written stride-1 (4-byte aligned is preferred, never
 * required, because no SIMD kernel runs here).  Orthogonal: 2 * rows * cols
 * bytes of Q15 scratch plus a 64-byte used-slot bitmap and a 32-lane reflection
 * vector on the caller's stack.
 */
namespace zzs::mllee::train
{
/** @brief Which distribution an initializer produced. */
enum class InitKind : uint8_t
{
    Uniform = 0u,
    Normal = 1u,
    XavierUniform = 2u,
    XavierNormal = 3u,
    KaimingUniform = 4u,
    KaimingNormal = 5u,
    HeUniform = 6u,
    HeNormal = 7u,
    FanInUniform = 8u,
    Orthogonal = 9u,
    NormalClamped = 10u,
    Constant = 11u,
};

/**
 * @brief What an initializer wrote and how to requantize it.
 *
 * @c _M_scale_q15 is the code-127 magnitude (wide Q15), @c _M_sigma_q15 the
 * intended standard deviation of the real weights in the same form, @c
 * _M_clamped the number of lanes that saturated into the code range.
 */
struct InitResult
{
    int32_t _M_scale_q15 = _S_q15_one;
    int32_t _M_sigma_q15 = 0;
    uint32_t _M_clamped = 0u;
    uint32_t _M_elements = 0u;
};

static_assert (sizeof (InitResult) == 16u,
               "an initializer result must stay four words for pool carving");

/** @brief The fan pair every named initializer is parametrized by. */
struct InitFans
{
    uint32_t _M_fan_in = 0u;
    uint32_t _M_fan_out = 0u;
};

static_assert (sizeof (InitFans) == 8u, "the fan pair is two plain counts");

/** @brief Normal truncation budget: 3.0 at 14 fraction bits. */
inline constexpr int32_t _S_normal_truncation_q14 = 98304;

/** @brief 1/sqrt(3) in Q15: the uniform half range to standard deviation. */
inline constexpr int32_t _S_uniform_sigma_q15 = 18919;

/** @brief Gain sqrt(2) (He/Kaiming for a rectifier) in 15 fraction bits. */
inline constexpr int32_t _S_he_gain_wide_q15 = 46341;

/** @brief Largest side @ref orthogonal_init reflects over. */
inline constexpr uint32_t _S_max_orthogonal_short_side = 32u;

/** @brief Largest side @ref orthogonal_init permutes over (512-bit bitmap). */
inline constexpr uint32_t _S_max_orthogonal_long_side = 512u;

/** @brief Householder reflections applied after the signed permutation. */
inline constexpr uint32_t _S_orthogonal_reflections = 6u;

/**
 * @brief Fan pair of a weight tensor, PyTorch's convention.
 *
 * Rank 2 [out, in] gives (in, out); rank >= 3 [out, in, k...] gives
 * (in * receptive, out * receptive) with the receptive field the trailing axes.
 */
[[nodiscard]] inline constexpr Status
fans_for_shape (const Shape &shape, InitFans &out) noexcept
{
    if (shape._M_rank < 2u || shape._M_rank > _S_max_rank)
        return Status::BAD_LAYOUT;
    uint32_t receptive = 1u;
    for (uint8_t axis = 2u; axis < shape._M_rank; ++axis)
        receptive *= shape._M_dims[axis];
    out._M_fan_in = static_cast<uint32_t> (shape._M_dims[1]) * receptive;
    out._M_fan_out = static_cast<uint32_t> (shape._M_dims[0]) * receptive;
    if (out._M_fan_in == 0u || out._M_fan_out == 0u)
        return Status::BAD_LAYOUT;
    return Status::OK;
}

/** @brief Publish an initializer's scale onto the tensor it filled. */
[[nodiscard]] inline Status
init_result_apply (const InitResult &result, Trainable &tensor) noexcept
{
    if (result._M_elements != tensor._M_count)
        return Status::BAD_LAYOUT;
    if (result._M_scale_q15 <= 0)
        return Status::OUT_OF_RANGE;
    tensor._M_scale_q15 = result._M_scale_q15;
    return Status::OK;
}

/**
 * @brief Round a wide-Q15 value onto the code grid whose code-127 is @p scale.
 *
 * Both operands are 15-fraction-bit magnitudes, so the quotient is the code
 * itself; the rounding is half away from zero and the result is left unclamped
 * so the caller can count the lanes that fell outside [-127, 127].
 */
[[nodiscard]] inline constexpr int32_t
wide_to_code (const int32_t value_wide_q15, const int32_t scale_wide_q15) noexcept
{
    if (scale_wide_q15 <= 0)
        return 0;
    const int64_t numerator = static_cast<int64_t> (value_wide_q15) * 127;
    const int64_t half = scale_wide_q15 / 2;
    const int64_t rounded = numerator >= 0 ? numerator + half : numerator - half;
    return static_cast<int32_t> (rounded / scale_wide_q15);
}

/**
 * @brief Uniform codes over the whole code range, half range @p limit_wide_q15.
 *
 * One Lemire draw per lane, so no modulo bias and no rejection branch; the
 * reported scale is the limit itself, which makes the real weights uniform on
 * [-limit, limit].
 */
template <typename Engine>
[[nodiscard]] inline Status
init_uniform (Engine &engine, const int32_t limit_wide_q15,
              int8_t *const codes, const uint32_t count,
              InitResult &out) noexcept
{
    if (codes == nullptr || count == 0u)
        return Status::BAD_LAYOUT;
    if (limit_wide_q15 <= 0 || count > _S_max_elements)
        return Status::OUT_OF_RANGE;
    for (uint32_t i = 0; i < count; ++i)
        codes[i] = static_cast<int8_t> (
            bounded_uniform (engine, 255u) - 127u);
    out._M_scale_q15 = limit_wide_q15;
    out._M_sigma_q15 = q15_scale_shift (limit_wide_q15, _S_uniform_sigma_q15,
                                        _S_grad_frac_bits);
    out._M_clamped = 0u;
    out._M_elements = count;
    return Status::OK;
}

/**
 * @brief Normal codes on a +/-3 sigma grid, deviates beyond it saturate.
 *
 * @param sigma_wide_q15 standard deviation in 15 fraction bits, may exceed 1.0.
 */
template <typename Engine>
[[nodiscard]] inline Status
init_normal (Engine &engine, const int32_t sigma_wide_q15,
             int8_t *const codes, const uint32_t count,
             InitResult &out) noexcept
{
    if (codes == nullptr || count == 0u)
        return Status::BAD_LAYOUT;
    if (sigma_wide_q15 <= 0 || count > _S_max_elements)
        return Status::OUT_OF_RANGE;
    const int32_t scale = q15_scale_shift (sigma_wide_q15,
                                           _S_normal_truncation_q14, 14u);
    uint32_t clamped = 0u;
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t deviate = normal_wide_q15 (engine);
            const int32_t value = q15_scale_shift (deviate, sigma_wide_q15,
                                                   _S_grad_frac_bits);
            const int32_t code = wide_to_code (value, scale);
            if (code > 127 || code < -127)
                {
                    ++clamped;
                    codes[i] = _S_i8_code (code > 0 ? 127 : -127);
                    continue;
                }
            codes[i] = _S_i8_code (code);
        }
    out._M_scale_q15 = scale;
    out._M_sigma_q15 = sigma_wide_q15;
    out._M_clamped = clamped;
    out._M_elements = count;
    return Status::OK;
}

/**
 * @brief Normal codes on an explicit grid, with an optional non-zero mean.
 *
 * The clamping form the task calls for: @p limit_wide_q15 is the code-127
 * magnitude, @p mean_wide_q15 the real mean, and every deviate outside
 * [-limit + mean, limit + mean] saturates onto the end codes.
 */
template <typename Engine>
[[nodiscard]] inline Status
init_normal_clamped (Engine &engine, const int32_t mean_wide_q15,
                     const int32_t sigma_wide_q15,
                     const int32_t limit_wide_q15, int8_t *const codes,
                     const uint32_t count, InitResult &out) noexcept
{
    if (codes == nullptr || count == 0u)
        return Status::BAD_LAYOUT;
    if (sigma_wide_q15 <= 0 || limit_wide_q15 <= 0 || count > _S_max_elements)
        return Status::OUT_OF_RANGE;
    uint32_t clamped = 0u;
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t deviate = normal_wide_q15 (engine);
            int64_t value = q15_scale_shift (deviate, sigma_wide_q15,
                                             _S_grad_frac_bits);
            value += mean_wide_q15;
            const int32_t bounded = value > limit_wide_q15
                                        ? limit_wide_q15
                                        : (value < -limit_wide_q15 ? -limit_wide_q15
                                                                    : static_cast<int32_t> (value));
            if (bounded != static_cast<int32_t> (value))
                ++clamped;
            codes[i] = _S_i8_code (wide_to_code (bounded, limit_wide_q15));
        }
    out._M_scale_q15 = limit_wide_q15;
    out._M_sigma_q15 = sigma_wide_q15;
    out._M_clamped = clamped;
    out._M_elements = count;
    return Status::OK;
}

/** @brief Constant fill: the gamma (1.0) and beta (0.0) starting points. */
[[nodiscard]] inline Status
init_constant (const int32_t value_wide_q15, const int32_t scale_wide_q15,
               int8_t *const codes, const uint32_t count,
               InitResult &out) noexcept
{
    if (codes == nullptr || count == 0u)
        return Status::BAD_LAYOUT;
    if (scale_wide_q15 <= 0 || count > _S_max_elements)
        return Status::OUT_OF_RANGE;
    const int32_t code = wide_to_code (value_wide_q15, scale_wide_q15);
    for (uint32_t i = 0; i < count; ++i)
        codes[i] = _S_i8_code (code);
    out._M_scale_q15 = scale_wide_q15;
    out._M_sigma_q15 = 0;
    out._M_clamped = code > 127 || code < -127 ? count : 0u;
    out._M_elements = count;
    return Status::OK;
}

/** @brief Normalization gain: every lane code 127 on a unit scale, i.e. 1.0. */
[[nodiscard]] inline Status
init_unit_gain (int8_t *const codes, const uint32_t count,
                InitResult &out) noexcept
{
    return init_constant (_S_q15_one, _S_q15_one, codes, count, out);
}

/** @brief Normalization bias: every lane zero. */
[[nodiscard]] inline Status
init_zero_bias (int8_t *const codes, const uint32_t count,
                InitResult &out) noexcept
{
    return init_constant (0, _S_q15_one, codes, count, out);
}

/** @brief Glorot/Xavier uniform: half range sqrt (6 / (fan_in + fan_out)). */
template <typename Engine>
[[nodiscard]] inline Status
xavier_uniform (Engine &engine, const InitFans fans, int8_t *const codes,
                const uint32_t count, InitResult &out) noexcept
{
    const uint32_t sum = fans._M_fan_in + fans._M_fan_out;
    if (sum < 2u || sum > 2u * _S_max_elements)
        return Status::OUT_OF_RANGE;
    const Status status
        = init_uniform (engine, sqrt_ratio_q15 (6u, sum), codes, count, out);
    return status;
}

/** @brief Glorot/Xavier normal: sigma sqrt (2 / (fan_in + fan_out)). */
template <typename Engine>
[[nodiscard]] inline Status
xavier_normal (Engine &engine, const InitFans fans, int8_t *const codes,
               const uint32_t count, InitResult &out) noexcept
{
    const uint32_t sum = fans._M_fan_in + fans._M_fan_out;
    if (sum < 2u || sum > 2u * _S_max_elements)
        return Status::OUT_OF_RANGE;
    return init_normal (engine, sqrt_ratio_q15 (2u, sum), codes, count, out);
}

/**
 * @brief Kaiming uniform: half range gain * sqrt (3 / fan_in).
 *
 * This is PyTorch's kaiming_uniform_ bound exactly (gain 1 for a linear
 * activation, sqrt 2 for a rectifier -- pass @ref _S_he_gain_wide_q15).
 */
template <typename Engine>
[[nodiscard]] inline Status
kaiming_uniform (Engine &engine, const uint32_t fan_in,
                 const int32_t gain_wide_q15, int8_t *const codes,
                 const uint32_t count, InitResult &out) noexcept
{
    if (fan_in == 0u || fan_in > _S_max_elements)
        return Status::OUT_OF_RANGE;
    if (gain_wide_q15 <= 0)
        return Status::BAD_LAYOUT;
    const int32_t bound
        = q15_scale_shift (sqrt_ratio_q15 (3u, fan_in), gain_wide_q15,
                           _S_grad_frac_bits);
    const Status status = init_uniform (engine, bound, codes, count, out);
    return status;
}

/** @brief Kaiming normal: sigma gain / sqrt (fan_in), +/-3 sigma grid. */
template <typename Engine>
[[nodiscard]] inline Status
kaiming_normal (Engine &engine, const uint32_t fan_in,
                const int32_t gain_wide_q15, int8_t *const codes,
                const uint32_t count, InitResult &out) noexcept
{
    if (fan_in == 0u || fan_in > _S_max_elements)
        return Status::OUT_OF_RANGE;
    if (gain_wide_q15 <= 0)
        return Status::BAD_LAYOUT;
    const int32_t sigma = q15_scale_shift (sqrt_ratio_q15 (1u, fan_in),
                                           gain_wide_q15, _S_grad_frac_bits);
    return init_normal (engine, sigma, codes, count, out);
}

/** @brief He uniform: the Kaiming form at the rectifier gain sqrt 2. */
template <typename Engine>
[[nodiscard]] inline Status
he_uniform (Engine &engine, const uint32_t fan_in, int8_t *const codes,
            const uint32_t count, InitResult &out) noexcept
{
    return kaiming_uniform (engine, fan_in, _S_he_gain_wide_q15, codes, count,
                            out);
}

/** @brief He normal: the Kaiming form at the rectifier gain sqrt 2. */
template <typename Engine>
[[nodiscard]] inline Status
he_normal (Engine &engine, const uint32_t fan_in, int8_t *const codes,
           const uint32_t count, InitResult &out) noexcept
{
    return kaiming_normal (engine, fan_in, _S_he_gain_wide_q15, codes, count,
                           out);
}

/** @brief Legacy PyTorch Linear default: half range 1 / sqrt (fan_in). */
template <typename Engine>
[[nodiscard]] inline Status
fan_in_uniform (Engine &engine, const uint32_t fan_in, int8_t *const codes,
                const uint32_t count, InitResult &out) noexcept
{
    if (fan_in == 0u || fan_in > _S_max_elements)
        return Status::OUT_OF_RANGE;
    return init_uniform (engine, sqrt_ratio_q15 (1u, fan_in), codes, count,
                         out);
}

/** @brief True when the two sides are usable by @ref orthogonal_init. */
[[nodiscard]] inline constexpr Status
orthogonal_shape_ok (const uint32_t rows, const uint32_t cols) noexcept
{
    if (rows == 0u || cols == 0u)
        return Status::BAD_LAYOUT;
    const uint32_t short_side = rows < cols ? rows : cols;
    const uint32_t long_side = rows < cols ? cols : rows;
    if (short_side > _S_max_orthogonal_short_side)
        return Status::OUT_OF_RANGE;
    if (long_side > _S_max_orthogonal_long_side)
        return Status::OUT_OF_RANGE;
    if (rows * cols > _S_max_elements)
        return Status::OUT_OF_RANGE;
    return Status::OK;
}

/** @brief First free column at or after @p start, wrapping once; 0xFFFF none. */
[[nodiscard]] inline uint32_t
orthogonal_probe (const uint32_t long_side, uint8_t *const used,
                  const uint32_t start) noexcept
{
    for (uint32_t step = 0u; step < long_side; ++step)
        {
            const uint32_t candidate = (start + step) % long_side;
            const uint32_t byte = candidate >> 3u;
            const uint8_t bit = static_cast<uint8_t> (1u << (candidate & 7u));
            if ((used[byte] & bit) == 0u)
                {
                    used[byte] = static_cast<uint8_t> (used[byte] | bit);
                    return candidate;
                }
        }
    return 0xFFFFu;
}

/**
 * @brief Semi-orthogonal init: signed permutation plus Householder reflections.
 *
 * The result has orthonormal rows when @p rows <= @p cols (and orthonormal
 * columns otherwise, the two cases being transposes of each other), each entry
 * bounded by @p gain_wide_q15 and reported as the code-127 scale -- so the
 * narrowing needs no gain term at all, only the wider scale.
 *
 * @param scratch caller-owned Q15 lane buffer, rows * cols lanes, 2 bytes each.
 *
 * The reflection vector is drawn from @ref uniform_q15 and one component is
 * forced away from zero, because u = 0 is the single input that would make the
 * reflection undefined; the division by u^T u then needs no guard.
 */
template <typename Engine>
[[nodiscard]] inline Status
orthogonal_init (Engine &engine, const uint32_t rows, const uint32_t cols,
                 const int32_t gain_wide_q15, int16_t *const scratch,
                 int8_t *const codes, InitResult &out) noexcept
{
    const Status shape = orthogonal_shape_ok (rows, cols);
    if (!status_ok (shape))
        return shape;
    if (scratch == nullptr || codes == nullptr || gain_wide_q15 <= 0)
        return Status::BAD_LAYOUT;
    const bool wide = rows <= cols;
    const uint32_t short_side = wide ? rows : cols;
    const uint32_t long_side = wide ? cols : rows;
    const uint32_t count = rows * cols;

    for (uint32_t i = 0; i < count; ++i)
        scratch[i] = 0;
    uint8_t used[_S_max_orthogonal_long_side / 8u] = { 0u };

    /* Stage 1: short_side entries of magnitude one on distinct long-side
       positions, each with a random sign.  Row-major placement when the matrix
       is wide, column-major when it is tall, so the starting matrix already has
       orthonormal rows (wide) or columns (tall). */
    for (uint32_t index = 0u; index < short_side; ++index)
        {
            const uint32_t slot = orthogonal_probe (
                long_side, used, bounded_uniform (engine, long_side));
            if (slot == 0xFFFFu)
                return Status::INTERNAL; /* unreachable while index < long_side */
            const int32_t sign = bernoulli_q15 (engine, _S_q15_one / 2) ? 1
                                                                       : -1;
            const uint32_t position
                = wide ? index * cols + slot : slot * cols + index;
            scratch[position] = static_cast<int16_t> (sign * _S_q15_max);
        }

    /* Stage 2: reflect along the orthonormal axis, which preserves the property
       exactly in real arithmetic (HA has the same row Gram matrix as A). */
    uint32_t clamped = 0u;
    const uint32_t reflections
        = short_side < _S_orthogonal_reflections ? short_side
                                                 : _S_orthogonal_reflections;
    int16_t u[_S_max_orthogonal_short_side];
    for (uint32_t reflection = 0u; reflection < reflections; ++reflection)
        {
            int64_t norm = 0;
            for (uint32_t i = 0; i < short_side; ++i)
                {
                    const int32_t draw = uniform_q15 (engine);
                    u[i] = static_cast<int16_t> (draw);
                    norm += static_cast<int64_t> (draw) * draw;
                }
            if (norm < static_cast<int64_t> (_S_q15_one))
                {
                    /* A near-zero u would make 2uu^T/u^Tu numerically huge; the
                       deterministic fallback is a half-unit basis vector. */
                    u[0] = static_cast<int16_t> (_S_q15_one / 2);
                    norm = static_cast<int64_t> (_S_q15_one / 2)
                          * (_S_q15_one / 2);
                }
            for (uint32_t line = 0u; line < long_side; ++line)
                {
                    int64_t projection = 0;
                    for (uint32_t i = 0; i < short_side; ++i)
                        projection
                            += static_cast<int64_t> (u[i])
                               * scratch[wide ? i * cols + line
                                              : line * cols + i];
                    /* factor = 2 (u . a) / (u . u) in 15 fraction bits.  A
                       reflection's factor is a ratio of two Q30 quantities, so
                       it is a real gain that legitimately exceeds 1.0 (up to
                       2 sqrt 3 for a two-component u) and therefore lives in
                       the wide form, never in a lane. */
                    const int64_t scaled
                        = (projection << (_S_grad_frac_bits + 1u)) / norm;
                    if (scaled > 0x40000000ll || scaled < -0x40000000ll)
                        ++clamped;
                    const int32_t factor = static_cast<int32_t> (
                        scaled > 0x40000000ll
                            ? 0x40000000ll
                            : (scaled < -0x40000000ll ? -0x40000000ll : scaled));
                    for (uint32_t i = 0; i < short_side; ++i)
                        {
                            const uint32_t position
                                = wide ? i * cols + line : line * cols + i;
                            const int32_t subtract = wide
                                                         ? q15_scale_shift (
                                                               u[i], factor,
                                                               _S_grad_frac_bits)
                                                         : q15_scale_shift (
                                                               factor, u[i],
                                                               _S_grad_frac_bits);
                            const int32_t value
                                = scratch[position] - subtract;
                            if (value > _S_q15_max || value < _S_q15_min)
                                ++clamped;
                            scratch[position] = q15_to_i16_sat (value);
                        }
                }
        }

    /* Stage 3: narrow to codes.  Entries are bounded by one unit, and code 127
       is the reported gain, so the narrowing is the constant factor 127/32768. */
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t code = _S_round_shift (
                static_cast<int64_t> (scratch[i]) * 127, _S_grad_frac_bits);
            if (code > 127 || code < -127)
                ++clamped;
            codes[i] = _S_i8_code (code);
        }
    const int32_t spread = sqrt_ratio_q15 (short_side, count);
    out._M_scale_q15 = gain_wide_q15;
    out._M_sigma_q15
        = q15_scale_shift (spread, gain_wide_q15, _S_grad_frac_bits);
    out._M_clamped = clamped;
    out._M_elements = count;
    return Status::OK;
}

/**
 * @brief Orthogonality left by @ref orthogonal_init, measured in code units.
 *
 * Over the orthonormal axis (rows when wide, columns when tall) a unit row is
 * @f$ 127^2 = 16129 @f$ of squared code and two distinct rows should dot to
 * zero, so the two reported numbers are the worst deviation from each.  The
 * scan is O (short_side^2 * long_side) and runs on the init path only.
 */
[[nodiscard]] inline Status
orthogonal_defect (const int8_t *const codes, const uint32_t rows,
                   const uint32_t cols, uint32_t &worst_norm_error,
                   uint32_t &worst_cross_dot) noexcept
{
    const Status shape = orthogonal_shape_ok (rows, cols);
    if (!status_ok (shape) || codes == nullptr)
        return codes == nullptr ? Status::BAD_LAYOUT : shape;
    const bool wide = rows <= cols;
    const uint32_t short_side = wide ? rows : cols;
    const uint32_t long_side = wide ? cols : rows;
    worst_norm_error = 0u;
    worst_cross_dot = 0u;
    for (uint32_t a = 0u; a < short_side; ++a)
        {
            for (uint32_t b = a; b < short_side; ++b)
                {
                    int64_t total = 0;
                    for (uint32_t line = 0u; line < long_side; ++line)
                        {
                            const uint32_t ia = wide ? a * cols + line
                                                     : line * cols + a;
                            const uint32_t ib = wide ? b * cols + line
                                                     : line * cols + b;
                            total += static_cast<int64_t> (codes[ia])
                                     * codes[ib];
                        }
                    if (a == b)
                        {
                            const int64_t deviation
                                = total > 16129 ? total - 16129
                                                : 16129 - total;
                            if (static_cast<uint64_t> (deviation) > worst_norm_error)
                                worst_norm_error = static_cast<uint32_t> (deviation);
                        }
                    else
                        {
                            const int64_t magnitude
                                = total < 0 ? -total : total;
                            if (static_cast<uint64_t> (magnitude) > worst_cross_dot)
                                worst_cross_dot = static_cast<uint32_t> (magnitude);
                        }
                }
        }
    return Status::OK;
}

/**
 * @brief One-shot dispatch over the named schemes, for a config-driven trainer.
 *
 * @p fans is ignored by the fan-only and orthogonal schemes; the count always
 * comes from the tensor, which the caller must have sized.
 */
template <typename Engine>
[[nodiscard]] inline Status
init_weights (const InitKind kind, Engine &engine, const InitFans fans,
              const uint32_t rows, const uint32_t cols,
              int16_t *const scratch, Trainable &tensor,
              InitResult &out) noexcept
{
    if (!status_ok (tensor.validate ()))
        return Status::BAD_LAYOUT;
    switch (kind)
        {
        case InitKind::XavierUniform:
            return xavier_uniform (engine, fans, tensor._M_code,
                                   tensor._M_count, out);
        case InitKind::XavierNormal:
            return xavier_normal (engine, fans, tensor._M_code,
                                  tensor._M_count, out);
        case InitKind::KaimingUniform:
        case InitKind::HeUniform:
            return kaiming_uniform (engine, fans._M_fan_in, _S_q15_one,
                                    tensor._M_code, tensor._M_count, out);
        case InitKind::HeNormal:
            return he_normal (engine, fans._M_fan_in, tensor._M_code,
                              tensor._M_count, out);
        case InitKind::KaimingNormal:
            return kaiming_normal (engine, fans._M_fan_in, _S_q15_one,
                                   tensor._M_code, tensor._M_count, out);
        case InitKind::FanInUniform:
            return fan_in_uniform (engine, fans._M_fan_in, tensor._M_code,
                                   tensor._M_count, out);
        case InitKind::Uniform:
            return init_uniform (engine, _S_q15_max, tensor._M_code,
                                 tensor._M_count, out);
        case InitKind::Normal:
            return init_normal (engine, _S_q15_one / 2, tensor._M_code,
                                tensor._M_count, out);
        case InitKind::NormalClamped:
            return init_normal_clamped (engine, 0, _S_q15_one / 4, _S_q15_max,
                                        tensor._M_code, tensor._M_count, out);
        case InitKind::Constant:
            return init_unit_gain (tensor._M_code, tensor._M_count, out);
        case InitKind::Orthogonal:
            return orthogonal_init (engine, rows, cols, _S_q15_one, scratch,
                                    tensor._M_code, out);
        }
    return Status::BAD_LAYOUT;
}
} // namespace zzs::mllee::train
