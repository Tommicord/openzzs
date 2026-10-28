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
 * @brief Non-aliasing hint for the hot kernel pointer pairs.
 *
 * The training kernels read one buffer and write another that provably cannot
 * overlap (both are carved from the same Pool at disjoint offsets), but the
 * compiler cannot see that from a call boundary.  `__restrict__` / `__restrict`
 * differ in spelling between GCC/Clang and MSVC, so the kernels use this one
 * name; an unknown compiler gets the empty expansion and the same semantics.
 */
#if defined (__GNUC__) || defined (__clang__)
#define ZZS_TRAIN_RESTRICT __restrict__
#else
#define ZZS_TRAIN_RESTRICT
#endif

/**
 * @brief Tell the optimizer an invariant the loop bound cannot prove.
 *
 * Only used where a `Status` check above the loop already established the
 * fact; a portable build simply drops the hint.
 */
#if defined (__clang__)
#define ZZS_TRAIN_ASSUMPTION(expression) __builtin_assume (expression)
#elif defined (__GNUC__)
#define ZZS_TRAIN_ASSUMPTION(expression)                                 \
    do                                                                  \
        {                                                               \
            if (!(expression))                                          \
                __builtin_unreachable ();                               \
        }                                                               \
    while (false)
#else
#define ZZS_TRAIN_ASSUMPTION(expression)                                 \
    do                                                                  \
        {                                                               \
        }                                                               \
    while (false)
#endif

/**
 * @brief Assert a pointer's alignment where the buffer is aligned by
 *        construction (a Pool carve or an `alignas` frame array).
 */
#if (defined (__GNUC__) || defined (__clang__)) && !defined (__apple__)
#define ZZS_TRAIN_ASSUME_ALIGNED(pointer, bytes)                         \
    __builtin_assume_aligned ((pointer), (bytes))
#else
#define ZZS_TRAIN_ASSUME_ALIGNED(pointer, bytes) (pointer)
#endif

/**
 * @brief Tensor views shared by every training kernel.
 *
 * A trainable tensor is one int8 code stream plus one Q15 gradient lane per
 * element, both carved from the caller's Pool.  Two separate buffers (instead of
 * a struct-of-arrays element) keep both streams stride-1 and 4-byte aligned,
 * which is what lets the SIMD kernels process a lane group at a time and keeps
 * a forward pass from pulling gradient bytes through the cache.
 *
 * Gradients saturate rather than wrap: a Q15 lane overflow is a modelling
 * error the caller can act on (see train_loss_scale.hpp), so it must never
 * silently change sign.
 */
namespace zzs::mllee::train
{
/** @brief NCHW/NHWC/time-series depth: four axes is the whole training opset. */
inline constexpr uint8_t _S_max_rank = 4u;

/** @brief Largest element count a tensor view addresses. */
inline constexpr uint16_t _S_max_elements = 65535u;

/** @brief Per-tensor behaviour switches. */
enum Flag : uint32_t
{
    RequiresGrad = 1u << 0,
    Frozen = 1u << 1,
    Quantized = 1u << 2,
    /** @brief Bias-like tensor: no weight decay, no quantization grid. */
    BiasLike = 1u << 3,
};

/**
 * @brief Row-major shape, rank at most four, no dynamic allocation.
 *
 * A zero-length axis is not a valid shape; a size-1 axis is, because the
 * broadcasting reductions (sum over a channel, mean over a batch) address it
 * explicitly.
 */
struct Shape
{
    uint16_t _M_dims[_S_max_rank] = { 1u, 1u, 1u, 1u };
    uint8_t _M_rank = 0u;

    [[nodiscard]] static constexpr Shape
    scalar () noexcept
    {
        Shape shape;
        shape._M_dims[0] = 1u;
        shape._M_rank = 1u;
        return shape;
    }

    [[nodiscard]] static constexpr Shape
    vector (const uint16_t c) noexcept
    {
        Shape shape;
        shape._M_dims[0] = c;
        shape._M_rank = 1u;
        return shape;
    }

    [[nodiscard]] static constexpr Shape
    matrix (const uint16_t rows, const uint16_t cols) noexcept
    {
        Shape shape;
        shape._M_dims[0] = rows;
        shape._M_dims[1] = cols;
        shape._M_rank = 2u;
        return shape;
    }

    [[nodiscard]] static constexpr Shape
    tensor3 (const uint16_t a, const uint16_t b, const uint16_t c) noexcept
    {
        Shape shape;
        shape._M_dims[0] = a;
        shape._M_dims[1] = b;
        shape._M_dims[2] = c;
        shape._M_rank = 3u;
        return shape;
    }

    [[nodiscard]] static constexpr Shape
    tensor4 (const uint16_t n, const uint16_t c, const uint16_t h,
             const uint16_t w) noexcept
    {
        Shape shape;
        shape._M_dims[0] = n;
        shape._M_dims[1] = c;
        shape._M_dims[2] = h;
        shape._M_dims[3] = w;
        shape._M_rank = 4u;
        return shape;
    }

    [[nodiscard]] constexpr uint32_t
    elements () const noexcept
    {
        uint32_t total = 1u;
        for (uint8_t axis = 0; axis < this->_M_rank; ++axis)
            total *= this->_M_dims[axis];
        return total;
    }

    /** @brief Stride of @p axis in elements, row-major (last axis is 1). */
    [[nodiscard]] constexpr uint32_t
    stride (const uint8_t axis) const noexcept
    {
        uint32_t stride = 1u;
        for (uint8_t trailing = static_cast<uint8_t> (this->_M_rank - 1u);
             trailing > axis; --trailing)
            stride *= this->_M_dims[trailing];
        return stride;
    }

    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (this->_M_rank == 0u || this->_M_rank > _S_max_rank)
            return Status::BAD_LAYOUT;
        uint32_t total = 1u;
        for (uint8_t axis = 0; axis < this->_M_rank; ++axis)
            {
                if (this->_M_dims[axis] == 0u)
                    return Status::BAD_LAYOUT;
                total *= this->_M_dims[axis];
            }
        return total <= _S_max_elements ? Status::OK : Status::OUT_OF_RANGE;
    }

    [[nodiscard]] constexpr bool
    operator== (const Shape &other) const noexcept
    {
        if (this->_M_rank != other._M_rank)
            return false;
        for (uint8_t axis = 0; axis < this->_M_rank; ++axis)
            if (this->_M_dims[axis] != other._M_dims[axis])
                return false;
        return true;
    }

    /** @brief Row-major index of a rank-2 coordinate pair. */
    [[nodiscard]] constexpr uint32_t
    index2 (const uint32_t row, const uint32_t col) const noexcept
    {
        return row * this->_M_dims[1] + col;
    }

    /** @brief Row-major index of a rank-4 coordinate. */
    [[nodiscard]] constexpr uint32_t
    index4 (const uint32_t n, const uint32_t c, const uint32_t h,
            const uint32_t w) const noexcept
    {
        return ((n * this->_M_dims[1] + c) * this->_M_dims[2] + h)
               * this->_M_dims[3] + w;
    }
};

static_assert (sizeof (Shape) <= 10u, "shape view must stay a copyable leaf");

/**
 * @brief One trainable tensor: int8 codes plus Q15 gradient lanes.
 *
 * The pointers are non-owning.  A tensor with no gradient buffer is a
 * forward-only view (an input or a frozen weight); kernels must check
 * has_grad() before accumulating into it.
 *
 * Writing a code or a gradient lane mutates the buffer, not the view, so those
 * members are const-qualified: a backward function receives its operands as
 * const views and still accumulates into them.
 */
struct Trainable
{
    int8_t *_M_code = nullptr;
    int16_t *_M_grad = nullptr;
    Shape _M_shape;
    uint32_t _M_count = 0u;
    /** @brief Real value of code c is c * _M_scale_q15 / (127 * 32768). */
    int32_t _M_scale_q15 = _S_q15_one;
    uint32_t _M_flags = 0u;

    [[nodiscard]] static Trainable
    make (int8_t *const code, int16_t *const grad, const Shape shape,
          const uint32_t flags) noexcept
    {
        Trainable view;
        view._M_code = code;
        view._M_grad = grad;
        view._M_shape = shape;
        view._M_count = shape.elements ();
        view._M_flags = flags;
        return view;
    }

    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        const Status shape_status = this->_M_shape.validate ();
        if (!status_ok (shape_status))
            return shape_status;
        if (this->_M_code == nullptr)
            return Status::BAD_LAYOUT;
        return this->_M_count == this->_M_shape.elements ()
                   ? Status::OK
                   : Status::BAD_LAYOUT;
    }

    [[nodiscard]] constexpr bool
    has_grad () const noexcept
    {
        return this->_M_grad != nullptr;
    }

    [[nodiscard]] constexpr bool
    requires_grad () const noexcept
    {
        return (this->_M_flags & Flag::RequiresGrad) != 0u
               && (this->_M_flags & Flag::Frozen) == 0u;
    }

    /** @brief True when both streams are 4-byte aligned, i.e. SIMD-ready. */
    [[nodiscard]] inline bool
    simd_ready () const noexcept
    {
        return (reinterpret_cast<uintptr_t> (this->_M_code) & 3u) == 0u
               && (this->_M_grad == nullptr
                   || (reinterpret_cast<uintptr_t> (this->_M_grad) & 7u) == 0u);
    }

    [[nodiscard]] constexpr int32_t
    code_at (const uint32_t index) const noexcept
    {
        return this->_M_code[index];
    }

    constexpr void
    set_code (const uint32_t index, const int32_t value) const noexcept
    {
        this->_M_code[index] = _S_i8_code (value);
    }

    /** @brief Q15 value of an element, i.e. its code on the tensor's scale. */
    [[nodiscard]] constexpr int32_t
    q15_at (const uint32_t index) const noexcept
    {
        return code_to_q15 (this->_M_code[index]);
    }

    [[nodiscard]] constexpr int32_t
    grad_at (const uint32_t index) const noexcept
    {
        return this->_M_grad == nullptr ? 0 : this->_M_grad[index];
    }

    /** @brief Overwrite one gradient lane with a saturated Q15 value. */
    constexpr void
    set_grad (const uint32_t index, const int32_t value_q15) const noexcept
    {
        if (this->_M_grad != nullptr)
            this->_M_grad[index] = q15_to_i16_sat (value_q15);
    }

    /**
     * @brief Saturating gradient accumulation, the one hot write of the module.
     *
     * @return true when the addition clamped, so a caller can count the
     *         saturation events that trigger a loss-scale backoff.
     */
    [[nodiscard]] constexpr bool
    add_grad (const uint32_t index, const int32_t delta_q15) const noexcept
    {
        if (this->_M_grad == nullptr)
            return false;
        const int32_t sum = this->_M_grad[index] + delta_q15;
        if (sum > _S_q15_max || sum < _S_q15_min)
            {
                this->_M_grad[index] = q15_to_i16_sat (sum);
                return true;
            }
        this->_M_grad[index] = static_cast<int16_t> (sum);
        return false;
    }

    constexpr void
    zero_grad () const noexcept
    {
        if (this->_M_grad == nullptr)
            return;
        for (uint32_t i = 0; i < this->_M_count; ++i)
            this->_M_grad[i] = 0;
    }

    /** @brief Bytes of pool this view occupies, used by the RAM gate. */
    [[nodiscard]] constexpr uint32_t
    bytes () const noexcept
    {
        return this->_M_count * (1u + (this->_M_grad != nullptr ? 2u : 0u));
    }
};

/** @brief int32 accumulator buffer, i.e. the pre-narrowing form of a gradient. */
struct Accum
{
    int32_t *_M_data = nullptr;
    uint32_t _M_count = 0u;

    constexpr void
    zero () const noexcept
    {
        for (uint32_t i = 0; i < this->_M_count; ++i)
            this->_M_data[i] = 0;
    }

    constexpr void
    add (const uint32_t index, const int32_t delta) const noexcept
    {
        this->_M_data[index] += delta;
    }

    /** @brief Narrow the accumulator into Q15 gradient lanes, right-shifting first. */
    constexpr void
    narrow_to (const Trainable &destination, const uint32_t shift) const noexcept
    {
        if (destination._M_grad == nullptr)
            return;
        for (uint32_t i = 0; i < this->_M_count; ++i)
            destination._M_grad[i] = q15_to_i16_sat (
                _S_round_shift (static_cast<int64_t> (this->_M_data[i]), shift));
    }
};

/**
 * @name Elementwise reductions over training buffers
 *
 * Straight stride-1 loops; the vectorized forms live in mllee/train_simd.hpp and
 * keep these semantics exactly.
 * @{
 */

/** @brief Sum of int8 codes. */
[[nodiscard]] inline int32_t
sum_codes (const int8_t *const values, const uint32_t count) noexcept
{
    int32_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += values[i];
    return total;
}

/** @brief Sum of Q15 lanes, saturated to the Q15 range. */
[[nodiscard]] inline int32_t
sum_q15 (const int16_t *const values, const uint32_t count) noexcept
{
    int64_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += values[i];
    return q15_clamp (static_cast<int32_t> (
        total > _S_q15_max ? _S_q15_max : (total < _S_q15_min ? _S_q15_min : total)));
}

/** @brief Mean of Q15 lanes as a Q15 value, rounded half away from zero. */
[[nodiscard]] inline int32_t
mean_q15 (const int16_t *const values, const uint32_t count) noexcept
{
    if (count == 0u)
        return 0;
    int64_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += values[i];
    const int64_t n = static_cast<int64_t> (count);
    const int64_t magnitude = total < 0 ? -total : total;
    const int64_t quotient = (magnitude + n / 2) / n;
    return q15_clamp (static_cast<int32_t> (total < 0 ? -quotient : quotient));
}

/** @brief Dot product of two int8 code streams, accumulated in int32. */
[[nodiscard]] inline int32_t
dot_codes (const int8_t *const lhs, const int8_t *const rhs,
           const uint32_t count) noexcept
{
    int32_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<int32_t> (lhs[i]) * rhs[i];
    return total;
}

/** @brief Dot product of two Q15 streams with an int64 accumulator, Q15 result. */
[[nodiscard]] inline int32_t
dot_q15 (const int16_t *const lhs, const int16_t *const rhs,
         const uint32_t count) noexcept
{
    int64_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<int64_t> (lhs[i]) * rhs[i];
    return q15_clamp (_S_round_shift (total, _S_grad_frac_bits));
}

/** @brief Sum of squares of Q15 lanes in int64, for gradient-norm computation. */
[[nodiscard]] inline int64_t
sum_squares_q15 (const int16_t *const values, const uint32_t count) noexcept
{
    int64_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<int64_t> (values[i]) * values[i];
    return total;
}

/** @brief Largest absolute gradient lane, Q15. */
[[nodiscard]] inline int32_t
max_abs_q15 (const int16_t *const values, const uint32_t count) noexcept
{
    int32_t maximum = 0;
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t magnitude = values[i] < 0 ? -values[i] : values[i];
            if (magnitude > maximum)
                maximum = magnitude;
        }
    return maximum;
}

/** @} */

/** @brief True when @p source can broadcast into @p destination elementwise. */
[[nodiscard]] inline Status
broadcast_ok (const Shape &source, const Shape &destination) noexcept
{
    if (source._M_rank > destination._M_rank)
        return Status::BAD_LAYOUT;
    const uint8_t offset
        = static_cast<uint8_t> (destination._M_rank - source._M_rank);
    for (uint8_t axis = 0; axis < source._M_rank; ++axis)
        {
            const uint16_t want = destination._M_dims[axis + offset];
            const uint16_t have = source._M_dims[axis];
            if (have != want && have != 1u)
                return Status::BAD_LAYOUT;
        }
    return Status::OK;
}

/** @brief Element index inside @p source of the flat @p destination position.
 *
 * Axes align on the right, as in PyTorch broadcasting: an axis @p source does
 * not have, or one it has as a size-1 axis, contributes nothing to the index and
 * therefore reads the same element for every coordinate along it.
 */
[[nodiscard]] inline uint32_t
broadcast_index (const uint32_t flat, const Shape &source,
                 const Shape &destination) noexcept
{
    const uint8_t offset
        = static_cast<uint8_t> (destination._M_rank - source._M_rank);
    uint32_t remaining = flat;
    uint32_t index = 0u;
    for (uint8_t axis = destination._M_rank; axis-- > 0u;)
        {
            const uint32_t coordinate
                = remaining % destination._M_dims[axis];
            remaining /= destination._M_dims[axis];
            if (axis < offset || source._M_dims[axis - offset] == 1u)
                continue;
            index += coordinate * source.stride (static_cast<uint8_t> (axis - offset));
        }
    return index;
}
} // namespace zzs::mllee::train
