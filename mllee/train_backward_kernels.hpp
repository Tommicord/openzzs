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
#include "mllee/train_simd.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief The shared numeric kernels every training op is built from.
 *
 * Layer modules (linear, conv, rnn, attention, embedding) are thin: they own the
 * shape arithmetic, the saved tensors and the backward routing, and they call
 * the kernels here for the actual products and reductions.  Keeping that
 * boundary in one place is what lets the assembly primitives stay the single
 * vectorized path -- anything that is a dot product, a lane add or a bulk
 * convert goes through mllee/train_simd.hpp and therefore gets SSE2/SMLAD
 * automatically, while anything that is a gather, a broadcast reduce or a
 * strided transpose keeps a predictable scalar form here instead of being
 * duplicated per op.
 *
 * Two accumulator widths are in use, and the choice is forced by overflow, not
 * taste: an int8 code product is at most 16129, so a code-domain GEMM can
 * accumulate over thousands of terms in an int32 lane and use the rank-1 update
 * form (contiguous inner loop, one multiply per element), leaving the
 * requantization to the caller's gemm_narrow_codes; a Q15 lane product is up to
 * 1.07e9, so the same accumulator would overflow after three terms, and the Q15
 * GEMM reduces each output element in an int64 local and stores the already
 * requantized lane (narrow it with shift 0).
 */
namespace zzs::mllee::train
{
/** @brief Edge length of a transpose tile: 8x8 codes = 64 bytes of frame. */
inline constexpr uint32_t _S_transpose_block = 8u;

/** @brief Largest k a code-domain accumulator is validated against (see gemm_codes). */
inline constexpr uint32_t _S_max_code_depth = 65535u;

/**
 * @brief Narrow an int64 Q30-scale reduction into a saturated Q15 lane.
 *
 * The clamp happens before the shift: a long reduction (a softmax over a
 * sequence, a linear layer over a batch) can exceed int32 range, and truncating
 * first would wrap the sign instead of saturating.
 */
[[nodiscard]] inline constexpr int32_t
narrow_sum_q15 (const int64_t total) noexcept
{
     constexpr int64_t limit = static_cast<int64_t> (_S_q15_max) << _S_grad_frac_bits;
    const int64_t bounded
        = total > limit ? limit : (total < -limit ? -limit : total);
    return q15_clamp (_S_round_shift (bounded, _S_grad_frac_bits));
}

/** @brief Saturate a plain (already Q15-scaled) lane sum, e.g. a bias reduction. */
[[nodiscard]] inline constexpr int32_t
clamp_sum_q15 (const int64_t total) noexcept
{
    return q15_clamp (static_cast<int32_t> (
        total > _S_q15_max ? _S_q15_max : (total < _S_q15_min ? _S_q15_min
                                                               : total)));
}

/**
 * @brief A strided matrix view over either element type.
 *
 * Exactly one of @p _M_codes / @p _M_lanes is non-null; the strides are in
 * elements, so a transposed matrix is the same view with the strides swapped and
 * no copy -- which is how the backward pass reads W^T without materializing it.
 */
struct Matrix
{
    const int8_t *_M_codes = nullptr;
    const int16_t *_M_lanes = nullptr;
    uint32_t _M_rows = 0u;
    uint32_t _M_cols = 0u;
    uint32_t _M_row_stride = 0u;
    uint32_t _M_col_stride = 1u;

    [[nodiscard]] constexpr int32_t
    code_at (const uint32_t row, const uint32_t col) const noexcept
    {
        return this->_M_codes[row * this->_M_row_stride
                              + col * this->_M_col_stride];
    }

    [[nodiscard]] constexpr int32_t
    lane_at (const uint32_t row, const uint32_t col) const noexcept
    {
        return this->_M_lanes[row * this->_M_row_stride
                              + col * this->_M_col_stride];
    }

    [[nodiscard]] constexpr bool
    is_code () const noexcept
    {
        return this->_M_codes != nullptr;
    }

    [[nodiscard]] constexpr bool
    is_lane () const noexcept
    {
        return this->_M_lanes != nullptr;
    }
};

/**
 * @brief View of a code matrix: @p rows x @p cols logically, @p transposed
 *        meaning the buffer is that matrix's transpose in row-major order.
 *
 * The caller always names the logical shape it wants, so a backward pass that
 * needs W^T passes the same rows and cols it passed for W and flips the flag.
 */
[[nodiscard]] inline constexpr Matrix
matrix_codes (const int8_t *const values, const uint32_t rows,
              const uint32_t cols, const bool transposed) noexcept
{
    Matrix view;
    view._M_codes = values;
    view._M_rows = rows;
    view._M_cols = cols;
    view._M_row_stride = transposed ? 1u : cols;
    view._M_col_stride = transposed ? rows : 1u;
    return view;
}

/** @brief Lane-matrix view with the same stride convention as matrix_codes. */
[[nodiscard]] inline constexpr Matrix
matrix_lanes (const int16_t *const values, const uint32_t rows,
              const uint32_t cols, const bool transposed) noexcept
{
    Matrix view;
    view._M_lanes = values;
    view._M_rows = rows;
    view._M_cols = cols;
    view._M_row_stride = transposed ? 1u : cols;
    view._M_col_stride = transposed ? rows : 1u;
    return view;
}

/** @brief Inner-dimension agreement between two multiplicands. */
[[nodiscard]] inline constexpr Status
matrix_inner_ok (const Matrix &a, const Matrix &b) noexcept
{
    if (!a.is_code () && !a.is_lane ())
        return Status::BAD_LAYOUT;
    if (!b.is_code () && !b.is_lane ())
        return Status::BAD_LAYOUT;
    if (a.is_code () != b.is_code ())
        return Status::BAD_LAYOUT;
    return a._M_cols == b._M_rows ? Status::OK : Status::BAD_LAYOUT;
}

/**
 * @brief Code-domain GEMM, rank-1 update form: C += A * B.
 *
 * @p c must already be zeroed (Accum::zero) and holds m * n int32 accumulators;
 * the caller requantizes with gemm_narrow.  Depth is bounded because the
 * accumulator is int32: 16129 * k must fit, hence the validation.
 */
[[nodiscard]] inline Status
gemm_codes (const Matrix &a, const Matrix &b, const Accum &c) noexcept
{
    const Status inner = matrix_inner_ok (a, b);
    if (!status_ok (inner))
        return inner;
    if (!c._M_data || c._M_count < a._M_rows * b._M_cols)
        return Status::BAD_LAYOUT;
    if (a._M_cols > _S_max_code_depth)
        return Status::OUT_OF_RANGE;
    for (uint32_t row = 0; row < a._M_rows; ++row)
        {
            int32_t *const destination = c._M_data + row * b._M_cols;
            for (uint32_t inner_index = 0; inner_index < a._M_cols; ++inner_index)
                {
                    const int32_t left = a.code_at (row, inner_index);
                    if (left == 0)
                        continue;
                    const int8_t *const source_row
                        = b._M_codes + inner_index * b._M_row_stride;
                    for (uint32_t col = 0; col < b._M_cols; ++col)
                        destination[col]
                            += left * source_row[col * b._M_col_stride];
                }
        }
    return Status::OK;
}

/**
 * @brief Q15 lane GEMM: C = sum_k A[i][k] * B[k][j], reduced in int64 per output.
 *
 * The int64 local is the only way to keep a long inner dimension (an attention
 * score over a sequence, a linear layer over a batch) exact, and it costs no
 * stack because it lives in one variable.
 */
[[nodiscard]] inline Status
gemm_q15 (const Matrix &a, const Matrix &b, const Accum &c) noexcept
{
    const Status inner = matrix_inner_ok (a, b);
    if (!status_ok (inner))
        return inner;
    if (!c._M_data || c._M_count < a._M_rows * b._M_cols)
        return Status::BAD_LAYOUT;
    if (a.is_code ())
        return Status::BAD_LAYOUT;
    for (uint32_t row = 0; row < a._M_rows; ++row)
        {
            int32_t *const destination = c._M_data + row * b._M_cols;
            for (uint32_t col = 0; col < b._M_cols; ++col)
                {
                    int64_t sum = 0;
                    for (uint32_t inner_index = 0; inner_index < a._M_cols; ++inner_index)
                        sum += static_cast<int64_t> (a.lane_at (row, inner_index))
                               * b.lane_at (inner_index, col);
                    destination[col] = narrow_sum_q15 (sum);
                }
        }
    return Status::OK;
}

/** @brief Narrow an int32 accumulator grid into requantized int8 codes. */
inline void
gemm_narrow_codes (const Accum &c, int8_t *const codes, const uint32_t shift) noexcept
{
    for (uint32_t i = 0; i < c._M_count; ++i)
        codes[i] = _S_i8_code (_S_round_shift (static_cast<int64_t> (c._M_data[i]),
                                               shift));
}

/** @brief Narrow an int32 accumulator grid into Q15 lanes. */
inline void
gemm_narrow_lanes (const Accum &c, int16_t *const lanes, const uint32_t shift) noexcept
{
    for (uint32_t i = 0; i < c._M_count; ++i)
        lanes[i] = q15_to_i16_sat (_S_round_shift (static_cast<int64_t> (c._M_data[i]),
                                                   shift));
}

/**
 * @brief Backward of Y = W X (+ b): both operand gradients in one pass.
 *
 * The weight layout is the inference one -- one row per output unit, so
 * W[o * in + i] -- which is what lets a trained tensor feed Core without a
 * transpose.
 *
 * grad_X[b][i] = sum_o grad_Y[b][o] * W[o * in + i]
 * grad_W[o][i] = sum_b grad_Y[b][o] * X[b][i]
 *
 * Gradients accumulate through Trainable::add_grad, so a tensor used by several
 * ops (a shared embedding table, a tied weight) collects every contribution
 * without a second buffer; @p saturated counts the lanes that clamped.
 */
inline Status
matmul_backward_q15 (const Trainable &x, const Trainable &weight,
                     const Trainable &grad_y, const uint32_t batch,
                     const uint32_t in_features, const uint32_t out_features,
                     uint32_t &saturated) noexcept
{
    if (x._M_count != batch * in_features
        || weight._M_count != in_features * out_features
        || grad_y._M_count != batch * out_features)
        return Status::BAD_LAYOUT;
    if (grad_y._M_grad == nullptr)
        return Status::BAD_LAYOUT;
    if (x._M_grad != nullptr)
        {
            for (uint32_t row = 0; row < batch; ++row)
                for (uint32_t col = 0; col < in_features; ++col)
                    {
                        int64_t sum = 0;
                        for (uint32_t out = 0; out < out_features; ++out)
                            sum += static_cast<int64_t> (
                                       grad_y._M_grad[row * out_features + out])
                                   * weight.q15_at (out * in_features + col);
                        if (x.add_grad (row * in_features + col, narrow_sum_q15 (sum)))
                            ++saturated;
                    }
        }
    if (weight._M_grad == nullptr)
        return Status::OK;
    for (uint32_t out = 0; out < out_features; ++out)
        for (uint32_t col = 0; col < in_features; ++col)
            {
                int64_t sum = 0;
                for (uint32_t row = 0; row < batch; ++row)
                    sum += static_cast<int64_t> (
                               grad_y._M_grad[row * out_features + out])
                           * x.q15_at (row * in_features + col);
                if (weight.add_grad (out * in_features + col, narrow_sum_q15 (sum)))
                    ++saturated;
            }
    return Status::OK;
}

/** @brief Sum of @p groups blocks of @p inner contiguous lanes into @p destination. */
inline void
reduce_axis_sum_q15 (const int16_t *const values, const uint32_t groups,
                     const uint32_t inner, int16_t *const destination,
                     uint32_t &saturated) noexcept
{
    for (uint32_t block = 0; block < groups; ++block)
        {
            int64_t total = 0;
            const uint32_t base = block * inner;
            for (uint32_t i = 0; i < inner; ++i)
                total += values[base + i];
            const int32_t lane = clamp_sum_q15 (total);
            if (static_cast<int64_t> (lane) != total)
                ++saturated;
            destination[block] = static_cast<int16_t> (lane);
        }
}

/**
 * @brief Bias backward: sum a [outer, channel, inner] gradient over every axis
 *        but @p channel.
 *
 * With inner == 1 (the NCHW case) the channel axis is the stride, so the reads
 * jump by the spatial size; with outer == 1 they are contiguous.  Both orders
 * keep one inner loop of stride @p inner, which is the best the layout allows
 * without a transpose.
 */
inline void
channel_backward (const int16_t *const grad, const uint32_t outer,
                  const uint32_t channels, const uint32_t inner,
                  int16_t *const destination, uint32_t &saturated) noexcept
{
    for (uint32_t channel = 0; channel < channels; ++channel)
        {
            int64_t total = 0;
            for (uint32_t block = 0; block < outer; ++block)
                {
                    const uint32_t base
                        = (block * channels + channel) * inner;
                    for (uint32_t i = 0; i < inner; ++i)
                        total += grad[base + i];
                }
            const int32_t clamped = clamp_sum_q15 (total);
            if (static_cast<int64_t> (clamped) != total)
                ++saturated;
            destination[channel] = static_cast<int16_t> (clamped);
        }
}

/**
 * @brief Generic broadcast backward: reduce a destination gradient into the
 *        operand it was broadcast from.
 *
 * This is the slow, always-correct form (one broadcast_index per element).  The
 * channel and axis reductions above exist precisely so the common layouts do not
 * have to pay for it.
 */
inline Status
broadcast_backward (const Trainable &grad_destination, const Trainable &source,
                    uint32_t &saturated) noexcept
{
    const Status broadcast
        = broadcast_ok (source._M_shape, grad_destination._M_shape);
    if (!status_ok (broadcast))
        return broadcast;
    if (source._M_grad == nullptr)
        return Status::OK;
    for (uint32_t flat = 0; flat < grad_destination._M_count; ++flat)
        {
            const uint32_t index = broadcast_index (
                flat, source._M_shape, grad_destination._M_shape);
            if (source.add_grad (index, grad_destination._M_grad[flat]))
                ++saturated;
        }
    return Status::OK;
}

/** @brief grad_in = mask ? grad_out : 0 -- the shared atom of dropout, max-pool and gate backward. */
inline void
route_by_mask (const int16_t *const grad_out, const uint8_t *const mask,
               int16_t *const grad_in, const uint32_t count) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        grad_in[i] = mask[i] != 0u ? grad_out[i] : 0;
}

/** @brief Scatter-add one gradient lane per index (gather, embedding and max-pool backward). */
inline void
scatter_by_index (const int16_t *const grad_out, const uint16_t *const index,
                  int16_t *const destination, const uint32_t destination_count,
                  uint32_t &saturated) noexcept
{
    for (uint32_t i = 0; i < destination_count; ++i)
        destination[i] = 0;
    for (uint32_t i = 0; i < destination_count; ++i)
        {
            const uint32_t target = index[i];
            if (target >= destination_count)
                continue;
            const int32_t sum = destination[target] + grad_out[i];
            if (sum > _S_q15_max || sum < _S_q15_min)
                {
                    destination[target] = q15_to_i16_sat (sum);
                    ++saturated;
                }
            else
                destination[target] = static_cast<int16_t> (sum);
        }
}

/** @brief Row gather from a code table (embedding and attention value lookup). */
inline void
gather_rows (const int8_t *const table, const uint32_t row_bytes,
             const uint16_t *const index, const uint32_t rows,
             int8_t *const destination) noexcept
{
    for (uint32_t row = 0; row < rows; ++row)
        {
            const int8_t *const source = table + static_cast<uint32_t> (index[row]) * row_bytes;
            for (uint32_t byte = 0; byte < row_bytes; ++byte)
                destination[row * row_bytes + byte] = source[byte];
        }
}

/** @brief Blocked int8 transpose: rows x cols -> cols x rows, 8x8 tiles. */
inline void
transpose_codes (const int8_t *const source, int8_t *const destination,
                 const uint32_t rows, const uint32_t cols) noexcept
{
    int8_t tile[_S_transpose_block * _S_transpose_block];
    for (uint32_t block_row = 0; block_row < rows; block_row += _S_transpose_block)
        for (uint32_t block_col = 0; block_col < cols;
             block_col += _S_transpose_block)
            {
                const uint32_t height
                    = rows - block_row < _S_transpose_block ? rows - block_row
                                                             : _S_transpose_block;
                const uint32_t width
                    = cols - block_col < _S_transpose_block ? cols - block_col
                                                             : _S_transpose_block;
                for (uint32_t r = 0; r < height; ++r)
                    for (uint32_t w = 0; w < width; ++w)
                        tile[r * _S_transpose_block + w]
                            = source[(block_row + r) * cols + block_col + w];
                for (uint32_t r = 0; r < height; ++r)
                    for (uint32_t w = 0; w < width; ++w)
                        destination[(block_col + w) * rows + block_row + r]
                            = tile[r * _S_transpose_block + w];
            }
}

/** @brief Blocked Q15 transpose, same tiling as the code form. */
inline void
transpose_lanes (const int16_t *const source, int16_t *const destination,
                 const uint32_t rows, const uint32_t cols) noexcept
{
    int16_t tile[_S_transpose_block * _S_transpose_block];
    for (uint32_t block_row = 0; block_row < rows; block_row += _S_transpose_block)
        for (uint32_t block_col = 0; block_col < cols;
             block_col += _S_transpose_block)
            {
                const uint32_t height
                    = rows - block_row < _S_transpose_block ? rows - block_row
                                                             : _S_transpose_block;
                const uint32_t width
                    = cols - block_col < _S_transpose_block ? cols - block_col
                                                             : _S_transpose_block;
                for (uint32_t r = 0; r < height; ++r)
                    for (uint32_t w = 0; w < width; ++w)
                        tile[r * _S_transpose_block + w]
                            = source[(block_row + r) * cols + block_col + w];
                for (uint32_t r = 0; r < height; ++r)
                    for (uint32_t w = 0; w < width; ++w)
                        destination[(block_col + w) * rows + block_row + r]
                            = tile[r * _S_transpose_block + w];
            }
}

/**
 * @name Lane buffers through the SIMD primitives
 *
 * Thin wrappers only: their whole purpose is that an op module never hand-rolls
 * a lane loop that the assembly already does eight at a time.
 * @{
 */

[[nodiscard]] inline uint32_t
add_lanes (int16_t *const destination, const int16_t *const source,
           const uint32_t count) noexcept
{
    return train_simd::add_grad (destination, source, count);
}

inline void
axpy_lanes (int16_t *const destination, const int16_t *const source,
            const int32_t scalar_q15, const uint32_t count) noexcept
{
    train_simd::axpy_q15 (destination, source, scalar_q15, count);
}

inline void
scale_lanes (int16_t *const values, const int32_t scalar_q15,
             const uint32_t count) noexcept
{
    train_simd::scale_q15 (values, scalar_q15, count);
}

inline void
fill_lanes (int16_t *const values, const int32_t scalar_q15,
            const uint32_t count) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        values[i] = q15_to_i16_sat (scalar_q15);
}

/** @brief L1 norm of a gradient buffer in Q15. */
[[nodiscard]] inline int32_t
l1_norm_q15 (const int16_t *const values, const uint32_t count) noexcept
{
    return train_simd::sum_abs_q15 (values, count);
}

/**
 * @brief L2 norm of a gradient buffer in Q15.
 *
 * The squared sum is Q30, so its integer square root is already a Q15 magnitude
 * (sqrt(g^2 * 2^30) = |g| * 2^15) and no second requantization step is needed.
 */
[[nodiscard]] inline int32_t
l2_norm_q15 (const int16_t *const values, const uint32_t count) noexcept
{
    const int32_t squares = train_simd::sum_sq_q15 (values, count);
    if (squares <= 0)
        return 0;
    return static_cast<int32_t> (_S_isqrt (static_cast<uint32_t> (squares)));
}

/** @brief Dot product of two lane buffers, Q15 result. */
[[nodiscard]] inline int32_t
dot_lanes (const int16_t *const lhs, const int16_t *const rhs,
           const uint32_t count) noexcept
{
    return train_simd::dot_q15 (lhs, rhs, count);
}

/** @} */
} // namespace zzs::mllee::train
