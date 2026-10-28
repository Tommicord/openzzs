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

#include "mllee/ops.hpp"

#include <cstdint>

namespace zzs::mllee
{
/**
 * @brief Arena-only tensor ops and the full-opset dispatcher.
 *
 * Every kernel here reads and writes activation bytes that already live in the
 * arena (or, for the embedding table, in Flash); none of them touch the packlow
 * reader, so they cost no codec state and keep the stack chain flat.
 */

/** @brief Saturated sum/difference of two int8 tensors, or a requantized product. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_binary (const Layer &layer, const int8_t *const lhs, const int8_t *const rhs,
           int8_t *const output) noexcept
{
    const uint32_t count = layer._M_in_dim;
    const int32_t multiplier = layer._M_requant_mult;
    const int32_t aux = _S_act_param (layer);
    if (multiplier == 32768 && layer._M_act == Act::NONE
        && (layer._M_op == Op::ADD || layer._M_op == Op::SUB))
        {
            if (layer._M_op == Op::ADD)
                simd::add_sat_i8 (lhs, rhs, output, count);
            else
                simd::sub_sat_i8 (lhs, rhs, output, count);
            return Status::OK;
        }
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t a = lhs[i];
            const int32_t b = rhs[i];
            const int64_t raw = layer._M_op == Op::SUB ? static_cast<int64_t> (a - b)
                                      : layer._M_op == Op::MUL ? static_cast<int64_t> (a * b)
                                                               : static_cast<int64_t> (a + b);
            output[i] = _S_i8_code (
                _S_apply_act (layer._M_act, simd::requant_q15 (raw, multiplier), aux));
        }
    return Status::OK;
}

/** @brief Clamp to the signed bounds carried in the kernel and stride bytes. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_clamp (const Layer &layer, const int8_t *const input, int8_t *const output) noexcept
{
    const int32_t low = static_cast<int8_t> (layer._M_kernel);
    const int32_t high = static_cast<int8_t> (layer._M_stride);
    if (low > high)
        return Status::BAD_LAYOUT;
    simd::clamp_i8 (input, output, layer._M_in_dim, static_cast<int8_t> (low),
                    static_cast<int8_t> (high));
    return Status::OK;
}

/** @brief Parametric ReLU with a per-element Q7 slope table. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_prelu (const Layer &layer, const uint8_t *const model, const int8_t *const input,
          int8_t *const output) noexcept
{
    const uint32_t count = layer._M_in_dim;
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t value = input[i];
            if (value >= 0)
                {
                    output[i] = static_cast<int8_t> (value);
                    continue;
                }
            const int32_t slope
                = layer._M_scale_off != 0 ? _S_i16 (model + layer._M_scale_off + i * 2u) : 32;
            output[i] = _S_i8_code (_S_round_shift (static_cast<int64_t> (value) * slope, 7u));
        }
    return Status::OK;
}

/** @brief BatchNorm-style per-channel normalization with per-channel affine. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_batch_norm (const Layer &layer, const uint8_t *const model, const int8_t *const input,
               int8_t *const output) noexcept
{
    const uint32_t count = layer._M_in_dim;
    const uint32_t window = layer._M_seq != 0 ? layer._M_seq : count;
    if (window == 0 || count % window != 0)
        return Status::BAD_LAYOUT;
    const bool rms = (layer._M_flags & NORM_RMS) != 0;
    const uint32_t channels = count / window;
    const int32_t aux = _S_act_param (layer);

    for (uint32_t channel = 0; channel < channels; ++channel)
        {
            int32_t mean = 0;
            uint32_t deviation = 0;
            _S_window_stats (input + channel * window, window, rms, mean, deviation);
            const _S_Div32 deviation_div = _S_div32_make (deviation);
            const int32_t gamma = layer._M_scale_off != 0
                                      ? _S_i16 (model + layer._M_scale_off + channel * 2u)
                                      : 128;
            const int32_t beta = layer._M_bias_off != 0
                                     ? _S_i16 (model + layer._M_bias_off + channel * 2u)
                                     : 0;
            for (uint32_t i = 0; i < window; ++i)
                {
                    const uint32_t index = channel * window + i;
                    output[index] = _S_i8_code (
                        _S_apply_act (layer._M_act,
                                      _S_norm_value (input[index], mean, deviation_div, rms,
                                                     gamma, layer._M_requant_mult, beta),
                                      aux));
                }
        }
    return Status::OK;
}

/** @brief GroupNorm: normalize contiguous groups, affine per group. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_group_norm (const Layer &layer, const uint8_t *const model, const int8_t *const input,
               int8_t *const output) noexcept
{
    const uint32_t count = layer._M_in_dim;
    const uint32_t window = _S_group_len (layer);
    if (window == 0 || count % window != 0)
        return Status::BAD_LAYOUT;
    const bool rms = (layer._M_flags & NORM_RMS) != 0;
    const int32_t aux = _S_act_param (layer);

    for (uint32_t group = 0; group * window < count; ++group)
        {
            const uint32_t base = group * window;
            int32_t mean = 0;
            uint32_t deviation = 0;
            _S_window_stats (input + base, window, rms, mean, deviation);
            const _S_Div32 deviation_div = _S_div32_make (deviation);
            const int32_t gamma
                = layer._M_scale_off != 0 ? _S_i16 (model + layer._M_scale_off + group * 2u) : 128;
            const int32_t beta
                = layer._M_bias_off != 0 ? _S_i16 (model + layer._M_bias_off + group * 2u) : 0;
            for (uint32_t i = 0; i < window; ++i)
                output[base + i] = _S_i8_code (
                    _S_apply_act (layer._M_act,
                                  _S_norm_value (input[base + i], mean, deviation_div, rms,
                                                 gamma, layer._M_requant_mult, beta),
                                  aux));
        }
    return Status::OK;
}

/** @brief L2-normalize each group so that the group norm is 127. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_l2_normalize (const Layer &layer, const int8_t *const input, int8_t *const output) noexcept
{
    const uint32_t count = layer._M_in_dim;
    const uint32_t window = _S_group_len (layer);
    if (window == 0 || count % window != 0)
        return Status::BAD_LAYOUT;
    for (uint32_t base = 0; base < count; base += window)
        {
            const int32_t square_sum = simd::sum_sq_i8 (input + base, window);
            const uint32_t norm
                = square_sum > 0 ? _S_isqrt (static_cast<uint32_t> (square_sum)) : 0u;
            const _S_Div32 norm_div = _S_div32_make (norm);
            for (uint32_t i = 0; i < window; ++i)
                output[base + i]
                    = norm != 0 ? _S_i8_code (
                          _S_div32_i (input[base + i] * 127, norm_div))
                                : 0;
        }
    return Status::OK;
}

/** @brief Max / average / global average pooling over per-channel windows. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_pool (const Layer &layer, const int8_t *const input, int8_t *const output) noexcept
{
    const uint32_t in_length = layer._M_head_dim;
    if (in_length == 0)
        return Status::BAD_LAYOUT;
    const uint32_t channels = layer._M_in_dim / in_length;
    const bool global = layer._M_op == Op::GLOBAL_AVG_POOL;
    const uint32_t out_length = global ? 1u : layer._M_seq;
    const uint32_t kernel = layer._M_kernel;
    const uint32_t stride = layer._M_stride;
    if (kernel == 0 || stride == 0 || out_length == 0 || layer._M_out_dim != channels * out_length)
        return Status::BAD_LAYOUT;
    const uint32_t padding
        = (layer._M_flags & CONV_CAUSAL) != 0 && !global ? kernel - 1u : 0u;
    const bool maximize = layer._M_op == Op::MAX_POOL1D;

    for (uint32_t channel = 0; channel < channels; ++channel)
        {
            const int8_t *const row = input + channel * in_length;
            uint32_t cached_samples = 0;
            _S_Div32 average_div{ 0u, 0u };
            for (uint32_t position = 0; position < out_length; ++position)
                {
                    const uint32_t start
                        = global ? 0u : position * stride;
                    const uint32_t span = global ? in_length : kernel;
                    int32_t best = maximize ? -128 : 0;
                    int32_t total = 0;
                    uint32_t samples = 0;
                    for (uint32_t tap = 0; tap < span; ++tap)
                        {
                            const uint32_t index = start + tap;
                            if (index < padding || index - padding >= in_length)
                                continue;
                            const int32_t value = row[index - padding];
                            ++samples;
                            total += value;
                            if (maximize && value > best)
                                best = value;
                        }
                    int32_t result = 0;
                    if (maximize)
                        result = samples == 0 ? 0 : best;
                    else if (samples != 0)
                        {
                            /* The divisor only changes on the padded edge, so a
                               whole interior row reuses one reciprocal. */
                            if (samples != cached_samples)
                                {
                                    cached_samples = samples;
                                    average_div
                                        = _S_div32_make (2u * samples);
                                }
                            result = _S_div32_i (
                                total * 2 + static_cast<int32_t> (samples), average_div);
                        }
                    output[channel * out_length + position] = _S_i8_code (result);
                }
        }
    return Status::OK;
}

/**
 * @brief Token embedding: gather head_dim codes per input index.
 *
 * A 3-bit row is byte-aligned (the validator requires a multiple of eight
 * codes); 2/4-bit rows are a continuous bit stream, so a row of odd dimension
 * starts in the high half of a shared byte.  Each row is expanded in 16-code
 * tiles by the width-specific kernel: nibble pshufb for I4/F4E2M1, four-way
 * extraction for I2, three-byte groups for the bit-packed I3/F3E1M1, and the
 * 256-entry gather for byte-wide codes.  No path extracts a packed code one at
 * a time.  The codes are still closed with _S_i8_code, so a -128 entry behaves
 * exactly as in the scalar form.
 */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_embedding (const Layer &layer, const uint8_t *const model, const int8_t *const input,
              int8_t *const output) noexcept
{
    const uint32_t dimension = layer._M_head_dim;
    if (dimension == 0)
        return Status::BAD_LAYOUT;
    const int8_t *const lut = _S_weight_lut (model, layer);
    const Dtype dtype = layer._M_dtype;
    const uint32_t bits = _S_code_bits (dtype);
    const uint8_t *const weights = model + layer._M_weight_off;
    for (uint32_t token = 0; token < layer._M_in_dim; ++token)
        {
            const int32_t index = input[token];
            if (index < 0)
                return Status::OUT_OF_RANGE;
            int8_t *const row = output + token * dimension;
            const uint32_t first_code = static_cast<uint32_t> (index) * dimension;
            const uint32_t first_bit = first_code * bits;
            const uint32_t first_byte = first_bit >> 3;
            const uint32_t row_end_bit = (first_code + dimension) * bits;
            if (((row_end_bit + 7u) >> 3) > layer._M_weight_len)
                return Status::OUT_OF_RANGE;
            const uint8_t *const source = weights + first_byte;
            if (bits == 8u)
                {
                    if (lut == nullptr)
                        {
                            for (uint32_t d = 0; d < dimension; ++d)
                                row[d] = _S_i8_code (static_cast<int8_t> (source[d]));
                            continue;
                        }
                    simd::gather_lut8 (lut, source, row, dimension);
                    for (uint32_t d = 0; d < dimension; ++d)
                        row[d] = _S_i8_code (row[d]);
                    continue;
                }
            /* 2/4-bit rows share a byte with the next row when the dimension is
               odd, so the code offset inside the first byte is kept; a 3-bit
               row is byte-aligned (the validator requires a multiple of 8). */
            for (uint32_t done = 0; done < dimension; done += 16u)
                {
                    const uint32_t want = dimension - done < 16u ? dimension - done : 16u;
                    const uint32_t start_bit = first_bit + done * bits;
                    const uint32_t shift = start_bit & 7u;
                    const uint8_t *const tile_source = weights + (start_bit >> 3);
                    int8_t tile[18];
                    if (bits == 4u)
                        {
                            const uint32_t bytes = (shift + want * 4u + 7u) >> 3;
                            simd::expand_codes (tile_source, tile, bytes, lut);
                        }
                    else if (bits == 2u)
                        {
                            const uint32_t bytes = (shift + want * 2u + 7u) >> 3;
                            simd::expand_codes2 (tile_source, tile, bytes, lut);
                        }
                    else
                        simd::expand_codes3 (tile_source, tile, want, lut);
                    const uint32_t skip = shift / bits;
                    for (uint32_t d = 0; d < want; ++d)
                        row[done + d] = _S_i8_code (tile[skip + d]);
                }
        }
    return Status::OK;
}

/** @brief Slice, concat, pad and flatten: pure arena-to-arena byte moves. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_shape_op (const Layer &layer, const int8_t *const input, const int8_t *const rhs,
             int8_t *const output) noexcept
{
    switch (layer._M_op)
        {
        case Op::FLATTEN:
            _S_copy_bytes (output, input, layer._M_in_dim);
            return Status::OK;
        case Op::SLICE:
            _S_copy_bytes (output, input + layer._M_kernel, layer._M_out_dim);
            return Status::OK;
        case Op::CONCAT:
            {
                const uint32_t second = _S_in2_len (layer);
                _S_copy_bytes (output, input, layer._M_in_dim);
                _S_copy_bytes (output + layer._M_in_dim, rhs, second);
                return Status::OK;
            }
        case Op::PAD1D:
            {
                const uint8_t fill = layer._M_aux;
                const uint32_t head = layer._M_kernel;
                const uint32_t tail = layer._M_in_dim + head;
                _S_fill_bytes (output, fill, head);
                _S_copy_bytes (output + head, input, layer._M_in_dim);
                if (layer._M_out_dim > tail)
                    _S_fill_bytes (output + tail, fill, layer._M_out_dim - tail);
                return Status::OK;
            }
        default:
            return Status::UNSUPPORTED;
        }
}

/** @brief Matrix product; the right operand is N rows of K codes in the arena. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_matmul (const Layer &layer, const int8_t *const input, const int8_t *const rhs,
           int8_t *const output) noexcept
{
    const uint32_t inner = layer._M_head_dim;
    if (inner == 0 || layer._M_in_dim % inner != 0)
        return Status::BAD_LAYOUT;
    const uint32_t rows = layer._M_in_dim / inner;
    const uint32_t rhs_length = _S_in2_len (layer);
    if (rhs_length == 0 || rhs_length % inner != 0)
        return Status::BAD_LAYOUT;
    const uint32_t columns = rhs_length / inner;
    if (rows * columns != layer._M_out_dim)
        return Status::BAD_LAYOUT;
    const int32_t aux = _S_act_param (layer);

    for (uint32_t row = 0; row < rows; ++row)
        {
            const int8_t *const a = input + row * inner;
            for (uint32_t column = 0; column < columns; ++column)
                {
                    const int32_t dot = simd::dot_i8 (a, rhs + column * inner, inner);
                    output[row * columns + column] = _S_i8_code (
                        _S_apply_act (layer._M_act,
                                      simd::requant_q15 (dot, layer._M_requant_mult), aux));
                }
        }
    return Status::OK;
}

/** @brief Row/column swap of a seq x head_dim arena tile. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_transpose2d (const Layer &layer, const int8_t *const input, int8_t *const output) noexcept
{
    const uint32_t rows = layer._M_seq;
    const uint32_t columns = layer._M_head_dim;
    if (rows == 0 || columns == 0 || rows * columns != layer._M_in_dim)
        return Status::BAD_LAYOUT;
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t column = 0; column < columns; ++column)
            output[column * rows + row] = input[row * columns + column];
    return Status::OK;
}

/** @brief Upsample with nearest neighbour: out[t] = in[t * in_len / out_len]. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_upsample (const Layer &layer, const int8_t *const input, int8_t *const output) noexcept
{
    const uint32_t in_length = layer._M_head_dim;
    const uint32_t out_length = layer._M_seq;
    if (in_length == 0 || out_length == 0)
        return Status::BAD_LAYOUT;
    const uint32_t channels = layer._M_in_dim / in_length;
    /* Nearest-neighbour source index is floor(pos * in / out).  Walking a
       Bresenham residue keeps the exact quotient without one hardware divide
       per output sample; the total carry work over the row is bounded by
       `in_length`, so the kernel stays linear. */
    for (uint32_t channel = 0; channel < channels; ++channel)
        {
            const int8_t *const source = input + channel * in_length;
            int8_t *const destination = output + channel * out_length;
            uint32_t index = 0;
            uint32_t residue = 0;
            for (uint32_t position = 0; position < out_length; ++position)
                {
                    destination[position] = source[index];
                    residue += in_length;
                    while (residue >= out_length)
                        {
                            residue -= out_length;
                            ++index;
                        }
                }
        }
    return Status::OK;
}

/** @brief Keep the input where the mask byte is zero, otherwise the fill code. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_masked_fill (const Layer &layer, const int8_t *const input, const int8_t *const mask,
                int8_t *const output) noexcept
{
    const int8_t fill = static_cast<int8_t> (layer._M_aux);
    simd::masked_fill_i8 (input, mask, output, layer._M_in_dim, fill);
    return Status::OK;
}

/**
 * @brief Inverted dropout with a deterministic xorshift32 mask stream.
 *
 * The seed is the 32-bit multiplier slot, so a given layer descriptor always
 * drops the same elements: the op stays bit-reproducible without a global RNG
 * and without reading anything the exporter did not put in the blob.
 */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_dropout (const Layer &layer, const int8_t *const input, int8_t *const output) noexcept
{
    const int32_t drop = _S_aux_param (layer);
    if (drop < 0)
        return Status::BAD_LAYOUT;
    const int32_t keep = 127 - drop;
    const int32_t keep_half = keep >> 1;
    const _S_Div32 keep_div = _S_div32_make (static_cast<uint32_t> (keep));
    uint32_t state = static_cast<uint32_t> (layer._M_requant_mult) | 1u;
    for (uint32_t i = 0; i < layer._M_in_dim; ++i)
        {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            if (static_cast<int32_t> ((state >> 17) & 0x7Fu) < drop || keep == 0)
                {
                    output[i] = 0;
                    continue;
                }
            const int32_t value = input[i];
            const int32_t scaled = value * 127;
            output[i] = _S_i8_code (
                _S_div32_i (scaled + (scaled >= 0 ? keep_half : -keep_half), keep_div));
        }
    return Status::OK;
}

/**
 * @brief Natural log of a softmax probability, in fixed-point units.
 *
 * The shared exp table decays by exp(-1/8) per index, so the table index is a
 * direct log2-scale measurement: one output code is 8/127 nats and -127 is the
 * floor for anything below exp(-8).
 */
[[nodiscard]] inline constexpr int32_t
_S_log_probability (const int32_t probability_q15) noexcept
{
    if (probability_q15 <= 0)
        return -127;
    uint32_t low = 0;
    uint32_t high = 255;
    while (low < high)
        {
            const uint32_t middle = (low + high + 1u) >> 1;
            if (_S_exp_table._M_value[middle] >= static_cast<uint32_t> (probability_q15))
                low = middle;
            else
                high = middle - 1u;
        }
    /* 64 = 2^6: the quarter-step rounding is a shift, never a divide. */
    const int32_t code = -static_cast<int32_t> ((low * 127u + 32u) >> 6);
    return code < -127 ? -127 : code;
}

/** @brief Log of the softmax over rows of _S_group_len elements. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_log_softmax (const Layer &layer, const int8_t *input, int8_t *output) noexcept
{
    const uint32_t width = _S_group_len (layer);
    if (width == 0 || layer._M_in_dim % width != 0)
        return Status::BAD_LAYOUT;
    for (uint32_t base = 0; base < layer._M_in_dim; base += width)
        {
            if (_S_softmax_row (input + base, output + base, width) == 0)
                return Status::INTERNAL;
            for (uint32_t i = 0; i < width; ++i)
                output[base + i] = static_cast<int8_t> (_S_log_probability (output[base + i] * 258));
        }
    return Status::OK;
}

/** @brief Sum / mean / max / min over the reduction groups. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_reduce (const Layer &layer, const int8_t *const input, int8_t *const output) noexcept
{
    const uint32_t window = _S_group_len (layer);
    if (window == 0 || layer._M_in_dim % window != 0)
        return Status::BAD_LAYOUT;
    const _S_Div32 window_div = _S_div32_make (window);
    for (uint32_t group = 0; group < layer._M_in_dim / window; ++group)
        {
            const int8_t *const row = input + group * window;
            switch (layer._M_aux)
                {
                case _S_reduce_sum:
                    output[group] = _S_i8_code (
                        simd::requant_q15 (simd::sum_i8 (row, window), layer._M_requant_mult));
                    break;
                case _S_reduce_mean:
                    output[group] = _S_i8_code (
                        _S_div32_i (simd::sum_i8 (row, window), window_div));
                    break;
                case _S_reduce_max:
                    output[group] = simd::max_i8 (row, window);
                    break;
                case _S_reduce_min:
                    {
                        int8_t minimum = 127;
                        for (uint32_t i = 0; i < window; ++i)
                            if (row[i] < minimum)
                                minimum = row[i];
                        output[group] = minimum;
                        break;
                    }
                default:
                    return Status::BAD_LAYOUT;
                }
        }
    return Status::OK;
}

/** @brief Index of the first maximum in each group, as a signed byte. */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_argmax (const Layer &layer, const int8_t *const input, int8_t *const output) noexcept
{
    const uint32_t window = _S_group_len (layer);
    if (window == 0 || window > 255 || layer._M_in_dim % window != 0)
        return Status::BAD_LAYOUT;
    for (uint32_t group = 0; group < layer._M_in_dim / window; ++group)
        {
            const int8_t *const row = input + group * window;
            /* One pass: the first strict improvement is exactly the first
               maximum `max_i8` would select, so the second scan disappears. */
            int8_t best = row[0];
            uint32_t index = 0;
            for (uint32_t i = 1; i < window; ++i)
                if (row[i] > best)
                    {
                        best = row[i];
                        index = i;
                    }
            output[group] = static_cast<int8_t> (index);
        }
    return Status::OK;
}

/**
 * @brief Run the parsed layer.
 *
 * @param arena the Core activation arena, used to locate second operands.
 */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_execute (Workspace &ws, const uint8_t *model, const int8_t *arena, const int8_t *input,
            int8_t *output) noexcept
{
    const Layer &layer = ws._M_layer;
    const bool residual_norm
        = layer._M_op == Op::NORM && (layer._M_flags & NORM_RESIDUAL) != 0;
    const int8_t *const rhs
        = (_S_two_operand (layer._M_op) || residual_norm)
              ? arena + _S_in2_off (layer)
              : nullptr;

    switch (layer._M_op)
        {
        case Op::LINEAR:
            return _S_linear (ws, model, input, output);
        case Op::CONV1D:
            return _S_conv1d (ws, model, input, output);
        case Op::NORM:
            return _S_norm (layer, model, input, residual_norm ? rhs : nullptr, output);
        case Op::ACTIVATION:
            return _S_activation (layer, input, output);
        case Op::SOFTMAX:
            return _S_softmax (layer, input, output);
        case Op::ATTENTION:
            return _S_attention (layer, input, output);
        case Op::ADD:
        case Op::SUB:
        case Op::MUL:
            return _S_binary (layer, input, rhs, output);
        case Op::CLAMP:
            return _S_clamp (layer, input, output);
        case Op::PRELU:
            return _S_prelu (layer, model, input, output);
        case Op::BATCH_NORM:
            return _S_batch_norm (layer, model, input, output);
        case Op::GROUP_NORM:
            return _S_group_norm (layer, model, input, output);
        case Op::L2_NORMALIZE:
            return _S_l2_normalize (layer, input, output);
        case Op::MAX_POOL1D:
        case Op::AVG_POOL1D:
        case Op::GLOBAL_AVG_POOL:
            return _S_pool (layer, input, output);
        case Op::EMBEDDING:
            return _S_embedding (layer, model, input, output);
        case Op::CONCAT:
        case Op::SLICE:
        case Op::PAD1D:
        case Op::FLATTEN:
            return _S_shape_op (layer, input, rhs, output);
        case Op::MATMUL:
            return _S_matmul (layer, input, rhs, output);
        case Op::TRANSPOSE2D:
            return _S_transpose2d (layer, input, output);
        case Op::UPSAMPLE_NEAREST:
            return _S_upsample (layer, input, output);
        case Op::MASKED_FILL:
            return _S_masked_fill (layer, input, rhs, output);
        case Op::DROPOUT:
            return _S_dropout (layer, input, output);
        case Op::LOG_SOFTMAX:
            return _S_log_softmax (layer, input, output);
        case Op::REDUCE:
            return _S_reduce (layer, input, output);
        case Op::ARGMAX:
            return _S_argmax (layer, input, output);
        case Op::COUNT:
            break;
        }
    return Status::UNSUPPORTED;
}
} // namespace zzs::mllee
