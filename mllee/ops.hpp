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

#include "mllee/format.hpp"
#include "mllee/packlow.hpp"
#include "mllee/quant.hpp"
#include "mllee/simd.hpp"
#include "mllee/status.hpp"

#include <cstdint>

namespace zzs::mllee
{
/**
 * @brief Writable execution state owned by Core.
 *
 * Keeping the header, the active layer and the codec state here (instead of in
 * local variables) is what keeps the core call chain inside the stack budget:
 * no op ever copies a 48-byte descriptor or a reader struct onto the stack.
 */
struct Workspace
{
    PacklowState _M_pk{};
    Header _M_header{};
    Layer _M_layer{};
};

[[nodiscard]] inline constexpr int32_t
_S_tanh_q15 (const int32_t x_q15) noexcept
{
    const int64_t x = x_q15;
    const int64_t x2 = (x * x) >> 15;
    const int64_t numerator = x * ((27LL << 15) + x2);
    const int64_t denominator = (27LL << 15) + 9 * x2;
    return static_cast<int32_t> (numerator / denominator);
}

struct _S_ActTable
{
    int8_t _M_tanh[256];
    int8_t _M_sigmoid[256];

    constexpr _S_ActTable () noexcept : _M_tanh (), _M_sigmoid ()
    {
        for (int code = 0; code < 256; ++code)
            {
                const int32_t input = (code - 127) * 258;
                const int32_t clamped = input < -32767 ? -32767 : (input > 32767 ? 32767 : input);
                const int32_t tanh = _S_tanh_q15 (clamped);
                this->_M_tanh[code] = _S_i8_code ((tanh * 127 + 16384) >> 15);
                const int32_t sigmoid_input = clamped / 2;
                const int32_t sigmoid = _S_tanh_q15 (sigmoid_input);
                this->_M_sigmoid[code] = static_cast<int8_t> (
                    ((sigmoid + 32768) * 127 + 32768) >> 16);
            }
    }
};

struct _S_ExpTable
{
    uint16_t _M_value[256];

    constexpr _S_ExpTable () noexcept : _M_value ()
    {
        uint32_t value = 32767;
        for (int i = 0; i < 256; ++i)
            {
                this->_M_value[i] = static_cast<uint16_t> (value);
                value = (value * 28919u + 16384u) >> 15;
                if (value < 1)
                    value = 1;
            }
    }
};

inline constexpr _S_ActTable _S_act_table{};
inline constexpr _S_ExpTable _S_exp_table{};

/** @brief Ratio of one exp-table step, i.e. exp(-_S_exp_step) per index. */
inline constexpr int32_t _S_exp_step_inverse = 8;

[[nodiscard]] inline constexpr int32_t
_S_table_index (const int32_t value) noexcept
{
    const int32_t index = value + 127;
    return index < 0 ? 0 : (index > 255 ? 255 : index);
}

[[nodiscard]] inline constexpr int32_t
_S_sigmoid_i8 (const int32_t value) noexcept
{
    return _S_act_table._M_sigmoid[_S_table_index (value)];
}

[[nodiscard]] inline constexpr int32_t
_S_tanh_i8 (const int32_t value) noexcept
{
    return _S_act_table._M_tanh[_S_table_index (value)];
}

/**
 * @brief Hard sigmoid: clamp((x + 1) / 2, 0, 1) with x = value / 127.
 *
 * The activation tables cover x in [-1, 1], so the PyTorch "x + 3, clamped to
 * [0, 6]" range is expressed in that same normalized domain.
 */
[[nodiscard]] inline constexpr int32_t
_S_hard_sigmoid (const int32_t value) noexcept
{
    const int32_t half = (value + 127) / 2;
    return half < 0 ? 0 : (half > 127 ? 127 : half);
}

/** @brief exp(-value / 127) as a Q15 value, from the shared exp table. */
[[nodiscard]] inline constexpr int32_t
_S_exp_q15 (const int32_t neg_value) noexcept
{
    int32_t index = (neg_value * _S_exp_step_inverse + 63) / 127;
    if (index < 0)
        index = 0;
    if (index > 255)
        index = 255;
    return _S_exp_table._M_value[static_cast<uint32_t> (index)];
}

/**
 * @brief Elementwise activation, in int8 code units.
 *
 * @param aux the layer aux byte read as a signed code: the PReLU-family slope
 *            (LEAKY_RELU) or the ELU alpha.  Unused by the other acts.
 */
[[nodiscard]] inline constexpr int32_t
_S_apply_act (const Act act, const int32_t value, const int32_t aux) noexcept
{
    switch (act)
        {
        case Act::RELU:
            return value < 0 ? 0 : value;
        case Act::RELU6:
            return value < 0 ? 0 : (value > 127 ? 127 : value);
        case Act::SIGMOID:
            return _S_sigmoid_i8 (value);
        case Act::TANH:
            return _S_tanh_i8 (value);
        case Act::GELU:
            {
                int32_t index = 127 + (value * 1702) / 1000;
                index = index < 0 ? 0 : (index > 255 ? 255 : index);
                const int32_t sigmoid = _S_act_table._M_sigmoid[index];
                return (value * sigmoid + 63) / 127;
            }
        case Act::LEAKY_RELU:
            return value < 0 ? value * aux / 127 : value;
        case Act::ELU:
            return value < 0
                       ? -_S_round_shift (aux * (32768 - _S_exp_q15 (-value)), 15u)
                       : value;
        case Act::SILU:
            return value * _S_sigmoid_i8 (value) / 127;
        case Act::HARDSWISH:
            return value * _S_hard_sigmoid (value) / 127;
        case Act::HARDSIGMOID:
            return _S_hard_sigmoid (value);
        case Act::NONE:
        default:
            return value;
        }
}

[[nodiscard]] inline constexpr int32_t
_S_apply_act (const Act act, const int32_t value) noexcept
{
    return _S_apply_act (act, value, 0);
}

[[nodiscard]] inline constexpr int32_t
_S_scale_multiplier (const uint8_t *model, const Layer &layer, const uint32_t index) noexcept
{
    if (layer._M_scale_off == 0)
        return layer._M_requant_mult;
    return _S_i32 (model + layer._M_scale_off + index * 4u);
}

[[nodiscard]] inline constexpr int32_t
_S_bias_value (const uint8_t *model, const Layer &layer, const uint32_t index) noexcept
{
    if (layer._M_bias_off == 0)
        return 0;
    return _S_i16 (model + layer._M_bias_off + index * 2u);
}

[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_linear (Workspace &ws, const uint8_t *model, const int8_t *input,
           int8_t *output) noexcept
{
    const Layer &layer = ws._M_layer;
    PacklowState &pk = ws._M_pk;
    const uint8_t *const dict =
        ws._M_header._M_dict_len != 0 ? model + ws._M_header._M_dict_off : nullptr;
    const uint16_t dict_len = ws._M_header._M_dict_len;
    const uint32_t groups = layer._M_group_size != 0 ? layer._M_groups : 1u;
    const uint32_t group_size =
        layer._M_group_size != 0 ? layer._M_group_size : layer._M_in_dim;
    const int32_t aux = _S_act_param (layer);

    for (uint32_t out_index = 0; out_index < layer._M_out_dim; ++out_index)
        {
            int32_t total = 0;
            for (uint32_t group = 0; group < groups; ++group)
                {
                    const uint32_t base = group * group_size;
                    const uint32_t count = base < layer._M_in_dim ? layer._M_in_dim - base : 0u;
                    const uint32_t bounded = count < group_size ? count : group_size;
                    int32_t accumulator = 0;
                    for (uint32_t k = 0; k < bounded; ++k)
                        {
                            const int32_t weight = _S_rd_code (pk, model, dict, dict_len);
                            if (weight == _S_pk_error)
                                return _S_pk_status (pk);
                            accumulator += weight * static_cast<int32_t> (input[base + k]);
                        }
                    total += simd::requant_q15 (
                        accumulator,
                        _S_scale_multiplier (model, layer, out_index * groups + group));
                }
            total += _S_bias_value (model, layer, out_index);
            output[out_index] = _S_i8_code (_S_apply_act (layer._M_act, total, aux));
        }

    if (_S_pk_status (pk) != Status::OK)
        return _S_pk_status (pk);
    if (pk._M_remaining != 0)
        return Status::BAD_LAYOUT;
    return Status::OK;
}

[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_conv1d (Workspace &ws, const uint8_t *model, const int8_t *input,
           int8_t *output) noexcept
{
    const Layer &layer = ws._M_layer;
    PacklowState &pk = ws._M_pk;
    if (layer._M_head_dim == 0 || layer._M_seq == 0 || layer._M_in_dim % layer._M_head_dim != 0
        || layer._M_out_dim % layer._M_seq != 0)
        return Status::BAD_LAYOUT;

    const uint32_t input_length = layer._M_head_dim;
    const uint32_t output_length = layer._M_seq;
    const uint32_t input_channels = layer._M_in_dim / input_length;
    const uint32_t output_channels = layer._M_out_dim / output_length;
    const bool depthwise = (layer._M_flags & CONV_DEPTHWISE) != 0;
    if (depthwise && output_channels != input_channels)
        return Status::BAD_LAYOUT;
    const uint32_t tap_channels = depthwise ? 1u : input_channels;
    const uint32_t dilation = _S_dilation (layer);
    const uint32_t row_codes = tap_channels * layer._M_kernel;
    const bool packed = layer._M_dtype == Dtype::I4 || layer._M_dtype == Dtype::F4E2M1;
    if (packed && (row_codes & 1u) != 0)
        return Status::UNSUPPORTED;
    const uint32_t row_bytes = packed ? row_codes / 2u : row_codes;
    if (row_bytes == 0 || row_bytes > _S_pk_window_bytes)
        return Status::UNSUPPORTED;

    const uint32_t padding
        = (layer._M_flags & CONV_CAUSAL) != 0 ? (static_cast<uint32_t> (layer._M_kernel) - 1u)
                                                    * dilation
                                             : 0u;
    const uint32_t groups = layer._M_group_size != 0 ? layer._M_groups : 1u;
    const uint32_t group_size =
        layer._M_group_size != 0 ? layer._M_group_size : row_codes;
    const uint8_t *const dict =
        ws._M_header._M_dict_len != 0 ? model + ws._M_header._M_dict_off : nullptr;
    const uint16_t dict_len = ws._M_header._M_dict_len;
    const int32_t aux = _S_act_param (layer);

    for (uint32_t out_channel = 0; out_channel < output_channels; ++out_channel)
        {
            for (uint32_t i = 0; i < row_bytes; ++i)
                {
                    const int32_t byte = _S_rd_byte (pk, model, dict, dict_len);
                    if (byte == _S_pk_error)
                        return _S_pk_status (pk);
                }
            for (uint32_t position = 0; position < output_length; ++position)
                {
                    int32_t total = 0;
                    for (uint32_t group = 0; group < groups; ++group)
                        {
                            const uint32_t group_base = group * group_size;
                            const uint32_t group_end = group_base + group_size;
                            int32_t accumulator = 0;
                            for (uint32_t channel = 0; channel < tap_channels; ++channel)
                                {
                                    const uint32_t source_channel
                                        = depthwise ? out_channel : channel;
                                    for (uint32_t tap = 0; tap < layer._M_kernel; ++tap)
                                        {
                                            const uint32_t code_index
                                                = channel * layer._M_kernel + tap;
                                            if (code_index < group_base
                                                || code_index >= group_end)
                                                continue;
                                            const uint32_t source_index
                                                = position * layer._M_stride + tap * dilation;
                                            if (source_index < padding
                                                || source_index - padding >= input_length)
                                                continue;
                                            const int32_t weight
                                                = _S_rd_staged (pk, code_index, row_bytes);
                                            const int32_t value
                                                = input[source_channel * input_length
                                                        + source_index - padding];
                                            accumulator += weight * value;
                                        }
                                }
                            total += simd::requant_q15 (
                                accumulator,
                                _S_scale_multiplier (model, layer, out_channel * groups + group));
                        }
                    total += _S_bias_value (model, layer, out_channel);
                    output[out_channel * output_length + position]
                        = _S_i8_code (_S_apply_act (layer._M_act, total, aux));
                }
        }

    if (_S_pk_status (pk) != Status::OK)
        return _S_pk_status (pk);
    if (pk._M_remaining != 0)
        return Status::BAD_LAYOUT;
    return Status::OK;
}

/**
 * @brief Mean and standard deviation of one int8 window, in integer units.
 *
 * Shared by LayerNorm, RMSNorm, instance/group/batch normalization.  The
 * variance is exact (int64) and the root is the floor integer square root, so
 * the result never depends on the host floating point environment.
 */
inline constexpr void
_S_window_stats (const int8_t *const values, const uint32_t count, const bool rms,
                 int32_t &mean, uint32_t &deviation) noexcept
{
    int32_t sum_codes = 0;
    int32_t square_codes = 0;
    simd::stats_i8 (values, count, sum_codes, square_codes);
    const int64_t sum = sum_codes;
    const int64_t square_sum = square_codes;
    mean = static_cast<int32_t> (sum / static_cast<int64_t> (count));
    int64_t variance;
    if (rms)
        variance = square_sum / static_cast<int64_t> (count);
    else
        {
            const int64_t numerator = square_sum * static_cast<int64_t> (count) - sum * sum;
            const int64_t denominator
                = static_cast<int64_t> (count) * static_cast<int64_t> (count);
            variance = numerator > 0 ? numerator / denominator : 0;
        }
    deviation = variance > 0 ? _S_isqrt (static_cast<uint32_t> (variance)) : 0u;
}

/** @brief One normalized, gamma/beta-scaled output code (256x fixed point). */
[[nodiscard]] inline constexpr int32_t
_S_norm_value (const int32_t value, const int32_t mean, const uint32_t deviation, const bool rms,
               const int32_t gamma_q7, const int32_t multiplier, const int32_t beta) noexcept
{
    const int32_t centered = rms ? value : value - mean;
    const int32_t normalized
        = deviation != 0 ? (centered * 256) / static_cast<int32_t> (deviation) : 0;
    const int32_t scaled = _S_round_shift (static_cast<int64_t> (normalized) * gamma_q7, 7u);
    const int32_t requantized = _S_round_shift (simd::requant_q15 (scaled, multiplier), 8u);
    return requantized + beta;
}

[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_norm (const Layer &layer, const uint8_t *model, const int8_t *input,
         int8_t *output) noexcept
{
    const uint32_t count = layer._M_in_dim;
    if (count == 0)
        return Status::BAD_LAYOUT;
    const bool rms = (layer._M_flags & NORM_RMS) != 0;
    const bool instance = (layer._M_flags & NORM_INSTANCE) != 0;
    const uint32_t window = instance && layer._M_group_size != 0 ? layer._M_group_size : count;
    if (window == 0 || count % window != 0)
        return Status::BAD_LAYOUT;
    const int32_t aux = _S_act_param (layer);

    for (uint32_t base = 0; base < count; base += window)
        {
            int32_t mean = 0;
            uint32_t deviation = 0;
            _S_window_stats (input + base, window, rms, mean, deviation);
            for (uint32_t i = 0; i < window; ++i)
                {
                    const uint32_t index = base + i;
                    const int32_t gamma = layer._M_scale_off != 0
                                              ? _S_i16 (model + layer._M_scale_off + index * 2u)
                                              : 128;
                    const int32_t beta = _S_bias_value (model, layer, index);
                    output[index] = _S_i8_code (
                        _S_apply_act (layer._M_act,
                                      _S_norm_value (input[index], mean, deviation, rms, gamma,
                                                     layer._M_requant_mult, beta),
                                      aux));
                }
        }
    return Status::OK;
}

[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_activation (const Layer &layer, const int8_t *input, int8_t *output) noexcept
{
    const uint32_t count = layer._M_out_dim < layer._M_in_dim ? layer._M_out_dim : layer._M_in_dim;
    const int32_t aux = _S_act_param (layer);
    for (uint32_t i = 0; i < count; ++i)
        output[i] = _S_i8_code (_S_apply_act (layer._M_act, input[i], aux));
    return Status::OK;
}

/**
 * @brief One softmax row: exp(max-shifted) codes normalized to sum to 127.
 *
 * @return the integer row sum, or 0 for an empty row (the caller maps that to
 *         INTERNAL) and writes the row itself through output.
 */
[[nodiscard]] inline int32_t
_S_softmax_row (const int8_t *const input, int8_t *const output, const uint32_t count) noexcept
{
    if (count == 0)
        return 0;
    const int32_t maximum = simd::max_i8 (input, count);
    int64_t sum = 0;
    for (uint32_t i = 0; i < count; ++i)
        {
            uint32_t distance = static_cast<uint32_t> (maximum - input[i]);
            if (distance > 255)
                distance = 255;
            sum += _S_exp_table._M_value[distance];
        }
    if (sum == 0)
        return 0;
    for (uint32_t i = 0; i < count; ++i)
        {
            uint32_t distance = static_cast<uint32_t> (maximum - input[i]);
            if (distance > 255)
                distance = 255;
            const int64_t numerator
                = static_cast<int64_t> (_S_exp_table._M_value[distance]) * 127 + sum / 2;
            output[i] = static_cast<int8_t> (numerator / sum);
        }
    return static_cast<int32_t> (sum);
}

/** @brief Softmax over rows of _S_group_len elements (a single row when unset). */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_softmax (const Layer &layer, const int8_t *input, int8_t *output) noexcept
{
    const uint32_t width = _S_group_len (layer);
    if (width == 0 || layer._M_in_dim % width != 0)
        return Status::BAD_LAYOUT;
    for (uint32_t base = 0; base < layer._M_in_dim; base += width)
        if (_S_softmax_row (input + base, output + base, width) == 0)
            return Status::INTERNAL;
    return Status::OK;
}

[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_attention (const Layer &layer, const int8_t *input, int8_t *output) noexcept
{
    const uint32_t sequence = layer._M_seq;
    const uint32_t head_dim = layer._M_head_dim;
    const uint32_t heads = _S_heads (layer);
    if (sequence == 0 || head_dim == 0 || head_dim > 8)
        return Status::UNSUPPORTED;
    const uint32_t stride = head_dim * heads;
    const bool causal = (layer._M_flags & ATTN_CAUSAL) != 0;
    const int32_t aux = _S_act_param (layer);

    for (uint32_t head = 0; head < heads; ++head)
        {
            const int8_t *const queries = input + head * head_dim;
            const int8_t *const keys = queries + sequence * stride;
            const int8_t *const values = keys + sequence * stride;
            const uint32_t score_scale = _S_isqrt (head_dim << 16);

            for (uint32_t query = 0; query < sequence; ++query)
                {
                    const int8_t *const query_row = queries + query * stride;
                    const uint32_t limit = causal ? query + 1u : sequence;
                    int32_t maximum = -2147483647;
                    for (uint32_t key = 0; key < limit; ++key)
                        {
                            const int32_t score
                                = simd::dot_i8 (query_row, keys + key * stride, head_dim);
                            if (score > maximum)
                                maximum = score;
                        }

                    int32_t accumulator[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                    int32_t denominator = 0;
                    for (uint32_t key = 0; key < limit; ++key)
                        {
                            const int32_t score
                                = simd::dot_i8 (query_row, keys + key * stride, head_dim);
                            uint32_t distance = static_cast<uint32_t> (maximum - score);
                            distance = (distance << 11) / score_scale;
                            if (distance > 255)
                                distance = 255;
                            const int32_t weight = _S_exp_table._M_value[distance];
                            denominator += weight;
                            const int8_t *const value_row = values + key * stride;
                            for (uint32_t d = 0; d < head_dim; ++d)
                                accumulator[d] += weight * value_row[d];
                        }
                    if (denominator == 0)
                        return Status::INTERNAL;
                    for (uint32_t d = 0; d < head_dim; ++d)
                        output[query * stride + head * head_dim + d] = _S_i8_code (
                            _S_apply_act (layer._M_act,
                          (accumulator[d] + denominator / 2) / denominator, aux));
                }
        }
    return Status::OK;
}
} // namespace zzs::mllee
