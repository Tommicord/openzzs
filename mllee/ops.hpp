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

#include "mllee/arith.hpp"
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
    /* value + 127 is in [0, 254], so the halving is a plain arithmetic shift:
       the truncation-toward-zero of `/ 2` and the floor of `>> 1` agree on the
       whole reachable range. */
    const int32_t half = (value + 127) >> 1;
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

/**
 * @brief Multiply a run of packed codes by the paired input codes.
 *
 * One dispatch covers every sub-byte format: two codes per byte (I4/F4E2M1),
 * four per byte (I2) and the bit-packed 3-bit groups (I3/F3E1M1).  `codes` is a
 * whole number of packing groups; the AVX2 kernels expand the codes in
 * registers and never touch a scratch buffer.
 */
[[nodiscard]] inline int32_t
_S_packed_dot (const uint8_t *const packed, const int8_t *const input,
               const uint32_t codes, const uint32_t bits,
               const int8_t *const lut) noexcept
{
    switch (bits)
        {
        case 2u:
            return simd::dot_i2 (packed, input, codes, lut);
        case 4u:
            return simd::dot_i4 (packed, input, codes, lut);
        default:
            return simd::dot_i3 (packed, input, codes, lut);
        }
}

/**
 * @brief Dot product over a byte-code row range: `[lo, hi)` of one weight row.
 *
 * I8 codes without a Flash table are their own bytes, so a contiguous span goes
 * straight to `simd::dot_i8` in the runs the ring wrap leaves.  A float8 table
 * has no shuffle form, so that layer keeps the scalar LUT gather.
 *
 * Out of line on purpose: the kernel is large (two ring runs, a LUT form and the
 * SIMD call), and inlining it into `_S_conv1d` would grow that frame past the
 * module's stack budget.  The dispatcher `_S_rd_row_dot` is the small part.
 */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline int32_t
_S_rd_row_dot_i8 (const PacklowState &pk, const uint8_t *const model,
                  const int8_t *const input_at_lo, const uint32_t lo, const uint32_t hi,
                  const uint32_t row_bytes) noexcept
{
    const int8_t *const lut = pk._M_lut;
    if (lut != nullptr)
        {
            int32_t total = 0;
            for (uint32_t code = lo; code < hi; ++code)
                total += lut[_S_rd_row_byte (pk, model, code, row_bytes)]
                         * static_cast<int32_t> (input_at_lo[code - lo]);
            return total;
        }
    if ((pk._M_flags & _S_pk_raw) != 0)
        return simd::dot_i8 (
            reinterpret_cast<const int8_t *> (model + pk._M_src_off - row_bytes + lo),
            input_at_lo, hi - lo);
    const uint32_t start = _S_rd_staged_start (pk, row_bytes);
    const uint32_t head = _S_rd_row_head (pk, row_bytes);
    uint32_t done = lo;
    int32_t total = 0;
    if (done < head)
        {
            const uint32_t count = hi < head ? hi - done : head - done;
            /* `done < head` keeps [start + done, start + head) inside the
               window, so the head run never wraps. */
            total += simd::dot_i8 (
                reinterpret_cast<const int8_t *> (&pk._M_ring[start + done]), input_at_lo,
                count);
            done += count;
        }
    if (done < hi)
        total += simd::dot_i8 (reinterpret_cast<const int8_t *> (pk._M_ring + (done - head)),
                               input_at_lo + (done - lo), hi - done);
    return total;
}

/**
 * @brief Dot product over a raw (uncompressed) sub-byte weight row.
 *
 * A RAW row already lies contiguous in Flash, so the packed kernels expand and
 * multiply whole packing groups straight from the blob.  The leading partial
 * group and the 3-bit tail use the bit-exact reader.
 *
 * Out of line for the stack budget: the packed kernels are large and inlining
 * them into `_S_linear` would push the module past its 384-byte frame ceiling.
 * A PACKLOW-compressed sub-byte row is decoded through the 64-byte ring by the
 * caller's scalar reader instead; that keeps the deepest compressed row chain
 * well below the raw one.
 */
[[nodiscard]] ZZS_MLLEE_NOINLINE inline int32_t
_S_rd_row_dot_packed (const PacklowState &pk, const uint8_t *const model,
                      const int8_t *const input_at_lo, const uint32_t lo, const uint32_t hi,
                      const uint32_t row_bytes) noexcept
{
    const Dtype dtype = static_cast<Dtype> (pk._M_dtype);
    const uint32_t bits = _S_code_bits (dtype);
    const int8_t *const lut = pk._M_lut;
    /* 2/4-bit: one byte per group; 3-bit: 8 codes per 3-byte group. */
    const uint32_t group_codes
        = _S_byte_divisible (dtype) ? _S_codes_per_byte (dtype) : 8u;
    uint32_t code = lo;
    int32_t total = 0;
    while (code < hi && (code % group_codes) != 0u)
        {
            total += _S_rd_row_code (pk, model, code, row_bytes)
                     * static_cast<int32_t> (input_at_lo[code - lo]);
            ++code;
        }
    if (code < hi)
        {
            const uint32_t middle = ((hi - code) / group_codes) * group_codes;
            if (middle != 0u)
                {
                    total += _S_packed_dot (
                        model + pk._M_src_off - row_bytes + code * bits / 8u,
                        input_at_lo + (code - lo), middle, bits, lut);
                    code += middle;
                }
            for (; code < hi; ++code)
                total += _S_rd_row_code (pk, model, code, row_bytes)
                         * static_cast<int32_t> (input_at_lo[code - lo]);
        }
    return total;
}

/**
 * @brief Dot product over the code range [lo, hi) of the staged weight row.
 *
 * `input_at_lo` is the operand element paired with code `lo`, so a caller whose
 * window starts inside its row never forms an out-of-range pointer.  This
 * dispatcher stays inline (a width test and a call); the two bodies are
 * out of line to keep the conv/norm frames inside the stack budget.
 */
[[nodiscard]] ZZS_MLLEE_INLINE int32_t
_S_rd_row_dot (const PacklowState &pk, const uint8_t *const model,
               const int8_t *const input_at_lo, const uint32_t lo, const uint32_t hi,
               const uint32_t row_bytes) noexcept
{
    if (hi <= lo)
        return 0;
    if (_S_code_bits (static_cast<Dtype> (pk._M_dtype)) == 8u)
        return _S_rd_row_dot_i8 (pk, model, input_at_lo, lo, hi, row_bytes);
    /* The packed kernels read a contiguous RAW row.  A compressed row lives in
       the ring by the time this runs, so the scalar reader is the safe form if
       a future caller reaches here without the RAW check. */
    if ((pk._M_flags & _S_pk_raw) == 0)
        {
            int32_t total = 0;
            for (uint32_t code = lo; code < hi; ++code)
                total += _S_rd_row_code (pk, model, code, row_bytes)
                         * static_cast<int32_t> (input_at_lo[code - lo]);
            return total;
        }
    return _S_rd_row_dot_packed (pk, model, input_at_lo, lo, hi, row_bytes);
}

/**
 * @brief One LINEAR weight group, staged then accumulated.
 *
 * `codes` codes are read as weight rows of at most one ring window: a raw
 * stream advances its Flash cursor, a packlow stream decodes into the ring, and
 * the multiply-accumulate then runs over the staged row instead of pulling one
 * range-coded byte at a time inside the MAC loop.
 */
[[nodiscard]] ZZS_MLLEE_INLINE Status
_S_linear_row (PacklowState &pk, const uint8_t *model, const uint8_t *dict,
               const uint16_t dict_len, const uint32_t codes, const int8_t *const input,
               const bool vector_mac, int32_t &accumulator) noexcept
{
    const Dtype dtype = static_cast<Dtype> (pk._M_dtype);
    const uint32_t bits = _S_code_bits (dtype);
    /* One staging window is a whole number of packing groups: 64 bytes for the
       byte-divisible formats, 63 bytes (168 codes) for bit-packed 3-bit. */
    const uint32_t group_codes
        = _S_byte_divisible (dtype) ? _S_codes_per_byte (dtype) : 8u;
    const uint32_t group_bytes = group_codes * bits / 8u;
    const uint32_t window_codes = (_S_pk_window_bytes / group_bytes) * group_codes;
    const uint32_t run = (pk._M_flags & _S_pk_raw) != 0 ? codes : window_codes;
    for (uint32_t done = 0; done < codes; done += run)
        {
            const uint32_t tile_codes = codes - done < run ? codes - done : run;
            const uint32_t tile_bytes = tile_codes / group_codes * group_bytes;
            const Status staged = _S_rd_row (pk, model, dict, dict_len, tile_bytes);
            if (staged != Status::OK)
                return staged;
            if (vector_mac)
                {
                    accumulator += _S_rd_row_dot (pk, model, input + done, 0u, tile_codes,
                                                  tile_bytes);
                    continue;
                }
            for (uint32_t i = 0; i < tile_codes; ++i)
                accumulator += _S_rd_row_code (pk, model, i, tile_bytes)
                               * static_cast<int32_t> (input[done + i]);
        }
    return Status::OK;
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
    /* Staging works a whole packing group at a time: one byte for the 1/2/4-bit
       formats that divide a byte, three bytes (eight codes) for bit-packed
       3-bit.  A group that crosses the input or a group boundary keeps the
       code-at-a-time reader, which carries the leftover bits across groups. */
    const Dtype dtype = static_cast<Dtype> (pk._M_dtype);
    const uint32_t bits = _S_code_bits (dtype);
    const uint32_t group_codes
        = _S_byte_divisible (dtype) ? _S_codes_per_byte (dtype) : 8u;
    const bool tileable
        = layer._M_in_dim % group_codes == 0u && group_size % group_codes == 0u;
    /* Sub-byte integer/float tables are expanded by the packed kernels straight
       from a RAW row; a PACKLOW row goes through the ring and the scalar reader,
       which keeps the compressed-row call chain inside the stack budget.  A
       byte-wide float8 table has no shuffle form, so it keeps the scalar LUT. */
    const bool vector_mac = tileable
                            && (bits == 8u ? pk._M_lut == nullptr
                                           : (pk._M_flags & _S_pk_raw) != 0);

    for (uint32_t out_index = 0; out_index < layer._M_out_dim; ++out_index)
        {
            int32_t total = 0;
            for (uint32_t group = 0; group < groups; ++group)
                {
                    const uint32_t base = group * group_size;
                    const uint32_t count = base < layer._M_in_dim ? layer._M_in_dim - base : 0u;
                    const uint32_t bounded = count < group_size ? count : group_size;
                    int32_t accumulator = 0;
                    if (tileable)
                        {
                            const Status filled = _S_linear_row (
                                pk, model, dict, dict_len, bounded, input + base, vector_mac,
                                accumulator);
                            if (filled != Status::OK)
                                return filled;
                        }
                    else
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
    const Dtype dtype = static_cast<Dtype> (pk._M_dtype);
    const uint32_t bits = _S_code_bits (dtype);
    const uint32_t group_codes
        = _S_byte_divisible (dtype) ? _S_codes_per_byte (dtype) : 8u;
    const uint32_t group_bytes = group_codes * bits / 8u;
    /* One row is staged as whole packing groups, so its codes have to fill
       them: 8 codes per 3-byte group for the bit-packed 3-bit formats. */
    if (row_codes % group_codes != 0u)
        return Status::UNSUPPORTED;
    const uint32_t row_bytes = row_codes / group_codes * group_bytes;
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
    /* With unit stride and dilation the taps of one output position pair with a
       contiguous input window, so an int8 row without a LUT is one dot product
       per channel instead of a code-at-a-time staged read. */
    const bool vector_mac = bits == 8u && pk._M_lut == nullptr
                            && layer._M_stride == 1u && dilation == 1u;
    for (uint32_t out_channel = 0; out_channel < output_channels; ++out_channel)
        {
            const Status staged = _S_rd_row (pk, model, dict, dict_len, row_bytes);
            if (staged != Status::OK)
                return staged;
            for (uint32_t position = 0; position < output_length; ++position)
                {
                    int32_t total = 0;
                    uint32_t tap_lo = 0;
                    uint32_t tap_hi = layer._M_kernel;
                    if (vector_mac)
                        {
                            tap_lo = padding > position ? padding - position : 0u;
                            if (tap_lo > layer._M_kernel)
                                tap_lo = layer._M_kernel;
                            tap_hi = position < input_length + padding
                                         ? input_length + padding - position
                                         : 0u;
                            if (tap_hi > layer._M_kernel)
                                tap_hi = layer._M_kernel;
                        }
                    for (uint32_t group = 0; group < groups; ++group)
                        {
                            const uint32_t group_base = group * group_size;
                            const uint32_t group_end = group_base + group_size;
                            int32_t accumulator = 0;
                            for (uint32_t channel = 0; channel < tap_channels; ++channel)
                                {
                                    const uint32_t source_channel
                                        = depthwise ? out_channel : channel;
                                    const int8_t *const row
                                        = input + source_channel * input_length;
                                    const uint32_t code_base = channel * layer._M_kernel;
                                    if (vector_mac)
                                        {
                                            uint32_t lo = group_base > code_base + tap_lo
                                                              ? group_base
                                                              : code_base + tap_lo;
                                            uint32_t hi = group_end < code_base + tap_hi
                                                              ? group_end
                                                              : code_base + tap_hi;
                                            if (hi > lo)
                                                accumulator += _S_rd_row_dot (
                                                    pk, model,
                                                    row + static_cast<int32_t> (lo - code_base)
                                                          - static_cast<int32_t> (padding)
                                                          + static_cast<int32_t> (position),
                                                    lo, hi, row_bytes);
                                            continue;
                                        }
                                    for (uint32_t tap = 0; tap < layer._M_kernel; ++tap)
                                        {
                                            const uint32_t code_index = code_base + tap;
                                            if (code_index < group_base
                                                || code_index >= group_end)
                                                continue;
                                            const uint32_t source_index
                                                = position * layer._M_stride + tap * dilation;
                                            if (source_index < padding
                                                || source_index - padding >= input_length)
                                                continue;
                                            const int32_t value
                                                = row[source_index - padding];
                                            accumulator
                                                += _S_rd_row_code (pk, model, code_index, row_bytes)
                                                   * value;
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
 *
 * The two 32-bit quotients use the reciprocal built once per window: the
 * window length is invariant across every output element, so the hardware
 * divide moves out of the kernel and the per-element cost becomes a multiply.
 * The variance quotient is 64-bit and still runs once per window.
 */
inline constexpr void
_S_window_finish (const int64_t sum, const int64_t square_sum,
                  const uint32_t count, const bool rms, int32_t &mean,
                  uint32_t &deviation) noexcept
{
    const _S_Div32 count_div = _S_div32_make (count);
    mean = _S_div32_i (static_cast<int32_t> (sum), count_div);
    int64_t variance;
    if (rms)
        variance = _S_div32_u (static_cast<uint32_t> (square_sum), count_div);
    else
        {
            const int64_t numerator = square_sum * static_cast<int64_t> (count) - sum * sum;
            const int64_t denominator
                = static_cast<int64_t> (count) * static_cast<int64_t> (count);
            variance = numerator > 0 ? numerator / denominator : 0;
        }
    deviation = variance > 0 ? _S_isqrt (static_cast<uint32_t> (variance)) : 0u;
}

inline constexpr void
_S_window_stats (const int8_t *const ZZS_MLLEE_RESTRICT values, const uint32_t count,
                 const bool rms, int32_t &mean, uint32_t &deviation) noexcept
{
    int32_t sum_codes = 0;
    int32_t square_codes = 0;
    simd::stats_i8 (values, count, sum_codes, square_codes);
    _S_window_finish (sum_codes, square_codes, count, rms, mean, deviation);
}

/**
 * @brief `_S_window_stats` over `input + residual`.
 *
 * The sum of two signed codes leaves the int8 domain, so the SIMD int8 stats
 * kernel cannot run here; the pair is still stride-1 and the window is bounded
 * by the arena, so the scalar walk stays inside the norm frame budget.
 */
inline constexpr void
_S_window_stats_sum (const int8_t *const ZZS_MLLEE_RESTRICT values,
                     const int8_t *const ZZS_MLLEE_RESTRICT residual,
                     const uint32_t count, const bool rms, int32_t &mean,
                     uint32_t &deviation) noexcept
{
    int64_t sum = 0;
    int64_t square_sum = 0;
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t value = static_cast<int32_t> (values[i])
                                  + static_cast<int32_t> (residual[i]);
            sum += value;
            square_sum += static_cast<int64_t> (value) * value;
        }
    _S_window_finish (sum, square_sum, count, rms, mean, deviation);
}

/**
 * @brief One normalized, gamma/beta-scaled output code (256x fixed point).
 *
 * `deviation_div` is `_S_div32_make(deviation)` computed once per window; a
 * zero deviation carries divisor zero and selects the exact-centre branch.
 */
[[nodiscard]] inline constexpr int32_t
_S_norm_value (const int32_t value, const int32_t mean, const _S_Div32 deviation_div,
               const bool rms, const int32_t gamma_q7, const int32_t multiplier,
               const int32_t beta) noexcept
{
    const int32_t centered = rms ? value : value - mean;
    const int32_t normalized
        = deviation_div._M_divisor != 0u
              ? _S_div32_i (centered * 256, deviation_div)
              : 0;
    const int32_t scaled = _S_round_shift (static_cast<int64_t> (normalized) * gamma_q7, 7u);
    const int32_t requantized = _S_round_shift (simd::requant_q15 (scaled, multiplier), 8u);
    return requantized + beta;
}

[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_norm (const Layer &layer, const uint8_t *model, const int8_t *input,
         const int8_t *residual, int8_t *output) noexcept
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
            if (residual != nullptr)
                _S_window_stats_sum (input + base, residual + base, window, rms, mean,
                                     deviation);
            else
                _S_window_stats (input + base, window, rms, mean, deviation);
            const _S_Div32 deviation_div = _S_div32_make (deviation);
            for (uint32_t i = 0; i < window; ++i)
                {
                    const uint32_t index = base + i;
                    const int32_t value
                        = residual != nullptr
                              ? static_cast<int32_t> (input[index])
                                    + static_cast<int32_t> (residual[index])
                              : static_cast<int32_t> (input[index]);
                    const int32_t gamma = layer._M_scale_off != 0
                                              ? _S_i16 (model + layer._M_scale_off + index * 2u)
                                              : 128;
                    const int32_t beta = _S_bias_value (model, layer, index);
                    output[index] = _S_i8_code (
                        _S_apply_act (layer._M_act,
                                      _S_norm_value (value, mean, deviation_div, rms,
                                                     gamma, layer._M_requant_mult, beta),
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
 * The sum fits in 32 bits (255 levels x 32767, i.e. below 2^23), so the row
 * accumulator is uint32 and the per-element normalization uses one reciprocal
 * instead of a hardware divide for every code.  The half-adder is a shift
 * because the sum is positive.
 *
 * @return the integer row sum, or 0 for an empty row (the caller maps that to
 *         INTERNAL) and writes the row itself through output.
 */
[[nodiscard]] inline int32_t
_S_softmax_row (const int8_t *const ZZS_MLLEE_RESTRICT input,
                int8_t *const ZZS_MLLEE_RESTRICT output, const uint32_t count) noexcept
{
    if (count == 0)
        return 0;
    const int32_t maximum = simd::max_i8 (input, count);
    uint32_t sum = 0;
    for (uint32_t i = 0; i < count; ++i)
        {
            uint32_t distance = static_cast<uint32_t> (maximum - input[i]);
            if (distance > 255u)
                distance = 255u;
            sum += _S_exp_table._M_value[distance];
        }
    if (sum == 0)
        return 0;
    const _S_Div32 sum_div = _S_div32_make (sum);
    const uint32_t half = sum >> 1;
    for (uint32_t i = 0; i < count; ++i)
        {
            uint32_t distance = static_cast<uint32_t> (maximum - input[i]);
            if (distance > 255u)
                distance = 255u;
            const uint32_t numerator = _S_exp_table._M_value[distance] * 127u + half;
            output[i] = static_cast<int8_t> (_S_div32_u (numerator, sum_div));
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

/**
 * @name Rotary position embedding
 *
 * Q and K are rotated in adjacent pairs by `theta_i = pos * 10000^(-2i/d)`.
 * The angle is carried in units of `2*pi/128`, the quarter-wave sine table has
 * 33 entries (0..pi/2 in 32 steps) and the other quadrants fold onto it, and
 * the per-pair angle step stays in Q16 so a 16-bit sequence index never
 * overflows the product.  head_dim is at most 8, so four pairs is the maximum.
 * @{
 */
inline constexpr int16_t _S_sin_table[33]
    = { 0,     1608,  3212,  4808,  6393,  7962,  9512,   11039, 12539,
        14010, 15446, 16846, 18204, 19519, 20787, 22005,  23170, 24279,
        25329, 26319, 27245, 28105, 28898, 29621, 30273,  30852, 31356,
        31785, 32137, 32412, 32609, 32728, 32767 };

/** @brief sin(2*pi*unit/128) in Q15, folded onto the quarter-wave table. */
[[nodiscard]] inline constexpr int32_t
_S_sin_unit (uint32_t unit) noexcept
{
    unit &= 127u;
    const uint32_t quadrant = unit >> 5;
    const uint32_t remainder = unit & 31u;
    const int32_t value = (quadrant == 1u || quadrant == 3u)
                              ? _S_sin_table[32u - remainder]
                              : _S_sin_table[remainder];
    return quadrant >= 2u ? -value : value;
}

/** @brief Q16 angle step per position, walked as [pairs - 1][pair] rows. */
inline constexpr int32_t _S_rope_step_q16[10] = {
    1335088,
    1335088, 13351,
    1335088, 61969, 2876,
    1335088, 133509, 13351, 1335,
};

/** @brief Rotate one Q/K row into an int16 lane buffer (head_dim <= 8). */
inline constexpr void
_S_rope_rotate_row (const int8_t *const ZZS_MLLEE_RESTRICT source, const uint32_t position,
                    int16_t *const ZZS_MLLEE_RESTRICT rotated, const uint32_t pairs) noexcept
{
    const uint32_t row = (pairs * (pairs - 1u)) >> 1;
    for (uint32_t pair = 0; pair < pairs; ++pair)
        {
            const int32_t step = _S_rope_step_q16[row + pair];
            const uint32_t angle
                = (static_cast<uint32_t> (
                       static_cast<int32_t> (position) * step)
                   >> 16)
                  & 127u;
            const int32_t cosine = _S_sin_unit (angle + 32u);
            const int32_t sine = _S_sin_unit (angle);
            const int32_t x = source[2u * pair];
            const int32_t y = source[2u * pair + 1u];
            rotated[2u * pair] = static_cast<int16_t> (_S_round_shift (
                static_cast<int64_t> (x) * cosine - static_cast<int64_t> (y) * sine,
                15u));
            rotated[2u * pair + 1u] = static_cast<int16_t> (_S_round_shift (
                static_cast<int64_t> (x) * sine + static_cast<int64_t> (y) * cosine,
                15u));
        }
}
/** @} */

/** @brief Scaled-dot score of two rows, optionally RoPE-rotated. */
[[nodiscard]] inline int32_t
_S_attention_score (const int8_t *const ZZS_MLLEE_RESTRICT query,
                    const int16_t *const ZZS_MLLEE_RESTRICT rotated_query,
                    const int8_t *const ZZS_MLLEE_RESTRICT key, const uint32_t key_pos,
                    const uint32_t head_dim, const bool rope) noexcept
{
    if (!rope)
        return simd::dot_i8 (query, key, head_dim);
    int16_t rotated_key[8];
    _S_rope_rotate_row (key, key_pos, rotated_key, head_dim / 2u);
    int32_t total = 0;
    for (uint32_t d = 0; d < head_dim; ++d)
        total += static_cast<int32_t> (rotated_query[d])
                 * static_cast<int32_t> (rotated_key[d]);
    return total;
}

[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_attention (const Layer &layer, const int8_t *input, int8_t *output) noexcept
{
    const uint32_t sequence = layer._M_seq;
    const uint32_t head_dim = layer._M_head_dim;
    const uint32_t heads = _S_heads (layer);
    if (sequence == 0 || head_dim - 1u > 7u)
        return Status::UNSUPPORTED;
    const uint32_t stride = head_dim * heads;
    const bool causal = (layer._M_flags & ATTN_CAUSAL) != 0;
    const bool rope = (layer._M_flags & ATTN_ROPE) != 0;
    if (rope && (head_dim & 1u) != 0u)
        return Status::UNSUPPORTED;
    const uint32_t pairs = head_dim / 2u;
    const int32_t aux = _S_act_param (layer);
    const _S_Div32 score_div = _S_div32_make (_S_isqrt (head_dim << 16));

    for (uint32_t head = 0; head < heads; ++head)
        {
            const int8_t *const queries = input + head * head_dim;
            const int8_t *const keys = queries + sequence * stride;
            const int8_t *const values = keys + sequence * stride;

            for (uint32_t query = 0; query < sequence; ++query)
                {
                    const int8_t *const query_row = queries + query * stride;
                    int16_t rotated_query[8];
                    if (rope)
                        _S_rope_rotate_row (query_row, query, rotated_query, pairs);
                    const uint32_t limit = causal ? query + 1u : sequence;
                    int32_t maximum = -2147483647;
                    for (uint32_t key = 0; key < limit; ++key)
                        {
                            const int32_t score = _S_attention_score (
                                query_row, rotated_query, keys + key * stride, key,
                                head_dim, rope);
                            if (score > maximum)
                                maximum = score;
                        }

                    int64_t accumulator[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
                    uint32_t denominator = 0;
                    for (uint32_t key = 0; key < limit; ++key)
                        {
                            const int32_t score = _S_attention_score (
                                query_row, rotated_query, keys + key * stride, key,
                                head_dim, rope);
                            uint32_t distance = static_cast<uint32_t> (maximum - score);
                            distance = _S_div32_u (distance << 11, score_div);
                            if (distance > 255u)
                                distance = 255u;
                            const int32_t weight = _S_exp_table._M_value[distance];
                            denominator += static_cast<uint32_t> (weight);
                            const int8_t *const value_row = values + key * stride;
                            for (uint32_t d = 0; d < head_dim; ++d)
                                accumulator[d] += weight * static_cast<int32_t> (value_row[d]);
                        }
                    if (denominator == 0)
                        return Status::INTERNAL;
                    const uint32_t half = denominator >> 1;
                    for (uint32_t d = 0; d < head_dim; ++d)
                        {
                            const int64_t numerator = accumulator[d] + half;
                            /* The int64 form also covers sequences the arena-sized
                               embedded models never reach; `denominator` itself fits
                               uint32 because the sequence field is uint16 and every
                               weight is at most 32767. */
                            const int64_t mixed
                                = numerator / static_cast<int64_t> (denominator);
                            output[query * stride + head * head_dim + d] = _S_i8_code (
                                _S_apply_act (layer._M_act,
                                              static_cast<int32_t> (mixed), aux));
                        }
                }
        }
    return Status::OK;
}
} // namespace zzs::mllee
