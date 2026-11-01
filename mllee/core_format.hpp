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

#include <cstdint>

namespace zzs::mllee
{
enum class Op : uint8_t
{
    LINEAR = 0,
    CONV1D,
    NORM,
    ACTIVATION,
    SOFTMAX,
    ATTENTION,
    ADD,
    SUB,
    MUL,
    CLAMP,
    PRELU,
    BATCH_NORM,
    GROUP_NORM,
    MAX_POOL1D,
    AVG_POOL1D,
    GLOBAL_AVG_POOL,
    EMBEDDING,
    CONCAT,
    SLICE,
    PAD1D,
    MATMUL,
    TRANSPOSE2D,
    REDUCE,
    ARGMAX,
    LOG_SOFTMAX,
    UPSAMPLE_NEAREST,
    MASKED_FILL,
    DROPOUT,
    FLATTEN,
    L2_NORMALIZE,
    COUNT,
};

enum class Act : uint8_t
{
    NONE = 0,
    RELU,
    RELU6,
    SIGMOID,
    TANH,
    GELU,
    LEAKY_RELU,
    ELU,
    SILU,
    HARDSWISH,
    HARDSIGMOID,
    COUNT,
};

enum class Codec : uint8_t
{
    RAW = 0,
    PACKLOW,
};

enum LayerFlag : uint8_t
{
    INPUT_EXTERNAL = 1u << 0,
    CONV_CAUSAL = 1u << 1,
    NORM_RMS = 1u << 2,
    ACT_IN_PLACE = 1u << 3,
    ATTN_CAUSAL = 1u << 4,
    CONV_DEPTHWISE = 1u << 5,
    NORM_INSTANCE = 1u << 6,
    /** @brief ATTENTION: rotate Q/K with RoPE (head_dim must be even). */
    ATTN_ROPE = 1u << 7,
    /** @brief NORM: normalize (input + second arena operand) instead of input. */
    NORM_RESIDUAL = 1u << 7,
};

inline constexpr uint8_t _S_model_version = 1;
inline constexpr uint8_t _S_op_count = static_cast<uint8_t> (Op::COUNT);
inline constexpr uint8_t _S_op_compat_min = 5;
inline constexpr uint8_t _S_layer_flag_mask = 0xFFu;
inline constexpr uint32_t _S_model_header_size = 32;
inline constexpr uint32_t _S_layer_desc_size = 48;

/**
 * @brief Writable RAM the module owns, in bytes (Cortex-M4 / 32-bit layout).
 *
 * The budget covers the resident `Core` object plus the deepest call chain the
 * module can build (`Core::run` -> `_S_execute` -> op -> codec reader).  The
 * chain lives in the caller's stack, so it is gated separately by
 * `_S_stack_budget_bytes`; the remainder, `_S_core_budget_bytes`, is the
 * ceiling for `sizeof (Core)`.
 */
inline constexpr uint32_t _S_ram_budget_bytes = 1024;
inline constexpr uint32_t _S_stack_budget_bytes = 384;
inline constexpr uint32_t _S_core_budget_bytes
    = _S_ram_budget_bytes - _S_stack_budget_bytes;
/** @brief Extra allowance for 64-bit hosts, where workspace pointers are padded. */
inline constexpr uint32_t _S_core_budget_host_bytes = _S_core_budget_bytes + 16;

/**
 * @brief Activation arena carried inside `Core`.
 *
 * Sized so that the fixed workspace, the arena and the status word fill
 * `_S_core_budget_bytes` exactly on a 32-bit target.  A model declares how much
 * of it it uses in header byte 22; anything larger is rejected with
 * `ARENA_OVERFLOW` before a single byte is written.
 */
inline constexpr uint16_t _S_arena_bytes
    = static_cast<uint16_t> (_S_core_budget_bytes - 196u);
inline constexpr uint16_t _S_external_input = 0xFFFFu;
inline constexpr uint32_t _S_dict_block_bytes = 32;
inline constexpr uint8_t _S_reduce_sum = 0;
inline constexpr uint8_t _S_reduce_mean = 1;
inline constexpr uint8_t _S_reduce_max = 2;
inline constexpr uint8_t _S_reduce_min = 3;

struct Header
{
    uint8_t _M_version = 0;
    uint8_t _M_flags = 0;
    uint8_t _M_layer_count = 0;
    uint8_t _M_default_dtype = 0;
    uint32_t _M_layer_table_off = 0;
    uint32_t _M_scale_table_off = 0;
    uint32_t _M_dict_off = 0;
    uint16_t _M_dict_len = 0;
    uint16_t _M_arena_bytes = 0;
};

struct Layer
{
    Op _M_op = Op::LINEAR;
    Act _M_act = Act::NONE;
    Dtype _M_dtype = Dtype::I8;
    Codec _M_codec = Codec::RAW;
    uint16_t _M_in_dim = 0;
    uint16_t _M_out_dim = 0;
    uint8_t _M_kernel = 0;
    uint8_t _M_stride = 0;
    uint8_t _M_groups = 0;
    uint8_t _M_group_size = 0;
    uint8_t _M_head_dim = 0;
    uint8_t _M_seq = 0;
    uint8_t _M_flags = 0;
    /** @brief Op-specific small parameter; see the table in mllee/README.md. */
    uint8_t _M_aux = 0;
    uint32_t _M_weight_off = 0;
    uint32_t _M_weight_len = 0;
    uint32_t _M_scale_off = 0;
    uint32_t _M_bias_off = 0;
    uint32_t _M_lut_off = 0;
    uint16_t _M_in_off = 0;
    uint16_t _M_out_off = 0;
    uint16_t _M_in_len = 0;
    uint16_t _M_out_len = 0;
    int32_t _M_requant_mult = 0;
};

[[nodiscard]] inline constexpr uint16_t
_S_u16 (const uint8_t *data) noexcept
{
    return static_cast<uint16_t> (static_cast<uint16_t> (data[0])
                                       | (static_cast<uint16_t> (data[1]) << 8));
}

[[nodiscard]] inline constexpr uint32_t
_S_u32 (const uint8_t *data) noexcept
{
    return static_cast<uint32_t> (data[0]) | (static_cast<uint32_t> (data[1]) << 8)
           | (static_cast<uint32_t> (data[2]) << 16)
           | (static_cast<uint32_t> (data[3]) << 24);
}

[[nodiscard]] inline constexpr int16_t
_S_i16 (const uint8_t *data) noexcept
{
    return static_cast<int16_t> (_S_u16 (data));
}

[[nodiscard]] inline constexpr int32_t
_S_i32 (const uint8_t *data) noexcept
{
    return static_cast<int32_t> (_S_u32 (data));
}

[[nodiscard]] inline constexpr bool
_S_span_ok (const uint32_t offset, const uint32_t span, const uint32_t total) noexcept
{
    return offset <= total && span <= total - offset;
}

/** @brief Ops whose operand lives in a Flash weight stream (packlow-framed). */
[[nodiscard]] inline constexpr bool
_S_weighted (const Op op) noexcept
{
    return op == Op::LINEAR || op == Op::CONV1D;
}

/** @brief Ops that read their second operand from a second arena slot. */
[[nodiscard]] inline constexpr bool
_S_two_operand (const Op op) noexcept
{
    return op == Op::ADD || op == Op::SUB || op == Op::MUL || op == Op::CONCAT
           || op == Op::MASKED_FILL || op == Op::MATMUL;
}

/** @brief Ops whose window size and step come from the kernel/stride bytes. */
[[nodiscard]] inline constexpr bool
_S_windowed (const Op op) noexcept
{
    return op == Op::CONV1D || op == Op::MAX_POOL1D || op == Op::AVG_POOL1D
           || op == Op::UPSAMPLE_NEAREST;
}

/** @brief Second arena operand base, carried in the unused weight offset low half. */
[[nodiscard]] inline constexpr uint16_t
_S_in2_off (const Layer &layer) noexcept
{
    return static_cast<uint16_t> (layer._M_weight_off & 0xFFFFu);
}

/** @brief Second arena operand length; zero means "same as the first operand". */
[[nodiscard]] inline constexpr uint16_t
_S_in2_len (const Layer &layer) noexcept
{
    return static_cast<uint16_t> (layer._M_weight_len & 0xFFFFu);
}

/** @brief Number of attention heads carried in the group count byte. */
[[nodiscard]] inline constexpr uint32_t
_S_heads (const Layer &layer) noexcept
{
    return layer._M_groups != 0 ? layer._M_groups : 1u;
}

/** @brief Length of one reduction / normalization group. */
[[nodiscard]] inline constexpr uint32_t
_S_group_len (const Layer &layer) noexcept
{
    return layer._M_group_size != 0 ? layer._M_group_size : layer._M_in_dim;
}

/** @brief The aux byte read as a signed code, i.e. aux/127 in normalized units. */
[[nodiscard]] inline constexpr int32_t
_S_aux_param (const Layer &layer) noexcept
{
    return static_cast<int32_t> (static_cast<int8_t> (layer._M_aux));
}

/**
 * @brief Convolution dilation.
 *
 * CONV1D is the one op that spends the aux byte on a shape parameter, so a
 * fused parametric activation there falls back to the PyTorch defaults.
 */
[[nodiscard]] inline constexpr uint32_t
_S_dilation (const Layer &layer) noexcept
{
    return layer._M_op == Op::CONV1D && layer._M_aux > 1 ? layer._M_aux : 1u;
}

/** @brief Parametric activation slope / alpha, in code units. */
[[nodiscard]] inline constexpr int32_t
_S_act_param (const Layer &layer) noexcept
{
    if (layer._M_op == Op::CONV1D)
        return layer._M_act == Act::ELU ? 127 : 32;
    return _S_aux_param (layer);
}

/**
 * @brief Flash LUT size a float weight format needs, or 0 for integer formats.
 *
 * The LUT maps the on-disk code to the normalized int8 weight the MAC consumes:
 * 8 entries for E1M1, 16 for E2M1 and 256 for either 8-bit float.
 */
[[nodiscard]] inline constexpr uint32_t
_S_lut_bytes (const Dtype dtype) noexcept
{
    switch (dtype)
        {
        case Dtype::F3E1M1:
            return 8u;
        case Dtype::F4E2M1:
            return 16u;
        case Dtype::F8E4M3:
        case Dtype::F8E5M2:
            return 256u;
        default:
            return 0u;
        }
}

[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
parse_header (const uint8_t *model, const uint32_t model_len, Header &header) noexcept
{
    if (model == nullptr || model_len < _S_model_header_size)
        return Status::TRUNCATED;
    if (model[0] != 'M' || model[1] != 'L' || model[2] != 'E' || model[3] != 'E')
        return Status::BAD_MAGIC;
    header._M_version = model[4];
    if (header._M_version != _S_model_version)
        return Status::BAD_VERSION;
    if (model[7] != _S_model_header_size)
        return Status::BAD_LAYOUT;
    if (_S_u16 (model + 28) != _S_crc16 (model, 28))
        return Status::CORRUPT_STREAM;

    header._M_flags = model[5];
    header._M_layer_count = model[6];
    header._M_default_dtype = model[24];
    header._M_layer_table_off = _S_u32 (model + 8);
    header._M_scale_table_off = _S_u32 (model + 12);
    header._M_dict_off = _S_u32 (model + 16);
    header._M_dict_len = _S_u16 (model + 20);
    header._M_arena_bytes = _S_u16 (model + 22);

    if (header._M_layer_count == 0)
        return Status::BAD_LAYOUT;
    if (header._M_default_dtype > _S_dtype_max)
        return Status::BAD_LAYOUT;
    if (header._M_arena_bytes > _S_arena_bytes)
        return Status::ARENA_OVERFLOW;
    if (model[25] < _S_op_compat_min || model[25] > _S_op_count)
        return Status::BAD_LAYOUT;
    if (!_S_span_ok (header._M_layer_table_off,
                     static_cast<uint32_t> (header._M_layer_count) * _S_layer_desc_size, model_len))
        return Status::BAD_LAYOUT;
    if (header._M_scale_table_off > model_len)
        return Status::BAD_LAYOUT;
    if (header._M_dict_len != 0
        && (!_S_span_ok (header._M_dict_off, header._M_dict_len, model_len)
            || header._M_dict_len < _S_dict_block_bytes))
        return Status::BAD_LAYOUT;
    return Status::OK;
}

[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
_S_validate_shape (const Layer &layer) noexcept
{
    const uint32_t in_dim = layer._M_in_dim;
    const uint32_t out_dim = layer._M_out_dim;
    switch (layer._M_op)
        {
        case Op::NORM:
        case Op::SOFTMAX:
        case Op::LOG_SOFTMAX:
        case Op::GROUP_NORM:
        case Op::L2_NORMALIZE:
        case Op::CLAMP:
        case Op::PRELU:
        case Op::BATCH_NORM:
        case Op::DROPOUT:
        case Op::FLATTEN:
        case Op::TRANSPOSE2D:
            if (out_dim != in_dim)
                return Status::BAD_LAYOUT;
            break;
        case Op::ADD:
        case Op::SUB:
        case Op::MUL:
        case Op::MASKED_FILL:
            /* The second arena operand is read element-wise over the whole
               input, so it has to cover at least in_dim bytes. */
            if (out_dim != in_dim || _S_in2_len (layer) < in_dim)
                return Status::BAD_LAYOUT;
            break;
        case Op::REDUCE:
        case Op::ARGMAX:
            if (const uint32_t group_len = _S_group_len (layer);
                group_len == 0 || in_dim % group_len != 0 || out_dim != in_dim / group_len)
                return Status::BAD_LAYOUT;
            break;
        case Op::EMBEDDING:
            if (layer._M_head_dim == 0 || in_dim * layer._M_head_dim != out_dim)
                return Status::BAD_LAYOUT;
            break;
        case Op::MAX_POOL1D:
        case Op::AVG_POOL1D:
        case Op::GLOBAL_AVG_POOL:
        case Op::UPSAMPLE_NEAREST:
            if (layer._M_head_dim == 0 || layer._M_seq == 0 || in_dim % layer._M_head_dim != 0)
                return Status::BAD_LAYOUT;
            if (const uint32_t channels = in_dim / layer._M_head_dim;
                layer._M_op == Op::GLOBAL_AVG_POOL ? out_dim != channels
                                                   : out_dim != channels * layer._M_seq)
                return Status::BAD_LAYOUT;
            break;
        case Op::SLICE:
            if (out_dim > in_dim
                || static_cast<uint32_t> (layer._M_kernel) + out_dim > in_dim)
                return Status::BAD_LAYOUT;
            break;
        case Op::PAD1D:
            if (in_dim + layer._M_kernel + layer._M_stride != out_dim)
                return Status::BAD_LAYOUT;
            break;
        case Op::CONCAT:
            if (_S_in2_len (layer) == 0 || in_dim + _S_in2_len (layer) != out_dim)
                return Status::BAD_LAYOUT;
            break;
        case Op::MATMUL:
            if (const uint32_t inner = layer._M_head_dim;
                inner != 0 && in_dim % inner == 0 && _S_in2_len (layer) % inner == 0)
            {
                if ((in_dim / inner) * (_S_in2_len (layer) / inner) != out_dim)
                    return Status::BAD_LAYOUT;
            }
            else
                return Status::BAD_LAYOUT;
            break;
        case Op::ATTENTION:
            if (layer._M_head_dim == 0 || layer._M_seq == 0 || layer._M_groups > 16)
                return Status::BAD_LAYOUT;
            if (layer._M_head_dim > 8)
                return Status::UNSUPPORTED;
            /* `_M_in_dim` is unused by the op (the QKV span comes from seq and
               head_dim), but the output is exactly one code per token/head. */
            if (out_dim != static_cast<uint32_t> (layer._M_seq)
                               * layer._M_head_dim * _S_heads (layer))
                return Status::BAD_LAYOUT;
            if ((layer._M_flags & ATTN_ROPE) != 0 && (layer._M_head_dim & 1u) != 0u)
                return Status::BAD_LAYOUT;
            break;
        case Op::ACTIVATION:
        case Op::LINEAR:
        case Op::CONV1D:
        default:
            break;
        }
    if (layer._M_op == Op::REDUCE && layer._M_aux > _S_reduce_min)
        return Status::BAD_LAYOUT;
    /* PAD1D, MASKED_FILL, DROPOUT and REDUCE all spend the aux byte on an op
       parameter, so they cannot also carry an activation slope. */
    if (layer._M_op == Op::PAD1D || layer._M_op == Op::MASKED_FILL
        || layer._M_op == Op::DROPOUT || layer._M_op == Op::REDUCE)
        {
            if (layer._M_act != Act::NONE)
                return Status::BAD_LAYOUT;
            if (layer._M_op == Op::DROPOUT && _S_aux_param (layer) < 0)
                return Status::BAD_LAYOUT;
        }
    return Status::OK;
}

/**
 * @brief Codes in one streamed weight row, or 0 when the op has no weight row.
 *
 * A packed row has to be byte-terminated before the next row starts, which for
 * 3-bit formats means a multiple of 8 codes.  LINEAR streams one `in_dim`-long
 * row per output channel, CONV1D one `tap_channels * kernel` row per output
 * channel (a depthwise conv taps one channel) and EMBEDDING one `head_dim`-long
 * table row per token.
 */
[[nodiscard]] inline constexpr uint32_t
_S_weight_row_codes (const Layer &layer) noexcept
{
    switch (layer._M_op)
        {
        case Op::LINEAR:
            return layer._M_in_dim;
        case Op::CONV1D:
            {
                if (layer._M_head_dim == 0
                    || layer._M_in_dim % layer._M_head_dim != 0)
                    return 0;
                const uint32_t in_channels = layer._M_in_dim / layer._M_head_dim;
                if (in_channels == 0)
                    return 0;
                const uint32_t taps
                    = (layer._M_flags & CONV_DEPTHWISE) != 0 ? 1u : in_channels;
                const uint32_t kernel = layer._M_kernel != 0 ? layer._M_kernel : 1u;
                return taps * kernel;
            }
        case Op::EMBEDDING:
            return layer._M_head_dim;
        default:
            return 0;
        }
}

[[nodiscard]] ZZS_MLLEE_NOINLINE inline Status
parse_layer (const uint8_t *model, const uint32_t model_len, const uint32_t offset,
             Layer &layer) noexcept
{
    if (!_S_span_ok (offset, _S_layer_desc_size, model_len))
        return Status::TRUNCATED;
    const uint8_t *const p = model + offset;
    if (p[0] >= static_cast<uint8_t> (Op::COUNT) || p[1] >= static_cast<uint8_t> (Act::COUNT)
        || p[2] > _S_dtype_max || p[3] > static_cast<uint8_t> (Codec::PACKLOW))
        return Status::BAD_LAYOUT;
    if ((p[14] & ~_S_layer_flag_mask) != 0)
        return Status::BAD_LAYOUT;
    const Op op = static_cast<Op> (p[0]);
    /* The packlow frame only wraps a Flash weight stream; every other operand is
       an arena tensor, so a codec byte there would skip 12 header bytes. */
    if (static_cast<Codec> (p[3]) == Codec::PACKLOW && !_S_weighted (op))
        return Status::BAD_LAYOUT;

    layer._M_op = op;
    layer._M_act = static_cast<Act> (p[1]);
    layer._M_dtype = static_cast<Dtype> (p[2]);
    layer._M_codec = static_cast<Codec> (p[3]);
    layer._M_in_dim = _S_u16 (p + 4);
    layer._M_out_dim = _S_u16 (p + 6);
    layer._M_kernel = p[8];
    layer._M_stride = p[9];
    layer._M_groups = p[10];
    layer._M_group_size = p[11];
    layer._M_head_dim = p[12];
    layer._M_seq = p[13];
    layer._M_flags = p[14];
    layer._M_aux = p[15];
    layer._M_weight_off = _S_u32 (p + 16);
    layer._M_weight_len = _S_u32 (p + 20);
    layer._M_scale_off = _S_u32 (p + 24);
    layer._M_bias_off = _S_u32 (p + 28);
    layer._M_lut_off = _S_u32 (p + 32);
    layer._M_in_off = _S_u16 (p + 36);
    layer._M_out_off = _S_u16 (p + 38);
    layer._M_in_len = _S_u16 (p + 40);
    layer._M_out_len = _S_u16 (p + 42);
    layer._M_requant_mult = _S_i32 (p + 44);

    if (layer._M_in_dim == 0 || layer._M_out_dim == 0)
        return Status::BAD_LAYOUT;
    if (_S_windowed (op))
        {
            if (layer._M_stride == 0)
                layer._M_stride = 1;
            if (layer._M_kernel == 0)
                layer._M_kernel = 1;
        }

    if (layer._M_in_off != _S_external_input
        && !_S_span_ok (layer._M_in_off, layer._M_in_len, _S_arena_bytes))
        return Status::ARENA_OVERFLOW;
    if (!_S_span_ok (layer._M_out_off, layer._M_out_len, _S_arena_bytes))
        return Status::ARENA_OVERFLOW;
    if (layer._M_in_len == 0)
        layer._M_in_len = layer._M_in_dim;
    if (layer._M_out_len == 0)
        layer._M_out_len = layer._M_out_dim;

    if (_S_weighted (op) || op == Op::EMBEDDING)
        {
            if (layer._M_weight_len == 0
                || !_S_span_ok (layer._M_weight_off, layer._M_weight_len, model_len))
                return Status::BAD_LAYOUT;
            /* A 3-bit row is only byte-terminated at a multiple of 8 codes; an
               unaligned one would share its final byte with the next row (or
               with the codec header), so reject it instead of mis-decoding. */
            if (const uint32_t row_codes = _S_weight_row_codes (layer);
                row_codes != 0 && !_S_codes_aligned (layer._M_dtype, row_codes))
                return Status::BAD_LAYOUT;
        }
    if (const uint32_t lut_bytes = _S_lut_bytes (layer._M_dtype);
        lut_bytes != 0
        && (layer._M_lut_off == 0 || !_S_span_ok (layer._M_lut_off, lut_bytes, model_len)))
        return Status::BAD_LAYOUT;
    if (_S_two_operand (op))
        {
            const uint32_t second
                = _S_in2_len (layer) != 0 ? _S_in2_len (layer) : layer._M_in_dim;
            if (!_S_span_ok (_S_in2_off (layer), second, _S_arena_bytes))
                return Status::ARENA_OVERFLOW;
        }
    if (op == Op::NORM && (layer._M_flags & NORM_RESIDUAL) != 0)
        {
            const uint32_t second
                = _S_in2_len (layer) != 0 ? _S_in2_len (layer) : layer._M_in_dim;
            if (!_S_span_ok (_S_in2_off (layer), second, _S_arena_bytes))
                return Status::ARENA_OVERFLOW;
        }
    if (layer._M_scale_off != 0 && layer._M_scale_off >= model_len)
        return Status::BAD_LAYOUT;
    if (layer._M_bias_off != 0 && layer._M_bias_off >= model_len)
        return Status::BAD_LAYOUT;

    if (layer._M_group_size != 0)
        {
            const uint32_t expected = (static_cast<uint32_t> (layer._M_in_dim)
                                            + layer._M_group_size - 1u)
                                           / layer._M_group_size;
            if (expected > 0xFFu)
                return Status::BAD_LAYOUT;
            if (layer._M_groups == 0)
                layer._M_groups = static_cast<uint8_t> (expected);
            else if (layer._M_groups != expected)
                return Status::BAD_LAYOUT;
        }
    else if (layer._M_groups != 0 && op != Op::ATTENTION)
        return Status::BAD_LAYOUT;
    return _S_validate_shape (layer);
}

[[nodiscard]] inline const int8_t *
_S_weight_lut (const uint8_t *model, const Layer &layer) noexcept
{
    return layer._M_lut_off != 0
               ? reinterpret_cast<const int8_t *> (model + layer._M_lut_off)
               : nullptr;
}

} // namespace zzs::mllee
