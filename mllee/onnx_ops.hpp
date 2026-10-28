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
#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "onnx_hash.hpp"
#include "onnx_node_info.hpp"
#include "onnx_string.hpp"
#include "onnx_types.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/** @brief Attribute names the lowering path understands. */
enum class AttrId : uint8_t
{
    NONE = 0,
    ALPHA,
    BETA,
    TRANSA,
    TRANSB,
    AXIS,
    AXES,
    KEEPDIMS,
    KERNEL_SHAPE,
    STRIDES,
    PADS,
    DILATIONS,
    GROUP,
    AUTO_PAD,
    EPSILON,
    MOMENTUM,
    COUNT_INCLUDE_PAD,
    CEIL_MODE,
    PERM,
    SPLIT,
    VALUE,
    TO,
    SEED,
    ELEMENTWISE_AFFINE,
    STORAGE_ORDER,
    LARGEST,
    SELECT_LAST_INDEX,
    NOOP_WITH_BIAS,
    TRAINING_MODE,
    OUTPUT_DTYPE,
};

struct AttrBinding
{
    uint32_t _M_hash;
    uint32_t _M_hash2;
    uint16_t _M_length;
    AttrId _M_id;
};

inline constexpr AttrBinding _S_attribute_registry[] = {
    { fnv1a_literal ("alpha"), djb2_literal ("alpha"), 5, AttrId::ALPHA },
    { fnv1a_literal ("beta"), djb2_literal ("beta"), 4, AttrId::BETA },
    { fnv1a_literal ("transA"), djb2_literal ("transA"), 6, AttrId::TRANSA },
    { fnv1a_literal ("transB"), djb2_literal ("transB"), 6, AttrId::TRANSB },
    { fnv1a_literal ("axis"), djb2_literal ("axis"), 4, AttrId::AXIS },
    { fnv1a_literal ("axes"), djb2_literal ("axes"), 4, AttrId::AXES },
    { fnv1a_literal ("keepdims"), djb2_literal ("keepdims"), 8, AttrId::KEEPDIMS },
    { fnv1a_literal ("kernel_shape"), djb2_literal ("kernel_shape"), 12,
      AttrId::KERNEL_SHAPE },
    { fnv1a_literal ("strides"), djb2_literal ("strides"), 7, AttrId::STRIDES },
    { fnv1a_literal ("pads"), djb2_literal ("pads"), 4, AttrId::PADS },
    { fnv1a_literal ("dilations"), djb2_literal ("dilations"), 9,
      AttrId::DILATIONS },
    { fnv1a_literal ("group"), djb2_literal ("group"), 5, AttrId::GROUP },
    { fnv1a_literal ("auto_pad"), djb2_literal ("auto_pad"), 8, AttrId::AUTO_PAD },
    { fnv1a_literal ("epsilon"), djb2_literal ("epsilon"), 7, AttrId::EPSILON },
    { fnv1a_literal ("momentum"), djb2_literal ("momentum"), 8,
      AttrId::MOMENTUM },
    { fnv1a_literal ("count_include_pad"), djb2_literal ("count_include_pad"), 17,
      AttrId::COUNT_INCLUDE_PAD },
    { fnv1a_literal ("ceil_mode"), djb2_literal ("ceil_mode"), 9, AttrId::CEIL_MODE },
    { fnv1a_literal ("perm"), djb2_literal ("perm"), 4, AttrId::PERM },
    { fnv1a_literal ("split"), djb2_literal ("split"), 5, AttrId::SPLIT },
    { fnv1a_literal ("value"), djb2_literal ("value"), 5, AttrId::VALUE },
    { fnv1a_literal ("to"), djb2_literal ("to"), 2, AttrId::TO },
    { fnv1a_literal ("seed"), djb2_literal ("seed"), 4, AttrId::SEED },
    { fnv1a_literal ("elementwise_affine"), djb2_literal ("elementwise_affine"), 17,
      AttrId::ELEMENTWISE_AFFINE },
    { fnv1a_literal ("storage_order"), djb2_literal ("storage_order"), 13,
      AttrId::STORAGE_ORDER },
    { fnv1a_literal ("largest"), djb2_literal ("largest"), 7, AttrId::LARGEST },
    { fnv1a_literal ("select_last_index"), djb2_literal ("select_last_index"), 17,
      AttrId::SELECT_LAST_INDEX },
    { fnv1a_literal ("noop_with_bias"), djb2_literal ("noop_with_bias"), 14,
      AttrId::NOOP_WITH_BIAS },
    { fnv1a_literal ("training_mode"), djb2_literal ("training_mode"), 13,
      AttrId::TRAINING_MODE },
    { fnv1a_literal ("output_dtype"), djb2_literal ("output_dtype"), 12,
      AttrId::OUTPUT_DTYPE },
};

[[nodiscard]] inline AttrId
attribute_id (const StringInfo &name) noexcept
{
    for (const AttrBinding &binding : _S_attribute_registry)
        if (binding._M_length == name._M_length
            && binding._M_hash == name._M_hash
            && binding._M_hash2 == name._M_hash2)
            return binding._M_id;
    return AttrId::NONE;
}

/** @brief An ONNX operator mapped onto the mllee executor. */
struct OpBinding
{
    uint32_t _M_hash;
    uint32_t _M_hash2;
    uint16_t _M_length;
    Op _M_op;
    Act _M_act;
    uint8_t _M_flags;
};

/** @brief Set on a binding whose ONNX semantics need an attribute rewrite. */
inline constexpr uint8_t _S_op_binding_transposed = 1u << 0;
inline constexpr uint8_t _S_op_binding_depthwise = 1u << 1;
inline constexpr uint8_t _S_op_binding_global = 1u << 2;

/**
 * @brief Operator-name registry in .rodata.
 *
 * A table entry costs 16 bytes of Flash and no RAM.  The linear scan touches
 * one cache line for a typical model and only runs once per node, after the
 * node's bytes have already been streamed; a hash-sorted binary search would
 * add a compile-time sort and several hundred bytes of code for no measurable
 * gain at embedded model sizes.
 */
inline constexpr OpBinding _S_operator_registry[] = {
    { fnv1a_literal ("Gemm"), djb2_literal ("Gemm"), 4, Op::LINEAR, Act::NONE, 0 },
    { fnv1a_literal ("MatMul"), djb2_literal ("MatMul"), 6, Op::MATMUL, Act::NONE, 0 },
    { fnv1a_literal ("Conv"), djb2_literal ("Conv"), 4, Op::CONV1D, Act::NONE, 0 },
    { fnv1a_literal ("Add"), djb2_literal ("Add"), 3, Op::ADD, Act::NONE, 0 },
    { fnv1a_literal ("Sub"), djb2_literal ("Sub"), 3, Op::SUB, Act::NONE, 0 },
    { fnv1a_literal ("Mul"), djb2_literal ("Mul"), 3, Op::MUL, Act::NONE, 0 },
    { fnv1a_literal ("Relu"), djb2_literal ("Relu"), 4, Op::ACTIVATION, Act::RELU, 0 },
    { fnv1a_literal ("Relu6"), djb2_literal ("Relu6"), 5, Op::ACTIVATION, Act::RELU6, 0 },
    { fnv1a_literal ("Sigmoid"), djb2_literal ("Sigmoid"), 7, Op::ACTIVATION,
      Act::SIGMOID, 0 },
    { fnv1a_literal ("Tanh"), djb2_literal ("Tanh"), 4, Op::ACTIVATION, Act::TANH, 0 },
    { fnv1a_literal ("Gelu"), djb2_literal ("Gelu"), 4, Op::ACTIVATION, Act::GELU, 0 },
    { fnv1a_literal ("LeakyRelu"), djb2_literal ("LeakyRelu"), 9, Op::ACTIVATION,
      Act::LEAKY_RELU, 0 },
    { fnv1a_literal ("Elu"), djb2_literal ("Elu"), 3, Op::ACTIVATION, Act::ELU, 0 },
    { fnv1a_literal ("Silu"), djb2_literal ("Silu"), 4, Op::ACTIVATION, Act::SILU, 0 },
    { fnv1a_literal ("HardSwish"), djb2_literal ("HardSwish"), 9, Op::ACTIVATION,
      Act::HARDSWISH, 0 },
    { fnv1a_literal ("HardSigmoid"), djb2_literal ("HardSigmoid"), 11,
      Op::ACTIVATION, Act::HARDSIGMOID, 0 },
    { fnv1a_literal ("Softmax"), djb2_literal ("Softmax"), 7, Op::SOFTMAX, Act::NONE, 0 },
    { fnv1a_literal ("LogSoftmax"), djb2_literal ("LogSoftmax"), 10,
      Op::LOG_SOFTMAX, Act::NONE, 0 },
    { fnv1a_literal ("Clip"), djb2_literal ("Clip"), 4, Op::CLAMP, Act::NONE, 0 },
    { fnv1a_literal ("PRelu"), djb2_literal ("PRelu"), 5, Op::PRELU, Act::NONE, 0 },
    { fnv1a_literal ("LayerNormalization"), djb2_literal ("LayerNormalization"), 18,
      Op::NORM, Act::NONE, 0 },
    { fnv1a_literal ("BatchNormalization"), djb2_literal ("BatchNormalization"), 18,
      Op::BATCH_NORM, Act::NONE, 0 },
    { fnv1a_literal ("GroupNormalization"), djb2_literal ("GroupNormalization"), 18,
      Op::GROUP_NORM, Act::NONE, 0 },
    { fnv1a_literal ("InstanceNormalization"), djb2_literal ("InstanceNormalization"),
      21, Op::NORM, Act::NONE, 0 },
    { fnv1a_literal ("MaxPool"), djb2_literal ("MaxPool"), 7, Op::MAX_POOL1D,
      Act::NONE, 0 },
    { fnv1a_literal ("AveragePool"), djb2_literal ("AveragePool"), 11,
      Op::AVG_POOL1D, Act::NONE, 0 },
    { fnv1a_literal ("GlobalAveragePool"), djb2_literal ("GlobalAveragePool"), 17,
      Op::GLOBAL_AVG_POOL, Act::NONE, _S_op_binding_global },
    { fnv1a_literal ("Gather"), djb2_literal ("Gather"), 6, Op::EMBEDDING, Act::NONE, 0 },
    { fnv1a_literal ("Concat"), djb2_literal ("Concat"), 6, Op::CONCAT, Act::NONE, 0 },
    { fnv1a_literal ("Slice"), djb2_literal ("Slice"), 5, Op::SLICE, Act::NONE, 0 },
    { fnv1a_literal ("Pad"), djb2_literal ("Pad"), 3, Op::PAD1D, Act::NONE, 0 },
    { fnv1a_literal ("Transpose"), djb2_literal ("Transpose"), 9, Op::TRANSPOSE2D,
      Act::NONE, 0 },
    { fnv1a_literal ("ReduceMean"), djb2_literal ("ReduceMean"), 10, Op::REDUCE,
      Act::NONE, 0 },
    { fnv1a_literal ("ReduceSum"), djb2_literal ("ReduceSum"), 9, Op::REDUCE,
      Act::NONE, 0 },
    { fnv1a_literal ("ReduceMax"), djb2_literal ("ReduceMax"), 9, Op::REDUCE,
      Act::NONE, 0 },
    { fnv1a_literal ("ReduceMin"), djb2_literal ("ReduceMin"), 9, Op::REDUCE,
      Act::NONE, 0 },
    { fnv1a_literal ("ArgMax"), djb2_literal ("ArgMax"), 6, Op::ARGMAX, Act::NONE, 0 },
    { fnv1a_literal ("Resize"), djb2_literal ("Resize"), 6, Op::UPSAMPLE_NEAREST,
      Act::NONE, 0 },
    { fnv1a_literal ("Upsample"), djb2_literal ("Upsample"), 8, Op::UPSAMPLE_NEAREST,
      Act::NONE, 0 },
    { fnv1a_literal ("Flatten"), djb2_literal ("Flatten"), 7, Op::FLATTEN, Act::NONE, 0 },
    { fnv1a_literal ("Reshape"), djb2_literal ("Reshape"), 7, Op::FLATTEN, Act::NONE, 0 },
    { fnv1a_literal ("Squeeze"), djb2_literal ("Squeeze"), 7, Op::FLATTEN, Act::NONE, 0 },
    { fnv1a_literal ("Unsqueeze"), djb2_literal ("Unsqueeze"), 9, Op::FLATTEN,
      Act::NONE, 0 },
    { fnv1a_literal ("Dropout"), djb2_literal ("Dropout"), 7, Op::DROPOUT, Act::NONE, 0 },
    { fnv1a_literal ("Attention"), djb2_literal ("Attention"), 9, Op::ATTENTION,
      Act::NONE, 0 },
    { fnv1a_literal ("Identity"), djb2_literal ("Identity"), 8, Op::FLATTEN, Act::NONE, 0 },
};

[[nodiscard]] inline OpBinding
lookup_op (const NodeInfo &node) noexcept
{
    if ((node._M_flags & NODE_FLAG_HAS_OP) == 0u
        || (node._M_flags & NODE_FLAG_DEFAULT_DOMAIN) == 0u)
        return OpBinding{ 0, 0, 0, Op::COUNT, Act::COUNT, 0 };
    for (const OpBinding &binding : _S_operator_registry)
        if (binding._M_length == node._M_op_length
            && binding._M_hash == node._M_op_hash
            && binding._M_hash2 == node._M_op_hash2)
            return binding;
    return OpBinding{ 0, 0, 0, Op::COUNT, Act::COUNT, 0 };
}

[[nodiscard]] inline const char *
operator_name (const Op op) noexcept
{
    switch (op)
        {
        case Op::LINEAR:
            return "Gemm";
        case Op::MATMUL:
            return "MatMul";
        case Op::CONV1D:
            return "Conv";
        case Op::ADD:
            return "Add";
        case Op::SUB:
            return "Sub";
        case Op::MUL:
            return "Mul";
        case Op::ACTIVATION:
            return "Activation";
        case Op::SOFTMAX:
            return "Softmax";
        case Op::LOG_SOFTMAX:
            return "LogSoftmax";
        case Op::CLAMP:
            return "Clip";
        case Op::PRELU:
            return "PRelu";
        case Op::NORM:
            return "Normalization";
        case Op::BATCH_NORM:
            return "BatchNormalization";
        case Op::GROUP_NORM:
            return "GroupNormalization";
        case Op::MAX_POOL1D:
            return "MaxPool";
        case Op::AVG_POOL1D:
            return "AveragePool";
        case Op::GLOBAL_AVG_POOL:
            return "GlobalAveragePool";
        case Op::EMBEDDING:
            return "Gather";
        case Op::CONCAT:
            return "Concat";
        case Op::SLICE:
            return "Slice";
        case Op::PAD1D:
            return "Pad";
        case Op::TRANSPOSE2D:
            return "Transpose";
        case Op::REDUCE:
            return "Reduce";
        case Op::ARGMAX:
            return "ArgMax";
        case Op::UPSAMPLE_NEAREST:
            return "Resize";
        case Op::FLATTEN:
            return "Flatten";
        case Op::DROPOUT:
            return "Dropout";
        case Op::ATTENTION:
            return "Attention";
        case Op::L2_NORMALIZE:
            return "L2Normalize";
        case Op::MASKED_FILL:
            return "MaskedFill";
        case Op::COUNT:
            return "Unknown";
        }
    return "Unknown";
}
} // namespace zzs::mllee::onnx
