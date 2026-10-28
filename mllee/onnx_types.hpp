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
#include "onnx_config.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/** @brief Protobuf wire types used by the ONNX schema. */
enum class WireType : uint8_t
{
    VARINT = 0,
    FIXED64 = 1,
    LENGTH = 2,
    START_GROUP = 3,
    END_GROUP = 4,
    FIXED32 = 5,
};

/** @brief ONNX TensorProto.DataType values, kept in the wire numbering. */
enum class DataType : uint8_t
{
    UNDEFINED = 0,
    FLOAT = 1,
    UINT8 = 2,
    INT8 = 3,
    UINT16 = 4,
    INT16 = 5,
    INT32 = 6,
    INT64 = 7,
    STRING = 8,
    BOOL = 9,
    FLOAT16 = 10,
    DOUBLE = 11,
    UINT32 = 12,
    UINT64 = 13,
    COMPLEX64 = 14,
    COMPLEX128 = 15,
    BFLOAT16 = 16,
    FLOAT8E4M3FN = 17,
    FLOAT8E4M3FNUZ = 18,
    FLOAT8E5M2 = 19,
    FLOAT8E5M2FNUZ = 20,
    UINT4 = 21,
    INT4 = 22,
    FLOAT4E2M1 = 23,
    FLOAT8E8M0 = 24,
    UINT2 = 25,
    INT2 = 26,
    FLOAT6E2M3 = 27,
    FLOAT6E3M2 = 28,
};

/** @brief AttributeProto.AttributeType values. */
enum class AttributeType : uint8_t
{
    UNDEFINED = 0,
    FLOAT = 1,
    INT = 2,
    STRING = 3,
    TENSOR = 4,
    GRAPH = 5,
    FLOATS = 6,
    INTS = 7,
    STRINGS = 8,
    TENSORS = 9,
    GRAPHS = 10,
    SPARSE_TENSOR = 11,
    SPARSE_TENSORS = 12,
    TYPE_PROTO = 13,
    TYPE_PROTOS = 14,
};

/** @brief Which part of a model the active TensorProto belongs to. */
enum class TensorRole : uint8_t
{
    NONE = 0,
    INITIALIZER,
    ATTRIBUTE,
    VALUE_INPUT,
    VALUE_OUTPUT,
    VALUE_INFO,
    SPARSE_INITIALIZER,
};

/** @brief TensorInfo flag bits shared by the parser and the event consumer. */
enum TensorFlag : uint8_t
{
    TENSOR_FLAG_EXTERNAL = 1u << 0,
    TENSOR_FLAG_DYNAMIC = 1u << 1,
    TENSOR_FLAG_OVERFLOW = 1u << 2,
    TENSOR_FLAG_RAW_DATA = 1u << 3,
    TENSOR_FLAG_TYPED_DATA = 1u << 4,
    TENSOR_FLAG_FRAGMENTED = 1u << 5,
    TENSOR_FLAG_SPARSE = 1u << 6,
    TENSOR_FLAG_INVALID = 1u << 7,
};

/** @brief TensorProto field numbers that can carry element payloads. */
enum class TensorDataField : uint8_t
{
    NONE = 0,
    FLOAT_DATA = 4,
    INT32_DATA = 5,
    STRING_DATA = 6,
    INT64_DATA = 7,
    RAW_DATA = 9,
    DOUBLE_DATA = 10,
    UINT64_DATA = 11,
    EXTERNAL_DATA = 13,
};

/** @brief ModelInfo flag bits. */
enum ModelFlag : uint8_t
{
    MODEL_FLAG_HAS_GRAPH = 1u << 0,
    MODEL_FLAG_HAS_NODES = 1u << 1,
    MODEL_FLAG_HAS_INITIALIZERS = 1u << 2,
    MODEL_FLAG_COUNT_OVERFLOW = 1u << 3,
    MODEL_FLAG_IR_TOO_NEW = 1u << 4,
    MODEL_FLAG_OPSET_TOO_NEW = 1u << 5,
    MODEL_FLAG_SUBGRAPH_SKIPPED = 1u << 6,
    MODEL_FLAG_EXTERNAL_DATA = 1u << 7,
};

[[nodiscard]] constexpr bool
data_type_valid (const uint32_t value) noexcept
{
    return value <= static_cast<uint32_t> (DataType::FLOAT6E3M2);
}

/** @brief Bits per element, or 0 for UNDEFINED/STRING. */
[[nodiscard]] constexpr uint8_t
data_type_bits (const DataType type) noexcept
{
    switch (type)
        {
        case DataType::FLOAT:
        case DataType::INT32:
        case DataType::UINT32:
            return 32;
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::BOOL:
        case DataType::FLOAT8E4M3FN:
        case DataType::FLOAT8E4M3FNUZ:
        case DataType::FLOAT8E5M2:
        case DataType::FLOAT8E5M2FNUZ:
        case DataType::FLOAT8E8M0:
            return 8;
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::FLOAT16:
        case DataType::BFLOAT16:
            return 16;
        case DataType::INT64:
        case DataType::UINT64:
        case DataType::DOUBLE:
        case DataType::COMPLEX64:
            return 64;
        case DataType::COMPLEX128:
            return 128;
        case DataType::UINT4:
        case DataType::INT4:
        case DataType::FLOAT4E2M1:
            return 4;
        case DataType::UINT2:
        case DataType::INT2:
            return 2;
        case DataType::FLOAT6E2M3:
        case DataType::FLOAT6E3M2:
            return 6;
        case DataType::UNDEFINED:
        case DataType::STRING:
            return 0;
        }
    return 0;
}

/** @brief Byte width when the type is byte aligned; 0 for packed/string types. */
[[nodiscard]] constexpr uint8_t
data_type_byte_size (const DataType type) noexcept
{
    const uint8_t bits = data_type_bits (type);
    return (bits != 0u && (bits & 7u) == 0u) ? static_cast<uint8_t> (bits / 8u) : 0u;
}

/**
 * @brief Elements represented by one int32_data entry.
 *
 * ONNX stores one element per int32 entry even for sub-byte types: a 4-bit or
 * 2-bit value occupies only the low bits of its own int32.  Raw data is the
 * field that is bit-packed tightly.  The function remains so the packing rule
 * has a name at the call sites that tally typed values.
 */
[[nodiscard]] constexpr uint8_t
data_type_int32_pack (const DataType) noexcept
{
    return 1;
}

[[nodiscard]] constexpr bool
data_type_is_float (const DataType type) noexcept
{
    switch (type)
        {
        case DataType::FLOAT:
        case DataType::FLOAT16:
        case DataType::DOUBLE:
        case DataType::BFLOAT16:
        case DataType::FLOAT8E4M3FN:
        case DataType::FLOAT8E4M3FNUZ:
        case DataType::FLOAT8E5M2:
        case DataType::FLOAT8E5M2FNUZ:
        case DataType::FLOAT4E2M1:
        case DataType::FLOAT8E8M0:
        case DataType::FLOAT6E2M3:
        case DataType::FLOAT6E3M2:
            return true;
        default:
            return false;
        }
}

[[nodiscard]] constexpr bool
data_type_is_integer (const DataType type) noexcept
{
    switch (type)
        {
        case DataType::UINT8:
        case DataType::INT8:
        case DataType::UINT16:
        case DataType::INT16:
        case DataType::INT32:
        case DataType::INT64:
        case DataType::UINT32:
        case DataType::UINT64:
        case DataType::UINT4:
        case DataType::INT4:
        case DataType::UINT2:
        case DataType::INT2:
            return true;
        default:
            return false;
        }
}

/** @brief Map an ONNX tensor type onto the mllee weight codec type. */
[[nodiscard]] constexpr bool
to_mllee_dtype (const DataType type, Dtype &out) noexcept
{
    switch (type)
        {
        case DataType::INT8:
            out = Dtype::I8;
            return true;
        case DataType::INT4:
            out = Dtype::I4;
            return true;
        case DataType::INT2:
            out = Dtype::I2;
            return true;
        case DataType::FLOAT8E4M3FN:
            out = Dtype::F8E4M3;
            return true;
        case DataType::FLOAT8E5M2:
            out = Dtype::F8E5M2;
            return true;
        case DataType::FLOAT4E2M1:
            out = Dtype::F4E2M1;
            return true;
        default:
            return false;
        }
}

[[nodiscard]] constexpr const char *
data_type_name (const DataType type) noexcept
{
    switch (type)
        {
        case DataType::UNDEFINED:
            return "undefined";
        case DataType::FLOAT:
            return "float";
        case DataType::UINT8:
            return "uint8";
        case DataType::INT8:
            return "int8";
        case DataType::UINT16:
            return "uint16";
        case DataType::INT16:
            return "int16";
        case DataType::INT32:
            return "int32";
        case DataType::INT64:
            return "int64";
        case DataType::STRING:
            return "string";
        case DataType::BOOL:
            return "bool";
        case DataType::FLOAT16:
            return "float16";
        case DataType::DOUBLE:
            return "double";
        case DataType::UINT32:
            return "uint32";
        case DataType::UINT64:
            return "uint64";
        case DataType::COMPLEX64:
            return "complex64";
        case DataType::COMPLEX128:
            return "complex128";
        case DataType::BFLOAT16:
            return "bfloat16";
        case DataType::FLOAT8E4M3FN:
            return "float8e4m3fn";
        case DataType::FLOAT8E4M3FNUZ:
            return "float8e4m3fnuz";
        case DataType::FLOAT8E5M2:
            return "float8e5m2";
        case DataType::FLOAT8E5M2FNUZ:
            return "float8e5m2fnuz";
        case DataType::UINT4:
            return "uint4";
        case DataType::INT4:
            return "int4";
        case DataType::FLOAT4E2M1:
            return "float4e2m1";
        case DataType::FLOAT8E8M0:
            return "float8e8m0";
        case DataType::UINT2:
            return "uint2";
        case DataType::INT2:
            return "int2";
        case DataType::FLOAT6E2M3:
            return "float6e2m3";
        case DataType::FLOAT6E3M2:
            return "float6e3m2";
        }
    return "unknown";
}

[[nodiscard]] constexpr const char *
attribute_type_name (const AttributeType type) noexcept
{
    switch (type)
        {
        case AttributeType::UNDEFINED:
            return "undefined";
        case AttributeType::FLOAT:
            return "float";
        case AttributeType::INT:
            return "int";
        case AttributeType::STRING:
            return "string";
        case AttributeType::TENSOR:
            return "tensor";
        case AttributeType::GRAPH:
            return "graph";
        case AttributeType::FLOATS:
            return "floats";
        case AttributeType::INTS:
            return "ints";
        case AttributeType::STRINGS:
            return "strings";
        case AttributeType::TENSORS:
            return "tensors";
        case AttributeType::GRAPHS:
            return "graphs";
        case AttributeType::SPARSE_TENSOR:
            return "sparse_tensor";
        case AttributeType::SPARSE_TENSORS:
            return "sparse_tensors";
        case AttributeType::TYPE_PROTO:
            return "type_proto";
        case AttributeType::TYPE_PROTOS:
            return "type_protos";
        }
    return "unknown";
}
} // namespace zzs::mllee::onnx
