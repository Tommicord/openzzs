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

#include "mllee/status.hpp"
#include "onnx_bounded.hpp"
#include "onnx_types.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief Rules shared by the TensorProto and AttributeProto parsers.
 *
 * Keeping the schema cross-checks here means the streaming parsers only deal
 * with wire framing while every "this field may not carry that data type" rule
 * lives in one constexpr table.  The functions return a bool instead of an
 * OnnxError because the caller knows which offset to attach to the rejection.
 */

/** @brief Whether `field` may be present for `type`. */
[[nodiscard]] constexpr bool
data_field_allowed (const DataType type, const TensorDataField field) noexcept
{
    switch (field)
        {
        case TensorDataField::NONE:
            return false;
        case TensorDataField::FLOAT_DATA:
            return type == DataType::FLOAT || type == DataType::COMPLEX64;
        case TensorDataField::DOUBLE_DATA:
            return type == DataType::DOUBLE || type == DataType::COMPLEX128;
        case TensorDataField::INT32_DATA:
            switch (type)
                {
                case DataType::INT32:
                case DataType::UINT8:
                case DataType::INT8:
                case DataType::UINT16:
                case DataType::INT16:
                case DataType::BOOL:
                case DataType::FLOAT16:
                case DataType::BFLOAT16:
                case DataType::FLOAT8E4M3FN:
                case DataType::FLOAT8E4M3FNUZ:
                case DataType::FLOAT8E5M2:
                case DataType::FLOAT8E5M2FNUZ:
                case DataType::UINT4:
                case DataType::INT4:
                case DataType::FLOAT4E2M1:
                case DataType::FLOAT8E8M0:
                case DataType::UINT2:
                case DataType::INT2:
                case DataType::FLOAT6E2M3:
                case DataType::FLOAT6E3M2:
                    return true;
                default:
                    return false;
                }
        case TensorDataField::INT64_DATA:
            return type == DataType::INT64;
        case TensorDataField::UINT64_DATA:
            return type == DataType::UINT64;
        case TensorDataField::STRING_DATA:
            return type == DataType::STRING;
        case TensorDataField::RAW_DATA:
            return type != DataType::UNDEFINED && type != DataType::STRING;
        case TensorDataField::EXTERNAL_DATA:
            return type != DataType::UNDEFINED && type != DataType::STRING;
        }
    return false;
}

/** @brief Number of scalar slots a typed field stores per tensor element. */
[[nodiscard]] constexpr bool
data_field_values (const TensorDataField field, const DataType type,
                   const uint64_t elements, uint64_t &out) noexcept
{
    switch (field)
        {
        case TensorDataField::FLOAT_DATA:
            return !mul_overflow_u64 (elements,
                                      type == DataType::COMPLEX64 ? 2u : 1u, out);
        case TensorDataField::DOUBLE_DATA:
            return !mul_overflow_u64 (elements,
                                      type == DataType::COMPLEX128 ? 2u : 1u, out);
        case TensorDataField::INT32_DATA:
        case TensorDataField::INT64_DATA:
        case TensorDataField::UINT64_DATA:
            out = elements;
            return true;
        case TensorDataField::NONE:
        case TensorDataField::STRING_DATA:
        case TensorDataField::RAW_DATA:
        case TensorDataField::EXTERNAL_DATA:
            return false;
        }
    return false;
}

/** @brief Fixed byte width of one value in a typed field, 0 if variable. */
[[nodiscard]] constexpr uint8_t
data_field_value_bytes (const TensorDataField field) noexcept
{
    switch (field)
        {
        case TensorDataField::FLOAT_DATA:
        case TensorDataField::INT32_DATA:
            return 4;
        case TensorDataField::DOUBLE_DATA:
        case TensorDataField::INT64_DATA:
        case TensorDataField::UINT64_DATA:
            return 8;
        case TensorDataField::NONE:
        case TensorDataField::STRING_DATA:
        case TensorDataField::RAW_DATA:
        case TensorDataField::EXTERNAL_DATA:
            return 0;
        }
    return 0;
}

/**
 * @brief Exact byte count of a typed payload.
 *
 * Returns false when the field is variable-width (string_data) or when the
 * value slot count would overflow, which the caller reports as BAD_TENSOR.
 */
[[nodiscard]] constexpr bool
typed_payload_bytes (const TensorDataField field, const DataType type,
                     const uint64_t elements, uint64_t &out) noexcept
{
    uint64_t values = 0;
    const uint8_t width = data_field_value_bytes (field);
    if (width == 0u || !data_field_values (field, type, elements, values))
        return false;
    return !mul_overflow_u64 (values, width, out);
}

/**
 * @brief Byte count of raw_data for a tensor shape.
 *
 * Sub-byte element types are bit-packed and rounded up to a byte, which is the
 * layout ONNX uses for int4/int2/float4/float6 weights.
 */
[[nodiscard]] constexpr bool
raw_payload_bytes (const DataType type, const uint64_t elements,
                   uint64_t &out) noexcept
{
    const uint8_t bits = data_type_bits (type);
    if (bits == 0u)
        return false;
    return packed_bytes (bits, elements, out);
}

/** @brief Whether an attribute payload field may carry this attribute type. */
[[nodiscard]] constexpr bool
attribute_field_allowed (const AttributeType type,
                         const uint32_t field) noexcept
{
    switch (field)
        {
        case 2:
            return type == AttributeType::FLOAT;
        case 3:
            return type == AttributeType::INT;
        case 4:
            return type == AttributeType::STRING;
        case 5:
            return type == AttributeType::TENSOR;
        case 6:
            return type == AttributeType::GRAPH;
        case 7:
            return type == AttributeType::FLOATS;
        case 8:
            return type == AttributeType::INTS;
        case 9:
            return type == AttributeType::STRINGS;
        case 10:
            return type == AttributeType::TENSORS;
        case 11:
            return type == AttributeType::GRAPHS;
        case 15:
            return type == AttributeType::TYPE_PROTOS;
        case 22:
            return type == AttributeType::SPARSE_TENSOR;
        case 23:
            return type == AttributeType::TYPE_PROTO;
        case 24:
            return type == AttributeType::SPARSE_TENSORS;
        default:
            return false;
        }
}

/** @brief Map an attribute payload field onto its AttributeType. */
[[nodiscard]] constexpr AttributeType
attribute_type_of_field (const uint32_t field) noexcept
{
    switch (field)
        {
        case 2:
            return AttributeType::FLOAT;
        case 3:
            return AttributeType::INT;
        case 4:
            return AttributeType::STRING;
        case 5:
            return AttributeType::TENSOR;
        case 6:
            return AttributeType::GRAPH;
        case 7:
            return AttributeType::FLOATS;
        case 8:
            return AttributeType::INTS;
        case 9:
            return AttributeType::STRINGS;
        case 10:
            return AttributeType::TENSORS;
        case 11:
            return AttributeType::GRAPHS;
        case 15:
            return AttributeType::TYPE_PROTOS;
        case 22:
            return AttributeType::SPARSE_TENSOR;
        case 23:
            return AttributeType::TYPE_PROTO;
        case 24:
            return AttributeType::SPARSE_TENSORS;
        default:
            return AttributeType::UNDEFINED;
        }
}
} // namespace zzs::mllee::onnx
