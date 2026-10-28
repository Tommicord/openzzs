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
#include "onnx_external.hpp"
#include "onnx_reader.hpp"
#include "onnx_scope.hpp"
#include "onnx_string.hpp"
#include "onnx_tensor_info.hpp"
#include "onnx_types.hpp"
#include "onnx_validate.hpp"
#include "onnx_varint.hpp"
#include "onnx_wire.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief Add a consumed-byte count to a running total.
 *
 * Every length that reaches TensorInfo goes through this helper; a repeated
 * field with a hostile number of occurrences cannot wrap the reported span.
 */
[[nodiscard]] inline bool
accumulate_bytes (uint64_t &total, const uint32_t bytes) noexcept
{
    return !add_overflow_u64 (total, bytes, total);
}

/**
 * @brief Fold one dim into TensorInfo and emit TENSOR_DIM.
 *
 * Both the packed and the unpacked protobuf encodings call this, so a shape is
 * reported identically however the producer serialized the repeated field.
 */
[[nodiscard]] inline OnnxError
note_tensor_dim (Context &context, TensorInfo &tensor, const TensorRole role,
                 const int64_t dimension) noexcept
{
    if (dimension < 0)
        return OnnxError (OnnxCode::BAD_SHAPE, context._M_stream.position ());
    const uint16_t dim_index = static_cast<uint16_t> (tensor._M_rank);
    tensor.note_dim (dimension, false);
    if (!context.emit (EventKind::TENSOR_DIM, role, tensor._M_shape_hash,
                       static_cast<uint64_t> (dimension), &tensor, 0,
                       dim_index, 0))
        return stopped (context._M_stream);
    return OnnxError::ok ();
}

/**
 * @brief Parse one TensorProto, emitting TENSOR_BEGIN/DIM/DATA/END.
 *
 * The tensor is never buffered.  Dims are folded into TensorInfo with checked
 * multiplication, every payload span is reported as an absolute offset, and the
 * final TENSOR_END carries the shape hash and the saturated element count.  A
 * typed field must agree with data_type and with the shape product; raw_data
 * must be exactly the bit-packed size.  String tensors emit one TENSOR_DATA per
 * element because their payload is not contiguous.
 */
[[nodiscard]] inline OnnxError
read_tensor (Context &context, const uint32_t limit, const TensorRole role,
             const uint16_t index) noexcept
{
    TensorInfo &tensor = context._M_tensor;
    tensor.reset (role);

    Scope scope{};
    const uint32_t message_offset = context._M_stream.position ();
    OnnxError error = open_scope (context._M_stream, limit, scope);
    if (error)
        return error;

    StringInfo name{};
    TensorDataField payload = TensorDataField::NONE;
    uint64_t payload_bytes = 0;
    uint64_t payload_values = 0;
    uint64_t string_elements = 0;
    uint32_t string_first_offset = 0;
    uint32_t payload_occurrences = 0;
    uint64_t external_length = 0;
    uint8_t external_flags = 0;
    bool has_external_length = false;
    bool segmented = false;

    if (!context.emit (EventKind::TENSOR_BEGIN, role, message_offset, 0,
                       &tensor, 0, index, 0))
        return stopped (context._M_stream);

    while (!scope_done (context._M_stream, scope))
        {
            Tag tag{};
            error = read_tag (context._M_stream, scope._M_end, tag);
            if (error)
                return error;

            switch (tag._M_field)
                {
                case 1:
                    {
                        if (tag.is_length ())
                            {
                                /* onnx.proto3 serializes repeated int64 as a
                                   packed run; onnx.proto leaves it unpacked. */
                                Scope packed{};
                                error = open_scope (context._M_stream,
                                                    scope._M_end, packed);
                                if (error)
                                    return error;
                                while (!scope_done (context._M_stream, packed))
                                    {
                                        int64_t dimension = 0;
                                        error = read_int64 (context._M_stream,
                                                            packed._M_end,
                                                            dimension);
                                        if (error)
                                            return error;
                                        error = note_tensor_dim (context,
                                                                 tensor, role,
                                                                 dimension);
                                        if (error)
                                            return error;
                                    }
                                break;
                            }
                        if (!tag.is_varint ())
                            return bad_field (context._M_stream);
                        int64_t dimension = 0;
                        error = read_int64 (context._M_stream, scope._M_end,
                                            dimension);
                        if (error)
                            return error;
                        error = note_tensor_dim (context, tensor, role,
                                                 dimension);
                        if (error)
                            return error;
                        break;
                    }
                case 2:
                    {
                        if (!tag.is_varint ())
                            return bad_field (context._M_stream);
                        uint32_t dtype = 0;
                        error = read_u32 (context._M_stream, scope._M_end, dtype);
                        if (error)
                            return error;
                        if (!data_type_valid (dtype)
                            || dtype == static_cast<uint32_t> (
                                DataType::UNDEFINED))
                            return OnnxError (OnnxCode::BAD_TYPE,
                                              context._M_stream.position ());
                        tensor._M_dtype = static_cast<uint8_t> (dtype);
                        break;
                    }
                case 3:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        Scope segment{};
                        error = open_scope (context._M_stream, scope._M_end,
                                            segment);
                        if (error)
                            return error;
                        int64_t begin = 0;
                        int64_t end = 0;
                        while (!scope_done (context._M_stream, segment))
                            {
                                Tag inner{};
                                error = read_tag (context._M_stream,
                                                  segment._M_end, inner);
                                if (error)
                                    return error;
                                if (inner._M_field == 1u
                                    || inner._M_field == 2u)
                                    {
                                        if (!inner.is_varint ())
                                            return bad_field (context._M_stream);
                                        int64_t value = 0;
                                        error = read_int64 (
                                            context._M_stream, segment._M_end,
                                            value);
                                        if (error)
                                            return error;
                                        if (inner._M_field == 1u)
                                            begin = value;
                                        else
                                            end = value;
                                    }
                                else
                                    {
                                        error = skip_value (
                                            context._M_stream, segment._M_end,
                                            inner._M_wire);
                                        if (error)
                                            return error;
                                    }
                            }
                        if (begin < 0 || end < begin)
                            return OnnxError (OnnxCode::BAD_TENSOR,
                                              segment._M_end);
                        segmented = true;
                        tensor._M_flags |= TENSOR_FLAG_FRAGMENTED;
                        break;
                    }
                case 4:
                case 5:
                case 6:
                case 7:
                case 9:
                case 10:
                case 11:
                    {
                        const TensorDataField field = static_cast<TensorDataField> (
                            tag._M_field);
                        if (payload != TensorDataField::NONE && payload != field)
                            return OnnxError (OnnxCode::BAD_TENSOR,
                                              context._M_stream.position ());
                        payload = field;
                        tensor.set_data_field (payload);
                        ++payload_occurrences;

                        uint32_t occurrence_offset = 0;
                        uint32_t occurrence_length = 0;
                        if (field == TensorDataField::RAW_DATA)
                            {
                                if (!tag.is_length ())
                                    return bad_field (context._M_stream);
                                error = read_length (context._M_stream,
                                                     scope._M_end,
                                                     occurrence_length);
                                if (error)
                                    return error;
                                occurrence_offset
                                    = context._M_stream.position ();
                                error = context._M_stream.skip (
                                    scope._M_end, occurrence_length);
                                if (error)
                                    return error;
                                if (!accumulate_bytes (payload_bytes,
                                                       occurrence_length))
                                    return OnnxError (OnnxCode::OVERFLOW,
                                                      occurrence_offset);
                            }
                        else if (field == TensorDataField::STRING_DATA)
                            {
                                if (!tag.is_length ())
                                    return bad_field (context._M_stream);
                                StringInfo element{};
                                error = read_string (context._M_stream,
                                                     scope._M_end, element);
                                if (error)
                                    return error;
                                if (string_elements == 0u)
                                    string_first_offset = element._M_offset;
                                if (!saturating_increment (
                                        string_elements,
                                        ~static_cast<uint64_t> (0u)))
                                    return OnnxError (OnnxCode::OVERFLOW,
                                                      element._M_offset);
                                tensor._M_flags
                                    |= TENSOR_FLAG_FRAGMENTED;
                                occurrence_offset = element._M_offset;
                                occurrence_length = element._M_length;
                                if (!context.emit (
                                        EventKind::TENSOR_DATA, role,
                                        occurrence_offset, string_elements - 1u,
                                        &tensor, occurrence_length,
                                        static_cast<uint16_t> (
                                            string_elements - 1u),
                                        static_cast<uint8_t> (field)))
                                    return stopped (context._M_stream);
                            }
                        else if (field == TensorDataField::FLOAT_DATA
                                 || field == TensorDataField::DOUBLE_DATA)
                            {
                                const uint8_t width
                                    = field == TensorDataField::FLOAT_DATA
                                          ? 4u
                                          : 8u;
                                if (tag.is_length ())
                                    {
                                        Scope packed{};
                                        error = open_scope (
                                            context._M_stream, scope._M_end,
                                            packed);
                                        if (error)
                                            return error;
                                        occurrence_offset
                                            = context._M_stream.position ();
                                        occurrence_length = packed._M_end
                                                            - occurrence_offset;
                                        if (occurrence_length % width != 0u)
                                            return OnnxError (
                                                OnnxCode::BAD_TENSOR,
                                                occurrence_offset);
                                        error = context._M_stream.skip (
                                            packed._M_end,
                                            occurrence_length);
                                        if (error)
                                            return error;
                                        payload_values
                                            += occurrence_length / width;
                                    }
                                else if ((field
                                              == TensorDataField::FLOAT_DATA
                                          && tag._M_wire
                                                 == WireType::FIXED32)
                                         || (field
                                                 == TensorDataField::DOUBLE_DATA
                                             && tag._M_wire
                                                    == WireType::FIXED64))
                                    {
                                        occurrence_offset
                                            = context._M_stream.position ();
                                        if (field
                                            == TensorDataField::FLOAT_DATA)
                                            {
                                                uint32_t bits = 0;
                                                error = read_fixed32 (
                                                    context._M_stream,
                                                    scope._M_end, bits);
                                            }
                                        else
                                            {
                                                uint64_t bits = 0;
                                                error = read_fixed64 (
                                                    context._M_stream,
                                                    scope._M_end, bits);
                                            }
                                        if (error)
                                            return error;
                                        occurrence_length = width;
                                        ++payload_values;
                                    }
                                else
                                    return bad_field (context._M_stream);
                                if (!accumulate_bytes (payload_bytes,
                                                       occurrence_length))
                                    return OnnxError (OnnxCode::OVERFLOW,
                                                      occurrence_offset);
                            }
                        else
                            {
                                if (tag.is_length ())
                                    {
                                        Scope packed{};
                                        error = open_scope (
                                            context._M_stream, scope._M_end,
                                            packed);
                                        if (error)
                                            return error;
                                        occurrence_offset
                                            = context._M_stream.position ();
                                        occurrence_length = packed._M_end
                                                            - occurrence_offset;
                                        while (!scope_done (context._M_stream,
                                                            packed))
                                            {
                                                uint64_t ignored = 0;
                                                error = read_varint64 (
                                                    context._M_stream,
                                                    packed._M_end, ignored);
                                                if (error)
                                                    return error;
                                                if (!saturating_increment (
                                                        payload_values,
                                                        ~static_cast<uint64_t> (
                                                            0u)))
                                                    return OnnxError (
                                                        OnnxCode::OVERFLOW,
                                                        occurrence_offset);
                                            }
                                    }
                                else if (tag.is_varint ())
                                    {
                                        occurrence_offset
                                            = context._M_stream.position ();
                                        uint64_t ignored = 0;
                                        error = read_varint64 (
                                            context._M_stream, scope._M_end,
                                            ignored);
                                        if (error)
                                            return error;
                                        occurrence_length
                                            = context._M_stream.position ()
                                              - occurrence_offset;
                                        ++payload_values;
                                    }
                                else
                                    return bad_field (context._M_stream);
                                if (!accumulate_bytes (payload_bytes,
                                                       occurrence_length))
                                    return OnnxError (OnnxCode::OVERFLOW,
                                                      occurrence_offset);
                            }

                        if (field != TensorDataField::STRING_DATA)
                            {
                                if (payload_occurrences == 1u)
                                    tensor._M_data_offset = occurrence_offset;
                                tensor._M_data_length = saturate_u32 (
                                    static_cast<uint64_t> (
                                        tensor._M_data_length)
                                    + occurrence_length);
                                if (!context.emit (
                                        EventKind::TENSOR_DATA, role,
                                        occurrence_offset,
                                        payload_occurrences - 1u, &tensor,
                                        occurrence_length,
                                        static_cast<uint16_t> (
                                            payload_occurrences - 1u),
                                        static_cast<uint8_t> (field)))
                                    return stopped (context._M_stream);
                            }
                        break;
                    }
                case 8:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        error = read_string (context._M_stream, scope._M_end,
                                             name);
                        if (error)
                            return error;
                        break;
                    }
                case 12:
                    {
                        StringInfo doc{};
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        error = read_string (context._M_stream, scope._M_end,
                                             doc);
                        if (error)
                            return error;
                        break;
                    }
                case 13:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        ExternalInfo external{};
                        error = read_external (context, scope._M_end, role,
                                               external);
                        if (error)
                            return error;
                        external_flags |= external._M_flags;
                        if ((external._M_flags & EXTERNAL_FLAG_LENGTH) != 0u)
                            {
                                external_length = external._M_length_value;
                                has_external_length = true;
                            }
                        break;
                    }
                case 14:
                    {
                        if (!tag.is_varint ())
                            return bad_field (context._M_stream);
                        uint32_t location = 0;
                        error = read_u32 (context._M_stream, scope._M_end,
                                          location);
                        if (error)
                            return error;
                        if (location > 1u)
                            return OnnxError (OnnxCode::BAD_TENSOR,
                                              context._M_stream.position ());
                        tensor.set_data_location (
                            static_cast<uint8_t> (location));
                        break;
                    }
                default:
                    error = skip_value (context._M_stream, scope._M_end,
                                        tag._M_wire);
                    if (error)
                        return error;
                    break;
                }
        }

    if ((tensor._M_flags & (TENSOR_FLAG_OVERFLOW | TENSOR_FLAG_INVALID))
        != 0u)
        return OnnxError (OnnxCode::OVERFLOW, scope._M_end);
    if (tensor._M_dtype == static_cast<uint8_t> (DataType::UNDEFINED))
        return OnnxError (OnnxCode::BAD_TYPE, scope._M_end);

    tensor.finish_shape ();
    const DataType dtype = tensor.dtype ();
    const uint64_t elements = tensor._M_elements;
    tensor._M_name_hash = name._M_hash;
    tensor._M_name_length = saturate_u16 (name._M_length);
    tensor.set_data_field (payload);

    const bool external_declared
        = tensor.data_location () == 1u
          || (external_flags & (EXTERNAL_FLAG_LOCATION | EXTERNAL_FLAG_OFFSET
                                | EXTERNAL_FLAG_LENGTH | EXTERNAL_FLAG_CHECKSUM))
                 != 0u;
    if (external_declared)
        {
            if (!context.allow (LOADER_ALLOW_EXTERNAL))
                return OnnxError (OnnxCode::UNSUPPORTED, scope._M_end);
            if ((external_flags & EXTERNAL_FLAG_LOCATION) == 0u)
                return OnnxError (OnnxCode::EXTERNAL_DATA, scope._M_end);
            if (payload != TensorDataField::NONE)
                return OnnxError (OnnxCode::BAD_TENSOR, scope._M_end);
            if (!data_field_allowed (dtype, TensorDataField::EXTERNAL_DATA))
                return OnnxError (OnnxCode::BAD_TENSOR, scope._M_end);
            uint64_t expected = 0;
            if (!segmented && raw_payload_bytes (dtype, elements, expected))
                {
                    if (has_external_length && external_length != expected)
                        return OnnxError (OnnxCode::BAD_TENSOR, scope._M_end);
                }
            tensor._M_flags |= TENSOR_FLAG_EXTERNAL;
            tensor._M_data_offset = 0;
            tensor._M_data_length = saturate_u32 (external_length);
            if (!context.emit (EventKind::TENSOR_DATA, role, 0, 0, &tensor,
                               tensor._M_data_length, index,
                               static_cast<uint8_t> (
                                   TensorDataField::EXTERNAL_DATA)))
                return stopped (context._M_stream);
        }
    else if (payload == TensorDataField::NONE)
        {
            /* No payload means every element is zero, which is legal for an
               optional tensor and for an initializer that relies on the
               default.  Nothing to report but the header. */
            tensor._M_data_offset = 0;
            tensor._M_data_length = 0;
        }
    else if (payload == TensorDataField::STRING_DATA)
        {
            if (dtype != DataType::STRING)
                return OnnxError (OnnxCode::BAD_TENSOR, scope._M_end);
            if (!segmented && string_elements != elements)
                return OnnxError (OnnxCode::BAD_TENSOR, scope._M_end);
            tensor._M_data_offset = string_first_offset;
            tensor._M_data_length = 0;
            tensor._M_flags |= TENSOR_FLAG_TYPED_DATA;
        }
    else if (payload == TensorDataField::RAW_DATA)
        {
            if (!data_field_allowed (dtype, payload))
                return OnnxError (OnnxCode::BAD_TENSOR, scope._M_end);
            uint64_t expected = 0;
            if (!raw_payload_bytes (dtype, elements, expected))
                return OnnxError (OnnxCode::BAD_TENSOR, scope._M_end);
            if (!segmented && payload_bytes != expected)
                return OnnxError (OnnxCode::BAD_TENSOR, scope._M_end);
            tensor._M_flags |= TENSOR_FLAG_RAW_DATA;
        }
    else
        {
            if (!data_field_allowed (dtype, payload))
                return OnnxError (OnnxCode::BAD_TENSOR, scope._M_end);
            uint64_t expected_values = 0;
            uint64_t expected_bytes = 0;
            if (!data_field_values (payload, dtype, elements, expected_values)
                || !typed_payload_bytes (payload, dtype, elements,
                                         expected_bytes))
                return OnnxError (OnnxCode::BAD_TENSOR, scope._M_end);
            /* int32_data/int64_data/uint64_data hold one varint per value, so
               the wire byte count varies with the magnitude; only the value
               count can be checked there.  float_data/double_data are packed
               fixed-width fields, where the byte count is exact too. */
            const bool fixed_width = payload == TensorDataField::FLOAT_DATA
                                     || payload == TensorDataField::DOUBLE_DATA;
            if (!segmented
                && (payload_values != expected_values
                    || (fixed_width && payload_bytes != expected_bytes)))
                return OnnxError (OnnxCode::BAD_TENSOR, scope._M_end);
            tensor._M_flags |= TENSOR_FLAG_TYPED_DATA;
        }

    if (!context.emit (EventKind::TENSOR_END, role, scope._M_end,
                       tensor._M_elements, &tensor, tensor._M_data_length,
                       index, tensor._M_flags))
        return stopped (context._M_stream);
    return OnnxError::ok ();
}

/**
 * @brief Parse SparseTensorProto without materializing its dense form.
 *
 * The values and indices tensors are ordinary TensorProto messages and are
 * reported through the same event stream; the dense dims are checked for
 * negative values and otherwise left to the consumer, which knows how to map
 * sparse coordinates onto its own layout.
 */
[[nodiscard]] inline OnnxError
read_sparse_tensor (Context &context, const uint32_t limit,
                    const TensorRole role, const uint16_t index) noexcept
{
    Scope scope{};
    OnnxError error = open_scope (context._M_stream, limit, scope);
    if (error)
        return error;
    uint32_t ordinal = 0;
    while (!scope_done (context._M_stream, scope))
        {
            Tag tag{};
            error = read_tag (context._M_stream, scope._M_end, tag);
            if (error)
                return error;
            switch (tag._M_field)
                {
                case 1:
                case 2:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        error = read_tensor (context, scope._M_end,
                                             TensorRole::SPARSE_INITIALIZER,
                                             index);
                        if (error)
                            return error;
                        ++ordinal;
                        break;
                    }
                case 3:
                    {
                        if (tag.is_length ())
                            {
                                Scope packed{};
                                error = open_scope (context._M_stream,
                                                    scope._M_end, packed);
                                if (error)
                                    return error;
                                while (!scope_done (context._M_stream, packed))
                                    {
                                        int64_t dimension = 0;
                                        error = read_int64 (context._M_stream,
                                                            packed._M_end,
                                                            dimension);
                                        if (error)
                                            return error;
                                        if (dimension < 0)
                                            return OnnxError (
                                                OnnxCode::BAD_SHAPE,
                                                context._M_stream.position ());
                                    }
                                break;
                            }
                        if (!tag.is_varint ())
                            return bad_field (context._M_stream);
                        int64_t dimension = 0;
                        error = read_int64 (context._M_stream, scope._M_end,
                                            dimension);
                        if (error)
                            return error;
                        if (dimension < 0)
                            return OnnxError (OnnxCode::BAD_SHAPE,
                                              context._M_stream.position ());
                        break;
                    }
                default:
                    error = skip_value (context._M_stream, scope._M_end,
                                        tag._M_wire);
                    if (error)
                        return error;
                    break;
                }
        }
    (void)role;
    (void)ordinal;
    return close_scope (context._M_stream, scope);
}
} // namespace zzs::mllee::onnx
