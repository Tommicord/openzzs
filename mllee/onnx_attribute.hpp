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
#include "onnx_config.hpp"
#include "onnx_reader.hpp"
#include "onnx_scope.hpp"
#include "onnx_shape.hpp"
#include "onnx_string.hpp"
#include "onnx_tensor.hpp"
#include "onnx_types.hpp"
#include "onnx_validate.hpp"
#include "onnx_varint.hpp"
#include "onnx_wire.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/* Defined in onnx_graph.hpp; an attribute may carry a subgraph and a subgraph
   may carry nodes with attributes, so the two parsers are mutually recursive
   and each declares the other before use. */
[[nodiscard]] inline OnnxError
read_graph (Context &context, const uint32_t limit,
            const uint32_t depth) noexcept;

enum AttributeFlag : uint8_t
{
    ATTRIBUTE_FLAG_HAS_NAME = 1u << 0,
    ATTRIBUTE_FLAG_REFERENCE = 1u << 1,
    ATTRIBUTE_FLAG_EMPTY = 1u << 2,
};

/**
 * @brief The small header emitted with ATTRIBUTE_BEGIN/END.
 *
 * Names are hashed like every other string; an attribute payload is delivered
 * as a sequence of ATTRIBUTE_ELEMENT events whose `_M_value` is the scalar or
 * the payload hash and whose `_M_offset`/`_M_length` locate a string or an
 * embedded message.  The loader owns no attribute name buffer.
 */
struct AttributeInfo
{
    uint32_t _M_name_hash = 0;
    uint32_t _M_name_hash2 = 0;
    uint32_t _M_count = 0;
    uint16_t _M_name_length = 0;
    uint8_t _M_type = 0;
    uint8_t _M_flags = 0;
};

static_assert (sizeof (AttributeInfo) <= 16u,
               "onnx::AttributeInfo exceeds its stack budget");

[[nodiscard]] inline bool
emit_element (Context &context, const AttributeInfo &info,
              const uint32_t offset, const uint64_t value,
              const uint32_t length, const uint16_t index,
              const AttributeType type) noexcept
{
    return context.emit (EventKind::ATTRIBUTE_ELEMENT, TensorRole::ATTRIBUTE,
                         offset, value, &info, length, index,
                         static_cast<uint8_t> (type));
}

[[nodiscard]] inline OnnxError
skip_subgraph (Context &context, const uint32_t limit, const Tag &tag,
               const char *const reason) noexcept
{
    (void)reason;
    if (context.strict ())
        return OnnxError (OnnxCode::UNSUPPORTED, context._M_stream.position ());
    context._M_model._M_flags |= MODEL_FLAG_SUBGRAPH_SKIPPED;
    return skip_value (context._M_stream, limit, tag._M_wire);
}

/**
 * @brief Parse one AttributeProto.
 *
 * Scalar attributes emit one element event.  Repeated numeric attributes emit
 * one element per value, packed or unpacked.  Strings are hashed and only their
 * spans are retained.  Tensors and subgraphs recurse through the same streaming
 * parser, which is why the depth parameter exists: an ONNX file can nest a
 * graph inside an attribute inside a node without a fixed bound otherwise.
 */
[[nodiscard]] inline OnnxError
read_attribute (Context &context, const uint32_t limit,
                const uint32_t depth, const uint16_t index) noexcept
{
    Scope scope{};
    OnnxError error = open_scope (context._M_stream, limit, scope);
    if (error)
        return error;

    AttributeInfo info{};
    uint32_t payload_field = 0;
    AttributeType declared = AttributeType::UNDEFINED;
    uint16_t element_index = 0;

    if (!context.emit (EventKind::ATTRIBUTE_BEGIN, TensorRole::ATTRIBUTE,
                       scope._M_end, 0, &info, 0, index, 0))
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
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        StringInfo name{};
                        error = read_string (context._M_stream, scope._M_end,
                                             name);
                        if (error)
                            return error;
                        info._M_name_hash = name._M_hash;
                        info._M_name_hash2 = name._M_hash2;
                        info._M_name_length = saturate_u16 (name._M_length);
                        info._M_flags |= ATTRIBUTE_FLAG_HAS_NAME;
                        break;
                    }
                case 13:
                case 21:
                    {
                        StringInfo text{};
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        error = read_string (context._M_stream, scope._M_end,
                                             text);
                        if (error)
                            return error;
                        if (tag._M_field == 21u)
                            info._M_flags |= ATTRIBUTE_FLAG_REFERENCE;
                        break;
                    }
                case 20:
                    {
                        if (!tag.is_varint ())
                            return bad_field (context._M_stream);
                        uint32_t type = 0;
                        error = read_u32 (context._M_stream, scope._M_end,
                                          type);
                        if (error)
                            return error;
                        if (type > static_cast<uint32_t> (
                                       AttributeType::TYPE_PROTOS))
                            return OnnxError (OnnxCode::BAD_ATTRIBUTE,
                                              context._M_stream.position ());
                        declared = static_cast<AttributeType> (type);
                        break;
                    }
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 15:
                case 22:
                case 23:
                case 24:
                    {
                        if (payload_field != 0u
                            && payload_field != tag._M_field)
                            return OnnxError (OnnxCode::BAD_ATTRIBUTE,
                                              context._M_stream.position ());
                        payload_field = tag._M_field;
                        const AttributeType implied
                            = attribute_type_of_field (tag._M_field);
                        if (declared != AttributeType::UNDEFINED
                            && declared != implied)
                            return OnnxError (OnnxCode::BAD_ATTRIBUTE,
                                              context._M_stream.position ());

                        switch (tag._M_field)
                            {
                            case 2:
                                {
                                    if (tag._M_wire != WireType::FIXED32)
                                        return bad_field (context._M_stream);
                                    uint32_t bits = 0;
                                    error = read_fixed32 (context._M_stream,
                                                          scope._M_end, bits);
                                    if (error)
                                        return error;
                                    ++info._M_count;
                                    if (!emit_element (context, info,
                                                       context._M_stream
                                                           .position ()
                                                           - 4u,
                                                       bits, 0, element_index++,
                                                       implied))
                                        return stopped (context._M_stream);
                                    break;
                                }
                            case 3:
                                {
                                    if (!tag.is_varint ())
                                        return bad_field (context._M_stream);
                                    const uint32_t offset
                                        = context._M_stream.position ();
                                    int64_t value = 0;
                                    error = read_int64 (context._M_stream,
                                                        scope._M_end, value);
                                    if (error)
                                        return error;
                                    ++info._M_count;
                                    if (!emit_element (
                                            context, info, offset,
                                            static_cast<uint64_t> (value), 0,
                                            element_index++, implied))
                                        return stopped (context._M_stream);
                                    break;
                                }
                            case 4:
                            case 9:
                                {
                                    if (!tag.is_length ())
                                        return bad_field (context._M_stream);
                                    StringInfo text{};
                                    error = read_string (context._M_stream,
                                                         scope._M_end, text);
                                    if (error)
                                        return error;
                                    ++info._M_count;
                                    if (!emit_element (
                                            context, info, text._M_offset,
                                            text._M_hash, text._M_length,
                                            element_index++, implied))
                                        return stopped (context._M_stream);
                                    break;
                                }
                            case 5:
                            case 10:
                                {
                                    if (!tag.is_length ())
                                        return bad_field (context._M_stream);
                                    error = read_tensor (
                                        context, scope._M_end,
                                        TensorRole::ATTRIBUTE, element_index);
                                    if (error)
                                        return error;
                                    ++info._M_count;
                                    ++element_index;
                                    break;
                                }
                            case 6:
                            case 11:
                                {
                                    if (!tag.is_length ())
                                        return bad_field (context._M_stream);
                                    if (!context.allow (
                                            LOADER_ALLOW_SUBGRAPHS))
                                        {
                                            error = skip_subgraph (
                                                context, scope._M_end, tag,
                                                "subgraph");
                                            if (error)
                                                return error;
                                            ++info._M_count;
                                            ++element_index;
                                            break;
                                        }
                                    if (depth >= _S_max_message_depth)
                                        return OnnxError (
                                            OnnxCode::UNSUPPORTED,
                                            context._M_stream.position ());
                                    error = read_graph (context, scope._M_end,
                                                        depth + 1u);
                                    if (error)
                                        return error;
                                    ++info._M_count;
                                    ++element_index;
                                    break;
                                }
                            case 7:
                                {
                                    if (tag._M_wire == WireType::FIXED32)
                                        {
                                            uint32_t bits = 0;
                                            error = read_fixed32 (
                                                context._M_stream,
                                                scope._M_end, bits);
                                            if (error)
                                                return error;
                                            ++info._M_count;
                                            if (!emit_element (
                                                    context, info,
                                                    context._M_stream
                                                            .position ()
                                                        - 4u,
                                                    bits, 0,
                                                    element_index++, implied))
                                                return stopped (
                                                    context._M_stream);
                                        }
                                    else if (tag.is_length ())
                                        {
                                            Scope packed{};
                                            error = open_scope (
                                                context._M_stream,
                                                scope._M_end, packed);
                                            if (error)
                                                return error;
                                            while (!scope_done (
                                                context._M_stream, packed))
                                                {
                                                    const uint32_t offset
                                                        = context._M_stream
                                                              .position ();
                                                    uint32_t bits = 0;
                                                    error = read_fixed32 (
                                                        context._M_stream,
                                                        packed._M_end, bits);
                                                    if (error)
                                                        return error;
                                                    ++info._M_count;
                                                    if (!emit_element (
                                                            context, info,
                                                            offset, bits, 0,
                                                            element_index++,
                                                            implied))
                                                        return stopped (
                                                            context
                                                                ._M_stream);
                                                }
                                        }
                                    else
                                        return bad_field (
                                            context._M_stream);
                                    break;
                                }
                            case 8:
                                {
                                    if (tag._M_wire == WireType::VARINT)
                                        {
                                            const uint32_t offset
                                                = context._M_stream.position ();
                                            int64_t value = 0;
                                            error = read_int64 (
                                                context._M_stream,
                                                scope._M_end, value);
                                            if (error)
                                                return error;
                                            ++info._M_count;
                                            if (!emit_element (
                                                    context, info, offset,
                                                    static_cast<uint64_t> (
                                                        value),
                                                    0, element_index++,
                                                    implied))
                                                return stopped (
                                                    context._M_stream);
                                        }
                                    else if (tag.is_length ())
                                        {
                                            Scope packed{};
                                            error = open_scope (
                                                context._M_stream,
                                                scope._M_end, packed);
                                            if (error)
                                                return error;
                                            while (!scope_done (
                                                context._M_stream, packed))
                                                {
                                                    const uint32_t offset
                                                        = context._M_stream
                                                              .position ();
                                                    int64_t value = 0;
                                                    error = read_int64 (
                                                        context._M_stream,
                                                        packed._M_end, value);
                                                    if (error)
                                                        return error;
                                                    ++info._M_count;
                                                    if (!emit_element (
                                                            context, info,
                                                            offset,
                                                            static_cast<
                                                                uint64_t> (
                                                                value),
                                                            0,
                                                            element_index++,
                                                            implied))
                                                        return stopped (
                                                            context
                                                                ._M_stream);
                                                }
                                        }
                                    else
                                        return bad_field (
                                            context._M_stream);
                                    break;
                                }
                            case 15:
                            case 23:
                                {
                                    if (!tag.is_length ())
                                        return bad_field (context._M_stream);
                                    TensorInfo type_info{};
                                    type_info.reset (TensorRole::ATTRIBUTE);
                                    bool has_type = false;
                                    error = read_type (context, scope._M_end,
                                                       type_info, has_type);
                                    if (error)
                                        return error;
                                    type_info.finish_shape ();
                                    ++info._M_count;
                                    if (!emit_element (
                                            context, info,
                                            context._M_stream.position (),
                                            type_info._M_elements,
                                            type_info._M_dtype,
                                            element_index++, implied))
                                        return stopped (context._M_stream);
                                    break;
                                }
                            case 22:
                            case 24:
                                {
                                    if (!tag.is_length ())
                                        return bad_field (context._M_stream);
                                    if (!context.allow (LOADER_ALLOW_SPARSE))
                                        {
                                            if (context.strict ())
                                                return OnnxError (
                                                    OnnxCode::UNSUPPORTED,
                                                    context._M_stream
                                                        .position ());
                                            error = skip_value (
                                                context._M_stream,
                                                scope._M_end, tag._M_wire);
                                            if (error)
                                                return error;
                                            ++info._M_count;
                                            ++element_index;
                                            break;
                                        }
                                    error = read_sparse_tensor (
                                        context, scope._M_end,
                                        TensorRole::SPARSE_INITIALIZER,
                                        element_index);
                                    if (error)
                                        return error;
                                    ++info._M_count;
                                    ++element_index;
                                    break;
                                }
                            default:
                                break;
                            }
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

    if (declared != AttributeType::UNDEFINED && payload_field != 0u
        && !attribute_field_allowed (declared, payload_field))
        return OnnxError (OnnxCode::BAD_ATTRIBUTE, scope._M_end);
    if (declared == AttributeType::UNDEFINED)
        declared = attribute_type_of_field (payload_field);
    info._M_type = static_cast<uint8_t> (declared);
    if (info._M_count == 0u)
        info._M_flags |= ATTRIBUTE_FLAG_EMPTY;

    if (!context.emit (EventKind::ATTRIBUTE_END, TensorRole::ATTRIBUTE,
                       scope._M_end, info._M_count, &info, 0, index,
                       info._M_type))
        return stopped (context._M_stream);
    return OnnxError::ok ();
}
} // namespace zzs::mllee::onnx
