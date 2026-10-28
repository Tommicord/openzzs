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
#include "onnx_attribute.hpp"
#include "onnx_config.hpp"
#include "onnx_ops.hpp"
#include "onnx_reader.hpp"
#include "onnx_scope.hpp"
#include "onnx_string.hpp"
#include "onnx_wire.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/* Defined in onnx_attribute.hpp; the include above makes the definition
   visible, but the declaration keeps the two headers independently usable. */
[[nodiscard]] inline OnnxError
read_attribute (Context &context, const uint32_t limit,
                const uint32_t depth, const uint16_t index) noexcept;

/**
 * @brief Parse one NodeProto.
 *
 * Inputs and outputs are emitted one by one with their hashes and source spans;
 * the operator spelling is hashed into NodeInfo and resolved against the
 * compile-time registry after the message has streamed past.  Attributes are
 * parsed in place, so an attribute tensor is reported with role ATTRIBUTE and a
 * subgraph re-enters the graph parser rather than being buffered.
 */
[[nodiscard]] inline OnnxError
read_node (Context &context, const uint32_t limit,
           const uint32_t depth) noexcept
{
    Scope scope{};
    OnnxError error = open_scope (context._M_stream, limit, scope);
    if (error)
        return error;

    NodeInfo &node = context._M_node;
    node.reset ();
    const uint32_t message_offset = scope._M_end;

    if (!context.emit (EventKind::NODE_BEGIN, TensorRole::NONE,
                       message_offset, 0, &node))
        return stopped (context._M_stream);

    uint16_t attribute_index = 0;
    uint16_t input_index = 0;
    uint16_t output_index = 0;

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
                        StringInfo value{};
                        error = read_string (context._M_stream, scope._M_end,
                                             value);
                        if (error)
                            return error;
                        const EventKind kind
                            = tag._M_field == 1u ? EventKind::NODE_INPUT
                                                 : EventKind::NODE_OUTPUT;
                        const uint64_t hashes
                            = (static_cast<uint64_t> (value._M_hash2) << 32)
                              | value._M_hash;
                        const uint16_t index = tag._M_field == 1u ? input_index
                                                                  : output_index;
                        if (tag._M_field == 1u)
                            node.note_input ();
                        else
                            node.note_output ();
                        if (!context.emit (kind, TensorRole::NONE,
                                           value._M_offset, hashes, &node,
                                           value._M_length, index, 0))
                            return stopped (context._M_stream);
                        if (tag._M_field == 1u)
                            {
                                if (input_index != 0xFFFFu)
                                    ++input_index;
                            }
                        else if (output_index != 0xFFFFu)
                            ++output_index;
                        break;
                    }
                case 3:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        StringInfo name{};
                        error = read_string (context._M_stream, scope._M_end,
                                             name);
                        if (error)
                            return error;
                        node.set_name (name._M_hash, name._M_length);
                        break;
                    }
                case 4:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        StringInfo op{};
                        error = read_string (context._M_stream, scope._M_end,
                                             op);
                        if (error)
                            return error;
                        node.set_op (op._M_hash, op._M_hash2, op._M_length);
                        break;
                    }
                case 5:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        error = read_attribute (context, scope._M_end, depth,
                                                attribute_index);
                        if (error)
                            return error;
                        node.note_attribute ();
                        if (attribute_index != 0xFFFFu)
                            ++attribute_index;
                        break;
                    }
                case 7:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        StringInfo domain{};
                        error = read_string (context._M_stream, scope._M_end,
                                             domain);
                        if (error)
                            return error;
                        node.set_domain (domain._M_hash, domain._M_length);
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

    if ((node._M_flags & NODE_FLAG_HAS_OP) == 0u)
        return OnnxError (OnnxCode::BAD_GRAPH, scope._M_end);

    const OpBinding binding = lookup_op (node);
    if (binding._M_op == Op::COUNT)
        node._M_flags |= NODE_FLAG_UNKNOWN_OP;
    const uint64_t dispatch
        = static_cast<uint32_t> (binding._M_op)
          | (static_cast<uint32_t> (binding._M_act) << 16);

    if (!context.emit (EventKind::NODE_END, TensorRole::NONE, scope._M_end,
                       dispatch, &node, 0, 0, node._M_flags))
        return stopped (context._M_stream);
    return OnnxError::ok ();
}
} // namespace zzs::mllee::onnx
