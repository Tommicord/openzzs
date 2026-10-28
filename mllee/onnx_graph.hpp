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
#include "onnx_node.hpp"
#include "onnx_node_info.hpp"
#include "onnx_reader.hpp"
#include "onnx_scope.hpp"
#include "onnx_string.hpp"
#include "onnx_tensor.hpp"
#include "onnx_types.hpp"
#include "onnx_value.hpp"
#include "onnx_wire.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/* Defined in onnx_node.hpp, which includes this header through the attribute
   parser; the declaration keeps the mutual recursion well-formed regardless of
   which header is included first. */
[[nodiscard]] inline OnnxError
read_node (Context &context, const uint32_t limit,
           const uint32_t depth) noexcept;

/**
 * @brief Save the single resident NodeInfo across a nested graph.
 *
 * The loader deliberately owns exactly one 28-byte NodeInfo.  A subgraph in an
 * attribute would otherwise overwrite the node that is currently being parsed,
 * so the caller's copy is restored by this guard when the subgraph returns.
 * The guard lives on the stack, never in the loader object.
 */
class NodeInfoGuard
{
  public:
    NodeInfoGuard (NodeInfo &node, const bool active) noexcept
        : _M_node (node), _M_saved (node), _M_active (active)
    {
    }
    NodeInfoGuard (const NodeInfoGuard &) = delete;
    NodeInfoGuard &operator= (const NodeInfoGuard &) = delete;
    ~NodeInfoGuard () noexcept
    {
        if (this->_M_active)
            this->_M_node = this->_M_saved;
    }

  private:
    NodeInfo &_M_node;
    NodeInfo _M_saved;
    bool _M_active;
};

[[nodiscard]] inline OnnxError
read_graph_body (Context &context, const Scope &scope,
                 const uint32_t depth) noexcept
{
    if (depth > _S_max_message_depth)
        return OnnxError (OnnxCode::UNSUPPORTED, scope._M_end);

    /* The top-level graph leaves its last NodeInfo in the loader so the caller
       can inspect it; a nested graph must restore the node that was being
       parsed when the attribute re-entered the graph parser. */
    NodeInfoGuard guard (context._M_node, depth != 0u);
    const uint32_t graph_offset = context._M_stream.position ();
    StringInfo name{};
    uint16_t tensor_index = 0;

    if (!context.emit (EventKind::GRAPH_BEGIN, TensorRole::NONE, graph_offset,
                       0, nullptr))
        return stopped (context._M_stream);

    while (!scope_done (context._M_stream, scope))
        {
            Tag tag{};
            OnnxError error = read_tag (context._M_stream, scope._M_end, tag);
            if (error)
                return error;

            switch (tag._M_field)
                {
                case 1:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        error = read_node (context, scope._M_end, depth);
                        if (error)
                            return error;
                        if (depth == 0u)
                            context._M_model.note_node ();
                        break;
                    }
                case 2:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        error = read_string (context._M_stream, scope._M_end,
                                             name);
                        if (error)
                            return error;
                        break;
                    }
                case 5:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        error = read_tensor (context, scope._M_end,
                                             TensorRole::INITIALIZER,
                                             tensor_index);
                        if (error)
                            return error;
                        if (depth == 0u)
                            context._M_model.note_initializer ();
                        if (tensor_index != 0xFFFFu)
                            ++tensor_index;
                        break;
                    }
                case 10:
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
                case 11:
                case 12:
                case 13:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        const TensorRole role
                            = tag._M_field == 11u
                                  ? TensorRole::VALUE_INPUT
                                  : (tag._M_field == 12u
                                         ? TensorRole::VALUE_OUTPUT
                                         : TensorRole::VALUE_INFO);
                        error = read_value (context, scope._M_end, role);
                        if (error)
                            return error;
                        if (depth == 0u)
                            {
                                if (tag._M_field == 11u)
                                    context._M_model.note_input ();
                                else if (tag._M_field == 12u)
                                    context._M_model.note_output ();
                            }
                        break;
                    }
                case 15:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        if (!context.allow (LOADER_ALLOW_SPARSE))
                            {
                                if (context.strict ())
                                    return OnnxError (
                                        OnnxCode::UNSUPPORTED,
                                        context._M_stream.position ());
                                error = skip_value (context._M_stream,
                                                    scope._M_end, tag._M_wire);
                                if (error)
                                    return error;
                                break;
                            }
                        error = read_sparse_tensor (
                            context, scope._M_end,
                            TensorRole::SPARSE_INITIALIZER, tensor_index);
                        if (error)
                            return error;
                        if (depth == 0u)
                            context._M_model.note_initializer ();
                        if (tensor_index != 0xFFFFu)
                            ++tensor_index;
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

    if (!context.emit (EventKind::GRAPH_END, TensorRole::NONE, scope._M_end,
                       name._M_hash, nullptr, name._M_length, 0, 0))
        return stopped (context._M_stream);
    return OnnxError::ok ();
}

[[nodiscard]] inline OnnxError
read_graph (Context &context, const uint32_t limit,
            const uint32_t depth) noexcept
{
    Scope scope{};
    OnnxError error = open_scope (context._M_stream, limit, scope);
    if (error)
        return error;
    error = read_graph_body (context, scope, depth);
    if (error)
        return error;
    return close_scope (context._M_stream, scope);
}
} // namespace zzs::mllee::onnx
