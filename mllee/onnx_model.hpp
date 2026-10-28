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
#include "onnx_graph.hpp"
#include "onnx_model_info.hpp"
#include "onnx_reader.hpp"
#include "onnx_scope.hpp"
#include "onnx_string.hpp"
#include "onnx_types.hpp"
#include "onnx_varint.hpp"
#include "onnx_wire.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief One OperatorSetIdProto header attached to an OPSET event.
 *
 * The domain is hashed, not stored.  A consumer that wants to confirm it saw
 * the default domain can compare `_M_domain_length` and the hash, or simply
 * read ModelInfo::_M_opset_version for the default domain.
 */
struct OpsetInfo
{
    uint32_t _M_domain_hash = 0;
    uint32_t _M_version = 0;
    uint16_t _M_domain_length = 0;
    uint8_t _M_flags = 0;
    uint8_t _M_reserved = 0;
};

static_assert (sizeof (OpsetInfo) <= 12u,
               "onnx::OpsetInfo exceeds its stack budget");

[[nodiscard]] inline bool
is_default_domain (const StringInfo &domain) noexcept
{
    return domain._M_length == 0u || string_is (domain, "ai.onnx");
}

[[nodiscard]] inline OnnxError
read_opset (Context &context, const uint32_t limit,
            const uint16_t index) noexcept
{
    Scope scope{};
    OnnxError error = open_scope (context._M_stream, limit, scope);
    if (error)
        return error;

    OpsetInfo info{};
    StringInfo domain{};
    bool has_version = false;
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
                        error = read_string (context._M_stream, scope._M_end,
                                             domain);
                        if (error)
                            return error;
                        break;
                    }
                case 2:
                    {
                        if (!tag.is_varint ())
                            return bad_field (context._M_stream);
                        int64_t version = 0;
                        error = read_int64 (context._M_stream, scope._M_end,
                                            version);
                        if (error)
                            return error;
                        if (version < 0
                            || static_cast<uint64_t> (version) > 0xFFFFFFFFull)
                            return OnnxError (OnnxCode::OVERFLOW,
                                              context._M_stream.position ());
                        info._M_version = static_cast<uint32_t> (version);
                        has_version = true;
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

    if (!has_version)
        return OnnxError (OnnxCode::BAD_MODEL, scope._M_end);
    info._M_domain_hash = domain._M_hash;
    info._M_domain_length = saturate_u16 (domain._M_length);
    if (is_default_domain (domain))
        {
            info._M_flags |= 1u;
            context._M_model._M_opset_version = info._M_version;
            if (info._M_version > _S_max_opset_version)
                {
                    context._M_model._M_flags |= MODEL_FLAG_OPSET_TOO_NEW;
                    return OnnxError (OnnxCode::UNSUPPORTED, scope._M_end);
                }
        }
    if (!context.emit (EventKind::OPSET, TensorRole::NONE, scope._M_end,
                       info._M_version, &info, info._M_domain_length, index,
                       info._M_flags))
        return stopped (context._M_stream);
    return OnnxError::ok ();
}

/**
 * @brief Parse the top-level ModelProto.
 *
 * The whole file is the protobuf message, so `limit` is normally
 * `Stream::size()`.  The graph span is recorded before its body is walked, which
 * is what lets a consumer stream the graph a second time or skip it later.  IR
 * and opset versions are rejected when they are newer than the constants in
 * onnx_config.hpp rather than being interpreted with older semantics.
 */
[[nodiscard]] inline OnnxError
read_model (Context &context, const uint32_t limit) noexcept
{
    ModelInfo &model = context._M_model;
    model.reset ();
    model._M_loader_flags = static_cast<uint8_t> (context._M_flags);

    if (!context.emit (EventKind::MODEL_BEGIN, TensorRole::NONE, 0, limit,
                       &model))
        return stopped (context._M_stream);

    uint16_t opset_index = 0;

    while (!scope_done (context._M_stream, Scope{ limit }))
        {
            Tag tag{};
            OnnxError error = read_tag (context._M_stream, limit, tag);
            if (error)
                return error;

            switch (tag._M_field)
                {
                case 1:
                    {
                        if (!tag.is_varint ())
                            return bad_field (context._M_stream);
                        uint64_t version = 0;
                        error = read_varint64 (context._M_stream, limit,
                                               version);
                        if (error)
                            return error;
                        if (version > 0xFFFFFFFFull)
                            return OnnxError (OnnxCode::OVERFLOW,
                                              context._M_stream.position ());
                        model._M_ir_version = static_cast<uint32_t> (version);
                        break;
                    }
                case 2:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        StringInfo producer{};
                        error = read_string (context._M_stream, limit,
                                             producer);
                        if (error)
                            return error;
                        model._M_producer_hash = producer._M_hash;
                        model._M_producer_length
                            = saturate_u16 (producer._M_length);
                        break;
                    }
                case 3:
                case 4:
                case 6:
                    {
                        StringInfo text{};
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        error = read_string (context._M_stream, limit, text);
                        if (error)
                            return error;
                        break;
                    }
                case 7:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        Scope graph{};
                        error = open_scope (context._M_stream, limit, graph);
                        if (error)
                            return error;
                        model._M_graph_offset
                            = context._M_stream.position ();
                        model._M_graph_length
                            = graph._M_end - model._M_graph_offset;
                        model._M_flags |= MODEL_FLAG_HAS_GRAPH;
                        error = read_graph_body (context, graph, 0u);
                        if (error)
                            return error;
                        error = close_scope (context._M_stream, graph);
                        if (error)
                            return error;
                        break;
                    }
                case 8:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        error = read_opset (context, limit, opset_index);
                        if (error)
                            return error;
                        if (opset_index != 0xFFFFu)
                            ++opset_index;
                        break;
                    }
                case 20:
                case 25:
                    {
                        /* training_info and functions carry secondary graphs.
                           They never change the inference graph that this
                           loader reports, so their bytes are skipped after the
                           length check instead of being buffered. */
                        error = skip_value (context._M_stream, limit,
                                            tag._M_wire);
                        if (error)
                            return error;
                        break;
                    }
                case 14:
                    {
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        error = skip_value (context._M_stream, limit,
                                            tag._M_wire);
                        if (error)
                            return error;
                        break;
                    }
                default:
                    error = skip_value (context._M_stream, limit,
                                        tag._M_wire);
                    if (error)
                        return error;
                    break;
                }
        }

    if (model._M_ir_version > _S_max_ir_version)
        {
            model._M_flags |= MODEL_FLAG_IR_TOO_NEW;
            return OnnxError (OnnxCode::UNSUPPORTED, context._M_stream.position ());
        }
    if ((model._M_flags & MODEL_FLAG_HAS_GRAPH) == 0u)
        return OnnxError (OnnxCode::BAD_MODEL, context._M_stream.position ());

    const uint32_t end = context._M_stream.position ();
    if (!context.emit (EventKind::MODEL_END, TensorRole::NONE, end,
                       model._M_node_count, &model, 0, 0, model._M_flags))
        return stopped (context._M_stream);
    if (!context.emit (EventKind::DONE, TensorRole::NONE, end, end, &model))
        return stopped (context._M_stream);
    return OnnxError::ok ();
}
} // namespace zzs::mllee::onnx
