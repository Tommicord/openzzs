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
#include "onnx_event.hpp"
#include "std/fmt_sink.hpp"
#include "std/log.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/** @brief Counters a trace visitor keeps on the caller's stack. */
struct TraceStats
{
    uint32_t _M_events = 0;
    uint32_t _M_nodes = 0;
    uint32_t _M_tensors = 0;
    uint32_t _M_values = 0;
    uint32_t _M_attributes = 0;
    uint32_t _M_external = 0;
};

[[nodiscard]] constexpr const char *
event_name (const EventKind kind) noexcept
{
    switch (kind)
        {
        case EventKind::NONE:
            return "none";
        case EventKind::MODEL_BEGIN:
            return "model.begin";
        case EventKind::OPSET:
            return "opset";
        case EventKind::MODEL_END:
            return "model.end";
        case EventKind::GRAPH_BEGIN:
            return "graph.begin";
        case EventKind::GRAPH_END:
            return "graph.end";
        case EventKind::NODE_BEGIN:
            return "node.begin";
        case EventKind::NODE_INPUT:
            return "node.input";
        case EventKind::NODE_OUTPUT:
            return "node.output";
        case EventKind::NODE_END:
            return "node.end";
        case EventKind::ATTRIBUTE_BEGIN:
            return "attr.begin";
        case EventKind::ATTRIBUTE_ELEMENT:
            return "attr.element";
        case EventKind::ATTRIBUTE_END:
            return "attr.end";
        case EventKind::TENSOR_BEGIN:
            return "tensor.begin";
        case EventKind::TENSOR_DIM:
            return "tensor.dim";
        case EventKind::TENSOR_DATA:
            return "tensor.data";
        case EventKind::TENSOR_END:
            return "tensor.end";
        case EventKind::VALUE_BEGIN:
            return "value.begin";
        case EventKind::VALUE_DIM:
            return "value.dim";
        case EventKind::VALUE_END:
            return "value.end";
        case EventKind::EXTERNAL_DATA:
            return "external";
        case EventKind::DONE:
            return "done";
        }
    return "invalid";
}

/**
 * @brief A ready-made visitor that logs one line per event.
 *
 * The line is built in a fixed FrameSink, so tracing a model in a debug build
 * touches the same 96 bytes of stack for every event and never allocates.
 * Passing a TraceStats pointer is optional; when present it is updated so a
 * caller can assert that the walk saw the expected node/tensor counts.
 */
inline bool
trace_event (void *const context, const Event &event) noexcept
{
    TraceStats *const stats = static_cast<TraceStats *> (context);
    if (stats != nullptr)
        {
            ++stats->_M_events;
            switch (event._M_kind)
                {
                case EventKind::NODE_END:
                    ++stats->_M_nodes;
                    break;
                case EventKind::TENSOR_END:
                    ++stats->_M_tensors;
                    break;
                case EventKind::VALUE_END:
                    ++stats->_M_values;
                    break;
                case EventKind::ATTRIBUTE_END:
                    ++stats->_M_attributes;
                    break;
                case EventKind::EXTERNAL_DATA:
                    ++stats->_M_external;
                    break;
                default:
                    break;
                }
        }

    zzs::base::FrameSink<96> sink;
    sink.write ("onnx ");
    sink.write (event_name (event._M_kind));
    sink.write (" off=");
    zzs::base::write_uint (sink, event._M_offset, 0);
    sink.write (" val=");
    zzs::base::write_uint (sink, event._M_value, 0);
    sink.write (" len=");
    zzs::base::write_uint (sink, event._M_length, 0);
    zzs::base::log_debug (sink.view ());
    return true;
}

/** @brief Bind the trace visitor without naming the function type. */
[[nodiscard]] inline Visitor
trace_visitor (TraceStats *const stats = nullptr) noexcept
{
    return visitor (&trace_event, stats);
}
} // namespace zzs::mllee::onnx
