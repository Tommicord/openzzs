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
#include "onnx_model_info.hpp"
#include "onnx_node_info.hpp"
#include "onnx_tensor_info.hpp"
#include "onnx_types.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/** @brief One streaming event emitted by the ONNX reader. */
enum class EventKind : uint8_t
{
    NONE = 0,
    MODEL_BEGIN,
    OPSET,
    MODEL_END,
    GRAPH_BEGIN,
    GRAPH_END,
    NODE_BEGIN,
    NODE_INPUT,
    NODE_OUTPUT,
    NODE_END,
    ATTRIBUTE_BEGIN,
    ATTRIBUTE_ELEMENT,
    ATTRIBUTE_END,
    TENSOR_BEGIN,
    TENSOR_DIM,
    TENSOR_DATA,
    TENSOR_END,
    VALUE_BEGIN,
    VALUE_DIM,
    VALUE_END,
    EXTERNAL_DATA,
    DONE,
};

/**
 * @brief A callback event, sized for the stack (32 bytes).
 *
 * `_M_info` points at the loader's ModelInfo/NodeInfo/TensorInfo, or at a
 * parser-local info struct.  The pointer is valid only for the duration of the
 * callback; a consumer that needs to retain data must copy it.  `_M_offset` and
 * `_M_length` identify string payloads in the source image so the consumer can
 * copy them with Loader::copy() without the loader buffering them.
 */
struct Event
{
    EventKind _M_kind = EventKind::NONE;
    TensorRole _M_role = TensorRole::NONE;
    uint16_t _M_index = 0;
    uint32_t _M_offset = 0;
    uint64_t _M_value = 0;
    const void *_M_info = nullptr;
    uint32_t _M_length = 0;
    uint8_t _M_flags = 0;
    uint8_t _M_reserved[3]{};

    [[nodiscard]] const ModelInfo *
    model () const noexcept
    {
        return static_cast<const ModelInfo *> (this->_M_info);
    }

    [[nodiscard]] const NodeInfo *
    node () const noexcept
    {
        return static_cast<const NodeInfo *> (this->_M_info);
    }

    [[nodiscard]] const TensorInfo *
    tensor () const noexcept
    {
        return static_cast<const TensorInfo *> (this->_M_info);
    }
};

static_assert (sizeof (Event) <= 32u, "onnx::Event exceeds its stack budget");

/** @brief Event sink. Returning false pauses the reader with OnnxCode::STOPPED. */
using EventFn = bool (*) (void *_M_context, const Event &_M_event) noexcept;

struct Visitor
{
    EventFn _M_on_event = nullptr;
    void *_M_context = nullptr;

    [[nodiscard]] bool
    emit (const Event &event) const noexcept
    {
        return this->_M_on_event == nullptr
                   ? true
                   : this->_M_on_event (this->_M_context, event);
    }
};

[[nodiscard]] inline Visitor
visitor (const EventFn function, void *const context = nullptr) noexcept
{
    return Visitor{ function, context };
}
} // namespace zzs::mllee::onnx
