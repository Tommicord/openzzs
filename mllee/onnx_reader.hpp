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
#include "onnx_event.hpp"
#include "onnx_model_info.hpp"
#include "onnx_node_info.hpp"
#include "onnx_stream.hpp"
#include "onnx_tensor_info.hpp"
#include "onnx_types.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief State shared by the structural parsers.
 *
 * A Context is created once per Loader::run() on the caller's stack and passed
 * by reference.  It owns no persistent state beyond references to the loader's
 * 128-byte object and the caller's Visitor, so adding a parser module does not
 * grow the resident footprint.
 */
struct Context
{
    Stream &_M_stream;
    ModelInfo &_M_model;
    NodeInfo &_M_node;
    TensorInfo &_M_tensor;
    Visitor _M_visitor;
    uint32_t _M_flags = 0;

    [[nodiscard]] bool
    allow (const LoaderFlag flag) const noexcept
    {
        return (this->_M_flags & static_cast<uint32_t> (flag)) != 0u;
    }

    [[nodiscard]] bool
    strict () const noexcept
    {
        return this->allow (LOADER_STRICT);
    }

    [[nodiscard]] bool
    emit (const EventKind kind, const TensorRole role, const uint32_t offset,
          const uint64_t value, const void *const info,
          const uint32_t length = 0, const uint16_t index = 0,
          const uint8_t flags = 0) const noexcept
    {
        Event event{};
        event._M_kind = kind;
        event._M_role = role;
        event._M_index = index;
        event._M_offset = offset;
        event._M_value = value;
        event._M_info = info;
        event._M_length = length;
        event._M_flags = flags;
        return this->_M_visitor.emit (event);
    }
};

/** @brief Emit a stop event when the visitor asked for it. */
[[nodiscard]] inline OnnxError
stopped (const Stream &stream) noexcept
{
    return OnnxError (OnnxCode::STOPPED, stream.position ());
}

[[nodiscard]] inline OnnxError
bad_field (const Stream &stream) noexcept
{
    return OnnxError (OnnxCode::BAD_FIELD, stream.position ());
}
} // namespace zzs::mllee::onnx
