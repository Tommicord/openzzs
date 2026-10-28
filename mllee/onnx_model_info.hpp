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
#include "onnx_config.hpp"
#include "onnx_types.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief The 32-byte ONNX model header retained by the loader.
 *
 * This is the only part of the loader that survives after `run()` and it is
 * deliberately exactly 32 bytes: IR/opset versions, the graph span, producer
 * identity and the graph counts.  Names and paths are hashed, not stored.
 */
struct ModelInfo
{
    uint32_t _M_ir_version = 0;
    uint32_t _M_opset_version = 0;
    uint32_t _M_graph_offset = 0;
    uint32_t _M_graph_length = 0;
    uint32_t _M_producer_hash = 0;
    uint16_t _M_node_count = 0;
    uint16_t _M_initializer_count = 0;
    uint16_t _M_input_count = 0;
    uint16_t _M_output_count = 0;
    uint16_t _M_producer_length = 0;
    uint8_t _M_flags = 0;
    uint8_t _M_loader_flags = 0;

    void
    reset () noexcept
    {
        this->_M_ir_version = 0;
        this->_M_opset_version = 0;
        this->_M_graph_offset = 0;
        this->_M_graph_length = 0;
        this->_M_producer_hash = 0;
        this->_M_node_count = 0;
        this->_M_initializer_count = 0;
        this->_M_input_count = 0;
        this->_M_output_count = 0;
        this->_M_producer_length = 0;
        this->_M_flags = 0;
        this->_M_loader_flags = 0;
    }

    [[nodiscard]] bool
    has_graph () const noexcept
    {
        return (this->_M_flags & MODEL_FLAG_HAS_GRAPH) != 0u;
    }

    void
    note_node () noexcept
    {
        this->_M_flags |= MODEL_FLAG_HAS_NODES;
        if (!saturating_increment (this->_M_node_count, static_cast<uint16_t> (0xFFFFu)))
            this->_M_flags |= MODEL_FLAG_COUNT_OVERFLOW;
    }

    void
    note_initializer () noexcept
    {
        this->_M_flags |= MODEL_FLAG_HAS_INITIALIZERS;
        if (!saturating_increment (this->_M_initializer_count,
                                    static_cast<uint16_t> (0xFFFFu)))
            this->_M_flags |= MODEL_FLAG_COUNT_OVERFLOW;
    }

    void
    note_input () noexcept
    {
        if (!saturating_increment (this->_M_input_count,
                                    static_cast<uint16_t> (0xFFFFu)))
            this->_M_flags |= MODEL_FLAG_COUNT_OVERFLOW;
    }

    void
    note_output () noexcept
    {
        if (!saturating_increment (this->_M_output_count,
                                    static_cast<uint16_t> (0xFFFFu)))
            this->_M_flags |= MODEL_FLAG_COUNT_OVERFLOW;
    }
};

static_assert (sizeof (ModelInfo) == _S_model_budget_bytes,
               "onnx::ModelInfo must stay 32 bytes");
} // namespace zzs::mllee::onnx
