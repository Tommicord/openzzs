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
#include "onnx_chunk.hpp"
#include "onnx_config.hpp"
#include "onnx_event.hpp"
#include "onnx_model.hpp"
#include "onnx_model_info.hpp"
#include "onnx_node_info.hpp"
#include "onnx_reader.hpp"
#include "onnx_status.hpp"
#include "onnx_stream.hpp"
#include "onnx_string.hpp"
#include "onnx_tensor_info.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief Streaming ONNX loader with a 128-byte resident footprint.
 *
 * The object is exactly four regions and nothing else:
 *
 *   Stream       40 B   cached caller-owned chunk and the absolute cursor
 *   ModelInfo    32 B   ONNX header (the "static" 32-byte part of the budget)
 *   NodeInfo     28 B   node currently being walked
 *   TensorInfo   28 B   tensor currently being walked
 *
 * No allocation, no module static and no whole-file buffer.  `begin()` binds a
 * chunk callback, `run()` walks the model and calls the visitor, and `copy()`
 * fetches a selected weight or string span later using the same callback.  The
 * visitor may return false from any event; the walk then stops with
 * OnnxCode::STOPPED but the loader object remains valid and can be run again.
 */
class Loader
{
  public:
    Loader () noexcept = default;
    Loader (const Loader &) = delete;
    Loader &operator= (const Loader &) = delete;

    /**
     * @brief Bind the source and reset the header state.
     *
     * `fetch` must return a caller-owned chunk whose first byte is `offset` and
     * whose memory stays valid until the next call.  Returning a null or empty
     * chunk means EOF.  The callback is invoked with absolute file offsets and
     * is never asked for more than the parser needs.
     */
    void
    begin (const ChunkFn fetch, void *const context, const uint32_t size,
           const uint32_t flags = loader_default_flags ()) noexcept
    {
        this->_M_stream.begin (fetch, context, size);
        this->_M_model.reset ();
        this->_M_model._M_loader_flags = static_cast<uint8_t> (flags);
        this->_M_node.reset ();
        this->_M_tensor.reset (TensorRole::NONE);
    }

    [[nodiscard]] uint32_t
    flags () const noexcept
    {
        return this->_M_model._M_loader_flags;
    }

    [[nodiscard]] bool
    ready () const noexcept
    {
        return this->_M_stream.source_ok () && this->_M_stream.size () != 0u;
    }

    /** @brief Walk the bound model and emit every event to `visitor`. */
    [[nodiscard]] OnnxError
    run (const Visitor &visitor) noexcept
    {
        if (!this->ready ())
            return OnnxError (OnnxCode::INTERNAL, 0);
        Context context{ this->_M_stream,
                         this->_M_model,
                         this->_M_node,
                         this->_M_tensor,
                         visitor,
                         static_cast<uint32_t> (this->_M_model._M_loader_flags) };
        return read_model (context, this->_M_stream.size ());
    }

    /** @brief Convenience overload for a plain function callback. */
    [[nodiscard]] OnnxError
    run (const EventFn function, void *const context = nullptr) noexcept
    {
        return this->run (visitor (function, context));
    }

    [[nodiscard]] const ModelInfo &
    model () const noexcept
    {
        return this->_M_model;
    }

    [[nodiscard]] const NodeInfo &
    node () const noexcept
    {
        return this->_M_node;
    }

    [[nodiscard]] const TensorInfo &
    tensor () const noexcept
    {
        return this->_M_tensor;
    }

    /**
     * @brief Copy an absolute source span without moving the parse cursor.
     *
     * `copied` receives the number of bytes written.  The caller asks for only
     * as much as its own buffer can hold; the loader never allocates one.  A
     * range that runs past the declared model size is rejected with
     * LIMIT_EXCEEDED instead of being clamped silently.
     */
    [[nodiscard]] OnnxError
    copy (const uint32_t offset, uint8_t *const destination,
          const uint32_t capacity, uint32_t &copied) const noexcept
    {
        return this->_M_stream.copy_at (offset, destination, capacity, copied);
    }

    /** @brief Copy a string whose span was reported by an event. */
    [[nodiscard]] OnnxError
    copy_string (const StringInfo &info, uint8_t *const destination,
                 const uint32_t capacity, uint32_t &written) const noexcept
    {
        return onnx::copy_string (this->_M_stream, info, destination, capacity,
                                  written);
    }

  private:
    Stream _M_stream;
    ModelInfo _M_model;
    NodeInfo _M_node;
    TensorInfo _M_tensor;

    static_assert (sizeof (Stream) + sizeof (ModelInfo) + sizeof (NodeInfo)
                       + sizeof (TensorInfo)
                   <= _S_ram_budget_bytes,
                   "onnx::Loader exceeds the 128-byte resident budget");
};

static_assert (sizeof (Loader)
                   == (sizeof (void *) == 4u ? 116u : _S_ram_budget_bytes),
               "onnx::Loader must be exactly 128 bytes on 64-bit / 116 on 32-bit");
static_assert (alignof (Loader) <= alignof (void *),
               "onnx::Loader must stay pointer-aligned");
static_assert (sizeof (Context) <= 64u,
               "onnx::Context must stay on the parser stack");
} // namespace zzs::mllee::onnx
