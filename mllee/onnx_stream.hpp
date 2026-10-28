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
#include "onnx_asm.hpp"
#include "onnx_chunk.hpp"
#include "onnx_config.hpp"
#include "onnx_status.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief Forward-only cursor over a chunked protobuf stream.
 *
 * The cursor keeps one caller-owned chunk cached.  Reads that cross a chunk
 * boundary call the source callback again; the parser never asks for the whole
 * model.  `limit` is the absolute end of the protobuf message currently being
 * parsed, so a nested length cannot make a read wander into the next message.
 *
 * Layout (64-bit host): fetch 8, context 8, data 8, base 4, chunk size 4,
 * chunk position 4, model size 4 = 40 bytes.  On a 32-bit target the three
 * pointers are 4 bytes, so the same cursor is 28 bytes.
 */
class Stream
{
  public:
    Stream () noexcept = default;
    Stream (const Stream &) = delete;
    Stream &operator= (const Stream &) = delete;

    void
    begin (const ChunkFn fetch, void *const context,
           const uint32_t size) noexcept
    {
        this->_M_fetch = fetch;
        this->_M_context = context;
        this->_M_data = nullptr;
        this->_M_base = 0;
        this->_M_chunk_size = 0;
        this->_M_chunk_pos = 0;
        this->_M_size = size == 0u ? 0u : (size > _S_max_model_bytes ? _S_max_model_bytes : size);
    }

    [[nodiscard]] uint32_t
    size () const noexcept
    {
        return this->_M_size;
    }

    [[nodiscard]] uint32_t
    position () const noexcept
    {
        return this->_M_base + this->_M_chunk_pos;
    }

    [[nodiscard]] uint32_t
    remaining (const uint32_t limit) const noexcept
    {
        const uint32_t here = this->position ();
        return here < limit ? limit - here : 0u;
    }

    [[nodiscard]] bool
    done (const uint32_t limit) const noexcept
    {
        return this->position () >= limit;
    }

    [[nodiscard]] bool
    source_ok () const noexcept
    {
        return this->_M_fetch != nullptr;
    }

    /**
     * @brief Make at least one byte visible, or report truncation.
     *
     * A zero-sized or null chunk is EOF.  A chunk that claims to run past the
     * declared model size is clamped rather than trusted, which keeps a buggy
     * flash driver from making the cursor address outside the model.
     */
    [[nodiscard]] OnnxError
    fill (const uint32_t limit) noexcept
    {
        if (this->_M_fetch == nullptr)
            return OnnxError (OnnxCode::INTERNAL, this->position ());
        if (this->_M_chunk_pos < this->_M_chunk_size)
            return OnnxError::ok ();

        const uint32_t next = this->_M_base + this->_M_chunk_size;
        if (next >= this->_M_size || next >= limit)
            return OnnxError (OnnxCode::TRUNCATED, next);
        const Chunk chunk = this->_M_fetch (this->_M_context, next);
        if (chunk._M_data == nullptr || chunk._M_size == 0u)
            return OnnxError (OnnxCode::TRUNCATED, next);

        uint32_t available = chunk._M_size;
        const uint32_t room = this->_M_size - next;
        if (available > room)
            available = room;
        if (available == 0u)
            return OnnxError (OnnxCode::TRUNCATED, next);
        this->_M_base = next;
        this->_M_data = chunk._M_data;
        this->_M_chunk_size = available;
        this->_M_chunk_pos = 0;
        return OnnxError::ok ();
    }

    /**
     * @brief Return a contiguous run at the cursor without consuming it.
     *
     * The varint fast path needs five (or ten) adjacent bytes.  When the
     * current chunk has them, the assembly decoder runs on the caller's buffer;
     * otherwise the byte-at-a-time path handles the boundary normally.
     */
    [[nodiscard]] OnnxError
    peek (const uint32_t limit, const uint8_t *&data,
          uint32_t &available) noexcept
    {
        if (this->position () >= limit)
            return OnnxError (OnnxCode::TRUNCATED, this->position ());
        if (this->_M_chunk_pos >= this->_M_chunk_size)
            {
                const OnnxError error = this->fill (limit);
                if (error)
                    return error;
            }
        const uint32_t here = this->position ();
        if (here >= limit)
            return OnnxError (OnnxCode::TRUNCATED, here);
        uint32_t run = this->_M_chunk_size - this->_M_chunk_pos;
        const uint32_t room = limit - here;
        if (run > room)
            run = room;
        data = this->_M_data + this->_M_chunk_pos;
        available = run;
        return OnnxError::ok ();
    }

    [[nodiscard]] OnnxError
    read_byte (const uint32_t limit, uint8_t &out) noexcept
    {
        const uint32_t here = this->position ();
        if (here >= limit)
            return OnnxError (OnnxCode::TRUNCATED, here);
        if (this->_M_chunk_pos >= this->_M_chunk_size)
            {
                const OnnxError error = this->fill (limit);
                if (error)
                    return error;
            }
        out = this->_M_data[this->_M_chunk_pos++];
        return OnnxError::ok ();
    }

    [[nodiscard]] OnnxError
    read_into (const uint32_t limit, uint8_t *const destination,
               uint32_t count) noexcept
    {
        uint8_t *cursor = destination;
        while (count != 0u)
            {
                const uint32_t here = this->position ();
                if (here >= limit)
                    return OnnxError (OnnxCode::TRUNCATED, here);
                if (this->_M_chunk_pos >= this->_M_chunk_size)
                    {
                        const OnnxError error = this->fill (limit);
                        if (error)
                            return error;
                    }
                uint32_t run = this->_M_chunk_size - this->_M_chunk_pos;
                const uint32_t room = limit - here;
                if (run > room)
                    run = room;
                if (run == 0u)
                    return OnnxError (OnnxCode::TRUNCATED, here);
                if (run > count)
                    run = count;
                copy_bytes (cursor, this->_M_data + this->_M_chunk_pos, run);
                this->_M_chunk_pos += run;
                cursor += run;
                count -= run;
            }
        return OnnxError::ok ();
    }

    [[nodiscard]] OnnxError
    skip (const uint32_t limit, uint32_t count) noexcept
    {
        while (count != 0u)
            {
                const uint32_t here = this->position ();
                if (here >= limit)
                    return OnnxError (OnnxCode::TRUNCATED, here);
                if (this->_M_chunk_pos >= this->_M_chunk_size)
                    {
                        const OnnxError error = this->fill (limit);
                        if (error)
                            return error;
                    }
                uint32_t run = this->_M_chunk_size - this->_M_chunk_pos;
                const uint32_t room = limit - here;
                if (run > room)
                    run = room;
                if (run > count)
                    run = count;
                if (run == 0u)
                    return OnnxError (OnnxCode::TRUNCATED, here);
                this->_M_chunk_pos += run;
                count -= run;
            }
        return OnnxError::ok ();
    }

    /**
     * @brief Copy an absolute range without disturbing the parse cursor.
     *
     * This is the only way a caller gets weight bytes: the loader reports the
     * offset and length in a TENSOR_DATA event, and the caller asks for exactly
     * the slice it can hold.  The source callback may return a short chunk; the
     * method loops until `capacity` bytes are copied or EOF is reached.
     */
    [[nodiscard]] OnnxError
    copy_at (const uint32_t offset, uint8_t *const destination,
             const uint32_t capacity, uint32_t &copied) const noexcept
    {
        copied = 0;
        if (capacity == 0u)
            return OnnxError::ok ();
        if (destination == nullptr)
            return OnnxError (OnnxCode::INTERNAL, offset);
        if (offset >= this->_M_size)
            return OnnxError (OnnxCode::TRUNCATED, offset);
        if (capacity > this->_M_size - offset)
            return OnnxError (OnnxCode::LIMIT_EXCEEDED, offset);

        while (copied < capacity)
            {
                if (this->_M_fetch == nullptr)
                    return OnnxError (OnnxCode::INTERNAL, offset + copied);
                const Chunk chunk = this->_M_fetch (this->_M_context, offset + copied);
                if (chunk._M_data == nullptr || chunk._M_size == 0u)
                    return OnnxError (OnnxCode::TRUNCATED, offset + copied);
                uint32_t run = chunk._M_size;
                const uint32_t room = this->_M_size - (offset + copied);
                if (run > room)
                    run = room;
                if (run > capacity - copied)
                    run = capacity - copied;
                if (run == 0u)
                    return OnnxError (OnnxCode::TRUNCATED, offset + copied);
                copy_bytes (destination + copied, chunk._M_data, run);
                copied += run;
            }
        return OnnxError::ok ();
    }

  private:
    ChunkFn _M_fetch = nullptr;
    void *_M_context = nullptr;
    const uint8_t *_M_data = nullptr;
    uint32_t _M_base = 0;
    uint32_t _M_chunk_size = 0;
    uint32_t _M_chunk_pos = 0;
    uint32_t _M_size = 0;
};

static_assert (sizeof (Stream) <= (sizeof (void *) == 4 ? 32u : _S_stream_budget_bytes),
               "onnx::Stream exceeds its share of the RAM budget");
static_assert (alignof (Stream) <= alignof (void *),
               "onnx::Stream must stay pointer-aligned");
} // namespace zzs::mllee::onnx
