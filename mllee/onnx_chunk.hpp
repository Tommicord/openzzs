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

#include "onnx_config.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief A caller-owned span of the ONNX byte stream.
 *
 * The callback contract is deliberately tiny: return a chunk whose first byte
 * is the byte at `offset`, or a null/zero chunk at EOF.  The returned memory
 * must stay valid until the next callback invocation; the loader never copies
 * it.  This is what keeps the resident state at 128 bytes while large weight
 * tensors stay in Flash, a memory-mapped partition or a network buffer chosen
 * by the caller.
 */
struct Chunk
{
    const uint8_t *_M_data = nullptr;
    uint32_t _M_size = 0;
};

/** @brief Chunk source callback. `offset` is absolute in the ONNX file. */
using ChunkFn = Chunk (*) (void *_M_context, uint32_t _M_offset) noexcept;

/** @brief A contiguous model image, useful for tests and memory-mapped files. */
struct MemorySource
{
    const uint8_t *_M_data = nullptr;
    uint32_t _M_size = 0;
};

[[nodiscard]] inline Chunk
memory_chunk (void *const context, const uint32_t offset) noexcept
{
    const MemorySource *const source = static_cast<const MemorySource *> (context);
    if (source == nullptr || source->_M_data == nullptr
        || offset >= source->_M_size)
        return Chunk{};
    return Chunk{ source->_M_data + offset, source->_M_size - offset };
}

/**
 * @brief A table of discontiguous caller-owned chunks.
 *
 * The loader's cursor is absolute, so a firmware update can expose an ONNX file
 * as, say, one header chunk in RAM and one weight chunk in external Flash
 * without merging them.  The lookup is linear; embedded models have a handful
 * of regions, and a binary search table would cost more Flash than it saves.
 */
struct ChunkList
{
    const Chunk *_M_chunks = nullptr;
    uint32_t _M_count = 0;
};

[[nodiscard]] inline Chunk
chunk_list_chunk (void *const context, const uint32_t offset) noexcept
{
    const ChunkList *const list = static_cast<const ChunkList *> (context);
    if (list == nullptr || list->_M_chunks == nullptr)
        return Chunk{};
    uint32_t base = 0;
    for (uint32_t i = 0; i < list->_M_count; ++i)
        {
            const Chunk chunk = list->_M_chunks[i];
            if (chunk._M_data == nullptr || chunk._M_size == 0u)
                continue;
            if (offset < base)
                return Chunk{};
            if (offset - base < chunk._M_size)
                return Chunk{ chunk._M_data + (offset - base),
                              chunk._M_size - (offset - base) };
            if (chunk._M_size > 0xFFFFFFFFu - base)
                return Chunk{};
            base += chunk._M_size;
        }
    return Chunk{};
}
} // namespace zzs::mllee::onnx
