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
#include "onnx_status.hpp"
#include "onnx_stream.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief Decode one protobuf varint32 from the stream.
 *
 * The fast path hands five adjacent bytes to the target kernel.  Only when the
 * chunk boundary falls inside the varint does the byte-at-a-time path run, so
 * an ordinary field tag or length costs one call and one bounds check.  Values
 * that need a sixth byte are rejected as OVERFLOW; ONNX never uses them.
 */
[[nodiscard]] inline OnnxError
read_varint32 (Stream &stream, const uint32_t limit, uint32_t &out) noexcept
{
    const uint8_t *data = nullptr;
    uint32_t available = 0;
    OnnxError error = stream.peek (limit, data, available);
    if (error)
        return error;
    if (available >= 5u)
        {
            const uint32_t used = varint32_bytes (data, available, out);
            if (used != 0u)
                {
                    error = stream.skip (limit, used);
                    return error;
                }
            return OnnxError (OnnxCode::OVERFLOW, stream.position ());
        }

    uint32_t result = 0;
    uint32_t shift = 0;
    for (uint32_t i = 0; i < 5u; ++i)
        {
            uint8_t byte = 0;
            error = stream.read_byte (limit, byte);
            if (error)
                return error;
            if (i == 4u && (byte & 0xF0u) != 0u)
                return OnnxError (OnnxCode::OVERFLOW, stream.position ());
            result |= static_cast<uint32_t> (byte & 0x7Fu) << shift;
            if ((byte & 0x80u) == 0u)
                {
                    out = result;
                    return OnnxError::ok ();
                }
            shift += 7u;
        }
    return OnnxError (OnnxCode::OVERFLOW, stream.position ());
}

/** @brief Decode one protobuf varint64 from the stream. */
[[nodiscard]] inline OnnxError
read_varint64 (Stream &stream, const uint32_t limit, uint64_t &out) noexcept
{
    const uint8_t *data = nullptr;
    uint32_t available = 0;
    OnnxError error = stream.peek (limit, data, available);
    if (error)
        return error;
    if (available >= 10u)
        {
            const uint32_t used = varint64_bytes (data, available, out);
            if (used != 0u)
                {
                    error = stream.skip (limit, used);
                    return error;
                }
            return OnnxError (OnnxCode::OVERFLOW, stream.position ());
        }

    uint64_t result = 0;
    uint32_t shift = 0;
    for (uint32_t i = 0; i < 10u; ++i)
        {
            uint8_t byte = 0;
            error = stream.read_byte (limit, byte);
            if (error)
                return error;
            if (i == 9u && (byte & 0xFEu) != 0u)
                return OnnxError (OnnxCode::OVERFLOW, stream.position ());
            result |= static_cast<uint64_t> (byte & 0x7Fu) << shift;
            if ((byte & 0x80u) == 0u)
                {
                    out = result;
                    return OnnxError::ok ();
                }
            shift += 7u;
        }
    return OnnxError (OnnxCode::OVERFLOW, stream.position ());
}

/** @brief protobuf zigzag decoding, used by signed attributes if a producer uses it. */
[[nodiscard]] constexpr int64_t
zigzag_decode (const uint64_t value) noexcept
{
    return static_cast<int64_t> ((value >> 1)
                                 ^ static_cast<uint64_t> (-static_cast<int64_t> (value & 1u)));
}

/** @brief Decode a signed 64-bit wire integer (two's complement varint). */
[[nodiscard]] inline OnnxError
read_int64 (Stream &stream, const uint32_t limit, int64_t &out) noexcept
{
    uint64_t raw = 0;
    const OnnxError error = read_varint64 (stream, limit, raw);
    if (error)
        return error;
    out = static_cast<int64_t> (raw);
    return OnnxError::ok ();
}

/** @brief Decode an unsigned 32-bit quantity from a 64-bit wire field. */
[[nodiscard]] inline OnnxError
read_u32 (Stream &stream, const uint32_t limit, uint32_t &out) noexcept
{
    uint64_t raw = 0;
    const OnnxError error = read_varint64 (stream, limit, raw);
    if (error)
        return error;
    if (raw > 0xFFFFFFFFull)
        return OnnxError (OnnxCode::OVERFLOW, stream.position ());
    out = static_cast<uint32_t> (raw);
    return OnnxError::ok ();
}
} // namespace zzs::mllee::onnx
