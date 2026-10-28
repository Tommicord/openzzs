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
#include "onnx_status.hpp"
#include "onnx_stream.hpp"
#include "onnx_types.hpp"
#include "onnx_varint.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/** @brief A decoded protobuf field tag. */
struct Tag
{
    uint32_t _M_field = 0;
    WireType _M_wire = WireType::VARINT;

    [[nodiscard]] constexpr bool
    is_varint () const noexcept
    {
        return this->_M_wire == WireType::VARINT;
    }

    [[nodiscard]] constexpr bool
    is_length () const noexcept
    {
        return this->_M_wire == WireType::LENGTH;
    }
};

[[nodiscard]] inline OnnxError
read_tag (Stream &stream, const uint32_t limit, Tag &tag) noexcept
{
    uint32_t raw = 0;
    const uint32_t offset = stream.position ();
    OnnxError error = read_varint32 (stream, limit, raw);
    if (error)
        return error;
    const uint32_t field = raw >> 3;
    const uint32_t wire = raw & 7u;
    if (field == 0u || field > _S_max_field_number)
        return OnnxError (OnnxCode::BAD_FIELD, offset);
    switch (wire)
        {
        case 0:
            tag._M_wire = WireType::VARINT;
            break;
        case 1:
            tag._M_wire = WireType::FIXED64;
            break;
        case 2:
            tag._M_wire = WireType::LENGTH;
            break;
        case 3:
        case 4:
            /* ONNX never emits protobuf groups; rejecting them keeps the
               skipper non-recursive and hostile input from nesting forever. */
            return OnnxError (OnnxCode::UNSUPPORTED, offset);
        case 5:
            tag._M_wire = WireType::FIXED32;
            break;
        default:
            return OnnxError (OnnxCode::BAD_WIRE_TYPE, offset);
        }
    tag._M_field = field;
    return OnnxError::ok ();
}

/**
 * @brief Read a length prefix and verify it fits in the enclosing message.
 *
 * `_S_max_string_bytes` is intentionally not applied here: a raw_data field can
 * legitimately be hundreds of megabytes.  Callers that hash strings apply their
 * own bound in onnx_string.hpp.
 */
[[nodiscard]] inline OnnxError
read_length (Stream &stream, const uint32_t limit, uint32_t &length) noexcept
{
    const uint32_t offset = stream.position ();
    OnnxError error = read_u32 (stream, limit, length);
    if (error)
        return error;
    if (length > stream.remaining (limit))
        return OnnxError (OnnxCode::TRUNCATED, offset);
    return OnnxError::ok ();
}

[[nodiscard]] inline OnnxError
read_fixed32 (Stream &stream, const uint32_t limit, uint32_t &out) noexcept
{
    if (stream.remaining (limit) < 4u)
        return OnnxError (OnnxCode::TRUNCATED, stream.position ());
    uint8_t bytes[4]{};
    const OnnxError error = stream.read_into (limit, bytes, 4u);
    if (error)
        return error;
    out = static_cast<uint32_t> (bytes[0])
          | (static_cast<uint32_t> (bytes[1]) << 8)
          | (static_cast<uint32_t> (bytes[2]) << 16)
          | (static_cast<uint32_t> (bytes[3]) << 24);
    return OnnxError::ok ();
}

[[nodiscard]] inline OnnxError
read_fixed64 (Stream &stream, const uint32_t limit, uint64_t &out) noexcept
{
    if (stream.remaining (limit) < 8u)
        return OnnxError (OnnxCode::TRUNCATED, stream.position ());
    uint8_t bytes[8]{};
    const OnnxError error = stream.read_into (limit, bytes, 8u);
    if (error)
        return error;
    out = static_cast<uint64_t> (bytes[0])
          | (static_cast<uint64_t> (bytes[1]) << 8)
          | (static_cast<uint64_t> (bytes[2]) << 16)
          | (static_cast<uint64_t> (bytes[3]) << 24)
          | (static_cast<uint64_t> (bytes[4]) << 32)
          | (static_cast<uint64_t> (bytes[5]) << 40)
          | (static_cast<uint64_t> (bytes[6]) << 48)
          | (static_cast<uint64_t> (bytes[7]) << 56);
    return OnnxError::ok ();
}

[[nodiscard]] inline OnnxError
skip_fixed (Stream &stream, const uint32_t limit,
            const uint32_t bytes) noexcept
{
    if (stream.remaining (limit) < bytes)
        return OnnxError (OnnxCode::TRUNCATED, stream.position ());
    return stream.skip (limit, bytes);
}

/** @brief Skip one field value of the given wire type. */
[[nodiscard]] inline OnnxError
skip_value (Stream &stream, const uint32_t limit, const WireType wire) noexcept
{
    switch (wire)
        {
        case WireType::VARINT:
            {
                uint64_t ignored = 0;
                return read_varint64 (stream, limit, ignored);
            }
        case WireType::FIXED64:
            return skip_fixed (stream, limit, 8u);
        case WireType::LENGTH:
            {
                uint32_t length = 0;
                OnnxError error = read_length (stream, limit, length);
                if (error)
                    return error;
                return stream.skip (limit, length);
            }
        case WireType::FIXED32:
            return skip_fixed (stream, limit, 4u);
        case WireType::START_GROUP:
        case WireType::END_GROUP:
            return OnnxError (OnnxCode::UNSUPPORTED, stream.position ());
        }
    return OnnxError (OnnxCode::BAD_WIRE_TYPE, stream.position ());
}
} // namespace zzs::mllee::onnx
