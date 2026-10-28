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
#include "onnx_reader.hpp"
#include "onnx_scope.hpp"
#include "onnx_string.hpp"
#include "onnx_types.hpp"
#include "onnx_wire.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
enum ExternalFlag : uint8_t
{
    EXTERNAL_FLAG_LOCATION = 1u << 0,
    EXTERNAL_FLAG_OFFSET = 1u << 1,
    EXTERNAL_FLAG_LENGTH = 1u << 2,
    EXTERNAL_FLAG_CHECKSUM = 1u << 3,
};

/**
 * @brief One StringStringEntryProto from TensorProto.external_data.
 *
 * Both strings are hashed and their absolute spans are retained, so the event
 * consumer can copy "location" or reject an unexpected key without the loader
 * owning a path buffer.  Offset and length values are parsed eagerly because a
 * wrong byte range must be caught before the application starts reading Flash.
 */
struct ExternalInfo
{
    StringInfo _M_key{};
    StringInfo _M_value{};
    uint64_t _M_offset_value = 0;
    uint64_t _M_length_value = 0;
    uint8_t _M_flags = 0;
    uint8_t _M_reserved[3]{};
};

[[nodiscard]] inline OnnxError
read_external (Context &context, const uint32_t limit, const TensorRole role,
               ExternalInfo &out) noexcept
{
    out = ExternalInfo{};
    Scope scope{};
    OnnxError error = open_scope (context._M_stream, limit, scope);
    if (error)
        return error;
    while (!scope_done (context._M_stream, scope))
        {
            Tag tag{};
            error = read_tag (context._M_stream, scope._M_end, tag);
            if (error)
                return error;
            switch (tag._M_field)
                {
                case 1:
                    if (!tag.is_length ())
                        return bad_field (context._M_stream);
                    error = read_string (context._M_stream, scope._M_end,
                                         out._M_key);
                    if (error)
                        return error;
                    break;
                case 2:
                    if (!tag.is_length ())
                        return bad_field (context._M_stream);
                    error = read_string (context._M_stream, scope._M_end,
                                         out._M_value);
                    if (error)
                        return error;
                    break;
                default:
                    error = skip_value (context._M_stream, scope._M_end,
                                        tag._M_wire);
                    if (error)
                        return error;
                    break;
                }
        }

    if (out._M_key.empty () || out._M_value.empty ())
        return OnnxError (OnnxCode::BAD_DATA, scope._M_end);

    if (string_is (out._M_key, "location"))
        {
            out._M_flags |= EXTERNAL_FLAG_LOCATION;
        }
    else if (string_is (out._M_key, "offset"))
        {
            out._M_flags |= EXTERNAL_FLAG_OFFSET;
            error = parse_decimal (context._M_stream, out._M_value._M_offset,
                                   out._M_value._M_length, out._M_offset_value);
            if (error)
                return error;
        }
    else if (string_is (out._M_key, "length"))
        {
            out._M_flags |= EXTERNAL_FLAG_LENGTH;
            error = parse_decimal (context._M_stream, out._M_value._M_offset,
                                   out._M_value._M_length, out._M_length_value);
            if (error)
                return error;
        }
    else if (string_is (out._M_key, "checksum"))
        {
            out._M_flags |= EXTERNAL_FLAG_CHECKSUM;
        }

    if (!context.emit (EventKind::EXTERNAL_DATA, role, out._M_value._M_offset,
                       out._M_key._M_hash, &out, out._M_value._M_length, 0,
                       out._M_flags))
        return stopped (context._M_stream);
    return OnnxError::ok ();
}
} // namespace zzs::mllee::onnx
