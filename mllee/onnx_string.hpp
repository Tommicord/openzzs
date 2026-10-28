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
#include "onnx_bounded.hpp"
#include "onnx_config.hpp"
#include "onnx_hash.hpp"
#include "onnx_status.hpp"
#include "onnx_stream.hpp"
#include "onnx_wire.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
enum StringFlag : uint8_t
{
    STRING_FLAG_EMPTY = 1u << 0,
    STRING_FLAG_HASHED = 1u << 1,
};

/**
 * @brief A protobuf string that was hashed while streaming past.
 *
 * The payload is never retained.  `_M_offset`/`_M_length` are absolute, so a
 * consumer that really needs the bytes (an external-data location, a node name
 * for diagnostics) can ask Loader::copy() for exactly that span.
 */
struct StringInfo
{
    uint32_t _M_hash = 0;
    uint32_t _M_hash2 = 0;
    uint32_t _M_offset = 0;
    uint32_t _M_length = 0;
    uint8_t _M_flags = 0;
    uint8_t _M_reserved[3]{};

    [[nodiscard]] bool
    empty () const noexcept
    {
        return this->_M_length == 0u;
    }
};

static_assert (sizeof (StringInfo) <= 20u,
               "onnx::StringInfo exceeds its stack budget");

/**
 * @brief Read and hash a length-delimited string.
 *
 * Fast path: the entire payload is in the current chunk, so the assembly
 * FNV-1a kernel runs once over it and djb2 runs over the same bytes.  Slow
 * path: the string crosses one or more chunk boundaries, and the two hashes
 * are folded byte by byte.  Neither path allocates or copies the string.
 */
[[nodiscard]] inline OnnxError
read_string (Stream &stream, const uint32_t limit, StringInfo &out) noexcept
{
    uint32_t length = 0;
    OnnxError error = read_length (stream, limit, length);
    if (error)
        return error;
    if (length > _S_max_string_bytes)
        return OnnxError (OnnxCode::LIMIT_EXCEEDED, stream.position ());

    out._M_offset = stream.position ();
    out._M_length = length;
    out._M_flags = length == 0u ? STRING_FLAG_EMPTY : 0u;

    uint32_t remaining = length;
    uint32_t fnv = _S_fnv_offset_basis;
    uint32_t djb = _S_djb2_basis;
    while (remaining != 0u)
        {
            const uint8_t *data = nullptr;
            uint32_t available = 0;
            error = stream.peek (limit, data, available);
            if (error)
                return error;
            if (available > remaining)
                available = remaining;
            if (available == length && length >= 8u)
                {
                    fnv = hash_bytes (data, length);
                    djb = djb2 (data, length);
                }
            else
                {
                    fnv = fnv1a_update (fnv, data, available);
                    djb = djb2_update (djb, data, available);
                }
            error = stream.skip (limit, available);
            if (error)
                return error;
            remaining -= available;
        }

    out._M_hash = fnv;
    out._M_hash2 = djb;
    out._M_flags |= STRING_FLAG_HASHED;
    return OnnxError::ok ();
}

/**
 * @brief Copy a string payload into a caller buffer.
 *
 * The caller gets at most `capacity` bytes and `written` reports the prefix
 * length.  This is how an external-data "location" value is recovered without
 * giving the loader a path buffer.
 */
[[nodiscard]] inline OnnxError
copy_string (const Stream &stream, const StringInfo &info,
             uint8_t *const destination, const uint32_t capacity,
             uint32_t &written) noexcept
{
    written = 0;
    if (info._M_length == 0u)
        return OnnxError::ok ();
    if (destination == nullptr && capacity != 0u)
        return OnnxError (OnnxCode::INTERNAL, info._M_offset);
    uint32_t take = info._M_length;
    if (take > capacity)
        take = capacity;
    return stream.copy_at (info._M_offset, destination, take, written);
}

/**
 * @brief Parse a decimal uint64 without allocating.
 *
 * ONNX stores external-data offsets and lengths as decimal strings.  Leading
 * zeros are accepted; a sign, whitespace, an empty value or an overflow is
 * rejected rather than coerced, because a silently wrong offset would read the
 * wrong Flash page.
 */
[[nodiscard]] inline OnnxError
parse_decimal (const Stream &stream, const uint32_t offset,
               const uint32_t length, uint64_t &out) noexcept
{
    if (length == 0u)
        return OnnxError (OnnxCode::BAD_DATA, offset);
    uint64_t value = 0;
    uint32_t consumed = 0;
    uint8_t buffer[16]{};
    while (consumed < length)
        {
            uint32_t run = length - consumed;
            if (run > sizeof (buffer))
                run = sizeof (buffer);
            uint32_t copied = 0;
            const OnnxError error = stream.copy_at (offset + consumed, buffer, run,
                                                     copied);
            if (error)
                return error;
            if (copied != run)
                return OnnxError (OnnxCode::TRUNCATED, offset + consumed);
            for (uint32_t i = 0; i < run; ++i)
                {
                    const uint8_t digit = buffer[i];
                    if (digit < static_cast<uint8_t> ('0')
                        || digit > static_cast<uint8_t> ('9'))
                        return OnnxError (OnnxCode::BAD_DATA, offset + consumed + i);
                    uint64_t scaled = 0;
                    uint64_t next = 0;
                    if (mul_overflow_u64 (value, 10u, scaled)
                        || add_overflow_u64 (scaled,
                                             static_cast<uint64_t> (
                                                 digit
                                                 - static_cast<uint8_t> ('0')),
                                             next))
                        return OnnxError (OnnxCode::OVERFLOW, offset + consumed + i);
                    value = next;
                }
            consumed += run;
        }
    out = value;
    return OnnxError::ok ();
}

/** @brief Compare an already-hashed string with a compile-time literal. */
template <unsigned int N>
[[nodiscard]] constexpr bool
string_is (const StringInfo &info, const char (&literal)[N]) noexcept
{
    return info._M_length == static_cast<uint32_t> (N - 1u)
           && info._M_hash == fnv1a_literal (literal)
           && info._M_hash2 == djb2_literal (literal);
}
} // namespace zzs::mllee::onnx
