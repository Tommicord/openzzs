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
#include "onnx_status.hpp"
#include "onnx_stream.hpp"
#include "onnx_wire.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief Absolute end of a length-delimited protobuf message.
 *
 * A scope is 4 bytes and lives on the parser's stack for the duration of the
 * message.  Nested scopes are ordinary C++ frames, bounded by
 * `_S_max_message_depth` in the structural parser; the wire skipper itself is
 * non-recursive, so a hostile stream cannot exhaust the stack with groups.
 */
struct Scope
{
    uint32_t _M_end = 0;
};

[[nodiscard]] inline OnnxError
open_scope (Stream &stream, const uint32_t limit, Scope &scope) noexcept
{
    uint32_t length = 0;
    const OnnxError error = read_length (stream, limit, length);
    if (error)
        return error;
    const uint32_t start = stream.position ();
    if (length > limit - start)
        return OnnxError (OnnxCode::TRUNCATED, start);
    scope._M_end = start + length;
    return OnnxError::ok ();
}

[[nodiscard]] inline bool
scope_done (const Stream &stream, const Scope &scope) noexcept
{
    return stream.position () >= scope._M_end;
}

/**
 * @brief Consume the remainder of a scope and verify the cursor landed on it.
 *
 * A parser normally exits its loop exactly at `_M_end`.  If a field was skipped
 * imperfectly, the leftover bytes are dropped here; crossing the end is a hard
 * error because the message would then describe the next message's bytes.
 */
[[nodiscard]] inline OnnxError
close_scope (Stream &stream, const Scope &scope) noexcept
{
    const uint32_t here = stream.position ();
    if (here > scope._M_end)
        return OnnxError (OnnxCode::TRUNCATED, here);
    if (here == scope._M_end)
        return OnnxError::ok ();
    return stream.skip (scope._M_end, scope._M_end - here);
}

/** @brief Require an exact end position (used by packed numeric fields). */
[[nodiscard]] inline OnnxError
require_scope_end (const Stream &stream, const Scope &scope) noexcept
{
    const uint32_t here = stream.position ();
    if (here != scope._M_end)
        return OnnxError (OnnxCode::BAD_FIELD, here);
    return OnnxError::ok ();
}
} // namespace zzs::mllee::onnx
