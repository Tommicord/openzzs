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
#include "onnx_reader.hpp"
#include "onnx_scope.hpp"
#include "onnx_shape.hpp"
#include "onnx_string.hpp"
#include "onnx_tensor_info.hpp"
#include "onnx_wire.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief Parse ValueInfoProto for a graph input, output or value_info entry.
 *
 * The name is hashed, the type is walked, and every dimension is emitted as a
 * VALUE_DIM event.  The final VALUE_END event carries the TensorInfo with the
 * saturated element count and shape hash.  The loader never stores the name or
 * the dimension vector.
 */
[[nodiscard]] inline OnnxError
read_value (Context &context, const uint32_t limit,
            const TensorRole role) noexcept
{
    Scope scope{};
    OnnxError error = open_scope (context._M_stream, limit, scope);
    if (error)
        return error;

    TensorInfo &tensor = context._M_tensor;
    tensor.reset (role);
    StringInfo name{};
    bool has_type = false;
    if (!context.emit (EventKind::VALUE_BEGIN, role, scope._M_end, 0, &tensor))
        return stopped (context._M_stream);

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
                    error = read_string (context._M_stream, scope._M_end, name);
                    if (error)
                        return error;
                    break;
                case 2:
                    if (!tag.is_length ())
                        return bad_field (context._M_stream);
                    error = read_type (context, scope._M_end, tensor, has_type);
                    if (error)
                        return error;
                    break;
                case 3:
                    {
                        StringInfo doc{};
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        error = read_string (context._M_stream, scope._M_end, doc);
                        if (error)
                            return error;
                        break;
                    }
                default:
                    error = skip_value (context._M_stream, scope._M_end,
                                        tag._M_wire);
                    if (error)
                        return error;
                    break;
                }
        }

    tensor.finish_shape ();
    tensor._M_name_hash = name._M_hash;
    tensor._M_name_length = saturate_u16 (name._M_length);
    if (has_type && tensor._M_dtype == static_cast<uint8_t> (DataType::UNDEFINED))
        return OnnxError (OnnxCode::BAD_TYPE, scope._M_end);
    if ((tensor._M_flags & TENSOR_FLAG_OVERFLOW) != 0u)
        return OnnxError (OnnxCode::OVERFLOW, scope._M_end);
    if (!context.emit (EventKind::VALUE_END, role, scope._M_end, 0, &tensor,
                       0, 0, tensor._M_flags))
        return stopped (context._M_stream);
    return OnnxError::ok ();
}
} // namespace zzs::mllee::onnx
