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
#include "onnx_string.hpp"
#include "onnx_tensor_info.hpp"
#include "onnx_types.hpp"
#include "onnx_varint.hpp"
#include "onnx_wire.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief Parse one TensorShapeProto.Dimension.
 *
 * A dimension carries exactly one of dim_value or dim_param.  Both present is
 * ambiguous and rejected; neither present is a producer bug and rejected too.
 * The product is folded into TensorInfo with overflow checking, so a shape such
 * as [2^32, 2^32, 2^32] cannot wrap into a small allocation.
 */
[[nodiscard]] inline OnnxError
read_dimension (Context &context, const uint32_t limit, const EventKind kind,
                const uint16_t index, TensorInfo &tensor) noexcept
{
    Scope scope{};
    OnnxError error = open_scope (context._M_stream, limit, scope);
    if (error)
        return error;
    int64_t value = 0;
    bool has_value = false;
    bool has_param = false;
    StringInfo param{};
    while (!scope_done (context._M_stream, scope))
        {
            Tag tag{};
            error = read_tag (context._M_stream, scope._M_end, tag);
            if (error)
                return error;
            switch (tag._M_field)
                {
                case 1:
                    if (!tag.is_varint ())
                        return bad_field (context._M_stream);
                    error = read_int64 (context._M_stream, scope._M_end, value);
                    if (error)
                        return error;
                    has_value = true;
                    break;
                case 2:
                    if (!tag.is_length ())
                        return bad_field (context._M_stream);
                    error = read_string (context._M_stream, scope._M_end, param);
                    if (error)
                        return error;
                    has_param = true;
                    break;
                case 3:
                    {
                        StringInfo denotation{};
                        if (!tag.is_length ())
                            return bad_field (context._M_stream);
                        error = read_string (context._M_stream, scope._M_end,
                                             denotation);
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

    if (has_value == has_param)
        return OnnxError (OnnxCode::BAD_SHAPE, scope._M_end);
    if (!has_param && value < 0)
        return OnnxError (OnnxCode::BAD_SHAPE, context._M_stream.position ());
    tensor.note_dim (has_param ? 0 : value, has_param);
    const uint32_t dim_value
        = has_param ? 0u : saturate_u32 (static_cast<uint64_t> (value));
    const uint8_t dim_flags = has_param ? TENSOR_FLAG_DYNAMIC : 0u;
    if (!context.emit (kind, tensor.role (), tensor._M_shape_hash, dim_value,
                       &tensor, 0, index, dim_flags))
        return stopped (context._M_stream);
    return OnnxError::ok ();
}

[[nodiscard]] inline OnnxError
read_shape (Context &context, const uint32_t limit, TensorInfo &tensor,
            const EventKind kind) noexcept
{
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
            if (tag._M_field == 1u)
                {
                    if (!tag.is_length ())
                        return bad_field (context._M_stream);
                    if (tensor._M_rank >= _S_max_tensor_rank)
                        {
                            tensor._M_flags |= TENSOR_FLAG_OVERFLOW;
                            error = skip_value (context._M_stream, scope._M_end,
                                                tag._M_wire);
                            if (error)
                                return error;
                            continue;
                        }
                    const uint16_t index = static_cast<uint16_t> (tensor._M_rank);
                    error = read_dimension (context, scope._M_end, kind, index,
                                            tensor);
                    if (error)
                        return error;
                }
            else
                {
                    error = skip_value (context._M_stream, scope._M_end,
                                        tag._M_wire);
                    if (error)
                        return error;
                }
        }
    return OnnxError::ok ();
}

/**
 * @brief Parse TypeProto.Tensor / TypeProto.SparseTensor.
 *
 * Only tensor and sparse-tensor types can be lowered to mllee.  A sequence,
 * map, optional or opaque type is reported as UNSUPPORTED in strict mode; in
 * tolerant mode the message is skipped and the value is marked invalid.
 */
[[nodiscard]] inline OnnxError
read_tensor_type (Context &context, const uint32_t limit, TensorInfo &tensor,
                  const bool sparse) noexcept
{
    Scope scope{};
    OnnxError error = open_scope (context._M_stream, limit, scope);
    if (error)
        return error;
    if (sparse)
        tensor._M_flags |= TENSOR_FLAG_SPARSE;
    while (!scope_done (context._M_stream, scope))
        {
            Tag tag{};
            error = read_tag (context._M_stream, scope._M_end, tag);
            if (error)
                return error;
            switch (tag._M_field)
                {
                case 1:
                    {
                        if (!tag.is_varint ())
                            return bad_field (context._M_stream);
                        uint32_t dtype = 0;
                        error = read_u32 (context._M_stream, scope._M_end, dtype);
                        if (error)
                            return error;
                        if (!data_type_valid (dtype))
                            return OnnxError (OnnxCode::BAD_TYPE,
                                              context._M_stream.position ());
                        tensor._M_dtype = static_cast<uint8_t> (dtype);
                        break;
                    }
                case 2:
                    if (!tag.is_length ())
                        return bad_field (context._M_stream);
                    error = read_shape (context, scope._M_end, tensor,
                                        EventKind::VALUE_DIM);
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
    return OnnxError::ok ();
}

[[nodiscard]] inline OnnxError
read_type (Context &context, const uint32_t limit, TensorInfo &tensor,
           bool &has_type) noexcept
{
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
                    has_type = true;
                    error = read_tensor_type (context, scope._M_end, tensor, false);
                    if (error)
                        return error;
                    break;
                case 8:
                    if (!tag.is_length ())
                        return bad_field (context._M_stream);
                    has_type = true;
                    error = read_tensor_type (context, scope._M_end, tensor, true);
                    if (error)
                        return error;
                    break;
                case 4:
                case 5:
                case 7:
                case 9:
                    if (context.strict ())
                        return OnnxError (OnnxCode::UNSUPPORTED,
                                          context._M_stream.position ());
                    tensor._M_flags |= TENSOR_FLAG_INVALID;
                    error = skip_value (context._M_stream, scope._M_end,
                                        tag._M_wire);
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
    return OnnxError::ok ();
}
} // namespace zzs::mllee::onnx
