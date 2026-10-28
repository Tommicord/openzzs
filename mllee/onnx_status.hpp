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

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief Detailed failure code of the streaming ONNX reader.
 *
 * mllee's own `Status` is intentionally coarse: it describes the blob format,
 * which the core already trusts.  A loader has to explain why hostile input was
 * rejected, so the ONNX path keeps one extra byte and the absolute stream
 * offset where the rejection happened.  `to_status()` folds the code back into
 * the module-wide enum at the public boundary.
 */
enum class OnnxCode : uint8_t
{
    OK = 0,
    TRUNCATED,
    MALFORMED_VARINT,
    OVERFLOW,
    BAD_WIRE_TYPE,
    BAD_FIELD,
    LIMIT_EXCEEDED,
    UNSUPPORTED,
    BAD_MODEL,
    BAD_GRAPH,
    BAD_TENSOR,
    BAD_ATTRIBUTE,
    BAD_TYPE,
    BAD_SHAPE,
    BAD_DATA,
    EXTERNAL_DATA,
    STOPPED,
    INTERNAL,
};

class OnnxError
{
  public:
    constexpr OnnxError () noexcept = default;

    explicit constexpr OnnxError (const OnnxCode code,
                                  const uint32_t offset = 0) noexcept
        : _M_code (code), _M_offset (offset)
    {
    }

    [[nodiscard]] static constexpr OnnxError
    ok () noexcept
    {
        return OnnxError{};
    }

    [[nodiscard]] constexpr OnnxCode
    code () const noexcept
    {
        return this->_M_code;
    }

    [[nodiscard]] constexpr uint32_t
    offset () const noexcept
    {
        return this->_M_offset;
    }

    [[nodiscard]] constexpr bool
    is_ok () const noexcept
    {
        return this->_M_code == OnnxCode::OK;
    }

    [[nodiscard]] constexpr bool
    is_stopped () const noexcept
    {
        return this->_M_code == OnnxCode::STOPPED;
    }

    [[nodiscard]] constexpr explicit
    operator bool () const noexcept
    {
        return !this->is_ok ();
    }

    [[nodiscard]] constexpr bool
    operator== (const OnnxError &other) const noexcept
    {
        return this->_M_code == other._M_code
               && this->_M_offset == other._M_offset;
    }

  private:
    OnnxCode _M_code = OnnxCode::OK;
    uint32_t _M_offset = 0;
};

[[nodiscard]] constexpr const char *
onnx_code_name (const OnnxCode code) noexcept
{
    switch (code)
        {
        case OnnxCode::OK:
            return "ok";
        case OnnxCode::TRUNCATED:
            return "truncated";
        case OnnxCode::MALFORMED_VARINT:
            return "malformed_varint";
        case OnnxCode::OVERFLOW:
            return "overflow";
        case OnnxCode::BAD_WIRE_TYPE:
            return "bad_wire_type";
        case OnnxCode::BAD_FIELD:
            return "bad_field";
        case OnnxCode::LIMIT_EXCEEDED:
            return "limit_exceeded";
        case OnnxCode::UNSUPPORTED:
            return "unsupported";
        case OnnxCode::BAD_MODEL:
            return "bad_model";
        case OnnxCode::BAD_GRAPH:
            return "bad_graph";
        case OnnxCode::BAD_TENSOR:
            return "bad_tensor";
        case OnnxCode::BAD_ATTRIBUTE:
            return "bad_attribute";
        case OnnxCode::BAD_TYPE:
            return "bad_type";
        case OnnxCode::BAD_SHAPE:
            return "bad_shape";
        case OnnxCode::BAD_DATA:
            return "bad_data";
        case OnnxCode::EXTERNAL_DATA:
            return "external_data";
        case OnnxCode::STOPPED:
            return "stopped";
        case OnnxCode::INTERNAL:
            return "internal";
        }
    return "unknown";
}

[[nodiscard]] constexpr Status
to_status (const OnnxCode code) noexcept
{
    switch (code)
        {
        case OnnxCode::OK:
        case OnnxCode::STOPPED:
            return Status::OK;
        case OnnxCode::TRUNCATED:
            return Status::TRUNCATED;
        case OnnxCode::MALFORMED_VARINT:
        case OnnxCode::OVERFLOW:
        case OnnxCode::BAD_WIRE_TYPE:
        case OnnxCode::BAD_FIELD:
            return Status::CORRUPT_STREAM;
        case OnnxCode::LIMIT_EXCEEDED:
            return Status::OUT_OF_RANGE;
        case OnnxCode::UNSUPPORTED:
            return Status::UNSUPPORTED;
        case OnnxCode::BAD_MODEL:
        case OnnxCode::BAD_GRAPH:
        case OnnxCode::BAD_TENSOR:
        case OnnxCode::BAD_ATTRIBUTE:
        case OnnxCode::BAD_TYPE:
        case OnnxCode::BAD_SHAPE:
        case OnnxCode::BAD_DATA:
        case OnnxCode::EXTERNAL_DATA:
            return Status::BAD_LAYOUT;
        case OnnxCode::INTERNAL:
            return Status::INTERNAL;
        }
    return Status::INTERNAL;
}

[[nodiscard]] constexpr OnnxError
onnx_ok () noexcept
{
    return OnnxError{};
}

[[nodiscard]] constexpr OnnxError
onnx_fail (const OnnxCode code, const uint32_t offset) noexcept
{
    return OnnxError (code, offset);
}
} // namespace zzs::mllee::onnx
