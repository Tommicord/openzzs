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
#include "onnx_config.hpp"
#include "onnx_types.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief The 28-byte TensorProto header retained while a tensor streams past.
 *
 * `_M_data_offset` is absolute in the ONNX image and points at the first
 * payload byte (the raw_data payload or the first packed element), so the
 * caller can copy the range later with Loader::copy().  The loader itself
 * never retains a weight byte.
 */
struct TensorInfo
{
    uint32_t _M_name_hash = 0;
    uint32_t _M_elements = 0;
    uint32_t _M_data_offset = 0;
    uint32_t _M_data_length = 0;
    uint32_t _M_shape_hash = 0;
    uint16_t _M_name_length = 0;
    uint8_t _M_dtype = 0;
    uint8_t _M_rank = 0;
    uint8_t _M_flags = 0;
    uint8_t _M_meta = 0;
    uint8_t _M_reserved[2]{};

    void
    reset (const TensorRole role) noexcept
    {
        this->_M_name_hash = 0;
        this->_M_elements = 1;
        this->_M_data_offset = 0;
        this->_M_data_length = 0;
        this->_M_shape_hash = 2166136261u;
        this->_M_name_length = 0;
        this->_M_dtype = 0;
        this->_M_rank = 0;
        this->_M_flags = 0;
        this->_M_meta = static_cast<uint8_t> (
            (static_cast<uint8_t> (role) & 0x07u) << 5);
        this->_M_reserved[0] = 0;
        this->_M_reserved[1] = 0;
    }

    [[nodiscard]] TensorRole
    role () const noexcept
    {
        return static_cast<TensorRole> (this->_M_meta >> 5);
    }

    [[nodiscard]] DataType
    dtype () const noexcept
    {
        return static_cast<DataType> (this->_M_dtype);
    }

    [[nodiscard]] TensorDataField
    data_field () const noexcept
    {
        return static_cast<TensorDataField> (this->_M_meta & 0x0Fu);
    }

    void
    set_data_field (const TensorDataField field) noexcept
    {
        this->_M_meta = static_cast<uint8_t> (
            (this->_M_meta & 0xF0u) | (static_cast<uint8_t> (field) & 0x0Fu));
    }

    [[nodiscard]] uint8_t
    data_location () const noexcept
    {
        return static_cast<uint8_t> ((this->_M_meta >> 4) & 0x01u);
    }

    void
    set_data_location (const uint8_t location) noexcept
    {
        this->_M_meta = static_cast<uint8_t> (
            (this->_M_meta & 0xEFu) | ((location & 0x01u) << 4));
    }

    [[nodiscard]] bool
    is_external () const noexcept
    {
        return (this->_M_flags & TENSOR_FLAG_EXTERNAL) != 0u;
    }

    [[nodiscard]] bool
    is_dynamic () const noexcept
    {
        return (this->_M_flags & TENSOR_FLAG_DYNAMIC) != 0u;
    }

    void
    note_dim (const int64_t value, const bool symbolic) noexcept
    {
        if (this->_M_rank >= _S_max_tensor_rank)
            {
                this->_M_flags |= TENSOR_FLAG_OVERFLOW;
                return;
            }
        ++this->_M_rank;
        if (symbolic)
            this->_M_flags |= TENSOR_FLAG_DYNAMIC;
        if (value < 0)
            {
                this->_M_flags |= TENSOR_FLAG_INVALID;
                return;
            }
        const uint64_t dimension = static_cast<uint64_t> (value);
        this->_M_shape_hash = (this->_M_shape_hash * 33u)
                              ^ static_cast<uint32_t> (dimension)
                              ^ static_cast<uint32_t> (dimension >> 32);
        uint64_t product = 0;
        if (mul_overflow_u64 (static_cast<uint64_t> (this->_M_elements),
                              dimension, product))
            {
                this->_M_flags |= TENSOR_FLAG_OVERFLOW;
                this->_M_elements = 0xFFFFFFFFu;
                return;
            }
        if (product > _S_max_tensor_elements)
            {
                this->_M_flags |= TENSOR_FLAG_OVERFLOW;
                this->_M_elements = 0xFFFFFFFFu;
                return;
            }
        this->_M_elements = static_cast<uint32_t> (product);
    }

    /** @brief Shape product for a tensor whose dims were all emitted. */
    void
    finish_shape () noexcept
    {
        if (this->_M_rank == 0u)
            this->_M_elements = 1u;
    }
};

static_assert (sizeof (TensorInfo) == _S_tensor_budget_bytes,
               "onnx::TensorInfo must stay 28 bytes");
} // namespace zzs::mllee::onnx
