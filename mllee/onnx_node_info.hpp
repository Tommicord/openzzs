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
#include "onnx_hash.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/** @brief NodeInfo flag bits. */
enum NodeFlag : uint8_t
{
    NODE_FLAG_HAS_OP = 1u << 0,
    NODE_FLAG_HAS_DOMAIN = 1u << 1,
    NODE_FLAG_HAS_NAME = 1u << 2,
    NODE_FLAG_DEFAULT_DOMAIN = 1u << 3,
    NODE_FLAG_COUNT_OVERFLOW = 1u << 4,
    NODE_FLAG_UNKNOWN_OP = 1u << 5,
};

/**
 * @brief The 28-byte NodeProto header retained while a node streams past.
 *
 * The loader does not keep the op_type string.  It stores two independent
 * hashes plus the length, which the registry in onnx_ops.hpp matches as a
 * tuple; the chance of both a FNV-1a and a djb2 collision at the same length is
 * below the error rate of the Flash the model lives in.  Input/output values
 * are emitted as events and never accumulated.
 */
struct NodeInfo
{
    uint32_t _M_op_hash = 0;
    uint32_t _M_op_hash2 = 0;
    uint32_t _M_domain_hash = 0;
    uint32_t _M_name_hash = 0;
    uint16_t _M_op_length = 0;
    uint16_t _M_domain_length = 0;
    uint16_t _M_name_length = 0;
    uint8_t _M_input_count = 0;
    uint8_t _M_output_count = 0;
    uint16_t _M_attribute_count = 0;
    uint8_t _M_flags = 0;
    uint8_t _M_reserved = 0;

    void
    reset () noexcept
    {
        this->_M_op_hash = 0;
        this->_M_op_hash2 = 0;
        this->_M_domain_hash = 0;
        this->_M_name_hash = 0;
        this->_M_op_length = 0;
        this->_M_domain_length = 0;
        this->_M_name_length = 0;
        this->_M_input_count = 0;
        this->_M_output_count = 0;
        this->_M_attribute_count = 0;
        this->_M_flags = NODE_FLAG_DEFAULT_DOMAIN;
        this->_M_reserved = 0;
    }

    void
    set_op (const uint32_t hash, const uint32_t hash2,
            const uint32_t length) noexcept
    {
        this->_M_op_hash = hash;
        this->_M_op_hash2 = hash2;
        this->_M_op_length = saturate_u16 (length);
        this->_M_flags |= NODE_FLAG_HAS_OP;
        if (length > 0xFFFFu)
            this->_M_flags |= NODE_FLAG_COUNT_OVERFLOW;
    }

    void
    set_domain (const uint32_t hash, const uint32_t length) noexcept
    {
        this->_M_domain_hash = hash;
        this->_M_domain_length = saturate_u16 (length);
        this->_M_flags |= NODE_FLAG_HAS_DOMAIN;
        this->_M_flags &= static_cast<uint8_t> (~NODE_FLAG_DEFAULT_DOMAIN);
        if (length == 0u)
            this->_M_flags |= NODE_FLAG_DEFAULT_DOMAIN;
    }

    void
    set_name (const uint32_t hash, const uint32_t length) noexcept
    {
        this->_M_name_hash = hash;
        this->_M_name_length = saturate_u16 (length);
        this->_M_flags |= NODE_FLAG_HAS_NAME;
    }

    void
    note_input () noexcept
    {
        if (!saturating_increment (this->_M_input_count, static_cast<uint8_t> (0xFFu)))
            this->_M_flags |= NODE_FLAG_COUNT_OVERFLOW;
    }

    void
    note_output () noexcept
    {
        if (!saturating_increment (this->_M_output_count, static_cast<uint8_t> (0xFFu)))
            this->_M_flags |= NODE_FLAG_COUNT_OVERFLOW;
    }

    void
    note_attribute () noexcept
    {
        if (!saturating_increment (this->_M_attribute_count,
                                    static_cast<uint16_t> (0xFFFFu)))
            this->_M_flags |= NODE_FLAG_COUNT_OVERFLOW;
    }

    [[nodiscard]] bool
    is_default_domain () const noexcept
    {
        return (this->_M_flags & NODE_FLAG_DEFAULT_DOMAIN) != 0u;
    }
};

static_assert (sizeof (NodeInfo) == _S_node_budget_bytes,
               "onnx::NodeInfo must stay 28 bytes");
} // namespace zzs::mllee::onnx
