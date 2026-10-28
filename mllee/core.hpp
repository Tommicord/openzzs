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

#include "mllee/format.hpp"
#include "mllee/ops.hpp"
#include "mllee/packlow.hpp"
#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/tensor_ops.hpp"
#include "std/maybe.hpp"

#include <cstdint>

namespace zzs::mllee
{
/**
 * @brief Heap-free model executor.
 *
 * All writable state is inside this object.  The model blob (weights, layer
 * descriptors, scales, biases, LUTs and the optional dictionary) is Flash
 * resident and is never copied.  Weights are streamed through the 64-byte
 * packlow ring, which is also the convolution row staging buffer.
 */
class Core
{
  public:
    struct Output
    {
        const int8_t *_M_data;
        uint16_t _M_len;
    };

    using RunResult = zzs::base::Maybe<Output, Error>;

    Core () noexcept = default;
    Core (const Core &) = delete;
    Core &operator= (const Core &) = delete;

    [[nodiscard]] ZZS_MLLEE_NOINLINE RunResult
    run (const uint8_t *model, const uint32_t model_len, const int8_t *input,
         const uint32_t input_len) noexcept
    {
        this->_M_status = static_cast<uint8_t> (Status::OK);
        Header &header = this->_M_ws._M_header;
        Layer &layer = this->_M_ws._M_layer;
        Status status = parse_header (model, model_len, header);
        if (status != Status::OK)
            return this->_M_fail (status);

        const int8_t *last_data =
            reinterpret_cast<const int8_t *> (this->_M_arena);
        uint16_t last_len = 0;

        for (uint32_t index = 0; index < header._M_layer_count; ++index)
            {
                const uint32_t layer_offset =
                    header._M_layer_table_off + index * _S_layer_desc_size;
                status = parse_layer (model, model_len, layer_offset, layer);
                if (status != Status::OK)
                    return this->_M_fail (status);

                status = _S_validate_tables (model_len, layer);
                if (status != Status::OK)
                    return this->_M_fail (status);

                const uint32_t required_input = _S_required_input (layer);
                const int8_t *source;
                if ((layer._M_flags & INPUT_EXTERNAL) != 0)
                    {
                        if (input == nullptr || input_len < required_input)
                            return this->_M_fail (Status::OUT_OF_RANGE);
                        source = input;
                    }
                else
                    {
                        if (required_input > layer._M_in_len
                            || !_S_span_ok (layer._M_in_off, required_input, header._M_arena_bytes))
                            return this->_M_fail (Status::ARENA_OVERFLOW);
                        source = reinterpret_cast<const int8_t *> (this->_M_arena
                                                                      + layer._M_in_off);
                    }

                const uint32_t required_output = _S_required_output (layer);
                if (required_output > layer._M_out_len
                    || !_S_span_ok (layer._M_out_off, required_output, header._M_arena_bytes))
                    return this->_M_fail (Status::ARENA_OVERFLOW);
                int8_t *const destination =
                    reinterpret_cast<int8_t *> (this->_M_arena + layer._M_out_off);

                if (layer._M_op == Op::LINEAR || layer._M_op == Op::CONV1D)
                    {
                        status = _S_pk_begin (this->_M_ws._M_pk, model, model_len,
                                              layer._M_weight_off, layer._M_weight_len,
                                              layer._M_codec);
                        if (status != Status::OK)
                            return this->_M_fail (status);
                        _S_rd_begin (this->_M_ws._M_pk, _S_weight_lut (model, layer),
                                     layer._M_dtype);
                    }

                status = _S_execute (this->_M_ws, model, this->arena (), source, destination);
                if (status != Status::OK)
                    return this->_M_fail (status);

                last_data = destination;
                last_len = layer._M_out_len;
            }

        return RunResult (Output{ last_data, last_len });
    }

    [[nodiscard]] Status
    status () const noexcept
    {
        return static_cast<Status> (this->_M_status);
    }

    [[nodiscard]] const int8_t *
    arena () const noexcept
    {
        return reinterpret_cast<const int8_t *> (this->_M_arena);
    }

    [[nodiscard]] int8_t *
    arena () noexcept
    {
        return reinterpret_cast<int8_t *> (this->_M_arena);
    }

    [[nodiscard]] static constexpr uint32_t
    ram_bytes () noexcept
    {
        return sizeof (Core);
    }

  private:
    [[nodiscard]] RunResult
    _M_fail (const Status status) noexcept
    {
        this->_M_status = static_cast<uint8_t> (status);
        return RunResult (Error (status));
    }

    [[nodiscard]] static constexpr uint32_t
    _S_required_input (const Layer &layer) noexcept
    {
        if (layer._M_op == Op::ATTENTION)
            return static_cast<uint32_t> (layer._M_seq) * layer._M_head_dim * _S_heads (layer) * 3u;
        return layer._M_in_dim;
    }

    [[nodiscard]] static constexpr uint32_t
    _S_required_output (const Layer &layer) noexcept
    {
        if (layer._M_op == Op::ATTENTION)
            return static_cast<uint32_t> (layer._M_seq) * layer._M_head_dim * _S_heads (layer);
        return layer._M_out_dim;
    }

    [[nodiscard]] static Status
    _S_validate_tables (const uint32_t model_len, const Layer &layer) noexcept
    {
        /* Number of 16-bit gamma and beta entries the kernel will index.  A
           negative result means the op carries no affine table at all. */
        int64_t affine = -1;
        switch (layer._M_op)
            {
            case Op::LINEAR:
            case Op::CONV1D:
                {
                    uint32_t rows = layer._M_out_dim;
                    if (layer._M_op == Op::CONV1D)
                        {
                            if (layer._M_seq == 0 || layer._M_out_dim % layer._M_seq != 0)
                                return Status::BAD_LAYOUT;
                            rows = layer._M_out_dim / layer._M_seq;
                        }
                    if (layer._M_scale_off != 0)
                        {
                            const uint32_t groups
                                = layer._M_group_size != 0 ? layer._M_groups : 1u;
                            if (!_S_span_ok (layer._M_scale_off, rows * groups * 4u, model_len))
                                return Status::BAD_LAYOUT;
                        }
                    if (layer._M_bias_off != 0 && !_S_span_ok (layer._M_bias_off, rows * 2u,
                                                               model_len))
                        return Status::BAD_LAYOUT;
                    return Status::OK;
                }
            case Op::SOFTMAX:
            case Op::LOG_SOFTMAX:
            case Op::CLAMP:
            case Op::DROPOUT:
            case Op::ADD:
            case Op::SUB:
            case Op::MUL:
            case Op::L2_NORMALIZE:
            case Op::MASKED_FILL:
            case Op::FLATTEN:
            case Op::SLICE:
            case Op::PAD1D:
            case Op::CONCAT:
            case Op::TRANSPOSE2D:
            case Op::UPSAMPLE_NEAREST:
            case Op::MAX_POOL1D:
            case Op::AVG_POOL1D:
            case Op::GLOBAL_AVG_POOL:
            case Op::REDUCE:
            case Op::ARGMAX:
            case Op::MATMUL:
            case Op::EMBEDDING:
            case Op::ATTENTION:
            case Op::ACTIVATION:
            default:
                return Status::OK;
            case Op::PRELU:
            case Op::NORM:
                affine = layer._M_in_dim;
                break;
            case Op::BATCH_NORM:
                {
                    const uint32_t window
                        = layer._M_seq != 0 ? layer._M_seq : layer._M_in_dim;
                    if (window == 0 || layer._M_in_dim % window != 0)
                        return Status::BAD_LAYOUT;
                    affine = static_cast<int64_t> (layer._M_in_dim / window);
                    break;
                }
            case Op::GROUP_NORM:
                {
                    const uint32_t window = _S_group_len (layer);
                    if (window == 0 || layer._M_in_dim % window != 0)
                        return Status::BAD_LAYOUT;
                    affine = static_cast<int64_t> (layer._M_in_dim / window);
                    break;
                }
            }
        if (layer._M_scale_off != 0
            && !_S_span_ok (layer._M_scale_off, static_cast<uint32_t> (affine) * 2u, model_len))
            return Status::BAD_LAYOUT;
        if (layer._M_bias_off != 0
            && !_S_span_ok (layer._M_bias_off, static_cast<uint32_t> (affine) * 2u, model_len))
            return Status::BAD_LAYOUT;
        return Status::OK;
    }

    Workspace _M_ws{};
    uint8_t _M_arena[_S_arena_bytes]{};
    uint8_t _M_status = static_cast<uint8_t> (Status::OK);
    uint8_t _M_pad[3]{};
};

// The module owns `_S_ram_budget_bytes` of writable RAM: this object plus the
// deepest call chain.  `_S_core_budget_bytes` is this object's share of that
// budget on a 32-bit target; the Cortex-M4 gate in tools/check_mllee_ram.sh
// measures the chain and enforces the sum.
static_assert (sizeof (Core)
                   <= (sizeof (void *) == 4 ? _S_core_budget_bytes
                                            : _S_core_budget_host_bytes),
               "mllee Core object exceeds the RAM budget");
static_assert (sizeof (void *) != 4 || sizeof (Core) <= _S_core_budget_bytes,
               "mllee Core exceeds its share of the module RAM budget on a 32-bit target");
static_assert (sizeof (void *) != 4 || alignof (Core) <= 4,
               "mllee Core must stay 4-byte aligned on a 32-bit target");
static_assert (sizeof (Workspace) <= 208, "mllee workspace exceeds its budget");
static_assert (sizeof (PacklowState) <= 136, "mllee codec state exceeds its budget");
static_assert (_S_arena_bytes == 444, "mllee activation arena size changed");
} // namespace zzs::mllee
