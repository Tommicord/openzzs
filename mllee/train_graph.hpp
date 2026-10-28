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
#include "mllee/train_fixed.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Reverse-mode gradient tape over caller-owned storage.
 *
 * The tape is the one piece of the training path that has to hold a graph, and
 * a graph in a heap-free module is a flat array with an ordering guarantee
 * instead of pointers: a node is appended only after every node that produces
 * its inputs, so recording order is already topological order and the backward
 * pass is one reverse walk.  No recursion, no allocation, no pointer chasing --
 * exactly the properties the inference core keeps for stack reasons.
 *
 * Contract for the op modules (train_linear.hpp, train_conv.hpp, ...):
 *
 *   - an op forward function computes its output tensor and calls Tape::record
 *     with its BackwardFn, its operand views and the saved tensors it needs to
 *     differentiate (activations, pre-activations, masks);
 *   - the BackwardFn receives the gradient of the loss with respect to its
 *     output and must accumulate into each operand's gradient lanes with
 *     Trainable::add_grad (saturating, never overwrite);
 *   - a backward function may not call Tape::record, allocate, or read a tensor
 *     the tape did not save.
 *
 * Gradients are Q15 lanes scaled by the tape's loss scale, so a loss term in
 * log units and a parameter in code units stay in the same integer domain
 * (mllee/train_loss_scale.hpp adjusts the scale, never the lanes).
 */
namespace zzs::mllee::train
{
/** @brief Largest fan-in one node may have (input, weight, bias, mask). */
inline constexpr uint32_t _S_max_operands = 4u;

/** @brief Default node capacity of a tape carved from the training pool. */
inline constexpr uint32_t _S_default_tape_nodes = 64u;

/** @brief Saved-tensor handle: a pool slab plus the element count it holds. */
struct Saved
{
    uint32_t _M_offset = 0u;
    uint32_t _M_bytes = 0u;
    uint32_t _M_count = 0u;

    [[nodiscard]] constexpr bool
    is_null () const noexcept
    {
        return this->_M_count == 0u || this->_M_bytes == 0u;
    }
};

/**
 * @brief Everything a backward function needs, passed by reference so no frame
 *        copies a node.
 *
 * @p operands are in the order recorded; @p saved is in the order the forward
 * call saved them.  A backward function must not write @p output.
 */
struct GradContext
{
    Pool *_M_pool = nullptr;
    const Trainable *_M_operands = nullptr;
    uint32_t _M_operand_count = 0u;
    const Saved *_M_saved = nullptr;
    uint32_t _M_saved_count = 0u;
    int32_t _M_loss_scale_q15 = _S_q15_one;
    uint32_t _M_saturation_events = 0u;

    /** @brief Typed view of a saved tensor, or null when the slot is not T-sized. */
    template <typename T>
    [[nodiscard]] T *
    saved_as (const uint32_t index) const noexcept
    {
        if (index >= this->_M_saved_count)
            return nullptr;
        const Saved &handle = this->_M_saved[index];
        if (handle._M_bytes < handle._M_count * sizeof (T))
            return nullptr;
        return reinterpret_cast<T *> (this->_M_pool->at (handle._M_offset));
    }
};

/** @brief Bytes a saved handle spans; kept for the RAM gate's accounting. */
[[nodiscard]] inline constexpr uint32_t
saved_bytes (const Saved &handle) noexcept
{
    return handle._M_bytes;
}

/** @brief Backward signature of a recorded op. */
using BackwardFn = Status (*)(const Trainable &output,
                              const Trainable &grad_output,
                              GradContext &context);

/** @brief One recorded operation: the op, its tensors, and its saved state. */
struct Node
{
    BackwardFn _M_backward = nullptr;
    Trainable _M_output;
    Trainable _M_operands[_S_max_operands];
    Saved _M_saved[_S_max_operands];
    uint16_t _M_operand_count = 0u;
    uint16_t _M_saved_count = 0u;
    uint32_t _M_op_id = 0u;
};

/**
 * @brief Tape counters, reported by the trainer and the RAM gate.
 *
 * @p _M_nodes, @p _M_saved_bytes and @p _M_saturation_events describe the pass
 * in progress and are cleared by begin(); only @p _M_peak_nodes is a run
 * total, because the RAM gate needs the high-water mark and nothing else does.
 */
struct TapeStats
{
    uint32_t _M_capacity = 0u;
    uint32_t _M_nodes = 0u;
    uint32_t _M_peak_nodes = 0u;
    uint32_t _M_saved_bytes = 0u;
    uint32_t _M_saturation_events = 0u;
};

/**
 * @brief Fixed-capacity forward recording buffer plus the backward walk.
 *
 * The node array and every saved tensor come from the same Pool the caller
 * measured, so tape capacity is a number the RAM gate can check instead of a
 * runtime surprise.
 */
class Tape
{
  public:
    /** @brief Bind to caller-owned node storage and the training pool. */
    Tape (Node *const nodes, const uint32_t capacity, Pool &pool) noexcept
        : _M_nodes (nodes), _M_capacity (capacity), _M_pool (&pool),
          _M_count (0u), _M_peak (0u), _M_saved_bytes (0u),
          _M_saturation (0u), _M_loss_scale_q15 (_S_q15_one), _M_open (false)
    {
    }

    Tape (const Tape &) = delete;
    Tape &operator= (const Tape &) = delete;

    /** @brief Start a pass; the caller must finish or reset before recording again. */
    [[nodiscard]] Status
    begin () noexcept
    {
        if (this->_M_nodes == nullptr || this->_M_capacity == 0u)
            return Status::BAD_LAYOUT;
        this->_M_count = 0u;
        this->_M_saved_bytes = 0u;
        this->_M_saturation = 0u;
        this->_M_open = true;
        return Status::OK;
    }

    /**
     * @brief True between begin() and reset().
     *
     * A forward hook that also serves evaluation uses this to skip recording, so
     * one hook body covers train and eval instead of two copies that can drift.
     */
    [[nodiscard]] constexpr bool
    is_open () const noexcept
    {
        return this->_M_open;
    }

    /** @brief Scale applied to every seeded gradient; owned by the loss-scale module. */
    void
    set_loss_scale (const int32_t scale_q15) noexcept
    {
        this->_M_loss_scale_q15 = q15_clamp (scale_q15);
    }

    [[nodiscard]] int32_t
    loss_scale () const noexcept
    {
        return this->_M_loss_scale_q15;
    }

    /** @brief Save one tensor's worth of state for a backward function. */
    template <typename T>
    [[nodiscard]] Status
    save (const uint32_t count, Saved &out, T *&values) noexcept
    {
        if (!this->_M_open)
            return Status::INTERNAL;
        Slab slab;
        const Status status
            = this->_M_pool->carve (count, static_cast<uint32_t> (sizeof (T)), slab);
        if (!status_ok (status))
            {
                out = Saved{};
                values = nullptr;
                return status;
            }
        values = this->_M_pool->template slab_view<T> (slab);
        if (values == nullptr)
            return Status::BAD_LAYOUT;
        for (uint32_t i = 0; i < count; ++i)
            values[i] = T{};
        out._M_offset = slab._M_offset;
        out._M_bytes = slab._M_bytes;
        out._M_count = count;
        this->_M_saved_bytes += slab._M_bytes;
        return Status::OK;
    }

    /** @brief Copy a source buffer into pool storage and register it as saved. */
    template <typename T>
    [[nodiscard]] Status
    save_copy (const T *const source, const uint32_t count, Saved &out) noexcept
    {
        T *values = nullptr;
        const Status status = this->save (count, out, values);
        if (!status_ok (status))
            return status;
        for (uint32_t i = 0; i < count; ++i)
            values[i] = source[i];
        return Status::OK;
    }

    /**
     * @brief Record one operation.
     *
     * The output must be a materialized tensor; operands with no gradient lane
     * are allowed (inputs and constants) and are simply skipped by a backward
     * function that writes through Trainable::add_grad.
     */
    [[nodiscard]] Status
    record (const BackwardFn backward_fn, const uint32_t op_id,
            const Trainable &output, const Trainable *const operands,
            const uint32_t operand_count, const Saved *const saved,
            const uint32_t saved_count) noexcept
    {
        if (!this->_M_open)
            return Status::INTERNAL;
        if (backward_fn == nullptr || operand_count > _S_max_operands
            || saved_count > _S_max_operands)
            return Status::BAD_LAYOUT;
        if (this->_M_count >= this->_M_capacity)
            return Status::ARENA_OVERFLOW;
        Node &node = this->_M_nodes[this->_M_count];
        node._M_backward = backward_fn;
        node._M_output = output;
        node._M_operand_count = static_cast<uint16_t> (operand_count);
        node._M_saved_count = static_cast<uint16_t> (saved_count);
        node._M_op_id = op_id;
        for (uint32_t i = 0; i < operand_count; ++i)
            node._M_operands[i] = operands[i];
        for (uint32_t i = 0; i < saved_count; ++i)
            node._M_saved[i] = saved[i];
        ++this->_M_count;
        if (this->_M_count > this->_M_peak)
            this->_M_peak = this->_M_count;
        return Status::OK;
    }

    /**
     * @brief Backward pass: seed @p loss with the loss scale, then walk nodes in
     *        reverse recording order.
     *
     * @p loss is the scalar output of a loss op (one element, gradient lanes of
     * length one).  Every node whose output has no gradient written stays inert,
     * which is how a frozen branch costs nothing.
     */
    [[nodiscard]] Status
    backward (const Trainable &loss) noexcept
    {
        if (!status_ok (loss.validate ()))
            return Status::BAD_LAYOUT;
        if (loss._M_grad == nullptr)
            return Status::BAD_LAYOUT;
        loss._M_grad[0] = static_cast<int16_t> (
            q15_clamp (this->_M_loss_scale_q15));
        for (uint32_t step = this->_M_count; step-- > 0u;)
            {
                Node &node = this->_M_nodes[step];
                const Trainable &grad_output = node._M_output;
                GradContext context;
                context._M_pool = this->_M_pool;
                context._M_operands = node._M_operands;
                context._M_operand_count = node._M_operand_count;
                context._M_saved = node._M_saved;
                context._M_saved_count = node._M_saved_count;
                context._M_loss_scale_q15 = this->_M_loss_scale_q15;
                context._M_saturation_events = 0u;
                const Status status
                    = node._M_backward (node._M_output, grad_output, context);
                this->_M_saturation += context._M_saturation_events;
                if (!status_ok (status))
                    return status;
            }
        return Status::OK;
    }

    /** @brief Drop the pass: nodes and saved bytes are pool-carved, so a reset is free. */
    void
    reset () noexcept
    {
        this->_M_count = 0u;
        this->_M_saved_bytes = 0u;
        this->_M_open = false;
        this->_M_pool->reset ();
    }

    [[nodiscard]] TapeStats
    stats () const noexcept
    {
        return TapeStats{ this->_M_capacity, this->_M_count, this->_M_peak,
                          this->_M_saved_bytes, this->_M_saturation };
    }

    [[nodiscard]] uint32_t
    node_count () const noexcept
    {
        return this->_M_count;
    }

    [[nodiscard]] Node *
    node (const uint32_t index) noexcept
    {
        return index < this->_M_count ? this->_M_nodes + index : nullptr;
    }

    [[nodiscard]] Pool &
    pool () noexcept
    {
        return *this->_M_pool;
    }

    /** @brief Zero the gradient lanes of every parameter in @p params. */
    static void
    zero_grad (const Trainable *const params, const uint32_t count) noexcept
    {
        for (uint32_t i = 0; i < count; ++i)
            params[i].zero_grad ();
    }

  private:
    Node *_M_nodes;
    uint32_t _M_capacity;
    Pool *_M_pool;
    uint32_t _M_count;
    uint32_t _M_peak;
    uint32_t _M_saved_bytes;
    uint32_t _M_saturation;
    int32_t _M_loss_scale_q15;
    bool _M_open;
};

/** @brief Bytes a tape of @p nodes nodes costs, for the training RAM gate. */
[[nodiscard]] inline constexpr uint32_t
tape_node_bytes (const uint32_t nodes) noexcept
{
    return nodes * static_cast<uint32_t> (sizeof (Node));
}
} // namespace zzs::mllee::train
