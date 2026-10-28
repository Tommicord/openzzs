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

#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief Lookup-table embedding, tied views and embedding bags.
 *
 * The module is the training twin of the inference EMBEDDING op
 * (`_S_embedding` in mllee/tensor_ops.hpp): a table of `vocab` rows of `dim`
 * int8 codes, addressed by token id, with the row copied out contiguously.  A
 * trained table therefore serializes and executes unchanged, which is the whole
 * point of keeping the gather in the code domain:
 *
 *   - the forward is `gather_rows (table, dim, index, rows, output)`, a pure
 *     copy, so an embedding layer costs no multiply and its output codes equal
 *     the table's codes for every non-padding row;
 *   - the backward is the transpose of that copy: a scatter-add of one output
 *     row's gradient lanes into the table row the index named.  It goes through
 *     `Trainable::add_grad` / `add_lanes` (never an overwrite) precisely because
 *     several tokens name the same row, and every clamped lane is counted.
 *
 * @section embed_padding Padding index
 *
 * `EmbedSpec::_M_padding_index` is a token id whose output row is all zeros and
 * whose table row receives no gradient at all, matching the inference convention
 * that a padding row is dead weight (an unused table row stays exactly where the
 * initializer put it).  -1 disables the behaviour.  The row is zeroed in the
 * forward rather than gathered from a zeroed table row, so a caller may reuse a
 * table whose padding row is not clean.
 *
 * @section embed_bag Bags
 *
 * A bag is one output row per variable-length token list: `offsets` holds
 * `rows + 1` monotonically increasing token positions and the row's codes are the
 * mean of the listed rows, `round (sum / length)` in code units.  An empty list
 * yields a zero row (there is nothing to average and no gradient to route).  The
 * backward applies the matching mean routing: each token in the list receives
 * `round (grad / length)` Q15 per lane, so a bag of length L costs each member
 * 1/L of the row's gradient -- the exact adjoint of the mean, not an approximation.
 * `EmbedSpec::_M_scale_grads` switches the division off for a sum-aggregation bag.
 *
 * @section embed_tying Tied views
 *
 * `tied (owner, shape, alias)` publishes a second view of a table that shares the
 * owner's code buffer *and* its gradient lanes: an output projection tied to the
 * embedding table accumulates into the same lanes the embedding writes, so one
 * optimizer pass updates both roles at once.  The two views are distinguished by
 * `grad_shared`, which an optimizer uses to visit a shared buffer only once.
 *
 * @section embed_pool Pool traffic
 *
 * With `N = rows * dim` output elements and `T` the index count (`rows` for a
 * lookup, `tokens` for a bag):
 *
 *   - saved index stream  : 2 T bytes (the backward re-reads the ids, and a
 *                           gather is not invertible without them);
 *   - saved offsets stream: 2 (rows + 1) bytes, bags only;
 *   - saved spec          : 16 bytes (four pool words).
 *
 * No accumulator or lane scratch is needed: the scatter writes straight into the
 * table's own gradient lanes.  The three regions are carved as three slabs, each
 * 4-byte aligned and stride-1, so `add_lanes` sees a SIMD-ready pair of streams
 * for every row fold and the table row it lands on is aligned whenever `dim` is
 * (a table with an odd `dim` still works: the fold is a saturating lane add, not
 * a vector load).
 */
namespace zzs::mllee::train
{
/** @brief Saved slot holding the spec copy the backward reads back. */
inline constexpr uint32_t _S_embed_spec = 0u;

/** @brief Saved slot holding the token id stream. */
inline constexpr uint32_t _S_embed_index = 1u;

/** @brief Saved slot holding the bag offsets, null for a plain lookup. */
inline constexpr uint32_t _S_embed_offsets = 2u;

/** @brief Saved slot kept empty: an embedding node never has a fourth operand. */
inline constexpr uint32_t _S_embed_unused = 3u;

/**
 * @brief One embedding layer: table geometry, id stream shape and padding rule.
 *
 * A plain POD saved in the pool, so the backward recovers the whole
 * configuration from the tape: no capture, no writable static.
 */
struct EmbedSpec
{
    /** @brief Output rows: tokens for a lookup, bags for a bag. */
    uint16_t _M_rows = 0u;
    /** @brief Index count: equal to @p _M_rows unless the layer is a bag. */
    uint16_t _M_tokens = 0u;
    /** @brief Table rows, i.e. the vocabulary size. */
    uint16_t _M_vocab = 0u;
    /** @brief Codes per table row. */
    uint16_t _M_dim = 0u;
    /** @brief Token id that gathers a zero row and collects no gradient; -1 = off. */
    int16_t _M_padding_index = -1;
    /** @brief Non-zero when the id stream is a bag list with offsets. */
    uint8_t _M_bag = 0u;
    /** @brief Bags only: divide the routed gradient by the list length. */
    uint8_t _M_scale_grads = 1u;
    /** @brief Op id recorded on the tape. */
    uint32_t _M_op_id = 0u;

    [[nodiscard]] constexpr bool
    is_bag () const noexcept
    {
        return this->_M_bag != 0u;
    }

    /** @brief True when a padding index is configured. */
    [[nodiscard]] constexpr bool
    has_padding () const noexcept
    {
        return this->_M_padding_index >= 0;
    }

    /** @brief Codes the table holds. */
    [[nodiscard]] constexpr uint32_t
    table_elements () const noexcept
    {
        return static_cast<uint32_t> (this->_M_vocab) * this->_M_dim;
    }

    /** @brief Codes the output holds. */
    [[nodiscard]] constexpr uint32_t
    output_elements () const noexcept
    {
        return static_cast<uint32_t> (this->_M_rows) * this->_M_dim;
    }

    /** @brief Ids the layer reads: @p _M_rows for a lookup, @p _M_tokens for a bag. */
    [[nodiscard]] constexpr uint32_t
    index_count () const noexcept
    {
        return this->is_bag () ? this->_M_tokens : this->_M_rows;
    }

    /** @brief Pool bytes one recorded layer needs, per the file header table. */
    [[nodiscard]] constexpr uint32_t
    bytes () const noexcept
    {
        const uint32_t index
            = align_up (2u * this->index_count (), _S_pool_alignment);
        const uint32_t offsets
            = this->is_bag ()
                  ? align_up (2u * (static_cast<uint32_t> (this->_M_rows) + 1u),
                              _S_pool_alignment)
                  : 0u;
        return index + offsets
               + align_up (static_cast<uint32_t> (sizeof (EmbedSpec)),
                           _S_pool_alignment);
    }

    [[nodiscard]] constexpr Shape
    output_shape () const noexcept
    {
        return Shape::matrix (this->_M_rows, this->_M_dim);
    }

    [[nodiscard]] constexpr Shape
    table_shape () const noexcept
    {
        return Shape::matrix (this->_M_vocab, this->_M_dim);
    }

    /**
     * @brief Check the geometry and the padding rule.
     *
     * A zero-length output row count is not a valid layer (the module has no
     * empty-tensor path, exactly like `Shape::validate`), and an index count a
     * `uint16` id stream cannot describe is OUT_OF_RANGE.
     *
     * @return BAD_LAYOUT for a zero vocabulary, dimension or row count, a lookup
     *         whose token count disagrees with its row count (the two are equal by
     *         construction, so a mismatch is a caller bug), or a padding id at or
     *         above the vocabulary; OUT_OF_RANGE for a table or output the runtime
     *         cannot address.
     */
    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (this->_M_rows == 0u || this->_M_vocab == 0u || this->_M_dim == 0u)
            return Status::BAD_LAYOUT;
        if (!this->is_bag () && this->_M_tokens != this->_M_rows)
            return Status::BAD_LAYOUT;
        if (this->has_padding ()
            && static_cast<uint32_t> (this->_M_padding_index) >= this->_M_vocab)
            return Status::BAD_LAYOUT;
        const uint32_t table = this->table_elements ();
        const uint32_t output = this->output_elements ();
        if (table > _S_max_elements || output > _S_max_elements)
            return Status::OUT_OF_RANGE;
        return Status::OK;
    }
};

static_assert (sizeof (EmbedSpec) == 16u,
               "a saved embedding spec is four pool words");

/**
 * @brief Validate one id stream against the table, and count the padding ids.
 *
 * @return OUT_OF_RANGE for an id at or above the vocabulary (a gather would read
 *         outside the table, so it is a shape error, not a value to clamp) and
 *         BAD_LAYOUT for a bag whose offsets do not start at 0, run backwards, or
 *         end past the token count.  Offsets are only checked when @p offsets is
 *         non-null, i.e. for the bag form.
 */
[[nodiscard]] inline Status
embedding_check_ids (const EmbedSpec &spec, const uint16_t *const index,
                     const uint16_t *const offsets) noexcept
{
    if (index == nullptr)
        return Status::BAD_LAYOUT;
    const uint32_t count = spec.index_count ();
    for (uint32_t row = 0; row < count; ++row)
        if (index[row] >= spec._M_vocab)
            return Status::OUT_OF_RANGE;
    if (offsets == nullptr)
        return Status::OK;
    if (offsets[0] != 0u)
        return Status::BAD_LAYOUT;
    for (uint32_t bag = 0; bag < spec._M_rows; ++bag)
        {
            const uint32_t begin = offsets[bag];
            const uint32_t end = offsets[bag + 1u];
            if (end < begin || end > spec._M_tokens)
                return Status::BAD_LAYOUT;
        }
    return Status::OK;
}

/**
 * @brief Forward pass with no tape recording.
 *
 * The lookup form gathers @p spec rows of the table; the bag form averages the
 * listed rows.  @p offsets must be `rows + 1` entries for a bag and null
 * otherwise.  The output gradient lanes are zeroed here, because a forward always
 * precedes the backward walk that accumulates into them.
 *
 * @return BAD_LAYOUT for a shape/count mismatch, OUT_OF_RANGE for a bad id or
 *         offset stream, OK otherwise.
 */
[[nodiscard]] inline Status
embedding_forward (const EmbedSpec &spec, const uint16_t *const index,
                   const uint16_t *const offsets, const Trainable &table,
                   Trainable &output) noexcept
{
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    if (table._M_code == nullptr || output._M_code == nullptr)
        return Status::BAD_LAYOUT;
    if (table._M_count != spec.table_elements ()
        || output._M_count != spec.output_elements ())
        return Status::BAD_LAYOUT;
    const bool bag = spec.is_bag ();
    if (bag != (offsets != nullptr))
        return Status::BAD_LAYOUT;
    const Status ids = embedding_check_ids (spec, index, offsets);
    if (!status_ok (ids))
        return ids;

    const uint32_t dim = spec._M_dim;
    if (!bag)
        {
            gather_rows (table._M_code, dim, index, spec._M_rows,
                         output._M_code);
            if (spec.has_padding ())
                for (uint32_t row = 0; row < spec._M_rows; ++row)
                    if (index[row]
                        == static_cast<uint16_t> (spec._M_padding_index))
                        for (uint32_t column = 0; column < dim; ++column)
                            output._M_code[row * dim + column] = 0;
            output.zero_grad ();
            return Status::OK;
        }
    for (uint32_t row = 0; row < spec._M_rows; ++row)
        {
            const uint32_t begin = offsets[row];
            const uint32_t end = offsets[row + 1u];
            const uint32_t length = end - begin;
            int8_t *const destination = output._M_code + row * dim;
            if (length == 0u)
                {
                    for (uint32_t column = 0; column < dim; ++column)
                        destination[column] = 0;
                    continue;
                }
            for (uint32_t column = 0; column < dim; ++column)
                {
                    int32_t total = 0;
                    for (uint32_t token = begin; token < end; ++token)
                        {
                            const uint16_t id = index[token];
                            if (spec.has_padding ()
                                && id == static_cast<uint16_t> (
                                             spec._M_padding_index))
                                continue;
                            total += table._M_code[id * dim + column];
                        }
                    destination[column] = _S_i8_code (
                        div_round (total, static_cast<int32_t> (length)));
                }
        }
    output.zero_grad ();
    return Status::OK;
}

/**
 * @brief Backward of the embedding, in the `BackwardFn` signature.
 *
 * Recorded node layout: operands [table]; saved [spec, index stream, offsets |
 * null, empty].  A lookup folds each output row's lanes into the table row the id
 * names with `add_lanes` (the saturating lane add `Trainable::add_grad` is a
 * single-lane instance of), so a table row named by several tokens collects every
 * contribution; a bag divides the row's lanes by its length first, which is the
 * adjoint of the mean the forward took.
 */
[[nodiscard]] inline Status
embedding_backward (const Trainable &output, const Trainable &grad_output,
                    GradContext &context) noexcept
{
    const EmbedSpec *const slot = context.saved_as<EmbedSpec> (_S_embed_spec);
    if (slot == nullptr)
        return Status::BAD_LAYOUT;
    const EmbedSpec spec = *slot;
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    if (context._M_operands == nullptr || context._M_operand_count < 1u)
        return Status::BAD_LAYOUT;
    const Trainable &table = context._M_operands[0];
    if (output._M_count != spec.output_elements ()
        || table._M_count != spec.table_elements ())
        return Status::BAD_LAYOUT;
    if (grad_output._M_grad == nullptr)
        return Status::BAD_LAYOUT;
    if (table._M_grad == nullptr)
        return Status::OK;
    const uint16_t *const index = context.saved_as<uint16_t> (_S_embed_index);
    if (index == nullptr)
        return Status::BAD_LAYOUT;
    const uint16_t *const offsets
        = spec.is_bag () ? context.saved_as<uint16_t> (_S_embed_offsets) : nullptr;
    if (spec.is_bag () && offsets == nullptr)
        return Status::BAD_LAYOUT;

    const uint32_t dim = spec._M_dim;
    const int32_t padding = spec.has_padding ()
                                ? static_cast<int32_t> (spec._M_padding_index)
                                : -1;
    uint32_t saturated = 0u;
    if (!spec.is_bag ())
        {
            for (uint32_t row = 0; row < spec._M_rows; ++row)
                {
                    const uint32_t id = index[row];
                    if (static_cast<int32_t> (id) == padding)
                        continue;
                    saturated += add_lanes (table._M_grad + id * dim,
                                            grad_output._M_grad + row * dim, dim);
                }
            context._M_saturation_events += saturated;
            return Status::OK;
        }
    for (uint32_t row = 0; row < spec._M_rows; ++row)
        {
            const uint32_t begin = offsets[row];
            const uint32_t end = offsets[row + 1u];
            const uint32_t length = end - begin;
            if (length == 0u)
                continue;
            const int16_t *const grad_row = grad_output._M_grad + row * dim;
            for (uint32_t token = begin; token < end; ++token)
                {
                    const uint32_t id = index[token];
                    if (static_cast<int32_t> (id) == padding)
                        continue;
                    const uint32_t base = id * dim;
                    for (uint32_t column = 0; column < dim; ++column)
                        {
                            const int32_t lane
                                = spec._M_scale_grads != 0u
                                      ? div_round (grad_row[column],
                                                   static_cast<int32_t> (length))
                                      : grad_row[column];
                            if (lane == 0)
                                continue;
                            if (table.add_grad (base + column, lane))
                                ++saturated;
                        }
                }
        }
    context._M_saturation_events += saturated;
    return Status::OK;
}

/**
 * @brief Save the id streams, run the forward and record the node.
 *
 * The ids are copied into the pool because a gather is only invertible with them
 * and a `BackwardFn` may not read a tensor the tape did not save.  @p offsets is
 * read only for the bag form (`EmbedSpec::_M_bag` non-zero) and @p index must be
 * `index_count ()` entries either way.
 *
 * @return BAD_LAYOUT for a shape mismatch or a missing offset stream,
 *         OUT_OF_RANGE for an id or offset outside the table, ARENA_OVERFLOW when
 *         the pool cannot supply the state, INTERNAL when the tape is not open.
 */
[[nodiscard]] inline Status
embedding_run (Tape &tape, const EmbedSpec &spec, const uint16_t *const index,
               const uint16_t *const offsets, const Trainable &table,
               Trainable &output) noexcept
{
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    if (index == nullptr || (spec.is_bag () != (offsets == nullptr)))
        return Status::BAD_LAYOUT;
    if (table._M_count != spec.table_elements ()
        || output._M_count != spec.output_elements ())
        return Status::BAD_LAYOUT;
    const Status ids = embedding_check_ids (spec, index, offsets);
    if (!status_ok (ids))
        return ids;

    Saved slots[_S_max_operands];
    Trainable operands[_S_max_operands];
    slots[_S_embed_unused] = Saved{};
    {
        EmbedSpec *saved_spec = nullptr;
        const Status save = tape.save (1u, slots[_S_embed_spec], saved_spec);
        if (!status_ok (save))
            return save;
        *saved_spec = spec;
    }
    {
        uint16_t *saved_index = nullptr;
        const Status save = tape.save (spec.index_count (),
                                       slots[_S_embed_index], saved_index);
        if (!status_ok (save))
            return save;
        for (uint32_t row = 0; row < spec.index_count (); ++row)
            saved_index[row] = index[row];
    }
    if (spec.is_bag ())
        {
            uint16_t *saved_offsets = nullptr;
            const Status save
                = tape.save (static_cast<uint32_t> (spec._M_rows) + 1u,
                             slots[_S_embed_offsets], saved_offsets);
            if (!status_ok (save))
                return save;
            for (uint32_t row = 0; row <= spec._M_rows; ++row)
                saved_offsets[row] = offsets[row];
        }

    const Status forward
        = embedding_forward (spec, index, offsets, table, output);
    if (!status_ok (forward))
        return forward;

    operands[0] = table;
    for (uint32_t index_slot = 1u; index_slot < _S_max_operands; ++index_slot)
        operands[index_slot] = Trainable{};
    return tape.record (embedding_backward, spec._M_op_id, output, operands, 1u,
                        slots, _S_max_operands);
}

/**
 * @brief A tied view: @p alias reads @p owner's codes and writes @p owner's
 *        gradient lanes.
 *
 * Weight tying is a storage fact, not an op: an output projection that shares the
 * embedding table is the same buffer under a different shape, so every op that
 * uses either name accumulates into one lane array and one optimizer step moves
 * both roles.  @p alias must describe the same element count as @p owner under
 * @p shape, and @p owner must own gradient lanes (a forward-only view cannot be
 * tied to anything).
 *
 * @return BAD_LAYOUT for a count mismatch or a missing gradient buffer.
 */
[[nodiscard]] inline constexpr Status
tied (const Trainable &owner, const Shape shape, Trainable &alias) noexcept
{
    if (owner._M_code == nullptr || owner._M_grad == nullptr)
        return Status::BAD_LAYOUT;
    const Status shape_status = shape.validate ();
    if (!status_ok (shape_status))
        return shape_status;
    if (shape.elements () != owner._M_count)
        return Status::BAD_LAYOUT;
    alias = owner;
    alias._M_shape = shape;
    alias._M_flags = owner._M_flags | Flag::RequiresGrad;
    return Status::OK;
}

/** @brief True when the two views accumulate into one gradient buffer. */
[[nodiscard]] inline constexpr bool
grad_shared (const Trainable &first, const Trainable &second) noexcept
{
    return first._M_grad != nullptr && first._M_grad == second._M_grad;
}

/** @brief True when the two views read one code buffer as well as one grad buffer. */
[[nodiscard]] inline constexpr bool
fully_tied (const Trainable &first, const Trainable &second) noexcept
{
    return grad_shared (first, second) && first._M_code == second._M_code;
}
} // namespace zzs::mllee::train
