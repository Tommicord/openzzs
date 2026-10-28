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
#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_tensor.hpp"
#include "mllee/train_trainer.hpp"

#include <cstdint>

/**
 * @brief Resumable run-state checkpoints for a device with no filesystem.
 *
 * This is the mllee twin of a torch checkpoint: parameter values, gradient
 * buffers and the trainer's progress counters in one self-validating byte
 * block.  A device has no file system and no heap to read one through, so the
 * checkpoint serialises into -- and restores from -- a caller-owned byte
 * buffer; shipping those bytes to flash, EEPROM or a peer is the caller's
 * business and this module never allocates or opens anything.
 *
 * Why the tape and pool state are deliberately NOT saved: a step's entire
 * forward graph (nodes, saved activations, scratch) is carved from the Pool
 * and the Trainer resets the pool and the tape on every exit path of every
 * step, including a failing hook.  Nothing that outlives a step lives in the
 * pool except the caller's own buffers (the parameter code and gradient
 * streams are application memory, never pool memory), so a resume only has to
 * reproduce the streams and the counters.  That reset-between-steps invariant
 * is what makes omitting graph state safe: after a restore the next step
 * records a fresh tape exactly as the step after a normal one would.
 *
 * Restore order around `Trainer::configure` (documented because configure
 * resets the run counters): configure() rebinds the parameter list and zeroes
 * the step index, saturation totals and loss extremes, while set_loss_scale()
 * -- the only public write path to the scale state machine -- resets the
 * clean-step patience window.  Therefore the surviving order is
 *
 *     trainer.configure (...)   // binds and resets; must run FIRST
 *     checkpoint.restore (...)  // streams, then step, then loss scale
 *     trainer.fit (...)         // continues from the restored step index
 *
 * restoring anything before configure() would simply be erased by it.  The
 * saved clean-step counter travels in the format for analysis and for a future
 * public write path, but after a restore the patience window restarts at zero
 * because set_loss_scale() clears it and no other API touches it; that is the
 * safe direction (it delays a scale growth, never triggers an early one).  The
 * best/first/last loss counters are likewise carried and returned in RunState;
 * the trainer rebuilds its private best tracker from the resumed run, so a
 * long-haul caller keeps the restored RunState values as its own truth.
 *
 * Every byte layout in this file is fixed-size little-endian, exactly like the
 * inference model container in mllee/format.hpp: fields are addressed through
 * the same _S_u16/_S_u32 accessors, the same CRC-16/CCITT folds the header
 * uses, and the same span checks (_S_span_ok) decide whether an offset is
 * trustworthy.  A checkpoint therefore survives a flash round trip with no
 * dependence on the host word order.
 *
 * Trust order on a restore: null buffer, then truncation, magic, version and
 * header size, then the length/offset/element table (alignment, monotonic
 * non-reordering streams, spans inside total_bytes), then the CRC over every
 * byte the structure promised, then the payload itself (-128 code rejection,
 * per-tensor element-count match, destination alignment).  A failing check
 * returns the Status that names it and touches nothing, so corrupt input is
 * structurally impossible to accept: every field is range-checked against the
 * caller's buffer length before a single payload byte is read.
 */
namespace zzs::mllee::train
{
/** @brief Checkpoint magic, the inference header's "MLEE" spelled for run state. */
inline constexpr uint32_t _S_ckpt_magic = 0x434B4C4Du; /* "MLKC" little-endian */

/** @brief Format version of this layout; a reader rejects anything else. */
inline constexpr uint32_t _S_ckpt_version = 1u;

/** @brief Bytes of the self-describing header block (eight words). */
inline constexpr uint32_t _S_ckpt_header_bytes = 32u;

/** @brief Words per parameter-table entry (code offset, grad offset, count, shape). */
inline constexpr uint32_t _S_ckpt_entry_words = 4u;

/** @brief Words of trainer run state carried next to the table (twelve, 48 bytes). */
inline constexpr uint32_t _S_ckpt_run_words = 12u;

/** @brief Header size field value: fixed prefix, so a reader can trust the table. */
inline constexpr uint32_t _S_ckpt_meta_bytes
    = _S_ckpt_header_bytes
      + _S_max_trainer_params * _S_ckpt_entry_words * 4u
      + _S_ckpt_run_words * 4u;

/**
 * @name Fixed-layout checkpoint structs
 *
 * These are the in-RAM mirrors of the serialised words.  They are PODs, and a
 * static_assert on every size pins the byte layout the reader and the writer
 * share; the serialised form is always addressed by word index, so the struct
 * padding on any one host is not part of the format.
 * @{
 */

/** @brief The header POD: magic, version, sizes, element total and checksum. */
struct CheckpointHeader
{
    uint32_t _M_magic = 0u;
    uint32_t _M_version = 0u;
    uint32_t _M_header_bytes = 0u;
    uint32_t _M_entry_count = 0u;
    uint32_t _M_element_total = 0u;
    uint32_t _M_total_bytes = 0u;
    uint32_t _M_reserved = 0u;
    uint32_t _M_checksum = 0u; /* CRC-16/CCITT in the low half; high half zero */
};

static_assert (sizeof (CheckpointHeader) == 32u,
               "checkpoint header is eight words; the tail pad is fixed too");

/** @brief One parameter's placement in the checkpoint streams. */
struct CheckpointEntry
{
    uint32_t _M_code_off = 0u;   /* byte offset of the int8 code stream */
    uint32_t _M_grad_off = 0u;   /* byte offset of the Q15 lane stream, 0 = none */
    uint32_t _M_count = 0u;      /* element count, the restore must match */
    uint32_t _M_shape_elements = 0u; /* Shape::elements () cross-check word */
};

static_assert (sizeof (CheckpointEntry) == 16u,
               "checkpoint entry is four words (offset, offset, count, shape)");

/** @brief Trainer progress carried verbatim so a fit run resumes, not restarts. */
struct RunState
{
    uint32_t _M_step = 0u;             /* context step index */
    int32_t _M_loss_scale_q15 = _S_q15_one; /* dynamic loss scale */
    uint32_t _M_clean_steps = 0u;      /* patience window at save time */
    int32_t _M_best_loss_q7 = 0;       /* best loss in Q7 log units */
    uint32_t _M_best_step = 0u;        /* step index that produced it */
    uint32_t _M_epochs = 0u;           /* completed epochs (fit sets 1 per call) */
    int32_t _M_first_loss_q7 = 0;
    int32_t _M_last_loss_q7 = 0;
    uint32_t _M_saturation_total = 0u;
    uint32_t _M_scale_halves = 0u;
    uint32_t _M_scale_grows = 0u;
    uint32_t _M_have_best = 0u;        /* bool widened to a word for layout */
};

static_assert (sizeof (RunState) == 48u, "run state is twelve checkpoint words");

/** @brief Parsed and fully validated checkpoint geometry. */
struct CheckpointInfo
{
    uint32_t _M_entry_count = 0u;
    uint32_t _M_element_total = 0u;
    uint32_t _M_param_off = 0u;   /* start of the int8 code stream */
    uint32_t _M_param_bytes = 0u; /* aligned span of the code stream */
    uint32_t _M_grad_off = 0u;    /* start of the Q15 lane stream */
    uint32_t _M_grad_bytes = 0u;  /* aligned span, 0 when no gradient carried */
    uint32_t _M_total_bytes = 0u;
};

static_assert (sizeof (CheckpointInfo) == 28u,
               "checkpoint info is seven words");

/**@}*/

/** @brief CRC-16/CCITT fold over one byte range, continuing a running value. */
[[nodiscard]] inline constexpr uint16_t
checkpoint_crc_fold (uint16_t crc, const uint8_t *const data,
                     const uint32_t length) noexcept
{
    for (uint32_t i = 0; i < length; ++i)
        crc = _S_crc16_update (crc, data[i]);
    return crc;
}

/**
 * @brief Byte count a checkpoint needs for @p params tensors holding @p elements
 *        codes in total (every tensor trainable, i.e. gradient lanes carried).
 *
 * A caller uses this to size its save buffer: the prefix is fixed, both
 * streams are aligned to the 4-byte pool alignment so the restore can copy
 * with the same stride-1 discipline the kernels use.  A parameter count the
 * serializer would reject, or a sum that overflows uint32, saturates the query
 * to UINT32_MAX instead of wrapping, so serialize fails closed against any real
 * buffer.
 */
[[nodiscard]] inline constexpr uint32_t
checkpoint_bytes (const uint32_t params, const uint32_t elements) noexcept
{
    if (params > _S_max_trainer_params)
        return 0xFFFFFFFFu;
    const uint64_t wide = elements;
    const uint64_t codes = (wide + _S_pool_alignment - 1u)
                           & ~static_cast<uint64_t> (_S_pool_alignment - 1u);
    const uint64_t lanes = (2u * wide + _S_pool_alignment - 1u)
                           & ~static_cast<uint64_t> (_S_pool_alignment - 1u);
    const uint64_t total = static_cast<uint64_t> (_S_ckpt_meta_bytes)
                           + codes + lanes;
    return total > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t> (total);
}

/**
 * @brief Fill a RunState from a trainer's public view of itself.
 *
 * Every field is read through the API Trainer exposes (context(), loss_scale(),
 * totals ()); _M_clean_steps has no public read path and is reported as the
 * trainer exposes it today (zero at a fresh configure), which the restore side
 * treats the same way.  Caller zeroes the struct first if it wants deterministic
 * unset fields; this writes all twelve.
 */
inline void
run_state_from_trainer (const Trainer &trainer, RunState &out) noexcept
{
    const StepContext &context = trainer.context ();
    const FitStats totals = trainer.totals ();
    out._M_step = context._M_step;
    out._M_loss_scale_q15 = trainer.loss_scale ();
    out._M_clean_steps = 0u; /* no public read; see the file header contract */
    out._M_best_loss_q7 = totals._M_best_loss_q7;
    out._M_best_step = totals._M_best_step;
    out._M_epochs = totals._M_epochs;
    out._M_first_loss_q7 = totals._M_first_loss_q7;
    out._M_last_loss_q7 = totals._M_last_loss_q7;
    out._M_saturation_total = totals._M_saturation_events;
    out._M_scale_halves = totals._M_scale_halves;
    out._M_scale_grows = totals._M_scale_grows;
    out._M_have_best = totals._M_best_loss_q7 != 0 || totals._M_best_step != 0u
                           ? 1u
                           : 0u;
}

/**
 * @brief Apply a RunState to a trainer: the surviving restore order.
 *
 * configure () first (it zeroes the step index and every private counter),
 * then this: step index and last loss go through the public context reference,
 * the loss scale through set_loss_scale (), which also re-arms the growth
 * patience at zero (the documented safe direction).  Best/epoch counters are
 * NOT pushed back in -- the trainer rebuilds them from the resumed run -- so
 * the caller keeps the RunState it restored as the cross-resume truth.
 */
inline void
run_state_to_trainer (Trainer &trainer, const RunState &state) noexcept
{
    trainer.context ()._M_step = state._M_step;
    trainer.context ()._M_loss_q7 = state._M_last_loss_q7;
    trainer.set_loss_scale (state._M_loss_scale_q15);
}

/**
 * @brief The resumable run-state object: bind once, save and restore repeatedly.
 *
 * bind () validates the parameter list and precomputes the stream offsets --
 * entries are laid out in caller order, and save/restore refuse any stream
 * that is not exactly at that offset, so a checkpoint can never silently
 * reorder the parameter list.
 */
class Checkpoint
{
  public:
    /**
     * @brief Validate @p params and precompute the layout.
     * @return BAD_LAYOUT for a null or invalid view, UNSUPPORTED for a list
     *         longer than _S_max_trainer_params (rejected before any math).
     */
    [[nodiscard]] Status
    bind (const Trainable *const params, const uint32_t param_count) noexcept
    {
        if (params == nullptr || param_count == 0u)
            return Status::BAD_LAYOUT;
        if (param_count > _S_max_trainer_params)
            return Status::UNSUPPORTED;
        uint32_t elements = 0u;
        for (uint32_t i = 0; i < param_count; ++i)
            {
                const Status status = params[i].validate ();
                if (!status_ok (status))
                    return status;
                elements += params[i]._M_count;
            }
        if (elements > _S_max_elements)
            return Status::OUT_OF_RANGE;
        this->_M_params = params;
        this->_M_count = param_count;
        this->_M_info._M_entry_count = param_count;
        this->_M_info._M_element_total = elements;
        this->_M_info._M_param_off = _S_ckpt_meta_bytes;
        this->_M_info._M_param_bytes = align_up (elements, _S_pool_alignment);
        this->_M_info._M_grad_off
            = this->_M_info._M_param_off + this->_M_info._M_param_bytes;
        uint32_t grad_bytes = 0u;
        for (uint32_t i = 0; i < param_count; ++i)
            if (params[i]._M_grad != nullptr)
                grad_bytes += align_up (2u * params[i]._M_count,
                                       _S_pool_alignment);
        this->_M_info._M_grad_bytes = grad_bytes;
        this->_M_info._M_total_bytes = this->_M_info._M_grad_off
                                       + this->_M_info._M_grad_bytes;
        return Status::OK;
    }

    [[nodiscard]] const CheckpointInfo &
    info () const noexcept
    {
        return this->_M_info;
    }

    /**
     * @brief Serialise streams and run state into the caller's buffer.
     *
     * @param out_bytes receives the exact byte count written.
     * @return BAD_LAYOUT when the checkpoint is not bound or a code is -128,
     *         ARENA_OVERFLOW when @p buffer_bytes is below the computed layout.
     */
    [[nodiscard]] Status
    serialize (uint8_t *const buffer, const uint32_t buffer_bytes,
               const RunState &run, uint32_t &out_bytes) const noexcept
    {
        if (this->_M_params == nullptr)
            return Status::BAD_LAYOUT;
        if (buffer == nullptr)
            return Status::BAD_LAYOUT;
        if (!simd_aligned (buffer))
            return Status::BAD_LAYOUT;
        if (buffer_bytes < this->_M_info._M_total_bytes)
            return Status::ARENA_OVERFLOW;
        /* The -128 rail is rejected before anything is written: inference
           never emits that code and a checkpoint must not smuggle one in. */
        for (uint32_t i = 0; i < this->_M_count; ++i)
            {
                const Trainable &param = this->_M_params[i];
                for (uint32_t k = 0; k < param._M_count; ++k)
                    if (param._M_code[k] == -128)
                        return Status::BAD_LAYOUT;
            }

        uint32_t *const words = reinterpret_cast<uint32_t *> (buffer);
        uint32_t code_off = this->_M_info._M_param_off;
        uint32_t grad_off = this->_M_info._M_grad_off;
        for (uint32_t i = 0; i < this->_M_count; ++i)
            {
                const Trainable &param = this->_M_params[i];
                this->write_entry (words, 8u + i * _S_ckpt_entry_words, param,
                                   code_off, grad_off);
                for (uint32_t k = 0; k < param._M_count; ++k)
                    buffer[code_off + k]
                        = static_cast<uint8_t> (param._M_code[k]);
                code_off += static_cast<uint32_t> (param._M_count);
                if (param._M_grad == nullptr)
                    continue;
                for (uint32_t k = 0; k < param._M_count; ++k)
                    {
                        const uint16_t lane
                            = static_cast<uint16_t> (param._M_grad[k]);
                        buffer[grad_off + 2u * k] = static_cast<uint8_t> (lane & 0xFFu);
                        buffer[grad_off + 2u * k + 1u]
                            = static_cast<uint8_t> (lane >> 8u);
                    }
                grad_off += static_cast<uint32_t> (
                    align_up (2u * param._M_count, _S_pool_alignment));
            }
        /* The code stream packs byte-adjacent (stride-1 copy), so the last
           tensor ends at the aligned boundary only when element_total is
           aligned; pad with 0xFF, a byte that can never be a code read back
           as part of a tensor because it sits past every entry's span. */
        for (uint32_t pad = code_off; pad < this->_M_info._M_grad_off; ++pad)
            buffer[pad] = 0xFFu;

        this->write_run (words, run);
        words[0] = _S_ckpt_magic;
        words[1] = _S_ckpt_version;
        words[2] = _S_ckpt_header_bytes;
        words[3] = this->_M_count;
        words[4] = this->_M_info._M_element_total;
        words[5] = this->_M_info._M_total_bytes;
        words[6] = 0u;
        words[7] = 0u;
        const uint16_t crc = this->checksum (buffer, this->_M_info._M_total_bytes);
        words[7] = crc;
        out_bytes = this->_M_info._M_total_bytes;
        return Status::OK;
    }

    /**
     * @brief Validate a checkpoint buffer and load it into the bound tensors.
     *
     * Runs the full trust chain from the file header on every rejection path,
     * in the documented order, and only copies payload bytes once the CRC and
     * the geometry agree.  Gradient lanes are restored raw (a rail lane is a
     * legitimate saved value); codes are re-checked for -128 as data, not as
     * structure, hence CORRUPT_STREAM rather than BAD_LAYOUT.
     */
    [[nodiscard]] Status
    deserialize (const uint8_t *const buffer, const uint32_t buffer_bytes,
                 RunState &run_out) const noexcept
    {
        CheckpointInfo info;
        const Status validation
            = Checkpoint::inspect (buffer, buffer_bytes, this->_M_params,
                                   this->_M_count, info, run_out);
        if (!status_ok (validation))
            return validation;
        for (uint32_t i = 0; i < this->_M_count; ++i)
            {
                const Trainable &param = this->_M_params[i];
                const uint32_t base = 8u + i * _S_ckpt_entry_words;
                const uint32_t code_off = buffer_words (buffer, base);
                const uint32_t grad_off = buffer_words (buffer, base + 1u);
                for (uint32_t k = 0; k < param._M_count; ++k)
                    param._M_code[k]
                        = static_cast<int8_t> (buffer[code_off + k]);
                if (grad_off == 0u || param._M_grad == nullptr)
                    continue;
                for (uint32_t k = 0; k < param._M_count; ++k)
                    param._M_grad[k] = _S_i16 (buffer + grad_off + 2u * k);
            }
        return Status::OK;
    }

    /**
     * @brief deserialize () plus the documented trainer-continuation order.
     *
     * Writes the streams first (a failed stream validation must not leave the
     * trainer half-resumed), then step/last-loss through the context, then the
     * loss scale.  After this returns OK, fit () continues the saved run.
     */
    [[nodiscard]] Status
    restore (const uint8_t *const buffer, const uint32_t buffer_bytes,
             Trainer &trainer, RunState &run_out) const noexcept
    {
        const Status status = this->deserialize (buffer, buffer_bytes, run_out);
        if (!status_ok (status))
            return status;
        run_state_to_trainer (trainer, run_out);
        return Status::OK;
    }

    /**
     * @brief Validate a checkpoint standalone and report geometry + run state.
     *
     * @param params destination list to match against (may be null for a pure
     *        corruption probe, in which case destination rules are skipped).
     * @return in trust order: BAD_LAYOUT (nulls), TRUNCATED (below the fixed
     *         prefix), BAD_MAGIC, BAD_VERSION, BAD_LAYOUT (header size field,
     *         unaligned or reordered or overflowing streams, count mismatch,
     *         misaligned destination), CORRUPT_STREAM (CRC, -128 codes).
     */
    [[nodiscard]] static Status
    inspect (const uint8_t *const buffer, const uint32_t buffer_bytes,
             const Trainable *const params, const uint32_t param_count,
             CheckpointInfo &info_out, RunState &run_out) noexcept
    {
        if (buffer == nullptr)
            return Status::BAD_LAYOUT;
        if (buffer_bytes < _S_ckpt_meta_bytes)
            return Status::TRUNCATED;
        if (!simd_aligned (buffer))
            return Status::BAD_LAYOUT;
        const uint32_t magic = buffer_words (buffer, 0u);
        if (magic != _S_ckpt_magic)
            return Status::BAD_MAGIC;
        const uint32_t version = buffer_words (buffer, 1u);
        if (version != _S_ckpt_version)
            return Status::BAD_VERSION;
        if (buffer_words (buffer, 2u) != _S_ckpt_header_bytes)
            return Status::BAD_LAYOUT;
        const uint32_t entry_count = buffer_words (buffer, 3u);
        if (entry_count > _S_max_trainer_params)
            return Status::BAD_LAYOUT;
        const uint32_t element_total = buffer_words (buffer, 4u);
        const uint32_t total_bytes = buffer_words (buffer, 5u);
        if (buffer_words (buffer, 6u) != 0u)
            return Status::BAD_LAYOUT;
        if (total_bytes < _S_ckpt_meta_bytes || total_bytes > buffer_bytes)
            return Status::BAD_LAYOUT;
        if (element_total == 0u || element_total > _S_max_elements)
            return Status::BAD_LAYOUT;

        const uint32_t param_off = _S_ckpt_meta_bytes;
        const uint32_t param_bytes = align_up (element_total, _S_pool_alignment);
        const uint32_t grad_off = param_off + param_bytes;
        uint32_t expect_code = param_off;
        uint32_t expect_grad = grad_off;
        uint32_t sum_counts = 0u;
        uint32_t grad_span = 0u;
        for (uint32_t i = 0; i < entry_count; ++i)
            {
                const uint32_t base = 8u + i * _S_ckpt_entry_words;
                const uint32_t code_off = buffer_words (buffer, base);
                const uint32_t entry_grad = buffer_words (buffer, base + 1u);
                const uint32_t count = buffer_words (buffer, base + 2u);
                const uint32_t shape_elements = buffer_words (buffer, base + 3u);
                if (count == 0u || count > _S_max_elements
                    || shape_elements != count)
                    return Status::BAD_LAYOUT;
                if (code_off != expect_code) /* offsets are explicit and in
                                                order; a reordered stream is
                                                corruption, not a permutation */
                    return Status::BAD_LAYOUT;
                if (!_S_span_ok (code_off, count, total_bytes))
                    return Status::BAD_LAYOUT;
                expect_code += count;
                sum_counts += count;
                if (entry_grad != 0u)
                    {
                        if ((entry_grad & (_S_pool_alignment - 1u)) != 0u)
                            return Status::BAD_LAYOUT;
                        if (entry_grad < grad_off
                            || !_S_span_ok (entry_grad, 2u * count, total_bytes))
                            return Status::BAD_LAYOUT;
                        if (entry_grad != expect_grad)
                            return Status::BAD_LAYOUT;
                        expect_grad
                            += align_up (2u * count, _S_pool_alignment);
                        grad_span = expect_grad - grad_off;
                    }
            }
        if (sum_counts != element_total)
            return Status::BAD_LAYOUT;
        if (grad_span != 0u
            && !_S_span_ok (grad_off, grad_span, total_bytes))
            return Status::BAD_LAYOUT;
        const uint32_t expected_total
            = expect_grad > grad_off ? expect_grad
                                      : param_off + param_bytes;
        if (total_bytes != expected_total)
            return Status::BAD_LAYOUT;

        if (checkpoint_crc_fold (checkpoint_crc_fold (0xFFFFu, buffer, 28u),
                                 buffer + 32u,
                                 total_bytes - 32u)
            != static_cast<uint16_t> (buffer_words (buffer, 7u)))
            return Status::CORRUPT_STREAM;

        if (params == nullptr && param_count != 0u)
            return Status::BAD_LAYOUT;
        if (param_count != 0u)
            {
                if (param_count != entry_count)
                    return Status::BAD_LAYOUT;
                for (uint32_t i = 0; i < param_count; ++i)
                    {
                        const Trainable &param = params[i];
                        if (!status_ok (param.validate ()))
                            return Status::BAD_LAYOUT;
                        const uint32_t base = 8u + i * _S_ckpt_entry_words;
                        const uint32_t count = buffer_words (buffer, base + 2u);
                        const uint32_t entry_grad = buffer_words (buffer, base + 1u);
                        if (count != param._M_count)
                            return Status::BAD_LAYOUT;
                        if ((entry_grad != 0u) != (param._M_grad != nullptr))
                            return Status::BAD_LAYOUT;
                        if (!param.simd_ready ())
                            return Status::BAD_LAYOUT;
                        const uint32_t code_off = buffer_words (buffer, base);
                        for (uint32_t k = 0; k < count; ++k)
                            if (static_cast<int8_t> (buffer[code_off + k]) == -128)
                                return Status::CORRUPT_STREAM;
                    }
            }
        read_run (buffer, run_out);
        info_out._M_entry_count = entry_count;
        info_out._M_element_total = element_total;
        info_out._M_param_off = param_off;
        info_out._M_param_bytes = param_bytes;
        info_out._M_grad_off = grad_off;
        info_out._M_grad_bytes = grad_span;
        info_out._M_total_bytes = total_bytes;
        return Status::OK;
    }

    /** @brief CRC-16/CCITT of the checkpoint: header minus the checksum word,
     *         then everything from payload start to total_bytes. */
    [[nodiscard]] static uint16_t
    checksum (const uint8_t *const buffer, const uint32_t total_bytes) noexcept
    {
        uint16_t crc = _S_crc16 (buffer, 28u);
        crc = checkpoint_crc_fold (crc, buffer + 32u, total_bytes - 32u);
        return crc;
    }

  private:
    static void
    write_entry (uint32_t *const words, const uint32_t base,
                 const Trainable &param, const uint32_t code_off,
                 const uint32_t grad_off) noexcept
    {
        words[base] = code_off;
        words[base + 1u] = param._M_grad != nullptr ? grad_off : 0u;
        words[base + 2u] = param._M_count;
        words[base + 3u] = param._M_shape.elements ();
    }

    static void
    write_run (uint32_t *const words, const RunState &run) noexcept
    {
        const uint32_t base = 8u + _S_max_trainer_params * _S_ckpt_entry_words;
        words[base] = run._M_step;
        words[base + 1u] = static_cast<uint32_t> (run._M_loss_scale_q15);
        words[base + 2u] = run._M_clean_steps;
        words[base + 3u] = static_cast<uint32_t> (run._M_best_loss_q7);
        words[base + 4u] = run._M_best_step;
        words[base + 5u] = run._M_epochs;
        words[base + 6u] = static_cast<uint32_t> (run._M_first_loss_q7);
        words[base + 7u] = static_cast<uint32_t> (run._M_last_loss_q7);
        words[base + 8u] = run._M_saturation_total;
        words[base + 9u] = run._M_scale_halves;
        words[base + 10u] = run._M_scale_grows;
        words[base + 11u] = run._M_have_best;
    }

    static void
    read_run (const uint8_t *const buffer, RunState &run) noexcept
    {
        const uint32_t base
            = 8u + _S_max_trainer_params * _S_ckpt_entry_words;
        run._M_step = buffer_words (buffer, base);
        run._M_loss_scale_q15 = static_cast<int32_t> (buffer_words (buffer, base + 1u));
        run._M_clean_steps = buffer_words (buffer, base + 2u);
        run._M_best_loss_q7 = static_cast<int32_t> (buffer_words (buffer, base + 3u));
        run._M_best_step = buffer_words (buffer, base + 4u);
        run._M_epochs = buffer_words (buffer, base + 5u);
        run._M_first_loss_q7 = static_cast<int32_t> (buffer_words (buffer, base + 6u));
        run._M_last_loss_q7 = static_cast<int32_t> (buffer_words (buffer, base + 7u));
        run._M_saturation_total = buffer_words (buffer, base + 8u);
        run._M_scale_halves = buffer_words (buffer, base + 9u);
        run._M_scale_grows = buffer_words (buffer, base + 10u);
        run._M_have_best = buffer_words (buffer, base + 11u);
    }

    /** @brief Little-endian word @p index of a validated buffer. */
    [[nodiscard]] static uint32_t
    buffer_words (const uint8_t *const buffer, const uint32_t word) noexcept
    {
        return _S_u32 (buffer + word * 4u);
    }

    const Trainable *_M_params = nullptr;
    uint32_t _M_count = 0u;
    CheckpointInfo _M_info;
};

/**
 * @name The frozen inference tensor bridge
 *
 * copy_out writes a trained parameter's code stream into exactly the bytes a
 * model container's Layer::_M_weight_off / _M_weight_len points at, and
 * copy_in reads a resident weight stream back into a trainable view for
 * distillation or fine-tuning.  Both go through the same span rule Core's
 * parser uses (_S_span_ok) and both refuse the -128 rail, so a checkpoint
 * export round-trips into the bytes Core validates without a conversion step.
 * @{
 */

/**
 * @brief Export one parameter's codes into a model weight stream.
 * @return BAD_LAYOUT for a null/misaligned model pointer or an out-of-range
 *         offset, OUT_OF_RANGE when the tensor is larger than @p weight_len or
 *         the span leaves the model, and any -128 code present is reported as
 *         CORRUPT_STREAM (the export must not silently clamp a bad training
 *         buffer into a deployable one).
 */
[[nodiscard]] inline Status
copy_out (const Trainable &param, uint8_t *const model,
          const uint32_t model_len, const uint32_t weight_off,
          const uint32_t weight_len) noexcept
{
    if (model == nullptr || !status_ok (param.validate ()))
        return Status::BAD_LAYOUT;
    if (!simd_aligned (model))
        return Status::BAD_LAYOUT;
    if (weight_len < param._M_count)
        return Status::OUT_OF_RANGE;
    if (!_S_span_ok (weight_off, param._M_count, model_len))
        return Status::OUT_OF_RANGE;
    for (uint32_t i = 0; i < param._M_count; ++i)
        {
            const int8_t code = param._M_code[i];
            if (code == -128)
                return Status::CORRUPT_STREAM;
            model[weight_off + i] = static_cast<uint8_t> (code);
        }
    /* Tail bytes the layer declares but the tensor does not fill are zeroed,
       so the reader never sees stale flash content inside the span. */
    for (uint32_t i = param._M_count; i < weight_len; ++i)
        model[weight_off + i] = 0u;
    return Status::OK;
}

/**
 * @brief Import a model weight stream into one parameter's code buffer.
 * @return the same rejection vocabulary as copy_out; a -128 byte in the stream
 *         is CORRUPT_STREAM because Core never validates such a code in.
 */
[[nodiscard]] inline Status
copy_in (const uint8_t *const model, const uint32_t model_len,
         const uint32_t weight_off, Trainable &param) noexcept
{
    if (model == nullptr || !status_ok (param.validate ()))
        return Status::BAD_LAYOUT;
    if (!simd_aligned (model))
        return Status::BAD_LAYOUT;
    if (!_S_span_ok (weight_off, param._M_count, model_len))
        return Status::OUT_OF_RANGE;
    for (uint32_t i = 0; i < param._M_count; ++i)
        {
            const int8_t code = static_cast<int8_t> (model[weight_off + i]);
            if (code == -128)
                return Status::CORRUPT_STREAM;
            param._M_code[i] = code;
        }
    return Status::OK;
}

/**@}*/

/* ==================================================================== *
 * Part B -- portable weight container ("MLCP")                          *
 * ==================================================================== *
 *
 * Part A above serialises a *run*: it is bound to a `Trainer`, it carries the
 * progress counters, and it restores in place.  Part B below serialises a
 * *model artifact*: the same code streams plus optional Q15 gradient lanes,
 * optional optimizer lanes and the per-tensor / per-channel Q15 multiplier
 * table, in a self-describing blob that needs no trainer to read it.  The two
 * share nothing but the file, the module's CRC-16 and the little-endian word
 * convention, and they use different magics ("MLKC" run state, "MLCP" weight
 * container) so one reader can never accept the other's bytes.
 *
 * ### Container layout (little-endian throughout, every stream 4-byte aligned)
 *
 * ```text
 *  0 +-----------------------------------------------------------+
 *    | header (56 bytes)                                         |
 *    |   0..3  magic "MLCP"   4 version   5 flags   6..7 params  |
 *    |   8..43 region byte counts, total size, step, loss scale  |
 *    |   44..45 header size   46..47 payload CRC-16              |
 *    |   48..49 header CRC-16  50..51 multiplier count           |
 *    |   52..55 reserved (zero)                                  |
 *    +-----------------------------------------------------------+
 *    | parameter table: params x 48-byte entries                 |
 *    |   shape (dims[4] + rank), flags, count, scale (Q15),      |
 *    |   code / grad / opt / scale offsets, opt lane count       |
 *    +-----------------------------------------------------------+
 *    | payload (offsets below are relative to its start):        |
 *    |   0..3 reserved zero word -- so 0 can mean "absent"       |
 *    |   multiplier table  (int32 Q15, per tensor then channel)  |
 *    |   code streams      (int8, one per tensor, padded to 4)   |
 *    |   gradient lanes    (int16 Q15, where present)            |
 *    |   optimizer lanes   (int16 Q15, where present)            |
 *    +-----------------------------------------------------------+
 * ```
 *
 * Every stream offset is payload-relative, so the payload can be moved (a flash
 * page copy, a second buffer) without rewriting the table.  Each stream is padded
 * up to a 4-byte boundary and the header and table sizes are multiples of 4, so a
 * caller that hands the reader a 4-byte-aligned blob gets a 4-byte-aligned
 * pointer for every stream and can feed it straight into the SIMD kernels.
 *
 * ### Validation order (why a bad blob is always a `Status`, never a fault)
 *
 * length, magic, version, header size, reserved word, header CRC, then structural
 * self-consistency: the region sizes must add up to the declared total, entries
 * must agree with their own shape and with the header's presence flags, and the
 * four streams of every tensor must tile their region in ascending, gap-free
 * order.  Only then is the payload CRC checked.  What survives all of that is
 * *addressable*, which is the only guarantee the accessors need.
 *
 * ### Interop with inference
 *
 * The code streams are plain int8 code runs, i.e. exactly the payload the packlow
 * codec consumes: `wck_code_span` hands the caller a (pointer, length) pair to
 * feed `PacklowEncoder::encode` without a copy.  The multiplier table is the same
 * run of little-endian int32 Q15 values `format.hpp` reads at
 * `Layer::_M_scale_off` through `_S_scale_multiplier`, and
 * `wck_copy_scale_table` produces exactly those bytes for a model buffer, so a
 * checkpoint's calibration travels into a deployed model unchanged.  (Part A's
 * `copy_out` covers the weight-span side of the same story.)
 *
 * Device-path rules honoured here: no heap, no exceptions, no RTTI, no recursion,
 * no writable statics, no float, errors as `Status` values, one pass per region.
 *
 * ### Pool cost (all caller-owned; nothing here allocates)
 *
 * | Object | Bytes |
 * |---|---|
 * | header | 56 |
 * | parameter table entry | 48 |
 * | payload reserved word | 4 |
 * | stream padding | 0..3 per stream, per region |
 * | `WckTensorSource` | 28 (device) |
 * | `WckWriteStats` | 24 |
 * | `WckView` | 68 (device) |
 *
 * A 4-tensor container of 256 lanes each, with gradients, one momentum lane per
 * weight and a per-tensor multiplier, costs 56 + 192 + 4 + 16 + 1024 + 2048 + 2048
 * = 5388 bytes of caller buffer; writing it needs no scratch at all.
 */

/** @brief Weight-container magic, stored as the four bytes 'M','L','C','P'. */
inline constexpr char _S_wck_magic[4] = { 'M', 'L', 'C', 'P' };

/** @brief The only container version this build can both write and read. */
inline constexpr uint8_t _S_wck_version = 1u;

/** @brief Byte size of the fixed header, also written into the header itself. */
inline constexpr uint16_t _S_wck_header_bytes = 56u;

/** @brief Byte size of one parameter table entry. */
inline constexpr uint16_t _S_wck_entry_bytes = 48u;

/** @brief Largest tensor count one container describes. */
inline constexpr uint32_t _S_wck_max_params = 1024u;

/** @brief Byte size of one multiplier table entry (a Q15 int32). */
inline constexpr uint32_t _S_wck_scale_entry_bytes = 4u;

/** @brief Bytes the payload starts with, so a stream offset of 0 means absent. */
inline constexpr uint32_t _S_wck_reserved_bytes = 4u;

/** @brief The stream alignment, i.e. what every region size is a multiple of. */
inline constexpr uint32_t _S_wck_alignment = 4u;

/** @brief Largest payload (streams plus tables) a blob may carry. */
inline constexpr uint32_t _S_wck_max_payload = 8u * 1024u * 1024u;

/** @brief Largest multiplier entry count one blob may carry (header field width). */
inline constexpr uint32_t _S_wck_max_scale_entries = 0xFFFFu;

/** @brief Header `flags` bits: which optional regions exist. */
enum WckFlag : uint8_t
{
    /** @brief At least one tensor carries Q15 gradient lanes. */
    WckHasGrad = 1u << 0,
    /** @brief At least one tensor carries optimizer state lanes. */
    WckHasOpt = 1u << 1,
    /** @brief A Q15 multiplier table follows the parameter table. */
    WckHasScaleTable = 1u << 2,
    /** @brief The run was at its best-known loss when this blob was taken. */
    WckIsBest = 1u << 3,
};

/**
 * @brief The fixed container header, in exactly the byte order it is stored in.
 *
 * The field order *is* the wire order and every field is naturally aligned, so
 * the `static_assert` below doubles as a layout test.  The accessors still read
 * the blob through explicit little-endian loads, so no reader ever depends on
 * that identity -- or on this machine's byte order.
 */
struct WckHeader
{
    uint8_t _M_magic[4]
        = { static_cast<uint8_t> (_S_wck_magic[0]),
            static_cast<uint8_t> (_S_wck_magic[1]),
            static_cast<uint8_t> (_S_wck_magic[2]),
            static_cast<uint8_t> (_S_wck_magic[3]) };
    uint8_t _M_version = _S_wck_version;
    /** @brief `WckFlag` bits. */
    uint8_t _M_flags = 0u;
    uint16_t _M_param_count = 0u;
    uint32_t _M_table_bytes = 0u;
    uint32_t _M_scale_bytes = 0u;
    uint32_t _M_code_bytes = 0u;
    uint32_t _M_grad_bytes = 0u;
    uint32_t _M_opt_bytes = 0u;
    /** @brief Reserved word plus all four regions. */
    uint32_t _M_payload_bytes = 0u;
    /** @brief Exact blob length; a longer buffer is allowed, a shorter is not. */
    uint32_t _M_total_bytes = 0u;
    uint32_t _M_step = 0u;
    /** @brief Loss scale in force, so a resume restarts the chain at the same gain. */
    int32_t _M_loss_scale_q15 = _S_q15_one;
    uint16_t _M_header_size = _S_wck_header_bytes;
    uint16_t _M_payload_crc = 0u;
    uint16_t _M_header_crc = 0u;
    /** @brief Total multiplier entries, so a table can be sized without a scan. */
    uint16_t _M_scale_entry_count = 0u;
    uint32_t _M_reserved = 0u;
};

static_assert (sizeof (WckHeader) == 56u,
               "the weight container header is the documented 56-byte layout");

/**
 * @brief One container parameter table entry: shape, flags, count, scale, offsets.
 *
 * Offsets are payload-relative; zero means "this tensor has no such stream",
 * which is unambiguous because the payload's first four bytes are a reserved
 * word.  A tensor's multiplier count is its own `_M_scale_entries`, so a
 * per-channel container and a per-tensor one differ only by that field.
 */
struct WckEntry
{
    uint16_t _M_dims[_S_max_rank] = { 1u, 1u, 1u, 1u };
    uint8_t _M_rank = 0u;
    uint8_t _M_pad = 0u;
    /** @brief The `Flag` bits of the tensor this entry describes. */
    uint16_t _M_flags = 0u;
    uint32_t _M_count = 0u;
    /** @brief Copied verbatim from `Trainable::_M_scale_q15`. */
    int32_t _M_scale_q15 = _S_q15_one;
    uint32_t _M_code_off = 0u;
    uint32_t _M_grad_off = 0u;
    uint32_t _M_opt_off = 0u;
    uint32_t _M_scale_off = 0u;
    uint16_t _M_scale_entries = 0u;
    uint16_t _M_pad2 = 0u;
    /** @brief Q15 lanes in the optimizer region: 0, count, or 2 · count. */
    uint32_t _M_opt_lanes = 0u;
    uint32_t _M_reserved = 0u;
};

static_assert (sizeof (WckEntry) == 48u,
               "one container parameter entry is 48 bytes on the wire");

/** @brief Marker for a multiplier table that is per-tensor rather than per-axis. */
inline constexpr uint32_t _S_wck_per_tensor = 0xFFFFFFFFu;

constexpr void
wck_put_u16 (uint8_t *const at, const uint16_t value) noexcept
{
    at[0] = static_cast<uint8_t> (value & 0xFFu);
    at[1] = static_cast<uint8_t> ((value >> 8) & 0xFFu);
}

constexpr void
wck_put_u32 (uint8_t *const at, const uint32_t value) noexcept
{
    at[0] = static_cast<uint8_t> (value & 0xFFu);
    at[1] = static_cast<uint8_t> ((value >> 8) & 0xFFu);
    at[2] = static_cast<uint8_t> ((value >> 16) & 0xFFu);
    at[3] = static_cast<uint8_t> ((value >> 24) & 0xFFu);
}

constexpr void
wck_put_i32 (uint8_t *const at, const int32_t value) noexcept
{
    wck_put_u32 (at, static_cast<uint32_t> (value));
}

/** @brief Round a byte count up to the stream alignment. */
[[nodiscard]] inline constexpr uint32_t
wck_align_span (const uint32_t bytes) noexcept
{
    return align_up (bytes, _S_wck_alignment);
}

/** @brief True when a payload-relative offset starts a SIMD-able stream. */
[[nodiscard]] inline constexpr bool
wck_aligned4 (const uint32_t offset) noexcept
{
    return (offset & (_S_wck_alignment - 1u)) == 0u;
}

/**
 * @brief Copy one aligned region, four bytes at a time, then the tail.
 *
 * Both sides are 4-byte aligned by construction (the caller's streams are, and so
 * are the payload offsets), so the bulk of every copy is one word per load with
 * no shifting -- which is what keeps a 5 KB container write off a byte loop.
 * Neither bound is ever exceeded: the word loop stops four bytes short and the
 * byte loop carries the remainder.
 */
constexpr void
wck_copy_region (uint8_t *const destination, const uint8_t *const source,
                 const uint32_t bytes) noexcept
{
    uint32_t index = 0u;
    for (; index + 4u <= bytes; index += 4u)
        {
            const uint32_t value
                = static_cast<uint32_t> (source[index])
                  | (static_cast<uint32_t> (source[index + 1u]) << 8u)
                  | (static_cast<uint32_t> (source[index + 2u]) << 16u)
                  | (static_cast<uint32_t> (source[index + 3u]) << 24u);
            wck_put_u32 (destination + index, value);
        }
    for (; index < bytes; ++index)
        destination[index] = source[index];
}

/** @brief Set a region to zero: padding has to be deterministic to be CRCd. */
constexpr void
wck_zero_region (uint8_t *const destination, const uint32_t bytes) noexcept
{
    uint32_t index = 0u;
    for (; index + 4u <= bytes; index += 4u)
        wck_put_u32 (destination + index, 0u);
    for (; index < bytes; ++index)
        destination[index] = 0u;
}

/**
 * @brief One tensor's container contribution, expressed as a `Trainable` view.
 *
 * The view is the source of truth for shape, codes, gradient lanes, scale and
 * flags, so a caller hands over the very tensors it trained: there is no second
 * description to keep in sync.  Only the two genuinely optional pieces sit beside
 * it -- optimizer lanes (a Q15 int16 run @c opt_lanes long, e.g. momentum, or m
 * and v interleaved) and the multiplier table inference will read back.
 */
struct WckTensorSource
{
    const Trainable *_M_param = nullptr;
    /** @brief Q15 optimizer lanes, or null when this tensor has none. */
    const int16_t *_M_opt = nullptr;
    /** @brief Q15 multipliers, per tensor or one per channel of @c scale_axis. */
    const int32_t *_M_scale_table = nullptr;
    uint32_t _M_opt_lanes = 0u;
    uint32_t _M_scale_entries = 0u;
    /** @brief Axis the table walks, or `_S_wck_per_tensor`. */
    uint32_t _M_scale_axis = _S_wck_per_tensor;
    uint8_t _M_pad[4] = { 0u, 0u, 0u, 0u };
};

static_assert (sizeof (WckTensorSource) == (sizeof (void *) == 4u ? 28u : 40u),
               "a container source is three pointers, three counters, a pad");

/** @brief Byte counters of one successful write; also the padding audit. */
struct WckWriteStats
{
    uint32_t _M_bytes_written = 0u;
    uint32_t _M_code_bytes = 0u;
    uint32_t _M_grad_bytes = 0u;
    uint32_t _M_opt_bytes = 0u;
    uint32_t _M_scale_bytes = 0u;
    /** @brief Reserved word plus every alignment pad: bytes that carry no data. */
    uint32_t _M_padding_bytes = 0u;
};

static_assert (sizeof (WckWriteStats) == 24u,
               "write statistics are six counters");

/**
 * @brief Reject a source array that cannot be written, before touching a buffer.
 *
 * Every rule here is a rule the reader re-checks, so a blob that was written
 * successfully can never fail validation for a reason the writer could have
 * avoided: a null view, an unaligned stream, a count that disagrees with its
 * shape, optimizer lanes without a count (or a count without lanes), a multiplier
 * table whose entry count disagrees with the axis it walks, or a tensor count past
 * the container limits.
 */
[[nodiscard]] inline Status
wck_validate_sources (const WckTensorSource *const sources,
                      const uint32_t count) noexcept
{
    if (sources == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (sources))
        return Status::BAD_LAYOUT;
    if (count == 0u || count > _S_wck_max_params)
        return Status::OUT_OF_RANGE;
    uint32_t scale_entries = 0u;
    for (uint32_t index = 0u; index < count; ++index)
        {
            const WckTensorSource &source = sources[index];
            if (source._M_param == nullptr)
                return Status::BAD_LAYOUT;
            const Status status = source._M_param->validate ();
            if (!status_ok (status))
                return status;
            if (!source._M_param->simd_ready ())
                return Status::BAD_LAYOUT;
            if (source._M_param->_M_flags > 0xFFFFu)
                return Status::OUT_OF_RANGE;
            if (source._M_opt != nullptr)
                {
                    if (source._M_opt_lanes == 0u)
                        return Status::BAD_LAYOUT;
                    if (source._M_opt_lanes > 4u * _S_max_elements)
                        return Status::OUT_OF_RANGE;
                    if (!simd_aligned (source._M_opt))
                        return Status::BAD_LAYOUT;
                }
            else if (source._M_opt_lanes != 0u)
                return Status::BAD_LAYOUT;
            if (source._M_scale_table != nullptr)
                {
                    if (source._M_scale_entries == 0u
                        || source._M_scale_entries > _S_wck_max_scale_entries)
                        return Status::OUT_OF_RANGE;
                    if (source._M_scale_axis == _S_wck_per_tensor)
                        {
                            if (source._M_scale_entries != 1u)
                                return Status::BAD_LAYOUT;
                        }
                    else
                        {
                            if (source._M_scale_axis >= _S_max_rank)
                                return Status::BAD_LAYOUT;
                            if (source._M_scale_entries
                                != source._M_param->_M_shape
                                       ._M_dims[source._M_scale_axis])
                                return Status::BAD_LAYOUT;
                        }
                    if (!simd_aligned (source._M_scale_table))
                        return Status::BAD_LAYOUT;
                    if (scale_entries > _S_wck_max_scale_entries
                        - source._M_scale_entries)
                        return Status::OUT_OF_RANGE;
                    scale_entries += source._M_scale_entries;
                }
            else if (source._M_scale_entries != 0u)
                return Status::BAD_LAYOUT;
        }
    return Status::OK;
}

/**
 * @brief Byte size the container of @p sources needs, before any buffer exists.
 *
 * The caller sizes its flash page or arena with this and nothing else.  A
 * `Status` of OUT_OF_RANGE means no buffer could hold the request, because the
 * payload would exceed `_S_wck_max_payload`.
 */
[[nodiscard]] inline Status
wck_bytes_needed (const WckTensorSource *const sources, const uint32_t count,
                  uint32_t &bytes_out) noexcept
{
    bytes_out = 0u;
    const Status status = wck_validate_sources (sources, count);
    if (!status_ok (status))
        return status;
    uint64_t code = 0u;
    uint64_t grad = 0u;
    uint64_t opt = 0u;
    uint64_t scale = 0u;
    for (uint32_t index = 0u; index < count; ++index)
        {
            const WckTensorSource &source = sources[index];
            const Trainable &param = *source._M_param;
            code += wck_align_span (param._M_count);
            if (param._M_grad != nullptr)
                grad += wck_align_span (param._M_count * 2u);
            if (source._M_opt != nullptr)
                opt += wck_align_span (source._M_opt_lanes * 2u);
            if (source._M_scale_table != nullptr)
                scale += source._M_scale_entries * _S_wck_scale_entry_bytes;
        }
    const uint64_t payload
        = _S_wck_reserved_bytes + scale + code + grad + opt;
    const uint64_t total
        = _S_wck_header_bytes
          + static_cast<uint64_t> (count) * _S_wck_entry_bytes + payload;
    if (payload > _S_wck_max_payload || total > 0xFFFFFFFFull)
        return Status::OUT_OF_RANGE;
    bytes_out = static_cast<uint32_t> (total);
    return Status::OK;
}

/**
 * @brief Serialize a weight container into a caller-owned buffer.
 *
 * @p buffer must be 4-byte aligned and at least `wck_bytes_needed` bytes wide; a
 * smaller capacity is TRUNCATED rather than a partial write, so a blob is either
 * complete and CRC-valid or absent.  Everything outside the data regions is
 * zeroed first, which is what makes the output deterministic: the same tensors
 * always produce byte-identical blobs, so a read-then-write round trip can be
 * proven by comparing bytes.
 */
[[nodiscard]] inline Status
wck_write (uint8_t *const buffer, const uint32_t capacity,
           const WckTensorSource *const sources, const uint32_t count,
           const uint32_t step, const int32_t loss_scale_q15,
           WckWriteStats &stats, const bool is_best = false) noexcept
{
    stats = WckWriteStats{};
    if (buffer == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (buffer))
        return Status::BAD_LAYOUT;
    uint32_t needed = 0u;
    const Status size_status = wck_bytes_needed (sources, count, needed);
    if (!status_ok (size_status))
        return size_status;
    if (capacity < needed)
        return Status::TRUNCATED;

    bool any_grad = false;
    bool any_opt = false;
    bool any_scale = false;
    const uint32_t table_bytes = count * _S_wck_entry_bytes;
    uint32_t scale_bytes = 0u;
    uint32_t code_bytes = 0u;
    uint32_t grad_bytes = 0u;
    uint32_t opt_bytes = 0u;
    uint32_t content = 0u;
    uint32_t scale_entries = 0u;
    for (uint32_t index = 0u; index < count; ++index)
        {
            const WckTensorSource &source = sources[index];
            const Trainable &param = *source._M_param;
            code_bytes += wck_align_span (param._M_count);
            content += param._M_count;
            if (param._M_grad != nullptr)
                {
                    any_grad = true;
                    grad_bytes += wck_align_span (param._M_count * 2u);
                    content += param._M_count * 2u;
                }
            if (source._M_opt != nullptr)
                {
                    any_opt = true;
                    opt_bytes += wck_align_span (source._M_opt_lanes * 2u);
                    content += source._M_opt_lanes * 2u;
                }
            if (source._M_scale_table != nullptr)
                {
                    any_scale = true;
                    scale_bytes += source._M_scale_entries
                                  * _S_wck_scale_entry_bytes;
                    content += source._M_scale_entries
                               * _S_wck_scale_entry_bytes;
                    scale_entries += source._M_scale_entries;
                }
        }
    const uint32_t payload_bytes
        = _S_wck_reserved_bytes + scale_bytes + code_bytes + grad_bytes
          + opt_bytes;
    const uint32_t total_bytes
        = _S_wck_header_bytes + table_bytes + payload_bytes;

    /* Zero once, up front: every byte that is not stream data must be
       deterministic before either CRC is computed. */
    wck_zero_region (buffer, total_bytes);

    uint8_t *const table = buffer + _S_wck_header_bytes;
    const uint32_t payload_off = _S_wck_header_bytes + table_bytes;
    uint32_t scale_run = _S_wck_reserved_bytes;
    uint32_t code_run = scale_run + scale_bytes;
    uint32_t grad_run = code_run + code_bytes;
    uint32_t opt_run = grad_run + grad_bytes;

    for (uint32_t index = 0u; index < count; ++index)
        {
            const WckTensorSource &source = sources[index];
            const Trainable &param = *source._M_param;
            WckEntry entry;
            for (uint8_t axis = 0u; axis < _S_max_rank; ++axis)
                entry._M_dims[axis] = param._M_shape._M_dims[axis];
            entry._M_rank = param._M_shape._M_rank;
            entry._M_flags = static_cast<uint16_t> (param._M_flags);
            entry._M_count = param._M_count;
            entry._M_scale_q15 = param._M_scale_q15;
            entry._M_code_off = code_run;
            code_run += wck_align_span (param._M_count);
            if (any_grad && param._M_grad != nullptr)
                {
                    entry._M_grad_off = grad_run;
                    grad_run += wck_align_span (param._M_count * 2u);
                }
            if (source._M_opt != nullptr)
                {
                    entry._M_opt_off = opt_run;
                    entry._M_opt_lanes = source._M_opt_lanes;
                    opt_run += wck_align_span (source._M_opt_lanes * 2u);
                }
            if (source._M_scale_table != nullptr)
                {
                    entry._M_scale_off = scale_run;
                    entry._M_scale_entries
                        = static_cast<uint16_t> (source._M_scale_entries);
                    scale_run
                        += source._M_scale_entries * _S_wck_scale_entry_bytes;
                }

            uint8_t *const at = table + index * _S_wck_entry_bytes;
            for (uint8_t axis = 0u; axis < _S_max_rank; ++axis)
                wck_put_u16 (at + axis * 2u, entry._M_dims[axis]);
            at[8] = entry._M_rank;
            wck_put_u16 (at + 10, entry._M_flags);
            wck_put_u32 (at + 12, entry._M_count);
            wck_put_i32 (at + 16, entry._M_scale_q15);
            wck_put_u32 (at + 20, entry._M_code_off);
            wck_put_u32 (at + 24, entry._M_grad_off);
            wck_put_u32 (at + 28, entry._M_opt_off);
            wck_put_u32 (at + 32, entry._M_scale_off);
            wck_put_u16 (at + 36, entry._M_scale_entries);
            wck_put_u32 (at + 40, entry._M_opt_lanes);

            wck_copy_region (buffer + payload_off + entry._M_code_off,
                             reinterpret_cast<const uint8_t *> (param._M_code),
                             param._M_count);
            if (entry._M_grad_off != 0u)
                wck_copy_region (
                    buffer + payload_off + entry._M_grad_off,
                    reinterpret_cast<const uint8_t *> (param._M_grad),
                    param._M_count * 2u);
            if (entry._M_opt_off != 0u)
                wck_copy_region (
                    buffer + payload_off + entry._M_opt_off,
                    reinterpret_cast<const uint8_t *> (source._M_opt),
                    source._M_opt_lanes * 2u);
            if (entry._M_scale_off != 0u)
                for (uint32_t lane = 0u; lane < source._M_scale_entries; ++lane)
                    wck_put_i32 (buffer + payload_off + entry._M_scale_off
                                     + lane * _S_wck_scale_entry_bytes,
                                 source._M_scale_table[lane]);
        }

    uint8_t flags = 0u;
    if (any_grad)
        flags |= static_cast<uint8_t> (WckFlag::WckHasGrad);
    if (any_opt)
        flags |= static_cast<uint8_t> (WckFlag::WckHasOpt);
    if (any_scale)
        flags |= static_cast<uint8_t> (WckFlag::WckHasScaleTable);
    if (is_best)
        flags |= static_cast<uint8_t> (WckFlag::WckIsBest);

    buffer[0] = static_cast<uint8_t> (_S_wck_magic[0]);
    buffer[1] = static_cast<uint8_t> (_S_wck_magic[1]);
    buffer[2] = static_cast<uint8_t> (_S_wck_magic[2]);
    buffer[3] = static_cast<uint8_t> (_S_wck_magic[3]);
    buffer[4] = _S_wck_version;
    buffer[5] = flags;
    wck_put_u16 (buffer + 6, static_cast<uint16_t> (count));
    wck_put_u32 (buffer + 8, table_bytes);
    wck_put_u32 (buffer + 12, scale_bytes);
    wck_put_u32 (buffer + 16, code_bytes);
    wck_put_u32 (buffer + 20, grad_bytes);
    wck_put_u32 (buffer + 24, opt_bytes);
    wck_put_u32 (buffer + 28, payload_bytes);
    wck_put_u32 (buffer + 32, total_bytes);
    wck_put_u32 (buffer + 36, step);
    wck_put_i32 (buffer + 40, loss_scale_q15);
    wck_put_u16 (buffer + 44, _S_wck_header_bytes);
    wck_put_u16 (buffer + 46,
                 _S_crc16 (buffer + payload_off, payload_bytes));
    wck_put_u16 (buffer + 50, static_cast<uint16_t> (scale_entries));

    stats._M_bytes_written = total_bytes;
    stats._M_code_bytes = code_bytes;
    stats._M_grad_bytes = grad_bytes;
    stats._M_opt_bytes = opt_bytes;
    stats._M_scale_bytes = scale_bytes;
    stats._M_padding_bytes = payload_bytes - content;

    /* The header CRC is the last store: it covers bytes [0, 48), which includes
       the payload CRC, so a tampered payload invalidates the container twice over
       and a tampered size field is caught before any offset is trusted. */
    wck_put_u16 (buffer + 48, _S_crc16 (buffer, 48));
    return Status::OK;
}

/**
 * @brief A validated window over a weight-container blob.
 *
 * The view borrows the buffer -- the caller keeps it alive and readable as long as
 * the view is used, exactly as with every other mllee view.  Building it *is* the
 * validation pass: after `wck_read` returns OK, every offset the accessors
 * dereference has been proven to lie inside the blob and to agree with the header
 * that framed it.
 */
struct WckView
{
    const uint8_t *_M_data = nullptr;
    uint32_t _M_length = 0u;
    uint32_t _M_payload_off = 0u;
    WckHeader _M_header{};
    uint8_t _M_valid = 0u;
    uint8_t _M_pad[3] = { 0u, 0u, 0u };
};

static_assert (sizeof (WckView)
                   == sizeof (WckHeader) + (sizeof (void *) == 4u ? 16u : 24u),
               "a container view is the decoded header plus two offsets");

/** @brief True when @p flag is one of the header's presence bits. */
[[nodiscard]] inline constexpr bool
wck_has (const WckHeader &header, const WckFlag flag) noexcept
{
    return (header._M_flags & static_cast<uint8_t> (flag)) != 0u;
}

/** @brief Decode one 48-byte container entry from a table that is already trusted. */
constexpr void
wck_entry_at (const uint8_t *const at, WckEntry &out) noexcept
{
    for (uint8_t axis = 0u; axis < _S_max_rank; ++axis)
        out._M_dims[axis] = _S_u16 (at + axis * 2u);
    out._M_rank = at[8];
    out._M_pad = at[9];
    out._M_flags = _S_u16 (at + 10);
    out._M_count = _S_u32 (at + 12);
    out._M_scale_q15 = static_cast<int32_t> (_S_u32 (at + 16));
    out._M_code_off = _S_u32 (at + 20);
    out._M_grad_off = _S_u32 (at + 24);
    out._M_opt_off = _S_u32 (at + 28);
    out._M_scale_off = _S_u32 (at + 32);
    out._M_scale_entries = _S_u16 (at + 36);
    out._M_pad2 = _S_u16 (at + 38);
    out._M_opt_lanes = _S_u32 (at + 40);
    out._M_reserved = _S_u32 (at + 44);
}

/** @brief Load one container table entry out of a validated view. */
[[nodiscard]] inline Status
wck_entry (const WckView &view, const uint32_t index, WckEntry &out) noexcept
{
    out = WckEntry{};
    if (view._M_valid == 0u || view._M_data == nullptr)
        return Status::BAD_LAYOUT;
    if (index >= view._M_header._M_param_count)
        return Status::OUT_OF_RANGE;
    wck_entry_at (view._M_data + _S_wck_header_bytes
                      + index * _S_wck_entry_bytes,
                  out);
    return Status::OK;
}

/**
 * @brief Parse, structurally validate and CRC-check a weight container.
 *
 * @p length may exceed the blob's own `total_bytes` (a flash page padded to a
 * write block): the reader checks that the declared size fits the buffer, then
 * validates against the declared size, never against the page size.
 */
[[nodiscard]] inline Status
wck_read (const uint8_t *const data, const uint32_t length,
          WckView &view) noexcept
{
    view = WckView{};
    if (data == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (data))
        return Status::BAD_LAYOUT;
    if (length < _S_wck_header_bytes)
        return Status::TRUNCATED;
    for (uint8_t index = 0u; index < 4u; ++index)
        if (data[index] != static_cast<uint8_t> (_S_wck_magic[index]))
            return Status::BAD_MAGIC;
    if (data[4] != _S_wck_version)
        return Status::BAD_VERSION;
    if (_S_u16 (data + 44) != _S_wck_header_bytes)
        return Status::BAD_LAYOUT;
    if (_S_u32 (data + 52) != 0u)
        return Status::BAD_LAYOUT;
    if (_S_u16 (data + 48) != _S_crc16 (data, 48))
        return Status::CORRUPT_STREAM;

    WckHeader &header = view._M_header;
    header._M_version = data[4];
    header._M_flags = data[5];
    header._M_param_count = _S_u16 (data + 6);
    header._M_table_bytes = _S_u32 (data + 8);
    header._M_scale_bytes = _S_u32 (data + 12);
    header._M_code_bytes = _S_u32 (data + 16);
    header._M_grad_bytes = _S_u32 (data + 20);
    header._M_opt_bytes = _S_u32 (data + 24);
    header._M_payload_bytes = _S_u32 (data + 28);
    header._M_total_bytes = _S_u32 (data + 32);
    header._M_step = _S_u32 (data + 36);
    header._M_loss_scale_q15 = static_cast<int32_t> (_S_u32 (data + 40));
    header._M_header_size = _S_u16 (data + 44);
    header._M_payload_crc = _S_u16 (data + 46);
    header._M_header_crc = _S_u16 (data + 48);
    header._M_scale_entry_count = _S_u16 (data + 50);
    header._M_reserved = _S_u32 (data + 52);

    if (header._M_param_count == 0u || header._M_param_count > _S_wck_max_params)
        return Status::BAD_LAYOUT;
    if (header._M_table_bytes
        != static_cast<uint32_t> (header._M_param_count) * _S_wck_entry_bytes)
        return Status::BAD_LAYOUT;
    if ((header._M_scale_bytes != 0u)
        != wck_has (header, WckFlag::WckHasScaleTable))
        return Status::BAD_LAYOUT;
    if ((header._M_grad_bytes != 0u) != wck_has (header, WckFlag::WckHasGrad))
        return Status::BAD_LAYOUT;
    if ((header._M_opt_bytes != 0u) != wck_has (header, WckFlag::WckHasOpt))
        return Status::BAD_LAYOUT;
    if (header._M_payload_bytes
        != _S_wck_reserved_bytes + header._M_scale_bytes + header._M_code_bytes
               + header._M_grad_bytes + header._M_opt_bytes)
        return Status::BAD_LAYOUT;
    const uint32_t payload_off = _S_wck_header_bytes + header._M_table_bytes;
    if (header._M_total_bytes != payload_off + header._M_payload_bytes)
        return Status::BAD_LAYOUT;
    if (header._M_total_bytes > length)
        return Status::TRUNCATED;
    if (_S_u32 (data + payload_off) != 0u)
        return Status::BAD_LAYOUT;
    if (header._M_payload_crc
        != _S_crc16 (data + payload_off, header._M_payload_bytes))
        return Status::CORRUPT_STREAM;

    /* Structure last: the four regions must be tiled by the entries themselves,
       in payload order with no gap and no overlap. */
    const uint8_t *const table = data + _S_wck_header_bytes;
    uint32_t scale_run = _S_wck_reserved_bytes;
    uint32_t code_run = _S_wck_reserved_bytes + header._M_scale_bytes;
    uint32_t grad_run = code_run + header._M_code_bytes;
    uint32_t opt_run = grad_run + header._M_grad_bytes;
    const uint32_t scale_end = code_run;
    const uint32_t code_end = grad_run;
    const uint32_t grad_end = opt_run;
    const uint32_t opt_end = opt_run + header._M_opt_bytes;
    uint32_t scale_entry_total = 0u;
    bool saw_grad = false;
    bool saw_opt = false;
    bool saw_scale = false;
    for (uint32_t index = 0u; index < header._M_param_count; ++index)
        {
            WckEntry entry;
            wck_entry_at (table + index * _S_wck_entry_bytes, entry);
            Shape shape;
            for (uint8_t axis = 0u; axis < _S_max_rank; ++axis)
                shape._M_dims[axis] = entry._M_dims[axis];
            shape._M_rank = entry._M_rank;
            if (!status_ok (shape.validate ()))
                return Status::BAD_LAYOUT;
            if (entry._M_count == 0u || entry._M_count > _S_max_elements
                || shape.elements () != entry._M_count)
                return Status::BAD_LAYOUT;
            if (entry._M_pad != 0u || entry._M_pad2 != 0u
                || entry._M_reserved != 0u)
                return Status::BAD_LAYOUT;
            if (entry._M_code_off != code_run || !wck_aligned4 (code_run))
                return Status::BAD_LAYOUT;
            code_run += wck_align_span (entry._M_count);
            if (entry._M_grad_off != 0u)
                {
                    if (entry._M_grad_off != grad_run
                        || !wck_aligned4 (grad_run))
                        return Status::BAD_LAYOUT;
                    grad_run += wck_align_span (entry._M_count * 2u);
                    saw_grad = true;
                }
            if (entry._M_opt_off != 0u)
                {
                    if (entry._M_opt_off != opt_run || !wck_aligned4 (opt_run)
                        || entry._M_opt_lanes == 0u
                        || entry._M_opt_lanes > 4u * _S_max_elements)
                        return Status::BAD_LAYOUT;
                    opt_run += wck_align_span (entry._M_opt_lanes * 2u);
                    saw_opt = true;
                }
            else if (entry._M_opt_lanes != 0u)
                return Status::BAD_LAYOUT;
            if (entry._M_scale_entries != 0u)
                {
                    if (entry._M_scale_off != scale_run
                        || !wck_aligned4 (scale_run))
                        return Status::BAD_LAYOUT;
                    if (entry._M_scale_entries > _S_max_elements
                        || scale_entry_total
                               > _S_wck_max_scale_entries - entry._M_scale_entries)
                        return Status::BAD_LAYOUT;
                    scale_run += entry._M_scale_entries
                                 * _S_wck_scale_entry_bytes;
                    scale_entry_total += entry._M_scale_entries;
                    saw_scale = true;
                }
            else if (entry._M_scale_off != 0u)
                return Status::BAD_LAYOUT;
        }
    if (scale_run != scale_end || code_run != code_end || grad_run != grad_end
        || opt_run != opt_end)
        return Status::BAD_LAYOUT;
    if (saw_scale != wck_has (header, WckFlag::WckHasScaleTable)
        || saw_grad != wck_has (header, WckFlag::WckHasGrad)
        || saw_opt != wck_has (header, WckFlag::WckHasOpt))
        return Status::BAD_LAYOUT;
    if (scale_entry_total != header._M_scale_entry_count)
        return Status::BAD_LAYOUT;

    view._M_data = data;
    view._M_length = length;
    view._M_payload_off = payload_off;
    view._M_valid = 1u;
    return Status::OK;
}

/** @brief The decoded header of a validated container view. */
[[nodiscard]] inline constexpr const WckHeader &
wck_header (const WckView &view) noexcept
{
    return view._M_header;
}

/** @brief The code stream of one tensor, as the packlow codec wants it. */
[[nodiscard]] inline Status
wck_code_span (const WckView &view, const uint32_t index,
               const uint8_t *&codes_out, uint32_t &length_out) noexcept
{
    codes_out = nullptr;
    length_out = 0u;
    WckEntry entry;
    const Status status = wck_entry (view, index, entry);
    if (!status_ok (status))
        return status;
    codes_out = view._M_data + view._M_payload_off + entry._M_code_off;
    length_out = entry._M_count;
    return Status::OK;
}

/** @brief The Q15 gradient lanes of one tensor, when the blob carries them. */
[[nodiscard]] inline Status
wck_grad_span (const WckView &view, const uint32_t index,
               const int16_t *&grads_out, uint32_t &lanes_out) noexcept
{
    grads_out = nullptr;
    lanes_out = 0u;
    WckEntry entry;
    const Status status = wck_entry (view, index, entry);
    if (!status_ok (status))
        return status;
    if (entry._M_grad_off == 0u)
        return Status::UNSUPPORTED;
    grads_out = reinterpret_cast<const int16_t *> (
        view._M_data + view._M_payload_off + entry._M_grad_off);
    lanes_out = entry._M_count;
    return Status::OK;
}

/** @brief The Q15 optimizer lanes of one tensor, when the blob carries them. */
[[nodiscard]] inline Status
wck_opt_span (const WckView &view, const uint32_t index,
              const int16_t *&opt_out, uint32_t &lanes_out) noexcept
{
    opt_out = nullptr;
    lanes_out = 0u;
    WckEntry entry;
    const Status status = wck_entry (view, index, entry);
    if (!status_ok (status))
        return status;
    if (entry._M_opt_off == 0u)
        return Status::UNSUPPORTED;
    opt_out = reinterpret_cast<const int16_t *> (
        view._M_data + view._M_payload_off + entry._M_opt_off);
    lanes_out = entry._M_opt_lanes;
    return Status::OK;
}

/** @brief One multiplier of a tensor's Q15 table, channel-indexed. */
[[nodiscard]] inline Status
wck_scale_at (const WckView &view, const uint32_t index, const uint32_t channel,
              int32_t &value_out) noexcept
{
    value_out = 0;
    WckEntry entry;
    const Status status = wck_entry (view, index, entry);
    if (!status_ok (status))
        return status;
    if (entry._M_scale_off == 0u || channel >= entry._M_scale_entries)
        return Status::OUT_OF_RANGE;
    value_out = static_cast<int32_t> (
        _S_u32 (view._M_data + view._M_payload_off + entry._M_scale_off
                + channel * _S_wck_scale_entry_bytes));
    return Status::OK;
}

/**
 * @brief Restore one tensor from the container into a live `Trainable`.
 *
 * The destination must already describe the same shape: the reader refuses a
 * mismatch instead of truncating or overrunning, which is what lets a caller
 * restore into a pool it re-carved from its own model description.  Codes, and
 * gradient lanes where the blob has them, are copied; scale and flags are
 * restored from the entry.
 */
[[nodiscard]] inline Status
wck_restore (const WckView &view, const uint32_t index,
             Trainable &destination) noexcept
{
    const Status destination_status = destination.validate ();
    if (!status_ok (destination_status))
        return destination_status;
    WckEntry entry;
    const Status status = wck_entry (view, index, entry);
    if (!status_ok (status))
        return status;
    Shape shape;
    for (uint8_t axis = 0u; axis < _S_max_rank; ++axis)
        shape._M_dims[axis] = entry._M_dims[axis];
    shape._M_rank = entry._M_rank;
    if (!(shape == destination._M_shape))
        return Status::BAD_LAYOUT;
    wck_copy_region (reinterpret_cast<uint8_t *> (destination._M_code),
                     view._M_data + view._M_payload_off + entry._M_code_off,
                     entry._M_count);
    if (entry._M_grad_off != 0u)
        {
            if (destination._M_grad == nullptr)
                return Status::BAD_LAYOUT;
            wck_copy_region (
                reinterpret_cast<uint8_t *> (destination._M_grad),
                view._M_data + view._M_payload_off + entry._M_grad_off,
                entry._M_count * 2u);
        }
    destination._M_scale_q15 = entry._M_scale_q15;
    destination._M_flags = entry._M_flags;
    return Status::OK;
}

/**
 * @brief Restore a whole parameter set from a container, in table order.
 *
 * On the first tensor whose shape disagrees the restore stops and reports which
 * index failed through @p failed_out: resuming against a changed architecture is
 * then a diagnosable event rather than a silently corrupt model.
 */
[[nodiscard]] inline Status
wck_restore_set (const WckView &view, Trainable *const parameters,
                 const uint32_t count, uint32_t &restored_out,
                 uint32_t &failed_out) noexcept
{
    restored_out = 0u;
    failed_out = 0u;
    if (view._M_valid == 0u || parameters == nullptr)
        return Status::BAD_LAYOUT;
    if (count != view._M_header._M_param_count)
        return Status::OUT_OF_RANGE;
    for (uint32_t index = 0u; index < count; ++index)
        {
            const Status status = wck_restore (view, index, parameters[index]);
            if (!status_ok (status))
                {
                    failed_out = index;
                    return status;
                }
            ++restored_out;
        }
    return Status::OK;
}

/**
 * @brief Copy one tensor's multiplier table into a model buffer.
 *
 * `Layer::_M_scale_off` addresses the *model* buffer, so a deployed model needs
 * the table copied rather than aliased; this produces exactly the bytes
 * `_S_scale_multiplier` reads and returns how many it wrote, which is what the
 * caller stores in the descriptor.  @p destination must be 4-byte aligned.
 */
[[nodiscard]] inline Status
wck_copy_scale_table (const WckView &view, const uint32_t tensor,
                      uint8_t *const destination, const uint32_t capacity,
                      uint32_t &bytes_out) noexcept
{
    bytes_out = 0u;
    if (destination == nullptr)
        return Status::BAD_LAYOUT;
    if (!simd_aligned (destination))
        return Status::BAD_LAYOUT;
    WckEntry entry;
    const Status status = wck_entry (view, tensor, entry);
    if (!status_ok (status))
        return status;
    if (entry._M_scale_off == 0u)
        return Status::UNSUPPORTED;
    const uint32_t span = entry._M_scale_entries * _S_wck_scale_entry_bytes;
    if (capacity < span)
        return Status::TRUNCATED;
    wck_copy_region (destination,
                     view._M_data + view._M_payload_off + entry._M_scale_off,
                     span);
    bytes_out = span;
    return Status::OK;
}

/**@}*/
} // namespace zzs::mllee::train
