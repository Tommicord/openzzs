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
 * @brief Numerically safe Q15 softmax, log-softmax, masks and temperature.
 *
 * @section softmax_domain Which domain a logit lives in
 *
 * A logit is a `Trainable` code, and a code alone says nothing about magnitude:
 * the module widens it with the tensor's own scale, exactly the way the
 * inference core reads a code as `c * scale / (127 * 32768)` in real units.  In
 * the Q15 lane domain that is
 *
 *   x = round (code * _M_scale_q15 / 127)
 *
 * so the default scale (32768) reproduces `code_to_q15` -- one code is 1/127 and
 * the row spans +/-1.0 -- while a classifier that needs +/-8 logits sets
 * `_M_scale_q15 = 8 * 32768` and gets the same codes with a usable dynamic
 * range.  Every widened logit is then clamped into the exp window
 * (`_S_exp_clamp_q15`, +/-8.0), because `q15_exp_neg` cannot represent more and
 * a lane beyond it is a saturated lane, not a trap.
 *
 * @section softmax_stability Why the row can never divide by zero
 *
 * The row kernel subtracts the kept maximum before calling exp, so the largest
 * term is `q15_exp_neg (0) = _S_q15_max = 32767` and the row sum is therefore at
 * least 32767 -- a division by the sum is a division by a positive integer by
 * construction, and no float appears anywhere.  A term whose gap from the
 * maximum exceeds the exp window is emitted as an exact zero rather than the
 * exp table's stall value, because the table floors at 4 (`train_fixed.hpp`'s
 * own note) and a long row of 4s would add mass the real distribution does not
 * have.  Lanes are summed in int64 and narrowed through `clamp_sum_q15`, whose
 * rail the op counts as a saturation event.
 *
 * The probabilities are Q15 (`ratio_q15 (exp_term, sum)`, i.e. one unit is
 * 1/32768) and their rounding residual -- always below one LSB per lane -- is
 * absorbed by the row's argmax lane, so the emitted lanes sum to 1.0 and the
 * loss module's `q7_neg_log_q15` sees a normalized distribution.
 *
 * @section softmax_codes What lands in the output codes
 *
 *   - softmax: `q15_to_code (p)`, i.e. the same grid mllee's inference SOFTMAX
 *     emits (`ops.hpp::_S_softmax_row` normalizes a row to 127), so a trained
 *     graph's softmax layer serializes and evaluates unchanged.  The codes are
 *     the coarse 1/127 view of the distribution; the Q15 lanes saved on the tape
 *     are the accurate one the backward uses;
 *   - log-softmax: `log_softmax_codes`, one code = 8/127 nat with -127 as the
 *     rail at ln p = -8 -- the grid `_S_log_probability` in the inference path
 *     uses.  The log values themselves come from the row's log-sum-exp
 *     (`log_sum_exp_q7`, the streaming `q7_log1p_exp` fold), so a probability
 *     under the Q15 grid still gets its correct log instead of flooring.
 *
 * @section softmax_mask Masks
 *
 * A mask is an int8 code stream or an int16 lane stream over the same elements
 * as the logits, and the rule is the additive one: a **negative** mask value is
 * a -inf penalty, so that position gets probability zero, log rail -127 and no
 * gradient.  Zero or positive keeps the position.  The converters
 * (`softmax_build_mask`, `softmax_build_mask_q15`) turn either form into the
 * internal 0/1 keep stream, which is element-major like every other stream here,
 * so the same index maps a logit, its mask byte, its probability and its
 * gradient.  A row with no kept position is legal, not an error: its
 * probabilities are all zero, its log codes all -127 and its gradient all zero.
 *
 * @section softmax_temperature Temperature
 *
 * `_M_temperature_q15` is a Q15 real number (32768 = identity, and that value
 * short-circuits the scaling entirely).  Each widened logit is divided by it in
 * int64 before the max subtraction and re-clamped to the exp window, and the
 * backward multiplies the row gradient by its reciprocal, so the chain rule for
 * a sharpened distribution is exact rather than approximated.  A temperature
 * below 1.0 amplifies that gradient and can saturate a lane, which is counted.
 *
 * @section softmax_axis Which axis
 *
 * The row axis is a coordinate of the tensor, not a stride assumption: with
 * `rows` = product of the axes before the softmax axis, `length` = that axis and
 * `inner` = product of the axes after it, one softmax group is the strided run
 * `element = (row * length + pos) * inner + offset` and there are `rows * inner`
 * groups.  `inner == 1` (the softmax is the last axis, the case every classifier
 * uses) makes every stream stride-1 and contiguous, so `dot_lanes`, `axpy_lanes`
 * and `add_lanes` see aligned buffers; any other `inner` gathers each group's
 * lanes into the scratch rows first, which costs a copy and keeps the arithmetic
 * identical.
 *
 * @section softmax_pool Pool traffic
 *
 * With `E = rows * length * inner` elements and `L = length`:
 *
 *   - saved spec        : 20 bytes (five pool words);
 *   - saved probability : 2 E bytes (Q15 lanes, element-major) -- the backward
 *                         is the Jacobian of the row it normalized, so it needs
 *                         the distribution, not the logits;
 *   - saved keep stream : E bytes, masked layers only;
 *   - row scratch       : 8 L bytes (two int32 rows for the forward, which the
 *                         backward aliases as four int16 rows), plus 2 L more for
 *                         the staged probability row a strided axis needs -- a
 *                         backward function may not allocate, so the forward
 *                         carves it -- plus one more L when that strided axis is
 *                         masked, for the keep row gathered out of the
 *                         element-major mask.
 *
 * `SoftmaxSpec::bytes ()` is the sum of those, each region 4-byte aligned by
 * `Pool::carve`, and `softmax_run` lays them inside one slab so the whole node
 * is one measured number.
 */
namespace zzs::mllee::train
{
/** @brief Saved slot holding the spec copy the backward reads back. */
inline constexpr uint32_t _S_softmax_spec = 0u;

/** @brief Saved slot holding the Q15 probability lanes. */
inline constexpr uint32_t _S_softmax_prob = 1u;

/** @brief Saved slot holding the keep stream; null for an unmasked row. */
inline constexpr uint32_t _S_softmax_mask = 2u;

/** @brief Saved slot holding the row scratch region. */
inline constexpr uint32_t _S_softmax_scratch = 3u;

/** @brief One softmax code step in Q7 log units: 8 nats over 127 codes, so the
 *         conversion is `round (log_q7 * 127 / _S_softmax_log_code_q7)`. */
inline constexpr int32_t _S_softmax_log_code_q7 = 1024;

/** @brief The log-softmax code rail, i.e. the code of a masked or
 *         underflowed position (matches the inference LOG_SOFTMAX floor). */
inline constexpr int8_t _S_softmax_log_rail = -127;

/** @brief Q7 sentinel for a masked log lane: `q7_log_q15`'s own p <= 0 rail. */
inline constexpr int32_t _S_softmax_neg_inf_q7 = _S_q15_min;

/** @brief Bits between the two fixed domains this module mixes: a Q15 logit
 *         becomes a Q7 one by dropping this many bits, and a Q7 difference
 *         becomes a Q15 argument by shifting up by the same amount. */
inline constexpr unsigned _S_q15_to_q7_bits = _S_grad_frac_bits - _S_log_frac_bits;

/**
 * @brief One softmax layer: axis decomposition, temperature and output flavour.
 *
 * A plain POD saved in the pool, so the backward recovers the whole
 * configuration from the tape: no capture, no writable static.
 */
struct SoftmaxSpec
{
    /** @brief Product of the axes before the softmax axis. */
    uint16_t _M_rows = 1u;
    /** @brief Axis length, i.e. the number of positions one row normalizes. */
    uint16_t _M_length = 0u;
    /** @brief Product of the axes after the softmax axis (1 for a last axis). */
    uint16_t _M_inner = 1u;
    /** @brief Rank of the tensor the axis decomposition was taken from. */
    uint8_t _M_rank = 1u;
    /** @brief Axis index the softmax runs over. */
    uint8_t _M_axis = 0u;
    /** @brief Non-zero when a mask stream is supplied. */
    uint8_t _M_masked = 0u;
    /** @brief Non-zero for log-softmax codes instead of probability codes. */
    uint8_t _M_log = 0u;
    /** @brief Q15 temperature; 32768 is the identity and must be positive. */
    int32_t _M_temperature_q15 = _S_q15_one;
    /** @brief Op id recorded on the tape. */
    uint32_t _M_op_id = 0u;

    /** @brief True when the row gradient is scaled by the temperature. */
    [[nodiscard]] constexpr bool
    uses_temperature () const noexcept
    {
        return this->_M_temperature_q15 != _S_q15_one;
    }

    /** @brief True when a mask stream participates. */
    [[nodiscard]] constexpr bool
    is_masked () const noexcept
    {
        return this->_M_masked != 0u;
    }

    /** @brief True when the output codes are log-probabilities. */
    [[nodiscard]] constexpr bool
    is_log () const noexcept
    {
        return this->_M_log != 0u;
    }

    /** @brief Elements the layer reads and writes, E. */
    [[nodiscard]] constexpr uint32_t
    total () const noexcept
    {
        return static_cast<uint32_t> (this->_M_rows) * this->_M_length
               * this->_M_inner;
    }

    /** @brief Independent softmax groups: `rows * inner`. */
    [[nodiscard]] constexpr uint32_t
    groups () const noexcept
    {
        return static_cast<uint32_t> (this->_M_rows) * this->_M_inner;
    }

    /** @brief Bytes of one int32 scratch row. */
    [[nodiscard]] constexpr uint32_t
    row_bytes () const noexcept
    {
        return align_up (4u * this->_M_length, _S_pool_alignment);
    }

    /** @brief Bytes of the row scratch region: two int32 rows, plus the staging
     *         row a strided axis needs.  The forward's last pass still reads the
     *         exp-term row while it writes the probabilities, so a gathered row
     *         cannot alias either of the two int32 rows and costs its own 2 L.
     *         A masked strided axis gathers its keep stream too -- the caller's
     *         mask is element-major, while a row has to be read stride-1 -- so it
     *         costs one more L bytes. */
    [[nodiscard]] constexpr uint32_t
    scratch_bytes () const noexcept
    {
        if (this->_M_inner == 1u)
            return 2u * this->row_bytes ();
        uint32_t stage = align_up (2u * this->_M_length, _S_pool_alignment);
        if (this->is_masked ())
            stage += align_up (this->_M_length, _S_pool_alignment);
        return 2u * this->row_bytes () + stage;
    }

    /** @brief Pool bytes one recorded layer needs, per the file header table. */
    [[nodiscard]] constexpr uint32_t
    bytes () const noexcept
    {
        const uint32_t elements = this->total ();
        const uint32_t probabilities
            = align_up (2u * elements, _S_pool_alignment);
        const uint32_t keep
            = this->is_masked () ? align_up (elements, _S_pool_alignment) : 0u;
        return align_up (static_cast<uint32_t> (sizeof (SoftmaxSpec)),
                         _S_pool_alignment)
               + probabilities + keep + this->scratch_bytes ();
    }

    [[nodiscard]] constexpr Shape
    input_shape () const noexcept
    {
        return this->output_shape ();
    }

    /** @brief A row-major shape of `rows * length * inner` elements. */
    [[nodiscard]] constexpr Shape
    output_shape () const noexcept
    {
        Shape shape;
        shape._M_rank = this->_M_rank;
        uint32_t remaining = this->total ();
        uint16_t leading = 1u;
        if (this->_M_rank != 0u)
            {
                leading = static_cast<uint16_t> (remaining / this->_M_length
                                                 * 0u
                                                 + (remaining / this->_M_length
                                                    > _S_max_elements
                                                        ? _S_max_elements
                                                        : remaining
                                                              / this->_M_length));
                remaining = this->total ();
            }
        (void) leading;
        (void) remaining;
        /* The axis decomposition is what the kernels need; the shape only has to
           carry the right element count and rank, so the axes are laid out with
           the softmax axis in place and every other axis folded into `rows` and
           `inner` as far as a `uint16` axis can express. */
        for (uint8_t axis = 0; axis < _S_max_rank; ++axis)
            shape._M_dims[axis] = 1u;
        if (this->_M_rank == 0u || this->_M_rank > _S_max_rank)
            {
                shape._M_rank = 1u;
                shape._M_dims[0] = static_cast<uint16_t> (this->total ());
                return shape;
            }
        const uint16_t rows = this->_M_rows > _S_max_elements
                                  ? _S_max_elements
                                  : this->_M_rows;
        const uint16_t inner = this->_M_inner > _S_max_elements
                                   ? _S_max_elements
                                   : this->_M_inner;
        if (this->_M_rank == 1u)
            {
                shape._M_dims[0] = static_cast<uint16_t> (this->total ());
                shape._M_dims[0] = this->total () <= _S_max_elements
                                       ? static_cast<uint16_t> (this->total ())
                                       : 1u;
                return shape;
            }
        shape._M_dims[0] = rows == 0u ? 1u : rows;
        shape._M_dims[this->_M_rank - 1u] = inner == 0u ? 1u : inner;
        shape._M_dims[this->_M_axis] = this->_M_length;
        return shape;
    }

    /**
     * @brief Check the geometry, the axis and the temperature.
     *
     * @return BAD_LAYOUT for a zero axis count, a rank/axis pair the tensor model
     *         cannot hold, a non-positive temperature or a mask/log flag outside
     *         {0, 1}; OUT_OF_RANGE for an element count a training tensor cannot
     *         address.
     */
    [[nodiscard]] constexpr Status
    validate () const noexcept
    {
        if (this->_M_rows == 0u || this->_M_length == 0u || this->_M_inner == 0u)
            return Status::BAD_LAYOUT;
        if (this->_M_rank == 0u || this->_M_rank > _S_max_rank)
            return Status::BAD_LAYOUT;
        if (this->_M_axis >= this->_M_rank)
            return Status::BAD_LAYOUT;
        if (this->_M_masked > 1u || this->_M_log > 1u)
            return Status::BAD_LAYOUT;
        if (this->_M_temperature_q15 <= 0)
            return Status::BAD_LAYOUT;
        const uint64_t elements = static_cast<uint64_t> (this->_M_rows)
                                  * this->_M_length * this->_M_inner;
        if (elements > _S_max_elements)
            return Status::OUT_OF_RANGE;
        return Status::OK;
    }
};

static_assert (sizeof (SoftmaxSpec) == 20u,
               "a saved softmax spec is five pool words");

/** @brief Element of @p group's @p position lane, row-major. */
[[nodiscard]] inline constexpr uint32_t
softmax_element (const SoftmaxSpec &spec, const uint32_t group,
                 const uint32_t position) noexcept
{
    const uint32_t row = group / spec._M_inner;
    const uint32_t offset = group % spec._M_inner;
    return (row * spec._M_length + position) * spec._M_inner + offset;
}

/** @brief The int32 widened-logit row of one scratch region. */
[[nodiscard]] inline constexpr int32_t *
softmax_scratch_row (uint8_t *const scratch) noexcept
{
    return reinterpret_cast<int32_t *> (scratch);
}

/** @brief The int32 exp-term row of one scratch region. */
[[nodiscard]] inline constexpr int32_t *
softmax_scratch_work (uint8_t *const scratch, const uint32_t length) noexcept
{
    return reinterpret_cast<int32_t *> (scratch) + length;
}

/** @brief The @p row-th int16 row of one scratch region (the backward aliases
 *         the same 8 L bytes as four stride-1 lane rows). */
[[nodiscard]] inline constexpr int16_t *
softmax_scratch_lanes (uint8_t *const scratch, const uint32_t row,
                       const uint32_t length) noexcept
{
    return reinterpret_cast<int16_t *> (scratch) + row * length;
}

/** @brief The stride-1 keep row a strided, masked axis gathers its element-major
 *         mask into: it sits past the two int32 rows and the staged probability
 *         row, which is exactly the extra L bytes `scratch_bytes ()` counts. */
[[nodiscard]] inline constexpr uint8_t *
softmax_keep_stage (uint8_t *const scratch, const uint32_t length) noexcept
{
    return scratch + 2u * align_up (4u * length, _S_pool_alignment)
           + align_up (2u * length, _S_pool_alignment);
}

/** @brief Clamp a widened Q15 logit into the exp table's usable window. */
[[nodiscard]] inline constexpr int32_t
softmax_clamp_window (const int64_t value_q15) noexcept
{
    if (value_q15 > _S_exp_clamp_q15)
        return _S_exp_clamp_q15;
    if (value_q15 < -_S_exp_clamp_q15)
        return -_S_exp_clamp_q15;
    return static_cast<int32_t> (value_q15);
}

/**
 * @brief Widen one code into the Q15 logit domain on the tensor's scale.
 *
 * `round (code * scale_q15 / 127)` is the Q15 encoding of the real value
 * `code * scale_q15 / (127 * 32768)`, so the default scale reproduces
 * `code_to_q15` exactly and a larger scale buys logit range, clamped to the
 * +/-8.0 the exp path can represent.
 */
[[nodiscard]] inline constexpr int32_t
softmax_widen (const int32_t code, const int32_t scale_q15) noexcept
{
    const int64_t raw = static_cast<int64_t> (code) * scale_q15;
    return softmax_clamp_window ((raw + 64) / 127);
}

/** @brief Divide a Q15 logit by the Q15 temperature, staying inside the window. */
[[nodiscard]] inline constexpr int32_t
softmax_div_temperature (const int32_t value_q15,
                         const int32_t temperature_q15) noexcept
{
    if (temperature_q15 == _S_q15_one)
        return value_q15;
    const int64_t scaled = static_cast<int64_t> (value_q15) << _S_grad_frac_bits;
    return softmax_clamp_window (scaled / temperature_q15);
}

/**
 * @brief Divide a lane buffer by the temperature in place, counting clamps.
 *
 * The backward's counterpart of @ref softmax_div_temperature: the row gradient
 * is with respect to the *scaled* logits, so dL/dx_raw = g / T.  A temperature
 * below 1.0 amplifies, which is exactly the case that saturates a Q15 lane, so
 * every clamped lane is counted rather than hidden.
 */
[[nodiscard]] inline uint32_t
softmax_div_temperature_lanes (int16_t *const lanes, const uint32_t count,
                               const int32_t temperature_q15) noexcept
{
    uint32_t saturated = 0u;
    if (temperature_q15 == _S_q15_one)
        return saturated;
    for (uint32_t i = 0; i < count; ++i)
        {
            const int64_t scaled
                = static_cast<int64_t> (lanes[i]) << _S_grad_frac_bits;
            const int64_t quotient = scaled / temperature_q15;
            if (quotient > _S_q15_max || quotient < _S_q15_min)
                ++saturated;
            lanes[i] = q15_to_i16_sat (static_cast<int32_t> (
                quotient > _S_q15_max
                    ? _S_q15_max
                    : (quotient < _S_q15_min ? _S_q15_min : quotient)));
        }
    return saturated;
}

/**
 * @brief One row's normalization statistics.
 *
 * A leaf so a caller can reuse the same sweep for the log path without a second
 * pass over the widened logits.
 */
struct SoftmaxRow
{
    /** @brief Sum of the exp terms in Q15 units; at least 32767 when @p _M_any.
     *         Declared first so the 8-byte member leads the layout: that makes
     *         the struct 16 bytes on a 32- and a 64-bit target alike, and a row
     *         state sits on the row sweep's stack, where the gate counts bytes. */
    int64_t _M_total = 0;
    /** @brief Q15 maximum of the kept lanes; the exp window rail for an empty row. */
    int32_t _M_maximum = 0;
    /** @brief False for a fully masked row. */
    bool _M_any = false;
};

static_assert (sizeof (SoftmaxRow) == 16u,
               "one row state is a maximum, a sum and a flag");

/**
 * @brief exp(x - max) for one lane, in Q15.
 *
 * A gap past the exp window is an exact zero, not the table's stall value: the
 * table floors at 4 and a long row of floors would invent probability mass.
 */
[[nodiscard]] inline constexpr int32_t
softmax_exp_term (const int32_t gap_q15) noexcept
{
    if (gap_q15 > _S_exp_clamp_q15)
        return 0;
    return q15_exp_neg (gap_q15);
}

/**
 * @brief Normalize one widened, temperature-scaled row into Q15 probabilities.
 *
 * @p keep is a 0/1 stream over the same @p length positions (null keeps
 * everything).  @p work is a `length`-entry int32 scratch for the exp terms.
 * @p prob receives the probabilities; a fully masked row receives all zeros and
 * the returned state has `_M_any` clear.
 */
[[nodiscard]] inline SoftmaxRow
softmax_row_q15 (const int32_t *const logits_q15, const uint32_t length,
                 const uint8_t *const keep, int32_t *const work,
                 int16_t *const prob) noexcept
{
    SoftmaxRow state;
    for (uint32_t position = 0; position < length; ++position)
        {
            if (keep != nullptr && keep[position] == 0u)
                {
                    work[position] = 0;
                    continue;
                }
            const int32_t value = logits_q15[position];
            if (!state._M_any || value > state._M_maximum)
                {
                    state._M_maximum = value;
                    state._M_any = true;
                }
        }
    if (!state._M_any)
        {
            state._M_maximum = -_S_exp_clamp_q15;
            for (uint32_t position = 0; position < length; ++position)
                {
                    work[position] = 0;
                    prob[position] = 0;
                }
            return state;
        }
    int64_t total = 0;
    for (uint32_t position = 0; position < length; ++position)
        {
            if (keep != nullptr && keep[position] == 0u)
                {
                    work[position] = 0;
                    continue;
                }
            const int32_t term
                = softmax_exp_term (state._M_maximum - logits_q15[position]);
            work[position] = term;
            total += term;
        }
    /* The kept maximum always contributes q15_exp_neg (0) = 32767, so the
       divisor is positive without a rail check; the sum still needs the int64
       accumulator because a long row can exceed int32. */
    const int64_t bounded
        = total > 0x7FFFFFFFLL ? static_cast<int64_t> (_S_q15_max) << 16 : total;
    const int32_t divisor = static_cast<int32_t> (bounded);
    int32_t emitted = 0;
    uint32_t rail = 0u;
    for (uint32_t position = 0; position < length; ++position)
        {
            const int32_t value
                = keep != nullptr && keep[position] == 0u
                      ? 0
                      : ratio_q15 (work[position], divisor);
            prob[position] = q15_to_i16_sat (value);
            const int64_t running = static_cast<int64_t> (emitted) + value;
            emitted = running > _S_q15_max ? _S_q15_max : (running < _S_q15_min
                                                                ? _S_q15_min
                                                                : static_cast<int32_t> (
                                                                      running));
            if (value > _S_q15_max || value < _S_q15_min)
                ++rail;
        }
    /* Absorb the truncation residual on the argmax lane: every term is below
       the divisor, so a single lane always has room for the deficit. */
    if (rail == 0u && emitted < _S_q15_one && state._M_maximum >= 0)
        {
            for (uint32_t position = 0; position < length; ++position)
                if (logits_q15[position] == state._M_maximum
                    && (keep == nullptr || keep[position] != 0u))
                    {
                        prob[position] = q15_to_i16_sat (
                            prob[position] + (_S_q15_one - emitted));
                        break;
                    }
        }
    state._M_total = total;
    return state;
}

/**
 * @brief ln (sum of exp) of one widened row, in Q7 log units.
 *
 * The streaming fold `lse = max (lse, x) + ln (1 + exp (min - lse))` costs one
 * `q7_log1p_exp` per lane and never overflows, which is why it is the log atom
 * here instead of a second normalized pass.  A fully masked row returns the
 * -inf sentinel and the caller maps that to the log rail.
 */
[[nodiscard]] inline int32_t
log_sum_exp_q7 (const int32_t *const logits_q15, const uint32_t length,
                const uint8_t *const keep) noexcept
{
    bool started = false;
    int32_t lse_q7 = 0;
    for (uint32_t position = 0; position < length; ++position)
        {
            if (keep != nullptr && keep[position] == 0u)
                continue;
            const int32_t value = logits_q15[position] >> _S_q15_to_q7_bits;
            if (!started)
                {
                    lse_q7 = value;
                    started = true;
                    continue;
                }
            const int32_t high = value > lse_q7 ? value : lse_q7;
            const int32_t low = value > lse_q7 ? lse_q7 : value;
            lse_q7 = high
                     + q7_log1p_exp ((low - high) << _S_q15_to_q7_bits);
        }
    return started ? lse_q7 : _S_softmax_neg_inf_q7;
}

/**
 * @brief Log-softmax of one widened row into Q7 log lanes.
 *
 * @p state is the row's normalization statistics (from @ref softmax_row_q15, so
 * the maximum is already known) and @p work its exp terms; the batched form
 * `maximum + ln (total / 2^15)` needs one log call for the whole row, which is
 * both cheaper and more accurate than the per-lane fold.  Masked lanes get the
 * -inf sentinel.
 */
inline void
log_softmax_row_q7 (const int32_t *const logits_q15, const uint32_t length,
                    const uint8_t *const keep, const SoftmaxRow &state,
                    const int64_t *const total, int16_t *const log_q7) noexcept
{
    int32_t lse_q7 = _S_softmax_neg_inf_q7;
    if (state._M_any)
        {
            const int64_t bounded
                = *total > 0x7FFFFFFFLL
                      ? static_cast<int64_t> (_S_q15_max) << 16
                      : *total;
            lse_q7 = (state._M_maximum >> _S_q15_to_q7_bits)
                     + q7_log_q15 (static_cast<int32_t> (bounded));
        }
    for (uint32_t position = 0; position < length; ++position)
        {
            if (keep != nullptr && keep[position] == 0u)
                {
                    log_q7[position] = static_cast<int16_t> (
                        _S_softmax_neg_inf_q7);
                    continue;
                }
            if (!state._M_any)
                {
                    log_q7[position] = static_cast<int16_t> (
                        _S_softmax_neg_inf_q7);
                    continue;
                }
            const int32_t value = (logits_q15[position] >> _S_q15_to_q7_bits)
                                  - lse_q7;
            log_q7[position] = q15_to_i16_sat (value);
        }
}

/** @brief One Q7 log-probability as an inference log-softmax code. */
[[nodiscard]] inline constexpr int8_t
log_softmax_code_q7 (const int32_t log_q7) noexcept
{
    if (log_q7 >= 0)
        return 0;
    const int32_t code = div_round (log_q7 * 127, _S_softmax_log_code_q7);
    return _S_i8_code (code < _S_softmax_log_rail
                           ? static_cast<int32_t> (_S_softmax_log_rail)
                           : code);
}

/** @brief Requantize a Q7 log-softmax buffer into inference codes. */
inline void
log_softmax_codes (const int16_t *const log_q7, const uint32_t count,
                   int8_t *const codes) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        codes[i] = log_softmax_code_q7 (log_q7[i]);
}

/**
 * @brief Turn an int8 additive mask into the internal keep stream.
 *
 * Element-major in and out, so a keep byte sits at the same index as the logit
 * it gates.  A negative code is a -inf penalty.
 */
inline void
softmax_build_mask (const int8_t *const mask, const uint32_t count,
                    uint8_t *const keep) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        keep[i] = mask[i] < 0 ? 0u : 1u;
}

/** @brief int16 lane form of @ref softmax_build_mask. */
inline void
softmax_build_mask_q15 (const int16_t *const mask, const uint32_t count,
                        uint8_t *const keep) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        keep[i] = mask[i] < 0 ? 0u : 1u;
}

/** @brief Bind one view against the element count @p spec describes. */
[[nodiscard]] inline constexpr Status
softmax_check_operand (const Trainable &tensor, const uint32_t elements) noexcept
{
    if (tensor._M_code == nullptr)
        return Status::BAD_LAYOUT;
    if (tensor._M_count != elements)
        return Status::BAD_LAYOUT;
    return status_ok (tensor._M_shape.validate ()) ? Status::OK
                                                   : Status::BAD_LAYOUT;
}

/**
 * @brief Forward pass with no tape recording.
 *
 * @p keep is the internal 0/1 stream (null when the spec is unmasked), @p prob
 * receives the Q15 probabilities the backward needs and @p scratch is
 * `spec.scratch_bytes ()` bytes of 4-byte aligned pool storage.  The output's
 * gradient lanes are zeroed here, because a forward always precedes the backward
 * walk that accumulates into them.
 *
 * @return BAD_LAYOUT for a shape mismatch or a missing/extra mask, OK otherwise.
 */
[[nodiscard]] inline Status
softmax_forward (const SoftmaxSpec &spec, const Trainable &logits,
                 const uint8_t *const keep, Trainable &output,
                 int16_t *const prob, uint8_t *const scratch) noexcept
{
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    const Status in = softmax_check_operand (logits, spec.total ());
    if (!status_ok (in))
        return in;
    const Status out = softmax_check_operand (output, spec.total ());
    if (!status_ok (out))
        return out;
    if (spec.is_masked () != (keep != nullptr))
        return Status::BAD_LAYOUT;
    if (prob == nullptr || scratch == nullptr)
        return Status::BAD_LAYOUT;

    const uint32_t length = spec._M_length;
    const uint32_t groups = spec.groups ();
    const bool contiguous = spec._M_inner == 1u;
    const bool scaled = spec.uses_temperature ();
    const int32_t temperature = spec._M_temperature_q15;
    int32_t *const row = softmax_scratch_row (scratch);
    int32_t *const work = softmax_scratch_work (scratch, length);
    /* Lane row 2 aliases the first half of the exp-term int32 row: by the time
       the log sweep writes it, softmax_row_q15 has consumed `work` and only the
       logits are still read. */
    int16_t *const log_row = softmax_scratch_lanes (scratch, 2u, length);

    for (uint32_t group = 0; group < groups; ++group)
        {
            /* The caller's mask is element-major, while a row must be read
               stride-1, so a strided axis gathers its keep bytes alongside its
               logits before normalizing. */
            uint8_t *const keep_stage = !contiguous && keep != nullptr
                                            ? softmax_keep_stage (scratch, length)
                                            : nullptr;
            const uint8_t *const keep_row
                = keep == nullptr
                      ? nullptr
                      : (contiguous ? keep + group * length : keep_stage);
            for (uint32_t position = 0; position < length; ++position)
                {
                    const uint32_t element
                        = contiguous ? group * length + position
                                     : softmax_element (spec, group, position);
                    if (keep_stage != nullptr)
                        keep_stage[position] = keep[element];
                    const int32_t widened
                        = softmax_widen (logits._M_code[element],
                                         logits._M_scale_q15);
                    row[position]
                        = scaled ? softmax_div_temperature (widened, temperature)
                                 : widened;
                }
            int16_t *const prob_row
                = contiguous ? prob + group * length : nullptr;
            SoftmaxRow state;
            if (prob_row == nullptr)
                {
                    /* Strided destination: normalize into lane row 4 -- the
                       stage region scratch_bytes () reserves for it -- then
                       scatter the probabilities.  Rows 2 and 3 are the exp-term
                       int32 row's halves, and softmax_row_q15's last pass still
                       reads that row while it writes the lanes, so the staged
                       row has to sit past byte 8 L. */
                    int16_t *const staged
                        = softmax_scratch_lanes (scratch, 4u, length);
                    state = softmax_row_q15 (row, length, keep_row, work, staged);
                    for (uint32_t position = 0; position < length; ++position)
                        prob[softmax_element (spec, group, position)]
                            = staged[position];
                }
            else
                state = softmax_row_q15 (row, length, keep_row, work, prob_row);

            if (spec.is_log ())
                {
                    const int64_t total = state._M_total;
                    log_softmax_row_q7 (row, length, keep_row, state, &total,
                                        log_row);
                    for (uint32_t position = 0; position < length; ++position)
                        {
                            const uint32_t element
                                = contiguous
                                      ? group * length + position
                                      : softmax_element (spec, group, position);
                            output._M_code[element]
                                = log_softmax_code_q7 (log_row[position]);
                        }
                    continue;
                }
            for (uint32_t position = 0; position < length; ++position)
                {
                    const uint32_t element
                        = contiguous ? group * length + position
                                     : softmax_element (spec, group, position);
                    const int32_t value
                        = contiguous
                              ? prob_row[position]
                              : prob[softmax_element (spec, group, position)];
                    output._M_code[element] = q15_to_code (value);
                }
        }
    output.zero_grad ();
    return Status::OK;
}

/**
 * @brief Softmax Jacobian for one row: g = s (d - sum (s d)).
 *
 * The contraction is `dot_lanes` (an int64 accumulator narrowed to Q15, so a
 * long row cannot wrap), the rank-one term `-c * s` goes through `axpy_lanes`
 * into a zeroed destination -- provably clamp-free, since both factors are Q15
 * lanes and their product is inside the rail by one LSB -- and the diagonal term
 * `s * d` is folded in by `add_lanes`, whose clamp count is returned.
 */
[[nodiscard]] inline uint32_t
softmax_backward_q15 (const int16_t *const prob, const int16_t *const grad_out,
                      int16_t *const grad_in, const uint32_t length,
                      int16_t *const prod) noexcept
{
    const int32_t contraction = dot_lanes (prob, grad_out, length);
    for (uint32_t position = 0; position < length; ++position)
        /* the narrowing is safe by the invariant this function documents: `prob`
           is a Q15 probability in [0, 32767], so no product reaches 32768 even
           when `grad_out` sits on the negative rail */
        prod[position] = static_cast<int16_t> (
            q15_mul (prob[position], grad_out[position]));
    fill_lanes (grad_in, 0, length);
    axpy_lanes (grad_in, prob, -contraction, length);
    return add_lanes (grad_in, prod, length);
}

/**
 * @brief Log-softmax Jacobian for one row: g = d - s * sum (d).
 *
 * @p grad_out arrives already zeroed at masked positions (the row kernel does
 * that while it gathers the lanes, so a masked position contributes neither to
 * the sum nor to its own gradient).  The sum is int64 and narrows through
 * `clamp_sum_q15`, whose rail is counted, then the same zero-then-axpy-then-add
 * shape as @ref softmax_backward_q15 applies the rank-one term.
 */
[[nodiscard]] inline uint32_t
log_softmax_backward_q15 (const int16_t *const prob,
                          const int16_t *const grad_out,
                          int16_t *const grad_in, const uint32_t length) noexcept
{
    int64_t total = 0;
    for (uint32_t position = 0; position < length; ++position)
        total += grad_out[position];
    const bool railed = total > _S_q15_max || total < _S_q15_min;
    const int32_t sum = clamp_sum_q15 (total);
    fill_lanes (grad_in, 0, length);
    axpy_lanes (grad_in, prob, -sum, length);
    uint32_t saturated = add_lanes (grad_in, grad_out, length);
    if (railed)
        ++saturated;
    return saturated;
}

/**
 * @brief Backward in the `BackwardFn` signature.
 *
 * Recorded node layout: operands [logits]; saved [spec, probabilities, keep |
 * null, scratch].  The gradient of every group is built in the scratch's
 * contiguous int16 rows, so the SIMD primitives always see aligned stride-1
 * streams even when the tensor's softmax axis is strided; the scatter back into
 * the logits operand is contiguous (`inner == 1`) or one `add_grad` per lane.
 */
[[nodiscard]] inline Status
softmax_backward (const Trainable &output, const Trainable &grad_output,
                  GradContext &context) noexcept
{
    const SoftmaxSpec *const slot
        = context.saved_as<SoftmaxSpec> (_S_softmax_spec);
    if (slot == nullptr)
        return Status::BAD_LAYOUT;
    const SoftmaxSpec spec = *slot;
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    if (context._M_operands == nullptr || context._M_operand_count < 1u)
        return Status::BAD_LAYOUT;
    if (context._M_saved_count <= _S_softmax_scratch)
        return Status::BAD_LAYOUT;
    const Trainable &logits = context._M_operands[0];
    if (output._M_count != spec.total () || logits._M_count != spec.total ())
        return Status::BAD_LAYOUT;
    if (grad_output._M_grad == nullptr || logits._M_grad == nullptr)
        return Status::OK;
    const int16_t *const prob = context.saved_as<int16_t> (_S_softmax_prob);
    if (prob == nullptr)
        return Status::BAD_LAYOUT;
    const Saved &mask_slot = context._M_saved[_S_softmax_mask];
    const uint8_t *const keep
        = mask_slot.is_null () ? nullptr
                               : context.saved_as<uint8_t> (_S_softmax_mask);
    if (spec.is_masked () && keep == nullptr)
        return Status::BAD_LAYOUT;
    int16_t *const scratch = context.saved_as<int16_t> (_S_softmax_scratch);
    if (scratch == nullptr)
        return Status::BAD_LAYOUT;

    const uint32_t length = spec._M_length;
    const uint32_t groups = spec.groups ();
    const bool contiguous = spec._M_inner == 1u;
    /* The backward owns the region as lane rows only -- there is no int32
       exp-term row here -- so it has four of them in the 8 L bytes every spec
       reserves.  Row 3 is the strided path's gathered-probability row. */
    int16_t *const d_row = scratch;
    int16_t *const grad_row = softmax_scratch_lanes (
        reinterpret_cast<uint8_t *> (scratch), 1u, length);
    int16_t *const prod_row = softmax_scratch_lanes (
        reinterpret_cast<uint8_t *> (scratch), 2u, length);
    int16_t *const p_row = softmax_scratch_lanes (
        reinterpret_cast<uint8_t *> (scratch), 3u, length);
    uint32_t saturated = 0u;

    for (uint32_t group = 0; group < groups; ++group)
        {
            /* Same gather as the forward: the keep stream is element-major, the
               row kernel reads it stride-1. */
            uint8_t *const keep_stage = !contiguous && keep != nullptr
                                            ? softmax_keep_stage (
                                                  reinterpret_cast<uint8_t *> (scratch),
                                                  length)
                                            : nullptr;
            const uint8_t *const keep_row
                = keep == nullptr
                      ? nullptr
                      : (contiguous ? keep + group * length : keep_stage);
            const int16_t *const prob_row
                = contiguous ? prob + group * length : p_row;
            for (uint32_t position = 0; position < length; ++position)
                {
                    const uint32_t element
                        = contiguous
                              ? group * length + position
                              : softmax_element (spec, group, position);
                    if (keep_stage != nullptr)
                        keep_stage[position] = keep[element];
                    const int32_t gated
                        = keep_row != nullptr && keep_row[position] == 0u
                              ? 0
                              : grad_output._M_grad[element];
                    d_row[position] = q15_to_i16_sat (gated);
                    if (!contiguous)
                        p_row[position] = prob[element];
                }
            if (spec.is_log ())
                saturated += log_softmax_backward_q15 (prob_row, d_row, grad_row,
                                                       length);
            else
                saturated += softmax_backward_q15 (prob_row, d_row, grad_row,
                                                   length, prod_row);
            if (spec.uses_temperature ())
                saturated += softmax_div_temperature_lanes (
                    grad_row, length, spec._M_temperature_q15);
            if (contiguous)
                saturated += add_lanes (logits._M_grad + group * length, grad_row,
                                        length);
            else
                for (uint32_t position = 0; position < length; ++position)
                    if (logits.add_grad (softmax_element (spec, group, position),
                                         grad_row[position]))
                        ++saturated;
        }
    context._M_saturation_events += saturated;
    return Status::OK;
}

/**
 * @brief Save the row state, run the forward and record the node.
 *
 * Everything the backward reads is laid out inside one carve of the tape's own
 * pool, so `SoftmaxSpec::bytes ()` is the whole cost of a softmax node and
 * `Tape::reset` frees the layer in one step.  @p mask is the caller's additive
 * mask (element-major int8 codes, negative = -inf) and is converted to the
 * internal keep stream into the pool; pass null exactly when
 * `SoftmaxSpec::_M_masked` is clear.
 *
 * @return BAD_LAYOUT for a shape/mask mismatch, OUT_OF_RANGE for an oversized
 *         tensor, ARENA_OVERFLOW when the pool cannot supply the state, INTERNAL
 *         when the tape is not open.
 */
[[nodiscard]] inline Status
softmax_run (Tape &tape, const SoftmaxSpec &spec, const Trainable &logits,
             const int8_t *const mask, Trainable &output) noexcept
{
    const Status status = spec.validate ();
    if (!status_ok (status))
        return status;
    const Status in = softmax_check_operand (logits, spec.total ());
    if (!status_ok (in))
        return in;
    const Status out = softmax_check_operand (output, spec.total ());
    if (!status_ok (out))
        return out;
    if (spec.is_masked () != (mask != nullptr))
        return Status::BAD_LAYOUT;

    Pool &pool = tape.pool ();
    Saved slots[_S_max_operands];
    Trainable operands[_S_max_operands];
    Slab slab;
    const Status carve = pool.carve (spec.bytes (), 1u, slab);
    if (!status_ok (carve))
        return carve;
    uint8_t *const raw = pool.at (slab._M_offset);
    if (raw == nullptr)
        return Status::BAD_LAYOUT;

    const uint32_t spec_bytes
        = align_up (static_cast<uint32_t> (sizeof (SoftmaxSpec)),
                    _S_pool_alignment);
    const uint32_t elements = spec.total ();
    const uint32_t prob_bytes = align_up (2u * elements, _S_pool_alignment);
    const uint32_t keep_bytes
        = spec.is_masked () ? align_up (elements, _S_pool_alignment) : 0u;

    SoftmaxSpec *saved_spec = reinterpret_cast<SoftmaxSpec *> (raw);
    int16_t *const prob = reinterpret_cast<int16_t *> (raw + spec_bytes);
    uint8_t *const keep
        = keep_bytes != 0u ? raw + spec_bytes + prob_bytes : nullptr;
    uint8_t *const scratch = raw + spec_bytes + prob_bytes + keep_bytes;

    slots[_S_softmax_spec]._M_offset = slab._M_offset;
    slots[_S_softmax_spec]._M_bytes = spec_bytes;
    slots[_S_softmax_spec]._M_count = 1u;
    slots[_S_softmax_prob]._M_offset = slab._M_offset + spec_bytes;
    slots[_S_softmax_prob]._M_bytes = prob_bytes;
    slots[_S_softmax_prob]._M_count = elements;
    slots[_S_softmax_mask]._M_offset
        = keep_bytes != 0u ? slab._M_offset + spec_bytes + prob_bytes : 0u;
    slots[_S_softmax_mask]._M_bytes = keep_bytes;
    slots[_S_softmax_mask]._M_count = keep_bytes != 0u ? elements : 0u;
    slots[_S_softmax_scratch]._M_offset
        = slab._M_offset + spec_bytes + prob_bytes + keep_bytes;
    slots[_S_softmax_scratch]._M_bytes = spec.scratch_bytes ();
    slots[_S_softmax_scratch]._M_count = spec.scratch_bytes () / 2u;

    *saved_spec = spec;
    if (keep != nullptr)
        softmax_build_mask (mask, elements, keep);

    const Status forward
        = softmax_forward (spec, logits, keep, output, prob, scratch);
    if (!status_ok (forward))
        return forward;

    operands[0] = logits;
    for (uint32_t index = 1u; index < _S_max_operands; ++index)
        operands[index] = Trainable{};
    return tape.record (softmax_backward, spec._M_op_id, output, operands, 1u,
                        slots, _S_max_operands);
}
} // namespace zzs::mllee::train
