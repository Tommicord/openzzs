/* Copyright(c) 2026 Tommicord
 *
 * Host unit tests for mllee/train_conv.hpp: the im2col convolution layer of
 * the mllee training stack.  Every expectation in this file is produced by an
 * independent naive reference written here, never by the library's own output.
 *
 * Covered: conv_axis_length, ConvSpec geometry/counts/validate(), the region
 * and stride helpers, conv_forward (codes domain, requant shift, bias,
 * layouts, groups) and the operand rejection paths.  Tape/backward coverage
 * is added in the later sections of this file.
 */

#include <catch2/catch_test_macros.hpp>

#include "mllee/train_conv.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>
using namespace zzs::mllee;
using namespace zzs::mllee::train;

namespace
{
    /** @brief Q15 unity multiplier, mirrored from the library contract. */
    constexpr int32_t k_q15_one = 32768;

    /** @brief Arithmetic shift right with round-half-up, int64 domain. */
    constexpr int64_t
    half_up_shift (const int64_t value, const uint32_t shift) noexcept
    {
        if (shift == 0u)
            return value;
        return (value + (int64_t{ 1 } << (shift - 1u))) >> shift;
    }

    /** @brief A flat operand view with no shape filled in (rank 0). */
    Trainable
    make_view (int8_t *const code, int16_t *const grad,
               const uint32_t count) noexcept
    {
        Trainable view;
        view._M_code = code;
        view._M_grad = grad;
        view._M_count = count;
        return view;
    }

    /** @brief Saturate to the emitted int8 code range [-127, 127]. */
    constexpr int8_t
    clamp_i8 (const int64_t value) noexcept
    {
        if (value > 127)
            return 127;
        if (value < -127)
            return -127;
        return static_cast<int8_t> (value);
    }

    /** @brief Code-domain activation, only the kinds these tests fuse. */
    constexpr int8_t
    act_expected (const Act act, const int32_t pre_code) noexcept
    {
        switch (act)
        {
        case Act::RELU:
            return pre_code < 0 ? 0 : clamp_i8 (pre_code);
        case Act::RELU6:
            return pre_code < 0 ? 0 : (pre_code > 127 ? 127 : clamp_i8 (pre_code));
        case Act::NONE:
        default:
            return clamp_i8 (pre_code);
        }
    }

    /** @brief Independent nested-loop forward in the code domain. */
    void
    naive_forward (const ConvSpec &spec, const int8_t *const input,
                   const int8_t *const weight, const int8_t *const bias,
                   int8_t *const output, int8_t *const pre = nullptr) noexcept
    {
        const uint32_t batch = spec._M_batch;
        const uint32_t ic = spec._M_in_channels;
        const uint32_t oc = spec._M_out_channels;
        const uint32_t in0 = spec._M_in[0];
        const uint32_t in1 = spec._M_in[1];
        const uint32_t o0 = spec.out_length (0u);
        const uint32_t o1 = spec.out_length (1u);
        const uint32_t ipg = spec.in_per_group ();
        const uint32_t opg = spec.out_per_group ();
        const uint32_t taps = spec.taps ();
        const uint32_t k0 = spec._M_kernel[0];
        const uint32_t k1 = spec._M_kernel[1];
        const bool cm = spec._M_layout == ConvLayout::CHANNEL_MAJOR;

        auto in_index = [cm, ic, in0, in1] (const uint32_t b, const uint32_t c,
                                            const uint32_t h,
                                            const uint32_t w) noexcept
        {
            return cm ? ((b * ic + c) * in0 + h) * in1 + w
                      : ((b * in0 + h) * in1 + w) * ic + c;
        };
        auto out_index = [cm, oc, o0, o1] (const uint32_t b, const uint32_t c,
                                           const uint32_t h,
                                           const uint32_t w) noexcept
        {
            return cm ? ((b * oc + c) * o0 + h) * o1 + w
                      : ((b * o0 + h) * o1 + w) * oc + c;
        };

        for (uint32_t b = 0; b < batch; ++b)
            for (uint32_t co = 0; co < oc; ++co)
                {
                    const uint32_t group = co / opg;
                    const uint32_t local = co % opg;
                    for (uint32_t a0 = 0; a0 < o0; ++a0)
                        for (uint32_t a1 = 0; a1 < o1; ++a1)
                            {
                                int64_t acc = 0;
                                for (uint32_t cl = 0; cl < ipg; ++cl)
                                    for (uint32_t t0 = 0; t0 < k0; ++t0)
                                        for (uint32_t t1 = 0; t1 < k1; ++t1)
                                            {
                                                const int32_t s0 = static_cast<
                                                    int32_t> (a0
                                                              * spec._M_stride[0])
                                                                   + static_cast<
                                                                       int32_t> (
                                                                       t0
                                                                       * spec
                                                                           ._M_dilation
                                                                               [0])
                                                                   - spec._M_pad[0];
                                                const int32_t s1 = static_cast<
                                                    int32_t> (a1
                                                              * spec._M_stride[1])
                                                                   + static_cast<
                                                                       int32_t> (
                                                                       t1
                                                                       * spec
                                                                           ._M_dilation
                                                                               [1])
                                                                   - spec._M_pad[2];
                                                if (s0 < 0
                                                    || static_cast<uint32_t> (s0)
                                                           >= in0
                                                    || s1 < 0
                                                    || static_cast<uint32_t> (s1)
                                                           >= in1)
                                                    continue;
                                                const uint32_t column
                                                    = (cl * k0 + t0) * k1 + t1;
                                                const uint32_t w_index
                                                    = (group * opg + local) * taps
                                                      + column;
                                                acc += static_cast<int64_t> (
                                                           input[in_index (
                                                               b, group * ipg + cl,
                                                               static_cast<uint32_t> (
                                                                   s0),
                                                               static_cast<uint32_t> (
                                                                   s1))])
                                                       * weight[w_index];
                                            }
                                int64_t value = half_up_shift (
                                    acc, spec._M_requant_shift);
                                if (spec._M_requant_mult_q15 != k_q15_one)
                                    value = half_up_shift (
                                        value * spec._M_requant_mult_q15, 15u);
                                if (spec.has_bias ())
                                    value += bias[co];
                                const int8_t code = clamp_i8 (value);
                                const uint32_t target = out_index (b, co, a0, a1);
                                if (spec.has_activation ())
                                    {
                                        if (pre != nullptr)
                                            pre[target] = code;
                                        output[target] = act_expected (spec._M_act,
                                                                        code);
                                    }
                                else
                                    output[target] = code;
                            }
                }
    }
    /** @brief One code step in Q15, mirrored from the documented contract. */
    constexpr int32_t k_code_step = 258;
    constexpr int32_t k_q15_max = 32767;
    constexpr int32_t k_q15_min = -32768;

    constexpr int32_t
    q15_clamp_ref (const int64_t value) noexcept
    {
        if (value > k_q15_max)
            return k_q15_max;
        if (value < k_q15_min)
            return k_q15_min;
        return static_cast<int32_t> (value);
    }

    /** @brief int64 Q30 reduction narrowed to a Q15 lane, clamp-before-shift. */
    constexpr int32_t
    narrow_sum_ref (const int64_t total) noexcept
    {
        const int64_t limit = static_cast<int64_t> (k_q15_max) << 15;
        const int64_t bounded
            = total > limit ? limit : (total < -limit ? -limit : total);
        return q15_clamp_ref (half_up_shift (bounded, 15u));
    }

    /** @brief Plain saturating Q15 sum used by the bias reduction. */
    constexpr int32_t
    clamp_sum_ref (const int64_t total) noexcept
    {
        return q15_clamp_ref (total);
    }

    /** @brief Saturating lane accumulation; true when it clamped. */
    bool
    sat_add_ref (int16_t *const lane, const int32_t delta) noexcept
    {
        const int32_t sum = static_cast<int32_t> (lane[0]) + delta;
        if (sum > k_q15_max || sum < k_q15_min)
            {
                lane[0] = static_cast<int16_t> (q15_clamp_ref (sum));
                return true;
            }
        lane[0] = static_cast<int16_t> (sum);
        return false;
    }

    /**
     * @brief Independent backward: dX (col2im), dW and db, mirroring the
     *        per-(item, group) block structure and the narrowing rules.
     *
     * @p grad is the already activation-routed lane stream in flat output
     *        order.  Null gradient buffers mean that operand stays frozen, as
     *        in the library.
     */
    void
    naive_backward (const ConvSpec &spec, const int8_t *const input,
                    const int8_t *const weight, const int16_t *const grad,
                    int16_t *const dx, int16_t *const dw, int16_t *const db,
                    uint32_t &saturated) noexcept
    {
        const uint32_t batch = spec._M_batch;
        const uint32_t ic = spec._M_in_channels;
        const uint32_t oc = spec._M_out_channels;
        const uint32_t in0 = spec._M_in[0];
        const uint32_t in1 = spec._M_in[1];
        const uint32_t o0 = spec.out_length (0u);
        const uint32_t o1 = spec.out_length (1u);
        const uint32_t ipg = spec.in_per_group ();
        const uint32_t opg = spec.out_per_group ();
        const uint32_t taps = spec.taps ();
        const uint32_t rows = spec.rows ();
        const uint32_t k0 = spec._M_kernel[0];
        const uint32_t k1 = spec._M_kernel[1];
        const bool cm = spec._M_layout == ConvLayout::CHANNEL_MAJOR;

        auto in_index = [cm, ic, in0, in1] (const uint32_t b, const uint32_t c,
                                            const uint32_t h,
                                            const uint32_t w) noexcept
        {
            return cm ? ((b * ic + c) * in0 + h) * in1 + w
                      : ((b * in0 + h) * in1 + w) * ic + c;
        };
        auto out_index = [cm, oc, o0, o1] (const uint32_t b, const uint32_t c,
                                           const uint32_t h,
                                           const uint32_t w) noexcept
        {
            return cm ? ((b * oc + c) * o0 + h) * o1 + w
                      : ((b * o0 + h) * o1 + w) * oc + c;
        };

        std::vector<int16_t> patch (static_cast<size_t> (rows) * taps);
        std::vector<int16_t> dy (static_cast<size_t> (rows) * opg);
        std::vector<int8_t> col (static_cast<size_t> (rows) * taps);

        for (uint32_t item = 0; item < batch; ++item)
            for (uint32_t group = 0; group < spec._M_groups; ++group)
                {
                    /* gather dY into the packed [rows, opg] block */
                    for (uint32_t row = 0; row < rows; ++row)
                        {
                            const uint32_t a0 = o1 != 0u ? row / o1 : 0u;
                            const uint32_t a1 = row % o1;
                            for (uint32_t local = 0; local < opg; ++local)
                                dy[row * opg + local]
                                    = grad[out_index (item, group * opg + local,
                                                      a0, a1)];
                        }
                    /* rebuild the im2col collection from the input codes */
                    uint32_t column = 0u;
                    for (uint32_t channel = 0; channel < ipg; ++channel)
                        for (uint32_t t0 = 0; t0 < k0; ++t0)
                            for (uint32_t t1 = 0; t1 < k1; ++t1)
                                {
                                    for (uint32_t row = 0; row < rows; ++row)
                                        {
                                            const uint32_t a0
                                                = o1 != 0u ? row / o1 : 0u;
                                            const uint32_t a1 = row % o1;
                                            const int32_t s0 = static_cast<
                                                int32_t> (a0 * spec._M_stride[0])
                                                + static_cast<int32_t> (
                                                      t0 * spec._M_dilation[0])
                                                - spec._M_pad[0];
                                            const int32_t s1 = static_cast<
                                                int32_t> (a1 * spec._M_stride[1])
                                                + static_cast<int32_t> (
                                                      t1 * spec._M_dilation[1])
                                                - spec._M_pad[2];
                                            col[row * taps + column]
                                                = (s0 >= 0
                                                   && static_cast<uint32_t> (s0)
                                                          < in0
                                                   && s1 >= 0
                                                   && static_cast<uint32_t> (s1)
                                                          < in1)
                                                      ? input[in_index (
                                                            item,
                                                            group * ipg + channel,
                                                            static_cast<uint32_t> (
                                                                s0),
                                                            static_cast<uint32_t> (
                                                                s1))]
                                                      : 0;
                                        }
                                    ++column;
                                }

                    /* dPatch = dY x W, in the kernel's row-major order */
                    if (dx != nullptr)
                        for (uint32_t row = 0; row < rows; ++row)
                            for (uint32_t c = 0; c < taps; ++c)
                                {
                                    int64_t sum = 0;
                                    for (uint32_t local = 0; local < opg;
                                         ++local)
                                        sum
                                            += static_cast<int64_t> (
                                                   dy[row * opg + local])
                                                 * static_cast<int64_t> (
                                                     weight[(group * opg
                                                             + local)
                                                            * taps
                                                            + c])
                                                 * k_code_step;
                                    /* the patch buffer starts zeroed, so this
                                       add is a plain set of a clamped lane */
                                    patch[row * taps + c] = static_cast<int16_t> (
                                        narrow_sum_ref (sum));
                                }

                    /* dW = dY^T x collection, accumulated per item */
                    if (dw != nullptr)
                        for (uint32_t local = 0; local < opg; ++local)
                            for (uint32_t c = 0; c < taps; ++c)
                                {
                                    int64_t sum = 0;
                                    for (uint32_t row = 0; row < rows; ++row)
                                        sum += static_cast<int64_t> (
                                                   dy[row * opg + local])
                                               * static_cast<int64_t> (
                                                   col[row * taps + c])
                                               * k_code_step;
                                    if (sat_add_ref (
                                            dw + (group * opg + local) * taps + c,
                                            narrow_sum_ref (sum)))
                                        ++saturated;
                                }

                    /* col2im in the library's (channel, tap, row) walk */
                    if (dx != nullptr)
                        {
                            column = 0u;
                            for (uint32_t channel = 0; channel < ipg; ++channel)
                                for (uint32_t t0 = 0; t0 < k0; ++t0)
                                    for (uint32_t t1 = 0; t1 < k1; ++t1)
                                        {
                                            for (uint32_t row = 0; row < rows;
                                                 ++row)
                                                {
                                                    const uint32_t a0 = o1 != 0u
                                                                              ? row
                                                                                / o1
                                                                              : 0u;
                                                    const uint32_t a1
                                                        = row % o1;
                                                    const int32_t s0
                                                        = static_cast<int32_t> (
                                                              a0
                                                              * spec._M_stride[0])
                                                          + static_cast<int32_t> (
                                                                t0
                                                                * spec._M_dilation
                                                                    [0])
                                                          - spec._M_pad[0];
                                                    const int32_t s1
                                                        = static_cast<int32_t> (
                                                              a1
                                                              * spec._M_stride[1])
                                                          + static_cast<int32_t> (
                                                                t1
                                                                * spec._M_dilation
                                                                    [1])
                                                          - spec._M_pad[2];
                                                    if (s0 >= 0
                                                        && static_cast<uint32_t> (
                                                               s0)
                                                               < in0
                                                        && s1 >= 0
                                                        && static_cast<uint32_t> (
                                                               s1)
                                                               < in1)
                                                        {
                                                            const uint32_t target
                                                                = in_index (
                                                                    item,
                                                                    group * ipg
                                                                        + channel,
                                                                    static_cast<
                                                                        uint32_t> (
                                                                        s0),
                                                                    static_cast<
                                                                        uint32_t> (
                                                                        s1));
                                                            const int16_t delta
                                                                = patch[row
                                                                        * taps
                                                                    + column];
                                                            if (sat_add_ref (
                                                                    dx + target,
                                                                    delta))
                                                                ++saturated;
                                                        }
                                                }
                                            ++column;
                                        }
                        }

                    /* db = channel sum of dY, then a per-group accumulate */
                    if (db != nullptr)
                        for (uint32_t local = 0; local < opg; ++local)
                            {
                                int64_t total = 0;
                                for (uint32_t row = 0; row < rows; ++row)
                                    total += dy[row * opg + local];
                                const int32_t clamped = clamp_sum_ref (total);
                                if (static_cast<int64_t> (clamped) != total)
                                    ++saturated;
                                if (sat_add_ref (db + group * opg + local,
                                                 clamped))
                                    ++saturated;
                            }
                }
    }
} // namespace

TEST_CASE ("conv_axis_length is floor(span/stride)+1 and clamps to 0",
           "[mllee][train][conv][geometry]")
{
    /* stride 1 */
    CHECK (conv_axis_length (5u, 3u, 1u, 1u, 0, 0) == 3);
    CHECK (conv_axis_length (7u, 3u, 1u, 1u, 0, 0) == 5);
    CHECK (conv_axis_length (1u, 1u, 1u, 1u, 0, 0) == 1);
    /* stride 2 */
    CHECK (conv_axis_length (5u, 3u, 2u, 1u, 0, 0) == 2);
    CHECK (conv_axis_length (6u, 3u, 2u, 1u, 0, 0) == 2);
    CHECK (conv_axis_length (7u, 3u, 2u, 1u, 0, 0) == 3);
    /* stride 3 */
    CHECK (conv_axis_length (5u, 3u, 3u, 1u, 0, 0) == 1);
    CHECK (conv_axis_length (6u, 3u, 3u, 1u, 0, 0) == 2);
    CHECK (conv_axis_length (8u, 3u, 3u, 1u, 0, 0) == 2);
    CHECK (conv_axis_length (9u, 3u, 3u, 1u, 0, 0) == 3);
    CHECK (conv_axis_length (4u, 2u, 10u, 1u, 0, 0) == 1);
    /* padding, including asymmetric */
    CHECK (conv_axis_length (5u, 3u, 1u, 1u, 1, 1) == 5);
    CHECK (conv_axis_length (5u, 3u, 1u, 1u, 2, 0) == 5);
    CHECK (conv_axis_length (5u, 3u, 1u, 1u, 0, 2) == 5);
    CHECK (conv_axis_length (5u, 3u, 2u, 1u, 1, 1) == 3);
    CHECK (conv_axis_length (6u, 5u, 1u, 1u, 2, 2) == 6);
    /* dilation 2 */
    CHECK (conv_axis_length (5u, 3u, 1u, 2u, 0, 0) == 1);
    CHECK (conv_axis_length (6u, 3u, 1u, 2u, 0, 0) == 2);
    CHECK (conv_axis_length (7u, 3u, 2u, 2u, 0, 0) == 2);
    CHECK (conv_axis_length (4u, 2u, 1u, 2u, 0, 0) == 2);
    CHECK (conv_axis_length (4u, 2u, 2u, 2u, 0, 0) == 1);
    CHECK (conv_axis_length (8u, 3u, 1u, 2u, 1, 1) == 6);
    /* negative numerator reports 0, never a wrapped value */
    CHECK (conv_axis_length (2u, 3u, 1u, 1u, 0, 0) == 0);
    CHECK (conv_axis_length (1u, 3u, 1u, 1u, 1, 0) == 0);
    CHECK (conv_axis_length (3u, 4u, 1u, 1u, 0, 0) == 0);
    CHECK (conv_axis_length (3u, 2u, 1u, 3u, 0, 0) == 0);
    CHECK (conv_axis_length (4u, 3u, 1u, 2u, 0, 0) == 0);
    /* boundary: span exactly 0 still places one window */
    CHECK (conv_axis_length (3u, 3u, 5u, 1u, 0, 0) == 1);

    static_assert (conv_axis_length (5u, 3u, 1u, 1u, 0, 0) == 3);
    static_assert (conv_axis_length (2u, 3u, 1u, 1u, 0, 0) == 0);
}

TEST_CASE ("ConvSpec derived geometry: out_length, rows, taps, groups",
           "[mllee][train][conv][geometry]")
{
    SECTION ("rank-1 defaults are a 1x1 pin on axis 1")
    {
        ConvSpec spec;
        spec._M_batch = 2u;
        spec._M_in_channels = 3u;
        spec._M_out_channels = 2u;
        spec._M_in[0] = 7u;
        spec._M_kernel[0] = 3u;
        spec._M_stride[0] = 2u;
        spec._M_pad[0] = 1;
        spec._M_pad[1] = 1;
        CHECK (spec.out_length (0u) == 4u);
        CHECK (spec.out_length (1u) == 1u);
        CHECK (spec.out_length (2u) == 1u);
        CHECK (spec.rows () == 4u);
        /* taps = in_per_group * k0 * k1 = 3 * 3 * 1 */
        CHECK (spec.taps () == 9u);
        CHECK (spec.in_per_group () == 3u);
        CHECK (spec.out_per_group () == 2u);
        CHECK (spec.input_elements () == 42u);
        CHECK (spec.output_elements () == 16u);
        CHECK (spec.weight_elements () == 18u);
        CHECK (spec.has_bias ());
        CHECK (!spec.has_activation ());
    }
    SECTION ("rank-2 geometry")
    {
        ConvSpec spec;
        spec._M_rank = 2u;
        spec._M_in_channels = 2u;
        spec._M_out_channels = 4u;
        spec._M_in[0] = 5u;
        spec._M_in[1] = 6u;
        spec._M_kernel[0] = 3u;
        spec._M_kernel[1] = 3u;
        spec._M_stride[0] = 2u;
        spec._M_stride[1] = 2u;
        spec._M_pad[0] = 1;
        spec._M_pad[1] = 1;
        spec._M_pad[2] = 1;
        spec._M_pad[3] = 1;
        CHECK (spec.out_length (0u) == 3u);
        CHECK (spec.out_length (1u) == 3u);
        CHECK (spec.out_length (2u) == 1u);
        CHECK (spec.rows () == 9u);
        CHECK (spec.taps () == 18u);
        CHECK (spec.input_elements () == 60u);
        CHECK (spec.output_elements () == 36u);
        CHECK (spec.weight_elements () == 72u);
    }
    SECTION ("groups split channels both sides")
    {
        ConvSpec spec;
        spec._M_in_channels = 6u;
        spec._M_out_channels = 4u;
        spec._M_groups = 2u;
        spec._M_in[0] = 5u;
        spec._M_kernel[0] = 3u;
        CHECK (spec.in_per_group () == 3u);
        CHECK (spec.out_per_group () == 2u);
        CHECK (spec.taps () == 9u);
        CHECK (spec.weight_elements () == 36u);
        CHECK (spec.input_elements () == 30u);
        /* out_length(0) = (5-3)/1+1 = 3, rows = 3 */
        CHECK (spec.output_elements () == 12u);
        spec._M_groups = 0u;
        CHECK (spec.in_per_group () == 0u);
        CHECK (spec.out_per_group () == 0u);
    }
    SECTION ("rank-2 grouped with dilation")
    {
        ConvSpec spec;
        spec._M_rank = 2u;
        spec._M_batch = 3u;
        spec._M_in_channels = 4u;
        spec._M_out_channels = 6u;
        spec._M_groups = 2u;
        spec._M_in[0] = 8u;
        spec._M_in[1] = 7u;
        spec._M_kernel[0] = 2u;
        spec._M_kernel[1] = 3u;
        spec._M_dilation[0] = 2u;
        CHECK (spec.in_per_group () == 2u);
        CHECK (spec.out_per_group () == 3u);
        CHECK (spec.taps () == 12u);
        /* axis 0: (8 - 2*(2-1) - 1)/1 + 1 = 6; axis 1: (7-3)/1+1 = 5 */
        CHECK (spec.out_length (0u) == 6u);
        CHECK (spec.out_length (1u) == 5u);
        CHECK (spec.rows () == 30u);
        CHECK (spec.weight_elements () == 72u);
        CHECK (spec.input_elements () == 672u);
        CHECK (spec.output_elements () == 540u);
    }
}

TEST_CASE ("ConvSpec::validate rejects each malformed geometry",
           "[mllee][train][conv][validate]")
{
    const auto base = [] () noexcept
    {
        ConvSpec spec;
        spec._M_in_channels = 2u;
        spec._M_out_channels = 2u;
        spec._M_in[0] = 5u;
        return spec;
    };
    REQUIRE (status_ok (base ().validate ()));

    SECTION ("rank out of range")
    {
        auto spec = base ();
        spec._M_rank = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec._M_rank = 3u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec._M_rank = 255u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("zero batch, channels or groups")
    {
        auto spec = base ();
        spec._M_batch = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_in_channels = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_out_channels = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_groups = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("zero spatial extent")
    {
        auto spec = base ();
        spec._M_in[0] = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_rank = 2u;
        spec._M_in[1] = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("requant shift beyond 31")
    {
        auto spec = base ();
        spec._M_requant_shift = 32u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec._M_requant_shift = 31u;
        CHECK (status_ok (spec.validate ()));
        spec._M_requant_shift = 7u;
        CHECK (status_ok (spec.validate ()));
    }
    SECTION ("requant multiplier outside the Q15 window")
    {
        auto spec = base ();
        spec._M_requant_mult_q15 = -1;
        CHECK (spec.validate () == Status::OUT_OF_RANGE);
        spec._M_requant_mult_q15 = 1048577;
        CHECK (spec.validate () == Status::OUT_OF_RANGE);
        spec._M_requant_mult_q15 = 1048576;
        CHECK (status_ok (spec.validate ()));
        spec._M_requant_mult_q15 = 0;
        CHECK (status_ok (spec.validate ()));
    }
    SECTION ("channels not divisible by groups")
    {
        auto spec = base ();
        spec._M_in_channels = 3u;
        spec._M_groups = 2u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_out_channels = 3u;
        spec._M_groups = 2u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("layout tag outside the enum")
    {
        auto spec = base ();
        spec._M_layout = static_cast<ConvLayout> (7u);
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec._M_layout = static_cast<ConvLayout> (1u);
        CHECK (status_ok (spec.validate ()));
    }
    SECTION ("zero kernel, stride or dilation on either axis")
    {
        auto spec = base ();
        spec._M_kernel[0] = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_stride[0] = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_dilation[0] = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_rank = 2u;
        spec._M_kernel[1] = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_rank = 2u;
        spec._M_stride[1] = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_rank = 2u;
        spec._M_dilation[1] = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("window that cannot be placed is BAD_LAYOUT")
    {
        auto spec = base ();
        spec._M_kernel[0] = 7u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_in[0] = 2u;
        spec._M_kernel[0] = 4u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_rank = 2u;
        spec._M_in[1] = 2u;
        spec._M_kernel[1] = 5u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("rank-1 axis-1 pin must stay at identity")
    {
        auto spec = base ();
        spec._M_in[1] = 2u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_kernel[1] = 2u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_stride[1] = 2u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_dilation[1] = 2u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_pad[2] = 1;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_pad[3] = 1;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("element counts the arena cannot address")
    {
        auto spec = base ();
        spec._M_batch = 255u;
        spec._M_in_channels = 255u;
        spec._M_out_channels = 255u;
        spec._M_in[0] = 255u;
        CHECK (spec.validate () == Status::OUT_OF_RANGE);
        spec = base ();
        spec._M_rank = 2u;
        spec._M_batch = 255u;
        spec._M_in_channels = 255u;
        spec._M_out_channels = 255u;
        spec._M_in[0] = 255u;
        spec._M_in[1] = 255u;
        CHECK (spec.validate () == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("conv_forward 1x1 rank-1 matches the naive code-domain reference",
           "[mllee][train][conv][forward]")
{
    ConvSpec spec;
    spec._M_batch = 2u;
    spec._M_in_channels = 3u;
    spec._M_out_channels = 2u;
    spec._M_in[0] = 3u;
    REQUIRE (status_ok (spec.validate ()));
    /* rows = 3 = taps (3 channels x 1x1 kernel): the im2col block is square. */
    REQUIRE (spec.rows () == 3u);
    REQUIRE (spec.taps () == 3u);
    REQUIRE (spec.input_elements () == 18u);
    REQUIRE (spec.weight_elements () == 6u);
    REQUIRE (spec.output_elements () == 12u);

    int8_t input[18];
    for (uint32_t i = 0; i < 18u; ++i)
        input[i] = static_cast<int8_t> ((i % 13u) - 6u);
    int8_t weight[6] = { 40, -30, 20, 15, -45, 35 };
    int8_t bias[2] = { 3, -5 };
    int8_t output[12] = { 0 };
    int8_t expected[12] = { 0 };
    alignas (64) int8_t scratch[2048];

    naive_forward (spec, input, weight, bias, expected);
    auto in_view = make_view (input, nullptr, 18u);
    auto w_view = make_view (weight, nullptr, 6u);
    auto b_view = make_view (bias, nullptr, 2u);
    auto out_view = make_view (output, nullptr, 12u);
    REQUIRE (conv_forward (spec, in_view, w_view, b_view, out_view, nullptr,
                           scratch)
             == Status::OK);
    for (uint32_t i = 0; i < 12u; ++i)
        CHECK (output[i] == expected[i]);

    /* every emitted code stays inside [-127, 127] */
    for (uint32_t i = 0; i < 12u; ++i)
        CHECK (output[i] != -128);

    SECTION ("saturation path stays inside the code range")
    {
        /* a big weight drives some accumulators past the int8 ceiling. */
        int8_t fat_weight[6] = { 127, 127, 127, -127, -127, -127 };
        int8_t out2[12] = { 0 };
        auto w2 = make_view (fat_weight, nullptr, 6u);
        auto o2 = make_view (out2, nullptr, 12u);
        REQUIRE (conv_forward (spec, in_view, w2, b_view, o2, nullptr, scratch)
                 == Status::OK);
        int8_t expected2[12];
        naive_forward (spec, input, fat_weight, bias, expected2);
        for (uint32_t i = 0; i < 12u; ++i)
        {
            CHECK (out2[i] == expected2[i]);
            CHECK (out2[i] >= -127);
            CHECK (out2[i] <= 127);
        }
    }
}

TEST_CASE ("conv_forward grouped spatial-major matches the reference",
           "[mllee][train][conv][forward]")
{
    ConvSpec spec;
    spec._M_batch = 2u;
    spec._M_in_channels = 2u;
    spec._M_out_channels = 6u;
    spec._M_groups = 2u;
    spec._M_in[0] = 5u;
    spec._M_kernel[0] = 3u;
    spec._M_pad[0] = 1;
    spec._M_pad[1] = 1;
    spec._M_layout = ConvLayout::SPATIAL_MAJOR;
    REQUIRE (status_ok (spec.validate ()));
    /* ipg = 1, taps = 3, opg = 3, rows = 5 */
    REQUIRE (spec.in_per_group () == 1u);
    REQUIRE (spec.taps () == 3u);
    REQUIRE (spec.out_per_group () == 3u);
    REQUIRE (spec.rows () == 5u);
    REQUIRE (spec.input_elements () == 20u);
    REQUIRE (spec.weight_elements () == 18u);
    REQUIRE (spec.output_elements () == 60u);

    int8_t input[20];
    for (uint32_t i = 0; i < 20u; ++i)
        input[i] = static_cast<int8_t> ((i * 5u % 19u) - 9u);
    int8_t weight[18];
    for (uint32_t i = 0; i < 18u; ++i)
        weight[i] = static_cast<int8_t> ((i * 11u % 23u) - 11u);
    int8_t bias[6] = { 1, -2, 3, -4, 5, -6 };
    int8_t output[60] = { 0 };
    int8_t expected[60] = { 0 };
    alignas (64) int8_t scratch[4096];

    naive_forward (spec, input, weight, bias, expected);
    auto in_view = make_view (input, nullptr, 20u);
    auto w_view = make_view (weight, nullptr, 18u);
    auto b_view = make_view (bias, nullptr, 6u);
    auto out_view = make_view (output, nullptr, 60u);
    REQUIRE (conv_forward (spec, in_view, w_view, b_view, out_view, nullptr,
                           scratch)
             == Status::OK);
    for (uint32_t i = 0; i < 60u; ++i)
        CHECK (output[i] == expected[i]);
}

TEST_CASE ("conv_forward stride 2 pad 1 dilation matches the reference",
           "[mllee][train][conv][forward]")
{
    ConvSpec spec;
    spec._M_batch = 2u;
    spec._M_in_channels = 2u;
    spec._M_out_channels = 4u;
    spec._M_in[0] = 11u;
    spec._M_kernel[0] = 3u;
    spec._M_stride[0] = 2u;
    spec._M_pad[0] = 1;
    spec._M_pad[1] = 1;
    REQUIRE (status_ok (spec.validate ()));
    /* span = 11 + 2 - 2 - 1 = 10, out = 10/2 + 1 = 6 = taps (2 channels x 3) */
    REQUIRE (spec.out_length (0u) == 6u);
    REQUIRE (spec.rows () == 6u);
    REQUIRE (spec.taps () == 6u);
    REQUIRE (spec.input_elements () == 44u);
    REQUIRE (spec.weight_elements () == 24u);
    REQUIRE (spec.output_elements () == 48u);

    int8_t input[44];
    for (uint32_t i = 0; i < 44u; ++i)
        input[i] = static_cast<int8_t> ((i * 7u % 17u) - 8u);
    int8_t weight[24];
    for (uint32_t i = 0; i < 24u; ++i)
        weight[i] = static_cast<int8_t> ((i * 3u % 21u) - 10u);
    int8_t bias[4] = { 2, -3, 1, 0 };
    int8_t output[48] = { 0 };
    int8_t expected[48] = { 0 };
    alignas (64) int8_t scratch[4096];

    naive_forward (spec, input, weight, bias, expected);
    auto in_view = make_view (input, nullptr, 44u);
    auto w_view = make_view (weight, nullptr, 24u);
    auto b_view = make_view (bias, nullptr, 4u);
    auto out_view = make_view (output, nullptr, 48u);
    REQUIRE (conv_forward (spec, in_view, w_view, b_view, out_view, nullptr,
                           scratch)
             == Status::OK);
    for (uint32_t i = 0; i < 48u; ++i)
        CHECK (output[i] == expected[i]);

    SECTION ("no bias output matches the bias-less reference")
    {
        auto nb = spec;
        nb._M_use_bias = 0u;
        int8_t out2[48] = { 0 };
        int8_t exp2[48] = { 0 };
        auto o2 = make_view (out2, nullptr, 48u);
        naive_forward (nb, input, weight, nullptr, exp2);
        REQUIRE (conv_forward (nb, in_view, w_view, b_view, o2, nullptr,
                               scratch)
                 == Status::OK);
        for (uint32_t i = 0; i < 48u; ++i)
            CHECK (out2[i] == exp2[i]);
    }

    SECTION ("requant shift other than 7 matches the reference")
    {
        auto rs = spec;
        rs._M_requant_shift = 4u;
        int8_t out3[48] = { 0 };
        int8_t exp3[48] = { 0 };
        auto o3 = make_view (out3, nullptr, 48u);
        naive_forward (rs, input, weight, bias, exp3);
        REQUIRE (conv_forward (rs, in_view, w_view, b_view, o3, nullptr,
                               scratch)
                 == Status::OK);
        for (uint32_t i = 0; i < 48u; ++i)
            CHECK (out3[i] == exp3[i]);
        /* the shift really changes the result: at least one code differs */
        bool any_diff = false;
        for (uint32_t i = 0; i < 48u; ++i)
            any_diff = any_diff || (out3[i] != expected[i]);
        CHECK (any_diff);
    }
}

TEST_CASE ("conv_forward rejects bad operands and scratch",
           "[mllee][train][conv][forward]")
{
    ConvSpec spec;
    spec._M_in_channels = 2u;
    spec._M_out_channels = 2u;
    spec._M_in[0] = 2u;
    spec._M_kernel[0] = 1u;
    REQUIRE (status_ok (spec.validate ()));
    REQUIRE (spec.rows () == 2u);
    REQUIRE (spec.taps () == 2u);

    int8_t input[4] = { 1, -2, 3, -4 };
    int8_t weight[4] = { 5, 6, -7, 8 };
    int8_t bias[2] = { 0, 1 };
    int8_t output[4] = { 0 };
    alignas (64) int8_t scratch[2048];
    auto ok_in = make_view (input, nullptr, 4u);
    auto ok_w = make_view (weight, nullptr, 4u);
    auto ok_b = make_view (bias, nullptr, 2u);
    auto ok_out = make_view (output, nullptr, 4u);
    REQUIRE (conv_forward (spec, ok_in, ok_w, ok_b, ok_out, nullptr, scratch)
             == Status::OK);

    SECTION ("null code pointer")
    {
        auto bad = make_view (nullptr, nullptr, 4u);
        CHECK (conv_forward (spec, bad, ok_w, ok_b, ok_out, nullptr, scratch)
               == Status::BAD_LAYOUT);
    }
    SECTION ("input count mismatch")
    {
        auto bad = make_view (input, nullptr, 3u);
        CHECK (conv_forward (spec, bad, ok_w, ok_b, ok_out, nullptr, scratch)
               == Status::BAD_LAYOUT);
    }
    SECTION ("weight count mismatch")
    {
        auto bad = make_view (weight, nullptr, 5u);
        CHECK (conv_forward (spec, ok_in, bad, ok_b, ok_out, nullptr, scratch)
               == Status::BAD_LAYOUT);
    }
    SECTION ("output count mismatch")
    {
        auto bad = make_view (output, nullptr, 5u);
        CHECK (conv_forward (spec, ok_in, ok_w, ok_b, bad, nullptr, scratch)
               == Status::BAD_LAYOUT);
    }
    SECTION ("bias count mismatch only when the bias is used")
    {
        auto bad = make_view (bias, nullptr, 3u);
        CHECK (conv_forward (spec, ok_in, ok_w, bad, ok_out, nullptr, scratch)
               == Status::BAD_LAYOUT);
        auto nb = spec;
        nb._M_use_bias = 0u;
        CHECK (status_ok (nb.validate ()));
        auto dummy_bias = make_view (bias, nullptr, 1u);
        CHECK (conv_forward (nb, ok_in, ok_w, dummy_bias, ok_out, nullptr,
                             scratch)
               == Status::OK);
    }
    SECTION ("null and misaligned scratch")
    {
        CHECK (conv_forward (spec, ok_in, ok_w, ok_b, ok_out, nullptr, nullptr)
               == Status::BAD_LAYOUT);
        CHECK (conv_forward (spec, ok_in, ok_w, ok_b, ok_out, nullptr,
                             scratch + 1)
               == Status::BAD_LAYOUT);
    }
    SECTION ("invalid spec short-circuits before any operand check")
    {
        auto bad_spec = spec;
        bad_spec._M_kernel[0] = 0u;
        CHECK (conv_forward (bad_spec, ok_in, ok_w, ok_b, ok_out, nullptr,
                             scratch)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("conv_regions and conv_element agree with the documented slab",
           "[mllee][train][conv][regions]")
{
    ConvSpec spec;
    spec._M_batch = 2u;
    spec._M_in_channels = 3u;
    spec._M_out_channels = 4u;
    spec._M_in[0] = 7u;
    spec._M_kernel[0] = 3u;
    spec._M_stride[0] = 2u;
    spec._M_pad[0] = 1;
    spec._M_pad[1] = 1;
    REQUIRE (status_ok (spec.validate ()));

    const auto region = conv_regions (spec);
    /* rows = 4, taps = 9, opg = 4, total = 2*4*4 = 32 */
    CHECK (spec.rows () == 4u);
    CHECK (spec.taps () == 9u);
    CHECK (region._M_block == 0u);
    CHECK (region._M_collection
           == ((4u * 4u * 4u + 3u) / 4u) * 4u);
    CHECK (region._M_patch
           == region._M_collection + ((4u * 9u + 3u) / 4u) * 4u);
    CHECK (region._M_grad
           == region._M_patch + ((2u * 4u * 9u + 3u) / 4u) * 4u);
    CHECK (region._M_routed
           == region._M_grad + ((2u * 4u * 4u + 3u) / 4u) * 4u);
    CHECK (region._M_bias
           == region._M_routed + ((2u * 32u + 3u) / 4u) * 4u);
    CHECK (region._M_bytes
           == region._M_bias + ((2u * 4u + 3u) / 4u) * 4u);
    CHECK (spec.scratch_bytes () == region._M_bytes);
    CHECK (region._M_bytes == spec.scratch_bytes ());
    CHECK (spec.bytes () == region._M_bytes + 44u + 32u + 36u);

    SECTION ("strides, channel major")
    {
        spec._M_rank = 2u;
        spec._M_in[1] = 5u;
        spec._M_kernel[1] = 2u;
        /* input [2, 3, 7, 5]: batch stride 3*7*5, channel stride 7*5 */
        const auto in_s = conv_input_strides (spec);
        CHECK (in_s._M_batch == 3u * 7u * 5u);
        CHECK (in_s._M_channel == 7u * 5u);
        CHECK (in_s._M_axis0 == 5u);
        CHECK (in_s._M_axis1 == 1u);
        CHECK (conv_element (in_s, 1u, 2u, 3u, 4u) == 1u * 105u + 2u * 35u + 3u * 5u + 4u);
        /* output [2, 4, 4, 4]: kernel 2 on axis 1 shortens it to 4, rows = 16 */
        REQUIRE (spec.out_length (0u) == 4u);
        REQUIRE (spec.out_length (1u) == 4u);
        REQUIRE (spec.rows () == 16u);
        const auto out_s = conv_output_strides (spec);
        CHECK (out_s._M_batch == 4u * 16u);
        CHECK (out_s._M_channel == 16u);
        CHECK (out_s._M_axis0 == 4u);
        CHECK (out_s._M_axis1 == 1u);
        CHECK (conv_element (out_s, 1u, 2u, 3u, 4u)
               == 1u * 64u + 2u * 16u + 3u * 4u + 4u);
    }
    SECTION ("strides, spatial major")
    {
        spec._M_layout = ConvLayout::SPATIAL_MAJOR;
        spec._M_rank = 2u;
        spec._M_in[1] = 5u;
        /* input [2, 7, 5, 3]: batch 7*5*3, channel 1, axis0 5*3, axis1 3 */
        const auto in_s = conv_input_strides (spec);
        CHECK (in_s._M_batch == 7u * 5u * 3u);
        CHECK (in_s._M_channel == 1u);
        CHECK (in_s._M_axis0 == 5u * 3u);
        CHECK (in_s._M_axis1 == 3u);
        CHECK (conv_element (in_s, 2u, 1u, 3u, 4u)
               == 2u * 105u + 1u + 3u * 15u + 4u * 3u);
        /* output [2, 4, 5, 4]: kernel[1] stays 1 so out_length(1) = 5 */
        REQUIRE (spec.out_length (1u) == 5u);
        REQUIRE (spec.rows () == 20u);
        const auto out_s = conv_output_strides (spec);
        CHECK (out_s._M_batch == 20u * 4u);
        CHECK (out_s._M_channel == 1u);
        CHECK (out_s._M_axis0 == 5u * 4u);
        CHECK (out_s._M_axis1 == 4u);
        CHECK (conv_element (out_s, 1u, 2u, 3u, 4u)
               == 1u * 80u + 2u + 3u * 20u + 4u * 4u);
    }
}

TEST_CASE ("conv_forward fused RELU writes pre-activation codes and rectified output",
           "[mllee][train][conv][activation]")
{
    ConvSpec spec;
    spec._M_batch = 2u;
    spec._M_in_channels = 3u;
    spec._M_out_channels = 2u;
    spec._M_in[0] = 3u;
    spec._M_act = Act::RELU;
    REQUIRE (status_ok (spec.validate ()));
    REQUIRE (spec.has_activation ());
    REQUIRE (spec.rows () == spec.taps ());

    int8_t input[18];
    for (uint32_t i = 0; i < 18u; ++i)
        input[i] = static_cast<int8_t> ((i % 13u) - 6u);
    int8_t weight[6] = { 40, -30, 20, 15, -45, 35 };
    int8_t bias[2] = { 3, -5 };
    int8_t output[12] = { 0 };
    int8_t pre[12] = { 0 };
    int8_t expected_out[12] = { 0 };
    int8_t expected_pre[12] = { 0 };
    alignas (64) int8_t scratch[2048];

    naive_forward (spec, input, weight, bias, expected_out, expected_pre);
    /* the data really exercises the rectifier: some pre-codes are negative */
    bool any_negative = false;
    for (uint32_t i = 0; i < 12u; ++i)
        any_negative = any_negative || (expected_pre[i] < 0);
    CHECK (any_negative);

    auto in_view = make_view (input, nullptr, 18u);
    auto w_view = make_view (weight, nullptr, 6u);
    auto b_view = make_view (bias, nullptr, 2u);
    auto out_view = make_view (output, nullptr, 12u);
    REQUIRE (conv_forward (spec, in_view, w_view, b_view, out_view, pre,
                           scratch)
             == Status::OK);
    for (uint32_t i = 0; i < 12u; ++i)
        {
            CHECK (pre[i] == expected_pre[i]);
            CHECK (output[i] == expected_out[i]);
            CHECK (output[i] == (pre[i] < 0 ? 0 : pre[i]));
        }

    SECTION ("missing pre-activation buffer is BAD_LAYOUT")
    {
        CHECK (conv_forward (spec, in_view, w_view, b_view, out_view, nullptr,
                             scratch)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("conv_forward fused RELU6 saturates in the code domain",
           "[mllee][train][conv][activation]")
{
    ConvSpec spec;
    spec._M_batch = 1u;
    spec._M_in_channels = 3u;
    spec._M_out_channels = 1u;
    spec._M_in[0] = 3u;
    spec._M_act = Act::RELU6;
    REQUIRE (status_ok (spec.validate ()));
    REQUIRE (spec.rows () == 3u);
    REQUIRE (spec.taps () == 3u);

    /* three 127*127 taps drive the pre-activation past the int8 ceiling */
    int8_t input[9] = { 127, 127, 127, 127, 127, 127, 127, 127, 127 };
    int8_t weight[3] = { 127, 127, 127 };
    int8_t bias[1] = { 0 };
    int8_t output[3] = { 0 };
    int8_t pre[3] = { 0 };
    int8_t expected_out[3];
    int8_t expected_pre[3];
    alignas (64) int8_t scratch[2048];
    naive_forward (spec, input, weight, bias, expected_out, expected_pre);
    REQUIRE (expected_pre[0] == 127);

    auto in_view = make_view (input, nullptr, 9u);
    auto w_view = make_view (weight, nullptr, 3u);
    auto b_view = make_view (bias, nullptr, 1u);
    auto out_view = make_view (output, nullptr, 3u);
    REQUIRE (conv_forward (spec, in_view, w_view, b_view, out_view, pre,
                           scratch)
             == Status::OK);
    for (uint32_t i = 0; i < 3u; ++i)
        {
            CHECK (pre[i] == expected_pre[i]);
            CHECK (output[i] == expected_out[i]);
            CHECK (output[i] <= 127);
            CHECK (output[i] >= 0);
        }

    SECTION ("negative pre-activation still rectifies to zero")
    {
        int8_t neg_input[9] = { -127, -127, -127, -127, -127,
                                -127, -127, -127, -127 };
        int8_t exp_out[3];
        int8_t exp_pre[3];
        naive_forward (spec, neg_input, weight, bias, exp_out, exp_pre);
        REQUIRE (exp_pre[0] == -127);
        auto neg_view = make_view (neg_input, nullptr, 9u);
        int8_t out2[3] = { 9 };
        int8_t pre2[3] = { 9 };
        auto o2 = make_view (out2, nullptr, 3u);
        REQUIRE (conv_forward (spec, neg_view, w_view, b_view, o2, pre2,
                               scratch)
                 == Status::OK);
        for (uint32_t i = 0; i < 3u; ++i)
            {
                CHECK (pre2[i] == -127);
                CHECK (out2[i] == 0);
            }
    }
}

TEST_CASE ("requant shift rounds half up and never emits the -128 code",
           "[mllee][train][conv][requant]")
{
    /* smallest legal geometry: one output position, one tap */
    ConvSpec spec;
    spec._M_batch = 1u;
    spec._M_in_channels = 1u;
    spec._M_out_channels = 1u;
    spec._M_in[0] = 1u;
    spec._M_kernel[0] = 1u;
    REQUIRE (status_ok (spec.validate ()));
    REQUIRE (spec.rows () == 1u);
    REQUIRE (spec.taps () == 1u);

    alignas (64) int8_t scratch[512];
    int8_t input[1];
    int8_t weight[1];
    int8_t bias[1] = { 5 };
    int8_t output[1];
    int8_t expected[1];

    /* (in, weight) pairs whose products hit the rounding boundaries */
    struct Pair
    {
        int8_t a;
        int8_t b;
    };
    const Pair pairs[] = {
        { 1, 64 }, { 1, -64 }, { 1, 63 }, { 1, 65 }, { 1, -63 }, { 1, -65 },
        { -1, 64 }, { -1, -64 }, { 127, 127 }, { -127, 127 }, { 127, -127 },
        { -127, -127 }, { 10, 20 }, { -10, 20 }, { 0, 127 }, { 3, 7 },
        { -3, -7 }, { 64, 64 }, { -64, 64 }, { 100, -100 }
    };

    SECTION ("shift 7 without bias")
    {
        spec._M_use_bias = 0u;
        for (const Pair &p : pairs)
            {
                input[0] = p.a;
                weight[0] = p.b;
                output[0] = 42;
                naive_forward (spec, input, weight, bias, expected);
                auto in_view = make_view (input, nullptr, 1u);
                auto w_view = make_view (weight, nullptr, 1u);
                auto b_view = make_view (bias, nullptr, 1u);
                auto out_view = make_view (output, nullptr, 1u);
                REQUIRE (conv_forward (spec, in_view, w_view, b_view, out_view,
                                       nullptr, scratch)
                         == Status::OK);
                CHECK (output[0] == expected[0]);
                CHECK (output[0] != -128);
                CHECK (output[0] >= -127);
                CHECK (output[0] <= 127);
            }
        /* the exact half at shift 7 rounds up: 64 -> 1, -64 -> 0 */
        input[0] = 1;
        weight[0] = 64;
        CHECK (half_up_shift (64, 7u) == 1);
        input[0] = 1;
        weight[0] = -64;
        CHECK (half_up_shift (-64, 7u) == 0);
    }

    SECTION ("shift 7 with bias adds the bias code after narrowing")
    {
        for (const Pair &p : pairs)
            {
                input[0] = p.a;
                weight[0] = p.b;
                output[0] = 42;
                naive_forward (spec, input, weight, bias, expected);
                auto in_view = make_view (input, nullptr, 1u);
                auto w_view = make_view (weight, nullptr, 1u);
                auto b_view = make_view (bias, nullptr, 1u);
                auto out_view = make_view (output, nullptr, 1u);
                REQUIRE (conv_forward (spec, in_view, w_view, b_view, out_view,
                                       nullptr, scratch)
                         == Status::OK);
                CHECK (output[0] == expected[0]);
                CHECK (output[0] != -128);
            }
    }

    SECTION ("shift 0 is the identity narrowing (with saturation)")
    {
        auto nb = spec;
        nb._M_use_bias = 0u;
        nb._M_requant_shift = 0u;
        for (const Pair &p : pairs)
            {
                input[0] = p.a;
                weight[0] = p.b;
                output[0] = 42;
                naive_forward (nb, input, weight, bias, expected);
                auto in_view = make_view (input, nullptr, 1u);
                auto w_view = make_view (weight, nullptr, 1u);
                auto b_view = make_view (bias, nullptr, 1u);
                auto out_view = make_view (output, nullptr, 1u);
                REQUIRE (conv_forward (nb, in_view, w_view, b_view, out_view,
                                       nullptr, scratch)
                         == Status::OK);
                CHECK (output[0] == expected[0]);
                CHECK (output[0] != -128);
            }
        /* 10*20 = 200 must saturate to 127, -200 to -127 */
        input[0] = 10;
        weight[0] = 20;
        auto in_view = make_view (input, nullptr, 1u);
        auto w_view = make_view (weight, nullptr, 1u);
        auto b_view = make_view (bias, nullptr, 1u);
        auto out_view = make_view (output, nullptr, 1u);
        REQUIRE (conv_forward (nb, in_view, w_view, b_view, out_view, nullptr,
                               scratch)
                 == Status::OK);
        CHECK (output[0] == 127);
        input[0] = -10;
        REQUIRE (conv_forward (nb, in_view, w_view, b_view, out_view, nullptr,
                               scratch)
                 == Status::OK);
        CHECK (output[0] == -127);
    }

    SECTION ("shift 31 reduces every small product to zero")
    {
        auto nb = spec;
        nb._M_use_bias = 0u;
        nb._M_requant_shift = 31u;
        for (const Pair &p : pairs)
            {
                input[0] = p.a;
                weight[0] = p.b;
                output[0] = 42;
                naive_forward (nb, input, weight, bias, expected);
                auto in_view = make_view (input, nullptr, 1u);
                auto w_view = make_view (weight, nullptr, 1u);
                auto b_view = make_view (bias, nullptr, 1u);
                auto out_view = make_view (output, nullptr, 1u);
                REQUIRE (conv_forward (nb, in_view, w_view, b_view, out_view,
                                       nullptr, scratch)
                         == Status::OK);
                CHECK (output[0] == expected[0]);
                CHECK (output[0] == 0);
            }
    }
}

TEST_CASE ("requant multiplier scales the narrowed accumulator in Q15",
           "[mllee][train][conv][requant]")
{
    ConvSpec spec;
    spec._M_batch = 1u;
    spec._M_in_channels = 1u;
    spec._M_out_channels = 1u;
    spec._M_in[0] = 1u;
    spec._M_kernel[0] = 1u;
    spec._M_use_bias = 0u;
    REQUIRE (status_ok (spec.validate ()));

    alignas (64) int8_t scratch[512];
    int8_t input[1];
    int8_t weight[1];
    int8_t bias[1] = { 0 };
    int8_t output[1];
    int8_t expected[1];

    struct Case
    {
        int8_t a;
        int8_t b;
        int32_t mult;
    };
    const Case cases[] = {
        { 1, 64, 16384 },   { 1, -64, 16384 },   { 3, 7, 16384 },
        { 127, 127, 16384 }, { 127, 127, 65536 }, { 100, 100, 65536 },
        { -100, 100, 65536 }, { 127, 127, 25907 }, { 64, 64, 1 },
        { -64, 64, 1 },      { 127, 127, 1048576 }, { 2, 3, 0 },
        { 127, -127, 1048576 }
    };
    for (const Case &c : cases)
        {
            auto rs = spec;
            rs._M_requant_mult_q15 = c.mult;
            input[0] = c.a;
            weight[0] = c.b;
            output[0] = 42;
            naive_forward (rs, input, weight, bias, expected);
            auto in_view = make_view (input, nullptr, 1u);
            auto w_view = make_view (weight, nullptr, 1u);
            auto b_view = make_view (bias, nullptr, 1u);
            auto out_view = make_view (output, nullptr, 1u);
            REQUIRE (conv_forward (rs, in_view, w_view, b_view, out_view,
                                   nullptr, scratch)
                     == Status::OK);
            CHECK (output[0] == expected[0]);
            CHECK (output[0] != -128);
        }

    SECTION ("unity multiplier path is untouched by the mult branch")
    {
        auto rs = spec;
        rs._M_requant_mult_q15 = k_q15_one;
        input[0] = 100;
        weight[0] = 100;
        naive_forward (rs, input, weight, bias, expected);
        auto in_view = make_view (input, nullptr, 1u);
        auto w_view = make_view (weight, nullptr, 1u);
        auto b_view = make_view (bias, nullptr, 1u);
        auto out_view = make_view (output, nullptr, 1u);
        REQUIRE (conv_forward (rs, in_view, w_view, b_view, out_view, nullptr,
                               scratch)
                 == Status::OK);
        CHECK (output[0] == expected[0]);
        CHECK (output[0] == 78); /* round(10000 / 128) */
    }
}

TEST_CASE ("conv_forward rank-2 channel-major with padding matches the reference",
           "[mllee][train][conv][forward]")
{
    ConvSpec spec;
    spec._M_batch = 2u;
    spec._M_in_channels = 2u;
    spec._M_out_channels = 4u;
    spec._M_rank = 2u;
    spec._M_in[0] = 3u;
    spec._M_in[1] = 2u;
    spec._M_kernel[0] = 3u;
    spec._M_kernel[1] = 1u;
    spec._M_pad[0] = 1;
    spec._M_pad[1] = 1;
    REQUIRE (status_ok (spec.validate ()));
    /* rows = out0 * out1 = 3 * 2 = 6 = taps (2 channels x 3 x 1) */
    REQUIRE (spec.out_length (0u) == 3u);
    REQUIRE (spec.out_length (1u) == 2u);
    REQUIRE (spec.rows () == 6u);
    REQUIRE (spec.taps () == 6u);
    REQUIRE (spec.input_elements () == 24u);
    REQUIRE (spec.weight_elements () == 24u);
    REQUIRE (spec.output_elements () == 48u);

    int8_t input[24];
    for (uint32_t i = 0; i < 24u; ++i)
        input[i] = static_cast<int8_t> ((i * 5u % 21u) - 10u);
    int8_t weight[24];
    for (uint32_t i = 0; i < 24u; ++i)
        weight[i] = static_cast<int8_t> ((i % 11u) - 5u);
    int8_t bias[4] = { 1, -1, 2, -2 };
    int8_t output[48] = { 0 };
    int8_t expected[48] = { 0 };
    alignas (64) int8_t scratch[4096];

    naive_forward (spec, input, weight, bias, expected);
    auto in_view = make_view (input, nullptr, 24u);
    auto w_view = make_view (weight, nullptr, 24u);
    auto b_view = make_view (bias, nullptr, 4u);
    auto out_view = make_view (output, nullptr, 48u);
    REQUIRE (conv_forward (spec, in_view, w_view, b_view, out_view, nullptr,
                           scratch)
             == Status::OK);
    for (uint32_t i = 0; i < 48u; ++i)
        CHECK (output[i] == expected[i]);
}

TEST_CASE ("conv_forward grouped rank-1 channel-major matches the reference",
           "[mllee][train][conv][forward]")
{
    ConvSpec spec;
    spec._M_batch = 2u;
    spec._M_in_channels = 4u;
    spec._M_out_channels = 4u;
    spec._M_groups = 2u;
    spec._M_in[0] = 8u;
    spec._M_kernel[0] = 3u;
    REQUIRE (status_ok (spec.validate ()));
    /* ipg = 2 -> taps = 6; rows = (8 - 2 - 1) + 1 = 6 */
    REQUIRE (spec.in_per_group () == 2u);
    REQUIRE (spec.out_per_group () == 2u);
    REQUIRE (spec.taps () == 6u);
    REQUIRE (spec.rows () == 6u);
    REQUIRE (spec.input_elements () == 64u);
    REQUIRE (spec.weight_elements () == 24u);
    REQUIRE (spec.output_elements () == 48u);

    int8_t input[64];
    for (uint32_t i = 0; i < 64u; ++i)
        input[i] = static_cast<int8_t> ((i * 3u % 15u) - 7u);
    int8_t weight[24];
    for (uint32_t i = 0; i < 24u; ++i)
        weight[i] = static_cast<int8_t> ((i * 7u % 19u) - 9u);
    int8_t bias[4] = { 4, -4, 2, -2 };
    int8_t output[48] = { 0 };
    int8_t expected[48] = { 0 };
    alignas (64) int8_t scratch[4096];

    naive_forward (spec, input, weight, bias, expected);
    auto in_view = make_view (input, nullptr, 64u);
    auto w_view = make_view (weight, nullptr, 24u);
    auto b_view = make_view (bias, nullptr, 4u);
    auto out_view = make_view (output, nullptr, 48u);
    REQUIRE (conv_forward (spec, in_view, w_view, b_view, out_view, nullptr,
                           scratch)
             == Status::OK);
    for (uint32_t i = 0; i < 48u; ++i)
        CHECK (output[i] == expected[i]);

    SECTION ("group isolation: zeroing group 1 weights leaves group 0 outputs")
    {
        /* out channels 0 and 1 come from in channels 0 and 1 only */
        int8_t half_weight[24];
        for (uint32_t i = 0; i < 12u; ++i)
            half_weight[i] = weight[i];
        for (uint32_t i = 12u; i < 24u; ++i)
            half_weight[i] = 0;
        int8_t out2[48] = { 0 };
        int8_t exp2[48] = { 0 };
        auto w2 = make_view (half_weight, nullptr, 24u);
        auto o2 = make_view (out2, nullptr, 48u);
        naive_forward (spec, input, half_weight, bias, exp2);
        REQUIRE (conv_forward (spec, in_view, w2, b_view, o2, nullptr, scratch)
                 == Status::OK);
        for (uint32_t i = 0; i < 48u; ++i)
            CHECK (out2[i] == exp2[i]);
        /* group 0 outputs unchanged from the full-weight run */
        for (uint32_t b = 0; b < 2u; ++b)
            for (uint32_t c = 0; c < 2u; ++c)
                for (uint32_t r = 0; r < 6u; ++r)
                    CHECK (out2[(b * 4u + c) * 6u + r]
                           == expected[(b * 4u + c) * 6u + r]);
    }
}

TEST_CASE ("conv_forward rank-2 grouped spatial-major matches the reference",
           "[mllee][train][conv][forward]"){
    ConvSpec spec;
    spec._M_batch = 2u;
    spec._M_in_channels = 2u;
    spec._M_out_channels = 6u;
    spec._M_groups = 2u;
    spec._M_rank = 2u;
    spec._M_in[0] = 5u;
    spec._M_in[1] = 2u;
    spec._M_kernel[0] = 3u;
    spec._M_kernel[1] = 1u;
    spec._M_layout = ConvLayout::SPATIAL_MAJOR;
    REQUIRE (status_ok (spec.validate ()));
    REQUIRE (spec.taps () == 3u);
    REQUIRE (spec.out_per_group () == 3u);
    /* out0 = (5-2-1)+1 = 3, out1 = (2-1)+1 = 2 -> rows = 6 */
    REQUIRE (spec.rows () == 6u);
    REQUIRE (spec.input_elements () == 40u);
    REQUIRE (spec.weight_elements () == 18u);
    REQUIRE (spec.output_elements () == 72u);

    int8_t input[40];
    for (uint32_t i = 0; i < 40u; ++i)
        input[i] = static_cast<int8_t> ((i * 7u % 13u) - 6u);
    int8_t weight[18];
    for (uint32_t i = 0; i < 18u; ++i)
        weight[i] = static_cast<int8_t> ((i % 9u) - 4u);
    int8_t bias[6] = { 1, 2, 3, -1, -2, -3 };
    int8_t output[72] = { 0 };
    int8_t expected[72] = { 0 };
    alignas (64) int8_t scratch[4096];

    naive_forward (spec, input, weight, bias, expected);
    auto in_view = make_view (input, nullptr, 40u);
    auto w_view = make_view (weight, nullptr, 18u);
    auto b_view = make_view (bias, nullptr, 6u);
    auto out_view = make_view (output, nullptr, 72u);
    REQUIRE (conv_forward (spec, in_view, w_view, b_view, out_view, nullptr,
                           scratch)
             == Status::OK);
    for (uint32_t i = 0; i < 72u; ++i)
        CHECK (output[i] == expected[i]);
}

namespace
{
    /** @brief Pool-backed GradContext wired for a direct conv_backward call. */
    struct BackwardFixture
    {
        alignas (64) uint8_t arena[16384];
        Pool pool { arena, static_cast<uint32_t> (sizeof (arena)) };
        Saved slots[4];
        Trainable operands[3];
        GradContext ctx;

        /**
         * @brief Carve the spec slot and the scratch slab; optionally a
         *        pre-activation slot the caller fills afterwards.
         */
        Status
        setup (const ConvSpec &spec, const Trainable &input,
               const Trainable &weight, const Trainable &bias,
               const uint32_t operand_count, const bool want_pre) noexcept
        {
            slots[0] = Saved{};
            slots[1] = Saved{};
            slots[2] = Saved{};
            slots[3] = Saved{};
            Slab s0;
            Status st
                = pool.carve (1u, static_cast<uint32_t> (sizeof (ConvSpec)), s0);
            if (!status_ok (st))
                return st;
            *reinterpret_cast<ConvSpec *> (pool.at (s0._M_offset)) = spec;
            slots[0]._M_offset = s0._M_offset;
            slots[0]._M_bytes = s0._M_bytes;
            slots[0]._M_count = 1u;
            if (want_pre)
                {
                    Slab s2;
                    st = pool.carve (spec.output_elements (), 1u, s2);
                    if (!status_ok (st))
                        return st;
                    std::fill_n (pool.at (s2._M_offset),
                                 spec.output_elements (), int8_t { 0 });
                    slots[2]._M_offset = s2._M_offset;
                    slots[2]._M_bytes = s2._M_bytes;
                    slots[2]._M_count = spec.output_elements ();
                }
            Slab s3;
            st = pool.carve (spec.scratch_bytes (), 1u, s3);
            if (!status_ok (st))
                return st;
            slots[3]._M_offset = s3._M_offset;
            slots[3]._M_bytes = s3._M_bytes;
            slots[3]._M_count = spec.scratch_bytes ();
            operands[0] = input;
            operands[1] = weight;
            operands[2] = bias;
            ctx._M_pool = &pool;
            ctx._M_operands = operands;
            ctx._M_operand_count = operand_count;
            ctx._M_saved = slots;
            ctx._M_saved_count = 4u;
            ctx._M_saturation_events = 0u;
            return Status::OK;
        }

        /** @brief Mutable byte view of a saved slot. */
        int8_t *
        saved_bytes_slot (const uint32_t index) noexcept
        {
            return slots[index].is_null ()
                       ? nullptr
                       : reinterpret_cast<int8_t *> (
                             pool.at (slots[index]._M_offset));
    }
    };
} // namespace

TEST_CASE ("conv_backward produces dX, dW and db matching the reference",
           "[mllee][train][conv][backward]")
{
    /* a stride-2 padded rank-1 layer: the backward needs no GEMM coincidence */
    ConvSpec spec;
    spec._M_batch = 2u;
    spec._M_in_channels = 3u;
    spec._M_out_channels = 2u;
    spec._M_in[0] = 7u;
    spec._M_kernel[0] = 3u;
    spec._M_stride[0] = 2u;
    spec._M_pad[0] = 1;
    spec._M_pad[1] = 1;
    REQUIRE (status_ok (spec.validate ()));
    REQUIRE (spec.rows () == 4u);
    REQUIRE (spec.taps () == 9u);

    int8_t input[42];
    for (uint32_t i = 0; i < 42u; ++i)
        input[i] = static_cast<int8_t> ((i % 11u) - 5u);
    int8_t weight[18];
    for (uint32_t i = 0; i < 18u; ++i)
        weight[i] = static_cast<int8_t> ((i * 3u % 13u) - 6u);
    int8_t bias[2] = { 1, -1 };
    int8_t output[16] = { 0 };
    int16_t input_grad[42];
    int16_t weight_grad[18];
    int16_t bias_grad[2];
    int16_t out_grad[16];
    for (uint32_t i = 0; i < 42u; ++i)
        input_grad[i] = 0;
    for (uint32_t i = 0; i < 18u; ++i)
        weight_grad[i] = 0;
    for (uint32_t i = 0; i < 2u; ++i)
        bias_grad[i] = 0;
    for (uint32_t i = 0; i < 16u; ++i)
        out_grad[i] = static_cast<int16_t> (int32_t (i % 5u) * 37 - 74);

    Trainable in_view;
    in_view._M_code = input;
    in_view._M_grad = input_grad;
    in_view._M_count = 42u;
    Trainable w_view;
    w_view._M_code = weight;
    w_view._M_grad = weight_grad;
    w_view._M_count = 18u;
    Trainable b_view;
    b_view._M_code = bias;
    b_view._M_grad = bias_grad;
    b_view._M_count = 2u;
    Trainable out_view;
    out_view._M_code = output;
    out_view._M_grad = out_grad;
    out_view._M_count = 16u;

    BackwardFixture fx;
    REQUIRE (fx.setup (spec, in_view, w_view, b_view, 3u, false)
             == Status::OK);
    REQUIRE (conv_backward (out_view, out_view, fx.ctx) == Status::OK);

    int16_t exp_dx[42];
    int16_t exp_dw[18];
    int16_t exp_db[2];
    std::fill_n (exp_dx, 42u, int16_t { 0 });
    std::fill_n (exp_dw, 18u, int16_t { 0 });
    std::fill_n (exp_db, 2u, int16_t { 0 });
    uint32_t sat_ref = 0u;
    naive_backward (spec, input, weight, out_grad, exp_dx, exp_dw, exp_db,
                    sat_ref);
    CHECK (sat_ref == 0u);
    CHECK (fx.ctx._M_saturation_events == 0u);
    for (uint32_t i = 0; i < 42u; ++i)
        CHECK (input_grad[i] == exp_dx[i]);
    for (uint32_t i = 0; i < 18u; ++i)
        CHECK (weight_grad[i] == exp_dw[i]);
    for (uint32_t i = 0; i < 2u; ++i)
        CHECK (bias_grad[i] == exp_db[i]);
    /* at least one gradient in each stream is non-zero, no silent no-op */
    bool nz = false;
    for (uint32_t i = 0; i < 42u; ++i)
        nz = nz || exp_dx[i] != 0;
    CHECK (nz);
    nz = false;
    for (uint32_t i = 0; i < 18u; ++i)
        nz = nz || exp_dw[i] != 0;
    CHECK (nz);
    nz = false;
    for (uint32_t i = 0; i < 2u; ++i)
        nz = nz || exp_db[i] != 0;
    CHECK (nz);
}

TEST_CASE ("conv_backward grouped rank-2 spatial-major matches the reference",
           "[mllee][train][conv][backward]")
{
    ConvSpec spec;
    spec._M_batch = 2u;
    spec._M_in_channels = 4u;
    spec._M_out_channels = 4u;
    spec._M_groups = 2u;
    spec._M_rank = 2u;
    spec._M_in[0] = 4u;
    spec._M_in[1] = 3u;
    spec._M_kernel[0] = 2u;
    spec._M_kernel[1] = 3u;
    spec._M_stride[0] = 2u;
    /* symmetric padding on both axes: axis 0 is [2, 3] -> [3, 3], axis 1 is
       [3, 3] -> [3, 3], so the block is the documented 9 rows */
    spec._M_pad[0] = 1;
    spec._M_pad[1] = 1;
    spec._M_pad[2] = 1;
    spec._M_pad[3] = 1;
    spec._M_layout = ConvLayout::SPATIAL_MAJOR;
    REQUIRE (status_ok (spec.validate ()));
    REQUIRE (spec.rows () == 9u);
    REQUIRE (spec.taps () == 12u);
    const uint32_t n_in = spec.input_elements ();
    const uint32_t n_w = spec.weight_elements ();
    const uint32_t n_out = spec.output_elements ();
    REQUIRE (n_in == 96u);
    REQUIRE (n_w == 48u);
    REQUIRE (n_out == 72u);

    std::vector<int8_t> input (n_in);
    std::vector<int8_t> weight (n_w);
    std::vector<int8_t> bias (spec._M_out_channels);
    std::vector<int8_t> output (n_out, 0);
    std::vector<int16_t> dx (n_in, 0);
    std::vector<int16_t> dw (n_w, 0);
    std::vector<int16_t> db (spec._M_out_channels, 0);
    std::vector<int16_t> dy (n_out);
    for (uint32_t i = 0; i < n_in; ++i)
        input[i] = static_cast<int8_t> ((i * 5u % 17u) - 8u);
    for (uint32_t i = 0; i < n_w; ++i)
        weight[i] = static_cast<int8_t> ((i % 9u) - 4u);
    for (uint32_t i = 0; i < n_out; ++i)
        dy[i] = static_cast<int16_t> ((i * 61 % 121) - 60);
    for (uint32_t i = 0; i < spec._M_out_channels; ++i)
        bias[i] = static_cast<int8_t> (int32_t (i) - 2);

    Trainable in_view;
    in_view._M_code = input.data ();
    in_view._M_grad = dx.data ();
    in_view._M_count = n_in;
    Trainable w_view;
    w_view._M_code = weight.data ();
    w_view._M_grad = dw.data ();
    w_view._M_count = n_w;
    Trainable b_view;
    b_view._M_code = bias.data ();
    b_view._M_grad = db.data ();
    b_view._M_count = spec._M_out_channels;
    Trainable out_view;
    out_view._M_code = output.data ();
    out_view._M_grad = dy.data ();
    out_view._M_count = n_out;

    BackwardFixture fx;
    REQUIRE (fx.setup (spec, in_view, w_view, b_view, 3u, false)
             == Status::OK);
    REQUIRE (conv_backward (out_view, out_view, fx.ctx) == Status::OK);

    std::vector<int16_t> exp_dx (n_in, 0);
    std::vector<int16_t> exp_dw (n_w, 0);
    std::vector<int16_t> exp_db (spec._M_out_channels, 0);
    uint32_t sat_ref = 0u;
    naive_backward (spec, input.data (), weight.data (), dy.data (),
                    exp_dx.data (), exp_dw.data (), exp_db.data (), sat_ref);
    CHECK (fx.ctx._M_saturation_events == sat_ref);
    for (uint32_t i = 0; i < n_in; ++i)
        CHECK (dx[i] == exp_dx[i]);
    for (uint32_t i = 0; i < n_w; ++i)
        CHECK (dw[i] == exp_dw[i]);
    for (uint32_t i = 0; i < spec._M_out_channels; ++i)
        CHECK (db[i] == exp_db[i]);
}

TEST_CASE ("conv_backward gradient accumulation saturates and counts events",
           "[mllee][train][conv][backward]")
{
    /* two items accumulating into one weight gradient force a clamp */
    ConvSpec spec;
    spec._M_batch = 2u;
    spec._M_in_channels = 1u;
    spec._M_out_channels = 1u;
    spec._M_in[0] = 3u;
    spec._M_kernel[0] = 3u;
    REQUIRE (status_ok (spec.validate ()));
    REQUIRE (spec.rows () == 1u);
    REQUIRE (spec.taps () == 3u);

    /* batch 2 x 3 input samples: each item feeds the same three taps */
    int8_t input[6] = { 127, 127, 127, 127, 127, 127 };
    int8_t weight[3] = { 100, 100, 100 };
    int8_t bias[1] = { 0 };
    int8_t output[2] = { 0 };
    /* both items carry a near-unit gradient: dW per tap = 2 * 32767 -> clamps */
    int16_t dy[2] = { 32767, 32767 };
    int16_t dx[6] = { 0, 0, 0, 0, 0, 0 };
    int16_t dw[3] = { 0, 0, 0 };
    int16_t db[1] = { 0 };

    Trainable in_view;
    in_view._M_code = input;
    in_view._M_grad = dx;
    in_view._M_count = 6u;
    Trainable w_view;
    w_view._M_code = weight;
    w_view._M_grad = dw;
    w_view._M_count = 3u;
    Trainable b_view;
    b_view._M_code = bias;
    b_view._M_grad = db;
    b_view._M_count = 1u;
    Trainable out_view;
    out_view._M_code = output;
    out_view._M_grad = dy;
    out_view._M_count = 2u;

    BackwardFixture fx;
    REQUIRE (fx.setup (spec, in_view, w_view, b_view, 3u, false)
             == Status::OK);
    REQUIRE (conv_backward (out_view, out_view, fx.ctx) == Status::OK);

    int16_t exp_dx[6] = { 0, 0, 0, 0, 0, 0 };
    int16_t exp_dw[3] = { 0, 0, 0 };
    int16_t exp_db[1] = { 0 };
    uint32_t sat_ref = 0u;
    naive_backward (spec, input, weight, dy, exp_dx, exp_dw, exp_db, sat_ref);
    /* the second item's saturated-add clamps once per weight tap and once for
       the bias accumulator; the mirrored reference counts the same events */
    CHECK (sat_ref > 0u);
    CHECK (fx.ctx._M_saturation_events == sat_ref);
    for (uint32_t i = 0; i < 3u; ++i)
        {
            CHECK (dw[i] == exp_dw[i]);
            CHECK (dw[i] == 32767);
        }
    for (uint32_t i = 0; i < 6u; ++i)
        CHECK (dx[i] == exp_dx[i]);
    CHECK (db[0] == exp_db[0]);
    CHECK (db[0] == 32767);
}

TEST_CASE ("conv_backward routes gradients through the saved pre-activation",
           "[mllee][train][conv][backward]")
{
    ConvSpec spec;
    spec._M_batch = 1u;
    spec._M_in_channels = 2u;
    spec._M_out_channels = 2u;
    spec._M_in[0] = 2u;
    spec._M_kernel[0] = 1u;
    spec._M_act = Act::RELU;
    REQUIRE (status_ok (spec.validate ()));
    REQUIRE (spec.rows () == 2u);
    REQUIRE (spec.taps () == 2u);
    REQUIRE (spec.output_elements () == 4u);

    int8_t input[4] = { 50, -60, 70, -80 };
    int8_t weight[4] = { 90, -30, 20, 45 };
    int8_t bias[2] = { 0, 0 };
    int8_t output[4] = { 11, 22, 33, 44 }; /* deliberately nonsense codes */
    int16_t dy[4] = { 1000, 2000, 3000, 4000 };
    int16_t dx[4] = { 0, 0, 0, 0 };
    int16_t dw[4] = { 0, 0, 0, 0 };
    int16_t db[2] = { 0, 0 };

    Trainable in_view;
    in_view._M_code = input;
    in_view._M_grad = dx;
    in_view._M_count = 4u;
    Trainable w_view;
    w_view._M_code = weight;
    w_view._M_grad = dw;
    w_view._M_count = 4u;
    Trainable b_view;
    b_view._M_code = bias;
    b_view._M_grad = db;
    b_view._M_count = 2u;
    Trainable out_view;
    out_view._M_code = output;
    out_view._M_grad = dy;
    out_view._M_count = 4u;

    BackwardFixture fx;
    REQUIRE (fx.setup (spec, in_view, w_view, b_view, 3u, true)
             == Status::OK);
    /* the saved pre-activation, not the (here nonsense) output, gates the flow:
       mask out elements 1 and 3 by storing negative pre-codes there */
    int8_t *const pre = fx.saved_bytes_slot (2u);
    REQUIRE (pre != nullptr);
    const int8_t saved_pre[4] = { 7, -1, 5, -2 };
    for (uint32_t i = 0; i < 4u; ++i)
        pre[i] = saved_pre[i];

    REQUIRE (conv_backward (out_view, out_view, fx.ctx) == Status::OK);

    int16_t routed[4];
    for (uint32_t i = 0; i < 4u; ++i)
        routed[i] = saved_pre[i] >= 0 ? dy[i] : 0;
    int16_t exp_dx[4] = { 0, 0, 0, 0 };
    int16_t exp_dw[4] = { 0, 0, 0, 0 };
    int16_t exp_db[2] = { 0, 0 };
    uint32_t sat_ref = 0u;
    naive_backward (spec, input, weight, routed, exp_dx, exp_dw, exp_db,
                    sat_ref);
    CHECK (fx.ctx._M_saturation_events == sat_ref);
    /* proof that the pre-activation was read: the masked lanes differ */
    CHECK (exp_db[0] != exp_db[1]);
    for (uint32_t i = 0; i < 4u; ++i)
        {
            CHECK (dx[i] == exp_dx[i]);
            CHECK (dw[i] == exp_dw[i]);
        }
    for (uint32_t i = 0; i < 2u; ++i)
        CHECK (db[i] == exp_db[i]);
}

TEST_CASE ("conv_backward routes a saturating hard-sigmoid gradient",
           "[mllee][train][conv][backward]")
{
    ConvSpec spec;
    spec._M_batch = 1u;
    spec._M_in_channels = 2u;
    spec._M_out_channels = 2u;
    spec._M_in[0] = 2u;
    spec._M_kernel[0] = 1u;
    spec._M_act = Act::HARDSIGMOID;
    REQUIRE (status_ok (spec.validate ()));

    int8_t input[4] = { 20, -40, 60, -80 };
    int8_t weight[4] = { 50, 60, -70, 80 };
    int8_t bias[2] = { 0, 0 };
    int8_t output[4] = { 0, 0, 0, 0 };
    int16_t dy[4] = { 2000, 4000, -6000, 8000 };
    int16_t dx[4] = { 0, 0, 0, 0 };
    int16_t dw[4] = { 0, 0, 0, 0 };
    int16_t db[2] = { 0, 0 };

    Trainable in_view;
    in_view._M_code = input;
    in_view._M_grad = dx;
    in_view._M_count = 4u;
    Trainable w_view;
    w_view._M_code = weight;
    w_view._M_grad = dw;
    w_view._M_count = 4u;
    Trainable b_view;
    b_view._M_code = bias;
    b_view._M_grad = db;
    b_view._M_count = 2u;
    Trainable out_view;
    out_view._M_code = output;
    out_view._M_grad = dy;
    out_view._M_count = 4u;

    BackwardFixture fx;
    REQUIRE (fx.setup (spec, in_view, w_view, b_view, 3u, true)
             == Status::OK);
    const int8_t saved_pre[4] = { 127, -127, 0, 1 };
    int8_t *const pre = fx.saved_bytes_slot (2u);
    REQUIRE (pre != nullptr);
    for (uint32_t i = 0; i < 4u; ++i)
        pre[i] = saved_pre[i];

    REQUIRE (conv_backward (out_view, out_view, fx.ctx) == Status::OK);

    /* hard-sigmoid derivative is 1/2 only strictly inside (0, 1), and the
       pre-code 127 maps to exactly the top of the range -> masked */
    int16_t routed[4];
    for (uint32_t i = 0; i < 4u; ++i)
        {
            const int32_t x = static_cast<int32_t> (saved_pre[i]) * k_code_step;
            int32_t y = (x + k_q15_one) >> 1;
            y = y < 0 ? 0 : (y > k_q15_max ? k_q15_max : y);
            const int32_t deriv = (y > 0 && y < k_q15_max) ? 16384 : 0;
            routed[i] = static_cast<int16_t> (
                q15_clamp_ref (half_up_shift (static_cast<int64_t> (dy[i])
                                              * deriv,
                                              15u)));
        }
    CHECK (routed[0] == 0);   /* pre 127 -> y = 32766? in (0, max): kept */
    int16_t exp_dx[4] = { 0, 0, 0, 0 };
    int16_t exp_dw[4] = { 0, 0, 0, 0 };
    int16_t exp_db[2] = { 0, 0 };
    uint32_t sat_ref = 0u;
    naive_backward (spec, input, weight, routed, exp_dx, exp_dw, exp_db,
                    sat_ref);
    CHECK (fx.ctx._M_saturation_events == sat_ref);
    for (uint32_t i = 0; i < 4u; ++i)
        {
            CHECK (dx[i] == exp_dx[i]);
            CHECK (dw[i] == exp_dw[i]);
        }
    for (uint32_t i = 0; i < 2u; ++i)
        CHECK (db[i] == exp_db[i]);
}

TEST_CASE ("conv_backward rejects malformed contexts and counts",
           "[mllee][train][conv][backward]")
{
    ConvSpec spec;
    spec._M_batch = 1u;
    spec._M_in_channels = 2u;
    spec._M_out_channels = 2u;
    spec._M_in[0] = 2u;
    spec._M_kernel[0] = 1u;
    REQUIRE (status_ok (spec.validate ()));

    int8_t input[4] = { 1, 2, 3, 4 };
    int8_t weight[4] = { 1, 1, 1, 1 };
    int8_t bias[2] = { 0, 0 };
    int8_t output[4] = { 0, 0, 0, 0 };
    int16_t dy[4] = { 10, 20, 30, 40 };
    int16_t dzeros[4] = { 0, 0, 0, 0 };

    const auto view_in = [&] () noexcept
    {
        Trainable t;
        t._M_code = input;
        t._M_count = 4u;
        return t;
    };
    const auto view_w = [&] () noexcept
    {
        Trainable t;
        t._M_code = weight;
        t._M_count = 4u;
        return t;
    };
    const auto view_b = [&] () noexcept
    {
        Trainable t;
        t._M_code = bias;
        t._M_count = 2u;
        return t;
    };
    auto view_out = [&] () noexcept
    {
        Trainable t;
        t._M_code = output;
        t._M_grad = dy;
        t._M_count = 4u;
        return t;
    };

    SECTION ("empty context has no saved spec")
    {
        GradContext ctx;
        const auto out = view_out ();
        CHECK (conv_backward (out, out, ctx) == Status::BAD_LAYOUT);
    }
    SECTION ("fewer than two operands")
    {
        BackwardFixture fx;
        REQUIRE (fx.setup (spec, view_in (), view_w (), view_b (), 1u, false)
                 == Status::OK);
        const auto out = view_out ();
        CHECK (conv_backward (out, out, fx.ctx) == Status::BAD_LAYOUT);
    }
    SECTION ("missing gradient lanes on the output")
    {
        BackwardFixture fx;
        REQUIRE (fx.setup (spec, view_in (), view_w (), view_b (), 3u, false)
                 == Status::OK);
        Trainable out;
        out._M_code = output;
        out._M_grad = nullptr;
        out._M_count = 4u;
        CHECK (conv_backward (out, out, fx.ctx) == Status::BAD_LAYOUT);
    }
    SECTION ("count mismatches against the spec")
    {
        BackwardFixture fx;
        REQUIRE (fx.setup (spec, view_in (), view_w (), view_b (), 3u, false)
                 == Status::OK);
        Trainable bad_in;
        bad_in._M_code = input;
        bad_in._M_count = 3u;
        fx.operands[0] = bad_in;
        auto out = view_out ();
        CHECK (conv_backward (out, out, fx.ctx) == Status::BAD_LAYOUT);

        Trainable bad_w;
        bad_w._M_code = weight;
        bad_w._M_count = 5u;
        fx.operands[0] = view_in ();
        fx.operands[1] = bad_w;
        CHECK (conv_backward (out, out, fx.ctx) == Status::BAD_LAYOUT);

        out._M_count = 2u;
        fx.operands[1] = view_w ();
        CHECK (conv_backward (out, out, fx.ctx) == Status::BAD_LAYOUT);
        out._M_grad = dzeros; /* keep a valid pointer for the next check */
    }
    SECTION ("missing scratch slot")
    {
        BackwardFixture fx;
        REQUIRE (fx.setup (spec, view_in (), view_w (), view_b (), 3u, false)
                 == Status::OK);
        fx.slots[3] = Saved{};
        const auto out = view_out ();
        CHECK (conv_backward (out, out, fx.ctx) == Status::BAD_LAYOUT);
    }
    SECTION ("activation without a saved pre-activation")
    {
        auto act_spec = spec;
        act_spec._M_act = Act::RELU;
        BackwardFixture fx;
        REQUIRE (fx.setup (act_spec, view_in (), view_w (), view_b (), 3u,
                           false)
                 == Status::OK);
        const auto out = view_out ();
        CHECK (conv_backward (out, out, fx.ctx) == Status::BAD_LAYOUT);
    }
    SECTION ("invalid saved spec is rejected by its own validate")
    {
        auto broken = spec;
        broken._M_kernel[0] = 0u;
        BackwardFixture fx;
        REQUIRE (fx.setup (broken, view_in (), view_w (), view_b (), 3u, false)
                 == Status::OK);
        const auto out = view_out ();
        CHECK (conv_backward (out, out, fx.ctx) == Status::BAD_LAYOUT);
    }
    SECTION ("frozen input and weight simply skip their halves")
    {
        Trainable in;
        in._M_code = input;
        in._M_grad = nullptr;
        in._M_count = 4u;
        Trainable w;
        w._M_code = weight;
        w._M_grad = nullptr;
        w._M_count = 4u;
        BackwardFixture fx;
        REQUIRE (fx.setup (spec, in, w, view_b (), 3u, false) == Status::OK);
        const auto out = view_out ();
        CHECK (conv_backward (out, out, fx.ctx) == Status::OK);
        CHECK (fx.ctx._M_saturation_events == 0u);
        /* only the bias collected a gradient */
        int16_t db[2] = { 0, 0 };
        uint32_t sat_ref = 0u;
        naive_backward (spec, input, weight, dy, nullptr, nullptr, db,
                        sat_ref);
        CHECK (sat_ref == 0u);
    }
}
