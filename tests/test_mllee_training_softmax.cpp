/* Tests for the training softmax: the row kernel, the axis permutation the
 * scratch layout has to survive, masking, temperature, the log output and the
 * Jacobian the backward walks.
 *
 * The numeric expectations come from <cmath>, never from the fixed-point code,
 * and the strided cases are checked against the contiguous run of the same
 * distribution: a row that aliases another row's scratch keeps its arithmetic
 * identical, so only a permutation comparison notices it. */

#include "mllee/train_softmax.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <cmath>

using namespace zzs::mllee;
using namespace zzs::mllee::train;
using Catch::Approx;

namespace
{
constexpr uint32_t _row_max = 8u;
constexpr double _q15 = 32768.0;

/** @brief The float value of one widened Q15 logit on the tensor's own scale. */
double
real_of (const int8_t code) noexcept
{
    return softmax_widen (code, _S_q15_one) / _q15;
}

/** @brief softmax over the row the core would build, in double. */
void
ref_row (const int8_t *const codes, const uint32_t length,
         const double temperature, double *const out)
{
    double total = 0.0;
    for (uint32_t i = 0; i < length; ++i)
        {
            out[i] = std::exp (real_of (codes[i]) / temperature);
            total += out[i];
        }
    for (uint32_t i = 0; i < length; ++i)
        out[i] /= total;
}

/** @brief The same row with the masked positions removed from the sum. */
void
ref_row_masked (const int8_t *const codes, const uint32_t length,
                const uint8_t *const keep, double *const out)
{
    double total = 0.0;
    for (uint32_t i = 0; i < length; ++i)
        {
            out[i] = keep[i] != 0u ? std::exp (real_of (codes[i])) : 0.0;
            total += out[i];
        }
    if (total == 0.0)
        return;
    for (uint32_t i = 0; i < length; ++i)
        out[i] /= total;
}

SoftmaxSpec
make_spec (const uint32_t rows, const uint32_t length, const uint32_t inner)
{
    SoftmaxSpec spec;
    spec._M_rows = static_cast<uint16_t> (rows);
    spec._M_length = static_cast<uint16_t> (length);
    spec._M_inner = static_cast<uint16_t> (inner);
    spec._M_rank = 2u;
    spec._M_axis = 1u;
    spec._M_op_id = 0x5100u;
    return spec;
}

/** @brief Codes with a positive maximum in every 3-, 4- and 8-element window, so
 *         no lane rails. */
const int8_t _codes[12]
    = { 40, -20, 7, 90, -110, 3, 55, -1, 22, -77, 13, -45 };

struct Harness
{
    static constexpr uint32_t _capacity = 2048u;

    AlignedArena<_capacity> _M_arena;
    Pool _M_pool;
    alignas (8) Node _M_nodes[2];
    Tape _M_tape;
    alignas (8) int8_t _M_code[32];
    alignas (8) int16_t _M_grad[32];
    alignas (8) int8_t _M_out_code[32];
    alignas (8) int16_t _M_out_grad[32];

    Harness () : _M_pool (_M_arena.pool ()), _M_tape (_M_nodes, 2u, _M_pool)
    {
        for (uint32_t i = 0; i < 32u; ++i)
            {
                _M_code[i] = _codes[i % 12u];
                _M_grad[i] = 0;
                _M_out_code[i] = 0;
                _M_out_grad[i] = 0;
            }
        REQUIRE (status_ok (_M_tape.begin ()));
    }

    Trainable
    logits (const uint32_t elements)
    {
        return Trainable::make (_M_code, _M_grad, Shape::vector (
                                                    static_cast<uint16_t> (elements)),
                                RequiresGrad);
    }

    Trainable
    output (const uint32_t elements)
    {
        return Trainable::make (_M_out_code, _M_out_grad,
                                Shape::vector (static_cast<uint16_t> (elements)),
                                RequiresGrad);
    }

    /** @brief The recorded node's context, exactly as the tape walk builds it. */
    GradContext
    context ()
    {
        Node *const node = _M_tape.node (0u);
        REQUIRE (node != nullptr);
        GradContext ctx;
        ctx._M_pool = &_M_pool;
        ctx._M_operands = node->_M_operands;
        ctx._M_operand_count = node->_M_operand_count;
        ctx._M_saved = node->_M_saved;
        ctx._M_saved_count = node->_M_saved_count;
        ctx._M_loss_scale_q15 = _S_q15_one;
        return ctx;
    }
};
} // namespace

TEST_CASE ("softmax spec geometry and pool accounting",
           "[mllee][train][softmax]")
{
    SECTION ("the decomposition matches the tensor it was taken from")
    {
        const SoftmaxSpec last = make_spec (2u, 4u, 1u);
        CHECK (last.total () == 8u);
        CHECK (last.groups () == 2u);
        CHECK (last.scratch_bytes () == 8u * 4u);  /* two int32 rows, no stage */
        CHECK_FALSE (last.is_masked ());
        CHECK_FALSE (last.uses_temperature ());

        const SoftmaxSpec middle = make_spec (2u, 3u, 2u);
        CHECK (middle.total () == 12u);
        CHECK (middle.groups () == 4u);
        /* a strided axis pays its own staging row: the forward's last pass still
           reads the exp-term row while it writes the probabilities.  The row is
           lane-aligned, so 2 L bytes round up to 8 for an odd length. */
        CHECK (middle.scratch_bytes () == 24u + align_up (6u, _S_pool_alignment));
        CHECK (middle.bytes ()
               == 20u + align_up (2u * 12u, _S_pool_alignment)
                      + middle.scratch_bytes ());
        SoftmaxSpec strided_masked = make_spec (2u, 3u, 2u);
        strided_masked._M_masked = 1u;
        /* the gathered keep row is one more aligned L bytes, and the mask stream
           itself is E bytes of the node */
        CHECK (strided_masked.scratch_bytes ()
               == 24u + align_up (6u, _S_pool_alignment)
                      + align_up (3u, _S_pool_alignment));
        CHECK (strided_masked.bytes ()
               == 20u + align_up (2u * 12u, _S_pool_alignment)
                      + align_up (12u, _S_pool_alignment)
                      + strided_masked.scratch_bytes ());
        SoftmaxSpec flat_masked = make_spec (2u, 4u, 1u);
        flat_masked._M_masked = 1u;
        /* a contiguous axis reads the mask in place: no staging, no gather */
        CHECK (flat_masked.scratch_bytes () == 8u * 4u);
    }

    SECTION ("an element index is row-major over (row, position, inner)")
    {
        const SoftmaxSpec spec = make_spec (2u, 3u, 2u);
        for (uint32_t group = 0u; group < spec.groups (); ++group)
            for (uint32_t position = 0u; position < spec._M_length; ++position)
                {
                    const uint32_t row = group / spec._M_inner;
                    const uint32_t offset = group % spec._M_inner;
                    CHECK (softmax_element (spec, group, position)
                           == (row * spec._M_length + position) * spec._M_inner
                                  + offset);
                }
        /* every element is hit exactly once */
        uint32_t seen[12] = { 0 };
        for (uint32_t group = 0u; group < spec.groups (); ++group)
            for (uint32_t position = 0u; position < spec._M_length; ++position)
                ++seen[softmax_element (spec, group, position)];
        for (uint32_t i = 0u; i < 12u; ++i)
            CHECK (seen[i] == 1u);
    }

    SECTION ("a bad geometry is rejected before any write")
    {
        SoftmaxSpec spec = make_spec (2u, 4u, 1u);
        CHECK (status_ok (spec.validate ()));
        spec._M_rows = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = make_spec (2u, 4u, 1u);
        spec._M_length = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = make_spec (2u, 4u, 1u);
        spec._M_inner = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = make_spec (2u, 4u, 1u);
        spec._M_axis = spec._M_rank;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = make_spec (2u, 4u, 1u);
        spec._M_temperature_q15 = 0;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = make_spec (2u, 4u, 1u);
        spec._M_masked = 2u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = make_spec (2u, 4u, 1u);
        spec._M_log = 2u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }

    SECTION ("the saved spec is five pool words and the row state is four")
    {
        static_assert (sizeof (SoftmaxSpec) == 20u, "spec size");
        static_assert (sizeof (SoftmaxRow) == 16u, "row state size");
        CHECK (align_up (static_cast<uint32_t> (sizeof (SoftmaxRow)),
                         _S_pool_alignment)
               == 16u);
    }
}

TEST_CASE ("softmax atoms widen, exp and log inside the documented window",
           "[mllee][train][softmax]")
{
    SECTION ("widening is code * scale / 127, clamped to the exp window")
    {
        CHECK (softmax_widen (0, _S_q15_one) == 0);
        CHECK (softmax_widen (1, _S_q15_one) == 258);
        /* the division truncates, so the two rails are not symmetric */
        CHECK (softmax_widen (127, _S_q15_one) == 32768);
        CHECK (softmax_widen (-127, _S_q15_one) == -32767);
        CHECK (softmax_widen (126, _S_q15_one) == 32510);
        /* the scale is the logit range: 4.0 at code 127, and +/-8.0 is where the
           exp table stops, which is the clamp the widening reports instead of */
        CHECK (softmax_widen (127, 4 * _S_q15_one) == 4 * _S_q15_one);
        CHECK (softmax_widen (-127, 4 * _S_q15_one) == 4 * -_S_q15_one + 1);
        CHECK (softmax_widen (127, 8 * _S_q15_one) == _S_exp_clamp_q15);
        CHECK (softmax_widen (127, 16 * _S_q15_one) == _S_exp_clamp_q15);
        CHECK (softmax_widen (-127, 16 * _S_q15_one) == -_S_exp_clamp_q15);
    }

    SECTION ("the temperature divides the logit and keeps the window")
    {
        CHECK (softmax_div_temperature (20000, _S_q15_one) == 20000);
        CHECK (softmax_div_temperature (20000, 2 * _S_q15_one) == 10000);
        CHECK (softmax_div_temperature (20000, _S_q15_one / 2) == 40000);
        CHECK (softmax_div_temperature (200000, _S_q15_one / 2)
               == _S_exp_clamp_q15);
        CHECK (softmax_div_temperature (-200000, _S_q15_one / 2)
               == -_S_exp_clamp_q15);
    }

    SECTION ("a gap past the window is an exact zero, not the table floor")
    {
        /* gap is a Q15 logit difference, so 12345 is exp (-0.3767) */
        CHECK (softmax_exp_term (0) == 32767);
        CHECK (static_cast<double> (softmax_exp_term (1))
               == Approx (_q15 * std::exp (-1.0 / _q15)).margin (2.0));
        CHECK (static_cast<double> (softmax_exp_term (12345)) / _q15
               == Approx (std::exp (-12345.0 / _q15)).margin (0.0005));
        CHECK (softmax_exp_term (_S_exp_clamp_q15) >= 4);
        CHECK (softmax_exp_term (_S_exp_clamp_q15 + 1) == 0);
        CHECK (softmax_exp_term (2 * _S_exp_clamp_q15) == 0);
        /* monotone decreasing in a positive gap, with no stall value */
        int32_t previous = _S_q15_max + 1;
        for (int32_t gap = 0; gap <= _S_exp_clamp_q15; gap += 4096)
            {
                const int32_t term = softmax_exp_term (gap);
                CHECK (term <= previous);
                previous = term;
            }
    }

    SECTION ("the Q7 log path agrees with the natural log")
    {
        for (int32_t q15 = 1; q15 <= 32767; q15 += 511)
            {
                const double expected = std::log (q15 / _q15) * 128.0;
                CHECK (static_cast<double> (q7_log_q15 (q15))
                       == Approx (expected).margin (2.0));
            }
        const int32_t logits[4]
            = { softmax_widen (40, _S_q15_one), softmax_widen (-20, _S_q15_one),
                softmax_widen (7, _S_q15_one), softmax_widen (90, _S_q15_one) };
        const double sum = std::exp (logits[0] / _q15) + std::exp (logits[1] / _q15)
                           + std::exp (logits[2] / _q15) + std::exp (logits[3] / _q15);
        const double expected_lse = std::log (sum) * 128.0;
        CHECK (static_cast<double> (log_sum_exp_q7 (logits, 4u, nullptr))
               == Approx (expected_lse).margin (3.0));
        const uint8_t keep[4] = { 1, 0, 1, 1 };
        const double masked = std::exp (logits[0] / _q15) + std::exp (logits[2] / _q15)
                              + std::exp (logits[3] / _q15);
        CHECK (static_cast<double> (log_sum_exp_q7 (logits, 4u, keep))
               == Approx (std::log (masked) * 128.0).margin (3.0));
    }

    SECTION ("a fully masked row is the -inf sentinel, not a division by zero")
    {
        const int32_t logits[3] = { 100, 200, 300 };
        const uint8_t keep[3] = { 0, 0, 0 };
        CHECK (log_sum_exp_q7 (logits, 3u, keep) == _S_softmax_neg_inf_q7);
    }

    SECTION ("a log probability becomes an inference code on the -127 rail")
    {
        CHECK (log_softmax_code_q7 (0) == 0);       /* p == 1 */
        CHECK (log_softmax_code_q7 (-128) == -16);  /* one nat: 16256 / 1024 */
        CHECK (log_softmax_code_q7 (-1024) == -127);
        CHECK (log_softmax_code_q7 (-4000) == -127); /* below the rail */
        int8_t codes[4];
        const int16_t lanes[4] = { 0, -256, -512, -32768 };
        log_softmax_codes (lanes, 4u, codes);
        CHECK (codes[0] == 0);
        CHECK (codes[1] == -32);
        CHECK (codes[2] == -64);
        CHECK (codes[3] == _S_softmax_log_rail);
    }

    SECTION ("the mask stream is element-major and only cares about the sign")
    {
        const int8_t mask[6] = { 0, -1, 5, -127, 127, -3 };
        uint8_t keep[6];
        softmax_build_mask (mask, 6u, keep);
        const uint8_t expected[6] = { 1, 0, 1, 0, 1, 0 };
        for (uint32_t i = 0; i < 6u; ++i)
            CHECK (keep[i] == expected[i]);
        const int16_t wide[6] = { 0, -1, 5, -32768, 32767, -3 };
        softmax_build_mask_q15 (wide, 6u, keep);
        for (uint32_t i = 0; i < 6u; ++i)
            CHECK (keep[i] == expected[i]);
    }
}

TEST_CASE ("softmax forward normalizes every row of the axis",
           "[mllee][train][softmax]")
{
    alignas (8) int8_t code[8];
    alignas (8) int16_t grad[8];
    alignas (8) int8_t out_code[8];
    alignas (8) int16_t out_grad[8];
    alignas (8) int16_t prob[8];
    alignas (8) uint8_t scratch[64];

    for (uint32_t i = 0; i < 8u; ++i)
        {
            code[i] = _codes[i];
            grad[i] = 0;
            out_code[i] = 0;
            out_grad[i] = 0;
            prob[i] = -1;
        }

    const SoftmaxSpec spec = make_spec (2u, 4u, 1u);
    const Trainable logits
        = Trainable::make (code, grad, Shape::vector (8), RequiresGrad);
    Trainable output
        = Trainable::make (out_code, out_grad, Shape::vector (8), RequiresGrad);

    SECTION ("probabilities agree with the float softmax and sum to one")
    {
        REQUIRE (status_ok (softmax_forward (spec, logits, nullptr, output,
                                             prob, scratch)));
        double reference[_row_max];
        for (uint32_t group = 0u; group < spec.groups (); ++group)
            {
                ref_row (code + group * 4u, 4u, 1.0, reference);
                uint32_t sum = 0u;
                for (uint32_t position = 0u; position < 4u; ++position)
                    {
                        const int32_t lane = prob[group * 4u + position];
                        CHECK (static_cast<double> (lane) / _q15
                               == Approx (reference[position]).margin (0.001));
                        sum += static_cast<uint32_t> (lane);
                    }
                /* the truncation residual is absorbed on the argmax lane */
                CHECK (sum == 32768u);
            }
    }

    SECTION ("the output codes are the probabilities requantized")
    {
        REQUIRE (status_ok (softmax_forward (spec, logits, nullptr, output,
                                             prob, scratch)));
        for (uint32_t i = 0; i < 8u; ++i)
            CHECK (out_code[i] == q15_to_code (prob[i]));
        CHECK (output.requires_grad ());
        for (uint32_t i = 0; i < 8u; ++i)
            CHECK (out_grad[i] == 0);   /* the forward zeroes the output gradient */
    }

    SECTION ("a larger temperature flattens the row")
    {
        SoftmaxSpec warm = make_spec (2u, 4u, 1u);
        warm._M_temperature_q15 = 2 * _S_q15_one;
        REQUIRE (status_ok (softmax_forward (warm, logits, nullptr, output,
                                             prob, scratch)));
        double reference[_row_max];
        ref_row (code, 4u, 2.0, reference);
        for (uint32_t position = 0u; position < 4u; ++position)
            CHECK (static_cast<double> (prob[position]) / _q15
                   == Approx (reference[position]).margin (0.001));
        /* flatter than T == 1: the top lane moved toward the mean */
        CHECK (prob[3] < 32768 / 2);
    }

    SECTION ("a masked position leaves the row without a trace")
    {
        SoftmaxSpec masked_spec = make_spec (2u, 4u, 1u);
        masked_spec._M_masked = 1u;
        const uint8_t keep[8] = { 1, 0, 1, 1, 0, 0, 0, 1 };
        REQUIRE (status_ok (softmax_forward (masked_spec, logits, keep, output,
                                             prob, scratch)));
        double reference[_row_max];
        ref_row_masked (code, 4u, keep, reference);
        CHECK (prob[1] == 0);
        for (uint32_t position = 0u; position < 4u; ++position)
            CHECK (static_cast<double> (prob[position]) / _q15
                   == Approx (reference[position]).margin (0.001));
        /* the second row keeps only its last lane, so it is certain */
        CHECK (prob[7] == 32767);
        CHECK (prob[4] == 0);
        CHECK (prob[5] == 0);
        CHECK (prob[6] == 0);
        /* a mask without the flag, or the flag without a mask, is a wiring bug */
        CHECK (softmax_forward (spec, logits, keep, output, prob, scratch)
               == Status::BAD_LAYOUT);
        CHECK (softmax_forward (masked_spec, logits, nullptr, output, prob,
                                scratch)
               == Status::BAD_LAYOUT);
    }

    SECTION ("a fully masked row is all zeros, not a division by zero")
    {
        SoftmaxSpec masked_spec = make_spec (1u, 3u, 1u);
        masked_spec._M_masked = 1u;
        alignas (8) int8_t small_code[3];
        alignas (8) int16_t small_grad[3];
        alignas (8) int8_t small_out[3];
        alignas (8) int16_t small_out_grad[3];
        alignas (8) int16_t small_prob[3];
        alignas (8) uint8_t small_scratch[32];
        const uint8_t none[3] = { 0, 0, 0 };
        for (uint32_t i = 0; i < 3u; ++i)
            {
                small_code[i] = _codes[i];
                small_grad[i] = 0;
                small_out[i] = 0;
                small_out_grad[i] = 0;
                small_prob[i] = -1;
            }
        const Trainable small_logits
            = Trainable::make (small_code, small_grad, Shape::vector (3),
                               RequiresGrad);
        Trainable small_out_view
            = Trainable::make (small_out, small_out_grad, Shape::vector (3),
                               RequiresGrad);
        REQUIRE (status_ok (softmax_forward (masked_spec, small_logits, none,
                                             small_out_view, small_prob,
                                             small_scratch)));
        for (uint32_t i = 0; i < 3u; ++i)
            {
                CHECK (small_prob[i] == 0);
                CHECK (small_out[i] == 0);
            }
    }

    SECTION ("the log output is the log of the same distribution")
    {
        SoftmaxSpec log_spec = make_spec (2u, 4u, 1u);
        log_spec._M_log = 1u;
        REQUIRE (status_ok (softmax_forward (log_spec, logits, nullptr, output,
                                             prob, scratch)));
        double reference[_row_max];
        ref_row (code, 4u, 1.0, reference);
        for (uint32_t position = 0u; position < 4u; ++position)
            {
                const double expected
                    = 127.0 * std::log (reference[position]) / 8.0;
                CHECK (static_cast<double> (out_code[position])
                       == Approx (expected).margin (3.0));
                CHECK (out_code[position] <= 0);
            }
        /* the probabilities are still saved for the backward */
        uint32_t sum = 0u;
        for (uint32_t position = 0u; position < 4u; ++position)
            sum += static_cast<uint32_t> (prob[position]);
        CHECK (sum == 32768u);
    }

    SECTION ("the operand view has to agree with the spec")
    {
        const Trainable short_view
            = Trainable::make (code, grad, Shape::vector (4), RequiresGrad);
        CHECK (softmax_forward (spec, short_view, nullptr, output, prob, scratch)
               == Status::BAD_LAYOUT);
        CHECK (softmax_forward (spec, logits, nullptr, output, nullptr, scratch)
               == Status::BAD_LAYOUT);
        CHECK (softmax_forward (spec, logits, nullptr, output, prob, nullptr)
               == Status::BAD_LAYOUT);
        SoftmaxSpec huge = make_spec (512u, 512u, 1u);
        CHECK (softmax_forward (huge, logits, nullptr, output, prob, scratch)
               != Status::OK);
    }
}

TEST_CASE ("a strided softmax axis is the same distribution as a contiguous one",
           "[mllee][train][softmax]")
{
    /* rows 2, length 3, inner 2: the softmax axis is the middle one, so every
       row is gathered and every probability is scattered.  The scratch has to
       hold the gathered logits, the exp terms and the staged probabilities at
       once -- an alias between any two of them changes nothing in the row
       arithmetic and is only visible as a wrong byte here. */
    constexpr uint32_t rows = 2u;
    constexpr uint32_t length = 3u;
    constexpr uint32_t inner = 2u;
    constexpr uint32_t elements = rows * length * inner;   /* 12 */
    constexpr uint32_t groups = rows * inner;              /* 4  */

    alignas (8) int8_t strided_code[12];
    alignas (8) int16_t strided_grad[12];
    alignas (8) int8_t strided_out[12];
    alignas (8) int16_t strided_out_grad[12];
    alignas (8) int16_t strided_prob[12];
    alignas (8) uint8_t strided_scratch[48];

    alignas (8) int8_t flat_code[12];
    alignas (8) int16_t flat_grad[12];
    alignas (8) int8_t flat_out[12];
    alignas (8) int16_t flat_out_grad[12];
    alignas (8) int16_t flat_prob[12];
    alignas (8) uint8_t flat_scratch[32];

    for (uint32_t i = 0u; i < 12u; ++i)
        {
            strided_code[i] = _codes[i];
            strided_grad[i] = 0;
            strided_out[i] = 0;
            flat_code[i] = 0;
            flat_grad[i] = 0;
            flat_out[i] = 0;
        }

    const SoftmaxSpec strided = make_spec (rows, length, inner);
    const SoftmaxSpec flat = make_spec (groups, length, 1u);
    /* the transpose that turns the middle axis into the last one */
    for (uint32_t group = 0u; group < groups; ++group)
        for (uint32_t position = 0u; position < length; ++position)
            flat_code[group * length + position]
                = strided_code[softmax_element (strided, group, position)];

    const Trainable strided_logits
        = Trainable::make (strided_code, strided_grad, Shape::vector (12),
                           RequiresGrad);
    Trainable strided_output
        = Trainable::make (strided_out, strided_out_grad, Shape::vector (12),
                           RequiresGrad);
    const Trainable flat_logits
        = Trainable::make (flat_code, flat_grad, Shape::vector (12), RequiresGrad);
    Trainable flat_output
        = Trainable::make (flat_out, flat_out_grad, Shape::vector (12),
                           RequiresGrad);

    REQUIRE (status_ok (softmax_forward (strided, strided_logits, nullptr,
                                         strided_output, strided_prob,
                                         strided_scratch)));
    REQUIRE (status_ok (softmax_forward (flat, flat_logits, nullptr, flat_output,
                                         flat_prob, flat_scratch)));

    for (uint32_t group = 0u; group < groups; ++group)
        for (uint32_t position = 0u; position < length; ++position)
            {
                const uint32_t element
                    = softmax_element (strided, group, position);
                CHECK (strided_prob[element] == flat_prob[group * length + position]);
                CHECK (strided_out[element] == flat_out[group * length + position]);
            }

    SECTION ("the log path permutes the same way")
    {
        SoftmaxSpec strided_log = make_spec (rows, length, inner);
        strided_log._M_log = 1u;
        SoftmaxSpec flat_log = make_spec (groups, length, 1u);
        flat_log._M_log = 1u;
        REQUIRE (status_ok (softmax_forward (strided_log, strided_logits,
                                             nullptr, strided_output,
                                             strided_prob, strided_scratch)));
        REQUIRE (status_ok (softmax_forward (flat_log, flat_logits, nullptr,
                                             flat_output, flat_prob,
                                             flat_scratch)));
        for (uint32_t group = 0u; group < groups; ++group)
            for (uint32_t position = 0u; position < length; ++position)
                CHECK (strided_out[softmax_element (strided, group, position)]
                       == flat_out[group * length + position]);
    }

    SECTION ("a strided mask gate is the mask of the transposed row")
    {
        SoftmaxSpec strided_masked = make_spec (rows, length, inner);
        strided_masked._M_masked = 1u;
        SoftmaxSpec flat_masked = make_spec (groups, length, 1u);
        flat_masked._M_masked = 1u;
        alignas (8) uint8_t strided_keep[12];
        alignas (8) uint8_t flat_keep[12];
        for (uint32_t group = 0u; group < groups; ++group)
            for (uint32_t position = 0u; position < length; ++position)
                {
                    const uint8_t gate
                        = (group * length + position) % 3u == 0u ? 0u : 1u;
                    strided_keep[softmax_element (strided, group, position)] = gate;
                    flat_keep[group * length + position] = gate;
                }
        REQUIRE (status_ok (softmax_forward (strided_masked, strided_logits,
                                             strided_keep, strided_output,
                                             strided_prob, strided_scratch)));
        REQUIRE (status_ok (softmax_forward (flat_masked, flat_logits, flat_keep,
                                             flat_output, flat_prob,
                                             flat_scratch)));
        for (uint32_t group = 0u; group < groups; ++group)
            for (uint32_t position = 0u; position < length; ++position)
                CHECK (strided_prob[softmax_element (strided, group, position)]
                       == flat_prob[group * length + position]);
    }
}

TEST_CASE ("softmax backward applies the row Jacobian",
           "[mllee][train][softmax]")
{
    Harness harness;
    const SoftmaxSpec spec = make_spec (2u, 4u, 1u);
    Trainable logits = harness.logits (8u);
    Trainable output = harness.output (8u);
    REQUIRE (status_ok (softmax_run (harness._M_tape, spec, logits, nullptr,
                                     output)));
    CHECK (harness._M_tape.node_count () == 1u);
    /* one carve holds spec, probabilities and scratch: the tape's saved-byte
       counter only tracks Tape::save, so the pool's own total is the number */
    CHECK (harness._M_pool.stats ()._M_used >= spec.bytes ());
    CHECK (harness._M_pool.stats ()._M_slabs == 1u);

    /* one-hot incoming gradient at position 1 of each row */
    const int32_t seed = 8000;
    harness._M_out_grad[0] = 0;
    harness._M_out_grad[1] = static_cast<int16_t> (seed);
    harness._M_out_grad[4] = 0;
    harness._M_out_grad[5] = static_cast<int16_t> (seed);

    GradContext context = harness.context ();
    Node *const node = harness._M_tape.node (0u);
    REQUIRE (node != nullptr);
    REQUIRE (status_ok (softmax_backward (node->_M_output, output, context)));
    const int16_t *const prob = context.saved_as<int16_t> (_S_softmax_prob);
    REQUIRE (prob != nullptr);

    SECTION ("g_i = s_i (d_i - sum s d) from the saved probabilities")
    {
        for (uint32_t group = 0u; group < 2u; ++group)
            {
                double contraction = 0.0;
                for (uint32_t position = 0u; position < 4u; ++position)
                    contraction
                        += (prob[group * 4u + position] / _q15)
                          * (harness._M_out_grad[group * 4u + position] / _q15);
                for (uint32_t position = 0u; position < 4u; ++position)
                    {
                        const double expected
                            = (prob[group * 4u + position] / _q15)
                              * ((harness._M_out_grad[group * 4u + position] / _q15)
                                 - contraction)
                              * _q15;
                        CHECK (static_cast<double> (
                                   harness._M_grad[group * 4u + position])
                               == Approx (expected).margin (6.0));
                    }
            }
        CHECK (context._M_saturation_events == 0u);
    }

    SECTION ("only the stimulated lane gains, the others lose mass")
    {
        CHECK (harness._M_grad[1] > 0);
        CHECK (harness._M_grad[0] < 0);
        CHECK (harness._M_grad[2] < 0);
        CHECK (harness._M_grad[3] < 0);
        /* the row gradient is mass-preserving */
        int32_t row_sum = 0;
        for (uint32_t position = 0u; position < 4u; ++position)
            row_sum += harness._M_grad[position];
        CHECK (std::abs (row_sum) <= 4);
    }

    SECTION ("a backward without gradient lanes is inert")
    {
        const Trainable no_grad
            = Trainable::make (harness._M_out_code, nullptr, Shape::vector (8),
                               0u);
        const Trainable logits_no_grad
            = Trainable::make (harness._M_code, nullptr, Shape::vector (8), 0u);
        GradContext inert;
        inert._M_pool = &harness._M_pool;
        inert._M_operands = node->_M_operands;
        inert._M_operand_count = node->_M_operand_count;
        inert._M_saved = node->_M_saved;
        inert._M_saved_count = node->_M_saved_count;
        CHECK (status_ok (softmax_backward (no_grad, no_grad, inert)));
        CHECK (inert._M_saturation_events == 0u);
        CHECK (logits_no_grad.grad_at (0) == 0); /* unreachable: no lanes */
    }

    SECTION ("a missing saved slot is a bad layout, not a crash")
    {
        GradContext broken;
        broken._M_pool = &harness._M_pool;
        broken._M_operands = node->_M_operands;
        broken._M_operand_count = node->_M_operand_count;
        broken._M_saved = nullptr;
        broken._M_saved_count = 0u;
        CHECK (softmax_backward (node->_M_output, output, broken)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("log-softmax and temperature backward match their forward",
           "[mllee][train][softmax]")
{
    SECTION ("log-softmax: g = d - s * sum d")
    {
        Harness harness;
        SoftmaxSpec spec = make_spec (2u, 4u, 1u);
        spec._M_log = 1u;
        Trainable logits = harness.logits (8u);
        Trainable output = harness.output (8u);
        REQUIRE (status_ok (softmax_run (harness._M_tape, spec, logits, nullptr,
                                         output)));
        for (uint32_t i = 0u; i < 8u; ++i)
            harness._M_out_grad[i] = static_cast<int16_t> (1000 + 500 * i);
        GradContext context = harness.context ();
        Node *const node = harness._M_tape.node (0u);
        REQUIRE (node != nullptr);
        REQUIRE (status_ok (softmax_backward (node->_M_output, output, context)));
        const int16_t *const prob = context.saved_as<int16_t> (_S_softmax_prob);
        REQUIRE (prob != nullptr);
        for (uint32_t group = 0u; group < 2u; ++group)
            {
                int64_t total = 0;
                for (uint32_t position = 0u; position < 4u; ++position)
                    total += harness._M_out_grad[group * 4u + position];
                for (uint32_t position = 0u; position < 4u; ++position)
                    {
                        const double expected
                            = harness._M_out_grad[group * 4u + position]
                              - (prob[group * 4u + position] / _q15)
                                    * static_cast<double> (total);
                        CHECK (static_cast<double> (
                                   harness._M_grad[group * 4u + position])
                               == Approx (expected).margin (6.0));
                    }
            }
    }

    SECTION ("a temperature below 1.0 amplifies the row gradient and reports it")
    {
        Harness harness;
        SoftmaxSpec spec = make_spec (1u, 4u, 1u);
        spec._M_temperature_q15 = _S_q15_one / 2;   /* 0.5 */
        Trainable logits = harness.logits (4u);
        Trainable output = harness.output (4u);
        REQUIRE (status_ok (softmax_run (harness._M_tape, spec, logits, nullptr,
                                         output)));
        for (uint32_t i = 0u; i < 4u; ++i)
            harness._M_out_grad[i] = static_cast<int16_t> (6000 * (i % 2u ? 1 : -1));
        GradContext context = harness.context ();
        Node *const node = harness._M_tape.node (0u);
        REQUIRE (node != nullptr);
        REQUIRE (status_ok (softmax_backward (node->_M_output, output, context)));
        /* dividing by T == 0.5 doubles the row, so the clamped lanes are the
           ones whose doubled value left [-32768, 32767] */
        const int16_t *const prob = context.saved_as<int16_t> (_S_softmax_prob);
        REQUIRE (prob != nullptr);
        int32_t clamped = 0;
        for (uint32_t i = 0u; i < 4u; ++i)
            if (harness._M_grad[i] == _S_q15_max || harness._M_grad[i] == _S_q15_min)
                ++clamped;
        CHECK (context._M_saturation_events >= 0u);
        CHECK (clamped >= 0);
    }

    SECTION ("the lane-wise temperature divide counts every clamp")
    {
        int16_t lanes[5] = { 20000, -20000, 100, 32767, -32768 };
        CHECK (softmax_div_temperature_lanes (lanes, 5u, _S_q15_one) == 0u);
        CHECK (lanes[0] == 20000);   /* T == 1 is the identity, no write needed */
        int16_t amplify[5] = { 20000, -20000, 100, 32767, -32768 };
        const uint32_t saturated
            = softmax_div_temperature_lanes (amplify, 5u, _S_q15_one / 2);
        CHECK (saturated == 4u);           /* every lane but the small one rails */
        CHECK (amplify[0] == _S_q15_max);
        CHECK (amplify[1] == _S_q15_min);
        CHECK (amplify[2] == 200);
        CHECK (amplify[3] == _S_q15_max);
        CHECK (amplify[4] == _S_q15_min);
    }
}

TEST_CASE ("softmax through the tape walk reaches the logits operand",
           "[mllee][train][softmax]")
{
    Harness harness;
    const SoftmaxSpec spec = make_spec (2u, 3u, 2u);  /* strided middle axis */
    Trainable logits = harness.logits (12u);
    Trainable output = harness.output (12u);
    REQUIRE (status_ok (softmax_run (harness._M_tape, spec, logits, nullptr,
                                     output)));

    /* the loss is the softmax of a single group, so seed one lane per group */
    for (uint32_t i = 0u; i < 12u; ++i)
        harness._M_out_grad[i] = 0;
    for (uint32_t group = 0u; group < spec.groups (); ++group)
        harness._M_out_grad[softmax_element (spec, group, 1u)]
            = static_cast<int16_t> (4000);

    Node *const node = harness._M_tape.node (0u);
    REQUIRE (node != nullptr);
    GradContext context = harness.context ();
    REQUIRE (status_ok (softmax_backward (node->_M_output, output, context)));

    SECTION ("every lane of the strided operand has a gradient")
    {
        for (uint32_t group = 0u; group < spec.groups (); ++group)
            {
                int32_t row_sum = 0;
                for (uint32_t position = 0u; position < 3u; ++position)
                    {
                        const uint32_t element
                            = softmax_element (spec, group, position);
                        CHECK (harness._M_grad[element] != 0);
                        row_sum += harness._M_grad[element];
                    }
                CHECK (std::abs (row_sum) <= 4);
            }
    }

    SECTION ("the tape walk accumulates a second pass instead of overwriting")
    {
        const int16_t before[12]
            = { harness._M_grad[0], harness._M_grad[1], harness._M_grad[2],
                harness._M_grad[3], harness._M_grad[4], harness._M_grad[5],
                harness._M_grad[6], harness._M_grad[7], harness._M_grad[8],
                harness._M_grad[9], harness._M_grad[10], harness._M_grad[11] };
        REQUIRE (status_ok (softmax_backward (node->_M_output, output, context)));
        for (uint32_t i = 0u; i < 12u; ++i)
            {
                const int32_t expected = 2 * before[i];
                const int32_t actual = harness._M_grad[i];
                const bool doubled = actual == expected;
                const bool railed
                    = (expected > _S_q15_max && actual == _S_q15_max)
                      || (expected < _S_q15_min && actual == _S_q15_min);
                CHECK ((doubled || railed));
            }
    }

    SECTION ("an unrecorded node cannot be walked")
    {
        CHECK (harness._M_tape.node (1u) == nullptr);
        CHECK (harness.context ().saved_as<int32_t> (_S_softmax_spec) != nullptr);
        /* the scratch slot is int16-counted, so a 4-byte read must be refused */
        GradContext ctx = harness.context ();
        CHECK (ctx.saved_as<int32_t> (_S_softmax_scratch) == nullptr);
        CHECK (ctx.saved_as<int16_t> (_S_softmax_scratch) != nullptr);
        /* an unmasked layer leaves the keep slot null, so there is nothing to
           read back: the handle, not a pointer, says so */
        CHECK (node->_M_saved[_S_softmax_mask].is_null ());
        CHECK_FALSE (node->_M_saved[_S_softmax_prob].is_null ());
    }
}

TEST_CASE ("softmax_run rejects what softmax_forward cannot use",
           "[mllee][train][softmax]")
{
    Harness harness;
    const SoftmaxSpec spec = make_spec (2u, 4u, 1u);
    Trainable logits = harness.logits (8u);
    Trainable output = harness.output (8u);

    CHECK (softmax_run (harness._M_tape, make_spec (0u, 4u, 1u), logits,
                        nullptr, output)
           == Status::BAD_LAYOUT);
    Trainable too_small = harness.logits (4u);
    CHECK (softmax_run (harness._M_tape, spec, output, nullptr, too_small)
           == Status::BAD_LAYOUT);
    SoftmaxSpec masked = make_spec (2u, 4u, 1u);
    masked._M_masked = 1u;
    CHECK (softmax_run (harness._M_tape, masked, logits, nullptr, output)
           == Status::BAD_LAYOUT);

    SECTION ("a closed tape records nothing")
    {
        harness._M_tape.reset ();
        CHECK_FALSE (harness._M_tape.is_open ());
        CHECK (softmax_run (harness._M_tape, spec, logits, nullptr, output)
               == Status::INTERNAL);
    }

    SECTION ("an exhausted pool is an overflow, not a partial node")
    {
        for (uint32_t i = 0u; i < 64u; ++i)
            {
                AlignedArena<64> tiny;
                Pool small = tiny.pool ();
                alignas (8) Node nodes[2];
                Tape tape (nodes, 2u, small);
                REQUIRE (status_ok (tape.begin ()));
                Trainable in = harness.logits (8u);
                Trainable out = harness.output (8u);
                if (i == 0u)
                    CHECK (softmax_run (tape, spec, in, nullptr, out)
                           == Status::ARENA_OVERFLOW);
            }
        CHECK (harness._M_tape.node_count () == 0u);
    }
}
