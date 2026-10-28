/* Tests for the QAT pair: mllee/train_fake_quant.hpp (quantize-dequantize plus the
 * exact straight-through estimator) and mllee/train_calibration.hpp (post-training
 * observers and grid selection).
 *
 * Every expectation in this file comes from a reference written *here*: plain
 * integer arithmetic that re-derives the rules the two headers document -- round
 * half up at 2^15, the 258-unit identity code step, the [-127, 127] rails (the
 * -128 code is never emitted), the ceil-div scale rules, the bin index
 * `(value + 32768) >> 8` over 256 equal bins, the 254-step affine grid, and the
 * Mse/Kl candidate sweep 127 down to `min_codes` with the first-strict-minimum
 * tie rule.  Nothing is asserted by calling the library and echoing its output:
 * the reference recomputes codes, clip windows, objectives and chosen grids from
 * the rules, and the library is then compared against it.  Where an
 * implementation and its own documented rule disagree, the test pins the
 * observed value, names the deviation in a comment, and the deviation is
 * reported rather than patched. */

#include "mllee/train_calibration.hpp"
#include "mllee/train_fake_quant.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>

using namespace zzs::mllee;
using namespace zzs::mllee::train;

namespace
{
/** @brief The documented value model, spelled with plain integers. */
constexpr int64_t R_Q15_ONE = 32768LL;
constexpr int32_t R_Q15_MAX = 32767;
constexpr int32_t R_Q15_MIN = -32768;
constexpr int64_t R_HALF_15 = 16384LL;
constexpr int32_t R_CODE_STEP = 258;
constexpr int32_t R_INT32_MAX = 2147483647;
constexpr int32_t R_INT32_MIN = -2147483647 - 1;

constexpr int64_t
r_half (const uint32_t shift) noexcept
{
    return static_cast<int64_t> (1) << (shift - 1u);
}

/** @brief round-half-up right shift, ties toward +infinity, as documented. */
constexpr int32_t
r_round_shift (const int64_t value, const uint32_t shift) noexcept
{
    if (shift == 0u)
        return static_cast<int32_t> (value);
    return static_cast<int32_t> ((value + r_half (shift)) >> shift);
}

constexpr int32_t
r_clamp_i8 (const int32_t value) noexcept
{
    return value < -127 ? -127 : (value > 127 ? 127 : value);
}

constexpr int8_t
r_i8 (const int32_t value) noexcept
{
    return static_cast<int8_t> (r_clamp_i8 (value));
}

constexpr int32_t
r_q15_clamp (const int32_t value) noexcept
{
    return value > R_Q15_MAX ? R_Q15_MAX : (value < R_Q15_MIN ? R_Q15_MIN : value);
}

constexpr int16_t
r_i16_sat (const int32_t value) noexcept
{
    return value > R_Q15_MAX
               ? static_cast<int16_t> (R_Q15_MAX)
               : (value < R_Q15_MIN ? static_cast<int16_t> (R_Q15_MIN)
                                    : static_cast<int16_t> (value));
}

constexpr int64_t
r_floor_div (const int64_t numerator, const int64_t denominator) noexcept
{
    const int64_t quotient = numerator / denominator;
    const int64_t remainder = numerator % denominator;
    return remainder != 0 && (remainder < 0) != (denominator < 0) ? quotient - 1
                                                                  : quotient;
}

constexpr int64_t
r_ceil_div (const int64_t numerator, const int64_t denominator) noexcept
{
    return -r_floor_div (-numerator, denominator);
}

constexpr int32_t
r_sat_i32 (const int64_t value) noexcept
{
    return value > R_INT32_MAX ? R_INT32_MAX
                               : (value < R_INT32_MIN ? R_INT32_MIN
                                                      : static_cast<int32_t> (value));
}

/** @brief The documented reciprocal: round(2^15 / scale). */
constexpr int32_t
r_multiplier_documented (const int32_t scale_q15) noexcept
{
    if (scale_q15 <= 0)
        return 0;
    return static_cast<int32_t> ((R_Q15_ONE + scale_q15 / 2) / scale_q15);
}

/** @brief The reciprocal `fq_multiplier_from_scale` actually computes. */
constexpr int64_t
r_multiplier_implemented (const int32_t scale_q15) noexcept
{
    if (scale_q15 <= 0)
        return 0;
    return (R_Q15_ONE * R_Q15_ONE + scale_q15 / 2) / scale_q15;
}

/** @brief The smallest symmetric step that reaches abs_max at code 127. */
constexpr int32_t
r_symmetric_scale (const int32_t abs_max_q15) noexcept
{
    if (abs_max_q15 <= 0)
        return R_CODE_STEP;
    const int64_t widened = r_ceil_div (abs_max_q15, 127);
    if (widened > R_Q15_MAX)
        return R_Q15_MAX;
    return widened < 1 ? 1 : static_cast<int32_t> (widened);
}

/** @brief The pre-clamp code of a Q15 value. */
constexpr int32_t
r_pre_clamp_code (const int32_t value_q15, const int32_t multiplier_q15,
                  const int32_t zero_point) noexcept
{
    return r_round_shift (static_cast<int64_t> (value_q15) * multiplier_q15, 15u)
           + zero_point;
}

constexpr int32_t
r_dequantize_q15 (const int32_t code, const int32_t scale_q15,
                  const int32_t zero_point) noexcept
{
    return r_q15_clamp ((code - zero_point) * scale_q15);
}

constexpr int32_t
r_dequantized_value (const int32_t value_q15, const int32_t scale_q15,
                     const int32_t multiplier_q15, const int32_t zero_point,
                     const int8_t qmin, const int8_t qmax, bool &clipped) noexcept
{
    const int32_t raw = r_pre_clamp_code (value_q15, multiplier_q15, zero_point);
    clipped = raw < static_cast<int32_t> (qmin) || raw > static_cast<int32_t> (qmax);
    return r_dequantize_q15 (r_clamp_i8 (raw), scale_q15, zero_point);
}

/** @brief The exact clip window, rebuilt from the rounding rule. */
constexpr ClipWindow
r_clip_window (const int32_t multiplier_q15, const int32_t zero_point,
               const int8_t qmin, const int8_t qmax) noexcept
{
    ClipWindow window;
    window._M_low = R_INT32_MIN;
    window._M_high = R_INT32_MAX;
    window._M_code_low = 0;
    window._M_code_high = 0;
    window._M_qmin = qmin;
    window._M_qmax = qmax;
    window._M_valid = 0u;
    window._M_pad = 0u;
    if (multiplier_q15 <= 0)
        return window;
    const int64_t low_level = static_cast<int64_t> (qmin) - zero_point;
    const int64_t high_level = static_cast<int64_t> (qmax) - zero_point;
    window._M_low = r_sat_i32 (r_ceil_div (low_level * R_Q15_ONE - R_HALF_15,
                                           multiplier_q15));
    window._M_high
        = r_sat_i32 (r_floor_div ((high_level + 1) * R_Q15_ONE - R_HALF_15 - 1,
                                  multiplier_q15));
    window._M_code_low = static_cast<int32_t> (low_level);
    window._M_code_high = static_cast<int32_t> (high_level);
    window._M_valid = 1u;
    return window;
}

/** @brief narrow_sum_q15: a Q30 total narrowed once with the rounding shift. */
constexpr int32_t
r_narrow_sum_q15 (const int64_t total) noexcept
{
    const int64_t limit = static_cast<int64_t> (R_Q15_MAX) << 15;
    const int64_t bounded
        = total > limit ? limit : (total < -limit ? -limit : total);
    return r_q15_clamp (r_round_shift (bounded, 15u));
}

constexpr int32_t
r_clamp_sum_q15 (const int64_t total) noexcept
{
    return r_q15_clamp (static_cast<int32_t> (
        total > R_Q15_MAX ? R_Q15_MAX : (total < R_Q15_MIN ? R_Q15_MIN : total)));
}

constexpr int32_t
r_q15_mul (const int32_t a, const int32_t b) noexcept
{
    return r_q15_clamp (r_round_shift (static_cast<int64_t> (a) * b, 15u));
}

constexpr int32_t
r_code_to_q15 (const int32_t code) noexcept
{
    return code * R_CODE_STEP;
}

/** @brief One observed (min,max) pair folded into a reference accumulator. */
struct RefMinMax
{
    int32_t min_q15 = 0;
    int32_t max_q15 = 0;
    int32_t abs_max_q15 = 0;
    int32_t sum_q15 = 0;
    uint32_t count = 0u;
    bool seeded = false;

    void
    feed (const int32_t value_q15) noexcept
    {
        if (!seeded)
            {
                min_q15 = value_q15;
                max_q15 = value_q15;
                seeded = true;
            }
        else
            {
                if (value_q15 < min_q15)
                    min_q15 = value_q15;
                if (value_q15 > max_q15)
                    max_q15 = value_q15;
            }
        const int32_t magnitude = value_q15 < 0 ? -value_q15 : value_q15;
        if (magnitude > abs_max_q15)
            abs_max_q15 = magnitude;
        sum_q15 = r_clamp_sum_q15 (static_cast<int64_t> (sum_q15) + magnitude);
        if (count < 0xFFFFu)
            ++count;
    }

    [[nodiscard]] MinMaxAccumulator
    as_accumulator () const noexcept
    {
        MinMaxAccumulator accumulator;
        accumulator._M_min_q15 = min_q15;
        accumulator._M_max_q15 = max_q15;
        accumulator._M_abs_max_q15 = abs_max_q15;
        accumulator._M_sum_q15 = sum_q15;
        accumulator._M_count = static_cast<uint16_t> (count);
        accumulator._M_seeded = seeded ? 1u : 0u;
        return accumulator;
    }
};

bool
same_accumulator (const MinMaxAccumulator &actual,
                  const MinMaxAccumulator &expected) noexcept
{
    return actual._M_min_q15 == expected._M_min_q15
           && actual._M_max_q15 == expected._M_max_q15
           && actual._M_abs_max_q15 == expected._M_abs_max_q15
           && actual._M_sum_q15 == expected._M_sum_q15
           && actual._M_count == expected._M_count
           && actual._M_seeded == expected._M_seeded;
}

/* narrow_clip_range: keep_fraction_bp of the observed extreme, never widening the
   incoming step; the asymmetric branch only reacts to a strictly positive span. */
constexpr int32_t
r_narrow_symmetric_scale (const int32_t current_scale, const int32_t abs_max_q15,
                          const int32_t keep_fraction_bp) noexcept
{
    const int32_t kept = r_sat_i32 (r_floor_div (static_cast<int64_t> (abs_max_q15)
                                                     * keep_fraction_bp,
                                                 10000));
    int32_t wanted = r_symmetric_scale (kept);
    if (wanted < 1)
        wanted = 1;
    return wanted > current_scale ? current_scale : wanted;
}

constexpr int32_t
r_narrow_asymmetric_scale (const int32_t current_scale, const int32_t min_q15,
                           const int32_t max_q15,
                           const int32_t keep_fraction_bp) noexcept
{
    int64_t wanted = current_scale;
    const int64_t span = static_cast<int64_t> (max_q15) - static_cast<int64_t> (min_q15);
    if (span > 0)
        wanted = r_ceil_div (span * keep_fraction_bp, 254LL * 10000);
    if (wanted < 1)
        wanted = 1;
    return wanted > current_scale ? current_scale : r_sat_i32 (wanted);
}

/* The narrow code range `narrow_clip_range` writes back for a symmetric grid:
   the code the new grid reaches for the observed abs max, clamped to [1, 127]. */
constexpr int8_t
r_narrow_reached_bound (const int32_t abs_max_q15, const int32_t scale_q15) noexcept
{
    const int64_t reached
        = r_round_shift (static_cast<int64_t> (abs_max_q15)
                             * r_multiplier_implemented (scale_q15),
                         15u);
    if (reached < 1)
        return 1;
    return reached > 127 ? static_cast<int8_t> (127) : static_cast<int8_t> (reached);
}

/* Scales used by the sweeps: the identity step, power-of-two steps (whose
   documented reciprocal is exact) and odd steps (whose reciprocal is not). */
constexpr int32_t _S_scale_table[] = { 1,    2,    4,    8,      16,   32,
                                       64,   128,  129,  256,    258,  300,
                                       516,  1000, 1024, 4096,   8192, 8191,
                                       32767, 17,   100, 3333,   16384 };

constexpr uint32_t
scale_table_size () noexcept
{
    return static_cast<uint32_t> (sizeof (_S_scale_table) / sizeof (int32_t));
}
} // namespace

/* ------------------------------------------------------------------ fake quant */

TEST_CASE ("fake-quant round trip is exact on grid points and half an LSB elsewhere",
           "[mllee][train][fakequant]")
{
    for (uint32_t entry = 0; entry < scale_table_size (); ++entry)
        {
            const int32_t scale = _S_scale_table[entry];
            const int32_t multiplier = r_multiplier_documented (scale);
            SECTION ("grid derived from the code step")
            {
                CAPTURE (scale, multiplier);
                const int32_t scales[1] = { scale };
                const int32_t mults[1] = { multiplier };
                const FakeQuantParams params
                    = FakeQuantParams::symmetric (scales, mults);
                REQUIRE (status_ok (params.validate ()));

                /* Every code of the grid is a representable value: the round trip
                   must be lossless and the emitted code must be the code itself. */
                for (int32_t code = -127; code <= 127; ++code)
                    {
                        const int32_t value = code * scale;
                        if (value > R_Q15_MAX || value < R_Q15_MIN)
                            continue;
                        bool clipped = true;
                        const int32_t back = fq_dequantized_value (
                            value, scale, multiplier, 0, -127, 127, clipped);
                        CHECK (clipped == false);
                        CHECK (back == value);
                        CHECK (fq_pre_clamp_code (value, multiplier, 0) == code);
                    }

                /* Any input lands within half an LSB of the scale, plus the
                   bounded skew of the reciprocal multiplier. */
                for (int32_t value = R_Q15_MIN; value <= R_Q15_MAX; value += 101)
                    {
                        bool clipped = true;
                        const int32_t back = fq_dequantized_value (
                            value, scale, multiplier, 0, -127, 127, clipped);
                        bool ref_clipped = false;
                        const int32_t ref_back = r_dequantized_value (
                            value, scale, multiplier, 0, -127, 127, ref_clipped);
                        CHECK (clipped == ref_clipped);
                        CHECK (back == ref_back);
                        if (ref_clipped)
                            continue;
                        const int32_t error = back - value;
                        const int32_t magnitude = error < 0 ? -error : error;
                        const int32_t abs_value = value < 0 ? -value : value;
                        const int64_t skew
                            = value < 0 ? static_cast<int64_t> (scale)
                                        : 0; /* half-up bias, one LSB worst case */
                        const int64_t bound2
                            = static_cast<int64_t> (scale) + skew
                              + r_ceil_div (static_cast<int64_t> (scale) * abs_value,
                                            R_Q15_ONE);
                        CHECK (2 * static_cast<int64_t> (magnitude) <= bound2);
                        /* The dequantized value sits on the grid unless the Q15
                           rail clipped it. */
                        if (back == (r_clamp_i8 (r_pre_clamp_code (
                                         value, multiplier, 0))) * scale)
                            CHECK (back % scale == 0);
                    }
            }
        }
}

TEST_CASE ("fake-quant params validate the documented contract",
           "[mllee][train][fakequant]")
{
    const int32_t good_scale[3] = { 258, 516, 1024 };
    const int32_t good_mult[3] = { 127, 64, 32 };
    const int8_t good_zp[3] = { 0, -5, 7 };

    SECTION ("accepted configurations")
    {
        const FakeQuantParams sym
            = FakeQuantParams::symmetric (good_scale, good_mult);
        CHECK (status_ok (sym.validate ()));
        CHECK (sym._M_mode == static_cast<uint8_t> (FakeQuantMode::Symmetric));
        CHECK (sym._M_axis == _S_fq_per_tensor);
        CHECK (sym._M_channels == 1u);
        CHECK (sym._M_qmin == -127);
        CHECK (sym._M_qmax == 127);
        CHECK_FALSE (sym.per_channel ());
        CHECK (sym.table_index (7u) == 0u);
        CHECK (sym.zero_point_at (1u) == 0);
        CHECK (sym.scale_at (2u) == 258);
        CHECK (sym.multiplier_at (2u) == 127);

        const FakeQuantParams asym
            = FakeQuantParams::asymmetric (good_scale, good_mult, good_zp);
        CHECK (status_ok (asym.validate ()));
        /* A per-tensor asymmetric grid only ever reads entry zero. */
        CHECK (asym.zero_point_at (0u) == 0);
        CHECK (asym.zero_point_at (2u) == 0);

        const FakeQuantParams channel_form = asym.on_axis (0u, 3u);
        CHECK (status_ok (channel_form.validate ()));
        CHECK (channel_form.per_channel ());
        CHECK (channel_form.table_index (1u) == 1u);
        CHECK (channel_form.scale_at (1u) == 516);
        CHECK (channel_form.multiplier_at (2u) == 32);
        CHECK (channel_form.zero_point_at (1u) == -5);
        CHECK (channel_form.zero_point_at (2u) == 7);

        /* The narrow symmetric clamp range the header allows. */
        FakeQuantParams narrow = sym;
        narrow._M_qmin = -8;
        narrow._M_qmax = 8;
        CHECK (status_ok (narrow.validate ()));

        /* An asymmetric descriptor without a zero-point table is legal and reads 0. */
        FakeQuantParams no_zp = asym;
        no_zp._M_zero_point = nullptr;
        CHECK (status_ok (no_zp.validate ()));
        CHECK (no_zp.zero_point_at (0u) == 0);
    }

    SECTION ("rejected configurations carry the documented Status")
    {
        FakeQuantParams missing = FakeQuantParams::symmetric (good_scale, good_mult);
        missing._M_scale_q15 = nullptr;
        CHECK (missing.validate () == Status::BAD_LAYOUT);
        missing = FakeQuantParams::symmetric (good_scale, good_mult);
        missing._M_multiplier_q15 = nullptr;
        CHECK (missing.validate () == Status::BAD_LAYOUT);

        FakeQuantParams channels = FakeQuantParams::symmetric (good_scale, good_mult);
        channels._M_channels = 0u;
        CHECK (channels.validate () == Status::OUT_OF_RANGE);
        channels._M_channels = 257u;
        CHECK (channels.validate () == Status::OUT_OF_RANGE);

        /* qmin > qmax is a layout error. */
        FakeQuantParams inverted = FakeQuantParams::symmetric (good_scale, good_mult);
        inverted._M_mode = static_cast<uint8_t> (FakeQuantMode::Asymmetric);
        inverted._M_qmin = 9;
        inverted._M_qmax = 5;
        CHECK (inverted.validate () == Status::BAD_LAYOUT);
        /* The mirror image is accepted: an inverted clamp only becomes an error
           when qmin strictly exceeds qmax. */
        inverted._M_qmin = 5;
        inverted._M_qmax = 9;
        CHECK (inverted.validate () == Status::OK);

        FakeQuantParams bad_mode = FakeQuantParams::symmetric (good_scale, good_mult);
        bad_mode._M_mode = 2u;
        CHECK (bad_mode.validate () == Status::BAD_LAYOUT);
        bad_mode._M_mode = 255u;
        CHECK (bad_mode.validate () == Status::BAD_LAYOUT);

        FakeQuantParams bad_axis = FakeQuantParams::symmetric (good_scale, good_mult);
        CHECK (bad_axis.on_axis (4u, 1u).validate () == Status::BAD_LAYOUT);
        CHECK (bad_axis.on_axis (255u, 1u).validate () == Status::OK);
        CHECK (bad_axis.on_axis (3u, 1u).validate () == Status::OK);

        /* -128 is the one out-of-range code the module never emits. */
        FakeQuantParams deep = FakeQuantParams::symmetric (good_scale, good_mult);
        deep._M_qmin = -128;
        deep._M_qmax = -128;
        CHECK (deep.validate () == Status::OUT_OF_RANGE);

        const int32_t zero_scale[1] = { 0 };
        FakeQuantParams flat = FakeQuantParams::symmetric (zero_scale, good_mult);
        CHECK (flat.validate () == Status::OUT_OF_RANGE);
        const int32_t negative_mult[1] = { -1 };
        flat = FakeQuantParams::symmetric (good_scale, negative_mult);
        CHECK (flat.validate () == Status::OUT_OF_RANGE);

        const int32_t second_negative[3] = { 127, 0, 32 };
        FakeQuantParams second_bad
            = FakeQuantParams::symmetric (good_scale, second_negative).on_axis (0u, 3u);
        CHECK (second_bad.validate () == Status::OUT_OF_RANGE);

        const int8_t minus_128[3] = { 3, -128, 0 };
        FakeQuantParams bad_zp
            = FakeQuantParams::asymmetric (good_scale, good_mult, minus_128).on_axis (
                0u, 3u);
        CHECK (bad_zp.validate () == Status::OUT_OF_RANGE);

        /* A symmetric grid must be balanced about zero and reach a positive code. */
        FakeQuantParams lopsided = FakeQuantParams::symmetric (good_scale, good_mult);
        lopsided._M_qmax = 64;
        CHECK (lopsided.validate () == Status::BAD_LAYOUT);
        lopsided._M_qmax = 127;
        lopsided._M_qmin = -100;
        CHECK (lopsided.validate () == Status::BAD_LAYOUT);
        lopsided._M_qmin = -127;
        CHECK (status_ok (lopsided.validate ()));
        lopsided._M_qmax = 0;
        CHECK (lopsided.validate () == Status::BAD_LAYOUT);
    }

    SECTION ("the descriptors keep their measured sizes")
    {
        CHECK (sizeof (FakeQuantParams) <= (sizeof (void *) == 4u ? 24u : 40u));
        CHECK (sizeof (FakeQuantLayout) == 20u);
        CHECK (sizeof (ClipWindow) == 20u);
        CHECK (sizeof (ClipWindowLane) == 24u);
        CHECK (sizeof (MinMaxAccumulator) == 20u);
        CHECK (sizeof (FakeQuantNodeState) == 24u);
        CHECK (alignof (ClipWindowLane) == 4u);
        CHECK (alignof (FakeQuantNodeState) == 4u);
    }
}

TEST_CASE ("fake-quant layout derives contiguous channel runs",
           "[mllee][train][fakequant]")
{
    const int32_t scales[3] = { 258, 516, 1024 };
    const int32_t mults[3] = { 127, 64, 32 };
    const FakeQuantParams per_tensor = FakeQuantParams::symmetric (scales, mults);
    const FakeQuantParams per_channel
        = FakeQuantParams::symmetric (scales, mults).on_axis (1u, 3u);

    SECTION ("per-tensor is the degenerate run")
    {
        FakeQuantLayout layout;
        REQUIRE (status_ok (fake_quant_layout (Shape::matrix (4, 5), per_tensor,
                                               layout)));
        CHECK (layout._M_valid == 1u);
        CHECK (layout._M_run == 20u);
        CHECK (layout._M_group == 20u);
        CHECK (layout._M_outer == 1u);
        CHECK (layout._M_count == 20u);
        CHECK (layout._M_channels == 1u);
        CHECK (layout.base (0u, 0u) == 0u);
        CHECK_FALSE (layout.per_channel ());
    }

    SECTION ("axis-wise runs repeat every dim times stride")
    {
        FakeQuantLayout layout;
        /* shape (2,3,4), axis 1: run = stride(1) = 4, group = 12, outer = 2. */
        REQUIRE (status_ok (fake_quant_layout (Shape::tensor3 (2, 3, 4), per_channel,
                                               layout)));
        CHECK (layout._M_run == 4u);
        CHECK (layout._M_group == 12u);
        CHECK (layout._M_outer == 2u);
        CHECK (layout._M_count == 24u);
        CHECK (layout._M_channels == 3u);
        CHECK (layout.per_channel ());
        for (uint32_t channel = 0; channel < 3u; ++channel)
            for (uint32_t block = 0; block < 2u; ++block)
                CHECK (layout.base (channel, block) == block * 12u + channel * 4u);
    }

    SECTION ("invalid shapes, axes and channel counts")
    {
        FakeQuantLayout layout;
        CHECK (fake_quant_layout (Shape {}, per_tensor, layout) == Status::BAD_LAYOUT);
        CHECK (fake_quant_layout (Shape::vector (0), per_tensor, layout)
               == Status::BAD_LAYOUT);
        /* Axis 2 is not an axis of a rank-2 shape. */
        CHECK (fake_quant_layout (Shape::matrix (4, 5),
                                  per_channel, layout)
               == Status::BAD_LAYOUT);
        /* dim(axis) must equal the table length. */
        CHECK (fake_quant_layout (
                  Shape::tensor3 (2, 3, 4),
                  FakeQuantParams::symmetric (scales, mults).on_axis (1u, 2u), layout)
               == Status::BAD_LAYOUT);
        /* An unvalidated parameter table never reaches the layout. */
        const int32_t zero_scale[3] = { 0, 516, 1024 };
        CHECK (fake_quant_layout (
                  Shape::tensor3 (2, 3, 4),
                  FakeQuantParams::symmetric (zero_scale, mults).on_axis (1u, 3u),
                  layout)
               == Status::OUT_OF_RANGE);
    }

    SECTION ("flat splits are channel-major")
    {
        FakeQuantLayout layout;
        REQUIRE (status_ok (fake_quant_layout_flat (24u, 3u, layout)));
        CHECK (layout._M_run == 8u);
        CHECK (layout._M_group == 24u);
        CHECK (layout._M_outer == 1u);
        CHECK (layout._M_axis == 0u);
        CHECK (layout._M_valid == 1u);
        CHECK (layout.base (2u, 0u) == 16u);
        CHECK (fake_quant_layout_flat (0u, 3u, layout) == Status::BAD_LAYOUT);
        CHECK (fake_quant_layout_flat (24u, 0u, layout) == Status::BAD_LAYOUT);
        CHECK (fake_quant_layout_flat (24u, 5u, layout) == Status::BAD_LAYOUT);
        CHECK (fake_quant_layout_flat (24u, 257u, layout) == Status::BAD_LAYOUT);
        CHECK (status_ok (fake_quant_layout_flat (24u, 1u, layout)));
    }
}

TEST_CASE ("fake-quant forward quantizes a lane stream through the grid",
           "[mllee][train][fakequant]")
{
    alignas (8) int16_t values[12];
    alignas (8) int16_t dequantized[12];
    alignas (8) int8_t codes[12];
    for (uint32_t i = 0; i < 12u; ++i)
        values[i] = static_cast<int16_t> (-30000 + static_cast<int32_t> (i) * 5000);

    const int32_t scale = 100;
    const int32_t multiplier = 328; /* round(32768 / 100) */
    const FakeQuantParams params = FakeQuantParams::symmetric (&scale, &multiplier);
    FakeQuantLayout layout;
    REQUIRE (status_ok (fake_quant_layout_flat (12u, 1u, layout)));

    SECTION ("codes, dequantized lanes and the clipped count")
    {
        /* `fake_quant_forward` adds to the caller's counter instead of clearing
           it (train_fake_quant.hpp:513-546), unlike the STE helpers that reset
           their out-params, so a stale value leaks into the result (reported). */
        uint32_t clipped = 99u;
        REQUIRE (status_ok (fake_quant_forward (values, codes, dequantized, params,
                                                layout, clipped)));
        uint32_t ref_clipped = 99u;
        for (uint32_t i = 0; i < 12u; ++i)
            {
                const int32_t raw = r_pre_clamp_code (values[i], multiplier, 0);
                const int8_t expected = r_i8 (raw);
                CHECK (codes[i] == expected);
                /* The -128 code is never emitted, whatever the raw value. */
                CHECK (codes[i] >= -127);
                CHECK (codes[i] <= 127);
                CHECK (dequantized[i]
                       == r_i16_sat (r_dequantize_q15 (expected, scale, 0)));
                if (raw < -127 || raw > 127)
                    ++ref_clipped;
            }
        CHECK (clipped == ref_clipped);
        CHECK (clipped == 106u);
        CHECK (ref_clipped - 99u == 7u);
        CHECK (codes[0] == -127);
        CHECK (codes[11] == 127);
    }

    SECTION ("either output may be null")
    {
        uint32_t clipped_lanes = 0u;
        REQUIRE (status_ok (fake_quant_forward (values, nullptr, dequantized, params,
                                                layout, clipped_lanes)));
        uint32_t clipped_codes = 0u;
        REQUIRE (status_ok (fake_quant_forward (values, codes, nullptr, params, layout,
                                                clipped_codes)));
        CHECK (clipped_lanes == clipped_codes);
        CHECK (clipped_lanes == 7u);
    }

    SECTION ("argument checks")
    {
        uint32_t clipped = 0u;
        CHECK (fake_quant_forward (nullptr, codes, dequantized, params, layout, clipped)
               == Status::BAD_LAYOUT);
        CHECK (fake_quant_forward (values, nullptr, nullptr, params, layout, clipped)
               == Status::BAD_LAYOUT);
        FakeQuantLayout stale;
        CHECK (fake_quant_forward (values, codes, dequantized, params, stale, clipped)
               == Status::BAD_LAYOUT);
        FakeQuantLayout wide;
        REQUIRE (status_ok (fake_quant_layout_flat (12u, 2u, wide)));
        CHECK (fake_quant_forward (values, codes, dequantized, params, wide, clipped)
               == Status::BAD_LAYOUT);
        /* Streams must be 4-byte aligned; an odd int16 offset is not. */
        CHECK (fake_quant_forward (values + 1, codes, dequantized, params, layout,
                                   clipped)
               == Status::BAD_LAYOUT);
        CHECK (fake_quant_forward (values, codes + 1, dequantized, params, layout,
                                   clipped)
               == Status::BAD_LAYOUT);
        CHECK (fake_quant_forward (values, codes, dequantized + 1, params, layout,
                                   clipped)
               == Status::BAD_LAYOUT);
        const int32_t zero_scale[1] = { 0 };
        const FakeQuantParams broken
            = FakeQuantParams::symmetric (zero_scale, &multiplier);
        CHECK (fake_quant_forward (values, codes, dequantized, broken, layout, clipped)
               == Status::OUT_OF_RANGE);
    }

    SECTION ("asymmetric grids offset the code but not the value")
    {
        const int8_t zp_table[1] = { -20 };
        const FakeQuantParams asym
            = FakeQuantParams::asymmetric (&scale, &multiplier, zp_table);
        uint32_t clipped = 0u;
        REQUIRE (status_ok (fake_quant_forward (values, codes, dequantized, asym,
                                                layout, clipped)));
        /* Two independent counts of the same twelve lanes: the documented rule
           tests the emitted code against [qmin, qmax]; `fake_quant_forward`
           (train_fake_quant.hpp:535-546) compares the same post-offset code
           against [qmin - zp, qmax - zp], i.e. it subtracts the zero point
           twice (reported). */
        uint32_t strict = 0u;
        uint32_t offset_bounds = 0u;
        for (uint32_t i = 0; i < 12u; ++i)
            {
                const int32_t raw = r_pre_clamp_code (values[i], multiplier, -20);
                CHECK (codes[i] == r_i8 (raw));
                CHECK (dequantized[i]
                       == r_i16_sat (r_dequantize_q15 (r_clamp_i8 (raw), scale, -20)));
                if (raw < -127 || raw > 127)
                    ++strict;
                if (raw < -107 || raw > 147)
                    ++offset_bounds;
            }
        CHECK (clipped == offset_bounds);
        CHECK (strict == 7u);
        CHECK (offset_bounds == 7u);
    }

    SECTION ("the clip counter disagrees with the window for an offset grid")
    {
        /* Lanes whose code lands in [-126, -108] are inside the documented clamp
           range but outside the implementation's, so the forward counter and the
           clip window (which is the documented inverse of the rounding rule)
           give different verdicts for the same lane. */
        alignas (8) int16_t probe[3] = { -9483, -9484, 15484 };
        alignas (8) int8_t probe_codes[3] = { 0, 0, 0 };
        alignas (8) int16_t probe_back[3] = { 0, 0, 0 };
        const int8_t zp_table[1] = { -20 };
        const FakeQuantParams asym
            = FakeQuantParams::asymmetric (&scale, &multiplier, zp_table);
        FakeQuantLayout flat;
        REQUIRE (status_ok (fake_quant_layout_flat (3u, 1u, flat)));
        uint32_t clipped = 0u;
        REQUIRE (status_ok (fake_quant_forward (probe, probe_codes, probe_back, asym,
                                                flat, clipped)));
        const ClipWindow window = r_clip_window (multiplier, -20, -127, 127);
        CHECK (window._M_valid == 1u);
        uint32_t strict = 0u;
        uint32_t offset_bounds = 0u;
        for (uint32_t i = 0; i < 3u; ++i)
            {
                const int32_t raw = r_pre_clamp_code (probe[i], multiplier, -20);
                CHECK (probe_codes[i] == r_i8 (raw));
                const bool outside_strict = raw < -127 || raw > 127;
                const bool outside_offset
                    = raw < window._M_code_low || raw > window._M_code_high;
                strict += outside_strict ? 1u : 0u;
                offset_bounds += outside_offset ? 1u : 0u;
                /* The window is the inverse of the rounding rule, so its verdict
                   is the strict one: `low <= v <= high` iff the code fits. */
                const bool in_window = probe[i] >= window._M_low
                                       && probe[i] <= window._M_high;
                CHECK (in_window == !outside_strict);
            }
        CHECK (strict == 1u);
        CHECK (offset_bounds == 2u);
        CHECK (clipped == offset_bounds);
        CHECK (clipped != strict);
    }
}

TEST_CASE ("fake-quant rails are the int8 domain, not the configured clamp range",
           "[mllee][train][fakequant]")
{
    /* The documented forward rule (train_fake_quant.hpp:54) is
       `code = clamp(round(v * m / 2^15) + zp, qmin, qmax)`.  The implementation
       clamps to the int8 rails instead (train_fake_quant.hpp:547), so a lane
       outside a narrowed clamp range still emits its unclamped code while the clip
       counter and the clip window treat it as clipped.  Pinned as observed. */
    alignas (8) int16_t values[6] = { 5160, -5160, 2064, -2064, 200, -200 };
    alignas (8) int16_t dequantized[6];
    alignas (8) int8_t codes[6];

    const int32_t scale = 258;
    const int32_t multiplier = 127;
    FakeQuantParams params = FakeQuantParams::symmetric (&scale, &multiplier);
    params._M_qmin = -8;
    params._M_qmax = 8;
    REQUIRE (status_ok (params.validate ()));

    FakeQuantLayout layout;
    REQUIRE (status_ok (fake_quant_layout_flat (6u, 1u, layout)));
    uint32_t clipped = 0u;
    REQUIRE (status_ok (fake_quant_forward (values, codes, dequantized, params, layout,
                                            clipped)));

    for (uint32_t i = 0; i < 6u; ++i)
        {
            const int32_t raw = r_pre_clamp_code (values[i], multiplier, 0);
            CHECK (codes[i] == r_i8 (raw));
            CHECK (dequantized[i] == r_i16_sat (r_dequantize_q15 (raw, scale, 0)));
        }
    CHECK (codes[0] == 20);
    CHECK (codes[1] == -20);
    CHECK (dequantized[0] == 5160);
    CHECK (dequantized[1] == -5160);
    /* The two lanes beyond the +-8 clamp range count as clipped. */
    CHECK (clipped == 2u);

    const ClipWindow window = clip_window_of (multiplier, 0, -8, 8);
    const ClipWindow expected = r_clip_window (multiplier, 0, -8, 8);
    CHECK (window._M_valid == 1u);
    CHECK (window._M_low == expected._M_low);
    CHECK (window._M_high == expected._M_high);
    CHECK (window._M_code_low == -8);
    CHECK (window._M_code_high == 8);
    CHECK (values[0] > window._M_high);
    CHECK (values[2] <= window._M_high);
    CHECK (values[3] >= window._M_low);

    SECTION ("the ste gate drops the gradient of a lane the forward kept")
    {
        alignas (8) int16_t grad_out[6] = { 100, 200, 300, 400, 500, 600 };
        alignas (8) int16_t grad_in[6] = { 0 };
        const uint32_t zeroed = fake_quant_backward_ste (values, grad_out, grad_in,
                                                         &window, params, layout);
        CHECK (grad_in[0] == 0);
        CHECK (grad_in[1] == 0);
        CHECK (grad_in[2] == 300);
        CHECK (grad_in[3] == 400);
        CHECK (zeroed == 2u);
    }
}

TEST_CASE ("clip window is the exact clip interval of the rounding rule",
           "[mllee][train][fakequant]")
{
    struct Case
    {
        int32_t multiplier;
        int32_t zero_point;
        int8_t qmin;
        int8_t qmax;
    };
    const Case cases[] = {
        { 127, 0, -127, 127 },      { 64, 0, -127, 127 },
        { 328, 0, -127, 127 },      { 1928, 7, -127, 127 },
        { 1928, -7, -127, 127 },    { 32, 0, -8, 8 },
        { 16384, 0, -127, 127 },    { 32768, 0, -127, 127 },
        { 3, 0, -127, 127 },        { 1, 127, -127, 127 },
        { 100, -127, -127, 127 },   { 127, -127, -127, 127 },
        { 63, 30, -100, 90 },       { 2147483647, 0, -127, 127 },
        { 2147483647, -127, -127, 127 }, { 2, -60, -127, 100 },
    };

    for (const Case &entry : cases)
        {
            SECTION ("window arithmetic")
            {
                CAPTURE (entry.multiplier, entry.zero_point, entry.qmin, entry.qmax);
                const ClipWindow actual
                    = clip_window_of (entry.multiplier, entry.zero_point, entry.qmin,
                                      entry.qmax);
                const ClipWindow expected
                    = r_clip_window (entry.multiplier, entry.zero_point, entry.qmin,
                                     entry.qmax);
                CHECK (actual._M_valid == expected._M_valid);
                CHECK (actual._M_low == expected._M_low);
                CHECK (actual._M_high == expected._M_high);
                CHECK (actual._M_code_low == expected._M_code_low);
                CHECK (actual._M_code_high == expected._M_code_high);
                CHECK (actual._M_qmin == entry.qmin);
                CHECK (actual._M_qmax == entry.qmax);

                /* The closed interval is exact on both edges. */
                const int32_t probes[6] = { actual._M_low - 1, actual._M_low,
                                            actual._M_low + 1, actual._M_high - 1,
                                            actual._M_high, actual._M_high + 1 };
                const bool should_clip[6]
                    = { true, false, false, false, false, true };
                for (uint32_t probe = 0; probe < 6u; ++probe)
                    {
                        const int32_t value = probes[probe];
                        if (value < R_Q15_MIN || value > R_Q15_MAX)
                            continue;
                        bool clipped = false;
                        bool expected_clipped = false;
                        const int32_t back
                            = fq_dequantized_value (value, 1, entry.multiplier,
                                                    entry.zero_point, entry.qmin,
                                                    entry.qmax, clipped);
                        CHECK (back
                               == r_dequantized_value (value, 1, entry.multiplier,
                                                       entry.zero_point, entry.qmin,
                                                       entry.qmax, expected_clipped));
                        CHECK (clipped == expected_clipped);
                        CHECK (clipped == should_clip[probe]);
                        const bool inside = value >= actual._M_low
                                            && value <= actual._M_high;
                        CHECK (inside == !clipped);
                    }
            }
        }

    SECTION ("a non-positive multiplier builds an invalid window")
    {
        const ClipWindow zero = clip_window_of (0, 0, -127, 127);
        CHECK (zero._M_valid == 0u);
        CHECK (zero._M_low == R_INT32_MIN);
        CHECK (zero._M_high == R_INT32_MAX);
        CHECK (zero._M_code_low == 0);
        CHECK (zero._M_code_high == 0);
        const ClipWindow negative = clip_window_of (-4, 0, -127, 127);
        CHECK (negative._M_valid == 0u);
        CHECK (negative._M_qmin == -127);
        CHECK (negative._M_qmax == 127);
    }

    SECTION ("the dividers and the narrowing are exact")
    {
        struct DivCase
        {
            int64_t numerator;
            int64_t denominator;
            int64_t floor;
            int64_t ceil;
        };
        const DivCase divs[] = {
            { 7, 2, 3, 4 },   { -7, 2, -4, -3 }, { 8, 2, 4, 4 },
            { -8, 2, -4, -4 }, { 0, 5, 0, 0 },   { 1, 5, 0, 1 },
            { -1, 5, -1, 0 },  { 4177919, 127, 32897, 32897 },
            { -4177920, 127, -32898, -32897 },
            { 1073741824, 258, 4161790, 4161791 },
            { -16384, 328, -50, -49 },
        };
        for (const DivCase &entry : divs)
            {
                CAPTURE (entry.numerator, entry.denominator);
                CHECK (fq_floor_div (entry.numerator, entry.denominator)
                       == r_floor_div (entry.numerator, entry.denominator));
                CHECK (fq_ceil_div (entry.numerator, entry.denominator)
                       == r_ceil_div (entry.numerator, entry.denominator));
                CHECK (fq_floor_div (entry.numerator, entry.denominator) == entry.floor);
                CHECK (fq_ceil_div (entry.numerator, entry.denominator) == entry.ceil);
            }
        CHECK (fq_narrow_i32_sat (static_cast<int64_t> (R_INT32_MAX) + 1000)
               == R_INT32_MAX);
        CHECK (fq_narrow_i32_sat (static_cast<int64_t> (R_INT32_MIN) - 1000)
               == R_INT32_MIN);
        CHECK (fq_narrow_i32_sat (0) == 0);
        CHECK (fq_narrow_i32_sat (-1) == -1);
        CHECK (fq_narrow_i32_sat (R_INT32_MAX) == R_INT32_MAX);
        CHECK (fq_narrow_i32_sat (R_INT32_MIN) == R_INT32_MIN);
    }

    SECTION ("the code helpers match the documented rules")
    {
        CHECK (fq_pre_clamp_code (12737, 328, 0) == 127);
        CHECK (fq_pre_clamp_code (12738, 328, 0) == 128);
        CHECK (fq_pre_clamp_code (-12737, 328, 0) == -127);
        CHECK (fq_pre_clamp_code (-12738, 328, 0) == -128);
        CHECK (fq_pre_clamp_code (1000, 1928, -3) == r_pre_clamp_code (1000, 1928, -3));
        CHECK (fq_dequantize_q15 (127, 258, 0) == 32766);
        CHECK (fq_dequantize_q15 (-127, 258, 0) == -32766);
        CHECK (fq_dequantize_q15 (0, 258, 0) == 0);
        CHECK (fq_dequantize_q15 (0, 258, 5) == -1290);
        CHECK (fq_dequantize_q15 (127, 32767, 0) == R_Q15_MAX);
        CHECK (fq_dequantize_q15 (127, 30000, -127) == R_Q15_MAX);
        CHECK (fq_dequantize_q15 (-127, 30000, 127) == R_Q15_MIN);
        /* Round half up: an exact .5 tie goes to +infinity, so a negative tie
           rounds toward zero -- the same rule `_S_round_shift` implements. */
        CHECK (fq_pre_clamp_code (16384, 1, 0) == 1);
        CHECK (fq_pre_clamp_code (-16384, 1, 0) == 0);
        CHECK (fq_pre_clamp_code (16383, 1, 0) == 0);
        CHECK (fq_pre_clamp_code (-16385, 1, 0) == -1);
        CHECK (fq_pre_clamp_code (163, 100, 0) == r_pre_clamp_code (163, 100, 0));
        CHECK (fq_pre_clamp_code (163, 100, 0) == 0);
        CHECK (fq_pre_clamp_code (-163, 100, 0) == 0);
        CHECK (fq_pre_clamp_code (0, 100, -5) == -5);
    }
}

TEST_CASE ("fake-quant clip window table reports capacity and layout errors",
           "[mllee][train][fakequant]")
{
    const int32_t scales[2] = { 258, 516 };
    const int32_t mults[2] = { 127, 64 };
    const FakeQuantParams per_tensor = FakeQuantParams::symmetric (scales, mults);
    const FakeQuantParams two_channels
        = FakeQuantParams::symmetric (scales, mults).on_axis (0u, 2u);
    ClipWindow windows[4];

    SECTION ("one entry per channel, hoisted once")
    {
        REQUIRE (status_ok (fake_quant_clip_windows (two_channels, windows, 4u)));
        const ClipWindow first = r_clip_window (127, 0, -127, 127);
        const ClipWindow second = r_clip_window (64, 0, -127, 127);
        CHECK (windows[0]._M_low == first._M_low);
        CHECK (windows[0]._M_high == first._M_high);
        CHECK (windows[1]._M_low == second._M_low);
        CHECK (windows[1]._M_high == second._M_high);
        CHECK (windows[0]._M_valid == 1u);
        CHECK (windows[1]._M_valid == 1u);
    }

    SECTION ("invalid parameters, null storage and short capacity")
    {
        CHECK (fake_quant_clip_windows (per_tensor, nullptr, 4u) == Status::BAD_LAYOUT);
        FakeQuantParams broken = per_tensor;
        broken._M_scale_q15 = nullptr;
        CHECK (broken.validate () == Status::BAD_LAYOUT);
        CHECK (fake_quant_clip_windows (broken, windows, 4u) == Status::BAD_LAYOUT);
        CHECK (fake_quant_clip_windows (two_channels, windows, 1u)
               == Status::ARENA_OVERFLOW);
        CHECK (fake_quant_clip_windows (two_channels, windows, 2u) == Status::OK);
        CHECK (status_ok (fake_quant_clip_windows (per_tensor, windows, 1u)));
    }
}

TEST_CASE ("fake-quant STE passes the gradient exactly inside the window",
           "[mllee][train][fakequant]")
{
    alignas (8) int16_t values[16];
    alignas (8) int16_t grad_out[16];
    alignas (8) int16_t grad_in[16];

    const int32_t scale = 100;
    const int32_t multiplier = 328;
    const ClipWindow window = r_clip_window (multiplier, 0, -127, 127);
    CHECK (window._M_low == -12737);
    CHECK (window._M_high == 12737);

    const int32_t pattern[16] = { -12738, -12737, -12736, -12735, -20000, 0, 12736,
                                  12737, 12738, 12800, 32767, -32768, 5000, -5000,
                                  -100, 100 };
    for (uint32_t i = 0; i < 16u; ++i)
        {
            values[i] = static_cast<int16_t> (pattern[i]);
            grad_out[i] = static_cast<int16_t> (100 + static_cast<int32_t> (i) * 10);
        }

    const FakeQuantParams params = FakeQuantParams::symmetric (&scale, &multiplier);
    FakeQuantLayout layout;
    REQUIRE (status_ok (fake_quant_layout_flat (16u, 1u, layout)));
    ClipWindow windows[2];
    REQUIRE (status_ok (fake_quant_clip_windows (params, windows, 2u)));

    SECTION ("identity inside, zero beyond, counted exactly")
    {
        CHECK (windows[0]._M_low == window._M_low);
        CHECK (windows[0]._M_high == window._M_high);
        const uint32_t zeroed = fake_quant_backward_ste (values, grad_out, grad_in,
                                                        windows, params, layout);
        uint32_t ref_zeroed = 0u;
        for (uint32_t i = 0; i < 16u; ++i)
            {
                const bool keep
                    = values[i] >= window._M_low && values[i] <= window._M_high;
                CHECK (grad_in[i] == (keep ? grad_out[i] : 0));
                if (!keep)
                    ++ref_zeroed;
            }
        CHECK (zeroed == ref_zeroed);
        CHECK (zeroed == 6u);
    }

    SECTION ("the STE gate equals the forward clip flag for every lane")
    {
        for (uint32_t i = 0; i < 16u; ++i)
            {
                bool clipped = false;
                bool reference_clipped = false;
                const int32_t back = fq_dequantized_value (
                    values[i], scale, multiplier, 0, -127, 127, clipped);
                const int32_t expected = r_dequantized_value (
                    values[i], scale, multiplier, 0, -127, 127, reference_clipped);
                CHECK (back == expected);
                CHECK (clipped == reference_clipped);
                const bool keep
                    = values[i] >= window._M_low && values[i] <= window._M_high;
                CHECK (clipped == !keep);
            }
    }

    SECTION ("null arguments produce no work")
    {
        CHECK (fake_quant_backward_ste (nullptr, grad_out, grad_in, windows, params,
                                        layout)
               == 0u);
        CHECK (fake_quant_backward_ste (values, nullptr, grad_in, windows, params,
                                        layout)
               == 0u);
        CHECK (fake_quant_backward_ste (values, grad_out, nullptr, windows, params,
                                        layout)
               == 0u);
        CHECK (fake_quant_backward_ste (values, grad_out, grad_in, nullptr, params,
                                        layout)
               == 0u);
        FakeQuantLayout stale;
        CHECK (fake_quant_backward_ste (values, grad_out, grad_in, windows, params,
                                        stale)
               == 0u);
        const int32_t three_scales[3] = { 100, 100, 100 };
        const int32_t three_mults[3] = { 328, 328, 328 };
        const FakeQuantParams wide
            = FakeQuantParams::symmetric (three_scales, three_mults).on_axis (0u, 3u);
        CHECK (fake_quant_backward_ste (values, grad_out, grad_in, windows, wide,
                                        layout)
               == 0u);
    }

    SECTION ("the accumulating form gates then saturates through the lane add")
    {
        alignas (8) int16_t scratch[16];
        alignas (8) int16_t accumulated[16];
        alignas (8) int16_t before[16];
        for (uint32_t i = 0; i < 16u; ++i)
            accumulated[i] = 32700;
        for (uint32_t i = 0; i < 16u; ++i)
            before[i] = accumulated[i];

        uint32_t zeroed = 0u;
        uint32_t saturated = 0u;
        REQUIRE (status_ok (fake_quant_backward_ste_accumulate (
                                values, grad_out, accumulated, scratch, 16u, windows,
                                params, layout, zeroed, saturated)));
        uint32_t ref_zeroed = 0u;
        uint32_t ref_saturated = 0u;
        for (uint32_t i = 0; i < 16u; ++i)
            {
                const bool keep
                    = values[i] >= window._M_low && values[i] <= window._M_high;
                const int32_t gated = keep ? grad_out[i] : 0;
                if (!keep)
                    ++ref_zeroed;
                const int32_t total = before[i] + gated;
                ref_saturated += total > R_Q15_MAX || total < R_Q15_MIN ? 1u : 0u;
                CHECK (accumulated[i] == r_i16_sat (total));
            }
        CHECK (zeroed == ref_zeroed);
        CHECK (saturated == ref_saturated);
        CHECK (zeroed == 6u);
        CHECK (saturated == 10u);
    }

    SECTION ("the accumulating form checks its arguments")
    {
        alignas (8) int16_t scratch[16];
        alignas (8) int16_t accumulated[16];
        uint32_t zeroed = 7u;
        uint32_t saturated = 7u;
        CHECK (fake_quant_backward_ste_accumulate (nullptr, grad_out, accumulated,
                                                   scratch, 16u, windows, params,
                                                   layout, zeroed, saturated)
               == Status::BAD_LAYOUT);
        CHECK (fake_quant_backward_ste_accumulate (values, grad_out, accumulated,
                                                   nullptr, 16u, windows, params,
                                                   layout, zeroed, saturated)
               == Status::BAD_LAYOUT);
        CHECK (fake_quant_backward_ste_accumulate (values, nullptr, accumulated,
                                                   scratch, 16u, windows, params,
                                                   layout, zeroed, saturated)
               == Status::BAD_LAYOUT);
        FakeQuantLayout stale;
        CHECK (fake_quant_backward_ste_accumulate (values, grad_out, accumulated,
                                                   scratch, 16u, windows, params,
                                                   stale, zeroed, saturated)
               == Status::BAD_LAYOUT);
        CHECK (fake_quant_backward_ste_accumulate (values, grad_out, accumulated,
                                                   scratch, 15u, windows, params,
                                                   layout, zeroed, saturated)
               == Status::ARENA_OVERFLOW);
        CHECK (fake_quant_backward_ste_accumulate (values + 1, grad_out, accumulated,
                                                   scratch, 16u, windows, params,
                                                   layout, zeroed, saturated)
               == Status::BAD_LAYOUT);
        const int32_t wide_scales[3] = { 100, 100, 100 };
        const int32_t wide_mults[3] = { 328, 328, 328 };
        const FakeQuantParams wide
            = FakeQuantParams::symmetric (wide_scales, wide_mults).on_axis (0u, 3u);
        CHECK (fake_quant_backward_ste_accumulate (values, grad_out, accumulated,
                                                   scratch, 16u, windows, wide,
                                                   layout, zeroed, saturated)
               == Status::ARENA_OVERFLOW);
    }
}

TEST_CASE ("fake-quant STE routes the window per channel",
           "[mllee][train][fakequant]")
{
    /* shape (2,3,4) along axis 1: three channels of four lanes, two blocks. */
    const int32_t scales[3] = { 100, 50, 25 };
    const int32_t mults[3] = { 328, 656, 1312 };
    const FakeQuantParams params
        = FakeQuantParams::symmetric (scales, mults).on_axis (1u, 3u);
    FakeQuantLayout layout;
    REQUIRE (status_ok (fake_quant_layout (Shape::tensor3 (2, 3, 4), params, layout)));
    CHECK (layout._M_run == 4u);
    CHECK (layout._M_outer == 2u);

    ClipWindow windows[3];
    REQUIRE (status_ok (fake_quant_clip_windows (params, windows, 3u)));
    for (uint32_t channel = 0; channel < 3u; ++channel)
        {
            const ClipWindow expected = r_clip_window (mults[channel], 0, -127, 127);
            CHECK (windows[channel]._M_low == expected._M_low);
            CHECK (windows[channel]._M_high == expected._M_high);
        }
    /* Three different rails, so the same value gates differently per channel. */
    CHECK (windows[0]._M_high > windows[1]._M_high);
    CHECK (windows[1]._M_high > windows[2]._M_high);

    alignas (8) int16_t values[24];
    alignas (8) int16_t grad_out[24];
    alignas (8) int16_t grad_in[24];
    const int32_t offsets[4] = { -1, 0, 1, 4000 };
    uint32_t index = 0u;
    for (uint32_t block = 0; block < 2u; ++block)
        for (uint32_t channel = 0; channel < 3u; ++channel)
            for (uint32_t lane = 0; lane < 4u; ++lane, ++index)
                {
                    const int32_t value = windows[channel]._M_high + offsets[lane];
                    values[index] = static_cast<int16_t> (r_q15_clamp (value));
                    grad_out[index] = static_cast<int16_t> (index + 1u);
                }

    SECTION ("each channel's lanes are gated by its own window")
    {
        const uint32_t zeroed = fake_quant_backward_ste (values, grad_out, grad_in,
                                                         windows, params, layout);
        uint32_t ref_zeroed = 0u;
        index = 0u;
        for (uint32_t block = 0; block < 2u; ++block)
            for (uint32_t channel = 0; channel < 3u; ++channel)
                for (uint32_t lane = 0; lane < 4u; ++lane, ++index)
                    {
                        const ClipWindow expected
                            = r_clip_window (mults[channel], 0, -127, 127);
                        const bool keep = values[index] >= expected._M_low
                                          && values[index] <= expected._M_high;
                        CHECK (grad_in[index] == (keep ? grad_out[index] : 0));
                        if (!keep)
                            ++ref_zeroed;
                    }
        CHECK (zeroed == ref_zeroed);
        CHECK (zeroed == 12u);
    }

    SECTION ("the forward clip count matches the same per-channel gates")
    {
        alignas (8) int8_t codes[24];
        alignas (8) int16_t dequantized[24];
        uint32_t clipped = 0u;
        REQUIRE (status_ok (fake_quant_forward (values, codes, dequantized, params,
                                                layout, clipped)));
        index = 0u;
        for (uint32_t block = 0; block < 2u; ++block)
            for (uint32_t channel = 0; channel < 3u; ++channel)
                for (uint32_t lane = 0; lane < 4u; ++lane, ++index)
                    {
                        const int32_t raw
                            = r_pre_clamp_code (values[index], mults[channel], 0);
                        CHECK (codes[index] == r_i8 (raw));
                        CHECK (dequantized[index]
                               == r_i16_sat (r_dequantize_q15 (r_clamp_i8 (raw),
                                                               scales[channel], 0)));
                    }
        CHECK (clipped == 12u);
    }

    SECTION ("one value, two channels, two verdicts")
    {
        /* A flat two-channel grid with rails +-12737 and +-6368: the very same
           lane is representable on one channel and clipped on the other. */
        const int32_t pair_scales[2] = { 100, 50 };
        const int32_t pair_mults[2] = { 328, 656 };
        const FakeQuantParams pair_params
            = FakeQuantParams::symmetric (pair_scales, pair_mults).on_axis (0u, 2u);
        ClipWindow pair_windows[2];
        REQUIRE (status_ok (fake_quant_clip_windows (pair_params, pair_windows, 2u)));
        CHECK (pair_windows[0]._M_high == 12737);
        CHECK (pair_windows[1]._M_high == r_clip_window (656, 0, -127, 127)._M_high);
        CHECK (pair_windows[1]._M_high == 6368);
        CHECK (pair_windows[1]._M_low == -6368);

        FakeQuantLayout flat;
        REQUIRE (status_ok (fake_quant_layout_flat (2u, 2u, flat)));
        CHECK (flat._M_run == 1u);
        CHECK (flat._M_group == 2u);
        CHECK (flat._M_outer == 1u);
        CHECK (flat.base (0u, 0u) == 0u);
        CHECK (flat.base (1u, 0u) == 1u);
        alignas (8) int16_t pair[2] = { 10368, 10368 };
        alignas (8) int16_t incoming[2] = { 4000, 4000 };
        alignas (8) int16_t routed[2] = { 0, 0 };
        const uint32_t zeroed = fake_quant_backward_ste (pair, incoming, routed,
                                                         pair_windows, pair_params,
                                                         flat);
        CHECK (routed[0] == 4000);
        CHECK (routed[1] == 0);
        CHECK (zeroed == 1u);

        alignas (8) int8_t codes[2] = { 0, 0 };
        alignas (8) int16_t dequantized[2] = { 0, 0 };
        uint32_t clipped = 0u;
        REQUIRE (status_ok (fake_quant_forward (pair, codes, dequantized, pair_params,
                                                flat, clipped)));
        const int32_t raw0 = r_pre_clamp_code (10368, 328, 0);
        const int32_t raw1 = r_pre_clamp_code (10368, 656, 0);
        CHECK (raw0 == 104);
        CHECK (raw1 == 208);
        CHECK (codes[0] == r_i8 (raw0));
        CHECK (codes[1] == r_i8 (raw1));
        CHECK (codes[1] == 127);
        CHECK (dequantized[0] == r_i16_sat (r_dequantize_q15 (r_clamp_i8 (raw0), 100,
                                                              0)));
        CHECK (dequantized[1] == r_i16_sat (r_dequantize_q15 (r_clamp_i8 (raw1), 50, 0)));
        CHECK (dequantized[0] == 10400);
        CHECK (dequantized[1] == 6350);
        CHECK (clipped == 1u);
    }
}

TEST_CASE ("min/max accumulator is order independent as a stream",
           "[mllee][train][fakequant]")
{
    const int16_t samples[] = { 1200, -4000, 32767, -32768, 0, 7, -258, 516, -516,
                                30000, -30000, 258, -1, 32766 };
    const uint32_t count = static_cast<uint32_t> (sizeof (samples) / sizeof (int16_t));

    SECTION ("one value at a time equals any other feed order")
    {
        MinMaxAccumulator streamed;
        for (uint32_t i = 0; i < count; ++i)
            observe_lanes (streamed, &samples[i], 1u);

        RefMinMax reference;
        for (uint32_t i = 0; i < count; ++i)
            reference.feed (samples[i]);
        CHECK (same_accumulator (streamed, reference.as_accumulator ()));
        CHECK (streamed._M_min_q15 == -32768);
        CHECK (streamed._M_max_q15 == 32767);
        CHECK (streamed._M_abs_max_q15 == 32768);
        CHECK (streamed._M_count == static_cast<uint16_t> (count));
        CHECK (streamed._M_seeded == 1u);

        alignas (8) int16_t reversed[16];
        for (uint32_t i = 0; i < count; ++i)
            reversed[i] = samples[count - 1u - i];
        MinMaxAccumulator backwards;
        observe_lanes (backwards, reversed, count);
        CHECK (same_accumulator (backwards, streamed));

        MinMaxAccumulator chunks;
        observe_lanes (chunks, samples, 5u);
        observe_lanes (chunks, samples + 5, 4u);
        observe_lanes (chunks, samples + 9, count - 9u);
        CHECK (same_accumulator (chunks, streamed));

        MinMaxAccumulator untouched;
        observe_lanes (untouched, nullptr, 4u);
        observe_lanes (untouched, samples, 0u);
        CHECK (untouched._M_seeded == 0u);
        CHECK (untouched._M_count == 0u);
    }

    SECTION ("the L1 accumulator saturates and reset clears")
    {
        MinMaxAccumulator accumulator;
        const int16_t big[4] = { 30000, 30000, 30000, 30000 };
        observe_lanes (accumulator, big, 4u);
        CHECK (accumulator._M_sum_q15 == R_Q15_MAX);
        CHECK (accumulator._M_abs_max_q15 == 30000);
        CHECK (accumulator._M_min_q15 == 30000);
        min_max_reset (&accumulator, 1u);
        CHECK (accumulator._M_min_q15 == 0);
        CHECK (accumulator._M_max_q15 == 0);
        CHECK (accumulator._M_abs_max_q15 == 0);
        CHECK (accumulator._M_sum_q15 == 0);
        CHECK (accumulator._M_count == 0u);
        CHECK (accumulator._M_seeded == 0u);
        min_max_reset (nullptr, 4u);

        /* An all-negative stream is not silently clipped to zero. */
        MinMaxAccumulator negative;
        const int16_t lanes[3] = { -1000, -2000, -3000 };
        observe_lanes (negative, lanes, 3u);
        CHECK (negative._M_min_q15 == -3000);
        CHECK (negative._M_max_q15 == -1000);
        CHECK (negative._M_abs_max_q15 == 3000);
        CHECK (negative._M_sum_q15 == 6000);

        /* Two accumulator arrays reset together. */
        MinMaxAccumulator pair[2];
        observe_lanes (pair[0], lanes, 3u);
        observe_lanes (pair[1], lanes, 3u);
        min_max_reset (pair, 2u);
        CHECK (pair[1]._M_seeded == 0u);
    }

    SECTION ("codes widen through the 258 identity step")
    {
        const int8_t codes[6] = { 127, -127, 0, 5, -5, 100 };
        MinMaxAccumulator observed;
        observe_codes (observed, codes, 6u);
        CHECK (observed._M_min_q15 == -32766);
        CHECK (observed._M_max_q15 == 32766);
        CHECK (observed._M_abs_max_q15 == 32766);
        int32_t ref_sum = 0;
        for (const int8_t code : codes)
            ref_sum = r_clamp_sum_q15 (
                static_cast<int64_t> (ref_sum)
                + (code < 0 ? -static_cast<int64_t> (r_code_to_q15 (code))
                            : static_cast<int64_t> (r_code_to_q15 (code))));
        CHECK (observed._M_sum_q15 == ref_sum);
        MinMaxAccumulator nothing;
        observe_codes (nothing, nullptr, 3u);
        CHECK (nothing._M_seeded == 0u);
    }

    SECTION ("the count saturates at one full uint16 of samples")
    {
        MinMaxAccumulator accumulator;
        accumulator._M_count = 0xFFFFu;
        const int16_t one[1] = { 5 };
        observe_lanes (accumulator, one, 1u);
        CHECK (accumulator._M_count == 0xFFFFu);
    }
}

TEST_CASE ("axiswise observation splits a tensor by channel",
           "[mllee][train][fakequant]")
{
    const int32_t scales[3] = { 258, 516, 1024 };
    const int32_t mults[3] = { 127, 64, 32 };
    const FakeQuantParams params
        = FakeQuantParams::symmetric (scales, mults).on_axis (1u, 3u);
    FakeQuantLayout layout;
    REQUIRE (status_ok (fake_quant_layout (Shape::tensor3 (2, 3, 4), params, layout)));

    alignas (8) int16_t values[24];
    alignas (8) int8_t codes[24];
    for (uint32_t i = 0; i < 24u; ++i)
        {
            const uint32_t channel = (i % 12u) / 4u;
            values[i] = static_cast<int16_t> ((static_cast<int32_t> (i) - 12)
                                              * static_cast<int32_t> (channel + 1));
            codes[i] = static_cast<int8_t> (r_clamp_i8 (static_cast<int32_t> (i) - 12));
        }

    SECTION ("each accumulator only ever sees its own runs")
    {
        MinMaxAccumulator observed[3];
        REQUIRE (status_ok (observe_lanes_axiswise (observed, values, layout)));
        for (uint32_t channel = 0; channel < 3u; ++channel)
            {
                RefMinMax reference;
                for (uint32_t block = 0; block < layout._M_outer; ++block)
                    for (uint32_t lane = 0; lane < layout._M_run; ++lane)
                        reference.feed (
                            values[layout.base (channel, block) + lane]);
                CHECK (same_accumulator (observed[channel],
                                         reference.as_accumulator ()));
            }
        CHECK (observe_lanes_axiswise (nullptr, values, layout) == Status::BAD_LAYOUT);
        CHECK (observe_lanes_axiswise (observed, nullptr, layout) == Status::BAD_LAYOUT);
        FakeQuantLayout stale;
        CHECK (observe_lanes_axiswise (observed, values, stale) == Status::BAD_LAYOUT);

        MinMaxAccumulator code_observed[3];
        REQUIRE (status_ok (observe_codes_axiswise (code_observed, codes, layout)));
        for (uint32_t channel = 0; channel < 3u; ++channel)
            {
                RefMinMax reference;
                for (uint32_t block = 0; block < layout._M_outer; ++block)
                    for (uint32_t lane = 0; lane < layout._M_run; ++lane)
                        reference.feed (r_code_to_q15 (
                            codes[layout.base (channel, block) + lane]));
                CHECK (same_accumulator (code_observed[channel],
                                         reference.as_accumulator ()));
            }
        CHECK (observe_codes_axiswise (code_observed, nullptr, layout)
               == Status::BAD_LAYOUT);
        CHECK (observe_codes_axiswise (nullptr, codes, layout) == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("scale helpers derive grids from an observation",
           "[mllee][train][fakequant]")
{
    SECTION ("the smallest symmetric step that reaches the extreme")
    {
        CHECK (fq_symmetric_scale (0) == R_CODE_STEP);
        CHECK (fq_symmetric_scale (-5) == R_CODE_STEP);
        CHECK (fq_symmetric_scale (1) == 1);
        CHECK (fq_symmetric_scale (127) == 1);
        CHECK (fq_symmetric_scale (128) == 2);
        CHECK (fq_symmetric_scale (254) == 2);
        CHECK (fq_symmetric_scale (32766) == 258);
        CHECK (fq_symmetric_scale (32767) == 259);
        CHECK (fq_symmetric_scale (5000000) == R_Q15_MAX);
        for (int32_t abs_max = 1; abs_max < 4000; abs_max += 7)
            {
                const int32_t scale = fq_symmetric_scale (abs_max);
                CHECK (scale == r_symmetric_scale (abs_max));
                CHECK (127 * scale >= abs_max);
                CHECK (127 * (scale - 1) < abs_max);
            }
    }

    SECTION ("the reciprocal multiplier of a code step")
    {
        /* train_fake_quant.hpp:48 documents `round(2^15 / scale_q15)`, which is
           what calib_multiplier_from_scale computes (up to one code); this helper
           divides 2^30 by the step instead, i.e. 2^15 times the documented
           reciprocal (reported as a defect). */
        CHECK (fq_multiplier_from_scale (258) == 4161790);
        CHECK (fq_multiplier_from_scale (0) == 0);
        CHECK (fq_multiplier_from_scale (-3) == 0);
        CHECK (fq_multiplier_from_scale (1) == 1073741824);
        for (const int32_t scale : { 2, 4, 100, 258, 32767, 17, 1024 })
            {
                CAPTURE (scale);
                CHECK (fq_multiplier_from_scale (scale)
                       == r_sat_i32 (r_multiplier_implemented (scale)));
                /* The documented reciprocal would be 2^15 / scale, i.e. 32768
                   times smaller than what the helper returns; pin the ratio so
                   the deviation stays visible. */
                /* The implemented reciprocal is the documented one scaled by 2^15:
                   the helper divides 2^30 by the step instead of 2^15 (reported). */
                CHECK (r_multiplier_implemented (scale) / R_Q15_ONE
                       >= static_cast<int64_t> (r_multiplier_documented (scale)) - 1);
                CHECK (r_multiplier_implemented (scale) / R_Q15_ONE
                       <= static_cast<int64_t> (r_multiplier_documented (scale)) + 1);
            }
    }

    SECTION ("scale_from_observed reads the abs max")
    {
        MinMaxAccumulator observed;
        observed._M_abs_max_q15 = 32766;
        CHECK (scale_from_observed (observed) == 258);
        observed._M_abs_max_q15 = 0;
        CHECK (scale_from_observed (observed) == R_CODE_STEP);
        observed._M_abs_max_q15 = 1000;
        CHECK (scale_from_observed (observed) == r_symmetric_scale (1000));
        observed._M_abs_max_q15 = 1;
        CHECK (scale_from_observed (observed) == 1);
    }
}

TEST_CASE ("asymmetric grid derivation clamps the zero point",
           "[mllee][train][fakequant]")
{
    SECTION ("a degenerate observation keeps the identity grid")
    {
        MinMaxAccumulator observed;
        int32_t scale = 0;
        int8_t zero_point = 42;
        REQUIRE (status_ok (scale_zero_from_observed (observed, scale, zero_point)));
        CHECK (scale == R_CODE_STEP);
        CHECK (zero_point == 0);
        observed._M_min_q15 = 100;
        observed._M_max_q15 = 100;
        REQUIRE (status_ok (scale_zero_from_observed (observed, scale, zero_point)));
        CHECK (scale == R_CODE_STEP);
        CHECK (zero_point == 0);
        observed._M_min_q15 = 500;
        observed._M_max_q15 = 10;
        REQUIRE (status_ok (scale_zero_from_observed (observed, scale, zero_point)));
        CHECK (scale == R_CODE_STEP);
        CHECK (zero_point == 0);
    }

    SECTION ("a non-negative span puts the low end on code -127")
    {
        /* With min == 0 the reciprocal never matters, so the documented rule
           `zp = clamp(-127 - round(min / scale), -127, 127)` is exact here.  The
           step is the span divided over the 254 codes the asymmetric grid has. */
        MinMaxAccumulator observed;
        observed._M_min_q15 = 0;
        int32_t scale = 0;
        int8_t zero_point = 0;
        for (int32_t high = 1; high <= 32767; high += 97)
            {
                observed._M_max_q15 = high;
                REQUIRE (status_ok (scale_zero_from_observed (observed, scale,
                                                              zero_point)));
                CAPTURE (high);
                CHECK (scale == static_cast<int32_t> (r_ceil_div (high, 254)));
                CHECK (scale >= 1);
                CHECK (scale <= R_Q15_MAX);
                CHECK (254 * static_cast<int64_t> (scale) >= high);
                CHECK (zero_point == -127);
            }
        observed._M_max_q15 = 32767;
        REQUIRE (status_ok (scale_zero_from_observed (observed, scale, zero_point)));
        CHECK (scale == static_cast<int32_t> (r_ceil_div (32767, 254)));
        CHECK (zero_point == -127);
        observed._M_max_q15 = 127;
        REQUIRE (status_ok (scale_zero_from_observed (observed, scale, zero_point)));
        CHECK (scale == 1);
        CHECK (zero_point == -127);
        /* Half an LSB of head-room is allowed, a full code more is not. */
        observed._M_max_q15 = 254 * 129;
        REQUIRE (status_ok (scale_zero_from_observed (observed, scale, zero_point)));
        CHECK (scale == 129);
        observed._M_max_q15 = 254 * 129 + 1;
        REQUIRE (status_ok (scale_zero_from_observed (observed, scale, zero_point)));
        CHECK (scale == 130);
    }

    SECTION ("a span wider than the grid is refused")
    {
        MinMaxAccumulator observed;
        observed._M_min_q15 = -5000000;
        observed._M_max_q15 = 5000000;
        int32_t scale = 0;
        int8_t zero_point = 0;
        CHECK (scale_zero_from_observed (observed, scale, zero_point)
               == Status::OUT_OF_RANGE);
        observed._M_min_q15 = -4200000;
        observed._M_max_q15 = 4200000;
        CHECK (scale_zero_from_observed (observed, scale, zero_point)
               == Status::OUT_OF_RANGE);
        observed._M_min_q15 = -4000000;
        observed._M_max_q15 = 4000000;
        REQUIRE (status_ok (scale_zero_from_observed (observed, scale, zero_point)));
        CHECK (scale == static_cast<int32_t> (r_ceil_div (8000000, 254)));
        CHECK (scale == 31497);
    }

    SECTION ("a negative low end saturates the zero point")
    {
        /* The documented rule would put the low end on code -127.  Because the
           multiplier this function derives is 2^15 too large, the rounded term is
           a huge magnitude and the clamp lands on +127 for every negative
           minimum (reported as a defect). */
        MinMaxAccumulator observed;
        observed._M_min_q15 = -2048;
        observed._M_max_q15 = 2048;
        int32_t scale = 0;
        int8_t zero_point = 0;
        REQUIRE (status_ok (scale_zero_from_observed (observed, scale, zero_point)));
        CHECK (scale == 17);
        CHECK (zero_point == 127);
        observed._M_min_q15 = -16000;
        observed._M_max_q15 = 16000;
        REQUIRE (status_ok (scale_zero_from_observed (observed, scale, zero_point)));
        CHECK (scale == 126);
        CHECK (zero_point == 127);
        observed._M_min_q15 = -1;
        observed._M_max_q15 = 508;
        REQUIRE (status_ok (scale_zero_from_observed (observed, scale, zero_point)));
        CHECK (scale == static_cast<int32_t> (r_ceil_div (509, 254)));
        CHECK (scale == 3);
        CHECK (zero_point == 127);
    }
}

TEST_CASE ("narrow_clip_range tightens the grid to the observation",
           "[mllee][train][fakequant]")
{
    const int32_t scales[2] = { 516, 1032 };
    const int32_t mults[2] = { 64, 32 };
    const FakeQuantParams symmetric
        = FakeQuantParams::symmetric (scales, mults).on_axis (0u, 2u);
    const int8_t zp[2] = { 3, -4 };
    const FakeQuantParams asymmetric
        = FakeQuantParams::asymmetric (scales, mults, zp).on_axis (0u, 2u);

    int32_t out_scale[2] = { 0, 0 };
    int32_t out_mult[2] = { 0, 0 };
    int8_t out_qmin[2] = { 0, 0 };
    int8_t out_qmax[2] = { 0, 0 };
    MinMaxAccumulator observed[2];
    observed[0]._M_abs_max_q15 = 12900;
    observed[0]._M_min_q15 = -12900;
    observed[0]._M_max_q15 = 12900;
    observed[1]._M_abs_max_q15 = 0;

    SECTION ("the symmetric form never re-expands the grid")
    {
        REQUIRE (status_ok (narrow_clip_range (symmetric, observed, 2u, 10000, out_scale,
                                               out_mult, out_qmin, out_qmax)));
        /* 100% of the observed extreme, ceil-div by 127, capped by the input. */
        CHECK (out_scale[0] == r_narrow_symmetric_scale (scales[0], 12900, 10000));
        CHECK (out_scale[0] == 102);
        CHECK (out_scale[0] <= scales[0]);
        CHECK (out_scale[1] == r_narrow_symmetric_scale (scales[1], 0, 10000));
        CHECK (out_scale[1] == R_CODE_STEP);
        CHECK (out_scale[1] <= scales[1]);
        CHECK (out_mult[0] == r_sat_i32 (r_multiplier_implemented (out_scale[0])));
        CHECK (out_mult[1] == r_sat_i32 (r_multiplier_implemented (out_scale[1])));
        /* The reached code uses the same oversized reciprocal as the table, so any
           nonzero extreme lands on the +-127 rails (reported with the scale bug). */
        CHECK (out_qmax[0] == r_narrow_reached_bound (12900, out_scale[0]));
        CHECK (out_qmax[0] == 127);
        CHECK (out_qmin[0] == -127);
        CHECK (out_qmax[1] == r_narrow_reached_bound (0, out_scale[1]));
        CHECK (out_qmax[1] == 1);
        CHECK (out_qmin[1] == -1);

        /* A 50% percentile halves the wanted span before the ceil. */
        REQUIRE (status_ok (narrow_clip_range (symmetric, observed, 2u, 5000, out_scale,
                                               out_mult, out_qmin, out_qmax)));
        CHECK (out_scale[0] == r_narrow_symmetric_scale (scales[0], 12900, 5000));
        CHECK (out_scale[0] == 51);

        /* A still-wider incoming grid is tightened to the observation, and the
           two channels narrow independently of each other. */
        MinMaxAccumulator wide[2];
        wide[0]._M_abs_max_q15 = 30000;
        wide[1]._M_abs_max_q15 = 30000;
        REQUIRE (status_ok (narrow_clip_range (symmetric, wide, 2u, 10000, out_scale,
                                               out_mult, out_qmin, out_qmax)));
        CHECK (out_scale[0] == r_narrow_symmetric_scale (scales[0], 30000, 10000));
        CHECK (out_scale[1] == r_narrow_symmetric_scale (scales[1], 30000, 10000));
        CHECK (out_scale[0] == 237);
        CHECK (out_scale[0] < scales[0]);
        CHECK (out_scale[1] < scales[1]);
        /* Narrowing is monotone: keeping less of the range gives a smaller step. */
        CHECK (out_qmax[0] == r_narrow_reached_bound (30000, out_scale[0]));
        CHECK (out_qmax[0] == 127);
    }

    SECTION ("the asymmetric form keeps the full code range")
    {
        REQUIRE (status_ok (narrow_clip_range (asymmetric, observed, 2u, 10000,
                                               out_scale, out_mult, out_qmin,
                                               out_qmax)));
        CHECK (out_scale[0] == r_narrow_asymmetric_scale (scales[0], -12900, 12900,
                                                          10000));
        CHECK (out_scale[0] == static_cast<int32_t> (r_ceil_div (25800, 254)));
        CHECK (out_scale[0] == 102);
        /* An empty span leaves the incoming step alone: the asymmetric branch
           only reacts to a strictly positive span. */
        CHECK (out_scale[1] == r_narrow_asymmetric_scale (scales[1], 0, 0, 10000));
        CHECK (out_scale[1] == scales[1]);
        CHECK (out_qmin[0] == -127);
        CHECK (out_qmax[0] == 127);
        CHECK (out_qmin[1] == -127);
        CHECK (out_qmax[1] == 127);
        CHECK (out_mult[1] == r_sat_i32 (r_multiplier_implemented (scales[1])));

        /* A degenerate observation keeps the incoming step on both channels. */
        MinMaxAccumulator quiet[2];
        REQUIRE (status_ok (narrow_clip_range (asymmetric, quiet, 2u, 10000, out_scale,
                                               out_mult, out_qmin, out_qmax)));
        CHECK (out_scale[0] == scales[0]);
        CHECK (out_scale[1] == scales[1]);

        /* A one-sided span is divided by the full 254-code travel. */
        MinMaxAccumulator biased[2];
        biased[0]._M_min_q15 = 1000;
        biased[0]._M_max_q15 = 6100;
        REQUIRE (status_ok (narrow_clip_range (asymmetric, biased, 2u, 10000, out_scale,
                                               out_mult, out_qmin, out_qmax)));
        CHECK (out_scale[0] == r_narrow_asymmetric_scale (scales[0], 1000, 6100, 10000));
        CHECK (out_scale[0] == static_cast<int32_t> (r_ceil_div (5100, 254)));
        CHECK (out_scale[0] == 21);
        CHECK (out_scale[1] == scales[1]);
    }

    SECTION ("argument and capacity checks")
    {
        CHECK (narrow_clip_range (symmetric, nullptr, 2u, 10000, out_scale, out_mult,
                                  out_qmin, out_qmax)
               == Status::BAD_LAYOUT);
        CHECK (narrow_clip_range (symmetric, observed, 2u, 10000, nullptr, out_mult,
                                  out_qmin, out_qmax)
               == Status::BAD_LAYOUT);
        CHECK (narrow_clip_range (symmetric, observed, 2u, 10000, out_scale, nullptr,
                                  out_qmin, out_qmax)
               == Status::BAD_LAYOUT);
        CHECK (narrow_clip_range (symmetric, observed, 2u, 10000, out_scale, out_mult,
                                  nullptr, out_qmax)
               == Status::BAD_LAYOUT);
        CHECK (narrow_clip_range (symmetric, observed, 2u, 10000, out_scale, out_mult,
                                  out_qmin, nullptr)
               == Status::BAD_LAYOUT);
        CHECK (narrow_clip_range (symmetric, observed, 1u, 10000, out_scale, out_mult,
                                  out_qmin, out_qmax)
               == Status::ARENA_OVERFLOW);
        CHECK (narrow_clip_range (symmetric, observed, 2u, 0, out_scale, out_mult,
                                  out_qmin, out_qmax)
               == Status::OUT_OF_RANGE);
        CHECK (narrow_clip_range (symmetric, observed, 2u, 10001, out_scale, out_mult,
                                  out_qmin, out_qmax)
               == Status::OUT_OF_RANGE);
        CHECK (narrow_clip_range (symmetric, observed, 2u, -1, out_scale, out_mult,
                                  out_qmin, out_qmax)
               == Status::OUT_OF_RANGE);
        FakeQuantParams broken = symmetric;
        broken._M_channels = 300u;
        CHECK (narrow_clip_range (broken, observed, 2u, 10000, out_scale, out_mult,
                                  out_qmin, out_qmax)
               == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("multiplier tables stay positive or are refused",
           "[mllee][train][fakequant]")
{
    const int32_t scales[3] = { 258, 516, 1032 };
    int32_t mults[3] = { 0, 0, 0 };

    SECTION ("a legal table is rebuilt channel by channel")
    {
        REQUIRE (status_ok (refresh_multiplier_table (scales, mults, 3u)));
        for (uint32_t channel = 0; channel < 3u; ++channel)
            {
                CHECK (mults[channel] > 0);
                CHECK (mults[channel] == r_sat_i32 (r_multiplier_implemented (
                                                         scales[channel])));
            }
        CHECK (mults[0] == 4161790);
    }

    SECTION ("a scale with no reciprocal aborts the table")
    {
        const int32_t broken[2] = { 258, 0 };
        int32_t out[2] = { 0, 0 };
        CHECK (refresh_multiplier_table (broken, out, 2u) == Status::OUT_OF_RANGE);
        CHECK (out[0] == 4161790);
        CHECK (out[1] == 0);
        const int32_t negative[1] = { -7 };
        int32_t single[1] = { 0 };
        CHECK (refresh_multiplier_table (negative, single, 1u) == Status::OUT_OF_RANGE);
        CHECK (single[0] == 0);
    }

    SECTION ("null storage and channel bounds")
    {
        CHECK (refresh_multiplier_table (nullptr, mults, 1u) == Status::BAD_LAYOUT);
        CHECK (refresh_multiplier_table (scales, nullptr, 1u) == Status::BAD_LAYOUT);
        CHECK (refresh_multiplier_table (scales, mults, 0u) == Status::OUT_OF_RANGE);
        CHECK (refresh_multiplier_table (scales, mults, 257u) == Status::OUT_OF_RANGE);
    }

    SECTION ("a refreshed table drives the forward pass")
    {
        int32_t built[3] = { 0, 0, 0 };
        REQUIRE (status_ok (refresh_multiplier_table (scales, built, 3u)));
        const FakeQuantParams params
            = FakeQuantParams::symmetric (scales, built).on_axis (0u, 3u);
        CHECK (status_ok (params.validate ()));
        /* The oversized reciprocal collapses every nonzero value onto a rail, so
           the documented half-LSB property only holds for zero (reported). */
        bool clipped = true;
        const int32_t back = fq_dequantized_value (1, 258, built[0], 0, -127, 127,
                                                   clipped);
        CHECK (back == 32766);
        CHECK (clipped == false);
        const ClipWindow window = clip_window_of (built[0], 0, -127, 127);
        const ClipWindow expected = r_clip_window (built[0], 0, -127, 127);
        CHECK (window._M_low == expected._M_low);
        CHECK (window._M_high == expected._M_high);
        /* The oversized reciprocal makes the grid 127 times steeper, so only the
           three lanes -1, 0 and 1 stay inside the code range at all. */
        CHECK (window._M_low == -1);
        CHECK (window._M_high == 1);
        CHECK (fq_pre_clamp_code (1, built[0], 0) == r_pre_clamp_code (1, built[0], 0));
        CHECK (fq_pre_clamp_code (1, built[0], 0) == 127);
        CHECK (fq_pre_clamp_code (2, built[0], 0) == 254);
    }
}

TEST_CASE ("learnable scale gradient projects onto the residual",
           "[mllee][train][fakequant]")
{
    /* A channel-major grid of two channels of six lanes, with scales whose
       documented reciprocal is exact so the grid points are lossless. */
    const int32_t scales[2] = { 128, 256 };
    const int32_t mults[2] = { r_multiplier_documented (128),
                               r_multiplier_documented (256) };
    const FakeQuantParams params
        = FakeQuantParams::symmetric (scales, mults).on_axis (0u, 2u);
    FakeQuantLayout layout;
    REQUIRE (status_ok (fake_quant_layout_flat (12u, 2u, layout)));

    alignas (8) int16_t values[12];
    alignas (8) int8_t codes[12];
    alignas (8) int16_t grad[12];
    alignas (8) int16_t grad_scale[2];
    for (uint32_t i = 0; i < 12u; ++i)
        {
            const uint32_t channel = i / 6u;
            const int32_t code = static_cast<int32_t> (i) - 5;
            codes[i] = static_cast<int8_t> (r_clamp_i8 (code));
            values[i] = static_cast<int16_t> (code * scales[channel]
                                              + static_cast<int32_t> (i % 3u));
            grad[i] = static_cast<int16_t> (1000 + static_cast<int32_t> (i) * 250);
        }

    SECTION ("per-channel projection accumulated in int64 and narrowed once")
    {
        REQUIRE (status_ok (learnable_scale_backward (values, codes, grad, grad_scale,
                                                      params, layout)));
        int64_t totals[2] = { 0, 0 };
        for (uint32_t i = 0; i < 12u; ++i)
            {
                const uint32_t channel = i / 6u;
                const int32_t dequantized
                    = r_dequantize_q15 (codes[i], scales[channel], 0);
                totals[channel] += static_cast<int64_t> (grad[i])
                                   * (values[i] - dequantized);
            }
        for (uint32_t channel = 0; channel < 2u; ++channel)
            {
                CHECK (grad_scale[channel]
                       == r_i16_sat (r_narrow_sum_q15 (totals[channel])));
                CHECK (grad_scale[channel] >= R_Q15_MIN);
                CHECK (grad_scale[channel] <= R_Q15_MAX);
            }
        CHECK ((totals[0] != 0 || totals[1] != 0));
    }

    SECTION ("argument checks")
    {
        CHECK (learnable_scale_backward (nullptr, codes, grad, grad_scale, params,
                                         layout)
               == Status::BAD_LAYOUT);
        CHECK (learnable_scale_backward (values, nullptr, grad, grad_scale, params,
                                         layout)
               == Status::BAD_LAYOUT);
        CHECK (learnable_scale_backward (values, codes, nullptr, grad_scale, params,
                                         layout)
               == Status::BAD_LAYOUT);
        CHECK (learnable_scale_backward (values, codes, grad, nullptr, params, layout)
               == Status::BAD_LAYOUT);
        FakeQuantLayout stale;
        CHECK (learnable_scale_backward (values, codes, grad, grad_scale, params,
                                         stale)
               == Status::BAD_LAYOUT);
        /* A per-tensor grid against a two-channel layout is a layout mismatch. */
        const FakeQuantParams per_tensor = FakeQuantParams::symmetric (scales, mults);
        CHECK (learnable_scale_backward (values, codes, grad, grad_scale, per_tensor,
                                         layout)
               == Status::BAD_LAYOUT);
        FakeQuantParams broken = params;
        broken._M_multiplier_q15 = nullptr;
        CHECK (learnable_scale_backward (values, codes, grad, grad_scale, broken,
                                         layout)
               == Status::BAD_LAYOUT);
        CHECK (learnable_scale_backward (values, codes, grad, grad_scale, params,
                                         layout)
               == Status::OK);
    }

    SECTION ("the dot form agrees with the fused form")
    {
        alignas (8) int16_t scratch[12];
        alignas (8) int16_t dot_scale[2];
        REQUIRE (status_ok (learnable_scale_backward_dot (values, codes, grad, dot_scale,
                                                          params, layout, scratch,
                                                          12u)));
        REQUIRE (status_ok (learnable_scale_backward (values, codes, grad, grad_scale,
                                                      params, layout)));
        for (uint32_t channel = 0; channel < 2u; ++channel)
            {
                const int32_t difference = dot_scale[channel] - grad_scale[channel];
                CHECK (difference <= 1);
                CHECK (difference >= -1);
            }
        CHECK (learnable_scale_backward_dot (values, codes, grad, dot_scale, params,
                                             layout, scratch, 11u)
               == Status::ARENA_OVERFLOW);
        CHECK (learnable_scale_backward_dot (values, codes, grad, dot_scale, params,
                                             layout, nullptr, 12u)
               == Status::BAD_LAYOUT);
        CHECK (learnable_scale_backward_dot (values + 1, codes, grad, dot_scale, params,
                                             layout, scratch, 12u)
               == Status::BAD_LAYOUT);
        FakeQuantLayout stale;
        CHECK (learnable_scale_backward_dot (values, codes, grad, dot_scale, params,
                                             stale, scratch, 12u)
               == Status::BAD_LAYOUT);
    }

    SECTION ("the scale step is a saturating Q15 update")
    {
        int32_t table[3] = { 1000, 5, 32767 };
        const int16_t gradients[3] = { 32767, -32767, 1000 };
        REQUIRE (status_ok (learnable_scale_step (table, gradients, 3u, 16384)));
        CHECK (table[0] == 1); /* 1000 + q15_mul (-32767, 16384) floors at 1 */
        CHECK (table[1] == 5 + r_q15_mul (32767, 16384));
        CHECK (table[1] == 16389);
        CHECK (table[2] == 32767 - r_q15_mul (1000, 16384));
        CHECK (table[2] == 32267);
        CHECK (learnable_scale_step (nullptr, gradients, 3u, 16384)
               == Status::BAD_LAYOUT);
        CHECK (learnable_scale_step (table, nullptr, 3u, 16384) == Status::BAD_LAYOUT);
        CHECK (learnable_scale_step (table, gradients, 0u, 16384)
               == Status::OUT_OF_RANGE);
        CHECK (learnable_scale_step (table, gradients, 257u, 16384)
               == Status::OUT_OF_RANGE);

        /* A zero gradient is a no-op; a full-scale push saturates at the rail. */
        int32_t steady[2] = { 258, 258 };
        const int16_t none[2] = { 0, 0 };
        REQUIRE (status_ok (learnable_scale_step (steady, none, 2u, 32767)));
        CHECK (steady[0] == 258);
        CHECK (steady[1] == 258);
        const int16_t push[2] = { -32768, -32768 };
        REQUIRE (status_ok (learnable_scale_step (steady, push, 2u, 32767)));
        CHECK (steady[0] == R_Q15_MAX);
        CHECK (steady[1] == R_Q15_MAX);
        int32_t rail[1] = { 1 };
        REQUIRE (status_ok (learnable_scale_step (rail, none, 1u, 32767)));
        CHECK (rail[0] == 1);
    }
}

TEST_CASE ("clip window lanes are six aligned words",
           "[mllee][train][fakequant]")
{
    SECTION ("packing and unpacking round trips")
    {
        const ClipWindow sources[] = {
            r_clip_window (127, 0, -127, 127),
            r_clip_window (328, 7, -127, 127),
            r_clip_window (32, 0, -8, 8),
            r_clip_window (-1, 0, -127, 127),
            r_clip_window (16384, -127, -127, 127),
            r_clip_window (3, 0, -1, 1),
        };
        for (const ClipWindow &source : sources)
            {
                ClipWindowLane lane;
                lane.set_window (source);
                CHECK (lane.valid () == (source._M_valid != 0u));
                CHECK (lane.qmin () == source._M_qmin);
                CHECK (lane.qmax () == source._M_qmax);
                CHECK (lane._M_words[0] == source._M_low);
                CHECK (lane._M_words[1] == source._M_high);
                CHECK (lane._M_words[2] == source._M_code_low);
                CHECK (lane._M_words[3] == source._M_code_high);
                CHECK (lane._M_words[5] == static_cast<int32_t> (source._M_valid));
                const ClipWindow back = lane.as_window ();
                CHECK (back._M_low == source._M_low);
                CHECK (back._M_high == source._M_high);
                CHECK (back._M_code_low == source._M_code_low);
                CHECK (back._M_code_high == source._M_code_high);
                CHECK (back._M_qmin == source._M_qmin);
                CHECK (back._M_qmax == source._M_qmax);
                CHECK (back._M_valid == source._M_valid);
                CHECK (back._M_pad == 0u);
            }
    }

    SECTION ("the two clamp codes share one word")
    {
        ClipWindow window = r_clip_window (127, 0, -127, 127);
        ClipWindowLane lane;
        lane.set_window (window);
        CHECK ((lane._M_words[4] & 0xFF) == 0x81);
        CHECK (((lane._M_words[4] >> 8) & 0xFF) == 0x7F);
        window._M_qmin = -1;
        window._M_qmax = 1;
        lane.set_window (window);
        CHECK (lane.qmin () == -1);
        CHECK (lane.qmax () == 1);
        CHECK (lane._M_words[5] == 1);
        window._M_valid = 0u;
        lane.set_window (window);
        CHECK (lane._M_words[5] == 0);
        CHECK_FALSE (lane.valid ());
        CHECK (lane.as_window ()._M_valid == 0u);
        /* A word that is not zero still reads as valid. */
        lane._M_words[5] = 7;
        CHECK (lane.valid ());
        CHECK (lane.as_window ()._M_valid == 1u);
    }

    SECTION ("a lane array is a plain int32 stream")
    {
        ClipWindowLane lanes[3];
        const int32_t *const words = &lanes[0]._M_words[0];
        CHECK (simd_aligned (words));
        CHECK (sizeof (lanes) == 3u * 24u);
        CHECK ((reinterpret_cast<uintptr_t> (&lanes[1])
                - reinterpret_cast<uintptr_t> (&lanes[0]))
               == 24u);
        for (uint32_t index = 0; index < 3u; ++index)
            CHECK (simd_aligned (&lanes[index]));
    }
}

TEST_CASE ("node state records and rebuilds a recorded layout",
           "[mllee][train][fakequant]")
{
    const int32_t scales[2] = { 258, 516 };
    const int32_t mults[2] = { 127, 64 };
    const FakeQuantParams params
        = FakeQuantParams::symmetric (scales, mults).on_axis (0u, 2u);
    FakeQuantLayout layout;
    REQUIRE (status_ok (fake_quant_layout (Shape::matrix (2, 6), params, layout)));

    FakeQuantNodeState state;
    fake_quant_node_state_set (state, layout);
    CHECK (state._M_run == layout._M_run);
    CHECK (state._M_group == layout._M_group);
    CHECK (state._M_outer == layout._M_outer);
    CHECK (state._M_count == layout._M_count);
    CHECK (state._M_channels == layout._M_channels);
    CHECK (state._M_pad == 0u);

    FakeQuantLayout rebuilt;
    REQUIRE (status_ok (fake_quant_node_layout (state, rebuilt)));
    CHECK (rebuilt._M_valid == 1u);
    CHECK (rebuilt._M_run == layout._M_run);
    CHECK (rebuilt._M_group == layout._M_group);
    CHECK (rebuilt._M_outer == layout._M_outer);
    CHECK (rebuilt._M_count == layout._M_count);
    CHECK (rebuilt._M_channels == layout._M_channels);
    CHECK (rebuilt._M_axis == 0u);
    for (uint32_t channel = 0; channel < 2u; ++channel)
        for (uint32_t block = 0; block < layout._M_outer; ++block)
            CHECK (rebuilt.base (channel, block) == layout.base (channel, block));

    SECTION ("a single channel rebuilds as a per-tensor grid")
    {
        const FakeQuantParams per_tensor = FakeQuantParams::symmetric (scales, mults);
        FakeQuantLayout flat;
        REQUIRE (status_ok (fake_quant_layout (Shape::vector (8), per_tensor, flat)));
        FakeQuantNodeState single;
        fake_quant_node_state_set (single, flat);
        CHECK (single._M_channels == 1u);
        FakeQuantLayout out;
        REQUIRE (status_ok (fake_quant_node_layout (single, out)));
        CHECK (out._M_axis == _S_fq_per_tensor);
        CHECK (out._M_channels == 1u);
        CHECK_FALSE (out.per_channel ());
    }

    SECTION ("a corrupt state is refused")
    {
        FakeQuantNodeState broken;
        FakeQuantLayout out;
        broken._M_run = 4u;
        broken._M_group = 8u;
        broken._M_outer = 1u;
        broken._M_count = 8u;
        broken._M_channels = 0u;
        /* A zero-channel state is not rejected: it rebuilds as a valid layout
           whose loops simply do not run (train_fake_quant.hpp:1178 only checks
           the run, the count and the 16-bit channel ceiling). */
        REQUIRE (status_ok (fake_quant_node_layout (broken, out)));
        CHECK (out._M_valid == 1u);
        CHECK (out._M_channels == 0u);
        CHECK (out._M_count == 8u);
        broken._M_channels = 1u;
        broken._M_run = 0u;
        CHECK (fake_quant_node_layout (broken, out) == Status::BAD_LAYOUT);
        broken._M_run = 4u;
        broken._M_count = 0u;
        CHECK (fake_quant_node_layout (broken, out) == Status::BAD_LAYOUT);
        broken._M_count = 8u;
        broken._M_channels = 65536u;
        CHECK (fake_quant_node_layout (broken, out) == Status::BAD_LAYOUT);
        broken._M_channels = 65535u;
        CHECK (status_ok (fake_quant_node_layout (broken, out)));
        CHECK (out._M_channels == 65535u);
    }
}

TEST_CASE ("a recorded fake-quant node runs the STE through the tape",
           "[mllee][train][fakequant]")
{
    const int32_t scale = 100;
    const int32_t multiplier = 328;
    const FakeQuantParams params = FakeQuantParams::symmetric (&scale, &multiplier);
    const ClipWindow window = r_clip_window (multiplier, 0, -127, 127);

    alignas (8) int8_t input_code[8];
    alignas (8) int16_t input_grad[8];
    alignas (8) int8_t output_code[8];
    alignas (8) int16_t output_grad[8];
    alignas (8) int16_t forward_values[8];
    for (uint32_t i = 0; i < 8u; ++i)
        {
            input_code[i] = 0;
            output_code[i] = 0;
            forward_values[i] = static_cast<int16_t> (window._M_high - 4
                                                      + static_cast<int32_t> (i) * 2);
            output_grad[i] = static_cast<int16_t> (100 + static_cast<int32_t> (i) * 10);
        }
    /* The last two lanes sit above the rail, so they must be excluded. */
    forward_values[6] = static_cast<int16_t> (R_Q15_MAX);
    forward_values[7] = static_cast<int16_t> (window._M_high + 1);
    CHECK (window._M_high == 12737);

    const Trainable input = Trainable::make (input_code, input_grad, Shape::vector (8),
                                             RequiresGrad);
    const Trainable output = Trainable::make (output_code, output_grad,
                                              Shape::vector (8), RequiresGrad);
    FakeQuantLayout layout;
    REQUIRE (status_ok (fake_quant_layout_flat (8u, 1u, layout)));

    alignas (8) Node nodes[4];
    AlignedArena<256> arena;
    Pool pool = arena.pool ();
    Tape tape (nodes, 4u, pool);
    REQUIRE (status_ok (tape.begin ()));

    alignas (8) int8_t loss_code[1] = { 0 };
    alignas (8) int16_t loss_grad[1] = { 0 };
    const Trainable loss
        = Trainable::make (loss_code, loss_grad, Shape::scalar (), 0u);

    SECTION ("pool cost is 2N + 24C + 24 and the walk routes the gradient")
    {
        for (uint32_t i = 0; i < 8u; ++i)
            input_grad[i] = 0;
        REQUIRE (status_ok (record_fake_quant (tape, input, output, forward_values,
                                               params, layout, 7u)));
        CHECK (tape.stats()._M_saved_bytes == 2u * 8u + 24u * 1u + 24u);
        CHECK (tape.stats()._M_saved_bytes == 64u);
        CHECK (tape.node_count () == 1u);
        REQUIRE (tape.node (0) != nullptr);
        CHECK (tape.node (0)->_M_saved_count == 3u);
        CHECK (tape.node (0)->_M_operand_count == 1u);
        CHECK (tape.node (0)->_M_op_id == 7u);
        CHECK (tape.stats()._M_nodes == 1u);

        REQUIRE (status_ok (tape.backward (loss)));
        for (uint32_t i = 0; i < 8u; ++i)
            {
                const bool keep = forward_values[i] >= window._M_low
                                  && forward_values[i] <= window._M_high;
                CHECK (input_grad[i] == (keep ? output_grad[i] : 0));
            }
        CHECK (input_grad[6] == 0);
        CHECK (input_grad[7] == 0);
        CHECK (input_grad[0] == output_grad[0]);
    }

    SECTION ("saturation is counted through the context")
    {
        for (uint32_t i = 0; i < 8u; ++i)
            input_grad[i] = 32700;
        REQUIRE (status_ok (record_fake_quant (tape, input, output, forward_values,
                                               params, layout)));
        REQUIRE (status_ok (tape.backward (loss)));
        uint32_t ref_saturated = 0u;
        for (uint32_t i = 0; i < 8u; ++i)
            {
                const bool keep = forward_values[i] >= window._M_low
                                  && forward_values[i] <= window._M_high;
                const int32_t total = 32700 + (keep ? output_grad[i] : 0);
                CHECK (input_grad[i] == r_i16_sat (total));
                if (total > R_Q15_MAX)
                    ++ref_saturated;
            }
        CHECK (tape.stats()._M_saturation_events == ref_saturated);
        CHECK (ref_saturated == 3u);
    }

    SECTION ("an arena that cannot hold the state reports overflow")
    {
        AlignedArena<48> small;
        Pool tight = small.pool ();
        alignas (8) Node few_nodes[2];
        Tape short_tape (few_nodes, 2u, tight);
        REQUIRE (status_ok (short_tape.begin ()));
        CHECK (record_fake_quant (short_tape, input, output, forward_values, params,
                                  layout)
               == Status::ARENA_OVERFLOW);
        CHECK (pool.stats()._M_capacity <= 256u);
    }

    SECTION ("record rejects malformed arguments")
    {
        CHECK (record_fake_quant (tape, input, output, nullptr, params, layout)
               == Status::BAD_LAYOUT);
        FakeQuantLayout stale;
        CHECK (record_fake_quant (tape, input, output, forward_values, params, stale)
               == Status::BAD_LAYOUT);
        FakeQuantLayout wide;
        REQUIRE (status_ok (fake_quant_layout_flat (8u, 2u, wide)));
        CHECK (record_fake_quant (tape, input, output, forward_values, params, wide)
               == Status::BAD_LAYOUT);
        const Trainable mismatched
            = Trainable::make (input_code, input_grad, Shape::vector (4), RequiresGrad);
        CHECK (record_fake_quant (tape, mismatched, output, forward_values, params,
                                  layout)
               == Status::BAD_LAYOUT);
        const Trainable no_grad
            = Trainable::make (output_code, nullptr, Shape::vector (8), 0u);
        CHECK (record_fake_quant (tape, input, no_grad, forward_values, params, layout)
               == Status::BAD_LAYOUT);
        CHECK (record_fake_quant (tape, input, output, forward_values + 1, params,
                                  layout)
               == Status::BAD_LAYOUT);
        FakeQuantParams broken = params;
        broken._M_scale_q15 = nullptr;
        CHECK (record_fake_quant (tape, input, output, forward_values, broken, layout)
               == Status::BAD_LAYOUT);
        const Trainable shape_mismatch
            = Trainable::make (input_code, input_grad, Shape {}, RequiresGrad);
        CHECK (record_fake_quant (tape, shape_mismatch, output, forward_values, params,
                                  layout)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("the fake-quant tape backward validates its context",
           "[mllee][train][fakequant]")
{
    const int32_t multiplier = 328;
    const ClipWindow window = r_clip_window (multiplier, 0, -127, 127);

    alignas (8) int8_t code[4];
    alignas (8) int16_t incoming[4];
    alignas (8) int16_t routed[4];
    for (uint32_t i = 0; i < 4u; ++i)
        {
            code[i] = 0;
            incoming[i] = static_cast<int16_t> (500 + static_cast<int32_t> (i) * 25);
            routed[i] = 0;
        }
    alignas (8) int16_t values[4] = { 12737, 12738, -12737, -12738 };

    const Trainable output = Trainable::make (code, incoming, Shape::vector (4),
                                              RequiresGrad);
    const Trainable input = Trainable::make (code, routed, Shape::vector (4),
                                             RequiresGrad);
    const Trainable wide_input = Trainable::make (code, routed, Shape::vector (8),
                                                  RequiresGrad);
    const Trainable no_grad_view = Trainable::make (code, nullptr, Shape::vector (4),
                                                   RequiresGrad);

    AlignedArena<256> arena;
    Pool pool = arena.pool ();
    /* Carve the saved streams in the order record_fake_quant would. */
    int16_t *saved_values = nullptr;
    ClipWindowLane *saved_lanes = nullptr;
    FakeQuantNodeState *saved_state = nullptr;
    REQUIRE (status_ok (pool.carve_typed (4u, saved_values)));
    REQUIRE (status_ok (pool.carve_typed (1u, saved_lanes)));
    REQUIRE (status_ok (pool.carve_typed (1u, saved_state)));
    for (uint32_t i = 0; i < 4u; ++i)
        saved_values[i] = values[i];
    saved_lanes[0].set_window (window);
    FakeQuantNodeState state;
    state._M_run = 4u;
    state._M_group = 4u;
    state._M_outer = 1u;
    state._M_count = 4u;
    state._M_channels = 1u;
    *saved_state = state;

    Saved saved[3] = { Saved{ 0u, 8u, 4u }, Saved{ 8u, 24u, 1u },
                       Saved{ 32u, 24u, 1u } };
    Trainable operands[1] = { input };
    GradContext context;
    context._M_pool = &pool;
    context._M_operands = operands;
    context._M_operand_count = 1u;
    context._M_saved = saved;
    context._M_saved_count = 3u;

    SECTION ("exact lanes pass and clipped lanes are dropped")
    {
        REQUIRE (status_ok (fake_quant_backward (output, output, context)));
        CHECK (routed[0] == incoming[0]);
        CHECK (routed[1] == 0);
        CHECK (routed[2] == incoming[2]);
        CHECK (routed[3] == 0);
        CHECK (context._M_saturation_events == 0u);
    }

    SECTION ("a missing operand or saved stream is a layout error")
    {
        GradContext thin = context;
        thin._M_operand_count = 0u;
        CHECK (fake_quant_backward (output, output, thin) == Status::BAD_LAYOUT);
        GradContext fewer = context;
        fewer._M_saved_count = 2u;
        CHECK (fake_quant_backward (output, output, fewer) == Status::BAD_LAYOUT);
        GradContext zero_saved = context;
        zero_saved._M_saved_count = 0u;
        CHECK (fake_quant_backward (output, output, zero_saved) == Status::BAD_LAYOUT);

        GradContext frozen_operand = context;
        Trainable operand_views[1] = { no_grad_view };
        frozen_operand._M_operands = operand_views;
        CHECK (status_ok (fake_quant_backward (output, output, frozen_operand)));

        GradContext mismatched = context;
        Trainable wrong[1] = { wide_input };
        mismatched._M_operands = wrong;
        CHECK (fake_quant_backward (output, output, mismatched) == Status::BAD_LAYOUT);
        CHECK (fake_quant_backward (no_grad_view, output, context)
               == Status::BAD_LAYOUT);
    }

    SECTION ("an invalid saved window stops the walk")
    {
        ClipWindowLane invalid;
        ClipWindow blank;
        blank._M_valid = 0u;
        blank._M_low = window._M_low;
        blank._M_high = window._M_high;
        invalid.set_window (blank);
        CHECK (invalid._M_words[5] == 0);
        saved_lanes[0] = invalid;
        CHECK (fake_quant_backward (output, output, context) == Status::BAD_LAYOUT);
        saved_lanes[0].set_window (window);

        FakeQuantNodeState broken = state;
        broken._M_run = 0u;
        *saved_state = broken;
        CHECK (fake_quant_backward (output, output, context) == Status::BAD_LAYOUT);
        broken._M_run = 4u;
        broken._M_count = 2u;
        *saved_state = broken;
        CHECK (fake_quant_backward (output, output, context) == Status::BAD_LAYOUT);
        *saved_state = state;

        /* A saved handle too small for its type reads as null. */
        Saved patched[3] = { saved[0], saved[1], saved[2] };
        patched[1]._M_bytes = 4u;
        GradContext short_handle = context;
        short_handle._M_saved = patched;
        CHECK (fake_quant_backward (output, output, short_handle) == Status::BAD_LAYOUT);
        patched[1]._M_bytes = saved[1]._M_bytes;
        patched[2]._M_count = 2u;
        short_handle._M_saved = patched;
        CHECK (fake_quant_backward (output, output, short_handle) == Status::BAD_LAYOUT);
    }
}

/* ---------------------------------------------------------------- calibration */

namespace
{
/** @brief The observer support: 256 bins of 256 Q15 units over the signed domain. */
constexpr uint32_t R_BINS = 256u;
constexpr int32_t R_BIN_WIDTH = 256;
constexpr int32_t R_IDENTITY_SCALE = 258;
constexpr int64_t R_MSE_CAP = 1LL << 60;
constexpr int64_t R_I32_MAX = 2147483647LL;

constexpr uint32_t
r_bin_of (const int32_t value_q15) noexcept
{
    return static_cast<uint32_t> (r_q15_clamp (value_q15) + 32768) >> 8u;
}

constexpr int32_t
r_bin_center (const uint32_t bin) noexcept
{
    return static_cast<int32_t> (bin) * R_BIN_WIDTH + R_BIN_WIDTH / 2 - 32768;
}

constexpr int32_t
r_bin_upper (const uint32_t bin) noexcept
{
    return (static_cast<int32_t> (bin) + 1) * R_BIN_WIDTH - 32768;
}

constexpr int32_t
r_bin_lower (const uint32_t bin) noexcept
{
    return static_cast<int32_t> (bin) * R_BIN_WIDTH - 32768;
}

constexpr int32_t
r_bin_magnitude (const uint32_t bin) noexcept
{
    return r_bin_center (bin) < 0 ? -r_bin_lower (bin) : r_bin_upper (bin);
}

/** @brief The observer the headers describe, rebuilt from their rules. */
struct RefHistogram
{
    uint32_t counts[R_BINS] = { 0u };
    uint32_t total = 0u;
    int32_t min_q15 = 0;
    int32_t max_q15 = 0;
    uint32_t batches = 0u;
    bool seeded = false;

    void
    fold_q15 (const int32_t value_q15) noexcept
    {
        const int32_t value = r_q15_clamp (value_q15);
        const uint32_t bin = r_bin_of (value);
        if (counts[bin] < 0xFFFFFFFFu)
            ++counts[bin];
        if (!seeded)
            {
                min_q15 = value;
                max_q15 = value;
                seeded = true;
            }
        else
            {
                if (value < min_q15)
                    min_q15 = value;
                if (value > max_q15)
                    max_q15 = value;
            }
        if (total < 0xFFFFFFFFu)
            ++total;
    }

    void
    fold_code (const int8_t code) noexcept
    {
        fold_q15 (r_code_to_q15 (code));
    }

    /* Fold the half-open element range [begin, end) of a Q15 lane buffer: the
       short tail run of a truncated calibration stream. */
    void
    fold_range (const int16_t *const values, const uint32_t begin,
                const uint32_t end) noexcept
    {
        for (uint32_t i = begin; i < end; ++i)
            fold_q15 (values[i]);
    }

    void
    end_batch (const uint32_t count) noexcept
    {
        if (count != 0u && batches < 0xFFFFu)
            ++batches;
    }

    [[nodiscard]] int32_t
    abs_max () const noexcept
    {
        const int32_t negative = min_q15 < 0 ? -min_q15 : 0;
        const int32_t positive = max_q15 > 0 ? max_q15 : 0;
        return positive > negative ? positive : negative;
    }

    [[nodiscard]] uint32_t
    occupied_bins () const noexcept
    {
        uint32_t seen = 0u;
        for (uint32_t bin = 0; bin < R_BINS; ++bin)
            if (counts[bin] != 0u)
                ++seen;
        return seen;
    }
};

/** @brief Fill a library observer from a reference histogram, field by field. */
void
apply_histogram (CalibHistogram &histogram, const RefHistogram &reference) noexcept
{
    calib_reset (histogram);
    for (uint32_t bin = 0; bin < R_BINS; ++bin)
        histogram._M_counts[bin] = reference.counts[bin];
    histogram._M_total = reference.total;
    histogram._M_min_q15 = static_cast<int16_t> (reference.min_q15);
    histogram._M_max_q15 = static_cast<int16_t> (reference.max_q15);
    histogram._M_batches = static_cast<uint16_t> (reference.batches);
}

void
check_histogram (const CalibHistogram &histogram,
                 const RefHistogram &reference) noexcept
{
    for (uint32_t bin = 0; bin < R_BINS; ++bin)
        CHECK (histogram._M_counts[bin] == reference.counts[bin]);
    CHECK (histogram._M_total == reference.total);
    CHECK (histogram._M_min_q15 == reference.min_q15);
    CHECK (histogram._M_max_q15 == reference.max_q15);
    CHECK (histogram._M_batches == reference.batches);
    CHECK (histogram.abs_max () == reference.abs_max ());
}

/** @brief `calib_scale_from_abs_max`: ceil by 127, floored by the caller's step. */
constexpr int32_t
r_calib_scale_from_abs_max (const int32_t abs_max_q15,
                            const int32_t min_scale_q15) noexcept
{
    const int32_t floor_scale = min_scale_q15 > 1 ? min_scale_q15 : 1;
    if (abs_max_q15 <= 0)
        return floor_scale;
    const int32_t scale = (abs_max_q15 + 126) / 127;
    return scale > floor_scale ? scale : floor_scale;
}

/** @brief The reciprocal `calib_multiplier_from_scale` documents. */
constexpr int32_t
r_calib_multiplier_documented (const int32_t scale_q15) noexcept
{
    if (scale_q15 <= 0)
        return 0;
    return static_cast<int32_t> ((65536LL + scale_q15) / (2LL * scale_q15));
}

/** @brief The reciprocal it actually returns: (2*32768 + 2*d) / (2*d). */
constexpr int32_t
r_calib_multiplier_implemented (const int32_t scale_q15) noexcept
{
    if (scale_q15 <= 0)
        return 0;
    const uint32_t denominator = static_cast<uint32_t> (scale_q15) * 2u;
    const uint32_t quotient = (65536u + denominator) / denominator;
    return quotient == 0u ? 1 : static_cast<int32_t> (quotient);
}

constexpr int32_t
r_calib_dequantize (const int32_t code, const int32_t scale_q15,
                    const int32_t zero_point) noexcept
{
    const int64_t value = static_cast<int64_t> (code - zero_point) * scale_q15;
    const int64_t bounded = value > R_I32_MAX ? R_I32_MAX : value;
    return r_q15_clamp (static_cast<int32_t> (bounded));
}

constexpr int32_t
r_calib_quantize_code (const int32_t value_q15, const int32_t multiplier_q15,
                       const int32_t zero_point) noexcept
{
    const int32_t raw
        = r_round_shift (static_cast<int64_t> (r_q15_clamp (value_q15))
                             * multiplier_q15,
                         15u);
    return r_clamp_i8 (raw + zero_point);
}

constexpr int32_t
r_calib_round_trip (const int32_t value_q15, const int32_t scale_q15,
                    const int32_t multiplier_q15,
                    const int32_t zero_point) noexcept
{
    return r_calib_dequantize (r_calib_quantize_code (value_q15, multiplier_q15,
                                                      zero_point),
                               scale_q15, zero_point);
}

constexpr Status
r_asymmetric_grid (const int32_t lo_q15, const int32_t hi_q15,
                   const int32_t min_scale_q15, int32_t &scale_q15,
                   int32_t &zero_point) noexcept
{
    if (hi_q15 < lo_q15)
        return Status::BAD_LAYOUT;
    const int64_t span = static_cast<int64_t> (hi_q15) - static_cast<int64_t> (lo_q15);
    int32_t scale = static_cast<int32_t> ((span + 253) / 254);
    if (scale < min_scale_q15)
        scale = min_scale_q15 > 1 ? min_scale_q15 : 1;
    const int32_t multiplier = r_calib_multiplier_implemented (scale);
    zero_point = r_clamp_i8 (r_round_shift (static_cast<int64_t> (-lo_q15)
                                               * multiplier,
                                           15u));
    scale_q15 = scale;
    return Status::OK;
}

/** @brief The histogram rule: the edge bound of the widest occupied bin. */
constexpr int32_t
r_histogram_bound (const RefHistogram &histogram) noexcept
{
    int32_t bound = 0;
    for (uint32_t bin = 0; bin < R_BINS; ++bin)
        {
            if (histogram.counts[bin] == 0u)
                continue;
            const int32_t magnitude = r_bin_magnitude (bin);
            if (magnitude > bound)
                bound = magnitude;
        }
    return bound < R_BIN_WIDTH ? R_BIN_WIDTH : bound;
}

/** @brief The percentile rule: equal-mass pairs dropped from both ends. */
constexpr Status
r_percentile_bound (const RefHistogram &histogram, const int32_t percentile_q15,
                    int32_t &out) noexcept
{
    out = 0;
    if (percentile_q15 < 0 || percentile_q15 > 32768)
        return Status::OUT_OF_RANGE;
    if (histogram.total == 0u)
        {
            out = R_BIN_WIDTH;
            return Status::OK;
        }
    const int64_t allowed = 32768 - percentile_q15;
    const int64_t budget = (static_cast<int64_t> (histogram.total) * allowed) / 32768;
    uint32_t low = 0u;
    uint32_t high = R_BINS - 1u;
    int64_t dropped = 0;
    while (low < high)
        {
            const int64_t pair
                = static_cast<int64_t> (histogram.counts[high])
                  + static_cast<int64_t> (histogram.counts[low]);
            if (dropped + pair > budget)
                break;
            dropped += pair;
            ++low;
            --high;
        }
    int32_t bound = r_bin_magnitude (high);
    const int32_t mirrored = r_bin_magnitude (low);
    if (mirrored > bound)
        bound = mirrored;
    const int32_t exact = histogram.abs_max ();
    if (bound > exact && exact >= R_BIN_WIDTH)
        bound = exact;
    if (bound < R_BIN_WIDTH)
        bound = R_BIN_WIDTH;
    out = bound;
    return Status::OK;
}

constexpr int32_t
r_worst_error (const RefHistogram &histogram, const int32_t scale_q15,
               const int32_t multiplier_q15, const int32_t zero_point) noexcept
{
    int32_t worst = 0;
    for (uint32_t bin = 0; bin < R_BINS; ++bin)
        {
            if (histogram.counts[bin] == 0u)
                continue;
            const int32_t value = r_bin_center (bin);
            const int32_t back = r_calib_round_trip (value, scale_q15, multiplier_q15,
                                                     zero_point);
            const int32_t error = back > value ? back - value : value - back;
            if (error > worst)
                worst = error;
        }
    return worst;
}

constexpr int64_t
r_mse_objective (const RefHistogram &histogram, const int32_t scale_q15,
                 const int32_t multiplier_q15,
                 const int32_t zero_point) noexcept
{
    int64_t total = 0;
    for (uint32_t bin = 0; bin < R_BINS; ++bin)
        {
            const uint32_t count = histogram.counts[bin];
            if (count == 0u)
                continue;
            const int32_t value = r_bin_center (bin);
            const int64_t back = r_calib_round_trip (value, scale_q15, multiplier_q15,
                                                     zero_point);
            const int64_t delta = back - value;
            total += delta * delta * static_cast<int64_t> (count);
            if (total > R_MSE_CAP)
                return R_MSE_CAP;
        }
    return total;
}

/** @brief KL(P||Q) in Q7 units with the module's own log-domain primitives. */
int32_t
r_kl_objective (const RefHistogram &histogram, const int32_t scale_q15,
                const int32_t multiplier_q15, const int32_t zero_point) noexcept
{
    if (histogram.total == 0u)
        return 0;
    uint32_t pushed[R_BINS] = { 0u };
    for (uint32_t bin = 0; bin < R_BINS; ++bin)
        {
            if (histogram.counts[bin] == 0u)
                continue;
            const int32_t back = r_calib_round_trip (r_bin_center (bin), scale_q15,
                                                    multiplier_q15, zero_point);
            const uint32_t target = r_bin_of (back);
            if (pushed[target] < 0xFFFFFFFFu)
                ++pushed[target];
        }
    const int32_t total = static_cast<int32_t> (histogram.total);
    int32_t divergence = 0;
    for (uint32_t bin = 0; bin < R_BINS; ++bin)
        {
            if (histogram.counts[bin] == 0u)
                continue;
            const int32_t p_q15 = ratio_q15 (static_cast<int32_t> (histogram.counts[bin]),
                                             total);
            const uint32_t moved = pushed[bin] == 0u ? 1u : pushed[bin];
            const int32_t q_q15 = ratio_q15 (static_cast<int32_t> (moved), total);
            const int32_t log_ratio
                = q7_log_q15 (p_q15 < 1 ? 1 : p_q15) - q7_log_q15 (q_q15 < 1 ? 1
                                                                     : q_q15);
            divergence = q15_add_sat (divergence, q15_mul (p_q15, log_ratio));
        }
    return divergence;
}

constexpr Status
r_grid_for_bound (const RefHistogram &histogram, const CalibConfig &config,
                  const int32_t bound_q15, int32_t &scale_q15,
                  int32_t &multiplier_q15, int32_t &zero_point) noexcept
{
    if (bound_q15 <= 0)
        return Status::BAD_LAYOUT;
    if (config._M_asymmetric != 0u)
        {
            int32_t lo = histogram.min_q15;
            int32_t hi = histogram.max_q15;
            if (hi > bound_q15)
                hi = bound_q15;
            if (lo < -bound_q15)
                lo = -bound_q15;
            const Status grid = r_asymmetric_grid (lo, hi, config._M_min_scale_q15,
                                                   scale_q15, zero_point);
            if (!status_ok (grid))
                return grid;
        }
    else
        {
            const int32_t ceil_127 = (bound_q15 + 126) / 127;
            scale_q15 = ceil_127 > config._M_min_scale_q15 ? ceil_127
                                                           : config._M_min_scale_q15;
            if (scale_q15 < 1)
                scale_q15 = 1;
            zero_point = 0;
        }
    multiplier_q15 = r_calib_multiplier_implemented (scale_q15);
    if (multiplier_q15 <= 0)
        return Status::OUT_OF_RANGE;
    return Status::OK;
}

/** @brief The rule's own config gate, restated from the documented contract. */
constexpr Status
r_validate_config (const CalibConfig &config) noexcept
{
    if (config._M_method > static_cast<uint8_t> (CalibMethod::Kl))
        return Status::UNSUPPORTED;
    if (config._M_percentile_q15 < 0 || config._M_percentile_q15 > 32768)
        return Status::OUT_OF_RANGE;
    if (config._M_min_scale_q15 < 1)
        return Status::OUT_OF_RANGE;
    if (config._M_min_codes < 1 || config._M_min_codes > 127)
        return Status::OUT_OF_RANGE;
    if (config._M_axis != _S_calib_per_tensor && config._M_axis >= _S_max_rank)
        return Status::BAD_LAYOUT;
    return Status::OK;
}

/** @brief One reference selection: bound, grid, objective and error. */
struct RefChoice
{
    int32_t scale_q15 = _S_code_to_q15_scale;
    int32_t multiplier_q15 = 127;
    int32_t zero_point = 0;
    int32_t bound_q15 = 0;
    int32_t worst_error_q15 = 0;
    int32_t objective_q15 = 0;
    uint8_t method = 0u;
    uint8_t asymmetric = 0u;
    uint8_t degenerate = 0u;
    uint8_t codes = 127u;
    Status status = Status::OK;
};

constexpr int32_t
r_sat_i32_from_i64 (const int64_t value) noexcept
{
    return value > R_I32_MAX ? static_cast<int32_t> (R_I32_MAX)
                             : (value < -R_I32_MAX ? static_cast<int32_t> (-R_I32_MAX)
                                                   : static_cast<int32_t> (value));
}

RefChoice
r_select (const RefHistogram &histogram, const CalibConfig &config) noexcept
{
    RefChoice choice;
    choice.status = r_validate_config (config);
    choice.method = config._M_method;
    if (!status_ok (choice.status))
        return choice;
    if (histogram.total == 0u)
        {
            choice.scale_q15 = _S_code_to_q15_scale;
            choice.multiplier_q15 = r_calib_multiplier_implemented (choice.scale_q15);
            choice.degenerate = 1u;
            choice.asymmetric = 0u;
            choice.codes = 127u;
            return choice;
        }
    int32_t bound = 0;
    switch (static_cast<CalibMethod> (config._M_method))
        {
        case CalibMethod::MinMax:
        case CalibMethod::Mse:
        case CalibMethod::Kl:
            bound = histogram.abs_max ();
            break;
        case CalibMethod::Histogram:
            bound = r_histogram_bound (histogram);
            break;
        case CalibMethod::Percentile:
            choice.status = r_percentile_bound (histogram, config._M_percentile_q15,
                                                bound);
            break;
        }
    if (!status_ok (choice.status))
        return choice;
    if (bound < R_BIN_WIDTH)
        bound = R_BIN_WIDTH;
    choice.bound_q15 = bound;
    choice.degenerate = 0u;
    choice.asymmetric = config._M_asymmetric;
    const bool search = config._M_method == static_cast<uint8_t> (CalibMethod::Mse)
                        || config._M_method == static_cast<uint8_t> (CalibMethod::Kl);
    if (!search)
        {
            int32_t scale = 0;
            int32_t multiplier = 0;
            int32_t zero_point = 0;
            choice.status = r_grid_for_bound (histogram, config, bound, scale,
                                              multiplier, zero_point);
            if (!status_ok (choice.status))
                return choice;
            choice.scale_q15 = scale;
            choice.multiplier_q15 = multiplier;
            choice.zero_point = zero_point;
            const int32_t reached = bound / scale;
            choice.codes = static_cast<uint8_t> (reached > 127 ? 127 : reached);
            choice.worst_error_q15 = r_worst_error (histogram, scale, multiplier,
                                                    zero_point);
            return choice;
        }
    int64_t best_objective = 0;
    int32_t best_scale = 0;
    int32_t best_multiplier = 0;
    int32_t best_zero_point = 0;
    int32_t best_codes = 0;
    for (int32_t codes = 127; codes >= config._M_min_codes; --codes)
        {
            const int64_t candidate_wide
                = (static_cast<int64_t> (bound) * codes + 126) / 127;
            if (candidate_wide <= 0)
                continue;
            const int32_t candidate = static_cast<int32_t> (
                candidate_wide > 32767 ? 32767 : candidate_wide);
            int32_t scale = 0;
            int32_t multiplier = 0;
            int32_t zero_point = 0;
            choice.status = r_grid_for_bound (histogram, config, candidate, scale,
                                              multiplier, zero_point);
            if (!status_ok (choice.status))
                return choice;
            const int64_t objective
                = config._M_method == static_cast<uint8_t> (CalibMethod::Kl)
                      ? static_cast<int64_t> (r_kl_objective (histogram, scale,
                                                              multiplier,
                                                              zero_point))
                      : r_mse_objective (histogram, scale, multiplier, zero_point);
            if (best_codes == 0 || objective < best_objective)
                {
                    best_codes = codes;
                    best_objective = objective;
                    best_scale = scale;
                    best_multiplier = multiplier;
                    best_zero_point = zero_point;
                }
        }
    if (best_codes == 0)
        {
            choice.status = Status::INTERNAL;
            return choice;
        }
    choice.scale_q15 = best_scale;
    choice.multiplier_q15 = best_multiplier;
    choice.zero_point = best_zero_point;
    choice.codes = static_cast<uint8_t> (best_codes);
    choice.objective_q15 = r_sat_i32_from_i64 (best_objective);
    choice.worst_error_q15 = r_worst_error (histogram, best_scale, best_multiplier,
                                            best_zero_point);
    return choice;
}

void
check_choice (const CalibChoice &actual, const RefChoice &expected) noexcept
{
    CHECK (actual._M_scale_q15 == expected.scale_q15);
    CHECK (actual._M_multiplier_q15 == expected.multiplier_q15);
    CHECK (actual._M_zero_point == expected.zero_point);
    CHECK (actual._M_bound_q15 == expected.bound_q15);
    CHECK (actual._M_worst_error_q15 == expected.worst_error_q15);
    CHECK (actual._M_objective_q15 == expected.objective_q15);
    CHECK (actual._M_method == expected.method);
    CHECK (actual._M_asymmetric == expected.asymmetric);
    CHECK (actual._M_degenerate == expected.degenerate);
    CHECK (actual._M_codes == expected.codes);
}

/** @brief A default configuration with one field changed. */
constexpr CalibConfig
config_with (const CalibMethod method, const uint8_t asymmetric) noexcept
{
    CalibConfig config;
    config._M_method = static_cast<uint8_t> (method);
    config._M_asymmetric = asymmetric;
    return config;
}

/* Deterministic lane fixtures. */
constexpr int16_t _S_lanes_spiky[]
    = { 0, 0, 0, 0, 0, 0, 40, -40, 300, -300, 512, -512, 1200, -1200, 2600,
        -2600, 31000, -31000, 77, -77 };

constexpr int16_t _S_lanes_positive[]
    = { 200, 900, 1500, 2000, 25000, 200, 900, 1500, 2000, 25000, 700, 700 };

constexpr uint32_t
lane_count (const int16_t *const lanes) noexcept
{
    uint32_t count = 0u;
    while (count < 64u && static_cast<int32_t> (lanes[count]) != 0x7FFF)
        ++count;
    return count;
}
} // namespace

TEST_CASE ("calibration binning cuts the signed Q15 domain into exact bins",
           "[mllee][train][calibration]")
{
    SECTION ("the bin index is the value lifted into the unsigned domain")
    {
        const int32_t edges[] = { -40000, -32768, -32767, -32513, -32512, -32511,
                                  -1, 0, 1, 127, 128, 255, 256, 16384, 32511, 32512,
                                  32513, 32766, 32767, 40000 };
        for (const int32_t value : edges)
            {
                CAPTURE (value);
                CHECK (CalibHistogram::bin_of (value) == r_bin_of (value));
                const int32_t clamped = r_q15_clamp (value);
                CHECK (clamped + 32768 >= 0);
                CHECK (CalibHistogram::bin_of (value) <= 255u);
            }
        CHECK (CalibHistogram::bin_of (-32768) == 0u);
        CHECK (CalibHistogram::bin_of (-32513) == 0u);
        CHECK (CalibHistogram::bin_of (-32512) == 1u);
        CHECK (CalibHistogram::bin_of (32512) == 255u);
        CHECK (CalibHistogram::bin_of (32767) == 255u);
        CHECK (CalibHistogram::bin_of (32768) == 255u); /* clamped, not wrapped */
    }

    SECTION ("bin centres and edges tile the domain without gaps")
    {
        CHECK (CalibHistogram::bin_lower (0u) == R_Q15_MIN);
        CHECK (CalibHistogram::bin_upper (255u) == 32768);
        CHECK (CalibHistogram::bin_center (0u) == -32640);
        CHECK (CalibHistogram::bin_center (128u) == 128);
        CHECK (CalibHistogram::bin_center (127u) == -128);
        for (uint32_t bin = 0; bin < R_BINS; ++bin)
            {
                CHECK (CalibHistogram::bin_lower (bin) == r_bin_lower (bin));
                CHECK (CalibHistogram::bin_upper (bin) == r_bin_upper (bin));
                CHECK (CalibHistogram::bin_center (bin) == r_bin_center (bin));
                CHECK (CalibHistogram::bin_magnitude (bin) == r_bin_magnitude (bin));
                /* The centre sits inside the bin and the bin is 256 wide. */
                CHECK (CalibHistogram::bin_center (bin)
                       == CalibHistogram::bin_lower (bin) + 128);
                CHECK (CalibHistogram::bin_upper (bin)
                       - CalibHistogram::bin_lower (bin) == 256);
                if (bin + 1u < R_BINS)
                    CHECK (CalibHistogram::bin_upper (bin)
                           == CalibHistogram::bin_lower (bin + 1u));
                /* Every centre bins back to its own bin. */
                CHECK (CalibHistogram::bin_of (CalibHistogram::bin_center (bin))
                       == bin);
                /* The magnitude is the farther edge, so the grid is symmetric. */
                const int32_t lower_distance = -CalibHistogram::bin_lower (bin);
                const int32_t upper_distance = CalibHistogram::bin_upper (bin);
                CHECK (CalibHistogram::bin_magnitude (bin)
                       == (lower_distance > upper_distance ? lower_distance
                                                            : upper_distance));
            }
    }

    SECTION ("the abs max reads the exact extremes, not the bins")
    {
        CalibHistogram histogram;
        calib_reset (histogram);
        CHECK (histogram.abs_max () == 0);
        histogram._M_min_q15 = -32000;
        histogram._M_max_q15 = 30000;
        CHECK (histogram.abs_max () == 32000);
        histogram._M_max_q15 = 32100;
        CHECK (histogram.abs_max () == 32100);
        /* The most negative lane an int16 can hold still reads a positive 32768. */
        histogram._M_min_q15 = -32768;
        histogram._M_max_q15 = 0;
        CHECK (histogram.abs_max () == 32768);
        histogram._M_min_q15 = -1;
        histogram._M_max_q15 = -1;
        CHECK (histogram.abs_max () == 1);
        /* The same -32768 edge through both implementations agrees. */
        RefHistogram reference;
        reference.min_q15 = -32768;
        reference.max_q15 = 0;
        histogram._M_min_q15 = -32768;
        histogram._M_max_q15 = 0;
        CHECK (histogram.abs_max () == reference.abs_max ());
    }
}

TEST_CASE ("an observer folds lanes with saturating counts",
           "[mllee][train][calibration]")
{
    SECTION ("a stream of lanes builds bins, extremes and a batch count")
    {
        alignas (8) const int16_t lanes[] = { 1000, -2000, 32767, -32768, 0, 7,
                                              255, 256, -256, -257, 12345, -12345 };
        RefHistogram reference;
        CalibHistogram histogram;
        calib_reset (histogram);
        for (uint32_t i = 0; i < 12u; ++i)
            {
                calib_fold_q15 (histogram, lanes[i]);
                reference.fold_q15 (lanes[i]);
            }
        check_histogram (histogram, reference);
        CHECK (histogram._M_batches == 0u); /* folding is not a batch */
        CHECK (histogram._M_pad == 0u);
        CHECK (histogram._M_counts[CalibHistogram::bin_of (1000)] == 1u);
        /* The bin is a 256-wide bucket, so 0, 7 and 255 all land in it. */
        CHECK (histogram._M_counts[CalibHistogram::bin_of (0)] == 3u);
        /* 255 and 256 straddle a bin edge. */
        CHECK (CalibHistogram::bin_of (255) == CalibHistogram::bin_of (0));
        CHECK (CalibHistogram::bin_of (256) != CalibHistogram::bin_of (255));
    }

    SECTION ("values beyond the Q15 domain are clamped, never wrapped")
    {
        RefHistogram reference;
        CalibHistogram histogram;
        calib_reset (histogram);
        const int32_t beyond[] = { 40000, -40000, 2147483647, -2147483647 - 1 };
        for (const int32_t value : beyond)
            {
                calib_fold_q15 (histogram, value);
                reference.fold_q15 (value);
            }
        check_histogram (histogram, reference);
        CHECK (histogram._M_min_q15 == R_Q15_MIN);
        CHECK (histogram._M_max_q15 == R_Q15_MAX);
        /* -40000 and INT32_MIN both clamp to the rail, +40000 and INT32_MAX both
           clamp to the other, so each rail bin holds two samples. */
        CHECK (histogram._M_counts[0u] == 2u);
        CHECK (histogram._M_counts[255u] == 2u);
        CHECK (histogram.abs_max () == 32768);
    }

    SECTION ("counts and totals saturate instead of wrapping")
    {
        CalibHistogram histogram;
        calib_reset (histogram);
        histogram._M_counts[40u] = 0xFFFFFFFFu;
        histogram._M_total = 0xFFFFFFFFu;
        histogram._M_min_q15 = -100;
        histogram._M_max_q15 = 100;
        /* A lane that lands on the full bin, with the total already full. */
        calib_fold_q15 (histogram, r_bin_center (40u));
        CHECK (histogram._M_counts[40u] == 0xFFFFFFFFu);
        CHECK (histogram._M_total == 0xFFFFFFFFu);
        /* Bin 40 is on the negative half of the signed domain, so the fold moves
           the minimum rail and leaves the maximum alone. */
        CHECK (histogram._M_min_q15 == r_bin_center (40u));
        CHECK (histogram._M_max_q15 == 100);
        calib_count_saturate (histogram._M_total);
        CHECK (histogram._M_total == 0xFFFFFFFFu);
        histogram._M_total = 0u;
        calib_count_saturate (histogram._M_total);
        CHECK (histogram._M_total == 1u);
    }

    SECTION ("the extremes seed on the first sample")
    {
        RefHistogram reference;
        CalibHistogram histogram;
        calib_reset (histogram);
        const int32_t negative_only[] = { -5000, -10, -40000 };
        for (uint32_t i = 0; i < 3u; ++i)
            {
                calib_fold_q15 (histogram, negative_only[i]);
                reference.fold_q15 (negative_only[i]);
            }
        /* Without seeding, an all-negative stream would pin the max at 0. */
        CHECK (histogram._M_max_q15 == -10);
        CHECK (histogram._M_min_q15 == R_Q15_MIN);
        check_histogram (histogram, reference);

        calib_reset (histogram);
        reference = RefHistogram{};
        calib_fold_q15 (histogram, 1200);
        reference.fold_q15 (1200);
        CHECK (histogram._M_min_q15 == 1200);
        CHECK (histogram._M_max_q15 == 1200);
        check_histogram (histogram, reference);
    }

    SECTION ("a folded batch counts itself once")
    {
        alignas (8) const int16_t lanes[] = { 100, 200, 300 };
        RefHistogram reference;
        CalibHistogram histogram;
        calib_reset (histogram);
        for (uint32_t round = 0; round < 3u; ++round)
            {
                REQUIRE (status_ok (calib_observe_lanes (histogram, lanes, 3u)));
                for (uint32_t i = 0; i < 3u; ++i)
                    reference.fold_q15 (lanes[i]);
                reference.end_batch (3u);
            }
        REQUIRE (status_ok (calib_observe_lanes (histogram, lanes, 0u)));
        check_histogram (histogram, reference);
        CHECK (histogram._M_batches == 3u);
        CHECK (histogram._M_total == 9u);
        /* The batch counter itself saturates at 16 bits. */
        histogram._M_batches = 0xFFFFu;
        REQUIRE (status_ok (calib_observe_lanes (histogram, lanes, 3u)));
        CHECK (histogram._M_batches == 0xFFFFu);
    }

    SECTION ("a lane buffer must exist and stay inside the element cap")
    {
        alignas (8) const int16_t lanes[] = { 1, 2, 3 };
        CalibHistogram histogram;
        calib_reset (histogram);
        CHECK (calib_observe_lanes (histogram, nullptr, 3u) == Status::BAD_LAYOUT);
        CHECK (calib_observe_lanes (histogram, lanes, _S_max_elements + 1u)
               == Status::OUT_OF_RANGE);
        CHECK (histogram._M_total == 0u);
    }
}

TEST_CASE ("calib_observe_codes widens int8 codes on the identity grid",
           "[mllee][train][calibration]")
{
    SECTION ("a code is exactly 256 times its value plus two")
    {
        alignas (8) const int8_t codes[] = { 127, -127, 0, 1, -1, 64, -64, 33 };
        RefHistogram reference;
        CalibHistogram histogram;
        calib_reset (histogram);
        REQUIRE (status_ok (calib_observe_codes (histogram, codes, 8u)));
        for (const int8_t code : codes)
            reference.fold_q15 (r_code_to_q15 (code));
        reference.end_batch (8u);
        check_histogram (histogram, reference);
        CHECK (histogram._M_max_q15 == 127 * R_CODE_STEP);
        CHECK (histogram._M_min_q15 == -127 * R_CODE_STEP);
        CHECK (histogram.abs_max () == 32766);
        CHECK (histogram._M_total == 8u);
        CHECK (histogram._M_batches == 1u);
        /* Code 127 is 32766, i.e. the identity grid never reaches the Q15 rail. */
        CHECK (CalibHistogram::bin_of (127 * R_CODE_STEP) == 255u);
        CHECK (CalibHistogram::bin_of (-127 * R_CODE_STEP) == 0u);
    }

    SECTION ("the code path shares the lane path's support")
    {
        alignas (8) const int8_t codes[] = { 5, -5, 100 };
        alignas (8) const int16_t lanes[]
            = { static_cast<int16_t> (5 * R_CODE_STEP),
                static_cast<int16_t> (-5 * R_CODE_STEP),
                static_cast<int16_t> (100 * R_CODE_STEP) };
        CalibHistogram from_codes;
        CalibHistogram from_lanes;
        calib_reset (from_codes);
        calib_reset (from_lanes);
        REQUIRE (status_ok (calib_observe_codes (from_codes, codes, 3u)));
        REQUIRE (status_ok (calib_observe_lanes (from_lanes, lanes, 3u)));
        for (uint32_t bin = 0; bin < R_BINS; ++bin)
            CHECK (from_codes._M_counts[bin] == from_lanes._M_counts[bin]);
        CHECK (from_codes._M_total == from_lanes._M_total);
        CHECK (from_codes._M_min_q15 == from_lanes._M_min_q15);
        CHECK (from_codes._M_max_q15 == from_lanes._M_max_q15);
    }

    SECTION ("argument checks")
    {
        alignas (8) const int8_t codes[] = { 1, 2, 3 };
        CalibHistogram histogram;
        calib_reset (histogram);
        CHECK (calib_observe_codes (histogram, nullptr, 3u) == Status::BAD_LAYOUT);
        CHECK (calib_observe_codes (histogram, codes, _S_max_elements + 1u)
               == Status::OUT_OF_RANGE);
        CHECK (histogram._M_total == 0u);
        CHECK (histogram._M_batches == 0u);
    }
}

TEST_CASE ("the observer is order independent and reset returns it to empty",
           "[mllee][train][calibration]")
{
    RefHistogram forward;
    CalibHistogram histogram;
    calib_reset (histogram);
    REQUIRE (status_ok (calib_observe_lanes (histogram, _S_lanes_spiky, 20u)));
    for (uint32_t i = 0; i < 20u; ++i)
        forward.fold_q15 (_S_lanes_spiky[i]);
    forward.end_batch (20u); /* one observe call is one batch */
    check_histogram (histogram, forward);

    SECTION ("a reversed stream gives the identical histogram")
    {
        alignas (8) int16_t reversed[20];
        for (uint32_t i = 0; i < 20u; ++i)
            reversed[i] = _S_lanes_spiky[19u - i];
        CalibHistogram other;
        calib_reset (other);
        REQUIRE (status_ok (calib_observe_lanes (other, reversed, 20u)));
        for (uint32_t bin = 0; bin < R_BINS; ++bin)
            CHECK (other._M_counts[bin] == histogram._M_counts[bin]);
        CHECK (memcmp (other._M_counts, histogram._M_counts,
                       sizeof (other._M_counts))
               == 0);
        CHECK (other._M_total == histogram._M_total);
        CHECK (other._M_min_q15 == histogram._M_min_q15);
        CHECK (other._M_max_q15 == histogram._M_max_q15);
    }

    SECTION ("splitting a stream into more batches changes nothing but the count")
    {
        CalibHistogram split;
        calib_reset (split);
        REQUIRE (status_ok (calib_observe_lanes (split, _S_lanes_spiky, 7u)));
        REQUIRE (status_ok (calib_observe_lanes (split, _S_lanes_spiky + 7, 6u)));
        REQUIRE (status_ok (calib_observe_lanes (split, _S_lanes_spiky + 13, 7u)));
        CHECK (split._M_batches == 3u);
        CHECK (memcmp (split._M_counts, histogram._M_counts,
                       sizeof (split._M_counts))
               == 0);
        CHECK (split._M_total == histogram._M_total);
    }

    SECTION ("reset clears bins, extremes, total and batches")
    {
        calib_reset (histogram);
        for (uint32_t bin = 0; bin < R_BINS; ++bin)
            CHECK (histogram._M_counts[bin] == 0u);
        CHECK (histogram._M_total == 0u);
        CHECK (histogram._M_min_q15 == 0);
        CHECK (histogram._M_max_q15 == 0);
        CHECK (histogram._M_batches == 0u);
        CHECK (histogram._M_pad == 0u);
        CHECK (histogram.abs_max () == 0);
    }

    SECTION ("the axis-wise reset clears every channel")
    {
        alignas (8) CalibHistogram channels[3] {};
        for (uint32_t channel = 0; channel < 3u; ++channel)
            REQUIRE (status_ok (calib_observe_lanes (channels[channel],
                                                    _S_lanes_spiky, 20u)));
        calib_reset_axiswise (nullptr, 3u); /* documented as a no-op */
        calib_reset_axiswise (channels, 0u);
        CHECK (channels[0]._M_total == 20u);
        calib_reset_axiswise (channels, 2u);
        CHECK (channels[0]._M_total == 0u);
        CHECK (channels[1]._M_total == 0u);
        CHECK (channels[2]._M_total == 20u);
        calib_reset_axiswise (channels, 3u);
        for (uint32_t channel = 0; channel < 3u; ++channel)
            CHECK (channels[channel]._M_total == 0u);
    }

    SECTION ("the observer slab is the measured 1036 bytes")
    {
        CHECK (sizeof (CalibHistogram) == 1036u);
        CHECK (calib_histogram_bytes (1u) == 1036u);
        CHECK (calib_histogram_bytes (3u) == 3108u);
        CHECK (calib_histogram_bytes (0u) == 0u);
        CHECK (calib_kl_scratch_bytes () == 1024u);
        CHECK (alignof (CalibHistogram) <= 4u);
    }
}

TEST_CASE ("calib_layout derives the channel runs of a shape",
           "[mllee][train][calibration]")
{
    SECTION ("a per-tensor layout is one run over the whole buffer")
    {
        CalibLayout layout;
        REQUIRE (status_ok (calib_layout (Shape::tensor3 (2, 3, 4),
                                         _S_calib_per_tensor, layout)));
        CHECK (layout._M_valid == 1u);
        CHECK (layout._M_count == 24u);
        CHECK (layout._M_run == 24u);
        CHECK (layout._M_period == 24u);
        CHECK (layout._M_channels == 1u);
        CHECK (layout._M_axis == 0u);
        CHECK (sizeof (CalibLayout) == 16u);
    }

    SECTION ("an axis-wise layout repeats one run per channel")
    {
        for (uint16_t axis = 0u; axis < 3u; ++axis)
            {
                CAPTURE (axis);
                CalibLayout layout;
                REQUIRE (status_ok (calib_layout (Shape::tensor3 (2, 3, 4), axis,
                                                  layout)));
                const Shape shape = Shape::tensor3 (2, 3, 4);
                CHECK (layout._M_valid == 1u);
                CHECK (layout._M_count == 24u);
                CHECK (layout._M_channels == shape._M_dims[axis]);
                CHECK (layout._M_run == shape.stride (static_cast<uint8_t> (axis)));
                CHECK (layout._M_period == layout._M_run * layout._M_channels);
                CHECK (layout._M_axis == axis);
                /* A whole number of periods of channel runs tiles the buffer:
                   this is exactly the walk calib_observe_axiswise performs. */
                CHECK (layout._M_count % layout._M_period == 0u);
                uint32_t covered = 0u;
                const uint32_t blocks = layout._M_count / layout._M_period;
                for (uint32_t block = 0u; block < blocks; ++block)
                    for (uint32_t channel = 0u; channel < layout._M_channels;
                         ++channel)
                        {
                            const uint32_t first
                                = block * layout._M_period
                                  + channel * layout._M_run;
                            CHECK (first + layout._M_run <= layout._M_count);
                            covered += layout._M_run;
                        }
                CHECK (covered == layout._M_count);
            }
        CalibLayout leading;
        REQUIRE (status_ok (calib_layout (Shape::tensor4 (2, 3, 4, 5), 1u, leading)));
        CHECK (leading._M_count == 120u);
        CHECK (leading._M_channels == 3u);
        CHECK (leading._M_run == 20u);
        CHECK (leading._M_period == 60u);
    }

    SECTION ("invalid shapes and axes are refused and mark the layout stale")
    {
        CalibLayout layout;
        REQUIRE (status_ok (calib_layout (Shape::vector (8u), 0u, layout)));
        CHECK (layout._M_valid == 1u);
        CHECK (calib_layout (Shape{}, 0u, layout) == Status::BAD_LAYOUT);
        CHECK (layout._M_valid == 0u);
        CHECK (calib_layout (Shape::vector (8u), 4u, layout) == Status::BAD_LAYOUT);
        CHECK (layout._M_valid == 0u);
        CHECK (calib_layout (Shape::vector (8u), 1u, layout) == Status::BAD_LAYOUT);
        CalibLayout zero;
        CHECK (calib_layout (Shape::matrix (0u, 4u), 0u, zero) == Status::BAD_LAYOUT);
        CHECK (zero._M_valid == 0u);
        CalibLayout huge;
        CHECK (calib_layout (Shape::matrix (300u, 300u), 0u, huge)
               == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("an axis-wise observer routes every run to its channel",
           "[mllee][train][calibration]")
{
    /* Shape (2, 3, 4) along axis 1: three channels, runs of four, two blocks. */
    alignas (8) int16_t lanes[24];
    for (uint32_t i = 0; i < 24u; ++i)
        lanes[i] = static_cast<int16_t> (-30000 + static_cast<int32_t> (i) * 2500);

    CalibLayout layout;
    REQUIRE (status_ok (calib_layout (Shape::tensor3 (2, 3, 4), 1u, layout)));
    CHECK (layout._M_run == 4u);
    CHECK (layout._M_channels == 3u);
    CHECK (layout._M_period == 12u);

    SECTION ("each channel sees exactly its own runs")
    {
        RefHistogram reference[3];
        /* Channel c owns [start, start+4) for start = block*12 + c*4. */
        for (uint32_t block = 0; block < 2u; ++block)
            for (uint32_t channel = 0; channel < 3u; ++channel)
                for (uint32_t lane = 0; lane < 4u; ++lane)
                    reference[channel].fold_q15 (
                        lanes[block * 12u + channel * 4u + lane]);
        for (uint32_t channel = 0; channel < 3u; ++channel)
            reference[channel].end_batch (8u);

        alignas (8) CalibHistogram histograms[3];
        for (uint32_t channel = 0; channel < 3u; ++channel)
            calib_reset (histograms[channel]);
        REQUIRE (status_ok (calib_observe_axiswise (histograms, layout, lanes,
                                                    nullptr)));
        for (uint32_t channel = 0; channel < 3u; ++channel)
            check_histogram (histograms[channel], reference[channel]);
        /* Different channels must not collapse onto one distribution. */
        CHECK_FALSE (memcmp (histograms[0]._M_counts, histograms[1]._M_counts,
                             sizeof (histograms[0]._M_counts))
                     == 0);
    }

    SECTION ("the code form routes the same way")
    {
        alignas (8) int8_t codes[24];
        for (uint32_t i = 0; i < 24u; ++i)
            codes[i] = static_cast<int8_t> (static_cast<int32_t> (i) - 12);
        RefHistogram reference[3];
        for (uint32_t block = 0; block < 2u; ++block)
            for (uint32_t channel = 0; channel < 3u; ++channel)
                for (uint32_t lane = 0; lane < 4u; ++lane)
                    reference[channel].fold_code (
                        codes[block * 12u + channel * 4u + lane]);
        for (uint32_t channel = 0; channel < 3u; ++channel)
            reference[channel].end_batch (8u);

        alignas (8) CalibHistogram histograms[3];
        for (uint32_t channel = 0; channel < 3u; ++channel)
            calib_reset (histograms[channel]);
        REQUIRE (status_ok (calib_observe_axiswise (histograms, layout, nullptr,
                                                    codes)));
        for (uint32_t channel = 0; channel < 3u; ++channel)
            check_histogram (histograms[channel], reference[channel]);
    }

    SECTION ("exactly one source stream must be given")
    {
        alignas (8) CalibHistogram histograms[3];
        alignas (8) int8_t codes[24];
        for (uint32_t i = 0; i < 24u; ++i)
            codes[i] = 0;
        CHECK (calib_observe_axiswise (histograms, layout, lanes, codes)
               == Status::BAD_LAYOUT);
        CHECK (calib_observe_axiswise (histograms, layout, nullptr, nullptr)
               == Status::BAD_LAYOUT);
        CHECK (calib_observe_axiswise (nullptr, layout, lanes, nullptr)
               == Status::BAD_LAYOUT);
        CalibLayout stale;
        CHECK (calib_observe_axiswise (histograms, stale, lanes, nullptr)
               == Status::BAD_LAYOUT);
    }

    SECTION ("a geometry whose runs leave the channel range is refused")
    {
        alignas (8) CalibHistogram histograms[2];
        calib_reset (histograms[0]);
        calib_reset (histograms[1]);
        CalibLayout ragged;
        /* Two channels, runs of four, but twelve elements of leading space: the
           walk reaches channel 2, which the array does not have. */
        ragged._M_count = 12u;
        ragged._M_period = 12u;
        ragged._M_run = 4u;
        ragged._M_channels = 2u;
        ragged._M_axis = 1u;
        ragged._M_valid = 1u;
        CHECK (calib_observe_axiswise (histograms, ragged, lanes, nullptr)
               == Status::BAD_LAYOUT);

        CalibLayout empty_run = layout;
        empty_run._M_run = 0u;
        CHECK (calib_observe_axiswise (histograms, empty_run, lanes, nullptr)
               == Status::BAD_LAYOUT);
        CalibLayout empty_period = layout;
        empty_period._M_period = 0u;
        CHECK (calib_observe_axiswise (histograms, empty_period, lanes, nullptr)
               == Status::BAD_LAYOUT);
        CalibLayout empty_channels = layout;
        empty_channels._M_channels = 0u;
        CHECK (calib_observe_axiswise (histograms, empty_channels, lanes, nullptr)
               == Status::BAD_LAYOUT);
    }

    SECTION ("a truncated buffer folds the short tail run")
    {
        CalibLayout tail;
        tail._M_count = 10u;
        tail._M_period = 12u;
        tail._M_run = 4u;
        tail._M_channels = 3u;
        tail._M_axis = 1u;
        tail._M_valid = 1u;
        alignas (8) CalibHistogram histograms[3];
        for (uint32_t channel = 0; channel < 3u; ++channel)
            calib_reset (histograms[channel]);
        REQUIRE (status_ok (calib_observe_axiswise (histograms, tail, lanes,
                                                    nullptr)));
        RefHistogram reference[3];
        reference[0].fold_range (lanes, 0u, 4u);
        reference[1].fold_range (lanes, 4u, 8u);
        reference[2].fold_range (lanes, 8u, 10u);
        reference[0].end_batch (4u);
        reference[1].end_batch (4u);
        reference[2].end_batch (2u);
        for (uint32_t channel = 0; channel < 3u; ++channel)
            check_histogram (histograms[channel], reference[channel]);
    }
}
