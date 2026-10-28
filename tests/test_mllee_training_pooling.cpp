/* Tests for mllee/train_pooling.hpp: max, average, L2, adaptive-average and
 * global pooling, rank 1 and rank 2, both layouts.
 *
 * Every expectation is computed by a reference written inside this file
 * (`ref_forward` / `ref_backward`), never by the library under test: the
 * reference implements the documented math -- first-tap-wins argmax, the
 * count_include_pad divisor choice, isqrt (round (sum of squares / area)) for
 * L2 and the per-kind backward factors -- with its own stride arithmetic, its
 * own round-half-up division and its own saturating accumulator.  A refactor
 * that moves a rounding rule or a tie-break has to disagree here. */

#include "mllee/train_graph.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_pooling.hpp"
#include "mllee/train_tensor.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

using namespace zzs::mllee;
using namespace zzs::mllee::train;

namespace
{
/* ---- independent fixed-point helpers ---------------------------------- */

constexpr int32_t _REF_Q15_MAX = 32767;
constexpr int32_t _REF_Q15_MIN = -32768;
constexpr int32_t _REF_CODE_MAX = 127;
constexpr int32_t _REF_CODE_MIN = -127;

/** @brief Round half away from zero, the rule the pooling docs state. */
int32_t
ref_div_round (int32_t numerator, const int32_t denominator)
{
    if (denominator == 0)
        return 0;
    const bool negative = (numerator < 0) != (denominator < 0);
    const int64_t n = numerator < 0 ? -static_cast<int64_t> (numerator)
                                    : static_cast<int64_t> (numerator);
    const int64_t d = denominator < 0 ? -static_cast<int64_t> (denominator)
                                      : static_cast<int64_t> (denominator);
    const int64_t quotient = (n + d / 2) / d;
    return static_cast<int32_t> (negative ? -quotient : quotient);
}

/** @brief Floor square root by bisection, independent of the library table. */
int32_t
ref_isqrt (const uint32_t value)
{
    uint32_t root = 0u;
    while ((root + 1u) * (root + 1u) <= value)
        ++root;
    return static_cast<int32_t> (root);
}

int32_t
ref_sat16 (const int32_t value)
{
    if (value > _REF_Q15_MAX)
        return _REF_Q15_MAX;
    if (value < _REF_Q15_MIN)
        return _REF_Q15_MIN;
    return value;
}

int32_t
ref_clamp_code (const int32_t value)
{
    if (value > _REF_CODE_MAX)
        return _REF_CODE_MAX;
    if (value < _REF_CODE_MIN)
        return _REF_CODE_MIN;
    return value;
}

/** @brief int64 floor division used by the adaptive window rule. */
uint32_t
ref_floor_div (const uint32_t a, const uint32_t b)
{
    return b == 0u ? 0u : a / b;
}

/** @brief int64 ceil division used by the adaptive window rule. */
uint32_t
ref_ceil_div (const uint32_t a, const uint32_t b)
{
    return b == 0u ? 0u : (a + b - 1u) / b;
}

/* ---- geometry, written from the documented formulas -------------------- */

uint32_t
ref_axes (const PoolSpec &spec)
{
    return spec._M_rank == 2u ? 2u : 1u;
}

bool
ref_is_global (const PoolKind kind)
{
    return kind == PoolKind::GLOBAL_MAX || kind == PoolKind::GLOBAL_AVERAGE
           || kind == PoolKind::GLOBAL_L2;
}

bool
ref_is_max (const PoolKind kind)
{
    return kind == PoolKind::MAX || kind == PoolKind::GLOBAL_MAX;
}

bool
ref_is_l2 (const PoolKind kind)
{
    return kind == PoolKind::L2 || kind == PoolKind::GLOBAL_L2;
}

uint32_t
ref_out_length (const PoolSpec &spec, const uint32_t axis)
{
    if (axis >= ref_axes (spec))
        return 1u;
    if (spec._M_out[axis] != 0u)
        return spec._M_out[axis];
    if (ref_is_global (spec._M_kind))
        return 1u;
    const int32_t span = static_cast<int32_t> (spec._M_in[axis])
                         + spec._M_pad[axis * 2u] + spec._M_pad[axis * 2u + 1u]
                         - spec._M_kernel[axis];
    if (span < 0)
        return 0u;
    const int32_t placed = span / spec._M_stride[axis] + 1;
    const int32_t limit = static_cast<int32_t> (spec._M_in[axis]);
    return static_cast<uint32_t> (placed < limit ? placed : limit);
}

/** @brief Window first coordinate + tap count of output position @p flat. */
struct RefWindow
{
    int32_t _M_start[2];
    uint32_t _M_length[2];
};

RefWindow
ref_window (const PoolSpec &spec, const uint32_t flat)
{
    const uint32_t out1 = ref_out_length (spec, 1u);
    RefWindow window{ { 0, 0 }, { 1u, 1u } };
    const uint32_t coord[2] = { flat / out1, flat % out1 };
    for (uint32_t axis = 0; axis < 2u; ++axis)
        {
            if (axis >= ref_axes (spec))
                {
                    window._M_start[axis] = 0;
                    window._M_length[axis] = 1u;
                    continue;
                }
            const uint32_t in = spec._M_in[axis];
            const uint32_t out = ref_out_length (spec, axis);
            if (ref_is_global (spec._M_kind))
                {
                    window._M_start[axis] = 0;
                    window._M_length[axis] = in;
                }
            else if (spec._M_kind == PoolKind::ADAPTIVE_AVERAGE)
                {
                    const uint32_t start = ref_floor_div (in * coord[axis], out);
                    uint32_t end = ref_ceil_div (in * (coord[axis] + 1u), out);
                    if (end <= start)
                        end = start + 1u;
                    window._M_start[axis] = static_cast<int32_t> (start);
                    window._M_length[axis] = end - start;
                }
            else
                {
                    window._M_start[axis]
                        = static_cast<int32_t> (coord[axis]
                                                * spec._M_stride[axis])
                          - spec._M_pad[axis * 2u];
                    window._M_length[axis] = spec._M_kernel[axis];
                }
        }
    return window;
}

/** @brief Input element address of (item, channel, a0, a1) in memory order. */
uint32_t
ref_input_index (const PoolSpec &spec, const uint32_t item, const uint32_t ch,
                 const uint32_t a0, const uint32_t a1)
{
    const uint32_t c = spec._M_channels;
    if (spec._M_layout == PoolLayout::CHANNEL_MAJOR)
        return item * c * spec._M_in[0] * spec._M_in[1]
               + ch * spec._M_in[0] * spec._M_in[1] + a0 * spec._M_in[1] + a1;
    return item * spec._M_in[0] * spec._M_in[1] * c + a0 * spec._M_in[1] * c
           + a1 * c + ch;
}

/** @brief Output element address of (item, channel, a0, a1) in memory order. */
uint32_t
ref_output_index (const PoolSpec &spec, const uint32_t item, const uint32_t ch,
                  const uint32_t a0, const uint32_t a1)
{
    const uint32_t c = spec._M_channels;
    const uint32_t o0 = ref_out_length (spec, 0u);
    const uint32_t o1 = ref_out_length (spec, 1u);
    if (spec._M_layout == PoolLayout::CHANNEL_MAJOR)
        return item * c * o0 * o1 + ch * o0 * o1 + a0 * o1 + a1;
    return item * o0 * o1 * c + a0 * o1 * c + a1 * c + ch;
}

/* ---- reference forward -------------------------------------------------- */

/** @brief Codes, saved window state and the argmax coordinates of a pass. */
struct RefForward
{
    std::vector<int8_t> _M_code;
    std::vector<uint16_t> _M_state;
    std::vector<uint32_t> _M_winner;
    std::vector<uint32_t> _M_area;
    std::vector<uint32_t> _M_inside;
};

RefForward
ref_forward (const PoolSpec &spec, const std::vector<int8_t> &codes)
{
    const uint32_t o0 = ref_out_length (spec, 0u);
    const uint32_t o1 = ref_out_length (spec, 1u);
    const uint32_t plane = o0 * o1;
    RefForward ref;
    ref._M_code.assign (static_cast<size_t> (spec._M_batch) * spec._M_channels
                            * plane,
                        int8_t (0));
    const size_t total = ref._M_code.size ();
    ref._M_state.assign (total, 0u);
    ref._M_winner.assign (total, 0u);
    ref._M_area.assign (total, 0u);
    ref._M_inside.assign (total, 0u);

    size_t cursor = 0u;
    for (uint32_t item = 0u; item < spec._M_batch; ++item)
        for (uint32_t ch = 0u; ch < spec._M_channels; ++ch)
            for (uint32_t flat = 0u; flat < plane; ++flat, ++cursor)
                {
                    const RefWindow window = ref_window (spec, flat);
                    int32_t sum = 0;
                    int32_t square = 0;
                    int32_t best = 0;
                    bool have_best = false;
                    uint32_t best_plane = 0u;
                    uint32_t inside = 0u;
                    for (uint32_t t0 = 0u; t0 < window._M_length[0]; ++t0)
                        for (uint32_t t1 = 0u; t1 < window._M_length[1]; ++t1)
                            {
                                const int32_t s0 = window._M_start[0]
                                                   + static_cast<int32_t> (t0);
                                const int32_t s1 = window._M_start[1]
                                                   + static_cast<int32_t> (t1);
                                if (s0 < 0 || s1 < 0
                                    || s0 >= static_cast<int32_t> (spec._M_in[0])
                                    || s1 >= static_cast<int32_t> (spec._M_in[1]))
                                    continue;
                                const uint32_t a0 = static_cast<uint32_t> (s0);
                                const uint32_t a1 = static_cast<uint32_t> (s1);
                                const int32_t value
                                    = codes[ref_input_index (spec, item, ch, a0,
                                                             a1)];
                                ++inside;
                                if (ref_is_max (spec._M_kind))
                                    {
                                        if (!have_best || value > best)
                                            {
                                                best = value;
                                                best_plane = a0 * spec._M_in[1]
                                                             + a1;
                                                have_best = true;
                                            }
                                        continue;
                                    }
                                sum += value;
                                square += value * value;
                            }
                    const uint32_t nominal
                        = window._M_length[0] * window._M_length[1];
                    const uint32_t area
                        = spec._M_count_include_pad != 0u
                              ? (nominal != 0u ? nominal : 1u)
                              : (inside != 0u ? inside : 1u);
                    ref._M_state[cursor] = ref_is_max (spec._M_kind)
                                               ? static_cast<uint16_t> (best_plane)
                                               : static_cast<uint16_t> (area);
                    ref._M_winner[cursor] = best_plane;
                    ref._M_area[cursor] = area;
                    ref._M_inside[cursor] = inside;
                    const uint32_t a0 = o1 != 0u ? flat / o1 : 0u;
                    const uint32_t a1 = o1 != 0u ? flat % o1 : flat;
                    int32_t code = 0;
                    if (ref_is_max (spec._M_kind))
                        code = best;
                    else if (ref_is_l2 (spec._M_kind))
                        code = ref_isqrt (static_cast<uint32_t> (
                            ref_div_round (square, static_cast<int32_t> (area))));
                    else
                        code = ref_div_round (sum, static_cast<int32_t> (area));
                    ref._M_code[ref_output_index (spec, item, ch, a0, a1)]
                        = static_cast<int8_t> (ref_clamp_code (code));
                }
    return ref;
}

/* ---- reference backward -------------------------------------------------- */

/** @brief Input gradient lanes in memory order plus the saturation count. */
struct RefBackward
{
    std::vector<int32_t> _M_grad;
    uint32_t _M_saturated = 0u;
};

/** @brief Saturating accumulate that counts every clamped write. */
void
ref_acc (std::vector<int32_t> &grad, const size_t index, const int32_t delta,
         uint32_t &saturated)
{
    const int64_t sum = static_cast<int64_t> (grad[index])
                        + static_cast<int64_t> (delta);
    if (sum > _REF_Q15_MAX || sum < _REF_Q15_MIN)
        {
            grad[index] = ref_sat16 (static_cast<int32_t> (
                sum > _REF_Q15_MAX ? _REF_Q15_MAX : _REF_Q15_MIN));
            ++saturated;
        }
    else
        grad[index] = static_cast<int32_t> (sum);
}

/**
 * @brief Independent backward replay of the documented factors.
 *
 * The max path folds every output lane into its saved winner before touching the
 * operand (one saturating add per input plane element, like the scatter), the
 * broadcast kinds add tap by tap in window order (like `route_by_mask`).
 */
RefBackward
ref_backward (const PoolSpec &spec, const RefForward &fwd,
              const std::vector<int8_t> &out_codes,
              const std::vector<int32_t> &grad_out,
              const std::vector<int8_t> &codes,
              const std::vector<int32_t> &start)
{
    const uint32_t o0 = ref_out_length (spec, 0u);
    const uint32_t o1 = ref_out_length (spec, 1u);
    const uint32_t plane = o0 * o1;
    RefBackward ref;
    ref._M_grad = start;
    size_t cursor = 0u;
    for (uint32_t item = 0u; item < spec._M_batch; ++item)
        for (uint32_t ch = 0u; ch < spec._M_channels; ++ch)
            {
                if (ref_is_max (spec._M_kind))
                    {
                        /* The library folds the plane into an int16 lane buffer
                           first (scatter_by_index) and only then adds each
                           non-zero lane into the operand, so the reference
                           replays exactly those two saturating stages. */
                        std::vector<int32_t> lane (spec.in_plane (), 0);
                        for (uint32_t flat = 0u; flat < plane; ++flat, ++cursor)
                            {
                                const uint32_t out_index
                                    = ref_output_index (spec, item, ch,
                                                        o1 ? flat / o1 : 0u,
                                                        o1 ? flat % o1 : flat);
                                const int64_t sum
                                    = static_cast<int64_t> (lane[fwd._M_winner[cursor]])
                                      + static_cast<int64_t> (grad_out[out_index]);
                                if (sum > _REF_Q15_MAX || sum < _REF_Q15_MIN)
                                    {
                                        lane[fwd._M_winner[cursor]]
                                            = sum > _REF_Q15_MAX ? _REF_Q15_MAX
                                                                  : _REF_Q15_MIN;
                                        ++ref._M_saturated;
                                    }
                                else
                                    lane[fwd._M_winner[cursor]]
                                        = static_cast<int32_t> (sum);
                            }
                        for (uint32_t q = 0u; q < spec.in_plane (); ++q)
                            {
                                if (lane[q] == 0)
                                    continue;
                                const uint32_t a0 = spec._M_in[1] ? q / spec._M_in[1] : 0u;
                                const uint32_t a1 = spec._M_in[1] ? q % spec._M_in[1] : q;
                                ref_acc (ref._M_grad,
                                         ref_input_index (spec, item, ch, a0, a1),
                                         lane[q], ref._M_saturated);
                            }
                        continue;
                    }
                for (uint32_t flat = 0u; flat < plane; ++flat, ++cursor)
                    {
                        const uint32_t out_index
                            = ref_output_index (spec, item, ch, o1 ? flat / o1 : 0u,
                                                o1 ? flat % o1 : flat);
                        const int32_t grad = grad_out[out_index];
                        const uint32_t area = fwd._M_area[cursor];
                        const int32_t y = out_codes[out_index];
                        if (area == 0u || grad == 0)
                            continue;
                        const RefWindow window = ref_window (spec, flat);
                        for (uint32_t t0 = 0u; t0 < window._M_length[0]; ++t0)
                            for (uint32_t t1 = 0u; t1 < window._M_length[1]; ++t1)
                                {
                                    const int32_t s0 = window._M_start[0]
                                                       + static_cast<int32_t> (t0);
                                    const int32_t s1 = window._M_start[1]
                                                       + static_cast<int32_t> (t1);
                                    if (s0 < 0 || s1 < 0
                                        || s0 >= static_cast<int32_t> (spec._M_in[0])
                                        || s1 >= static_cast<int32_t> (spec._M_in[1]))
                                        continue;
                                    const uint32_t a0 = static_cast<uint32_t> (s0);
                                    const uint32_t a1 = static_cast<uint32_t> (s1);
                                    const uint32_t in_index
                                        = ref_input_index (spec, item, ch, a0, a1);
                                    int32_t lane = 0;
                                    if (ref_is_l2 (spec._M_kind))
                                        lane = y > 0
                                                   ? ref_div_round (
                                                       grad * static_cast<int32_t> (
                                                           codes[in_index]),
                                                       static_cast<int32_t> (area)
                                                           * y)
                                                   : 0;
                                    else
                                        lane = ref_div_round (grad,
                                                              static_cast<int32_t> (area));
                                    const int32_t stored = ref_sat16 (lane);
                                    if (stored == 0)
                                        continue;
                                    ref_acc (ref._M_grad, in_index, stored,
                                             ref._M_saturated);
                                }
                    }
            }
    return ref;
}

/* ---- buffers and tape harness -------------------------------------------- */

struct Rig
{
    alignas (16) uint8_t _M_arena[16384];
    Pool _M_pool { _M_arena, sizeof (Rig::_M_arena) };
    Node _M_nodes[8];
    Tape _M_tape { _M_nodes, 8u, _M_pool };
};

/** @brief Backward context view of the single recorded node of @p rig. */
GradContext
context_of (Rig &rig)
{
    Node *const node = rig._M_tape.node (0u);
    REQUIRE (node != nullptr);
    GradContext context;
    context._M_pool = &rig._M_pool;
    context._M_operands = node->_M_operands;
    context._M_operand_count = node->_M_operand_count;
    context._M_saved = node->_M_saved;
    context._M_saved_count = node->_M_saved_count;
    context._M_loss_scale_q15 = _S_q15_one;
    return context;
}

struct Views
{
    alignas (16) int8_t _M_in_code[4096];
    alignas (16) int16_t _M_in_grad[4096];
    alignas (16) int8_t _M_out_code[4096];
    alignas (16) int16_t _M_out_grad[4096];
};

Shape
shape_of (const PoolSpec &spec, const bool input)
{
    const uint32_t l0 = input ? spec._M_in[0] : ref_out_length (spec, 0u);
    const uint32_t l1 = input ? spec._M_in[1] : ref_out_length (spec, 1u);
    return spec._M_rank == 1u
               ? Shape::tensor3 (spec._M_batch, spec._M_channels,
                                 static_cast<uint16_t> (l0))
               : Shape::tensor4 (spec._M_batch, spec._M_channels,
                                 static_cast<uint16_t> (l0),
                                 static_cast<uint16_t> (l1));
}

/** @brief Codes 1..N folded into [-127,127] with a visible sign pattern. */
std::vector<int8_t>
fill_codes (const uint32_t count, const uint32_t seed)
{
    std::vector<int8_t> codes (count);
    for (uint32_t i = 0u; i < count; ++i)
        codes[i] = static_cast<int8_t> (
            ref_clamp_code (static_cast<int32_t> ((i * 17u + seed * 11u) % 240u)
                            - 119));
    return codes;
}

/** @brief Run one forward and compare it, element by element, to the reference. */
void
check_forward_matches (const PoolSpec &spec, const uint32_t seed)
{
    REQUIRE (status_ok (spec.validate ()));
    Views views;
    const std::vector<int8_t> codes = fill_codes (spec.input_elements (), seed);
    for (uint32_t i = 0u; i < spec.input_elements (); ++i)
        views._M_in_code[i] = codes[i];
    const RefForward ref = ref_forward (spec, codes);
    std::vector<uint16_t> state (spec.output_elements (), 0xFFFFu);
    Trainable input
        = Trainable::make (views._M_in_code, nullptr, shape_of (spec, true), 0u);
    Trainable output
        = Trainable::make (views._M_out_code, views._M_out_grad,
                           shape_of (spec, false), 0u);
    REQUIRE (status_ok (pool_forward (spec, input, output, state.data ())));
    REQUIRE (ref._M_code.size () == spec.output_elements ());
    for (uint32_t i = 0u; i < spec.output_elements (); ++i)
        {
            CHECK (output._M_code[i] == ref._M_code[i]);
            CHECK (state[i] == ref._M_state[i]);
            /* the -128 code is never emitted */
            CHECK (output._M_code[i] != -128);
        }
    for (uint32_t i = 0u; i < spec.input_elements (); ++i)
        CHECK (views._M_in_code[i] == codes[i]);
}

/** @brief A non-zero gradient pattern the Q15 lanes can hold exactly. */
std::vector<int32_t>
fill_grads (const uint32_t count, const uint32_t seed, const uint32_t span)
{
    std::vector<int32_t> grads (count);
    for (uint32_t i = 0u; i < count; ++i)
        {
            int32_t value = static_cast<int32_t> ((i * 611u + seed * 977u)
                                                  % (2u * span + 1u))
                            - static_cast<int32_t> (span);
            if (value == 0)
                value = 7;
            grads[i] = value;
        }
    return grads;
}

/**
 * @brief Record a real pooling node and replay its backward against the
 *        reference, comparing every input lane and the saturation count.
 */
void
check_backward_matches (const PoolSpec &spec, const uint32_t seed,
                        const uint32_t grad_span)
{
    REQUIRE (status_ok (spec.validate ()));
    Rig rig;
    Views views;
    const std::vector<int8_t> codes = fill_codes (spec.input_elements (), seed);
    for (uint32_t i = 0u; i < spec.input_elements (); ++i)
        {
            views._M_in_code[i] = codes[i];
            views._M_in_grad[i] = 0;
        }
    Trainable input = Trainable::make (views._M_in_code, views._M_in_grad,
                                       shape_of (spec, true), Flag::RequiresGrad);
    Trainable output = Trainable::make (views._M_out_code, views._M_out_grad,
                                        shape_of (spec, false),
                                        Flag::RequiresGrad);
    REQUIRE (status_ok (rig._M_tape.begin ()));
    REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));

    const std::vector<int32_t> grads
        = fill_grads (spec.output_elements (), seed, grad_span);
    for (uint32_t i = 0u; i < spec.output_elements (); ++i)
        output._M_grad[i] = static_cast<int16_t> (grads[i]);

    GradContext context = context_of (rig);
    const RefForward fwd = ref_forward (spec, codes);
    const uint16_t *const state = context.saved_as<uint16_t> (_S_pool_window);
    REQUIRE (state != nullptr);
    for (uint32_t i = 0u; i < spec.output_elements (); ++i)
        {
            CHECK (state[i] == fwd._M_state[i]);
            CHECK (output._M_code[i] == fwd._M_code[i]);
        }
    const RefBackward ref
        = ref_backward (spec, fwd, fwd._M_code, grads, codes,
                        std::vector<int32_t> (spec.input_elements (), 0));
    REQUIRE (status_ok (pool_backward (output, output, context)));
    for (uint32_t i = 0u; i < spec.input_elements (); ++i)
        CHECK (input._M_grad[i] == ref._M_grad[i]);
    CHECK (context._M_saturation_events == ref._M_saturated);
}
} // namespace

TEST_CASE ("PoolSpec derives output lengths from kernel, stride and padding")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::MAX;
    spec._M_rank = 1u;
    spec._M_in[0] = 8u;
    spec._M_kernel[0] = 2u;
    spec._M_stride[0] = 2u;

    SECTION ("exact fit")
    {
        REQUIRE (spec.out_length (0u) == 4u);
        REQUIRE (spec.out_length (1u) == 1u);
        REQUIRE (spec.out_length (2u) == 1u);
    }
    SECTION ("stride one keeps in - kernel + 1 positions")
    {
        spec._M_stride[0] = 1u;
        REQUIRE (spec.out_length (0u) == 7u);
    }
    SECTION ("asymmetric padding enters the span")
    {
        spec._M_pad[0] = 1;
        spec._M_pad[1] = 2;
        /* span = 8 + 3 - 2 = 9, so floor (9 / 2) + 1 = 5 placements */
        REQUIRE (spec.out_length (0u) == 5u);
        REQUIRE (ref_out_length (spec, 0u) == 5u);
    }
    SECTION ("the placed < limit clamp caps a kernel-1 window")
    {
        spec._M_kernel[0] = 1u;
        spec._M_stride[0] = 1u;
        spec._M_pad[0] = 4;
        spec._M_pad[1] = 4;
        /* span / stride + 1 = 15 but no output may start past the input */
        REQUIRE (spec.out_length (0u) == 8u);
    }
    SECTION ("a span shorter than the input still reports the placed count")
    {
        spec._M_kernel[0] = 3u;
        spec._M_stride[0] = 3u;
        REQUIRE (spec.out_length (0u) == 2u);
    }
    SECTION ("a negative span is a zero length")
    {
        spec._M_kernel[0] = 9u;
        REQUIRE (spec.out_length (0u) == 0u);
    }
    SECTION ("an explicit output wins over the derivation")
    {
        spec._M_out[0] = 3u;
        REQUIRE (spec.out_length (0u) == 3u);
    }
}

TEST_CASE ("Global and adaptive kinds report their own output lengths")
{
    PoolSpec spec;
    spec._M_rank = 1u;
    spec._M_in[0] = 12u;
    spec._M_kernel[0] = 3u;

    SECTION ("every global kind returns one position per axis")
    {
        for (const PoolKind kind : { PoolKind::GLOBAL_MAX,
                                     PoolKind::GLOBAL_AVERAGE,
                                     PoolKind::GLOBAL_L2 })
            {
                spec._M_kind = kind;
                spec._M_out[0] = 0u;
                REQUIRE (spec.out_length (0u) == 1u);
                REQUIRE (spec.out_length (1u) == 1u);
                REQUIRE (spec.out_plane () == 1u);
            }
    }
    SECTION ("adaptive reads the requested length")
    {
        spec._M_kind = PoolKind::ADAPTIVE_AVERAGE;
        spec._M_out[0] = 5u;
        REQUIRE (spec.out_length (0u) == 5u);
        REQUIRE (spec.input_elements () == 12u);
        REQUIRE (spec.output_elements () == 5u);
    }
    SECTION ("a global kind with an explicit output still reports it")
    {
        spec._M_kind = PoolKind::GLOBAL_AVERAGE;
        spec._M_out[0] = 2u;
        REQUIRE (spec.out_length (0u) == 2u);
        /* ... and validate rejects it, see the validate test case */
        REQUIRE (spec.validate () == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("Kind predicates agree with the documented sets")
{
    SECTION ("max kinds")
    {
        REQUIRE (pool_is_max (PoolKind::MAX));
        REQUIRE (pool_is_max (PoolKind::GLOBAL_MAX));
        REQUIRE_FALSE (pool_is_max (PoolKind::AVERAGE));
        REQUIRE_FALSE (pool_is_max (PoolKind::L2));
        REQUIRE_FALSE (pool_is_max (PoolKind::ADAPTIVE_AVERAGE));
        REQUIRE_FALSE (pool_is_max (PoolKind::GLOBAL_AVERAGE));
        REQUIRE_FALSE (pool_is_max (PoolKind::GLOBAL_L2));
    }
    SECTION ("global kinds")
    {
        REQUIRE_FALSE (pool_is_global (PoolKind::MAX));
        REQUIRE_FALSE (pool_is_global (PoolKind::AVERAGE));
        REQUIRE_FALSE (pool_is_global (PoolKind::L2));
        REQUIRE_FALSE (pool_is_global (PoolKind::ADAPTIVE_AVERAGE));
        REQUIRE (pool_is_global (PoolKind::GLOBAL_MAX));
        REQUIRE (pool_is_global (PoolKind::GLOBAL_AVERAGE));
        REQUIRE (pool_is_global (PoolKind::GLOBAL_L2));
    }
    SECTION ("input-recording kinds are exactly the L2 pair")
    {
        REQUIRE_FALSE (pool_needs_input (PoolKind::MAX));
        REQUIRE_FALSE (pool_needs_input (PoolKind::AVERAGE));
        REQUIRE_FALSE (pool_needs_input (PoolKind::ADAPTIVE_AVERAGE));
        REQUIRE_FALSE (pool_needs_input (PoolKind::GLOBAL_MAX));
        REQUIRE_FALSE (pool_needs_input (PoolKind::GLOBAL_AVERAGE));
        REQUIRE (pool_needs_input (PoolKind::L2));
        REQUIRE (pool_needs_input (PoolKind::GLOBAL_L2));
        REQUIRE (pool_is_l2 (PoolKind::L2));
        REQUIRE (pool_is_l2 (PoolKind::GLOBAL_L2));
        REQUIRE_FALSE (pool_is_l2 (PoolKind::AVERAGE));
    }
    SECTION ("the enum range is the valid kind range")
    {
        for (uint8_t raw = 0u; raw <= 6u; ++raw)
            REQUIRE (pool_kind_valid (static_cast<PoolKind> (raw)));
        REQUIRE_FALSE (pool_kind_valid (static_cast<PoolKind> (7u)));
        REQUIRE_FALSE (pool_kind_valid (static_cast<PoolKind> (200u)));
    }
    SECTION ("the tags stay single bytes")
    {
        static_assert (sizeof (PoolKind) == 1u);
        static_assert (sizeof (PoolLayout) == 1u);
        static_assert (sizeof (PoolSpec) == 28u);
        static_assert (sizeof (PoolWindow) == 16u);
        static_assert (sizeof (PoolTap) == 12u);
        static_assert (sizeof (PoolStrides) == 16u);
        REQUIRE (static_cast<uint8_t> (PoolKind::MAX) == 0u);
        REQUIRE (static_cast<uint8_t> (PoolKind::GLOBAL_L2) == 6u);
        REQUIRE (static_cast<uint8_t> (PoolLayout::SPATIAL_MAJOR) == 1u);
    }
}

TEST_CASE ("Max pooling rank 1 channel-major forward matches the reference")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::MAX;
    spec._M_rank = 1u;
    spec._M_batch = 2u;
    spec._M_channels = 3u;
    spec._M_in[0] = 8u;
    spec._M_kernel[0] = 2u;
    spec._M_stride[0] = 2u;
    spec._M_layout = PoolLayout::CHANNEL_MAJOR;
    REQUIRE (status_ok (spec.validate ()));

    Views views;
    const std::vector<int8_t> codes = fill_codes (spec.input_elements (), 3u);
    for (uint32_t i = 0u; i < spec.input_elements (); ++i)
        views._M_in_code[i] = codes[i];

    const RefForward ref = ref_forward (spec, codes);
    std::vector<uint16_t> state (spec.output_elements (), 0xFFFFu);
    Trainable input
        = Trainable::make (views._M_in_code, nullptr, shape_of (spec, true), 0u);
    Trainable output
        = Trainable::make (views._M_out_code, views._M_out_grad,
                           shape_of (spec, false), 0u);

    SECTION ("the forward succeeds and codes every window maximum")
    {
        REQUIRE (status_ok (pool_forward (spec, input, output, state.data ())));
        REQUIRE (ref._M_code.size () == spec.output_elements ());
        for (uint32_t i = 0u; i < spec.output_elements (); ++i)
            {
                REQUIRE (output._M_code[i] == ref._M_code[i]);
                REQUIRE (static_cast<int32_t> (output._M_code[i]) >= -127);
                REQUIRE (static_cast<int32_t> (output._M_code[i]) <= 127);
            }
    }
    SECTION ("the saved state is the winner plane coordinate")
    {
        REQUIRE (status_ok (pool_forward (spec, input, output, state.data ())));
        for (uint32_t i = 0u; i < spec.output_elements (); ++i)
            REQUIRE (state[i] == ref._M_state[i]);
    }
    SECTION ("the forward zeroes the output gradient")
    {
        for (uint32_t i = 0u; i < spec.output_elements (); ++i)
            output._M_grad[i] = static_cast<int16_t> (i + 1u);
        REQUIRE (status_ok (pool_forward (spec, input, output, state.data ())));
        for (uint32_t i = 0u; i < spec.output_elements (); ++i)
            REQUIRE (output._M_grad[i] == 0);
    }
    SECTION ("a null window state is refused")
    {
        REQUIRE (pool_forward (spec, input, output, nullptr)
                 == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("pool_window reproduces every documented window rule")
{
    SECTION ("fixed windows step by stride and shift by the low pad")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::MAX;
        spec._M_rank = 1u;
        spec._M_in[0] = 16u;
        spec._M_kernel[0] = 3u;
        spec._M_stride[0] = 2u;
        spec._M_pad[0] = 1;
        const uint32_t out = spec.out_length (0u);
        REQUIRE (out == 8u);
        for (uint32_t flat = 0u; flat < out; ++flat)
            {
                const PoolWindow window = pool_window (spec, flat);
                const RefWindow ref = ref_window (spec, flat);
                CHECK (window._M_start[0] == ref._M_start[0]);
                CHECK (window._M_length[0] == ref._M_length[0]);
                CHECK (window._M_start[0] == static_cast<int32_t> (flat * 2u) - 1);
                CHECK (window._M_length[0] == 3u);
                /* rank 1 pins the second axis */
                CHECK (window._M_start[1] == 0);
                CHECK (window._M_length[1] == 1u);
            }
    }
    SECTION ("rank 2 decodes flat into (a0, a1) with the row-major out plane")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::AVERAGE;
        spec._M_rank = 2u;
        spec._M_batch = 2u;
        spec._M_channels = 2u;
        spec._M_in[0] = 5u;
        spec._M_in[1] = 4u;
        spec._M_kernel[0] = 3u;
        spec._M_kernel[1] = 2u;
        spec._M_stride[0] = 2u;
        spec._M_stride[1] = 1u;
        REQUIRE (spec.out_length (0u) == 2u);
        REQUIRE (spec.out_length (1u) == 3u);
        REQUIRE (spec.out_plane () == 6u);
        for (uint32_t flat = 0u; flat < spec.out_plane (); ++flat)
            {
                const PoolWindow window = pool_window (spec, flat);
                const RefWindow ref = ref_window (spec, flat);
                CHECK (window._M_start[0] == ref._M_start[0]);
                CHECK (window._M_start[1] == ref._M_start[1]);
                CHECK (window._M_length[0] == 3u);
                CHECK (window._M_length[1] == 2u);
                CHECK (window._M_start[0]
                       == static_cast<int32_t> ((flat / 3u) * 2u));
                CHECK (window._M_start[1] == static_cast<int32_t> (flat % 3u));
            }
    }
    SECTION ("adaptive windows start at floor and end at ceil")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::ADAPTIVE_AVERAGE;
        spec._M_rank = 1u;
        spec._M_in[0] = 5u;
        spec._M_out[0] = 3u;
        const uint32_t start_ref[3] = { 0u, 1u, 3u };
        const uint32_t length_ref[3] = { 2u, 3u, 2u };
        for (uint32_t flat = 0u; flat < 3u; ++flat)
            {
                const PoolWindow window = pool_window (spec, flat);
                CHECK (window._M_start[0]
                       == static_cast<int32_t> (start_ref[flat]));
                CHECK (window._M_length[0] == length_ref[flat]);
                CHECK (window._M_start[0] == ref_window (spec, flat)._M_start[0]);
                CHECK (window._M_length[0] == ref_window (spec, flat)._M_length[0]);
            }
        /* consecutive windows share at most one position */
        CHECK (pool_window (spec, 0u)._M_start[0] + 2u
               == pool_window (spec, 1u)._M_start[0] + 1u);
        CHECK (pool_window (spec, 1u)._M_length[0] == 3u);
    }
    SECTION ("global windows cover the whole plane")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::GLOBAL_AVERAGE;
        spec._M_rank = 2u;
        spec._M_in[0] = 3u;
        spec._M_in[1] = 4u;
        spec._M_kernel[0] = 9u;
        spec._M_stride[0] = 5u;
        const PoolWindow window = pool_window (spec, 0u);
        CHECK (window._M_start[0] == 0);
        CHECK (window._M_start[1] == 0);
        CHECK (window._M_length[0] == 3u);
        CHECK (window._M_length[1] == 4u);
        CHECK (spec.out_plane () == 1u);
        /* rank 1 global pins axis 1 */
        spec._M_rank = 1u;
        spec._M_in[1] = 1u;
        const PoolWindow line = pool_window (spec, 0u);
        CHECK (line._M_length[0] == 3u);
        CHECK (line._M_length[1] == 1u);
    }
    SECTION ("a zero output length never divides the flat coordinate")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::AVERAGE;
        spec._M_rank = 1u;
        spec._M_in[0] = 2u;
        spec._M_kernel[0] = 5u;
        REQUIRE (spec.out_length (0u) == 0u);
        const PoolWindow window = pool_window (spec, 0u);
        CHECK (window._M_start[0] == 0);
        CHECK (window._M_length[0] == 5u);
    }
}

TEST_CASE ("pool_tap reports padding instead of addressing it")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::AVERAGE;
    spec._M_rank = 2u;
    spec._M_batch = 2u;
    spec._M_channels = 3u;
    spec._M_in[0] = 4u;
    spec._M_in[1] = 5u;
    spec._M_kernel[0] = 3u;
    spec._M_kernel[1] = 3u;
    spec._M_stride[0] = 1u;
    spec._M_stride[1] = 1u;
    spec._M_pad[0] = 1;
    spec._M_pad[1] = 1;
    spec._M_pad[2] = 1;
    spec._M_pad[3] = 1;
    REQUIRE (spec.validate () == Status::OK);

    const PoolStrides strides = pool_input_strides (spec);
    uint32_t checked_inside = 0u;
    uint32_t checked_outside = 0u;
    for (uint32_t item = 0u; item < spec._M_batch; ++item)
        for (uint32_t ch = 0u; ch < spec._M_channels; ++ch)
            for (uint32_t flat = 0u; flat < spec.out_plane (); ++flat)
                {
                    const PoolWindow window = pool_window (spec, flat);
                    for (uint32_t t0 = 0u; t0 < window._M_length[0]; ++t0)
                        for (uint32_t t1 = 0u; t1 < window._M_length[1]; ++t1)
                            {
                                const PoolTap tap = pool_tap (spec, strides,
                                                              window, item, ch,
                                                              t0, t1);
                                const int32_t s0 = window._M_start[0]
                                                   + static_cast<int32_t> (t0);
                                const int32_t s1 = window._M_start[1]
                                                   + static_cast<int32_t> (t1);
                                const bool inside = s0 >= 0 && s1 >= 0
                                                    && s0
                                                       < static_cast<int32_t> (
                                                           spec._M_in[0])
                                                    && s1
                                                       < static_cast<int32_t> (
                                                           spec._M_in[1]);
                                CHECK (tap._M_inside == inside);
                                if (!inside)
                                    {
                                        ++checked_outside;
                                        /* a rejected tap keeps the default value,
                                           so no read or write can leave the
                                           tensor */
                                        CHECK (tap._M_element == 0u);
                                        CHECK (tap._M_plane_index == 0u);
                                        continue;
                                    }
                                    ++checked_inside;
                                CHECK (tap._M_plane_index
                                       == static_cast<uint32_t> (s0) * spec._M_in[1]
                                              + static_cast<uint32_t> (s1));
                                CHECK (tap._M_element == ref_input_index (
                                             spec, item, ch,
                                             static_cast<uint32_t> (s0),
                                             static_cast<uint32_t> (s1)));
                            }
                }
    CHECK (checked_inside > 0u);
    CHECK (checked_outside > 0u);
}

TEST_CASE ("PoolStrides and pool_element agree with the layout element order")
{
    for (const PoolLayout layout : { PoolLayout::CHANNEL_MAJOR,
                                     PoolLayout::SPATIAL_MAJOR })
        for (const uint8_t rank : { 1u, 2u })
            {
                PoolSpec spec;
                spec._M_kind = PoolKind::AVERAGE;
                spec._M_rank = rank;
                spec._M_batch = 2u;
                spec._M_channels = 3u;
                spec._M_in[0] = 5u;
                spec._M_in[1] = rank == 2u ? 4u : 1u;
                spec._M_kernel[0] = 3u;
                spec._M_kernel[1] = 1u;
                spec._M_stride[0] = 2u;
                spec._M_stride[1] = 1u;
                spec._M_layout = layout;
                REQUIRE (spec.validate () == Status::OK);

                const PoolStrides in = pool_input_strides (spec);
                const PoolStrides out = pool_output_strides (spec);
                const uint32_t o0 = spec.out_length (0u);
                const uint32_t o1 = spec.out_length (1u);
                for (uint32_t item = 0u; item < spec._M_batch; ++item)
                    for (uint32_t ch = 0u; ch < spec._M_channels; ++ch)
                        {
                            for (uint32_t a0 = 0u; a0 < spec._M_in[0]; ++a0)
                                for (uint32_t a1 = 0u; a1 < spec._M_in[1]; ++a1)
                                    CHECK (pool_element (in, item, ch, a0, a1)
                                           == ref_input_index (spec, item, ch,
                                                               a0, a1));
                            for (uint32_t a0 = 0u; a0 < o0; ++a0)
                                for (uint32_t a1 = 0u; a1 < o1; ++a1)
                                    CHECK (pool_element (out, item, ch, a0, a1)
                                           == ref_output_index (spec, item, ch,
                                                               a0, a1));
                        }
                CHECK (in._M_batch == spec.input_elements () / spec._M_batch);
                CHECK (out._M_batch == spec.output_elements () / spec._M_batch);
            }
}

TEST_CASE ("Average pooling divides by the nominal area with count_include_pad")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::AVERAGE;
    spec._M_rank = 1u;
    spec._M_in[0] = 3u;
    spec._M_kernel[0] = 3u;
    spec._M_stride[0] = 1u;
    spec._M_pad[0] = 1;
    REQUIRE (spec.out_length (0u) == 2u);

    Views views;
    views._M_in_code[0] = 1;
    views._M_in_code[1] = -2;
    views._M_in_code[2] = 3;
    Trainable input
        = Trainable::make (views._M_in_code, nullptr, shape_of (spec, true), 0u);
    Trainable output
        = Trainable::make (views._M_out_code, views._M_out_grad,
                           shape_of (spec, false), 0u);
    std::vector<uint16_t> state (spec.output_elements (), 0u);

    SECTION ("padding counts in the divisor and the border mean shrinks")
    {
        spec._M_count_include_pad = 1u;
        REQUIRE (status_ok (pool_forward (spec, input, output, state.data ())));
        /* window 0 reads pad, 1, -2: sum -1 over the nominal area 3 */
        CHECK (state[0] == 3u);
        CHECK (output._M_code[0] == 0);
        CHECK (output._M_code[1] == 1);
        CHECK (ref_div_round (-1, 3) == 0);
    }
    SECTION ("counting only in-border taps keeps the border mean honest")
    {
        spec._M_count_include_pad = 0u;
        REQUIRE (status_ok (pool_forward (spec, input, output, state.data ())));
        CHECK (state[0] == 2u);
        CHECK (state[1] == 3u);
        CHECK (output._M_code[0] == -1);
        CHECK (output._M_code[1] == 1);
        CHECK (ref_div_round (-1, 2) == -1);
    }
    SECTION ("the full sweep agrees with the reference for both rules")
    {
        for (const uint8_t include_pad : { 0u, 1u })
            {
                spec._M_count_include_pad = include_pad;
                spec._M_channels = 2u;
                spec._M_in[0] = 8u;
                spec._M_kernel[0] = 3u;
                spec._M_pad[0] = 1;
                spec._M_pad[1] = 1;
                spec._M_out[0] = 0u;
                check_forward_matches (spec, include_pad);
            }
    }
}

TEST_CASE ("Average pooling rank 2 border windows report the clipped divisor")
{
    for (const PoolLayout layout : { PoolLayout::CHANNEL_MAJOR,
                                     PoolLayout::SPATIAL_MAJOR })
        for (const uint8_t include_pad : { 0u, 1u })
            {
                PoolSpec spec;
                spec._M_kind = PoolKind::AVERAGE;
                spec._M_rank = 2u;
                spec._M_batch = 2u;
                spec._M_channels = 3u;
                spec._M_in[0] = 5u;
                spec._M_in[1] = 4u;
                spec._M_kernel[0] = 3u;
                spec._M_kernel[1] = 3u;
                spec._M_stride[0] = 1u;
                spec._M_stride[1] = 2u;
                spec._M_pad[0] = 1;
                spec._M_pad[1] = 1;
                spec._M_pad[2] = 1;
                spec._M_pad[3] = 1;
                spec._M_layout = layout;
                spec._M_count_include_pad = include_pad;
                INFO ("layout " << static_cast<int> (layout) << " pad "
                                << int (include_pad));
                CHECK (spec.out_length (0u) == 5u);
                CHECK (spec.out_length (1u) == 2u);
                check_forward_matches (spec, include_pad + 2u);
            }
}

TEST_CASE ("Max pooling rank 2 matches the reference in both layouts")
{
    for (const PoolLayout layout : { PoolLayout::CHANNEL_MAJOR,
                                     PoolLayout::SPATIAL_MAJOR })
        {
            PoolSpec spec;
            spec._M_kind = PoolKind::MAX;
            spec._M_rank = 2u;
            spec._M_batch = 2u;
            spec._M_channels = 3u;
            spec._M_in[0] = 6u;
            spec._M_in[1] = 5u;
            spec._M_kernel[0] = 3u;
            spec._M_kernel[1] = 2u;
            spec._M_stride[0] = 2u;
            spec._M_stride[1] = 2u;
            spec._M_layout = layout;
            INFO ("layout " << static_cast<int> (layout));
            CHECK (spec.out_length (0u) == 2u);
            CHECK (spec.out_length (1u) == 2u);
            CHECK (spec.max_window () == 6u);
            check_forward_matches (spec, 5u);
        }

    SECTION ("a padded max window ignores the padding entirely")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::MAX;
        spec._M_rank = 1u;
        spec._M_in[0] = 6u;
        spec._M_kernel[0] = 3u;
        spec._M_stride[0] = 1u;
        spec._M_pad[0] = 2;
        spec._M_pad[1] = 2;
        REQUIRE (spec.out_length (0u) == 6u);
        check_forward_matches (spec, 1u);
    }
}

TEST_CASE ("Max pooling ties go to the first tap in window order")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::MAX;
    spec._M_rank = 1u;
    spec._M_in[0] = 6u;
    spec._M_kernel[0] = 4u;
    spec._M_stride[0] = 2u;
    REQUIRE (spec.out_length (0u) == 2u);

    Views views;
    for (uint32_t i = 0u; i < 6u; ++i)
        views._M_in_code[i] = 7;
    views._M_in_code[3] = 20;
    views._M_in_code[5] = 20;
    Trainable input
        = Trainable::make (views._M_in_code, nullptr, shape_of (spec, true), 0u);
    Trainable output
        = Trainable::make (views._M_out_code, views._M_out_grad,
                           shape_of (spec, false), 0u);
    uint16_t state[2] = { 999u, 999u };
    REQUIRE (status_ok (pool_forward (spec, input, output, state)));

    SECTION ("the first equal maximum wins the window")
    {
        CHECK (output._M_code[0] == 20);
        CHECK (output._M_code[1] == 20);
        /* window 0 covers 0..3, so the tie is decided by index 3 */
        CHECK (state[0] == 3u);
        /* window 1 covers 2..5, and the first 20 it meets is still index 3 */
        CHECK (state[1] == 3u);
    }
    SECTION ("the recorded state is exactly what the backward replays")
    {
        const RefForward ref = ref_forward (spec, { views._M_in_code,
                                                    views._M_in_code + 6 });
        CHECK (ref._M_state[0] == state[0]);
        CHECK (ref._M_state[1] == state[1]);
        CHECK (ref._M_code[0] == output._M_code[0]);
        CHECK (ref._M_code[1] == output._M_code[1]);
    }
}

TEST_CASE ("L2 pooling is the integer root of the rounded mean square")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::L2;
    spec._M_rank = 1u;
    spec._M_in[0] = 4u;
    spec._M_kernel[0] = 2u;
    spec._M_stride[0] = 2u;
    REQUIRE (spec.out_length (0u) == 2u);

    Views views;
    views._M_in_code[0] = 3;
    views._M_in_code[1] = 4;
    views._M_in_code[2] = 0;
    views._M_in_code[3] = -5;
    Trainable input
        = Trainable::make (views._M_in_code, nullptr, shape_of (spec, true), 0u);
    Trainable output
        = Trainable::make (views._M_out_code, views._M_out_grad,
                           shape_of (spec, false), 0u);
    uint16_t state[2] = { 0u, 0u };
    REQUIRE (status_ok (pool_forward (spec, input, output, state)));

    SECTION ("the mean is rounded before the root")
    {
        /* (9 + 16) / 2 = 12.5 -> 13 -> floor root 3 */
        CHECK (state[0] == 2u);
        CHECK (output._M_code[0] == 3);
        CHECK (ref_isqrt (static_cast<uint32_t> (ref_div_round (25, 2))) == 3);
        /* (0 + 25) / 2 = 12.5 -> 13 -> floor root 3 */
        CHECK (state[1] == 2u);
        CHECK (output._M_code[1] == 3);
    }
    SECTION ("rank 2 and both layouts follow the same reference")
    {
        for (const PoolLayout layout : { PoolLayout::CHANNEL_MAJOR,
                                         PoolLayout::SPATIAL_MAJOR })
            {
                PoolSpec wide;
                wide._M_kind = PoolKind::L2;
                wide._M_rank = 2u;
                wide._M_batch = 2u;
                wide._M_channels = 2u;
                wide._M_in[0] = 5u;
                wide._M_in[1] = 4u;
                wide._M_kernel[0] = 3u;
                wide._M_kernel[1] = 2u;
                wide._M_stride[0] = 2u;
                wide._M_stride[1] = 1u;
                wide._M_layout = layout;
                check_forward_matches (wide, 4u);
            }
        PoolSpec line;
        line._M_kind = PoolKind::L2;
        line._M_rank = 1u;
        line._M_batch = 2u;
        line._M_channels = 2u;
        line._M_in[0] = 12u;
        line._M_kernel[0] = 3u;
        line._M_stride[0] = 1u;
        line._M_pad[0] = 1;
        line._M_pad[1] = 1;
        line._M_count_include_pad = 0u;
        check_forward_matches (line, 6u);
        line._M_count_include_pad = 1u;
        check_forward_matches (line, 6u);
    }
}

TEST_CASE ("Adaptive average windows overlap by at most one position")
{
    for (const uint8_t rank : { 1u, 2u })
        for (const PoolLayout layout : { PoolLayout::CHANNEL_MAJOR,
                                         PoolLayout::SPATIAL_MAJOR })
            {
                PoolSpec spec;
                spec._M_kind = PoolKind::ADAPTIVE_AVERAGE;
                spec._M_rank = rank;
                spec._M_batch = 2u;
                spec._M_channels = 2u;
                spec._M_in[0] = 7u;
                spec._M_in[1] = rank == 2u ? 5u : 1u;
                spec._M_out[0] = 3u;
                spec._M_out[1] = rank == 2u ? 2u : 0u;
                spec._M_layout = layout;
                INFO ("rank " << int (rank) << " layout "
                              << static_cast<int> (spec._M_layout));
                REQUIRE (spec.validate () == Status::OK);
                CHECK (spec.out_length (0u) == 3u);
                CHECK (spec.out_length (1u) == (rank == 2u ? 2u : 1u));
                CHECK (spec.output_elements () == 3u * spec.out_length (1u) * 4u);
                /* padding is not part of the adaptive rule */
                spec._M_pad[0] = 1;
                CHECK (spec.validate () == Status::BAD_LAYOUT);
                spec._M_pad[0] = 0;
                /* a kernel wider than the input is rejected too */
                spec._M_kernel[0] = 8u;
                CHECK (spec.validate () == Status::BAD_LAYOUT);
                spec._M_kernel[0] = 1u;
                check_forward_matches (spec, 9u);
            }

    SECTION ("overlapping windows split a shared tap across both outputs")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::ADAPTIVE_AVERAGE;
        spec._M_rank = 1u;
        spec._M_in[0] = 5u;
        spec._M_out[0] = 3u;
        const uint32_t starts[3] = { 0u, 1u, 3u };
        const uint32_t ends[3] = { 2u, 4u, 5u };
        for (uint32_t flat = 0u; flat < 3u; ++flat)
            {
                const PoolWindow window = pool_window (spec, flat);
                CHECK (window._M_start[0] == static_cast<int32_t> (starts[flat]));
                CHECK (static_cast<uint32_t> (window._M_start[0]
                                              + static_cast<int32_t> (
                                                  window._M_length[0]))
                       == ends[flat]);
                if (flat + 1u < 3u)
                    CHECK (window._M_start[0] + static_cast<int32_t> (
                               window._M_length[0])
                           >= pool_window (spec, flat + 1u)._M_start[0]);
            }
        CHECK (spec.max_window () >= 3u);
        check_forward_matches (spec, 2u);
    }
}

TEST_CASE ("Global kinds reduce the whole plane to one position per channel")
{
    for (const PoolKind kind : { PoolKind::GLOBAL_MAX, PoolKind::GLOBAL_AVERAGE,
                                 PoolKind::GLOBAL_L2 })
        for (const uint8_t rank : { 1u, 2u })
            for (const PoolLayout layout : { PoolLayout::CHANNEL_MAJOR,
                                             PoolLayout::SPATIAL_MAJOR })
                {
                    PoolSpec spec;
                    spec._M_kind = kind;
                    spec._M_rank = rank;
                    spec._M_batch = 2u;
                    spec._M_channels = 3u;
                    spec._M_in[0] = 5u;
                    spec._M_in[1] = rank == 2u ? 4u : 1u;
                    spec._M_kernel[0] = 3u;
                    spec._M_stride[0] = 2u;
                    spec._M_layout = layout;
                    INFO ("kind " << static_cast<int> (kind) << " rank "
                                  << int (rank));
                    REQUIRE (spec.validate () == Status::OK);
                    CHECK (spec.out_plane () == 1u);
                    CHECK (spec.output_elements () == 6u);
                    CHECK (spec.input_elements () == 6u * spec.in_plane ());
                    /* the global kinds ignore kernel, stride and a wide plane */
                    CHECK (spec.out_length (0u) == 1u);
                    check_forward_matches (spec, rank + static_cast<uint32_t> (kind));
                }

    SECTION ("a global plane wider than the window bound still validates")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::GLOBAL_AVERAGE;
        spec._M_rank = 1u;
        spec._M_in[0] = 200u;
        REQUIRE (spec.max_window () > _S_pool_window_max);
        REQUIRE (spec.validate () == Status::OK);
        check_forward_matches (spec, 7u);
    }
    SECTION ("max_window reports the axis extent for a global layer")
    {
        /* the forward never bounds a global window, so the number is only
           informational; it stays the axis-0 extent even for rank 2 */
        PoolSpec spec;
        spec._M_kind = PoolKind::GLOBAL_MAX;
        spec._M_rank = 2u;
        spec._M_in[0] = 3u;
        spec._M_in[1] = 4u;
        CHECK (spec.max_window () == 3u);
        CHECK (spec.in_plane () == 12u);
    }
}

TEST_CASE ("Max pooling backward scatters into the saved argmax coordinate")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::MAX;
    spec._M_rank = 1u;
    spec._M_in[0] = 5u;
    spec._M_kernel[0] = 3u;
    spec._M_stride[0] = 1u;
    REQUIRE (spec.out_length (0u) == 3u);

    Rig rig;
    Views views;
    const int8_t hand[5] = { 1, 5, 2, 5, 0 };
    for (uint32_t i = 0u; i < 5u; ++i)
        {
            views._M_in_code[i] = hand[i];
            views._M_in_grad[i] = 0;
        }
    Trainable input
        = Trainable::make (views._M_in_code, views._M_in_grad,
                           shape_of (spec, true), Flag::RequiresGrad);
    Trainable output
        = Trainable::make (views._M_out_code, views._M_out_grad,
                           shape_of (spec, false), Flag::RequiresGrad);
    REQUIRE (status_ok (rig._M_tape.begin ()));
    REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));

    SECTION ("the forward recorded the first maximum of each window")
    {
        CHECK (output._M_code[0] == 5);
        CHECK (output._M_code[1] == 5);
        CHECK (output._M_code[2] == 5);
        GradContext context = context_of (rig);
        const uint16_t *const state
            = context.saved_as<uint16_t> (_S_pool_window);
        REQUIRE (state != nullptr);
        CHECK (state[0] == 1u);
        CHECK (state[1] == 1u);
        CHECK (state[2] == 3u);
    }
    SECTION ("each output lane lands on its own winner")
    {
        output._M_grad[0] = 1000;
        output._M_grad[1] = 0;
        output._M_grad[2] = 0;
        GradContext context = context_of (rig);
        REQUIRE (status_ok (pool_backward (output, output, context)));
        CHECK (input._M_grad[1] == 1000);
        CHECK (input._M_grad[0] == 0);
        CHECK (input._M_grad[2] == 0);
        CHECK (input._M_grad[3] == 0);
        CHECK (input._M_grad[4] == 0);
        CHECK (context._M_saturation_events == 0u);
    }
    SECTION ("a winner that no output picked keeps a zero lane")
    {
        output._M_grad[0] = 0;
        output._M_grad[1] = 0;
        output._M_grad[2] = 333;
        GradContext context = context_of (rig);
        REQUIRE (status_ok (pool_backward (output, output, context)));
        CHECK (input._M_grad[3] == 333);
        CHECK (input._M_grad[1] == 0);
        CHECK (input._M_grad[2] == 0);
    }
    SECTION ("the full pattern matches the independent replay")
    {
        const std::vector<int32_t> grads = { 1000, 2000, 4000 };
        for (uint32_t i = 0u; i < 3u; ++i)
            output._M_grad[i] = static_cast<int16_t> (grads[i]);
        GradContext context = context_of (rig);
        REQUIRE (status_ok (pool_backward (output, output, context)));
        /* hand-computed: outputs 0 and 1 both picked input 1 */
        CHECK (input._M_grad[1] == 3000);
        CHECK (input._M_grad[3] == 4000);
        const RefForward fwd = ref_forward (spec, { hand, hand + 5 });
        const RefBackward ref
            = ref_backward (spec, fwd, fwd._M_code, grads, { hand, hand + 5 },
                            std::vector<int32_t> (5, 0));
        for (uint32_t i = 0u; i < 5u; ++i)
            CHECK (input._M_grad[i] == ref._M_grad[i]);
    }
}

TEST_CASE ("Overlapping max windows both land on the shared input")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::MAX;
    spec._M_rank = 1u;
    spec._M_batch = 2u;
    spec._M_channels = 2u;
    spec._M_in[0] = 9u;
    spec._M_kernel[0] = 3u;
    spec._M_stride[0] = 1u;
    REQUIRE (spec.out_length (0u) == 7u);
    REQUIRE (spec.validate () == Status::OK);

    Rig rig;
    Views views;
    /* one tall spike per plane so every window that covers it picks it */
    for (uint32_t i = 0u; i < spec.input_elements (); ++i)
        {
            views._M_in_code[i] = static_cast<int8_t> (
                ref_clamp_code (static_cast<int32_t> (i % 5u) * 3 - 6));
            views._M_in_grad[i] = 0;
        }
    for (uint32_t item = 0u; item < 2u; ++item)
        for (uint32_t ch = 0u; ch < 2u; ++ch)
            views._M_in_code[item * 18u + ch * 9u + 4u] = 100;
    Trainable input
        = Trainable::make (views._M_in_code, views._M_in_grad,
                           shape_of (spec, true), Flag::RequiresGrad);
    Trainable output
        = Trainable::make (views._M_out_code, views._M_out_grad,
                           shape_of (spec, false), Flag::RequiresGrad);
    REQUIRE (status_ok (rig._M_tape.begin ()));
    REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));

    const std::vector<int32_t> grads
        = fill_grads (spec.output_elements (), 3u, 900u);
    for (uint32_t i = 0u; i < spec.output_elements (); ++i)
        output._M_grad[i] = static_cast<int16_t> (grads[i]);
    GradContext context = context_of (rig);
    const RefForward fwd = ref_forward (
        spec, { views._M_in_code, views._M_in_code + spec.input_elements () });
    const uint16_t *const state = context.saved_as<uint16_t> (_S_pool_window);
    REQUIRE (state != nullptr);
    for (uint32_t i = 0u; i < spec.output_elements (); ++i)
        CHECK (state[i] == fwd._M_state[i]);
    REQUIRE (status_ok (pool_backward (output, output, context)));

    SECTION ("the spike collects one contribution per window that chose it")
    {
        /* windows 2, 3 and 4 of every plane cover plane index 4, so the spike
           receives three lanes -- the scatter adds, it never overwrites */
        uint32_t pickers = 0u;
        for (uint32_t i = 0u; i < spec.output_elements (); ++i)
            if (state[i] == 4u)
                ++pickers;
        /* three windows per plane, four planes */
        CHECK (pickers == 12u);
        for (uint32_t item = 0u; item < 2u; ++item)
            for (uint32_t ch = 0u; ch < 2u; ++ch)
                {
                    int32_t expected = 0;
                    uint32_t counted = 0u;
                    for (uint32_t flat = 0u; flat < 7u; ++flat)
                        {
                            const uint32_t cursor
                                = (item * 2u + ch) * 7u + flat;
                            if (state[cursor] != 4u)
                                continue;
                            expected += grads[cursor];
                            ++counted;
                        }
                    CHECK (counted == 3u);
                    CHECK (input._M_grad[item * 18u + ch * 9u + 4u] == expected);
                }
    }
    SECTION ("every lane matches the reference and nothing saturated")
    {
        const std::vector<int8_t> codes { views._M_in_code,
                                          views._M_in_code
                                              + spec.input_elements () };
        const RefBackward ref
            = ref_backward (spec, fwd, fwd._M_code, grads, codes,
                            std::vector<int32_t> (spec.input_elements (), 0));
        for (uint32_t i = 0u; i < spec.input_elements (); ++i)
            CHECK (input._M_grad[i] == ref._M_grad[i]);
        CHECK (context._M_saturation_events == 0u);
        CHECK (ref._M_saturated == 0u);
    }
}

TEST_CASE ("Backward matches the reference for every kind, rank and layout")
{
    for (const PoolKind kind : { PoolKind::MAX, PoolKind::AVERAGE, PoolKind::L2,
                                 PoolKind::ADAPTIVE_AVERAGE, PoolKind::GLOBAL_MAX,
                                 PoolKind::GLOBAL_AVERAGE, PoolKind::GLOBAL_L2 })
        for (const uint8_t rank : { 1u, 2u })
            for (const PoolLayout layout : { PoolLayout::CHANNEL_MAJOR,
                                             PoolLayout::SPATIAL_MAJOR })
                {
                    PoolSpec spec;
                    spec._M_kind = kind;
                    spec._M_rank = rank;
                    spec._M_batch = 2u;
                    spec._M_channels = 2u;
                    spec._M_in[0] = 12u;
                    spec._M_in[1] = rank == 2u ? 6u : 1u;
                    spec._M_kernel[0] = 3u;
                    spec._M_kernel[1] = rank == 2u ? 2u : 1u;
                    spec._M_stride[0] = 2u;
                    spec._M_stride[1] = 1u;
                    spec._M_layout = layout;
                    if (kind == PoolKind::ADAPTIVE_AVERAGE)
                        {
                            spec._M_out[0] = 3u;
                            spec._M_out[1] = rank == 2u ? 2u : 0u;
                        }
                    INFO ("kind " << static_cast<int> (kind) << " rank "
                                  << int (rank) << " layout "
                                  << static_cast<int> (layout));
                    CHECK (spec.validate () == Status::OK);
                    check_backward_matches (spec, rank + static_cast<uint32_t> (kind),
                                            900u);
                    spec._M_count_include_pad = 0u;
                    if (kind == PoolKind::AVERAGE || kind == PoolKind::L2)
                        {
                            spec._M_pad[0] = 1;
                            spec._M_pad[1] = 1;
                            if (rank == 2u)
                                {
                                    spec._M_pad[2] = 1;
                                    spec._M_pad[3] = 1;
                                }
                            check_backward_matches (spec, 5u, 900u);
                        }
                }
}

TEST_CASE ("The L2 backward factor vanishes where the pooled value is zero")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::L2;
    spec._M_rank = 1u;
    spec._M_in[0] = 6u;
    spec._M_kernel[0] = 2u;
    spec._M_stride[0] = 2u;
    REQUIRE (spec.out_length (0u) == 3u);

    Rig rig;
    Views views;
    const int8_t hand[6] = { 0, 0, 3, -4, 6, 8 };
    for (uint32_t i = 0u; i < 6u; ++i)
        {
            views._M_in_code[i] = hand[i];
            views._M_in_grad[i] = 0;
        }
    Trainable input
        = Trainable::make (views._M_in_code, views._M_in_grad,
                           shape_of (spec, true), Flag::RequiresGrad);
    Trainable output
        = Trainable::make (views._M_out_code, views._M_out_grad,
                           shape_of (spec, false), Flag::RequiresGrad);
    REQUIRE (status_ok (rig._M_tape.begin ()));
    REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));

    SECTION ("a zero pooled output has no slope")
    {
        CHECK (output._M_code[0] == 0);
        /* (9 + 16) / 2 = 12.5 -> 13 -> floor root 3 */
        CHECK (output._M_code[1] == 3);
        CHECK (output._M_code[2] == 7);
        CHECK (ref_isqrt (static_cast<uint32_t> (ref_div_round (25, 2))) == 3);
        CHECK (ref_isqrt (static_cast<uint32_t> (ref_div_round (100, 2))) == 7);
        output._M_grad[0] = 5000;
        output._M_grad[1] = 0;
        output._M_grad[2] = 0;
        GradContext context = context_of (rig);
        REQUIRE (status_ok (pool_backward (output, output, context)));
        CHECK (input._M_grad[0] == 0);
        CHECK (input._M_grad[1] == 0);
        CHECK (context._M_saturation_events == 0u);
    }
    SECTION ("a non-zero output splits x_t / (area * y) onto each tap")
    {
        output._M_grad[0] = 0;
        output._M_grad[1] = 8000;
        output._M_grad[2] = 0;
        GradContext context = context_of (rig);
        REQUIRE (status_ok (pool_backward (output, output, context)));
        /* window 1 = (3, -4), y = 3, area = 2: 8000 * 3 / 6 = 4000 */
        CHECK (output._M_code[1] == 3);
        CHECK (ref_div_round (8000 * 3, 2 * 3) == 4000);
        CHECK (input._M_grad[2] == 4000);
        /* 8000 * -4 / 6 rounds half up on the magnitude to -5333 */
        CHECK (ref_div_round (-32000, 6) == -5333);
        CHECK (input._M_grad[3] == -5333);
        CHECK (input._M_grad[4] == 0);
        const std::vector<int8_t> codes { hand, hand + 6 };
        const RefForward fwd = ref_forward (spec, codes);
        const RefBackward ref = ref_backward (
            spec, fwd, fwd._M_code, std::vector<int32_t> { 0, 8000, 0 }, codes,
            std::vector<int32_t> (6, 0));
        for (uint32_t i = 0u; i < 6u; ++i)
            CHECK (input._M_grad[i] == ref._M_grad[i]);
    }
    SECTION ("the saved input copy is what the L2 backward reads")
    {
        GradContext context = context_of (rig);
        const int8_t *const saved = context.saved_as<int8_t> (_S_pool_input);
        REQUIRE (saved != nullptr);
        for (uint32_t i = 0u; i < 6u; ++i)
            CHECK (saved[i] == hand[i]);
        CHECK (context.saved_as<uint16_t> (_S_pool_window) != nullptr);
        check_backward_matches (spec, 1u, 700u);
    }
}

TEST_CASE ("Saturating adds are counted where gradients exceed a Q15 lane")
{
    SECTION ("two pickers of a max window clamp the fold exactly once")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::MAX;
        spec._M_rank = 1u;
        spec._M_in[0] = 5u;
        spec._M_kernel[0] = 3u;
        spec._M_stride[0] = 1u;
        Rig rig;
        Views views;
        const int8_t hand[5] = { 1, 5, 2, 5, 0 };
        for (uint32_t i = 0u; i < 5u; ++i)
            {
                views._M_in_code[i] = hand[i];
                views._M_in_grad[i] = 0;
            }
        Trainable input
            = Trainable::make (views._M_in_code, views._M_in_grad,
                               shape_of (spec, true), Flag::RequiresGrad);
        Trainable output = Trainable::make (views._M_out_code,
                                            views._M_out_grad,
                                            shape_of (spec, false), 0u);
        REQUIRE (status_ok (rig._M_tape.begin ()));
        REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));
        output._M_grad[0] = 20000;
        output._M_grad[1] = 20000;
        output._M_grad[2] = 100;
        GradContext context = context_of (rig);
        REQUIRE (status_ok (pool_backward (output, output, context)));
        /* the int16 fold of the shared winner clamps once, then the operand add
           is exact because the lane already fits */
        CHECK (input._M_grad[1] == 32767);
        CHECK (input._M_grad[3] == 100);
        CHECK (context._M_saturation_events == 1u);
        const std::vector<int8_t> codes { hand, hand + 5 };
        const RefForward fwd = ref_forward (spec, codes);
        const RefBackward ref
            = ref_backward (spec, fwd, fwd._M_code,
                            std::vector<int32_t> { 20000, 20000, 100 }, codes,
                            std::vector<int32_t> (5, 0));
        CHECK (ref._M_saturated == 1u);
        for (uint32_t i = 0u; i < 5u; ++i)
            CHECK (input._M_grad[i] == ref._M_grad[i]);
    }
    SECTION ("an average split accumulates past the lane ceiling")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::AVERAGE;
        spec._M_rank = 1u;
        spec._M_in[0] = 4u;
        spec._M_kernel[0] = 2u;
        spec._M_stride[0] = 1u;
        REQUIRE (spec.out_length (0u) == 3u);
        Rig rig;
        Views views;
        for (uint32_t i = 0u; i < 4u; ++i)
            {
                views._M_in_code[i] = static_cast<int8_t> (i + 1);
                views._M_in_grad[i] = 0;
            }
        Trainable input
            = Trainable::make (views._M_in_code, views._M_in_grad,
                               shape_of (spec, true), Flag::RequiresGrad);
        Trainable output = Trainable::make (views._M_out_code,
                                            views._M_out_grad,
                                            shape_of (spec, false), 0u);
        REQUIRE (status_ok (rig._M_tape.begin ()));
        REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));
        for (uint32_t i = 0u; i < 3u; ++i)
            output._M_grad[i] = 32767;
        GradContext context = context_of (rig);
        REQUIRE (status_ok (pool_backward (output, output, context)));
        /* each window of 2 contributes (32767 + 1) / 2 = 16384 per tap, so an
           interior input gets 2 * 16384 = 32768 and clamps */
        CHECK (input._M_grad[0] == 16384);
        CHECK (input._M_grad[1] == 32767);
        CHECK (input._M_grad[2] == 32767);
        CHECK (input._M_grad[3] == 16384);
        CHECK (context._M_saturation_events == 2u);
    }
    SECTION ("the reference sees the same clamp count on a wide sweep")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::MAX;
        spec._M_rank = 2u;
        spec._M_batch = 2u;
        spec._M_channels = 2u;
        spec._M_in[0] = 5u;
        spec._M_in[1] = 4u;
        spec._M_kernel[0] = 3u;
        spec._M_kernel[1] = 2u;
        spec._M_stride[0] = 1u;
        spec._M_stride[1] = 1u;
        check_backward_matches (spec, 4u, 32000u);
        spec._M_kind = PoolKind::AVERAGE;
        check_backward_matches (spec, 4u, 32000u);
    }
}

TEST_CASE ("Global pooling backward broadcasts over the whole plane")
{
    for (const PoolKind kind : { PoolKind::GLOBAL_MAX, PoolKind::GLOBAL_AVERAGE,
                                 PoolKind::GLOBAL_L2 })
        for (const uint8_t rank : { 1u, 2u })
            {
                PoolSpec spec;
                spec._M_kind = kind;
                spec._M_rank = rank;
                spec._M_batch = 2u;
                spec._M_channels = 2u;
                spec._M_in[0] = 9u;
                spec._M_in[1] = rank == 2u ? 5u : 1u;
                INFO ("kind " << static_cast<int> (kind) << " rank " << int (rank));
                REQUIRE (spec.validate () == Status::OK);
                CHECK (spec.out_plane () == 1u);
                check_backward_matches (spec, rank, 1500u);
            }

    SECTION ("a global average divides by the plane area, no window bound")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::GLOBAL_AVERAGE;
        spec._M_rank = 1u;
        spec._M_in[0] = 200u;
        spec._M_channels = 1u;
        spec._M_batch = 1u;
        REQUIRE (spec.max_window () > _S_pool_window_max);
        REQUIRE (spec.validate () == Status::OK);
        Rig rig;
        Views views;
        const std::vector<int8_t> codes = fill_codes (200u, 2u);
        for (uint32_t i = 0u; i < 200u; ++i)
            {
                views._M_in_code[i] = codes[i];
                views._M_in_grad[i] = 0;
            }
        Trainable input
            = Trainable::make (views._M_in_code, views._M_in_grad,
                               shape_of (spec, true), Flag::RequiresGrad);
        Trainable output = Trainable::make (views._M_out_code,
                                            views._M_out_grad,
                                            shape_of (spec, false), 0u);
        REQUIRE (status_ok (rig._M_tape.begin ()));
        REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));
        GradContext context = context_of (rig);
        const uint16_t *const state
            = context.saved_as<uint16_t> (_S_pool_window);
        REQUIRE (state != nullptr);
        CHECK (state[0] == 200u);
        output._M_grad[0] = 3000;
        REQUIRE (status_ok (pool_backward (output, output, context)));
        const int32_t lane = ref_div_round (3000, 200);
        CHECK (lane == 15);
        uint32_t non_zero = 0u;
        for (uint32_t i = 0u; i < 200u; ++i)
            {
                CHECK (input._M_grad[i] == lane);
                if (input._M_grad[i] != 0)
                    ++non_zero;
            }
        CHECK (non_zero == 200u);
        CHECK (context._M_saturation_events == 0u);
    }
}

TEST_CASE ("pool_backward re-validates the saved spec and the recorded slots")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::MAX;
    spec._M_rank = 1u;
    spec._M_in[0] = 8u;
    spec._M_kernel[0] = 2u;
    spec._M_stride[0] = 2u;
    Rig rig;
    Views views;
    for (uint32_t i = 0u; i < spec.input_elements (); ++i)
        {
            views._M_in_code[i] = static_cast<int8_t> (i - 3);
            views._M_in_grad[i] = 0;
        }
    Trainable input = Trainable::make (views._M_in_code, views._M_in_grad,
                                       shape_of (spec, true),
                                       Flag::RequiresGrad);
    Trainable output
        = Trainable::make (views._M_out_code, views._M_out_grad,
                           shape_of (spec, false), 0u);
    REQUIRE (status_ok (rig._M_tape.begin ()));
    REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));
    Node *const node = rig._M_tape.node (0u);
    REQUIRE (node != nullptr);

    auto context_with = [&](const Saved *saved, const uint32_t saved_count,
                            const Trainable *operands,
                            const uint32_t operand_count) {
        GradContext context;
        context._M_pool = &rig._M_pool;
        context._M_operands = operands;
        context._M_operand_count = operand_count;
        context._M_saved = saved;
        context._M_saved_count = saved_count;
        context._M_loss_scale_q15 = _S_q15_one;
        return context;
    };

    SECTION ("a missing spec slot is refused")
    {
        GradContext context = context_with (nullptr, 0u, node->_M_operands, 1u);
        REQUIRE (pool_backward (output, output, context)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("a spec slot too small for the struct is refused")
    {
        Saved shrunk = node->_M_saved[_S_pool_spec];
        shrunk._M_bytes = shrunk._M_bytes - 4u;
        GradContext context = context_with (&shrunk, 1u, node->_M_operands, 1u);
        REQUIRE (pool_backward (output, output, context)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("no operands is refused")
    {
        GradContext context = context_with (node->_M_saved,
                                            node->_M_saved_count, nullptr, 0u);
        REQUIRE (pool_backward (output, output, context)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("a frozen input with no gradient lanes is inert, not an error")
    {
        Trainable frozen = input;
        frozen._M_grad = nullptr;
        Trainable single[1] = { frozen };
        GradContext context = context_with (node->_M_saved,
                                            node->_M_saved_count, single, 1u);
        REQUIRE (pool_backward (output, output, context) == Status::OK);
    }
    SECTION ("a null grad_output stream is refused")
    {
        Trainable bare = Trainable::make (views._M_out_code, nullptr,
                                          shape_of (spec, false), 0u);
        GradContext context = context_with (node->_M_saved,
                                            node->_M_saved_count,
                                            node->_M_operands, 1u);
        REQUIRE (pool_backward (output, bare, context) == Status::BAD_LAYOUT);
    }
    SECTION ("counts that disagree with the spec are refused")
    {
        Trainable wrong_input = input;
        wrong_input._M_count = spec.input_elements () + 1u;
        Trainable single[1] = { wrong_input };
        GradContext context = context_with (node->_M_saved,
                                            node->_M_saved_count, single, 1u);
        REQUIRE (pool_backward (output, output, context)
                 == Status::BAD_LAYOUT);
        Trainable wrong_output = output;
        wrong_output._M_count = spec.output_elements () - 1u;
        GradContext wrong = context_with (node->_M_saved, node->_M_saved_count,
                                          node->_M_operands, 1u);
        REQUIRE (pool_backward (wrong_output, output, wrong)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("a window coordinate beyond the plane is refused")
    {
        uint16_t *const state = reinterpret_cast<uint16_t *> (
            rig._M_pool.at (node->_M_saved[_S_pool_window]._M_offset));
        REQUIRE (state != nullptr);
        const uint16_t keep = state[0];
        state[0] = static_cast<uint16_t> (spec.in_plane ());
        GradContext context = context_with (node->_M_saved,
                                            node->_M_saved_count,
                                            node->_M_operands, 1u);
        REQUIRE (pool_backward (output, output, context)
                 == Status::BAD_LAYOUT);
        state[0] = static_cast<uint16_t> (spec.in_plane () + 7u);
        REQUIRE (pool_backward (output, output, context)
                 == Status::BAD_LAYOUT);
        state[0] = keep;
        REQUIRE (status_ok (pool_backward (output, output, context)));
    }
    SECTION ("a max node without its scratch slab is refused")
    {
        Saved broken[4];
        for (uint32_t i = 0u; i < 4u; ++i)
            broken[i] = node->_M_saved[i];
        broken[_S_pool_scratch] = Saved{};
        GradContext context = context_with (broken, 4u, node->_M_operands, 1u);
        REQUIRE (pool_backward (output, output, context)
                 == Status::BAD_LAYOUT);
    }
    SECTION ("a spec with a 65-tap window is refused by the head validation")
    {
        /* max_window () equals the kernel product for the fixed kinds, so a
           window past _S_pool_window_max never reaches the frame loop: the
           re-validation at the head of pool_backward answers it first, and with
           the same status the guard documents. */
        PoolSpec wide;
        wide._M_kind = PoolKind::MAX;
        wide._M_rank = 1u;
        wide._M_in[0] = 70u;
        wide._M_kernel[0] = 65u;
        wide._M_stride[0] = 1u;
        REQUIRE (wide.out_length (0u) == 6u);
        REQUIRE (wide.max_window () == 65u);
        REQUIRE (wide.validate () == Status::OUT_OF_RANGE);
        alignas (16) uint8_t raw[64] = {};
        Pool scratch_pool (raw, sizeof (raw));
        Slab slab;
        REQUIRE (status_ok (scratch_pool.carve (1u, 28u, slab)));
        PoolSpec *const forged = reinterpret_cast<PoolSpec *> (
            scratch_pool.at (slab._M_offset));
        *forged = wide;
        Saved slot;
        slot._M_offset = slab._M_offset;
        slot._M_bytes = 28u;
        slot._M_count = 1u;
        GradContext context;
        context._M_pool = &scratch_pool;
        context._M_operands = node->_M_operands;
        context._M_operand_count = 1u;
        context._M_saved = &slot;
        context._M_saved_count = 1u;
        REQUIRE (pool_backward (output, output, context)
                 == Status::OUT_OF_RANGE);
        /* and the forward refuses the same geometry before it walks a tap */
        Trainable big_in = Trainable::make (views._M_in_code, nullptr,
                                            Shape::tensor3 (1u, 1u, 70u), 0u);
        Trainable big_out = Trainable::make (views._M_out_code, nullptr,
                                             Shape::tensor3 (1u, 1u, 6u), 0u);
        uint16_t state[6] = {};
        REQUIRE (pool_forward (wide, big_in, big_out, state)
                 == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("validate rejects exactly the documented geometry errors")
{
    auto base = [] {
        PoolSpec spec;
        spec._M_kind = PoolKind::MAX;
        spec._M_rank = 1u;
        spec._M_batch = 2u;
        spec._M_channels = 2u;
        spec._M_in[0] = 8u;
        spec._M_in[1] = 1u;
        spec._M_kernel[0] = 2u;
        spec._M_stride[0] = 2u;
        return spec;
    };
    REQUIRE (base ().validate () == Status::OK);

    SECTION ("an unsupported kind")
    {
        PoolSpec spec = base ();
        spec._M_kind = static_cast<PoolKind> (7u);
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec._M_kind = static_cast<PoolKind> (255u);
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("a rank outside one and two")
    {
        PoolSpec spec = base ();
        spec._M_rank = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec._M_rank = 2u;
        spec._M_in[1] = 4u;
        CHECK (spec.validate () == Status::OK);
        spec._M_rank = 3u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("a zero batch or channel count")
    {
        PoolSpec spec = base ();
        spec._M_batch = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec._M_batch = 2u;
        spec._M_channels = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("a zero spatial extent")
    {
        PoolSpec spec = base ();
        spec._M_in[0] = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec._M_in[0] = 8u;
        spec._M_in[1] = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("an unknown layout")
    {
        PoolSpec spec = base ();
        spec._M_layout = static_cast<PoolLayout> (2u);
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec._M_layout = PoolLayout::SPATIAL_MAJOR;
        CHECK (spec.validate () == Status::OK);
    }
    SECTION ("rank one must pin the second axis")
    {
        PoolSpec loose = base ();
        loose._M_rank = 2u;
        loose._M_in[1] = 4u;
        loose._M_kernel[1] = 2u;
        loose._M_stride[1] = 2u;
        loose._M_out[1] = 2u;
        loose._M_pad[2] = 1;
        loose._M_pad[3] = 1;
        REQUIRE (loose.validate () == Status::OK);
        PoolSpec spec = loose;
        spec._M_rank = 1u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_in[1] = 3u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_kernel[1] = 2u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_stride[1] = 2u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_out[1] = 2u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_pad[2] = 1;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_pad[3] = -1;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("a zero kernel or stride")
    {
        PoolSpec spec = base ();
        spec._M_kernel[0] = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec = base ();
        spec._M_stride[0] = 0u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("an output length below one")
    {
        PoolSpec spec = base ();
        spec._M_in[0] = 2u;
        spec._M_kernel[0] = 5u;
        spec._M_stride[0] = 1u;
        CHECK (spec.out_length (0u) == 0u);
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("a caller output wider than the input")
    {
        PoolSpec spec = base ();
        spec._M_out[0] = 9u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec._M_out[0] = 8u;
        CHECK (spec.validate () == Status::OK);
        spec._M_out[0] = 1u;
        CHECK (spec.validate () == Status::OK);
    }
    SECTION ("padding on an adaptive layer")
    {
        PoolSpec spec = base ();
        spec._M_kind = PoolKind::ADAPTIVE_AVERAGE;
        spec._M_out[0] = 4u;
        REQUIRE (spec.validate () == Status::OK);
        spec._M_pad[0] = 1;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec._M_pad[0] = 0;
        spec._M_pad[1] = -1;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec._M_pad[1] = 0;
        spec._M_kernel[0] = 9u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("a global layer pinned to one output per axis")
    {
        PoolSpec spec = base ();
        spec._M_kind = PoolKind::GLOBAL_MAX;
        REQUIRE (spec.validate () == Status::OK);
        spec._M_out[0] = 2u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
        spec._M_out[0] = 0u;
        spec._M_in[1] = 6u;
        spec._M_rank = 2u;
        spec._M_out[1] = 3u;
        CHECK (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("element counts beyond a training tensor")
    {
        PoolSpec spec = base ();
        spec._M_rank = 2u;
        spec._M_in[0] = 255u;
        spec._M_in[1] = 255u;
        spec._M_kernel[0] = 3u;
        spec._M_kernel[1] = 3u;
        spec._M_stride[0] = 1u;
        spec._M_stride[1] = 1u;
        spec._M_batch = 2u;
        spec._M_channels = 2u;
        REQUIRE (spec.input_elements () > _S_max_elements);
        CHECK (spec.validate () == Status::OUT_OF_RANGE);
        /* one plane per tensor: the input_elements guard fires first, so the
           in_plane guard behind it only ever agrees with it */
        spec._M_batch = 1u;
        spec._M_channels = 1u;
        spec._M_in[0] = 300u;
        spec._M_in[1] = 300u;
        spec._M_kernel[0] = 3u;
        spec._M_kernel[1] = 3u;
        REQUIRE (spec.in_plane () > _S_max_elements);
        REQUIRE (spec.input_elements () == spec.in_plane ());
        CHECK (spec.validate () == Status::OUT_OF_RANGE);
        CHECK (spec.out_length (0u) == 298u);
    }
    SECTION ("a window wider than the frame bound")
    {
        PoolSpec spec = base ();
        spec._M_in[0] = 64u;
        spec._M_kernel[0] = 64u;
        spec._M_stride[0] = 64u;
        REQUIRE (spec.max_window () == _S_pool_window_max);
        CHECK (spec.validate () == Status::OK);
        spec._M_kernel[0] = 65u;
        spec._M_in[0] = 70u;
        CHECK (spec.validate () == Status::OUT_OF_RANGE);
        spec._M_rank = 2u;
        spec._M_in[1] = 9u;
        spec._M_kernel[1] = 9u;
        spec._M_stride[1] = 9u;
        spec._M_kernel[0] = 8u;
        spec._M_in[0] = 8u;
        REQUIRE (spec.max_window () == 72u);
        CHECK (spec.validate () == Status::OUT_OF_RANGE);
        /* the same plane as a global layer has no tap bound */
        spec._M_kind = PoolKind::GLOBAL_MAX;
        CHECK (spec.validate () == Status::OK);
    }
}

TEST_CASE ("A window of exactly the frame bound is coverable, 65 taps is not")
{
    auto full64 = [] {
        PoolSpec spec;
        spec._M_kind = PoolKind::MAX;
        spec._M_rank = 1u;
        spec._M_in[0] = 64u;
        spec._M_kernel[0] = 64u;
        spec._M_stride[0] = 64u;
        return spec;
    };

    SECTION ("the bound is 64 and the accounting stays inside it")
    {
        CHECK (_S_pool_window_max == 64u);
        PoolSpec spec = full64 ();
        CHECK (spec.max_window () == _S_pool_window_max);
        CHECK (spec.out_length (0u) == 1u);
        REQUIRE (spec.validate () == Status::OK);
        spec._M_kernel[0] = 63u;
        CHECK (spec.max_window () == 63u);
        CHECK (spec.validate () == Status::OK);
        spec._M_kernel[0] = 65u;
        spec._M_in[0] = 66u;
        CHECK (spec.max_window () == 65u);
        CHECK (spec.validate () == Status::OUT_OF_RANGE);
    }
    SECTION ("a one-axis 64-tap window forward and backward both replay")
    {
        PoolSpec spec = full64 ();
        check_forward_matches (spec, 3u);
        check_backward_matches (spec, 3u, 800u);
        /* the single output reads the whole plane */
        CHECK (spec.in_plane () == 64u);
        CHECK (spec.out_plane () == 1u);
        const std::vector<int8_t> codes = fill_codes (64u, 3u);
        const RefForward fwd = ref_forward (spec, codes);
        int32_t best = -128;
        uint32_t at = 0u;
        for (uint32_t i = 0u; i < 64u; ++i)
            if (codes[i] > best)
                {
                    best = codes[i];
                    at = i;
                }
        CHECK (fwd._M_state[0] == at);
        CHECK (fwd._M_code[0] == static_cast<int8_t> (best));
    }
    SECTION ("a 64-tap 2-D window (8 by 8) fits and 8 by 9 does not")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::AVERAGE;
        spec._M_rank = 2u;
        spec._M_in[0] = 8u;
        spec._M_in[1] = 8u;
        spec._M_kernel[0] = 8u;
        spec._M_kernel[1] = 8u;
        spec._M_stride[0] = 8u;
        spec._M_stride[1] = 8u;
        REQUIRE (spec.max_window () == 64u);
        REQUIRE (spec.validate () == Status::OK);
        check_forward_matches (spec, 2u);
        check_backward_matches (spec, 2u, 800u);
        spec._M_in[1] = 9u;
        spec._M_in[0] = 9u;
        spec._M_kernel[1] = 9u;
        spec._M_kernel[0] = 8u;
        REQUIRE (spec.max_window () == 72u);
        CHECK (spec.validate () == Status::OUT_OF_RANGE);
    }
    SECTION ("a stride-1 64-tap sweep keeps every tap addressable")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::MAX;
        spec._M_rank = 1u;
        spec._M_in[0] = 70u;
        spec._M_kernel[0] = 64u;
        spec._M_stride[0] = 1u;
        REQUIRE (spec.out_length (0u) == 7u);
        REQUIRE (spec.validate () == Status::OK);
        check_forward_matches (spec, 6u);
        check_backward_matches (spec, 6u, 500u);
    }
}

TEST_CASE ("A window that lands entirely in the padding is refused by the forward")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::MAX;
    spec._M_rank = 1u;
    spec._M_in[0] = 4u;
    spec._M_kernel[0] = 1u;
    spec._M_stride[0] = 1u;
    spec._M_pad[0] = 5;
    /* four placements, every one of them left of the real border */
    REQUIRE (spec.out_length (0u) == 4u);
    REQUIRE (spec.validate () == Status::OK);
    for (uint32_t flat = 0u; flat < 4u; ++flat)
    {
        const PoolWindow window = pool_window (spec, flat);
        CHECK (window._M_start[0] + static_cast<int32_t> (window._M_length[0])
               <= 0);
    }

    Views views;
    for (uint32_t i = 0u; i < 4u; ++i)
        views._M_in_code[i] = static_cast<int8_t> (i + 1);
    Trainable input
        = Trainable::make (views._M_in_code, nullptr, shape_of (spec, true), 0u);
    Trainable output = Trainable::make (views._M_out_code, views._M_out_grad,
                                        shape_of (spec, false), 0u);
    uint16_t state[4] = { 0u, 0u, 0u, 0u };
    SECTION ("the max forward reports the layout it cannot reduce")
    {
        CHECK (pool_forward (spec, input, output, state) == Status::BAD_LAYOUT);
    }
    SECTION ("the average forward refuses the undefined mean")
    {
        spec._M_kind = PoolKind::AVERAGE;
        CHECK (pool_forward (spec, input, output, state) == Status::BAD_LAYOUT);
    }
    SECTION ("a partially covered window is fine")
    {
        spec._M_kind = PoolKind::AVERAGE;
        spec._M_kernel[0] = 3u;
        spec._M_pad[0] = 2;
        REQUIRE (spec.out_length (0u) == 4u);
        REQUIRE (spec.validate () == Status::OK);
        state[0] = 0xFFFFu;
        CHECK (status_ok (pool_forward (spec, input, output, state)));
        /* every window touches the data: only the two left taps of the first one
           are padding, so the nominal divisor of 3 still shrinks its mean */
        CHECK (output._M_code[0] == 0);
        CHECK (output._M_code[1] == 1);
        CHECK (output._M_code[2] == 2);
        CHECK (output._M_code[3] == 3);
        CHECK (state[0] == 3u);
        CHECK (state[3] == 3u);
        const RefForward fwd = ref_forward (spec, { views._M_in_code,
                                                    views._M_in_code + 4 });
        for (uint32_t i = 0u; i < 4u; ++i)
            CHECK (output._M_code[i] == fwd._M_code[i]);
    }
}

TEST_CASE ("Pooled codes stay inside the minus-127 to 127 contract")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::MAX;
    spec._M_rank = 1u;
    spec._M_in[0] = 6u;
    spec._M_kernel[0] = 3u;
    spec._M_stride[0] = 3u;
    REQUIRE (spec.out_length (0u) == 2u);

    Views views;
    for (uint32_t i = 0u; i < 6u; ++i)
        views._M_in_code[i] = -128; /* an out-of-contract code a caller could feed */
    Trainable input
        = Trainable::make (views._M_in_code, nullptr, shape_of (spec, true), 0u);
    Trainable output = Trainable::make (views._M_out_code, views._M_out_grad,
                                        shape_of (spec, false), 0u);
    uint16_t state[2] = { 0u, 0u };

    SECTION ("the forward never emits the minus-128 code")
    {
        REQUIRE (status_ok (pool_forward (spec, input, output, state)));
        for (uint32_t i = 0u; i < 2u; ++i)
            {
                /* the code the layer reports is the clamped one, so a caller that
                   fed the illegal -128 sees -127 and never -128 */
                CHECK (output._M_code[i] == -127);
                CHECK (output._M_code[i] != -128);
                /* the argmax probe starts at -128, so nothing is ever strictly
                   greater than an all -128 plane and the recorded coordinate
                   collapses to 0 -- out of contract, but it must not crash */
                CHECK (state[i] == 0u);
            }
    }
    SECTION ("a saturating input still rounds inside the grid")
    {
        for (uint32_t i = 0u; i < 6u; ++i)
            views._M_in_code[i] = 127;
        spec._M_kind = PoolKind::AVERAGE;
        REQUIRE (status_ok (pool_forward (spec, input, output, state)));
        CHECK (output._M_code[0] == 127);
        CHECK (output._M_code[1] == 127);
        spec._M_kind = PoolKind::L2;
        REQUIRE (status_ok (pool_forward (spec, input, output, state)));
        CHECK (output._M_code[0] == 127);
        CHECK (ref_isqrt (static_cast<uint32_t> (
            ref_div_round (3 * 127 * 127, 3))) == 127);
        for (uint32_t i = 0u; i < 2u; ++i)
            CHECK (state[i] == 3u);
    }
}

TEST_CASE ("Average backward divides by the area the forward recorded")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::AVERAGE;
    spec._M_rank = 1u;
    spec._M_in[0] = 3u;
    spec._M_kernel[0] = 3u;
    spec._M_stride[0] = 1u;
    spec._M_pad[0] = 1;
    REQUIRE (spec.out_length (0u) == 2u);

    Rig rig;
    Views views;
    views._M_in_code[0] = 4;
    views._M_in_code[1] = 8;
    views._M_in_code[2] = 12;
    for (uint32_t i = 0u; i < 3u; ++i)
        views._M_in_grad[i] = 0;
    Trainable input
        = Trainable::make (views._M_in_code, views._M_in_grad,
                           shape_of (spec, true), Flag::RequiresGrad);
    Trainable output = Trainable::make (views._M_out_code, views._M_out_grad,
                                        shape_of (spec, false), 0u);
    REQUIRE (status_ok (rig._M_tape.begin ()));
    REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));
    GradContext context = context_of (rig);
    const uint16_t *const state = context.saved_as<uint16_t> (_S_pool_window);
    REQUIRE (state != nullptr);

    SECTION ("padding counts, so the border window splits the lane by three")
    {
        CHECK (state[0] == 3u);
        CHECK (state[1] == 3u);
        CHECK (output._M_code[0] == ref_div_round (12, 3));
        output._M_grad[0] = 3000;
        output._M_grad[1] = 0;
        REQUIRE (status_ok (pool_backward (output, output, context)));
        /* window 0 = pad, 4, 8 -> two real taps, divisor 3 */
        CHECK (input._M_grad[0] == 1000);
        CHECK (input._M_grad[1] == 1000);
        CHECK (input._M_grad[2] == 0);
    }
    SECTION ("excluding padding keeps the border mean and lane honest")
    {
        spec._M_count_include_pad = 0u;
        Rig second;
        Views more;
        more._M_in_code[0] = 4;
        more._M_in_code[1] = 8;
        more._M_in_code[2] = 12;
        for (uint32_t i = 0u; i < 3u; ++i)
            more._M_in_grad[i] = 0;
        Trainable in2 = Trainable::make (more._M_in_code, more._M_in_grad,
                                         shape_of (spec, true),
                                         Flag::RequiresGrad);
        Trainable out2
            = Trainable::make (more._M_out_code, more._M_out_grad,
                               shape_of (spec, false), 0u);
        REQUIRE (status_ok (second._M_tape.begin ()));
        REQUIRE (status_ok (pool_run (second._M_tape, spec, in2, out2)));
        GradContext ctx2 = context_of (second);
        const uint16_t *const state2
            = ctx2.saved_as<uint16_t> (_S_pool_window);
        REQUIRE (state2 != nullptr);
        CHECK (state2[0] == 2u);
        CHECK (state2[1] == 3u);
        CHECK (out2._M_code[0] == ref_div_round (12, 2));
        CHECK (out2._M_code[1] == ref_div_round (24, 3));
        out2._M_grad[0] = 3000;
        out2._M_grad[1] = 0;
        REQUIRE (status_ok (pool_backward (out2, out2, ctx2)));
        CHECK (in2._M_grad[0] == 1500);
        CHECK (in2._M_grad[1] == 1500);
        CHECK (in2._M_grad[2] == 0);
        CHECK (ctx2._M_saturation_events == 0u);
    }
}

TEST_CASE ("PoolSpec accounting covers what a real backward carves")
{
    auto pad4 = [](const uint32_t value) { return (value + 3u) & ~3u; };
    struct Geom
    {
        PoolKind _M_kind;
        uint8_t _M_rank;
        uint16_t _M_in0;
        uint16_t _M_in1;
        uint16_t _M_k0;
        uint16_t _M_k1;
        uint16_t _M_s0;
        uint16_t _M_s1;
        uint16_t _M_out0;
    };
    const std::vector<Geom> geoms
        = { { PoolKind::MAX, 1u, 16u, 1u, 2u, 1u, 2u, 1u, 0u },
            { PoolKind::MAX, 2u, 5u, 4u, 3u, 2u, 1u, 1u, 0u },
            { PoolKind::MAX, 2u, 4u, 4u, 2u, 2u, 2u, 2u, 0u },
            { PoolKind::AVERAGE, 1u, 12u, 1u, 3u, 1u, 1u, 1u, 0u },
            { PoolKind::L2, 2u, 5u, 4u, 3u, 2u, 1u, 1u, 0u },
            { PoolKind::L2, 1u, 9u, 1u, 3u, 1u, 3u, 1u, 0u },
            { PoolKind::ADAPTIVE_AVERAGE, 1u, 12u, 1u, 1u, 1u, 1u, 1u, 4u },
            { PoolKind::GLOBAL_MAX, 2u, 5u, 4u, 1u, 1u, 1u, 1u, 0u },
            { PoolKind::GLOBAL_AVERAGE, 1u, 20u, 1u, 1u, 1u, 1u, 1u, 0u },
            { PoolKind::GLOBAL_L2, 2u, 4u, 3u, 1u, 1u, 1u, 1u, 0u } };

    for (const Geom &geom : geoms)
        {
            PoolSpec spec;
            spec._M_kind = geom._M_kind;
            spec._M_rank = geom._M_rank;
            spec._M_batch = 2u;
            spec._M_channels = 2u;
            spec._M_in[0] = geom._M_in0;
            spec._M_in[1] = geom._M_rank == 2u ? geom._M_in1 : 1u;
            spec._M_kernel[0] = geom._M_k0;
            spec._M_kernel[1] = geom._M_rank == 2u ? geom._M_k1 : 1u;
            spec._M_stride[0] = geom._M_s0;
            spec._M_stride[1] = 1u;
            spec._M_out[0] = geom._M_out0;
            spec._M_op_id = 42u;
            INFO ("kind " << static_cast<int> (geom._M_kind) << " rank "
                          << int (geom._M_rank));
            REQUIRE (spec.validate () == Status::OK);

            /* the invariant the sentinel tail of the max scatter relies on */
            CHECK (spec.out_plane () <= spec.in_plane ());

            const uint32_t elements = spec.output_elements ();
            const uint32_t in_plane = spec.in_plane ();
            const uint32_t out_plane = spec.out_plane ();
            const uint32_t want_spec = pad4 (static_cast<uint32_t> (sizeof (PoolSpec)));
            const uint32_t want_window = pad4 (2u * elements);
            const uint32_t want_scratch
                = pool_is_max (spec._M_kind)
                      ? 2u * pad4 (2u * in_plane) + pad4 (2u * out_plane)
                      : 0u;
            const uint32_t want_input
                = pool_needs_input (spec._M_kind)
                      ? pad4 (spec.input_elements ())
                      : 0u;
            const uint32_t want_total
                = want_spec + want_window + want_scratch + want_input;

            CHECK (spec.scratch_bytes () == want_scratch);
            CHECK (spec.bytes () == want_total);
            CHECK (want_window >= 2u * elements);
            const bool input_covers
                = want_input == 0u || want_input >= spec.input_elements ();
            CHECK (input_covers);
            const bool scratch_covers
                = want_scratch == 0u || 2u * in_plane <= want_scratch;
            CHECK (scratch_covers);
            CHECK (want_spec >= sizeof (PoolSpec));

            Rig rig;
            std::fill (std::begin (rig._M_arena), std::end (rig._M_arena),
                       uint8_t (0xAAu));
            Views views;
            const std::vector<int8_t> codes = fill_codes (spec.input_elements (), 2u);
            for (uint32_t i = 0u; i < spec.input_elements (); ++i)
                {
                    views._M_in_code[i] = codes[i];
                    views._M_in_grad[i] = 0;
                }
            Trainable input = Trainable::make (views._M_in_code,
                                               views._M_in_grad,
                                               shape_of (spec, true),
                                               Flag::RequiresGrad);
            Trainable output = Trainable::make (views._M_out_code,
                                                views._M_out_grad,
                                                shape_of (spec, false), 0u);
            const uint32_t used_before = rig._M_pool.stats()._M_used;
            REQUIRE (status_ok (rig._M_tape.begin ()));
            REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));
            CHECK (rig._M_pool.stats()._M_used - used_before == spec.bytes ());

            Node *const node = rig._M_tape.node (0u);
            REQUIRE (node != nullptr);
            CHECK (node->_M_saved[_S_pool_spec]._M_bytes == want_spec);
            CHECK (node->_M_saved[_S_pool_window]._M_bytes == want_window);
            CHECK (node->_M_saved[_S_pool_scratch]._M_bytes == want_scratch);
            CHECK (node->_M_saved[_S_pool_input]._M_bytes == want_input);
            CHECK (node->_M_saved[_S_pool_window]._M_count == elements);
            CHECK (node->_M_saved[_S_pool_input]._M_count
                   == (want_input != 0u ? spec.input_elements () : 0u));
            uint32_t recorded = 0u;
            for (uint32_t i = 0u; i < _S_max_operands; ++i)
                {
                    recorded += node->_M_saved[i]._M_bytes;
                    /* the saved_as<T> precondition: a slot never promises more
                       than it holds */
                    CHECK (node->_M_saved[i]._M_bytes
                           >= node->_M_saved[i]._M_count);
                }
            CHECK (recorded <= spec.bytes ());

            const std::vector<int32_t> grads
                = fill_grads (elements, 5u, 900u);
            for (uint32_t i = 0u; i < elements; ++i)
                output._M_grad[i] = static_cast<int16_t> (grads[i]);
            GradContext context = context_of (rig);
            REQUIRE (status_ok (pool_backward (output, output, context)));

            /* nothing moved outside the single carve: the spec numbers cover the
               lane plane, the index plane and the staged output lanes */
            const uint32_t slab_end = used_before + spec.bytes ();
            CHECK (slab_end <= sizeof (Rig::_M_arena));
            uint32_t untouched = 0u;
            for (uint32_t byte = slab_end;
                 byte < slab_end + 256u && byte < sizeof (Rig::_M_arena); ++byte)
                {
                    if (rig._M_arena[byte] == 0xAAu)
                        ++untouched;
                }
            CHECK (untouched > 0u);
            for (uint32_t byte = slab_end;
                 byte < slab_end + 256u && byte < sizeof (Rig::_M_arena); ++byte)
                CHECK (rig._M_arena[byte] == 0xAAu);
            if (pool_is_max (spec._M_kind))
                {
                    const uint8_t *const scratch
                        = rig._M_pool.at (node->_M_saved[_S_pool_scratch]._M_offset);
                    REQUIRE (scratch != nullptr);
                    CHECK (simd_aligned (scratch));
                    CHECK (simd_aligned (scratch + pad4 (2u * in_plane)));
                    CHECK (simd_aligned (scratch + 2u * pad4 (2u * in_plane)));
                    CHECK (2u * in_plane <= want_scratch);
                    CHECK (2u * out_plane <= want_scratch);
                }
        }
}

TEST_CASE ("pool_run wires the tape node and saved_as round-trips the spec")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::MAX;
    spec._M_rank = 2u;
    spec._M_batch = 2u;
    spec._M_channels = 2u;
    spec._M_in[0] = 5u;
    spec._M_in[1] = 4u;
    spec._M_kernel[0] = 3u;
    spec._M_kernel[1] = 2u;
    spec._M_stride[0] = 2u;
    spec._M_stride[1] = 1u;
    spec._M_op_id = 777u;
    REQUIRE (spec.validate () == Status::OK);

    Rig rig;
    Views views;
    const std::vector<int8_t> codes = fill_codes (spec.input_elements (), 1u);
    for (uint32_t i = 0u; i < spec.input_elements (); ++i)
        {
            views._M_in_code[i] = codes[i];
            views._M_in_grad[i] = 0;
        }
    Trainable input = Trainable::make (views._M_in_code, views._M_in_grad,
                                       shape_of (spec, true),
                                       Flag::RequiresGrad);
    Trainable output = Trainable::make (views._M_out_code, views._M_out_grad,
                                        shape_of (spec, false),
                                        Flag::RequiresGrad);
    REQUIRE (rig._M_tape.node_count () == 0u);
    REQUIRE (status_ok (rig._M_tape.begin ()));
    REQUIRE (rig._M_tape.is_open ());

    SECTION ("the recorded node carries one operand and four saved slots")
    {
        REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));
        CHECK (rig._M_tape.node_count () == 1u);
        Node *const node = rig._M_tape.node (0u);
        REQUIRE (node != nullptr);
        CHECK (node->_M_operand_count == 1u);
        CHECK (node->_M_saved_count == _S_max_operands);
        CHECK (node->_M_saved_count == 4u);
        CHECK (node->_M_op_id == 777u);
        CHECK (node->_M_output._M_count == spec.output_elements ());
        CHECK (node->_M_operands[0]._M_count == spec.input_elements ());
        /* the second slot is deliberately empty so the node never looks like a
           two-tensor op */
        CHECK (node->_M_operands[1]._M_code == nullptr);
        CHECK (node->_M_operands[2]._M_code == nullptr);
        CHECK (node->_M_operands[3]._M_code == nullptr);
    }
    SECTION ("the spec survives the arena round trip byte for byte")
    {
        REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));
        GradContext context = context_of (rig);
        const PoolSpec *const saved = context.saved_as<PoolSpec> (_S_pool_spec);
        REQUIRE (saved != nullptr);
        CHECK (saved->_M_batch == spec._M_batch);
        CHECK (saved->_M_channels == spec._M_channels);
        CHECK (saved->_M_in[0] == spec._M_in[0]);
        CHECK (saved->_M_in[1] == spec._M_in[1]);
        CHECK (saved->_M_out[0] == spec._M_out[0]);
        CHECK (saved->_M_out[1] == spec._M_out[1]);
        CHECK (saved->_M_kernel[0] == spec._M_kernel[0]);
        CHECK (saved->_M_kernel[1] == spec._M_kernel[1]);
        CHECK (saved->_M_stride[0] == spec._M_stride[0]);
        CHECK (saved->_M_stride[1] == spec._M_stride[1]);
        CHECK (saved->_M_pad[0] == spec._M_pad[0]);
        CHECK (saved->_M_pad[3] == spec._M_pad[3]);
        CHECK (saved->_M_kind == PoolKind::MAX);
        CHECK (saved->_M_layout == PoolLayout::CHANNEL_MAJOR);
        CHECK (saved->_M_rank == 2u);
        CHECK (saved->_M_count_include_pad == 1u);
        CHECK (saved->_M_op_id == 777u);
        CHECK (context.saved_as<PoolSpec> (_S_max_operands) == nullptr);
        CHECK (context.saved_as<uint16_t> (4u) == nullptr);
    }
    SECTION ("the forward ran through the recorded node")
    {
        REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));
        const RefForward fwd = ref_forward (spec, codes);
        for (uint32_t i = 0u; i < spec.output_elements (); ++i)
            CHECK (output._M_code[i] == fwd._M_code[i]);
        CHECK (input._M_grad[0] == 0);
    }
    SECTION ("a shape mismatch is refused before any arena traffic")
    {
        Trainable wrong = input;
        wrong._M_count = spec.input_elements () - 1u;
        const uint32_t used_before = rig._M_pool.stats()._M_used;
        CHECK (pool_run (rig._M_tape, spec, wrong, output)
               == Status::BAD_LAYOUT);
        CHECK (rig._M_pool.stats()._M_used == used_before);
        CHECK (rig._M_tape.node_count () == 0u);
        wrong = input;
        wrong._M_count = spec.input_elements ();
        Trainable bad_out = output;
        bad_out._M_count = spec.output_elements () + 2u;
        CHECK (pool_run (rig._M_tape, spec, wrong, bad_out)
               == Status::BAD_LAYOUT);
        PoolSpec broken = spec;
        broken._M_kernel[0] = 0u;
        CHECK (pool_run (rig._M_tape, broken, input, output)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("pool_run refuses a closed tape and a starved arena")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::AVERAGE;
    spec._M_rank = 1u;
    spec._M_batch = 1u;
    spec._M_channels = 1u;
    spec._M_in[0] = 8u;
    spec._M_kernel[0] = 2u;
    spec._M_stride[0] = 2u;
    REQUIRE (spec.validate () == Status::OK);

    SECTION ("a tape that never began answers INTERNAL")
    {
        Rig rig;
        Views views;
        for (uint32_t i = 0u; i < 8u; ++i)
            views._M_in_code[i] = static_cast<int8_t> (i);
        Trainable input = Trainable::make (views._M_in_code, nullptr,
                                           shape_of (spec, true), 0u);
        Trainable output = Trainable::make (views._M_out_code,
                                            views._M_out_grad,
                                            shape_of (spec, false), 0u);
        REQUIRE_FALSE (rig._M_tape.is_open ());
        CHECK (pool_run (rig._M_tape, spec, input, output) == Status::INTERNAL);
        CHECK (rig._M_tape.node_count () == 0u);
        /* reset closes the pass, so recording stops working again */
        REQUIRE (status_ok (rig._M_tape.begin ()));
        REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, output)));
        CHECK (rig._M_tape.node_count () == 1u);
        rig._M_tape.reset ();
        CHECK_FALSE (rig._M_tape.is_open ());
        CHECK (rig._M_tape.node_count () == 0u);
        CHECK (rig._M_pool.remaining () == sizeof (Rig::_M_arena));
        CHECK (pool_run (rig._M_tape, spec, input, output) == Status::INTERNAL);
    }
    SECTION ("an arena too small for spec.bytes answers ARENA_OVERFLOW")
    {
        alignas (16) uint8_t raw[128] = {};
        Pool pool (raw, sizeof (raw));
        Node nodes[4];
        Tape tape (nodes, 4u, pool);
        REQUIRE (status_ok (tape.begin ()));
        /* a max layer needs its window state, its spec and the scatter scratch */
        PoolSpec greedy;
        greedy._M_kind = PoolKind::MAX;
        greedy._M_rank = 1u;
        greedy._M_batch = 1u;
        greedy._M_channels = 1u;
        greedy._M_in[0] = 64u;
        greedy._M_kernel[0] = 2u;
        greedy._M_stride[0] = 2u;
        REQUIRE (greedy.validate () == Status::OK);
        CHECK (greedy.scratch_bytes () == 2u * 128u + 64u);
        CHECK (greedy.bytes () == 64u + 320u + 28u);
        Views views;
        for (uint32_t i = 0u; i < 64u; ++i)
            views._M_in_code[i] = static_cast<int8_t> (i % 13u);
        Trainable input = Trainable::make (views._M_in_code, nullptr,
                                           Shape::tensor3 (1u, 1u, 64u), 0u);
        Trainable output = Trainable::make (views._M_out_code, nullptr,
                                            Shape::tensor3 (1u, 1u, 32u), 0u);
        REQUIRE (greedy.bytes () > sizeof (raw));
        CHECK (pool_run (tape, greedy, input, output)
               == Status::ARENA_OVERFLOW);
        CHECK (tape.node_count () == 0u);
        CHECK (pool.stats()._M_overflows == 1u);
        CHECK (pool.remaining () == sizeof (raw));
        /* the same geometry runs perfectly well when the caller owns the state,
           so the refusal is arena traffic and nothing else */
        uint16_t local_state[32] = {};
        CHECK (status_ok (pool_forward (greedy, input, output, local_state)));
    }
    SECTION ("a second node that does not fit is refused without corrupting the first")
    {
        alignas (16) uint8_t raw[256] = {};
        Pool pool (raw, sizeof (raw));
        Node nodes[4];
        Tape tape (nodes, 4u, pool);
        REQUIRE (status_ok (tape.begin ()));
        Views views;
        for (uint32_t i = 0u; i < 8u; ++i)
            views._M_in_code[i] = static_cast<int8_t> (i);
        Trainable input = Trainable::make (views._M_in_code, nullptr,
                                           shape_of (spec, true), 0u);
        Trainable output = Trainable::make (views._M_out_code,
                                            views._M_out_grad,
                                            shape_of (spec, false), 0u);
        REQUIRE (spec.bytes () < sizeof (raw));
        REQUIRE (status_ok (pool_run (tape, spec, input, output)));
        const uint32_t used = pool.stats()._M_used;
        CHECK (used == spec.bytes ());
        PoolSpec greedy;
        greedy._M_kind = PoolKind::MAX;
        greedy._M_rank = 1u;
        greedy._M_batch = 1u;
        greedy._M_channels = 1u;
        greedy._M_in[0] = 64u;
        greedy._M_kernel[0] = 2u;
        greedy._M_stride[0] = 2u;
        REQUIRE (greedy.validate () == Status::OK);
        Trainable big_in = Trainable::make (views._M_in_code, nullptr,
                                           Shape::tensor3 (1u, 1u, 64u), 0u);
        Trainable big_out = Trainable::make (views._M_out_code, nullptr,
                                             Shape::tensor3 (1u, 1u, 32u), 0u);
        CHECK (greedy.bytes () > pool.remaining ());
        CHECK (pool_run (tape, greedy, big_in, big_out)
               == Status::ARENA_OVERFLOW);
        CHECK (pool.stats()._M_used == used);
        CHECK (tape.node_count () == 1u);
        CHECK (pool.stats()._M_slabs == 1u);
        CHECK (tape.node (0u)->_M_saved[_S_pool_window]._M_bytes
               == align_up (2u * spec.output_elements (), 4u));
    }
}

TEST_CASE ("pool_forward binds operands to the spec it was handed")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::MAX;
    spec._M_rank = 2u;
    spec._M_batch = 1u;
    spec._M_channels = 2u;
    spec._M_in[0] = 4u;
    spec._M_in[1] = 3u;
    spec._M_kernel[0] = 2u;
    spec._M_kernel[1] = 2u;
    spec._M_stride[0] = 2u;
    spec._M_stride[1] = 1u;
    REQUIRE (spec.validate () == Status::OK);

    Views views;
    for (uint32_t i = 0u; i < spec.input_elements (); ++i)
        views._M_in_code[i] = static_cast<int8_t> ((i % 9u) - 4);
    uint16_t state[8] = {};

    SECTION ("a null code stream is refused")
    {
        Trainable null_in = Trainable::make (nullptr, nullptr,
                                             shape_of (spec, true), 0u);
        Trainable out = Trainable::make (views._M_out_code, nullptr,
                                         shape_of (spec, false), 0u);
        CHECK (pool_forward (spec, null_in, out, state) == Status::BAD_LAYOUT);
        Trainable in = Trainable::make (views._M_in_code, nullptr,
                                        shape_of (spec, true), 0u);
        Trainable null_out = Trainable::make (nullptr, nullptr,
                                              shape_of (spec, false), 0u);
        CHECK (pool_forward (spec, in, null_out, state) == Status::BAD_LAYOUT);
    }
    SECTION ("a count that disagrees with the spec is refused")
    {
        Trainable in = Trainable::make (views._M_in_code, nullptr,
                                        shape_of (spec, true), 0u);
        in._M_count = spec.input_elements () + 1u;
        Trainable out = Trainable::make (views._M_out_code, nullptr,
                                         shape_of (spec, false), 0u);
        CHECK (pool_forward (spec, in, out, state) == Status::BAD_LAYOUT);
        in._M_count = spec.input_elements ();
        out._M_count = 1u;
        CHECK (pool_forward (spec, in, out, state) == Status::BAD_LAYOUT);
    }
    SECTION ("a rank that disagrees with the spec is refused")
    {
        Trainable in = Trainable::make (views._M_in_code, nullptr,
                                        Shape::tensor3 (1u, 2u, 12u), 0u);
        in._M_count = spec.input_elements ();
        Trainable out = Trainable::make (views._M_out_code, nullptr,
                                         shape_of (spec, false), 0u);
        CHECK (pool_forward (spec, in, out, state) == Status::BAD_LAYOUT);
    }
    SECTION ("a rank-agnostic view is accepted")
    {
        Trainable in;
        in._M_code = views._M_in_code;
        in._M_count = spec.input_elements ();
        Trainable out;
        out._M_code = views._M_out_code;
        out._M_count = spec.output_elements ();
        CHECK (status_ok (pool_forward (spec, in, out, state)));
        const RefForward fwd = ref_forward (
            spec, { views._M_in_code,
                    views._M_in_code + spec.input_elements () });
        for (uint32_t i = 0u; i < spec.output_elements (); ++i)
            CHECK (out._M_code[i] == fwd._M_code[i]);
    }
    SECTION ("an invalid spec answers with its own status")
    {
        PoolSpec bad = spec;
        bad._M_kind = static_cast<PoolKind> (9u);
        Trainable in = Trainable::make (views._M_in_code, nullptr,
                                        shape_of (spec, true), 0u);
        Trainable out = Trainable::make (views._M_out_code, nullptr,
                                         shape_of (spec, false), 0u);
        CHECK (pool_forward (bad, in, out, state) == Status::BAD_LAYOUT);
        bad._M_kind = PoolKind::MAX;
        bad._M_in[0] = 65u;
        bad._M_kernel[0] = 65u;
        bad._M_stride[0] = 1u;
        bad._M_in[1] = 1u;
        bad._M_kernel[1] = 1u;
        bad._M_rank = 1u;
        REQUIRE (bad.validate () == Status::OUT_OF_RANGE);
        CHECK (pool_forward (bad, in, out, state) == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("A pooling node replays through Tape::backward end to end")
{
    PoolSpec spec;
    spec._M_kind = PoolKind::GLOBAL_AVERAGE;
    spec._M_rank = 1u;
    spec._M_batch = 1u;
    spec._M_channels = 1u;
    spec._M_in[0] = 5u;
    REQUIRE (spec.validate () == Status::OK);

    Rig rig;
    Views views;
    const int8_t hand[5] = { 10, -20, 30, -40, 50 };
    for (uint32_t i = 0u; i < 5u; ++i)
        {
            views._M_in_code[i] = hand[i];
            views._M_in_grad[i] = 0;
        }
    Trainable input = Trainable::make (views._M_in_code, views._M_in_grad,
                                       shape_of (spec, true),
                                       Flag::RequiresGrad);
    /* the pooled scalar is the loss itself, so the tape can seed it */
    Trainable loss = Trainable::make (views._M_out_code, views._M_out_grad,
                                      shape_of (spec, false), 0u);
    REQUIRE (loss._M_count == 1u);
    REQUIRE (status_ok (rig._M_tape.begin ()));
    REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, loss)));

    SECTION ("the forward is the mean of the plane")
    {
        CHECK (loss._M_code[0] == 6);
        CHECK (ref_div_round (10 - 20 + 30 - 40 + 50, 5) == 6);
    }
    SECTION ("the backward broadcasts the seeded lane")
    {
        REQUIRE (status_ok (rig._M_tape.backward (loss)));
        CHECK (loss._M_grad[0] == _S_q15_max);
        const int32_t lane = ref_div_round (_S_q15_max, 5);
        CHECK (lane == 6553);
        for (uint32_t i = 0u; i < 5u; ++i)
            CHECK (input._M_grad[i] == lane);
        CHECK (rig._M_tape.stats ()._M_saturation_events == 0u);
    }
    SECTION ("a second pass starts from a fresh tape")
    {
        REQUIRE (status_ok (pool_run (rig._M_tape, spec, input, loss)));
        CHECK (rig._M_tape.node_count () == 2u);
        CHECK (rig._M_tape.stats ()._M_peak_nodes == 2u);
        rig._M_tape.reset ();
        CHECK (rig._M_tape.node_count () == 0u);
        CHECK (rig._M_pool.remaining () == sizeof (Rig::_M_arena));
    }
}

TEST_CASE ("The pooling contract constants match the arena documentation")
{
    CHECK (_S_train_pool_bytes == 8192u);
    CHECK (_S_pool_alignment == 4u);
    CHECK (_S_max_operands == 4u);
    CHECK (_S_pool_window_max == 64u);
    CHECK (_S_pool_spec == 0u);
    CHECK (_S_pool_window == 1u);
    CHECK (_S_pool_scratch == 2u);
    CHECK (_S_pool_input == 3u);
    CHECK (sizeof (PoolSpec) == 28u);
    CHECK (align_up (1u, _S_pool_alignment) == 4u);
    CHECK (align_up (4u, _S_pool_alignment) == 4u);
    CHECK (align_up (5u, _S_pool_alignment) == 8u);

    SECTION ("a realistic layer fits the default training arena")
    {
        PoolSpec spec;
        spec._M_kind = PoolKind::MAX;
        spec._M_rank = 2u;
        spec._M_batch = 1u;
        spec._M_channels = 16u;
        spec._M_in[0] = 8u;
        spec._M_in[1] = 8u;
        spec._M_kernel[0] = 2u;
        spec._M_kernel[1] = 2u;
        spec._M_stride[0] = 2u;
        spec._M_stride[1] = 2u;
        REQUIRE (spec.validate () == Status::OK);
        CHECK (spec.output_elements () == 16u * 16u);
        CHECK (spec.in_plane () == 64u);
        /* two int16 planes of in_plane plus one staged int16 plane of out_plane,
           each rounded up to the pool alignment */
        CHECK (spec.scratch_bytes () == 2u * 128u + 32u);
        CHECK (spec.bytes () < _S_train_pool_bytes);
        CHECK (spec.bytes ()
               == align_up (2u * spec.output_elements (), 4u)
                  + spec.scratch_bytes ()
                  + align_up (static_cast<uint32_t> (sizeof (PoolSpec)), 4u));
        /* an L2 layer of the same size also carries its input codes */
        PoolSpec l2 = spec;
        l2._M_kind = PoolKind::L2;
        CHECK (l2.scratch_bytes () == 0u);
        CHECK (l2.bytes ()
               == spec.bytes () - spec.scratch_bytes ()
                  + align_up (l2.input_elements (), 4u));
        CHECK (l2.bytes () < _S_train_pool_bytes);
    }
    SECTION ("the pool the tape is built on keeps its word counters")
    {
        alignas (16) uint8_t raw[_S_train_pool_bytes] = {};
        Pool pool (raw, _S_train_pool_bytes);
        CHECK (pool.stats()._M_capacity == _S_train_pool_bytes);
        CHECK (pool.remaining () == _S_train_pool_bytes);
        Slab first;
        Slab second;
        REQUIRE (status_ok (pool.carve (28u, 1u, first)));
        REQUIRE (status_ok (pool.carve (7u, 2u, second)));
        CHECK (first._M_bytes == 28u);
        CHECK (second._M_offset == 28u);
        CHECK (second._M_bytes == 16u);
        CHECK (pool.stats()._M_slabs == 2u);
        CHECK (pool.stats()._M_peak == 44u);
        pool.reset ();
        CHECK (pool.stats()._M_slabs == 0u);
        CHECK (pool.stats()._M_used == 0u);
        CHECK (pool.stats()._M_peak == 44u);
    }
}



