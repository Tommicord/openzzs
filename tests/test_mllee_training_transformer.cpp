/* Tests for the two training-side transformer modules added after the
 * inference transformer work: mllee/train_norm.hpp (LayerNorm / RMSNorm, the
 * residual add) and mllee/train_attention.hpp (multi-head scaled dot-product
 * attention, causal mask, RoPE).
 *
 * Reference independence.  The forward values are recomputed with a plain
 * double-precision model of the documented rules (real = code / 127, integer
 * root of the Q30 variance, round-half-away narrowing), and the analytic
 * Jacobians with the textbook LayerNorm / softmax-backprop formulas.  The
 * attention forward is additionally pinned to the inference kernel: the test
 * packs Q, K and V into a model blob, runs Core, and demands byte-identical
 * codes from the training forward.  That is a stronger statement than a
 * tolerance check and is exactly the train/deploy property this header exists
 * to keep.
 *
 * The library stays heap-free; only the test's std::vector buffers live on the
 * host heap. */

#ifndef ZZS_MLLEE_HOST
#define ZZS_MLLEE_HOST
#endif

#include "mllee/mllee.hpp"
#include "mllee/train_attention.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_norm.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"
#include "mllee_test_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

using namespace zzs::mllee;
using namespace zzs::mllee::train;

namespace
{
/* ------------------------------------------------------------------ */
/* Independent integer / double references.                            */
/* ------------------------------------------------------------------ */

constexpr uint32_t
ref_align_up (const uint32_t value)
{
    return (value + _S_pool_alignment - 1u) & ~(_S_pool_alignment - 1u);
}

int32_t
ref_clamp_lane (const double value)
{
    const long rounded = std::lround (value);
    return rounded > 32767 ? 32767 : (rounded < -32768 ? -32768
                                                       : static_cast<int32_t> (rounded));
}

int32_t
ref_i8 (const double value)
{
    const long rounded = std::lround (value);
    return rounded < -127 ? -127 : (rounded > 127 ? 127
                                                  : static_cast<int32_t> (rounded));
}

/* One trainable tensor with its own code and gradient storage. */
struct Buf
{
    std::vector<int8_t> code;
    std::vector<int16_t> grad;
};

Buf
make_buf (std::vector<int8_t> codes, const bool with_grad = true)
{
    Buf buffer;
    buffer.code = std::move (codes);
    if (with_grad)
        buffer.grad.assign (buffer.code.size (), 0);
    return buffer;
}

Trainable
view (Buf &buffer, const Shape shape, const bool with_grad = true)
{
    return Trainable::make (buffer.code.data (),
                            with_grad ? buffer.grad.data () : nullptr, shape,
                            with_grad ? Flag::RequiresGrad : 0u);
}

/* Double model of one normalized window. */
struct RefNorm
{
    std::vector<double> xhat; /* one per element */
    std::vector<double> dev;  /* one per window */
    std::vector<int8_t> out;
};

RefNorm
ref_norm_forward (const std::vector<int8_t> &x, const uint32_t batch,
                  const uint32_t features, const uint32_t window,
                  const bool rms, const uint32_t eps_q30,
                  const std::vector<int8_t> &gamma,
                  const std::vector<int8_t> &beta, const bool use_gamma,
                  const bool use_bias)
{
    const uint32_t per_row = features / window;
    RefNorm result;
    result.xhat.assign (x.size (), 0.0);
    result.dev.assign (batch * per_row, 0.0);
    result.out.assign (x.size (), 0);
    for (uint32_t row = 0; row < batch; ++row)
        for (uint32_t w = 0; w < per_row; ++w)
            {
                const uint32_t base = row * features + w * window;
                double sum = 0.0;
                double squares = 0.0;
                for (uint32_t i = 0; i < window; ++i)
                    {
                        const double value = x[base + i] / 127.0;
                        sum += value;
                        squares += value * value;
                    }
                const double mean = rms ? 0.0 : sum / window;
                double variance = 0.0;
                for (uint32_t i = 0; i < window; ++i)
                    {
                        const double centered = x[base + i] / 127.0 - mean;
                        variance += centered * centered;
                    }
                variance /= window;
                const double deviation
                    = std::sqrt (variance + static_cast<double> (eps_q30)
                                                 / 1073741824.0);
                result.dev[row * per_row + w] = deviation;
                for (uint32_t i = 0; i < window; ++i)
                    {
                        const uint32_t feature = w * window + i;
                        const double centered = x[base + i] / 127.0 - mean;
                        const double xhat = deviation > 0.0 ? centered / deviation
                                                           : 0.0;
                        result.xhat[base + i] = xhat;
                        const double g = use_gamma ? gamma[feature] / 127.0 : 1.0;
                        const double b = use_bias ? beta[feature] / 127.0 : 0.0;
                        result.out[base + i] = static_cast<int8_t> (
                            ref_i8 ((g * xhat + b) * 127.0));
                    }
            }
    return result;
}

struct RefNormGrad
{
    std::vector<double> dx;
    std::vector<double> dgamma;
    std::vector<double> dbeta;
};

RefNormGrad
ref_norm_backward (const std::vector<int8_t> &x, const RefNorm &forward,
                   const std::vector<int16_t> &grad_out, const uint32_t batch,
                   const uint32_t features, const uint32_t window,
                   const bool rms, const std::vector<int8_t> &gamma,
                   const bool use_gamma, const bool use_bias)
{
    const uint32_t per_row = features / window;
    RefNormGrad result;
    result.dx.assign (x.size (), 0.0);
    result.dgamma.assign (features, 0.0);
    result.dbeta.assign (features, 0.0);
    for (uint32_t row = 0; row < batch; ++row)
        for (uint32_t w = 0; w < per_row; ++w)
            {
                const uint32_t base = row * features + w * window;
                const double deviation = forward.dev[row * per_row + w];
                double mean_routed = 0.0;
                double mean_routed_xhat = 0.0;
                for (uint32_t i = 0; i < window; ++i)
                    {
                        const uint32_t feature = w * window + i;
                        const double g = grad_out[base + i] / 32768.0;
                        const double gamma_real
                            = use_gamma ? gamma[feature] / 127.0 : 1.0;
                        const double routed = g * gamma_real;
                        mean_routed += routed;
                        mean_routed_xhat += routed * forward.xhat[base + i];
                        result.dgamma[feature] += g * forward.xhat[base + i];
                        if (use_bias)
                            result.dbeta[feature] += g;
                    }
                mean_routed /= window;
                mean_routed_xhat /= window;
                for (uint32_t i = 0; i < window; ++i)
                    {
                        const uint32_t feature = w * window + i;
                        const double g = grad_out[base + i] / 32768.0;
                        const double gamma_real
                            = use_gamma ? gamma[feature] / 127.0 : 1.0;
                        const double routed = g * gamma_real;
                        const double dx
                            = deviation > 0.0
                                  ? (routed - mean_routed
                                     - forward.xhat[base + i]
                                           * mean_routed_xhat)
                                        / deviation
                                  : 0.0;
                        result.dx[base + i] = dx;
                    }
            }
    return result;
}

/* Double model of the non-rotary attention backward. */
void
ref_attention_backward (const std::vector<int8_t> &q,
                        const std::vector<int8_t> &k,
                        const std::vector<int8_t> &v,
                        const std::vector<int16_t> &grad_out,
                        const uint32_t sequence, const uint32_t head_dim,
                        const uint32_t heads, const bool causal,
                        std::vector<double> &dq, std::vector<double> &dk,
                        std::vector<double> &dv)
{
    const uint32_t stride = heads * head_dim;
    dq.assign (q.size (), 0.0);
    dk.assign (k.size (), 0.0);
    dv.assign (v.size (), 0.0);
    const double root = std::sqrt (static_cast<double> (head_dim));
    for (uint32_t head = 0; head < heads; ++head)
        for (uint32_t i = 0; i < sequence; ++i)
            {
                const uint32_t limit = causal ? i + 1u : sequence;
                std::vector<double> score (limit, 0.0);
                double maximum = -1e300;
                for (uint32_t j = 0; j < limit; ++j)
                    {
                        double dot = 0.0;
                        for (uint32_t d = 0; d < head_dim; ++d)
                            dot += (q[i * stride + head * head_dim + d] / 127.0)
                                   * (k[j * stride + head * head_dim + d] / 127.0);
                        score[j] = dot / root;
                        maximum = std::max (maximum, score[j]);
                    }
                double total = 0.0;
                for (uint32_t j = 0; j < limit; ++j)
                    {
                        score[j] = std::exp (score[j] - maximum);
                        total += score[j];
                    }
                for (uint32_t j = 0; j < limit; ++j)
                    score[j] /= total;

                std::vector<double> dprob (limit, 0.0);
                for (uint32_t j = 0; j < limit; ++j)
                    {
                        double dot = 0.0;
                        for (uint32_t d = 0; d < head_dim; ++d)
                            dot += (grad_out[i * stride + head * head_dim + d]
                                    / 32768.0)
                                   * (v[j * stride + head * head_dim + d] / 127.0);
                        dprob[j] = dot;
                        for (uint32_t d = 0; d < head_dim; ++d)
                            dv[j * stride + head * head_dim + d]
                                += score[j]
                                   * (grad_out[i * stride + head * head_dim + d]
                                      / 32768.0);
                    }
                double contraction = 0.0;
                for (uint32_t j = 0; j < limit; ++j)
                    contraction += score[j] * dprob[j];
                for (uint32_t j = 0; j < limit; ++j)
                    {
                        const double ds
                            = score[j] * (dprob[j] - contraction) / root;
                        for (uint32_t d = 0; d < head_dim; ++d)
                            {
                                dq[i * stride + head * head_dim + d]
                                    += ds
                                       * (k[j * stride + head * head_dim + d]
                                          / 127.0);
                                dk[j * stride + head * head_dim + d]
                                    += ds
                                       * (q[i * stride + head * head_dim + d]
                                          / 127.0);
                            }
                    }
            }
}

/* Largest absolute deviation between a lane buffer and a double reference. */
double
max_lane_error (const std::vector<int16_t> &lanes,
                const std::vector<double> &reference, const bool scale)
{
    double worst = 0.0;
    for (std::size_t i = 0; i < lanes.size (); ++i)
        {
            const double expected = scale ? reference[i] * 32768.0
                                          : reference[i];
            worst = std::max (worst, std::abs (static_cast<double> (lanes[i])
                                               - expected));
        }
    return worst;
}

/* Packed inference ATTENTION, used only to pin the training forward. */
struct InferenceResult
{
    bool ok = false;
    std::vector<int8_t> values;
};

InferenceResult
run_inference_attention (const std::vector<int8_t> &packed,
                         const uint16_t sequence, const uint8_t head_dim,
                         const uint8_t heads, const bool causal,
                         const bool rope)
{
    mllee_test::LayerSpec layer;
    layer._M_op = Op::ATTENTION;
    layer._M_in_dim = static_cast<uint16_t> (packed.size ());
    layer._M_out_dim = static_cast<uint16_t> (sequence * head_dim * heads);
    layer._M_head_dim = head_dim;
    layer._M_seq = sequence;
    layer._M_groups = heads;
    layer._M_in_off = _S_external_input;
    layer._M_out_off = 0;
    layer._M_in_len = static_cast<uint16_t> (packed.size ());
    layer._M_out_len = layer._M_out_dim;
    layer._M_flags = INPUT_EXTERNAL;
    if (causal)
        layer._M_flags |= ATTN_CAUSAL;
    if (rope)
        layer._M_flags |= ATTN_ROPE;

    const std::vector<uint8_t> blob
        = mllee_test::BlobBuilder ({ layer }).build (_S_arena_bytes);
    Core core;
    auto result = core.run (blob.data (), static_cast<uint32_t> (blob.size ()),
                            packed.data (),
                            static_cast<uint32_t> (packed.size ()));
    InferenceResult out;
    if (result.has_value ())
        {
            out.ok = true;
            out.values.assign (result.value ()._M_data,
                               result.value ()._M_data + result.value ()._M_len);
        }
    return out;
}

/* Run one tape pass with a constant upstream lane K, returning the operand
   gradients the backward accumulated.  A constant seed also keeps lane 0 of
   the output at K, so the gradient the forward wrote is not disturbed. */
template <typename RunFn>
Status
run_with_constant_grad (const int16_t seed, RunFn &&run, Trainable &output)
{
    AlignedArena<4096> arena;
    Pool pool = arena.pool ();
    Node nodes[8];
    Tape tape (nodes, 8u, pool);
    const Status begin = tape.begin ();
    if (!status_ok (begin))
        return begin;
    tape.set_loss_scale (seed);
    const Status recorded = run (tape);
    if (!status_ok (recorded))
        return recorded;
    for (uint32_t i = 0; i < output._M_count; ++i)
        output._M_grad[i] = seed;
    return tape.backward (output);
}
} // namespace

/* ================================================================== */
/* Norm geometry and pool accounting                                  */
/* ================================================================== */

TEST_CASE ("NormSpec validate covers geometry and flags")
{
    NormSpec spec;
    spec._M_features = 8u;
    REQUIRE (spec.validate () == Status::OK);
    REQUIRE (spec.window () == 8u);
    REQUIRE (spec.windows_per_row () == 1u);
    REQUIRE (spec.elements () == 8u);
    REQUIRE (spec.input_shape () == Shape::matrix (1u, 8u));

    spec._M_batch = 3u;
    spec._M_window = 4u;
    REQUIRE (spec.windows_per_row () == 2u);
    REQUIRE (spec.windows () == 6u);

    SECTION ("a window that does not divide the feature axis is rejected")
    {
        spec._M_window = 3u;
        REQUIRE (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("zero axes and out-of-range counts")
    {
        spec._M_window = 0u;
        spec._M_features = 0u;
        REQUIRE (spec.validate () == Status::BAD_LAYOUT);
        spec._M_features = 8u;
        spec._M_batch = 8191u;
        REQUIRE (spec.validate () == Status::OK); /* 65528 elements */
        spec._M_batch = 8192u;
        REQUIRE (spec.validate () == Status::OUT_OF_RANGE); /* 65536 */
    }
    SECTION ("flag bytes outside {0, 1} are rejected")
    {
        NormSpec flags;
        flags._M_features = 4u;
        flags._M_rms = 2u;
        REQUIRE (flags.validate () == Status::BAD_LAYOUT);
        flags._M_rms = 0u;
        flags._M_use_gamma = 2u;
        REQUIRE (flags.validate () == Status::BAD_LAYOUT);
        flags._M_use_gamma = 0u;
        flags._M_use_bias = 2u;
        REQUIRE (flags.validate () == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("NormSpec::bytes is the exact pool cost of a recorded layer")
{
    constexpr NormSpec spec = []
    {
        NormSpec s;
        s._M_batch = 2u;
        s._M_features = 4u;
        s._M_use_bias = 1u;
        s._M_rms = 1u;
        return s;
    } ();
    static_assert (spec.windows () == 2u, "one window per row");
    constexpr uint32_t stats = ref_align_up (2u * sizeof (NormStats));
    constexpr uint32_t scratch = ref_align_up (2u * 4u * 4u);
    constexpr uint32_t total
        = stats + scratch + ref_align_up (static_cast<uint32_t> (sizeof (NormSpec)));
    static_assert (total == 68u, "16 + 32 + 20 bytes");

    Buf input = make_buf ({ 10, -20, 30, 40, -50, 60, -70, 80 });
    Buf gamma = make_buf ({ 127, 120, 100, 80 });
    Buf bias = make_buf ({ 2, -3, 4, -5 });
    Buf output = make_buf (std::vector<int8_t> (8, 0));

    AlignedArena<total> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);
    Trainable xv = view (input, spec.input_shape ());
    Trainable gv = view (gamma, spec.gamma_shape ());
    Trainable bv = view (bias, spec.gamma_shape ());
    Trainable ov = view (output, spec.input_shape ());
    REQUIRE (spec.bytes () == total);
    REQUIRE (norm_run (tape, spec, xv, gv, bv, ov) == Status::OK);
    REQUIRE (tape.node_count () == 1u);
    REQUIRE (pool.stats ()._M_used == total);
    REQUIRE (pool.remaining () == 0u);

    SECTION ("one slab short reports ARENA_OVERFLOW and records nothing")
    {
        AlignedArena<total - 4u> small;
        Pool short_pool = small.pool ();
        Node short_nodes[2];
        Tape short_tape (short_nodes, 2u, short_pool);
        REQUIRE (short_tape.begin () == Status::OK);
        REQUIRE (norm_run (short_tape, spec, xv, gv, bv, ov)
                 == Status::ARENA_OVERFLOW);
        REQUIRE (short_tape.node_count () == 0u);
    }
}

/* ================================================================== */
/* Norm forward                                                       */
/* ================================================================== */

TEST_CASE ("layer norm forward matches a double reference")
{
    const std::vector<int8_t> codes = { 30, -20, 10, 70, -60, 40, 5, -15 };
    const std::vector<int8_t> gamma = { 100, 120, 127, 80 };
    const std::vector<int8_t> beta = { 3, -4, 2, 0 };
    const RefNorm reference
        = ref_norm_forward (codes, 2u, 4u, 4u, false, 0u, gamma, beta, true,
                            true);

    Buf input = make_buf (codes);
    Buf gamma_buf = make_buf (gamma);
    Buf beta_buf = make_buf (beta);
    Buf output = make_buf (std::vector<int8_t> (codes.size (), 0));
    NormSpec spec;
    spec._M_batch = 2u;
    spec._M_features = 4u;
    spec._M_use_bias = 1u;

    AlignedArena<256> arena;
    Pool pool = arena.pool ();
    NormStats stats[2];
    Trainable xv = view (input, spec.input_shape ());
    Trainable gv = view (gamma_buf, spec.gamma_shape ());
    Trainable bv = view (beta_buf, spec.gamma_shape ());
    Trainable ov = view (output, spec.input_shape ());
    REQUIRE (norm_forward (spec, xv, gv, bv, ov, stats) == Status::OK);
    for (std::size_t i = 0; i < codes.size (); ++i)
        CHECK (std::abs (output.code[i] - reference.out[i]) <= 1);
    REQUIRE (stats[0]._M_deviation > 0);
    /* (30 - 20 + 10 + 70) * 258 / 4 = 5805 exactly. */
    CHECK (stats[0]._M_mean == 5805);
}

TEST_CASE ("layer norm is invariant to a constant shift, RMS norm is not")
{
    const std::vector<int8_t> first = { 10, 20, 30, 40 };
    std::vector<int8_t> shifted = first;
    for (int8_t &code : shifted)
        code = static_cast<int8_t> (code + 20);

    const auto run = [] (const std::vector<int8_t> &codes) {
        Buf input = make_buf (codes);
        Buf output = make_buf (std::vector<int8_t> (codes.size (), 0));
        NormSpec spec;
        spec._M_features = 4u;
        spec._M_use_gamma = 0u;
        AlignedArena<128> arena;
        Pool pool = arena.pool ();
        NormStats stats[1];
        Trainable xv = view (input, spec.input_shape ());
        Trainable ov = view (output, spec.input_shape ());
        REQUIRE (norm_forward (spec, xv, Trainable{}, Trainable{}, ov, stats)
                 == Status::OK);
        return output.code;
    };

    CHECK (run (first) == run (shifted));

    const std::vector<int8_t> rms_first = { 10, 20, 30, 40 };
    std::vector<int8_t> rms_shifted = rms_first;
    for (int8_t &code : rms_shifted)
        code = static_cast<int8_t> (code + 20);
    const auto run_rms = [] (const std::vector<int8_t> &codes) {
        Buf input = make_buf (codes);
        Buf output = make_buf (std::vector<int8_t> (codes.size (), 0));
        NormSpec spec;
        spec._M_features = 4u;
        spec._M_use_gamma = 0u;
        spec._M_rms = 1u;
        AlignedArena<128> arena;
        Pool pool = arena.pool ();
        NormStats stats[1];
        Trainable xv = view (input, spec.input_shape ());
        Trainable ov = view (output, spec.input_shape ());
        REQUIRE (norm_forward (spec, xv, Trainable{}, Trainable{}, ov, stats)
                 == Status::OK);
        return output.code;
    };
    CHECK (run_rms (rms_first) != run_rms (rms_shifted));
}

/* ================================================================== */
/* Norm backward                                                      */
/* ================================================================== */

TEST_CASE ("norm backward matches the analytic LayerNorm Jacobian")
{
    const std::vector<int8_t> codes = { 30, -20, 10, 70, -60, 40, 5, -15 };
    const std::vector<int8_t> gamma = { 100, 120, 127, 80 };
    const std::vector<int8_t> beta = { 3, -4, 2, 0 };
    const std::vector<int16_t> upstream (codes.size (), 8192);
    const RefNorm forward = ref_norm_forward (codes, 2u, 4u, 4u, false, 0u,
                                              gamma, beta, true, true);
    const RefNormGrad reference = ref_norm_backward (
        codes, forward, upstream, 2u, 4u, 4u, false, gamma, true, true);

    Buf input = make_buf (codes);
    Buf gamma_buf = make_buf (gamma);
    Buf beta_buf = make_buf (beta);
    Buf output = make_buf (std::vector<int8_t> (codes.size (), 0));
    NormSpec spec;
    spec._M_batch = 2u;
    spec._M_features = 4u;
    spec._M_use_bias = 1u;

    Trainable xv = view (input, spec.input_shape ());
    Trainable gv = view (gamma_buf, spec.gamma_shape ());
    Trainable bv = view (beta_buf, spec.gamma_shape ());
    Trainable ov = view (output, spec.input_shape ());
    REQUIRE (run_with_constant_grad (
                 8192,
                 [&] (Tape &tape) {
                     return norm_run (tape, spec, xv, gv, bv, ov);
                 },
                 ov)
             == Status::OK);
    CHECK (max_lane_error (input.grad, reference.dx, true) <= 16.0);
    CHECK (max_lane_error (gamma_buf.grad, reference.dgamma, true) <= 2.0);
    for (uint32_t i = 0; i < 4u; ++i)
        CHECK (beta_buf.grad[i]
               == static_cast<int16_t> (2 * 8192)); /* one per batch row */

    /* The centered Jacobian sums to zero across a window. */
    for (uint32_t row = 0; row < 2u; ++row)
        {
            int32_t total = 0;
            for (uint32_t i = 0; i < 4u; ++i)
                total += input.grad[row * 4u + i];
            CHECK (std::abs (total) <= 8);
        }
}

TEST_CASE ("norm backward matches the analytic RMSNorm Jacobian")
{
    const std::vector<int8_t> codes = { 30, -20, 10, 70, -60, 40, 5, -15 };
    const std::vector<int8_t> gamma = { 127, 100, 90, 70 };
    const std::vector<int16_t> upstream (codes.size (), 4096);
    const RefNorm forward = ref_norm_forward (codes, 2u, 4u, 4u, true, 0u,
                                              gamma, {}, true, false);
    const RefNormGrad reference = ref_norm_backward (
        codes, forward, upstream, 2u, 4u, 4u, true, gamma, true, false);

    Buf input = make_buf (codes);
    Buf gamma_buf = make_buf (gamma);
    Buf output = make_buf (std::vector<int8_t> (codes.size (), 0));
    NormSpec spec;
    spec._M_batch = 2u;
    spec._M_features = 4u;
    spec._M_rms = 1u;

    Trainable xv = view (input, spec.input_shape ());
    Trainable gv = view (gamma_buf, spec.gamma_shape ());
    Trainable ov = view (output, spec.input_shape ());
    REQUIRE (run_with_constant_grad (
                 4096,
                 [&] (Tape &tape) {
                     return norm_run (tape, spec, xv, gv, Trainable{}, ov);
                 },
                 ov)
             == Status::OK);
    CHECK (max_lane_error (input.grad, reference.dx, true) <= 16.0);
    CHECK (max_lane_error (gamma_buf.grad, reference.dgamma, true) <= 2.0);
}

TEST_CASE ("a constant window takes the exact-centre branch")
{
    const std::vector<int8_t> codes (4, 25);
    const std::vector<int8_t> gamma = { 127, 127, 127, 127 };
    const std::vector<int16_t> upstream (4, 16384);

    Buf input = make_buf (codes);
    Buf gamma_buf = make_buf (gamma);
    Buf beta_buf = make_buf ({ 1, 2, 3, 4 });
    Buf output = make_buf (std::vector<int8_t> (4, 0));
    NormSpec spec;
    spec._M_features = 4u;
    spec._M_use_bias = 1u;

    Trainable xv = view (input, spec.input_shape ());
    Trainable gv = view (gamma_buf, spec.gamma_shape ());
    Trainable bv = view (beta_buf, spec.gamma_shape ());
    Trainable ov = view (output, spec.input_shape ());
    AlignedArena<128> arena;
    Pool pool = arena.pool ();
    NormStats stats[1];
    REQUIRE (norm_forward (spec, xv, gv, bv, ov, stats) == Status::OK);
    REQUIRE (stats[0]._M_deviation == 0);
    /* y = gamma * 0 + beta, so the output is the bias row itself. */
    CHECK (output.code == beta_buf.code);

    REQUIRE (run_with_constant_grad (
                 16384,
                 [&] (Tape &tape) {
                     return norm_run (tape, spec, xv, gv, bv, ov);
                 },
                 ov)
             == Status::OK);
    for (uint32_t i = 0; i < 4u; ++i)
        {
            CHECK (input.grad[i] == 0);
            CHECK (gamma_buf.grad[i] == 0);
            CHECK (beta_buf.grad[i] == 16384);
        }
}

/* ================================================================== */
/* Residual add                                                       */
/* ================================================================== */

TEST_CASE ("residual add saturates and routes to both branches")
{
    Buf a = make_buf ({ 100, -100, 50, -50 });
    Buf b = make_buf ({ 100, -100, -80, 120 });
    Buf out = make_buf (std::vector<int8_t> (4, 0));
    Trainable av = view (a, Shape::vector (4u));
    Trainable bv = view (b, Shape::vector (4u));
    Trainable ov = view (out, Shape::vector (4u));

    REQUIRE (residual_forward (av, bv, ov) == Status::OK);
    CHECK (out.code == std::vector<int8_t> ({ 127, -127, -30, 70 }));

    REQUIRE (run_with_constant_grad (
                 4096,
                 [&] (Tape &tape) { return residual_run (tape, av, bv, ov); },
                 ov)
             == Status::OK);
    CHECK (a.grad == b.grad);
    CHECK (a.grad == std::vector<int16_t> ({ 4096, 4096, 4096, 4096 }));

    SECTION ("a mismatched shape is rejected")
    {
        Buf other = make_buf ({ 1, 2, 3, 4, 5, 6, 7, 8 });
        Trainable wrong = view (other, Shape::vector (8u));
        CHECK (residual_forward (av, wrong, ov) == Status::BAD_LAYOUT);
        CHECK (residual_forward (av, Trainable{}, ov) == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("residual plus norm equals a norm of the summed codes")
{
    Buf a = make_buf ({ 40, -70, 20, 5 });
    Buf b = make_buf ({ 10, 60, -30, 15 });
    Buf sum = make_buf (std::vector<int8_t> (4, 0));
    Buf fused = make_buf (std::vector<int8_t> (4, 0));
    NormSpec spec;
    spec._M_features = 4u;
    spec._M_use_gamma = 0u;
    AlignedArena<128> arena;
    Pool pool = arena.pool ();
    NormStats stats[1];
    Trainable av = view (a, spec.input_shape ());
    Trainable bv = view (b, spec.input_shape ());
    Trainable sv = view (sum, spec.input_shape ());
    Trainable fv = view (fused, spec.input_shape ());
    REQUIRE (residual_forward (av, bv, sv) == Status::OK);
    REQUIRE (norm_forward (spec, sv, Trainable{}, Trainable{}, fv, stats)
             == Status::OK);
    std::vector<int8_t> summed_codes = a.code;
    for (std::size_t i = 0; i < summed_codes.size (); ++i)
        summed_codes[i] = static_cast<int8_t> (
            ref_i8 (static_cast<int> (a.code[i]) + b.code[i]));
    REQUIRE (sum.code == summed_codes);
    const RefNorm reference
        = ref_norm_forward (summed_codes, 1u, 4u, 4u, false, 0u, {}, {}, false,
                            false);
    for (std::size_t i = 0; i < fused.code.size (); ++i)
        CHECK (std::abs (fused.code[i] - reference.out[i]) <= 1);
}

/* ================================================================== */
/* Attention geometry                                                 */
/* ================================================================== */

TEST_CASE ("AttentionSpec validate covers geometry and flags")
{
    AttentionSpec spec;
    REQUIRE (spec.validate () == Status::BAD_LAYOUT); /* zero sequence */
    spec._M_seq = 3u;
    spec._M_head_dim = 2u;
    spec._M_heads = 2u;
    REQUIRE (spec.validate () == Status::OK);
    REQUIRE (spec.stride () == 4u);
    REQUIRE (spec.elements () == 12u);
    REQUIRE (spec.token_shape () == Shape::matrix (3u, 4u));

    SECTION ("head dimension above eight")
    {
        spec._M_head_dim = 9u;
        REQUIRE (spec.validate () == Status::BAD_LAYOUT);
        spec._M_head_dim = 8u;
        REQUIRE (spec.validate () == Status::OK);
    }
    SECTION ("RoPE needs an even head dimension")
    {
        spec._M_head_dim = 3u;
        spec._M_rope = 1u;
        REQUIRE (spec.validate () == Status::BAD_LAYOUT);
        spec._M_head_dim = 4u;
        REQUIRE (spec.validate () == Status::OK);
    }
    SECTION ("flag bytes outside {0, 1}")
    {
        spec._M_causal = 2u;
        REQUIRE (spec.validate () == Status::BAD_LAYOUT);
        spec._M_causal = 0u;
        spec._M_rope = 2u;
        REQUIRE (spec.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("the token grid must fit a training view")
    {
        spec._M_seq = 65535u;
        spec._M_heads = 2u;
        REQUIRE (spec.validate () == Status::OUT_OF_RANGE);
        spec._M_heads = 1u;
        spec._M_head_dim = 1u;
        REQUIRE (spec.validate () == Status::OK);
    }
}

TEST_CASE ("attention forward is byte-identical to inference")
{
    const uint16_t sequence = 3u;
    const uint8_t head_dim = 2u;
    const uint8_t heads = 2u;
    const uint32_t stride = head_dim * heads;
    const std::vector<int8_t> q = {
        60, -30, 10, 45,  -20, 70, -55, 15,  30, 40, -25, 5,
    };
    const std::vector<int8_t> k = {
        20, 50, -40, 15,  65, -10, 35, -60,  -5, 25, 70, -35,
    };
    const std::vector<int8_t> v = {
        100, -20, 30, -45,  -60, 80, -15, 55,  25, -70, 45, 10,
    };
    std::vector<int8_t> packed = q;
    packed.insert (packed.end (), k.begin (), k.end ());
    packed.insert (packed.end (), v.begin (), v.end ());

    for (const bool causal : { false, true })
        for (const bool rope : { false, true })
            {
                INFO ("causal " << causal << " rope " << rope);
                const InferenceResult reference = run_inference_attention (
                    packed, sequence, head_dim, heads, causal, rope);
                REQUIRE (reference.ok);

                Buf qb = make_buf (q);
                Buf kb = make_buf (k);
                Buf vb = make_buf (v);
                Buf out = make_buf (
                    std::vector<int8_t> (sequence * stride, 0));
                AttentionSpec spec;
                spec._M_seq = sequence;
                spec._M_heads = heads;
                spec._M_head_dim = head_dim;
                spec._M_causal = causal ? 1u : 0u;
                spec._M_rope = rope ? 1u : 0u;
                Trainable qv = view (qb, spec.token_shape ());
                Trainable kv = view (kb, spec.token_shape ());
                Trainable vv = view (vb, spec.token_shape ());
                Trainable ov = view (out, spec.token_shape ());
                REQUIRE (attention_forward (spec, qv, kv, vv, ov)
                         == Status::OK);
                CHECK (out.code == reference.values);
            }
}

/* ================================================================== */
/* Attention backward                                                 */
/* ================================================================== */

TEST_CASE ("a single key gives zero query and key gradients")
{
    for (const bool rope : { false, true })
        {
            INFO ("rope " << rope);
            const uint8_t head_dim = 2u;
            Buf q = make_buf ({ 127, 0 });
            Buf k = make_buf ({ 127, 0 });
            Buf v = make_buf ({ 100, -30 });
            Buf out = make_buf ({ 0, 0 });
            AttentionSpec spec;
            spec._M_seq = 1u;
            spec._M_heads = 1u;
            spec._M_head_dim = head_dim;
            spec._M_rope = rope ? 1u : 0u;
            Trainable qv = view (q, spec.token_shape ());
            Trainable kv = view (k, spec.token_shape ());
            Trainable vv = view (v, spec.token_shape ());
            Trainable ov = view (out, spec.token_shape ());
            REQUIRE (run_with_constant_grad (
                         8192,
                         [&] (Tape &tape) {
                             return attention_run (tape, spec, qv, kv, vv, ov);
                         },
                         ov)
                     == Status::OK);
            CHECK (q.grad[0] == 0);
            CHECK (q.grad[1] == 0);
            CHECK (k.grad[0] == 0);
            CHECK (k.grad[1] == 0);
            /* p == 1, so dV is the upstream lane (one LSB of the ratio). */
            CHECK (std::abs (v.grad[0] - 8192) <= 1);
            CHECK (std::abs (v.grad[1] - 8192) <= 1);
        }
}

TEST_CASE ("attention backward isolates heads and honours the causal mask")
{
    const uint16_t sequence = 3u;
    const uint8_t head_dim = 2u;
    const uint8_t heads = 2u;
    const uint32_t stride = head_dim * heads;
    /* Every key must score exactly the same against a query, otherwise the
       exp table (one step per 1/8 nat) flattens every non-maximum weight to
       its floor and the softmax Jacobian rounds to zero.  Q is [a, 0] and K is
       [1, c], so q.k == a for every key regardless of c; the keys still carry
       distinct second lanes, which is what dQ needs. */
    const std::vector<int8_t> q_codes = {
        10, 0, 5, 0,  20, 0, 6, 0,  30, 0, 7, 0,
    };
    const std::vector<int8_t> k_codes = {
        1, 0, 1, 0,  1, 60, 1, 60,  1, -120, 1, -120,
    };
    const std::vector<int8_t> v_codes = {
        100, -20, 30, -45,  30, -60, -60, 80,  -40, 80, -15, 55,
    };

    /* Only head 0 receives upstream signal. */
    std::vector<int16_t> upstream (sequence * stride, 0);
    for (uint32_t position = 0; position < sequence; ++position)
        for (uint32_t d = 0; d < head_dim; ++d)
            upstream[position * stride + d] = 2048;

    const auto run = [&] (const bool causal) {
        Buf q = make_buf (q_codes);
        Buf k = make_buf (k_codes);
        Buf v = make_buf (v_codes);
        Buf out = make_buf (std::vector<int8_t> (sequence * stride, 0));
        AttentionSpec spec;
        spec._M_seq = sequence;
        spec._M_heads = heads;
        spec._M_head_dim = head_dim;
        spec._M_causal = causal ? 1u : 0u;
        Trainable qv = view (q, spec.token_shape ());
        Trainable kv = view (k, spec.token_shape ());
        Trainable vv = view (v, spec.token_shape ());
        Trainable ov = view (out, spec.token_shape ());
        AlignedArena<512> arena;
        Pool pool = arena.pool ();
        Node nodes[4];
        Tape tape (nodes, 4u, pool);
        REQUIRE (tape.begin () == Status::OK);
        tape.set_loss_scale (2048);
        REQUIRE (attention_run (tape, spec, qv, kv, vv, ov) == Status::OK);
        for (uint32_t i = 0; i < ov._M_count; ++i)
            ov._M_grad[i] = upstream[i];
        REQUIRE (tape.backward (ov) == Status::OK);
        return std::make_tuple (q.grad, k.grad, v.grad);
    };

    const auto [causal_q, causal_k, causal_v] = run (true);
    const auto [plain_q, plain_k, plain_v] = run (false);

    /* Head 1 never got a gradient lane. */
    for (uint32_t position = 0; position < sequence; ++position)
        for (uint32_t d = 0; d < head_dim; ++d)
            {
                const uint32_t lane = position * stride + head_dim + d;
                CHECK (causal_q[lane] == 0);
                CHECK (causal_k[lane] == 0);
                CHECK (causal_v[lane] == 0);
            }
    /* Query 0 attends only key 0, so its single-key softmax has dQ == 0,
       while the unmasked forward mixes the other keys into the gradient. */
    CHECK (causal_q[0] == 0);
    CHECK (causal_q[1] == 0);
    CHECK ((plain_q[0] != 0 || plain_q[1] != 0));
    /* The causal value gradient only sums the rows allowed to see that key. */
    CHECK (causal_v != plain_v);
}

TEST_CASE ("attention backward matches a double reference")
{
    const uint16_t sequence = 3u;
    const uint8_t head_dim = 2u;
    const uint8_t heads = 1u;
    const uint32_t stride = head_dim;
    /* Balanced scores, as in the test above: q = [a, 0], k = [1, c] makes the
       integer score a for every key of a query, so the table softmax is the
       uniform 1/3 the double reference computes.  The keys still differ in
       their second lane, so dQ is not trivially zero. */
    const std::vector<int8_t> q_codes = { 10, 0, 20, 0, 30, 0 };
    const std::vector<int8_t> k_codes = { 1, 0, 1, 60, 1, -120 };
    const std::vector<int8_t> v_codes = { 100, -20, 30, -60, -40, 80 };
    std::vector<int16_t> upstream (sequence * stride, 0);
    for (std::size_t i = 0; i < upstream.size (); ++i)
        upstream[i] = static_cast<int16_t> (1024 + 512 * static_cast<int> (i));

    std::vector<double> dq, dk, dv;
    ref_attention_backward (q_codes, k_codes, v_codes, upstream, sequence,
                            head_dim, heads, false, dq, dk, dv);

    Buf q = make_buf (q_codes);
    Buf k = make_buf (k_codes);
    Buf v = make_buf (v_codes);
    Buf out = make_buf (std::vector<int8_t> (sequence * stride, 0));
    AttentionSpec spec;
    spec._M_seq = sequence;
    spec._M_heads = heads;
    spec._M_head_dim = head_dim;
    Trainable qv = view (q, spec.token_shape ());
    Trainable kv = view (k, spec.token_shape ());
    Trainable vv = view (v, spec.token_shape ());
    Trainable ov = view (out, spec.token_shape ());
    AlignedArena<512> arena;
    Pool pool = arena.pool ();
    Node nodes[4];
    Tape tape (nodes, 4u, pool);
    REQUIRE (tape.begin () == Status::OK);
    /* Tape::backward re-seeds lane 0 with the loss scale, so a multi-lane
       upstream vector is only reproduced when the scale equals its lane 0. */
    tape.set_loss_scale (upstream[0]);
    REQUIRE (attention_run (tape, spec, qv, kv, vv, ov) == Status::OK);
    for (uint32_t i = 0; i < ov._M_count; ++i)
        ov._M_grad[i] = upstream[i];
    REQUIRE (tape.backward (ov) == Status::OK);

    /* The exp table quantizes the softmax argument to 1/8 steps, so the
       probability vector differs from the exact double softmax; the check is
       relative to the largest reference lane plus a quantisation floor. */
    const auto close = [] (const std::vector<int16_t> &lanes,
                           const std::vector<double> &reference,
                           const double scale, const double relative,
                           const double floor) {
        double peak = 0.0;
        for (const double value : reference)
            peak = std::max (peak, std::abs (value) * scale);
        for (std::size_t i = 0; i < lanes.size (); ++i)
            if (std::abs (static_cast<double> (lanes[i])
                          - reference[i] * scale)
                > floor + relative * peak)
                return false;
        return true;
    };
    CHECK (close (v.grad, dv, 32768.0, 0.05, 8.0));
    CHECK (close (q.grad, dq, 32768.0, 0.20, 64.0));
    CHECK (close (k.grad, dk, 32768.0, 0.20, 64.0));
}

TEST_CASE ("attention_run records one node and its exact pool cost")
{
    constexpr AttentionSpec spec = []
    {
        AttentionSpec s;
        s._M_seq = 3u;
        s._M_heads = 2u;
        s._M_head_dim = 2u;
        return s;
    } ();
    static_assert (spec.elements () == 12u, "three tokens of width four");
    static_assert (spec.bytes () == 24u, "12 scratch + 12 spec");

    Buf q = make_buf (std::vector<int8_t> (12, 3));
    Buf k = make_buf (std::vector<int8_t> (12, -2));
    Buf v = make_buf (std::vector<int8_t> (12, 5));
    Buf out = make_buf (std::vector<int8_t> (12, 0));
    Trainable qv = view (q, spec.token_shape ());
    Trainable kv = view (k, spec.token_shape ());
    Trainable vv = view (v, spec.token_shape ());
    Trainable ov = view (out, spec.token_shape ());

    AlignedArena<spec.bytes ()> arena;
    Pool pool = arena.pool ();
    Node nodes[2];
    Tape tape (nodes, 2u, pool);
    REQUIRE (tape.begin () == Status::OK);
    REQUIRE (attention_run (tape, spec, qv, kv, vv, ov) == Status::OK);
    REQUIRE (tape.node_count () == 1u);
    REQUIRE (pool.stats ()._M_used == spec.bytes ());
    REQUIRE (pool.remaining () == 0u);

    const AttentionSpec *const saved
        = tape.node (0u)->_M_saved[_S_attention_spec].is_null ()
              ? nullptr
              : reinterpret_cast<const AttentionSpec *> (
                    pool.at (tape.node (0u)->_M_saved[_S_attention_spec]
                                 ._M_offset));
    REQUIRE (saved != nullptr);
    CHECK (saved->_M_seq == 3u);
    CHECK (saved->_M_heads == 2u);

    AlignedArena<spec.bytes () - 4u> small;
    Pool short_pool = small.pool ();
    Node short_nodes[2];
    Tape short_tape (short_nodes, 2u, short_pool);
    REQUIRE (short_tape.begin () == Status::OK);
    CHECK (attention_run (short_tape, spec, qv, kv, vv, ov)
           == Status::ARENA_OVERFLOW);
    CHECK (short_tape.node_count () == 0u);
}

TEST_CASE ("attention gradient reaches all three projections")
{
    AttentionSpec spec;
    spec._M_seq = 2u;
    spec._M_heads = 1u;
    spec._M_head_dim = 2u;
    Buf q = make_buf ({ 10, 0, 20, 0 });
    Buf k = make_buf ({ 1, 0, 1, 60 });
    Buf v = make_buf ({ 100, -20, 30, -60 });
    Buf out = make_buf ({ 0, 0, 0, 0 });
    Trainable qv = view (q, spec.token_shape ());
    Trainable kv = view (k, spec.token_shape ());
    Trainable vv = view (v, spec.token_shape ());
    Trainable ov = view (out, spec.token_shape ());
    REQUIRE (run_with_constant_grad (
                 2048,
                 [&] (Tape &tape) {
                     return attention_run (tape, spec, qv, kv, vv, ov);
                 },
                 ov)
             == Status::OK);
    CHECK (std::any_of (q.grad.begin (), q.grad.end (),
                        [] (const int16_t lane) { return lane != 0; }));
    CHECK (std::any_of (k.grad.begin (), k.grad.end (),
                        [] (const int16_t lane) { return lane != 0; }));
    CHECK (std::any_of (v.grad.begin (), v.grad.end (),
                        [] (const int16_t lane) { return lane != 0; }));
}
