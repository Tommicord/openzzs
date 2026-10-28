/* Tests for the training weight modules: magnitude pruning (mllee/train_prune.hpp),
 * model averaging (mllee/train_model_averaging.hpp) and knowledge distillation
 * (mllee/train_distill.hpp).
 *
 * Every expectation is derived from an independent reference written inside this
 * file: the integer references implement the documented rounding contract
 * (round-half-up shifts, the 258-unit code step, round-half-away-from-zero
 * averages, saturating rails) with their own code, and a <cmath> reference
 * implements the real-domain softmax / KL that the Q7 log domain approximates.
 * Nothing here asks the library what it just answered; a refactor that changes a
 * documented rule fails here instead of silently moving the reference. */

#include "mllee/format.hpp"
#include "mllee/train_distill.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_loss.hpp"
#include "mllee/train_model_averaging.hpp"
#include "mllee/train_prune.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <vector>

using namespace zzs::mllee;
using namespace zzs::mllee::train;

namespace
{
/* --------------------------------------------------------------------------
 * Integer references: the documented rounding rules, re-implemented here.
 * -------------------------------------------------------------------------- */

/* (v + 2^(shift-1)) >> shift: round half up (ties toward +infinity). */
constexpr int64_t
ref_round_shift (const int64_t v, const unsigned shift)
{
    return (v + (INT64_C (1) << (shift - 1u))) >> shift;
}

constexpr int32_t
ref_clamp_q15 (const int64_t v)
{
    if (v > 32767)
        return 32767;
    if (v < -32768)
        return -32768;
    return static_cast<int32_t> (v);
}

/* Q15 product narrowed with the documented half-up remainder, saturating. */
constexpr int32_t
ref_q15_mul (const int64_t a, const int64_t b)
{
    return ref_clamp_q15 (ref_round_shift (a * b, 15u));
}

/* int16 saturating store of a Q15 intermediate. */
constexpr int32_t
ref_q15_to_i16_sat (const int64_t v)
{
    return ref_clamp_q15 (v);
}

/* The documented code narrowing: clamp(round(v * 127 / 2^15)) to [-127, 127]. */
constexpr int8_t
ref_q15_to_code (const int64_t v_q15)
{
    const int64_t raw = ref_round_shift (v_q15 * 127, 15u);
    if (raw > 127)
        return 127;
    if (raw < -127)
        return -127;
    return static_cast<int8_t> (raw);
}

/* Plain (documented) code widening: code * 258. */
constexpr int64_t
ref_code_to_q15 (const int32_t code)
{
    return static_cast<int64_t> (code) * 258;
}

/* Round half away from zero, integer form, for the averaging accumulator. */
constexpr int64_t
ref_div_round_away (const int64_t n, const uint64_t d)
{
    if (d == 0u)
        return 0;
    const bool negative = n < 0;
    const uint64_t magnitude = negative ? static_cast<uint64_t> (-n)
                                        : static_cast<uint64_t> (n);
    const uint64_t quotient = (magnitude + d / 2u) / d;
    return negative ? -static_cast<int64_t> (quotient)
                    : static_cast<int64_t> (quotient);
}

/* num / den as Q15, truncating, saturating as documented. */
constexpr int32_t
ref_ratio_q15 (const int64_t num, const int64_t den)
{
    if (den == 0)
        return 0;
    const int64_t scaled = (num << 15) / den;
    if (scaled > 32767)
        return 32767;
    if (scaled < -32768)
        return -32768;
    return static_cast<int32_t> (scaled);
}

/* The code magnitude class: |code| clamped to 127. */
constexpr uint32_t
ref_magnitude (const int32_t code)
{
    const int64_t magnitude = code < 0 ? -static_cast<int64_t> (code)
                                       : static_cast<int64_t> (code);
    return magnitude > 127 ? 127u : static_cast<uint32_t> (magnitude);
}

/* floor(2^30 / x) saturating to Q15, 32767 for x <= 0. */
constexpr int32_t
ref_q15_reciprocal (const int64_t x_q15)
{
    if (x_q15 <= 0)
        return 32767;
    const int64_t scaled = (INT64_C (1) << 30) / x_q15;
    return scaled > 32767 ? 32767 : static_cast<int32_t> (scaled);
}

/* --------------------------------------------------------------------------
 * Prune selection references.
 * -------------------------------------------------------------------------- */

/* Threshold walk over magnitude classes 127..0: first class t with
   count(>t) <= keep <= count(>=t); tie budget keep - count(>t). */
std::pair<uint32_t, uint32_t>
ref_threshold_walk (const std::vector<uint32_t>& mags, const uint32_t keep)
{
    uint32_t bins[128] = { 0u };
    for (const uint32_t m : mags)
        ++bins[m];
    uint32_t above = 0u;
    for (uint32_t bin = 128u; bin-- > 0u;)
    {
        const uint32_t here = bins[bin];
        if (above <= keep && above + here >= keep)
            return { bin, keep - above };
        above += here;
    }
    return { 0u, keep };
}

struct SelRef
{
    std::vector<uint8_t> mask;
    uint32_t pruned = 0u;
    uint32_t threshold = 0u;
    uint32_t tie_group = 0u;
    uint32_t tie_kept = 0u;
};

/* The documented unstructured selection: survivors are mag > t, plus the
   first (keep - count(mag > t)) lanes of magnitude t in ascending index
   order. */
SelRef
ref_select (const std::vector<uint32_t>& mags, const uint32_t keep)
{
    const auto [t, tk] = ref_threshold_walk (mags, keep);
    SelRef r;
    r.threshold = t;
    r.tie_kept = tk;
    r.tie_group = static_cast<uint32_t> (
        std::count (mags.begin (), mags.end (), t));
    uint32_t used = 0u;
    for (const uint32_t m : mags)
    {
        uint8_t survives = 0u;
        if (m > t)
            survives = 1u;
        else if (m == t && used < tk)
        {
            survives = 1u;
            ++used;
        }
        r.mask.push_back (survives);
        if (survives == 0u)
            ++r.pruned;
    }
    return r;
}

SelRef
ref_select_codes (const std::vector<int8_t>& codes, const uint32_t keep)
{
    std::vector<uint32_t> mags;
    for (const int8_t c : codes)
        mags.push_back (ref_magnitude (c));
    return ref_select (mags, keep);
}

struct NmRef
{
    std::vector<uint8_t> mask;
    uint32_t pruned = 0u;
    uint32_t blocks = 0u;
    uint32_t partial = 0u;
};

/* N:M reference: every aligned block of m keeps its n largest magnitudes,
   ties to the lower lane index; a short tail keeps its own n largest. */
NmRef
ref_ninm (const std::vector<int8_t>& codes, const uint32_t n, const uint32_t m)
{
    NmRef r;
    const uint32_t count = static_cast<uint32_t> (codes.size ());
    uint32_t index = 0u;
    while (index < count)
    {
        const uint32_t width = std::min (m, count - index);
        if (width < m)
            ++r.partial;
        ++r.blocks;
        std::vector<uint32_t> order (width);
        std::iota (order.begin (), order.end (), 0u);
        std::stable_sort (order.begin (), order.end (),
                          [&](const uint32_t a, const uint32_t b) {
                              return ref_magnitude (codes[index + a])
                                     > ref_magnitude (codes[index + b]);
                          });
        std::vector<uint8_t> keep (width, 0u);
        const uint32_t survivors = std::min (n, width);
        for (uint32_t k = 0u; k < survivors; ++k)
            keep[order[k]] = 1u;
        for (uint32_t lane = 0u; lane < width; ++lane)
        {
            r.mask.push_back (keep[lane]);
            if (keep[lane] == 0u)
                ++r.pruned;
        }
        index += width;
    }
    return r;
}

/* Row L1 scores: row-major codes, `blocks` periods of `rows` rows of
   `row_len` lanes; saturating int32. */
std::vector<int32_t>
ref_row_scores (const std::vector<int8_t>& codes, const uint32_t rows,
                const uint32_t row_len, const uint32_t blocks)
{
    std::vector<int32_t> scores (rows, 0);
    uint32_t pos = 0u;
    for (uint32_t block = 0u; block < blocks; ++block)
        for (uint32_t row = 0u; row < rows; ++row)
        {
            int64_t acc = scores[row];
            for (uint32_t lane = 0u; lane < row_len; ++lane)
            {
                const int32_t c = codes[pos++];
                acc += c < 0 ? -static_cast<int64_t> (c)
                             : static_cast<int64_t> (c);
            }
            scores[row] = acc > 2147483647 ? 2147483647
                                           : static_cast<int32_t> (acc);
        }
    return scores;
}

/* Documented score-to-class fold: floor(score * 127 / max), 0 when max is 0. */
std::vector<uint32_t>
ref_row_mags (const std::vector<int32_t>& scores)
{
    int64_t max_score = 0;
    for (const int32_t s : scores)
        if (s > max_score)
            max_score = s;
    std::vector<uint32_t> mags;
    for (const int32_t s : scores)
        mags.push_back (max_score > 0
                            ? static_cast<uint32_t> (s * 127 / max_score)
                            : 0u);
    return mags;
}

/* --------------------------------------------------------------------------
 * Small tensor fixtures (lane counts stay far below the module maxima).
 * -------------------------------------------------------------------------- */

struct W
{
    alignas (16) int8_t code[64];
    alignas (16) int16_t grad[64];
    uint32_t n = 0u;

    void
    fill (const std::vector<int8_t>& codes, const int16_t grad_value = 0)
    {
        n = static_cast<uint32_t> (codes.size ());
        for (uint32_t i = 0u; i < n; ++i)
        {
            code[i] = codes[i];
            grad[i] = grad_value;
        }
    }

    [[nodiscard]] std::vector<int8_t>
    codes () const
    {
        return std::vector<int8_t> (code, code + n);
    }

    [[nodiscard]] std::vector<int32_t>
    grads () const
    {
        return std::vector<int32_t> (grad, grad + n);
    }

    void
    clear_grad ()
    {
        for (uint32_t i = 0u; i < n; ++i)
            grad[i] = 0;
    }
};

Trainable
view (W& w, const Shape shape)
{
    return Trainable::make (w.code, w.grad, shape, Flag::RequiresGrad);
}

Trainable
view_forward_only (W& w, const Shape shape)
{
    return Trainable::make (w.code, nullptr, shape, Flag::Frozen);
}

PruneMask
make_mask (uint8_t* const bits, const uint32_t count)
{
    PruneMask mask;
    mask._M_bits = bits;
    mask._M_count = count;
    return mask;
}

/* --------------------------------------------------------------------------
 * Distillation float references (real domain).
 * -------------------------------------------------------------------------- */

/* The documented softened Q15 logit of one code. */
constexpr double
ref_logit_real (const int32_t code, const int32_t inv_t_q15)
{
    return ref_q15_mul (ref_code_to_q15 (code), inv_t_q15) / 32768.0;
}

std::vector<double>
ref_softmax (const std::vector<int8_t>& codes, const int32_t inv_t_q15)
{
    double max_z = -1e300;
    for (const int8_t c : codes)
        max_z = std::max (max_z, ref_logit_real (c, inv_t_q15));
    std::vector<double> p;
    double sum = 0.0;
    for (const int8_t c : codes)
    {
        const double e = std::exp (ref_logit_real (c, inv_t_q15) - max_z);
        p.push_back (e);
        sum += e;
    }
    for (double& v : p)
        v /= sum;
    return p;
}

double
ref_kl (const std::vector<double>& pt, const std::vector<double>& ps)
{
    double total = 0.0;
    for (size_t i = 0; i < pt.size (); ++i)
        total += pt[i] * (std::log (pt[i]) - std::log (ps[i]));
    return total;
}

/* --------------------------------------------------------------------------
 * Hand-built backward harness: pool-carved DistillMeta + GradContext, so the
 * backward functions can be called with full control of the meta fields.
 * -------------------------------------------------------------------------- */

struct BackwardHarness
{
    alignas (16) uint8_t pool_mem[512];
    Pool pool { pool_mem, sizeof (pool_mem) };
    Slab slab;
    Saved saved;
    DistillMeta* meta = nullptr;
    Trainable operands[2];
    GradContext context;
    alignas (16) int8_t seed_code[4] = { 0, 0, 0, 0 };
    alignas (16) int16_t seed_grad[4] = { 0, 0, 0, 0 };

    void
    bind (const Trainable& student, const Trainable& teacher)
    {
        pool.reset ();
        const Status status = pool.carve (1u, static_cast<uint32_t> (sizeof (DistillMeta)), slab);
        REQUIRE (status == Status::OK);
        saved._M_offset = slab._M_offset;
        saved._M_bytes = slab._M_bytes;
        saved._M_count = 1u;
        meta = reinterpret_cast<DistillMeta*> (pool.at (slab._M_offset));
        REQUIRE (meta != nullptr);
        *meta = DistillMeta{};
        operands[0] = student;
        operands[1] = teacher;
        context = GradContext{};
        context._M_pool = &pool;
        context._M_operands = operands;
        context._M_operand_count = 2u;
        context._M_saved = &saved;
        context._M_saved_count = 1u;
    }

    [[nodiscard]] Trainable
    seed (const int32_t value)
    {
        seed_grad[0] = static_cast<int16_t> (ref_clamp_q15 (value));
        return Trainable::make (seed_code, seed_grad, Shape::scalar (),
                                Flag::RequiresGrad);
    }
};

/* Documented distill_backward student-lane chain, from probabilities the test
   verified separately against <cmath>. */
int32_t
ref_distill_student_lane (const int32_t p_s, const int32_t p_t,
                          const int32_t inv_t, const int32_t alpha,
                          const uint32_t kl_shift, const bool is_label,
                          const int32_t seed)
{
    const int32_t one_hot = is_label ? 32767 : 0;
    const int32_t one_minus = 32768 - 1 - alpha;
    const int64_t hard_arm
        = (static_cast<int64_t> (ref_clamp_q15 (p_s - one_hot)) * inv_t * alpha)
          >> 30;
    const int64_t soft_raw
        = (static_cast<int64_t> (ref_clamp_q15 (p_s - p_t)) * inv_t * one_minus)
          >> 30;
    const int64_t summed = hard_arm + (soft_raw << kl_shift);
    return ref_q15_mul (ref_clamp_q15 (summed), seed);
}

/* Documented routed-teacher lane chain. */
int32_t
ref_distill_teacher_lane (const int32_t p_t, const int32_t delta_q7,
                          const int32_t kl_q7, const int32_t inv_t,
                          const int32_t alpha, const uint32_t kl_shift,
                          const int32_t seed)
{
    const int32_t one_minus = 32768 - 1 - alpha;
    const int32_t outer = ref_clamp_q15 (
        static_cast<int64_t> (delta_q7 - kl_q7) << 8);
    const int32_t grad_raw
        = ref_q15_mul (ref_q15_mul (ref_q15_mul (p_t, inv_t), outer), one_minus);
    const int64_t wide = static_cast<int64_t> (grad_raw) << kl_shift;
    return ref_q15_mul (ref_clamp_q15 (wide), seed);
}

/* --------------------------------------------------------------------------
 * Averaging references: the documented accumulator, division and narrowing.
 * -------------------------------------------------------------------------- */

/* int64 accumulator ceiling of the averaging pass. */
constexpr int64_t _ref_avg_acc_max = (INT64_C (1) << 60) - 1;

/* Uniform / SWA mean over the picked snapshots, re-implemented: widen each code
   by 258, accumulate in int64 with a per-add rail clamp, divide
   half-away-from-zero, then narrow with the fixed-point round-half-up code rule
   (the rule avg_uniform and avg_swa use). */
std::vector<int8_t>
ref_avg_codes (const std::vector<std::vector<int8_t>>& streams,
               const std::vector<uint32_t>& picks)
{
    std::vector<int8_t> out;
    if (picks.empty ())
        return out;
    const uint32_t count = static_cast<uint32_t> (streams[picks[0]].size ());
    for (uint32_t lane = 0u; lane < count; ++lane)
    {
        int64_t accumulator = 0;
        for (const uint32_t pick : picks)
        {
            accumulator
                += ref_code_to_q15 (static_cast<int32_t> (streams[pick][lane]));
            if (accumulator > _ref_avg_acc_max)
                accumulator = _ref_avg_acc_max;
            else if (accumulator < -_ref_avg_acc_max)
                accumulator = -_ref_avg_acc_max;
        }
        out.push_back (ref_q15_to_code (
            ref_div_round_away (accumulator, picks.size ())));
    }
    return out;
}

/* The train_simd::q15_to_codes narrowing, i.e. round-half-away-from-zero on the
   magnitude: it differs from q15_to_code at an exact negative tie, so the EMA
   read (which goes through the SIMD helper) is checked against this one. */
constexpr int8_t
ref_simd_q15_to_code (const int32_t value_q15)
{
    const int32_t magnitude = value_q15 < 0 ? -value_q15 : value_q15;
    const int32_t scaled = (magnitude * 127 + 16384) >> 15;
    const int32_t signed_value = value_q15 < 0 ? -scaled : scaled;
    if (signed_value < -127)
        return -127;
    if (signed_value > 127)
        return 127;
    return static_cast<int8_t> (signed_value);
}

/* One documented EMA fold: s += (t - s) * (32767 - decay) / 2^15 with the
   half-up product, saturating int16 store, reporting whether it clamped. */
constexpr int32_t
ref_ema_next (const int32_t held, const int32_t target, const int32_t decay,
              bool &saturated_out)
{
    const int32_t move = 32768 - 1 - decay;
    const int32_t step = ref_clamp_q15 (
        ref_round_shift (static_cast<int64_t> (target - held) * move, 15u));
    const int64_t next = static_cast<int64_t> (held) + step;
    saturated_out = next > 32767 || next < -32768;
    return ref_clamp_q15 (next);
}

/* The decay^n bias correction, re-implemented from the documented loop. */
std::vector<int16_t>
ref_bias_corrected (const std::vector<int16_t>& lanes, const int32_t decay,
                    const uint32_t updates)
{
    if (updates == 0u)
        return lanes;
    int32_t power = decay;
    int32_t one_minus = 32768 - 1 - decay;
    for (uint32_t update = 1u; update < updates; ++update)
    {
        power = ref_q15_mul (power, decay);
        one_minus = 32768 - 1 - power;
    }
    if (one_minus <= 0)
        return lanes;
    const int32_t reciprocal = ref_q15_reciprocal (one_minus);
    std::vector<int16_t> out;
    for (const int16_t lane : lanes)
        out.push_back (static_cast<int16_t> (
            ref_clamp_q15 (ref_q15_mul (lane, reciprocal))));
    return out;
}

/* Independent accept/reject trace of the metric gate. */
struct GateTrace
{
    std::vector<uint32_t> _M_accepted;
    uint32_t _M_rejected = 0u;
    uint32_t _M_compared = 0u;
    int32_t _M_best = 0;
};

GateTrace
ref_gate (const std::vector<int32_t>& metrics, const int32_t min_delta,
          const bool higher_is_better)
{
    GateTrace trace;
    bool has_best = false;
    for (size_t i = 0; i < metrics.size (); ++i)
    {
        ++trace._M_compared;
        if (!has_best)
        {
            has_best = true;
            trace._M_best = metrics[i];
            trace._M_accepted.push_back (static_cast<uint32_t> (i));
            continue;
        }
        const int64_t margin = higher_is_better
                                   ? static_cast<int64_t> (metrics[i])
                                                     - trace._M_best
                                   : static_cast<int64_t> (trace._M_best)
                                                     - metrics[i];
        if (margin >= min_delta)
        {
            trace._M_best = metrics[i];
            trace._M_accepted.push_back (static_cast<uint32_t> (i));
        }
        else
            ++trace._M_rejected;
    }
    return trace;
}

/* Per-snapshot sign-flip counts against an average stream: a lane diverges only
   when both sides are non-zero and their signs disagree. */
std::vector<uint32_t>
ref_divergence (const std::vector<int8_t>& average,
                const std::vector<std::vector<int8_t>>& snapshots)
{
    std::vector<uint32_t> counts;
    for (const std::vector<int8_t>& snapshot : snapshots)
    {
        uint32_t diverging = 0u;
        for (size_t lane = 0; lane < snapshot.size (); ++lane)
        {
            const int32_t a = average[lane];
            const int32_t s = snapshot[lane];
            if ((a > 0 && s < 0) || (a < 0 && s > 0))
                ++diverging;
        }
        counts.push_back (diverging);
    }
    return counts;
}

/* Lanes an average differs from its source, and the largest code-step gap. */
std::pair<uint32_t, int32_t>
ref_drift (const std::vector<int8_t>& average, const std::vector<int8_t>& source)
{
    uint32_t drift = 0u;
    int32_t maximum = 0;
    for (size_t lane = 0; lane < average.size (); ++lane)
    {
        const int32_t a = average[lane];
        const int32_t s = source[lane];
        const int32_t difference = a > s ? a - s : s - a;
        if (difference != 0)
            ++drift;
        if (difference > maximum)
            maximum = difference;
    }
    return { drift, maximum };
}

/* --------------------------------------------------------------------------
 * Distillation float references, continued.
 * -------------------------------------------------------------------------- */

/* log p of the same softened softmax the Q7 domain approximates. */
std::vector<double>
ref_log_probs (const std::vector<int8_t>& codes, const int32_t inv_t_q15)
{
    std::vector<double> logs;
    for (const double p : ref_softmax (codes, inv_t_q15))
        logs.push_back (std::log (p));
    return logs;
}

/* KL(teacher || student) in nats from the float softmaxes. */
double
ref_kl_real (const std::vector<int8_t>& student,
             const std::vector<int8_t>& teacher, const int32_t inv_t_q15)
{
    return ref_kl (ref_softmax (teacher, inv_t_q15),
                   ref_softmax (student, inv_t_q15));
}

/* One class of the documented KL backward chain, student side. */
constexpr int32_t
ref_kl_student_lane (const int32_t p_s, const int32_t p_t, const int32_t inv_t,
                      const int32_t seed)
{
    return ref_q15_mul (
        ref_q15_mul (ref_clamp_q15 (static_cast<int64_t> (p_s) - p_t), inv_t),
        seed);
}
} // namespace

TEST_CASE ("prune POD layouts and defaults are the documented ones")
{
    STATIC_REQUIRE (sizeof (PruneConfig) == 32u);
    STATIC_REQUIRE (sizeof (PruneHistogram) == 516u);
    STATIC_REQUIRE (sizeof (PruneReport) == 40u);
    STATIC_REQUIRE (sizeof (PruneRunReport) == 36u);
    STATIC_REQUIRE (sizeof (PruneRowGeom) == 24u);
    STATIC_REQUIRE (sizeof (DimRewrite) == 12u);

    PruneConfig config;
    CHECK (config._M_target_sparsity_q15 == 0);
    CHECK (config._M_pattern == static_cast<uint8_t> (PrunePattern::Unstructured));
    CHECK (config._M_block_n == 2u);
    CHECK (config._M_block_m == 4u);
    CHECK (config._M_schedule == static_cast<uint8_t> (SparsitySchedule::Constant));
    CHECK (config.validate () == Status::OK);
    PruneHistogram histogram;
    histogram.reset (); /* the POD has no default initialiser: zero it first */
    histogram.add (3u);
    histogram.add (128u); /* both saturate into the top bin */
    REQUIRE (histogram._M_total == 2u);
    histogram.reset ();
    uint32_t live_bins = 0u;
    for (const uint32_t bin : histogram._M_bins)
        live_bins += bin;
    CHECK (live_bins == 0u);
    CHECK (histogram._M_total == 0u);
    CHECK (PruneReport {}._M_pruned == 0u);
    CHECK (PruneRunReport {}._M_achieved_sparsity_q15 == 0);
    CHECK (DimRewrite {}._M_valid == 0u);
    StructuredPlan plan;
    CHECK (plan._M_keep == nullptr);
    CHECK (plan._M_kept_count == 0u);
}

TEST_CASE ("PruneConfig validate accepts the documented grid")
{
    SECTION ("unstructured and schedules")
    {
        for (const uint8_t schedule : { static_cast<uint8_t> (SparsitySchedule::Constant),
                                        static_cast<uint8_t> (SparsitySchedule::Linear),
                                        static_cast<uint8_t> (SparsitySchedule::ThreeStage) })
        {
            PruneConfig c;
            c._M_schedule = schedule;
            c._M_ramp_end_step = 10u;
            c._M_ramp_start_step = 2u;
            CHECK (c.validate () == Status::OK);
        }
    }
    SECTION ("NinM only accepts 2:4 and 4:8")
    {
        PruneConfig c;
        c._M_pattern = static_cast<uint8_t> (PrunePattern::NinM);
        c._M_block_n = 2u;
        c._M_block_m = 4u;
        CHECK (c.validate () == Status::OK);
        c._M_block_n = 4u;
        c._M_block_m = 8u;
        CHECK (c.validate () == Status::OK);
        c._M_block_n = 1u;
        c._M_block_m = 2u;
        CHECK (c.validate () == Status::UNSUPPORTED);
        c._M_block_n = 3u;
        c._M_block_m = 6u;
        CHECK (c.validate () == Status::UNSUPPORTED);
        c._M_block_n = 5u;
        c._M_block_m = 4u;
        CHECK (c.validate () == Status::UNSUPPORTED);
    }
    SECTION ("range and structure errors")
    {
        PruneConfig c;
        c._M_target_sparsity_q15 = 32768; /* > 32767 ceiling */
        CHECK (c.validate () == Status::OUT_OF_RANGE);
        c._M_target_sparsity_q15 = -1;
        CHECK (c.validate () == Status::OUT_OF_RANGE);
        c = PruneConfig {};
        c._M_final_sparsity_q15 = 32768;
        CHECK (c.validate () == Status::OUT_OF_RANGE);
        c = PruneConfig {};
        c._M_pattern = 3;
        CHECK (c.validate () == Status::BAD_LAYOUT);
        c._M_pattern = 0;
        c._M_schedule = 3;
        CHECK (c.validate () == Status::BAD_LAYOUT);
        c = PruneConfig {};
        c._M_axis = 4;
        CHECK (c.validate () == Status::BAD_LAYOUT);
        c._M_axis = 3;
        c._M_schedule = static_cast<uint8_t> (SparsitySchedule::Linear);
        c._M_ramp_end_step = 0u;
        CHECK (c.validate () == Status::BAD_LAYOUT);
        c._M_ramp_end_step = 5u;
        c._M_ramp_start_step = 6u;
        c._M_schedule = static_cast<uint8_t> (SparsitySchedule::ThreeStage);
        CHECK (c.validate () == Status::BAD_LAYOUT);
    }
    SECTION ("32767 sparsity is the allowed ceiling")
    {
        PruneConfig c;
        c._M_target_sparsity_q15 = 32767;
        CHECK (c.validate () == Status::OK);
    }
}

TEST_CASE ("prune_magnitude maps the whole code grid to |code| classes")
{
    SECTION ("every code in [-127, 127]")
    {
        for (int32_t code = -127; code <= 127; ++code)
        {
            const uint32_t expected = static_cast<uint32_t> (code < 0 ? -code : code);
            CHECK (prune_magnitude (code) == expected);
        }
    }
    SECTION ("out-of-grid inputs clamp to 127")
    {
        CHECK (prune_magnitude (-128) == 127u);
        CHECK (prune_magnitude (200) == 127u);
        CHECK (prune_magnitude (-1000) == 127u);
        CHECK (prune_magnitude (0) == 0u);
    }
    SECTION ("magnitude ranking is the Q15 ranking: step 258 is monotone")
    {
        /* Widening is code * 258, so one code step is exactly one 258-unit Q15
           step: the |value| ranking is the magnitude-class ranking. */
        for (int32_t code = -127; code < 127; ++code)
        {
            const int32_t next = code + 1;
            CHECK (ref_code_to_q15 (next) - ref_code_to_q15 (code) == 258);
            if (code < 0 && next <= 0)
            {
                /* negative codes climb toward zero, |q15| shrinks */
                CHECK (ref_code_to_q15 (next) >= ref_code_to_q15 (code));
                CHECK (-ref_code_to_q15 (next) <= -ref_code_to_q15 (code));
            }
            else if (code >= 0)
                CHECK (ref_code_to_q15 (next) >= ref_code_to_q15 (code));
            else /* code == -1 -> next == 0 crosses the sign boundary */
                CHECK (ref_code_to_q15 (next) >= ref_code_to_q15 (code));
        }
        /* Strictly increasing away from zero on both signs, class by class. */
        for (uint32_t mag = 0u; mag < 127u; ++mag)
        {
            const int32_t c = static_cast<int32_t> (mag);
            CHECK (ref_code_to_q15 (c + 1) > ref_code_to_q15 (c));
            CHECK (ref_code_to_q15 (-c - 1) < ref_code_to_q15 (-c));
            CHECK (prune_magnitude (c + 1) == mag + 1u);
        }
        /* The two rankings agree lane for lane over the whole grid. */
        const auto q15_abs = [](const int64_t v) -> int64_t {
            return v < 0 ? -v : v;
        };
        for (int32_t code = -127; code < 127; ++code)
        {
            const bool q15_grows
                = q15_abs (ref_code_to_q15 (code + 1))
                  >= q15_abs (ref_code_to_q15 (code));
            const bool class_grows
                = prune_magnitude (code + 1) >= prune_magnitude (code);
            CHECK (q15_grows == class_grows);
        }
    }
}

TEST_CASE ("PruneHistogram counts bins and saturates bin 127")
{
    PruneHistogram h;
    h.reset ();
    for (uint32_t bin = 0; bin < 128u; ++bin)
        CHECK (h._M_bins[bin] == 0u);
    CHECK (h._M_total == 0u);

    h.add (0u);
    h.add (5u);
    h.add (5u);
    h.add (127u);
    h.add (128u); /* clamped class */
    h.add (5000u); /* clamped class */
    CHECK (h._M_bins[0] == 1u);
    CHECK (h._M_bins[5] == 2u);
    CHECK (h._M_bins[127] == 3u);
    CHECK (h._M_total == 6u);

    SECTION ("reset clears everything")
    {
        h.reset ();
        CHECK (h._M_bins[5] == 0u);
        CHECK (h._M_total == 0u);
    }
    SECTION ("histogram of a code stream matches the independent counts")
    {
        const std::vector<int8_t> codes = { 5, -3, -5, 2, 7, 0, -1, 4,
                                            4, -4, 7, -7, 0, 12, 3, -2 };
        PruneHistogram hs;
        CHECK (prune_histogram_of_codes (hs, codes.data (),
                                         static_cast<uint32_t> (codes.size ()))
              == Status::OK);
        CHECK (hs._M_total == 16u);
        std::vector<uint32_t> expected (128u, 0u);
        for (const int8_t c : codes)
            ++expected[ref_magnitude (c)];
        for (uint32_t bin = 0; bin < 128u; ++bin)
            CHECK (hs._M_bins[bin] == expected[bin]);
    }
    SECTION ("input validation")
    {
        CHECK (prune_histogram_of_codes (h, nullptr, 4u) == Status::BAD_LAYOUT);
        CHECK (prune_histogram_of_codes (h, reinterpret_cast<const int8_t*> (1),
                                         _S_prune_max_count + 1u)
              == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("prune_sparsity_at_q15 implements the documented schedules")
{
    SECTION ("constant is the target at every step")
    {
        PruneConfig c;
        c._M_target_sparsity_q15 = 8192;
        for (const uint32_t step : { 0u, 1u, 7u, 1000u })
            CHECK (prune_sparsity_at_q15 (c, step) == 8192);
    }

    SECTION ("linear ramps from initial at 0 to final at ramp_end")
    {
        /* Documented: initial + round-half-up((final-initial)*(step)/(span)). */
        struct Case
        {
            int32_t from;
            int32_t to;
            uint32_t end;
        };
        const Case cases[] = { { 0, 16384, 4u }, { 0, 8192, 3u },
                               { 4000, 12000, 7u }, { 16384, 0, 4u },
                               { 12000, 3000, 5u }, { 0, 32767, 6u } };
        for (const Case& k : cases)
        {
            PruneConfig c;
            c._M_schedule = static_cast<uint8_t> (SparsitySchedule::Linear);
            c._M_initial_sparsity_q15 = k.from;
            c._M_final_sparsity_q15 = k.to;
            c._M_ramp_end_step = k.end;
            const int64_t span = k.end;
            int32_t previous = prune_sparsity_at_q15 (c, 0u);
            CHECK (previous == k.from);
            for (uint32_t step = 0u; step <= k.end + 2u; ++step)
            {
                const int64_t num
                    = (static_cast<int64_t> (k.to) - k.from)
                      * static_cast<int64_t> (step);
                int64_t expected;
                if (step == 0u)
                    expected = k.from;
                else if (static_cast<int64_t> (step) >= span)
                    expected = k.to;
                else
                {
                    const int64_t mag = num < 0 ? -num : num;
                    const int64_t q = (mag + span / 2) / span;
                    expected = k.from + (num < 0 ? -q : q);
                }
                const int32_t got = prune_sparsity_at_q15 (c, step);
                CHECK (got == ref_clamp_q15 (expected));
                if (k.from <= k.to)
                    CHECK (got >= previous);
                else
                    CHECK (got <= previous);
                previous = got;
            }
            CHECK (prune_sparsity_at_q15 (c, k.end) == k.to);
        }
    }

    SECTION ("three stage plateaus at both ends")
    {
        PruneConfig c;
        c._M_schedule = static_cast<uint8_t> (SparsitySchedule::ThreeStage);
        c._M_initial_sparsity_q15 = 2048;
        c._M_final_sparsity_q15 = 16384;
        c._M_ramp_start_step = 10u;
        c._M_ramp_end_step = 20u;
        for (uint32_t step = 0u; step <= 10u; ++step)
            CHECK (prune_sparsity_at_q15 (c, step) == 2048);
        /* Documented: initial + round-half-up((final-initial)*(step-start)
           /(end-start)) = 2048 + (14336*5 + 5)/10 = 2048 + 7168. */
        CHECK (prune_sparsity_at_q15 (c, 15u) == 9216);
        CHECK (prune_sparsity_at_q15 (c, 15u)
               == 2048 + (14336 * 5 + 5) / 10);
        const int32_t mid = prune_sparsity_at_q15 (c, 15u);
        CHECK (mid > 2048);
        CHECK (mid < 16384);
        /* The ramp halves exactly at the quarter points of its own span. */
        CHECK (prune_sparsity_at_q15 (c, 12u) == 2048 + (14336 * 2 + 5) / 10);
        CHECK (prune_sparsity_at_q15 (c, 18u) == 2048 + (14336 * 8 + 5) / 10);
        /* Monotone across the whole schedule, including the two plateaus. */
        int32_t previous = prune_sparsity_at_q15 (c, 0u);
        for (uint32_t step = 1u; step <= 40u; ++step)
        {
            const int32_t got = prune_sparsity_at_q15 (c, step);
            INFO ("step=" << step);
            CHECK (got >= previous);
            CHECK (got >= 2048);
            CHECK (got <= 16384);
            previous = got;
        }
        for (uint32_t step = 20u; step <= 30u; ++step)
            CHECK (prune_sparsity_at_q15 (c, step) == 16384);
    }

    SECTION ("ramp reaches its exact target at the documented step")
    {
        PruneConfig c;
        c._M_schedule = static_cast<uint8_t> (SparsitySchedule::Linear);
        c._M_initial_sparsity_q15 = 0;
        c._M_final_sparsity_q15 = 8192;
        c._M_ramp_end_step = 8u;
        CHECK (prune_sparsity_at_q15 (c, 4u) == 4096);
        CHECK (prune_sparsity_at_q15 (c, 3u) == 3072);
        CHECK (prune_sparsity_at_q15 (c, 8u) == 8192);
        CHECK (prune_sparsity_at_q15 (c, 9u) == 8192);
    }

    SECTION ("ramp_end 0 is treated as span 1 by the pure function")
    {
        PruneConfig c;
        c._M_schedule = static_cast<uint8_t> (SparsitySchedule::Linear);
        c._M_initial_sparsity_q15 = 100;
        c._M_final_sparsity_q15 = 200;
        c._M_ramp_end_step = 0u;
        CHECK (prune_sparsity_at_q15 (c, 0u) == 100);
        CHECK (prune_sparsity_at_q15 (c, 1u) == 200);
        CHECK (prune_sparsity_at_q15 (c, 2u) == 200);
    }
}

TEST_CASE ("prune_target_pruned floors the request as documented")
{
    struct Case
    {
        uint32_t count;
        int32_t sparsity;
        uint32_t expected;
    };
    const Case cases[] = {
        { 16u, 16384, 8u },   { 16u, 0, 0u },       { 16u, -5, 0u },
        { 8u, 32767, 7u },    { 7u, 8192, 1u },     { 100u, 12000, 36u },
        { 0u, 16384, 0u },    { 3u, 32767, 2u },    { 1u, 32767, 0u },
        { 256u, 4096, 32u },  { 255u, 4096, 31u },  { 1024u, 16384, 512u },
        { 65535u, 32767, 65533u },
    };
    for (const Case& k : cases)
    {
        const uint64_t expected = k.count == 0u || k.sparsity <= 0
                                      ? 0u
                                      : std::min (static_cast<uint64_t> (k.count),
                                                  (static_cast<uint64_t> (k.count)
                                                   * static_cast<uint64_t> (k.sparsity))
                                                        / 32768u);
        CHECK (prune_target_pruned (k.count, k.sparsity)
               == static_cast<uint32_t> (expected));
        CHECK (prune_target_pruned (k.count, k.sparsity) == k.expected);
    }
}

TEST_CASE ("unstructured prune hits the exact sparsity and the documented ties")
{
    struct Tensor
    {
        std::vector<int8_t> codes;
    };
    const std::vector<Tensor> tensors = {
        { { 5, -3, -5, 2, 7, 0, -1, 4, 4, -4, 7, -7, 0, 12, 3, -2 } },
        { { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 } },
        { { -100, 100, -100, 100, 1, -1, 1, -1, 50, -50, 50, -50 } },
        { { 0, 1, -1, 2, -2, 0, 3, -3, 127, -127, 126, -126, 127 } },
    };
    const int32_t sparsities[] = { 1638, 8192, 16384, 24576, 27306, 32767 };

    for (const Tensor& t : tensors)
    {
        for (const int32_t sparsity : sparsities)
        {
            W w;
            w.fill (t.codes);
            Trainable weight = view (w, Shape::vector (static_cast<uint16_t> (w.n)));
            alignas (16) uint8_t bits[64] = { 0u };
            PruneMask mask = make_mask (bits, w.n);
            PruneHistogram hist;
            PruneReport report;

            PruneConfig c;
            c._M_target_sparsity_q15 = sparsity;

            REQUIRE (prune_apply (c, 0u, weight, hist, mask, report)
                     == Status::OK);

            const uint32_t count = w.n;
            const uint32_t target = static_cast<uint32_t> (
                static_cast<uint64_t> (count) * static_cast<uint64_t> (sparsity)
                / 32768u);
            const uint32_t keep = count - target;
            const SelRef ref = ref_select_codes (t.codes, keep);

            INFO ("count=" << count << " sparsity=" << sparsity);

            /* Exact request honoured, to one-lane granularity. */
            CHECK (report._M_elements == count);
            CHECK (report._M_target_pruned == target);
            CHECK (report._M_pruned == target);
            CHECK (mask._M_pruned == ref.pruned);
            CHECK (report._M_requested_sparsity_q15 == sparsity);

            /* The mask matches the independent selection byte for byte. */
            for (uint32_t i = 0u; i < count; ++i)
                CHECK (bits[i] == ref.mask[i]);

            /* Threshold / tie group / tie kept per the documented walk. */
            CHECK (static_cast<uint32_t> (report._M_threshold_magnitude)
                   == ref.threshold);
            CHECK (report._M_tie_group == ref.tie_group);
            CHECK (report._M_tie_kept == ref.tie_kept);

            /* Forward applied: pruned lanes are zero, kept lanes untouched. */
            for (uint32_t i = 0u; i < count; ++i)
            {
                if (ref.mask[i] != 0u)
                    CHECK (w.code[i] == t.codes[i]);
                else
                    CHECK (w.code[i] == 0);
            }

            /* Achieved sparsity: the mask ratio, verified independently. */
            CHECK (report._M_achieved_sparsity_q15 == mask.sparsity_q15 ());
            CHECK (mask.sparsity_q15 ()
                   == ref_ratio_q15 (static_cast<int64_t> (ref.pruned),
                                     static_cast<int64_t> (count)));
            const double approx = static_cast<double> (ref.pruned)
                                  / static_cast<double> (count) * 32768.0;
            CHECK (std::fabs (report._M_achieved_sparsity_q15 - approx) < 1.0);
        }
    }
}

TEST_CASE ("unstructured tie-break goes to the lowest indices")
{
    SECTION ("all-tied tensor keeps the first keep lanes")
    {
        const std::vector<int8_t> codes = { 6, -6, 6, -6, 6, 6, -6, 6 };
        W w;
        w.fill (codes);
        Trainable weight = view (w, Shape::vector (8u));
        alignas (16) uint8_t bits[16] = { 0u };
        PruneMask mask = make_mask (bits, 8u);
        PruneHistogram hist;
        uint32_t threshold = 99u;
        uint32_t tie_group = 99u;
        uint32_t tie_kept = 99u;
        REQUIRE (prune_mask_unstructured (w.code, 8u, 3u, hist, mask, threshold,
                                          tie_group, tie_kept)
                 == Status::OK);
        const uint8_t expected[8] = { 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u };
        for (uint32_t i = 0u; i < 8u; ++i)
            CHECK (bits[i] == expected[i]);
        CHECK (threshold == 6u);
        CHECK (tie_group == 8u);
        CHECK (tie_kept == 3u);
        CHECK (mask._M_pruned == 5u);
        CHECK (mask._M_valid == 1u);
        CHECK (mask._M_pattern == static_cast<uint8_t> (PrunePattern::Unstructured));

        /* The survivor lanes are the ones a forward pass must still see. */
        REQUIRE (prune_apply_forward (mask, weight._M_code) == Status::OK);
        const int8_t kept_codes[8] = { 6, -6, 6, 0, 0, 0, 0, 0 };
        for (uint32_t i = 0u; i < 8u; ++i)
            CHECK (w.code[i] == kept_codes[i]);
    }

    SECTION ("all-zero tensor keeps the first keep lanes at threshold 0")
    {
        const std::vector<int8_t> codes (8, 0);
        W w;
        w.fill (codes);
        alignas (16) uint8_t bits[16] = { 0u };
        PruneMask mask = make_mask (bits, 8u);
        PruneHistogram hist;
        uint32_t threshold = 99u;
        uint32_t tie_group = 99u;
        uint32_t tie_kept = 99u;
        REQUIRE (prune_mask_unstructured (w.code, 8u, 2u, hist, mask, threshold,
                                          tie_group, tie_kept)
                 == Status::OK);
        CHECK (threshold == 0u);
        CHECK (tie_group == 8u);
        CHECK (tie_kept == 2u);
        const uint8_t expected[8] = { 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u };
        for (uint32_t i = 0u; i < 8u; ++i)
            CHECK (bits[i] == expected[i]);
    }

    SECTION ("keep 0 prunes everything, keep all prunes nothing")
    {
        const std::vector<int8_t> codes = { 3, -1, 0, 7, -7 };
        W w;
        w.fill (codes);
        alignas (16) uint8_t bits[16] = { 0u };
        PruneMask mask = make_mask (bits, 5u);
        PruneHistogram hist;
        uint32_t threshold = 99u;
        uint32_t tie_group = 99u;
        uint32_t tie_kept = 99u;
        REQUIRE (prune_mask_unstructured (w.code, 5u, 0u, hist, mask, threshold,
                                          tie_group, tie_kept)
                 == Status::OK);
        CHECK (threshold == 127u); /* the documented top-bin corner */
        CHECK (tie_kept == 0u);
        CHECK (mask._M_pruned == 5u);

        REQUIRE (prune_mask_unstructured (w.code, 5u, 5u, hist, mask, threshold,
                                          tie_group, tie_kept)
                 == Status::OK);
        CHECK (mask._M_pruned == 0u);
        for (uint32_t i = 0u; i < 5u; ++i)
            CHECK (bits[i] == 1u);
    }

    SECTION ("rerun over identical weights reproduces the mask byte for byte")
    {
        const std::vector<int8_t> codes = { 5, -5, 1, -1, 5, 5, -2, 2 };
        uint8_t first[8] = { 0u };
        uint8_t second[8] = { 0u };
        for (int round = 0; round < 2; ++round)
        {
            W w;
            w.fill (codes);
            alignas (16) uint8_t bits[16] = { 0u };
            PruneMask mask = make_mask (bits, 8u);
            PruneHistogram hist;
            uint32_t t = 0u;
            uint32_t g = 0u;
            uint32_t k = 0u;
            REQUIRE (prune_mask_unstructured (w.code, 8u, 4u, hist, mask, t, g, k)
                     == Status::OK);
            uint8_t* out = round == 0 ? first : second;
            for (uint32_t i = 0u; i < 8u; ++i)
                out[i] = bits[i];
        }
        for (uint32_t i = 0u; i < 8u; ++i)
            CHECK (first[i] == second[i]);
    }

    SECTION ("selection input validation")
    {
        W w;
        w.fill (std::vector<int8_t> { 1, 2, 3 });
        alignas (16) uint8_t bits[4] = { 0u };
        PruneMask mask = make_mask (bits, 2u); /* count mismatch */
        PruneHistogram hist;
        uint32_t t = 0u;
        uint32_t g = 0u;
        uint32_t k = 0u;
        CHECK (prune_mask_unstructured (w.code, 3u, 2u, hist, mask, t, g, k)
               == Status::BAD_LAYOUT);
        CHECK (prune_mask_unstructured (nullptr, 3u, 2u, hist, mask, t, g, k)
               == Status::BAD_LAYOUT);
        static uint8_t unaligned_storage[8];
        PruneMask bad = make_mask (unaligned_storage + 1, 3u);
        CHECK (prune_mask_unstructured (w.code, 3u, 2u, hist, bad, t, g, k)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("prune_select_threshold walks bins exactly")
{
    const std::vector<int8_t> codes = { 5, -3, -5, 2, 7, 0, -1, 4,
                                        4, -4, 7, -7, 0, 12, 3, -2 };
    PruneHistogram hist;
    REQUIRE (prune_histogram_of_codes (hist, codes.data (), 16u) == Status::OK);

    struct Case
    {
        uint32_t keep;
        uint32_t threshold;
        uint32_t tie_keep;
    };
    /* Hand-walked over the bin counts of `codes`
       (0:2, 1:1, 2:2, 3:2, 4:3, 5:2, 7:3, 12:1): the first class t, from the
       top down, with count(>t) <= keep <= count(>=t) is the threshold and the
       tie budget is keep - count(>t). */
    const Case cases[] = { { 16u, 0u, 2u },  { 15u, 0u, 1u },  { 14u, 1u, 1u },
                           { 13u, 2u, 2u },  { 12u, 2u, 1u },  { 11u, 3u, 2u },
                           { 10u, 3u, 1u },  { 9u, 4u, 3u },   { 8u, 4u, 2u },
                           { 7u, 4u, 1u },   { 6u, 5u, 2u },   { 5u, 5u, 1u },
                           { 4u, 7u, 3u },   { 3u, 7u, 2u },   { 2u, 7u, 1u },
                           { 1u, 12u, 1u },  { 0u, 127u, 0u } };
    for (const Case& k : cases)
    {
        uint32_t threshold = 99u;
        uint32_t tie_keep = 99u;
        const SelRef ref = ref_select_codes (codes, k.keep);
        INFO ("keep=" << k.keep);
        REQUIRE (prune_select_threshold (hist, k.keep, threshold, tie_keep)
                 == Status::OK);
        CHECK (threshold == ref.threshold);
        CHECK (tie_keep == ref.tie_kept);
        CHECK (threshold == k.threshold);
        CHECK (tie_keep == k.tie_keep);
    }

    SECTION ("keep above total is out of range")
    {
        uint32_t threshold = 0u;
        uint32_t tie_keep = 0u;
        CHECK (prune_select_threshold (hist, 17u, threshold, tie_keep)
               == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("prune_apply zero-request still yields an all-keep valid mask")
{
    W w;
    w.fill (std::vector<int8_t> { 4, -4, 2, 0, 9 });
    const std::vector<int8_t> before = w.codes ();
    Trainable weight = view (w, Shape::vector (5u));
    alignas (16) uint8_t bits[8] = { 0u };
    PruneMask mask = make_mask (bits, 5u);
    PruneHistogram hist;
    PruneReport report;
    PruneConfig c; /* target 0 */
    REQUIRE (prune_apply (c, 123u, weight, hist, mask, report) == Status::OK);
    CHECK (mask._M_valid == 1u);
    for (uint32_t i = 0u; i < 5u; ++i)
        CHECK (bits[i] == 1u);
    CHECK (report._M_pruned == 0u);
    CHECK (report._M_achieved_sparsity_q15 == 0);
    CHECK (report._M_requested_sparsity_q15 == 0);
    CHECK (w.codes () == before);
    /* The mask survives so the backward hook can read it on every step. */
    CHECK (mask.validate () == Status::OK);
}

TEST_CASE ("NinM pruning removes exactly the documented lanes")
{
    SECTION ("2:4 keeps the two largest per aligned block")
    {
        const std::vector<int8_t> codes = { 1, 9, 2, 8,   /* block 0: keep 9,8 */
                                           -3, -10, 4, 3, /* block 1: keep -10,4 */
                                           0, 0, 0, 7,   /* block 2: keep 7, tie at 0 */
                                           5, -5 };      /* tail of 2: keep both */
        W w;
        w.fill (codes);
        alignas (16) uint8_t bits[16] = { 0u };
        PruneMask mask = make_mask (bits, 14u);
        uint32_t blocks = 0u;
        uint32_t partial = 0u;
        REQUIRE (prune_mask_ninm (w.code, 14u, 2u, 4u, mask, blocks, partial)
                 == Status::OK);
        const NmRef ref = ref_ninm (codes, 2u, 4u);
        for (uint32_t i = 0u; i < 14u; ++i)
            CHECK (bits[i] == ref.mask[i]);
        CHECK (blocks == 4u);
        CHECK (partial == 1u);
        CHECK (mask._M_pruned == ref.pruned);
        CHECK (mask._M_pattern == static_cast<uint8_t> (PrunePattern::NinM));
        const uint8_t expected[14] = { 0u, 1u, 0u, 1u,  0u, 1u, 1u, 0u,
                                       1u, 0u, 0u, 1u,  1u, 1u };
        for (uint32_t i = 0u; i < 14u; ++i)
            CHECK (bits[i] == expected[i]);
    }

    SECTION ("4:8 keeps four per block of eight")
    {
        const std::vector<int8_t> codes = { 1, 2, 3, 4, 5, 6, 7, 8,
                                           8, 7, 6, 5, 4, 3, 2, 1,
                                           -9, -1, -9, -1, -9, -1, -9, -1 };
        W w;
        w.fill (codes);
        alignas (16) uint8_t bits[32] = { 0u };
        PruneMask mask = make_mask (bits, 24u);
        uint32_t blocks = 0u;
        uint32_t partial = 0u;
        REQUIRE (prune_mask_ninm (w.code, 24u, 4u, 8u, mask, blocks, partial)
                 == Status::OK);
        const NmRef ref = ref_ninm (codes, 4u, 8u);
        for (uint32_t i = 0u; i < 24u; ++i)
            CHECK (bits[i] == ref.mask[i]);
        CHECK (blocks == 3u);
        CHECK (partial == 0u);
        CHECK (mask._M_pruned == 12u);
        /* First block keeps lanes 4..7, second keeps 8..11 (ties go low). */
        const uint8_t expected[24] = { 0u, 0u, 0u, 0u, 1u, 1u, 1u, 1u,
                                       1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u,
                                       1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u };
        for (uint32_t i = 0u; i < 24u; ++i)
            CHECK (bits[i] == expected[i]);
    }

    SECTION ("prune_apply NinM reports blocks and exact per-block sparsity")
    {
        const std::vector<int8_t> codes = { 4, 1, 3, 2, 8, 7, 6, 5 };
        W w;
        w.fill (codes);
        Trainable weight = view (w, Shape::vector (8u));
        alignas (16) uint8_t bits[16] = { 0u };
        PruneMask mask = make_mask (bits, 8u);
        PruneHistogram hist;
        PruneReport report;
        PruneConfig c;
        c._M_pattern = static_cast<uint8_t> (PrunePattern::NinM);
        c._M_target_sparsity_q15 = 16384; /* ignored by the block rule */
        REQUIRE (prune_apply (c, 0u, weight, hist, mask, report) == Status::OK);
        const NmRef ref = ref_ninm (codes, 2u, 4u);
        for (uint32_t i = 0u; i < 8u; ++i)
            CHECK (bits[i] == ref.mask[i]);
        CHECK (report._M_blocks_total == 2u);
        CHECK (report._M_blocks_partial == 0u);
        CHECK (report._M_pruned == ref.pruned);
        CHECK (report._M_achieved_sparsity_q15 == mask.sparsity_q15 ());
        for (uint32_t i = 0u; i < 8u; ++i)
            CHECK (w.code[i] == (ref.mask[i] != 0u ? codes[i] : int8_t { 0 }));
    }

    SECTION ("unsupported block shapes and validation")
    {
        W w;
        w.fill (std::vector<int8_t> { 1, 2, 3, 4 });
        alignas (16) uint8_t bits[8] = { 0u };
        PruneMask mask = make_mask (bits, 4u);
        uint32_t b = 0u;
        uint32_t p = 0u;
        CHECK (prune_mask_ninm (w.code, 4u, 0u, 4u, mask, b, p)
               == Status::UNSUPPORTED);
        CHECK (prune_mask_ninm (w.code, 4u, 2u, 2u, mask, b, p)
               == Status::UNSUPPORTED);
        CHECK (prune_mask_ninm (w.code, 4u, 2u, 9u, mask, b, p)
               == Status::UNSUPPORTED);
        CHECK (prune_mask_ninm (nullptr, 4u, 2u, 4u, mask, b, p)
               == Status::BAD_LAYOUT);
        mask._M_count = 3u;
        CHECK (prune_mask_ninm (w.code, 4u, 2u, 4u, mask, b, p)
               == Status::BAD_LAYOUT);
    }

    SECTION ("NinM is deterministic and idempotent on the code stream")
    {
        const std::vector<int8_t> codes = { 5, -5, 1, -1, 4, 4, -9, 0,
                                           2, -2, 7, 7, 1, 0, 1, -8 };
        uint8_t first[16] = { 0u };
        std::vector<int8_t> after_forward;
        {
            W w;
            w.fill (codes);
            alignas (16) uint8_t bits[16] = { 0u };
            PruneMask mask = make_mask (bits, 16u);
            uint32_t b = 0u;
            uint32_t p = 0u;
            REQUIRE (prune_mask_ninm (w.code, 16u, 2u, 4u, mask, b, p)
                     == Status::OK);
            for (uint32_t i = 0u; i < 16u; ++i)
                first[i] = bits[i];
            REQUIRE (prune_apply_forward (mask, w.code) == Status::OK);
            after_forward = w.codes ();
        }
        /* Re-pruning the zeroed stream keeps the same lanes alive. */
        W w;
        w.fill (after_forward);
        alignas (16) uint8_t bits[16] = { 0u };
        PruneMask mask = make_mask (bits, 16u);
        uint32_t b = 0u;
        uint32_t p = 0u;
        REQUIRE (prune_mask_ninm (w.code, 16u, 2u, 4u, mask, b, p) == Status::OK);
        REQUIRE (prune_apply_forward (mask, w.code) == Status::OK);
        for (uint32_t i = 0u; i < 16u; ++i)
            CHECK (bits[i] == first[i]);
        CHECK (w.codes () == after_forward);
    }
}

TEST_CASE ("pruning twice changes nothing (mask idempotence)")
{
    SECTION ("prune_apply twice on the same tensor leaves codes and counts fixed")
    {
        const std::vector<int8_t> codes = { 9, -1, 4, 4, -7, 0, 5, -3,
                                           12, -12, 2, -8, 7, 0, -5, 6 };
        W w;
        w.fill (codes);
        Trainable weight = view (w, Shape::vector (16u));
        alignas (16) uint8_t bits[16] = { 0u };
        PruneMask mask = make_mask (bits, 16u);
        PruneHistogram hist;
        PruneReport report;
        PruneConfig c;
        c._M_target_sparsity_q15 = 16384;
        REQUIRE (prune_apply (c, 0u, weight, hist, mask, report) == Status::OK);
        const std::vector<int8_t> once = w.codes ();
        const uint32_t pruned_once = report._M_pruned;
        const uint8_t mask_once[16] = { 0u };
        std::copy (bits, bits + 16, const_cast<uint8_t*> (mask_once));
        hist.reset ();
        REQUIRE (prune_apply (c, 0u, weight, hist, mask, report) == Status::OK);
        CHECK (w.codes () == once);
        CHECK (report._M_pruned == pruned_once);
        /* Every lane that went to zero stayed zero. */
        for (uint32_t i = 0u; i < 16u; ++i)
            if (mask_once[i] == 0u)
                CHECK (w.code[i] == 0);
    }

    SECTION ("enforcing the same mask twice is a no-op")
    {
        W w;
        w.fill (std::vector<int8_t> { 1, -2, 3, -4, 5, -6, 7, -8 });
        w.grad[3] = 111;
        w.grad[6] = 222;
        Trainable weight = view (w, Shape::vector (8u));
        alignas (16) uint8_t bits[8] = { 0u };
        PruneMask mask = make_mask (bits, 8u);
        const uint8_t pattern[8] = { 1u, 0u, 1u, 0u, 1u, 1u, 0u, 1u };
        std::copy (pattern, pattern + 8, bits);
        mask._M_pruned = 3u;
        mask._M_valid = 1u;
        REQUIRE (prune_enforce (mask, weight) == Status::OK);
        const std::vector<int8_t> codes_after = w.codes ();
        const std::vector<int32_t> grads_after = w.grads ();
        REQUIRE (prune_enforce (mask, weight) == Status::OK);
        CHECK (w.codes () == codes_after);
        CHECK (w.grads () == grads_after);
        CHECK (w.code[1] == 0);
        CHECK (w.code[3] == 0);
        CHECK (w.code[6] == 0);
        CHECK (w.grad[1] == 0);
        CHECK (w.grad[3] == 0);
        CHECK (w.grad[6] == 0);
        CHECK (w.grad[0] == 0);
        CHECK (w.grad[7] == 0);
    }
}

TEST_CASE ("PruneMask validate and sparsity_q15 follow the contract")
{
    alignas (16) uint8_t bits[16] = { 0u };
    PruneMask mask = make_mask (bits, 8u);
    CHECK (mask.validate () == Status::BAD_LAYOUT); /* no bits yet */
    mask._M_bits = nullptr;
    mask._M_count = 8u;
    mask._M_valid = 1u;
    CHECK (mask.validate () == Status::BAD_LAYOUT);

    mask = make_mask (bits, 8u);
    mask._M_valid = 1u;
    mask._M_pruned = 3u;
    CHECK (mask.validate () == Status::OK);
    CHECK (mask.sparsity_q15 () == ref_ratio_q15 (3, 8));
    CHECK (mask.sparsity_q15 ()
           == ((static_cast<int32_t> (3) << 15) / static_cast<int32_t> (8)));

    mask._M_pruned = 9u; /* more pruned than lanes */
    CHECK (mask.validate () == Status::CORRUPT_STREAM);

    mask._M_pruned = 8u;
    mask._M_count = 0u;
    /* An empty mask cannot have pruned lanes: the counter outranks the count. */
    CHECK (mask.validate () == Status::CORRUPT_STREAM);
    /* sparsity_q15 is the pure ratio and count 0 divides by 1. */
    CHECK (mask.sparsity_q15 () == ref_ratio_q15 (8, 1));
    CHECK (mask.sparsity_q15 () == 32767); /* ratio saturated to the Q15 max */
    mask._M_pruned = 0u;
    CHECK (mask.validate () == Status::OK);
    CHECK (mask.sparsity_q15 () == 0);

    mask._M_count = _S_prune_max_count + 1u;
    CHECK (mask.validate () == Status::OUT_OF_RANGE);

    mask._M_count = 4u;
    const uint8_t pattern[16] = { 1u, 0u, 1u, 1u, 0u, 0u, 1u, 0u,
                                  0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u };
    std::copy (pattern, pattern + 16, bits);
    CHECK (mask.keeps (0u));
    CHECK (!mask.keeps (1u));
    CHECK (mask.keeps (2u));
    CHECK (!mask.keeps (7u));
    CHECK (mask.keeps (15u) == (pattern[15] != 0u));
}

TEST_CASE ("mask pack / unpack / invert round trips bit-exactly")
{
    for (const uint32_t count : { 1u, 3u, 7u, 8u, 9u, 15u, 16u, 17u, 31u })
    {
        alignas (16) uint8_t bits[32];
        alignas (16) uint8_t packed[16];
        alignas (16) uint8_t out[32];
        /* Deterministic pattern: lane kept iff the low bits of i are not 0 or 3. */
        uint32_t pruned = 0u;
        for (uint32_t i = 0u; i < 32u; ++i)
        {
            bits[i] = (i % 4u == 0u || i % 4u == 3u) ? 0u : 1u;
            if (i < count && bits[i] == 0u)
                ++pruned;
        }
        PruneMask mask = make_mask (bits, count);
        mask._M_pruned = pruned;
        mask._M_valid = 1u;
        REQUIRE (mask.validate () == Status::OK);

        const uint32_t needed = (count + 7u) / 8u;
        REQUIRE (prune_mask_pack (mask, packed, needed - (needed > 0u ? 1u : 0u))
                 == Status::TRUNCATED);
        REQUIRE (prune_mask_pack (mask, packed, needed) == Status::OK);

        /* Independent packed form: bit i set iff lane i kept, low bit first. */
        for (uint32_t byte = 0u; byte < needed; ++byte)
        {
            uint8_t expected = 0u;
            for (uint32_t bit = 0u; bit < 8u; ++bit)
            {
                const uint32_t i = byte * 8u + bit;
                if (i < count && bits[i] != 0u)
                    expected = static_cast<uint8_t> (expected | (1u << bit));
            }
            INFO ("count=" << count << " byte=" << byte);
            CHECK (packed[byte] == expected);
        }
        for (uint32_t i = 0u; i < count; ++i)
            CHECK (prune_mask_packed_keeps (packed, i) == (bits[i] != 0u));
        CHECK (!prune_mask_packed_keeps (nullptr, 0u));

        /* Unpack rebuilds the byte mask and the pruned counter. */
        PruneMask restored;
        REQUIRE (prune_mask_unpack (packed, count, out, count, restored)
                 == Status::OK);
        CHECK (restored._M_count == count);
        CHECK (restored._M_pruned == pruned);
        CHECK (restored._M_valid == 1u);
        for (uint32_t i = 0u; i < count; ++i)
            CHECK (out[i] == bits[i]);

        /* Inverted mask: pruned lanes set, spare tail bits cleared. */
        alignas (16) uint8_t inverted[16];
        REQUIRE (prune_mask_pack_inverted (packed, count, inverted, needed)
                 == Status::OK);
        for (uint32_t i = 0u; i < count; ++i)
        {
            const bool set = (inverted[i >> 3] & (1u << (i & 7u))) != 0u;
            CHECK (set == (bits[i] == 0u));
        }
        if (needed > 0u)
        {
            const uint32_t spare = needed * 8u - count;
            if (spare != 0u)
            {
                const uint8_t tail = inverted[needed - 1u];
                for (uint32_t bit = 8u - spare; bit < 8u; ++bit)
                    CHECK ((tail & (1u << bit)) == 0u);
            }
        }
        CHECK (prune_mask_pack_inverted (packed, count, inverted, needed - 1u)
               == Status::BAD_LAYOUT);
        CHECK (prune_mask_pack_inverted (nullptr, count, inverted, needed)
               == Status::BAD_LAYOUT);
    }
    SECTION ("packing rejects a null destination")
    {
        alignas (16) uint8_t bits[4] = { 1u, 0u, 1u, 0u };
        PruneMask mask = make_mask (bits, 4u);
        mask._M_pruned = 2u;
        mask._M_valid = 1u;
        CHECK (prune_mask_pack (mask, nullptr, 1u) == Status::TRUNCATED);
        PruneMask invalid;
        CHECK (prune_mask_pack (invalid, reinterpret_cast<uint8_t*> (1), 1u)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("prune_apply_forward / backward / accumulate honour the mask")
{
    SECTION ("forward zeroes pruned codes only")
    {
        W w;
        w.fill (std::vector<int8_t> { 3, -4, 0, 5, -1, 2, 7, -8, 1, 1, 1, 1 });
        alignas (16) uint8_t bits[12] = { 0u };
        const uint8_t pattern[12] = { 1u, 0u, 1u, 0u, 1u, 1u, 0u, 0u,
                                      1u, 1u, 0u, 1u };
        std::copy (pattern, pattern + 12, bits);
        PruneMask mask = make_mask (bits, 12u);
        mask._M_pruned = 5u;
        mask._M_valid = 1u;
        REQUIRE (prune_apply_forward (mask, w.code) == Status::OK);
        for (uint32_t i = 0u; i < 12u; ++i)
            CHECK (w.code[i] == (pattern[i] != 0u ? int32_t { w.code[i] }
                                                  : int32_t { 0 }));
        CHECK (w.code[0] == 3);
        CHECK (w.code[1] == 0);
        CHECK (w.code[2] == 0);
        CHECK (w.code[4] == -1);
        CHECK (w.code[6] == 0);
        CHECK (w.code[11] == 1);
        CHECK (prune_apply_forward (mask, nullptr) == Status::BAD_LAYOUT);
    }

    SECTION ("backward zeroes pruned gradient lanes only")
    {
        W w;
        w.fill (std::vector<int8_t> (8, 0), 0);
        const int16_t g[8] = { 10, -20, 30, -40, 50, -60, 70, -80 };
        std::copy (g, g + 8, w.grad);
        Trainable weight = view (w, Shape::vector (8u));
        alignas (16) uint8_t bits[8] = { 1u, 0u, 1u, 1u, 0u, 1u, 0u, 1u };
        PruneMask mask = make_mask (bits, 8u);
        mask._M_pruned = 3u;
        mask._M_valid = 1u;
        REQUIRE (prune_apply_backward (mask, weight) == Status::OK);
        const int32_t expected[8] = { 10, 0, 30, -40, 0, -60, 0, -80 };
        for (uint32_t i = 0u; i < 8u; ++i)
            CHECK (w.grad[i] == expected[i]);
        weight._M_count = 9u; /* count mismatch */
        CHECK (prune_apply_backward (mask, weight) == Status::BAD_LAYOUT);
    }

    SECTION ("masked accumulation saturating sum matches the integer reference")
    {
        W w;
        w.fill (std::vector<int8_t> (8, 1), 0);
        Trainable weight = view (w, Shape::vector (8u));
        alignas (16) int16_t grad[8] = { 30000, -30000, 1000, -1000,
                                         32767, -32768, 5, -5 };
        weight.set_grad (0u, 10000);
        weight.set_grad (1u, -10000);
        weight.set_grad (2u, 32000);
        weight.set_grad (3u, -32000);
        weight.set_grad (4u, 0);
        weight.set_grad (5u, 0);
        weight.set_grad (6u, 32767);
        weight.set_grad (7u, -32768);
        alignas (16) uint8_t mask[8] = { 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u };
        uint32_t saturated = 99u;
        REQUIRE (prune_accumulate_masked (weight, mask, grad, 8u, saturated)
                 == Status::OK);
        /* Lanes 0..3 accumulate, 4..7 are gated off. */
        CHECK (w.grad[0] == ref_q15_to_i16_sat (10000 + 30000));
        CHECK (w.grad[1] == ref_q15_to_i16_sat (-10000 - 30000));
        CHECK (w.grad[2] == ref_q15_to_i16_sat (32000 + 1000));
        CHECK (w.grad[3] == ref_q15_to_i16_sat (-32000 - 1000));
        CHECK (w.grad[4] == 0);
        CHECK (w.grad[5] == 0);
        CHECK (w.grad[6] == 32767);
        CHECK (w.grad[7] == -32768);
        CHECK (saturated == 4u); /* every accumulated lane clamped: +/-40000, +/-33000 */

        SECTION ("only the clamped lanes are counted as saturations")
        {
            W m;
            m.fill (std::vector<int8_t> (8, 1), 0);
            Trainable mw = view (m, Shape::vector (8u));
            mw.set_grad (0u, 1000);
            mw.set_grad (1u, 32000);
            mw.set_grad (2u, -32700);
            mw.set_grad (3u, 0);
            mw.set_grad (4u, 0);
            mw.set_grad (5u, 0);
            mw.set_grad (6u, 0);
            mw.set_grad (7u, 0);
            alignas (16) int16_t mixed[8] = { 200, 2000, -2000, 7, 32767,
                                              -32768, 5, -5 };
            uint32_t counted = 99u;
            REQUIRE (prune_accumulate_masked (mw, mask, mixed, 8u, counted)
                     == Status::OK);
            CHECK (m.grad[0] == 1200);
            CHECK (m.grad[1] == ref_q15_to_i16_sat (34000));
            CHECK (m.grad[2] == ref_q15_to_i16_sat (-34700));
            CHECK (m.grad[3] == 7);
            /* Lanes 4..7 are gated off, so their rails are not accumulation
               events: exactly the two overflowing lanes are counted. */
            CHECK (counted == 2u);
            CHECK (m.grad[4] == 0);
            CHECK (m.grad[7] == 0);
        }

        SECTION ("vectorized route_and_add matches the same reference")
        {
            W v;
            v.fill (std::vector<int8_t> (8, 1), 0);
            Trainable vw = view (v, Shape::vector (8u));
            vw.set_grad (0u, 10000);
            vw.set_grad (1u, -10000);
            vw.set_grad (2u, 32000);
            vw.set_grad (3u, -32000);
            vw.set_grad (4u, 0);
            vw.set_grad (5u, 0);
            vw.set_grad (6u, 32767);
            vw.set_grad (7u, -32768);
            alignas (16) int16_t scratch[8] = { 0 };
            uint32_t vec_saturated = 99u;
            REQUIRE (prune_route_and_add (vw, mask, grad, scratch, 8u,
                                          vec_saturated)
                     == Status::OK);
            for (uint32_t i = 0u; i < 8u; ++i)
                CHECK (v.grad[i] == w.grad[i]);
            CHECK (vec_saturated == saturated);
            /* Scratch holds the gated gradient: zeros where the mask is 0. */
            for (uint32_t i = 0u; i < 8u; ++i)
                CHECK (scratch[i] == (mask[i] != 0u ? grad[i] : int16_t { 0 }));
            CHECK (prune_route_and_add (vw, nullptr, grad, scratch, 8u,
                                        vec_saturated)
                   == Status::BAD_LAYOUT);
        }
        SECTION ("count and weight mismatches are rejected")
        {
            alignas (16) int16_t scratch[8] = { 0 };
            uint32_t vec_saturated = 99u;
            CHECK (prune_accumulate_masked (weight, mask, grad, 7u, saturated)
                   == Status::BAD_LAYOUT);
            CHECK (prune_route_and_add (weight, mask, grad, scratch, 7u,
                                        vec_saturated)
                   == Status::BAD_LAYOUT); /* count != weight._M_count */
            Trainable unaligned = Trainable::make (w.code, w.grad + 1,
                                                  Shape::vector (8u),
                                                  Flag::RequiresGrad);
            CHECK (prune_route_and_add (unaligned, mask, grad, scratch, 8u,
                                        vec_saturated)
                   == Status::BAD_LAYOUT); /* the vector lanes must be aligned */
            CHECK (prune_route_and_add (weight, mask, grad, scratch + 1, 8u,
                                        vec_saturated)
                   == Status::BAD_LAYOUT);
            CHECK (prune_route_and_add (weight, nullptr, grad, scratch, 8u,
                                        vec_saturated)
                   == Status::BAD_LAYOUT);
            /* The same call with every operand in order is accepted, so the
             * rejections above are the mismatches and not a blanket failure. */
            CHECK (prune_route_and_add (weight, mask, grad, scratch, 8u,
                                        vec_saturated)
                   == Status::OK);
        }
    }
}

TEST_CASE ("prune_backward_hook gates accumulation and clears pruned lanes")
{
    const std::vector<int8_t> codes = { 0, -4, 0, 3, 0, 0, 9, 0 };
    W weight_w;
    weight_w.fill (codes, 0);
    Trainable weight = view (weight_w, Shape::vector (8u));
    W source_w;
    source_w.fill (std::vector<int8_t> (8, 0), 0);
    const int16_t incoming[8] = { 77, 55, -66, 44, 33, -22, 11, -99 };
    std::copy (incoming, incoming + 8, source_w.grad);
    Trainable source = view (source_w, Shape::vector (8u));

    alignas (16) uint8_t bits[8] = { 0u };
    const uint8_t pattern[8] = { 0u, 1u, 0u, 1u, 0u, 0u, 1u, 0u };
    std::copy (pattern, pattern + 8, bits);
    PruneMask mask = make_mask (bits, 8u);
    mask._M_pruned = 5u;
    mask._M_valid = 1u;

    uint32_t saturated = 99u;
    REQUIRE (prune_backward_hook (weight, mask, source, saturated) == Status::OK);
    /* Kept lanes accumulate the source gradient; pruned lanes stay 0. */
    for (uint32_t i = 0u; i < 8u; ++i)
        CHECK (weight_w.grad[i] == (pattern[i] != 0u ? incoming[i]
                                                     : int16_t { 0 }));
    CHECK (saturated == 0u);

    SECTION ("a second hook on saturated lanes reports clamp events")
    {
        weight_w.grad[1] = 30000;
        weight_w.grad[3] = -30000;
        weight_w.grad[6] = 5;
        const int16_t again[8] = { 0, 10000, 0, -10000, 0, 0, 32767, 0 };
        std::copy (again, again + 8, source_w.grad);
        REQUIRE (prune_backward_hook (weight, mask, source, saturated)
                 == Status::OK);
        CHECK (weight_w.grad[1] == ref_q15_to_i16_sat (40000));
        CHECK (weight_w.grad[3] == ref_q15_to_i16_sat (-40000));
        CHECK (weight_w.grad[6] == 32767);
        CHECK (saturated == 3u);
        /* Pruned lanes cannot come back to life. */
        CHECK (weight_w.grad[0] == 0);
        CHECK (weight_w.grad[4] == 0);
        CHECK (weight_w.grad[7] == 0);
    }

    SECTION ("size mismatches and invalid masks are refused")
    {
        source._M_count = 7u;
        CHECK (prune_backward_hook (weight, mask, source, saturated)
               == Status::BAD_LAYOUT);
        source._M_count = 8u;
        PruneMask invalid = make_mask (bits, 8u);
        CHECK (prune_backward_hook (weight, invalid, source, saturated)
               == Status::BAD_LAYOUT);
        W no_grad = weight_w;
        no_grad.n = 8u;
        Trainable frozen = view_forward_only (no_grad, Shape::vector (8u));
        CHECK (prune_backward_hook (frozen, mask, source, saturated)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("prune_validate_mask recomputes and cross-checks the codes")
{
    alignas (16) uint8_t bits[8] = { 1u, 0u, 1u, 0u, 1u, 1u, 1u, 1u };
    PruneMask mask = make_mask (bits, 8u);
    mask._M_pruned = 2u;
    mask._M_valid = 1u;

    SECTION ("consistent masked tensor passes")
    {
        W w;
        w.fill (std::vector<int8_t> { 5, 0, 4, 0, -2, 9, 0, 1 });
        w.code[6] = 7; /* non-zero under a keep bit: fine */
        Trainable weight = view (w, Shape::vector (8u));
        CHECK (prune_validate_mask (mask, weight, true) == Status::OK);
        CHECK (prune_validate_mask (mask, weight, false) == Status::OK);
    }

    SECTION ("a pruned lane with a live code is corrupt")
    {
        W w;
        w.fill (std::vector<int8_t> { 5, 3, 4, 0, -2, 9, 7, 1 });
        Trainable weight = view (w, Shape::vector (8u));
        CHECK (prune_validate_mask (mask, weight, true)
               == Status::CORRUPT_STREAM);
        /* Without the recompute the counter mismatch is not checked. */
        CHECK (prune_validate_mask (mask, weight, false) == Status::OK);
    }

    SECTION ("a stale pruned counter is corrupt")
    {
        W w;
        w.fill (std::vector<int8_t> { 5, 0, 4, 0, -2, 9, 7, 0 });
        Trainable weight = view (w, Shape::vector (8u));
        PruneMask stale = mask;
        stale._M_pruned = 3u;
        CHECK (prune_validate_mask (stale, weight, true)
               == Status::CORRUPT_STREAM);
    }

    SECTION ("count mismatch against the weight")
    {
        W w;
        w.fill (std::vector<int8_t> { 5, 0, 4 });
        Trainable weight = view (w, Shape::vector (3u));
        CHECK (prune_validate_mask (mask, weight, true) == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("prune_count_zeros reports zeros and the Q15 fraction")
{
    struct Case
    {
        std::vector<int8_t> codes;
        uint32_t zeros;
    };
    const std::vector<Case> cases = {
        { { 0, 0, 0, 0 }, 4u },
        { { 1, -1, 0, 2 }, 1u },
        { { 0, 5, -9, 0, 0, 3, 0, -1 }, 4u },
        { { 7, 8, 9 }, 0u },
    };
    for (const Case& k : cases)
    {
        uint32_t zeros = 99u;
        int32_t sparsity = 99999;
        REQUIRE (prune_count_zeros (k.codes.data (),
                                    static_cast<uint32_t> (k.codes.size ()),
                                    zeros, sparsity)
                 == Status::OK);
        const uint32_t count = static_cast<uint32_t> (k.codes.size ());
        CHECK (zeros == k.zeros);
        CHECK (sparsity == ref_ratio_q15 (static_cast<int64_t> (k.zeros),
                                          static_cast<int64_t> (count)));
    }
    uint32_t zeros = 0u;
    int32_t sparsity = 0;
    CHECK (prune_count_zeros (nullptr, 4u, zeros, sparsity) == Status::BAD_LAYOUT);
    CHECK (prune_count_zeros (reinterpret_cast<const int8_t*> (1),
                              _S_prune_max_count + 1u, zeros, sparsity)
           == Status::OUT_OF_RANGE);
    /* count 0 divides by 1 per the documented rule. */
    REQUIRE (prune_count_zeros (reinterpret_cast<const int8_t*> (
                                   &zeros), 0u, zeros, sparsity)
             == Status::OK);
    CHECK (zeros == 0u);
    CHECK (sparsity == 0);
}

TEST_CASE ("prune_row_geom derives the documented row geometry")
{
    struct Case
    {
        Shape shape;
        uint8_t axis;
        uint32_t row_len;
        uint32_t rows;
        uint32_t period;
        uint32_t blocks;
    };
    const std::vector<Case> cases = {
        { Shape::matrix (4u, 3u), 0u, 3u, 4u, 12u, 1u },
        { Shape::matrix (4u, 3u), 1u, 1u, 3u, 3u, 4u },
        { Shape::tensor3 (2u, 3u, 4u), 0u, 12u, 2u, 24u, 1u },
        { Shape::tensor3 (2u, 3u, 4u), 1u, 4u, 3u, 12u, 2u },
        { Shape::tensor3 (2u, 3u, 4u), 2u, 1u, 4u, 4u, 6u },
        { Shape::tensor4 (2u, 3u, 2u, 2u), 1u, 4u, 3u, 12u, 2u },
        { Shape::vector (6u), 0u, 1u, 6u, 6u, 1u },
    };
    for (const Case& k : cases)
    {
        PruneRowGeom geom;
        INFO ("axis " << unsigned (k.axis));
        REQUIRE (prune_row_geom (k.shape, k.axis, geom) == Status::OK);
        CHECK (geom._M_valid == 1u);
        CHECK (geom._M_row_len == k.row_len);
        CHECK (geom._M_rows == k.rows);
        CHECK (geom._M_period == k.period);
        CHECK (geom._M_blocks == k.blocks);
        CHECK (geom._M_count == k.shape.elements ());
        CHECK (geom._M_axis == k.axis);
    }

    SECTION ("invalid axes and shapes are refused")
    {
        PruneRowGeom geom;
        CHECK (prune_row_geom (Shape::matrix (4u, 3u), 2u, geom)
               == Status::BAD_LAYOUT);
        CHECK (prune_row_geom (Shape::matrix (4u, 3u), 4u, geom)
               == Status::BAD_LAYOUT);
        Shape broken;
        CHECK (prune_row_geom (broken, 0u, geom) == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("structured row selection matches the independent L1 reference")
{
    struct Scenario
    {
        std::vector<int8_t> codes; /* row-major, axis 0 of a matrix */
        uint32_t rows;
        uint32_t row_len;
        uint32_t keep;
    };
    const std::vector<Scenario> scenarios = {
        { { 1, 2, 3, 100, -100, 100, 0, 0, 0, 5, 5, 5 }, 4u, 3u, 3u },
        { { 1, 2, 3, 100, -100, 100, 0, 0, 0, 5, 5, 5 }, 4u, 3u, 2u },
        { { 1, 2, 3, 100, -100, 100, 0, 0, 0, 5, 5, 5 }, 4u, 3u, 4u },
        { { 7, 7, 1, 1, 1, 1, 4, 4 }, 4u, 2u, 1u },
        { { 0, 0, 0, 0, 0, 0 }, 3u, 2u, 1u }, /* all-zero scores */
    };

    for (const Scenario& s : scenarios)
    {
        alignas (16) int32_t scores[8] = { 0 };
        alignas (16) uint8_t keep_rows[8] = { 0u };
        PruneHistogram hist;
        StructuredPlan plan;
        uint32_t threshold = 99u;
        uint32_t tie_group = 99u;
        uint32_t tie_kept = 99u;

        PruneRowGeom geom;
        geom._M_row_len = s.row_len;
        geom._M_rows = s.rows;
        geom._M_period = s.row_len * s.rows;
        geom._M_blocks = 1u;
        geom._M_count = static_cast<uint32_t> (s.codes.size ());
        geom._M_axis = 0u;
        geom._M_valid = 1u;

        REQUIRE (prune_select_rows (s.codes.data (), geom, s.keep, scores,
                                    keep_rows, hist, plan, threshold, tie_group,
                                    tie_kept)
                 == Status::OK);

        const std::vector<int32_t> ref_scores
            = ref_row_scores (s.codes, s.rows, s.row_len, 1u);
        for (uint32_t row = 0u; row < s.rows; ++row)
            CHECK (scores[row] == ref_scores[row]);

        const std::vector<uint32_t> mags = ref_row_mags (ref_scores);
        const SelRef ref = ref_select (mags, s.keep);
        INFO ("rows=" << s.rows << " keep=" << s.keep);
        for (uint32_t row = 0u; row < s.rows; ++row)
            CHECK (keep_rows[row] == ref.mask[row]);
        CHECK (threshold == ref.threshold);
        CHECK (tie_group == ref.tie_group);
        CHECK (tie_kept == ref.tie_kept);
        CHECK (plan._M_rows_kept == s.keep);
        CHECK (plan._M_kept_count == s.keep * s.row_len);
        CHECK (plan._M_keep == keep_rows);

        /* Compaction: the kept rows, packed, in order. */
        std::vector<int8_t> expected_compacted;
        for (uint32_t row = 0u; row < s.rows; ++row)
            if (ref.mask[row] != 0u)
                for (uint32_t lane = 0u; lane < s.row_len; ++lane)
                    expected_compacted.push_back (s.codes[row * s.row_len + lane]);
        std::vector<int8_t> compacted (expected_compacted.size () + 4u, 42);
        REQUIRE (prune_compact_codes (s.codes.data (), plan, compacted.data (),
                                      static_cast<uint32_t> (
                                          expected_compacted.size ()))
                 == Status::OK);
        CHECK (std::equal (expected_compacted.begin (), expected_compacted.end (),
                           compacted.begin ()));
        /* Insufficient capacity is TRUNCATED, not a crash. */
        if (!expected_compacted.empty ())
            CHECK (prune_compact_codes (s.codes.data (), plan, compacted.data (),
                                        static_cast<uint32_t> (
                                            expected_compacted.size () - 1u))
                   == Status::TRUNCATED);
    }
}

TEST_CASE ("structured compaction works over multi-block tensors and lanes")
{
    /* tensor3 (2, 3, 4): channel axis 1, row_len 4, 3 rows, 2 blocks. */
    const std::vector<int8_t> codes = {
        /* block 0 */ 1, 1, 1, 1, 50, 50, 50, 50, 2, 2, 2, 2,
        /* block 1 */ 3, 3, 3, 3, 60, 60, 60, 60, -4, -4, -4, -4,
    };
    PruneRowGeom geom;
    REQUIRE (prune_row_geom (Shape::tensor3 (2u, 3u, 4u), 1u, geom) == Status::OK);
    CHECK (geom._M_blocks == 2u);

    alignas (16) int32_t scores[4] = { 0 };
    alignas (16) uint8_t keep_rows[4] = { 0u };
    PruneHistogram hist;
    StructuredPlan plan;
    uint32_t threshold = 99u;
    uint32_t tie_group = 99u;
    uint32_t tie_kept = 99u;
    /* Keep 2 of 3 rows. */
    REQUIRE (prune_select_rows (codes.data (), geom, 2u, scores, keep_rows, hist,
                                plan, threshold, tie_group, tie_kept)
             == Status::OK);
    const std::vector<int32_t> ref_scores = ref_row_scores (codes, 3u, 4u, 2u);
    CHECK (ref_scores[0] == 16);
    CHECK (ref_scores[1] == 440); /* the strong channel, both blocks */
    CHECK (ref_scores[2] == 24);
    for (uint32_t row = 0u; row < 3u; ++row)
        CHECK (scores[row] == ref_scores[row]);

    const std::vector<uint32_t> mags = ref_row_mags (ref_scores);
    const SelRef ref = ref_select (mags, 2u);
    for (uint32_t row = 0u; row < 3u; ++row)
        CHECK (keep_rows[row] == ref.mask[row]);
    CHECK (plan._M_rows_kept == 2u);
    CHECK (plan._M_kept_count == 2u * 2u * 4u); /* blocks * kept rows * row_len */

    std::vector<int8_t> compacted (plan._M_kept_count, 42);
    REQUIRE (prune_compact_codes (codes.data (), plan, compacted.data (),
                                  plan._M_kept_count)
             == Status::OK);
    std::vector<int8_t> expected;
    for (uint32_t block = 0u; block < 2u; ++block)
        for (uint32_t row = 0u; row < 3u; ++row)
            if (ref.mask[row] != 0u)
                for (uint32_t lane = 0u; lane < 4u; ++lane)
                    expected.push_back (codes[block * 12u + row * 4u + lane]);
    CHECK (compacted == expected);

    SECTION ("lane compaction mirrors the code compaction")
    {
        std::vector<int16_t> lanes (24u);
        for (uint32_t i = 0u; i < 24u; ++i)
            lanes[i] = static_cast<int16_t> (i);
        std::vector<int16_t> out_lanes (plan._M_kept_count, -1);
        REQUIRE (prune_compact_lanes (lanes.data (), plan, out_lanes.data (),
                                      plan._M_kept_count)
                 == Status::OK);
        std::vector<int16_t> expected_lanes;
        for (uint32_t block = 0u; block < 2u; ++block)
            for (uint32_t row = 0u; row < 3u; ++row)
                if (ref.mask[row] != 0u)
                    for (uint32_t lane = 0u; lane < 4u; ++lane)
                        expected_lanes.push_back (
                            lanes[block * 12u + row * 4u + lane]);
        CHECK (out_lanes == expected_lanes);
        CHECK (prune_compact_lanes (lanes.data (), plan, out_lanes.data (),
                                    plan._M_kept_count - 1u)
               == Status::TRUNCATED);
    }

    SECTION ("null inputs are refused")
    {
        StructuredPlan empty;
        CHECK (prune_compact_codes (codes.data (), empty,
                                    reinterpret_cast<int8_t*> (1), 1u)
               == Status::BAD_LAYOUT);
        CHECK (prune_select_rows (nullptr, geom, 2u, scores, keep_rows, hist,
                                  plan, threshold, tie_group, tie_kept)
               == Status::BAD_LAYOUT);
        PruneRowGeom invalid;
        CHECK (prune_select_rows (codes.data (), invalid, 2u, scores, keep_rows,
                                  hist, plan, threshold, tie_group, tie_kept)
               == Status::INTERNAL);
    }
}

TEST_CASE ("structured rewrite updates the shape, dims and layer")
{
    /* Build a plan: matrix (4, 3) axis 0, keep rows {0, 2, 3} (drop row 1). */
    PruneRowGeom geom;
    geom._M_row_len = 3u;
    geom._M_rows = 4u;
    geom._M_period = 12u;
    geom._M_blocks = 1u;
    geom._M_count = 12u;
    geom._M_axis = 0u;
    geom._M_valid = 1u;
    alignas (16) uint8_t keep_rows[4] = { 1u, 0u, 1u, 1u };
    StructuredPlan plan;
    plan._M_geom = geom;
    plan._M_keep = keep_rows;
    plan._M_rows_kept = 3u;
    plan._M_kept_count = 9u;

    SECTION ("shape rewrite shrinks the channel axis")
    {
        Shape shape = Shape::matrix (4u, 3u);
        REQUIRE (prune_rewrite_shape (plan, shape) == Status::OK);
        CHECK (shape._M_dims[0] == 3u);
        CHECK (shape._M_dims[1] == 3u);
        CHECK (shape.elements () == 9u);
        CHECK (shape == Shape::matrix (3u, 3u));

        /* A shape whose axis disagrees with the plan geometry is refused. */
        Shape other = Shape::matrix (5u, 3u);
        CHECK (prune_rewrite_shape (plan, other) == Status::BAD_LAYOUT);
        StructuredPlan empty;
        CHECK (prune_rewrite_shape (empty, shape) == Status::BAD_LAYOUT);
        /* Dropping every row leaves nothing to rewrite to. */
        StructuredPlan zero_rows = plan;
        zero_rows._M_rows_kept = 0u;
        Shape full = Shape::matrix (4u, 3u);
        CHECK (prune_rewrite_shape (zero_rows, full) == Status::OUT_OF_RANGE);
        /* An axis the shape does not have is refused before the dim compare. */
        Shape scalar = Shape::scalar ();
        StructuredPlan wide_axis = plan;
        wide_axis._M_geom._M_axis = 1u;
        CHECK (prune_rewrite_shape (wide_axis, scalar) == Status::BAD_LAYOUT);
    }

    SECTION ("axis 0 shrinks out_dim, axis 1 shrinks in_dim")
    {
        DimRewrite out;
        REQUIRE (prune_dims_rewrite (plan, 3u, 4u, out) == Status::OK);
        CHECK (out._M_valid == 1u);
        CHECK (out._M_axis == 0u);
        CHECK (out._M_in_from == 3u);
        CHECK (out._M_in_to == 3u);
        CHECK (out._M_out_from == 4u);
        CHECK (out._M_out_to == 3u);

        /* Input axis: matrix (2, 6) axis 1, drop 2 of 6 channels, row_len 1. */
        PruneRowGeom in_geom;
        in_geom._M_row_len = 1u;
        in_geom._M_rows = 6u;
        in_geom._M_period = 6u;
        in_geom._M_blocks = 2u;
        in_geom._M_count = 12u;
        in_geom._M_axis = 1u;
        in_geom._M_valid = 1u;
        alignas (16) uint8_t in_keep[6] = { 1u, 1u, 0u, 1u, 0u, 1u };
        StructuredPlan in_plan;
        in_plan._M_geom = in_geom;
        in_plan._M_keep = in_keep;
        in_plan._M_rows_kept = 4u;
        in_plan._M_kept_count = 8u;
        DimRewrite in_out;
        REQUIRE (prune_dims_rewrite (in_plan, 6u, 2u, in_out) == Status::OK);
        CHECK (in_out._M_in_from == 6u);
        CHECK (in_out._M_in_to == 4u);
        CHECK (in_out._M_out_to == 2u);

        CHECK (prune_dims_rewrite (plan, 0u, 4u, out) == Status::OUT_OF_RANGE);
        CHECK (prune_dims_rewrite (plan, 3u, 0u, out) == Status::OUT_OF_RANGE);
        CHECK (prune_dims_rewrite (plan, 3u, 70000u, out)
               == Status::OUT_OF_RANGE); /* dims are uint16-sized */
        /* Dropping every output channel: the dropped count equals the width,
           so the fold records out_to 0 and the layer step is the gate that
           refuses it (see the layer-fold section). */
        DimRewrite wiped;
        CHECK (prune_dims_rewrite (plan, 3u, 1u, wiped) == Status::OK);
        CHECK (wiped._M_out_to == 0u);
        CHECK (wiped._M_valid == 1u);
        /* More rows dropped than the descriptor has outputs. */
        StructuredPlan heavy = plan;
        heavy._M_rows_kept = 0u; /* dropped = 4 > out_dim 3 */
        CHECK (prune_dims_rewrite (heavy, 3u, 3u, out) == Status::OUT_OF_RANGE);
        /* An input-side prune that removes more lanes than in_dim. */
        StructuredPlan heavy_in = in_plan;
        heavy_in._M_rows_kept = 1u; /* dropped 5 * row_len 1 = 5 < 6: fits */
        CHECK (prune_dims_rewrite (heavy_in, 4u, 2u, out) == Status::OUT_OF_RANGE);
        StructuredPlan empty;
        CHECK (prune_dims_rewrite (empty, 3u, 4u, out) == Status::BAD_LAYOUT);
    }

    SECTION ("layer fold updates dims, weight length and arena lengths")
    {
        DimRewrite rewrite;
        REQUIRE (prune_dims_rewrite (plan, 3u, 4u, rewrite) == Status::OK);
        Layer layer {};
        layer._M_in_dim = 3u;
        layer._M_out_dim = 4u;
        layer._M_weight_len = 12u;
        layer._M_in_len = 3u;
        layer._M_out_len = 4u;
        REQUIRE (prune_apply_rewrite (rewrite, layer) == Status::OK);
        CHECK (layer._M_in_dim == 3u);
        CHECK (layer._M_out_dim == 3u);
        CHECK (layer._M_weight_len == 9u);
        CHECK (layer._M_in_len == 3u);
        CHECK (layer._M_out_len == 3u);

        /* A descriptor that does not match the rewrite's from-dims is refused. */
        Layer mismatch {};
        mismatch._M_in_dim = 5u;
        mismatch._M_out_dim = 4u;
        CHECK (prune_apply_rewrite (rewrite, mismatch) == Status::BAD_LAYOUT);

        DimRewrite invalid;
        CHECK (prune_apply_rewrite (invalid, layer) == Status::BAD_LAYOUT);

        /* A zero target dim is refused before the fold.  The descriptor must
           still match the rewrite's from-dims, so use a fresh one here. */
        Layer fresh {};
        fresh._M_in_dim = 3u;
        fresh._M_out_dim = 4u;
        fresh._M_weight_len = 12u;
        DimRewrite to_zero = rewrite;
        to_zero._M_out_to = 0u;
        CHECK (prune_apply_rewrite (to_zero, fresh) == Status::OUT_OF_RANGE);
        DimRewrite from_zero = rewrite;
        from_zero._M_in_to = 0u;
        CHECK (prune_apply_rewrite (from_zero, fresh) == Status::OUT_OF_RANGE);
        /* The refused fold leaves the descriptor untouched. */
        CHECK (fresh._M_in_dim == 3u);
        CHECK (fresh._M_out_dim == 4u);
        CHECK (fresh._M_weight_len == 12u);
    }
}

TEST_CASE ("prune_apply_structured end to end matches the row reference")
{
    const std::vector<int8_t> codes = {
        1, 1, 1,      /* row 0, L1 3 */
        100, -100, 2, /* row 1, L1 202 */
        0, 0, 0,      /* row 2, L1 0 */
        5, 5, 5,      /* row 3, L1 15 */
    };
    W w;
    w.fill (codes);
    Trainable weight = view (w, Shape::matrix (4u, 3u));
    PruneConfig c;
    c._M_pattern = static_cast<uint8_t> (PrunePattern::StructuredRows);
    c._M_axis = 0u;
    c._M_target_sparsity_q15 = 8192; /* floor(4 * 8192 / 32768) = 1 row */

    alignas (16) int32_t scores[4] = { 0 };
    alignas (16) uint8_t keep_rows[4] = { 0u };
    PruneHistogram hist;
    StructuredPlan plan;
    PruneReport report;
    std::vector<int8_t> compacted (12u, 42);
    REQUIRE (prune_apply_structured (c, 0u, weight, scores, keep_rows, hist,
                                     compacted.data (), 12u, plan, report)
             == Status::OK);

    /* Independent reference: scores 3, 202, 0, 15 -> classes via floor(*127/202). */
    const std::vector<int32_t> ref_scores = ref_row_scores (codes, 4u, 3u, 1u);
    CHECK (ref_scores[0] == 3);
    CHECK (ref_scores[1] == 202);
    CHECK (ref_scores[2] == 0);
    CHECK (ref_scores[3] == 15);
    for (uint32_t row = 0u; row < 4u; ++row)
        CHECK (scores[row] == ref_scores[row]);
    const SelRef ref = ref_select (ref_row_mags (ref_scores), 3u);
    for (uint32_t row = 0u; row < 4u; ++row)
        CHECK (keep_rows[row] == ref.mask[row]);
    CHECK (plan._M_rows_kept == 3u);
    CHECK (plan._M_kept_count == 9u);

    CHECK (report._M_elements == 12u);
    CHECK (report._M_requested_sparsity_q15 == 8192);
    CHECK (report._M_target_pruned == 3u); /* 1 row * 3 lanes */
    CHECK (report._M_pruned == 3u);
    CHECK (report._M_achieved_sparsity_q15
           == ref_ratio_q15 (3, 12));

    /* Compacted stream: rows 0, 1, 3 packed (row 2 was the zero row). */
    const int8_t expected[9] = { 1, 1, 1, 100, -100, 2, 5, 5, 5 };
    for (uint32_t i = 0u; i < 9u; ++i)
        CHECK (compacted[i] == expected[i]);

    SECTION ("capacity below the kept count is TRUNCATED")
    {
        StructuredPlan p2;
        PruneReport r2;
        CHECK (prune_apply_structured (c, 0u, weight, scores, keep_rows, hist,
                                       compacted.data (), 8u, p2, r2)
               == Status::TRUNCATED);
    }

    SECTION ("pattern mismatch is refused")
    {
        PruneConfig wrong = c;
        wrong._M_pattern = static_cast<uint8_t> (PrunePattern::Unstructured);
        CHECK (prune_apply_structured (wrong, 0u, weight, scores, keep_rows,
                                       hist, compacted.data (), 12u, plan,
                                       report)
               == Status::BAD_LAYOUT);
    }

    SECTION ("prune_apply refuses the structured pattern")
    {
        alignas (16) uint8_t bits[16] = { 0u };
        PruneMask mask = make_mask (bits, 12u);
        PruneReport r;
        CHECK (prune_apply (c, 0u, weight, hist, mask, r) == Status::UNSUPPORTED);
    }

    SECTION ("an all-zero tensor yields an all-zero row selection")
    {
        W z;
        z.fill (std::vector<int8_t> (12u, 0));
        Trainable zw = view (z, Shape::matrix (4u, 3u));
        PruneConfig zc = c;
        zc._M_target_sparsity_q15 = 16384; /* drop 2 rows */
        StructuredPlan zp;
        PruneReport zr;
        std::vector<int8_t> zc_buf (12u, 7);
        REQUIRE (prune_apply_structured (zc, 0u, zw, scores, keep_rows, hist,
                                         zc_buf.data (), 12u, zp, zr)
                 == Status::OK);
        /* All scores 0 -> all classes 0 -> ties break to the lowest rows. */
        CHECK (zp._M_rows_kept == 2u);
        CHECK (keep_rows[0] == 1u);
        CHECK (keep_rows[1] == 1u);
        CHECK (keep_rows[2] == 0u);
        CHECK (keep_rows[3] == 0u);
        CHECK (zr._M_pruned == 6u);
        CHECK (zr._M_threshold_magnitude == 0);
    }
}

TEST_CASE ("prune_apply_set aggregates the run report exactly")
{
    std::vector<int8_t> c0 = { 5, -3, -5, 2, 7, 0, -1, 4 };
    std::vector<int8_t> c1 = { -9, 9, -9, 9, 1, -1, 0, 3, -3, 3, 3, -8 };
    W w0;
    w0.fill (c0);
    W w1;
    w1.fill (c1);
    Trainable weights[2] = { view (w0, Shape::vector (8u)),
                             view (w1, Shape::vector (12u)) };
    PruneConfig configs[2];
    configs[0]._M_target_sparsity_q15 = 16384;
    configs[1]._M_target_sparsity_q15 = 8192;
    alignas (16) uint8_t bits0[16] = { 0u };
    alignas (16) uint8_t bits1[16] = { 0u };
    PruneMask masks[2] = { make_mask (bits0, 0u), make_mask (bits1, 0u) };
    PruneHistogram hist;
    PruneReport reports[2];
    PruneRunReport run;

    REQUIRE (prune_apply_set (weights, configs, 2u, 0u, hist, masks, reports,
                              run)
             == Status::OK);

    const uint32_t t0 = 8u * 16384u / 32768u; /* 4 */
    const uint32_t t1 = 12u * 8192u / 32768u; /* 3 */
    CHECK (run._M_tensors == 2u);
    CHECK (run._M_elements == 20u);
    CHECK (run._M_pruned == t0 + t1);
    CHECK (reports[0]._M_pruned == t0);
    CHECK (reports[1]._M_pruned == t1);
    CHECK (run._M_achieved_sparsity_q15
           == ref_ratio_q15 (static_cast<int64_t> (t0 + t1), 20));
    CHECK (run._M_step_sparsity_q15 == 16384); /* configs[0], constant */
    CHECK (run._M_rows_before == 0u);
    CHECK (run._M_rows_after == 0u);
    CHECK (run._M_blocks_total == 0u);
    CHECK (masks[0]._M_pruned == t0);
    CHECK (masks[1]._M_pruned == t1);

    SECTION ("independent per-lane check of both masks")
    {
        const SelRef r0 = ref_select_codes (c0, 8u - t0);
        const SelRef r1 = ref_select_codes (c1, 12u - t1);
        for (uint32_t i = 0u; i < 8u; ++i)
            CHECK (bits0[i] == r0.mask[i]);
        for (uint32_t i = 0u; i < 12u; ++i)
            CHECK (bits1[i] == r1.mask[i]);
    }

    SECTION ("validation and error propagation")
    {
        CHECK (prune_apply_set (nullptr, configs, 2u, 0u, hist, masks, reports,
                                 run)
               == Status::BAD_LAYOUT);
        CHECK (prune_apply_set (weights, configs, 0u, 0u, hist, masks, reports,
                                run)
               == Status::OUT_OF_RANGE);
        CHECK (prune_apply_set (weights, configs, _S_prune_max_count + 1u, 0u,
                                hist, masks, reports, run)
               == Status::OUT_OF_RANGE);
        /* A bad config on the second tensor aborts with its status. */
        PruneConfig bad[2] = { configs[0], configs[1] };
        bad[1]._M_target_sparsity_q15 = 40000;
        CHECK (prune_apply_set (weights, bad, 2u, 0u, hist, masks, reports, run)
               == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("prune scratch sizing fits the documented pool costs")
{
    CHECK (prune_mask_bytes (10u) == 12u);
    CHECK (prune_mask_bytes (8u) == 8u);
    CHECK (prune_mask_packed_bytes (10u) == 4u);
    CHECK (prune_mask_packed_bytes (64u) == 8u);
    CHECK (prune_row_scratch_bytes (4u) == 20u);

    SECTION ("one unstructured prune of 1024 lanes costs 516 + 1024 bytes")
    {
        alignas (16) uint8_t arena[_S_train_pool_bytes];
        Pool pool (arena, sizeof (arena));
        PruneHistogram* hist = nullptr;
        uint8_t* mask = nullptr;
        REQUIRE (pool.carve_typed<PruneHistogram> (1u, hist) == Status::OK);
        REQUIRE (pool.carve_typed<uint8_t> (prune_mask_bytes (1024u), mask)
                 == Status::OK);
        CHECK (pool.stats ()._M_used == 516u + 1024u);
        CHECK (516u + 1024u == 1540u);
        CHECK (1540u <= _S_train_pool_bytes);

        SECTION ("the mask carves aligned, so it is SIMD-ready")
        {
            CHECK (simd_aligned (mask));
            CHECK (simd_aligned (hist));
        }
    }

    SECTION ("an oversized request surfaces ARENA_OVERFLOW, never a crash")
    {
        alignas (16) uint8_t tiny[64];
        Pool pool (tiny, sizeof (tiny));
        /* The 516-byte histogram does not fit a 64-byte arena at all, and the
           pointer stays null instead of aliasing somebody else's slab. */
        PruneHistogram* hist = nullptr;
        CHECK (pool.carve_typed<PruneHistogram> (1u, hist)
               == Status::ARENA_OVERFLOW);
        CHECK (hist == nullptr);
        CHECK (pool.stats ()._M_overflows == 1u);

        uint8_t* mask = nullptr;
        REQUIRE (pool.carve_typed<uint8_t> (prune_mask_bytes (48u), mask)
                 == Status::OK);
        CHECK (simd_aligned (mask));
        uint8_t* over = nullptr;
        CHECK (pool.carve_typed<uint8_t> (prune_mask_bytes (32u), over)
               == Status::ARENA_OVERFLOW);
        CHECK (over == nullptr);
        CHECK (pool.stats ()._M_overflows == 2u);
        /* The first slab is untouched by the failed request. */
        CHECK (pool.stats ()._M_used == 48u);
        /* A reset frees the arena, so the same request now fits. */
        pool.reset ();
        uint8_t* again = nullptr;
        CHECK (pool.carve_typed<uint8_t> (prune_mask_bytes (32u), again)
               == Status::OK);
        CHECK (pool.stats ()._M_used == 32u);
        CHECK (pool.stats ()._M_overflows == 2u);
    }

    SECTION ("a full-size arena holds the documented worst-case prune set")
    {
        alignas (16) uint8_t arena[_S_train_pool_bytes];
        Pool pool (arena, sizeof (arena));
        CHECK (_S_train_pool_bytes == 8192u);
        PruneHistogram* hist = nullptr;
        REQUIRE (pool.carve_typed<PruneHistogram> (1u, hist) == Status::OK);
        int32_t* scores = nullptr;
        REQUIRE (pool.carve_typed<int32_t> (prune_row_scratch_bytes (64u) / 4u,
                                            scores)
                 == Status::OK);
        uint8_t* rows = nullptr;
        REQUIRE (pool.carve_typed<uint8_t> (prune_mask_bytes (64u), rows)
                 == Status::OK);
        uint8_t* mask = nullptr;
        REQUIRE (pool.carve_typed<uint8_t> (prune_mask_bytes (6000u), mask)
                 == Status::OK);
        CHECK (pool.stats ()._M_used
               <= _S_train_pool_bytes);
        uint8_t* spill = nullptr;
        CHECK (pool.carve_typed<uint8_t> (prune_mask_bytes (2000u), spill)
               == Status::ARENA_OVERFLOW);
        CHECK (spill == nullptr);
        pool.reset ();
        CHECK (pool.stats ()._M_used == 0u);
    }
}

TEST_CASE ("prune_apply input validation table")
{
    alignas (16) uint8_t bits[16] = { 0u };
    PruneHistogram hist;
    PruneReport report;

    SECTION ("zero-count tensor is refused")
    {
        alignas (16) int8_t code[4] = { 0, 0, 0, 0 };
        alignas (16) int16_t grad[4] = { 0, 0, 0, 0 };
        Trainable weight = Trainable::make (code, grad, Shape::scalar (),
                                            Flag::RequiresGrad);
        PruneConfig c;
        c._M_target_sparsity_q15 = 0;
        PruneMask mask = make_mask (bits, 1u);
        /* count 1 with sparsity 0: fine, all-keep. */
        CHECK (prune_apply (c, 0u, weight, hist, mask, report) == Status::OK);
        CHECK (report._M_elements == 1u);
    }

    SECTION ("bad config and missing mask")
    {
        W w;
        w.fill (std::vector<int8_t> { 1, 2, 3, 4 });
        Trainable weight = view (w, Shape::vector (4u));
        PruneConfig bad;
        bad._M_target_sparsity_q15 = 40000;
        PruneMask mask = make_mask (bits, 4u);
        CHECK (prune_apply (bad, 0u, weight, hist, mask, report)
               == Status::OUT_OF_RANGE);
        PruneConfig c;
        c._M_target_sparsity_q15 = 16384;
        PruneMask no_bits = make_mask (nullptr, 4u);
        CHECK (prune_apply (c, 0u, weight, hist, no_bits, report)
               == Status::BAD_LAYOUT);
        static uint8_t unaligned[8];
        PruneMask skewed = make_mask (unaligned + 1, 4u);
        CHECK (prune_apply (c, 0u, weight, hist, skewed, report)
               == Status::BAD_LAYOUT);
    }

    SECTION ("invalid weight (shape mismatch) is refused")
    {
        W w;
        w.fill (std::vector<int8_t> { 1, 2, 3, 4 });
        Trainable weight = view (w, Shape::vector (4u));
        weight._M_count = 3u;
        PruneConfig c;
        c._M_target_sparsity_q15 = 16384;
        PruneMask mask = make_mask (bits, 4u);
        CHECK (prune_apply (c, 0u, weight, hist, mask, report)
               == Status::BAD_LAYOUT);
    }
}

/* ======================================================================== */
/* Model averaging (mllee/train_model_averaging.hpp)                         */
/* ======================================================================== */

TEST_CASE ("averaging POD layouts, defaults and pool helpers are documented")
{
    /* The header pins the 32-bit device layouts; a pointer-holding view grows by
       its trailing alignment on the 64-bit host, which the same assert allows. */
    STATIC_REQUIRE (sizeof (SnapshotSet)
                    == (sizeof (void *) == 4u ? 16u : 24u));
    STATIC_REQUIRE (sizeof (ParamSet)
                    == (sizeof (void *) == 4u ? 16u : 24u));
    STATIC_REQUIRE (sizeof (EmaState)
                    == (sizeof (void *) == 4u ? 24u : 32u));
    STATIC_REQUIRE (sizeof (AvgGate) == 24u);
    STATIC_REQUIRE (sizeof (DivergenceReport) == 24u);
    STATIC_REQUIRE (sizeof (AveragingReport) == 32u);

    CHECK (_S_avg_max_count == 65535u);
    CHECK (_S_avg_max_snapshots == 4096u);
    CHECK (_S_avg_acc_max == (INT64_C (1) << 60) - 1);
    CHECK (static_cast<uint8_t> (AveragingMode::Uniform) == 0u);
    CHECK (static_cast<uint8_t> (AveragingMode::Ema) == 1u);
    CHECK (static_cast<uint8_t> (AveragingMode::Swa) == 2u);

    SnapshotSet empty_set;
    CHECK (empty_set._M_streams == nullptr);
    CHECK (empty_set._M_snapshots == 0u);
    CHECK (empty_set._M_count == 0u);
    CHECK (empty_set._M_valid == 0u);

    ParamSet empty_params;
    CHECK (empty_params._M_params == nullptr);
    CHECK (empty_params._M_tensors == 0u);
    CHECK (empty_params._M_lane_total == 0u);
    CHECK (empty_params.validate () == Status::BAD_LAYOUT);

    EmaState empty_state;
    CHECK (empty_state._M_lanes == nullptr);
    CHECK (empty_state._M_count == 0u);
    CHECK (empty_state._M_updates == 0u);
    CHECK (empty_state._M_saturation_events == 0u);
    CHECK (empty_state._M_valid == 0u);
    /* Documented default decay: 0.994 in Q15. */
    CHECK (empty_state._M_decay_q15 == 32572);
    CHECK (empty_state.validate () == Status::BAD_LAYOUT);

    AvgGate gate;
    CHECK (gate._M_best_q15 == 0);
    CHECK (gate._M_min_delta_q15 == 1);
    CHECK (gate._M_accepted == 0u);
    CHECK (gate._M_rejected == 0u);
    CHECK (gate._M_compared == 0u);
    CHECK (gate._M_higher_is_better == 0u);
    CHECK (gate._M_has_best == 0u);
    CHECK (gate.validate () == Status::OK);
    gate._M_min_delta_q15 = -1;
    CHECK (gate.validate () == Status::OUT_OF_RANGE);
    gate._M_min_delta_q15 = 0;
    CHECK (gate.validate () == Status::OK);

    DivergenceReport divergence;
    CHECK (divergence._M_lanes_compared == 0u);
    CHECK (divergence._M_lanes_diverging == 0u);
    CHECK (divergence._M_worst_snapshot == 0u);
    CHECK (divergence._M_worst_snapshot_lanes == 0u);
    CHECK (divergence._M_worst_fraction_q15 == 0);
    CHECK (divergence._M_limit_q15 == 0);

    AveragingReport report;
    CHECK (report._M_tensors == 0u);
    CHECK (report._M_lanes == 0u);
    CHECK (report._M_snapshots_used == 0u);
    CHECK (report._M_gate_accepted == 0u);
    CHECK (report._M_gate_rejected == 0u);
    CHECK (report._M_max_code_drift == 0);
    CHECK (report._M_drifted_lanes == 0u);
    CHECK (report._M_mode == static_cast<uint8_t> (AveragingMode::Uniform));

    SECTION ("the documented caller-storage sizes hold inside the training pool")
    {
        CHECK (_S_pool_alignment == 4u);
        CHECK (_S_train_pool_bytes == 8192u);
        CHECK (avg_ema_state_bytes (0u) == 0u);
        CHECK (avg_ema_state_bytes (5u) == align_up (10u, _S_pool_alignment));
        CHECK (avg_ema_state_bytes (256u) == 512u);
        /* Documented worst case: a model-parallel EMA over 4 tensors of 256
           lanes costs 4 * 512 = 2048 pool bytes, a quarter of the arena. */
        CHECK (4u * avg_ema_state_bytes (256u) == 2048u);
        CHECK (4u * avg_ema_state_bytes (256u)
               <= static_cast<uint32_t> (_S_train_pool_bytes));
        CHECK (avg_snapshot_table_bytes (0u) == 0u);
        CHECK (avg_snapshot_table_bytes (1u)
               == align_up (static_cast<uint32_t> (sizeof (void *)),
                            _S_pool_alignment));
        CHECK (avg_snapshot_table_bytes (8u)
               == align_up (8u * static_cast<uint32_t> (sizeof (void *)),
                            _S_pool_alignment));
    }
}

TEST_CASE ("avg_div_round divides half-away-from-zero and saturates the int32 rail")
{
    struct Row
    {
        int64_t _M_num;
        uint32_t _M_den;
    };

    /* Every accumulator the code grid can produce, both signs, plus the exact
       ties where half-up and half-away disagree. */
    const Row rows[] = {
        { 0, 1u },         { 129, 2u },     { -129, 2u },   { 258, 2u },
        { -258, 2u },      { 259, 2u },     { -259, 2u },   { 5, 10u },
        { -5, 10u },       { 1, 2u },       { -1, 2u },     { 100, 3u },
        { -100, 3u },      { 32766, 3u },   { -32766, 3u }, { 516, 4u },
        { -516, 4u },      { 774, 5u },     { -774, 5u },   { 2580, 10u },
        { -2580, 10u },    { 2147483647LL, 3u }, { -2147483647LL, 3u },
        { INT64_C (32766) * 4096, 4096u },
        { -INT64_C (32766) * 4096, 4096u },
        { INT64_C (32766) * 4095, 4095u },
        { INT64_C (1) << 59, 7u },
        { -(INT64_C (1) << 59), 7u },
    };
    /* The reference divides without a rail, so clamp it to int32 before
       comparing: past that point avg_div_round saturates instead of wrapping. */
    const auto clamp_i32 = [](const int64_t v) -> int32_t {
        if (v > 2147483647LL)
            return 2147483647;
        if (v < -2147483648LL)
            return -2147483647 - 1;
        return static_cast<int32_t> (v);
    };
    for (const Row &row : rows)
        CHECK (avg_div_round (row._M_num, row._M_den)
               == clamp_i32 (ref_div_round_away (row._M_num, row._M_den)));

    /* A negative tie leaves the module's default round-half-up: -259 / 2 is
       -130 here, while _S_round_shift would have produced -129. */
    CHECK (avg_div_round (-259, 2u) == -130);
    CHECK (ref_round_shift (-259, 1u) == -129);
    CHECK (avg_div_round (259, 2u) == 130);

    CHECK (avg_div_round (12345, 0u) == 0);
    CHECK (avg_div_round (0, 0u) == 0);

    /* Round-half-away can push the magnitude past int32: saturate, never wrap. */
    CHECK (avg_div_round (INT64_C (2147483648), 1u) == 2147483647);
    CHECK (avg_div_round (INT64_C (2147483649), 1u) == 2147483647);
    CHECK (avg_div_round (-INT64_C (2147483649), 1u) == (-2147483647 - 1));
    CHECK (avg_div_round (INT64_MIN, 1u) == (-2147483647 - 1));
    CHECK (avg_div_round (INT64_MAX, 1u) == 2147483647);
    CHECK (avg_div_round (-INT64_C (1) << 60, 1u) == (-2147483647 - 1));
    CHECK (avg_div_round (INT64_C (1) << 60, 3u) == 2147483647);
}

TEST_CASE ("avg_snapshots_open rejects null, misaligned and oversized sets")
{
    W a, b, c;
    a.fill ({ 1, 2, 3, 4 });
    b.fill ({ -1, -2, -3, -4 });
    c.fill ({ 0, 0, 0, 0 });
    alignas (16) const int8_t *streams[3] = { a.code, b.code, c.code };
    SnapshotSet set;

    SECTION ("a valid binding publishes the view")
    {
        REQUIRE (avg_snapshots_open (set, streams, 3u, 4u) == Status::OK);
        CHECK (set._M_valid == 1u);
        CHECK (set._M_snapshots == 3u);
        CHECK (set._M_count == 4u);
        CHECK (set._M_streams == streams);
    }

    SECTION ("the pointer table itself must exist")
    {
        CHECK (avg_snapshots_open (set, nullptr, 3u, 4u) == Status::BAD_LAYOUT);
        CHECK (set._M_valid == 0u);
    }

    SECTION ("counts are bounded by the module maxima")
    {
        CHECK (avg_snapshots_open (set, streams, 0u, 4u)
               == Status::OUT_OF_RANGE);
        CHECK (avg_snapshots_open (set, streams, 3u, 0u) == Status::OUT_OF_RANGE);
        CHECK (avg_snapshots_open (set, streams, _S_avg_max_snapshots + 1u, 4u)
               == Status::OUT_OF_RANGE);
        CHECK (avg_snapshots_open (set, streams, 3u, _S_avg_max_count + 1u)
               == Status::OUT_OF_RANGE);
        CHECK (set._M_valid == 0u);
    }

    SECTION ("every stream is validated, not just the first")
    {
        alignas (16) const int8_t *holes[3] = { a.code, nullptr, c.code };
        CHECK (avg_snapshots_open (set, holes, 3u, 4u) == Status::BAD_LAYOUT);
        CHECK (set._M_valid == 0u);

        alignas (16) const int8_t *ragged[2]
            = { a.code, reinterpret_cast<const int8_t *> (b.code) + 1 };
        CHECK (avg_snapshots_open (set, ragged, 2u, 4u) == Status::BAD_LAYOUT);
        CHECK (set._M_valid == 0u);

        /* A byte-offset table pointer is the only way to break the 4-byte
           alignment contract here. */
        const int8_t *const *misaligned
            = reinterpret_cast<const int8_t *const *> (
                  reinterpret_cast<const uint8_t *> (streams) + 1u);
        CHECK (avg_snapshots_open (set, misaligned, 3u, 4u)
               == Status::BAD_LAYOUT);
    }

    SECTION ("a failed rebind clears the previously valid view")
    {
        REQUIRE (avg_snapshots_open (set, streams, 3u, 4u) == Status::OK);
        CHECK (avg_snapshots_open (set, streams, 0u, 4u) == Status::OUT_OF_RANGE);
        CHECK (set._M_valid == 0u);
        CHECK (set._M_streams == nullptr);
        CHECK (set._M_snapshots == 0u);
        CHECK (set._M_count == 0u);
    }
}

TEST_CASE ("avg_uniform reproduces an independent int64 mean narrowed to codes")
{
    W first, second, third, fourth;
    first.fill ({ 10, -20, 30, 40, -50, 0, 127, -127 });
    second.fill ({ 12, -18, 34, 36, -54, 4, 127, -126 });
    third.fill ({ 8, -22, 28, 44, -46, -4, 126, -127 });
    fourth.fill ({ -10, 20, -30, -40, 50, 0, 127, -127 });
    alignas (16) const int8_t *streams[4]
        = { first.code, second.code, third.code, fourth.code };
    SnapshotSet set;
    REQUIRE (avg_snapshots_open (set, streams, 4u, 8u) == Status::OK);
    alignas (16) int8_t out[8] = { 0 };

    SECTION ("the full window matches the re-implemented accumulator")
    {
        REQUIRE (avg_uniform (set, 0u, 4u, out) == Status::OK);
        const std::vector<std::vector<int8_t>> all
            = { first.codes (), second.codes (), third.codes (),
                fourth.codes () };
        const std::vector<int8_t> expected = ref_avg_codes (all, { 0u, 1u, 2u, 3u });
        for (uint32_t lane = 0u; lane < 8u; ++lane)
            CHECK (out[lane] == expected[lane]);
    }

    SECTION ("a sub-window only averages the snapshots it covers")
    {
        const std::vector<std::vector<int8_t>> all
            = { first.codes (), second.codes (), third.codes (),
                fourth.codes () };
        struct Window
        {
            uint32_t _M_from;
            uint32_t _M_to;
        };
        const Window windows[] = { { 0u, 1u }, { 1u, 3u }, { 2u, 4u },
                                   { 0u, 2u }, { 3u, 4u } };
        for (const Window &w : windows)
        {
            REQUIRE (avg_uniform (set, w._M_from, w._M_to, out) == Status::OK);
            std::vector<uint32_t> picks;
            for (uint32_t i = w._M_from; i < w._M_to; ++i)
                picks.push_back (i);
            const std::vector<int8_t> expected = ref_avg_codes (all, picks);
            for (uint32_t lane = 0u; lane < 8u; ++lane)
                CHECK (out[lane] == expected[lane]);
            INFO ("window " << w._M_from << ".." << w._M_to);
        }
    }

    SECTION ("a one-snapshot window is an exact copy")
    {
        REQUIRE (avg_uniform (set, 2u, 3u, out) == Status::OK);
        for (uint32_t lane = 0u; lane < 8u; ++lane)
            CHECK (out[lane] == third.code[lane]);
    }

    SECTION ("the rails narrow to +/-127 and never to the forbidden -128")
    {
        REQUIRE (avg_uniform (set, 0u, 3u, out) == Status::OK);
        /* Lanes 6 and 7 average (127,127,126) and (-127,-126,-127). */
        CHECK (out[6] == 127);
        CHECK (out[7] == -127);
        for (const int8_t code : out)
            CHECK (code != -128);

        alignas (16) const int8_t *rails[2] = { first.code, fourth.code };
        SnapshotSet rail_set;
        REQUIRE (avg_snapshots_open (rail_set, rails, 2u, 8u) == Status::OK);
        REQUIRE (avg_uniform (rail_set, 0u, 2u, out) == Status::OK);
        /* (127 + 127) / 2 = 127, (-127 + -127) / 2 = -127, round trip exact. */
        CHECK (out[6] == 127);
        CHECK (out[7] == -127);
        /* (10 + -10) / 2 = 0 for the surviving lanes of the opposite pair. */
        CHECK (out[0] == 0);
        CHECK (out[4] == 0);
        for (const int8_t code : out)
            CHECK (code != -128);
    }

    SECTION ("an invalid set or destination is refused without writing")
    {
        alignas (16) int8_t untouched[8] = { 9, 9, 9, 9, 9, 9, 9, 9 };
        SnapshotSet bad;
        CHECK (avg_uniform (bad, 0u, 1u, untouched) == Status::BAD_LAYOUT);
        CHECK (avg_uniform (set, 0u, 1u, nullptr) == Status::BAD_LAYOUT);
        CHECK (avg_uniform (set, 0u, 1u,
                            reinterpret_cast<int8_t *> (untouched) + 1)
               == Status::BAD_LAYOUT);
        /* Empty and inverted windows, and a window past the last snapshot. */
        CHECK (avg_uniform (set, 2u, 2u, untouched) == Status::OUT_OF_RANGE);
        CHECK (avg_uniform (set, 3u, 1u, untouched) == Status::OUT_OF_RANGE);
        CHECK (avg_uniform (set, 0u, 5u, untouched) == Status::OUT_OF_RANGE);
        CHECK (avg_uniform (set, 4u, 5u, untouched) == Status::OUT_OF_RANGE);
        for (const int8_t code : untouched)
            CHECK (code == 9);
    }

    SECTION ("the accumulator ceiling keeps the int64 sum monotone in the count")
    {
        /* 4096 identical streams is the largest set a SnapshotSet can hold; the
           sum stays 8 orders of magnitude below the documented ceiling, so the
           per-add clamp cannot wrap and the mean is the exact code value. */
        CHECK (_S_avg_acc_max / 32766 > 4096);
        W big;
        big.fill ({ 127, -127, 1, -1, 0, 40, -40, 63 });
        std::vector<int8_t> expected = ref_avg_codes (
            { big.codes (), big.codes (), big.codes () }, { 0u, 1u, 2u });
        alignas (16) const int8_t *same[3]
            = { big.code, big.code, big.code };
        SnapshotSet triple;
        REQUIRE (avg_snapshots_open (triple, same, 3u, 8u) == Status::OK);
        REQUIRE (avg_uniform (triple, 0u, 3u, out) == Status::OK);
        for (uint32_t lane = 0u; lane < 8u; ++lane)
            CHECK (out[lane] == expected[lane]);
    }
}

TEST_CASE ("avg_swa lets a duplicated snapshot index weigh twice")
{
    W early, middle, late;
    early.fill ({ 10, -30, 50, 0 });
    middle.fill ({ 20, -20, 40, 8 });
    late.fill ({ 30, -10, 30, 16 });
    alignas (16) const int8_t *streams[3]
        = { early.code, middle.code, late.code };
    SnapshotSet set;
    REQUIRE (avg_snapshots_open (set, streams, 3u, 4u) == Status::OK);
    alignas (16) int8_t out[4] = { 0 };
    const std::vector<std::vector<int8_t>> all
        = { early.codes (), middle.codes (), late.codes () };

    SECTION ("uniform window and the equivalent index list agree")
    {
        alignas (16) const uint32_t indices[3] = { 0u, 1u, 2u };
        REQUIRE (avg_swa (set, indices, 3u, out) == Status::OK);
        alignas (16) int8_t window[4] = { 0 };
        REQUIRE (avg_uniform (set, 0u, 3u, window) == Status::OK);
        for (uint32_t lane = 0u; lane < 4u; ++lane)
            CHECK (out[lane] == window[lane]);
    }

    SECTION ("repeating the late snapshot pulls the average toward it")
    {
        alignas (16) const uint32_t indices[3] = { 0u, 2u, 2u };
        REQUIRE (avg_swa (set, indices, 3u, out) == Status::OK);
        const std::vector<int8_t> weighted = ref_avg_codes (all, { 0u, 2u, 2u });
        for (uint32_t lane = 0u; lane < 4u; ++lane)
            CHECK (out[lane] == weighted[lane]);
        alignas (16) const uint32_t pair[2] = { 0u, 2u };
        alignas (16) int8_t plain[4] = { 0 };
        REQUIRE (avg_swa (set, pair, 2u, plain) == Status::OK);
        const std::vector<int8_t> unweighted = ref_avg_codes (all, { 0u, 2u });
        for (uint32_t lane = 0u; lane < 4u; ++lane)
            CHECK (plain[lane] == unweighted[lane]);
        /* Lane 0 is (10, 30, 30) weighted against (10, 30) unweighted: the
           duplicate really does move the result toward the late snapshot. */
        CHECK (weighted[0] > unweighted[0]);
        CHECK (static_cast<int32_t> (weighted[0]) == 23);
        CHECK (static_cast<int32_t> (unweighted[0]) == 20);
        /* A one-entry list is a copy of that snapshot. */
        alignas (16) const uint32_t single[1] = { 2u };
        alignas (16) int8_t copied[4] = { 0 };
        REQUIRE (avg_swa (set, single, 1u, copied) == Status::OK);
        for (uint32_t lane = 0u; lane < 4u; ++lane)
            CHECK (copied[lane] == late.code[lane]);
    }

    SECTION ("index and length validation")
    {
        alignas (16) const uint32_t past[2] = { 0u, 3u };
        alignas (16) int8_t untouched[4] = { 9, 9, 9, 9 };
        CHECK (avg_swa (set, past, 2u, untouched) == Status::OUT_OF_RANGE);
        alignas (16) const uint32_t negative_like[2] = { 1u, 0xFFFFFFFFu };
        CHECK (avg_swa (set, negative_like, 2u, untouched)
               == Status::OUT_OF_RANGE);
        CHECK (avg_swa (set, past, 0u, untouched) == Status::OUT_OF_RANGE);
        CHECK (avg_swa (set, past, _S_avg_max_snapshots + 1u, untouched)
               == Status::OUT_OF_RANGE);
        CHECK (avg_swa (set, nullptr, 1u, untouched) == Status::BAD_LAYOUT);
        CHECK (avg_swa (set, past, 1u, nullptr) == Status::BAD_LAYOUT);
        CHECK (avg_swa (
                  set, reinterpret_cast<const uint32_t *> (
                           reinterpret_cast<const uint8_t *> (past) + 1u),
                  1u, untouched)
               == Status::BAD_LAYOUT);
        CHECK (avg_swa (set, past, 2u,
                        reinterpret_cast<int8_t *> (untouched) + 1)
               == Status::BAD_LAYOUT);
        SnapshotSet invalid;
        CHECK (avg_swa (invalid, past, 1u, untouched) == Status::BAD_LAYOUT);
        for (const int8_t code : untouched)
            CHECK (code == 9);
    }

    SECTION ("a first-out-of-range index aborts before any lane is written")
    {
        alignas (16) const uint32_t indices[4] = { 0u, 1u, 2u, 2u };
        alignas (16) const uint32_t broken[4] = { 0u, 1u, 2u, 9u };
        alignas (16) int8_t good[4] = { 0 };
        REQUIRE (avg_swa (set, indices, 4u, good) == Status::OK);
        alignas (16) int8_t refused[4] = { 1, 2, 3, 4 };
        CHECK (avg_swa (set, broken, 4u, refused) == Status::OUT_OF_RANGE);
        CHECK (refused[0] == 1);
        CHECK (refused[3] == 4);
    }
}

TEST_CASE ("avg_ema_step folds the documented difference form step by step")
{
    W seed_stream, target_stream;
    seed_stream.fill ({ 20, -7, 0, 127, -127 });
    target_stream.fill ({ 50, -60, 0, 100, -100 });
    alignas (16) int16_t lanes[8] = { 0 };
    EmaState state;
    REQUIRE (avg_ema_open (state, lanes, 5u, 32000) == Status::OK);
    CHECK (state._M_valid == 1u);
    CHECK (state._M_count == 5u);
    CHECK (state._M_decay_q15 == 32000);

    SECTION ("the seed is the plain code widening and is unbiased at n = 0")
    {
        REQUIRE (avg_ema_seed (state, seed_stream.code) == Status::OK);
        for (uint32_t lane = 0u; lane < 5u; ++lane)
            CHECK (state._M_lanes[lane]
                   == static_cast<int16_t> (ref_code_to_q15 (
                          seed_stream.code[lane])));
        CHECK (state._M_updates == 0u);
        CHECK (state._M_saturation_events == 0u);
        alignas (16) int16_t corrected[8] = { 0 };
        REQUIRE (avg_ema_bias_corrected (state, corrected) == Status::OK);
        for (uint32_t lane = 0u; lane < 5u; ++lane)
            CHECK (corrected[lane] == state._M_lanes[lane]);
    }

    SECTION ("each step is exactly s += (t - s) * (1 - decay) / 2^15")
    {
        REQUIRE (avg_ema_seed (state, seed_stream.code) == Status::OK);
        std::vector<int16_t> held (state._M_lanes, state._M_lanes + 5);
        for (uint32_t step = 1u; step <= 6u; ++step)
        {
            INFO ("ema step " << step);
            REQUIRE (avg_ema_step (state, target_stream.code) == Status::OK);
            CHECK (state._M_updates == step);
            std::vector<int16_t> want;
            for (uint32_t lane = 0u; lane < 5u; ++lane)
            {
                bool saturated = false;
                const int32_t next = ref_ema_next (
                    held[lane],
                    static_cast<int32_t> (ref_code_to_q15 (
                        target_stream.code[lane])),
                    32000, saturated);
                CHECK (saturated == false);
                want.push_back (static_cast<int16_t> (next));
            }
            for (uint32_t lane = 0u; lane < 5u; ++lane)
                CHECK (state._M_lanes[lane] == want[lane]);
            held = want;
        }
        /* The decay pulls every lane toward its target but never reaches it. */
        const auto iabs = [](const int64_t v) -> int64_t {
            return v < 0 ? -v : v;
        };
        for (uint32_t lane = 0u; lane < 5u; ++lane)
        {
            const int64_t target
                = ref_code_to_q15 (target_stream.code[lane]);
            const int64_t origin
                = ref_code_to_q15 (seed_stream.code[lane]);
            if (target == origin)
                continue; /* a converged lane has no distance left to close */
            INFO ("lane " << lane);
            CHECK (iabs (static_cast<int64_t> (state._M_lanes[lane]) - target)
                   < iabs (origin - target));
            /* And it never overshoots: the sign of the residual is preserved. */
            CHECK ((static_cast<int64_t> (state._M_lanes[lane]) - target)
                   * (origin - target)
                   > 0);
        }
    }

    SECTION ("a lane already at its target does not move at all")
    {
        REQUIRE (avg_ema_seed (state, target_stream.code) == Status::OK);
        const std::vector<int16_t> frozen (state._M_lanes,
                                           state._M_lanes + 5);
        for (uint32_t step = 0u; step < 4u; ++step)
        {
            REQUIRE (avg_ema_step (state, target_stream.code) == Status::OK);
            for (uint32_t lane = 0u; lane < 5u; ++lane)
                CHECK (state._M_lanes[lane] == frozen[lane]);
        }
        CHECK (state._M_updates == 4u);
        CHECK (state._M_saturation_events == 0u);
    }

    SECTION ("decay 0 jumps straight to the target, decay 1 freezes it")
    {
        REQUIRE (avg_ema_open (state, lanes, 5u, 0) == Status::OK);
        REQUIRE (avg_ema_seed (state, seed_stream.code) == Status::OK);
        REQUIRE (avg_ema_step (state, target_stream.code) == Status::OK);
        for (uint32_t lane = 0u; lane < 5u; ++lane)
            CHECK (state._M_lanes[lane]
                   == static_cast<int16_t> (ref_code_to_q15 (
                          target_stream.code[lane])));

        REQUIRE (avg_ema_set_decay (state, _S_q15_max) == Status::OK);
        const std::vector<int16_t> before (state._M_lanes,
                                           state._M_lanes + 5);
        REQUIRE (avg_ema_step (state, seed_stream.code) == Status::OK);
        for (uint32_t lane = 0u; lane < 5u; ++lane)
            CHECK (state._M_lanes[lane] == before[lane]);
        CHECK (state._M_updates == 2u);
    }

    SECTION ("the difference form can never cross a rail, so it cannot wrap")
    {
        /* The state lane is caller-owned: start it exactly on the int16 rails. */
        REQUIRE (avg_ema_open (state, lanes, 5u, 0) == Status::OK);
        const int16_t lane_held_start[5] = { -32768, 32767, -32768, 32767, 0 };
        for (uint32_t lane = 0u; lane < 5u; ++lane)
            state._M_lanes[lane] = lane_held_start[lane];
        alignas (16) int8_t codes[5] = { 127, -127, -127, 127, 0 };
        REQUIRE (avg_ema_step (state, codes) == Status::OK);
        bool any_saturated = false;
        for (uint32_t lane = 0u; lane < 5u; ++lane)
        {
            bool saturated = false;
            const int32_t want = ref_ema_next (
                lane_held_start[lane],
                static_cast<int32_t> (ref_code_to_q15 (codes[lane])), 0,
                saturated);
            INFO ("rail lane " << lane);
            CHECK (state._M_lanes[lane] == static_cast<int16_t> (want));
            CHECK (saturated == false);
            CHECK (state._M_lanes[lane] >= -32768);
            CHECK (state._M_lanes[lane] <= 32767);
            any_saturated = any_saturated || saturated;
        }
        /* (t - s) * (1 - d) is bounded by the remaining distance, so the rail is
           approached, never overshot: the documented saturation counter stays 0.
           Even a full Q15 step from the rail clamps inside the range. */
        CHECK (any_saturated == false);
        CHECK (state._M_saturation_events == 0u);
        CHECK (state._M_lanes[0] == -1);
        CHECK (state._M_lanes[1] == -1);
        CHECK (state._M_lanes[2] == -32766);
        CHECK (state._M_lanes[3] == 32766);
        CHECK (state._M_lanes[4] == 0);
        alignas (16) int8_t read[5] = { 0 };
        REQUIRE (avg_ema_read (state, read) == Status::OK);
        for (uint32_t lane = 0u; lane < 5u; ++lane)
        {
            CHECK (read[lane] == ref_simd_q15_to_code (state._M_lanes[lane]));
            CHECK (read[lane] != -128);
        }
        CHECK (read[2] == -127); /* the int16 rail narrows to -127, not -128 */
        CHECK (read[3] == 127);
        CHECK (read[4] == 0);
    }

    SECTION ("bias correction divides by 1 - decay^n")
    {
        REQUIRE (avg_ema_open (state, lanes, 5u, 32572) == Status::OK);
        REQUIRE (avg_ema_seed (state, seed_stream.code) == Status::OK);
        std::vector<int16_t> held (state._M_lanes, state._M_lanes + 5);
        alignas (16) int16_t corrected[8] = { 0 };
        for (uint32_t update = 1u; update <= 5u; ++update)
        {
            REQUIRE (avg_ema_step (state, target_stream.code) == Status::OK);
            held.assign (state._M_lanes, state._M_lanes + 5);
            REQUIRE (avg_ema_bias_corrected (state, corrected) == Status::OK);
            const std::vector<int16_t> want
                = ref_bias_corrected (held, 32572, update);
            for (uint32_t lane = 0u; lane < 5u; ++lane)
                CHECK (corrected[lane] == want[lane]);
            /* The correction only ever divides by a factor >= 1 - decay, so a
               seed-sized lane grows with the update count, never shrinks. */
            const int32_t gained = corrected[0] < 0 ? -corrected[0] : corrected[0];
            const int32_t raw = held[0] < 0 ? static_cast<int32_t> (-held[0])
                                           : static_cast<int32_t> (held[0]);
            if (update > 1u)
                CHECK (gained >= raw);
        }
        /* One update at decay 1.0 leaves 1 - decay^n == 0: the copy path. */
        REQUIRE (avg_ema_open (state, lanes, 5u, _S_q15_max) == Status::OK);
        REQUIRE (avg_ema_seed (state, seed_stream.code) == Status::OK);
        state._M_updates = 1u;
        REQUIRE (avg_ema_bias_corrected (state, corrected) == Status::OK);
        for (uint32_t lane = 0u; lane < 5u; ++lane)
            CHECK (corrected[lane] == state._M_lanes[lane]);
        /* Two updates make the factor positive (decay^2 rounds one LSB below the
           rail), so the reciprocal path runs and only a rail-sized lane loses
           one LSB. */
        state._M_updates = 2u;
        REQUIRE (avg_ema_bias_corrected (state, corrected) == Status::OK);
        const std::vector<int16_t> raw (state._M_lanes, state._M_lanes + 5);
        const std::vector<int16_t> want
            = ref_bias_corrected (raw, _S_q15_max, 2u);
        for (uint32_t lane = 0u; lane < 5u; ++lane)
            CHECK (corrected[lane] == want[lane]);
        CHECK (want[3] == raw[3] - 1);
        CHECK (want[4] == raw[4] + 1);
        CHECK (want[2] == raw[2]); /* a zero lane is its own correction */
    }

    SECTION ("open, seed and step validate their inputs")
    {
        CHECK (avg_ema_open (state, nullptr, 4u, 100) == Status::BAD_LAYOUT);
        CHECK (avg_ema_open (state, lanes, 0u, 100) == Status::OUT_OF_RANGE);
        CHECK (avg_ema_open (state, lanes, _S_avg_max_count + 1u, 100)
               == Status::OUT_OF_RANGE);
        CHECK (avg_ema_open (state, lanes, 4u, -1) == Status::OUT_OF_RANGE);
        CHECK (avg_ema_open (state, lanes, 4u, _S_q15_one)
               == Status::OUT_OF_RANGE);
        CHECK (avg_ema_open (state, lanes, 4u, _S_q15_max) == Status::OK);
        CHECK (state._M_valid == 1u);
        CHECK (avg_ema_open (state, reinterpret_cast<int16_t *> (lanes) + 1, 4u,
                             100)
               == Status::BAD_LAYOUT);
        CHECK (state._M_valid == 0u);

        REQUIRE (avg_ema_open (state, lanes, 5u, 32000) == Status::OK);
        CHECK (avg_ema_seed (state, nullptr) == Status::BAD_LAYOUT);
        CHECK (avg_ema_seed (state, reinterpret_cast<const int8_t *> (
                                   seed_stream.code)
                                 + 1)
               == Status::BAD_LAYOUT);
        CHECK (avg_ema_step (state, nullptr) == Status::BAD_LAYOUT);
        CHECK (avg_ema_step (state, reinterpret_cast<const int8_t *> (
                                    target_stream.code)
                                  + 1)
               == Status::BAD_LAYOUT);
        EmaState closed;
        alignas (16) int8_t read[8] = { 0 };
        alignas (16) int16_t corrected[8] = { 0 };
        CHECK (avg_ema_seed (closed, seed_stream.code) == Status::BAD_LAYOUT);
        CHECK (avg_ema_step (closed, target_stream.code) == Status::BAD_LAYOUT);
        CHECK (avg_ema_read (closed, read) == Status::BAD_LAYOUT);
        CHECK (avg_ema_bias_corrected (closed, corrected) == Status::BAD_LAYOUT);
        CHECK (avg_ema_bias_corrected (state, nullptr) == Status::BAD_LAYOUT);
        CHECK (avg_ema_bias_corrected (
                  state, reinterpret_cast<int16_t *> (
                             reinterpret_cast<uint8_t *> (corrected) + 1u))
               == Status::BAD_LAYOUT);
        CHECK (avg_ema_read (state, reinterpret_cast<int8_t *> (read) + 1)
               == Status::BAD_LAYOUT);
        CHECK (avg_ema_set_decay (closed, 100) == Status::BAD_LAYOUT);
        CHECK (avg_ema_set_decay (state, -1) == Status::OUT_OF_RANGE);
        CHECK (avg_ema_set_decay (state, _S_q15_one) == Status::OUT_OF_RANGE);
        CHECK (state._M_decay_q15 == 32000);
        /* A state whose count exceeds the module maximum is refused by every
           lane-touching entry point. */
        REQUIRE (avg_ema_open (state, lanes, 5u, 32000) == Status::OK);
        state._M_count = _S_avg_max_count + 1u;
        CHECK (state.validate () == Status::OUT_OF_RANGE);
        CHECK (avg_ema_seed (state, seed_stream.code) == Status::OUT_OF_RANGE);
        CHECK (avg_ema_step (state, target_stream.code) == Status::OUT_OF_RANGE);
        CHECK (avg_ema_read (state, read) == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("avg_round_to_codes uses the SIMD narrowing and never emits -128")
{
    alignas (16) int16_t lanes[9];
    const int16_t values[9] = { -32768, -16384, -16383, -1, 0, 1, 16384, 32766,
                                32767 };
    for (uint32_t i = 0u; i < 9u; ++i)
        lanes[i] = values[i];
    alignas (16) int8_t codes[9] = { 0 };

    SECTION ("every lane narrows the way train_simd promises")
    {
        REQUIRE (avg_round_to_codes (lanes, codes, 9u) == Status::OK);
        for (uint32_t i = 0u; i < 9u; ++i)
        {
            INFO ("lane " << i << " value " << values[i]);
            CHECK (codes[i] == ref_simd_q15_to_code (values[i]));
            CHECK (codes[i] != -128);
            CHECK (codes[i] >= -127);
            CHECK (codes[i] <= 127);
        }
        CHECK (codes[0] == -127); /* the int16 rail clamps, it does not alias */
        CHECK (codes[3] == 0);
        CHECK (codes[4] == 0);
        CHECK (codes[5] == 0);
        CHECK (codes[7] == 127);
        CHECK (codes[8] == 127);
    }

    SECTION ("the exact negative tie separates the two documented rules")
    {
        REQUIRE (avg_round_to_codes (lanes, codes, 9u) == Status::OK);
        /* The SIMD helper rounds the magnitude half-away-from-zero, so -16384
           (-0.5 exactly) becomes -64 while the fixed-point q15_to_code, which
           rounds half toward +infinity, returns -63 for the same lane. */
        CHECK (codes[1] == -64);
        CHECK (q15_to_code (-16384) == -63);
        CHECK (codes[6] == q15_to_code (16384)); /* positive ties agree */
    }

    SECTION ("validation")
    {
        CHECK (avg_round_to_codes (nullptr, codes, 4u) == Status::BAD_LAYOUT);
        CHECK (avg_round_to_codes (lanes, nullptr, 4u) == Status::BAD_LAYOUT);
        CHECK (avg_round_to_codes (lanes, codes, _S_avg_max_count + 1u)
               == Status::OUT_OF_RANGE);
        CHECK (avg_round_to_codes (reinterpret_cast<const int16_t *> (lanes) + 1,
                                   codes, 4u)
               == Status::BAD_LAYOUT);
        CHECK (avg_round_to_codes (lanes,
                                   reinterpret_cast<int8_t *> (codes) + 1, 4u)
               == Status::BAD_LAYOUT);
        /* count 0 is a legal empty pass. */
        CHECK (avg_round_to_codes (lanes, codes, 0u) == Status::OK);
    }
}

TEST_CASE ("AvgGate accepts a snapshot only when the metric improves enough")
{
    AvgGate gate;
    bool accept = false;

    SECTION ("the first sample is always accepted")
    {
        REQUIRE (avg_gate_consider (gate, 5000, accept) == Status::OK);
        CHECK (accept);
        CHECK (gate._M_best_q15 == 5000);
        CHECK (gate._M_accepted == 1u);
        CHECK (gate._M_rejected == 0u);
        CHECK (gate._M_compared == 1u);
        CHECK (gate._M_has_best == 1u);
    }

    SECTION ("a lower loss wins by at least min_delta in the default direction")
    {
        REQUIRE (avg_gate_consider (gate, 1000, accept) == Status::OK);
        CHECK (accept);
        CHECK (gate._M_best_q15 == 1000);
        /* Exactly min_delta improvement: accepted. */
        REQUIRE (avg_gate_consider (gate, 999, accept) == Status::OK);
        CHECK (accept);
        CHECK (gate._M_best_q15 == 999);
        /* min_delta - 1 (here a tie): rejected, and the best value is kept. */
        REQUIRE (avg_gate_consider (gate, 999, accept) == Status::OK);
        CHECK (!accept);
        CHECK (gate._M_best_q15 == 999);
        REQUIRE (avg_gate_consider (gate, 1000, accept) == Status::OK);
        CHECK (!accept);
        /* A full-scale improvement moves the best value and is accepted. */
        REQUIRE (avg_gate_consider (gate, 500, accept) == Status::OK);
        CHECK (accept);
        CHECK (gate._M_best_q15 == 500);
        CHECK (gate._M_accepted == 3u);
        CHECK (gate._M_rejected == 2u);
        CHECK (gate._M_compared == 5u);
        /* A worse sample never overwrites the best. */
        REQUIRE (avg_gate_consider (gate, 5000, accept) == Status::OK);
        CHECK (!accept);
        CHECK (gate._M_best_q15 == 500);
        CHECK (gate._M_rejected == 3u);
    }

    SECTION ("min_delta 0 accepts a tie")
    {
        gate._M_min_delta_q15 = 0;
        REQUIRE (avg_gate_consider (gate, -100, accept) == Status::OK);
        CHECK (accept);
        REQUIRE (avg_gate_consider (gate, -100, accept) == Status::OK);
        CHECK (accept);
        CHECK (gate._M_accepted == 2u);
    }

    SECTION ("higher_is_better flips the comparison")
    {
        gate._M_higher_is_better = 1u;
        gate._M_min_delta_q15 = 10;
        REQUIRE (avg_gate_consider (gate, 100, accept) == Status::OK);
        CHECK (accept);
        REQUIRE (avg_gate_consider (gate, 109, accept) == Status::OK);
        CHECK (!accept);
        REQUIRE (avg_gate_consider (gate, 110, accept) == Status::OK);
        CHECK (accept);
        CHECK (gate._M_best_q15 == 110);
        REQUIRE (avg_gate_consider (gate, 90, accept) == Status::OK);
        CHECK (!accept);
    }

    SECTION ("a negative min_delta is refused before any counter moves")
    {
        gate._M_min_delta_q15 = -5;
        CHECK (avg_gate_consider (gate, 10, accept) == Status::OUT_OF_RANGE);
        CHECK (gate._M_compared == 0u);
        CHECK (gate._M_accepted == 0u);
    }

    SECTION ("reset clears the trend but keeps the configured direction")
    {
        gate._M_min_delta_q15 = 7;
        gate._M_higher_is_better = 1u;
        REQUIRE (avg_gate_consider (gate, 100, accept) == Status::OK);
        REQUIRE (avg_gate_consider (gate, 50, accept) == Status::OK);
        CHECK (gate._M_has_best == 1u);
        avg_gate_reset (gate);
        CHECK (gate._M_best_q15 == 0);
        CHECK (gate._M_accepted == 0u);
        CHECK (gate._M_rejected == 0u);
        CHECK (gate._M_compared == 0u);
        CHECK (gate._M_has_best == 0u);
        CHECK (gate._M_pad == 0u);
        CHECK (gate._M_min_delta_q15 == 7);
        CHECK (gate._M_higher_is_better == 1u);
        /* A reset gate accepts its next sample as the new first one. */
        REQUIRE (avg_gate_consider (gate, 5000, accept) == Status::OK);
        CHECK (accept);
        CHECK (gate._M_best_q15 == 5000);
        /* A negative threshold cannot survive a reset. */
        gate._M_min_delta_q15 = -1;
        avg_gate_reset (gate);
        CHECK (gate._M_min_delta_q15 == 1);
        CHECK (gate.validate () == Status::OK);
    }

    SECTION ("a long metric series traces the independent gate")
    {
        const std::vector<int32_t> metrics
            = { 900, 900, 850, 860, 800, 800, 799, 1000, 700, 710 };
        const GateTrace want = ref_gate (metrics, 1, false);
        AvgGate traced;
        for (const int32_t metric : metrics)
        {
            bool answer = false;
            REQUIRE (avg_gate_consider (traced, metric, answer) == Status::OK);
        }
        CHECK (traced._M_accepted
               == static_cast<uint32_t> (want._M_accepted.size ()));
        CHECK (traced._M_rejected == want._M_rejected);
        CHECK (traced._M_compared == want._M_compared);
        CHECK (traced._M_best_q15 == want._M_best);
        /* The improving samples are 900, 850, 800, 799 and 700; a tie and every
           jitter against the best are dropped. */
        CHECK (want._M_accepted.size () == 5u);
        const uint32_t accepted[5] = { 0u, 2u, 4u, 6u, 8u };
        for (uint32_t i = 0u; i < 5u; ++i)
            CHECK (want._M_accepted[i] == accepted[i]);
        CHECK (want._M_rejected == 5u);
        CHECK (want._M_compared == 10u);
        CHECK (want._M_best == 700);
    }
}

TEST_CASE ("avg_gate_record turns metrics into a deterministic index list")
{
    alignas (16) int32_t metrics[6] = { 100, 90, 95, 80, 80, 70 };
    alignas (16) uint32_t indices[6] = { 0 };
    uint32_t used = 99u;
    AvgGate gate;
    gate._M_min_delta_q15 = 1;

    SECTION ("improving snapshots are indexed, jitter is dropped")
    {
        REQUIRE (avg_gate_record (gate, metrics, 6u, indices, used)
                 == Status::OK);
        CHECK (used == 4u);
        const uint32_t want[4] = { 0u, 1u, 3u, 5u };
        for (uint32_t i = 0u; i < used; ++i)
            CHECK (indices[i] == want[i]);
        CHECK (gate._M_accepted == 4u);
        CHECK (gate._M_rejected == 2u);
        CHECK (gate._M_compared == 6u);
        CHECK (gate._M_best_q15 == 70);
    }

    SECTION ("the trace is reproducible from a reset gate")
    {
        const GateTrace want = ref_gate (
            std::vector<int32_t> (metrics, metrics + 6), 1, false);
        uint32_t second_used = 0u;
        alignas (16) uint32_t again[6] = { 0 };
        AvgGate fresh;
        fresh._M_min_delta_q15 = 1;
        REQUIRE (avg_gate_record (fresh, metrics, 6u, again, second_used)
                 == Status::OK);
        REQUIRE (second_used
                 == static_cast<uint32_t> (want._M_accepted.size ()));
        for (uint32_t i = 0u; i < second_used; ++i)
            CHECK (again[i] == want._M_accepted[i]);
    }

    SECTION ("higher_is_better with a margin drops the jitter in between")
    {
        AvgGate accuracy;
        accuracy._M_higher_is_better = 1u;
        accuracy._M_min_delta_q15 = 5;
        alignas (16) int32_t scores[5] = { 10, 14, 15, 21, 20 };
        uint32_t picked = 0u;
        alignas (16) uint32_t list[5] = { 0 };
        REQUIRE (avg_gate_record (accuracy, scores, 5u, list, picked)
                 == Status::OK);
        /* 10 seeds the gate, 14 is only +4, 15 is exactly +5, 21 improves again
           and 20 is below the best. */
        CHECK (picked == 3u);
        CHECK (list[0] == 0u);
        CHECK (list[1] == 2u);
        CHECK (list[2] == 3u);
        CHECK (accuracy._M_best_q15 == 21);
        CHECK (accuracy._M_rejected == 2u);
        CHECK (accuracy._M_accepted == 3u);
        const GateTrace want
            = ref_gate (std::vector<int32_t> (scores, scores + 5), 5, true);
        CHECK (static_cast<uint32_t> (want._M_accepted.size ()) == picked);
        for (uint32_t i = 0u; i < picked; ++i)
            CHECK (list[i] == want._M_accepted[i]);
    }

    SECTION ("validation order and the empty-list contract")
    {
        CHECK (avg_gate_record (gate, nullptr, 6u, indices, used)
               == Status::BAD_LAYOUT);
        CHECK (avg_gate_record (gate, metrics, 6u, nullptr, used)
               == Status::BAD_LAYOUT);
        CHECK (avg_gate_record (gate, metrics, 0u, indices, used)
               == Status::OUT_OF_RANGE);
        CHECK (avg_gate_record (gate, metrics, _S_avg_max_snapshots + 1u,
                                indices, used)
               == Status::OUT_OF_RANGE);
        CHECK (avg_gate_record (
                  gate, reinterpret_cast<const int32_t *> (
                            reinterpret_cast<const uint8_t *> (metrics) + 1u),
                  6u, indices, used)
               == Status::BAD_LAYOUT);
        CHECK (avg_gate_record (
                  gate, metrics, 6u,
                  reinterpret_cast<uint32_t *> (
                      reinterpret_cast<uint8_t *> (indices) + 1u),
                  used)
               == Status::BAD_LAYOUT);
        /* A broken gate is caught before the pointer checks. */
        gate._M_min_delta_q15 = -1;
        CHECK (avg_gate_record (gate, nullptr, 0u, nullptr, used)
               == Status::OUT_OF_RANGE);
        gate._M_min_delta_q15 = 1;
        /* The first sample is always accepted, so an all-reject trace needs a
           pre-seeded gate; then the caller gets OUT_OF_RANGE with used 0. */
        AvgGate seeded;
        seeded._M_min_delta_q15 = 1;
        seeded._M_has_best = 1u;
        seeded._M_best_q15 = -100000;
        alignas (16) int32_t worse[2] = { 10, 20 };
        CHECK (avg_gate_record (seeded, worse, 2u, indices, used)
               == Status::OUT_OF_RANGE);
        CHECK (used == 0u);
        CHECK (seeded._M_rejected == 2u);
        CHECK (seeded._M_accepted == 0u);
        CHECK (seeded._M_compared == 2u);
        CHECK (seeded._M_best_q15 == -100000);
    }
}

TEST_CASE ("avg_check_divergence counts sign flips against the average")
{
    W average, calm, gentle, flipped, opposite;
    average.fill ({ 5, -3, 0, 7, -2 });
    calm.fill ({ 4, -2, 0, 6, -1 });
    gentle.fill ({ 6, -4, 0, 8, -1 });
    flipped.fill ({ -5, 3, 0, -7, 2 });
    opposite.fill ({ 5, -3, 0, 7, 9 });
    alignas (16) const int8_t *two[2] = { calm.code, gentle.code };
    alignas (16) const int8_t *three[3]
        = { calm.code, flipped.code, opposite.code };
    SnapshotSet calm_set, mixed_set;
    REQUIRE (avg_snapshots_open (calm_set, two, 2u, 5u) == Status::OK);
    REQUIRE (avg_snapshots_open (mixed_set, three, 3u, 5u) == Status::OK);
    DivergenceReport report;

    SECTION ("a same-basin set passes and reports zero flips")
    {
        REQUIRE (avg_check_divergence (average.code, calm_set, 32767, report)
                 == Status::OK);
        CHECK (report._M_lanes_compared == 10u);
        CHECK (report._M_lanes_diverging == 0u);
        CHECK (report._M_worst_snapshot == 0u);
        CHECK (report._M_worst_snapshot_lanes == 0u);
        CHECK (report._M_worst_fraction_q15 == 0);
        CHECK (report._M_limit_q15 == 32767);
    }

    SECTION ("the sign-flipped snapshot is named with its exact fraction")
    {
        const std::vector<int8_t> avg = average.codes ();
        const std::vector<std::vector<int8_t>> snaps
            = { calm.codes (), flipped.codes (), opposite.codes () };
        const std::vector<uint32_t> want = ref_divergence (avg, snaps);
        REQUIRE (avg_check_divergence (average.code, mixed_set, 32767, report)
                 == Status::OK);
        CHECK (report._M_lanes_compared == 15u);
        CHECK (report._M_lanes_diverging
               == want[0] + want[1] + want[2]);
        CHECK (report._M_worst_snapshot == 1u);
        CHECK (report._M_worst_snapshot_lanes == want[1]);
        /* Lane 2 is zero on both sides: a code of 0 carries no sign. */
        CHECK (want[1] == 4u);
        CHECK (report._M_worst_fraction_q15 == ref_ratio_q15 (want[1], 5));
        CHECK (report._M_worst_fraction_q15 == 26214);
        CHECK (report._M_limit_q15 == 32767);
    }

    SECTION ("crossing the limit fails with the report still filled")
    {
        const int32_t exact = ref_ratio_q15 (4, 5);
        REQUIRE (avg_check_divergence (average.code, mixed_set, exact, report)
                 == Status::OK); /* the limit is inclusive: fraction > limit */
        CHECK (avg_check_divergence (average.code, mixed_set, exact - 1, report)
               == Status::OUT_OF_RANGE);
        CHECK (report._M_worst_snapshot_lanes == 4u);
        CHECK (report._M_worst_fraction_q15 == exact);
        CHECK (report._M_limit_q15 == exact - 1);
        CHECK (avg_check_divergence (average.code, mixed_set, 0, report)
               == Status::OUT_OF_RANGE);
    }

    SECTION ("zero lanes are exempt in both directions")
    {
        W zeroed, signed_streams;
        zeroed.fill ({ 0, 0, 0, 0, 0 });
        signed_streams.fill ({ 9, -9, 0, 9, -9 });
        alignas (16) const int8_t *pair[2] = { signed_streams.code,
                                               signed_streams.code };
        SnapshotSet set;
        REQUIRE (avg_snapshots_open (set, pair, 2u, 5u) == Status::OK);
        REQUIRE (avg_check_divergence (zeroed.code, set, 0, report) == Status::OK);
        CHECK (report._M_lanes_diverging == 0u);
        CHECK (report._M_worst_fraction_q15 == 0);
        /* And the converse: an averaged-away lane against a signed snapshot. */
        CHECK (avg_lane_diverges (0, 5) == false);
        CHECK (avg_lane_diverges (0, -5) == false);
        CHECK (avg_lane_diverges (5, 0) == false);
        CHECK (avg_lane_diverges (-5, 0) == false);
        CHECK (avg_lane_diverges (5, 0) == false);
        CHECK (avg_lane_diverges (5, -5) == true);
        CHECK (avg_lane_diverges (-5, 5) == true);
        CHECK (avg_lane_diverges (-5, -5) == false);
    }

    SECTION ("the worst snapshot is the first one to reach the maximum")
    {
        W twin;
        twin.fill ({ -5, 3, 0, -7, 2 });
        alignas (16) const int8_t *pair[2] = { twin.code, flipped.code };
        SnapshotSet set;
        REQUIRE (avg_snapshots_open (set, pair, 2u, 5u) == Status::OK);
        REQUIRE (avg_check_divergence (average.code, set, 32767, report)
                 == Status::OK);
        CHECK (report._M_worst_snapshot == 0u);
        CHECK (report._M_worst_snapshot_lanes == 4u);
        CHECK (report._M_lanes_diverging == 8u);
    }

    SECTION ("validation")
    {
        CHECK (avg_check_divergence (nullptr, mixed_set, 100, report)
               == Status::BAD_LAYOUT);
        CHECK (avg_check_divergence (
                  reinterpret_cast<const int8_t *> (average.code) + 1, mixed_set,
                  100, report)
               == Status::BAD_LAYOUT);
        SnapshotSet invalid;
        CHECK (avg_check_divergence (average.code, invalid, 100, report)
               == Status::BAD_LAYOUT);
        CHECK (avg_check_divergence (average.code, mixed_set, -1, report)
               == Status::OUT_OF_RANGE);
        CHECK (report._M_limit_q15 == -1); /* echoed before the range check */
        CHECK (report._M_lanes_compared == 0u);
        CHECK (avg_check_divergence (average.code, mixed_set, _S_q15_max + 1,
                                    report)
               == Status::OUT_OF_RANGE);
        /* A fresh report is written even on the rejection paths. */
        CHECK (avg_check_divergence (average.code, mixed_set, 0, report)
               == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("avg_code_drift measures how far an average sits from its newest input")
{
    W average, source, identical;
    average.fill ({ 5, -3, 0, 7, -2 });
    source.fill ({ 4, -2, 0, 6, -1 });
    identical.fill ({ 5, -3, 0, 7, -2 });
    uint32_t drift = 99u;
    int32_t maximum = -1;

    SECTION ("counts and the largest code step")
    {
        const std::pair<uint32_t, int32_t> want
            = ref_drift (average.codes (), source.codes ());
        REQUIRE (avg_code_drift (average.code, source.code, 5u, drift, maximum)
                 == Status::OK);
        CHECK (drift == want.first);
        CHECK (maximum == want.second);
        CHECK (drift == 4u);
        CHECK (maximum == 1);
    }

    SECTION ("identical streams are a no-op average")
    {
        REQUIRE (avg_code_drift (average.code, identical.code, 5u, drift,
                                 maximum)
                 == Status::OK);
        CHECK (drift == 0u);
        CHECK (maximum == 0);
    }

    SECTION ("a rail-to-rail gap reports the whole code span")
    {
        W high, low;
        high.fill ({ 127, 0, 127, -127, 0 });
        low.fill ({ -127, 0, -127, 127, 0 });
        REQUIRE (avg_code_drift (high.code, low.code, 5u, drift, maximum)
                 == Status::OK);
        CHECK (drift == 3u);
        CHECK (maximum == 254);
        const std::pair<uint32_t, int32_t> want
            = ref_drift (high.codes (), low.codes ());
        CHECK (std::make_pair (drift, maximum) == want);
    }

    SECTION ("zero lanes and validation")
    {
        REQUIRE (avg_code_drift (average.code, source.code, 0u, drift, maximum)
                 == Status::OK);
        CHECK (drift == 0u);
        CHECK (maximum == 0);
        CHECK (avg_code_drift (nullptr, source.code, 5u, drift, maximum)
               == Status::BAD_LAYOUT);
        CHECK (avg_code_drift (average.code, nullptr, 5u, drift, maximum)
               == Status::BAD_LAYOUT);
        CHECK (avg_code_drift (average.code, source.code,
                               _S_avg_max_count + 1u, drift, maximum)
               == Status::OUT_OF_RANGE);
        /* The boundary count needs a buffer that actually spans it; the 64-byte
           `W` fixture is only valid for the short cases above. */
        const std::vector<int8_t> wide (_S_avg_max_count, 3);
        const std::vector<int8_t> wide_shifted (_S_avg_max_count, 4);
        CHECK (avg_code_drift (wide.data (), wide_shifted.data (),
                               _S_avg_max_count, drift, maximum)
               == Status::OK);
        CHECK (drift == _S_avg_max_count);
        CHECK (maximum == 1);
    }
}

TEST_CASE ("avg_average_set averages a parameter set and reports the pass")
{
    W t0, t1;
    t0.fill ({ 10, -20, 30 });
    t1.fill ({ -5, 5, 0 });
    W s0, s1, s2;
    s0.fill ({ 40, -40, 0 });
    s1.fill ({ 8, 12, -16 });
    s2.fill ({ -2, 2, 2 });
    /* The destination code streams are separate buffers from the snapshots: a
       tensor that doubles as its own snapshot stream would alias. */
    W u0, u1, u2;
    u0.fill ({ 16, -16, 16 });
    u1.fill ({ -8, 4, -2 });
    u2.fill ({ 2, 2, 2 });
    alignas (16) Trainable params[2];
    params[0] = view (t0, Shape::vector (3));
    params[1] = view (t1, Shape::vector (3));
    ParamSet destination;
    REQUIRE (avg_params_open (destination, params, 2u) == Status::OK);
    CHECK (destination._M_valid == 1u);
    CHECK (destination._M_tensors == 2u);
    CHECK (destination._M_lane_total == 6u);
    CHECK (destination.validate () == Status::OK);

    alignas (16) const int8_t *streams0[3] = { s0.code, s1.code, s2.code };
    alignas (16) const int8_t *streams1[3] = { u0.code, u1.code, u2.code };
    alignas (16) SnapshotSet sets[2];
    REQUIRE (avg_snapshots_open (sets[0], streams0, 3u, 3u) == Status::OK);
    REQUIRE (avg_snapshots_open (sets[1], streams1, 3u, 3u) == Status::OK);
    AveragingReport report;

    SECTION ("uniform averages every snapshot of every tensor")
    {
        REQUIRE (avg_average_set (sets, destination, AveragingMode::Uniform,
                                  nullptr, 0u, report)
                 == Status::OK);
        const std::vector<std::vector<int8_t>> group0
            = { s0.codes (), s1.codes (), s2.codes () };
        const std::vector<std::vector<int8_t>> group1
            = { u0.codes (), u1.codes (), u2.codes () };
        const std::vector<int8_t> want0 = ref_avg_codes (group0, { 0u, 1u, 2u });
        const std::vector<int8_t> want1 = ref_avg_codes (group1, { 0u, 1u, 2u });
        for (uint32_t lane = 0u; lane < 3u; ++lane)
        {
            CHECK (t0.code[lane] == want0[lane]);
            CHECK (t1.code[lane] == want1[lane]);
        }
        CHECK (report._M_mode
               == static_cast<uint8_t> (AveragingMode::Uniform));
        CHECK (report._M_tensors == 2u);
        CHECK (report._M_lanes == 6u);
        CHECK (report._M_snapshots_used == 3u);
        CHECK (report._M_gate_accepted == 0u);
        CHECK (report._M_gate_rejected == 0u);
        /* Drift is measured against the newest snapshot of each set. */
        const std::pair<uint32_t, int32_t> d0
            = ref_drift (t0.codes (), s2.codes ());
        const std::pair<uint32_t, int32_t> d1
            = ref_drift (t1.codes (), u2.codes ());
        CHECK (report._M_drifted_lanes == d0.first + d1.first);
        CHECK (report._M_max_code_drift == std::max (d0.second, d1.second));
        CHECK (report._M_drifted_lanes > 0u);
    }

    SECTION ("swa uses the caller's index list, duplicates and all")
    {
        alignas (16) const uint32_t indices[2] = { 0u, 0u };
        REQUIRE (avg_average_set (sets, destination, AveragingMode::Swa, indices,
                                  2u, report)
                 == Status::OK);
        const std::vector<int8_t> want0 = ref_avg_codes (
            { s0.codes (), s1.codes (), s2.codes () }, { 0u, 0u });
        const std::vector<int8_t> want1 = ref_avg_codes (
            { u0.codes (), u1.codes (), u2.codes () }, { 0u, 0u });
        for (uint32_t lane = 0u; lane < 3u; ++lane)
        {
            /* A duplicated index is an exact copy of that snapshot. */
            CHECK (t0.code[lane] == want0[lane]);
            CHECK (t0.code[lane] == s0.code[lane]);
            CHECK (t1.code[lane] == want1[lane]);
            CHECK (t1.code[lane] == u0.code[lane]);
        }
        CHECK (report._M_mode == static_cast<uint8_t> (AveragingMode::Swa));
        CHECK (report._M_snapshots_used == 2u);
        CHECK (report._M_lanes == 6u);
        /* Drift is still against the newest snapshot, so it is not zero here. */
        CHECK (report._M_drifted_lanes == ref_drift (t0.codes (), s2.codes ()).first
                                           + ref_drift (t1.codes (), u2.codes ()).first);
        CHECK (report._M_max_code_drift > 0);
    }

    SECTION ("ema is refused because the state lives outside a code pass")
    {
        CHECK (avg_average_set (sets, destination, AveragingMode::Ema, nullptr,
                                0u, report)
               == Status::UNSUPPORTED);
        CHECK (report._M_mode == static_cast<uint8_t> (AveragingMode::Ema));
        CHECK (report._M_tensors == 0u);
    }

    SECTION ("layout mismatches are caught per tensor")
    {
        alignas (16) const uint32_t picks[3] = { 0u, 1u, 2u };
        CHECK (avg_average_set (nullptr, destination, AveragingMode::Uniform,
                                nullptr, 0u, report)
               == Status::BAD_LAYOUT);
        CHECK (avg_average_set (sets, destination, AveragingMode::Uniform,
                                picks, 3u, report)
               == Status::BAD_LAYOUT); /* uniform with an index list */
        CHECK (avg_average_set (sets, destination, AveragingMode::Swa, nullptr,
                                3u, report)
               == Status::BAD_LAYOUT);
        CHECK (avg_average_set (sets, destination, AveragingMode::Swa, picks,
                                0u, report)
               == Status::BAD_LAYOUT);
        ParamSet closed;
        CHECK (avg_average_set (sets, closed, AveragingMode::Uniform, nullptr,
                                0u, report)
               == Status::BAD_LAYOUT);
        CHECK (avg_average_set (
                  reinterpret_cast<SnapshotSet *> (
                      reinterpret_cast<uint8_t *> (sets) + 1u),
                  destination, AveragingMode::Uniform, nullptr, 0u, report)
               == Status::BAD_LAYOUT);
        /* A lane-count mismatch between a set and the tensor it feeds. */
        alignas (16) SnapshotSet short_set;
        REQUIRE (avg_snapshots_open (short_set, streams0, 3u, 2u) == Status::OK);
        alignas (16) SnapshotSet mismatched[2];
        mismatched[0] = short_set;
        mismatched[1] = sets[1];
        CHECK (avg_average_set (mismatched, destination, AveragingMode::Uniform,
                                nullptr, 0u, report)
               == Status::BAD_LAYOUT);
        CHECK (report._M_tensors == 0u);
        /* An unbound set in the parallel array. */
        mismatched[0] = sets[0];
        mismatched[1] = SnapshotSet{};
        CHECK (avg_average_set (mismatched, destination, AveragingMode::Uniform,
                                nullptr, 0u, report)
               == Status::BAD_LAYOUT);
    }

    SECTION ("avg_params_open validates every tensor it measures")
    {
        ParamSet fresh;
        CHECK (avg_params_open (fresh, nullptr, 2u) == Status::BAD_LAYOUT);
        CHECK (avg_params_open (fresh, params, 0u) == Status::OUT_OF_RANGE);
        CHECK (avg_params_open (fresh, params, _S_avg_max_count + 1u)
               == Status::OUT_OF_RANGE);
        CHECK (avg_params_open (
                  fresh, reinterpret_cast<Trainable *> (
                             reinterpret_cast<uint8_t *> (params) + 1u),
                  2u)
               == Status::BAD_LAYOUT);
        alignas (16) Trainable broken[1];
        broken[0] = Trainable::make (nullptr, t0.grad, Shape::vector (3u),
                                     Flag::RequiresGrad);
        CHECK (avg_params_open (fresh, broken, 1u) == Status::BAD_LAYOUT);
        CHECK (fresh._M_valid == 0u);
        /* A tensor whose shape exceeds the element count the view holds. */
        alignas (16) Trainable ragged[1];
        ragged[0] = Trainable::make (t0.code, t0.grad, Shape::vector (4u),
                                     Flag::RequiresGrad);
        ragged[0]._M_count = 3u;
        CHECK (avg_params_open (fresh, ragged, 1u) == Status::BAD_LAYOUT);
        CHECK (avg_params_open (fresh, params, 2u) == Status::OK);
    }
}

TEST_CASE ("avg_gated_average averages only the improving snapshots")
{
    W a0, a1, a2;
    a0.fill ({ 10, -20, 30, 0 });
    a1.fill ({ 20, -10, 40, 8 });
    a2.fill ({ 30, 0, 50, 16 });
    W dst;
    dst.fill ({ 0, 0, 0, 0 });
    alignas (16) Trainable params[1];
    params[0] = view (dst, Shape::vector (4u));
    ParamSet destination;
    REQUIRE (avg_params_open (destination, params, 1u) == Status::OK);
    alignas (16) const int8_t *streams[3] = { a0.code, a1.code, a2.code };
    alignas (16) SnapshotSet sets[1];
    REQUIRE (avg_snapshots_open (sets[0], streams, 3u, 4u) == Status::OK);
    alignas (16) int32_t metrics[3] = { 100, 120, 90 };
    alignas (16) uint32_t scratch[3] = { 0 };
    AvgGate gate;
    gate._M_min_delta_q15 = 1;
    AveragingReport report;

    SECTION ("a snapshot that got worse is dropped from the average")
    {
        REQUIRE (avg_gated_average (sets, destination, gate, metrics, 3u, scratch,
                                    report)
                 == Status::OK);
        CHECK (gate._M_accepted == 2u);
        CHECK (gate._M_rejected == 1u);
        CHECK (report._M_gate_accepted == 2u);
        CHECK (report._M_gate_rejected == 1u);
        CHECK (report._M_mode == static_cast<uint8_t> (AveragingMode::Swa));
        CHECK (report._M_tensors == 1u);
        CHECK (report._M_lanes == 4u);
        CHECK (report._M_snapshots_used == 2u);
        CHECK (scratch[0] == 0u);
        CHECK (scratch[1] == 2u);
        const std::vector<int8_t> want
            = ref_avg_codes ({ a0.codes (), a1.codes (), a2.codes () },
                             { 0u, 2u });
        for (uint32_t lane = 0u; lane < 4u; ++lane)
            CHECK (dst.code[lane] == want[lane]);
        CHECK (report._M_drifted_lanes == ref_drift (dst.codes (),
                                                     a2.codes ()).first);
        CHECK (report._M_max_code_drift == ref_drift (dst.codes (),
                                                      a2.codes ()).second);
    }

    SECTION ("an improving tail accepts everything")
    {
        alignas (16) int32_t improving[3] = { 90, 80, 70 };
        REQUIRE (avg_gated_average (sets, destination, gate, improving, 3u,
                                    scratch, report)
                 == Status::OK);
        /* A reset gate carries no history from the previous section: every
           sample improves, so the average covers the whole set. */
        CHECK (gate._M_accepted == 3u);
        CHECK (gate._M_rejected == 0u);
        CHECK (report._M_gate_accepted == 3u);
        CHECK (report._M_snapshots_used == 3u);
        CHECK (scratch[0] == 0u);
        CHECK (scratch[1] == 1u);
        CHECK (scratch[2] == 2u);
        alignas (16) int8_t plain[4] = { 0 };
        REQUIRE (avg_uniform (sets[0], 0u, 3u, plain) == Status::OK);
        for (uint32_t lane = 0u; lane < 4u; ++lane)
            CHECK (dst.code[lane] == plain[lane]);
    }

    SECTION ("validation happens before the gate is consulted")
    {
        CHECK (avg_gated_average (nullptr, destination, gate, metrics, 3u,
                                  scratch, report)
               == Status::BAD_LAYOUT);
        CHECK (avg_gated_average (sets, destination, gate, nullptr, 3u, scratch,
                                  report)
               == Status::BAD_LAYOUT);
        CHECK (avg_gated_average (sets, destination, gate, metrics, 3u, nullptr,
                                  report)
               == Status::BAD_LAYOUT);
        CHECK (avg_gated_average (sets, destination, gate, metrics, 0u, scratch,
                                  report)
               == Status::OUT_OF_RANGE);
        CHECK (avg_gated_average (sets, destination, gate, metrics,
                                  _S_avg_max_snapshots + 1u, scratch, report)
               == Status::OUT_OF_RANGE);
        /* More metrics than the set has snapshots is a layout error: the index
           list must stay valid for every tensor. */
        ParamSet closed;
        CHECK (avg_gated_average (sets, closed, gate, metrics, 3u, scratch,
                                  report)
               == Status::BAD_LAYOUT);
        CHECK (report._M_mode == static_cast<uint8_t> (AveragingMode::Swa));
        CHECK (avg_gated_average (sets, destination, gate, metrics, 4u, scratch,
                                  report)
               == Status::BAD_LAYOUT);
        CHECK (avg_gated_average (
                  sets, destination, gate,
                  reinterpret_cast<const int32_t *> (
                      reinterpret_cast<const uint8_t *> (metrics) + 1u),
                  3u, scratch, report)
               == Status::BAD_LAYOUT);
        CHECK (avg_gated_average (
                  sets, destination, gate, metrics, 3u,
                  reinterpret_cast<uint32_t *> (
                      reinterpret_cast<uint8_t *> (scratch) + 1u),
                  report)
               == Status::BAD_LAYOUT);
        CHECK (avg_gated_average (
                  reinterpret_cast<SnapshotSet *> (
                      reinterpret_cast<uint8_t *> (sets) + 1u),
                  destination, gate, metrics, 3u, scratch, report)
               == Status::BAD_LAYOUT);
    }

    SECTION ("a gate that accepts nothing reports the counters and bails out")
    {
        gate._M_has_best = 1u;
        gate._M_best_q15 = -100000;
        gate._M_accepted = 0u;
        gate._M_rejected = 0u;
        alignas (16) int32_t worse[2] = { 10, 20 };
        CHECK (avg_gated_average (sets, destination, gate, worse, 2u, scratch,
                                  report)
               == Status::OUT_OF_RANGE);
        CHECK (report._M_gate_accepted == 0u);
        CHECK (report._M_gate_rejected == 2u);
        CHECK (report._M_tensors == 0u);
    }
}

TEST_CASE ("avg_ema_update_set and avg_snapshot_codes drive the EMA loop body")
{
    W first, second;
    first.fill ({ 10, -20, 30 });
    second.fill ({ -5, 5, 0 });
    W snapshot0, snapshot1;
    snapshot0.fill ({ 40, -40, 0 });
    snapshot1.fill ({ 8, 12, -16 });
    alignas (16) Trainable params[2];
    params[0] = view (first, Shape::vector (3u));
    params[1] = view (second, Shape::vector (3u));
    ParamSet destination;
    REQUIRE (avg_params_open (destination, params, 2u) == Status::OK);
    alignas (16) int16_t lanes0[4], lanes1[4];
    alignas (16) EmaState states[2];
    REQUIRE (avg_ema_open (states[0], lanes0, 3u, 32000) == Status::OK);
    REQUIRE (avg_ema_open (states[1], lanes1, 3u, 32000) == Status::OK);
    REQUIRE (avg_ema_seed (states[0], snapshot0.code) == Status::OK);
    REQUIRE (avg_ema_seed (states[1], snapshot0.code) == Status::OK);
    alignas (16) const int8_t *snapshots[2] = { snapshot1.code,
                                                snapshot1.code };
    AveragingReport report;

    SECTION ("one call steps every state and republishes the codes")
    {
        REQUIRE (avg_ema_update_set (states, destination, snapshots, report)
                 == Status::OK);
        CHECK (report._M_mode == static_cast<uint8_t> (AveragingMode::Ema));
        CHECK (report._M_tensors == 2u);
        CHECK (report._M_lanes == 6u);
        CHECK (report._M_snapshots_used == 1u);
        CHECK (states[0]._M_updates == 1u);
        CHECK (states[1]._M_updates == 1u);
        CHECK (report._M_gate_accepted == 0u);
        for (uint32_t tensor = 0u; tensor < 2u; ++tensor)
        {
            const std::vector<int16_t> held (states[tensor]._M_lanes,
                                             states[tensor]._M_lanes + 3);
            for (uint32_t lane = 0u; lane < 3u; ++lane)
            {
                bool saturated = false;
                const int32_t want = ref_ema_next (
                    static_cast<int32_t> (ref_code_to_q15 (
                        snapshot0.code[lane])),
                    static_cast<int32_t> (ref_code_to_q15 (
                        snapshot1.code[lane])),
                    32000, saturated);
                INFO ("tensor " << tensor << " lane " << lane);
                CHECK (held[lane] == static_cast<int16_t> (want));
                CHECK (params[tensor]._M_code[lane]
                       == ref_simd_q15_to_code (want));
                CHECK (params[tensor]._M_code[lane] != -128);
            }
        }
    }

    SECTION ("a second step raises the update counter the report tracks")
    {
        REQUIRE (avg_ema_update_set (states, destination, snapshots, report)
                 == Status::OK);
        REQUIRE (avg_ema_update_set (states, destination, snapshots, report)
                 == Status::OK);
        CHECK (states[0]._M_updates == 2u);
        CHECK (report._M_snapshots_used == 2u);
        CHECK (report._M_lanes == 6u);
        alignas (16) int16_t corrected[4] = { 0 };
        REQUIRE (avg_ema_bias_corrected (states[0], corrected) == Status::OK);
        const std::vector<int16_t> raw (states[0]._M_lanes,
                                        states[0]._M_lanes + 3);
        const std::vector<int16_t> want
            = ref_bias_corrected (raw, 32000, states[0]._M_updates);
        for (uint32_t lane = 0u; lane < 3u; ++lane)
            CHECK (corrected[lane] == want[lane]);
    }

    SECTION ("state, tensor and snapshot mismatches are refused")
    {
        alignas (16) EmaState closed[2];
        CHECK (avg_ema_update_set (closed, destination, snapshots, report)
               == Status::BAD_LAYOUT);
        CHECK (avg_ema_update_set (states, destination, nullptr, report)
               == Status::BAD_LAYOUT);
        CHECK (avg_ema_update_set (nullptr, destination, snapshots, report)
               == Status::BAD_LAYOUT);
        ParamSet empty;
        CHECK (avg_ema_update_set (states, empty, snapshots, report)
               == Status::BAD_LAYOUT);
        CHECK (avg_ema_update_set (
                  reinterpret_cast<EmaState *> (
                      reinterpret_cast<uint8_t *> (states) + 1u),
                  destination, snapshots, report)
               == Status::BAD_LAYOUT);
        CHECK (avg_ema_update_set (
                  states, destination,
                  reinterpret_cast<const int8_t *const *> (
                      reinterpret_cast<const uint8_t *> (snapshots) + 1u),
                  report)
               == Status::BAD_LAYOUT);
        /* A null snapshot stream for the second tensor. */
        alignas (16) const int8_t *holes[2] = { snapshot1.code, nullptr };
        CHECK (avg_ema_update_set (states, destination, holes, report)
               == Status::BAD_LAYOUT);
        /* Misaligned stream for the first tensor. */
        alignas (16) const int8_t *ragged[2]
            = { reinterpret_cast<const int8_t *> (snapshot1.code) + 1,
                snapshot1.code };
        CHECK (avg_ema_update_set (states, destination, ragged, report)
               == Status::BAD_LAYOUT);
        /* A state whose lane count does not match the tensor it feeds. */
        REQUIRE (avg_ema_open (states[0], lanes0, 2u, 32000) == Status::OK);
        CHECK (avg_ema_update_set (states, destination, snapshots, report)
               == Status::BAD_LAYOUT);
        REQUIRE (avg_ema_open (states[0], lanes0, 3u, 32000) == Status::OK);
    }

    SECTION ("the snapshot writer publishes codes into caller-owned buffers")
    {
        alignas (16) int8_t ring0[4], ring1[4];
        alignas (16) int8_t *ring[2] = { ring0, ring1 };
        REQUIRE (avg_snapshot_codes (destination, ring) == Status::OK);
        for (uint32_t lane = 0u; lane < 3u; ++lane)
        {
            CHECK (ring0[lane] == first.code[lane]);
            CHECK (ring1[lane] == second.code[lane]);
        }
        CHECK (avg_snapshot_codes (destination, nullptr) == Status::BAD_LAYOUT);
        ParamSet empty;
        CHECK (avg_snapshot_codes (empty, ring) == Status::BAD_LAYOUT);
        CHECK (avg_snapshot_codes (
                  destination, reinterpret_cast<int8_t *const *> (
                      reinterpret_cast<uint8_t *> (ring) + 1u))
               == Status::BAD_LAYOUT);
        alignas (16) int8_t *ragged_ring[2]
            = { ring0, reinterpret_cast<int8_t *> (ring1) + 1 };
        CHECK (avg_snapshot_codes (destination, ragged_ring)
               == Status::BAD_LAYOUT);
        /* The ring is a plain copy, so binding it as a snapshot set works and
           averaging it reproduces the published codes exactly. */
        SnapshotSet rebound;
        alignas (16) const int8_t *view_ring[2] = { ring0, ring1 };
        REQUIRE (avg_snapshots_open (rebound, view_ring, 2u, 3u) == Status::OK);
        alignas (16) int8_t one[3] = { 0 };
        REQUIRE (avg_uniform (rebound, 1u, 2u, one) == Status::OK);
        for (uint32_t lane = 0u; lane < 3u; ++lane)
            CHECK (one[lane] == second.code[lane]);
    }

    SECTION ("the documented pool cost of a four-tensor EMA fits the arena")
    {
        /* 4 tensors x 256 lanes: the caller carves exactly this much state. */
        uint32_t total = 0u;
        for (uint32_t tensor = 0u; tensor < 4u; ++tensor)
            total += avg_ema_state_bytes (256u);
        CHECK (total == 2048u);
        alignas (16) uint8_t arena[4 * 512];
        Pool pool (arena, sizeof (arena));
        uint32_t used_before = pool.stats ()._M_used;
        for (uint32_t tensor = 0u; tensor < 4u; ++tensor)
        {
            Slab slab;
            REQUIRE (pool.carve (256u, 2u, slab) == Status::OK);
            CHECK (slab._M_bytes == 512u);
        }
        CHECK (pool.stats ()._M_used == used_before + 2048u);
        CHECK (pool.stats ()._M_overflows == 0u);
        /* One byte more than the arena is a Status, never a crash. */
        Slab spill;
        CHECK (pool.carve (1u, 2049u, spill) == Status::ARENA_OVERFLOW);
        CHECK (pool.stats ()._M_overflows == 1u);
    }
}
/* ======================================================================== */
/* Knowledge distillation (mllee/train_distill.hpp)                          */
/* ======================================================================== */

/* The documented lane set both backwards recompute from the code streams. The
   probabilities themselves are checked against <cmath> in their own case, so
   these helpers only re-do the integer composition. */
struct DistillLanes
{
    std::vector<int32_t> _M_p_s;
    std::vector<int32_t> _M_p_t;
    std::vector<int32_t> _M_delta_q7;
    std::vector<int32_t> _M_log_s;
};

DistillLanes
ref_distill_lanes (const Trainable &student, const Trainable &teacher,
                   const int32_t inv_t)
{
    const SoftmaxState student_state
        = softmax_state_from_codes (student, inv_t);
    const SoftmaxState teacher_state
        = softmax_state_from_codes (teacher, inv_t);
    DistillLanes lanes;
    for (uint32_t i = 0u; i < teacher._M_count; ++i)
    {
        lanes._M_p_t.push_back (prob_q15 (teacher, i, inv_t, teacher_state));
        lanes._M_p_s.push_back (prob_q15 (student, i, inv_t, student_state));
        lanes._M_log_s.push_back (
            log_prob_q7 (student, i, inv_t, student_state));
        lanes._M_delta_q7.push_back (
            log_prob_q7 (teacher, i, inv_t, teacher_state)
            - log_prob_q7 (student, i, inv_t, student_state));
    }
    return lanes;
}

/* sum_c ((log p_t - log p_s) * p_t) >> 15, skipping the zero-target terms.
   The products are summed before the single rounded shift, so no per-class
   truncation can push the divergence below zero. */
int32_t
ref_kl_from_lanes (const DistillLanes &lanes)
{
    int64_t products = 0;
    for (size_t i = 0; i < lanes._M_p_t.size (); ++i)
    {
        if (lanes._M_p_t[i] <= 0)
            continue;
        products += static_cast<int64_t> (lanes._M_delta_q7[i])
                    * lanes._M_p_t[i];
    }
    return products > 0 ? static_cast<int32_t> ((products + (1LL << 14)) >> 15) : 0;
}

/* The hard arm the combined loss uses: -log p of the label, in Q7. */
int32_t
ref_hard_q7 (const Trainable &student, const uint32_t label, const int32_t inv_t)
{
    const SoftmaxState state = softmax_state_from_codes (student, inv_t);
    return -log_prob_q7 (student, label, inv_t, state);
}

TEST_CASE ("distill POD layouts, op ids and defaults are the documented ones")
{
    STATIC_REQUIRE (sizeof (DistillMeta) == 16u);
    STATIC_REQUIRE (sizeof (SoftmaxState) == 8u);
    STATIC_REQUIRE (sizeof (DistillResult) == sizeof (LossResult) + 8u);

    CHECK (_S_op_kl == 0x4C20u);
    CHECK (_S_op_distill == 0x4C21u);
    CHECK (_S_op_distill == _S_op_kl + 1u);
    CHECK (_S_distill_max_classes == 255u);
    /* The class count, the route flag and the shift are byte-wide fields, so the
       module maxima have to fit an 8-bit unsigned. */
    STATIC_REQUIRE (_S_distill_max_classes <= 255u);
    STATIC_REQUIRE (_S_max_operands >= 2u);

    DistillMeta meta;
    CHECK (meta._M_inv_temperature_q15 == _S_q15_max);
    CHECK (meta._M_alpha_q15 == 16384);
    CHECK (meta._M_label == 0u);
    CHECK (meta._M_classes == 0u);
    CHECK (meta._M_route_teacher == 0u);
    CHECK (meta._M_kl_shift == 0u);
    CHECK (meta._M_pad == 0u);
    meta._M_route_teacher = 1u;
    meta._M_kl_shift = 4u;
    CHECK (meta._M_route_teacher == 1u);
    CHECK (meta._M_kl_shift == 4u); /* the largest accepted shift fits the byte */

    SoftmaxState state;
    CHECK (state._M_max_q15 == 0);
    CHECK (state._M_lse_q7 == 0);

    DistillResult result;
    CHECK (result._M_kl_q7 == 0);
    CHECK (result._M_hard_q7 == 0);
    CHECK (result._M_total._M_loss_q7 == 0);
    CHECK (result._M_total._M_loss_q15 == 0);
    CHECK (result._M_total._M_node._M_code == nullptr);
    CHECK (result._M_total._M_node._M_count == 0u);
    CHECK (result._M_total._M_node.validate () == Status::BAD_LAYOUT);
}

TEST_CASE ("q7_log_pos_int tracks the natural log in Q7 and floors at zero")
{
    /* The documented -inf sentinel: log of nothing is the Q15 minimum. */
    CHECK (q7_log_pos_int (0u) == _S_q15_min);
    CHECK (q7_log_pos_int (1u) == 0);
    CHECK (q7_log_pos_int (2u) == _S_ln2_q7);
    CHECK (_S_ln2_q7 == 89);

    SECTION ("every power of two rounds n * ln 2 exactly once")
    {
        /* The octave term stays in Q15 until the single narrowing, so the
           expectation is the correctly rounded n * ln 2 in Q7, not n times the
           already-rounded `_S_ln2_q7`. */
        for (uint32_t exponent = 0u; exponent <= 16u; ++exponent)
            CHECK (q7_log_pos_int (1u << exponent)
                   == static_cast<int32_t> (
                       std::lround (static_cast<double> (exponent)
                                    * std::log (2.0) * 128.0)));
    }

    SECTION ("the mantissa path tracks std::log")
    {
        const uint32_t values[] = { 1u,    2u,    3u,    5u,    7u,    10u,
                                    16u,   63u,   100u,  127u,  255u,  256u,
                                    1000u, 4096u, 8191u, 32767u };
        for (const uint32_t value : values)
        {
            const double got
                = static_cast<double> (q7_log_pos_int (value)) / 128.0;
            const double want = std::log (static_cast<double> (value));
            INFO ("value " << value);
            /* The Q7 domain quantizes at 1/128 nats and the octave term is
               narrowed once from Q15, so the error is a fraction of a Q7. */
            CHECK (std::fabs (got - want) < 0.05);
        }
    }

    SECTION ("monotone over the whole log-sum range a softmax can reach")
    {
        int32_t previous = q7_log_pos_int (1u);
        for (uint32_t value = 2u; value <= 512u; ++value)
        {
            const int32_t got = q7_log_pos_int (value);
            CHECK (got >= previous);
            previous = got;
        }
        /* 255 classes at the widest code spread is the documented worst case for
           the accumulator, and it stays inside the Q7 window. */
        CHECK (q7_log_pos_int (255u) > q7_log_pos_int (128u));
        CHECK (q7_log_pos_int (_S_avg_max_count) < 32767);
    }
}

TEST_CASE ("soften_teacher widens codes and scales them by the reciprocal temperature")
{
    W teacher;
    teacher.fill ({ 127, 60, 3, 0, -60, -127 });
    const Trainable view_teacher = view (teacher, Shape::vector (6u));
    alignas (16) int16_t dest[8] = { 0 };

    SECTION ("T = 1 leaves the widened codes, invT = 1/2 halves them")
    {
        REQUIRE (soften_teacher (view_teacher, _S_q15_max, dest) == Status::OK);
        for (uint32_t i = 0u; i < 6u; ++i)
        {
            const int32_t widened
                = static_cast<int32_t> (ref_code_to_q15 (teacher.code[i]));
            INFO ("lane " << i);
            CHECK (dest[i] == ref_q15_mul (widened, _S_q15_max));
            /* The softened lane is the same value the per-element helper gives. */
            CHECK (softened_logit_q15 (view_teacher, i, _S_q15_max) == dest[i]);
            /* |x| <= 16384 keeps the half-up product exact; outside that window
               the 32767/32768 scale costs exactly one LSB, toward zero. */
            if (widened >= -16383 && widened <= 16384)
                CHECK (dest[i] == widened);
            else
                CHECK (dest[i] == widened + (widened > 0 ? -1 : 1));
        }
        CHECK (dest[0] == 32765);
        CHECK (dest[5] == -32765);

        REQUIRE (soften_teacher (view_teacher, 16384, dest) == Status::OK);
        for (uint32_t i = 0u; i < 6u; ++i)
        {
            CHECK (dest[i] == ref_q15_mul (
                              static_cast<int32_t> (ref_code_to_q15 (
                                  teacher.code[i])),
                              16384));
            CHECK (softened_logit_q15 (view_teacher, i, 16384) == dest[i]);
        }
        /* T = 2 pulls the rails toward zero but keeps the ordering. */
        CHECK (dest[0] > dest[1]);
        CHECK (dest[1] > dest[2]);
        CHECK (dest[2] > dest[3]);
        CHECK (dest[4] < dest[3]);
        CHECK (dest[5] < dest[4]);
    }

    SECTION ("a colder temperature flattens the whole buffer")
    {
        const int32_t sweep[5] = { _S_q15_max, 16384, 8192, 4096, 1 };
        int32_t previous_span = 0;
        for (const int32_t inv_t : sweep)
        {
            REQUIRE (soften_teacher (view_teacher, inv_t, dest) == Status::OK);
            int32_t highest = dest[0];
            int32_t lowest = dest[0];
            for (uint32_t i = 1u; i < 6u; ++i)
            {
                CHECK (dest[i] >= -32768);
                CHECK (dest[i] <= 32767);
                highest = std::max (highest, static_cast<int32_t> (dest[i]));
                lowest = std::min (lowest, static_cast<int32_t> (dest[i]));
            }
            const int32_t span = highest - lowest;
            INFO ("invT " << inv_t);
            CHECK ((span <= previous_span || previous_span == 0));
            CHECK (span >= 0);
            previous_span = span;
            if (inv_t == 1)
            {
                /* 1/32768 temperature: every lane collapses into the rail of one
                   Q15 unit, i.e. a flat softmax. */
                CHECK (span <= 2);
            }
        }
    }

    SECTION ("the accepted range stops just past 1.0 in Q15")
    {
        CHECK (soften_teacher (view_teacher, 0, dest) == Status::OUT_OF_RANGE);
        CHECK (soften_teacher (view_teacher, -1, dest) == Status::OUT_OF_RANGE);
        CHECK (soften_teacher (view_teacher, _S_q15_one + 1, dest)
               == Status::OUT_OF_RANGE);
        CHECK (soften_teacher (view_teacher, 16384, nullptr)
               == Status::BAD_LAYOUT);
        W broken;
        broken.fill ({ 1, 2, 3 });
        const Trainable no_code
            = Trainable::make (nullptr, broken.grad, Shape::vector (3u),
                               Flag::RequiresGrad);
        CHECK (soften_teacher (no_code, 16384, dest) == Status::BAD_LAYOUT);
        /* A frozen teacher still has codes to soften. */
        const Trainable frozen
            = Trainable::make (teacher.code, nullptr, Shape::vector (6u),
                               Flag::Frozen);
        REQUIRE (soften_teacher (frozen, 16384, dest) == Status::OK);
        CHECK (dest[0] == ref_q15_mul (32766, 16384));
        /* inv_temperature == 32768 is exactly 1.0 in Q15, so the lane keeps
           its widened value (the +2^14 rounding term stays below one lsb and
           the top and bottom are the widened rails, not clipped to +/-32768). */
        REQUIRE (soften_teacher (view_teacher, _S_q15_one, dest) == Status::OK);
        CHECK (dest[1] == 60 * 258);
        CHECK (dest[0] == 127 * 258);  /* 32766, the largest widened code */
        CHECK (dest[5] == -127 * 258); /* and the symmetric floor */
    }
}

TEST_CASE ("the Q7 log-softmax matches a float softmax sweep, hot to cold")
{
    W student;
    student.fill ({ 60, -20, 3, 100, 0, -127 });
    const Trainable view_student = view (student, Shape::vector (6u));
    const std::vector<int8_t> codes = student.codes ();
    /* T = 1, 2, 4, 8 plus the degenerate invT = 1. */
    const int32_t sweep[5] = { _S_q15_max, 16384, 8192, 4096, 1 };

    SECTION ("each class' log-probability is the float log-softmax")
    {
        for (const int32_t inv_t : sweep)
        {
            const SoftmaxState state
                = softmax_state_from_codes (view_student, inv_t);
            const std::vector<double> p = ref_softmax (codes, inv_t);
            /* The state carries log-sum-exp of the *logits*, which is not the
               log of the normalized probability sum (that is always zero). */
            double max_z = -1e300;
            for (const int8_t c : codes)
                max_z = std::max (max_z, ref_logit_real (c, inv_t));
            double z_sum = 0.0;
            for (const int8_t c : codes)
                z_sum += std::exp (ref_logit_real (c, inv_t) - max_z);
            const double want_lse = std::log (z_sum) + max_z;
            int64_t prob_total = 0;
            for (uint32_t i = 0u; i < 6u; ++i)
            {
                const double got_log
                    = static_cast<double> (
                          log_prob_q7 (view_student, i, inv_t, state))
                      / 128.0;
                const double want_log = std::log (p[i]);
                INFO ("invT " << inv_t << " class " << i);
                /* Q7 quantization is 1/128 nats and the exp table interpolates
                   on 1/8-nat steps. */
                CHECK (std::fabs (got_log - want_log) < 0.05);
                const int32_t prob = prob_q15 (view_student, i, inv_t, state);
                CHECK (prob >= 0);
                CHECK (prob <= _S_q15_max);
                CHECK (std::fabs (static_cast<double> (prob) / 32768.0 - p[i])
                       < 0.01);
                prob_total += prob;
            }
            /* The sweep is a probability simplex: the sum stays within one
               Q7 LSB per class of 1.0. */
            CHECK (prob_total > 32768 - 6 * 260);
            CHECK (prob_total < 32768 + 6 * 260);
            CHECK (std::fabs (static_cast<double> (state._M_lse_q7) / 128.0
                              - want_lse)
                   < 0.05);
            /* The state's max is the largest softened logit, exactly. */
            int32_t want_max = ref_q15_mul (
                static_cast<int32_t> (ref_code_to_q15 (codes[0])), inv_t);
            for (const int8_t c : codes)
                want_max = std::max (
                    want_max, ref_q15_mul (
                                  static_cast<int32_t> (ref_code_to_q15 (c)),
                                  inv_t));
            CHECK (state._M_max_q15 == want_max);
        }
    }

    SECTION ("raising the temperature flattens the distribution")
    {
        int32_t previous_max = 0;
        int32_t previous_min = 0;
        bool first = true;
        for (const int32_t inv_t : sweep)
        {
            const SoftmaxState state
                = softmax_state_from_codes (view_student, inv_t);
            int32_t highest = prob_q15 (view_student, 0u, inv_t, state);
            int32_t lowest = highest;
            for (uint32_t i = 1u; i < 6u; ++i)
            {
                const int32_t p = prob_q15 (view_student, i, inv_t, state);
                highest = std::max (highest, p);
                lowest = std::min (lowest, p);
            }
            if (!first)
            {
                INFO ("invT " << inv_t);
                /* A colder teacher/student moves probability mass from the peak
                   to the tail: the peak falls and the floor rises. */
                CHECK (highest <= previous_max);
                CHECK (lowest >= previous_min);
            }
            previous_max = highest;
            previous_min = lowest;
            first = false;
        }
        /* Ordering of the log-probabilities follows the code ordering. */
        const SoftmaxState state
            = softmax_state_from_codes (view_student, _S_q15_max);
        CHECK (log_prob_q7 (view_student, 3u, _S_q15_max, state)
               > log_prob_q7 (view_student, 0u, _S_q15_max, state));
        CHECK (log_prob_q7 (view_student, 0u, _S_q15_max, state)
               > log_prob_q7 (view_student, 2u, _S_q15_max, state));
        CHECK (log_prob_q7 (view_student, 2u, _S_q15_max, state)
               > log_prob_q7 (view_student, 1u, _S_q15_max, state));
        CHECK (log_prob_q7 (view_student, 5u, _S_q15_max, state)
               < log_prob_q7 (view_student, 4u, _S_q15_max, state));
        /* Every log-probability is <= 0 in the log domain, and the widest
           code-only spread (2.0 nats) plus ln C stays far above the rail. */
        for (uint32_t i = 0u; i < 6u; ++i)
            CHECK (log_prob_q7 (view_student, i, _S_q15_max, state) <= 0);
        /* The code grid cannot produce a zero probability: the smallest lane the
           exp table can still resolve sits far above the skip threshold. */
        CHECK (prob_q15 (view_student, 5u, _S_q15_max, state) > 0);
        CHECK (q15_exp_q7 (log_prob_q7 (view_student, 5u, _S_q15_max, state))
               > 0);
    }

    SECTION ("log_softmax_q7 and log_sum_exp_lanes agree on a lane buffer")
    {
        alignas (16) int16_t lanes[6];
        for (uint32_t i = 0u; i < 6u; ++i)
            lanes[i] = static_cast<int16_t> (ref_code_to_q15 (codes[i]));
        alignas (16) int16_t out_q7[6] = { 0 };
        SoftmaxState state;
        REQUIRE (log_softmax_q7 (lanes, 6u, out_q7, state) == Status::OK);
        const SoftmaxState direct = log_sum_exp_lanes (lanes, 6u);
        CHECK (state._M_max_q15 == direct._M_max_q15);
        CHECK (state._M_lse_q7 == direct._M_lse_q7);
        for (uint32_t i = 0u; i < 6u; ++i)
        {
            const int32_t logit_q7 = static_cast<int32_t> (
                static_cast<int64_t> (lanes[i]) >> 8);
            CHECK (out_q7[i]
                   == static_cast<int16_t> (ref_q15_to_i16_sat (
                          logit_q7 - state._M_lse_q7)));
            const double want = static_cast<double> (logit_q7) / 128.0
                                - static_cast<double> (state._M_lse_q7) / 128.0;
            const double got = static_cast<double> (out_q7[i]) / 128.0;
            CHECK (std::fabs (got - want) < 1e-9);
        }
        /* The uncentered lane domain is the same softmax as the code domain. */
        CHECK (direct._M_lse_q7
               == softmax_state_from_codes (view_student, _S_q15_one)._M_lse_q7
                      / 1);
        CHECK (log_sum_exp_lanes (nullptr, 6u)._M_max_q15 == 0);
        CHECK (log_sum_exp_lanes (lanes, 0u)._M_lse_q7 == 0);
        CHECK (log_softmax_q7 (nullptr, 6u, out_q7, state) == Status::BAD_LAYOUT);
        CHECK (log_softmax_q7 (lanes, 0u, out_q7, state) == Status::BAD_LAYOUT);
        CHECK (log_softmax_q7 (lanes, 6u, nullptr, state) == Status::BAD_LAYOUT);
        /* A single-class softmax is exactly 0 in the log domain: the Q15 sum is
           one exp term at 2^15, so lse is the maximum in Q7 and the log
           probability is zero. */
        SoftmaxState single;
        alignas (16) int16_t one_out[2] = { 0, 0 };
        REQUIRE (log_softmax_q7 (lanes, 1u, one_out, single) == Status::OK);
        CHECK (single._M_max_q15 == lanes[0]);
        CHECK (single._M_lse_q7 == (lanes[0] >> 8));
        CHECK (one_out[0] == static_cast<int16_t> (ref_q15_to_i16_sat (
                                 (lanes[0] >> 8) - single._M_lse_q7)));
        CHECK (one_out[0] <= 0);
        CHECK (one_out[0] > -3); /* ln 1 = 0 up to the Q7 rounding */
    }

    SECTION ("softmax_state_from_codes is empty for an empty operand")
    {
        W nothing;
        nothing.fill ({ 1 });
        Trainable empty = view (nothing, Shape::vector (1u));
        empty._M_count = 0u;
        const SoftmaxState state = softmax_state_from_codes (empty, _S_q15_max);
        CHECK (state._M_max_q15 == 0);
        CHECK (state._M_lse_q7 == 0);
    }
}

TEST_CASE ("kl_sum_q7 sums the Q7 divergence, skipping zero targets")
{
    W student, teacher;
    student.fill ({ 60, -20, 3, 100, 0, -100 });
    teacher.fill ({ 50, -10, 10, 90, 5, -60 });
    const Trainable view_student = view (student, Shape::vector (6u));
    const Trainable view_teacher = view (teacher, Shape::vector (6u));

    SECTION ("it equals the documented lane sum and approximates the float KL")
    {
        const int32_t inv_t = _S_q15_max;
        const SoftmaxState s_state
            = softmax_state_from_codes (view_student, inv_t);
        const SoftmaxState t_state
            = softmax_state_from_codes (view_teacher, inv_t);
        const DistillLanes lanes = ref_distill_lanes (view_student,
                                                      view_teacher, inv_t);
        const int32_t got
            = kl_sum_q7 (view_student, view_teacher, inv_t, s_state, t_state);
        CHECK (got == ref_kl_from_lanes (lanes));
        CHECK (got >= 0);
        const double nats = static_cast<double> (got) / 128.0;
        const double want
            = ref_kl_real (student.codes (), teacher.codes (), inv_t);
        INFO ("kl " << got << " want " << want * 128.0);
        /* Per-term Q7 truncation and the Q7 log quantization bound the error. */
        CHECK (std::fabs (nats - want) < 0.1);
        /* No term is skipped: every teacher probability is resolvable. */
        for (const int32_t p : lanes._M_p_t)
            CHECK (p > 0);
    }

    SECTION ("identical operands give exactly zero")
    {
        const int32_t inv_t = _S_q15_max;
        const SoftmaxState state
            = softmax_state_from_codes (view_student, inv_t);
        CHECK (kl_sum_q7 (view_student, view_student, inv_t, state, state) == 0);
        CHECK (kl_sum_q7 (view_teacher, view_teacher, inv_t,
                          softmax_state_from_codes (view_teacher, inv_t),
                          softmax_state_from_codes (view_teacher, inv_t))
               == 0);
    }

    SECTION ("the divergence tracks the float KL as the student moves")
    {
        /* The fixtures soften the logits (invT = 0.5) and clamp the moving
           student at -127, so the true divergence is *not* monotone: it peaks
           early and returns near zero once every channel saturates.  What the
           integer path must do is follow the float reference across the whole
           sweep, including the near-tie at the end. */
        const int32_t inv_t = 16384;
        const SoftmaxState t_state
            = softmax_state_from_codes (view_teacher, inv_t);
        W moved;
        for (uint32_t step = 0u; step < 6u; ++step)
        {
            std::vector<int8_t> codes = teacher.codes ();
            for (uint32_t i = 0u; i < 6u; ++i)
                codes[i] = static_cast<int8_t> (codes[i] - step * 20 < -127
                                                    ? -127
                                                    : codes[i] - step * 20);
            moved.fill (codes);
            const Trainable view_moved = view (moved, Shape::vector (6u));
            const SoftmaxState s_state
                = softmax_state_from_codes (view_moved, inv_t);
            const int32_t got = kl_sum_q7 (view_moved, view_teacher, inv_t,
                                          s_state, t_state);
            const double want
                = ref_kl_real (codes, teacher.codes (), inv_t) * 128.0;
            INFO ("step " << step << " kl " << got << " want " << want);
            CHECK (got >= 0);
            /* Every term carries a sub-Q7 log truncation, so a whole sweep is
               allowed to drift two Q7 from the float product. */
            CHECK (std::fabs (static_cast<double> (got) - want) < 2.5);
        }
    }

    SECTION ("a zero-target sweep is the documented limit, not a division")
    {
        /* The guard is structural: a teacher state whose probabilities all fall
           below the exp table's floor contributes no term at all. */
        const int32_t inv_t = _S_q15_max;
        SoftmaxState s_state
            = softmax_state_from_codes (view_student, inv_t);
        SoftmaxState t_state = softmax_state_from_codes (view_teacher, inv_t);
        t_state._M_lse_q7 += 5000; /* raises lse, so every log p falls below
                                      the exp table's floor */
        for (uint32_t i = 0u; i < 6u; ++i)
            CHECK (prob_q15 (view_teacher, i, inv_t, t_state) == 0);
        CHECK (kl_sum_q7 (view_student, view_teacher, inv_t, s_state, t_state)
               == 0);
        /* A zero student probability saturates its term at the rail, finite. */
        s_state._M_lse_q7 += 5000;
        const int32_t saturated
            = kl_sum_q7 (view_student, view_teacher, inv_t, s_state, t_state);
        CHECK (saturated == 0); /* still every target skipped */
        /* And with the teacher intact but the student railed, the sum stays
           inside int32 instead of trapping. */
        const SoftmaxState broken_student
            = softmax_state_from_codes (view_student, inv_t);
        SoftmaxState railed = broken_student;
        railed._M_lse_q7 += 5000;
        const int32_t away
            = kl_sum_q7 (view_student, view_teacher, inv_t, railed,
                         softmax_state_from_codes (view_teacher, inv_t));
        CHECK (away > 0);
        CHECK (away < INT32_MAX);
        CHECK (railed._M_lse_q7 == broken_student._M_lse_q7 + 5000);
        /* The exp table's own floor, which the guard exists to catch. */
        CHECK (q15_exp (0) == _S_q15_max);
        CHECK (q15_exp_neg (32 * _S_q15_one) == 0);
        CHECK (q15_exp_neg (31 * _S_q15_one) > 0);
    }

    SECTION ("the summed form is not averaged over classes")
    {
        /* Documented: C * 0.5 nats routinely exceeds one Q15 lane, so the value
           is reported in the Q7 accumulator domain. */
        const int32_t inv_t = _S_q15_max;
        const int32_t kl = kl_sum_q7 (
            view_student, view_teacher, inv_t,
            softmax_state_from_codes (view_student, inv_t),
            softmax_state_from_codes (view_teacher, inv_t));
        CHECK (kl > 0);
        CHECK (kl < _S_q15_one * 64); /* far past one Q15 lane, still in Q7 */
        const double per_class = static_cast<double> (kl) / 6.0;
        CHECK (per_class > 0.0);
    }
}

TEST_CASE ("kl_divergence_forward records a two-operand node and its backward")
{
    alignas (16) uint8_t memory[4096];
    Pool pool (memory, sizeof (memory));
    alignas (16) Node nodes[4];
    Tape tape (nodes, 4u, pool);
    REQUIRE (tape.begin () == Status::OK);

    W student, teacher;
    student.fill ({ 60, -20, 3, 100, 0, -100 });
    teacher.fill ({ 50, -10, 10, 90, 5, -60 });
    const Trainable view_student = view (student, Shape::vector (6u));
    const Trainable view_teacher = view (teacher, Shape::vector (6u));
    LossResult out;

    SECTION ("the loss is the Q7 sum and the save carries the meta")
    {
        REQUIRE (kl_divergence_forward (pool, &tape, view_student, view_teacher,
                                        _S_q15_max, out)
                 == Status::OK);
        const DistillLanes lanes = ref_distill_lanes (view_student,
                                                      view_teacher, _S_q15_max);
        CHECK (out._M_loss_q7 == ref_kl_from_lanes (lanes));
        CHECK (out._M_loss_q15 == ref_clamp_q15 (
                                      static_cast<int64_t> (out._M_loss_q7)
                                      << 8));
        REQUIRE (out._M_node.validate () == Status::OK);
        CHECK (out._M_node._M_count == 1u);
        CHECK (out._M_node.grad_at (0) == 0);
        const TapeStats stats = tape.stats ();
        CHECK (stats._M_nodes == 1u);
        CHECK (stats._M_peak_nodes == 1u);
        CHECK (stats._M_saved_bytes == 16u);
        CHECK (stats._M_capacity == 4u);
        CHECK (stats._M_saturation_events == 0u);
        Node *const node = tape.node (0u);
        REQUIRE (node != nullptr);
        CHECK (node->_M_op_id == _S_op_kl);
        CHECK (node->_M_operand_count == 2u);
        CHECK (node->_M_operand_count <= _S_max_operands);
        CHECK (node->_M_saved_count == 1u);
        CHECK (node->_M_saved_count <= _S_max_operands);
        const DistillMeta *const meta
            = reinterpret_cast<const DistillMeta *> (
                  pool.at (node->_M_saved[0]._M_offset));
        REQUIRE (meta != nullptr);
        CHECK (meta->_M_inv_temperature_q15 == _S_q15_max);
        CHECK (meta->_M_alpha_q15 == 0);
        CHECK (meta->_M_label == 0u);
        CHECK (meta->_M_classes == 6u);
        CHECK (meta->_M_route_teacher == 0u);
        CHECK (meta->_M_kl_shift == 0u);
        CHECK (node->_M_saved[0]._M_bytes == 16u);
        CHECK (node->_M_saved[0]._M_count == 1u);
        /* The pool cost is exactly the documented 8-byte node plus 16-byte save. */
        CHECK (pool.stats ()._M_used == 24u);
        CHECK (pool.stats ()._M_overflows == 0u);
    }

    SECTION ("backward routes seed * invT * (p_s - p_t) to the student only")
    {
        REQUIRE (kl_divergence_forward (pool, &tape, view_student, view_teacher,
                                        16384, out)
                 == Status::OK);
        REQUIRE (tape.backward (out._M_node) == Status::OK);
        const int32_t inv_t = 16384;
        const DistillLanes lanes = ref_distill_lanes (view_student,
                                                      view_teacher, inv_t);
        const int32_t seed = q15_clamp (tape.loss_scale ());
        CHECK (seed == _S_q15_max);
        CHECK (out._M_node.grad_at (0) == seed);
        for (uint32_t i = 0u; i < 6u; ++i)
        {
            INFO ("class " << i);
            CHECK (student.grad[i]
                   == static_cast<int16_t> (ref_kl_student_lane (
                          lanes._M_p_s[i], lanes._M_p_t[i], inv_t, seed)));
            /* A frozen (unrouted) teacher keeps its lanes. */
            CHECK (teacher.grad[i] == 0);
        }
        int64_t total = 0;
        for (const int32_t g : student.grads ())
            total += g;
        /* Both softmaxes are simplexes, so the gradient sums to ~0. */
        CHECK (total > -6 * 40);
        CHECK (total < 6 * 40);
    }

    SECTION ("route_teacher adds the analytic derivative to the teacher")
    {
        tape.reset ();
        REQUIRE (tape.begin () == Status::OK);
        REQUIRE (kl_divergence_forward (pool, &tape, view_student, view_teacher,
                                        _S_q15_max, out, true)
                 == Status::OK);
        const Node *const node = tape.node (0u);
        REQUIRE (node != nullptr);
        CHECK (reinterpret_cast<const DistillMeta *> (
                   pool.at (node->_M_saved[0]._M_offset))
                   ->_M_route_teacher
               == 1u);
        REQUIRE (tape.backward (out._M_node) == Status::OK);
        const int32_t inv_t = _S_q15_max;
        const DistillLanes lanes = ref_distill_lanes (view_student,
                                                      view_teacher, inv_t);
        const int32_t kl_q7 = ref_kl_from_lanes (lanes);
        const int32_t seed = _S_q15_max;
        for (uint32_t i = 0u; i < 6u; ++i)
        {
            INFO ("routed class " << i);
            const int32_t outer = ref_clamp_q15 (
                static_cast<int64_t> (lanes._M_delta_q7[i] - kl_q7) << 8);
            const int32_t want
                = ref_q15_mul (ref_q15_mul (lanes._M_p_t[i], inv_t), outer);
            CHECK (teacher.grad[i]
                   == static_cast<int16_t> (ref_q15_mul (want, seed)));
            CHECK (student.grad[i]
                   == static_cast<int16_t> (ref_kl_student_lane (
                          lanes._M_p_s[i], lanes._M_p_t[i], inv_t, seed)));
        }
        /* Routing must not change the student arm. */
        tape.reset ();
        REQUIRE (tape.begin () == Status::OK);
        LossResult unrouted;
        W student_copy;
        student_copy.fill (student.codes ());
        const Trainable view_copy = view (student_copy, Shape::vector (6u));
        REQUIRE (kl_divergence_forward (pool, &tape, view_copy, view_teacher,
                                        _S_q15_max, unrouted)
                 == Status::OK);
        REQUIRE (tape.backward (unrouted._M_node) == Status::OK);
        CHECK (unrouted._M_loss_q7 == out._M_loss_q7);
        for (uint32_t i = 0u; i < 6u; ++i)
            CHECK (student_copy.grad[i] == student.grad[i]);
    }

    SECTION ("a student with no gradient lanes still validates")
    {
        tape.reset ();
        REQUIRE (tape.begin () == Status::OK);
        const Trainable frozen
            = Trainable::make (student.code, nullptr, Shape::vector (6u),
                               Flag::Frozen);
        const Trainable frozen_teacher
            = Trainable::make (teacher.code, nullptr, Shape::vector (6u),
                               Flag::Frozen);
        LossResult eval_out;
        REQUIRE (kl_divergence_forward (pool, nullptr, frozen, frozen_teacher,
                                        _S_q15_max, eval_out)
                 == Status::OK);
        CHECK (eval_out._M_loss_q7 > 0);
        REQUIRE (kl_divergence_forward (pool, &tape, frozen, frozen_teacher,
                                        _S_q15_max, eval_out)
                 == Status::OK);
        CHECK (tape.stats ()._M_nodes == 1u);
        REQUIRE (tape.backward (eval_out._M_node) == Status::OK);
        CHECK (tape.stats ()._M_saturation_events == 0u);
    }

    SECTION ("a null tape evaluates without recording")
    {
        LossResult eval_out;
        REQUIRE (kl_divergence_forward (pool, nullptr, view_student,
                                        view_teacher, _S_q15_max, eval_out)
                 == Status::OK);
        CHECK (eval_out._M_loss_q7 > 0);
        CHECK (eval_out._M_node._M_count == 1u);
    }

    SECTION ("class count, temperature and pair validation")
    {
        LossResult refused;
        W wide_student, wide_teacher;
        alignas (16) int8_t s_codes[256];
        alignas (16) int16_t s_grad[256];
        alignas (16) int8_t t_codes[256];
        alignas (16) int16_t t_grad[256];
        for (uint32_t i = 0u; i < 256u; ++i)
        {
            s_codes[i] = static_cast<int8_t> (static_cast<int>(i % 60u) - 30);
            t_codes[i] = static_cast<int8_t> (static_cast<int>(i % 50u) - 25);
            s_grad[i] = 0;
            t_grad[i] = 0;
        }
        wide_student.fill ({ 1 });
        wide_teacher.fill ({ 1 });
        const Trainable too_wide_student
            = Trainable::make (s_codes, s_grad, Shape::vector (256u),
                               Flag::RequiresGrad);
        const Trainable too_wide_teacher
            = Trainable::make (t_codes, t_grad, Shape::vector (256u),
                               Flag::RequiresGrad);
        CHECK (too_wide_student._M_count == 256u);
        CHECK (kl_divergence_forward (pool, &tape, too_wide_student,
                                      too_wide_teacher, _S_q15_max, refused)
               == Status::OUT_OF_RANGE);
        /* The documented maximum itself is accepted. */
        const Trainable widest_student
            = Trainable::make (s_codes, s_grad, Shape::vector (255u),
                               Flag::RequiresGrad);
        const Trainable widest_teacher
            = Trainable::make (t_codes, t_grad, Shape::vector (255u),
                               Flag::RequiresGrad);
        CHECK (kl_divergence_forward (pool, &tape, widest_student, widest_teacher,
                                      _S_q15_max, refused)
               == Status::OK);
        CHECK (refused._M_loss_q7 > 0);
        CHECK (refused._M_node._M_count == 1u);

        /* Mismatched operands are caught by the shared loss pair check. */
        W short_teacher;
        short_teacher.fill ({ 1, 2, 3 });
        const Trainable view_short
            = view (short_teacher, Shape::vector (3u));
        CHECK (kl_divergence_forward (pool, &tape, view_student, view_short,
                                      _S_q15_max, refused)
               == Status::BAD_LAYOUT);
        W empty;
        empty.fill ({ 1 });
        Trainable view_zero = view (empty, Shape::vector (1u));
        view_zero._M_count = 0u;
        CHECK (kl_divergence_forward (pool, &tape, view_zero, view_zero,
                                      _S_q15_max, refused)
               == Status::BAD_LAYOUT);
        CHECK (kl_divergence_forward (pool, &tape, view_student, view_teacher, 0,
                                      refused)
               == Status::OUT_OF_RANGE);
        CHECK (kl_divergence_forward (pool, &tape, view_student, view_teacher,
                                      _S_q15_one + 1, refused)
               == Status::OUT_OF_RANGE);
        /* The pair check runs first, so a bad operand beats a bad temperature. */
        CHECK (kl_divergence_forward (pool, &tape, view_student, view_short, 0,
                                      refused)
               == Status::BAD_LAYOUT);
    }

    SECTION ("the tape's node capacity is a Status, not a crash")
    {
        alignas (16) uint8_t big[_S_train_pool_bytes];
        Pool full_pool (big, sizeof (big));
        alignas (16) Node many[_S_default_tape_nodes + 1u];
        Tape busy (many, _S_default_tape_nodes, full_pool);
        REQUIRE (busy.begin () == Status::OK);
        LossResult step;
        for (uint32_t i = 0u; i < _S_default_tape_nodes; ++i)
            REQUIRE (kl_divergence_forward (full_pool, &busy, view_student,
                                            view_teacher, _S_q15_max, step)
                     == Status::OK);
        CHECK (busy.stats ()._M_nodes == _S_default_tape_nodes);
        /* 64 nodes x (8 + 16) bytes still fits the documented 8 KiB arena. */
        CHECK (full_pool.stats ()._M_used
               <= static_cast<uint32_t> (_S_train_pool_bytes));
        CHECK (full_pool.stats ()._M_overflows == 0u);
        CHECK (kl_divergence_forward (full_pool, &busy, view_student,
                                      view_teacher, _S_q15_max, step)
               == Status::ARENA_OVERFLOW);
        CHECK (busy.stats ()._M_nodes == _S_default_tape_nodes);
        CHECK (busy.node_count () == _S_default_tape_nodes);
        REQUIRE (busy.backward (step._M_node) == Status::OK);
        /* An unwritten node array is refused before any recording. */
        Tape none (many, 0u, full_pool);
        CHECK (none.begin () == Status::BAD_LAYOUT);
        CHECK (kl_divergence_forward (full_pool, &none, view_student,
                                      view_teacher, _S_q15_max, step)
               == Status::INTERNAL);
        /* Backward on a loss node that never came from the tape is rejected. */
        Tape closed (many, _S_default_tape_nodes, full_pool);
        REQUIRE (closed.begin () == Status::OK);
        CHECK (closed.is_open ());
        closed.reset ();
        CHECK (!closed.is_open ());
        CHECK (kl_divergence_forward (full_pool, &closed, view_student,
                                      view_teacher, _S_q15_max, step)
               == Status::INTERNAL);
    }
}
//__CHUNK5__

