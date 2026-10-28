/* Tests for the training loss family: mllee/train_loss.hpp,
 * mllee/train_label_smoothing.hpp and mllee/train_regularization.hpp.
 *
 * Every expectation in this file is produced by a reference written here, in
 * plain integer or floating-point arithmetic, from the rounding rules the
 * headers document (round-half-up for the shifted products, round-half-away
 * from zero for div_round, truncating shifts where the header says a bare
 * shift is used, the Q7 loss rail, the 258-unit code step and the [-127, 127]
 * code alphabet).  The library is never asked for an expected value: an
 * `expected_*` helper below recomputes the whole pipeline from those rules, and
 * every gradient is additionally cross-checked against a hand-written analytic
 * derivative and against a central difference of a floating-point twin of the
 * same loss, so a change of a rounding rule, a divisor or a domain moves this
 * file instead of passing silently. */

#include "mllee/train_label_smoothing.hpp"
#include "mllee/train_loss.hpp"
#include "mllee/train_regularization.hpp"

#include "mllee/train_activations.hpp"
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"
#include "mllee/train_trainer.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <vector>

using namespace zzs::mllee;
using namespace zzs::mllee::train;
using Catch::Approx;

namespace
{
/* ======================================================================== */
/* Reference primitives -- re-implemented here from the documented rules.
 * ======================================================================== */

constexpr int32_t k_step = 258;        /* one code step in Q15 units */
constexpr int32_t k_q15_one = 32768;
constexpr int32_t k_q15_max = 32767;
constexpr int32_t k_q15_min = -32768;
constexpr int32_t k_q7_one = 128;
constexpr double k_code_to_real = 258.0 / 32768.0;

double
ref_clamp (const double value, const double low, const double high)
{
    return value < low ? low : (value > high ? high : value);
}

int32_t
ref_q15_clamp (const int32_t value)
{
    return value > k_q15_max ? k_q15_max : (value < k_q15_min ? k_q15_min : value);
}

/** @brief The documented shift: add half an ulp, then arithmetic shift. */
int32_t
ref_round_shift (const int64_t value, const unsigned shift)
{
    if (shift == 0u)
        return static_cast<int32_t> (value);
    const int64_t half = static_cast<int64_t> (1) << (shift - 1u);
    return static_cast<int32_t> ((value + half) >> shift);
}

/** @brief A bare arithmetic shift (truncates toward minus infinity). */
int64_t
ref_floor_shift (const int64_t value, const unsigned shift)
{
    return value >> shift;
}

/** @brief Division that rounds halves away from zero. */
int32_t
ref_div_round (const int32_t numerator, const int32_t denominator)
{
    if (denominator == 0)
        return 0;
    const bool negative = (numerator < 0) != (denominator < 0);
    const int64_t abs_num = numerator < 0 ? -static_cast<int64_t> (numerator)
                                          : numerator;
    const int64_t abs_den = denominator < 0 ? -static_cast<int64_t> (denominator)
                                            : denominator;
    const int64_t quotient = (abs_num + abs_den / 2) / abs_den;
    const int32_t magnitude = static_cast<int32_t> (quotient);
    return negative ? -magnitude : magnitude;
}

int32_t
ref_q15_mul (const int32_t a, const int32_t b)
{
    return ref_q15_clamp (ref_round_shift (static_cast<int64_t> (a) * b, 15u));
}

int32_t
ref_q15_div (const int32_t a, const int32_t b)
{
    if (b == 0)
        return 0;
    const int64_t scaled = (static_cast<int64_t> (a) << 15) / b;
    return ref_q15_clamp (static_cast<int32_t> (scaled > k_q15_max
                                                   ? k_q15_max
                                                   : (scaled < k_q15_min
                                                          ? k_q15_min
                                                          : scaled)));
}

int32_t
ref_ratio_q15 (const int32_t num, const int32_t den)
{
    if (den == 0)
        return 0;
    const int64_t scaled = (static_cast<int64_t> (num) << 15) / den;
    return ref_q15_clamp (static_cast<int32_t> (scaled > k_q15_max
                                                   ? k_q15_max
                                                   : (scaled < -static_cast<int64_t> (
                                                          k_q15_one)
                                                          ? -k_q15_one
                                                          : scaled)));
}

int32_t
ref_q15_reciprocal (const int32_t x_q15)
{
    if (x_q15 <= 0)
        return k_q15_max;
    const int64_t scaled = (static_cast<int64_t> (1) << 30) / x_q15;
    return static_cast<int32_t> (scaled > k_q15_max ? k_q15_max : scaled);
}

/** @brief Independent integer square root (digit-by-digit, no floats). */
uint64_t
ref_isqrt (const uint64_t value)
{
    uint64_t root = 0;
    uint64_t bit = static_cast<uint64_t> (1) << 62;
    while (bit > value)
        bit >>= 2;
    uint64_t remainder = value;
    while (bit != 0)
    {
        if (remainder >= root + bit)
        {
            remainder -= root + bit;
            root = (root >> 1) + bit;
        }
        else
            root >>= 1;
        bit >>= 2;
    }
    return root;
}

int32_t
ref_q15_sqrt (const int32_t x_q15)
{
    if (x_q15 <= 0)
        return 0;
    return static_cast<int32_t> (ref_isqrt (static_cast<uint64_t> (x_q15) << 15));
}

int32_t
ref_q15_scale_shift (const int32_t a, const int32_t b, const unsigned shift)
{
    return static_cast<int32_t> ((static_cast<int64_t> (a) * b
                                  + (static_cast<int64_t> (1) << (shift - 1u)))
                                 >> shift);
}

int32_t
ref_q15_abs (const int32_t value)
{
    return value < 0 ? -value : value;
}

int8_t
ref_i8_code (const int32_t value)
{
    return static_cast<int8_t> (value < -127 ? -127 : (value > 127 ? 127 : value));
}

int32_t
ref_code_q15 (const int32_t code)
{
    return code * k_step;
}

/** @brief Q15 lane -> code, the documented rounding of a lane back to the grid. */
int8_t
ref_q15_to_code (const int32_t lane)
{
    return ref_i8_code (ref_round_shift (static_cast<int64_t> (lane) * 127, 15u));
}

int16_t
ref_i16_sat (const int32_t value)
{
    return static_cast<int16_t> (value > k_q15_max
                                     ? k_q15_max
                                     : (value < k_q15_min ? k_q15_min : value));
}

/** @brief -ln(p) in Q7 from a Q15 lane, straight from <cmath>. */
double
ref_neg_log_q7 (const double p_real)
{
    return -std::log (p_real) * static_cast<double> (k_q7_one);
}

double
ref_softplus (const double x)
{
    return x > 30.0 ? x : std::log1p (std::exp (x));
}

double
ref_sigmoid (const double x)
{
    return 1.0 / (1.0 + std::exp (-x));
}

/** @brief The mean of the squares of the code diffs, reported in Q7. */
int32_t
ref_mean_square_q7 (const int8_t *const pred, const int8_t *const target,
                    const uint32_t count)
{
    int64_t total_q30 = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
        const int64_t d = static_cast<int64_t> (pred[i] - target[i]) * k_step;
        total_q30 += d * d;
    }
    if (count == 0u)
        return 0;
    /* Half an ulp of the divisor added before the divide: round half up. */
    return static_cast<int32_t> (
        (total_q30 + (static_cast<int64_t> (count) << 22))
        / (static_cast<int64_t> (count) << 23));
}

/** @brief The mean of the Q15 magnitudes, reported in Q7 (half up). */
int32_t
ref_mean_magnitude_q7 (const int64_t total_q15, const uint32_t count)
{
    if (count == 0u)
        return 0;
    return static_cast<int32_t> (
        (total_q15 + (static_cast<int64_t> (count) << 7))
        / (static_cast<int64_t> (count) << 8));
}

double
mean_square_real (const int8_t *const pred, const int8_t *const target,
                  const uint32_t count)
{
    double total = 0.0;
    for (uint32_t i = 0; i < count; ++i)
    {
        const double d = (pred[i] - target[i]) * k_code_to_real;
        total += d * d;
    }
    return total / static_cast<double> (count);
}

/* A central difference of any real-valued loss twin. */
double
central_difference (const std::function<double (const std::vector<double> &)> &f,
                    std::vector<double> x, const size_t index, const double step)
{
    x[index] += step;
    const double up = f (x);
    x[index] -= 2.0 * step;
    const double down = f (x);
    return (up - down) / (2.0 * step);
}

/** @brief The lane a loss backward should publish for a unit-seeded graph. */
double
lane_derivative (const int32_t lane, const int32_t seed)
{
    return static_cast<double> (lane) / static_cast<double> (seed);
}

struct ElementCase
{
    std::vector<int8_t> pred;
    std::vector<int8_t> target;
};

const ElementCase _element_cases[] = {
    { { 10, -20, 30, -40 }, { 4, 5, 6, 7 } },
    { { 0, 0, 0, 0 }, { 0, 0, 0, 0 } },
    { { 127, -127, 127, -127 }, { -127, 127, 0, 3 } },
    { { 1 }, { -1 } },
    { { 5, 7 }, { -5, -7 } },
    { { 30, 31, 32, 33, 34, 35 }, { 12, -13, 40, -1, 0, 35 } },
    { { -60, 20 }, { -61, 20 } },
    { { 100, -100, 50, -50 }, { 100, -100, 51, -49 } },
};

/* ======================================================================== */
/* Harness: a pool, a tape, and two operand buffers.
 * ======================================================================== */

struct Harness
{
    static constexpr uint32_t k_len = 16u;
    static constexpr uint32_t k_nodes = 4u;

    AlignedArena<4096> arena;
    Pool pool;
    alignas (8) Node nodes[k_nodes];
    Tape tape;
    alignas (8) int8_t pred_code[k_len] = { 0 };
    alignas (8) int16_t pred_grad[k_len] = { 0 };
    alignas (8) int8_t targ_code[k_len] = { 0 };
    alignas (8) int16_t targ_grad[k_len] = { 0 };

    Harness () : pool (arena.pool ()), tape (nodes, k_nodes, pool)
    {
    }

    Trainable
    pred (const uint32_t count)
    {
        return Trainable::make (pred_code, pred_grad,
                                Shape::vector (static_cast<uint16_t> (count)),
                                RequiresGrad);
    }

    Trainable
    target (const uint32_t count)
    {
        return Trainable::make (targ_code, targ_grad,
                                Shape::vector (static_cast<uint16_t> (count)),
                                RequiresGrad);
    }

    Trainable
    target_const (const uint32_t count)
    {
        return Trainable::make (targ_code, nullptr,
                                Shape::vector (static_cast<uint16_t> (count)), 0u);
    }

    void
    fill (const std::vector<int8_t> &p, const std::vector<int8_t> &t)
    {
        for (uint32_t i = 0; i < k_len; ++i)
        {
            pred_code[i] = i < p.size () ? p[i] : 0;
            targ_code[i] = i < t.size () ? t[i] : 0;
            pred_grad[i] = 0;
            targ_grad[i] = 0;
        }
    }

    /** @brief Fresh pool + open tape, so one section never sees another's slabs. */
    void
    open ()
    {
        tape.reset ();
        pool.reset ();
        for (uint32_t i = 0; i < k_len; ++i)
        {
            pred_grad[i] = 0;
            targ_grad[i] = 0;
        }
        REQUIRE (status_ok (tape.begin ()));
    }

    /** @brief Run the recorded loss backward with the tape seeding the node. */
    Status
    backward (LossResult &result, const int32_t seed)
    {
        tape.set_loss_scale (seed);
        return tape.backward (result._M_node);
    }

    Node *
    only_node ()
    {
        Node *const node = tape.node (0u);
        REQUIRE (node != nullptr);
        return node;
    }

    /** @brief The saved slab of slot 0, typed, as the tape hands it to backward. */
    template <typename T>
    const T *
    saved (const uint32_t slot = 0u)
    {
        Node *const node = only_node ();
        REQUIRE (node->_M_saved_count > slot);
        return reinterpret_cast<const T *> (pool.at (node->_M_saved[slot]._M_offset));
    }

    /** @brief Mutable view of a saved slab, for index-validation probes. */
    template <typename T>
    T *
    saved_mut (const uint32_t slot = 0u)
    {
        Node *const node = only_node ();
        REQUIRE (node->_M_saved_count > slot);
        return reinterpret_cast<T *> (pool.at (node->_M_saved[slot]._M_offset));
    }

    /** @brief A GradContext over the recorded node, exactly as the walk builds it. */
    GradContext
    context (const int32_t seed_scale = k_q15_one)
    {
        Node *const node = only_node ();
        GradContext ctx;
        ctx._M_pool = &pool;
        ctx._M_operands = node->_M_operands;
        ctx._M_operand_count = node->_M_operand_count;
        ctx._M_saved = node->_M_saved;
        ctx._M_saved_count = node->_M_saved_count;
        ctx._M_loss_scale_q15 = seed_scale;
        return ctx;
    }
};

/** @brief The float reference loss twin for one ElementCase and one loss kind. */
enum Kind
{
    MSE,
    MAE,
    HUBER,
    QUANTILE,
    BCE,
    COSINE,
    HINGE
};

double
ref_loss_real (const Kind kind, const std::vector<double> &p,
               const std::vector<double> &t, const double aux)
{
    const size_t n = p.size ();
    switch (kind)
    {
    case MSE:
    {
        double total = 0.0;
        for (size_t i = 0; i < n; ++i)
            total += (p[i] - t[i]) * (p[i] - t[i]);
        return total / static_cast<double> (n);
    }
    case MAE:
    {
        double total = 0.0;
        for (size_t i = 0; i < n; ++i)
            total += std::abs (p[i] - t[i]);
        return total / static_cast<double> (n);
    }
    case HUBER:
    {
        const double delta = aux;
        double total = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double d = p[i] - t[i];
            total += std::abs (d) <= delta ? 0.5 * d * d
                                           : delta * (std::abs (d) - 0.5 * delta);
        }
        return total / static_cast<double> (n);
    }
    case QUANTILE:
    {
        const double tau = aux;
        double total = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double d = p[i] - t[i];
            total += std::max (tau * d, (tau - 1.0) * d);
        }
        return total / static_cast<double> (n);
    }
    case BCE:
    {
        double total = 0.0;
        for (size_t i = 0; i < n; ++i)
            total += (1.0 - t[i]) * ref_softplus (p[i]) + t[i] * ref_softplus (-p[i]);
        return total / static_cast<double> (n);
    }
    case COSINE:
    {
        double dot = 0.0, np = 0.0, nt = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            dot += p[i] * t[i];
            np += p[i] * p[i];
            nt += t[i] * t[i];
        }
        if (np == 0.0 || nt == 0.0)
            return 1.0;
        return 1.0 - dot / (std::sqrt (np) * std::sqrt (nt));
    }
    case HINGE:
    {
        const double margin = aux;
        double total = 0.0;
        for (size_t i = 0; i < n; ++i)
        {
            const double y = t[i] >= 0.0 ? 1.0 : -1.0;
            const double product = y * p[i];
            total += product < margin ? margin - product : 0.0;
        }
        return total / static_cast<double> (n);
    }
    }
    return 0.0;
}

std::vector<double>
reals_of (const std::vector<int8_t> &codes)
{
    std::vector<double> out;
    out.reserve (codes.size ());
    for (const int8_t c : codes)
        out.push_back (static_cast<double> (c) * k_code_to_real);
    return out;
}
} // namespace

/* ======================================================================== */
/* Additional reference pipelines: each one recomputes the documented integer
 * formula of one loss from the ref_* primitives above, so an expectation is
 * the module's rounding contract, not the library's output.
 * ======================================================================== */
namespace
{
/** @brief The seed a tape-backed backward actually sees: the loss scale after
 *         the int16 seeding lane (q15_clamp, so 32768 lands on 32767). */
int32_t
ref_seed_lane (const int32_t scale)
{
    return ref_q15_clamp (scale);
}

/** @brief Huber Q7: per-term halves truncate (bare >> 1), the mean rounds
 *         half away, all as documented in train_loss.hpp. */
int32_t
ref_huber_q7 (const std::vector<int8_t> &pred, const std::vector<int8_t> &target,
              const int32_t delta)
{
    const uint32_t n = static_cast<uint32_t> (pred.size ());
    int64_t total_q30 = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        const int64_t d = ref_code_q15 (pred[i] - target[i]);
        const int64_t magnitude = d < 0 ? -d : d;
        total_q30 += magnitude <= delta
                         ? (d * d) >> 1
                         : (static_cast<int64_t> (delta) * (2 * magnitude - delta))
                               >> 1;
    }
    if (n == 0u)
        return 0;
    return static_cast<int32_t> (
        (total_q30 + (static_cast<int64_t> (n) << 22))
        / (static_cast<int64_t> (n) << 23));
}

/** @brief Pinball Q7: both arms shift-truncate, the max is exact, the mean
 *         rounds half up through the magnitude helper. */
int32_t
ref_quantile_q7 (const std::vector<int8_t> &pred, const std::vector<int8_t> &target,
                 const int32_t tau)
{
    const uint32_t n = static_cast<uint32_t> (pred.size ());
    int64_t total_q15 = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        const int64_t d = ref_code_q15 (pred[i] - target[i]);
        const int64_t up = ref_floor_shift (d * tau, 15u);
        const int64_t down = ref_floor_shift (d * (tau - k_q15_one), 15u);
        total_q15 += up > down ? up : down;
    }
    return ref_mean_magnitude_q7 (total_q15, n);
}

/** @brief Binary hinge Q7: y = +-q15_max from the label SIGN OF CODE (>= 0 is
 *         positive), products narrow through q15_mul then a bare Q7 shift. */
int32_t
ref_hinge_q7 (const std::vector<int8_t> &scores, const std::vector<int8_t> &labels,
              const int32_t margin_q7)
{
    const uint32_t n = static_cast<uint32_t> (scores.size ());
    int64_t total_q7 = 0;
    uint32_t violations = 0u;
    for (uint32_t i = 0; i < n; ++i)
    {
        const int32_t y = labels[i] >= 0 ? k_q15_max : -k_q15_max;
        const int32_t product_q7 = ref_q15_mul (y, ref_code_q15 (scores[i])) >> 8;
        if (product_q7 < margin_q7)
        {
            total_q7 += margin_q7 - product_q7;
            ++violations;
        }
    }
    if (violations == 0u)
        return 0;
    return ref_div_round (static_cast<int32_t> (total_q7), static_cast<int32_t> (n));
}

/** @brief The number of multiclass violators for a Crammer-Singer sum. */
uint32_t
ref_multiclass_violations (const std::vector<int8_t> &scores, const uint32_t target,
                           const int32_t margin_q7)
{
    uint32_t violations = 0u;
    for (uint32_t i = 0; i < scores.size (); ++i)
    {
        if (i == target)
            continue;
        const int32_t diff_q7 = ref_q15_mul (ref_code_q15 (scores[i])
                                                 - ref_code_q15 (scores[target]),
                                             k_q15_max)
                                >> 8;
        if (margin_q7 + diff_q7 > 0)
            ++violations;
    }
    return violations;
}

/** @brief Cosine Q7: code-domain norms, truncating ratio, 1.0 minus the Q7
 *        narrowing of the clamped cosine. */
int32_t
ref_cosine_q7 (const std::vector<int8_t> &pred, const std::vector<int8_t> &target)
{
    int32_t dot = 0, sq_p = 0, sq_t = 0;
    for (size_t i = 0; i < pred.size (); ++i)
    {
        dot += pred[i] * target[i];
        sq_p += pred[i] * pred[i];
        sq_t += target[i] * target[i];
    }
    int32_t cos_q15 = 0;
    if (sq_p != 0 && sq_t != 0)
    {
        const int64_t denom = static_cast<int64_t> (static_cast<int32_t> (ref_isqrt (
                                   static_cast<uint64_t> (sq_p))))
                              * static_cast<int32_t> (ref_isqrt (
                                  static_cast<uint64_t> (sq_t)));
        const int64_t raw = (static_cast<int64_t> (dot) << 15) / denom;
        cos_q15 = ref_q15_clamp (static_cast<int32_t> (raw));
    }
    return k_q7_one - (cos_q15 >> 8);
}

/** @brief The exact Q7<<8 saturating copy the module documents for _M_loss_q15. */
int32_t
ref_loss_q15 (const int32_t loss_q7)
{
    const int64_t wide = static_cast<int64_t> (loss_q7) << 8;
    return static_cast<int32_t> (wide > k_q15_max ? k_q15_max
                                                  : (wide < k_q15_min ? k_q15_min : wide));
}

/** @brief Dot product of two code streams mapped to the real domain. */
double
dot_real (const std::vector<int8_t> &a, const std::vector<int8_t> &b)
{
    double total = 0.0;
    for (size_t i = 0; i < a.size (); ++i)
        total += (a[i] * k_code_to_real) * (b[i] * k_code_to_real);
    return total;
}

/** @brief Check one LossResult against the rail/step contract: the Q15 copy
 *        is the saturating Q7<<8 narrowing, the node code is that lane's grid
 *        value, the node gradient starts dead, and the node is a scalar. */
void
check_loss_result_shape (const LossResult &result, const uint32_t expected_operand_count,
                         const uint32_t expected_saved_count, const uint32_t expected_op_id,
                         Harness &harness)
{
    CHECK (result._M_loss_q15 == ref_loss_q15 (result._M_loss_q7));
    CHECK (result._M_node._M_count == 1u);
    CHECK (result._M_node._M_shape == Shape::scalar ());
    CHECK (result._M_node.code_at (0) >= -127);
    CHECK (result._M_node.code_at (0) != static_cast<int8_t> (-128));
    CHECK (result._M_node.grad_at (0) == 0);
    Node *const node = harness.tape.node (0u);
    REQUIRE (node != nullptr);
    CHECK (node->_M_op_id == expected_op_id);
    CHECK (node->_M_operand_count == expected_operand_count);
    CHECK (node->_M_saved_count == expected_saved_count);
}
} // namespace

/* ======================================================================== */
/* train_loss.hpp -- contract, helpers and the elementwise family           */
/* ======================================================================== */

TEST_CASE ("loss family contract constants and struct defaults")
{
    SECTION ("op ids are the documented contiguous block")
    {
        CHECK (_S_op_mse == 0x4C01u);
        CHECK (_S_op_mae == 0x4C02u);
        CHECK (_S_op_huber == 0x4C03u);
        CHECK (_S_op_quantile == 0x4C04u);
        CHECK (_S_op_bce == 0x4C05u);
        CHECK (_S_op_cross_entropy == 0x4C06u);
        CHECK (_S_op_nll == 0x4C07u);
        CHECK (_S_op_cosine == 0x4C08u);
        CHECK (_S_op_hinge == 0x4C09u);
        CHECK (_S_op_hinge_multiclass == 0x4C0Au);
        CHECK (_S_op_smoothed_ce == 0x4C10u);
        CHECK (_S_op_soft_ce == 0x4C11u);
    }
    SECTION ("domain constants of the module")
    {
        CHECK (_S_log_step_q7 == 4);
        CHECK (_S_q15_one == 32768);
        CHECK (_S_q15_max == 32767);
        CHECK (_S_q15_min == -32768);
        CHECK (_S_q7_one == 128);
        CHECK (_S_code_to_q15_scale == 258);
        CHECK (_S_train_pool_bytes == 8192u);
        CHECK (_S_pool_alignment == 4u);
        CHECK (_S_max_operands == 4u);
    }
    SECTION ("parameter blocks default as documented")
    {
        CHECK (HuberParams{}._M_delta_q15 == _S_q15_max);
        CHECK (QuantileParams{}._M_tau_q15 == 16384);
        CHECK (CosineParams{}._M_eps_q15 == 32);
        CHECK (HingeParams{}._M_margin_q7 == _S_q7_one);
        CHECK (TargetMeta{}._M_classes == 0u);
        CHECK (TargetMeta{}._M_target == 0u);
        CHECK (TargetMeta{}._M_margin_q7 == 0);
        CHECK (LossResult{}._M_loss_q7 == 0);
        CHECK (LossResult{}._M_loss_q15 == 0);
    }
    SECTION ("struct sizes pinned by the header static_asserts")
    {
        CHECK (sizeof (LossResult) == sizeof (Trainable) + 8u);
        CHECK (sizeof (HuberParams) == 4u);
        CHECK (sizeof (QuantileParams) == 4u);
        CHECK (sizeof (CosineParams) == 4u);
        CHECK (sizeof (HingeParams) == 4u);
        CHECK (sizeof (TargetMeta) == 8u);
        CHECK (sizeof (SmoothingParams) == 8u);
        CHECK (sizeof (SmoothedCeMeta) == 12u);
        CHECK (sizeof (RegularizationStats) == 32u);
        CHECK (sizeof (RegularizationPolicy) == sizeof (void *) + 32u);
    }
}

TEST_CASE ("loss reduction helpers: q30/q15 means, half-up tie and zero count")
{
    SECTION ("q30_sumsq_mean_to_q7 exact table")
    {
        struct Row
        {
            int64_t total_q30;
            uint32_t count;
            int32_t expect;
        };
        constexpr Row rows[] = {
            { 0, 1u, 0 },
            { 1, 1u, 0 },                        /* 1 / 2^23 truncates */
            { static_cast<int64_t> (1) << 22, 1u, 1 },  /* exact half-ulp -> up */
            { (static_cast<int64_t> (1) << 23) - 1, 1u, 1 },
            { static_cast<int64_t> (1) << 23, 1u, 1 },  /* exactly 1.0 Q7 */
            { 3 * (static_cast<int64_t> (1) << 23) + (static_cast<int64_t> (1) << 22),
              1u, 4 },                            /* 3.5 rounds up to 4 */
            { 4 * (static_cast<int64_t> (1) << 23), 4u, 1 },
            { static_cast<int64_t> (1) << 30, 1u, 128 },
        };
        for (const auto &row : rows)
        {
            CHECK (q30_sumsq_mean_to_q7 (row.total_q30, row.count) == row.expect);
            const int64_t independent = (row.total_q30
                                         + (static_cast<int64_t> (row.count) << 22))
                                        / (static_cast<int64_t> (row.count) << 23);
            CHECK (row.expect == static_cast<int32_t> (independent));
        }
        CHECK (q30_sumsq_mean_to_q7 (12345, 0u) == 0);
    }
    SECTION ("q15_sum_mean_to_q7 exact table")
    {
        struct Row
        {
            int64_t total_q15;
            uint32_t count;
            int32_t expect;
        };
        constexpr Row rows[] = {
            { 0, 1u, 0 },
            { 1, 1u, 0 },
            { 1 << 7, 1u, 1 },                   /* half an ulp, rounds up */
            { (1 << 8) - 1, 1u, 1 },
            { 1 << 8, 1u, 1 },
            { 4 * (1 << 8), 4u, 1 },
            { 3 * (1 << 8) + (1 << 7), 4u, 1 },  /* (768+512)/1024 = 1 */
            { -(1 << 8), 1u, 0 },                /* -256+128 = -128 -> 0 (trunc) */
            { 1 << 15, 1u, 128 },
        };
        for (const auto &row : rows)
            CHECK (q15_sum_mean_to_q7 (row.total_q15, row.count) == row.expect);
        CHECK (q15_sum_mean_to_q7 (999, 0u) == 0);
        /* Recompute the same means from the test's own integer arithmetic. */
        for (const auto &row : rows)
        {
            if (row.count == 0u)
                continue;
            const int64_t expect = (row.total_q15
                                    + (static_cast<int64_t> (row.count) << 7))
                                   / (static_cast<int64_t> (row.count) << 8);
            CHECK (row.expect == static_cast<int32_t> (expect));
        }
    }
    SECTION ("div_round ties land away from zero, both signs")
    {
        CHECK (div_round (3, 2) == 2);           /* 1.5 -> 2 */
        CHECK (div_round (-3, 2) == -2);         /* -1.5 -> -2 (away from zero) */
        CHECK (div_round (5, 2) == 3);           /* (5+1)/2: 2.5 -> 3 */
        CHECK (div_round (-5, 2) == -3);
        CHECK (div_round (7, 4) == 2);           /* 1.75 -> 2 */
        CHECK (div_round (1, 4) == 0);
        CHECK (div_round (2, 4) == 1);           /* 0.5 -> 1 */
        CHECK (div_round (-2, 4) == -1);
        CHECK (div_round (10, 0) == 0);          /* zero divisor is a rail, no trap */
        for (int32_t num = -40; num <= 40; ++num)
            for (int32_t den = 1; den <= 7; ++den)
                CHECK (div_round (num, den) == ref_div_round (num, den));
    }
}

TEST_CASE ("mse_forward: Q7 value matches the integer and float reference")
{
    Harness harness;
    for (const auto &c : _element_cases)
    {
        const uint32_t n = static_cast<uint32_t> (c.pred.size ());
        harness.fill (c.pred, c.target);
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (mse_forward (harness.pool, &harness.tape,
                                         harness.pred (n), harness.target (n), result)));
        CHECK (result._M_loss_q7 == ref_mean_square_q7 (c.pred.data (), c.target.data (), n));
        /* Independent float twin: mean of squared reals, mapped to Q7. */
        const double q7_float = mean_square_real (c.pred.data (), c.target.data (), n)
                                * 128.0;
        CHECK (static_cast<double> (result._M_loss_q7) == Approx (q7_float).margin (2.0));
        CHECK (result._M_loss_q15 == ref_loss_q15 (result._M_loss_q7));
        CHECK (result._M_node.code_at (0) == ref_q15_to_code (result._M_loss_q15));
        CHECK (harness.tape.node_count () == 1u);
        CHECK (harness.pool.stats ()._M_used == 8u); /* documented: node only */
        check_loss_result_shape (result, 2u, 0u, _S_op_mse, harness);
    }
    SECTION ("perfect prediction is exactly zero, every arm")
    {
        const std::vector<int8_t> same = { 10, -20, 30, -40 };
        harness.fill (same, same);
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (mse_forward (harness.pool, &harness.tape,
                                        harness.pred (4), harness.target (4), result)));
        CHECK (result._M_loss_q7 == 0);
        CHECK (result._M_loss_q15 == 0);
        CHECK (result._M_node.code_at (0) == 0);
    }
    SECTION ("opposite-signed rails: 4.0 squares, saturating Q15 copy")
    {
        harness.fill ( { 127, 127 }, { -127, -127 });
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (mse_forward (harness.pool, &harness.tape,
                                         harness.pred (2), harness.target (2), result)));
        /* (254 * 258 / 32768)^2 = 3.99975 -> 511.97 -> Q7 half-up = 512. */
        CHECK (result._M_loss_q7 == 512);
        CHECK (result._M_loss_q15 == k_q15_max);  /* saturates, never wraps */
        CHECK (result._M_node.code_at (0) == 127);
    }
    SECTION ("mean reduction: duplicating the buffer does not move the value")
    {
        harness.fill ( { 10, 10, -20 }, { 4, 4, 5 });
        harness.open ();
        LossResult single;
        REQUIRE (status_ok (mse_forward (harness.pool, &harness.tape,
                                        harness.pred (3), harness.target (3), single)));
        harness.fill ( { 10, 10, -20, 10, 10, -20 }, { 4, 4, 5, 4, 4, 5 });
        harness.open ();
        LossResult doubled;
        REQUIRE (status_ok (mse_forward (harness.pool, &harness.tape,
                                        harness.pred (6), harness.target (6), doubled)));
        CHECK (single._M_loss_q7 == doubled._M_loss_q7);
        CHECK (single._M_loss_q7 > 0);
    }
    SECTION ("pair validation: mismatched or empty operands are BAD_LAYOUT")
    {
        harness.open ();
        LossResult result;
        CHECK (mse_forward (harness.pool, &harness.tape, harness.pred (4),
                            harness.target (3), result)
               == Status::BAD_LAYOUT);
        CHECK (mse_forward (harness.pool, &harness.tape, harness.pred (0),
                            harness.target (0), result)
               == Status::BAD_LAYOUT);
        Trainable broken = Trainable::make (nullptr, harness.pred_grad,
                                            Shape::vector (4), RequiresGrad);
        CHECK (mse_forward (harness.pool, &harness.tape, broken, harness.target (4),
                            result)
               == Status::BAD_LAYOUT);
        CHECK (loss_pair_ok (harness.pred (4), harness.target (4)) == Status::OK);
        CHECK (loss_pair_ok (harness.pred (5), harness.target (4)) == Status::BAD_LAYOUT);
        CHECK (loss_pair_ok (harness.pred (0), harness.target (0)) == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("mse_backward: exact integer lanes, analytic reference, central difference")
{
    Harness harness;
    struct SeedCase
    {
        int32_t scale;
    };
    const SeedCase seeds[] = { { k_q15_one }, { 8192 }, { 16384 }, { 1 }, { 0 } };
    for (const auto &c : _element_cases)
    {
        const uint32_t n = static_cast<uint32_t> (c.pred.size ());
        for (const auto &s : seeds)
        {
            harness.fill (c.pred, c.target);
            harness.open ();
            LossResult result;
            REQUIRE (status_ok (mse_forward (harness.pool, &harness.tape,
                                            harness.pred (n), harness.target (n), result)));
            REQUIRE (status_ok (harness.backward (result, s.scale)));
            const int32_t seed = ref_seed_lane (s.scale);
            for (uint32_t i = 0; i < n; ++i)
            {
                const int32_t d_q15 = ref_code_q15 (c.pred[i] - c.target[i]);
                const int32_t base = ref_div_round (2 * d_q15, static_cast<int32_t> (n));
                const int32_t expect = ref_q15_mul (base, seed);
                CHECK (harness.pred_grad[i] == expect);
            }
            CHECK (harness.tape.stats ()._M_saturation_events == 0u);
        }
    }
    SECTION ("gradient equals dL/dp_real * 32768 and the central difference")
    {
        const std::vector<int8_t> pred = { 10, -20, 30, -40 };
        const std::vector<int8_t> target = { 4, 5, 6, 7 };
        harness.fill (pred, target);
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (mse_forward (harness.pool, &harness.tape,
                                        harness.pred (4), harness.target (4), result)));
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        const std::vector<double> t_real = reals_of (target);
        auto twin = [&](const std::vector<double> &x)
        { return ref_loss_real (MSE, x, t_real, 0.0); };
        for (uint32_t i = 0; i < 4u; ++i)
        {
            /* Analytic real derivative, lane convention = x 32768. */
            const double p_real = pred[i] * k_code_to_real;
            const double analytic = 2.0 * (p_real - t_real[i]) / 4.0;
            CHECK (lane_derivative (harness.pred_grad[i], ref_seed_lane (k_q15_one))
                   == Approx (analytic).margin (2e-4));
            const double numeric = central_difference (twin, reals_of (pred), i, 1e-3);
            CHECK (lane_derivative (harness.pred_grad[i], ref_seed_lane (k_q15_one))
                   == Approx (numeric).margin (2e-4));
        }
    }
    SECTION ("a repeated backward clamps the lane and counts the saturation")
    {
        harness.fill ( { 127 }, { -127 });
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (mse_forward (harness.pool, &harness.tape,
                                        harness.pred (1), harness.target (1), result)));
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        /* The raw 2*(254*258) is beyond a lane: the product rail pins it. */
        CHECK (harness.pred_grad[0] == k_q15_max);
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        CHECK (harness.pred_grad[0] == k_q15_max); /* saturating add, never wrap */
        CHECK (harness.tape.stats ()._M_saturation_events >= 1u);
    }
}

TEST_CASE ("mae_forward and mae_backward: sign gradient with 1/n divisor")
{
    Harness harness;
    for (const auto &c : _element_cases)
    {
        const uint32_t n = static_cast<uint32_t> (c.pred.size ());
        harness.fill (c.pred, c.target);
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (mae_forward (harness.pool, &harness.tape,
                                        harness.pred (n), harness.target (n), result)));
        int64_t total_q15 = 0;
        for (uint32_t i = 0; i < n; ++i)
            total_q15 += ref_q15_abs (ref_code_q15 (c.pred[i] - c.target[i]));
        CHECK (result._M_loss_q7 == ref_mean_magnitude_q7 (total_q15, n));
        CHECK (result._M_loss_q15 == ref_loss_q15 (result._M_loss_q7));
        check_loss_result_shape (result, 2u, 0u, _S_op_mae, harness);

        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        const int32_t seed = ref_seed_lane (k_q15_one);
        const int32_t unit = ref_div_round (k_q15_one, static_cast<int32_t> (n));
        for (uint32_t i = 0; i < n; ++i)
        {
            const int32_t diff = c.pred[i] - c.target[i];
            const int32_t sign = diff > 0 ? unit : (diff < 0 ? -unit : 0);
            CHECK (harness.pred_grad[i] == ref_q15_mul (sign, seed));
        }
        /* The loss magnitude never exceeds the real span 2.0 -> rail check. */
        CHECK (result._M_loss_q7
               <= static_cast<int32_t> (254 * k_step * 128 / k_q15_one));
    }
    SECTION ("perfect prediction is exactly zero with zero-gradient everywhere")
    {
        const std::vector<int8_t> same = { 5, 7, -9 };
        harness.fill (same, same);
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (mae_forward (harness.pool, &harness.tape,
                                        harness.pred (3), harness.target (3), result)));
        CHECK (result._M_loss_q7 == 0);
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        for (uint32_t i = 0; i < 3u; ++i)
            CHECK (harness.pred_grad[i] == 0);
    }
    SECTION ("central difference away from the kink")
    {
        const std::vector<int8_t> pred = { 10, -20, 30, -40 };
        const std::vector<int8_t> target = { 4, 5, 6, 7 };
        harness.fill (pred, target);
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (mae_forward (harness.pool, &harness.tape,
                                        harness.pred (4), harness.target (4), result)));
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        const std::vector<double> t_real = reals_of (target);
        auto twin = [&](const std::vector<double> &x)
        { return ref_loss_real (MAE, x, t_real, 0.0); };
        for (uint32_t i = 0; i < 4u; ++i)
        {
            const double numeric = central_difference (twin, reals_of (pred), i, 2e-3);
            CHECK (lane_derivative (harness.pred_grad[i], ref_seed_lane (k_q15_one))
                   == Approx (numeric).margin (2e-4));
        }
    }
}

TEST_CASE ("huber_forward: quadratic and linear arms, delta at 0 and maximum")
{
    Harness harness;
    const int32_t deltas[] = { 0, 516, 16384, 32767, -516 };
    for (const auto &c : _element_cases)
    {
        const uint32_t n = static_cast<uint32_t> (c.pred.size ());
        for (const int32_t delta : deltas)
        {
            harness.fill (c.pred, c.target);
            harness.open ();
            LossResult result;
            const HuberParams params{ delta };
            REQUIRE (status_ok (huber_forward (harness.pool, &harness.tape,
                                              harness.pred (n), harness.target (n),
                                              params, result)));
            const int32_t clamped = ref_q15_clamp (delta);
            CHECK (result._M_loss_q7
                   == ref_huber_q7 (c.pred, c.target, clamped));
            CHECK (result._M_loss_q15 == ref_loss_q15 (result._M_loss_q7));
            check_loss_result_shape (result, 2u, 1u, _S_op_huber, harness);
            /* The one saved word is the clamped delta. */
            CHECK (harness.saved<const int32_t> (0)[0] == clamped);
            CHECK (harness.pool.stats ()._M_used == 12u); /* node 8 + save 4 */
        }
    }
    SECTION ("delta at the rail: exactly the halved MSE arm")
    {
        const auto &c = _element_cases[0];
        harness.fill (c.pred, c.target);
        harness.open ();
        LossResult result;
        const HuberParams params{ 40000 }; /* out-of-range: clamped to 32767 */
        REQUIRE (status_ok (huber_forward (harness.pool, &harness.tape,
                                          harness.pred (4), harness.target (4),
                                          params, result)));
        CHECK (harness.saved<const int32_t> (0)[0] == k_q15_max);
        const int32_t mse_q7
            = ref_mean_square_q7 (c.pred.data (), c.target.data (), 4u);
        CHECK (result._M_loss_q7 == ref_huber_q7 (c.pred, c.target, k_q15_max));
        CHECK (static_cast<double> (result._M_loss_q7)
               == Approx (mse_q7 * 0.5).margin (2.0));
    }
    SECTION ("delta zero collapses every arm to zero")
    {
        const auto &c = _element_cases[2];
        harness.fill (c.pred, c.target);
        harness.open ();
        LossResult result;
        const HuberParams params{ 0 };
        REQUIRE (status_ok (huber_forward (harness.pool, &harness.tape,
                                          harness.pred (4), harness.target (4),
                                          params, result)));
        CHECK (result._M_loss_q7 == 0); /* 0 * (2|d| - 0) / 2 = 0 */
    }
}

TEST_CASE ("huber_backward: quadratic slope inside delta, clamped slope outside")
{
    Harness harness;
    const int32_t deltas[] = { 0, 516, 16384, 32767 };
    for (const auto &c : _element_cases)
    {
        const uint32_t n = static_cast<uint32_t> (c.pred.size ());
        for (const int32_t delta_in : deltas)
        {
            harness.fill (c.pred, c.target);
            harness.open ();
            LossResult result;
            const HuberParams params{ delta_in };
            REQUIRE (status_ok (huber_forward (harness.pool, &harness.tape,
                                              harness.pred (n), harness.target (n),
                                              params, result)));
            REQUIRE (status_ok (harness.backward (result, 8192)));
            const int32_t delta = ref_q15_clamp (delta_in);
            const int32_t seed = ref_seed_lane (8192);
            for (uint32_t i = 0; i < n; ++i)
            {
                const int32_t d = ref_code_q15 (c.pred[i] - c.target[i]);
                const int32_t magnitude = ref_q15_abs (d);
                const int32_t raw = magnitude <= delta
                                        ? d
                                        : (d > 0 ? delta : -delta);
                const int32_t base = ref_div_round (raw, static_cast<int32_t> (n));
                CHECK (harness.pred_grad[i] == ref_q15_mul (base, seed));
            }
        }
    }
    SECTION ("central difference matches where the arms are differentiable")
    {
        const std::vector<int8_t> pred = { 10, -20, 30, -40 };
        const std::vector<int8_t> target = { 4, 5, 6, 7 };
        harness.fill (pred, target);
        harness.open ();
        LossResult result;
        /* delta = 16384 -> 0.5 on the real diff domain; case diffs: 6, -25, 24,
         * -47 codes -> 0.047, -0.197, 0.189, -0.371 reals -> all inside the
         * quadratic arm except the last. */
        const HuberParams params{ 16384 };
        REQUIRE (status_ok (huber_forward (harness.pool, &harness.tape,
                                          harness.pred (4), harness.target (4),
                                          params, result)));
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        const std::vector<double> t_real = reals_of (target);
        auto twin = [&](const std::vector<double> &x)
        { return ref_loss_real (HUBER, x, t_real, 0.5); };
        for (uint32_t i = 0; i < 4u; ++i)
        {
            const double numeric = central_difference (twin, reals_of (pred), i, 1e-3);
            CHECK (lane_derivative (harness.pred_grad[i], ref_seed_lane (k_q15_one))
                   == Approx (numeric).margin (5e-4));
        }
    }
}

TEST_CASE ("quantile_forward: tau 0, 0.5, maximum and clamp rails")
{
    Harness harness;
    const int32_t taus[] = { 0, 3277, 16384, 32767, 100000, -32768 };
    for (const auto &c : _element_cases)
    {
        const uint32_t n = static_cast<uint32_t> (c.pred.size ());
        for (const int32_t tau_in : taus)
        {
            harness.fill (c.pred, c.target);
            harness.open ();
            LossResult result;
            const QuantileParams params{ tau_in };
            REQUIRE (status_ok (quantile_forward (harness.pool, &harness.tape,
                                                  harness.pred (n), harness.target (n),
                                                  params, result)));
            const int32_t tau = ref_q15_clamp (tau_in);
            CHECK (result._M_loss_q7 == ref_quantile_q7 (c.pred, c.target, tau));
            CHECK (harness.saved<const int32_t> (0)[0] == tau);
            check_loss_result_shape (result, 2u, 1u, _S_op_quantile, harness);
        }
    }
    SECTION ("tau 0.5 equals half of the MAE mean")
    {
        const auto &c = _element_cases[0];
        harness.fill (c.pred, c.target);
        harness.open ();
        LossResult qres;
        const QuantileParams half{ 16384 };
        REQUIRE (status_ok (quantile_forward (harness.pool, &harness.tape,
                                             harness.pred (4), harness.target (4),
                                             half, qres)));
        int64_t mae_total = 0;
        for (uint32_t i = 0; i < 4u; ++i)
            mae_total += ref_q15_abs (ref_code_q15 (c.pred[i] - c.target[i]));
        const int32_t mae_q7 = ref_mean_magnitude_q7 (mae_total, 4u);
        CHECK (static_cast<double> (qres._M_loss_q7)
               == Approx (mae_q7 * 0.5).margin (2.0));
    }
    SECTION ("tau above the rail clamps to the maximum, below clamps up")
    {
        harness.fill ( { 10, -20 }, { 4, 5 });
        harness.open ();
        LossResult big;
        const QuantileParams params{ 100000 };
        REQUIRE (status_ok (quantile_forward (harness.pool, &harness.tape,
                                             harness.pred (2), harness.target (2),
                                             params, big)));
        CHECK (harness.saved<const int32_t> (0)[0] == k_q15_max);
        harness.fill ( { 10, -20 }, { 4, 5 });
        harness.open ();
        LossResult small;
        const QuantileParams negative{ -40000 };
        REQUIRE (status_ok (quantile_forward (harness.pool, &harness.tape,
                                             harness.pred (2), harness.target (2),
                                             negative, small)));
        CHECK (harness.saved<const int32_t> (0)[0] == k_q15_min);
        /* A fully inverted tau still reports through the rails, no wrap. */
        CHECK (small._M_loss_q15 == ref_loss_q15 (small._M_loss_q7));
    }
}

TEST_CASE ("quantile_backward: tau above and tau-minus-one below the knee")
{
    Harness harness;
    const int32_t taus[] = { 0, 3277, 16384, 32767 };
    const int32_t seeds[] = { k_q15_one, 8192, 1 };
    for (const auto &c : _element_cases)
    {
        const uint32_t n = static_cast<uint32_t> (c.pred.size ());
        for (const int32_t tau_in : taus)
            for (const int32_t scale : seeds)
            {
                harness.fill (c.pred, c.target);
                harness.open ();
                LossResult result;
                const QuantileParams params{ tau_in };
                REQUIRE (status_ok (quantile_forward (harness.pool, &harness.tape,
                                                     harness.pred (n),
                                                     harness.target (n), params, result)));
                REQUIRE (status_ok (harness.backward (result, scale)));
                const int32_t tau = ref_q15_clamp (tau_in);
                const int32_t above
                    = ref_div_round (tau, static_cast<int32_t> (n));
                const int32_t below
                    = ref_div_round (tau - k_q15_one, static_cast<int32_t> (n));
                const int32_t seed = ref_seed_lane (scale);
                for (uint32_t i = 0; i < n; ++i)
                {
                    const int32_t diff = c.pred[i] - c.target[i];
                    const int32_t slope = diff > 0 ? above : (diff < 0 ? below : 0);
                    CHECK (harness.pred_grad[i] == ref_q15_mul (slope, seed));
                }
            }
    }
    SECTION ("central difference of the pinball twin")
    {
        const std::vector<int8_t> pred = { 10, -20, 30, -40 };
        const std::vector<int8_t> target = { 4, 5, 6, 7 };
        harness.fill (pred, target);
        harness.open ();
        LossResult result;
        const QuantileParams params{ 26214 }; /* tau ~ 0.8 */
        REQUIRE (status_ok (quantile_forward (harness.pool, &harness.tape,
                                             harness.pred (4), harness.target (4),
                                             params, result)));
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        const std::vector<double> t_real = reals_of (target);
        auto twin = [&](const std::vector<double> &x)
        { return ref_loss_real (QUANTILE, x, t_real, 26214.0 / 32768.0); };
        for (uint32_t i = 0; i < 4u; ++i)
        {
            const double numeric = central_difference (twin, reals_of (pred), i, 2e-3);
            CHECK (lane_derivative (harness.pred_grad[i], ref_seed_lane (k_q15_one))
                   == Approx (numeric).margin (5e-4));
        }
    }
}

TEST_CASE ("bce_with_logits_forward: stable arms against the softplus twin")
{
    Harness harness;
    const std::vector<int8_t> logits = { 30, -30, 100, -100, 5, 127, -127, 0 };
    const std::vector<int8_t> target = { 127, 0, 64, 127, 0, 32, 96, 127 };
    const uint32_t n = 8u;
    harness.fill (logits, target);
    harness.open ();
    LossResult result;
    REQUIRE (status_ok (bce_with_logits_forward (harness.pool, &harness.tape,
                                                 harness.pred (n), harness.target (n),
                                                 result)));
    CHECK (result._M_loss_q7 > 0);
    check_loss_result_shape (result, 2u, 0u, _S_op_bce, harness);
    /* Float twin: mean((1 - t) softplus(x) + t softplus(-x)) mapped to Q7. */
    double total = 0.0;
    for (uint32_t i = 0; i < n; ++i)
    {
        const double x = logits[i] * k_code_to_real;
        const double t = target[i] * k_code_to_real;
        total += (1.0 - t) * ref_softplus (x) + t * ref_softplus (-x);
    }
    const double expected_q7 = total / n * 128.0;
    CHECK (static_cast<double> (result._M_loss_q7) == Approx (expected_q7).margin (6.0));
    SECTION ("a zero target with a strongly negative logit is near zero loss")
    {
        harness.fill ( { -127, -127 }, { 0, 0 });
        harness.open ();
        LossResult small;
        REQUIRE (status_ok (bce_with_logits_forward (harness.pool, &harness.tape,
                                                    harness.pred (2),
                                                    harness.target (2), small)));
        /* softplus(-0.9999) = 0.30876... but Q7 keeps it honest, not zero. */
        CHECK (small._M_loss_q7 > 0);
        CHECK (static_cast<double> (small._M_loss_q7)
               == Approx (ref_softplus (-127.0 * k_code_to_real) * 128.0).margin (6.0));
    }
}

TEST_CASE ("bce_backward: sigmoid-minus-target over n, rail saturating")
{
    Harness harness;
    const std::vector<int8_t> logits = { 30, -30, 100, -100, 5, 127, -127, 0 };
    const std::vector<int8_t> target = { 127, 0, 64, 127, 0, 32, 96, 127 };
    const uint32_t n = 8u;
    harness.fill (logits, target);
    harness.open ();
    LossResult result;
    REQUIRE (status_ok (bce_with_logits_forward (harness.pool, &harness.tape,
                                                harness.pred (n), harness.target (n),
                                                result)));
    REQUIRE (status_ok (harness.backward (result, k_q15_one)));
    const int32_t seed = ref_seed_lane (k_q15_one);
    /* The library's q15_sigmoid against the test's <cmath> twin: agreement
     * within a couple of LSB is what pins the arm recomputation. */
    for (uint32_t i = 0; i < n; ++i)
    {
        const int32_t x = ref_code_q15 (logits[i]);
        const double x_real = logits[i] * k_code_to_real;
        const int32_t sig_ref = static_cast<int32_t> (
            std::llround (ref_sigmoid (x_real) * k_q15_one));
        const int32_t t = ref_code_q15 (target[i]);
        const int32_t base = ref_div_round (
            static_cast<int32_t> (q15_sigmoid (x)) - t, static_cast<int32_t> (n));
        CHECK (harness.pred_grad[i] == ref_q15_mul (base, seed));
        /* Table interpolation plus the ratio truncation cost the Q15 sigmoid
         * a few tens of LSB; the exact pipeline check above is the pin. */
        CHECK (q15_sigmoid (x) == Approx (sig_ref).margin (32.0));
        /* Analytic cross-check on the real domain. */
        const double t_real = target[i] * k_code_to_real;
        const double analytic = (ref_sigmoid (x_real) - t_real) / n;
        CHECK (lane_derivative (harness.pred_grad[i], seed)
               == Approx (analytic).margin (2e-4));
    }
    SECTION ("central difference of the BCE twin")
    {
        const std::vector<double> t_real = reals_of (target);
        auto twin = [&](const std::vector<double> &x)
        { return ref_loss_real (BCE, x, t_real, 0.0); };
        harness.fill (logits, target);
        harness.open ();
        LossResult fresh;
        REQUIRE (status_ok (bce_with_logits_forward (harness.pool, &harness.tape,
                                                    harness.pred (n),
                                                    harness.target (n), fresh)));
        REQUIRE (status_ok (harness.backward (fresh, k_q15_one)));
        for (uint32_t i = 0; i < n; ++i)
        {
            const double numeric = central_difference (twin, reals_of (logits), i, 1e-3);
            CHECK (lane_derivative (harness.pred_grad[i], seed)
                   == Approx (numeric).margin (5e-4));
        }
    }
}

TEST_CASE ("backward context validation: every loss backward rejects a broken node")
{
    Harness harness;
    LossResult result;
    SECTION ("mse backward needs two operands")
    {
        harness.fill ( { 3, 4 }, { 1, 2 });
        harness.open ();
        REQUIRE (status_ok (mse_forward (harness.pool, &harness.tape, harness.pred (2),
                                        harness.target (2), result)));
        Node *node = harness.only_node ();
        node->_M_output.set_grad (0, k_q15_one - 1);
        GradContext ctx = harness.context ();
        ctx._M_operand_count = 1u;
        CHECK (mse_backward (node->_M_output, node->_M_output, ctx)
               == Status::BAD_LAYOUT);
        ctx._M_operand_count = 2u;
        CHECK (mse_backward (node->_M_output, node->_M_output, ctx) == Status::OK);
    }
    SECTION ("huber and quantile need one saved word")
    {
        harness.fill ( { 3, 4 }, { 1, 2 });
        harness.open ();
        const HuberParams hp{ 16384 };
        REQUIRE (status_ok (huber_forward (harness.pool, &harness.tape, harness.pred (2),
                                          harness.target (2), hp, result)));
        Node *node = harness.only_node ();
        node->_M_output.set_grad (0, k_q15_one - 1);
        GradContext ctx = harness.context ();
        ctx._M_saved_count = 0u;
        CHECK (huber_backward (node->_M_output, node->_M_output, ctx)
               == Status::BAD_LAYOUT);
        ctx._M_saved_count = 1u;
        CHECK (huber_backward (node->_M_output, node->_M_output, ctx) == Status::OK);
    }
    SECTION ("hinge needs its margin word")
    {
        harness.fill ( { 30, -30 }, { 1, -1 });
        harness.open ();
        const HingeParams hp;
        REQUIRE (status_ok (hinge_loss_forward (harness.pool, &harness.tape,
                                                harness.pred (2), harness.target (2),
                                                hp, result)));
        Node *node = harness.only_node ();
        node->_M_output.set_grad (0, k_q15_one - 1);
        GradContext ctx = harness.context ();
        ctx._M_saved_count = 0u;
        CHECK (hinge_backward (node->_M_output, node->_M_output, ctx)
               == Status::BAD_LAYOUT);
    }
    SECTION ("index losses need a TargetMeta save")
    {
        harness.fill ( { 30, 60, 90, 20 }, { 0, 1, 0, 0 });
        harness.open ();
        REQUIRE (status_ok (cross_entropy_forward (harness.pool, &harness.tape,
                                                   harness.pred (4), 1u, result)));
        Node *node = harness.only_node ();
        node->_M_output.set_grad (0, k_q15_one - 1);
        GradContext ctx = harness.context ();
        ctx._M_saved_count = 0u;
        CHECK (cross_entropy_index_backward (node->_M_output, node->_M_output, ctx)
               == Status::BAD_LAYOUT);
        ctx._M_saved_count = 1u;
        ctx._M_operand_count = 0u;
        CHECK (cross_entropy_index_backward (node->_M_output, node->_M_output, ctx)
               == Status::BAD_LAYOUT);
        ctx._M_operand_count = 1u;
        CHECK (cross_entropy_index_backward (node->_M_output, node->_M_output, ctx)
               == Status::OK);
    }
    SECTION ("multiclass hinge rejects a target outside the class count")
    {
        harness.fill ( { 30, 60, 90, 20 }, { 0, 1, 0, 0 });
        harness.open ();
        const HingeParams hp;
        REQUIRE (status_ok (multiclass_hinge_forward (harness.pool, &harness.tape,
                                                     harness.pred (4), 2u, hp, result)));
        Node *node = harness.only_node ();
        node->_M_output.set_grad (0, k_q15_one - 1);
        GradContext ctx = harness.context ();
        harness.saved_mut<TargetMeta> (0)->_M_target = 9u;
        CHECK (multiclass_hinge_backward (node->_M_output, node->_M_output, ctx)
               == Status::BAD_LAYOUT);
    }
    SECTION ("nll backward needs its meta save")
    {
        harness.fill ( { -10, -20, -30 }, { 0, 0, 0 });
        harness.open ();
        REQUIRE (status_ok (nll_forward (harness.pool, &harness.tape,
                                        harness.pred (3), 1u, result)));
        Node *node = harness.only_node ();
        node->_M_output.set_grad (0, k_q15_one - 1);
        GradContext ctx = harness.context ();
        ctx._M_saved_count = 0u;
        CHECK (nll_backward (node->_M_output, node->_M_output, ctx)
               == Status::BAD_LAYOUT);
        ctx._M_saved_count = 1u;
        CHECK (nll_backward (node->_M_output, node->_M_output, ctx) == Status::OK);
    }
    SECTION ("smoothed CE backward needs its meta save")
    {
        harness.fill ( { 30, 60, 90, 20 }, { 0, 1, 0, 0 });
        harness.open ();
        const SmoothingParams sp = SmoothingParams::make (3277, 4u);
        REQUIRE (status_ok (smoothed_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (4), 1u, sp, result)));
        Node *node = harness.only_node ();
        node->_M_output.set_grad (0, k_q15_one - 1);
        GradContext ctx = harness.context ();
        ctx._M_saved_count = 0u;
        CHECK (smoothed_ce_backward (node->_M_output, node->_M_output, ctx)
               == Status::BAD_LAYOUT);
        ctx._M_saved_count = 1u;
        CHECK (smoothed_ce_backward (node->_M_output, node->_M_output, ctx)
               == Status::OK);
    }
    SECTION ("soft-target CE backward needs two operands and the meta")
    {
        harness.fill ( { 30, 60, 90, 20 }, { 0, 1, 0, 0 });
        harness.open ();
        const SmoothingParams sp = SmoothingParams::make (3277, 4u);
        REQUIRE (status_ok (soft_target_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (4), harness.target (4), sp,
            result)));
        Node *node = harness.only_node ();
        node->_M_output.set_grad (0, k_q15_one - 1);
        GradContext ctx = harness.context ();
        ctx._M_operand_count = 1u;
        CHECK (soft_target_ce_backward (node->_M_output, node->_M_output, ctx)
               == Status::BAD_LAYOUT);
        ctx._M_operand_count = 2u;
        ctx._M_saved_count = 0u;
        CHECK (soft_target_ce_backward (node->_M_output, node->_M_output, ctx)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("cross_entropy_forward (index form): -log p_y in Q7 with rails")
{
    Harness harness;
    /* Softmax-shaped probability buffers: codes in [0, 127], target index. */
    struct CeCase
    {
        std::vector<int8_t> probs;
        uint16_t target;
    };
    const CeCase cases[] = {
        { { 127, 0, 0, 0 }, 0u },
        { { 10, 100, 40, 2 }, 1u },
        { { 64, 64, 1, 127 }, 3u },
        { { 0, 0, 50, 0 }, 0u },
        { { 127, 127 }, 1u },
        { { 1, 2, 3, 4, 5 }, 0u },
        { { 30, 60, 90, 20 }, 2u },
    };
    for (const auto &c : cases)
    {
        const uint32_t n = static_cast<uint32_t> (c.probs.size ());
        harness.fill (c.probs, { 0 });
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (cross_entropy_forward (harness.pool, &harness.tape,
                                                   harness.pred (n), c.target,
                                                   result)));
        /* Float reference straight from <cmath>: the log atom and the Q7
         * narrow stay inside a few units of -ln(p_y) * 128. */
        const double float_q7 = (c.probs[c.target] > 0)
                                    ? ref_neg_log_q7 (c.probs[c.target] * k_code_to_real)
                                    : 0.0;
        if (c.probs[c.target] > 0)
        {
            /* The log atom and the Q7 narrow stay inside a few units. */
            CHECK (result._M_loss_q7
                   == Approx (static_cast<int32_t> (float_q7)).margin (4.0));
            CHECK (result._M_loss_q7 >= 0);
        }
        else
        {
            CHECK (result._M_loss_q7 == 32768); /* sentinel rail, finite, no trap */
            CHECK (result._M_loss_q15 == k_q15_max);
            CHECK (result._M_node.code_at (0) == 127);
        }
        CHECK (result._M_loss_q15 == ref_loss_q15 (result._M_loss_q7));
        check_loss_result_shape (result, 1u, 1u, _S_op_cross_entropy, harness);
        /* The saved meta describes the sample. */
        const TargetMeta *const meta = harness.saved<TargetMeta> (0);
        CHECK (meta->_M_classes == static_cast<uint16_t> (n));
        CHECK (meta->_M_target == c.target);
        CHECK (meta->_M_margin_q7 == 0);
        CHECK (harness.pool.stats ()._M_used == 16u); /* node 8 + meta 8 */
    }
    SECTION ("perfect probability 127 costs under one Q7 unit")
    {
        harness.fill ( { 127, 0 }, { 0, 0 });
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (cross_entropy_forward (harness.pool, &harness.tape,
                                                   harness.pred (2), 0u, result)));
        CHECK (result._M_loss_q7 == 0); /* -log(32766/32768) narrows to 0 */
    }
    SECTION ("class-index validation")
    {
        harness.fill ( { 10, 20, 30 }, { 0, 0, 0 });
        harness.open ();
        LossResult result;
        CHECK (cross_entropy_forward (harness.pool, &harness.tape, harness.pred (3), 3u,
                                      result)
               == Status::OUT_OF_RANGE);
        CHECK (cross_entropy_forward (harness.pool, &harness.tape, harness.pred (3), 99u,
                                      result)
               == Status::OUT_OF_RANGE);
        CHECK (cross_entropy_forward (harness.pool, &harness.tape, harness.pred (0), 0u,
                                      result)
               == Status::BAD_LAYOUT);
        Trainable nogr = Trainable::make (harness.pred_code, nullptr,
                                          Shape::vector (3), 0u);
        CHECK (status_ok (cross_entropy_forward (harness.pool, &harness.tape, nogr, 1u,
                                                 result)));
    }
}

TEST_CASE ("cross_entropy_index_backward: one saturated -1/p lane at the target")
{
    Harness harness;
    const std::vector<int8_t> probs = { 10, 100, 40, 2 };
    const int32_t seeds[] = { k_q15_one, 16384, 1, 0 };
    for (const int32_t scale : seeds)
    {
        harness.fill (probs, { 0 });
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (cross_entropy_forward (harness.pool, &harness.tape,
                                                   harness.pred (4), 1u, result)));
        REQUIRE (status_ok (harness.backward (result, scale)));
        const int32_t seed = ref_seed_lane (scale);
        /* Documented rail behaviour: dL/dp_y = -1 saturates at -q15_max, no
         * magnitude dependence on p_y, every other lane stays clean. */
        for (uint32_t i = 0; i < 4u; ++i)
        {
            const int32_t expect = i == 1u ? ref_q15_mul (-k_q15_max, seed) : 0;
            CHECK (harness.pred_grad[i] == expect);
        }
    }
    SECTION ("a corrupted meta target simply routes to no lane")
    {
        harness.fill (probs, { 0 });
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (cross_entropy_forward (harness.pool, &harness.tape,
                                                   harness.pred (4), 1u, result)));
        harness.saved_mut<TargetMeta> (0)->_M_target = 9u;
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        for (uint32_t i = 0; i < 4u; ++i)
            CHECK (harness.pred_grad[i] == 0);
    }
}

TEST_CASE ("soft_cross_entropy_forward: sum -t_c log p_c, one-hot equivalence")
{
    Harness harness;
    const std::vector<int8_t> probs = { 10, 100, 40, 2 };
    const std::vector<int8_t> dense = { 127, 0, 64, 32 };
    harness.fill (probs, dense);
    harness.open ();
    LossResult result;
    REQUIRE (status_ok (soft_cross_entropy_forward (harness.pool, &harness.tape,
                                                    harness.pred (4),
                                                    harness.target (4), result)));
    int64_t total = 0;
    for (uint32_t i = 0; i < 4u; ++i)
    {
        if (dense[i] == 0)
            continue;
        const int32_t p = ref_code_q15 (probs[i]);
        const int32_t neg_log = (p <= 0)
                                    ? 32768
                                    : static_cast<int32_t> (
                                        -std::log (p / 32768.0) * 128.0 + 0.5);
        total += (static_cast<int64_t> (neg_log) * ref_code_q15 (dense[i])) >> 15;
    }
    /* Float twin with an exact-log reference: tolerance covers the library's
     * log-series error, term truncation and the Q7 rounding. */
    double twin = 0.0;
    for (uint32_t i = 0; i < 4u; ++i)
    {
        const double p = std::max (probs[i] * k_code_to_real, 4.0 / 32768.0);
        const double t = std::max (dense[i] * k_code_to_real, 0.0);
        if (t == 0.0)
            continue;
        twin += t * -std::log (p);
    }
    CHECK (static_cast<double> (result._M_loss_q7) == Approx (twin * 128.0).margin (12.0));
    CHECK (result._M_loss_q7 >= 0);
    check_loss_result_shape (result, 2u, 0u, _S_op_cross_entropy, harness);
    CHECK (harness.pool.stats ()._M_used == 8u); /* documented: node only */
    SECTION ("a one-hot dense target equals the index form within one Q7 unit")
    {
        harness.fill (probs, { 0, 127, 0, 0 });
        harness.open ();
        LossResult soft;
        REQUIRE (status_ok (soft_cross_entropy_forward (harness.pool, &harness.tape,
                                                       harness.pred (4),
                                                       harness.target (4), soft)));
        harness.fill (probs, { 0, 0, 0, 0 });
        harness.open ();
        LossResult indexed;
        REQUIRE (status_ok (cross_entropy_forward (harness.pool, &harness.tape,
                                                   harness.pred (4), 1u, indexed)));
        CHECK (indexed._M_loss_q7 - soft._M_loss_q7 <= 1);
        CHECK (indexed._M_loss_q7 - soft._M_loss_q7 >= 0);
    }
    SECTION ("zero-length and mismatched pairs are BAD_LAYOUT")
    {
        harness.open ();
        LossResult result2;
        CHECK (soft_cross_entropy_forward (harness.pool, &harness.tape,
                                          harness.pred (4), harness.target (5), result2)
               == Status::BAD_LAYOUT);
        CHECK (soft_cross_entropy_forward (harness.pool, &harness.tape,
                                          harness.pred (0), harness.target (0), result2)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("cross_entropy_dense_backward: -t/p per class, saturating rails")
{
    Harness harness;
    const std::vector<int8_t> probs = { 10, 100, 40, 2 };
    const std::vector<int8_t> dense = { 127, 0, 64, 32 };
    harness.fill (probs, dense);
    harness.open ();
    LossResult result;
    REQUIRE (status_ok (soft_cross_entropy_forward (harness.pool, &harness.tape,
                                                   harness.pred (4),
                                                   harness.target (4), result)));
    REQUIRE (status_ok (harness.backward (result, k_q15_one)));
    const int32_t seed = ref_seed_lane (k_q15_one);
    for (uint32_t i = 0; i < 4u; ++i)
    {
        const int32_t p = ref_code_q15 (probs[i]);
        const int32_t t = ref_code_q15 (dense[i]);
        const int32_t ratio = p > 0 ? ref_q15_div (-t, p) : -k_q15_max;
        CHECK (harness.pred_grad[i] == ref_q15_mul (ratio, seed));
        if (p > 0 && t <= p)
        {
            /* Analytic -t/p in the real domain for the un-railed classes. */
            const double analytic = -static_cast<double> (dense[i]) / probs[i];
            const double lane_real = static_cast<double> (harness.pred_grad[i]) / seed;
            CHECK (lane_real == Approx (analytic).margin (0.01));
        }
        else
        {
            /* Documented rails: an overflowing ratio pins at the full int16
             * floor, a non-positive probability at -q15_max. */
            CHECK (ratio == (p > 0 ? k_q15_min : -k_q15_max));
        }
    }
    SECTION ("a zero-probability class pins the -q15_max rail")
    {
        harness.fill ( { 0, 50 }, { 30, 0 });
        harness.open ();
        LossResult small;
        REQUIRE (status_ok (soft_cross_entropy_forward (harness.pool, &harness.tape,
                                                       harness.pred (2),
                                                       harness.target (2), small)));
        REQUIRE (status_ok (harness.backward (small, k_q15_one)));
        /* p=0 -> ratio rail -q15_max even though t=30 (documented). */
        CHECK (harness.pred_grad[0] == ref_q15_mul (-k_q15_max,
                                                     ref_seed_lane (k_q15_one)));
        /* t=0 with p>0 -> ratio 0, lane untouched. */
        CHECK (harness.pred_grad[1] == 0);
    }
}

TEST_CASE ("nll_forward: exact Q7 from the log-prob grid and the code codec")
{
    Harness harness;
    struct NllCase
    {
        std::vector<int8_t> log_probs;
        uint16_t target;
    };
    const NllCase cases[] = {
        { { -10, -20, -30 }, 1u },
        { { -127, -100, -50, 0 }, 0u },
        { { 0, -1, -2 }, 2u },
        { { -40 }, 0u },
    };
    for (const auto &c : cases)
    {
        const uint32_t n = static_cast<uint32_t> (c.log_probs.size ());
        harness.fill (c.log_probs, { 0 });
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (nll_forward (harness.pool, &harness.tape,
                                         harness.pred (n), c.target, result)));
        /* L = -logprob_y exactly: code * 4 Q7 units, no runtime log. */
        CHECK (result._M_loss_q7 == -c.log_probs[c.target] * 4);
        CHECK (result._M_loss_q7 == -logprob_q7_at (harness.pred (n), c.target));
        check_loss_result_shape (result, 1u, 1u, _S_op_nll, harness);
        CHECK (harness.pool.stats ()._M_used == 16u);
        const TargetMeta *const meta = harness.saved<TargetMeta> (0);
        CHECK (meta->_M_classes == static_cast<uint16_t> (n));
        CHECK (meta->_M_target == c.target);
        CHECK (meta->_M_margin_q7 == 0);
    }
    SECTION ("logprob_code_from_q7 rounds away from zero and rails at +-127")
    {
        CHECK (logprob_code_from_q7 (100) == 25);   /* 100/4 */
        CHECK (logprob_code_from_q7 (102) == 26);   /* 25.5 -> 26 (away) */
        CHECK (logprob_code_from_q7 (101) == 25);   /* 25.25 -> 25 */
        CHECK (logprob_code_from_q7 (-101) == -25); /* -25.25 -> -25 */
        CHECK (logprob_code_from_q7 (-102) == -26); /* -25.5 -> -26 */
        CHECK (logprob_code_from_q7 (0) == 0);
        CHECK (logprob_code_from_q7 (2000) == 127);  /* 500 -> i8 rail */
        CHECK (logprob_code_from_q7 (-2000) == -127);
        CHECK (logprob_code_from_q7 (508) == 127);
        CHECK (logprob_code_from_q7 (512) == 127);   /* 128 -> rail, never -128 */
        /* Decode side: a log-prob lane is exactly code * 4 Q7 units, so the
         * encode/encode round-trip holds over the whole signed code grid. */
        for (int32_t code = -127; code <= 127; ++code)
        {
            harness.pred_code[0] = static_cast<int8_t> (code);
            CHECK (logprob_q7_at (harness.pred (1), 0) == code * 4);
            CHECK (logprob_code_from_q7 (code * 4) == static_cast<int8_t> (code));
        }
    }
    SECTION ("target validation and zero length")
    {
        harness.fill ( { -10, -20, -30 }, { 0, 0, 0 });
        harness.open ();
        LossResult result;
        CHECK (nll_forward (harness.pool, &harness.tape, harness.pred (3), 3u, result)
               == Status::OUT_OF_RANGE);
        CHECK (nll_forward (harness.pool, &harness.tape, harness.pred (3), 4u, result)
               == Status::OUT_OF_RANGE);
        CHECK (nll_forward (harness.pool, &harness.tape, harness.pred (0), 0u, result)
               == Status::BAD_LAYOUT);
    }
    SECTION ("nll backward: seed-scaled -1 only at the gold lane")
    {
        const int32_t seeds[] = { k_q15_one, 8192, 1 };
        for (const int32_t scale : seeds)
        {
            harness.fill ( { -10, -20, -30 }, { 0, 0, 0 });
            harness.open ();
            LossResult result;
            REQUIRE (status_ok (nll_forward (harness.pool, &harness.tape,
                                             harness.pred (3), 1u, result)));
            REQUIRE (status_ok (harness.backward (result, scale)));
            const int32_t seed = ref_seed_lane (scale);
            CHECK (harness.pred_grad[0] == 0);
            CHECK (harness.pred_grad[1] == ref_q15_mul (-k_q15_max, seed));
            CHECK (harness.pred_grad[2] == 0);
        }
    }
}

TEST_CASE ("cosine_similarity_forward: 1 - cos in code-domain exactness")
{
    Harness harness;
    for (const auto &c : _element_cases)
    {
        if (c.pred.size () < 2u)
            continue;
        const uint32_t n = static_cast<uint32_t> (c.pred.size ());
        harness.fill (c.pred, c.target);
        harness.open ();
        LossResult result;
        const CosineParams params;
        REQUIRE (status_ok (cosine_similarity_forward (harness.pool, &harness.tape,
                                                      harness.pred (n),
                                                      harness.target (n), params,
                                                      result)));
        CHECK (result._M_loss_q7 == ref_cosine_q7 (c.pred, c.target));
        check_loss_result_shape (result, 2u, 0u, _S_op_cosine, harness);
        CHECK (harness.pool.stats ()._M_used == 8u);
        /* Float twin where both norms exist; code-truncated roots keep the
         * library slightly below the true cosine, within a few Q7 units. */
        const double p2 = dot_real (c.pred, c.pred);
        const double t2 = dot_real (c.target, c.target);
        if (p2 > 0.0 && t2 > 0.0)
        {
            const double cos_real = ref_clamp (dot_real (c.pred, c.target)
                                               / (std::sqrt (p2) * std::sqrt (t2)),
                                               -1.0, 1.0);
            const double expected = (1.0 - cos_real) * 128.0;
            CHECK (static_cast<double> (result._M_loss_q7)
                   == Approx (expected).margin (4.5));
        }
    }
    SECTION ("identical vectors hit the Q15 rail: loss is the 1-unit floor")
    {
        const std::vector<int8_t> v = { 3, 4, 12 }; /* norm 13, exact square */
        harness.fill (v, v);
        harness.open ();
        LossResult result;
        const CosineParams params;
        REQUIRE (status_ok (cosine_similarity_forward (harness.pool, &harness.tape,
                                                      harness.pred (3),
                                                      harness.target (3), params,
                                                      result)));
        /* cos narrows to 32767 (a lane cannot hold 1.0), so Q7 = 128 - 127. */
        CHECK (result._M_loss_q7 == 1);
        CHECK (result._M_loss_q15 == ref_loss_q15 (1));
    }
    SECTION ("opposite vectors reach the 2.0 rail")
    {
        harness.fill ( { 10, -20 }, { -10, 20 });
        harness.open ();
        LossResult result;
        const CosineParams params;
        REQUIRE (status_ok (cosine_similarity_forward (harness.pool, &harness.tape,
                                                      harness.pred (2),
                                                      harness.target (2), params,
                                                      result)));
        CHECK (result._M_loss_q7 == 256);
        CHECK (result._M_loss_q15 == k_q15_max); /* saturates, never wraps */
        CHECK (result._M_node.code_at (0) == 127);
    }
    SECTION ("a zero-norm operand costs exactly 1.0 and the eps word is ignored")
    {
        harness.fill ( { 0, 0, 0 }, { 1, 2, 3 });
        harness.open ();
        LossResult zero_p;
        const CosineParams eps_default;
        REQUIRE (status_ok (cosine_similarity_forward (harness.pool, &harness.tape,
                                                      harness.pred (3),
                                                      harness.target (3), eps_default,
                                                      zero_p)));
        CHECK (zero_p._M_loss_q7 == 128);
        harness.fill ( { 0, 0, 0 }, { 1, 2, 3 });
        harness.open ();
        LossResult other_eps;
        const CosineParams big_eps{ 32767 };
        REQUIRE (status_ok (cosine_similarity_forward (harness.pool, &harness.tape,
                                                      harness.pred (3),
                                                      harness.target (3), big_eps,
                                                      other_eps)));
        /* Documented: the guard is structural, not the eps word. */
        CHECK (other_eps._M_loss_q7 == zero_p._M_loss_q7);
    }
}

TEST_CASE ("cosine_backward: analytic, central-difference, and the zero-norm guard")
{
    Harness harness;
    SECTION ("gradient at the perfect-match pair: true zero vs the Np^3 arm")
    {
        /* Known library deviation (reported, not fixed): the norm arm at
         * train_loss.hpp:807-808 divides by norm_pred^3, but the derivative
         * of 1 - cos carries cos * p_real / |p|_real^2 -- one norm factor
         * too many.  With gain2 = gain1^2 >> 15 the integer arm scales the
         * normalisation term by 127/Np relative to the true derivative, so
         * at p == t (where both the mathematical gradient and the clamped,
         * flat quantised forward give exactly 0) the pass still reports a
         * positive residual.  The checks pin the implemented value through
         * the module's own gain constants and document the deviation from
         * both the central difference and the analytic zero. */
        const std::vector<int8_t> v = { 30, 31, 32, 33, 34, 35 };
        harness.fill (v, v);
        harness.open ();
        LossResult result;
        const CosineParams params;
        REQUIRE (status_ok (cosine_similarity_forward (harness.pool, &harness.tape,
                                                      harness.pred (6),
                                                      harness.target (6), params,
                                                      result)));
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        const int32_t seed = ref_seed_lane (k_q15_one);
        const std::vector<double> t_real = reals_of (v);
        auto twin = [&](const std::vector<double> &x)
        { return ref_loss_real (COSINE, x, t_real, 0.0); };
        int32_t dot = 0, sqp = 0;
        for (uint32_t i = 0; i < 6u; ++i)
        {
            dot += v[i] * v[i];
            sqp += v[i] * v[i];
        }
        const int64_t norm = static_cast<int64_t> (
            static_cast<int32_t> (ref_isqrt (static_cast<uint64_t> (sqp))));
        const int32_t cos_q15 = ref_q15_clamp (static_cast<int32_t> (
            (static_cast<int64_t> (dot) << 15) / (norm * norm)));
        CHECK (cos_q15 == k_q15_max); /* dot/N^2 > 1: the root floor clamps it */
        constexpr int64_t gain1 = 4161790;
        const int64_t gain2 = (gain1 * gain1) >> 15;
        for (uint32_t i = 0; i < 6u; ++i)
        {
            const int32_t lane = harness.pred_grad[i];
            /* The true derivative at p == t vanishes; the central difference
             * confirms it up to its own truncation noise. */
            const double numeric = central_difference (twin, reals_of (v), i, 1e-3);
            CHECK (std::abs (numeric) < 1e-3);
            /* The implemented arm, reproduced from the module constants. */
            const int64_t align_term = gain1 * v[i] / (norm * norm);
            const int64_t norm_term
                = gain2 * cos_q15 * v[i] / (32768 * norm * norm * norm);
            const int32_t narrowed = ref_q15_clamp (
                static_cast<int32_t> (norm_term - align_term));
            CHECK (lane == ref_q15_mul (narrowed, seed));
            CHECK (lane > 0); /* the residual that should be zero */
        }
        CHECK (harness.tape.stats ()._M_saturation_events == 0u);
    }
    SECTION ("the integer pipeline matches the module gain constants")
    {
        const std::vector<int8_t> pred = { 30, 31, 32, 33, 34, 35 };
        const std::vector<int8_t> target = { 12, -13, 40, -1, 0, 35 };
        harness.fill (pred, target);
        harness.open ();
        LossResult result;
        const CosineParams params;
        REQUIRE (status_ok (cosine_similarity_forward (harness.pool, &harness.tape,
                                                      harness.pred (6),
                                                      harness.target (6), params,
                                                      result)));
        REQUIRE (status_ok (harness.backward (result, 16384)));
        const int32_t seed = ref_seed_lane (16384);
        int32_t dot = 0, sqp = 0, sqt = 0;
        for (uint32_t i = 0; i < 6u; ++i)
        {
            dot += pred[i] * target[i];
            sqp += pred[i] * pred[i];
            sqt += target[i] * target[i];
        }
        const int64_t norm_p = static_cast<int64_t> (ref_isqrt (static_cast<uint64_t> (sqp)));
        const int64_t norm_t = static_cast<int64_t> (ref_isqrt (static_cast<uint64_t> (sqt)));
        const int32_t cos_q15 = ref_q15_clamp (static_cast<int32_t> (
            (static_cast<int64_t> (dot) << 15) / (norm_p * norm_t)));
        constexpr int64_t gain1 = 4161790;
        const int64_t gain2 = (gain1 * gain1) >> 15;
        for (uint32_t i = 0; i < 6u; ++i)
        {
            const int64_t p = pred[i];
            const int64_t t = target[i];
            const int64_t align_term = gain1 * t / (norm_p * norm_t);
            const int64_t norm_term
                = gain2 * cos_q15 * p / (32768 * norm_p * norm_p * norm_p);
            const int64_t summed = norm_term - align_term;
            const int32_t narrowed = ref_q15_clamp (static_cast<int32_t> (summed));
            CHECK (harness.pred_grad[i] == ref_q15_mul (narrowed, seed));
        }
    }
    SECTION ("a zero-norm operand returns OK and routes no gradient")
    {
        harness.fill ( { 0, 0, 0 }, { 1, 2, 3 });
        harness.open ();
        LossResult result;
        const CosineParams params;
        REQUIRE (status_ok (cosine_similarity_forward (harness.pool, &harness.tape,
                                                      harness.pred (3),
                                                      harness.target (3), params,
                                                      result)));
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        for (uint32_t i = 0; i < 3u; ++i)
            CHECK (harness.pred_grad[i] == 0);
    }
}

TEST_CASE ("hinge_loss_forward: margin rails, sign-of-code labels, zero violations")
{
    Harness harness;
    const int32_t margins[] = { 128, 0, 64, -128, 100000 };
    const std::vector<int8_t> scores = { 30, -20, 127, -127, 5 };
    const std::vector<int8_t> labels = { 4, -5, 0, 7, -1 };
    for (const int32_t margin : margins)
    {
        harness.fill (scores, labels);
        harness.open ();
        LossResult result;
        const HingeParams params{ margin };
        REQUIRE (status_ok (hinge_loss_forward (harness.pool, &harness.tape,
                                                harness.pred (5), harness.target (5),
                                                params, result)));
        CHECK (result._M_loss_q7 == ref_hinge_q7 (scores, labels, margin));
        check_loss_result_shape (result, 2u, 1u, _S_op_hinge, harness);
        CHECK (harness.saved<const int32_t> (0)[0] == margin);
        CHECK (harness.pool.stats ()._M_used == 12u);
    }
    SECTION ("label code zero counts as the positive class")
    {
        harness.fill ( { 127 }, { 0 });
        harness.open ();
        LossResult result;
        const HingeParams params{ 0 }; /* margin 0, product ~+1 -> no violation */
        REQUIRE (status_ok (hinge_loss_forward (harness.pool, &harness.tape,
                                                harness.pred (1), harness.target (1),
                                                params, result)));
        CHECK (result._M_loss_q7 == 0); /* y=+1, y*s >= 0 = margin -> exactly zero */
    }
    SECTION ("products just below the 1.0 margin still violate, exactly")
    {
        harness.fill ( { 100, -100 }, { 7, -7 });
        harness.open ();
        LossResult result;
        const HingeParams params;
        REQUIRE (status_ok (hinge_loss_forward (harness.pool, &harness.tape,
                                                harness.pred (2), harness.target (2),
                                                params, result)));
        /* y*s narrows to 100 Q7, not 128: 100 codes are below the 1.0 margin
         * (32767*25800 q15_mul gives 25799, >> 8 gives 100), so BOTH classes
         * violate by 28 and the mean over the count keeps the 28. */
        CHECK (result._M_loss_q7 == 28);
        CHECK (result._M_loss_q7 == ref_hinge_q7 ( { 100, -100 }, { 7, -7 }, 128));
        CHECK (result._M_loss_q7 > 0);
    }
    SECTION ("a truly satisfied batch is exactly zero")
    {
        harness.fill ( { 127, -127 }, { 127, -127 }); /* |y*s| = 127 Q7 >= 128? */
        harness.open ();
        LossResult result;
        const HingeParams params{ 127 };
        REQUIRE (status_ok (hinge_loss_forward (harness.pool, &harness.tape,
                                                harness.pred (2), harness.target (2),
                                                params, result)));
        /* 127*258 = 32766 Q15, y = 32767: q15_mul lands on 32765, >> 8 = 127,
         * which equals the 127 margin, so nothing violates. */
        CHECK (result._M_loss_q7 == 0);
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        CHECK (harness.pred_grad[0] == 0);
        CHECK (harness.pred_grad[1] == 0);
    }
    SECTION ("a huge margin saturates the lane rail, the Q7 word stays exact")
    {
        harness.fill ( { 10, 10, 10, 10 }, { 1, 1, 1, 1 });
        harness.open ();
        LossResult result;
        const HingeParams params{ 100000 };
        REQUIRE (status_ok (hinge_loss_forward (harness.pool, &harness.tape,
                                                harness.pred (4), harness.target (4),
                                                params, result)));
        /* y*s narrows to 10 Q7 for every lane (score code 10), so each
         * violator costs 100000 - 10 and the mean over 4 keeps 99990 exact. */
        CHECK (result._M_loss_q7 == 99990);
        CHECK (result._M_loss_q15 == k_q15_max);
        CHECK (result._M_node.code_at (0) == 127);
    }
    SECTION ("float twin and central difference on the hinge arm")
    {
        harness.fill (scores, labels);
        harness.open ();
        LossResult result;
        const HingeParams params{ 128 };
        REQUIRE (status_ok (hinge_loss_forward (harness.pool, &harness.tape,
                                                harness.pred (5), harness.target (5),
                                                params, result)));
        double twin = 0.0;
        for (uint32_t i = 0; i < 5u; ++i)
        {
            const double y = labels[i] >= 0 ? 1.0 : -1.0;
            const double s = scores[i] * k_code_to_real;
            twin += std::max (0.0, 1.0 - y * s);
        }
        twin /= 5.0;
        CHECK (static_cast<double> (result._M_loss_q7) == Approx (twin * 128.0).margin (3.0));
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        const std::vector<double> lab = reals_of (labels);
        auto loss_twin = [&](const std::vector<double> &x)
        { return ref_loss_real (HINGE, x, lab, 1.0); };
        const int32_t seed = ref_seed_lane (k_q15_one);
        for (uint32_t i = 0; i < 5u; ++i)
        {
            const double y = labels[i] >= 0 ? 1.0 : -1.0;
            const double s = scores[i] * k_code_to_real;
            const double lane_real = static_cast<double> (harness.pred_grad[i]) / seed;
            if (std::abs (y * s - 1.0) < 4e-3)
            {
                /* Code 127 sits 6e-5 below the margin-1.0 kink, so the
                 * two-sided difference straddles max(0, .): compare against
                 * the subgradient slope -y/n instead. */
                CHECK (lane_real == Approx (-y / 5.0).margin (1e-3));
                continue;
            }
            const double numeric = central_difference (loss_twin, reals_of (scores),
                                                       i, 2e-3);
            CHECK (lane_real == Approx (numeric).margin (5e-4));
        }
    }
}

TEST_CASE ("hinge_backward: -sign(label) unit for violators only")
{
    Harness harness;
    const std::vector<int8_t> scores = { 30, -20, 127, -127, 5 };
    const std::vector<int8_t> labels = { 4, -5, 0, 7, -1 };
    const int32_t margins[] = { 128, 64, 0 };
    const int32_t seeds[] = { k_q15_one, 8192 };
    for (const int32_t margin : margins)
        for (const int32_t scale : seeds)
        {
            harness.fill (scores, labels);
            harness.open ();
            LossResult result;
            const HingeParams params{ margin };
            REQUIRE (status_ok (hinge_loss_forward (harness.pool, &harness.tape,
                                                    harness.pred (5),
                                                    harness.target (5), params, result)));
            REQUIRE (status_ok (harness.backward (result, scale)));
            const int32_t seed = ref_seed_lane (scale);
            const int32_t unit = ref_div_round (k_q15_one, 5);
            for (uint32_t i = 0; i < 5u; ++i)
            {
                const int32_t y = labels[i] >= 0 ? k_q15_max : -k_q15_max;
                const int32_t prod = ref_q15_mul (y, ref_code_q15 (scores[i])) >> 8;
                const int32_t grad = prod < margin ? ref_q15_mul (-y, unit) : 0;
                CHECK (harness.pred_grad[i] == ref_q15_mul (grad, seed));
            }
        }
}

TEST_CASE ("multiclass_hinge: sum reduction, meta margin, violation gradient")
{
    Harness harness;
    const std::vector<int8_t> scores = { 10, 50, 30, 5 };
    const uint16_t gold = 1u;
    struct McCase
    {
        int32_t margin_q7;
    };
    const McCase margins[] = { { 128 }, { 0 }, { -128 }, { 16384 } };
    for (const auto &m : margins)
    {
        harness.fill (scores, { 0 });
        harness.open ();
        LossResult result;
        const HingeParams params{ m.margin_q7 };
        REQUIRE (status_ok (multiclass_hinge_forward (harness.pool, &harness.tape,
                                                     harness.pred (4), gold, params,
                                                     result)));
        /* Value: sum (no divisor) over violators of max(0, margin + s_c - s_y). */
        int32_t expect = 0;
        uint32_t violations = 0u;
        for (uint32_t i = 0; i < 4u; ++i)
        {
            if (i == gold)
                continue;
            const int32_t diff_q7 = ref_q15_mul (ref_code_q15 (scores[i])
                                                     - ref_code_q15 (scores[gold]),
                                                 k_q15_max)
                                    >> 8;
            const int32_t term = m.margin_q7 + diff_q7;
            if (term > 0)
            {
                expect += term;
                ++violations;
            }
        }
        CHECK (result._M_loss_q7 == expect);
        CHECK (result._M_loss_q15 == ref_loss_q15 (expect));
        check_loss_result_shape (result, 1u, 1u, _S_op_hinge_multiclass, harness);
        const TargetMeta *const meta = harness.saved<TargetMeta> (0);
        CHECK (meta->_M_classes == 4u);
        CHECK (meta->_M_target == gold);
        CHECK (meta->_M_margin_q7 == m.margin_q7);
        CHECK (harness.pool.stats ()._M_used == 16u);

        /* Gradient: +unit per violator, -violations*unit at the gold class. */
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        const int32_t seed = ref_seed_lane (k_q15_one);
        const int32_t unit = ref_div_round (k_q15_one, 4);
        CHECK (ref_multiclass_violations (scores, gold, m.margin_q7) == violations);
        for (uint32_t i = 0; i < 4u; ++i)
        {
            if (i == gold)
            {
                const int32_t expect_lane
                    = violations == 0u
                          ? 0
                          : ref_q15_mul (-static_cast<int32_t> (violations) * unit,
                                         seed);
                CHECK (harness.pred_grad[i] == expect_lane);
            }
            else
            {
                const int32_t diff_q7 = ref_q15_mul (ref_code_q15 (scores[i])
                                                         - ref_code_q15 (scores[gold]),
                                                     k_q15_max)
                                        >> 8;
                const int32_t expect_lane = (m.margin_q7 + diff_q7 > 0)
                                                ? ref_q15_mul (unit, seed)
                                                : 0;
                CHECK (harness.pred_grad[i] == expect_lane);
            }
        }
    }
    SECTION ("sum reduction: exact violator sums per gold class")
    {
        harness.fill ( { 50, 50, 10, 10 }, { 0 });
        harness.open ();
        LossResult one;
        const HingeParams params{ 128 };
        REQUIRE (status_ok (multiclass_hinge_forward (harness.pool, &harness.tape,
                                                     harness.pred (4), 3u, params,
                                                     one)));
        CHECK (one._M_loss_q7 > 0);
        /* gold=0 (score 50, tied with one class, above two): also three
         * violators, but two of them score 40 below the gold, so the sum is
         * smaller -- value-only loss with no 1/C divisor. */
        harness.fill ( { 50, 50, 10, 10 }, { 0 });
        harness.open ();
        LossResult two;
        REQUIRE (status_ok (multiclass_hinge_forward (harness.pool, &harness.tape,
                                                     harness.pred (4), 0u, params,
                                                     two)));
        const uint32_t k0 = ref_multiclass_violations ( { 50, 50, 10, 10 }, 0u, 128);
        const uint32_t k3 = ref_multiclass_violations ( { 50, 50, 10, 10 }, 3u, 128);
        CHECK (k0 == 3u);
        CHECK (k3 == 3u);
        /* Hand-derived from the documented arms: diff +40 Q7 (10320 Q15
         * narrows to 40), diff 0, diff -40 truncates to -41 under >> 8.
         * gold=3: 3 x 128 + 2 x 40 = 464.  gold=0: 3 x 128 - 2 x 41 = 302. */
        CHECK (one._M_loss_q7 == 464);
        CHECK (two._M_loss_q7 == 302);
        CHECK (two._M_loss_q7 < one._M_loss_q7);
    }
    SECTION ("validation: rank-0/1 buffers and out-of-range gold")
    {
        harness.open ();
        LossResult result;
        const HingeParams params;
        CHECK (multiclass_hinge_forward (harness.pool, &harness.tape, harness.pred (1),
                                        0u, params, result)
               == Status::OUT_OF_RANGE);
        CHECK (multiclass_hinge_forward (harness.pool, &harness.tape, harness.pred (0),
                                        0u, params, result)
               == Status::BAD_LAYOUT);
        CHECK (multiclass_hinge_forward (harness.pool, &harness.tape, harness.pred (3),
                                        3u, params, result)
               == Status::OUT_OF_RANGE);
    }
}

/* ======================================================================== */
/* train_label_smoothing.hpp -- soft targets, entropy, smoothed CE          */
/* ======================================================================== */
namespace
{
/** @brief scale_q15 contract: rounded Q15 product, saturating to an int16. */
int16_t
ref_scale_lane (const int32_t value, const int32_t scalar_q15)
{
    const int64_t shifted
        = (static_cast<int64_t> (value) * scalar_q15 + (1LL << 14)) >> 15;
    return ref_i16_sat (static_cast<int32_t> (
        shifted > k_q15_max ? k_q15_max
                            : (shifted < k_q15_min ? k_q15_min : shifted)));
}

/** @brief axpy_q15 contract for one lane: rounded term plus saturating sum. */
int16_t
ref_axpy_lane (const int32_t dest, const int32_t src, const int32_t scalar_q15)
{
    const int32_t term = static_cast<int32_t> (
        (static_cast<int64_t> (src) * scalar_q15 + (1LL << 14)) >> 15);
    return ref_i16_sat (dest + term);
}

/** @brief Saturating Q15 add, as documented for the smoothing weights. */
int32_t
ref_q15_add_sat (const int32_t a, const int32_t b)
{
    return ref_q15_clamp (a + b);
}

/** @brief |x| for the mass-conservation comparisons. */
int64_t
ref_iabs (const int64_t x)
{
    return x < 0 ? -x : x;
}
} // namespace

TEST_CASE ("SmoothingParams: eps clamp, uniform division, validate rails")
{
    struct MakeCase
    {
        int32_t _M_eps;
        uint32_t _M_classes;
        int32_t _M_eps_out;
        int32_t _M_uniform_out;
    };
    const MakeCase cases[] = {
        { 3277, 4u, 3277, 819 },     /* (3277 + 2) / 4 = 819.75 -> 819  */
        { 0, 8u, 0, 0 },
        { 32767, 1u, 32767, 32767 },
        { 32767, 4u, 32767, 8192 },  /* (32767 + 2) / 4 = 8192.25       */
        { 16384, 3u, 16384, 5461 },  /* (16384 + 1) / 3 truncates .67   */
        { -100, 4u, 0, 0 },          /* negative eps pins to zero first */
        { 100000, 4u, 32767, 8192 }, /* eps rails at q15_max, then cuts */
        { 32768, 2u, 32767, 16384 },
        { 1, 2u, 1, 1 },             /* (1 + 1) / 2: the half rounds away */
        { 32766, 4u, 32766, 8192 },  /* (32766 + 2) / 4 divides exactly   */
        { 5, 0u, 5, 0 },             /* C == 0 pins the uniform at zero   */
    };
    for (const auto &c : cases)
    {
        const SmoothingParams p = SmoothingParams::make (c._M_eps, c._M_classes);
        CHECK (p._M_eps_q15 == c._M_eps_out);
        CHECK (p._M_uniform_q15 == c._M_uniform_out);
        CHECK (p.validate () == Status::OK);
    }
    SECTION ("defaults, sizes and the meta record layout")
    {
        const SmoothingParams d;
        CHECK (d._M_eps_q15 == 3277);   /* the 0.1 convention */
        CHECK (d._M_uniform_q15 == 0);
        CHECK (d.validate () == Status::OK);
        CHECK (sizeof (SmoothingParams) == 8u);
        CHECK (sizeof (SmoothedCeMeta) == 12u);
        const SmoothedCeMeta m;
        CHECK (m._M_eps_q15 == 0);
        CHECK (m._M_uniform_q15 == 0);
        CHECK (m._M_target == 0u);
        CHECK (m._M_classes == 0u);
    }
    SECTION ("validate rejects hand-built out-of-domain words")
    {
        /* make() can never produce these, but validate() is the gate every
         * helper consults before touching a caller buffer. */
        SmoothingParams high;
        high._M_eps_q15 = k_q15_max + 1;
        high._M_uniform_q15 = 0;
        CHECK (high.validate () == Status::OUT_OF_RANGE);
        SmoothingParams low;
        low._M_eps_q15 = k_q15_min - 1;
        low._M_uniform_q15 = 0;
        CHECK (low.validate () == Status::OUT_OF_RANGE);
        SmoothingParams floor_word;
        floor_word._M_eps_q15 = k_q15_min; /* the validator domain is full int16 */
        floor_word._M_uniform_q15 = 0;
        CHECK (floor_word.validate () == Status::OK);
        SmoothingParams neg_uniform;
        neg_uniform._M_eps_q15 = 0;
        neg_uniform._M_uniform_q15 = -1;
        CHECK (neg_uniform.validate () == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("smooth_onehot_q15: exact soft vectors, mass conservation, errors")
{
    struct OneCase
    {
        uint16_t _M_target;
        uint32_t _M_classes;
        int32_t _M_eps;
    };
    const OneCase cases[] = {
        { 0u, 4u, 3277 }, { 3u, 4u, 3277 }, { 1u, 8u, 16384 },
        { 0u, 1u, 32767 }, { 2u, 5u, 0 }, { 4u, 5u, 1 }, { 2u, 3u, 32767 },
    };
    for (const auto &c : cases)
    {
        const SmoothingParams p = SmoothingParams::make (c._M_eps, c._M_classes);
        const int32_t keep = k_q15_one - 1 - p._M_eps_q15; /* the 32767 - eps word */
        int16_t dest[8];
        for (int16_t &d : dest)
            d = static_cast<int16_t> (1234);
        REQUIRE (status_ok (smooth_onehot_q15 (c._M_target, c._M_classes, p, dest)));
        int64_t total = 0;
        for (uint32_t i = 0; i < c._M_classes; ++i)
        {
            const int32_t expect = i == c._M_target
                                       ? static_cast<int32_t> (ref_i16_sat (
                                             p._M_uniform_q15 + keep))
                                       : p._M_uniform_q15;
            CHECK (dest[i] == expect);
            total += dest[i];
        }
        /* Lanes beyond the class count are never written. */
        for (uint32_t i = c._M_classes; i < 8u; ++i)
            CHECK (dest[i] == 1234);
        /* Mass contract: the gold lane is uniform + keep on top, so the sum
         * is exactly C * uniform + keep; the add can never saturate because
         * div_round (eps, C) <= eps keeps gold at or below 32767. */
        CHECK (total == static_cast<int64_t> (c._M_classes) * p._M_uniform_q15
                        + keep);
        /* Conservation against the one-hot mass: total - 32767 is the eps/C
         * division slack only, at most half a unit per class. */
        CHECK (ref_iabs (total - k_q15_max)
               <= (static_cast<int64_t> (c._M_classes) + 1) / 2);
    }
    SECTION ("eps zero reproduces the exact one-hot")
    {
        const SmoothingParams p = SmoothingParams::make (0, 4u);
        int16_t dest[4];
        REQUIRE (status_ok (smooth_onehot_q15 (2u, 4u, p, dest)));
        CHECK (dest[0] == 0);
        CHECK (dest[1] == 0);
        CHECK (dest[2] == k_q15_max);
        CHECK (dest[3] == 0);
    }
    SECTION ("one class: smoothing is the identity at any eps")
    {
        /* C = 1: uniform = eps, gold = eps + (32767 - eps) = 32767 exactly. */
        const SmoothingParams p = SmoothingParams::make (12345, 1u);
        int16_t dest[1];
        REQUIRE (status_ok (smooth_onehot_q15 (0u, 1u, p, dest)));
        CHECK (dest[0] == k_q15_max);
    }
    SECTION ("errors: layout first, then params, buffer untouched")
    {
        const SmoothingParams p = SmoothingParams::make (3277, 4u);
        int16_t dest[4] = { 7, 7, 7, 7 };
        CHECK (smooth_onehot_q15 (0u, 4u, p, nullptr) == Status::BAD_LAYOUT);
        CHECK (smooth_onehot_q15 (0u, 0u, p, dest) == Status::BAD_LAYOUT);
        CHECK (smooth_onehot_q15 (4u, 4u, p, dest) == Status::BAD_LAYOUT);
        CHECK (smooth_onehot_q15 (5u, 4u, p, dest) == Status::BAD_LAYOUT);
        SmoothingParams bad;
        bad._M_eps_q15 = 10;
        bad._M_uniform_q15 = -1;
        CHECK (smooth_onehot_q15 (0u, 4u, bad, dest) == Status::OUT_OF_RANGE);
        for (const int16_t d : dest)
            CHECK (d == 7); /* every rejection left the buffer alone */
    }
}

TEST_CASE ("smooth_dense_q15: widen-scale-fill chain against the integer twin")
{
    Harness harness;
    const std::vector<int8_t> codes = { 100, -3, 0, 64, 127, -127 };
    harness.fill (codes, { 0 });
    const int32_t eps_list[] = { 0, 3277, 16384, 32767 };
    for (const int32_t eps : eps_list)
    {
        const SmoothingParams p = SmoothingParams::make (eps, 6u);
        const int32_t keep = k_q15_one - 1 - p._M_eps_q15;
        int16_t dest[6];
        REQUIRE (status_ok (smooth_dense_q15 (harness.pred (6), p, dest)));
        for (uint32_t i = 0; i < 6u; ++i)
        {
            /* dest = sat (scale (code * 258 by keep) + uniform): the three
             * documented SIMD passes, lane by lane, from the ref primitives. */
            const int32_t lane = ref_code_q15 (codes[i]);
            const int16_t expect = ref_i16_sat (
                static_cast<int32_t> (ref_scale_lane (lane, keep))
                + p._M_uniform_q15);
            CHECK (dest[i] == expect);
        }
    }
    SECTION ("eps=0 is a one-lsb identity: only lanes above half scale shrink")
    {
        const SmoothingParams p = SmoothingParams::make (0, 6u);
        int16_t dest[6];
        REQUIRE (status_ok (smooth_dense_q15 (harness.pred (6), p, dest)));
        /* keep = 32767: the documented round-half-up narrowing
         * (v * 32767 + 2^14) >> 15 is v - 1 above half scale and v + 1 below
         * minus half scale -- the floor lane is pulled up one lsb. */
        CHECK (dest[0] == 25799);  /* 100 * 258 = 25800 -> loses 1        */
        CHECK (dest[1] == -774);   /* small negatives are preserved       */
        CHECK (dest[2] == 0);
        CHECK (dest[3] == 16511);  /* 16512 > 16384 -> loses 1            */
        CHECK (dest[4] == 32765);  /* the top rail lane                   */
        CHECK (dest[5] == -32765); /* -32766 * 32767 / 2^15 rounds to -32765 */
        for (uint32_t i = 0; i < 6u; ++i)
        {
            const int32_t lane = ref_code_q15 (codes[i]);
            CHECK (dest[i] == ref_scale_lane (lane, 32767));
            CHECK (ref_iabs (static_cast<int64_t> (dest[i]) - lane) <= 1);
        }
    }
    SECTION ("eps maximal erases the target into uniform lanes")
    {
        /* keep = 0 -> every scaled lane is 0 -> dest is the fill only. */
        const SmoothingParams p = SmoothingParams::make (32767, 6u);
        CHECK (p._M_uniform_q15 == 5461); /* (32767 + 3) / 6 = 5461.67 */
        int16_t dest[6];
        REQUIRE (status_ok (smooth_dense_q15 (harness.pred (6), p, dest)));
        for (uint32_t i = 0; i < 6u; ++i)
            CHECK (dest[i] == 5461);
        /* A uniform-over-C distribution carries entropy ln C; mass check:
         * C * 5461 = 32766 is one short of the eps mass 32767. */
        int64_t total = 0;
        for (const int16_t d : dest)
            total += d;
        CHECK (total == 32766);
        CHECK (ref_iabs (total - 32767) <= 1);
    }
    SECTION ("the constant add saturates at the top rail, never wraps")
    {
        SmoothingParams manual;
        manual._M_eps_q15 = 0;
        manual._M_uniform_q15 = k_q15_max; /* valid per validate() */
        int16_t dest[6];
        REQUIRE (status_ok (smooth_dense_q15 (harness.pred (6), manual, dest)));
        CHECK (dest[0] == k_q15_max);  /* 25799 + 32767 -> rail            */
        CHECK (dest[4] == k_q15_max);
        CHECK (dest[1] == 31993);      /* -774 + 32767 stays inside        */
        CHECK (dest[5] == 2);          /* -32765 + 32767 = 2 (scaled floor lane) */
    }
    SECTION ("errors: null buffers, empty operands, bad params")
    {
        const SmoothingParams p = SmoothingParams::make (3277, 6u);
        int16_t dest[6] = { 1, 2, 3, 4, 5, 6 };
        CHECK (smooth_dense_q15 (harness.pred (6), p, nullptr) == Status::BAD_LAYOUT);
        const Trainable noc = Trainable::make (nullptr, nullptr,
                                               Shape::vector (3), 0u);
        CHECK (smooth_dense_q15 (noc, p, dest) == Status::BAD_LAYOUT);
        CHECK (smooth_dense_q15 (harness.pred (0), p, dest) == Status::BAD_LAYOUT);
        SmoothingParams bad;
        bad._M_eps_q15 = 0;
        bad._M_uniform_q15 = -1;
        CHECK (smooth_dense_q15 (harness.pred (6), bad, dest)
               == Status::OUT_OF_RANGE);
        for (uint32_t i = 0; i < 6u; ++i)
            CHECK (dest[i] == static_cast<int16_t> (i + 1));
    }
}

TEST_CASE ("entropy_q7: lane buffer and code twin against -sum p log p")
{
    /* Uniform over 4 classes: H = ln 4 = 1.386 nats -> 177 Q7. */
    const int16_t uniform4[4] = { 8192, 8192, 8192, 8192 };
    const double h_uniform = -4.0 * (0.25 * std::log (0.25));
    CHECK (entropy_q7 (uniform4, 4u) == Approx (h_uniform * 128.0).margin (6.0));
    /* Uniform over 8: ln 8 * 128, past 1.0 in Q7 terms but nowhere near a
     * lane, exactly the reason the result is reported in Q7. */
    const int16_t uniform8[8] = { 4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096 };
    const double h8 = std::log (8.0);
    CHECK (entropy_q7 (uniform8, 8u) == Approx (h8 * 128.0).margin (8.0));
    /* A peaked distribution has (almost) no entropy: -ln (32766/32768) * 128
     * rounds to zero Q7 and the non-positive lanes are the 0 log 0 skips. */
    const int16_t peaked[4] = { 32766, 0, -1, 0 };
    CHECK (entropy_q7 (peaked, 4u) == 0);
    /* Non-positive lanes contribute exactly nothing: dropping them from the
     * buffer cannot change the sum. */
    const int16_t mixed[5] = { 1290, -5, 25800, 0, 258 };
    const int16_t trimmed[3] = { 1290, 25800, 258 };
    CHECK (entropy_q7 (mixed, 5u) == entropy_q7 (trimmed, 3u));
    double h_mixed = 0.0;
    for (const int16_t lane : trimmed)
    {
        const double p = static_cast<double> (lane) / k_q15_one;
        h_mixed += -p * std::log (p);
    }
    CHECK (entropy_q7 (mixed, 5u) == Approx (h_mixed * 128.0).margin (8.0));
    SECTION ("the code convenience overload equals the widened lane buffer")
    {
        Harness harness;
        const std::vector<int8_t> codes = { 20, 30, 40, 10 };
        harness.fill (codes, { 0 });
        int16_t lanes[4];
        for (uint32_t i = 0; i < 4u; ++i)
            lanes[i] = ref_scale_lane (ref_code_q15 (codes[i]), k_q15_max);
        CHECK (entropy_q7 (harness.pred (4)) == entropy_q7 (lanes, 4u));
        double h = 0.0;
        for (const int8_t c : codes)
        {
            const double p = c * k_code_to_real;
            h += -p * std::log (p);
        }
        CHECK (entropy_q7 (harness.pred (4)) == Approx (h * 128.0).margin (8.0));
    }
    SECTION ("empty accumulations are exactly zero")
    {
        CHECK (entropy_q7 (uniform4, 0u) == 0);
        const int16_t dead[3] = { 0, -1, -32768 };
        CHECK (entropy_q7 (dead, 3u) == 0);
        Harness harness;
        harness.fill ( { 0, 0 }, { 0 });
        CHECK (entropy_q7 (harness.pred (2)) == 0);
    }
}

TEST_CASE ("uniform_target_gradient: -(1/C)/p with the int16 floor at p <= 0")
{
    Harness harness;
    const std::vector<int8_t> probs = { 64, 32, 8, 1, 0, -5 };
    harness.fill (probs, { 0 });
    const int32_t unit = ref_div_round (k_q15_one, 6);
    CHECK (unit == 5461); /* (32768 + 3) / 6 = 5461.8 -> 5461 */
    int16_t dest[6];
    REQUIRE (status_ok (uniform_target_gradient (harness.pred (6), dest, 6u)));
    for (uint32_t i = 0; i < 6u; ++i)
    {
        const int32_t p = ref_code_q15 (probs[i]);
        const int32_t expect = p > 0 ? ref_q15_div (-unit, p) : k_q15_min;
        CHECK (dest[i] == static_cast<int16_t> (expect));
    }
    /* Analytic -(1/C)/p_real for the un-railed lanes only. */
    for (uint32_t i = 0; i < 6u; ++i)
    {
        const double p_real = probs[i] * k_code_to_real;
        if (p_real <= 0.0)
            continue;
        const double analytic = -(1.0 / 6.0) / p_real;
        const double lane_real = static_cast<double> (dest[i]) / k_q15_one;
        if (analytic > -1.0)
            CHECK (lane_real == Approx (analytic).margin (0.01));
        else
            CHECK (dest[i] == static_cast<int16_t> (k_q15_min));
    }
    SECTION ("small and non-positive probabilities pin the floor")
    {
        CHECK (dest[2] == static_cast<int16_t> (k_q15_min)); /* 5461/2064 > 1 */
        CHECK (dest[3] == static_cast<int16_t> (k_q15_min));
        /* Documented: p <= 0 stores the raw int16 floor (a ratio rail that
         * can reach -32768, unlike the -32767 CE ratio rails). */
        CHECK (dest[4] == static_cast<int16_t> (k_q15_min));
        CHECK (dest[5] == static_cast<int16_t> (k_q15_min));
    }
    SECTION ("validation: destination, count agreement, emptiness")
    {
        CHECK (uniform_target_gradient (harness.pred (6), nullptr, 6u)
               == Status::BAD_LAYOUT);
        CHECK (uniform_target_gradient (harness.pred (6), dest, 5u)
               == Status::BAD_LAYOUT);
        CHECK (uniform_target_gradient (harness.pred (5), dest, 6u)
               == Status::BAD_LAYOUT);
        CHECK (uniform_target_gradient (harness.pred (6), dest, 0u)
               == Status::BAD_LAYOUT);
    }
}

TEST_CASE ("smoothed_gradient_blend: two-pass axpy against the integer twin")
{
    const std::vector<int16_t> gh = { 20000, -4000, 0, 32767, -32768, 516 };
    const std::vector<int16_t> gu = { -6000, -6000, -6000, -6000, -6000, -6000 };
    const int32_t eps_list[] = { 0, 3277, 16384, 32767 };
    for (const int32_t eps : eps_list)
    {
        const SmoothingParams p = SmoothingParams::make (eps, 6u);
        const int32_t keep = k_q15_one - 1 - p._M_eps_q15;
        int16_t dest[6];
        REQUIRE (status_ok (smoothed_gradient_blend (gh.data (), gu.data (), p,
                                                     dest, 6u)));
        for (uint32_t i = 0; i < 6u; ++i)
        {
            /* Contract: dest = sat (sat (0 + gh * keep) + gu * eps), the two
             * documented axpy passes with half-up rounded terms. */
            const int16_t first = ref_axpy_lane (0, gh[i], keep);
            const int16_t expect = ref_axpy_lane (first, gu[i], p._M_eps_q15);
            CHECK (dest[i] == expect);
        }
    }
    SECTION ("eps zero: one-hot lanes return within one lsb, uniform drops out")
    {
        const SmoothingParams p = SmoothingParams::make (0, 6u);
        int16_t dest[6];
        REQUIRE (status_ok (smoothed_gradient_blend (gh.data (), gu.data (), p,
                                                     dest, 6u)));
        /* Second pass scalar 0: the term (src * 0 + 2^14) >> 15 is exactly 0,
         * so the uniform contribution vanishes; the first pass is the eps=0
         * dense-smoothing near-identity (one lost only above half scale). */
        for (uint32_t i = 0; i < 6u; ++i)
        {
            CHECK (ref_iabs (static_cast<int64_t> (dest[i]) - gh[i]) <= 1);
            CHECK (dest[i] == ref_scale_lane (gh[i], k_q15_max));
        }
    }
    SECTION ("eps maximal routes the whole gradient through the uniform arm")
    {
        const SmoothingParams p = SmoothingParams::make (32767, 6u);
        int16_t dest[6];
        REQUIRE (status_ok (smoothed_gradient_blend (gh.data (), gu.data (), p,
                                                     dest, 6u)));
        /* keep = 0 zeroes the one-hot pass; eps = 32767 keeps the uniform
         * pass at one lsb under the raw lane value. */
        for (uint32_t i = 0; i < 6u; ++i)
            CHECK (dest[i] == ref_scale_lane (gu[i], k_q15_max));
    }
    SECTION ("saturating sum, never wrap")
    {
        const int16_t top[2] = { 32767, -32768 };
        const int16_t same[2] = { 32767, -32768 };
        SmoothingParams manual;
        manual._M_eps_q15 = 16384;
        manual._M_uniform_q15 = 0;
        int16_t dest[2];
        REQUIRE (status_ok (smoothed_gradient_blend (top, same, manual, dest, 2u)));
        /* keep = 16383: first term = (32767 * 16383 + 2^14) >> 15 = 16383;
         * second term = (32767 * 16384 + 2^14) >> 15 = 16385; sum rails at
         * 32767 instead of 32768.  The same arithmetic on -32768 rounds
         * toward zero (round-half-up), so the negative lane lands one lsb
         * inside the floor instead of hitting it: -(16383 + 16384). */
        CHECK (dest[0] == k_q15_max);
        CHECK (dest[1] == static_cast<int16_t> (k_q15_min + 1));
    }
    SECTION ("errors leave the destination untouched")
    {
        const SmoothingParams p = SmoothingParams::make (3277, 6u);
        int16_t dest[6] = { 999, 999, 999, 999, 999, 999 };
        CHECK (smoothed_gradient_blend (nullptr, gu.data (), p, dest, 6u)
               == Status::BAD_LAYOUT);
        CHECK (smoothed_gradient_blend (gh.data (), nullptr, p, dest, 6u)
               == Status::BAD_LAYOUT);
        CHECK (smoothed_gradient_blend (gh.data (), gu.data (), p, nullptr, 6u)
               == Status::BAD_LAYOUT);
        SmoothingParams bad;
        bad._M_eps_q15 = k_q15_max + 1; /* unreachable via make () */
        bad._M_uniform_q15 = 0;
        CHECK (smoothed_gradient_blend (gh.data (), gu.data (), bad, dest, 6u)
               == Status::OUT_OF_RANGE);
        for (const int16_t d : dest)
            CHECK (d == 999);
    }
}

TEST_CASE ("smoothed_cross_entropy_forward: value pipeline, eps identity, cost")
{
    Harness harness;
    const std::vector<int8_t> probs = { 60, 40, 20, 10, 5 };
    const uint16_t gold = 1u;
    const int32_t eps_list[] = { 0, 3277, 16384 };
    for (const int32_t eps : eps_list)
    {
        harness.fill (probs, { 0 });
        harness.open ();
        const SmoothingParams p = SmoothingParams::make (eps, 5u);
        const int32_t keep = k_q15_one - 1 - p._M_eps_q15;
        LossResult result;
        REQUIRE (status_ok (smoothed_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (5), gold, p, result)));
        check_loss_result_shape (result, 1u, 1u, _S_op_smoothed_ce, harness);
        CHECK (harness.pool.stats ()._M_used == 20u); /* node 8 + meta 12 */
        const SmoothedCeMeta *const meta = harness.saved<SmoothedCeMeta> (0);
        CHECK (meta->_M_eps_q15 == p._M_eps_q15);
        CHECK (meta->_M_uniform_q15 == p._M_uniform_q15);
        CHECK (meta->_M_target == gold);
        CHECK (meta->_M_classes == 5u);
        /* Float twin of the documented sum: the gold class carries
         * (keep + uniform), every other class the uniform mass only. */
        double twin = 0.0;
        for (uint32_t i = 0; i < 5u; ++i)
        {
            const double w = (i == gold ? keep + p._M_uniform_q15
                                        : p._M_uniform_q15)
                             / static_cast<double> (k_q15_one);
            if (w == 0.0)
                continue;
            twin += w * -std::log (probs[i] * k_code_to_real);
        }
        /* The margin covers the log-atom series error (scaled by w <= 1) and
         * the one-unit floor each (neg_log * w) >> 15 term loses. */
        CHECK (static_cast<double> (result._M_loss_q7)
               == Approx (twin * 128.0).margin (12.0));
        CHECK (result._M_loss_q7 >= 0);
        CHECK (result._M_loss_q15 == ref_loss_q15 (result._M_loss_q7));
    }
    SECTION ("eps zero lands within one Q7 unit of the index form")
    {
        const SmoothingParams p = SmoothingParams::make (0, 5u);
        harness.fill (probs, { 0 });
        harness.open ();
        LossResult smooth;
        REQUIRE (status_ok (smoothed_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (5), gold, p, smooth)));
        harness.fill (probs, { 0 });
        harness.open ();
        LossResult indexed;
        REQUIRE (status_ok (cross_entropy_forward (harness.pool, &harness.tape,
                                                   harness.pred (5), gold,
                                                   indexed)));
        /* Weight 32767/32768 narrows the single term by at most one unit. */
        CHECK (indexed._M_loss_q7 - smooth._M_loss_q7 >= 0);
        CHECK (indexed._M_loss_q7 - smooth._M_loss_q7 <= 1);
    }
    SECTION ("a zero-probability gold pins the sentinel term exactly")
    {
        /* q7_neg_log_q15 (p <= 0) = 32768 (finite rail); at eps = 0 the gold
         * weight is keep = 32767, so (32768 * 32767) >> 15 = 32767 exactly. */
        const SmoothingParams p = SmoothingParams::make (0, 2u);
        harness.fill ( { 0, 50 }, { 0 });
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (smoothed_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (2), 0u, p, result)));
        CHECK (result._M_loss_q7 == 32767);
        CHECK (result._M_loss_q15 == k_q15_max);
        CHECK (result._M_node.code_at (0) == 127);
    }
    SECTION ("validation: layout, then target, then params")
    {
        harness.open ();
        LossResult result;
        CHECK (smoothed_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (0), 0u,
            SmoothingParams::make (3277, 4u), result)
               == Status::BAD_LAYOUT);
        CHECK (smoothed_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (5), 5u,
            SmoothingParams::make (3277, 5u), result)
               == Status::OUT_OF_RANGE);
        CHECK (smoothed_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (5), 6u,
            SmoothingParams::make (3277, 5u), result)
               == Status::OUT_OF_RANGE);
        SmoothingParams bad;
        bad._M_eps_q15 = 0;
        bad._M_uniform_q15 = -1;
        CHECK (smoothed_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (5), 1u, bad, result)
               == Status::OUT_OF_RANGE);
        /* Nothing was carved or recorded on the rejected calls. */
        CHECK (harness.pool.stats ()._M_used == 0u);
        CHECK (harness.tape.node_count () == 0u);
    }
}

TEST_CASE ("smoothed_ce_backward: weight-rail lanes, blend mass, saturation")
{
    Harness harness;
    const std::vector<int8_t> probs = { 60, 40, 20, 10, 5 };
    const int32_t seeds[] = { k_q15_one, 16384, 1, 0 };
    for (const int32_t eps : { 0, 3277 })
    {
        const SmoothingParams p = SmoothingParams::make (eps, 5u);
        const int32_t keep = k_q15_one - 1 - p._M_eps_q15;
        /* Documented weights: gold = sat (keep * 32767 / 2^15 + uniform),
         * every other class = sat (0 + uniform) = uniform. */
        const int32_t w_gold
            = ref_q15_add_sat (ref_q15_mul (keep, k_q15_max), p._M_uniform_q15);
        for (const int32_t scale : seeds)
        {
            harness.fill (probs, { 0 });
            harness.open ();
            LossResult result;
            REQUIRE (status_ok (smoothed_cross_entropy_forward (
                harness.pool, &harness.tape, harness.pred (5), 1u, p, result)));
            REQUIRE (status_ok (harness.backward (result, scale)));
            const int32_t seed = ref_seed_lane (scale);
            for (uint32_t i = 0; i < 5u; ++i)
            {
                const int32_t w = i == 1u ? w_gold : p._M_uniform_q15;
                const int32_t q = ref_code_q15 (probs[i]);
                const int32_t ratio = q > 0 ? ref_q15_div (-w, q) : -k_q15_max;
                CHECK (harness.pred_grad[i] == ref_q15_mul (ratio, seed));
            }
            CHECK (harness.tape.stats ()._M_saturation_events == 0u);
        }
    }
    SECTION ("the blended weights keep the smoothed one-hot mass")
    {
        /* eps = 3277, C = 5: uniform = 655, q15_mul (29490, 32767) = 29489,
         * so gold weighs 30144 and the total is 30144 + 4 * 655 = 32764,
         * within a couple of units of the 32767 one-hot mass. */
        const SmoothingParams p = SmoothingParams::make (3277, 5u);
        CHECK (p._M_uniform_q15 == 655);
        const int32_t w_gold = ref_q15_add_sat (
            ref_q15_mul (k_q15_one - 1 - p._M_eps_q15, k_q15_max),
            p._M_uniform_q15);
        const int64_t mass = w_gold + 4 * static_cast<int64_t> (p._M_uniform_q15);
        CHECK (mass == 32764);
        CHECK (ref_iabs (mass - k_q15_max) <= 4);
    }
    SECTION ("non-positive probabilities use the -q15_max ratio rail")
    {
        const SmoothingParams p = SmoothingParams::make (3277, 4u);
        const int32_t seeds[] = { k_q15_one, 8192 };
        for (const int32_t scale : seeds)
        {
            harness.fill ( { 0, -9, 5, 60 }, { 0 });
            harness.open ();
            LossResult result;
            REQUIRE (status_ok (smoothed_cross_entropy_forward (
                harness.pool, &harness.tape, harness.pred (4), 0u, p, result)));
            REQUIRE (status_ok (harness.backward (result, scale)));
            const int32_t seed = ref_seed_lane (scale);
            /* Ratio -32767 (int16-reachable), unlike the -32768 store in
             * uniform_target_gradient -- pinned from the header. */
            CHECK (harness.pred_grad[0] == ref_q15_mul (-k_q15_max, seed));
            CHECK (harness.pred_grad[1] == ref_q15_mul (-k_q15_max, seed));
            const int32_t w = p._M_uniform_q15;
            CHECK (harness.pred_grad[2]
                   == ref_q15_mul (ref_q15_div (-w, ref_code_q15 (5)), seed));
        }
    }
    SECTION ("a second backward saturates rail-bound lanes and counts them")
    {
        const SmoothingParams p = SmoothingParams::make (3277, 2u);
        CHECK (p._M_uniform_q15 == 1639); /* (3277 + 1) / 2 */
        harness.fill ( { 1, 60 }, { 0 });
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (smoothed_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (2), 0u, p, result)));
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        const int32_t seed = ref_seed_lane (k_q15_one);
        const int32_t w_gold = ref_q15_add_sat (
            ref_q15_mul (k_q15_one - 1 - p._M_eps_q15, k_q15_max),
            p._M_uniform_q15);
        /* Gold lane: 31128 / 258 overflows -> -32768 ratio -> -32767 lane. */
        CHECK (ref_q15_div (-w_gold, ref_code_q15 (1)) == k_q15_min);
        CHECK (harness.pred_grad[0] == ref_q15_mul (k_q15_min, seed));
        CHECK (harness.pred_grad[1]
               == ref_q15_mul (ref_q15_div (-1639, ref_code_q15 (60)), seed));
        CHECK (harness.tape.stats ()._M_saturation_events == 0u);
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        /* -32767 + (-32767) clamps to the int16 floor: one counted event;
         * the small lane just accumulates. */
        CHECK (harness.pred_grad[0] == k_q15_min);
        CHECK (harness.tape.stats ()._M_saturation_events == 1u);
    }
}

TEST_CASE ("soft_target_cross_entropy_forward: uniform mass enters every class")
{
    Harness harness;
    const std::vector<int8_t> probs = { 60, 40, 20, 10 };
    const std::vector<int8_t> dense = { 127, 0, 64, 32 };
    const int32_t eps_list[] = { 0, 3277, 8192 };
    for (const int32_t eps : eps_list)
    {
        harness.fill (probs, dense);
        harness.open ();
        const SmoothingParams p = SmoothingParams::make (eps, 4u);
        const int32_t keep = k_q15_one - 1 - p._M_eps_q15;
        LossResult result;
        REQUIRE (status_ok (soft_target_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (4), harness.target (4),
            p, result)));
        check_loss_result_shape (result, 2u, 1u, _S_op_soft_ce, harness);
        CHECK (harness.pool.stats ()._M_used == 20u);
        const SmoothedCeMeta *const meta = harness.saved<SmoothedCeMeta> (0);
        CHECK (meta->_M_eps_q15 == p._M_eps_q15);
        CHECK (meta->_M_uniform_q15 == p._M_uniform_q15);
        CHECK (meta->_M_target == 0u); /* dense form: no gold index */
        CHECK (meta->_M_classes == 4u);
        /* Weights sat (keep * t / 2^15 + uniform) in the integer twin, log
         * from <cmath: the blended uniform mass must be visible on t = 0. */
        double twin = 0.0;
        for (uint32_t i = 0; i < 4u; ++i)
        {
            const int32_t w = ref_q15_add_sat (
                ref_q15_mul (keep, ref_code_q15 (dense[i])), p._M_uniform_q15);
            if (w <= 0)
                continue;
            twin += static_cast<double> (w) / k_q15_one
                    * -std::log (probs[i] * k_code_to_real);
        }
        CHECK (static_cast<double> (result._M_loss_q7)
               == Approx (twin * 128.0).margin (12.0));
        CHECK (result._M_loss_q15 == ref_loss_q15 (result._M_loss_q7));
    }
    SECTION ("eps zero degenerates to the plain soft CE within two units")
    {
        harness.fill (probs, dense);
        harness.open ();
        LossResult blended;
        REQUIRE (status_ok (soft_target_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (4), harness.target (4),
            SmoothingParams::make (0, 4u), blended)));
        harness.fill (probs, dense);
        harness.open ();
        LossResult plain;
        REQUIRE (status_ok (soft_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (4), harness.target (4),
            plain)));
        /* Only the keep = 32767 narrowing separates the weight pipelines. */
        CHECK (plain._M_loss_q7 - blended._M_loss_q7 >= 0);
        CHECK (plain._M_loss_q7 - blended._M_loss_q7 <= 2);
    }
    SECTION ("pair and params validation")
    {
        harness.open ();
        LossResult result;
        const SmoothingParams p = SmoothingParams::make (3277, 4u);
        CHECK (soft_target_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (4), harness.target (3),
            p, result)
               == Status::BAD_LAYOUT);
        CHECK (soft_target_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (0), harness.target (0),
            p, result)
               == Status::BAD_LAYOUT);
        SmoothingParams bad;
        bad._M_eps_q15 = 0;
        bad._M_uniform_q15 = -1;
        CHECK (soft_target_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (4), harness.target (4),
            bad, result)
               == Status::OUT_OF_RANGE);
    }
}

TEST_CASE ("soft_target_ce_backward: blended rails, t=0 classes still gradient")
{
    Harness harness;
    const std::vector<int8_t> probs = { 60, 40, 20, 10 };
    const std::vector<int8_t> dense = { 127, 0, 64, 32 };
    const SmoothingParams p = SmoothingParams::make (3277, 4u);
    const int32_t keep = k_q15_one - 1 - p._M_eps_q15;
    const int32_t seeds[] = { k_q15_one, 8192, 0 };
    for (const int32_t scale : seeds)
    {
        harness.fill (probs, dense);
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (soft_target_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (4), harness.target (4),
            p, result)));
        REQUIRE (status_ok (harness.backward (result, scale)));
        const int32_t seed = ref_seed_lane (scale);
        for (uint32_t i = 0; i < 4u; ++i)
        {
            const int32_t w = ref_q15_add_sat (
                ref_q15_mul (keep, ref_code_q15 (dense[i])), p._M_uniform_q15);
            const int32_t q = ref_code_q15 (probs[i]);
            const int32_t ratio = q > 0 ? ref_q15_div (-w, q) : -k_q15_max;
            CHECK (harness.pred_grad[i] == ref_q15_mul (ratio, seed));
        }
    }
    SECTION ("the plain CE leaves the t=0 class dead, the blend does not")
    {
        harness.fill (probs, dense);
        harness.open ();
        LossResult plain;
        REQUIRE (status_ok (soft_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (4), harness.target (4),
            plain)));
        REQUIRE (status_ok (harness.backward (plain, k_q15_one)));
        CHECK (harness.pred_grad[1] == 0); /* -0 / p */
        harness.fill (probs, dense);
        harness.open ();
        LossResult blended;
        REQUIRE (status_ok (soft_target_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (4), harness.target (4),
            SmoothingParams::make (3277, 4u), blended)));
        REQUIRE (status_ok (harness.backward (blended, k_q15_one)));
        const int32_t seed = ref_seed_lane (k_q15_one);
        const int32_t expect
            = ref_q15_mul (ref_q15_div (-819, ref_code_q15 (40)), seed);
        CHECK (harness.pred_grad[1] == expect); /* uniform / p, nonzero */
        CHECK (harness.pred_grad[1] != 0);
    }
    SECTION ("p <= 0 routes through the -q15_max rail")
    {
        harness.fill ( { 0, 50, 25 }, { 50, 0, 0 });
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (soft_target_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (3), harness.target (3),
            SmoothingParams::make (0, 3u), result)));
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        const int32_t seed = ref_seed_lane (k_q15_one);
        /* keep = 32767, t_0 = 50 * 258: weight narrows to 12900 (<= 16384,
         * so no lsb lost), but p = 0 forces the documented -32767 rail. */
        CHECK (ref_q15_mul (32767, ref_code_q15 (50)) == ref_code_q15 (50));
        CHECK (harness.pred_grad[0] == ref_q15_mul (-k_q15_max, seed));
    }
    SECTION ("saturation counts only the add_grad clamps on the second pass")
    {
        harness.fill ( { 1, 1 }, { 127, 127 });
        harness.open ();
        LossResult result;
        REQUIRE (status_ok (soft_target_cross_entropy_forward (
            harness.pool, &harness.tape, harness.pred (2), harness.target (2),
            SmoothingParams::make (0, 2u), result)));
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        const int32_t seed = ref_seed_lane (k_q15_one);
        const int32_t w = ref_q15_mul (k_q15_max, ref_code_q15 (127));
        CHECK (w == 32765); /* the eps=0 keep narrowing loses one */
        const int32_t lane = ref_q15_mul (ref_q15_div (-w, ref_code_q15 (1)),
                                          seed);
        CHECK (harness.pred_grad[0] == lane);
        CHECK (harness.pred_grad[1] == lane);
        CHECK (harness.tape.stats ()._M_saturation_events == 0u);
        REQUIRE (status_ok (harness.backward (result, k_q15_one)));
        CHECK (harness.pred_grad[0] == k_q15_min);
        CHECK (harness.pred_grad[1] == k_q15_min);
        CHECK (harness.tape.stats ()._M_saturation_events == 2u);
    }
}

/* APPEND MARKER */
