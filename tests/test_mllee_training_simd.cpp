/* Tests for the training SIMD primitives in mllee/train_simd.hpp: all ten
 * kernels over lane sweeps that include both Q15 rails, zeros and mixed signs,
 * at counts that run the eight-lane SSE2 body, its tail and a lane-shifted
 * (non-SIMD-ready) window of the same data.
 *
 * This build links mllee/asm-x86_64/mllee_train_q15_sse2.S and compiles the
 * portable fallback out, so the exact-integer transcriptions below -- not the
 * fallback -- are what pins the assembly.  The header's contract is that both
 * bodies produce byte-identical results, so every reference here mirrors the
 * documented recurrence (int64 products, (x + 2^14) >> 15 with floor
 * semantics, the 258-unit code grid with round-half-away, saturation instead
 * of wrapping) rather than a floating model. */

#include "mllee/train_pool.hpp"
#include "mllee/train_simd.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

using namespace zzs::mllee;
using namespace zzs::mllee::train;

namespace
{
constexpr uint32_t _stream = 33u;   /* longest sweep: 4 groups plus one tail lane */
constexpr uint32_t _guard = 8u;     /* lanes of sentinel on each side             */
constexpr int32_t _one = 32768;     /* _S_q15_one                                 */
constexpr int32_t _hi = 32767;      /* _S_q15_max                                 */
constexpr int32_t _lo = -32768;     /* _S_q15_min                                 */
constexpr int32_t _step = 258;      /* _S_code_step_q15, 32768 / 127 rounded up   */
constexpr int16_t _sentinel = static_cast<int16_t> (0x5A5A);
constexpr int8_t _sentinel8 = static_cast<int8_t> (0x5A);

const uint32_t _counts[]
    = { 0u, 1u, 2u, 3u, 7u, 8u, 9u, 15u, 16u, 17u, 33u };

/* Q15 scalars / learning rates: zero, one ulp, the half-code and code
   quantums, the identity, both rails and their negatives. */
const int32_t _scalars[] = { 0, 1, 2, 128, 129, 257, 258, 1032, 1638, 3277,
                             8192, 16383, 16384, 32767, 32768,
                             -1, -129, -16384, -16385, -32768 };

const int32_t _lane_a[_stream] = {
    _hi, _lo, 0, 1, -1, 129, -129, 258,
    -258, 16384, -16384, 8192, -8192, _hi - 1, -_hi, 3,
    1000, -1000, 4096, -4096, 255, -255, 64, -64,
    _hi, _hi, _lo, _lo, 129, 130, -128, 2, -2
};

const int32_t _lane_b[_stream] = {
    1, -1, _hi, _hi, _hi, _lo, _lo, 0,
    2, -2, 129, -129, 258, -258, 16384, 16384,
    -16384, 258, 1, -1, 127, -127, 254, -254,
    0, 1, 2, -1, 3, -3, 1, 1, 0
};

/* Forward int8 codes for the ReLU gate, including the -128 code the module
   never emits itself: the gate is a sign test, so it must read as negative. */
const int32_t _codes_in[_stream] = {
    0, 1, -1, 127, -127, -1, 40, -40,
    2, -2, 0, 127, -127, 3, -3, 64,
    -64, 1, -1, 0, 5, -5, 100, -100,
    7, -7, 0, 1, -1, 127, -128, 2, -2
};

/** @brief (x + 2^14) >> 15 on an int64 accumulator: the kernel's rounding,
 *         i.e. ties toward +infinity (floor of x / 2^15 + 0.5). */
int64_t
round15 (const int64_t x) noexcept
{
    return (x + 16384) >> 15;
}

/** @brief Saturation into the Q15 lane rails, never a wrap. */
int64_t
clamp16 (const int64_t x) noexcept
{
    return x > _hi ? _hi : (x < _lo ? _lo : x);
}

/** @brief Saturation of a reduction into int32 (the dot / norm rail). */
int64_t
clamp32 (const int64_t x) noexcept
{
    return x > 2147483647LL ? 2147483647LL : (x < -2147483648LL ? -2147483648LL : x);
}

/** @brief The code grid move of sgd_step_i8: round-half-away division by 258. */
int64_t
grid_move (const int64_t code_step) noexcept
{
    return code_step >= 0 ? (code_step + 129) / _step
                          : -((-code_step + 129) / _step);
}

/** @brief _S_i8_code: the [-127, 127] code rail, so -128 is never emitted. */
int64_t
i8_rail (const int64_t x) noexcept
{
    return x > 127 ? 127 : (x < -127 ? -127 : x);
}

/** @brief q15_to_codes: (|v| * 127 + 2^14) >> 15 with the sign restored. */
int64_t
narrow_code (const int64_t v) noexcept
{
    const int64_t magnitude = v < 0 ? -v : v;
    const int64_t code = (magnitude * 127 + 16384) >> 15;
    return i8_rail (v < 0 ? -code : code);
}

/** @brief First lane whose int16 value differs from the reference, else -1. */
int
first_diff (const int16_t *const got, const int64_t *const want,
            const uint32_t n) noexcept
{
    for (uint32_t i = 0; i < n; ++i)
        if (got[i] != want[i])
            return static_cast<int> (i);
    return -1;
}

/** @brief First lane whose int8 code differs from the reference, else -1. */
int
first_diff8 (const int8_t *const got, const int64_t *const want,
             const uint32_t n) noexcept
{
    for (uint32_t i = 0; i < n; ++i)
        if (got[i] != static_cast<int8_t> (want[i]))
            return static_cast<int> (i);
    return -1;
}

/** @brief A lane stream with a sentinel moat, so an over-read or an
 *         over-write by the eight-lane body is visible. */
struct Lanes
{
    alignas (8) int16_t _raw[_guard + _stream + _guard];
    uint32_t _n = 0u;

    int16_t *
    lanes () noexcept
    {
        return _raw + _guard;
    }

    const int16_t *
    lanes () const noexcept
    {
        return _raw + _guard;
    }

    void
    prime (const int32_t *const pattern, const uint32_t n) noexcept
    {
        _n = n;
        for (uint32_t i = 0; i < _guard + _stream + _guard; ++i)
            _raw[i] = _sentinel;
        for (uint32_t i = 0; i < n; ++i)
            lanes ()[i] = static_cast<int16_t> (pattern[i]);
    }

    /** @brief True when nothing outside the primed window was written. */
    bool
    untouched () const noexcept
    {
        for (uint32_t i = 0; i < _guard; ++i)
            if (_raw[i] != _sentinel || _raw[_guard + _stream + i] != _sentinel)
                return false;
        for (uint32_t i = _n; i < _stream; ++i)
            if (lanes ()[i] != _sentinel)
                return false;
        return true;
    }
};

/** @brief The same moat around an int8 code stream (one byte per lane). */
struct Codes
{
    alignas (8) int8_t _raw[_guard + _stream + _guard];
    uint32_t _n = 0u;

    int8_t *
    codes () noexcept
    {
        return _raw + _guard;
    }

    const int8_t *
    codes () const noexcept
    {
        return _raw + _guard;
    }

    void
    prime (const int32_t *const pattern, const uint32_t n) noexcept
    {
        _n = n;
        for (uint32_t i = 0; i < _guard + _stream + _guard; ++i)
            _raw[i] = _sentinel8;
        for (uint32_t i = 0; i < n; ++i)
            codes ()[i] = static_cast<int8_t> (pattern[i]);
    }

    bool
    untouched () const noexcept
    {
        for (uint32_t i = 0; i < _guard; ++i)
            if (_raw[i] != _sentinel8
                || _raw[_guard + _stream + i] != _sentinel8)
                return false;
        for (uint32_t i = _n; i < _stream; ++i)
            if (codes ()[i] != _sentinel8)
                return false;
        return true;
    }
};

/** @brief Fill the lanes of a plain (unguarded) buffer from a pattern. */
void
fill (int16_t *const dst, const int32_t *const pattern, const uint32_t n) noexcept
{
    for (uint32_t i = 0; i < n; ++i)
        dst[i] = static_cast<int16_t> (pattern[i]);
}
} // namespace

TEST_CASE ("train_simd code step and identity constants are the documented grid",
           "[mllee][train][simd]")
{
    /* 32768 / 127 = 257.54, rounded to 258: code 127 is 32766, one ulp under
       1.0, and 127 * 258 is the widest a code-lane product can reach. */
    CHECK (train_simd::_S_code_step_q15 == 258);
    CHECK (_step * 127 == 32766);
    CHECK (train_simd::_S_code_step_q15 == _S_code_to_q15_scale);
    CHECK (_one / 127 == 258);
}

TEST_CASE ("dot_q15 is the rounded int64 product sum, saturated to int32",
           "[mllee][train][simd]")
{
    for (const uint32_t n : _counts)
    {
        Lanes lhs;
        Lanes rhs;
        lhs.prime (_lane_a, n);
        rhs.prime (_lane_b, n);

        int64_t total = 0;
        for (uint32_t i = 0; i < n; ++i)
            total += static_cast<int64_t> (_lane_a[i]) * _lane_b[i];
        const int64_t want = clamp32 (round15 (total));
        const int32_t got = train_simd::dot_q15 (lhs.lanes (), rhs.lanes (), n);
        INFO ("count " << n << " dot " << got << " want " << want);
        CHECK (static_cast<int64_t> (got) == want);
        if (n == 0u)
            CHECK (got == 0);
        CHECK (lhs.untouched ());
        CHECK (rhs.untouched ());

        /* the same data read one lane further in (a 2 mod 8, i.e. not
           SIMD-ready pointer) must reduce identically */
        if (n >= 2u)
        {
            int64_t shift_total = 0;
            for (uint32_t i = 1; i < n; ++i)
                shift_total += static_cast<int64_t> (_lane_a[i]) * _lane_b[i];
            const int32_t shifted
                = train_simd::dot_q15 (lhs.lanes () + 1, rhs.lanes () + 1, n - 1u);
            CHECK (static_cast<int64_t> (shifted) == clamp32 (round15 (shift_total)));
        }
    }

    SECTION ("hand-derived rails and the floor rounding of a negative tie")
    {
        alignas (8) int16_t lhs[4];
        alignas (8) int16_t rhs[4];
        fill (lhs, _lane_a, 4u);
        fill (rhs, _lane_b, 4u);
        /* 32767 * 1 + (-32768) * (-1) + 0 * 32767 + 1 * 32767 = 32767 + 32768
           + 0 + 32767 = 98302, (98302 + 16384) >> 15 = 114686 >> 15 = 3 */
        CHECK (train_simd::dot_q15 (lhs, rhs, 4u) == 3);

        const int32_t neg_lhs[2] = { -32768, 1 };
        const int32_t neg_rhs[2] = { 1, -1 };
        Lanes a;
        Lanes b;
        a.prime (neg_lhs, 2u);
        b.prime (neg_rhs, 2u);
        /* total = -32768 - 1 = -32769, +16384 = -16385, and -16385 >> 15 is
           -1: the shift floors, so a half-and-a-bit below zero rounds down */
        CHECK (train_simd::dot_q15 (a.lanes (), b.lanes (), 2u) == -1);
        /* -16384 + 16384 = 0 exactly: the symmetric half-lsb lands on zero */
        a.prime (neg_lhs, 1u);
        b.prime (neg_rhs, 1u);
        CHECK (train_simd::dot_q15 (a.lanes (), b.lanes (), 1u) == -1);
        const int32_t tie_lhs[1] = { -1 };
        const int32_t tie_rhs[1] = { 16384 };
        a.prime (tie_lhs, 1u);
        b.prime (tie_rhs, 1u);
        /* total = -16384: (-16384 + 16384) >> 15 = 0, ties toward +infinity */
        CHECK (train_simd::dot_q15 (a.lanes (), b.lanes (), 1u) == 0);
    }

    SECTION ("both int32 rails are reachable and a zero count reads nothing")
    {
        /* One -32768 x -32768 product is 2^30, so after the >> 15 each lane
           contributes exactly 32768 and the total only passes INT32_MAX at
           65536 lanes: 65535 * 32768 = 2147450880, 65536 * 32768 = 2147483648.
           The widest negative product is -32768 x 32767 = -32767 * 2^15, i.e.
           -32767 per lane, so the floor needs 65539 lanes:
           65538 * 32767 = 2147483646 (two ulps of headroom) and
           65539 * 32767 = 2147516413 (past -2147483648). */
        static int16_t big_lhs[65540];
        static int16_t big_rhs[65540];
        for (uint32_t i = 0u; i < 65540u; ++i)
            {
                big_lhs[i] = static_cast<int16_t> (_lo);
                big_rhs[i] = static_cast<int16_t> (_lo);
            }
        CHECK (train_simd::dot_q15 (big_lhs, big_rhs, 65535u) == 2147450880);
        CHECK (train_simd::dot_q15 (big_lhs, big_rhs, 65536u) == 2147483647);

        for (uint32_t i = 0u; i < 65540u; ++i)
            big_rhs[i] = static_cast<int16_t> (_hi);
        CHECK (train_simd::dot_q15 (big_lhs, big_rhs, 65538u) == -2147483646);
        CHECK (train_simd::dot_q15 (big_lhs, big_rhs, 65539u) == -2147483648);

        Lanes lhs;
        Lanes rhs;
        lhs.prime (_lane_a, 0u);
        rhs.prime (_lane_b, 0u);
        CHECK (train_simd::dot_q15 (lhs.lanes (), rhs.lanes (), 0u) == 0);
        CHECK (lhs.untouched ());
        CHECK (rhs.untouched ());
    }
}

TEST_CASE ("add_grad saturates lane-wise and reports every clamped lane",
           "[mllee][train][simd]")
{
    for (const uint32_t n : _counts)
    {
        Lanes dst;
        Lanes src;
        dst.prime (_lane_a, n);
        src.prime (_lane_b, n);
        int64_t want[_stream];
        uint32_t want_sat = 0u;
        for (uint32_t i = 0; i < n; ++i)
            {
                const int64_t sum = static_cast<int64_t> (_lane_a[i]) + _lane_b[i];
                want[i] = clamp16 (sum);
                if (want[i] != sum)
                    ++want_sat;
            }
        const uint32_t sat = train_simd::add_grad (dst.lanes (), src.lanes (), n);
        const int bad = first_diff (dst.lanes (), want, n);
        INFO ("count " << n << " lane " << bad);
        CHECK (bad == -1);
        CHECK (sat == want_sat);
        CHECK (src.untouched ());
        CHECK (dst.untouched ());
        /* the source stream is read-only and the destination keeps its own
           lanes beyond count */
        for (uint32_t i = 0; i < n; ++i)
            CHECK (src.lanes ()[i] == static_cast<int16_t> (_lane_b[i]));
    }

    SECTION ("exact rail hits are not clamp events, near misses are")
    {
        const int32_t a[8] = { _hi, _lo, -_hi, 100, -100, 0, _hi, _lo };
        const int32_t b[8] = { 1, -1, -1, 32667, -32568, 0, 100, -100 };
        Lanes dst;
        Lanes src;
        dst.prime (a, 8u);
        src.prime (b, 8u);
        const uint32_t sat = train_simd::add_grad (dst.lanes (), src.lanes (), 8u);
        /* 32767 + 1 = 32768 clamps and -32768 - 1 = -32769 clamps, while
           -32767 - 1 = -32768 and 100 + 32667 = 32767 land exactly on a rail:
           stored but NOT counted.  Lanes 6 (32767 + 100) and 7 (-32768 - 100)
           clamp too, so the group has four clamp events -- two in each 64-bit
           half, which psadbw turns into 256 per half and the final shr 7 back
           into 4. */
        CHECK (dst.lanes ()[0] == _hi);
        CHECK (dst.lanes ()[1] == _lo);
        CHECK (dst.lanes ()[2] == _lo);
        CHECK (dst.lanes ()[3] == 32767);
        CHECK (dst.lanes ()[4] == -32668);
        CHECK (dst.lanes ()[5] == 0);
        CHECK (dst.lanes ()[6] == _hi);
        CHECK (dst.lanes ()[7] == _lo);
        CHECK (sat == 4u);
        /* a zero-count call over the same buffers is a no-op */
        const int16_t after[8] = { _hi, _lo, _lo, 32767, -32668, 0, _hi, _lo };
        CHECK (train_simd::add_grad (dst.lanes (), src.lanes (), 0u) == 0u);
        for (uint32_t i = 0; i < 8u; ++i)
            CHECK (dst.lanes ()[i] == after[i]);
    }

    SECTION ("only bit 15 of the overflow witness feeds the count")
    {
        /* 16383 + 16383 = 32766 never clamps, but the raw witness
           (a ^ sum) & (b ^ sum) is 0x4001, i.e. 65 psadbw units of carry
           noise per lane.  Left unmasked, eight of them would report a phantom
           count of 4; the pand against the sign constant makes it 0. */
        const int32_t a[8] = { 16383, 16383, 16383, 16383, 16383, 16383, 16383, 16383 };
        Lanes dst;
        Lanes src;
        dst.prime (a, 8u);
        src.prime (a, 8u);
        const uint32_t sat = train_simd::add_grad (dst.lanes (), src.lanes (), 8u);
        CHECK (sat == 0u);
        for (uint32_t i = 0; i < 8u; ++i)
            CHECK (dst.lanes ()[i] == 32766);
    }

    SECTION ("a lane-shifted destination accumulates the same per-lane sums")
    {
        Lanes dst;
        Lanes src;
        dst.prime (_lane_a, _stream);
        src.prime (_lane_b, _stream);

        int64_t want[_stream];
        uint32_t want_front = 0u;
        for (uint32_t i = 0; i < _stream; ++i)
            {
                const int64_t sum = static_cast<int64_t> (_lane_a[i]) + _lane_b[i];
                want[i] = clamp16 (sum);
                if (want[i] != sum)
                    ++want_front;
            }
        const uint32_t front
            = train_simd::add_grad (dst.lanes (), src.lanes (), _stream);
        CHECK (front == want_front);
        CHECK (first_diff (dst.lanes (), want, _stream) == -1);

        /* the same window one lane deeper: 32 lanes whose base pointer is 2
           mod 8, so the movdqu body runs off the lane grid and lane 0 drops out
           of both the sums and the clamp count */
        dst.prime (_lane_a, _stream);
        src.prime (_lane_b, _stream);
        int64_t shifted[_stream];
        uint32_t want_back = 0u;
        for (uint32_t i = 1u; i < _stream; ++i)
            {
                const int64_t sum = static_cast<int64_t> (_lane_a[i]) + _lane_b[i];
                shifted[i - 1u] = clamp16 (sum);
                if (shifted[i - 1u] != sum)
                    ++want_back;
            }
        const uint32_t back
            = train_simd::add_grad (dst.lanes () + 1, src.lanes () + 1, _stream - 1u);
        CHECK (back == want_back);
        CHECK (first_diff (dst.lanes () + 1, shifted, _stream - 1u) == -1);
        /* the dropped lane is 32767 + 1, which clamps, so the two runs differ
           by exactly that one event and the equality of the rest is not
           vacuous */
        CHECK (front == back + 1u);
        CHECK (dst.lanes ()[0] == static_cast<int16_t> (_lane_a[0]));
    }

    SECTION ("a zero count writes nothing and reports no clamp")
    {
        CHECK (train_simd::add_grad (nullptr, nullptr, 0u) == 0u);
        Lanes dst;
        Lanes src;
        dst.prime (_lane_a, 0u);
        src.prime (_lane_b, 0u);
        CHECK (train_simd::add_grad (dst.lanes (), src.lanes (), 0u) == 0u);
        CHECK (dst.untouched ());
        CHECK (src.untouched ());
    }
}

TEST_CASE ("axpy_q15 adds the rounded scalar product with lane saturation",
           "[mllee][train][simd]")
{
    for (const uint32_t n : _counts)
        for (const int32_t k : _scalars)
        {
            Lanes dst;
            Lanes src;
            dst.prime (_lane_a, n);
            src.prime (_lane_b, n);
            int64_t want[_stream];
            for (uint32_t i = 0; i < n; ++i)
                want[i] = clamp16 (static_cast<int64_t> (_lane_a[i])
                                   + round15 (static_cast<int64_t> (_lane_b[i]) * k));
            train_simd::axpy_q15 (dst.lanes (), src.lanes (), k, n);
            const int bad = first_diff (dst.lanes (), want, n);
            INFO ("count " << n << " scalar " << k << " lane " << bad);
            CHECK (bad == -1);
            CHECK (dst.untouched ());
            CHECK (src.untouched ());
        }

    SECTION ("zero scalar is the identity, one ulp is one lane of drift")
    {
        alignas (8) int16_t dst[9];
        alignas (8) int16_t src[9];
        fill (dst, _lane_a, 9u);
        fill (src, _lane_b, 9u);
        int16_t before[9];
        for (uint32_t i = 0; i < 9u; ++i)
            before[i] = dst[i];
        train_simd::axpy_q15 (dst, src, 0, 9u);
        for (uint32_t i = 0; i < 9u; ++i)
            CHECK (dst[i] == before[i]);
        train_simd::axpy_q15 (dst, src, 1, 0u);
        for (uint32_t i = 0; i < 9u; ++i)
            CHECK (dst[i] == before[i]);
    }

    SECTION ("hand-derived lane arithmetic")
    {
        const int32_t d[3] = { 0, _hi, _lo };
        const int32_t s[3] = { 8000, 16384, 32767 };
        Lanes dst;
        Lanes src;
        dst.prime (d, 3u);
        src.prime (s, 3u);
        /* 8000 * 16384 + 2^14 = 131088384, >> 15 = 4000 (the half-lsb of
           4000.5 floors away), so lane 0 becomes 0 + 4000 */
        train_simd::axpy_q15 (dst.lanes (), src.lanes (), 16384, 3u);
        CHECK (dst.lanes ()[0] == 4000);
        /* 32767 + (16384 * 16384 + 2^14 >> 15) = 32767 + 8192 = 40959 -> rail */
        CHECK (dst.lanes ()[1] == _hi);
        /* 32767 * 16384 = 2^29 - 2^14, so +2^14 makes the quotient exactly
           2^14 = 16384 and -32768 + 16384 = -16384 */
        CHECK (dst.lanes ()[2] == -16384);
    }

    SECTION ("a scalar at or past the half-split edge stays exact")
    {
        /* one width test per call, never per element: the split needs
           |m| <= 32767, i.e. |k| < 2^30, so 1073741823 and -1073741824 stay on
           the vector path while 1073741824 and -1073741825 send the whole call
           to the scalar loop.  Both must still produce the documented rounded
           product, which is why the reference below is int64 arithmetic. */
        const int32_t wide[] = { 1073741823, -1073741824, 1073741824,
                                 -1073741825 };
        for (const int32_t k : wide)
            {
                Lanes dst;
                Lanes src;
                dst.prime (_lane_a, _stream);
                src.prime (_lane_b, _stream);
                int64_t want[_stream];
                for (uint32_t i = 0; i < _stream; ++i)
                    want[i] = clamp16 (static_cast<int64_t> (_lane_a[i])
                                       + round15 (static_cast<int64_t> (_lane_b[i]) * k));
                train_simd::axpy_q15 (dst.lanes (), src.lanes (), k, _stream);
                INFO ("scalar " << k);
                CHECK (first_diff (dst.lanes (), want, _stream) == -1);
                CHECK (dst.untouched ());
            }

        /* the widest lane against the widest in-range scalar: -32768 * 2^15 =
           -2^30 of Q15, so lane 1 of the destination rails downward exactly */
        const int32_t d[2] = { 0, 1 };
        const int32_t s[2] = { -32768, -32768 };
        Lanes dst;
        Lanes src;
        dst.prime (d, 2u);
        src.prime (s, 2u);
        train_simd::axpy_q15 (dst.lanes (), src.lanes (), 32768, 2u);
        /* round15(-32768 * 32768) = -32768: 0 + (-32768) */
        CHECK (dst.lanes ()[0] == _lo);
        /* 1 - 32768 = -32767, one ulp off the floor */
        CHECK (dst.lanes ()[1] == -32767);
    }
}

TEST_CASE ("scale_q15 rounds in place and rails at both Q15 edges",
           "[mllee][train][simd]")
{
    for (const uint32_t n : _counts)
        for (const int32_t k : _scalars)
        {
            Lanes values;
            values.prime (_lane_a, n);
            int64_t want[_stream];
            for (uint32_t i = 0; i < n; ++i)
                want[i] = clamp16 (round15 (static_cast<int64_t> (_lane_a[i]) * k));
            train_simd::scale_q15 (values.lanes (), k, n);
            const int bad = first_diff (values.lanes (), want, n);
            INFO ("count " << n << " scalar " << k << " lane " << bad);
            CHECK (bad == -1);
            CHECK (values.untouched ());
        }

    SECTION ("the identity scalar is exactly 32768 and is idempotent")
    {
        Lanes a;
        Lanes b;
        a.prime (_lane_a, _stream);
        b.prime (_lane_a, _stream);
        /* (v * 32768 + 2^14) >> 15 = v + 0.5 floored = v for every lane */
        train_simd::scale_q15 (a.lanes (), _one, _stream);
        train_simd::scale_q15 (a.lanes (), _one, _stream);
        for (uint32_t i = 0; i < _stream; ++i)
            CHECK (a.lanes ()[i] == static_cast<int16_t> (_lane_a[i]));
        train_simd::scale_q15 (b.lanes (), 0, _stream);
        for (uint32_t i = 0; i < _stream; ++i)
            CHECK (b.lanes ()[i] == 0);
        train_simd::scale_q15 (b.lanes (), 0, 0u);
        for (uint32_t i = 0; i < _stream; ++i)
            CHECK (b.lanes ()[i] == 0);
    }

    SECTION ("hand-derived quarter and half scales")
    {
        const int32_t v[4] = { 32767, -32768, 129, 1 };
        Lanes values;
        values.prime (v, 4u);
        train_simd::scale_q15 (values.lanes (), 16384, 4u);
        /* 32767 * 16384 + 16384 = 536887296, >> 15 = 16384 (not 16383: the
           rounding term lifts the half-lsb up) */
        CHECK (values.lanes ()[0] == 16384);
        /* -32768 * 16384 + 16384 = -536854528, >> 15 = -16384 */
        CHECK (values.lanes ()[1] == -16384);
        /* 129 * 16384 + 2^14 = 2129920, >> 15 = 65: a half-code quantum
           rounds to 65 lane units, not to the 64 a truncating divide gives */
        CHECK (values.lanes ()[2] == 65);
        /* 1 * 16384 + 16384 = 32768, >> 15 = 1 */
        CHECK (values.lanes ()[3] == 1);
        train_simd::scale_q15 (values.lanes (), 32767, 0u);
        CHECK (values.lanes ()[0] == 16384);
    }

    SECTION ("negative scalars rail at -32768 and never past it")
    {
        Lanes values;
        values.prime (_lane_a, _stream);
        train_simd::scale_q15 (values.lanes (), -32768, _stream);
        for (uint32_t i = 0; i < _stream; ++i)
            {
                /* -v for the identity scale, with the +2^14 term still
                   flooring, so a positive lane keeps its exact magnitude */
                const int64_t want = clamp16 (round15 (-static_cast<int64_t> (_lane_a[i]) * 32768));
                CHECK (static_cast<int64_t> (values.lanes ()[i]) == want);
                CHECK (values.lanes ()[i] >= _lo);
                CHECK (values.lanes ()[i] <= _hi);
            }
    }
}

TEST_CASE ("sgd_step_i8 moves the code grid with round-half-away and rails",
           "[mllee][train][simd]")
{
    for (const uint32_t n : _counts)
        for (const int32_t lr : _scalars)
        {
            Codes params;
            Lanes grad;
            params.prime (_codes_in, n);
            grad.prime (_lane_a, n);
            int64_t want[_stream];
            for (uint32_t i = 0; i < n; ++i)
                want[i] = i8_rail (static_cast<int64_t> (static_cast<int8_t> (
                                       _codes_in[i]))
                                   + grid_move (round15 (static_cast<int64_t> (_lane_a[i]) * lr)));
            train_simd::sgd_step_i8 (params.codes (), grad.lanes (), lr, n);
            const int bad = first_diff8 (params.codes (), want, n);
            INFO ("count " << n << " lr " << lr << " lane " << bad);
            CHECK (bad == -1);
            CHECK (params.untouched ());
            /* the gradient buffer is read-only here, so a second step on the
               same lanes is a second move, not a fresh product */
            for (uint32_t i = 0; i < n; ++i)
                CHECK (grad.lanes ()[i] == static_cast<int16_t> (_lane_a[i]));
            for (uint32_t i = 0; i < n; ++i)
                {
                    CHECK (params.codes ()[i] >= -127);
                    CHECK (params.codes ()[i] <= 127);
                }
        }

    SECTION ("grid quantums: half-a-code ties move away from zero")
    {
        /* at lr 32768 the Q15 product is the gradient verbatim, so each lane's
           code step is exactly its own value and only the /258 rule is under
           test here */
        const int32_t c[6] = { 0, 0, 0, 0, 127, -127 };
        const int32_t g[6] = { 129, 128, -129, -128, 32767, -32768 };
        Codes params;
        Lanes grad;
        params.prime (c, 6u);
        grad.prime (g, 6u);
        train_simd::sgd_step_i8 (params.codes (), grad.lanes (), 32768, 6u);
        /* code step 129 is exactly half a grid quantum: (129 + 129) / 258 = 1 */
        CHECK (params.codes ()[0] == 1);
        /* one unit short: (128 + 129) / 258 = 0 */
        CHECK (params.codes ()[1] == 0);
        /* -129 mirrors it: -((129 + 129) / 258) = -1, away from zero */
        CHECK (params.codes ()[2] == -1);
        CHECK (params.codes ()[3] == 0);
        /* 32767 code step = (32767 + 129) / 258 = 127 moves, and 127 + 127
           rails back to 127: never -128, never past the grid */
        CHECK (params.codes ()[4] == 127);
        CHECK (params.codes ()[5] == -127);
        for (uint32_t i = 0; i < 6u; ++i)
            CHECK (params.codes ()[i] != -128);

        /* a railed gradient at a half scale still lands on a whole code move:
           8000 * 16384 + 2^14 >> 15 = 4000, (4000 + 129) / 258 = 16 */
        const int32_t c2[2] = { 0, 111 };
        const int32_t g2[2] = { 8000, -8000 };
        params.prime (c2, 2u);
        grad.prime (g2, 2u);
        train_simd::sgd_step_i8 (params.codes (), grad.lanes (), 16384, 2u);
        CHECK (params.codes ()[0] == 16);
        CHECK (params.codes ()[1] == 95);   /* 111 - 16 */

        /* and a zero-count call on the same buffers is a no-op */
        train_simd::sgd_step_i8 (params.codes (), grad.lanes (), 16384, 0u);
        CHECK (params.codes ()[0] == 16);
        CHECK (params.codes ()[1] == 95);
        CHECK (params.untouched ());
    }

    SECTION ("the widest in-range move rails whichever parameter it starts from")
    {
        /* lr 32768 with a railed gradient is a code step of 32767, i.e. 127
           codes; from code -127 that is a move to 0, from 0 to 127 */
        const int32_t c[8] = { -127, -100, -3, 0, 3, 100, 127, -127 };
        const int32_t g[8] = { 32767, 32767, 32767, 32767, 32767, 32767, 32767, -32768 };
        Codes params;
        Lanes grad;
        params.prime (c, 8u);
        grad.prime (g, 8u);
        train_simd::sgd_step_i8 (params.codes (), grad.lanes (), 32768, 8u);
        CHECK (params.codes ()[0] == 0);      /* -127 + 127 */
        CHECK (params.codes ()[1] == 27);     /* -100 + 127 */
        CHECK (params.codes ()[2] == 124);    /* -3 + 127 */
        CHECK (params.codes ()[3] == 127);    /* 0 + 127 */
        CHECK (params.codes ()[4] == 127);    /* 3 + 127, railed */
        CHECK (params.codes ()[5] == 127);
        CHECK (params.codes ()[6] == 127);
        /* -32768 * 32768 + 2^14 >> 15 = -32768, -((32768 + 129) / 258) = -127,
           so -127 - 127 rails at -127 and never becomes the -128 code */
        CHECK (params.codes ()[7] == -127);
        for (uint32_t i = 0; i < 8u; ++i)
            CHECK (params.codes ()[i] != -128);

        /* Past the grid the assembly pre-rails the move at +-254 (packssdw plus
           pminsw/pmaxsw) while the fallback adds the full move.  For any valid
           code that is the same answer: 65790 * 32767 + 2^14 >> 15 = 65788 code
           units, (65788 + 129) / 258 = 255 moves, and both -127 + 254 and
           -127 + 255 sit on or past the +127 rail.  An lr this wide is outside
           the Q15 range, so this is the overflow edge, not a normal step. */
        const int32_t c2[8] = { -127, -126, -3, 0, 3, 126, 127, -127 };
        const int32_t g2[8] = { 32767, 32767, -32768, 32767, -32768, 32767, 32767, -32768 };
        params.prime (c2, 8u);
        grad.prime (g2, 8u);
        train_simd::sgd_step_i8 (params.codes (), grad.lanes (), 65790, 8u);
        CHECK (params.codes ()[0] == 127);
        CHECK (params.codes ()[1] == 127);
        CHECK (params.codes ()[2] == -127);
        CHECK (params.codes ()[3] == 127);
        CHECK (params.codes ()[4] == -127);
        CHECK (params.codes ()[5] == 127);
        CHECK (params.codes ()[6] == 127);
        CHECK (params.codes ()[7] == -127);
        for (uint32_t i = 0; i < 8u; ++i)
            CHECK (params.codes ()[i] != -128);
    }

    SECTION ("lr zero moves nothing and normalises the one illegal code")
    {
        Codes params;
        Lanes grad;
        params.prime (_codes_in, _stream);
        grad.prime (_lane_a, _stream);
        train_simd::sgd_step_i8 (params.codes (), grad.lanes (), 0, _stream);
        for (uint32_t i = 0; i < _stream; ++i)
            {
                /* a zero code step still passes the sum through _S_i8_code, so
                   the stray -128 code in the stream is rewritten to -127: the
                   update never leaves an out-of-grid code behind */
                CHECK (params.codes ()[i] == i8_rail (_codes_in[i]));
                CHECK (params.codes ()[i] >= -127);
                CHECK (params.codes ()[i] <= 127);
                CHECK (params.codes ()[i] != -128);
            }
        CHECK (grad.untouched ());
    }
}

TEST_CASE ("relu_backward routes the gradient by the forward sign",
           "[mllee][train][simd]")
{
    for (const uint32_t n : _counts)
    {
        Lanes out;
        Codes input;
        Lanes in;
        out.prime (_lane_a, n);
        input.prime (_codes_in, n);
        in.prime (_lane_b, n);
        int64_t want[_stream];
        for (uint32_t i = 0; i < n; ++i)
            want[i] = _codes_in[i] >= 0 ? _lane_a[i] : 0;
        train_simd::relu_backward (out.lanes (), input.codes (), in.lanes (), n);
        const int bad = first_diff (in.lanes (), want, n);
        INFO ("count " << n << " lane " << bad);
        CHECK (bad == -1);
        CHECK (in.untouched ());
        CHECK (out.untouched ());
        /* a gated lane is written as zero, not left at its prior value: the
           destination window is fully owned by the kernel */
        for (uint32_t i = 0; i < n; ++i)
            CHECK (in.lanes ()[i] == static_cast<int16_t> (want[i]));
    }

    SECTION ("the gate keeps the zero code and rejects -128 as negative")
    {
        const int32_t c[4] = { 0, -128, 127, -1 };
        const int32_t g[4] = { 12345, 23456, -32768, _hi };
        Lanes out;
        Codes input;
        Lanes in;
        out.prime (g, 4u);
        input.prime (c, 4u);
        in.prime (_lane_a, 4u);
        train_simd::relu_backward (out.lanes (), input.codes (), in.lanes (), 4u);
        CHECK (in.lanes ()[0] == 12345);   /* code 0 >= 0: routed */
        CHECK (in.lanes ()[1] == 0);       /* -128 < 0: gated */
        CHECK (in.lanes ()[2] == -32768);  /* positive code: routed verbatim */
        CHECK (in.lanes ()[3] == 0);
    }

    SECTION ("an empty gate pass writes no lane")
    {
        Lanes out;
        Codes input;
        Lanes in;
        out.prime (_lane_a, 8u);
        input.prime (_codes_in, 8u);
        in.prime (_lane_b, 8u);
        train_simd::relu_backward (out.lanes (), input.codes (), in.lanes (), 0u);
        CHECK (in.untouched ());
        for (uint32_t i = 0; i < 8u; ++i)
            CHECK (in.lanes ()[i] == static_cast<int16_t> (_lane_b[i]));
    }
}

TEST_CASE ("sum_abs_q15 is the saturating L1 lane reduction",
           "[mllee][train][simd]")
{
    for (const uint32_t n : _counts)
    {
        Lanes values;
        values.prime (_lane_a, n);
        int64_t total = 0;
        for (uint32_t i = 0; i < n; ++i)
            {
                const int64_t v = _lane_a[i];
                total += v < 0 ? -v : v;
            }
        const int32_t got = train_simd::sum_abs_q15 (values.lanes (), n);
        INFO ("count " << n << " sum " << got << " want " << total);
        CHECK (static_cast<int64_t> (got) == (total > 2147483647LL ? 2147483647LL : total));
        CHECK (values.untouched ());

        /* the same magnitudes one lane further in, on a 2 mod 8 pointer */
        if (n >= 2u)
            {
                int64_t shift_total = 0;
                for (uint32_t i = 1; i < n; ++i)
                    shift_total += _lane_a[i] < 0 ? -static_cast<int64_t> (_lane_a[i])
                                                  : _lane_a[i];
                CHECK (static_cast<int64_t> (
                           train_simd::sum_abs_q15 (values.lanes () + 1, n - 1u))
                       == (shift_total > 2147483647LL ? 2147483647LL : shift_total));
            }
    }

    SECTION ("the -32768 lane contributes 32768, not -32768")
    {
        const int32_t v[4] = { _lo, _lo, _hi, 1 };
        Lanes values;
        values.prime (v, 4u);
        /* 32768 + 32768 + 32767 + 1 = 98304: the rail needs 17 bits, which is
           why the kernel widens before accumulating */
        CHECK (train_simd::sum_abs_q15 (values.lanes (), 4u) == 98304);
        CHECK (train_simd::sum_abs_q15 (values.lanes (), 0u) == 0);
        CHECK (values.untouched ());
    }

    SECTION ("the int32 rail saturates upward only")
    {
        /* 65536 lanes of -32768 are 2147483648 magnitudes, one past INT32_MAX */
        static int16_t big[_stream + 66000u];
        for (uint32_t i = 0; i < 66000u; ++i)
            big[i] = static_cast<int16_t> (_lo);
        CHECK (train_simd::sum_abs_q15 (big, 65536u) == 2147483647);
        /* 65535 lanes stay one ulp under the rail: 65535 * 32768 = 2147450880 */
        CHECK (train_simd::sum_abs_q15 (big, 65535u) == 2147450880);
    }
}

TEST_CASE ("sum_sq_q15 is the saturating Q30 square reduction",
           "[mllee][train][simd]")
{
    for (const uint32_t n : _counts)
    {
        Lanes values;
        values.prime (_lane_a, n);
        int64_t total = 0;
        bool railed = false;
        for (uint32_t i = 0; i < n; ++i)
            {
                total += static_cast<int64_t> (_lane_a[i]) * _lane_a[i];
                if (total > 2147483647LL)
                    railed = true;
            }
        const int32_t got = train_simd::sum_sq_q15 (values.lanes (), n);
        const int32_t want = railed ? 2147483647 : static_cast<int32_t> (total);
        INFO ("count " << n << " sum " << got << " want " << want);
        CHECK (got == want);
        CHECK (values.untouched ());
        if (n >= 2u)
            {
                int64_t shift_total = 0;
                bool shift_railed = false;
                for (uint32_t i = 1; i < n; ++i)
                    {
                        shift_total += static_cast<int64_t> (_lane_a[i]) * _lane_a[i];
                        if (shift_total > 2147483647LL)
                            shift_railed = true;
                    }
                CHECK (train_simd::sum_sq_q15 (values.lanes () + 1, n - 1u)
                       == (shift_railed ? 2147483647 : static_cast<int32_t> (shift_total)));
            }
    }

    SECTION ("hand-derived squares and the saturation exactly one past the rail")
    {
        const int32_t v[4] = { 300, -300, 16384, _hi };
        Lanes values;
        values.prime (v, 4u);
        /* 90000 + 90000 + 268435456 + 1073676289 = 1342291745, still under
           INT32_MAX, so the reduction reports the exact Q30 total */
        CHECK (train_simd::sum_sq_q15 (values.lanes (), 4u) == 1342291745);
        CHECK (train_simd::sum_sq_q15 (values.lanes (), 0u) == 0);

        /* the widest square an int16 lane can hold is -32768 squared = 2^30, so
           one lane is 1073741824 and two lanes are 2147483648 -- exactly one
           past INT32_MAX, which is why the rail reports 2147483647 */
        const int32_t rail[2] = { _lo, _lo };
        Lanes halves;
        halves.prime (rail, 2u);
        CHECK (train_simd::sum_sq_q15 (halves.lanes (), 1u) == 1073741824);
        CHECK (train_simd::sum_sq_q15 (halves.lanes (), 2u) == 2147483647);

        /* eight lanes of 16384 squares are 2147483648, one past the rail, and
           the crossing happens at the group collapse: seven lanes -- a
           pure-tail run -- stay exact at 1879048192 */
        const int32_t edge[9] = { 16384, 16384, 16384, 16384, 16384, 16384,
                                  16384, 16384, 1000 };
        Lanes wide;
        wide.prime (edge, 9u);
        CHECK (train_simd::sum_sq_q15 (wide.lanes (), 7u) == 1879048192);
        CHECK (train_simd::sum_sq_q15 (wide.lanes (), 8u) == 2147483647);
        /* once clamped the tail cannot pull the total back down: 1000 squared
           is added and the clamp re-applies */
        CHECK (train_simd::sum_sq_q15 (wide.lanes (), 9u) == 2147483647);
    }
}

TEST_CASE ("codes_to_q15 widens the code grid exactly, -128 included",
           "[mllee][train][simd]")
{
    for (const uint32_t n : _counts)
    {
        Codes codes;
        Lanes values;
        codes.prime (_codes_in, n);
        values.prime (_lane_b, n);
        int64_t want[_stream];
        for (uint32_t i = 0; i < n; ++i)
            {
                const int64_t product = static_cast<int64_t> (static_cast<int8_t> (_codes_in[i])) * _step;
                /* the widening is a 16-bit wrap, exactly like the kernel: for
                   the -128 code the product leaves the lane */
                want[i] = static_cast<int64_t> (static_cast<int16_t> (static_cast<uint16_t> (
                    static_cast<uint32_t> (product) & 0xFFFFu)));
            }
        train_simd::codes_to_q15 (codes.codes (), values.lanes (), n);
        const int bad = first_diff (values.lanes (), want, n);
        INFO ("count " << n << " lane " << bad);
        CHECK (bad == -1);
        CHECK (values.untouched ());
        CHECK (codes.untouched ());
    }

    SECTION ("the grid rails are 32766 and -32766, and -128 wraps")
    {
        const int32_t c[8] = { 127, -127, 1, -1, 0, 64, -64, -128 };
        Codes codes;
        Lanes values;
        codes.prime (c, 8u);
        values.prime (_lane_b, 8u);
        train_simd::codes_to_q15 (codes.codes (), values.lanes (), 8u);
        CHECK (values.lanes ()[0] == 32766);   /* 127 * 258 */
        CHECK (values.lanes ()[1] == -32766);
        CHECK (values.lanes ()[2] == 258);
        CHECK (values.lanes ()[3] == -258);
        CHECK (values.lanes ()[4] == 0);
        CHECK (values.lanes ()[5] == 16512);   /* 64 * 258 */
        CHECK (values.lanes ()[6] == -16512);
        /* -128 * 258 = -33024, which as an int16 lane reads 32512: an invalid
           code produces a wrapped lane in both bodies, never a trap */
        CHECK (values.lanes ()[7] == 32512);
        train_simd::codes_to_q15 (codes.codes (), values.lanes (), 0u);
        CHECK (values.untouched ());
    }

    SECTION ("all 256 int8 inputs widen deterministically, group and tail alike")
    {
        alignas (8) int8_t stream[8];
        int32_t mismatches = 0;
        int32_t seen = 0;
        for (int32_t base = -128; base < 128; base += 8)
            {
                for (int32_t k = 0; k < 8; ++k)
                    stream[k] = static_cast<int8_t> (base + k);
                alignas (8) int16_t out[8];
                fill (out, _lane_b, 8u);
                /* one eight-code group: the movdqu body */
                train_simd::codes_to_q15 (stream, out, 8u);
                for (int32_t k = 0; k < 8; ++k)
                    {
                        ++seen;
                        const int64_t want = static_cast<int64_t> (static_cast<int16_t> (
                            static_cast<uint16_t> (static_cast<uint32_t> (
                                static_cast<int64_t> (base + k) * _step) & 0xFFFFu)));
                        if (out[k] != want)
                            ++mismatches;
                    }
                /* the same codes one at a time: the scalar tail loop */
                for (int32_t k = 0; k < 8; ++k)
                    {
                        alignas (8) int16_t solo[8];
                        fill (solo, _lane_b, 8u);
                        train_simd::codes_to_q15 (stream + k, solo, 1u);
                        ++seen;
                        const int64_t want = static_cast<int64_t> (static_cast<int16_t> (
                            static_cast<uint16_t> (static_cast<uint32_t> (
                                static_cast<int64_t> (base + k) * _step) & 0xFFFFu)));
                        if (solo[0] != want)
                            ++mismatches;
                        for (int32_t j = 1; j < 8; ++j)
                            if (solo[j] != static_cast<int16_t> (_lane_b[j]))
                                ++mismatches;
                    }
            }
        /* the sweep covered every int8 value on both paths, twice, and nothing
           disagreed -- including the -128 code that leaves the grid */
        CHECK (seen == 512);
        CHECK (mismatches == 0);
    }

    SECTION ("a zero count writes no lane")
    {
        Codes codes;
        Lanes values;
        codes.prime (_codes_in, 0u);
        values.prime (_lane_a, 0u);
        train_simd::codes_to_q15 (codes.codes (), values.lanes (), 0u);
        CHECK (values.untouched ());
        CHECK (codes.untouched ());
        train_simd::codes_to_q15 (nullptr, values.lanes (), 0u);
        CHECK (values.untouched ());
    }
}

TEST_CASE ("q15_to_codes narrows with rounding and never emits -128",
           "[mllee][train][simd]")
{
    for (const uint32_t n : _counts)
    {
        Lanes values;
        Codes codes;
        values.prime (_lane_a, n);
        codes.prime (_codes_in, n);
        int64_t want[_stream];
        for (uint32_t i = 0; i < n; ++i)
            want[i] = narrow_code (_lane_a[i]);
        train_simd::q15_to_codes (values.lanes (), codes.codes (), n);
        const int bad = first_diff8 (codes.codes (), want, n);
        INFO ("count " << n << " lane " << bad);
        CHECK (bad == -1);
        CHECK (codes.untouched ());
        CHECK (values.untouched ());
        for (uint32_t i = 0; i < n; ++i)
            {
                CHECK (codes.codes ()[i] >= -127);
                CHECK (codes.codes ()[i] <= 127);
                CHECK (codes.codes ()[i] != -128);
            }
    }

    SECTION ("the first full code step is 130, not half of 258")
    {
        const int32_t v[6] = { 32767, _lo, 16384, 16383, 129, 130 };
        Lanes values;
        Codes codes;
        values.prime (v, 6u);
        codes.prime (_codes_in, 6u);
        train_simd::q15_to_codes (values.lanes (), codes.codes (), 6u);
        /* (32767 * 127 + 16384) >> 15 = 4177920 >> 15 = 127 */
        CHECK (codes.codes ()[0] == 127);
        /* (-32768) uses the magnitude: (32768 * 127 + 16384) >> 15 = 127, so
           the rail narrows to -127 and the -128 code is never produced */
        CHECK (codes.codes ()[1] == -127);
        CHECK (codes.codes ()[2] == 64);   /* (16384 * 127 + 16384) = 2097152, >> 15 = 64 */
        CHECK (codes.codes ()[3] == 63);   /* 2080768 + 16384 = 2097025, >> 15 = 63 */
        /* one half of the 258-unit grid step is 129, yet 129 * 127 + 16384 =
           32767 falls one unit short of 2^15, so it narrows to code 0 */
        CHECK (codes.codes ()[4] == 0);
        /* 130 is the first magnitude that reaches code 1: 16510 + 16384 =
           32894, >> 15 = 1 */
        CHECK (codes.codes ()[5] == 1);
        train_simd::q15_to_codes (values.lanes (), codes.codes (), 0u);
        CHECK (codes.untouched ());
    }

    SECTION ("every int16 value narrows on the grid, SIMD body and tail")
    {
        alignas (8) int16_t window[8];
        int32_t mismatches = 0;
        int32_t minus128 = 0;
        int32_t seen = 0;
        for (int32_t base = -32768; base < 32768; base += 8)
            {
                for (int32_t k = 0; k < 8; ++k)
                    window[k] = static_cast<int16_t> (base + k);
                alignas (8) int8_t out[8];
                for (int32_t k = 0; k < 8; ++k)
                    out[k] = _sentinel8;
                train_simd::q15_to_codes (window, out, 8u);
                for (int32_t k = 0; k < 8; ++k)
                    {
                        ++seen;
                        if (out[k] != narrow_code (base + k))
                            ++mismatches;
                        if (out[k] == -128)
                            ++minus128;
                    }
                /* the same values through the scalar tail, one at a time from a
                   deliberately unaligned base */
                for (int32_t k = 0; k < 8; ++k)
                    {
                        alignas (8) int8_t solo[8];
                        for (int32_t j = 0; j < 8; ++j)
                            solo[j] = _sentinel8;
                        train_simd::q15_to_codes (window + k, solo, 1u);
                        ++seen;
                        if (solo[0] != narrow_code (base + k))
                            ++mismatches;
                        if (solo[0] == -128)
                            ++minus128;
                        for (int32_t j = 1; j < 8; ++j)
                            if (solo[j] != _sentinel8)
                                ++mismatches;
                    }
            }
        /* the -128 code is unreachable from both paths, over the whole int16
           domain, and the doubling trick (|v| * 254 + 32768) >> 16 agrees with
           the fallback's (|v| * 127 + 16384) >> 15 everywhere */
        CHECK (seen == 131072);
        CHECK (mismatches == 0);
        CHECK (minus128 == 0);
    }

    SECTION ("the code/Q15 round trip is lossless on the grid")
    {
        Codes codes;
        Lanes wide;
        Codes back;
        /* a stream of every valid code in grid order */
        int32_t ramp[_stream];
        for (uint32_t i = 0; i < _stream; ++i)
            ramp[i] = static_cast<int32_t> (i) - 16;
        codes.prime (ramp, _stream);
        wide.prime (_lane_b, _stream);
        back.prime (ramp, _stream);
        train_simd::codes_to_q15 (codes.codes (), wide.lanes (), _stream);
        train_simd::q15_to_codes (wide.lanes (), back.codes (), _stream);
        for (uint32_t i = 0; i < _stream; ++i)
            CHECK (back.codes ()[i] == static_cast<int8_t> (ramp[i]));
    }
}

TEST_CASE ("train_simd calls are alignment-stable and allocation-free",
           "[mllee][train][simd]")
{
    /* one alignment test per call: a stream whose base pointer is 2 mod 8 (not
       the 4-byte lane alignment the pool carves) produces the same per-lane
       results as the 8-aligned base. */
    alignas (8) int16_t aligned[_stream];
    alignas (8) int16_t shifted[_stream];
    fill (aligned, _lane_a, _stream);
    shifted[0] = _sentinel;
    for (uint32_t i = 1u; i < _stream; ++i)
        shifted[i] = static_cast<int16_t> (_lane_a[i]);

    train_simd::scale_q15 (aligned, 8192, _stream);
    train_simd::scale_q15 (shifted + 1, 8192, _stream - 1u);
    int32_t diverged = 0;
    for (uint32_t i = 1u; i < _stream; ++i)
        if (shifted[i] != aligned[i])
            ++diverged;
    CHECK (diverged == 0);
    /* the lane the deeper run skipped keeps its primed value, so the equality
       above really compares 32 off-grid lanes and is not vacuous */
    CHECK (shifted[0] == _sentinel);
    /* 32767 * 8192 = 2^29 - 8, and +2^14 makes the quotient 8192.5, whose
       floor is 8192: the rounding term lifts it a whole unit above half scale */
    CHECK (aligned[0] == 8192);

    /* axpy over a lane-shifted destination and a differently shifted source */
    Lanes src;
    src.prime (_lane_b, _stream);
    int64_t want[_stream];
    for (uint32_t i = 0; i + 1 < _stream; ++i)
        want[i] = clamp16 (static_cast<int64_t> (aligned[i + 1])
                           + round15 (static_cast<int64_t> (_lane_b[i]) * 3277));
    train_simd::axpy_q15 (aligned + 1, src.lanes (), 3277, _stream - 1u);
    CHECK (first_diff (aligned + 1, want, _stream - 1u) == -1);

    /* sgd over an odd-byte code stream: int8 lanes have no alignment
       requirement, so the same codes at any offset move identically */
    Codes params;
    params.prime (_codes_in, _stream);
    Codes odd;
    odd.prime (_codes_in + 1, _stream - 1u);
    Lanes grad;
    grad.prime (_lane_a, _stream);
    train_simd::sgd_step_i8 (params.codes (), grad.lanes (), 3277, _stream);
    grad.prime (_lane_a + 1, _stream - 1u);
    train_simd::sgd_step_i8 (odd.codes (), grad.lanes (), 3277, _stream - 1u);
    int32_t codes_diverged = 0;
    for (uint32_t i = 0; i + 1 < _stream; ++i)
        if (odd.codes ()[i] != params.codes ()[i + 1])
            ++codes_diverged;
    CHECK (codes_diverged == 0);
    CHECK (odd.untouched ());
    /* lane 0: 32767 * 3277 + 2^14 >> 15 = 3277 of Q15 code step, and
       (3277 + 129) / 258 = 13 codes from the zero code it starts at.  Lane 1
       mirrors it negative: -32768 * 3277 floors to -3277, so 1 moves to -12. */
    CHECK (params.codes ()[0] == 13);
    CHECK (params.codes ()[1] == -12);

    /* nothing in this header touches a pool: the training arena stays empty
       across a full sweep of every primitive */
    AlignedArena<256> arena;
    Pool pool = arena.pool ();
    const PoolStats before = pool.stats ();
    Lanes values;
    Codes codes;
    values.prime (_lane_a, _stream);
    codes.prime (_codes_in, _stream);
    for (const uint32_t n : _counts)
        {
            volatile int32_t sink = 0;
            sink = train_simd::dot_q15 (values.lanes (), values.lanes (), n);
            sink += static_cast<int32_t> (train_simd::add_grad (
                values.lanes (), values.lanes (), n));
            train_simd::axpy_q15 (values.lanes (), values.lanes (), 258, n);
            train_simd::scale_q15 (values.lanes (), 258, n);
            train_simd::sgd_step_i8 (codes.codes (), values.lanes (), 258, n);
            train_simd::relu_backward (values.lanes (), codes.codes (),
                                       values.lanes (), n);
            sink += train_simd::sum_abs_q15 (values.lanes (), n);
            sink += train_simd::sum_sq_q15 (values.lanes (), n);
            train_simd::codes_to_q15 (codes.codes (), values.lanes (), n);
            train_simd::q15_to_codes (values.lanes (), codes.codes (), n);
            (void) sink;
        }
    const PoolStats after = pool.stats ();
    CHECK (after._M_used == before._M_used);
    CHECK (after._M_slabs == before._M_slabs);
    CHECK (after._M_overflows == before._M_overflows);
    CHECK (before._M_used == 0u);
}

TEST_CASE ("train_simd primitives ignore a zero count without dereferencing",
           "[mllee][train][simd]")
{
    /* count == 0 is the one case every kernel must reach without touching a
       single buffer: nullptr arguments are the strictest probe, and the two
       reductions must report their empty value. */
    CHECK (train_simd::dot_q15 (nullptr, nullptr, 0u) == 0);
    CHECK (train_simd::add_grad (nullptr, nullptr, 0u) == 0u);
    CHECK (train_simd::sum_abs_q15 (nullptr, 0u) == 0);
    CHECK (train_simd::sum_sq_q15 (nullptr, 0u) == 0);

    int16_t guard_lane = _sentinel;
    int8_t guard_code = _sentinel8;
    train_simd::axpy_q15 (&guard_lane, nullptr, 3277, 0u);
    train_simd::scale_q15 (&guard_lane, 3277, 0u);
    train_simd::relu_backward (nullptr, nullptr, &guard_lane, 0u);
    CHECK (guard_lane == _sentinel);
    train_simd::sgd_step_i8 (&guard_code, nullptr, 3277, 0u);
    train_simd::codes_to_q15 (nullptr, &guard_lane, 0u);
    train_simd::q15_to_codes (&guard_lane, &guard_code, 0u);
    CHECK (guard_code == _sentinel8);
    CHECK (guard_lane == _sentinel);
}

