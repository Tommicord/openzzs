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

/*
 * mllee Cortex-M4 assembly self-test.
 *
 * Runs the Thumb kernels from mllee/asm-cortex-m4 on the QEMU `mps2-an385`
 * Cortex-M4 model (tools/run_mllee_m4_qemu.sh) and compares every one of them
 * against the core's documented semantics, recomputed here in plain C++ with
 * the same int32 accumulator widths the portable fallbacks use.  The point is
 * to prove the kernels are legal ARMv7E-M code that produces byte-identical
 * results, because the x86 test suite can only ever exercise the fallbacks.
 *
 * Freestanding: no libc, no C++ runtime, output through semihosting.
 */

#include <cstdint>

extern "C"
{
    int32_t zzs_mllee_dot_i8 (const int8_t *lhs, const int8_t *rhs,
                              uint32_t count);
    int32_t zzs_mllee_requant_q15 (int32_t accumulator, int32_t multiplier);
    void zzs_mllee_unpack_i4 (const uint8_t *source, int8_t *destination,
                              uint32_t bytes);
    void zzs_mllee_add_sat_i8 (const int8_t *lhs, const int8_t *rhs,
                               int8_t *destination, uint32_t count);
    void zzs_mllee_sub_sat_i8 (const int8_t *lhs, const int8_t *rhs,
                               int8_t *destination, uint32_t count);
    int32_t zzs_mllee_sum_i8 (const int8_t *values, uint32_t count);
    int8_t zzs_mllee_max_i8 (const int8_t *values, uint32_t count);
    void zzs_mllee_stats_i8 (const int8_t *values, uint32_t count, int32_t *sum,
                             int32_t *square_sum);

    /* Training lane kernels (mllee/asm-cortex-m4/mllee_train_q15_*.S).  The
       x86 suite can only ever run train_simd's portable fallbacks, so these are
       the routines that most need on-device proof. */
    uint32_t zzs_mllee_train_add_grad_q15 (int16_t *destination,
                                          const int16_t *source, uint32_t count);
    void zzs_mllee_train_axpy_q15 (int16_t *destination, const int16_t *source,
                                   int32_t scalar_q15, uint32_t count);
    void zzs_mllee_train_scale_q15 (int16_t *values, int32_t scalar_q15,
                                    uint32_t count);
    void zzs_mllee_train_relu_bwd (const int16_t *grad_out, const int8_t *input,
                                   int16_t *grad_in, uint32_t count);
    int32_t zzs_mllee_train_sum_abs_q15 (const int16_t *values, uint32_t count);
    int32_t zzs_mllee_train_sum_sq_q15 (const int16_t *values, uint32_t count);
    int32_t zzs_mllee_train_dot_q15 (const int16_t *lhs, const int16_t *rhs,
                                     uint32_t count);
    void zzs_mllee_train_sgd_step_i8 (int8_t *parameters, const int16_t *grad,
                                      int32_t lr_q15, uint32_t count);
    void zzs_mllee_train_codes_to_q15 (const int8_t *codes, int16_t *values,
                                       uint32_t count);
    void zzs_mllee_train_q15_to_codes (const int16_t *values, int8_t *codes,
                                       uint32_t count);
    void zzs_mllee_shm_write0 (const char *text);
    void zzs_mllee_shm_exit (uint32_t failing);

    /* One guarded instruction each; 1 = the model executes it, 0 = it faults. */
    int zzs_m4_probe_sxtb16 (void);
    int zzs_m4_probe_smlad (void);
    int zzs_m4_probe_smlabb (void);
    int zzs_m4_probe_ssat (void);
    int zzs_m4_probe_usada8 (void);
}

namespace
{
constexpr uint32_t _S_cap = 520u;
constexpr uint32_t _S_rows = 2u;

alignas (4) int8_t _S_lhs[_S_cap];
alignas (4) int8_t _S_rhs[_S_cap];
alignas (4) uint8_t _S_packed[_S_cap];
alignas (4) int8_t _S_out[_S_cap * 2u + 4u];
alignas (4) int32_t _S_stats[_S_rows][2];
alignas (4) int16_t _S_lane[_S_cap];
alignas (4) int16_t _S_lane_b[_S_cap];
alignas (4) int16_t _S_lane_ref[_S_cap];
alignas (4) int16_t _S_lane_bref[_S_cap];
alignas (4) int16_t _S_grad[_S_cap];
alignas (4) int8_t _S_code[_S_cap];
alignas (4) int8_t _S_code_ref[_S_cap];

uint32_t _S_seed = 0x13579bdfu;
uint32_t _S_failures = 0;
uint32_t _S_checks = 0;
uint32_t _S_skipped_lane = 0;

/* True once the model is known to execute the SXTB16 + SMLAD lane pair that the
 * dot, sum and stats word loops are built from. */
bool _S_sxtb16_ok = false;
bool _S_smlad_ok = false;
bool _S_lane_ok = false;

char _S_line[96];
uint32_t _S_pos;

[[nodiscard]] uint32_t
rnd () noexcept
{
    _S_seed = _S_seed * 1664525u + 1013904223u;
    return _S_seed >> 24u;
}

void
emit (const char *const text) noexcept
{
    zzs_mllee_shm_write0 (text);
}

void
put (const char c) noexcept
{
    _S_line[_S_pos++] = c;
    _S_line[_S_pos] = '\0';
}

void
put_text (const char *const text) noexcept
{
    for (uint32_t i = 0; text[i] != '\0'; ++i)
        put (text[i]);
}

void
put_dec (const int32_t value) noexcept
{
    char digits[12];
    uint32_t n = 0;
    uint32_t magnitude;
    if (value < 0)
        {
            put ('-');
            magnitude = static_cast<uint32_t> (-static_cast<int64_t> (value));
        }
    else
        magnitude = static_cast<uint32_t> (value);
    do
        {
            digits[n++] = static_cast<char> ('0' + magnitude % 10u);
            magnitude /= 10u;
        }
    while (magnitude != 0u);
    while (n != 0)
        put (digits[--n]);
}

void
flush_line () noexcept
{
    _S_line[_S_pos] = '\0';
    emit (_S_line);
    _S_pos = 0;
}

void
report (const char *const name, const uint32_t offset, const uint32_t count,
        const int32_t expected, const int32_t actual) noexcept
{
    ++_S_failures;
    _S_pos = 0;
    put_text ("FAIL ");
    put_text (name);
    put_text (" off=");
    put_dec (static_cast<int32_t> (offset));
    put_text (" n=");
    put_dec (static_cast<int32_t> (count));
    put_text (" expected=");
    put_dec (expected);
    put_text (" got=");
    put_dec (actual);
    put ('\n');
    flush_line ();
}

/* --- Reference semantics: what the core promises, independent of any kernel. */

[[nodiscard]] int32_t
ref_dot (const int8_t *const lhs, const int8_t *const rhs,
         const uint32_t count) noexcept
{
    uint32_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<uint32_t> (static_cast<int32_t> (lhs[i]) * rhs[i]);
    return static_cast<int32_t> (total);
}

[[nodiscard]] int32_t
ref_requant (const int32_t accumulator, const int32_t multiplier) noexcept
{
    const int64_t product
        = static_cast<int64_t> (accumulator) * static_cast<int64_t> (multiplier);
    return static_cast<int32_t> (
        static_cast<uint32_t> (static_cast<uint64_t> ((product + 16384) >> 15) & 0xFFFFFFFFull));
}

[[nodiscard]] int8_t
ref_from_int4 (const uint8_t nibble) noexcept
{
    return static_cast<int8_t> ((nibble & 0x08u) != 0u
                                    ? static_cast<int32_t> (nibble) - 16
                                    : static_cast<int32_t> (nibble));
}

[[nodiscard]] int8_t
ref_code (const int32_t value) noexcept
{
    if (value < -127)
        return -127;
    if (value > 127)
        return 127;
    return static_cast<int8_t> (value);
}

[[nodiscard]] int8_t
ref_add_sat (const int8_t lhs, const int8_t rhs) noexcept
{
    return ref_code (static_cast<int32_t> (lhs) + rhs);
}

[[nodiscard]] int8_t
ref_sub_sat (const int8_t lhs, const int8_t rhs) noexcept
{
    return ref_code (static_cast<int32_t> (lhs) - rhs);
}

[[nodiscard]] int32_t
ref_sum (const int8_t *const values, const uint32_t count) noexcept
{
    uint32_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<uint32_t> (values[i]);
    return static_cast<int32_t> (total);
}

[[nodiscard]] int32_t
ref_squares (const int8_t *const values, const uint32_t count) noexcept
{
    uint32_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<uint32_t> (static_cast<int32_t> (values[i]) * values[i]);
    return static_cast<int32_t> (total);
}

[[nodiscard]] int8_t
ref_max (const int8_t *const values, const uint32_t count) noexcept
{
    int8_t maximum = -128;
    for (uint32_t i = 0; i < count; ++i)
        if (values[i] > maximum)
            maximum = values[i];
    return maximum;
}

/* Lengths that cover the word loop, the 4-byte step, every tail and zero. */
constexpr uint32_t _S_lengths[]
    = { 0u,  1u,  2u,  3u,  4u,  5u,  6u,  7u,  8u,  9u,   12u,  15u,
        16u, 17u, 23u, 24u, 25u, 31u, 32u, 33u, 64u, 65u,  127u, 128u,
        129u, 255u, 256u, 511u, 512u };

void
fill () noexcept
{
    for (uint32_t i = 0; i < _S_cap; ++i)
        {
            _S_lhs[i] = static_cast<int8_t> (rnd () & 0xFFu);
            _S_rhs[i] = static_cast<int8_t> (rnd () & 0xFFu);
            _S_packed[i] = static_cast<uint8_t> (rnd () & 0xFFu);
        }
    /* A few exact-code and saturating patterns on top of the noise. */
    for (uint32_t i = 0; i < 16u; ++i)
        {
            _S_lhs[i] = i < 4u ? -127 : (i < 8u ? 127 : static_cast<int8_t> (i));
            _S_rhs[i] = i % 2u == 0u ? 127 : -127;
        }
}

template <typename Body>
void
for_each_shape (const Body &body) noexcept
{
    for (uint32_t offset = 0; offset < 4u; ++offset)
        for (const uint32_t count : _S_lengths)
            body (offset, count);
}

/* Aligned inputs of four bytes or more take the SXTB16/SMLAD lane loop; every
 * other shape stays on scalar code the model executes. */
[[nodiscard]] bool
lane_case (const uint32_t offset, const uint32_t count) noexcept
{
    if (offset == 0u && count >= 4u && !_S_lane_ok)
        {
            ++_S_skipped_lane;
            return true;
        }
    return false;
}

void
print_caps () noexcept
{
    _S_pos = 0;
    put_text ("CAPS sxtb16=");
    put (_S_sxtb16_ok ? 'y' : 'n');
    put_text (" smlad=");
    put (_S_smlad_ok ? 'y' : 'n');
    put_text (" smlabb=");
    put (zzs_m4_probe_smlabb () != 0 ? 'y' : 'n');
    put_text (" ssat=");
    put (zzs_m4_probe_ssat () != 0 ? 'y' : 'n');
    put_text (" usada8=");
    put (zzs_m4_probe_usada8 () != 0 ? 'y' : 'n');
    put ('\n');
    flush_line ();
}

void
test_dot () noexcept
{
    for_each_shape ([] (const uint32_t offset, const uint32_t count) {
        if (lane_case (offset, count))
            return;
        const int32_t got = zzs_mllee_dot_i8 (_S_lhs + offset, _S_rhs + offset, count);
        const int32_t want = ref_dot (_S_lhs + offset, _S_rhs + offset, count);
        ++_S_checks;
        if (got != want)
            report ("dot_i8", offset, count, want, got);
    });
}

void
test_requant () noexcept
{
    static constexpr int32_t _S_accumulators[]
        = { 0, 1, -1, 127, -127, 16383, 16384, 16385, -16384, 1000000, -1000000,
            2147483647, -2147483648, 12345678, -87654321 };
    static constexpr int32_t _S_multipliers[]
        = { 1, 2, 100, 16384, 16385, 32767, 32768, 65536, -1, -32768, 2147483647 };
    for (const int32_t accumulator : _S_accumulators)
        for (const int32_t multiplier : _S_multipliers)
            {
                const int32_t got = zzs_mllee_requant_q15 (accumulator, multiplier);
                const int32_t want = ref_requant (accumulator, multiplier);
                ++_S_checks;
                if (got != want)
                    report ("requant_q15", static_cast<uint32_t> (accumulator),
                            static_cast<uint32_t> (multiplier), want, got);
            }
}

void
test_unpack () noexcept
{
    for (uint32_t offset = 0; offset < 4u; ++offset)
        for (const uint32_t bytes : _S_lengths)
            {
                zzs_mllee_unpack_i4 (_S_packed + offset, _S_out + offset, bytes);
                for (uint32_t i = 0; i < bytes * 2u; ++i)
                    {
                        const uint8_t source
                            = _S_packed[offset + i / 2u];
                        const int8_t want = i % 2u == 0u
                                                ? ref_from_int4 (source & 0x0Fu)
                                                : ref_from_int4 (source >> 4);
                        ++_S_checks;
                        if (_S_out[offset + i] != want)
                            report ("unpack_i4", offset, bytes, want, _S_out[offset + i]);
                    }
            }
}

void
test_sat (const bool additive) noexcept
{
    for_each_shape ([&] (const uint32_t offset, const uint32_t count) {
        if (additive)
            zzs_mllee_add_sat_i8 (_S_lhs + offset, _S_rhs + offset, _S_out + offset, count);
        else
            zzs_mllee_sub_sat_i8 (_S_lhs + offset, _S_rhs + offset, _S_out + offset, count);
        for (uint32_t i = 0; i < count; ++i)
            {
                const int8_t want = additive
                                        ? ref_add_sat (_S_lhs[offset + i], _S_rhs[offset + i])
                                        : ref_sub_sat (_S_lhs[offset + i], _S_rhs[offset + i]);
                ++_S_checks;
                if (_S_out[offset + i] != want)
                    report (additive ? "add_sat_i8" : "sub_sat_i8", offset, count, want,
                            _S_out[offset + i]);
            }
    });
}

void
test_sum_max () noexcept
{
    for_each_shape ([] (const uint32_t offset, const uint32_t count) {
        const bool skip_lane = lane_case (offset, count);
        if (!skip_lane)
            {
                const int32_t got = zzs_mllee_sum_i8 (_S_lhs + offset, count);
                const int32_t want = ref_sum (_S_lhs + offset, count);
                ++_S_checks;
                if (got != want)
                    report ("sum_i8", offset, count, want, got);
            }

        const int8_t got_max = zzs_mllee_max_i8 (_S_lhs + offset, count);
        const int8_t want_max = ref_max (_S_lhs + offset, count);
        ++_S_checks;
        if (got_max != want_max)
            report ("max_i8", offset, count, want_max, got_max);
    });
}

void
test_stats () noexcept
{
    for_each_shape ([] (const uint32_t offset, const uint32_t count) {
        if (lane_case (offset, count))
            return;
        _S_stats[0][0] = 0x5A5A5A5A;
        _S_stats[0][1] = 0x5A5A5A5A;
        zzs_mllee_stats_i8 (_S_lhs + offset, count, &_S_stats[0][0], &_S_stats[0][1]);
        ++_S_checks;
        if (_S_stats[0][0] != ref_sum (_S_lhs + offset, count))
            report ("stats_i8.sum", offset, count, ref_sum (_S_lhs + offset, count),
                    _S_stats[0][0]);
        ++_S_checks;
        if (_S_stats[0][1] != ref_squares (_S_lhs + offset, count))
            report ("stats_i8.squares", offset, count, ref_squares (_S_lhs + offset, count),
                    _S_stats[0][1]);
    });
}
} // namespace

extern "C" void *
memset (void *const destination, const int value, const uint32_t length) noexcept
{
    unsigned char *const bytes = static_cast<unsigned char *> (destination);
    for (uint32_t i = 0; i < length; ++i)
        bytes[i] = static_cast<unsigned char> (value);
    return destination;
}

extern "C" void *
memcpy (void *const destination, const void *const source,
        const uint32_t length) noexcept
{
    unsigned char *const to = static_cast<unsigned char *> (destination);
    const unsigned char *const from = static_cast<const unsigned char *> (source);
    for (uint32_t i = 0; i < length; ++i)
        to[i] = from[i];
    return destination;
}

/* --- Training lane kernels: the portable fallbacks' contract, recomputed here
       with the same int64 accumulators, so a divergence is a kernel bug rather
       than a difference of convention. */

[[nodiscard]] int16_t
ref_lane (const int32_t value) noexcept
{
    return static_cast<int16_t> (value > 32767 ? 32767
                                               : (value < -32768 ? -32768 : value));
}

[[nodiscard]] uint32_t
ref_add_grad (int16_t *const destination, const int16_t *const source,
              const uint32_t count) noexcept
{
    uint32_t saturated = 0u;
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t sum = destination[i] + source[i];
            if (sum > 32767 || sum < -32768)
                {
                    destination[i] = ref_lane (sum);
                    ++saturated;
                }
            else
                destination[i] = static_cast<int16_t> (sum);
        }
    return saturated;
}

[[nodiscard]] int32_t
ref_dot_q15 (const int16_t *const lhs, const int16_t *const rhs,
             const uint32_t count) noexcept
{
    int64_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<int64_t> (lhs[i]) * rhs[i];
    const int64_t shifted = (total + (1LL << 14)) >> 15;
    if (shifted > 2147483647LL)
        return 2147483647;
    if (shifted < -2147483648LL)
        return -2147483647 - 1;
    return static_cast<int32_t> (shifted);
}

void
ref_axpy (int16_t *const destination, const int16_t *const source,
          const int32_t scalar_q15, const uint32_t count) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        {
            const int64_t product = static_cast<int64_t> (source[i]) * scalar_q15;
            destination[i] = ref_lane (destination[i]
                                       + static_cast<int32_t> (
                                             (product + (1LL << 14)) >> 15));
        }
}

void
ref_scale (int16_t *const values, const int32_t scalar_q15,
           const uint32_t count) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        {
            const int64_t product = static_cast<int64_t> (values[i]) * scalar_q15;
            const int64_t shifted = (product + (1LL << 14)) >> 15;
            /* The wrapper clamps the int64 here and truncates it in axpy; the
               reference keeps that asymmetry so a kernel that quietly folds
               differently shows up as a divergence. */
            values[i] = static_cast<int16_t> (
                shifted > 32767 ? 32767 : (shifted < -32768 ? -32768 : shifted));
        }
}

void
ref_relu_bwd (const int16_t *const grad_out, const int8_t *const input,
              int16_t *const grad_in, const uint32_t count) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        grad_in[i] = input[i] >= 0 ? grad_out[i] : 0;
}

[[nodiscard]] int32_t
ref_sum_abs_q15 (const int16_t *const values, const uint32_t count) noexcept
{
    int64_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += values[i] < 0 ? -static_cast<int64_t> (values[i]) : values[i];
    return total > 2147483647LL ? 2147483647 : static_cast<int32_t> (total);
}

[[nodiscard]] int32_t
ref_sum_sq_q15 (const int16_t *const values, const uint32_t count) noexcept
{
    int64_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        {
            total += static_cast<int64_t> (values[i]) * values[i];
            if (total > 2147483647LL)
                return 2147483647;
        }
    return static_cast<int32_t> (total);
}

/* The reference divides on its own: a nostdlib image has no __aeabi_ldivmod, and
   linking libgcc for it would put the answer inside the code under test. */
[[nodiscard]] uint64_t
ref_udiv (uint64_t dividend, const uint32_t divisor) noexcept
{
    uint32_t top = 64u;
    while (top != 0u && ((dividend >> (top - 1u)) & 1u) == 0u)
        --top;
    uint64_t quotient = 0;
    uint64_t remainder = 0;
    for (uint32_t bit = top; bit-- > 0;)
        {
            remainder = (remainder << 1) | ((dividend >> bit) & 1u);
            quotient <<= 1;
            if (remainder >= divisor)
                {
                    remainder -= divisor;
                    quotient |= 1u;
                }
        }
    return quotient;
}

[[nodiscard]] int64_t
ref_div_trunc (const int64_t value, const uint32_t divisor) noexcept
{
    const bool negative = value < 0;
    /* 0 - |x| in unsigned arithmetic, so INT64_MIN has no special case. */
    const uint64_t magnitude
        = negative ? 0ull - static_cast<uint64_t> (value)
                   : static_cast<uint64_t> (value);
    const uint64_t quotient = ref_udiv (magnitude, divisor);
    return negative ? -static_cast<int64_t> (quotient)
                    : static_cast<int64_t> (quotient);
}

void
ref_sgd_step (int8_t *const parameters, const int16_t *const grad,
              const int32_t lr_q15, const uint32_t count) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        {
            const int64_t product = static_cast<int64_t> (grad[i]) * lr_q15;
            const int64_t code_step = (product + (1LL << 14)) >> 15;
            const int32_t updated
                = static_cast<int32_t> (parameters[i])
                  + static_cast<int32_t> (ref_div_trunc (
                        code_step >= 0 ? code_step + 129 : code_step - 129, 258u));
            parameters[i] = ref_code (updated);
        }
}

void
ref_codes_to_q15 (const int8_t *const codes, int16_t *const values,
                  const uint32_t count) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        values[i] = static_cast<int16_t> (codes[i] * 258);
}

void
ref_q15_to_codes (const int16_t *const values, int8_t *const codes,
                  const uint32_t count) noexcept
{
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t value = values[i];
            codes[i] = ref_code (value >= 0 ? (value * 127 + 16384) >> 15
                                            : -(((-value) * 127 + 16384) >> 15));
        }
}

void
fill_lanes () noexcept
{
    for (uint32_t i = 0; i < _S_cap; ++i)
        {
            _S_lane[i] = static_cast<int16_t> (rnd () & 0xFFFFu);
            _S_lane_b[i] = static_cast<int16_t> (rnd () & 0xFFFFu);
            _S_code[i] = static_cast<int8_t> (rnd () & 0xFFu);
        }
    /* Rails and the exact values the rounding rules turn on. */
    static constexpr int16_t _S_rails[]
        = { 0, 1, -1, 127, -127, 128, -128, 129, 258, -258, 16383, 16384,
            16385, -16384, -16385, 32767, -32768, 32766, -32767, 8192 };
    for (uint32_t i = 0; i < 40u; ++i)
        {
            _S_lane[i] = _S_rails[i % 20u];
            _S_lane_b[i] = _S_rails[(i + 7u) % 20u];
            _S_code[i] = static_cast<int8_t> (
                i % 20u < 10u ? -127 + static_cast<int32_t> (i % 10u)
                              : 127 - static_cast<int32_t> (i % 10u));
        }
}

void
test_train_lanes () noexcept
{
    static constexpr int32_t _S_scalars[]
        = { 0, 1, -1, 258, 1638, 16384, -16384, 32767, -32768, 65536, -65536 };

    for_each_shape ([&] (const uint32_t offset, const uint32_t count) {
        /* The reference buffers are copies of the same window the kernel sees,
           at the same indices: a copy that stops at count would leave the tail of
           an offset window holding last case's result and report a false match. */
        for (uint32_t i = 0; i < count; ++i)
            {
                _S_lane_ref[offset + i] = _S_lane[offset + i];
                _S_lane_bref[offset + i] = _S_lane_b[offset + i];
            }

        const uint32_t got_saturated
            = zzs_mllee_train_add_grad_q15 (_S_lane + offset, _S_lane_b + offset,
                                            count);
        const uint32_t want_saturated
            = ref_add_grad (_S_lane_ref + offset, _S_lane_bref + offset, count);
        ++_S_checks;
        if (got_saturated != want_saturated)
            report ("train_add_grad_count", offset, count,
                    static_cast<int32_t> (want_saturated),
                    static_cast<int32_t> (got_saturated));
        /* The lanes too, not just the count: these buffers feed every case below,
           so a wrong value here otherwise shows up as a dozen unrelated
           failures. */
        for (uint32_t i = 0; i < count; ++i)
            {
                ++_S_checks;
                if (_S_lane[offset + i] != _S_lane_ref[offset + i])
                    report ("train_add_grad_lane", offset, count,
                            _S_lane_ref[offset + i], _S_lane[offset + i]);
            }

        const int32_t got_dot
            = zzs_mllee_train_dot_q15 (_S_lane + offset, _S_lane_b + offset, count);
        const int32_t want_dot
            = ref_dot_q15 (_S_lane_ref + offset, _S_lane_bref + offset, count);
        ++_S_checks;
        if (got_dot != want_dot)
            report ("train_dot_q15", offset, count, want_dot, got_dot);

        const int32_t got_abs
            = zzs_mllee_train_sum_abs_q15 (_S_lane + offset, count);
        const int32_t want_abs = ref_sum_abs_q15 (_S_lane_ref + offset, count);
        ++_S_checks;
        if (got_abs != want_abs)
            report ("train_sum_abs_q15", offset, count, want_abs, got_abs);

        const int32_t got_sq = zzs_mllee_train_sum_sq_q15 (_S_lane + offset, count);
        const int32_t want_sq = ref_sum_sq_q15 (_S_lane_ref + offset, count);
        ++_S_checks;
        if (got_sq != want_sq)
            report ("train_sum_sq_q15", offset, count, want_sq, got_sq);

        zzs_mllee_train_relu_bwd (_S_lane + offset, _S_code + offset,
                                  _S_lane_b + offset, count);
        ref_relu_bwd (_S_lane_ref + offset, _S_code + offset, _S_lane_bref + offset,
                      count);
        for (uint32_t i = 0; i < count; ++i)
            {
                ++_S_checks;
                if (_S_lane_b[offset + i] != _S_lane_bref[offset + i])
                    report ("train_relu_bwd", offset, count, _S_lane_bref[offset + i],
                            _S_lane_b[offset + i]);
            }

        for (const int32_t scalar : _S_scalars)
            {
                zzs_mllee_train_scale_q15 (_S_lane + offset, scalar, count);
                ref_scale (_S_lane_ref + offset, scalar, count);
                for (uint32_t i = 0; i < count; ++i)
                    {
                        ++_S_checks;
                        if (_S_lane[offset + i] != _S_lane_ref[offset + i])
                            report ("train_scale_q15", offset, count,
                                    _S_lane_ref[offset + i], _S_lane[offset + i]);
                    }

                zzs_mllee_train_axpy_q15 (_S_lane + offset, _S_lane_b + offset,
                                          scalar, count);
                ref_axpy (_S_lane_ref + offset, _S_lane_bref + offset, scalar, count);
                for (uint32_t i = 0; i < count; ++i)
                    {
                        ++_S_checks;
                        if (_S_lane[offset + i] != _S_lane_ref[offset + i])
                            report ("train_axpy_q15", offset, count,
                                    _S_lane_ref[offset + i], _S_lane[offset + i]);
                    }

                for (uint32_t i = 0; i < count; ++i)
                    {
                        _S_code_ref[offset + i] = _S_code[offset + i];
                        _S_grad[offset + i] = _S_lane[offset + i];
                    }
                zzs_mllee_train_sgd_step_i8 (_S_code + offset, _S_grad + offset,
                                             scalar, count);
                ref_sgd_step (_S_code_ref + offset, _S_grad + offset, scalar, count);
                for (uint32_t i = 0; i < count; ++i)
                    {
                        ++_S_checks;
                        if (_S_code[offset + i] != _S_code_ref[offset + i])
                            report ("train_sgd_step_i8", offset, count,
                                    _S_code_ref[offset + i], _S_code[offset + i]);
                    }
            }

        for (uint32_t i = 0; i < count; ++i)
            _S_lane_ref[offset + i] = 0;
        zzs_mllee_train_codes_to_q15 (_S_code + offset, _S_lane + offset, count);
        ref_codes_to_q15 (_S_code + offset, _S_lane_ref + offset, count);
        for (uint32_t i = 0; i < count; ++i)
            {
                ++_S_checks;
                if (_S_lane[offset + i] != _S_lane_ref[offset + i])
                    report ("train_codes_to_q15", offset, count,
                            _S_lane_ref[offset + i], _S_lane[offset + i]);
            }

        for (uint32_t i = 0; i < count; ++i)
            _S_code_ref[offset + i] = 0;
        zzs_mllee_train_q15_to_codes (_S_lane + offset, _S_code + offset, count);
        ref_q15_to_codes (_S_lane + offset, _S_code_ref + offset, count);
        for (uint32_t i = 0; i < count; ++i)
            {
                ++_S_checks;
                if (_S_code[offset + i] != _S_code_ref[offset + i])
                    report ("train_q15_to_codes", offset, count,
                            _S_code_ref[offset + i], _S_code[offset + i]);
            }
    });
}

extern "C" int
main ()
{
    _S_sxtb16_ok = zzs_m4_probe_sxtb16 () != 0;
    _S_smlad_ok = zzs_m4_probe_smlad () != 0;
    _S_lane_ok = _S_sxtb16_ok && _S_smlad_ok;
    print_caps ();

    fill ();
    test_dot ();
    test_requant ();
    test_unpack ();
    test_sat (true);
    test_sat (false);
    test_sum_max ();
    test_stats ();
    fill_lanes ();
    test_train_lanes ();

    _S_pos = 0;
    put_text ("M4-SELFTEST: ");
    put_text (_S_failures == 0u ? "PASS" : "FAIL");
    put_text (" checks=");
    put_dec (static_cast<int32_t> (_S_checks));
    put_text (" failures=");
    put_dec (static_cast<int32_t> (_S_failures));
    put_text (" lane_cases_skipped=");
    put_dec (static_cast<int32_t> (_S_skipped_lane));
    put ('\n');
    flush_line ();
    if (_S_failures == 0u && _S_skipped_lane != 0u)
        emit ("M4-NOTE: SXTB16 lane loops not executable on this model;"
              " verified against ARMv7E-M semantics by inspection\n");
    return _S_failures == 0u ? 0 : 1;
}
