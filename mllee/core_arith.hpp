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

#include <cstdint>

/**
 * @section Integer division by an invariant
 *
 * Several kernels divide a whole row by the same small denominator: the norm
 * deviation, the softmax sum, the attention denominator, an average window.
 * A hardware divide is a multi-cycle variable-latency operation on every
 * target this module supports (20-40 cycles on Cortex-M4, 20+ on x86-64), and
 * the old loops paid it once per element.  The row is instead divided by one
 * reciprocal at the top of the loop and each element costs a 32x32->64
 * multiply, a shift and at most one branch-free correction.
 *
 * The reciprocal is the round-up magic `ceil(2^32 / d)`.  For `n < 2^32` the
 * product is at most one quotient above the exact value, so a single
 * underflow correction makes the result bit-identical to `/`.  The correction
 * is written as an unsigned wrap test, so it compiles to a compare plus a
 * conditional move (or a `sbb`, on x86) rather than a second multiply.
 */
namespace zzs::mllee
{
struct _S_Div32
{
    uint32_t _M_magic;
    uint32_t _M_divisor;
};

[[nodiscard]] inline constexpr _S_Div32
_S_div32_make (const uint32_t divisor) noexcept
{
    if (divisor <= 1u)
        return _S_Div32{ 0u, divisor };
    return _S_Div32{ static_cast<uint32_t> (((1ull << 32) + divisor - 1u) / divisor),
                     divisor };
}

/** @brief High 32 bits of an unsigned 32x32 product. */
[[nodiscard]] inline constexpr uint32_t
_S_mulhi_u32 (const uint32_t lhs, const uint32_t rhs) noexcept
{
#if defined(__SIZEOF_INT128__)
    return static_cast<uint32_t> (
        (static_cast<unsigned __int128> (lhs) * rhs) >> 32);
#else
    return static_cast<uint32_t> ((static_cast<uint64_t> (lhs) * rhs) >> 32);
#endif
}

/** @brief Exact `numerator / divisor` for the reciprocal built by `_S_div32_make`. */
[[nodiscard]] inline constexpr uint32_t
_S_div32_u (const uint32_t numerator, const _S_Div32 reciprocal) noexcept
{
    if (reciprocal._M_divisor <= 1u)
        return numerator;
    const uint32_t quotient = _S_mulhi_u32 (numerator, reciprocal._M_magic);
    const uint32_t remainder = numerator - quotient * reciprocal._M_divisor;
    /* A quotient that overshot by one wraps `remainder`; the wrap is visible as
       `remainder > numerator` because the true remainder never exceeds it. */
    const uint32_t overshot = remainder > numerator ? 1u : 0u;
    return quotient - overshot;
}

/** @brief Truncating signed division (`/` semantics) through the same reciprocal. */
[[nodiscard]] inline constexpr int32_t
_S_div32_i (const int32_t numerator, const _S_Div32 reciprocal) noexcept
{
    if (numerator >= 0)
        return static_cast<int32_t> (_S_div32_u (static_cast<uint32_t> (numerator), reciprocal));
    return -static_cast<int32_t> (
        _S_div32_u (0u - static_cast<uint32_t> (numerator), reciprocal));
}

/** @brief Exact power-of-two division, kept as a shift after a compile-time check. */
[[nodiscard]] inline constexpr uint32_t
_S_div_pow2_u (const uint32_t value, const uint32_t shift) noexcept
{
    return value >> shift;
}

/**
 * @brief Bulk byte copies and fills.
 *
 * FLATTEN, SLICE, CONCAT, PAD1D and the arena zeroing path are pure byte
 * moves.  GCC/Clang lower `__builtin_memcpy`/`__builtin_memset` to one
 * `rep movsb`/`rep stosb` (or an inline vector move for small constants), so a
 * whole tensor costs one libc-independent primitive instead of a per-byte
 * load/store loop that the size-optimized build will not vectorize.
 */
inline void
_S_copy_bytes (void *const destination, const void *const source,
               const uint32_t count) noexcept
{
    if (count == 0u)
        return;
#if defined(__GNUC__) || defined(__clang__)
    __builtin_memcpy (destination, source, static_cast<unsigned long> (count));
#else
    auto *to = static_cast<uint8_t *> (destination);
    const auto *from = static_cast<const uint8_t *> (source);
    for (uint32_t i = 0; i < count; ++i)
        to[i] = from[i];
#endif
}

inline void
_S_fill_bytes (void *const destination, const uint8_t value,
               const uint32_t count) noexcept
{
    if (count == 0u)
        return;
#if defined(__GNUC__) || defined(__clang__)
    __builtin_memset (destination, value, static_cast<unsigned long> (count));
#else
    auto *to = static_cast<uint8_t *> (destination);
    for (uint32_t i = 0; i < count; ++i)
        to[i] = value;
#endif
}

[[nodiscard]] inline constexpr int32_t
_S_abs_i32 (const int32_t value) noexcept
{
    return value < 0 ? -value : value;
}

[[nodiscard]] inline constexpr int32_t
_S_clamp_i32 (const int32_t value, const int32_t low, const int32_t high) noexcept
{
    return value < low ? low : (value > high ? high : value);
}

/** @brief Branch-free min/max on signed and unsigned words. */
[[nodiscard]] inline constexpr int32_t
_S_min_i32 (const int32_t lhs, const int32_t rhs) noexcept
{
    return rhs < lhs ? rhs : lhs;
}

[[nodiscard]] inline constexpr int32_t
_S_max_i32 (const int32_t lhs, const int32_t rhs) noexcept
{
    return lhs < rhs ? rhs : lhs;
}

[[nodiscard]] inline constexpr uint32_t
_S_min_u32 (const uint32_t lhs, const uint32_t rhs) noexcept
{
    return rhs < lhs ? rhs : lhs;
}

[[nodiscard]] inline constexpr uint32_t
_S_max_u32 (const uint32_t lhs, const uint32_t rhs) noexcept
{
    return lhs < rhs ? rhs : lhs;
}

/** @brief floor(log2(value)); 0 for `value <= 1`. */
[[nodiscard]] inline constexpr uint32_t
_S_log2_u32 (const uint32_t value) noexcept
{
#if defined(__has_builtin) && __has_builtin(__builtin_clz)
    if !consteval {
        if (value > 1u)
            return 31u - static_cast<uint32_t> (__builtin_clz (value));
        return 0u;
    }
#endif
    uint32_t bits = 0;
    while ((value >> bits) > 1u)
        ++bits;
    return bits;
}

/** @brief ceil(log2(value)); 0 for `value <= 1`. */
[[nodiscard]] inline constexpr uint32_t
_S_ceil_log2_u32 (const uint32_t value) noexcept
{
    if (value <= 1u)
        return 0u;
    return _S_log2_u32 (value - 1u) + 1u;
}

/** @brief log2 of a power-of-two constant, via ctz where the compiler has it. */
[[nodiscard]] inline constexpr uint32_t
_S_ctz_u32 (const uint32_t value) noexcept
{
#if defined(__has_builtin) && __has_builtin(__builtin_ctz)
    if !consteval {
        if (value != 0u)
            return static_cast<uint32_t> (__builtin_ctz (value));
        return 32u;
    }
#endif
    uint32_t count = 0;
    uint32_t probe = value;
    while ((probe & 1u) == 0u && probe != 0u)
        {
            probe >>= 1;
            ++count;
        }
    return count;
}

[[nodiscard]] inline constexpr uint32_t
_S_popcount_u32 (const uint32_t value) noexcept
{
#if defined(__has_builtin) && __has_builtin(__builtin_popcount)
    if !consteval {
        return static_cast<uint32_t> (__builtin_popcount (value));
    }
#endif
    uint32_t count = 0;
    for (uint32_t probe = value; probe != 0u; probe >>= 1)
        count += probe & 1u;
    return count;
}

/**
 * @brief High half of a signed 64x64 product where the compiler exposes it.
 *
 * GCC only provides `__builtin_smulh` on targets with a widening signed high
 * multiply (the Arm/AArch64 family).  On other hosts the 128-bit fallback is a
 * single `imul` when the compiler has a 128-bit type, and the scalar shift
 * form is the last resort.  No caller in this module depends on the builtin
 * being present: every use has a value-level equivalent.
 */
[[nodiscard]] inline constexpr int64_t
_S_mulhi_s64 (const int64_t lhs, const int64_t rhs) noexcept
{
#if defined(__has_builtin) && __has_builtin(__builtin_smulh)
    if !consteval {
        return static_cast<int64_t> (__builtin_smulh (lhs, rhs));
    }
#endif
#if defined(__SIZEOF_INT128__)
    return static_cast<int64_t> (
        (static_cast<__int128> (lhs) * static_cast<__int128> (rhs)) >> 64);
#else
    const uint64_t a = static_cast<uint64_t> (lhs);
    const uint64_t b = static_cast<uint64_t> (rhs);
    const uint64_t a_lo = a & 0xFFFFFFFFu;
    const uint64_t a_hi = a >> 32;
    const uint64_t b_lo = b & 0xFFFFFFFFu;
    const uint64_t b_hi = b >> 32;
    const uint64_t lo_lo = a_lo * b_lo;
    const uint64_t hi_lo = a_hi * b_lo;
    const uint64_t lo_hi = a_lo * b_hi;
    const uint64_t hi_hi = a_hi * b_hi;
    const uint64_t cross = (lo_lo >> 32) + (hi_lo & 0xFFFFFFFFu) + (lo_hi & 0xFFFFFFFFu);
    return static_cast<int64_t> (hi_hi + (hi_lo >> 32) + (lo_hi >> 32) + (cross >> 32));
#endif
}
} // namespace zzs::mllee
