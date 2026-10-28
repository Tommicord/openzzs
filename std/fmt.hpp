/*
 * Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

#pragma once

#include "std/string_view.hpp"

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>

#if defined(__aarch64__) || defined(__ARM64)
#define ZZS_IS_AARCH64 1
#define ZZS_IS_X64 0
#elif defined(__x86_64__) || defined(__X64)
#define ZZS_IS_AARCH64 0
#define ZZS_IS_X64 1
#else
#define ZZS_IS_AARCH64 0
#define ZZS_IS_X64 0
#endif

#ifndef ZZS_USE_NEON
#if ZZS_IS_AARCH64 && (defined(__ARM_NEON__) || defined(__ARM_NEON))
#include <arm_neon.h>
#define ZZS_USE_NEON 1
#else
#define ZZS_USE_NEON 0
#endif
#endif

#ifndef ZZS_USE_SSE2
#if defined(__SSE2__) && __SSE2__
#include <immintrin.h>
#define ZZS_USE_SSE2 1
#else
#define ZZS_USE_SSE2 0
#endif
#endif

#if ZZS_USE_SSE2 || ZZS_USE_NEON
#define ZZS_USE_NEON_OR_SSE2 1
#else
#define ZZS_USE_NEON_OR_SSE2 0
#endif

#ifndef ZZS_USE_SSE3
#if ZZS_USE_SSE2 && defined(__SSE3__) && __SSE3__
#define ZZS_USE_SSE3 1
#else
#define ZZS_USE_SSE3 0
#endif
#endif

#ifndef ZZS_USE_SSE4_1
#if ZZS_USE_SSE3 && defined(__SSE4_1__) && __SSE4_1__
#define ZZS_USE_SSE4_1 1
#else
#define ZZS_USE_SSE4_1 0
#endif
#endif

#ifndef ZZS_NO_MEMMOVE
#if ZZS_USE_SSE2 && (defined(__AVX512F__) && defined(__AVX512IFMA__) && defined(__AVX512VBMI__))
#define ZZS_NO_MEMMOVE 1
#elif ZZS_USE_SSE2 && (ZZS_USE_SSE4_1 || (defined(__SSSE3__) && __SSSE3__))
#define ZZS_NO_MEMMOVE 1
#elif ZZS_USE_NEON && defined(__APPLE__) && defined(__arm64__)
#define ZZS_NO_MEMMOVE 0
#elif ZZS_USE_NEON
#define ZZS_NO_MEMMOVE 1
#else
#define ZZS_NO_MEMMOVE 0
#endif
#endif

#if ZZS_NO_MEMMOVE
#define ZZS_SHUFFLER_PARAM , move_shuffler
#define ZZS_SHUFFLER_DECL , const uint8_t *move_shuffler
#else
#define ZZS_SHUFFLER_PARAM
#define ZZS_SHUFFLER_DECL
#endif

#if ZZS_NO_MEMMOVE || ZZS_USE_NEON
#define ZZS_NOT_REMOVE_FIRST_ZERO 1
#else
#define ZZS_NOT_REMOVE_FIRST_ZERO 0
#endif

namespace zzs::base {
enum class NumberBase : unsigned int { binary = 2, octal = 8, decimal = 10, hexadecimal = 16 };

template <typename> struct _S_FloatingStringWidth {};

// Floating-point types: based on __FLT/DBL/LDBL macros
template <> struct _S_FloatingStringWidth<float> {
    // Maximum decimal string length for float
    // Format: _M_sign + max significand digits + decimal point + exponent marker + exponent
    // __FLT_MANT_DIG__ = 24 (bits), so ~7 decimal digits
    // __FLT_MAX_10_EXP__ = 38
    // Worst case: "-1.234567e+38" = 1 + 7 + 1 + 1 + 2 + 2 = 14
    // Using a conservative value to handle various formats
    static constexpr unsigned int size = 32;
};
template <> struct _S_FloatingStringWidth<double> {
    // Maximum decimal string length for double
    // __DBL_MANT_DIG__ = 53 (bits), so ~15-16 decimal digits
    // __DBL_MAX_10_EXP__ = 308
    // Worst case: "-1.234567890123456e+308" = 1 + 16 + 1 + 1 + 3 + 3 = 25
    // Using a conservative value to handle various formats
    static constexpr unsigned int size = 64;
};
template <> struct _S_FloatingStringWidth<long double> {
    // Maximum decimal string length for long double
    // __LDBL_MANT_DIG__ = 64 (bits on x86), so ~19-20 decimal digits
    // __LDBL_MAX_10_EXP__ = 4932 (on x86 extended precision)
    // Worst case: "-1.234567890123456789e+4932" = 1 + 19 + 1 + 1 + 4 + 4 = 30
    // Using a conservative value to handle various formats
    static constexpr unsigned int size = 128;
};
#if defined(__SIZEOF_INT128__)
using uint128_t = __uint128_t;
#endif
constexpr int
_S_u64_lz_bits (uint64_t x) {
#if defined(__has_builtin) && __has_builtin(__builtin_clzll)
    if !consteval {
        return __builtin_clzll (x);
    }
#endif
    int n = 64;
    for (; x > 0; x >>= 1)
        --n;
    return n;
}

constexpr int
_S_u32_lz_bits (uint32_t x) {
#if defined(__has_builtin) && __has_builtin(__builtin_clz)
    if !consteval {
        return __builtin_clz (x);
    }
#endif
    int n = 32;
    for (; x > 0; x >>= 1)
        --n;
    return n;
}

constexpr int
_S_u64_tz_bits (uint64_t x) {
#if defined(__has_builtin) && __has_builtin(__builtin_ctzll)
    if !consteval {
        return __builtin_ctzll (x);
    }
#endif
    int n = 64;
    for (; x > 0; x <<= 1)
        --n;
    return n;
}

constexpr uint64_t
_S_umul128_hi64_fallback (uint64_t x, uint64_t y) {
    uint64_t a = x >> 32;
    uint64_t b = static_cast<uint32_t> (x);
    uint64_t c = y >> 32;
    uint64_t d = static_cast<uint32_t> (y);

    uint64_t ac = a * c;
    uint64_t bc = b * c;
    uint64_t ad = a * d;
    uint64_t bd = b * d;

    uint64_t cs = (bd >> 32) + static_cast<uint32_t> (ad) + static_cast<uint32_t> (bc);
    return ac + (ad >> 32) + (bc >> 32) + (cs >> 32);
}

constexpr void
_S_umul128_hi64_lo64_fallback (uint64_t x, uint64_t y, uint64_t &hi, uint64_t &lo) {
    uint64_t a = x >> 32;
    uint64_t b = static_cast<uint32_t> (x);
    uint64_t c = y >> 32;
    uint64_t d = static_cast<uint32_t> (y);

    uint64_t ac = a * c;
    uint64_t bc = b * c;
    uint64_t ad = a * d;
    uint64_t bd = b * d;

    uint64_t cs = (bd >> 32) + static_cast<uint32_t> (ad) + static_cast<uint32_t> (bc);
    hi = ac + (ad >> 32) + (bc >> 32) + (cs >> 32);
    lo = (cs << 32) + static_cast<uint32_t> (bd);
}

constexpr void
_S_umul128_u64_hi128 (uint64_t a_high, uint64_t a_low, uint64_t b, uint64_t *result_high, uint64_t *result_mid) {
    uint64_t low_high, low_low;
    uint64_t high_high, high_low;
    if !consteval {
#if defined(_MSC_VER) && defined(__X64)
        low_low = _umul128 (a_low, b, &low_high);
        high_low = _umul128 (a_high, b, &high_high);
        unsigned char carry1 = _addcarry_u64 (0, low_high, high_low, result_mid);
        _addcarry_u64 (carry1, high_high, 0, result_high);
#elif defined(_MSC_VER) && defined(__ARM64)
        low_high = __umulh (a_low, b);
        high_high = __umulh (a_high, b);
        high_low = a_high * b;
        *result_mid = low_high + high_low;
        *result_high = high_high + (*result_mid < low_high);
#elif defined(__SIZEOF_INT128__)
        uint128_t hi128 = (static_cast<uint128_t> (b) * a_high + ((static_cast<uint128_t> (b) * a_low) >> 64));
        *result_high = hi128 >> 64;
        *result_mid = static_cast<uint64_t> (hi128);
#else
        _S_umul128_hi64_lo64_fallback (a_low, b, low_high, low_low);
        _S_umul128_hi64_lo64_fallback (a_high, b, high_high, high_low);
        *result_mid = low_high + high_low;
        *result_high = high_high + (*result_mid < low_high);
#endif
    } else {
        _S_umul128_hi64_lo64_fallback (a_low, b, low_high, low_low);
        _S_umul128_hi64_lo64_fallback (a_high, b, high_high, high_low);
        *result_mid = low_high + high_low;
        *result_high = high_high + (*result_mid < low_high);
    }
}

constexpr uint64_t
_S_umul128_hi64_xjb (uint64_t a, uint64_t b) {
    if !consteval {
#if defined(_MSC_VER) && (defined(__X64) || defined(__ARM64))
        return __umulh (a, b);
#elif defined(__SIZEOF_INT128__)
        return (static_cast<uint128_t> (a) * b) >> 64;
#endif
    }
    return _S_umul128_hi64_fallback (a, b);
}

constexpr uint64_t
_S_u128_madd_hi64 (uint64_t a, uint64_t b, uint64_t c) {
    if !consteval {
#if defined(_MSC_VER) && defined(__X64)
        uint64_t hi, lo, null_value;
        lo = _umul128 (a, b, &hi);
        unsigned char carry1 = _addcarry_u64 (0, lo, c, &null_value);
        return hi + carry1;
#elif defined(_MSC_VER) && defined(__ARM64)
        uint64_t hi = __umulh (a, b);
        uint64_t lo = a * b;
        return hi + ((lo + c) < c);
#elif defined(__SIZEOF_INT128__)
        return (static_cast<uint128_t> (a) * b + c) >> 64;
#endif
    }
    uint64_t hi, lo;
    _S_umul128_hi64_lo64_fallback (a, b, hi, lo);
    return hi + ((lo + c) < c);
}

constexpr uint64_t
_S_byteswap64_xjb (uint64_t x) {
    if !consteval {
#if defined(__has_builtin) && __has_builtin(__builtin_bswap64)
        return __builtin_bswap64 (x);
#elif defined(_MSC_VER)
        return _byteswap_uint64 (x);
#endif
    }
    return ((x & 0xff00000000000000) >> 56) | ((x & 0x00ff000000000000) >> 40) | ((x & 0x0000ff0000000000) >> 24)
           | ((x & 0x000000ff00000000) >> 8) | ((x & 0x00000000ff000000) << 8) | ((x & 0x0000000000ff0000) << 24)
           | ((x & 0x000000000000ff00) << 40) | ((x & 0x00000000000000ff) << 56);
}

constexpr uint64_t
_S_is_little_endian () {
    if !consteval {
        const int n = 1;
        return *reinterpret_cast<const char *> (&n) == 1;
    } else {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        return 0;
#else
        return 1;
#endif
    }
}

constexpr uint64_t
_S_cmov_branchless (const uint64_t condition, const uint64_t true_value, const uint64_t false_value) {
#if !XJB_IS_REAL_GCC || !defined(__amd64__)
    return condition ? true_value : false_value;
#else
    asm volatile ("test %2, %2\n\t"
                  "cmovne %1, %0\n\t"
                  : "+r"(false_value)
                  : "r"(true_value), "r"(condition)
                  : "cc");
    return false_value;
#endif
}

template <typename T>
constexpr T *
_S_constexpr_memcpy (T *dest, const T *src, size_t count) {
    for (size_t i = 0; i < count; ++i)
        dest[i] = src[i];
    return dest;
}

struct ShortestAscii16 {
#if ZZS_USE_NEON
    uint64x2_t ascii16;
#elif ZZS_USE_SSE2
    __m128i ascii16;
#else
    uint64_t hi;
    uint64_t lo;
#endif
    uint64_t dec_sig_len_sub1;
#if ZZS_NO_MEMMOVE
    int32_t trailing_byte;
#endif
};
struct ShortestAscii {
    uint64_t ascii;
    uint64_t dec_sig_len_sub1;
};

struct _S_DoubleTable {
    static constexpr int e10_DN = -4;
    static constexpr int e10_UP = 15;
    static constexpr int max_dec_sig_len = 17;
    static constexpr int num_pow10 = 323 - (-293) + 1;
    static constexpr int max_first_sig_pos = 5;
    static constexpr int max_one_offset = 17;
    static constexpr int max_buffer_requirement = ZZS_NO_MEMMOVE ? 33 : 34;
    static constexpr int max_valid_output_len = 24;

    uint64_t pow10_double[num_pow10 * 2];
    uint64_t exp_result_double[324 + 308 + 1];
    alignas (64) unsigned char e10_variable_data[e10_UP - e10_DN + 1 + 1][ZZS_NO_MEMMOVE ? 64 : 32];
    unsigned char h7[2048];

    consteval _S_DoubleTable () : pow10_double{}, exp_result_double{}, e10_variable_data{}, h7{} {
        struct uint192 {
            uint64_t w0, w1, w2;
        };
        uint192 current = {0xb2e28cedd086d011, 0x1e53ed49a96272c8, 0xcc5fc196fefd7d0c};
        constexpr uint64_t ten = 0xa000000000000000;

        for (int i = 0; i < num_pow10; ++i) {
            int e10 = i - 293;
            this->pow10_double[(num_pow10 - 1 - i) * 2 + 0] =
                e10 == 0 ? 1ull << 63 : current.w2 + (e10 >= 0 && e10 <= 27);
            this->pow10_double[(num_pow10 - 1 - i) * 2 + 1] = current.w1 + 1;

            uint64_t h0 = _S_umul128_hi64_fallback (current.w0, ten);
            uint64_t h1 = _S_umul128_hi64_fallback (current.w1, ten);
            uint64_t c0 = h0 + current.w1 * ten;
            uint64_t c1 = (c0 < h0) + h1 + current.w2 * ten;
            uint64_t c2 = (c1 < h1) + _S_umul128_hi64_fallback (current.w2, ten);
            if (c2 >> 63)
                current = {c0, c1, c2};
            else
                current = {c0 << 1, c1 << 1 | c0 >> 63, c2 << 1 | c1 >> 63};
        }

        for (int e10 = -324; e10 <= 308; ++e10) {
            uint64_t e = e10 < 0 ? ('e' + '-' * 256) : 'e' + '+' * 256;
            uint64_t e10_abs = e10 < 0 ? -e10 : e10;
            uint64_t a = e10_abs / 100;
            uint64_t bc = e10_abs - a * 100;
            uint64_t b = bc / 10;
            uint64_t c = bc - b * 10;
            uint64_t exp_len = 4 + (e10_abs >= 100);
            uint64_t e10_abs_ascii =
                (e10_abs >= 100) ? (a + '0') + ((b + '0') << 8) + ((c + '0') << 16) : (b + '0') + ((c + '0') << 8);
            uint64_t exp_res = e + (e10_abs_ascii << 16) + (exp_len << 56);
            exp_res = (e10 > e10_DN && e10 < e10_UP) ? 0 : exp_res;
            this->exp_result_double[e10 + 324] = exp_res;
        }

        for (int e10 = e10_DN; e10 <= e10_UP + 1; e10++) {
            const int tmp_data_ofs = e10 - e10_DN;
            auto &current_line = this->e10_variable_data[tmp_data_ofs];
            uint64_t first_sig_pos = (e10_DN <= e10 && e10 <= -1) ? 1 - e10 : 0;
            uint64_t dot_pos = (0 <= e10 && e10 <= e10_UP) ? 1 + e10 : 1;
            uint64_t move_pos = dot_pos + (0 <= e10 || e10 < e10_DN);
            current_line[max_dec_sig_len + 0] = static_cast<unsigned char> (first_sig_pos);
            current_line[max_dec_sig_len + 1] = static_cast<unsigned char> (dot_pos);
            current_line[max_dec_sig_len + 2] = static_cast<unsigned char> (move_pos);

            for (uint8_t D17 = 0; D17 <= 1; D17++) {
                const unsigned char one_offset = 15 + D17 + (move_pos > dot_pos && dot_pos <= 15 + D17);
                current_line[max_dec_sig_len + 3 + D17] = one_offset;
            }

            for (int dec_sig_len = 1; dec_sig_len <= max_dec_sig_len; dec_sig_len++) {
                uint64_t exp_pos =
                    (e10_DN <= e10 && e10 <= -1)
                        ? dec_sig_len
                        : (0 <= e10 && e10 <= e10_UP ? (e10 + 3 > dec_sig_len + 1 ? e10 + 3 : dec_sig_len + 1)
                                                     : (dec_sig_len + 1 - (dec_sig_len == 1)));
                current_line[dec_sig_len - 1] = static_cast<unsigned char> (exp_pos);
            }
#if ZZS_NO_MEMMOVE
            uint8_t v = 0xf;
            for (uint64_t j = 0; j < 0x10; ++j)
                current_line[0x20 + 0x10 + j] = v--;
            if (move_pos > dot_pos) {
                for (uint64_t j = 0xf; j > dot_pos && j > 0; --j)
                    current_line[j + 0x20 + 0x10] = current_line[j + 0x20 + 0x10 - 1];
            }
            for (uint64_t j = 0; j < 0x10; ++j) {
                auto v = current_line[j + 0x20 + 0x10];
                current_line[j + 0x20] = v ? (v - 1) : 0xf;
            }
#endif
        }
        for (int exp = 0; exp < 2048; ++exp) {
            constexpr int offset = 9;
            int q = exp - 1075 + (exp == 0);
            int k = (q * 78913) >> 18;
            int h = q + (((-k - 1) * 217707) >> 16);
            h7[exp] = static_cast<unsigned char> (h + 1 + offset);
        }
    }
};

struct _S_FloatTable {
    static constexpr int e10_DN = -3;
    static constexpr int e10_UP = 6;
    static constexpr int max_dec_sig_len = 9;
    static constexpr int max_first_sig_pos = 4;
    static constexpr int max_valid_output_len = 15;
    static constexpr int max_buffer_requirement = 21;
    static constexpr int num_pow10 = 44 - (-32) + 1;

    uint64_t pow10_float_reverse[44 - (-32) + 1];
    uint32_t exp_result_float[45 + 38 + 1];
    unsigned char e10_variable_data[e10_UP - (e10_DN) + 1 + 1][16];
    unsigned char h37[256];

    struct const_value_float {
#if ZZS_IS_AARCH64
        uint64_t c1 =
            ((static_cast<uint64_t> ('0' + '0' * 256) << (36)) + ((static_cast<uint64_t> (1) << (36 - 1)) - 7));
#else
        uint64_t c1 =
            ((static_cast<uint64_t> ('0' + '0' * 256) << (36 - 1)) + ((static_cast<uint64_t> (1) << (36 - 2)) - 7));
#endif
        uint64_t div10000 = 1844674407370956;
        uint64_t m = (1ull << 32) - 10000;
        uint32_t e7 = 10000000;
        uint32_t e6 = 1000000;
#if ZZS_USE_NEON
        int32x4_t m32_4 = {0x147b000, -100 + 0x10000, 0xce0, -10 + 0x100};
#else
        int32_t m32_4[4] = {0x147b000, -100 + 0x10000, 0xce0, -10 + 0x100};
#endif
    };

    const_value_float constants_float;

    consteval _S_FloatTable ()
        : pow10_float_reverse{}, exp_result_float{}, e10_variable_data{}, h37{}, constants_float{} {
        struct uint128_xjb {
            uint64_t w0, w1;
        };
        uint128_xjb current = {0x67de18eda5814af3, 0xcfb11ead453994ba};
        const uint64_t ten = 0xa000000000000000;

        for (int i = 0; i < num_pow10; ++i) {
            int e10 = i - 32;
            pow10_float_reverse[num_pow10 - i - 1] = e10 == 0 ? 1ull << 63 : current.w1 + 1;
            uint64_t h0 = _S_umul128_hi64_fallback (current.w0, ten);
            uint64_t c0 = h0 + current.w1 * ten;
            uint64_t c1 = (c0 < h0) + _S_umul128_hi64_fallback (current.w1, ten);
            if (c1 >> 63)
                current = {c0, c1};
            else
                current = {c0 << 1, c1 << 1 | c0 >> 63};
        }

        for (int e10 = -45; e10 <= 38; e10++) {
            uint64_t e = e10 < 0 ? ('e' + '-' * 256) : ('e' + '+' * 256);
            uint64_t e10_abs = e10 < 0 ? -e10 : e10;
            uint64_t a = e10_abs / 10;
            uint64_t b = e10_abs - a * 10;
            uint64_t e10_abs_ascii = (a + '0') + ((b + '0') << 8);
            uint64_t exp_res = e + (e10_abs_ascii << 16);
            exp_res = (e10 >= e10_DN && e10 <= e10_UP) ? 0 : exp_res;
            exp_result_float[e10 + 45] = static_cast<uint32_t> (exp_res);
        }

        for (int e10 = e10_DN; e10 <= e10_UP + 1; e10++) {
            int tmp_data_ofs = e10 - e10_DN;
            uint64_t first_sig_pos = (e10_DN <= e10 && e10 <= -1) ? 1 - e10 : 0;
            uint64_t dot_pos = (0 <= e10 && e10 <= e10_UP) ? 1 + e10 : 1;
            uint64_t move_pos = dot_pos + ((0 <= e10 || e10 < e10_DN));
            auto &current_line = e10_variable_data[tmp_data_ofs];
            current_line[max_dec_sig_len + 0] = static_cast<unsigned char> (first_sig_pos);
            current_line[max_dec_sig_len + 1] = static_cast<unsigned char> (dot_pos);
            current_line[max_dec_sig_len + 2] = static_cast<unsigned char> (move_pos);

            for (int dec_sig_len = 1; dec_sig_len <= max_dec_sig_len; dec_sig_len++) {
                const uint64_t exp_pos =
                    (e10_DN <= e10 && e10 <= -1)
                        ? dec_sig_len
                        : (0 <= e10 && e10 <= e10_UP ? (e10 + 3 > dec_sig_len + 1 ? e10 + 3 : dec_sig_len + 1)
                                                     : (dec_sig_len + 1 - (dec_sig_len == 1)));
                current_line[dec_sig_len - 1] = static_cast<unsigned char> (exp_pos);
            }
        }

        for (int exp = 0; exp < 256; exp++) {
            const int exp_bin = exp - 150 + (exp == 0);
            int k = (exp_bin * 1233) >> 12;
            int h37_precalc = (36 + 1) + exp_bin + ((k * -1701 + (-1701)) >> 9);
            h37[exp] = static_cast<unsigned char> (h37_precalc);
        }
    }
};
struct _S_ConstValueDouble {
    static constexpr int e10_DN = _S_DoubleTable::e10_DN;
    static constexpr int e10_UP = _S_DoubleTable::e10_UP;
    static constexpr int max_dec_sig_len = _S_DoubleTable::max_dec_sig_len;
    static constexpr int num_pow10 = _S_DoubleTable::num_pow10;

    uint64_t c1 = 78913ull << (64 - 18);
    uint64_t c2 = static_cast<uint64_t> (-217707);
    uint64_t c3 = static_cast<uint64_t> (1e15) - 1;
    uint64_t c4 = (1ull << 63) + 6;
    uint64_t c5 = static_cast<uint64_t> (-131072);
    uint64_t c6 = (1 << 9) - 1;
    uint64_t mul_const = 0xabcc77118461cefd;
    int64_t hundred_million = -100000000;
    uint64_t div10000 = 1844674407370956;
    uint64_t div10000_m = 0x100000000 - 10000;
    double div10000_2_d = static_cast<double> (-10000 + 0x100000000);
    int64_t div10000_2 = 0xd1b7176000;
#if ZZS_USE_NEON
    int32x4_t multipliers32 = {0x68db8bb, -10000 + 0x10000, 0x147b000, -100 + 0x10000};
    int16x8_t multipliers16 = {0xce0, -10 + 0x100, '0' + '0' * 256};
#else
    int32_t multipliers32[4] = {0x68db8bb, -10000 + 0x10000, 0x147b000, -100 + 0x10000};
    int16_t multipliers16[8] = {0xce0, -10 + 0x100, '0' + '0' * 256};
#endif
#if ZZS_USE_NEON
    uint8_t shuffle_table_neon[32] = {
        6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
    };
    uint8_t shuffle_table_memmove[32] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0,
                                         0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15};
#endif
#if ZZS_USE_NEON && ZZS_NO_MEMMOVE
    uint8_t reverse_shuffle_table[17] = {0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
#endif
#if ZZS_USE_NEON
    uint8_t shuffle_table_neon[32] = {
        6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
    };
    uint8_t shuffle_table_memmove[32] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0,
                                         0, 1, 2, 3, 4, 5, 6, 7, 8, 9,  10, 11, 12, 13, 14, 15};
#endif
#if (ZZS_NO_MEMMOVE || ZZS_USE_NEON) && (defined(__SSSE3__) && __SSSE3__)
    uint8_t shuffle_table[32] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0};
#endif
};
constexpr _S_ConstValueDouble constants_double;
alignas (64) constexpr _S_DoubleTable double_table;
alignas (64) constexpr _S_FloatTable float_table;

constexpr uint64_t
_S_compute_double_dec_sig_len (uint64_t up_down, int tz, uint64_t D17) {
    uint64_t len = (ZZS_NOT_REMOVE_FIRST_ZERO ? 14 + D17 : 15) - static_cast<uint64_t> (tz);
    const uint64_t ret = _S_cmov_branchless (up_down, len, 15 + D17);
    return ret;
}

constexpr uint64_t
_S_compute_double_dec_sig_len_sse2 (uint64_t up_down, int tz, uint64_t D17) {
    uint64_t len = (0 ? 14 + D17 : 15) + 48 - static_cast<uint64_t> (tz);
    const uint64_t ret = _S_cmov_branchless (up_down, len, 15 + D17);
    return ret;
}

constexpr uint64_t
_S_compute_float_dec_sig_len (uint64_t up_down, int tz, uint64_t lz) {
    const uint64_t ret = _S_cmov_branchless (up_down, (7 - lz) - tz, 8 - lz);
    return ret;
}

constexpr ShortestAscii
_S_to_ascii8_final (uint64_t abcdefgh_BCD, uint32_t lz, uint32_t up_down) {
    const uint64_t ZERO = 0x3030303030303030;
    uint32_t tz = _S_u64_lz_bits (abcdefgh_BCD) >> 3;
    abcdefgh_BCD = abcdefgh_BCD >> (lz << 3);
    abcdefgh_BCD = _S_is_little_endian () ? abcdefgh_BCD : _S_byteswap64_xjb (abcdefgh_BCD);
    return {abcdefgh_BCD | ZERO, _S_cmov_branchless (up_down, (7 ^ lz) - tz, 8 - lz)};
}

constexpr ShortestAscii
_S_to_ascii8_buffer (const uint64_t m, const uint32_t up_down, uint32_t &lz,
                     const _S_FloatTable::const_value_float *c) {
    uint64_t abcdefgh_BCD = 0;
    if !consteval {
#if ZZS_USE_NEON
        int32x2_t tenthousands = vreinterpret_s32_u64 (vcreate_u64 (m + c->m * ((m * (uint128_t) c->div10000) >> 64)));
        int16x4_t hundreds = vreinterpret_s16_s32 (
            vmla_n_s32 (tenthousands, vqdmulh_s32 (tenthousands, vdup_n_s32 (c->m32_4[0])), c->m32_4[1]));
        int16x4_t BCD_big_endian = vmla_n_s16 (hundreds, vqdmulh_s16 (hundreds, vdup_n_s16 (c->m32_4[2])), c->m32_4[3]);
        uint64_t hgfedcba_BCD = vget_lane_u64 (vreinterpret_u64_s16 (BCD_big_endian), 0);
        uint64_t abcdefgh_BCD = _S_byteswap64_xjb (hgfedcba_BCD);
#elif ZZS_USE_SSE2
        uint64_t aabb_ccdd_merge =
            (m << 32) + (1 - (10000ull << 32)) * ((m * static_cast<uint128_t> (1844674407370956)) >> 64);
        __m128i y = _mm_set1_epi64x (aabb_ccdd_merge);
        __m128i y_div_100 = _mm_srli_epi16 (_mm_mulhi_epu16 (y, _mm_set1_epi16 (0x147b)), 3);
        __m128i y_mod_100 = _mm_sub_epi16 (y, _mm_mullo_epi16 (y_div_100, _mm_set1_epi16 (100)));
        __m128i z = _mm_or_si128 (y_div_100, _mm_slli_epi32 (y_mod_100, 16));
        __m128i z_div_10 = _mm_mulhi_epi16 (z, _mm_set1_epi16 (0x199a));
        __m128i tmp = _mm_sub_epi16 (_mm_slli_epi16 (z, 8), _mm_mullo_epi16 (_mm_set1_epi16 (2559), z_div_10));
        abcdefgh_BCD = _mm_cvtsi128_si64 (tmp);
#endif
    } else {
        int64_t aabb_ccdd_merge = (m << 32) + (1 - (10000ull << 32)) * ((m * 109951163) >> 40);
        int64_t aa_bb_cc_dd_merge =
            (aabb_ccdd_merge << 16)
            + (1 - (100ull << 16)) * (((aabb_ccdd_merge * 10486) >> 20) & ((0x7Full << 32) | 0x7Full));
        abcdefgh_BCD =
            (aa_bb_cc_dd_merge << 8)
            + (1 - (10ull << 8))
                  * (((aa_bb_cc_dd_merge * 103) >> 10) & ((0xFull << 48) | (0xFull << 32) | (0xFull << 16) | 0xFull));
    }
    return _S_to_ascii8_final (abcdefgh_BCD, lz, up_down);
}

constexpr ShortestAscii16
_S_to_ascii16_buffer (char *buf, const uint64_t m, const uint64_t up_down, const uint64_t D17,
                      const _S_ConstValueDouble *cv ZZS_SHUFFLER_DECL) {
    constexpr uint64_t ZERO = 0x3030303030303030;
    uint32_t abcdefgh = _S_umul128_hi64_xjb (m, cv->mul_const) >> (90 - 64);
    int64_t hundred_million = cv->hundred_million;
    uint32_t ijklmnop = m + abcdefgh * hundred_million;

    if !consteval {
#if ZZS_USE_NEON
#if ZZS_NO_MEMMOVE
        uint64x1_t hundredmillions = {ijklmnop | ((uint64_t) abcdefgh << 32)};
#else
        uint64x1_t hundredmillions = {abcdefgh | ((uint64_t) ijklmnop << 32)};
#endif
        int32x2_t hundredmillions_s32 = vreinterpret_s32_u64 (hundredmillions);
                        int32x2_t high_10000 = vreinterpret_s32_u32( vshr_n_u32( vreinterpret_u32_s32( vqdmulh_s32( hundredmillions_s32, vdup_n_s32( cv->multipliers32[0] ) ), 9 ) );
                        int32x2_t tenthousands = vmla_s32( hundredmillions_s32, high_10000, vdup_n_s32( cv->multipliers32[1] ) );
                        int32x4_t extended = vreinterpretq_s32_u32( vshll_n_u16( vreinterpret_u16_s32( tenthousands ), 0 ) );
                        int32x4_t high_100 = vqdmulhq_s32( extended, vdupq_n_s32( cv->multipliers32[2] ) );
                        int16x8_t hundreds = vreinterpretq_s16_s32( vmlaq_s32( extended, high_100, vdupq_n_s32( cv->multipliers32[3] ) ) );
                        int16x8_t high_10 = vqdmulhq_s16( hundreds, vdupq_n_s16( cv->multipliers16[0] ) );
                        int16x8_t BCD_big_endian = vmlaq_s16( hundreds, high_10, vdupq_n_s16( cv->multipliers16[1] ) );
#if ZZS_NO_MEMMOVE
                        int8x16_t ascii16_swapped = vorrq_s8( vreinterpretq_s8_s16( BCD_big_endian ), vdupq_n_s8( '0' ) );
                        uint16x8_t is_not_zero = vreinterpretq_u16_u8( vcgtzq_s8( vreinterpretq_s8_s16( BCD_big_endian ) ) );
                        int8x16_t ascii16 = vreinterpretq_s8_u8( vqtbl1q_u8( vreinterpretq_u8_s8( ascii16_swapped ), vld1q_u8( move_shuffler ) ) );
                        uint64_t zeroes = vget_lane_u64( vreinterpret_u64_u8( vshrn_n_u16( is_not_zero, 4 ) ), 0 );
                        int tz = _S_u64_tz_bits( zeroes ) >> 2;
                        uint64_t dec_sig_len = up_down ? (ZZS_NOT_REMOVE_FIRST_ZERO ? (14 + D17) - tz : 15 - tz) : 15 + D17;
                        return {vreinterpretq_u64_s8( ascii16 ), dec_sig_len, vgetq_lane_s32( vreinterpretq_s32_s8( ascii16_swapped ), 0 )};
#else
                        int8x16_t BCD_little_endian = vreinterpretq_s8_u8( vqtbl1q_u8( vreinterpretq_u8_s16( BCD_big_endian ),
                                                                  vld1q_u8( &cv->shuffle_table_neon[(15 + D17) & 16] ) ) );
                        int8x16_t ascii16 = vorrq_s8( BCD_little_endian, vdupq_n_s8( '0' ) );
                        vst1q_s8( (int8_t*)buf, vdupq_n_s8( '0' ) );
                        vst1q_s8( (int8_t*)(buf + 16), vdupq_n_s8( '0' ) );
                        uint16x8_t is_not_zero = vreinterpretq_u16_u8( vcgtzq_s8( BCD_little_endian ) );
                        uint64_t zeroes = vget_lane_u64( vreinterpret_u64_u8( vshrn_n_u16( is_not_zero, 4 ) ), 0 );
                        uint32_t tz = _S_u64_lz_bits( zeroes ) >> 2;
                        uint32_t dec_sig_len = up_down ? ((0) ? (14 + D17) - tz : 15 - tz) : 15 + D17;
                        return {vreinterpretq_u64_s8( ascii16 ), dec_sig_len};
#endif
#elif ZZS_USE_SSE2
        __m128i x = _mm_unpacklo_epi64 (_mm_cvtsi32_si128 (ijklmnop), _mm_cvtsi32_si128 (abcdefgh));
        __m128i y =
            _mm_add_epi64 (x, _mm_mul_epu32 (_mm_set1_epi64x ((1ull << 32) - 10000),
                                             _mm_srli_epi64 (_mm_mul_epu32 (x, _mm_set1_epi64x (109951163)), 40)));
        __m128i y_div_100 = _mm_srli_epi16 (_mm_mulhi_epu16 (y, _mm_set1_epi16 (0x147b)), 3);
        __m128i y_mod_100 = _mm_sub_epi16 (y, _mm_mullo_epi16 (y_div_100, _mm_set1_epi16 (100)));
        __m128i z = _mm_or_si128 (y_div_100, _mm_slli_epi32 (y_mod_100, 16));
        __m128i z_div_10 = _mm_mulhi_epi16 (z, _mm_set1_epi16 (0x199a));
        __m128i bcd_swapped = _mm_sub_epi16 (_mm_slli_epi16 (z, 8), _mm_mullo_epi16 (_mm_set1_epi16 (2559), z_div_10));
        __m128i little_endian_bcd = _mm_shuffle_epi32 (bcd_swapped, _MM_SHUFFLE (0, 1, 2, 3));

#if ZZS_NOT_REMOVE_FIRST_ZERO
        __m128i move_mask;
        memcpy (&move_mask, &(cv->shuffle_table[D17 ? 0 : 1]), 16);
        little_endian_bcd = _mm_shuffle_epi8 (little_endian_bcd, move_mask);
#endif

        int mask = _mm_movemask_epi8 (_mm_cmpgt_epi8 (little_endian_bcd, _mm_setzero_si128 ()));
        int tz = _S_u64_lz_bits (mask);

        __m128i ascii16 = _mm_add_epi8 (little_endian_bcd, _mm_set1_epi8 ('0'));
        memset (buf, '0', 32);
        return {ascii16, _S_compute_double_dec_sig_len_sse2 (up_down, tz, D17)};
#endif
    } else {
        uint64_t abcd_efgh = abcdefgh + (0x100000000 - 10000) * ((abcdefgh * 0x68db8bbull) >> 40);
        uint64_t ijkl_mnop = ijklmnop + (0x100000000 - 10000) * ((ijklmnop * 0x68db8bbull) >> 40);
        uint64_t ab_cd_ef_gh = abcd_efgh + (0x10000 - 100) * (((abcd_efgh * 0x147b) >> 19) & 0x7f0000007f);
        uint64_t ij_kl_mn_op = ijkl_mnop + (0x10000 - 100) * (((ijkl_mnop * 0x147b) >> 19) & 0x7f0000007f);
        uint64_t a_b_c_d_e_f_g_h = ab_cd_ef_gh + (0x100 - 10) * (((ab_cd_ef_gh * 0x67) >> 10) & 0xf000f000f000f);
        uint64_t i_j_k_l_m_n_o_p = ij_kl_mn_op + (0x100 - 10) * (((ij_kl_mn_op * 0x67) >> 10) & 0xf000f000f000f);
        int abcdefgh_tz = _S_u64_tz_bits (a_b_c_d_e_f_g_h);
        int ijklmnop_tz = _S_u64_tz_bits (i_j_k_l_m_n_o_p);
        uint64_t abcdefgh_bcd = _S_is_little_endian () ? _S_byteswap64_xjb (a_b_c_d_e_f_g_h) : a_b_c_d_e_f_g_h;
        uint64_t ijklmnop_bcd = _S_is_little_endian () ? _S_byteswap64_xjb (i_j_k_l_m_n_o_p) : i_j_k_l_m_n_o_p;
        int tz = (ijklmnop == 0) ? 64 + abcdefgh_tz : ijklmnop_tz;
        tz = tz / 8;
        for (int i = 0; i < 32; ++i)
            buf[i] = '0';
        ShortestAscii16 result{};
#if ZZS_USE_NEON || ZZS_USE_SSE2
        struct M128IUnion {
            uint64_t lo;
            uint64_t hi;
        };
        M128IUnion m_union;
        m_union.hi = abcdefgh_bcd | ZERO;
        m_union.lo = ijklmnop_bcd | ZERO;
        result.ascii16 = std::bit_cast<__m128i> (m_union);
        result.dec_sig_len_sub1 = _S_compute_double_dec_sig_len (up_down, tz, D17);
#else
        result.hi = abcdefgh_bcd | ZERO;
        result.lo = ijklmnop_bcd | ZERO;
        result.dec_sig_len_sub1 = _S_compute_double_dec_sig_len (up_down, tz, D17);
#endif
        return result;
    }
}

constexpr char *
_S_xjb64 (char *buff, const double v) {
    const _S_ConstValueDouble *cv = &constants_double;
    constexpr auto t = &double_table;
#if ZZS_IS_AARCH64 && (defined(__clang__) || defined(__GNUC__))
    asm ("" : "+r"(cv));
#endif
    uint64_t vi;
    if !consteval {
        memcpy (&vi, &v, sizeof (v));
    } else {
        vi = std::bit_cast<uint64_t> (v);
    }
    *buff = '-';
    buff += vi >> 63;
    uint64_t sig = vi & ((1ull << 52) - 1);
    uint64_t exp = (vi << 1) >> 53;
    int64_t q = static_cast<int64_t> (exp) - 1075;
    uint64_t c = sig | (1ull << 52);
#if ZZS_IS_X64
    if (((exp + 1) & 2047) <= 1) [[unlikely]]
#endif
    {
        if (exp == 0) {
            if (sig <= 1) {
                const char *str = sig ? "5e-324\0" : "0.0\0\0\0\0";
                if !consteval {
                    return static_cast<char *> (memcpy (buff, str, 8)) + (sig ? 6 : 3);
                } else {
                    _S_constexpr_memcpy (buff, str, sig ? 6 : 3);
                    return buff + (sig ? 6 : 3);
                }
            }
            c = sig;
            q = 1 - 1075;
        }
        if (exp == 2047) {
            if !consteval {
                return static_cast<char *> (memcpy (buff, sig ? "nan" : "inf", 4)) + 3;
            } else {
                _S_constexpr_memcpy (buff, sig ? "nan" : "inf", 4);
                return buff + 3;
            }
        }
    }
    unsigned char h7_precalc = t->h7[exp];
    constexpr int offset = 9;
    bool irregular = sig == 0;
    int64_t k = 0;
    uint64_t m_up = 0;
    uint32_t one = 0;
    uint32_t up_down = 0;
    {
        uint64_t hi64 = 0, lo64 = 0, pow10_hi = 0, pow10_lo = 0;
#if defined(__SIZEOF_INT128__) && ZZS_IS_AARCH64
        k = ((int64_t) q * (uint128_t) (78913ull << (64 - 18))) >> 64;
#else
        k = (q * 78913) >> 18;
#endif

        const uint64_t *pow10_ptr = t->pow10_double + 323 * 2 + 2 + k * 2;
        pow10_hi = pow10_ptr[0];
        pow10_lo = pow10_ptr[1];
        _S_umul128_u64_hi128 (pow10_hi, pow10_lo, c << h7_precalc, &hi64, &lo64);
        uint64_t dot_one = (hi64 << (64 - offset)) | (lo64 >> offset);
        uint64_t half_ulp = (pow10_hi >> ((1 + offset) - h7_precalc)) + ((c + 1) & 1);
        bool up = half_ulp > ~0 - dot_one;
        bool down = half_ulp > dot_one;
        m_up = (hi64 >> offset) + up;
        up_down = up + down;
        uint64_t half = (dot_one == (1ull << 62)) ? 0 : cv->c4;
        one = static_cast<uint32_t> (_S_u128_madd_hi64 (dot_one, 10, half));
    }
    if (irregular) [[unlikely]] {
        k = (q * 315653 - 131072) >> 20;
        int64_t h = q + ((k * -217707 - 217707) >> 16);
        uint64_t pow10_hi = t->pow10_double[323 * 2 + 2 + k * 2];
        uint64_t half_ulp = pow10_hi >> (-h);
        uint64_t dot_one = (pow10_hi << (53 + h));
        uint64_t up = half_ulp > ~0 - dot_one;
        uint64_t down = (half_ulp >> 1) > dot_one;
        m_up = (pow10_hi >> (11 - h)) + up;
        up_down = up + down;
        one = ((dot_one >> (53 + h)) * 5 + (1 << (9 - h))) >> (10 - h);
        if ((((dot_one >> 54) * 5) & ((1 << 9) - 1)) > ((half_ulp >> 55) * 5))
            one = (((dot_one >> 54) * 5) >> 9) + 1;
        if (dot_one == (1ull << 62))
            one = 2;
    }
    uint64_t D17 = m_up > static_cast<uint64_t> (cv->c3);
    uint64_t mr = D17 ? m_up : m_up * 10;

    int64_t e10 = k + (15 + D17);
    constexpr int64_t e10_DN = t->e10_DN;
    constexpr int64_t e10_UP = t->e10_UP;
    constexpr uint64_t interval = e10_UP - e10_DN + 1;
#if ZZS_IS_AARCH64
    uint32_t e10_3 = static_cast<int32_t> (e10) + static_cast<int32_t> (-e10_DN);
    uint32_t e10_data_ofs = e10_3 < interval ? e10_3 : interval;
#else
    uint64_t e10_3 = e10 + (-e10_DN);
    uint64_t e10_data_ofs = e10_3 < interval ? e10_3 : interval;
#endif

#if ZZS_NO_MEMMOVE
    if !consteval {
        memset (buff, '0', 8);
    } else {
        for (int i = 0; i < 8; ++i)
            buff[i] = '0';
    }
    const uint8_t *move_shuffler = &(t->e10_variable_data[e10_data_ofs][32 + D17 * 16]);
#else
    if !consteval {
        memset (buff, '0', 32);
    } else {
        for (int i = 0; i < 32; ++i)
            buff[i] = '0';
    }
#endif
    ShortestAscii16 s =
        _S_to_ascii16_buffer (buff, ZZS_NOT_REMOVE_FIRST_ZERO ? m_up : mr, up_down, D17, cv ZZS_SHUFFLER_PARAM);
    uint64_t first_sig_pos = t->e10_variable_data[e10_data_ofs][17 + 0];
    uint64_t dot_pos = t->e10_variable_data[e10_data_ofs][17 + 1];
    uint64_t one_offset = t->e10_variable_data[e10_data_ofs][17 + 3 + D17];
    uint64_t move_pos = t->e10_variable_data[e10_data_ofs][17 + 2];
    uint64_t exp_pos = t->e10_variable_data[e10_data_ofs][s.dec_sig_len_sub1];
    char *buf_origin = buff;
    buff += first_sig_pos;
#if ZZS_USE_NEON_OR_SSE2
    if !consteval {
        memcpy (buff, &(s.ascii16), 16);
    } else {
        struct M128IUnion {
            uint64_t lo;
            uint64_t hi;
        };
        auto m = std::bit_cast<M128IUnion> (s.ascii16);
        struct uint64_bytes {
            char b[sizeof (uint64_t)];
        };
        auto hi_bytes = std::bit_cast<uint64_bytes> (m.hi);
        auto lo_bytes = std::bit_cast<uint64_bytes> (m.lo);
        for (int i = 0; i < 8; ++i)
            buff[i] = hi_bytes.b[i];
        for (int i = 0; i < 8; ++i)
            buff[8 + i] = lo_bytes.b[i];
    }
#else
    if !consteval {
        memcpy (buff + 0, &(s.hi), 8);
        memcpy (buff + 8, &(s.lo), 8);
    } else {
        struct Uint64Bytes {
            char b[sizeof (uint64_t)];
        };
        auto hi_bytes = std::bit_cast<Uint64Bytes> (s.hi);
        auto lo_bytes = std::bit_cast<Uint64Bytes> (s.lo);
        for (int i = 0; i < 8; ++i)
            buff[i] = hi_bytes.b[i];
        for (int i = 0; i < 8; ++i)
            buff[8 + i] = lo_bytes.b[i];
    }
#endif

    one |= 0x30303030;
    if !consteval {
        memcpy (&buff[15 + D17], &one, 4);
        memmove (&buff[move_pos], &buff[dot_pos], 16);
    } else {
        struct CharUint32Bytes {
            char b[sizeof (uint32_t)];
        };
        auto one_bytes = std::bit_cast<CharUint32Bytes> (one);
        for (int i = 0; i < 4; ++i)
            buff[15 + D17 + i] = one_bytes.b[i];
        if (move_pos < dot_pos) {
            for (int i = 0; i < 16; ++i)
                buff[move_pos + i] = buff[dot_pos + i];
        } else if (move_pos > dot_pos) {
            for (int i = 16; i > 0; --i)
                buff[move_pos + i - 1] = buff[dot_pos + i - 1];
        }
    }
    buf_origin[dot_pos] = '.';

#if ZZS_IS_AARCH64
    if (exp == 0) [[unlikely]]
#endif
        if (m_up < static_cast<uint64_t> (1e14)) [[unlikely]] {
            uint64_t lz = 0;
            while (buff[2 + lz] == '0')
                lz++;
            lz += 2;
            e10 -= lz - 1;
            buff[0] = buff[lz];
            if !consteval {
                memmove (&buff[2], &buff[lz + 1], 16);
            } else {
                if (2 < lz + 1) {
                    for (int i = 0; i < 16; ++i)
                        buff[2 + i] = buff[lz + 1 + i];
                } else if (2 > lz + 1) {
                    for (int i = 16; i > 0; --i)
                        buff[2 + i - 1] = buff[lz + 1 + i - 1];
                }
            }
            exp_pos = exp_pos - lz + (exp_pos - lz != 1);
        }

    uint64_t exp_result = t->exp_result_double[e10 + 324];
    buff += exp_pos;
    if !consteval {
        memcpy (buff, &exp_result, 8);
    } else {
        struct CharUint64Bytes {
            char b[sizeof (uint64_t)];
        };
        auto exp_result_bytes = std::bit_cast<CharUint64Bytes> (exp_result);
        for (int i = 0; i < 8; ++i)
            buff[i] = exp_result_bytes.b[i];
    }
    const uint64_t exp_len = exp_result >> 56;
    return buff + exp_len;
}

constexpr char *
_S_xjb32 (char *buff, const float v) {
    constexpr auto t = &float_table;
    const _S_FloatTable::const_value_float *c = &t->constants_float;

    uint32_t vi;
    if !consteval {
        memcpy (&vi, &v, 4);
    } else {
        vi = std::bit_cast<uint32_t> (v);
    }
    buff[0] = '-';
    buff += vi >> 31;
    const uint32_t sig = vi & ((1 << 23) - 1);
    const uint64_t exp = (vi << 1) >> 24;
    uint64_t sig_bin = sig | (1 << 23);
    int64_t exp_bin = static_cast<int64_t> (exp) - 150;
    if (exp == 0) {
        if (sig == 0) {
            constexpr SimpleStringView zero = "0.0";
            if !consteval {
                return static_cast<char *> (memcpy (buff, zero.data (), 4)) + zero.size ();
            } else {
                _S_constexpr_memcpy (buff, zero.data (), zero.size ());
                return buff + zero.size ();
            }
        }
        exp_bin = 1 - 150;
        sig_bin = sig;
    }
    if (exp == 255) {
        constexpr SimpleStringView nan = "nan";
        constexpr SimpleStringView inf = "inf";
        if !consteval {
            return static_cast<char *> (memcpy (buff, sig ? nan.data () : inf.data (), 4)) + 3;
        } else {
            _S_constexpr_memcpy (buff, sig ? nan.data () : inf.data (), 4);
            return buff + 3;
        }
    }
    uint32_t h37_precalc = t->h37[exp];
    bool irregular = sig == 0;
    constexpr int BIT = 36;

    int64_t k = (exp_bin * 1233) >> 12;
    if (irregular) {
        k = (exp_bin * 1233 - 512) >> 12;
        h37_precalc = (BIT + 1) + exp_bin + ((k * -1701 + (-1701)) >> 9);
    }
    uint64_t pow10_hi = t->pow10_float_reverse[45 + k];
    uint64_t cb = sig_bin << h37_precalc;
    uint64_t hi64 = _S_umul128_hi64_xjb (cb, pow10_hi);
    uint64_t half_ulp = (pow10_hi >> (65 - h37_precalc)) + ((sig + 1) & 1);
    uint64_t dot_one_36bit = hi64 & ((static_cast<uint64_t> (1) << BIT) - 1);
    uint32_t m_up = static_cast<uint32_t> ((hi64 + half_ulp) >> BIT);
    uint32_t up_down = m_up > static_cast<uint32_t> ((hi64 - (half_ulp >> 0)) >> BIT);
#if ZZS_IS_AARCH64
    uint32_t one = (dot_one_36bit * 10 + c->c1 + (dot_one_36bit >> (BIT - 4))) >> (BIT);
#else
    uint32_t one = (dot_one_36bit * 5 + c->c1 + (dot_one_36bit >> (BIT - 4))) >> (BIT - 1);
#endif

    if (irregular) {
        if ((exp_bin == 31 - 150) | (exp_bin == 214 - 150) | (exp_bin == 217 - 150))
            ++one;
        up_down = m_up > static_cast<uint32_t> ((hi64 - (half_ulp >> 1)) >> BIT);
    }
    uint32_t lz = (m_up < c->e7) + (m_up < c->e6);
    if !consteval {
        memset (buff, '0', 16);
    } else {
        for (int i = 0; i < 16; ++i)
            buff[i] = '0';
    }
    ShortestAscii s = _S_to_ascii8_buffer (m_up, up_down, lz, c);
    int32_t e10 = static_cast<int32_t> (k) + (8 - lz);
    constexpr int64_t e10_DN = t->e10_DN;
    constexpr int64_t e10_UP = t->e10_UP;
    constexpr uint64_t interval = e10_UP - e10_DN + 1;
    uint32_t e10_3 = e10 + (-e10_DN);
    uint32_t e10_data_ofs = e10_3 < interval ? e10_3 : interval;
    uint64_t first_sig_pos = t->e10_variable_data[e10_data_ofs][9 + 0];
    uint64_t dot_pos = t->e10_variable_data[e10_data_ofs][9 + 1];
    uint64_t move_pos = t->e10_variable_data[e10_data_ofs][9 + 2];
    uint64_t exp_pos = t->e10_variable_data[e10_data_ofs][s.dec_sig_len_sub1];
    char *buf_origin = buff;
    buff += first_sig_pos;
    if !consteval {
        memcpy (buff, &(s), 8);
        memcpy (&buff[8 - lz], &one, 4);
        memmove (&buff[move_pos], &buff[dot_pos], 8);
    } else {
        struct TmpAscii {
constexpr unsigned long long
uitoa (char *buff, unsigned long long value, const unsigned int base, const bool upper_case) {
    if (buff == nullptr)
        return 0;
    const unsigned int radix = (base < 2 || base > 36) ? 10u : base;
    char tmp[64];
    unsigned int len = 0;
    do {
        const unsigned int digit = static_cast<unsigned int> (value % radix);
        if (digit < 10)
            tmp[len++] = static_cast<char> ('0' + digit);
        else
            tmp[len++] = static_cast<char> ((upper_case ? 'A' : 'a') + digit - 10);
        value /= radix;
    } while (value != 0);
    for (unsigned int i = 0; i < len; ++i)
        buff[i] = tmp[len - i - 1];
    buff[len] = '\0';
    return len;
}
constexpr unsigned long long
uitoa (char *buff, const unsigned long long value, const unsigned int base) {
    return uitoa (buff, value, base, false);
}
constexpr unsigned long long
uitoa (char *buff, const unsigned long long value) {
    return uitoa (buff, value, 10, false);
}
constexpr unsigned long long
itoa (char *buff, const long long value, const unsigned int base, const bool upper_case) {
    if (buff == nullptr)
        return 0;
    else {
        if (value < 0) {
            buff[0] = '-';
            return 1ull + uitoa (buff + 1, 0ull - static_cast<unsigned long long> (value), base, upper_case);
        }
    }
    return uitoa (buff, static_cast<unsigned long long> (value), base, upper_case);
}
constexpr unsigned long long
itoa (char *buff, const long long value, const unsigned int base) {
    return itoa (buff, value, base, false);
}
constexpr unsigned long long
itoa (char *buff, const long long value) {
    return itoa (buff, value, 10, false);
}
constexpr unsigned long long
uitoa (char *buff, const unsigned long long value, const NumberBase base, const bool upper_case) {
    return uitoa (buff, value, static_cast<unsigned int> (base), upper_case);
}
constexpr unsigned long long
uitoa (char *buff, const unsigned long long value, const NumberBase base) {
    return uitoa (buff, value, base, false);
}
constexpr unsigned long long
itoa (char *buff, const long long value, const NumberBase base, const bool upper_case) {
    return itoa (buff, value, static_cast<unsigned int> (base), upper_case);
}
constexpr unsigned long long
itoa (char *buff, const long long value, const NumberBase base) {
    return itoa (buff, value, base, false);
}
            ShortestAscii ascii;
        };
        const TmpAscii values{s};
        struct CharTmpAsciiBytes {
            char b[sizeof (TmpAscii)];
        };
        struct CharUin32Bytes {
            char b[sizeof (uint32_t)];
        };
        auto s_bytes = std::bit_cast<CharTmpAsciiBytes> (values.ascii);
        for (int i = 0; i < 8; ++i)
            buff[i] = s_bytes.b[i];
        auto one_bytes = std::bit_cast<CharUin32Bytes> (one);
        for (int i = 0; i < 4; ++i)
            buff[8 - lz + i] = one_bytes.b[i];
        if (move_pos < dot_pos) {
            for (int i = 0; i < 8; ++i)
                buff[move_pos + i] = buff[dot_pos + i];
        } else if (move_pos > dot_pos) {
            for (int i = 8; i > 0; --i)
                buff[move_pos + i - 1] = buff[dot_pos + i - 1];
        }
    }
    buf_origin[dot_pos] = '.';

#if ZZS_IS_AARCH64
    if (exp == 0)
#endif
        if (m_up < 100000) {
            uint64_t lz2 = 0;
            uint64_t u;
            if !consteval {
                memcpy (&u, &buff[2], 8);
            } else {
                for (int i = 0; i < 8; ++i)
                    reinterpret_cast<char *> (&u)[i] = buff[2 + i];
            }
            u = _S_is_little_endian () ? u : _S_byteswap64_xjb (u);
            lz2 = _S_u64_tz_bits (u & 0x0f0f0f0f0f0f0f0f) / 8;
            lz2 += 2;
            e10 -= lz2 - 1;
            buff[0] = buff[lz2];
            if !consteval {
                memmove (&buff[2], &buff[lz2 + 1], 8);
            } else {
                if (2 < lz2 + 1) {
                    for (int i = 0; i < 8; ++i)
                        buff[2 + i] = buff[lz2 + 1 + i];
                } else if (2 > lz2 + 1) {
                    for (int i = 8; i > 0; --i)
                        buff[2 + i - 1] = buff[lz2 + 1 + i - 1];
                }
            }
            exp_pos = exp_pos - lz2 + (exp_pos - lz2 != 1);
        }
    uint32_t exp_result_u32 = t->exp_result_float[45 + e10];
    if constexpr (!_S_is_little_endian ())
        exp_result_u32 = ((exp_result_u32 & 0xff000000) >> 24) | ((exp_result_u32 & 0x00ff0000) >> 8)
                         | ((exp_result_u32 & 0x0000ff00) << 8) | ((exp_result_u32 & 0x0000000ff) << 24);
    const uint64_t exp_result_u64 =
        _S_is_little_endian () ? exp_result_u32 : static_cast<uint64_t> (exp_result_u32) << 32;
    buff += exp_pos;
    if !consteval {
        memcpy (buff, &exp_result_u64, 8);
    } else {
        struct char_uint64_bytes {
            char b[sizeof (uint64_t)];
        };
        auto exp_result_bytes = std::bit_cast<char_uint64_bytes> (exp_result_u64);
        for (int i = 0; i < 8; ++i)
            buff[i] = exp_result_bytes.b[i];
    }
    return buff + (exp_result_u64 & 4);
}

template <typename ValFn>
constexpr unsigned long long
_S_dtoa (char *buff, const double value, ValFn &&fn) {
    if (buff == nullptr)
        return 0;
    char *result = fn (buff, value);
    unsigned long long len = static_cast<size_t> (result - buff);
    buff[len] = '\0';
    return len;
}
template <typename ValFn>
constexpr unsigned long long
_S_dtoa (char *buff, const double value, const int precision, ValFn &&fn) {
    if (buff == nullptr)
        return 0;
    if (precision < 0)
        return _S_dtoa (buff, value, fn);
    unsigned long long len = _S_dtoa (buff, value, fn);
    char *dot_pos;
    if !consteval {
        dot_pos = static_cast<char *> (memchr (buff, '.', len));
    } else {
        dot_pos = nullptr;
        for (size_t i = 0; i < len; ++i)
            if (buff[i] == '.') {
                dot_pos = buff + i;
                break;
            }
    }
    if (dot_pos != nullptr) {
        const int decimal_places = static_cast<int> (len - (dot_pos - buff + 1));
        if (decimal_places > precision)
            len = (dot_pos - buff + 1) + precision;
        else if (decimal_places < precision) {
            if (len + (precision - decimal_places) < 64) {
                if !consteval {
                    memset (buff + len, '0', precision - decimal_places);
                } else {
                    for (int i = 0; i < precision - decimal_places; ++i)
                        buff[len + i] = '0';
                }
                len += precision - decimal_places;
            }
        }
    } else {
        if (len + precision + 1 < 64) {
            buff[len] = '.';
            ++len;
            if !consteval {
                memset (buff + len, '0', precision);
            } else {
                for (int i = 0; i < precision; ++i)
                    buff[len + i] = '0';
            }
            len += precision;
        }
    }
    buff[len] = '\0';
    return len;
}
template <typename ValFn>
constexpr unsigned long long
_S_ftoa (char *buff, const float value, ValFn &&fn) {
    if (buff == nullptr)
        return 0;
    const char *result = fn (buff, value);
    unsigned long long len = static_cast<size_t> (result - buff);
    buff[len] = '\0';
    return len;
}
template <typename ValFn>
constexpr unsigned long long
_S_ftoa (char *buff, const float value, const int precision, ValFn &&fn) {
    if (precision < 0)
        return _S_ftoa (buff, value, fn);
    else
        return _S_dtoa (buff, value, precision, fn);
}

constexpr unsigned long long
dtoa (char *buff, const double value) {
    return _S_dtoa (buff, value, &_S_xjb64);
}
constexpr unsigned long long
dtoa (char *buff, const double value, const int precision) {
    return _S_dtoa (buff, value, precision, &_S_xjb64);
}
constexpr unsigned long long
ftoa (char *buff, const float value) {
    return _S_ftoa (buff, value, &_S_xjb32);
}
constexpr unsigned long long
ftoa (char *buff, const float value, const int precision) {
    return _S_ftoa (buff, value, precision, &_S_xjb32);
}

template <typename T> struct FormatArg {
    T v;
    constexpr FormatArg (T val) : v (val) {
    }
};

constexpr unsigned long long
uitoa (char *buff, unsigned long long value, const unsigned int base, const bool upper_case) {
    if (buff == nullptr)
        return 0;
    const unsigned int radix = (base < 2 || base > 36) ? 10u : base;
    char tmp[64];
    unsigned int len = 0;
    do {
        const unsigned int digit = static_cast<unsigned int> (value % radix);
        if (digit < 10)
            tmp[len++] = static_cast<char> ('0' + digit);
        else
            tmp[len++] = static_cast<char> ((upper_case ? 'A' : 'a') + digit - 10);
        value /= radix;
    } while (value != 0);
    for (unsigned int i = 0; i < len; ++i)
        buff[i] = tmp[len - i - 1];
    buff[len] = '\0';
    return len;
}
constexpr unsigned long long
uitoa (char *buff, const unsigned long long value, const unsigned int base) {
    return uitoa (buff, value, base, false);
}
constexpr unsigned long long
uitoa (char *buff, const unsigned long long value) {
    return uitoa (buff, value, 10, false);
}
constexpr unsigned long long
itoa (char *buff, const long long value, const unsigned int base, const bool upper_case) {
    if (buff == nullptr)
        return 0;
    else {
        if (value < 0) {
            buff[0] = '-';
            return 1ull + uitoa (buff + 1, 0ull - static_cast<unsigned long long> (value), base, upper_case);
        }
    }
    return uitoa (buff, static_cast<unsigned long long> (value), base, upper_case);
}
constexpr unsigned long long
itoa (char *buff, const long long value, const unsigned int base) {
    return itoa (buff, value, base, false);
}
constexpr unsigned long long
itoa (char *buff, const long long value) {
    return itoa (buff, value, 10, false);
}
constexpr unsigned long long
uitoa (char *buff, const unsigned long long value, const NumberBase base, const bool upper_case) {
    return uitoa (buff, value, static_cast<unsigned int> (base), upper_case);
}
constexpr unsigned long long
uitoa (char *buff, const unsigned long long value, const NumberBase base) {
    return uitoa (buff, value, base, false);
}
constexpr unsigned long long
itoa (char *buff, const long long value, const NumberBase base, const bool upper_case) {
    return itoa (buff, value, static_cast<unsigned int> (base), upper_case);
}
constexpr unsigned long long
itoa (char *buff, const long long value, const NumberBase base) {
    return itoa (buff, value, base, false);
}

template <unsigned int N, typename CharTy> struct FormatArg<const CharTy[N]> {
    using value_type = CharTy *;
    BasicTemplatedStringView<N, CharTy> v;
    constexpr FormatArg (const CharTy (&str)[N]) noexcept : v (str) {
    }
    constexpr
    operator value_type () const noexcept {
        return v.data ();
    }
};
template <typename T> FormatArg (T) -> FormatArg<T>;
template <unsigned int N, typename CharTy> FormatArg (const CharTy (&)[N]) -> FormatArg<const CharTy[N]>;

enum class SpecAlign : unsigned char { NONE, LHS, RHS, CENTER };
enum class SpecSign : unsigned char { MINUS, PLUS, SPACE };
enum class SpecFlag : unsigned char { NONE = 0, ALT = 1, ZERO_PAD = 2 };
enum class FloatKind : unsigned char { NUMBER, INFINITY, NOT_A_NUMBER };

constexpr SpecFlag
operator| (const SpecFlag lhs, const SpecFlag rhs) noexcept {
    return static_cast<SpecFlag> (static_cast<unsigned int> (lhs) | static_cast<unsigned int> (rhs));
}

constexpr SpecFlag
operator& (const SpecFlag lhs, const SpecFlag rhs) noexcept {
    return static_cast<SpecFlag> (static_cast<unsigned int> (lhs) & static_cast<unsigned int> (rhs));
}

constexpr bool
has_flag (const SpecFlag flags, const SpecFlag flag) noexcept {
    return (flags & flag) != SpecFlag::NONE;
}

template <auto...> inline constexpr bool format_argument_error = false;

template <typename CharTy> struct FormatSpec {
    CharTy _M_fill = static_cast<CharTy> (' ');
    CharTy _M_type = static_cast<CharTy> ('\0');
    SpecAlign _M_align = SpecAlign::NONE;
    SpecSign _M_sign = SpecSign::MINUS;
    SpecFlag _M_flags = SpecFlag::NONE;
    unsigned short _M_width = 0;
    short _M_precision = -1;
};

template <typename CharTy> struct FormatSink {
    CharTy *_M_dst = nullptr;
    unsigned int _M_capacity = 0;
    unsigned int _M_pos = 0;
    bool _M_stopped = false;

    constexpr void
    put (const CharTy ch) noexcept {
        if (this->_M_stopped)
            return;
        if (this->_M_dst != nullptr && this->_M_pos < this->_M_capacity)
            this->_M_dst[this->_M_pos] = ch;
        ++this->_M_pos;
    }

    constexpr void
    fill (unsigned int count, const CharTy ch) noexcept {
        while (count-- > 0)
            this->put (ch);
    }
};

template <typename CharTy>
constexpr void
put_ascii (FormatSink<CharTy> &sink, const char *text, const unsigned int len) noexcept {
    for (unsigned int i = 0; i < len; ++i)
        sink.put (static_cast<CharTy> (text[i]));
}

template <typename CharTy>
constexpr unsigned int
text_length (const CharTy *text) noexcept {
    unsigned int len = 0;
    while (text[len] != static_cast<CharTy> ('\0'))
        ++len;
    return len;
}

template <typename CharTy>
constexpr CharTy
pick_sign (const bool negative, const FormatSpec<CharTy> &spec) noexcept {
    if (negative)
        return static_cast<CharTy> ('-');
    if (spec._M_sign == SpecSign::PLUS)
        return static_cast<CharTy> ('+');
    if (spec._M_sign == SpecSign::SPACE)
        return static_cast<CharTy> (' ');
    return static_cast<CharTy> ('\0');
}

template <typename CharTy>
constexpr void
text_spec (FormatSpec<CharTy> &spec) noexcept {
    if (has_flag (spec._M_flags, SpecFlag::ZERO_PAD) && spec._M_align == SpecAlign::NONE) {
        spec._M_fill = static_cast<CharTy> ('0');
        spec._M_align = SpecAlign::RHS;
    }
}

template <typename CharTy, typename EmitPrefix, typename EmitBody>
constexpr void
emit_padded (FormatSink<CharTy> &sink, const FormatSpec<CharTy> &spec, const SpecAlign default_align,
             const CharTy sign_ch, const unsigned int prefix_len, const unsigned int body_len, EmitPrefix &&emit_prefix,
             EmitBody &&emit_body) noexcept {
    const unsigned int sign_len = sign_ch != static_cast<CharTy> ('\0') ? 1u : 0u;
    const unsigned int total = sign_len + prefix_len + body_len;
    const unsigned int pad = spec._M_width > total ? static_cast<unsigned int> (spec._M_width) - total : 0u;
    const auto emit_sign = [&] () noexcept {
        if (sign_len != 0)
            sink.put (sign_ch);
    };
    if (has_flag (spec._M_flags, SpecFlag::ZERO_PAD) && spec._M_align == SpecAlign::NONE) {
        emit_sign ();
        emit_prefix ();
        sink.fill (pad, static_cast<CharTy> ('0'));
        emit_body ();
        return;
    }
    const SpecAlign align = spec._M_align != SpecAlign::NONE ? spec._M_align : default_align;
    if (align == SpecAlign::LHS) {
        emit_sign ();
        emit_prefix ();
        emit_body ();
        sink.fill (pad, spec._M_fill);
    } else if (align == SpecAlign::CENTER) {
        const unsigned int left_pad = pad / 2;
        sink.fill (left_pad, spec._M_fill);
        emit_sign ();
        emit_prefix ();
        emit_body ();
        sink.fill (pad - left_pad, spec._M_fill);
    } else {
        sink.fill (pad, spec._M_fill);
        emit_sign ();
        emit_prefix ();
        emit_body ();
    }
}

template <typename CharTy, typename EmitBody>
constexpr void
emit_padded (FormatSink<CharTy> &sink, const FormatSpec<CharTy> &spec, const SpecAlign default_align,
             const CharTy sign_ch, const unsigned int body_len, EmitBody &&emit_body) noexcept {
    emit_padded (sink, spec, default_align, sign_ch, 0u, body_len, [] () noexcept {}, emit_body);
}

template <typename CharTy>
constexpr bool
is_integer_type (const CharTy type) noexcept {
    return type == static_cast<CharTy> ('\0') || type == static_cast<CharTy> ('d') || type == static_cast<CharTy> ('u')
           || type == static_cast<CharTy> ('x') || type == static_cast<CharTy> ('X')
           || type == static_cast<CharTy> ('b') || type == static_cast<CharTy> ('B')
           || type == static_cast<CharTy> ('o') || type == static_cast<CharTy> ('c');
}

template <typename CharTy>
constexpr bool
is_floating_type (const CharTy type) noexcept {
    return type == static_cast<CharTy> ('\0') || type == static_cast<CharTy> ('f') || type == static_cast<CharTy> ('F')
           || type == static_cast<CharTy> ('e') || type == static_cast<CharTy> ('E')
           || type == static_cast<CharTy> ('g') || type == static_cast<CharTy> ('G');
}

template <typename CharTy, typename Reader>
constexpr bool
parse_format_spec (const Reader &rd, const unsigned int begin, const unsigned int end,
                   FormatSpec<CharTy> &spec) noexcept {
    const auto is_digit = [] (const CharTy c) noexcept {
        return c >= static_cast<CharTy> ('0') && c <= static_cast<CharTy> ('9');
    };
    const auto is_align = [] (const CharTy c) noexcept {
        return c == static_cast<CharTy> ('<') || c == static_cast<CharTy> ('>') || c == static_cast<CharTy> ('^');
    };
    const auto align_of = [] (const CharTy c) noexcept {
        if (c == static_cast<CharTy> ('<'))
            return SpecAlign::LHS;
        if (c == static_cast<CharTy> ('>'))
            return SpecAlign::RHS;
        return SpecAlign::CENTER;
    };
    if (begin == end)
        return true;
    if (rd[begin] != static_cast<CharTy> (':'))
        return false;
    unsigned int i = begin + 1;
    if (i == end)
        return true;
    if (i + 1 < end && is_align (rd[i + 1])) {
        spec._M_fill = rd[i];
        spec._M_align = align_of (rd[i + 1]);
        i += 2;
    } else if (is_align (rd[i])) {
        spec._M_align = align_of (rd[i]);
        i += 1;
    }
    if (i < end && (rd[i] == static_cast<CharTy> ('+') || rd[i] == static_cast<CharTy> (' '))) {
        spec._M_sign = rd[i] == static_cast<CharTy> ('+') ? SpecSign::PLUS : SpecSign::SPACE;
        i += 1;
    }
    if (i < end && rd[i] == static_cast<CharTy> ('#')) {
        spec._M_flags = spec._M_flags | SpecFlag::ALT;
        i += 1;
    }
    if (i < end && rd[i] == static_cast<CharTy> ('0')) {
        spec._M_flags = spec._M_flags | SpecFlag::ZERO_PAD;
        i += 1;
    }
    if (i < end && is_digit (rd[i])) {
        unsigned int width = 0;
        while (i < end && is_digit (rd[i])) {
            width = width * 10u + static_cast<unsigned int> (rd[i] - static_cast<CharTy> ('0'));
            if (width > 65535u)
                width = 65535u;
            i += 1;
        }
        spec._M_width = static_cast<unsigned short> (width);
    }
    if (i < end && rd[i] == static_cast<CharTy> ('.')) {
        i += 1;
        if (i >= end || !is_digit (rd[i]))
            return false;
        int precision = 0;
        while (i < end && is_digit (rd[i])) {
            if (precision < 32767)
                precision = precision * 10 + static_cast<int> (rd[i] - static_cast<CharTy> ('0'));
            if (precision > 32767)
                precision = 32767;
            i += 1;
        }
        spec._M_precision = static_cast<short> (precision);
    }
    if (i < end) {
        spec._M_type = rd[i];
        i += 1;
    }
    return i == end;
}

template <typename CharTy>
constexpr void
emit_text (FormatSink<CharTy> &sink, const char *text, FormatSpec<CharTy> spec,
           const SpecAlign default_align) noexcept {
    text_spec (spec);
    const unsigned int len = text_length (text);
    emit_padded (sink, spec, default_align, static_cast<CharTy> ('\0'), len,
                 [&] () noexcept { put_ascii (sink, text, len); });
}

template <typename CharTy>
constexpr void
emit_char (FormatSink<CharTy> &sink, const CharTy ch, FormatSpec<CharTy> spec) noexcept {
    text_spec (spec);
    emit_padded (sink, spec, SpecAlign::LHS, static_cast<CharTy> ('\0'), 1u, [&] () noexcept { sink.put (ch); });
}

template <typename CharTy>
constexpr void
emit_text_content (FormatSink<CharTy> &sink, const BasicStringView<CharTy> text, FormatSpec<CharTy> spec) noexcept {
    unsigned int len = static_cast<unsigned int> (text.size ());
    if (len != 0 && text[len - 1] == static_cast<CharTy> ('\0'))
        --len;
    if (spec._M_precision >= 0 && static_cast<unsigned int> (spec._M_precision) < len)
        len = static_cast<unsigned int> (spec._M_precision);
    text_spec (spec);
    emit_padded (sink, spec, SpecAlign::LHS, static_cast<CharTy> ('\0'), len, [&] () noexcept {
        for (unsigned int i = 0; i < len; ++i)
            sink.put (static_cast<CharTy> (text[i]));
    });
}

struct FloatRepr {
    char _M_digits[64] = {};
    unsigned int _M_ndigits = 0;
    int _M_exp10 = 0;
    bool _M_negative = false;
    FloatKind _M_kind = FloatKind::NUMBER;
};

constexpr FloatRepr
parse_float_repr (const char *text, const unsigned int len) noexcept {
    FloatRepr repr;
    unsigned int i = 0;
    if (i < len && (text[i] == '-' || text[i] == '+')) {
        repr._M_negative = text[i] == '-';
        i += 1;
    }
    if (i < len && (text[i] == 'i' || text[i] == 'I')) {
        repr._M_kind = FloatKind::INFINITY;
        return repr;
    }
    if (i < len && (text[i] == 'n' || text[i] == 'N')) {
        repr._M_kind = FloatKind::NOT_A_NUMBER;
        return repr;
    }
    int exp = 0;
    bool has_exp = false;
    unsigned int pre_dot = 0;
    bool seen_dot = false;
    unsigned int nd = 0;
    for (; i < len; ++i) {
        const char c = text[i];
        if (c >= '0' && c <= '9') {
            if (!seen_dot)
                pre_dot += 1;
            if (nd < 63) {
                repr._M_digits[nd] = c;
                nd += 1;
            }
        } else if (c == '.') {
            seen_dot = true;
        } else if (c == 'e' || c == 'E') {
            has_exp = true;
            i += 1;
            bool neg = false;
            if (i < len && (text[i] == '-' || text[i] == '+')) {
                neg = text[i] == '-';
                i += 1;
            }
            int e = 0;
            for (; i < len && text[i] >= '0' && text[i] <= '9'; ++i) {
                if (e < 100000)
                    e = e * 10 + (text[i] - '0');
            }
            exp = neg ? -e : e;
            break;
        }
    }
    repr._M_ndigits = nd;
    repr._M_exp10 = (has_exp ? exp : 0) + static_cast<int> (pre_dot) - 1;
    unsigned int start = 0;
    while (start + 1 < repr._M_ndigits && repr._M_digits[start] == '0')
        start += 1;
    if (start != 0) {
        repr._M_exp10 -= static_cast<int> (start);
        for (unsigned int k = 0; k + start < repr._M_ndigits; ++k)
            repr._M_digits[k] = repr._M_digits[k + start];
        repr._M_ndigits -= start;
    }
    bool all_zero = true;
    for (unsigned int k = 0; k < repr._M_ndigits; ++k) {
        if (repr._M_digits[k] != '0')
            all_zero = false;
    }
    if (repr._M_ndigits == 0 || all_zero) {
        repr._M_digits[0] = '0';
        repr._M_ndigits = 1;
        repr._M_exp10 = 0;
    }
    return repr;
}

constexpr void
round_float_repr (FloatRepr &repr, const int keep) noexcept {
    if (keep < 0) {
        repr._M_digits[0] = '0';
        repr._M_ndigits = 1;
        repr._M_exp10 = 0;
        return;
    }
    if (keep == 0) {
        if (repr._M_digits[0] >= '5') {
            repr._M_digits[0] = '1';
            repr._M_exp10 += 1;
        } else {
            repr._M_digits[0] = '0';
            repr._M_exp10 = 0;
        }
        repr._M_ndigits = 1;
        return;
    }
    if (static_cast<int> (repr._M_ndigits) <= keep)
        return;
    const unsigned int k = static_cast<unsigned int> (keep);
    if (repr._M_digits[k] >= '5') {
        int i = static_cast<int> (k) - 1;
        for (; i >= 0; --i) {
            if (repr._M_digits[i] != '9') {
                repr._M_digits[i] = static_cast<char> (repr._M_digits[i] + 1);
                break;
            }
            repr._M_digits[i] = '0';
        }
        if (i < 0) {
            repr._M_digits[0] = '1';
            repr._M_exp10 += 1;
        }
    }
    repr._M_ndigits = k;
}

constexpr void
pad_float_repr (FloatRepr &repr, const unsigned int count) noexcept {
    while (repr._M_ndigits < count && repr._M_ndigits < 63) {
        repr._M_digits[repr._M_ndigits] = '0';
        repr._M_ndigits += 1;
    }
}

constexpr void
strip_float_repr (FloatRepr &repr) noexcept {
    while (repr._M_ndigits > 1 && repr._M_digits[repr._M_ndigits - 1] == '0')
        repr._M_ndigits -= 1;
}

template <typename CharTy>
constexpr void
emit_float_fixed (FormatSink<CharTy> &sink, const FormatSpec<CharTy> &spec, const CharTy sign_ch, const FloatRepr &repr,
                  const unsigned int frac_target, const bool alt) noexcept {
    const int e = repr._M_exp10;
    const unsigned int int_len = e >= 0 ? static_cast<unsigned int> (e) + 1u : 1u;
    const unsigned int frac_len = (frac_target > 0u || alt) ? frac_target + 1u : 0u;
    const unsigned int body_len = int_len + frac_len;
    emit_padded (sink, spec, SpecAlign::RHS, sign_ch, body_len, [&] () noexcept {
        if (e >= 0) {
            const unsigned int int_digits = static_cast<unsigned int> (e) + 1u;
            for (unsigned int i = 0; i < int_digits; ++i)
                sink.put (i < repr._M_ndigits ? static_cast<CharTy> (repr._M_digits[i]) : static_cast<CharTy> ('0'));
        } else {
            sink.put (static_cast<CharTy> ('0'));
        }
        if (frac_len != 0u) {
            sink.put (static_cast<CharTy> ('.'));
            for (unsigned int k = 1u; k <= frac_target; ++k) {
                const int idx = e + static_cast<int> (k);
                const bool inside = idx >= 0 && static_cast<unsigned int> (idx) < repr._M_ndigits;
                sink.put (inside ? static_cast<CharTy> (repr._M_digits[idx]) : static_cast<CharTy> ('0'));
            }
        }
    });
}

template <typename CharTy>
constexpr void
emit_float_sci (FormatSink<CharTy> &sink, const FormatSpec<CharTy> &spec, const CharTy sign_ch, const FloatRepr &repr,
                const unsigned int frac_count, const bool alt, const CharTy exp_char) noexcept {
    char exp_digits[8];
    unsigned int exp_len = 0;
    const bool exp_negative = repr._M_exp10 < 0;
    unsigned int exp_abs =
        exp_negative ? static_cast<unsigned int> (-repr._M_exp10) : static_cast<unsigned int> (repr._M_exp10);
    do {
        exp_digits[exp_len] = static_cast<char> ('0' + static_cast<int> (exp_abs % 10u));
        exp_len += 1;
        exp_abs /= 10u;
    } while (exp_abs != 0u && exp_len < 8u);
    while (exp_len < 2u) {
        exp_digits[exp_len] = '0';
        exp_len += 1;
    }
    const unsigned int frac_len = (frac_count > 0u || alt) ? frac_count + 1u : 0u;
    const unsigned int body_len = 1u + frac_len + 2u + exp_len;
    emit_padded (sink, spec, SpecAlign::RHS, sign_ch, body_len, [&] () noexcept {
        sink.put (repr._M_ndigits != 0u ? static_cast<CharTy> (repr._M_digits[0]) : static_cast<CharTy> ('0'));
        if (frac_len != 0u) {
            sink.put (static_cast<CharTy> ('.'));
            for (unsigned int k = 1u; k <= frac_count; ++k)
                sink.put (k < repr._M_ndigits ? static_cast<CharTy> (repr._M_digits[k]) : static_cast<CharTy> ('0'));
        }
        sink.put (exp_char);
        sink.put (exp_negative ? static_cast<CharTy> ('-') : static_cast<CharTy> ('+'));
        for (unsigned int i = exp_len; i > 0u; --i)
            sink.put (static_cast<CharTy> (exp_digits[i - 1u]));
    });
}

template <typename CharTy, typename P>
constexpr void
format_pointer (FormatSink<CharTy> &sink, const P ptr, const FormatSpec<CharTy> &spec) noexcept {
    if (ptr == nullptr) {
        emit_text (sink, "nullptr", spec, SpecAlign::RHS);
        return;
    }
    if !consteval {
        if constexpr (std::is_pointer_v<P>) {
            char raw[24];
            const unsigned long long raw_len =
                uitoa (raw, static_cast<unsigned long long> (reinterpret_cast<uintptr_t> (ptr)),
                       NumberBase::hexadecimal, false);
            emit_padded (
                sink, spec, SpecAlign::RHS, static_cast<CharTy> ('\0'), 2u, static_cast<unsigned int> (raw_len),
                [&] () noexcept { put_ascii (sink, "0x", 2u); },
                [&] () noexcept { put_ascii (sink, raw, static_cast<unsigned int> (raw_len)); });
        } else {
            sink._M_stopped = true;
        }
    } else {
        sink._M_stopped = true;
    }
}

template <typename CharTy, typename T>
constexpr void
format_integer (FormatSink<CharTy> &sink, const T value, const FormatSpec<CharTy> &spec) noexcept {
    const CharTy type = spec._M_type;
    if (type == static_cast<CharTy> ('c')) {
        emit_char (sink, static_cast<CharTy> (value), spec);
        return;
    }
    auto base = NumberBase::decimal;
    bool upper = false;
    if (type == static_cast<CharTy> ('x')) {
        base = NumberBase::hexadecimal;
    } else if (type == static_cast<CharTy> ('X')) {
        base = NumberBase::hexadecimal;
        upper = true;
    } else if (type == static_cast<CharTy> ('o')) {
        base = NumberBase::octal;
    } else if (type == static_cast<CharTy> ('b')) {
        base = NumberBase::binary;
    } else if (type == static_cast<CharTy> ('B')) {
        base = NumberBase::binary;
        upper = true;
    } else if (type != static_cast<CharTy> ('\0') && type != static_cast<CharTy> ('d')
               && type != static_cast<CharTy> ('u')) {
        sink._M_stopped = true;
        return;
    }
    bool negative = false;
    unsigned long long magnitude = 0;
    if constexpr (std::is_signed_v<T> && !std::is_same_v<T, bool>) {
        if (value < 0) {
            negative = true;
            magnitude = base != NumberBase::decimal ? static_cast<unsigned long long> (value)
                                                    : 0ull - static_cast<unsigned long long> (value);
        } else {
            magnitude = static_cast<unsigned long long> (value);
        }
    } else {
        magnitude = static_cast<unsigned long long> (value);
    }
    const bool allow_sign = type == static_cast<CharTy> ('\0') || type == static_cast<CharTy> ('d');
    const CharTy sign_ch = allow_sign ? pick_sign (negative, spec) : static_cast<CharTy> ('\0');
    char raw[72];
    const unsigned int digits_len = static_cast<unsigned int> (uitoa (raw, magnitude, base, upper));
    unsigned int body_len = digits_len;
    if (spec._M_precision >= 0 && static_cast<unsigned int> (spec._M_precision) > digits_len)
        body_len = static_cast<unsigned int> (spec._M_precision);
    CharTy prefix[2];
    unsigned int prefix_len = 0;
    if (has_flag (spec._M_flags, SpecFlag::ALT) || type == static_cast<CharTy> ('B')) {
        if (base == NumberBase::hexadecimal) {
            prefix[0] = static_cast<CharTy> ('0');
            prefix[1] = upper ? static_cast<CharTy> ('X') : static_cast<CharTy> ('x');
            prefix_len = 2;
        } else if (base == NumberBase::binary) {
            prefix[0] = static_cast<CharTy> ('0');
            prefix[1] = upper ? static_cast<CharTy> ('B') : static_cast<CharTy> ('b');
            prefix_len = 2;
        } else if (base == NumberBase::octal) {
            const bool has_zero = body_len > digits_len || (digits_len != 0 && raw[0] == '0');
            if (!has_zero) {
                prefix[0] = static_cast<CharTy> ('0');
                prefix_len = 1;
            }
        }
    }
    emit_padded (
        sink, spec, SpecAlign::RHS, sign_ch, prefix_len, body_len,
        [&] () noexcept {
            for (unsigned int i = 0; i < prefix_len; ++i)
                sink.put (prefix[i]);
        },
        [&] () noexcept {
            if (body_len > digits_len)
                sink.fill (body_len - digits_len, static_cast<CharTy> ('0'));
            put_ascii (sink, raw, digits_len);
        });
}

template <typename CharTy, typename T>
constexpr void
format_floating (FormatSink<CharTy> &sink, const T value, const FormatSpec<CharTy> &spec) noexcept {
    const CharTy type = spec._M_type;
    if (!is_floating_type (type)) {
        sink._M_stopped = true;
        return;
    }
    const bool upper =
        type == static_cast<CharTy> ('F') || type == static_cast<CharTy> ('E') || type == static_cast<CharTy> ('G');
    char raw[96];
    unsigned long long raw_len = 0;
    if constexpr (std::is_same_v<T, float>)
        raw_len = ftoa (raw, value);
    else
        raw_len = dtoa (raw, static_cast<double> (value));
    const FloatRepr repr = parse_float_repr (raw, static_cast<unsigned int> (raw_len));
    const CharTy sign_ch = pick_sign (repr._M_negative, spec);
    const bool alt = has_flag (spec._M_flags, SpecFlag::ALT);
    if (repr._M_kind != FloatKind::NUMBER) {
        const char *text = repr._M_kind == FloatKind::INFINITY ? (upper ? "INF" : "inf") : (upper ? "NAN" : "nan");
        const unsigned int text_len = text_length (text);
        emit_padded (sink, spec, SpecAlign::RHS, sign_ch, text_len,
                     [&] () noexcept { put_ascii (sink, text, text_len); });
        return;
    }
    CharTy t = type;
    if (t == static_cast<CharTy> ('\0') && spec._M_precision < 0) {
        const unsigned int body_len = static_cast<unsigned int> (raw_len);
        bool has_dot = false;
        for (unsigned int i = 0; i < body_len; ++i) {
            if (raw[i] == '.')
                has_dot = true;
        }
        const bool add_dot = alt && !has_dot;
        emit_padded (sink, spec, SpecAlign::RHS, sign_ch, body_len + (add_dot ? 2u : 0u), [&] () noexcept {
            for (unsigned int i = 0; i < body_len; ++i) {
                const char c = raw[i];
                sink.put (upper && c >= 'a' && c <= 'z' ? static_cast<CharTy> (c - 'a' + 'A')
                                                        : static_cast<CharTy> (c));
            }
            if (add_dot) {
                sink.put (static_cast<CharTy> ('.'));
                sink.put (static_cast<CharTy> ('0'));
            }
        });
        return;
    }
    if (t == static_cast<CharTy> ('\0'))
        t = static_cast<CharTy> ('g');
    if (t == static_cast<CharTy> ('f') || t == static_cast<CharTy> ('F')) {
        const unsigned int p = spec._M_precision >= 0 ? static_cast<unsigned int> (spec._M_precision) : 6u;
        FloatRepr rounded = repr;
        round_float_repr (rounded, rounded._M_exp10 + static_cast<int> (p) + 1);
        emit_float_fixed (sink, spec, sign_ch, rounded, p, alt);
    } else if (t == static_cast<CharTy> ('e') || t == static_cast<CharTy> ('E')) {
        const unsigned int p = spec._M_precision >= 0 ? static_cast<unsigned int> (spec._M_precision) : 6u;
        FloatRepr rounded = repr;
        round_float_repr (rounded, static_cast<int> (p) + 1);
        pad_float_repr (rounded, p + 1u);
        emit_float_sci (sink, spec, sign_ch, rounded, p, alt,
                        upper ? static_cast<CharTy> ('E') : static_cast<CharTy> ('e'));
    } else {
        unsigned int p = spec._M_precision >= 0 ? static_cast<unsigned int> (spec._M_precision) : 6u;
        if (p == 0u)
            p = 1u;
        FloatRepr rounded = repr;
        round_float_repr (rounded, static_cast<int> (p));
        if (alt)
            pad_float_repr (rounded, p);
        else
            strip_float_repr (rounded);
        if (rounded._M_exp10 < -4 || rounded._M_exp10 >= static_cast<int> (p)) {
            emit_float_sci (sink, spec, sign_ch, rounded, rounded._M_ndigits - 1u, alt,
                            upper ? static_cast<CharTy> ('E') : static_cast<CharTy> ('e'));
        } else {
            const int frac_i = static_cast<int> (rounded._M_ndigits) - rounded._M_exp10 - 1;
            const unsigned int frac = frac_i > 0 ? static_cast<unsigned int> (frac_i) : 0u;
            emit_float_fixed (sink, spec, sign_ch, rounded, frac, alt);
        }
    }
}

template <typename CharTy, typename T>
constexpr void
format_value_impl (FormatSink<CharTy> &sink, const T &value, const FormatSpec<CharTy> &spec) noexcept;

template <typename CharTy, typename T>
constexpr void
format_value_impl (FormatSink<CharTy> &sink, const T &value, const FormatSpec<CharTy> &spec) noexcept {
    using U = std::remove_cvref_t<T>;
    if constexpr (std::is_same_v<U, bool>) {
        const CharTy type = spec._M_type;
        if (type == static_cast<CharTy> ('\0') || type == static_cast<CharTy> ('s')) {
            emit_text (sink, value ? "true" : "false", spec, SpecAlign::LHS);
        } else if (type == static_cast<CharTy> ('c')) {
            emit_char (sink, value ? static_cast<CharTy> ('1') : static_cast<CharTy> ('0'), spec);
        } else if (is_integer_type (type)) {
            format_integer (sink, value, spec);
        } else {
            sink._M_stopped = true;
        }
    } else if constexpr (std::is_enum_v<U>) {
        format_value_impl (sink, static_cast<std::underlying_type_t<U>> (value), spec);
    } else if constexpr (std::is_floating_point_v<U>) {
        format_floating (sink, value, spec);
    } else if constexpr (std::is_same_v<U, std::nullptr_t>) {
        format_pointer (sink, value, spec);
    } else if constexpr (std::is_same_v<U, CharTy>) {
        const CharTy type = spec._M_type;
        if (type == static_cast<CharTy> ('\0') || type == static_cast<CharTy> ('s')
            || type == static_cast<CharTy> ('c')) {
            emit_char (sink, value, spec);
        } else if (is_integer_type (type)) {
            format_integer (sink, value, spec);
        } else {
            sink._M_stopped = true;
        }
    } else if constexpr (std::is_integral_v<U>) {
        if (is_integer_type (spec._M_type))
            format_integer (sink, value, spec);
        else
            sink._M_stopped = true;
    } else if constexpr (requires { BasicStringView<CharTy>{value}; }) {
        if (spec._M_type == static_cast<CharTy> ('\0') || spec._M_type == static_cast<CharTy> ('s')) {
            if constexpr (std::is_pointer_v<U>) {
                if (value == nullptr) {
                    emit_text (sink, "nullptr", spec, SpecAlign::RHS);
                    return;
                }
            }
            emit_text_content (sink, BasicStringView<CharTy>{value}, spec);
        } else {
            sink._M_stopped = true;
        }
    } else if constexpr (std::is_pointer_v<U>) {
        if (spec._M_type == static_cast<CharTy> ('\0') || spec._M_type == static_cast<CharTy> ('p'))
            format_pointer (sink, value, spec);
        else
            sink._M_stopped = true;
    } else {
        sink._M_stopped = true;
    }
}

template <typename CharTy>
constexpr void
emit_format_text (FormatSink<CharTy> &sink, const CharTy *&fmt) noexcept {
    while (*fmt != static_cast<CharTy> ('\0')) {
        if (*fmt == static_cast<CharTy> ('{')) {
            if (*(fmt + 1) == static_cast<CharTy> ('{')) {
                sink.put (static_cast<CharTy> ('{'));
                fmt += 2;
                continue;
            }
            return;
        }
        if (*fmt == static_cast<CharTy> ('}')) {
            if (*(fmt + 1) == static_cast<CharTy> ('}')) {
                sink.put (static_cast<CharTy> ('}'));
                fmt += 2;
                continue;
            }
            sink._M_stopped = true;
            return;
        }
        sink.put (*fmt);
        ++fmt;
    }
}

template <typename CharTy>
constexpr void
parse_impl (FormatSink<CharTy> &sink, const CharTy *fmt) noexcept {
    emit_format_text (sink, fmt);
    if (*fmt != static_cast<CharTy> ('\0'))
        sink._M_stopped = true;
}

template <typename CharTy, typename First, typename... Rest>
constexpr void
parse_impl (FormatSink<CharTy> &sink, const CharTy *fmt, const First &first, const Rest &...rest) noexcept {
    emit_format_text (sink, fmt);
    if (sink._M_stopped)
        return;
    if (*fmt == static_cast<CharTy> ('\0'))
        return;
    if (*fmt != static_cast<CharTy> ('{')) {
        sink._M_stopped = true;
        return;
    }
    const CharTy *close = fmt + 1;
    while (*close != static_cast<CharTy> ('\0') && *close != static_cast<CharTy> ('}'))
        ++close;
    if (*close != static_cast<CharTy> ('}')) {
        sink._M_stopped = true;
        return;
    }
    FormatSpec<CharTy> spec{};
    const BasicStringView<CharTy> field (fmt + 1, static_cast<size_t> (close - (fmt + 1)));
    if (!parse_format_spec (field, 0u, static_cast<unsigned int> (field.size ()), spec)) {
        sink._M_stopped = true;
        return;
    }
    format_value_impl (sink, first, spec);
    if (sink._M_stopped)
        return;
    parse_impl (sink, close + 1, rest...);
}

template <unsigned int DstSize, typename CharTy, typename... Args>
constexpr unsigned int
format_into (CharTy (&dst)[DstSize], const CharTy *fmt, const Args &...args) noexcept {
    FormatSink<CharTy> sink{};
    sink._M_dst = dst;
    sink._M_capacity = DstSize - 1u;
    parse_impl (sink, fmt, args...);
    const unsigned int end = sink._M_pos < DstSize - 1u ? sink._M_pos : DstSize - 1u;
    dst[end] = static_cast<CharTy> ('\0');
    return sink._M_pos;
}

template <unsigned int Pos, unsigned int ArgIdx, const BasicTemplatedStringView Fmt, FormatArg... Args>
struct FormatJob {
    using FmtTy = std::remove_cv_t<decltype (Fmt)>;
    using FmtChar = typename FmtTy::value_type;

    static constexpr FmtChar
    _S_at (const unsigned int index) noexcept {
        return index < Fmt.size () ? Fmt[index] : static_cast<FmtChar> ('\0');
    }

    static constexpr unsigned int
    _S_close_pos () noexcept {
        unsigned int i = Pos + 1;
        while (i < Fmt.size () && Fmt[i] != static_cast<FmtChar> ('}') && Fmt[i] != static_cast<FmtChar> ('\0'))
            ++i;
        return i;
    }

    static constexpr unsigned int
    _S_literal_run () noexcept {
        unsigned int i = Pos;
        while (i < Fmt.size ()) {
            const FmtChar c = Fmt[i];
            if (c == static_cast<FmtChar> ('\0') || c == static_cast<FmtChar> ('{') || c == static_cast<FmtChar> ('}'))
                break;
            ++i;
        }
        return i - Pos;
    }

    template <unsigned int I, typename First, typename... Rest>
    static constexpr decltype (auto)
    _S_nth (const First &first, const Rest &...rest) noexcept {
        if constexpr (I == 0)
            return first;
        else
            return _S_nth<I - 1> (rest...);
    }

    static constexpr auto
    _S_current_arg () noexcept {
        return _S_nth<ArgIdx> (Args...).v;
    }

    static constexpr unsigned int field_close = _S_close_pos ();
    static constexpr bool field_has_args = ArgIdx < sizeof...(Args);
    static constexpr bool field_closed = field_close < Fmt.size () && _S_at (field_close) == static_cast<FmtChar> ('}');
    static constexpr bool field_open = field_has_args && field_closed;

    static constexpr bool
    _S_field_valid () noexcept {
        if (!field_open)
            return false;
        FormatSpec<FmtChar> spec{};
        if (!parse_format_spec (Fmt, Pos + 1, field_close, spec))
            return false;
        FormatSink<FmtChar> sink{};
        format_value_impl (sink, _S_current_arg (), spec);
        return !sink._M_stopped;
    }

    static constexpr unsigned int
    _S_field_size () noexcept {
        FormatSpec<FmtChar> spec{};
        parse_format_spec (Fmt, Pos + 1, field_close, spec);
        FormatSink<FmtChar> sink{};
        format_value_impl (sink, _S_current_arg (), spec);
        return sink._M_pos;
    }

    static constexpr unsigned int
    _S_compute_size () noexcept {
        if constexpr (_S_at (Pos) == static_cast<FmtChar> ('\0')) {
            return 0;
        } else if constexpr (_S_at (Pos) == static_cast<FmtChar> ('{')) {
            if constexpr (_S_at (Pos + 1) == static_cast<FmtChar> ('{')) {
                return 1u + FormatJob<Pos + 2, ArgIdx, Fmt, Args...>::size;
            } else if constexpr (!field_has_args) {
                static_assert (format_argument_error<Fmt, Pos, ArgIdx>, "not enough arguments for format string");
                return 0;
            } else if constexpr (!field_closed) {
                static_assert (format_argument_error<Fmt, Pos, ArgIdx>, "unterminated format replacement field");
                return 0;
            } else if constexpr (!_S_field_valid ()) {
                static_assert (format_argument_error<Fmt, Pos, ArgIdx>,
                               "invalid format specification or unsupported argument");
                return 0;
            } else {
                return _S_field_size () + FormatJob<field_close + 1, ArgIdx + 1, Fmt, Args...>::size;
            }
        } else if constexpr (_S_at (Pos) == static_cast<FmtChar> ('}')) {
            if constexpr (_S_at (Pos + 1) == static_cast<FmtChar> ('}')) {
                return 1u + FormatJob<Pos + 2, ArgIdx, Fmt, Args...>::size;
            } else {
                static_assert (format_argument_error<Fmt, Pos, ArgIdx>, "unmatched '}' in format string");
                return 0;
            }
        } else {
            return _S_literal_run () + FormatJob<Pos + _S_literal_run (), ArgIdx, Fmt, Args...>::size;
        }
    }

    static constexpr void
    _S_write_into (FormatSink<FmtChar> &sink) noexcept {
        if constexpr (_S_at (Pos) == static_cast<FmtChar> ('\0')) {
            return;
        } else if constexpr (_S_at (Pos) == static_cast<FmtChar> ('{')) {
            if constexpr (_S_at (Pos + 1) == static_cast<FmtChar> ('{')) {
                sink.put (static_cast<FmtChar> ('{'));
                FormatJob<Pos + 2, ArgIdx, Fmt, Args...>::_S_write_into (sink);
            } else if constexpr (!field_open) {
                return;
            } else if constexpr (!_S_field_valid ()) {
                return;
            } else {
                FormatSpec<FmtChar> spec{};
                parse_format_spec (Fmt, Pos + 1, field_close, spec);
                format_value_impl (sink, _S_current_arg (), spec);
                FormatJob<field_close + 1, ArgIdx + 1, Fmt, Args...>::_S_write_into (sink);
            }
        } else if constexpr (_S_at (Pos) == static_cast<FmtChar> ('}')) {
            if constexpr (_S_at (Pos + 1) == static_cast<FmtChar> ('}')) {
                sink.put (static_cast<FmtChar> ('}'));
                FormatJob<Pos + 2, ArgIdx, Fmt, Args...>::_S_write_into (sink);
            }
        } else {
            const unsigned int run = _S_literal_run ();
            for (unsigned int i = 0; i < run; ++i)
                sink.put (Fmt[Pos + i]);
            FormatJob<Pos + run, ArgIdx, Fmt, Args...>::_S_write_into (sink);
        }
    }

    static constexpr unsigned int size = _S_compute_size ();
};

template <const BasicTemplatedStringView Fmt, FormatArg... Args>
consteval auto
format () {
    using FmtChar = typename std::remove_cv_t<decltype (Fmt)>::value_type;
    constexpr unsigned int length = FormatJob<0, 0, Fmt, Args...>::size;
    if constexpr (length == 0) {
        return BasicTemplatedStringView<0, FmtChar>{};
    } else {
        FmtChar buffer[length]{};
        FormatSink<FmtChar> sink{};
        sink._M_dst = buffer;
        sink._M_capacity = length;
        FormatJob<0, 0, Fmt, Args...>::_S_write_into (sink);
        return BasicTemplatedStringView<length, FmtChar>{buffer};
    }
}

template <unsigned int DstSize, typename CharTy, typename First, typename... Rest>
constexpr unsigned int
format (CharTy (&dst)[DstSize], const CharTy *fmt, First &&first, Rest &&...rest) noexcept {
    return format_into (dst, fmt, first, rest...);
}

template <unsigned int DstSize, typename CharTy, typename... Args>
constexpr unsigned int
parse (CharTy (&dst)[DstSize], const CharTy *fmt, Args &&...args) noexcept {
    return format_into (dst, fmt, args...);
}

template <unsigned int DstSize, typename CharTy>
constexpr unsigned int
parse (CharTy (&dst)[DstSize], const CharTy *fmt) noexcept {
    return format_into (dst, fmt);
}
} // namespace zzs::base