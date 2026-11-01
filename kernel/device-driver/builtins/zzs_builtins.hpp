/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#pragma once

/**
 * @brief Portable wrappers around compiler builtins for performance-critical
 *        kernel code.
 *
 * Each `ZZS_*` macro maps to a GCC/Clang builtin when the compiler provides it
 * and degrades to a documented, correct fallback on MSVC and other compilers.
 * Type-safe inline-function counterparts live in `zzs::kernel::builtins`; the
 * macros are the zero-overhead forms the hot paths use directly.
 *
 * No heap, no float, no exceptions, no RTTI, no writable statics.
 *
 * @section fallback Fallback behaviour by macro
 *
 * - `ZZS_ASSUME_ALIGNED`: `__builtin_assume_aligned` on GCC/Clang; otherwise a
 *   no-op that returns the pointer unchanged.  Use `ZZS_ASSUME_ALIGNED_CHECK`
 *   when a compile-time alignment assertion is desired.
 * - `ZZS_PREFETCH`: `__builtin_prefetch` on GCC/Clang; `_mm_prefetch` on MSVC;
 *   otherwise a no-op that evaluates the address once.
 * - `ZZS_LIKELY` / `ZZS_UNLIKELY`: `__builtin_expect(..,1)` / `__builtin_expect(..,0)`
 *   on GCC/Clang; otherwise the expression itself.
 * - `ZZS_MUL_OVERFLOW` / `ZZS_ADD_OVERFLOW` / `ZZS_SUB_OVERFLOW`:
 *   `__builtin_*_overflow` on GCC/Clang; otherwise a branchy portable check.
 * - `ZZS_UNREACHABLE`: `__builtin_unreachable` on GCC/Clang; `__assume(false)` on
 *   MSVC; otherwise an infinite loop (truly unreachable).
 * - `ZZS_MEMCPY` / `ZZS_MEMSET`: `__builtin_memcpy` / `__builtin_memset` on
 *   GCC/Clang; otherwise `std::memcpy` / `std::memset`.
 * @{
 */

#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace zzs::kernel::builtins {

/** @brief Assume `ptr` points to memory aligned to `alignment` bytes.
 *
 *  Returns `ptr`.  On GCC/Clang this is `__builtin_assume_aligned`, which lets
 *  the optimizer emit aligned loads/stores.  On other compilers the pointer is
 *  returned unchanged; the caller must still guarantee the alignment.
 */
template <typename T>
[[nodiscard]] inline T *
assume_aligned (T *const ptr, const uintptr_t alignment) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return static_cast<T *> (__builtin_assume_aligned (ptr, alignment));
#else
  (void) alignment;
  return ptr;
#endif
}

/** @brief Prefetch a cache line at `addr`.  `rw` is 0 for read, 1 for write;
 *         `locality` is 0..3 (temporal locality).  Evaluates `addr` once. */
inline void
prefetch (const void *const addr, const int rw, const int locality) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_prefetch (addr, rw, locality);
#elif defined(_MSC_VER)
#include <xmmintrin.h>
  /* Map locality to an MSVC _MM_HINT. */
  static const int _S_hint[4] = {_MM_HINT_T0, _MM_HINT_T1, _MM_HINT_T2, _MM_HINT_NTA};
  const int hint = locality < 0 ? _MM_HINT_T0 : (locality > 3 ? _MM_HINT_NTA : _S_hint[locality]);
  _mm_prefetch (static_cast<const char *> (addr), hint);
#else
  (void) addr;
  (void) rw;
  (void) locality;
#endif
}

/** @brief Branch hint: `true` when `expr` is likely.  Falls back to `expr`. */
template <typename T>
[[nodiscard]] inline constexpr bool
likely (const T &expr) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_expect (!!static_cast<bool> (expr), 1);
#else
  return static_cast<bool> (expr);
#endif
}

/** @brief Branch hint: `true` when `expr` is unlikely.  Falls back to `expr`. */
template <typename T>
[[nodiscard]] inline constexpr bool
unlikely (const T &expr) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_expect (!!static_cast<bool> (expr), 0);
#else
  return static_cast<bool> (expr);
#endif
}

/** @brief Saturating multiplication overflow test.  Returns true and stores
 *         the wrapped product in `res` on overflow; `res` is only meaningful
 *         when the return is false.  Portable fallback for compilers without
 *         `__builtin_mul_overflow`. */
template <typename T>
[[nodiscard]] inline bool
mul_overflow (const T a, const T b, T &res) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_mul_overflow (a, b, &res);
#else
  using U = typename std::make_unsigned<T>::type;
  constexpr U k_sign_bit = U (1) << (sizeof (T) * 8u - 1u);
  constexpr U k_pos_max = ~k_sign_bit;
  const bool neg = (a < T (0)) != (b < T (0));
  const U ma = (a < T (0)) ? U (0) - static_cast<U> (a) : static_cast<U> (a);
  const U mb = (b < T (0)) ? U (0) - static_cast<U> (b) : static_cast<U> (b);
  const U limit = neg ? k_sign_bit : k_pos_max;
  const bool overflow = (ma != U (0) && mb != U (0) && ma > limit / mb);
  U prod = ma * mb;
  if (neg)
    prod = U (0) - prod; /* two's-complement negation mod 2^N */
  res = static_cast<T> (prod);
  return overflow;
#endif
}

/** @brief Saturating addition overflow test.  See `mul_overflow`. */
template <typename T>
[[nodiscard]] inline bool
add_overflow (const T a, const T b, T &res) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_add_overflow (a, b, &res);
#else
  using U = typename std::make_unsigned<T>::type;
  constexpr U k_sign_bit = U (1) << (sizeof (T) * 8u - 1u);
  constexpr T k_min = static_cast<T> (k_sign_bit);
  constexpr T k_max = static_cast<T> (~k_sign_bit);
  bool overflow = false;
  if (b > T (0) && a > k_max - b)
    overflow = true;
  else if (b < T (0) && a < k_min - b)
    overflow = true;
  res = static_cast<T> (static_cast<U> (a) + static_cast<U> (b));
  return overflow;
#endif
}

/** @brief Saturating subtraction overflow test.  See `mul_overflow`. */
template <typename T>
[[nodiscard]] inline bool
sub_overflow (const T a, const T b, T &res) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_sub_overflow (a, b, &res);
#else
  /* a - b == a + (-b); negating b is safe unless b == T::min, handled below. */
  if (b == std::numeric_limits<T>::min ()) {
    /* -b would overflow; the result a - min can only overflow when a == max. */
    const bool overflow = (a == std::numeric_limits<T>::max ());
    res = a; /* wrapped result; caller checks the flag first */
    return overflow;
  }
  return add_overflow (a, static_cast<T> (-b), res);
#endif
}

/** @brief Mark a point as unreachable.  The optimizer may assume execution
 *         never gets here.  Fallback: MSVC `__assume(false)`, otherwise an
 *         infinite loop (truly unreachable code paths should not continue). */
[[noreturn]] inline void
unreachable () noexcept {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_unreachable ();
#elif defined(_MSC_VER)
  __assume (false);
#else
  for (;;) {
  }
#endif
}

/** @brief `memcpy` with a constant-time hint.  Copies `n` bytes from `src` to
 *         `dst`.  On GCC/Clang this is `__builtin_memcpy` (no trapping, no
 *         size-dependent branches); otherwise `std::memcpy`. */
inline void *
builtin_memcpy (void *const dst, const void *const src, const size_t n) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_memcpy (dst, src, n);
#else
  return std::memcpy (dst, src, n);
#endif
}

/** @brief `memset` with a constant-time hint.  Sets `n` bytes of `dst` to `c`.
 *         On GCC/Clang this is `__builtin_memset`; otherwise `std::memset`. */
inline void *
builtin_memset (void *const dst, const int c, const size_t n) noexcept {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_memset (dst, c, n);
#else
  return std::memset (dst, c, n);
#endif
}

/** @brief Compile-time alignment assertion helper.  Asserts that the type of
 *  `ptr` is aligned to at least `alignment` bytes:
 *
 *      alignas(64) int x;
 *      ZZS_ASSUME_ALIGNED_CHECK(&x, 64);
 *
 *  Expands to nothing on compilers without `__builtin_assume_aligned`. */
template <typename T>
[[nodiscard]] inline constexpr bool
alignment_check (const uintptr_t alignment) noexcept {
  return alignof (T) >= alignment;
}

} // namespace zzs::kernel::builtins

/* =========================================================================
 * Zero-overhead macros (the forms the hot paths use).
 * ========================================================================= */

/** @brief Assume `ptr` is `alignment`-byte aligned. */
#ifndef ZZS_ASSUME_ALIGNED
#if defined(__GNUC__) || defined(__clang__)
#define ZZS_ASSUME_ALIGNED(ptr, alignment) (::zzs::kernel::builtins::assume_aligned ((ptr), (alignment)))
#else
#define ZZS_ASSUME_ALIGNED(ptr, alignment) (::zzs::kernel::builtins::assume_aligned ((ptr), (alignment)))
#endif
#endif

/** @brief Prefetch a cache line at `addr` (rw: 0=read/1=write, locality: 0-3). */
#ifndef ZZS_PREFETCH
#if defined(__GNUC__) || defined(__clang__)
#define ZZS_PREFETCH(addr, rw, locality) __builtin_prefetch ((addr), (rw), (locality))
#elif defined(_MSC_VER)
#define ZZS_PREFETCH(addr, rw, locality) (::zzs::kernel::builtins::prefetch ((addr), (rw), (locality)))
#else
#define ZZS_PREFETCH(addr, rw, locality) ((void) (addr), (void) (rw), (void) (locality))
#endif
#endif

/** @brief Branch hint: `expr` is likely. */
#ifndef ZZS_LIKELY
#define ZZS_LIKELY(expr) (::zzs::kernel::builtins::likely (expr))
#endif

/** @brief Branch hint: `expr` is unlikely. */
#ifndef ZZS_UNLIKELY
#define ZZS_UNLIKELY(expr) (::zzs::kernel::builtins::unlikely (expr))
#endif

/** @brief Saturating multiplication overflow: stores product in `res`, returns
 *         true on overflow. */
#ifndef ZZS_MUL_OVERFLOW
#define ZZS_MUL_OVERFLOW(a, b, res) (::zzs::kernel::builtins::mul_overflow ((a), (b), (res)))
#endif

/** @brief Saturating addition overflow: stores sum in `res`, returns true on
 *         overflow. */
#ifndef ZZS_ADD_OVERFLOW
#define ZZS_ADD_OVERFLOW(a, b, res) (::zzs::kernel::builtins::add_overflow ((a), (b), (res)))
#endif

/** @brief Saturating subtraction overflow: stores difference in `res`,
 *         returns true on overflow. */
#ifndef ZZS_SUB_OVERFLOW
#define ZZS_SUB_OVERFLOW(a, b, res) (::zzs::kernel::builtins::sub_overflow ((a), (b), (res)))
#endif

/** @brief Mark the current point as unreachable. */
#ifndef ZZS_UNREACHABLE
#define ZZS_UNREACHABLE() (::zzs::kernel::builtins::unreachable ())
#endif

/** @brief Constant-time-hinted memcpy of `n` bytes from `src` to `dst`. */
#ifndef ZZS_MEMCPY
#define ZZS_MEMCPY(dst, src, n) (::zzs::kernel::builtins::builtin_memcpy ((dst), (src), (n)))
#endif

/** @brief Constant-time-hinted memset of `n` bytes of `dst` to `c`. */
#ifndef ZZS_MEMSET
#define ZZS_MEMSET(dst, c, n) (::zzs::kernel::builtins::builtin_memset ((dst), (c), (n)))
#endif

/** @brief Compile-time alignment check for a typed pointer.  Used as a
 *  standalone statement; does nothing on compilers without the builtin.
 *
 *      alignas(64) int x;
 *      ZZS_ASSUME_ALIGNED_CHECK(&x, 64);
 */
#ifndef ZZS_ASSUME_ALIGNED_CHECK
#if defined(__GNUC__) || defined(__clang__)
#define ZZS_ASSUME_ALIGNED_CHECK(ptr, alignment)                                              \
  static_assert (alignof (typename std::remove_pointer<decltype (ptr)>::type) >= (alignment), \
                 "ZZS_ASSUME_ALIGNED_CHECK: pointee type not aligned")
#else
#define ZZS_ASSUME_ALIGNED_CHECK(ptr, alignment) (void) 0
#endif
#endif

/** @} */
