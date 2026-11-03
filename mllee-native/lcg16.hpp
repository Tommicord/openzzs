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

#include "mllee-native/q11_q15_q31.hpp"

#include <cstdint>

/**
 * @brief Compile-time pseudo-random number generator for @mllee-native/.
 *
 * A 16-bit LCG (Linear Congruential Generator) with Hull-Dobell parameters,
 * fully usable in consteval contexts.  No std:: dependencies.  Produces
 * deterministic streams from a 16-bit seed for reproducible dataset generation.
 * @{
 */
namespace zzs::mllee_native::rng {

/** @brief 16-bit LCG state. */
struct Lcg16 {
    uint16_t _M_state;

    [[nodiscard]] static constexpr Lcg16
    seed (uint16_t s) noexcept {
        /* Ensure state is never 0 (LCG degenerate). */
        return Lcg16{static_cast<uint16_t> (s == 0 ? 0xACE1u : s)};
    }

    /** @brief Advance and return next 16-bit value. */
    [[nodiscard]] constexpr uint16_t
    next () noexcept {
        /* Numerical Recipes 16-bit LCG: state = 1664525 * state + 1013904223 (mod 2^16).
           Truncated mod 2^16: multiplier 0x09D5 (but 1664525 mod 65536 = 13669 = 0x357D),
           increment 1013904223 mod 65536 = 23453 = 0x5B85. */
        _M_state = static_cast<uint16_t> (static_cast<uint32_t> (_M_state) * 0x357Du + 0x5B85u);
        return _M_state;
    }

    /** @brief Return value in [0, 1) as Q15. */
    [[nodiscard]] constexpr zzs::mllee_native::fx::Q15
    next_q15 () noexcept {
        return zzs::mllee_native::fx::Q15{static_cast<int16_t> (next ())};
    }

    /** @brief Return value in [0, 1) as Q31. */
    [[nodiscard]] constexpr zzs::mllee_native::fx::Q31
    next_q31 () noexcept {
        return zzs::mllee_native::fx::Q31{static_cast<int32_t> (static_cast<uint32_t> (next ()) << 16)};
    }

    /** @brief Return integer in [lo, hi]. */
    [[nodiscard]] constexpr uint16_t
    range (uint16_t lo, uint16_t hi) noexcept {
        return lo + (next () % (hi - lo + 1u));
    }
};

/** @brief Xorshift16 alternative. */
struct XorShift16 {
    uint16_t _M_state;

    [[nodiscard]] static constexpr XorShift16
    seed (uint16_t s) noexcept {
        return XorShift16{static_cast<uint16_t> (s == 0 ? 0xBEEFu : s)};
    }

    [[nodiscard]] constexpr uint16_t
    next () noexcept {
        uint16_t x = _M_state;
        x ^= static_cast<uint16_t> (x << 7);
        x ^= static_cast<uint16_t> (x >> 9);
        x ^= static_cast<uint16_t> (x << 8);
        _M_state = x;
        return x;
    }

    [[nodiscard]] constexpr zzs::mllee_native::fx::Q15
    next_q15 () noexcept {
        return zzs::mllee_native::fx::Q15{static_cast<int16_t> (next ())};
    }
};

/** @} */
} // namespace zzs::mllee_native::rng
