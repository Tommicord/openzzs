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

#include "mllee/status.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief Overflow-checked arithmetic used by every length/shape computation.
 *
 * ONNX lengths are signed 64-bit values and dimensions are products of signed
 * 64-bit values.  A malicious model can put INT64_MAX in both, so each step is
 * checked before it is allowed to influence a buffer offset.  The helpers are
 * constexpr and lower to the target's carry/overflow flags where available.
 */
[[nodiscard]] constexpr bool
add_overflow_u32 (const uint32_t lhs, const uint32_t rhs,
                  uint32_t &out) noexcept
{
    if (rhs > 0xFFFFFFFFu - lhs)
        return true;
    out = lhs + rhs;
    return false;
}

[[nodiscard]] constexpr bool
add_overflow_u64 (const uint64_t lhs, const uint64_t rhs,
                  uint64_t &out) noexcept
{
    if (rhs > 0xFFFFFFFFFFFFFFFFull - lhs)
        return true;
    out = lhs + rhs;
    return false;
}

[[nodiscard]] constexpr bool
mul_overflow_u64 (const uint64_t lhs, const uint64_t rhs,
                  uint64_t &out) noexcept
{
    if (lhs != 0u && rhs > 0xFFFFFFFFFFFFFFFFull / lhs)
        return true;
    out = lhs * rhs;
    return false;
}

[[nodiscard]] constexpr uint32_t
saturate_u32 (const uint64_t value) noexcept
{
    return value > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t> (value);
}

[[nodiscard]] constexpr uint16_t
saturate_u16 (const uint32_t value) noexcept
{
    return value > 0xFFFFu ? 0xFFFFu : static_cast<uint16_t> (value);
}

[[nodiscard]] constexpr uint8_t
saturate_u8 (const uint32_t value) noexcept
{
    return value > 0xFFu ? 0xFFu : static_cast<uint8_t> (value);
}

/** @brief Saturating increment that also reports that it stuck at the maximum. */
template <typename T>
[[nodiscard]] constexpr bool
saturating_increment (T &value, const T maximum) noexcept
{
    if (value == maximum)
        return false;
    ++value;
    return true;
}

/** @brief Convert a signed wire integer to a non-negative 32-bit length. */
[[nodiscard]] constexpr bool
non_negative_u32 (const int64_t value, uint32_t &out) noexcept
{
    if (value < 0)
        return false;
    if (static_cast<uint64_t> (value) > 0xFFFFFFFFull)
        return false;
    out = static_cast<uint32_t> (value);
    return true;
}

/**
 * @brief Byte length of a packed tensor, rounded up to a whole byte.
 *
 * `bits` is 2, 4, 6, 8, ... and `elements` is already overflow-checked.  The
 * addition before the division cannot overflow because elements is bounded by
 * `_S_max_tensor_elements` at the call site.
 */
[[nodiscard]] constexpr bool
packed_bytes (const uint8_t bits, const uint64_t elements,
              uint64_t &out) noexcept
{
    if (bits == 0u)
        return false;
    uint64_t product = 0;
    if (mul_overflow_u64 (elements, bits, product))
        return false;
    if (add_overflow_u64 (product, 7u, product))
        return false;
    out = product / 8u;
    return true;
}
} // namespace zzs::mllee::onnx
