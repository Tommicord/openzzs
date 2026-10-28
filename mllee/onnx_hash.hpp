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

#include "std/string_view.hpp"

#include <cstddef>
#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief FNV-1a and djb2 hashes used to identify protobuf strings.
 *
 * The loader never retains a name, path or operator spelling.  It hashes the
 * bytes while they stream past and keeps two independent 32-bit digests for
 * operator names; the registry compares both digests plus the length, so a
 * single FNV collision cannot dispatch the wrong kernel.  djb2 is not used on
 * the hot path for arbitrary strings, only for the short NodeProto header
 * fields.  Both functions are constexpr so the registry in onnx_ops.hpp is a
 * compile-time table in .rodata.
 */
inline constexpr uint32_t _S_fnv_offset_basis = 2166136261u;
inline constexpr uint32_t _S_fnv_prime = 16777619u;
inline constexpr uint32_t _S_djb2_basis = 5381u;

[[nodiscard]] constexpr uint32_t
fnv1a_update (uint32_t hash, const uint8_t *const data,
              const uint32_t length) noexcept
{
    for (uint32_t i = 0; i < length; ++i)
        {
            hash ^= data[i];
            hash *= _S_fnv_prime;
        }
    return hash;
}

[[nodiscard]] constexpr uint32_t
djb2_update (uint32_t hash, const uint8_t *const data,
             const uint32_t length) noexcept
{
    for (uint32_t i = 0; i < length; ++i)
        hash = (hash * 33u) ^ data[i];
    return hash;
}

[[nodiscard]] constexpr uint32_t
fnv1a (const uint8_t *const data, const uint32_t length) noexcept
{
    return fnv1a_update (_S_fnv_offset_basis, data, length);
}

[[nodiscard]] constexpr uint32_t
djb2 (const uint8_t *const data, const uint32_t length) noexcept
{
    return djb2_update (_S_djb2_basis, data, length);
}

[[nodiscard]] constexpr uint32_t
fnv1a (const char *const data, const uint32_t length) noexcept
{
    uint32_t hash = _S_fnv_offset_basis;
    for (uint32_t i = 0; i < length; ++i)
        {
            hash ^= static_cast<uint8_t> (data[i]);
            hash *= _S_fnv_prime;
        }
    return hash;
}

[[nodiscard]] constexpr uint32_t
djb2 (const char *const data, const uint32_t length) noexcept
{
    uint32_t hash = _S_djb2_basis;
    for (uint32_t i = 0; i < length; ++i)
        hash = (hash * 33u) ^ static_cast<uint8_t> (data[i]);
    return hash;
}

template <unsigned int N>
[[nodiscard]] constexpr uint32_t
fnv1a_literal (const char (&text)[N]) noexcept
{
    return fnv1a (text, static_cast<uint32_t> (N - 1u));
}

template <unsigned int N>
[[nodiscard]] constexpr uint32_t
djb2_literal (const char (&text)[N]) noexcept
{
    return djb2 (text, static_cast<uint32_t> (N - 1u));
}

[[nodiscard]] inline uint32_t
fnv1a (const zzs::base::SimpleStringView text) noexcept
{
    return fnv1a (text.data (), static_cast<uint32_t> (text.size ()));
}

[[nodiscard]] inline uint32_t
djb2 (const zzs::base::SimpleStringView text) noexcept
{
    return djb2 (text.data (), static_cast<uint32_t> (text.size ()));
}
} // namespace zzs::mllee::onnx
