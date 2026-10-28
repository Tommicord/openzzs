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

#ifndef ZZS_MLLEE_HOST
#define ZZS_MLLEE_HOST
#endif

#include "mllee/quant.hpp"
#include "mllee/simd.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <vector>

namespace
{
[[nodiscard]] std::uint32_t
next_random (std::uint32_t &state) noexcept
{
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

[[nodiscard]] std::int32_t
reference_i4 (const std::vector<std::uint8_t> &packed, const std::vector<std::int8_t> &rhs,
              const std::uint32_t codes, const std::int8_t *lut) noexcept
{
    std::int32_t total = 0;
    for (std::uint32_t i = 0; i < codes; ++i)
        {
            const std::uint8_t byte = packed[i >> 1];
            const std::uint8_t code = (i & 1u) != 0u ? (byte >> 4) : (byte & 0x0Fu);
            const std::int32_t weight
                = lut != nullptr ? lut[code] : zzs::mllee::_S_from_int4 (code);
            total += weight * rhs[i];
        }
    return total;
}

[[nodiscard]] std::int32_t
reference_i2 (const std::vector<std::uint8_t> &packed, const std::vector<std::int8_t> &rhs,
              const std::uint32_t codes, const std::int8_t *lut) noexcept
{
    std::int32_t total = 0;
    for (std::uint32_t i = 0; i < codes; ++i)
        {
            const std::uint8_t code = (packed[i >> 2] >> ((i & 3u) * 2u)) & 0x03u;
            const std::int32_t weight
                = lut != nullptr ? lut[code] : zzs::mllee::_S_from_int2 (code);
            total += weight * rhs[i];
        }
    return total;
}

[[nodiscard]] std::int32_t
reference_i3 (const std::vector<std::uint8_t> &packed, const std::vector<std::int8_t> &rhs,
              const std::uint32_t codes, const std::int8_t *lut) noexcept
{
    std::int32_t total = 0;
    for (std::uint32_t i = 0; i < codes; ++i)
        {
            const std::uint8_t code
                = static_cast<std::uint8_t> (zzs::mllee::_S_code_at (packed.data (), i, 3));
            const std::int32_t weight
                = lut != nullptr ? lut[code] : zzs::mllee::_S_from_int3 (code);
            total += weight * rhs[i];
        }
    return total;
}
} // namespace

TEST_CASE ("packed int4 dot matches the scalar reference", "[mllee][simd][packed]")
{
    std::uint32_t state = 0x12345678u;
    std::int8_t lut[16];
    for (auto &value : lut)
        value = static_cast<std::int8_t> (static_cast<std::int32_t> (next_random (state) % 255u) - 127);

    for (std::uint32_t codes = 0; codes <= 130; ++codes)
        {
            std::vector<std::uint8_t> packed ((codes + 1u) / 2u);
            std::vector<std::int8_t> rhs (codes);
            for (auto &byte : packed)
                byte = static_cast<std::uint8_t> (next_random (state));
            for (auto &value : rhs)
                value = static_cast<std::int8_t> (static_cast<std::int32_t> (next_random (state) % 255u) - 127);

            REQUIRE (zzs::mllee::simd::dot_i4 (packed.data (), rhs.data (), codes, nullptr)
                     == reference_i4 (packed, rhs, codes, nullptr));
            REQUIRE (zzs::mllee::simd::dot_i4 (packed.data (), rhs.data (), codes, lut)
                     == reference_i4 (packed, rhs, codes, lut));
        }
}

TEST_CASE ("packed int2 dot matches the scalar reference", "[mllee][simd][packed]")
{
    std::uint32_t state = 0x9E3779B9u;
    std::int8_t lut[4];
    for (auto &value : lut)
        value = static_cast<std::int8_t> (static_cast<std::int32_t> (next_random (state) % 255u) - 127);

    for (std::uint32_t codes = 0; codes <= 260; ++codes)
        {
            std::vector<std::uint8_t> packed ((codes + 3u) / 4u);
            std::vector<std::int8_t> rhs (codes);
            for (auto &byte : packed)
                byte = static_cast<std::uint8_t> (next_random (state));
            for (auto &value : rhs)
                value = static_cast<std::int8_t> (static_cast<std::int32_t> (next_random (state) % 255u) - 127);

            REQUIRE (zzs::mllee::simd::dot_i2 (packed.data (), rhs.data (), codes, nullptr)
                     == reference_i2 (packed, rhs, codes, nullptr));
            REQUIRE (zzs::mllee::simd::dot_i2 (packed.data (), rhs.data (), codes, lut)
                     == reference_i2 (packed, rhs, codes, lut));
        }
}

TEST_CASE ("packed int3 dot matches the scalar reference", "[mllee][simd][packed]")
{
    std::uint32_t state = 0xDEADBEEFu;
    std::int8_t lut[8];
    for (auto &value : lut)
        value = static_cast<std::int8_t> (static_cast<std::int32_t> (next_random (state) % 255u) - 127);

    for (std::uint32_t codes = 0; codes <= 200; ++codes)
        {
            const std::uint32_t bytes = (codes * 3u + 7u) / 8u;
            std::vector<std::uint8_t> packed (bytes);
            std::vector<std::int8_t> rhs (codes);
            for (auto &byte : packed)
                byte = static_cast<std::uint8_t> (next_random (state));
            for (auto &value : rhs)
                value = static_cast<std::int8_t> (static_cast<std::int32_t> (next_random (state) % 255u) - 127);

            REQUIRE (zzs::mllee::simd::dot_i3 (packed.data (), rhs.data (), codes, nullptr)
                     == reference_i3 (packed, rhs, codes, nullptr));
            REQUIRE (zzs::mllee::simd::dot_i3 (packed.data (), rhs.data (), codes, lut)
                     == reference_i3 (packed, rhs, codes, lut));
        }
}

TEST_CASE ("packed expansion agrees with the bit-level reader", "[mllee][simd][packed]")
{
    std::uint32_t state = 0xC001D00Du;
    for (std::uint32_t groups = 0; groups <= 32; ++groups)
        {
            const std::uint32_t codes = groups * 8u;
            const std::uint32_t bytes = codes * 3u / 8u;
            std::vector<std::uint8_t> packed (bytes + 1u);
            for (auto &byte : packed)
                byte = static_cast<std::uint8_t> (next_random (state));
            std::vector<std::int8_t> expanded (codes + 8u);

            zzs::mllee::simd::expand_codes3 (packed.data (), expanded.data (), codes, nullptr);
            for (std::uint32_t i = 0; i < codes; ++i)
                REQUIRE (expanded[i]
                         == zzs::mllee::_S_from_int3 (static_cast<std::uint8_t> (
                             zzs::mllee::_S_code_at (packed.data (), i, 3))));

            std::int8_t lut[8];
            for (auto &value : lut)
                value = static_cast<std::int8_t> (next_random (state) & 0x7Fu);
            zzs::mllee::simd::expand_codes3 (packed.data (), expanded.data (), codes, lut);
            for (std::uint32_t i = 0; i < codes; ++i)
                REQUIRE (expanded[i]
                         == lut[zzs::mllee::_S_code_at (packed.data (), i, 3)]);

            const std::uint32_t quad_bytes = (groups * 8u + 3u) / 4u;
            std::vector<std::uint8_t> packed2 (quad_bytes + 1u);
            for (auto &byte : packed2)
                byte = static_cast<std::uint8_t> (next_random (state));
            zzs::mllee::simd::expand_codes2 (packed2.data (), expanded.data (), quad_bytes,
                                             nullptr);
            for (std::uint32_t i = 0; i < quad_bytes * 4u; ++i)
                REQUIRE (expanded[i]
                         == zzs::mllee::_S_from_int2 (static_cast<std::uint8_t> (
                             (packed2[i >> 2] >> ((i & 3u) * 2u)) & 3u)));
        }
}

TEST_CASE ("packed dot stays exact over the full int8 range", "[mllee][simd][packed]")
{
    constexpr std::uint32_t codes = 4096;
    std::vector<std::uint8_t> packed4 (codes / 2u, 0x88u);
    std::vector<std::int8_t> rhs (codes, -128);

    const std::int32_t expected
        = static_cast<std::int32_t> (codes) * (-8) * (-128);
    REQUIRE (zzs::mllee::simd::dot_i4 (packed4.data (), rhs.data (), codes, nullptr)
             == expected);

    std::vector<std::uint8_t> packed2 (codes / 4u, 0xFFu);
    const std::int32_t expected2 = static_cast<std::int32_t> (codes) * (-1) * (-128);
    REQUIRE (zzs::mllee::simd::dot_i2 (packed2.data (), rhs.data (), codes, nullptr)
             == expected2);
}
