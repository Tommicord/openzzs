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

#include "kernel/kernel_config.hpp"

#include <cstdint>

/**
 * @brief CRC-16/CCITT for the kernel packet layer.
 *
 * Table-based, branch-free, identical on every target (no run-time dispatch,
 * no carry-less multiply ladder).  Used to protect the packet payload; the
 * header magic and version are separate integrity checks.  Polynomial 0x1021,
 * init 0xFFFF, no reflection.
 *
 * No heap, no float, no exceptions, no RTTI.
 * @{
 */
namespace zzs::kernel::crc {

/** @brief CRC-16/CCITT polynomial (x^16 + x^12 + x^5 + 1). */
inline constexpr uint16_t _S_crc16_poly = 0x1021u;

/** @brief Build the CRC-16 lookup table at compile time. */
struct _S_Crc16Table {
    uint16_t _M_table[256];

    constexpr _S_Crc16Table () noexcept : _M_table{} {
        for (uint32_t i = 0; i < 256u; ++i) {
            uint16_t crc = static_cast<uint16_t> (i << 8);
            for (uint32_t bit = 0; bit < 8u; ++bit) {
                const bool top = (crc & 0x8000u) != 0u;
                crc = static_cast<uint16_t> (top ? (crc << 1) ^ _S_crc16_poly : (crc << 1));
            }
            this->_M_table[i] = crc;
        }
    }
};

/** @brief Compile-time CRC-16 lookup table (256 entries). */
inline constexpr _S_Crc16Table _S_crc16_table{};

/** @brief CRC-16/CCITT over `count` bytes starting from `init`. */
[[nodiscard]] inline constexpr uint16_t
crc16_ccitt (const uint8_t *const data, const uint32_t count, const uint16_t init = 0xFFFFu) noexcept {
    uint16_t crc = init;
    for (uint32_t i = 0; i < count; ++i)
        crc = static_cast<uint16_t> ((crc << 8) ^ _S_crc16_table._M_table[((crc >> 8) ^ data[i]) & 0xFFu]);
    return crc;
}

/** @brief CRC-16/CCITT over a single byte. */
[[nodiscard]] inline constexpr uint16_t
crc16_ccitt_byte (const uint16_t crc, const uint8_t byte) noexcept {
    return static_cast<uint16_t> ((crc << 8) ^ _S_crc16_table._M_table[((crc >> 8) ^ byte) & 0xFFu]);
}

/** @} */
} // namespace zzs::kernel::crc
