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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

#pragma once

#include <cstdint>

/**
 * @brief NMEA 0183 sentence checksum (8-bit XOR).
 *
 * The NMEA checksum is the XOR of all bytes between '$' (or '!') and '*'
 * (exclusive), rendered as two uppercase hex digits after '*'.  Integer-only,
 * constexpr-friendly, no heap.
 * @{
 */
namespace zzs::kernel::device::driver::gnss::nmea {

/** @brief Start-of-sentence marker. */
inline constexpr char NmeaStartDollar = '$';
/** @brief Alternate start-of-sentence marker (some receivers use '!'). */
inline constexpr char NmeaStartBang = '!';
/** @brief Checksum delimiter. */
inline constexpr char NmeaChecksumSep = '*';
/** @brief End-of-sentence markers (CR/LF). */
inline constexpr char NmeaEndCr = '\r';
inline constexpr char NmeaEndLf = '\n';

/** @brief Compute the NMEA XOR checksum over `[first, last)`. */
[[nodiscard]] inline uint8_t
nmea_checksum (const char *first, const char *last) noexcept {
    uint8_t x = 0u;
    while (first != last) {
        x = static_cast<uint8_t> (x ^ static_cast<uint8_t> (*first));
        ++first;
    }
    return x;
}

/** @brief Compute the NMEA XOR checksum over a fixed buffer of `len` bytes. */
[[nodiscard]] inline uint8_t
nmea_checksum_buf (const uint8_t *buf, uint16_t len) noexcept {
    uint8_t x = 0u;
    for (uint16_t i = 0u; i < len; ++i)
        x = static_cast<uint8_t> (x ^ buf[i]);
    return x;
}

/** @brief Convert a hex nibble char to its value, or 0xff when invalid. */
[[nodiscard]] inline uint8_t
hex_nibble (char c) noexcept {
    if (c >= '0' && c <= '9')
        return static_cast<uint8_t> (c - '0');
    if (c >= 'A' && c <= 'F')
        return static_cast<uint8_t> (c - 'A' + 10u);
    if (c >= 'a' && c <= 'f')
        return static_cast<uint8_t> (c - 'a' + 10u);
    return 0xffu;
}

/** @brief Parse two hex digits starting at `p` into `*out`; returns true on
 *         success. */
[[nodiscard]] inline bool
parse_hex2 (const char *p, uint8_t &out) noexcept {
    uint8_t hi = hex_nibble (p[0]);
    uint8_t lo = hex_nibble (p[1]);
    if (hi == 0xffu || lo == 0xffu)
        return false;
    out = static_cast<uint8_t> ((hi << 4) | lo);
    return true;
}

/** @} */
} // namespace zzs::kernel::device::driver::gnss::nmea
