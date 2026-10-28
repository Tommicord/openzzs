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
 * @brief Inlining controls for the RAM-budgeted path.
 *
 * Nested calls each cost a register-save frame, so the layer parser and the
 * op kernels are kept out of line (a flat chain) while the decode primitives
 * are folded into the codec entry point.
 */
#if defined(__GNUC__) || defined(__clang__)
#define ZZS_MLLEE_NOINLINE __attribute__ ((noinline))
#define ZZS_MLLEE_INLINE __attribute__ ((always_inline)) inline
#else
#define ZZS_MLLEE_NOINLINE
#define ZZS_MLLEE_INLINE inline
#endif

namespace zzs::mllee
{
/**
 * @brief Result code used by every mllee internal path.
 *
 * The core never throws: errors are values.  The public boundary wraps this
 * code in zzs::base::Maybe<Output, Error>; hot paths return Status directly so
 * no storage beyond a single byte is needed.
 */
struct Error
{
    enum class Kind : uint8_t
    {
        OK = 0,
        TRUNCATED,
        CORRUPT_STREAM,
        BAD_MAGIC,
        BAD_VERSION,
        BAD_LAYOUT,
        OUT_OF_RANGE,
        ARENA_OVERFLOW,
        UNSUPPORTED,
        INTERNAL,
    };

    Error () = delete;
    explicit constexpr Error (const Kind kind) noexcept : _M_kind (kind)
    {
    }

    [[nodiscard]] constexpr Kind
    kind () const noexcept
    {
        return this->_M_kind;
    }

    [[nodiscard]] constexpr bool
    is_ok () const noexcept
    {
        return this->_M_kind == Kind::OK;
    }

    [[nodiscard]] constexpr bool
    operator== (const Error &other) const noexcept
    {
        return this->_M_kind == other._M_kind;
    }

    [[nodiscard]] constexpr explicit
    operator bool () const noexcept
    {
        return this->_M_kind != Kind::OK;
    }

  private:
    Kind _M_kind;
};

using Status = Error::Kind;

[[nodiscard]] constexpr bool
status_ok (const Status status) noexcept
{
    return status == Status::OK;
}

/**
 * @brief CRC-16/CCITT byte table: the residue of each of the 256 possible
 *        eight-bit shifts, generated at compile time.
 *
 * The bit-at-a-time form below it needs eight data-dependent tests per byte,
 * which is the least predictable loop in the module; one load and one xor per
 * byte is the same result with a fixed sequence.  512 bytes of .rodata buys
 * that, and the table is const so it stays out of the RAM budget.
 */
struct Crc16Table
{
    uint16_t _M_bytes[256];

    constexpr Crc16Table () noexcept : _M_bytes ()
    {
        for (uint16_t byte = 0; byte < 256; ++byte)
            {
                uint16_t value = static_cast<uint16_t> (byte << 8);
                for (int bit = 0; bit < 8; ++bit)
                    value = (value & 0x8000u) != 0
                                ? static_cast<uint16_t> ((value << 1) ^ 0x1021u)
                                : static_cast<uint16_t> (value << 1);
                this->_M_bytes[byte] = value;
            }
    }
};

inline constexpr Crc16Table _S_crc16_table{};

/** @brief One table-driven CRC-16/CCITT step (polynomial 0x1021, init 0xFFFF). */
[[nodiscard]] constexpr uint16_t
_S_crc16_update (uint16_t crc, const uint8_t byte) noexcept
{
    return static_cast<uint16_t> ((crc << 8)
                                  ^ _S_crc16_table._M_bytes[((crc >> 8) ^ byte) & 0xFFu]);
}

[[nodiscard]] constexpr uint16_t
_S_crc16 (const uint8_t *data, const uint32_t length) noexcept
{
    uint16_t crc = 0xFFFFu;
    for (uint32_t i = 0; i < length; ++i)
        crc = _S_crc16_update (crc, data[i]);
    return crc;
}

[[nodiscard]] constexpr const char *
status_name (const Status status) noexcept
{
    switch (status)
        {
        case Status::OK:
            return "ok";
        case Status::TRUNCATED:
            return "truncated";
        case Status::CORRUPT_STREAM:
            return "corrupt_stream";
        case Status::BAD_MAGIC:
            return "bad_magic";
        case Status::BAD_VERSION:
            return "bad_version";
        case Status::BAD_LAYOUT:
            return "bad_layout";
        case Status::OUT_OF_RANGE:
            return "out_of_range";
        case Status::ARENA_OVERFLOW:
            return "arena_overflow";
        case Status::UNSUPPORTED:
            return "unsupported";
        case Status::INTERNAL:
            return "internal";
        }
    return "unknown";
}
} // namespace zzs::mllee
