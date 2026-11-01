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
#include "kernel/kernel_util_crc.hpp"

#include <cstdint>

/**
 * @brief Packet framing: magic / header / control bytes / payload / CRC-16 with
 *        byte stuffing inside the payload.
 *
 * Wire format (one frame):
 *   [MAGIC 0x5A][VERSION][TYPE][LENGTH hi][LENGTH lo][SEQ][FLAGS][RESERVED]  <- 8-byte header
 *   [payload ...]  <- up to _S_packet_max_payload bytes, byte-stuffed
 *   [CRC16 hi][CRC16 lo]  <- CRC-16/CCITT over header+payload
 *
 * The header carries the layout and control bytes the receiver uses to decode
 * the payload (feature row, sensor sample, config, or stream control).  Only
 * hex magic numbers are used and they are documented here.
 * @{
 */
namespace zzs::kernel::pkt
{

/** @brief Frame magic byte (0x5A = OpenZZS kernel). Documented, not arbitrary. */
inline constexpr uint8_t _S_packet_magic = 0x5Au;

/** @brief Protocol version this build decodes. */
inline constexpr uint8_t _S_packet_version = 1u;

/** @brief Packet types the kernel can decode. */
enum class Type : uint8_t
{
    SAMPLE = 0x01,   /**< One resampled grid-tick sensor sample */
    FEATURE_ROW = 0x02, /**< One packed int8 feature row */
    CONFIG = 0x03,   /**< Runtime configuration (rate, gain, mains grid) */
    STREAM_OPEN = 0x04, /**< Open a packet stream */
    STREAM_CLOSE = 0x05, /**< Close a packet stream */
    STREAM_ACK = 0x06,   /**< Stream acknowledgment */
    KEEPALIVE = 0x07,  /**< Stream keepalive */
};

/** @brief Header flags. */
enum class Flags : uint8_t
{
    NONE = 0x00,
    STUFFED = 0x01, /**< Payload contains stuffed bytes */
    COMPRESSED = 0x02, /**< Payload is packlow-compressed */
    LAST = 0x80,    /**< Last packet of a message */
};

/** @brief Fixed packet header, 8 bytes. */
struct _S_PacketHeader
{
    uint8_t _M_magic;       /* 0x5A */
    uint8_t _M_version;     /* 1 */
    uint8_t _M_type;        /* Type */
    uint8_t _M_length_hi;   /* payload length high */
    uint8_t _M_length_lo;   /* payload length low */
    uint8_t _M_sequence;    /* packet sequence */
    uint8_t _M_flags;       /* Flags */
    uint8_t _M_reserved;    /* reserved, 0 */
};

/** @brief Byte-stuffing escape markers (only inside payload). */
inline constexpr uint8_t _S_stuff_escape = 0x7Du;
inline constexpr uint8_t _S_stuff_flag = 0x7Eu;

/** @brief Stuff one byte into `out`, returning bytes written (1 or 2).
 *
 *  The escape byte (0x7D), the flag byte (0x7E) and the magic byte (0x5A) are
 *  escaped so they never appear raw inside a stuffed payload; this keeps the
 *  receiver's magic scan free of false positives during resynchronization.  An
 *  escaped byte `b` is written as [0x7D, b ^ 0x20].
 */
[[nodiscard]] inline uint32_t
stuff_byte (const uint8_t byte, uint8_t *const out) noexcept
{
    if (byte == _S_stuff_flag || byte == _S_stuff_escape || byte == _S_packet_magic)
        {
            out[0] = _S_stuff_escape;
            out[1] = static_cast<uint8_t> (byte ^ 0x20u);
            return 2u;
        }
    out[0] = byte;
    return 1u;
}

/** @brief Unstuff one byte from `in`; advances `in` by 1 or 2.
 *
 *  Returns false on malformed input (escape with no following byte).  `consumed`
 *  is set to the number of source bytes consumed (1 or 2).
 */
[[nodiscard]] inline bool
unstuff_byte (const uint8_t *const in, const uint32_t in_remaining, uint8_t &out,
              uint32_t &consumed) noexcept
{
    if (in_remaining == 0u)
        return false;
    if (in[0] == _S_stuff_escape)
        {
            if (in_remaining < 2u)
                return false;
            out = static_cast<uint8_t> (in[1] ^ 0x20u);
            consumed = 2u;
            return true;
        }
    out = in[0];
    consumed = 1u;
    return true;
}

/** @brief Header length in bytes. */
[[nodiscard]] inline constexpr uint32_t
header_bytes (void) noexcept
{
    return zzs::kernel::_S_packet_header_bytes;
}

/** @brief CRC length in bytes. */
[[nodiscard]] inline constexpr uint32_t
crc_bytes (void) noexcept
{
    return zzs::kernel::_S_packet_crc_bytes;
}

/** @brief Max payload in bytes. */
[[nodiscard]] inline constexpr uint32_t
max_payload (void) noexcept
{
    return zzs::kernel::_S_packet_max_payload;
}

/** @} */
} // namespace zzs::kernel::pkt
