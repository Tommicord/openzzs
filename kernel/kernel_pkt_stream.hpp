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
#include "kernel/kernel_pkt_frame.hpp"
#include "kernel/kernel_pkt_queue.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief Packet stream: open a stream, receive frames until closed.
 *
 * The stream is the main packet communication path.  A source callback supplies
 * raw bytes; the stream runs a framing state machine that hunts for the magic
 * byte, decodes the 8-byte header, unstuffs the payload, verifies the CRC-16,
 * and pushes complete packets into a receive queue.  The consumer pops packets
 * until the source signals STREAM_CLOSE or the stream is closed.
 *
 * States: HUNT_MAGIC -> READ_HEADER -> READ_PAYLOAD -> READ_CRC.
 * A bad magic/version/length/CRC resynchronizes by scanning for the next magic.
 * @{
 */
namespace zzs::kernel::pkt {

/** @brief Byte source callback: fills `buf` with up to `size` bytes; returns
 *         bytes written, 0 when no data is available right now, or a negative
 *         value when the source is closed. */
using SourceFn = int32_t (*) (uint8_t *const buf, const uint32_t size, void *const context) noexcept;

/** @brief Framing state machine states. */
enum class _S_FrameState : uint8_t {
        HUNT_MAGIC = 0,
        READ_HEADER = 1,
        READ_PAYLOAD = 2,
        READ_CRC = 3,
};

/** @brief Stream state. */
struct _S_Stream {
        SourceFn _M_source;
        void *_M_context;
        _S_PacketQueue *_M_queue;
        _S_FrameState _M_state;
        _S_PacketHeader _M_header;
        /* In-frame assembly buffers.  The header is kept raw (8 bytes); the payload
           is unstuffed as it arrives so the CRC can be checked over the original
           bytes. */
        uint8_t _M_raw_header[_S_packet_header_bytes];
        uint32_t _M_raw_header_len;
        uint8_t _M_payload[_S_packet_max_payload];
        uint32_t _M_payload_collected;
        uint32_t _M_payload_expected;
        uint8_t _M_crc[_S_packet_crc_bytes];
        uint32_t _M_crc_collected;
        bool _M_escape_pending; /* true when the last payload byte was 0x7D */
        bool _M_closed;
        bool _M_error;
};

/** @brief Initialize a stream over a source callback and a receive queue. */
[[nodiscard]] inline zzs::kernel::Status
stream_init (_S_Stream &stream, const SourceFn source, void *const context, _S_PacketQueue &queue) noexcept {
        if (source == nullptr)
                return zzs::kernel::Status::OUT_OF_RANGE;
        stream._M_source = source;
        stream._M_context = context;
        stream._M_queue = &queue;
        stream._M_state = _S_FrameState::HUNT_MAGIC;
        stream._M_raw_header_len = 0u;
        stream._M_payload_collected = 0u;
        stream._M_payload_expected = 0u;
        stream._M_crc_collected = 0u;
        stream._M_escape_pending = false;
        stream._M_closed = false;
        stream._M_error = false;
        return zzs::kernel::Status::OK;
}

/** @brief Close the stream; no further bytes are pulled. */
inline void
stream_close (_S_Stream &stream) noexcept {
        stream._M_closed = true;
}

/** @brief True when the stream has been closed. */
[[nodiscard]] inline bool
stream_closed (const _S_Stream &stream) noexcept {
        return stream._M_closed;
}

/** @brief True when the stream is in the middle of a frame (not hunting magic). */
[[nodiscard]] inline bool
stream_in_frame (const _S_Stream &stream) noexcept {
        return stream._M_state != _S_FrameState::HUNT_MAGIC;
}

/** @brief Push one assembled, CRC-valid packet into the stream's queue. */
[[nodiscard]] inline zzs::kernel::Status
stream_emit_packet (_S_Stream &stream) noexcept {
        _S_QueuedPacket qp;
        qp._M_header = stream._M_header;
        qp._M_payload_len = stream._M_payload_expected;
        for (uint32_t i = 0; i < stream._M_payload_expected; ++i)
                qp._M_payload[i] = stream._M_payload[i];
        qp._M_crc[0] = stream._M_crc[0];
        qp._M_crc[1] = stream._M_crc[1];
        qp._M_valid = true;
        return queue_push (*stream._M_queue, qp);
}

/** @brief Verify the CRC-16 over the raw header plus the unstuffed payload. */
[[nodiscard]] inline bool
stream_verify_crc (const _S_Stream &stream) noexcept {
        const uint16_t computed = zzs::kernel::crc::crc16_ccitt (stream._M_raw_header, _S_packet_header_bytes, 0xFFFFu);
        /* Fold the payload into the same running CRC. */
        uint16_t crc = computed;
        for (uint32_t i = 0; i < stream._M_payload_expected; ++i)
                crc = zzs::kernel::crc::crc16_ccitt_byte (crc, stream._M_payload[i]);
        const uint16_t received =
            (static_cast<uint16_t> (stream._M_crc[0]) << 8) | static_cast<uint16_t> (stream._M_crc[1]);
        return crc == received;
}

/**
 * @brief Pump the stream: pull bytes from the source and emit complete packets
 *        into the queue until the source blocks, the stream closes, or an error
 *        occurs.  Returns Status::OK when more data may be available, NOT_READY
 *        when the source is drained, or an error kind.
 */
[[nodiscard]] inline zzs::kernel::Status
stream_pump (_S_Stream &stream) noexcept {
        if (stream._M_error)
                return zzs::kernel::Status::TRUNCATED;
        if (stream._M_closed && stream._M_state == _S_FrameState::HUNT_MAGIC)
                return zzs::kernel::Status::OK;

        while (true) {
                uint8_t byte = 0u;
                const int32_t got = stream._M_source (&byte, 1u, stream._M_context);
                if (got <= 0) {
                        if (stream._M_state != _S_FrameState::HUNT_MAGIC)
                                return zzs::kernel::Status::TRUNCATED; /* incomplete frame */
                        return stream._M_closed ? zzs::kernel::Status::OK : zzs::kernel::Status::NOT_READY;
                }

                switch (stream._M_state) {
                case _S_FrameState::HUNT_MAGIC:
                        if (byte == _S_packet_magic) {
                                stream._M_raw_header[0] = byte;
                                stream._M_raw_header_len = 1u;
                                stream._M_state = _S_FrameState::READ_HEADER;
                        }
                        break;

                case _S_FrameState::READ_HEADER:
                        stream._M_raw_header[stream._M_raw_header_len++] = byte;
                        if (stream._M_raw_header_len >= _S_packet_header_bytes) {
                                _S_PacketHeader &h = stream._M_header;
                                h._M_magic = stream._M_raw_header[0];
                                h._M_version = stream._M_raw_header[1];
                                h._M_type = stream._M_raw_header[2];
                                h._M_length_hi = stream._M_raw_header[3];
                                h._M_length_lo = stream._M_raw_header[4];
                                h._M_sequence = stream._M_raw_header[5];
                                h._M_flags = stream._M_raw_header[6];
                                h._M_reserved = stream._M_raw_header[7];

                                if (h._M_magic != _S_packet_magic || h._M_version != _S_packet_version) {
                                        /* Bad magic or version: resynchronize. */
                                        stream._M_state = _S_FrameState::HUNT_MAGIC;
                                        stream._M_raw_header_len = 0u;
                                        stream._M_escape_pending = false;
                                        break;
                                }

                                stream._M_payload_expected =
                                    (static_cast<uint32_t> (h._M_length_hi) << 8) | h._M_length_lo;
                                if (stream._M_payload_expected > _S_packet_max_payload) {
                                        /* Bad length: resynchronize. */
                                        stream._M_state = _S_FrameState::HUNT_MAGIC;
                                        stream._M_raw_header_len = 0u;
                                        stream._M_escape_pending = false;
                                        break;
                                }

                                stream._M_payload_collected = 0u;
                                stream._M_crc_collected = 0u;
                                stream._M_escape_pending = false;
                                stream._M_state = _S_FrameState::READ_PAYLOAD;
                                if (stream._M_payload_expected == 0u)
                                        stream._M_state = _S_FrameState::READ_CRC;
                        }
                        break;

                case _S_FrameState::READ_PAYLOAD:
                        if (stream._M_escape_pending) {
                                /* This byte completes the escape sequence. */
                                stream._M_payload[stream._M_payload_collected++] = static_cast<uint8_t> (byte ^ 0x20u);
                                stream._M_escape_pending = false;
                        } else if (byte == _S_stuff_escape) {
                                stream._M_escape_pending = true;
                        } else {
                                stream._M_payload[stream._M_payload_collected++] = byte;
                        }

                        if (stream._M_payload_collected >= stream._M_payload_expected) {
                                stream._M_payload_collected = stream._M_payload_expected;
                                stream._M_state = _S_FrameState::READ_CRC;
                        }
                        break;

                case _S_FrameState::READ_CRC:
                        stream._M_crc[stream._M_crc_collected++] = byte;
                        if (stream._M_crc_collected >= _S_packet_crc_bytes) {
                                if (!stream_verify_crc (stream)) {
                                        /* Bad CRC: resynchronize. */
                                        stream._M_state = _S_FrameState::HUNT_MAGIC;
                                        stream._M_raw_header_len = 0u;
                                        stream._M_escape_pending = false;
                                        break;
                                }
                                /* Emit the packet into the queue. */
                                const zzs::kernel::Status st = stream_emit_packet (stream);
                                if (st != zzs::kernel::Status::OK) {
                                        /* Backpressure: resynchronize at the next frame boundary. */
                                        stream._M_state = _S_FrameState::HUNT_MAGIC;
                                        stream._M_raw_header_len = 0u;
                                        stream._M_escape_pending = false;
                                        return st;
                                }
                                stream._M_state = _S_FrameState::HUNT_MAGIC;
                                stream._M_raw_header_len = 0u;
                        }
                        break;
                }
        }
        return zzs::kernel::Status::OK;
}

/**
 * @brief Pump a contiguous chunk of bytes through the framing state machine.
 *
 * This is the data-streaming entry point: instead of pulling one byte per
 * source call (`stream_pump`), the caller supplies an already-read buffer and
 * every byte is driven through the state machine in a tight loop.  This is the
 * form to use when a DMA/interrupt layer has filled a block of bytes and the
 * stream should drain it without per-byte callbacks.
 *
 * Returns the number of bytes consumed (always <= size), and a status:
 *   - OK:          all bytes consumed, more data may follow
 *   - TRUNCATED:   the stream hit an error or an incomplete frame at end-of-chunk
 *   - QUEUE_FULL:  the receive queue filled mid-chunk; consumption stopped
 *   - NOT_READY:   nothing consumed (stream closed and already idle)
 *
 * On QUEUE_FULL or TRUNCATED the stream state is left consistent; the caller
 * can resume with the remaining bytes after draining the queue / resetting.
 */
[[nodiscard]] inline zzs::kernel::Status
stream_pump_chunk (_S_Stream &stream, const uint8_t *const buf, const uint32_t size,
                   uint32_t *const consumed) noexcept {
        if (consumed == nullptr)
                return zzs::kernel::Status::OUT_OF_RANGE;
        *consumed = 0u;
        if (stream._M_error)
                return zzs::kernel::Status::TRUNCATED;
        if (stream._M_closed && stream._M_state == _S_FrameState::HUNT_MAGIC)
                return zzs::kernel::Status::NOT_READY;
        if (buf == nullptr || size == 0u)
                return zzs::kernel::Status::OK;

        while (*consumed < size) {
                const uint8_t byte = buf[*consumed];
                ++(*consumed);

                switch (stream._M_state) {
                case _S_FrameState::HUNT_MAGIC:
                        if (byte == _S_packet_magic) {
                                stream._M_raw_header[0] = byte;
                                stream._M_raw_header_len = 1u;
                                stream._M_state = _S_FrameState::READ_HEADER;
                        }
                        break;

                case _S_FrameState::READ_HEADER:
                        stream._M_raw_header[stream._M_raw_header_len++] = byte;
                        if (stream._M_raw_header_len >= _S_packet_header_bytes) {
                                _S_PacketHeader &h = stream._M_header;
                                h._M_magic = stream._M_raw_header[0];
                                h._M_version = stream._M_raw_header[1];
                                h._M_type = stream._M_raw_header[2];
                                h._M_length_hi = stream._M_raw_header[3];
                                h._M_length_lo = stream._M_raw_header[4];
                                h._M_sequence = stream._M_raw_header[5];
                                h._M_flags = stream._M_raw_header[6];
                                h._M_reserved = stream._M_raw_header[7];

                                if (h._M_magic != _S_packet_magic || h._M_version != _S_packet_version) {
                                        stream._M_state = _S_FrameState::HUNT_MAGIC;
                                        stream._M_raw_header_len = 0u;
                                        break;
                                }

                                stream._M_payload_expected =
                                    (static_cast<uint32_t> (h._M_length_hi) << 8) | h._M_length_lo;
                                if (stream._M_payload_expected > _S_packet_max_payload) {
                                        stream._M_state = _S_FrameState::HUNT_MAGIC;
                                        stream._M_raw_header_len = 0u;
                                        break;
                                }

                                stream._M_payload_collected = 0u;
                                stream._M_crc_collected = 0u;
                                stream._M_escape_pending = false;
                                stream._M_state = _S_FrameState::READ_PAYLOAD;
                                if (stream._M_payload_expected == 0u)
                                        stream._M_state = _S_FrameState::READ_CRC;
                        }
                        break;

                case _S_FrameState::READ_PAYLOAD:
                        if (stream._M_escape_pending) {
                                stream._M_payload[stream._M_payload_collected++] = static_cast<uint8_t> (byte ^ 0x20u);
                                stream._M_escape_pending = false;
                        } else if (byte == _S_stuff_escape) {
                                stream._M_escape_pending = true;
                        } else {
                                stream._M_payload[stream._M_payload_collected++] = byte;
                        }

                        if (stream._M_payload_collected >= stream._M_payload_expected) {
                                stream._M_payload_collected = stream._M_payload_expected;
                                stream._M_state = _S_FrameState::READ_CRC;
                        }
                        break;

                case _S_FrameState::READ_CRC:
                        stream._M_crc[stream._M_crc_collected++] = byte;
                        if (stream._M_crc_collected >= _S_packet_crc_bytes) {
                                if (!stream_verify_crc (stream)) {
                                        stream._M_state = _S_FrameState::HUNT_MAGIC;
                                        stream._M_raw_header_len = 0u;
                                        stream._M_escape_pending = false;
                                        break;
                                }
                                const zzs::kernel::Status st = stream_emit_packet (stream);
                                if (st != zzs::kernel::Status::OK) {
                                        stream._M_state = _S_FrameState::HUNT_MAGIC;
                                        stream._M_raw_header_len = 0u;
                                        stream._M_escape_pending = false;
                                        return st; /* QUEUE_FULL: stop consuming */
                                }
                                stream._M_state = _S_FrameState::HUNT_MAGIC;
                                stream._M_raw_header_len = 0u;
                                stream._M_escape_pending = false;
                        }
                        break;
                }
        }
        return zzs::kernel::Status::OK;
}

/**
 * @brief Pump the stream: pull a chunk from the source, then drive every byte
 *        through the framing state machine.  Returns when the source blocks,
 *        the stream closes, an error occurs, or the queue fills.
 *
 * This is the chunked counterpart of `stream_pump`: it issues one source call
 * per pump with a caller-provided scratch buffer (up to
 * `_S_packet_max_frame_bytes` bytes, the worst case between two packet
 * boundaries) instead of one-byte-at-a-time callbacks.  `bytes_pulled` reports
 * how many bytes the source actually filled.
 */
[[nodiscard]] inline zzs::kernel::Status
stream_pump_bytes (_S_Stream &stream, uint8_t *const scratch, const uint32_t scratch_size,
                   uint32_t *const bytes_pulled) noexcept {
        if (bytes_pulled == nullptr)
                return zzs::kernel::Status::OUT_OF_RANGE;
        *bytes_pulled = 0u;
        if (stream._M_error)
                return zzs::kernel::Status::TRUNCATED;
        if (stream._M_closed && stream._M_state == _S_FrameState::HUNT_MAGIC)
                return zzs::kernel::Status::NOT_READY;
        if (scratch == nullptr || scratch_size == 0u)
                return zzs::kernel::Status::OUT_OF_RANGE;

        const int32_t got = stream._M_source (scratch, scratch_size, stream._M_context);
        if (got <= 0) {
                if (stream._M_state != _S_FrameState::HUNT_MAGIC)
                        return zzs::kernel::Status::TRUNCATED;
                return stream._M_closed ? zzs::kernel::Status::OK : zzs::kernel::Status::NOT_READY;
        }
        *bytes_pulled = static_cast<uint32_t> (got);

        uint32_t consumed = 0u;
        return stream_pump_chunk (stream, scratch, *bytes_pulled, &consumed);
}

/** @brief Pop the oldest complete packet from the stream's queue. */
[[nodiscard]] inline zzs::kernel::Status
stream_pop (_S_Stream &stream, _S_QueuedPacket &out) noexcept {
        return queue_pop (*stream._M_queue, out);
}

/** @} */
} // namespace zzs::kernel::pkt
