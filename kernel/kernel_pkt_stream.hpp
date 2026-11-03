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
 * and pushes complete packets into an owned receive queue.  The consumer pops
 * packets until the source signals STREAM_CLOSE or the stream is closed.
 *
 * States: HUNT_MAGIC -> READ_HEADER -> READ_PAYLOAD -> READ_CRC.
 * A bad magic/version/length/CRC resynchronizes by scanning for the next magic.
 *
 * The stream owns its framing buffers and its receive queue as fixed members
 * (RAII): nothing is allocated, nothing is copied, and construction takes the
 * source callback and context so there is no two-phase `init()`.  Copy/move are
 * deleted because the framing position is meaningful state that must not be
 * duplicated.
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

/** @brief Packet stream.  Capacity of the owned receive queue is fixed at
 *         `_S_packet_queue_capacity`. */
class Stream {
    /* In-frame assembly buffers.  The header is kept raw (8 bytes); the payload
       is unstuffed as it arrives so the CRC can be checked over the original
       bytes.  The payload is aligned so `ZZS_ASSUME_ALIGNED` is meaningful. */
    SourceFn _M_source = nullptr;
    void *_M_context = nullptr;
    PacketQueue<_S_packet_queue_capacity> _M_queue;
    _S_FrameState _M_state = _S_FrameState::HUNT_MAGIC;
    _S_PacketHeader _M_header{};
    uint8_t _M_raw_header[_S_packet_header_bytes]{};
    uint32_t _M_raw_header_len = 0u;
    alignas (4) uint8_t _M_payload[_S_packet_max_payload]{};
    uint32_t _M_payload_collected = 0u;
    uint32_t _M_payload_expected = 0u;
    uint8_t _M_crc[_S_packet_crc_bytes]{};
    uint32_t _M_crc_collected = 0u;
    bool _M_escape_pending = false; /* true when the last payload byte was 0x7D */
    bool _M_closed = false;
    bool _M_error = false;

  public:
    /** @brief Construct a stream over a source callback and a context. */
    inline Stream (const SourceFn source, void *const context) noexcept : _M_source (source), _M_context (context) {
    }

    Stream (const Stream &) = delete;
    Stream &
    operator= (const Stream &) = delete;
    Stream (Stream &&) = delete;
    Stream &
    operator= (Stream &&) = delete;

    /** @brief Close the stream; no further bytes are pulled. */
    inline void
    close () noexcept {
        _M_closed = true;
    }

    /** @brief True when the stream has been closed. */
    [[nodiscard]] inline bool
    closed () const noexcept {
        return _M_closed;
    }

    /** @brief True when the stream is in the middle of a frame (not hunting magic). */
    [[nodiscard]] inline bool
    in_frame () const noexcept {
        return _M_state != _S_FrameState::HUNT_MAGIC;
    }

    /** @brief Pop the oldest complete packet from the stream's queue. */
    [[nodiscard]] inline zzs::kernel::Status
    pop (_S_QueuedPacket &out) noexcept {
        return _M_queue.pop (out);
    }

    /**
     * @brief Pump the stream: pull bytes from the source and emit complete packets
     *        into the queue until the source blocks, the stream closes, or an error
     *        occurs.  Returns Status::OK when more data may be available, NOT_READY
     *        when the source is drained, or an error kind.
     */
    [[nodiscard]] inline zzs::kernel::Status
    pump () noexcept {
        if (ZZS_UNLIKELY (_M_error))
            return zzs::kernel::Status::TRUNCATED;
        if (ZZS_UNLIKELY (_M_closed && _M_state == _S_FrameState::HUNT_MAGIC))
            return zzs::kernel::Status::OK;

        while (true) {
            uint8_t byte = 0u;
            const int32_t got = _M_source (&byte, 1u, _M_context);
            if (got <= 0) {
                if (_M_state != _S_FrameState::HUNT_MAGIC)
                    return zzs::kernel::Status::TRUNCATED; /* incomplete frame */
                return _M_closed ? zzs::kernel::Status::OK : zzs::kernel::Status::NOT_READY;
            }

            switch (_M_state) {
            case _S_FrameState::HUNT_MAGIC:
                if (byte == _S_packet_magic) {
                    _M_raw_header[0] = byte;
                    _M_raw_header_len = 1u;
                    _M_state = _S_FrameState::READ_HEADER;
                }
                break;

            case _S_FrameState::READ_HEADER:
                _M_raw_header[_M_raw_header_len++] = byte;
                if (_M_raw_header_len >= _S_packet_header_bytes) {
                    _S_PacketHeader &h = _M_header;
                    h._M_magic = _M_raw_header[0];
                    h._M_version = _M_raw_header[1];
                    h._M_type = _M_raw_header[2];
                    h._M_length_hi = _M_raw_header[3];
                    h._M_length_lo = _M_raw_header[4];
                    h._M_sequence = _M_raw_header[5];
                    h._M_flags = _M_raw_header[6];
                    h._M_reserved = _M_raw_header[7];

                    if (h._M_magic != _S_packet_magic || h._M_version != _S_packet_version) {
                        /* Bad magic or version: resynchronize. */
                        _M_state = _S_FrameState::HUNT_MAGIC;
                        _M_raw_header_len = 0u;
                        _M_escape_pending = false;
                        break;
                    }

                    _M_payload_expected = (static_cast<uint32_t> (h._M_length_hi) << 8) | h._M_length_lo;
                    if (_M_payload_expected > _S_packet_max_payload) {
                        /* Bad length: resynchronize. */
                        _M_state = _S_FrameState::HUNT_MAGIC;
                        _M_raw_header_len = 0u;
                        _M_escape_pending = false;
                        break;
                    }

                    _M_payload_collected = 0u;
                    _M_crc_collected = 0u;
                    _M_escape_pending = false;
                    _M_state = _S_FrameState::READ_PAYLOAD;
                    if (_M_payload_expected == 0u)
                        _M_state = _S_FrameState::READ_CRC;
                }
                break;

            case _S_FrameState::READ_PAYLOAD:
                if (_M_escape_pending) {
                    /* This byte completes the escape sequence. */
                    _M_payload[_M_payload_collected++] = static_cast<uint8_t> (byte ^ 0x20u);
                    _M_escape_pending = false;
                } else if (byte == _S_stuff_escape) {
                    _M_escape_pending = true;
                } else {
                    _M_payload[_M_payload_collected++] = byte;
                }

                if (_M_payload_collected >= _M_payload_expected) {
                    _M_payload_collected = _M_payload_expected;
                    _M_state = _S_FrameState::READ_CRC;
                }
                break;

            case _S_FrameState::READ_CRC:
                _M_crc[_M_crc_collected++] = byte;
                if (_M_crc_collected >= _S_packet_crc_bytes) {
                    if (!verify_crc ()) {
                        /* Bad CRC: resynchronize. */
                        _M_state = _S_FrameState::HUNT_MAGIC;
                        _M_raw_header_len = 0u;
                        _M_escape_pending = false;
                        break;
                    }
                    /* Emit the packet into the queue. */
                    const zzs::kernel::Status st = emit_packet ();
                    if (st != zzs::kernel::Status::OK) {
                        /* Backpressure: resynchronize at the next frame boundary. */
                        _M_state = _S_FrameState::HUNT_MAGIC;
                        _M_raw_header_len = 0u;
                        _M_escape_pending = false;
                        return st;
                    }
                    _M_state = _S_FrameState::HUNT_MAGIC;
                    _M_raw_header_len = 0u;
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
     * source call (`pump`), the caller supplies an already-read buffer and every
     * byte is driven through the state machine in a tight loop.  This is the form
     * to use when a DMA/interrupt layer has filled a block of bytes and the stream
     * should drain it without per-byte callbacks.
     *
     * Returns the number of bytes consumed (always <= size), and a status:
     *   - OK:          all bytes consumed, more data may follow
     *   - TRUNCATED:   the stream hit an error or an incomplete frame at end-of-chunk
     *   - QUEUE_FULL:  the receive queue filled mid-chunk; consumption stopped
     *   - NOT_READY:   nothing consumed (stream closed and already idle)
     */
    [[nodiscard]] inline zzs::kernel::Status
    pump_chunk (const uint8_t *const buf, const uint32_t size, uint32_t *const consumed) noexcept {
        if (ZZS_UNLIKELY (consumed == nullptr))
            return zzs::kernel::Status::OUT_OF_RANGE;
        *consumed = 0u;
        if (ZZS_UNLIKELY (_M_error))
            return zzs::kernel::Status::TRUNCATED;
        if (ZZS_UNLIKELY (_M_closed && _M_state == _S_FrameState::HUNT_MAGIC))
            return zzs::kernel::Status::NOT_READY;
        if (ZZS_UNLIKELY (buf == nullptr || size == 0u))
            return zzs::kernel::Status::OK;

        while (*consumed < size) {
            const uint8_t byte = buf[*consumed];
            ++(*consumed);

            switch (_M_state) {
            case _S_FrameState::HUNT_MAGIC:
                if (byte == _S_packet_magic) {
                    _M_raw_header[0] = byte;
                    _M_raw_header_len = 1u;
                    _M_state = _S_FrameState::READ_HEADER;
                }
                break;

            case _S_FrameState::READ_HEADER:
                _M_raw_header[_M_raw_header_len++] = byte;
                if (_M_raw_header_len >= _S_packet_header_bytes) {
                    _S_PacketHeader &h = _M_header;
                    h._M_magic = _M_raw_header[0];
                    h._M_version = _M_raw_header[1];
                    h._M_type = _M_raw_header[2];
                    h._M_length_hi = _M_raw_header[3];
                    h._M_length_lo = _M_raw_header[4];
                    h._M_sequence = _M_raw_header[5];
                    h._M_flags = _M_raw_header[6];
                    h._M_reserved = _M_raw_header[7];

                    if (h._M_magic != _S_packet_magic || h._M_version != _S_packet_version) {
                        _M_state = _S_FrameState::HUNT_MAGIC;
                        _M_raw_header_len = 0u;
                        break;
                    }

                    _M_payload_expected = (static_cast<uint32_t> (h._M_length_hi) << 8) | h._M_length_lo;
                    if (_M_payload_expected > _S_packet_max_payload) {
                        _M_state = _S_FrameState::HUNT_MAGIC;
                        _M_raw_header_len = 0u;
                        break;
                    }

                    _M_payload_collected = 0u;
                    _M_crc_collected = 0u;
                    _M_escape_pending = false;
                    _M_state = _S_FrameState::READ_PAYLOAD;
                    if (_M_payload_expected == 0u)
                        _M_state = _S_FrameState::READ_CRC;
                }
                break;

            case _S_FrameState::READ_PAYLOAD:
                if (_M_escape_pending) {
                    _M_payload[_M_payload_collected++] = static_cast<uint8_t> (byte ^ 0x20u);
                    _M_escape_pending = false;
                } else if (byte == _S_stuff_escape) {
                    _M_escape_pending = true;
                } else {
                    _M_payload[_M_payload_collected++] = byte;
                }

                if (_M_payload_collected >= _M_payload_expected) {
                    _M_payload_collected = _M_payload_expected;
                    _M_state = _S_FrameState::READ_CRC;
                }
                break;

            case _S_FrameState::READ_CRC:
                _M_crc[_M_crc_collected++] = byte;
                if (_M_crc_collected >= _S_packet_crc_bytes) {
                    if (!verify_crc ()) {
                        _M_state = _S_FrameState::HUNT_MAGIC;
                        _M_raw_header_len = 0u;
                        _M_escape_pending = false;
                        break;
                    }
                    const zzs::kernel::Status st = emit_packet ();
                    if (st != zzs::kernel::Status::OK) {
                        _M_state = _S_FrameState::HUNT_MAGIC;
                        _M_raw_header_len = 0u;
                        _M_escape_pending = false;
                        return st; /* QUEUE_FULL: stop consuming */
                    }
                    _M_state = _S_FrameState::HUNT_MAGIC;
                    _M_raw_header_len = 0u;
                    _M_escape_pending = false;
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
     * This is the chunked counterpart of `pump`: it issues one source call per
     * pump with a caller-provided scratch buffer (up to `_S_packet_max_frame_bytes`
     * bytes, the worst case between two packet boundaries) instead of one-byte
     * at-a-time callbacks.  `bytes_pulled` reports how many bytes the source
     * actually filled.
     */
    [[nodiscard]] inline zzs::kernel::Status
    pump_bytes (uint8_t *const scratch, const uint32_t scratch_size, uint32_t *const bytes_pulled) noexcept {
        if (ZZS_UNLIKELY (bytes_pulled == nullptr))
            return zzs::kernel::Status::OUT_OF_RANGE;
        *bytes_pulled = 0u;
        if (ZZS_UNLIKELY (_M_error))
            return zzs::kernel::Status::TRUNCATED;
        if (ZZS_UNLIKELY (_M_closed && _M_state == _S_FrameState::HUNT_MAGIC))
            return zzs::kernel::Status::NOT_READY;
        if (ZZS_UNLIKELY (scratch == nullptr || scratch_size == 0u))
            return zzs::kernel::Status::OUT_OF_RANGE;

        const int32_t got = _M_source (scratch, scratch_size, _M_context);
        if (got <= 0) {
            if (_M_state != _S_FrameState::HUNT_MAGIC)
                return zzs::kernel::Status::TRUNCATED;
            return _M_closed ? zzs::kernel::Status::OK : zzs::kernel::Status::NOT_READY;
        }
        *bytes_pulled = static_cast<uint32_t> (got);

        uint32_t consumed = 0u;
        return pump_chunk (scratch, *bytes_pulled, &consumed);
    }

  private:
    /** @brief Push one assembled, CRC-valid packet into the stream's queue. */
    [[nodiscard]] inline zzs::kernel::Status
    emit_packet () noexcept {
        _S_QueuedPacket qp;
        qp._M_header = _M_header;
        qp._M_payload_len = _M_payload_expected;
        /* The unstuffed payload is copied with a builtin hint instead of a hand
           rolled byte loop. */
        uint8_t *const payload = ZZS_ASSUME_ALIGNED (_M_payload, 4);
        ZZS_MEMCPY (qp._M_payload, payload, _M_payload_expected * sizeof (uint8_t));
        qp._M_crc[0] = _M_crc[0];
        qp._M_crc[1] = _M_crc[1];
        qp._M_valid = true;
        return _M_queue.push (qp);
    }

    /** @brief Verify the CRC-16 over the raw header plus the unstuffed payload. */
    [[nodiscard]] inline bool
    verify_crc () const noexcept {
        const uint16_t computed = zzs::kernel::crc::crc16_ccitt (_M_raw_header, _S_packet_header_bytes, 0xFFFFu);
        /* Fold the payload into the same running CRC. */
        uint16_t crc = computed;
        for (uint32_t i = 0; i < _M_payload_expected; ++i)
            crc = zzs::kernel::crc::crc16_ccitt_byte (crc, _M_payload[i]);
        const uint16_t received = (static_cast<uint16_t> (_M_crc[0]) << 8) | static_cast<uint16_t> (_M_crc[1]);
        return crc == received;
    }
};

/** @} */
} // namespace zzs::kernel::pkt
