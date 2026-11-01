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
#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief Fixed-depth receive queue for decoded packets.
 *
 * Power-of-two capacity (_S_packet_queue_capacity = 8) so the wrap mask is free.
 * The decoder pushes fully-decoded packets here; the consumer pops them in order.
 * QUEUE_FULL is reported when the consumer falls behind. The caller owns the
 * packet storage buffer.
 * @{
 */
namespace zzs::kernel::pkt {

/** @brief One stored packet: header plus stuffed payload plus CRC. */
struct _S_QueuedPacket {
        _S_PacketHeader _M_header;
        uint8_t _M_payload[_S_packet_max_payload];
        uint8_t _M_crc[_S_packet_crc_bytes];
        uint32_t _M_payload_len;
        bool _M_valid;
};

/** @brief Queue state; caller owns the packet storage array. */
struct _S_PacketQueue {
        _S_QueuedPacket *_M_buf; /* caller-owned array of _M_capacity packets */
        uint32_t _M_capacity;    /* power of two */
        uint32_t _M_head;        /* push index */
        uint32_t _M_tail;        /* pop index */
        uint32_t _M_count;       /* entries in queue */
};

/** @brief Initialize over caller-owned storage. Capacity must be pow2. */
[[nodiscard]] inline zzs::kernel::Status
queue_init (_S_PacketQueue &queue, _S_QueuedPacket *const buf, const uint32_t capacity) noexcept {
        if (buf == nullptr || capacity == 0u || (capacity & (capacity - 1u)) != 0u)
                return zzs::kernel::Status::OUT_OF_RANGE;
        queue._M_buf = buf;
        queue._M_capacity = capacity;
        queue._M_head = 0;
        queue._M_tail = 0;
        queue._M_count = 0;
        return zzs::kernel::Status::OK;
}

/** @brief Push a decoded packet into the queue. Returns QUEUE_FULL if full. */
[[nodiscard]] inline zzs::kernel::Status
queue_push (_S_PacketQueue &queue, const _S_QueuedPacket &packet) noexcept {
        if (queue._M_count >= queue._M_capacity)
                return zzs::kernel::Status::QUEUE_FULL;
        queue._M_buf[queue._M_head] = packet;
        queue._M_buf[queue._M_head]._M_valid = true;
        queue._M_head = (queue._M_head + 1u) & (queue._M_capacity - 1u);
        ++queue._M_count;
        return zzs::kernel::Status::OK;
}

/** @brief Pop the oldest packet. Returns QUEUE_FULL/... actually EMPTY via NOT_READY. */
[[nodiscard]] inline zzs::kernel::Status
queue_pop (_S_PacketQueue &queue, _S_QueuedPacket &out) noexcept {
        if (queue._M_count == 0u)
                return zzs::kernel::Status::NOT_READY;
        out = queue._M_buf[queue._M_tail];
        queue._M_buf[queue._M_tail]._M_valid = false;
        queue._M_tail = (queue._M_tail + 1u) & (queue._M_capacity - 1u);
        --queue._M_count;
        return zzs::kernel::Status::OK;
}

/** @brief Entries currently in the queue. */
[[nodiscard]] inline uint32_t
queue_count (const _S_PacketQueue &queue) noexcept {
        return queue._M_count;
}

/** @brief True when the queue is empty. */
[[nodiscard]] inline bool
queue_empty (const _S_PacketQueue &queue) noexcept {
        return queue._M_count == 0u;
}

/** @brief True when the queue is full. */
[[nodiscard]] inline bool
queue_full (const _S_PacketQueue &queue) noexcept {
        return queue._M_count >= queue._M_capacity;
}

/** @} */
} // namespace zzs::kernel::pkt
