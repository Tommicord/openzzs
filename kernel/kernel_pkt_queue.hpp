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
 * Power-of-two capacity so the wrap mask is free.  The decoder pushes fully
 * decoded packets here; the consumer pops them in order.  QUEUE_FULL is
 * reported when the consumer falls behind.
 *
 * The queue owns its packet storage as a fixed member array (RAII): nothing is
 * allocated and nothing is copied.  Capacity is a compile-time configuration
 * expressed as a template argument.  Copy/move are deleted because the head /
 * tail position is meaningful state that must not be duplicated.
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

/** @brief Fixed-capacity receive queue owning `_M_buf[Capacity]` packets. */
template <uint32_t Capacity> class PacketQueue {
  static_assert (Capacity != 0u && (Capacity & (Capacity - 1u)) == 0u,
                 "PacketQueue capacity must be a non-zero power of two");

  _S_QueuedPacket _M_buf[Capacity]; /* owned packet storage */
  uint32_t _M_head = 0u;            /* push index */
  uint32_t _M_tail = 0u;            /* pop index */
  uint32_t _M_count = 0u;           /* entries in queue */

public:
  PacketQueue () = default;
  PacketQueue (const PacketQueue &) = delete;
  PacketQueue &
  operator= (const PacketQueue &) = delete;
  PacketQueue (PacketQueue &&) = delete;
  PacketQueue &
  operator= (PacketQueue &&) = delete;

  /** @brief Capacity of the queue (packets). */
  [[nodiscard]] inline uint32_t
  capacity () const noexcept {
    return Capacity;
  }

  /** @brief Entries currently in the queue. */
  [[nodiscard]] inline uint32_t
  count () const noexcept {
    return _M_count;
  }

  /** @brief True when the queue is empty. */
  [[nodiscard]] inline bool
  empty () const noexcept {
    return _M_count == 0u;
  }

  /** @brief True when the queue is full. */
  [[nodiscard]] inline bool
  full () const noexcept {
    return _M_count >= Capacity;
  }

  /** @brief Push a decoded packet into the queue. Returns QUEUE_FULL if full. */
  [[nodiscard]] inline zzs::kernel::Status
  push (const _S_QueuedPacket &packet) noexcept {
    if (ZZS_UNLIKELY (_M_count >= Capacity))
      return zzs::kernel::Status::QUEUE_FULL;
    _M_buf[_M_head] = packet;
    _M_buf[_M_head]._M_valid = true;
    _M_head = (_M_head + 1u) & (Capacity - 1u);
    ++_M_count;
    return zzs::kernel::Status::OK;
  }

  /** @brief Pop the oldest packet. Returns NOT_READY when empty. */
  [[nodiscard]] inline zzs::kernel::Status
  pop (_S_QueuedPacket &out) noexcept {
    if (ZZS_UNLIKELY (_M_count == 0u))
      return zzs::kernel::Status::NOT_READY;
    out = _M_buf[_M_tail];
    _M_buf[_M_tail]._M_valid = false;
    _M_tail = (_M_tail + 1u) & (Capacity - 1u);
    --_M_count;
    return zzs::kernel::Status::OK;
  }
};

/** @} */
} // namespace zzs::kernel::pkt
