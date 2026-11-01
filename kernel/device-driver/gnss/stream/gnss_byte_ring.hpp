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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHER DEALINGS IN
 * THE SOFTWARE. */

#pragma once

#include "kernel/device-driver/gnss/gnss_types.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>
#include <cstring>

/**
 * @brief Fixed-size byte ring buffer for the GNSS streaming path.
 *
 * Decouples the byte producer (ISR / HAL `read_bytes`) from the protocol
 * parser (NMEA / UBX).  Fixed C array, power-of-two depth for a free mask,
 * drop-oldest policy when full (never blocks, never allocates).  All methods
 * `noexcept`, integer-only.
 * @{
 */
namespace zzs::kernel::device::driver::gnss::stream {

/** @brief Maximum ring depth (power of two). */
inline constexpr uint16_t GnssRingMaxDepth = 256u;

/** @brief Default ring depth. */
inline constexpr uint16_t GnssRingDefaultDepth = 64u;

/** @brief One byte slot. */
using Byte = uint8_t;

/** @brief Fixed-size ring of raw GNSS receiver bytes. */
class GnssByteRing {
public:
  /** @brief Construct with a fixed depth (clamped to [1, GnssRingMaxDepth]). */
  explicit GnssByteRing (uint16_t depth) noexcept
      : _M_depth (clamp_depth (depth)) {
    _M_clear ();
  }

  GnssByteRing (const GnssByteRing &) = delete;
  GnssByteRing &operator= (const GnssByteRing &) = delete;
  GnssByteRing (GnssByteRing &&) = delete;
  GnssByteRing &operator= (GnssByteRing &&) = delete;

  /** @brief Push one byte; overwrites the oldest when full. */
  [[nodiscard]] zzs::kernel::Status
  push (uint8_t b) noexcept {
    _M_buf[_M_head] = b;
    _M_head = static_cast<uint16_t> ((_M_head + 1u) & _M_mask);
    if (_M_count < _M_depth)
      _M_count += 1u;
    else
      _M_tail = static_cast<uint16_t> ((_M_tail + 1u) & _M_mask);
    return zzs::kernel::Status::OK;
  }

  /** @brief Pop the oldest byte into `out`. Returns STALE when empty. */
  [[nodiscard]] zzs::kernel::Status
  pop (uint8_t &out) noexcept {
    if (_M_count == 0u)
      return zzs::kernel::Status::STALE;
    out = _M_buf[_M_tail];
    _M_tail = static_cast<uint16_t> ((_M_tail + 1u) & _M_mask);
    _M_count -= 1u;
    return zzs::kernel::Status::OK;
  }

  /** @brief Peek at `count` bytes starting at `tail` without consuming,
   *         writing them linearly into `dst` (wraps internally). Returns the
   *         number of bytes peeked (<= count, <= filled). */
  [[nodiscard]] uint16_t
  peek (uint8_t *dst, uint16_t count) const noexcept {
    if (count > _M_count)
      count = _M_count;
    uint16_t n = 0u;
    uint16_t idx = _M_tail;
    while (n < count) {
      dst[n++] = _M_buf[idx];
      idx = static_cast<uint16_t> ((idx + 1u) & _M_mask);
    }
    return n;
  }

  /** @brief Consume `count` bytes from the tail. */
  void
  consume (uint16_t count) noexcept {
    if (count > _M_count)
      count = _M_count;
    _M_tail = static_cast<uint16_t> ((_M_tail + count) & _M_mask);
    _M_count = static_cast<uint16_t> (_M_count - count);
  }

  /** @brief Number of bytes currently in the ring. */
  [[nodiscard]] uint16_t
  filled () const noexcept {
    return _M_count;
  }

  [[nodiscard]] bool
  empty () const noexcept {
    return _M_count == 0u;
  }

  [[nodiscard]] bool
  full () const noexcept {
    return _M_count == _M_depth;
  }

  /** @brief Reset the ring to empty. */
  void
  clear () noexcept {
    _M_clear ();
  }

private:
  [[nodiscard]] static uint16_t
  clamp_depth (uint16_t depth) noexcept {
    if (depth == 0u)
      return 1u;
    if (depth > GnssRingMaxDepth)
      return GnssRingMaxDepth;
    return depth;
  }

  void
  _M_clear () noexcept {
    _M_head = 0u;
    _M_tail = 0u;
    _M_count = 0u;
    _M_mask = static_cast<uint16_t> (_M_depth - 1u);
    for (uint16_t i = 0u; i < GnssRingMaxDepth; ++i)
      _M_buf[i] = 0u;
  }

  uint8_t _M_buf[GnssRingMaxDepth] = {};
  uint16_t _M_depth = 1u;
  uint16_t _M_head = 0u;
  uint16_t _M_tail = 0u;
  uint16_t _M_count = 0u;
  uint16_t _M_mask = 0u;
};

/** @} */
} // namespace zzs::kernel::device::driver::gnss::stream
