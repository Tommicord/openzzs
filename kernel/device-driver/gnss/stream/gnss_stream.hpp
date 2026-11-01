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

#include "kernel/device-driver/gnss/gnss_driver.hpp"
#include "kernel/device-driver/gnss/nmea/nmea_parser.hpp"
#include "kernel/device-driver/gnss/stream/gnss_byte_ring.hpp"
#include "kernel/device-driver/gnss/ubx/ubx_parser.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief GNSS streaming pipeline: stream bus -> byte ring -> NMEA + UBX
 *        parsers -> cached Fix / Satellite table.
 *
 * Owns the fixed-size byte ring and both protocol parsers.  `poll()` drains
 * the stream bus into the ring, then runs the ring through both parsers in
 * lockstep, publishing any new fix/satellite data to the driver base class.
 * All buffers are fixed C arrays; no heap, no exceptions.
 * @{
 */
namespace zzs::kernel::device::driver::gnss::stream {

/** @brief Configuration for the GNSS stream pipeline. */
struct GnssStreamConfig {
  uint16_t _M_byte_ring_depth = 256u; /**< Depth of the byte ring (power of 2). */
  uint8_t _M_protocol_mask = 0x03u;   /**< Bit0=NMEA, bit1=UBX. */
};

/** @brief Streaming GNSS pipeline. */
class GnssStream {
public:
  /** @brief Construct with a stream bus and optional clock. */
  GnssStream (zzs::kernel::device::driver::StreamBusInterface *bus, uint64_t (*clock) (void) = nullptr,
              const GnssStreamConfig &config = GnssStreamConfig{}) noexcept
      : _M_bus (bus), _M_clock (clock), _M_byte_ring (config._M_byte_ring_depth),
        _M_protocol_mask (config._M_protocol_mask) {
  }

  GnssStream (const GnssStream &) = delete;
  GnssStream &
  operator= (const GnssStream &) = delete;
  GnssStream (GnssStream &&) = delete;
  GnssStream &
  operator= (GnssStream &&) = delete;

  /** @brief Drain the stream bus into the byte ring and run both parsers.
   *         Returns OK when at least one new message was decoded, STALE when
   *         nothing new, NOT_READY when the bus is not open, or a bus error. */
  [[nodiscard]] zzs::kernel::Status
  poll () noexcept;

  /** @brief Send a raw command to the receiver (e.g. UBX CFG or NMEA
   *         PUBX). */
  [[nodiscard]] zzs::kernel::Status
  send_command (const uint8_t *buf, uint16_t len) noexcept;

  /** @brief Flush the receiver's receive buffer. */
  [[nodiscard]] zzs::kernel::Status
  flush_rx () noexcept;

  /** @brief Return true when the underlying stream is open. */
  [[nodiscard]] bool
  is_open () const noexcept;

  /** @brief Access the NMEA parser (read-only). */
  [[nodiscard]] const nmea::Parser &
  nmea_parser () const noexcept {
    return _M_nmea;
  }

  /** @brief Access the UBX parser (read-only). */
  [[nodiscard]] const ubx::Parser &
  ubx_parser () const noexcept {
    return _M_ubx;
  }

  /** @brief Number of bytes currently buffered in the ring. */
  [[nodiscard]] uint16_t
  buffered_bytes () const noexcept {
    return _M_byte_ring.filled ();
  }

private:
  [[nodiscard]] zzs::kernel::Status
  _M_feed_parsers (uint8_t b) noexcept;

  zzs::kernel::device::driver::StreamBusInterface *_M_bus = nullptr;
  uint64_t (*_M_clock) (void) = nullptr;
  GnssStreamConfig _M_config{};
  stream::GnssByteRing _M_byte_ring{1u};
  nmea::Parser _M_nmea{};
  ubx::Parser _M_ubx{};
  uint8_t _M_protocol_mask = 0x03u;
  uint32_t _M_bytes_drained = 0u;
};

/** @} */
} // namespace zzs::kernel::device::driver::gnss::stream
