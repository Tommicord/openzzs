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

#include "kernel/device-driver/stream_bus.hpp"
#include "kernel/device-driver/wifi/wifi_driver.hpp"
#include "kernel/device-driver/wifi/wifi_frame.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief WiFi streaming pipeline: stream bus -> byte ring -> 802.11 frame
 *        parser -> cached Frame / BSS table.
 *
 * Reads raw bytes from the stream bus into a fixed byte ring, then reassembles
 * frames.  The radio stream is expected to deliver each frame prefixed with a
 * 16-bit little-endian length (a common WiFi chip SPI/UART framing convention).
 * Each reassembled frame is parsed by `wifi_parse_frame` and delivered through
 * a caller-supplied frame callback.  All buffers are fixed C arrays; no heap,
 * no exceptions, no writable statics.
 *
 * Threading model: single-producer / single-consumer.  `poll()` is the only
 * consumer (drains the bus and reassembles), and `send_frame()` is the only
 * producer.  There is no locking; the caller must not call `poll()` and
 * `send_frame()` concurrently from different contexts.  The ring is
 * drop-oldest when full.
 * @{
 */
namespace zzs::kernel::device::driver::wifi::stream {

/** @brief Callback invoked when a complete 802.11 frame has been parsed. */
using FrameCallback = void (*) (const Frame &frame, void *context) noexcept;

/** @brief Configuration for the WiFi stream pipeline. */
struct WifiStreamConfig {
    uint16_t _M_byte_ring_depth = 256u; /**< Depth of the byte ring (power of 2). */
};

/** @brief Streaming WiFi pipeline. */
class WifiStream {
  public:
    /** @brief Construct with a stream bus, optional clock, and config. */
    WifiStream (zzs::kernel::device::driver::StreamBusInterface *bus, uint64_t (*clock) (void) = nullptr,
                const WifiStreamConfig &config = WifiStreamConfig{}) noexcept
        : _M_bus (bus), _M_clock (clock), _M_config (config) {
        _M_ring_depth = valid_ring_depth (config._M_byte_ring_depth);
        _M_ring_mask = static_cast<uint16_t> (_M_ring_depth - 1u);
    }

    WifiStream (const WifiStream &) = delete;
    WifiStream &
    operator= (const WifiStream &) = delete;
    WifiStream (WifiStream &&) = delete;
    WifiStream &
    operator= (WifiStream &&) = delete;

    /** @brief Register a frame-delivery callback and context pointer. */
    void
    set_frame_callback (FrameCallback cb, void *context) noexcept {
        _M_cb = cb;
        _M_ctx = context;
    }

    /** @brief Reconfigure the byte-ring depth and reset the ring.
     *
     * The depth must be a power of two in [1, kRingMaxDepth]. Returns
     * `OUT_OF_RANGE` when the requested depth is invalid and leaves the ring
     * unchanged; otherwise applies the new depth and returns OK. */
    [[nodiscard]] zzs::kernel::Status
    reset (const WifiStreamConfig &config) noexcept {
        uint16_t depth = valid_ring_depth (config._M_byte_ring_depth);
        if (depth != config._M_byte_ring_depth)
            return zzs::kernel::Status::OUT_OF_RANGE;
        _M_config = config;
        _M_ring_depth = depth;
        _M_ring_mask = static_cast<uint16_t> (depth - 1u);
        _M_ring_head = 0u;
        _M_ring_tail = 0u;
        _M_ring_count = 0u;
        for (uint16_t i = 0u; i < kRingMaxDepth; ++i)
            _M_ring_buf[i] = 0u;
        return zzs::kernel::Status::OK;
    }

    /** @brief Return the configured byte-ring depth (always a valid power of two). */
    [[nodiscard]] uint16_t
    ring_depth () const noexcept {
        return _M_ring_depth;
    }

    /** @brief Drain the stream bus into the byte ring and reassemble/parse
     *         frames. Returns OK when at least one frame was parsed, STALE when
     *         nothing new, NOT_READY when the bus is not open, or a bus error. */
    [[nodiscard]] zzs::kernel::Status
    poll () noexcept;

    /** @brief Send a raw frame/command to the radio. */
    [[nodiscard]] zzs::kernel::Status
    send_frame (const uint8_t *buf, uint16_t len) noexcept;

    /** @brief Flush the radio's receive buffer. */
    [[nodiscard]] zzs::kernel::Status
    flush_rx () noexcept;

    /** @brief Return true when the underlying stream is open. */
    [[nodiscard]] bool
    is_open () const noexcept;

    /** @brief Number of bytes currently buffered in the ring. */
    [[nodiscard]] uint16_t
    buffered_bytes () const noexcept {
        return _M_ring_count;
    }

    /** @brief Drop-oldest byte-ring push. Returns OK when the byte was stored,
     *         QUEUE_FULL when the ring is full (oldest byte dropped to make
     *         room). */
    [[nodiscard]] zzs::kernel::Status
    ring_push (uint8_t byte) noexcept;

    /** @brief Pop one byte from the ring. Returns STALE when empty. */
    [[nodiscard]] zzs::kernel::Status
    ring_pop (uint8_t &byte) noexcept;

  private:
    [[nodiscard]] zzs::kernel::Status
    _M_reassemble_frames () noexcept;

    /** @brief Clamp a requested ring depth to a valid power of two in [1, kRingMaxDepth].
     *         Used by the constructor (which cannot fail) so an invalid config
     *         degrades gracefully to the maximum valid depth instead of
     *         corrupting ring indices. */
    [[nodiscard]] static constexpr uint16_t
    valid_ring_depth (uint16_t depth) noexcept {
        if (depth == 0u || depth > kRingMaxDepth)
            return kRingMaxDepth;
        /* Round down to the largest power of two <= depth. */
        uint16_t p2 = 1u;
        while (p2 * 2u <= depth)
            p2 = static_cast<uint16_t> (p2 * 2u);
        return p2;
    }

    // -- fixed-size byte ring (drop-oldest, power-of-two mask) --
    static constexpr uint16_t kRingMaxDepth = 256u;

    uint8_t _M_ring_buf[kRingMaxDepth] = {};
    uint16_t _M_ring_depth = 1u;
    uint16_t _M_ring_head = 0u;
    uint16_t _M_ring_tail = 0u;
    uint16_t _M_ring_count = 0u;
    uint16_t _M_ring_mask = 0u;

    // -- length-prefixed frame reassembly state (per-instance, no statics) --
    bool _M_in_length = true;
    uint8_t _M_len_lo = 0u;
    uint8_t _M_len_hi = 0u;
    uint16_t _M_frame_len = 0u;
    uint16_t _M_frame_received = 0u;
    uint16_t _M_frame_skip = 0u; /**< Payload bytes left to discard for an oversized frame. */
    uint8_t _M_frame_buf[WifiMaxFrameBytes] = {};

    zzs::kernel::device::driver::StreamBusInterface *_M_bus = nullptr;
    uint64_t (*_M_clock) (void) = nullptr;
    WifiStreamConfig _M_config{};
    uint32_t _M_bytes_drained = 0u;

    FrameCallback _M_cb = nullptr;
    void *_M_ctx = nullptr;
};

/** @} */

} // namespace zzs::kernel::device::driver::wifi::stream
