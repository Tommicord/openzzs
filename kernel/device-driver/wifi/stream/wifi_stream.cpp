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

#include "kernel/device-driver/wifi/stream/wifi_stream.hpp"

#include "kernel/kernel_status.hpp"

#include <cstdint>
#include <cstring>

namespace zzs::kernel::device::driver::wifi::stream {

[[nodiscard]] zzs::kernel::Status
WifiStream::poll () noexcept {
    if (_M_bus == nullptr)
        return zzs::kernel::Status::INTERNAL;
    if (!_M_bus->is_open ())
        return zzs::kernel::Status::NOT_READY;

    /* Drain the bus into the byte ring. */
    uint8_t buf[64];
    uint16_t filled = 0u;
    bool any_read = false;
    for (;;) {
        filled = 0u;
        zzs::kernel::Status st = _M_bus->read_bytes (buf, static_cast<uint16_t> (sizeof (buf)), &filled);
        if (st == zzs::kernel::Status::STALE)
            break;
        if (!zzs::kernel::status_ok (st))
            return st;
        if (filled == 0u)
            break;
        any_read = true;
        for (uint16_t i = 0u; i < filled; ++i) {
            zzs::kernel::Status rst = ring_push (buf[i]);
            if (!zzs::kernel::status_ok (rst) && rst != zzs::kernel::Status::QUEUE_FULL)
                return rst;
        }
        _M_bytes_drained += filled;
    }

    /* Reassemble and parse frames from the ring. */
    zzs::kernel::Status rst = _M_reassemble_frames ();
    if (!zzs::kernel::status_ok (rst) && rst != zzs::kernel::Status::STALE)
        return rst;

    if (any_read || rst == zzs::kernel::Status::OK)
        return zzs::kernel::Status::OK;
    return zzs::kernel::Status::STALE;
}

[[nodiscard]] zzs::kernel::Status
WifiStream::send_frame (const uint8_t *buf, uint16_t len) noexcept {
    if (buf == nullptr && len > 0u)
        return zzs::kernel::Status::OUT_OF_RANGE;
    if (_M_bus == nullptr)
        return zzs::kernel::Status::INTERNAL;
    if (!_M_bus->is_open ())
        return zzs::kernel::Status::NOT_READY;
    return _M_bus->write_bytes (buf, len);
}

[[nodiscard]] zzs::kernel::Status
WifiStream::flush_rx () noexcept {
    if (_M_bus == nullptr)
        return zzs::kernel::Status::INTERNAL;
    return _M_bus->flush_rx ();
}

[[nodiscard]] bool
WifiStream::is_open () const noexcept {
    return _M_bus != nullptr && _M_bus->is_open ();
}

[[nodiscard]] zzs::kernel::Status
WifiStream::ring_push (uint8_t byte) noexcept {
    if (_M_ring_depth == 0u)
        return zzs::kernel::Status::INTERNAL;
    _M_ring_buf[_M_ring_head] = byte;
    _M_ring_head = static_cast<uint16_t> ((_M_ring_head + 1u) & _M_ring_mask);
    if (_M_ring_count < _M_ring_depth) {
        _M_ring_count += 1u;
        return zzs::kernel::Status::OK;
    }
    /* Drop-oldest: ring full, advance tail too. */
    _M_ring_tail = static_cast<uint16_t> ((_M_ring_tail + 1u) & _M_ring_mask);
    return zzs::kernel::Status::QUEUE_FULL;
}

[[nodiscard]] zzs::kernel::Status
WifiStream::ring_pop (uint8_t &byte) noexcept {
    if (_M_ring_count == 0u)
        return zzs::kernel::Status::STALE;
    byte = _M_ring_buf[_M_ring_tail];
    _M_ring_tail = static_cast<uint16_t> ((_M_ring_tail + 1u) & _M_ring_mask);
    _M_ring_count -= 1u;
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
WifiStream::_M_reassemble_frames () noexcept {
    /* Length-prefixed frame protocol: [len_lo][len_hi][payload...].
     * Drain bytes from the ring, accumulate into _M_frame_buf until the
     * declared length is reached, then parse and deliver via callback.
     * Oversized frames (> WifiMaxFrameBytes) are discarded byte-for-byte so the
     * ring stays byte-aligned and the next frame's length prefix is read
     * correctly. */
    bool any_parsed = false;
    for (;;) {
        /* Discarding an oversized frame's payload: pop bytes without storing them. */
        if (_M_frame_skip > 0u) {
            uint8_t byte = 0u;
            zzs::kernel::Status st = ring_pop (byte);
            if (st == zzs::kernel::Status::STALE)
                break; /* Wait for more bytes to discard. */
            if (!zzs::kernel::status_ok (st))
                return st;
            _M_frame_skip -= 1u;
            continue;
        }

        if (_M_in_length) {
            /* Need two length bytes. */
            if (_M_frame_received == 0u) {
                zzs::kernel::Status st = ring_pop (_M_len_lo);
                if (st == zzs::kernel::Status::STALE)
                    break; /* No more length bytes available. */
                if (!zzs::kernel::status_ok (st))
                    return st;
                _M_frame_received = 1u;
            }
            if (_M_frame_received == 1u) {
                zzs::kernel::Status st = ring_pop (_M_len_hi);
                if (st == zzs::kernel::Status::STALE)
                    break; /* Waiting for second length byte. */
                if (!zzs::kernel::status_ok (st))
                    return st;
                _M_frame_len = static_cast<uint16_t> (_M_len_lo) | (static_cast<uint16_t> (_M_len_hi) << 8);
                _M_frame_received = 2u;
                /* Validate the declared length. */
                if (_M_frame_len == 0u || _M_frame_len > WifiMaxFrameBytes) {
                    /* Oversized/nonsensical length: discard exactly _M_frame_len payload
                     * bytes from the ring so the stream stays byte-aligned, then resync
                     * in length mode.  Zero length needs no discard. */
                    _M_frame_skip = _M_frame_len;
                    _M_in_length = true;
                    _M_frame_received = 0u;
                    _M_frame_len = 0u;
                    _M_len_lo = 0u;
                    _M_len_hi = 0u;
                    continue;
                }
                _M_in_length = false;
                _M_frame_received = 0u;
            }
        }

        /* Payload accumulation. */
        while (_M_frame_received < _M_frame_len) {
            uint8_t byte = 0u;
            zzs::kernel::Status st = ring_pop (byte);
            if (st == zzs::kernel::Status::STALE)
                break; /* Wait for more payload bytes. */
            if (!zzs::kernel::status_ok (st))
                return st;
            _M_frame_buf[_M_frame_received] = byte;
            _M_frame_received += 1u;
        }

        if (_M_frame_received < _M_frame_len)
            break; /* Incomplete frame, wait for more data. */

        /* Full frame accumulated: deliver via callback.  The callback is
         * responsible for distinguishing module protocol frames (ScanDone,
         * BssEntry, ConnectResult, Deauth, etc.) from real 802.11 MAC frames and
         * handling each accordingly.  Always invoke it so short protocol frames
         * (which fail 802.11 parsing) are still dispatched. */
        Frame f = frame_make_empty ();
        f._M_len = _M_frame_len;
        for (uint16_t i = 0u; i < _M_frame_len; ++i)
            f._M_buf[i] = _M_frame_buf[i];
        if (_M_clock != nullptr)
            f._M_timestamp_ns = _M_clock ();

        if (_M_cb != nullptr) {
            _M_cb (f, _M_ctx);
            any_parsed = true;
        }

        /* Also attempt 802.11 parsing for the stream's own parse-error tracking;
         * failures here are expected for protocol frames and are ignored.  Use a
         * distinct scratch frame so the source buffer is never aliased with the
         * output (wifi_parse_frame zeroes its output before copying, which would
         * otherwise always "succeed" on a zeroed buffer). */
        Frame scratch = frame_make_empty ();
        (void) wifi_parse_frame (f._M_buf, f._M_len, scratch);

        /* Clear the frame buffer before resetting the length so the clear loop
         * actually runs (the buffer is overwritten next frame, but clearing avoids
         * leaking stale payload bytes beyond _M_frame_len). */
        for (uint16_t i = 0u; i < _M_frame_len; ++i)
            _M_frame_buf[i] = 0u;
        /* Reset for the next frame. */
        _M_in_length = true;
        _M_frame_received = 0u;
        _M_frame_len = 0u;
        _M_len_lo = 0u;
        _M_len_hi = 0u;
    }

    return any_parsed ? zzs::kernel::Status::OK : zzs::kernel::Status::STALE;
}

/** @} */

} // namespace zzs::kernel::device::driver::wifi::stream
