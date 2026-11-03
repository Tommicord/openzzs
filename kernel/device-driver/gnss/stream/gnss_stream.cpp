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

#include "kernel/device-driver/gnss/stream/gnss_stream.hpp"

namespace zzs::kernel::device::driver::gnss::stream {

[[nodiscard]] zzs::kernel::Status
GnssStream::poll () noexcept {
    if (_M_bus == nullptr || !_M_bus->is_open ())
        return zzs::kernel::Status::NOT_READY;

    /* Drain the stream bus into the fixed byte ring. */
    uint8_t buf[64];
    bool any_decoded = false;
    for (;;) {
        uint16_t filled = 0u;
        zzs::kernel::Status st = _M_bus->read_bytes (buf, sizeof (buf), &filled);
        if (st == zzs::kernel::Status::STALE)
            break; // no more bytes available right now
        if (!zzs::kernel::status_ok (st))
            return st;
        if (filled == 0u)
            break;
        for (uint16_t i = 0u; i < filled; ++i) {
            (void) _M_byte_ring.push (buf[i]);
            _M_bytes_drained += 1u;
        }
    }

    /* Run the ring through both parsers in lockstep. */
    uint8_t b = 0u;
    while (_M_byte_ring.pop (b) == zzs::kernel::Status::OK) {
        if (_M_feed_parsers (b) == zzs::kernel::Status::OK)
            any_decoded = true;
    }

    return any_decoded ? zzs::kernel::Status::OK : zzs::kernel::Status::STALE;
}

[[nodiscard]] zzs::kernel::Status
GnssStream::send_command (const uint8_t *buf, uint16_t len) noexcept {
    if (buf == nullptr && len > 0u)
        return zzs::kernel::Status::OUT_OF_RANGE;
    if (_M_bus == nullptr)
        return zzs::kernel::Status::INTERNAL;
    return _M_bus->write_bytes (buf, len);
}

[[nodiscard]] zzs::kernel::Status
GnssStream::flush_rx () noexcept {
    if (_M_bus == nullptr)
        return zzs::kernel::Status::INTERNAL;
    return _M_bus->flush_rx ();
}

[[nodiscard]] bool
GnssStream::is_open () const noexcept {
    return _M_bus != nullptr && _M_bus->is_open ();
}

[[nodiscard]] zzs::kernel::Status
GnssStream::_M_feed_parsers (uint8_t b) noexcept {
    bool any_ok = false;
    if ((_M_protocol_mask & 0x01u) != 0u) {
        zzs::kernel::Status st = _M_nmea.feed (b);
        if (st == zzs::kernel::Status::OK)
            any_ok = true;
    }
    if ((_M_protocol_mask & 0x02u) != 0u) {
        zzs::kernel::Status st = _M_ubx.feed (b);
        if (st == zzs::kernel::Status::OK)
            any_ok = true;
    }
    return any_ok ? zzs::kernel::Status::OK : zzs::kernel::Status::STALE;
}

} // namespace zzs::kernel::device::driver::gnss::stream
