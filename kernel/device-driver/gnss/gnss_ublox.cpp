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

#include "kernel/device-driver/gnss/gnss_ublox.hpp"

#include "kernel/device-driver/gnss/ubx/ubx_framing.hpp"
#include "kernel/kernel_clock_source.hpp"

#include <cstdint>
#include <cstring>

namespace zzs::kernel::device::driver::gnss {

[[nodiscard]] zzs::kernel::Status
UbloxDriver::init () noexcept {
    if (_M_bus == nullptr)
        return zzs::kernel::Status::INTERNAL;
    if (!_M_bus->is_open ())
        return zzs::kernel::Status::NOT_READY;

    _M_status = GnssReceiverStatus::Starting;
    _M_next_handle = 1u;
    _M_fix = fix_make_invalid ();
    _M_satellite_count = 0u;
    _M_consecutive_errors = 0u;

    /* Configure the receiver (baud, NMEA/UBX output messages, nav rate). */
    zzs::kernel::Status st = _M_configure_receiver ();
    if (!zzs::kernel::status_ok (st)) {
        _M_status = GnssReceiverStatus::Error;
        return st;
    }
    _M_configured = true;
    _M_status = GnssReceiverStatus::Searching;
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
UbloxDriver::deinit () noexcept {
    _M_configured = false;
    _M_status = GnssReceiverStatus::Off;
    _M_fix = fix_make_invalid ();
    _M_satellite_count = 0u;
    (void) _M_stream.flush_rx ();
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
UbloxDriver::poll () noexcept {
    if (!_M_configured)
        return zzs::kernel::Status::NOT_READY;

    /* Snapshot parser error counts before polling so we can detect new errors. */
    const uint32_t nmea_errors_before = _M_stream.nmea_parser ().errors ();
    const uint32_t ubx_errors_before = _M_stream.ubx_parser ().errors ();

    zzs::kernel::Status st = _M_stream.poll ();
    if (!zzs::kernel::status_ok (st) && st != zzs::kernel::Status::STALE)
        return st;

    /* Record any new parser errors and update the error state. */
    if (_M_stream.nmea_parser ().errors () > nmea_errors_before
        || _M_stream.ubx_parser ().errors () > ubx_errors_before)
        _M_record_error ();

    /* Pull the latest UBX NAV-PVT fix into the driver cache. */
    Fix fix = fix_make_invalid ();
    st = _M_stream.ubx_parser ().decode_nav_pvt (fix);
    if (zzs::kernel::status_ok (st) && fix_is_usable (fix)) {
        _M_publish_fix (fix);
        _M_clear_errors ();
    }

    /* Pull satellites from NAV-SAT if available. */
    Satellite sats[GnssMaxSatellites] = {};
    uint8_t filled = 0u;
    st = _M_stream.ubx_parser ().decode_nav_sat (sats, GnssMaxSatellites, &filled);
    if (zzs::kernel::status_ok (st) && filled > 0u)
        _M_publish_satellites (sats, filled);

    /* Pull NAV-DOP if available and update fix DOP fields. */
    int16_t pdop_t = 0, hdop_t = 0, vdop_t = 0;
    st = _M_stream.ubx_parser ().decode_nav_dop (pdop_t, hdop_t, vdop_t);
    if (zzs::kernel::status_ok (st)) {
        _M_fix._M_pdop_tenths = pdop_t;
        _M_fix._M_hdop_tenths = hdop_t;
        _M_fix._M_vdop_tenths = vdop_t;
    }

    /* If NMEA produced a GGA/RMC fix and we have no UBX fix yet, use it. */
    if (!fix_has_valid_status (_M_fix)) {
        const auto &nmea = _M_stream.nmea_parser ().last_sentence ();
        if (nmea._M_checksum_ok
            && (nmea._M_type == nmea::SentenceType::Gga || nmea._M_type == nmea::SentenceType::Rmc)) {
            Fix f = fix_make_invalid ();
            f._M_timestamp_ns = timestamp_ns ();
            if (nmea._M_type == nmea::SentenceType::Gga) {
                const auto &gga = nmea._M_gga;
                f._M_lat = gga._M_lat;
                f._M_lon = gga._M_lon;
                f._M_alt_mm = gga._M_alt_mm;
                f._M_num_satellites = gga._M_num_satellites;
                f._M_hdop_tenths = gga._M_hdop_tenths;
                f._M_status = gga._M_fix_quality != 0u ? FixStatus::Gps3d : FixStatus::NoFix;
            } else {
                const auto &rmc = nmea._M_rmc;
                if (!rmc._M_valid) {
                    f._M_status = FixStatus::NoFix;
                } else {
                    f._M_lat = rmc._M_lat;
                    f._M_lon = rmc._M_lon;
                    f._M_speed_mm_s = rmc._M_speed_knots_tenths * 1852ll * 100ll / 3600ll; // knots->mm/s
                    f._M_course_deg_tenths = rmc._M_course_deg_tenths;
                    f._M_status = FixStatus::Gps3d;
                }
            }
            if (fix_has_valid_status (f))
                _M_publish_fix (f);
        }
    }

    if (_M_status == GnssReceiverStatus::Searching && fix_has_valid_status (_M_fix))
        _M_status = GnssReceiverStatus::Fix;

    return (st == zzs::kernel::Status::OK) ? zzs::kernel::Status::OK : zzs::kernel::Status::STALE;
}

[[nodiscard]] zzs::kernel::Status
UbloxDriver::read_fix (Fix &out) noexcept {
    if (!fix_is_usable (_M_fix))
        return zzs::kernel::Status::STALE;
    out = _M_fix;
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
UbloxDriver::read_satellites (Satellite *out, uint8_t max_count, uint8_t *filled) noexcept {
    if (out == nullptr || filled == nullptr)
        return zzs::kernel::Status::OUT_OF_RANGE;
    if (_M_satellite_count == 0u) {
        *filled = 0u;
        return zzs::kernel::Status::STALE;
    }
    uint8_t n = _M_satellite_count < max_count ? _M_satellite_count : max_count;
    for (uint8_t i = 0u; i < n; ++i)
        out[i] = _M_satellites[i];
    *filled = n;
    return zzs::kernel::Status::OK;
}

[[nodiscard]] bool
UbloxDriver::self_test () noexcept {
    if (!_M_configured)
        return false;
    /* A self-test passes when a usable fix has been observed. */
    return fix_has_valid_status (_M_fix) && fix_coordinates_valid (_M_fix);
}

[[nodiscard]] zzs::kernel::Status
UbloxDriver::send_ubx (uint8_t msg_class, uint8_t msg_id, const uint8_t *payload, uint16_t payload_len) noexcept {
    if (payload == nullptr && payload_len > 0u)
        return zzs::kernel::Status::OUT_OF_RANGE;
    if (payload_len > ubx::UbxMaxPayloadBytes)
        return zzs::kernel::Status::BAD_LENGTH;
    if (_M_bus == nullptr)
        return zzs::kernel::Status::INTERNAL;

    uint8_t frame[6u + ubx::UbxMaxPayloadBytes + 2u];
    uint16_t idx = 0u;
    frame[idx++] = ubx::UbxSync1;
    frame[idx++] = ubx::UbxSync2;
    frame[idx++] = msg_class;
    frame[idx++] = msg_id;
    frame[idx++] = static_cast<uint8_t> (payload_len & 0xffu);
    frame[idx++] = static_cast<uint8_t> ((payload_len >> 8) & 0xffu);
    for (uint16_t i = 0u; i < payload_len; ++i)
        frame[idx++] = payload[i];

    uint16_t ck = ubx::ubx_checksum (&frame[2], 4u + payload_len);
    frame[idx++] = static_cast<uint8_t> (ck & 0xffu);
    frame[idx++] = static_cast<uint8_t> ((ck >> 8) & 0xffu);

    return _M_bus->write_bytes (frame, idx);
}

[[nodiscard]] zzs::kernel::Status
UbloxDriver::_M_wait_for_ack () noexcept {
    /* Bounded wait for the UBX ACK-ACK/ACK-NAK response to the last config
     * command.  Reads bytes from the bus one at a time (so multiple queued
     * responses are consumed one-per-config-call rather than drained in one
     * gulp), feeds them into a local UBX parser, and inspects the decoded ACK
     * message.  No infinite loop: bail after kMaxAttempts with a delay between
     * idle attempts. */
    constexpr uint32_t kMaxAttempts = 200u;
    constexpr uint32_t kDelayMs = 10u;
    ubx::Parser ack_parser;

    for (uint32_t attempt = 0u; attempt < kMaxAttempts; ++attempt) {
        uint8_t b = 0u;
        uint16_t filled = 0u;
        zzs::kernel::Status st = bus_read (&b, 1u, &filled);
        if (zzs::kernel::status_ok (st) && filled == 1u) {
            if (ack_parser.feed (b) == zzs::kernel::Status::OK) {
                const auto &msg = ack_parser.last_message ();
                if (msg._M_is_ack) {
                    return msg._M_ack_ok ? zzs::kernel::Status::OK : zzs::kernel::Status::INTERNAL;
                }
            }
        } else {
            /* No byte available right now: wait a little before retrying. */
            delay_ms (kDelayMs);
        }
    }
    return zzs::kernel::Status::INTERNAL; // timeout waiting for ACK
}

[[nodiscard]] zzs::kernel::Status
UbloxDriver::_M_configure_receiver () noexcept {
    /* CFG-PRT: set UART1 to configured baud, NMEA + UBX in/out.
     * Real UBX-CFG-PRT payload (20 bytes):
     *   portID(1) | reserved0(1) | txReady(2) | mode(4) | baudRate(4) |
     *   inProtoMask(2) | outProtoMask(2) | flags(2) | reserved1(2)
     * baudRate is at offset 8 (little-endian). */
    {
        uint8_t cfg_prt[20] = {
            0x00,                   // portID (UART1)
            0x00,                   // reserved0
            0x00, 0x00,             // txReady
            0x00, 0x80, 0x00, 0x00, // mode: 8N1, no parity (0x00008000 LE)
            0x00, 0x00, 0x00, 0x00, // baudRate (filled below, little-endian)
            0x03, 0x00,             // inProtoMask: NMEA + UBX
            0x03, 0x00,             // outProtoMask: NMEA + UBX
            0x00, 0x00,             // flags
            0x00, 0x00,             // reserved1
        };
        /* Fill baud rate (little-endian) at offset 8. */
        cfg_prt[8] = static_cast<uint8_t> (_M_config._M_baud_rate & 0xffu);
        cfg_prt[9] = static_cast<uint8_t> ((_M_config._M_baud_rate >> 8) & 0xffu);
        cfg_prt[10] = static_cast<uint8_t> ((_M_config._M_baud_rate >> 16) & 0xffu);
        cfg_prt[11] = static_cast<uint8_t> ((_M_config._M_baud_rate >> 24) & 0xffu);
        zzs::kernel::Status st = send_ubx (0x06u, 0x00u, cfg_prt, sizeof (cfg_prt));
        if (!zzs::kernel::status_ok (st))
            return st;
        st = _M_wait_for_ack ();
        if (!zzs::kernel::status_ok (st))
            return st;
        delay_ms (10u);
    }

    /* CFG-RATE: navigation rate (ms). */
    {
        uint8_t cfg_rate[] = {
            0x64, 0x00, // measRate (1000 ms)
            0x64, 0x00, // navRate (1 cycle)
            0x01, 0x00, // timeRef (UTC)
        };
        cfg_rate[0] = static_cast<uint8_t> (_M_config._M_nav_rate_ms & 0xffu);
        cfg_rate[1] = static_cast<uint8_t> ((_M_config._M_nav_rate_ms >> 8) & 0xffu);
        zzs::kernel::Status st = send_ubx (0x06u, 0x08u, cfg_rate, sizeof (cfg_rate));
        if (!zzs::kernel::status_ok (st))
            return st;
        st = _M_wait_for_ack ();
        if (!zzs::kernel::status_ok (st))
            return st;
        delay_ms (10u);
    }

    /* CFG-MSG: enable NAV-PVT on UART1. */
    {
        uint8_t cfg_msg[] = {0x01u, 0x07u, 0x01u}; // class, id, rate(UART1)=1
        zzs::kernel::Status st = send_ubx (0x06u, 0x01u, cfg_msg, sizeof (cfg_msg));
        if (!zzs::kernel::status_ok (st))
            return st;
        st = _M_wait_for_ack ();
        if (!zzs::kernel::status_ok (st))
            return st;
        delay_ms (10u);
    }

    return zzs::kernel::Status::OK;
}

} // namespace zzs::kernel::device::driver::gnss
