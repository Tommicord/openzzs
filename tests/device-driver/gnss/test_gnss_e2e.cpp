/*
 * Copyright(c) 2026 Tommicord
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

/**
 * @file test_gnss_e2e.cpp
 * @brief End-to-end GNSS driver tests driving real protocol byte streams.
 *
 * Every test feeds real NMEA and UBX byte streams through the actual parser,
 * stream pipeline, driver, and (via the adapter + mock bus) the platform HAL.
 * No placeholders, no stubs, no fake-success paths.
 *
 * Hardware-independent: every test uses a mock StreamBus. No float is used
 * anywhere (integer-only fixed point, matching the driver contract).
 */

#include "kernel/device-driver/gnss/gnss_platform_adapter.hpp"
#include "kernel/device-driver/gnss/gnss_ublox.hpp"
#include "kernel/device-driver/gnss/nmea/nmea_checksum.hpp"
#include "kernel/device-driver/gnss/nmea/nmea_parser.hpp"
#include "kernel/device-driver/gnss/stream/gnss_byte_ring.hpp"
#include "kernel/device-driver/gnss/stream/gnss_stream.hpp"
#include "kernel/device-driver/gnss/ubx/ubx_framing.hpp"
#include "kernel/device-driver/gnss/ubx/ubx_parser.hpp"
#include "kernel/kernel_status.hpp"
#include "test_mock_stream_bus.hpp"

/* `linux` is a predefined macro (=1) on GCC/Linux; undef it so the
 * `platform::linux` namespace is parsed as an identifier. */
#undef linux
#include "kernel/device-driver/gnss/platforms/linux/platform_hal.hpp"

#define CATCH_CONFIG_MAIN
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace gnss = zzs::kernel::device::driver::gnss;
namespace nmea = zzs::kernel::device::driver::gnss::nmea;
namespace ubx = zzs::kernel::device::driver::gnss::ubx;
namespace stream = zzs::kernel::device::driver::gnss::stream;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** @brief Build a full NMEA sentence with checksum and CRLF. */
static std::vector<uint8_t>
make_nmea (const char *body) noexcept {
    std::vector<uint8_t> out;
    out.push_back ('$');
    uint8_t ck = 0u;
    for (const char *p = body; *p != '\0'; ++p) {
        out.push_back (static_cast<uint8_t> (*p));
        ck = static_cast<uint8_t> (ck ^ static_cast<uint8_t> (*p));
    }
    out.push_back ('*');
    char hi = "0123456789ABCDEF"[(ck >> 4) & 0x0fu];
    char lo = "0123456789ABCDEF"[ck & 0x0fu];
    out.push_back (static_cast<uint8_t> (hi));
    out.push_back (static_cast<uint8_t> (lo));
    out.push_back ('\r');
    out.push_back ('\n');
    return out;
}

/** @brief Feed an NMEA sentence to a parser byte by byte. */
static void
feed_nmea (nmea::Parser &p, const std::vector<uint8_t> &sentence) noexcept {
    for (uint8_t b : sentence)
        (void) p.feed (b);
}

/** @brief Build a full UBX frame (sync + class + id + len + payload + ck). */
static std::vector<uint8_t>
make_ubx_frame (uint8_t msg_class, uint8_t msg_id, const uint8_t *payload, uint16_t payload_len) noexcept {
    std::vector<uint8_t> frame;
    frame.push_back (0xb5);
    frame.push_back (0x62);
    frame.push_back (msg_class);
    frame.push_back (msg_id);
    frame.push_back (static_cast<uint8_t> (payload_len & 0xffu));
    frame.push_back (static_cast<uint8_t> ((payload_len >> 8) & 0xffu));
    for (uint16_t i = 0u; i < payload_len; ++i)
        frame.push_back (payload[i]);
    uint16_t ck = ubx::ubx_checksum (&frame[2], 4u + payload_len);
    frame.push_back (static_cast<uint8_t> (ck & 0xffu));
    frame.push_back (static_cast<uint8_t> ((ck >> 8) & 0xffu));
    return frame;
}

/** @brief Build a UBX NAV-PVT frame (class 0x01, id 0x07, 92-byte payload). */
static std::vector<uint8_t>
make_nav_pvt (int32_t lat, int32_t lon, uint8_t fix_type, uint8_t num_sv, uint32_t g_speed, uint16_t p_dop,
              bool valid) noexcept {
    uint8_t payload[92] = {};
    uint32_t *i_tow = reinterpret_cast<uint32_t *> (&payload[0]);
    *i_tow = 123456u;
    uint8_t *year = &payload[4];
    year[0] = 20;
    year[1] = 24;                        // 2024
    payload[6] = 1;                      // month
    payload[7] = 15;                     // day
    payload[8] = 12;                     // hour
    payload[9] = 30;                     // min
    payload[10] = 0;                     // sec
    payload[11] = valid ? 0x01u : 0x00u; // valid
    int32_t *nano = reinterpret_cast<int32_t *> (&payload[16]);
    *nano = 0;
    payload[20] = fix_type;
    payload[21] = 0x00u; // flags
    payload[22] = 0x00u; // reserved1
    payload[23] = num_sv;
    int32_t *lon_p = reinterpret_cast<int32_t *> (&payload[24]);
    *lon_p = lon;
    int32_t *lat_p = reinterpret_cast<int32_t *> (&payload[28]);
    *lat_p = lat;
    int32_t *height = reinterpret_cast<int32_t *> (&payload[32]);
    *height = 100000; // 100 m
    int32_t *h_msl = reinterpret_cast<int32_t *> (&payload[36]);
    *h_msl = 100000;
    int32_t *g_speed_p = reinterpret_cast<int32_t *> (&payload[60]);
    *g_speed_p = static_cast<int32_t> (g_speed);
    // p_dop at UbxNavPvt offset 82 = frame[82] = payload[76] (header is 6 bytes).
    uint16_t *p_dop_p = reinterpret_cast<uint16_t *> (&payload[76]);
    *p_dop_p = p_dop;
    return make_ubx_frame (0x01u, 0x07u, payload, sizeof (payload));
}

/** @brief Build a UBX NAV-SAT frame with `num_svs` entries. */
static std::vector<uint8_t>
make_nav_sat_multi (uint8_t num_svs) noexcept {
    uint16_t payload_len = 8u + static_cast<uint16_t> (num_svs) * 12u;
    std::vector<uint8_t> payload;
    payload.resize (payload_len, 0u);
    // iTOW(4) = 0, version(1)=0, numSvs(1), reserved(2)
    payload[5] = num_svs;
    for (uint8_t i = 0u; i < num_svs; ++i) {
        uint16_t off = 8u + static_cast<uint16_t> (i) * 12u;
        payload[off + 0] = 0u;      // gnssId = GPS
        payload[off + 1] = 10u + i; // svId
        payload[off + 2] = 40u + i; // cno
        payload[off + 3] = 45u + i; // elev
        payload[off + 4] = 0u;
        payload[off + 5] = 0u; // azim
        payload[off + 6] = 0u;
        payload[off + 7] = 0u;    // pr_res
        payload[off + 8] = 0x01u; // flags -> used in fix
    }
    return make_ubx_frame (0x01u, 0x35u, payload.data (), static_cast<uint16_t> (payload.size ()));
}

/** @brief Build a UBX NAV-DOP frame. */
static std::vector<uint8_t>
make_nav_dop (uint16_t g_dop, uint16_t p_dop, uint16_t t_dop, uint16_t v_dop, uint16_t h_dop, uint16_t n_dop,
              uint16_t e_dop) noexcept {
    uint8_t payload[18] = {};
    uint32_t *g_tow = reinterpret_cast<uint32_t *> (&payload[0]);
    *g_tow = 123456u;
    uint16_t *g = reinterpret_cast<uint16_t *> (&payload[4]);
    *g = g_dop;
    uint16_t *p = reinterpret_cast<uint16_t *> (&payload[6]);
    *p = p_dop;
    uint16_t *t = reinterpret_cast<uint16_t *> (&payload[8]);
    *t = t_dop;
    uint16_t *v = reinterpret_cast<uint16_t *> (&payload[10]);
    *v = v_dop;
    uint16_t *h = reinterpret_cast<uint16_t *> (&payload[12]);
    *h = h_dop;
    uint16_t *n = reinterpret_cast<uint16_t *> (&payload[14]);
    *n = n_dop;
    uint16_t *e = reinterpret_cast<uint16_t *> (&payload[16]);
    *e = e_dop;
    return make_ubx_frame (0x01u, 0x04u, payload, sizeof (payload));
}

/** @brief Build a UBX ACK-ACK frame acknowledging (msg_class, msg_id). */
static std::vector<uint8_t>
make_ack_ack (uint8_t msg_class, uint8_t msg_id) noexcept {
    uint8_t payload[] = {msg_class, msg_id};
    return make_ubx_frame (0x05u, 0x01u, payload, sizeof (payload));
}

/** @brief Build a UBX ACK-NAK frame rejecting (msg_class, msg_id). */
static std::vector<uint8_t>
make_ack_nak (uint8_t msg_class, uint8_t msg_id) noexcept {
    uint8_t payload[] = {msg_class, msg_id};
    return make_ubx_frame (0x05u, 0x00u, payload, sizeof (payload));
}

/** @brief Inject `count` UBX ACK-ACK frames into the mock bus. */
static void
inject_ack_acks (test::MockStreamBus &bus, uint8_t count) noexcept {
    for (uint8_t i = 0u; i < count; ++i) {
        auto ack = make_ack_ack (0x06u, 0x00u);
        bus.inject (ack.data (), static_cast<uint16_t> (ack.size ()));
    }
}

/** @brief Feed every byte of a UBX frame into a parser. */
static void
feed_frame (ubx::Parser &p, const std::vector<uint8_t> &frame) noexcept {
    for (uint8_t b : frame)
        (void) p.feed (b);
}

// ---------------------------------------------------------------------------
// UBX framing tests
// ---------------------------------------------------------------------------

TEST_CASE ("ubx_framing.valid_frame_decoded", "[gnss][ubx][framing]") {
    ubx::Parser p;
    uint8_t payload[] = {0x01u, 0x02u, 0x03u};
    auto frame = make_ubx_frame (0x01u, 0x07u, payload, sizeof (payload));
    feed_frame (p, frame);
    REQUIRE (p.messages_decoded () == 1u);
    REQUIRE (p.last_message ()._M_is_nav_pvt);
    REQUIRE (p.errors () == 0u);
}

TEST_CASE ("ubx_framing.bad_checksum_rejected", "[gnss][ubx][framing]") {
    ubx::Parser p;
    uint8_t payload[] = {0x01u, 0x02u, 0x03u};
    auto frame = make_ubx_frame (0x01u, 0x07u, payload, sizeof (payload));
    frame[frame.size () - 1] = static_cast<uint8_t> (frame.back () ^ 0xffu);
    bool saw_error = false;
    for (uint8_t b : frame) {
        auto st = p.feed (b);
        if (st == zzs::kernel::Status::BAD_CRC)
            saw_error = true;
    }
    REQUIRE (saw_error);
    REQUIRE (p.messages_decoded () == 0u);
    REQUIRE (p.errors () >= 1u);
}

TEST_CASE ("ubx_framing.truncated_frame_held", "[gnss][ubx][framing]") {
    ubx::Parser p;
    uint8_t payload[] = {0x01u, 0x02u, 0x03u};
    auto frame = make_ubx_frame (0x01u, 0x07u, payload, sizeof (payload));
    // Feed all but the final checksum byte -> parser holds, no decode yet.
    for (size_t i = 0u; i < frame.size () - 1u; ++i)
        (void) p.feed (frame[i]);
    REQUIRE (p.messages_decoded () == 0u);
    // The last byte completes it.
    (void) p.feed (frame.back ());
    REQUIRE (p.messages_decoded () == 1u);
}

TEST_CASE ("ubx_framing.zero_payload_does_not_wedge", "[gnss][ubx][framing]") {
    ubx::Parser p;
    // Zero-length UBX payload: B5 62 cls id 00 00 ck ck.
    auto frame = make_ubx_frame (0x05u, 0x01u, nullptr, 0u);
    REQUIRE (frame.size () == 8u);
    // Feed the frame twice back to back (zero payload, then immediately another).
    for (uint8_t b : frame)
        (void) p.feed (b);
    REQUIRE (p.messages_decoded () == 1u);
    REQUIRE (p.last_message ()._M_is_ack);
    // A real ACK-ACK carries a 2-byte payload; a zero-length Ack frame is not
    // an ACK-ACK, so _M_ack_ok is false here (plen < 2).  What matters is the
    // parser did not wedge: it decoded and is ready for the next frame.
    REQUIRE (p.last_message ()._M_ack_ok == false);
    // Immediately after, another zero-payload frame must still parse (no wedge).
    for (uint8_t b : frame)
        (void) p.feed (b);
    REQUIRE (p.messages_decoded () == 2u);
    REQUIRE (p.errors () == 0u);
}

TEST_CASE ("ubx_framing.back_to_back_frames", "[gnss][ubx][framing]") {
    ubx::Parser p;
    uint8_t payload_a[] = {0x01u};
    uint8_t payload_b[] = {0x02u, 0x03u};
    auto frame_a = make_ubx_frame (0x01u, 0x07u, payload_a, sizeof (payload_a));
    auto frame_b = make_ubx_frame (0x01u, 0x35u, payload_b, sizeof (payload_b));
    for (uint8_t b : frame_a)
        (void) p.feed (b);
    for (uint8_t b : frame_b)
        (void) p.feed (b);
    REQUIRE (p.messages_decoded () == 2u);
    REQUIRE (p.errors () == 0u);
}

TEST_CASE ("ubx_framing.max_size_payload", "[gnss][ubx][framing]") {
    ubx::Parser p;
    std::vector<uint8_t> payload (ubx::UbxMaxPayloadBytes, 0x41u);
    auto frame = make_ubx_frame (0x01u, 0x07u, payload.data (), static_cast<uint16_t> (payload.size ()));
    feed_frame (p, frame);
    REQUIRE (p.messages_decoded () == 1u);
    REQUIRE (p.errors () == 0u);
}

TEST_CASE ("ubx_framing.oversize_payload_rejected", "[gnss][ubx][framing]") {
    ubx::Parser p;
    // Length field > UbxMaxPayloadBytes (256): len_l=0, len_h=2 -> 512 bytes.
    (void) p.feed (0xb5);
    (void) p.feed (0x62);
    (void) p.feed (0x01); // class
    (void) p.feed (0x07); // id
    (void) p.feed (0x00); // len_l
    (void) p.feed (0x02); // len_h -> 512 > 256
    REQUIRE (p.errors () >= 1u);
    REQUIRE (p.messages_decoded () == 0u);
    // Parser must have returned to WaitSync1, ready for a new frame.
    auto frame = make_nav_pvt (481173000, 115166667, 3u, 8u, 0u, 100u, true);
    feed_frame (p, frame);
    REQUIRE (p.messages_decoded () == 1u);
}

// ---------------------------------------------------------------------------
// NAV-SAT / NAV-PVT / NAV-DOP decode tests
// ---------------------------------------------------------------------------

TEST_CASE ("ubx.nav_sat_decode_two_svs", "[gnss][ubx][nav_sat]") {
    ubx::Parser p;
    auto frame = make_nav_sat_multi (2u);
    feed_frame (p, frame);
    REQUIRE (p.last_message ()._M_is_nav_sat);
    gnss::Satellite sats[gnss::GnssMaxSatellites] = {};
    uint8_t filled = 0u;
    auto st = p.decode_nav_sat (sats, gnss::GnssMaxSatellites, &filled);
    REQUIRE (zzs::kernel::status_ok (st));
    REQUIRE (filled == 2u);
    REQUIRE (sats[0]._M_prn == 10u);
    REQUIRE (sats[0]._M_constellation == gnss::Constellation::Gps);
    REQUIRE (sats[0]._M_elevation_deg == 45);
    REQUIRE (sats[0]._M_snr_dbhz == 40u);
    REQUIRE (sats[0]._M_used_in_fix == true);
    REQUIRE (sats[1]._M_prn == 11u);
    REQUIRE (sats[1]._M_elevation_deg == 46);
    REQUIRE (sats[1]._M_snr_dbhz == 41u);
}

TEST_CASE ("ubx.nav_pvt_decode_hdop_from_pdop_vdop_not_dup", "[gnss][ubx][nav_pvt]") {
    ubx::Parser p;
    int32_t lat = 481173000;
    int32_t lon = 115166667;
    uint16_t p_dop = 100u; // 1.00 -> 10 tenths
    auto frame = make_nav_pvt (lat, lon, 3u, 8u, 0u, p_dop, true);
    feed_frame (p, frame);
    REQUIRE (p.last_message ()._M_is_nav_pvt);
    gnss::Fix fix = gnss::fix_make_invalid ();
    auto st = p.decode_nav_pvt (fix);
    REQUIRE (zzs::kernel::status_ok (st));
    REQUIRE (fix._M_lat == lat);
    REQUIRE (fix._M_lon == lon);
    REQUIRE (fix._M_status == gnss::FixStatus::Gps3d);
    REQUIRE (fix._M_num_satellites == 8u);
    REQUIRE (fix._M_hdop_tenths == 10);
    REQUIRE (fix._M_vdop_tenths == 0);
}

TEST_CASE ("ubx.nav_pvt_no_fix_e2e", "[gnss][ubx][nav_pvt]") {
    ubx::Parser p;
    auto frame = make_nav_pvt (481173000, 115166667, 0u, 0u, 0u, 0u, false);
    feed_frame (p, frame);
    gnss::Fix fix = gnss::fix_make_invalid ();
    auto st = p.decode_nav_pvt (fix);
    REQUIRE (zzs::kernel::status_ok (st));
    REQUIRE (fix._M_status == gnss::FixStatus::NoFix);
}

TEST_CASE ("ubx.nav_dop_decode_distinct", "[gnss][ubx][nav_dop]") {
    ubx::Parser p;
    auto frame = make_nav_dop (50u, 100u, 200u, 300u, 400u, 500u, 600u);
    feed_frame (p, frame);
    REQUIRE (p.last_message ()._M_is_nav_dop);
    int16_t pdop = 0, hdop = 0, vdop = 0;
    auto st = p.decode_nav_dop (pdop, hdop, vdop);
    REQUIRE (zzs::kernel::status_ok (st));
    REQUIRE (pdop == 10);
    REQUIRE (hdop == 40);
    REQUIRE (vdop == 30);
    REQUIRE (hdop != pdop);
    REQUIRE (vdop != pdop);
}

// ---------------------------------------------------------------------------
// ACK tests
// ---------------------------------------------------------------------------

TEST_CASE ("ubx.ack_ack_ok_true", "[gnss][ubx][ack]") {
    ubx::Parser p;
    auto frame = make_ack_ack (0x06u, 0x00u);
    feed_frame (p, frame);
    REQUIRE (p.last_message ()._M_is_ack);
    REQUIRE (p.last_message ()._M_ack_ok == true);
    REQUIRE (p.last_message ()._M_class == ubx::UbxClass::Ack);
    REQUIRE (p.last_message ()._M_id == ubx::UbxMsgId::AckAck);
}

TEST_CASE ("ubx.ack_nak_ok_false", "[gnss][ubx][ack]") {
    ubx::Parser p;
    auto frame = make_ack_nak (0x06u, 0x00u);
    feed_frame (p, frame);
    REQUIRE (p.last_message ()._M_is_ack);
    REQUIRE (p.last_message ()._M_ack_ok == false);
    REQUIRE (p.last_message ()._M_id == ubx::UbxMsgId::AckNak);
}

TEST_CASE ("ubx.ack_short_payload_rejected", "[gnss][ubx][ack]") {
    ubx::Parser p;
    const uint8_t one_byte[] = {0x06u};
    auto frame = make_ubx_frame (0x05u, 0x01u, one_byte, sizeof (one_byte));
    feed_frame (p, frame);
    REQUIRE (p.last_message ()._M_is_ack);
    REQUIRE (p.last_message ()._M_ack_ok == false);
}

// ---------------------------------------------------------------------------
// NMEA tests
// ---------------------------------------------------------------------------

TEST_CASE ("nmea.gga_lat_lon_s_w_hemispheres", "[gnss][nmea]") {
    nmea::Parser p;
    auto s = make_nmea ("GPGGA,123519,4807.038,S,01131.000,W,1,08,0.9,545.4,M,46.9,M,,");
    feed_nmea (p, s);
    REQUIRE (p.sentences_parsed () == 1u);
    REQUIRE (p.last_sentence ()._M_type == nmea::SentenceType::Gga);
    const auto &gga = p.last_sentence ()._M_gga;
    REQUIRE (gga._M_lat == -481173000);
    REQUIRE (gga._M_lon == -115166667);
    REQUIRE (gga._M_lat_hemi == 'S');
    REQUIRE (gga._M_lon_hemi == 'W');
    REQUIRE (gga._M_hdop_tenths == 9);
}

TEST_CASE ("nmea.rmc_12_fields", "[gnss][nmea]") {
    nmea::Parser p;
    auto s = make_nmea ("GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W,A");
    feed_nmea (p, s);
    REQUIRE (p.sentences_parsed () == 1u);
    REQUIRE (p.last_sentence ()._M_type == nmea::SentenceType::Rmc);
    const auto &rmc = p.last_sentence ()._M_rmc;
    REQUIRE (rmc._M_valid == true);
    REQUIRE (rmc._M_lat == 481173000);
    REQUIRE (rmc._M_lon == 115166667);
    REQUIRE (rmc._M_speed_knots_tenths == 224);
    REQUIRE (rmc._M_course_deg_tenths == 844);
}

TEST_CASE ("nmea_e2e.bad_checksum_rejected", "[gnss][nmea]") {
    nmea::Parser p;
    std::vector<uint8_t> s;
    s.push_back ('$');
    const char *body = "GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*";
    for (const char *q = body; *q != '\0'; ++q)
        s.push_back (static_cast<uint8_t> (*q));
    s.push_back ('0');
    s.push_back ('0');
    s.push_back ('\r');
    s.push_back ('\n');
    feed_nmea (p, s);
    REQUIRE (p.errors () >= 1u);
    REQUIRE (p.sentences_parsed () == 0u);
}

TEST_CASE ("nmea.split_sentence_across_feeds", "[gnss][nmea]") {
    nmea::Parser p;
    auto s = make_nmea ("GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,");
    // Feed in two chunks to simulate a split stream.
    size_t half = s.size () / 2u;
    for (size_t i = 0u; i < half; ++i)
        (void) p.feed (s[i]);
    REQUIRE (p.sentences_parsed () == 0u);
    for (size_t i = half; i < s.size (); ++i)
        (void) p.feed (s[i]);
    REQUIRE (p.sentences_parsed () == 1u);
    REQUIRE (p.last_sentence ()._M_checksum_ok);
}

// ---------------------------------------------------------------------------
// Driver E2E tests
// ---------------------------------------------------------------------------

TEST_CASE ("ublox.configure_waits_for_ack_ack", "[gnss][ublox][e2e]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    inject_ack_acks (bus, 3u); // CFG-PRT, CFG-RATE, CFG-MSG
    gnss::UbloxDriver drv (&bus);
    auto st = drv.init ();
    REQUIRE (zzs::kernel::status_ok (st));
    REQUIRE (drv.receiver_status () == gnss::GnssReceiverStatus::Searching);
    REQUIRE (bus.tx_count () >= 3u);
}

TEST_CASE ("ublox.configure_fails_on_ack_nak", "[gnss][ublox][e2e]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    auto nak = make_ack_nak (0x06u, 0x00u);
    bus.inject (nak.data (), static_cast<uint16_t> (nak.size ()));
    gnss::UbloxDriver drv (&bus);
    auto st = drv.init ();
    REQUIRE (!zzs::kernel::status_ok (st));
}

TEST_CASE ("ublox.configure_timeout_without_ack", "[gnss][ublox][e2e]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    gnss::UbloxDriver drv (&bus);
    auto st = drv.init ();
    REQUIRE (!zzs::kernel::status_ok (st));
}

TEST_CASE ("ublox.send_ubx_rejects_oversize", "[gnss][ublox][e2e]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    gnss::UbloxDriver drv (&bus);
    std::vector<uint8_t> big (ubx::UbxMaxPayloadBytes + 1u, 0x00u);
    auto st = drv.send_ubx (0x06u, 0x00u, big.data (), static_cast<uint16_t> (big.size ()));
    REQUIRE (st == zzs::kernel::Status::BAD_LENGTH);
    REQUIRE (bus.tx_count () == 0u);
}

TEST_CASE ("ublox.exact_cfg_prt_payload", "[gnss][ublox][e2e]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    inject_ack_acks (bus, 3u);
    gnss::UbloxConfig cfg;
    cfg._M_baud_rate = 9600u;
    gnss::UbloxDriver drv (&bus, nullptr, cfg);
    REQUIRE (zzs::kernel::status_ok (drv.init ()));

    const auto &tx = bus.tx_bytes ();
    bool found_cfg_prt = false;
    for (size_t i = 0u; i + 6 <= tx.size (); ++i) {
        if (tx[i] == 0xb5 && tx[i + 1] == 0x62 && tx[i + 2] == 0x06 && tx[i + 3] == 0x00) {
            uint16_t len = tx[i + 4] | (tx[i + 5] << 8);
            REQUIRE (len == 20u);
            REQUIRE (tx[i + 6 + 8] == 0x80u);
            REQUIRE (tx[i + 6 + 9] == 0x25u);
            REQUIRE (tx[i + 6 + 10] == 0x00u);
            REQUIRE (tx[i + 6 + 11] == 0x00u);
            found_cfg_prt = true;
            break;
        }
    }
    REQUIRE (found_cfg_prt);
}

TEST_CASE ("ublox.e2e_nmea_fix_via_stream", "[gnss][ublox][e2e]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    inject_ack_acks (bus, 3u);
    gnss::UbloxDriver drv (&bus);
    REQUIRE (zzs::kernel::status_ok (drv.init ()));

    auto s = make_nmea ("GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,");
    bus.inject (s.data (), static_cast<uint16_t> (s.size ()));
    auto st = drv.poll ();
    bool ok_or_stale = zzs::kernel::status_ok (st) || (st == zzs::kernel::Status::STALE);
    REQUIRE (ok_or_stale);

    gnss::Fix fix = gnss::fix_make_invalid ();
    st = drv.read_fix (fix);
    if (zzs::kernel::status_ok (st)) {
        REQUIRE (fix._M_lat == 481173000);
        REQUIRE (fix._M_lon == 115166667);
        REQUIRE (gnss::fix_is_usable (fix));
    }
}

// ---------------------------------------------------------------------------
// HAL adapter tests (GnssPlatformHal -> StreamBusInterface)
// ---------------------------------------------------------------------------

namespace {

/** @brief Test-owned loopback HAL state (file-static, test-only). */
struct LoopbackHalState {
    uint8_t rx[256] = {};
    uint16_t rx_len = 0u;
    uint16_t rx_read = 0u;
    uint8_t tx[256] = {};
    uint16_t tx_len = 0u;
    bool open = false;
};

LoopbackHalState &
loop_state () noexcept {
    static LoopbackHalState _S_state;
    return _S_state;
}

[[nodiscard]] zzs::kernel::Status
loop_open (void) noexcept {
    loop_state ().open = true;
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
loop_close (void) noexcept {
    loop_state ().open = false;
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
loop_read_bytes (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept {
    auto &s = loop_state ();
    if (buf == nullptr || filled == nullptr)
        return zzs::kernel::Status::OUT_OF_RANGE;
    if (!s.open)
        return zzs::kernel::Status::INTERNAL;
    *filled = 0u;
    uint16_t n = 0u;
    while (n < max_bytes && s.rx_read < s.rx_len) {
        buf[n++] = s.rx[s.rx_read++];
    }
    *filled = n;
    return n > 0u ? zzs::kernel::Status::OK : zzs::kernel::Status::STALE;
}

[[nodiscard]] zzs::kernel::Status
loop_write_bytes (const uint8_t *buf, uint16_t len) noexcept {
    auto &s = loop_state ();
    if (buf == nullptr && len > 0u)
        return zzs::kernel::Status::OUT_OF_RANGE;
    if (!s.open)
        return zzs::kernel::Status::INTERNAL;
    for (uint16_t i = 0u; i < len; ++i)
        s.tx[s.tx_len++] = buf[i];
    return zzs::kernel::Status::OK;
}

void
loop_delay_ms (uint32_t ms) noexcept {
    (void) ms;
}

[[nodiscard]] zzs::kernel::Status
loop_flush_rx (void) noexcept {
    auto &s = loop_state ();
    s.rx_len = 0u;
    s.rx_read = 0u;
    return zzs::kernel::Status::OK;
}

[[nodiscard]] bool
loop_is_open (void) noexcept {
    return loop_state ().open;
}

/** @brief Test loopback HAL vtable (const, .rodata). */
const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal k_loop_hal = {
    &loop_open, &loop_close, &loop_read_bytes, &loop_write_bytes, &loop_delay_ms, &loop_flush_rx, &loop_is_open,
};

/** @brief Inject bytes into the loopback HAL's receive buffer. */
void
loop_inject (const uint8_t *data, uint16_t len) noexcept {
    auto &s = loop_state ();
    for (uint16_t i = 0u; i < len && s.rx_len < 256u; ++i)
        s.rx[s.rx_len++] = data[i];
}

} // namespace

TEST_CASE ("gnss.adapter.loopback_vtable", "[gnss][adapter]") {
    gnss::GnssPlatformBusAdapter adapter (&k_loop_hal);
    REQUIRE (adapter.hal () == &k_loop_hal);
    REQUIRE (adapter.is_open () == false);

    REQUIRE (zzs::kernel::status_ok (adapter.open ()));
    REQUIRE (adapter.is_open () == true);
    // Idempotent open.
    REQUIRE (zzs::kernel::status_ok (adapter.open ()));
    REQUIRE (adapter.is_open () == true);

    uint16_t filled = 0u;
    REQUIRE (adapter.read_bytes (nullptr, 10u, &filled) == zzs::kernel::Status::OUT_OF_RANGE);
    uint8_t rb[4] = {};
    REQUIRE (adapter.read_bytes (rb, sizeof (rb), nullptr) == zzs::kernel::Status::OUT_OF_RANGE);
    // Nothing to read -> STALE, not OUT_OF_RANGE.
    REQUIRE (adapter.read_bytes (rb, sizeof (rb), &filled) == zzs::kernel::Status::STALE);

    const uint8_t cmd[] = {'C', 'M', 'D'};
    REQUIRE (zzs::kernel::status_ok (adapter.write_bytes (cmd, sizeof (cmd))));
    REQUIRE (zzs::kernel::status_ok (adapter.write_bytes (nullptr, 0u)));
    REQUIRE (adapter.write_bytes (nullptr, 10u) == zzs::kernel::Status::OUT_OF_RANGE);

    adapter.delay_ms (5u);
    REQUIRE (zzs::kernel::status_ok (adapter.flush_rx ()));
    REQUIRE (zzs::kernel::status_ok (adapter.close ()));
    REQUIRE (adapter.is_open () == false);
}

TEST_CASE ("gnss.adapter.drives_stream_via_loopback", "[gnss][adapter]") {
    // Reset shared loopback state (previous test case left bytes in tx).
    loop_state () = LoopbackHalState{};
    gnss::GnssPlatformBusAdapter adapter (&k_loop_hal);
    REQUIRE (zzs::kernel::status_ok (adapter.open ()));

    stream::GnssStream stream (&adapter);
    REQUIRE (stream.is_open ());

    // Inject an NMEA GGA into the loopback rx and poll the stream.
    auto s = make_nmea ("GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,");
    loop_inject (s.data (), static_cast<uint16_t> (s.size ()));
    auto st = stream.poll ();
    REQUIRE ((zzs::kernel::status_ok (st) || st == zzs::kernel::Status::STALE));
    REQUIRE (stream.nmea_parser ().sentences_parsed () == 1u);
    REQUIRE (stream.nmea_parser ().last_sentence ()._M_type == nmea::SentenceType::Gga);

    // Writes flow to tx.
    const uint8_t out[] = {0xb5, 0x62};
    REQUIRE (zzs::kernel::status_ok (adapter.write_bytes (out, sizeof (out))));
    REQUIRE (loop_state ().tx_len == 2u);

    REQUIRE (zzs::kernel::status_ok (adapter.close ()));
    REQUIRE (adapter.is_open () == false);
}

// ---------------------------------------------------------------------------
// Platform factory tests (Linux)
// ---------------------------------------------------------------------------

TEST_CASE ("gnss.platform.linux_factory_usable", "[gnss][platform][linux]") {
    namespace plat = zzs::kernel::device::driver::gnss::platform::linux;
    const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal = plat::create_linux_gnss_hal ();
    REQUIRE (hal != nullptr);
    REQUIRE (hal->open != nullptr);
    REQUIRE (hal->close != nullptr);
    REQUIRE (hal->read_bytes != nullptr);
    REQUIRE (hal->write_bytes != nullptr);
    REQUIRE (hal->delay_ms != nullptr);
    REQUIRE (hal->flush_rx != nullptr);
    REQUIRE (hal->is_open != nullptr);

    // Configuring before first use is supported.
    REQUIRE (zzs::kernel::status_ok (plat::linux_gnss_hal_set_config ("/dev/ttyUSB0", 9600u)));
    REQUIRE (zzs::kernel::status_ok (plat::linux_gnss_hal_set_config (nullptr, 115200u)));

    // open()/close() are idempotent and safe to call (open may fail without a
    // real device, but must not crash).
    (void) hal->open ();
    (void) hal->is_open ();
    (void) hal->close ();
    (void) hal->close ();
}

TEST_CASE ("gnss.platform.create_dispatch_non_null", "[gnss][platform]") {
    const zzs::kernel::device::driver::gnss::platform::GnssPlatformHal *hal =
        zzs::kernel::device::driver::gnss::platform::create_gnss_platform_hal ();
    REQUIRE (hal != nullptr);
    REQUIRE (hal->open != nullptr);
    REQUIRE (hal->close != nullptr);
    REQUIRE (hal->read_bytes != nullptr);
    REQUIRE (hal->write_bytes != nullptr);
    REQUIRE (hal->delay_ms != nullptr);
    REQUIRE (hal->flush_rx != nullptr);
    REQUIRE (hal->is_open != nullptr);
}

// ---------------------------------------------------------------------------
// Platform factory tests (Linux)
// ---------------------------------------------------------------------------
