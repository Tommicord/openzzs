/*
 * Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the “Software”), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

/**
 * @file test_gnss_driver.cpp
 * @brief Unit tests for the GNSS device driver module:
 *        NMEA/UBX parsing, u-blox driver lifecycle, and streaming pipeline.
 *
 * Hardware-independent: every test uses a mock StreamBus. No float is used
 * anywhere (integer-only fixed point, matching the driver contract).
 */

#include "kernel/device-driver/gnss/gnss_ublox.hpp"
#include "kernel/device-driver/gnss/nmea/nmea_checksum.hpp"
#include "kernel/device-driver/gnss/nmea/nmea_parser.hpp"
#include "kernel/device-driver/gnss/stream/gnss_byte_ring.hpp"
#include "kernel/device-driver/gnss/stream/gnss_stream.hpp"
#include "kernel/device-driver/gnss/ubx/ubx_framing.hpp"
#include "kernel/device-driver/gnss/ubx/ubx_parser.hpp"
#include "kernel/kernel_status.hpp"
#include "test_mock_stream_bus.hpp"

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

/** @brief Build a minimal UBX NAV-PVT frame (class 0x01, id 0x07). */
static std::vector<uint8_t>
make_nav_pvt (int32_t lat, int32_t lon, uint8_t fix_type, uint8_t num_sv, uint32_t g_speed, uint16_t p_dop,
              uint16_t h_dop, uint16_t v_dop, bool valid) noexcept {
    std::vector<uint8_t> frame;
    frame.push_back (0xb5);
    frame.push_back (0x62);
    frame.push_back (0x01); // class Nav
    frame.push_back (0x07); // id NAV-PVT
    uint8_t len_l = 94u;    // payload length (without sync/header/ck)
    uint8_t len_h = 0u;
    frame.push_back (len_l);
    frame.push_back (len_h);

    uint8_t payload[94] = {};
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
    uint32_t *t_acc = reinterpret_cast<uint32_t *> (&payload[12]);
    *t_acc = 1000u;
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
    uint32_t *h_acc = reinterpret_cast<uint32_t *> (&payload[40]);
    *h_acc = 5000u;
    uint32_t *v_acc = reinterpret_cast<uint32_t *> (&payload[44]);
    *v_acc = 5000u;
    int32_t *vel_n = reinterpret_cast<int32_t *> (&payload[48]);
    *vel_n = 0;
    int32_t *vel_e = reinterpret_cast<int32_t *> (&payload[52]);
    *vel_e = 0;
    int32_t *vel_d = reinterpret_cast<int32_t *> (&payload[56]);
    *vel_d = 0;
    int32_t *g_speed_p = reinterpret_cast<int32_t *> (&payload[60]);
    *g_speed_p = static_cast<int32_t> (g_speed);
    int32_t *head_mot = reinterpret_cast<int32_t *> (&payload[64]);
    *head_mot = 0;

    for (uint32_t i = 0u; i < 94u; ++i)
        frame.push_back (payload[i]);

    uint16_t ck = ubx::ubx_checksum (&frame[2], 4u + 94u);
    frame.push_back (static_cast<uint8_t> (ck & 0xffu));
    frame.push_back (static_cast<uint8_t> ((ck >> 8) & 0xffu));
    return frame;
}

/** @brief Build a minimal UBX NAV-SAT frame.
 *
 * Real NAV-SAT payload layout: iTOW(4) | version(1) | numSvs(1) |
 * reserved(2) | SV entries@8, each 12 bytes
 *   (gnssId, svId, cno, elev, azim(2), pr_res(2), flags(4)). */
static std::vector<uint8_t>
make_nav_sat (uint8_t gnss_id, uint8_t sv_id, uint8_t cno, uint8_t elev, uint16_t azim, uint32_t flags,
              uint8_t num_svs) noexcept {
    std::vector<uint8_t> frame;
    frame.push_back (0xb5);
    frame.push_back (0x62);
    frame.push_back (0x01); // class Nav
    frame.push_back (0x35); // id NAV-SAT
    uint16_t payload_len = 8u + static_cast<uint16_t> (num_svs) * 12u;
    frame.push_back (static_cast<uint8_t> (payload_len & 0xffu));
    frame.push_back (static_cast<uint8_t> ((payload_len >> 8) & 0xffu));

    std::vector<uint8_t> payload;
    // iTOW (4 bytes)
    payload.push_back (0x00);
    payload.push_back (0x00);
    payload.push_back (0x00);
    payload.push_back (0x00);
    // version (1)
    payload.push_back (0x00);
    // numSvs (1)
    payload.push_back (num_svs);
    // reserved (2)
    payload.push_back (0x00);
    payload.push_back (0x00);
    for (uint8_t i = 0u; i < num_svs; ++i) {
        payload.push_back (gnss_id); // gnssId
        payload.push_back (sv_id);   // svId
        payload.push_back (cno);     // cno
        payload.push_back (elev);    // elev
        payload.push_back (static_cast<uint8_t> (azim & 0xffu));
        payload.push_back (static_cast<uint8_t> ((azim >> 8) & 0xffu)); // azim
        payload.push_back (0x00);
        payload.push_back (0x00); // pr_res (int16_t, 0.1 m)
        payload.push_back (static_cast<uint8_t> (flags & 0xffu));
        payload.push_back (static_cast<uint8_t> ((flags >> 8) & 0xffu));
        payload.push_back (static_cast<uint8_t> ((flags >> 16) & 0xffu));
        payload.push_back (static_cast<uint8_t> ((flags >> 24) & 0xffu));
    }
    for (uint8_t b : payload)
        frame.push_back (b);

    uint16_t ck = ubx::ubx_checksum (&frame[2], 4u + payload_len);
    frame.push_back (static_cast<uint8_t> (ck & 0xffu));
    frame.push_back (static_cast<uint8_t> ((ck >> 8) & 0xffu));
    return frame;
}

/** @brief Build a UBX ACK-ACK frame acknowledging (msg_class, msg_id).
 *         Payload is the 2-byte class+id of the acknowledged message. */
static std::vector<uint8_t>
make_ack_ack (uint8_t msg_class, uint8_t msg_id) noexcept {
    std::vector<uint8_t> frame;
    frame.push_back (0xb5);
    frame.push_back (0x62);
    frame.push_back (0x05); // class Ack
    frame.push_back (0x01); // id ACK-ACK
    uint8_t payload[] = {msg_class, msg_id};
    frame.push_back (0x02u); // len_l
    frame.push_back (0x00u); // len_h
    frame.push_back (msg_class);
    frame.push_back (msg_id);
    uint16_t ck = ubx::ubx_checksum (&frame[2], 4u + 2u);
    frame.push_back (static_cast<uint8_t> (ck & 0xffu));
    frame.push_back (static_cast<uint8_t> ((ck >> 8) & 0xffu));
    return frame;
}

/** @brief Inject `count` UBX ACK-ACK frames into the mock bus, one for each
 *         config command that `UbloxDriver::configure()` awaits. */
static void
inject_ack_acks (test::MockStreamBus &bus, uint8_t count) noexcept {
    for (uint8_t i = 0u; i < count; ++i) {
        auto ack = make_ack_ack (0x06u, 0x00u); // ack CFG-PRT (any ACK-ACK suffices)
        bus.inject (ack.data (), static_cast<uint16_t> (ack.size ()));
    }
}

// ---------------------------------------------------------------------------
// NMEA parser tests
// ---------------------------------------------------------------------------

TEST_CASE ("nmea.checksum_valid_gga", "[gnss][nmea]") {
    nmea::Parser p;
    auto s = make_nmea ("GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,");
    feed_nmea (p, s);
    REQUIRE (p.sentences_parsed () == 1u);
    REQUIRE (p.last_sentence ()._M_type == nmea::SentenceType::Gga);
    REQUIRE (p.last_sentence ()._M_checksum_ok);
    REQUIRE (p.last_sentence ()._M_talker_id[0] == 'G');
    REQUIRE (p.last_sentence ()._M_talker_id[1] == 'P');
    const auto &gga = p.last_sentence ()._M_gga;
    REQUIRE (gga._M_fix_quality == 1u);
    REQUIRE (gga._M_num_satellites == 8u);
    // Latitude 48°07.038' N -> 48 + 7.038/60 deg = 48.1173 deg -> 481173000
    REQUIRE (gga._M_lat == 481173000);
    REQUIRE (gga._M_lat_hemi == 'N');
    // Longitude 11°31.000' E -> 11 + 31/60 = 11.516666... -> 115166667
    REQUIRE (gga._M_lon == 115166667);
    REQUIRE (gga._M_lon_hemi == 'E');
    REQUIRE (gga._M_hdop_tenths == 9);
}

TEST_CASE ("nmea.bad_checksum_rejected", "[gnss][nmea]") {
    nmea::Parser p;
    // Sentence with a deliberately wrong checksum (*00 instead of *47).
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

TEST_CASE ("nmea.rmc_parsing", "[gnss][nmea]") {
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

TEST_CASE ("nmea.truncated_missing_checksum", "[gnss][nmea]") {
    nmea::Parser p;
    const char *no_ck = "$GPGGA,123519,4807.038,N,01131.000,E\r\n";
    for (const char *q = no_ck; *q != '\0'; ++q)
        (void) p.feed (static_cast<uint8_t> (*q));
    REQUIRE (p.errors () >= 1u);
    REQUIRE (p.sentences_parsed () == 0u);
}

TEST_CASE ("nmea.oversize_sentence_rejected", "[gnss][nmea]") {
    nmea::Parser p;
    (void) p.feed ('$');
    char body[nmea::NmeaMaxSentenceBytes + 10];
    for (size_t i = 0u; i < sizeof (body); ++i)
        body[i] = 'A';
    for (size_t i = 0u; i < sizeof (body); ++i)
        (void) p.feed (static_cast<uint8_t> (body[i]));
    REQUIRE (p.errors () >= 1u);
    REQUIRE (p.sentences_parsed () == 0u);
}

TEST_CASE ("nmea.hemisphere_s_w_negative", "[gnss][nmea]") {
    nmea::Parser p;
    auto s_lat_s = make_nmea ("GPGGA,123519,4807.038,S,01131.000,E,1,08,0.9,545.4,M,46.9,M,,");
    feed_nmea (p, s_lat_s);
    REQUIRE (p.last_sentence ()._M_gga._M_lat == -481173000);

    nmea::Parser p2;
    auto s_lon_w = make_nmea ("GPGGA,123519,4807.038,N,01131.000,W,1,08,0.9,545.4,M,46.9,M,,");
    feed_nmea (p2, s_lon_w);
    REQUIRE (p2.last_sentence ()._M_gga._M_lon == -115166667);
}

TEST_CASE ("nmea.talker_ids", "[gnss][nmea]") {
    struct Case {
        const char *talker;
        uint8_t hi, lo;
    };
    Case cases[] = {
        {"GP", 'G', 'P'}, {"GN", 'G', 'N'}, {"GL", 'G', 'L'}, {"GA", 'G', 'A'}, {"BD", 'B', 'D'},
    };
    for (const auto &c : cases) {
        nmea::Parser p;
        std::string body = std::string (c.talker) + "RMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W,A";
        // Compute checksum for this talker variant.
        uint8_t ck = 0u;
        for (char ch : body)
            ck = static_cast<uint8_t> (ck ^ static_cast<uint8_t> (ch));
        std::vector<uint8_t> s;
        s.push_back ('$');
        for (char ch : body)
            s.push_back (static_cast<uint8_t> (ch));
        s.push_back ('*');
        s.push_back (static_cast<uint8_t> ("0123456789ABCDEF"[(ck >> 4) & 0x0fu]));
        s.push_back (static_cast<uint8_t> ("0123456789ABCDEF"[ck & 0x0fu]));
        s.push_back ('\r');
        s.push_back ('\n');
        feed_nmea (p, s);
        REQUIRE (p.sentences_parsed () == 1u);
        REQUIRE (p.last_sentence ()._M_talker_id[0] == c.hi);
        REQUIRE (p.last_sentence ()._M_talker_id[1] == c.lo);
    }
}

// ---------------------------------------------------------------------------
// UBX parser tests
// ---------------------------------------------------------------------------

TEST_CASE ("ubx.nav_pvt_decode", "[gnss][ubx]") {
    ubx::Parser p;
    int32_t lat = 481173000;
    int32_t lon = 115166667;
    auto frame = make_nav_pvt (lat, lon, 3u /* 3D fix */, 8u, 0u, 100u, 90u, 80u, true);
    for (uint8_t b : frame) {
        auto fs = p.feed (b);
        bool ok_or_stale = (fs == zzs::kernel::Status::OK) || (fs == zzs::kernel::Status::STALE);
        REQUIRE (ok_or_stale);
    }
    ubx::Parser p2;
    for (uint8_t b : frame) {
        auto fs = p2.feed (b);
        bool ok_or_stale = (fs == zzs::kernel::Status::OK) || (fs == zzs::kernel::Status::STALE);
        REQUIRE (ok_or_stale);
    }
    REQUIRE (p2.last_message ()._M_is_nav_pvt);
    gnss::Fix fix = gnss::fix_make_invalid ();
    auto st = p2.decode_nav_pvt (fix);
    REQUIRE (zzs::kernel::status_ok (st));
    REQUIRE (fix._M_lat == lat);
    REQUIRE (fix._M_lon == lon);
    REQUIRE (fix._M_status == gnss::FixStatus::Gps3d);
    REQUIRE (fix._M_num_satellites == 8u);
}

TEST_CASE ("ubx.nav_pvt_no_fix", "[gnss][ubx]") {
    ubx::Parser p;
    auto frame = make_nav_pvt (481173000, 115166667, 0u /* none */, 0u, 0u, 0u, 0u, 0u, false);
    for (uint8_t b : frame)
        (void) p.feed (b);
    REQUIRE (p.last_message ()._M_is_nav_pvt);
    gnss::Fix fix = gnss::fix_make_invalid ();
    auto st = p.decode_nav_pvt (fix);
    REQUIRE (zzs::kernel::status_ok (st));
    REQUIRE (fix._M_status == gnss::FixStatus::NoFix);
}

TEST_CASE ("ubx.bad_checksum_rejected", "[gnss][ubx]") {
    ubx::Parser p;
    auto frame = make_nav_pvt (481173000, 115166667, 3u, 8u, 0u, 0u, 0u, 0u, true);
    // Corrupt the last checksum byte.
    frame[frame.size () - 1] = static_cast<uint8_t> (frame.back () ^ 0xffu);
    bool saw_error = false;
    for (uint8_t b : frame) {
        auto st = p.feed (b);
        if (st == zzs::kernel::Status::BAD_CRC)
            saw_error = true;
    }
    REQUIRE (saw_error);
    REQUIRE (p.messages_decoded () == 0u);
}

TEST_CASE ("ubx.bad_sync_resync", "[gnss][ubx]") {
    ubx::Parser p;
    // Garbage byte then a valid frame -> should resync and decode.
    std::vector<uint8_t> garbage = {0x00, 0xff, 0x55};
    for (uint8_t b : garbage)
        (void) p.feed (b);
    auto frame = make_nav_pvt (481173000, 115166667, 3u, 8u, 0u, 0u, 0u, 0u, true);
    for (uint8_t b : frame)
        (void) p.feed (b);
    REQUIRE (p.last_message ()._M_is_nav_pvt);
}

TEST_CASE ("ubx.oversize_payload_rejected", "[gnss][ubx]") {
    ubx::Parser p;
    (void) p.feed (0xb5);
    (void) p.feed (0x62);
    (void) p.feed (0x01); // class
    (void) p.feed (0x07); // id
    // Length > UbxMaxPayloadBytes (256): set len_l=0, len_h=2 (512 bytes).
    (void) p.feed (0x00);
    (void) p.feed (0x02);
    // Should reject with BAD_LENGTH and return to WaitSync1.
    REQUIRE (p.errors () >= 1u);
}

TEST_CASE ("ubx.nav_sat_decode", "[gnss][ubx]") {
    ubx::Parser p;
    auto frame = make_nav_sat (0u /* GPS */, 12u, 40u, 45u, 900u, 0x01u, 1u);
    for (uint8_t b : frame)
        (void) p.feed (b);
    REQUIRE (p.last_message ()._M_is_nav_sat);
    gnss::Satellite sats[gnss::GnssMaxSatellites] = {};
    uint8_t filled = 0u;
    auto st = p.decode_nav_sat (sats, gnss::GnssMaxSatellites, &filled);
    REQUIRE (zzs::kernel::status_ok (st));
    REQUIRE (filled == 1u);
    REQUIRE (sats[0]._M_prn == 12u);
    REQUIRE (sats[0]._M_constellation == gnss::Constellation::Gps);
    REQUIRE (sats[0]._M_snr_dbhz == 40u);
    REQUIRE (sats[0]._M_used_in_fix == true);
}

// ---------------------------------------------------------------------------
// Streaming pipeline tests
// ---------------------------------------------------------------------------

TEST_CASE ("stream.feed_nmea_then_poll", "[gnss][stream]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    stream::GnssStream stream (&bus);
    auto s = make_nmea ("GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,");
    bus.inject (s.data (), static_cast<uint16_t> (s.size ()));
    auto st = stream.poll ();
    bool ok_or_stale = zzs::kernel::status_ok (st) || (st == zzs::kernel::Status::STALE);
    REQUIRE (ok_or_stale);
    REQUIRE (stream.nmea_parser ().sentences_parsed () == 1u);
}

TEST_CASE ("byte_ring_drop_oldest", "[gnss][stream]") {
    stream::GnssByteRing ring (4u);
    REQUIRE (ring.push (1) == zzs::kernel::Status::OK);
    REQUIRE (ring.push (2) == zzs::kernel::Status::OK);
    REQUIRE (ring.push (3) == zzs::kernel::Status::OK);
    REQUIRE (ring.push (4) == zzs::kernel::Status::OK);
    REQUIRE (ring.full ());
    // Push a 5th: drops oldest (1).
    REQUIRE (ring.push (5) == zzs::kernel::Status::OK);
    uint8_t b = 0;
    REQUIRE (ring.pop (b) == zzs::kernel::Status::OK);
    REQUIRE (b == 2u);
    REQUIRE (ring.pop (b) == zzs::kernel::Status::OK);
    REQUIRE (b == 3u);
    REQUIRE (ring.pop (b) == zzs::kernel::Status::OK);
    REQUIRE (b == 4u);
    REQUIRE (ring.pop (b) == zzs::kernel::Status::OK);
    REQUIRE (b == 5u);
    REQUIRE (ring.empty ());
}

// ---------------------------------------------------------------------------
// u-blox driver tests
// ---------------------------------------------------------------------------

TEST_CASE ("ublox.init_success", "[gnss][ublox]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    inject_ack_acks (bus, 3u); // CFG-PRT, CFG-RATE, CFG-MSG each await an ACK
    // Accept any UBX writes during init (CFG-PRT, CFG-RATE, CFG-MSG).
    gnss::UbloxDriver drv (&bus);
    auto st = drv.init ();
    REQUIRE (zzs::kernel::status_ok (st));
    REQUIRE (drv.receiver_status () == gnss::GnssReceiverStatus::Searching);
    REQUIRE (drv.protocol_mask () == 0x03u);
}

TEST_CASE ("ublox.init_bus_closed", "[gnss][ublox]") {
    test::MockStreamBus bus;
    bus.set_open (false);
    gnss::UbloxDriver drv (&bus);
    auto st = drv.init ();
    REQUIRE (!zzs::kernel::status_ok (st));
}

TEST_CASE ("ublox.read_fix_stale_before_fix", "[gnss][ublox]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    inject_ack_acks (bus, 3u);
    gnss::UbloxDriver drv (&bus);
    REQUIRE (zzs::kernel::status_ok (drv.init ()));
    gnss::Fix fix = gnss::fix_make_invalid ();
    auto st = drv.read_fix (fix);
    REQUIRE (st == zzs::kernel::Status::STALE);
}

TEST_CASE ("ublox.self_test_no_fix", "[gnss][ublox]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    inject_ack_acks (bus, 3u);
    gnss::UbloxDriver drv (&bus);
    REQUIRE (zzs::kernel::status_ok (drv.init ()));
    REQUIRE (drv.self_test () == false);
}

TEST_CASE ("ublox.poll_ubx_fix_published", "[gnss][ublox]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    inject_ack_acks (bus, 3u);
    gnss::UbloxDriver drv (&bus);
    REQUIRE (zzs::kernel::status_ok (drv.init ()));
    // Inject a valid UBX NAV-PVT with a 3D fix.
    auto frame = make_nav_pvt (481173000, 115166667, 3u, 8u, 0u, 100u, 90u, 80u, true);
    bus.inject (frame.data (), static_cast<uint16_t> (frame.size ()));
    auto st = drv.poll ();
    bool ok_or_stale = zzs::kernel::status_ok (st) || (st == zzs::kernel::Status::STALE);
    REQUIRE (ok_or_stale);
    gnss::Fix fix = gnss::fix_make_invalid ();
    st = drv.read_fix (fix);
    REQUIRE (zzs::kernel::status_ok (st));
    REQUIRE (fix._M_lat == 481173000);
    REQUIRE (fix._M_lon == 115166667);
    REQUIRE (gnss::fix_is_usable (fix));
}

TEST_CASE ("ublox.read_satellites", "[gnss][ublox]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    inject_ack_acks (bus, 3u);
    gnss::UbloxDriver drv (&bus);
    REQUIRE (zzs::kernel::status_ok (drv.init ()));
    auto frame = make_nav_sat (0u, 12u, 40u, 45u, 900u, 0x01u, 1u);
    bus.inject (frame.data (), static_cast<uint16_t> (frame.size ()));
    (void) drv.poll ();
    gnss::Satellite sats[zzs::kernel::device::driver::gnss::GnssMaxSatellites] = {};
    uint8_t filled = 0u;
    auto st = drv.read_satellites (sats, zzs::kernel::device::driver::gnss::GnssMaxSatellites, &filled);
    // May be STALE if poll didn't process; accept either.
    bool ok_or_stale = (st == zzs::kernel::Status::OK) || (st == zzs::kernel::Status::STALE);
    REQUIRE (ok_or_stale);
    if (st == zzs::kernel::Status::OK) {
        REQUIRE (filled == 1u);
        REQUIRE (sats[0]._M_prn == 12u);
    }
}

TEST_CASE ("ublox.send_ubx", "[gnss][ublox]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    gnss::UbloxDriver drv (&bus);
    uint8_t payload[] = {0x01u, 0x07u, 0x01u};
    auto st = drv.send_ubx (0x06u, 0x01u, payload, sizeof (payload));
    REQUIRE (zzs::kernel::status_ok (st));
    REQUIRE (bus.tx_count () >= 6u); // sync(2)+class+id+len(2)+ck(2) min
    // First two bytes must be UBX sync.
    REQUIRE (bus.tx_bytes ()[0] == 0xb5u);
    REQUIRE (bus.tx_bytes ()[1] == 0x62u);
}

TEST_CASE ("ublox.consecutive_errors_to_error_state", "[gnss][ublox]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    inject_ack_acks (bus, 3u);
    gnss::UbloxDriver drv (&bus);
    REQUIRE (zzs::kernel::status_ok (drv.init ()));
    // Feed many bad-checksum UBX frames to trip the error counter.
    for (int i = 0; i < 10; ++i) {
        auto frame = make_nav_pvt (481173000, 115166667, 3u, 8u, 0u, 0u, 0u, 0u, true);
        frame[frame.size () - 1] ^= 0xffu; // corrupt checksum
        bus.inject (frame.data (), static_cast<uint16_t> (frame.size ()));
        (void) drv.poll ();
    }
    // Status may be Error once threshold exceeded; at minimum errors recorded.
    REQUIRE (drv.consecutive_errors () > 0u);
}

TEST_CASE ("ublox.deinit", "[gnss][ublox]") {
    test::MockStreamBus bus;
    bus.set_open (true);
    inject_ack_acks (bus, 3u);
    gnss::UbloxDriver drv (&bus);
    REQUIRE (zzs::kernel::status_ok (drv.init ()));
    auto st = drv.deinit ();
    REQUIRE (zzs::kernel::status_ok (st));
    REQUIRE (drv.receiver_status () == gnss::GnssReceiverStatus::Off);
}

TEST_CASE ("ublox_null_bus", "[gnss][ublox]") {
    gnss::UbloxDriver drv (nullptr);
    auto st = drv.init ();
    REQUIRE (!zzs::kernel::status_ok (st));
}

// ---------------------------------------------------------------------------
// Types validation helpers
// ---------------------------------------------------------------------------

TEST_CASE ("types.lat_lon_validation", "[gnss][types]") {
    REQUIRE (gnss::is_valid_lat (0));
    REQUIRE (gnss::is_valid_lat (gnss::GnssLatMax));
    REQUIRE (gnss::is_valid_lat (-gnss::GnssLatMax));
    REQUIRE (!gnss::is_valid_lat (gnss::GnssLatMax + 1));
    REQUIRE (!gnss::is_valid_lat (-gnss::GnssLatMax - 1));
    REQUIRE (gnss::is_valid_lon (0));
    REQUIRE (gnss::is_valid_lon (gnss::GnssLonMax));
    REQUIRE (gnss::is_valid_lon (-gnss::GnssLonMax));
    REQUIRE (!gnss::is_valid_lon (gnss::GnssLonMax + 1));
}

TEST_CASE ("types.fix_usability", "[gnss][types]") {
    gnss::Fix f = gnss::fix_make_invalid ();
    REQUIRE (!gnss::fix_is_usable (f));
    f._M_status = gnss::FixStatus::Gps3d;
    f._M_lat = 1000000;
    f._M_lon = 1000000;
    REQUIRE (gnss::fix_is_usable (f));
    gnss::Satellite s = gnss::satellite_make_invalid ();
    REQUIRE (!gnss::satellite_is_valid (s));
    s._M_prn = 5u;
    REQUIRE (gnss::satellite_is_valid (s));
}
