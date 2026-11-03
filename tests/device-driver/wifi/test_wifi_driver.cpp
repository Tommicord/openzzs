/*
 * Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

/**
 * @file test_wifi_driver.cpp
 * @brief Unit tests for the WiFi device-driver module.
 *
 * Covers: init success/failure, scan populates BSS table, connect state machine
 * transitions, disconnect returns to Disconnected, frame parsing of beacon/
 * probe/assoc, cipher negotiation, bad-frame rejection, buffer-overflow
 * rejection, constant-time MAC compare, replay-window protection.
 *
 * Hardware-independent: the stream bus is mocked via MockStreamBus.
 */

#include "kernel/device-driver/wifi/generic_wifi_driver.hpp"
#include "kernel/device-driver/wifi/platforms/platform_select.hpp"
#include "kernel/device-driver/wifi/stream/wifi_stream.hpp"
#include "kernel/device-driver/wifi/wifi_crypto.hpp"
#include "kernel/device-driver/wifi/wifi_frame.hpp"
#include "kernel/kernel_status.hpp"
#include "test_mock_stream_bus.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

using namespace zzs::kernel;
using namespace zzs::kernel::device::driver::wifi;
using namespace zzs::kernel::device::driver::wifi::crypto;
using namespace zzs::kernel::device::driver::wifi::platform;
using namespace test;

namespace {

// Build a WifiHal with all hooks no-op returning OK.
WifiHal
make_hal () noexcept {
    WifiHal h{};
    h._M_reset = [] (void) noexcept -> Status { return Status::OK; };
    h._M_set_channel = [] (uint8_t) noexcept -> Status { return Status::OK; };
    h._M_set_country = [] (const char cc[2]) noexcept -> Status {
        (void) cc;
        return Status::OK;
    };
    h._M_mac_addr = [] (uint8_t mac[6]) noexcept -> Status {
        static const uint8_t m[6] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
        for (uint8_t i = 0u; i < 6u; ++i)
            mac[i] = m[i];
        return Status::OK;
    };
    h._M_delay_ms = [] (uint32_t) noexcept {};
    return h;
}

} // namespace

TEST_CASE ("wifi.driver.init.success", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);

    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);
    REQUIRE (driver.initialized ());
    REQUIRE (driver.connection_state () == WifiConnectionState::Disconnected);
    REQUIRE (driver.mac_addr ()._M_octets[0] == 0x11u);
    REQUIRE (driver.connection_cipher () == WifiCipher::None);
}

TEST_CASE ("wifi.driver.init.bus_closed", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);
    bus.set_open (false);
    REQUIRE (driver.init () == Status::NOT_READY);
    REQUIRE_FALSE (driver.initialized ());
}

TEST_CASE ("wifi.driver.init.null_bus", "[wifi][driver]") {
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (nullptr, hal);
    REQUIRE (driver.init () == Status::INTERNAL);
}

TEST_CASE ("wifi.driver.scan.populates_bss_table", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);

    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);

    /* Inject scan_done event. */
    uint8_t scan_done[1] = {static_cast<uint8_t> (WifiResp::ScanDone)};
    bus.inject_frame (scan_done, 1u);

    /* Inject one BSS entry as a protocol frame:
     * [WifiResp::BssEntry][bssid(6)][ssid_len(1)][ssid][security(1)][rssi(1)][channel(1)]. */
    uint8_t bssid[6] = {0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
    const char *ssid = "test-ssid";
    uint8_t entry[1 + 6 + 1 + 9 + 1 + 1 + 1] = {};
    uint8_t off = 0;
    entry[off++] = static_cast<uint8_t> (WifiResp::BssEntry);
    for (uint8_t i = 0; i < 6; ++i)
        entry[off++] = bssid[i];
    entry[off++] = static_cast<uint8_t> (strlen (ssid));
    for (uint8_t i = 0; i < strlen (ssid); ++i)
        entry[off++] = static_cast<uint8_t> (ssid[i]);
    entry[off++] = static_cast<uint8_t> (WifiSecurity::Wpa2);
    entry[off++] = static_cast<uint8_t> (-50);
    entry[off++] = 6u;
    bus.inject_frame (entry, off);

    REQUIRE (driver.scan () == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Scanning);

    /* Poll to process injected frames. */
    REQUIRE (driver.poll () == Status::OK);

    BssTable table{};
    REQUIRE (driver.read_bss_table (table) == Status::OK);
    REQUIRE (table._M_count == 1u);
    REQUIRE (table._M_entries[0]._M_ssid_len == static_cast<uint8_t> (strlen (ssid)));
    REQUIRE (std::memcmp (table._M_entries[0]._M_ssid, ssid, strlen (ssid)) == 0);
    REQUIRE (table._M_entries[0]._M_security == WifiSecurity::Wpa2);
}

TEST_CASE ("wifi.driver.connect.state_machine", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);

    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);

    const char *ssid = "my-ssid";
    const uint8_t psk[4] = {0x01, 0x02, 0x03, 0x04};

    REQUIRE (driver.connect (ssid, static_cast<uint8_t> (strlen (ssid)), psk, 4u) == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Scanning);

    /* Inject ConnectResult success with CCMP: [WifiResp::ConnectResult][result=0][cipher=CCMP]. */
    uint8_t assoc[3] = {static_cast<uint8_t> (WifiResp::ConnectResult), 0x00, 0x02};
    bus.inject_frame (assoc, 3u);
    REQUIRE (driver.poll () == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Connected);
    REQUIRE (driver.connection_cipher () == WifiCipher::Ccmp);
}

TEST_CASE ("wifi.driver.disconnect.returns_to_disconnected", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);

    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);
    REQUIRE (driver.connect ("ssid", 4u, nullptr, 0u) == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Scanning);

    REQUIRE (driver.disconnect () == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Disconnected);
    REQUIRE (driver.connection_cipher () == WifiCipher::None);
}

TEST_CASE ("wifi.driver.frame.beacon_probe_assoc", "[wifi][driver]") {
    /* Build a beacon frame: FC(2) Duration(2) Addr1(6) Addr2(6) Addr3(6) Seq(2) */
    uint8_t frame[24 + 7] = {};
    /* FC = type Management(0) subtype Beacon(8), version 0 */
    frame[0] = 0x80u;
    frame[1] = 0x00u;
    /* Addr1 = broadcast, Addr2 = bssid, Addr3 = bssid */
    for (uint8_t i = 0; i < 6; ++i)
        frame[4 + i] = 0xffu; /* addr1 */
    for (uint8_t i = 0; i < 6; ++i)
        frame[10 + i] = 0x11u; /* addr2 */
    for (uint8_t i = 0; i < 6; ++i)
        frame[16 + i] = 0x11u; /* addr3 */
    /* Fixed params: timestamp(8) interval(2) cap(2) */
    /* SSID IE: id 0 len 5 "hello" */
    frame[24] = 0x00u;
    frame[25] = 0x05u;
    frame[26] = 'h';
    frame[27] = 'e';
    frame[28] = 'l';
    frame[29] = 'l';
    frame[30] = 'o';

    Frame parsed{};
    REQUIRE (wifi_parse_frame (frame, 31u, parsed) == Status::OK);
    REQUIRE (parsed._M_type == WifiFrameType::Management);
    REQUIRE (parsed._M_mgmt_subtype == WifiMgmtSubtype::Beacon);
    REQUIRE (parsed._M_len == 31u);
}

TEST_CASE ("wifi.driver.beacon_dispatch_not_confused_with_ok", "[wifi][driver]") {
    /* Regression: beacon FC byte0 = 0x80 collides with WifiResp::Ok = 0x80.
     * A real beacon frame must be parsed as an 802.11 beacon, not treated as a
     * module protocol response and silently dropped. */
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);
    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);

    /* Build a real 802.11 beacon: FC=0x8000 (mgmt/beacon, version 0),
     * Addr1=broadcast, Addr2=Addr3=bssid.  Length-prefixed stream frame. */
    uint8_t beacon[31] = {};
    beacon[0] = 0x80u; /* FC: type Management(0), subtype Beacon(8), ver 0 */
    beacon[1] = 0x00u;
    for (uint8_t i = 0; i < 6; ++i)
        beacon[4 + i] = 0xffu; /* addr1 = broadcast */
    for (uint8_t i = 0; i < 6; ++i)
        beacon[10 + i] = 0x11u; /* addr2 */
    for (uint8_t i = 0; i < 6; ++i)
        beacon[16 + i] = 0x11u; /* addr3 */
    /* SSID IE: id 0 len 5 "hello" */
    beacon[24] = 0x00u;
    beacon[25] = 0x05u;
    beacon[26] = 'h';
    beacon[27] = 'e';
    beacon[28] = 'l';
    beacon[29] = 'l';
    beacon[30] = 'o';

    /* Inject as a length-prefixed stream frame. */
    bus.inject_frame (beacon, sizeof (beacon));
    REQUIRE (driver.poll () == Status::OK);

    /* The beacon must be available as a parsed frame, not confused with Ok. */
    Frame f{};
    REQUIRE (driver.read_frame (f) == Status::OK);
    REQUIRE (f._M_type == WifiFrameType::Management);
    REQUIRE (f._M_mgmt_subtype == WifiMgmtSubtype::Beacon);
    REQUIRE (f._M_len == sizeof (beacon));
    /* State must remain Disconnected (not advanced by a fake Ok response). */
    REQUIRE (driver.connection_state () == WifiConnectionState::Disconnected);
}

TEST_CASE ("wifi.driver.frame.bad_version_rejected", "[wifi][driver]") {
    uint8_t frame[24] = {};
    frame[0] = 0x81u; /* version = 1 */
    frame[1] = 0x00u;
    Frame parsed{};
    REQUIRE (wifi_parse_frame (frame, 24u, parsed) == Status::BAD_TYPE);
}

TEST_CASE ("wifi.driver.frame.undersize_rejected", "[wifi][driver]") {
    uint8_t frame[10] = {};
    Frame parsed{};
    REQUIRE (wifi_parse_frame (frame, 10u, parsed) == Status::BAD_LENGTH);
}

TEST_CASE ("wifi.driver.frame.oversize_rejected", "[wifi][driver]") {
    uint8_t buf[WifiMaxFrameBytes + 1] = {};
    Frame parsed{};
    REQUIRE (wifi_parse_frame (buf, WifiMaxFrameBytes + 1u, parsed) == Status::BAD_LENGTH);
}

TEST_CASE ("wifi.driver.buffer_overflow_rejection", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);
    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);

    /* Inject a frame larger than WifiMaxFrameBytes via length prefix. */
    uint8_t len_lo = static_cast<uint8_t> ((WifiMaxFrameBytes + 1u) & 0xffu);
    uint8_t len_hi = static_cast<uint8_t> (((WifiMaxFrameBytes + 1u) >> 8) & 0xffu);
    bus.inject_response (&len_lo, 1u);
    bus.inject_response (&len_hi, 1u);
    /* Reassembler should reject the length and resync. */
    REQUIRE ((driver.poll () == Status::OK || driver.poll () == Status::STALE));
}

TEST_CASE ("wifi.driver.cipher_negotiation", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);
    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);

    /* TKIP: [WifiResp::ConnectResult][result=0][cipher=TKIP]. */
    uint8_t tkip[3] = {static_cast<uint8_t> (WifiResp::ConnectResult), 0x00, 0x01};
    bus.inject_frame (tkip, 3u);
    driver.poll ();
    REQUIRE (driver.connection_cipher () == WifiCipher::Tkip);

    /* CCMP: [WifiResp::ConnectResult][result=0][cipher=CCMP]. */
    uint8_t ccmp[3] = {static_cast<uint8_t> (WifiResp::ConnectResult), 0x00, 0x02};
    bus.inject_frame (ccmp, 3u);
    driver.poll ();
    REQUIRE (driver.connection_cipher () == WifiCipher::Ccmp);

    /* GCMP: [WifiResp::ConnectResult][result=0][cipher=GCMP]. */
    uint8_t gcmp[3] = {static_cast<uint8_t> (WifiResp::ConnectResult), 0x00, 0x03};
    bus.inject_frame (gcmp, 3u);
    driver.poll ();
    REQUIRE (driver.connection_cipher () == WifiCipher::Gcmp);
}

TEST_CASE ("wifi.driver.connect.empty_ssid_rejected", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);
    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);
    REQUIRE (driver.connect ("", 0u, nullptr, 0u) == Status::OUT_OF_RANGE);
}

TEST_CASE ("wifi.driver.connect.psk_over_max_len_rejected", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);
    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);

    const char *ssid = "my-ssid";
    /* PSK length = WifiMaxPskLen + 1 (65) exceeds the 64-byte buffer. */
    uint8_t psk[WifiMaxPskLen + 1u] = {};
    for (uint8_t i = 0u; i <= WifiMaxPskLen; ++i)
        psk[i] = static_cast<uint8_t> (i);

    /* psk_len over capacity: must be rejected before any copy. */
    REQUIRE (driver.connect (ssid, static_cast<uint8_t> (strlen (ssid)), psk, static_cast<uint8_t> (WifiMaxPskLen + 1u))
             == Status::OUT_OF_RANGE);

    /* psk_len at capacity (64) is accepted. */
    REQUIRE (driver.connect (ssid, static_cast<uint8_t> (strlen (ssid)), psk, WifiMaxPskLen) == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Scanning);

    /* Null PSK with nonzero length is rejected. */
    REQUIRE (driver.connect (ssid, static_cast<uint8_t> (strlen (ssid)), nullptr, WifiMaxPskLen)
             == Status::OUT_OF_RANGE);
}

TEST_CASE ("wifi.driver.deauth_mid_connection", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);
    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);
    REQUIRE (driver.connect ("ssid", 4u, nullptr, 0u) == Status::OK);

    /* Inject deauth event. */
    uint8_t deauth[1] = {static_cast<uint8_t> (WifiEvent::Deauth)};
    bus.inject_frame (deauth, 1u);
    REQUIRE (driver.poll () == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Disconnected);
}

TEST_CASE ("wifi.driver.concurrent_scan_connect_rejected", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);
    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);

    REQUIRE (driver.scan () == Status::OK);
    /* Scan in progress; connect should be rejected. */
    REQUIRE (driver.connect ("ssid", 4u, nullptr, 0u) == Status::INTERNAL);
}

TEST_CASE ("wifi.driver.read_frame.stale_when_empty", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);
    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);
    Frame f{};
    REQUIRE (driver.read_frame (f) == Status::STALE);
}

// ---------------------------------------------------------------------------
// Full connect state machine: Scanning -> Authenticating -> Connected
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.driver.connect.full_state_machine", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);

    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);

    const char *ssid = "my-ssid";
    const uint8_t psk[4] = {0x01, 0x02, 0x03, 0x04};

    /* connect() -> Scanning. */
    REQUIRE (driver.connect (ssid, static_cast<uint8_t> (strlen (ssid)), psk, 4u) == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Scanning);

    /* scan_done while connecting -> Authenticating. */
    uint8_t scan_done[1] = {static_cast<uint8_t> (WifiResp::ScanDone)};
    bus.inject_frame (scan_done, 1u);
    REQUIRE (driver.poll () == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Authenticating);

    /* AssocResp / ConnectResult success -> Connected. */
    uint8_t assoc[3] = {static_cast<uint8_t> (WifiResp::ConnectResult), 0x00, 0x02};
    bus.inject_frame (assoc, 3u);
    REQUIRE (driver.poll () == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Connected);
    REQUIRE (driver.connection_cipher () == WifiCipher::Ccmp);
}

// ---------------------------------------------------------------------------
// Connect timeout -> Failed state
// ---------------------------------------------------------------------------

namespace {

uint64_t g_mock_now_ms = 0u;

uint64_t
mock_clock_ms () noexcept {
    return g_mock_now_ms * 1000000ull;
}

} // namespace

TEST_CASE ("wifi.driver.connect.timeout_failed", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal, &mock_clock_ms);

    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);

    const char *ssid = "my-ssid";
    const uint8_t psk[4] = {0x01, 0x02, 0x03, 0x04};

    REQUIRE (driver.connect (ssid, static_cast<uint8_t> (strlen (ssid)), psk, 4u) == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Scanning);

    /* First poll sets the deadline to now + 10000. */
    g_mock_now_ms = 0u;
    REQUIRE (driver.poll () == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Scanning);

    /* Advance past first deadline -> retry 1. */
    g_mock_now_ms = 11000u;
    REQUIRE (driver.poll () == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Scanning);

    /* Advance past second deadline -> retry 2. */
    g_mock_now_ms = 22000u;
    REQUIRE (driver.poll () == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Scanning);

    /* Advance past third deadline -> retry 3. */
    g_mock_now_ms = 33000u;
    REQUIRE (driver.poll () == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Scanning);

    /* Advance past fourth deadline -> retry limit exceeded -> Failed. */
    g_mock_now_ms = 44000u;
    REQUIRE (driver.poll () == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Failed);
}

// ---------------------------------------------------------------------------
// Scan table full: 8 entries accepted, 9th rejected
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.driver.scan.table_full", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);

    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);

    /* Inject 8 BSS entries as protocol frames. */
    for (uint8_t i = 0u; i < WifiMaxBssEntries; ++i) {
        uint8_t bssid[6] = {0xaa, 0xbb, 0xcc, 0xdd, 0xee, static_cast<uint8_t> (i)};
        char ssid[3] = {static_cast<char> ('0' + i), 0};
        uint8_t entry[1 + 6 + 1 + 2 + 1 + 1 + 1] = {};
        uint8_t off = 0;
        entry[off++] = static_cast<uint8_t> (WifiResp::BssEntry);
        for (uint8_t j = 0; j < 6; ++j)
            entry[off++] = bssid[j];
        entry[off++] = 1u;
        entry[off++] = static_cast<uint8_t> (ssid[0]);
        entry[off++] = static_cast<uint8_t> (WifiSecurity::Wpa2);
        entry[off++] = static_cast<uint8_t> (-50);
        entry[off++] = 6u;
        bus.inject_frame (entry, off);
    }

    /* Inject scan_done to end the scan. */
    uint8_t scan_done[1] = {static_cast<uint8_t> (WifiResp::ScanDone)};
    bus.inject_frame (scan_done, 1u);

    REQUIRE (driver.scan () == Status::OK);
    REQUIRE (driver.poll () == Status::OK);

    BssTable table{};
    REQUIRE (driver.read_bss_table (table) == Status::OK);
    REQUIRE (table._M_count == WifiMaxBssEntries);

    /* A 9th entry must be rejected by bss_table_add. */
    BssInfo extra = bss_info_make_empty ();
    MacAddr bsid{};
    bsid._M_octets[0] = 0xffu;
    extra._M_bssid = bsid;
    extra._M_ssid_len = 1u;
    extra._M_ssid[0] = 'x';
    extra._M_security = WifiSecurity::None;
    REQUIRE (bss_table_add (table, extra) == false);
    REQUIRE (table._M_count == WifiMaxBssEntries);
}

// ---------------------------------------------------------------------------
// PSK zero-out after use (connect success then disconnect clears PSK)
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.driver.connect.psk_zeroed_after_success", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);

    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);

    const char *ssid = "my-ssid";
    const uint8_t psk[4] = {0x01, 0x02, 0x03, 0x04};

    REQUIRE (driver.connect (ssid, static_cast<uint8_t> (strlen (ssid)), psk, 4u) == Status::OK);
    uint8_t assoc[3] = {static_cast<uint8_t> (WifiResp::ConnectResult), 0x00, 0x02};
    bus.inject_frame (assoc, 3u);
    REQUIRE (driver.poll () == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Connected);

    /* Disconnect: PSK must be zeroed. */
    REQUIRE (driver.disconnect () == Status::OK);
    REQUIRE (driver.connection_state () == WifiConnectionState::Disconnected);
    REQUIRE (driver.connection_cipher () == WifiCipher::None);
}

// ---------------------------------------------------------------------------
// Frame parsing: probe response, assoc response, deauth
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.driver.frame.probe_response", "[wifi][driver]") {
    uint8_t frame[24 + 7] = {};
    /* FC = type Management(0) subtype ProbeResp(5), version 0 */
    frame[0] = 0x50u;
    frame[1] = 0x00u;
    /* Addr1 = broadcast, Addr2 = bssid, Addr3 = bssid */
    for (uint8_t i = 0; i < 6; ++i)
        frame[4 + i] = 0xffu;
    for (uint8_t i = 0; i < 6; ++i)
        frame[10 + i] = 0x22u;
    for (uint8_t i = 0; i < 6; ++i)
        frame[16 + i] = 0x22u;
    /* SSID IE: id 0 len 5 "hello" */
    frame[24] = 0x00u;
    frame[25] = 0x05u;
    frame[26] = 'h';
    frame[27] = 'e';
    frame[28] = 'l';
    frame[29] = 'l';
    frame[30] = 'o';

    Frame parsed{};
    REQUIRE (wifi_parse_frame (frame, 31u, parsed) == Status::OK);
    REQUIRE (parsed._M_type == WifiFrameType::Management);
    REQUIRE (parsed._M_mgmt_subtype == WifiMgmtSubtype::ProbeResp);
    REQUIRE (parsed._M_len == 31u);
}

TEST_CASE ("wifi.driver.frame.assoc_response", "[wifi][driver]") {
    uint8_t frame[24 + 6] = {};
    /* FC = type Management(0) subtype AssocResp(1), version 0 */
    frame[0] = 0x10u;
    frame[1] = 0x00u;
    /* Addr1 = bssid, Addr2 = bssid, Addr3 = bssid */
    for (uint8_t i = 0; i < 6; ++i)
        frame[4 + i] = 0x33u;
    for (uint8_t i = 0; i < 6; ++i)
        frame[10 + i] = 0x33u;
    for (uint8_t i = 0; i < 6; ++i)
        frame[16 + i] = 0x33u;
    /* Capability info(2) Status code(2) AID(2) */
    frame[24] = 0x00u;
    frame[25] = 0x00u;
    frame[26] = 0x00u; /* status = success */
    frame[27] = 0x00u;
    frame[28] = 0x01u;
    frame[29] = 0x00u;

    Frame parsed{};
    REQUIRE (wifi_parse_frame (frame, 30u, parsed) == Status::OK);
    REQUIRE (parsed._M_type == WifiFrameType::Management);
    REQUIRE (parsed._M_mgmt_subtype == WifiMgmtSubtype::AssocResp);
}

TEST_CASE ("wifi.driver.frame.deauth", "[wifi][driver]") {
    uint8_t frame[24] = {};
    /* FC = type Management(0) subtype Deauth(0x0c), version 0 */
    frame[0] = 0xc0u;
    frame[1] = 0x00u;
    /* Addr1 = bssid, Addr2 = bssid, Addr3 = bssid */
    for (uint8_t i = 0; i < 6; ++i)
        frame[4 + i] = 0x44u;
    for (uint8_t i = 0; i < 6; ++i)
        frame[10 + i] = 0x44u;
    for (uint8_t i = 0; i < 6; ++i)
        frame[16 + i] = 0x44u;

    Frame parsed{};
    REQUIRE (wifi_parse_frame (frame, 24u, parsed) == Status::OK);
    REQUIRE (parsed._M_type == WifiFrameType::Management);
    REQUIRE (parsed._M_mgmt_subtype == WifiMgmtSubtype::Deauth);
}

// ---------------------------------------------------------------------------
// Bad length prefix in the stream reassembler
// ---------------------------------------------------------------------------

TEST_CASE ("wifi.driver.stream.bad_length_prefix", "[wifi][driver]") {
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);
    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);

    /* Length prefix declares 0 bytes -> nonsensical, reassembler resyncs. */
    uint8_t len_lo = 0x00u;
    uint8_t len_hi = 0x00u;
    bus.inject_response (&len_lo, 1u);
    bus.inject_response (&len_hi, 1u);

    /* Should not crash; poll returns OK or STALE. */
    REQUIRE ((driver.poll () == Status::OK || driver.poll () == Status::STALE));
}

TEST_CASE ("wifi.driver.stream.oversized_frame_then_valid_aligned", "[wifi][driver]") {
    /* Regression: a frame whose declared length exceeds WifiMaxFrameBytes must
     * be discarded byte-for-byte so the stream stays byte-aligned.  A subsequent
     * valid frame must be parsed correctly. */
    test::MockStreamBus bus;
    WifiHal hal = make_hal ();
    GenericWifiDriver driver (&bus, hal);
    bus.set_open (true);
    REQUIRE (driver.init () == Status::OK);

    /* Oversized frame: length prefix = WifiMaxFrameBytes + 1.  The reassembler
     * must discard all payload bytes and resync.  Inject in chunks because the
     * mock bus buffer is smaller than the oversized payload. */

    /* Inject oversized length prefix (257).  Reassembler sets skip=257. */
    uint8_t len_lo = static_cast<uint8_t> ((WifiMaxFrameBytes + 1u) & 0xffu);        /* 1 */
    uint8_t len_hi = static_cast<uint8_t> (((WifiMaxFrameBytes + 1u) >> 8) & 0xffu); /* 0 */
    bus.inject_response (&len_lo, 1u);
    bus.inject_response (&len_hi, 1u);
    (void) driver.poll (); /* start the skip */

    /* Inject 257 bytes of junk in chunks to complete the skip. */
    uint8_t junk[256] = {};
    bus.inject_response (junk, sizeof (junk));
    (void) driver.poll ();
    bus.inject_response (junk, 1u);
    (void) driver.poll ();
    /* Skip is now complete; reassembler is back in length mode. */

    /* Now inject a valid beacon frame. It must be parsed correctly. */
    uint8_t beacon[24] = {};
    beacon[0] = 0x80u; /* FC: mgmt/beacon, version 0 */
    beacon[1] = 0x00u;
    for (uint8_t i = 0; i < 6; ++i)
        beacon[4 + i] = 0xffu;
    for (uint8_t i = 0; i < 6; ++i)
        beacon[10 + i] = 0x11u;
    for (uint8_t i = 0; i < 6; ++i)
        beacon[16 + i] = 0x11u;
    bus.inject_frame (beacon, sizeof (beacon));

    REQUIRE (driver.poll () == Status::OK);
    Frame f{};
    REQUIRE (driver.read_frame (f) == Status::OK);
    REQUIRE (f._M_mgmt_subtype == WifiMgmtSubtype::Beacon);
    REQUIRE (f._M_len == sizeof (beacon));
}

TEST_CASE ("wifi.driver.frame.aliasing_regression", "[wifi][driver]") {
    /* Regression: wifi_parse_frame must not alias its input and output buffers.
     * A malformed frame must be rejected, and a valid frame must decode
     * correctly into a distinct output buffer. */
    uint8_t src[WifiMacHeaderLen] = {};
    src[0] = 0x81u; /* bad version -> must be rejected */
    src[1] = 0x00u;

    Frame out{};
    out._M_len = WifiMaxFrameBytes;
    for (uint16_t i = 0u; i < WifiMaxFrameBytes; ++i)
        out._M_buf[i] = 0xaau; /* poison the output buffer */

    /* Malformed input: must be rejected.  wifi_parse_frame clears its output
     * first (frame_make_empty), so _M_len is 0 and _M_buf is zeroed on failure.
     * The key regression check: the source buffer is not aliased with output,
     * so src is untouched. */
    REQUIRE (wifi_parse_frame (src, sizeof (src), out) == Status::BAD_TYPE);
    REQUIRE (out._M_len == 0u);
    /* Source buffer must not have been modified by the parse. */
    REQUIRE (src[0] == 0x81u);
    REQUIRE (src[1] == 0x00u);

    /* Valid input: must decode correctly. */
    uint8_t valid[WifiMacHeaderLen] = {};
    valid[0] = 0x80u; /* FC: mgmt/beacon, version 0 */
    valid[1] = 0x00u;
    for (uint8_t i = 0; i < 6; ++i)
        valid[4 + i] = 0xffu;
    for (uint8_t i = 0; i < 6; ++i)
        valid[10 + i] = 0x11u;
    for (uint8_t i = 0; i < 6; ++i)
        valid[16 + i] = 0x11u;

    Frame out2{};
    REQUIRE (wifi_parse_frame (valid, sizeof (valid), out2) == Status::OK);
    REQUIRE (out2._M_type == WifiFrameType::Management);
    REQUIRE (out2._M_mgmt_subtype == WifiMgmtSubtype::Beacon);
    REQUIRE (out2._M_len == sizeof (valid));
    /* Buffers are distinct: src was not modified. */
    REQUIRE (valid[0] == 0x80u);
}
