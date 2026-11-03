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
 * @file test_wifi_crypto.cpp
 * @brief Unit tests for the WiFi crypto module: RSN IE parsing, constant-time
 *        MAC comparison, replay-window protection, and 4-way handshake model.
 */

#include "kernel/device-driver/wifi/wifi_crypto.hpp"
#include "kernel/kernel_status.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstring>

using namespace zzs::kernel;
using namespace zzs::kernel::device::driver::wifi;
using namespace zzs::kernel::device::driver::wifi::crypto;

TEST_CASE ("wifi.crypto.memcmp_eq.equal", "[wifi][crypto]") {
    uint8_t a[6] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    uint8_t b[6] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    REQUIRE (wifi_crypto_memcmp_eq (a, b, 6u) == true);
}

TEST_CASE ("wifi.crypto.memcmp_eq.not_equal", "[wifi][crypto]") {
    uint8_t a[6] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    uint8_t b[6] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x67};
    REQUIRE (wifi_crypto_memcmp_eq (a, b, 6u) == false);
}

TEST_CASE ("wifi.crypto.memcmp_eq.null", "[wifi][crypto]") {
    uint8_t a[6] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66};
    REQUIRE (wifi_crypto_memcmp_eq (nullptr, a, 6u) == false);
    REQUIRE (wifi_crypto_memcmp_eq (a, nullptr, 6u) == false);
    REQUIRE (wifi_crypto_memcmp_eq (nullptr, nullptr, 6u) == false);
}

TEST_CASE ("wifi.crypto.memcmp_eq.fixed_time_no_early_exit", "[wifi][crypto]") {
    /* All 256 possible first-byte differences must still compare all bytes. */
    uint8_t base[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    for (uint8_t d = 0u; d < 255u; ++d) {
        uint8_t a[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        uint8_t b[8] = {static_cast<uint8_t> (base[0] ^ d), 1, 2, 3, 4, 5, 6, 7};
        bool eq = wifi_crypto_memcmp_eq (a, b, 8u);
        REQUIRE ((eq == (d == 0u)));
    }
}

TEST_CASE ("wifi.crypto.rsn_ie.parse_wpa2_ccmp_psk", "[wifi][crypto]") {
    /* RSN IE: id(1)=0x30 len(1)=18, version(2)=1, group cipher(4)=CCMP,
     * pairwise count(2)=1, pairwise(4)=CCMP, AKM count(2)=1, AKM(4)=PSK. */
    uint8_t ie[] = {
        0x30, 18,               /* id=RSN, len=18 */
        0x01, 0x00,             /* version 1 */
        0x00, 0x0f, 0xac, 0x04, /* group cipher = CCMP */
        0x01, 0x00,             /* pairwise count = 1 */
        0x00, 0x0f, 0xac, 0x04, /* pairwise = CCMP */
        0x01, 0x00,             /* AKM count = 1 */
        0x00, 0x0f, 0xac, 0x02, /* AKM = PSK */
    };
    WifiRsnCipher cipher = WifiRsnCipher::None;
    WifiRsnAkm akm = WifiRsnAkm::Psk;
    REQUIRE (wifi_rsn_ie_parse (ie, sizeof (ie), cipher, akm) == Status::OK);
    REQUIRE (cipher == WifiRsnCipher::Ccmp);
    REQUIRE (akm == WifiRsnAkm::Psk);
}

TEST_CASE ("wifi.crypto.rsn_ie.bad_oui", "[wifi][crypto]") {
    uint8_t ie[] = {
        0x30, 16,   0x01, 0x00, 0x11, 0x22, 0x33, 0x04, /* bad OUI */
        0x01, 0x00, 0x00, 0x0f, 0xac, 0x04, 0x01, 0x00, 0x00, 0x0f, 0xac, 0x02,
    };
    WifiRsnCipher cipher = WifiRsnCipher::None;
    WifiRsnAkm akm = WifiRsnAkm::Psk;
    REQUIRE (wifi_rsn_ie_parse (ie, sizeof (ie), cipher, akm) == Status::BAD_TYPE);
}

TEST_CASE ("wifi.crypto.rsn_ie.truncated", "[wifi][crypto]") {
    uint8_t ie[] = {0x30, 0x02, 0x01, 0x00}; /* only version, truncated */
    WifiRsnCipher cipher = WifiRsnCipher::None;
    WifiRsnAkm akm = WifiRsnAkm::Psk;
    REQUIRE (wifi_rsn_ie_parse (ie, sizeof (ie), cipher, akm) == Status::BAD_LENGTH);
}

TEST_CASE ("wifi.crypto.replay.new_counter_accepted", "[wifi][crypto]") {
    ReplayWindow window{};
    uint8_t ctr[8] = {0, 0, 0, 0, 0, 0, 0, 1};
    REQUIRE (wifi_replay_check (window, ctr, 8u) == Status::OK);
    REQUIRE (window._M_count == 1u);
}

TEST_CASE ("wifi.crypto.replay.replay_rejected", "[wifi][crypto]") {
    ReplayWindow window{};
    uint8_t ctr1[8] = {0, 0, 0, 0, 0, 0, 0, 5};
    uint8_t ctr2[8] = {0, 0, 0, 0, 0, 0, 0, 5}; /* same as ctr1 -> replay */
    REQUIRE (wifi_replay_check (window, ctr1, 8u) == Status::OK);
    REQUIRE (wifi_replay_check (window, ctr2, 8u) == Status::BAD_SEQUENCE);
}

TEST_CASE ("wifi.crypto.replay.lower_counter_rejected", "[wifi][crypto]") {
    ReplayWindow window{};
    uint8_t hi[8] = {0, 0, 0, 0, 0, 0, 0, 10};
    uint8_t lo[8] = {0, 0, 0, 0, 0, 0, 0, 3};
    REQUIRE (wifi_replay_check (window, hi, 8u) == Status::OK);
    REQUIRE (wifi_replay_check (window, lo, 8u) == Status::BAD_SEQUENCE);
}

TEST_CASE ("wifi.crypto.replay.window_wraps_drop_oldest", "[wifi][crypto]") {
    ReplayWindow window{};
    /* Fill the window beyond WifiReplayWindow (8). */
    for (uint8_t i = 1u; i <= 12u; ++i) {
        uint8_t ctr[8] = {0, 0, 0, 0, 0, 0, 0, i};
        REQUIRE (wifi_replay_check (window, ctr, 8u) == Status::OK);
    }
    /* Window now holds counters 5..12. Counter 1 was dropped but is lower than
     * every entry still in the window, so re-submitting it is a replay. */
    uint8_t ctr1[8] = {0, 0, 0, 0, 0, 0, 0, 1};
    REQUIRE (wifi_replay_check (window, ctr1, 8u) == Status::BAD_SEQUENCE);
    /* A new higher counter is accepted and drops the oldest (5). */
    uint8_t ctr13[8] = {0, 0, 0, 0, 0, 0, 0, 13};
    REQUIRE (wifi_replay_check (window, ctr13, 8u) == Status::OK);
}

TEST_CASE ("wifi.crypto.hanldshake.make_empty_zeros_key_material", "[wifi][crypto]") {
    Handshake h = wifi_handshake_make_empty ();
    REQUIRE (h._M_state == WifiHandshakeState::Disconnected);
    REQUIRE (h._M_pmk._M_valid == false);
    REQUIRE (h._M_ptk._M_valid == false);
    REQUIRE (h._M_anonce._M_valid == false);
    REQUIRE (h._M_snonce._M_valid == false);
}

TEST_CASE ("wifi.crypto.derive_ptk.valid", "[wifi][crypto]") {
    Pmk pmk{};
    pmk._M_len = 32u;
    for (uint8_t i = 0; i < 32; ++i)
        pmk._M_octets[i] = static_cast<uint8_t> (i);
    pmk._M_valid = true;

    Nonce anonce{};
    anonce._M_len = 32u;
    for (uint8_t i = 0; i < 32; ++i)
        anonce._M_octets[i] = static_cast<uint8_t> (i + 1);
    anonce._M_valid = true;

    Nonce snonce{};
    snonce._M_len = 32u;
    for (uint8_t i = 0; i < 32; ++i)
        snonce._M_octets[i] = static_cast<uint8_t> (i + 2);
    snonce._M_valid = true;

    MacAddr aa{};
    for (uint8_t i = 0; i < 6; ++i)
        aa._M_octets[i] = static_cast<uint8_t> (0x10 + i);
    MacAddr sa{};
    for (uint8_t i = 0; i < 6; ++i)
        sa._M_octets[i] = static_cast<uint8_t> (0x20 + i);

    Ptk ptk{};
    REQUIRE (wifi_crypto_derive_ptk (pmk, anonce, snonce, aa, sa, ptk) == Status::OK);
    REQUIRE (ptk._M_valid == true);
    REQUIRE (ptk._M_len == 32u);
}

TEST_CASE ("wifi.crypto.derive_ptk.invalid_pmk", "[wifi][crypto]") {
    Pmk pmk{}; /* not valid */
    Nonce anonce{};
    anonce._M_len = 32u;
    anonce._M_valid = true;
    Nonce snonce{};
    snonce._M_len = 32u;
    snonce._M_valid = true;
    MacAddr aa{}, sa{};
    Ptk ptk{};
    REQUIRE (wifi_crypto_derive_ptk (pmk, anonce, snonce, aa, sa, ptk) != Status::OK);
}

TEST_CASE ("wifi.crypto.derive_ptk.zero_length_key_material", "[wifi][crypto]") {
    /* Regression: % pmk._M_len / % ptk._M_len divide-by-zero when _M_len == 0.
     * Every modulus/division is guarded; a defined error is returned. */
    Pmk pmk{};
    pmk._M_len = 0u; /* zero length -> div-by-zero guard */
    pmk._M_valid = true;
    Nonce anonce{};
    anonce._M_len = 32u;
    anonce._M_valid = true;
    Nonce snonce{};
    snonce._M_len = 32u;
    snonce._M_valid = true;
    MacAddr aa{}, sa{};
    Ptk ptk{};
    REQUIRE (wifi_crypto_derive_ptk (pmk, anonce, snonce, aa, sa, ptk) == Status::INTERNAL);

    /* Zero-length ANonce. */
    Pmk pmk2{};
    pmk2._M_len = 32u;
    pmk2._M_valid = true;
    Nonce anonce2{};
    anonce2._M_len = 0u;
    anonce2._M_valid = true;
    Nonce snonce2{};
    snonce2._M_len = 32u;
    snonce2._M_valid = true;
    Ptk ptk2{};
    REQUIRE (wifi_crypto_derive_ptk (pmk2, anonce2, snonce2, aa, sa, ptk2) == Status::INTERNAL);

    /* Zero-length SNonce. */
    Nonce anonce3{};
    anonce3._M_len = 32u;
    anonce3._M_valid = true;
    Nonce snonce3{};
    snonce3._M_len = 0u;
    snonce3._M_valid = true;
    Ptk ptk3{};
    REQUIRE (wifi_crypto_derive_ptk (pmk2, anonce3, snonce3, aa, sa, ptk3) == Status::INTERNAL);
}

TEST_CASE ("wifi.crypto.mic_compute.zero_ptk_len", "[wifi][crypto]") {
    /* Regression: mic_compute guards against ptk._M_len == 0. */
    Ptk ptk{};
    ptk._M_len = 0u;
    ptk._M_valid = true;
    uint8_t data[4] = {1, 2, 3, 4};
    uint8_t mic[16] = {};
    uint8_t mic_len = 0u;
    REQUIRE (wifi_crypto_mic_compute (ptk, data, sizeof (data), mic, mic_len) == Status::INTERNAL);
}

TEST_CASE ("wifi.crypto.rsn_ie.akm_count_after_pairwise_match", "[wifi][crypto]") {
    /* Regression: pairwise match branch must advance the cursor (body += 4)
     * in every path so the AKM count is read from the correct offset, not from
     * inside a pairwise suite.  Two pairwise suites: first is TKIP (matched),
     * second is CCMP.  AKM count follows after both pairwise suites. */
    uint8_t ie[] = {
        0x30, 22,               /* id=RSN, len=22 */
        0x01, 0x00,             /* version 1 */
        0x00, 0x0f, 0xac, 0x04, /* group cipher = CCMP */
        0x02, 0x00,             /* pairwise count = 2 */
        0x00, 0x0f, 0xac, 0x02, /* pairwise[0] = TKIP (matched) */
        0x00, 0x0f, 0xac, 0x04, /* pairwise[1] = CCMP (skipped after match) */
        0x01, 0x00,             /* AKM count = 1 */
        0x00, 0x0f, 0xac, 0x02, /* AKM = PSK */
    };
    WifiRsnCipher cipher = WifiRsnCipher::None;
    WifiRsnAkm akm = WifiRsnAkm::Psk;
    REQUIRE (wifi_rsn_ie_parse (ie, sizeof (ie), cipher, akm) == Status::OK);
    REQUIRE (cipher == WifiRsnCipher::Tkip);
    REQUIRE (akm == WifiRsnAkm::Psk);
}

TEST_CASE ("wifi.crypto.rsn_ie.akm_count_skips_unmatched_pairwise", "[wifi][crypto]") {
    /* Regression: when the first pairwise suite is unmatched, the cursor must
     * still advance past it to read the second pairwise suite and then the AKM
     * count correctly. */
    uint8_t ie[] = {
        0x30, 22,               /* id=RSN, len=22 */
        0x01, 0x00,             /* version 1 */
        0x00, 0x0f, 0xac, 0x04, /* group cipher = CCMP */
        0x02, 0x00,             /* pairwise count = 2 */
        0x11, 0x22, 0x33, 0x05, /* pairwise[0] = non-RSN OUI (skipped) */
        0x00, 0x0f, 0xac, 0x02, /* pairwise[1] = TKIP (matched) */
        0x01, 0x00,             /* AKM count = 1 */
        0x00, 0x0f, 0xac, 0x02, /* AKM = PSK */
    };
    WifiRsnCipher cipher = WifiRsnCipher::None;
    WifiRsnAkm akm = WifiRsnAkm::Psk;
    REQUIRE (wifi_rsn_ie_parse (ie, sizeof (ie), cipher, akm) == Status::OK);
    REQUIRE (cipher == WifiRsnCipher::Tkip);
    REQUIRE (akm == WifiRsnAkm::Psk);
}
