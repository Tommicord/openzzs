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

#include "kernel/device-driver/wifi/generic_wifi_driver.hpp"

#include "kernel/kernel_status.hpp"

#include <cstdint>
#include <cstring>

namespace zzs::kernel::device::driver::wifi {

GenericWifiDriver::GenericWifiDriver (zzs::kernel::device::driver::StreamBusInterface *bus,
                                      const platform::WifiHal &hal, uint64_t (*clock) (void)) noexcept
    : WifiDriver (bus, clock), _M_stream (bus, clock), _M_hal (&hal) {
    _M_stream.set_frame_callback (&GenericWifiDriver::_M_frame_cb, this);
}

[[nodiscard]] zzs::kernel::Status
GenericWifiDriver::init () noexcept {
    if (_M_bus == nullptr)
        return zzs::kernel::Status::INTERNAL;
    if (!_M_bus->is_open ())
        return zzs::kernel::Status::NOT_READY;

    /* Radio reset via the platform HAL. */
    if (_M_hal != nullptr && _M_hal->_M_reset != nullptr) {
        zzs::kernel::Status st = _M_hal->_M_reset ();
        if (!zzs::kernel::status_ok (st))
            return st;
    }
    if (_M_hal != nullptr && _M_hal->_M_delay_ms != nullptr)
        _M_hal->_M_delay_ms (10u);

    /* Set regulatory country (default "US"). */
    if (_M_hal != nullptr && _M_hal->_M_set_country != nullptr) {
        char cc[2] = {'U', 'S'};
        zzs::kernel::Status st = _M_hal->_M_set_country (cc);
        if (!zzs::kernel::status_ok (st))
            return st;
    }

    /* Set operating channel (default 6). */
    if (_M_hal != nullptr && _M_hal->_M_set_channel != nullptr) {
        zzs::kernel::Status st = _M_hal->_M_set_channel (6u);
        if (!zzs::kernel::status_ok (st))
            return st;
    }

    /* Read station MAC address from the HAL. */
    if (_M_hal != nullptr && _M_hal->_M_mac_addr != nullptr) {
        uint8_t mac[WifiMacAddrLen] = {};
        zzs::kernel::Status st = _M_hal->_M_mac_addr (mac);
        if (!zzs::kernel::status_ok (st))
            return st;
        for (uint8_t i = 0u; i < WifiMacAddrLen; ++i)
            _M_mac_addr._M_octets[i] = mac[i];
    } else {
        /* Fallback: fixed local MAC (no static storage - stack array only). */
        const uint8_t fallback_mac[WifiMacAddrLen] = {0x02, 0x00, 0x00, 0x12, 0x34, 0x56};
        for (uint8_t i = 0u; i < WifiMacAddrLen; ++i)
            _M_mac_addr._M_octets[i] = fallback_mac[i];
    }

    /* Send reset command to the module. */
    uint8_t payload[1] = {0x00};
    zzs::kernel::Status st = _M_send_cmd (WifiCmd::Reset, payload, 1u);
    if (!zzs::kernel::status_ok (st))
        return st;

    _M_state = WifiConnectionState::Disconnected;
    _M_cipher = WifiCipher::None;
    _M_initialized = true;
    _M_clear_errors ();
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
GenericWifiDriver::deinit () noexcept {
    /* Send disconnect command, ignoring failures. */
    if (_M_bus != nullptr && _M_bus->is_open ()) {
        (void) _M_send_cmd (WifiCmd::Disconnect, nullptr, 0u);
    }

    /* Tear down the handshake, zeroing key material. */
    _M_handshake = crypto::wifi_handshake_make_empty ();

    _M_state = WifiConnectionState::Disconnected;
    _M_cipher = WifiCipher::None;
    _M_scan_in_progress = false;
    _M_initialized = false;
    _M_connect_ssid_len = 0u;
    _M_connect_psk_len = 0u;
    for (uint8_t i = 0u; i < WifiMaxPskLen; ++i)
        _M_connect_psk[i] = 0u;
    for (uint8_t i = 0u; i < WifiMaxSsidLen + 1; ++i)
        _M_connect_ssid[i] = 0;

    if (_M_bus != nullptr)
        (void) _M_bus->flush_rx ();
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
GenericWifiDriver::scan () noexcept {
    if (!_M_initialized)
        return zzs::kernel::Status::NOT_READY;
    if (_M_scan_in_progress)
        return zzs::kernel::Status::INTERNAL;

    /* Reset the BSS table before a new scan. */
    _M_bss_table = BssTable{};
    _M_scan_in_progress = true;

    zzs::kernel::Status st = _M_send_cmd (WifiCmd::Scan, nullptr, 0u);
    if (!zzs::kernel::status_ok (st)) {
        _M_scan_in_progress = false;
        return st;
    }
    _M_state = WifiConnectionState::Scanning;
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
GenericWifiDriver::poll () noexcept {
    if (!_M_initialized)
        return zzs::kernel::Status::NOT_READY;

    /* Drain the stream and parse any frames/events. */
    zzs::kernel::Status st = _M_stream.poll ();
    if (!zzs::kernel::status_ok (st) && st != zzs::kernel::Status::STALE)
        return st;

    /* Drive the connect state machine. */
    _M_state_machine ();

    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
GenericWifiDriver::connect (const char *ssid, uint8_t ssid_len, const uint8_t *psk, uint8_t psk_len) noexcept {
    if (!_M_initialized)
        return zzs::kernel::Status::NOT_READY;
    if (ssid == nullptr || ssid_len == 0u || ssid_len > WifiMaxSsidLen)
        return zzs::kernel::Status::OUT_OF_RANGE;
    if (psk_len > WifiMaxPskLen)
        return zzs::kernel::Status::OUT_OF_RANGE;
    if (psk_len > 0u && psk == nullptr)
        return zzs::kernel::Status::OUT_OF_RANGE;
    if (_M_state == WifiConnectionState::Scanning || _M_state == WifiConnectionState::Authenticating
        || _M_state == WifiConnectionState::Associating)
        return zzs::kernel::Status::INTERNAL; /* concurrent scan+connect rejected */

    /* Store connection parameters (zero-out any previous PSK first). */
    for (uint8_t i = 0u; i < WifiMaxPskLen; ++i)
        _M_connect_psk[i] = 0u;
    _M_connect_ssid_len = ssid_len;
    for (uint8_t i = 0u; i < ssid_len; ++i)
        _M_connect_ssid[i] = ssid[i];
    _M_connect_ssid[ssid_len] = 0;
    _M_connect_psk_len = psk_len;
    for (uint8_t i = 0u; i < psk_len; ++i)
        _M_connect_psk[i] = psk[i];

    /* Build connect command payload: [ssid_len][ssid...][psk_len][psk...]. */
    uint8_t payload[1 + WifiMaxSsidLen + 1 + WifiMaxPskLen] = {};
    uint8_t off = 0u;
    payload[off++] = ssid_len;
    for (uint8_t i = 0u; i < ssid_len; ++i)
        payload[off++] = static_cast<uint8_t> (ssid[i]);
    payload[off++] = psk_len;
    for (uint8_t i = 0u; i < psk_len; ++i)
        payload[off++] = psk[i];

    zzs::kernel::Status st = _M_send_cmd (WifiCmd::Connect, payload, off);
    if (!zzs::kernel::status_ok (st))
        return st;

    _M_state = WifiConnectionState::Scanning;
    _M_connect_retries = 0u;
    _M_connect_deadline_ms = 0u; /* poll() will set the deadline */
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
GenericWifiDriver::disconnect () noexcept {
    if (!_M_initialized)
        return zzs::kernel::Status::NOT_READY;

    zzs::kernel::Status st = _M_send_cmd (WifiCmd::Disconnect, nullptr, 0u);
    if (!zzs::kernel::status_ok (st))
        return st;

    _M_handshake = crypto::wifi_handshake_make_empty ();
    _M_state = WifiConnectionState::Disconnected;
    _M_cipher = WifiCipher::None;
    _M_scan_in_progress = false;
    _M_connect_retries = 0u;
    _M_connect_ssid_len = 0u;
    _M_connect_psk_len = 0u;
    for (uint8_t i = 0u; i < WifiMaxPskLen; ++i)
        _M_connect_psk[i] = 0u;
    for (uint8_t i = 0u; i < WifiMaxSsidLen + 1; ++i)
        _M_connect_ssid[i] = 0;
    _M_clear_errors ();
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
GenericWifiDriver::read_frame (Frame &out) noexcept {
    out = frame_make_empty ();
    if (_M_frame._M_len == 0u)
        return zzs::kernel::Status::STALE;
    out = _M_frame;
    _M_frame = frame_make_empty ();
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
GenericWifiDriver::read_bss_table (BssTable &out) noexcept {
    out = _M_bss_table;
    return zzs::kernel::Status::OK;
}

// ---------------------------------------------------------------------------
// Command protocol
// ---------------------------------------------------------------------------

[[nodiscard]] zzs::kernel::Status
GenericWifiDriver::_M_send_cmd (WifiCmd cmd, const uint8_t *payload, uint8_t payload_len) noexcept {
    if (_M_bus == nullptr)
        return zzs::kernel::Status::INTERNAL;
    if (!_M_bus->is_open ())
        return zzs::kernel::Status::NOT_READY;
    if (payload == nullptr && payload_len > 0u)
        return zzs::kernel::Status::OUT_OF_RANGE;

    /* Frame: [len_lo][len_hi][cmd][payload...].  Total length = 2 + 1 + payload. */
    uint16_t frame_len = static_cast<uint16_t> (3u + payload_len);
    if (frame_len > WifiMaxFrameBytes)
        return zzs::kernel::Status::BAD_LENGTH;

    uint8_t frame[WifiMaxFrameBytes] = {};
    uint16_t off = 0u;
    frame[off++] = static_cast<uint8_t> (frame_len & 0xffu);
    frame[off++] = static_cast<uint8_t> ((frame_len >> 8) & 0xffu);
    frame[off++] = static_cast<uint8_t> (cmd);
    for (uint8_t i = 0u; i < payload_len; ++i)
        frame[off++] = payload[i];

    zzs::kernel::Status st = _M_bus->write_bytes (frame, frame_len);
    /* Zero out the local frame buffer (command payload may contain PSK). */
    for (uint16_t i = 0u; i < frame_len; ++i)
        frame[i] = 0u;
    return st;
}

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------

void
GenericWifiDriver::_M_state_machine () noexcept {
    /* Set the entry deadline when first entering a connect sub-state. */
    if (_M_connect_deadline_ms == 0u
        && (_M_state == WifiConnectionState::Scanning || _M_state == WifiConnectionState::Authenticating
            || _M_state == WifiConnectionState::Associating)) {
        uint64_t now = timestamp_ns ();
        _M_connect_deadline_ms = static_cast<uint32_t> (now / 1000000ull) + WifiConnectTimeoutMs;
    }

    switch (_M_state) {
    case WifiConnectionState::Scanning:
        /* Wait for scan_done event from _M_handle_scan_done, which advances to
         * Authenticating (if connecting) or Disconnected (if standalone scan). */
        break;
    case WifiConnectionState::Authenticating:
        /* Auth handshake completed by the module; the module issues AssocResp to
         * advance to Associating. */
        break;
    case WifiConnectionState::Associating:
        /* Wait for connect result (AssocResp) -> Connected or Failed. */
        break;
    case WifiConnectionState::Connected:
        /* Wait for DHCP/IP -> ConnectedIp.  The module indicates IP acquisition
         * via an event; here we simply hold Connected until the application
         * drives further. */
        break;
    case WifiConnectionState::ConnectedIp:
        break;
    case WifiConnectionState::Disconnected:
    case WifiConnectionState::Disconnecting:
    case WifiConnectionState::Failed:
        _M_connect_deadline_ms = 0u;
        break;
    default:
        break;
    }

    /* Connect timeout handling with retry limit. */
    if (_M_state == WifiConnectionState::Scanning || _M_state == WifiConnectionState::Authenticating
        || _M_state == WifiConnectionState::Associating) {
        if (_M_connect_deadline_ms != 0u) {
            uint64_t now = timestamp_ns ();
            uint32_t now_ms = static_cast<uint32_t> (now / 1000000ull);
            if (now_ms >= _M_connect_deadline_ms) {
                if (_M_connect_retries < WifiMaxConnectRetries) {
                    _M_connect_retries += 1u;
                    _M_connect_deadline_ms = now_ms + WifiConnectTimeoutMs;
                    /* Re-issue connect command. */
                    uint8_t payload[1 + WifiMaxSsidLen + 1 + WifiMaxPskLen] = {};
                    uint8_t off = 0u;
                    payload[off++] = _M_connect_ssid_len;
                    for (uint8_t i = 0u; i < _M_connect_ssid_len; ++i)
                        payload[off++] = static_cast<uint8_t> (_M_connect_ssid[i]);
                    payload[off++] = _M_connect_psk_len;
                    for (uint8_t i = 0u; i < _M_connect_psk_len; ++i)
                        payload[off++] = _M_connect_psk[i];
                    (void) _M_send_cmd (WifiCmd::Connect, payload, off);
                    for (uint16_t i = 0u; i < off; ++i)
                        payload[i] = 0u;
                } else {
                    _M_state = WifiConnectionState::Failed;
                    _M_connect_deadline_ms = 0u;
                    _M_record_error ();
                    _M_scan_in_progress = false;
                    /* Zero out PSK on failure. */
                    _M_connect_psk_len = 0u;
                    for (uint8_t i = 0u; i < WifiMaxPskLen; ++i)
                        _M_connect_psk[i] = 0u;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Event / response handlers
// ---------------------------------------------------------------------------

[[nodiscard]] zzs::kernel::Status
GenericWifiDriver::_M_handle_scan_done (const uint8_t *payload, uint8_t len) noexcept {
    (void) payload;
    (void) len;
    _M_scan_in_progress = false;
    if (_M_state == WifiConnectionState::Scanning) {
        /* Scan complete: if we were connecting, pick the best BSS and proceed. */
        if (_M_connect_ssid_len > 0u) {
            _M_state = WifiConnectionState::Authenticating;
            _M_connect_deadline_ms = 0u; /* reset deadline for next sub-state */
        } else {
            _M_state = WifiConnectionState::Disconnected;
        }
    }
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
GenericWifiDriver::_M_handle_bss_entry (const uint8_t *payload, uint8_t len) noexcept {
    if (payload == nullptr || len < 1u)
        return zzs::kernel::Status::OUT_OF_RANGE;

    BssInfo bss = bss_info_make_empty ();

    /* [bssid(6)][ssid_len(1)][ssid...][security(1)][rssi(1)][channel(1)] */
    uint8_t off = 0u;
    for (uint8_t i = 0u; i < WifiMacAddrLen && off + WifiMacAddrLen <= len; ++i)
        bss._M_bssid._M_octets[i] = payload[off + i];
    off += WifiMacAddrLen;

    if (off >= len)
        return zzs::kernel::Status::BAD_LENGTH;
    uint8_t ssid_len = payload[off++];
    if (ssid_len > WifiMaxSsidLen)
        ssid_len = WifiMaxSsidLen;
    if (off + ssid_len > len)
        return zzs::kernel::Status::BAD_LENGTH;
    for (uint8_t i = 0u; i < ssid_len; ++i)
        bss._M_ssid[i] = static_cast<char> (payload[off + i]);
    bss._M_ssid[ssid_len] = 0;
    bss._M_ssid_len = ssid_len;
    off += ssid_len;

    if (off >= len)
        return zzs::kernel::Status::BAD_LENGTH;
    uint8_t security = payload[off++];
    switch (security) {
    case 0:
        bss._M_security = WifiSecurity::None;
        break;
    case 1:
        bss._M_security = WifiSecurity::Wep;
        break;
    case 2:
        bss._M_security = WifiSecurity::Wpa;
        break;
    case 3:
        bss._M_security = WifiSecurity::Wpa2;
        break;
    case 4:
        bss._M_security = WifiSecurity::Wpa3;
        break;
    default:
        bss._M_security = WifiSecurity::Unknown;
        break;
    }

    if (off < len) {
        bss._M_signal._M_rssi_dbm = static_cast<int8_t> (payload[off++]);
    }
    if (off < len) {
        bss._M_signal._M_channel = payload[off++];
    }

    bss._M_timestamp_ns = timestamp_ns ();
    bss._M_beacon_seen = true;

    (void) bss_table_add (_M_bss_table, bss);
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
GenericWifiDriver::_M_handle_connect_result (const uint8_t *payload, uint8_t len) noexcept {
    if (payload == nullptr || len < 1u)
        return zzs::kernel::Status::OUT_OF_RANGE;

    uint8_t result = payload[0];
    uint8_t cipher_enum = (len > 1) ? payload[1] : 0u;

    if (result == 0u) {
        /* Success. */
        switch (cipher_enum) {
        case 0:
            _M_cipher = WifiCipher::None;
            break;
        case 1:
            _M_cipher = WifiCipher::Tkip;
            break;
        case 2:
            _M_cipher = WifiCipher::Ccmp;
            break;
        case 3:
            _M_cipher = WifiCipher::Gcmp;
            break;
        default:
            _M_cipher = WifiCipher::None;
            break;
        }
        _M_state = WifiConnectionState::Connected;
        _M_connect_deadline_ms = 0u;
        _M_clear_errors ();
        _M_scan_in_progress = false;
        /* Connection established: zero out the PSK material we no longer need. */
        _M_connect_psk_len = 0u;
        for (uint8_t i = 0u; i < WifiMaxPskLen; ++i)
            _M_connect_psk[i] = 0u;
    } else {
        /* Failure. */
        _M_state = WifiConnectionState::Failed;
        _M_connect_deadline_ms = 0u;
        _M_record_error ();
        _M_scan_in_progress = false;
        _M_connect_psk_len = 0u;
        for (uint8_t i = 0u; i < WifiMaxPskLen; ++i)
            _M_connect_psk[i] = 0u;
    }
    return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
GenericWifiDriver::_M_handle_event_frame (const Frame &frame) noexcept {
    if (frame._M_len < 1u)
        return zzs::kernel::Status::BAD_LENGTH;

    WifiEvent event = static_cast<WifiEvent> (frame._M_buf[0]);
    const uint8_t *payload = frame._M_buf + 1;
    uint8_t payload_len = static_cast<uint8_t> (frame._M_len - 1u);

    switch (event) {
    case WifiEvent::Beacon:
    case WifiEvent::ProbeResp:
        /* Beacons and probe responses carry BSS entries. */
        return _M_handle_bss_entry (payload, payload_len);
    case WifiEvent::AssocResp:
        return _M_handle_connect_result (payload, payload_len);
    case WifiEvent::Deauth:
        _M_state = WifiConnectionState::Disconnected;
        _M_cipher = WifiCipher::None;
        _M_handshake = crypto::wifi_handshake_make_empty ();
        _M_clear_errors ();
        _M_scan_in_progress = false;
        return zzs::kernel::Status::OK;
    default:
        break;
    }
    return zzs::kernel::Status::OK;
}

// ---------------------------------------------------------------------------
// Stream frame callback
// ---------------------------------------------------------------------------

void
GenericWifiDriver::_M_frame_cb (const Frame &frame, void *context) noexcept {
    GenericWifiDriver *self = static_cast<GenericWifiDriver *> (context);
    if (self == nullptr)
        return;

    if (frame._M_len == 0u) {
        self->_M_record_error ();
        return;
    }

    /* Distinguish a real 802.11 MAC frame from a module protocol/response/event
     * frame.  A real 802.11 frame has a full MAC header and a Frame Control
     * version field of 0.  Module protocol frames use raw WifiResp/WifiEvent
     * code bytes as their first octet (e.g. WifiResp::Ok = 0x80), which collides
     * with the 802.11 beacon FC byte (0x80).  Requiring both a full header and a
     * zero FC version keeps real beacons from being mistaken for protocol
     * responses and silently dropped. */
    bool is_80211 = false;
    if (frame._M_len >= WifiMacHeaderLen) {
        uint16_t fc = static_cast<uint16_t> (frame._M_buf[0]) | (static_cast<uint16_t> (frame._M_buf[1]) << 8);
        if ((fc & WifiFcVersionMask) == 0u)
            is_80211 = true;
    }

    if (is_80211) {
        /* Real 802.11 frame: parse and cache it. */
        Frame parsed = frame_make_empty ();
        zzs::kernel::Status pst = wifi_parse_frame (frame._M_buf, frame._M_len, parsed);
        if (zzs::kernel::status_ok (pst)) {
            self->_M_frame = parsed;
            self->_M_clear_errors ();
        } else {
            self->_M_record_error ();
        }
        return;
    }

    /* Module protocol frame: first byte is a WifiResp/WifiEvent code. */
    uint8_t first = frame._M_buf[0];
    bool is_protocol_frame =
        (first == static_cast<uint8_t> (WifiResp::Ok) || first == static_cast<uint8_t> (WifiResp::Fail)
         || first == static_cast<uint8_t> (WifiResp::ScanDone) || first == static_cast<uint8_t> (WifiResp::BssEntry)
         || first == static_cast<uint8_t> (WifiResp::ConnectResult) || first == static_cast<uint8_t> (WifiEvent::Beacon)
         || first == static_cast<uint8_t> (WifiEvent::ProbeResp) || first == static_cast<uint8_t> (WifiEvent::AssocResp)
         || first == static_cast<uint8_t> (WifiEvent::Deauth));

    if (is_protocol_frame) {
        switch (first) {
        case static_cast<uint8_t> (WifiResp::ScanDone):
            (void) self->_M_handle_scan_done (frame._M_buf + 1, static_cast<uint8_t> (frame._M_len - 1u));
            break;
        case static_cast<uint8_t> (WifiResp::BssEntry):
            (void) self->_M_handle_bss_entry (frame._M_buf + 1, static_cast<uint8_t> (frame._M_len - 1u));
            break;
        case static_cast<uint8_t> (WifiResp::ConnectResult):
            (void) self->_M_handle_connect_result (frame._M_buf + 1, static_cast<uint8_t> (frame._M_len - 1u));
            break;
        case static_cast<uint8_t> (WifiResp::Ok):
        case static_cast<uint8_t> (WifiResp::Fail):
            break;
        default:
            (void) self->_M_handle_event_frame (frame);
            break;
        }
    } else {
        /* Unknown frame: neither valid 802.11 nor recognised protocol code. */
        self->_M_record_error ();
    }
}

/** @} */

} // namespace zzs::kernel::device::driver::wifi
