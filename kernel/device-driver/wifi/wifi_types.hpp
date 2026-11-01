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

#include "kernel/kernel_status.hpp"

#include <cstdint>
#include <cstring>

/**
 * @brief WiFi (802.11) antenna driver types.
 *
 * Integer-only, fixed-size, no-heap types describing MAC addresses, 802.11
 * frames, security modes and signal information.  No floating point anywhere.
 * @{
 */
namespace zzs::kernel::device::driver::wifi {

/** @brief Length of an 802.11 MAC address (6 octets). */
inline constexpr uint8_t WifiMacAddrLen = 6u;

/** @brief Broadcast MAC address (FF:FF:FF:FF:FF:FF). */
inline constexpr uint8_t WifiMacBroadcast[WifiMacAddrLen] = {
  0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

/** @brief Maximum SSID length (802.11). */
inline constexpr uint8_t WifiMaxSsidLen = 32u;

/** @brief Maximum BSSID table entries the driver caches. */
inline constexpr uint8_t WifiMaxBssEntries = 8u;

/** @brief Maximum scan result entries. */
inline constexpr uint8_t WifiMaxScanResults = 16u;

// ---------------------------------------------------------------------------
// MAC address
// ---------------------------------------------------------------------------

/** @brief An 802.11 MAC address (6 octets, fixed array, comparable). */
struct MacAddr {
  uint8_t _M_octets[WifiMacAddrLen] = {0};
};

/** @brief Compare two MAC addresses. */
[[nodiscard]] inline constexpr bool
mac_addr_equal (const MacAddr &a, const MacAddr &b) noexcept {
  return std::memcmp (a._M_octets, b._M_octets, WifiMacAddrLen) == 0;
}

/** @brief Copy a MAC address. */
[[nodiscard]] inline MacAddr
mac_addr_copy (const MacAddr &src) noexcept {
  MacAddr dst{};
  std::memcpy (dst._M_octets, src._M_octets, WifiMacAddrLen);
  return dst;
}

/** @brief True when `a` is the broadcast address. */
[[nodiscard]] inline constexpr bool
mac_addr_is_broadcast (const MacAddr &a) noexcept {
  for (uint8_t i = 0u; i < WifiMacAddrLen; ++i) {
    if (a._M_octets[i] != WifiMacBroadcast[i])
      return false;
  }
  return true;
}

/** @brief True when `a` is all zeros (unspecified). */
[[nodiscard]] inline constexpr bool
mac_addr_is_unspecified (const MacAddr &a) noexcept {
  for (uint8_t i = 0u; i < WifiMacAddrLen; ++i) {
    if (a._M_octets[i] != 0x00u)
      return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Security / auth
// ---------------------------------------------------------------------------

/** @brief WiFi security / authentication mode. */
enum class WifiSecurity : uint8_t {
  None = 0,    /**< Open network, no authentication. */
  Wep = 1,     /**< WEP (deprecated). */
  Wpa = 2,     /**< WPA / TKIP. */
  Wpa2 = 3,    /**< WPA2 / CCMP (AES). */
  Wpa3 = 4,    /**< WPA3 / SAE. */
  Unknown = 0xff,
};

/** @brief Return a short name for a security mode. */
[[nodiscard]] inline const char *
wifi_security_name (WifiSecurity s) noexcept {
  switch (s) {
  case WifiSecurity::None:
    return "none";
  case WifiSecurity::Wep:
    return "wep";
  case WifiSecurity::Wpa:
    return "wpa";
  case WifiSecurity::Wpa2:
    return "wpa2";
  case WifiSecurity::Wpa3:
    return "wpa3";
  default:
    return "unknown";
  }
}

// ---------------------------------------------------------------------------
// 802.11 frame types
// ---------------------------------------------------------------------------

/** @brief 802.11 frame type (type sub-field of the Frame Control field). */
enum class WifiFrameType : uint8_t {
  Management = 0x00,
  Control = 0x01,
  Data = 0x02,
  Reserved = 0x03,
};

/** @brief 802.11 management frame subtype. */
enum class WifiMgmtSubtype : uint8_t {
  AssocReq = 0x00,
  AssocResp = 0x01,
  ReassocReq = 0x02,
  ReassocResp = 0x03,
  ProbeReq = 0x04,
  ProbeResp = 0x05,
  Beacon = 0x08,
  Atim = 0x09,
  Disassoc = 0x0a,
  Auth = 0x0b,
  Deauth = 0x0c,
  Action = 0x0d,
};

/** @brief 802.11 control frame subtype. */
enum class WifiCtrlSubtype : uint8_t {
  BlockAckReq = 0x08,
  BlockAck = 0x09,
  PsPoll = 0x0a,
  Rts = 0x0b,
  Cts = 0x0c,
  Ack = 0x0d,
  CfEnd = 0x0e,
  CfEndCfAck = 0x0f,
};

/** @brief Return a short name for a frame type. */
[[nodiscard]] inline const char *
wifi_frame_type_name (WifiFrameType t) noexcept {
  switch (t) {
  case WifiFrameType::Management:
    return "management";
  case WifiFrameType::Control:
    return "control";
  case WifiFrameType::Data:
    return "data";
  default:
    return "reserved";
  }
}

// ---------------------------------------------------------------------------
// Signal / BSS info
// ---------------------------------------------------------------------------

/** @brief Signal strength information for one BSS. */
struct BssSignal {
  int8_t _M_rssi_dbm = 0;       /**< Received signal strength indicator, dBm. */
  int8_t _M_noise_dbm = 0;      /**< Noise floor, dBm. */
  uint8_t _M_channel = 0;       /**< Operating channel (1..165). */
  uint8_t _M_antenna = 0;       /**< Antenna index used for the measurement. */
};

/** @brief One scanned BSS (access point / station). */
struct BssInfo {
  MacAddr _M_bssid{};           /**< BSSID (AP MAC). */
  MacAddr _M_source{};          /**< Source address (TA). */
  char _M_ssid[WifiMaxSsidLen + 1] = {0};
  uint8_t _M_ssid_len = 0u;
  WifiSecurity _M_security = WifiSecurity::Unknown;
  BssSignal _M_signal{};        /**< Last measured signal. */
  uint64_t _M_timestamp_ns = 0ull;
  bool _M_beacon_seen = false;  /**< True when a beacon from this BSS was received. */
};

/** @brief Clear a BSS info to empty. */
[[nodiscard]] inline BssInfo
bss_info_make_empty () noexcept {
  BssInfo b{};
  b._M_security = WifiSecurity::Unknown;
  return b;
}

/** @brief True when the BSS carries a non-empty SSID. */
[[nodiscard]] inline constexpr bool
bss_info_has_ssid (const BssInfo &b) noexcept {
  return b._M_ssid_len > 0u && b._M_ssid_len <= WifiMaxSsidLen;
}

/** @brief Table of scanned BSSes (fixed array, no heap). */
struct BssTable {
  BssInfo _M_entries[WifiMaxBssEntries] = {};
  uint8_t _M_count = 0u;
};

/** @brief Add or update a BSS entry (replaces on matching BSSID). */
[[nodiscard]] inline bool
bss_table_add (BssTable &table, const BssInfo &bss) noexcept {
  if (table._M_count >= WifiMaxBssEntries)
    return false;
  for (uint8_t i = 0u; i < table._M_count; ++i) {
    if (mac_addr_equal (table._M_entries[i]._M_bssid, bss._M_bssid)) {
      table._M_entries[i] = bss;
      return true;
    }
  }
  table._M_entries[table._M_count++] = bss;
  return true;
}

/** @brief Find a BSS entry by BSSID. Returns index or WifiMaxBssEntries. */
[[nodiscard]] inline uint8_t
bss_table_find (const BssTable &table, const MacAddr &bssid) noexcept {
  for (uint8_t i = 0u; i < table._M_count; ++i) {
    if (mac_addr_equal (table._M_entries[i]._M_bssid, bssid))
      return i;
  }
  return WifiMaxBssEntries;
}

// ---------------------------------------------------------------------------
// Scan result
// ---------------------------------------------------------------------------

/** @brief One scan result (summarized BSS). */
struct ScanResult {
  MacAddr _M_bssid{};
  char _M_ssid[WifiMaxSsidLen + 1] = {0};
  uint8_t _M_ssid_len = 0u;
  WifiSecurity _M_security = WifiSecurity::Unknown;
  int8_t _M_rssi_dbm = 0;
  uint8_t _M_channel = 0;
};

/** @brief WiFi connection state. */
enum class WifiConnectionState : uint8_t {
  Disconnected = 0,
  Scanning = 1,
  Authenticating = 2,
  Associating = 3,
  Connected = 4,
  ConnectedIp = 5,   /**< Connected and has an IP (DHCP). */
  Disconnecting = 6,
  Failed = 7,
};

/** @brief Return a short name for a connection state. */
[[nodiscard]] inline const char *
wifi_connection_state_name (WifiConnectionState s) noexcept {
  switch (s) {
  case WifiConnectionState::Disconnected:
    return "disconnected";
  case WifiConnectionState::Scanning:
    return "scanning";
  case WifiConnectionState::Authenticating:
    return "authenticating";
  case WifiConnectionState::Associating:
    return "associating";
  case WifiConnectionState::Connected:
    return "connected";
  case WifiConnectionState::ConnectedIp:
    return "connected_ip";
  case WifiConnectionState::Disconnecting:
    return "disconnecting";
  case WifiConnectionState::Failed:
    return "failed";
  default:
    return "unknown";
  }
}

/** @} */
} // namespace zzs::kernel::device::driver::wifi
