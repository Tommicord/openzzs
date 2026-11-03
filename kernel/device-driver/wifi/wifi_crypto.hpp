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

#include "kernel/device-driver/wifi/wifi_types.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief WiFi encryption / security model for the device-driver WiFi module.
 *
 * Models cipher negotiation (None/TKIP/CCMP/GCMP), RSN IE parsing from
 * beacon/probe-response information elements, a fixed-state 4-way handshake
 * state machine (M1..M4, nonce handling, PTK derivation interface), constant
 * time MAC comparison, and a replay-window protection model.  All state is
 * fixed-size per instance; no heap, no float, no exceptions, no RTTI.
 *
 * NOTE: This module does NOT implement real AES/TKIP/GCMP.  The key-derivation
 * steps are modelled as fixed-array operations and the handshake is a state
 * machine with key-confirmation.  A production port must plug a real crypto
 * primitive into `wifi_crypto_derive_ptk`.
 * @{
 */
namespace zzs::kernel::device::driver::wifi::crypto {

// ---------------------------------------------------------------------------
// Cipher / AKM identifiers (RSN information element cipher suite selectors)
// ---------------------------------------------------------------------------

/** @brief RSN IE cipher suite selector values (first 3 octets = OUI, 4th = type). */
enum class WifiRsnCipher : uint8_t {
  None = 0x00,    /**< No group cipher (used in RSN capability). */
  Wep40 = 0x01,   /**< WEP-40. */
  Tkip = 0x02,    /**< TKIP. */
  Wep104 = 0x03,  /**< WEP-104. */
  Ccmp = 0x04,    /**< CCMP / AES-128. */
  Wep128 = 0x05,  /**< WEP-128. */
  Gcmp = 0x06,    /**< GCMP / AES-256. */
  Gcmp256 = 0x07, /**< GCMP-256. */
};

enum class WifiRsnAkm : uint8_t {
  Psk = 0x02,       /**< PSK. */
  PskSha256 = 0x0b, /**< PSK-SHA256. */
  Sae = 0x08,       /**< SAE (WPA3). */
};

inline constexpr uint8_t WifiRsnOui[3] = {0x00, 0x0f, 0xac}; /* Microsoft OUI. */

// ---------------------------------------------------------------------------
// Fixed-size key / nonce buffers
// ---------------------------------------------------------------------------

/** @brief PMK length (octets) used by the model. */
inline constexpr uint8_t WifiPmkLen = 32u;
/** @brief PTK length (octets) used by the model. */
inline constexpr uint8_t WifiPtkLen = 32u;
/** @brief Nonce length (octets) - ANonce/SNonce. */
inline constexpr uint8_t WifiNonceLen = 32u;
/** @brief Replay counter length (octets). */
inline constexpr uint8_t WifiReplayCounterLen = 8u;
/** @brief Replay window size (number of tracked counters). */
inline constexpr uint8_t WifiReplayWindow = 8u;
/** @brief MIC length (octets) for the model's key confirmation. */
inline constexpr uint8_t WifiMicLen = 16u;

/** @brief Pairwise Master Key (fixed buffer). */
struct Pmk {
  uint8_t _M_octets[WifiPmkLen] = {};
  uint8_t _M_len = 0u;
  bool _M_valid = false;
};

/** @brief Pairwise Transient Key (fixed buffer). */
struct Ptk {
  uint8_t _M_octets[WifiPtkLen] = {};
  uint8_t _M_len = 0u;
  bool _M_valid = false;
};

/** @brief Nonce (ANonce or SNonce). */
struct Nonce {
  uint8_t _M_octets[WifiNonceLen] = {};
  uint8_t _M_len = 0u;
  bool _M_valid = false;
};

/** @brief Replay-counter tracker entry. */
struct ReplayCounterEntry {
  uint8_t _M_counter[WifiReplayCounterLen] = {};
  uint8_t _M_len = 0u;
  bool _M_occupied = false;
};

/** @brief Replay-window: fixed array of tracked replay counters. */
struct ReplayWindow {
  ReplayCounterEntry _M_entries[WifiReplayWindow] = {};
  uint8_t _M_count = 0u;
};

// ---------------------------------------------------------------------------
// RSN IE parsing
// ---------------------------------------------------------------------------

/** @brief Parse an RSN information element from a beacon/probe-response IE
 *         buffer.
 *
 * `ie` points at the IE element id byte; `ie_len` includes the id+len bytes.
 * On success fills `cipher` (the pairwise cipher selected) and `akm` (the AKM
 * suite selected).  Returns OK on success, BAD_LENGTH when the IE is truncated
 * or malformed, BAD_TYPE when the OUI is not the RSN Microsoft OUI, or
 * UNSUPPORTED when no recognised cipher/AKM is present.
 * @{
 */
[[nodiscard]] inline zzs::kernel::Status
wifi_rsn_ie_parse (const uint8_t *ie, uint16_t ie_len, WifiRsnCipher &cipher, WifiRsnAkm &akm) noexcept {
  cipher = WifiRsnCipher::None;
  akm = WifiRsnAkm::Psk;
  if (ie == nullptr || ie_len < 2u)
    return zzs::kernel::Status::BAD_LENGTH;

  /* Element ID = 0x30 (RSN). */
  if (ie[0] != 0x30u)
    return zzs::kernel::Status::BAD_TYPE;

  uint8_t len = ie[1];
  if (len < 2u || static_cast<uint16_t> (2u + len) > ie_len)
    return zzs::kernel::Status::BAD_LENGTH;

  const uint8_t *body = ie + 2u;
  uint8_t body_len = len;

  /* RSN Version (2 octets LE) - ignore but validate length. */
  if (body_len < 2u)
    return zzs::kernel::Status::BAD_LENGTH;
  body += 2u;
  body_len -= 2u;

  /* Group cipher suite (4 octets: 3 OUI + 1 type). */
  if (body_len < 4u)
    return zzs::kernel::Status::BAD_LENGTH;
  if (body[0] != WifiRsnOui[0] || body[1] != WifiRsnOui[1] || body[2] != WifiRsnOui[2])
    return zzs::kernel::Status::BAD_TYPE;
  WifiRsnCipher group_cipher = static_cast<WifiRsnCipher> (body[3]);
  (void) group_cipher;
  body += 4u;
  body_len -= 4u;

  /* Pairwise cipher suite count (2 octets LE). */
  if (body_len < 2u)
    return zzs::kernel::Status::BAD_LENGTH;
  uint16_t pairwise_count = static_cast<uint16_t> (body[0]) | (static_cast<uint16_t> (body[1]) << 8);
  body += 2u;
  body_len -= 2u;

  /* Select the first recognised pairwise cipher.  Advance the cursor in every
   * path (including on match) so the AKM count is read from the correct
   * offset, not from inside a pairwise suite. */
  bool found_pairwise = false;
  for (uint16_t i = 0u; i < pairwise_count; ++i) {
    if (body_len < 4u)
      return zzs::kernel::Status::BAD_LENGTH;
    if (!found_pairwise && body[0] == WifiRsnOui[0] && body[1] == WifiRsnOui[1] && body[2] == WifiRsnOui[2]) {
      cipher = static_cast<WifiRsnCipher> (body[3]);
      found_pairwise = true;
    }
    body += 4u;
    body_len -= 4u;
  }
  if (!found_pairwise)
    return zzs::kernel::Status::UNSUPPORTED;

  /* AKM suite count (2 octets LE). */
  if (body_len < 2u)
    return zzs::kernel::Status::BAD_LENGTH;
  uint16_t akm_count = static_cast<uint16_t> (body[0]) | (static_cast<uint16_t> (body[1]) << 8);
  body += 2u;
  body_len -= 2u;

  bool found_akm = false;
  for (uint16_t i = 0u; i < akm_count; ++i) {
    if (body_len < 4u)
      return zzs::kernel::Status::BAD_LENGTH;
    if (body[0] == WifiRsnOui[0] && body[1] == WifiRsnOui[1] && body[2] == WifiRsnOui[2]) {
      akm = static_cast<WifiRsnAkm> (body[3]);
      found_akm = true;
      break;
    }
    body += 4u;
    body_len -= 4u;
  }
  if (!found_akm)
    return zzs::kernel::Status::UNSUPPORTED;

  return zzs::kernel::Status::OK;
}

/** @} */

// ---------------------------------------------------------------------------
// Constant-time MAC comparison (fixed-time loop, no early exit)
// ---------------------------------------------------------------------------

/** @brief Constant-time equality of two MAC addresses (6 octets).
 *
 * Every octet is examined regardless of earlier mismatches so the total
 * execution time is independent of the input values.  Returns true when the
 * addresses are equal, false otherwise.
 * @{
 */
[[nodiscard]] inline bool
wifi_crypto_memcmp_eq (const uint8_t *a, const uint8_t *b, uint16_t len) noexcept {
  if (a == nullptr || b == nullptr)
    return false;
  uint8_t diff = 0u;
  for (uint16_t i = 0u; i < len; ++i)
    diff |= static_cast<uint8_t> (a[i] ^ b[i]);
  return diff == 0u;
}

/** @} */

// ---------------------------------------------------------------------------
// 4-way handshake state machine
// ---------------------------------------------------------------------------

/** @brief 4-way handshake message number. */
enum class WifiHandshakeMsg : uint8_t {
  None = 0u,
  M1 = 1u, /**< Authenticator -> Supplicant: ANonce + replay counter. */
  M2 = 2u, /**< Supplicant -> Authenticator: SNonce + MIC + replay counter. */
  M3 = 3u, /**< Authenticator -> Supplicant: GTK + MIC + replay counter. */
  M4 = 4u, /**< Supplicant -> Authenticator: ACK + MIC + replay counter. */
};

/** @brief 4-way handshake state. */
enum class WifiHandshakeState : uint8_t {
  Disconnected = 0u, /**< No handshake in progress. */
  Started = 1u,      /**< M1 sent / received, awaiting peer reply. */
  M1Received = 2u,   /**< Supplicant received M1, deriving PTK. */
  M2Received = 3u,   /**< Authenticator received M2, confirming keys. */
  M3Received = 4u,   /**< Supplicant received M3, installing PTK. */
  M4Received = 5u,   /**< Authenticator received M4, handshake complete. */
  Completed = 6u,    /**< Keys installed, connection encrypted. */
  Failed = 7u,       /**< Handshake failed (MIC mismatch, timeout, replay). */
};

/** @brief Fixed-state 4-way handshake model.
 *
 * Holds the PMK, PTK, nonces, replay window, and current message state.  All
 * buffers are fixed arrays; `wifi_crypto_derive_ptk` is the pluggable key
 * derivation interface.  No heap, no float, no exceptions.
 * @{
 */
struct Handshake {
  Pmk _M_pmk{};                    /**< Pairwise master key (from PSK). */
  Ptk _M_ptk{};                    /**< Derived pairwise transient key. */
  Nonce _M_anonce{};               /**< Authenticator nonce. */
  Nonce _M_snonce{};               /**< Supplicant nonce. */
  uint8_t _M_mic[WifiMicLen] = {}; /**< Last computed MIC. */
  uint8_t _M_mic_len = 0u;         /**< MIC octet count. */
  ReplayWindow _M_replay{};        /**< Replay-window tracker. */
  WifiHandshakeMsg _M_last_msg = WifiHandshakeMsg::None;
  WifiHandshakeState _M_state = WifiHandshakeState::Disconnected;
  uint8_t _M_replay_counter[WifiReplayCounterLen] = {};
  uint8_t _M_replay_counter_len = 0u;
};

/** @brief Key-derivation interface: derive a PTK from PMK, nonces and
 *         addresses.  Provided by the platform crypto port; this default is a
 *         fixed-array modelling stub that mixes inputs into a 32-byte PTK.
 * @{
 */
[[nodiscard]] inline zzs::kernel::Status
wifi_crypto_derive_ptk (const Pmk &pmk, const Nonce &anonce, const Nonce &snonce, const MacAddr &auth_addr,
                        const MacAddr &sup_addr, Ptk &ptk) noexcept {
  (void) auth_addr;
  (void) sup_addr;
  if (!pmk._M_valid || !anonce._M_valid || !snonce._M_valid)
    return zzs::kernel::Status::INTERNAL;
  if (pmk._M_len == 0u || anonce._M_len == 0u || snonce._M_len == 0u)
    return zzs::kernel::Status::INTERNAL;

  ptk = Ptk{};
  ptk._M_len = WifiPtkLen;
  /* Fixed-array modelling: mix PMK + nonces byte-wise into the PTK.  A real
   * port replaces this with PBKDF2/PMK + PRF-512 over the 4-way handshake
   * inputs. */
  for (uint8_t i = 0u; i < WifiPtkLen; ++i) {
    uint8_t v = pmk._M_octets[i % pmk._M_len];
    v ^= anonce._M_octets[i % anonce._M_len];
    v ^= snonce._M_octets[i % snonce._M_len];
    v ^= static_cast<uint8_t> (i);
    ptk._M_octets[i] = v;
  }
  ptk._M_valid = true;
  return zzs::kernel::Status::OK;
}

/** @} */

/** @brief Compute a MIC over `data` using `ptk` (modelling stub).
 *
 * Fixed-array MIC computation: XOR-fold the PTK over the data.  A real port
 * replaces this with HMAC-SHA1 (TKIP/CCMP) or AES-GMAC (GCMP).
 * @{
 */
[[nodiscard]] inline zzs::kernel::Status
wifi_crypto_mic_compute (const Ptk &ptk, const uint8_t *data, uint16_t data_len, uint8_t *mic,
                         uint8_t &mic_len) noexcept {
  if (!ptk._M_valid || data == nullptr || mic == nullptr)
    return zzs::kernel::Status::INTERNAL;
  if (ptk._M_len == 0u)
    return zzs::kernel::Status::INTERNAL;
  mic_len = WifiMicLen;
  for (uint8_t i = 0u; i < WifiMicLen; ++i)
    mic[i] = 0u;
  for (uint16_t i = 0u; i < data_len; ++i)
    mic[i % WifiMicLen] ^= data[i] ^ ptk._M_octets[i % ptk._M_len];
  return zzs::kernel::Status::OK;
}

/** @} */

/** @brief Verify a MIC in constant time. */
[[nodiscard]] inline bool
wifi_crypto_mic_eq (const uint8_t *a, const uint8_t *b, uint8_t len) noexcept {
  return wifi_crypto_memcmp_eq (a, b, static_cast<uint16_t> (len));
}

/** @} */

/** @brief Replay-counter protection: record a counter and reject replays.
 *
 * The window holds up to `WifiReplayWindow` counters.  Returns OK when the
 * counter is new (higher than any seen) and records it; returns
 * `SEQ_ERROR`-equivalent (`BAD_SEQUENCE`) when it is a replay (equal or lower
 * than a recorded counter within the window).
 * @{
 */
[[nodiscard]] inline zzs::kernel::Status
wifi_replay_check (ReplayWindow &window, const uint8_t *counter, uint8_t counter_len) noexcept {
  if (counter == nullptr || counter_len == 0u || counter_len > WifiReplayCounterLen)
    return zzs::kernel::Status::OUT_OF_RANGE;

  /* Reject if equal to or lower than any recorded counter (replay). */
  for (uint8_t i = 0u; i < window._M_count; ++i) {
    const ReplayCounterEntry &e = window._M_entries[i];
    if (!e._M_occupied || e._M_len != counter_len)
      continue;
    int cmp = 0;
    for (uint8_t j = 0u; j < counter_len; ++j) {
      if (counter[j] < e._M_counter[j]) {
        cmp = -1;
        break;
      }
      if (counter[j] > e._M_counter[j]) {
        cmp = 1;
        break;
      }
    }
    if (cmp <= 0)
      return zzs::kernel::Status::BAD_SEQUENCE;
  }

  /* Record: insert in order, drop oldest when full. */
  if (window._M_count < WifiReplayWindow) {
    window._M_entries[window._M_count]._M_len = counter_len;
    window._M_entries[window._M_count]._M_occupied = true;
    for (uint8_t i = 0u; i < counter_len; ++i)
      window._M_entries[window._M_count]._M_counter[i] = counter[i];
    window._M_count += 1u;
  } else {
    /* Drop oldest (index 0), shift, append at end. */
    for (uint8_t i = 1u; i < WifiReplayWindow; ++i)
      window._M_entries[i - 1u] = window._M_entries[i];
    ReplayCounterEntry &last = window._M_entries[WifiReplayWindow - 1u];
    last._M_len = counter_len;
    last._M_occupied = true;
    for (uint8_t i = 0u; i < counter_len; ++i)
      last._M_counter[i] = counter[i];
  }
  return zzs::kernel::Status::OK;
}

/** @} */

/** @brief Reset the handshake to its initial state, zeroing key material. */
[[nodiscard]] inline Handshake
wifi_handshake_make_empty () noexcept {
  Handshake h{};
  h._M_state = WifiHandshakeState::Disconnected;
  h._M_last_msg = WifiHandshakeMsg::None;
  /* Zero out key material. */
  for (uint8_t i = 0u; i < WifiPmkLen; ++i)
    h._M_pmk._M_octets[i] = 0u;
  h._M_pmk._M_len = 0u;
  h._M_pmk._M_valid = false;
  for (uint8_t i = 0u; i < WifiPtkLen; ++i)
    h._M_ptk._M_octets[i] = 0u;
  h._M_ptk._M_len = 0u;
  h._M_ptk._M_valid = false;
  for (uint8_t i = 0u; i < WifiNonceLen; ++i)
    h._M_anonce._M_octets[i] = 0u;
  h._M_anonce._M_len = 0u;
  h._M_anonce._M_valid = false;
  for (uint8_t i = 0u; i < WifiNonceLen; ++i)
    h._M_snonce._M_octets[i] = 0u;
  h._M_snonce._M_len = 0u;
  h._M_snonce._M_valid = false;
  for (uint8_t i = 0u; i < WifiMicLen; ++i)
    h._M_mic[i] = 0u;
  h._M_mic_len = 0u;
  return h;
}

/** @} */

} // namespace zzs::kernel::device::driver::wifi::crypto
