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

#include "kernel/device-driver/stream_bus.hpp"
#include "kernel/device-driver/wifi/platforms/platform_hal.hpp"
#include "kernel/device-driver/wifi/stream/wifi_stream.hpp"
#include "kernel/device-driver/wifi/wifi_crypto.hpp"
#include "kernel/device-driver/wifi/wifi_driver.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief Generic UART/SPI WiFi module driver (concrete WifiDriver).
 *
 * Implements the full station lifecycle over a length-prefixed command/event
 * protocol streamed through a `StreamBusInterface`:
 *   init -> reset radio, set country/channel, enable interface, read MAC
 *   scan   -> issue scan command, collect up to WifiMaxBssEntries BSSes
 *   connect -> Disconnected -> Scanning -> Authenticating -> Associating ->
 *              Connected -> ConnectedIp, with timeouts and retry limits
 *   disconnect -> deauth/disassoc, return to Disconnected
 *   poll -> drive state machine, drain stream, parse frames (beacons, probe
 *           responses, assoc responses, deauths)
 *
 * All state is fixed per-instance; no heap, no float, no exceptions.  The
 * radio-control hooks come from a `WifiHal` supplied by the platform HAL.
 * @{
 */
namespace zzs::kernel::device::driver::wifi {

// ---------------------------------------------------------------------------
// Command / event protocol codes (length-prefixed stream frames)
// ---------------------------------------------------------------------------

/** @brief Command codes sent to the WiFi module. */
enum class WifiCmd : uint8_t {
  Reset = 0x01,
  SetChannel = 0x02,
  SetCountry = 0x03,
  Scan = 0x04,
  Connect = 0x05,
  Disconnect = 0x06,
};

/** @brief Response codes returned by the WiFi module. */
enum class WifiResp : uint8_t {
  Ok = 0x80,
  Fail = 0x81,
  ScanDone = 0x82,
  BssEntry = 0x83,
  ConnectResult = 0x84,
};

/** @brief Async event codes from the WiFi module. */
enum class WifiEvent : uint8_t {
  Beacon = 0xA0,
  ProbeResp = 0xA1,
  AssocResp = 0xA2,
  Deauth = 0xA3,
};

/** @brief Maximum PSK octets the driver buffers. */
inline constexpr uint8_t WifiMaxPskLen = 64u;

/** @brief Connect timeout in milliseconds. */
inline constexpr uint32_t WifiConnectTimeoutMs = 10000u;

/** @brief Maximum connect retries. */
inline constexpr uint8_t WifiMaxConnectRetries = 3u;

// ---------------------------------------------------------------------------
// Concrete driver
// ---------------------------------------------------------------------------

/** @brief Generic WiFi station driver. */
class GenericWifiDriver : public WifiDriver {
public:
  /** @brief Construct with a stream bus, platform HAL radio hooks, and clock.
   *         The bus and HAL are non-owning. */
  explicit GenericWifiDriver (zzs::kernel::device::driver::StreamBusInterface *bus, const platform::WifiHal &hal,
                              uint64_t (*clock) (void) = nullptr) noexcept;

  GenericWifiDriver (const GenericWifiDriver &) = delete;
  GenericWifiDriver &
  operator= (const GenericWifiDriver &) = delete;
  GenericWifiDriver (GenericWifiDriver &&) = delete;
  GenericWifiDriver &
  operator= (GenericWifiDriver &&) = delete;
  ~GenericWifiDriver () override = default;

  /** @brief Initialize the WiFi radio: reset, set country/channel, enable
   *         interface, read MAC address. */
  [[nodiscard]] zzs::kernel::Status
  init () noexcept override;

  /** @brief Deinitialize / power down the radio. */
  [[nodiscard]] zzs::kernel::Status
  deinit () noexcept override;

  /** @brief Start a scan for BSSes. */
  [[nodiscard]] zzs::kernel::Status
  scan () noexcept override;

  /** @brief Poll the radio: drive the connect state machine, drain the
   *         stream, parse incoming frames/events. */
  [[nodiscard]] zzs::kernel::Status
  poll () noexcept override;

  /** @brief Connect to an SSID. */
  [[nodiscard]] zzs::kernel::Status
  connect (const char *ssid, uint8_t ssid_len, const uint8_t *psk, uint8_t psk_len) noexcept override;

  /** @brief Disconnect from the current AP. */
  [[nodiscard]] zzs::kernel::Status
  disconnect () noexcept override;

  /** @brief Read the latest received frame into `out`. */
  [[nodiscard]] zzs::kernel::Status
  read_frame (Frame &out) noexcept override;

  /** @brief Return the scanned BSS table. */
  [[nodiscard]] zzs::kernel::Status
  read_bss_table (BssTable &out) noexcept override;

  /** @brief Return the current connection state. */
  [[nodiscard]] WifiConnectionState
  connection_state () const noexcept override {
    return _M_state;
  }

  /** @brief Return the negotiated cipher for the active connection. */
  [[nodiscard]] WifiCipher
  connection_cipher () const noexcept override {
    return _M_cipher;
  }

  /** @brief Return the station MAC address. */
  [[nodiscard]] MacAddr
  mac_addr () const noexcept override {
    return _M_mac_addr;
  }

  /** @brief True when the driver has been initialized. */
  [[nodiscard]] bool
  initialized () const noexcept {
    return _M_initialized;
  }

private:
  // -- command protocol --
  [[nodiscard]] zzs::kernel::Status
  _M_send_cmd (WifiCmd cmd, const uint8_t *payload, uint8_t payload_len) noexcept;

  // -- state machine --
  void
  _M_state_machine () noexcept;

  [[nodiscard]] zzs::kernel::Status
  _M_handle_scan_done (const uint8_t *payload, uint8_t len) noexcept;

  [[nodiscard]] zzs::kernel::Status
  _M_handle_bss_entry (const uint8_t *payload, uint8_t len) noexcept;

  [[nodiscard]] zzs::kernel::Status
  _M_handle_connect_result (const uint8_t *payload, uint8_t len) noexcept;

  [[nodiscard]] zzs::kernel::Status
  _M_handle_event_frame (const Frame &frame) noexcept;

  // -- frame callback for the stream pipeline --
  static void
  _M_frame_cb (const Frame &frame, void *context) noexcept;

  // -- members --
  stream::WifiStream _M_stream;
  const platform::WifiHal *_M_hal = nullptr;
  WifiConnectionState _M_state = WifiConnectionState::Disconnected;
  WifiCipher _M_cipher = WifiCipher::None;
  MacAddr _M_mac_addr{};
  crypto::Handshake _M_handshake = crypto::wifi_handshake_make_empty ();
  uint32_t _M_connect_deadline_ms = 0u;
  uint8_t _M_connect_retries = 0u;
  char _M_connect_ssid[WifiMaxSsidLen + 1] = {0};
  uint8_t _M_connect_ssid_len = 0u;
  uint8_t _M_connect_psk[WifiMaxPskLen] = {};
  uint8_t _M_connect_psk_len = 0u;
  bool _M_scan_in_progress = false;
  bool _M_initialized = false;
};

/** @} */

} // namespace zzs::kernel::device::driver::wifi
