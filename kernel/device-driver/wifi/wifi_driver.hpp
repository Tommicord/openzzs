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
#include "kernel/device-driver/wifi/wifi_types.hpp"
#include "kernel/kernel_status.hpp"
#include "kernel/kernel_clock_source.hpp"

#include <cstdint>

/**
 * @brief WiFi (802.11) antenna/radio driver base class.
 *
 * Mirrors `kernel/sensor-driver/sensors/sensor_driver.hpp`: an abstract OOP
 * base with a non-owning stream-bus pointer, fixed-size state, `noexcept`
 * virtual methods, integer-only arithmetic, and no heap allocation.  Concrete
 * WiFi chipsets (ESP32, esp8266, Realtek RTL871x, etc.) derive from this class
 * and implement `init`, `scan`, `connect`, `poll`, `read_frame` and
 * protocol-specific 802.11 frame handling.
 *
 * All state lives in a fixed per-instance arena: one latest frame buffer, a
 * scanned BSS table, and connection state.  No heap, no exceptions, no float.
 * @{
 */
namespace zzs::kernel::device::driver::wifi {

/** @brief Maximum 802.11 frame payload the driver buffers. */
inline constexpr uint16_t WifiMaxFrameBytes = 256u;

/** @brief Maximum consecutive frame errors before reporting error state. */
inline constexpr uint8_t WifiMaxConsecutiveErrors = 8u;

/** @brief Encryption cipher for a connection. */
enum class WifiCipher : uint8_t {
  None = 0,     /**< No encryption (open). */
  Tkip = 1,     /**< TKIP (WPA). */
  Ccmp = 2,     /**< CCMP/AES (WPA2). */
  Gcmp = 3,     /**< GCMP/AES-256 (WPA3). */
  Unknown = 0xff,
};

/** @brief Return a short name for a cipher. */
[[nodiscard]] inline const char *
wifi_cipher_name (WifiCipher c) noexcept {
  switch (c) {
  case WifiCipher::None:
    return "none";
  case WifiCipher::Tkip:
    return "tkip";
  case WifiCipher::Ccmp:
    return "ccmp";
  case WifiCipher::Gcmp:
    return "gcmp";
  default:
    return "unknown";
  }
}

/** @brief One received 802.11 frame (fixed buffer). */
struct Frame {
  uint8_t _M_buf[WifiMaxFrameBytes] = {};
  uint16_t _M_len = 0u;
  uint64_t _M_timestamp_ns = 0ull;
  int8_t _M_rssi_dbm = 0;
  uint8_t _M_channel = 0;
  WifiFrameType _M_type = WifiFrameType::Management;
  WifiMgmtSubtype _M_mgmt_subtype = WifiMgmtSubtype::Beacon;
};

/** @brief Clear a frame to empty. */
[[nodiscard]] inline Frame
frame_make_empty () noexcept {
  Frame f{};
  return f;
}

/** @brief Abstract base class every WiFi driver implements. */
class WifiDriver {
public:
  /** @brief Construct with a non-owning stream-bus pointer and optional clock
   *         source. */
  explicit WifiDriver (zzs::kernel::device::driver::StreamBusInterface *bus,
                       uint64_t (*clock) (void) = nullptr) noexcept
      : _M_bus (bus), _M_clock (clock) {
  }

  /** @brief Virtual destructor, defaulted. */
  virtual ~WifiDriver () = default;

  WifiDriver (const WifiDriver &) = delete;
  WifiDriver &operator= (const WifiDriver &) = delete;
  WifiDriver (WifiDriver &&) = delete;
  WifiDriver &operator= (WifiDriver &&) = delete;

  /** @brief Initialize the WiFi radio: reset, set country/channel, enable
   *         the interface. Returns OK when the radio responds. */
  [[nodiscard]] virtual zzs::kernel::Status
  init () noexcept = 0;

  /** @brief Deinitialize / power down the radio. */
  [[nodiscard]] virtual zzs::kernel::Status
  deinit () noexcept = 0;

  /** @brief Start a scan for BSSes. Returns OK when the scan is initiated. */
  [[nodiscard]] virtual zzs::kernel::Status
  scan () noexcept = 0;

  /** @brief Poll the radio for new frames/events. Reads available bytes from
   *         the bus, parses 802.11 frames, updates the BSS table and
   *         connection state. Returns OK when at least one frame/event was
   *         processed, STALE when nothing new, NOT_READY while warming up. */
  [[nodiscard]] virtual zzs::kernel::Status
  poll () noexcept = 0;

  /** @brief Connect to an SSID. Returns OK when the association completes
   *         (synchronously or initiated; poll() drives completion). */
  [[nodiscard]] virtual zzs::kernel::Status
  connect (const char *ssid, uint8_t ssid_len,
           const uint8_t *psk, uint8_t psk_len) noexcept = 0;

  /** @brief Disconnect from the current AP. */
  [[nodiscard]] virtual zzs::kernel::Status
  disconnect () noexcept = 0;

  /** @brief Read the latest received frame into `out`. Returns OK when a
   *         frame is available, STALE when none. */
  [[nodiscard]] virtual zzs::kernel::Status
  read_frame (Frame &out) noexcept = 0;

  /** @brief Return the scanned BSS table. */
  [[nodiscard]] virtual zzs::kernel::Status
  read_bss_table (BssTable &out) noexcept = 0;

  /** @brief Return the current connection state. */
  [[nodiscard]] virtual WifiConnectionState
  connection_state () const noexcept = 0;

  /** @brief Return the current cipher negotiated for the active connection. */
  [[nodiscard]] virtual WifiCipher
  connection_cipher () const noexcept = 0;

  /** @brief Return the MAC address of this station. */
  [[nodiscard]] virtual MacAddr
  mac_addr () const noexcept = 0;

  /** @brief Number of consecutive frame parse errors since the last good
   *         frame. Useful for fault detection. */
  [[nodiscard]] uint8_t
  consecutive_errors () const noexcept {
    return _M_consecutive_errors;
  }

protected:
  /** @brief Stream bus the driver talks to; non-owning, set by the constructor. */
  zzs::kernel::device::driver::StreamBusInterface *_M_bus = nullptr;

  /** @brief Clock source callback for timestamps (may be nullptr). */
  uint64_t (*_M_clock) (void) = nullptr;

  /** @brief Latest received frame (cached by `poll`). */
  Frame _M_frame = frame_make_empty ();

  /** @brief Scanned BSS table (cached by `poll`). */
  BssTable _M_bss_table{};

  /** @brief Current connection state. */
  WifiConnectionState _M_state = WifiConnectionState::Disconnected;

  /** @brief Negotiated cipher for the active connection. */
  WifiCipher _M_cipher = WifiCipher::None;

  /** @brief This station's MAC address. */
  MacAddr _M_mac_addr{};

  /** @brief Number of consecutive frame parse errors. */
  uint8_t _M_consecutive_errors = 0u;

  /** @brief Return the current timestamp: per-instance clock callback when
   *         supplied, otherwise the global kernel clock source, or 0. */
  [[nodiscard]] inline uint64_t
  timestamp_ns () const noexcept {
    if (_M_clock != nullptr)
      return _M_clock ();
    return zzs::kernel::clock::clock_source_now_ns ();
  }

  /** @brief Read up to `max_bytes` from the stream bus into `buf`; `*filled`
   *         is set to the number of bytes actually read. */
  [[nodiscard]] inline zzs::kernel::Status
  bus_read (uint8_t *buf, uint16_t max_bytes, uint16_t *filled) noexcept {
    if (buf == nullptr || filled == nullptr)
      return zzs::kernel::Status::OUT_OF_RANGE;
    if (_M_bus == nullptr)
      return zzs::kernel::Status::INTERNAL;
    return _M_bus->read_bytes (buf, max_bytes, filled);
  }

  /** @brief Write `len` bytes from `buf` to the stream bus. */
  [[nodiscard]] inline zzs::kernel::Status
  bus_write (const uint8_t *buf, uint16_t len) noexcept {
    if (buf == nullptr && len > 0u)
      return zzs::kernel::Status::OUT_OF_RANGE;
    if (_M_bus == nullptr)
      return zzs::kernel::Status::INTERNAL;
    return _M_bus->write_bytes (buf, len);
  }

  /** @brief Block for at least `ms` milliseconds via the bus. */
  inline void
  delay_ms (uint32_t ms) noexcept {
    if (_M_bus != nullptr)
      _M_bus->delay_ms (ms);
  }

  /** @brief Flush the radio's receive buffer. */
  [[nodiscard]] inline zzs::kernel::Status
  bus_flush_rx () noexcept {
    if (_M_bus == nullptr)
      return zzs::kernel::Status::INTERNAL;
    return _M_bus->flush_rx ();
  }

  /** @brief Record a frame parse error and transition to Failed when the
   *         threshold is exceeded. */
  inline void
  _M_record_error () noexcept {
    if (_M_consecutive_errors < WifiMaxConsecutiveErrors)
      _M_consecutive_errors += 1u;
    if (_M_consecutive_errors >= WifiMaxConsecutiveErrors)
      _M_state = WifiConnectionState::Failed;
  }

  /** @brief Clear the error counter. */
  inline void
  _M_clear_errors () noexcept {
    _M_consecutive_errors = 0u;
    if (_M_state == WifiConnectionState::Failed)
      _M_state = WifiConnectionState::Disconnected;
  }

  /** @brief Publish a received frame: cache it and clear errors. */
  inline void
  _M_publish_frame (const Frame &frame) noexcept {
    _M_frame = frame;
    _M_clear_errors ();
  }

  /** @brief Publish a scanned BSS table. */
  inline void
  _M_publish_bss_table (const BssTable &table) noexcept {
    _M_bss_table = table;
    _M_clear_errors ();
  }
};

/** @} */
} // namespace zzs::kernel::device::driver::wifi
