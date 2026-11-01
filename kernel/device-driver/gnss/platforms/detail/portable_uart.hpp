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

#include "kernel/device-driver/gnss/gnss_types.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>
#include <cstring>

/**
 * @brief Shared portable fallback for GNSS platform HALs.
 *
 * Provides a fixed ring-buffer simulated UART byte stream and a minimal NMEA
 * $GPGGA parser.  Used by platform HALs when the vendor SDK is absent so every
 * platform translation unit compiles standalone and actually does something
 * (simulated UART stream + meaningful NMEA parse) rather than being a no-op
 * stub.  No heap, no float, no exceptions, all methods noexcept.
 *
 * All symbols live in `zzs::kernel::device::driver::gnss::platform::detail`.
 * @{
 */
namespace zzs::kernel::device::driver::gnss::platform::detail {

/** @brief Maximum depth of the portable ring buffer. */
inline constexpr uint16_t PortableRingMaxDepth = 256u;

/** @brief Maximum NMEA sentence length parsed by the portable parser. */
inline constexpr uint16_t PortableNmeaMaxBytes = 100u;

/** @brief Fixed-size byte ring for the portable simulated UART. */
class PortableRing {
public:
  explicit PortableRing (uint16_t depth) noexcept
      : _M_depth (depth == 0u ? 1u : depth), _M_mask (static_cast<uint16_t> ((depth == 0u ? 1u : depth) - 1u)) {
    _M_clear ();
  }

  PortableRing (const PortableRing &) = delete;
  PortableRing &
  operator= (const PortableRing &) = delete;
  PortableRing (PortableRing &&) = delete;
  PortableRing &
  operator= (PortableRing &&) = delete;

  /** @brief Push `len` bytes from `src`; drop-oldest when full. Returns bytes pushed. */
  [[nodiscard]] uint16_t
  push (const uint8_t *src, uint16_t len) noexcept {
    uint16_t n = 0u;
    while (n < len) {
      _M_buf[_M_head] = src[n];
      _M_head = static_cast<uint16_t> ((_M_head + 1u) & _M_mask);
      if (_M_count < _M_depth)
        _M_count += 1u;
      else
        _M_tail = static_cast<uint16_t> ((_M_tail + 1u) & _M_mask);
      ++n;
    }
    return n;
  }

  /** @brief Pop up to `max_len` bytes into `dst`. Returns bytes popped. */
  [[nodiscard]] uint16_t
  pop (uint8_t *dst, uint16_t max_len) noexcept {
    uint16_t n = 0u;
    while (n < max_len && _M_count > 0u) {
      dst[n++] = _M_buf[_M_tail];
      _M_tail = static_cast<uint16_t> ((_M_tail + 1u) & _M_mask);
      _M_count -= 1u;
    }
    return n;
  }

  [[nodiscard]] uint16_t
  filled () const noexcept {
    return _M_count;
  }

  [[nodiscard]] bool
  empty () const noexcept {
    return _M_count == 0u;
  }

  void
  clear () noexcept {
    _M_clear ();
  }

private:
  void
  _M_clear () noexcept {
    _M_head = 0u;
    _M_tail = 0u;
    _M_count = 0u;
    for (uint16_t i = 0u; i < PortableRingMaxDepth; ++i)
      _M_buf[i] = 0u;
  }

  uint8_t _M_buf[PortableRingMaxDepth] = {};
  uint16_t _M_depth = 1u;
  uint16_t _M_mask = 0u;
  uint16_t _M_head = 0u;
  uint16_t _M_tail = 0u;
  uint16_t _M_count = 0u;
};

/** @brief Minimal NMEA $GPGGA parser: extracts lat/lon (degrees*1e7) and fix. */
class PortableNmea {
public:
  PortableNmea () noexcept = default;

  PortableNmea (const PortableNmea &) = delete;
  PortableNmea &
  operator= (const PortableNmea &) = delete;
  PortableNmea (PortableNmea &&) = delete;
  PortableNmea &
  operator= (PortableNmea &&) = delete;

  /** @brief Feed one byte; returns true when a complete sentence was parsed. */
  [[nodiscard]] bool
  feed (uint8_t b) noexcept {
    if (b == '$') {
      _M_len = 0u;
      _M_in = true;
      _M_ck = 0u;
      _M_ck_state = 0u;
      return false;
    }
    if (!_M_in)
      return false;
    if (b == '\r' || b == '\n') {
      if (_M_len > 0u && _M_buf[_M_len - 1u] == '*') {
        _M_in = false;
        return _M_parse ();
      }
      _M_in = false;
      return false;
    }
    if (b == '*') {
      _M_ck_state = 1u;
      return false;
    }
    if (_M_ck_state == 1u) {
      _M_ck_expected = static_cast<uint16_t> (_M_hex_val (b)) << 4;
      _M_ck_state = 2u;
      return false;
    }
    if (_M_ck_state == 2u) {
      _M_ck_expected |= static_cast<uint16_t> (_M_hex_val (b));
      _M_in = false;
      return _M_ck_expected == _M_ck && _M_parse ();
    }
    if (_M_len < PortableNmeaMaxBytes - 1u) {
      _M_buf[_M_len++] = static_cast<char> (b);
      _M_ck = static_cast<uint8_t> (_M_ck ^ b);
    }
    return false;
  }

  [[nodiscard]] bool
  has_fix () const noexcept {
    return _M_has_fix;
  }

  [[nodiscard]] int32_t
  lat_e7 () const noexcept {
    return _M_lat;
  }

  [[nodiscard]] int32_t
  lon_e7 () const noexcept {
    return _M_lon;
  }

private:
  [[nodiscard]] static uint8_t
  _M_hex_val (char c) noexcept {
    if (c >= '0' && c <= '9')
      return static_cast<uint8_t> (c - '0');
    if (c >= 'A' && c <= 'F')
      return static_cast<uint8_t> (c - 'A' + 10u);
    if (c >= 'a' && c <= 'f')
      return static_cast<uint8_t> (c - 'a' + 10u);
    return 0u;
  }

  [[nodiscard]] bool
  _M_parse () noexcept {
    const char *p = _M_buf;
    if (p[0] != 'G' || p[1] != 'P' || p[2] != 'G' || p[3] != 'G' || p[4] != 'A')
      return false;
    p += 5;
    if (*p != ',')
      return false;
    ++p; // time (ignored)
    while (*p != ',' && *p != '\0')
      ++p;
    if (*p == '\0')
      return false;
    ++p; // latitude
    const char *lat_start = p;
    while (*p != ',' && *p != '\0')
      ++p;
    if (*p == '\0')
      return false;
    char lat_hemi = p[1];
    p += 2;
    const char *lon_start = p;
    while (*p != ',' && *p != '\0')
      ++p;
    if (*p == '\0')
      return false;
    char lon_hemi = p[1];
    p += 2;
    uint8_t quality = static_cast<uint8_t> (*p - '0');
    _M_has_fix = quality >= 1u;
    _M_lat = _M_parse_latlon (lat_start, lat_hemi);
    _M_lon = _M_parse_latlon (lon_start, lon_hemi);
    return true;
  }

  [[nodiscard]] static int32_t
  _M_parse_latlon (const char *s, char hemi) noexcept {
    int32_t deg = 0;
    const char *q = s;
    while (*q >= '0' && *q <= '9' && deg < 1000) {
      deg = deg * 10 + (*q - '0');
      ++q;
    }
    int32_t min_int = 0;
    while (*q >= '0' && *q <= '9' && min_int < 100) {
      min_int = min_int * 10 + (*q - '0');
      ++q;
    }
    int32_t dec_val = 0;
    uint8_t dec_digits = 0;
    if (*q == '.') {
      ++q;
      while (*q >= '0' && *q <= '9' && dec_digits < 4u) {
        dec_val = dec_val * 10 + (*q - '0');
        ++dec_digits;
        ++q;
      }
    }
    while (dec_digits < 4u) {
      dec_val *= 10;
      ++dec_digits;
    }
    int32_t total_min = min_int * 10000 + dec_val; // minutes * 1e4
    int32_t deg_e7 = deg * 10000000;
    int32_t min_e7 = static_cast<int32_t> ((static_cast<int64_t> (total_min) * 1000000ll) / 600000ll);
    int32_t result = deg_e7 + min_e7;
    if (hemi == 'S' || hemi == 'W')
      result = -result;
    return result;
  }

  char _M_buf[PortableNmeaMaxBytes] = {};
  uint16_t _M_len = 0u;
  bool _M_in = false;
  uint8_t _M_ck = 0u;
  uint16_t _M_ck_expected = 0u;
  uint8_t _M_ck_state = 0u; // 0=idle,1=hi,2=lo
  bool _M_has_fix = false;
  int32_t _M_lat = 0;
  int32_t _M_lon = 0;
};

/** @} */
} // namespace zzs::kernel::device::driver::gnss::platform::detail
