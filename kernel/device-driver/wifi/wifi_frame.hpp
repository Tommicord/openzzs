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
#include "kernel/device-driver/wifi/wifi_driver.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>
#include <cstring>

/**
 * @brief 802.11 MAC frame parser.
 *
 * Parses raw 802.11 MAC frames (the bytes delivered by a WiFi radio's stream
 * bus) into the common `Frame` type: frame control type/subtype, duration,
 * address fields (addr1/addr2/addr3), sequence control, and payload.  All
 * header access is done byte-by-byte to avoid unaligned/aliasing issues on
 * Cortex-M-class MCUs.  No heap, no float, no exceptions.
 * @{
 */
namespace zzs::kernel::device::driver::wifi {

/** @brief 802.11 MAC header fixed size (24 bytes for most frames). */
inline constexpr uint16_t WifiMacHeaderLen = 24u;

/** @brief Frame Control field offset within the MAC header. */
inline constexpr uint16_t WifiFcOffset = 0u;
/** @brief Duration ID field offset. */
inline constexpr uint16_t WifiDurOffset = 2u;
/** @brief Address 1 offset. */
inline constexpr uint16_t WifiAddr1Offset = 4u;
/** @brief Address 2 offset. */
inline constexpr uint16_t WifiAddr2Offset = 10u;
/** @brief Address 3 offset. */
inline constexpr uint16_t WifiAddr3Offset = 16u;
/** @brief Sequence control offset. */
inline constexpr uint16_t WifiSeqCtrlOffset = 22u;

/** @brief Frame Control bit masks. */
enum WifiFcMask : uint16_t {
  WifiFcVersionMask = 0x0003u,      // bits 0-1
  WifiFcTypeMask = 0x000cu,         // bits 2-3
  WifiFcSubtypeMask = 0x00f0u,      // bits 4-7
  WifiFcToDsMask = 0x0100u,         // bit 8
  WifiFcFromDsMask = 0x0200u,       // bit 9
  WifiFcMoreFragMask = 0x0400u,     // bit 10
  WifiFcRetryMask = 0x0800u,        // bit 11
  WifiFcPwrMgmtMask = 0x1000u,      // bit 12
  WifiFcMoreDataMask = 0x2000u,     // bit 13
  WifiFcProtectedMask = 0x4000u,    // bit 14
  WifiFcOrderMask = 0x8000u,        // bit 15
};

/** @brief Parse the Frame Control type bits. */
[[nodiscard]] inline WifiFrameType
wifi_fc_type (uint16_t fc) noexcept {
  return static_cast<WifiFrameType> ((fc & WifiFcTypeMask) >> 2);
}

/** @brief Parse the Frame Control subtype bits. */
[[nodiscard]] inline uint8_t
wifi_fc_subtype (uint16_t fc) noexcept {
  return static_cast<uint8_t> ((fc & WifiFcSubtypeMask) >> 4);
}

/** @brief Read a little-endian uint16 from a buffer at `off`. */
[[nodiscard]] inline uint16_t
wifi_read_u16_le (const uint8_t *buf, uint16_t off) noexcept {
  return static_cast<uint16_t> (buf[off]) | (static_cast<uint16_t> (buf[off + 1]) << 8);
}

/** @brief Copy a 6-byte MAC address from a buffer at `off` into a MacAddr. */
[[nodiscard]] inline MacAddr
wifi_read_mac (const uint8_t *buf, uint16_t off) noexcept {
  MacAddr m{};
  for (uint8_t i = 0u; i < WifiMacAddrLen; ++i)
    m._M_octets[i] = buf[off + i];
  return m;
}

/** @brief Parse a raw 802.11 MAC frame into a `Frame`.
 *
 * Returns OK when `len` is at least the MAC header length and the frame
 * control version field is 0 (802.11).  Otherwise returns BAD_TYPE or
 * BAD_LENGTH.  The frame's address fields and type/subtype are filled in;
 * `rssi_dbm` and `channel` are supplied by the caller (radio metadata).
 * @{
 */
[[nodiscard]] inline zzs::kernel::Status
wifi_parse_frame (const uint8_t *buf, uint16_t len, Frame &out) noexcept {
  out = frame_make_empty ();
  if (len < WifiMacHeaderLen)
    return zzs::kernel::Status::BAD_LENGTH;
  if (len > WifiMaxFrameBytes)
    return zzs::kernel::Status::BAD_LENGTH;

  uint16_t fc = wifi_read_u16_le (buf, WifiFcOffset);
  if ((fc & WifiFcVersionMask) != 0u)
    return zzs::kernel::Status::BAD_TYPE;

  out._M_type = wifi_fc_type (fc);
  out._M_mgmt_subtype = static_cast<WifiMgmtSubtype> (wifi_fc_subtype (fc));

  /* Copy the raw frame into the fixed frame buffer. */
  std::memcpy (out._M_buf, buf, len);
  out._M_len = len;
  return zzs::kernel::Status::OK;
}

/** @} */
} // namespace zzs::kernel::device::driver::wifi
