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
 * @brief u-blox UBX binary protocol framing and CRC.
 *
 * UBX frame: 0xB5 0x62 | msgClass | msgId | lengthLE(2B) | payload | CK_A | CK_B.
 * Fletcher-8 checksum over msgClass..payload.  Integer-only, constexpr,
 * no heap.  This header is protocol framing only; concrete message parsing
 * (NAV-PVT, NAV-SAT, NAV-DOP, MON-VER, etc.) lives in ubx_parser.hpp.
 * @{
 */
namespace zzs::kernel::device::driver::gnss::ubx {

/** @brief UBX sync chars. */
inline constexpr uint8_t UbxSync1 = 0xb5u;
inline constexpr uint8_t UbxSync2 = 0x62u;

/** @brief Maximum UBX payload length we accept (frames larger than this are
 *         rejected as malformed). */
inline constexpr uint16_t UbxMaxPayloadBytes = 256u;

/** @brief UBX message classes we handle. */
enum class UbxClass : uint8_t {
  Nav = 0x01u,
  Rxm = 0x02u,
  Inf = 0x04u,
  Ack = 0x05u,
  Cfg = 0x06u,
  Mon = 0x0au,
};

/** @brief UBX message IDs (subset). */
enum class UbxMsgId : uint8_t {
  NavPvt = 0x07u,   /**< NAV-PVT: position, velocity, time. */
  NavSat = 0x35u,   /**< NAV-SAT: satellite info. */
  NavDop = 0x04u,   /**< NAV-DOP: dilution of precision. */
  MonVer = 0x04u,   /**< MON-VER: software/version info. */
  AckAck = 0x01u,   /**< ACK-ACK. */
  AckNak = 0x00u,   /**< ACK-NAK. */
};

/** @brief UBX NAV-PVT fixed flags bitmask. */
enum UbxPvtFlags : uint16_t {
  UbxPvtFixValid = 0x0001u,
  UbxPvtDiffSoln = 0x0002u,
  UbxPvtWknSet = 0x0004u,
  UbxPvtTowSet = 0x0008u,
};

/** @brief UBX fixType values. */
enum UbxPvtFixType : uint8_t {
  UbxPvtFixNone = 0,
  UbxPvtFixDr = 1,
  UbxPvtFix2d = 2,
  UbxPvtFix3d = 3,
  UbxPvtFixGpsDr = 4,
  UbxPvtFixTimeOnly = 5,
};

/** @brief Fletcher-8 UBX checksum: returns CK_A in low byte, CK_B in high
 *         byte. */
[[nodiscard]] inline uint16_t
ubx_checksum (const uint8_t *buf, uint16_t len) noexcept {
  uint8_t ck_a = 0u;
  uint8_t ck_b = 0u;
  for (uint16_t i = 0u; i < len; ++i) {
    ck_a = static_cast<uint8_t> (ck_a + buf[i]);
    ck_b = static_cast<uint8_t> (ck_b + ck_a);
  }
  return static_cast<uint16_t> (ck_a) | (static_cast<uint16_t> (ck_b) << 8);
}

/** @brief Verify a complete UBX frame (sync..payload) including the two
 *         checksum bytes. Returns true when valid. */
[[nodiscard]] inline bool
ubx_frame_valid (const uint8_t *frame, uint16_t frame_len) noexcept {
  if (frame_len < 8u)
    return false;
  if (frame[0] != UbxSync1 || frame[1] != UbxSync2)
    return false;
  uint16_t len = static_cast<uint16_t> (frame[4]) | (static_cast<uint16_t> (frame[5]) << 8);
  if (static_cast<uint16_t> (frame_len) != 6u + len + 2u)
    return false;
  uint16_t ck = ubx_checksum (&frame[2], 4u + len);
  uint8_t ck_a = static_cast<uint8_t> (ck & 0xffu);
  uint8_t ck_b = static_cast<uint8_t> ((ck >> 8) & 0xffu);
  return frame[6u + len] == ck_a && frame[7u + len] == ck_b;
}

/** @brief Extract payload length from a UBX frame header (bytes 4-5). */
[[nodiscard]] inline uint16_t
ubx_payload_len (const uint8_t *frame) noexcept {
  return static_cast<uint16_t> (frame[4]) | (static_cast<uint16_t> (frame[5]) << 8);
}

/** @brief Message class from a UBX frame header (byte 2). */
[[nodiscard]] inline uint8_t
ubx_msg_class (const uint8_t *frame) noexcept {
  return frame[2];
}

/** @brief Message ID from a UBX frame header (byte 3). */
[[nodiscard]] inline uint8_t
ubx_msg_id (const uint8_t *frame) noexcept {
  return frame[3];
}

/** @brief NAV-PVT message (little-endian packed). Layout matches the
 *         u-blox specification exactly; fields are read via accessors to
 *         avoid strict-aliasing / unaligned-access issues on Cortex-M4. */
#pragma pack(push, 1)
struct UbxNavPvt {
  uint8_t _M_syn0;          // 0xB5
  uint8_t _M_syn1;          // 0x62
  uint8_t _M_msg_class;     // 0x01
  uint8_t _M_msg_id;        // 0x07
  uint8_t _M_len_l;         // 0x64
  uint8_t _M_len_h;         // 0x00
  uint32_t _M_i_tow;        // 8
  uint16_t _M_year;         // 12
  uint8_t _M_month;         // 14
  uint8_t _M_day;           // 15
  uint8_t _M_hour;          // 16
  uint8_t _M_min;           // 17
  uint8_t _M_sec;           // 18
  uint8_t _M_valid;         // 19
  uint32_t _M_t_acc;        // 20
  int32_t _M_nano;          // 24
  uint8_t _M_fix_type;      // 28
  uint8_t _M_flags;         // 29
  uint8_t _M_reserved1;     // 30
  uint8_t _M_num_sv;        // 31
  int32_t _M_lon;           // 32  (1e-7 deg)
  int32_t _M_lat;           // 36  (1e-7 deg)
  int32_t _M_height;        // 40  (mm)
  int32_t _M_h_msl;         // 44  (mm)
  uint32_t _M_h_acc;        // 48
  uint32_t _M_v_acc;        // 52
  int32_t _M_vel_n;         // 56  (mm/s)
  int32_t _M_vel_e;         // 60  (mm/s)
  int32_t _M_vel_d;         // 64  (mm/s)
  int32_t _M_g_speed;       // 68  (mm/s)
  int32_t _M_head_mot;      // 72  (1e-5 deg)
  uint32_t _M_s_acc;        // 76
  uint32_t _M_head_acc;     // 80
  uint16_t _M_p_dop;        // 82  (0.01)
  uint8_t _M_reserved2[6];  // 84
  uint32_t _M_head_veh;     // 90  (1e-5 deg)
  int16_t _M_mag_dec;       // 94  (0.01 deg)
  uint16_t _M_mag_acc;      // 96  (0.01 deg)
  uint8_t _M_ck_a;          // 98
  uint8_t _M_ck_b;          // 99
};
#pragma pack(pop)

static_assert (sizeof (UbxNavPvt) == 100u, "UBX NAV-PVT must be 100 bytes");
static_assert (offsetof (UbxNavPvt, _M_lat) == 34u, "UBX NAV-PVT lat offset");
static_assert (offsetof (UbxNavPvt, _M_lon) == 30u, "UBX NAV-PVT lon offset");

/** @brief NAV-SAT satellite info (one SV entry). */
#pragma pack(push, 1)
struct UbxNavSatSv {
  uint8_t _M_gnss_id;
  uint8_t _M_sv_id;
  uint8_t _M_cno;
  uint8_t _M_elev;
  int16_t _M_azim;          // 0.01 deg
  int16_t _M_pr_res;        // 0.1 m
  uint32_t _M_flags;
};
#pragma pack(pop)
static_assert (sizeof (UbxNavSatSv) == 12u, "UBX NAV-SAT SV entry is 12 bytes");

/** @brief NAV-DOP message. */
#pragma pack(push, 1)
struct UbxNavDop {
  uint8_t _M_syn0;
  uint8_t _M_syn1;
  uint8_t _M_msg_class;
  uint8_t _M_msg_id;
  uint8_t _M_len_l;
  uint8_t _M_len_h;
  uint32_t _M_g_tow;
  uint16_t _M_g_dop;   // 0.01
  uint16_t _M_p_dop;   // 0.01
  uint16_t _M_t_dop;   // 0.01
  uint16_t _M_v_dop;   // 0.01
  uint16_t _M_h_dop;   // 0.01
  uint16_t _M_n_dop;   // 0.01
  uint16_t _M_e_dop;   // 0.01
  uint8_t _M_ck_a;
  uint8_t _M_ck_b;
};
#pragma pack(pop)
static_assert (sizeof (UbxNavDop) == 26u, "UBX NAV-DOP must be 26 bytes");

/** @} */
} // namespace zzs::kernel::device::driver::gnss::ubx
