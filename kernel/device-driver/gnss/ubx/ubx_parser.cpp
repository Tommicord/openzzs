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

#include "kernel/device-driver/gnss/ubx/ubx_parser.hpp"

namespace zzs::kernel::device::driver::gnss::ubx {

[[nodiscard]] zzs::kernel::Status
Parser::feed (uint8_t b) noexcept {
  switch (_M_state) {
  case UbxParseState::WaitSync1:
    if (b == UbxSync1) {
      _M_frame[0] = b;
      _M_frame_len = 1u;
      _M_state = UbxParseState::WaitSync2;
    }
    return zzs::kernel::Status::STALE;

  case UbxParseState::WaitSync2:
    if (b == UbxSync2) {
      _M_frame[1] = b;
      _M_frame_len = 2u;
      _M_state = UbxParseState::ReadHeader;
    } else {
      /* Resync: this byte might itself be Sync1. */
      _M_state = (b == UbxSync1) ? UbxParseState::WaitSync2 : UbxParseState::WaitSync1;
      if (b == UbxSync1) {
        _M_frame[0] = b;
        _M_frame_len = 1u;
      }
    }
    return zzs::kernel::Status::STALE;

  case UbxParseState::ReadHeader:
    _M_frame[_M_frame_len++] = b;
    if (_M_frame_len == 6u) { // sync(2) + class + id + len(2)
      _M_payload_len = ubx_payload_len (_M_frame);
      if (_M_payload_len > UbxMaxPayloadBytes) {
        _M_state = UbxParseState::WaitSync1;
        _M_errors += 1u;
        return zzs::kernel::Status::BAD_LENGTH;
      }
      _M_state = UbxParseState::ReadPayload;
    }
    return zzs::kernel::Status::STALE;

  case UbxParseState::ReadPayload:
    _M_frame[_M_frame_len++] = b;
    if (_M_frame_len == 6u + _M_payload_len) {
      _M_state = UbxParseState::ReadCkA;
    }
    return zzs::kernel::Status::STALE;

  case UbxParseState::ReadCkA:
    _M_frame[_M_frame_len++] = b;
    _M_ck_a = b;
    _M_state = UbxParseState::ReadCkB;
    return zzs::kernel::Status::STALE;

  case UbxParseState::ReadCkB: {
    _M_frame[_M_frame_len++] = b;
    _M_ck_b = b;
    _M_state = UbxParseState::WaitSync1;
    return _M_complete_frame ();
  }
  }

  _M_state = UbxParseState::WaitSync1;
  _M_errors += 1u;
  return zzs::kernel::Status::BAD_LENGTH;
}

[[nodiscard]] zzs::kernel::Status
Parser::_M_complete_frame () noexcept {
  if (!ubx_frame_valid (_M_frame, _M_frame_len)) {
    _M_errors += 1u;
    return zzs::kernel::Status::BAD_CRC;
  }

  _M_last = DecodedMessage{};
  _M_last._M_class = static_cast<UbxClass> (ubx_msg_class (_M_frame));
  _M_last._M_id = static_cast<UbxMsgId> (ubx_msg_id (_M_frame));
  _M_messages_decoded += 1u;

  const uint8_t *payload = &_M_frame[6];
  uint16_t plen = ubx_payload_len (_M_frame);

  if (_M_last._M_class == UbxClass::Nav && _M_last._M_id == UbxMsgId::NavPvt) {
    _M_last._M_is_nav_pvt = true;
  } else if (_M_last._M_class == UbxClass::Nav && _M_last._M_id == UbxMsgId::NavSat) {
    _M_last._M_is_nav_sat = true;
  } else if (_M_last._M_class == UbxClass::Nav && _M_last._M_id == UbxMsgId::NavDop) {
    _M_last._M_is_nav_dop = true;
  } else if (_M_last._M_class == UbxClass::Ack) {
    _M_last._M_is_ack = true;
    _M_last._M_ack_ok = (plen >= 2u && payload[1] == 0x01u); // ACK-ACK
  }

  return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
Parser::decode_nav_pvt (Fix &out) const noexcept {
  if (!_M_last._M_is_nav_pvt)
    return zzs::kernel::Status::STALE;

  const uint8_t *payload = &_M_frame[6];
  uint16_t plen = ubx_payload_len (_M_frame);
  if (plen < static_cast<uint16_t> (sizeof (UbxNavPvt) - 8u)) // subtract header+checksum
    return zzs::kernel::Status::BAD_LENGTH;

  const UbxNavPvt &pvt = *reinterpret_cast<const UbxNavPvt *> (_M_frame);

  out = fix_make_invalid ();
  out._M_lat = pvt._M_lat;
  out._M_lon = pvt._M_lon;
  out._M_alt_mm = pvt._M_height;
  out._M_alt_msl_mm = pvt._M_h_msl;
  out._M_speed_mm_s = pvt._M_g_speed;
  out._M_course_deg_tenths = static_cast<int32_t> (pvt._M_head_mot) / 10000ll; // 1e-5 deg -> tenths of deg
  out._M_num_satellites = pvt._M_num_sv;
  out._M_hdop_tenths = static_cast<int16_t> ((static_cast<int32_t> (pvt._M_p_dop) * 10ll) / 100ll);
  out._M_vdop_tenths = static_cast<int16_t> ((static_cast<int32_t> (pvt._M_p_dop) * 10ll) / 100ll);

  /* Fix status from fixType and valid flags. */
  bool valid = (pvt._M_valid & UbxPvtFixValid) != 0u;
  out._M_status = valid ? _M_fix_type_to_status (pvt._M_fix_type) : FixStatus::NoFix;
  return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
Parser::decode_nav_sat (Satellite *out, uint8_t max_count, uint8_t *filled) const noexcept {
  if (out == nullptr || filled == nullptr)
    return zzs::kernel::Status::OUT_OF_RANGE;
  if (!_M_last._M_is_nav_sat)
    return zzs::kernel::Status::STALE;

  const uint8_t *payload = &_M_frame[6];
  uint16_t plen = ubx_payload_len (_M_frame);
  if (plen < 2u)
    return zzs::kernel::Status::BAD_LENGTH;

  /* NAV-SAT payload: numSvs (1 byte), reserved (1 byte), then 12-byte entries. */
  uint8_t num_svs = payload[0];
  uint16_t entry_off = 2u;
  uint8_t n = 0u;
  for (uint8_t i = 0u; i < num_svs && n < max_count; ++i) {
    if (entry_off + sizeof (UbxNavSatSv) > plen)
      break;
    const UbxNavSatSv &sv = *reinterpret_cast<const UbxNavSatSv *> (payload + entry_off);
    out[n]._M_prn = sv._M_sv_id;
    out[n]._M_constellation = _M_gnss_id_to_constellation (sv._M_gnss_id);
    out[n]._M_elevation_deg = static_cast<int8_t> (sv._M_elev);
    out[n]._M_azimuth_deg_tenths = static_cast<int16_t> (sv._M_azim);
    out[n]._M_snr_dbhz = sv._M_cno;
    out[n]._M_used_in_fix = (sv._M_flags & 0x01u) != 0u;
    entry_off += static_cast<uint16_t> (sizeof (UbxNavSatSv));
    ++n;
  }
  *filled = n;
  return zzs::kernel::Status::OK;
}

[[nodiscard]] zzs::kernel::Status
Parser::decode_nav_dop (int16_t &p_dop_tenths, int16_t &h_dop_tenths, int16_t &v_dop_tenths) const noexcept {
  if (!_M_last._M_is_nav_dop)
    return zzs::kernel::Status::STALE;

  const UbxNavDop &dop = *reinterpret_cast<const UbxNavDop *> (_M_frame);
  /* DOP fields are in 0.01 units; convert to tenths (0.1). */
  p_dop_tenths = static_cast<int16_t> ((static_cast<int32_t> (dop._M_p_dop) * 10ll) / 100ll);
  h_dop_tenths = static_cast<int16_t> ((static_cast<int32_t> (dop._M_h_dop) * 10ll) / 100ll);
  v_dop_tenths = static_cast<int16_t> ((static_cast<int32_t> (dop._M_v_dop) * 10ll) / 100ll);
  return zzs::kernel::Status::OK;
}

[[nodiscard]] FixStatus
Parser::_M_fix_type_to_status (uint8_t fix_type) noexcept {
  switch (fix_type) {
  case UbxPvtFixNone:
    return FixStatus::NoFix;
  case UbxPvtFixDr:
    return FixStatus::DeadReckoning;
  case UbxPvtFix2d:
    return FixStatus::Gps2d;
  case UbxPvtFix3d:
    return FixStatus::Gps3d;
  case UbxPvtFixGpsDr:
    return FixStatus::GpsDr;
  case UbxPvtFixTimeOnly:
    return FixStatus::TimeOnly;
  default:
    return FixStatus::NoFix;
  }
}

[[nodiscard]] Constellation
Parser::_M_gnss_id_to_constellation (uint8_t gnss_id) noexcept {
  switch (gnss_id) {
  case 0u:
    return Constellation::Gps;
  case 1u:
    return Constellation::Sbas;
  case 2u:
    return Constellation::Galileo;
  case 3u:
    return Constellation::BeiDou;
  case 5u:
    return Constellation::Qzss;
  case 6u:
    return Constellation::Glonass;
  default:
    return Constellation::Unknown;
  }
}

} // namespace zzs::kernel::device::driver::gnss::ubx
