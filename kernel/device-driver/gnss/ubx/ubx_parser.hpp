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
#include "kernel/device-driver/gnss/ubx/ubx_framing.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief UBX binary message decoder.
 *
 * Incrementally assembles UBX frames byte-by-byte (`feed`), validates the
 * Fletcher-8 checksum, and decodes NAV-PVT (fix), NAV-SAT (satellites) and
 * NAV-DOP (dilution of precision) into the common `Fix` / `Satellite` types.
 * Fixed per-instance frame buffer; no heap, no float, no exceptions.
 * @{
 */
namespace zzs::kernel::device::driver::gnss::ubx {

/** @brief UBX frame parser state machine. */
enum class UbxParseState : uint8_t {
  WaitSync1 = 0,
  WaitSync2 = 1,
  ReadHeader = 2,   // class, id, len_l, len_h
  ReadPayload = 3,
  ReadCkA = 4,
  ReadCkB = 5,
};

/** @brief Result of decoding one UBX message. */
struct DecodedMessage {
  UbxClass _M_class = UbxClass::Nav;
  UbxMsgId _M_id = UbxMsgId::NavPvt;
  bool _M_is_nav_pvt = false;
  bool _M_is_nav_sat = false;
  bool _M_is_nav_dop = false;
  bool _M_is_ack = false;
  bool _M_ack_ok = false;
};

/** @brief Incremental UBX frame parser. */
class Parser {
public:
  Parser () noexcept = default;
  Parser (const Parser &) = delete;
  Parser &operator= (const Parser &) = delete;

  /** @brief Feed one raw byte. Returns OK when a complete valid UBX message
   *         was decoded and stored, STALE when more bytes are needed, or a
   *         protocol error (bad checksum, bad sync, oversize). */
  [[nodiscard]] zzs::kernel::Status
  feed (uint8_t b) noexcept;

  /** @brief Return the last decoded message. */
  [[nodiscard]] const DecodedMessage &
  last_message () const noexcept {
    return _M_last;
  }

  /** @brief Decode the last NAV-PVT into a `Fix`. Returns OK when the last
   *         message was NAV-PVT and carried a usable fix. */
  [[nodiscard]] zzs::kernel::Status
  decode_nav_pvt (Fix &out) const noexcept;

  /** @brief Decode the last NAV-SAT into satellites. Returns OK + count. */
  [[nodiscard]] zzs::kernel::Status
  decode_nav_sat (Satellite *out, uint8_t max_count, uint8_t *filled) const noexcept;

  /** @brief Decode the last NAV-DOP into DOP tenths. */
  [[nodiscard]] zzs::kernel::Status
  decode_nav_dop (int16_t &p_dop_tenths, int16_t &h_dop_tenths,
                  int16_t &v_dop_tenths) const noexcept;

  /** @brief Number of messages decoded. */
  [[nodiscard]] uint32_t
  messages_decoded () const noexcept {
    return _M_messages_decoded;
  }

  /** @brief Number of checksum / framing errors. */
  [[nodiscard]] uint32_t
  errors () const noexcept {
    return _M_errors;
  }

private:
  [[nodiscard]] zzs::kernel::Status
  _M_complete_frame () noexcept;

  [[nodiscard]] static FixStatus
  _M_fix_type_to_status (uint8_t fix_type) noexcept;
  [[nodiscard]] static Constellation
  _M_gnss_id_to_constellation (uint8_t gnss_id) noexcept;

  UbxParseState _M_state = UbxParseState::WaitSync1;
  uint8_t _M_frame[8u + UbxMaxPayloadBytes + 2u] = {};
  uint16_t _M_frame_len = 0u;
  uint16_t _M_payload_len = 0u;
  uint8_t _M_ck_a = 0u;
  uint8_t _M_ck_b = 0u;

  DecodedMessage _M_last{};
  uint32_t _M_messages_decoded = 0u;
  uint32_t _M_errors = 0u;
};

/** @} */
} // namespace zzs::kernel::device::driver::gnss::ubx
