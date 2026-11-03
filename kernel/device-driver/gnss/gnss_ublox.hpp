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

#include "kernel/device-driver/gnss/gnss_driver.hpp"
#include "kernel/device-driver/gnss/stream/gnss_stream.hpp"
#include "kernel/device-driver/gnss/ubx/ubx_parser.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief Concrete u-blox GNSS receiver driver (ZED-F9P, NEO-M9N, etc.).
 *
 * Configures the receiver over UBX (NAV-PVT / NAV-SAT / NAV-DOP at 1 Hz,
 * NMEA GGA/RMC/GSA/GSV/VTG) and decodes both UBX and NMEA in the streaming
 * pipeline.  Derives from GnssDriver, owns a fixed GnssStream, and adds
 * u-blox-specific configuration commands (CFG-PRT, CFG-MSG, CFG-RATE).
 *
 * All state is fixed-size; no heap, no exceptions, noexcept throughout.
 * @{
 */
namespace zzs::kernel::device::driver::gnss {

/** @brief u-blox driver configuration. */
struct UbloxConfig {
    uint32_t _M_baud_rate = 9600u;    /**< UART baud rate. */
    uint16_t _M_nav_rate_ms = 1000u;  /**< Navigation solution rate (ms). */
    uint8_t _M_protocol_mask = 0x03u; /**< Bit0=NMEA, bit1=UBX. */
};

/** @brief Concrete u-blox GNSS driver. */
class UbloxDriver : public GnssDriver {
  public:
    /** @brief Construct with a stream bus, optional clock, and config. */
    UbloxDriver (zzs::kernel::device::driver::StreamBusInterface *bus, uint64_t (*clock) (void) = nullptr,
                 const UbloxConfig &config = UbloxConfig{}) noexcept
        : GnssDriver (bus, clock), _M_stream (bus, clock), _M_config (config) {
    }

    ~UbloxDriver () override = default;
    UbloxDriver (const UbloxDriver &) = delete;
    UbloxDriver &
    operator= (const UbloxDriver &) = delete;
    UbloxDriver (UbloxDriver &&) = delete;
    UbloxDriver &
    operator= (UbloxDriver &&) = delete;

    // -- GnssDriver --
    [[nodiscard]] zzs::kernel::Status
    init () noexcept override;

    [[nodiscard]] zzs::kernel::Status
    deinit () noexcept override;

    [[nodiscard]] zzs::kernel::Status
    poll () noexcept override;

    [[nodiscard]] zzs::kernel::Status
    read_fix (Fix &out) noexcept override;

    [[nodiscard]] zzs::kernel::Status
    read_satellites (Satellite *out, uint8_t max_count, uint8_t *filled) noexcept override;

    [[nodiscard]] GnssReceiverStatus
    receiver_status () const noexcept override {
        return _M_status;
    }

    [[nodiscard]] bool
    self_test () noexcept override;

    [[nodiscard]] uint8_t
    protocol_mask () const noexcept override {
        return _M_config._M_protocol_mask;
    }

    /** @brief Send a raw UBX command payload (message class/id + payload),
     *         computes and appends the Fletcher-8 checksum. */
    [[nodiscard]] zzs::kernel::Status
    send_ubx (uint8_t msg_class, uint8_t msg_id, const uint8_t *payload, uint16_t payload_len) noexcept;

    /** @brief Access the underlying stream (read-only). */
    [[nodiscard]] const stream::GnssStream &
    stream () const noexcept {
        return _M_stream;
    }

  private:
    [[nodiscard]] zzs::kernel::Status
    _M_configure_receiver () noexcept;

    /** @brief Block until an UBX ACK-ACK/ACK-NAK for the last config command is
     *         observed on the bus, or a timeout elapses. */
    [[nodiscard]] zzs::kernel::Status
    _M_wait_for_ack () noexcept;

    stream::GnssStream _M_stream;
    UbloxConfig _M_config;
    bool _M_configured = false;
};

/** @} */
} // namespace zzs::kernel::device::driver::gnss
