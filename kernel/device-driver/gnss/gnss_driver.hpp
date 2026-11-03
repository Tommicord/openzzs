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
#include "kernel/device-driver/stream_bus.hpp"
#include "kernel/kernel_clock_source.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief GNSS (GPS/GLONASS/Galileo/BeiDou) receiver driver base class.
 *
 * Mirrors `kernel/sensor-driver/sensors/sensor_driver.hpp`: an abstract OOP
 * base with a non-owning stream-bus pointer, fixed-size state, `noexcept`
 * virtual methods, integer-only arithmetic, and no heap allocation.  Concrete
 * receivers (u-blox ZED-F9P, Quectel L76, etc.) derive from this class and
 * implement `init`, `poll`, `read_fix` and the protocol-specific frame parser.
 *
 * All state lives in a fixed per-instance arena: one latest `Fix`, one
 * satellite table, and a small byte buffer for incomplete NMEA/UBX frames.
 * @{
 */
namespace zzs::kernel::device::driver::gnss {

/** @brief Maximum consecutive protocol errors before the driver reports an
 *         error state. */
inline constexpr uint8_t GnssMaxConsecutiveErrors = 8u;

/** @brief Abstract base class every GNSS receiver driver implements. */
class GnssDriver {
  public:
    /** @brief Construct with a non-owning stream-bus pointer and optional clock
     *         source. */
    explicit GnssDriver (zzs::kernel::device::driver::StreamBusInterface *bus,
                         uint64_t (*clock) (void) = nullptr) noexcept
        : _M_bus (bus), _M_clock (clock) {
    }

    /** @brief Virtual destructor, defaulted. */
    virtual ~GnssDriver () = default;

    GnssDriver (const GnssDriver &) = delete;
    GnssDriver &
    operator= (const GnssDriver &) = delete;
    GnssDriver (GnssDriver &&) = delete;
    GnssDriver &
    operator= (GnssDriver &&) = delete;

    /** @brief Initialize the receiver: baud rate, NMEA/UBX output messages,
     *         power mode. Returns OK when the device responds. */
    [[nodiscard]] virtual zzs::kernel::Status
    init () noexcept = 0;

    /** @brief Deinitialize / power down the receiver. */
    [[nodiscard]] virtual zzs::kernel::Status
    deinit () noexcept = 0;

    /** @brief Poll the receiver for new data.  Reads all available bytes from
     *         the bus, feeds them through the protocol parser, and updates the
     *         cached fix/satellite table. Returns OK when at least one new fix
     *         or update was processed, STALE when nothing new is ready,
     *         NOT_READY while the receiver is warming up, or a bus/protocol
     *         error. */
    [[nodiscard]] virtual zzs::kernel::Status
    poll () noexcept = 0;

    /** @brief Return the latest usable fix (copy into `out`). Returns OK when a
     *         usable fix is cached, STALE when none is available. */
    [[nodiscard]] virtual zzs::kernel::Status
    read_fix (Fix &out) noexcept = 0;

    /** @brief Return the latest satellite-in-view table. */
    [[nodiscard]] virtual zzs::kernel::Status
    read_satellites (Satellite *out, uint8_t max_count, uint8_t *filled) noexcept = 0;

    /** @brief Return the receiver's current operational status. */
    [[nodiscard]] virtual GnssReceiverStatus
    receiver_status () const noexcept = 0;

    /** @brief Run the built-in self-test (cold-start / warm-start verification).
     *         Returns true when the receiver reports a valid fix within the
     *         self-test timeout. */
    [[nodiscard]] virtual bool
    self_test () noexcept = 0;

    /** @brief Return the receiver protocol mask (NMEA / UBX / both). */
    [[nodiscard]] virtual uint8_t
    protocol_mask () const noexcept = 0;

    /** @brief Number of consecutive parser errors since the last successful
     *         message. Useful for fault detection. */
    [[nodiscard]] uint8_t
    consecutive_errors () const noexcept {
        return _M_consecutive_errors;
    }

  protected:
    /** @brief Stream bus the driver talks to; non-owning, set by the constructor. */
    zzs::kernel::device::driver::StreamBusInterface *_M_bus = nullptr;

    /** @brief Clock source callback for timestamps (may be nullptr). */
    uint64_t (*_M_clock) (void) = nullptr;

    /** @brief Latest usable fix (cached by `poll`). */
    Fix _M_fix = fix_make_invalid ();

    /** @brief Latest satellite table (cached by `poll`). */
    Satellite _M_satellites[GnssMaxSatellites] = {};
    uint8_t _M_satellite_count = 0u;

    /** @brief Receiver operational status. */
    GnssReceiverStatus _M_status = GnssReceiverStatus::Off;

    /** @brief Number of consecutive protocol errors. */
    uint8_t _M_consecutive_errors = 0u;

    /** @brief Handle counter for newly produced fixes. */
    uint8_t _M_next_handle = 1u;

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

    /** @brief Flush the receiver's receive buffer. */
    [[nodiscard]] inline zzs::kernel::Status
    bus_flush_rx () noexcept {
        if (_M_bus == nullptr)
            return zzs::kernel::Status::INTERNAL;
        return _M_bus->flush_rx ();
    }

    /** @brief Record a parser error and update `_M_status` when the threshold is
     *         exceeded. */
    inline void
    _M_record_error () noexcept {
        if (_M_consecutive_errors < GnssMaxConsecutiveErrors)
            _M_consecutive_errors += 1u;
        if (_M_consecutive_errors >= GnssMaxConsecutiveErrors)
            _M_status = GnssReceiverStatus::Error;
    }

    /** @brief Clear the error counter and restore tracking/fix status. */
    inline void
    _M_clear_errors () noexcept {
        _M_consecutive_errors = 0u;
        if (_M_status == GnssReceiverStatus::Error)
            _M_status = GnssReceiverStatus::Tracking;
    }

    /** @brief Publish a new fix: assign a handle, cache it, update status. */
    inline void
    _M_publish_fix (const Fix &fix) noexcept {
        Fix f = fix;
        if (f._M_handle == GnssInvalidHandle)
            f._M_handle = _M_next_handle++;
        _M_fix = f;
        _M_status = GnssReceiverStatus::Fix;
        _M_clear_errors ();
    }

    /** @brief Publish satellites from a parsed message. */
    inline void
    _M_publish_satellites (const Satellite *sats, uint8_t count) noexcept {
        if (count > GnssMaxSatellites)
            count = GnssMaxSatellites;
        for (uint8_t i = 0u; i < count; ++i)
            _M_satellites[i] = sats[i];
        _M_satellite_count = count;
    }
};

/** @} */
} // namespace zzs::kernel::device::driver::gnss
