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

#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_status.hpp"
#include "kernel/sensor-driver/sensors/sensor_driver.hpp"

#include <cstdint>
#include <cstring>

/**
 * @brief Streaming sensor-driver abstraction (lower level than the platform HAL).
 *
 * A single bus burst read produces one timestamped scan frame covering all
 * enabled axes of a sensor at the same instant.  A fixed-size ring buffer
 * (ScanRing) decouples the producer (ISR / HAL read) from the consumer
 * (preprocessing bank) without any heap allocation, RTTI, std::string,
 * std::vector, std::array or std::function.  All buffers are fixed C arrays
 * sized at compile time; every method is `noexcept` and integer-only.
 * @{
 */
namespace zzs::kernel::sensor::driver::stream {

/** @brief Maximum number of channels carried in one scan frame. */
inline constexpr uint32_t kMaxScanChannels = 8u;

/** @brief Default depth of the fixed scan ring (power of two for a free mask). */
inline constexpr uint32_t kDefaultRingDepth = 16u;

/** @brief One timestamped scan frame: all channels sampled at the same instant. */
struct ScanFrame {
    /** @brief Monotonic nanosecond timestamp captured after the bus burst. */
    uint64_t _M_timestamp_ns = 0ull;
    /** @brief Q15 sample values, one per channel. */
    int16_t _M_channels[kMaxScanChannels] = {0};
    /** @brief Number of valid channels in `_M_channels`. */
    uint8_t _M_channel_count = 0u;
};

/** @brief Streaming configuration: ring depth and consumer watermark. */
struct StreamConfig {
    /** @brief Fixed ring depth (must be a power of two). */
    uint16_t _M_depth = static_cast<uint16_t> (kDefaultRingDepth);
    /** @brief Consumer is notified/wakes when `filled >= _M_watermark`. */
    uint16_t _M_watermark = 1u;
};

/**
 * @brief Fixed-size ring of timestamped sensor samples.
 *
 * Producer pushes samples (from `read_samples()` burst fills); consumer pops
 * them.  Overwrites the oldest sample when full (drop-oldest policy), so the
 * ring never blocks and never allocates.  Depth is fixed at construction and
 * stored in a C array.
 * @{
 */
class ScanRing {
  public:
    /** @brief Construct with a fixed depth (clamped to [1, kMaxRingDepth]). */
    explicit ScanRing (uint16_t depth) noexcept : _M_depth (clamp_depth (depth)), _M_watermark (1u) {
        _M_clear ();
    }

    ScanRing (const ScanRing &) = delete;
    ScanRing &
    operator= (const ScanRing &) = delete;
    ScanRing (ScanRing &&) = delete;
    ScanRing &
    operator= (ScanRing &&) = delete;

    /** @brief Push one sample; overwrites the oldest when full. */
    [[nodiscard]] zzs::kernel::Status
    push (const sensor::Sample &sample) noexcept {
        _M_buf[_M_head] = sample;
        _M_head = static_cast<uint16_t> ((_M_head + 1u) % _M_depth);
        if (_M_count < _M_depth)
            _M_count += 1u;
        else
            _M_tail = static_cast<uint16_t> ((_M_tail + 1u) % _M_depth);
        return zzs::kernel::Status::OK;
    }

    /** @brief Pop the oldest sample into `out`. Returns STALE when empty. */
    [[nodiscard]] zzs::kernel::Status
    pop (sensor::Sample &out) noexcept {
        if (_M_count == 0u)
            return zzs::kernel::Status::STALE;
        out = _M_buf[_M_tail];
        _M_tail = static_cast<uint16_t> ((_M_tail + 1u) % _M_depth);
        _M_count -= 1u;
        return zzs::kernel::Status::OK;
    }

    /** @brief Number of samples currently in the ring. */
    [[nodiscard]] uint16_t
    filled () const noexcept {
        return _M_count;
    }

    [[nodiscard]] bool
    empty () const noexcept {
        return _M_count == 0u;
    }

    [[nodiscard]] bool
    full () const noexcept {
        return _M_count == _M_depth;
    }

    /** @brief Set the consumer watermark (1..depth). */
    void
    set_watermark (uint16_t watermark) noexcept {
        _M_watermark = (watermark == 0u || watermark > _M_depth) ? _M_depth : watermark;
    }

    /** @brief True when `filled() >= watermark`. */
    [[nodiscard]] bool
    watermark_reached () const noexcept {
        return _M_count >= _M_watermark;
    }

    /** @brief Reset the ring to empty. */
    void
    clear () noexcept {
        _M_clear ();
    }

  private:
    [[nodiscard]] static uint16_t
    clamp_depth (uint16_t depth) noexcept {
        if (depth == 0u)
            return 1u;
        if (depth > kMaxRingDepth)
            return kMaxRingDepth;
        return depth;
    }

    void
    _M_clear () noexcept {
        _M_head = 0u;
        _M_tail = 0u;
        _M_count = 0u;
        for (uint16_t i = 0u; i < kMaxRingDepth; ++i)
            _M_buf[i] = sensor::Sample{};
    }

    static constexpr uint16_t kMaxRingDepth = 64u;

    sensor::Sample _M_buf[kMaxRingDepth] = {};
    uint16_t _M_depth = 1u;
    uint16_t _M_head = 0u;
    uint16_t _M_tail = 0u;
    uint16_t _M_count = 0u;
    uint16_t _M_watermark = 1u;
};
/** @} */

/**
 * @brief Streaming sensor driver: fills a caller-provided sample buffer in one
 *        call, bursting multiple axes from a single bus transaction where the
 *        hardware supports it.
 *
 * Concrete drivers override `read_samples` to issue a single burst register
 * read (one chip-select / I2C address phase) and convert the block into several
 * `sensor::Sample` values sharing one timestamp.  The base class default
 * implementation loops `read_sample()` so existing drivers keep working.
 * @{
 */
class StreamingSensorDriver : public sensors::SensorDriver {
  public:
    /** @brief Construct with a non-owning bus pointer and optional clock. */
    explicit StreamingSensorDriver (bus::BusInterface *bus, uint64_t (*clock) (void) = nullptr) noexcept
        : sensors::SensorDriver (bus, clock) {
    }

    /** @brief Virtual destructor. */
    ~StreamingSensorDriver () override = default;

    StreamingSensorDriver (const StreamingSensorDriver &) = delete;
    StreamingSensorDriver &
    operator= (const StreamingSensorDriver &) = delete;
    StreamingSensorDriver (StreamingSensorDriver &&) = delete;
    StreamingSensorDriver &
    operator= (StreamingSensorDriver &&) = delete;

    /**
     * @brief Read up to `max_samples` into `out`; `*filled` is set to the
     *        number of samples written.  Returns OK when at least one sample
     *        was produced, STALE when none are ready, NOT_READY while warming
     *        up, or a bus error.
     *
     * The default implementation loops `read_sample()`; override it to burst
     * multiple axes from one bus transaction.
     */
    [[nodiscard]] virtual zzs::kernel::Status
    read_samples (sensor::Sample *out, uint16_t max_samples, uint16_t *filled) noexcept {
        if (out == nullptr || filled == nullptr)
            return zzs::kernel::Status::OUT_OF_RANGE;
        if (max_samples == 0u) {
            *filled = 0u;
            return zzs::kernel::Status::OK;
        }
        uint16_t n = 0u;
        while (n < max_samples) {
            sensor::Sample s;
            zzs::kernel::Status st = read_sample (s);
            if (!zzs::kernel::status_ok (st)) {
                if (st == zzs::kernel::Status::STALE || st == zzs::kernel::Status::NOT_READY)
                    break;
                return st;
            }
            out[n++] = s;
        }
        *filled = n;
        return (n > 0u) ? zzs::kernel::Status::OK : zzs::kernel::Status::STALE;
    }

  protected:
    /** @brief Burst-read `len` consecutive registers from `reg` into `buf`. */
    [[nodiscard]] inline zzs::kernel::Status
    bus_read_block (uint8_t reg, uint8_t *buf, uint16_t len) const noexcept {
        return _M_bus == nullptr ? zzs::kernel::Status::INTERNAL : _M_bus->read_reg (reg, buf, len);
    }
};
/** @} */

/** @} */
} // namespace zzs::kernel::sensor::driver::stream
