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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#pragma once

#include "kernel/device-driver/builtins/zzs_builtins.hpp"
#include "kernel/kernel_config.hpp"

#include <cstdint>

/**
 * @brief Fixed-capacity ring buffer of Q15 int16 samples.
 *
 * A stage (PLHR, vibration, flicker, acoustic) owns one ring per channel axis and
 * fills it one grid tick at a time.  When the ring reaches its window length the
 * stage drains a contiguous window into a caller buffer and keeps the remainder
 * for overlap.  Capacity is a power of two so the wrap mask is free.
 *
 * The ring owns its sample storage as a fixed member array (RAII): nothing is
 * allocated, nothing is copied, and the capacity is a compile-time configuration
 * expressed as a template argument.  Copy/move are deleted because a ring's
 * read/write position is meaningful state that must not be duplicated.
 * @{
 */
namespace zzs::kernel::win {

/** @brief Ring buffer of `Capacity` Q15 samples.  `Capacity` must be pow2. */
template <uint32_t Capacity> class Ring {
    static_assert (Capacity != 0u && (Capacity & (Capacity - 1u)) == 0u,
                   "Ring capacity must be a non-zero power of two");

    int16_t _M_buf[Capacity]{}; /* owned sample storage */
    uint32_t _M_write = 0u;     /* next write index */
    uint32_t _M_count = 0u;     /* samples currently stored */

  public:
    Ring () = default;
    Ring (const Ring &) = delete;
    Ring &
    operator= (const Ring &) = delete;
    Ring (Ring &&) = delete;
    Ring &
    operator= (Ring &&) = delete;

    /** @brief Capacity of the ring (samples). */
    [[nodiscard]] inline uint32_t
    capacity () const noexcept {
        return Capacity;
    }

    /** @brief Samples available for reading. */
    [[nodiscard]] inline uint32_t
    count () const noexcept {
        return _M_count;
    }

    /** @brief True when the ring holds a full capacity of samples. */
    [[nodiscard]] inline bool
    full () const noexcept {
        return _M_count == Capacity;
    }

    /** @brief True when the ring is empty. */
    [[nodiscard]] inline bool
    empty () const noexcept {
        return _M_count == 0u;
    }

    /** @brief Push one sample; overwrites the oldest when full (drop-tail). */
    inline void
    push (const int16_t sample_q15) noexcept {
        _M_buf[_M_write] = sample_q15;
        _M_write = (_M_write + 1u) & (Capacity - 1u);
        if (ZZS_LIKELY (_M_count < Capacity))
            ++_M_count;
    }

    /**
     * @brief Read `count` contiguous samples starting at logical index `start`
     *        (0 = oldest) into `out`, handling wrap.  Returns the number actually
     *        read (capped at available).  `out` must hold `count` samples.
     *
     *  The two wrapped segments are copied with `ZZS_MEMCPY` instead of hand
     *  rolled byte loops, and the wrap mask is a free power-of-two AND.
     */
    [[nodiscard]] inline uint32_t
    read (const uint32_t start, int16_t *const out, const uint32_t count) noexcept {
        if (ZZS_UNLIKELY (start >= _M_count || out == nullptr))
            return 0u;
        const uint32_t avail = _M_count - start;
        const uint32_t to_read = count < avail ? count : avail;
        const uint32_t mask = Capacity - 1u;
        const uint32_t begin = (_M_write - _M_count + start) & mask;
        const uint32_t first = Capacity - begin; /* samples from `begin` to end of buffer */
        const uint32_t n_first = first < to_read ? first : to_read;

        ZZS_MEMCPY (out, &_M_buf[begin], n_first * sizeof (int16_t));
        if (ZZS_LIKELY (n_first < to_read)) {
            /* Prefetch the line the wrap segment will touch. */
            ZZS_PREFETCH (&_M_buf[0], 0, 0);
            ZZS_MEMCPY (out + n_first, &_M_buf[0], (to_read - n_first) * sizeof (int16_t));
        }
        return to_read;
    }

    /** @brief Discard `n` oldest samples. */
    inline void
    consume (const uint32_t n) noexcept {
        const uint32_t to_consume = n < _M_count ? n : _M_count;
        _M_count -= to_consume;
    }
};

/** @} */
} // namespace zzs::kernel::win
