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

#include "kernel/kernel_config.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>

/**
 * @brief Fixed-capacity ring buffer of Q15 int16 samples.
 *
 * A stage (PLHR, vibration, flicker, acoustic) owns one ring per channel axis and
 * fills it one grid tick at a time.  When the ring reaches its window length the
 * stage drains a contiguous window into a caller buffer and keeps the remainder
 * for overlap.  Capacity is a power of two so the wrap mask is free.
 * @{
 */
namespace zzs::kernel::win {

/** @brief Ring buffer state; the caller owns the sample storage. */
struct _S_Ring {
        int16_t *_M_buf;      /* caller-owned storage of _M_capacity samples */
        uint32_t _M_capacity; /* power of two */
        uint32_t _M_write;    /* next write index */
        uint32_t _M_count;    /* samples currently stored */
};

/** @brief Construct a ring over caller-owned storage.  Capacity must be pow2. */
[[nodiscard]] inline zzs::kernel::Status
ring_init (_S_Ring &ring, int16_t *const buf, const uint32_t capacity) noexcept {
        if (buf == nullptr || capacity == 0u || (capacity & (capacity - 1u)) != 0u)
                return zzs::kernel::Status::OUT_OF_RANGE;
        ring._M_buf = buf;
        ring._M_capacity = capacity;
        ring._M_write = 0;
        ring._M_count = 0;
        return zzs::kernel::Status::OK;
}

/** @brief Push one sample; overwrites the oldest when full (drop-tail). */
inline void
ring_push (_S_Ring &ring, const int16_t sample_q15) noexcept {
        ring._M_buf[ring._M_write] = sample_q15;
        ring._M_write = (ring._M_write + 1u) & (ring._M_capacity - 1u);
        if (ring._M_count < ring._M_capacity)
                ++ring._M_count;
}

/** @brief Samples available for reading. */
[[nodiscard]] inline uint32_t
ring_count (const _S_Ring &ring) noexcept {
        return ring._M_count;
}

/** @brief Capacity of the ring. */
[[nodiscard]] inline uint32_t
ring_capacity (const _S_Ring &ring) noexcept {
        return ring._M_capacity;
}

/**
 * @brief Read `count` contiguous samples starting at logical index `start`
 *        (0 = oldest) into `out`, handling wrap.  Returns the number actually
 *        read (capped at available).  Caller must ensure out has `count` room.
 */
[[nodiscard]] inline uint32_t
ring_read (const _S_Ring &ring, const uint32_t start, int16_t *const out, const uint32_t count) noexcept {
        if (start >= ring._M_count || out == nullptr)
                return 0u;
        const uint32_t avail = ring._M_count - start;
        const uint32_t to_read = count < avail ? count : avail;
        const uint32_t begin = (ring._M_write - ring._M_count + start) & (ring._M_capacity - 1u);
        const uint32_t first = ring._M_capacity - begin;
        const uint32_t n_first = first < to_read ? first : to_read;
        for (uint32_t i = 0; i < n_first; ++i)
                out[i] = ring._M_buf[(begin + i) & (ring._M_capacity - 1u)];
        for (uint32_t i = n_first; i < to_read; ++i)
                out[i] = ring._M_buf[(begin + i) & (ring._M_capacity - 1u)];
        return to_read;
}

/** @brief Discard `n` oldest samples. */
inline void
ring_consume (_S_Ring &ring, const uint32_t n) noexcept {
        const uint32_t to_consume = n < ring._M_count ? n : ring._M_count;
        ring._M_count -= to_consume;
        /* write index is unchanged; logical oldest moves forward implicitly. */
}

/** @} */
} // namespace zzs::kernel::win
