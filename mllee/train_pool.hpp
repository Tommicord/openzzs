/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#pragma once

#include "mllee/status.hpp"

#include <cstdint>

/**
 * @brief Caller-owned fixed-size storage for the training path.
 *
 * Training needs more RAM than the 1024-byte inference module: a gradient per
 * element, optimizer moments, and the tape.  None of it may come from a heap,
 * and none of it may be a writable module static, so every training object lives
 * in a Pool carved out of a buffer the application owns (a linker-placed .bss
 * array, a DTCM region, or a stack frame the caller can measure).
 *
 * The allocator is a deterministic bump pointer: allocation order is the only
 * scheduling decision, it never fragments, and the used/peak word counters make
 * a capacity regression visible instead of silent.  A request that does not fit
 * returns Status::ARENA_OVERFLOW and leaves the pool untouched.
 */
namespace zzs::mllee::train
{
/** @brief Alignment every training buffer is carved at, i.e. one SIMD lane. */
inline constexpr uint32_t _S_pool_alignment = 4u;

/**
 * @brief Default training arena size.
 *
 * This is the training-mode budget and is reported next to the inference
 * budget by tools/check_mllee_train_ram.sh; it does not change the 1024-byte
 * inference module gate.
 */
inline constexpr uint32_t _S_train_pool_bytes = 8192u;

/** @brief One carved region: a halfword offset into the pool and its length. */
struct Slab
{
    uint32_t _M_offset = 0u;
    uint32_t _M_bytes = 0u;

    [[nodiscard]] constexpr bool
    is_null () const noexcept
    {
        return this->_M_bytes == 0u;
    }
};

/** @brief Counter snapshot for the RAM gate and for telemetry. */
struct PoolStats
{
    uint32_t _M_capacity = 0u;
    uint32_t _M_used = 0u;
    uint32_t _M_peak = 0u;
    uint32_t _M_slabs = 0u;
    uint32_t _M_overflows = 0u;
};

/**
 * @brief Bump allocator over one caller-owned buffer.
 *
 * Every block is aligned to _S_pool_alignment so a kernel can assume a 4-byte
 * aligned, stride-1 stream: the SIMD kernels in mllee/train_simd.hpp branch on
 * alignment once per call, exactly like the inference kernels do, and a
 * misaligned carve would silently cost the whole vectorized win.
 */
class Pool
{
  public:
    Pool (uint8_t *const backing, const uint32_t bytes) noexcept
        : _M_backing (backing),
          _M_capacity (bytes - (bytes % _S_pool_alignment)),
          _M_used (0u), _M_peak (0u), _M_slabs (0u), _M_overflows (0u)
    {
    }

    Pool (const Pool &) = delete;
    Pool &operator= (const Pool &) = delete;

    /** @brief Carve @p count elements of @p element_bytes from the pool. */
    [[nodiscard]] Status
    carve (const uint32_t count, const uint32_t element_bytes, Slab &out) noexcept
    {
        const uint64_t requested
            = static_cast<uint64_t> (count) * element_bytes;
        if (requested > 0xFFFFFFFFull)
            return Status::OUT_OF_RANGE;
        const uint32_t aligned
            = align_up (static_cast<uint32_t> (requested), _S_pool_alignment);
        const uint32_t start = align_up (this->_M_used, _S_pool_alignment);
        if (start + aligned > this->_M_capacity)
            {
                ++this->_M_overflows;
                return Status::ARENA_OVERFLOW;
            }
        out._M_offset = start;
        out._M_bytes = aligned;
        this->_M_used = start + aligned;
        if (this->_M_used > this->_M_peak)
            this->_M_peak = this->_M_used;
        ++this->_M_slabs;
        return Status::OK;
    }

    /** @brief Carve and zero @p count elements, returning a typed pointer. */
    template <typename T>
    [[nodiscard]] Status
    carve_typed (const uint32_t count, T *&out) noexcept
    {
        Slab slab;
        const Status status
            = this->carve (count, static_cast<uint32_t> (sizeof (T)), slab);
        if (!status_ok (status))
            {
                out = nullptr;
                return status;
            }
        out = reinterpret_cast<T *> (this->_M_backing + slab._M_offset);
        return this->zero (out, count);
    }

    /** @brief Zero @p count elements of a pool-carved array. */
    template <typename T>
    [[nodiscard]] static Status
    zero (T *const values, const uint32_t count) noexcept
    {
        if (values == nullptr && count != 0u)
            return Status::INTERNAL;
        for (uint32_t i = 0; i < count; ++i)
            values[i] = T{};
        return Status::OK;
    }

    /** @brief Pointer to the raw byte at @p offset, or null when out of range. */
    [[nodiscard]] uint8_t *
    at (const uint32_t offset) noexcept
    {
        return offset <= this->_M_capacity ? this->_M_backing + offset : nullptr;
    }

    [[nodiscard]] const uint8_t *
    at (const uint32_t offset) const noexcept
    {
        return offset <= this->_M_capacity ? this->_M_backing + offset : nullptr;
    }

    /** @brief Typed view of a previously carved slab. */
    template <typename T>
    [[nodiscard]] T *
    slab_view (const Slab slab) noexcept
    {
        if (slab._M_bytes % sizeof (T) != 0u)
            return nullptr;
        return reinterpret_cast<T *> (this->_M_backing + slab._M_offset);
    }

    /** @brief Return every slab: the tape and the trainer reset between passes. */
    void
    reset () noexcept
    {
        this->_M_used = 0u;
        this->_M_slabs = 0u;
    }

    [[nodiscard]] PoolStats
    stats () const noexcept
    {
        return PoolStats{ this->_M_capacity, this->_M_used, this->_M_peak,
                          this->_M_slabs, this->_M_overflows };
    }

    [[nodiscard]] uint32_t
    remaining () const noexcept
    {
        const uint32_t used = align_up (this->_M_used, _S_pool_alignment);
        return used >= this->_M_capacity ? 0u : this->_M_capacity - used;
    }

    [[nodiscard]] uint8_t *
    backing () noexcept
    {
        return this->_M_backing;
    }

  private:
    /** @brief Round @p value up to the next multiple of @p alignment. */
    [[nodiscard]] static constexpr uint32_t
    align_up (const uint32_t value, const uint32_t alignment) noexcept
    {
        return (value + alignment - 1u) & ~(alignment - 1u);
    }

    uint8_t *_M_backing;
    uint32_t _M_capacity;
    uint32_t _M_used;
    uint32_t _M_peak;
    uint32_t _M_slabs;
    uint32_t _M_overflows;
};

/**
 * @brief Storage helper: a pool-sized array with the alignment a kernel needs.
 *
 * Applications place this at file scope (it is writable state they own and
 * measure) or on a stack frame they have sized; tests use it to keep the arena
 * off the default 8 MiB stack assumptions of the host harness.
 */
template <uint32_t Bytes>
struct AlignedArena
{
    static_assert (Bytes % _S_pool_alignment == 0u,
                   "training arena size must be a multiple of the SIMD lane width");

    alignas (_S_pool_alignment) uint8_t _M_data[Bytes];

    [[nodiscard]] Pool
    pool () noexcept
    {
        return Pool (this->_M_data, Bytes);
    }
};

/** @brief True when @p pointer is usable by the 4-byte SIMD training kernels. */
[[nodiscard]] inline bool
simd_aligned (const void *const pointer) noexcept
{
    return (reinterpret_cast<uintptr_t> (pointer) & (_S_pool_alignment - 1u)) == 0u;
}
} // namespace zzs::mllee::train
