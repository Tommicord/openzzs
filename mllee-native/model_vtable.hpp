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

#include "std/swiss_table.hpp"

#include <cstdint>

/**
 * @brief Model VTable: a fixed-capacity collection of model blobs indexed by a
 *        SwissTable keyed on ModelKey.
 *
 * Instead of a single hardcoded model blob, the VTable holds many model blobs.
 * Each entry begins with a 4-byte type tag (e.g. 0x0008 = OUTPRED).  Entries
 * are looked up by key through an embedded SwissTable that maps a ModelKey to
 * a slot index; the entry itself is then read with `operator[]` on that index.
 *
 * All state lives inside the object; no heap, no writable statics, no
 * exceptions.  Blob pointers reference .rodata and are never copied.
 * @{
 */
namespace zzs::mllee_native::model {

using zzs::base::SwissTable;

/** @brief Maximum number of entries the VTable can hold. */
inline constexpr uint32_t kMaxModelEntries = 16u;

/** @brief SwissTable slot count: power of two, multiple of 16. */
inline constexpr uint32_t kModelVTableIndexCapacity = 32u;

/** @brief Model type codes. Stored as the first 4 bytes of each VTable entry. */
enum class ModelType : uint32_t {
    kNone = 0x0000u,      /*< No model / invalid entry */
    kOutPred = 0x0008u,   /*< OUTPRED: outage predictor (the original MLP blob) */
    kConvAttn = 0x0010u,  /*< Tier 2 CONV1D + Attention blob */
    kMultiTask = 0x0018u, /*< Tier 3 multi-task blob */
};

/** @brief Opaque lookup key for the model VTable/SwissTable.
 *
 * Wraps a 64-bit hash of a ModelType (and, in future, version/attributes).
 */
class ModelKey {
  public:
    constexpr ModelKey () noexcept : _M_hash (0u) {
    } /* kNone / empty sentinel */
    explicit constexpr ModelKey (ModelType type) noexcept : _M_hash (static_cast<uint64_t> (type)) {
    }
    explicit constexpr ModelKey (uint64_t hash) noexcept : _M_hash (hash) {
    }

    [[nodiscard]] constexpr uint64_t
    hash () const noexcept {
        return this->_M_hash;
    }

    [[nodiscard]] constexpr bool
    operator== (const ModelKey &other) const noexcept {
        return this->_M_hash == other._M_hash;
    }

    [[nodiscard]] constexpr bool
    operator!= (const ModelKey &other) const noexcept {
        return this->_M_hash != other._M_hash;
    }

  private:
    uint64_t _M_hash;
};

/** @brief A single model-blob entry in the VTable. */
struct VTableEntry {
    /** @brief First 4 bytes: model type tag (e.g. 0x0008 = OUTPRED). */
    uint32_t _M_type;
    /** @brief Size of the referenced model blob, in bytes. */
    uint32_t _M_blob_size;
    /** @brief Pointer to the model blob (resides in .rodata). */
    const uint8_t *_M_blob;

    VTableEntry () noexcept : _M_type (static_cast<uint32_t> (ModelType::kNone)), _M_blob_size (0u), _M_blob (nullptr) {
    }

    constexpr VTableEntry (const ModelType type, const uint32_t blob_size, const uint8_t *const blob) noexcept
        : _M_type (static_cast<uint32_t> (type)), _M_blob_size (blob_size), _M_blob (blob) {
    }

    [[nodiscard]] constexpr bool
    is_valid () const noexcept {
        return this->_M_blob != nullptr && this->_M_blob_size > 0u;
    }

    [[nodiscard]] constexpr ModelType
    type () const noexcept {
        return static_cast<ModelType> (this->_M_type);
    }
};

/**
 * @brief OOP model VTable container.
 *
 * Owns a fixed array of `VTableEntry` and a SwissTable mapping
 * `ModelKey` -> slot index.  Instantiate with the desired capacity (defaults to
 * `kMaxModelEntries` entries / `kModelVTableIndexCapacity` index slots).
 *
 * @tparam kMaxEntries   max entries in the table.
 * @tparam kIndexCapacity SwissTable capacity (power of two, multiple of 16).
 */
template <uint32_t kMaxEntries = kMaxModelEntries, uint32_t kIndexCapacity = kModelVTableIndexCapacity>
class ModelVTable {
  public:
    static_assert (kMaxEntries > 0, "model vtable must hold at least one entry");
    static_assert (kIndexCapacity % 16u == 0, "model vtable index capacity must be a multiple of 16");
    static_assert ((kIndexCapacity & (kIndexCapacity - 1u)) == 0, "model vtable index capacity must be a power of two");
    static_assert (kMaxEntries <= kIndexCapacity, "model vtable index capacity must be >= max entries");

    using entry_type = VTableEntry;
    using key_type = ModelKey;
    using index_type = uint32_t;
    using size_type = uint32_t;

    static constexpr size_type k_max_entries = kMaxEntries;
    static constexpr size_type k_index_capacity = kIndexCapacity;
    static constexpr index_type k_npos = kIndexCapacity; /* not-found sentinel */

    ModelVTable () noexcept = default;
    ModelVTable (const ModelVTable &) = delete;
    ModelVTable &
    operator= (const ModelVTable &) = delete;
    ModelVTable (ModelVTable &&) = delete;
    ModelVTable &
    operator= (ModelVTable &&) = delete;
    ~ModelVTable () noexcept = default;

    // -----------------------------------------------------------------------
    // Capacity
    // -----------------------------------------------------------------------

    [[nodiscard]] constexpr size_type
    size () const noexcept {
        return this->_M_index.size ();
    }

    [[nodiscard]] constexpr size_type
    capacity () const noexcept {
        return kMaxEntries;
    }

    [[nodiscard]] constexpr bool
    empty () const noexcept {
        return this->_M_index.empty ();
    }

    [[nodiscard]] constexpr bool
    full () const noexcept {
        return this->_M_index.size () == kMaxEntries;
    }

    // -----------------------------------------------------------------------
    // Lookup by key -> index
    // -----------------------------------------------------------------------

    /** @brief Find the slot index for `key`, or k_npos if absent. */
    [[nodiscard]] constexpr index_type
    find (const ModelKey &key) const noexcept {
        return this->_M_index.find (key);
    }

    /** @brief Whether `key` is present in the table. */
    [[nodiscard]] constexpr bool
    contains (const ModelKey &key) const noexcept {
        return this->_M_index.contains (key);
    }

    /** @brief Number of entries matching `key` (0 or 1). */
    [[nodiscard]] constexpr size_type
    count (const ModelKey &key) const noexcept {
        return this->_M_index.count (key);
    }

    // -----------------------------------------------------------------------
    // Indexed access (operator[]) -- the preferred entry accessor
    // -----------------------------------------------------------------------

    /** @brief Const access to the entry at slot `index`.
     *
     * No bounds checking; callers must ensure `index < capacity()` and that the
     * slot is occupied (use find()/contains() first).
     */
    [[nodiscard]] constexpr const entry_type &
    operator[] (const size_type index) const noexcept {
        return this->_M_entries[index];
    }

    /** @brief Mutable access to the entry at slot `index`. */
    [[nodiscard]] constexpr entry_type &
    operator[] (const size_type index) noexcept {
        return this->_M_entries[index];
    }

    /** @brief Const access to the entry's type tag at slot `index`. */
    [[nodiscard]] constexpr uint32_t
    type_at (const size_type index) const noexcept {
        return this->_M_entries[index]._M_type;
    }

    /** @brief Const access to the entry's blob pointer at slot `index`. */
    [[nodiscard]] constexpr const uint8_t *
    blob_at (const size_type index) const noexcept {
        return this->_M_entries[index]._M_blob;
    }

    /** @brief Const access to the entry's blob size at slot `index`. */
    [[nodiscard]] constexpr uint32_t
    blob_size_at (const size_type index) const noexcept {
        return this->_M_entries[index]._M_blob_size;
    }

    // -----------------------------------------------------------------------
    // Insertion
    // -----------------------------------------------------------------------

    /** @brief Insert or replace the entry for `key`.
     *
     * If `key` already present, its entry is overwritten.  Otherwise a new slot
     * is allocated.  Returns the slot index on success, or k_npos if the table is
     * full.
     */
    [[nodiscard]] constexpr index_type
    insert (const ModelKey &key, const entry_type &entry) noexcept {
        const index_type existing = this->_M_index.find (key);
        if (existing != k_npos) {
            this->_M_entries[existing] = entry;
            return existing;
        }
        if (this->full ())
            return k_npos;
        /* Append at the next free slot (linear scan for the first empty slot). */
        index_type slot = 0u;
        for (; slot < kMaxEntries; ++slot) {
            if (!this->_M_entries[slot].is_valid ())
                break;
        }
        this->_M_entries[slot] = entry;
        static_cast<void> (this->_M_index.insert_or_assign (key, slot));
        return slot;
    }

    /** @brief Remove the entry for `key` (leaves a tombstone in the index). */
    constexpr bool
    erase (const ModelKey &key) noexcept {
        const index_type slot = this->_M_index.find (key);
        if (slot == k_npos)
            return false;
        this->_M_entries[slot] = entry_type{};
        this->_M_index.erase (key);
        return true;
    }

    /** @brief Remove all entries. */
    constexpr void
    clear () noexcept {
        for (size_type i = 0u; i < kMaxEntries; ++i)
            this->_M_entries[i] = entry_type{};
        this->_M_index.clear ();
    }

    // -----------------------------------------------------------------------
    // Iteration
    // -----------------------------------------------------------------------

    [[nodiscard]] constexpr index_type
    first_occupied () const noexcept {
        return this->_M_index.first_occupied ();
    }

    [[nodiscard]] constexpr index_type
    next_occupied (const index_type index) const noexcept {
        return this->_M_index.next_occupied (index);
    }

  private:
    entry_type _M_entries[kMaxEntries] = {};
    SwissTable<ModelKey, index_type, kIndexCapacity> _M_index{};
};

/** @} */
} // namespace zzs::mllee_native::model
