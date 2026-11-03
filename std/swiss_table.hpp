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

#include <cstdint>
#include <utility>

namespace zzs::base {

/**
 * @brief Fixed-capacity open-addressing hash table in the style of SwissTable
 *        (Abseil flat_hash_map / CtrlHashMap).
 *
 * Designed for embedded use: no heap allocations, no writable static storage,
 * no exceptions, no RTTI, and all state lives inside the object instance.  The
 * capacity is a compile-time template parameter and must be a power of two and
 * a multiple of @c kGroupSize (16).
 *
 * Layout
 * ------
 * The table is divided into groups of @c kGroupSize slots.  Each slot owns one
 * control byte and one value:
 *
 *   - @c _M_ctrl[i] : control byte for slot @c i.
 *                     0   = empty slot
 *                     0xF = deleted ("tombstone") slot
 *                     1..0xE = hash suffix (h2) of the key stored in slot i
 *   - @C _M_entries[i] : the value stored in slot i (meaningful only when the
 *                        control byte is full).
 *
 * Lookup hashes the key, maps it to a group, and compares the 8-bit hash suffix
 * (h2) of the key against all 16 control bytes of the group in a single scan.
 * Matching candidates are then compared for full-key equality.  On a miss the
 * probe advances to the next group (linear probing over groups).
 *
 * Requirements on template arguments
 * ----------------------------------
 *   - Key must provide `uint64_t hash() const noexcept` and
 *     `bool operator==(const Key&) const noexcept`.
 *   - Value must be default-constructible and copyable/movable (trivially
 *     copyable is preferred for embedded).
 *
 * All operations are noexcept and bounded in time by the capacity.
 * @{
 */
namespace zzs::base {

/** @brief Sentinel returned by find() when a key is not present. */
template <uint32_t kCapacity>
inline constexpr uint32_t swiss_table_npos = kCapacity;

/** @brief Control byte meaning: slot is empty. */
inline constexpr uint8_t swiss_table_ctrl_empty = 0x00u;
/** @brief Control byte meaning: slot is deleted (tombstone). */
inline constexpr uint8_t swiss_table_ctrl_deleted = 0xF0u;
/** @brief Smallest valid hash suffix (h2) stored in a full slot. */
inline constexpr uint8_t swiss_table_ctrl_h2_min = 0x01u;
/** @brief Largest valid hash suffix (h2) stored in a full slot. */
inline constexpr uint8_t swiss_table_ctrl_h2_max = 0x0Eu;

/** @brief Number of slots per control group (must be a power of two). */
inline constexpr uint32_t swiss_table_group_size = 16u;

/**
 * @brief Compute the hash suffix (h2) stored in a control byte from a 64-bit
 *        hash.  The result is clamped to [1, 0xE] so that 0 (empty) and 0xF
 *        (deleted) are never confused with a real suffix.
 */
[[nodiscard]] inline constexpr uint8_t
swiss_table_h2 (const uint64_t hash) noexcept {
  uint8_t h = static_cast<uint8_t> (hash >> 56u); /* top 8 bits */
  if (h < swiss_table_ctrl_h2_min)
    h = swiss_table_ctrl_h2_min;
  if (h > swiss_table_ctrl_h2_max)
    h = swiss_table_ctrl_h2_max;
  return h;
}

/**
 * @brief Group index for a 64-bit hash within a table of `kCapacity` slots.
 */
template <uint32_t kCapacity>
[[nodiscard]] inline constexpr uint32_t
swiss_table_group_index (const uint64_t hash) noexcept {
  static_assert (kCapacity % swiss_table_group_size == 0,
                 "swiss_table capacity must be a multiple of the group size");
  return static_cast<uint32_t> (hash % (kCapacity / swiss_table_group_size));
}

/**
 * @brief Mask used to test which slots within a group match a given h2.
 *
 * Compares one control byte against all 16 control bytes of a group and
 * returns a 16-bit mask where bit i is set when the i-th control byte equals
 * `h` and is a full (non-empty, non-deleted) slot.
 */
[[nodiscard]] inline constexpr uint16_t
swiss_table_match_mask (const uint8_t *const group_ctrl, const uint8_t h) noexcept {
  uint16_t mask = 0u;
  for (uint32_t i = 0u; i < swiss_table_group_size; ++i) {
    const uint8_t c = group_ctrl[i];
    if (c != swiss_table_ctrl_empty && c != swiss_table_ctrl_deleted && c == h)
      mask |= static_cast<uint16_t> (1u << i);
  }
  return mask;
}

/**
 * @brief Index of the lowest set bit in a 16-bit mask, or 16 if none.
 */
[[nodiscard]] inline constexpr uint32_t
swiss_table_lowest_set_bit (const uint16_t mask) noexcept {
  for (uint32_t i = 0u; i < swiss_table_group_size; ++i)
    if ((mask >> i) & 1u)
      return i;
  return swiss_table_group_size;
}

/**
 * @brief SwissTable: fixed-capacity hash map with O(1) expected lookup.
 *
 * @tparam Key      key type (must expose hash() and operator==).
 * @tparam Value    mapped value type.
 * @tparam kCapacity total slot count (power of two, multiple of 16).
 */
template <typename Key, typename Value, uint32_t kCapacity> class SwissTable {
public:
  static_assert (kCapacity > 0, "swiss_table capacity must be positive");
  static_assert (kCapacity % swiss_table_group_size == 0,
                 "swiss_table capacity must be a multiple of the group size (16)");
  static_assert ((kCapacity & (kCapacity - 1u)) == 0,
                 "swiss_table capacity must be a power of two");

  using key_type = Key;
  using mapped_type = Value;
  using value_type = Value;
  using size_type = uint32_t;
  using reference = Value &;
  using const_reference = const Value &;
  using pointer = Value *;
  using const_pointer = const Value *;

  static constexpr size_type k_group_size = swiss_table_group_size;
  static constexpr size_type k_capacity = kCapacity;
  static constexpr size_type k_group_count = kCapacity / swiss_table_group_size;
  static constexpr size_type k_npos = swiss_table_npos<kCapacity>;

  /** @brief Default construct an empty table (all slots empty). */
  SwissTable () noexcept = default;

  SwissTable (const SwissTable &) = default;
  SwissTable (SwissTable &&) = default;
  SwissTable &
  operator= (const SwissTable &) = default;
  SwissTable &
  operator= (SwissTable &&) = default;
  ~SwissTable () noexcept = default;

  // -----------------------------------------------------------------------
  // Capacity
  // -----------------------------------------------------------------------

  /** @brief Number of occupied (non-empty, non-deleted) slots. */
  [[nodiscard]] constexpr size_type
  size () const noexcept {
    return this->_M_size;
  }

  /** @brief Total slot count (capacity). */
  [[nodiscard]] static constexpr size_type
  capacity () noexcept {
    return kCapacity;
  }

  /** @brief Whether no slots are occupied. */
  [[nodiscard]] constexpr bool
  empty () const noexcept {
    return this->_M_size == 0u;
  }

  /** @brief Whether the table is full (no empty or deleted slots remain). */
  [[nodiscard]] constexpr bool
  full () const noexcept {
    return this->_M_size == kCapacity;
  }

  // -----------------------------------------------------------------------
  // Lookup
  // -----------------------------------------------------------------------

  /** @brief Find the slot index of `key`, or k_npos if absent. */
  [[nodiscard]] constexpr size_type
  find (const Key &key) const noexcept {
    const uint64_t hash = key.hash ();
    const uint8_t h = swiss_table_h2 (hash);
    size_type group = swiss_table_group_index<kCapacity> (hash);

    for (size_type probe = 0u; probe < k_group_count; ++probe) {
      const uint8_t *const group_ctrl = this->_M_ctrl + group * k_group_size;
      const uint16_t mask = swiss_table_match_mask (group_ctrl, h);
      uint16_t m = mask;
      while (m != 0u) {
        const size_type offset = swiss_table_lowest_set_bit (m);
        m &= static_cast<uint16_t> (m - 1u); /* clear lowest set bit */
        const size_type index = group * k_group_size + offset;
        if (this->_M_entries[index].first == key)
          return index;
      }
      /* Advance to next group (linear probe over groups). */
      group = (group + 1u) % k_group_count;
    }
    return k_npos;
  }

  /** @brief Whether `key` is present in the table. */
  [[nodiscard]] constexpr bool
  contains (const Key &key) const noexcept {
    return this->find (key) != k_npos;
  }

  /** @brief Number of elements matching `key` (0 or 1 for a set-like map). */
  [[nodiscard]] constexpr size_type
  count (const Key &key) const noexcept {
    return this->contains (key) ? 1u : 0u;
  }

  // -----------------------------------------------------------------------
  // Indexed access (operator[])
  // -----------------------------------------------------------------------

  /** @brief Const access to the value stored at slot `index`.
   *
   * No bounds checking is performed; callers must ensure `index < capacity()`
   * and that the slot is occupied (use find() first).
   */
  [[nodiscard]] constexpr const_reference
  operator[] (const size_type index) const noexcept {
    return this->_M_entries[index].first;
  }

  /** @brief Mutable access to the value stored at slot `index`.
   *
   * No bounds checking is performed; callers must ensure `index < capacity()`
   * and that the slot is occupied (use find() first).
   */
  [[nodiscard]] constexpr reference
  operator[] (const size_type index) noexcept {
    return this->_M_entries[index].first;
  }

  /** @brief Const access to the key stored at slot `index`. */
  [[nodiscard]] constexpr const Key &
  key_at (const size_type index) const noexcept {
    return this->_M_entries[index].second;
  }

  /** @brief Mutable access to the key stored at slot `index`. */
  [[nodiscard]] constexpr Key &
  key_at (const size_type index) noexcept {
    return this->_M_entries[index].second;
  }

  // -----------------------------------------------------------------------
  // Insertion
  // -----------------------------------------------------------------------

  /** @brief Insert or assign `key` -> `value`.
   *
   * If `key` is already present, its value is overwritten.  Otherwise the key
   * is inserted into the first empty/deleted slot found while probing from the
   * key's home group.
   *
   * @returns Status::OK on success, OUT_OF_RANGE if the table is full and the
   *          key is not already present.
   */
  [[nodiscard]] constexpr int
  insert_or_assign (const Key &key, const Value &value) noexcept {
    const size_type existing = this->find (key);
    if (existing != k_npos) {
      this->_M_entries[existing].first = value;
      return 0; /* OK */
    }

    if (this->full ())
      return -1; /* OUT_OF_RANGE */

    const uint64_t hash = key.hash ();
    const uint8_t h = swiss_table_h2 (hash);
    size_type group = swiss_table_group_index<kCapacity> (hash);
    size_type first_deleted = k_npos;

    for (size_type probe = 0u; probe < k_group_count; ++probe) {
      const uint8_t *const group_ctrl = this->_M_ctrl + group * k_group_size;
      for (size_type i = 0u; i < k_group_size; ++i) {
        const size_type index = group * k_group_size + i;
        const uint8_t c = group_ctrl[i];
        if (c == swiss_table_ctrl_empty) {
          /* Prefer the earliest deleted slot if we saw one; otherwise use this
           * empty slot. */
          const size_type insert_at = (first_deleted != k_npos) ? first_deleted : index;
          this->_M_ctrl[insert_at] = h;
          this->_M_entries[insert_at].first = value;
          this->_M_entries[insert_at].second = key;
          this->_M_size++;
          return 0; /* OK */
        }
        if (c == swiss_table_ctrl_deleted && first_deleted == k_npos)
          first_deleted = index;
      }
      group = (group + 1u) % k_group_count;
    }

    /* Should be unreachable when !full(), but be safe. */
    return -1; /* OUT_OF_RANGE */
  }

  /** @brief Insert `key` -> `value` only if `key` is not already present.
   *
   * @returns Status::OK on successful insertion, 0 (already present) if the
   *          key exists (its value is left unchanged), OUT_OF_RANGE if full.
   */
  [[nodiscard]] constexpr int
  insert (const Key &key, const Value &value) noexcept {
    const size_type existing = this->find (key);
    if (existing != k_npos)
      return 0; /* already present */
    return this->insert_or_assign (key, value);
  }

  /** @brief Erase the slot holding `key`, converting it to a tombstone.
   *
   * @returns true if the key was found and erased, false if absent.
   */
  constexpr bool
  erase (const Key &key) noexcept {
    const size_type index = this->find (key);
    if (index == k_npos)
      return false;
    this->_M_ctrl[index] = swiss_table_ctrl_deleted;
    this->_M_entries[index] = Value{}; /* drop the value */
    this->_M_size--;
    return true;
  }

  /** @brief Erase the slot at `index` if it is occupied. */
  constexpr void
  erase_at (const size_type index) noexcept {
    if (index >= kCapacity)
      return;
    const uint8_t c = this->_M_ctrl[index];
    if (c == swiss_table_ctrl_empty || c == swiss_table_ctrl_deleted)
      return;
    this->_M_ctrl[index] = swiss_table_ctrl_deleted;
    this->_M_entries[index] = Value{};
    this->_M_size--;
  }

  /** @brief Remove all elements, returning every slot to the empty state. */
  constexpr void
  clear () noexcept {
    for (uint32_t i = 0u; i < kCapacity; ++i) {
      this->_M_ctrl[i] = swiss_table_ctrl_empty;
      this->_M_entries[i] = Value{};
    }
    this->_M_size = 0u;
  }

  // -----------------------------------------------------------------------
  // Iteration over occupied slots
  // -----------------------------------------------------------------------

  /** @brief Index of the first occupied slot, or k_npos if the table is empty. */
  [[nodiscard]] constexpr size_type
  first_occupied () const noexcept {
    for (size_type i = 0u; i < kCapacity; ++i) {
      const uint8_t c = this->_M_ctrl[i];
      if (c != swiss_table_ctrl_empty && c != swiss_table_ctrl_deleted)
        return i;
    }
    return k_npos;
  }

  /** @brief Index of the next occupied slot after `index`, or k_npos. */
  [[nodiscard]] constexpr size_type
  next_occupied (const size_type index) const noexcept {
    for (size_type i = index + 1u; i < kCapacity; ++i) {
      const uint8_t c = this->_M_ctrl[i];
      if (c != swiss_table_ctrl_empty && c != swiss_table_ctrl_deleted)
        return i;
    }
    return k_npos;
  }

private:
  /* Control bytes: one per slot.  0 = empty, 0xF = deleted, 1..0xE = h2. */
  uint8_t _M_ctrl[kCapacity] = {};
  /* Each slot stores (value, key).  Ordered (value first) so operator[] on the
   * slot yields the mapped value directly. */
  struct Slot {
    Value first;  /* mapped value */
    Key second;   /* key */
  };
  Slot _M_entries[kCapacity] = {};
  uint32_t _M_size = 0u;
};

/** @} */
} // namespace zzs::base

} // namespace zzs
