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

/**
 * @brief Model type identifier: the first 4 bytes of a VTable entry.
 *
 * Every entry in the model VTable begins with a 4-byte type tag so the blob
 * can be dispatched by kind rather than assumed to be a single hardcoded
 * outage-predictor blob.
 * @{
 */
namespace zzs::mllee_native::model {

/** @brief Model type codes. Stored as the first 4 bytes of each VTable entry. */
enum class ModelType : uint32_t {
  kNone = 0x0000u,       /*< No model / invalid entry */
  kOutPred = 0x0008u,    /*< OUTPRED: outage predictor (the original MLP blob) */
  kConvAttn = 0x0010u,   /*< Tier 2 CONV1D + Attention blob */
  kMultiTask = 0x0018u,  /*< Tier 3 multi-task blob */
};

/** @brief Opaque lookup key for the model VTable/SwissTable.
 *
 * Wraps a 64-bit hash of a ModelType (and, in future, version/attributes).
 */
class ModelKey {
public:
  explicit constexpr ModelKey (ModelType type) noexcept : _M_hash (static_cast<uint64_t> (type)) {}
  explicit constexpr ModelKey (uint64_t hash) noexcept : _M_hash (hash) {}

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

/** @} */
} // namespace zzs::mllee_native::model
