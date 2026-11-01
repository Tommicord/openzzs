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

#include "kernel/sensor-driver/bus/bus_interface.hpp"

#include <cstdint>

/**
 * @brief Singleton registry of bus instances by device path/name.
 *
 * Platform code registers a concrete `BusInterface` under a device path
 * (e.g. "/dev/i2c-1", "/dev/spidev0.0") and sensor drivers look it up by
 * that path.  The registry owns no instances -- callers retain ownership --
 * it only stores non-owning pointers for lookup.
 *
 * No heap, no exceptions, no RTTI.
 * @{
 */
namespace zzs::kernel::sensor::driver::bus {

/** @brief Register a bus instance under `path`. Replaces any prior registration. */
[[nodiscard]] inline Status
bus_registry_register (const char *path, BusInterface *bus) noexcept;

/** @brief Look up a previously registered bus by `path`. Returns nullptr if absent. */
[[nodiscard]] inline BusInterface *
bus_registry_lookup (const char *path) noexcept;

/** @brief Remove the registration for `path`, if any. Returns true if removed. */
[[nodiscard]] inline bool
bus_registry_unregister (const char *path) noexcept;

/** @brief Fixed-capacity open-addressing map of path -> BusInterface *. */
struct _S_BusRegistryEntry {
  const char *_M_path = nullptr;
  BusInterface *_M_bus = nullptr;
};

inline constexpr uint32_t _S_bus_registry_capacity = 16u;

inline _S_BusRegistryEntry _S_bus_registry[_S_bus_registry_capacity];

/** @brief Simple FNV-1a hash of a NUL-terminated string. */
[[nodiscard]] inline constexpr uint32_t
_S_bus_hash (const char *str) noexcept {
  uint32_t h = 2166136261u;
  while (*str != '\0') {
    h ^= static_cast<uint32_t> (static_cast<unsigned char> (*str));
    h *= 16777619u;
    ++str;
  }
  return h;
}

[[nodiscard]] inline Status
bus_registry_register (const char *path, BusInterface *bus) noexcept {
  if (path == nullptr || bus == nullptr)
    return Status::OUT_OF_RANGE;
  uint32_t idx = _S_bus_hash (path) & (_S_bus_registry_capacity - 1u);
  uint32_t start = idx;
  for (;;) {
    if (_S_bus_registry[idx]._M_path == nullptr) {
      _S_bus_registry[idx]._M_path = path;
      _S_bus_registry[idx]._M_bus = bus;
      return Status::OK;
    }
    idx = (idx + 1u) & (_S_bus_registry_capacity - 1u);
    if (idx == start)
      return Status::BUDGET_EXCEEDED;
  }
}

[[nodiscard]] inline BusInterface *
bus_registry_lookup (const char *path) noexcept {
  if (path == nullptr)
    return nullptr;
  uint32_t idx = _S_bus_hash (path) & (_S_bus_registry_capacity - 1u);
  uint32_t start = idx;
  for (;;) {
    if (_S_bus_registry[idx]._M_path == nullptr)
      return nullptr;
    if (_S_bus_registry[idx]._M_path == path)
      return _S_bus_registry[idx]._M_bus;
    idx = (idx + 1u) & (_S_bus_registry_capacity - 1u);
    if (idx == start)
      return nullptr;
  }
}

[[nodiscard]] inline bool
bus_registry_unregister (const char *path) noexcept {
  if (path == nullptr)
    return false;
  uint32_t idx = _S_bus_hash (path) & (_S_bus_registry_capacity - 1u);
  uint32_t start = idx;
  for (;;) {
    if (_S_bus_registry[idx]._M_path == nullptr)
      return false;
    if (_S_bus_registry[idx]._M_path == path) {
      _S_bus_registry[idx]._M_path = nullptr;
      _S_bus_registry[idx]._M_bus = nullptr;
      return true;
    }
    idx = (idx + 1u) & (_S_bus_registry_capacity - 1u);
    if (idx == start)
      return false;
  }
}

/** @} */
} // namespace zzs::kernel::sensor::driver::bus
