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
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Monotonic clock source abstraction.
 *
 * The kernel aligns every sensor to a common 100 Hz grid and needs a monotonically
 * increasing nanosecond timebase that is not affected by wall-clock adjustments:
 *   - Linux/embedded: CLOCK_MONOTONIC_RAW
 *   - Android: SystemClock.elapsedRealtimeNanos()
 *   - iOS: mach_absolute_time()
 *   - host/tests: a software counter
 *
 * Each platform HAL implements `clock_source_now_ns`; the kernel itself never
 * calls the OS clock directly, which keeps the module testable and portable.
 * No heap, no float, no RTTI, no exceptions.
 * @{
 */
namespace zzs::kernel::clock {

/** @brief Nanosecond timestamp, monotonic, never reset by wall-clock changes. */
using Nanoseconds = uint64_t;

/** @brief Function pointer type for the platform clock source. */
using NowFn = Nanoseconds (*) (void) noexcept;

/**
 * @brief Return the current monotonic time in nanoseconds.
 *
 * The platform HAL registers its implementation through `clock_source_install`.
 * Until then this returns 0.
 */
[[nodiscard]] inline Nanoseconds
clock_source_now_ns (void) noexcept;

/** @brief Install the platform's clock source (called once at HAL init). */
[[nodiscard]] inline zzs::kernel::Status
clock_source_install (const NowFn now_fn) noexcept;

inline NowFn _S_now_fn = nullptr;

[[nodiscard]] inline Nanoseconds
clock_source_now_ns (void) noexcept {
  if (_S_now_fn != nullptr)
    return _S_now_fn ();
  return 0ull;
}

[[nodiscard]] inline zzs::kernel::Status
clock_source_install (const NowFn now_fn) noexcept {
  if (now_fn == nullptr) {
    zzs::base::log_error ("clock source install failed ({})", status_name (zzs::kernel::Status::OUT_OF_RANGE));
    return zzs::kernel::Status::OUT_OF_RANGE;
  }
  _S_now_fn = now_fn;
  zzs::base::log_info ("clock source installed");
  return zzs::kernel::Status::OK;
}
/** @} */
} // namespace zzs::kernel::clock
