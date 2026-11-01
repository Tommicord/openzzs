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
#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_status.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Sensor HAL interface.
 *
 * The platform HAL (Android/iOS/Linux-IIO/STM32-Zephyr) implements this vtable
 * and registers it with the preprocessing bank.  The bank calls `read_sample`
 * once per available hardware event; the HAL fills a `sensor::Sample` (Q15
 * int16, channel id, monotonic ns timestamp) and returns Status::OK, STALE when
 * nothing new is available, or NOT_READY when the channel is warming up.
 *
 * The HAL is also responsible for the platform clock source
 * (`clock_source_install` from kernel_clock_source.hpp).  No heap, no float,
 * no RTTI, no exceptions.
 * @{
 */
namespace zzs::kernel::hal {

/** @brief Vtable implemented by each platform HAL. */
struct _S_SensorHal {
  /** @brief Read the next available sample, or return STALE/NOT_READY. */
  zzs::kernel::Status (*_M_read_sample) (sensor::Sample &out) noexcept;

  /** @brief Enable/disable a channel. */
  zzs::kernel::Status (*_M_set_channel_enabled) (const sensor::Channel channel, const bool enabled) noexcept;

  /** @brief Return the channel's configured sample rate in Hz. */
  uint32_t (*_M_channel_rate_hz) (const sensor::Channel channel) noexcept;

  /** @brief Called once at bank init; may install the clock source. */
  zzs::kernel::Status (*_M_init) (void) noexcept;
};

/** @brief The installed HAL; null until a platform registers. */
inline const _S_SensorHal *_S_hal = nullptr;
inline bool _S_installed = false;

/** @brief Query whether a HAL has been installed. */
[[nodiscard]] inline bool
hal_is_installed (void) noexcept {
  return _S_installed;
}

/** @brief Register the platform HAL (called once at init). */
[[nodiscard]] inline zzs::kernel::Status
hal_install (const _S_SensorHal *const hal) noexcept {
  if (hal == nullptr)
    return zzs::kernel::Status::OUT_OF_RANGE;
  if (_S_installed)
    return zzs::kernel::Status::INTERNAL;
  _S_hal = hal;
  _S_installed = true;
  return zzs::kernel::Status::OK;
}

/** @brief Read the next available sample via the installed HAL. */
[[nodiscard]] inline zzs::kernel::Status
hal_read_sample (sensor::Sample &sample) noexcept {
  if (_S_hal == nullptr || _S_hal->_M_read_sample == nullptr)
    return zzs::kernel::Status::UNSUPPORTED;
  zzs::kernel::Status st = _S_hal->_M_read_sample (sample);
  if (!zzs::kernel::status_ok (st))
    zzs::base::log_error ("hal read_sample ch={} failed: {}", static_cast<uint8_t> (sample._M_channel),
                          zzs::kernel::status_name (st));
  return st;
}

/** @brief Enable/disable a channel via the installed HAL. */
[[nodiscard]] inline zzs::kernel::Status
hal_set_channel_enabled (const sensor::Channel channel, const bool enabled) noexcept {
  if (_S_hal == nullptr || _S_hal->_M_set_channel_enabled == nullptr)
    return zzs::kernel::Status::UNSUPPORTED;
  return _S_hal->_M_set_channel_enabled (channel, enabled);
}

/** @brief Channel rate via the installed HAL, returned as a Status. */
[[nodiscard]] inline zzs::kernel::Status
hal_channel_rate_hz_status (const sensor::Channel channel, uint32_t &rate_hz) noexcept {
  if (rate_hz == 0u)
    return zzs::kernel::Status::OUT_OF_RANGE;
  if (_S_hal == nullptr || _S_hal->_M_channel_rate_hz == nullptr)
    return zzs::kernel::Status::UNSUPPORTED;
  rate_hz = _S_hal->_M_channel_rate_hz (channel);
  if (rate_hz == 0u)
    return zzs::kernel::Status::UNSUPPORTED;
  return zzs::kernel::Status::OK;
}

/** @brief Channel rate via the installed HAL (falls back to the compile-time table). */
[[nodiscard]] inline uint32_t
hal_channel_rate_hz (const sensor::Channel channel) noexcept {
  if (_S_hal == nullptr || _S_hal->_M_channel_rate_hz == nullptr)
    return sensor::channel_rate_hz (channel);
  return _S_hal->_M_channel_rate_hz (channel);
}

/** @brief Initialize the installed HAL. */
[[nodiscard]] inline zzs::kernel::Status
hal_init (void) noexcept {
  if (_S_hal == nullptr || _S_hal->_M_init == nullptr)
    return zzs::kernel::Status::UNSUPPORTED;
  zzs::kernel::Status st = _S_hal->_M_init ();
  if (zzs::kernel::status_ok (st))
    zzs::base::log_info ("hal init ok");
  else
    zzs::base::log_error ("hal init failed: {}", zzs::kernel::status_name (st));
  return st;
}

/** @} */
} // namespace zzs::kernel::hal
