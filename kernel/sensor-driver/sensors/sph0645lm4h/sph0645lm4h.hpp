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
#include "sph0645lm4h_reg.h"

#include <cstdint>

/** @brief Function supplied by the platform to read one I2S/PDM audio sample
 *         (16-bit PCM, Q15 format). */
using PdmSampleFn = int16_t (*) (void) noexcept;

/**
 * @brief Knowles SPH0645LM4H PDM microphone sound driver (SOUND_ENVELOPE channel).
 *
 * Reads PDM/PCM audio samples from an I2S input (provided via a caller-supplied
 * read function), accumulates them into envelope blocks and computes a Q15
 * sound envelope via mean-absolute-value with exponential smoothing.
 * Integer-only; no float.
 * @{
 */
namespace zzs::kernel::sensor::driver::sensors {

/** @brief SPH0645LM4H sound driver (SOUND_ENVELOPE channel). */
class Sph0645lm4hSoundDriver : public SensorDriver {
public:
  /** @brief Construct with a non-owning bus pointer, an optional PDM sample
   *         read function and an optional clock source callback for timestamps. */
  Sph0645lm4hSoundDriver (bus::BusInterface *bus, PdmSampleFn pdm_read = nullptr,
                          uint64_t (*clock) (void) = nullptr) noexcept
      : SensorDriver (bus, clock) {
    _M_pdm_read = pdm_read;
  }

  [[nodiscard]] zzs::kernel::Status
  init () noexcept override {
    /* No device registers to probe for a PDM microphone.  Mark the device
     * as present if a PDM read function was supplied. */
    _M_device_id = (_M_pdm_read != nullptr) ? 0x49u : 0x00u;
    _M_envelope_q15 = 0;
    _M_block_sum = 0;
    _M_block_count = 0;
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] zzs::kernel::Status
  read_sample (sensor::Sample &out) noexcept override {
    if (_M_pdm_read == nullptr)
      return zzs::kernel::Status::UNSUPPORTED;

    /* Accumulate one audio sample into the current envelope block. */
    const int16_t sample = _M_pdm_read ();
    const int32_t abs_sample = sample >= 0 ? sample : -sample;
    if (__builtin_add_overflow (_M_block_sum, abs_sample, &_M_block_sum))
      _M_block_sum = abs_sample > 0 ? INT32_MAX : INT32_MIN;
    _M_block_count += 1u;

    if (_M_block_count < sph0645lm4h::EnvelopeBlock)
      return zzs::kernel::Status::STALE;

    /* Compute mean absolute value of the block. */
    const int32_t mean_abs = static_cast<int32_t> (_M_block_sum / static_cast<int32_t> (_M_block_count));
    _M_block_sum = 0;
    _M_block_count = 0;

    /* Exponential smoothing: env = alpha * mean_abs + (1 - alpha) * env. */
    int32_t new_env = 0;
    if (__builtin_mul_overflow (mean_abs, sph0645lm4h::EnvelopeAlphaQ15, &new_env))
      new_env = mean_abs > 0 ? INT32_MAX : INT32_MIN;
    new_env = zzs::kernel::sensor::driver::fixed::q15_mul (new_env >> 15, sph0645lm4h::EnvelopeAlphaQ15);

    int32_t one_minus_alpha = 0;
    if (__builtin_sub_overflow (static_cast<int32_t> (32768), sph0645lm4h::EnvelopeAlphaQ15, &one_minus_alpha))
      one_minus_alpha = INT32_MAX;
    int32_t old_env = 0;
    if (__builtin_mul_overflow (_M_envelope_q15, one_minus_alpha, &old_env))
      old_env = _M_envelope_q15 > 0 ? INT32_MAX : INT32_MIN;
    old_env = zzs::kernel::sensor::driver::fixed::q15_mul (old_env >> 15, one_minus_alpha);

    int32_t env_sum = 0;
    if (__builtin_add_overflow (new_env, old_env, &env_sum))
      env_sum = new_env > 0 ? INT32_MAX : INT32_MIN;
    _M_envelope_q15 = zzs::kernel::sensor::driver::fixed::q15_clamp (env_sum);

    out._M_channel = sensor::Channel::SOUND_ENVELOPE;
    out._M_sample_q15 = zzs::kernel::sensor::driver::fixed::q15_to_raw (_M_envelope_q15);
    out._M_timestamp_ns = timestamp_ns ();
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] zzs::kernel::Status
  set_channel_enabled (sensor::Channel ch, bool enabled) noexcept override {
    if (ch != sensor::Channel::SOUND_ENVELOPE)
      return zzs::kernel::Status::OUT_OF_RANGE;
    _M_enabled = enabled;
    return zzs::kernel::Status::OK;
  }

  [[nodiscard]] uint32_t
  channel_rate_hz (sensor::Channel ch) noexcept override {
    (void) ch;
    return sph0645lm4h::EnvelopeRateHz;
  }

  [[nodiscard]] uint8_t
  device_id () const noexcept override {
    return _M_device_id;
  }

  [[nodiscard]] bool
  self_test () noexcept override {
    /* Basic self-test: confirm the PDM read function is present and
     * produces a plausible (non-saturating) sample. */
    if (_M_pdm_read == nullptr)
      return false;
    const int16_t s = _M_pdm_read ();
    return s != 0 && s != 32767 && s != -32768;
  }

private:
  /** @brief Platform-supplied PDM sample read function, or nullptr. */
  PdmSampleFn _M_pdm_read = nullptr;
  /** @brief Whether the sound envelope channel is enabled. */
  bool _M_enabled = true;
  /** @brief Current smoothed envelope (Q15). */
  int32_t _M_envelope_q15 = 0;
  /** @brief Running sum of absolute sample values in current block. */
  int32_t _M_block_sum = 0;
  /** @brief Number of samples accumulated in current block. */
  uint32_t _M_block_count = 0;
  /** @brief Device ID byte. */
  uint8_t _M_device_id = 0;
};

/** @} */
} // namespace zzs::kernel::sensor::driver::sensors
