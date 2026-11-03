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

/** @file probe_main.cpp
 *  @brief Probe native entry point for the OpenZZS magnetometer probe.
 *
 *  Uses the kernel sensor-driver Android platform HAL
 *  (kernel/sensor-driver/platforms/android/) to read real LIS3MDL
 *  magnetometer samples over I2C and log them via zzs::base::log_info /
 *  log_error (which routes to Android logcat on __ANDROID__).
 *
 *  This proves whether the kernel sensor-driver is receiving real sensor
 *  data from the hardware I2C bus, or whether it is producing hardcoded /
 *  mock values.
 *
 *  View output with:
 *    adb logcat -s OpenZZSProbe
 */

#include <unistd.h>

#include <cstdint>

#include "kernel/kernel_sensor_hal.hpp"
#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_status.hpp"
#include "platforms/android/platform_hal.hpp"
#include "std/log.hpp"

namespace {

[[nodiscard]] int
run_probe () noexcept {
  using namespace zzs::kernel;
  using namespace zzs::kernel::sensor;
  using namespace zzs::kernel::hal;
  using namespace zzs::kernel::sensor::driver::platform;
  using namespace zzs::kernel::sensor::driver::platform::android;

  zzs::base::log_info ("{}", "OpenZZS magnetometer probe starting");

  // Build the Android sensor HAL from the kernel's platform code.
  const _S_SensorHal *hal = create_android_sensor_hal ();
  if (hal == nullptr) {
    zzs::base::log_error ("{}", "create_android_sensor_hal returned nullptr");
    return 1;
  }
  zzs::base::log_info ("HAL vtable obtained");

  // Install the HAL into the kernel.
  zzs::kernel::Status st = hal_install (hal);
  if (!status_ok (st)) {
    zzs::base::log_error ("hal_install failed: {}", status_name (st));
    return 1;
  }
  zzs::base::log_info ("HAL installed");

  // Initialize the HAL (this initializes the LIS3MDL magnetometer on I2C bus 1,
  // address 0x1C, configures continuous conversion mode at 20 Hz, ±4 gauss).
  st = hal_init ();
  if (!status_ok (st)) {
    zzs::base::log_error ("hal_init failed: {}", status_name (st));
    return 1;
  }
  zzs::base::log_info ("HAL init ok (LIS3MDL configured)");

  // Enable all three magnetometer axes explicitly.
  for (sensor::Channel ch : {sensor::Channel::MAG_X, sensor::Channel::MAG_Y,
                             sensor::Channel::MAG_Z}) {
    st = hal_set_channel_enabled (ch, true);
    zzs::base::log_info ("enable MAG axis {}: {}", static_cast<uint32_t> (ch),
                         status_name (st));
  }

  // Read and log magnetometer samples.  The LIS3MDL runs at 20 Hz; the HAL
  // round-robins accel/gyro/mag/light/sound, so we poll until we collect a
  // few samples per axis.
  zzs::base::log_info ("Reading magnetometer samples (Q15 milligauss)...");

  int mag_samples_seen[3] = {0, 0, 0};
  const int target_per_axis = 5;
  int total_reads = 0;
  int total_ok = 0;
  int total_stale = 0;

  while (total_reads < 200 &&
         (mag_samples_seen[0] < target_per_axis ||
          mag_samples_seen[1] < target_per_axis ||
          mag_samples_seen[2] < target_per_axis)) {
    sensor::Sample sample{};
    total_reads++;
    st = hal_read_sample (sample);
    if (status_ok (st)) {
      total_ok++;
      if (sample._M_channel >= sensor::Channel::MAG_X &&
          sample._M_channel <= sensor::Channel::MAG_Z) {
        uint32_t axis = static_cast<uint32_t> (sample._M_channel) -
                        static_cast<uint32_t> (sensor::Channel::MAG_X);
        if (mag_samples_seen[axis] < target_per_axis) {
          int16_t q15 = sample._M_sample_q15;
          // Convert Q15 to milligauss: Q15 * 4789 / 32768
          int32_t mg = (static_cast<int32_t> (q15) * 4789) / 32768;
          zzs::base::log_info (
              "MAG axis {}: raw_q15={} ({} milliG), ts_ns={}",
              axis == 0u   ? "X"
              : axis == 1u ? "Y"
                           : "Z",
              q15, mg, static_cast<long long> (sample._M_timestamp_ns));
          mag_samples_seen[axis]++;
        }
      }
    } else if (st == zzs::kernel::Status::STALE) {
      total_stale++;
    } else {
      zzs::base::log_warn ("hal_read_sample status: {}", status_name (st));
    }
    usleep (50000); // 50 ms between polls
  }

  zzs::base::log_info (
      "Probe done: reads={} ok={} stale={} | MAG_X={} MAG_Y={} MAG_Z={}",
      total_reads, total_ok, total_stale, mag_samples_seen[0],
      mag_samples_seen[1], mag_samples_seen[2]);

  // Print a verdict line that is easy to grep from logcat.
  if (mag_samples_seen[0] > 0 && mag_samples_seen[1] > 0 &&
      mag_samples_seen[2] > 0) {
    zzs::base::log_info (
        "VERDICT: REAL magnetometer data received from LIS3MDL via kernel "
        "sensor-driver (not hardcoded)");
  } else {
    zzs::base::log_warn (
        "VERDICT: No magnetometer samples received -- I2C bus may be "
        "inaccessible (needs root) or LIS3MDL not present");
  }

  return 0;
}

} // namespace

extern "C" [[maybe_unused]] void
ANativeActivity_onCreate (void * /*savedState*/, size_t /*savedStateSize*/) noexcept {
  run_probe ();
}
