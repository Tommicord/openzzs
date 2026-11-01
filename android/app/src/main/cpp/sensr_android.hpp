/*
 * Copyright(c) 2026 Tommicord
 *
 * MIT License
 */

#pragma once

#include <android/looper.h>
#include <android/sensor.h>
#include <cstdint>
#include <cstring>
#include <jni.h>
#include <thread>

namespace zzs::kernel::sensor::android {

/** @brief ASensorManager and ASensorEventQueue wrappers for the Android HAL. */
struct AndroidSensorContext {
  ASensorManager *_M_manager = nullptr;
  ASensorEventQueue *_M_queue = nullptr;
  ALooper *_M_looper = nullptr;
  ASensor const *_M_mag_sensor = nullptr;
  ASensor const *_M_accel_sensor = nullptr;
  ASensor const *_M_gyro_sensor = nullptr;
  ASensor const *_M_light_sensor = nullptr;
  ASensor const *_M_sound_sensor = nullptr;
  ASensor const *_M_battery_sensor = nullptr;
  std::thread _M_thread;
  bool _M_enabled = false;
};

/** @brief Initialize Android sensor subsystem.
 *  @returns true on success. */
bool
android_sensor_init (AndroidSensorContext &ctx) noexcept;

/** @brief Enable all sensors and register event callbacks on the looper. */
bool
android_sensor_enable (AndroidSensorContext &ctx, ALooper *looper, int ident) noexcept;

/** @brief Disable all sensors. */
void
android_sensor_disable (AndroidSensorContext &ctx) noexcept;

/** @brief Process pending sensor events and feed them to the preprocessing bank. */
void
android_sensor_poll (AndroidSensorContext &ctx) noexcept;

/** @brief Shut down the Android sensor subsystem. */
void
android_sensor_shutdown (AndroidSensorContext &ctx) noexcept;

} // namespace zzs::kernel::sensor::android
