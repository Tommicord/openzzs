/*
 * Copyright(c) 2026 Tommicord
 *
 * MIT License
 *
 * Android sensor polling thread using the NDK sensor API (ASensorManager).
 * Polls magnetic field, accelerometer, gyroscope, and light sensors at their
 * hardware rates and feeds Q15-normalized samples into the OpenZZS
 * preprocessing bank via feed_native_sample().
 */

#include "sensr_android.hpp"

#include "sensr_bridge.h"

#include <android/log.h>
#include <android/looper.h>
#include <android/sensor.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <thread>

#define LOGI(...) __android_log_print (ANDROID_LOG_INFO, "OpenZZS", __VA_ARGS__)
#define LOGE(...) __android_log_print (ANDROID_LOG_ERROR, "OpenZZS", __VA_ARGS__)

namespace zzs::kernel::sensor::android {

/** @brief Convert an Android sensor event value to Q15 fixed-point. */
static inline int16_t
to_q15 (float value, float scale) noexcept {
    const float clamped = value / scale;
    const float bounded = clamped > 1.0f ? 1.0f : (clamped < -1.0f ? -1.0f : clamped);
    return static_cast<int16_t> (bounded * 32767.0f);
}

/** @brief Sensor polling thread function. */
static int
_sensor_poller (void *ctx) noexcept {
    auto *context = static_cast<AndroidSensorContext *> (ctx);
    if (context == nullptr)
        return -1;

    ALooper *looper = context->_M_looper;
    if (looper == nullptr) {
        looper = ALooper_forThread ();
        if (looper == nullptr)
            looper = ALooper_prepare (ALOOPER_PREPARE_ALLOW_NON_CALLBACKS);
        context->_M_looper = looper;
    }

    ASensorManager *manager = ASensorManager_getInstance ();
    if (manager == nullptr) {
        LOGE ("Failed to get ASensorManager instance");
        return -1;
    }
    context->_M_manager = manager;

    ASensorEventQueue *queue = ASensorManager_createEventQueue (manager, looper, 0, nullptr, nullptr);
    if (queue == nullptr) {
        LOGE ("Failed to create sensor event queue");
        return -1;
    }
    context->_M_queue = queue;

    auto enable_sensor = [&] (ASensor const *sensor) {
        if (sensor != nullptr) {
            ASensorEventQueue_enableSensor (queue, sensor);
            int min_delay = ASensor_getMinDelay (sensor);
            ASensorEventQueue_setEventRate (queue, sensor, min_delay);
            LOGI ("Enabled sensor: %s (min_delay=%d us)", ASensor_getName (sensor), min_delay);
        }
    };

    context->_M_mag_sensor = ASensorManager_getDefaultSensor (manager, ASENSOR_TYPE_MAGNETIC_FIELD);
    context->_M_accel_sensor = ASensorManager_getDefaultSensor (manager, ASENSOR_TYPE_ACCELEROMETER);
    context->_M_gyro_sensor = ASensorManager_getDefaultSensor (manager, ASENSOR_TYPE_GYROSCOPE);
    context->_M_light_sensor = ASensorManager_getDefaultSensor (manager, ASENSOR_TYPE_LIGHT);

    enable_sensor (context->_M_mag_sensor);
    enable_sensor (context->_M_accel_sensor);
    enable_sensor (context->_M_gyro_sensor);
    enable_sensor (context->_M_light_sensor);

    context->_M_enabled = true;
    LOGI ("Android sensor HAL polling thread started");

    ASensorEvent event;
    while (context->_M_enabled) {
        int ident = ALooper_pollOnce (10, nullptr, nullptr, nullptr);
        if (ident < 0)
            continue;

        while (ASensorEventQueue_getEvents (queue, &event, 1) > 0) {
            uint64_t ts = static_cast<uint64_t> (event.timestamp);

            switch (event.type) {
            case ASENSOR_TYPE_MAGNETIC_FIELD:
                feed_native_sample (zzs::kernel::sensor::Channel::MAG_X, to_q15 (event.data[0], 100.0f), ts);
                feed_native_sample (zzs::kernel::sensor::Channel::MAG_Y, to_q15 (event.data[1], 100.0f), ts);
                feed_native_sample (zzs::kernel::sensor::Channel::MAG_Z, to_q15 (event.data[2], 100.0f), ts);
                break;

            case ASENSOR_TYPE_ACCELEROMETER:
                feed_native_sample (zzs::kernel::sensor::Channel::ACCEL_X, to_q15 (event.data[0], 20.0f), ts);
                feed_native_sample (zzs::kernel::sensor::Channel::ACCEL_Y, to_q15 (event.data[1], 20.0f), ts);
                feed_native_sample (zzs::kernel::sensor::Channel::ACCEL_Z, to_q15 (event.data[2], 20.0f), ts);
                break;

            case ASENSOR_TYPE_GYROSCOPE:
                feed_native_sample (zzs::kernel::sensor::Channel::GYRO_X, to_q15 (event.data[0], 20.0f), ts);
                feed_native_sample (zzs::kernel::sensor::Channel::GYRO_Y, to_q15 (event.data[1], 20.0f), ts);
                feed_native_sample (zzs::kernel::sensor::Channel::GYRO_Z, to_q15 (event.data[2], 20.0f), ts);
                break;

            case ASENSOR_TYPE_LIGHT:
                feed_native_sample (zzs::kernel::sensor::Channel::LIGHT, to_q15 (event.data[0], 10000.0f), ts);
                break;

            default:
                break;
            }
        }
    }

    ASensorEventQueue_disableSensor (queue, context->_M_mag_sensor);
    ASensorEventQueue_disableSensor (queue, context->_M_accel_sensor);
    ASensorEventQueue_disableSensor (queue, context->_M_gyro_sensor);
    ASensorEventQueue_disableSensor (queue, context->_M_light_sensor);
    ASensorManager_destroyEventQueue (manager, queue);
    context->_M_queue = nullptr;
    context->_M_enabled = false;

    LOGI ("Android sensor HAL polling thread stopped");
    return 0;
}

bool
android_sensor_init (AndroidSensorContext &ctx) noexcept {
    ctx._M_manager = ASensorManager_getInstance ();
    if (ctx._M_manager == nullptr) {
        LOGE ("Failed to get ASensorManager instance");
        return false;
    }
    return true;
}

bool
android_sensor_enable (AndroidSensorContext &ctx, ALooper *looper, int ident) noexcept {
    (void) looper;
    (void) ident;
    if (ctx._M_manager == nullptr)
        return false;

    ctx._M_thread = std::thread (_sensor_poller, &ctx);
    if (!ctx._M_thread.joinable ()) {
        LOGE ("Failed to start sensor polling thread");
        return false;
    }
    return true;
}

void
android_sensor_disable (AndroidSensorContext &ctx) noexcept {
    ctx._M_enabled = false;
    if (ctx._M_thread.joinable ())
        ctx._M_thread.join ();
}

void
android_sensor_poll (AndroidSensorContext &ctx) noexcept {
    (void) ctx; // Polling is done in the background thread
}

void
android_sensor_shutdown (AndroidSensorContext &ctx) noexcept {
    android_sensor_disable (ctx);
    ctx._M_manager = nullptr;
    ctx._M_queue = nullptr;
}

} // namespace zzs::kernel::sensor::android
