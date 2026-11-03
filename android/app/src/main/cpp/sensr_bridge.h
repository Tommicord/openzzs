/*
 * Copyright(c) 2026 Tommicord
 *
 * MIT License
 *
 * Shared declaration for the native sample feed function used by the
 * Android sensor polling thread (sensr_android.cpp) and the JNI layer
 * (sensr_jni.cpp).
 */

#pragma once

#include "kernel/kernel_sensor_types.hpp"

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Feed one Q15 sensor sample into the OpenZZS preprocessing bank.
 *
 * The sample is resampled to the 100 Hz grid (if needed), fed through all
 * registered preprocessing stages, and when a feature row is complete, mllee
 * inference is run and the result is delivered to the Java listener.
 *
 * This function is thread-safe and may be called from any thread.
 *
 * @param channel   Sensor channel id (see sensor::Channel)
 * @param sample_q15 Q15 fixed-point sample value
 * @param timestamp_ns Monotonic nanosecond timestamp
 */
void
feed_native_sample (zzs::kernel::sensor::Channel channel, int16_t sample_q15, uint64_t timestamp_ns) noexcept;

#ifdef __cplusplus
} // extern "C"
#endif
