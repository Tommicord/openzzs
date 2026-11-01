/*
 * Copyright(c) 2026 Tommicord
 *
 * MIT License
 *
 * JNI bridge between the Android UI and the OpenZZS preprocessing bank
 * + mllee inference core. Uses the kernel/sensor-driver platform HAL for
 * Android NDK sensor access, resamples every channel to the 100 Hz grid,
 * runs feature extraction and mllee inference, and returns results to Java.
 */

#include "kernel/kernel_bank.hpp"
#include "kernel/kernel_clock_grid.hpp"
#include "kernel/kernel_clock_source.hpp"
#include "kernel/kernel_config.hpp"
#include "kernel/kernel_dsp_resampler.hpp"
#include "kernel/kernel_feat_acoustic.hpp"
#include "kernel/kernel_feat_flicker.hpp"
#include "kernel/kernel_feat_pack.hpp"
#include "kernel/kernel_feat_plhr.hpp"
#include "kernel/kernel_feat_vibration.hpp"
#include "kernel/kernel_meta_ids.hpp"
#include "kernel/kernel_sensor_hal.hpp"
#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_seq_frames.hpp"
#include "kernel/kernel_util_fixed.hpp"
#include "mllee-native/model.hpp"
#include "mllee/core.hpp"
#include "sensr_bridge.h"

#include <android/log.h>
#include <android/looper.h>
#include <android/sensor.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <jni.h>
#include <string>
#include <time.h>

#define LOGI(...) __android_log_print (ANDROID_LOG_INFO, "OpenZZS", __VA_ARGS__)
#define LOGE(...) __android_log_print (ANDROID_LOG_ERROR, "OpenZZS", __VA_ARGS__)
#define LOGW(...) __android_log_print (ANDROID_LOG_WARN, "OpenZZS", __VA_ARGS__)

namespace {

/** @brief Grid tracker: 100 Hz common grid for all channels. */
static zzs::kernel::clock::_S_GridTracker _S_grid;

/** @brief Preprocessing bank state. */
static zzs::kernel::_S_BankConfig _S_bank_config{};
static zzs::kernel::_S_PreprocessingBank _S_bank{};
static zzs::kernel::frames::FeatureFrame _S_frame{};

/** @brief Per-channel resamplers to the 100 Hz grid. */
static zzs::kernel::dsp::_S_Resampler _S_resampler_mag_x{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_mag_y{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_mag_z{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_accel_x{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_accel_y{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_accel_z{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_light{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_sound{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_battery{};

/** @brief Stage state storage. */
static __attribute__ ((aligned (4))) int8_t _S_plhr_state[1024]{};
static __attribute__ ((aligned (4))) int8_t _S_vib_state[1024]{};
static __attribute__ ((aligned (4))) int8_t _S_flicker_state[1024]{};
static __attribute__ ((aligned (4))) int8_t _S_acoustic_state[1024]{};

/** @brief mllee model blob (extern, defined in generated model_blob_generated.cpp) and core. */
static zzs::mllee::Core _S_core{};

/** @brief JNI callback references. */
static JavaVM *_S_vm = nullptr;
static jobject _S_listener = nullptr;
static jmethodID _M_on_features = nullptr;
static jmethodID _M_on_inference = nullptr;

/**
 * @brief Feed one resampled grid-tick sample into the bank and run inference
 *        when a feature row is ready.
 */
static void
_feed_grid_sample (zzs::kernel::sensor::Channel channel, int16_t sample_q15, uint64_t timestamp_ns) noexcept {
    zzs::kernel::sensor::Sample s;
    s._M_channel = channel;
    s._M_sample_q15 = sample_q15;
    s._M_timestamp_ns = timestamp_ns;

    auto st = zzs::kernel::bank_feed_sample (_S_bank, s);
    if (!zzs::kernel::status_ok (st) && st != zzs::kernel::Status::NOT_READY)
        LOGE ("bank_feed_sample failed: %d", static_cast<int> (st));

    int8_t row[zzs::kernel::_S_feature_count];
    st = zzs::kernel::bank_produce_row (_S_bank, row);
    if (st == zzs::kernel::Status::NOT_READY)
        return; // No window finished yet
    if (!zzs::kernel::status_ok (st)) {
        LOGE ("bank_produce_row failed: %d", static_cast<int> (st));
        return;
    }

    // Append to sequence frame (seq_len=5)
    if (!zzs::kernel::frames::frame_append_row (_S_frame, row))
        return; // Frame not complete yet

    // Frame complete: run inference
    if (zzs::mllee_native::model::g_model_blob[0] != 'M') {
        LOGW ("Model blob not initialized, skipping inference");
        zzs::kernel::frames::frame_reset (_S_frame);
        return;
    }

    struct timespec ts_start;
    clock_gettime (CLOCK_MONOTONIC, &ts_start);

    const auto result =
        _S_core.run (zzs::mllee_native::model::g_model_blob, zzs::mllee_native::model::kBlobSize,
                     reinterpret_cast<const int8_t *> (_S_frame._M_row[0]), zzs::mllee_native::model::kInputDim);
    if (!result) {
        LOGE ("mllee inference failed: %d", static_cast<int> (result.error ().kind ()));
        zzs::kernel::frames::frame_reset (_S_frame);
        return;
    }

    struct timespec ts_end;
    clock_gettime (CLOCK_MONOTONIC, &ts_end);
    const uint64_t start_ns =
        static_cast<uint64_t> (ts_start.tv_sec) * 1000000000ull + static_cast<uint64_t> (ts_start.tv_nsec);
    const uint64_t end_ns =
        static_cast<uint64_t> (ts_end.tv_sec) * 1000000000ull + static_cast<uint64_t> (ts_end.tv_nsec);
    const uint64_t elapsed_us = (end_ns - start_ns) / 1000ull;
    const uint64_t elapsed_ms = elapsed_us / 1000ull;

    // Read 3 softmax output codes
    int8_t prob_codes[3];
    for (uint32_t i = 0; i < 3u; ++i)
        prob_codes[i] = result.value ()._M_data[i];

    // Convert to probabilities (0-100%)
    float probs[3];
    float total = 0.0f;
    for (uint32_t i = 0; i < 3u; ++i) {
        float p = static_cast<float> (prob_codes[i] + 127) / 254.0f;
        if (p < 0.0f)
            p = 0.0f;
        if (p > 1.0f)
            p = 1.0f;
        probs[i] = p;
        total += p;
    }
    if (total > 0.0f) {
        for (uint32_t i = 0; i < 3u; ++i)
            probs[i] /= total;
    } else {
        probs[0] = 1.0f;
        probs[1] = 0.0f;
        probs[2] = 0.0f;
    }

    // Find predicted class
    uint32_t predicted = 0u;
    for (uint32_t i = 1u; i < 3u; ++i)
        if (probs[i] > probs[predicted])
            predicted = i;

    LOGI ("Inference: %llu ms (%llu us), P_normal=%.3f P_outage=%.3f "
          "P_transition=%.3f -> %s",
          static_cast<unsigned long long> (elapsed_ms), static_cast<unsigned long long> (elapsed_us), probs[0],
          probs[1], probs[2], predicted == 0 ? "normal" : (predicted == 1 ? "outage" : "transition"));

    if (elapsed_ms > 10u)
        LOGW ("Inference slow: %llu ms (%llu us)", static_cast<unsigned long long> (elapsed_ms),
              static_cast<unsigned long long> (elapsed_us));

    // Deliver to Java on the UI thread
    JNIEnv *env;
    if (_S_vm->GetEnv (reinterpret_cast<void **> (&env), JNI_VERSION_1_6) != JNI_OK)
        return;
    if (_S_listener == nullptr || _M_on_inference == nullptr)
        return;

    jfloatArray jprobs = env->NewFloatArray (3);
    env->SetFloatArrayRegion (jprobs, 0, 3, probs);
    env->CallVoidMethod (_S_listener, _M_on_inference, jprobs);
    env->DeleteLocalRef (jprobs);

    zzs::kernel::frames::frame_reset (_S_frame);
}

/**
 * @brief Resampler push wrapper for each channel.
 */
static void
_push_sample (zzs::kernel::dsp::_S_Resampler &resampler, zzs::kernel::sensor::Channel channel, int16_t sample_q15,
              uint64_t timestamp_ns) noexcept {
    bool produced = false;
    int16_t out = 0;
    auto st = zzs::kernel::dsp::resampler_push (resampler, sample_q15, &produced, &out);
    if (!zzs::kernel::status_ok (st)) {
        LOGE ("resampler_push failed: %d", static_cast<int> (st));
        return;
    }
    if (produced)
        _feed_grid_sample (channel, out, timestamp_ns);
}

/** @brief Public feed function called by sensr_android.cpp and Java. */
extern "C" void
feed_native_sample (zzs::kernel::sensor::Channel channel, int16_t sample_q15, uint64_t timestamp_ns) noexcept {
    // Route through the appropriate resampler to the 100 Hz grid
    switch (channel) {
    case zzs::kernel::sensor::Channel::MAG_X:
        _push_sample (_S_resampler_mag_x, channel, sample_q15, timestamp_ns);
        break;
    case zzs::kernel::sensor::Channel::MAG_Y:
        _push_sample (_S_resampler_mag_y, channel, sample_q15, timestamp_ns);
        break;
    case zzs::kernel::sensor::Channel::MAG_Z:
        _push_sample (_S_resampler_mag_z, channel, sample_q15, timestamp_ns);
        break;
    case zzs::kernel::sensor::Channel::ACCEL_X:
        _push_sample (_S_resampler_accel_x, channel, sample_q15, timestamp_ns);
        break;
    case zzs::kernel::sensor::Channel::ACCEL_Y:
        _push_sample (_S_resampler_accel_y, channel, sample_q15, timestamp_ns);
        break;
    case zzs::kernel::sensor::Channel::ACCEL_Z:
        _push_sample (_S_resampler_accel_z, channel, sample_q15, timestamp_ns);
        break;
    case zzs::kernel::sensor::Channel::LIGHT:
        _push_sample (_S_resampler_light, channel, sample_q15, timestamp_ns);
        break;
    case zzs::kernel::sensor::Channel::SOUND_ENVELOPE:
        _push_sample (_S_resampler_sound, channel, sample_q15, timestamp_ns);
        break;
    case zzs::kernel::sensor::Channel::BATTERY:
        _push_sample (_S_resampler_battery, channel, sample_q15, timestamp_ns);
        break;
    default:
        // GYRO channels: feed directly (not used by stages yet)
        _feed_grid_sample (channel, sample_q15, timestamp_ns);
        break;
    }
}

} // anonymous namespace

extern "C" {

JNIEXPORT jint JNICALL
Java_com_openzzs_SensrBridge_initNative (JNIEnv *env, jobject thiz) noexcept {
    jint rv = env->GetJavaVM (&_S_vm);
    if (rv != JNI_OK || _S_vm == nullptr)
        return JNI_ERR;

    // Keep a global reference to the listener object
    _S_listener = env->NewGlobalRef (thiz);
    jclass cls = env->GetObjectClass (thiz);
    if (cls == nullptr) {
        env->DeleteGlobalRef (_S_listener);
        _S_listener = nullptr;
        return JNI_ERR;
    }
    _M_on_features = env->GetMethodID (cls, "onFeatures", "([Ljava/lang/String;)V");
    _M_on_inference = env->GetMethodID (cls, "onInference", "([F)V");
    if (_M_on_features == nullptr || _M_on_inference == nullptr) {
        env->DeleteGlobalRef (_S_listener);
        _S_listener = nullptr;
        env->DeleteLocalRef (cls);
        return JNI_ERR;
    }
    env->DeleteLocalRef (cls);

    // Initialize clock source (Android: CLOCK_MONOTONIC via clock_gettime)
    auto st = zzs::kernel::clock::clock_source_install ([] () noexcept -> zzs::kernel::clock::Nanoseconds {
        struct timespec ts;
        if (clock_gettime (CLOCK_MONOTONIC, &ts) == 0)
            return static_cast<uint64_t> (ts.tv_sec) * 1000000000ull + static_cast<uint64_t> (ts.tv_nsec);
        return 0ull;
    });
    if (!zzs::kernel::status_ok (st)) {
        LOGE ("clock_source_install failed: %d", static_cast<int> (st));
        return JNI_ERR;
    }

    // Initialize grid tracker
    st = zzs::kernel::clock::grid_tracker_init (_S_grid, 0ull);
    if (!zzs::kernel::status_ok (st)) {
        LOGE ("grid_tracker_init failed: %d", static_cast<int> (st));
        return JNI_ERR;
    }

    // Build model blob from trained or init weights
    auto model_st = zzs::mllee_native::model::init_model_blob ();
    if (!zzs::mllee::status_ok (model_st)) {
        LOGE ("init_model_blob failed: %d", static_cast<int> (model_st));
        return JNI_ERR;
    }

    // Configure resamplers for each channel (rate -> 100 Hz grid)
    st = zzs::kernel::dsp::resampler_configure (_S_resampler_mag_x, 200, 100);
    if (!zzs::kernel::status_ok (st))
        LOGE ("mag_x resampler configure failed: %d", static_cast<int> (st));
    st = zzs::kernel::dsp::resampler_configure (_S_resampler_mag_y, 200, 100);
    if (!zzs::kernel::status_ok (st))
        LOGE ("mag_y resampler configure failed: %d", static_cast<int> (st));
    st = zzs::kernel::dsp::resampler_configure (_S_resampler_mag_z, 200, 100);
    if (!zzs::kernel::status_ok (st))
        LOGE ("mag_z resampler configure failed: %d", static_cast<int> (st));

    st = zzs::kernel::dsp::resampler_configure (_S_resampler_accel_x, 200, 100);
    if (!zzs::kernel::status_ok (st))
        LOGE ("accel_x resampler configure failed: %d", static_cast<int> (st));
    st = zzs::kernel::dsp::resampler_configure (_S_resampler_accel_y, 200, 100);
    if (!zzs::kernel::status_ok (st))
        LOGE ("accel_y resampler configure failed: %d", static_cast<int> (st));
    st = zzs::kernel::dsp::resampler_configure (_S_resampler_accel_z, 200, 100);
    if (!zzs::kernel::status_ok (st))
        LOGE ("accel_z resampler configure failed: %d", static_cast<int> (st));

    st = zzs::kernel::dsp::resampler_configure (_S_resampler_light, 1, 100);
    if (!zzs::kernel::status_ok (st))
        LOGE ("light resampler configure failed: %d", static_cast<int> (st));

    st = zzs::kernel::dsp::resampler_configure (_S_resampler_sound, 200, 100);
    if (!zzs::kernel::status_ok (st))
        LOGE ("sound resampler configure failed: %d", static_cast<int> (st));

    st = zzs::kernel::dsp::resampler_configure (_S_resampler_battery, 1, 100);
    if (!zzs::kernel::status_ok (st))
        LOGE ("battery resampler configure failed: %d", static_cast<int> (st));

    // Register stages in the bank config
    _S_bank_config._M_stages[0]._M_vtable = &zzs::kernel::feat::plhr_vtable ();
    _S_bank_config._M_stages[0]._M_state = _S_plhr_state;
    _S_bank_config._M_stages[1]._M_vtable = &zzs::kernel::feat::vibration_vtable ();
    _S_bank_config._M_stages[1]._M_state = _S_vib_state;
    _S_bank_config._M_stages[2]._M_vtable = &zzs::kernel::feat::flicker_vtable ();
    _S_bank_config._M_stages[2]._M_state = _S_flicker_state;
    _S_bank_config._M_stages[3]._M_vtable = &zzs::kernel::feat::acoustic_vtable ();
    _S_bank_config._M_stages[3]._M_state = _S_acoustic_state;
    _S_bank_config._M_stage_count = 4;

    st = zzs::kernel::bank_init (_S_bank, _S_bank_config);
    if (!zzs::kernel::status_ok (st)) {
        LOGE ("bank_init failed: %d", static_cast<int> (st));
        return JNI_ERR;
    }

    LOGI ("OpenZZS bridge initialized (model=%u bytes, 4 stages)", zzs::mllee_native::model::kBlobSize);
    return JNI_OK;
}

JNIEXPORT void JNICALL
Java_com_openzzs_SensrBridge_shutdownNative (JNIEnv *env, jobject thiz) noexcept {
    if (!_S_listener)
        return;

    zzs::kernel::bank_reset (_S_bank);
    env->DeleteGlobalRef (_S_listener);
    _S_listener = nullptr;
    _M_on_features = nullptr;
    _M_on_inference = nullptr;
    LOGI ("OpenZZS bridge shutdown");
}

JNIEXPORT void JNICALL
Java_com_openzzs_SensrBridge_feedSampleNative (JNIEnv *env, jobject thiz, jint channel, jshort sample_q15,
                                               jlong timestamp_ns) noexcept {
    if (!_S_listener)
        return;

    auto ch = static_cast<zzs::kernel::sensor::Channel> (channel);
    auto sq15 = static_cast<int16_t> (sample_q15);
    auto ts = static_cast<uint64_t> (timestamp_ns);

    feed_native_sample (ch, sq15, ts);
}

JNIEXPORT jobjectArray JNICALL
Java_com_openzzs_SensrBridge_getFeaturesNative (JNIEnv *env, jobject thiz) noexcept {
    if (!_S_listener)
        return env->NewObjectArray (0, env->FindClass ("java/lang/String"), nullptr);

    jclass string_cls = env->FindClass ("java/lang/String");
    jobjectArray result = env->NewObjectArray (zzs::kernel::_S_feature_count, string_cls, nullptr);

    int8_t row[zzs::kernel::_S_feature_count];
    auto st = zzs::kernel::bank_produce_row (_S_bank, row);
    if (st == zzs::kernel::Status::NOT_READY)
        return result;
    if (!zzs::kernel::status_ok (st)) {
        LOGE ("bank_produce_row failed: %d", static_cast<int> (st));
        return result;
    }

    for (uint32_t i = 0; i < zzs::kernel::_S_feature_count; ++i) {
        const char *name = zzs::kernel::meta::feat_name (i).data ();
        std::string entry = std::string (name) + "=" + std::to_string (static_cast<int> (row[i]));
        jstring jentry = env->NewStringUTF (entry.c_str ());
        env->SetObjectArrayElement (result, i, jentry);
        env->DeleteLocalRef (jentry);
    }

    return result;
}

JNIEXPORT jint JNICALL
JNI_OnLoad (JavaVM *vm, void *reserved) {
    JNIEnv *env;
    if (vm->GetEnv (reinterpret_cast<void **> (&env), JNI_VERSION_1_6) != JNI_OK)
        return JNI_ERR;

    jclass cls = env->FindClass ("com/openzzs/SensrBridge");
    if (cls == nullptr)
        return JNI_ERR;

    static const JNINativeMethod methods[] = {
        {"initNative", "()I", reinterpret_cast<void *> (Java_com_openzzs_SensrBridge_initNative)},
        {"shutdownNative", "()V", reinterpret_cast<void *> (Java_com_openzzs_SensrBridge_shutdownNative)},
        {"feedSampleNative", "(ISJ)V", reinterpret_cast<void *> (Java_com_openzzs_SensrBridge_feedSampleNative)},
        {"getFeaturesNative", "()[Ljava/lang/String;",
         reinterpret_cast<void *> (Java_com_openzzs_SensrBridge_getFeaturesNative)},
    };

    if (env->RegisterNatives (cls, methods, sizeof (methods) / sizeof (methods[0])) < 0)
        return JNI_ERR;

    return JNI_VERSION_1_6;
}

} // extern "C"
