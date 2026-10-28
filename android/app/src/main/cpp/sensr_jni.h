/*
 * Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
 * Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
 * WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. */

#ifndef OPENZZS_SENSR_JNI_H
#define OPENZZS_SENSR_JNI_H

#include <jni.h>
#include <android/log.h>

#include "sensr/sensr.h"
#include "sensr/sensr_android.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOG_TAG "Openzzs-Sensr"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

/**
 * @brief Opaque JNI-side state owned by SensrBridge.java.
 *
 * The Java handle is the numeric address of this struct. All native entry
 * points validate it before dereferencing so a stale handle fails cleanly.
 */
struct sensr_jni_state
{
        struct sensr_backend *backend;
        int running;
        int sensorsEnabled;
};

typedef struct sensr_jni_state sensr_jni_state_t;

JNIEXPORT jlong JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeCreate (JNIEnv *env, jclass clazz, jint backendKind);

JNIEXPORT jboolean JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeInitRuntime (JNIEnv *env, jclass clazz, jint heapSizeBytes);

JNIEXPORT void JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeShutdownRuntime (JNIEnv *env, jclass clazz);

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeBackendKind (JNIEnv *env, jclass clazz, jlong handle);

JNIEXPORT void JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeDestroy (JNIEnv *env, jclass clazz, jlong handle);

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeEnableSensor (JNIEnv *env, jclass clazz, jlong handle, jint sensorType,
                                                        jint rateHz);

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeDisableSensor (JNIEnv *env, jclass clazz, jlong handle, jint sensorType);

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeStart (JNIEnv *env, jclass clazz, jlong handle);

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeStop (JNIEnv *env, jclass clazz, jlong handle);

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeReadEvent (JNIEnv *env, jclass clazz, jlong handle, jint timeoutMs,
                                                     jobject outEvent);

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativePollEvents (JNIEnv *env, jclass clazz, jlong handle, jobjectArray outEvents,
                                                      jint maxCount);

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeAvailable (JNIEnv *env, jclass clazz, jlong handle);

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeFlush (JNIEnv *env, jclass clazz, jlong handle);

JNIEXPORT void JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeGetStats (JNIEnv *env, jclass clazz, jlong handle, jlongArray outStats);

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeSensorCount (JNIEnv *env, jclass clazz, jlong handle);

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeGetSensors (JNIEnv *env, jclass clazz, jlong handle,
                                                      jobjectArray outSensors);

JNIEXPORT jboolean JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeHasSensor (JNIEnv *env, jclass clazz, jlong handle, jint sensorType);

JNIEXPORT void JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeSetDefaultRate (JNIEnv *env, jclass clazz, jlong handle, jint rateHz);

#ifdef __cplusplus
}
#endif

#endif // OPENZZS_SENSR_JNI_H
