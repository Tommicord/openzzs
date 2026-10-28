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

#include "sensr_jni.h"

#include "sensr/sensr_mock.h"

#include "std/heap_allocator.h"
#include "std/log.h"

#include <stdint.h>
#include <string.h>

/* Keep these tables in sync with SensrBridge.java. */
static enum sensr_sensor_type
java_to_sensr_type (jint javaType)
{
        switch (javaType)
                {
                case 1:
                        return SENSR_SENSOR_ACCELEROMETER;
                case 2:
                        return SENSR_SENSOR_GYROSCOPE;
                case 3:
                        return SENSR_SENSOR_MAGNETOMETER;
                case 4:
                        return SENSR_SENSOR_LIGHT;
                case 5:
                        return SENSR_SENSOR_PROXIMITY;
                case 6:
                        return SENSR_SENSOR_PRESSURE;
                case 7:
                        return SENSR_SENSOR_TEMPERATURE;
                case 8:
                        return SENSR_SENSOR_HUMIDITY;
                case 9:
                        return SENSR_SENSOR_GRAVITY;
                case 10:
                        return SENSR_SENSOR_LINEAR_ACCELERATION;
                case 11:
                        return SENSR_SENSOR_ROTATION_VECTOR;
                case 12:
                        return SENSR_SENSOR_GAME_ROTATION_VECTOR;
                case 13:
                        return SENSR_SENSOR_MAGNETIC_FIELD_UNCALIBRATED;
                case 14:
                        return SENSR_SENSOR_GYROSCOPE_UNCALIBRATED;
                case 15:
                        return SENSR_SENSOR_STEP_COUNTER;
                case 16:
                        return SENSR_SENSOR_STEP_DETECTOR;
                case 17:
                        return SENSR_SENSOR_SIGNIFICANT_MOTION;
                case 18:
                        return SENSR_SENSOR_HEART_RATE;
                case 19:
                        return SENSR_SENSOR_SOUND_LEVEL;
                case 20:
                        return SENSR_SENSOR_BATTERY_LEVEL;
                default:
                        return SENSR_SENSOR_UNKNOWN;
                }
}

static jint
sensr_to_java_type (enum sensr_sensor_type type)
{
        switch (type)
                {
                case SENSR_SENSOR_ACCELEROMETER:
                        return 1;
                case SENSR_SENSOR_GYROSCOPE:
                        return 2;
                case SENSR_SENSOR_MAGNETOMETER:
                        return 3;
                case SENSR_SENSOR_LIGHT:
                        return 4;
                case SENSR_SENSOR_PROXIMITY:
                        return 5;
                case SENSR_SENSOR_PRESSURE:
                        return 6;
                case SENSR_SENSOR_TEMPERATURE:
                        return 7;
                case SENSR_SENSOR_HUMIDITY:
                        return 8;
                case SENSR_SENSOR_GRAVITY:
                        return 9;
                case SENSR_SENSOR_LINEAR_ACCELERATION:
                        return 10;
                case SENSR_SENSOR_ROTATION_VECTOR:
                        return 11;
                case SENSR_SENSOR_GAME_ROTATION_VECTOR:
                        return 12;
                case SENSR_SENSOR_MAGNETIC_FIELD_UNCALIBRATED:
                        return 13;
                case SENSR_SENSOR_GYROSCOPE_UNCALIBRATED:
                        return 14;
                case SENSR_SENSOR_STEP_COUNTER:
                        return 15;
                case SENSR_SENSOR_STEP_DETECTOR:
                        return 16;
                case SENSR_SENSOR_SIGNIFICANT_MOTION:
                        return 17;
                case SENSR_SENSOR_HEART_RATE:
                        return 18;
                case SENSR_SENSOR_SOUND_LEVEL:
                        return 19;
                case SENSR_SENSOR_BATTERY_LEVEL:
                        return 20;
                default:
                        return 0;
                }
}

static enum ZZS_sensr_backend_kind
java_to_backend_kind (jint javaKind)
{
        switch (javaKind)
                {
                case 1:
                        return ZZS_sensr_backend_MOCK;
                case 2:
                        return ZZS_sensr_backend_ANDROID;
                case 3:
                        return ZZS_sensr_backend_IOS;
                default:
                        return ZZS_sensr_backend_UNKNOWN;
                }
}

static inline sensr_jni_state_t *
get_state (jlong handle)
{
        return (sensr_jni_state_t *) (uintptr_t) handle;
}

static void
fill_event (JNIEnv *env, jobject out, const struct sensr_event *event)
{
        if (!out || !event)
                return;

        jclass cls = (*env)->GetObjectClass (env, out);
        if (!cls)
                return;

        jfieldID fType = (*env)->GetFieldID (env, cls, "type", "I");
        jfieldID fTimestamp = (*env)->GetFieldID (env, cls, "timestampNs", "J");
        jfieldID fAccuracy = (*env)->GetFieldID (env, cls, "accuracy", "I");
        jfieldID fValueCount = (*env)->GetFieldID (env, cls, "valueCount", "I");
        jfieldID fSource = (*env)->GetFieldID (env, cls, "source", "I");
        jfieldID fSequence = (*env)->GetFieldID (env, cls, "sequence", "J");
        jfieldID fValues = (*env)->GetFieldID (env, cls, "values", "[F");

        if (fType)
                (*env)->SetIntField (env, out, fType, sensr_to_java_type (event->sample.type));
        if (fTimestamp)
                (*env)->SetLongField (env, out, fTimestamp, (jlong) event->sample.timestampNs);
        if (fAccuracy)
                (*env)->SetIntField (env, out, fAccuracy, (jint) event->sample.accuracy);
        if (fValueCount)
                (*env)->SetIntField (env, out, fValueCount, (jint) event->sample.valueCount);
        if (fSource)
                (*env)->SetIntField (env, out, fSource, (jint) event->source);
        if (fSequence)
                (*env)->SetLongField (env, out, fSequence, (jlong) event->sequence);

        if (fValues)
                {
                        jfloatArray values = (jfloatArray) (*env)->GetObjectField (env, out, fValues);
                        if (values)
                                {
                                        jsize length = (*env)->GetArrayLength (env, values);
                                        jsize count = (jsize) event->sample.valueCount;
                                        if (count > length)
                                                count = length;
                                        if (count > 0)
                                                (*env)->SetFloatArrayRegion (env, values, 0, count,
                                                                             event->sample.values);
                                        (*env)->DeleteLocalRef (env, values);
                                }
                }

        (*env)->DeleteLocalRef (env, cls);
}

static void
fill_sensor_info (JNIEnv *env, jobject out, const struct sensr_sensor_info *info)
{
        if (!out || !info)
                return;

        jclass cls = (*env)->GetObjectClass (env, out);
        if (!cls)
                return;

        jfieldID fType = (*env)->GetFieldID (env, cls, "type", "I");
        jfieldID fName = (*env)->GetFieldID (env, cls, "name", "Ljava/lang/String;");
        jfieldID fVendor = (*env)->GetFieldID (env, cls, "vendor", "Ljava/lang/String;");
        jfieldID fMinDelay = (*env)->GetFieldID (env, cls, "minDelayUs", "I");
        jfieldID fMaxDelay = (*env)->GetFieldID (env, cls, "maxDelayUs", "I");
        jfieldID fResolution = (*env)->GetFieldID (env, cls, "resolution", "F");
        jfieldID fMaxRange = (*env)->GetFieldID (env, cls, "maxRange", "F");
        jfieldID fReportingMode = (*env)->GetFieldID (env, cls, "reportingMode", "I");
        jfieldID fWakeUp = (*env)->GetFieldID (env, cls, "wakeUp", "I");
        jfieldID fValueCount = (*env)->GetFieldID (env, cls, "valueCount", "I");

        if (fType)
                (*env)->SetIntField (env, out, fType, sensr_to_java_type (info->type));
        if (fMinDelay)
                (*env)->SetIntField (env, out, fMinDelay, (jint) info->minDelayUs);
        if (fMaxDelay)
                (*env)->SetIntField (env, out, fMaxDelay, (jint) info->maxDelayUs);
        if (fResolution)
                (*env)->SetFloatField (env, out, fResolution, (jfloat) info->resolution);
        if (fMaxRange)
                (*env)->SetFloatField (env, out, fMaxRange, (jfloat) info->maxRange);
        if (fReportingMode)
                (*env)->SetIntField (env, out, fReportingMode, (jint) info->reportingMode);
        if (fWakeUp)
                (*env)->SetIntField (env, out, fWakeUp, (jint) info->wakeUp);
        if (fValueCount)
                (*env)->SetIntField (env, out, fValueCount, (jint) info->valueCount);

        if (fName)
                {
                        jstring name = (*env)->NewStringUTF (env, info->name);
                        if (name)
                                {
                                        (*env)->SetObjectField (env, out, fName, name);
                                        (*env)->DeleteLocalRef (env, name);
                                }
                }
        if (fVendor)
                {
                        jstring vendor = (*env)->NewStringUTF (env, info->vendor);
                        if (vendor)
                                {
                                        (*env)->SetObjectField (env, out, fVendor, vendor);
                                        (*env)->DeleteLocalRef (env, vendor);
                                }
                }

        (*env)->DeleteLocalRef (env, cls);
}

JNIEXPORT jboolean JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeInitRuntime (JNIEnv *env, jclass clazz, jint heapSizeBytes)
{
        (void) env;
        (void) clazz;

        if (heapSizeBytes <= 0)
                heapSizeBytes = 8 * 1024 * 1024;

        int rc = ZZS_std_heap_init ((size_t) heapSizeBytes);
        if (rc != ZZS_std_OK && rc != ZZS_std_ERROR_HEAP_ALREADY_INITIALIZED)
                {
                        LOGE ("ZZS_std_heap_init failed: %d", rc);
                        return JNI_FALSE;
                }

        rc = ZZS_std_log_init ();
        if (rc != ZZS_std_OK)
                {
                        LOGE ("ZZS_std_log_init failed: %d", rc);
                        return JNI_FALSE;
                }

        ZZS_std_log_set_min_level (ZZS_LOG_LEVEL_DEBUG);
        return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeShutdownRuntime (JNIEnv *env, jclass clazz)
{
        (void) env;
        (void) clazz;

        ZZS_std_log_shutdown ();
        ZZS_std_heap_shutdown ();
}

JNIEXPORT jlong JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeCreate (JNIEnv *env, jclass clazz, jint backendKind)
{
        (void) env;
        (void) clazz;

        enum ZZS_sensr_backend_kind kind = java_to_backend_kind (backendKind);

        struct ZZS_sensr_backend_config config;
        ZZS_sensr_backend_config_init (&config);

        struct sensr_backend *backend = NULL;

        if (kind == ZZS_sensr_backend_ANDROID || kind == ZZS_sensr_backend_UNKNOWN)
                {
                        backend = ZZS_sensr_android_backend_create (&config);
                        if (backend)
                                LOGI ("Android NDK backend created");
                        else
                                LOGW ("Android NDK backend unavailable");
                }

        /* Automatic mode falls back to the deterministic mock backend, an
         * explicit Android request does not, so tests can detect real data. */
        if (!backend && kind == ZZS_sensr_backend_UNKNOWN)
                {
                        backend = sensr_mock_backend_create (&config);
                        if (backend)
                                LOGI ("Mock backend created (fallback)");
                }

        if (!backend)
                {
                        LOGE ("failed to create sensor backend (kind=%d)", (int) kind);
                        return 0;
                }

        sensr_jni_state_t *state = (sensr_jni_state_t *) ZZS_std_heap_alloc (sizeof (sensr_jni_state_t));
        if (!state)
                {
                        ZZS_sensr_backend_destroy (backend);
                        LOGE ("failed to allocate JNI state");
                        return 0;
                }

        memset (state, 0, sizeof (*state));
        state->backend = backend;

        return (jlong) (uintptr_t) state;
}

JNIEXPORT void JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeDestroy (JNIEnv *env, jclass clazz, jlong handle)
{
        (void) env;
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state)
                return;

        if (state->running)
                (void) ZZS_sensr_backend_stop (state->backend);

        ZZS_sensr_backend_destroy (state->backend);
        ZZS_std_heap_free (state);
}

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeBackendKind (JNIEnv *env, jclass clazz, jlong handle)
{
        (void) env;
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state)
                return (jint) ZZS_sensr_backend_UNKNOWN;

        return (jint) ZZS_sensr_backend_get_kind (state->backend);
}

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeEnableSensor (JNIEnv *env, jclass clazz, jlong handle, jint sensorType,
                                                        jint rateHz)
{
        (void) env;
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state)
                return SENSR_ERROR_INVALID_ARGUMENT;

        enum sensr_sensor_type type = java_to_sensr_type (sensorType);
        if (type == SENSR_SENSOR_UNKNOWN)
                return SENSR_ERROR_INVALID_ARGUMENT;

        int rc = ZZS_sensr_backend_enable (state->backend, type, (uint32_t) rateHz);
        if (rc == SENSR_OK)
                state->sensorsEnabled = 1;
        return rc;
}

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeDisableSensor (JNIEnv *env, jclass clazz, jlong handle, jint sensorType)
{
        (void) env;
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state)
                return SENSR_ERROR_INVALID_ARGUMENT;

        enum sensr_sensor_type type = java_to_sensr_type (sensorType);
        if (type == SENSR_SENSOR_UNKNOWN)
                return SENSR_ERROR_INVALID_ARGUMENT;

        return ZZS_sensr_backend_disable (state->backend, type);
}

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeStart (JNIEnv *env, jclass clazz, jlong handle)
{
        (void) env;
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state)
                return SENSR_ERROR_INVALID_ARGUMENT;

        if (state->running)
                return SENSR_OK;

        int rc = ZZS_sensr_backend_start (state->backend);
        if (rc == SENSR_OK)
                state->running = 1;
        return rc;
}

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeStop (JNIEnv *env, jclass clazz, jlong handle)
{
        (void) env;
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state)
                return SENSR_ERROR_INVALID_ARGUMENT;

        if (!state->running)
                return SENSR_OK;

        int rc = ZZS_sensr_backend_stop (state->backend);
        if (rc == SENSR_OK)
                state->running = 0;
        return rc;
}

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeReadEvent (JNIEnv *env, jclass clazz, jlong handle, jint timeoutMs,
                                                     jobject outEvent)
{
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state || !outEvent)
                return SENSR_ERROR_INVALID_ARGUMENT;

        struct sensr_event event;
        int rc = ZZS_sensr_backend_read (state->backend, &event, (int) timeoutMs);
        if (rc == SENSR_OK)
                fill_event (env, outEvent, &event);
        return rc;
}

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativePollEvents (JNIEnv *env, jclass clazz, jlong handle, jobjectArray outEvents,
                                                      jint maxCount)
{
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state || !outEvents || maxCount <= 0)
                return SENSR_ERROR_INVALID_ARGUMENT;

        jsize arrayLength = (*env)->GetArrayLength (env, outEvents);
        if (maxCount > arrayLength)
                maxCount = arrayLength;

        struct sensr_event stackEvents[32];
        struct sensr_event *events = stackEvents;
        int heapAllocated = 0;

        if (maxCount > (jint) (sizeof (stackEvents) / sizeof (stackEvents[0])))
                {
                        events = (struct sensr_event *) ZZS_std_heap_alloc ((size_t) maxCount * sizeof (*events));
                        if (!events)
                                return SENSR_ERROR_OUT_OF_MEMORY;
                        heapAllocated = 1;
                }

        size_t count = 0;
        int rc = ZZS_sensr_backend_poll (state->backend, events, (size_t) maxCount, &count);
        if (rc != SENSR_OK)
                goto done;

        if (count > (size_t) maxCount)
                count = (size_t) maxCount;

        for (size_t i = 0; i < count; ++i)
                {
                        jobject item = (*env)->GetObjectArrayElement (env, outEvents, (jsize) i);
                        if (item)
                                {
                                        fill_event (env, item, &events[i]);
                                        (*env)->DeleteLocalRef (env, item);
                                }
                }

        rc = (int) count;

done:
        if (heapAllocated)
                ZZS_std_heap_free (events);
        return rc;
}

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeAvailable (JNIEnv *env, jclass clazz, jlong handle)
{
        (void) env;
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state)
                return SENSR_ERROR_INVALID_ARGUMENT;

        return (jint) ZZS_sensr_backend_available (state->backend);
}

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeFlush (JNIEnv *env, jclass clazz, jlong handle)
{
        (void) env;
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state)
                return SENSR_ERROR_INVALID_ARGUMENT;

        return ZZS_sensr_backend_flush (state->backend);
}

JNIEXPORT void JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeGetStats (JNIEnv *env, jclass clazz, jlong handle, jlongArray outStats)
{
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state || !outStats)
                return;

        struct sensr_stats stats;
        if (ZZS_sensr_backend_get_stats (state->backend, &stats) != SENSR_OK)
                return;

        jlong values[4] = {(jlong) stats.enqueued, (jlong) stats.dropped, (jlong) stats.dequeued,
                           (jlong) stats.produced};
        jsize length = (*env)->GetArrayLength (env, outStats);
        if (length > 4)
                length = 4;
        (*env)->SetLongArrayRegion (env, outStats, 0, length, values);
}

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeSensorCount (JNIEnv *env, jclass clazz, jlong handle)
{
        (void) env;
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state)
                return 0;

        return (jint) ZZS_sensr_backend_sensor_count (state->backend);
}

JNIEXPORT jint JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeGetSensors (JNIEnv *env, jclass clazz, jlong handle,
                                                      jobjectArray outSensors)
{
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state || !outSensors)
                return SENSR_ERROR_INVALID_ARGUMENT;

        jsize length = (*env)->GetArrayLength (env, outSensors);
        jint filled = 0;

        for (jsize i = 0; i < length; ++i)
                {
                        struct sensr_sensor_info info;
                        if (ZZS_sensr_backend_sensor_at (state->backend, (size_t) i, &info) != SENSR_OK)
                                break;

                        jobject item = (*env)->GetObjectArrayElement (env, outSensors, i);
                        if (!item)
                                continue;

                        fill_sensor_info (env, item, &info);
                        (*env)->DeleteLocalRef (env, item);
                        ++filled;
                }

        return filled;
}

JNIEXPORT jboolean JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeHasSensor (JNIEnv *env, jclass clazz, jlong handle, jint sensorType)
{
        (void) env;
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state)
                return JNI_FALSE;

        enum sensr_sensor_type type = java_to_sensr_type (sensorType);
        if (type == SENSR_SENSOR_UNKNOWN)
                return JNI_FALSE;

        return ZZS_sensr_backend_has_sensor (state->backend, type) ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_com_openzzs_sensr_SensrBridge_nativeSetDefaultRate (JNIEnv *env, jclass clazz, jlong handle, jint rateHz)
{
        (void) env;
        (void) clazz;

        sensr_jni_state_t *state = get_state (handle);
        if (!state)
                return;

        state->backend->config.sampleRateHz = (uint32_t) rateHz;
}
