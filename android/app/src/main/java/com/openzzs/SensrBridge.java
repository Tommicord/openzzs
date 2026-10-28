package com.openzzs;

/**
 * Java facade over the native sensr sensor backend.
 *
 * <p>The native library owns a std heap and a background sampler thread. Call
 * {@link #initRuntime(int)} once before constructing a bridge, and
 * {@link #shutdownRuntime()} when the process is done with sensors.
 *
 * <p>Instances are not thread-safe; all calls should happen on the same thread
 * (the UI thread in the sample {@link MainActivity}).
 */
public final class SensrBridge implements AutoCloseable {
    static {
        System.loadLibrary("openzzs-jni");
    }

    // Backend kinds (mirror enum ZZS_sensr_backend_kind).
    public static final int BACKEND_UNKNOWN = 0;
    public static final int BACKEND_MOCK = 1;
    public static final int BACKEND_ANDROID = 2;
    public static final int BACKEND_IOS = 3;
    /** Let the native layer choose the platform backend, falling back to mock. */
    public static final int BACKEND_DEFAULT = BACKEND_UNKNOWN;

    // Sensor types (mirror enum sensr_sensor_type).
    public static final int SENSOR_UNKNOWN = 0;
    public static final int SENSOR_ACCELEROMETER = 1;
    public static final int SENSOR_GYROSCOPE = 2;
    public static final int SENSOR_MAGNETOMETER = 3;
    public static final int SENSOR_LIGHT = 4;
    public static final int SENSOR_PROXIMITY = 5;
    public static final int SENSOR_PRESSURE = 6;
    public static final int SENSOR_TEMPERATURE = 7;
    public static final int SENSOR_HUMIDITY = 8;
    public static final int SENSOR_GRAVITY = 9;
    public static final int SENSOR_LINEAR_ACCELERATION = 10;
    public static final int SENSOR_ROTATION_VECTOR = 11;
    public static final int SENSOR_GAME_ROTATION_VECTOR = 12;
    public static final int SENSOR_MAGNETIC_FIELD_UNCALIBRATED = 13;
    public static final int SENSOR_GYROSCOPE_UNCALIBRATED = 14;
    public static final int SENSOR_STEP_COUNTER = 15;
    public static final int SENSOR_STEP_DETECTOR = 16;
    public static final int SENSOR_SIGNIFICANT_MOTION = 17;
    public static final int SENSOR_HEART_RATE = 18;
    public static final int SENSOR_SOUND_LEVEL = 19;
    public static final int SENSOR_BATTERY_LEVEL = 20;

    // Accuracy values (mirror enum sensr_accuracy).
    public static final int ACCURACY_NO_CONTACT = -1;
    public static final int ACCURACY_UNRELIABLE = 0;
    public static final int ACCURACY_LOW = 1;
    public static final int ACCURACY_MEDIUM = 2;
    public static final int ACCURACY_HIGH = 3;

    // Reporting modes (mirror enum sensr_reporting_mode).
    public static final int REPORTING_INVALID = -1;
    public static final int REPORTING_CONTINUOUS = 0;
    public static final int REPORTING_ON_CHANGE = 1;
    public static final int REPORTING_ONE_SHOT = 2;
    public static final int REPORTING_SPECIAL_TRIGGER = 3;

    private static boolean runtimeInitialized = false;

    private long handle;

    /**
     * Initialize the native std runtime (heap + log). Idempotent.
     *
     * @param heapSizeBytes Size of the native heap; pass 0 for the 8 MiB default.
     * @return true on success.
     */
    public static synchronized boolean initRuntime(int heapSizeBytes) {
        if (runtimeInitialized) {
            return true;
        }
        runtimeInitialized = nativeInitRuntime(heapSizeBytes);
        return runtimeInitialized;
    }

    public static synchronized boolean isRuntimeInitialized() {
        return runtimeInitialized;
    }

    /** Release the native runtime. No bridge may be in use afterwards. */
    public static synchronized void shutdownRuntime() {
        if (runtimeInitialized) {
            nativeShutdownRuntime();
            runtimeInitialized = false;
        }
    }

    /** Create a bridge using the automatic backend selection. */
    public SensrBridge() {
        this(BACKEND_DEFAULT);
    }

    public SensrBridge(int backendKind) {
        if (!runtimeInitialized) {
            throw new IllegalStateException("SensrBridge.initRuntime() must be called first");
        }
        handle = nativeCreate(backendKind);
        if (handle == 0) {
            throw new IllegalStateException("native sensor backend creation failed (kind=" + backendKind + ")");
        }
    }

    public boolean isValid() {
        return handle != 0;
    }

    public int backendKind() {
        return handle == 0 ? BACKEND_UNKNOWN : nativeBackendKind(handle);
    }

    public int sensorCount() {
        return handle == 0 ? 0 : nativeSensorCount(handle);
    }

    public SensrSensorInfo[] getSensors() {
        int count = sensorCount();
        SensrSensorInfo[] sensors = new SensrSensorInfo[count];
        for (int i = 0; i < count; i++) {
            sensors[i] = new SensrSensorInfo();
        }
        if (handle != 0 && count > 0) {
            nativeGetSensors(handle, sensors);
        }
        return sensors;
    }

    public boolean hasSensor(int sensorType) {
        return handle != 0 && nativeHasSensor(handle, sensorType);
    }

    public int enableSensor(int sensorType, int rateHz) {
        return handle == 0 ? -1 : nativeEnableSensor(handle, sensorType, rateHz);
    }

    public int disableSensor(int sensorType) {
        return handle == 0 ? -1 : nativeDisableSensor(handle, sensorType);
    }

    public int start() {
        return handle == 0 ? -1 : nativeStart(handle);
    }

    public int stop() {
        return handle == 0 ? -1 : nativeStop(handle);
    }

    public int readEvent(SensrEvent out, int timeoutMs) {
        return handle == 0 ? -1 : nativeReadEvent(handle, timeoutMs, out);
    }

    /** Poll up to {@code out.length} events. Returns the number written. */
    public int pollEvents(SensrEvent[] out) {
        return handle == 0 ? -1 : nativePollEvents(handle, out, out.length);
    }

    public int available() {
        return handle == 0 ? 0 : nativeAvailable(handle);
    }

    public int flush() {
        return handle == 0 ? -1 : nativeFlush(handle);
    }

    /** Returns [enqueued, dropped, dequeued, produced]. */
    public long[] getStats() {
        long[] stats = new long[4];
        if (handle != 0) {
            nativeGetStats(handle, stats);
        }
        return stats;
    }

    public void setDefaultRate(int rateHz) {
        if (handle != 0) {
            nativeSetDefaultRate(handle, rateHz);
        }
    }

    @Override
    public void close() {
        if (handle != 0) {
            nativeDestroy(handle);
            handle = 0;
        }
    }

    @Override
    protected void finalize() throws Throwable {
        try {
            close();
        } finally {
            super.finalize();
        }
    }

    // ---- Naming helpers (mirror the C string tables) ----

    public static String backendName(int kind) {
        switch (kind) {
            case BACKEND_MOCK: return "mock";
            case BACKEND_ANDROID: return "android";
            case BACKEND_IOS: return "ios";
            default: return "unknown";
        }
    }

    public static String sensorName(int type) {
        switch (type) {
            case SENSOR_ACCELEROMETER: return "accelerometer";
            case SENSOR_GYROSCOPE: return "gyroscope";
            case SENSOR_MAGNETOMETER: return "magnetometer";
            case SENSOR_LIGHT: return "light";
            case SENSOR_PROXIMITY: return "proximity";
            case SENSOR_PRESSURE: return "pressure";
            case SENSOR_TEMPERATURE: return "temperature";
            case SENSOR_HUMIDITY: return "humidity";
            case SENSOR_GRAVITY: return "gravity";
            case SENSOR_LINEAR_ACCELERATION: return "linear_acceleration";
            case SENSOR_ROTATION_VECTOR: return "rotation_vector";
            case SENSOR_GAME_ROTATION_VECTOR: return "game_rotation_vector";
            case SENSOR_MAGNETIC_FIELD_UNCALIBRATED: return "magnetic_field_uncalibrated";
            case SENSOR_GYROSCOPE_UNCALIBRATED: return "gyroscope_uncalibrated";
            case SENSOR_STEP_COUNTER: return "step_counter";
            case SENSOR_STEP_DETECTOR: return "step_detector";
            case SENSOR_SIGNIFICANT_MOTION: return "significant_motion";
            case SENSOR_HEART_RATE: return "heart_rate";
            case SENSOR_SOUND_LEVEL: return "sound_level";
            case SENSOR_BATTERY_LEVEL: return "battery_level";
            default: return "unknown";
        }
    }

    public static String accuracyName(int accuracy) {
        switch (accuracy) {
            case ACCURACY_NO_CONTACT: return "no_contact";
            case ACCURACY_UNRELIABLE: return "unreliable";
            case ACCURACY_LOW: return "low";
            case ACCURACY_MEDIUM: return "medium";
            case ACCURACY_HIGH: return "high";
            default: return "unknown";
        }
    }

    public static String errorName(int error) {
        switch (error) {
            case 0: return "OK";
            case -1: return "invalid_argument";
            case -2: return "out_of_memory";
            case -3: return "not_supported";
            case -4: return "not_found";
            case -5: return "not_initialized";
            case -6: return "already_running";
            case -7: return "not_running";
            case -8: return "queue_full";
            case -9: return "queue_empty";
            case -10: return "timeout";
            case -11: return "would_block";
            case -12: return "not_enabled";
            case -13: return "platform_failure";
            case -14: return "permission_denied";
            case -15: return "overflow";
            case -16: return "already_initialized";
            default: return "error(" + error + ")";
        }
    }

    // ---- Native entry points ----

    private static native boolean nativeInitRuntime(int heapSizeBytes);

    private static native void nativeShutdownRuntime();

    private static native long nativeCreate(int backendKind);

    private static native void nativeDestroy(long handle);

    private static native int nativeBackendKind(long handle);

    private static native int nativeSensorCount(long handle);

    private static native int nativeGetSensors(long handle, SensrSensorInfo[] outSensors);

    private static native boolean nativeHasSensor(long handle, int sensorType);

    private static native int nativeEnableSensor(long handle, int sensorType, int rateHz);

    private static native int nativeDisableSensor(long handle, int sensorType);

    private static native int nativeStart(long handle);

    private static native int nativeStop(long handle);

    private static native int nativeReadEvent(long handle, int timeoutMs, SensrEvent outEvent);

    private static native int nativePollEvents(long handle, SensrEvent[] outEvents, int maxCount);

    private static native int nativeAvailable(long handle);

    private static native int nativeFlush(long handle);

    private static native void nativeGetStats(long handle, long[] outStats);

    private static native void nativeSetDefaultRate(long handle, int rateHz);
}
