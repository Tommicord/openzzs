package com.openzzs;

import java.util.Locale;

/**
 * A single sensor sample delivered by the native sensr backend.
 *
 * Instances are reused by {@link SensrBridge#pollEvents(SensrEvent[])} to avoid
 * allocating on every frame/sample.
 */
public final class SensrEvent {
    public static final int MAX_VALUES = 16;

    /** Sensor type, matching the {@code SensrBridge.SENSOR_*} constants. */
    public int type;
    /** Monotonic timestamp in nanoseconds. */
    public long timestampNs;
    /** Accuracy, matching the {@code SensrBridge.ACCURACY_*} constants. */
    public int accuracy;
    /** Number of meaningful entries in {@link #values}. */
    public int valueCount;
    /** Backend that produced the sample, matching {@code SensrBridge.BACKEND_*}. */
    public int source;
    /** Monotonic per-backend sequence number. */
    public long sequence;

    /** Sensor payload. Unused entries are left untouched. */
    public float[] values = new float[MAX_VALUES];

    public String typeName() {
        return SensrBridge.sensorName(type);
    }

    public String accuracyName() {
        return SensrBridge.accuracyName(accuracy);
    }

    public String toDisplayString() {
        StringBuilder builder = new StringBuilder(96);
        builder.append(String.format(Locale.US, "%-24s t=%d acc=%s", typeName(), timestampNs, accuracyName()));
        int count = Math.min(valueCount, values.length);
        for (int i = 0; i < count; i++) {
            builder.append(String.format(Locale.US, " %.4f", values[i]));
        }
        return builder.toString();
    }

    @Override
    public String toString() {
        return toDisplayString();
    }
}
