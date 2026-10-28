package com.openzzs;

import java.util.Locale;

/**
 * Static description of a sensor exposed by the native backend.
 */
public final class SensrSensorInfo {
    /** Sensor type, matching the {@code SensrBridge.SENSOR_*} constants. */
    public int type;
    public String name;
    public String vendor;
    /** Minimum sampling period in microseconds (0 when unknown). */
    public int minDelayUs;
    /** Maximum sampling period in microseconds (0 when unknown). */
    public int maxDelayUs;
    public float resolution;
    public float maxRange;
    /** Reporting mode, matching the {@code SensrBridge.REPORTING_*} constants. */
    public int reportingMode;
    /** Non-zero for wake-up sensors. */
    public int wakeUp;
    /** Number of meaningful values per sample. */
    public int valueCount;

    public String typeName() {
        return SensrBridge.sensorName(type);
    }

    public String toDisplayString() {
        return String.format(Locale.US,
                "%-24s name=\"%s\" vendor=\"%s\" minDelay=%dus mode=%d wake=%d values=%d",
                typeName(), name, vendor, minDelayUs, reportingMode, wakeUp, valueCount);
    }

    @Override
    public String toString() {
        return toDisplayString();
    }
}
