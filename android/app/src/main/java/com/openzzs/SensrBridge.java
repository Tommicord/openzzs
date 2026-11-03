package com.openzzs;

import android.content.Context;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorListener;
import android.hardware.SensorManager;
import android.os.Handler;
import android.os.Looper;
import android.util.Log;

/**
 * JNI bridge between the Android UI and the OpenZZS native preprocessing bank
 * + mllee inference core.
 */
public class SensrBridge {
    private static final String TAG = "SensrBridge";

    static {
        System.loadLibrary("openzzs-android");
        Log.i(TAG, "Loading native library openzzzs-android");
    }

    public interface FeaturesListener {
        void onFeatures(String[] features);
        void onInference(float[] probabilities);
        void onError(String message);
    }

    /** Callback for every raw sensor sample fed into the native pipeline. */
    public interface SampleListener {
        void onSample(int channel, short sampleQ15, long timestampNs);
    }

    private final FeaturesListener mListener;
    private final SensorManager mSensorManager;
    private final Handler mHandler = new Handler(Looper.getMainLooper());

    private SampleListener mSampleListener;

    private boolean mInitialized = false;
    private boolean mRunning = false;

    public SensrBridge(Context context, FeaturesListener listener) {
        mListener = listener;
        mSensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
    }

    public native int initNative();
    public native void shutdownNative();
    public native void feedSampleNative(int channel, short sampleQ15, long timestampNs);
    public native String[] getFeaturesNative();

    public boolean init() {
        Log.i(TAG, "init() called");
        int rc = initNative();
        Log.i(TAG, "initNative returned rc=" + rc + " (0=OK, model loaded from MLEE blob)");
        if (rc != 0) {
            Log.e(TAG, "initNative failed: " + rc);
            return false;
        }
        mInitialized = true;
        Log.i(TAG, "SensrBridge initialized successfully");
        return true;
    }

    public void shutdown() {
        Log.i(TAG, "shutdown() called");
        stop();
        shutdownNative();
        mInitialized = false;
        Log.i(TAG, "SensrBridge shutdown complete");
    }

    public void start() {
        if (!mInitialized) return;
        mRunning = true;
        registerSensors();
    }

    public void stop() {
        mRunning = false;
        unregisterSensors();
    }

    private void registerSensors() {
        if (mSensorManager == null) return;

        Sensor mag = mSensorManager.getDefaultSensor(Sensor.TYPE_MAGNETIC_FIELD);
        Sensor accel = mSensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        Sensor gyro = mSensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        Sensor light = mSensorManager.getDefaultSensor(Sensor.TYPE_LIGHT);

        Log.i(TAG, "registerSensors: mag=" + (mag != null ? mag.getName() : "null") +
                ", accel=" + (accel != null ? accel.getName() : "null") +
                ", gyro=" + (gyro != null ? gyro.getName() : "null") +
                ", light=" + (light != null ? light.getName() : "null"));

        SensorEventListener listener = new SensorEventListener() {
            private long _M_lastLogTs = 0;
            private int _M_sampleCount = 0;
            private final int[] _M_channelCounts = new int[12];

            @Override
            public void onSensorChanged(SensorEvent event) {
                if (!mRunning) return;
                long ts = event.timestamp;
                int channel;
                switch (event.sensor.getType()) {
                    case Sensor.TYPE_MAGNETIC_FIELD:
                        channel = 0;
                        feedSampleNative(channel, toQ15(event.values[0], 100f), ts); // MAG_X
                        feedSampleNative(channel + 1, toQ15(event.values[1], 100f), ts); // MAG_Y
                        feedSampleNative(channel + 2, toQ15(event.values[2], 100f), ts); // MAG_Z
                        if (mSampleListener != null) {
                            mSampleListener.onSample(channel, toQ15(event.values[0], 100f), ts);
                            mSampleListener.onSample(channel + 1, toQ15(event.values[1], 100f), ts);
                            mSampleListener.onSample(channel + 2, toQ15(event.values[2], 100f), ts);
                        }
                        _M_channelCounts[channel] += 3;
                        break;
                    case Sensor.TYPE_ACCELEROMETER:
                        channel = 3;
                        feedSampleNative(channel, toQ15(event.values[0], 20f), ts);  // ACCEL_X
                        feedSampleNative(channel + 1, toQ15(event.values[1], 20f), ts);  // ACCEL_Y
                        feedSampleNative(channel + 2, toQ15(event.values[2], 20f), ts);  // ACCEL_Z
                        if (mSampleListener != null) {
                            mSampleListener.onSample(channel, toQ15(event.values[0], 20f), ts);
                            mSampleListener.onSample(channel + 1, toQ15(event.values[1], 20f), ts);
                            mSampleListener.onSample(channel + 2, toQ15(event.values[2], 20f), ts);
                        }
                        _M_channelCounts[channel] += 3;
                        break;
                    case Sensor.TYPE_GYROSCOPE:
                        channel = 6;
                        feedSampleNative(channel, toQ15(event.values[0], 20f), ts);  // GYRO_X
                        feedSampleNative(channel + 1, toQ15(event.values[1], 20f), ts);  // GYRO_Y
                        feedSampleNative(channel + 2, toQ15(event.values[2], 20f), ts);  // GYRO_Z
                        if (mSampleListener != null) {
                            mSampleListener.onSample(channel, toQ15(event.values[0], 20f), ts);
                            mSampleListener.onSample(channel + 1, toQ15(event.values[1], 20f), ts);
                            mSampleListener.onSample(channel + 2, toQ15(event.values[2], 20f), ts);
                        }
                        _M_channelCounts[channel] += 3;
                        break;
                    case Sensor.TYPE_LIGHT:
                        channel = 9;
                        feedSampleNative(channel, toQ15(event.values[0], 10000f), ts); // LIGHT
                        if (mSampleListener != null) {
                            mSampleListener.onSample(channel, toQ15(event.values[0], 10000f), ts);
                        }
                        _M_channelCounts[channel]++;
                        break;
                    default:
                        break;
                }
                // Log a per-channel summary every ~2 seconds (throttled)
                if (ts - _M_lastLogTs > 2_000_000_000L) {
                    StringBuilder sb = new StringBuilder();
                    sb.append("Sensor feed: samples=").append(_M_sampleCount);
                    for (int i = 0; i < 12; i++) {
                        if (_M_channelCounts[i] > 0) {
                            sb.append(" ch=").append(i).append("=").append(_M_channelCounts[i]);
                        }
                    }
                    Log.i(TAG, sb.toString());
                    _M_lastLogTs = ts;
                    _M_sampleCount = 0;
                    java.util.Arrays.fill(_M_channelCounts, 0);
                }
            }

            @Override
            public void onAccuracyChanged(Sensor sensor, int accuracy) {
                // not used
            }
        };

        if (mag != null) mSensorManager.registerListener(listener, mag, SensorManager.SENSOR_DELAY_FASTEST);
        if (accel != null) mSensorManager.registerListener(listener, accel, SensorManager.SENSOR_DELAY_FASTEST);
        if (gyro != null) mSensorManager.registerListener(listener, gyro, SensorManager.SENSOR_DELAY_FASTEST);
        if (light != null) mSensorManager.registerListener(listener, light, SensorManager.SENSOR_DELAY_NORMAL);
    }

    private void unregisterSensors() {
        if (mSensorManager != null) {
            mSensorManager.unregisterListener((SensorListener) mListener);
        }
    }

    private static short toQ15(float value, float scale) {
        float clamped = value / scale;
        if (clamped > 1.0f) clamped = 1.0f;
        if (clamped < -1.0f) clamped = -1.0f;
        return (short) (clamped * 32767.0f);
    }

    /**
     * Call from the UI thread to pull the latest feature row.
     * Returns an array of "name=value" strings, or null if not ready.
     */
    public String[] pollFeatures() {
        if (!mInitialized) return null;
        String[] features = getFeaturesNative();
        if (features == null || features.length == 0) return null;
        if (mListener != null) mListener.onFeatures(features);
        return features;
    }

    public boolean isRunning() {
        return mRunning;
    }

    /** Set a listener to receive every raw sensor sample (channel, Q15, timestamp). */
    public void setSampleListener(SampleListener listener) {
        this.mSampleListener = listener;
    }

    // Callback methods for JNI to invoke — delegates to the registered listener
    public void onFeatures (String[] features) {
        if (mListener != null) mListener.onFeatures (features);
    }

    public void onInference (float[] probabilities) {
        if (mListener != null) {
            if (probabilities != null && probabilities.length >= 3) {
                Log.i(TAG, "onInference: P_normal=" + String.format("%.4f", probabilities[0]) +
                        " P_outage=" + String.format("%.4f", probabilities[1]) +
                        " P_transition=" + String.format("%.4f", probabilities[2]));
            }
            mListener.onInference (probabilities);
        }
    }

    public void onError (String message) {
        if (mListener != null) mListener.onError (message);
    }
}
