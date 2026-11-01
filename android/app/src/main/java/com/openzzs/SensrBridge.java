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
    }

    public interface FeaturesListener {
        void onFeatures(String[] features);
        void onInference(float[] probabilities);
        void onError(String message);
    }

    private final FeaturesListener mListener;
    private final SensorManager mSensorManager;
    private final Handler mHandler = new Handler(Looper.getMainLooper());

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
        int rc = initNative();
        if (rc != 0) {
            Log.e(TAG, "initNative failed: " + rc);
            return false;
        }
        mInitialized = true;
        Log.i(TAG, "SensrBridge initialized");
        return true;
    }

    public void shutdown() {
        stop();
        shutdownNative();
        mInitialized = false;
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

        SensorEventListener listener = new SensorEventListener() {
            @Override
            public void onSensorChanged(SensorEvent event) {
                if (!mRunning) return;
                long ts = event.timestamp;
                switch (event.sensor.getType()) {
                    case Sensor.TYPE_MAGNETIC_FIELD:
                        feedSampleNative(0, toQ15(event.values[0], 100f), ts); // MAG_X
                        feedSampleNative(1, toQ15(event.values[1], 100f), ts); // MAG_Y
                        feedSampleNative(2, toQ15(event.values[2], 100f), ts); // MAG_Z
                        break;
                    case Sensor.TYPE_ACCELEROMETER:
                        feedSampleNative(3, toQ15(event.values[0], 20f), ts);  // ACCEL_X
                        feedSampleNative(4, toQ15(event.values[1], 20f), ts);  // ACCEL_Y
                        feedSampleNative(5, toQ15(event.values[2], 20f), ts);  // ACCEL_Z
                        break;
                    case Sensor.TYPE_GYROSCOPE:
                        feedSampleNative(6, toQ15(event.values[0], 20f), ts);  // GYRO_X
                        feedSampleNative(7, toQ15(event.values[1], 20f), ts);  // GYRO_Y
                        feedSampleNative(8, toQ15(event.values[2], 20f), ts);  // GYRO_Z
                        break;
                    case Sensor.TYPE_LIGHT:
                        feedSampleNative(9, toQ15(event.values[0], 10000f), ts); // LIGHT
                        break;
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

    /**
     * Convert Q15 feature codes to probabilities for the 3 classes:
     * [P_normal, P_outage, P_transition].
     * The last 3 features (indices 32-34) are the acoustic features which
     * include sound_level_dB used as a simple proxy.
     */
    public static float[] softmaxFromFeatures(String[] features) {
        // Simple heuristic: use the last 3 feature values as logits
        float[] logits = new float[3];
        try {
            logits[0] = Float.parseFloat(features[features.length - 3].split("=")[1]);
            logits[1] = Float.parseFloat(features[features.length - 2].split("=")[1]);
            logits[2] = Float.parseFloat(features[features.length - 1].split("=")[1]);
        } catch (Exception e) {
            return new float[]{1.0f, 0.0f, 0.0f};
        }

        // Softmax
        float max = logits[0];
        for (float l : logits) if (l > max) max = l;
        float sum = 0f;
        for (int i = 0; i < 3; i++) {
            logits[i] = (float) Math.exp(logits[i] - max);
            sum += logits[i];
        }
        for (int i = 0; i < 3; i++) logits[i] /= sum;
        return logits;
    }

    public boolean isRunning() {
        return mRunning;
    }

    // Callback methods for JNI to invoke — delegates to the registered listener
    public void onFeatures (String[] features) {
        if (mListener != null) mListener.onFeatures (features);
    }

    public void onInference (float[] probabilities) {
        if (mListener != null) mListener.onInference (probabilities);
    }

    public void onError (String message) {
        if (mListener != null) mListener.onError (message);
    }
}
