package com.openzzs;

import android.os.Bundle;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;

import java.util.Locale;

/**
 * Main activity for OpenZZS power outage detection.
 * Displays real-time sensor features and ML inference probabilities.
 */
public class SensrActivity extends AppCompatActivity implements SensrBridge.FeaturesListener {

    private static final String TAG = "SensrActivity";

    private TextView mStatusText;
    private TextView mFeatureText;
    private TextView mInferenceText;

    private SensrBridge mBridge;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mStatusText = findViewById(R.id.status_text);
        mFeatureText = findViewById(R.id.feature_text);
        mInferenceText = findViewById(R.id.inference_text);

        mBridge = new SensrBridge(this, this);
        if (!mBridge.init()) {
            mStatusText.setText("Init failed");
            return;
        }
        mStatusText.setText("Initialized");
        mBridge.start();
    }

    @Override
    protected void onDestroy() {
        if (mBridge != null) {
            mBridge.stop();
            mBridge.shutdown();
        }
        super.onDestroy();
    }

    @Override
    public void onFeatures(String[] features) {
        runOnUiThread(() -> {
            StringBuilder sb = new StringBuilder();
            for (String f : features) {
                sb.append(f).append("\n");
            }
            mFeatureText.setText(sb.toString());

            float[] probs = SensrBridge.softmaxFromFeatures(features);
            String inf = String.format(Locale.US,
                    "P_normal=%.3f  P_outage=%.3f  P_transition=%.3f",
                    probs[0], probs[1], probs[2]);
            mInferenceText.setText(inf);
        });
    }

    @Override
    public void onInference(float[] probabilities) {
        // Native inference pushes real probabilities here on every frame via the
        // onInference([F)V JNI callback (sensr_jni.cpp: _feed_grid_sample).
        runOnUiThread(() -> {
            if (probabilities == null || probabilities.length < 3) {
                return;
            }
            String inf = String.format(Locale.US,
                    "P_normal=%.3f  P_outage=%.3f  P_transition=%.3f",
                    probabilities[0], probabilities[1], probabilities[2]);
            mInferenceText.setText(inf);
        });
    }

    @Override
    public void onError(String message) {
        runOnUiThread(() -> mStatusText.setText("Error: " + message));
    }
}
