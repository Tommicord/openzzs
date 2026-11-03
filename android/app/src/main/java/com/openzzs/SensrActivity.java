package com.openzzs;

import android.app.Activity;
import android.os.Bundle;
import android.util.Log;
import android.widget.TextView;

import java.util.Locale;

/**
 * Main activity for OpenZZS power outage detection.
 *
 * Minimal launcher: loads the native OpenZZS library, starts sensor sampling
 * and inference, and displays the live inference probabilities.  All heavy
 * lifting (sensor I/O, preprocessing, mllee inference) happens in native code
 * via SensrBridge.
 */
public class SensrActivity extends Activity implements SensrBridge.FeaturesListener {

  private static final String TAG = "SensrActivity";

  private TextView mStatusText;
  private TextView mInferenceText;
  private SensrBridge mBridge;

  @Override
  protected void onCreate (Bundle savedInstanceState) {
    super.onCreate (savedInstanceState);
    setContentView (R.layout.activity_main);

    mStatusText = findViewById (R.id.status_text);
    mInferenceText = findViewById (R.id.inference_text);

    Log.i (TAG, "onCreate: creating SensrBridge");
    mBridge = new SensrBridge (this, this);
    if (!mBridge.init ()) {
      mStatusText.setText ("Init failed");
      Log.e (TAG, "onCreate: SensrBridge init failed");
      return;
    }
    mStatusText.setText ("Initialized");
    Log.i (TAG, "onCreate: SensrBridge initialized, starting sensors");
    mBridge.start ();
  }

  @Override
  protected void onDestroy () {
    Log.i (TAG, "onDestroy: stopping SensrBridge");
    if (mBridge != null) {
      mBridge.stop ();
      mBridge.shutdown ();
    }
    super.onDestroy ();
  }

  @Override
  public void onFeatures (String[] features) {
    // Not used in the minimal launcher; kept for interface compliance.
  }

  @Override
  public void onInference (float[] probabilities) {
    runOnUiThread (() -> {
      if (probabilities == null || probabilities.length < 3) {
        Log.w (TAG, "onInference: received null or short probabilities array");
        return;
      }
      Log.i (TAG, "onInference UI: P_normal=" + String.format ("%.4f", probabilities[0]) +
              " P_outage=" + String.format ("%.4f", probabilities[1]) +
              " P_transition=" + String.format ("%.4f", probabilities[2]));
      String inf = String.format (Locale.US,
          "P_normal=%.3f  P_outage=%.3f  P_transition=%.3f",
          probabilities[0], probabilities[1], probabilities[2]);
      mInferenceText.setText (inf);
    });
  }

  @Override
  public void onError (String message) {
    Log.e (TAG, "onError: " + message);
    runOnUiThread (() -> mStatusText.setText ("Error: " + message));
  }
}
