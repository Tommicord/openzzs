package com.openzzs.probe;

import android.app.NativeActivity;
import android.os.Bundle;

/** @brief Simple NativeActivity that loads the OpenZZS probe native library.
 *
 *  The native entry point ANativeActivity_onCreate in probe_main.cpp runs the
 *  magnetometer probe automatically when the activity starts.  Results are
 *  visible via:
 *    adb logcat -s OpenZZSProbe
 */
public class ProbeActivity extends NativeActivity {
    static {
        System.loadLibrary("openzzs_probe");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
    }
}
