package com.openzzs;

import android.content.Context;
import android.util.Log;

import com.opencsv.CSVWriter;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Locale;

/**
 * Collects real-world sensor samples into a labeled CSV file for model training.
 *
 * Flow:
 *   1. User selects a label once ({@link Label#NORMAL}, {@link Label#BLACKOUT},
 *      or {@link Label#TRANSITION}).
 *   2. Collector accumulates sensor samples until {@link #stop()} is called.
 *   3. On stop, writes a CSV to {@link Context#getFilesDir()}/openzzs_dataset_&lt;label&gt;_&lt;ts&gt;.csv
 *
 * CSV columns: timestamp_ns, channel, sample_q15, label
 *
 * No Android permissions are required: the file lives in the app-private
 * files/ directory.
 */
public class DatasetCollector {
    private static final String TAG = "DatasetCollector";

    /** Label chosen by the user before collection starts. */
    public enum Label {
        NORMAL,      // 0 — mains power present, normal conditions
        BLACKOUT,    // 1 — power outage / blackout
        TRANSITION   // 2 — power cutting or restoring
    }

    private final Context mContext;
    private final SensrBridge mBridge;

    private Label mCurrentLabel = null;
    private boolean mCollecting = false;
    private final List<String[]> mRows = new ArrayList<>();
    private long mStartTimeNs = 0;

    private CollectionListener mListener;

    public interface CollectionListener {
        void onCollectionStarted(Label label);
        void onCollectionStopped(File csvFile, int sampleCount);
        void onCollectionError(String message);
    }

    public DatasetCollector(Context context, SensrBridge bridge) {
        mContext = context.getApplicationContext();
        mBridge = bridge;
        // Hook into the bridge's sensor sample stream so we capture every axis.
        mBridge.setSampleListener(mSampleListener);
    }

    public void setListener(CollectionListener listener) {
        mListener = listener;
    }

    public boolean isCollecting() {
        return mCollecting;
    }

    public Label getCurrentLabel() {
        return mCurrentLabel;
    }

    /**
     * User selects the label once before starting.  Call this before {@link #start()}.
     */
    public void selectLabel(Label label) {
        mCurrentLabel = label;
        Log.i(TAG, "Label selected: " + label);
    }

    /** Begin accumulating samples.  Requires a label to have been selected. */
    public void start() {
        if (mCurrentLabel == null) {
            Log.e(TAG, "start() called with no label selected");
            if (mListener != null) mListener.onCollectionError("No label selected");
            return;
        }
        if (mCollecting) {
            Log.w(TAG, "Already collecting");
            return;
        }
        mCollecting = true;
        mRows.clear();
        mStartTimeNs = System.nanoTime();
        Log.i(TAG, "Collection started: label=" + mCurrentLabel);
        if (mListener != null) mListener.onCollectionStarted(mCurrentLabel);
    }

    /** Stop collection and write the CSV file. */
    public void stop() {
        if (!mCollecting) {
            Log.w(TAG, "stop() called but not collecting");
            return;
        }
        mCollecting = false;

        File csvFile = writeCsv();
        Log.i(TAG, "Collection stopped: " + mRows.size() + " samples -> " + csvFile);
        if (mListener != null) mListener.onCollectionStopped(csvFile, mRows.size());
    }

    /**
     * Called by SensrBridge for every sensor sample.  Runs on the UI thread
     * (sensor callback thread), so we only append to a list — no file I/O here.
     */
    public void onSample(int channel, short sampleQ15, long timestampNs) {
        if (!mCollecting || mCurrentLabel == null) return;
        long relativeNs = timestampNs - mStartTimeNs;
        mRows.add(new String[]{
            String.valueOf(relativeNs),
            String.valueOf(channel),
            String.valueOf(sampleQ15),
            String.valueOf(mCurrentLabel.ordinal())
        });
    }

    private File writeCsv() {
        File dir = mContext.getFilesDir();
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        File csvFile = new File(dir,
                "openzzs_dataset_" + mCurrentLabel.name().toLowerCase(Locale.ROOT) + "_" + timestamp + ".csv");

        try (CSVWriter writer = new CSVWriter(new FileWriter(csvFile))) {
            writer.writeNext(new String[]{"timestamp_ns", "channel", "sample_q15", "label"});
            for (String[] row : mRows) {
                writer.writeNext(row);
            }
            writer.flush();
        } catch (IOException e) {
            Log.e(TAG, "Failed to write CSV dataset", e);
            if (mListener != null) mListener.onCollectionError(e.getMessage());
        }
        return csvFile;
    }

    /* ------------------------------------------------------------------ */
    /* Bridge sample listener                                             */
    /* ------------------------------------------------------------------ */
    private final SensrBridge.SampleListener mSampleListener =
            new SensrBridge.SampleListener() {
                @Override
                public void onSample(int channel, short sampleQ15, long timestampNs) {
                    DatasetCollector.this.onSample(channel, sampleQ15, timestampNs);
                }
            };
}
