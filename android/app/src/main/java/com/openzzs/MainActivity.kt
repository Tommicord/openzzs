package com.openzzs

import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.widget.Button
import android.widget.ScrollView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.util.Locale

/**
 * Minimal host activity that exercises the real Android sensor backend.
 *
 * It performs three jobs only:
 *  1. initializes the native runtime and creates the NDK sensor backend,
 *  2. enables the sensors that are actually present on the device,
 *  3. polls the native event queue and prints every sample it receives.
 *
 * No data is persisted or uploaded; the activity is a live sensor feed used to
 * validate the C backend against real hardware.
 */
class MainActivity : AppCompatActivity() {

    private companion object {
        const val HEAP_SIZE_BYTES = 8 * 1024 * 1024
        const val DEFAULT_RATE_HZ = 50
        const val POLL_BATCH = 64
        const val POLL_INTERVAL_MS = 100L
        const val MAX_LOG_CHARS = 24_000
        const val TRIM_KEEP_CHARS = 12_000
    }

    private var bridge: SensrBridge? = null

    private lateinit var logView: TextView
    private lateinit var logScroll: ScrollView
    private lateinit var statusView: TextView

    private val handler = Handler(Looper.getMainLooper())
    private val eventBuffer = Array(POLL_BATCH) { SensrEvent() }

    @Volatile
    private var polling = false

    private val pollTask = object : Runnable {
        override fun run() {
            if (!polling) {
                return
            }
            val activeBridge = bridge ?: return

            val count = activeBridge.pollEvents(eventBuffer)
            if (count > 0) {
                val builder = StringBuilder()
                for (i in 0 until count) {
                    builder.append(eventBuffer[i].toDisplayString()).append('\n')
                }
                appendLog(builder.toString())
            }

            updateStatus(activeBridge)
            handler.postDelayed(this, POLL_INTERVAL_MS)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        logView = findViewById(R.id.logView)
        logScroll = findViewById(R.id.logScroll)
        statusView = findViewById(R.id.statusView)

        findViewById<Button>(R.id.btnStart).setOnClickListener { startSampling() }
        findViewById<Button>(R.id.btnStop).setOnClickListener { stopSampling() }
        findViewById<Button>(R.id.btnFlush).setOnClickListener { flushQueue() }
        findViewById<Button>(R.id.btnSensors).setOnClickListener { listSensors() }

        try {
            if (!SensrBridge.initRuntime(HEAP_SIZE_BYTES)) {
                appendLog("ERROR: native runtime initialization failed")
                return
            }

            appendLog("runtime initialized (${HEAP_SIZE_BYTES / (1024 * 1024)} MiB heap)")

            val created = SensrBridge(SensrBridge.BACKEND_DEFAULT)
            bridge = created
            appendLog("backend: ${SensrBridge.backendName(created.backendKind())}")
            listSensors()
            enableDefaultSensors(created)
            startSampling()
        } catch (error: Throwable) {
            // Covers a missing JNI library as well as backend creation failures.
            appendLog("ERROR: ${error.message ?: error.javaClass.simpleName}")
        }
    }

    override fun onDestroy() {
        stopSampling()
        bridge?.close()
        bridge = null

        // The runtime is process-wide; only tear it down when the activity is
        // really going away rather than being recreated for a config change.
        if (isFinishing) {
            SensrBridge.shutdownRuntime()
        }

        super.onDestroy()
    }

    private fun enableDefaultSensors(activeBridge: SensrBridge) {
        val wanted = intArrayOf(
            SensrBridge.SENSOR_ACCELEROMETER,
            SensrBridge.SENSOR_GYROSCOPE,
            SensrBridge.SENSOR_MAGNETOMETER,
            SensrBridge.SENSOR_ROTATION_VECTOR,
            SensrBridge.SENSOR_GRAVITY,
            SensrBridge.SENSOR_LINEAR_ACCELERATION,
            SensrBridge.SENSOR_LIGHT,
            SensrBridge.SENSOR_PROXIMITY,
            SensrBridge.SENSOR_STEP_COUNTER,
        )

        for (type in wanted) {
            if (!activeBridge.hasSensor(type)) {
                continue
            }
            val result = activeBridge.enableSensor(type, DEFAULT_RATE_HZ)
            appendLog(
                String.format(
                    Locale.US,
                    "enable %-24s @ %d Hz -> %s",
                    SensrBridge.sensorName(type),
                    DEFAULT_RATE_HZ,
                    SensrBridge.errorName(result),
                ),
            )
        }
    }

    private fun listSensors() {
        val activeBridge = bridge ?: return
        val sensors = activeBridge.getSensors()
        appendLog("sensors (${sensors.size}):")
        for (sensor in sensors) {
            appendLog("  ${sensor.toDisplayString()}")
        }
    }

    private fun startSampling() {
        val activeBridge = bridge ?: return
        if (polling) {
            return
        }

        val result = activeBridge.start()
        appendLog("start sampler -> ${SensrBridge.errorName(result)}")
        if (result != 0) {
            return
        }

        polling = true
        handler.post(pollTask)
        appendLog("streaming sensor data...")
    }

    private fun stopSampling() {
        polling = false
        handler.removeCallbacks(pollTask)

        val activeBridge = bridge ?: return
        val result = activeBridge.stop()
        appendLog("stop sampler -> ${SensrBridge.errorName(result)}")
        updateStatus(activeBridge)
    }

    private fun flushQueue() {
        val activeBridge = bridge ?: return
        val result = activeBridge.flush()
        appendLog("flush -> ${SensrBridge.errorName(result)}")
        updateStatus(activeBridge)
    }

    private fun updateStatus(activeBridge: SensrBridge) {
        val stats = activeBridge.getStats()
        statusView.text = String.format(
            Locale.US,
            "backend=%s  sensors=%d  queued=%d  enqueued=%d  dropped=%d  dequeued=%d  produced=%d",
            SensrBridge.backendName(activeBridge.backendKind()),
            activeBridge.sensorCount(),
            activeBridge.available(),
            stats[0],
            stats[1],
            stats[2],
            stats[3],
        )
    }

    private fun appendLog(line: String) {
        if (line.isEmpty()) {
            return
        }

        logView.append(line.trimEnd())
        logView.append("\n")

        if (logView.text.length > MAX_LOG_CHARS) {
            val start = logView.text.length - TRIM_KEEP_CHARS
            logView.text = logView.text.subSequence(start, logView.text.length)
        }

        logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }
    }
}
