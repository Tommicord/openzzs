/* Copyright(c) 2026 Tommicord
 *
 * MIT License
 *
 * Magnetometer mains-harmonics probe for Android.
 *
 * Reads ASENSOR_TYPE_MAGNETIC_FIELD via the NDK sensor API and measures how
 * much energy sits in the 50/60 Hz mains band (and its first harmonics)
 * compared to the broadband noise floor of the sensor.
 *
 * Build:
 *   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android36-clang++ \
 *       -std=gnu++2b -O2 -DANDROID tools/mag_probe/mag_probe.cpp -o mag_probe -llog -landroid
 *   adb push mag_probe /data/local/tmp/ && adb shell chmod +x /data/local/tmp/mag_probe
 *   adb shell LD_LIBRARY_PATH=/data/local/tmp /data/local/tmp/mag_probe
 */

#include <android/log.h>
#include <android/looper.h>
#include <android/sensor.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>

#define TAG "MagProbe"
#define LOGI(...) __android_log_print (ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print (ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

constexpr uint32_t _S_window = 1600; /* samples per axis */
constexpr float _S_pi = 3.14159265358979323846f;

/* Direct DFT power at `target_hz` for a real, DC-removed signal.
 * Works even when target_hz lands on the Nyquist bin (fs/2), where the
 * Goertzel recurrence degenerates.  Returns mean-square amplitude. */
float
dft_power (const float *samples, uint32_t n, float sample_rate_hz, float target_hz) noexcept {
    if (n == 0u || sample_rate_hz <= 0.f)
        return 0.f;
    float re = 0.f;
    float im = 0.f;
    const float w = 2.f * _S_pi * target_hz / sample_rate_hz;
    for (uint32_t i = 0; i < n; ++i) {
        const float phase = w * static_cast<float> (i);
        re += samples[i] * cosf (phase);
        im += samples[i] * sinf (phase);
    }
    const float mag = sqrtf (re * re + im * im);
    /* Single-sided amplitude^2/2 == mean square power of that component. */
    return (mag * mag) / (2.f * static_cast<float> (n) * static_cast<float> (n));
}

float
rms (const float *samples, uint32_t n) noexcept {
    float sum = 0.f;
    for (uint32_t i = 0; i < n; ++i)
        sum += samples[i] * samples[i];
    return sqrtf (sum / static_cast<float> (n));
}

/* Average DFT power over a band [lo, hi] using 1 Hz spaced bins.  Used as a
 * broadband noise-floor estimate that excludes the mains bins. */
float
band_floor_power (const float *samples, uint32_t n, float sample_rate_hz, float lo, float hi) noexcept {
    float sum = 0.f;
    uint32_t bins = 0u;
    for (float f = lo; f <= hi; f += 1.f) {
        /* skip the mains bins we are measuring */
        if ((f > 48.f && f < 52.f) || (f > 58.f && f < 62.f))
            continue;
        sum += dft_power (samples, n, sample_rate_hz, f);
        ++bins;
    }
    return bins == 0u ? 0.f : sum / static_cast<float> (bins);
}

ASensorManager *
open_manager () noexcept {
    /* Package-scoped instance: a bare shell process has no attribution source on
     * modern Android, which silently yields zero sensor events. */
    ASensorManager *manager = ASensorManager_getInstanceForPackage ("com.openzzs");
    if (manager == nullptr)
        manager = ASensorManager_getInstance ();
    return manager;
}

} // namespace

int
main () {
    ASensorManager *const manager = open_manager ();
    if (manager == nullptr) {
        LOGE ("ASensorManager unavailable");
        return 1;
    }

    const ASensor *const mag = ASensorManager_getDefaultSensor (manager, ASENSOR_TYPE_MAGNETIC_FIELD);
    if (mag == nullptr) {
        LOGE ("no magnetic field sensor");
        return 1;
    }

    int32_t min_delay_us = ASensor_getMinDelay (mag);
    if (min_delay_us <= 0)
        min_delay_us = 10000;
    const float sample_rate_hz = 1000000.f / static_cast<float> (min_delay_us);

    LOGI ("sensor=%s minDelay=%d us (%.1f Hz) resolution=%.4f uT", ASensor_getName (mag), min_delay_us, sample_rate_hz,
          ASensor_getResolution (mag));

    ALooper *const looper = ALooper_prepare (ALOOPER_PREPARE_ALLOW_NON_CALLBACKS);
    ASensorEventQueue *const queue = ASensorManager_createEventQueue (manager, looper, 0, nullptr, nullptr);
    if (queue == nullptr) {
        LOGE ("could not create sensor event queue");
        return 1;
    }
    ASensorEventQueue_enableSensor (queue, mag);
    ASensorEventQueue_setEventRate (queue, mag, min_delay_us);

    static float mag_x[_S_window];
    static float mag_y[_S_window];
    static float mag_z[_S_window];
    uint32_t idx = 0u;
    uint32_t poll_timeouts = 0u;

    const timespec start = [] {
        timespec t{};
        clock_gettime (CLOCK_MONOTONIC, &t);
        return t;
    }();

    LOGI ("capturing %u samples (~%.1f s)...", _S_window,
          static_cast<double> (_S_window) / static_cast<double> (sample_rate_hz));

    while (idx < _S_window) {
        const int ident = ALooper_pollOnce (200, nullptr, nullptr, nullptr);
        if (ident == ALOOPER_POLL_TIMEOUT) {
            if ((++poll_timeouts % 10u) == 0u)
                LOGI ("waiting: %u samples, %u timeouts", idx, poll_timeouts);
            continue;
        }
        if (ident == ALOOPER_POLL_ERROR) {
            LOGE ("looper error");
            break;
        }
        ASensorEvent event{};
        while (ASensorEventQueue_getEvents (queue, &event, 1) > 0) {
            if (event.type != ASENSOR_TYPE_MAGNETIC_FIELD)
                continue;
            mag_x[idx] = event.magnetic.x;
            mag_y[idx] = event.magnetic.y;
            mag_z[idx] = event.magnetic.z;
            if ((++idx % 200u) == 0u)
                LOGI ("captured %u/%u", idx, _S_window);
            if (idx >= _S_window)
                break;
        }
    }

    ASensorEventQueue_disableSensor (queue, mag);

    const timespec end = [] {
        timespec t{};
        clock_gettime (CLOCK_MONOTONIC, &t);
        return t;
    }();
    const double elapsed_s =
        static_cast<double> (end.tv_sec - start.tv_sec) + static_cast<double> (end.tv_nsec - start.tv_nsec) * 1e-9;

    if (idx < _S_window) {
        LOGE ("only captured %u samples in %.2f s - no usable data", idx, elapsed_s);
        return 1;
    }

    LOGI ("captured %u samples in %.2f s -> effective %.1f Hz", idx, elapsed_s, static_cast<double> (idx) / elapsed_s);

    /* AC-couple: remove the DC (Earth field) component per axis. */
    float mean_x = 0.f, mean_y = 0.f, mean_z = 0.f;
    for (uint32_t i = 0; i < _S_window; ++i) {
        mean_x += mag_x[i];
        mean_y += mag_y[i];
        mean_z += mag_z[i];
    }
    mean_x /= static_cast<float> (_S_window);
    mean_y /= static_cast<float> (_S_window);
    mean_z /= static_cast<float> (_S_window);
    for (uint32_t i = 0; i < _S_window; ++i) {
        mag_x[i] -= mean_x;
        mag_y[i] -= mean_y;
        mag_z[i] -= mean_z;
    }

    const float rms_x = rms (mag_x, _S_window);
    const float rms_y = rms (mag_y, _S_window);
    const float rms_z = rms (mag_z, _S_window);
    const float total_ms = rms_x * rms_x + rms_y * rms_y + rms_z * rms_z;

    LOGI ("DC field: [%.2f, %.2f, %.2f] uT", static_cast<double> (mean_x), static_cast<double> (mean_y),
          static_cast<double> (mean_z));
    LOGI ("AC RMS  : x=%.4f y=%.4f z=%.4f uT (noise-floor resolution is %.3f uT)", rms_x, rms_y, rms_z,
          ASensor_getResolution (mag));

    const float mains[] = {50.f, 60.f, 100.f, 120.f};
    const char *const label[] = {"50 Hz", "60 Hz", "100 Hz", "120 Hz"};
    float ratio[4] = {0.f, 0.f, 0.f, 0.f};

    LOGI ("%-7s %10s %10s %10s %12s %10s", "freq", "powX", "powY", "powZ", "total", "%ofAC");
    for (uint32_t f = 0; f < 4u; ++f) {
        const float px = dft_power (mag_x, _S_window, sample_rate_hz, mains[f]);
        const float py = dft_power (mag_y, _S_window, sample_rate_hz, mains[f]);
        const float pz = dft_power (mag_z, _S_window, sample_rate_hz, mains[f]);
        const float total = px + py + pz;
        ratio[f] = total_ms > 0.f ? 100.f * total / total_ms : 0.f;
        LOGI ("%-7s %10.6f %10.6f %10.6f %12.6f %9.2f%%", label[f], px, py, pz, total, ratio[f]);
    }

    /* Broadband noise floor (1 Hz bins, mains excluded). */
    const float floor_x = band_floor_power (mag_x, _S_window, sample_rate_hz, 10.f, 45.f);
    const float floor_y = band_floor_power (mag_y, _S_window, sample_rate_hz, 10.f, 45.f);
    const float floor_z = band_floor_power (mag_z, _S_window, sample_rate_hz, 10.f, 45.f);
    const float floor_total = floor_x + floor_y + floor_z;

    LOGI ("noise floor per 1 Hz bin: x=%.8f y=%.8f z=%.8f (sum %.8f)", floor_x, floor_y, floor_z, floor_total);

    for (uint32_t f = 0; f < 2u; ++f) {
        const float px = dft_power (mag_x, _S_window, sample_rate_hz, mains[f]);
        const float py = dft_power (mag_y, _S_window, sample_rate_hz, mains[f]);
        const float pz = dft_power (mag_z, _S_window, sample_rate_hz, mains[f]);
        const float total = px + py + pz;
        const float snr = floor_total > 0.f ? total / floor_total : 0.f;
        LOGI ("%s: %.2f%% of AC power, %.1fx the 1 Hz noise floor", label[f], ratio[f], snr);
    }

    /* Alias-aware peak scan.  With fs = 100 Hz (this sensor's max ODR), a real
     * 50 Hz field folds onto DC/Nyquist and a 60 Hz field folds onto 40 Hz, so
     * scanning every 1 Hz bin between 15-45 Hz (above the 1/f drift band) is the
     * honest test for "is any coherent AC line signal reaching this sensor". */
    float best_f = 0.f;
    float best_p = 0.f;
    for (float f = 15.f; f <= 45.f; f += 1.f) {
        const float p = dft_power (mag_x, _S_window, sample_rate_hz, f)
                        + dft_power (mag_y, _S_window, sample_rate_hz, f)
                        + dft_power (mag_z, _S_window, sample_rate_hz, f);
        if (p > best_p) {
            best_p = p;
            best_f = f;
        }
    }
    const float floor_1hz = (floor_x + floor_y + floor_z);
    LOGI ("strongest 1 Hz bin in 15-45 Hz: %.0f Hz (%.2fx the mean noise floor)", best_f,
          floor_1hz > 0.f ? best_p / floor_1hz : 0.f);
    LOGI ("  (fs=%.0f Hz: 50 Hz mains aliases to DC/Nyquist, 60 Hz mains aliases to %.0f Hz)", sample_rate_hz,
          sample_rate_hz > 60.f ? sample_rate_hz - 60.f : 60.f - sample_rate_hz);

    const bool coherent_peak = floor_1hz > 0.f && (best_p / floor_1hz) > 10.f;
    const bool detected = (ratio[0] > 20.f) || (ratio[1] > 20.f) || coherent_peak;
    LOGI ("VERDICT: mains harmonics %s on this magnetometer", detected ? "DETECTABLE" : "NOT DETECTABLE");
    if (!detected) {
        LOGI ("=> the raw magnetic signal is dominated by broadband noise/motion; a 50/60 Hz");
        LOGI ("   component this small cannot be recovered at %.0f Hz ODR", sample_rate_hz);
    }
    return 0;
}
