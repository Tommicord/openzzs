/* Copyright(c) 2026 Tommicord
 *
 * MIT License
 *
 * Acoustic mains-hum probe for Android (AAudio, no AGC / no noise suppression).
 *
 * Power lines and every transformer/PSU on them radiate at the mains frequency
 * and its harmonics through magnetostriction (100/120 Hz, 150/180 Hz, ...).
 * A phone microphone has kilohertz of bandwidth, so unlike the 100 Hz-capped
 * magnetometer this path can actually resolve the harmonic ladder.
 *
 * Build:
 *   $NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android36-clang++ \
 *       -std=gnu++2b -O2 -DANDROID tools/mic_probe/mic_probe.cpp -o mic_probe -laaudio -llog
 *   adb push mic_probe /data/local/tmp/ && adb shell chmod +x /data/local/tmp/mic_probe
 *   adb shell /data/local/tmp/mic_probe
 */

#include <aaudio/AAudio.h>
#include <android/log.h>
#include <cmath>
#include <cstdint>
#include <vector>

#define TAG "MicProbe"
#define LOGI(...) __android_log_print (ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print (ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

namespace {

constexpr float _S_pi = 3.14159265358979323846f;
constexpr int32_t _S_rate = 48000;
constexpr uint32_t _S_seconds = 10;

/* Goertzel: mean-square amplitude of the component at target_hz. */
float
tone_power (const std::vector<float> &x, float fs, float target_hz) noexcept {
    const uint32_t n = static_cast<uint32_t> (x.size ());
    if (n == 0u)
        return 0.f;
    const float w = 2.f * _S_pi * target_hz / fs;
    const float c = 2.f * cosf (w);
    float q1 = 0.f;
    float q2 = 0.f;
    for (uint32_t i = 0; i < n; ++i) {
        const float q0 = c * q1 - q2 + x[i];
        q2 = q1;
        q1 = q0;
    }
    const float re = q1 - q2 * cosf (w);
    const float im = q2 * sinf (w);
    return 2.f * (re * re + im * im) / (static_cast<float> (n) * static_cast<float> (n));
}

float
rms (const std::vector<float> &x) noexcept {
    double sum = 0.0;
    for (float v : x)
        sum += static_cast<double> (v) * static_cast<double> (v);
    return static_cast<float> (sqrt (sum / static_cast<double> (x.size ())));
}

} // namespace

int
main () {
    AAudioStreamBuilder *builder = nullptr;
    if (AAudio_createStreamBuilder (&builder) != AAUDIO_OK || builder == nullptr) {
        LOGE ("AAudio_createStreamBuilder failed");
        return 1;
    }
    AAudioStreamBuilder_setDirection (builder, AAUDIO_DIRECTION_INPUT);
    AAudioStreamBuilder_setSampleRate (builder, _S_rate);
    AAudioStreamBuilder_setChannelCount (builder, 1);
    AAudioStreamBuilder_setFormat (builder, AAUDIO_FORMAT_PCM_FLOAT);
    AAudioStreamBuilder_setPerformanceMode (builder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
    /* UNPROCESSED = no AGC, no noise suppression, no HPF that would eat the hum. */
    AAudioStreamBuilder_setInputPreset (builder, AAUDIO_INPUT_PRESET_UNPROCESSED);

    AAudioStream *stream = nullptr;
    aaudio_result_t rc = AAudioStreamBuilder_openStream (builder, &stream);
    AAudioStreamBuilder_delete (builder);
    if (rc != AAUDIO_OK || stream == nullptr) {
        LOGE ("openStream failed (%s), retrying with VOICE_RECOGNITION", AAudio_convertResultToText (rc));
        AAudio_createStreamBuilder (&builder);
        AAudioStreamBuilder_setDirection (builder, AAUDIO_DIRECTION_INPUT);
        AAudioStreamBuilder_setSampleRate (builder, _S_rate);
        AAudioStreamBuilder_setChannelCount (builder, 1);
        AAudioStreamBuilder_setFormat (builder, AAUDIO_FORMAT_PCM_FLOAT);
        AAudioStreamBuilder_setInputPreset (builder, AAUDIO_INPUT_PRESET_VOICE_RECOGNITION);
        rc = AAudioStreamBuilder_openStream (builder, &stream);
        AAudioStreamBuilder_delete (builder);
        if (rc != AAUDIO_OK || stream == nullptr) {
            LOGE ("openStream failed again (%s)", AAudio_convertResultToText (rc));
            return 1;
        }
    }

    const int32_t fs = AAudioStream_getSampleRate (stream);
    const int32_t chans = AAudioStream_getChannelCount (stream);
    LOGI ("mic: %d Hz, %d ch, preset=%d", fs, chans, AAudioStream_getInputPreset (stream));

    rc = AAudioStream_requestStart (stream);
    if (rc != AAUDIO_OK) {
        LOGE ("requestStart failed (%s)", AAudio_convertResultToText (rc));
        return 1;
    }

    const uint32_t target = static_cast<uint32_t> (_S_seconds) * static_cast<uint32_t> (fs);
    std::vector<float> buf (static_cast<size_t> (target) * static_cast<size_t> (chans));
    std::vector<float> mono;
    mono.reserve (target);
    std::vector<float> chunk (4096 * static_cast<size_t> (chans));

    LOGI ("recording %us...", _S_seconds);
    while (mono.size () < target) {
        const int32_t got = AAudioStream_read (
            stream, chunk.data (), static_cast<int32_t> (chunk.size () / static_cast<size_t> (chans)), 1000000000);
        if (got < 0) {
            LOGE ("read failed (%s)", AAudio_convertResultToText (got));
            break;
        }
        for (int32_t i = 0; i < got; ++i)
            mono.push_back (chunk[static_cast<size_t> (i) * static_cast<size_t> (chans)]); /* ch 0 */
    }
    AAudioStream_requestStop (stream);
    AAudioStream_close (stream);

    const uint32_t n = static_cast<uint32_t> (mono.size ());
    if (n < static_cast<uint32_t> (fs)) {
        LOGE ("only %u samples captured", n);
        return 1;
    }
    LOGI ("captured %u samples (%.1f s)", n, static_cast<double> (n) / static_cast<double> (fs));

    /* DC / sub-audio removal: subtract the mean and high-pass with a 1-pole
     * differentiator-like filter so HVAC rumble does not dominate the floor. */
    double mean = 0.0;
    for (float v : mono)
        mean += v;
    mean /= static_cast<double> (n);
    float prev = 0.f;
    for (uint32_t i = 0; i < n; ++i) {
        const float v = mono[i] - static_cast<float> (mean);
        mono[i] = v - prev * 0.9f; /* crude HP at ~1 kHz/2pi, keeps 100 Hz+ intact */
        prev = v;
    }

    const float r = rms (mono);
    const float db = 20.f * log10f (r + 1e-9f);
    LOGI ("AC RMS = %.6f (%.1f dBFS)", r, db);

    /* Harmonic ladder of 50 Hz and 60 Hz mains. */
    const float targets[] = {50.f, 60.f, 100.f, 120.f, 150.f, 180.f, 200.f, 250.f, 300.f, 350.f, 400.f};
    const uint32_t ntargets = sizeof (targets) / sizeof (targets[0]);

    /* Noise floor: mean 1 Hz-bin power sampled every 5 Hz between 450-1500 Hz. */
    double floor_sum = 0.0;
    uint32_t floor_bins = 0;
    for (float f = 450.f; f <= 1500.f; f += 5.f) {
        floor_sum += static_cast<double> (tone_power (mono, static_cast<float> (fs), f));
        ++floor_bins;
    }
    const float floor_p = floor_bins > 0u ? static_cast<float> (floor_sum / floor_bins) : 0.f;

    float best_p = 0.f;
    float best_f = 0.f;
    for (uint32_t i = 0; i < ntargets; ++i) {
        const float p = tone_power (mono, static_cast<float> (fs), targets[i]);
        if (p > best_p) {
            best_p = p;
            best_f = targets[i];
        }
        LOGI ("%5.0f Hz: power=%12.4e %8.2fx floor", targets[i], p, floor_p > 0.f ? p / floor_p : 0.f);
    }

    LOGI ("strongest mains harmonic: %.0f Hz (%.2fx floor)", best_f, floor_p > 0.f ? best_p / floor_p : 0.f);

    /* The 100/120 Hz pair is the reliable transformer/PSU signature. */
    const float p100 = tone_power (mono, static_cast<float> (fs), 100.f);
    const float p120 = tone_power (mono, static_cast<float> (fs), 120.f);
    const float hum = p100 > p120 ? p100 : p120;
    const bool detected = floor_p > 0.f && (hum / floor_p) > 20.f;
    LOGI ("VERDICT: mains hum %s (100/120 Hz = %.1fx noise floor)", detected ? "DETECTED" : "NOT DETECTED",
          floor_p > 0.f ? hum / floor_p : 0.f);
    return 0;
}
