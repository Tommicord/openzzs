/*
 * Copyright(c) 2026 Tommicord
 *
 * MIT License
 *
 * JNI bridge between the Android UI and the OpenZZS preprocessing bank
 * + mllee inference core. Uses the kernel/sensor-driver platform HAL for
 * Android NDK sensor access, resamples every channel to the 100 Hz grid,
 * runs feature extraction and mllee inference, and returns results to Java.
 */

#include "kernel/kernel_bank.hpp"
#include "kernel/kernel_clock_grid.hpp"
#include "kernel/kernel_clock_source.hpp"
#include "kernel/kernel_config.hpp"
#include "kernel/kernel_dsp_resampler.hpp"
#include "kernel/kernel_feat_acoustic.hpp"
#include "kernel/kernel_feat_flicker.hpp"
#include "kernel/kernel_feat_pack.hpp"
#include "kernel/kernel_feat_plhr.hpp"
#include "kernel/kernel_feat_vibration.hpp"
#include "kernel/kernel_meta_ids.hpp"
#include "kernel/kernel_sensor_hal.hpp"
#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_seq_frames.hpp"
#include "kernel/kernel_util_fixed.hpp"
#include "mllee-native/model.hpp"
#include "mllee/core.hpp"
#include "sensr_bridge.h"

#include <aaudio/AAudio.h>
#include <android/log.h>
#include <android/looper.h>
#include <android/sensor.h>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <jni.h>
#include <mutex>
#include <pthread.h>
#include <string>
#include <time.h>

#define LOGI(...) __android_log_print (ANDROID_LOG_INFO, "openzzs", __VA_ARGS__)
#define LOGE(...) __android_log_print (ANDROID_LOG_ERROR, "openzzs", __VA_ARGS__)
#define LOGW(...) __android_log_print (ANDROID_LOG_WARN, "openzzs", __VA_ARGS__)

namespace {

/** @brief Grid tracker: 100 Hz common grid for all channels. */
static zzs::kernel::clock::_S_GridTracker _S_grid;

/** @brief Preprocessing bank state. */
static zzs::kernel::_S_BankConfig _S_bank_config{};
static zzs::kernel::_S_PreprocessingBank _S_bank{};
static zzs::kernel::frames::Sequence _S_frame{};

/** @brief Per-channel resamplers to the 100 Hz grid. */
static zzs::kernel::dsp::_S_Resampler _S_resampler_mag_x{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_mag_y{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_mag_z{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_accel_x{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_accel_y{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_accel_z{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_gyro_x{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_gyro_y{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_gyro_z{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_light{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_sound{};
static zzs::kernel::dsp::_S_Resampler _S_resampler_battery{};

/** @brief Stage state storage. */
static __attribute__ ((aligned (4))) int8_t _S_plhr_state[sizeof (zzs::kernel::feat::_S_PlhrState)]{};
static __attribute__ ((aligned (4))) int8_t _S_vib_state[sizeof (zzs::kernel::feat::_S_VibrationState)]{};
static __attribute__ ((aligned (4))) int8_t _S_flicker_state[sizeof (zzs::kernel::feat::_S_FlickerState)]{};
static __attribute__ ((aligned (4))) int8_t _S_acoustic_state[sizeof (zzs::kernel::feat::_S_AcousticState)]{};

/** @brief AAudio mic capture for the sound-envelope PLHR path. */
static AAudioStream *_S_aaudio_stream = nullptr;
static pthread_t _S_aaudio_thread{};
static std::atomic<bool> _S_aaudio_running{false};

/** @brief Mutex protecting shared bank/grid state across the AAudio callback
 *         thread and the JNI sensor-feed thread. */
static std::mutex _S_bank_mutex;

/** @brief Resampler push wrapper for each channel. Forward-declared here so
 *         the AAudio callback (defined below) can call it. */
static void
_push_sample (zzs::kernel::dsp::_S_Resampler &resampler, zzs::kernel::sensor::Channel channel, int16_t sample_q15,
              uint64_t timestamp_ns) noexcept;

/** @brief AAudio PCM accumulation buffer for block-RMS envelope (float, mono). */
static float _S_aaudio_buf[4096];
static uint32_t _S_aaudio_buf_len = 0;

/** @brief Sound envelope timestamp counter (grid-aligned). */
static uint64_t _S_sound_ts = 0;

/** @brief AAudio data callback: accumulate PCM, downsample to 1 kHz,
 *         feed directly to acoustic stage (Goertzel resolves 120 Hz hum).
 */
static aaudio_data_callback_result_t
_aaudio_callback (AAudioStream * /*stream*/, void * /*user_data*/, void *audio_data, int32_t num_frames) noexcept {
  const float *const pcm = static_cast<const float *> (audio_data);
  uint32_t frames_left = static_cast<uint32_t> (num_frames);
  const float *const pcm_end = pcm + frames_left;

  const float *ptr = pcm;
  while (frames_left > 0u) {
    const uint32_t space =
        static_cast<uint32_t> (sizeof (_S_aaudio_buf) / sizeof (_S_aaudio_buf[0])) - _S_aaudio_buf_len;
    const uint32_t copy = frames_left < space ? frames_left : space;
    for (uint32_t i = 0; i < copy; ++i)
      _S_aaudio_buf[_S_aaudio_buf_len + i] = ptr[i];
    _S_aaudio_buf_len += copy;
    ptr += copy;
    frames_left -= copy;

    /* Downsample 48 kHz -> 1 kHz: take one sample every 48 frames.
     * Feed directly to the acoustic stage (Goertzel resolves 120 Hz hum).
     * Also feed the same 1 kHz PCM directly into the PLHR stage as
     * MAG_X/Y/Z so the existing PLHR bandpass + Goertzel harmonic banks
     * process mic data instead of dead magnetometer data (mag is HAL-capped
     * at 100 Hz, Nyquist = 50 Hz, so 60 Hz PLHR is unreachable via
     * magnetometer).
     *
     * The PLHR stage advances its window tick on MAG_Z, so all three axes
     * must be pushed in order: X, Y, Z.  All three receive the same mono PCM
     * since we have a single microphone channel.
     */
    while (_S_aaudio_buf_len >= 48u) {
      /* Simple decimation: average 48 samples for anti-aliasing. */
      double sum = 0.0;
      for (uint32_t i = 0; i < 48u; ++i)
        sum += static_cast<double> (_S_aaudio_buf[i]);
      const int16_t sample_q15 = static_cast<int16_t> (sum / 48.0 * 32767.0);
      for (uint32_t i = 48u; i < _S_aaudio_buf_len; ++i)
        _S_aaudio_buf[i - 48u] = _S_aaudio_buf[i];
      _S_aaudio_buf_len -= 48u;

      /* Feed both stages under the grid lock. */
      std::lock_guard<std::mutex> lock (_S_bank_mutex);
      if (const zzs::kernel::Status afst = zzs::kernel::feat::acoustic_feed (
              _S_acoustic_state, zzs::kernel::sensor::Channel::SOUND_ENVELOPE, sample_q15);
          !zzs::kernel::status_ok (afst) && afst != zzs::kernel::Status::NOT_READY)
        LOGE ("acoustic_feed failed: %d", static_cast<int> (afst));

      /* Feed PLHR stage directly at 1 kHz (no resampler decimation that
       * would destroy the 120 Hz hum).  All three axes get the same PCM;
       * the PLHR stage advances its window tick on MAG_Z. */
      static uint64_t _S_plhr_ts = 0;
      _S_plhr_ts += zzs::kernel::_S_grid_period_ns;
      if (const zzs::kernel::Status pfst =
              zzs::kernel::feat::plhr_feed (_S_plhr_state, zzs::kernel::sensor::Channel::MAG_X, sample_q15);
          !zzs::kernel::status_ok (pfst) && pfst != zzs::kernel::Status::NOT_READY)
        LOGE ("plhr_feed MAG_X failed: %d", static_cast<int> (pfst));
      if (const zzs::kernel::Status pfst =
              zzs::kernel::feat::plhr_feed (_S_plhr_state, zzs::kernel::sensor::Channel::MAG_Y, sample_q15);
          !zzs::kernel::status_ok (pfst) && pfst != zzs::kernel::Status::NOT_READY)
        LOGE ("plhr_feed MAG_Y failed: %d", static_cast<int> (pfst));
      if (const zzs::kernel::Status pfst =
              zzs::kernel::feat::plhr_feed (_S_plhr_state, zzs::kernel::sensor::Channel::MAG_Z, sample_q15);
          !zzs::kernel::status_ok (pfst) && pfst != zzs::kernel::Status::NOT_READY)
        LOGE ("plhr_feed MAG_Z failed: %d", static_cast<int> (pfst));
    }
  }
  return AAUDIO_CALLBACK_RESULT_CONTINUE;
}

/** @brief AAudio thread entry (kept for lifecycle symmetry; the callback does
 *         the work). */
static void *
_aaudio_thread_fn (void *) noexcept {
  // The callback is driven by AAudio's internal thread; this thread just
  // keeps the process alive while the stream is open.  No-op here.
  return nullptr;
}

/** @brief Start AAudio mic capture (UNPROCESSED, 48 kHz, mono, float). */
[[nodiscard]] static zzs::kernel::Status
_aaudio_start () noexcept {
  AAudioStreamBuilder *builder = nullptr;
  if (AAudio_createStreamBuilder (&builder) != AAUDIO_OK || builder == nullptr)
    return zzs::kernel::Status::INTERNAL;
  AAudioStreamBuilder_setDirection (builder, AAUDIO_DIRECTION_INPUT);
  AAudioStreamBuilder_setSampleRate (builder, 48000);
  AAudioStreamBuilder_setChannelCount (builder, 1);
  AAudioStreamBuilder_setFormat (builder, AAUDIO_FORMAT_PCM_FLOAT);
  AAudioStreamBuilder_setPerformanceMode (builder, AAUDIO_PERFORMANCE_MODE_LOW_LATENCY);
#if __ANDROID_API__ >= 28
  AAudioStreamBuilder_setInputPreset (builder, AAUDIO_INPUT_PRESET_UNPROCESSED);
#endif
  AAudioStreamBuilder_setDataCallback (builder, _aaudio_callback, nullptr);

  aaudio_result_t res = AAudioStreamBuilder_openStream (builder, &_S_aaudio_stream);
  AAudioStreamBuilder_delete (builder);
  if (res != AAUDIO_OK || _S_aaudio_stream == nullptr)
    return zzs::kernel::Status::INTERNAL;

  _S_aaudio_running = true;
  if (AAudioStream_requestStart (_S_aaudio_stream) != AAUDIO_OK) {
    _S_aaudio_running = false;
    AAudioStream_close (_S_aaudio_stream);
    _S_aaudio_stream = nullptr;
    return zzs::kernel::Status::INTERNAL;
  }
  return zzs::kernel::Status::OK;
}

/** @brief Stop AAudio mic capture. */
static void
_aaudio_stop () noexcept {
  if (_S_aaudio_stream == nullptr)
    return;
  _S_aaudio_running = false;
  AAudioStream_requestStop (_S_aaudio_stream);
  AAudioStream_close (_S_aaudio_stream);
  _S_aaudio_stream = nullptr;
}

/** @brief mllee model blob (extern, defined in generated model_blob_generated.cpp) and core. */
static zzs::mllee::Core _S_core{};

/** @brief JNI callback references. */
static JavaVM *_S_vm = nullptr;
static jobject _S_listener = nullptr;
static jmethodID _M_on_features = nullptr;
static jmethodID _M_on_inference = nullptr;

/**
 * @brief Feed one resampled grid-tick sample into the bank and run inference
 *        when a feature row is ready.
 */
static void
_feed_grid_sample (zzs::kernel::sensor::Channel channel, int16_t sample_q15, uint64_t timestamp_ns) noexcept {
  // Deterministic grid timestamps: Android sensor timestamps may be 0 or
  // unreliable on user builds, so advance by one grid period per sample.
  static uint64_t _S_grid_ts = 0;
  _S_grid_ts += zzs::kernel::_S_grid_period_ns;
  (void) timestamp_ns;

  static uint32_t _S_grid_debug_count = 0;
  if (_S_grid_debug_count < 40u) {
    LOGE ("DEBUG grid: ch=%u q15=%d tick=%u", static_cast<unsigned int> (channel), sample_q15,
          static_cast<unsigned int> (_S_grid._M_tick_count));
    _S_grid_debug_count++;
  }

  zzs::kernel::sensor::Sample s;
  s._M_channel = channel;
  s._M_sample_q15 = sample_q15;
  s._M_timestamp_ns = _S_grid_ts;

  // Advance grid deterministically (Android timestamps may be 0 on user builds).
  {
    std::lock_guard<std::mutex> lock (_S_bank_mutex);
    _S_grid._M_tick_count += 1u;
    _S_grid._M_last_tick_ns = s._M_timestamp_ns;
  }

  auto st = zzs::kernel::bank_feed_sample (_S_bank, s);
  if (!zzs::kernel::status_ok (st) && st != zzs::kernel::Status::NOT_READY)
    LOGE ("bank_feed_sample failed: %d", static_cast<int> (st));

  int8_t row[zzs::kernel::_S_feature_count];
  st = zzs::kernel::bank_produce_row (_S_bank, row);
  if (st == zzs::kernel::Status::NOT_READY)
    return; // No window finished yet
  if (!zzs::kernel::status_ok (st)) {
    LOGE ("bank_produce_row failed: %d", static_cast<int> (st));
    return;
  }

  // Append to sequence frame (seq_len=5)
  if (!_S_frame.append_row (row))
    return; // Frame not complete yet

  // Frame complete: run inference using the OUTPRED model blob from the VTable.
  if (const zzs::mllee::Status st = zzs::mllee_native::model::init_model_vtable (); !zzs::mllee::status_ok (st)) {
    LOGW ("init_model_vtable failed: %d, skipping inference", static_cast<int> (st));
    _S_frame.reset ();
    return;
  }
  const uint32_t idx = zzs::mllee_native::model::find_outpred ();
  if (idx == zzs::mllee_native::model::ModelVTable<>::k_npos) {
    LOGW ("outpred model not registered in VTable, skipping inference");
    _S_frame.reset ();
    return;
  }
  const zzs::mllee_native::model::VTableEntry &entry = zzs::mllee_native::model::g_model_vtable[idx];
  if (entry._M_blob[0] != 'M') {
    LOGW ("Model blob not initialized, skipping inference");
    _S_frame.reset ();
    return;
  }

  LOGE ("DEBUG row: [%d %d %d %d %d %d %d %d %d %d | P%u V%u %u %u F%u %u %u %u A%u %u %u R%u %u %u]",
        _S_frame.data ()[0], _S_frame.data ()[1], _S_frame.data ()[2], _S_frame.data ()[3], _S_frame.data ()[4],
        _S_frame.data ()[5], _S_frame.data ()[6], _S_frame.data ()[7], _S_frame.data ()[8], _S_frame.data ()[9],
        static_cast<unsigned int> (_S_frame.data ()[20]), static_cast<unsigned int> (_S_frame.data ()[21]),
        static_cast<unsigned int> (_S_frame.data ()[22]), static_cast<unsigned int> (_S_frame.data ()[23]),
        static_cast<unsigned int> (_S_frame.data ()[24]), static_cast<unsigned int> (_S_frame.data ()[25]),
        static_cast<unsigned int> (_S_frame.data ()[26]), static_cast<unsigned int> (_S_frame.data ()[27]),
        static_cast<unsigned int> (_S_frame.data ()[28]), static_cast<unsigned int> (_S_frame.data ()[29]),
        static_cast<unsigned int> (_S_frame.data ()[30]), static_cast<unsigned int> (_S_frame.data ()[31]),
        static_cast<unsigned int> (_S_frame.data ()[32]), static_cast<unsigned int> (_S_frame.data ()[33]),
        static_cast<unsigned int> (_S_frame.data ()[34]), static_cast<unsigned int> (_S_frame.data ()[35]),
        static_cast<unsigned int> (_S_frame.data ()[36]), static_cast<unsigned int> (_S_frame.data ()[37]));

  struct timespec ts_start;
  clock_gettime (CLOCK_MONOTONIC, &ts_start);

  const auto result =
      _S_core.run (entry._M_blob, entry._M_blob_size, reinterpret_cast<const int8_t *> (_S_frame.data ()),
                   zzs::mllee_native::model::kInputDim);
  if (!result) {
    LOGE ("mllee inference failed: %d", static_cast<int> (result.error ().kind ()));
    _S_frame.reset ();
    return;
  }

  struct timespec ts_end;
  clock_gettime (CLOCK_MONOTONIC, &ts_end);
  const uint64_t start_ns =
      static_cast<uint64_t> (ts_start.tv_sec) * 1000000000ull + static_cast<uint64_t> (ts_start.tv_nsec);
  const uint64_t end_ns =
      static_cast<uint64_t> (ts_end.tv_sec) * 1000000000ull + static_cast<uint64_t> (ts_end.tv_nsec);
  const uint64_t elapsed_us = (end_ns - start_ns) / 1000ull;
  const uint64_t elapsed_ms = elapsed_us / 1000ull;

  // Read 3 softmax output codes
  int8_t prob_codes[3];
  for (uint32_t i = 0; i < 3u; ++i)
    prob_codes[i] = result.value ()._M_data[i];

  LOGE ("DEBUG raw codes: [%d, %d, %d] blob[0]=%c kRequantMult=%d", prob_codes[0], prob_codes[1], prob_codes[2],
        entry._M_blob[0], zzs::mllee_native::model::kRequantMult);

  // Convert to probabilities (0-100%).
  // Softmax output codes are in [0, 127] (0 = 0% probability, 127 = 100%),
  // NOT in [-127, 127], so we scale code/127 directly.  All math is Q15
  // fixed-point; no float anywhere (mllee/ and std/ are float-free).
  //
  // Apply temperature scaling in Q15: p^(1/T) with T=2 softens near-0/near-1
  // outputs by taking the integer square root of (p * Q15_ONE), then apply a
  // minimum probability floor so no class collapses to exactly 0%.
  int32_t probs_q15[3];
  int32_t total_q15 = 0;
  for (uint32_t i = 0; i < 3u; ++i) {
    int32_t p = static_cast<int32_t> (prob_codes[i]);
    if (p < 0)
      p = 0;
    if (p > 127)
      p = 127;
    /* Temperature scaling: p_scaled_q15 = isqrt(p_q15 * Q15_ONE).
       With T=2, near-1 values stay near 1 (isqrt(32768*32768)=32768),
       near-0.5 values become isqrt(16384*32768)=23170 (Q15 ≈ 0.707),
       softening extreme outputs while staying in Q15. */
    const int32_t p_q15 = p * 32768 / 127; /* code -> Q15 */
    const int32_t p_sqrt_q15 = zzs::mllee::_S_isqrt (static_cast<uint32_t> (p_q15) * 32768u);
    probs_q15[i] = p_sqrt_q15;
    total_q15 += p_sqrt_q15;
  }
  /* Minimum probability floor: every class gets at least 5% (1638 Q15). */
  constexpr int32_t kMinProbQ15 = 1638; /* 0.05 * 32768 */
  total_q15 = 0;
  for (uint32_t i = 0; i < 3u; ++i) {
    if (probs_q15[i] < kMinProbQ15)
      probs_q15[i] = kMinProbQ15;
    total_q15 += probs_q15[i];
  }
  /* Normalize: probs_q15[i] /= total_q15, all in Q15, then convert to float
     for the JNI callback. */
  float probs[3];
  for (uint32_t i = 0; i < 3u; ++i)
    probs[i] = static_cast<float> (probs_q15[i]) / static_cast<float> (total_q15);

  // Find predicted class
  uint32_t predicted = 0u;
  for (uint32_t i = 1u; i < 3u; ++i)
    if (probs[i] > probs[predicted])
      predicted = i;

  LOGI ("Inference: %llu ms (%llu us), P_normal=%.3f P_outage=%.3f "
        "P_transition=%.3f -> %s",
        static_cast<unsigned long long> (elapsed_ms), static_cast<unsigned long long> (elapsed_us), probs[0], probs[1],
        probs[2], predicted == 0 ? "normal" : (predicted == 1 ? "blackout" : "transition"));

  if (elapsed_ms > 10u)
    LOGW ("Inference slow: %llu ms (%llu us)", static_cast<unsigned long long> (elapsed_ms),
          static_cast<unsigned long long> (elapsed_us));

  // Reset frame for next inference cycle.
  _S_frame.reset ();

  // Deliver to Java on the UI thread
  JNIEnv *env;
  if (_S_vm == nullptr || _S_vm->GetEnv (reinterpret_cast<void **> (&env), JNI_VERSION_1_6) != JNI_OK)
    return;
  if (_S_listener == nullptr || _M_on_inference == nullptr)
    return;

  jfloatArray jprobs = env->NewFloatArray (3);
  env->SetFloatArrayRegion (jprobs, 0, 3, probs);
  env->CallVoidMethod (_S_listener, _M_on_inference, jprobs);
  env->DeleteLocalRef (jprobs);

  _S_frame.reset ();
}

/**
 * @brief Resampler push wrapper for each channel.
 */
static void
_push_sample (zzs::kernel::dsp::_S_Resampler &resampler, zzs::kernel::sensor::Channel channel, int16_t sample_q15,
              uint64_t timestamp_ns) noexcept {
  // For upsampling channels (q=1, p>1): sample-and-hold at grid rate.
  // Slow sensors (e.g. light at 1 Hz) produce one input sample per second;
  // repeat it p times at grid spacing so the grid advances and stages
  // accumulate windows at 100 Hz.
  if (resampler._M_q == 1u && resampler._M_p > 1u) {
    for (uint32_t i = 0u; i < resampler._M_p; ++i) {
      _feed_grid_sample (channel, sample_q15,
                         timestamp_ns + static_cast<uint64_t> (i) * zzs::kernel::_S_grid_period_ns);
    }
    return;
  }

  // Normal path: resampler handles decimation/passthrough (q>1).
  bool produced = false;
  int16_t out = 0;
  auto st = zzs::kernel::dsp::resampler_push (resampler, sample_q15, &produced, &out);
  if (!zzs::kernel::status_ok (st)) {
    LOGE ("resampler_push failed: %d", static_cast<int> (st));
    return;
  }
  if (produced)
    _feed_grid_sample (channel, out, timestamp_ns);
}

/** @brief Public feed function called by sensr_android.cpp and Java. */
extern "C" void
feed_native_sample (zzs::kernel::sensor::Channel channel, int16_t sample_q15, uint64_t timestamp_ns) noexcept {
  static uint32_t _S_feed_debug_count = 0;
  if (_S_feed_debug_count < 8u) {
    LOGE ("DEBUG feed: ch=%u q15=%d ts=%llu", static_cast<unsigned int> (channel), sample_q15,
          static_cast<unsigned long long> (timestamp_ns));
    _S_feed_debug_count++;
  }
  // Fallback: Android sensor timestamps may be 0 on some devices; use a
  // monotonic counter so the grid tracker advances.
  static uint64_t _S_fallback_ts = 0;
  if (timestamp_ns == 0)
    timestamp_ns = _S_fallback_ts += zzs::kernel::_S_grid_period_ns;
  // Route through the appropriate resampler to the 100 Hz grid
  switch (channel) {
  case zzs::kernel::sensor::Channel::MAG_X:
    _push_sample (_S_resampler_mag_x, channel, sample_q15, timestamp_ns);
    break;
  case zzs::kernel::sensor::Channel::MAG_Y:
    _push_sample (_S_resampler_mag_y, channel, sample_q15, timestamp_ns);
    break;
  case zzs::kernel::sensor::Channel::MAG_Z:
    _push_sample (_S_resampler_mag_z, channel, sample_q15, timestamp_ns);
    break;
  case zzs::kernel::sensor::Channel::ACCEL_X:
    _push_sample (_S_resampler_accel_x, channel, sample_q15, timestamp_ns);
    break;
  case zzs::kernel::sensor::Channel::ACCEL_Y:
    _push_sample (_S_resampler_accel_y, channel, sample_q15, timestamp_ns);
    break;
  case zzs::kernel::sensor::Channel::ACCEL_Z:
    _push_sample (_S_resampler_accel_z, channel, sample_q15, timestamp_ns);
    break;
  case zzs::kernel::sensor::Channel::LIGHT:
    _push_sample (_S_resampler_light, channel, sample_q15, timestamp_ns);
    break;
  case zzs::kernel::sensor::Channel::SOUND_ENVELOPE:
    _push_sample (_S_resampler_sound, channel, sample_q15, timestamp_ns);
    break;
  case zzs::kernel::sensor::Channel::BATTERY:
    _push_sample (_S_resampler_battery, channel, sample_q15, timestamp_ns);
    break;
  case zzs::kernel::sensor::Channel::GYRO_X:
    _push_sample (_S_resampler_gyro_x, channel, sample_q15, timestamp_ns);
    break;
  case zzs::kernel::sensor::Channel::GYRO_Y:
    _push_sample (_S_resampler_gyro_y, channel, sample_q15, timestamp_ns);
    break;
  case zzs::kernel::sensor::Channel::GYRO_Z:
    _push_sample (_S_resampler_gyro_z, channel, sample_q15, timestamp_ns);
    break;
  default:
    LOGE ("feed_native_sample: unknown channel %d", static_cast<int> (channel));
    break;
  }
}

} // anonymous namespace

extern "C" {

JNIEXPORT jint JNICALL
Java_com_openzzs_SensrBridge_initNative (JNIEnv *env, jobject thiz) noexcept {
  jint rv = env->GetJavaVM (&_S_vm);
  if (rv != JNI_OK || _S_vm == nullptr)
    return JNI_ERR;

  // Keep a global reference to the listener object
  _S_listener = env->NewGlobalRef (thiz);
  LOGE ("DEBUG init: listener=%p", static_cast<void *> (_S_listener));
  jclass cls = env->GetObjectClass (thiz);
  if (cls == nullptr) {
    env->DeleteGlobalRef (_S_listener);
    _S_listener = nullptr;
    return JNI_ERR;
  }
  _M_on_features = env->GetMethodID (cls, "onFeatures", "([Ljava/lang/String;)V");
  _M_on_inference = env->GetMethodID (cls, "onInference", "([F)V");
  if (_M_on_features == nullptr || _M_on_inference == nullptr) {
    env->DeleteGlobalRef (_S_listener);
    _S_listener = nullptr;
    env->DeleteLocalRef (cls);
    return JNI_ERR;
  }
  env->DeleteLocalRef (cls);

  // Initialize clock source (Android: CLOCK_MONOTONIC via clock_gettime)
  auto st = zzs::kernel::clock::clock_source_install ([] () noexcept -> zzs::kernel::clock::Nanoseconds {
    struct timespec ts;
    if (clock_gettime (CLOCK_MONOTONIC, &ts) == 0)
      return static_cast<uint64_t> (ts.tv_sec) * 1000000000ull + static_cast<uint64_t> (ts.tv_nsec);
    return 0ull;
  });
  if (!zzs::kernel::status_ok (st)) {
    LOGE ("clock_source_install failed: %d", static_cast<int> (st));
    return JNI_ERR;
  }

  // Initialize grid tracker
  st = zzs::kernel::clock::grid_tracker_init (_S_grid, 0ull);
  if (!zzs::kernel::status_ok (st)) {
    LOGE ("grid_tracker_init failed: %d", static_cast<int> (st));
    return JNI_ERR;
  }

  // Build model blob from trained or init weights
  auto model_st = zzs::mllee_native::model::init_model_vtable ();
  if (!zzs::mllee::status_ok (model_st)) {
    LOGE ("init_model_blob failed: %d", static_cast<int> (model_st));
    return JNI_ERR;
  }

  // Configure resamplers for each channel (rate -> 100 Hz grid)
  st = zzs::kernel::dsp::resampler_configure (_S_resampler_mag_x, 200, 100);
  if (!zzs::kernel::status_ok (st))
    LOGE ("mag_x resampler configure failed: %d", static_cast<int> (st));
  st = zzs::kernel::dsp::resampler_configure (_S_resampler_mag_y, 200, 100);
  if (!zzs::kernel::status_ok (st))
    LOGE ("mag_y resampler configure failed: %d", static_cast<int> (st));
  st = zzs::kernel::dsp::resampler_configure (_S_resampler_mag_z, 200, 100);
  if (!zzs::kernel::status_ok (st))
    LOGE ("mag_z resampler configure failed: %d", static_cast<int> (st));

  /* PLHR stage runs at 1 kHz (fed directly from AAudio PCM downsampled to
   * 1 kHz), not at the 100 Hz grid.  The magnetometer is HAL-capped at 100
   * Hz (Nyquist = 50 Hz), so 60 Hz PLHR is unreachable via magnetometer.
   * The mic at 1 kHz has Nyquist = 500 Hz, well above 120 Hz.  Override the
   * default sample rate (_S_grid_rate_hz = 100) that plhr_configure sets. */
  auto *plhr_state = reinterpret_cast<zzs::kernel::feat::_S_PlhrState *> (_S_plhr_state);
  plhr_state->_M_sample_rate_hz = 1000u;

  st = zzs::kernel::dsp::resampler_configure (_S_resampler_accel_x, 200, 100);
  if (!zzs::kernel::status_ok (st))
    LOGE ("accel_x resampler configure failed: %d", static_cast<int> (st));
  st = zzs::kernel::dsp::resampler_configure (_S_resampler_accel_y, 200, 100);
  if (!zzs::kernel::status_ok (st))
    LOGE ("accel_y resampler configure failed: %d", static_cast<int> (st));
  st = zzs::kernel::dsp::resampler_configure (_S_resampler_accel_z, 200, 100);
  if (!zzs::kernel::status_ok (st))
    LOGE ("accel_z resampler configure failed: %d", static_cast<int> (st));

  st = zzs::kernel::dsp::resampler_configure (_S_resampler_light, 1, 100);
  if (!zzs::kernel::status_ok (st))
    LOGE ("light resampler configure failed: %d", static_cast<int> (st));
  zzs::kernel::dsp::resampler_set_linear_mode (_S_resampler_light, true);

  st = zzs::kernel::dsp::resampler_configure (_S_resampler_sound, 200, 100);
  if (!zzs::kernel::status_ok (st))
    LOGE ("sound resampler configure failed: %d", static_cast<int> (st));

  st = zzs::kernel::dsp::resampler_configure (_S_resampler_battery, 1, 100);
  if (!zzs::kernel::status_ok (st))
    LOGE ("battery resampler configure failed: %d", static_cast<int> (st));
  zzs::kernel::dsp::resampler_set_linear_mode (_S_resampler_battery, true);

  st = zzs::kernel::dsp::resampler_configure (_S_resampler_gyro_x, 200, 100);
  if (!zzs::kernel::status_ok (st))
    LOGE ("gyro_x resampler configure failed: %d", static_cast<int> (st));
  st = zzs::kernel::dsp::resampler_configure (_S_resampler_gyro_y, 200, 100);
  if (!zzs::kernel::status_ok (st))
    LOGE ("gyro_y resampler configure failed: %d", static_cast<int> (st));
  st = zzs::kernel::dsp::resampler_configure (_S_resampler_gyro_z, 200, 100);
  if (!zzs::kernel::status_ok (st))
    LOGE ("gyro_z resampler configure failed: %d", static_cast<int> (st));

  // Register stages in the bank config (no acoustic stage on Android:
  // no sound data is fed, so it would never complete).
  _S_bank_config._M_stages[0]._M_vtable = &zzs::kernel::feat::plhr_vtable ();
  _S_bank_config._M_stages[0]._M_state = _S_plhr_state;
  _S_bank_config._M_stages[1]._M_vtable = &zzs::kernel::feat::vibration_vtable ();
  _S_bank_config._M_stages[1]._M_state = _S_vib_state;
  _S_bank_config._M_stages[2]._M_vtable = &zzs::kernel::feat::flicker_vtable ();
  _S_bank_config._M_stages[2]._M_state = _S_flicker_state;
  _S_bank_config._M_stage_count = 3;

  st = zzs::kernel::bank_init (_S_bank, _S_bank_config);
  if (!zzs::kernel::status_ok (st)) {
    LOGE ("bank_init failed: %d", static_cast<int> (st));
    return JNI_ERR;
  }

  // Register the acoustic (microphone) stage — the mic is the primary PLHR
  // receiver on a stock phone (ENF/hum path), unlike the 100 Hz-capped
  // magnetometer which cannot resolve 60 Hz (Nyquist = 50 Hz).
  _S_bank_config._M_stages[3]._M_vtable = &zzs::kernel::feat::acoustic_vtable ();
  _S_bank_config._M_stages[3]._M_state = _S_acoustic_state;
  _S_bank_config._M_stage_count = 4;
  // Configure + reset the newly added acoustic stage (bank_init already did
  // stages 0-2 when stage_count was 3).
  if (const zzs::kernel::Status cst =
          _S_bank_config._M_stages[3]._M_vtable->_M_configure (_S_bank_config._M_stages[3]._M_state);
      cst != zzs::kernel::Status::OK)
    LOGE ("acoustic configure failed: %d", static_cast<int> (cst));
  _S_bank_config._M_stages[3]._M_vtable->_M_reset (_S_bank_config._M_stages[3]._M_state);

  // Start AAudio mic capture (UNPROCESSED, 48 kHz, mono, float).
  auto aaudio_st = _aaudio_start ();
  if (!zzs::kernel::status_ok (aaudio_st))
    LOGE ("aaudio_start failed: %d, acoustic stage will produce zeros", static_cast<int> (aaudio_st));

  LOGI ("OpenZZS bridge initialized (model=%u bytes, 4 stages)", zzs::mllee_native::model::kBlobSize);
  return JNI_OK;
}

JNIEXPORT void JNICALL
Java_com_openzzs_SensrBridge_shutdownNative (JNIEnv *env, jobject thiz) noexcept {
  _aaudio_stop ();
  if (!_S_listener)
    return;

  zzs::kernel::bank_reset (_S_bank);
  env->DeleteGlobalRef (_S_listener);
  _S_listener = nullptr;
  _M_on_features = nullptr;
  _M_on_inference = nullptr;
  LOGI ("OpenZZS bridge shutdown");
}

JNIEXPORT void JNICALL
Java_com_openzzs_SensrBridge_feedSampleNative (JNIEnv *env, jobject thiz, jint channel, jshort sample_q15,
                                               jlong timestamp_ns) noexcept {
  static uint32_t _S_jni_debug_count = 0;
  if (_S_jni_debug_count < 8u) {
    LOGE ("DEBUG jni_feed: ch=%d q15=%d ts=%llu listener=%p", channel, sample_q15,
          static_cast<unsigned long long> (timestamp_ns), static_cast<void *> (_S_listener));
    _S_jni_debug_count++;
  }
  if (!_S_listener)
    return;

  auto ch = static_cast<zzs::kernel::sensor::Channel> (channel);
  auto sq15 = static_cast<int16_t> (sample_q15);
  auto ts = static_cast<uint64_t> (timestamp_ns);

  feed_native_sample (ch, sq15, ts);
}

JNIEXPORT jobjectArray JNICALL
Java_com_openzzs_SensrBridge_getFeaturesNative (JNIEnv *env, jobject thiz) noexcept {
  if (!_S_listener)
    return env->NewObjectArray (0, env->FindClass ("java/lang/String"), nullptr);

  jclass string_cls = env->FindClass ("java/lang/String");
  jobjectArray result = env->NewObjectArray (zzs::kernel::_S_feature_count, string_cls, nullptr);

  int8_t row[zzs::kernel::_S_feature_count];
  auto st = zzs::kernel::bank_produce_row (_S_bank, row);
  if (st == zzs::kernel::Status::NOT_READY)
    return result;
  if (!zzs::kernel::status_ok (st)) {
    LOGE ("bank_produce_row failed: %d", static_cast<int> (st));
    return result;
  }

  for (uint32_t i = 0; i < zzs::kernel::_S_feature_count; ++i) {
    const char *name = zzs::kernel::meta::feat_name (i).data ();
    std::string entry = std::string (name) + "=" + std::to_string (static_cast<int> (row[i]));
    jstring jentry = env->NewStringUTF (entry.c_str ());
    env->SetObjectArrayElement (result, i, jentry);
    env->DeleteLocalRef (jentry);
  }

  return result;
}

JNIEXPORT jint JNICALL
JNI_OnLoad (JavaVM *vm, void *reserved) {
  JNIEnv *env;
  if (vm->GetEnv (reinterpret_cast<void **> (&env), JNI_VERSION_1_6) != JNI_OK)
    return JNI_ERR;

  jclass cls = env->FindClass ("com/openzzs/SensrBridge");
  if (cls == nullptr)
    return JNI_ERR;

  static const JNINativeMethod methods[] = {
      {"initNative", "()I", reinterpret_cast<void *> (Java_com_openzzs_SensrBridge_initNative)},
      {"shutdownNative", "()V", reinterpret_cast<void *> (Java_com_openzzs_SensrBridge_shutdownNative)},
      {"feedSampleNative", "(ISJ)V", reinterpret_cast<void *> (Java_com_openzzs_SensrBridge_feedSampleNative)},
      {"getFeaturesNative", "()[Ljava/lang/String;",
       reinterpret_cast<void *> (Java_com_openzzs_SensrBridge_getFeaturesNative)},
  };

  if (env->RegisterNatives (cls, methods, sizeof (methods) / sizeof (methods[0])) < 0)
    return JNI_ERR;

  return JNI_VERSION_1_6;
}

} // extern "C"
