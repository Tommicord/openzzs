/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

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
#include "kernel/kernel_status.hpp"
#include "kernel/kernel_util_fixed.hpp"
#include "mllee-native/model.hpp"
#include "mllee/core.hpp"
#include "std/fmt.hpp"
#include "std/io.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

#if defined(__ANDROID__)
#include "kernel/sensor-driver/platforms/android/platform_hal.hpp"

#include <android/log.h>
#define MLOGI(...) __android_log_print (ANDROID_LOG_INFO, "OpenZZS", __VA_ARGS__)
#define MLOGE(...) __android_log_print (ANDROID_LOG_ERROR, "OpenZZS", __VA_ARGS__)
#else
#define MLOGI(...) std::printf (__VA_ARGS__)
#define MLOGE(...) std::fprintf (stderr, __VA_ARGS__)
#endif

namespace zzs::mllee_native::model {} // namespace zzs::mllee_native::model

namespace zzs::monitor {

/** @brief Host-side mock HAL: emits deterministic samples on enabled channels. */
class _S_MockHal {
  public:
    _S_MockHal () noexcept = default;

    [[nodiscard]] zzs::kernel::Status
    init () noexcept {
        _M_ts = 0ull;
        _M_cursor = 0u;
        for (uint32_t c = 0u; c < zzs::kernel::sensor::channel_count (); ++c)
            _M_enabled[c] = false;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] zzs::kernel::Status
    set_enabled (const zzs::kernel::sensor::Channel channel, const bool enabled) noexcept {
        _M_enabled[static_cast<uint32_t> (channel)] = enabled;
        return zzs::kernel::Status::OK;
    }

    [[nodiscard]] uint32_t
    rate_hz (const zzs::kernel::sensor::Channel channel) noexcept {
        return zzs::kernel::sensor::channel_rate_hz (channel);
    }

    /** @brief Read the next sample on a round-robin channel schedule. */
    [[nodiscard]] zzs::kernel::Status
    read_sample (zzs::kernel::sensor::Sample &out) noexcept {
        while (true) {
            const uint32_t index = _M_cursor % zzs::kernel::sensor::channel_count ();
            _M_cursor++;
            if (!_M_enabled[index])
                continue;
            out._M_channel = static_cast<zzs::kernel::sensor::Channel> (index);
            out._M_sample_q15 = _S_code_for (index, _M_ts);
            out._M_timestamp_ns = _M_ts;
            _M_ts += zzs::kernel::_S_grid_period_ns;
            return zzs::kernel::Status::OK;
        }
    }

  private:
    [[nodiscard]] static int16_t
    _S_code_for (const uint32_t channel, const uint64_t ts_ns) noexcept {
        const uint64_t mix = ts_ns * 0x9E3779B97F4A7C15ull + static_cast<uint64_t> (channel) * 2654435761u;
        const uint32_t bits = static_cast<uint32_t> (mix ^ (mix >> 32));
        const int32_t code = static_cast<int32_t> ((bits % 4096u) - 2048u);
        return static_cast<int16_t> (code);
    }

    uint64_t _M_ts = 0ull;
    uint32_t _M_cursor = 0u;
    bool _M_enabled[zzs::kernel::sensor::channel_count ()]{};
};

} // namespace zzs::monitor

namespace {

/** @brief The mock HAL instance (file-local, owned by main). */
zzs::monitor::_S_MockHal g_mock_hal;

/** @brief Vtable free functions that delegate to g_mock_hal. */
[[nodiscard]] zzs::kernel::Status
mock_read_sample (zzs::kernel::sensor::Sample &out) noexcept {
    return g_mock_hal.read_sample (out);
}

[[nodiscard]] zzs::kernel::Status
mock_set_channel_enabled (const zzs::kernel::sensor::Channel channel, const bool enabled) noexcept {
    return g_mock_hal.set_enabled (channel, enabled);
}

[[nodiscard]] uint32_t
mock_channel_rate_hz (const zzs::kernel::sensor::Channel channel) noexcept {
    return g_mock_hal.rate_hz (channel);
}

[[nodiscard]] zzs::kernel::Status
mock_init (void) noexcept {
    return g_mock_hal.init ();
}

/** @brief The mock HAL vtable registered with the kernel HAL layer. */
const zzs::kernel::hal::_S_SensorHal g_hal_vtable = {
    mock_read_sample,
    mock_set_channel_enabled,
    mock_channel_rate_hz,
    mock_init,
};

} // namespace

/** @brief Run the preprocessing+inference loop using the given HAL. */
[[nodiscard]] int
run_monitor (zzs::kernel::hal::_S_SensorHal const *hal) {
    zzs::kernel::clock::_S_GridTracker grid{};
    zzs::kernel::_S_BankConfig bank_config{};
    zzs::kernel::_S_PreprocessingBank bank{};
    zzs::kernel::frames::Sequence frame{};
    zzs::mllee::Core core{};

    /* Install the HAL. */
    if (const zzs::kernel::Status st = zzs::kernel::hal::hal_install (hal); !zzs::kernel::status_ok (st)) {
        MLOGE ("hal_install failed: %s\n", zzs::kernel::status_name (st).data ());
        return 1;
    }

    /* Initialize the HAL and clock source. */
    if (const zzs::kernel::Status st = hal->_M_init (); !zzs::kernel::status_ok (st)) {
        MLOGE ("hal_init failed: %s\n", zzs::kernel::status_name (st).data ());
        return 1;
    }
    if (const zzs::kernel::Status st = zzs::kernel::clock::clock_source_install (
            [] () noexcept -> zzs::kernel::clock::Nanoseconds { return 0ull; });
        !zzs::kernel::status_ok (st)) {
        MLOGE ("clock_source_install failed: %s\n", zzs::kernel::status_name (st).data ());
        return 1;
    }
    if (const zzs::kernel::Status st = zzs::kernel::clock::grid_tracker_init (grid, 0ull);
        !zzs::kernel::status_ok (st)) {
        MLOGE ("grid_tracker_init failed: %s\n", zzs::kernel::status_name (st).data ());
        return 1;
    }

    /* Build the model blob from trained or init weights. */
    if (const zzs::mllee::Status st = zzs::mllee_native::model::init_model_vtable (); !zzs::mllee::status_ok (st)) {
        MLOGE ("init_model_vtable failed: %s\n", zzs::mllee::status_name (st));
        return 1;
    }

    /* Configure resamplers per channel (rate -> 100 Hz grid). */
    zzs::kernel::dsp::_S_Resampler resamplers[zzs::kernel::sensor::channel_count ()]{};
    constexpr uint32_t k_grid_rate_hz = zzs::kernel::_S_grid_rate_hz;
    for (uint32_t c = 0u; c < zzs::kernel::sensor::channel_count (); ++c) {
        const uint32_t in_rate = zzs::kernel::sensor::channel_rate_hz (static_cast<zzs::kernel::sensor::Channel> (c));
        if (const zzs::kernel::Status st =
                zzs::kernel::dsp::resampler_configure (resamplers[c], in_rate, k_grid_rate_hz);
            !zzs::kernel::status_ok (st)) {
            MLOGE ("resampler_configure ch=%u failed: %s\n", c, zzs::kernel::status_name (st).data ());
            return 1;
        }
    }

    /* Enable MAG/ACCEL/LIGHT channels. */
    constexpr zzs::kernel::sensor::Channel k_enabled_channels[] = {
        zzs::kernel::sensor::Channel::MAG_X,   zzs::kernel::sensor::Channel::MAG_Y,
        zzs::kernel::sensor::Channel::MAG_Z,   zzs::kernel::sensor::Channel::ACCEL_X,
        zzs::kernel::sensor::Channel::ACCEL_Y, zzs::kernel::sensor::Channel::ACCEL_Z,
        zzs::kernel::sensor::Channel::LIGHT,   zzs::kernel::sensor::Channel::SOUND_ENVELOPE,
    };
#if defined(__ANDROID__)
    /* On Android, use the HAL's set_channel_enabled. */
    for (const zzs::kernel::sensor::Channel ch : k_enabled_channels) {
        hal->_M_set_channel_enabled (ch, true);
    }
#else
    for (const zzs::kernel::sensor::Channel ch : k_enabled_channels) {
        static_cast<void> (g_mock_hal.set_enabled (ch, true));
    }
#endif

    /* Stage state storage (stack-allocated, sized to each stage's state struct). */
    alignas (4) int8_t plhr_state[sizeof (zzs::kernel::feat::_S_PlhrState)]{};
    alignas (4) int8_t vib_state[sizeof (zzs::kernel::feat::_S_VibrationState)]{};
    alignas (4) int8_t flicker_state[sizeof (zzs::kernel::feat::_S_FlickerState)]{};
    alignas (4) int8_t acoustic_state[sizeof (zzs::kernel::feat::_S_AcousticState)]{};

    /* Register stages in the bank config. */
    bank_config._M_stages[0]._M_vtable = &zzs::kernel::feat::plhr_vtable ();
    bank_config._M_stages[0]._M_state = plhr_state;
    bank_config._M_stages[1]._M_vtable = &zzs::kernel::feat::vibration_vtable ();
    bank_config._M_stages[1]._M_state = vib_state;
    bank_config._M_stages[2]._M_vtable = &zzs::kernel::feat::flicker_vtable ();
    bank_config._M_stages[2]._M_state = flicker_state;
    bank_config._M_stages[3]._M_vtable = &zzs::kernel::feat::acoustic_vtable ();
    bank_config._M_stages[3]._M_state = acoustic_state;
    bank_config._M_stage_count = 4u;
    bank_config._M_grid = grid;

    if (const zzs::kernel::Status st = zzs::kernel::bank_init (bank, bank_config); !zzs::kernel::status_ok (st)) {
        MLOGE ("bank_init failed: %s\n", zzs::kernel::status_name (st).data ());
        return 1;
    }

    /* Main loop: feed samples until a frame completes, then run inference. */
    const uint32_t k_max_iterations = 1000000u;
    uint32_t inferences = 0u;
    for (uint32_t it = 0u; it < k_max_iterations && inferences < 1000u; ++it) {
        zzs::kernel::sensor::Sample sample{};
        const zzs::kernel::Status st = hal->_M_read_sample (sample);
        if (!zzs::kernel::status_ok (st)) {
            continue; /* Skip unavailable samples on Android. */
        }

        /* Route through the channel resampler to the grid; only resampled grid
           ticks are fed into the preprocessing bank. */
        bool produced = false;
        int16_t out_q15 = 0;
        const zzs::kernel::Status rst = zzs::kernel::dsp::resampler_push (
            resamplers[static_cast<uint32_t> (sample._M_channel)], sample._M_sample_q15, &produced, &out_q15);
        if (!zzs::kernel::status_ok (rst)) {
            MLOGE ("resampler_push failed: %s\n", zzs::kernel::status_name (rst).data ());
            return 1;
        }
        if (!produced)
            continue;

        zzs::kernel::sensor::Sample grid_sample{};
        grid_sample._M_channel = sample._M_channel;
        grid_sample._M_sample_q15 = out_q15;
        grid_sample._M_timestamp_ns = sample._M_timestamp_ns;
        const zzs::kernel::Status fst = zzs::kernel::bank_feed_sample (bank, grid_sample);
        if (fst == zzs::kernel::Status::NOT_READY)
            continue;
        if (!zzs::kernel::status_ok (fst)) {
            MLOGE ("bank_feed_sample failed: %s\n", zzs::kernel::status_name (fst).data ());
            return 1;
        }

        int8_t row[zzs::kernel::_S_feature_count];
        const zzs::kernel::Status pst = zzs::kernel::bank_produce_row (bank, row);
        if (pst == zzs::kernel::Status::NOT_READY)
            continue;
        if (!zzs::kernel::status_ok (pst)) {
            MLOGE ("bank_produce_row failed: %s\n", zzs::kernel::status_name (pst).data ());
            return 1;
        }

        if (!frame.apply_row (row))
            continue; /* frame not complete yet */

        /* Frame complete: run mllee inference on the 190 int8 codes using the
           OUTPRED model blob from the VTable. */
        if (const zzs::mllee::Status st = zzs::mllee_native::model::init_model_vtable (); !zzs::mllee::status_ok (st)) {
            MLOGE ("init_model_vtable failed: %s\n", zzs::mllee::status_name (st));
            frame.reset ();
            continue;
        }
        const uint32_t idx = zzs::mllee_native::model::find_outpred ();
        if (idx == zzs::mllee_native::model::ModelVTable<>::k_npos) {
            MLOGE ("outpred model not registered in VTable\n");
            frame.reset ();
            continue;
        }
        const zzs::mllee_native::model::VTableEntry &entry = zzs::mllee_native::model::g_model_vtable[idx];
        const auto result =
            core.run (entry._M_blob, entry._M_blob_size, frame.data (), zzs::mllee_native::model::kInputDim);
        if (!result) {
            MLOGE ("inference failed: %s\n",
                   zzs::mllee::status_name (static_cast<zzs::mllee::Status> (result.error ().kind ())));
            frame.reset ();
            continue;
        }

        const int8_t *const prob_codes = result.value ()._M_data;
        uint32_t probs[3];
        uint32_t predicted = 0u;
        for (uint32_t i = 0u; i < 3u; ++i) {
            probs[i] = (static_cast<uint32_t> (prob_codes[i]) * 100u) / 127u;
            if (probs[i] > 100u)
                probs[i] = 100u;
            if (prob_codes[i] > prob_codes[predicted])
                predicted = i;
        }

        const char *label = (predicted == 0u) ? "normal" : (predicted == 1u) ? "outage" : "transition";
        MLOGI ("inference %u: %s P_normal=%u%% P_outage=%u%% P_transition=%u%%\n", inferences + 1u, label, probs[0],
               probs[1], probs[2]);

        inferences++;
        frame.reset ();
    }

    MLOGI ("monitor: %u inferences complete\n", inferences);
    return 0;
}

int
main (int argc, char **argv) {
    (void) argc;
    (void) argv;

#if defined(__ANDROID__)
    /* On Android, use the platform sensor HAL. */
    auto hal = zzs::kernel::sensor::driver::platform::android::create_android_sensor_hal ();
    if (hal == nullptr) {
        MLOGE ("Failed to create Android sensor HAL\n");
        return 1;
    }
    return run_monitor (hal);
#else
    /* On host, use the mock HAL. */
    return run_monitor (&g_hal_vtable);
#endif
}

/* ======================================================================== */
/* JNI entry points for Android (compiled into libopenzzs-android.so).      */
/* ======================================================================== */

#if defined(__ANDROID__)
#include <jni.h>

namespace {

/** @brief Global pipeline state for Android JNI. */
struct _S_JniState {
    zzs::kernel::clock::_S_GridTracker grid{};
    zzs::kernel::_S_BankConfig bank_config{};
    zzs::kernel::_S_PreprocessingBank bank{};
    zzs::kernel::frames::Sequence frame{};
    zzs::mllee::Core core{};
    zzs::kernel::dsp::_S_Resampler resamplers[zzs::kernel::sensor::channel_count ()]{};
    const zzs::kernel::hal::_S_SensorHal *hal = nullptr;
    alignas (4) int8_t plhr_state[sizeof (zzs::kernel::feat::_S_PlhrState)]{};
    alignas (4) int8_t vib_state[sizeof (zzs::kernel::feat::_S_VibrationState)]{};
    alignas (4) int8_t flicker_state[sizeof (zzs::kernel::feat::_S_FlickerState)]{};
    alignas (4) int8_t acoustic_state[sizeof (zzs::kernel::feat::_S_AcousticState)]{};
    bool initialized = false;
};

_S_JniState g_jni;

} // namespace

extern "C" {

JNIEXPORT jint JNICALL
Java_com_openzzs_SensrBridge_initNative (JNIEnv *env, jobject thiz) {
    (void) env;
    (void) thiz;

    g_jni.hal = zzs::kernel::sensor::driver::platform::android::create_android_sensor_hal ();
    if (g_jni.hal == nullptr) {
        MLOGE ("Failed to create Android sensor HAL\n");
        return -1;
    }

    if (const zzs::kernel::Status st = zzs::kernel::hal::hal_install (g_jni.hal); !zzs::kernel::status_ok (st)) {
        MLOGE ("hal_install failed: %s\n", zzs::kernel::status_name (st).data ());
        return -1;
    }

    if (const zzs::kernel::Status st = g_jni.hal->_M_init (); !zzs::kernel::status_ok (st)) {
        MLOGE ("hal_init failed: %s\n", zzs::kernel::status_name (st).data ());
        return -1;
    }
    if (const zzs::kernel::Status st = zzs::kernel::clock::clock_source_install (
            [] () noexcept -> zzs::kernel::clock::Nanoseconds { return 0ull; });
        !zzs::kernel::status_ok (st)) {
        MLOGE ("clock_source_install failed: %s\n", zzs::kernel::status_name (st).data ());
        return -1;
    }
    if (const zzs::kernel::Status st = zzs::kernel::clock::grid_tracker_init (g_jni.grid, 0ull);
        !zzs::kernel::status_ok (st)) {
        MLOGE ("grid_tracker_init failed: %s\n", zzs::kernel::status_name (st).data ());
        return -1;
    }

    if (const zzs::mllee::Status st = zzs::mllee_native::model::init_model_vtable (); !zzs::mllee::status_ok (st)) {
        MLOGE ("init_model_vtable failed: %s\n", zzs::mllee::status_name (st));
        return -1;
    }

    constexpr uint32_t k_grid_rate_hz = zzs::kernel::_S_grid_rate_hz;
    for (uint32_t c = 0u; c < zzs::kernel::sensor::channel_count (); ++c) {
        const uint32_t in_rate = zzs::kernel::sensor::channel_rate_hz (static_cast<zzs::kernel::sensor::Channel> (c));
        if (const zzs::kernel::Status st =
                zzs::kernel::dsp::resampler_configure (g_jni.resamplers[c], in_rate, k_grid_rate_hz);
            !zzs::kernel::status_ok (st)) {
            MLOGE ("resampler_configure ch=%u failed: %s\n", c, zzs::kernel::status_name (st).data ());
            return -1;
        }
    }

    constexpr zzs::kernel::sensor::Channel k_enabled_channels[] = {
        zzs::kernel::sensor::Channel::MAG_X,   zzs::kernel::sensor::Channel::MAG_Y,
        zzs::kernel::sensor::Channel::MAG_Z,   zzs::kernel::sensor::Channel::ACCEL_X,
        zzs::kernel::sensor::Channel::ACCEL_Y, zzs::kernel::sensor::Channel::ACCEL_Z,
        zzs::kernel::sensor::Channel::LIGHT,   zzs::kernel::sensor::Channel::SOUND_ENVELOPE,
    };
    for (const zzs::kernel::sensor::Channel ch : k_enabled_channels) {
        g_jni.hal->_M_set_channel_enabled (ch, true);
    }

    g_jni.bank_config._M_stages[0]._M_vtable = &zzs::kernel::feat::plhr_vtable ();
    g_jni.bank_config._M_stages[0]._M_state = g_jni.plhr_state;
    g_jni.bank_config._M_stages[1]._M_vtable = &zzs::kernel::feat::vibration_vtable ();
    g_jni.bank_config._M_stages[1]._M_state = g_jni.vib_state;
    g_jni.bank_config._M_stages[2]._M_vtable = &zzs::kernel::feat::flicker_vtable ();
    g_jni.bank_config._M_stages[2]._M_state = g_jni.flicker_state;
    g_jni.bank_config._M_stages[3]._M_vtable = &zzs::kernel::feat::acoustic_vtable ();
    g_jni.bank_config._M_stages[3]._M_state = g_jni.acoustic_state;
    g_jni.bank_config._M_stage_count = 4u;
    g_jni.bank_config._M_grid = g_jni.grid;

    if (const zzs::kernel::Status st = zzs::kernel::bank_init (g_jni.bank, g_jni.bank_config);
        !zzs::kernel::status_ok (st)) {
        MLOGE ("bank_init failed: %s\n", zzs::kernel::status_name (st).data ());
        return -1;
    }

    g_jni.initialized = true;
    MLOGI ("openzzs Android JNI initialized\n");
    return 0;
}

JNIEXPORT jint JNICALL
Java_com_openzzs_SensrBridge_feedSampleNative (JNIEnv *env, jobject thiz, jint channel, jshort sample_q15,
                                               jlong timestamp_ns) {
    (void) env;
    (void) thiz;

    if (!g_jni.initialized) {
        return -1;
    }

    const auto ch = static_cast<zzs::kernel::sensor::Channel> (channel);

    bool produced = false;
    int16_t out_q15 = 0;
    const auto st = zzs::kernel::dsp::resampler_push (g_jni.resamplers[static_cast<uint32_t> (ch)], sample_q15,
                                                      &produced, &out_q15);
    if (!zzs::kernel::status_ok (st)) {
        return -1;
    }
    if (!produced)
        return 0;

    zzs::kernel::sensor::Sample grid_sample{};
    grid_sample._M_channel = ch;
    grid_sample._M_sample_q15 = out_q15;
    grid_sample._M_timestamp_ns = static_cast<uint64_t> (timestamp_ns);
    const auto fst = zzs::kernel::bank_feed_sample (g_jni.bank, grid_sample);
    if (fst == zzs::kernel::Status::NOT_READY)
        return 0;
    if (!zzs::kernel::status_ok (fst)) {
        return -1;
    }

    int8_t row[zzs::kernel::_S_feature_count];
    const auto pst = zzs::kernel::bank_produce_row (g_jni.bank, row);
    if (pst == zzs::kernel::Status::NOT_READY)
        return 0;
    if (!zzs::kernel::status_ok (pst)) {
        return -1;
    }

    if (!g_jni.frame.apply_row (row))
        return 0;

    const uint32_t jni_idx = zzs::mllee_native::model::find_outpred ();
    if (jni_idx == zzs::mllee_native::model::ModelVTable<>::k_npos)
        return -1;
    const zzs::mllee_native::model::VTableEntry &jni_entry = zzs::mllee_native::model::g_model_vtable[jni_idx];
    const auto result = g_jni.core.run (jni_entry._M_blob, jni_entry._M_blob_size, g_jni.frame.data (),
                                        zzs::mllee_native::model::kInputDim);
    if (!result) {
        return -1;
    }

    const int8_t *const prob_codes = result.value ()._M_data;
    uint32_t probs[3];
    uint32_t predicted = 0u;
    for (uint32_t i = 0u; i < 3u; ++i) {
        probs[i] = (static_cast<uint32_t> (prob_codes[i]) * 100u) / 127u;
        if (probs[i] > 100u)
            probs[i] = 100u;
        if (prob_codes[i] > prob_codes[predicted])
            predicted = i;
    }

    const char *label = (predicted == 0u) ? "normal" : (predicted == 1u) ? "outage" : "transition";
    MLOGI ("inference: %s P_normal=%u%% P_outage=%u%% P_transition=%u%%\n", label, probs[0], probs[1], probs[2]);

    g_jni.frame.reset ();
    return static_cast<jint> (predicted);
}

JNIEXPORT void JNICALL
Java_com_openzzs_SensrBridge_shutdownNative (JNIEnv *env, jobject thiz) {
    (void) env;
    (void) thiz;
    g_jni.initialized = false;
    g_jni.hal = nullptr;
    MLOGI ("openzzs Android JNI shutdown\n");
}

} // extern "C"
#endif // __ANDROID__