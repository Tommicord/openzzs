/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#pragma once

#include "mllee-native/dataset_types.hpp"
#include "mllee-native/lcg16.hpp"
#include "mllee-native/q11_q15_q31.hpp"

#include <cstdint>

/**
 * @brief Compile-time dataset generator for power outage detection.
 *
 * Generates synthetic 38-feature windows across 5 overlapping time steps,
 * labelled normal/outage/transition.  All generation happens in consteval,
 * and the resulting dataset is stored in .rodata.  No heap, no float,
 * no std:: dependencies.
 * @{
 */
namespace zzs::mllee_native::dataset_gen {

using zzs::mllee_native::dataset::Dataset;
using zzs::mllee_native::dataset::FeatureWindow;
using zzs::mllee_native::dataset::GenParam;
using zzs::mllee_native::dataset::kFeatureCount;
using zzs::mllee_native::dataset::kSeqLen;
using zzs::mllee_native::dataset::Label;
using zzs::mllee_native::dataset::TrainingSample;
using zzs::mllee_native::fx::Q11;
using zzs::mllee_native::fx::q11_add;
using zzs::mllee_native::fx::q11_mul;
using zzs::mllee_native::fx::Q15;
using zzs::mllee_native::fx::q15_abs;
using zzs::mllee_native::fx::q15_add;
using zzs::mllee_native::fx::q15_mul;
using zzs::mllee_native::fx::Q31;
using zzs::mllee_native::rng::Lcg16;

/** @brief Feature block offsets within the 38-feature vector. */
namespace feat {
inline constexpr uint32_t plhr_start = 0u;      /* 0-23: PLHR 3 axes x 4 harmonics x 2 grids */
inline constexpr uint32_t vib_start = 24u;      /* 24-27: vibration 4 features */
inline constexpr uint32_t flicker_start = 28u;  /* 28-31: flicker 4 features */
inline constexpr uint32_t acoustic_start = 32u; /* 32-34: acoustic 3 features */
inline constexpr uint32_t reserved_start = 35u; /* 35-37: reserved */
} // namespace feat

/** @brief Clamp an int32 to int16 range. */
[[nodiscard]] constexpr int16_t
clamp_i16 (int32_t v) noexcept {
    return v > 32767 ? 32767 : (v < -32768 ? -32768 : static_cast<int16_t> (v));
}

/** @brief Generate one Q15 feature value in [-1, 1) scaled by gain. */
[[nodiscard]] constexpr int16_t
gen_feature_q15 (Lcg16 &rng, int16_t gain_q15) noexcept {
    /* next_q15 returns [0, 32767]; center and scale by gain. */
    int32_t raw = static_cast<int32_t> (rng.next_q15 ()._M_raw) - 16384; /* [-16384, 16383] */
    int32_t scaled = (raw * static_cast<int32_t> (gain_q15)) >> 15;
    return static_cast<int16_t> (clamp_i16 (scaled));
}

/** @brief Fill a window with class-appropriate feature patterns. */
constexpr void
fill_window (FeatureWindow &win, Lcg16 &rng, Label label, uint32_t window_index) noexcept {
    /* Class-specific base levels (Q15). */
    int16_t plhr_level, vib_level, flicker_level, acoustic_level;

    switch (label) {
    case Label::Normal:
        plhr_level = 20000; /* strong grid hum */
        vib_level = 8000;
        flicker_level = 5000;
        acoustic_level = 12000;
        break;
    case Label::Outage:
        plhr_level = 500; /* hum gone */
        vib_level = 2000;
        flicker_level = 25000; /* light DC drop / high flicker */
        acoustic_level = 3000;
        break;
    case Label::Transition:
        plhr_level = 10000;    /* sagging hum */
        vib_level = 15000;     /* mechanical stress */
        flicker_level = 15000; /* flickering */
        acoustic_level = 8000;
        break;
    }

    /* Window index adds slight temporal drift (0-4). */
    int16_t drift = static_cast<int16_t> (window_index * 1000);

    /* PLHR block: 3 axes x 4 harmonics x 2 grids = 24 features (0-23). */
    for (uint32_t a = 0; a < 3u; ++a) {
        for (uint32_t g = 0; g < 2u; ++g) {
            for (uint32_t h = 0; h < 4u; ++h) {
                uint32_t idx = feat::plhr_start + a * 8u + g * 4u + h;
                int16_t harmonic_atten = static_cast<int16_t> (32767 - (h * 6000));
                int16_t base = static_cast<int16_t> ((static_cast<int32_t> (plhr_level) * harmonic_atten) >> 15);
                base = static_cast<int16_t> ((static_cast<int32_t> (base) + drift) > 32767
                                                 ? 32767
                                                 : ((static_cast<int32_t> (base) + drift) < -32768
                                                        ? -32768
                                                        : static_cast<int16_t> (base + drift)));
                win._M_features[idx] = q15_mul (Q15{base}, Q15{gen_feature_q15 (rng, 16000)})._M_raw;
            }
        }
    }

    /* Vibration block: 4 features (24-27). */
    win._M_features[feat::vib_start + 0] =
        q15_mul (Q15{vib_level}, Q15{gen_feature_q15 (rng, 20000)})._M_raw; /* rms_xyz */
    win._M_features[feat::vib_start + 1] =
        q15_mul (Q15{static_cast<int16_t> (vib_level + 5000)}, Q15{gen_feature_q15 (rng, 18000)})._M_raw; /* peak_xyz */
    win._M_features[feat::vib_start + 2] =
        q15_mul (Q15{12000}, Q15{gen_feature_q15 (rng, 10000)})._M_raw; /* spectral_centroid */
    win._M_features[feat::vib_start + 3] =
        q15_mul (Q15{plhr_level}, Q15{gen_feature_q15 (rng, 15000)})._M_raw; /* power_50hz */

    /* Flicker block: 4 features (28-31). */
    win._M_features[feat::flicker_start + 0] =
        q15_mul (Q15{flicker_level}, Q15{gen_feature_q15 (rng, 20000)})._M_raw; /* dc_level */
    win._M_features[feat::flicker_start + 1] =
        q15_mul (Q15{15000}, Q15{gen_feature_q15 (rng, 12000)})._M_raw; /* 100hz_power */
    win._M_features[feat::flicker_start + 2] =
        q15_mul (Q15{static_cast<int16_t> (flicker_level >> 1)}, Q15{gen_feature_q15 (rng, 10000)})
            ._M_raw; /* flicker_index */
    win._M_features[feat::flicker_start + 3] =
        q15_mul (Q15{10000}, Q15{gen_feature_q15 (rng, 14000)})._M_raw; /* flicker_drop */

    /* Acoustic block: 3 features (32-34). */
    win._M_features[feat::acoustic_start + 0] =
        q15_mul (Q15{acoustic_level}, Q15{gen_feature_q15 (rng, 20000)})._M_raw; /* 100hz_power */
    win._M_features[feat::acoustic_start + 1] = q15_mul (Q15{8000}, Q15{gen_feature_q15 (rng, 10000)})._M_raw; /* thd */
    win._M_features[feat::acoustic_start + 2] =
        q15_mul (Q15{acoustic_level}, Q15{gen_feature_q15 (rng, 18000)})._M_raw; /* level_db */

    /* Reserved: 3 features (35-37) = 0. */
    win._M_features[35] = 0;
    win._M_features[36] = 0;
    win._M_features[37] = 0;
}

/** @brief Determine label for sample index based on class ratios (per-mille). */
[[nodiscard]] constexpr Label
label_for_index (uint32_t idx, uint32_t total, const GenParam &p) noexcept {
    uint32_t cum_normal = p._M_normal_ratio;
    uint32_t cum_outage = cum_normal + p._M_outage_ratio;
    uint32_t position = (static_cast<uint64_t> (idx) * 1000u) / total;

    if (position < cum_normal)
        return Label::Normal;
    if (position < cum_outage)
        return Label::Outage;
    return Label::Transition;
}

/** @brief Generate a complete dataset at compile time. */
[[nodiscard]] constexpr auto
generate_dataset (const GenParam &p) noexcept {
    Dataset<64> ds{}; /* Fixed-size compile-time dataset */
    Lcg16 rng = Lcg16::seed (p._M_seed);

    const uint32_t n = p._M_num_samples > 64 ? 64 : p._M_num_samples;
    for (uint32_t i = 0; i < n; ++i) {
        Label lbl = label_for_index (i, n, p);
        for (uint32_t w = 0; w < kSeqLen; ++w) {
            fill_window (ds._M_samples[i]._M_windows[w], rng, lbl, w);
        }
        ds._M_samples[i]._M_label = static_cast<uint8_t> (lbl);
    }
    ds._M_count = n;
    return ds;
}

/** @} */
} // namespace zzs::mllee_native::dataset_gen
