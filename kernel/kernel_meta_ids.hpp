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

#include "kernel/kernel_config.hpp"
#include "std/log.hpp"
#include "std/string_view.hpp"

#include <cstdint>

/**
 * @brief Feature index table (feat_ids).
 *
 * The model's input vector is `_S_feature_count` (38) int8-coded features.  This
 * table is the single place that defines the order; appending a feature is
 * allowed only at the end so the exported blob's metadata stays compatible.
 * Indices are referenced by the feature stages and the mllee blob metadata.
 *
 * Layout (35 used, 3 reserved at the end):
 *   PLHR (mag):        0..23   = 24  (3 axes x 4 harmonics x 2 grids)
 *   Vibration (accel): 24..27  = 4   (rms_xyz, peak_xyz, spectral_centroid, 50hz_power)
 *   Flicker (light):   28..31  = 4   (dc_level, 100hz_power, flicker_index, flicker_drop)
 *   Acoustic (sound):  32..34  = 3   (100hz_power, thd, level_db)
 *   Reserved:          35..37  = 3
 * @{
 */
namespace zzs::kernel::meta {

/** @brief Total feature count (must equal _S_feature_count). */
inline constexpr uint32_t _S_feature_total = zzs::kernel::_S_feature_count;

/* ---- PLHR feature block (mag axes X,Y,Z) ---- */

/** @brief Start index of the PLHR block. */
inline constexpr uint32_t _S_feat_plhr_start = 0u;
/** @brief PLHR per-axis harmonic power count (4 harmonics x 2 grids). */
inline constexpr uint32_t _S_feat_plhr_per_axis = 8u;
/** @brief PLHR block length = 3 axes * 8. */
inline constexpr uint32_t _S_feat_plhr_count = 24u;

/** @brief Index of mag axis `axis` harmonic `harm` on grid `grid` (0=50Hz,1=60Hz). */
[[nodiscard]] inline constexpr uint32_t
feat_plhr_index (const uint32_t axis, const uint32_t grid, const uint32_t harm) noexcept {
  return _S_feat_plhr_start + axis * _S_feat_plhr_per_axis + grid * 4u + harm;
}

/* ---- Vibration feature block (accel) ---- */

/** @brief Start index of the vibration block. */
inline constexpr uint32_t _S_feat_vib_start = _S_feat_plhr_start + _S_feat_plhr_count;
/** @brief Vibration block length. */
inline constexpr uint32_t _S_feat_vib_count = 4u;

namespace vib {
inline constexpr uint32_t rms_xyz = 0;           /* accel_rms_xyz */
inline constexpr uint32_t peak_xyz = 1;          /* accel_peak_xyz */
inline constexpr uint32_t spectral_centroid = 2; /* accel_spectral_centroid */
inline constexpr uint32_t power_50hz = 3;        /* accel_50hz_power */
} // namespace vib

/* ---- Flicker feature block (light) ---- */

/** @brief Start index of the flicker block. */
inline constexpr uint32_t _S_feat_flicker_start = _S_feat_vib_start + _S_feat_vib_count;
/** @brief Flicker block length. */
inline constexpr uint32_t _S_feat_flicker_count = 4u;

namespace flicker {
inline constexpr uint32_t dc_level = 0;      /* light_dc_level */
inline constexpr uint32_t power_100hz = 1;   /* light_100hz_power */
inline constexpr uint32_t flicker_index = 2; /* light_flicker_index */
inline constexpr uint32_t flicker_drop = 3;  /* light_flicker_drop */
} // namespace flicker

/* ---- Acoustic feature block (sound) ---- */

/** @brief Start index of the acoustic block. */
inline constexpr uint32_t _S_feat_acoustic_start = _S_feat_flicker_start + _S_feat_flicker_count;
/** @brief Acoustic block length. */
inline constexpr uint32_t _S_feat_acoustic_count = 3u;

namespace acoustic {
inline constexpr uint32_t power_100hz = 0; /* sound_100hz_power */
inline constexpr uint32_t thd = 1;         /* sound_thd */
inline constexpr uint32_t level_db = 2;    /* sound_level_dB */
} // namespace acoustic

/** @brief First reserved index. */
inline constexpr uint32_t _S_feat_reserved_start = _S_feat_acoustic_start + _S_feat_acoustic_count;

/** @brief Verify the block lengths do not exceed the feature count. */
static_assert (_S_feat_plhr_count + _S_feat_vib_count + _S_feat_flicker_count + _S_feat_acoustic_count
                   <= _S_feature_total,
               "feature block lengths must not exceed _S_feature_count");

/** @brief Name of a feature index, for the host-side telemetry boundary. */
[[nodiscard]] inline zzs::base::SimpleStringView
feat_name (const uint32_t index) noexcept {
  switch (index) {
  /* PLHR mag X */
  case 0:
    return "plhr_mag_x_h50_h1";
  case 1:
    return "plhr_mag_x_h50_h2";
  case 2:
    return "plhr_mag_x_h50_h3";
  case 3:
    return "plhr_mag_x_h50_h4";
  case 4:
    return "plhr_mag_x_h60_h1";
  case 5:
    return "plhr_mag_x_h60_h2";
  case 6:
    return "plhr_mag_x_h60_h3";
  case 7:
    return "plhr_mag_x_h60_h4";
  /* PLHR mag Y */
  case 8:
    return "plhr_mag_y_h50_h1";
  case 9:
    return "plhr_mag_y_h50_h2";
  case 10:
    return "plhr_mag_y_h50_h3";
  case 11:
    return "plhr_mag_y_h50_h4";
  case 12:
    return "plhr_mag_y_h60_h1";
  case 13:
    return "plhr_mag_y_h60_h2";
  case 14:
    return "plhr_mag_y_h60_h3";
  case 15:
    return "plhr_mag_y_h60_h4";
  /* PLHR mag Z */
  case 16:
    return "plhr_mag_z_h50_h1";
  case 17:
    return "plhr_mag_z_h50_h2";
  case 18:
    return "plhr_mag_z_h50_h3";
  case 19:
    return "plhr_mag_z_h50_h4";
  case 20:
    return "plhr_mag_z_h60_h1";
  case 21:
    return "plhr_mag_z_h60_h2";
  case 22:
    return "plhr_mag_z_h60_h3";
  case 23:
    return "plhr_mag_z_h60_h4";
  /* Vibration */
  case 24:
    return "vib_accel_rms_xyz";
  case 25:
    return "vib_accel_peak_xyz";
  case 26:
    return "vib_accel_spectral_centroid";
  case 27:
    return "vib_accel_50hz_power";
  /* Flicker */
  case 28:
    return "flicker_light_dc_level";
  case 29:
    return "flicker_light_100hz_power";
  case 30:
    return "flicker_light_flicker_index";
  case 31:
    return "flicker_light_flicker_drop";
  /* Acoustic */
  case 32:
    return "acoustic_sound_100hz_power";
  case 33:
    return "acoustic_sound_thd";
  case 34:
    return "acoustic_sound_level_db";
  default:
    return "reserved";
  }
}

/** @} */
} // namespace zzs::kernel::meta
