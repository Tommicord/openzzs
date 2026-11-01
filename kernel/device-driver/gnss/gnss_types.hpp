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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

#pragma once

#include "kernel/kernel_status.hpp"

#include <cstdint>
#include <cstring>

/**
 * @brief GNSS (GPS / GLONASS / Galileo / BeiDou) types.
 *
 * All coordinates are integer fixed-point: latitude/longitude are stored as
 * degrees * 10^7 (int32, resolution ~1 cm), altitude and speed as millimetre
 * / millimetre-per-second int32.  No floating point is used anywhere in the
 * GNSS module, matching the kernel's integer-only, no-heap, noexcept contract.
 * @{
 */
namespace zzs::kernel::device::driver::gnss {

/** @brief Resolution of lat/lon storage: degrees * 10^7. */
inline constexpr int64_t GnssLatLonScale = 10000000ll;

/** @brief Maximum valid latitude magnitude (90 deg * 1e7). */
inline constexpr int32_t GnssLatMax = 900000000;

/** @brief Maximum valid longitude magnitude (180 deg * 1e7). */
inline constexpr int32_t GnssLonMax = 1800000000;

/** @brief Maximum number of satellites tracked in a fix report. */
inline constexpr uint8_t GnssMaxSatellites = 16u;

/** @brief Invalid / missing satellite PRN sentinel. */
inline constexpr uint8_t GnssInvalidPrn = 0u;

/** @brief Invalid fix handle / handle not yet assigned. */
inline constexpr uint8_t GnssInvalidHandle = 0xffu;

// ---------------------------------------------------------------------------
// Constellation
// ---------------------------------------------------------------------------

/** @brief GNSS constellation a satellite or fix belongs to. */
enum class Constellation : uint8_t {
  Unknown = 0,
  Gps = 1,      /**< GPS (USA) */
  Glonass = 2,  /**< GLONASS (Russia) */
  Galileo = 3,  /**< Galileo (EU) */
  BeiDou = 4,   /**< BeiDou (China) */
  Qzss = 5,     /**< QZSS (Japan) */
  Sbas = 6,     /**< SBAS / WAAS / EGNOS */
};

/** @brief Return a short name for a constellation. */
[[nodiscard]] inline const char *
constellation_name (Constellation c) noexcept {
  switch (c) {
  case Constellation::Gps:
    return "gps";
  case Constellation::Glonass:
    return "glonass";
  case Constellation::Galileo:
    return "galileo";
  case Constellation::BeiDou:
    return "beidou";
  case Constellation::Qzss:
    return "qzss";
  case Constellation::Sbas:
    return "sbas";
  default:
    return "unknown";
  }
}

// ---------------------------------------------------------------------------
// Fix status
// ---------------------------------------------------------------------------

/** @brief GNSS fix quality / mode. */
enum class FixStatus : uint8_t {
  NoFix = 0,           /**< No fix available. */
  DeadReckoning = 1,   /**< Dead reckoning only. */
  Gps2d = 2,           /**< 2D fix (latitude/longitude only). */
  Gps3d = 3,           /**< 3D fix (latitude/longitude/altitude). */
  GpsDr = 4,           /**< GPS + dead reckoning. */
  TimeOnly = 5,        /**< Time-only fix. */
};

/** @brief Return a short name for a fix status. */
[[nodiscard]] inline const char *
fix_status_name (FixStatus s) noexcept {
  switch (s) {
  case FixStatus::NoFix:
    return "no_fix";
  case FixStatus::DeadReckoning:
    return "dead_reckoning";
  case FixStatus::Gps2d:
    return "gps_2d";
  case FixStatus::Gps3d:
    return "gps_3d";
  case FixStatus::GpsDr:
    return "gps_dr";
  case FixStatus::TimeOnly:
    return "time_only";
  default:
    return "unknown";
  }
}

// ---------------------------------------------------------------------------
// Coordinate
// ---------------------------------------------------------------------------

/** @brief Latitude or longitude in degrees * 10^7 (int32). */
using LatLon = int32_t;

/**
 * @brief Validate a latitude value.
 * @return true when -90e7 <= value <= +90e7.
 */
[[nodiscard]] inline constexpr bool
is_valid_lat (LatLon lat) noexcept {
  return lat >= -GnssLatMax && lat <= GnssLatMax;
}

/**
 * @brief Validate a longitude value.
 * @return true when -180e7 <= value <= +180e7.
 */
[[nodiscard]] inline constexpr bool
is_valid_lon (LatLon lon) noexcept {
  return lon >= -GnssLonMax && lon <= GnssLonMax;
}

/** @brief Convert degrees * 10^7 to a floating-point degrees value (host
 *         side only; not used in the hot path). */
[[nodiscard]] inline double
lat_lon_to_degrees (LatLon value) noexcept {
  return static_cast<double> (value) / static_cast<double> (GnssLatLonScale);
}

// ---------------------------------------------------------------------------
// Satellite
// ---------------------------------------------------------------------------

/** @brief One tracked satellite in view. */
struct Satellite {
  uint8_t _M_prn = GnssInvalidPrn;     /**< Satellite PRN / slot number. */
  Constellation _M_constellation = Constellation::Unknown;
  int8_t _M_elevation_deg = 0;         /**< Elevation in degrees (-90..90). */
  int16_t _M_azimuth_deg_tenths = 0;   /**< Azimuth in tenths of a degree (0..3599). */
  uint8_t _M_snr_dbhz = 0;             /**< Signal-to-noise ratio in dB-Hz. */
  bool _M_used_in_fix = false;         /**< Whether this SV contributed to the fix. */
};

/** @brief Clear a satellite to its invalid / empty state. */
[[nodiscard]] inline Satellite
satellite_make_invalid () noexcept {
  Satellite s{};
  s._M_prn = GnssInvalidPrn;
  s._M_constellation = Constellation::Unknown;
  return s;
}

/** @brief True when the satellite entry carries a valid PRN. */
[[nodiscard]] inline constexpr bool
satellite_is_valid (const Satellite &s) noexcept {
  return s._M_prn != GnssInvalidPrn;
}

// ---------------------------------------------------------------------------
// Fix
// ---------------------------------------------------------------------------

/** @brief One GNSS fix solution. All fields are integer fixed-point. */
struct Fix {
  uint8_t _M_handle = GnssInvalidHandle;  /**< Handle assigned by the driver. */
  FixStatus _M_status = FixStatus::NoFix;
  uint64_t _M_timestamp_ns = 0ull;        /**< Receiver time-of-fix, monotonic ns. */
  LatLon _M_lat = 0;                      /**< Latitude, degrees * 1e7. */
  LatLon _M_lon = 0;                      /**< Longitude, degrees * 1e7. */
  int32_t _M_alt_mm = 0;                  /**< Altitude above ellipsoid, mm. */
  int32_t _M_alt_msl_mm = 0;              /**< Altitude above mean sea level, mm. */
  int32_t _M_speed_mm_s = 0;              /**< Ground speed, mm/s. */
  int32_t _M_course_deg_tenths = 0;       /**< Course over ground, tenths of deg. */
  uint8_t _M_num_satellites = 0u;         /**< Satellites used in the fix. */
  Satellite _M_satellites[GnssMaxSatellites] = {};
  int16_t _M_hdop_tenths = 0;             /**< Horizontal dilution, tenths. */
  int16_t _M_vdop_tenths = 0;             /**< Vertical dilution, tenths. */
  int16_t _M_pdop_tenths = 0;             /**< Position dilution, tenths. */
};

/** @brief Clear a fix to its invalid / no-fix state. */
[[nodiscard]] inline Fix
fix_make_invalid () noexcept {
  Fix f{};
  f._M_handle = GnssInvalidHandle;
  f._M_status = FixStatus::NoFix;
  return f;
}

/** @brief True when the fix carries a valid (non-no-fix) status. */
[[nodiscard]] inline constexpr bool
fix_has_valid_status (const Fix &f) noexcept {
  return f._M_status != FixStatus::NoFix;
}

/** @brief True when lat/lon are within legal ranges. */
[[nodiscard]] inline constexpr bool
fix_coordinates_valid (const Fix &f) noexcept {
  return is_valid_lat (f._M_lat) && is_valid_lon (f._M_lon);
}

/** @brief True when the fix is usable (valid status and coordinates). */
[[nodiscard]] inline constexpr bool
fix_is_usable (const Fix &f) noexcept {
  return fix_has_valid_status (f) && fix_coordinates_valid (f);
}

// ---------------------------------------------------------------------------
// GNSS status / error
// ---------------------------------------------------------------------------

/** @brief GNSS receiver operational status. */
enum class GnssReceiverStatus : uint8_t {
  Off = 0,          /**< Receiver powered off. */
  Starting = 1,     /**< Receiver booting / cold start. */
  Searching = 2,    /**< Searching for satellites. */
  Tracking = 3,     /**< Tracking satellites, no fix yet. */
  Fix = 4,          /**< Fix available. */
  Error = 5,        /**< Receiver in an error state. */
};

/** @brief Return a short name for a receiver status. */
[[nodiscard]] inline const char *
gnss_receiver_status_name (GnssReceiverStatus s) noexcept {
  switch (s) {
  case GnssReceiverStatus::Off:
    return "off";
  case GnssReceiverStatus::Starting:
    return "starting";
  case GnssReceiverStatus::Searching:
    return "searching";
  case GnssReceiverStatus::Tracking:
    return "tracking";
  case GnssReceiverStatus::Fix:
    return "fix";
  case GnssReceiverStatus::Error:
    return "error";
  default:
    return "unknown";
  }
}

/** @} */
} // namespace zzs::kernel::device::driver::gnss
