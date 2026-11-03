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

#include "kernel/device-driver/gnss/gnss_types.hpp"
#include "kernel/device-driver/gnss/nmea/nmea_checksum.hpp"
#include "kernel/kernel_status.hpp"

#include <cstdint>
#include <cstring>

/**
 * @brief NMEA 0183 sentence parser.
 *
 * Receives raw receiver bytes one at a time (`feed`), assembles a single
 * sentence into a fixed buffer, validates the `*CC` checksum, splits on
 * commas, and dispatches to sentence-specific handlers ($GPGGA, $GPRMC,
 * $GPGSA, $GPGSV, $GPVTG).  All state is a fixed per-instance sentence
 * buffer; no heap, no float, no exceptions.
 *
 * Supported sentences:
 *   GGA — 3D fix, latitude/longitude, altitude, satellites used, HDOP
 *   RMC — recommended minimum, time, lat/lon, speed over ground, course, mode
 *   GSA — DOP values, fix mode, PRNs of satellites used
 *   GSV — satellites in view (PRN, elevation, azimuth, SNR)
 *   VTG — course over ground and ground speed
 * @{
 */
namespace zzs::kernel::device::driver::gnss::nmea {

/** @brief Maximum NMEA sentence length including '$...**CC\r\n'. */
inline constexpr uint16_t NmeaMaxSentenceBytes = 100u;

/** @brief Maximum comma-separated fields in a sentence. */
inline constexpr uint8_t NmeaMaxFields = 16u;

/** @brief NMEA sentence type. */
enum class SentenceType : uint8_t {
    Unknown = 0,
    Gga = 1,
    Rmc = 2,
    Gsa = 3,
    Gsv = 4,
    Vtg = 5,
};

/** @brief Fix mode indicator (from RMC/GSA 'mode' field). */
enum class FixMode : uint8_t {
    Autonomous = 1,   /**< Autonomous (no differential). */
    Differential = 2, /**< Differential (DGPS). */
    Estimated = 3,    /**< Estimated / dead reckoning. */
    Invalid = 6,      /**< Data not valid. */
};

/** @brief Parsed GGA sentence fields. */
struct Gga {
    uint32_t _M_time_hhmmss = 0u;   /**< UTC time hhmmss. */
    LatLon _M_lat = 0;              /**< Latitude, degrees * 1e7. */
    char _M_lat_hemi = 0;           /**< 'N' or 'S'. */
    LatLon _M_lon = 0;              /**< Longitude, degrees * 1e7. */
    char _M_lon_hemi = 0;           /**< 'E' or 'W'. */
    uint8_t _M_fix_quality = 0u;    /**< 0=invalid,1=GPS,2=DGPS. */
    uint8_t _M_num_satellites = 0u; /**< Satellites used. */
    int16_t _M_hdop_tenths = 0;     /**< HDOP, tenths. */
    int32_t _M_alt_mm = 0;          /**< Antenna altitude above ellipsoid, mm. */
    int32_t _M_geo_sep_mm = 0;      /**< Geoidal separation, mm. */
};

/** @brief Parsed RMC sentence fields. */
struct Rmc {
    uint32_t _M_time_hhmmss = 0u; /**< UTC time hhmmss. */
    bool _M_valid = false;        /**< 'A' = valid, 'V' = warning. */
    LatLon _M_lat = 0;
    char _M_lat_hemi = 0;
    LatLon _M_lon = 0;
    char _M_lon_hemi = 0;
    int32_t _M_speed_knots_tenths = 0;  /**< Speed over ground, knots*tenths. */
    int32_t _M_course_deg_tenths = 0;   /**< Course over ground, tenths deg. */
    uint32_t _M_date_ddmmyy = 0u;       /**< Date ddmmyy. */
    FixMode _M_mode = FixMode::Invalid; /**< Positioning mode. */
};

/** @brief Parsed GSA sentence fields. */
struct Gsa {
    FixMode _M_mode = FixMode::Invalid; /**< Fix mode: 1=no fix,2=2D,3=3D. */
    int16_t _M_pdop_tenths = 0;         /**< PDOP, tenths. */
    int16_t _M_hdop_tenths = 0;         /**< HDOP, tenths. */
    int16_t _M_vdop_tenths = 0;         /**< VDOP, tenths. */
    uint8_t _M_sat_prns[GnssMaxSatellites] = {};
    uint8_t _M_num_satellites = 0u;
};

/** @brief One satellite in view (from GSV). */
struct GsvSat {
    uint8_t _M_prn = GnssInvalidPrn;
    uint8_t _M_elevation_deg = 0u;
    uint16_t _M_azimuth_deg_tenths = 0u;
    uint8_t _M_snr_dbhz = 0u;
};

/** @brief Parsed GSV sentence fields (satellites in view). */
struct Gsv {
    uint8_t _M_num_messages = 0u;
    uint8_t _M_message_number = 0u;
    uint8_t _M_total_satellites = 0u;
    GsvSat _M_satellites[4u] = {}; /**< Up to 4 satellites per GSV message. */
    uint8_t _M_num_satellites = 0u;
};

/** @brief Parsed VTG sentence fields. */
struct Vtg {
    int32_t _M_course_true_deg_tenths = 0; /**< Course over ground, true, tenths. */
    int32_t _M_course_mag_deg_tenths = 0;  /**< Course over ground, magnetic. */
    int32_t _M_speed_knots_tenths = 0;     /**< Speed over ground, knots. */
    int32_t _M_speed_kmph_tenths = 0;      /**< Speed over ground, km/h. */
};

/** @brief Result of parsing one sentence: the type plus the decoded union. */
struct ParsedSentence {
    SentenceType _M_type = SentenceType::Unknown;
    uint8_t _M_talker_id[2] = {0, 0}; /**< e.g. "GP", "GN". */
    bool _M_checksum_ok = false;

    Gga _M_gga{};
    Rmc _M_rmc{};
    Gsa _M_gsa{};
    Gsv _M_gsv{};
    Vtg _M_vtg{};
};

/** @brief Callback signature for a fully parsed, checksum-valid sentence. */
using SentenceCallback = void (*) (const ParsedSentence &, void *context) noexcept;

/** @brief Incremental NMEA sentence parser. */
class Parser {
  public:
    /** @brief Construct with a fixed sentence buffer. */
    Parser () noexcept = default;

    Parser (const Parser &) = delete;
    Parser &
    operator= (const Parser &) = delete;

    /** @brief Feed one raw byte. Returns OK when a complete valid sentence was
     *         parsed and dispatched, STALE when more bytes are needed, or a
     *         protocol error (bad checksum, oversize, framing). */
    [[nodiscard]] zzs::kernel::Status
    feed (uint8_t b) noexcept;

    /** @brief Register a callback invoked for every valid parsed sentence. */
    void
    set_callback (SentenceCallback cb, void *context) noexcept {
        _M_cb = cb;
        _M_ctx = context;
    }

    /** @brief Return the last parsed sentence (valid only after feed() returns
     *         OK). */
    [[nodiscard]] const ParsedSentence &
    last_sentence () const noexcept {
        return _M_last;
    }

    /** @brief Number of sentences successfully parsed. */
    [[nodiscard]] uint32_t
    sentences_parsed () const noexcept {
        return _M_sentences_parsed;
    }

    /** @brief Number of checksum / framing errors. */
    [[nodiscard]] uint32_t
    errors () const noexcept {
        return _M_errors;
    }

  private:
    [[nodiscard]] zzs::kernel::Status
    _M_complete_sentence () noexcept;

    [[nodiscard]] static SentenceType
    _M_sentence_type (const char *id) noexcept;

    [[nodiscard]] static uint16_t
    _M_split_fields (char *buf, char *fields[NmeaMaxFields]) noexcept;

    [[nodiscard]] static bool
    _M_parse_gga (char *fields[], uint8_t nfields, Gga &out) noexcept;
    [[nodiscard]] static bool
    _M_parse_rmc (char *fields[], uint8_t nfields, Rmc &out) noexcept;
    [[nodiscard]] static bool
    _M_parse_gsa (char *fields[], uint8_t nfields, Gsa &out) noexcept;
    [[nodiscard]] static bool
    _M_parse_gsv (char *fields[], uint8_t nfields, Gsv &out) noexcept;
    [[nodiscard]] static bool
    _M_parse_vtg (char *fields[], uint8_t nfields, Vtg &out) noexcept;

    [[nodiscard]] static LatLon
    _M_lat_lon_from_nmea (const char *field, char hemi, bool is_lat) noexcept;
    [[nodiscard]] static FixMode
    _M_fix_mode_from_char (char c) noexcept;
    [[nodiscard]] static int32_t
    _M_parse_int_tenths (const char *field) noexcept;
    [[nodiscard]] static uint32_t
    _M_parse_uint (const char *field) noexcept;

    SentenceCallback _M_cb = nullptr;
    void *_M_ctx = nullptr;

    char _M_buf[NmeaMaxSentenceBytes] = {};
    uint16_t _M_len = 0u;
    bool _M_in_sentence = false;
    bool _M_after_star = false;
    uint8_t _M_ck_expected = 0u;
    uint8_t _M_ck_digit_count = 0u;

    ParsedSentence _M_last{};
    uint32_t _M_sentences_parsed = 0u;
    uint32_t _M_errors = 0u;
};

/** @} */
} // namespace zzs::kernel::device::driver::gnss::nmea
