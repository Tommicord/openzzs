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

#include "kernel/device-driver/gnss/nmea/nmea_parser.hpp"

namespace zzs::kernel::device::driver::gnss::nmea {

[[nodiscard]] zzs::kernel::Status
Parser::feed (uint8_t b) noexcept {
    char c = static_cast<char> (b);

    /* Wait for a sentence start. */
    if (!_M_in_sentence) {
        if (c == NmeaStartDollar || c == NmeaStartBang) {
            _M_in_sentence = true;
            _M_after_star = false;
            _M_ck_expected = 0u;
            _M_ck_digit_count = 0u;
            _M_len = 0u;
            _M_buf[0] = '\0';
        }
        return zzs::kernel::Status::STALE;
    }

    /* Checksum hex-digit phase. */
    if (_M_after_star) {
        uint8_t nibble = hex_nibble (c);
        if (nibble == 0xffu) {
            /* Malformed checksum: abort this sentence. */
            _M_in_sentence = false;
            _M_errors += 1u;
            return zzs::kernel::Status::BAD_CRC;
        }
        if (_M_ck_digit_count == 0u) {
            _M_ck_expected = static_cast<uint8_t> (nibble << 4);
            _M_ck_digit_count = 1u;
            return zzs::kernel::Status::STALE;
        }
        _M_ck_expected = static_cast<uint8_t> (_M_ck_expected | nibble);
        _M_ck_digit_count = 0u;
        _M_after_star = false;

        /* Sentence is complete; validate checksum and dispatch. */
        return _M_complete_sentence ();
    }

    /* Inside a sentence body. */
    if (c == NmeaChecksumSep) {
        /* Transition to checksum phase. */
        _M_after_star = true;
        _M_ck_digit_count = 0u;
        return zzs::kernel::Status::STALE;
    }

    if (c == NmeaEndCr || c == NmeaEndLf) {
        /* End-of-line without a checksum: framing error. */
        _M_in_sentence = false;
        _M_errors += 1u;
        return zzs::kernel::Status::BAD_LENGTH;
    }

    /* Accumulate body byte. */
    if (_M_len + 1u >= NmeaMaxSentenceBytes) {
        _M_in_sentence = false;
        _M_errors += 1u;
        return zzs::kernel::Status::BAD_LENGTH;
    }
    _M_buf[_M_len++] = c;
    _M_buf[_M_len] = '\0';
    _M_ck_expected = static_cast<uint8_t> (_M_ck_expected ^ static_cast<uint8_t> (c));
    return zzs::kernel::Status::STALE;
}

[[nodiscard]] zzs::kernel::Status
Parser::_M_complete_sentence () noexcept {
    _M_in_sentence = false;

    /* Validate the NMEA XOR checksum. */
    uint8_t ck_actual = nmea_checksum (_M_buf, _M_buf + _M_len);
    if (ck_actual != _M_ck_expected) {
        _M_errors += 1u;
        return zzs::kernel::Status::BAD_CRC;
    }

    /* Null-terminate and split on commas. */
    char *fields[NmeaMaxFields];
    for (uint8_t i = 0u; i < NmeaMaxFields; ++i)
        fields[i] = nullptr;
    uint16_t nfields = _M_split_fields (_M_buf, fields);
    if (nfields == 0u) {
        _M_errors += 1u;
        return zzs::kernel::Status::BAD_LENGTH;
    }

    /* The first field is "talkerID + sentenceID", e.g. "GPGGA". */
    const char *first = fields[0];
    if (first == nullptr || first[0] == '\0') {
        _M_errors += 1u;
        return zzs::kernel::Status::BAD_TYPE;
    }

    /* Talker ID = first two chars. */
    _M_last._M_talker_id[0] = static_cast<uint8_t> (first[0]);
    _M_last._M_talker_id[1] = static_cast<uint8_t> (first[1]);

    /* Sentence ID starts at char 2. */
    const char *sid = first + 2;
    SentenceType type = _M_sentence_type (sid);
    _M_last._M_type = type;
    _M_last._M_checksum_ok = true;

    bool ok = false;
    switch (type) {
    case SentenceType::Gga:
        ok = _M_parse_gga (fields, static_cast<uint8_t> (nfields), _M_last._M_gga);
        break;
    case SentenceType::Rmc:
        ok = _M_parse_rmc (fields, static_cast<uint8_t> (nfields), _M_last._M_rmc);
        break;
    case SentenceType::Gsa:
        ok = _M_parse_gsa (fields, static_cast<uint8_t> (nfields), _M_last._M_gsa);
        break;
    case SentenceType::Gsv:
        ok = _M_parse_gsv (fields, static_cast<uint8_t> (nfields), _M_last._M_gsv);
        break;
    case SentenceType::Vtg:
        ok = _M_parse_vtg (fields, static_cast<uint8_t> (nfields), _M_last._M_vtg);
        break;
    default:
        /* Unknown sentence type: still a valid NMEA sentence, just not one we
         * decode.  Return OK so the caller knows the stream is healthy. */
        break;
    }

    if (!ok && type != SentenceType::Unknown) {
        _M_errors += 1u;
        return zzs::kernel::Status::BAD_LENGTH;
    }

    _M_sentences_parsed += 1u;
    if (_M_cb != nullptr)
        _M_cb (_M_last, _M_ctx);
    return zzs::kernel::Status::OK;
}

[[nodiscard]] SentenceType
Parser::_M_sentence_type (const char *id) noexcept {
    if (id[0] == '\0')
        return SentenceType::Unknown;
    if (id[0] == 'G' && id[1] == 'G' && id[2] == 'A' && id[3] == '\0')
        return SentenceType::Gga;
    if (id[0] == 'R' && id[1] == 'M' && id[2] == 'C' && id[3] == '\0')
        return SentenceType::Rmc;
    if (id[0] == 'G' && id[1] == 'S' && id[2] == 'A' && id[3] == '\0')
        return SentenceType::Gsa;
    if (id[0] == 'G' && id[1] == 'S' && id[2] == 'V' && id[3] == '\0')
        return SentenceType::Gsv;
    if (id[0] == 'V' && id[1] == 'T' && id[2] == 'G' && id[3] == '\0')
        return SentenceType::Vtg;
    return SentenceType::Unknown;
}

[[nodiscard]] uint16_t
Parser::_M_split_fields (char *buf, char *fields[NmeaMaxFields]) noexcept {
    uint16_t n = 0u;
    fields[n++] = buf;
    char *p = buf;
    while (*p != '\0' && n < NmeaMaxFields) {
        if (*p == ',') {
            *p = '\0';
            fields[n++] = p + 1;
        }
        ++p;
    }
    return n;
}

// ---------------------------------------------------------------------------
// Numeric field parsers
// ---------------------------------------------------------------------------

[[nodiscard]] uint32_t
Parser::_M_parse_uint (const char *field) noexcept {
    if (field == nullptr || *field == '\0')
        return 0u;
    const char *p = field;
    uint32_t v = 0u;
    while (*p != '\0' && *p >= '0' && *p <= '9') {
        v = v * 10u + static_cast<uint32_t> (*p - '0');
        ++p;
    }
    return v;
}

[[nodiscard]] int32_t
Parser::_M_parse_int_tenths (const char *field) noexcept {
    if (field == nullptr || *field == '\0')
        return 0;
    const char *p = field;
    bool neg = false;
    if (*p == '-') {
        neg = true;
        ++p;
    }
    uint32_t int_part = 0u;
    while (*p != '\0' && *p >= '0' && *p <= '9') {
        int_part = int_part * 10u + static_cast<uint32_t> (*p - '0');
        ++p;
    }
    uint32_t frac = 0u;
    if (*p == '.') {
        ++p;
        uint8_t frac_digits = 0u;
        while (*p != '\0' && *p >= '0' && *p <= '9' && frac_digits < 1u) {
            frac = static_cast<uint32_t> (*p - '0');
            ++frac_digits;
            ++p;
        }
        /* Only one fractional digit (tenths) is kept. */
    }
    int32_t v = static_cast<int32_t> (int_part * 10u + frac);
    return neg ? -v : v;
}

[[nodiscard]] LatLon
Parser::_M_lat_lon_from_nmea (const char *field, char hemi, bool is_lat) noexcept {
    if (field == nullptr || *field == '\0')
        return 0;
    const char *p = field;

    /* Locate the decimal point to split degrees/minutes. */
    const char *dot = nullptr;
    for (const char *q = p; *q != '\0'; ++q) {
        if (*q == '.') {
            dot = q;
            break;
        }
    }

    uint32_t degrees = 0u;
    if (is_lat) {
        /* Latitude: DDMM.MMMM -> 2 deg digits. */
        if (static_cast<uint32_t> (dot ? (dot - p) : 0u) >= 2u) {
            degrees = static_cast<uint32_t> ((p[0] - '0') * 10 + (p[1] - '0'));
            p += 2;
        }
    } else {
        /* Longitude: DDDMM.MMMM -> 3 deg digits. */
        if (static_cast<uint32_t> (dot ? (dot - p) : 0u) >= 3u) {
            degrees = static_cast<uint32_t> ((p[0] - '0') * 100 + (p[1] - '0') * 10 + (p[2] - '0'));
            p += 3;
        }
    }

    /* Minutes: parse all digits, track fractional digits, scale to 1/10000 min. */
    uint32_t minutes_val = 0u;
    uint8_t frac_digits = 0u;
    bool saw_dot = false;
    for (const char *q = p; *q != '\0'; ++q) {
        if (*q == '.') {
            saw_dot = true;
            continue;
        }
        if (*q < '0' || *q > '9')
            break;
        minutes_val = minutes_val * 10u + static_cast<uint32_t> (*q - '0');
        if (saw_dot)
            ++frac_digits;
    }
    /* Scale minutes to 1/10000: minutes_val * 10^(4 - frac_digits). */
    uint32_t scale = 1u;
    for (uint8_t i = frac_digits; i < 4u; ++i)
        scale *= 10u;
    if (frac_digits > 4u) {
        /* Truncate extra fractional digits. */
        for (uint8_t i = 4u; i < frac_digits; ++i)
            minutes_val /= 10u;
        scale = 1u;
    }
    uint32_t minutes_scaled = minutes_val * scale; /* minutes * 10000 */

    /* degrees*1e7 + minutes_scaled*50/3 (round-half-up). */
    int64_t microdeg =
        static_cast<int64_t> (degrees) * 10000000ll + (static_cast<int64_t> (minutes_scaled) * 50ll + 1ll) / 3ll;

    if (hemi == 'S' || hemi == 's')
        microdeg = -microdeg;
    else if (hemi == 'W' || hemi == 'w')
        microdeg = -microdeg;

    /* Clamp to legal range. */
    if (microdeg > GnssLatMax)
        microdeg = GnssLatMax;
    else if (microdeg < -GnssLatMax)
        microdeg = -GnssLatMax;
    return static_cast<LatLon> (microdeg);
}

[[nodiscard]] FixMode
Parser::_M_fix_mode_from_char (char c) noexcept {
    switch (c) {
    case 'A':
        return FixMode::Autonomous;
    case 'D':
        return FixMode::Differential;
    case 'E':
        return FixMode::Estimated;
    case 'N':
        return FixMode::Invalid;
    default:
        return FixMode::Invalid;
    }
}

// ---------------------------------------------------------------------------
// Sentence parsers
// ---------------------------------------------------------------------------

[[nodiscard]] bool
Parser::_M_parse_gga (char *fields[], uint8_t nfields, Gga &out) noexcept {
    out = Gga{};
    if (nfields < 15u)
        return false;

    /* $GPGGA,time,lat,hemi,lon,hemi,quality,sats,hdop,alt,altunit,sep,sepunit,diffage,diffstation,checksum */
    out._M_time_hhmmss = _M_parse_uint (fields[1]);
    out._M_lat = _M_lat_lon_from_nmea (fields[2], fields[3] ? fields[3][0] : 0, true);
    out._M_lat_hemi = fields[3] ? fields[3][0] : 0;
    out._M_lon = _M_lat_lon_from_nmea (fields[4], fields[5] ? fields[5][0] : 0, false);
    out._M_lon_hemi = fields[5] ? fields[5][0] : 0;
    out._M_fix_quality = static_cast<uint8_t> (_M_parse_uint (fields[6]));
    out._M_num_satellites = static_cast<uint8_t> (_M_parse_uint (fields[7]));
    out._M_hdop_tenths = _M_parse_int_tenths (fields[8]);
    /* Altitude in meters; fields[9] may be decimal -> tenths -> *100 = mm. */
    int32_t alt_tenths = _M_parse_int_tenths (fields[9]);
    out._M_alt_mm = alt_tenths * 100;
    int32_t sep_tenths = _M_parse_int_tenths (fields[11]);
    out._M_geo_sep_mm = sep_tenths * 100;
    return true;
}

[[nodiscard]] bool
Parser::_M_parse_rmc (char *fields[], uint8_t nfields, Rmc &out) noexcept {
    out = Rmc{};
    if (nfields < 13u)
        return false;

    /* $GPRMC,time,status,lat,hemi,lon,hemi,speed,course,date,magvar,mvdir,mode,checksum */
    out._M_time_hhmmss = _M_parse_uint (fields[1]);
    out._M_valid = (fields[2] != nullptr && fields[2][0] == 'A');
    out._M_lat = _M_lat_lon_from_nmea (fields[3], fields[4] ? fields[4][0] : 0, true);
    out._M_lat_hemi = fields[4] ? fields[4][0] : 0;
    out._M_lon = _M_lat_lon_from_nmea (fields[5], fields[6] ? fields[6][0] : 0, false);
    out._M_lon_hemi = fields[6] ? fields[6][0] : 0;
    out._M_speed_knots_tenths = _M_parse_int_tenths (fields[7]);
    out._M_course_deg_tenths = _M_parse_int_tenths (fields[8]);
    out._M_date_ddmmyy = _M_parse_uint (fields[9]);
    out._M_mode = _M_fix_mode_from_char (fields[12] ? fields[12][0] : 'N');
    return true;
}

[[nodiscard]] bool
Parser::_M_parse_gsa (char *fields[], uint8_t nfields, Gsa &out) noexcept {
    out = Gsa{};
    if (nfields < 18u)
        return false;

    /* $GPGSA,mode,fixmode,prn,prn,...,prn,pdop,hdop,vdop,checksum */
    out._M_mode = _M_fix_mode_from_char (fields[1] ? fields[1][0] : 'N');
    /* Fix mode: 1=no fix,2=2D,3=3D (field[1] for some talkers) — already parsed. */
    uint8_t sat_idx = 0u;
    for (uint8_t i = 2u; i < nfields && sat_idx < GnssMaxSatellites; ++i) {
        if (fields[i] == nullptr || fields[i][0] == '\0')
            continue;
        uint32_t prn = _M_parse_uint (fields[i]);
        if (prn >= 1u && prn <= 255u)
            out._M_sat_prns[sat_idx++] = static_cast<uint8_t> (prn);
    }
    out._M_num_satellites = sat_idx;
    /* PDOP/HDOP/VDOP are the last three numeric fields. */
    int32_t pdop = _M_parse_int_tenths (fields[nfields - 3]);
    int32_t hdop = _M_parse_int_tenths (fields[nfields - 2]);
    int32_t vdop = _M_parse_int_tenths (fields[nfields - 1]);
    out._M_pdop_tenths = static_cast<int16_t> (pdop);
    out._M_hdop_tenths = static_cast<int16_t> (hdop);
    out._M_vdop_tenths = static_cast<int16_t> (vdop);
    return true;
}

[[nodiscard]] bool
Parser::_M_parse_gsv (char *fields[], uint8_t nfields, Gsv &out) noexcept {
    out = Gsv{};
    if (nfields < 5u)
        return false;

    /* $GPGSV,num_messages,msg_number,total_sats,prn,elev,azimuth,snr,...,checksum */
    out._M_num_messages = static_cast<uint8_t> (_M_parse_uint (fields[1]));
    out._M_message_number = static_cast<uint8_t> (_M_parse_uint (fields[2]));
    out._M_total_satellites = static_cast<uint8_t> (_M_parse_uint (fields[3]));

    uint8_t sat_idx = 0u;
    uint8_t i = 4u;
    while (i + 3u < nfields && sat_idx < 4u) {
        uint32_t prn = _M_parse_uint (fields[i]);
        uint32_t elev = _M_parse_uint (fields[i + 1]);
        uint32_t az = _M_parse_uint (fields[i + 2]);
        uint32_t snr = _M_parse_uint (fields[i + 3]);
        if (prn >= 1u && prn <= 255u) {
            out._M_satellites[sat_idx]._M_prn = static_cast<uint8_t> (prn);
            out._M_satellites[sat_idx]._M_elevation_deg = static_cast<uint8_t> (elev);
            out._M_satellites[sat_idx]._M_azimuth_deg_tenths = static_cast<uint16_t> (az);
            out._M_satellites[sat_idx]._M_snr_dbhz = static_cast<uint8_t> (snr);
            ++sat_idx;
        }
        i += 4u;
    }
    out._M_num_satellites = sat_idx;
    return true;
}

[[nodiscard]] bool
Parser::_M_parse_vtg (char *fields[], uint8_t nfields, Vtg &out) noexcept {
    out = Vtg{};
    if (nfields < 9u)
        return false;

    /* $GPVTG,course_true,T,course_mag,M,speed_knots,N,speed_kmph,K,mode,checksum */
    out._M_course_true_deg_tenths = _M_parse_int_tenths (fields[1]);
    out._M_course_mag_deg_tenths = _M_parse_int_tenths (fields[3]);
    out._M_speed_knots_tenths = _M_parse_int_tenths (fields[5]);
    out._M_speed_kmph_tenths = _M_parse_int_tenths (fields[7]);
    return true;
}

} // namespace zzs::kernel::device::driver::gnss::nmea
