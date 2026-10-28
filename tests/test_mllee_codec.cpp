/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the “Software”), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#ifndef ZZS_MLLEE_HOST
#define ZZS_MLLEE_HOST
#endif

#include "mllee/mllee.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <random>
#include <vector>

namespace
{
struct Decoded
{
    std::vector<uint8_t> _M_bytes;
    zzs::mllee::Status _M_begin = zzs::mllee::Status::OK;
    zzs::mllee::Status _M_status = zzs::mllee::Status::OK;
};

Decoded
decode (const std::vector<uint8_t> &stream, const std::vector<uint8_t> &dictionary = {})
{
    Decoded result;
    zzs::mllee::PacklowState state{};
    result._M_begin = zzs::mllee::_S_pk_begin (state, stream.data (),
                                               static_cast<uint32_t> (stream.size ()), 0,
                                               static_cast<uint32_t> (stream.size ()),
                                               zzs::mllee::Codec::PACKLOW);
    if (result._M_begin != zzs::mllee::Status::OK)
        {
            result._M_status = result._M_begin;
            return result;
        }
    const uint8_t *const dict = dictionary.empty () ? nullptr : dictionary.data ();
    for (;;)
        {
            const std::int32_t byte =
                zzs::mllee::_S_pk_next (state, stream.data (), dict,
                                        static_cast<uint16_t> (dictionary.size ()));
            if (byte < 0)
                break;
            result._M_bytes.push_back (static_cast<uint8_t> (byte));
        }
    result._M_status = zzs::mllee::_S_pk_status (state);
    return result;
}

std::vector<uint8_t>
encode (const std::vector<uint8_t> &raw,
        const zzs::mllee::PacklowOptions &options = zzs::mllee::PacklowOptions{})
{
    std::vector<uint8_t> output (raw.size () * 4 + 1024, 0);
    const uint8_t dummy = 0;
    const uint8_t *const source = raw.empty () ? &dummy : raw.data ();
    const uint32_t size = zzs::mllee::PacklowEncoder::encode (
        source, static_cast<uint32_t> (raw.size ()), options, output.data (),
        static_cast<uint32_t> (output.size ()));
    output.resize (size);
    return output;
}

std::vector<uint8_t>
random_bytes (const std::size_t count, const uint32_t seed)
{
    std::mt19937 generator (seed);
    std::vector<uint8_t> bytes (count);
    for (uint8_t &byte : bytes)
        byte = static_cast<uint8_t> (generator () & 0xFFu);
    return bytes;
}
} // namespace

TEST_CASE ("packlow round-trips random data", "[mllee][codec]")
{
    for (uint32_t seed = 1; seed <= 8; ++seed)
        {
            const std::vector<uint8_t> raw = random_bytes (257 + seed * 17, seed);
            const std::vector<uint8_t> stream = encode (raw);
            REQUIRE (stream.size () >= zzs::mllee::_S_pk_header_bytes);
            const Decoded decoded = decode (stream);
            REQUIRE (decoded._M_begin == zzs::mllee::Status::OK);
            REQUIRE (decoded._M_status == zzs::mllee::Status::OK);
            REQUIRE (decoded._M_bytes == raw);
        }
}

TEST_CASE ("packlow round-trips empty input", "[mllee][codec]")
{
    const std::vector<uint8_t> raw;
    const std::vector<uint8_t> stream = encode (raw);
    REQUIRE (stream.size () == zzs::mllee::_S_pk_header_bytes + 5);
    const Decoded decoded = decode (stream);
    REQUIRE (decoded._M_begin == zzs::mllee::Status::OK);
    REQUIRE (decoded._M_status == zzs::mllee::Status::OK);
    REQUIRE (decoded._M_bytes.empty ());
}

TEST_CASE ("packlow compresses zero runs", "[mllee][codec]")
{
    const std::vector<uint8_t> raw (1000, 0);
    zzs::mllee::PacklowStats stats;
    zzs::mllee::PacklowOptions options;
    options._M_stats = &stats;
    const std::vector<uint8_t> stream = encode (raw, options);
    REQUIRE (stream.size () < raw.size () / 8);
    REQUIRE (stats._M_copy >= 1);
    const Decoded decoded = decode (stream);
    REQUIRE (decoded._M_status == zzs::mllee::Status::OK);
    REQUIRE (decoded._M_bytes == raw);
}

TEST_CASE ("packlow delta path triggers on near-identical blocks", "[mllee][codec]")
{
    std::vector<uint8_t> raw (512);
    for (std::size_t block = 0; block < 16; ++block)
        for (std::size_t k = 0; k < 32; ++k)
            raw[block * 32 + k] =
                static_cast<uint8_t> (block + k + (k == 15 ? 1 : 0));

    zzs::mllee::PacklowStats stats;
    zzs::mllee::PacklowOptions options;
    options._M_stats = &stats;
    const std::vector<uint8_t> stream = encode (raw, options);
    REQUIRE (stats._M_delta >= 1);
    REQUIRE (stream.size () < raw.size ());

    const Decoded decoded = decode (stream);
    REQUIRE (decoded._M_status == zzs::mllee::Status::OK);
    REQUIRE (decoded._M_bytes == raw);
}

TEST_CASE ("packlow delta can be disabled", "[mllee][codec]")
{
    std::vector<uint8_t> raw (512);
    for (std::size_t block = 0; block < 16; ++block)
        for (std::size_t k = 0; k < 32; ++k)
            raw[block * 32 + k] =
                static_cast<uint8_t> (block + k + (k == 15 ? 1 : 0));

    zzs::mllee::PacklowStats stats;
    zzs::mllee::PacklowOptions options;
    options._M_delta = false;
    options._M_stats = &stats;
    const std::vector<uint8_t> stream = encode (raw, options);
    REQUIRE (stats._M_delta == 0);
    const Decoded decoded = decode (stream);
    REQUIRE (decoded._M_status == zzs::mllee::Status::OK);
    REQUIRE (decoded._M_bytes == raw);
}

TEST_CASE ("packlow dictionary copies use flash blocks", "[mllee][codec]")
{
    std::vector<uint8_t> dictionary = random_bytes (96, 1234);
    std::vector<uint8_t> raw;
    for (int repeat = 0; repeat < 6; ++repeat)
        for (std::size_t block = 0; block < dictionary.size (); block += 32)
            raw.insert (raw.end (), dictionary.begin () + static_cast<std::ptrdiff_t> (block),
                        dictionary.begin () + static_cast<std::ptrdiff_t> (block + 32));

    zzs::mllee::PacklowStats stats;
    zzs::mllee::PacklowOptions options;
    options._M_dict = dictionary.data ();
    options._M_dict_len = static_cast<uint16_t> (dictionary.size ());
    options._M_stats = &stats;
    const std::vector<uint8_t> stream = encode (raw, options);
    REQUIRE (stats._M_dict >= 1);
    REQUIRE (stream.size () < raw.size ());

    const Decoded decoded = decode (stream, dictionary);
    REQUIRE (decoded._M_status == zzs::mllee::Status::OK);
    REQUIRE (decoded._M_bytes == raw);
}

TEST_CASE ("packlow rejects bad header fields", "[mllee][codec]")
{
    const std::vector<uint8_t> stream = encode (random_bytes (64, 7));
    REQUIRE (!stream.empty ());

    std::vector<uint8_t> corrupted = stream;
    corrupted[0] = 'X';
    REQUIRE (decode (corrupted)._M_begin == zzs::mllee::Status::BAD_MAGIC);

    corrupted = stream;
    corrupted[4] = static_cast<uint8_t> (zzs::mllee::_S_pk_version + 1u);
    REQUIRE (decode (corrupted)._M_begin == zzs::mllee::Status::BAD_VERSION);

    corrupted = stream;
    REQUIRE (decode (std::vector<uint8_t> (corrupted.begin (), corrupted.begin () + 11))._M_begin
             == zzs::mllee::Status::TRUNCATED);
}

TEST_CASE ("packlow reports CRC corruption", "[mllee][codec]")
{
    const std::vector<uint8_t> raw = random_bytes (200, 99);
    std::vector<uint8_t> stream = encode (raw);
    stream[10] ^= 0x01u;
    const Decoded decoded = decode (stream);
    REQUIRE (decoded._M_status == zzs::mllee::Status::CORRUPT_STREAM);
    REQUIRE (decoded._M_bytes == raw);
}

TEST_CASE ("packlow survives truncation at every byte", "[mllee][codec]")
{
    const std::vector<uint8_t> raw = random_bytes (120, 42);
    const std::vector<uint8_t> stream = encode (raw);
    for (std::size_t cut = 0; cut < stream.size (); ++cut)
        {
            const std::vector<uint8_t> truncated (stream.begin (),
                                                       stream.begin () + static_cast<std::ptrdiff_t> (cut));
            const Decoded decoded = decode (truncated);
            if (decoded._M_begin == zzs::mllee::Status::OK
                && decoded._M_status == zzs::mllee::Status::OK)
                REQUIRE (decoded._M_bytes == raw);
            else
                REQUIRE (decoded._M_status != zzs::mllee::Status::OK);
        }
}

TEST_CASE ("packlow detects payload corruption or changes output", "[mllee][codec]")
{
    const std::vector<uint8_t> raw = random_bytes (300, 2026);
    std::vector<uint8_t> stream = encode (raw);
    REQUIRE (stream.size () > zzs::mllee::_S_pk_header_bytes + 8);
    stream[zzs::mllee::_S_pk_header_bytes + 6] ^= 0x40u;
    const Decoded decoded = decode (stream);
    REQUIRE ((decoded._M_status != zzs::mllee::Status::OK || decoded._M_bytes != raw));
}

TEST_CASE ("packlow encoder rejects tiny output buffers", "[mllee][codec]")
{
    const std::vector<uint8_t> raw = random_bytes (128, 5);
    uint8_t output[8] = {};
    REQUIRE (zzs::mllee::PacklowEncoder::encode (
                 raw.data (), static_cast<uint32_t> (raw.size ()),
                 zzs::mllee::PacklowOptions{}, output, sizeof (output))
             == 0);
}

namespace
{
using zzs::mllee::Codec;
using zzs::mllee::Dtype;
using zzs::mllee::PacklowState;
using zzs::mllee::Status;

std::vector<uint8_t>
pack_rows (const Dtype dtype, const std::vector<std::int8_t> &codes)
{
    std::vector<uint8_t> bytes (zzs::mllee::_S_row_bytes (
                                    dtype, static_cast<uint32_t> (codes.size ())),
                                0);
    zzs::mllee::_S_pack_codes (bytes.data (), codes.data (),
                               static_cast<uint32_t> (codes.size ()),
                               zzs::mllee::_S_code_bits (dtype));
    return bytes;
}

std::vector<std::int8_t>
make_codes (const uint32_t count, const uint32_t seed)
{
    const auto bits = 3u;
    std::mt19937 generator (seed);
    std::vector<std::int8_t> codes (count);
    for (std::int8_t &code : codes)
        code = static_cast<std::int8_t> (generator () & ((1u << bits) - 1u));
    return codes;
}
} // namespace

TEST_CASE ("reader keeps int8 byte semantics", "[mllee][codec]")
{
    const std::vector<uint8_t> bytes = { 0, 1, 127, 128, 255 };
    PacklowState state{};
    REQUIRE (zzs::mllee::_S_pk_begin (state, bytes.data (),
                                      static_cast<uint32_t> (bytes.size ()), 0,
                                      static_cast<uint32_t> (bytes.size ()), Codec::RAW)
             == Status::OK);
    zzs::mllee::_S_rd_begin (state, nullptr, Dtype::I8);
    const std::int32_t expected[5] = { 0, 1, 127, -128, -1 };
    for (const std::int32_t value : expected)
        REQUIRE (zzs::mllee::_S_rd_code (state, bytes.data (), nullptr, 0) == value);
}

TEST_CASE ("reservoir reads unaligned int2 and int3 groups", "[mllee][codec]")
{
    SECTION ("int2, groups of 6/6/5 codes")
    {
        const std::vector<std::int8_t> codes = make_codes (17, 11);
        const std::vector<uint8_t> bytes = pack_rows (Dtype::I2, codes);
        PacklowState state{};
        REQUIRE (zzs::mllee::_S_pk_begin (state, bytes.data (),
                                          static_cast<uint32_t> (bytes.size ()), 0,
                                          static_cast<uint32_t> (bytes.size ()), Codec::RAW)
                 == Status::OK);
        zzs::mllee::_S_rd_begin (state, nullptr, Dtype::I2);
        const uint32_t groups[] = { 6u, 6u, 5u };
        uint32_t index = 0;
        for (const uint32_t group : groups)
            for (uint32_t k = 0; k < group; ++k, ++index)
                REQUIRE (zzs::mllee::_S_rd_code (state, bytes.data (), nullptr, 0)
                         == zzs::mllee::_S_from_int2 (
                             static_cast<uint8_t> (codes[index])));
        REQUIRE (zzs::mllee::_S_pk_status (state) == Status::OK);
    }

    SECTION ("int3, groups of 5/5/3 codes")
    {
        const std::vector<std::int8_t> codes = make_codes (13, 23);
        const std::vector<uint8_t> bytes = pack_rows (Dtype::I3, codes);
        PacklowState state{};
        REQUIRE (zzs::mllee::_S_pk_begin (state, bytes.data (),
                                          static_cast<uint32_t> (bytes.size ()), 0,
                                          static_cast<uint32_t> (bytes.size ()), Codec::RAW)
                 == Status::OK);
        zzs::mllee::_S_rd_begin (state, nullptr, Dtype::I3);
        const uint32_t groups[] = { 5u, 5u, 3u };
        uint32_t index = 0;
        for (const uint32_t group : groups)
            for (uint32_t k = 0; k < group; ++k, ++index)
                REQUIRE (zzs::mllee::_S_rd_code (state, bytes.data (), nullptr, 0)
                         == zzs::mllee::_S_from_int3 (
                             static_cast<uint8_t> (codes[index])));
        REQUIRE (zzs::mllee::_S_pk_status (state) == Status::OK);
    }
}

TEST_CASE ("row reader sign-extends int2/int3/int4 and uses the float LUT",
           "[mllee][codec]")
{
    SECTION ("int2 rows of 12 codes")
    {
        const std::vector<std::int8_t> codes = make_codes (24, 31);
        const std::vector<uint8_t> bytes = pack_rows (Dtype::I2, codes);
        const uint32_t row_bytes = zzs::mllee::_S_row_bytes (Dtype::I2, 12);
        PacklowState state{};
        REQUIRE (zzs::mllee::_S_pk_begin (state, bytes.data (),
                                          static_cast<uint32_t> (bytes.size ()), 0,
                                          static_cast<uint32_t> (bytes.size ()), Codec::RAW)
                 == Status::OK);
        zzs::mllee::_S_rd_begin (state, nullptr, Dtype::I2);
        for (uint32_t row = 0; row < 2; ++row)
            {
                REQUIRE (zzs::mllee::_S_rd_row (state, bytes.data (), nullptr, 0, row_bytes)
                         == Status::OK);
                for (uint32_t i = 0; i < 12; ++i)
                    REQUIRE (zzs::mllee::_S_rd_row_code (state, bytes.data (), i, row_bytes)
                             == zzs::mllee::_S_from_int2 (static_cast<uint8_t> (
                                 codes[row * 12u + i])));
            }
    }

    SECTION ("int3 rows of 16 codes, including straddling codes")
    {
        const std::vector<std::int8_t> codes = make_codes (32, 37);
        const std::vector<uint8_t> bytes = pack_rows (Dtype::I3, codes);
        const uint32_t row_bytes = zzs::mllee::_S_row_bytes (Dtype::I3, 16);
        PacklowState state{};
        REQUIRE (zzs::mllee::_S_pk_begin (state, bytes.data (),
                                          static_cast<uint32_t> (bytes.size ()), 0,
                                          static_cast<uint32_t> (bytes.size ()), Codec::RAW)
                 == Status::OK);
        zzs::mllee::_S_rd_begin (state, nullptr, Dtype::I3);
        for (uint32_t row = 0; row < 2; ++row)
            {
                REQUIRE (zzs::mllee::_S_rd_row (state, bytes.data (), nullptr, 0, row_bytes)
                         == Status::OK);
                for (uint32_t i = 0; i < 16; ++i)
                    REQUIRE (zzs::mllee::_S_rd_row_code (state, bytes.data (), i, row_bytes)
                             == zzs::mllee::_S_from_int3 (static_cast<uint8_t> (
                                 codes[row * 16u + i])));
            }
    }

    SECTION ("int4 rows of 10 codes")
    {
        const std::vector<std::int8_t> codes = make_codes (20, 41);
        const std::vector<uint8_t> bytes = pack_rows (Dtype::I4, codes);
        const uint32_t row_bytes = zzs::mllee::_S_row_bytes (Dtype::I4, 10);
        PacklowState state{};
        REQUIRE (zzs::mllee::_S_pk_begin (state, bytes.data (),
                                          static_cast<uint32_t> (bytes.size ()), 0,
                                          static_cast<uint32_t> (bytes.size ()), Codec::RAW)
                 == Status::OK);
        zzs::mllee::_S_rd_begin (state, nullptr, Dtype::I4);
        for (uint32_t row = 0; row < 2; ++row)
            {
                REQUIRE (zzs::mllee::_S_rd_row (state, bytes.data (), nullptr, 0, row_bytes)
                         == Status::OK);
                for (uint32_t i = 0; i < 10; ++i)
                    REQUIRE (zzs::mllee::_S_rd_row_code (state, bytes.data (), i, row_bytes)
                             == zzs::mllee::_S_from_int4 (static_cast<uint8_t> (
                                 codes[row * 10u + i])));
            }
    }

    SECTION ("float3 row decodes through the 8-entry LUT")
    {
        const uint8_t code_values[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
        std::vector<std::int8_t> codes (code_values, code_values + 8);
        codes.insert (codes.end (), codes.begin (), codes.end ());
        const std::vector<uint8_t> bytes = pack_rows (Dtype::F3E1M1, codes);
        const uint32_t row_bytes = zzs::mllee::_S_row_bytes (Dtype::F3E1M1, 8);
        std::int8_t lut[8] = {};
        zzs::mllee::build_weight_lut_e1m1 (1.5f, lut);
        PacklowState state{};
        REQUIRE (zzs::mllee::_S_pk_begin (state, bytes.data (),
                                          static_cast<uint32_t> (bytes.size ()), 0,
                                          static_cast<uint32_t> (bytes.size ()), Codec::RAW)
                 == Status::OK);
        zzs::mllee::_S_rd_begin (state, lut, Dtype::F3E1M1);
        for (uint32_t row = 0; row < 2; ++row)
            {
                REQUIRE (zzs::mllee::_S_rd_row (state, bytes.data (), nullptr, 0, row_bytes)
                         == Status::OK);
                for (uint32_t i = 0; i < 8; ++i)
                    REQUIRE (zzs::mllee::_S_rd_row_code (state, bytes.data (), i, row_bytes)
                             == lut[codes[row * 8u + i] & 0x07]);
            }
    }
}

TEST_CASE ("packlow-compressed int3 stream reads through the reservoir", "[mllee][codec]")
{
    const std::vector<std::int8_t> codes = make_codes (29, 53);
    const std::vector<uint8_t> raw = pack_rows (Dtype::I3, codes);
    const std::vector<uint8_t> stream = encode (raw);
    PacklowState state{};
    REQUIRE (zzs::mllee::_S_pk_begin (state, stream.data (),
                                      static_cast<uint32_t> (stream.size ()), 0,
                                      static_cast<uint32_t> (stream.size ()), Codec::PACKLOW)
             == Status::OK);
    zzs::mllee::_S_rd_begin (state, nullptr, Dtype::I3);
    for (const std::int8_t code : codes)
        REQUIRE (zzs::mllee::_S_rd_code (state, stream.data (), nullptr, 0)
                 == zzs::mllee::_S_from_int3 (static_cast<uint8_t> (code)));
    REQUIRE (zzs::mllee::_S_pk_status (state) == Status::OK);
}

TEST_CASE ("reservoir reports truncation mid-code", "[mllee][codec]")
{
    const std::vector<uint8_t> bytes = { 0x88u }; // eight I3 bits
    PacklowState state{};
    REQUIRE (zzs::mllee::_S_pk_begin (state, bytes.data (),
                                      static_cast<uint32_t> (bytes.size ()), 0,
                                      static_cast<uint32_t> (bytes.size ()), Codec::RAW)
             == Status::OK);
    zzs::mllee::_S_rd_begin (state, nullptr, Dtype::I3);
    REQUIRE (zzs::mllee::_S_rd_code (state, bytes.data (), nullptr, 0) == 0);
    REQUIRE (zzs::mllee::_S_rd_code (state, bytes.data (), nullptr, 0) == 1);
    // The third code needs a ninth bit that the stream does not have.
    REQUIRE (zzs::mllee::_S_rd_code (state, bytes.data (), nullptr, 0)
             == zzs::mllee::_S_pk_error);
    REQUIRE (zzs::mllee::_S_pk_status (state) == Status::TRUNCATED);
}
