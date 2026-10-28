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

#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>

namespace
{
using zzs::mllee::Codec;
using zzs::mllee::PacklowEncoder;
using zzs::mllee::PacklowOptions;
using zzs::mllee::PacklowState;
using zzs::mllee::Status;

constexpr uint32_t _S_size = 4096;
constexpr uint32_t _S_iterations = 200;

struct Pattern
{
    const char *_M_name;
    uint8_t (*_M_fill) (uint32_t);
};

uint8_t
random_byte (const uint32_t index)
{
    uint32_t state = index * 1664525u + 1013904223u;
    state ^= state >> 13;
    return static_cast<uint8_t> (state & 0xFFu);
}

uint8_t
constant_byte (const uint32_t)
{
    return 7;
}

uint8_t
ramp_byte (const uint32_t index)
{
    return static_cast<uint8_t> (index & 0xFFu);
}

/** @brief int4-like weights with slowly varying blocks (delta friendly). */
uint8_t
weights_byte (const uint32_t index)
{
    const uint32_t block = index / 32u;
    return static_cast<uint8_t> ((block + (index % 32u) + (index % 15u == 0 ? 1u : 0u)) & 0xFFu);
}

bool
decode (const std::vector<uint8_t> &stream, const uint32_t raw_size,
        std::vector<uint8_t> &out)
{
    PacklowState state{};
    if (zzs::mllee::_S_pk_begin (state, stream.data (), static_cast<uint32_t> (stream.size ()),
                                 0, static_cast<uint32_t> (stream.size ()), Codec::PACKLOW)
        != Status::OK)
        return false;
    out.resize (raw_size);
    for (uint32_t i = 0; i < raw_size; ++i)
        {
            const std::int32_t byte = zzs::mllee::_S_pk_next (state, stream.data (), nullptr, 0);
            if (byte < 0)
                return false;
            out[i] = static_cast<uint8_t> (byte);
        }
    return zzs::mllee::_S_pk_status (state) == Status::OK;
}

template <typename Clock>
double
milliseconds (const typename Clock::time_point start)
{
    const auto elapsed = Clock::now () - start;
    return std::chrono::duration<double, std::milli> (elapsed).count ();
}
} // namespace

int
main ()
{
    const Pattern patterns[] = {
        { "random", random_byte },
        { "constant", constant_byte },
        { "ramp", ramp_byte },
        { "weights", weights_byte },
    };

    std::printf ("| pattern | raw B | packed B | ratio | encode MB/s | decode MB/s |\n");
    std::printf ("|---|---:|---:|---:|---:|---:|\n");

    PacklowOptions options;
    options._M_delta = true;
    options._M_dict_copy = false;

    bool ok = true;
    for (const Pattern &pattern : patterns)
        {
            std::vector<uint8_t> raw (_S_size);
            for (uint32_t i = 0; i < _S_size; ++i)
                raw[i] = pattern._M_fill (i);

            std::vector<uint8_t> encoded (_S_size * 2u + 512u);
            const auto encode_start = std::chrono::steady_clock::now ();
            uint32_t packed = 0;
            for (uint32_t iteration = 0; iteration < _S_iterations; ++iteration)
                packed = PacklowEncoder::encode (raw.data (), _S_size, options, encoded.data (),
                                                 static_cast<uint32_t> (encoded.size ()));
            const double encode_ms = milliseconds<std::chrono::steady_clock> (encode_start);

            std::vector<uint8_t> decoded;
            const auto decode_start = std::chrono::steady_clock::now ();
            for (uint32_t iteration = 0; iteration < _S_iterations; ++iteration)
                {
                    if (!decode (encoded, _S_size, decoded))
                        {
                            ok = false;
                            break;
                        }
                }
            const double decode_ms = milliseconds<std::chrono::steady_clock> (decode_start);

            if (packed == 0 || decoded.size () != raw.size ()
                || std::memcmp (decoded.data (), raw.data (), raw.size ()) != 0)
                {
                    ok = false;
                    std::printf ("| %s | FAILED round-trip |\n", pattern._M_name);
                    continue;
                }

            const double encode_mb = (static_cast<double> (_S_size) * _S_iterations / 1048576.0)
                                     / (encode_ms / 1000.0);
            const double decode_mb = (static_cast<double> (_S_size) * _S_iterations / 1048576.0)
                                     / (decode_ms / 1000.0);
            std::printf ("| %s | %u | %u | %.3f | %.1f | %.1f |\n", pattern._M_name,
                         static_cast<unsigned int> (_S_size), static_cast<unsigned int> (packed),
                         static_cast<double> (packed) / static_cast<double> (_S_size), encode_mb,
                         decode_mb);
        }

    return ok ? 0 : 1;
}
