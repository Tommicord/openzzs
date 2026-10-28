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

/**
 * @brief Reference ticks for the op kernels.
 *
 * The counters are the host's invariant time base, so the numbers below are
 * reference ticks per weight, comparable between builds on one machine but not
 * a Cortex-M4 cycle count.
 */
[[nodiscard]] inline uint64_t
_S_tick () noexcept
{
#if defined (__x86_64__) || defined (__i386__)
    return __builtin_ia32_rdtsc ();
#else
    return static_cast<uint64_t> (
        std::chrono::duration_cast<std::chrono::nanoseconds> (
            std::chrono::steady_clock::now ().time_since_epoch ())
            .count ());
#endif
}

constexpr uint32_t _S_ops_iterations = 300;

/** @brief Seeded byte with the small amplitude of a symmetric int8 weight tensor. */
int8_t
_S_weight_code (const uint32_t index)
{
    uint32_t state = index * 2246822519u + 3266489917u;
    state ^= state >> 13;
    state *= 2246822519u;
    state ^= state >> 16;
    return static_cast<int8_t> (state % 7u) - 3;
}

int8_t
_S_activation_code (const uint32_t index)
{
    uint32_t state = index * 1664525u + 1013904223u;
    state ^= state >> 15;
    return static_cast<int8_t> ((state >> 8) % 255u) - 127;
}

/**
 * @brief One measured kernel call: a workspace, its operand buffers and the
 *        per-element work the kernel does (weights, taps or values).
 */
struct OpsCase
{
    const char *_M_name;
    zzs::mllee::Workspace _M_ws{};
    std::vector<uint8_t> _M_model;
    std::vector<int8_t> _M_input;
    std::vector<int8_t> _M_output;
    uint64_t _M_elements = 0;
    Status (*_M_run) (OpsCase &) noexcept = nullptr;
};

/** @brief Fill the descriptor fields shared by every weight-stream case. */
void
_S_fill_weight_layer (zzs::mllee::Layer &layer, const zzs::mllee::Op op,
                      const uint32_t in_dim, const uint32_t out_dim) noexcept
{
    layer = {};
    layer._M_op = op;
    layer._M_in_dim = static_cast<uint16_t> (in_dim);
    layer._M_out_dim = static_cast<uint16_t> (out_dim);
    layer._M_dtype = zzs::mllee::Dtype::I8;
    layer._M_kernel = 1;
    layer._M_stride = 1;
    layer._M_requant_mult = 32768;
}

Status
_S_run_linear (OpsCase &c) noexcept
{
    return zzs::mllee::_S_linear (c._M_ws, c._M_model.data (), c._M_input.data (),
                                  c._M_output.data ());
}

Status
_S_run_conv1d (OpsCase &c) noexcept
{
    return zzs::mllee::_S_conv1d (c._M_ws, c._M_model.data (), c._M_input.data (),
                                  c._M_output.data ());
}

Status
_S_run_norm (OpsCase &c) noexcept
{
    return zzs::mllee::_S_norm (c._M_ws._M_layer, c._M_model.data (), c._M_input.data (),
                                c._M_output.data ());
}

Status
_S_run_softmax (OpsCase &c) noexcept
{
    return zzs::mllee::_S_softmax (c._M_ws._M_layer, c._M_input.data (),
                                   c._M_output.data ());
}

Status
_S_run_activation (OpsCase &c) noexcept
{
    return zzs::mllee::_S_activation (c._M_ws._M_layer, c._M_input.data (),
                                      c._M_output.data ());
}

Status
_S_run_attention (OpsCase &c) noexcept
{
    return zzs::mllee::_S_attention (c._M_ws._M_layer, c._M_input.data (),
                                     c._M_output.data ());
}

/** @brief Attach a weight stream to a case, raw or packlow-encoded. */
void
_S_attach_weights (OpsCase &c, const std::vector<int8_t> &codes,
                   const zzs::mllee::Codec codec)
{
    std::vector<uint8_t> raw;
    raw.reserve (codes.size ());
    for (const int8_t code : codes)
        raw.push_back (static_cast<uint8_t> (code));

    c._M_model.assign (zzs::mllee::_S_model_header_size, 0);
    const uint32_t offset = static_cast<uint32_t> (c._M_model.size ());
    if (codec == zzs::mllee::Codec::PACKLOW)
        {
            std::vector<uint8_t> encoded (raw.size () * 2u + 512u);
            zzs::mllee::PacklowOptions options;
            options._M_delta = true;
            options._M_dict_copy = false;
            const uint32_t packed = PacklowEncoder::encode (
                raw.data (), static_cast<uint32_t> (raw.size ()), options, encoded.data (),
                static_cast<uint32_t> (encoded.size ()));
            if (packed != 0)
                encoded.resize (packed);
            else
                {
                    c._M_model.insert (c._M_model.end (), raw.begin (), raw.end ());
                    c._M_ws._M_layer._M_codec = zzs::mllee::Codec::RAW;
                }
            c._M_model.insert (c._M_model.end (), encoded.begin (), encoded.end ());
        }
    else
        c._M_model.insert (c._M_model.end (), raw.begin (), raw.end ());

    c._M_ws._M_layer._M_weight_off = offset;
    c._M_ws._M_layer._M_weight_len =
        static_cast<uint32_t> (c._M_model.size ()) - offset;
    c._M_ws._M_layer._M_codec = codec;
    const zzs::mllee::Status status = zzs::mllee::_S_pk_begin (
        c._M_ws._M_pk, c._M_model.data (), static_cast<uint32_t> (c._M_model.size ()),
        c._M_ws._M_layer._M_weight_off, c._M_ws._M_layer._M_weight_len, codec);
    if (status != zzs::mllee::Status::OK)
        std::printf ("  setup failed: %s\n", zzs::mllee::status_name (status));
    zzs::mllee::_S_rd_begin (c._M_ws._M_pk, nullptr, c._M_ws._M_layer._M_dtype);
}

/** @brief Re-arm the codec reader without rebuilding the blob. */
void
_S_rewind (OpsCase &c)
{
    if (c._M_ws._M_layer._M_op != zzs::mllee::Op::LINEAR
        && c._M_ws._M_layer._M_op != zzs::mllee::Op::CONV1D)
        return;
    const zzs::mllee::Layer &layer = c._M_ws._M_layer;
    const zzs::mllee::Status status = zzs::mllee::_S_pk_begin (
        c._M_ws._M_pk, c._M_model.data (), static_cast<uint32_t> (c._M_model.size ()),
        layer._M_weight_off, layer._M_weight_len, layer._M_codec);
    if (status != zzs::mllee::Status::OK)
        std::printf ("  rewind failed: %s\n", zzs::mllee::status_name (status));
    zzs::mllee::_S_rd_begin (c._M_ws._M_pk, nullptr, layer._M_dtype);
}

/** @brief One LINEAR row block: in_dim x out_dim int8 weights. */
OpsCase
_S_make_linear (const char *name, const uint32_t in_dim, const uint32_t out_dim,
                const zzs::mllee::Codec codec)
{
    OpsCase c;
    c._M_name = name;
    c._M_run = _S_run_linear;
    _S_fill_weight_layer (c._M_ws._M_layer, zzs::mllee::Op::LINEAR, in_dim, out_dim);
    std::vector<int8_t> codes (static_cast<std::size_t> (in_dim) * out_dim);
    for (std::size_t i = 0; i < codes.size (); ++i)
        codes[i] = _S_weight_code (static_cast<uint32_t> (i));
    _S_attach_weights (c, codes, codec);
    c._M_input.resize (in_dim);
    for (uint32_t i = 0; i < in_dim; ++i)
        c._M_input[i] = _S_activation_code (i);
    c._M_output.resize (out_dim);
    c._M_elements = static_cast<uint64_t> (in_dim) * out_dim;
    return c;
}

/** @brief One depthwise CONV1D: channels x length, kernel taps per row. */
OpsCase
_S_make_conv1d (const char *name, const uint32_t channels, const uint32_t length,
                const uint32_t kernel, const zzs::mllee::Codec codec)
{
    OpsCase c;
    c._M_name = name;
    c._M_run = _S_run_conv1d;
    _S_fill_weight_layer (c._M_ws._M_layer, zzs::mllee::Op::CONV1D, channels * length,
                          channels * length);
    c._M_ws._M_layer._M_head_dim = static_cast<uint8_t> (length);
    c._M_ws._M_layer._M_seq = static_cast<uint8_t> (length);
    c._M_ws._M_layer._M_kernel = static_cast<uint8_t> (kernel);
    c._M_ws._M_layer._M_flags = zzs::mllee::CONV_DEPTHWISE | zzs::mllee::CONV_CAUSAL;
    std::vector<int8_t> codes (static_cast<std::size_t> (channels) * kernel);
    for (std::size_t i = 0; i < codes.size (); ++i)
        codes[i] = _S_weight_code (static_cast<uint32_t> (i));
    _S_attach_weights (c, codes, codec);
    c._M_input.resize (static_cast<std::size_t> (channels) * length);
    for (std::size_t i = 0; i < c._M_input.size (); ++i)
        c._M_input[i] = _S_activation_code (static_cast<uint32_t> (i));
    c._M_output.resize (static_cast<std::size_t> (channels) * length);
    c._M_elements = static_cast<uint64_t> (channels) * length * kernel;
    return c;
}

/** @brief One NORM / SOFTMAX / ACTIVATION case over `count` arena values. */
OpsCase
_S_make_elementwise (const char *name, const zzs::mllee::Op op, const uint32_t count,
                     const uint32_t window, const zzs::mllee::Act act)
{
    OpsCase c;
    c._M_name = name;
    _S_fill_weight_layer (c._M_ws._M_layer, op, count, count);
    c._M_ws._M_layer._M_group_size = static_cast<uint8_t> (window);
    c._M_ws._M_layer._M_groups = window != 0 ? static_cast<uint8_t> (count / window) : 0;
    c._M_ws._M_layer._M_act = act;
    c._M_model.assign (zzs::mllee::_S_model_header_size, 0);
    c._M_input.resize (count);
    for (uint32_t i = 0; i < count; ++i)
        c._M_input[i] = _S_activation_code (i);
    c._M_output.resize (count);
    c._M_elements = count;
    c._M_run = op == zzs::mllee::Op::NORM
                   ? _S_run_norm
                   : (op == zzs::mllee::Op::SOFTMAX ? _S_run_softmax : _S_run_activation);
    return c;
}

/** @brief One ATTENTION case over `seq` x `head_dim` x heads Q/K/V rows. */
OpsCase
_S_make_attention (const char *name, const uint32_t seq, const uint32_t head_dim,
                   const uint32_t heads)
{
    OpsCase c;
    c._M_name = name;
    c._M_run = _S_run_attention;
    const uint32_t length = seq * head_dim * heads;
    _S_fill_weight_layer (c._M_ws._M_layer, zzs::mllee::Op::ATTENTION, length, length);
    c._M_ws._M_layer._M_seq = static_cast<uint8_t> (seq);
    c._M_ws._M_layer._M_head_dim = static_cast<uint8_t> (head_dim);
    c._M_ws._M_layer._M_groups = static_cast<uint8_t> (heads);
    c._M_model.assign (zzs::mllee::_S_model_header_size, 0);
    c._M_input.resize (3u * length);
    for (std::size_t i = 0; i < c._M_input.size (); ++i)
        c._M_input[i] = _S_activation_code (static_cast<uint32_t> (i));
    c._M_output.resize (length);
    /* One score dot plus one value accumulate per key/query pair and dimension. */
    c._M_elements = static_cast<uint64_t> (seq) * seq * head_dim * heads;
    return c;
}

/** @brief Run one case and print reference ticks per element. */
bool
_S_measure (OpsCase &c)
{
    _S_rewind (c);
    if (c._M_run (c) != Status::OK)
        return false;

    uint64_t checksum = 0;
    const uint64_t begin = _S_tick ();
    for (uint32_t iteration = 0; iteration < _S_ops_iterations; ++iteration)
        {
            _S_rewind (c);
            if (c._M_run (c) != Status::OK)
                return false;
        }
    const uint64_t end = _S_tick ();
    for (const int8_t value : c._M_output)
        checksum += static_cast<uint64_t> (static_cast<int32_t> (value)) + 128u;

    const double per_element
        = static_cast<double> (end - begin) / (static_cast<double> (_S_ops_iterations)
                                               * static_cast<double> (c._M_elements));
    std::printf ("| %-28s | %10llu | %8.2f | %16llu |\n", c._M_name,
                 static_cast<unsigned long long> (c._M_elements), per_element,
                 static_cast<unsigned long long> (checksum));
    return true;
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

    OpsCase cases[] = {
        _S_make_linear ("LINEAR 128x128 i8 raw", 128u, 128u, Codec::RAW),
        _S_make_linear ("LINEAR 128x128 i8 packlow", 128u, 128u, Codec::PACKLOW),
        _S_make_linear ("LINEAR 64x64 i8 raw", 64u, 64u, Codec::RAW),
        _S_make_conv1d ("CONV1D dw k5 64->64 raw", 64u, 64u, 5u, Codec::RAW),
        _S_make_conv1d ("CONV1D dw k5 64->64 packlow", 64u, 64u, 5u, Codec::PACKLOW),
        _S_make_elementwise ("NORM 4096 win64", zzs::mllee::Op::NORM, 4096u, 64u,
                             zzs::mllee::Act::NONE),
        _S_make_elementwise ("SOFTMAX 4096 win64", zzs::mllee::Op::SOFTMAX, 4096u, 64u,
                             zzs::mllee::Act::NONE),
        _S_make_elementwise ("ACTIVATION relu 4096", zzs::mllee::Op::ACTIVATION, 4096u, 0u,
                             zzs::mllee::Act::RELU),
        _S_make_attention ("ATTENTION seq8 hd8 h4", 8u, 8u, 4u),
    };

    std::printf ("\n| op case | elements | ticks/element | output checksum |\n");
    std::printf ("|---|---:|---:|---:|\n");
    for (OpsCase &c : cases)
        if (!_S_measure (c))
            {
                ok = false;
                std::printf ("| %s | FAILED |\n", c._M_name);
            }

    return ok ? 0 : 1;
}