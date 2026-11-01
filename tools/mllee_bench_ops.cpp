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

#include "kernel/kernel_dsp_biquad.hpp"
#include "kernel/kernel_dsp_goertzel.hpp"
#include "kernel/kernel_dsp_polyphase.hpp"
#include "kernel/kernel_kernel_fixed_simd.hpp"
#include "mllee/mllee.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {
using zzs::kernel::dsp::_S_BiquadCascade;
using zzs::kernel::dsp::_S_BiquadCoeffs;
using zzs::kernel::dsp::_S_BiquadState;
using zzs::kernel::dsp::_S_GoertzelBin;
using zzs::kernel::dsp::_S_PolyBranch;
using zzs::kernel::dsp::_S_PolyphaseFir;
using zzs::kernel::simd::dot_widen_q15;
using zzs::mllee::Act;
using zzs::mllee::Codec;
using zzs::mllee::Dtype;
using zzs::mllee::Layer;
using zzs::mllee::Op;
using zzs::mllee::PacklowEncoder;
using zzs::mllee::PacklowOptions;
using zzs::mllee::PacklowState;
using zzs::mllee::Status;
using zzs::mllee::Workspace;

constexpr uint32_t _S_iterations = 1000;
constexpr uint32_t _S_repeats = 3;

[[nodiscard]] inline uint64_t
_S_tick () noexcept {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_ia32_rdtsc ();
#else
    return static_cast<uint64_t> (
        std::chrono::duration_cast<std::chrono::nanoseconds> (std::chrono::steady_clock::now ().time_since_epoch ())
            .count ());
#endif
}

[[nodiscard]] inline double
_S_seconds (const uint64_t start, const uint64_t end) noexcept {
#if defined(__x86_64__) || defined(__i386__)
    /* rdtsc: host's invariant TSC, report ticks directly (no divide). */
    return 0.0;
#else
    return static_cast<double> (end - start) / 1.0e9;
#endif
}

int8_t
_S_weight_code (const uint32_t index) {
    uint32_t state = index * 2246822519u + 3266489917u;
    state ^= state >> 13;
    state *= 2246822519u;
    state ^= state >> 16;
    return static_cast<int8_t> (state % 7u) - 3;
}

int8_t
_S_activation_code (const uint32_t index) {
    uint32_t state = index * 1664525u + 1013904223u;
    state ^= state >> 15;
    return static_cast<int8_t> ((state >> 8) % 255u) - 127;
}

[[nodiscard]] inline int8_t
_S_i8 (const int8_t value) noexcept {
    return value < -127 ? -127 : (value > 127 ? 127 : value);
}

/** @brief Pack int8 codes into the on-disk layout for `dtype` (I4/I2/I3). */
std::vector<uint8_t>
_S_pack_codes (const std::vector<int8_t> &codes, const Dtype dtype) noexcept {
    const uint32_t bits = zzs::mllee::_S_code_bits (dtype);
    std::vector<uint8_t> packed;
    if (dtype == Dtype::I3) {
        packed.resize ((static_cast<uint64_t> (codes.size ()) * bits + 7u) / 8u);
        for (std::size_t i = 0; i < codes.size (); ++i) {
            const uint32_t bit = static_cast<uint32_t> (i) * bits;
            packed[bit >> 3] |=
                static_cast<uint8_t> ((static_cast<uint32_t> (static_cast<uint8_t> (codes[i])) & 7u) << (bit & 7u));
        }
    } else if (bits == 4u) {
        packed.resize (codes.size () / 2u);
        for (std::size_t i = 0; i + 1 < codes.size (); i += 2u)
            packed[i / 2] = static_cast<uint8_t> ((static_cast<uint8_t> (codes[i]) & 0xFu)
                                                  | (static_cast<uint8_t> (codes[i + 1]) & 0xFu) << 4u);
    } else // bits == 2
    {
        packed.resize (codes.size () / 4u);
        for (std::size_t i = 0; i + 3 < codes.size (); i += 4u) {
            uint8_t b = 0;
            for (uint32_t k = 0; k < 4u; ++k)
                b |= static_cast<uint8_t> ((static_cast<uint8_t> (codes[i + k]) & 0x3u) << (k * 2u));
            packed[i / 4] = b;
        }
    }
    return packed;
}

/** @brief Fill the descriptor fields shared by every weight-stream case. */
void
_S_fill_weight_layer (Layer &layer, const Op op, const uint32_t in_dim, const uint32_t out_dim) noexcept {
    layer = {};
    layer._M_op = op;
    layer._M_in_dim = static_cast<uint16_t> (in_dim);
    layer._M_out_dim = static_cast<uint16_t> (out_dim);
    layer._M_dtype = Dtype::I8;
    layer._M_kernel = 1;
    layer._M_stride = 1;
    layer._M_requant_mult = 32768;
}

/** @brief One measured kernel case: workspace, model blob, operands, runner. */
struct BenchCase {
    const char *_M_name;
    Workspace _M_ws{};
    std::vector<uint8_t> _M_model;
    std::vector<int8_t> _M_input;
    std::vector<int8_t> _M_residual;
    std::vector<int8_t> _M_output;
    uint64_t _M_elements = 0;
    Status (*_M_run) (BenchCase &) noexcept = nullptr;
};

/** @brief Attach a weight stream to a case, raw or packlow-encoded. */
void
_S_attach_weights (BenchCase &c, const std::vector<int8_t> &codes, const Codec codec) noexcept {
    std::vector<uint8_t> raw;
    raw.reserve (codes.size ());
    for (const int8_t code : codes)
        raw.push_back (static_cast<uint8_t> (code));

    c._M_model.assign (zzs::mllee::_S_model_header_size, 0);
    const uint32_t offset = static_cast<uint32_t> (c._M_model.size ());
    if (codec == Codec::PACKLOW) {
        std::vector<uint8_t> encoded (raw.size () * 2u + 512u);
        PacklowOptions options;
        options._M_delta = true;
        options._M_dict_copy = false;
        const uint32_t packed = PacklowEncoder::encode (raw.data (), static_cast<uint32_t> (raw.size ()), options,
                                                        encoded.data (), static_cast<uint32_t> (encoded.size ()));
        if (packed != 0)
            encoded.resize (packed);
        else {
            c._M_model.insert (c._M_model.end (), raw.begin (), raw.end ());
            c._M_ws._M_layer._M_codec = Codec::RAW;
        }
        c._M_model.insert (c._M_model.end (), encoded.begin (), encoded.end ());
    } else
        c._M_model.insert (c._M_model.end (), raw.begin (), raw.end ());

    c._M_ws._M_layer._M_weight_off = offset;
    c._M_ws._M_layer._M_weight_len = static_cast<uint32_t> (c._M_model.size ()) - offset;
    c._M_ws._M_layer._M_codec = codec;
    const Status status =
        zzs::mllee::_S_pk_begin (c._M_ws._M_pk, c._M_model.data (), static_cast<uint32_t> (c._M_model.size ()),
                                 c._M_ws._M_layer._M_weight_off, c._M_ws._M_layer._M_weight_len, codec);
    if (status != zzs::mllee::Status::OK)
        std::printf ("  setup failed: %s\n", zzs::mllee::status_name (status));
    zzs::mllee::_S_rd_begin (c._M_ws._M_pk, nullptr, c._M_ws._M_layer._M_dtype);
}

/** @brief Re-arm the codec reader without rebuilding the blob. */
void
_S_rewind (BenchCase &c) {
    if (c._M_ws._M_layer._M_op != Op::LINEAR && c._M_ws._M_layer._M_op != Op::CONV1D)
        return;
    const Layer &layer = c._M_ws._M_layer;
    const Status status =
        zzs::mllee::_S_pk_begin (c._M_ws._M_pk, c._M_model.data (), static_cast<uint32_t> (c._M_model.size ()),
                                 layer._M_weight_off, layer._M_weight_len, layer._M_codec);
    if (status != zzs::mllee::Status::OK)
        std::printf ("  rewind failed: %s\n", zzs::mllee::status_name (status));
    zzs::mllee::_S_rd_begin (c._M_ws._M_pk, nullptr, layer._M_dtype);
}

/** @brief Attach already-packed weight bytes as a RAW stream of `dtype`. */
void
_S_attach_packed_weights (BenchCase &c, const std::vector<uint8_t> &packed, const Dtype dtype) noexcept {
    c._M_model.assign (zzs::mllee::_S_model_header_size, 0);
    const uint32_t offset = static_cast<uint32_t> (c._M_model.size ());
    c._M_model.insert (c._M_model.end (), packed.begin (), packed.end ());
    c._M_ws._M_layer._M_weight_off = offset;
    c._M_ws._M_layer._M_weight_len = static_cast<uint32_t> (packed.size ());
    c._M_ws._M_layer._M_codec = Codec::RAW;
    c._M_ws._M_layer._M_dtype = dtype;
    const Status status =
        zzs::mllee::_S_pk_begin (c._M_ws._M_pk, c._M_model.data (), static_cast<uint32_t> (c._M_model.size ()),
                                 c._M_ws._M_layer._M_weight_off, c._M_ws._M_layer._M_weight_len, Codec::RAW);
    if (status != zzs::mllee::Status::OK)
        std::printf ("  setup failed: %s\n", zzs::mllee::status_name (status));
    zzs::mllee::_S_rd_begin (c._M_ws._M_pk, nullptr, dtype);
}

Status
_S_run_linear (BenchCase &c) noexcept {
    return zzs::mllee::_S_linear (c._M_ws, c._M_model.data (), c._M_input.data (), c._M_output.data ());
}

Status
_S_run_conv1d (BenchCase &c) noexcept {
    return zzs::mllee::_S_conv1d (c._M_ws, c._M_model.data (), c._M_input.data (), c._M_output.data ());
}

Status
_S_run_norm (BenchCase &c) noexcept {
    const int8_t *const residual = c._M_residual.empty () ? nullptr : c._M_residual.data ();
    return zzs::mllee::_S_norm (c._M_ws._M_layer, c._M_model.data (), c._M_input.data (), residual,
                                c._M_output.data ());
}

Status
_S_run_batch_norm (BenchCase &c) noexcept {
    return zzs::mllee::_S_batch_norm (c._M_ws._M_layer, c._M_model.data (), c._M_input.data (), c._M_output.data ());
}

Status
_S_run_group_norm (BenchCase &c) noexcept {
    return zzs::mllee::_S_group_norm (c._M_ws._M_layer, c._M_model.data (), c._M_input.data (), c._M_output.data ());
}

Status
_S_run_softmax (BenchCase &c) noexcept {
    return zzs::mllee::_S_softmax (c._M_ws._M_layer, c._M_input.data (), c._M_output.data ());
}

Status
_S_run_activation (BenchCase &c) noexcept {
    return zzs::mllee::_S_activation (c._M_ws._M_layer, c._M_input.data (), c._M_output.data ());
}

Status
_S_run_attention (BenchCase &c) noexcept {
    return zzs::mllee::_S_attention (c._M_ws._M_layer, c._M_input.data (), c._M_output.data ());
}

Status
_S_run_binary (BenchCase &c) noexcept {
    return zzs::mllee::_S_binary (c._M_ws._M_layer, c._M_input.data (), c._M_output.data (), c._M_output.data ());
}

Status
_S_run_pool (BenchCase &c) noexcept {
    return zzs::mllee::_S_pool (c._M_ws._M_layer, c._M_input.data (), c._M_output.data ());
}

/** @brief Print the markdown header row for the results table. */
void
_S_print_header () {
    std::printf ("| op | config | elements | ticks/element | ticks/call | checksum |\n");
    std::printf ("|---|---|---:|---:|---:|---:|\n");
}

/** @brief One LINEAR row block: in_dim x out_dim int8 weights, raw or packlow. */
BenchCase
_S_make_linear (const char *name, const uint32_t in_dim, const uint32_t out_dim, const Codec codec) {
    BenchCase c;
    c._M_name = name;
    c._M_run = _S_run_linear;
    _S_fill_weight_layer (c._M_ws._M_layer, Op::LINEAR, in_dim, out_dim);
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
BenchCase
_S_make_conv1d (const char *name, const uint32_t channels, const uint32_t length, const uint32_t kernel,
                const Codec codec) {
    BenchCase c;
    c._M_name = name;
    c._M_run = _S_run_conv1d;
    _S_fill_weight_layer (c._M_ws._M_layer, Op::CONV1D, channels * length, channels * length);
    c._M_ws._M_layer._M_head_dim = static_cast<uint8_t> (length);
    c._M_ws._M_layer._M_seq = static_cast<uint8_t> (length);
    c._M_ws._M_layer._M_kernel = static_cast<uint8_t> (kernel);
    c._M_ws._M_layer._M_flags = static_cast<uint8_t> (zzs::mllee::CONV_DEPTHWISE | zzs::mllee::CONV_CAUSAL);
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

/** @brief One LINEAR case with a pre-packed sub-byte weight stream (I4/I2/I3). */
BenchCase
_S_make_packed_linear (const char *name, const uint32_t in_dim, const uint32_t out_dim, const Dtype dtype) {
    BenchCase c;
    c._M_name = name;
    c._M_run = _S_run_linear;
    _S_fill_weight_layer (c._M_ws._M_layer, Op::LINEAR, in_dim, out_dim);
    c._M_ws._M_layer._M_dtype = dtype;
    std::vector<int8_t> codes (static_cast<std::size_t> (in_dim) * out_dim);
    for (std::size_t i = 0; i < codes.size (); ++i)
        codes[i] = _S_weight_code (static_cast<uint32_t> (i));
    const std::vector<uint8_t> packed = _S_pack_codes (codes, dtype);
    _S_attach_packed_weights (c, packed, dtype);
    c._M_input.resize (in_dim);
    for (uint32_t i = 0; i < in_dim; ++i)
        c._M_input[i] = _S_activation_code (i);
    c._M_output.resize (out_dim);
    c._M_elements = static_cast<uint64_t> (in_dim) * out_dim;
    return c;
}

/** @brief One NORM / BATCH_NORM / GROUP_NORM / SOFTMAX / ACTIVATION case. */
BenchCase
_S_make_elementwise (const char *name, const Op op, const uint32_t count, const uint32_t window, const Act act,
                     const bool residual) {
    BenchCase c;
    c._M_name = name;
    _S_fill_weight_layer (c._M_ws._M_layer, op, count, count);
    const bool instance_norm = op == Op::NORM && window != 0;
    const bool batch_norm = op == Op::BATCH_NORM && window != 0;
    const bool group_norm = op == Op::GROUP_NORM && window != 0;
    c._M_ws._M_layer._M_group_size = static_cast<uint8_t> (window);
    c._M_ws._M_layer._M_groups = window != 0 ? static_cast<uint8_t> (count / window) : 0;
    if (batch_norm)
        c._M_ws._M_layer._M_seq = static_cast<uint8_t> (window);
    if (instance_norm)
        c._M_ws._M_layer._M_flags = static_cast<uint8_t> (c._M_ws._M_layer._M_flags | zzs::mllee::NORM_INSTANCE);
    c._M_ws._M_layer._M_act = act;
    c._M_ws._M_layer._M_requant_mult = 32768;
    c._M_input.resize (count);
    for (uint32_t i = 0; i < count; ++i)
        c._M_input[i] = _S_activation_code (i);
    c._M_output.resize (count);
    if (residual) {
        c._M_residual.resize (count);
        for (uint32_t i = 0; i < count; ++i)
            c._M_residual[i] = _S_activation_code (count + i);
        c._M_ws._M_layer._M_flags = static_cast<uint8_t> (c._M_ws._M_layer._M_flags | zzs::mllee::NORM_RESIDUAL);
    }
    c._M_elements = count;
    c._M_run = op == Op::NORM
                   ? _S_run_norm
                   : (op == Op::BATCH_NORM
                          ? _S_run_batch_norm
                          : (op == Op::GROUP_NORM ? _S_run_group_norm
                                                  : (op == Op::SOFTMAX ? _S_run_softmax : _S_run_activation)));
    return c;
}

/** @brief One ATTENTION case over seq x head_dim x heads Q/K/V rows. */
BenchCase
_S_make_attention (const char *name, const uint32_t seq, const uint32_t head_dim, const uint32_t heads) {
    BenchCase c;
    c._M_name = name;
    c._M_run = _S_run_attention;
    const uint32_t length = seq * head_dim * heads;
    _S_fill_weight_layer (c._M_ws._M_layer, Op::ATTENTION, length, length);
    c._M_ws._M_layer._M_seq = static_cast<uint8_t> (seq);
    c._M_ws._M_layer._M_head_dim = static_cast<uint8_t> (head_dim);
    c._M_ws._M_layer._M_groups = static_cast<uint8_t> (heads);
    c._M_ws._M_layer._M_flags = static_cast<uint8_t> (zzs::mllee::ATTN_CAUSAL);
    c._M_input.resize (3u * length);
    for (std::size_t i = 0; i < c._M_input.size (); ++i)
        c._M_input[i] = _S_activation_code (static_cast<uint32_t> (i));
    c._M_output.resize (length);
    /* One score dot plus one value accumulate per key/query pair and dimension. */
    c._M_elements = static_cast<uint64_t> (seq) * seq * head_dim * heads;
    return c;
}

/** @brief One BINARY (ADD/SUB/MUL) case over count elements. */
BenchCase
_S_make_binary (const char *name, const Op op, const uint32_t count) {
    BenchCase c;
    c._M_name = name;
    c._M_run = _S_run_binary;
    _S_fill_weight_layer (c._M_ws._M_layer, op, count, count);
    c._M_ws._M_layer._M_requant_mult = 32768;
    c._M_input.resize (count);
    for (uint32_t i = 0; i < count; ++i)
        c._M_input[i] = _S_activation_code (i);
    c._M_output.resize (count);
    c._M_elements = count;
    return c;
}

/** @brief One POOL (MAX/AVG/GLOBAL_AVG) case over count elements.
 *
 * `_M_head_dim` is uint8, so the per-channel length must fit in 255.  Pick a
 * head length that divides `count` and report `count` as the element count.
 */
BenchCase
_S_make_pool (const char *name, const Op op, const uint32_t count, const uint32_t kernel, const uint32_t seq) {
    BenchCase c;
    c._M_name = name;
    c._M_run = _S_run_pool;
    /* Pick a head length <= 255 that divides count; channels = count / head. */
    uint32_t head = 64u;
    while (head > 1u && (count % head != 0 || head > 255u))
        --head;
    const uint32_t channels = count / head;
    const uint32_t out_length = op == Op::GLOBAL_AVG_POOL ? 1u : seq;
    _S_fill_weight_layer (c._M_ws._M_layer, op, count, count);
    c._M_ws._M_layer._M_head_dim = static_cast<uint8_t> (head);
    c._M_ws._M_layer._M_seq = static_cast<uint8_t> (out_length);
    c._M_ws._M_layer._M_kernel = static_cast<uint8_t> (kernel);
    c._M_ws._M_layer._M_stride = 1;
    c._M_ws._M_layer._M_out_dim = channels * out_length;
    c._M_input.resize (count);
    for (uint32_t i = 0; i < count; ++i)
        c._M_input[i] = _S_activation_code (i);
    c._M_output.resize (channels * out_length);
    c._M_elements = count;
    return c;
}

/** @brief Codec tax: time _S_rd_row_code per code vs staged simd::dot_i8. */
void
_S_measure_codec_tax (const char *name, const uint32_t codes) {
    BenchCase c;
    c._M_name = name;
    c._M_run = _S_run_linear;
    _S_fill_weight_layer (c._M_ws._M_layer, Op::LINEAR, codes, codes);
    std::vector<int8_t> weight_codes (codes);
    for (uint32_t i = 0; i < codes; ++i)
        weight_codes[i] = _S_weight_code (i);
    _S_attach_weights (c, weight_codes, Codec::PACKLOW);
    c._M_input.resize (codes);
    for (uint32_t i = 0; i < codes; ++i)
        c._M_input[i] = _S_activation_code (i);
    c._M_output.resize (codes);
    c._M_elements = codes;

    if (zzs::mllee::_S_rd_row (c._M_ws._M_pk, c._M_model.data (), nullptr, 0, codes) != Status::OK)
        return;

    uint64_t best_code = UINT64_MAX;
    uint64_t best_dot = UINT64_MAX;
    for (uint32_t repeat = 0; repeat < _S_repeats; ++repeat) {
        const uint64_t t0 = _S_tick ();
        int32_t acc_code = 0;
        for (uint32_t iteration = 0; iteration < _S_iterations; ++iteration) {
            acc_code = 0;
            for (uint32_t i = 0; i < codes; ++i)
                acc_code += zzs::mllee::_S_rd_row_code (c._M_ws._M_pk, c._M_model.data (), i, codes)
                            * static_cast<int32_t> (c._M_input[i]);
        }
        const uint64_t e_code = _S_tick () - t0;
        if (e_code < best_code)
            best_code = e_code;

        const uint64_t t1 = _S_tick ();
        int32_t acc_dot = 0;
        for (uint32_t iteration = 0; iteration < _S_iterations; ++iteration) {
            acc_dot = zzs::mllee::simd::dot_i8 (reinterpret_cast<const int8_t *> (c._M_ws._M_pk._M_ring),
                                                c._M_input.data (), codes);
        }
        const uint64_t e_dot = _S_tick () - t1;
        if (e_dot < best_dot)
            best_dot = e_dot;
    }

    (void) zzs::mllee::_S_i8_code (static_cast<int8_t> (best_code));
    (void) zzs::mllee::_S_i8_code (static_cast<int8_t> (best_dot));

    const double ticks_code_per_code =
        static_cast<double> (best_code) / (static_cast<double> (_S_iterations) * static_cast<double> (codes));
    const double ticks_dot_per_code =
        static_cast<double> (best_dot) / (static_cast<double> (_S_iterations) * static_cast<double> (codes));

    std::printf ("| %-22s | codec-tax %u codes | %11u | %13.2f | %10.2f | %15u |\n", name, codes, codes,
                 ticks_code_per_code, ticks_dot_per_code, 0u);
}

/** @brief Run `case` min-over-`repeats` after one warm-up, print one row. */
bool
_S_measure (BenchCase &c, const char *config) {
    _S_rewind (c);
    if (c._M_run (c) != Status::OK)
        return false;

    /* Warm-up iteration (not timed). */
    _S_rewind (c);
    if (c._M_run (c) != Status::OK)
        return false;

    uint64_t best = UINT64_MAX;
    for (uint32_t repeat = 0; repeat < _S_repeats; ++repeat) {
        _S_rewind (c);
        const uint64_t begin = _S_tick ();
        for (uint32_t iteration = 0; iteration < _S_iterations; ++iteration) {
            _S_rewind (c);
            if (c._M_run (c) != Status::OK)
                return false;
        }
        const uint64_t elapsed = _S_tick () - begin;
        if (elapsed < best)
            best = elapsed;
    }

    uint64_t checksum = 0;
    for (const int8_t value : c._M_output)
        checksum += static_cast<uint64_t> (static_cast<int32_t> (value)) + 128u;

    const double ticks_per_element =
        static_cast<double> (best) / (static_cast<double> (_S_iterations) * static_cast<double> (c._M_elements));
    const double ticks_per_call = static_cast<double> (best) / static_cast<double> (_S_iterations);

    std::printf ("| %-22s | %-24s | %11llu | %13.2f | %10.2f | %15llu |\n", c._M_name, config,
                 static_cast<unsigned long long> (c._M_elements), ticks_per_element, ticks_per_call,
                 static_cast<unsigned long long> (checksum));
    return true;
}

/** @brief DSP: biquad_step / biquad_cascade_step over 100000 samples. */
void
_S_bench_biquad () {
    _S_BiquadCoeffs coeffs{};
    if (zzs::kernel::dsp::bandpass_coeffs (100u, 1000u, 32768, coeffs) != zzs::kernel::Status::OK)
        return;

    _S_BiquadState state{};
    std::vector<int16_t> samples (100000);
    for (uint32_t i = 0; i < samples.size (); ++i)
        samples[i] = static_cast<int16_t> ((i * 37u) % 65536 - 32768);

    uint64_t best_single = UINT64_MAX;
    uint64_t best_cascade = UINT64_MAX;
    for (uint32_t repeat = 0; repeat < _S_repeats; ++repeat) {
        zzs::kernel::dsp::biquad_reset (state);
        const uint64_t t0 = _S_tick ();
        int16_t y = 0;
        for (uint32_t i = 0; i < samples.size (); ++i)
            y = zzs::kernel::dsp::biquad_step (coeffs, state, samples[i]);
        const uint64_t e = _S_tick () - t0;
        if (e < best_single)
            best_single = e;
        (void) y;

        _S_BiquadCascade cascade{};
        cascade._M_section_count = 2u;
        cascade._M_coeff[0] = coeffs;
        cascade._M_coeff[1] = coeffs;
        zzs::kernel::dsp::biquad_cascade_reset (cascade);
        const uint64_t t1 = _S_tick ();
        for (uint32_t i = 0; i < samples.size (); ++i)
            y = zzs::kernel::dsp::biquad_cascade_step (cascade, samples[i]);
        const uint64_t e2 = _S_tick () - t1;
        if (e2 < best_cascade)
            best_cascade = e2;
        (void) y;
    }

    const double ticks_single = static_cast<double> (best_single) / static_cast<double> (samples.size ());
    const double ticks_cascade = static_cast<double> (best_cascade) / static_cast<double> (samples.size ());
    std::printf ("| %-22s | %-24s | %11u | %13.2f | %10.2f | %15llu |\n", "BIQUAD", "step 100k samples",
                 static_cast<unsigned int> (samples.size ()), ticks_single, ticks_single,
                 static_cast<unsigned long long> (best_single));
    std::printf ("| %-22s | %-24s | %11u | %13.2f | %10.2f | %15llu |\n", "BIQUAD_CASCADE", "step 100k samples",
                 static_cast<unsigned int> (samples.size ()), ticks_cascade, ticks_cascade,
                 static_cast<unsigned long long> (best_cascade));
}

/** @brief DSP: goertzel_bin_step over 100000 samples per bin, 4 bins. */
void
_S_bench_goertzel () {
    const uint32_t freqs[] = {50u, 60u, 100u, 120u};
    const uint32_t sample_rate = 200u;
    std::vector<int16_t> samples (100000);
    for (uint32_t i = 0; i < samples.size (); ++i)
        samples[i] = static_cast<int16_t> ((i * 97u) % 65536 - 32768);

    for (const uint32_t f : freqs) {
        _S_GoertzelBin bin{};
        if (zzs::kernel::dsp::goertzel_bin_configure (bin, f, sample_rate) != zzs::kernel::Status::OK)
            continue;
        uint64_t best = UINT64_MAX;
        int16_t last = 0;
        for (uint32_t repeat = 0; repeat < _S_repeats; ++repeat) {
            zzs::kernel::dsp::goertzel_bin_reset (bin);
            const uint64_t t0 = _S_tick ();
            for (uint32_t i = 0; i < samples.size (); ++i)
                zzs::kernel::dsp::goertzel_bin_step (bin, samples[i]);
            const uint64_t e = _S_tick () - t0;
            if (e < best)
                best = e;
            last = static_cast<int16_t> (zzs::kernel::dsp::goertzel_bin_magnitude_q7 (bin));
        }
        (void) last;
        const double ticks = static_cast<double> (best) / static_cast<double> (samples.size ());
        std::printf ("| %-22s | %-24s | %11u | %13.2f | %10.2f | %15llu |\n", "GOERTZEL", "bin 50/60/100/120 Hz",
                     static_cast<unsigned int> (f), ticks, ticks, static_cast<unsigned long long> (best));
    }
}

/** @brief DSP: polyphase_step_branch over 10000 outputs. */
void
_S_bench_polyphase () {
    _S_PolyphaseFir fir{};
    fir._M_branch_count = 4u;
    fir._M_delay_len = 8u;
    fir._M_delay_idx = 0u;
    for (uint32_t i = 0; i < fir._M_delay_len; ++i)
        fir._M_delay[i] = 0;
    int16_t coeffs[8] = {100, 200, 300, 400, 400, 300, 200, 100};
    if (zzs::kernel::dsp::polyphase_load_branch (fir, 0, coeffs, 8u) != zzs::kernel::Status::OK)
        return;

    std::vector<int16_t> samples (10000 + 8);
    for (uint32_t i = 0; i < samples.size (); ++i)
        samples[i] = static_cast<int16_t> ((i * 53u) % 65536 - 32768);

    uint64_t best = UINT64_MAX;
    int16_t last = 0;
    for (uint32_t i = 0; i < 8u; ++i) {
        const zzs::kernel::Status st = zzs::kernel::dsp::polyphase_push (fir, samples[i]);
        if (st != zzs::kernel::Status::OK)
            return;
    }
    for (uint32_t repeat = 0; repeat < _S_repeats; ++repeat) {
        const uint64_t t0 = _S_tick ();
        for (uint32_t i = 8u; i < samples.size (); ++i) {
            const zzs::kernel::Status st = zzs::kernel::dsp::polyphase_push (fir, samples[i]);
            if (st != zzs::kernel::Status::OK)
                return;
            last = zzs::kernel::dsp::polyphase_step_branch (fir, 0u);
        }
        const uint64_t e = _S_tick () - t0;
        if (e < best)
            best = e;
    }
    (void) last;
    const double ticks = static_cast<double> (best) / 10000.0;
    std::printf ("| %-22s | %-24s | %11u | %13.2f | %10.2f | %15llu |\n", "POLYPHASE", "step_branch 10000 outputs",
                 10000u, ticks, ticks, static_cast<unsigned long long> (best));
}

/** @brief DSP: dot_widen_q15 over 1024/4096 samples. */
void
_S_bench_dot_widen (const uint32_t count) {
    std::vector<int16_t> lhs (count), rhs (count);
    for (uint32_t i = 0; i < count; ++i) {
        lhs[i] = static_cast<int16_t> ((i * 11u) % 65536 - 32768);
        rhs[i] = static_cast<int16_t> ((i * 13u) % 65536 - 32768);
    }
    uint64_t best = UINT64_MAX;
    int64_t acc = 0;
    for (uint32_t repeat = 0; repeat < _S_repeats; ++repeat) {
        const uint64_t t0 = _S_tick ();
        for (uint32_t iteration = 0; iteration < _S_iterations; ++iteration)
            acc = zzs::kernel::simd::dot_widen_q15 (lhs.data (), rhs.data (), count);
        const uint64_t e = _S_tick () - t0;
        if (e < best)
            best = e;
    }
    (void) acc;
    const double ticks =
        static_cast<double> (best) / (static_cast<double> (_S_iterations) * static_cast<double> (count));
    std::printf ("| %-22s | %-24s | %11u | %13.2f | %10.2f | %15llu |\n", "DOT_WIDEN_Q15", "int16 dot product", count,
                 ticks, ticks, static_cast<unsigned long long> (best));
}

} // namespace

int
main () {
    _S_print_header ();
    bool ok = true;

    /* ---- mllee core ops: LINEAR ---- */
    BenchCase linear_cases[] = {
        _S_make_linear ("LINEAR", 64u, 64u, Codec::RAW),   _S_make_linear ("LINEAR", 64u, 64u, Codec::PACKLOW),
        _S_make_linear ("LINEAR", 128u, 128u, Codec::RAW), _S_make_linear ("LINEAR", 128u, 128u, Codec::PACKLOW),
        _S_make_linear ("LINEAR", 256u, 256u, Codec::RAW), _S_make_linear ("LINEAR", 256u, 256u, Codec::PACKLOW),
    };
    for (BenchCase &c : linear_cases)
        if (!_S_measure (c, "i8")) {
            ok = false;
            std::printf ("| %s | FAILED |\n", c._M_name);
        }

    /* ---- mllee core ops: LINEAR packed i4/i2/i3 (RAW codec, packed dot path) ---- */
    BenchCase packed_cases[] = {
        _S_make_packed_linear ("LINEAR", 128u, 128u, Dtype::I4),
        _S_make_packed_linear ("LINEAR", 128u, 128u, Dtype::I2),
        _S_make_packed_linear ("LINEAR", 128u, 128u, Dtype::I3),
    };
    for (BenchCase &c : packed_cases)
        if (!_S_measure (c, "packed dot")) {
            ok = false;
            std::printf ("| %s | FAILED |\n", c._M_name);
        }

    /* ---- mllee core ops: CONV1D ---- */
    BenchCase conv_cases[] = {
        _S_make_conv1d ("CONV1D", 32u, 64u, 3u, Codec::RAW),  _S_make_conv1d ("CONV1D", 32u, 64u, 3u, Codec::PACKLOW),
        _S_make_conv1d ("CONV1D", 32u, 64u, 5u, Codec::RAW),  _S_make_conv1d ("CONV1D", 32u, 64u, 5u, Codec::PACKLOW),
        _S_make_conv1d ("CONV1D", 32u, 64u, 7u, Codec::RAW),  _S_make_conv1d ("CONV1D", 64u, 128u, 3u, Codec::RAW),
        _S_make_conv1d ("CONV1D", 64u, 128u, 5u, Codec::RAW), _S_make_conv1d ("CONV1D", 64u, 128u, 7u, Codec::RAW),
    };
    for (BenchCase &c : conv_cases)
        if (!_S_measure (c, "depthwise")) {
            ok = false;
            std::printf ("| %s | FAILED |\n", c._M_name);
        }

    /* ---- mllee core ops: NORM / BATCH_NORM / GROUP_NORM ---- */
    BenchCase norm_cases[] = {
        _S_make_elementwise ("NORM", Op::NORM, 1024u, 16u, Act::NONE, false),
        _S_make_elementwise ("NORM", Op::NORM, 1024u, 64u, Act::NONE, false),
        _S_make_elementwise ("NORM", Op::NORM, 1024u, 256u, Act::NONE, false),
        _S_make_elementwise ("NORM", Op::NORM, 4096u, 16u, Act::NONE, false),
        _S_make_elementwise ("NORM", Op::NORM, 4096u, 64u, Act::NONE, false),
        _S_make_elementwise ("NORM", Op::NORM, 4096u, 256u, Act::NONE, false),
        _S_make_elementwise ("BATCH_NORM", Op::BATCH_NORM, 1024u, 16u, Act::NONE, false),
        _S_make_elementwise ("BATCH_NORM", Op::BATCH_NORM, 1024u, 64u, Act::NONE, false),
        _S_make_elementwise ("BATCH_NORM", Op::BATCH_NORM, 4096u, 64u, Act::NONE, false),
        _S_make_elementwise ("GROUP_NORM", Op::GROUP_NORM, 1024u, 16u, Act::NONE, false),
        _S_make_elementwise ("GROUP_NORM", Op::GROUP_NORM, 1024u, 64u, Act::NONE, false),
        _S_make_elementwise ("GROUP_NORM", Op::GROUP_NORM, 4096u, 256u, Act::NONE, false),
        _S_make_elementwise ("NORM residual", Op::NORM, 4096u, 64u, Act::NONE, true),
        _S_make_elementwise ("NORM no-residual", Op::NORM, 4096u, 64u, Act::NONE, false),
    };
    for (BenchCase &c : norm_cases)
        if (!_S_measure (c, "norm")) {
            ok = false;
            std::printf ("| %s | FAILED |\n", c._M_name);
        }

    /* ---- mllee core ops: SOFTMAX ---- */
    BenchCase softmax_cases[] = {
        _S_make_elementwise ("SOFTMAX", Op::SOFTMAX, 256u, 16u, Act::NONE, false),
        _S_make_elementwise ("SOFTMAX", Op::SOFTMAX, 256u, 64u, Act::NONE, false),
        _S_make_elementwise ("SOFTMAX", Op::SOFTMAX, 256u, 256u, Act::NONE, false),
        _S_make_elementwise ("SOFTMAX", Op::SOFTMAX, 1024u, 64u, Act::NONE, false),
        _S_make_elementwise ("SOFTMAX", Op::SOFTMAX, 1024u, 256u, Act::NONE, false),
    };
    for (BenchCase &c : softmax_cases)
        if (!_S_measure (c, "softmax")) {
            ok = false;
            std::printf ("| %s | FAILED |\n", c._M_name);
        }

    /* ---- mllee core ops: ACTIVATION, every Act ---- */
    const Act acts[] = {
        Act::RELU, Act::SIGMOID, Act::TANH,      Act::GELU,        Act::LEAKY_RELU,
        Act::ELU,  Act::SILU,    Act::HARDSWISH, Act::HARDSIGMOID,
    };
    const char *const act_names[] = {
        "relu", "sigmoid", "tanh", "gelu", "leaky_relu", "elu", "silu", "hardswish", "hardsigmoid",
    };
    for (std::size_t a = 0; a < sizeof (acts) / sizeof (acts[0]); ++a) {
        BenchCase c = _S_make_elementwise ("ACTIVATION", Op::ACTIVATION, 4096u, 0u, acts[a], false);
        if (!_S_measure (c, act_names[a])) {
            ok = false;
            std::printf ("| ACTIVATION %s | FAILED |\n", act_names[a]);
        }
    }

    /* ---- mllee core ops: ATTENTION ---- */
    BenchCase attn_cases[] = {
        _S_make_attention ("ATTENTION", 4u, 4u, 2u),  _S_make_attention ("ATTENTION", 4u, 8u, 2u),
        _S_make_attention ("ATTENTION", 8u, 4u, 4u),  _S_make_attention ("ATTENTION", 8u, 8u, 4u),
        _S_make_attention ("ATTENTION", 16u, 4u, 2u), _S_make_attention ("ATTENTION", 16u, 8u, 4u),
    };
    for (BenchCase &c : attn_cases)
        if (!_S_measure (c, "self-attn")) {
            ok = false;
            std::printf ("| %s | FAILED |\n", c._M_name);
        }

    /* ---- mllee core ops: BINARY ---- */
    BenchCase binary_cases[] = {
        _S_make_binary ("BINARY", Op::ADD, 4096u),
        _S_make_binary ("BINARY", Op::SUB, 4096u),
        _S_make_binary ("BINARY", Op::MUL, 4096u),
    };
    for (BenchCase &c : binary_cases)
        if (!_S_measure (c, "elementwise")) {
            ok = false;
            std::printf ("| %s | FAILED |\n", c._M_name);
        }

    /* ---- mllee core ops: POOL ---- */
    BenchCase pool_cases[] = {
        _S_make_pool ("POOL", Op::MAX_POOL1D, 1024u, 4u, 16u),
        _S_make_pool ("POOL", Op::AVG_POOL1D, 1024u, 4u, 16u),
        _S_make_pool ("POOL", Op::GLOBAL_AVG_POOL, 1024u, 4u, 1u),
    };
    for (BenchCase &c : pool_cases)
        if (!_S_measure (c, "pool")) {
            ok = false;
            std::printf ("| %s | FAILED |\n", c._M_name);
        }

    /* ---- mllee core ops: packed-codec tax (_S_rd_row_code vs staged SIMD) ---- */
    _S_measure_codec_tax ("LINEAR", 64u);
    _S_measure_codec_tax ("LINEAR", 128u);
    _S_measure_codec_tax ("LINEAR", 256u);

    std::printf ("\n");

    /* ---- kernel/ DSP ---- */
    _S_bench_biquad ();
    _S_bench_goertzel ();
    _S_bench_polyphase ();
    _S_bench_dot_widen (1024u);
    _S_bench_dot_widen (4096u);

    return ok ? 0 : 1;
}
