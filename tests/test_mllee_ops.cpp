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
#include "mllee_test_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace
{
using zzs::mllee::Act;
using zzs::mllee::Codec;
using zzs::mllee::Dtype;
using zzs::mllee::Op;
using zzs::mllee::Status;

struct Executed
{
    bool _M_ok = false;
    Status _M_status = Status::OK;
    std::vector<std::int8_t> _M_values;
};

Executed
execute (std::vector<mllee_test::LayerSpec> layers, const std::vector<std::int8_t> &input,
         const uint16_t arena_bytes = zzs::mllee::_S_arena_bytes)
{
    const std::vector<uint8_t> blob =
        mllee_test::BlobBuilder (std::move (layers)).build (arena_bytes);
    zzs::mllee::Core core;
    auto result = core.run (blob.data (), static_cast<uint32_t> (blob.size ()), input.data (),
                            static_cast<uint32_t> (input.size ()));
    Executed executed;
    if (result.has_value ())
        {
            executed._M_ok = true;
            executed._M_values.assign (result.value ()._M_data,
                                       result.value ()._M_data + result.value ()._M_len);
        }
    else
        executed._M_status = result.error ().kind ();
    return executed;
}

Status
status_of (const std::vector<uint8_t> &blob, const std::vector<std::int8_t> &input)
{
    zzs::mllee::Core core;
    auto result = core.run (blob.data (), static_cast<uint32_t> (blob.size ()), input.data (),
                            static_cast<uint32_t> (input.size ()));
    return result.has_value () ? Status::OK : result.error ().kind ();
}

mllee_test::LayerSpec
external_layer (const Op op, const uint16_t in_dim, const uint16_t out_dim)
{
    mllee_test::LayerSpec layer;
    layer._M_op = op;
    layer._M_in_dim = in_dim;
    layer._M_out_dim = out_dim;
    layer._M_in_off = zzs::mllee::_S_external_input;
    layer._M_out_off = 0;
    layer._M_flags = zzs::mllee::INPUT_EXTERNAL;
    return layer;
}

void
append_bits (std::vector<std::int8_t> &weights, const uint8_t bits)
{
    weights.push_back (static_cast<std::int8_t> (bits));
}

std::vector<uint8_t>
single_blob (const mllee_test::LayerSpec &layer)
{
    return mllee_test::BlobBuilder ({ layer }).build (zzs::mllee::_S_arena_bytes);
}
} // namespace

TEST_CASE ("linear int8 accumulates, requantizes and clamps", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 4, 2);
    layer._M_weights = { 1, 2, 3, 4, -1, 0, 1, -2 };
    layer._M_bias = { 5, -5 };
    const Executed executed = execute ({ layer }, { 10, -20, 30, 40 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 127, -65 });
}

TEST_CASE ("linear per-output multipliers", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 4, 2);
    layer._M_weights = { 1, 2, 3, 4, -1, 0, 1, -2 };
    layer._M_multipliers = { 32768, 16384 };
    const Executed executed = execute ({ layer }, { 10, -20, 30, 40 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 127, -30 });
}

TEST_CASE ("linear applies activation after bias", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 4, 2);
    layer._M_weights = { 1, 2, 3, 4, -1, 0, 1, -2 };
    layer._M_act = Act::RELU;
    const Executed executed = execute ({ layer }, { 10, -20, 30, 40 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 127, 0 });
}

TEST_CASE ("linear group scales", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 4, 1);
    layer._M_weights = { 1, 2, 3, 4 };
    layer._M_group_size = 2;
    layer._M_groups = 2;
    layer._M_multipliers = { 32768, 16384 };
    const Executed executed = execute ({ layer }, { 10, -20, 30, 40 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 95 });
}

TEST_CASE ("linear int4 weights", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 4, 2);
    layer._M_dtype = Dtype::I4;
    layer._M_weights = { 1, -2, 3, -4, -7, 7, 0, -1 };
    const Executed executed = execute ({ layer }, { 2, 3, 4, 5 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ -12, 2 });
}

TEST_CASE ("linear float8 e4m3 weights use the flash LUT", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 4, 2);
    layer._M_dtype = Dtype::F8E4M3;
    layer._M_lut.resize (256);
    zzs::mllee::build_weight_lut_e4m3 (1.5f, layer._M_lut.data ());
    append_bits (layer._M_weights, 0x30);
    append_bits (layer._M_weights, 0xB0);
    append_bits (layer._M_weights, 0x38);
    append_bits (layer._M_weights, 0xB8);
    append_bits (layer._M_weights, 0x30);
    append_bits (layer._M_weights, 0x30);
    append_bits (layer._M_weights, 0x30);
    append_bits (layer._M_weights, 0x30);
    const Executed executed = execute ({ layer }, { 1, 2, 3, 4 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ -127, 127 });
}

TEST_CASE ("linear float8 e5m2 weights use the flash LUT", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 4, 2);
    layer._M_dtype = Dtype::F8E5M2;
    layer._M_lut.resize (256);
    zzs::mllee::build_weight_lut_e5m2 (2.0f, layer._M_lut.data ());
    append_bits (layer._M_weights, 0x3C);
    append_bits (layer._M_weights, 0xBC);
    append_bits (layer._M_weights, 0x40);
    append_bits (layer._M_weights, 0xC0);
    append_bits (layer._M_weights, 0x3C);
    append_bits (layer._M_weights, 0x3C);
    append_bits (layer._M_weights, 0x3C);
    append_bits (layer._M_weights, 0x3C);
    const Executed executed = execute ({ layer }, { 1, 1, 1, 1 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 0, 127 });
}

TEST_CASE ("linear float4 e2m1 weights use the flash LUT", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 4, 2);
    layer._M_dtype = Dtype::F4E2M1;
    layer._M_lut.resize (16);
    zzs::mllee::build_weight_lut_e2m1 (6.0f, layer._M_lut.data ());
    layer._M_weights = { 1, 2, 3, 4, 7, 7, 7, 7 };
    const Executed executed = execute ({ layer }, { 1, 1, 1, 1 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 106, 127 });
}

TEST_CASE ("linear int2 weights", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 4, 2);
    layer._M_dtype = Dtype::I2;
    layer._M_weights = { 1, -2, 1, -1, -1, -1, 1, 1 };
    const Executed executed = execute ({ layer }, { 2, 3, 4, 5 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ -5, 4 });
}

TEST_CASE ("linear int3 weights", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 8, 1);
    layer._M_dtype = Dtype::I3;
    layer._M_weights = { -4, -3, -2, -1, 0, 1, 2, 3 };
    const Executed executed = execute ({ layer }, { 1, 2, 3, 4, 5, 6, 7, 8 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 24 });
}

TEST_CASE ("linear int3 group scales", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 8, 1);
    layer._M_dtype = Dtype::I3;
    layer._M_group_size = 4;
    layer._M_groups = 2;
    layer._M_multipliers = { 32768, 16384 };
    layer._M_weights = { 1, 1, 1, 1, 1, 1, 1, 1 };
    /* 4 + (4 >> 1) = 6: the second group runs at half the requant multiplier. */
    const Executed executed = execute ({ layer }, { 1, 1, 1, 1, 1, 1, 1, 1 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 6 });
}

TEST_CASE ("linear float3 e1m1 weights use the flash LUT", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 8, 2);
    layer._M_dtype = Dtype::F3E1M1;
    layer._M_lut.resize (8);
    zzs::mllee::build_weight_lut_e1m1 (1.5f, layer._M_lut.data ());
    layer._M_weights = { 0, 1, 2, 3, 0, 0, 0, 0, 4, 5, 6, 7, 0, 0, 0, 0 };
    const Executed executed = execute ({ layer }, std::vector<std::int8_t> (8, 1));
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 127, -127 });
}

TEST_CASE ("embedding reads sub-byte rows", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::EMBEDDING, 3, 24);
    layer._M_head_dim = 8;
    layer._M_dtype = Dtype::I2;
    /* Rows 0..2 of eight int2 codes. */
    layer._M_weights = { 1,  1,  1,  1,  -1, -1, -1, -1,  0, 0, 0, 0, 1, 1, 1, 1,
                         -2, -2, -2, -2, 0,  0,  0,  0 };
    const Executed executed = execute ({ layer }, { 0, 1, 2 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 1,  1,  1,  1,  -1, -1, -1, -1,
                                                             0,  0,  0,  0,  1,  1,  1,  1,
                                                             -2, -2, -2, -2, 0,  0,  0,  0 });
}

TEST_CASE ("embedding reads bit-packed 3-bit rows", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::EMBEDDING, 2, 16);
    layer._M_head_dim = 8;
    layer._M_dtype = Dtype::I3;
    layer._M_weights = { -4, -3, -2, -1, 0, 1, 2, 3, 3, 2, 1, 0, -1, -2, -3, -4 };
    const Executed executed = execute ({ layer }, { 0, 1 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ -4, -3, -2, -1, 0, 1, 2, 3,
                                                             3,  2,  1,  0,  -1, -2, -3, -4 });
}

TEST_CASE ("embedding applies the float3 flash LUT", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::EMBEDDING, 2, 16);
    layer._M_head_dim = 8;
    layer._M_dtype = Dtype::F3E1M1;
    layer._M_lut.resize (8);
    zzs::mllee::build_weight_lut_e1m1 (1.5f, layer._M_lut.data ());
    layer._M_weights = { 0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0 };
    const Executed executed = execute ({ layer }, { 0, 1 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values
             == std::vector<std::int8_t>{ 0, 42, 85, 127, 0, -42, -85, -127,
                                          -127, -85, -42, 0, 127, 85, 42, 0 });
}

TEST_CASE ("conv1d int4 weights", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::CONV1D, 4, 4);
    layer._M_head_dim = 4;
    layer._M_seq = 4;
    layer._M_kernel = 2;
    layer._M_dtype = Dtype::I4;
    layer._M_weights = { 2, -3 };
    const Executed executed = execute ({ layer }, { 1, 2, 3, 4 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ -4, -5, -6, 8 });
}

TEST_CASE ("conv1d bit-packed int3 weights", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::CONV1D, 8, 8);
    layer._M_head_dim = 8;
    layer._M_seq = 8;
    layer._M_kernel = 8;
    layer._M_dtype = Dtype::I3;
    layer._M_weights = { 1, 2, 3, -4, -3, -2, -1, 3 };
    const Executed executed = execute ({ layer }, std::vector<std::int8_t> (8, 1));
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values
             == std::vector<std::int8_t>{ -1, -4, -3, -1, 2, 6, 3, 1 });
}

TEST_CASE ("linear packed int4 packlow rows match raw", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 64, 8);
    layer._M_dtype = Dtype::I4;
    for (std::uint32_t out = 0; out < 8; ++out)
        for (std::uint32_t i = 0; i < 64; ++i)
            layer._M_weights.push_back (
                static_cast<std::int8_t> (static_cast<std::int32_t> ((out + i) % 8u) - 4));
    std::vector<std::int8_t> input (64);
    for (std::uint32_t i = 0; i < 64; ++i)
        input[i] = static_cast<std::int8_t> (static_cast<std::int32_t> (i % 7u) - 3);

    const Executed raw = execute ({ layer }, input);
    REQUIRE (raw._M_ok);

    layer._M_use_packlow = true;
    const std::vector<uint8_t> blob = single_blob (layer);
    REQUIRE (blob[zzs::mllee::_S_model_header_size + 3]
             == static_cast<uint8_t> (Codec::PACKLOW));
    const Executed packed = execute ({ layer }, input);
    REQUIRE (packed._M_ok);
    REQUIRE (packed._M_values == raw._M_values);
}

TEST_CASE ("linear bit-packed int3 packlow rows match raw", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 64, 8);
    layer._M_dtype = Dtype::I3;
    for (std::uint32_t out = 0; out < 8; ++out)
        for (std::uint32_t i = 0; i < 64; ++i)
            layer._M_weights.push_back (
                static_cast<std::int8_t> (static_cast<std::int32_t> ((out + i) % 8u) - 4));
    std::vector<std::int8_t> input (64);
    for (std::uint32_t i = 0; i < 64; ++i)
        input[i] = static_cast<std::int8_t> (static_cast<std::int32_t> (i % 5u) - 2);

    const Executed raw = execute ({ layer }, input);
    REQUIRE (raw._M_ok);

    layer._M_use_packlow = true;
    const std::vector<uint8_t> blob = single_blob (layer);
    REQUIRE (blob[zzs::mllee::_S_model_header_size + 3]
             == static_cast<uint8_t> (Codec::PACKLOW));
    const Executed packed = execute ({ layer }, input);
    REQUIRE (packed._M_ok);
    REQUIRE (packed._M_values == raw._M_values);
}

TEST_CASE ("conv1d causal and valid padding", "[mllee][ops]")
{
    mllee_test::LayerSpec causal = external_layer (Op::CONV1D, 4, 4);
    causal._M_head_dim = 4;
    causal._M_seq = 4;
    causal._M_kernel = 2;
    causal._M_stride = 1;
    causal._M_flags |= zzs::mllee::CONV_CAUSAL;
    causal._M_weights = { 2, -1 };
    Executed executed = execute ({ causal }, { 1, 2, 3, 4 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ -1, 0, 1, 2 });

    mllee_test::LayerSpec valid = causal;
    valid._M_flags = zzs::mllee::INPUT_EXTERNAL;
    executed = execute ({ valid }, { 1, 2, 3, 4 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 0, 1, 2, 8 });
}

TEST_CASE ("conv1d multi-channel and stride", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::CONV1D, 4, 2);
    layer._M_head_dim = 2;
    layer._M_seq = 2;
    layer._M_kernel = 2;
    layer._M_stride = 1;
    layer._M_flags |= zzs::mllee::CONV_CAUSAL;
    layer._M_weights = { 1, 1, 1, 1 };
    Executed executed = execute ({ layer }, { 1, 2, 3, 4 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 4, 10 });

    mllee_test::LayerSpec strided = external_layer (Op::CONV1D, 4, 2);
    strided._M_head_dim = 4;
    strided._M_seq = 2;
    strided._M_kernel = 2;
    strided._M_stride = 2;
    strided._M_flags |= zzs::mllee::CONV_CAUSAL;
    strided._M_weights = { 1, 1 };
    executed = execute ({ strided }, { 1, 2, 3, 4 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 1, 5 });
}

TEST_CASE ("conv1d group multipliers", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::CONV1D, 4, 2);
    layer._M_head_dim = 2;
    layer._M_seq = 2;
    layer._M_kernel = 2;
    layer._M_stride = 1;
    layer._M_flags |= zzs::mllee::CONV_CAUSAL;
    layer._M_group_size = 2;
    layer._M_groups = 2;
    layer._M_weights = { 1, 2, 3, 4 };
    layer._M_multipliers = { 32768, 16384 };
    const Executed executed = execute ({ layer }, { 1, 2, 3, 4 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 8, 18 });
}

TEST_CASE ("conv1d int4 staging buffer", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::CONV1D, 4, 4);
    layer._M_dtype = Dtype::I4;
    layer._M_head_dim = 4;
    layer._M_seq = 4;
    layer._M_kernel = 2;
    layer._M_stride = 1;
    layer._M_flags |= zzs::mllee::CONV_CAUSAL;
    layer._M_weights = { 3, -2 };
    const Executed executed = execute ({ layer }, { 1, 2, 3, 4 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ -2, -1, 0, 1 });
}

// The packlow CONV1D path re-reads a staged row from the 64-byte ring once per
// output position, so a window that starts past the ring wrap exercises the
// tail branch of the staged dot.  This pins it to the raw stream, which reads
// the same codes straight from Flash.
TEST_CASE ("conv1d packlow staged row matches raw across ring wrap", "[mllee][ops]")
{
    // Depthwise causal, kernel 5: 26 rows of 5 codes are decoded into one ring,
    // so the row window wraps and many positions start inside the wrapped tail.
    const std::vector<std::int8_t> weights
        = [] {
              std::vector<std::int8_t> codes (26u * 5u);
              for (std::size_t i = 0; i < codes.size (); ++i)
                  codes[i] = static_cast<std::int8_t> ((i / 5u) % 9u) - 4;
              return codes;
          } ();
    const std::vector<std::int8_t> input
        = [] {
              std::vector<std::int8_t> values (26u * 16u);
              for (std::size_t i = 0; i < values.size (); ++i)
                  values[i] = static_cast<std::int8_t> ((i % 11u) * 9 - 45);
              return values;
          } ();

    mllee_test::LayerSpec raw = external_layer (Op::CONV1D, 26 * 16, 26 * 16);
    raw._M_head_dim = 16;
    raw._M_seq = 16;
    raw._M_kernel = 5;
    raw._M_stride = 1;
    raw._M_flags |= zzs::mllee::CONV_DEPTHWISE | zzs::mllee::CONV_CAUSAL;
    raw._M_weights = weights;

    mllee_test::LayerSpec packed = raw;
    packed._M_use_packlow = true;

    // Only a real packlow blob exercises the ring reader; a fallback to raw
    // would silently pass without testing the staged path.
    const std::vector<uint8_t> blob = single_blob (packed);
    REQUIRE (blob[zzs::mllee::_S_model_header_size + 3]
             == static_cast<uint8_t> (Codec::PACKLOW));

    const Executed raw_executed = execute ({ raw }, input);
    const Executed packed_executed = execute ({ packed }, input);
    REQUIRE (raw_executed._M_ok);
    REQUIRE (packed_executed._M_ok);
    REQUIRE (raw_executed._M_values == packed_executed._M_values);
}

TEST_CASE ("layernorm matches integer reference", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::NORM, 4, 4);
    layer._M_scale16 = { 128, 128, 128, 128 };
    layer._M_bias = { 1, 0, 0, -1 };
    Executed executed = execute ({ layer }, { -3, -1, 1, 3 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 0, 0, 1, 1 });

    layer._M_scale16 = { 192, 192, 192, 192 };
    layer._M_bias = { 0, 0, 0, 0 };
    executed = execute ({ layer }, { -3, -1, 1, 3 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ -2, -1, 1, 2 });
}

TEST_CASE ("rmsnorm matches integer reference", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::NORM, 4, 4);
    layer._M_flags |= zzs::mllee::NORM_RMS;
    const Executed executed = execute ({ layer }, { -3, -1, 1, 3 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ -1, 0, 1, 2 });
}

TEST_CASE ("norm handles zero variance", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::NORM, 4, 4);
    layer._M_bias = { 7, -7, 0, 3 };
    const Executed executed = execute ({ layer }, { 5, 5, 5, 5 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 7, -7, 0, 3 });
}

TEST_CASE ("activation kernels", "[mllee][ops]")
{
    mllee_test::LayerSpec relu = external_layer (Op::ACTIVATION, 5, 5);
    relu._M_act = Act::RELU;
    Executed executed = execute ({ relu }, { -5, 0, 5, 127, -127 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 0, 0, 5, 127, 0 });

    mllee_test::LayerSpec sigmoid = external_layer (Op::ACTIVATION, 3, 3);
    sigmoid._M_act = Act::SIGMOID;
    executed = execute ({ sigmoid }, { -127, 0, 127 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 34, 64, 93 });

    mllee_test::LayerSpec tanh_layer = external_layer (Op::ACTIVATION, 3, 3);
    tanh_layer._M_act = Act::TANH;
    executed = execute ({ tanh_layer }, { -127, 0, 127 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ -99, 0, 99 });

    mllee_test::LayerSpec gelu = external_layer (Op::ACTIVATION, 3, 3);
    gelu._M_act = Act::GELU;
    executed = execute ({ gelu }, { -127, 0, 127 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values[1] == 0);
    REQUIRE (executed._M_values[2] >= 0);
}

TEST_CASE ("softmax normalizes to 127", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::SOFTMAX, 3, 3);
    const Executed executed = execute ({ layer }, { -1, 0, 1 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 37, 42, 48 });
}

TEST_CASE ("softmax clamps large distances", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::SOFTMAX, 2, 2);
    const Executed executed = execute ({ layer }, { -127, 127 });
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values[1] >= 126);
    REQUIRE (executed._M_values[0] <= 1);
    REQUIRE (executed._M_values[0] + executed._M_values[1] == 127);
}

TEST_CASE ("single-head attention mixes values", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::ATTENTION, 12, 4);
    layer._M_head_dim = 2;
    layer._M_seq = 2;
    const std::vector<std::int8_t> input = {
        1, 0, 0, 1,      // queries
        1, 0, 0, 1,      // keys
        100, 0, 0, 100,  // values
    };
    const Executed executed = execute ({ layer }, input);
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 65, 35, 35, 65 });
}

TEST_CASE ("attention with equal scores averages values", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::ATTENTION, 12, 4);
    layer._M_head_dim = 2;
    layer._M_seq = 2;
    const std::vector<std::int8_t> input = {
        0, 0, 0, 0,
        0, 0, 0, 0,
        100, 0, 0, 100,
    };
    const Executed executed = execute ({ layer }, input);
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{ 50, 50, 50, 50 });
}

TEST_CASE ("attention rejects head dimensions above eight", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::ATTENTION, 27, 9);
    layer._M_head_dim = 9;
    layer._M_seq = 1;
    const Executed executed = execute ({ layer }, std::vector<std::int8_t> (27, 0));
    REQUIRE (!executed._M_ok);
    REQUIRE (executed._M_status == Status::UNSUPPORTED);
}

/* Double-precision twin of the rotary attention path: the kernel quantizes the
   rotation and the exp table at 1/8 steps, so the test compares within a few
   codes rather than bit-exactly. */
std::vector<std::int8_t>
rope_reference (const std::vector<std::int8_t> &input, const uint32_t sequence,
                const uint32_t head_dim, const bool causal)
{
    const uint32_t stride = head_dim;
    const uint32_t pairs = head_dim / 2u;
    const auto rotate = [&] (const uint32_t base, const uint32_t position) {
        std::vector<double> row (head_dim, 0.0);
        for (uint32_t pair = 0; pair < pairs; ++pair)
            {
                const double theta
                    = static_cast<double> (position)
                      * std::pow (10000.0,
                                  -2.0 * static_cast<double> (pair)
                                      / static_cast<double> (head_dim));
                const double cosine = std::cos (theta);
                const double sine = std::sin (theta);
                const double x = input[base + position * stride + 2u * pair];
                const double y = input[base + position * stride + 2u * pair + 1u];
                row[2u * pair] = x * cosine - y * sine;
                row[2u * pair + 1u] = x * sine + y * cosine;
            }
        return row;
    };
    const uint32_t q_base = 0u;
    const uint32_t k_base = sequence * stride;
    const uint32_t v_base = 2u * sequence * stride;
    std::vector<std::int8_t> out (sequence * stride, 0);
    for (uint32_t query = 0; query < sequence; ++query)
        {
            const std::vector<double> q = rotate (q_base, query);
            const uint32_t limit = causal ? query + 1u : sequence;
            std::vector<double> scores (limit, 0.0);
            double maximum = -1e300;
            for (uint32_t key = 0; key < limit; ++key)
                {
                    const std::vector<double> k = rotate (k_base, key);
                    for (uint32_t d = 0; d < head_dim; ++d)
                        scores[key] += q[d] * k[d];
                    maximum = std::max (maximum, scores[key]);
                }
            double sum = 0.0;
            for (uint32_t key = 0; key < limit; ++key)
                {
                    scores[key] = std::exp ((scores[key] - maximum)
                                             / std::sqrt (static_cast<double> (head_dim)));
                    sum += scores[key];
                }
            for (uint32_t d = 0; d < head_dim; ++d)
                {
                    double mixed = 0.0;
                    for (uint32_t key = 0; key < limit; ++key)
                        mixed += scores[key]
                                 * input[v_base + key * stride + d];
                    mixed /= sum;
                    const int32_t code = std::lround (mixed);
                    out[query * stride + d] = static_cast<std::int8_t> (
                        code < -127 ? -127 : (code > 127 ? 127 : code));
                }
        }
    return out;
}

TEST_CASE ("rotary attention rotates queries and keys per position", "[mllee][ops]")
{
    const std::vector<std::int8_t> input = {
        60, 0,   0, 60,    /* queries */
        60, 0,   60, 60,   /* keys */
        100, 0,  0, 100,   /* values */
    };
    mllee_test::LayerSpec plain = external_layer (Op::ATTENTION, 12, 4);
    plain._M_head_dim = 2;
    plain._M_seq = 2;
    const Executed plain_out = execute ({ plain }, input);
    REQUIRE (plain_out._M_ok);

    mllee_test::LayerSpec rotated = plain;
    rotated._M_flags |= zzs::mllee::ATTN_ROPE;
    const Executed rotated_out = execute ({ rotated }, input);
    REQUIRE (rotated_out._M_ok);
    REQUIRE (rotated_out._M_values != plain_out._M_values);

    const std::vector<std::int8_t> want
        = rope_reference (input, 2u, 2u, false);
    REQUIRE (rotated_out._M_values.size () == want.size ());
    for (std::size_t i = 0; i < want.size (); ++i)
        CHECK (std::abs (rotated_out._M_values[i] - want[i]) <= 3);

    /* A causal row at position 0 can only see its own value, rotation or not. */
    mllee_test::LayerSpec causal = rotated;
    causal._M_flags |= zzs::mllee::ATTN_CAUSAL;
    const Executed causal_out = execute ({ causal }, input);
    REQUIRE (causal_out._M_ok);
    REQUIRE (causal_out._M_values[0] == want[0]);
    REQUIRE (causal_out._M_values[1] == want[1]);
}

TEST_CASE ("rotary attention rejects an odd head dimension", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::ATTENTION, 9, 3);
    layer._M_head_dim = 3;
    layer._M_seq = 1;
    layer._M_flags |= zzs::mllee::ATTN_ROPE;
    const Executed executed = execute ({ layer }, std::vector<std::int8_t> (9, 0));
    REQUIRE (!executed._M_ok);
    REQUIRE (executed._M_status == Status::BAD_LAYOUT);
}

TEST_CASE ("residual norm normalizes the sum of both operands", "[mllee][ops]")
{
    /* Layer 0 copies the external row into the arena; layer 1 normalizes the
       external row plus that residual, which must equal a plain norm of the
       element-wise sum. */
    mllee_test::LayerSpec copy = external_layer (Op::ACTIVATION, 4, 4);
    copy._M_out_off = 0;
    copy._M_out_len = 4;

    mllee_test::LayerSpec residual_norm = external_layer (Op::NORM, 4, 4);
    residual_norm._M_out_off = 4;
    residual_norm._M_out_len = 4;
    residual_norm._M_flags |= zzs::mllee::NORM_RESIDUAL;
    residual_norm._M_in2_off = 0;
    residual_norm._M_in2_len = 4;

    const Executed residual = execute ({ copy, residual_norm }, { -3, -1, 1, 3 });
    REQUIRE (residual._M_ok);

    mllee_test::LayerSpec plain = external_layer (Op::NORM, 4, 4);
    const Executed summed = execute ({ plain }, { -6, -2, 2, 6 });
    REQUIRE (summed._M_ok);
    REQUIRE (residual._M_values == summed._M_values);

    /* The same composition with RMSNorm, so the residual feed is exercised on
       both variance paths. */
    mllee_test::LayerSpec rms_residual = residual_norm;
    rms_residual._M_flags |= zzs::mllee::NORM_RMS;
    const Executed rms = execute ({ copy, rms_residual }, { -3, -1, 1, 3 });
    REQUIRE (rms._M_ok);
    mllee_test::LayerSpec rms_plain = plain;
    rms_plain._M_flags |= zzs::mllee::NORM_RMS;
    const Executed rms_summed = execute ({ rms_plain }, { -6, -2, 2, 6 });
    REQUIRE (rms_summed._M_ok);
    REQUIRE (rms._M_values == rms_summed._M_values);
}

TEST_CASE ("packlow path matches raw weights", "[mllee][ops]")
{
    mllee_test::LayerSpec raw = external_layer (Op::LINEAR, 16, 4);
    raw._M_weights.assign (64, 7);
    raw._M_bias.assign (4, 0);

    mllee_test::LayerSpec compressed = raw;
    compressed._M_use_packlow = true;

    const std::vector<std::int8_t> input (16, 3);
    const Executed raw_executed = execute ({ raw }, input);
    const Executed packed_executed = execute ({ compressed }, input);
    REQUIRE (raw_executed._M_ok);
    REQUIRE (packed_executed._M_ok);
    REQUIRE (raw_executed._M_values == packed_executed._M_values);

    const std::vector<uint8_t> blob = single_blob (compressed);
    REQUIRE (blob[zzs::mllee::_S_model_header_size + 3]
             == static_cast<uint8_t> (Codec::PACKLOW));
}

TEST_CASE ("core rejects malformed blobs", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 4, 2);
    layer._M_weights = { 1, 0, 0, 1, 0, 1, 1, 0 };
    const std::vector<uint8_t> blob = single_blob (layer);
    const std::vector<std::int8_t> input = { 1, 2, 3, 4 };

    REQUIRE (status_of (blob, input) == Status::OK);

    std::vector<uint8_t> corrupted = blob;
    corrupted[0] = 'X';
    REQUIRE (status_of (corrupted, input) == Status::BAD_MAGIC);

    corrupted = blob;
    corrupted[4] = 2;
    REQUIRE (status_of (corrupted, input) == Status::BAD_VERSION);

    corrupted = blob;
    corrupted[9] ^= 0x01u;
    REQUIRE (status_of (corrupted, input) == Status::CORRUPT_STREAM);

    const std::vector<uint8_t> truncated (blob.begin (), blob.begin () + 20);
    REQUIRE (status_of (truncated, input) == Status::TRUNCATED);

    corrupted = blob;
    const uint16_t oversized = zzs::mllee::_S_arena_bytes
                               + static_cast<uint16_t> (32);
    corrupted[22] = static_cast<uint8_t> (oversized & 0xFFu);
    corrupted[23] = static_cast<uint8_t> ((oversized >> 8) & 0xFFu);
    const uint16_t crc = zzs::mllee::_S_crc16 (corrupted.data (), 28);
    corrupted[28] = static_cast<uint8_t> (crc & 0xFFu);
    corrupted[29] = static_cast<uint8_t> ((crc >> 8) & 0xFFu);
    REQUIRE (status_of (corrupted, input) == Status::ARENA_OVERFLOW);

    corrupted = blob;
    corrupted[zzs::mllee::_S_model_header_size + 20] = 0xFFu;
    corrupted[zzs::mllee::_S_model_header_size + 21] = 0xFFu;
    REQUIRE (status_of (corrupted, input) == Status::BAD_LAYOUT);
}

TEST_CASE ("core rejects bad input and arena layouts", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 4, 2);
    layer._M_weights = { 1, 0, 0, 1, 0, 1, 1, 0 };
    const std::vector<uint8_t> blob = single_blob (layer);

    zzs::mllee::Core core;
    auto result = core.run (blob.data (), static_cast<uint32_t> (blob.size ()), nullptr, 0);
    REQUIRE (result.has_error ());
    REQUIRE (result.error ().kind () == Status::OUT_OF_RANGE);

    const std::vector<std::int8_t> short_input = { 1, 2, 3 };
    result = core.run (blob.data (), static_cast<uint32_t> (blob.size ()), short_input.data (),
                       static_cast<uint32_t> (short_input.size ()));
    REQUIRE (result.has_error ());
    REQUIRE (result.error ().kind () == Status::OUT_OF_RANGE);

    mllee_test::LayerSpec overflow = layer;
    overflow._M_out_off = zzs::mllee::_S_arena_bytes - 8;
    overflow._M_out_len = 16;
    REQUIRE (status_of (single_blob (overflow), { 1, 2, 3, 4 }) == Status::ARENA_OVERFLOW);
}

TEST_CASE ("core rejects unused weight bytes", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 4, 2);
    layer._M_weights = { 1, 0, 0, 1, 0, 1, 1, 0, 9 };
    const Executed executed = execute ({ layer }, { 1, 2, 3, 4 });
    REQUIRE (!executed._M_ok);
    REQUIRE (executed._M_status == Status::BAD_LAYOUT);
}

TEST_CASE ("core reports packlow CRC failure", "[mllee][ops]")
{
    mllee_test::LayerSpec layer = external_layer (Op::LINEAR, 16, 4);
    layer._M_weights.assign (64, 7);
    layer._M_use_packlow = true;
    std::vector<uint8_t> blob = single_blob (layer);
    REQUIRE (blob[zzs::mllee::_S_model_header_size + 3]
             == static_cast<uint8_t> (Codec::PACKLOW));
    const uint32_t weight_off =
        zzs::mllee::_S_u32 (blob.data () + zzs::mllee::_S_model_header_size + 16);
    blob[weight_off + 10] ^= 0x01u;
    REQUIRE (status_of (blob, std::vector<std::int8_t> (16, 1)) == Status::CORRUPT_STREAM);
}

TEST_CASE ("pipeline runs end to end", "[mllee][ops][golden]")
{
    const std::vector<mllee_test::LayerSpec> layers = mllee_test::make_pipeline ();
    const std::vector<uint8_t> blob = mllee_test::BlobBuilder (layers).build (
        zzs::mllee::_S_arena_bytes);
    const std::vector<std::int8_t> input = { 1, 2, 3, 4, 5, 6, 7, 8 };

    zzs::mllee::Core core;
    auto result = core.run (blob.data (), static_cast<uint32_t> (blob.size ()), input.data (),
                            static_cast<uint32_t> (input.size ()));
    REQUIRE (result.has_value ());
    REQUIRE (result.value ()._M_len == 4);

    const std::int8_t *const arena = core.arena ();
    const std::vector<std::int8_t> expected_conv = { -1, -2, -2, -2, -2, -2, -2, -2,
                                                     1,  3,  6,  9,  12, 15, 18, 21 };
    REQUIRE (std::vector<std::int8_t> (arena, arena + 16) == expected_conv);

    const std::vector<std::int8_t> attention (arena + 52, arena + 64);
    REQUIRE (std::adjacent_find (attention.begin (), attention.end (),
                                 std::not_equal_to<std::int8_t>{})
             != attention.end ());

    int sum = 0;
    for (std::int8_t value : std::vector<std::int8_t> (result.value ()._M_data,
                                                       result.value ()._M_data + 4))
        sum += value;
    REQUIRE (sum >= 124);
    REQUIRE (sum <= 130);

    const std::vector<std::int8_t> first (result.value ()._M_data,
                                          result.value ()._M_data + result.value ()._M_len);
    result = core.run (blob.data (), static_cast<uint32_t> (blob.size ()), input.data (),
                       static_cast<uint32_t> (input.size ()));
    REQUIRE (result.has_value ());
    REQUIRE (std::vector<std::int8_t> (result.value ()._M_data,
                                       result.value ()._M_data + result.value ()._M_len)
             == first);
}
