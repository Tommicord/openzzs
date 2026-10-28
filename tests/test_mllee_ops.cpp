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

TEST_CASE ("readme pipeline runs end to end", "[mllee][ops][golden]")
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
