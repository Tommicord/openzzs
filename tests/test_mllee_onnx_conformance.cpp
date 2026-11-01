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
#include <cstdint>
#include <utility>
#include <vector>

namespace {
using zzs::mllee::Act;
using zzs::mllee::Codec;
using zzs::mllee::Dtype;
using zzs::mllee::Op;
using zzs::mllee::Status;

struct Executed {
    bool _M_ok = false;
    Status _M_status = Status::OK;
    std::vector<std::int8_t> _M_values;
};

Executed
execute (std::vector<mllee_test::LayerSpec> layers, const std::vector<std::int8_t> &input,
         const uint16_t arena_bytes = zzs::mllee::_S_arena_bytes) {
    const std::vector<uint8_t> blob = mllee_test::BlobBuilder (std::move (layers)).build (arena_bytes);
    zzs::mllee::Core core;
    auto result = core.run (blob.data (), static_cast<uint32_t> (blob.size ()), input.data (),
                            static_cast<uint32_t> (input.size ()));
    Executed executed;
    if (result.has_value ()) {
        executed._M_ok = true;
        executed._M_values.assign (result.value ()._M_data, result.value ()._M_data + result.value ()._M_len);
    } else
        executed._M_status = result.error ().kind ();
    return executed;
}

[[nodiscard]] Status
status_of (const std::vector<uint8_t> &blob, const std::vector<std::int8_t> &input) noexcept {
    zzs::mllee::Core core;
    auto result = core.run (blob.data (), static_cast<uint32_t> (blob.size ()), input.data (),
                            static_cast<uint32_t> (input.size ()));
    return result.has_value () ? Status::OK : result.error ().kind ();
}

/** @brief A single external-input LINEAR layer, int8, identity requant. */
mllee_test::LayerSpec
external_linear (const uint16_t in_dim, const uint16_t out_dim) noexcept {
    mllee_test::LayerSpec layer;
    layer._M_op = Op::LINEAR;
    layer._M_act = Act::NONE;
    layer._M_dtype = Dtype::I8;
    layer._M_codec = Codec::RAW;
    layer._M_in_dim = in_dim;
    layer._M_out_dim = out_dim;
    layer._M_kernel = 1;
    layer._M_stride = 1;
    layer._M_in_off = zzs::mllee::_S_external_input;
    layer._M_out_off = 0;
    layer._M_in_len = in_dim;
    layer._M_out_len = out_dim;
    layer._M_flags = zzs::mllee::INPUT_EXTERNAL;
    layer._M_requant_mult = 32768;
    return layer;
}

} // namespace

TEST_CASE ("onnx conformance: valid linear model runs and matches expected output", "[mllee][onnx][conformance]") {
    /* in_dim = 2, out_dim = 2, input = {2, 3}.
     * Row 0 weights {1, 2}: 1*2 + 2*3 = 8, requant(8, 32768) = 8, bias 0 -> 8.
     * Row 1 weights {3, 4}: 3*2 + 4*3 = 18, requant(18, 32768) = 18, bias 0 -> 18. */
    mllee_test::LayerSpec layer = external_linear (2, 2);
    layer._M_weights = {1, 2, 3, 4};
    layer._M_bias = {0, 0};
    const Executed executed = execute ({layer}, {2, 3});
    REQUIRE (executed._M_ok);
    REQUIRE (executed._M_values == std::vector<std::int8_t>{8, 18});
}

TEST_CASE ("onnx conformance: bad magic is rejected", "[mllee][onnx][conformance]") {
    mllee_test::LayerSpec layer = external_linear (2, 2);
    layer._M_weights = {1, 2, 3, 4};
    layer._M_bias = {0, 0};
    std::vector<uint8_t> blob = mllee_test::BlobBuilder ({layer}).build (zzs::mllee::_S_arena_bytes);
    const std::vector<std::int8_t> input = {2, 3};

    REQUIRE (status_of (blob, input) == Status::OK);

    blob[0] = 'X';
    REQUIRE (status_of (blob, input) == Status::BAD_MAGIC);
}

TEST_CASE ("onnx conformance: bad version is rejected", "[mllee][onnx][conformance]") {
    mllee_test::LayerSpec layer = external_linear (2, 2);
    layer._M_weights = {1, 2, 3, 4};
    layer._M_bias = {0, 0};
    std::vector<uint8_t> blob = mllee_test::BlobBuilder ({layer}).build (zzs::mllee::_S_arena_bytes);
    const std::vector<std::int8_t> input = {2, 3};

    REQUIRE (status_of (blob, input) == Status::OK);

    blob[4] = 2;
    REQUIRE (status_of (blob, input) == Status::BAD_VERSION);
}

TEST_CASE ("onnx conformance: bad CRC is rejected", "[mllee][onnx][conformance]") {
    mllee_test::LayerSpec layer = external_linear (2, 2);
    layer._M_weights = {1, 2, 3, 4};
    layer._M_bias = {0, 0};
    std::vector<uint8_t> blob = mllee_test::BlobBuilder ({layer}).build (zzs::mllee::_S_arena_bytes);
    const std::vector<std::int8_t> input = {2, 3};

    REQUIRE (status_of (blob, input) == Status::OK);

    /* Flip a header byte (flags) without recomputing the CRC-16 over bytes
     * [0, 28), so parse_header detects the mismatch. */
    blob[5] ^= 0x01u;
    REQUIRE (status_of (blob, input) == Status::CORRUPT_STREAM);
}

TEST_CASE ("onnx conformance: truncated input is rejected", "[mllee][onnx][conformance]") {
    mllee_test::LayerSpec layer = external_linear (2, 2);
    layer._M_weights = {1, 2, 3, 4};
    layer._M_bias = {0, 0};
    const std::vector<uint8_t> blob = mllee_test::BlobBuilder ({layer}).build (zzs::mllee::_S_arena_bytes);
    const std::vector<std::int8_t> input = {2, 3};

    REQUIRE (status_of (blob, input) == Status::OK);

    const std::vector<uint8_t> truncated (blob.begin (), blob.begin () + 16);
    REQUIRE (status_of (truncated, input) == Status::TRUNCATED);
}

TEST_CASE ("onnx conformance: bad layout is rejected", "[mllee][onnx][conformance]") {
    mllee_test::LayerSpec layer = external_linear (2, 2);
    layer._M_weights = {1, 2, 3, 4};
    layer._M_bias = {0, 0};
    std::vector<uint8_t> blob = mllee_test::BlobBuilder ({layer}).build (zzs::mllee::_S_arena_bytes);
    const std::vector<std::int8_t> input = {2, 3};

    REQUIRE (status_of (blob, input) == Status::OK);

    /* Header byte 7 must equal the header size (32).  Corrupting it trips
     * BAD_LAYOUT before the CRC check, so no CRC fix-up is needed. */
    blob[7] = 0;
    REQUIRE (status_of (blob, input) == Status::BAD_LAYOUT);
}
