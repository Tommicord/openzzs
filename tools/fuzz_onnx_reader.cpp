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

/** @file
 * @brief libFuzzer target for the native ONNX / MLEE blob reader.
 *
 * Feeds the raw fuzz bytes to zzs::mllee::Core::run(), which parses the blob
 * and returns a RunResult (an error-or-value, never an exception).  Every
 * status is handled, so malformed input never crashes the process.  A dummy
 * input buffer is supplied because some layers read external input.
 */

#include "mllee/mllee.hpp"

#include <cstdint>

namespace {
// Dummy input buffer handed to Core::run() on every invocation.  Kept small so
// the fuzzer's RSS stays flat; the contents are irrelevant because a malformed
// model fails before any layer consumes external input.
alignas (64) int8_t _S_dummy_input[64]{};
} // namespace

extern "C" int
LLVMFuzzerTestOneInput (const uint8_t *data, const size_t size) noexcept {
    if (data == nullptr || size == 0u) {
        return 0;
    }

    // Clamp to the uint32_t model length Core::run() accepts.  The parser
    // validates every span internally, so an oversized length is rejected
    // cleanly rather than read out of bounds.
    const uint32_t model_len = size > static_cast<size_t> (UINT32_MAX) ? UINT32_MAX : static_cast<uint32_t> (size);

    zzs::mllee::Core core;
    const zzs::mllee::Core::RunResult result = core.run (data, model_len, _S_dummy_input, sizeof (_S_dummy_input));

    // Consume the result so the call is not optimized away.  All paths are
    // noexcept: a false result carries an Error::Kind, never an exception.
    if (result) {
        const zzs::mllee::Core::Output &output = result.value ();
        volatile const int8_t *const sink = output._M_data;
        volatile const uint16_t len = output._M_len;
        (void) sink;
        (void) len;
    } else {
        const zzs::mllee::Status kind = result.error ().kind ();
        (void) kind;
    }

    return 0;
}
