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

#pragma once

#include "mllee/status.hpp"

#include <cstdint>

namespace zzs::mllee::onnx
{
/**
 * @brief Resident RAM budget of the streaming ONNX reader.
 *
 * The loader object is partitioned into four fixed regions, all of which are
 * part of the caller-visible object:
 *
 *   Stream       40 bytes (28 on a 32-bit target)  source cursor + cached chunk
 *   ModelInfo    32 bytes                         ONNX model header
 *   NodeInfo     28 bytes                         active NodeProto header
 *   TensorInfo   28 bytes                         active TensorProto header
 *   ---------------------------------------------------------------
 *   Loader      128 bytes on a 64-bit host, 116 on a 32-bit target
 *
 * No writable module static, heap allocation or hidden scratch buffer exists.
 * Strings are hashed while they stream past; only headers are retained.  The
 * caller owns every byte returned by the chunk callback, including the raw
 * tensor payloads, and can copy selected ranges with Loader::copy().
 */
inline constexpr uint32_t _S_ram_budget_bytes = 128;
/** @brief Share of the budget occupied by the model header ("static memory"). */
inline constexpr uint32_t _S_static_budget_bytes = 32;
inline constexpr uint32_t _S_stream_budget_bytes = 40;
inline constexpr uint32_t _S_model_budget_bytes = 32;
inline constexpr uint32_t _S_node_budget_bytes = 28;
inline constexpr uint32_t _S_tensor_budget_bytes = 28;

/** @brief Largest model accepted by the 32-bit absolute-offset cursor. */
inline constexpr uint32_t _S_max_model_bytes = 0xFFFFFFFFu;
/** @brief Largest protobuf field number (29 bits). */
inline constexpr uint32_t _S_max_field_number = (1u << 29) - 1u;
/** @brief Bound on nested messages the structural parser will follow. */
inline constexpr uint8_t _S_max_message_depth = 8;
/** @brief Bound on a single string hash read; longer strings are rejected. */
inline constexpr uint32_t _S_max_string_bytes = 0x01000000u;
/** @brief Bound on the rank retained by TensorInfo. */
inline constexpr uint8_t _S_max_tensor_rank = 64;
/** @brief Bound on decoded elements before an overflow flag is raised. */
inline constexpr uint64_t _S_max_tensor_elements = 0xFFFFFFFFull;
/** @brief Highest ONNX IR version understood by this reader. */
inline constexpr uint8_t _S_max_ir_version = 14;
/** @brief Highest default-domain opset understood by this reader. */
inline constexpr uint8_t _S_max_opset_version = 25;
/** @brief A callback chunk should return at least this many bytes when able. */
inline constexpr uint32_t _S_chunk_hint_bytes = 64;

/**
 * @brief Flags accepted by Loader::begin().
 *
 * The default is strict: unknown operator domains, subgraphs and sparse
 * initializers are reported as unsupported through the event stream.  Tolerant
 * mode skips them instead, which is what a firmware update path wants when it
 * only needs the dense part of a model.
 */
enum LoaderFlag : uint32_t
{
    LOADER_STRICT = 1u << 0,
    LOADER_ALLOW_SUBGRAPHS = 1u << 1,
    LOADER_ALLOW_SPARSE = 1u << 2,
    LOADER_ALLOW_EXTERNAL = 1u << 3,
    LOADER_ALLOW_TRAILING_BYTES = 1u << 4,
};

[[nodiscard]] constexpr uint32_t
loader_default_flags () noexcept
{
    return LOADER_ALLOW_TRAILING_BYTES;
}
} // namespace zzs::mllee::onnx
