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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

#pragma once

#include <cstdint>

/**
 * @brief Compile-time model topology for mllee-native.
 *
 * Captures all model dimensions as static constexpr members so every module
 * (model, inference, training pipeline) derives its sizes from a single
 * template parameter instead of hardcoded `inline constexpr` constants.
 *
 * Two topologies are supported:
 *
 *   - MLPConfig: the original flat MLP, LINEAR(190->16) + RELU + LINEAR(16->3)
 *   - ConvAttnConfig: Tier 2, CONV1D(38->8,k=3,causal) -> LINEAR(40->120,QKV)
 *                     -> ATTENTION(5,4,2) -> GLOBAL_AVG_POOL -> LINEAR(8->3)
 *                     -> SOFTMAX
 *
 * The ConvAttnConfig default fits the 444-byte activation arena: conv output
 * (40) + QKV projection (120) + attention output (40) + pooled (8) + output
 * (3) = 211 int8 codes.
 * @{
 */
namespace zzs::mllee_native {

/** @brief Original flat MLP topology: LINEAR(190->16) + RELU + LINEAR(16->3). */
template <uint32_t InputDim, uint32_t HiddenDim, uint32_t OutputDim, uint32_t SeqLen, uint32_t FeatureCount>
struct ModelConfig {
        static constexpr uint32_t input_dim = InputDim;
        static constexpr uint32_t hidden_dim = HiddenDim;
        static constexpr uint32_t output_dim = OutputDim;
        static constexpr uint32_t seq_len = SeqLen;
        static constexpr uint32_t feature_count = FeatureCount;

        /** @brief Flattened input size: seq_len * feature_count. */
        static constexpr uint32_t flat_input = seq_len * feature_count;

        /** @brief Input size of the hidden layer (equals flat_input). */
        static constexpr uint32_t hidden_input = flat_input;

        /** @brief Output size of the hidden layer (equals hidden_dim). */
        static constexpr uint32_t hidden_output = hidden_dim;
};

/** @brief Conv1D + Attention topology.
 *
 *  Layer chain:
 *    CONV1D:   [seq, feature_count] external -> [seq, conv_channels], causal
 *    LINEAR:   [seq*conv_channels] -> [seq*heads*head_dim*3] (QKV projection)
 *    ATTENTION: [seq, heads*head_dim*3] -> [seq, heads*head_dim]
 *    GLOBAL_AVG_POOL: [seq, conv_channels] -> [conv_channels]
 *    LINEAR:   [conv_channels] -> [output_dim]
 *    SOFTMAX:  [output_dim] -> [output_dim] (in-place)
 *
 *  The inference ATTENTION op reads Q, K, V as three consecutive blocks from its
 *  arena input (seq * head_dim * heads * 3 codes), so the preceding LINEAR
 *  projects the conv output to QKV.  ATTENTION itself carries no weights.
 */
template <uint32_t InputDim, uint32_t SeqLen, uint32_t FeatureCount, uint32_t ConvChannels, uint32_t ConvKernel,
          uint32_t AttnHeadDim, uint32_t AttnHeads, uint32_t OutputDim>
struct ConvAttnConfig {
        static constexpr uint32_t input_dim = InputDim;
        static constexpr uint32_t seq_len = SeqLen;
        static constexpr uint32_t feature_count = FeatureCount;
        static constexpr uint32_t conv_channels = ConvChannels;
        static constexpr uint32_t conv_kernel = ConvKernel;
        static constexpr uint32_t attn_head_dim = AttnHeadDim;
        static constexpr uint32_t attn_heads = AttnHeads;
        static constexpr uint32_t output_dim = OutputDim;

        /** @brief Flattened input size: seq_len * feature_count. */
        static constexpr uint32_t flat_input = seq_len * feature_count;

        /** @brief Conv1D output codes: seq_len * conv_channels. */
        static constexpr uint32_t conv_output = seq_len * conv_channels;

        /** @brief QKV projection output codes: seq_len * attn_heads * attn_head_dim * 3. */
        static constexpr uint32_t qkv_output = seq_len * attn_heads * attn_head_dim * 3u;

        /** @brief Attention output codes: seq_len * attn_heads * attn_head_dim. */
        static constexpr uint32_t attn_output = seq_len * attn_heads * attn_head_dim;

        /** @brief Global average pool output codes: conv_channels. */
        static constexpr uint32_t pooled = conv_channels;

        /** @brief MLP head input size (equals pooled). */
        static constexpr uint32_t head_input = pooled;
};

/** @brief Default model topology: LINEAR(190->3) + SOFTMAX(3->3). */
using DefaultConfig = ModelConfig<190u, 16u, 3u, 5u, 38u>;

/** @brief Default Tier 2 topology: CONV1D + ATTENTION head. */
using ConvAttnDefaultConfig = ConvAttnConfig<190u, 5u, 38u, 8u, 3u, 4u, 2u, 3u>;

/** @} */
} // namespace zzs::mllee_native
