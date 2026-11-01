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

#include "mllee-native/model_config.hpp"
#include "mllee/core_format.hpp"
#include "mllee/core_quant.hpp"
#include "mllee/core_status.hpp"

#include <cstdint>

/**
 * @brief Model export: serialize trained weights into the native `"MLEE"` blob
 *        format consumed by `zzs::mllee::Core::run()`.
 *
 * Builds a model blob in a caller-owned buffer.  The blob layout is:
 *
 *   [0..31]     Header (32 bytes, magic "MLEE" + CRC-16)
 *   [32..223]   Layer descriptors (4 x 48 bytes: LINEAR + ACTIVATION + LINEAR + SOFTMAX)
 *   [224..3311] LINEAR weights  (190*16 + 16*3 = 3088 int8 codes, RAW codec)
 *   [3312..3343] LINEAR bias    (16 + 3 = 19 int8 codes -> 38 bytes as int16)
 *
 * Total: 3350 bytes for the default config, all in .rodata at runtime.  No heap,
 * no float, no ONNX dependency (the ONNX layer in mllee/ is read-only; we write
 * the native blob directly).
 *
 * Model topology (default):
 *   LINEAR:  190 inputs (5 windows x 38 features) -> 16 hidden, act=NONE
 *   ACTIVATION: RELU, in-place on 16 hidden codes
 *   LINEAR:  16 hidden -> 3 outputs, act=NONE
 *   SOFTMAX: 3 inputs -> 3 outputs (P_normal, P_outage, P_transition)
 * @{
 */
namespace zzs::mllee_native::export_ {

using zzs::mllee::_S_crc16;
using zzs::mllee::_S_i8_code;
using zzs::mllee::_S_layer_desc_size;
using zzs::mllee::_S_model_header_size;
using zzs::mllee::_S_model_version;
using zzs::mllee::Act;
using zzs::mllee::ACT_IN_PLACE;
using zzs::mllee::Codec;
using zzs::mllee::Dtype;
using zzs::mllee::Layer;
using zzs::mllee::LayerFlag;
using zzs::mllee::Op;
using zzs::mllee::Status;
using zzs::mllee::status_ok;

using zzs::mllee_native::ConvAttnConfig;
using zzs::mllee_native::DefaultConfig;
using zzs::mllee_native::ModelConfig;

/** @brief Input dimension: 5 windows x 38 features = 190. */
inline constexpr uint32_t kInputDim = DefaultConfig::input_dim;
/** @brief Hidden dimension: 16 hidden units. */
inline constexpr uint32_t kHiddenDim = DefaultConfig::hidden_dim;
/** @brief Hidden layer input size (equals flat_input). */
inline constexpr uint32_t kHiddenInput = DefaultConfig::hidden_input;
/** @brief Output dimension: 3 classes. */
inline constexpr uint32_t kOutputDim = DefaultConfig::output_dim;

/** @brief Blob size: header + 4 layers + weights (3088) + bias (19 int16 = 38 bytes). */
inline constexpr uint32_t kBlobSize = _S_model_header_size + 4u * _S_layer_desc_size + kHiddenInput * kHiddenDim
                                      + kHiddenDim * kOutputDim + (kHiddenDim + kOutputDim) * 2u;

/** @brief Offsets within the MLP blob. */
namespace offset {
inline constexpr uint32_t header = 0u;
inline constexpr uint32_t layer_table = _S_model_header_size; /* 32 */
/* Start of all weights: hidden weights first, then output weights. */
inline constexpr uint32_t weights = layer_table + 4u * _S_layer_desc_size;             /* 224 */
inline constexpr uint32_t hidden_weights = weights;                                    /* 224 */
inline constexpr uint32_t output_weights = hidden_weights + kHiddenInput * kHiddenDim; /* 224 + 3040 = 3264 */
/* Start of all bias: hidden bias first (int16), then output bias (int16). */
inline constexpr uint32_t bias = output_weights + kHiddenDim * kOutputDim; /* 3264 + 48 = 3312 */
inline constexpr uint32_t hidden_bias = bias;                              /* 3312 */
inline constexpr uint32_t output_bias = hidden_bias + kHiddenDim * 2u;     /* 3312 + 32 = 3344 */
} // namespace offset

// ---------------------------------------------------------------------------
// Tier 2: CONV1D + Attention over a 5-window sequence.
//
//   Layer chain:
//     CONV1D(38->8, kernel=3, causal): arena[0..40]
//     LINEAR(40->120) QKV projection:  arena[40..160]
//     ATTENTION(seq=5, head_dim=4, heads=2): arena[160..200]
//     GLOBAL_AVG_POOL:                 arena[200..208]
//     LINEAR(8->3):                    arena[208..211]
//     SOFTMAX:                         arena[208..211] (in-place)
//
// Total arena: 211 bytes <= 444.
// ---------------------------------------------------------------------------
namespace conv_attn {

using zzs::mllee_native::ConvAttnDefaultConfig;

inline constexpr uint32_t kSeqLen = ConvAttnDefaultConfig::seq_len;               /* 5 */
inline constexpr uint32_t kFeatureCount = ConvAttnDefaultConfig::feature_count;   /* 38 */
inline constexpr uint32_t kConvChannels = ConvAttnDefaultConfig::conv_channels;   /* 8 */
inline constexpr uint32_t kConvKernel = ConvAttnDefaultConfig::conv_kernel;       /* 3 */
inline constexpr uint32_t kAttnHeadDim = ConvAttnDefaultConfig::attn_head_dim;    /* 4 */
inline constexpr uint32_t kAttnHeads = ConvAttnDefaultConfig::attn_heads;         /* 2 */
inline constexpr uint32_t kOutputDimConvAttn = ConvAttnDefaultConfig::output_dim; /* 3 */

inline constexpr uint32_t kConvOutput = kSeqLen * kConvChannels;                 /* 40 */
inline constexpr uint32_t kQKVOutput = kSeqLen * kAttnHeads * kAttnHeadDim * 3u; /* 120 */
inline constexpr uint32_t kAttnOutput = kSeqLen * kAttnHeads * kAttnHeadDim;     /* 40 */
inline constexpr uint32_t kPooled = kConvChannels;                               /* 8 */

/** @brief Blob size: header + 6 layers + conv weights + QKV weights + output weights + biases. */
inline constexpr uint32_t kConvAttnBlobSize =
    _S_model_header_size + 6u * _S_layer_desc_size +
    /* CONV1D weights: output_channels * input_channels * kernel = 8*38*3 = 912 */
    kConvChannels * kFeatureCount * kConvKernel +
    /* QKV LINEAR weights: 40*120 = 4800 */
    kConvOutput * kQKVOutput +
    /* Output LINEAR weights: 8*3 = 24 */
    kPooled * kOutputDimConvAttn +
    /* Biases as int16: conv 8 + QKV 120 + output 3 = 131 * 2 = 262 bytes */
    (kConvChannels + kQKVOutput + kOutputDimConvAttn) * 2u;

/** @brief Offsets within the conv-attn blob. */
namespace offset {
inline constexpr uint32_t header = 0u;
inline constexpr uint32_t layer_table = _S_model_header_size; /* 32 */
/* Start of all weights after the 6 layer descriptors. */
inline constexpr uint32_t weights = layer_table + 6u * _S_layer_desc_size; /* 32 + 288 = 320 */

/* CONV1D: 8 output channels x 38 input channels x 3 kernel = 912 codes. */
inline constexpr uint32_t conv_weights = weights;                                                 /* 320 */
inline constexpr uint32_t conv_bias = conv_weights + kConvChannels * kFeatureCount * kConvKernel; /* 320 + 912 = 1232 */

/* QKV LINEAR: 40*120 = 4800 codes. */
inline constexpr uint32_t qkv_weights = conv_bias + kConvChannels * 2u;      /* 1232 + 16 = 1248 */
inline constexpr uint32_t qkv_bias = qkv_weights + kConvOutput * kQKVOutput; /* 1248 + 4800 = 6048 */

/* Output LINEAR: 8*3 = 24 codes. */
inline constexpr uint32_t out_weights = qkv_bias + kQKVOutput * 2u;              /* 6048 + 240 = 6288 */
inline constexpr uint32_t out_bias = out_weights + kPooled * kOutputDimConvAttn; /* 6288 + 24 = 6312 */
} // namespace offset

/* Sanity checks. */
static_assert (kConvAttnBlobSize == offset::out_bias + kOutputDimConvAttn * 2u,
               "conv-attn blob size must equal the end of the output bias");
static_assert (kConvAttnBlobSize <= 8192, "conv-attn blob exceeds the flash budget");

} // namespace conv_attn

// ---------------------------------------------------------------------------
// Tier 3 Part A: Multi-task model — shared Conv1D + Attention encoder with two
// regression heads after the shared encoder.
//
//   Layer chain:
//     CONV1D(38->8, kernel=3, causal):       arena[0..40]
//     LINEAR(40->120) QKV projection:        arena[40..160]
//     ATTENTION(seq=5, head_dim=4, heads=2): arena[160..200]
//     GLOBAL_AVG_POOL:                       arena[200..208]
//     LINEAR(8->3) head 1 (classification):  arena[208..211]
//     SOFTMAX head 1:                        arena[208..211] (in-place)
//     LINEAR(8->1) head 2 (confidence):      arena[211..212]
//     LINEAR(8->1) head 3 (time-to-outage):  arena[212..213]
//
// Total arena: 213 bytes <= 444.
// ---------------------------------------------------------------------------
namespace multitask {

using zzs::mllee_native::ConvAttnDefaultConfig;

inline constexpr uint32_t kSeqLen = ConvAttnDefaultConfig::seq_len;             /* 5 */
inline constexpr uint32_t kFeatureCount = ConvAttnDefaultConfig::feature_count; /* 38 */
inline constexpr uint32_t kConvChannels = ConvAttnDefaultConfig::conv_channels; /* 8 */
inline constexpr uint32_t kConvKernel = ConvAttnDefaultConfig::conv_kernel;     /* 3 */
inline constexpr uint32_t kAttnHeadDim = ConvAttnDefaultConfig::attn_head_dim;  /* 4 */
inline constexpr uint32_t kAttnHeads = ConvAttnDefaultConfig::attn_heads;       /* 2 */

inline constexpr uint32_t kConvOutput = kSeqLen * kConvChannels;                 /* 40 */
inline constexpr uint32_t kQKVOutput = kSeqLen * kAttnHeads * kAttnHeadDim * 3u; /* 120 */
inline constexpr uint32_t kAttnOutput = kSeqLen * kAttnHeads * kAttnHeadDim;     /* 40 */
inline constexpr uint32_t kPooled = kConvChannels;                               /* 8 */

/** @brief Head 1 output size: 3 classes (normal/outage/transition). */
inline constexpr uint32_t kHead1Out = 3u;
/** @brief Head 2 output size: 1 confidence code. */
inline constexpr uint32_t kHead2Out = 1u;
/** @brief Head 3 output size: 1 time-to-outage code. */
inline constexpr uint32_t kHead3Out = 1u;

inline constexpr uint32_t kMultiTaskOutputs = kHead1Out + kHead2Out + kHead3Out; /* 5 */

/** @brief Blob size: header + 8 layers + conv weights + QKV weights + head weights + biases. */
inline constexpr uint32_t kMultiTaskBlobSize =
    _S_model_header_size + 8u * _S_layer_desc_size +
    /* CONV1D weights: output_channels * input_channels * kernel = 8*38*3 = 912 */
    kConvChannels * kFeatureCount * kConvKernel +
    /* QKV LINEAR weights: 40*120 = 4800 */
    kConvOutput * kQKVOutput +
    /* Head 1 LINEAR weights: 8*3 = 24 */
    kPooled * kHead1Out +
    /* Head 2 LINEAR weights: 8*1 = 8 */
    kPooled * kHead2Out +
    /* Head 3 LINEAR weights: 8*1 = 8 */
    kPooled * kHead3Out +
    /* Biases as int16: conv 8 + QKV 120 + head1 3 + head2 1 + head3 1 = 133 * 2 = 266 bytes */
    (kConvChannels + kQKVOutput + kHead1Out + kHead2Out + kHead3Out) * 2u;

/** @brief Offsets within the multitask blob. */
namespace offset {
inline constexpr uint32_t header = 0u;
inline constexpr uint32_t layer_table = _S_model_header_size; /* 32 */
/* Start of all weights after the 8 layer descriptors. */
inline constexpr uint32_t weights = layer_table + 8u * _S_layer_desc_size; /* 32 + 384 = 416 */

/* CONV1D: 8 output channels x 38 input channels x 3 kernel = 912 codes. */
inline constexpr uint32_t conv_weights = weights;                                                 /* 416 */
inline constexpr uint32_t conv_bias = conv_weights + kConvChannels * kFeatureCount * kConvKernel; /* 416 + 912 = 1328 */

/* QKV LINEAR: 40*120 = 4800 codes. */
inline constexpr uint32_t qkv_weights = conv_bias + kConvChannels * 2u;      /* 1328 + 16 = 1344 */
inline constexpr uint32_t qkv_bias = qkv_weights + kConvOutput * kQKVOutput; /* 1344 + 4800 = 6144 */

/* Head 1 LINEAR: 8*3 = 24 codes. */
inline constexpr uint32_t head1_weights = qkv_bias + kQKVOutput * 2u;       /* 6144 + 240 = 6384 */
inline constexpr uint32_t head1_bias = head1_weights + kPooled * kHead1Out; /* 6384 + 24 = 6408 */

/* Head 2 LINEAR: 8*1 = 8 codes. */
inline constexpr uint32_t head2_weights = head1_bias + kHead1Out * 2u;      /* 6408 + 6 = 6414 */
inline constexpr uint32_t head2_bias = head2_weights + kPooled * kHead2Out; /* 6414 + 8 = 6422 */

/* Head 3 LINEAR: 8*1 = 8 codes. */
inline constexpr uint32_t head3_weights = head2_bias + kHead2Out * 2u;      /* 6422 + 2 = 6424 */
inline constexpr uint32_t head3_bias = head3_weights + kPooled * kHead3Out; /* 6424 + 8 = 6432 */
} // namespace offset

/* Sanity checks. */
static_assert (kMultiTaskBlobSize == offset::head3_bias + kHead3Out * 2u,
               "multitask blob size must equal the end of the head3 bias");
static_assert (kMultiTaskBlobSize <= 8192, "multitask blob exceeds the flash budget");

} // namespace multitask

/** @brief Write a little-endian u16 into `dst`. */
inline void
_S_write_u16 (uint8_t *dst, uint16_t value) noexcept {
  dst[0] = static_cast<uint8_t> (value & 0xFFu);
  dst[1] = static_cast<uint8_t> ((value >> 8) & 0xFFu);
}

/** @brief Write a little-endian u32 into `dst`. */
inline void
_S_write_u32 (uint8_t *dst, uint32_t value) noexcept {
  dst[0] = static_cast<uint8_t> (value & 0xFFu);
  dst[1] = static_cast<uint8_t> ((value >> 8) & 0xFFu);
  dst[2] = static_cast<uint8_t> ((value >> 16) & 0xFFu);
  dst[3] = static_cast<uint8_t> ((value >> 24) & 0xFFu);
}

/** @brief Write a little-endian i32 into `dst`. */
inline void
_S_write_i32 (uint8_t *dst, int32_t value) noexcept {
  _S_write_u32 (dst, static_cast<uint32_t> (value));
}

/** @brief Build the 32-byte header into `dst`.  Returns Status::OK. */
[[nodiscard]] inline Status
build_header (uint8_t *dst, uint8_t layer_count, uint32_t layer_table_off, uint32_t arena_bytes) noexcept {
  if (dst == nullptr)
    return Status::BAD_LAYOUT;
  dst[0] = 'M';
  dst[1] = 'L';
  dst[2] = 'E';
  dst[3] = 'E';
  dst[4] = _S_model_version;                            /* version */
  dst[5] = 0;                                           /* flags */
  dst[6] = layer_count;                                 /* layer_count */
  dst[7] = static_cast<uint8_t> (_S_model_header_size); /* reserved = header size */
  _S_write_u32 (dst + 8, layer_table_off);
  _S_write_u32 (dst + 12, 0); /* scale_table_off (unused) */
  _S_write_u32 (dst + 16, 0); /* dict_off (unused, RAW codec) */
  _S_write_u16 (dst + 20, 0); /* dict_len */
  _S_write_u16 (dst + 22, static_cast<uint16_t> (arena_bytes));
  dst[24] = static_cast<uint8_t> (Dtype::I8); /* default_dtype */
  dst[25] = static_cast<uint8_t> (Op::COUNT); /* op_count_compat */
  _S_write_u16 (dst + 26, 0);                 /* reserved */
  /* CRC-16 over bytes [0, 28). */
  const uint16_t crc = _S_crc16 (dst, 28);
  _S_write_u16 (dst + 28, crc);
  return Status::OK;
}

/** @brief Build one layer descriptor into `dst` (48 bytes). */
[[nodiscard]] inline Status
build_layer (uint8_t *dst, uint32_t index, Op op, Act act, Dtype dtype, Codec codec, uint16_t in_dim, uint16_t out_dim,
             uint32_t weight_off, uint32_t weight_len, uint32_t bias_off, uint32_t bias_len,
             int32_t requant_mult) noexcept {
  if (dst == nullptr)
    return Status::BAD_LAYOUT;
  uint8_t *const p = dst + index * _S_layer_desc_size;
  p[0] = static_cast<uint8_t> (op);
  p[1] = static_cast<uint8_t> (act);
  p[2] = static_cast<uint8_t> (dtype);
  p[3] = static_cast<uint8_t> (codec);
  _S_write_u16 (p + 4, in_dim);
  _S_write_u16 (p + 6, out_dim);
  p[8] = 0;  /* kernel */
  p[9] = 0;  /* stride */
  p[10] = 0; /* groups */
  p[11] = 0; /* group_size */
  p[12] = 0; /* head_dim */
  p[13] = 0; /* seq */
  p[14] = 0; /* flags */
  p[15] = 0; /* aux */
  _S_write_u32 (p + 16, weight_off);
  _S_write_u32 (p + 20, weight_len);
  _S_write_u32 (p + 24, 0); /* scale_off */
  _S_write_u32 (p + 28, bias_off);
  _S_write_u32 (p + 32, 0);                               /* lut_off */
  _S_write_u16 (p + 36, 0);                               /* in_off (set by caller after arena layout) */
  _S_write_u16 (p + 38, 0);                               /* out_off (set by caller after arena layout) */
  _S_write_u16 (p + 40, static_cast<uint16_t> (in_dim));  /* in_len */
  _S_write_u16 (p + 42, static_cast<uint16_t> (out_dim)); /* out_len */
  _S_write_i32 (p + 44, requant_mult);
  return Status::OK;
}

/** @brief Serialize the complete model blob into `dst` (kBlobSize bytes).
 *
 * `hidden_weights` points to kHiddenInput*kHiddenDim int8 codes (row-major: w[out][in]).
 * `hidden_bias` points to kHiddenDim int8 codes.
 * `output_weights` points to kHiddenDim*kOutputDim int8 codes (row-major: w[out][in]).
 * `output_bias` points to kOutputDim int8 codes.
 * `requant_mult` is the per-output-channel requantization multiplier (Q15,
 * typically 32768 = 1.0 for identity).
 *
 * Returns Status::OK, BAD_LAYOUT if `dst` is null, or OUT_OF_RANGE if the
 * buffers are too small.
 */
[[nodiscard]] inline Status
serialize_model (uint8_t *dst, const int8_t *hidden_weights, const int8_t *hidden_bias, const int8_t *output_weights,
                 const int8_t *output_bias, int32_t requant_mult) noexcept {
  if (dst == nullptr || hidden_weights == nullptr || hidden_bias == nullptr || output_weights == nullptr
      || output_bias == nullptr)
    return Status::BAD_LAYOUT;

  /* Header: 4 layers. */
  Status st = build_header (dst, 4u, offset::layer_table, zzs::mllee::_S_arena_bytes);
  if (!status_ok (st))
    return st;

  /* Layer 0: LINEAR 190->16, RAW weights, external input. */
  st = build_layer (dst + offset::layer_table, 0u, Op::LINEAR, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kInputDim), static_cast<uint16_t> (kHiddenDim), offset::hidden_weights,
                    kHiddenInput * kHiddenDim, offset::hidden_bias, kHiddenDim, requant_mult);
  if (!status_ok (st))
    return st;
  /* LINEAR reads input from the caller buffer (external), not the arena. */
  dst[offset::layer_table + 14] |= 1u;                    /* INPUT_EXTERNAL flag */
  _S_write_u16 (dst + offset::layer_table + 36, 0xFFFFu); /* in_off = external */
  _S_write_u16 (dst + offset::layer_table + 38, 0u);      /* out_off = arena[0] */

  /* Layer 1: ACTIVATION RELU 16->16, in-place. */
  st = build_layer (dst + offset::layer_table, 1u, Op::ACTIVATION, Act::RELU, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kHiddenDim), static_cast<uint16_t> (kHiddenDim), 0, 0, 0, 0, 0);
  if (!status_ok (st))
    return st;
  _S_write_u16 (dst + offset::layer_table + 48 + 36, 0u); /* in_off = arena[0] */
  _S_write_u16 (dst + offset::layer_table + 48 + 38, 0u); /* out_off = arena[0] (in-place) */
  _S_write_u16 (dst + offset::layer_table + 48 + 40, static_cast<uint16_t> (kHiddenDim)); /* in_len */
  _S_write_u16 (dst + offset::layer_table + 48 + 42, static_cast<uint16_t> (kHiddenDim)); /* out_len */
  dst[offset::layer_table + 48 + 14] |= static_cast<uint8_t> (ACT_IN_PLACE);              /* ACT_IN_PLACE flag */

  /* Layer 2: LINEAR 16->3, RAW weights. */
  st = build_layer (dst + offset::layer_table, 2u, Op::LINEAR, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kHiddenDim), static_cast<uint16_t> (kOutputDim), offset::output_weights,
                    kHiddenDim * kOutputDim, offset::output_bias, kOutputDim, requant_mult);
  if (!status_ok (st))
    return st;
  _S_write_u16 (dst + offset::layer_table + 96 + 36, 0u);  /* in_off = arena[0] (hidden) */
  _S_write_u16 (dst + offset::layer_table + 96 + 38, 16u); /* out_off = arena[16] (output) */

  /* Layer 3: SOFTMAX 3->3, in-place. */
  st = build_layer (dst + offset::layer_table, 3u, Op::SOFTMAX, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kOutputDim), static_cast<uint16_t> (kOutputDim), 0, 0, 0, 0, 0);
  if (!status_ok (st))
    return st;
  _S_write_u16 (dst + offset::layer_table + 144 + 36, 16u); /* in_off = arena[16] */
  _S_write_u16 (dst + offset::layer_table + 144 + 38, 16u); /* out_off = arena[16] (in-place) */

  /* Copy hidden weights (row-major: w[out * in_dim + in]). */
  for (uint32_t i = 0; i < kHiddenInput * kHiddenDim; ++i)
    dst[offset::hidden_weights + i] = _S_i8_code (hidden_weights[i]);

  /* Copy output weights (row-major: w[out * in_dim + in]). */
  for (uint32_t i = 0; i < kHiddenDim * kOutputDim; ++i)
    dst[offset::output_weights + i] = _S_i8_code (output_weights[i]);

  /* Copy hidden bias as int16 (2 bytes each, as expected by _S_bias_value). */
  for (uint32_t i = 0; i < kHiddenDim; ++i) {
    _S_write_u16 (dst + offset::hidden_bias + i * 2u, static_cast<uint16_t> (static_cast<int16_t> (hidden_bias[i])));
  }

  /* Copy output bias as int16 (2 bytes each). */
  for (uint32_t i = 0; i < kOutputDim; ++i) {
    _S_write_u16 (dst + offset::output_bias + i * 2u, static_cast<uint16_t> (static_cast<int16_t> (output_bias[i])));
  }

  return Status::OK;
}

/** @brief Validate a model blob: magic, version, CRC, layer count. */
[[nodiscard]] inline Status
validate_model (const uint8_t *model, uint32_t model_len) noexcept {
  if (model == nullptr || model_len < _S_model_header_size)
    return Status::TRUNCATED;
  if (model[0] != 'M' || model[1] != 'L' || model[2] != 'E' || model[3] != 'E')
    return Status::BAD_MAGIC;
  if (model[4] != _S_model_version)
    return Status::BAD_VERSION;
  if (model[7] != static_cast<uint8_t> (_S_model_header_size))
    return Status::BAD_LAYOUT;
  if (_S_crc16 (model, 28) != (static_cast<uint16_t> (model[28]) | (static_cast<uint16_t> (model[29]) << 8)))
    return Status::CORRUPT_STREAM;
  const uint8_t layer_count = model[6];
  if (layer_count == 0)
    return Status::BAD_LAYOUT;
  if (model_len < _S_model_header_size + static_cast<uint32_t> (layer_count) * _S_layer_desc_size)
    return Status::TRUNCATED;
  return Status::OK;
}

/** @brief Serialize the Tier 2 CONV1D + Attention model blob into `dst`.
 *
 * `conv_weights` points to kConvOutput*kFeatureCount*kConvKernel int8 codes
 * (row-major: w[out_channel][in_channel * kernel + tap], RAW codec).
 * `conv_bias` points to kConvChannels int8 codes.
 * `qkv_weights` points to kConvOutput*kQKVOutput int8 codes
 * (row-major: w[out * in_dim + in], RAW codec).
 * `qkv_bias` points to kQKVOutput/3 int8 codes (one per QKV channel group;
 * the head_dim dimension is folded into the group count).
 * `out_weights` points to kPooled*kOutputDimConvAttn int8 codes
 * (row-major: w[out][in], RAW codec).
 * `out_bias` points to kOutputDimConvAttn int8 codes.
 * `requant_mult` is the per-output-channel requantization multiplier (Q15,
 * typically 32768 = 1.0 for identity).
 *
 * Arena layout used by the layer descriptors:
 *   CONV1D   in  = external ([5*38]=190), out = arena[0..40]
 *   LINEAR   QKV in  = arena[0..40],     out = arena[40..160]
 *   ATTENTION in  = arena[40..160] (120=Q+K+V), out = arena[160..200] (40)
 *   GLOBAL_AVG_POOL in  = arena[160..200] (40), out = arena[200..208] (8)
 *   LINEAR   out = arena[200..208], out = arena[208..211] (3)
 *   SOFTMAX  in  = arena[208..211], out = arena[208..211] (in-place)
 *
 * Returns Status::OK, BAD_LAYOUT if `dst` or any weight/bias pointer is null.
 */
[[nodiscard]] inline Status
serialize_conv_attn_model (uint8_t *dst, const int8_t *conv_w, const int8_t *conv_b, const int8_t *qkv_w,
                           const int8_t *qkv_b, const int8_t *out_w, const int8_t *out_b,
                           int32_t requant_mult) noexcept {
  if (dst == nullptr || conv_w == nullptr || conv_b == nullptr || qkv_w == nullptr || qkv_b == nullptr
      || out_w == nullptr || out_b == nullptr)
    return Status::BAD_LAYOUT;

  using namespace conv_attn;
  using namespace conv_attn::offset;

  /* Header: 6 layers. */
  Status st = build_header (dst, 6u, layer_table, zzs::mllee::_S_arena_bytes);
  if (!status_ok (st))
    return st;

  /* Layer 0: CONV1D 38->8, kernel=3, causal, RAW weights, external input.
   * in_dim = input_channels * input_length = 38 * 1 (per position), out_dim = output_channels * seq = 8 * 5 = 40. */
  st = build_layer (dst + layer_table, 0u, Op::CONV1D, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kFeatureCount), static_cast<uint16_t> (kConvOutput), conv_weights,
                    kConvChannels * kFeatureCount * kConvKernel, conv_bias, kConvChannels, requant_mult);
  if (!status_ok (st))
    return st;
  uint8_t *const l0 = dst + layer_table;
  l0[8] = static_cast<uint8_t> (kConvKernel); /* kernel = 3 */
  l0[9] = 1u;                                 /* stride = 1 */
  l0[12] = 1u;                                /* head_dim = input_length = 1 */
  l0[13] = static_cast<uint8_t> (kSeqLen);    /* seq = 5 */
  l0[14] |= static_cast<uint8_t> (LayerFlag::INPUT_EXTERNAL | LayerFlag::CONV_CAUSAL);
  _S_write_u16 (l0 + 36, 0xFFFFu);                               /* in_off = external */
  _S_write_u16 (l0 + 38, 0u);                                    /* out_off = arena[0] */
  _S_write_u16 (l0 + 40, static_cast<uint16_t> (kFeatureCount)); /* in_len */
  _S_write_u16 (l0 + 42, static_cast<uint16_t> (kConvOutput));   /* out_len */

  /* Layer 1: LINEAR 40->120, QKV projection, RAW weights. */
  st = build_layer (dst + layer_table, 1u, Op::LINEAR, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kConvOutput), static_cast<uint16_t> (kQKVOutput), qkv_weights,
                    kConvOutput * kQKVOutput, qkv_bias, kQKVOutput, requant_mult);
  if (!status_ok (st))
    return st;
  uint8_t *const l1 = dst + layer_table + 1u * _S_layer_desc_size;
  _S_write_u16 (l1 + 36, 0u);                                  /* in_off = arena[0] (conv output) */
  _S_write_u16 (l1 + 38, kConvOutput);                         /* out_off = arena[40] */
  _S_write_u16 (l1 + 40, static_cast<uint16_t> (kConvOutput)); /* in_len */
  _S_write_u16 (l1 + 42, static_cast<uint16_t> (kQKVOutput));  /* out_len */

  /* Layer 2: ATTENTION(seq=5, head_dim=4, heads=2), parameter-free.
   * in_dim unused by op; out_dim = seq*heads*head_dim = 40. */
  st = build_layer (dst + layer_table, 2u, Op::ATTENTION, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kQKVOutput), static_cast<uint16_t> (kAttnOutput), 0, 0, 0, 0, 0);
  if (!status_ok (st))
    return st;
  uint8_t *const l2 = dst + layer_table + 2u * _S_layer_desc_size;
  l2[10] = static_cast<uint8_t> (kAttnHeads);                  /* groups = heads */
  l2[12] = static_cast<uint8_t> (kAttnHeadDim);                /* head_dim = 4 */
  l2[13] = static_cast<uint8_t> (kSeqLen);                     /* seq = 5 */
  l2[14] = 0;                                                  /* no causal/RoPE flags */
  _S_write_u16 (l2 + 36, kConvOutput);                         /* in_off = arena[40] */
  _S_write_u16 (l2 + 38, kConvOutput + kQKVOutput);            /* out_off = arena[160] */
  _S_write_u16 (l2 + 40, static_cast<uint16_t> (kQKVOutput));  /* in_len */
  _S_write_u16 (l2 + 42, static_cast<uint16_t> (kAttnOutput)); /* out_len */

  /* Layer 3: GLOBAL_AVG_POOL(seq=5, channels=8): in  = arena[160..200] (40),
   * out = arena[200..208] (8). head_dim = seq (5), so channels = in_dim/head_dim = 40/5 = 8 = out_dim. */
  st = build_layer (dst + layer_table, 3u, Op::GLOBAL_AVG_POOL, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kAttnOutput), static_cast<uint16_t> (kPooled), 0, 0, 0, 0, 0);
  if (!status_ok (st))
    return st;
  uint8_t *const l3 = dst + layer_table + 3u * _S_layer_desc_size;
  l3[8] = 1u;                              /* kernel = 1 */
  l3[9] = 1u;                              /* stride = 1 */
  l3[12] = static_cast<uint8_t> (kSeqLen); /* head_dim = seq = 5 (channels = 40/5 = 8) */
  l3[13] = static_cast<uint8_t> (kSeqLen); /* seq = 5 */
  l3[14] = 0;
  _S_write_u16 (l3 + 36, kConvOutput + kQKVOutput);               /* in_off = arena[160] */
  _S_write_u16 (l3 + 38, kConvOutput + kQKVOutput + kAttnOutput); /* out_off = arena[200] */
  _S_write_u16 (l3 + 40, static_cast<uint16_t> (kAttnOutput));    /* in_len */
  _S_write_u16 (l3 + 42, static_cast<uint16_t> (kPooled));        /* out_len */

  /* Layer 4: LINEAR 8->3, RAW weights. */
  st = build_layer (dst + layer_table, 4u, Op::LINEAR, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kPooled), static_cast<uint16_t> (kOutputDimConvAttn), out_weights,
                    kPooled * kOutputDimConvAttn, out_bias, kOutputDimConvAttn, requant_mult);
  if (!status_ok (st))
    return st;
  uint8_t *const l4 = dst + layer_table + 4u * _S_layer_desc_size;
  _S_write_u16 (l4 + 36, kConvOutput + kQKVOutput + kAttnOutput);           /* in_off = arena[200] */
  _S_write_u16 (l4 + 38, kConvOutput + kQKVOutput + kAttnOutput + kPooled); /* out_off = arena[208] */
  _S_write_u16 (l4 + 40, static_cast<uint16_t> (kPooled));                  /* in_len */
  _S_write_u16 (l4 + 42, static_cast<uint16_t> (kOutputDimConvAttn));       /* out_len */

  /* Layer 5: SOFTMAX 3->3, in-place. */
  st = build_layer (dst + layer_table, 5u, Op::SOFTMAX, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kOutputDimConvAttn), static_cast<uint16_t> (kOutputDimConvAttn), 0, 0, 0, 0,
                    0);
  if (!status_ok (st))
    return st;
  uint8_t *const l5 = dst + layer_table + 5u * _S_layer_desc_size;
  _S_write_u16 (l5 + 36, kConvOutput + kQKVOutput + kAttnOutput + kPooled); /* in_off = arena[208] */
  _S_write_u16 (l5 + 38, kConvOutput + kQKVOutput + kAttnOutput + kPooled); /* out_off = arena[208] (in-place) */

  /* Copy CONV1D weights (row-major: w[out_channel * in_channels * kernel + ...]). */
  for (uint32_t i = 0; i < kConvChannels * kFeatureCount * kConvKernel; ++i)
    dst[conv_weights + i] = _S_i8_code (conv_w[i]);

  /* Copy CONV1D bias as int16 (2 bytes each). */
  for (uint32_t i = 0; i < kConvChannels; ++i)
    _S_write_u16 (dst + conv_bias + i * 2u, static_cast<uint16_t> (static_cast<int16_t> (conv_b[i])));

  /* Copy QKV weights (row-major: w[out * in_dim + in]). */
  for (uint32_t i = 0; i < kConvOutput * kQKVOutput; ++i)
    dst[qkv_weights + i] = _S_i8_code (qkv_w[i]);

  /* Copy QKV bias as int16 (2 bytes each). */
  for (uint32_t i = 0; i < kQKVOutput; ++i)
    _S_write_u16 (dst + qkv_bias + i * 2u, static_cast<uint16_t> (static_cast<int16_t> (qkv_b[i])));

  /* Copy output weights (row-major: w[out * in_dim + in]). */
  for (uint32_t i = 0; i < kPooled * kOutputDimConvAttn; ++i)
    dst[out_weights + i] = _S_i8_code (out_w[i]);

  /* Copy output bias as int16 (2 bytes each). */
  for (uint32_t i = 0; i < kOutputDimConvAttn; ++i)
    _S_write_u16 (dst + out_bias + i * 2u, static_cast<uint16_t> (static_cast<int16_t> (out_b[i])));

  return Status::OK;
}

/** @brief Serialize the Tier 3 multi-task model blob into `dst`.
 *
 * `conv_weights` points to kConvOutput*kFeatureCount*kConvKernel int8 codes
 * (row-major: w[out_channel][in_channel * kernel + tap], RAW codec).
 * `conv_bias` points to kConvChannels int8 codes.
 * `qkv_weights` points to kConvOutput*kQKVOutput int8 codes
 * (row-major: w[out * in_dim + in], RAW codec).
 * `qkv_bias` points to kQKVOutput int8 codes.
 * `head1_weights` points to kPooled*kHead1Out int8 codes (head 1: classification).
 * `head1_bias` points to kHead1Out int8 codes.
 * `head2_weights` points to kPooled*kHead2Out int8 codes (head 2: confidence).
 * `head2_bias` points to kHead2Out int8 codes.
 * `head3_weights` points to kPooled*kHead3Out int8 codes (head 3: time-to-outage).
 * `head3_bias` points to kHead3Out int8 codes.
 * `requant_mult` is the per-output-channel requantization multiplier (Q15,
 * typically 32768 = 1.0 for identity).
 *
 * Arena layout used by the layer descriptors:
 *   CONV1D        in  = external ([5*38]=190), out = arena[0..40]
 *   LINEAR QKV    in  = arena[0..40],          out = arena[40..160]
 *   ATTENTION     in  = arena[40..160] (120),  out = arena[160..200] (40)
 *   GLOBAL_AVG_POOL in  = arena[160..200] (40), out = arena[200..208] (8)
 *   LINEAR head1  in  = arena[200..208], out = arena[208..211] (3)
 *   SOFTMAX head1 in  = arena[208..211], out = arena[208..211] (in-place)
 *   LINEAR head2  in  = arena[200..208], out = arena[211..212] (1)
 *   LINEAR head3  in  = arena[200..208], out = arena[212..213] (1)
 *
 * Returns Status::OK, BAD_LAYOUT if `dst` or any weight/bias pointer is null.
 */
[[nodiscard]] inline Status
serialize_multitask_model (uint8_t *dst, const int8_t *conv_w, const int8_t *conv_b, const int8_t *qkv_w,
                           const int8_t *qkv_b, const int8_t *head1_w, const int8_t *head1_b, const int8_t *head2_w,
                           const int8_t *head2_b, const int8_t *head3_w, const int8_t *head3_b,
                           int32_t requant_mult) noexcept {
  if (dst == nullptr || conv_w == nullptr || conv_b == nullptr || qkv_w == nullptr || qkv_b == nullptr
      || head1_w == nullptr || head1_b == nullptr || head2_w == nullptr || head2_b == nullptr || head3_w == nullptr
      || head3_b == nullptr)
    return Status::BAD_LAYOUT;

  using namespace multitask;
  using namespace multitask::offset;

  /* Header: 8 layers. */
  Status st = build_header (dst, 8u, layer_table, zzs::mllee::_S_arena_bytes);
  if (!status_ok (st))
    return st;

  /* Layer 0: CONV1D 38->8, kernel=3, causal, RAW weights, external input.
   * in_dim = input_channels * input_length = 38 * 1 (per position), out_dim = output_channels * seq = 8 * 5 = 40. */
  st = build_layer (dst + layer_table, 0u, Op::CONV1D, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kFeatureCount), static_cast<uint16_t> (kConvOutput), conv_weights,
                    kConvChannels * kFeatureCount * kConvKernel, conv_bias, kConvChannels, requant_mult);
  if (!status_ok (st))
    return st;
  uint8_t *const l0 = dst + layer_table;
  l0[8] = static_cast<uint8_t> (kConvKernel); /* kernel = 3 */
  l0[9] = 1u;                                 /* stride = 1 */
  l0[12] = 1u;                                /* head_dim = input_length = 1 */
  l0[13] = static_cast<uint8_t> (kSeqLen);    /* seq = 5 */
  l0[14] |= static_cast<uint8_t> (LayerFlag::INPUT_EXTERNAL | LayerFlag::CONV_CAUSAL);
  _S_write_u16 (l0 + 36, 0xFFFFu);                               /* in_off = external */
  _S_write_u16 (l0 + 38, 0u);                                    /* out_off = arena[0] */
  _S_write_u16 (l0 + 40, static_cast<uint16_t> (kFeatureCount)); /* in_len */
  _S_write_u16 (l0 + 42, static_cast<uint16_t> (kConvOutput));   /* out_len */

  /* Layer 1: LINEAR 40->120, QKV projection, RAW weights. */
  st = build_layer (dst + layer_table, 1u, Op::LINEAR, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kConvOutput), static_cast<uint16_t> (kQKVOutput), qkv_weights,
                    kConvOutput * kQKVOutput, qkv_bias, kQKVOutput, requant_mult);
  if (!status_ok (st))
    return st;
  uint8_t *const l1 = dst + layer_table + 1u * _S_layer_desc_size;
  _S_write_u16 (l1 + 36, 0u);                                  /* in_off = arena[0] (conv output) */
  _S_write_u16 (l1 + 38, kConvOutput);                         /* out_off = arena[40] */
  _S_write_u16 (l1 + 40, static_cast<uint16_t> (kConvOutput)); /* in_len */
  _S_write_u16 (l1 + 42, static_cast<uint16_t> (kQKVOutput));  /* out_len */

  /* Layer 2: ATTENTION(seq=5, head_dim=4, heads=2), parameter-free.
   * in_dim unused by op; out_dim = seq*heads*head_dim = 40. */
  st = build_layer (dst + layer_table, 2u, Op::ATTENTION, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kQKVOutput), static_cast<uint16_t> (kAttnOutput), 0, 0, 0, 0, 0);
  if (!status_ok (st))
    return st;
  uint8_t *const l2 = dst + layer_table + 2u * _S_layer_desc_size;
  l2[10] = static_cast<uint8_t> (kAttnHeads);                  /* groups = heads */
  l2[12] = static_cast<uint8_t> (kAttnHeadDim);                /* head_dim = 4 */
  l2[13] = static_cast<uint8_t> (kSeqLen);                     /* seq = 5 */
  l2[14] = 0;                                                  /* no causal/RoPE flags */
  _S_write_u16 (l2 + 36, kConvOutput);                         /* in_off = arena[40] */
  _S_write_u16 (l2 + 38, kConvOutput + kQKVOutput);            /* out_off = arena[160] */
  _S_write_u16 (l2 + 40, static_cast<uint16_t> (kQKVOutput));  /* in_len */
  _S_write_u16 (l2 + 42, static_cast<uint16_t> (kAttnOutput)); /* out_len */

  /* Layer 3: GLOBAL_AVG_POOL(seq=5, channels=8): in  = arena[160..200] (40),
   * out = arena[200..208] (8). head_dim = seq (5), so channels = in_dim/head_dim = 40/5 = 8 = out_dim. */
  st = build_layer (dst + layer_table, 3u, Op::GLOBAL_AVG_POOL, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kAttnOutput), static_cast<uint16_t> (kPooled), 0, 0, 0, 0, 0);
  if (!status_ok (st))
    return st;
  uint8_t *const l3 = dst + layer_table + 3u * _S_layer_desc_size;
  l3[8] = 1u;                              /* kernel = 1 */
  l3[9] = 1u;                              /* stride = 1 */
  l3[12] = static_cast<uint8_t> (kSeqLen); /* head_dim = seq = 5 (channels = 40/5 = 8) */
  l3[13] = static_cast<uint8_t> (kSeqLen); /* seq = 5 */
  l3[14] = 0;
  _S_write_u16 (l3 + 36, kConvOutput + kQKVOutput);               /* in_off = arena[160] */
  _S_write_u16 (l3 + 38, kConvOutput + kQKVOutput + kAttnOutput); /* out_off = arena[200] */
  _S_write_u16 (l3 + 40, static_cast<uint16_t> (kAttnOutput));    /* in_len */
  _S_write_u16 (l3 + 42, static_cast<uint16_t> (kPooled));        /* out_len */

  /* Layer 4: LINEAR 8->3 head 1 (classification), RAW weights. */
  st = build_layer (dst + layer_table, 4u, Op::LINEAR, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kPooled), static_cast<uint16_t> (kHead1Out), head1_weights,
                    kPooled * kHead1Out, head1_bias, kHead1Out, requant_mult);
  if (!status_ok (st))
    return st;
  uint8_t *const l4 = dst + layer_table + 4u * _S_layer_desc_size;
  _S_write_u16 (l4 + 36, kConvOutput + kQKVOutput + kAttnOutput);           /* in_off = arena[200] */
  _S_write_u16 (l4 + 38, kConvOutput + kQKVOutput + kAttnOutput + kPooled); /* out_off = arena[208] */
  _S_write_u16 (l4 + 40, static_cast<uint16_t> (kPooled));                  /* in_len */
  _S_write_u16 (l4 + 42, static_cast<uint16_t> (kHead1Out));                /* out_len */

  /* Layer 5: SOFTMAX head 1 (3->3), in-place. */
  st = build_layer (dst + layer_table, 5u, Op::SOFTMAX, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kHead1Out), static_cast<uint16_t> (kHead1Out), 0, 0, 0, 0, 0);
  if (!status_ok (st))
    return st;
  uint8_t *const l5 = dst + layer_table + 5u * _S_layer_desc_size;
  _S_write_u16 (l5 + 36, kConvOutput + kQKVOutput + kAttnOutput + kPooled); /* in_off = arena[208] */
  _S_write_u16 (l5 + 38, kConvOutput + kQKVOutput + kAttnOutput + kPooled); /* out_off = arena[208] (in-place) */

  /* Layer 6: LINEAR 8->1 head 2 (confidence), RAW weights. */
  st = build_layer (dst + layer_table, 6u, Op::LINEAR, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kPooled), static_cast<uint16_t> (kHead2Out), head2_weights,
                    kPooled * kHead2Out, head2_bias, kHead2Out, requant_mult);
  if (!status_ok (st))
    return st;
  uint8_t *const l6 = dst + layer_table + 6u * _S_layer_desc_size;
  _S_write_u16 (l6 + 36, kConvOutput + kQKVOutput + kAttnOutput);                       /* in_off = arena[200] */
  _S_write_u16 (l6 + 38, kConvOutput + kQKVOutput + kAttnOutput + kPooled + kHead1Out); /* out_off = arena[211] */
  _S_write_u16 (l6 + 40, static_cast<uint16_t> (kPooled));                              /* in_len */
  _S_write_u16 (l6 + 42, static_cast<uint16_t> (kHead2Out));                            /* out_len */

  /* Layer 7: LINEAR 8->1 head 3 (time-to-outage), RAW weights. */
  st = build_layer (dst + layer_table, 7u, Op::LINEAR, Act::NONE, Dtype::I8, Codec::RAW,
                    static_cast<uint16_t> (kPooled), static_cast<uint16_t> (kHead3Out), head3_weights,
                    kPooled * kHead3Out, head3_bias, kHead3Out, requant_mult);
  if (!status_ok (st))
    return st;
  uint8_t *const l7 = dst + layer_table + 7u * _S_layer_desc_size;
  _S_write_u16 (l7 + 36, kConvOutput + kQKVOutput + kAttnOutput); /* in_off = arena[200] */
  _S_write_u16 (l7 + 38,
                kConvOutput + kQKVOutput + kAttnOutput + kPooled + kHead1Out + kHead2Out); /* out_off = arena[212] */
  _S_write_u16 (l7 + 40, static_cast<uint16_t> (kPooled));                                 /* in_len */
  _S_write_u16 (l7 + 42, static_cast<uint16_t> (kHead3Out));                               /* out_len */

  /* Copy CONV1D weights (row-major: w[out_channel * in_channels * kernel + ...]). */
  for (uint32_t i = 0; i < kConvChannels * kFeatureCount * kConvKernel; ++i)
    dst[conv_weights + i] = _S_i8_code (conv_w[i]);

  /* Copy CONV1D bias as int16 (2 bytes each). */
  for (uint32_t i = 0; i < kConvChannels; ++i)
    _S_write_u16 (dst + conv_bias + i * 2u, static_cast<uint16_t> (static_cast<int16_t> (conv_b[i])));

  /* Copy QKV weights (row-major: w[out * in_dim + in]). */
  for (uint32_t i = 0; i < kConvOutput * kQKVOutput; ++i)
    dst[qkv_weights + i] = _S_i8_code (qkv_w[i]);

  /* Copy QKV bias as int16 (2 bytes each). */
  for (uint32_t i = 0; i < kQKVOutput; ++i)
    _S_write_u16 (dst + qkv_bias + i * 2u, static_cast<uint16_t> (static_cast<int16_t> (qkv_b[i])));

  /* Copy head 1 weights (row-major: w[out * in_dim + in]). */
  for (uint32_t i = 0; i < kPooled * kHead1Out; ++i)
    dst[head1_weights + i] = _S_i8_code (head1_w[i]);

  /* Copy head 1 bias as int16 (2 bytes each). */
  for (uint32_t i = 0; i < kHead1Out; ++i)
    _S_write_u16 (dst + head1_bias + i * 2u, static_cast<uint16_t> (static_cast<int16_t> (head1_b[i])));

  /* Copy head 2 weights (row-major: w[out * in_dim + in]). */
  for (uint32_t i = 0; i < kPooled * kHead2Out; ++i)
    dst[head2_weights + i] = _S_i8_code (head2_w[i]);

  /* Copy head 2 bias as int16 (2 bytes each). */
  _S_write_u16 (dst + head2_bias, static_cast<uint16_t> (static_cast<int16_t> (head2_b[0])));

  /* Copy head 3 weights (row-major: w[out * in_dim + in]). */
  for (uint32_t i = 0; i < kPooled * kHead3Out; ++i)
    dst[head3_weights + i] = _S_i8_code (head3_w[i]);

  /* Copy head 3 bias as int16 (2 bytes each). */
  _S_write_u16 (dst + head3_bias, static_cast<uint16_t> (static_cast<int16_t> (head3_b[0])));

  return Status::OK;
}

/** @} */
} // namespace zzs::mllee_native::export_
