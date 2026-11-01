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
#include "mllee/core_packlow_codec.hpp"
#include "mllee/core_status.hpp"

#include <cstdint>

namespace zzs::mllee_native::quant_packlow {

using zzs::mllee::Dtype;
using zzs::mllee::PacklowBufferSink;
using zzs::mllee::PacklowDecoder;
using zzs::mllee::PacklowEncoder;
using zzs::mllee::PacklowOptions;
using zzs::mllee::PacklowSink;
using zzs::mllee::Status;
using zzs::mllee::status_ok;

/** @brief Per-channel affine quantization parameters. */
struct QuantScale {
  int8_t _M_zero_point = 0;
  float _M_scale = 0.0f;
};

/** @brief Max positive code for an integer dtype: 127/7/3/1 for I8/I4/I3/I2. */
[[nodiscard]] inline constexpr int32_t
_S_qmax (const Dtype dtype) noexcept {
  return dtype == Dtype::I8 ? 127 : dtype == Dtype::I4 ? 7 : dtype == Dtype::I3 ? 3 : dtype == Dtype::I2 ? 1 : 0;
}

/** @brief Max absolute code magnitude for an integer dtype: 128/8/4/2 for I8/I4/I3/I2. */
[[nodiscard]] inline constexpr int32_t
_S_qmag (const Dtype dtype) noexcept {
  return dtype == Dtype::I8 ? 128 : dtype == Dtype::I4 ? 8 : dtype == Dtype::I3 ? 4 : dtype == Dtype::I2 ? 2 : 0;
}

/** @brief Codes per stream byte for a byte-divisible dtype, else 0. */
[[nodiscard]] inline constexpr uint32_t
_S_codes_per_byte (const Dtype dtype) noexcept {
  const uint32_t bits = zzs::mllee::_S_code_bits (dtype);
  return (8u % bits) == 0u ? 8u / bits : 0u;
}

/**
 * @brief Quantize float32 weights to int codes with per-channel scales.
 *
 * `src` is treated as `num_channels` consecutive groups of
 * `count / num_channels` weights each (the last group may be smaller).  For
 * each group `scale = max_abs / qmax` and every weight is rounded to the
 * nearest code in `[-qmax, qmax]` and divided by `scale` to recover the code.
 *
 * Integer dtypes only (I8/I4/I3/I2); returns 0 for a float dtype or on bad
 * input.  `dst` must hold `count` int8_t codes.  `scales` must hold
 * `num_channels` entries.
 */
[[nodiscard]] inline uint32_t
quantize_weights (const float *src, const uint32_t count, const Dtype dtype, int8_t *dst, QuantScale *scales,
                  const uint32_t num_channels) noexcept {
  if (src == nullptr || dst == nullptr || scales == nullptr || count == 0 || num_channels == 0
      || !zzs::mllee::_S_integer_dtype (dtype))
    return 0;
  const int32_t qmax = _S_qmax (dtype);
  if (qmax <= 0)
    return 0;

  const uint32_t per_channel = (count + num_channels - 1u) / num_channels;
  uint32_t written = 0;

  for (uint32_t ch = 0; ch < num_channels && written < count; ++ch) {
    const uint32_t begin = ch * per_channel;
    const uint32_t end = begin + per_channel < count ? begin + per_channel : count;
    float max_abs = 0.0f;
    for (uint32_t i = begin; i < end; ++i) {
      const float mag = src[i] < 0.0f ? -src[i] : src[i];
      if (mag > max_abs)
        max_abs = mag;
    }
    QuantScale &scale = scales[ch];
    scale._M_zero_point = 0;
    if (!(max_abs > 0.0f)) {
      scale._M_scale = 0.0f;
      for (uint32_t i = begin; i < end; ++i)
        dst[written++] = 0;
      continue;
    }
    scale._M_scale = max_abs / static_cast<float> (qmax);
    const float inv_scale = 1.0f / scale._M_scale;
    for (uint32_t i = begin; i < end; ++i) {
      const float quotient = src[i] * inv_scale;
      int32_t code = static_cast<int32_t> (quotient >= 0.0f ? quotient + 0.5f : quotient - 0.5f);
      if (code < -qmax)
        code = -qmax;
      else if (code > qmax)
        code = qmax;
      dst[written++] = static_cast<int8_t> (code);
    }
  }
  return written;
}

/**
 * @brief Pack int8 codes into a Packlow stream.
 *
 * The codes are byte-encoded (one byte per code) regardless of `dtype`; the
 * dtype is recorded only so the matching reader can unpack them back through
 * `unpack_and_dequantize`.  Returns the number of packed bytes written, or 0
 * on failure / insufficient capacity.
 */
[[nodiscard]] inline uint32_t
pack_quantized_weights (const int8_t *codes, const uint32_t count, PacklowEncoder &encoder, uint8_t *output,
                        const uint32_t capacity) noexcept {
  if (codes == nullptr || output == nullptr || capacity < zzs::mllee::_S_pk_header_bytes)
    return 0;
  PacklowOptions options{};
  PacklowBufferSink buffer{output, capacity, 0, false};
  PacklowSink sink{&buffer, zzs::mllee::_S_pk_buffer_write};
  if (encoder.begin (reinterpret_cast<const uint8_t *> (codes), count, options, sink) != zzs::mllee::Status::OK)
    return 0;
  if (encoder.run () != zzs::mllee::Status::OK || buffer._M_overflow)
    return 0;
  return buffer._M_size;
}

/**
 * @brief Unpack Packlow bytes back to float32 weights with per-channel scales.
 *
 * `packed` is the stream produced by `pack_quantized_weights`; `packed_len` is
 * its length (the Packlow header carries the raw code count).  The codes are
 * read through `PacklowDecoder` and dequantized as `code * scale[channel]`.
 * `dst` must hold `dst_capacity` floats; returns the number of floats written.
 */
[[nodiscard]] inline uint32_t
unpack_and_dequantize (const uint8_t *packed, const uint32_t packed_len, const Dtype dtype, const QuantScale *scales,
                       const uint32_t num_channels, float *dst, const uint32_t dst_capacity) noexcept {
  if (packed == nullptr || packed_len < zzs::mllee::_S_pk_header_bytes || scales == nullptr || dst == nullptr
      || num_channels == 0 || !zzs::mllee::_S_integer_dtype (dtype))
    return 0;
  PacklowDecoder decoder;
  if (decoder.begin (packed, packed_len) != Status::OK)
    return 0;
  if (decoder.begin_weights (packed, packed_len, 0, packed_len, dtype, nullptr) != Status::OK)
    return 0;

  const int32_t qmax = _S_qmax (dtype);
  if (qmax <= 0)
    return 0;

  /* Count how many codes (one byte each) the stream decodes to. */
  uint32_t code_count = 0;
  {
    PacklowDecoder counter;
    if (counter.begin (packed, packed_len) != Status::OK)
      return 0;
    if (counter.begin_weights (packed, packed_len, 0, packed_len, dtype, nullptr) != Status::OK)
      return 0;
    while (counter.next_byte () >= 0)
      ++code_count;
  }
  if (code_count == 0 || code_count > dst_capacity)
    return 0;

  /* Per-channel groups: channel c covers codes [c*per_channel, (c+1)*per_channel). */
  const uint32_t per_channel = (code_count + num_channels - 1u) / num_channels;

  uint32_t written = 0;
  uint32_t channel = 0;
  uint32_t index = 0;
  for (;;) {
    const int32_t byte = decoder.next_byte ();
    if (byte < 0)
      break;
    if (index != 0 && index % per_channel == 0)
      ++channel;
    const int32_t code = static_cast<int8_t> (byte);
    const float scale = scales[channel]._M_scale;
    dst[written++] = static_cast<float> (code) * scale;
    ++index;
    if (written >= dst_capacity)
      break;
  }
  return written;
}

} // namespace zzs::mllee_native::quant_packlow
