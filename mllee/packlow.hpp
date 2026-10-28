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

#include "mllee/format.hpp"
#include "mllee/status.hpp"

#include <cstdint>

namespace zzs::mllee
{
inline constexpr uint32_t _S_pk_header_bytes = 12;
inline constexpr uint32_t _S_pk_window_bytes = 64;
inline constexpr uint32_t _S_pk_dict_block_bytes = 32;
inline constexpr uint32_t _S_pk_stage_bytes = 64;
inline constexpr uint32_t _S_pk_max_match = 4096;
inline constexpr uint32_t _S_pk_chain_max = 8;
inline constexpr uint8_t _S_pk_version = 1;
inline constexpr uint16_t _S_pk_prob_init = 2048;
inline constexpr uint16_t _S_pk_prob_min = 1;
inline constexpr uint16_t _S_pk_prob_max = 4095;
inline constexpr uint8_t _S_pk_prob_shift = 5;
inline constexpr uint8_t _S_pk_flag_delta = 1u << 0;
inline constexpr uint8_t _S_pk_flag_dict = 1u << 1;
inline constexpr uint8_t _S_pk_raw = 1u << 0;
inline constexpr uint8_t _S_pk_half = 1u << 1;
inline constexpr int32_t _S_pk_error = -1000;

enum _S_PkProb : uint8_t
{
    _S_PK_PROB_TOKEN0 = 0,
    _S_PK_PROB_TOKEN1 = 1,
    _S_PK_PROB_TOKEN2 = 2,
    _S_PK_PROB_LITERAL0 = 3,
    _S_PK_PROB_LITERAL1 = 4,
    _S_PK_PROB_LITERAL2 = 5,
    _S_PK_PROB_LITERAL3 = 6,
    _S_PK_PROB_GAMMA = 7,
    _S_PK_PROB_DELTA_SIGN = 8,
    _S_PK_PROB_DELTA_MSB = 9,
    _S_PK_PROB_COUNT = 10,
};

enum class _S_PkToken : uint8_t
{
    LITERAL = 0,
    COPY = 1,
    DELTA = 2,
    DICT_COPY = 3,
};

enum _S_PkKind : uint8_t
{
    _S_PK_KIND_NONE = 0,
    _S_PK_KIND_RING = 1,
    _S_PK_KIND_DICT = 2,
    _S_PK_KIND_DELTA = 3,
};

/**
 * @brief Decoder state.  Exactly 116 bytes on a 32-bit target.
 *
 * The 64-byte ring is both the LZ window and the weight staging buffer used by
 * the convolution kernel: a whole output row is decoded into the ring and then
 * consumed before the next row is decoded.
 */
struct PacklowState
{
    const int8_t *_M_lut;
    uint32_t _M_range;
    uint32_t _M_code;
    uint32_t _M_src_off;
    uint32_t _M_src_end;
    uint32_t _M_remaining;
    uint16_t _M_prob[_S_PK_PROB_COUNT];
    uint16_t _M_crc;
    uint16_t _M_crc_expected;
    uint16_t _M_pending;
    uint16_t _M_dict_off;
    uint8_t _M_ring[_S_pk_window_bytes];
    uint8_t _M_ring_pos;
    uint8_t _M_kind;
    uint8_t _M_dist;
    uint8_t _M_status;
    uint8_t _M_dtype;
    uint8_t _M_nibble;
    uint8_t _M_flags;
};

static_assert (sizeof (PacklowState) <= 136, "mllee packlow state exceeds its RAM budget");

[[nodiscard]] inline constexpr uint32_t
_S_pk_u32 (const uint8_t *data) noexcept
{
    return static_cast<uint32_t> (data[0]) | (static_cast<uint32_t> (data[1]) << 8)
           | (static_cast<uint32_t> (data[2]) << 16)
           | (static_cast<uint32_t> (data[3]) << 24);
}

[[nodiscard]] inline constexpr uint16_t
_S_pk_u16 (const uint8_t *data) noexcept
{
    return static_cast<uint16_t> (static_cast<uint16_t> (data[0])
                                       | (static_cast<uint16_t> (data[1]) << 8));
}

[[nodiscard]] inline Status
_S_pk_begin (PacklowState &state, const uint8_t *model, const uint32_t model_len,
             const uint32_t weight_off, const uint32_t weight_len, const Codec codec) noexcept
{
    state = {};
    if (model == nullptr || !_S_span_ok (weight_off, weight_len, model_len))
        return Status::TRUNCATED;
    if (codec == Codec::RAW)
        {
            state._M_flags = _S_pk_raw;
            state._M_src_off = weight_off;
            state._M_src_end = weight_off + weight_len;
            state._M_remaining = weight_len;
            state._M_status = static_cast<uint8_t> (Status::OK);
            return Status::OK;
        }
    if (weight_len < _S_pk_header_bytes)
        return Status::TRUNCATED;
    const uint8_t *const header = model + weight_off;
    if (header[0] != 'M' || header[1] != 'L' || header[2] != 'P' || header[3] != 'K')
        return Status::BAD_MAGIC;
    if (header[4] != _S_pk_version)
        return Status::BAD_VERSION;
    state._M_remaining = _S_pk_u32 (header + 6);
    state._M_crc_expected = _S_pk_u16 (header + 10);
    state._M_crc = 0xFFFFu;
    state._M_src_off = weight_off + _S_pk_header_bytes;
    state._M_src_end = weight_off + weight_len;
    state._M_range = 0xFFFFFFFFu;
    state._M_code = 0;
    for (unsigned int i = 0; i < 5; ++i)
        {
            if (state._M_src_off >= state._M_src_end)
                {
                    state._M_status = static_cast<uint8_t> (Status::TRUNCATED);
                    return Status::TRUNCATED;
                }
            state._M_code = (state._M_code << 8) | model[state._M_src_off++];
        }
    for (unsigned int i = 0; i < _S_PK_PROB_COUNT; ++i)
        state._M_prob[i] = _S_pk_prob_init;
    state._M_status = static_cast<uint8_t> (Status::OK);
    return Status::OK;
}

[[nodiscard]] inline Status
_S_pk_status (const PacklowState &state) noexcept
{
    return static_cast<Status> (state._M_status);
}

inline void
_S_pk_set_status (PacklowState &state, const Status status) noexcept
{
    state._M_status = static_cast<uint8_t> (status);
}

[[nodiscard]] ZZS_MLLEE_INLINE int32_t
_S_pk_src_byte (PacklowState &state, const uint8_t *model) noexcept
{
    if (state._M_src_off >= state._M_src_end)
        {
            _S_pk_set_status (state, Status::TRUNCATED);
            return -1;
        }
    return model[state._M_src_off++];
}

[[nodiscard]] ZZS_MLLEE_INLINE int32_t
_S_pk_decode_bit (PacklowState &state, const uint8_t *model, uint16_t &prob) noexcept
{
    const uint32_t bound = (state._M_range >> 12) * static_cast<uint32_t> (prob);
    uint32_t bit = 0;
    if (state._M_code < bound)
        {
            state._M_range = bound;
            prob = static_cast<uint16_t> (prob + ((_S_pk_prob_max + 1 - prob) >> _S_pk_prob_shift));
            if (prob > _S_pk_prob_max)
                prob = _S_pk_prob_max;
        }
    else
        {
            state._M_code -= bound;
            state._M_range -= bound;
            prob = static_cast<uint16_t> (prob - (prob >> _S_pk_prob_shift));
            if (prob < _S_pk_prob_min)
                prob = _S_pk_prob_min;
            bit = 1;
        }
    while (state._M_range < (1u << 24))
        {
            const int32_t byte = _S_pk_src_byte (state, model);
            if (byte < 0)
                return -1;
            state._M_range <<= 8;
            state._M_code = (state._M_code << 8) | static_cast<uint32_t> (byte);
        }
    return static_cast<int32_t> (bit);
}

[[nodiscard]] ZZS_MLLEE_INLINE int32_t
_S_pk_decode_fixed_bit (PacklowState &state, const uint8_t *model) noexcept
{
    uint16_t prob = _S_pk_prob_init;
    return _S_pk_decode_bit (state, model, prob);
}

[[nodiscard]] ZZS_MLLEE_INLINE int32_t
_S_pk_decode_gamma (PacklowState &state, const uint8_t *model, const uint32_t limit) noexcept
{
    uint32_t prefix = 0;
    for (;;)
        {
            const int32_t bit = _S_pk_decode_bit (state, model, state._M_prob[_S_PK_PROB_GAMMA]);
            if (bit < 0)
                return -1;
            if (bit != 0)
                break;
            if (++prefix > 24)
                {
                    _S_pk_set_status (state, Status::CORRUPT_STREAM);
                    return -1;
                }
        }
    uint32_t value = 1;
    for (uint32_t i = 0; i < prefix; ++i)
        {
            const int32_t bit = _S_pk_decode_fixed_bit (state, model);
            if (bit < 0)
                return -1;
            value = (value << 1) | static_cast<uint32_t> (bit);
        }
    if (value > limit)
        {
            _S_pk_set_status (state, Status::CORRUPT_STREAM);
            return -1;
        }
    return static_cast<int32_t> (value);
}

[[nodiscard]] ZZS_MLLEE_INLINE int32_t
_S_pk_decode_byte (PacklowState &state, const uint8_t *model) noexcept
{
    uint32_t value = 0;
    for (int shift = 3; shift >= 0; --shift)
        {
            const int32_t bit = _S_pk_decode_bit (state, model, state._M_prob[_S_PK_PROB_LITERAL0 + shift]);
            if (bit < 0)
                return -1;
            value |= static_cast<uint32_t> (bit) << shift;
        }
    value <<= 4;
    for (int shift = 3; shift >= 0; --shift)
        {
            const int32_t bit = _S_pk_decode_bit (state, model, state._M_prob[_S_PK_PROB_LITERAL0 + shift]);
            if (bit < 0)
                return -1;
            value |= static_cast<uint32_t> (bit) << shift;
        }
    return static_cast<int32_t> (value);
}

[[nodiscard]] ZZS_MLLEE_INLINE int32_t
_S_pk_decode_correction (PacklowState &state, const uint8_t *model) noexcept
{
    const int32_t sign = _S_pk_decode_bit (state, model, state._M_prob[_S_PK_PROB_DELTA_SIGN]);
    if (sign < 0)
        return -256;
    uint32_t magnitude = 0;
    const int32_t msb = _S_pk_decode_bit (state, model, state._M_prob[_S_PK_PROB_DELTA_MSB]);
    if (msb < 0)
        return -256;
    magnitude = static_cast<uint32_t> (msb) << 3;
    for (int shift = 2; shift >= 0; --shift)
        {
            const int32_t bit = _S_pk_decode_fixed_bit (state, model);
            if (bit < 0)
                return -256;
            magnitude |= static_cast<uint32_t> (bit) << shift;
        }
    return sign != 0 ? -static_cast<int32_t> (magnitude)
                     : static_cast<int32_t> (magnitude);
}

[[nodiscard]] ZZS_MLLEE_INLINE int32_t
_S_pk_emit (PacklowState &state, const uint8_t byte) noexcept
{
    if (state._M_remaining == 0)
        {
            _S_pk_set_status (state, Status::CORRUPT_STREAM);
            return -1;
        }
    state._M_ring[state._M_ring_pos] = byte;
    state._M_ring_pos = static_cast<uint8_t> ((state._M_ring_pos + 1u) & (_S_pk_window_bytes - 1u));
    --state._M_remaining;
    state._M_crc = _S_crc16_update (state._M_crc, byte);
    if (state._M_remaining == 0 && state._M_crc != state._M_crc_expected)
        _S_pk_set_status (state, Status::CORRUPT_STREAM);
    return static_cast<int32_t> (byte);
}

[[nodiscard]] inline int32_t
_S_pk_next (PacklowState &state, const uint8_t *model, const uint8_t *dict,
            const uint16_t dict_len) noexcept
{
    if (static_cast<Status> (state._M_status) != Status::OK || state._M_remaining == 0)
        return -1;
    if ((state._M_flags & _S_pk_raw) != 0)
        {
            const int32_t byte = _S_pk_src_byte (state, model);
            if (byte < 0)
                return -1;
            --state._M_remaining;
            state._M_ring[state._M_ring_pos] = static_cast<uint8_t> (byte);
            state._M_ring_pos =
                static_cast<uint8_t> ((state._M_ring_pos + 1u) & (_S_pk_window_bytes - 1u));
            return byte;
        }

    for (;;)
        {
            if (state._M_pending == 0)
                {
                    const int32_t first = _S_pk_decode_bit (state, model, state._M_prob[_S_PK_PROB_TOKEN0]);
                    if (first < 0)
                        return -1;
                    _S_PkToken token;
                    if (first == 0)
                        {
                            const int32_t second =
                                _S_pk_decode_bit (state, model, state._M_prob[_S_PK_PROB_TOKEN1]);
                            if (second < 0)
                                return -1;
                            token = second != 0 ? _S_PkToken::COPY : _S_PkToken::LITERAL;
                        }
                    else
                        {
                            const int32_t second =
                                _S_pk_decode_bit (state, model, state._M_prob[_S_PK_PROB_TOKEN2]);
                            if (second < 0)
                                return -1;
                            token = second != 0 ? _S_PkToken::DICT_COPY : _S_PkToken::DELTA;
                        }

                    if (token == _S_PkToken::LITERAL)
                        {
                            const int32_t byte = _S_pk_decode_byte (state, model);
                            if (byte < 0)
                                return -1;
                            return _S_pk_emit (state, static_cast<uint8_t> (byte));
                        }

                    if (token == _S_PkToken::COPY)
                        {
                            const int32_t distance =
                                _S_pk_decode_gamma (state, model, _S_pk_window_bytes);
                            const int32_t length = _S_pk_decode_gamma (state, model, state._M_remaining);
                            if (distance < 0 || length < 0)
                                return -1;
                            if (distance < 1 || static_cast<uint32_t> (length) > state._M_remaining)
                                {
                                    _S_pk_set_status (state, Status::CORRUPT_STREAM);
                                    return -1;
                                }
                            state._M_kind = _S_PK_KIND_RING;
                            state._M_dist = static_cast<uint8_t> (distance);
                            state._M_pending = static_cast<uint16_t> (length);
                        }
                    else if (token == _S_PkToken::DELTA)
                        {
                            const int32_t distance =
                                _S_pk_decode_gamma (state, model, _S_pk_window_bytes);
                            const int32_t length = _S_pk_decode_gamma (state, model, state._M_remaining);
                            if (distance < 0 || length < 0)
                                return -1;
                            if (distance < 1 || static_cast<uint32_t> (length) > state._M_remaining)
                                {
                                    _S_pk_set_status (state, Status::CORRUPT_STREAM);
                                    return -1;
                                }
                            state._M_kind = _S_PK_KIND_DELTA;
                            state._M_dist = static_cast<uint8_t> (distance);
                            state._M_pending = static_cast<uint16_t> (length);
                        }
                    else
                        {
                            const int32_t index =
                                _S_pk_decode_gamma (state, model, 0xFFFFu);
                            const int32_t length = _S_pk_decode_gamma (state, model, state._M_remaining);
                            if (index < 0 || length < 0 || dict == nullptr)
                                {
                                    _S_pk_set_status (state, Status::CORRUPT_STREAM);
                                    return -1;
                                }
                            const uint32_t block = static_cast<uint32_t> (index - 1)
                                                        * _S_pk_dict_block_bytes;
                            if (block >= dict_len
                                || static_cast<uint32_t> (length) > dict_len - block)
                                {
                                    _S_pk_set_status (state, Status::CORRUPT_STREAM);
                                    return -1;
                                }
                            state._M_kind = _S_PK_KIND_DICT;
                            state._M_dict_off = static_cast<uint16_t> (block);
                            state._M_pending = static_cast<uint16_t> (length);
                        }
                    continue;
                }

            uint8_t byte;
            if (state._M_kind == _S_PK_KIND_RING)
                {
                    byte = state._M_ring[(state._M_ring_pos + _S_pk_window_bytes - state._M_dist)
                                         & (_S_pk_window_bytes - 1u)];
                }
            else if (state._M_kind == _S_PK_KIND_DICT)
                {
                    if (dict == nullptr || state._M_dict_off >= dict_len)
                        {
                            _S_pk_set_status (state, Status::CORRUPT_STREAM);
                            return -1;
                        }
                    byte = dict[state._M_dict_off++];
                }
            else
                {
                    const int32_t correction = _S_pk_decode_correction (state, model);
                    if (correction == -256)
                        return -1;
                    const uint8_t base =
                        state._M_ring[(state._M_ring_pos + _S_pk_window_bytes - state._M_dist)
                                      & (_S_pk_window_bytes - 1u)];
                    byte = static_cast<uint8_t> (static_cast<int32_t> (base) + correction);
                }
            --state._M_pending;
            const int32_t emitted = _S_pk_emit (state, byte);
            if (emitted >= 0)
                return emitted;
            return -1;
        }
}

/** @brief Arm the weight reader: LUT and dtype live in the codec state, no extra struct. */
inline void
_S_rd_begin (PacklowState &state, const int8_t *lut, const Dtype dtype) noexcept
{
    state._M_lut = lut;
    state._M_dtype = static_cast<uint8_t> (dtype);
    state._M_nibble = 0;
    state._M_flags = static_cast<uint8_t> (state._M_flags & ~_S_pk_half);
}

[[nodiscard]] ZZS_MLLEE_INLINE int32_t
_S_rd_byte (PacklowState &state, const uint8_t *model, const uint8_t *dict,
            const uint16_t dict_len) noexcept
{
    if (static_cast<Status> (state._M_status) != Status::OK)
        return _S_pk_error;
    const int32_t byte = _S_pk_next (state, model, dict, dict_len);
    if (byte < 0)
        {
            if (static_cast<Status> (state._M_status) == Status::OK)
                _S_pk_set_status (state, Status::TRUNCATED);
            return _S_pk_error;
        }
    return byte;
}

[[nodiscard]] ZZS_MLLEE_INLINE int32_t
_S_rd_code (PacklowState &state, const uint8_t *model, const uint8_t *dict,
            const uint16_t dict_len) noexcept
{
    const bool packed = state._M_dtype == static_cast<uint8_t> (Dtype::I4)
                        || state._M_dtype == static_cast<uint8_t> (Dtype::F4E2M1);
    if (packed)
        {
            uint8_t nibble;
            if ((state._M_flags & _S_pk_half) == 0)
                {
                    const int32_t byte = _S_rd_byte (state, model, dict, dict_len);
                    if (byte == _S_pk_error)
                        return _S_pk_error;
                    state._M_nibble = static_cast<uint8_t> (byte);
                    state._M_flags = static_cast<uint8_t> (state._M_flags | _S_pk_half);
                    nibble = static_cast<uint8_t> (byte & 0x0F);
                }
            else
                {
                    nibble = static_cast<uint8_t> ((state._M_nibble >> 4) & 0x0F);
                    state._M_flags = static_cast<uint8_t> (state._M_flags & ~_S_pk_half);
                }
            return state._M_lut != nullptr ? state._M_lut[nibble] : _S_from_int4 (nibble);
        }
    const int32_t byte = _S_rd_byte (state, model, dict, dict_len);
    if (byte == _S_pk_error)
        return _S_pk_error;
    if (state._M_lut != nullptr)
        return state._M_lut[static_cast<uint8_t> (byte)];
    return static_cast<int8_t> (static_cast<uint8_t> (byte));
}

/** @brief Read a code from the current 64-byte weight row staged in the ring. */
[[nodiscard]] ZZS_MLLEE_INLINE int32_t
_S_rd_staged (const PacklowState &state, const uint32_t index, const uint32_t row_bytes) noexcept
{
    const uint32_t start =
        (static_cast<uint32_t> (state._M_ring_pos) + _S_pk_window_bytes - row_bytes)
        & (_S_pk_window_bytes - 1u);
    if (state._M_dtype == static_cast<uint8_t> (Dtype::I4)
        || state._M_dtype == static_cast<uint8_t> (Dtype::F4E2M1))
        {
            const uint8_t byte =
                state._M_ring[(start + (index >> 1)) & (_S_pk_window_bytes - 1u)];
            const uint8_t nibble =
                (index & 1u) != 0 ? static_cast<uint8_t> ((byte >> 4) & 0x0F)
                                  : static_cast<uint8_t> (byte & 0x0F);
            return state._M_lut != nullptr ? state._M_lut[nibble] : _S_from_int4 (nibble);
        }
    const uint8_t byte = state._M_ring[(start + index) & (_S_pk_window_bytes - 1u)];
    return state._M_lut != nullptr ? state._M_lut[byte] : static_cast<int8_t> (byte);
}

struct PacklowStats
{
    uint32_t _M_literal = 0;
    uint32_t _M_copy = 0;
    uint32_t _M_delta = 0;
    uint32_t _M_dict = 0;
};

struct PacklowOptions
{
    const uint8_t *_M_dict = nullptr;
    uint16_t _M_dict_len = 0;
    bool _M_delta = true;
    bool _M_dict_copy = true;
    PacklowStats *_M_stats = nullptr;
};

/** @brief Caller-owned output callback; no allocation is hidden in the codec. */
struct PacklowSink
{
    void *_M_context = nullptr;
    bool (*_M_write) (void *context, const uint8_t *data, uint32_t length) noexcept = nullptr;
};

/** @brief Sink over a caller-owned byte buffer, for one-shot encoding. */
struct PacklowBufferSink
{
    uint8_t *_M_data = nullptr;
    uint32_t _M_capacity = 0;
    uint32_t _M_size = 0;
    bool _M_overflow = false;
};

[[nodiscard]] inline bool
_S_pk_buffer_write (void *context, const uint8_t *data, const uint32_t length) noexcept
{
    auto &sink = *static_cast<PacklowBufferSink *> (context);
    if (length > sink._M_capacity - sink._M_size)
        {
            sink._M_overflow = true;
            return false;
        }
    for (uint32_t i = 0; i < length; ++i)
        sink._M_data[sink._M_size + i] = data[i];
    sink._M_size += length;
    return true;
}

/**
 * @brief Pull-based streaming packlow decoder.
 *
 * The compressed bytes stay where the caller put them (normally Flash); the
 * decoder reads them one at a time and writes decoded bytes into caller-owned
 * storage.  No buffer proportional to the input or output is allocated, so the
 * same object works on the host and on a Cortex-M4.
 */
class PacklowDecoder
{
  public:
    PacklowDecoder () noexcept = default;

    [[nodiscard]] Status
    begin (const uint8_t *stream, const uint32_t stream_len, const uint8_t *dict = nullptr,
           const uint16_t dict_len = 0) noexcept
    {
        return this->begin (stream, stream_len, 0, stream_len, Codec::PACKLOW, dict, dict_len);
    }

    [[nodiscard]] Status
    begin (const uint8_t *stream, const uint32_t stream_len, const uint32_t offset,
           const uint32_t length, const Codec codec, const uint8_t *dict = nullptr,
           const uint16_t dict_len = 0) noexcept
    {
        this->_M_stream = stream;
        this->_M_dict = dict;
        this->_M_dict_len = dict_len;
        return _S_pk_begin (this->_M_state, stream, stream_len, offset, length, codec);
    }

    [[nodiscard]] Status
    begin_weights (const uint8_t *stream, const uint32_t stream_len, const uint32_t offset,
                   const uint32_t length, const Dtype dtype, const int8_t *lut = nullptr,
                   const uint8_t *dict = nullptr, const uint16_t dict_len = 0) noexcept
    {
        const Status status =
            this->begin (stream, stream_len, offset, length, Codec::PACKLOW, dict, dict_len);
        if (status == Status::OK)
            _S_rd_begin (this->_M_state, lut, dtype);
        return status;
    }

    /** @brief Decode one byte, or -1 when the stream ended or failed. */
    [[nodiscard]] int32_t
    next_byte () noexcept
    {
        return _S_pk_next (this->_M_state, this->_M_stream, this->_M_dict, this->_M_dict_len);
    }

    /** @brief Decode one weight code through the configured dtype/LUT. */
    [[nodiscard]] int32_t
    next_code () noexcept
    {
        const int32_t code =
            _S_rd_code (this->_M_state, this->_M_stream, this->_M_dict, this->_M_dict_len);
        return code == _S_pk_error ? -1 : code;
    }

    /** @brief Fill `output` with up to `capacity` bytes; returns the count written. */
    [[nodiscard]] uint32_t
    read (uint8_t *output, const uint32_t capacity) noexcept
    {
        if (output == nullptr)
            return 0;
        uint32_t produced = 0;
        while (produced < capacity)
            {
                const int32_t byte = this->next_byte ();
                if (byte < 0)
                    break;
                output[produced++] = static_cast<uint8_t> (byte);
            }
        return produced;
    }

    /** @brief Fill `output` with up to `capacity` weight codes. */
    [[nodiscard]] uint32_t
    read (int8_t *output, const uint32_t capacity) noexcept
    {
        if (output == nullptr)
            return 0;
        uint32_t produced = 0;
        while (produced < capacity)
            {
                const int32_t code = this->next_code ();
                if (code < 0)
                    break;
                output[produced++] = static_cast<int8_t> (code);
            }
        return produced;
    }

    [[nodiscard]] Status
    status () const noexcept
    {
        return _S_pk_status (this->_M_state);
    }

    [[nodiscard]] uint32_t
    remaining () const noexcept
    {
        return this->_M_state._M_remaining;
    }

    [[nodiscard]] bool
    done () const noexcept
    {
        return this->status () != Status::OK || this->_M_state._M_remaining == 0;
    }

    [[nodiscard]] const PacklowState &
    state () const noexcept
    {
        return this->_M_state;
    }

  private:
    PacklowState _M_state{};
    const uint8_t *_M_stream = nullptr;
    const uint8_t *_M_dict = nullptr;
    uint16_t _M_dict_len = 0;
};

/**
 * @brief Heap-free packlow encoder.
 *
 * The source is read in place (normally Flash-resident weights) and the packed
 * bytes are streamed to a caller-owned sink.  The match finder uses bounded
 * tables over the decoder's 64-byte window, so the encoder never allocates and
 * its RAM footprint does not depend on the input size.
 *
 * `begin()` writes the header immediately; `pump()` advances the token stream by
 * at most the requested number of source bytes; `finish()` flushes the range
 * coder.  `run()` does all three.
 */
class PacklowEncoder
{
  public:
    PacklowEncoder () noexcept = default;

    /** @brief One-shot encode into a caller-owned buffer; returns 0 on failure. */
    [[nodiscard]] static uint32_t
    encode (const uint8_t *source, const uint32_t length, const PacklowOptions &options,
            uint8_t *output, const uint32_t capacity) noexcept
    {
        if (output == nullptr || capacity < _S_pk_header_bytes)
            return 0;
        if (source == nullptr && length != 0)
            return 0;
        PacklowBufferSink buffer{ output, capacity, 0, false };
        PacklowEncoder encoder;
        const PacklowSink sink{ &buffer, _S_pk_buffer_write };
        if (encoder.begin (source, length, options, sink) != Status::OK)
            return 0;
        if (encoder.run () != Status::OK || buffer._M_overflow)
            return 0;
        return buffer._M_size;
    }

    [[nodiscard]] Status
    begin (const uint8_t *source, const uint32_t length, const PacklowOptions &options,
           const PacklowSink sink) noexcept
    {
        this->_M_status = static_cast<uint8_t> (Status::OK);
        this->_M_finished = false;
        this->_M_source = source;
        this->_M_length = length;
        this->_M_position = 0;
        this->_M_output_size = 0;
        this->_M_options = options;
        this->_M_sink = sink;
        this->_M_low = 0;
        this->_M_range = 0xFFFFFFFFu;
        this->_M_cache = 0;
        this->_M_cache_size = 1;
        this->_M_stage_len = 0;
        for (uint32_t i = 0; i < _S_PK_PROB_COUNT; ++i)
            this->_M_prob[i] = _S_pk_prob_init;
        for (uint32_t i = 0; i < 256; ++i)
            {
                this->_M_head[i] = 0;
                this->_M_first[i] = 0;
                this->_M_dict_head[i] = 0;
            }
        for (uint32_t i = 0; i < _S_pk_window_bytes; ++i)
            {
                this->_M_prev[i] = 0;
                this->_M_first_prev[i] = 0;
            }
        if (sink._M_write == nullptr || (source == nullptr && length != 0))
            return this->_M_fail (Status::BAD_LAYOUT);
        if (options._M_dict_len != 0 && options._M_dict == nullptr)
            return this->_M_fail (Status::BAD_LAYOUT);
        this->_M_build_dict_index ();
        this->_M_write_header ();
        return this->status ();
    }

    /** @brief Encode at most `byte_budget` source bytes; 0 means until the end. */
    [[nodiscard]] Status
    pump (const uint32_t byte_budget) noexcept
    {
        if (this->_M_status != static_cast<uint8_t> (Status::OK))
            return this->status ();
        const uint32_t budget = byte_budget == 0 ? 0xFFFFFFFFu : byte_budget;
        uint32_t consumed = 0;
        while (this->_M_position < this->_M_length && consumed < budget
               && this->_M_status == static_cast<uint8_t> (Status::OK))
            {
                const uint32_t before = this->_M_position;
                this->_M_encode_at (this->_M_position);
                consumed += this->_M_position - before;
            }
        return this->status ();
    }

    /** @brief Flush the range coder and the output stage. */
    [[nodiscard]] Status
    finish () noexcept
    {
        if (this->_M_finished)
            return this->status ();
        this->_M_finished = true;
        if (this->_M_status == static_cast<uint8_t> (Status::OK))
            {
                for (unsigned int i = 0; i < 5; ++i)
                    this->_M_shift_low ();
                this->_M_flush_stage ();
            }
        return this->status ();
    }

    /** @brief Encode the whole source and finish. */
    [[nodiscard]] Status
    run () noexcept
    {
        const Status pumped = this->pump (0);
        if (pumped != Status::OK)
            return pumped;
        return this->finish ();
    }

    [[nodiscard]] Status
    status () const noexcept
    {
        return static_cast<Status> (this->_M_status);
    }

    [[nodiscard]] bool
    done () const noexcept
    {
        return this->_M_finished || this->_M_status != static_cast<uint8_t> (Status::OK);
    }

    [[nodiscard]] uint32_t
    encoded_size () const noexcept
    {
        return this->_M_output_size;
    }

    [[nodiscard]] uint32_t
    position () const noexcept
    {
        return this->_M_position;
    }

    [[nodiscard]] uint32_t
    source_length () const noexcept
    {
        return this->_M_length;
    }

  private:
    [[nodiscard]] Status
    _M_fail (const Status status) noexcept
    {
        this->_M_status = static_cast<uint8_t> (status);
        return status;
    }

    [[nodiscard]] bool
    _M_ok () const noexcept
    {
        return this->_M_status == static_cast<uint8_t> (Status::OK);
    }

    void
    _M_put (const uint8_t byte) noexcept
    {
        if (!this->_M_ok ())
            return;
        this->_M_stage[this->_M_stage_len++] = byte;
        ++this->_M_output_size;
        if (this->_M_stage_len == _S_pk_stage_bytes)
            this->_M_flush_stage ();
    }

    void
    _M_flush_stage () noexcept
    {
        if (this->_M_stage_len == 0)
            return;
        if (!this->_M_ok ())
            {
                this->_M_stage_len = 0;
                return;
            }
        if (!this->_M_sink._M_write (this->_M_sink._M_context, this->_M_stage,
                                     this->_M_stage_len))
            {
                this->_M_status = static_cast<uint8_t> (Status::OUT_OF_RANGE);
                this->_M_stage_len = 0;
                return;
            }
        this->_M_stage_len = 0;
    }

    void
    _M_write_header () noexcept
    {
        const uint8_t flags =
            static_cast<uint8_t> ((this->_M_options._M_delta ? _S_pk_flag_delta : 0u)
                                  | (this->_M_options._M_dict_copy ? _S_pk_flag_dict : 0u));
        this->_M_put ('M');
        this->_M_put ('L');
        this->_M_put ('P');
        this->_M_put ('K');
        this->_M_put (_S_pk_version);
        this->_M_put (flags);
        const uint32_t raw = this->_M_length;
        this->_M_put (static_cast<uint8_t> (raw & 0xFFu));
        this->_M_put (static_cast<uint8_t> ((raw >> 8) & 0xFFu));
        this->_M_put (static_cast<uint8_t> ((raw >> 16) & 0xFFu));
        this->_M_put (static_cast<uint8_t> ((raw >> 24) & 0xFFu));
        const uint16_t crc =
            this->_M_source != nullptr || this->_M_length == 0
                ? _S_crc16 (this->_M_source, this->_M_length)
                : 0xFFFFu;
        this->_M_put (static_cast<uint8_t> (crc & 0xFFu));
        this->_M_put (static_cast<uint8_t> ((crc >> 8) & 0xFFu));
    }

    void
    _M_shift_low () noexcept
    {
        const uint32_t carry = static_cast<uint32_t> (this->_M_low >> 32);
        if (static_cast<uint32_t> (this->_M_low) < 0xFF000000u || carry != 0)
            {
                uint8_t temp = this->_M_cache;
                do
                    {
                        this->_M_put (static_cast<uint8_t> (temp + carry));
                        temp = 0xFF;
                    }
                while (--this->_M_cache_size != 0);
                this->_M_cache = static_cast<uint8_t> (this->_M_low >> 24);
            }
        ++this->_M_cache_size;
        this->_M_low = static_cast<uint64_t> (static_cast<uint32_t> (this->_M_low) << 8);
    }

    void
    _M_encode_bit (uint16_t &prob, const uint32_t bit) noexcept
    {
        if (!this->_M_ok ())
            return;
        const uint32_t bound = (this->_M_range >> 12) * static_cast<uint32_t> (prob);
        if (bit == 0)
            {
                this->_M_range = bound;
                prob = static_cast<uint16_t> (prob + ((_S_pk_prob_max + 1 - prob) >> _S_pk_prob_shift));
                if (prob > _S_pk_prob_max)
                    prob = _S_pk_prob_max;
            }
        else
            {
                this->_M_low += bound;
                this->_M_range -= bound;
                prob = static_cast<uint16_t> (prob - (prob >> _S_pk_prob_shift));
                if (prob < _S_pk_prob_min)
                    prob = _S_pk_prob_min;
            }
        while (this->_M_range < (1u << 24))
            {
                this->_M_range <<= 8;
                this->_M_shift_low ();
            }
    }

    void
    _M_encode_fixed_bit (const uint32_t bit) noexcept
    {
        uint16_t prob = _S_pk_prob_init;
        this->_M_encode_bit (prob, bit);
    }

    void
    _M_encode_token (const _S_PkToken token) noexcept
    {
        if (token == _S_PkToken::LITERAL || token == _S_PkToken::COPY)
            {
                this->_M_encode_bit (this->_M_prob[_S_PK_PROB_TOKEN0], 0);
                this->_M_encode_bit (this->_M_prob[_S_PK_PROB_TOKEN1],
                                     token == _S_PkToken::COPY ? 1u : 0u);
            }
        else
            {
                this->_M_encode_bit (this->_M_prob[_S_PK_PROB_TOKEN0], 1);
                this->_M_encode_bit (this->_M_prob[_S_PK_PROB_TOKEN2],
                                     token == _S_PkToken::DICT_COPY ? 1u : 0u);
            }
    }

    void
    _M_encode_literal (const uint8_t byte) noexcept
    {
        for (int shift = 7; shift >= 0; --shift)
            this->_M_encode_bit (this->_M_prob[_S_PK_PROB_LITERAL0 + (shift & 3)],
                                 (static_cast<uint32_t> (byte) >> shift) & 1u);
    }

    void
    _M_encode_gamma (const uint32_t value) noexcept
    {
        uint32_t prefix = 0;
        for (uint32_t probe = value; probe > 1; probe >>= 1)
            ++prefix;
        for (uint32_t i = 0; i < prefix; ++i)
            this->_M_encode_bit (this->_M_prob[_S_PK_PROB_GAMMA], 0);
        this->_M_encode_bit (this->_M_prob[_S_PK_PROB_GAMMA], 1);
        for (int shift = static_cast<int> (prefix) - 1; shift >= 0; --shift)
            this->_M_encode_fixed_bit ((value >> shift) & 1u);
    }

    void
    _M_encode_correction (const int32_t correction) noexcept
    {
        const uint32_t magnitude =
            static_cast<uint32_t> (correction < 0 ? -correction : correction);
        this->_M_encode_bit (this->_M_prob[_S_PK_PROB_DELTA_SIGN], correction < 0 ? 1u : 0u);
        this->_M_encode_bit (this->_M_prob[_S_PK_PROB_DELTA_MSB], (magnitude >> 3) & 1u);
        for (int shift = 2; shift >= 0; --shift)
            this->_M_encode_fixed_bit ((magnitude >> shift) & 1u);
    }

    [[nodiscard]] static constexpr uint32_t
    _M_hash (const uint8_t *data) noexcept
    {
        return (static_cast<uint32_t> (data[0]) * 251u
                + static_cast<uint32_t> (data[1]) * 31u + static_cast<uint32_t> (data[2]))
               & 0xFFu;
    }

    /** @brief Distance to the previous chain entry, or 0 when it is out of range. */
    [[nodiscard]] static uint8_t
    _M_chain_gap (const uint16_t current, const uint16_t previous) noexcept
    {
        if (previous == 0)
            return 0;
        const uint16_t distance = static_cast<uint16_t> (current - previous);
        return distance <= _S_pk_window_bytes ? static_cast<uint8_t> (distance) : 0;
    }

    void
    _M_build_dict_index () noexcept
    {
        if (!this->_M_options._M_dict_copy || this->_M_options._M_dict == nullptr
            || this->_M_options._M_dict_len < _S_pk_dict_block_bytes)
            return;
        for (uint32_t block = 0; block + _S_pk_dict_block_bytes <= this->_M_options._M_dict_len;
             block += _S_pk_dict_block_bytes)
            {
                const uint32_t hash = _M_hash (this->_M_options._M_dict + block);
                this->_M_dict_head[hash] =
                    static_cast<uint16_t> (block / _S_pk_dict_block_bytes + 1u);
            }
    }

    void
    _M_insert (const uint32_t position) noexcept
    {
        if (position + 2 >= this->_M_length)
            return;
        const uint16_t encoded = static_cast<uint16_t> (position + 1u);
        if (encoded == 0)
            return;
        const uint32_t hash = _M_hash (this->_M_source + position);
        const uint32_t first = this->_M_source[position];
        const uint32_t slot = position & (_S_pk_window_bytes - 1u);
        this->_M_prev[slot] = _M_chain_gap (encoded, this->_M_head[hash]);
        this->_M_head[hash] = encoded;
        this->_M_first_prev[slot] = _M_chain_gap (encoded, this->_M_first[first]);
        this->_M_first[first] = encoded;
    }

    [[nodiscard]] uint32_t
    _M_match (const uint32_t position, const uint32_t distance,
              const uint32_t limit) const noexcept
    {
        const uint8_t *const ahead = this->_M_source + position;
        const uint8_t *const behind = ahead - distance;
        uint32_t length = 0;
        while (length < limit && ahead[length] == behind[length])
            ++length;
        return length;
    }

    [[nodiscard]] uint32_t
    _M_delta_match (const uint32_t position, const uint32_t distance,
                    const uint32_t limit) const noexcept
    {
        uint32_t length = 0;
        while (length < limit)
            {
                const int base =
                    static_cast<int> (this->_M_source[position - distance + length]);
                int difference = static_cast<int> (this->_M_source[position + length]) - base;
                if (difference < -128)
                    difference += 256;
                else if (difference > 127)
                    difference -= 256;
                if (difference < -15 || difference > 15)
                    break;
                ++length;
            }
        return length;
    }

    [[nodiscard]] uint32_t
    _M_dict_match (const uint32_t position, const uint32_t block,
                   const uint32_t limit) const noexcept
    {
        const uint8_t *const dict = this->_M_options._M_dict;
        const uint32_t dict_len = this->_M_options._M_dict_len;
        uint32_t length = 0;
        while (length < limit && block + length < dict_len
               && this->_M_source[position + length] == dict[block + length])
            ++length;
        return length;
    }

    [[nodiscard]] uint32_t
    _M_search_copy (const uint32_t position, uint32_t &best_distance) const noexcept
    {
        best_distance = 0;
        if (position + 2 >= this->_M_length)
            return 0;
        uint32_t limit = this->_M_length - position;
        if (limit > _S_pk_max_match)
            limit = _S_pk_max_match;
        const uint32_t hash = _M_hash (this->_M_source + position);
        const uint16_t current = static_cast<uint16_t> (position + 1u);
        uint16_t candidate = this->_M_head[hash];
        uint32_t best = 0;
        for (uint32_t step = 0; candidate != 0 && step < _S_pk_chain_max; ++step)
            {
                const uint32_t distance = static_cast<uint16_t> (current - candidate);
                if (distance == 0 || distance > _S_pk_window_bytes)
                    break;
                const uint32_t length = this->_M_match (position, distance, limit);
                if (length > best)
                    {
                        best = length;
                        best_distance = distance;
                        if (best == limit)
                            break;
                    }
                const uint8_t gap =
                    this->_M_prev[(position - distance) & (_S_pk_window_bytes - 1u)];
                if (gap == 0)
                    break;
                candidate = static_cast<uint16_t> (candidate - gap);
            }
        return best;
    }

    [[nodiscard]] uint32_t
    _M_search_delta (const uint32_t position, uint32_t &best_distance) const noexcept
    {
        best_distance = 0;
        if (position == 0)
            return 0;
        uint32_t limit = this->_M_length - position;
        if (limit > _S_pk_window_bytes)
            limit = _S_pk_window_bytes;
        const uint16_t current = static_cast<uint16_t> (position + 1u);
        uint16_t candidate = this->_M_first[this->_M_source[position]];
        uint32_t best = 0;
        for (uint32_t step = 0; candidate != 0 && step < _S_pk_chain_max; ++step)
            {
                const uint32_t distance = static_cast<uint16_t> (current - candidate);
                if (distance == 0 || distance > _S_pk_window_bytes)
                    break;
                const uint32_t length = this->_M_delta_match (position, distance, limit);
                if (length > best)
                    {
                        best = length;
                        best_distance = distance;
                        if (best == limit)
                            break;
                    }
                const uint8_t gap =
                    this->_M_first_prev[(position - distance) & (_S_pk_window_bytes - 1u)];
                if (gap == 0)
                    break;
                candidate = static_cast<uint16_t> (candidate - gap);
            }
        return best;
    }

    [[nodiscard]] uint32_t
    _M_search_dict (const uint32_t position, const uint32_t limit,
                    uint32_t &block_out) const noexcept
    {
        block_out = 0;
        if (!this->_M_options._M_dict_copy || this->_M_options._M_dict == nullptr
            || this->_M_options._M_dict_len < _S_pk_dict_block_bytes
            || position + 2 >= this->_M_length)
            return 0;
        const uint16_t encoded = this->_M_dict_head[_M_hash (this->_M_source + position)];
        if (encoded == 0)
            return 0;
        const uint32_t block =
            (static_cast<uint32_t> (encoded) - 1u) * _S_pk_dict_block_bytes;
        if (block >= this->_M_options._M_dict_len)
            return 0;
        const uint32_t length = this->_M_dict_match (position, block, limit);
        if (length != 0)
            block_out = block;
        return length;
    }

    void
    _M_encode_at (const uint32_t position) noexcept
    {
        uint32_t copy_distance = 0;
        uint32_t delta_distance = 0;
        uint32_t dict_block = 0;

        const uint32_t copy = this->_M_search_copy (position, copy_distance);
        const uint32_t delta =
            this->_M_options._M_delta ? this->_M_search_delta (position, delta_distance) : 0u;
        uint32_t dict_limit = this->_M_length - position;
        if (dict_limit > _S_pk_max_match)
            dict_limit = _S_pk_max_match;
        const uint32_t dict = this->_M_search_dict (position, dict_limit, dict_block);

        const bool use_dict = dict >= 3u && dict >= copy + 2u;
        const bool use_delta = !use_dict && delta >= 4u && delta > copy + 2u;
        const bool use_copy = !use_dict && !use_delta && copy >= 3u;

        if (use_dict)
            {
                this->_M_encode_token (_S_PkToken::DICT_COPY);
                this->_M_encode_gamma (dict_block / _S_pk_dict_block_bytes + 1u);
                this->_M_encode_gamma (dict);
                if (this->_M_options._M_stats != nullptr)
                    ++this->_M_options._M_stats->_M_dict;
                for (uint32_t i = 0; i < dict && this->_M_ok (); ++i)
                    this->_M_insert (position + i);
                this->_M_position = position + dict;
            }
        else if (use_delta)
            {
                this->_M_encode_token (_S_PkToken::DELTA);
                this->_M_encode_gamma (delta_distance);
                this->_M_encode_gamma (delta);
                for (uint32_t i = 0; i < delta && this->_M_ok (); ++i)
                    {
                        int difference =
                            static_cast<int> (this->_M_source[position + i])
                            - static_cast<int> (this->_M_source[position - delta_distance + i]);
                        if (difference < -128)
                            difference += 256;
                        else if (difference > 127)
                            difference -= 256;
                        this->_M_encode_correction (difference);
                    }
                if (this->_M_options._M_stats != nullptr)
                    ++this->_M_options._M_stats->_M_delta;
                for (uint32_t i = 0; i < delta && this->_M_ok (); ++i)
                    this->_M_insert (position + i);
                this->_M_position = position + delta;
            }
        else if (use_copy)
            {
                this->_M_encode_token (_S_PkToken::COPY);
                this->_M_encode_gamma (copy_distance);
                this->_M_encode_gamma (copy);
                if (this->_M_options._M_stats != nullptr)
                    ++this->_M_options._M_stats->_M_copy;
                for (uint32_t i = 0; i < copy && this->_M_ok (); ++i)
                    this->_M_insert (position + i);
                this->_M_position = position + copy;
            }
        else
            {
                this->_M_encode_token (_S_PkToken::LITERAL);
                this->_M_encode_literal (this->_M_source[position]);
                if (this->_M_options._M_stats != nullptr)
                    ++this->_M_options._M_stats->_M_literal;
                this->_M_insert (position);
                this->_M_position = position + 1u;
            }
    }

    PacklowSink _M_sink{};
    const uint8_t *_M_source = nullptr;
    uint32_t _M_length = 0;
    uint32_t _M_position = 0;
    uint32_t _M_output_size = 0;
    uint64_t _M_low = 0;
    uint32_t _M_range = 0xFFFFFFFFu;
    uint32_t _M_cache_size = 1;
    uint8_t _M_cache = 0;
    uint8_t _M_status = static_cast<uint8_t> (Status::OK);
    uint16_t _M_head[256]{};
    uint16_t _M_first[256]{};
    uint16_t _M_dict_head[256]{};
    uint8_t _M_prev[_S_pk_window_bytes]{};
    uint8_t _M_first_prev[_S_pk_window_bytes]{};
    uint16_t _M_prob[_S_PK_PROB_COUNT]{};
    uint8_t _M_stage[_S_pk_stage_bytes]{};
    uint32_t _M_stage_len = 0;
    PacklowOptions _M_options{};
    bool _M_finished = false;
};
} // namespace zzs::mllee
