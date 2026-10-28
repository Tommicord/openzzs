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

#include "mllee/mllee.hpp"

#include <cstdint>
#include <cstring>
#include <vector>

namespace mllee_test
{
struct LayerSpec
{
    zzs::mllee::Op _M_op = zzs::mllee::Op::LINEAR;
    zzs::mllee::Act _M_act = zzs::mllee::Act::NONE;
    zzs::mllee::Dtype _M_dtype = zzs::mllee::Dtype::I8;
    zzs::mllee::Codec _M_codec = zzs::mllee::Codec::RAW;
    uint16_t _M_in_dim = 0;
    uint16_t _M_out_dim = 0;
    uint8_t _M_kernel = 1;
    uint8_t _M_stride = 1;
    uint8_t _M_groups = 0;
    uint8_t _M_group_size = 0;
    uint8_t _M_head_dim = 0;
    uint8_t _M_seq = 0;
    uint8_t _M_flags = 0;
    uint8_t _M_aux = 0;
    uint16_t _M_in_off = zzs::mllee::_S_external_input;
    uint16_t _M_out_off = 0;
    uint16_t _M_in_len = 0;
    uint16_t _M_out_len = 0;
    /** @brief Arena offset of the second tensor operand (ADD, MATMUL, ...). */
    uint16_t _M_in2_off = 0;
    /** @brief Length in bytes of the second tensor operand. */
    uint16_t _M_in2_len = 0;
    std::int32_t _M_requant_mult = 32768;
    std::vector<std::int8_t> _M_weights;
    std::vector<std::int32_t> _M_multipliers;
    std::vector<std::int16_t> _M_scale16;
    std::vector<std::int16_t> _M_bias;
    std::vector<std::int8_t> _M_lut;
    bool _M_use_packlow = false;
};

class BlobBuilder
{
  public:
    explicit BlobBuilder (std::vector<LayerSpec> layers, std::vector<uint8_t> dictionary = {})
        : _M_layers (std::move (layers)), _M_dictionary (std::move (dictionary))
    {
    }

    [[nodiscard]] std::vector<uint8_t>
    build (const uint16_t arena_bytes) const
    {
        std::vector<uint8_t> blob (zzs::mllee::_S_model_header_size, 0);
        const uint32_t layer_table = static_cast<uint32_t> (blob.size ());
        blob.resize (blob.size () + _M_layers.size () * zzs::mllee::_S_layer_desc_size, 0);
        const uint32_t scale_table = static_cast<uint32_t> (blob.size ());

        std::vector<uint32_t> scale_offsets (_M_layers.size (), 0);
        for (std::size_t i = 0; i < _M_layers.size (); ++i)
            {
                if (_M_layers[i]._M_multipliers.empty () && _M_layers[i]._M_scale16.empty ())
                    continue;
                scale_offsets[i] = static_cast<uint32_t> (blob.size ());
                if (!_M_layers[i]._M_scale16.empty ())
                    for (const std::int16_t value : _M_layers[i]._M_scale16)
                        _put_i16 (blob, value);
                else
                    for (const std::int32_t value : _M_layers[i]._M_multipliers)
                        _put_i32 (blob, value);
            }

        std::vector<uint32_t> bias_offsets (_M_layers.size (), 0);
        for (std::size_t i = 0; i < _M_layers.size (); ++i)
            {
                if (_M_layers[i]._M_bias.empty ())
                    continue;
                bias_offsets[i] = static_cast<uint32_t> (blob.size ());
                for (const std::int16_t value : _M_layers[i]._M_bias)
                    _put_i16 (blob, value);
            }

        std::vector<uint32_t> lut_offsets (_M_layers.size (), 0);
        for (std::size_t i = 0; i < _M_layers.size (); ++i)
            {
                if (_M_layers[i]._M_lut.empty ())
                    continue;
                lut_offsets[i] = static_cast<uint32_t> (blob.size ());
                for (const std::int8_t value : _M_layers[i]._M_lut)
                    blob.push_back (static_cast<uint8_t> (value));
            }

        uint32_t dictionary_offset = 0;
        if (!_M_dictionary.empty ())
            {
                dictionary_offset = static_cast<uint32_t> (blob.size ());
                blob.insert (blob.end (), _M_dictionary.begin (), _M_dictionary.end ());
            }

        std::vector<uint32_t> weight_offsets (_M_layers.size (), 0);
        std::vector<uint32_t> weight_lengths (_M_layers.size (), 0);
        std::vector<zzs::mllee::Codec> codecs (_M_layers.size (), zzs::mllee::Codec::RAW);
        for (std::size_t i = 0; i < _M_layers.size (); ++i)
            {
                const LayerSpec &layer = _M_layers[i];
                if (layer._M_weights.empty ())
                    continue;
                std::vector<uint8_t> raw = _pack_weights (layer);
                weight_offsets[i] = static_cast<uint32_t> (blob.size ());
                if (layer._M_use_packlow)
                    {
                        std::vector<uint8_t> encoded (raw.size () * 4 + 512);
                        zzs::mllee::PacklowOptions options;
                        options._M_delta = true;
                        options._M_dict_copy = !_M_dictionary.empty ();
                        if (!_M_dictionary.empty ())
                            {
                                options._M_dict = _M_dictionary.data ();
                                options._M_dict_len =
                                    static_cast<uint16_t> (_M_dictionary.size ());
                            }
                        const uint32_t size = zzs::mllee::PacklowEncoder::encode (
                            raw.data (), static_cast<uint32_t> (raw.size ()), options,
                            encoded.data (), static_cast<uint32_t> (encoded.size ()));
                        if (size != 0 && size < raw.size ())
                            {
                                blob.insert (blob.end (), encoded.begin (), encoded.begin () + size);
                                codecs[i] = zzs::mllee::Codec::PACKLOW;
                                weight_lengths[i] = size;
                                continue;
                            }
                    }
                blob.insert (blob.end (), raw.begin (), raw.end ());
                codecs[i] = zzs::mllee::Codec::RAW;
                weight_lengths[i] = static_cast<uint32_t> (raw.size ());
            }

        for (std::size_t i = 0; i < _M_layers.size (); ++i)
            {
                const uint32_t offset =
                    layer_table + static_cast<uint32_t> (i) * zzs::mllee::_S_layer_desc_size;
                _write_layer (blob, offset, _M_layers[i], scale_offsets[i], bias_offsets[i],
                              lut_offsets[i], weight_offsets[i], weight_lengths[i], codecs[i]);
            }

        blob[0] = 'M';
        blob[1] = 'L';
        blob[2] = 'E';
        blob[3] = 'E';
        blob[4] = zzs::mllee::_S_model_version;
        blob[5] = 0;
        blob[6] = static_cast<uint8_t> (_M_layers.size ());
        blob[7] = static_cast<uint8_t> (zzs::mllee::_S_model_header_size);
        _write_u32 (blob, 8, layer_table);
        _write_u32 (blob, 12, scale_table);
        _write_u32 (blob, 16, dictionary_offset);
        _write_u16 (blob, 20, static_cast<uint16_t> (_M_dictionary.size ()));
        _write_u16 (blob, 22, arena_bytes);
        blob[24] = static_cast<uint8_t> (zzs::mllee::Dtype::I8);
        blob[25] = zzs::mllee::_S_op_count;
        _write_u16 (blob, 26, 0);
        _write_u16 (blob, 28, zzs::mllee::_S_crc16 (blob.data (), 28));
        _write_u16 (blob, 30, 0);
        return blob;
    }

  private:
    [[nodiscard]] static std::vector<uint8_t>
    _pack_weights (const LayerSpec &layer)
    {
        std::vector<uint8_t> raw;
        if (layer._M_dtype == zzs::mllee::Dtype::I4 || layer._M_dtype == zzs::mllee::Dtype::F4E2M1)
            {
                for (std::size_t i = 0; i < layer._M_weights.size (); i += 2)
                    {
                        const std::int8_t low = layer._M_weights[i];
                        const std::int8_t high =
                            i + 1 < layer._M_weights.size () ? layer._M_weights[i + 1] : 0;
                        raw.push_back (zzs::mllee::_S_pack_int4 (low, high));
                    }
            }
        else
            {
                for (const std::int8_t value : layer._M_weights)
                    raw.push_back (static_cast<uint8_t> (value));
            }
        return raw;
    }

    static void
    _write_layer (std::vector<uint8_t> &blob, const uint32_t offset,
                  const LayerSpec &layer, const uint32_t scale_off, const uint32_t bias_off,
                  const uint32_t lut_off, const uint32_t weight_off,
                  const uint32_t weight_len, const zzs::mllee::Codec codec)
    {
        blob[offset + 0] = static_cast<uint8_t> (layer._M_op);
        blob[offset + 1] = static_cast<uint8_t> (layer._M_act);
        blob[offset + 2] = static_cast<uint8_t> (layer._M_dtype);
        blob[offset + 3] = static_cast<uint8_t> (codec);
        _write_u16 (blob, offset + 4, layer._M_in_dim);
        _write_u16 (blob, offset + 6, layer._M_out_dim);
        blob[offset + 8] = layer._M_kernel;
        blob[offset + 9] = layer._M_stride;
        blob[offset + 10] =
            layer._M_groups != 0
                ? layer._M_groups
                : (layer._M_group_size != 0
                       ? static_cast<uint8_t> ((layer._M_in_dim + layer._M_group_size - 1)
                                                    / layer._M_group_size)
                       : 0);
        blob[offset + 11] = layer._M_group_size;
        blob[offset + 12] = layer._M_head_dim;
        blob[offset + 13] = layer._M_seq;
        blob[offset + 14] = layer._M_flags;
        blob[offset + 15] = layer._M_aux;
        /* Two-operand tensor ops reuse the weight slot for the arena-resident
           right-hand operand, exactly like the runtime decoder does. */
        const bool second_operand = zzs::mllee::_S_two_operand (layer._M_op);
        _write_u32 (blob, offset + 16,
                    second_operand && weight_off == 0 ? layer._M_in2_off : weight_off);
        _write_u32 (blob, offset + 20,
                    second_operand && weight_len == 0 ? layer._M_in2_len : weight_len);
        _write_u32 (blob, offset + 24, scale_off);
        _write_u32 (blob, offset + 28, bias_off);
        _write_u32 (blob, offset + 32, lut_off);
        _write_u16 (blob, offset + 36, layer._M_in_off);
        _write_u16 (blob, offset + 38, layer._M_out_off);
        const uint16_t in_len = layer._M_in_len != 0 ? layer._M_in_len : layer._M_in_dim;
        const uint16_t out_len = layer._M_out_len != 0 ? layer._M_out_len : layer._M_out_dim;
        _write_u16 (blob, offset + 40, in_len);
        _write_u16 (blob, offset + 42, out_len);
        _write_u32 (blob, offset + 44, static_cast<uint32_t> (layer._M_requant_mult));
    }

    static void
    _put_i32 (std::vector<uint8_t> &blob, const std::int32_t value)
    {
        for (int shift = 0; shift < 32; shift += 8)
            blob.push_back (static_cast<uint8_t> ((static_cast<uint32_t> (value) >> shift) & 0xFFu));
    }

    static void
    _put_i16 (std::vector<uint8_t> &blob, const std::int16_t value)
    {
        blob.push_back (static_cast<uint8_t> (static_cast<uint16_t> (value) & 0xFFu));
        blob.push_back (static_cast<uint8_t> ((static_cast<uint16_t> (value) >> 8) & 0xFFu));
    }

    static void
    _write_u16 (std::vector<uint8_t> &blob, const uint32_t offset, const uint16_t value)
    {
        blob[offset] = static_cast<uint8_t> (value & 0xFFu);
        blob[offset + 1] = static_cast<uint8_t> ((value >> 8) & 0xFFu);
    }

    static void
    _write_u32 (std::vector<uint8_t> &blob, const uint32_t offset, const uint32_t value)
    {
        blob[offset] = static_cast<uint8_t> (value & 0xFFu);
        blob[offset + 1] = static_cast<uint8_t> ((value >> 8) & 0xFFu);
        blob[offset + 2] = static_cast<uint8_t> ((value >> 16) & 0xFFu);
        blob[offset + 3] = static_cast<uint8_t> ((value >> 24) & 0xFFu);
    }

    std::vector<LayerSpec> _M_layers;
    std::vector<uint8_t> _M_dictionary;
};

inline std::vector<LayerSpec>
make_pipeline ()
{
    std::vector<LayerSpec> layers;

    LayerSpec conv;
    conv._M_op = zzs::mllee::Op::CONV1D;
    conv._M_in_dim = 8;
    conv._M_out_dim = 16;
    conv._M_head_dim = 8;
    conv._M_seq = 8;
    conv._M_kernel = 3;
    conv._M_stride = 1;
    conv._M_flags = static_cast<uint8_t> (zzs::mllee::INPUT_EXTERNAL | zzs::mllee::CONV_CAUSAL);
    conv._M_in_off = zzs::mllee::_S_external_input;
    conv._M_out_off = 0;
    conv._M_in_len = 8;
    conv._M_out_len = 16;
    conv._M_weights = { 1, 0, -1, 1, 1, 1 };
    conv._M_multipliers = { 32768, 32768 };
    conv._M_bias = { 0, 0 };
    layers.push_back (conv);

    for (int branch = 0; branch < 3; ++branch)
        {
            LayerSpec linear;
            linear._M_op = zzs::mllee::Op::LINEAR;
            linear._M_in_dim = 16;
            linear._M_out_dim = 12;
            linear._M_in_off = 0;
            linear._M_out_off = static_cast<uint16_t> (16 + branch * 12);
            linear._M_in_len = 16;
            linear._M_out_len = 12;
            linear._M_requant_mult = 32768;
            linear._M_weights.resize (16 * 12);
            for (std::size_t i = 0; i < linear._M_weights.size (); ++i)
                linear._M_weights[i] =
                    static_cast<std::int8_t> ((i * 37 + static_cast<std::size_t> (branch) * 53) % 7) - 3;
            layers.push_back (linear);
        }

    LayerSpec attention;
    attention._M_op = zzs::mllee::Op::ATTENTION;
    attention._M_in_dim = 12;
    attention._M_out_dim = 12;
    attention._M_head_dim = 6;
    attention._M_seq = 2;
    attention._M_in_off = 16;
    attention._M_out_off = 52;
    attention._M_in_len = 36;
    attention._M_out_len = 12;
    layers.push_back (attention);

    LayerSpec dense;
    dense._M_op = zzs::mllee::Op::LINEAR;
    dense._M_in_dim = 12;
    dense._M_out_dim = 4;
    dense._M_in_off = 52;
    dense._M_out_off = 64;
    dense._M_in_len = 12;
    dense._M_out_len = 4;
    dense._M_requant_mult = 32768;
    dense._M_weights.resize (12 * 4);
    for (std::size_t i = 0; i < dense._M_weights.size (); ++i)
        dense._M_weights[i] = static_cast<std::int8_t> ((i * 7) % 3) - 1;
    layers.push_back (dense);

    LayerSpec softmax;
    softmax._M_op = zzs::mllee::Op::SOFTMAX;
    softmax._M_in_dim = 4;
    softmax._M_out_dim = 4;
    softmax._M_in_off = 64;
    softmax._M_out_off = 64;
    softmax._M_in_len = 4;
    softmax._M_out_len = 4;
    layers.push_back (softmax);

    return layers;
}
} // namespace mllee_test
