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

#include "mllee/onnx.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace
{
namespace onnx = zzs::mllee::onnx;
using Bytes = std::vector<uint8_t>;

/* ------------------------------------------------------------------------- */
/* Minimal protobuf writer.  The fixtures are hand-built so the test does not
   depend on protoc or on an ONNX Python runtime. */
/* ------------------------------------------------------------------------- */

void
put_varint (Bytes &out, uint64_t value)
{
    while (value >= 0x80u)
        {
            out.push_back (static_cast<uint8_t> (value | 0x80u));
            value >>= 7;
        }
    out.push_back (static_cast<uint8_t> (value));
}

void
put_tag (Bytes &out, const uint32_t field, const uint32_t wire)
{
    put_varint (out, (static_cast<uint64_t> (field) << 3) | wire);
}

void
put_uint (Bytes &out, const uint32_t field, const uint64_t value)
{
    put_tag (out, field, 0);
    put_varint (out, value);
}

void
put_fixed32 (Bytes &out, const uint32_t field, const uint32_t value)
{
    put_tag (out, field, 5);
    out.push_back (static_cast<uint8_t> (value));
    out.push_back (static_cast<uint8_t> (value >> 8));
    out.push_back (static_cast<uint8_t> (value >> 16));
    out.push_back (static_cast<uint8_t> (value >> 24));
}

void
put_bytes (Bytes &out, const uint32_t field, const uint8_t *const data,
           const size_t size)
{
    put_tag (out, field, 2);
    put_varint (out, size);
    out.insert (out.end (), data, data + size);
}

void
put_string (Bytes &out, const uint32_t field, const std::string_view text)
{
    put_bytes (out, field, reinterpret_cast<const uint8_t *> (text.data ()),
               text.size ());
}

void
put_message (Bytes &out, const uint32_t field, const Bytes &message)
{
    put_bytes (out, field, message.data (), message.size ());
}

void
put_packed_varints (Bytes &out, const uint32_t field,
                    const std::vector<int64_t> &values)
{
    Bytes packed;
    for (const int64_t value : values)
        put_varint (packed, static_cast<uint64_t> (value));
    put_message (out, field, packed);
}

void
put_packed_fixed32 (Bytes &out, const uint32_t field,
                    const std::vector<uint32_t> &values)
{
    Bytes packed;
    for (const uint32_t value : values)
        {
            packed.push_back (static_cast<uint8_t> (value));
            packed.push_back (static_cast<uint8_t> (value >> 8));
            packed.push_back (static_cast<uint8_t> (value >> 16));
            packed.push_back (static_cast<uint8_t> (value >> 24));
        }
    put_message (out, field, packed);
}

/* ------------------------------------------------------------------------- */
/* ONNX message builders. */
/* ------------------------------------------------------------------------- */

Bytes
make_tensor_raw (const uint32_t dtype, const std::vector<int64_t> &dims,
                 const std::string_view name, const Bytes &raw)
{
    Bytes tensor;
    for (const int64_t dim : dims)
        put_uint (tensor, 1, static_cast<uint64_t> (dim));
    put_uint (tensor, 2, dtype);
    if (!name.empty ())
        put_string (tensor, 8, name);
    if (!raw.empty ())
        put_bytes (tensor, 9, raw.data (), raw.size ());
    return tensor;
}

Bytes
make_tensor_int32 (const uint32_t dtype, const std::vector<int64_t> &dims,
                   const std::string_view name,
                   const std::vector<int64_t> &values)
{
    Bytes tensor;
    for (const int64_t dim : dims)
        put_uint (tensor, 1, static_cast<uint64_t> (dim));
    put_uint (tensor, 2, dtype);
    if (!name.empty ())
        put_string (tensor, 8, name);
    put_packed_varints (tensor, 5, values);
    return tensor;
}

Bytes
make_tensor_float (const std::vector<int64_t> &dims,
                   const std::string_view name,
                   const std::vector<float> &values)
{
    Bytes tensor;
    for (const int64_t dim : dims)
        put_uint (tensor, 1, static_cast<uint64_t> (dim));
    put_uint (tensor, 2, static_cast<uint32_t> (onnx::DataType::FLOAT));
    if (!name.empty ())
        put_string (tensor, 8, name);
    std::vector<uint32_t> bits;
    bits.reserve (values.size ());
    for (const float value : values)
        {
            uint32_t raw = 0;
            std::memcpy (&raw, &value, sizeof (raw));
            bits.push_back (raw);
        }
    put_packed_fixed32 (tensor, 4, bits);
    return tensor;
}

Bytes
make_tensor_string (const std::vector<int64_t> &dims,
                    const std::string_view name,
                    const std::vector<std::string_view> &values)
{
    Bytes tensor;
    for (const int64_t dim : dims)
        put_uint (tensor, 1, static_cast<uint64_t> (dim));
    put_uint (tensor, 2, static_cast<uint32_t> (onnx::DataType::STRING));
    if (!name.empty ())
        put_string (tensor, 8, name);
    for (const std::string_view value : values)
        put_string (tensor, 6, value);
    return tensor;
}

Bytes
make_external_entry (const std::string_view key, const std::string_view value)
{
    Bytes entry;
    put_string (entry, 1, key);
    put_string (entry, 2, value);
    return entry;
}

Bytes
make_tensor_external (const uint32_t dtype,
                      const std::vector<int64_t> &dims,
                      const std::string_view name, const uint32_t length)
{
    Bytes tensor;
    for (const int64_t dim : dims)
        put_uint (tensor, 1, static_cast<uint64_t> (dim));
    put_uint (tensor, 2, dtype);
    if (!name.empty ())
        put_string (tensor, 8, name);
    put_message (tensor, 13,
                 make_external_entry ("location", "weights.bin"));
    put_message (tensor, 13, make_external_entry ("offset", "0"));
    put_message (tensor, 13,
                 make_external_entry ("length", std::to_string (length)));
    put_uint (tensor, 14, 1);
    return tensor;
}

Bytes
make_attribute_float (const std::string_view name, const float value)
{
    Bytes attribute;
    put_string (attribute, 1, name);
    uint32_t bits = 0;
    std::memcpy (&bits, &value, sizeof (bits));
    put_fixed32 (attribute, 2, bits);
    put_uint (attribute, 20, static_cast<uint32_t> (onnx::AttributeType::FLOAT));
    return attribute;
}

Bytes
make_attribute_int (const std::string_view name, const int64_t value)
{
    Bytes attribute;
    put_string (attribute, 1, name);
    put_uint (attribute, 3, static_cast<uint64_t> (value));
    put_uint (attribute, 20, static_cast<uint32_t> (onnx::AttributeType::INT));
    return attribute;
}

Bytes
make_attribute_tensor (const std::string_view name, const Bytes &tensor)
{
    Bytes attribute;
    put_string (attribute, 1, name);
    put_message (attribute, 5, tensor);
    put_uint (attribute, 20,
              static_cast<uint32_t> (onnx::AttributeType::TENSOR));
    return attribute;
}

Bytes
make_attribute_graph (const std::string_view name, const Bytes &graph)
{
    Bytes attribute;
    put_string (attribute, 1, name);
    put_message (attribute, 6, graph);
    put_uint (attribute, 20,
              static_cast<uint32_t> (onnx::AttributeType::GRAPH));
    return attribute;
}

Bytes
make_node (const std::string_view op_type,
           const std::vector<std::string_view> &inputs,
           const std::vector<std::string_view> &outputs,
           const std::vector<Bytes> &attributes,
           const std::string_view name = {})
{
    Bytes node;
    for (const std::string_view input : inputs)
        put_string (node, 1, input);
    for (const std::string_view output : outputs)
        put_string (node, 2, output);
    if (!name.empty ())
        put_string (node, 3, name);
    put_string (node, 4, op_type);
    for (const Bytes &attribute : attributes)
        put_message (node, 5, attribute);
    return node;
}

Bytes
make_value_info (const std::string_view name, const uint32_t dtype,
                 const std::vector<int64_t> &dims)
{
    Bytes value;
    put_string (value, 1, name);

    Bytes shape;
    for (const int64_t dim : dims)
        {
            Bytes dimension;
            put_uint (dimension, 1, static_cast<uint64_t> (dim));
            put_message (shape, 1, dimension);
        }

    Bytes tensor_type;
    put_uint (tensor_type, 1, dtype);
    put_message (tensor_type, 2, shape);

    Bytes type;
    put_message (type, 1, tensor_type);
    put_message (value, 2, type);
    return value;
}

Bytes
make_graph (const std::vector<Bytes> &nodes, const std::vector<Bytes> &initializers,
            const std::vector<Bytes> &inputs, const std::vector<Bytes> &outputs,
            const std::string_view name, const std::vector<Bytes> &value_info = {})
{
    Bytes graph;
    for (const Bytes &node : nodes)
        put_message (graph, 1, node);
    if (!name.empty ())
        put_string (graph, 2, name);
    for (const Bytes &initializer : initializers)
        put_message (graph, 5, initializer);
    for (const Bytes &input : inputs)
        put_message (graph, 11, input);
    for (const Bytes &output : outputs)
        put_message (graph, 12, output);
    for (const Bytes &value : value_info)
        put_message (graph, 13, value);
    return graph;
}

Bytes
make_model (const uint32_t ir_version, const uint32_t opset,
            const std::string_view producer, const Bytes &graph,
            const bool include_opset = true)
{
    Bytes model;
    put_uint (model, 1, ir_version);
    put_string (model, 2, producer);
    put_message (model, 7, graph);
    if (include_opset)
        {
            Bytes import;
            put_uint (import, 2, opset);
            put_message (model, 8, import);
        }
    return model;
}

/* ------------------------------------------------------------------------- */
/* Event capture and chunk sources. */
/* ------------------------------------------------------------------------- */

struct Captured
{
    onnx::EventKind _M_kind = onnx::EventKind::NONE;
    onnx::TensorRole _M_role = onnx::TensorRole::NONE;
    uint16_t _M_index = 0;
    uint32_t _M_offset = 0;
    uint64_t _M_value = 0;
    uint32_t _M_length = 0;
    uint8_t _M_flags = 0;
    onnx::ModelInfo _M_model{};
    onnx::NodeInfo _M_node{};
    onnx::TensorInfo _M_tensor{};
};

struct Capture
{
    std::vector<Captured> _M_events;
    onnx::EventKind _M_stop = onnx::EventKind::NONE;

    static bool
    callback (void *const context, const onnx::Event &event) noexcept
    {
        Capture *const self = static_cast<Capture *> (context);
        Captured saved{};
        saved._M_kind = event._M_kind;
        saved._M_role = event._M_role;
        saved._M_index = event._M_index;
        saved._M_offset = event._M_offset;
        saved._M_value = event._M_value;
        saved._M_length = event._M_length;
        saved._M_flags = event._M_flags;
        switch (event._M_kind)
            {
            case onnx::EventKind::MODEL_BEGIN:
            case onnx::EventKind::MODEL_END:
            case onnx::EventKind::DONE:
                if (event.model () != nullptr)
                    saved._M_model = *event.model ();
                break;
            case onnx::EventKind::NODE_BEGIN:
            case onnx::EventKind::NODE_INPUT:
            case onnx::EventKind::NODE_OUTPUT:
            case onnx::EventKind::NODE_END:
                if (event.node () != nullptr)
                    saved._M_node = *event.node ();
                break;
            case onnx::EventKind::TENSOR_BEGIN:
            case onnx::EventKind::TENSOR_DIM:
            case onnx::EventKind::TENSOR_DATA:
            case onnx::EventKind::TENSOR_END:
                if (event.tensor () != nullptr)
                    saved._M_tensor = *event.tensor ();
                break;
            default:
                break;
            }
        self->_M_events.push_back (saved);
        return event._M_kind != self->_M_stop;
    }

    [[nodiscard]] onnx::Visitor
    visitor () noexcept
    {
        return onnx::visitor (&callback, this);
    }

    [[nodiscard]] size_t
    count (const onnx::EventKind kind) const noexcept
    {
        size_t total = 0;
        for (const Captured &event : this->_M_events)
            if (event._M_kind == kind)
                ++total;
        return total;
    }

    [[nodiscard]] const Captured *
    first (const onnx::EventKind kind) const noexcept
    {
        for (const Captured &event : this->_M_events)
            if (event._M_kind == kind)
                return &event;
        return nullptr;
    }
};

onnx::Chunk
one_byte_chunk (void *const context, const uint32_t offset) noexcept
{
    const onnx::MemorySource *const source
        = static_cast<const onnx::MemorySource *> (context);
    if (source == nullptr || source->_M_data == nullptr
        || offset >= source->_M_size)
        return onnx::Chunk{};
    return onnx::Chunk{ source->_M_data + offset, 1 };
}

/* The standard two-tensor model used by most cases. */
Bytes
make_sample_model ()
{
    const std::vector<int64_t> weight_dims{ 2, 2 };
    const Bytes weights{ 1, 2, 3, 4 };
    const Bytes initializer
        = make_tensor_raw (static_cast<uint32_t> (onnx::DataType::INT8),
                           weight_dims, "w", weights);

    const std::vector<Bytes> attributes{
        make_attribute_float ("alpha", 0.5f),
        make_attribute_int ("transB", 1),
    };
    const Bytes node = make_node ("Gemm", { "x", "w" }, { "y" }, attributes,
                                  "fc");

    const Bytes input = make_value_info (
        "x", static_cast<uint32_t> (onnx::DataType::FLOAT), { 1, 2 });
    const Bytes output = make_value_info (
        "y", static_cast<uint32_t> (onnx::DataType::FLOAT), { 1, 2 });

    const Bytes graph
        = make_graph ({ node }, { initializer }, { input }, { output }, "g");
    return make_model (8, 13, "openzzs-test", graph);
}

} // namespace

TEST_CASE ("onnx: resident layout")
{
    STATIC_REQUIRE (sizeof (onnx::Loader) == 128u);
    STATIC_REQUIRE (sizeof (onnx::Stream) == 40u);
    STATIC_REQUIRE (sizeof (onnx::ModelInfo) == 32u);
    STATIC_REQUIRE (sizeof (onnx::NodeInfo) == 28u);
    STATIC_REQUIRE (sizeof (onnx::TensorInfo) == 28u);
    STATIC_REQUIRE (sizeof (onnx::Event) <= 32u);
    STATIC_REQUIRE (onnx::_S_static_budget_bytes == sizeof (onnx::ModelInfo));
}

TEST_CASE ("onnx: sample model streams to completion")
{
    const Bytes model = make_sample_model ();
    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };

    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()),
                  onnx::LOADER_STRICT | onnx::LOADER_ALLOW_TRAILING_BYTES);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (!error);
    CHECK (error.code () == onnx::OnnxCode::OK);

    CHECK (capture.count (onnx::EventKind::MODEL_BEGIN) == 1u);
    CHECK (capture.count (onnx::EventKind::OPSET) == 1u);
    CHECK (capture.count (onnx::EventKind::GRAPH_BEGIN) == 1u);
    CHECK (capture.count (onnx::EventKind::GRAPH_END) == 1u);
    CHECK (capture.count (onnx::EventKind::NODE_BEGIN) == 1u);
    CHECK (capture.count (onnx::EventKind::NODE_INPUT) == 2u);
    CHECK (capture.count (onnx::EventKind::NODE_OUTPUT) == 1u);
    CHECK (capture.count (onnx::EventKind::NODE_END) == 1u);
    CHECK (capture.count (onnx::EventKind::ATTRIBUTE_END) == 2u);
    CHECK (capture.count (onnx::EventKind::TENSOR_BEGIN) == 1u);
    CHECK (capture.count (onnx::EventKind::TENSOR_DIM) == 2u);
    CHECK (capture.count (onnx::EventKind::TENSOR_DATA) == 1u);
    CHECK (capture.count (onnx::EventKind::TENSOR_END) == 1u);
    CHECK (capture.count (onnx::EventKind::VALUE_END) == 2u);
    CHECK (capture.count (onnx::EventKind::MODEL_END) == 1u);
    CHECK (capture.count (onnx::EventKind::DONE) == 1u);

    const onnx::ModelInfo &info = loader.model ();
    CHECK (info._M_ir_version == 8u);
    CHECK (info._M_opset_version == 13u);
    CHECK (info._M_node_count == 1u);
    CHECK (info._M_initializer_count == 1u);
    CHECK (info._M_input_count == 1u);
    CHECK (info._M_output_count == 1u);
    CHECK (info._M_producer_hash == onnx::fnv1a_literal ("openzzs-test"));
    CHECK (info.has_graph ());

    const onnx::NodeInfo &node = loader.node ();
    CHECK (node._M_op_length == 4u);
    CHECK (node._M_op_hash == onnx::fnv1a_literal ("Gemm"));
    CHECK (node._M_op_hash2 == onnx::djb2_literal ("Gemm"));
    CHECK (node._M_input_count == 2u);
    CHECK (node._M_output_count == 1u);
    CHECK (node._M_attribute_count == 2u);
    CHECK (node.is_default_domain ());
    CHECK (onnx::lookup_op (node)._M_op == zzs::mllee::Op::LINEAR);

    const Captured *const data
        = capture.first (onnx::EventKind::TENSOR_DATA);
    REQUIRE (data != nullptr);
    CHECK (data->_M_role == onnx::TensorRole::INITIALIZER);
    CHECK (data->_M_length == 4u);
    CHECK (data->_M_tensor.dtype () == onnx::DataType::INT8);
    CHECK (data->_M_tensor._M_elements == 4u);
    CHECK (data->_M_tensor.data_field ()
           == onnx::TensorDataField::RAW_DATA);

    uint8_t weights[4]{};
    uint32_t copied = 0;
    REQUIRE (!loader.copy (data->_M_offset, weights, sizeof (weights), copied));
    CHECK (copied == 4u);
    CHECK (weights[0] == 1u);
    CHECK (weights[1] == 2u);
    CHECK (weights[2] == 3u);
    CHECK (weights[3] == 4u);
}

TEST_CASE ("onnx: one-byte chunks cross every boundary")
{
    const Bytes model = make_sample_model ();
    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };

    onnx::Loader loader;
    Capture capture;
    loader.begin (one_byte_chunk, &source,
                  static_cast<uint32_t> (model.size ()), onnx::LOADER_STRICT);
    REQUIRE (!loader.run (capture.visitor ()));
    CHECK (capture.count (onnx::EventKind::DONE) == 1u);

    const Captured *const data
        = capture.first (onnx::EventKind::TENSOR_DATA);
    REQUIRE (data != nullptr);
    uint8_t weights[4]{};
    uint32_t copied = 0;
    REQUIRE (!loader.copy (data->_M_offset, weights, 4u, copied));
    CHECK (copied == 4u);
    CHECK ((weights[0] == 1u && weights[3] == 4u));
}

TEST_CASE ("onnx: chunk list joins discontiguous regions")
{
    const Bytes model = make_sample_model ();
    const size_t split = model.size () / 2u;
    const onnx::Chunk chunks[] = {
        onnx::Chunk{ model.data (), static_cast<uint32_t> (split) },
        onnx::Chunk{ model.data () + split,
                     static_cast<uint32_t> (model.size () - split) },
    };
    onnx::ChunkList list{ chunks, 2u };

    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::chunk_list_chunk, &list,
                  static_cast<uint32_t> (model.size ()), onnx::LOADER_STRICT);
    REQUIRE (!loader.run (capture.visitor ()));
    CHECK (capture.count (onnx::EventKind::NODE_END) == 1u);
}

TEST_CASE ("onnx: stale trailing bytes are truncated")
{
    const Bytes model = make_sample_model ();
    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size () - 2u) };

    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size () - 2u),
                  onnx::LOADER_STRICT);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::TRUNCATED);
}

TEST_CASE ("onnx: malformed top-level tag")
{
    Bytes model;
    model.push_back (0xFFu);
    model.push_back (0xFFu);
    model.push_back (0xFFu);
    model.push_back (0xFFu);
    model.push_back (0xFFu);

    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };
    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()), onnx::LOADER_STRICT);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::OVERFLOW);
}

TEST_CASE ("onnx: tensor shape overflow is rejected")
{
    Bytes model = make_model (
        8, 13, "overflow",
        make_graph ({}, { make_tensor_raw (
                              static_cast<uint32_t> (onnx::DataType::INT8),
                              { 0x40000000ll, 0x40000000ll }, "huge", {}) },
                    {}, {}, "g"));

    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };
    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()),
                  onnx::LOADER_STRICT | onnx::LOADER_ALLOW_TRAILING_BYTES);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::OVERFLOW);
}

TEST_CASE ("onnx: negative dimensions are rejected")
{
    Bytes model = make_model (
        8, 13, "negative",
        make_graph ({}, { make_tensor_raw (
                              static_cast<uint32_t> (onnx::DataType::INT8),
                              { -1, 2 }, "bad", {}) },
                    {}, {}, "g"));

    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };
    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()),
                  onnx::LOADER_STRICT | onnx::LOADER_ALLOW_TRAILING_BYTES);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::BAD_SHAPE);
}

TEST_CASE ("onnx: a value shape overflow is rejected")
{
    const Bytes input = make_value_info (
        "x", static_cast<uint32_t> (onnx::DataType::FLOAT),
        { 0x40000000ll, 0x40000000ll });
    Bytes model = make_model (8, 13, "overflow",
                              make_graph ({}, {}, { input }, {}, "g"));

    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };
    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()),
                  onnx::LOADER_STRICT | onnx::LOADER_ALLOW_TRAILING_BYTES);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::OVERFLOW);
}

TEST_CASE ("onnx: raw payload size must match the shape")
{
    Bytes bad = make_tensor_raw (static_cast<uint32_t> (onnx::DataType::FLOAT),
                                 { 2 }, "short", { 1, 2, 3 });
    Bytes model
        = make_model (8, 13, "short",
                      make_graph ({}, { bad }, {}, {}, "g"));

    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };
    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()),
                  onnx::LOADER_STRICT | onnx::LOADER_ALLOW_TRAILING_BYTES);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::BAD_TENSOR);
}

TEST_CASE ("onnx: typed and string tensors are reported")
{
    const Bytes floats = make_tensor_float ({ 2 }, "f", { 1.0f, 2.0f });
    const Bytes strings
        = make_tensor_string ({ 2 }, "s", { "alpha", "beta" });
    const Bytes model
        = make_model (8, 13, "typed",
                      make_graph ({}, { floats, strings }, {}, {}, "g"));

    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };
    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()),
                  onnx::LOADER_STRICT | onnx::LOADER_ALLOW_TRAILING_BYTES);
    REQUIRE (!loader.run (capture.visitor ()));

    CHECK (capture.count (onnx::EventKind::TENSOR_DATA) == 3u);
    size_t string_events = 0;
    for (const Captured &event : capture._M_events)
        if (event._M_kind == onnx::EventKind::TENSOR_DATA
            && event._M_tensor.data_field ()
                   == onnx::TensorDataField::STRING_DATA)
            {
                ++string_events;
                CHECK (event._M_length == (string_events == 1u ? 5u : 4u));
            }
    CHECK (string_events == 2u);

    const Captured *const end
        = capture.first (onnx::EventKind::TENSOR_END);
    REQUIRE (end != nullptr);

    bool saw_string_end = false;
    for (const Captured &event : capture._M_events)
        if (event._M_kind == onnx::EventKind::TENSOR_END
            && event._M_tensor.data_field ()
                   == onnx::TensorDataField::STRING_DATA)
            {
                saw_string_end = true;
                CHECK ((event._M_tensor._M_flags
                        & onnx::TENSOR_FLAG_FRAGMENTED)
                       != 0u);
            }
    CHECK (saw_string_end);
}

TEST_CASE ("onnx: packed tensor dims are accepted")
{
    const Bytes raw{ 1, 2, 3, 4, 5, 6 };
    Bytes tensor;
    put_packed_varints (tensor, 1, { 2, 3 });
    put_uint (tensor, 2, static_cast<uint32_t> (onnx::DataType::INT8));
    put_string (tensor, 8, "p");
    put_bytes (tensor, 9, raw.data (), raw.size ());

    const Bytes model
        = make_model (8, 13, "packed", make_graph ({}, { tensor }, {}, {}, "g"));
    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };
    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()), onnx::LOADER_STRICT);
    REQUIRE (!loader.run (capture.visitor ()));
    CHECK (capture.count (onnx::EventKind::TENSOR_DIM) == 2u);
    const Captured *const end
        = capture.first (onnx::EventKind::TENSOR_END);
    REQUIRE (end != nullptr);
    CHECK (end->_M_tensor._M_elements == 6u);
    CHECK (end->_M_tensor._M_data_length == 6u);
}

TEST_CASE ("onnx: external data is opt-in and validated")
{
    const Bytes initializer = make_tensor_external (
        static_cast<uint32_t> (onnx::DataType::INT8), { 2, 2 }, "ext", 4u);
    const Bytes model
        = make_model (8, 13, "external",
                      make_graph ({}, { initializer }, {}, {}, "g"));

    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };

    SECTION ("without the flag")
    {
        onnx::Loader loader;
        Capture capture;
        loader.begin (onnx::memory_chunk, &source,
                      static_cast<uint32_t> (model.size ()),
                      onnx::LOADER_STRICT);
        const onnx::OnnxError error = loader.run (capture.visitor ());
        REQUIRE (error);
        CHECK (error.code () == onnx::OnnxCode::UNSUPPORTED);
    }

    SECTION ("with the flag")
    {
        onnx::Loader loader;
        Capture capture;
        loader.begin (onnx::memory_chunk, &source,
                      static_cast<uint32_t> (model.size ()),
                      onnx::LOADER_STRICT | onnx::LOADER_ALLOW_EXTERNAL);
        REQUIRE (!loader.run (capture.visitor ()));
        CHECK (capture.count (onnx::EventKind::EXTERNAL_DATA) == 3u);

        const Captured *const data
            = capture.first (onnx::EventKind::TENSOR_DATA);
        REQUIRE (data != nullptr);
        CHECK ((data->_M_tensor._M_flags & onnx::TENSOR_FLAG_EXTERNAL) != 0u);
        CHECK (data->_M_tensor.data_location () == 1u);
        CHECK (data->_M_tensor._M_data_length == 4u);
        CHECK (data->_M_length == 4u);
    }
}

TEST_CASE ("onnx: subgraphs are skipped or parsed by flag")
{
    const Bytes inner_node
        = make_node ("Relu", { "inner_in" }, { "inner_out" }, {}, "inner");
    const Bytes inner_graph
        = make_graph ({ inner_node }, {}, {}, {}, "sub");
    const Bytes outer_node
        = make_node ("If", { "x" }, { "y" },
                     { make_attribute_graph ("then_branch", inner_graph) },
                     "outer");
    const Bytes model
        = make_model (8, 13, "subgraph",
                      make_graph ({ outer_node }, {}, {}, {}, "g"));

    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };

    SECTION ("tolerant build skips the subgraph")
    {
        onnx::Loader loader;
        Capture capture;
        loader.begin (onnx::memory_chunk, &source,
                      static_cast<uint32_t> (model.size ()),
                      onnx::LOADER_ALLOW_TRAILING_BYTES);
        REQUIRE (!loader.run (capture.visitor ()));
        CHECK (capture.count (onnx::EventKind::NODE_BEGIN) == 1u);
        CHECK ((loader.model ()._M_flags
                & onnx::MODEL_FLAG_SUBGRAPH_SKIPPED)
               != 0u);
    }

    SECTION ("strict build rejects it")
    {
        onnx::Loader loader;
        Capture capture;
        loader.begin (onnx::memory_chunk, &source,
                      static_cast<uint32_t> (model.size ()),
                      onnx::LOADER_STRICT);
        const onnx::OnnxError error = loader.run (capture.visitor ());
        REQUIRE (error);
        CHECK (error.code () == onnx::OnnxCode::UNSUPPORTED);
    }

    SECTION ("allow flag parses the nested node")
    {
        onnx::Loader loader;
        Capture capture;
        loader.begin (onnx::memory_chunk, &source,
                      static_cast<uint32_t> (model.size ()),
                      onnx::LOADER_STRICT | onnx::LOADER_ALLOW_SUBGRAPHS);
        REQUIRE (!loader.run (capture.visitor ()));
        CHECK (capture.count (onnx::EventKind::NODE_BEGIN) == 2u);
        CHECK (capture.count (onnx::EventKind::GRAPH_BEGIN) == 2u);
    }
}

TEST_CASE ("onnx: a visitor can stop the walk")
{
    const Bytes model = make_sample_model ();
    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };

    onnx::Loader loader;
    Capture capture;
    capture._M_stop = onnx::EventKind::NODE_BEGIN;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()), onnx::LOADER_STRICT);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::STOPPED);
    CHECK (capture.count (onnx::EventKind::NODE_BEGIN) == 1u);
    CHECK (capture.count (onnx::EventKind::NODE_END) == 0u);
}

TEST_CASE ("onnx: opset newer than the loader is rejected")
{
    const Bytes model
        = make_model (8, 99, "future",
                      make_graph ({}, {}, {}, {}, "g"));
    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };

    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()), onnx::LOADER_STRICT);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::UNSUPPORTED);
    CHECK ((loader.model ()._M_flags & onnx::MODEL_FLAG_OPSET_TOO_NEW) != 0u);
}

TEST_CASE ("onnx: copy rejects out-of-range spans")
{
    const Bytes model = make_sample_model ();
    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };

    onnx::Loader loader;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()), onnx::LOADER_STRICT);

    uint8_t byte = 0;
    uint32_t copied = 0;
    onnx::OnnxError error
        = loader.copy (static_cast<uint32_t> (model.size ()), &byte, 1u, copied);
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::TRUNCATED);

    error = loader.copy (static_cast<uint32_t> (model.size ()) - 1u, &byte,
                         2u, copied);
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::LIMIT_EXCEEDED);
}

TEST_CASE ("onnx: sparse initializers are gated by flag")
{
    const Bytes values = make_tensor_raw (
        static_cast<uint32_t> (onnx::DataType::INT8), { 2 }, "v", { 7, 9 });
    Bytes indices;
    put_uint (indices, 1, 2u);
    put_uint (indices, 2, static_cast<uint32_t> (onnx::DataType::INT64));
    put_string (indices, 8, "i");
    put_packed_varints (indices, 7, { 0, 3 });

    Bytes sparse;
    put_message (sparse, 1, values);
    put_message (sparse, 2, indices);
    put_packed_varints (sparse, 3, { 4 });

    Bytes graph = make_graph ({}, {}, {}, {}, "g");
    put_message (graph, 15, sparse);
    const Bytes model = make_model (8, 13, "sparse", graph);

    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };

    SECTION ("tolerant build skips the sparse initializer")
    {
        onnx::Loader loader;
        Capture capture;
        loader.begin (onnx::memory_chunk, &source,
                      static_cast<uint32_t> (model.size ()));
        REQUIRE (!loader.run (capture.visitor ()));
        CHECK (capture.count (onnx::EventKind::TENSOR_BEGIN) == 0u);
        CHECK (loader.model ()._M_initializer_count == 0u);
    }

    SECTION ("strict build rejects it")
    {
        onnx::Loader loader;
        Capture capture;
        loader.begin (onnx::memory_chunk, &source,
                      static_cast<uint32_t> (model.size ()),
                      onnx::LOADER_STRICT);
        const onnx::OnnxError error = loader.run (capture.visitor ());
        REQUIRE (error);
        CHECK (error.code () == onnx::OnnxCode::UNSUPPORTED);
    }

    SECTION ("the allow flag parses values and indices")
    {
        onnx::Loader loader;
        Capture capture;
        loader.begin (onnx::memory_chunk, &source,
                      static_cast<uint32_t> (model.size ()),
                      onnx::LOADER_STRICT | onnx::LOADER_ALLOW_SPARSE);
        REQUIRE (!loader.run (capture.visitor ()));
        CHECK (capture.count (onnx::EventKind::TENSOR_BEGIN) == 2u);
        CHECK (loader.model ()._M_initializer_count == 1u);
    }
}

TEST_CASE ("onnx: a newer IR version is rejected")
{
    const Bytes graph = make_graph ({}, {}, {}, {}, "g");
    const Bytes model
        = make_model (onnx::_S_max_ir_version + 1u, 13, "ir", graph);

    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };
    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()), onnx::LOADER_STRICT);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::UNSUPPORTED);
    CHECK ((loader.model ()._M_flags & onnx::MODEL_FLAG_IR_TOO_NEW) != 0u);
}

TEST_CASE ("onnx: protobuf groups are rejected")
{
    Bytes model;
    put_tag (model, 1, 3u);
    put_varint (model, 1u);

    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };
    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()), onnx::LOADER_STRICT);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::UNSUPPORTED);
}

TEST_CASE ("onnx: a missing graph is rejected")
{
    Bytes imports;
    put_uint (imports, 2, 13u);
    Bytes model;
    put_uint (model, 1, 8u);
    put_message (model, 8, imports);

    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };
    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()), onnx::LOADER_STRICT);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::BAD_MODEL);
}

TEST_CASE ("onnx: external length must match the shape")
{
    const Bytes initializer = make_tensor_external (
        static_cast<uint32_t> (onnx::DataType::INT8), { 2, 2 }, "ext", 5u);
    const Bytes model
        = make_model (8, 13, "external",
                      make_graph ({}, { initializer }, {}, {}, "g"));

    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };
    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()),
                  onnx::LOADER_STRICT | onnx::LOADER_ALLOW_EXTERNAL);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::BAD_TENSOR);
}

TEST_CASE ("onnx: an external offset that overflows is rejected")
{
    Bytes tensor;
    put_uint (tensor, 1, 2u);
    put_uint (tensor, 1, 2u);
    put_uint (tensor, 2, static_cast<uint32_t> (onnx::DataType::INT8));
    put_string (tensor, 8, "ext");
    put_message (tensor, 13, make_external_entry ("location", "weights.bin"));
    put_message (tensor, 13,
                 make_external_entry ("offset", "30000000000000000000"));
    put_message (tensor, 13, make_external_entry ("length", "4"));
    put_uint (tensor, 14, 1u);

    const Bytes model
        = make_model (8, 13, "external",
                      make_graph ({}, { tensor }, {}, {}, "g"));
    onnx::MemorySource source{ model.data (),
                               static_cast<uint32_t> (model.size ()) };
    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::memory_chunk, &source,
                  static_cast<uint32_t> (model.size ()),
                  onnx::LOADER_STRICT | onnx::LOADER_ALLOW_EXTERNAL);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::OVERFLOW);
}

TEST_CASE ("onnx: a chunk list gap is truncation")
{
    const Bytes model = make_sample_model ();
    const size_t half = model.size () / 2u;
    const onnx::Chunk chunks[] = {
        onnx::Chunk{ model.data (), static_cast<uint32_t> (half) },
        onnx::Chunk{ nullptr, 0 },
    };
    onnx::ChunkList list{ chunks, 2u };

    onnx::Loader loader;
    Capture capture;
    loader.begin (onnx::chunk_list_chunk, &list,
                  static_cast<uint32_t> (model.size ()), onnx::LOADER_STRICT);
    const onnx::OnnxError error = loader.run (capture.visitor ());
    REQUIRE (error);
    CHECK (error.code () == onnx::OnnxCode::TRUNCATED);
}
