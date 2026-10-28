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

/*
 * Streaming ONNX loader.
 *
 *   zzs::mllee::onnx::Loader loader;          // 128 bytes of resident state
 *   loader.begin (fetch, context, size);
 *   loader.run (visitor);
 *
 * The caller supplies the bytes through a chunk callback and owns every buffer;
 * the loader keeps only the model, node and tensor headers while the protobuf
 * stream passes through.  Weights and strings are reported as absolute spans
 * and fetched later with Loader::copy() or Loader::copy_string().
 *
 * Tiers:
 *   onnx_config.hpp    budgets, flags, version ceilings
 *   onnx_status.hpp    OnnxCode/OnnxError
 *   onnx_types.hpp     ONNX enums and dtype tables
 *   onnx_hash.hpp      FNV-1a and djb2
 *   onnx_asm.hpp       hand-written primitive declarations/fallbacks
 *   onnx_chunk.hpp     Chunk, ChunkList, memory source
 *   onnx_stream.hpp    absolute cursor over caller-owned chunks
 *   onnx_bounded.hpp   checked length/shape arithmetic
 *   onnx_varint.hpp    protobuf varint decoding
 *   onnx_wire.hpp      tags, lengths, fixed fields, generic skip
 *   onnx_scope.hpp     length-delimited message bounds
 *   onnx_string.hpp    hashed strings and decimal offsets
 *   onnx_model_info.hpp / onnx_node_info.hpp / onnx_tensor_info.hpp
 *   onnx_event.hpp     event stream and visitor
 *   onnx_reader.hpp    Context shared by the structural parsers
 *   onnx_ops.hpp       operator/attribute name registries
 *   onnx_validate.hpp  schema cross-checks for tensor payloads
 *   onnx_external.hpp  external-data entries
 *   onnx_shape.hpp     TensorShapeProto/TypeProto
 *   onnx_value.hpp     ValueInfoProto
 *   onnx_tensor.hpp    TensorProto and SparseTensorProto
 *   onnx_attribute.hpp AttributeProto
 *   onnx_node.hpp      NodeProto
 *   onnx_graph.hpp     GraphProto
 *   onnx_model.hpp     ModelProto and opset imports
 *   onnx_loader.hpp    the 128-byte Loader facade
 *   onnx_trace.hpp     optional logging visitor built on std/log.hpp; it is a
 *                      separate include because bare-metal targets may not have
 *                      the lock-free atomics that std/log.hpp requires
 */

#include "mllee/status.hpp"
#include "onnx_asm.hpp"
#include "onnx_attribute.hpp"
#include "onnx_bounded.hpp"
#include "onnx_chunk.hpp"
#include "onnx_config.hpp"
#include "onnx_event.hpp"
#include "onnx_external.hpp"
#include "onnx_graph.hpp"
#include "onnx_hash.hpp"
#include "onnx_loader.hpp"
#include "onnx_model.hpp"
#include "onnx_model_info.hpp"
#include "onnx_node.hpp"
#include "onnx_node_info.hpp"
#include "onnx_ops.hpp"
#include "onnx_reader.hpp"
#include "onnx_scope.hpp"
#include "onnx_shape.hpp"
#include "onnx_status.hpp"
#include "onnx_stream.hpp"
#include "onnx_string.hpp"
#include "onnx_tensor.hpp"
#include "onnx_tensor_info.hpp"
#include "onnx_types.hpp"
#include "onnx_validate.hpp"
#include "onnx_value.hpp"
#include "onnx_varint.hpp"
#include "onnx_wire.hpp"
