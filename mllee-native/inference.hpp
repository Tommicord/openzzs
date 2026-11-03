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

#include "mllee-native/dataset.hpp"
#include "mllee-native/inference_detail.hpp"
#include "mllee-native/model.hpp"
#include "mllee/core.hpp"
#include "mllee/core_status.hpp"
#include "std/string_view.hpp"

#include <cstdint>
#include <cstdio>

/**
 * @brief Inference path for power outage detection.
 *
 * Runs the native `"MLEE"` model blob through `zzs::mllee::Core::run()`.
 * Flattens 5 windows x 38 features = 190 int8 input codes, feeds them to the
 * model, and reads 3 softmax output codes interpreted as probabilities:
 *
 *   output[0] -> P_normal
 *   output[1] -> P_outage
 *   output[2] -> P_transition
 *
 * All arithmetic is fixed-point (int8 codes).  No heap, no float, no RTTI,
 * no exceptions.  Caller-owned buffers only.
 * @{
 */
namespace zzs::mllee_native::inference {

using zzs::base::SimpleStringView;
using zzs::mllee::Core;
using zzs::mllee::Status;
using zzs::mllee::status_ok;
using zzs::mllee_native::dataset::TrainingSample;
using zzs::mllee_native::export_::validate_model;
using zzs::mllee_native::export_::conv_attn::kConvAttnBlobSize;
using zzs::mllee_native::export_::multitask::kAttnOutput;
using zzs::mllee_native::export_::multitask::kConvOutput;
using zzs::mllee_native::export_::multitask::kHead1Out;
using zzs::mllee_native::export_::multitask::kHead2Out;
using zzs::mllee_native::export_::multitask::kHead3Out;
using zzs::mllee_native::export_::multitask::kMultiTaskBlobSize;
using zzs::mllee_native::export_::multitask::kPooled;
using zzs::mllee_native::export_::multitask::kQKVOutput;
using zzs::mllee_native::model::find_outpred;
using zzs::mllee_native::model::g_model_vtable;
using zzs::mllee_native::model::init_model_vtable;
using zzs::mllee_native::model::kBlobSize;
using zzs::mllee_native::model::kInputDim;
using zzs::mllee_native::model::kOutputDim;
using zzs::mllee_native::model::ModelVTable;
using zzs::mllee_native::model::VTableEntry;

/** @brief Flatten `seq_len` windows x `feature_count` features into int8 codes.
 *
 * Declared before InferenceEngine so the template method can find it.
 */
inline void
flatten_sample (const TrainingSample &sample, int8_t *const flat) noexcept {
    zzs::mllee_native::detail::flatten_sample (sample, flat);
}

/**
 * @brief OOP inference engine: owns input/output code buffers and runs the
 *        model blob through `zzs::mllee::Core::run()`.
 *
 * All state is inside this object; no globals.  Instantiate with the desired
 * `Config` (defaults to `DefaultConfig`).
 */
class InferenceEngine {
  public:
    InferenceEngine () noexcept = default;
    InferenceEngine (const InferenceEngine &) = delete;
    InferenceEngine &
    operator= (const InferenceEngine &) = delete;

    /** @brief Run inference on one sample.
     *
     * Returns the predicted class index (0=normal, 1=outage, 2=transition) and
     * writes the output_dim softmax output codes into `output`.
     *
     * @returns Status::OK on success, BAD_LAYOUT if the model blob is invalid,
     *          or another mllee Status error.
     */
    [[nodiscard]] Status
    run (const TrainingSample &sample, int8_t *const output, uint32_t *const predicted_class) noexcept {
        if (output == nullptr || predicted_class == nullptr)
            return Status::BAD_LAYOUT;
        /* Ensure the OUTPRED model blob is serialized and registered in the VTable. */
        if (const Status st = init_model_vtable (); !status_ok (st))
            return st;
        const uint32_t idx = find_outpred ();
        if (idx == ModelVTable<>::k_npos)
            return Status::OUT_OF_RANGE;
        const VTableEntry &entry = g_model_vtable[idx];
        if (const Status st = validate_model (entry._M_blob, entry._M_blob_size); !status_ok (st))
            return st;
        flatten_sample (sample, this->_M_input);
        Core core;
        const auto result = core.run (entry._M_blob, entry._M_blob_size, this->_M_input, kInputDim);
        if (!result)
            return result.error ().kind ();
        for (uint32_t i = 0; i < kOutputDim; ++i)
            output[i] = result.value ()._M_data[i];
        int32_t best_code = output[0];
        uint32_t best_idx = 0u;
        for (uint32_t c = 1u; c < kOutputDim; ++c) {
            if (output[c] > best_code) {
                best_code = output[c];
                best_idx = c;
            }
        }
        *predicted_class = best_idx;
        return Status::OK;
    }

  private:
    alignas (4) int8_t _M_input[kInputDim] = {};
    alignas (4) int8_t _M_output[kOutputDim] = {};
};

/** @brief Run inference on one sample (free-function wrapper).
 *
 * Returns the predicted class index and writes the softmax output codes into
 * `output`.  Uses a stack-local `InferenceEngine`; no globals.
 *
 * @returns Status::OK on success, BAD_LAYOUT if the model blob is invalid.
 */
[[nodiscard]] inline Status
run_inference (const TrainingSample &sample, int8_t *const output, uint32_t *const predicted_class) noexcept {
    InferenceEngine engine;
    return engine.run (sample, output, predicted_class);
}

/** @brief Run Tier 2 CONV1D + Attention inference on one sample.
 *
 * Runs the supplied conv-attn model blob through `zzs::mllee::Core::run()`.
 * `model_blob` must be `kConvAttnBlobSize` bytes and have been serialized by
 * `serialize_conv_attn_model()` / `Model::build_conv_attn_blob()`.
 *
 * @returns Status::OK on success, BAD_LAYOUT if the model blob is invalid,
 *          or another mllee Status error.
 */
[[nodiscard]] inline Status
run_conv_attn_inference (const TrainingSample &sample, const uint8_t *model_blob, uint32_t model_blob_size,
                         int8_t *const output, uint32_t *const predicted_class) noexcept {
    if (output == nullptr || predicted_class == nullptr || model_blob == nullptr)
        return Status::BAD_LAYOUT;
    if (const Status st = validate_model (model_blob, model_blob_size); !status_ok (st))
        return st;
    alignas (4) int8_t input[kInputDim] = {};
    flatten_sample (sample, input);
    Core core;
    const auto result = core.run (model_blob, model_blob_size, input, kInputDim);
    if (!result)
        return result.error ().kind ();
    for (uint32_t i = 0; i < kOutputDim; ++i)
        output[i] = result.value ()._M_data[i];
    int32_t best_code = output[0];
    uint32_t best_idx = 0u;
    for (uint32_t c = 1u; c < kOutputDim; ++c) {
        if (output[c] > best_code) {
            best_code = output[c];
            best_idx = c;
        }
    }
    *predicted_class = best_idx;
    return Status::OK;
}

/** @brief Run Tier 3 multi-task inference on one sample.
 *
 * Runs the supplied multi-task model blob through `zzs::mllee::Core::run()`.
 * `model_blob` must be `kMultiTaskBlobSize` bytes and have been serialized by
 * `serialize_multitask_model()` / `Model::build_multitask_blob()`.
 *
 * The shared encoder produces a pooled 8-code vector that feeds three heads:
 *   head 1 (arena[208..211]): 3-class classification (softmax)
 *   head 2 (arena[211..212]): confidence regression (1 code)
 *   head 3 (arena[212..213]): time-to-outage regression (1 code)
 *
 * `predicted_class` receives the argmax of head 1 (0=normal, 1=outage,
 * 2=transition). `confidence_code` and `time_code` receive the raw int8
 * output codes from head 2 and head 3 respectively.
 *
 * @returns Status::OK on success, BAD_LAYOUT if the model blob is invalid,
 *          or another mllee Status error.
 */
[[nodiscard]] inline Status
run_multitask_inference (const TrainingSample &sample, const uint8_t *model_blob, uint32_t model_blob_size,
                         uint32_t *const predicted_class, int8_t *const confidence_code,
                         int8_t *const time_code) noexcept {
    if (predicted_class == nullptr || confidence_code == nullptr || time_code == nullptr || model_blob == nullptr)
        return Status::BAD_LAYOUT;
    if (const Status st = validate_model (model_blob, model_blob_size); !status_ok (st))
        return st;
    alignas (4) int8_t input[kInputDim] = {};
    flatten_sample (sample, input);
    Core core;
    const auto result = core.run (model_blob, model_blob_size, input, kInputDim);
    if (!result)
        return result.error ().kind ();

    const int8_t *const arena = core.arena ();
    /* Head 1 classification output: arena[208..211] (3 codes, softmax). */
    const uint32_t head1_off = kConvOutput + kQKVOutput + kAttnOutput + kPooled; /* 208 */
    int32_t best_code = arena[head1_off];
    uint32_t best_idx = 0u;
    for (uint32_t c = 1u; c < kHead1Out; ++c) {
        if (arena[head1_off + c] > best_code) {
            best_code = arena[head1_off + c];
            best_idx = c;
        }
    }
    *predicted_class = best_idx;

    /* Head 2 confidence output: arena[211..212] (1 code). */
    const uint32_t head2_off = head1_off + kHead1Out; /* 211 */
    *confidence_code = arena[head2_off];

    /* Head 3 time-to-outage output: arena[212..213] (1 code). */
    const uint32_t head3_off = head2_off + kHead2Out; /* 212 */
    *time_code = arena[head3_off];

    return Status::OK;
}

/** @brief Convert an output code to a probability percentage (0-100).
 *
 * Codes are in [-127, 127]; map to [0, 100] linearly.
 */
[[nodiscard]] inline uint32_t
code_to_probability (const int8_t code) noexcept {
    const int32_t pct = (static_cast<int32_t> (code) + 127) * 100 + 127;
    const int32_t v = pct / 254;
    return static_cast<uint32_t> (v > 100 ? 100 : (v < 0 ? 0 : v));
}

/** @brief Class name for a class index. */
[[nodiscard]] inline SimpleStringView
class_name (const uint32_t class_index) noexcept {
    switch (class_index) {
    case 0:
        return "normal";
    case 1:
        return "outage";
    case 2:
        return "transition";
    default:
        return "unknown";
    }
}

/** @} */
} // namespace zzs::mllee_native::inference
