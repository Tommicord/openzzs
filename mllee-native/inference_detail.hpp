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

#include "mllee-native/dataset_types.hpp"
#include "mllee-native/model_config.hpp"
#include "mllee-native/q11_q15_q31.hpp"

#include <cstdint>

/**
 * @brief Shared implementation details for mllee-native inference and training.
 *
 * Contains the single canonical `flatten_sample` implementation used by both
 * the inference engine and the training pipeline, eliminating the previous
 * duplication.
 * @{
 */
namespace zzs::mllee_native::detail {

using zzs::mllee_native::dataset::TrainingSample;
using zzs::mllee_native::fx::Q15;

/** @brief Q15 code step: 32768 / 127 ≈ 258. */
inline constexpr int32_t kCodeStep = 258;

/** @brief Convert Q15 raw to int8 code. */
[[nodiscard]] constexpr int8_t
q15_raw_to_code (const int16_t q15_raw) noexcept {
    const int32_t code = static_cast<int32_t> (q15_raw) / kCodeStep;
    return static_cast<int8_t> (code > 127 ? 127 : (code < -127 ? -127 : code));
}

/** @brief Flatten `seq_len` windows x `feature_count` features into int8 codes.
 *
 * This is the single canonical implementation shared by inference and the
 * training pipeline.  Uses the same Q15->int8 code conversion as the training
 * pipeline so the inference input matches the training input format exactly.
 */
inline void
flatten_sample (const TrainingSample &sample, int8_t *const flat) noexcept {
    uint32_t out = 0u;
    for (uint32_t w = 0; w < zzs::mllee_native::dataset::kSeqLen; ++w) {
        for (uint32_t f = 0; f < zzs::mllee_native::dataset::kFeatureCount; ++f) {
            flat[out++] = q15_raw_to_code (sample._M_windows[w]._M_features[f]);
        }
    }
}

/** @} */
} // namespace zzs::mllee_native::detail
