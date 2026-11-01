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
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHER WARRANTIES,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
 * USE OR OTHER DEALINGS IN THE SOFTWARE. */

#pragma once

#include "mllee-native/lcg16.hpp"
#include "mllee-native/model_config.hpp"
#include "mllee-native/q11_q15_q31.hpp"

#include <cstdint>

/**
 * @brief Dataset types and generation parameters for power outage detection.
 *
 * Each training sample is a sequence of 5 windows (seq_len=5), each window
 * containing 38 Q15 features matching the preprocessing bank output layout.
 * All data is generated at compile time (consteval) and stored in .rodata.
 * @{
 */
namespace zzs::mllee_native::dataset {

/** @brief Number of overlapping windows per sequence (matches kernel config). */
inline constexpr uint32_t kSeqLen = 5u;

/** @brief Number of features per window (matches kernel config). */
inline constexpr uint32_t kFeatureCount = 38u;

/** @brief Number of classes: normal(0), outage(1), transition(2). */
inline constexpr uint32_t kClassCount = 3u;

static_assert (kSeqLen == zzs::mllee_native::DefaultConfig::seq_len, "kSeqLen must match DefaultConfig::seq_len");
static_assert (kFeatureCount == zzs::mllee_native::DefaultConfig::feature_count,
               "kFeatureCount must match DefaultConfig::feature_count");
static_assert (kClassCount == zzs::mllee_native::DefaultConfig::output_dim,
               "kClassCount must match DefaultConfig::output_dim");

/** @brief Label encoding. */
enum class Label : uint8_t {
    Normal = 0u,
    Outage = 1u,
    Transition = 2u,
};

/** @brief One feature window: 38 Q15 values. */
struct FeatureWindow {
    int16_t _M_features[kFeatureCount];
};

/** @brief One training sample: 5 windows + label. */
struct TrainingSample {
    FeatureWindow _M_windows[kSeqLen];
    uint8_t _M_label;
};

/** @brief Dataset generation parameters. */
struct GenerationParams {
    uint16_t _M_seed;
    uint32_t _M_num_samples;
    uint32_t _M_normal_ratio;     /* per-mille (0-1000) */
    uint32_t _M_outage_ratio;     /* per-mille (0-1000) */
    uint32_t _M_transition_ratio; /* per-mille (0-1000) */
};

/** @brief Full compile-time dataset (stored in .rodata). */
template <uint32_t N> struct Dataset {
    TrainingSample _M_samples[N];
    uint32_t _M_count;
};

/** @} */
} // namespace zzs::mllee_native::dataset
