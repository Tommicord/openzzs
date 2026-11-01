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

#include "mllee-native/dataset_gen.hpp"

#include <cstdint>

/**
 * @brief Compile-time instantiated power outage dataset stored in .rodata.
 *
 * Generated with a fixed 16-bit seed so the dataset is deterministic and
 * identical across all builds/targets.  The consteval evaluation happens at
 * compile time; the resulting array lives in .rodata and consumes no RAM.
 * @{
 */
namespace zzs::mllee_native {

/** @brief Default seed for the dataset generator. */
inline constexpr uint16_t kDefaultSeed = 0xC0DEu;

/** @brief Default class distribution (per-mille): 60% normal, 25% outage, 15% transition. */
inline constexpr zzs::mllee_native::dataset::GenerationParams kDefaultParams{
    kDefaultSeed, 64u, /* num_samples */
    600u,              /* normal ratio */
    250u,              /* outage ratio */
    150u,              /* transition ratio */
};

/** @brief The compile-time generated dataset (stored in .rodata). */
inline constexpr auto g_dataset = zzs::mllee_native::dataset_gen::generate_dataset (kDefaultParams);

/** @brief Number of samples in the compile-time dataset. */
[[nodiscard]] constexpr uint32_t
dataset_count () noexcept {
    return g_dataset._M_count;
}

/** @brief Read-only access to sample i. */
[[nodiscard]] constexpr const zzs::mllee_native::dataset::TrainingSample &
dataset_sample (uint32_t i) noexcept {
    return g_dataset._M_samples[i];
}

/** @} */
} // namespace zzs::mllee_native
