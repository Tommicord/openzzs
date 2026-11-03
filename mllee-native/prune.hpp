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

#include <cmath>
#include <cstdint>
#include <cstdlib>

/**
 * @brief Heap-free magnitude pruning utilities for int8 weight arrays.
 *
 * All functions are `noexcept`, allocate no heap memory, use no exceptions and
 * no RTTI.  State is caller-owned; there are no statics or globals.
 *
 *   prune_magnitude   - element-wise magnitude pruning
 *   prune_rows        - structural row pruning for [out_dim x in_dim] matrices
 *   prune_sparsity    - fraction of zero weights
 *
 * @{
 */
namespace zzs::mllee_native::prune {

/** @brief Maximum output dimension supported by `prune_rows`.
 *
 * The row selector uses a fixed-size stack buffer, so `out_dim` must not exceed
 * this limit.  Exceeding it is treated as "nothing pruned" to stay heap-free.
 */
constexpr uint32_t kMaxOutDim = 4096;

/** @brief Element-wise magnitude pruning.
 *
 * Zeroes the `target = round(count * sparsity_ratio)` weights with the smallest
 * magnitude `|w|`.  `sparsity_ratio` is clamped to [0, 1] (NaN -> 0).
 *
 * @returns The number of weights zeroed.
 * @param mask_out  If non-null, writes `1` for kept weights and `0` for pruned
 *                  weights, one byte per weight.
 */
[[nodiscard]] inline uint32_t
prune_magnitude (int8_t *weights, uint32_t count, double sparsity_ratio, uint8_t *mask_out = nullptr) noexcept {
    if (count == 0u) {
        return 0u;
    }

    double r = sparsity_ratio;
    if (!(r >= 0.0)) { /* covers NaN */
        r = 0.0;
    }
    if (r > 1.0) {
        r = 1.0;
    }

    const uint32_t target = static_cast<uint32_t> (std::round (static_cast<double> (count) * r));
    const uint32_t target_pruned = target > count ? count : target;

    if (target_pruned == 0u) {
        if (mask_out != nullptr) {
            for (uint32_t i = 0u; i < count; ++i) {
                mask_out[i] = 1u;
            }
        }
        return 0u;
    }

    if (target_pruned == count) {
        for (uint32_t i = 0u; i < count; ++i) {
            weights[i] = 0;
        }
        if (mask_out != nullptr) {
            for (uint32_t i = 0u; i < count; ++i) {
                mask_out[i] = 0u;
            }
        }
        return count;
    }

    /* Histogram of magnitudes.  int8_t magnitude ranges 0..128 (|-128| = 128). */
    uint32_t hist[129] = {};
    for (uint32_t i = 0u; i < count; ++i) {
        const int v = static_cast<int> (weights[i]);
        uint32_t m = static_cast<uint32_t> (v < 0 ? -v : v);
        if (m > 128u) {
            m = 128u;
        }
        ++hist[m];
    }

    /* Find the cutoff magnitude: smallest m such that cum(m) >= target_pruned. */
    uint32_t cum = 0u;
    uint32_t cutoff = 0u;
    for (uint32_t m = 0u; m <= 128u; ++m) {
        if (cum + hist[m] >= target_pruned) {
            cutoff = m;
            break;
        }
        cum += hist[m];
    }

    /* cum == count of weights with |w| < cutoff. */
    uint32_t need_at_cutoff = target_pruned - cum;
    uint32_t pruned = 0u;

    for (uint32_t i = 0u; i < count; ++i) {
        const int v = static_cast<int> (weights[i]);
        uint32_t m = static_cast<uint32_t> (v < 0 ? -v : v);
        if (m > 128u) {
            m = 128u;
        }

        const bool do_prune = (m < cutoff) || (m == cutoff && need_at_cutoff > 0u);

        if (do_prune) {
            weights[i] = 0;
            if (need_at_cutoff > 0u) {
                --need_at_cutoff;
            }
            ++pruned;
            if (mask_out != nullptr) {
                mask_out[i] = 0u;
            }
        } else if (mask_out != nullptr) {
            mask_out[i] = 1u;
        }
    }

    return pruned;
}

/** @brief Structural row pruning for a row-major [out_dim x in_dim] matrix.
 *
 * Computes the L1 norm (sum of `|w|`) of each row, then zeroes the
 * `target = round(out_dim * sparsity_ratio)` rows with the smallest norm.
 * Entire pruned rows are zeroed.
 *
 * @returns The number of rows zeroed.
 * @param row_mask_out  If non-null, writes `1` for kept rows and `0` for pruned
 *                      rows, one byte per row.
 */
[[nodiscard]] inline uint32_t
prune_rows (int8_t *weights, uint32_t out_dim, uint32_t in_dim, double sparsity_ratio,
            uint8_t *row_mask_out = nullptr) noexcept {
    if (out_dim == 0u || in_dim == 0u) {
        return 0u;
    }
    if (out_dim > kMaxOutDim) {
        return 0u;
    }

    double r = sparsity_ratio;
    if (!(r >= 0.0)) {
        r = 0.0;
    }
    if (r > 1.0) {
        r = 1.0;
    }

    const uint32_t target = static_cast<uint32_t> (std::round (static_cast<double> (out_dim) * r));
    const uint32_t target_rows = target > out_dim ? out_dim : target;

    if (target_rows == 0u) {
        if (row_mask_out != nullptr) {
            for (uint32_t i = 0u; i < out_dim; ++i) {
                row_mask_out[i] = 1u;
            }
        }
        return 0u;
    }

    if (target_rows == out_dim) {
        for (uint32_t i = 0u; i < out_dim; ++i) {
            int8_t *const row = weights + i * in_dim;
            for (uint32_t j = 0u; j < in_dim; ++j) {
                row[j] = 0;
            }
        }
        if (row_mask_out != nullptr) {
            for (uint32_t i = 0u; i < out_dim; ++i) {
                row_mask_out[i] = 0u;
            }
        }
        return out_dim;
    }

    /* L1 norm of each row. */
    int32_t norm[kMaxOutDim];
    for (uint32_t i = 0u; i < out_dim; ++i) {
        const int8_t *const row = weights + i * in_dim;
        int32_t sum = 0;
        for (uint32_t j = 0u; j < in_dim; ++j) {
            const int v = static_cast<int> (row[j]);
            sum += v < 0 ? -v : v;
        }
        norm[i] = sum;
    }

    uint8_t pruned_flag[kMaxOutDim] = {};
    uint32_t rows_pruned = 0u;

    for (uint32_t t = 0u; t < target_rows; ++t) {
        uint32_t best_idx = 0u;
        int32_t best_norm = 0;
        bool found = false;
        for (uint32_t i = 0u; i < out_dim; ++i) {
            if (pruned_flag[i] != 0u) {
                continue;
            }
            if (!found || norm[i] < best_norm) {
                best_idx = i;
                best_norm = norm[i];
                found = true;
            }
        }
        if (!found) {
            break;
        }
        pruned_flag[best_idx] = 1u;
        int8_t *const row = weights + best_idx * in_dim;
        for (uint32_t j = 0u; j < in_dim; ++j) {
            row[j] = 0;
        }
        ++rows_pruned;
    }

    if (row_mask_out != nullptr) {
        for (uint32_t i = 0u; i < out_dim; ++i) {
            row_mask_out[i] = pruned_flag[i] == 0u ? 1u : 0u;
        }
    }

    return rows_pruned;
}

/** @brief Fraction of zero weights in `weights`.
 *
 * @returns `zero_count / count`, or `0.0` when `count == 0`.
 */
[[nodiscard]] inline double
prune_sparsity (const int8_t *weights, uint32_t count) noexcept {
    if (count == 0u || weights == nullptr) {
        return 0.0;
    }
    uint32_t zeros = 0u;
    for (uint32_t i = 0u; i < count; ++i) {
        if (weights[i] == 0) {
            ++zeros;
        }
    }
    return static_cast<double> (zeros) / static_cast<double> (count);
}

/** @} */
} // namespace zzs::mllee_native::prune
