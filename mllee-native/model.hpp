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
#include "mllee-native/model_config.hpp"
#include "mllee-native/model_export.hpp"
#include "mllee-native/q11_q15_q31.hpp"

#include <cstdint>

/**
 * @brief Compile-time model blob for power outage detection.
 *
 * Weights are generated at compile time from the training pipeline's trained
 * parameters (or fall back to deterministic initialization if not yet trained).
 * The blob lives in .rodata and is consumed by `zzs::mllee::Core::run()`.
 *
 * Model: LINEAR(190->16, act=NONE) + ACTIVATION(RELU) + LINEAR(16->3, act=NONE) + SOFTMAX(3->3)
 * Output: [P_normal, P_outage, P_transition] as int8 codes [-127, 127]
 * @{
 */
namespace zzs::mllee_native::model {

using zzs::mllee::Status;
using zzs::mllee::status_ok;
using zzs::mllee_native::ConvAttnDefaultConfig;
using zzs::mllee_native::DefaultConfig;
using zzs::mllee_native::ModelConfig;
using zzs::mllee_native::export_::kBlobSize;
using zzs::mllee_native::export_::kHiddenDim;
using zzs::mllee_native::export_::kHiddenInput;
using zzs::mllee_native::export_::kInputDim;
using zzs::mllee_native::export_::kOutputDim;
using zzs::mllee_native::export_::serialize_conv_attn_model;
using zzs::mllee_native::export_::serialize_model;
using zzs::mllee_native::export_::serialize_multitask_model;
using zzs::mllee_native::export_::validate_model;
using zzs::mllee_native::export_::conv_attn::kConvAttnBlobSize;
using zzs::mllee_native::export_::multitask::kMultiTaskBlobSize;

/** @brief Requantization multiplier (Q15, 32768 = 1.0 identity). */
inline constexpr int32_t kRequantMult = 32768;

/**
 * @brief OOP model container: weights, bias, and trained state, all sized by
 *        Config.  This is the preferred way to hold model state; the global
 *        aliases below are kept for backward compatibility with existing tests.
 */
template <typename Config = DefaultConfig> class Model {
  public:
    Model () noexcept = default;
    Model (const Model &) = delete;
    Model &
    operator= (const Model &) = delete;

    /** @brief Build the complete model blob into `dst`. */
    [[nodiscard]] Status
    build_blob (uint8_t *dst) const noexcept {
        const int8_t *hidden_weights = _M_trained ? _M_hidden_weights : nullptr;
        const int8_t *hidden_bias = _M_trained ? _M_hidden_bias : nullptr;
        const int8_t *output_weights = _M_trained ? _M_weights : nullptr;
        const int8_t *output_bias = _M_trained ? _M_bias : nullptr;

        /* Deterministic fallback weights (same as training init). */
        alignas (4) int8_t init_hidden_weights[Config::hidden_input * Config::hidden_dim];
        alignas (4) int8_t init_hidden_bias[Config::hidden_dim];
        alignas (4) int8_t init_output_weights[Config::hidden_dim * Config::output_dim];
        alignas (4) int8_t init_output_bias[Config::output_dim];

        if (!_M_trained) {
            for (uint32_t i = 0; i < Config::hidden_input * Config::hidden_dim; ++i)
                init_hidden_weights[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
            for (uint32_t i = 0; i < Config::hidden_dim; ++i)
                init_hidden_bias[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);
            for (uint32_t i = 0; i < Config::hidden_dim * Config::output_dim; ++i)
                init_output_weights[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
            for (uint32_t i = 0; i < Config::output_dim; ++i)
                init_output_bias[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);
            hidden_weights = init_hidden_weights;
            hidden_bias = init_hidden_bias;
            output_weights = init_output_weights;
            output_bias = init_output_bias;
        }

        return serialize_model (dst, hidden_weights, hidden_bias, output_weights, output_bias, kRequantMult);
    }

    /** @brief Serialize the current weights/bias into `dst`. */
    [[nodiscard]] Status
    serialize (uint8_t *dst) const noexcept {
        return serialize_model (dst, _M_hidden_weights, _M_hidden_bias, _M_weights, _M_bias, kRequantMult);
    }

    /** @brief Build the Tier 2 CONV1D + Attention model blob into `dst`.
     *
     * Uses trained weights if `_M_trained` is true, otherwise deterministic
     * initialization.  The blob is `kConvAttnBlobSize` bytes.
     */
    [[nodiscard]] Status
    build_conv_attn_blob (uint8_t *dst) const noexcept {
        const int8_t *conv_weights = _M_trained ? _M_conv_weights : nullptr;
        const int8_t *conv_bias = _M_trained ? _M_conv_bias : nullptr;
        const int8_t *qkv_weights = _M_trained ? _M_qkv_weights : nullptr;
        const int8_t *qkv_bias = _M_trained ? _M_qkv_bias : nullptr;
        const int8_t *out_weights = _M_trained ? _M_out_weights : nullptr;
        const int8_t *out_bias = _M_trained ? _M_out_bias : nullptr;

        /* Deterministic fallback weights (same as training init). */
        alignas (4) int8_t init_conv_weights[ConvAttnDefaultConfig::conv_channels * ConvAttnDefaultConfig::feature_count
                                             * ConvAttnDefaultConfig::conv_kernel];
        alignas (4) int8_t init_conv_bias[ConvAttnDefaultConfig::conv_channels];
        alignas (4) int8_t init_qkv_weights[ConvAttnDefaultConfig::conv_output * ConvAttnDefaultConfig::qkv_output];
        alignas (4) int8_t init_qkv_bias[ConvAttnDefaultConfig::qkv_output];
        alignas (4) int8_t init_out_weights[ConvAttnDefaultConfig::pooled * ConvAttnDefaultConfig::output_dim];
        alignas (4) int8_t init_out_bias[ConvAttnDefaultConfig::output_dim];

        if (!_M_trained) {
            for (uint32_t i = 0; i < ConvAttnDefaultConfig::conv_channels * ConvAttnDefaultConfig::feature_count
                                         * ConvAttnDefaultConfig::conv_kernel;
                 ++i)
                init_conv_weights[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
            for (uint32_t i = 0; i < ConvAttnDefaultConfig::conv_channels; ++i)
                init_conv_bias[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);
            for (uint32_t i = 0; i < ConvAttnDefaultConfig::conv_output * ConvAttnDefaultConfig::qkv_output; ++i)
                init_qkv_weights[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
            for (uint32_t i = 0; i < ConvAttnDefaultConfig::qkv_output; ++i)
                init_qkv_bias[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);
            for (uint32_t i = 0; i < ConvAttnDefaultConfig::pooled * ConvAttnDefaultConfig::output_dim; ++i)
                init_out_weights[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
            for (uint32_t i = 0; i < ConvAttnDefaultConfig::output_dim; ++i)
                init_out_bias[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);
            conv_weights = init_conv_weights;
            conv_bias = init_conv_bias;
            qkv_weights = init_qkv_weights;
            qkv_bias = init_qkv_bias;
            out_weights = init_out_weights;
            out_bias = init_out_bias;
        }

        return serialize_conv_attn_model (dst, conv_weights, conv_bias, qkv_weights, qkv_bias, out_weights, out_bias,
                                          kRequantMult);
    }

    /** @brief Build the Tier 3 multi-task model blob into `dst`.
     *
     * Uses trained weights if `_M_trained` is true, otherwise deterministic
     * initialization.  The blob is `kMultiTaskBlobSize` bytes.
     */
    [[nodiscard]] Status
    build_multitask_blob (uint8_t *dst) const noexcept {
        const int8_t *conv_weights = _M_trained ? _M_conv_weights : nullptr;
        const int8_t *conv_bias = _M_trained ? _M_conv_bias : nullptr;
        const int8_t *qkv_weights = _M_trained ? _M_qkv_weights : nullptr;
        const int8_t *qkv_bias = _M_trained ? _M_qkv_bias : nullptr;
        const int8_t *head1_weights = _M_trained ? _M_head1_weights : nullptr;
        const int8_t *head1_bias = _M_trained ? _M_head1_bias : nullptr;
        const int8_t *head2_weights = _M_trained ? _M_head2_weights : nullptr;
        const int8_t *head2_bias = _M_trained ? _M_head2_bias : nullptr;
        const int8_t *head3_weights = _M_trained ? _M_head3_weights : nullptr;
        const int8_t *head3_bias = _M_trained ? _M_head3_bias : nullptr;

        /* Deterministic fallback weights (same as training init). */
        alignas (4) int8_t init_conv_weights[ConvAttnDefaultConfig::conv_channels * ConvAttnDefaultConfig::feature_count
                                             * ConvAttnDefaultConfig::conv_kernel];
        alignas (4) int8_t init_conv_bias[ConvAttnDefaultConfig::conv_channels];
        alignas (4) int8_t init_qkv_weights[ConvAttnDefaultConfig::conv_output * ConvAttnDefaultConfig::qkv_output];
        alignas (4) int8_t init_qkv_bias[ConvAttnDefaultConfig::qkv_output];
        alignas (4) int8_t init_head1_weights[ConvAttnDefaultConfig::pooled * 3u];
        alignas (4) int8_t init_head1_bias[3u];
        alignas (4) int8_t init_head2_weights[ConvAttnDefaultConfig::pooled * 1u];
        alignas (4) int8_t init_head2_bias[1u];
        alignas (4) int8_t init_head3_weights[ConvAttnDefaultConfig::pooled * 1u];
        alignas (4) int8_t init_head3_bias[1u];

        if (!_M_trained) {
            for (uint32_t i = 0; i < ConvAttnDefaultConfig::conv_channels * ConvAttnDefaultConfig::feature_count
                                         * ConvAttnDefaultConfig::conv_kernel;
                 ++i)
                init_conv_weights[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
            for (uint32_t i = 0; i < ConvAttnDefaultConfig::conv_channels; ++i)
                init_conv_bias[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);
            for (uint32_t i = 0; i < ConvAttnDefaultConfig::conv_output * ConvAttnDefaultConfig::qkv_output; ++i)
                init_qkv_weights[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
            for (uint32_t i = 0; i < ConvAttnDefaultConfig::qkv_output; ++i)
                init_qkv_bias[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);
            for (uint32_t i = 0; i < ConvAttnDefaultConfig::pooled * 3u; ++i)
                init_head1_weights[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
            for (uint32_t i = 0; i < 3u; ++i)
                init_head1_bias[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);
            for (uint32_t i = 0; i < ConvAttnDefaultConfig::pooled * 1u; ++i)
                init_head2_weights[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
            for (uint32_t i = 0; i < 1u; ++i)
                init_head2_bias[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);
            for (uint32_t i = 0; i < ConvAttnDefaultConfig::pooled * 1u; ++i)
                init_head3_weights[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
            for (uint32_t i = 0; i < 1u; ++i)
                init_head3_bias[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);
            conv_weights = init_conv_weights;
            conv_bias = init_conv_bias;
            qkv_weights = init_qkv_weights;
            qkv_bias = init_qkv_bias;
            head1_weights = init_head1_weights;
            head1_bias = init_head1_bias;
            head2_weights = init_head2_weights;
            head2_bias = init_head2_bias;
            head3_weights = init_head3_weights;
            head3_bias = init_head3_bias;
        }

        return serialize_multitask_model (dst, conv_weights, conv_bias, qkv_weights, qkv_bias, head1_weights,
                                          head1_bias, head2_weights, head2_bias, head3_weights, head3_bias,
                                          kRequantMult);
    }

    /** @brief Validate a model blob: magic, version, CRC, layer count. */
    [[nodiscard]] static Status
    validate (const uint8_t *blob, uint32_t blob_len) noexcept {
        return validate_model (blob, blob_len);
    }

    /** @brief Number of bytes in the model blob. */
    [[nodiscard]] static constexpr uint32_t
    blob_bytes () noexcept {
        return kBlobSize;
    }

    /** @brief Whether the weights have been trained. */
    [[nodiscard]] constexpr bool
    trained () const noexcept {
        return _M_trained;
    }

    /** @brief Set the trained flag. */
    constexpr void
    set_trained (bool trained) noexcept {
        this->_M_trained = trained;
    }

    /** @brief Direct access to hidden weight codes: [out * in_dim + in]. */
    [[nodiscard]] int8_t *
    hidden_weights () noexcept {
        return this->_M_hidden_weights;
    }

    [[nodiscard]] const int8_t *
    hidden_weights () const noexcept {
        return this->_M_hidden_weights;
    }

    /** @brief Direct access to hidden bias codes: [out]. */
    [[nodiscard]] int8_t *
    hidden_bias () noexcept {
        return this->_M_hidden_bias;
    }

    [[nodiscard]] const int8_t *
    hidden_bias () const noexcept {
        return this->_M_hidden_bias;
    }

    /** @brief Direct access to output weight codes: [out * in_dim + in]. */
    [[nodiscard]] int8_t *
    weights () noexcept {
        return this->_M_weights;
    }

    [[nodiscard]] const int8_t *
    weights () const noexcept {
        return this->_M_weights;
    }

    /** @brief Direct access to output bias codes: [out]. */
    [[nodiscard]] int8_t *
    bias () noexcept {
        return this->_M_bias;
    }

    [[nodiscard]] const int8_t *
    bias () const noexcept {
        return this->_M_bias;
    }

    /** @brief Direct access to conv weight codes: [out_channel * in_channels * kernel + ...]. */
    [[nodiscard]] int8_t *
    conv_weights () noexcept {
        return this->_M_conv_weights;
    }

    [[nodiscard]] const int8_t *
    conv_weights () const noexcept {
        return this->_M_conv_weights;
    }

    /** @brief Direct access to conv bias codes: [out_channel]. */
    [[nodiscard]] int8_t *
    conv_bias () noexcept {
        return this->_M_conv_bias;
    }

    [[nodiscard]] const int8_t *
    conv_bias () const noexcept {
        return this->_M_conv_bias;
    }

    /** @brief Direct access to QKV weight codes: [out * in_dim + in]. */
    [[nodiscard]] int8_t *
    qkv_weights () noexcept {
        return this->_M_qkv_weights;
    }

    [[nodiscard]] const int8_t *
    qkv_weights () const noexcept {
        return this->_M_qkv_weights;
    }

    /** @brief Direct access to QKV bias codes: [out]. */
    [[nodiscard]] int8_t *
    qkv_bias () noexcept {
        return this->_M_qkv_bias;
    }

    [[nodiscard]] const int8_t *
    qkv_bias () const noexcept {
        return this->_M_qkv_bias;
    }

    /** @brief Direct access to output weight codes: [out * in_dim + in]. */
    [[nodiscard]] int8_t *
    out_weights () noexcept {
        return this->_M_out_weights;
    }

    [[nodiscard]] const int8_t *
    out_weights () const noexcept {
        return this->_M_out_weights;
    }

    /** @brief Direct access to output bias codes: [out]. */
    [[nodiscard]] int8_t *
    out_bias () noexcept {
        return this->_M_out_bias;
    }

    [[nodiscard]] const int8_t *
    out_bias () const noexcept {
        return this->_M_out_bias;
    }

    /** @brief Direct access to head 1 weight codes: [out * in_dim + in]. */
    [[nodiscard]] int8_t *
    head1_weights () noexcept {
        return this->_M_head1_weights;
    }

    [[nodiscard]] const int8_t *
    head1_weights () const noexcept {
        return this->_M_head1_weights;
    }

    /** @brief Direct access to head 1 bias codes: [out]. */
    [[nodiscard]] int8_t *
    head1_bias () noexcept {
        return this->_M_head1_bias;
    }

    [[nodiscard]] const int8_t *
    head1_bias () const noexcept {
        return this->_M_head1_bias;
    }

    /** @brief Direct access to head 2 weight codes: [out * in_dim + in]. */
    [[nodiscard]] int8_t *
    head2_weights () noexcept {
        return this->_M_head2_weights;
    }

    [[nodiscard]] const int8_t *
    head2_weights () const noexcept {
        return this->_M_head2_weights;
    }

    /** @brief Direct access to head 2 bias codes: [out]. */
    [[nodiscard]] int8_t *
    head2_bias () noexcept {
        return this->_M_head2_bias;
    }

    [[nodiscard]] const int8_t *
    head2_bias () const noexcept {
        return this->_M_head2_bias;
    }

    /** @brief Direct access to head 3 weight codes: [out * in_dim + in]. */
    [[nodiscard]] int8_t *
    head3_weights () noexcept {
        return this->_M_head3_weights;
    }

    [[nodiscard]] const int8_t *
    head3_weights () const noexcept {
        return this->_M_head3_weights;
    }

    /** @brief Direct access to head 3 bias codes: [out]. */
    [[nodiscard]] int8_t *
    head3_bias () noexcept {
        return this->_M_head3_bias;
    }

    [[nodiscard]] const int8_t *
    head3_bias () const noexcept {
        return this->_M_head3_bias;
    }

    /** @brief Number of bytes in the multi-task model blob. */
    [[nodiscard]] static constexpr uint32_t
    multitask_blob_bytes () noexcept {
        return kMultiTaskBlobSize;
    }

    /** @brief Number of bytes in the conv-attn model blob. */
    [[nodiscard]] static constexpr uint32_t
    conv_attn_blob_bytes () noexcept {
        return kConvAttnBlobSize;
    }

  private:
    alignas (4) int8_t _M_hidden_weights[Config::hidden_input * Config::hidden_dim] = {};
    alignas (4) int8_t _M_hidden_bias[Config::hidden_dim] = {};
    alignas (4) int8_t _M_weights[Config::hidden_dim * Config::output_dim] = {};
    alignas (4) int8_t _M_bias[Config::output_dim] = {};
    /* Tier 2 CONV1D + Attention weights (sized by ConvAttnDefaultConfig). */
    alignas (4) int8_t _M_conv_weights[ConvAttnDefaultConfig::conv_channels * ConvAttnDefaultConfig::feature_count
                                       * ConvAttnDefaultConfig::conv_kernel] = {};
    alignas (4) int8_t _M_conv_bias[ConvAttnDefaultConfig::conv_channels] = {};
    alignas (4) int8_t _M_qkv_weights[ConvAttnDefaultConfig::conv_output * ConvAttnDefaultConfig::qkv_output] = {};
    alignas (4) int8_t _M_qkv_bias[ConvAttnDefaultConfig::qkv_output] = {};
    alignas (4) int8_t _M_out_weights[ConvAttnDefaultConfig::pooled * ConvAttnDefaultConfig::output_dim] = {};
    alignas (4) int8_t _M_out_bias[ConvAttnDefaultConfig::output_dim] = {};
    /* Tier 3 multi-task head weights (shared encoder output is kPooled = 8). */
    alignas (4) int8_t _M_head1_weights[ConvAttnDefaultConfig::pooled * 3u] = {}; /* head 1: 8->3 */
    alignas (4) int8_t _M_head1_bias[3u] = {};
    alignas (4) int8_t _M_head2_weights[ConvAttnDefaultConfig::pooled * 1u] = {}; /* head 2: 8->1 */
    alignas (4) int8_t _M_head2_bias[1u] = {};
    alignas (4) int8_t _M_head3_weights[ConvAttnDefaultConfig::pooled * 1u] = {}; /* head 3: 8->1 */
    alignas (4) int8_t _M_head3_bias[1u] = {};
    bool _M_trained = false;
};

/** @brief Trained hidden weight codes: [out * in_dim + in] = 3040 int8 codes. */
alignas (4) inline int8_t g_model_hidden_weights[kHiddenInput * kHiddenDim] = {};

/** @brief Trained hidden bias codes: [out] = 16 int8 codes. */
alignas (4) inline int8_t g_model_hidden_bias[kHiddenDim] = {};

/** @brief Trained output weight codes: [out * in_dim + in] = 48 int8 codes. */
alignas (4) inline int8_t g_model_weights[kHiddenDim * kOutputDim] = {};

/** @brief Trained output bias codes: [out] = 3 int8 codes. */
alignas (4) inline int8_t g_model_bias[kOutputDim] = {};

/** @brief Whether the weights have been trained (set by training pipeline). */
inline bool g_model_trained = false;

/** @brief Build the complete model blob at compile time.
 *
 * Returns the blob in a caller-owned buffer.  Uses trained weights if
 * g_model_trained is true, otherwise deterministic initialization.
 */
[[nodiscard]] inline Status
build_model_blob (uint8_t *blob) noexcept {
    if (blob == nullptr)
        return zzs::mllee::Status::BAD_LAYOUT;

    /* Use trained weights if available, else deterministic initialization. */
    const int8_t *hidden_weights = g_model_trained ? g_model_hidden_weights : nullptr;
    const int8_t *hidden_bias = g_model_trained ? g_model_hidden_bias : nullptr;
    const int8_t *output_weights = g_model_trained ? g_model_weights : nullptr;
    const int8_t *output_bias = g_model_trained ? g_model_bias : nullptr;

    /* Deterministic fallback weights (same as training init). */
    alignas (4) int8_t init_hidden_weights[kHiddenInput * kHiddenDim];
    alignas (4) int8_t init_hidden_bias[kHiddenDim];
    alignas (4) int8_t init_output_weights[kHiddenDim * kOutputDim];
    alignas (4) int8_t init_output_bias[kOutputDim];

    if (!g_model_trained) {
        for (uint32_t i = 0; i < kHiddenInput * kHiddenDim; ++i)
            init_hidden_weights[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
        for (uint32_t i = 0; i < kHiddenDim; ++i)
            init_hidden_bias[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);
        for (uint32_t i = 0; i < kHiddenDim * kOutputDim; ++i)
            init_output_weights[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
        for (uint32_t i = 0; i < kOutputDim; ++i)
            init_output_bias[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);
        hidden_weights = init_hidden_weights;
        hidden_bias = init_hidden_bias;
        output_weights = init_output_weights;
        output_bias = init_output_bias;
    }

    return serialize_model (blob, hidden_weights, hidden_bias, output_weights, output_bias, kRequantMult);
}

/** @brief The compile-time model blob (stored in .rodata).
 *
 * Defined in a build-time generated source file (model_blob_generated.cpp)
 * so all translation units share the same serialized instance.  The test
 * translation unit provides the definition; other TUs see this declaration.
 */
extern uint8_t g_model_blob[kBlobSize];

/** @brief Initialize the blob from trained or init weights. */
[[nodiscard]] inline Status
init_model_blob () noexcept {
    return build_model_blob (g_model_blob);
}

/** @brief Validate the current blob. */
[[nodiscard]] inline Status
validate_model_blob () noexcept {
    return validate_model (g_model_blob, kBlobSize);
}

/** @brief Number of bytes in the model blob. */
[[nodiscard]] inline constexpr uint32_t
model_blob_bytes () noexcept {
    return kBlobSize;
}

/** @} */
} // namespace zzs::mllee_native::model
