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
#include "mllee-native/q11_q15_q31.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"
#include "mllee/train_trainer.hpp"
#include "std/io.hpp"

#include <cstdint>

/**
 * @brief Training pipeline for power outage detection using mllee/.
 *
 * Two-layer MLP: LINEAR(190->16) + ACTIVATION(RELU) + LINEAR(16->3) with
 * mean-squared-error loss, trained by SGD through the mllee::train::Trainer.
 * All storage is caller-owned; nothing allocates.  Training progress is logged
 * via zzs::base::log_* (redirect stdout to ./.tmp/training.log at the shell).
 * @{
 */
namespace zzs::mllee_native::pipeline {

using zzs::mllee::Status;
using zzs::mllee::status_ok;
using zzs::mllee::train::Flag;
using zzs::mllee::train::Pool;
using zzs::mllee::train::Shape;
using zzs::mllee::train::StepContext;
using zzs::mllee::train::StepHooks;
using zzs::mllee::train::StepStats;
using zzs::mllee::train::Tape;
using zzs::mllee::train::Trainable;
using zzs::mllee::train::Trainer;

using zzs::mllee_native::dataset::kClassCount;
using zzs::mllee_native::dataset::kFeatureCount;
using zzs::mllee_native::dataset::kSeqLen;
using zzs::mllee_native::dataset::Label;
using zzs::mllee_native::dataset::TrainingSample;
using zzs::mllee_native::fx::Q15;
using zzs::mllee_native::fx::q15_abs;
using zzs::mllee_native::fx::q15_add;
using zzs::mllee_native::fx::q15_mul;

using zzs::mllee_native::DefaultConfig;
using zzs::mllee_native::ModelConfig;

/** @brief Input dimension: flattened input size (seq_len * feature_count). */
inline constexpr uint32_t kInputDim = DefaultConfig::flat_input;
/** @brief Hidden dimension: number of hidden units. */
inline constexpr uint32_t kHiddenDim = DefaultConfig::hidden_dim;
/** @brief Hidden layer input size (equals flat_input). */
inline constexpr uint32_t kHiddenInput = DefaultConfig::hidden_input;
/** @brief Output dimension: number of classes. */
inline constexpr uint32_t kOutputDim = DefaultConfig::output_dim;
/** @brief Flattened input size: seq_len * feature_count. */
inline constexpr uint32_t kFlatInput = DefaultConfig::flat_input;

/** @brief Forward hook: two-layer MLP + MSE loss. */
[[nodiscard]] Status
mlp_forward_hook (StepContext &context) noexcept;

/** @brief Backward function for the two-layer MLP + MSE layer. */
[[nodiscard]] Status
mlp_backward_fn (const Trainable &output, const Trainable &grad_output,
                 zzs::mllee::train::GradContext &context) noexcept;

/** @brief Schedule hook: constant learning rate (0.05 in Q15). */
[[nodiscard]] int32_t
constant_lr_schedule (uint32_t step) noexcept;

/** @brief Update hook: SGD (w -= lr * grad). */
[[nodiscard]] Status
sgd_update_hook (StepContext &context) noexcept;

/** @brief Post-grad hook: count saturations, optionally clip. */
[[nodiscard]] Status
post_grad_hook (StepContext &context, uint32_t saturated) noexcept;

/** @brief One-hot encode a label as int8 codes (127 for true class). */
constexpr void
one_hot_target (const uint8_t label, int8_t *const target) noexcept {
  for (uint32_t i = 0; i < DefaultConfig::output_dim; ++i)
    target[i] = (static_cast<uint32_t> (i) == static_cast<uint32_t> (label)) ? 127 : -127;
}

/** @brief Trainable parameter block for the hidden linear layer. */
struct HiddenParams {
  Trainable _M_weight; /* [hidden_dim, hidden_input], requires grad */
  Trainable _M_bias;   /* [hidden_dim],   requires grad */
};

/** @brief Trainable parameter block for the output linear layer. */
struct OutputParams {
  Trainable _M_weight; /* [output_dim, hidden_dim], requires grad */
  Trainable _M_bias;   /* [output_dim],   requires grad */
  Trainable _M_output; /* [output_dim],   forward-only (no grad) */
  Trainable _M_loss;   /* scalar, forward-only */
};

/**
 * @brief All caller-owned storage for training, sized by Config.
 *
 * Replaces the old hardcoded POD struct with fixed 16KB arena, 256 tape nodes,
 * and fixed weight/grad buffers.  All dimensions come from Config.
 */
template <typename Config = DefaultConfig> struct TrainingStorage {
  /* Arena: 16 KiB bump-allocated pool for tape + saved tensors. */
  alignas (4) uint8_t _M_arena[16384];
  Pool _M_pool;

  /* Tape nodes: 256 nodes is plenty for a two-layer MLP. */
  alignas (8) zzs::mllee::train::Node _M_nodes[256];
  Tape _M_tape;

  /* Hidden layer weight/bias code + grad buffers, sized by Config. */
  alignas (4) int8_t _M_hidden_w_code[Config::hidden_input * Config::hidden_dim];
  alignas (8) int16_t _M_hidden_w_grad[Config::hidden_input * Config::hidden_dim];
  alignas (4) int8_t _M_hidden_b_code[Config::hidden_dim];
  alignas (8) int16_t _M_hidden_b_grad[Config::hidden_dim];

  /* Output layer weight/bias code + grad buffers, sized by Config. */
  alignas (4) int8_t _M_w_code[Config::hidden_dim * Config::output_dim];
  alignas (8) int16_t _M_w_grad[Config::hidden_dim * Config::output_dim];
  alignas (4) int8_t _M_b_code[Config::output_dim];
  alignas (8) int16_t _M_b_grad[Config::output_dim];

  /* Output + loss buffers. */
  alignas (4) int8_t _M_out_code[Config::output_dim];
  alignas (8) int16_t _M_out_grad[Config::output_dim];
  alignas (4) int8_t _M_loss_code[1];
  alignas (8) int16_t _M_loss_grad[1];

  /* Input + target views (set per step). */
  alignas (4) int8_t _M_input_code[Config::flat_input];
  alignas (4) int8_t _M_target_code[Config::output_dim];

  HiddenParams _M_hidden_params;
  OutputParams _M_output_params;

  TrainingStorage () noexcept;
  void
  reset () noexcept;
};

/** @brief Non-template alias for default-config storage (backward compatibility). */
using TrainingStorageDefault = TrainingStorage<DefaultConfig>;

/**
 * @brief OOP training pipeline: wraps TrainingStorage and owns the step loop.
 *
 * Instantiating a TrainingPipeline bundles the storage, pool, tape, trainer
 * and hooks into one object.  All state is inside the object; no globals.
 */
template <typename Config = DefaultConfig> class TrainingPipeline {
public:
  TrainingPipeline () noexcept = default;
  TrainingPipeline (const TrainingPipeline &) = delete;
  TrainingPipeline &
  operator= (const TrainingPipeline &) = delete;

  /** @brief Run training for @p steps on the compile-time dataset. */
  [[nodiscard]] Status
  run_training (uint32_t steps, zzs::base::Stdout &out) noexcept;

  /** @brief Evaluate the model on the full dataset, report accuracy. */
  [[nodiscard]] Status
  evaluate_model (zzs::base::Stdout &out) noexcept;

  /** @brief Access the underlying storage. */
  [[nodiscard]] TrainingStorage<Config> &
  storage () noexcept {
    return this->_M_storage;
  }

  [[nodiscard]] const TrainingStorage<Config> &
  storage () const noexcept {
    return this->_M_storage;
  }

private:
  TrainingStorage<Config> _M_storage;
};

/** @brief Non-template alias for the default-config pipeline. */
using TrainingPipelineDefault = TrainingPipeline<DefaultConfig>;

/** @brief Run training for @p steps on the compile-time dataset. */
[[nodiscard]] Status
run_training (TrainingStorageDefault &storage, uint32_t steps, zzs::base::Stdout &out) noexcept;

/** @brief Evaluate the model on the full dataset, report accuracy. */
[[nodiscard]] Status
evaluate_model (TrainingStorageDefault &storage, zzs::base::Stdout &out) noexcept;

/* ======================================================================== */
/* Tier 2: CONV1D + Attention training pipeline                             */
/* ======================================================================== */

/** @brief Forward hook: CONV1D -> LINEAR(QKV) -> ATTENTION -> GLOBAL_AVG_POOL
 *         -> LINEAR -> SOFTMAX with MSE loss. */
[[nodiscard]] Status
conv_attn_forward_hook (StepContext &context) noexcept;

/** @brief Backward function for the MSE loss over the conv-attn output. */
[[nodiscard]] Status
conv_attn_backward_fn (const Trainable &output, const Trainable &grad_output,
                       zzs::mllee::train::GradContext &context) noexcept;

/** @brief Trainable parameter block for the conv-attn pipeline. */
struct ConvAttnParams {
  /* Layer weights (require grad). */
  Trainable _M_conv_w; /* [conv_channels, feature_count * conv_kernel] */
  Trainable _M_conv_b; /* [conv_channels] */
  Trainable _M_qkv_w;  /* [qkv_output, conv_output] */
  Trainable _M_qkv_b;  /* [qkv_output] */
  Trainable _M_out_w;  /* [output_dim, pooled] */
  Trainable _M_out_b;  /* [output_dim] */

  /* Layer activations (forward output + grad lanes; frozen from updates). */
  Trainable _M_conv_out;  /* [conv_output]  = 40 */
  Trainable _M_qkv_out;   /* [qkv_output]   = 120 (Q+K+V concatenated) */
  Trainable _M_attn_out;  /* [attn_output]  = 40 */
  Trainable _M_pool_out;  /* [pooled]       = 8 */
  Trainable _M_model_out; /* [output_dim]   = 3 */

  /* Loss node. */
  Trainable _M_loss; /* scalar */
};

/**
 * @brief All caller-owned storage for conv-attn training, sized by Config.
 *
 * 16 KiB bump pool for tape + saved tensors.  Weight and grad buffers are
 * sized by Config.  Activation buffers live in separate arrays so the tape
 * pool is never used for activation storage -- only for saved tensors and
 * scratch.
 */
template <typename Config = ConvAttnDefaultConfig> struct ConvAttnStorage {
  /* Arena: 16 KiB bump-allocated pool for tape + saved tensors. */
  alignas (4) uint8_t _M_arena[16384];
  Pool _M_pool;

  /* Tape nodes: enough for 7 layer nodes plus the loss node. */
  alignas (8) zzs::mllee::train::Node _M_nodes[256];
  Tape _M_tape;

  /* Conv1D weights: [conv_channels, feature_count * conv_kernel]. */
  alignas (4) int8_t _M_conv_w_code[Config::conv_channels * Config::feature_count * Config::conv_kernel];
  alignas (8) int16_t _M_conv_w_grad[Config::conv_channels * Config::feature_count * Config::conv_kernel];
  alignas (4) int8_t _M_conv_b_code[Config::conv_channels];
  alignas (8) int16_t _M_conv_b_grad[Config::conv_channels];

  /* QKV LINEAR weights: [qkv_output, conv_output]. */
  alignas (4) int8_t _M_qkv_w_code[Config::qkv_output * Config::conv_output];
  alignas (8) int16_t _M_qkv_w_grad[Config::qkv_output * Config::conv_output];
  alignas (4) int8_t _M_qkv_b_code[Config::qkv_output];
  alignas (8) int16_t _M_qkv_b_grad[Config::qkv_output];

  /* Output LINEAR weights: [output_dim, pooled]. */
  alignas (4) int8_t _M_out_w_code[Config::output_dim * Config::pooled];
  alignas (8) int16_t _M_out_w_grad[Config::output_dim * Config::pooled];
  alignas (4) int8_t _M_out_b_code[Config::output_dim];
  alignas (8) int16_t _M_out_b_grad[Config::output_dim];

  /* Activation code + grad buffers. */
  alignas (4) int8_t _M_conv_out_code[Config::conv_output];
  alignas (8) int16_t _M_conv_out_grad[Config::conv_output];
  alignas (4) int8_t _M_qkv_out_code[Config::qkv_output];
  alignas (8) int16_t _M_qkv_out_grad[Config::qkv_output];
  alignas (4) int8_t _M_attn_out_code[Config::attn_output];
  alignas (8) int16_t _M_attn_out_grad[Config::attn_output];
  alignas (4) int8_t _M_pool_out_code[Config::pooled];
  alignas (8) int16_t _M_pool_out_grad[Config::pooled];
  alignas (4) int8_t _M_model_out_code[Config::output_dim];
  alignas (8) int16_t _M_model_out_grad[Config::output_dim];

  /* Input + target (set per step). */
  alignas (4) int8_t _M_input_code[Config::flat_input];
  alignas (4) int8_t _M_target_code[Config::output_dim];

  /* Loss. */
  alignas (4) int8_t _M_loss_code[1];
  alignas (8) int16_t _M_loss_grad[1];

  ConvAttnParams _M_params;

  ConvAttnStorage () noexcept;
  void
  reset () noexcept;
};

/** @brief Non-template alias for default-config storage. */
using ConvAttnStorageDefault = ConvAttnStorage<ConvAttnDefaultConfig>;

/**
 * @brief OOP training pipeline for the Tier 2 CONV1D + Attention model.
 *
 * Wraps ConvAttnStorage and owns the step loop.  All state is inside the
 * object; no globals.
 */
template <typename Config = ConvAttnDefaultConfig> class ConvAttnPipeline {
public:
  ConvAttnPipeline () noexcept = default;
  ConvAttnPipeline (const ConvAttnPipeline &) = delete;
  ConvAttnPipeline &
  operator= (const ConvAttnPipeline &) = delete;

  /** @brief Run training for @p steps on the compile-time dataset. */
  [[nodiscard]] Status
  run_training (uint32_t steps, zzs::base::Stdout &out) noexcept;

  /** @brief Evaluate the model on the full dataset, report accuracy. */
  [[nodiscard]] Status
  evaluate_model (zzs::base::Stdout &out) noexcept;

  /** @brief Access the underlying storage. */
  [[nodiscard]] ConvAttnStorage<Config> &
  storage () noexcept {
    return this->_M_storage;
  }

  [[nodiscard]] const ConvAttnStorage<Config> &
  storage () const noexcept {
    return this->_M_storage;
  }

private:
  ConvAttnStorage<Config> _M_storage;
};

/** @brief Non-template alias for the default-config pipeline. */
using ConvAttnPipelineDefault = ConvAttnPipeline<ConvAttnDefaultConfig>;

/** @brief Run training for @p steps on the compile-time dataset. */
[[nodiscard]] Status
run_training (ConvAttnStorageDefault &storage, uint32_t steps, zzs::base::Stdout &out) noexcept;

/** @brief Evaluate the model on the full dataset, report accuracy. */
[[nodiscard]] Status
evaluate_model (ConvAttnStorageDefault &storage, zzs::base::Stdout &out) noexcept;

/* ======================================================================== */
/* Tier 4: Multi-task training pipeline                                     */
/* ======================================================================== */

/** @brief Forward hook: CONV1D -> LINEAR(QKV) -> three LINEAR heads. */
[[nodiscard]] Status
multitask_forward_hook (StepContext &context) noexcept;

/** @brief Backward function for the multi-task loss. */
[[nodiscard]] Status
multitask_backward_fn (const Trainable &output, const Trainable &grad_output,
                       zzs::mllee::train::GradContext &context) noexcept;

/** @brief Forward hook: student forward with teacher logits for distillation. */
[[nodiscard]] Status
distill_forward_hook (StepContext &context, const int8_t *teacher_logits, uint32_t teacher_count) noexcept;

/** @brief Trainable parameter block for the multi-task pipeline. */
struct MultiTaskParams {
  /* Layer weights (require grad). */
  Trainable _M_conv_w;  /* [conv_channels, feature_count * conv_kernel] */
  Trainable _M_conv_b;  /* [conv_channels] */
  Trainable _M_qkv_w;   /* [qkv_output, conv_output] */
  Trainable _M_qkv_b;   /* [qkv_output] */
  Trainable _M_head1_w; /* [head1_out, pooled] */
  Trainable _M_head1_b; /* [head1_out] */
  Trainable _M_head2_w; /* [head2_out, pooled] */
  Trainable _M_head2_b; /* [head2_out] */
  Trainable _M_head3_w; /* [head3_out, pooled] */
  Trainable _M_head3_b; /* [head3_out] */

  /* Layer activations / outputs (forward output + grad lanes). */
  Trainable _M_output; /* [output_dim] */
  Trainable _M_loss;   /* scalar */
};

/* ======================================================================== */
/* Tier 5: Distillation training pipeline                                   */
/* ======================================================================== */

/** @brief Forward hook: student CONV1D+Attention+3-head forward with a
 *         teacher-blended soft target and MSE distillation loss. */
[[nodiscard]] Status
distill_forward_hook (StepContext &context, const int8_t *teacher_logits, uint32_t teacher_count) noexcept;

/** @brief Trainable parameter block for the distill pipeline.
 *
 * Param layout: 0=conv_w 1=conv_b 2=qkv_w 3=qkv_b
 *               4=head1_w 5=head1_b 6=head2_w 7=head2_b
 *               8=head3_w 9=head3_b 10=output 11=loss (12 params)
 */
struct DistillParams {
  /* Layer weights (require grad). */
  Trainable _M_conv_w;  /* [conv_channels, feature_count * conv_kernel] */
  Trainable _M_conv_b;  /* [conv_channels] */
  Trainable _M_qkv_w;   /* [qkv_output, conv_output] */
  Trainable _M_qkv_b;   /* [qkv_output] */
  Trainable _M_head1_w; /* [head1_out, pooled] */
  Trainable _M_head1_b; /* [head1_out] */
  Trainable _M_head2_w; /* [head2_out, pooled] */
  Trainable _M_head2_b; /* [head2_out] */
  Trainable _M_head3_w; /* [head3_out, pooled] */
  Trainable _M_head3_b; /* [head3_out] */

  /* Layer activations / outputs (forward output + grad lanes). */
  Trainable _M_output; /* [output_dim] */
  Trainable _M_loss;   /* scalar */
};

/**
 * @brief All caller-owned storage for distill training, sized by Config.
 *
 * Mirrors ConvAttnStorage: 16 KiB bump pool, 256 tape nodes, weight/code +
 * grad buffers sized by ConvAttnDefaultConfig, activation buffers,
 * input/target, loss.
 */
template <typename Config = ConvAttnDefaultConfig> struct DistillStorage {
  /* Arena: 16 KiB bump-allocated pool for tape + saved tensors. */
  alignas (4) uint8_t _M_arena[16384];
  Pool _M_pool;

  /* Tape nodes: 256 nodes. */
  alignas (8) zzs::mllee::train::Node _M_nodes[256];
  Tape _M_tape;

  /* Conv1D weights: [conv_channels, feature_count * conv_kernel]. */
  alignas (4) int8_t _M_conv_w_code[Config::conv_channels * Config::feature_count * Config::conv_kernel];
  alignas (8) int16_t _M_conv_w_grad[Config::conv_channels * Config::feature_count * Config::conv_kernel];
  alignas (4) int8_t _M_conv_b_code[Config::conv_channels];
  alignas (8) int16_t _M_conv_b_grad[Config::conv_channels];

  /* QKV LINEAR weights: [qkv_output, conv_output]. */
  alignas (4) int8_t _M_qkv_w_code[Config::qkv_output * Config::conv_output];
  alignas (8) int16_t _M_qkv_w_grad[Config::qkv_output * Config::conv_output];
  alignas (4) int8_t _M_qkv_b_code[Config::qkv_output];
  alignas (8) int16_t _M_qkv_b_grad[Config::qkv_output];

  /* Output LINEAR weights: [output_dim, pooled]. */
  alignas (4) int8_t _M_head1_w_code[Config::output_dim * Config::pooled];
  alignas (8) int16_t _M_head1_w_grad[Config::output_dim * Config::pooled];
  alignas (4) int8_t _M_head1_b_code[Config::output_dim];
  alignas (8) int16_t _M_head1_b_grad[Config::output_dim];

  alignas (4) int8_t _M_head2_w_code[Config::pooled];
  alignas (8) int16_t _M_head2_w_grad[Config::pooled];
  alignas (4) int8_t _M_head2_b_code[1];
  alignas (8) int16_t _M_head2_b_grad[1];

  alignas (4) int8_t _M_head3_w_code[Config::pooled];
  alignas (8) int16_t _M_head3_w_grad[Config::pooled];
  alignas (4) int8_t _M_head3_b_code[1];
  alignas (8) int16_t _M_head3_b_grad[1];

  /* Activation code + grad buffers. */
  alignas (4) int8_t _M_conv_out_code[Config::conv_output];
  alignas (8) int16_t _M_conv_out_grad[Config::conv_output];
  alignas (4) int8_t _M_qkv_out_code[Config::qkv_output];
  alignas (8) int16_t _M_qkv_out_grad[Config::qkv_output];
  alignas (4) int8_t _M_attn_out_code[Config::attn_output];
  alignas (8) int16_t _M_attn_out_grad[Config::attn_output];
  alignas (4) int8_t _M_pool_out_code[Config::pooled];
  alignas (8) int16_t _M_pool_out_grad[Config::pooled];
  alignas (4) int8_t _M_output_code[Config::output_dim];
  alignas (8) int16_t _M_output_grad[Config::output_dim];

  /* Input + target (set per step). */
  alignas (4) int8_t _M_input_code[Config::flat_input];
  alignas (4) int8_t _M_target_code[Config::output_dim];

  /* Loss. */
  alignas (4) int8_t _M_loss_code[1];
  alignas (8) int16_t _M_loss_grad[1];

  DistillParams _M_params;

  DistillStorage () noexcept;
  void
  reset () noexcept;
};

/** @brief Non-template alias for default-config storage. */
using DistillStorageDefault = DistillStorage<ConvAttnDefaultConfig>;

/**
 * @brief OOP training pipeline for distillation.
 *
 * Wraps DistillStorage and owns the step loop.  run_training wires
 * hooks._M_forward = distill_forward_hook.  All state is inside the object;
 * no globals.
 */
template <typename Config = ConvAttnDefaultConfig> class DistillPipeline {
public:
  DistillPipeline () noexcept = default;
  DistillPipeline (const DistillPipeline &) = delete;
  DistillPipeline &
  operator= (const DistillPipeline &) = delete;

  /** @brief Run training for @p steps on the compile-time dataset. */
  [[nodiscard]] Status
  run_training (uint32_t steps, zzs::base::Stdout &out) noexcept;

  /** @brief Evaluate the model on the full dataset, report accuracy. */
  [[nodiscard]] Status
  evaluate_model (zzs::base::Stdout &out) noexcept;

  /** @brief Access the underlying storage. */
  [[nodiscard]] DistillStorage<Config> &
  storage () noexcept {
    return this->_M_storage;
  }

  [[nodiscard]] const DistillStorage<Config> &
  storage () const noexcept {
    return this->_M_storage;
  }

private:
  DistillStorage<Config> _M_storage;
};

/** @brief Non-template alias for the default-config pipeline. */
using DistillPipelineDefault = DistillPipeline<ConvAttnDefaultConfig>;

/** @brief Run training for @p steps on the compile-time dataset. */
[[nodiscard]] Status
run_training (DistillStorageDefault &storage, uint32_t steps, zzs::base::Stdout &out) noexcept;

/** @brief Evaluate the model on the full dataset, report accuracy. */
[[nodiscard]] Status
evaluate_model (DistillStorageDefault &storage, zzs::base::Stdout &out) noexcept;

/** @brief Wire up the DistillParams Trainable views to storage buffers. */
template <typename Config>
void
setup_distill_params (DistillParams &params, int8_t *conv_w_code, int16_t *conv_w_grad, int8_t *conv_b_code,
                      int16_t *conv_b_grad, int8_t *qkv_w_code, int16_t *qkv_w_grad, int8_t *qkv_b_code,
                      int16_t *qkv_b_grad, int8_t *head1_w_code, int16_t *head1_w_grad, int8_t *head1_b_code,
                      int16_t *head1_b_grad, int8_t *head2_w_code, int16_t *head2_w_grad, int8_t *head2_b_code,
                      int16_t *head2_b_grad, int8_t *head3_w_code, int16_t *head3_w_grad, int8_t *head3_b_code,
                      int16_t *head3_b_grad, int8_t *output_code, int16_t *output_grad, int8_t *loss_code,
                      int16_t *loss_grad) noexcept;

/** @} */
} // namespace zzs::mllee_native::pipeline
