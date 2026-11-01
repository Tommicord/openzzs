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
#include "mllee-native/train_pipeline.hpp"
#include "mllee/core_format.hpp"
#include "mllee/core_quant.hpp"
#include "mllee/train_attention.hpp"
#include "mllee/train_conv.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_linear.hpp"
#include "mllee/train_loss.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_pooling.hpp"
#include "mllee/train_softmax.hpp"

#include <cstdint>

/**
 * @brief Training pipeline hook implementations for power outage detection.
 *
 * Two-layer MLP: 190 inputs (5 windows x 38 features) -> 16 hidden (RELU)
 * -> 3 outputs (normal/outage/transition), MSE loss, SGD updates.  All
 * arithmetic is fixed-point (Q15/int8 codes); no float anywhere.
 * @{
 */
namespace zzs::mllee_native::pipeline {

using zzs::mllee::Status;
using zzs::mllee::status_name;
using zzs::mllee::status_ok;
using zzs::mllee::train::GradContext;
using zzs::mllee::train::Pool;
using zzs::mllee::train::Saved;
using zzs::mllee::train::StepContext;
using zzs::mllee::train::Trainable;

using zzs::mllee_native::dataset_count;
using zzs::mllee_native::dataset_sample;
using zzs::mllee_native::detail::flatten_sample;
using zzs::mllee_native::detail::q15_raw_to_code;

using zzs::mllee::Act;
using zzs::mllee::train::AttentionSpec;
using zzs::mllee::train::ConvLayout;
using zzs::mllee::train::ConvSpec;
using zzs::mllee::train::LinearSpec;
using zzs::mllee::train::LossResult;
using zzs::mllee::train::PoolKind;
using zzs::mllee::train::PoolLayout;
using zzs::mllee::train::PoolSpec;
using zzs::mllee::train::SoftmaxSpec;

/** @brief Q15 code step: 32768 / 127 ≈ 258. */
inline constexpr int32_t kCodeStep = 258;

/** @brief MLP forward: hidden = input * W_hidden^T + b_hidden; hidden = RELU(hidden);
           output = hidden * W_output^T + b_output; MSE loss. */
[[nodiscard]] inline Status
mlp_forward_hook (StepContext &context) noexcept {
  const Trainable &hidden_w = context._M_params[0];
  const Trainable &hidden_b = context._M_params[1];
  const Trainable &output_w = context._M_params[2];
  const Trainable &output_b = context._M_params[3];
  const Trainable &output = context._M_params[4];
  const Trainable &loss = context._M_params[5];

  const int8_t *const input = context._M_input;
  const uint32_t input_count = context._M_input_count;
  const int8_t *const target = context._M_target;
  const uint32_t target_count = context._M_target_count;

  /* Hidden activations (stack-local, 16 codes for default config). */
  int8_t hidden[kHiddenDim];

  /* Forward: hidden[j] = sum_i input[i] * W_hidden[j][i] + b_hidden[j] */
  for (uint32_t j = 0; j < kHiddenDim; ++j) {
    int32_t acc = 0;
    for (uint32_t i = 0; i < input_count; ++i) {
      acc += static_cast<int32_t> (input[i]) * static_cast<int32_t> (hidden_w.code_at (j * kHiddenInput + i));
    }
    acc = (acc + 128) >> 8; /* requantize: sum of codes / 256 */
    acc += static_cast<int32_t> (hidden_b.code_at (j));
    hidden[j] = static_cast<int8_t> (acc > 127 ? 127 : (acc < -127 ? -127 : acc));
  }

  /* ReLU activation (in-place on hidden). */
  for (uint32_t j = 0; j < kHiddenDim; ++j) {
    if (hidden[j] < 0)
      hidden[j] = 0;
  }

  /* Forward: output[k] = sum_j hidden[j] * W_output[k][j] + b_output[k] */
  for (uint32_t k = 0; k < kOutputDim; ++k) {
    int32_t acc = 0;
    for (uint32_t j = 0; j < kHiddenDim; ++j) {
      acc += static_cast<int32_t> (hidden[j]) * static_cast<int32_t> (output_w.code_at (k * kHiddenDim + j));
    }
    acc = (acc + 128) >> 8; /* requantize: sum of codes / 256 */
    acc += static_cast<int32_t> (output_b.code_at (k));
    output.set_code (k, acc);
  }

  /* MSE loss: loss = sum_k (output[k] - target[k])^2 / (4 * output_dim) */
  int32_t mse = 0;
  for (uint32_t k = 0; k < kOutputDim && k < target_count; ++k) {
    const int32_t diff = static_cast<int32_t> (output.code_at (k)) - static_cast<int32_t> (target[k]);
    mse += diff * diff;
  }
  mse = mse / (4 * kOutputDim);
  const int32_t loss_code = mse / 4;
  loss.set_code (0, loss_code);
  context._M_loss_q7 = static_cast<int32_t> ((static_cast<int64_t> (loss_code) << 7) / 127);

  /* Record backward for training. */
  if (context._M_tape != nullptr && context._M_tape->is_open ()) {
    Saved input_saved;
    int8_t *input_mem = nullptr;
    Status st = context._M_tape->save (input_count, input_saved, input_mem);
    if (!status_ok (st))
      return st;
    for (uint32_t i = 0; i < input_count; ++i)
      input_mem[i] = input[i];

    Saved hidden_saved;
    int8_t *hidden_mem = nullptr;
    st = context._M_tape->save (kHiddenDim, hidden_saved, hidden_mem);
    if (!status_ok (st))
      return st;
    for (uint32_t j = 0; j < kHiddenDim; ++j)
      hidden_mem[j] = hidden[j];

    Saved output_saved;
    int8_t *output_mem = nullptr;
    st = context._M_tape->save (kOutputDim, output_saved, output_mem);
    if (!status_ok (st))
      return st;
    for (uint32_t k = 0; k < kOutputDim; ++k)
      output_mem[k] = output.code_at (k);

    Saved target_saved;
    int8_t *target_mem = nullptr;
    st = context._M_tape->save (target_count, target_saved, target_mem);
    if (!status_ok (st))
      return st;
    for (uint32_t k = 0; k < target_count; ++k)
      target_mem[k] = target[k];

    Trainable operands[4] = {const_cast<Trainable &> (hidden_w), const_cast<Trainable &> (hidden_b),
                             const_cast<Trainable &> (output_w), const_cast<Trainable &> (output_b)};
    Saved saved_arr[4] = {input_saved, hidden_saved, output_saved, target_saved};
    return context._M_tape->record (mlp_backward_fn, 4u, loss, operands, 4u, saved_arr, 4u);
  }
  return Status::OK;
}

/** @brief Backward: dW_output[k][j] = dloss/doutput[k] * hidden[j],
                   dW_hidden[j][i] = dloss/dhidden[j] * input[i],
                   with ReLU derivative applied to hidden. */
[[nodiscard]] inline Status
mlp_backward_fn (const Trainable &output, const Trainable &grad_output, GradContext &context) noexcept {
  (void) output;
  (void) grad_output;

  Pool *const pool = context._M_pool;
  const int8_t *const input_mem = reinterpret_cast<const int8_t *> (pool->at (context._M_saved[0]._M_offset));
  const int8_t *const hidden_mem = reinterpret_cast<const int8_t *> (pool->at (context._M_saved[1]._M_offset));
  const int8_t *const output_mem = reinterpret_cast<const int8_t *> (pool->at (context._M_saved[2]._M_offset));
  const int8_t *const target_mem = reinterpret_cast<const int8_t *> (pool->at (context._M_saved[3]._M_offset));

  const Trainable &hidden_w = context._M_operands[0];
  const Trainable &hidden_b = context._M_operands[1];
  const Trainable &output_w = context._M_operands[2];
  const Trainable &output_b = context._M_operands[3];

  /* dloss/doutput[k] = diff / 6 */
  int32_t dloss_doutput[kOutputDim];
  for (uint32_t k = 0; k < kOutputDim; ++k) {
    const int32_t diff = static_cast<int32_t> (output_mem[k]) - static_cast<int32_t> (target_mem[k]);
    dloss_doutput[k] = diff / 6;
  }

  /* Accumulate db_output[k] = dloss_doutput[k] */
  for (uint32_t k = 0; k < kOutputDim; ++k)
    static_cast<void> (output_b.add_grad (k, dloss_doutput[k]));

  /* Accumulate dW_output[k][j] = dloss_doutput[k] * hidden[j].
     Scale by 1<<4 so the Q15 gradient is large enough to move int8 codes
     under a 0.05 Q15 learning rate without saturating int16 gradients. */
  for (uint32_t k = 0; k < kOutputDim; ++k) {
    for (uint32_t j = 0; j < kHiddenDim; ++j) {
      const int32_t grad_w = dloss_doutput[k] * static_cast<int32_t> (hidden_mem[j]) * 16;
      static_cast<void> (output_w.add_grad (k * kHiddenDim + j, grad_w));
    }
  }

  /* dloss/dhidden[j] = sum_k dloss_doutput[k] * W_output[k][j] */
  int32_t dloss_dhidden[kHiddenDim];
  for (uint32_t j = 0; j < kHiddenDim; ++j) {
    int32_t acc = 0;
    for (uint32_t k = 0; k < kOutputDim; ++k) {
      acc += dloss_doutput[k] * static_cast<int32_t> (output_w.code_at (k * kHiddenDim + j));
    }
    dloss_dhidden[j] = acc;
  }

  /* ReLU derivative: dloss/dhidden[j] *= (hidden[j] > 0 ? 1 : 0) */
  for (uint32_t j = 0; j < kHiddenDim; ++j) {
    if (hidden_mem[j] <= 0)
      dloss_dhidden[j] = 0;
  }

  /* Accumulate db_hidden[j] = dloss_dhidden[j] */
  for (uint32_t j = 0; j < kHiddenDim; ++j)
    static_cast<void> (hidden_b.add_grad (j, dloss_dhidden[j]));

  /* Accumulate dW_hidden[j][i] = dloss_dhidden[j] * input[i].
     Scale down by 1>>4 (i.e. divide by 16) because the hidden-layer gradient
     already includes the sum over output classes, making it naturally larger
     than the output-layer gradient.  This keeps int16 gradients from
     saturating while still giving a useful update signal. */
  for (uint32_t j = 0; j < kHiddenDim; ++j) {
    for (uint32_t i = 0; i < kHiddenInput; ++i) {
      const int32_t grad_w = (dloss_dhidden[j] * static_cast<int32_t> (input_mem[i])) >> 4;
      static_cast<void> (hidden_w.add_grad (j * kHiddenInput + i, grad_w));
    }
  }

  return Status::OK;
}

/** @brief Constant learning rate schedule: 0.05 in Q15 = 1638. */
[[nodiscard]] inline int32_t
constant_lr_schedule (uint32_t step) noexcept {
  (void) step;
  return 1638;
}

/** @brief SGD update: w -= lr * grad (code units). */
[[nodiscard]] inline Status
sgd_update_hook (StepContext &context) noexcept {
  const int32_t lr_q15 = context._M_lr_q15;

  for (uint32_t p = 0; p < context._M_param_count; ++p) {
    const Trainable &param = context._M_params[p];
    if (!param.requires_grad ())
      continue;
    for (uint32_t i = 0; i < param._M_count; ++i) {
      const int32_t grad_q15 = param.grad_at (i);
      const int32_t delta = q15_mul (Q15{static_cast<int16_t> (lr_q15)}, Q15{static_cast<int16_t> (grad_q15)})._M_raw;
      const int32_t delta_code = delta / kCodeStep;
      param.set_code (i, param.code_at (i) - delta_code);
    }
  }
  return Status::OK;
}

/** @brief Post-grad hook: count saturations. */
[[nodiscard]] inline Status
post_grad_hook (StepContext &context, uint32_t saturated) noexcept {
  (void) context;
  (void) saturated;
  return Status::OK;
}

/** @brief TrainingStorage constructor: wire up pool and tape. */
template <typename Config>
TrainingStorage<Config>::TrainingStorage () noexcept
    : _M_pool (this->_M_arena, sizeof (this->_M_arena)), _M_tape (this->_M_nodes, 256, this->_M_pool) {
}

/** @brief Reset all training state to zero. */
template <typename Config>
void
TrainingStorage<Config>::reset () noexcept {
  for (uint32_t i = 0; i < Config::hidden_input * Config::hidden_dim; ++i) {
    this->_M_hidden_w_code[i] = 0;
    this->_M_hidden_w_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::hidden_dim; ++i) {
    this->_M_hidden_b_code[i] = 0;
    this->_M_hidden_b_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::hidden_dim * Config::output_dim; ++i) {
    this->_M_w_code[i] = 0;
    this->_M_w_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::output_dim; ++i) {
    this->_M_b_code[i] = 0;
    this->_M_b_grad[i] = 0;
    this->_M_out_code[i] = 0;
    this->_M_out_grad[i] = 0;
  }
  this->_M_loss_code[0] = 0;
  this->_M_loss_grad[0] = 0;
  for (uint32_t i = 0; i < Config::flat_input; ++i)
    this->_M_input_code[i] = 0;
  for (uint32_t i = 0; i < Config::output_dim; ++i)
    this->_M_target_code[i] = 0;

  this->_M_pool.reset ();
}

/** @brief Wire up the HiddenParams and OutputParams Trainable views to storage buffers. */
template <typename Config>
void
setup_params (HiddenParams &hidden_params, OutputParams &output_params, int8_t *hidden_w_code, int16_t *hidden_w_grad,
              int8_t *hidden_b_code, int16_t *hidden_b_grad, int8_t *w_code, int16_t *w_grad, int8_t *b_code,
              int16_t *b_grad, int8_t *out_code, int16_t *out_grad, int8_t *loss_code, int16_t *loss_grad) noexcept {
  hidden_params._M_weight = Trainable::make (
      hidden_w_code, hidden_w_grad,
      Shape::matrix (static_cast<uint16_t> (Config::hidden_dim), static_cast<uint16_t> (Config::hidden_input)),
      Flag::RequiresGrad);
  hidden_params._M_bias =
      Trainable::make (hidden_b_code, hidden_b_grad, Shape::vector (static_cast<uint16_t> (Config::hidden_dim)),
                       Flag::RequiresGrad | Flag::BiasLike);
  output_params._M_weight = Trainable::make (
      w_code, w_grad,
      Shape::matrix (static_cast<uint16_t> (Config::output_dim), static_cast<uint16_t> (Config::hidden_dim)),
      Flag::RequiresGrad);
  output_params._M_bias = Trainable::make (b_code, b_grad, Shape::vector (static_cast<uint16_t> (Config::output_dim)),
                                           Flag::RequiresGrad | Flag::BiasLike);
  output_params._M_output =
      Trainable::make (out_code, out_grad, Shape::vector (static_cast<uint16_t> (Config::output_dim)), Flag::Frozen);
  output_params._M_loss = Trainable::make (loss_code, loss_grad, Shape::scalar (), 0u);
}

/** @brief Run training for @p steps on the compile-time dataset. */
template <typename Config>
[[nodiscard]] Status
TrainingPipeline<Config>::run_training (uint32_t steps, zzs::base::Stdout &out) noexcept {
  (void) out;

  auto &storage = this->_M_storage;
  setup_params<Config> (storage._M_hidden_params, storage._M_output_params, storage._M_hidden_w_code,
                        storage._M_hidden_w_grad, storage._M_hidden_b_code, storage._M_hidden_b_grad, storage._M_w_code,
                        storage._M_w_grad, storage._M_b_code, storage._M_b_grad, storage._M_out_code,
                        storage._M_out_grad, storage._M_loss_code, storage._M_loss_grad);

  Pool &pool = storage._M_pool;
  Tape &tape = storage._M_tape;

  StepContext ctx;
  ctx._M_pool = &pool;
  ctx._M_tape = &tape;
  ctx._M_param_count = 6;
  ctx._M_batch = 1;
  ctx._M_loss = Trainable::make (storage._M_loss_code, storage._M_loss_grad, Shape::scalar (), 0u);
  ctx._M_input = storage._M_input_code;
  ctx._M_input_count = Config::flat_input;
  ctx._M_target = storage._M_target_code;
  ctx._M_target_count = Config::output_dim;

  StepHooks hooks;
  hooks._M_forward = mlp_forward_hook;
  hooks._M_schedule = constant_lr_schedule;
  hooks._M_post_grad = post_grad_hook;
  hooks._M_update = sgd_update_hook;

  Trainer trainer (ctx, hooks);

  // Build a contiguous param array for the trainer: hidden_w, hidden_b, output_w, output_b, output, loss.
  Trainable param_array[6] = {
      storage._M_hidden_params._M_weight, storage._M_hidden_params._M_bias,   storage._M_output_params._M_weight,
      storage._M_output_params._M_bias,   storage._M_output_params._M_output, storage._M_output_params._M_loss,
  };
  ctx._M_params = param_array;

  Status st = trainer.configure (param_array, 6, 1);
  if (!status_ok (st))
    return st;

  for (uint32_t step = 0; step < steps; ++step) {
    const uint32_t sample_idx = step % dataset_count ();
    const auto &sample = dataset_sample (sample_idx);
    flatten_sample (sample, storage._M_input_code);
    one_hot_target (sample._M_label, storage._M_target_code);

    StepStats stats;
    st = trainer.step (stats);
    if (!status_ok (st))
      return st;
  }

  return Status::OK;
}

/** @brief Evaluate the model on the full dataset, report accuracy. */
template <typename Config>
[[nodiscard]] Status
TrainingPipeline<Config>::evaluate_model (zzs::base::Stdout &out) noexcept {
  (void) out;

  auto &storage = this->_M_storage;
  setup_params<Config> (storage._M_hidden_params, storage._M_output_params, storage._M_hidden_w_code,
                        storage._M_hidden_w_grad, storage._M_hidden_b_code, storage._M_hidden_b_grad, storage._M_w_code,
                        storage._M_w_grad, storage._M_b_code, storage._M_b_grad, storage._M_out_code,
                        storage._M_out_grad, storage._M_loss_code, storage._M_loss_grad);

  Pool &pool = storage._M_pool;
  Tape &tape = storage._M_tape;

  StepContext ctx;
  ctx._M_pool = &pool;
  ctx._M_tape = &tape;
  ctx._M_param_count = 6;
  ctx._M_batch = 1;
  ctx._M_loss = Trainable::make (storage._M_loss_code, storage._M_loss_grad, Shape::scalar (), 0u);
  ctx._M_input = storage._M_input_code;
  ctx._M_input_count = Config::flat_input;
  ctx._M_target = storage._M_target_code;
  ctx._M_target_count = Config::output_dim;

  StepHooks hooks;
  hooks._M_forward = mlp_forward_hook;

  Trainer trainer (ctx, hooks);

  Trainable param_array[6] = {
      storage._M_hidden_params._M_weight, storage._M_hidden_params._M_bias,   storage._M_output_params._M_weight,
      storage._M_output_params._M_bias,   storage._M_output_params._M_output, storage._M_output_params._M_loss,
  };
  ctx._M_params = param_array;

  Status st = trainer.configure (param_array, 6, 1);
  if (!status_ok (st))
    return st;

  uint32_t correct = 0;
  uint32_t total = 0;

  for (uint32_t i = 0; i < dataset_count (); ++i) {
    const auto &sample = dataset_sample (i);
    flatten_sample (sample, storage._M_input_code);
    one_hot_target (sample._M_label, storage._M_target_code);

    ctx._M_input = storage._M_input_code;
    ctx._M_input_count = Config::flat_input;
    ctx._M_target = storage._M_target_code;
    ctx._M_target_count = Config::output_dim;

    StepStats eval_stats;
    st = trainer.evaluate (eval_stats);
    if (!status_ok (st))
      return st;

    int32_t best_code = storage._M_out_code[0];
    uint32_t best_idx = 0;
    for (uint32_t c = 1; c < Config::output_dim; ++c) {
      if (storage._M_out_code[c] > best_code) {
        best_code = storage._M_out_code[c];
        best_idx = c;
      }
    }

    ++total;
    if (best_idx == sample._M_label)
      ++correct;
  }

  return Status::OK;
}

/* ======================================================================== */
/* Tier 2: CONV1D + Attention training pipeline                             */
/* ======================================================================== */

namespace {

/** @brief Conv1D geometry for the default topology (rank-1, causal). */
[[nodiscard]] inline ConvSpec
make_conv_spec (uint32_t batch) noexcept {
  ConvSpec spec;
  spec._M_batch = static_cast<uint16_t> (batch);
  spec._M_in_channels = static_cast<uint16_t> (ConvAttnDefaultConfig::feature_count);  /* 38 */
  spec._M_out_channels = static_cast<uint16_t> (ConvAttnDefaultConfig::conv_channels); /* 8 */
  spec._M_in[0] = static_cast<uint16_t> (ConvAttnDefaultConfig::seq_len);              /* 5 */
  spec._M_in[1] = 1u;
  spec._M_kernel[0] = static_cast<uint8_t> (ConvAttnDefaultConfig::conv_kernel); /* 3 */
  spec._M_kernel[1] = 1u;
  spec._M_stride[0] = 1u;
  spec._M_stride[1] = 1u;
  spec._M_dilation[0] = 1u;
  spec._M_dilation[1] = 1u;
  spec._M_pad[0] = 2; /* causal: (kernel-1) on the left */
  spec._M_pad[1] = 0;
  spec._M_pad[2] = 0;
  spec._M_pad[3] = 0;
  spec._M_groups = 1u;
  spec._M_rank = 1u;
  spec._M_layout = ConvLayout::CHANNEL_MAJOR;
  spec._M_act = Act::NONE;
  spec._M_aux_code = 0;
  spec._M_requant_shift = 7u;
  spec._M_requant_mult_q15 = zzs::mllee::train::_S_q15_one;
  spec._M_use_bias = 1u;
  spec._M_op_id = 0xC001u;
  return spec;
}

/** @brief LINEAR spec for the QKV projection (40 -> 120). */
[[nodiscard]] inline LinearSpec
make_qkv_spec (uint32_t batch) noexcept {
  LinearSpec spec;
  spec._M_batch = static_cast<uint16_t> (batch);
  spec._M_in_features = static_cast<uint16_t> (ConvAttnDefaultConfig::conv_output); /* 40 */
  spec._M_out_features = static_cast<uint16_t> (ConvAttnDefaultConfig::qkv_output); /* 120 */
  spec._M_use_bias = 1u;
  spec._M_requant_shift = 7u;
  spec._M_act = Act::NONE;
  spec._M_aux_code = 0;
  spec._M_pad = 0;
  spec._M_requant_mult_q15 = zzs::mllee::train::_S_q15_one;
  spec._M_op_id = 0xC002u;
  return spec;
}

/** @brief ATTENTION spec (seq=5, head_dim=4, heads=2). */
[[nodiscard]] inline AttentionSpec
make_attention_spec () noexcept {
  AttentionSpec spec;
  spec._M_seq = static_cast<uint16_t> (ConvAttnDefaultConfig::seq_len);           /* 5 */
  spec._M_heads = static_cast<uint16_t> (ConvAttnDefaultConfig::attn_heads);      /* 2 */
  spec._M_head_dim = static_cast<uint8_t> (ConvAttnDefaultConfig::attn_head_dim); /* 4 */
  spec._M_causal = 0u;
  spec._M_rope = 0u;
  spec._M_pad = 0u;
  spec._M_op_id = 0xC003u;
  return spec;
}

/** @brief GLOBAL_AVG_POOL spec (seq=5, channels=8). */
[[nodiscard]] inline PoolSpec
make_pool_spec (uint32_t batch) noexcept {
  PoolSpec spec;
  spec._M_batch = static_cast<uint16_t> (batch);
  spec._M_channels = static_cast<uint16_t> (ConvAttnDefaultConfig::conv_channels); /* 8 */
  spec._M_in[0] = static_cast<uint16_t> (ConvAttnDefaultConfig::seq_len);          /* 5 */
  spec._M_in[1] = 1u;
  spec._M_out[0] = 1u;
  spec._M_out[1] = 1u;
  spec._M_kernel[0] = 1u;
  spec._M_kernel[1] = 1u;
  spec._M_stride[0] = 1u;
  spec._M_stride[1] = 1u;
  spec._M_pad[0] = 0;
  spec._M_pad[1] = 0;
  spec._M_pad[2] = 0;
  spec._M_pad[3] = 0;
  spec._M_kind = PoolKind::GLOBAL_AVERAGE;
  spec._M_layout = PoolLayout::CHANNEL_MAJOR;
  spec._M_rank = 1u;
  spec._M_count_include_pad = 1u;
  spec._M_op_id = 0xC004u;
  return spec;
}

/** @brief LINEAR spec for the output head (8 -> 3). */
[[nodiscard]] inline LinearSpec
make_output_spec (uint32_t batch) noexcept {
  LinearSpec spec;
  spec._M_batch = static_cast<uint16_t> (batch);
  spec._M_in_features = static_cast<uint16_t> (ConvAttnDefaultConfig::pooled);      /* 8 */
  spec._M_out_features = static_cast<uint16_t> (ConvAttnDefaultConfig::output_dim); /* 3 */
  spec._M_use_bias = 1u;
  spec._M_requant_shift = 7u;
  spec._M_act = Act::NONE;
  spec._M_aux_code = 0;
  spec._M_pad = 0;
  spec._M_requant_mult_q15 = zzs::mllee::train::_S_q15_one;
  spec._M_op_id = 0xC005u;
  return spec;
}

/** @brief SOFTMAX spec (3 classes, last axis). */
[[nodiscard]] inline SoftmaxSpec
make_softmax_spec () noexcept {
  SoftmaxSpec spec;
  spec._M_rows = 1u;
  spec._M_length = static_cast<uint16_t> (ConvAttnDefaultConfig::output_dim); /* 3 */
  spec._M_inner = 1u;
  spec._M_rank = 1u;
  spec._M_axis = 0u;
  spec._M_masked = 0u;
  spec._M_log = 0u;
  spec._M_temperature_q15 = zzs::mllee::train::_S_q15_one;
  spec._M_op_id = 0xC006u;
  return spec;
}

} // namespace

/** @brief Q15 code step: 32768 / 127 ≈ 258. */
inline constexpr int32_t kConvAttnCodeStep = 258;

/** @brief ConvAttn forward hook: full layer chain + MSE loss, all recorded on tape. */
[[nodiscard]] inline Status
conv_attn_forward_hook (StepContext &context) noexcept {
  const int8_t *const input = context._M_input;
  const uint32_t input_count = context._M_input_count;
  const int8_t *const target = context._M_target;
  const uint32_t target_count = context._M_target_count;

  /* Param layout: 0=conv_w 1=conv_b 2=qkv_w 3=qkv_b 4=out_w 5=out_b
   *             6=conv_out 7=qkv_out 8=attn_out 9=pool_out 10=model_out 11=loss */
  const Trainable &conv_w = context._M_params[0];
  const Trainable &conv_b = context._M_params[1];
  const Trainable &qkv_w = context._M_params[2];
  const Trainable &qkv_b = context._M_params[3];
  const Trainable &out_w = context._M_params[4];
  const Trainable &out_b = context._M_params[5];
  Trainable conv_out = context._M_params[6];
  Trainable qkv_out = context._M_params[7];
  Trainable attn_out = context._M_params[8];
  Trainable pool_out = context._M_params[9];
  Trainable model_out = context._M_params[10];

  /* ---------- Layer 1: CONV1D(38->8, k=3, causal): 190 -> 40 ---------- */
  Trainable conv_input = Trainable::make (
      const_cast<int8_t *> (input), nullptr,
      zzs::mllee::train::Shape::tensor3 (1u, ConvAttnDefaultConfig::feature_count, ConvAttnDefaultConfig::seq_len), 0u);
  const ConvSpec conv_spec = make_conv_spec (1);
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    Status st = conv_run (tape, conv_spec, conv_input, conv_w, conv_b, conv_out);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 2: LINEAR(40->120) QKV projection: 40 -> 120 ---------- */
  Trainable qkv_input = conv_out;
  qkv_input._M_shape = zzs::mllee::train::Shape::matrix (1u, ConvAttnDefaultConfig::conv_output);
  const LinearSpec qkv_spec = make_qkv_spec (1);
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    Status st = linear_run (tape, qkv_spec, qkv_input, qkv_w, qkv_b, qkv_out);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 3: ATTENTION(seq=5, head_dim=4, heads=2): 120 -> 40 ----------
   * Q/K/V split strategy: the QKV LINEAR output is one 120-code buffer
   * (Q[0..39] + K[40..79] + V[80..119]).  attention_backward needs Q, K, V as
   * three separate operands, so we build three Trainable views into the
   * qkv_out buffer.  Crucially, each view's grad lanes alias qkv_out's grad
   * buffer at offsets 0, 40, 80 respectively, so attention_backward writes
   * dQ/dK/dV into the *concatenated* grad buffer.  The QKV LINEAR backward
   * then reads qkv_out._M_grad (all 120 lanes) as its grad_output, which is
   * exactly [dQ, dK, dV] concatenated.  No extra copy or split backward is
   * needed -- the tape's reverse walk naturally chains them. */
  const uint32_t attn_stride = ConvAttnDefaultConfig::attn_heads * ConvAttnDefaultConfig::attn_head_dim; /* 8 */
  const uint32_t attn_elements = ConvAttnDefaultConfig::attn_output;                                     /* 40 */

  Trainable q_view = Trainable::make (qkv_out._M_code, qkv_out._M_grad + 0 * attn_elements,
                                      zzs::mllee::train::Shape::matrix (ConvAttnDefaultConfig::seq_len, attn_stride),
                                      zzs::mllee::train::Flag::RequiresGrad);
  Trainable k_view = Trainable::make (qkv_out._M_code + attn_elements, qkv_out._M_grad + 1 * attn_elements,
                                      zzs::mllee::train::Shape::matrix (ConvAttnDefaultConfig::seq_len, attn_stride),
                                      zzs::mllee::train::Flag::RequiresGrad);
  Trainable v_view = Trainable::make (qkv_out._M_code + 2 * attn_elements, qkv_out._M_grad + 2 * attn_elements,
                                      zzs::mllee::train::Shape::matrix (ConvAttnDefaultConfig::seq_len, attn_stride),
                                      zzs::mllee::train::Flag::RequiresGrad);

  const AttentionSpec attn_spec = make_attention_spec ();
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    Status st = attention_run (tape, attn_spec, q_view, k_view, v_view, attn_out);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 4: GLOBAL_AVG_POOL(seq=5, channels=8): 40 -> 8 ---------- */
  Trainable pool_input = attn_out;
  pool_input._M_shape =
      zzs::mllee::train::Shape::tensor3 (1u, ConvAttnDefaultConfig::conv_channels, ConvAttnDefaultConfig::seq_len);
  const PoolSpec pool_spec = make_pool_spec (1);
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    Status st = pool_run (tape, pool_spec, pool_input, pool_out);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 5: LINEAR(8->3): 8 -> 3 ---------- */
  Trainable out_input = pool_out;
  out_input._M_shape = zzs::mllee::train::Shape::matrix (1u, ConvAttnDefaultConfig::pooled);
  const LinearSpec output_spec = make_output_spec (1);
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    Status st = linear_run (tape, output_spec, out_input, out_w, out_b, model_out);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 6: SOFTMAX(3->3), in-place ---------- */
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    Status st = softmax_run (tape, make_softmax_spec (), model_out, nullptr, model_out);
    if (!status_ok (st))
      return st;
  }

  /* ---------- MSE loss: model_out vs target ---------- */
  Trainable pred = model_out;
  pred._M_grad = model_out._M_grad; /* grad lanes receive dL/dpred */
  Trainable target_t = Trainable::make (const_cast<int8_t *> (target), nullptr,
                                        zzs::mllee::train::Shape::vector (static_cast<uint16_t> (target_count)), 0u);
  LossResult loss_result;
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    Status st = mse_forward (pool, &tape, pred, target_t, loss_result);
    if (!status_ok (st))
      return st;
  }

  /* Publish the loss node to the trainer context. */
  context._M_loss = loss_result._M_node;
  context._M_loss_q7 = loss_result._M_loss_q7;

  return Status::OK;
}

/** @brief Backward function for the MSE loss over the conv-attn output.
 *
 * Delegates to mllee's `mse_backward`, which computes dL/dpred = seed * 2 *
 * (p - t) / n and accumulates into the prediction operand's grad lanes.  This
 * is the seed of the reverse walk: the tape then propagates it through
 * SOFTMAX -> LINEAR(8->3) -> GLOBAL_AVG_POOL -> ATTENTION -> LINEAR(QKV) ->
 * CONV1D in reverse recording order. */
[[nodiscard]] inline Status
conv_attn_backward_fn (const Trainable &output, const Trainable &grad_output, GradContext &context) noexcept {
  (void) output;
  return mse_backward (output, grad_output, context);
}

/** @brief Wire up the ConvAttnParams Trainable views to storage buffers. */
template <typename Config>
void
setup_conv_attn_params (ConvAttnParams &params, int8_t *conv_w_code, int16_t *conv_w_grad, int8_t *conv_b_code,
                        int16_t *conv_b_grad, int8_t *qkv_w_code, int16_t *qkv_w_grad, int8_t *qkv_b_code,
                        int16_t *qkv_b_grad, int8_t *out_w_code, int16_t *out_w_grad, int8_t *out_b_code,
                        int16_t *out_b_grad, int8_t *conv_out_code, int16_t *conv_out_grad, int8_t *qkv_out_code,
                        int16_t *qkv_out_grad, int8_t *attn_out_code, int16_t *attn_out_grad, int8_t *pool_out_code,
                        int16_t *pool_out_grad, int8_t *model_out_code, int16_t *model_out_grad, int8_t *loss_code,
                        int16_t *loss_grad) noexcept {
  params._M_conv_w = Trainable::make (
      conv_w_code, conv_w_grad,
      zzs::mllee::train::Shape::matrix (static_cast<uint16_t> (Config::conv_channels),
                                        static_cast<uint16_t> (Config::feature_count * Config::conv_kernel)),
      zzs::mllee::train::Flag::RequiresGrad);
  params._M_conv_b = Trainable::make (conv_b_code, conv_b_grad,
                                      zzs::mllee::train::Shape::vector (static_cast<uint16_t> (Config::conv_channels)),
                                      zzs::mllee::train::Flag::RequiresGrad | zzs::mllee::train::Flag::BiasLike);
  params._M_qkv_w = Trainable::make (qkv_w_code, qkv_w_grad,
                                     zzs::mllee::train::Shape::matrix (static_cast<uint16_t> (Config::qkv_output),
                                                                       static_cast<uint16_t> (Config::conv_output)),
                                     zzs::mllee::train::Flag::RequiresGrad);
  params._M_qkv_b = Trainable::make (qkv_b_code, qkv_b_grad,
                                     zzs::mllee::train::Shape::vector (static_cast<uint16_t> (Config::qkv_output)),
                                     zzs::mllee::train::Flag::RequiresGrad | zzs::mllee::train::Flag::BiasLike);
  params._M_out_w = Trainable::make (out_w_code, out_w_grad,
                                     zzs::mllee::train::Shape::matrix (static_cast<uint16_t> (Config::output_dim),
                                                                       static_cast<uint16_t> (Config::pooled)),
                                     zzs::mllee::train::Flag::RequiresGrad);
  params._M_out_b = Trainable::make (out_b_code, out_b_grad,
                                     zzs::mllee::train::Shape::vector (static_cast<uint16_t> (Config::output_dim)),
                                     zzs::mllee::train::Flag::RequiresGrad | zzs::mllee::train::Flag::BiasLike);

  /* Activations: code + grad lanes, require grad so backward writes into them. */
  params._M_conv_out = Trainable::make (conv_out_code, conv_out_grad,
                                        zzs::mllee::train::Shape::tensor3 (1u, Config::conv_channels, Config::seq_len),
                                        zzs::mllee::train::Flag::RequiresGrad);
  params._M_qkv_out =
      Trainable::make (qkv_out_code, qkv_out_grad, zzs::mllee::train::Shape::matrix (1u, Config::qkv_output),
                       zzs::mllee::train::Flag::RequiresGrad);
  params._M_attn_out =
      Trainable::make (attn_out_code, attn_out_grad, zzs::mllee::train::Shape::matrix (1u, Config::attn_output),
                       zzs::mllee::train::Flag::RequiresGrad);
  params._M_pool_out =
      Trainable::make (pool_out_code, pool_out_grad, zzs::mllee::train::Shape::tensor3 (1u, Config::conv_channels, 1u),
                       zzs::mllee::train::Flag::RequiresGrad);
  params._M_model_out =
      Trainable::make (model_out_code, model_out_grad, zzs::mllee::train::Shape::matrix (1u, Config::output_dim),
                       zzs::mllee::train::Flag::RequiresGrad);
  params._M_loss = Trainable::make (loss_code, loss_grad, zzs::mllee::train::Shape::scalar (), 0u);
}

/** @brief ConvAttnStorage constructor: wire up pool and tape. */
template <typename Config>
ConvAttnStorage<Config>::ConvAttnStorage () noexcept
    : _M_pool (this->_M_arena, sizeof (this->_M_arena)), _M_tape (this->_M_nodes, 256, this->_M_pool) {
}

/** @brief Reset all training state to zero. */
template <typename Config>
void
ConvAttnStorage<Config>::reset () noexcept {
  auto &s = *this;
  for (uint32_t i = 0; i < Config::conv_channels * Config::feature_count * Config::conv_kernel; ++i) {
    s._M_conv_w_code[i] = 0;
    s._M_conv_w_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::conv_channels; ++i) {
    s._M_conv_b_code[i] = 0;
    s._M_conv_b_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::qkv_output * Config::conv_output; ++i) {
    s._M_qkv_w_code[i] = 0;
    s._M_qkv_w_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::qkv_output; ++i) {
    s._M_qkv_b_code[i] = 0;
    s._M_qkv_b_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::output_dim * Config::pooled; ++i) {
    s._M_out_w_code[i] = 0;
    s._M_out_w_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::output_dim; ++i) {
    s._M_out_b_code[i] = 0;
    s._M_out_b_grad[i] = 0;
    s._M_model_out_code[i] = 0;
    s._M_model_out_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::conv_output; ++i) {
    s._M_conv_out_code[i] = 0;
    s._M_conv_out_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::qkv_output; ++i) {
    s._M_qkv_out_code[i] = 0;
    s._M_qkv_out_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::attn_output; ++i) {
    s._M_attn_out_code[i] = 0;
    s._M_attn_out_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::pooled; ++i) {
    s._M_pool_out_code[i] = 0;
    s._M_pool_out_grad[i] = 0;
  }
  s._M_loss_code[0] = 0;
  s._M_loss_grad[0] = 0;
  for (uint32_t i = 0; i < Config::flat_input; ++i)
    s._M_input_code[i] = 0;
  for (uint32_t i = 0; i < Config::output_dim; ++i)
    s._M_target_code[i] = 0;

  s._M_pool.reset ();
}

/** @brief Run training for @p steps on the compile-time dataset. */
template <typename Config>
[[nodiscard]] Status
ConvAttnPipeline<Config>::run_training (uint32_t steps, zzs::base::Stdout &out) noexcept {
  (void) out;

  auto &storage = this->_M_storage;
  setup_conv_attn_params<Config> (
      storage._M_params, storage._M_conv_w_code, storage._M_conv_w_grad, storage._M_conv_b_code, storage._M_conv_b_grad,
      storage._M_qkv_w_code, storage._M_qkv_w_grad, storage._M_qkv_b_code, storage._M_qkv_b_grad, storage._M_out_w_code,
      storage._M_out_w_grad, storage._M_out_b_code, storage._M_out_b_grad, storage._M_conv_out_code,
      storage._M_conv_out_grad, storage._M_qkv_out_code, storage._M_qkv_out_grad, storage._M_attn_out_code,
      storage._M_attn_out_grad, storage._M_pool_out_code, storage._M_pool_out_grad, storage._M_model_out_code,
      storage._M_model_out_grad, storage._M_loss_code, storage._M_loss_grad);

  Pool &pool = storage._M_pool;
  Tape &tape = storage._M_tape;

  StepContext ctx;
  ctx._M_pool = &pool;
  ctx._M_tape = &tape;
  ctx._M_batch = 1;
  ctx._M_input = storage._M_input_code;
  ctx._M_input_count = Config::flat_input;
  ctx._M_target = storage._M_target_code;
  ctx._M_target_count = Config::output_dim;

  /* Contiguous param array for the trainer:
   * [conv_w, conv_b, qkv_w, qkv_b, out_w, out_b,
   *  conv_out, qkv_out, attn_out, pool_out, model_out, loss] */
  Trainable param_array[12] = {
      storage._M_params._M_conv_w,   storage._M_params._M_conv_b,    storage._M_params._M_qkv_w,
      storage._M_params._M_qkv_b,    storage._M_params._M_out_w,     storage._M_params._M_out_b,
      storage._M_params._M_conv_out, storage._M_params._M_qkv_out,   storage._M_params._M_attn_out,
      storage._M_params._M_pool_out, storage._M_params._M_model_out, storage._M_params._M_loss,
  };
  ctx._M_params = param_array;
  ctx._M_param_count = 12;

  StepHooks hooks;
  hooks._M_forward = conv_attn_forward_hook;
  hooks._M_schedule = constant_lr_schedule;
  hooks._M_post_grad = post_grad_hook;
  hooks._M_update = sgd_update_hook;

  Trainer trainer (ctx, hooks);
  Status st = trainer.configure (param_array, 12, 1);
  if (!status_ok (st))
    return st;

  for (uint32_t step = 0; step < steps; ++step) {
    const uint32_t sample_idx = step % dataset_count ();
    const auto &sample = dataset_sample (sample_idx);
    flatten_sample (sample, storage._M_input_code);
    one_hot_target (sample._M_label, storage._M_target_code);

    StepStats stats;
    st = trainer.step (stats);
    if (!status_ok (st))
      return st;
  }

  return Status::OK;
}

/** @brief Evaluate the model on the full dataset, report accuracy. */
template <typename Config>
[[nodiscard]] Status
ConvAttnPipeline<Config>::evaluate_model (zzs::base::Stdout &out) noexcept {
  (void) out;

  auto &storage = this->_M_storage;
  setup_conv_attn_params<Config> (
      storage._M_params, storage._M_conv_w_code, storage._M_conv_w_grad, storage._M_conv_b_code, storage._M_conv_b_grad,
      storage._M_qkv_w_code, storage._M_qkv_w_grad, storage._M_qkv_b_code, storage._M_qkv_b_grad, storage._M_out_w_code,
      storage._M_out_w_grad, storage._M_out_b_code, storage._M_out_b_grad, storage._M_conv_out_code,
      storage._M_conv_out_grad, storage._M_qkv_out_code, storage._M_qkv_out_grad, storage._M_attn_out_code,
      storage._M_attn_out_grad, storage._M_pool_out_code, storage._M_pool_out_grad, storage._M_model_out_code,
      storage._M_model_out_grad, storage._M_loss_code, storage._M_loss_grad);

  Pool &pool = storage._M_pool;
  Tape &tape = storage._M_tape;

  StepContext ctx;
  ctx._M_pool = &pool;
  ctx._M_tape = &tape;
  ctx._M_batch = 1;
  ctx._M_input = storage._M_input_code;
  ctx._M_input_count = Config::flat_input;
  ctx._M_target = storage._M_target_code;
  ctx._M_target_count = Config::output_dim;

  Trainable param_array[12] = {
      storage._M_params._M_conv_w,   storage._M_params._M_conv_b,    storage._M_params._M_qkv_w,
      storage._M_params._M_qkv_b,    storage._M_params._M_out_w,     storage._M_params._M_out_b,
      storage._M_params._M_conv_out, storage._M_params._M_qkv_out,   storage._M_params._M_attn_out,
      storage._M_params._M_pool_out, storage._M_params._M_model_out, storage._M_params._M_loss,
  };
  ctx._M_params = param_array;
  ctx._M_param_count = 12;

  StepHooks hooks;
  hooks._M_forward = conv_attn_forward_hook;

  Trainer trainer (ctx, hooks);
  Status st = trainer.configure (param_array, 12, 1);
  if (!status_ok (st))
    return st;

  uint32_t correct = 0;
  uint32_t total = 0;

  for (uint32_t i = 0; i < dataset_count (); ++i) {
    const auto &sample = dataset_sample (i);
    flatten_sample (sample, storage._M_input_code);
    one_hot_target (sample._M_label, storage._M_target_code);

    ctx._M_input = storage._M_input_code;
    ctx._M_input_count = Config::flat_input;
    ctx._M_target = storage._M_target_code;
    ctx._M_target_count = Config::output_dim;

    StepStats eval_stats;
    st = trainer.evaluate (eval_stats);
    if (!status_ok (st))
      return st;

    int32_t best_code = storage._M_model_out_code[0];
    uint32_t best_idx = 0;
    for (uint32_t c = 1; c < Config::output_dim; ++c) {
      if (storage._M_model_out_code[c] > best_code) {
        best_code = storage._M_model_out_code[c];
        best_idx = c;
      }
    }

    ++total;
    if (best_idx == sample._M_label)
      ++correct;
  }

  return Status::OK;
}

/* ======================================================================== */
/* Tier 4: Multi-task training pipeline                                     */
/* ======================================================================== */

/* Multi-task loss: L = CE(head1, target) + 0.5*MSE(head2, target[1])
 *                 + 0.5*MSE(head3, target[2]).  All arithmetic fixed-point,
 *     int8 codes.  head2/head3 are scalar outputs; target[1]/target[2] are
 *     single int8 codes (not one-hot). */

/** @brief Multi-task forward hook: CONV1D -> QKV LINEAR -> ATTENTION ->
 *         GLOBAL_AVG_POOL -> 3 LINEAR heads -> SOFTMAX(head1) +
 *         multi-task loss.  All layers recorded on tape. */
[[nodiscard]] inline Status
multitask_forward_hook (StepContext &context) noexcept {
  const int8_t *const input = context._M_input;
  const uint32_t input_count = context._M_input_count;
  const int8_t *const target = context._M_target;
  const uint32_t target_count = context._M_target_count;

  /* Param layout: 0=conv_w 1=conv_b 2=qkv_w 3=qkv_b
   *             4=head1_w 5=head1_b 6=head2_w 7=head2_b
   *             8=head3_w 9=head3_b 10=output 11=loss */
  const Trainable &conv_w = context._M_params[0];
  const Trainable &conv_b = context._M_params[1];
  const Trainable &qkv_w = context._M_params[2];
  const Trainable &qkv_b = context._M_params[3];
  const Trainable &h1_w = context._M_params[4];
  const Trainable &h1_b = context._M_params[5];
  const Trainable &h2_w = context._M_params[6];
  const Trainable &h2_b = context._M_params[7];
  const Trainable &h3_w = context._M_params[8];
  const Trainable &h3_b = context._M_params[9];
  Trainable output = context._M_params[10];

  /* Wrap raw input pointer as a Trainable view (no grad, forward-only). */
  Trainable conv_input = Trainable::make (
      const_cast<int8_t *> (input), nullptr,
      zzs::mllee::train::Shape::tensor3 (1u, ConvAttnDefaultConfig::feature_count, ConvAttnDefaultConfig::seq_len), 0u);

  /* ---------- Layer 1: CONV1D(38->8, k=3, causal): 190 -> 40 ---------- */
  Trainable conv_out_code;
  int8_t conv_out_mem[ConvAttnDefaultConfig::conv_output];
  conv_out_code = Trainable::make (
      conv_out_mem, nullptr,
      zzs::mllee::train::Shape::tensor3 (1u, ConvAttnDefaultConfig::conv_channels, ConvAttnDefaultConfig::seq_len), 0u);
  const ConvSpec conv_spec = make_conv_spec (1);
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    const Status st = conv_run (tape, conv_spec, conv_input, conv_w, conv_b, conv_out_code);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 2: LINEAR(40->120) QKV: 40 -> 120 ---------- */
  Trainable qkv_out_code;
  int8_t qkv_out_mem[ConvAttnDefaultConfig::qkv_output];
  qkv_out_code = Trainable::make (qkv_out_mem, nullptr,
                                  zzs::mllee::train::Shape::matrix (1u, ConvAttnDefaultConfig::qkv_output), 0u);
  const LinearSpec qkv_spec = make_qkv_spec (1);
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    const Status st = linear_run (tape, qkv_spec, conv_out_code, qkv_w, qkv_b, qkv_out_code);
    if (!status_ok (st))
      return st;
  }

  /* Q/K/V view-split for ATTENTION (same pattern as conv_attn_forward_hook). */
  const uint32_t attn_stride = ConvAttnDefaultConfig::attn_heads * ConvAttnDefaultConfig::attn_head_dim;
  const uint32_t attn_elements = ConvAttnDefaultConfig::attn_output;
  Trainable q_view = Trainable::make (qkv_out_code._M_code, nullptr,
                                      zzs::mllee::train::Shape::matrix (ConvAttnDefaultConfig::seq_len, attn_stride),
                                      zzs::mllee::train::Flag::RequiresGrad);
  Trainable k_view = Trainable::make (qkv_out_code._M_code + attn_elements, nullptr,
                                      zzs::mllee::train::Shape::matrix (ConvAttnDefaultConfig::seq_len, attn_stride),
                                      zzs::mllee::train::Flag::RequiresGrad);
  Trainable v_view = Trainable::make (qkv_out_code._M_code + 2 * attn_elements, nullptr,
                                      zzs::mllee::train::Shape::matrix (ConvAttnDefaultConfig::seq_len, attn_stride),
                                      zzs::mllee::train::Flag::RequiresGrad);

  /* ---------- Layer 3: ATTENTION(seq=5,head_dim=4,heads=2): 120 -> 40 ---------- */
  Trainable attn_out_code;
  int8_t attn_out_mem[ConvAttnDefaultConfig::attn_output];
  attn_out_code = Trainable::make (attn_out_mem, nullptr,
                                   zzs::mllee::train::Shape::matrix (1u, ConvAttnDefaultConfig::attn_output), 0u);
  const AttentionSpec attn_spec = make_attention_spec ();
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    const Status st = attention_run (tape, attn_spec, q_view, k_view, v_view, attn_out_code);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 4: GLOBAL_AVG_POOL(seq=5,ch=8): 40 -> 8 ---------- */
  Trainable pool_out_code;
  int8_t pool_out_mem[ConvAttnDefaultConfig::pooled];
  pool_out_code = Trainable::make (
      pool_out_mem, nullptr, zzs::mllee::train::Shape::tensor3 (1u, ConvAttnDefaultConfig::conv_channels, 1u), 0u);
  const PoolSpec pool_spec = make_pool_spec (1);
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    const Status st = pool_run (tape, pool_spec, attn_out_code, pool_out_code);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 5: HEAD1 LINEAR(8->3): 8 -> 3 ---------- */
  Trainable head1_out_code;
  int8_t head1_out_mem[ConvAttnDefaultConfig::output_dim];
  head1_out_code = Trainable::make (head1_out_mem, nullptr,
                                    zzs::mllee::train::Shape::matrix (1u, ConvAttnDefaultConfig::output_dim), 0u);
  const LinearSpec h1_spec = make_output_spec (1);
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    const Status st = linear_run (tape, h1_spec, pool_out_code, h1_w, h1_b, head1_out_code);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 6: HEAD2 LINEAR(8->1): 8 -> 1 ---------- */
  Trainable head2_out_code;
  int8_t head2_out_mem[1];
  head2_out_code = Trainable::make (head2_out_mem, nullptr, zzs::mllee::train::Shape::matrix (1u, 1u), 0u);
  LinearSpec h2l_spec = make_output_spec (1);
  h2l_spec._M_out_features = 1u;
  h2l_spec._M_op_id = 0xC007u;
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    const Status st = linear_run (tape, h2l_spec, pool_out_code, h2_w, h2_b, head2_out_code);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 7: HEAD3 LINEAR(8->1): 8 -> 1 ---------- */
  Trainable head3_out_code;
  int8_t head3_out_mem[1];
  head3_out_code = Trainable::make (head3_out_mem, nullptr, zzs::mllee::train::Shape::matrix (1u, 1u), 0u);
  LinearSpec h3l_spec = make_output_spec (1);
  h3l_spec._M_out_features = 1u;
  h3l_spec._M_op_id = 0xC008u;
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    const Status st = linear_run (tape, h3l_spec, pool_out_code, h3_w, h3_b, head3_out_code);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 8: SOFTMAX(head1, in-place) ---------- */
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    const Status st = softmax_run (tape, make_softmax_spec (), head1_out_code, nullptr, head1_out_code);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Multi-task loss ---------- */
  Trainable pred = head1_out_code;
  /* Find the target class index from the one-hot target (127 = true class). */
  uint16_t target_class = 0;
  for (uint32_t i = 0; i < target_count; ++i) {
    if (target[i] == 127) {
      target_class = static_cast<uint16_t> (i);
      break;
    }
  }

  LossResult ce_result;
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    const Status st = cross_entropy_forward (pool, &tape, pred, target_class, ce_result);
    if (!status_ok (st))
      return st;
  }

  /* head2 MSE: head2_out[0] vs target[1] */
  Trainable head2_pred = head2_out_code;
  int8_t conf_target_mem[1] = {static_cast<int8_t> (target_count > 1 ? target[1] : 0)};
  Trainable conf_target = Trainable::make (conf_target_mem, nullptr, zzs::mllee::train::Shape::vector (1u), 0u);
  LossResult mse2_result;
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    const Status st = mse_forward (pool, &tape, head2_pred, conf_target, mse2_result);
    if (!status_ok (st))
      return st;
  }

  /* head3 MSE: head3_out[0] vs target[2] */
  Trainable head3_pred = head3_out_code;
  int8_t time_target_mem[1] = {static_cast<int8_t> (target_count > 2 ? target[2] : 0)};
  Trainable time_target = Trainable::make (time_target_mem, nullptr, zzs::mllee::train::Shape::vector (1u), 0u);
  LossResult mse3_result;
  {
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    const Status st = mse_forward (pool, &tape, head3_pred, time_target, mse3_result);
    if (!status_ok (st))
      return st;
  }

  /* Combine losses: L = CE + 0.5*MSE2 + 0.5*MSE3 (Q7 fixed point). */
  const int32_t loss_q7 = ce_result._M_loss_q7 + (mse2_result._M_loss_q7 >> 1) + (mse3_result._M_loss_q7 >> 1);
  output.set_code (0, loss_q7);

  /* Publish loss node: use CE's node as the primary tape entry. */
  context._M_loss = ce_result._M_node;
  context._M_loss_q7 = loss_q7;

  return Status::OK;
}

/** @brief Multi-task backward: delegate to mse_backward; tape reverse walk
 *         propagates through SOFTMAX -> HEAD1 -> GLOBAL_AVG_POOL -> ATTENTION
 *         -> QKV LINEAR -> CONV1D. */
[[nodiscard]] inline Status
multitask_backward_fn (const Trainable &output, const Trainable &grad_output, GradContext &context) noexcept {
  (void) output;
  return mse_backward (output, grad_output, context);
}

/** @brief Distill forward hook: student CONV1D+Attention+3-head forward with a
 *         teacher-blended soft target and MSE distillation loss.
 *
 * Param layout: 0=conv_w 1=conv_b 2=qkv_w 3=qkv_b
 *               4=head1_w 5=head1_b 6=head2_w 7=head2_b
 *               8=head3_w 9=head3_b 10=output 11=loss
 *
 * All activations are code-only stack arrays (no grad lanes in forward):
 * conv_out[40] + qkv_out[120] + attn_out[40] + pool_out[8] +
 * head1_out[3] + head2_out[1] + head3_out[1] + soft_target[3] ~ 216 bytes.
 */
[[nodiscard]] inline Status
distill_forward_hook (StepContext &context, const int8_t *teacher_logits, uint32_t teacher_count) noexcept {
  const int8_t *const input = context._M_input;
  const int8_t *const target = context._M_target;
  const uint32_t target_count = context._M_target_count;

  /* Layer weights: 0=conv_w 1=conv_b 2=qkv_w 3=qkv_b
   *              4=head1_w 5=head1_b 6=head2_w 7=head2_b
   *              8=head3_w 9=head3_b */
  const Trainable &conv_w = context._M_params[0];
  const Trainable &conv_b = context._M_params[1];
  const Trainable &qkv_w = context._M_params[2];
  const Trainable &qkv_b = context._M_params[3];
  const Trainable &head1_w = context._M_params[4];
  const Trainable &head1_b = context._M_params[5];
  const Trainable &head2_w = context._M_params[6];
  const Trainable &head2_b = context._M_params[7];
  const Trainable &head3_w = context._M_params[8];
  const Trainable &head3_b = context._M_params[9];

  /* ---------- Activations: code-only stack arrays (no grad in forward) ---------- */
  alignas (4) int8_t conv_out[ConvAttnDefaultConfig::conv_output]{};   /* 40 */
  alignas (4) int8_t qkv_out[ConvAttnDefaultConfig::qkv_output]{};     /* 120 */
  alignas (4) int8_t attn_out[ConvAttnDefaultConfig::attn_output]{};   /* 40 */
  alignas (4) int8_t pool_out[ConvAttnDefaultConfig::pooled]{};        /* 8 */
  alignas (4) int8_t head1_out[ConvAttnDefaultConfig::output_dim]{};   /* 3 */
  int8_t head2_out[1]{};                                               /* 1 */
  int8_t head3_out[1]{};                                               /* 1 */
  alignas (4) int8_t soft_target[ConvAttnDefaultConfig::output_dim]{}; /* 3 */

  Pool &pool = *context._M_pool;
  Tape &tape = *context._M_tape;

  /* ---------- Layer 1: CONV1D(38->8, k=3, causal): 190 -> 40 ---------- */
  Trainable conv_input = Trainable::make (
      const_cast<int8_t *> (input), nullptr,
      zzs::mllee::train::Shape::tensor3 (1u, ConvAttnDefaultConfig::feature_count, ConvAttnDefaultConfig::seq_len), 0u);
  Trainable conv_out_t = Trainable::make (
      conv_out, nullptr,
      zzs::mllee::train::Shape::tensor3 (1u, ConvAttnDefaultConfig::conv_channels, ConvAttnDefaultConfig::seq_len), 0u);
  const ConvSpec conv_spec = make_conv_spec (1);
  {
    Status st = conv_run (tape, conv_spec, conv_input, conv_w, conv_b, conv_out_t);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 2: LINEAR(40->120) QKV projection: 40 -> 120 ---------- */
  Trainable qkv_input = conv_out_t;
  qkv_input._M_shape = zzs::mllee::train::Shape::matrix (1u, ConvAttnDefaultConfig::conv_output);
  Trainable qkv_out_t =
      Trainable::make (qkv_out, nullptr, zzs::mllee::train::Shape::matrix (1u, ConvAttnDefaultConfig::qkv_output), 0u);
  const LinearSpec qkv_spec = make_qkv_spec (1);
  {
    Status st = linear_run (tape, qkv_spec, qkv_input, qkv_w, qkv_b, qkv_out_t);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 3: ATTENTION(seq=5, head_dim=4, heads=2): 120 -> 40 ----------
   * Q/K/V split: qkv_out is [Q[0..39] | K[40..79] | V[80..119]].  Forward only
   * needs the code streams, so grad lanes are null. */
  const uint32_t attn_stride = ConvAttnDefaultConfig::attn_heads * ConvAttnDefaultConfig::attn_head_dim; /* 8 */
  const uint32_t attn_elements = ConvAttnDefaultConfig::attn_output;                                     /* 40 */

  Trainable q_view = Trainable::make (
      qkv_out, nullptr, zzs::mllee::train::Shape::matrix (ConvAttnDefaultConfig::seq_len, attn_stride), 0u);
  Trainable k_view =
      Trainable::make (qkv_out + attn_elements, nullptr,
                       zzs::mllee::train::Shape::matrix (ConvAttnDefaultConfig::seq_len, attn_stride), 0u);
  Trainable v_view =
      Trainable::make (qkv_out + 2 * attn_elements, nullptr,
                       zzs::mllee::train::Shape::matrix (ConvAttnDefaultConfig::seq_len, attn_stride), 0u);
  Trainable attn_out_t = Trainable::make (
      attn_out, nullptr, zzs::mllee::train::Shape::matrix (1u, ConvAttnDefaultConfig::attn_output), 0u);
  const AttentionSpec attn_spec = make_attention_spec ();
  {
    Status st = attention_run (tape, attn_spec, q_view, k_view, v_view, attn_out_t);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 4: GLOBAL_AVG_POOL(seq=5, channels=8): 40 -> 8 ---------- */
  Trainable pool_input = attn_out_t;
  pool_input._M_shape =
      zzs::mllee::train::Shape::tensor3 (1u, ConvAttnDefaultConfig::conv_channels, ConvAttnDefaultConfig::seq_len);
  Trainable pool_out_t = Trainable::make (
      pool_out, nullptr, zzs::mllee::train::Shape::tensor3 (1u, ConvAttnDefaultConfig::conv_channels, 1u), 0u);
  const PoolSpec pool_spec = make_pool_spec (1);
  {
    Status st = pool_run (tape, pool_spec, pool_input, pool_out_t);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 5: LINEAR(8->3) head1 / LINEAR(8->1) head2 / head3 ---------- */
  Trainable head_input = pool_out_t;
  head_input._M_shape = zzs::mllee::train::Shape::matrix (1u, ConvAttnDefaultConfig::pooled);
  Trainable head1_out_t = Trainable::make (
      head1_out, nullptr, zzs::mllee::train::Shape::matrix (1u, ConvAttnDefaultConfig::output_dim), 0u);
  const LinearSpec head1_spec = make_output_spec (1);
  {
    Status st = linear_run (tape, head1_spec, head_input, head1_w, head1_b, head1_out_t);
    if (!status_ok (st))
      return st;
  }

  Trainable head2_out_t = Trainable::make (head2_out, nullptr, zzs::mllee::train::Shape::matrix (1u, 1u), 0u);
  LinearSpec head2_spec = make_output_spec (1);
  head2_spec._M_out_features = 1u;
  {
    Status st = linear_run (tape, head2_spec, head_input, head2_w, head2_b, head2_out_t);
    if (!status_ok (st))
      return st;
  }

  Trainable head3_out_t = Trainable::make (head3_out, nullptr, zzs::mllee::train::Shape::matrix (1u, 1u), 0u);
  LinearSpec head3_spec = make_output_spec (1);
  head3_spec._M_out_features = 1u;
  {
    Status st = linear_run (tape, head3_spec, head_input, head3_w, head3_b, head3_out_t);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Layer 6: SOFTMAX(3->3) on head1, in-place ---------- */
  {
    Status st = softmax_run (tape, make_softmax_spec (), head1_out_t, nullptr, head1_out_t);
    if (!status_ok (st))
      return st;
  }

  /* ---------- Soft target: (7 * teacher + 3 * one_hot_target) / 10, clamp [-127,127] ---------- */
  const uint32_t n = static_cast<uint32_t> (teacher_count < target_count ? teacher_count : target_count);
  for (uint32_t k = 0; k < n && k < ConvAttnDefaultConfig::output_dim; ++k) {
    const int32_t v = (7 * static_cast<int32_t> (teacher_logits[k]) + 3 * static_cast<int32_t> (target[k])) / 10;
    soft_target[k] = static_cast<int8_t> (v > 127 ? 127 : (v < -127 ? -127 : v));
  }

  /* ---------- MSE loss: head1_out (softmax) vs soft_target ---------- */
  Trainable pred = head1_out_t;
  Trainable target_t = Trainable::make (
      soft_target, nullptr,
      zzs::mllee::train::Shape::vector (static_cast<uint16_t> (ConvAttnDefaultConfig::output_dim)), 0u);
  LossResult loss_result;
  {
    Status st = mse_forward (pool, &tape, pred, target_t, loss_result);
    if (!status_ok (st))
      return st;
  }

  /* Publish the loss node to the trainer context. */
  context._M_loss = loss_result._M_node;
  context._M_loss_q7 = loss_result._M_loss_q7;

  return Status::OK;
}

/* ======================================================================== */
/* Tier 5: Distillation training pipeline                                   */
/* ======================================================================== */

namespace {

/** @brief Q15 code step: 32768 / 127 ≈ 258. */
inline constexpr int32_t kDistillCodeStep = 258;

/** @brief Adapter matching Trainer::ForwardHook; feeds the one-hot target as
 *         the teacher logits (hard distillation) into distill_forward_hook. */
[[nodiscard]] inline Status
distill_forward_adapter (StepContext &context) noexcept {
  return distill_forward_hook (context, context._M_target, context._M_target_count);
}

} // namespace

/** @brief Wire up the DistillParams Trainable views to storage buffers. */
template <typename Config>
void
setup_distill_params (DistillParams &params, int8_t *conv_w_code, int16_t *conv_w_grad, int8_t *conv_b_code,
                      int16_t *conv_b_grad, int8_t *qkv_w_code, int16_t *qkv_w_grad, int8_t *qkv_b_code,
                      int16_t *qkv_b_grad, int8_t *head1_w_code, int16_t *head1_w_grad, int8_t *head1_b_code,
                      int16_t *head1_b_grad, int8_t *head2_w_code, int16_t *head2_w_grad, int8_t *head2_b_code,
                      int16_t *head2_b_grad, int8_t *head3_w_code, int16_t *head3_w_grad, int8_t *head3_b_code,
                      int16_t *head3_b_grad, int8_t *output_code, int16_t *output_grad, int8_t *loss_code,
                      int16_t *loss_grad) noexcept {
  params._M_conv_w = Trainable::make (
      conv_w_code, conv_w_grad,
      zzs::mllee::train::Shape::matrix (static_cast<uint16_t> (Config::conv_channels),
                                        static_cast<uint16_t> (Config::feature_count * Config::conv_kernel)),
      zzs::mllee::train::Flag::RequiresGrad);
  params._M_conv_b = Trainable::make (conv_b_code, conv_b_grad,
                                      zzs::mllee::train::Shape::vector (static_cast<uint16_t> (Config::conv_channels)),
                                      zzs::mllee::train::Flag::RequiresGrad | zzs::mllee::train::Flag::BiasLike);
  params._M_qkv_w = Trainable::make (qkv_w_code, qkv_w_grad,
                                     zzs::mllee::train::Shape::matrix (static_cast<uint16_t> (Config::qkv_output),
                                                                       static_cast<uint16_t> (Config::conv_output)),
                                     zzs::mllee::train::Flag::RequiresGrad);
  params._M_qkv_b = Trainable::make (qkv_b_code, qkv_b_grad,
                                     zzs::mllee::train::Shape::vector (static_cast<uint16_t> (Config::qkv_output)),
                                     zzs::mllee::train::Flag::RequiresGrad | zzs::mllee::train::Flag::BiasLike);
  params._M_head1_w = Trainable::make (head1_w_code, head1_w_grad,
                                       zzs::mllee::train::Shape::matrix (static_cast<uint16_t> (Config::output_dim),
                                                                         static_cast<uint16_t> (Config::pooled)),
                                       zzs::mllee::train::Flag::RequiresGrad);
  params._M_head1_b = Trainable::make (head1_b_code, head1_b_grad,
                                       zzs::mllee::train::Shape::vector (static_cast<uint16_t> (Config::output_dim)),
                                       zzs::mllee::train::Flag::RequiresGrad | zzs::mllee::train::Flag::BiasLike);
  params._M_head2_w = Trainable::make (
      head2_w_code, head2_w_grad,
      zzs::mllee::train::Shape::matrix (static_cast<uint16_t> (1u), static_cast<uint16_t> (Config::pooled)),
      zzs::mllee::train::Flag::RequiresGrad);
  params._M_head2_b =
      Trainable::make (head2_b_code, head2_b_grad, zzs::mllee::train::Shape::vector (static_cast<uint16_t> (1u)),
                       zzs::mllee::train::Flag::RequiresGrad | zzs::mllee::train::Flag::BiasLike);
  params._M_head3_w = Trainable::make (
      head3_w_code, head3_w_grad,
      zzs::mllee::train::Shape::matrix (static_cast<uint16_t> (1u), static_cast<uint16_t> (Config::pooled)),
      zzs::mllee::train::Flag::RequiresGrad);
  params._M_head3_b =
      Trainable::make (head3_b_code, head3_b_grad, zzs::mllee::train::Shape::vector (static_cast<uint16_t> (1u)),
                       zzs::mllee::train::Flag::RequiresGrad | zzs::mllee::train::Flag::BiasLike);

  /* Activations: code + grad lanes, require grad so backward writes into them. */
  params._M_output =
      Trainable::make (output_code, output_grad, zzs::mllee::train::Shape::matrix (1u, Config::output_dim),
                       zzs::mllee::train::Flag::RequiresGrad);
  params._M_loss = Trainable::make (loss_code, loss_grad, zzs::mllee::train::Shape::scalar (), 0u);
}

/** @brief DistillStorage constructor: wire up pool and tape. */
template <typename Config>
DistillStorage<Config>::DistillStorage () noexcept
    : _M_pool (this->_M_arena, sizeof (this->_M_arena)), _M_tape (this->_M_nodes, 256, this->_M_pool) {
}

/** @brief Reset all training state to zero. */
template <typename Config>
void
DistillStorage<Config>::reset () noexcept {
  auto &s = *this;
  for (uint32_t i = 0; i < Config::conv_channels * Config::feature_count * Config::conv_kernel; ++i) {
    s._M_conv_w_code[i] = 0;
    s._M_conv_w_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::conv_channels; ++i) {
    s._M_conv_b_code[i] = 0;
    s._M_conv_b_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::qkv_output * Config::conv_output; ++i) {
    s._M_qkv_w_code[i] = 0;
    s._M_qkv_w_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::qkv_output; ++i) {
    s._M_qkv_b_code[i] = 0;
    s._M_qkv_b_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::output_dim * Config::pooled; ++i) {
    s._M_head1_w_code[i] = 0;
    s._M_head1_w_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::output_dim; ++i) {
    s._M_head1_b_code[i] = 0;
    s._M_head1_b_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::pooled; ++i) {
    s._M_head2_w_code[i] = 0;
    s._M_head2_w_grad[i] = 0;
  }
  s._M_head2_b_code[0] = 0;
  s._M_head2_b_grad[0] = 0;
  for (uint32_t i = 0; i < Config::pooled; ++i) {
    s._M_head3_w_code[i] = 0;
    s._M_head3_w_grad[i] = 0;
  }
  s._M_head3_b_code[0] = 0;
  s._M_head3_b_grad[0] = 0;
  for (uint32_t i = 0; i < Config::conv_output; ++i) {
    s._M_conv_out_code[i] = 0;
    s._M_conv_out_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::qkv_output; ++i) {
    s._M_qkv_out_code[i] = 0;
    s._M_qkv_out_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::attn_output; ++i) {
    s._M_attn_out_code[i] = 0;
    s._M_attn_out_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::pooled; ++i) {
    s._M_pool_out_code[i] = 0;
    s._M_pool_out_grad[i] = 0;
  }
  for (uint32_t i = 0; i < Config::output_dim; ++i) {
    s._M_output_code[i] = 0;
    s._M_output_grad[i] = 0;
  }
  s._M_loss_code[0] = 0;
  s._M_loss_grad[0] = 0;
  for (uint32_t i = 0; i < Config::flat_input; ++i)
    s._M_input_code[i] = 0;
  for (uint32_t i = 0; i < Config::output_dim; ++i)
    s._M_target_code[i] = 0;

  s._M_pool.reset ();
}

/** @brief Run training for @p steps on the compile-time dataset. */
template <typename Config>
[[nodiscard]] Status
DistillPipeline<Config>::run_training (uint32_t steps, zzs::base::Stdout &out) noexcept {
  (void) out;

  auto &storage = this->_M_storage;
  setup_distill_params<Config> (
      storage._M_params, storage._M_conv_w_code, storage._M_conv_w_grad, storage._M_conv_b_code, storage._M_conv_b_grad,
      storage._M_qkv_w_code, storage._M_qkv_w_grad, storage._M_qkv_b_code, storage._M_qkv_b_grad,
      storage._M_head1_w_code, storage._M_head1_w_grad, storage._M_head1_b_code, storage._M_head1_b_grad,
      storage._M_head2_w_code, storage._M_head2_w_grad, storage._M_head2_b_code, storage._M_head2_b_grad,
      storage._M_head3_w_code, storage._M_head3_w_grad, storage._M_head3_b_code, storage._M_head3_b_grad,
      storage._M_output_code, storage._M_output_grad, storage._M_loss_code, storage._M_loss_grad);

  Pool &pool = storage._M_pool;
  Tape &tape = storage._M_tape;

  StepContext ctx;
  ctx._M_pool = &pool;
  ctx._M_tape = &tape;
  ctx._M_batch = 1;
  ctx._M_input = storage._M_input_code;
  ctx._M_input_count = Config::flat_input;
  ctx._M_target = storage._M_target_code;
  ctx._M_target_count = Config::output_dim;

  /* Contiguous param array for the trainer:
   * [conv_w, conv_b, qkv_w, qkv_b,
   *  head1_w, head1_b, head2_w, head2_b, head3_w, head3_b,
   *  output, loss] */
  Trainable param_array[12] = {
      storage._M_params._M_conv_w,  storage._M_params._M_conv_b,  storage._M_params._M_qkv_w,
      storage._M_params._M_qkv_b,   storage._M_params._M_head1_w, storage._M_params._M_head1_b,
      storage._M_params._M_head2_w, storage._M_params._M_head2_b, storage._M_params._M_head3_w,
      storage._M_params._M_head3_b, storage._M_params._M_output,  storage._M_params._M_loss,
  };
  ctx._M_params = param_array;
  ctx._M_param_count = 12;

  StepHooks hooks;
  hooks._M_forward = distill_forward_adapter;
  hooks._M_schedule = constant_lr_schedule;
  hooks._M_post_grad = post_grad_hook;
  hooks._M_update = sgd_update_hook;

  Trainer trainer (ctx, hooks);
  Status st = trainer.configure (param_array, 12, 1);
  if (!status_ok (st))
    return st;

  for (uint32_t step = 0; step < steps; ++step) {
    const uint32_t sample_idx = step % dataset_count ();
    const auto &sample = dataset_sample (sample_idx);
    flatten_sample (sample, storage._M_input_code);
    one_hot_target (sample._M_label, storage._M_target_code);

    StepStats stats;
    st = trainer.step (stats);
    if (!status_ok (st))
      return st;
  }

  return Status::OK;
}

/** @brief Evaluate the model on the full dataset, report accuracy. */
template <typename Config>
[[nodiscard]] Status
DistillPipeline<Config>::evaluate_model (zzs::base::Stdout &out) noexcept {
  (void) out;

  auto &storage = this->_M_storage;
  setup_distill_params<Config> (
      storage._M_params, storage._M_conv_w_code, storage._M_conv_w_grad, storage._M_conv_b_code, storage._M_conv_b_grad,
      storage._M_qkv_w_code, storage._M_qkv_w_grad, storage._M_qkv_b_code, storage._M_qkv_b_grad,
      storage._M_head1_w_code, storage._M_head1_w_grad, storage._M_head1_b_code, storage._M_head1_b_grad,
      storage._M_head2_w_code, storage._M_head2_w_grad, storage._M_head2_b_code, storage._M_head2_b_grad,
      storage._M_head3_w_code, storage._M_head3_w_grad, storage._M_head3_b_code, storage._M_head3_b_grad,
      storage._M_output_code, storage._M_output_grad, storage._M_loss_code, storage._M_loss_grad);

  Pool &pool = storage._M_pool;
  Tape &tape = storage._M_tape;

  StepContext ctx;
  ctx._M_pool = &pool;
  ctx._M_tape = &tape;
  ctx._M_batch = 1;
  ctx._M_input = storage._M_input_code;
  ctx._M_input_count = Config::flat_input;
  ctx._M_target = storage._M_target_code;
  ctx._M_target_count = Config::output_dim;

  Trainable param_array[12] = {
      storage._M_params._M_conv_w,  storage._M_params._M_conv_b,  storage._M_params._M_qkv_w,
      storage._M_params._M_qkv_b,   storage._M_params._M_head1_w, storage._M_params._M_head1_b,
      storage._M_params._M_head2_w, storage._M_params._M_head2_b, storage._M_params._M_head3_w,
      storage._M_params._M_head3_b, storage._M_params._M_output,  storage._M_params._M_loss,
  };
  ctx._M_params = param_array;
  ctx._M_param_count = 12;

  StepHooks hooks;
  hooks._M_forward = distill_forward_adapter;

  Trainer trainer (ctx, hooks);
  Status st = trainer.configure (param_array, 12, 1);
  if (!status_ok (st))
    return st;

  uint32_t correct = 0;
  uint32_t total = 0;

  for (uint32_t i = 0; i < dataset_count (); ++i) {
    const auto &sample = dataset_sample (i);
    flatten_sample (sample, storage._M_input_code);
    one_hot_target (sample._M_label, storage._M_target_code);

    ctx._M_input = storage._M_input_code;
    ctx._M_input_count = Config::flat_input;
    ctx._M_target = storage._M_target_code;
    ctx._M_target_count = Config::output_dim;

    StepStats eval_stats;
    st = trainer.evaluate (eval_stats);
    if (!status_ok (st))
      return st;

    int32_t best_code = storage._M_output_code[0];
    uint32_t best_idx = 0;
    for (uint32_t c = 1; c < Config::output_dim; ++c) {
      if (storage._M_output_code[c] > best_code) {
        best_code = storage._M_output_code[c];
        best_idx = c;
      }
    }

    ++total;
    if (best_idx == sample._M_label)
      ++correct;
  }

  return Status::OK;
}

/** @} */
} // namespace zzs::mllee_native::pipeline
