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

#include "mllee/train.hpp"

#include <cstdint>

/**
 * @brief Stack probe for a convolutional training model.
 *
 * The linear probe measures the cheapest model that still exercises the trainer,
 * which is not the stack a real one costs: a conv layer's backward, a max-pool
 * backward that stages its window, and a dropout backward that re-reads its mask
 * are all larger frames, and none of them appears in the linear chain.  This
 * probe builds the smallest model that reaches every one of them --
 * conv(relu) -> max-pool -> dropout -> MSE -- so the gate adds their frames to
 * the same budget instead of assuming they behave like `linear_backward`.
 *
 * Compiled with -fstack-usage and never linked: only the .su rows are read.
 */
namespace
{
using namespace zzs::mllee;
using namespace zzs::mllee::train;

/** @brief Convolution geometry, fixed so the frames below are deterministic. */
constexpr uint16_t _S_deep_batch = 1u;
constexpr uint16_t _S_deep_in_channels = 2u;
constexpr uint16_t _S_deep_out_channels = 3u;
constexpr uint16_t _S_deep_in_length = 4u;
constexpr uint16_t _S_deep_kernel = 3u;
/** @brief Padding 1 with stride 1 keeps the conv output at the input length. */
constexpr uint16_t _S_deep_conv_length = 4u;
constexpr uint16_t _S_deep_pool_kernel = 2u;
constexpr uint16_t _S_deep_pool_length = 2u;

constexpr uint32_t _S_deep_pooled
    = static_cast<uint32_t> (_S_deep_batch) * _S_deep_out_channels
      * _S_deep_pool_length;

/**
 * @brief The deep forward hook: one conv, one pool, one dropout, one MSE.
 *
 * Every operand is carved from the step's own pool and saved through the tape,
 * exactly as a caller's step would, so the frames the gate reads are the ones a
 * device run reaches -- including the arena traffic of the max-pool scratch.
 */
Status
deep_forward (StepContext &context)
{
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    const Trainable &weight = context._M_params[0];
    const Trainable &bias = context._M_params[1];

    ConvSpec conv;
    conv._M_batch = _S_deep_batch;
    conv._M_in_channels = _S_deep_in_channels;
    conv._M_out_channels = _S_deep_out_channels;
    conv._M_in[0] = _S_deep_in_length;
    conv._M_kernel[0] = _S_deep_kernel;
    /* A stride-1, pad-1, 3-tap conv keeps its length, so the pool below sees the
       geometry the probe names. */
    conv._M_pad[0] = 1;
    conv._M_pad[1] = 1;
    conv._M_act = Act::RELU;

    PoolSpec pooled;
    pooled._M_batch = _S_deep_batch;
    pooled._M_channels = _S_deep_out_channels;
    pooled._M_in[0] = _S_deep_conv_length;
    pooled._M_kernel[0] = _S_deep_pool_kernel;
    pooled._M_stride[0] = _S_deep_pool_kernel;
    pooled._M_kind = PoolKind::MAX;

    DropoutConfig dropped;
    const Status init = dropout_config_init (
        dropped, DropoutKind::Elementwise, true, _S_q15_one / 5u,
        0x9E3779B97F4A7C15ull, _S_deep_pooled, 1u, _S_deep_pooled, 0);
    if (!status_ok (init))
        return init;

    int8_t *input_codes = nullptr;
    int8_t *hidden_codes = nullptr;
    int16_t *hidden_grad = nullptr;
    int8_t *pooled_codes = nullptr;
    int16_t *pooled_grad = nullptr;
    int8_t *dropped_codes = nullptr;
    int16_t *dropped_grad = nullptr;
    int8_t *target_codes = nullptr;
    if (!status_ok (pool.carve_typed (conv.input_elements (), input_codes))
        || !status_ok (pool.carve_typed (conv.output_elements (), hidden_codes))
        || !status_ok (pool.carve_typed (conv.output_elements (), hidden_grad))
        || !status_ok (pool.carve_typed (pooled.output_elements (), pooled_codes))
        || !status_ok (pool.carve_typed (pooled.output_elements (), pooled_grad))
        || !status_ok (pool.carve_typed (_S_deep_pooled, dropped_codes))
        || !status_ok (pool.carve_typed (_S_deep_pooled, dropped_grad))
        || !status_ok (pool.carve_typed (_S_deep_pooled, target_codes)))
        return Status::ARENA_OVERFLOW;

    const Trainable input = Trainable::make (input_codes, nullptr,
                                             conv.input_shape (), Flag::Frozen);
    Trainable hidden = Trainable::make (hidden_codes, hidden_grad,
                                        conv.output_shape (), Flag::RequiresGrad);
    Trainable downsampled = Trainable::make (pooled_codes, pooled_grad,
                                             pooled.output_shape (),
                                             Flag::RequiresGrad);
    Trainable noisy = Trainable::make (dropped_codes, dropped_grad,
                                       Shape::vector (static_cast<uint16_t> (
                                           _S_deep_pooled)),
                                       Flag::RequiresGrad);
    const Trainable target = Trainable::make (target_codes, nullptr,
                                              Shape::vector (static_cast<uint16_t> (
                                                  _S_deep_pooled)),
                                              Flag::Frozen);

    const Status run_conv = conv_run (tape, conv, input, weight, bias, hidden);
    if (!status_ok (run_conv))
        return run_conv;
    const Status run_pool = pool_run (tape, pooled, hidden, downsampled);
    if (!status_ok (run_pool))
        return run_pool;
    const Status run_drop = dropout_run (tape, dropped, downsampled, noisy);
    if (!status_ok (run_drop))
        return run_drop;

    LossResult result;
    const Status loss = mse_forward (pool, &tape, noisy, target, result);
    if (!status_ok (loss))
        return loss;
    context._M_loss = result._M_node;
    context._M_loss_q7 = result._M_loss_q7;
    return Status::OK;
}

Status
deep_update (StepContext &context)
{
    for (uint32_t p = 0u; p < context._M_param_count; ++p)
        {
            const Trainable &param = context._M_params[p];
            for (uint32_t i = 0u; i < param._M_count; ++i)
                param.set_code (i, param.code_at (i)
                                      - q15_to_code (q15_mul (context._M_lr_q15,
                                                              param.grad_at (i))));
        }
    return Status::OK;
}

int32_t
deep_schedule (const uint32_t)
{
    return _S_q15_one / 16;
}

/** @brief Batch the deep probe configures its trainer with. */
constexpr uint32_t _S_deep_batch_rows = 2u;
} // namespace

/**
 * @brief Deepest step of a convolutional model.
 *
 * @param trainer a Trainer already configured over caller-owned pool, tape and
 *                parameter storage.
 */
extern "C" int
mllee_train_worst_deep_step (Trainer &trainer, StepStats &out,
                             const Trainable *params, const uint32_t param_count)
{
    const StepHooks hooks{ deep_forward, deep_schedule, nullptr, deep_update };
    Trainer local (trainer.context (), hooks);
    if (!status_ok (local.configure (params, param_count, _S_deep_batch_rows)))
        return 0;
    return status_ok (local.step (out)) ? 1 : 0;
}
