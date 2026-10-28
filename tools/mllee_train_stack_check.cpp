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
 * @brief Training stack-usage probe.
 *
 * Compile with -fstack-usage and read the .su output for the frames the training
 * gate adds up.  Everything with a footprint -- the arena, the node array, the
 * parameter views, the Trainer itself -- is owned by the caller and reached
 * through a reference, so no probe entry point is charged for it and the file
 * keeps the module's own no-writable-statics property (the .o is checked for an
 * empty .data/.bss exactly like the inference probe is).
 *
 * The forward hook is a real layer, not a stub: a linear op into an MSE loss.
 * Measuring the trainer frame alone would understate the worst case, because the
 * deepest stack a step reaches is inside the recorded op's own forward and
 * backward, and those are the frames the gate has to bound.
 */
namespace
{
using namespace zzs::mllee;
using namespace zzs::mllee::train;

/** @brief Layer geometry the probe fixes so the frames below are deterministic. */
constexpr uint16_t _S_probe_in_features = 8u;
constexpr uint16_t _S_probe_out_features = 4u;
constexpr uint32_t _S_probe_batch = 2u;

/**
 * @brief The worst-case step body: one recorded linear op, one recorded loss op.
 *
 * `linear_run` then `mse_forward` is the shortest chain that still reaches both
 * a tensor-layer backward and a loss backward, which is what a real model's step
 * costs.  Operands are carved from the step's own pool, so the probe exercises
 * the same arena path a caller does.
 */
Status
probe_forward (StepContext &context)
{
    Pool &pool = *context._M_pool;
    Tape &tape = *context._M_tape;
    const Trainable &weight = context._M_params[0];

    LinearSpec spec;
    spec._M_batch = static_cast<uint16_t> (context._M_batch);
    spec._M_in_features = _S_probe_in_features;
    spec._M_out_features = _S_probe_out_features;

    int8_t *input_codes = nullptr;
    int16_t *input_grad = nullptr;
    int8_t *pred_codes = nullptr;
    int16_t *pred_grad = nullptr;
    int8_t *target_codes = nullptr;
    if (!status_ok (pool.carve_typed (static_cast<uint32_t> (_S_probe_batch)
                                          * _S_probe_in_features,
                                      input_codes)))
        return Status::ARENA_OVERFLOW;
    if (!status_ok (pool.carve_typed (weight._M_count, input_grad)))
        return Status::ARENA_OVERFLOW;
    if (!status_ok (pool.carve_typed (spec.output_elements (), pred_codes)))
        return Status::ARENA_OVERFLOW;
    if (!status_ok (pool.carve_typed (spec.output_elements (), pred_grad)))
        return Status::ARENA_OVERFLOW;
    if (!status_ok (pool.carve_typed (spec.output_elements (), target_codes)))
        return Status::ARENA_OVERFLOW;

    const Trainable input = Trainable::make (input_codes, nullptr,
                                             spec.input_shape (),
                                             Flag::Frozen);
    Trainable pred = Trainable::make (pred_codes, pred_grad,
                                      spec.output_shape (),
                                      Flag::RequiresGrad);
    const Trainable target = Trainable::make (target_codes, nullptr,
                                              spec.output_shape (),
                                              Flag::Frozen);

    const Status run = linear_run (tape, spec, input, weight,
                                   context._M_params[1], pred);
    if (!status_ok (run))
        return run;

    LossResult result;
    const Status loss = mse_forward (pool, &tape, pred, target, result);
    if (!status_ok (loss))
        return loss;
    context._M_loss = result._M_node;
    context._M_loss_q7 = result._M_loss_q7;
    return Status::OK;
}

Status
probe_update (StepContext &context)
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
probe_schedule (const uint32_t)
{
    return _S_q15_one / 16;
}
} // namespace

/**
 * @brief Deepest single training step.
 *
 * @param trainer a Trainer already configured over caller-owned pool, tape and
 *                parameter storage.
 */
extern "C" int
mllee_train_worst_step (Trainer &trainer, StepStats &out, const Trainable *params,
                        const uint32_t param_count)
{
    const StepHooks hooks{ probe_forward, probe_schedule, nullptr, probe_update };
    Trainer local (trainer.context (), hooks);
    if (!status_ok (local.configure (params, param_count, _S_probe_batch)))
        return 0;
    return status_ok (local.step (out)) ? 1 : 0;
}

/**
 * @brief A whole fit run: proves `fit` adds no frame of its own beyond `step`.
 */
extern "C" int
mllee_train_worst_fit (Trainer &trainer, FitStats &out, const Trainable *params,
                       const uint32_t param_count)
{
    const StepHooks hooks{ probe_forward, probe_schedule, nullptr, probe_update };
    Trainer local (trainer.context (), hooks);
    if (!status_ok (local.configure (params, param_count, _S_probe_batch)))
        return 0;
    return status_ok (local.fit (4u, true, out)) ? 1 : 0;
}
