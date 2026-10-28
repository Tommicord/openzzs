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

#pragma once

#include "mllee/status.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"

#include <cstdint>

/**
 * @brief The step loop that ties a tape, an optimizer and a loss together.
 *
 * The trainer owns policy, not math: it knows the order of a step (zero, record,
 * backward, unscale, clip, update, account) and what to do when a step goes bad
 * (a saturated gradient skips the update instead of applying a clipped one), and
 * it knows nothing about which loss or which optimizer is in use.  Those arrive
 * as function pointers, so this header stays independent of every other training
 * module and a device build links only the hooks it actually calls.
 *
 * Function pointers rather than virtual dispatch, because the module forbids
 * RTTI and the training path forbids a heap: one `StepHooks` struct of plain
 * pointers is the whole polymorphism budget, and it is const, so a trainer can
 * live in .rodata next to the model.
 *
 * The pool is reset between steps, which is the only reason a fixed 8 KiB
 * training arena is enough for an unbounded number of steps: forward state is
 * scratch, and only the tensors the caller keeps pointers to survive a step.
 */
namespace zzs::mllee::train
{
/** @brief Largest parameter list one step may update. */
inline constexpr uint32_t _S_max_trainer_params = 64u;

/** @brief Clean steps before the loss scale is allowed to grow. */
inline constexpr uint32_t _S_scale_patience = 2000u;

/** @brief Growth once the scale is stable: a quarter, which is 1.25 and does not
 *        fit a Q15 lane, so it is applied as a shift rather than a multiplier. */
inline constexpr uint32_t _S_scale_growth_shift = 2u;

/** @brief Training pool budget a step is validated against. */
inline constexpr uint32_t _S_trainer_pool_bytes = 8192u;

/**
 * @brief Everything one step touches, in one struct so a hook never needs a
 *        closure or a heap allocation to reach its inputs.
 *
 * The forward hook fills @p _M_loss and @p _M_loss_q7; the update hook reads the
 * learning rate the trainer took from the schedule hook.
 */
struct StepContext
{
    Pool *_M_pool = nullptr;
    Tape *_M_tape = nullptr;
    const Trainable *_M_params = nullptr;
    uint32_t _M_param_count = 0u;
    const int8_t *_M_input = nullptr;
    uint32_t _M_input_count = 0u;
    const int8_t *_M_target = nullptr;
    uint32_t _M_target_count = 0u;
    uint32_t _M_batch = 1u;
    uint32_t _M_step = 0u;
    Trainable _M_loss;
    int32_t _M_loss_q7 = 0;
    int32_t _M_lr_q15 = 0;
    int32_t _M_loss_scale_q15 = _S_q15_one;
};

/** @brief Records the forward pass and produces the loss node. */
using ForwardHook = Status (*)(StepContext &context);

/** @brief Maps a step index to a Q15 learning rate. */
using ScheduleHook = int32_t (*)(uint32_t step);

/**
 * @brief Post-backward gradient work: unscaling, clipping, penalty injection.
 *
 * @p saturated is the lane count the backward reported, so a hook can decide to
 * zero the update instead; returning a status other than OK aborts the step.
 */
using PostGradHook = Status (*)(StepContext &context, uint32_t saturated);

/** @brief Applies the parameter update with @p context._M_lr_q15. */
using UpdateHook = Status (*)(StepContext &context);

/** @brief The four hooks a step needs; a null hook is skipped, not fatal. */
struct StepHooks
{
    ForwardHook _M_forward = nullptr;
    ScheduleHook _M_schedule = nullptr;
    PostGradHook _M_post_grad = nullptr;
    UpdateHook _M_update = nullptr;
};

/** @brief Per-step outcome, aggregated by the epoch counters. */
struct StepStats
{
    uint32_t _M_step = 0u;
    uint32_t _M_nodes = 0u;
    uint32_t _M_saved_bytes = 0u;
    uint32_t _M_pool_peak = 0u;
    uint32_t _M_saturation_events = 0u;
    uint32_t _M_skipped_updates = 0u;
    int32_t _M_loss_q7 = 0;
    int32_t _M_loss_scale_q15 = _S_q15_one;
    int32_t _M_lr_q15 = 0;
    Status _M_status = Status::OK;
};

/** @brief Aggregates a fit run, for the log line and the budget gate. */
struct FitStats
{
    uint32_t _M_steps = 0u;
    uint32_t _M_epochs = 0u;
    uint32_t _M_skipped_updates = 0u;
    uint32_t _M_saturation_events = 0u;
    uint32_t _M_pool_peak = 0u;
    uint32_t _M_tape_peak_nodes = 0u;
    uint32_t _M_scale_halves = 0u;
    uint32_t _M_scale_grows = 0u;
    int32_t _M_first_loss_q7 = 0;
    int32_t _M_last_loss_q7 = 0;
    int32_t _M_best_loss_q7 = 0;
    uint32_t _M_best_step = 0u;
    /** @brief Last non-OK step status, so a continue-on-error run keeps its reason. */
    Status _M_status = Status::OK;
};

/**
 * @brief Step policy: the loop, the loss-scale adaptation and the accounting.
 *
 * A loss value is Q7 (a Q15 lane cannot hold a magnitude of 1 or more, and a
 * cross-entropy term spans several octaves), while gradients stay Q15 lanes and
 * parameters stay int8 codes; the trainer never converts between the two, it only
 * sequences the hooks that do.
 */
class Trainer
{
  public:
    Trainer (StepContext &base, const StepHooks &hooks) noexcept
        : _M_context (base), _M_hooks (hooks), _M_scale_q15 (_S_q15_one),
          _M_clean_steps (0u), _M_halves (0u), _M_grows (0u),
          _M_saturation_total (0u), _M_first_loss_q7 (0), _M_best_loss_q7 (0),
          _M_best_step (0u), _M_have_best (false)
    {
    }

    Trainer (const Trainer &) = delete;
    Trainer &operator= (const Trainer &) = delete;

    /**
     * @brief Bind the fixed parts of a run and restart its progress counters.
     *
     * Checked once, not per step.  The step index, the saturation total and the
     * loss extremes belong to one run, so a trainer reused over the same
     * context cannot report a best step from the previous run; the loss scale is
     * deliberately left alone so a checkpoint restore works in either order.
     */
    [[nodiscard]] Status
    configure (const Trainable *const params, const uint32_t param_count,
               const uint32_t batch) noexcept
    {
        if (this->_M_context._M_pool == nullptr || this->_M_context._M_tape == nullptr)
            return Status::BAD_LAYOUT;
        if (this->_M_hooks._M_forward == nullptr)
            return Status::BAD_LAYOUT;
        if (params == nullptr || param_count == 0u
            || param_count > _S_max_trainer_params)
            return Status::BAD_LAYOUT;
        if (batch == 0u)
            return Status::BAD_LAYOUT;
        for (uint32_t i = 0; i < param_count; ++i)
            if (!status_ok (params[i].validate ()))
                return Status::BAD_LAYOUT;
        this->_M_context._M_params = params;
        this->_M_context._M_param_count = param_count;
        this->_M_context._M_batch = batch;
        this->_M_context._M_step = 0u;
        this->_M_clean_steps = 0u;
        this->_M_halves = 0u;
        this->_M_grows = 0u;
        this->_M_saturation_total = 0u;
        this->_M_first_loss_q7 = 0;
        this->_M_best_loss_q7 = 0;
        this->_M_best_step = 0u;
        this->_M_have_best = false;
        return Status::OK;
    }

    /** @brief Set the dynamic loss scale explicitly (e.g. restored from a checkpoint). */
    void
    set_loss_scale (const int32_t scale_q15) noexcept
    {
        this->_M_scale_q15 = q15_clamp (scale_q15);
        this->_M_clean_steps = 0u;
    }

    [[nodiscard]] int32_t
    loss_scale () const noexcept
    {
        return this->_M_scale_q15;
    }

    /**
     * @brief One step: reset, record, backward, post-process, update.
     *
     * The pool and tape are reset on every exit path, including a failing hook,
     * so a bad step cannot leak forward state into the next one.
     */
    [[nodiscard]] Status
    step (StepStats &out) noexcept
    {
        StepContext &context = this->_M_context;
        Pool &pool = *context._M_pool;
        Tape &tape = *context._M_tape;

        for (uint32_t i = 0; i < context._M_param_count; ++i)
            context._M_params[i].zero_grad ();

        const Status begun = tape.begin ();
        if (!status_ok (begun))
            {
                pool.reset ();
                out._M_status = begun;
                return begun;
            }

        context._M_loss_scale_q15 = this->_M_scale_q15;
        tape.set_loss_scale (this->_M_scale_q15);
        context._M_lr_q15
            = this->_M_hooks._M_schedule != nullptr
                  ? q15_clamp (this->_M_hooks._M_schedule (context._M_step))
                  : _M_default_lr_q15;

        const Status forward = this->_M_hooks._M_forward (context);
        if (!status_ok (forward))
            return this->abort (out, forward);
        if (!status_ok (context._M_loss.validate ())
            || context._M_loss._M_grad == nullptr)
            return this->abort (out, Status::BAD_LAYOUT);

        const Status backward = tape.backward (context._M_loss);
        if (!status_ok (backward))
            return this->abort (out, backward);

        const TapeStats tape_stats = tape.stats ();
        /* The tape counts saturations per pass, so this is already this step's
           total; the run accumulator is the trainer's own, which is why a
           second trainer over the same tape cannot inherit the first one's. */
        const uint32_t saturated = tape_stats._M_saturation_events;
        this->_M_saturation_total += saturated;

        /* A saturated lane means the Q15 gradient clipped, so the update that
           would follow is wrong in an unknown direction: skip it, shrink the
           scale and let the next step try again with a smaller amplification. */
        const bool usable = saturated == 0u;
        if (!usable)
            {
                this->_M_scale_q15 = q15_clamp (this->_M_scale_q15 / 2);
                ++this->_M_halves;
                this->_M_clean_steps = 0u;
            }
        else if (this->_M_scale_q15 < _S_q15_max
                 && ++this->_M_clean_steps >= _S_scale_patience)
            {
                this->_M_scale_q15 = q15_clamp (
                    this->_M_scale_q15 + (this->_M_scale_q15 >> _S_scale_growth_shift));
                ++this->_M_grows;
                this->_M_clean_steps = 0u;
            }

        if (this->_M_hooks._M_post_grad != nullptr)
            {
                const Status post = this->_M_hooks._M_post_grad (context, saturated);
                if (!status_ok (post))
                    return this->abort (out, post);
            }

        if (usable && this->_M_hooks._M_update != nullptr)
            {
                const Status update = this->_M_hooks._M_update (context);
                if (!status_ok (update))
                    return this->abort (out, update);
            }

        out._M_step = context._M_step;
        out._M_nodes = tape_stats._M_nodes;
        out._M_saved_bytes = tape_stats._M_saved_bytes;
        out._M_pool_peak = pool.stats ()._M_peak;
        out._M_saturation_events = saturated;
        out._M_skipped_updates = usable ? 0u : 1u;
        out._M_loss_q7 = context._M_loss_q7;
        out._M_loss_scale_q15 = this->_M_scale_q15;
        out._M_lr_q15 = context._M_lr_q15;
        out._M_status = Status::OK;

        this->note_progress (context._M_loss_q7);
        ++context._M_step;

        pool.reset ();
        tape.reset ();
        return Status::OK;
    }

    /**
     * @brief Run @p steps steps, or until a hook fails.
     *
     * @param stop_on_error when false, a failing step is counted and the run
     *        continues, which is what a long device run wants: one corrupt
     *        batch must not end training.
     */
    [[nodiscard]] Status
    fit (const uint32_t steps, const bool stop_on_error, FitStats &out) noexcept
    {
        if (steps == 0u)
            return Status::BAD_LAYOUT;
        out = FitStats{};
        int32_t first = 0;
        bool have_first = false;
        for (uint32_t step = 0; step < steps; ++step)
            {
                StepStats stats;
                const Status status = this->step (stats);
                if (!have_first)
                    {
                        first = stats._M_loss_q7;
                        have_first = true;
                    }
                ++out._M_steps;
                out._M_saturation_events += stats._M_saturation_events;
                out._M_skipped_updates += stats._M_skipped_updates;
                if (stats._M_pool_peak > out._M_pool_peak)
                    out._M_pool_peak = stats._M_pool_peak;
                if (stats._M_nodes > out._M_tape_peak_nodes)
                    out._M_tape_peak_nodes = stats._M_nodes;
                out._M_last_loss_q7 = stats._M_loss_q7;
                if (!status_ok (status))
                    {
                        out._M_status = status;
                        if (stop_on_error)
                            return status;
                    }
            }
        out._M_first_loss_q7 = first;
        out._M_epochs = 1u;
        out._M_best_loss_q7 = this->_M_best_loss_q7;
        out._M_best_step = this->_M_best_step;
        out._M_scale_halves = this->_M_halves;
        out._M_scale_grows = this->_M_grows;
        return out._M_status;
    }

    /**
     * @brief Train until the loss reaches @p target_q7 or @p max_steps runs out.
     *
     * @param steps_out receives the steps actually taken, so the caller can tell
     *        an early stop from a budget stop.
     */
    [[nodiscard]] Status
    fit_until (const uint32_t max_steps, const int32_t target_q7,
               const bool stop_on_error, FitStats &out, uint32_t &steps_out) noexcept
    {
        const Status status = this->fit (max_steps, stop_on_error, out);
        steps_out = out._M_steps;
        if (!status_ok (status))
            return status;
        return out._M_last_loss_q7 <= target_q7 ? Status::OK : Status::OUT_OF_RANGE;
    }

    /**
     * @brief Forward-only pass over one batch; records nothing.
     *
     * The tape is not begun, so a hook that calls Tape::record during evaluation
     * fails with INTERNAL instead of silently growing the node array; a hook that
     * serves both paths guards on Tape::is_open().
     */
    [[nodiscard]] Status
    evaluate (StepStats &out) noexcept
    {
        StepContext &context = this->_M_context;
        if (this->_M_hooks._M_forward == nullptr)
            return Status::BAD_LAYOUT;
        context._M_loss_scale_q15 = this->_M_scale_q15;
        const Status forward = this->_M_hooks._M_forward (context);
        out._M_loss_q7 = context._M_loss_q7;
        out._M_status = forward;
        out._M_pool_peak = context._M_pool->stats ()._M_peak;
        context._M_pool->reset ();
        return forward;
    }

    /** @brief Counters a long run reports periodically. */
    [[nodiscard]] FitStats
    totals () const noexcept
    {
        FitStats totals;
        totals._M_steps = this->_M_context._M_step;
        totals._M_saturation_events = this->_M_saturation_total;
        totals._M_scale_halves = this->_M_halves;
        totals._M_scale_grows = this->_M_grows;
        totals._M_first_loss_q7 = this->_M_first_loss_q7;
        totals._M_last_loss_q7 = this->_M_context._M_loss_q7;
        totals._M_best_loss_q7 = this->_M_best_loss_q7;
        totals._M_best_step = this->_M_best_step;
        totals._M_pool_peak = this->_M_context._M_pool->stats ()._M_peak;
        return totals;
    }

    [[nodiscard]] StepContext &
    context () noexcept
    {
        return this->_M_context;
    }

    [[nodiscard]] const StepContext &
    context () const noexcept
    {
        return this->_M_context;
    }

    /**
     * @brief Fixed bytes a run costs outside the pool: the tape's node array
     *        plus the caller's parameter view array.
     *
     * Both are caller-owned static storage, so this is the part of the training
     * footprint the pool's own peak never shows.  Saved tensors and the
     * backward's scratch come out of the pool and are measured separately.
     */
    [[nodiscard]] static constexpr uint32_t
    budget (const uint32_t params, const uint32_t nodes) noexcept
    {
        return tape_node_bytes (nodes)
               + params * static_cast<uint32_t> (sizeof (Trainable));
    }

  private:
    /** @brief Learning rate when no schedule hook is installed (0.05 in Q15). */
    static constexpr int32_t _M_default_lr_q15 = 1638;

    [[nodiscard]] Status
    abort (StepStats &out, const Status status) noexcept
    {
        out._M_status = status;
        out._M_step = this->_M_context._M_step;
        this->_M_context._M_pool->reset ();
        this->_M_context._M_tape->reset ();
        return status;
    }

    void
    note_progress (const int32_t loss_q7) noexcept
    {
        if (this->_M_context._M_step == 0u)
            this->_M_first_loss_q7 = loss_q7;
        if (!this->_M_have_best || loss_q7 < this->_M_best_loss_q7)
            {
                this->_M_best_loss_q7 = loss_q7;
                this->_M_best_step = this->_M_context._M_step;
                this->_M_have_best = true;
            }
    }

    StepContext &_M_context;
    StepHooks _M_hooks;
    int32_t _M_scale_q15;
    uint32_t _M_clean_steps;
    uint32_t _M_halves;
    uint32_t _M_grows;
    uint32_t _M_saturation_total;
    int32_t _M_first_loss_q7;
    int32_t _M_best_loss_q7;
    uint32_t _M_best_step;
    bool _M_have_best;
};

static_assert (sizeof (StepContext) <= 140u, "step context is copied by reference, not by value");
static_assert (sizeof (StepStats) <= 40u, "per-step stats stay a leaf");
static_assert (sizeof (FitStats) <= 64u, "run stats stay a leaf");
static_assert (sizeof (StepHooks) <= 32u, "four plain function pointers, no vtable");
} // namespace zzs::mllee::train
