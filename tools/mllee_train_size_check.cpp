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
#include <cstdio>

/**
 * @brief Training-memory probe.
 *
 * Prints the numbers the training gate reads, as `key=value` so the shell
 * wrapper can parse them, and returns non-zero when the documented budget
 * arithmetic stops holding.  The three things it pins are the pool size, the
 * per-node tape cost and the fixed control-plane objects -- the training path is
 * allowed to be larger than inference (it runs offline), but it must stay
 * *bounded*, and a run that cannot be described by `Trainer::budget` is exactly
 * the kind of surprise an embedded caller cannot afford at 3 a.m. on a device.
 */
namespace
{
using namespace zzs::mllee;
using namespace zzs::mllee::train;

/* A tape node is the unit of the budget: it is what one recorded op costs, and
   it is fixed-size, so the pool arithmetic is closed-form rather than a
   "depends how deep your graph is" estimate. */
constexpr uint32_t _S_probe_params = 16u;

/** @brief Largest tape capacity that still fits beside @p params parameters. */
[[nodiscard]] constexpr uint32_t
max_nodes (const uint32_t params) noexcept
{
    for (uint32_t nodes = 1u; nodes <= 4096u; ++nodes)
        if (Trainer::budget (params, nodes) > _S_train_pool_bytes)
            return nodes - 1u;
    return 4096u;
}

void
print (const char *const key, const uint64_t value) noexcept
{
    std::printf ("%s=%llu\n", key, static_cast<unsigned long long> (value));
}
} // namespace

int
main ()
{
    bool ok = true;

    print ("pool_bytes", _S_train_pool_bytes);
    print ("trainer_pool_bytes", _S_trainer_pool_bytes);
    print ("node_bytes", sizeof (Node));
    print ("trainable_bytes", sizeof (Trainable));
    print ("tape_bytes", sizeof (Tape));
    print ("pool_obj_bytes", sizeof (Pool));
    print ("trainer_bytes", sizeof (Trainer));
    print ("step_context_bytes", sizeof (StepContext));
    print ("step_stats_bytes", sizeof (StepStats));
    print ("fit_stats_bytes", sizeof (FitStats));
    print ("step_hooks_bytes", sizeof (StepHooks));
    print ("saved_bytes", sizeof (Saved));
    print ("grad_context_bytes", sizeof (GradContext));

    const uint32_t default_fit = max_nodes (_S_probe_params);
    print ("max_nodes_default", default_fit);
    print ("budget_default_fit", Trainer::budget (_S_probe_params, default_fit));
    print ("max_nodes_one", max_nodes (1u));
    print ("max_params_one_node", [] {
        for (uint32_t params = 1u; params <= 4096u; ++params)
            if (Trainer::budget (params, 1u) > _S_train_pool_bytes)
                return params - 1u;
        return 4096u;
    } ());

    /* The pool budget the trainer validates against must be the pool itself, or
       a step would be checked against a number nothing allocates from. */
    if (_S_train_pool_bytes != _S_trainer_pool_bytes)
        {
            std::printf ("FAIL: pool budget %u != trainer budget %u\n",
                         _S_train_pool_bytes, _S_trainer_pool_bytes);
            ok = false;
        }

    /* The documented default shape has to fit and one more node must not: this
       is the boundary a caller reads in the README, so it is gated, not noted. */
    if (Trainer::budget (_S_probe_params, default_fit) > _S_train_pool_bytes)
        {
            std::printf ("FAIL: %u params x %u nodes exceeds the pool\n",
                         _S_probe_params, default_fit);
            ok = false;
        }
    if (Trainer::budget (_S_probe_params, default_fit + 1u)
        <= _S_train_pool_bytes)
        {
            std::printf ("FAIL: the reported node ceiling is not the ceiling\n");
            ok = false;
        }

    /* Control plane: the objects a training run owns *besides* the pool.  The
       pool is scratch and is budgeted separately; these are the resident
       bookkeeping objects, and they stay small on purpose. */
    constexpr uint32_t control_bytes
        = sizeof (Tape) + sizeof (Pool) + sizeof (Trainer) + sizeof (StepContext)
          + sizeof (StepStats) + sizeof (FitStats) + sizeof (StepHooks);
    print ("control_bytes", control_bytes);
    if (control_bytes > 512u)
        {
            std::printf ("FAIL: training control plane is %u bytes, limit 512\n",
                         control_bytes);
            ok = false;
        }

    /* Backing arena: the training pool is a fixed array, so its footprint is
       the budget plus the arena's own alignment padding and nothing dynamic. */
    print ("arena_bytes", sizeof (AlignedArena<_S_train_pool_bytes>));

    if (!ok)
        return 1;
    std::printf ("train budget check: OK\n");
    return 0;
}
