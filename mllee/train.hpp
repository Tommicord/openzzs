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

/**
 * @brief The training-side aggregate include.
 *
 * Deliberately separate from mllee.hpp.  Inference pays for the 1024-byte module
 * RAM budget and a 16 KiB flash budget, and a device that only runs a model has
 * no reason to see a tape, an optimizer or a backward kernel -- an aggregate that
 * pulled training in would make the flash gate a function of the training module
 * count.  `mllee.hpp` therefore stays inference-only and this header is the
 * training caller's single include.
 *
 * The order is the dependency order, not alphabetical, so a TU that includes only
 * what it needs still compiles:
 *
 *   - foundation: the value model (train_fixed), the arena (train_pool), the
 *     gradient-bearing view (train_tensor), the tape (train_graph);
 *   - math: activations, backward kernels, and the SIMD primitives both use;
 *   - policy: the step loop (train_trainer);
 *   - layers, losses, optimizers, regularizers, data and lifecycle.
 *
 * Nothing here is a translation unit: every header is `inline`/`constexpr`, so
 * including the whole set costs compile time but no link time, and a linker still
 * keeps only the kernels a caller actually calls.
 */

#include "mllee/train_fixed.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"
#include "mllee/train_graph.hpp"

#include "mllee/train_activations.hpp"
#include "mllee/train_backward_kernels.hpp"
#include "mllee/train_simd.hpp"

#include "mllee/train_trainer.hpp"

#include "mllee/train_calibration.hpp"
#include "mllee/train_checkpoint.hpp"
#include "mllee/train_conv.hpp"
#include "mllee/train_distill.hpp"
#include "mllee/train_dropout.hpp"
#include "mllee/train_embedding.hpp"
#include "mllee/train_fake_quant.hpp"
#include "mllee/train_grad_check.hpp"
#include "mllee/train_grad_clip.hpp"
#include "mllee/train_init.hpp"
#include "mllee/train_label_smoothing.hpp"
#include "mllee/train_linear.hpp"
#include "mllee/train_loss.hpp"
#include "mllee/train_loss_scale.hpp"
#include "mllee/train_lr_schedule.hpp"
#include "mllee/train_model_averaging.hpp"
#include "mllee/train_optim.hpp"
#include "mllee/train_pooling.hpp"
#include "mllee/train_prune.hpp"
#include "mllee/train_random.hpp"
#include "mllee/train_regularization.hpp"
#include "mllee/train_running_stats.hpp"
#include "mllee/train_softmax.hpp"
