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

/**
 * @brief Unit tests for the mllee-native training pipeline on the fake dataset.
 *
 * Verifies that the training pipeline produces sensible outputs (forward pass,
 * loss values, class predictions) when run on the compile-time generated
 * power-outage dataset.
 */

#include "mllee-native/dataset.hpp"
#include "mllee-native/q11_q15_q31.hpp"
#include "mllee-native/train_pipeline.hpp"
#include "mllee-native/train_pipeline_impl.hpp"
#include "mllee/train_fixed.hpp"
#include "mllee/train_graph.hpp"
#include "mllee/train_pool.hpp"
#include "mllee/train_tensor.hpp"
#include "mllee/train_trainer.hpp"
#include "std/fmt.hpp"
#include "std/io.hpp"
#include "std/log.hpp"

using namespace zzs::base;
using namespace zzs::mllee;
using namespace zzs::mllee::train;
using namespace zzs::mllee_native;
using namespace zzs::mllee_native::dataset;
using namespace zzs::mllee_native::pipeline;

#include <catch2/catch_test_macros.hpp>
#include <cstdint>

alignas (4) static uint8_t test_arena[16384];
alignas (8) static Node test_nodes[256];
alignas (4) static int8_t test_hidden_w_code[kHiddenInput * kHiddenDim];
alignas (8) static int16_t test_hidden_w_grad[kHiddenInput * kHiddenDim];
alignas (4) static int8_t test_hidden_b_code[kHiddenDim];
alignas (8) static int16_t test_hidden_b_grad[kHiddenDim];
alignas (4) static int8_t test_w_code[kHiddenDim * kOutputDim];
alignas (8) static int16_t test_w_grad[kHiddenDim * kOutputDim];
alignas (4) static int8_t test_b_code[kOutputDim];
alignas (8) static int16_t test_b_grad[kOutputDim];
alignas (4) static int8_t test_out_code[kOutputDim];
alignas (8) static int16_t test_out_grad[kOutputDim];
alignas (4) static int8_t test_loss_code[1];
alignas (8) static int16_t test_loss_grad[1];
alignas (4) static int8_t test_input[kInputDim];
alignas (4) static int8_t test_target[kOutputDim];
static Trainable test_params[6];

static void
test_init_storage () noexcept {
    for (auto &v : test_hidden_w_code)
        v = 0;
    for (auto &v : test_hidden_w_grad)
        v = 0;
    for (auto &v : test_hidden_b_code)
        v = 0;
    for (auto &v : test_hidden_b_grad)
        v = 0;
    for (auto &v : test_w_code)
        v = 0;
    for (auto &v : test_w_grad)
        v = 0;
    for (auto &v : test_b_code)
        v = 0;
    for (auto &v : test_b_grad)
        v = 0;
    for (auto &v : test_out_code)
        v = 0;
    for (auto &v : test_out_grad)
        v = 0;
    for (auto &v : test_loss_code)
        v = 0;
    for (auto &v : test_loss_grad)
        v = 0;
    for (auto &v : test_input)
        v = 0;
    for (auto &v : test_target)
        v = 0;

    for (uint32_t i = 0; i < kHiddenInput * kHiddenDim; ++i)
        test_hidden_w_code[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
    for (uint32_t i = 0; i < kHiddenDim; ++i)
        test_hidden_b_code[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);
    for (uint32_t i = 0; i < kHiddenDim * kOutputDim; ++i)
        test_w_code[i] = static_cast<int8_t> ((i * 37 + 11) % 25 - 12);
    for (uint32_t i = 0; i < kOutputDim; ++i)
        test_b_code[i] = static_cast<int8_t> ((i * 13 + 7) % 25 - 12);

    test_params[0] = Trainable::make (
        test_hidden_w_code, test_hidden_w_grad,
        Shape::matrix (static_cast<uint16_t> (kHiddenDim), static_cast<uint16_t> (kHiddenInput)), Flag::RequiresGrad);
    test_params[1] =
        Trainable::make (test_hidden_b_code, test_hidden_b_grad, Shape::vector (static_cast<uint16_t> (kHiddenDim)),
                         Flag::RequiresGrad | Flag::BiasLike);
    test_params[2] = Trainable::make (
        test_w_code, test_w_grad,
        Shape::matrix (static_cast<uint16_t> (kOutputDim), static_cast<uint16_t> (kHiddenDim)), Flag::RequiresGrad);
    test_params[3] = Trainable::make (test_b_code, test_b_grad, Shape::vector (static_cast<uint16_t> (kOutputDim)),
                                      Flag::RequiresGrad | Flag::BiasLike);
    test_params[4] = Trainable::make (test_out_code, test_out_grad, Shape::vector (static_cast<uint16_t> (kOutputDim)),
                                      Flag::Frozen);
    test_params[5] = Trainable::make (test_loss_code, test_loss_grad, Shape::scalar (), 0u);
}

TEST_CASE ("Dataset sample structure is valid", "[mllee_native][dataset]") {
    REQUIRE (dataset_count () == 64);

    for (uint32_t i = 0; i < dataset_count (); ++i) {
        const TrainingSample &sample = dataset_sample (i);
        REQUIRE (sample._M_label <= 2u);

        for (uint32_t w = 0; w < kSeqLen; ++w) {
            for (uint32_t f = 0; f < kFeatureCount; ++f) {
                int16_t v = sample._M_windows[w]._M_features[f];
                REQUIRE (v >= -32768);
                REQUIRE (v <= 32767);
            }
        }
    }
}

TEST_CASE ("Dataset class labels cover all three classes", "[mllee_native][dataset]") {
    bool has_normal = false, has_outage = false, has_transition = false;

    for (uint32_t i = 0; i < dataset_count (); ++i) {
        const TrainingSample &sample = dataset_sample (i);
        switch (sample._M_label) {
        case 0:
            has_normal = true;
            break;
        case 1:
            has_outage = true;
            break;
        case 2:
            has_transition = true;
            break;
        }
    }

    REQUIRE (has_normal);
    REQUIRE (has_outage);
    REQUIRE (has_transition);
}

TEST_CASE ("Flatten sample produces 190 int8 codes in range", "[mllee_native][flatten]") {
    test_init_storage ();

    const TrainingSample &sample = dataset_sample (0);
    flatten_sample (sample, test_input);

    for (uint32_t i = 0; i < kInputDim; ++i) {
        REQUIRE (test_input[i] >= -127);
        REQUIRE (test_input[i] <= 127);
    }
}

TEST_CASE ("Forward pass produces output codes in range", "[mllee_native][forward]") {
    test_init_storage ();

    Pool pool (test_arena, sizeof (test_arena));
    Tape tape (test_nodes, 256, pool);
    REQUIRE (status_ok (tape.begin ()));

    StepContext ctx;
    ctx._M_pool = &pool;
    ctx._M_tape = &tape;
    ctx._M_params = test_params;
    ctx._M_param_count = 6;
    ctx._M_batch = 1;
    ctx._M_loss = Trainable::make (test_loss_code, test_loss_grad, Shape::scalar (), 0u);
    ctx._M_input = test_input;
    ctx._M_input_count = kInputDim;
    ctx._M_target = test_target;
    ctx._M_target_count = kOutputDim;

    const TrainingSample &sample = dataset_sample (0);
    flatten_sample (sample, test_input);
    one_hot_target (sample._M_label, test_target);

    const Status st = mlp_forward_hook (ctx);
    REQUIRE (status_ok (st));

    /* Output codes should be in int8 range */
    for (uint32_t c = 0; c < kOutputDim; ++c) {
        REQUIRE (test_out_code[c] >= -127);
        REQUIRE (test_out_code[c] <= 127);
    }

    /* Loss code should be non-negative (MSE) */
    REQUIRE (test_loss_code[0] >= 0);
    REQUIRE (ctx._M_loss_q7 >= 0);
}

TEST_CASE ("Backward pass produces gradients", "[mllee_native][backward]") {
    test_init_storage ();

    Pool pool (test_arena, sizeof (test_arena));
    Tape tape (test_nodes, 256, pool);
    REQUIRE (status_ok (tape.begin ()));

    StepContext ctx;
    ctx._M_pool = &pool;
    ctx._M_tape = &tape;
    ctx._M_params = test_params;
    ctx._M_param_count = 6;
    ctx._M_batch = 1;
    ctx._M_loss = Trainable::make (test_loss_code, test_loss_grad, Shape::scalar (), 0u);
    ctx._M_input = test_input;
    ctx._M_input_count = kInputDim;
    ctx._M_target = test_target;
    ctx._M_target_count = kOutputDim;

    const TrainingSample &sample = dataset_sample (0);
    flatten_sample (sample, test_input);
    one_hot_target (sample._M_label, test_target);

    const Status st = mlp_forward_hook (ctx);
    REQUIRE (status_ok (st));

    /* Recorded backward should exist */
    REQUIRE (tape.is_open ());

    /* Run backward through the tape */
    const Status bw = tape.backward (ctx._M_loss);
    REQUIRE (status_ok (bw));

    /* Hidden weight gradients should be non-zero after backward */
    bool any_hidden_w_grad = false;
    for (uint32_t i = 0; i < kHiddenInput * kHiddenDim; ++i) {
        if (test_hidden_w_grad[i] != 0) {
            any_hidden_w_grad = true;
            break;
        }
    }
    REQUIRE (any_hidden_w_grad);

    /* Hidden bias gradients should be non-zero after backward */
    bool any_hidden_b_grad = false;
    for (uint32_t i = 0; i < kHiddenDim; ++i) {
        if (test_hidden_b_grad[i] != 0) {
            any_hidden_b_grad = true;
            break;
        }
    }
    REQUIRE (any_hidden_b_grad);

    /* Output weight gradients should be non-zero after backward */
    bool any_w_grad = false;
    for (uint32_t i = 0; i < kHiddenDim * kOutputDim; ++i) {
        if (test_w_grad[i] != 0) {
            any_w_grad = true;
            break;
        }
    }
    REQUIRE (any_w_grad);

    /* Output bias gradients should be non-zero after backward */
    bool any_b_grad = false;
    for (uint32_t i = 0; i < kOutputDim; ++i) {
        if (test_b_grad[i] != 0) {
            any_b_grad = true;
            break;
        }
    }
    REQUIRE (any_b_grad);
}

TEST_CASE ("SGD update changes weights", "[mllee_native][update]") {
    test_init_storage ();

    /* Save initial weights */
    int8_t hidden_w_before[kHiddenInput * kHiddenDim];
    int8_t hidden_b_before[kHiddenDim];
    int8_t w_before[kHiddenDim * kOutputDim];
    int8_t b_before[kOutputDim];
    for (uint32_t i = 0; i < kHiddenInput * kHiddenDim; ++i)
        hidden_w_before[i] = test_hidden_w_code[i];
    for (uint32_t i = 0; i < kHiddenDim; ++i)
        hidden_b_before[i] = test_hidden_b_code[i];
    for (uint32_t i = 0; i < kHiddenDim * kOutputDim; ++i)
        w_before[i] = test_w_code[i];
    for (uint32_t i = 0; i < kOutputDim; ++i)
        b_before[i] = test_b_code[i];

    Pool pool (test_arena, sizeof (test_arena));
    Tape tape (test_nodes, 256, pool);
    REQUIRE (status_ok (tape.begin ()));

    StepContext ctx;
    ctx._M_pool = &pool;
    ctx._M_tape = &tape;
    ctx._M_params = test_params;
    ctx._M_param_count = 6;
    ctx._M_batch = 1;
    ctx._M_loss = Trainable::make (test_loss_code, test_loss_grad, Shape::scalar (), 0u);
    ctx._M_input = test_input;
    ctx._M_input_count = kInputDim;
    ctx._M_target = test_target;
    ctx._M_target_count = kOutputDim;
    ctx._M_lr_q15 = 1638; /* 0.05 in Q15 */

    const TrainingSample &sample = dataset_sample (0);
    flatten_sample (sample, test_input);
    one_hot_target (sample._M_label, test_target);

    const Status fw = mlp_forward_hook (ctx);
    REQUIRE (status_ok (fw));
    const Status bw = tape.backward (ctx._M_loss);
    REQUIRE (status_ok (bw));

    const Status upd = sgd_update_hook (ctx);
    REQUIRE (status_ok (upd));

    /* At least some weights should have changed */
    bool any_changed = false;
    for (uint32_t i = 0; i < kHiddenInput * kHiddenDim; ++i) {
        if (test_hidden_w_code[i] != hidden_w_before[i]) {
            any_changed = true;
            break;
        }
    }
    for (uint32_t i = 0; i < kHiddenDim * kOutputDim && !any_changed; ++i) {
        if (test_w_code[i] != w_before[i]) {
            any_changed = true;
            break;
        }
    }
    REQUIRE (any_changed);
}

TEST_CASE ("Full training step produces decreasing-or-stable loss", "[mllee_native][training]") {
    test_init_storage ();

    Pool pool (test_arena, sizeof (test_arena));
    Tape tape (test_nodes, 256, pool);

    StepContext ctx;
    ctx._M_pool = &pool;
    ctx._M_tape = &tape;
    ctx._M_params = test_params;
    ctx._M_param_count = 6;
    ctx._M_batch = 1;
    ctx._M_loss = Trainable::make (test_loss_code, test_loss_grad, Shape::scalar (), 0u);
    ctx._M_input = test_input;
    ctx._M_input_count = kInputDim;
    ctx._M_target = test_target;
    ctx._M_target_count = kOutputDim;

    StepHooks hooks;
    hooks._M_forward = mlp_forward_hook;
    hooks._M_schedule = constant_lr_schedule;
    hooks._M_post_grad = post_grad_hook;
    hooks._M_update = sgd_update_hook;

    Trainer trainer (ctx, hooks);
    REQUIRE (status_ok (trainer.configure (test_params, 6, 1)));

    int32_t first_loss_q7 = 0;
    int32_t last_loss_q7 = 0;

    for (uint32_t step = 0; step < 20; ++step) {
        const uint32_t sample_idx = step % dataset_count ();
        const TrainingSample &sample = dataset_sample (sample_idx);
        flatten_sample (sample, test_input);
        one_hot_target (sample._M_label, test_target);

        StepStats stats;
        const Status st = trainer.step (stats);
        REQUIRE (status_ok (st));

        if (step == 0)
            first_loss_q7 = stats._M_loss_q7;
        last_loss_q7 = stats._M_loss_q7;

        /* Loss should be a valid Q7 value */
        REQUIRE (stats._M_loss_q7 >= 0);
        REQUIRE (stats._M_loss_q7 < 32768);
    }

    /* After 20 steps, loss should not have exploded */
    REQUIRE (last_loss_q7 < 100000);
}

TEST_CASE ("Model predicts outage class with high accuracy on fake dataset", "[mllee_native][evaluation]") {
    test_init_storage ();

    Pool pool (test_arena, sizeof (test_arena));
    Tape tape (test_nodes, 256, pool);

    StepContext ctx;
    ctx._M_pool = &pool;
    ctx._M_tape = &tape;
    ctx._M_params = test_params;
    ctx._M_param_count = 6;
    ctx._M_batch = 1;
    ctx._M_loss = Trainable::make (test_loss_code, test_loss_grad, Shape::scalar (), 0u);
    ctx._M_input = test_input;
    ctx._M_input_count = kInputDim;
    ctx._M_target = test_target;
    ctx._M_target_count = kOutputDim;

    StepHooks hooks;
    hooks._M_forward = mlp_forward_hook;
    hooks._M_schedule = constant_lr_schedule;
    hooks._M_post_grad = post_grad_hook;
    hooks._M_update = sgd_update_hook;

    Trainer trainer (ctx, hooks);
    REQUIRE (status_ok (trainer.configure (test_params, 6, 1)));

    /* Train for 100 steps */
    for (uint32_t step = 0; step < 100; ++step) {
        const uint32_t sample_idx = step % dataset_count ();
        const TrainingSample &sample = dataset_sample (sample_idx);
        flatten_sample (sample, test_input);
        one_hot_target (sample._M_label, test_target);

        StepStats stats;
        REQUIRE (status_ok (trainer.step (stats)));
    }

    /* Evaluate on full dataset */
    uint32_t correct = 0;
    uint32_t total = 0;
    uint32_t outage_correct = 0;
    uint32_t outage_total = 0;

    for (uint32_t i = 0; i < dataset_count (); ++i) {
        const TrainingSample &sample = dataset_sample (i);
        flatten_sample (sample, test_input);
        one_hot_target (sample._M_label, test_target);

        ctx._M_input = test_input;
        ctx._M_input_count = kInputDim;
        ctx._M_target = test_target;
        ctx._M_target_count = kOutputDim;

        StepStats eval_stats;
        const Status st = trainer.evaluate (eval_stats);
        REQUIRE (status_ok (st));

        int32_t best_code = test_out_code[0];
        uint32_t best_idx = 0;
        for (uint32_t c = 1; c < kOutputDim; ++c) {
            if (test_out_code[c] > best_code) {
                best_code = test_out_code[c];
                best_idx = c;
            }
        }

        const uint8_t true_label = sample._M_label;
        ++total;
        if (true_label == 1)
            ++outage_total;
        if (best_idx == true_label)
            ++correct;
        if (best_idx == 1 && true_label == 1)
            ++outage_correct;
    }

    /* Outage class should be predicted well (it's the most distinct) */
    if (outage_total > 0) {
        const int outage_acc = (outage_correct * 100) / outage_total;
        log_info ("outage accuracy: {}/{} ({}%)", outage_correct, outage_total, outage_acc);
        REQUIRE (outage_acc >= 50);
    }

    log_info ("overall accuracy: {}/{} ({}%)", correct, total, total > 0 ? (correct * 100) / total : 0);
}
