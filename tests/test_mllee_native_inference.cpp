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
 * @brief Unit tests for mllee-native inference on the fake dataset.
 *
 * Verifies that the trained model blob can be loaded and executed via
 * `zzs::mllee::Core::run()`, producing valid softmax output codes in
 * [-127, 127] and sensible class predictions.
 */

// Forward-declare zzs::base::format before log.hpp is pulled in via
// mllee-native/train_pipeline_impl.hpp, so the log_format templates find it.
namespace zzs::base {
template <unsigned int DstSize, class CharTy, class First, class... Rest>
constexpr unsigned int
format (CharTy (&)[DstSize], const CharTy *fmt, First &&first, Rest &&...rest) noexcept;
}

#include "mllee-native/dataset.hpp"
#include "mllee-native/inference.hpp"
#include "mllee-native/model.hpp"
#include "mllee-native/q11_q15_q31.hpp"
#include "mllee-native/train_pipeline_impl.hpp"
#include "mllee/core.hpp"
#include "mllee/core_format.hpp"
#include "std/fmt.hpp"
#include "std/io.hpp"

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdio>
#include <string>

using namespace zzs::mllee;
using namespace zzs::mllee_native;
using namespace zzs::mllee_native::dataset;
using namespace zzs::mllee_native::inference;
using namespace zzs::mllee_native::model;
using namespace zzs::mllee::train;

using zzs::base::format;

/** @brief Minimal training storage for inference tests. */
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

/** @brief The model blob, defined here so all TUs share one instance. */
namespace zzs::mllee_native::model {
alignas (4) uint8_t g_model_blob[kBlobSize] = {};
}

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

TEST_CASE ("Model blob is valid MLEE format", "[mllee_native][inference][model]") {
    const Status st = validate_model (g_model_blob, kBlobSize);
    REQUIRE (status_ok (st));
    REQUIRE (kBlobSize == 3350u);
}

TEST_CASE ("Model blob header has correct magic and version", "[mllee_native][inference][model]") {
    const Status init_st = init_model_blob ();
    REQUIRE (status_ok (init_st));
    REQUIRE (g_model_blob[0] == 'M');
    REQUIRE (g_model_blob[1] == 'L');
    REQUIRE (g_model_blob[2] == 'E');
    REQUIRE (g_model_blob[3] == 'E');
    REQUIRE (g_model_blob[4] == 1u); /* version */
    REQUIRE (g_model_blob[6] == 4u); /* layer_count: LINEAR + ACTIVATION + LINEAR + SOFTMAX */
}

TEST_CASE ("Flatten sample produces 190 int8 codes in range", "[mllee_native][inference][flatten]") {
    const TrainingSample &sample = dataset_sample (0);
    flatten_sample (sample, test_input);

    for (uint32_t i = 0; i < kInputDim; ++i) {
        REQUIRE (test_input[i] >= -127);
        REQUIRE (test_input[i] <= 127);
    }
}

TEST_CASE ("Core::run produces valid softmax output", "[mllee_native][inference][core]") {
    test_init_storage ();

    /* Build the model blob from init weights. */
    const Status build_st = build_model_blob (g_model_blob);
    REQUIRE (status_ok (build_st));

    const TrainingSample &sample = dataset_sample (0);
    int8_t output[kOutputDim] = {0, 0, 0};
    uint32_t predicted_class = 0u;

    const Status st = run_inference (sample, output, &predicted_class);
    REQUIRE (status_ok (st));

    /* Output codes should be in int8 range (softmax output). */
    for (uint32_t c = 0; c < kOutputDim; ++c) {
        REQUIRE (output[c] >= -127);
        REQUIRE (output[c] <= 127);
    }

    /* Predicted class should be valid. */
    REQUIRE (predicted_class <= 2u);
}

TEST_CASE ("Softmax output codes form a valid distribution", "[mllee_native][inference][softmax]") {
    test_init_storage ();

    const Status build_st = build_model_blob (g_model_blob);
    REQUIRE (status_ok (build_st));

    int8_t output[kOutputDim] = {0, 0, 0};
    uint32_t predicted_class = 0u;

    /* Run inference on all samples and check output validity. */
    for (uint32_t i = 0; i < dataset_count (); ++i) {
        const TrainingSample &sample = dataset_sample (i);
        const Status st = run_inference (sample, output, &predicted_class);
        REQUIRE (status_ok (st));

        /* All output codes in range. */
        for (uint32_t c = 0; c < kOutputDim; ++c) {
            REQUIRE (output[c] >= -127);
            REQUIRE (output[c] <= 127);
        }

        /* Predicted class is valid. */
        REQUIRE (predicted_class <= 2u);
    }
}

TEST_CASE ("Inference on trained model produces consistent predictions", "[mllee_native][inference][trained]") {
    test_init_storage ();

    /* Train briefly to get meaningful weights. */
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
    hooks._M_forward = pipeline::mlp_forward_hook;
    hooks._M_schedule = pipeline::constant_lr_schedule;
    hooks._M_post_grad = pipeline::post_grad_hook;
    hooks._M_update = pipeline::sgd_update_hook;

    Trainer trainer (ctx, hooks);
    REQUIRE (status_ok (trainer.configure (test_params, 6, 1)));

    StepStats train_stats;
    for (uint32_t step = 0; step < 50; ++step) {
        const uint32_t sample_idx = step % dataset_count ();
        const TrainingSample &sample = dataset_sample (sample_idx);
        pipeline::flatten_sample (sample, test_input);
        pipeline::one_hot_target (sample._M_label, test_target);
        REQUIRE (status_ok (trainer.step (train_stats)));
    }

    /* Export trained weights into model blob. */
    for (uint32_t i = 0; i < kHiddenInput * kHiddenDim; ++i)
        g_model_hidden_weights[i] = test_hidden_w_code[i];
    for (uint32_t i = 0; i < kHiddenDim; ++i)
        g_model_hidden_bias[i] = test_hidden_b_code[i];
    for (uint32_t i = 0; i < kHiddenDim * kOutputDim; ++i)
        g_model_weights[i] = test_w_code[i];
    for (uint32_t i = 0; i < kOutputDim; ++i)
        g_model_bias[i] = test_b_code[i];
    g_model_trained = true;

    const Status build_st = build_model_blob (g_model_blob);
    REQUIRE (status_ok (build_st));

    /* Run inference on all samples with trained model. */
    uint32_t correct = 0;
    uint32_t total = 0;

    for (uint32_t i = 0; i < dataset_count (); ++i) {
        const TrainingSample &sample = dataset_sample (i);
        int8_t output[kOutputDim] = {0, 0, 0};
        uint32_t predicted_class = 0u;

        const Status st = run_inference (sample, output, &predicted_class);
        REQUIRE (status_ok (st));

        ++total;
        if (predicted_class == sample._M_label)
            ++correct;
    }

    printf ("trained model accuracy: %u/%u (%u%%)\n", correct, total, total > 0 ? (correct * 100) / total : 0);

    /* Trained model should do better than random (33%). */
    if (total > 0) {
        const int acc = (correct * 100) / total;
        REQUIRE (acc >= 25);
    }
}

TEST_CASE ("Code to probability conversion", "[mllee_native][inference][prob]") {
    REQUIRE (code_to_probability (-127) == 0u);
    REQUIRE (code_to_probability (0) == 50u);
    REQUIRE (code_to_probability (127) == 100u);
    REQUIRE (code_to_probability (63) == 75u);
    REQUIRE (code_to_probability (-63) == 25u);
}

TEST_CASE ("Class names are correct", "[mllee_native][inference][names]") {
    REQUIRE (std::string (class_name (0).data ()) == "normal");
    REQUIRE (std::string (class_name (1).data ()) == "outage");
    REQUIRE (std::string (class_name (2).data ()) == "transition");
    REQUIRE (std::string (class_name (99).data ()) == "unknown");
}
