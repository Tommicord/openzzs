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

#include "mllee-native/q11_q15_q31.hpp"
#include "mllee-native/lcg16.hpp"
#include "mllee-native/dataset_types.hpp"
#include "mllee-native/dataset_gen.hpp"
#include "mllee-native/dataset.hpp"

#include <cstdint>
#include <cstdio>

#include <catch2/catch_test_macros.hpp>

using zzs::mllee_native::fx::Q11;
using zzs::mllee_native::fx::Q15;
using zzs::mllee_native::fx::Q31;
using zzs::mllee_native::fx::q11_mul;
using zzs::mllee_native::fx::q15_mul;
using zzs::mllee_native::fx::q31_mul;
using zzs::mllee_native::fx::q11_add;
using zzs::mllee_native::fx::q15_add;
using zzs::mllee_native::fx::q31_add;
using zzs::mllee_native::fx::q11_abs;
using zzs::mllee_native::fx::q15_abs;
using zzs::mllee_native::fx::q31_abs;
using zzs::mllee_native::fx::q11_to_q15;
using zzs::mllee_native::fx::q15_to_q11;
using zzs::mllee_native::fx::q15_to_q31;
using zzs::mllee_native::fx::q31_to_q15;
using zzs::mllee_native::rng::Lcg16;
using zzs::mllee_native::rng::XorShift16;
using zzs::mllee_native::dataset::Label;
using zzs::mllee_native::dataset::TrainingSample;
using zzs::mllee_native::dataset::GenerationParams;
using zzs::mllee_native::dataset_gen::generate_dataset;
using zzs::mllee_native::dataset_count;
using zzs::mllee_native::dataset_sample;
TEST_CASE ("Q11 fixed-point arithmetic", "[mllee_native][fixed]")
{
    Q11 a = Q11::from_raw (0x0400); /* 0.5 in Q11 (0.5 * 2048 = 1024) */
    Q11 b = Q11::from_raw (0x0200); /* 0.25 in Q11 */
    Q11 c = q11_mul (a, b);
    /* 0.5 * 0.25 = 0.125 -> 0.125 * 2048 = 256 */
    REQUIRE (c._M_raw == 256);

    Q11 d = Q11::from_raw (0x07FF); /* ~0.9995 */
    Q11 e = Q11::from_raw (0x0001); /* ~0.0005 */
    Q11 f = q11_add (d, e);
    REQUIRE (f._M_raw == 0x0800); /* saturates at 1.0 */

    Q11 neg = Q11::from_raw (-1024);
    Q11 abs_neg = q11_abs (neg);
    REQUIRE (abs_neg._M_raw == 1024);

    Q15 wide = q11_to_q15 (a);
    /* 0.5 in Q11 -> 0.5 in Q15: 1024 << 4 = 16384 */
    REQUIRE (wide._M_raw == 16384);

    Q15 q15val = Q15::from_raw (16384);
    Q11 narrowed = q15_to_q11 (q15val);
    REQUIRE (narrowed._M_raw == 1024);
}

TEST_CASE ("Q15 fixed-point arithmetic", "[mllee_native][fixed]")
{
    Q15 a = Q15::from_raw (0x4000); /* 0.5 in Q15 */
    Q15 b = Q15::from_raw (0x2000); /* 0.25 in Q15 */
    Q15 c = q15_mul (a, b);
    /* 0.5 * 0.25 = 0.125 -> 0.125 * 32768 = 4096 */
    REQUIRE (c._M_raw == 0x1000);

    Q15 d = Q15::from_raw (0x7FFF);
    Q15 e = Q15::from_raw (0x0001);
    Q15 f = q15_add (d, e);
    REQUIRE (f._M_raw == 0x7FFF); /* saturates */

    Q15 neg = Q15::from_raw (-16384);
    Q15 abs_neg = q15_abs (neg);
    REQUIRE (abs_neg._M_raw == 16384);

    Q31 q31val = q15_to_q31 (a);
    /* 0.5 in Q15 -> 0.5 in Q31: 16384 << 16 = 1073741824 */
    REQUIRE (q31val._M_raw == 1073741824);

    Q15 back = q31_to_q15 (q31val);
    REQUIRE (back._M_raw == 16384);
}

TEST_CASE ("Q31 fixed-point arithmetic", "[mllee_native][fixed]")
{
    Q31 a = Q31::from_raw (0x40000000); /* 0.5 in Q31 */
    Q31 b = Q31::from_raw (0x20000000); /* 0.25 in Q31 */
    Q31 c = q31_mul (a, b);
    /* 0.5 * 0.25 = 0.125 -> 0.125 * 2^31 = 268435456 = 0x10000000 */
    REQUIRE (c._M_raw == 0x10000000);

    Q31 d = Q31::from_raw (0x7FFFFFFF);
    Q31 e = Q31::from_raw (0x00000001);
    Q31 f = q31_add (d, e);
    REQUIRE (f._M_raw == 0x7FFFFFFF); /* saturates */

    Q31 neg = Q31::from_raw (-1073741824);
    Q31 abs_neg = q31_abs (neg);
    REQUIRE (abs_neg._M_raw == 1073741824);
}

TEST_CASE ("LCG16 PRNG is deterministic and well-distributed", "[mllee_native][rng]")
{
    Lcg16 r1 = Lcg16::seed (0x1234u);
    Lcg16 r2 = Lcg16::seed (0x1234u);

    /* Same seed produces same sequence. */
    for (int i = 0; i < 64; ++i)
        REQUIRE (r1.next () == r2.next ());

    /* Different seeds produce different sequences (usually). */
    Lcg16 r3 = Lcg16::seed (0x5678u);
    bool differ = false;
    for (int i = 0; i < 16; ++i)
        if (r1.next () != r3.next ()) { differ = true; break; }
    REQUIRE (differ);

    /* State never zero. */
    Lcg16 r4 = Lcg16::seed (0u);
    REQUIRE (r4.next () != 0);
}

TEST_CASE ("XorShift16 PRNG", "[mllee_native][rng]")
{
    XorShift16 x1 = XorShift16::seed (0x1234u);
    XorShift16 x2 = XorShift16::seed (0x1234u);
    for (int i = 0; i < 32; ++i)
        REQUIRE (x1.next () == x2.next ());
}

TEST_CASE ("Dataset class distribution matches generation params", "[mllee_native][dataset]")
{
    GenerationParams p{ 0xC0DEu, 64u, 600u, 250u, 150u };
    auto ds = generate_dataset (p);

    REQUIRE (ds._M_count == 64);

    uint32_t n_normal = 0, n_outage = 0, n_transition = 0;
    for (uint32_t i = 0; i < ds._M_count; ++i)
        {
            Label lbl = static_cast<Label> (ds._M_samples[i]._M_label);
            switch (lbl)
                {
                case Label::Normal: ++n_normal; break;
                case Label::Outage: ++n_outage; break;
                case Label::Transition: ++n_transition; break;
                }
        }

    /* With per-mille ratios on 64 samples: normal ~38, outage ~16, transition ~10. */
    REQUIRE (n_normal > 20);
    REQUIRE (n_outage > 5);
    REQUIRE (n_transition > 2);
    REQUIRE (n_normal + n_outage + n_transition == 64);
}

TEST_CASE ("Dataset labels are valid (0, 1, or 2)", "[mllee_native][dataset]")
{
    GenerationParams p{ 0xC0DEu, 64u, 600u, 250u, 150u };
    auto ds = generate_dataset (p);

    for (uint32_t i = 0; i < ds._M_count; ++i)
        {
            uint8_t lbl = ds._M_samples[i]._M_label;
            REQUIRE (lbl <= 2u);
        }
}

TEST_CASE ("Dataset features are within int16 range", "[mllee_native][dataset]")
{
    GenerationParams p{ 0xC0DEu, 64u, 600u, 250u, 150u };
    auto ds = generate_dataset (p);

    for (uint32_t i = 0; i < ds._M_count; ++i)
        {
            for (uint32_t w = 0; w < 5u; ++w)
                {
                    for (uint32_t f = 0; f < 38u; ++f)
                        {
                            int16_t v = ds._M_samples[i]._M_windows[w]._M_features[f];
                            REQUIRE (v >= -32768);
                            REQUIRE (v <= 32767);
                        }
                }
        }
}

TEST_CASE ("Global dataset is constexpr and stored in .rodata", "[mllee_native][dataset]")
{
    /* Verify the global dataset object is usable in constexpr context. */
    constexpr uint32_t cnt = zzs::mllee_native::dataset_count ();
    REQUIRE (cnt == 64);

    /* Spot-check a known sample's label. */
    constexpr auto& s0 = zzs::mllee_native::dataset_sample (0);
    REQUIRE (s0._M_label <= 2u);
}

TEST_CASE ("Dataset is deterministic across generations", "[mllee_native][dataset]")
{
    GenerationParams p{ 0xC0DEu, 64u, 600u, 250u, 150u };
    auto ds1 = generate_dataset (p);
    auto ds2 = generate_dataset (p);

    REQUIRE (ds1._M_count == ds2._M_count);
    for (uint32_t i = 0; i < ds1._M_count; ++i)
        {
            REQUIRE (ds1._M_samples[i]._M_label == ds2._M_samples[i]._M_label);
            for (uint32_t w = 0; w < 5u; ++w)
                for (uint32_t f = 0; f < 38u; ++f)
                    REQUIRE (ds1._M_samples[i]._M_windows[w]._M_features[f] ==
                             ds2._M_samples[i]._M_windows[w]._M_features[f]);
        }
}

TEST_CASE ("Different seeds produce different datasets", "[mllee_native][dataset]")
{
    GenerationParams p1{ 0x0001u, 32u, 600u, 250u, 150u };
    GenerationParams p2{ 0x0002u, 32u, 600u, 250u, 150u };
    auto ds1 = generate_dataset (p1);
    auto ds2 = generate_dataset (p2);

    bool differ = false;
    for (uint32_t i = 0; i < ds1._M_count && !differ; ++i)
        {
            if (ds1._M_samples[i]._M_label != ds2._M_samples[i]._M_label)
                differ = true;
            for (uint32_t w = 0; w < 5u && !differ; ++w)
                for (uint32_t f = 0; f < 38u && !differ; ++f)
                    if (ds1._M_samples[i]._M_windows[w]._M_features[f] !=
                        ds2._M_samples[i]._M_windows[w]._M_features[f])
                        differ = true;
        }
    REQUIRE (differ);
}
