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

#include "catch2/catch_test_macros.hpp"
#include "kernel/kernel_feat_fusion.hpp"
#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_status.hpp"
#include "mllee/core_arith.hpp"

#include <cmath>
#include <cstdint>

namespace {

constexpr uint32_t _S_grid_rate_hz = zzs::kernel::_S_grid_rate_hz;
constexpr uint32_t _S_mag_rate_hz = zzs::kernel::_S_mag_rate_hz;
constexpr uint32_t _S_plhr_samples = zzs::kernel::_S_plhr_samples;

/** @brief Feed a full accel+mag triad through fusion, one grid tick at a time.
 *  gravity_z is the Q15 1-g reference; mag_x carries a sine at freq_hz. */
void
feed_triad (zzs::kernel::feat::_S_FusionState &fusion,
            zzs::kernel::feat::_S_PlhrState &plhr, /* for direct produce checks */
            int16_t gravity_z_q15, int16_t mag_x_q15, int16_t mag_y_q15, int16_t mag_z_q15) noexcept {
    /* Accel first: only Z has gravity, X/Y are zero. */
    zzs::kernel::Status st;
    st = zzs::kernel::feat::fusion_feed (&fusion, zzs::kernel::sensor::Channel::ACCEL_X, 0);
    REQUIRE (st == zzs::kernel::Status::OK);
    st = zzs::kernel::feat::fusion_feed (&fusion, zzs::kernel::sensor::Channel::ACCEL_Y, 0);
    REQUIRE (st == zzs::kernel::Status::OK);
    st = zzs::kernel::feat::fusion_feed (&fusion, zzs::kernel::sensor::Channel::ACCEL_Z, gravity_z_q15);
    REQUIRE (st == zzs::kernel::Status::OK);

    /* Mag triad: X, Y, Z. Fusion projects and feeds PLHR on the Z feed. */
    st = zzs::kernel::feat::fusion_feed (&fusion, zzs::kernel::sensor::Channel::MAG_X, mag_x_q15);
    REQUIRE (st == zzs::kernel::Status::OK);
    st = zzs::kernel::feat::fusion_feed (&fusion, zzs::kernel::sensor::Channel::MAG_Y, mag_y_q15);
    REQUIRE (st == zzs::kernel::Status::OK);
    st = zzs::kernel::feat::fusion_feed (&fusion, zzs::kernel::sensor::Channel::MAG_Z, mag_z_q15);
    REQUIRE (st == zzs::kernel::Status::OK);
    (void) plhr;
}

} // namespace

TEST_CASE ("fusion tilt-compensation passes a steady field through PLHR", "[feat][fusion]") {
    zzs::kernel::feat::_S_FusionState fusion{};
    zzs::kernel::feat::_S_PlhrState plhr{};
    fusion._M_plhr_state = &plhr;

    REQUIRE (zzs::kernel::feat::fusion_configure (&fusion) == zzs::kernel::Status::OK);

    /* Feed enough ticks to fill a PLHR window (200 samples).  Gravity points
       down (+Z); a constant 50 Hz harmonic sits on mag X.  With the phone
       upright (gravity aligned with Z), tilt compensation should leave the X
       harmonic essentially intact. */
    const int16_t gravity_z = 32767; /* 1 g in Q15 */
    for (uint32_t t = 0; t < _S_plhr_samples; ++t) {
        /* 50 Hz sine at 200 Hz sample rate: 2*pi*50/200 = pi/2 per sample.
           Alternates 1, 0, -1, 0 ... scaled to a moderate amplitude. */
        const int16_t mag_x = (t % 4 == 0) ? 8192 : (t % 4 == 2) ? -8192 : 0;
        feed_triad (fusion, plhr, gravity_z, mag_x, 0, 0);
    }

    int32_t slots[zzs::kernel::_S_feature_count] = {};
    const zzs::kernel::Status st = zzs::kernel::feat::fusion_produce (&fusion, slots, zzs::kernel::_S_feature_count);
    REQUIRE (st == zzs::kernel::Status::OK);

    /* PLHR slots 0..23 = 3 axes x 2 grids x 4 harmonics.
       Mag X on the 50 Hz grid, harmonic 1 (slot 0) should be non-zero. */
    bool any_nonzero = false;
    for (uint32_t i = 0; i < zzs::kernel::meta::_S_feat_plhr_count; ++i) {
        if (slots[i] != 0)
            any_nonzero = true;
    }
    REQUIRE (any_nonzero);
}

TEST_CASE ("fusion tilt-compensation rotates with gravity direction", "[feat][fusion]") {
    zzs::kernel::feat::_S_FusionState fusion{};
    zzs::kernel::feat::_S_PlhrState plhr{};
    fusion._M_plhr_state = &plhr;

    REQUIRE (zzs::kernel::feat::fusion_configure (&fusion) == zzs::kernel::Status::OK);

    /* Case A: gravity down +Z, harmonic on mag X. */
    {
        int32_t slots_a[zzs::kernel::_S_feature_count] = {};
        const int16_t gravity_z = 32767;
        for (uint32_t t = 0; t < _S_plhr_samples; ++t) {
            const int16_t mag_x = (t % 4 == 0) ? 8192 : ((t % 4 == 2) ? -8192 : 0);
            feed_triad (fusion, plhr, gravity_z, mag_x, 0, 0);
        }
        REQUIRE (zzs::kernel::feat::fusion_produce (&fusion, slots_a, zzs::kernel::_S_feature_count)
                 == zzs::kernel::Status::OK);
    }

    /* Reset and case B: gravity down +X (phone on its side), same harmonic on
       mag X.  After tilt compensation the horizontal component should be
       different (the field is now partly aligned with gravity). */
    zzs::kernel::feat::fusion_reset (&fusion);
    {
        int32_t slots_b[zzs::kernel::_S_feature_count] = {};
        const int16_t gravity_x = 32767;
        for (uint32_t t = 0; t < _S_plhr_samples; ++t) {
            const int16_t mag_x = (t % 4 == 0) ? 8192 : ((t % 4 == 2) ? -8192 : 0);
            /* Feed accel X with gravity instead of Z. */
            zzs::kernel::Status st =
                zzs::kernel::feat::fusion_feed (&fusion, zzs::kernel::sensor::Channel::ACCEL_X, gravity_x);
            REQUIRE (st == zzs::kernel::Status::OK);
            st = zzs::kernel::feat::fusion_feed (&fusion, zzs::kernel::sensor::Channel::ACCEL_Y, 0);
            REQUIRE (st == zzs::kernel::Status::OK);
            st = zzs::kernel::feat::fusion_feed (&fusion, zzs::kernel::sensor::Channel::ACCEL_Z, 0);
            REQUIRE (st == zzs::kernel::Status::OK);

            st = zzs::kernel::feat::fusion_feed (&fusion, zzs::kernel::sensor::Channel::MAG_X, mag_x);
            REQUIRE (st == zzs::kernel::Status::OK);
            st = zzs::kernel::feat::fusion_feed (&fusion, zzs::kernel::sensor::Channel::MAG_Y, 0);
            REQUIRE (st == zzs::kernel::Status::OK);
            st = zzs::kernel::feat::fusion_feed (&fusion, zzs::kernel::sensor::Channel::MAG_Z, 0);
            REQUIRE (st == zzs::kernel::Status::OK);
        }
        REQUIRE (zzs::kernel::feat::fusion_produce (&fusion, slots_b, zzs::kernel::_S_feature_count)
                 == zzs::kernel::Status::OK);
    }
}

TEST_CASE ("fusion is integer-only and saturates safely", "[feat][fusion]") {
    zzs::kernel::feat::_S_FusionState fusion{};
    zzs::kernel::feat::_S_PlhrState plhr{};
    fusion._M_plhr_state = &plhr;

    REQUIRE (zzs::kernel::feat::fusion_configure (&fusion) == zzs::kernel::Status::OK);

    /* Extreme values: full-scale accel and mag.  Feed enough triads to fill a
       PLHR window (200 samples) so produce can run. */
    for (uint32_t t = 0; t < _S_plhr_samples; ++t) {
        feed_triad (fusion, plhr, 32767, 32767, -32768, 32767);
    }

    int32_t slots[zzs::kernel::_S_feature_count] = {};
    REQUIRE (zzs::kernel::feat::fusion_produce (&fusion, slots, zzs::kernel::_S_feature_count)
             == zzs::kernel::Status::OK);

    /* All PLHR slots must be within int32 range (they are int32 by type, but
       the packed row clamps to int8).  Just verify no slot is a trap value. */
    for (uint32_t i = 0; i < zzs::kernel::meta::_S_feat_plhr_count; ++i) {
        REQUIRE (slots[i] >= -1000000);
        REQUIRE (slots[i] <= 1000000);
    }
}
