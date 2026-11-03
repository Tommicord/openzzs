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

#include "kernel/kernel_config.hpp"
#include "kernel/kernel_feat_pack.hpp"
#include "kernel/kernel_feat_plhr.hpp"
#include "kernel/kernel_kernel_stage.hpp"
#include "kernel/kernel_meta_ids.hpp"
#include "kernel/kernel_sensor_accel.hpp"
#include "kernel/kernel_sensor_types.hpp"
#include "kernel/kernel_status.hpp"
#include "kernel/kernel_util_fixed.hpp"
#include "std/log.hpp"

#include <cstdint>

/**
 * @brief Geometric tilt-compensation stage for the magnetometer.
 *
 * Phone-held magnetometer readings rotate with the device: the same 50/60 Hz
 * harmonic lands on different axes depending on orientation.  PLHR bandpasses
 * at 45-250 Hz which removes the static geomagnetic DC, but the AC harmonic
 * energy is still split across axes in an orientation-dependent way.
 *
 * This stage removes the gravity-aligned component of the magnetometer vector
 * by projecting it onto the plane perpendicular to the estimated gravity
 * vector (from the accelerometer low-pass).  The result is the horizontal
 * magnetic component, which is orientation-stable: rotating the phone around
 * the vertical axis spins the horizontal vector but does not change its
 * magnitude or which axis carries the fundamental vs. harmonic energy.
 *
 * The projection is computed without any sin/cos/atan2/sqrt per sample:
 *
 *   g       = low-pass filtered accelerometer (gravity estimate, Q15)
 *   g_sq    = gx^2 + gy^2 + gz^2              (Q30)
 *   dot     = mx*gx + my*gy + mz*gz           (Q30)
 *   coeff   = dot / g_sq                      (Q15, dimensionless)
 *   mag_perp[i] = mx[i] - q15_mul(g[i], coeff)  (Q15)
 *
 * Because PLHR reports dB-relative harmonic power, the residual scale factor
 * from skipping the |g| normalization is a constant dB offset that the
 * classifier absorbs.  The division by g_sq uses mllee's magic reciprocal so
 * each mag triad costs one 32x32->64 multiply instead of a hardware divide.
 *
 * All arithmetic is integer-only; no float, no double.  Fusion wraps PLHR:
 * fusion_configure calls plhr_configure, fusion_reset calls plhr_reset, and
 * fusion_produce calls plhr_produce.  Register fusion (not PLHR) in the bank,
 * and pass PLHR's state pointer to fusion so samples flow accel->gravity->
 * project->plhr_feed->plhr_produce.
 *
 * Cost per accel sample: 3 q15_mul + 3 adds (gravity LPF) + g_sq recompute.
 * Cost per mag triad:    3 mul + 2 add (dot) + 1 magic-div (coeff) +
 *                        3 q15_mul + 3 sub (projection) + PLHR feed.
 * @{
 */
namespace zzs::kernel::feat {

namespace sm = zzs::kernel::sensor::mag;
namespace ac = zzs::kernel::sensor::accel;
using zzs::kernel::fx::clamp_q15;
using zzs::kernel::fx::q15_add_sat;
using zzs::kernel::fx::q15_mul;
using zzs::kernel::fx::q15_sub_sat;
using zzs::mllee::_S_div32_make;
using zzs::mllee::_S_div32_u;

/** @brief Low-pass coefficient applied to the accelerometer gravity estimate.
 *
 * Q15.  0.1 = 3277 gives a slow-moving gravity estimate that tracks phone
 * orientation changes but not motion.  Overridable at construction. */
inline constexpr int16_t _S_fusion_lpf_coeff_q15 = 3277; /* 0.1 in Q15 */

/** @brief Fusion stage state.  Wraps a PLHR state and feeds it compensated
 *         magnetometer samples. */
struct _S_FusionState {
    /* Gravity estimate from low-pass filtered accelerometer, Q15 per axis. */
    int16_t _M_gravity_q15[ac::COUNT];

    /* Squared magnitude of the gravity vector, Q30.  Refreshed on each accel
       sample so the mag path reuses it without recomputing. */
    int32_t _M_gravity_sq_q30;

    /* Buffered magnetometer triad, Q15.  Filled as MAG_X/Y/Z arrive. */
    int16_t _M_mag_buf[sm::COUNT];
    uint32_t _M_mag_count;

    /* PLHR state: fusion feeds compensated samples into it and delegates
       configure/reset/produce to it.  Set by the caller at construction. */
    _S_PlhrState *_M_plhr_state;

    /* Low-pass coefficient for the gravity filter, Q15. */
    int16_t _M_lpf_coeff_q15;

    bool _M_configured;
};

/** @brief Configure fusion and its wrapped PLHR stage. */
[[nodiscard]] inline zzs::kernel::Status
fusion_configure (void *const state) noexcept {
    const auto s = static_cast<_S_FusionState *> (state);
    if (s == nullptr || s->_M_plhr_state == nullptr)
        return zzs::kernel::Status::OUT_OF_RANGE;

    for (uint32_t a = 0; a < ac::COUNT; ++a)
        s->_M_gravity_q15[a] = 0;
    s->_M_gravity_sq_q30 = 0;
    s->_M_mag_count = 0;
    for (uint32_t a = 0; a < sm::COUNT; ++a)
        s->_M_mag_buf[a] = 0;
    if (s->_M_lpf_coeff_q15 == 0)
        s->_M_lpf_coeff_q15 = _S_fusion_lpf_coeff_q15;

    /* Delegate to PLHR configure so its filters/resonators are set up. */
    const zzs::kernel::Status st = plhr_configure (s->_M_plhr_state);
    if (st != zzs::kernel::Status::OK) {
        zzs::base::log_error ("fusion_configure: plhr_configure={}", zzs::kernel::status_name (st));
        return st;
    }

    s->_M_configured = true;
    zzs::base::log_debug ("fusion_configure: configured");
    return zzs::kernel::Status::OK;
}

/** @brief Reset fusion and its wrapped PLHR stage. */
inline void
fusion_reset (void *const state) noexcept {
    const auto s = static_cast<_S_FusionState *> (state);
    if (s == nullptr)
        return;
    for (uint32_t a = 0; a < ac::COUNT; ++a)
        s->_M_gravity_q15[a] = 0;
    s->_M_gravity_sq_q30 = 0;
    s->_M_mag_count = 0;
    for (uint32_t a = 0; a < sm::COUNT; ++a)
        s->_M_mag_buf[a] = 0;
    plhr_reset (s->_M_plhr_state);
}

/** @brief Feed one resampled Q15 sample.
 *
 * Accelerometer samples update the gravity estimate.  Magnetometer samples are
 * buffered; when the Z axis completes the triad the compensated vector is
 * computed and fed directly into the PLHR state.  Non-accel/mag channels are
 * ignored.
 */
[[nodiscard]] inline zzs::kernel::Status
fusion_feed (void *const state, const sensor::Channel channel, const int16_t sample_q15) noexcept {
    const auto s = static_cast<_S_FusionState *> (state);
    if (s == nullptr || !s->_M_configured)
        return zzs::kernel::Status::NOT_READY;

    /* g[i] += alpha * (raw[i] - g[i])   one q15_mul + one add per axis. ---- */
    const uint32_t accel_axis = static_cast<uint32_t> (channel) - static_cast<uint32_t> (sensor::Channel::ACCEL_X);
    if (accel_axis < ac::COUNT) {
        const int32_t err = static_cast<int32_t> (sample_q15) - s->_M_gravity_q15[accel_axis];
        const int32_t step = q15_mul (static_cast<int32_t> (s->_M_lpf_coeff_q15), err);
        const int32_t new_g = q15_add_sat (s->_M_gravity_q15[accel_axis], step);
        s->_M_gravity_q15[accel_axis] = static_cast<int16_t> (new_g);

        /* Recompute |g|^2 in Q30.  Each square is Q30; the sum stays Q30.
           int64 accumulation prevents overflow; Q30 full-scale fits int32. */
        int64_t g_sq = 0;
        for (uint32_t a = 0; a < ac::COUNT; ++a) {
            const int32_t g = s->_M_gravity_q15[a];
            g_sq += static_cast<int64_t> (g) * g;
        }
        s->_M_gravity_sq_q30 = static_cast<int32_t> (g_sq);
        return zzs::kernel::Status::OK;
    }

    /* The Z axis is the last axis of the triad (matching PLHR's tick convention). ---- */
    const uint32_t mag_axis = static_cast<uint32_t> (channel) - static_cast<uint32_t> (sensor::Channel::MAG_X);
    if (mag_axis < sm::COUNT) {
        s->_M_mag_buf[mag_axis] = sample_q15;
        ++s->_M_mag_count;

        if (mag_axis == static_cast<uint32_t> (sensor::Channel::MAG_Z) && s->_M_mag_count >= sm::COUNT) {
            /* dot = mx*gx + my*gy + mz*gz  (Q30) */
            int64_t dot = 0;
            for (uint32_t a = 0; a < sm::COUNT; ++a)
                dot += static_cast<int64_t> (s->_M_mag_buf[a]) * s->_M_gravity_q15[a];

            /* g_sq in Q30; guard against freefall / no gravity. */
            const int32_t g_sq = s->_M_gravity_sq_q30;
            int32_t coeff_q15 = 0;
            if (g_sq > 0) {
                /* coeff = dot / g_sq as Q15.
                   Reduce both to Q15 first to keep the magic-divide
                   numerator inside uint32: (dot>>15) << 15 / (g_sq>>15).
                   This loses ~15 fraction bits of the ratio, which is
                   irrelevant because PLHR reports dB-relative power. */
                const int32_t abs_dot = dot >= 0 ? static_cast<int32_t> (dot) : -static_cast<int32_t> (dot);
                const int32_t abs_dot_q15 = abs_dot >> 15;
                const int32_t g_sq_q15 = g_sq >> 15;
                const uint32_t divisor = g_sq_q15 == 0 ? 1u : static_cast<uint32_t> (g_sq_q15);
                const zzs::mllee::_S_Div32 recip = _S_div32_make (divisor);
                const uint32_t quot = _S_div32_u (static_cast<uint32_t> (abs_dot_q15 << 15), recip);
                coeff_q15 = dot >= 0 ? static_cast<int32_t> (quot) : -static_cast<int32_t> (quot);
                /* Clamp to [-1, +1] in Q15. */
                if (coeff_q15 > zzs::kernel::_S_q15_one)
                    coeff_q15 = zzs::kernel::_S_q15_one;
                else if (coeff_q15 < -zzs::kernel::_S_q15_one)
                    coeff_q15 = -zzs::kernel::_S_q15_one;
            }

            /* mag_perp[i] = mx[i] - q15_mul(g[i], coeff)  (Q15) */
            int16_t comp[sm::COUNT];
            for (uint32_t a = 0; a < sm::COUNT; ++a) {
                const int32_t removed = q15_mul (s->_M_gravity_q15[a], coeff_q15);
                const int32_t perp = q15_sub_sat (static_cast<int32_t> (s->_M_mag_buf[a]), removed);
                comp[a] = clamp_q15 (perp);
            }

            /* Feed the compensated triad into PLHR.  Order X, Y, Z so
               PLHR's window tick advances on the Z feed. */
            for (uint32_t a = 0; a < sm::COUNT; ++a) {
                const sensor::Channel ch =
                    static_cast<sensor::Channel> (static_cast<uint32_t> (sensor::Channel::MAG_X) + a);
                const zzs::kernel::Status st = plhr_feed (s->_M_plhr_state, ch, comp[a]);
                (void) st;
                (void) ch;
            }

            s->_M_mag_count = 0;
        }
        return zzs::kernel::Status::OK;
    }

    return zzs::kernel::Status::OK;
}

/** @brief Produce features: delegate to PLHR.  Returns PLHR's status so the
 *         bank knows whether a window completed. */
[[nodiscard]] inline zzs::kernel::Status
fusion_produce (void *const state, int32_t *const slots, const uint32_t slot_count) noexcept {
    _S_FusionState *const s = static_cast<_S_FusionState *> (state);
    if (s == nullptr || !s->_M_configured || s->_M_plhr_state == nullptr)
        return zzs::kernel::Status::NOT_READY;
    const zzs::kernel::Status st = plhr_produce (s->_M_plhr_state, slots, slot_count);
    if (st == zzs::kernel::Status::OK)
        zzs::base::log_debug ("fusion_produce: window complete (via plhr)");
    else if (st != zzs::kernel::Status::NOT_READY)
        zzs::base::log_error ("fusion_produce: plhr_produce={}", zzs::kernel::status_name (st));
    return st;
}

/** @brief Fusion stage vtable. */
[[nodiscard]] inline const _S_StageVtable &
fusion_vtable (void) noexcept {
    static const _S_StageVtable vt = {fusion_configure, fusion_reset, fusion_feed, fusion_produce};
    return vt;
}

/** @} */
} // namespace zzs::kernel::feat
