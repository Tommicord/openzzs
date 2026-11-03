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

#include "mllee/core_arith.hpp"
#include "mllee/core_status.hpp"

#include <cstdint>

/**
 * @brief Module-wide constants and build-time tier selection for the kernel.
 *
 * The kernel is the preprocessing bank between the sensor HAL and the mllee
 * inference core: it turns raw sensor events into the fixed-size feature vector
 * the model consumes.  Everything that several stages must agree on -- sample
 * rates, window lengths, mains frequencies, the feature count, the numeric
 * formats and the RAM budget -- lives here, because a mismatch between two of
 * those is a silent numerical bug rather than a compile error.
 *
 * @section Numeric formats
 *
 * The module is integer-only end to end.  Four widths are in play and every
 * other value in the kernel is one of them:
 *
 *   - `_S_sample_frac_bits` (Q15 int16): a normalized sensor sample.  One unit
 *     is 1/32768 of the channel's full scale; this is the wire format and the
 *     format a stage receives.
 *   - `_S_state_frac_bits` (Q15 int32): a filter state or an accumulator.  The
 *     extra 16 exponent bits are what lets a biquad hold a resonant pole without
 *     a per-sample renormalization.
 *   - `_S_resonator_frac_bits` (Q7 int16): a Goertzel resonator state.  A
 *     resonator grows linearly with the window length, so the state width, not
 *     the sample width, decides the input scale -- see `dsp_goertzel.hpp`.
 *   - `_S_coeff_frac_bits` (Q13/Q14 int16): a filter or resonator coefficient.
 *     `2*cos(theta)` peaks at 2.0, which is why the coefficient is not Q15.
 *
 * @section Budget
 *
 * The module owns `_S_ram_budget_bytes` of writable RAM on a 32-bit target.
 * That is the resident `PreprocessingBank` object plus the deepest call chain
 * it can build; the gate in tools/check_kernel_ram.sh measures both and the
 * static_assert in kernel_bank.hpp refuses a build that overflows the object's
 * share.  Nothing here allocates, and no header declares a writable static.
 */
namespace zzs::kernel {
/**
 * @name SIMD tiers
 *
 * Selected at build time from the compiler's own target macros, never probed at
 * run time: the hot loops are fixed-length and the point of a preprocessing
 * bank on a Cortex-M is that its instruction count per sample does not depend
 * on what CPU it happens to be running on.  Each tier is guarded so a build can
 * force the portable path with `-DZZS_KERNEL_SCALAR=1` and diff the results;
 * the portable path and every tier compute the same int32 intermediate, which
 * is what makes that diff meaningful.
 *
 * The tiers are mutually exclusive and ordered by width for the int16 lane
 * kernels (8 lanes at SSE2, 16 at AVX2, 32 at AVX-512BW, 8 at NEON) and for the
 * widening int16x16 -> int32 product, which is the primitive the whole module
 * is built on.
 * @{
 */
#if !defined(ZZS_KERNEL_SCALAR)

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#define ZZS_KERNEL_NEON 1
#else
#define ZZS_KERNEL_NEON 0
#endif

#if defined(__x86_64__) || defined(__i386__)
#if defined(__AVX512BW__) && defined(__AVX512F__)
#define ZZS_KERNEL_AVX512 1
#else
#define ZZS_KERNEL_AVX512 0
#endif
#if defined(__AVX2__)
#define ZZS_KERNEL_AVX2 1
#else
#define ZZS_KERNEL_AVX2 0
#endif
#if defined(__SSSE3__)
#define ZZS_KERNEL_SSSE3 1
#else
#define ZZS_KERNEL_SSSE3 0
#endif
#if defined(__SSE4_1__)
#define ZZS_KERNEL_SSE41 1
#else
#define ZZS_KERNEL_SSE41 0
#endif
#if defined(__SSE2__)
#define ZZS_KERNEL_SSE2 1
#else
#define ZZS_KERNEL_SSE2 0
#endif
#else
#define ZZS_KERNEL_AVX512 0
#define ZZS_KERNEL_AVX2 0
#define ZZS_KERNEL_SSSE3 0
#define ZZS_KERNEL_SSE41 0
#define ZZS_KERNEL_SSE2 0
#endif

} // namespace zzs::kernel
#if ZZS_KERNEL_NEON
#include <arm_neon.h>
#endif
#if ZZS_KERNEL_SSE2 || ZZS_KERNEL_AVX2 || ZZS_KERNEL_AVX512
#include <immintrin.h>
#endif
namespace zzs::kernel {

#else /* ZZS_KERNEL_SCALAR */
#define ZZS_KERNEL_NEON 0
#define ZZS_KERNEL_AVX512 0
#define ZZS_KERNEL_AVX2 0
#define ZZS_KERNEL_SSSE3 0
#define ZZS_KERNEL_SSE41 0
#define ZZS_KERNEL_SSE2 0
#endif

/** @brief int16 lanes per vector for the lane kernels that exist on this target. */
#if ZZS_KERNEL_AVX512
inline constexpr uint32_t _S_simd_lanes = 32u;
#elif ZZS_KERNEL_AVX2
inline constexpr uint32_t _S_simd_lanes = 16u;
#elif ZZS_KERNEL_NEON
inline constexpr uint32_t _S_simd_lanes = 8u;
#elif ZZS_KERNEL_SSE2
inline constexpr uint32_t _S_simd_lanes = 8u;
#else
inline constexpr uint32_t _S_simd_lanes = 1u;
#endif

/** @brief True when the target has a vector widening int16x16 -> int32 multiply. */
#if ZZS_KERNEL_NEON || ZZS_KERNEL_SSE2 || ZZS_KERNEL_AVX2 || ZZS_KERNEL_AVX512
inline constexpr bool _S_has_widening_mac = true;
#else
inline constexpr bool _S_has_widening_mac = false;
#endif
/** @} */

/**
 * @name Calling convention
 *
 * The bank is called once per grid tick, so a nested helper that materializes a
 * lane buffer costs a frame on the chain the budget counts.  The DSP hot loops
 * are forced inline (they have no local state worth a frame); the stage entry
 * points are kept out of line so the call chain stays flat and the stack gate
 * in tools/check_kernel_ram.sh can attribute a frame to each stage.
 * @{
 */
#if defined(__GNUC__) || defined(__clang__)
#define ZZS_KERNEL_NOINLINE __attribute__ ((noinline))
#define ZZS_KERNEL_INLINE __attribute__ ((always_inline)) inline
#else
#define ZZS_KERNEL_NOINLINE
#define ZZS_KERNEL_INLINE inline
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ZZS_KERNEL_RESTRICT __restrict__
#else
#define ZZS_KERNEL_RESTRICT
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ZZS_KERNEL_ASSUME_ALIGNED(pointer, bytes) __builtin_assume_aligned ((pointer), (bytes))
#else
#define ZZS_KERNEL_ASSUME_ALIGNED(pointer, bytes) (pointer)
#endif
/** @} */

/**
 * @name Sensor rates and the common grid
 *
 * Every channel is resampled to `_S_grid_rate_hz` before any feature is
 * computed, which is what lets one window index address all six channels.  The
 * rates are the hardware maxima the HAL table documents; a channel that arrives
 * slower is interpolated up, a channel that arrives faster is decimated down,
 * and both cases are the resampler's job rather than the stage's.
 * @{
 */
inline constexpr uint32_t _S_grid_rate_hz = 100u;
inline constexpr uint32_t _S_mag_rate_hz = 200u;
inline constexpr uint32_t _S_accel_rate_hz = 200u;
inline constexpr uint32_t _S_gyro_rate_hz = 200u;
inline constexpr uint32_t _S_light_rate_hz = 1u;
inline constexpr uint32_t _S_battery_rate_hz = 1u;
inline constexpr uint32_t _S_sound_rate_hz = 16000u;

/** @brief Envelope rate: the 16 kHz audio path is block-RMS decimated to this. */
inline constexpr uint32_t _S_envelope_rate_hz = 200u;

/** @brief Audio block size: 16000 / 200, so one envelope sample per block. */
inline constexpr uint32_t _S_envelope_block = 80u;

/** @brief Nanoseconds per grid tick, the denominator every stage shares. */
inline constexpr uint64_t _S_grid_period_ns = 1000000000ull / _S_grid_rate_hz;
/** @} */

/**
 * @name Window lengths
 *
 * Counts of grid-rate samples, not seconds: a stage that wants a 1 s window
 * takes `_S_grid_rate_hz` samples, and a stage that takes `_S_plhr_samples` at
 * the mag rate is documented as such.  All four are power of two apart from the
 * 200-sample windows, which is why the DFT is optional and the Goertzel bank,
 * which works for any length, is the default spectral path.
 * @{
 */
inline constexpr uint32_t _S_plhr_samples = 200u;
inline constexpr uint32_t _S_vibration_samples = 200u;
inline constexpr uint32_t _S_flicker_samples = 100u;
inline constexpr uint32_t _S_acoustic_samples = 200u;

/** @brief Longest resonator window any stage uses; the overflow bound is stated against it. */
inline constexpr uint32_t _S_resonator_max_samples = 256u;
/** @} */

/**
 * @name Mains grid
 *
 * The deployment region is not known at build time, so the mains frequency is
 * detected at run time from the sensor data (Goertzel scan at 50/60 Hz in the
 * PLHR stage).  The detected frequency is stored in PLHR stage state
 * (_M_mains_freq_hz), not as a module-level static, to avoid writable static
 * storage (AGENTS.md §4.1).
 * @{
 */
inline constexpr uint32_t _S_harmonics = 4u;

/**
 * @brief Detect the mains grid frequency with fine resolution.
 *
 * Runs a Goertzel scan across a range around the nominal mains frequency
 * (50 Hz or 60 Hz grid) in fine steps, then uses parabolic interpolation on
 * the peak bin to estimate the sub-bin frequency.  Returns the estimated
 * frequency in millihertz (e.g. 60000 = 60.000 Hz).
 *
 * Integer-only: resonator coefficients come from the fixed-point
 * trigonometry table; no float, no heap, no exceptions.
 *
 * @param samples          window of Q15 int16 samples
 * @param count            number of samples
 * @param sample_rate_hz   sample rate of the samples
 * @param nominal_hz       nominal mains frequency (50 or 60) to scan around
 * @returns Estimated mains frequency in millihertz, or nominal_hz * 1000 on
 *          invalid input.
 */
[[nodiscard]] inline uint32_t
detect_mains_freq_fine (const int16_t *const samples, const uint32_t count, const uint32_t sample_rate_hz,
                        const uint32_t nominal_hz) noexcept;

/** @brief PLHR passband: above the 50/60 Hz fundamental, below the mag alias. */
inline constexpr uint32_t _S_plhr_band_low_hz = 45u;
inline constexpr uint32_t _S_plhr_band_high_hz = 250u;

/** @brief Vibration passband for the accelerometer. */
inline constexpr uint32_t _S_vibration_band_low_hz = 10u;
inline constexpr uint32_t _S_vibration_band_high_hz = 100u;

/** @brief Acoustic hum band: the 2x mains line and its first sideband. */
inline constexpr uint32_t _S_acoustic_hum_low_hz = 100u;
inline constexpr uint32_t _S_acoustic_hum_high_hz = 240u;
/** @} */

/**
 * @name Feature vector
 *
 * `_S_feature_count` is the model's input width, and `_S_seq_len` is the number
 * of overlapping windows the sequence builder hands the Conv1D layer, so 5 x 1 s
 * windows give the model 5 s of context beyond its own window.  A feature index
 * is part of the exported blob's metadata, so appending a feature is allowed
 * only at the end -- the id table in feat_ids.hpp is the single place that
 * order is written down.
 * @{
 */
inline constexpr uint32_t _S_feature_count = 38u;
inline constexpr uint32_t _S_seq_len = 5u;

/** @brief Bytes of one packed feature row when the codes are int8. */
inline constexpr uint32_t _S_feature_row_bytes = _S_feature_count;

/** @brief Bytes of the sequence buffer the model input needs. */
inline constexpr uint32_t _S_sequence_bytes = _S_seq_len * _S_feature_count;
/** @} */

/**
 * @name Packet layer
 *
 * The wire format is a fixed 8-byte header plus a payload plus a CRC-16, with
 * byte stuffing only inside the payload.  The limits bound the receive state
 * machine's worst case, which is what the budget gate charges; `_S_packet_max_payload`
 * is sized so one packet carries exactly one 200-sample grid tick of the widest
 * channel without fragmentation.
 * @{
 */
inline constexpr uint32_t _S_packet_header_bytes = 8u;
inline constexpr uint32_t _S_packet_crc_bytes = 2u;
inline constexpr uint32_t _S_packet_max_payload = 128u;

/** @brief Worst case bytes between two packet boundaries, including stuffing. */
inline constexpr uint32_t _S_packet_max_frame_bytes =
    (_S_packet_header_bytes + _S_packet_max_payload + _S_packet_crc_bytes) * 2u;

/** @brief Receive queue depth in packets; a power of two so the mask is free. */
inline constexpr uint32_t _S_packet_queue_capacity = 8u;
/** @} */

/**
 * @name RAM budget
 *
 * `_S_ram_budget_bytes` is the module's whole writable-RAM claim: the resident
 * bank object plus the deepest stage chain.  The object's share is the
 * remainder, and the host builds get the same number plus the pointer-width
 * slack mllee uses, so a 64-bit host can never pass a gate a 32-bit target
 * fails.
 * @{
 */
inline constexpr uint32_t _S_ram_budget_bytes = 2048u;
inline constexpr uint32_t _S_stack_budget_bytes = 384u;
inline constexpr uint32_t _S_bank_budget_bytes = _S_ram_budget_bytes - _S_stack_budget_bytes;
inline constexpr uint32_t _S_bank_budget_host_bytes = _S_bank_budget_bytes + 16u;
/** @} */

/**
 * @name Small compile-time helpers
 *
 * A capacity that is not a power of two silently turns every `& mask` wrap into
 * a modulo, which on a Cortex-M4 is a subroutine call in the innermost loop.
 * These are the two checks the module's own declarations use.
 * @{
 */
[[nodiscard]] inline constexpr bool
_S_is_pow2 (const uint32_t value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

[[nodiscard]] inline constexpr uint32_t
_S_align_up (const uint32_t value, const uint32_t alignment) noexcept {
    return (value + alignment - 1u) & ~(alignment - 1u);
}
/** @} */
} // namespace zzs::kernel
