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

#include "mllee/quant.hpp"
#include "mllee/status.hpp"
#include "mllee/train_fixed.hpp"

#include <cstdint>

/*
 * Training SIMD primitives.
 *
 * Same build-time selection rule as mllee/simd.hpp: the assembly is chosen by a
 * compile define, never by CPU probing, and every primitive has a portable C++
 * fallback that produces byte-identical results.  A caller therefore gets the
 * same numbers from the asm and the no-asm build, which is what lets the host
 * test suite cover semantics that the QEMU model cannot execute.
 *
 * Conventions shared by the fallback and both assembly targets:
 *
 *   - Q15 lanes are int16 in memory and int32/int64 inside a kernel;
 *   - every reduction saturates to the stated width instead of wrapping, and
 *     reports nothing but its value (a saturating sum is enough for a norm, and
 *     the caller watches gradient saturation separately);
 *   - round-half-up shifts, i.e. (value + (1 << (shift - 1))) >> shift applied
 *     to an int64 intermediate, never a floating multiply;
 *   - the code step is 258 Q15 units (32768 / 127), so a gradient-to-code
 *     conversion divides by 258 with rounding;
 *   - alignment is tested once per call, never per element, and a stream that is
 *     not lane-aligned takes the scalar body for its whole length.
 */
#if defined(ZZS_MLLEE_ASM_X86_64) || defined(ZZS_MLLEE_ASM_CORTEX_M4)
#define ZZS_MLLEE_HAS_TRAIN_ASM 1
extern "C"
{
    /** @brief Sum of lane products, Q15 result, saturating (SSE2 pmaddwd / M4 SMLAD). */
    int32_t zzs_mllee_train_dot_q15 (const int16_t *lhs, const int16_t *rhs,
                                     uint32_t count);
    /** @brief Saturating Q15 lane add; returns the number of lanes that clamped. */
    uint32_t zzs_mllee_train_add_grad_q15 (int16_t *destination,
                                          const int16_t *source, uint32_t count);
    /** @brief destination += scalar * source, Q15, saturating. */
    void zzs_mllee_train_axpy_q15 (int16_t *destination, const int16_t *source,
                                   int32_t scalar_q15, uint32_t count);
    /** @brief values *= scalar, Q15, saturating (momentum and weight decay). */
    void zzs_mllee_train_scale_q15 (int16_t *values, int32_t scalar_q15,
                                    uint32_t count);
    /** @brief param += round(scalar_q15 * grad / 258) in code units, [-127, 127]. */
    void zzs_mllee_train_sgd_step_i8 (int8_t *parameters, const int16_t *grad,
                                      int32_t lr_q15, uint32_t count);
    /** @brief grad_in = grad_out where input >= 0, else 0 (relu/relu6 backward). */
    void zzs_mllee_train_relu_bwd (const int16_t *grad_out, const int8_t *input,
                                   int16_t *grad_in, uint32_t count);
    /** @brief Sum of |values| in Q15, int32 result, saturating. */
    int32_t zzs_mllee_train_sum_abs_q15 (const int16_t *values, uint32_t count);
    /** @brief Sum of squares in Q15, int32 result, saturating. */
    int32_t zzs_mllee_train_sum_sq_q15 (const int16_t *values, uint32_t count);
    /** @brief Bulk int8 code -> Q15 lane widening (x 258). */
    void zzs_mllee_train_codes_to_q15 (const int8_t *codes, int16_t *values,
                                       uint32_t count);
    /** @brief Bulk Q15 lane -> int8 code with rounding and saturation. */
    void zzs_mllee_train_q15_to_codes (const int16_t *values, int8_t *codes,
                                       uint32_t count);
}
#endif

namespace zzs::mllee::train_simd
{
/** @brief The code step in Q15 units, shared by the code/gradient conversions. */
inline constexpr int32_t _S_code_step_q15 = 258;

/** @brief Sum of lane products with a Q15 result, saturated to int32. */
[[nodiscard]] inline int32_t
dot_q15 (const int16_t *const lhs, const int16_t *const rhs,
         const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_TRAIN_ASM)
    return zzs_mllee_train_dot_q15 (lhs, rhs, count);
#else
    int64_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += static_cast<int64_t> (lhs[i]) * rhs[i];
    const int64_t shifted = (total + (1LL << 14)) >> 15;
    if (shifted > 2147483647LL)
        return 2147483647;
    if (shifted < -2147483648LL)
        return -2147483647 - 1;
    return static_cast<int32_t> (shifted);
#endif
}

/**
 * @brief Accumulate one gradient buffer into another with Q15 saturation.
 *
 * @return the number of lanes that clamped; the trainer feeds that count to the
 *         loss-scale backoff, so a saturated gradient is never invisible.
 */
[[nodiscard]] inline uint32_t
add_grad (int16_t *const destination, const int16_t *const source,
          const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_TRAIN_ASM)
    return zzs_mllee_train_add_grad_q15 (destination, source, count);
#else
    uint32_t saturated = 0u;
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t sum = destination[i] + source[i];
            if (sum > 32767 || sum < -32768)
                {
                    destination[i] = sum > 32767 ? 32767 : -32768;
                    ++saturated;
                }
            else
                destination[i] = static_cast<int16_t> (sum);
        }
    return saturated;
#endif
}

/** @brief destination += scalar_q15 * source, rounded and saturating. */
inline void
axpy_q15 (int16_t *const destination, const int16_t *const source,
          const int32_t scalar_q15, const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_TRAIN_ASM)
    zzs_mllee_train_axpy_q15 (destination, source, scalar_q15, count);
#else
    for (uint32_t i = 0; i < count; ++i)
        {
            const int64_t product = static_cast<int64_t> (source[i]) * scalar_q15;
            const int32_t term = static_cast<int32_t> ((product + (1LL << 14)) >> 15);
            const int32_t sum = destination[i] + term;
            destination[i] = sum > 32767 ? 32767 : (sum < -32768 ? -32768
                                                                : static_cast<int16_t> (sum));
        }
#endif
}

/** @brief values *= scalar_q15 in place, rounded and saturating. */
inline void
scale_q15 (int16_t *const values, const int32_t scalar_q15,
           const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_TRAIN_ASM)
    zzs_mllee_train_scale_q15 (values, scalar_q15, count);
#else
    for (uint32_t i = 0; i < count; ++i)
        {
            const int64_t product = static_cast<int64_t> (values[i]) * scalar_q15;
            const int64_t shifted = (product + (1LL << 14)) >> 15;
            values[i] = static_cast<int16_t> (
                shifted > 32767 ? 32767 : (shifted < -32768 ? -32768 : shifted));
        }
#endif
}

/**
 * @brief The parameter update: one code step per gradient step.
 *
 * param[i] += round(lr_q15 * grad[i] / (32768 * 258) * 32768) expressed in code
 * units, i.e. the Q15 product lr*grad is narrowed to a code step by dividing by
 * 258, then saturated into [-127, 127] because mllee never emits the -128 code.
 */
inline void
sgd_step_i8 (int8_t *const parameters, const int16_t *const grad,
             const int32_t lr_q15, const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_TRAIN_ASM)
    zzs_mllee_train_sgd_step_i8 (parameters, grad, lr_q15, count);
#else
    for (uint32_t i = 0; i < count; ++i)
        {
            const int64_t product = static_cast<int64_t> (grad[i]) * lr_q15;
            const int64_t code_step = (product + (1LL << 14)) >> 15;
            const int32_t updated = static_cast<int32_t> (parameters[i])
                                    + static_cast<int32_t> (
                                        (code_step >= 0 ? code_step + 129 : code_step - 129) / 258);
            parameters[i] = _S_i8_code (updated);
        }
#endif
}

/** @brief ReLU backward: route the incoming gradient by the forward sign. */
inline void
relu_backward (const int16_t *const grad_out, const int8_t *const input,
               int16_t *const grad_in, const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_TRAIN_ASM)
    zzs_mllee_train_relu_bwd (grad_out, input, grad_in, count);
#else
    for (uint32_t i = 0; i < count; ++i)
        grad_in[i] = input[i] >= 0 ? grad_out[i] : 0;
#endif
}

/** @brief L1 norm of a gradient buffer, saturating to int32. */
[[nodiscard]] inline int32_t
sum_abs_q15 (const int16_t *const values, const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_TRAIN_ASM)
    return zzs_mllee_train_sum_abs_q15 (values, count);
#else
    int64_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        total += values[i] < 0 ? -static_cast<int64_t> (values[i]) : values[i];
    return total > 2147483647LL ? 2147483647 : static_cast<int32_t> (total);
#endif
}

/** @brief Sum of squares of a gradient buffer, saturating to int32. */
[[nodiscard]] inline int32_t
sum_sq_q15 (const int16_t *const values, const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_TRAIN_ASM)
    return zzs_mllee_train_sum_sq_q15 (values, count);
#else
    int64_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
        {
            total += static_cast<int64_t> (values[i]) * values[i];
            if (total > 2147483647LL)
                return 2147483647;
        }
    return static_cast<int32_t> (total);
#endif
}

/** @brief Widen int8 codes into Q15 lanes (multiply by the code step). */
inline void
codes_to_q15 (const int8_t *const codes, int16_t *const values,
              const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_TRAIN_ASM)
    zzs_mllee_train_codes_to_q15 (codes, values, count);
#else
    for (uint32_t i = 0; i < count; ++i)
        values[i] = static_cast<int16_t> (codes[i] * _S_code_step_q15);
#endif
}

/** @brief Narrow Q15 lanes into int8 codes with rounding and saturation. */
inline void
q15_to_codes (const int16_t *const values, int8_t *const codes,
              const uint32_t count) noexcept
{
#if defined(ZZS_MLLEE_HAS_TRAIN_ASM)
    zzs_mllee_train_q15_to_codes (values, codes, count);
#else
    for (uint32_t i = 0; i < count; ++i)
        {
            const int32_t value = values[i];
            codes[i] = _S_i8_code (value >= 0 ? (value * 127 + 16384) >> 15
                                              : -(((-value) * 127 + 16384) >> 15));
        }
#endif
}
} // namespace zzs::mllee::train_simd
