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

/*
 * mllee Cortex-M4 (ARMv7E-M + DSP) reduction and saturating-add kernels.
 *
 * Leaf routines, no literal pools, no writable static, no calls.  Cortex-M has
 * no byte-wide SIMD add, so the reductions get their throughput from the
 * SXTB16 + SMLAD lane split while the element-wise saturating ops stay
 * byte-wise; SSAT is ARMv6.1 and is implemented on ARMv7E-M.
 *
 * The core's code range is [-127, 127]: -128 is never a valid int8 code, so
 * every saturating result of -128 is lifted to -127 exactly as the portable
 * C++ fallback does.
 */

.syntax unified
.thumb
.cpu cortex-m4
.section .text.mllee.m4.reduce

/*
 * int32_t zzs_mllee_sum_i8(const int8_t *values, uint32_t count)
 *
 * SMLAD against the lane constant 0x00010001 adds the two int16 lanes of a
 * sign-extended word, which is a two-element sum in one instruction.
 */
    .globl zzs_mllee_sum_i8
    .type zzs_mllee_sum_i8, %function
    .thumb_func
zzs_mllee_sum_i8:
    push {r4, r5}
    movs r4, #0                 /* running sum */
    cmp r1, #0
    beq .Lsum_out
    mov.w r5, #0x00010001        /* two int16 lanes of value 1 */
    tst r0, #3
    bne .Lsum_scalar
.Lsum_w4:
    cmp r1, #4
    blt .Lsum_scalar
    ldr r2, [r0], #4
    sxtb16 r3, r2
    sxtb16 r2, r2, ror #8
    smlad r4, r3, r5, r4
    smlad r4, r2, r5, r4
    subs r1, r1, #4
    bne .Lsum_w4
    b .Lsum_out
.Lsum_scalar:
    cbz r1, .Lsum_out
.Lsum_scalar_loop:
    ldrsb r2, [r0], #1
    adds r4, r4, r2
    subs r1, r1, #1
    bne .Lsum_scalar_loop
.Lsum_out:
    mov r0, r4
    pop {r4, r5}
    bx lr
    .size zzs_mllee_sum_i8, .-zzs_mllee_sum_i8

/*
 * int8_t zzs_mllee_max_i8(const int8_t *values, uint32_t count)
 *
 * Byte-wise signed maximum: ARMv7E-M has no UMAX8/SMAX8, and the SWAR
 * unsigned-max trick needs a bias flip per word plus a borrow cascade, which
 * costs more than the four-instruction loop below.
 */
    .globl zzs_mllee_max_i8
    .type zzs_mllee_max_i8, %function
    .thumb_func
zzs_mllee_max_i8:
    mvn r2, #127                /* -128, lower than any valid code */
    cbz r1, .Lmax_out
.Lmax_loop:
    ldrsb r3, [r0], #1
    cmp r3, r2
    it gt
    movgt r2, r3
    subs r1, r1, #1
    bne .Lmax_loop
.Lmax_out:
    mov r0, r2
    bx lr
    .size zzs_mllee_max_i8, .-zzs_mllee_max_i8

/*
 * void zzs_mllee_add_sat_i8(const int8_t *lhs, const int8_t *rhs,
 *                           int8_t *destination, uint32_t count)
 *
 * SSAT clamps to [-128, 127]; the -128 corner is lifted to -127 so the result
 * matches the core's [-127, 127] code range.
 */
    .globl zzs_mllee_add_sat_i8
    .type zzs_mllee_add_sat_i8, %function
    .thumb_func
zzs_mllee_add_sat_i8:
    push {r4, r5}
    cbz r3, .Ladd_out
.Ladd_loop:
    ldrsb r12, [r0], #1
    ldrsb r4, [r1], #1
    adds r12, r12, r4
    ssat r12, #8, r12
    cmn r12, #128
    bne .Ladd_store
    mvn r12, #126               /* -127 */
.Ladd_store:
    strb r12, [r2], #1
    subs r3, r3, #1
    bne .Ladd_loop
.Ladd_out:
    pop {r4, r5}
    bx lr
    .size zzs_mllee_add_sat_i8, .-zzs_mllee_add_sat_i8

/* @brief Saturating int8 subtraction with the same [-127, 127] clamp. */
    .globl zzs_mllee_sub_sat_i8
    .type zzs_mllee_sub_sat_i8, %function
    .thumb_func
zzs_mllee_sub_sat_i8:
    push {r4, r5}
    cbz r3, .Lsub_out
.Lsub_loop:
    ldrsb r12, [r0], #1
    ldrsb r4, [r1], #1
    subs r12, r12, r4
    ssat r12, #8, r12
    cmn r12, #128
    bne .Lsub_store
    mvn r12, #126               /* -127 */
.Lsub_store:
    strb r12, [r2], #1
    subs r3, r3, #1
    bne .Lsub_loop
.Lsub_out:
    pop {r4, r5}
    bx lr
    .size zzs_mllee_sub_sat_i8, .-zzs_mllee_sub_sat_i8

.section .note.GNU-stack,"",%progbits
