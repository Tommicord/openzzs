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
 * mllee Cortex-M4 (ARMv7E-M + DSP) training kernels, reduction and update group.
 *
 * Same frozen interface as mllee/train_simd.hpp's portable fallbacks, same
 * byte-for-byte results; the fallback is the specification and
 * tools/mllee_m4_selftest.cpp runs the two against each other on the QEMU
 * mps2-an385 model.
 *
 * These are reductions plus one element-wise update, so the accumulator has to
 * be wider than int32: dot() sums up to count products of two Q15 lanes, each up
 * to 2^30, and the documented semantics saturate only after the final
 * (total + 2^14) >> 15.  Every kernel therefore keeps a 64-bit accumulator in a
 * register pair and folds it down once, at the end.
 *
 * Why SMLAD is NOT used for the dot product, unlike the int8 kernels: SMLAD
 * accumulates both halfword products into one 32-bit addend, and the sum of two
 * int16 products can be exactly +2^31 (both lanes at -32768), which is the one
 * value that wraps.  SMULL/MUL per lane plus ADDS/ADCS never wraps, and the
 * dual-lane saving is two instructions out of eight, so the predictable form
 * wins.  What the hardware is used for is the multiply itself: MUL and SMULL are
 * single-cycle on Cortex-M4, which is what makes an int64 accumulator affordable,
 * and the element loop stays free of the ARMv6 media group this model faults on
 * (see mllee_train_q15_lanes_cortex_m4.S).
 *
 * The divide by 258 in the SGD step is the reciprocal ceil(2^22 / 258) = 16257.
 * It reproduces floor(x / 258) while x < 2^22 (258 * 16257 = 2^22 + 2, so the
 * error term is x / (258 * 2^22)) and its product with x stays inside 32 bits
 * while x < 264192, so the kernel tests x against 254 * 258 = 65532 and takes the
 * saturating path above that.  The saturating path is not an approximation: a
 * code step of at least 254 puts a [-127, 127] parameter on its rail regardless
 * of where it started, which is exactly what the fallback's division computes.
 *
 * Callee-saved registers are pushed and popped, and r15 is never an accumulator:
 * SMULL/SMMUL with RdHi = 15 is UNDEFINED on ARMv7-M, so every 64-bit product
 * here owns two real register pairs.
 */

.syntax unified
.thumb
.cpu cortex-m4
.section .text.mllee.m4.trainmac

/*
 * int32_t zzs_mllee_train_dot_q15(const int16_t *lhs, const int16_t *rhs,
 *                                 uint32_t count)
 *
 * 64-bit accumulator, one rounded shift-right-15, then the int32 rails.  The
 * rails are reachable in principle (count * 2^15 exceeds 2^31 past about 65536
 * lanes) even though a real layer never gets that wide, so they are implemented
 * rather than assumed.
 */
    .globl zzs_mllee_train_dot_q15
    .type zzs_mllee_train_dot_q15, %function
    .thumb_func
zzs_mllee_train_dot_q15:
    push {r4, r5, r6, r7}
    movs r4, #0                 /* accumulator, low word */
    movs r5, #0                 /* accumulator, high word */
    cbz r2, .Ltdt_fold
.Ltdt_loop:
    ldrsh r6, [r0], #2
    ldrsh r7, [r1], #2
    mul r6, r6, r7              /* exact int32 product of two Q15 lanes */
    movs r7, r6, asr #31
    adds r4, r6
    adcs r5, r7
    subs r2, #1
    bne .Ltdt_loop
.Ltdt_fold:
    movw r6, #16384             /* 0x4000 is not an ADD-immediate, so it is loaded */
    adds r4, r6
    adcs r5, #0
    mov r0, r4, lsr #15
    orr r0, r0, r5, lsl #17     /* low 32 bits of (total + 2^14) >> 15 */
    asrs r6, r5, #15            /* its high 32 bits: total >> 47, sign-extended */
    movs r7, r0, asr #31
    cmp r6, r7
    bne .Ltdt_sat
    pop {r4, r5, r6, r7}
    bx lr
.Ltdt_sat:
    cmp r0, #0
    blt .Ltdt_min
    movw r0, #0xFFFF
    movt r0, #0x7FFF
    pop {r4, r5, r6, r7}
    bx lr
.Ltdt_min:
    movw r0, #0
    movt r0, #0x8000
    pop {r4, r5, r6, r7}
    bx lr
    .size zzs_mllee_train_dot_q15, .-zzs_mllee_train_dot_q15

/*
 * int32_t zzs_mllee_train_sum_abs_q15(const int16_t *values, uint32_t count)
 *
 * The absolute value is the ASR mask / EOR / SUB triple, so there is no branch
 * on the lane sign; the sum is non-negative, so a non-zero high word alone is
 * the overflow test and INT32_MAX is the only rail.
 */
    .globl zzs_mllee_train_sum_abs_q15
    .type zzs_mllee_train_sum_abs_q15, %function
    .thumb_func
zzs_mllee_train_sum_abs_q15:
    push {r4, r5, r6, r7}
    movs r4, #0
    movs r5, #0
    cbz r1, .Ltas_fold
.Ltas_loop:
    ldrsh r6, [r0], #2
    movs r7, r6, asr #31
    eors r6, r7
    subs r6, r7                 /* |value|, including the -32768 corner */
    adds r4, r6
    adcs r5, #0
    subs r1, #1
    bne .Ltas_loop
.Ltas_fold:
    mov r0, r4
    movw r6, #0xFFFF
    movt r6, #0x7FFF
    cmp r5, #0
    bne .Ltas_rail
    cmp r4, r6
    bhi .Ltas_rail
    pop {r4, r5, r6, r7}
    bx lr
.Ltas_rail:
    mov r0, r6
    pop {r4, r5, r6, r7}
    bx lr
    .size zzs_mllee_train_sum_abs_q15, .-zzs_mllee_train_sum_abs_q15

/*
 * int32_t zzs_mllee_train_sum_sq_q15(const int16_t *values, uint32_t count)
 *
 * A square is never negative and never above 2^30, so MUL's 32-bit result is the
 * whole product and the carry into the high word is the only extension needed.
 */
    .globl zzs_mllee_train_sum_sq_q15
    .type zzs_mllee_train_sum_sq_q15, %function
    .thumb_func
zzs_mllee_train_sum_sq_q15:
    push {r4, r5, r6, r7}
    movs r4, #0
    movs r5, #0
    cbz r1, .Ltss_fold
.Ltss_loop:
    ldrsh r6, [r0], #2
    mul r6, r6, r6
    adds r4, r6
    adcs r5, #0
    subs r1, #1
    bne .Ltss_loop
.Ltss_fold:
    mov r0, r4
    movw r6, #0xFFFF
    movt r6, #0x7FFF
    cmp r5, #0
    bne .Ltss_rail
    cmp r4, r6
    bhi .Ltss_rail
    pop {r4, r5, r6, r7}
    bx lr
.Ltss_rail:
    mov r0, r6
    pop {r4, r5, r6, r7}
    bx lr
    .size zzs_mllee_train_sum_sq_q15, .-zzs_mllee_train_sum_sq_q15

/*
 * void zzs_mllee_train_sgd_step_i8(int8_t *parameters, const int16_t *grad,
 *                                  int32_t lr_q15, uint32_t count)
 *
 * code_step = (grad * lr + 2^14) >> 15 in int64, then the code delta is
 * round-half-away-from-zero(code_step / 258), i.e. floor((|code_step| + 129) /
 * 258) with the sign restored -- the shape of the fallback's two-sided division.
 * The parameter clamp is [-127, 127] and explicit, never SSAT: SSAT's 8-bit rail
 * is [-128, 127] and -128 is a code this core never emits.
 */
    .globl zzs_mllee_train_sgd_step_i8
    .type zzs_mllee_train_sgd_step_i8, %function
    .thumb_func
zzs_mllee_train_sgd_step_i8:
    push {r4, r5, r6, r7, r8, r9, r10, r11}
    cmp r3, #0                    /* CBZ cannot reach the epilogue of a loop this long */
    beq .Ltsg_out
    mov r4, r2                  /* lr_q15 */
    movw r5, #16257             /* ceil(2^22 / 258) */
    movs r6, #129               /* the rounding offset of the 258 division */
    movw r7, #16384             /* the rounding addend of the Q15 shift */
    movw r11, #0xFFFC           /* 254 * 258: the first magnitude that rails */
.Ltsg_loop:
    ldrsh r8, [r1], #2
    smull r8, r10, r8, r4       /* the whole 64-bit lr * grad product */
    adds r8, r7
    adcs r10, #0
    mov r9, r10, asr #15        /* code_step's high word */
    lsr r8, #15
    orr r8, r8, r10, lsl #17    /* code_step's low word */
    mov r10, r9, asr #31        /* sign mask, survives into the rail path */
    cmp r9, #0
    bge .Ltsg_mag
    mvn r9, r9                    /* Thumb-2 has no RSCS: ~hi, then -lo, then +carry */
    rsbs r8, r8, #0
    adcs r9, #0                   /* |code_step| */
.Ltsg_mag:
    cmp r9, #0
    bne .Ltsg_rail              /* magnitude >= 2^32: far past any rail */
    cmp r8, r11
    bhs .Ltsg_rail              /* delta >= 254: the parameter is on its rail */
    add r8, r6
    umull r8, r12, r8, r5       /* the product fits 32 bits by the test above */
    lsr r8, #22                 /* floor((|code_step| + 129) / 258) */
    eors r8, r10
    subs r8, r10                /* signed code delta */
    ldrsb r9, [r0]
    adds r9, r8
    movs r12, #127
    cmp r9, r12
    bgt .Ltsg_high
    rsbs r12, r12, #0
    cmp r9, r12
    blt .Ltsg_low
    strb r9, [r0]
    b .Ltsg_next
.Ltsg_rail:
    cmp r10, #0
    blt .Ltsg_low
.Ltsg_high:
    movs r9, #127
    strb r9, [r0]
    b .Ltsg_next
.Ltsg_low:
    movs r9, #127
    rsbs r9, r9, #0
    strb r9, [r0]
.Ltsg_next:
    adds r0, #1
    subs r3, #1
    bne .Ltsg_loop
.Ltsg_out:
    pop {r4, r5, r6, r7, r8, r9, r10, r11}
    bx lr
    .size zzs_mllee_train_sgd_step_i8, .-zzs_mllee_train_sgd_step_i8

.section .note.GNU-stack,"",%progbits
