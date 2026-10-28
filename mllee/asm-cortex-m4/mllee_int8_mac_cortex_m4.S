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
 * mllee Cortex-M4 (ARMv7E-M + DSP) multiply-accumulate kernels.
 *
 * Fixed-sequence leaf routines: no literal pools, no writable static, no
 * calls, no CPU probing.  Only ARMv7E-M instructions are used -- the ARMv6
 * byte/halfword SIMD group (ADD8, ADD16, SEL...) is NOT implemented on
 * Cortex-M, so the byte-wise kernels stay scalar and the word kernels get
 * their parallelism from SXTB16 + SMLAD instead.  SXTB16/SMLAD are the
 * documented Armv7E-M DSP instructions (CMSIS __SXTB16 / __SMLAD for
 * Cortex-M4/M7); QEMU's Cortex-M4 model does not decode SXTB16, so
 * tools/run_mllee_m4_qemu.sh probes for it and reports the lane paths as
 * unverified instead of passing them silently.
 *
 * Alignment is gated once per call, never per element: an unaligned LDR faults
 * on ARMv7-M, so a stream that is not 4-byte aligned takes the scalar path for
 * its whole length instead of branching inside the loop.
 */

.syntax unified
.thumb
.cpu cortex-m4
.section .text.mllee.m4.mac

/*
 * int32_t zzs_mllee_dot_i8(const int8_t *lhs, const int8_t *rhs, uint32_t count)
 *
 * SXTB16 splits a packed word into two int16 lanes, and SMLAD adds the two
 * lane-wise products into a 32-bit accumulator in one instruction: eight
 * MACs per four loads, 19 instructions per 8 elements.  The tail and the
 * misaligned case use MUL/ADD, which every ARM architecture level decodes,
 * instead of the SMLABB single-cycle MAC.
 */
    .globl zzs_mllee_dot_i8
    .type zzs_mllee_dot_i8, %function
    .thumb_func
zzs_mllee_dot_i8:
    push {r4, r5, r6, r7, r8, lr}
    movs r3, #0                 /* accumulator */
    cmp r2, #0
    beq .Ldot_out
    orr r4, r0, r1
    tst r4, #3
    bne .Ldot_scalar
.Ldot_w8:
    cmp r2, #8
    blt .Ldot_w4
    ldr r4, [r0], #4
    ldr r5, [r1], #4
    sxtb16 r6, r4
    sxtb16 r7, r4, ror #8
    sxtb16 r4, r5
    sxtb16 r5, r5, ror #8
    smlad r3, r6, r4, r3
    smlad r3, r7, r5, r3
    ldr r4, [r0], #4
    ldr r5, [r1], #4
    sxtb16 r6, r4
    sxtb16 r7, r4, ror #8
    sxtb16 r4, r5
    sxtb16 r5, r5, ror #8
    smlad r3, r6, r4, r3
    smlad r3, r7, r5, r3
    subs r2, r2, #8
    bne .Ldot_w8
    b .Ldot_out
.Ldot_w4:
    cmp r2, #4
    blt .Ldot_scalar
    ldr r4, [r0], #4
    ldr r5, [r1], #4
    sxtb16 r6, r4
    sxtb16 r7, r4, ror #8
    sxtb16 r4, r5
    sxtb16 r5, r5, ror #8
    smlad r3, r6, r4, r3
    smlad r3, r7, r5, r3
    subs r2, r2, #4
    bne .Ldot_w4
    b .Ldot_out
.Ldot_scalar:
    cbz r2, .Ldot_out
.Ldot_scalar_loop:
    ldrsb r4, [r0], #1
    ldrsb r5, [r1], #1
    mul r6, r4, r5
    adds r3, r6
    subs r2, r2, #1
    bne .Ldot_scalar_loop
.Ldot_out:
    mov r0, r3
    pop {r4, r5, r6, r7, r8, lr}
    bx lr
    .size zzs_mllee_dot_i8, .-zzs_mllee_dot_i8

/*
 * int32_t zzs_mllee_requant_q15(int32_t accumulator, int32_t multiplier)
 *
 * (accumulator * multiplier + 16384) >> 15, rounded half up.  SMULL builds the
 * full 64-bit product so the shift never loses a bit; the low 32 bits of an
 * arithmetic shift by 15 are exactly (low >> 15) | (high << 17), which is what
 * the pair of shifts below reassembles.
 */
    .globl zzs_mllee_requant_q15
    .type zzs_mllee_requant_q15, %function
    .thumb_func
zzs_mllee_requant_q15:
    smull r2, r3, r0, r1
    movw r12, #0x4000
    adds r2, r2, r12
    adc r3, r3, #0
    lsrs r0, r2, #15
    orr r0, r0, r3, lsl #17
    bx lr
    .size zzs_mllee_requant_q15, .-zzs_mllee_requant_q15

/*
 * void zzs_mllee_stats_i8(const int8_t *values, uint32_t count,
 *                         int32_t *sum, int32_t *square_sum)
 *
 * Both reductions come from the same lane split: SMLAD against 0x00010001 is a
 * two-lane sum, SMLAD against itself is a two-lane sum of squares.  Four
 * elements cost one load, two SXTB16 and four SMLAD.
 */
    .globl zzs_mllee_stats_i8
    .type zzs_mllee_stats_i8, %function
    .thumb_func
zzs_mllee_stats_i8:
    push {r4, r5, r6, r7, r8, lr}
    mov r6, r2                  /* sum pointer */
    mov r7, r3                  /* square-sum pointer */
    movs r4, #0                 /* sum */
    movs r5, #0                 /* square sum */
    cmp r1, #0
    beq .Lst_store
    tst r0, #3
    bne .Lst_scalar
    mov.w r8, #0x00010001
.Lst_w4:
    cmp r1, #4
    blt .Lst_scalar
    ldr r2, [r0], #4
    sxtb16 r3, r2
    sxtb16 r2, r2, ror #8
    smlad r4, r3, r8, r4
    smlad r4, r2, r8, r4
    smlad r5, r3, r3, r5
    smlad r5, r2, r2, r5
    subs r1, r1, #4
    bne .Lst_w4
    b .Lst_store
.Lst_scalar:
    cbz r1, .Lst_store
.Lst_scalar_loop:
    ldrsb r2, [r0], #1
    adds r4, r4, r2
    mul r3, r2, r2
    adds r5, r5, r3
    subs r1, r1, #1
    bne .Lst_scalar_loop
.Lst_store:
    str r4, [r6]
    str r5, [r7]
    pop {r4, r5, r6, r7, r8, lr}
    bx lr
    .size zzs_mllee_stats_i8, .-zzs_mllee_stats_i8

.section .note.GNU-stack,"",%progbits
