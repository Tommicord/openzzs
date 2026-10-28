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
 * mllee Cortex-M4 (ARMv7E-M + DSP) training kernels, lane arithmetic group.
 *
 * Same frozen interface as mllee/train_simd.hpp's portable fallbacks, same
 * byte-for-byte results.  Fixed-sequence leaf routines: no literal pools, no
 * writable static, no calls, no CPU probing.
 *
 * The instruction set here is narrower than the header comment of
 * mllee_int8_mac_cortex_m4.S assumed.  QEMU's mps2-an385 Cortex-M4 model UNDEFs
 * the whole sign/zero-extend-halfword group (SXTB16, SXTAB16, UXTB16, UXTH16 --
 * one encoding family), and ARMv7E-M has no lane-wise int16 add at all: SADD16
 * and friends are ARMv6 media instructions that Cortex-M never implemented, and
 * QADD/QSUB fault here too.  Two int16 lanes per word therefore have to be
 * unpacked with shifts, which costs the same as keeping the element loop scalar,
 * so every kernel in this file runs one lane per pass and stays free of any
 * extend/add SIMD mnemonic.  The hardware is used where it is unconditionally
 * available and actually saves work: SMULL for the 64-bit products, SSAT for the
 * Q15 rails, BIC for the branch-free ReLU select, and shift/add for the 258-unit
 * code step so no division reaches the hot loop at all.
 *
 * Lane streams are read with LDRSH/STRH, which need 2-byte alignment; the
 * self-test walks its windows by element, so every case is aligned by
 * construction.  The int8 code streams need nothing.
 *
 * Callee-saved registers are pushed in even counts so the frame stays
 * double-word aligned, and no SMULL here uses r15 as its high accumulator.
 */

.syntax unified
.thumb
.cpu cortex-m4
.section .text.mllee.m4.trainlanes

/*
 * uint32_t zzs_mllee_train_add_grad_q15(int16_t *destination,
 *                                       const int16_t *source, uint32_t count)
 *
 * Two Q15 lanes sum inside [-65536, 65534], so the int32 add never wraps and the
 * only work is the rail.  The rails are explicit rather than SSAT because the
 * count of clamped lanes has to come back out, and "the value changed" is a
 * cheaper test than re-deriving the pre-clamp sum.
 */
    .globl zzs_mllee_train_add_grad_q15
    .type zzs_mllee_train_add_grad_q15, %function
    .thumb_func
zzs_mllee_train_add_grad_q15:
    push {r4, r5, r6, r7}
    movs r6, #0                     /* clamped lanes */
    cbz r2, .Ltag_ret
    movw r4, #32767
    rsbs r5, r4, #0
    subs r5, #1                 /* -32768: the negative rail is one below -r4 */
.Ltag_loop:
    ldrsh r3, [r0]
    ldrsh r7, [r1]
    adds r3, r7
    cmp r3, r4
    bgt .Ltag_high
    cmp r3, r5
    blt .Ltag_low
    strh r3, [r0]
    b .Ltag_next
.Ltag_high:
    strh r4, [r0]
    adds r6, #1
    b .Ltag_next
.Ltag_low:
    strh r5, [r0]
    adds r6, #1
.Ltag_next:
    adds r0, #2
    adds r1, #2
    subs r2, #1
    bne .Ltag_loop
.Ltag_ret:
    mov r0, r6
    pop {r4, r5, r6, r7}
    bx lr
    .size zzs_mllee_train_add_grad_q15, .-zzs_mllee_train_add_grad_q15

/*
 * void zzs_mllee_train_axpy_q15(int16_t *destination, const int16_t *source,
 *                               int32_t scalar_q15, uint32_t count)
 *
 * The fallback narrows the rounded 64-bit product with a plain int32 cast, so
 * the fold below is not a truncation the kernel invents -- it is the same one.
 * After that the destination add is int32 and SSAT supplies the [-32768, 32767]
 * rail the wrapper documents.
 */
    .globl zzs_mllee_train_axpy_q15
    .type zzs_mllee_train_axpy_q15, %function
    .thumb_func
zzs_mllee_train_axpy_q15:
    push {r4, r5, r6, r7, r8, lr}
    cbz r3, .Ltax_out
    mov r4, r2                      /* scalar */
    movw r7, #16384                 /* the rounding addend, as a register */
.Ltax_loop:
    ldrsh r5, [r1], #2
    ldrsh r6, [r0]
    smull r12, r8, r5, r4           /* the whole 64-bit product */
    adds r12, r7
    adcs r8, #0
    lsr r12, #15
    orr r12, r12, r8, lsl #17       /* (product + 2^14) >> 15, low 32 bits */
    adds r12, r6
    ssat r12, #16, r12
    strh r12, [r0]
    adds r0, #2
    subs r3, #1
    bne .Ltax_loop
.Ltax_out:
    pop {r4, r5, r6, r7, r8, lr}
    bx lr
    .size zzs_mllee_train_axpy_q15, .-zzs_mllee_train_axpy_q15

/*
 * void zzs_mllee_train_scale_q15(int16_t *values, int32_t scalar_q15,
 *                                uint32_t count)
 *
 * Same product fold as axpy, but this wrapper clamps the int64 shift rather than
 * casting it, so the kernel checks whether the folded word carries the whole
 * result: if the shifted value's high word disagrees with the low word's sign,
 * the magnitude is past int32 and the Q15 rail is decided by its sign alone.
 * Both paths are the fallback's, one branch for each of its two narrowings.
 */
    .globl zzs_mllee_train_scale_q15
    .type zzs_mllee_train_scale_q15, %function
    .thumb_func
zzs_mllee_train_scale_q15:
    push {r4, r5, r6, r7}
    cbz r2, .Ltsl_out
    mov r4, r1                      /* scalar */
    movw r6, #16384
.Ltsl_loop:
    ldrsh r5, [r0]
    smull r12, r7, r5, r4
    adds r12, r6
    adcs r7, #0
    mov r5, r7, asr #15             /* the shifted value's high word */
    lsr r12, #15
    orr r12, r12, r7, lsl #17       /* its low word */
    mov r7, r12, asr #31
    cmp r5, r7
    bne .Ltsl_rail
    ssat r12, #16, r12
    b .Ltsl_store
.Ltsl_rail:
    cmp r5, #0
    blt .Ltsl_min
    movw r12, #32767
    b .Ltsl_store
.Ltsl_min:
    movs r12, #1
    lsls r12, #15                   /* 0x8000, whose low halfword is the -32768 rail */
.Ltsl_store:
    strh r12, [r0]
    adds r0, #2
    subs r2, #1
    bne .Ltsl_loop
.Ltsl_out:
    pop {r4, r5, r6, r7}
    bx lr
    .size zzs_mllee_train_scale_q15, .-zzs_mllee_train_scale_q15

/*
 * void zzs_mllee_train_relu_bwd(const int16_t *grad_out, const int8_t *input,
 *                                int16_t *grad_in, uint32_t count)
 *
 * LDRSB's sign bit is the predicate, so the select is an ASR mask and a BIC:
 * no SEL (which the model UNDEFs) and no branch on the activation sign.  The
 * gradient and the mask cannot share r12, which is what IP aliases to, so the
 * pair lives in two saved registers instead.
 */
    .globl zzs_mllee_train_relu_bwd
    .type zzs_mllee_train_relu_bwd, %function
    .thumb_func
zzs_mllee_train_relu_bwd:
    push {r4, r5}
    cbz r3, .Ltrl_out
.Ltrl_loop:
    ldrsh r4, [r0], #2              /* grad_out */
    ldrsb r5, [r1], #1              /* the forward code, whose sign is the gate */
    movs r5, r5, asr #31            /* all ones where the code is negative */
    bic r4, r4, r5
    strh r4, [r2], #2
    subs r3, #1
    bne .Ltrl_loop
.Ltrl_out:
    pop {r4, r5}
    bx lr
    .size zzs_mllee_train_relu_bwd, .-zzs_mllee_train_relu_bwd

/*
 * void zzs_mllee_train_codes_to_q15(const int8_t *codes, int16_t *values,
 *                                   uint32_t count)
 *
 * 258 = 256 + 2, so the widening is (code << 8) + (code << 1): two shifts, one
 * add and no multiply.  The store keeps the fallback's int16 wrap for a -128
 * code, which the core never emits but the wrapper does not filter.
 */
    .globl zzs_mllee_train_codes_to_q15
    .type zzs_mllee_train_codes_to_q15, %function
    .thumb_func
zzs_mllee_train_codes_to_q15:
    cbz r2, .Ltcw_out
.Ltcw_loop:
    ldrsb r3, [r0], #1
    mov r12, r3, lsl #8
    lsls r3, #1
    adds r3, r12                    /* 256 * code + 2 * code */
    strh r3, [r1], #2
    subs r2, #1
    bne .Ltcw_loop
.Ltcw_out:
    bx lr
    .size zzs_mllee_train_codes_to_q15, .-zzs_mllee_train_codes_to_q15

/*
 * void zzs_mllee_train_q15_to_codes(const int16_t *values, int8_t *codes,
 *                                    uint32_t count)
 *
 * The fallback is (|v| * 127 + 2^14) >> 15 with the sign restored -- note the
 * 32768 divisor, which is NOT floor((|v| + 129) / 258): they disagree at v = 129.
 * So the kernel evaluates the fallback's own expression, and it needs no
 * reciprocal: 127 * m fits 23 bits for every lane, the rounding addend fits a
 * register hoisted out of the loop, and a logical shift is already the
 * round-half-up the wrapper documents.  The largest result is 127 (at m = 32768),
 * so the [-127, 127] rail needs no test and never produces the -128 code.
 */
    .globl zzs_mllee_train_q15_to_codes
    .type zzs_mllee_train_q15_to_codes, %function
    .thumb_func
zzs_mllee_train_q15_to_codes:
    push {r4, r5}
    cbz r2, .Ltnc_out
    movs r4, #1
    lsls r4, #14                    /* 16384: not encodable as an ADD immediate */
.Ltnc_loop:
    ldrsh r3, [r0], #2
    movs r5, r3, asr #31
    eors r3, r5
    subs r3, r5                     /* magnitude, 0..32768 */
    mov r12, r3, lsl #7
    subs r3, r12, r3                /* 127 * magnitude */
    adds r3, r4
    lsrs r3, #15                    /* code magnitude, at most 127 */
    eors r3, r5
    subs r3, r5                     /* signed code */
    strb r3, [r1], #1
    subs r2, #1
    bne .Ltnc_loop
.Ltnc_out:
    pop {r4, r5}
    bx lr
    .size zzs_mllee_train_q15_to_codes, .-zzs_mllee_train_q15_to_codes

.section .note.GNU-stack,"",%progbits
