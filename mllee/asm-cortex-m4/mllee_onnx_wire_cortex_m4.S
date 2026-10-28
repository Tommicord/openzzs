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
 * mllee ONNX wire primitives (Cortex-M4, ARMv7E-M).
 *
 * The same four leaf routines as the x86-64 file.  The 32-bit varint and the
 * hash use only Thumb-2 data processing.  The 64-bit varint splits the value
 * into low/high halves around a 32-bit shift boundary because ARMv7-M has no
 * 64-bit shift instruction; the loop stays branch-predictable and runs once per
 * varint byte (at most ten).  The copy uses word moves only when both pointers
 * are aligned, then a byte tail.
 *
 * No literal pool, no calls, no writable static.
 */

.syntax unified
.thumb
.cpu cortex-m4
.section .text.mllee.m4.onnx

/*
 * uint32_t zzs_mllee_onnx_varint32(const uint8_t *src, uint32_t available,
 *                                  uint32_t *out)
 */
    .globl zzs_mllee_onnx_varint32
    .type zzs_mllee_onnx_varint32, %function
    .thumb_func
zzs_mllee_onnx_varint32:
    push {r4, r5, r6, r7}
    movs r3, #0                 /* accumulated value          */
    movs r4, #0                 /* index                      */
    movs r5, #0                 /* shift                      */
    cmp r1, #0
    beq .Lv32_fail
    cmp r1, #5
    bls .Lv32_loop
    movs r1, #5
.Lv32_loop:
    ldrb r6, [r0, r4]
    cmp r4, #4
    bne .Lv32_no_fifth
    tst r6, #0xF0
    bne .Lv32_fail
.Lv32_no_fifth:
    and r7, r6, #0x7F
    lsls r7, r7, r5
    orrs r3, r3, r7
    lsls r6, r6, #24            /* bit 7 -> N */
    bpl .Lv32_done
    adds r5, r5, #7
    adds r4, r4, #1
    cmp r4, r1
    bcc .Lv32_loop
.Lv32_fail:
    movs r0, #0
    pop {r4, r5, r6, r7}
    bx lr
.Lv32_done:
    str r3, [r2]
    adds r0, r4, #1
    pop {r4, r5, r6, r7}
    bx lr
    .size zzs_mllee_onnx_varint32, .-zzs_mllee_onnx_varint32

/*
 * uint32_t zzs_mllee_onnx_varint64(const uint8_t *src, uint32_t available,
 *                                  uint64_t *out)
 *
 * r3:r4 is the accumulator, r5 the shift, r6 the payload bits, r7 the index
 * and r8 the output pointer.  The 10th byte may contribute only one bit.
 */
    .globl zzs_mllee_onnx_varint64
    .type zzs_mllee_onnx_varint64, %function
    .thumb_func
zzs_mllee_onnx_varint64:
    push {r4, r5, r6, r7, r8}
    mov r8, r2
    movs r3, #0
    movs r4, #0
    movs r5, #0
    movs r7, #0
    cmp r1, #0
    beq .Lv64_fail
    cmp r1, #10
    bls .Lv64_loop
    movs r1, #10
.Lv64_loop:
    ldrb r12, [r0, r7]
    cmp r7, #9
    bne .Lv64_no_tenth
    tst r12, #0xFE
    bne .Lv64_fail
.Lv64_no_tenth:
    and r6, r12, #0x7F
    cmp r5, #32
    bcs .Lv64_high
    lsls r2, r6, r5
    orrs r3, r3, r2
    cbz r5, .Lv64_shifted
    rsb r2, r5, #32
    lsrs r2, r6, r2
    orrs r4, r4, r2
    b .Lv64_shifted
.Lv64_high:
    subs r2, r5, #32
    lsls r2, r6, r2
    orrs r4, r4, r2
.Lv64_shifted:
    lsls r2, r12, #24           /* continuation bit -> N */
    bpl .Lv64_done
    adds r5, r5, #7
    adds r7, r7, #1
    cmp r7, r1
    bcc .Lv64_loop
.Lv64_fail:
    movs r0, #0
    pop {r4, r5, r6, r7, r8}
    bx lr
.Lv64_done:
    str r3, [r8]
    str r4, [r8, #4]
    adds r0, r7, #1
    pop {r4, r5, r6, r7, r8}
    bx lr
    .size zzs_mllee_onnx_varint64, .-zzs_mllee_onnx_varint64

/*
 * uint32_t zzs_mllee_onnx_hash32(const uint8_t *src, uint32_t length)
 */
    .globl zzs_mllee_onnx_hash32
    .type zzs_mllee_onnx_hash32, %function
    .thumb_func
zzs_mllee_onnx_hash32:
    movw r2, #:lower16:2166136261
    movt r2, #:upper16:2166136261
    movw r3, #:lower16:16777619
    movt r3, #:upper16:16777619
    cbz r1, .Lhash32_done
.Lhash32_loop:
    ldrb r12, [r0], #1
    eor r2, r2, r12
    mul r2, r3, r2
    subs r1, r1, #1
    bne .Lhash32_loop
.Lhash32_done:
    mov r0, r2
    bx lr
    .size zzs_mllee_onnx_hash32, .-zzs_mllee_onnx_hash32

/*
 * void zzs_mllee_onnx_copy(uint8_t *dst, const uint8_t *src, uint32_t length)
 */
    .globl zzs_mllee_onnx_copy
    .type zzs_mllee_onnx_copy, %function
    .thumb_func
zzs_mllee_onnx_copy:
    cbz r2, .Lcopy_done
    orr r3, r0, r1
    tst r3, #3
    bne .Lcopy_byte
.Lcopy_word:
    cmp r2, #4
    blt .Lcopy_byte
    ldr r3, [r1], #4
    str r3, [r0], #4
    subs r2, r2, #4
    bne .Lcopy_word
    bx lr
.Lcopy_byte:
    ldrb r3, [r1], #1
    strb r3, [r0], #1
    subs r2, r2, #1
    bne .Lcopy_byte
.Lcopy_done:
    bx lr
    .size zzs_mllee_onnx_copy, .-zzs_mllee_onnx_copy

.section .note.GNU-stack,"",%progbits
