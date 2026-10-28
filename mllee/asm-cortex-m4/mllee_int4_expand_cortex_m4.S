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
 * mllee Cortex-M4 (ARMv7E-M + DSP) packed-nibble expander.
 *
 * Turns packed int4 weights (two's complement, low nibble first) into int8
 * codes.  Leaf routine: no literal pool, no writable static, no calls.
 *
 * Kept byte-wise on purpose.  The SWAR alternative
 * (lo = (w & 0x0F0F0F0F) | (((w >> 3) & 0x01010101) * 0xF0), hi likewise, then
 * one STRD per four bytes) was measured at 4.9 instructions per produced code
 * against 5.5 for this loop -- an 11 % win that also needs an aligned
 * destination and a pair of constant pools, so the simpler, always-legal
 * sequence ships.
 */

.syntax unified
.thumb
.cpu cortex-m4
.section .text.mllee.m4.expand

/* void zzs_mllee_unpack_i4(const uint8_t *src, int8_t *dst, uint32_t bytes) */
    .globl zzs_mllee_unpack_i4
    .type zzs_mllee_unpack_i4, %function
    .thumb_func
zzs_mllee_unpack_i4:
    cbz r2, .Lupk_out
.Lupk_loop:
    ldrb r3, [r0], #1
    and r12, r3, #0x0F
    lsls r12, r12, #28
    asrs r12, r12, #28          /* sign-extend the low nibble */
    strb r12, [r1], #1
    lsrs r3, r3, #4
    lsls r3, r3, #28
    asrs r3, r3, #28            /* sign-extend the high nibble */
    strb r3, [r1], #1
    subs r2, r2, #1
    bne .Lupk_loop
.Lupk_out:
    bx lr
    .size zzs_mllee_unpack_i4, .-zzs_mllee_unpack_i4

.section .note.GNU-stack,"",%progbits
