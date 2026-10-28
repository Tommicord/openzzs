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
 * mllee x86-64 norm statistics (SSE2).
 *
 * _S_window_stats needs the sum and the sum of squares of every code in a
 * normalization window; the portable form does both with a widening multiply
 * per element.  Here pmaddwd with the operand pair produces the int16 square
 * sums directly and pmaddwd against a vector of ones produces the plain sums,
 * so one 16-byte group costs four multiplies and no branch per element.
 *
 * Leaf routine: no stack frame, no callee-saved register, no writable static.
 */

.intel_syntax noprefix

.section .rodata
.align 16
.Lst_ones16:
    .word 1, 1, 1, 1, 1, 1, 1, 1

.text

/*
 * void zzs_mllee_stats_i8(const int8_t *values, uint32_t count, int32_t *sum,
 *                         int32_t *square_sum)
 *
 * Both results are int32.  The square accumulator holds at most
 * count / 8 * 16129 per lane, so any count up to 65535 -- the widest a layer
 * dimension can be -- stays far inside the lane.
 */
.globl zzs_mllee_stats_i8
.type zzs_mllee_stats_i8, @function
zzs_mllee_stats_i8:
    mov r9, rcx
    pxor xmm0, xmm0                 /* int32 sum lanes                   */
    pxor xmm1, xmm1                 /* int32 square-sum lanes            */
    pxor xmm3, xmm3                 /* zero, the sign-extension operand  */
    movdqa xmm2, XMMWORD PTR [rip + .Lst_ones16]
    xor eax, eax
    xor r8d, r8d
    cmp esi, 16
    jb .Lstats_tail

.Lstats_loop:
    movdqu xmm4, XMMWORD PTR [rdi]
    movdqa xmm5, xmm3
    movdqa xmm6, xmm4
    pcmpgtb xmm5, xmm4              /* 0xFF where the code is negative   */
    punpcklbw xmm4, xmm5            /* low 8 codes widened to int16      */
    punpckhbw xmm6, xmm5            /* high 8 codes widened to int16     */
    movdqa xmm7, xmm4
    movdqa xmm8, xmm6
    pmaddwd xmm4, xmm2              /* a + b of each int16 pair          */
    pmaddwd xmm6, xmm2
    pmaddwd xmm7, xmm7              /* a*a + b*b of each int16 pair      */
    pmaddwd xmm8, xmm8
    paddd xmm0, xmm4
    paddd xmm0, xmm6
    paddd xmm1, xmm7
    paddd xmm1, xmm8
    add rdi, 16
    sub esi, 16
    cmp esi, 16
    jae .Lstats_loop

    movdqa xmm4, xmm0
    movhlps xmm5, xmm4
    paddd xmm4, xmm5
    pshufd xmm5, xmm4, 0x55
    paddd xmm4, xmm5
    movd eax, xmm4
    movdqa xmm6, xmm1
    movhlps xmm7, xmm6
    paddd xmm6, xmm7
    pshufd xmm7, xmm6, 0x55
    paddd xmm6, xmm7
    movd r8d, xmm6

.Lstats_tail:
    test esi, esi
    jz .Lstats_store
.Lstats_tail_loop:
    movsx ecx, BYTE PTR [rdi]
    add eax, ecx
    imul ecx, ecx
    add r8d, ecx
    inc rdi
    dec esi
    jnz .Lstats_tail_loop
.Lstats_store:
    mov DWORD PTR [rdx], eax
    mov DWORD PTR [r9], r8d
    ret
.size zzs_mllee_stats_i8, .-zzs_mllee_stats_i8

.section .note.GNU-stack,"",@progbits
