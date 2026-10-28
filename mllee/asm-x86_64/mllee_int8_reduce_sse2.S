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
 * mllee x86-64 int8 element-wise and reduction kernels (SSE2 baseline).
 *
 * These are the kernels the tensor ops call once per layer: saturating ADD/SUB
 * over an arena tensor (ADD/SUB ops, residual connections), the row max that
 * softmax/log-softmax need, and the integer sum behind LayerNorm and REDUCE.
 *
 * Signed ordering on byte lanes is obtained by biasing with 0x80 and using the
 * unsigned pmaxub/psubusb lanes, so no lane is ever compared one at a time.
 * Saturation follows the mllee code contract: results clamp to [-127, 127],
 * the range every C++ fallback in mllee/quant.hpp produces.
 */

.intel_syntax noprefix

.section .rodata
.align 16
.Lrs_ones:
    .word 1, 1, 1, 1, 1, 1, 1, 1
.Lrs_sign:
    .byte 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
    .byte 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
.Lrs_oneb:
    .byte 1, 1, 1, 1, 1, 1, 1, 1
    .byte 1, 1, 1, 1, 1, 1, 1, 1
.Lrs_min8:
    .byte 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80
    .byte 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80

.text

/*
 * void zzs_mllee_add_sat_i8(const int8_t *lhs, const int8_t *rhs,
 *                           int8_t *destination, uint32_t count)
 *
 * 16 codes per paddsb; the -128 lanes paddsb can produce are lifted to -127 so
 * the asm and the portable path return the same bytes.
 */
.globl zzs_mllee_add_sat_i8
.type zzs_mllee_add_sat_i8, @function
zzs_mllee_add_sat_i8:
    movdqa xmm7, XMMWORD PTR [rip + .Lrs_min8]
    movdqa xmm6, XMMWORD PTR [rip + .Lrs_oneb]
    cmp ecx, 16
    jb .Laddv_tail
.Laddv_loop:
    movdqu xmm0, XMMWORD PTR [rdi]
    movdqu xmm1, XMMWORD PTR [rsi]
    paddsb xmm0, xmm1
    movdqa xmm2, xmm0
    pcmpeqb xmm2, xmm7
    pand xmm2, xmm6
    paddb xmm0, xmm2
    movdqu XMMWORD PTR [rdx], xmm0
    add rdi, 16
    add rsi, 16
    add rdx, 16
    sub ecx, 16
    cmp ecx, 16
    jae .Laddv_loop
.Laddv_tail:
    test ecx, ecx
    jz .Laddv_done
.Laddv_tail_loop:
    movsx eax, BYTE PTR [rdi]
    movsx r8d, BYTE PTR [rsi]
    add eax, r8d
    cmp eax, 127
    jle .Laddv_low_ok
    mov eax, 127
.Laddv_low_ok:
    cmp eax, -127
    jge .Laddv_store
    mov eax, -127
.Laddv_store:
    mov BYTE PTR [rdx], al
    inc rdi
    inc rsi
    inc rdx
    dec ecx
    jnz .Laddv_tail_loop
.Laddv_done:
    ret
.size zzs_mllee_add_sat_i8, .-zzs_mllee_add_sat_i8

/*
 * void zzs_mllee_sub_sat_i8(const int8_t *lhs, const int8_t *rhs,
 *                           int8_t *destination, uint32_t count)
 */
.globl zzs_mllee_sub_sat_i8
.type zzs_mllee_sub_sat_i8, @function
zzs_mllee_sub_sat_i8:
    movdqa xmm7, XMMWORD PTR [rip + .Lrs_min8]
    movdqa xmm6, XMMWORD PTR [rip + .Lrs_oneb]
    cmp ecx, 16
    jb .Lsubv_tail
.Lsubv_loop:
    movdqu xmm0, XMMWORD PTR [rdi]
    movdqu xmm1, XMMWORD PTR [rsi]
    psubsb xmm0, xmm1
    movdqa xmm2, xmm0
    pcmpeqb xmm2, xmm7
    pand xmm2, xmm6
    paddb xmm0, xmm2
    movdqu XMMWORD PTR [rdx], xmm0
    add rdi, 16
    add rsi, 16
    add rdx, 16
    sub ecx, 16
    cmp ecx, 16
    jae .Lsubv_loop
.Lsubv_tail:
    test ecx, ecx
    jz .Lsubv_done
.Lsubv_tail_loop:
    movsx eax, BYTE PTR [rdi]
    movsx r8d, BYTE PTR [rsi]
    sub eax, r8d
    cmp eax, 127
    jle .Lsubv_low_ok
    mov eax, 127
.Lsubv_low_ok:
    cmp eax, -127
    jge .Lsubv_store
    mov eax, -127
.Lsubv_store:
    mov BYTE PTR [rdx], al
    inc rdi
    inc rsi
    inc rdx
    dec ecx
    jnz .Lsubv_tail_loop
.Lsubv_done:
    ret
.size zzs_mllee_sub_sat_i8, .-zzs_mllee_sub_sat_i8

/* int32_t zzs_mllee_sum_i8(const int8_t *values, uint32_t count) */
.globl zzs_mllee_sum_i8
.type zzs_mllee_sum_i8, @function
zzs_mllee_sum_i8:
    xor eax, eax
    pxor xmm4, xmm4
    pxor xmm7, xmm7
    movdqa xmm6, XMMWORD PTR [rip + .Lrs_ones]
    cmp esi, 16
    jb .Lsumv_tail
.Lsumv_loop:
    movdqu xmm0, XMMWORD PTR [rdi]
    movdqa xmm1, xmm7
    pcmpgtb xmm1, xmm0
    movdqa xmm2, xmm0
    punpcklbw xmm0, xmm1
    punpckhbw xmm2, xmm1
    pmaddwd xmm0, xmm6
    pmaddwd xmm2, xmm6
    paddd xmm0, xmm2
    paddd xmm4, xmm0
    add rdi, 16
    sub esi, 16
    cmp esi, 16
    jae .Lsumv_loop
    movdqa xmm0, xmm4
    movhlps xmm1, xmm0
    paddd xmm0, xmm1
    pshufd xmm1, xmm0, 0x55
    paddd xmm0, xmm1
    movd ecx, xmm0
    add eax, ecx
.Lsumv_tail:
    test esi, esi
    jz .Lsumv_done
.Lsumv_tail_loop:
    movsx ecx, BYTE PTR [rdi]
    add eax, ecx
    inc rdi
    dec esi
    jnz .Lsumv_tail_loop
.Lsumv_done:
    ret
.size zzs_mllee_sum_i8, .-zzs_mllee_sum_i8

/*
 * int8_t zzs_mllee_max_i8(const int8_t *values, uint32_t count)
 *
 * Bias to unsigned once per 16-byte block (pxor 0x80), fold the block max with
 * pmaxub, and un-bias at the end.  16 bytes per iteration, no per-lane branch.
 */
.globl zzs_mllee_max_i8
.type zzs_mllee_max_i8, @function
zzs_mllee_max_i8:
    mov eax, -128
    test esi, esi
    jz .Lmaxv_done
    cmp esi, 16
    jb .Lmaxv_scalar
    movdqa xmm7, XMMWORD PTR [rip + .Lrs_sign]
    movdqu xmm0, XMMWORD PTR [rdi]
    pxor xmm0, xmm7
    movdqa xmm1, xmm0
    add rdi, 16
    sub esi, 16
.Lmaxv_loop:
    movdqu xmm2, XMMWORD PTR [rdi]
    pxor xmm2, xmm7
    pmaxub xmm1, xmm2
    add rdi, 16
    sub esi, 16
    cmp esi, 16
    jae .Lmaxv_loop
    /* fold 16 lanes into lane 0 with the unsigned byte ordering */
    movdqa xmm0, xmm1
    psrlq xmm0, 32
    pmaxub xmm1, xmm0
    movdqa xmm0, xmm1
    psrlq xmm0, 16
    pmaxub xmm1, xmm0
    movdqa xmm0, xmm1
    psrlq xmm0, 8
    pmaxub xmm1, xmm0
    movd ecx, xmm1
    movzx ecx, cl
    xor ecx, 0x80
    movsx eax, cl
    test esi, esi
    jz .Lmaxv_done
.Lmaxv_scalar:
    movsx ecx, BYTE PTR [rdi]
    cmp ecx, eax
    cmovg eax, ecx
    inc rdi
    dec esi
    jnz .Lmaxv_scalar
.Lmaxv_done:
    ret
.size zzs_mllee_max_i8, .-zzs_mllee_max_i8

.section .note.GNU-stack,"",@progbits
