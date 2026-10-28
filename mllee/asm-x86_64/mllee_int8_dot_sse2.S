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
 * mllee x86-64 baseline MAC / quantization kernels (SSE2 only, the x86-64
 * architectural floor).
 *
 * Every function is a leaf: no stack frame, no callee-saved register, no
 * writable static.  The routines are deliberately fixed-sequence (no CPUID,
 * no dispatch) so the core's instruction count is predictable.  Element-wise
 * set arithmetic in mllee_int8_reduce_sse2.S, the norm statistics reduction in
 * mllee_int8_stats_sse2.S, the shuffle-based codec expansion in
 * mllee_packlow_shuffle_ssse3.S; -DOPENZZS_MLLEE_AVX2=ON adds the AVX2
 * variants.  CRC-16/CCITT stays portable C++ (see the note in CMakeLists.txt).
 */

.intel_syntax noprefix

.section .rodata
.align 16
.Lmllee_mask0f:
    .byte 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F
    .byte 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F
.Lmllee_mask08:
    .byte 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08
    .byte 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08
.Lmllee_min127:
    .word -127, -127, -127, -127, -127, -127, -127, -127
.Lmllee_max127:
    .word 127, 127, 127, 127, 127, 127, 127, 127

.text

/* int32_t zzs_mllee_dot_i8(const int8_t *lhs, const int8_t *rhs, uint32_t count) */
.globl zzs_mllee_dot_i8
.type zzs_mllee_dot_i8, @function
zzs_mllee_dot_i8:
    xor eax, eax
    pxor xmm4, xmm4
    pxor xmm7, xmm7
    cmp edx, 16
    jb .Ldot_tail

.Ldot_loop:
    movdqu xmm0, XMMWORD PTR [rdi]
    movdqu xmm1, XMMWORD PTR [rsi]
    movdqa xmm2, xmm0
    movdqa xmm3, xmm1
    movdqa xmm5, xmm7
    movdqa xmm6, xmm7
    pcmpgtb xmm5, xmm0
    pcmpgtb xmm6, xmm1
    punpcklbw xmm0, xmm5
    punpcklbw xmm1, xmm6
    punpckhbw xmm2, xmm5
    punpckhbw xmm3, xmm6
    pmaddwd xmm0, xmm1
    pmaddwd xmm2, xmm3
    paddd xmm0, xmm2
    paddd xmm4, xmm0
    add rdi, 16
    add rsi, 16
    sub edx, 16
    cmp edx, 16
    jae .Ldot_loop

    movdqa xmm0, xmm4
    movhlps xmm1, xmm0
    paddd xmm0, xmm1
    pshufd xmm1, xmm0, 0x55
    paddd xmm0, xmm1
    movd ecx, xmm0
    add eax, ecx

.Ldot_tail:
    test edx, edx
    jz .Ldot_done
.Ldot_tail_loop:
    movsx ecx, BYTE PTR [rdi]
    movsx r8d, BYTE PTR [rsi]
    imul ecx, r8d
    add eax, ecx
    inc rdi
    inc rsi
    dec edx
    jnz .Ldot_tail_loop
.Ldot_done:
    ret
.size zzs_mllee_dot_i8, .-zzs_mllee_dot_i8

/* int32_t zzs_mllee_requant_q15(int32_t accumulator, int32_t multiplier) */
.globl zzs_mllee_requant_q15
.type zzs_mllee_requant_q15, @function
zzs_mllee_requant_q15:
    movsxd rax, edi
    movsxd rcx, esi
    imul rax, rcx
    add rax, 16384
    sar rax, 15
    ret
.size zzs_mllee_requant_q15, .-zzs_mllee_requant_q15

/* void zzs_mllee_unpack_i4(const uint8_t *src, int8_t *dst, uint32_t bytes) */
.globl zzs_mllee_unpack_i4
.type zzs_mllee_unpack_i4, @function
zzs_mllee_unpack_i4:
    movdqa xmm7, XMMWORD PTR [rip + .Lmllee_mask0f]
    movdqa xmm6, XMMWORD PTR [rip + .Lmllee_mask08]
    cmp edx, 16
    jb .Lunpack_tail

.Lunpack_loop:
    movdqu xmm0, XMMWORD PTR [rdi]
    movdqa xmm1, xmm0
    pand xmm0, xmm7
    psrlw xmm1, 4
    pand xmm1, xmm7
    movdqa xmm2, xmm0
    punpcklbw xmm2, xmm1
    punpckhbw xmm0, xmm1
    pxor xmm2, xmm6
    psubb xmm2, xmm6
    pxor xmm0, xmm6
    psubb xmm0, xmm6
    movdqu XMMWORD PTR [rsi], xmm2
    movdqu XMMWORD PTR [rsi + 16], xmm0
    add rdi, 16
    add rsi, 32
    sub edx, 16
    cmp edx, 16
    jae .Lunpack_loop

.Lunpack_tail:
    test edx, edx
    jz .Lunpack_done
.Lunpack_tail_loop:
    movzx eax, BYTE PTR [rdi]
    mov ecx, eax
    and ecx, 0x0F
    shl ecx, 28
    sar ecx, 28
    mov BYTE PTR [rsi], cl
    shr eax, 4
    shl eax, 28
    sar eax, 28
    mov BYTE PTR [rsi + 1], al
    inc rdi
    add rsi, 2
    dec edx
    jnz .Lunpack_tail_loop
.Lunpack_done:
    ret
.size zzs_mllee_unpack_i4, .-zzs_mllee_unpack_i4




/* void zzs_mllee_quantize_affine_i8(const float *src, int8_t *dst, uint32_t n,
 *                                   float scale, int32_t zero_point) */
.globl zzs_mllee_quantize_affine_i8
.type zzs_mllee_quantize_affine_i8, @function
zzs_mllee_quantize_affine_i8:
    xorps xmm2, xmm2
    ucomiss xmm0, xmm2
    jbe .Lquant_zero
    shufps xmm0, xmm0, 0
    movd xmm4, ecx
    pshufd xmm4, xmm4, 0
    movdqa xmm5, XMMWORD PTR [rip + .Lmllee_min127]
    movdqa xmm6, XMMWORD PTR [rip + .Lmllee_max127]
    cmp edx, 4
    jb .Lquant_tail
.Lquant_loop:
    movups xmm1, XMMWORD PTR [rdi]
    divps xmm1, xmm0
    cvtps2dq xmm1, xmm1
    paddd xmm1, xmm4
    packssdw xmm1, xmm1
    pmaxsw xmm1, xmm5
    pminsw xmm1, xmm6
    packsswb xmm1, xmm1
    movd DWORD PTR [rsi], xmm1
    add rdi, 16
    add rsi, 4
    sub edx, 4
    cmp edx, 4
    jae .Lquant_loop
.Lquant_tail:
    test edx, edx
    jz .Lquant_done
.Lquant_scalar:
    movss xmm1, DWORD PTR [rdi]
    divss xmm1, xmm0
    cvtss2si eax, xmm1
    add eax, ecx
    cmp eax, 127
    jle .Lquant_low_ok
    mov eax, 127
.Lquant_low_ok:
    cmp eax, -127
    jge .Lquant_store
    mov eax, -127
.Lquant_store:
    mov BYTE PTR [rsi], al
    add rdi, 4
    inc rsi
    dec edx
    jnz .Lquant_scalar
.Lquant_done:
    ret
.Lquant_zero:
    test edx, edx
    jz .Lquant_done
.Lquant_zero_loop:
    mov BYTE PTR [rsi], 0
    inc rsi
    dec edx
    jnz .Lquant_zero_loop
    ret
.size zzs_mllee_quantize_affine_i8, .-zzs_mllee_quantize_affine_i8

/* void zzs_mllee_dequantize_i8(const int8_t *src, float *dst, uint32_t n, float scale) */
.globl zzs_mllee_dequantize_i8
.type zzs_mllee_dequantize_i8, @function
zzs_mllee_dequantize_i8:
    test edx, edx
    jz .Ldequant_done
.Ldequant_loop:
    movsx eax, BYTE PTR [rdi]
    cvtsi2ss xmm1, eax
    mulss xmm1, xmm0
    movss DWORD PTR [rsi], xmm1
    inc rdi
    add rsi, 4
    dec edx
    jnz .Ldequant_loop
.Ldequant_done:
    ret
.size zzs_mllee_dequantize_i8, .-zzs_mllee_dequantize_i8

.section .note.GNU-stack,"",@progbits
