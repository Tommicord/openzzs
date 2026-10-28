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
 * AVX2 arena element-wise kernels: clamp, mask-blend and sum-of-squares.
 *
 * These three ops were scalar per-element loops.  Clamp uses the native signed
 * byte min/max; masked fill uses one pcmpeqb plus a blend (no branch per
 * element, which also keeps the timing data-independent); sum-of-squares uses
 * the same pmaddwd fold as the SSE2 stats kernel but without paying for the
 * plain sum the L2 path never reads.
 *
 * Leaf routines, 32 lanes per iteration, scalar tail.
 */

.intel_syntax noprefix
.text

/* void zzs_mllee_clamp_i8(const int8_t *src, int8_t *dst, uint32_t count,
 *                         int8_t low, int8_t high) */
.globl zzs_mllee_clamp_i8
.type zzs_mllee_clamp_i8, @function
zzs_mllee_clamp_i8:
    vpbroadcastb ymm6, BYTE PTR [rsp + 8]   /* fifth argument: low  */
    vpbroadcastb ymm7, BYTE PTR [rsp + 16]  /* sixth argument: high */
    cmp     edx, 32
    jb      .Lclamp_tail
    .p2align 4
.Lclamp_loop:
    vmovdqu ymm0, YMMWORD PTR [rdi]
    vpmaxsb ymm0, ymm0, ymm6
    vpminsb ymm0, ymm0, ymm7
    vmovdqu YMMWORD PTR [rsi], ymm0
    add     rdi, 32
    add     rsi, 32
    sub     edx, 32
    cmp     edx, 32
    jae     .Lclamp_loop
.Lclamp_tail:
    test    edx, edx
    jz      .Lclamp_done
    movsx   eax, BYTE PTR [rsp + 8]
    movsx   r8d, BYTE PTR [rsp + 16]
.Lclamp_tail_loop:
    movsx   ecx, BYTE PTR [rdi]
    cmp     ecx, eax
    cmovl   ecx, eax
    cmp     ecx, r8d
    cmovg   ecx, r8d
    mov     BYTE PTR [rsi], cl
    inc     rdi
    inc     rsi
    dec     edx
    jnz     .Lclamp_tail_loop
.Lclamp_done:
    vzeroupper
    ret
.size zzs_mllee_clamp_i8, .-zzs_mllee_clamp_i8

/* void zzs_mllee_masked_fill_i8(const int8_t *src, const int8_t *mask,
 *                               int8_t *dst, uint32_t count, int8_t fill) */
.globl zzs_mllee_masked_fill_i8
.type zzs_mllee_masked_fill_i8, @function
zzs_mllee_masked_fill_i8:
    vpbroadcastb ymm7, BYTE PTR [rsp + 8]
    vpxor   ymm6, ymm6, ymm6
    cmp     ecx, 32
    jb      .Lmf_tail
    .p2align 4
.Lmf_loop:
    vmovdqu ymm0, YMMWORD PTR [rdi]
    vmovdqu ymm1, YMMWORD PTR [rsi]
    vpcmpeqb ymm1, ymm1, ymm6          /* 0xFF where the mask is zero */
    vpblendvb ymm0, ymm7, ymm0, ymm1   /* keep src where mask == 0     */
    vmovdqu YMMWORD PTR [rdx], ymm0
    add     rdi, 32
    add     rsi, 32
    add     rdx, 32
    sub     ecx, 32
    cmp     ecx, 32
    jae     .Lmf_loop
.Lmf_tail:
    test    ecx, ecx
    jz      .Lmf_done
    movsx   r8d, BYTE PTR [rsp + 8]
.Lmf_tail_loop:
    movsx   eax, BYTE PTR [rdi]
    cmp     BYTE PTR [rsi], 0
    cmovne  eax, r8d
    mov     BYTE PTR [rdx], al
    inc     rdi
    inc     rsi
    inc     rdx
    dec     ecx
    jnz     .Lmf_tail_loop
.Lmf_done:
    vzeroupper
    ret
.size zzs_mllee_masked_fill_i8, .-zzs_mllee_masked_fill_i8

/* int32_t zzs_mllee_sum_sq_i8(const int8_t *values, uint32_t count) */
.globl zzs_mllee_sum_sq_i8
.type zzs_mllee_sum_sq_i8, @function
zzs_mllee_sum_sq_i8:
    vpxor   ymm4, ymm4, ymm4
    vpxor   ymm7, ymm7, ymm7
    xor     eax, eax
    cmp     esi, 16
    jb      .Lsq_tail
    .p2align 4
.Lsq_loop:
    vmovdqu xmm0, XMMWORD PTR [rdi]
    vpmovsxbw ymm0, xmm0
    vpmaddwd ymm0, ymm0, ymm0
    vpaddd  ymm4, ymm4, ymm0
    add     rdi, 16
    sub     esi, 16
    cmp     esi, 16
    jae     .Lsq_loop
    vextracti128 xmm0, ymm4, 1
    vpaddd  xmm0, xmm0, xmm4
    vpshufd xmm1, xmm0, 0x4E
    vpaddd  xmm0, xmm0, xmm1
    vpshufd xmm1, xmm0, 0xB1
    vpaddd  xmm0, xmm0, xmm1
    vmovd   eax, xmm0
    jmp     .Lsq_tail
.Lsq_tail:
    test    esi, esi
    jz      .Lsq_done
.Lsq_tail_loop:
    movsx   ecx, BYTE PTR [rdi]
    imul    ecx, ecx
    add     eax, ecx
    inc     rdi
    dec     esi
    jnz     .Lsq_tail_loop
.Lsq_done:
    vzeroupper
    ret
.size zzs_mllee_sum_sq_i8, .-zzs_mllee_sum_sq_i8

.section .note.GNU-stack,"",@progbits
