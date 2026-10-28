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
 * AVX2 int8 dot product, selected at build time by -DOPENZZS_MLLEE_AVX2=ON and
 * never probed at run time.
 *
 * Both operands are signed, so every lane stays on the sign-extend path:
 * `vpmovsxbw` widens 16 int8 codes to 16 int16 lanes and `vpmaddwd` multiplies
 * adjacent pairs into int32.  A pair sums at most 2 * 127 * 127 = 32258, which
 * cannot overflow the int32 lane, so the kernel is exact and never saturates --
 * unlike a `vpmaddubsw` form, whose unsigned x signed lanes clamp at 32767 as
 * soon as two biased products land in the same word.
 *
 * Two accumulator chains let the second widen/multiply pair issue while the
 * first `vpaddd` is still retiring; the horizontal fold runs once at the end.
 * The int32 accumulator is exact while |sum| fits, i.e. for
 * count <= 2147483647 / 16129 = 133143 codes, which covers every row the blob
 * format can describe (its dimensions are uint16).
 */

.intel_syntax noprefix
.text

/* int32_t zzs_mllee_dot_i8_avx2(const int8_t *lhs, const int8_t *rhs, uint32_t count) */
.globl zzs_mllee_dot_i8_avx2
.type zzs_mllee_dot_i8_avx2, @function
zzs_mllee_dot_i8_avx2:
    vpxor   xmm4, xmm4, xmm4          /* accumulator A */
    vpxor   xmm5, xmm5, xmm5          /* accumulator B */
    cmp     edx, 32
    jb      .Ldot_avx2_16

    .p2align 4
.Ldot_avx2_32:
    vpmovsxbw ymm6, XMMWORD PTR [rdi]
    vpmovsxbw ymm7, XMMWORD PTR [rsi]
    vpmaddwd  ymm6, ymm6, ymm7
    vpaddd    ymm4, ymm4, ymm6
    vpmovsxbw ymm6, XMMWORD PTR [rdi + 16]
    vpmovsxbw ymm7, XMMWORD PTR [rsi + 16]
    vpmaddwd  ymm6, ymm6, ymm7
    vpaddd    ymm5, ymm5, ymm6
    add       rdi, 32
    add       rsi, 32
    sub       edx, 32
    cmp       edx, 32
    jae       .Ldot_avx2_32

.Ldot_avx2_16:
    cmp     edx, 16
    jb      .Ldot_avx2_fold
    vpmovsxbw ymm6, XMMWORD PTR [rdi]
    vpmovsxbw ymm7, XMMWORD PTR [rsi]
    vpmaddwd  ymm6, ymm6, ymm7
    vpaddd    ymm4, ymm4, ymm6
    add       rdi, 16
    add       rsi, 16
    sub       edx, 16

.Ldot_avx2_fold:
    vpaddd    ymm4, ymm4, ymm5
    vextracti128 xmm6, ymm4, 1
    vpaddd    xmm6, xmm6, xmm4
    vpshufd   xmm7, xmm6, 0x4E        /* swap 64-bit halves */
    vpaddd    xmm6, xmm6, xmm7
    vpshufd   xmm7, xmm6, 0xB1        /* swap adjacent dwords */
    vpaddd    xmm6, xmm6, xmm7
    vmovd     eax, xmm6
    vzeroupper

    test    edx, edx
    jz      .Ldot_avx2_done
.Ldot_avx2_tail:
    movsx   ecx, BYTE PTR [rdi]
    movsx   r8d, BYTE PTR [rsi]
    imul    ecx, r8d
    add     eax, ecx
    inc     rdi
    inc     rsi
    dec     edx
    jnz     .Ldot_avx2_tail
.Ldot_avx2_done:
    ret
.size zzs_mllee_dot_i8_avx2, .-zzs_mllee_dot_i8_avx2

.section .note.GNU-stack,"",@progbits
