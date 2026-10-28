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
 * Optional AVX2 int8 dot product.  Selected at build time by
 * -DOPENZZS_MLLEE_AVX2=ON; never probed at run time.
 */

.intel_syntax noprefix
.text

/* int32_t zzs_mllee_dot_i8_avx2(const int8_t *lhs, const int8_t *rhs, uint32_t count) */
.globl zzs_mllee_dot_i8_avx2
.type zzs_mllee_dot_i8_avx2, @function
zzs_mllee_dot_i8_avx2:
    xor eax, eax
    vpxor ymm4, ymm4, ymm4
    cmp edx, 16
    jb .Ldot_avx2_tail
.Ldot_avx2_loop:
    vpmovsxbw ymm0, XMMWORD PTR [rdi]
    vpmovsxbw ymm1, XMMWORD PTR [rsi]
    vpmaddwd ymm0, ymm0, ymm1
    vpaddd ymm4, ymm4, ymm0
    add rdi, 16
    add rsi, 16
    sub edx, 16
    cmp edx, 16
    jae .Ldot_avx2_loop
    vextracti128 xmm0, ymm4, 1
    vpaddd xmm0, xmm0, xmm4
    vpshufd xmm1, xmm0, 0x4E
    vpaddd xmm0, xmm0, xmm1
    vpshufd xmm1, xmm0, 0xB1
    vpaddd xmm0, xmm0, xmm1
    vmovd ecx, xmm0
    add eax, ecx
    vzeroupper
.Ldot_avx2_tail:
    test edx, edx
    jz .Ldot_avx2_done
.Ldot_avx2_tail_loop:
    movsx ecx, BYTE PTR [rdi]
    movsx r8d, BYTE PTR [rsi]
    imul ecx, r8d
    add eax, ecx
    inc rdi
    inc rsi
    dec edx
    jnz .Ldot_avx2_tail_loop
.Ldot_avx2_done:
    ret
.size zzs_mllee_dot_i8_avx2, .-zzs_mllee_dot_i8_avx2

.section .note.GNU-stack,"",@progbits
