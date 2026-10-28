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
 * AVX-512 VNNI int8 dot product, selected at build time by
 * -DOPENZZS_MLLEE_AVX512=ON (-mavx512f -mavx512bw -mavx512vnni).  Like the
 * AVX-VNNI kernel, vpdpbusd takes an unsigned lhs and a signed rhs, so the
 * signed lhs is rebased with XOR 0x80 and a second accumulator subtracts
 * 128 * sum(rhs) once at the end:
 *
 *     sum (a ^ 0x80) * b = sum a*b + 128 * sum b
 *
 * The 512-bit form holds four products per dword lane over 64 bytes per
 * instruction.  Two accumulator pairs (dot and sum-of-rhs) keep both ports
 * busy; the horizontal fold runs once.
 *
 * Exactness window: the biased lane stays inside int32 while
 * count <= 2147483647 / (255 * 128) = 65793 codes, the same bound as the
 * AVX-VNNI form.  Every blob row is far below that (dimensions are uint16,
 * staged conv rows are 64 codes); longer spans fall back to the portable path
 * in mllee/simd.hpp.
 */

.intel_syntax noprefix
.text

/* int32_t zzs_mllee_dot_i8_avx512(const int8_t *lhs, const int8_t *rhs,
 *                                  uint32_t count) */
.globl zzs_mllee_dot_i8_avx512
.type zzs_mllee_dot_i8_avx512, @function
zzs_mllee_dot_i8_avx512:
    vpxord  zmm2, zmm2, zmm2           /* biased dot accumulator A */
    vpxord  zmm3, zmm3, zmm3           /* sum-of-rhs accumulator A */
    vpxord  zmm4, zmm4, zmm4           /* biased dot accumulator B */
    vpxord  zmm5, zmm5, zmm5           /* sum-of-rhs accumulator B */
    vpbroadcastb zmm8, BYTE PTR [rip + .Lavx512_bias]
    vpbroadcastb zmm9, BYTE PTR [rip + .Lavx512_one]
    cmp     edx, 128
    jb      .Lavx512_64

    .p2align 4
.Lavx512_128:
    vmovdqu8 zmm0, ZMMWORD PTR [rdi]
    vmovdqu8 zmm1, ZMMWORD PTR [rsi]
    vpxord   zmm0, zmm0, zmm8
    vpdpbusd zmm2, zmm0, zmm1
    vpdpbusd zmm3, zmm9, zmm1
    vmovdqu8 zmm6, ZMMWORD PTR [rdi + 64]
    vmovdqu8 zmm7, ZMMWORD PTR [rsi + 64]
    vpxord   zmm6, zmm6, zmm8
    vpdpbusd zmm4, zmm6, zmm7
    vpdpbusd zmm5, zmm9, zmm7
    add     rdi, 128
    add     rsi, 128
    sub     edx, 128
    cmp     edx, 128
    jae     .Lavx512_128

.Lavx512_64:
    cmp     edx, 64
    jb      .Lavx512_32
    vmovdqu8 zmm0, ZMMWORD PTR [rdi]
    vmovdqu8 zmm1, ZMMWORD PTR [rsi]
    vpxord   zmm0, zmm0, zmm8
    vpdpbusd zmm2, zmm0, zmm1
    vpdpbusd zmm3, zmm9, zmm1
    add     rdi, 64
    add     rsi, 64
    sub     edx, 64

.Lavx512_32:
    cmp     edx, 32
    jb      .Lavx512_16
    vmovdqu ymm0, YMMWORD PTR [rdi]
    vmovdqu ymm1, YMMWORD PTR [rsi]
    vpxor   ymm0, ymm0, ymm8
    vpdpbusd ymm2, ymm0, ymm1
    vpdpbusd ymm3, ymm9, ymm1
    add     rdi, 32
    add     rsi, 32
    sub     edx, 32

.Lavx512_16:
    cmp     edx, 16
    jb      .Lavx512_fold
    vmovdqu xmm0, XMMWORD PTR [rdi]
    vmovdqu xmm1, XMMWORD PTR [rsi]
    vpxor   xmm0, xmm0, xmm8
    vpdpbusd xmm2, xmm0, xmm1
    vpdpbusd xmm3, xmm9, xmm1
    add     rdi, 16
    add     rsi, 16
    sub     edx, 16

.Lavx512_fold:
    vpaddd  zmm2, zmm2, zmm4           /* coarse dot fold */
    vpaddd  zmm3, zmm3, zmm5           /* coarse sum fold */
    vextracti64x4 ymm6, zmm2, 1
    vpaddd  ymm2, ymm2, ymm6
    vextracti128 xmm6, ymm2, 1
    vpaddd  xmm2, xmm2, xmm6
    vpshufd xmm6, xmm2, 0x4E
    vpaddd  xmm2, xmm2, xmm6
    vpshufd xmm6, xmm2, 0xB1
    vpaddd  xmm2, xmm2, xmm6
    vmovd   r9d, xmm2

    vextracti64x4 ymm6, zmm3, 1
    vpaddd  ymm3, ymm3, ymm6
    vextracti128 xmm6, ymm3, 1
    vpaddd  xmm3, xmm3, xmm6
    vpshufd xmm6, xmm3, 0x4E
    vpaddd  xmm3, xmm3, xmm6
    vpshufd xmm6, xmm3, 0xB1
    vpaddd  xmm3, xmm3, xmm6
    vmovd   r10d, xmm3

    shl     r10d, 7                    /* 128 * sum(rhs) */
    sub     r9d, r10d

    test    edx, edx
    jz      .Lavx512_done
    .p2align 4
.Lavx512_tail:
    movsx   ecx, BYTE PTR [rdi]
    movsx   r8d, BYTE PTR [rsi]
    imul    ecx, r8d
    add     r9d, ecx
    inc     rdi
    inc     rsi
    dec     edx
    jnz     .Lavx512_tail
.Lavx512_done:
    mov     eax, r9d
    vzeroupper
    ret
.size zzs_mllee_dot_i8_avx512, .-zzs_mllee_dot_i8_avx512

.section .rodata
.Lavx512_bias:
    .byte 0x80
.Lavx512_one:
    .byte 0x01

.section .note.GNU-stack,"",@progbits
