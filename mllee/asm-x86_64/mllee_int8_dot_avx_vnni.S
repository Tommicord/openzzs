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
 * AVX-VNNI int8 dot product (four products per dword lane, no widening),
 * selected at build time by -DOPENZZS_MLLEE_AVX_VNNI=ON and never probed at
 * run time.  It implies AVX2, so the build adds -mavxvnni -mavx2.
 *
 * VPDPBUSD multiplies an UNSIGNED byte vector by a SIGNED byte vector and
 * accumulates four products into each int32 lane, so the signed lhs has to be
 * rebased.  XOR with 0x80 is exactly that rebase: for a in [-128, 127] the
 * byte a ^ 0x80 is the unsigned value a + 128, and
 *
 *     sum (a + 128) * b  =  sum a * b  +  128 * sum b
 *
 * The second accumulator therefore tracks sum b with the same instruction --
 * ones as the unsigned operand, b as the signed one -- and the correction is
 * applied once after the horizontal fold.  No product is ever clamped: the
 * dword lanes hold signed sums and cannot saturate before overflow.
 *
 * Exactness window: the biased accumulator swallows the bias term too, so its
 * magnitude is bounded by count * 255 * 128 = count * 32640 rather than by the
 * true dot product.  That keeps every lane inside int32 while count <= 65793,
 * far above the 444-code rows and 64-code conv windows this format can carry
 * (all blob dimensions are uint16).  Callers with longer spans get the portable
 * path; the range is documented in mllee/simd.hpp next to the dispatch.
 */

.intel_syntax noprefix
.text

/* int32_t zzs_mllee_dot_i8_avx_vnni(const int8_t *lhs, const int8_t *rhs,
 *                                    uint32_t count) */
.globl zzs_mllee_dot_i8_avx_vnni
.type zzs_mllee_dot_i8_avx_vnni, @function
zzs_mllee_dot_i8_avx_vnni:
    vpxor   xmm2, xmm2, xmm2          /* biased dot accumulator   */
    vpxor   xmm3, xmm3, xmm3          /* sum-of-rhs accumulator   */
    vmovdqa ymm8, YMMWORD PTR [rip + .Ldot_vnni_bias]
    vmovdqa ymm9, YMMWORD PTR [rip + .Ldot_vnni_ones]
    cmp     edx, 32
    jb      .Ldot_vnni_16

    .p2align 4
.Ldot_vnni_32:
    vmovdqu ymm0, YMMWORD PTR [rdi]
    vmovdqu ymm1, YMMWORD PTR [rsi]
    vpxor   ymm0, ymm0, ymm8          /* u = a + 128, unsigned     */
    vpdpbusd ymm2, ymm0, ymm1         /* sum (a + 128) * b         */
    vpdpbusd ymm3, ymm9, ymm1         /* sum 1 * b                 */
    add     rdi, 32
    add     rsi, 32
    sub     edx, 32
    cmp     edx, 32
    jae     .Ldot_vnni_32

.Ldot_vnni_16:
    cmp     edx, 16
    jb      .Ldot_vnni_fold
    vmovdqu xmm0, XMMWORD PTR [rdi]
    vmovdqu xmm1, XMMWORD PTR [rsi]
    vpxor   xmm0, xmm0, xmm8
    vpdpbusd xmm2, xmm0, xmm1
    vpdpbusd xmm3, xmm9, xmm1
    add     rdi, 16
    add     rsi, 16
    sub     edx, 16

.Ldot_vnni_fold:
    vextracti128 xmm4, ymm2, 1
    vpaddd    xmm4, xmm4, xmm2
    vpshufd   xmm5, xmm4, 0x4E
    vpaddd    xmm4, xmm4, xmm5
    vpshufd   xmm5, xmm4, 0xB1
    vpaddd    xmm4, xmm4, xmm5
    vmovd     ecx, xmm4               /* biased total             */

    vextracti128 xmm6, ymm3, 1
    vpaddd    xmm6, xmm6, xmm3
    vpshufd   xmm7, xmm6, 0x4E
    vpaddd    xmm6, xmm6, xmm7
    vpshufd   xmm7, xmm6, 0xB1
    vpaddd    xmm6, xmm6, xmm7
    vmovd     eax, xmm6               /* sum of the rhs codes     */

    imul    eax, eax, 128
    sub     ecx, eax
    mov     eax, ecx
    vzeroupper

    test    edx, edx
    jz      .Ldot_vnni_done
.Ldot_vnni_tail:
    movsx   ecx, BYTE PTR [rdi]
    movsx   r8d, BYTE PTR [rsi]
    imul    ecx, r8d
    add     eax, ecx
    inc     rdi
    inc     rsi
    dec     edx
    jnz     .Ldot_vnni_tail
.Ldot_vnni_done:
    ret
.size zzs_mllee_dot_i8_avx_vnni, .-zzs_mllee_dot_i8_avx_vnni

.section .rodata
    .p2align 5
.Ldot_vnni_bias:
    .fill 32, 1, 0x80
.Ldot_vnni_ones:
    .fill 32, 1, 0x01

.section .note.GNU-stack,"",@progbits
