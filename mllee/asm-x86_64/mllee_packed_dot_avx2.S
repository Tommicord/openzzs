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
 * AVX2 packed-weight dot products for the 2- and 4-bit formats.
 *
 * A dot over a packed row used to cost one range-decode, one nibble extract, a
 * table lookup and one scalar multiply per code.  These kernels expand the
 * packed codes in registers with pshufb and feed the resulting int8 codes to
 * the same sign-extending vpmaddwd fold as the int8 kernel, so no scratch
 * buffer (and no stack array) is needed:
 *
 *   i4  one pshufb pair per 16 packed bytes -> 32 codes
 *   i2  four pshufb of the four bit fields -> 64 codes from 16 bytes
 *
 * `lut` is the model's 16-entry (i4/f4) or 4-entry (i2) code->int8 table.  A
 * null LUT selects the two's-complement integer mapping, exactly like the
 * scalar `_S_from_int_bits` path.  Codes are consumed low-bits-first, the
 * layout the exporter and the packlow reader share.
 *
 * The accumulators are int32 and exact for every row the blob can describe
 * (dimensions are uint16; the worst-case product is 16129 per code).
 */

.intel_syntax noprefix

.section .rodata
    .p2align 4
.Lpd_mask0f:
    .fill 16, 1, 0x0F
.Lpd_mask03:
    .fill 16, 1, 0x03
    .p2align 4
.Lpd_i4_table:
    .byte 0, 1, 2, 3, 4, 5, 6, 7
    .byte -8, -7, -6, -5, -4, -3, -2, -1
.Lpd_i2_table:
    .byte 0, 1, -2, -1
    .fill 12, 1, 0

.text

/* int32_t zzs_mllee_dot_i4_avx2(const uint8_t *packed, const int8_t *input,
 *                               uint32_t codes, const int8_t *lut16) */
.globl zzs_mllee_dot_i4_avx2
.type zzs_mllee_dot_i4_avx2, @function
zzs_mllee_dot_i4_avx2:
    test    ecx, ecx
    jz      .Lpd_i4_default
    vmovdqu xmm1, XMMWORD PTR [rcx]
    lea     r8, [rcx]
    jmp     .Lpd_i4_table_ready
.Lpd_i4_default:
    vmovdqa xmm1, XMMWORD PTR [rip + .Lpd_i4_table]
    lea     r8, [rip + .Lpd_i4_table]
.Lpd_i4_table_ready:
    vmovdqa xmm7, XMMWORD PTR [rip + .Lpd_mask0f]
    vpxor   xmm0, xmm0, xmm0
    cmp     edx, 32
    jb      .Lpd_i4_tail

    .p2align 4
.Lpd_i4_loop:
    vmovdqu xmm2, XMMWORD PTR [rdi]
    vpand   xmm3, xmm2, xmm7           /* low nibbles  */
    vpsrlw  xmm2, xmm2, 4
    vpand   xmm2, xmm2, xmm7           /* high nibbles */
    vpshufb xmm3, xmm1, xmm3           /* c0, c2, ...  */
    vpshufb xmm2, xmm1, xmm2           /* c1, c3, ...  */
    vpunpcklbw xmm4, xmm3, xmm2        /* c0..c15      */
    vpunpckhbw xmm5, xmm3, xmm2        /* c16..c31     */
    vpmovsxbw ymm4, xmm4
    vpmovsxbw ymm6, XMMWORD PTR [rsi]
    vpmaddwd  ymm4, ymm4, ymm6
    vpmovsxbw ymm5, xmm5
    vpmovsxbw ymm6, XMMWORD PTR [rsi + 16]
    vpmaddwd  ymm5, ymm5, ymm6
    vpaddd    ymm4, ymm4, ymm5
    vpaddd    ymm0, ymm0, ymm4
    add     rdi, 16
    add     rsi, 32
    sub     edx, 32
    cmp     edx, 32
    jae     .Lpd_i4_loop

.Lpd_i4_tail:
    /* Fold the vector accumulator to one int32. */
    vextracti128 xmm2, ymm0, 1
    vpaddd  xmm0, xmm0, xmm2
    vpshufd xmm2, xmm0, 0x4E
    vpaddd  xmm0, xmm0, xmm2
    vpshufd xmm2, xmm0, 0xB1
    vpaddd  xmm0, xmm0, xmm2
    vmovd   r9d, xmm0
    /* Two codes per byte, low nibble first; an odd final code is ignored. */
    test    edx, edx
    jz      .Lpd_i4_done
    .p2align 4
.Lpd_i4_pair:
    movzx   r10d, BYTE PTR [rdi]
    mov     ecx, r10d
    and     ecx, 15
    movsx   ecx, BYTE PTR [r8 + rcx]
    movsx   r11d, BYTE PTR [rsi]
    imul    ecx, r11d
    add     r9d, ecx
    inc     rsi
    dec     edx
    jz      .Lpd_i4_done
    shr     r10d, 4
    movsx   ecx, BYTE PTR [r8 + r10]
    movsx   r11d, BYTE PTR [rsi]
    imul    ecx, r11d
    add     r9d, ecx
    inc     rsi
    inc     rdi
    sub     edx, 1
    jnz     .Lpd_i4_pair
.Lpd_i4_done:
    mov     eax, r9d
    vzeroupper
    ret
.size zzs_mllee_dot_i4_avx2, .-zzs_mllee_dot_i4_avx2

/* int32_t zzs_mllee_dot_i2_avx2(const uint8_t *packed, const int8_t *input,
 *                               uint32_t codes, const int8_t *lut4) */
.globl zzs_mllee_dot_i2_avx2
.type zzs_mllee_dot_i2_avx2, @function
zzs_mllee_dot_i2_avx2:
    test    ecx, ecx
    jz      .Lpd_i2_default
    vmovdqu xmm1, XMMWORD PTR [rcx]
    lea     r8, [rcx]
    jmp     .Lpd_i2_table_ready
.Lpd_i2_default:
    vmovdqa xmm1, XMMWORD PTR [rip + .Lpd_i2_table]
    lea     r8, [rip + .Lpd_i2_table]
.Lpd_i2_table_ready:
    vmovdqa xmm7, XMMWORD PTR [rip + .Lpd_mask03]
    vpxor   xmm0, xmm0, xmm0
    vpxor   xmm6, xmm6, xmm6
    cmp     edx, 64
    jb      .Lpd_i2_tail

    .p2align 4
.Lpd_i2_loop:
    vmovdqu xmm2, XMMWORD PTR [rdi]    /* 16 bytes -> 64 codes */
    vpand   xmm3, xmm2, xmm7
    vpsrlw  xmm4, xmm2, 2
    vpand   xmm4, xmm4, xmm7
    vpsrlw  xmm5, xmm2, 4
    vpand   xmm5, xmm5, xmm7
    vpsrlw  xmm2, xmm2, 6
    vpand   xmm2, xmm2, xmm7
    vpshufb xmm3, xmm1, xmm3           /* field 0 */
    vpshufb xmm4, xmm1, xmm4           /* field 1 */
    vpshufb xmm5, xmm1, xmm5           /* field 2 */
    vpshufb xmm2, xmm1, xmm2           /* field 3 */
    vpunpcklbw xmm8, xmm3, xmm4        /* f0,f1 bytes 0..7  */
    vpunpckhbw xmm3, xmm3, xmm4        /* f0,f1 bytes 8..15 */
    vpunpcklbw xmm9, xmm5, xmm2        /* f2,f3 bytes 0..7  */
    vpunpckhbw xmm5, xmm5, xmm2        /* f2,f3 bytes 8..15 */
    vpunpcklwd xmm4, xmm8, xmm9        /* codes 0..15   */
    vpunpckhwd xmm8, xmm8, xmm9        /* codes 16..31  */
    vpunpcklwd xmm2, xmm3, xmm5        /* codes 32..47  */
    vpunpckhwd xmm3, xmm3, xmm5        /* codes 48..63  */

    vpmovsxbw ymm4, xmm4
    vpmovsxbw ymm9, XMMWORD PTR [rsi]
    vpmaddwd  ymm4, ymm4, ymm9
    vpaddd    ymm0, ymm0, ymm4

    vpmovsxbw ymm8, xmm8
    vpmovsxbw ymm9, XMMWORD PTR [rsi + 16]
    vpmaddwd  ymm8, ymm8, ymm9
    vpaddd    ymm6, ymm6, ymm8

    vpmovsxbw ymm2, xmm2
    vpmovsxbw ymm9, XMMWORD PTR [rsi + 32]
    vpmaddwd  ymm2, ymm2, ymm9
    vpaddd    ymm0, ymm0, ymm2

    vpmovsxbw ymm3, xmm3
    vpmovsxbw ymm9, XMMWORD PTR [rsi + 48]
    vpmaddwd  ymm3, ymm3, ymm9
    vpaddd    ymm6, ymm6, ymm3

    add     rdi, 16
    add     rsi, 64
    sub     edx, 64
    cmp     edx, 64
    jae     .Lpd_i2_loop

.Lpd_i2_tail:
    vpaddd  ymm0, ymm0, ymm6
    vextracti128 xmm2, ymm0, 1
    vpaddd  xmm0, xmm0, xmm2
    vpshufd xmm2, xmm0, 0x4E
    vpaddd  xmm0, xmm0, xmm2
    vpshufd xmm2, xmm0, 0xB1
    vpaddd  xmm0, xmm0, xmm2
    vmovd   r9d, xmm0
    test    edx, edx
    jz      .Lpd_i2_done
    .p2align 4
.Lpd_i2_quad:
    movzx   r10d, BYTE PTR [rdi]
    mov     ecx, r10d
    and     ecx, 3
    movsx   ecx, BYTE PTR [r8 + rcx]
    movsx   r11d, BYTE PTR [rsi]
    imul    ecx, r11d
    add     r9d, ecx
    inc     rsi
    dec     edx
    jz      .Lpd_i2_done
    shr     r10d, 2
    mov     ecx, r10d
    and     ecx, 3
    movsx   ecx, BYTE PTR [r8 + rcx]
    movsx   r11d, BYTE PTR [rsi]
    imul    ecx, r11d
    add     r9d, ecx
    inc     rsi
    dec     edx
    jz      .Lpd_i2_done
    shr     r10d, 2
    mov     ecx, r10d
    and     ecx, 3
    movsx   ecx, BYTE PTR [r8 + rcx]
    movsx   r11d, BYTE PTR [rsi]
    imul    ecx, r11d
    add     r9d, ecx
    inc     rsi
    dec     edx
    jz      .Lpd_i2_done
    shr     r10d, 2
    and     r10d, 3
    movsx   ecx, BYTE PTR [r8 + r10]
    movsx   r11d, BYTE PTR [rsi]
    imul    ecx, r11d
    add     r9d, ecx
    inc     rsi
    inc     rdi
    sub     edx, 1
    jnz     .Lpd_i2_quad
.Lpd_i2_done:
    mov     eax, r9d
    vzeroupper
    ret
.size zzs_mllee_dot_i2_avx2, .-zzs_mllee_dot_i2_avx2

.section .note.GNU-stack,"",@progbits
