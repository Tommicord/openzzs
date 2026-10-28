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
 * mllee x86-64 packlow / quantized-weight expansion (SSSE3).
 *
 * The packed weight path reads int4 and float4 codes: two signed nibbles per
 * byte, low nibble first.  The portable form does that per code with a shift, a
 * mask and a sign test -- sequential bit extraction.  Here pshufb is used as a
 * 16-entry lookup table instead, so one shuffle both splits the byte and maps a
 * nibble onto its int8 code, and for a float4 layer the very same shuffle
 * applies the model's own flash table.
 *
 * Selected at configure time by -DOPENZZS_MLLEE_SSSE3=ON, never by CPUID at run
 * time.  Leaf routines: no stack frame, no callee-saved registers, no writable
 * static, and no data-dependent branches except the fixed tail loop.
 */

.intel_syntax noprefix

.section .rodata
.align 16
/* int4 nibble -> int8 code: 0..7 stay, 8..15 become -8..-1. */
.Lpk_int4_table:
    .byte 0, 1, 2, 3, 4, 5, 6, 7
    .byte 0xF8, 0xF9, 0xFA, 0xFB, 0xFC, 0xFD, 0xFE, 0xFF
.Lpk_nibble:
    .byte 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F
    .byte 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F
.Lpk_lane_even:
    .byte 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00
    .byte 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00
.Lpk_lane_odd:
    .byte 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF
    .byte 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0xFF

.text

/*
 * void zzs_mllee_expand_codes_ssse3(const uint8_t *source, int8_t *destination,
 *                                   uint32_t bytes, const int8_t *lut16)
 *
 * Expand `bytes` packed bytes into 2*bytes int8 codes, low nibble first.  When
 * `lut16` is not null it is the model's 16-entry float4 flash table; otherwise
 * the built-in int4 table decodes the nibble.  Eight source bytes -- sixteen
 * codes -- per shuffle group.
 */
/*
 * void zzs_mllee_expand_codes_ssse3(const uint8_t *source, int8_t *destination,
 *                                   uint32_t bytes, const int8_t *lut16)
 *
 * Expand `bytes` packed bytes into 2*bytes int8 codes, low nibble first.  When
 * `lut16` is not null it is the model's 16-entry float4 flash table; otherwise
 * the built-in int4 table decodes the nibble.  Eight source bytes -- sixteen
 * codes -- per shuffle group.
 */
.globl zzs_mllee_expand_codes_ssse3
.type zzs_mllee_expand_codes_ssse3, @function
zzs_mllee_expand_codes_ssse3:
    lea r8, [rip + .Lpk_int4_table]
    test rcx, rcx
    jz .Lexp_arm
    mov r8, rcx
.Lexp_arm:
    movdqu xmm4, XMMWORD PTR [r8]     /* 16-entry nibble table              */
    movdqa xmm5, XMMWORD PTR [rip + .Lpk_nibble]
    movdqa xmm6, XMMWORD PTR [rip + .Lpk_lane_even]
    movdqa xmm7, XMMWORD PTR [rip + .Lpk_lane_odd]
    cmp edx, 8
    jb .Lexp_tail
.Lexp_loop:
    movq xmm0, QWORD PTR [rdi]        /* b0..b7                            */
    punpcklbw xmm0, xmm0             /* b0 b0 b1 b1 ... b7 b7              */
    movdqa xmm1, xmm0
    psrlw xmm1, 4                    /* high nibble now in every lane       */
    pand xmm0, xmm5                  /* low-nibble indices, bit7 cleared    */
    pand xmm1, xmm5                  /* high-nibble indices                 */
    movdqa xmm2, xmm0
    movdqa xmm3, xmm1
    pshufb xmm2, xmm4                /* lo(b0) lo(b0) lo(b1) ...            */
    pshufb xmm3, xmm4                /* hi(b0) hi(b0) hi(b1) ...            */
    pand xmm2, xmm6                  /* keep the even (low) lanes           */
    pand xmm3, xmm7                  /* keep the odd (high) lanes           */
    por xmm2, xmm3
    movdqu XMMWORD PTR [rsi], xmm2
    add rdi, 8
    add rsi, 16
    sub edx, 8
    cmp edx, 8
    jae .Lexp_loop
.Lexp_tail:
    test edx, edx
    jz .Lexp_done
.Lexp_tail_loop:
    movzx eax, BYTE PTR [rdi]
    movzx ecx, al
    and ecx, 0x0F
    movzx ecx, BYTE PTR [r8 + rcx]
    mov BYTE PTR [rsi], cl
    shr eax, 4
    and eax, 0x0F
    movzx eax, BYTE PTR [r8 + rax]
    mov BYTE PTR [rsi + 1], al
    inc rdi
    add rsi, 2
    dec edx
    jnz .Lexp_tail_loop
.Lexp_done:
    ret
.size zzs_mllee_expand_codes_ssse3, .-zzs_mllee_expand_codes_ssse3

/*
 * void zzs_mllee_gather_lut8(const int8_t *lut256, const uint8_t *codes,
 *                            int8_t *destination, uint32_t count)
 *
 * Float8 weight codes address a 256-entry flash table.  pshufb indexes only 16
 * lanes, so a byte-wide gather cannot be a single shuffle; this is the four-way
 * unrolled gather instead.  The body holds no data-dependent branch, which is
 * the property that keeps the timing flat, and mllee keeps the vector form for
 * the int4/float4 tables where the lane count does allow one.
 */
.globl zzs_mllee_gather_lut8
.type zzs_mllee_gather_lut8, @function
zzs_mllee_gather_lut8:
    cmp ecx, 4
    jb .Lgather_tail
.Lgather_loop:
    movzx r8d, BYTE PTR [rsi]
    movzx r9d, BYTE PTR [rsi + 1]
    movzx r10d, BYTE PTR [rsi + 2]
    movzx r11d, BYTE PTR [rsi + 3]
    movzx eax, BYTE PTR [rdi + r8]
    movzx r8d, BYTE PTR [rdi + r9]
    movzx r9d, BYTE PTR [rdi + r10]
    movzx r10d, BYTE PTR [rdi + r11]
    shl r8d, 8
    or eax, r8d
    shl r9d, 16
    or eax, r9d
    shl r10d, 24
    or eax, r10d
    mov DWORD PTR [rdx], eax
    add rsi, 4
    add rdx, 4
    sub ecx, 4
    cmp ecx, 4
    jae .Lgather_loop
.Lgather_tail:
    test ecx, ecx
    jz .Lgather_done
.Lgather_tail_loop:
    movzx eax, BYTE PTR [rsi]
    movzx eax, BYTE PTR [rdi + rax]
    mov BYTE PTR [rdx], al
    inc rsi
    inc rdx
    dec ecx
    jnz .Lgather_tail_loop
.Lgather_done:
    ret
.size zzs_mllee_gather_lut8, .-zzs_mllee_gather_lut8

.section .note.GNU-stack,"",@progbits
