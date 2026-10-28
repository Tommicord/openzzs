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
 * mllee x86-64 training kernels (SSE2 int16 lanes, the x86-64 architectural
 * floor).
 *
 * The interface in mllee/train_simd.hpp is frozen, so every kernel here mirrors
 * its portable fallback bit for bit -- tails, saturation edges and the zero-count
 * case included.  Eight int16 lanes travel through one xmm operation: pmullw and
 * pmulhw for the exact 16x16 products, pmulhuw for the unsigned high half a Q15
 * magnitude needs, pmuludq for the 32x32->64 products the code-step reciprocal
 * needs, and paddsw/psubsw/psraw/packssdw/packsswb for the lane arithmetic and
 * its rails.  Streams are read with movdqu, so nothing is peeled or aligned per
 * element and only the count % 8 tail runs scalar.
 *
 * Four shapes recur, and each one exists because the obvious shape is wrong.
 *
 *   - pmaddwd is not used for the dot product: its two-lane sum wraps at 2^31,
 *     two adjacent -32768 lanes hit exactly that, and the fallback keeps the
 *     running total in int64.  The eight int32 products are rebuilt from
 *     pmullw/pmulhw and flushed into the int64 lanes every group.
 *   - A Q15 multiply by an int32 scalar cannot use the scalar's low half as a
 *     lane: read as an int16 lane it is signed, so k = high * 65536 + low is not
 *     high * 65536 + lane(low).  Splitting k into m = 2 * high + (low >> 15) and
 *     v = low & 0x7FFF makes both halves lane-safe and gives
 *     round(a * k / 2^15) = a * m + round(a * v / 2^15) exactly.  |m| <= 32767
 *     needs |k| < 2^30, so a wider scalar takes the scalar body for the whole
 *     call -- one test per call, exactly like the alignment test, never per
 *     element.
 *   - Division by the 258-unit code step is folded into a multiply, so neither
 *     idiv nor a per-lane branch appears in the loop.  The reciprocal is the
 *     over-estimate ceil(2^s / 258), exact while value * (258 * M - 2^s) < 2^s:
 *     M = 4261672976 with s = 40 (error term 32) covers every int32 numerator.
 *     The exponent is capped at 40 because pmuludq reads only the low 32 bits of
 *     each operand, so an s = 41 reciprocal -- 8523345952, a 34-bit value --
 *     silently becomes its own low half and every code step comes out at half
 *     magnitude.  The fallback's +129 rides in as the constant 129 * M added to
 *     the 64-bit product, because |code step| may reach 2^31 - 1 and adding 129
 *     to that in an int32 lane would wrap.
 *   - The narrowing conversion is (|v| * 254 + 32768) >> 16, which is the
 *     fallback's (|v| * 127 + 16384) >> 15 with every term doubled.  That stays
 *     in 16-bit lanes -- pmulhuw gives the quotient and bit 15 of pmullw the one
 *     rounding step -- whereas floor((|v| + 129) / 258) would differ from the
 *     fallback for roughly one magnitude in eighty.
 *
 * Leaf routines: no stack frame and no red-zone use (the red zone belongs to the
 * caller's own leaf frame, so a leaf kernel must not touch memory below rsp), no
 * callee-saved register, no writable static, no CPUID, and count == 0
 * dereferences nothing.  An int8 stream moves eight bytes at a time through
 * movq, because a group of eight codes is eight bytes and a 16-byte access would
 * read or write past the caller's buffer.
 */

.intel_syntax noprefix

.section .rodata
.align 16
.Ltr_sign16:
    .word 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000, 0x8000
.Ltr_c258:
    .word 258, 258, 258, 258, 258, 258, 258, 258
.Ltr_c254:
    .word 254, 254, 254, 254, 254, 254, 254, 254
.Ltr_cm254:
    .word -254, -254, -254, -254, -254, -254, -254, -254
.Ltr_c127:
    .word 127, 127, 127, 127, 127, 127, 127, 127
.Ltr_cm127:
    .word -127, -127, -127, -127, -127, -127, -127, -127
.Ltr_c16384_32:
    .long 16384, 16384, 16384, 16384
.Ltr_magic40:
    .quad 4261672976, 4261672976
.Ltr_129magic40:
    .quad 549755813904, 549755813904

.text

/*
 * int32_t zzs_mllee_train_dot_q15(const int16_t *lhs, const int16_t *rhs,
 *                                 uint32_t count)
 *
 * The int64 lanes hold the running total, so a 4-billion-lane dot needs no
 * intermediate correction; the collapse is two movq and an add.
 */
.globl zzs_mllee_train_dot_q15
.type zzs_mllee_train_dot_q15, @function
zzs_mllee_train_dot_q15:
    pxor xmm4, xmm4                 /* zero, the widening operand             */
    pxor xmm6, xmm6                 /* int64 lane pair accumulator            */
    cmp edx, 8
    jb .Ltr_dot_collapse
.Ltr_dot_loop:
    movdqu xmm0, XMMWORD PTR [rdi]
    movdqu xmm1, XMMWORD PTR [rsi]
    movdqa xmm2, xmm0
    pmullw xmm0, xmm1               /* low halves of the eight products       */
    pmulhw xmm2, xmm1               /* high halves; 16x16 never needs 33 bits */
    movdqa xmm3, xmm0
    punpcklwd xmm0, xmm2            /* products 0..3 as int32 lanes           */
    punpckhwd xmm3, xmm2            /* products 4..7                          */
    movdqa xmm5, xmm0
    movdqa xmm1, xmm0
    psrad xmm1, 31
    punpckldq xmm0, xmm1            /* the sign dword widens each product     */
    punpckhdq xmm5, xmm1
    paddq xmm6, xmm0
    paddq xmm6, xmm5
    movdqa xmm5, xmm3
    movdqa xmm1, xmm3
    psrad xmm1, 31
    punpckldq xmm3, xmm1
    punpckhdq xmm5, xmm1
    paddq xmm6, xmm3
    paddq xmm6, xmm5
    add rdi, 16
    add rsi, 16
    sub edx, 8
    cmp edx, 8
    jae .Ltr_dot_loop
.Ltr_dot_collapse:
    movq rax, xmm6
    psrldq xmm6, 8
    movq rcx, xmm6
    add rax, rcx
    test edx, edx
    jz .Ltr_dot_round
.Ltr_dot_tail:
    movsx rcx, WORD PTR [rdi]
    movsx r8, WORD PTR [rsi]
    imul rcx, r8
    add rax, rcx
    add rdi, 2
    add rsi, 2
    dec edx
    jnz .Ltr_dot_tail
.Ltr_dot_round:
    add rax, 16384
    sar rax, 15
    mov rcx, 2147483647
    cmp rax, rcx
    cmovg rax, rcx
    movabs rcx, -2147483648
    cmp rax, rcx
    cmovl rax, rcx
    ret
    .size zzs_mllee_train_dot_q15, .-zzs_mllee_train_dot_q15

/*
 * uint32_t zzs_mllee_train_add_grad_q15(int16_t *destination,
 *                                       const int16_t *source, uint32_t count)
 *
 * paddsw saturates silently, so the clamp is read off the *wrapping* sum:
 * ((a ^ sum) & (b ^ sum)) carries the overflow decision in bit 15, and only in
 * bit 15 -- the low bits of that witness are noise, which is why it is masked
 * before psadbw turns one 0x80 byte per clamped lane into lanes * 128.  The
 * shift at the end is the whole count reduction: no popcount, no per-lane
 * branch, and only the saturating sum is ever stored.
 */
.globl zzs_mllee_train_add_grad_q15
.type zzs_mllee_train_add_grad_q15, @function
zzs_mllee_train_add_grad_q15:
    movdqa xmm7, XMMWORD PTR [rip + .Ltr_sign16]
    pxor xmm5, xmm5                 /* psadbw's zero operand                  */
    pxor xmm6, xmm6                 /* lane counter, one 0x80 per clamp       */
    xor eax, eax
    cmp edx, 8
    jb .Ltr_add_collapse
.Ltr_add_loop:
    movdqu xmm0, XMMWORD PTR [rdi]
    movdqu xmm1, XMMWORD PTR [rsi]
    movdqa xmm2, xmm0
    paddw xmm2, xmm1                /* wrapping sum, the overflow witness     */
    movdqa xmm3, xmm0
    pxor xmm3, xmm2
    movdqa xmm4, xmm1
    pxor xmm4, xmm2
    pand xmm3, xmm4
    pand xmm3, xmm7
    paddsw xmm0, xmm1
    movdqu XMMWORD PTR [rdi], xmm0
    psadbw xmm3, xmm5
    paddq xmm6, xmm3
    add rdi, 16
    add rsi, 16
    sub edx, 8
    cmp edx, 8
    jae .Ltr_add_loop
.Ltr_add_collapse:
    movq rax, xmm6
    psrldq xmm6, 8
    movq rcx, xmm6
    add rax, rcx
    shr rax, 7
    test edx, edx
    jz .Ltr_add_done
.Ltr_add_tail:
    movsx ecx, WORD PTR [rdi]
    movsx r8d, WORD PTR [rsi]
    add ecx, r8d
    mov r9d, ecx
    cmp ecx, 32767
    jle .Ltr_add_tail_low
    mov ecx, 32767
    jmp .Ltr_add_tail_store
.Ltr_add_tail_low:
    cmp ecx, -32768
    jge .Ltr_add_tail_store
    mov ecx, -32768
.Ltr_add_tail_store:
    cmp ecx, r9d
    setne r10b
    movzx r10d, r10b
    add eax, r10d
    mov [rdi], cx
    add rdi, 2
    add rsi, 2
    dec edx
    jnz .Ltr_add_tail
.Ltr_add_done:
    ret
    .size zzs_mllee_train_add_grad_q15, .-zzs_mllee_train_add_grad_q15

/*
 * void zzs_mllee_train_axpy_q15(int16_t *destination, const int16_t *source,
 *                                int32_t scalar_q15, uint32_t count)
 *
 * term = a*m + round(a*v / 2^15) with the scalar split into the two lane-safe
 * halves m and v; packssdw is the saturating add's clamp, which is why this
 * kernel needs no rail vector.
 */
.globl zzs_mllee_train_axpy_q15
.type zzs_mllee_train_axpy_q15, @function
zzs_mllee_train_axpy_q15:
    movsxd r10, edx                 /* the scalar, widened for the tail       */
    mov r11d, edx
    sar r11d, 16
    add r11d, r11d
    mov r9d, edx
    shr r9d, 15
    and r9d, 1
    add r11d, r9d                   /* m = 2 * high + bit 15 of low           */
    mov r8d, edx
    and r8d, 0x7FFF                 /* v = low & 0x7FFF, never a sign lane    */
    movd xmm14, r11d
    punpcklwd xmm14, xmm14
    pshufd xmm14, xmm14, 0
    movd xmm15, r8d
    punpcklwd xmm15, xmm15
    pshufd xmm15, xmm15, 0
    movdqa xmm13, XMMWORD PTR [rip + .Ltr_c16384_32]
    cmp ecx, 8
    jb .Ltr_axpy_tail
    cmp r10, -1073741824
    jl .Ltr_axpy_tail
    cmp r10, 1073741823
    jg .Ltr_axpy_tail
.Ltr_axpy_loop:
    movdqu xmm0, XMMWORD PTR [rsi]
    movdqu xmm7, XMMWORD PTR [rdi]
    movdqa xmm1, xmm0
    movdqa xmm2, xmm0
    movdqa xmm3, xmm0
    pmullw xmm0, xmm14
    pmulhw xmm1, xmm14
    pmullw xmm2, xmm15
    pmulhw xmm3, xmm15
    movdqa xmm4, xmm0
    punpcklwd xmm0, xmm1            /* a * m, lanes 0..3                      */
    punpckhwd xmm4, xmm1            /* a * m, lanes 4..7                      */
    movdqa xmm1, xmm2
    punpcklwd xmm2, xmm3            /* a * v, lanes 0..3                      */
    punpckhwd xmm1, xmm3            /* a * v, lanes 4..7                      */
    paddd xmm2, xmm13
    paddd xmm1, xmm13
    psrad xmm2, 15
    psrad xmm1, 15
    paddd xmm0, xmm2
    paddd xmm4, xmm1
    movdqa xmm8, xmm7
    psraw xmm8, 15
    movdqa xmm9, xmm7
    punpcklwd xmm7, xmm8
    punpckhwd xmm9, xmm8
    paddd xmm7, xmm0
    paddd xmm9, xmm4
    packssdw xmm7, xmm9
    movdqu XMMWORD PTR [rdi], xmm7
    add rdi, 16
    add rsi, 16
    sub ecx, 8
    cmp ecx, 8
    jae .Ltr_axpy_loop
.Ltr_axpy_tail:
    test ecx, ecx
    jz .Ltr_axpy_done
.Ltr_axpy_tail_loop:
    movsx r8, WORD PTR [rsi]
    imul r8, r10
    add r8, 16384
    sar r8, 15
    movsx r9, WORD PTR [rdi]
    add r9, r8
    cmp r9, 32767
    jle .Ltr_axpy_tail_low
    mov r9d, 32767
    jmp .Ltr_axpy_tail_store
.Ltr_axpy_tail_low:
    cmp r9, -32768
    jge .Ltr_axpy_tail_store
    mov r9d, -32768
.Ltr_axpy_tail_store:
    mov [rdi], r9w
    add rdi, 2
    add rsi, 2
    dec ecx
    jnz .Ltr_axpy_tail_loop
.Ltr_axpy_done:
    ret
    .size zzs_mllee_train_axpy_q15, .-zzs_mllee_train_axpy_q15

/*
 * void zzs_mllee_train_scale_q15(int16_t *values, int32_t scalar_q15,
 *                                uint32_t count)
 *
 * The same half-split product as axpy; packssdw now clamps the value itself.
 */
.globl zzs_mllee_train_scale_q15
.type zzs_mllee_train_scale_q15, @function
zzs_mllee_train_scale_q15:
    movsxd r10, esi
    mov r11d, esi
    sar r11d, 16
    add r11d, r11d
    mov r9d, esi
    shr r9d, 15
    and r9d, 1
    add r11d, r9d
    mov r8d, esi
    and r8d, 0x7FFF
    movd xmm14, r11d
    punpcklwd xmm14, xmm14
    pshufd xmm14, xmm14, 0
    movd xmm15, r8d
    punpcklwd xmm15, xmm15
    pshufd xmm15, xmm15, 0
    movdqa xmm13, XMMWORD PTR [rip + .Ltr_c16384_32]
    cmp edx, 8
    jb .Ltr_scale_tail
    cmp r10, -1073741824
    jl .Ltr_scale_tail
    cmp r10, 1073741823
    jg .Ltr_scale_tail
.Ltr_scale_loop:
    movdqu xmm0, XMMWORD PTR [rdi]
    movdqa xmm1, xmm0
    movdqa xmm2, xmm0
    movdqa xmm3, xmm0
    pmullw xmm0, xmm14
    pmulhw xmm1, xmm14
    pmullw xmm2, xmm15
    pmulhw xmm3, xmm15
    movdqa xmm4, xmm0
    punpcklwd xmm0, xmm1
    punpckhwd xmm4, xmm1
    movdqa xmm1, xmm2
    punpcklwd xmm2, xmm3
    punpckhwd xmm1, xmm3
    paddd xmm2, xmm13
    paddd xmm1, xmm13
    psrad xmm2, 15
    psrad xmm1, 15
    paddd xmm0, xmm2
    paddd xmm4, xmm1
    packssdw xmm0, xmm4
    movdqu XMMWORD PTR [rdi], xmm0
    add rdi, 16
    sub edx, 8
    cmp edx, 8
    jae .Ltr_scale_loop
.Ltr_scale_tail:
    test edx, edx
    jz .Ltr_scale_done
.Ltr_scale_tail_loop:
    movsx r8, WORD PTR [rdi]
    imul r8, r10
    add r8, 16384
    sar r8, 15
    cmp r8, 32767
    jle .Ltr_scale_tail_low
    mov r8d, 32767
    jmp .Ltr_scale_tail_store
.Ltr_scale_tail_low:
    cmp r8, -32768
    jge .Ltr_scale_tail_store
    mov r8d, -32768
.Ltr_scale_tail_store:
    mov [rdi], r8w
    add rdi, 2
    dec edx
    jnz .Ltr_scale_tail_loop
.Ltr_scale_done:
    ret
    .size zzs_mllee_train_scale_q15, .-zzs_mllee_train_scale_q15

/*
 * void zzs_mllee_train_sgd_step_i8(int8_t *parameters, const int16_t *grad,
 *                                 int32_t lr_q15, uint32_t count)
 *
 * The update is the truncated division of the code step by 258, so it cannot
 * reuse a round-to-nearest multiply: the fallback adds +129/-129 before
 * truncating, which is round-half-away-from-zero.  Both that offset and the
 * reciprocal live in the int64 lanes, since a code step may be as wide as
 * 2^31 - 1 and the int32 lane would wrap on the way in.  packssdw's own
 * saturation plus the +-254 rails are exact here: a code step past 254 pushes
 * any parameter onto the [-127, 127] rail whichever rail value is used.
 */
.globl zzs_mllee_train_sgd_step_i8
.type zzs_mllee_train_sgd_step_i8, @function
zzs_mllee_train_sgd_step_i8:
    movsxd r10, edx
    mov r11d, edx
    sar r11d, 16
    add r11d, r11d
    mov r9d, edx
    shr r9d, 15
    and r9d, 1
    add r11d, r9d
    mov r8d, edx
    and r8d, 0x7FFF
    movd xmm14, r11d
    punpcklwd xmm14, xmm14
    pshufd xmm14, xmm14, 0
    movd xmm15, r8d
    punpcklwd xmm15, xmm15
    pshufd xmm15, xmm15, 0
    pxor xmm5, xmm5                 /* zero, the code sign compare            */
    movdqa xmm13, XMMWORD PTR [rip + .Ltr_c16384_32]
    movdqa xmm12, XMMWORD PTR [rip + .Ltr_magic40]
    movdqa xmm11, XMMWORD PTR [rip + .Ltr_129magic40]
    movdqa xmm10, XMMWORD PTR [rip + .Ltr_c254]
    movdqa xmm9, XMMWORD PTR [rip + .Ltr_cm254]
    movdqa xmm8, XMMWORD PTR [rip + .Ltr_c127]
    movdqa xmm7, XMMWORD PTR [rip + .Ltr_cm127]
    cmp ecx, 8
    jb .Ltr_sgd_tail
    cmp r10, -1073741824
    jl .Ltr_sgd_tail
    cmp r10, 1073741823
    jg .Ltr_sgd_tail
.Ltr_sgd_loop:
    movdqu xmm0, XMMWORD PTR [rsi]
    movdqa xmm1, xmm0
    movdqa xmm2, xmm0
    movdqa xmm3, xmm0
    pmullw xmm0, xmm14
    pmulhw xmm1, xmm14
    pmullw xmm2, xmm15
    pmulhw xmm3, xmm15
    movdqa xmm4, xmm0
    punpcklwd xmm0, xmm1
    punpckhwd xmm4, xmm1
    movdqa xmm1, xmm2
    punpcklwd xmm2, xmm3
    punpckhwd xmm1, xmm3
    paddd xmm2, xmm13
    paddd xmm1, xmm13
    psrad xmm2, 15
    psrad xmm1, 15
    paddd xmm0, xmm2                /* code step, lanes 0..3                  */
    paddd xmm4, xmm1                /* code step, lanes 4..7                  */
    /* magnitude, reciprocal of 258 with the +129 folded in, sign restored */
    movdqa xmm1, xmm0
    psrad xmm1, 31
    movdqa xmm2, xmm0
    pxor xmm2, xmm1
    psubd xmm2, xmm1
    movdqa xmm6, xmm2
    psrldq xmm6, 4
    pmuludq xmm2, xmm12
    pmuludq xmm6, xmm12
    paddq xmm2, xmm11
    paddq xmm6, xmm11
    psrlq xmm2, 40
    psrlq xmm6, 40
    pslldq xmm6, 4                  /* the quotients sit in each pair's low   */
    por xmm2, xmm6                  /* dword, so one shift merges the two     */
    paddd xmm2, xmm1
    pxor xmm2, xmm1
    movdqa xmm6, xmm2
    movdqa xmm1, xmm4
    psrad xmm1, 31
    movdqa xmm2, xmm4
    pxor xmm2, xmm1
    psubd xmm2, xmm1
    movdqa xmm3, xmm2
    psrldq xmm3, 4
    pmuludq xmm2, xmm12
    pmuludq xmm3, xmm12
    paddq xmm2, xmm11
    paddq xmm3, xmm11
    psrlq xmm2, 40
    psrlq xmm3, 40
    pslldq xmm3, 4
    por xmm2, xmm3
    paddd xmm2, xmm1
    pxor xmm2, xmm1
    packssdw xmm6, xmm2             /* code steps as int16 lanes              */
    pminsw xmm6, xmm10
    pmaxsw xmm6, xmm9
    movq xmm2, QWORD PTR [rdi]
    movdqa xmm3, xmm5
    pcmpgtb xmm3, xmm2              /* 0xFF where the code is negative        */
    punpcklbw xmm2, xmm3            /* parameters widened to int16 lanes      */
    paddw xmm6, xmm2
    pminsw xmm6, xmm8
    pmaxsw xmm6, xmm7
    packsswb xmm6, xmm6
    movq QWORD PTR [rdi], xmm6
    add rdi, 8
    add rsi, 16
    sub ecx, 8
    cmp ecx, 8
    jae .Ltr_sgd_loop
.Ltr_sgd_tail:
    test ecx, ecx
    jz .Ltr_sgd_done
.Ltr_sgd_tail_loop:
    movsx r8, WORD PTR [rsi]
    imul r8, r10
    add r8, 16384
    sar r8, 15                      /* the code step, exactly as the fallback */
    mov rax, r8
    xor r11d, r11d
    test rax, rax
    jns .Ltr_sgd_tail_div
    neg rax
    mov r11d, 1
.Ltr_sgd_tail_div:
    add rax, 129
    xor edx, edx
    mov r9, 258
    div r9
    test r11d, r11d
    jz .Ltr_sgd_tail_add
    neg rax
.Ltr_sgd_tail_add:
    movsx r11, BYTE PTR [rdi]
    add eax, r11d
    cmp eax, 127
    jle .Ltr_sgd_tail_low
    mov eax, 127
.Ltr_sgd_tail_low:
    cmp eax, -127
    jge .Ltr_sgd_tail_store
    mov eax, -127
.Ltr_sgd_tail_store:
    mov BYTE PTR [rdi], al
    inc rdi
    add rsi, 2
    dec ecx
    jnz .Ltr_sgd_tail_loop
.Ltr_sgd_done:
    ret
    .size zzs_mllee_train_sgd_step_i8, .-zzs_mllee_train_sgd_step_i8

/*
 * void zzs_mllee_train_relu_bwd(const int16_t *grad_out, const int8_t *input,
 *                               int16_t *grad_in, uint32_t count)
 *
 * Duplicating a code byte into both halves of a lane puts the forward sign in
 * bit 15, so one psraw is the whole predicate and pandn routes the gradient.
 * Only the low eight bytes of the code stream are ever touched: a group is
 * eight codes, not sixteen.
 */
.globl zzs_mllee_train_relu_bwd
.type zzs_mllee_train_relu_bwd, @function
zzs_mllee_train_relu_bwd:
    pxor xmm5, xmm5
    cmp ecx, 8
    jb .Ltr_relu_tail
.Ltr_relu_loop:
    movdqu xmm0, XMMWORD PTR [rdi]
    movq xmm2, QWORD PTR [rsi]
    punpcklbw xmm2, xmm2
    psraw xmm2, 15                  /* 0xFFFF where the code is negative      */
    pandn xmm2, xmm0
    movdqu XMMWORD PTR [rdx], xmm2
    add rdi, 16
    add rsi, 8
    add rdx, 16
    sub ecx, 8
    cmp ecx, 8
    jae .Ltr_relu_loop
.Ltr_relu_tail:
    test ecx, ecx
    jz .Ltr_relu_done
    xor r9d, r9d
.Ltr_relu_tail_loop:
    movsx r8, BYTE PTR [rsi]
    movzx eax, WORD PTR [rdi]
    test r8d, r8d
    cmovs eax, r9d
    mov WORD PTR [rdx], ax
    add rdi, 2
    add rsi, 1
    add rdx, 2
    dec ecx
    jnz .Ltr_relu_tail_loop
.Ltr_relu_done:
    ret
    .size zzs_mllee_train_relu_bwd, .-zzs_mllee_train_relu_bwd

/*
 * int32_t zzs_mllee_train_sum_abs_q15(const int16_t *values, uint32_t count)
 *
 * |v| = max(v,0) - min(v,0) needs 17 bits for the single -32768 lane, which is
 * why the magnitudes widen as unsigned (punpck?wd against zero) before the
 * int64 lanes take them; a sign dword would turn that lane back into -32768.
 */
.globl zzs_mllee_train_sum_abs_q15
.type zzs_mllee_train_sum_abs_q15, @function
zzs_mllee_train_sum_abs_q15:
    pxor xmm4, xmm4                 /* zero, the rail and the widening operand */
    pxor xmm5, xmm5                 /* int64 lane pair accumulator             */
    cmp esi, 8
    jb .Ltr_abs_collapse
.Ltr_abs_loop:
    movdqu xmm0, XMMWORD PTR [rdi]
    movdqa xmm1, xmm0
    pmaxsw xmm1, xmm4
    pminsw xmm0, xmm4
    psubw xmm1, xmm0
    movdqa xmm2, xmm1
    punpcklwd xmm1, xmm4
    punpckhwd xmm2, xmm4
    paddd xmm1, xmm2                /* two magnitudes per int32 lane          */
    movdqa xmm3, xmm1
    punpckldq xmm1, xmm4
    punpckhdq xmm3, xmm4
    paddq xmm5, xmm1
    paddq xmm5, xmm3
    add rdi, 16
    sub esi, 8
    cmp esi, 8
    jae .Ltr_abs_loop
.Ltr_abs_collapse:
    movq rax, xmm5
    psrldq xmm5, 8
    movq rcx, xmm5
    add rax, rcx
    test esi, esi
    jz .Ltr_abs_sat
.Ltr_abs_tail:
    movsx ecx, WORD PTR [rdi]
    test ecx, ecx
    jns .Ltr_abs_tail_add
    neg ecx
.Ltr_abs_tail_add:
    add rax, rcx
    add rdi, 2
    dec esi
    jnz .Ltr_abs_tail
.Ltr_abs_sat:
    mov rcx, 2147483647
    cmp rax, rcx
    cmovg rax, rcx
    ret
    .size zzs_mllee_train_sum_abs_q15, .-zzs_mllee_train_sum_abs_q15

/*
 * int32_t zzs_mllee_train_sum_sq_q15(const int16_t *values, uint32_t count)
 *
 * pmullw/pmulhw of a lane against itself is the exact square -- never pmaddwd,
 * whose pair sum of two 2^30 squares wraps -- and the int64 lanes then hold the
 * whole total, so the fallback's saturate-and-return reads as one clamp.
 */
.globl zzs_mllee_train_sum_sq_q15
.type zzs_mllee_train_sum_sq_q15, @function
zzs_mllee_train_sum_sq_q15:
    pxor xmm4, xmm4                 /* zero, the widening operand             */
    pxor xmm5, xmm5                 /* int64 lane pair accumulator            */
    cmp esi, 8
    jb .Ltr_sq_collapse
.Ltr_sq_loop:
    movdqu xmm0, XMMWORD PTR [rdi]
    movdqa xmm1, xmm0
    pmullw xmm0, xmm0
    pmulhw xmm1, xmm1               /* a square is non-negative, so the high  */
    movdqa xmm2, xmm0               /* half never needs an unsigned read      */
    punpcklwd xmm0, xmm1
    punpckhwd xmm2, xmm1
    movdqa xmm3, xmm0
    punpckldq xmm0, xmm4
    punpckhdq xmm3, xmm4
    paddq xmm5, xmm0
    paddq xmm5, xmm3
    movdqa xmm3, xmm2
    punpckldq xmm2, xmm4
    punpckhdq xmm3, xmm4
    paddq xmm5, xmm2
    paddq xmm5, xmm3
    add rdi, 16
    sub esi, 8
    cmp esi, 8
    jae .Ltr_sq_loop
.Ltr_sq_collapse:
    movq rax, xmm5
    psrldq xmm5, 8
    movq rcx, xmm5
    add rax, rcx
    mov ecx, 2147483647
    cmp rax, rcx
    cmovg rax, rcx
    test esi, esi
    jz .Ltr_sq_done
.Ltr_sq_tail:
    movsx rcx, WORD PTR [rdi]
    imul rcx, rcx
    add rax, rcx
    mov ecx, 2147483647
    cmp rax, rcx
    cmovg rax, rcx
    add rdi, 2
    dec esi
    jnz .Ltr_sq_tail
.Ltr_sq_done:
    ret
    .size zzs_mllee_train_sum_sq_q15, .-zzs_mllee_train_sum_sq_q15

/*
 * void zzs_mllee_train_codes_to_q15(const int8_t *codes, int16_t *values,
 *                                   uint32_t count)
 *
 * The product 127 * 258 = 32766 is the widest a code can reach, so pmullw
 * already holds it and no high half has to be recovered.
 */
.globl zzs_mllee_train_codes_to_q15
.type zzs_mllee_train_codes_to_q15, @function
zzs_mllee_train_codes_to_q15:
    movdqa xmm6, XMMWORD PTR [rip + .Ltr_c258]
    pxor xmm5, xmm5
    cmp edx, 8
    jb .Ltr_widen_tail
.Ltr_widen_loop:
    movq xmm0, QWORD PTR [rdi]
    movdqa xmm1, xmm5
    pcmpgtb xmm1, xmm0
    punpcklbw xmm0, xmm1            /* eight codes widened to eight lanes     */
    pmullw xmm0, xmm6
    movdqu XMMWORD PTR [rsi], xmm0
    add rdi, 8
    add rsi, 16
    sub edx, 8
    cmp edx, 8
    jae .Ltr_widen_loop
.Ltr_widen_tail:
    test edx, edx
    jz .Ltr_widen_done
.Ltr_widen_tail_loop:
    movsx ecx, BYTE PTR [rdi]
    imul ecx, ecx, 258
    mov WORD PTR [rsi], cx
    inc rdi
    add rsi, 2
    dec edx
    jnz .Ltr_widen_tail_loop
.Ltr_widen_done:
    ret
    .size zzs_mllee_train_codes_to_q15, .-zzs_mllee_train_codes_to_q15

/*
 * void zzs_mllee_train_q15_to_codes(const int16_t *values, int8_t *codes,
 *                                   uint32_t count)
 *
 * (|v| * 127 + 16384) >> 15 with every term doubled is
 * (|v| * 254 + 32768) >> 16: pmulhuw gives that quotient's floor and bit 15 of
 * pmullw the one rounding step it needs, so the whole conversion stays in
 * 16-bit lanes and its result never exceeds 127.
 */
.globl zzs_mllee_train_q15_to_codes
.type zzs_mllee_train_q15_to_codes, @function
zzs_mllee_train_q15_to_codes:
    movdqa xmm7, XMMWORD PTR [rip + .Ltr_c254]
    cmp edx, 8
    jb .Ltr_narrow_tail
.Ltr_narrow_loop:
    movdqu xmm0, XMMWORD PTR [rdi]
    movdqa xmm1, xmm0
    psraw xmm1, 15
    movdqa xmm2, xmm0
    pxor xmm2, xmm1
    psubw xmm2, xmm1                /* magnitude bits; 0x8000 is the |rail|   */
    movdqa xmm3, xmm2
    pmulhuw xmm3, xmm7              /* unsigned high half: 32768 is not -32768 */
    pmullw xmm2, xmm7
    psrlw xmm2, 15                  /* the rounding step folded into the term  */
    paddw xmm2, xmm3
    pxor xmm2, xmm1
    psubw xmm2, xmm1
    packsswb xmm2, xmm2
    movq QWORD PTR [rsi], xmm2
    add rdi, 16
    add rsi, 8
    sub edx, 8
    cmp edx, 8
    jae .Ltr_narrow_loop
.Ltr_narrow_tail:
    test edx, edx
    jz .Ltr_narrow_done
.Ltr_narrow_tail_loop:
    movsx ecx, WORD PTR [rdi]
    test ecx, ecx
    js .Ltr_narrow_tail_neg
    imul ecx, ecx, 127
    add ecx, 16384
    sar ecx, 15
    jmp .Ltr_narrow_tail_store
.Ltr_narrow_tail_neg:
    neg ecx
    imul ecx, ecx, 127
    add ecx, 16384
    sar ecx, 15
    neg ecx
.Ltr_narrow_tail_store:
    mov BYTE PTR [rsi], cl
    add rdi, 2
    inc rsi
    dec edx
    jnz .Ltr_narrow_tail_loop
.Ltr_narrow_done:
    ret
    .size zzs_mllee_train_q15_to_codes, .-zzs_mllee_train_q15_to_codes

.section .note.GNU-stack,"",@progbits
