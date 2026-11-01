/* Copyright(c) 2026 Tommicord
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. */

/*
 * std/secure_channel_aes_gcm_x86_64.S
 *
 * x86-64 AES-NI hot-path primitives for the secure channel AES-256-GCM.
 *
 * Selected at build time (never run-time CPUID).  Provides:
 *   zzs_scc_aes_keyexp256  - AES-256 key schedule (round keys into caller buffer)
 *   zzs_scc_aes_enc_block  - single-block AES encryption (16 bytes)
 *   zzs_scc_ghash_clmul    - GHASH over 16-byte blocks via PCLMULQDQ
 *
 * Every routine is a leaf (no stack frame beyond the red zone, no
 * callee-saved registers, no writable static).  Fixed instruction sequence,
 * no data-dependent branches on secret data.
 */

.intel_syntax noprefix

.section .rodata
.align 16
.Lscc_rcon:
    .byte 0x01, 0x00, 0x00, 0x00
    .byte 0x02, 0x00, 0x00, 0x00
    .byte 0x04, 0x00, 0x00, 0x00
    .byte 0x08, 0x00, 0x00, 0x00
    .byte 0x10, 0x00, 0x00, 0x00
    .byte 0x20, 0x00, 0x00, 0x00
    .byte 0x40, 0x00, 0x00, 0x00
    .byte 0x80, 0x00, 0x00, 0x00
    .byte 0x1b, 0x00, 0x00, 0x00
    .byte 0x36, 0x00, 0x00, 0x00
    .byte 0x6c, 0x00, 0x00, 0x00
    .byte 0xd8, 0x00, 0x00, 0x00
    .byte 0xab, 0x00, 0x00, 0x00
    .byte 0x4d, 0x00, 0x00, 0x00
    .byte 0x9a, 0x00, 0x00, 0x00

.section .text
.align 16

/* void zzs_scc_aes_keyexp256(const uint8_t *key32, uint8_t *rk240)
 * Produces 15 round keys x 16 bytes = 240 bytes using AES-NI
 * (aeskeygenassist + pshufd + aesimc for inverse-round conversion). */
.globl zzs_scc_aes_keyexp256
.type zzs_scc_aes_keyexp256, @function
zzs_scc_aes_keyexp256:
    /* rdi = key32, rsi = rk240 */
    movdqu xmm0, XMMWORD PTR [rdi + 0]   /* rk[0..3]  */
    movdqu xmm1, XMMWORD PTR [rdi + 16]  /* rk[4..7]  */
    movdqu XMMWORD PTR [rsi + 0], xmm0
    movdqu XMMWORD PTR [rsi + 16], xmm1
    xor ecx, ecx                         /* rcon index */
.Lscc_ke_loop:
    cmp ecx, 7
    jge .Lscc_ke_done
    /* t = aeskeygenassist(xmm1, 0) -> [w3 w2 w1 w0] shuffled */
    aeskeygenassist xmm2, xmm1, 0
    /* rcon = .Lscc_rcon[ecx*4] */
    movdqu xmm3, XMMWORD PTR [.Lscc_rcon + rcx * 4]
    pshufd xmm2, xmm2, 0xff              /* broadcast w3 */
    pxor xmm2, xmm3
    pxor xmm0, xmm2                      /* new w0 = w0 ^ t ^ rcon */
    /* rotate t */
    aeskeygenassist xmm2, xmm1, 0
    pshufd xmm2, xmm2, 0xaa              /* broadcast w2 */
    pxor xmm0, xmm2                      /* new w1 */
    aeskeygenassist xmm2, xmm1, 0
    pshufd xmm2, xmm2, 0x55              /* broadcast w1 */
    pxor xmm0, xmm2                      /* new w2 */
    aeskeygenassist xmm2, xmm1, 0
    pshufd xmm2, xmm2, 0x00              /* broadcast w0 */
    pxor xmm0, xmm2                      /* new w3 */
    mov rax, rcx
    shl rax, 5                           /* rax = rcx * 32 */
    movdqu XMMWORD PTR [rsi + rax + 32], xmm0   /* store rk[(cx+1)*4 .. +3] */
    /* second half: key2 derived from key1 */
    aesimc xmm4, xmm0                    /* inverse mix columns for key2 */
    aeskeygenassist xmm2, xmm4, 0
    pshufd xmm2, xmm2, 0xff
    pxor xmm2, xmm3
    pxor xmm1, xmm2
    aeskeygenassist xmm2, xmm4, 0
    pshufd xmm2, xmm2, 0xaa
    pxor xmm1, xmm2
    aeskeygenassist xmm2, xmm4, 0
    pshufd xmm2, xmm2, 0x55
    pxor xmm1, xmm2
    aeskeygenassist xmm2, xmm4, 0
    pshufd xmm2, xmm2, 0x00
    pxor xmm1, xmm2
    movdqu XMMWORD PTR [rsi + rax + 48], xmm1   /* store rk[(cx+1)*4+4 .. +7] */
    inc ecx
    jmp .Lscc_ke_loop
.Lscc_ke_done:
    ret
.size zzs_scc_aes_keyexp256, .-zzs_scc_aes_keyexp256

.align 16
/* void zzs_scc_aes_enc_block(const uint8_t *in16, const uint8_t *rk240, uint8_t *out16)
 * Single AES-256 encryption block (14 rounds) using AES-NI. */
.globl zzs_scc_aes_enc_block
.type zzs_scc_aes_enc_block, @function
zzs_scc_aes_enc_block:
    /* rdi = in, rsi = rk, rdx = out */
    movdqu xmm0, XMMWORD PTR [rdi]
    pxor xmm0, XMMWORD PTR [rsi + 0]     /* AddRoundKey rk0 */
    mov ecx, 1
.Lscc_enc_loop:
    mov rax, rcx
    shl rax, 4                           /* rax = rcx * 16 */
    aesenc xmm0, XMMWORD PTR [rsi + rax]
    inc ecx
    cmp ecx, 14
    jle .Lscc_enc_loop
    aesdeclast xmm0, XMMWORD PTR [rsi + 14 * 16]   /* final round */
    movdqu XMMWORD PTR [rdx], xmm0
    ret
.size zzs_scc_aes_enc_block, .-zzs_scc_aes_enc_block

.align 16
/* void zzs_scc_ghash_clmul(const uint8_t *h16, const uint8_t *data, size_t len,
 *                          uint8_t *y16)
 * GHASH over len bytes (multiple of 16) using PCLMULQDQ carry-less multiply
 * in GF(2^128) with the GCM reduction polynomial. */
.globl zzs_scc_ghash_clmul
.type zzs_scc_ghash_clmul, @function
zzs_scc_ghash_clmul:
    /* rdi = h, rsi = data, rdx = len, rcx = y (in/out) */
    movdqu xmm4, XMMWORD PTR [rdi]       /* H */
    /* reduction constant: x^128 + x^7 + x^2 + x + 1 = 0x011b -> low part */
    mov rax, 0xc200000000000000
    movq xmm5, rax
    pshufd xmm5, xmm5, 0                 /* broadcast R = 0xc20000... */
    xor eax, eax                         /* y = 0 initially (caller zeroes) */
    test rdx, rdx
    jz .Lscc_ghash_done
.Lscc_ghash_loop:
    movdqu xmm0, XMMWORD PTR [rsi]
    pxor xmm0, XMMWORD PTR [rcx]         /* y ^= block */
    /* xmm0 = y block, xmm4 = H -> clmul */
    movdqa xmm1, xmm0
    movdqa xmm2, xmm0
    pclmulqdq xmm1, xmm4, 0x00           /* low*low */
    pclmulqdq xmm2, xmm4, 0x11           /* high*low */
    pclmulqdq xmm0, xmm4, 0x10           /* low*high */
    pxor xmm1, xmm2                      /* cross */
    pxor xmm0, xmm1
    /* reduction: fold high 128 bits down */
    movdqa xmm1, xmm0
    pshufd xmm1, xmm1, 0xee              /* high part */
    pclmulqdq xmm1, xmm5, 0x00
    pslldq xmm1, 8
    pxor xmm0, xmm1
    movdqa xmm1, xmm0
    pshufd xmm1, xmm1, 0xee
    pclmulqdq xmm1, xmm5, 0x00
    pslldq xmm1, 4
    pxor xmm0, xmm1
    movdqu XMMWORD PTR [rcx], xmm0       /* store y */
    add rsi, 16
    sub rdx, 16
    jnz .Lscc_ghash_loop
.Lscc_ghash_done:
    ret
.size zzs_scc_ghash_clmul, .-zzs_scc_ghash_clmul

.section .note.GNU-stack,"",@progbits
