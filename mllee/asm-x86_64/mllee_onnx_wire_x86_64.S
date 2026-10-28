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
 * mllee ONNX wire primitives (x86-64).
 *
 * Four leaf routines used by the streaming loader:
 *   zzs_mllee_onnx_varint32 / varint64   protobuf varint decode
 *   zzs_mllee_onnx_hash32                FNV-1a over a caller-owned string
 *   zzs_mllee_onnx_copy                  non-overlapping byte copy
 *
 * No stack frame, no callee-saved register, no writable static.  The varint
 * decoders return the number of bytes consumed, or zero when the range ends
 * early, the value does not fit, or a reserved high bit is set in the final
 * byte.  That is the same contract as the portable fallback in onnx_asm.hpp.
 */

.intel_syntax noprefix

.text

/*
 * uint32_t zzs_mllee_onnx_varint32(const uint8_t *src, uint32_t available,
 *                                  uint32_t *out)
 */
.globl zzs_mllee_onnx_varint32
.type zzs_mllee_onnx_varint32, @function
zzs_mllee_onnx_varint32:
    xor eax, eax                    /* accumulated value          */
    xor ecx, ecx                    /* shift                      */
    xor r8d, r8d                    /* index                      */
    test esi, esi
    jz .Lvarint32_fail
    cmp esi, 5
    jbe .Lvarint32_loop
    mov esi, 5
.Lvarint32_loop:
    movzx r9d, BYTE PTR [rdi + r8]
    cmp r8d, 4
    jne .Lvarint32_no_fifth
    test r9b, 0xF0
    jnz .Lvarint32_fail
.Lvarint32_no_fifth:
    mov r10d, r9d
    and r10d, 0x7F
    shl r10d, cl
    or eax, r10d
    test r9b, 0x80
    jz .Lvarint32_done
    add ecx, 7
    inc r8d
    cmp r8d, esi
    jb .Lvarint32_loop
.Lvarint32_fail:
    xor eax, eax
    ret
.Lvarint32_done:
    mov DWORD PTR [rdx], eax
    lea eax, [r8 + 1]
    ret
.size zzs_mllee_onnx_varint32, .-zzs_mllee_onnx_varint32

/*
 * uint32_t zzs_mllee_onnx_varint64(const uint8_t *src, uint32_t available,
 *                                  uint64_t *out)
 */
.globl zzs_mllee_onnx_varint64
.type zzs_mllee_onnx_varint64, @function
zzs_mllee_onnx_varint64:
    xor eax, eax                    /* accumulated value          */
    xor ecx, ecx                    /* shift                      */
    xor r8d, r8d                    /* index                      */
    test esi, esi
    jz .Lvarint64_fail
    cmp esi, 10
    jbe .Lvarint64_loop
    mov esi, 10
.Lvarint64_loop:
    movzx r9d, BYTE PTR [rdi + r8]
    cmp r8d, 9
    jne .Lvarint64_no_tenth
    test r9b, 0xFE
    jnz .Lvarint64_fail
.Lvarint64_no_tenth:
    mov r10, r9
    and r10, 0x7F
    shl r10, cl
    or rax, r10
    test r9b, 0x80
    jz .Lvarint64_done
    add ecx, 7
    inc r8d
    cmp r8d, esi
    jb .Lvarint64_loop
.Lvarint64_fail:
    xor eax, eax
    ret
.Lvarint64_done:
    mov QWORD PTR [rdx], rax
    lea eax, [r8 + 1]
    ret
.size zzs_mllee_onnx_varint64, .-zzs_mllee_onnx_varint64

/*
 * uint32_t zzs_mllee_onnx_hash32(const uint8_t *src, uint32_t length)
 *
 * FNV-1a: hash = (hash ^ byte) * 16777619.  The multiply is a real dependency
 * chain, so the loop is kept branch-tight instead of being unrolled.
 */
.globl zzs_mllee_onnx_hash32
.type zzs_mllee_onnx_hash32, @function
zzs_mllee_onnx_hash32:
    mov eax, 2166136261
    test esi, esi
    jz .Lhash32_done
    xor ecx, ecx
.Lhash32_loop:
    movzx edx, BYTE PTR [rdi + rcx]
    xor eax, edx
    imul eax, eax, 16777619
    inc ecx
    cmp ecx, esi
    jb .Lhash32_loop
.Lhash32_done:
    ret
.size zzs_mllee_onnx_hash32, .-zzs_mllee_onnx_hash32

/*
 * void zzs_mllee_onnx_copy(uint8_t *dst, const uint8_t *src, uint32_t length)
 *
 * Short spans (headers, names, one tensor element) use a byte loop with no
 * setup cost.  Longer spans use rep movsb, which modern x86 cores run at or
 * near the memory bandwidth even for the 16-64 byte range.  The direction flag
 * is clear on entry as required by the ABI, so no cld is needed.
 */
.globl zzs_mllee_onnx_copy
.type zzs_mllee_onnx_copy, @function
zzs_mllee_onnx_copy:
    test edx, edx
    jz .Lcopy_done
    cmp edx, 64
    jae .Lcopy_rep
.Lcopy_bytes:
    mov al, BYTE PTR [rsi]
    mov BYTE PTR [rdi], al
    inc rsi
    inc rdi
    dec edx
    jnz .Lcopy_bytes
    ret
.Lcopy_rep:
    mov ecx, edx
    rep movsb
.Lcopy_done:
    ret
.size zzs_mllee_onnx_copy, .-zzs_mllee_onnx_copy

.section .note.GNU-stack,"",@progbits
