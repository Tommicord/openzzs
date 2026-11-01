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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

# WiFi bus assembly primitives for x86_64.
# Selected at build time when OPENZZS_DEVICE_DRIVER_WIFI_BUS_ASM is ON and the
# target is x86_64.  Falls back to wifi_bus_asm_cpp.cpp otherwise.
#
# These routines toggle GPIO bits and chip-select lines via MMIO.  They are
# timing primitives only; real hardware should use a proper timer.

.intel_syntax noprefix
.text
.globl zzs_wifi_bus_gpio_toggle_delay
.globl zzs_wifi_bus_spi_cs_set

# void zzs_wifi_bus_gpio_toggle_delay(volatile uint32_t *base, uint32_t bit,
#                                     uint32_t count) noexcept
zzs_wifi_bus_gpio_toggle_delay:
        test    rdi, rdi                  # null base?
        jz      .Lgpio_ret
        cmp     esi, 32                   # bit < 32?
        jae     .Lgpio_ret
        mov     ecx, edx                  # count
        mov     eax, 1
        mov     cl, sil                   # bit count into CL
        shl     eax, cl                   # mask = 1 << bit
.Lgpio_loop:
        mov     r8d, DWORD PTR [rdi]      # odr = *base
        xor     r8d, eax                  # toggle bit
        mov     DWORD PTR [rdi], r8d      # *base = odr ^ mask
        loop    .Lgpio_loop
.Lgpio_ret:
        ret

# void zzs_wifi_bus_spi_cs_set(volatile uint32_t *base, uint32_t bit,
#                              int asserted) noexcept
zzs_wifi_bus_spi_cs_set:
        test    rdi, rdi                  # null base?
        jz      .Lcs_ret
        cmp     esi, 32                   # bit < 32?
        jae     .Lcs_ret
        mov     eax, 1
        mov     cl, sil                   # mask = 1 << bit
        shl     eax, cl
        mov     r8d, DWORD PTR [rdi]      # odr = *base
        test    edx, edx                  # asserted?
        jnz     .Lcs_assert
        or      r8d, eax                  # deassert: odr | mask
        jmp     .Lcs_store
.Lcs_assert:
        not     eax                       # ~mask
        and     r8d, eax                  # assert: odr & ~mask
.Lcs_store:
        mov     DWORD PTR [rdi], r8d      # *base = result
.Lcs_ret:
        ret
