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

# WiFi bus assembly primitives for Cortex-M4 (Thumb-2).
# Selected at build time when OPENZZS_DEVICE_DRIVER_WIFI_BUS_ASM is ON and the
# target is cortex-m4.  Falls back to wifi_bus_asm_cpp.cpp otherwise.
#
# These routines toggle GPIO bits and chip-select lines via MMIO.  They are
# timing primitives only; real hardware should use a proper timer.

.syntax unified
.cpu cortex-m4
.thumb
.thumb_func
.global zzs_wifi_bus_gpio_toggle_delay
.thumb_func
.global zzs_wifi_bus_spi_cs_set

# void zzs_wifi_bus_gpio_toggle_delay(volatile uint32_t *base, uint32_t bit,
#                                     uint32_t count) noexcept
# r0 = base, r1 = bit, r2 = count
zzs_wifi_bus_gpio_toggle_delay:
        cmp     r0, #0
        it      eq
        beq     .Lgpio_ret
        cmp     r1, #32
        it      hs
        bhs     .Lgpio_ret
        lsl     r3, r1, #1                # mask = 1 << bit
.Lgpio_loop:
        cmp     r2, #0
        it      eq
        beq     .Lgpio_ret
        ldr     r4, [r0]                  # odr = *base
        eor     r4, r4, r3                # toggle bit
        str     r4, [r0]                  # *base = odr ^ mask
        subs    r2, r2, #1
        bne     .Lgpio_loop
.Lgpio_ret:
        bx      lr

# void zzs_wifi_bus_spi_cs_set(volatile uint32_t *base, uint32_t bit,
#                              int asserted) noexcept
# r0 = base, r1 = bit, r2 = asserted
zzs_wifi_bus_spi_cs_set:
        cmp     r0, #0
        it      eq
        beq     .Lcs_ret
        cmp     r1, #32
        it      hs
        bhs     .Lcs_ret
        lsl     r3, r1, #1                # mask = 1 << bit
        ldr     r4, [r0]                  # odr = *base
        cmp     r2, #0
        it      ne
        orrne   r4, r4, r3                # deassert: odr | mask
        beq     .Lcs_store
        mvn     r3, r3                    # ~mask
        and     r4, r4, r3                # assert: odr & ~mask
.Lcs_store:
        str     r4, [r0]                  # *base = result
.Lcs_ret:
        bx      lr
