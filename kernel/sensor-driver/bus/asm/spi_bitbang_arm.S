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

/*
 * ARM SPI bit-bang assembly routines (Cortex-M / Cortex-A).
 *
 * Direct hardware communication via memory-mapped GPIO registers.  GPIO
 * primitives (bus_gpio_set / bus_gpio_get) are defined in
 * i2c_bitbang_arm.S and referenced here as external symbols.
 *
 * Thumb-2 instruction set.  Fixed-sequence code with predictable instruction
 * count; no writable static, no runtime CPU probing.
 *
 * GPIO register model (per bus, base address in r0):
 *   +0x00 GPSET  write 1<<pin to set pin high   (write-only)
 *   +0x04 GPCLR  write 1<<pin to set pin low    (write-only)
 *   +0x08 GPLEV  read pin level                 (read-only)
 *
 * Calling convention: AAPCS.
 *   r0 = arg1, r1 = arg2, r2 = arg3, r3 = arg4, r4 = arg5
 *   Return in r0.
 */

.syntax unified
.thumb
.cpu cortex-m4
.fpu softvfp
.section .text.spi_bitbang_arm

/* External GPIO primitives defined in i2c_bitbang_arm.S */
.globl bus_gpio_set
.globl bus_gpio_get

/* ------------------------------------------------------------------ */
/* SPI delay                                                          */
/* ------------------------------------------------------------------ */

/*
 * void bus_spi_delay_us(uint32_t us)
 */
.globl bus_spi_delay_us
.type bus_spi_delay_us, %function
.thumb_func
bus_spi_delay_us:
    /* r0 = us */
    cmp r0, #0
    beq .Lspi_delay_done
    mov r1, r0                  /* r1 = remaining us */
.Lspi_delay_loop:
    mov r2, #200
.Lspi_delay_inner:
    subs r2, r2, #1
    bne .Lspi_delay_inner
    subs r1, r1, #1
    bne .Lspi_delay_loop
.Lspi_delay_done:
    bx lr
.size bus_spi_delay_us, .-bus_spi_delay_us

/* ------------------------------------------------------------------ */
/* SPI chip select                                                    */
/* ------------------------------------------------------------------ */

/*
 * void bus_spi_cs_assert(uint32_t gpio_base, uint32_t cs_pin)
 *
 * Assert chip select (drive CS low -- active low by convention).
 */
.globl bus_spi_cs_assert
.type bus_spi_cs_assert, %function
.thumb_func
bus_spi_cs_assert:
    /* r0 = gpio_base, r1 = cs_pin */
    add r0, r0, #4              /* GPCLR -> CS low (assert) */
    mov r2, #1
    lsl r2, r2, r1              /* r2 = 1 << cs_pin */
    str r2, [r0]
    bx lr
.size bus_spi_cs_assert, .-bus_spi_cs_assert

/*
 * void bus_spi_cs_deassert(uint32_t gpio_base, uint32_t cs_pin)
 *
 * Deassert chip select (drive CS high).
 */
.globl bus_spi_cs_deassert
.type bus_spi_cs_deassert, %function
.thumb_func
bus_spi_cs_deassert:
    /* r0 = gpio_base, r1 = cs_pin */
    add r0, r0, #0              /* GPSET -> CS high (deassert) */
    mov r2, #1
    lsl r2, r2, r1              /* r2 = 1 << cs_pin */
    str r2, [r0]
    bx lr
.size bus_spi_cs_deassert, .-bus_spi_cs_deassert

/* ------------------------------------------------------------------ */
/* SPI bit read/write                                                 */
/* ------------------------------------------------------------------ */

/*
 * void bus_spi_write_bit(uint32_t gpio_base, uint32_t mosi_pin,
 *                        uint32_t sclk_pin, uint8_t bit)
 *
 * Drive MOSI to `bit`, then toggle SCLK high then low.
 * Mode 0: data is valid before rising edge, sampled on rising edge.
 */
.globl bus_spi_write_bit
.type bus_spi_write_bit, %function
.thumb_func
bus_spi_write_bit:
    /* r0 = gpio_base, r1 = mosi_pin, r2 = sclk_pin, r3 = bit */
    push {r4, r5, r6, lr}
    mov r4, r0                  /* r4 = gpio_base */
    mov r5, r1                  /* r5 = mosi_pin */
    mov r6, r2                  /* r6 = sclk_pin */
    /* Drive MOSI according to bit */
    cmp r3, #0
    ite eq
    addeq r0, r4, #4            /* bit==0: GPCLR (MOSI low) */
    addne r0, r4, #0            /* bit!=0: GPSET (MOSI high) */
    mov r1, #1
    lsl r1, r1, r5              /* r1 = 1 << mosi_pin */
    str r1, [r0]
    /* tSU: setup time before clock edge */
    mov r0, #1
    bl bus_spi_delay_us
    /* Toggle SCLK high */
    add r0, r4, #0              /* GPSET */
    mov r1, #1
    lsl r1, r1, r6              /* r1 = 1 << sclk_pin */
    str r1, [r0]
    mov r0, #1
    bl bus_spi_delay_us
    /* Toggle SCLK low */
    add r0, r4, #4              /* GPCLR */
    mov r1, #1
    lsl r1, r1, r6
    str r1, [r0]
    mov r0, #1
    bl bus_spi_delay_us
    pop {r4, r5, r6, pc}
.size bus_spi_write_bit, .-bus_spi_write_bit

/*
 * uint8_t bus_spi_read_bit(uint32_t gpio_base, uint32_t miso_pin,
 *                          uint32_t sclk_pin)
 *
 * Toggle SCLK high, sample MISO while SCLK high, then SCLK low.
 * Returns the data bit in r0.
 */
.globl bus_spi_read_bit
.type bus_spi_read_bit, %function
.thumb_func
bus_spi_read_bit:
    /* r0 = gpio_base, r1 = miso_pin, r2 = sclk_pin */
    push {r4, r5, r6, lr}
    mov r4, r0                  /* r4 = gpio_base */
    mov r5, r1                  /* r5 = miso_pin */
    mov r6, r2                  /* r6 = sclk_pin */
    /* Toggle SCLK high */
    add r0, r4, #0              /* GPSET */
    mov r1, #1
    lsl r1, r1, r6              /* r1 = 1 << sclk_pin */
    str r1, [r0]
    mov r0, #1
    bl bus_spi_delay_us
    /* Sample MISO while SCLK high */
    add r0, r4, #8              /* GPLEV */
    ldr r1, [r0]
    lsr r1, r1, r5
    and r0, r1, #1              /* r0 = data bit */
    /* Toggle SCLK low */
    add r0, r4, #4              /* GPCLR */
    mov r1, #1
    lsl r1, r1, r6
    str r1, [r0]
    mov r0, #1
    bl bus_spi_delay_us
    pop {r4, r5, r6, pc}        /* r0 = data bit */
.size bus_spi_read_bit, .-bus_spi_read_bit

/* ------------------------------------------------------------------ */
/* SPI byte transfer                                                    */
/* ------------------------------------------------------------------ */

/*
 * uint8_t bus_spi_transfer_byte(uint32_t gpio_base, uint32_t mosi_pin,
 *                                uint32_t miso_pin, uint32_t sclk_pin,
 *                                uint8_t byte)
 *
 * Full-duplex byte transfer, MSB first, mode 0 (CPOL=0, CPHA=0).
 * Data changes on falling edge, sampled on rising edge.
 * Returns the received byte in r0.
 */
.globl bus_spi_transfer_byte
.type bus_spi_transfer_byte, %function
.thumb_func
bus_spi_transfer_byte:
    /* r0 = gpio_base, r1 = mosi_pin, r2 = miso_pin, r3 = sclk_pin, r4 = byte */
    push {r5, r6, r7, r8, lr}
    mov r5, r0                  /* r5 = gpio_base */
    mov r6, r1                  /* r6 = mosi_pin */
    mov r7, r2                  /* r7 = miso_pin */
    mov r8, r3                  /* r8 = sclk_pin */
    /* r4 = byte to transmit (already in r4) */
    mov r3, #0                  /* r3 = received byte (zeroed) */
    mov r1, #8                  /* bit counter in r1 */
.Lspi_transfer_loop:
    /* Extract MSB of transmit byte: test bit 7 of r4 */
    tst r4, #0x80
    it ne
    addeq r0, r5, #4            /* bit=0: GPCLR (MOSI low) */
    addne r0, r5, #0            /* bit=1: GPSET (MOSI high) */
    mov r2, #1
    lsl r2, r2, r6              /* r2 = 1 << mosi_pin */
    str r2, [r0]
    lsl r4, r4, #1              /* shift tx byte left for next bit */
    /* tSU: setup time before clock edge */
    mov r0, #1
    bl bus_spi_delay_us
    /* Toggle SCLK high */
    add r0, r5, #0              /* GPSET */
    mov r2, #1
    lsl r2, r2, r8              /* r2 = 1 << sclk_pin */
    str r2, [r0]
    mov r0, #1
    bl bus_spi_delay_us
    /* Sample MISO while SCLK high */
    add r0, r5, #8              /* GPLEV */
    ldr r2, [r0]
    lsr r2, r2, r7
    and r2, r2, #1              /* r2 = data bit */
    /* Shift into accumulator */
    lsl r3, r3, #1
    orr r3, r3, r2
    /* Toggle SCLK low */
    add r0, r5, #4              /* GPCLR */
    mov r2, #1
    lsl r2, r2, r8
    str r2, [r0]
    mov r0, #1
    bl bus_spi_delay_us
    subs r1, r1, #1
    bne .Lspi_transfer_loop
    mov r0, r3                  /* r0 = received byte */
    pop {r5, r6, r7, r8, pc}
.size bus_spi_transfer_byte, .-bus_spi_transfer_byte
