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
 * ARM I2C bit-bang assembly routines (Cortex-M / Cortex-A).
 *
 * Direct hardware communication via memory-mapped GPIO registers.  Thumb-2
 * instruction set.  All routines are leaf functions with predictable
 * instruction count; no writable static, no runtime CPU probing.
 *
 * GPIO register model (per bus, base address in r0):
 *   +0x00 GPSET  write 1<<pin to set pin high   (write-only)
 *   +0x04 GPCLR  write 1<<pin to set pin low    (write-only)
 *   +0x08 GPLEV  read pin level                 (read-only)
 *
 * Calling convention: AAPCS.
 *   r0 = arg1, r1 = arg2, r2 = arg3, r3 = arg4
 *   Return in r0.
 */

.syntax unified
.thumb
.cpu cortex-m4
.fpu softvfp
.section .text.i2c_bitbang_arm

/* ------------------------------------------------------------------ */
/* Low-level GPIO primitives (shared with SPI)                        */
/* ------------------------------------------------------------------ */

/*
 * void bus_gpio_set(uint32_t gpio_base, uint32_t pin, uint8_t value)
 *
 * Set a single GPIO pin high (value != 0) or low (value == 0).
 */
.globl bus_gpio_set
.type bus_gpio_set, %function
.thumb_func
bus_gpio_set:
    /* r0 = gpio_base, r1 = pin, r2 = value */
    cmp r2, #0
    ite ne                      /* if (r2 != 0) high, else low */
    addne r3, r0, #0            /* r3 = base + GPSET */
    addeq r3, r0, #4            /* r3 = base + GPCLR */
    mov r4, #1
    lsl r4, r4, r1              /* r4 = 1 << pin */
    str r4, [r3]
    bx lr
.size bus_gpio_set, .-bus_gpio_set

/*
 * uint8_t bus_gpio_get(uint32_t gpio_base, uint32_t pin)
 *
 * Read a single GPIO pin.  Returns 0 or 1 in r0.
 */
.globl bus_gpio_get
.type bus_gpio_get, %function
.thumb_func
bus_gpio_get:
    /* r0 = gpio_base, r1 = pin */
    add r2, r0, #8              /* r2 = base + GPLEV */
    ldr r3, [r2]                /* r3 = level register */
    lsr r3, r3, r1              /* shift pin bit to bit 0 */
    and r0, r3, #1              /* mask to single bit */
    bx lr
.size bus_gpio_get, .-bus_gpio_get

/* ------------------------------------------------------------------ */
/* I2C delay                                                          */
/* ------------------------------------------------------------------ */

/*
 * void bus_i2c_delay_us(uint32_t us)
 *
 * Cycle-calibrated delay using nested subtract-and-branch loops.
 * On a Cortex-M4 at 48 MHz the inner loop is ~4 instructions per pass.
 */
.globl bus_i2c_delay_us
.type bus_i2c_delay_us, %function
.thumb_func
bus_i2c_delay_us:
    /* r0 = us */
    cmp r0, #0
    beq .Ldelay_done
    mov r1, r0                  /* r1 = remaining us */
.Ldelay_loop:
    mov r2, #200                /* ~200 iterations per us at 48 MHz */
.Ldelay_inner:
    subs r2, r2, #1
    bne .Ldelay_inner
    subs r1, r1, #1
    bne .Ldelay_loop
.Ldelay_done:
    bx lr
.size bus_i2c_delay_us, .-bus_i2c_delay_us

/* ------------------------------------------------------------------ */
/* I2C start / stop conditions                                        */
/* ------------------------------------------------------------------ */

/*
 * void bus_i2c_start(uint32_t gpio_base, uint32_t sda_pin, uint32_t scl_pin)
 *
 * Generate I2C start condition: SDA goes low while SCL is high.
 * Assumes SCL is high and SDA is high (bus idle) on entry.
 */
.globl bus_i2c_start
.type bus_i2c_start, %function
.thumb_func
bus_i2c_start:
    /* r0 = gpio_base, r1 = sda_pin, r2 = scl_pin */
    push {r4, r5, lr}           /* save across bl calls */
    mov r4, r0                  /* r4 = gpio_base */
    mov r5, r1                  /* r5 = sda_pin */
    /* Ensure SDA high while SCL high */
    add r0, r4, #0              /* GPSET */
    mov r1, #1
    lsl r1, r1, r5              /* r1 = 1 << sda_pin */
    str r1, [r0]
    /* tSU:STA setup time */
    mov r0, #4
    bl bus_i2c_delay_us
    /* Pull SDA low while SCL high -> start condition */
    add r0, r4, #4              /* GPCLR */
    mov r1, #1
    lsl r1, r1, r5
    str r1, [r0]
    mov r0, #4
    bl bus_i2c_delay_us
    pop {r4, r5, pc}            /* restore and return */
.size bus_i2c_start, .-bus_i2c_start

/*
 * void bus_i2c_stop(uint32_t gpio_base, uint32_t sda_pin, uint32_t scl_pin)
 *
 * Generate I2C stop condition: SDA goes high while SCL is high.
 * Assumes SCL is high and SDA is low on entry.
 */
.globl bus_i2c_stop
.type bus_i2c_stop, %function
.thumb_func
bus_i2c_stop:
    /* r0 = gpio_base, r1 = sda_pin, r2 = scl_pin */
    push {r4, r5, lr}
    mov r4, r0                  /* r4 = gpio_base */
    mov r5, r1                  /* r5 = sda_pin */
    /* Ensure SDA low while SCL high */
    add r0, r4, #4              /* GPCLR */
    mov r1, #1
    lsl r1, r1, r5
    str r1, [r0]
    mov r0, #4
    bl bus_i2c_delay_us
    /* Release SDA high while SCL high -> stop condition */
    add r0, r4, #0              /* GPSET */
    mov r1, #1
    lsl r1, r1, r5
    str r1, [r0]
    mov r0, #4
    bl bus_i2c_delay_us
    pop {r4, r5, pc}
.size bus_i2c_stop, .-bus_i2c_stop

/* ------------------------------------------------------------------ */
/* I2C bit read/write                                                 */
/* ------------------------------------------------------------------ */

/*
 * uint8_t bus_i2c_write_bit(uint32_t gpio_base, uint32_t sda_pin,
 *                           uint32_t scl_pin, uint8_t bit)
 *
 * Drive SDA to `bit`, pulse SCL high then low.  Sample SDA while SCL
 * high for ACK/NACK.  Returns 0 for ACK, 1 for NACK.
 */
.globl bus_i2c_write_bit
.type bus_i2c_write_bit, %function
.thumb_func
bus_i2c_write_bit:
    /* r0 = gpio_base, r1 = sda_pin, r2 = scl_pin, r3 = bit */
    push {r4, r5, r6, lr}
    mov r4, r0                  /* r4 = gpio_base */
    mov r5, r1                  /* r5 = sda_pin */
    mov r6, r2                  /* r6 = scl_pin */
    /* Drive SDA according to bit */
    cmp r3, #0
    ite eq
    addeq r0, r4, #4            /* bit==0: GPCLR */
    addne r0, r4, #0            /* bit!=0: GPSET */
    mov r1, #1
    lsl r1, r1, r5              /* r1 = 1 << sda_pin */
    str r1, [r0]
    /* tSU:DAT setup time */
    mov r0, #2
    bl bus_i2c_delay_us
    /* Pulse SCL high */
    add r0, r4, #0              /* GPSET */
    mov r1, #1
    lsl r1, r1, r6              /* r1 = 1 << scl_pin */
    str r1, [r0]
    /* tHIGH: SCL high time */
    mov r0, #2
    bl bus_i2c_delay_us
    /* Read SDA while SCL high (sample ACK/NACK) */
    add r0, r4, #8              /* GPLEV */
    ldr r1, [r0]
    lsr r1, r1, r5
    and r0, r1, #1              /* r0 = 0=ACK, 1=NACK */
    /* Pulse SCL low */
    add r0, r4, #4              /* GPCLR */
    mov r1, #1
    lsl r1, r1, r6
    str r1, [r0]
    /* tLOW: SCL low time */
    mov r0, #2
    bl bus_i2c_delay_us
    pop {r4, r5, r6, pc}        /* r0 = ACK/NACK */
.size bus_i2c_write_bit, .-bus_i2c_write_bit

/*
 * uint8_t bus_i2c_read_bit(uint32_t gpio_base, uint32_t sda_pin,
 *                          uint32_t scl_pin)
 *
 * Release SDA, pulse SCL high, read SDA while SCL high, then SCL low.
 * Returns the data bit in r0.
 */
.globl bus_i2c_read_bit
.type bus_i2c_read_bit, %function
.thumb_func
bus_i2c_read_bit:
    /* r0 = gpio_base, r1 = sda_pin, r2 = scl_pin */
    push {r4, r5, r6, lr}
    mov r4, r0                  /* r4 = gpio_base */
    mov r5, r1                  /* r5 = sda_pin */
    mov r6, r2                  /* r6 = scl_pin */
    /* Release SDA (set high) for read */
    add r0, r4, #0              /* GPSET */
    mov r1, #1
    lsl r1, r1, r5
    str r1, [r0]
    /* tSU:DAT setup time */
    mov r0, #2
    bl bus_i2c_delay_us
    /* Pulse SCL high */
    add r0, r4, #0              /* GPSET */
    mov r1, #1
    lsl r1, r1, r6
    str r1, [r0]
    /* tHIGH: read SDA while SCL high */
    mov r0, #2
    bl bus_i2c_delay_us
    /* Sample SDA */
    add r0, r4, #8              /* GPLEV */
    ldr r1, [r0]
    lsr r1, r1, r5
    and r0, r1, #1              /* r0 = data bit */
    /* Pulse SCL low */
    add r0, r4, #4              /* GPCLR */
    mov r1, #1
    lsl r1, r1, r6
    str r1, [r0]
    /* tLOW */
    mov r0, #2
    bl bus_i2c_delay_us
    pop {r4, r5, r6, pc}        /* r0 = data bit */
.size bus_i2c_read_bit, .-bus_i2c_read_bit

/* ------------------------------------------------------------------ */
/* I2C byte read/write                                                */
/* ------------------------------------------------------------------ */

/*
 * uint8_t bus_i2c_write_byte(uint32_t gpio_base, uint32_t sda_pin,
 *                             uint32_t scl_pin, uint8_t byte)
 *
 * Write MSB-first, 8 bits, then read ACK/NACK.  Returns 0 for ACK, 1 for NACK.
 */
.globl bus_i2c_write_byte
.type bus_i2c_write_byte, %function
.thumb_func
bus_i2c_write_byte:
    /* r0 = gpio_base, r1 = sda_pin, r2 = scl_pin, r3 = byte */
    push {r4, r5, r6, r7, lr}
    mov r4, r0                  /* r4 = gpio_base */
    mov r5, r1                  /* r5 = sda_pin */
    mov r6, r2                  /* r6 = scl_pin */
    mov r7, r3                  /* r7 = byte to write */
    mov r3, #8                  /* bit counter */
.Lwrite_byte_loop:
    /* Test bit 7 of r7 (MSB) */
    tst r7, #0x80
    it ne
    addeq r0, r4, #4            /* bit=0: GPCLR */
    addne r0, r4, #0            /* bit=1: GPSET */
    mov r1, #1
    lsl r1, r1, r5              /* r1 = 1 << sda_pin */
    str r1, [r0]
    lsl r7, r7, #1              /* shift left for next bit */
    /* Pulse SCL high */
    add r0, r4, #0              /* GPSET */
    mov r1, #1
    lsl r1, r1, r6
    str r1, [r0]
    mov r0, #2
    bl bus_i2c_delay_us
    subs r3, r3, #1
    bne .Lwrite_byte_loop
    /* 8 bits done: read ACK/NACK */
    /* Release SDA (set high) */
    add r0, r4, #0              /* GPSET */
    mov r1, #1
    lsl r1, r1, r5
    str r1, [r0]
    mov r0, #2
    bl bus_i2c_delay_us
    /* Pulse SCL high for ACK slot */
    add r0, r4, #0              /* GPSET */
    mov r1, #1
    lsl r1, r1, r6
    str r1, [r0]
    mov r0, #2
    bl bus_i2c_delay_us
    /* Sample SDA (ACK=0, NACK=1) */
    add r0, r4, #8              /* GPLEV */
    ldr r1, [r0]
    lsr r1, r1, r5
    and r0, r1, #1              /* r0 = ACK/NACK */
    /* Pulse SCL low */
    add r0, r4, #4              /* GPCLR */
    mov r1, #1
    lsl r1, r1, r6
    str r1, [r0]
    mov r0, #2
    bl bus_i2c_delay_us
    pop {r4, r5, r6, r7, pc}    /* r0 = ACK/NACK */
.size bus_i2c_write_byte, .-bus_i2c_write_byte

/*
 * uint8_t bus_i2c_read_byte(uint32_t gpio_base, uint32_t sda_pin,
 *                            uint32_t scl_pin, uint8_t ack)
 *
 * Read 8 bits MSB-first, then send ACK (ack=0) or NACK (ack=1).
 * Returns the byte read in r0.
 */
.globl bus_i2c_read_byte
.type bus_i2c_read_byte, %function
.thumb_func
bus_i2c_read_byte:
    /* r0 = gpio_base, r1 = sda_pin, r2 = scl_pin, r3 = ack */
    push {r4, r5, r6, r7, lr}
    mov r4, r0                  /* r4 = gpio_base */
    mov r5, r1                  /* r5 = sda_pin */
    mov r6, r2                  /* r6 = scl_pin */
    mov r7, r3                  /* r7 = ack */
    mov r3, #0                  /* r3 = received byte */
    mov r1, #8                  /* bit counter in r1 */
.Lread_byte_loop:
    /* Release SDA for read */
    add r0, r4, #0              /* GPSET */
    mov r2, #1
    lsl r2, r2, r5
    str r2, [r0]
    /* Pulse SCL high */
    add r0, r4, #0              /* GPSET */
    mov r2, #1
    lsl r2, r2, r6
    str r2, [r0]
    mov r0, #2
    bl bus_i2c_delay_us
    /* Sample SDA */
    add r0, r4, #8              /* GPLEV */
    ldr r2, [r0]
    lsr r2, r2, r5
    and r2, r2, #1              /* r2 = data bit */
    /* Shift into accumulator */
    lsl r3, r3, #1
    orr r3, r3, r2
    /* Pulse SCL low */
    add r0, r4, #4              /* GPCLR */
    mov r2, #1
    lsl r2, r2, r6
    str r2, [r0]
    mov r0, #2
    bl bus_i2c_delay_us
    subs r1, r1, #1
    bne .Lread_byte_loop
    /* 8 bits read: send ACK or NACK */
    cmp r7, #0
    ite eq
    addeq r0, r4, #4            /* ack==0: GPCLR (SDA low) */
    addne r0, r4, #0            /* ack!=0: GPSET (SDA high) */
    mov r1, #1
    lsl r1, r1, r5
    str r1, [r0]
    mov r0, #2
    bl bus_i2c_delay_us
    /* Pulse SCL high for ACK/NACK bit */
    add r0, r4, #0              /* GPSET */
    mov r1, #1
    lsl r1, r1, r6
    str r1, [r0]
    mov r0, #2
    bl bus_i2c_delay_us
    /* Pulse SCL low */
    add r0, r4, #4              /* GPCLR */
    mov r1, #1
    lsl r1, r1, r6
    str r1, [r0]
    mov r0, #2
    bl bus_i2c_delay_us
    mov r0, r3                  /* r0 = received byte */
    pop {r4, r5, r6, r7, pc}
.size bus_i2c_read_byte, .-bus_i2c_read_byte
