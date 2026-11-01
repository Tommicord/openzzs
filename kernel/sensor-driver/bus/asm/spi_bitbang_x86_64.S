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
 * x86-64 SPI bit-bang assembly routines.
 *
 * Direct hardware communication via memory-mapped GPIO registers.  GPIO
 * primitives (bus_gpio_set / bus_gpio_get) are defined in
 * i2c_bitbang_x86_64.S and referenced here as external symbols.
 *
 * GPIO register model (per bus, base address in %rdi):
 *   +0x00 GPSET  write 1<<pin to set pin high   (write-only)
 *   +0x04 GPCLR  write 1<<pin to set pin low    (write-only)
 *   +0x08 GPLEV  read pin level                 (read-only)
 *
 * Calling convention: System V AMD64 ABI.
 *   %rdi = arg1, %rsi = arg2, %rdx = arg3, %rcx = arg4, %r8 = arg5
 *   Return in %al / %rax.
 */

.intel_syntax noprefix

.text

/* External GPIO primitives defined in i2c_bitbang_x86_64.S */
.globl bus_gpio_set
.globl bus_gpio_get

/* ------------------------------------------------------------------ */
/* SPI delay                                                          */
/* ------------------------------------------------------------------ */

/*
 * void bus_spi_delay_us(uint32_t us)
 */
.globl bus_spi_delay_us
.type bus_spi_delay_us, @function
bus_spi_delay_us:
    test edi, edi                 /* edi = us */
    jz .Lspi_delay_done
    mov ecx, edi                  /* ecx = remaining us */
.Lspi_delay_loop:
    rep nop
    dec ecx
    jnz .Lspi_delay_loop
.Lspi_delay_done:
    ret
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
.type bus_spi_cs_assert, @function
bus_spi_cs_assert:
    /* %rdi = gpio_base, %rsi = cs_pin */
    mov ecx, esi
    mov eax, 1
    shl eax, cl
    mov r8d, edi
    add r8, 0x04                 /* GPCLR -> CS low (assert) */
    mov [r8], eax
    ret
.size bus_spi_cs_assert, .-bus_spi_cs_assert

/*
 * void bus_spi_cs_deassert(uint32_t gpio_base, uint32_t cs_pin)
 *
 * Deassert chip select (drive CS high).
 */
.globl bus_spi_cs_deassert
.type bus_spi_cs_deassert, @function
bus_spi_cs_deassert:
    /* %rdi = gpio_base, %rsi = cs_pin */
    mov ecx, esi
    mov eax, 1
    shl eax, cl
    mov r8d, edi
    add r8, 0x00                 /* GPSET -> CS high (deassert) */
    mov [r8], eax
    ret
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
.type bus_spi_write_bit, @function
bus_spi_write_bit:
    /* %rdi = gpio_base, %rsi = mosi_pin, %rdx = sclk_pin, %rcx = bit */
    mov r8, rdi                   /* r8 = gpio_base */
    mov r9, rsi                   /* r9 = mosi_pin */
    mov r10, rdx                  /* r10 = sclk_pin */
    /* Drive MOSI according to bit */
    test cl, cl
    jz .Lspi_write_bit_mosi_low
    /* MOSI high */
    mov ecx, r9d
    mov eax, 1
    shl eax, cl
    mov r11, r8
    add r11, 0x00                 /* GPSET */
    mov [r11], eax
    jmp .Lspi_write_bit_mosi_done
.Lspi_write_bit_mosi_low:
    mov ecx, r9d
    mov eax, 1
    shl eax, cl
    mov r11, r8
    add r11, 0x04                 /* GPCLR */
    mov [r11], eax
.Lspi_write_bit_mosi_done:
    /* tSU: setup time before clock edge */
    mov edi, 1
    call bus_spi_delay_us
    /* Toggle SCLK high */
    mov ecx, r10d
    mov eax, 1
    shl eax, cl
    mov r11, r8
    add r11, 0x00                 /* GPSET */
    mov [r11], eax
    mov edi, 1
    call bus_spi_delay_us
    /* Toggle SCLK low */
    mov ecx, r10d
    mov eax, 1
    shl eax, cl
    mov r11, r8
    add r11, 0x04                 /* GPCLR */
    mov [r11], eax
    mov edi, 1
    call bus_spi_delay_us
    ret
.size bus_spi_write_bit, .-bus_spi_write_bit

/*
 * uint8_t bus_spi_read_bit(uint32_t gpio_base, uint32_t miso_pin,
 *                          uint32_t sclk_pin)
 *
 * Toggle SCLK high, sample MISO while SCLK high, then SCLK low.
 * Returns the data bit in %al.
 */
.globl bus_spi_read_bit
.type bus_spi_read_bit, @function
bus_spi_read_bit:
    /* %rdi = gpio_base, %rsi = miso_pin, %rdx = sclk_pin */
    mov r8, rdi                   /* r8 = gpio_base */
    mov r9, rsi                   /* r9 = miso_pin */
    mov r10, rdx                  /* r10 = sclk_pin */
    /* Toggle SCLK high */
    mov ecx, r10d
    mov eax, 1
    shl eax, cl
    mov r11, r8
    add r11, 0x00                 /* GPSET */
    mov [r11], eax
    mov edi, 1
    call bus_spi_delay_us
    /* Sample MISO while SCLK high */
    mov ecx, r9d
    mov r11, r8
    add r11, 0x08                 /* GPLEV */
    mov eax, [r11]
    shr eax, cl
    and eax, 1                    /* al = data bit */
    /* Toggle SCLK low */
    mov ecx, r10d
    mov eax, 1
    shl eax, cl
    mov r11, r8
    add r11, 0x04                 /* GPCLR */
    mov [r11], eax
    mov edi, 1
    call bus_spi_delay_us
    ret                           /* al = data bit */
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
 * Returns the received byte in %al.
 */
.globl bus_spi_transfer_byte
.type bus_spi_transfer_byte, @function
bus_spi_transfer_byte:
    /* %rdi = gpio_base, %rsi = mosi_pin, %rdx = miso_pin,
       %rcx = sclk_pin, %r8 = byte */
    mov r9, rdi                   /* r9 = gpio_base (saved) */
    mov r10, rsi                  /* r10 = mosi_pin */
    mov r11, rdx                  /* r11 = miso_pin */
    mov r12, rcx                  /* r12 = sclk_pin */
    mov r13d, r8d                  /* r13d = byte to transmit */
    xor r14d, r14d                /* r14d = received byte (zeroed) */
    mov ecx, 8                    /* ecx = bit counter */
.Lspi_transfer_loop:
    /* Extract MSB of transmit byte: shift left, CF = old bit 7 */
    shl r13d, 1
    jc .Lspi_tx_bit_one
    /* MOSI low */
    mov eax, 1
    mov ecx, r10d                 /* cl = mosi_pin */
    shl eax, cl                   /* eax = 1 << mosi_pin */
    mov r15, r9
    add r15, 0x04                 /* GPCLR */
    mov [r15], eax
    jmp .Lspi_tx_bit_done
.Lspi_tx_bit_one:
    /* MOSI high */
    mov eax, 1
    mov ecx, r10d                 /* cl = mosi_pin */
    shl eax, cl                   /* eax = 1 << mosi_pin */
    mov r15, r9
    add r15, 0x00                 /* GPSET */
    mov [r15], eax
.Lspi_tx_bit_done:
    /* tSU: setup time before clock edge */
    mov edi, 1
    call bus_spi_delay_us
    /* Toggle SCLK high */
    mov eax, 1
    mov ecx, r12d                 /* cl = sclk_pin */
    shl eax, cl                   /* eax = 1 << sclk_pin */
    mov r15, r9
    add r15, 0x00                 /* GPSET */
    mov [r15], eax
    mov edi, 1
    call bus_spi_delay_us
    /* Sample MISO while SCLK high */
    mov r15, r9
    add r15, 0x08                 /* GPLEV */
    mov eax, [r15]                /* eax = level register */
    mov ecx, r11d                 /* cl = miso_pin */
    shr eax, cl
    and eax, 1                    /* al = data bit */
    /* Shift into accumulator */
    shl r14d, 1
    or r14b, al
    /* Toggle SCLK low */
    mov eax, 1
    mov ecx, r12d                 /* cl = sclk_pin */
    shl eax, cl                   /* eax = 1 << sclk_pin */
    mov r15, r9
    add r15, 0x04                 /* GPCLR */
    mov [r15], eax
    mov edi, 1
    call bus_spi_delay_us
    dec ecx                       /* decrement bit counter */
    jnz .Lspi_transfer_loop
    movzx eax, r14b               /* al = received byte */
    ret
.size bus_spi_transfer_byte, .-bus_spi_transfer_byte
