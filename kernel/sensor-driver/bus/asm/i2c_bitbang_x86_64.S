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
 * x86-64 I2C bit-bang assembly routines for Linux userspace.
 *
 * Uses the Linux kernel I2C device interface (/dev/i2c-N) via ioctl()
 * I2C_RDWR instead of memory-mapped GPIO.  Direct physical address access
 * (mov [r8], eax) would segfault on Linux userspace because arbitrary
 * physical addresses are not mapped into the process address space.
 *
 * GPIO primitives (bus_gpio_set / bus_gpio_get) are no-ops here: the kernel
 * I2C driver owns the GPIO lines and userspace must not toggle them.
 *
 * Calling convention: System V AMD64 ABI.
 *   %rdi = arg1, %rsi = arg2, %rdx = arg3, %rcx = arg4
 *   Return in %al / %rax.
 *   %rbx, %rbp, %r12-%r15 are callee-saved; we preserve them where used.
 */

.intel_syntax noprefix

.text

/* i2c_msg and i2c_rdrw_ioctl_data layouts (struct packing is natural):
 *
 * struct i2c_msg {
 *     uint16_t addr;   // offset 0
 *     uint16_t flags;  // offset 2
 *     uint16_t len;    // offset 4
 *     uint8_t  *buf;   // offset 8 (pointer)
 * };
 * sizeof(struct i2c_msg) = 16
 *
 * struct i2c_rdwr_ioctl_data {
 *     struct i2c_msg *msgs;  // offset 0
 *     uint32_t      nmsgs;   // offset 8
 * };
 * sizeof(struct i2c_rdwr_ioctl_data) = 16
 *
 * I2C_RDWR = 0x0707
 * I2C_SLAVE = 0x0703
 */

#define I2C_RDWR 0x0707
#define I2C_SLAVE 0x0703
#define I2C_M_RD 0x0001

/* ------------------------------------------------------------------ */
/* Low-level GPIO primitives (no-ops on Linux userspace)              */
/* ------------------------------------------------------------------ */

/*
 * void bus_gpio_set(uint32_t gpio_base, uint32_t pin, uint8_t value)
 *
 * No-op on Linux userspace: the kernel I2C driver owns the GPIO lines.
 * Kept for ABI compatibility with the shared SPI assembly and with ARM.
 */
.globl bus_gpio_set
.type bus_gpio_set, @function
bus_gpio_set:
    ret
.size bus_gpio_set, .-bus_gpio_set

/*
 * uint8_t bus_gpio_get(uint32_t gpio_base, uint32_t pin)
 *
 * No-op on Linux userspace.  Returns 0.
 */
.globl bus_gpio_get
.type bus_gpio_get, @function
bus_gpio_get:
    xor eax, eax
    ret
.size bus_gpio_get, .-bus_gpio_get

/* ------------------------------------------------------------------ */
/* Linux userspace I2C device open / ioctl helpers                    */
/* ------------------------------------------------------------------ */

/*
 * int32_t bus_i2c_linux_open(const char *path)
 *
 * Calls open(path, O_RDWR).  Returns fd in eax, or -1 on error.
 */
.globl bus_i2c_linux_open
.type bus_i2c_linux_open, @function
bus_i2c_linux_open:
    /* SYS_openat = 256; use open(2) via syscall 2 (SYS_open) is 64-bit? */
    /* On x86-64 Linux: syscall number for openat is 256. We use openat(AT_FDCWD=...). */
    mov eax, 256                 /* SYS_openat */
    mov edi, -100                /* AT_FDCWD */
    mov rsi, rdi                 /* const char *pathname */
    mov edx, 0x0002              /* O_RDWR */
    mov r10, 0                   /* mode (unused for O_RDWR) */
    syscall
    ret
.size bus_i2c_linux_open, .-bus_i2c_linux_open

/*
 * void bus_i2c_linux_close(int32_t fd)
 *
 * Calls close(fd).
 */
.globl bus_i2c_linux_close
.type bus_i2c_linux_close, @function
bus_i2c_linux_close:
    mov eax, 3                   /* SYS_close */
    mov edi, edi                 /* fd already in edi */
    syscall
    ret
.size bus_i2c_linux_close, .-bus_i2c_linux_close

/*
 * int32_t bus_i2c_linux_ioctl(int32_t fd, uint32_t request, void *arg)
 *
 * Calls ioctl(fd, request, arg).  Returns result in eax.
 */
.globl bus_i2c_linux_ioctl
.type bus_i2c_linux_ioctl, @function
bus_i2c_linux_ioctl:
    mov eax, 16                  /* SYS_ioctl */
    /* edi = fd, esi = request, edx = arg already set by caller */
    syscall
    ret
.size bus_i2c_linux_ioctl, .-bus_i2c_linux_ioctl

/* ------------------------------------------------------------------ */
/* I2C device handle management                                       */
/* ------------------------------------------------------------------ */

/*
 * The caller (C++ layer) holds a file descriptor for /dev/i2c-N and the
 * slave address.  Because the assembly routines are stateless leaf
 * functions, the C++ wrapper passes the fd and address explicitly.
 *
 * void bus_i2c_start(int32_t fd, uint16_t addr)
 *   -> issues I2C_SLAVE ioctl to set the slave address (start of a transfer).
 *
 * void bus_i2c_stop(int32_t fd)
 *   -> no-op on Linux (no stop condition needed at userspace level).
 *
 * uint8_t bus_i2c_write_byte(int32_t fd, uint16_t addr, uint8_t byte)
 *   -> builds a single i2c_msg (write) and issues I2C_RDWR ioctl.
 *
 * uint8_t bus_i2c_read_byte(int32_t fd, uint16_t addr, uint8_t ack)
 *   -> builds a single i2c_msg (read) and issues I2C_RDWR ioctl.
 *
 * The existing C++ API signatures use (gpio_base, sda_pin, scl_pin, ...).
 * To keep the header declarations unchanged we reinterpret:
 *   rdi = fd        (was gpio_base)
 *   rsi = addr      (was sda_pin - now carries the 7-bit slave address)
 *   rdx = byte/ack  (was scl_pin - now carries byte or ack flag)
 * This is an internal ABI between the C++ wrapper and this assembly file.
 */

/*
 * void bus_i2c_start(int32_t fd, uint16_t addr)
 *
 * Set the I2C slave address via I2C_SLAVE ioctl.  This is the "start" of
 * a Linux userspace I2C transaction.
 */
.globl bus_i2c_start
.type bus_i2c_start, @function
bus_i2c_start:
    /* %rdi = fd, %esi = addr (7-bit in low byte) */
    mov edx, I2C_SLAVE
    mov esi, esi                 /* addr already in esi */
    mov eax, 16                  /* SYS_ioctl */
    syscall
    ret
.size bus_i2c_start, .-bus_i2c_start

/*
 * void bus_i2c_stop(int32_t fd)
 *
 * No-op on Linux userspace I2C device: the kernel generates STOP
 * automatically at the end of each I2C_RDWR message sequence.
 */
.globl bus_i2c_stop
.type bus_i2c_stop, @function
bus_i2c_stop:
    ret
.size bus_i2c_stop, .-bus_i2c_stop

/* ------------------------------------------------------------------ */
/* I2C byte read/write via ioctl(I2C_RDWR)                            */
/* ------------------------------------------------------------------ */

/*
 * uint8_t bus_i2c_write_byte(int32_t fd, uint16_t addr, uint8_t byte)
 *
 * Build a one-message write transfer:
 *   i2c_msg { addr, flags=0, len=1, buf=&byte }
 *   i2c_rdwr_ioctl_data { &msg, nmsgs=1 }
 * Issue ioctl(I2C_RDWR).  Returns 0 for ACK, 1 for NACK/error.
 */
.globl bus_i2c_write_byte
.type bus_i2c_write_byte, @function
bus_i2c_write_byte:
    /* %rdi = fd, %rsi = addr, %rdx = byte */
    push rbp
    mov rbp, rsp
    sub rsp, 32                  /* stack space for i2c_msg(16) + rdwr(16) */
    /* i2c_msg at rsp+0 */
    mov word ptr [rsp+0], si         /* addr */
    mov word ptr [rsp+2], 0          /* flags = 0 (write) */
    mov word ptr [rsp+4], 1          /* len = 1 */
    mov qword ptr [rsp+8], r10       /* buf = pointer to the byte */
    /* We need a pointer to the byte.  Store the byte on stack and point buf at it. */
    mov byte ptr [rsp+24], dl        /* spill byte to a known stack slot */
    lea r10, [rsp+24]                /* r10 = pointer to the byte */
    mov qword ptr [rsp+8], r10       /* i2c_msg.buf = r10 */
    /* i2c_rdwr_ioctl_data at rsp+16 */
    lea r10, [rsp+0]                 /* msgs = &i2c_msg */
    mov qword ptr [rsp+16], r10
    mov dword ptr [rsp+24], 1        /* nmsgs = 1 */
    /* ioctl(fd, I2C_RDWR, &rdwr) */
    mov eax, 16                  /* SYS_ioctl */
    mov esi, I2C_RDWR
    lea rdx, [rsp+16]            /* arg = &i2c_rdwr_ioctl_data */
    syscall
    /* eax = return value: 0 on success, -1 on error */
    test eax, eax
    jns .Lwrite_byte_ok          /* >= 0 => success (ACK) */
    mov eax, 1                   /* error => NACK */
    jmp .Lwrite_byte_done
.Lwrite_byte_ok:
    xor eax, eax                 /* success => ACK (0) */
.Lwrite_byte_done:
    add rsp, 32
    pop rbp
    ret
.size bus_i2c_write_byte, .-bus_i2c_write_byte

/*
 * uint8_t bus_i2c_read_byte(int32_t fd, uint16_t addr, uint8_t ack)
 *
 * Build a one-message read transfer:
 *   i2c_msg { addr, flags=I2C_M_RD, len=1, buf=&byte }
 *   i2c_rdwr_ioctl_data { &msg, nmsgs=1 }
 * Issue ioctl(I2C_RDWR).  Returns the byte read in al.
 * (The ack/nak is handled by the kernel; userspace reads one byte.)
 */
.globl bus_i2c_read_byte
.type bus_i2c_read_byte, @function
bus_i2c_read_byte:
    /* %rdi = fd, %rsi = addr, %rdx = ack (ignored by kernel read) */
    push rbp
    mov rbp, rsp
    sub rsp, 32                  /* stack space for i2c_msg(16) + rdwr(16) + byte */
    /* i2c_msg at rsp+0 */
    mov word ptr [rsp+0], si         /* addr */
    mov word ptr [rsp+2], I2C_M_RD   /* flags = read */
    mov word ptr [rsp+4], 1          /* len = 1 */
    lea r10, [rsp+28]                /* buf = &stack byte slot */
    mov qword ptr [rsp+8], r10
    /* i2c_rdwr_ioctl_data at rsp+16 */
    lea r10, [rsp+0]                 /* msgs = &i2c_msg */
    mov qword ptr [rsp+16], r10
    mov dword ptr [rsp+24], 1        /* nmsgs = 1 */
    /* ioctl(fd, I2C_RDWR, &rdwr) */
    mov eax, 16                  /* SYS_ioctl */
    mov esi, I2C_RDWR
    lea rdx, [rsp+16]            /* arg = &i2c_rdwr_ioctl_data */
    syscall
    test eax, eax
    jns .Lread_byte_ok
    xor eax, eax                 /* on error return 0 */
    jmp .Lread_byte_done
.Lread_byte_ok:
    movzx eax, byte ptr [rsp+28]     /* eax = byte read */
.Lread_byte_done:
    add rsp, 32
    pop rbp
    ret
.size bus_i2c_read_byte, .-bus_i2c_read_byte

/* ------------------------------------------------------------------ */
/* I2C delay                                                          */
/* ------------------------------------------------------------------ */

/*
 * void bus_i2c_delay_us(uint32_t us)
 *
 * Best-effort microsecond delay using a nop loop.  On Linux userspace a
 * real delay would use clock_nanosleep, but a tight nop loop keeps the
 * assembly leaf and avoids extra syscall overhead for the tiny bit-bang
 * delays the I2C layer uses between operations.
 */
.globl bus_i2c_delay_us
.type bus_i2c_delay_us, @function
bus_i2c_delay_us:
    test edi, edi                /* edi = us */
    jz .Ldelay_done
    mov ecx, edi                 /* ecx = remaining us */
.Ldelay_loop:
    rep nop
    dec ecx
    jnz .Ldelay_loop
.Ldelay_done:
    ret
.size bus_i2c_delay_us, .-bus_i2c_delay_us
