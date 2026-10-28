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
 * mllee Cortex-M4 startup and instruction probe for the QEMU `mps2-an385`
 * self-test.
 *
 * The vector table is linked at 0x0 (the AN385 flash alias window) and the
 * stack starts in the 4 MiB SRAM at 0x20000000.  Output goes through ARM
 * semihosting, so the test needs no UART driver and no C runtime.
 *
 * The probe matters because emulators are not complete: QEMU's Cortex-M4 model
 * executes SMLAD/SMLADX/USAD8/SSAT but leaves out SXTB16, SMLABB and QADD,
 * which a real Armv7E-M core implements (CMSIS documents __SXTB16 and __QADD
 * for Cortex-M4/M7).  Each probe runs one guarded instruction; the fault
 * handler skips it and reports the instruction as unavailable, so the test
 * knows which kernel paths it can actually exercise instead of passing them
 * unverified.
 */

.syntax unified
.thumb
.cpu cortex-m4

    .section .isr_vector, "a", %progbits
    .align 2
    .globl _vector_table
_vector_table:
    .word _stack_top
    .word _start + 1
    .word bad_handler + 1         /* NMI */
    .word bad_handler + 1         /* HardFault */
    .size _vector_table, .-_vector_table

    .section .bss
    .align 2
    .globl zzs_m4_in_probe
zzs_m4_in_probe:
    .word 0
    .globl zzs_m4_probe_ok
zzs_m4_probe_ok:
    .word 0
    .align 2
zzs_m4_hexbuf:
    .byte 0, 0, 0, 0
    .align 2
zzs_m4_exit_reason:
    .word 0, 0

    .section .text
    .align 2

/* extern "C" void zzs_mllee_shm_write0(const char *text)
 *
 * SYS_WRITE0 (0x04): r1 is a NUL-terminated string.  QEMU prints it to stdout.
 */
    .globl zzs_mllee_shm_write0
    .type zzs_mllee_shm_write0, %function
    .thumb_func
zzs_mllee_shm_write0:
    mov r1, r0
    movs r0, #4
    bkpt #0xab
    bx lr
    .size zzs_mllee_shm_write0, .-zzs_mllee_shm_write0

/* extern "C" void zzs_mllee_shm_exit(uint32_t failing)
 *
 * SYS_EXIT (0x18) with ARMv7-M's block form: r1 points at {Angel reason, exit
 * code}.  ADP_Stopped_ApplicationExit (0xF0002) is the normal-exit reason.  Note
 * that QEMU 10.2's mps2-an385 returns exit status 1 for this path regardless of
 * the reason, which is why tools/run_mllee_m4_qemu.sh judges the printed
 * verdict line rather than the emulator's status.
 */
    .globl zzs_mllee_shm_exit
    .type zzs_mllee_shm_exit, %function
    .thumb_func
zzs_mllee_shm_exit:
    ldr r1, =zzs_m4_exit_reason
    ldr r2, =0xF0002
    cmp r0, #0
    beq 1f
    ldr r2, =0xF0001
1:
    str r2, [r1, #0]
    movs r2, #0
    str r2, [r1, #4]
    movs r0, #0x18
    bkpt #0xab
    b .
    .size zzs_mllee_shm_exit, .-zzs_mllee_shm_exit

/* Entry point: hand over to the C++ self-test, then report its result. */
    .globl _start
    .type _start, %function
    .thumb_func
_start:
    bl main
    bl zzs_mllee_shm_exit
    b .
    .size _start, .-_start

/*
 * HardFault handler.  Inside a probe it skips the offending instruction and
 * clears zzs_m4_probe_ok; anywhere else it reports a fault and stops, so a
 * genuine kernel bug never looks like a pass.
 */
    .globl bad_handler
    .type bad_handler, %function
    .thumb_func
bad_handler:
    ldr r3, =zzs_m4_in_probe
    ldr r2, [r3]
    cbz r2, .Lbad_fatal
    movs r2, #0
    str r2, [r3]                  /* leave probe mode */
    ldr r3, =zzs_m4_probe_ok
    movs r2, #0
    str r2, [r3]                  /* mark the instruction unavailable */
    ldr r2, [r13, #24]
    ldrh r3, [r2]
    lsrs r3, r3, #11              /* top five bits: 28..31 means 32-bit encoding */
    movs r12, #2
    cmp r3, #27
    it hi
    addhi r12, #2
    add r2, r12
    str r2, [r13, #24]
    bx lr
.Lbad_fatal:
    ldr r0, =.Lmsg_fault
    bl zzs_mllee_shm_write0
    ldr r0, [r13, #24]            /* stacked PC: the faulting instruction */
    bl print_hex32
    ldr r0, =.Lmsg_lr
    bl zzs_mllee_shm_write0
    ldr r0, [r13, #20]            /* stacked LR: its caller */
    bl print_hex32
    ldr r0, =.Lmsg_nl
    bl zzs_mllee_shm_write0
    movs r0, #1
    bl zzs_mllee_shm_exit
    b .
    .size bad_handler, .-bad_handler

/* print_hex32: semihost-print r0 as 0x........, most significant nibble first.
 * Reached only from the fatal path, so a syscall per nibble is affordable. */
    .type print_hex32, %function
    .thumb_func
print_hex32:
    push {r4, r5, r6, lr}
    mov r4, r0                  /* value */
    ldr r0, =.Lmsg_hex
    bl zzs_mllee_shm_write0
    movs r5, #28                /* shift of the nibble to emit */
    ldr r6, =zzs_m4_hexbuf
.Lph_loop:
    mov r0, r4
    lsr r0, r0, r5
    ands r0, r0, #15
    ldr r1, =.Lhexdigits
    ldrb r0, [r1, r0]
    movs r2, #0
    strb r0, [r6]
    strb r2, [r6, #1]
    mov r0, r6
    bl zzs_mllee_shm_write0
    subs r5, #4
    bpl .Lph_loop
    pop {r4, r5, r6, lr}
    bx lr
    .size print_hex32, .-print_hex32

/*
 * probe <name>, <instruction>
 *
 * Returns 1 when the model executes the instruction, 0 when it faults.  r0 and
 * r1 are the probe operands and r13 points at a private 8-byte scratch slot, so
 * a store probe cannot clobber its caller.  The guarded instruction must not
 * depend on anything else.
 */
    .macro probe_name label, instruction:vararg
    .globl \label
    .type \label, %function
    .thumb_func
\label:
    push {r4, r5, lr}
    ldr r3, =zzs_m4_probe_ok
    movs r2, #1
    str r2, [r3]
    ldr r3, =zzs_m4_in_probe
    movs r2, #1
    str r2, [r3]
    movs r0, #0x11
    movs r1, #0x22
    \instruction
    ldr r3, =zzs_m4_in_probe
    movs r2, #0
    str r2, [r3]
    ldr r3, =zzs_m4_probe_ok
    ldr r0, [r3]
    pop {r4, r5, pc}
    .size \label, .-\label
    .endm

    probe_name zzs_m4_probe_sxtb16, sxtb16 r0, r1
    probe_name zzs_m4_probe_sxtab16, sxtab16 r0, r1, r0
    probe_name zzs_m4_probe_smlad, smlad r0, r1, r0, r1
    probe_name zzs_m4_probe_smlabb, smlabb r0, r1, r0, r1
    probe_name zzs_m4_probe_smulbb, smulbb r0, r1, r0
    probe_name zzs_m4_probe_ssat, ssat r0, #8, r1
    probe_name zzs_m4_probe_usada8, usada8 r0, r1, r0, r1
    probe_name zzs_m4_probe_qadd, qadd r0, r1, r0
    probe_name zzs_m4_probe_strd, strd r0, r1, [r13]

.Lmsg_fault:
    .asciz "M4-SELFTEST: HARDFAULT pc="
.Lmsg_lr:
    .asciz " lr="
.Lmsg_nl:
    .asciz "\n"
.Lmsg_hex:
    .asciz "0x"
.Lhexdigits:
    .ascii "0123456789abcdef"
    .align 2
    .ltorg

.section .note.GNU-stack,"",%progbits
