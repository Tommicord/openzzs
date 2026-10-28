/* Link script for the mllee Cortex-M4 self-test on QEMU's mps2-an385.
 *
 * Code and read-only data sit in the flash alias window at 0x00000000, which
 * is where the Cortex-M4 fetches its vector table; writable data sits in the
 * 4 MiB SRAM at 0x20000000.
 */

MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 4M
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 4M
}

ENTRY(_start)

SECTIONS
{
    .text : {
        KEEP(*(.isr_vector))
        *(.text.mllee.*)
        *(.text*)
        *(.rodata*)
        . = ALIGN(4);
    } > FLASH

    _data_lma = LOADADDR(.data);

    .data : {
        . = ALIGN(4);
        _data_start = .;
        *(.data*)
        . = ALIGN(4);
        _data_end = .;
    } > RAM AT > FLASH

    .bss (NOLOAD) : {
        . = ALIGN(4);
        _bss_start = .;
        *(.bss*)
        *(COMMON)
        . = ALIGN(4);
        _bss_end = .;
    } > RAM

    _stack_top = ORIGIN(RAM) + 64K;

    /DISCARD/ : {
        *(.ARM.exidx*)
        *(.ARM.attributes)
        *(.comment)
    }
}
