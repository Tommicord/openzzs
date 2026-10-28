#!/usr/bin/env bash
# Copyright(c) 2026 Tommicord
# SPDX-License-Identifier: MIT
#
# Runs the mllee Cortex-M4 assembly kernels on QEMU's mps2-an385 (a real
# ARMv7E-M model) and checks them against the core's documented semantics.
#
# This is the only place the Thumb kernels are executed: the host suite links
# the x86-64 kernels, so it can never see SMLAD/SXTB16 encodings, and an
# assembler pass does not prove that they are legal on Cortex-M or that the
# lane tricks produce the same bytes as the C++ path.
#
# Usage: tools/run_mllee_m4_qemu.sh [output-dir]
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:-"$root/build-mllee-m4"}
mkdir -p "$out"

cxx=${ZZS_MLLEE_CXX:-arm-none-eabi-g++}
cc=${ZZS_MLLEE_CC:-arm-none-eabi-gcc}
qemu=${ZZS_MLLEE_QEMU:-qemu-system-arm}

for tool in "$cxx" "$cc" "$qemu"; do
    command -v "$tool" >/dev/null 2>&1 || { echo "SKIP: $tool not installed" >&2; exit 0; }
done

arch=(-mcpu=cortex-m4 -mthumb -mfloat-abi=soft)
# The reference side of the test is built for baseline ARMv7-M on purpose: GCC
# happily turns a dot product into SMLABB (which even this model faults on), and
# then the harness would be checking the compiler against itself.
cxx_arch=(-mthumb -march=armv7-m -mfloat-abi=soft)
warn=(-Wall -Wextra)
flags=("${cxx_arch[@]}" -Os -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti
    -fno-asynchronous-unwind-tables -I"$root" "${warn[@]}")

echo "mllee M4 self-test: $cxx on $qemu (mps2-an385)"

"$cc" "${arch[@]}" -c "$root"/mllee/asm-cortex-m4/mllee_int8_mac_cortex_m4.S \
    -o "$out/mac.o"
"$cc" "${arch[@]}" -c "$root"/mllee/asm-cortex-m4/mllee_int8_reduce_cortex_m4.S \
    -o "$out/reduce.o"
"$cc" "${arch[@]}" -c "$root"/mllee/asm-cortex-m4/mllee_int4_expand_cortex_m4.S \
    -o "$out/expand.o"
"$cc" "${arch[@]}" -c "$root"/mllee/asm-cortex-m4/mllee_train_q15_lanes_cortex_m4.S \
    -o "$out/train_lanes.o"
"$cc" "${arch[@]}" -c "$root"/mllee/asm-cortex-m4/mllee_train_q15_mac_cortex_m4.S \
    -o "$out/train_mac.o"
"$cxx" "${flags[@]}" -ffreestanding -c "$root/tools/mllee_m4_selftest.cpp" \
    -o "$out/selftest.o"
"$cc" "${arch[@]}" -c "$root/tools/mllee_m4_startup.S" -o "$out/startup.o"

"$cxx" "${arch[@]}" -nostdlib -nostartfiles -Wl,--gc-sections -Wl,-z,noexecstack \
    -T "$root/tools/mllee_m4.ld" \
    "$out/startup.o" "$out/selftest.o" "$out/mac.o" "$out/reduce.o" "$out/expand.o" \
    "$out/train_lanes.o" "$out/train_mac.o" \
    -o "$out/selftest.elf"

output=$(timeout 120 "$qemu" -machine mps2-an385 -nographic -semihosting \
    -kernel "$out/selftest.elf" 2>&1 | tr -d '\r') || true
echo "$output"
# QEMU's exit status is not informative for a semihosted ARMv7-M SYS_EXIT, so
# the guest's own verdict line decides.
if grep -q '^M4-SELFTEST: PASS ' <<<"$output" && ! grep -qE 'HARDFAULT|^FAIL ' <<<"$output"; then
    echo "mllee M4 self-test: PASS"
else
    echo "mllee M4 self-test: FAIL" >&2
    exit 1
fi
