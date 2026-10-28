#!/usr/bin/env bash
# Copyright(c) 2026 Tommicord
# SPDX-License-Identifier: MIT
#
# mllee RAM / Flash gate.
#
# The module owns ZZS_MLLEE_RAM_BUDGET bytes (default 1024) of writable RAM on a
# 32-bit target (Cortex-M4 layout).  That budget is the resident Core object
# (640 bytes, including the 444-byte activation arena) plus the deepest call
# chain the module can build (ceiling 384 bytes, ZZS_MLLEE_STACK_LIMIT).  The
# module owns no heap, no writable statics and no global mutable state.
#
# Usage: tools/check_mllee_ram.sh [output-dir]
#        ZZS_MLLEE_CXX=clang++ tools/check_mllee_ram.sh
#        ZZS_MLLEE_STACK_LIMIT=288 tools/check_mllee_ram.sh
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:-"$root/build-mllee-ram"}
mkdir -p "$out"

cxx=${ZZS_MLLEE_CXX:-}
if [[ -z $cxx ]]; then
    if command -v arm-none-eabi-g++ >/dev/null 2>&1; then
        cxx=arm-none-eabi-g++
    else
        cxx=${CXX:-g++}
    fi
fi

is_arm=0
target=()
size_tool=size
if [[ $cxx == *arm-none-eabi* ]]; then
    is_arm=1
    target=(-mcpu=cortex-m4 -mthumb)
    size_tool=arm-none-eabi-size
fi

flags=(-std=c++23 -Os -fomit-frame-pointer -fstack-usage -fno-exceptions -fno-rtti
    -fno-stack-protector -ffunction-sections -fdata-sections -I"$root")

echo "mllee RAM gate: $cxx ${target[*]}"

"$cxx" "${target[@]}" "${flags[@]}" -c "$root/tools/mllee_size_check.cpp" \
    -o "$out/mllee_size_check.o"

if (( is_arm == 0 )); then
    "$cxx" "${target[@]}" "${flags[@]}" "$root/tools/mllee_size_check.cpp" \
        -o "$out/mllee_size_check"
    sizes=$("$out/mllee_size_check") || { echo "$sizes" >&2; exit 1; }
    echo "$sizes"
    core_bytes=$(sed -n 's/.*sizeof(Core)=\([0-9][0-9]*\).*/\1/p' <<<"$sizes" | head -n1)
    core_bytes=${core_bytes:-0}
else
    # The 32-bit layout is enforced by static_assert in mllee_size_check.cpp.
    echo "layout: 32-bit assertions passed (Core == 640, Workspace == 192, PacklowState == 124, onnx::Loader == 116)"
    core_bytes=640
fi

"$cxx" "${target[@]}" "${flags[@]}" -c "$root/tools/mllee_stack_check.cpp" \
    -o "$out/mllee_stack_check.o"

su="$out/mllee_stack_check.su"
if [[ ! -f $su ]]; then
    echo "FAIL: $su was not produced" >&2
    exit 1
fi

frame () {
    local value
    value=$(awk -v needle="$1" -F'\t' 'index($1, needle) { print $2; exit }' "$su")
    echo "${value:-0}"
}

run=$(frame "Core::run(")
execute=$(frame "_S_execute(")
conv=$(frame "_S_conv1d(")
linear=$(frame "_S_linear(")
attention=$(frame "_S_attention(")
norm=$(frame "_S_norm(")
softmax=$(frame "_S_softmax(")
activation=$(frame "_S_activation(")
parse=$(frame "parse_layer(")
pk_next=$(frame "_S_pk_next(")
rd_code=$(frame "_S_rd_code(")
rd_byte=$(frame "_S_rd_byte(")
rd_dot_i8=$(frame "_S_rd_row_dot_i8(")
rd_dot_packed=$(frame "_S_rd_row_dot_packed(")
rd_packed_dot=$(frame "_S_packed_dot(")

# The codec reader and the vector dot are sibling children of a kernel: a row is
# either decoded a code at a time or expanded in SIMD groups, never both at
# once.  The vector child is format-dependent, so conv1d (I8 only) and linear
# (I8 or packed) take their own maximum.
codec_child=$((rd_code + rd_byte + pk_next))
packed_child=$((rd_dot_packed + rd_packed_dot))
conv_child=$rd_dot_i8
(( codec_child > conv_child )) && conv_child=$codec_child
linear_child=$rd_dot_i8
(( packed_child > linear_child )) && linear_child=$packed_child
(( codec_child > linear_child )) && linear_child=$codec_child

weight=$((run + execute + conv + conv_child))
linear_chain=$((run + execute + linear + linear_child))
attention_chain=$((run + execute + attention))
norm_chain=$((run + execute + norm))
softmax_chain=$((run + execute + softmax))
activation_chain=$((run + execute + activation))
parse_chain=$((run + parse))
frame_begin=$(frame "_S_pk_begin(")
begin_chain=$((run + frame_begin))

worst=$weight
worst_name="conv1d+codec"
for entry in "linear+codec:$linear_chain" "attention:$attention_chain" "norm:$norm_chain" \
    "softmax:$softmax_chain" "activation:$activation_chain" "parse_layer:$parse_chain" \
    "pk_begin:$begin_chain"; do
    name=${entry%%:*}
    value=${entry##*:}
    if (( value > worst )); then
        worst=$value
        worst_name=$name
    fi
done

sections=$("$size_tool" -A "$out/mllee_stack_check.o")
data=$(awk '$1 == ".data" { sum += $2 } END { print sum + 0 }' <<<"$sections")
bss=$(awk '$1 == ".bss" { sum += $2 } END { print sum + 0 }' <<<"$sections")
flash=$(awk '$1 ~ /^\.text(\.|$)/ || $1 ~ /^\.rodata(\.|$)/ { sum += $2 } END { print sum + 0 }' \
    <<<"$sections")
text=$(awk '$1 ~ /^\.text(\.|$)/ { sum += $2 } END { print sum + 0 }' <<<"$sections")
rodata=$(awk '$1 ~ /^\.rodata(\.|$)/ { sum += $2 } END { print sum + 0 }' <<<"$sections")

stack_limit=${ZZS_MLLEE_STACK_LIMIT:-384}
ram_budget=${ZZS_MLLEE_RAM_BUDGET:-1024}
flash_limit=16384

status=0

printf 'writable statics : .data=%s .bss=%s (must be 0)\n' "$data" "$bss"
if (( data != 0 || bss != 0 )); then
    echo "FAIL: mllee must not own writable static storage" >&2
    status=1
fi

printf 'core stack       : %d B (worst chain: %s, limit %d)\n' "$worst" "$worst_name" "$stack_limit"
if (( worst > stack_limit )); then
    echo "FAIL: core stack chain exceeds $stack_limit bytes" >&2
    status=1
fi

printf 'flash (code+lut) : %d B text + %d B rodata = %d B (limit %d)\n' \
    "$text" "$rodata" "$flash" "$flash_limit"
if (( flash > flash_limit )); then
    echo "FAIL: mllee code and tables exceed the $flash_limit-byte flash budget" >&2
    status=1
fi

printf 'static RAM       : %d B (Core object)\n' "$core_bytes"
printf 'module RAM       : %d B (Core + deepest chain, budget %d)\n' \
    "$((core_bytes + worst))" "$ram_budget"
if (( core_bytes + worst > ram_budget )); then
    echo "FAIL: mllee exceeds the $ram_budget-byte RAM budget" >&2
    status=1
fi

if (( status != 0 )); then
    exit 1
fi
echo "mllee RAM gate: PASS"
