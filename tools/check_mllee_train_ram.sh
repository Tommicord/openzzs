#!/usr/bin/env bash
# Copyright(c) 2026 Tommicord
# SPDX-License-Identifier: MIT
#
# mllee training RAM / Flash gate.
#
# Training is a separate, larger budget than inference on purpose: it runs
# offline, and an on-device trainer must still be describable by closed-form
# arithmetic.  The three pieces add up to the declared budget:
#
#   pool       ZZS_MLLEE_TRAIN_POOL_BYTES (default 8192) scratch arena
#   control    the resident bookkeeping objects (Tape + Pool + Trainer + the
#              step/fit stat structs), capped at 512 B by the size probe
#   stack      the deepest call chain a step can build, capped at
#              ZZS_MLLEE_TRAIN_STACK_LIMIT (default 512) for module frames, plus
#              the caller's own hook frames up to
#              ZZS_MLLEE_TRAIN_FULL_STACK_LIMIT (default 1024)
#
# 8192 + 512 + 1024 = 9728 <= the 10240-byte training RAM budget, and the module
# owns no heap and no writable statics exactly as the inference path does -- the
# .o is checked for an empty .data/.bss.
#
# Usage: tools/check_mllee_train_ram.sh [output-dir]
#        ZZS_MLLEE_CXX=clang++ tools/check_mllee_train_ram.sh
#        ZZS_MLLEE_TRAIN_STACK_LIMIT=640 tools/check_mllee_train_ram.sh
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
out=${1:-"$root/build-mllee-train-ram"}
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

echo "mllee training RAM gate: $cxx ${target[*]}"

# ---------------------------------------------------------------------------
# 1. The closed-form budget arithmetic.
# ---------------------------------------------------------------------------
if (( is_arm == 0 )); then
    "$cxx" "${target[@]}" "${flags[@]}" "$root/tools/mllee_train_size_check.cpp" \
        -o "$out/mllee_train_size_check"
    report=$("$out/mllee_train_size_check") || { echo "$report" >&2; exit 1; }
else
    # The 32-bit layout is fixed by static_assert in the probe; run it natively
    # anyway so the printed ceilings are the ones a host reader can reproduce.
    "${CXX:-g++}" -std=c++23 -O1 -Wall -Wextra -Werror -DZZS_MLLEE_HOST \
        -I"$root" "$root/tools/mllee_train_size_check.cpp" \
        -o "$out/mllee_train_size_check"
    report=$("$out/mllee_train_size_check") || { echo "$report" >&2; exit 1; }
fi
echo "$report"

value () { sed -n "s/^$1=\([0-9][0-9]*\)$/\1/p" <<<"$report" | head -n1; }

pool_bytes=$(value pool_bytes)
control_bytes=$(value control_bytes)
node_bytes=$(value node_bytes)
trainer_bytes=$(value trainer_bytes)
max_nodes=$(value max_nodes_default)
pool_bytes=${pool_bytes:-0}
control_bytes=${control_bytes:-0}

pool_budget=${ZZS_MLLEE_TRAIN_POOL_BYTES:-8192}
control_limit=${ZZS_MLLEE_TRAIN_CONTROL_LIMIT:-512}

status=0

printf 'pool arena       : %d B (budget %d)\n' "$pool_bytes" "$pool_budget"
if (( pool_bytes > pool_budget )); then
    echo "FAIL: training pool is $pool_bytes bytes, budget $pool_budget" >&2
    status=1
fi

printf 'control plane    : %d B (limit %d)\n' "$control_bytes" "$control_limit"
if (( control_bytes > control_limit )); then
    echo "FAIL: training control plane is $control_bytes bytes, limit $control_limit" >&2
    status=1
fi

printf 'tape node        : %d B x %d nodes fit beside 16 params\n' \
    "$node_bytes" "$max_nodes"

# ---------------------------------------------------------------------------
# 2. The deepest step chain, from -fstack-usage.
# ---------------------------------------------------------------------------
"$cxx" "${target[@]}" "${flags[@]}" -c "$root/tools/mllee_train_stack_check.cpp" \
    -o "$out/mllee_train_stack_check.o"

su="$out/mllee_train_stack_check.su"
if [[ ! -f $su ]]; then
    echo "FAIL: $su was not produced" >&2
    exit 1
fi

frame () {
    local value
    value=$(awk -v needle="$1" -F'\t' 'index($1, needle) { print $2; exit }' "$su")
    echo "${value:-0}"
}

step=$(frame "Trainer::step(")
configure=$(frame "Trainer::configure(")
linear_run=$(frame "linear_run(")
linear_forward=$(frame "linear_forward(")
linear_backward=$(frame "linear_backward(")
mse_backward=$(frame "mse_backward(")
act_backward=$(frame "act_backward_codes(")
tape_record=$(frame "Tape::record(")
tape_save=$(frame "Tape::save(")
pool_carve=$(frame "Pool::carve(")
hook=$(frame "probe_forward(")
entry=$(frame "mllee_train_worst_step(")

# A step is forward-then-backward, and the frames nest: Trainer::step calls the
# caller's hook, which calls linear_run, which calls *one* of {forward kernel,
# tape save, tape record} at a time -- siblings share the frame budget, so only
# the deepest child is added.  Saving is outlined per slab and record is called
# once per node, so they are checked as children of linear_run too.
forward_child=$linear_forward
for child in "$tape_record" "$tape_save" "$pool_carve"; do
    (( child > forward_child )) && forward_child=$child
done
forward=$((step + linear_run + forward_child))

backward_child=$((linear_backward + act_backward))
(( mse_backward > backward_child )) && backward_child=$mse_backward
backward=$((step + backward_child))

module_worst=$forward
module_name="forward (linear kernel)"
if (( backward > module_worst )); then
    module_worst=$backward
    module_name="backward (linear+activation)"
fi
# The hook is caller code, but it is on the device stack while the module runs,
# so the probe reports the whole chain too.
full=$((entry + hook + module_worst))

stack_limit=${ZZS_MLLEE_TRAIN_STACK_LIMIT:-512}
full_limit=${ZZS_MLLEE_TRAIN_FULL_STACK_LIMIT:-1024}

printf 'forward chain    : %d B (step %d + linear_run %d + deepest child %d)\n' \
    "$forward" "$step" "$linear_run" "$forward_child"
printf 'backward chain   : %d B (step %d + deepest node %d)\n' \
    "$backward" "$step" "$backward_child"
printf 'module stack     : %d B (worst chain: %s, limit %d)\n' \
    "$module_worst" "$module_name" "$stack_limit"
if (( module_worst > stack_limit )); then
    echo "FAIL: training module stack chain exceeds $stack_limit bytes" >&2
    status=1
fi

printf 'probe stack      : %d B (caller hook included, limit %d)\n' \
    "$full" "$full_limit"
if (( full > full_limit )); then
    echo "FAIL: the measured probe chain exceeds $full_limit bytes" >&2
    status=1
fi

# ---------------------------------------------------------------------------
# 3. Statics, flash, and the total.
# ---------------------------------------------------------------------------
sections=$("$size_tool" -A "$out/mllee_train_stack_check.o")
data=$(awk '$1 == ".data" { sum += $2 } END { print sum + 0 }' <<<"$sections")
bss=$(awk '$1 == ".bss" { sum += $2 } END { print sum + 0 }' <<<"$sections")
text=$(awk '$1 ~ /^\.text(\.|$)/ { sum += $2 } END { print sum + 0 }' <<<"$sections")
rodata=$(awk '$1 ~ /^\.rodata(\.|$)/ { sum += $2 } END { print sum + 0 }' <<<"$sections")
flash=$((text + rodata))
flash_limit=${ZZS_MLLEE_TRAIN_FLASH_LIMIT:-8192}

printf 'writable statics : .data=%s .bss=%s (must be 0)\n' "$data" "$bss"
if (( data != 0 || bss != 0 )); then
    echo "FAIL: mllee must not own writable static storage" >&2
    status=1
fi

printf 'flash (one op)   : %d B text + %d B rodata = %d B (limit %d)\n' \
    "$text" "$rodata" "$flash" "$flash_limit"
if (( flash > flash_limit )); then
    echo "FAIL: a linear+MSE training step exceeds the $flash_limit-byte flash budget" >&2
    status=1
fi

ram_budget=${ZZS_MLLEE_TRAIN_RAM_BUDGET:-10240}
total=$((pool_bytes + control_bytes + full))
printf 'training RAM     : %d B (pool + control + deepest chain, budget %d)\n' \
    "$total" "$ram_budget"
if (( total > ram_budget )); then
    echo "FAIL: training exceeds the $ram_budget-byte RAM budget" >&2
    status=1
fi

printf 'trainer object   : %d B\n' "$trainer_bytes"

if (( status != 0 )); then
    exit 1
fi
echo "mllee training RAM gate: PASS"
