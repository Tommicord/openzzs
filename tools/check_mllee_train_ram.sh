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

# Read one frame out of a named .su file.
frame_in () {
    local value
    value=$(awk -v needle="$2" -F'\t' 'index($1, needle) { print $2; exit }' "$1")
    echo "${value:-0}"
}

# A frame the compiler inlined away would silently shrink the chain it belongs
# to, so a chain is only measured once every name it adds up is present.
require_frames () {
    local file=$1
    shift
    local needle
    for needle in "$@"; do
        if ! grep -qF -- "$needle" "$file"; then
            echo "FAIL: $file has no frame for '$needle'; the gate cannot bound that chain" >&2
            exit 1
        fi
    done
}

max_of () {
    local best=0 value
    for value in "$@"; do
        (( value > best )) && best=$value
    done
    echo "$best"
}

require_frames "$su" "Trainer::step(" "linear_run(" "linear_forward(" \
    "linear_backward(" "mse_backward(" "act_backward_codes(" "Tape::record(" \
    "Tape::save(" "Pool::carve(" "probe_forward(" "mllee_train_worst_step("

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
# 2b. The convolutional chain: conv -> max-pool -> dropout -> MSE.
#
# The linear chain above is the cheapest model that reaches the trainer, and it
# is not what a device runs.  A conv layer, a max-pool that stages its window
# and a dropout that re-reads its mask are separate, larger frames, so a gate
# that only adds `linear_backward` reports a model that never exists.
# ---------------------------------------------------------------------------
"$cxx" "${target[@]}" "${flags[@]}" -c "$root/tools/mllee_train_deep_stack_check.cpp" \
    -o "$out/mllee_train_deep_stack_check.o"

dsu="$out/mllee_train_deep_stack_check.su"
if [[ ! -f $dsu ]]; then
    echo "FAIL: $dsu was not produced" >&2
    exit 1
fi

require_frames "$dsu" "Trainer::step(" "mllee_train_worst_deep_step(" \
    "deep_forward(" "conv_run(" "conv_record(" "conv_forward(" "conv_backward(" \
    "act_backward_codes(" "pool_run(" "pool_forward(" "pool_backward(" \
    "pool_backward_max(" "pool_backward_window(" \
    "dropout_run(" "dropout_backward(" "mse_backward(" "Tape::record(" \
    "Tape::save(" "Pool::carve("

dstep=$(frame_in "$dsu" "Trainer::step(")
dentry=$(frame_in "$dsu" "mllee_train_worst_deep_step(")
dhook=$(frame_in "$dsu" "deep_forward(")
dconv_run=$(frame_in "$dsu" "conv_run(")
dconv_record=$(frame_in "$dsu" "conv_record(")
dconv_forward=$(frame_in "$dsu" "conv_forward(")
dconv_backward=$(frame_in "$dsu" "conv_backward(")
dact_backward=$(frame_in "$dsu" "act_backward_codes(")
dpool_run=$(frame_in "$dsu" "pool_run(")
dpool_forward=$(frame_in "$dsu" "pool_forward(")
dpool_backward=$(frame_in "$dsu" "pool_backward(")
dpool_backward_max=$(frame_in "$dsu" "pool_backward_max(")
dpool_backward_window=$(frame_in "$dsu" "pool_backward_window(")
ddrop_run=$(frame_in "$dsu" "dropout_run(")
ddrop_backward=$(frame_in "$dsu" "dropout_backward(")
dmse_backward=$(frame_in "$dsu" "mse_backward(")
dtape_record=$(frame_in "$dsu" "Tape::record(")
dtape_save=$(frame_in "$dsu" "Tape::save(")
dpool_carve=$(frame_in "$dsu" "Pool::carve(")

# The tape walks its nodes in the trainer's frame, so every backward is a direct
# child of step; forward ops nest one level deeper inside their own run.
# `pool_backward` is a dispatcher: the max scatter and the window broadcast are
# siblings that never run together, so the chain is the dispatch frame plus the
# deeper of the two, exactly like a run frame plus its forward.
dpool_backward_chain=$(( dpool_backward \
                         + $(max_of "$dpool_backward_max" "$dpool_backward_window") ))
dtape_child=$(max_of "$dtape_record" "$dtape_save" "$dpool_carve")
# `conv_run` writes the node from its own `conv_record` helper, so the tape frame
# nests one level below that helper instead of below the run frame the forward
# shares with it.
dconv_record_chain=$(( dconv_record + dtape_record ))
dconv_chain=$(( dconv_run \
                + $(max_of "$dconv_forward" "$dtape_child" "$dconv_record_chain") ))
dpool_chain=$(( dpool_run + $(max_of "$dpool_forward" "$dtape_child") ))
ddrop_chain=$(( ddrop_run + dtape_child ))
dforward=$(( dstep + $(max_of "$dconv_chain" "$dpool_chain" "$ddrop_chain") ))
dbackward=$(( dstep + $(max_of $(( dconv_backward + dact_backward )) \
                            "$dpool_backward_chain" "$ddrop_backward" "$dmse_backward") ))

deep_module=$(max_of "$dforward" "$dbackward")
deep_name="conv+pool forward"
if (( dbackward > deep_module )); then
    deep_module=$dbackward
    deep_name="pool_backward"
fi

# The caller's hook owns three spec structs and eight arena pointers, so a
# convolutional probe sits above the single-op budget by design; it is bounded
# separately and only the module chain is compared against the 512-byte limit.
deep_full=$(( dentry + dhook + deep_module ))
deep_full_limit=${ZZS_MLLEE_TRAIN_DEEP_FULL_STACK_LIMIT:-2048}

printf 'conv forward     : %d B (step %d + deepest op chain %d)\n' \
    "$dforward" "$dstep" "$(( dforward - dstep ))"
printf 'conv backward    : %d B (step %d + pool %d|%d | conv %d + act %d)\n' \
    "$dbackward" "$dstep" "$dpool_backward" "$dpool_backward_window" \
    "$dconv_backward" "$dact_backward"
printf 'conv module      : %d B (worst chain: %s, limit %d)\n' \
    "$deep_module" "$deep_name" "$stack_limit"
if (( deep_module > stack_limit )); then
    echo "FAIL: a convolutional training step exceeds the $stack_limit-byte module budget" >&2
    status=1
fi
printf 'conv probe       : %d B (caller hook included, limit %d)\n' \
    "$deep_full" "$deep_full_limit"
if (( deep_full > deep_full_limit )); then
    echo "FAIL: the convolutional probe chain exceeds $deep_full_limit bytes" >&2
    status=1
fi

if (( deep_full > full )); then
    worst_full=$deep_full
else
    worst_full=$full
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
total=$((pool_bytes + control_bytes + worst_full))
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
