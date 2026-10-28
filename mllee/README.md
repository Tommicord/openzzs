# mllee — a heap-free int8 inference core with an on-device training path

`mllee` is a header-only C++23 core for the OpenZZS sensor pipeline
(Conv1D → attention → dense). It uses only integer arithmetic, streams weights
from Flash, and keeps every byte of writable state inside one `Core` object.
`mllee/train.hpp` adds the reverse path — tape, optimizers, losses and the
PyTorch-style operators around them — so the same blob a device runs is the
blob it can fine-tune.

- No heap, no exceptions, no RTTI, no recursion, no writable module statics.
- Weights are either raw quantized bytes or compressed with the improved
  packlow range coder (bounded 64-byte window).
- int8, int4, float8 (`e4m3fn`, `e5m2`) and float4 (`e2m1`) weights are all
  reduced to int8 × int8 MACs; float formats use a Flash-resident LUT.
- Norm, activations, softmax and single-head attention are built in.

```
OFFLINE                                      RUNTIME (Cortex-M)
PyTorch model + calibration                  Core::run(model, len, input, in_len)
  → quantize (i8 / i4 / f8 / f4)               for each Flash descriptor:
  → pack nibbles + scales/biases                 decode weight stream (64 B ring)
  → packlow encode (host-only)                   MAC int8 × int8 → int32
  → model blob (Flash / .rodata)                 requantize → int8 arena
                                               return pointer into arena
```

## RAM and Flash budget

Measured with `arm-none-eabi-g++ 14.2`, `-mcpu=cortex-m4 -mthumb -Os
-fomit-frame-pointer -fno-exceptions -fno-rtti -fno-stack-protector`, using
`tools/check_mllee_ram.sh` (`-fstack-usage` + section sizes).

The module budget is **1024 bytes of writable RAM** (`_S_ram_budget_bytes`),
split into a static part that the library owns and a dynamic part the library
can put on the caller's stack:

| Writable static region                                        |                            Bytes |
|---------------------------------------------------------------|---------------------------------:|
| `PacklowState` — range/code, probabilities, 64 B ring, reader |                              124 |
| `Header` — parsed model metadata                              |                               20 |
| `Layer` — active 48-byte descriptor                           |                               48 |
| Activation arena                                              |                              444 |
| Status + padding                                              |                                4 |
| **`sizeof(Core)`**                                            | **640** (`_S_core_budget_bytes`) |
| Module `.data` / `.bss`                                       |                        **0 / 0** |
| Heap                                                          |                            **0** |

Flash (not counted against the 1024 B):

| Flash item                                                         |                      Bytes |
|--------------------------------------------------------------------|---------------------------:|
| Core + ops + codec + asm `.text`                                   |                     10 574 |
| Activation / exp / CRC LUTs (`.rodata`)                            |                      1 536 |
| **Total code + tables**                                            | **12 110** (budget 16 KiB) |
| Layer descriptors, scales, biases, model LUTs, dictionary, weights |            model-dependent |

### How the 1024 bytes are split

`Core` is the only writable static RAM the module owns. The deepest call chain
it can build is charged against the same budget because it is the only other
RAM the module touches while running:

| Region | Bytes |
|---|---:|
| Core object (static) | 640 |
| Worst core stack chain (`run → _S_execute → _S_conv1d → _S_pk_next`) | 304 |
| **Total RAM at run time** | **944** of 1024 |

The chain is gated on its own by `ZZS_MLLEE_STACK_LIMIT` (default 384) and the
sum by `ZZS_MLLEE_RAM_BUDGET` (default 1024); per-frame numbers are printed by
the gate script. Frames of interest: `_S_conv1d` 160, `_S_attention` 136,
`_S_batch_norm` / `_S_group_norm` 104, `_S_embedding` 96, `_S_linear` 88,
`_S_norm` 88, `_S_pool` 64, `_S_matmul` 56, `_S_binary` 48, `_S_pk_next` 48.

The arena is derived from the budget (`_S_arena_bytes = _S_core_budget_bytes -
196`, where 196 is the workspace plus status footprint), so lowering
`_S_arena_bytes` in `format.hpp` shrinks `Core` without touching the stack
gate. Growing it past `_S_core_budget_bytes` fails the `static_assert` in
`core.hpp` before anything is compiled.

## Model blob format

All fields are little-endian and parsed with byte loads; no packed structs.
The 32-byte header carries a CRC-16-CCITT over its first 28 bytes.

| Offset | Header field                              |
|-------:|-------------------------------------------|
|      0 | magic `"MLEE"`                            |
|      4 | format version (1)                        |
|      5 | flags                                     |
|      6 | layer count                               |
|      7 | header size (32)                          |
|      8 | layer table offset (u32)                  |
|     12 | scale table offset (u32)                  |
|     16 | dictionary offset (u32)                   |
|     20 | dictionary length (u16)                   |
|     22 | arena bytes used by the model (u16, ≤444) |
|     24 | default dtype                             |
|     25 | op count (5)                              |
|     26 | reserved                                  |
|     28 | header CRC-16                             |
|     30 | reserved                                  |

Each layer descriptor is 48 bytes:

| Offset | Field                                                                |
|-------:|----------------------------------------------------------------------|
|    0-3 | op, activation, dtype, codec                                         |
|    4-7 | `in_dim`, `out_dim` (u16)                                            |
|   8-14 | `kernel`, `stride`, `groups`, `group_size`, `head_dim`, `seq`, flags |
|  16-35 | `weight_off`, `weight_len`, `scale_off`, `bias_off`, `lut_off` (u32) |
|  36-43 | `in_off`, `out_off`, `in_len`, `out_len` (u16)                       |
|     44 | `requant_mult` (i32, Q15)                                            |

`in_off == 0xFFFF` marks an external input (the caller's feature buffer).
Layers write activations into the 444-byte arena at `out_off`.

Scale tables hold `rows × groups` int32 Q15 multipliers (per-channel or
per-group); norm layers use int16 gamma instead. Biases are int16 and are
pre-scaled to the output quantization scale. Float weights point at a LUT:
256 bytes for e4m3/e5m2, 16 bytes for e2m1.

Compressed weights begin with a 12-byte stream header:

```
"MLPK" | version u8 | flags u8 | raw_bytes u32 | crc16 u16 | payload
```

`raw_bytes` terminates decoding; the CRC-16 is updated over decoded bytes.
The packer stores a tensor raw when compression does not shrink it.

## Quantization

PyTorch observers are the numerical reference; the core only consumes integers.

Affine activation quantizer:

```
scale = (max - min) / (qmax - qmin)
zero_point = clamp(qmin - round(min / scale), qmin, qmax)
q = clamp(round(x / scale) + zero_point, qmin, qmax)
x_hat = (q - zero_point) * scale
```

Symmetric weights use `zero_point = 0` and `scale = max(|x|) / qmax`.

| Dtype  | Layout                                          | Scale                    | Range          |
|--------|-------------------------------------------------|--------------------------|----------------|
| int8   | two's complement, 1/byte                        | per-channel / per-tensor | [-127, 127]    |
| int4   | two's complement, 2/byte, **low nibble first**  | per-group (32)           | [-7, 7]        |
| e4m3fn | 1-4-3, bias 7, max 448, no inf, `0x7F/0xFF` NaN | per-channel              | via LUT → int8 |
| e5m2   | 1-5-2, bias 15, max 57344                       | per-channel              | via LUT → int8 |
| e2m1   | {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6}             | shared E8M0 per 32       | via LUT → int8 |

Requantization uses one Q15 multiplier folding `s_weight · s_input / s_output`:

```
y = clamp(round_half_up(acc · M / 2^15) + bias, -127, 127)
```

Per-group scale application happens per 32-weight group, keeping the
accumulator in int32.

## Improved packlow codec

The decoder is a C++ port of the range coder from `codevar-packlow` with the
memory model replaced:

| Reference crate                         | mllee port                                            |
|-----------------------------------------|-------------------------------------------------------|
| `Vec<u8>` whole-output decode           | streaming decode, caller-owned output                 |
| 2 KiB block dedup table                 | 64-byte ring (two 32-byte blocks)                     |
| 18-bit probabilities                    | 12-bit adaptive probabilities                         |
| shared token probability                | 2-bit token tree (LITERAL / COPY / DELTA / DICT_COPY) |
| 64-bit whole-output hash                | streaming CRC-16-CCITT                                |
| dead delta path (`min_similarity=0xCC`) | explicit ratio, active and regression-tested          |
| decoder-side MTF table                  | Flash-resident 32-byte-block dictionary               |
| heap allocations per delta              | none; all state in `PacklowState`                     |

Tokens copy from the 64-byte ring (overlap allowed), from the Flash dictionary,
or apply 5-bit corrections to a ring byte. Bounds checks reject a distance
outside `1..64`, a dictionary block past the end, a length larger than the
remaining output, a truncated stream, or a CRC mismatch.

Measured with `tools/mllee_bench.cpp` on a desktop host (4096-byte tensors,
200 iterations; MB/s vary with optimization flags and are host numbers, not
Cortex-M cycles):

| pattern       | packed B | ratio | encode MB/s | decode MB/s |
|---------------|---------:|------:|------------:|------------:|
| random        |    4 196 | 1.024 |          ~8 |         ~14 |
| constant      |       21 | 0.005 |        ~110 |        ~130 |
| ramp          |    3 908 | 0.954 |         ~10 |         ~16 |
| block weights |    2 097 | 0.512 |         ~25 |         ~50 |

The blob writer compares the packed and raw sizes and stores the smaller one,
so a tensor never expands. The reference crate measured 6.7–10.4 MB/s unpack
on the same kind of data; the streaming decoder avoids whole-output buffers
and per-delta allocations.

## Ops and envelope

| Op           | Behaviour                                                   |
|--------------|-------------------------------------------------------------|
| `LINEAR`     | per-output weight stream, group scales, bias, activation    |
| `CONV1D`     | causal or valid, stride, groups; one row staged in the ring |
| `NORM`       | LayerNorm / RMSNorm with int16 gamma and bias               |
| `ACTIVATION` | ReLU, ReLU6, sigmoid, tanh, GELU via 512-byte LUTs          |
| `SOFTMAX`    | two-pass integer exp table over the row                     |
| `ATTENTION`  | single-head, online normalisation, no score buffer          |

Arena limits with the default 444-byte arena:

- One activation row (linear / norm / softmax / activation) ≤ 444 values.
- Attention consumes `4 · seq · head_dim` bytes (Q, K, V and output), so
  `seq · head_dim ≤ 47`; `head_dim` itself is limited to 8.
- A Conv1D weight row must fit the 64-byte ring: `in_channels · kernel ≤ 64`
  int8 codes, or ≤ 128 int4/e2m1 codes (packed two per byte).

These are validation failures (`ARENA_OVERFLOW` / `UNSUPPORTED`), never
out-of-bounds accesses.

## Training path

`mllee/train.hpp` is the aggregate include for the 31 training headers. Every
one of them keeps the inference invariants: no heap, no writable module
statics, no exceptions, no recursion, errors returned as `Status`, all
numerics in fixed point (int8 codes on the `[-127, 127]` grid, Q15 `int16_t`
lanes, int32/int64 intermediates), and every buffer carved from a
caller-owned pool.

| Layer | Headers |
|---|---|
| Vocabulary | `train_fixed` (Q15 mul/div/sqrt, rounding, rails), `train_tensor` (views, shapes, flags), `train_pool` (8 KiB carve arena), `train_random` (deterministic engines, masks, index permutations) |
| Autograd | `train_graph` (reverse-mode tape over caller storage), `train_backward_kernels` (matmul / reduction / elementwise backward), `train_grad_check` (torch.autograd.gradcheck without a heap) |
| Layers | `train_linear`, `train_conv` (1-D/2-D, stride, padding, dilation, groups), `train_pooling` (max, avg, adaptive-avg, L2, global), `train_embedding` (tied views, embedding bags), `train_running_stats` (BN/GN buffers) |
| Numerics | `train_softmax` (masked, temperature, log path), `train_activations`, `train_dropout` (inverted, saved mask), `train_fake_quant` (QAT and the STE), `train_init` (code-space initializers that report their scale) |
| Objective | `train_loss` (MSE, MAE, Huber, quantile, BCE-with-logits, cross-entropy, soft CE, NLL, cosine, hinge, multiclass hinge), `train_label_smoothing`, `train_regularization` (installable post-grad hooks), `train_distill` (temperature, KL, alpha blend), `train_loss_scale` (static and dynamic) |
| Optimizer | `train_optim` (SGD + momentum/dampening/nesterov/weight-decay, Adam, AdamW, Adagrad, RMSprop, Adadelta, Lion), `train_lr_schedule` (constexpr per-step arithmetic), `train_grad_clip` (value, global-norm, per-group, L1), `train_model_averaging` (uniform, EMA, SWA, gated) |
| Runtime | `train_trainer` (the step loop: schedule, forward, backward, skip-or-unscale, clip-and-penalize, update, account), `train_calibration` (float-free observers and grid selection), `train_prune` (magnitude masks, blocks, schedules), `train_checkpoint` (resumable run state, no filesystem), `train_simd` (the ten lane primitives) |

The numerical contract is the load-bearing part of the design: each `train_simd`
primitive has a portable C++ fallback documented as byte-identical to the
assembly, so a build without assembly produces the same codes, and a mismatch
between the two is a defect rather than a tolerance.

### Training RAM budget

Training is deliberately a larger, separate budget from the 1 KiB inference
module (`tools/check_mllee_train_ram.sh`):

| Region | Bytes | Limit |
|---|---:|---:|
| Pool arena (tapes, states, activations, scratch) | 8 192 | 8 192 |
| Control plane (trainer object, optimizer params, hooks) | 412 | 512 |
| Deepest forward chain (step + `linear_run` + child) | 424 | — |
| Deepest backward chain (step + tape node) | 264 | — |
| Module stack (worst chain, siblings share frames) | 424 | 512 |
| Probe stack (caller hook included) | 912 | 1 024 |
| Flash, one training op | 6 286 text + 1 282 rodata = 7 568 | 8 192 |
| Writable statics | `.data=0 .bss=0` | must be 0 |
| **Training RAM at run time** | **9 516** | **10 240** |

The closed form is `pool + control limit + full-stack limit = 8192 + 512 +
1024 = 9728 ≤ 10240`; the gate reports the measured 9 516 against it, which is
the 8192-byte pool plus the 412-byte control plane plus the 912-byte probe
stack. A tape node is 264 B, so 28 nodes fit beside 16 parameters in the
default pool.
`ZZS_MLLEE_TRAIN_POOL_BYTES`, `ZZS_MLLEE_TRAIN_CONTROL_LIMIT`,
`ZZS_MLLEE_TRAIN_STACK_LIMIT` and `ZZS_MLLEE_TRAIN_FULL_STACK_LIMIT` retarget
those limits for a smaller part.

## Assembly kernels

`mllee/asm-<target>/` holds the hand-written kernels; CMake compiles them into
a static `openzzs-mllee-asm` library and defines the macro the headers dispatch
on. The portable C++ path stays compiled-in whenever the option is off.

| Target | Sources | Macro |
|---|---|---|
| x86-64 baseline (SSE2) | `mllee_int8_dot_sse2.S`, `mllee_int8_reduce_sse2.S`, `mllee_int8_stats_sse2.S`, `mllee_train_q15_sse2.S` | `ZZS_MLLEE_ASM_X86_64` |
| x86-64 SSSE3 (opt-in, default on) | `mllee_packlow_shuffle_ssse3.S` (`pshufb` nibble expand, no sequential bit extraction) | `ZZS_MLLEE_ASM_SSSE3` |
| x86-64 AVX2 (opt-in, default off) | `mllee_int8_dot_avx2.S` | `ZZS_MLLEE_ASM_AVX2` |
| Cortex-M4 (ARMv7E-M DSP) | `mllee_int8_mac_cortex_m4.S` (SMLAD), `mllee_int8_reduce_cortex_m4.S`, `mllee_int4_expand_cortex_m4.S` (SXTB16/USADA8), `mllee_train_q15_lanes_cortex_m4.S`, `mllee_train_q15_mac_cortex_m4.S` | `ZZS_MLLEE_ASM_CORTEX_M4` |

```sh
cmake -S . -B build -DOPENZZS_MLLEE_ASM=OFF                        # portable only
cmake -S . -B build -DOPENZZS_MLLEE_ASM_TARGET=cortex-m4           # force a target
cmake -S . -B build -DOPENZZS_MLLEE_AVX2=ON -DOPENZZS_MLLEE_SSSE3=OFF
```

`OPENZZS_MLLEE_ASM_TARGET=auto` picks from `CMAKE_SYSTEM_PROCESSOR` and the
compiler name. Two rules the kernels follow and the headers re-document:
a `pmuludq`-style 32×32→64 multiply caps a Granlund–Montgomery reciprocal
exponent at 40 (an `s = 41` magic for the 258 code step is a 34-bit value and
silently becomes its own low half, halving every code step), and flag-testing
branches must not be preceded by an instruction that clobbers the flags they
read.

CRC-16/CCITT has no kernel on purpose: the fold needs a carry-less multiply
ladder whose constants are easy to get wrong, the blob it guards is a few
hundred bytes at load time, and no Cortex-M4 has PCLMULQDQ. The table form in
`mllee/status.hpp` is branch-free and identical on every target.

### Executing the Thumb kernels

The host suite can only ever see the x86 kernels, so
`tools/run_mllee_m4_qemu.sh` builds the Cortex-M4 sources for QEMU's
`mps2-an385` (a real ARMv7E-M model) and checks them against the documented
semantics there — an assembler pass alone does not prove that SMLAD/SXTB16
encodings are legal on Cortex-M or that the lane tricks produce the same bytes
as the C++ path.

```sh
tools/run_mllee_m4_qemu.sh            # prints CAPS, then M4-SELFTEST: PASS
```

The reference side is compiled for baseline ARMv7-M on purpose: GCC will
happily turn a dot product into SMLABB (which this model faults on), and then
the harness would be checking the compiler against itself.

## API

```cpp
#include "mllee/mllee.hpp"

zzs::mllee::Core core;                      // 640 B, zero-initialised
auto result = core.run (model, model_len, input, input_len);
if (!result.has_value ())
    return result.error ();                 // Error::Kind

const int8_t *output = result.value ()._M_data;  // points into the arena
const uint16_t output_len = result.value ()._M_len;
```

The returned pointer is valid until the next `run` on the same `Core`.
The model blob must stay alive (it is normally a `constexpr` Flash array).
`core.status ()` reports the last error as a `Status`.

Errors use `zzs::base::Maybe`; nothing throws, and the hot path stores status
in a single byte.

### Training

```cpp
#include "mllee/train.hpp"

namespace tr = zzs::mllee::train;

alignas (tr::_S_pool_alignment) uint8_t backing[tr::_S_train_pool_bytes];
tr::Pool pool (backing, sizeof (backing));

/* one parameter tensor: int8 codes plus Q15 gradient lanes, carved from the
   caller's pool, so the module owns no heap and no statics */
int8_t *codes = nullptr;
int16_t *grad = nullptr;
if (!status_ok (pool.carve_typed (64u, codes))            // ARENA_OVERFLOW or
    || !status_ok (pool.carve_typed (64u, grad)))         // OUT_OF_RANGE: the
    return zzs::mllee::Status::ARENA_OVERFLOW;            // pool is untouched

const tr::Trainable param = tr::Trainable::make (
    codes, grad, tr::Shape::vector (64u),
    static_cast<uint32_t> (tr::Flag::RequiresGrad));

tr::OptimState state;                                     // carved beside the pool
if (!status_ok (tr::optim_state_init (pool, tr::OptimKind::SGD, param, state)))
    return zzs::mllee::Status::BAD_LAYOUT;

tr::Optimizer opt;                                        // SGD, lr 3277 (~0.1)
return opt.step (param, state);                           // codes move on the 258 grid
```

Every step returns a `Status` and refuses before writing when the state was
never carved, when a gradient pointer is missing, or when the pool cannot hold
the requested buffers — a refused step leaves the parameters byte-identical.
`tr::Tape` records the forward pass into pool slabs and `tr::Trainer` sequences
schedule, forward, backward, unscale, clip, update and accounting; the whole
loop runs inside the 8 KiB pool and the stack budget the training gate prints.

## Build and verification

```sh
cmake -S . -B build-mllee -DCMAKE_BUILD_TYPE=Release -DOPENZZS_BUILD_TESTS=ON
cmake --build build-mllee --target test_mllee
ctest --test-dir build-mllee -R mllee --output-on-failure

cmake --build build-mllee --target check_mllee_ram        # inference budget
cmake --build build-mllee --target check_mllee_train_ram  # training budget
cmake --build build-mllee --target check_mllee_m4_asm     # QEMU kernel self-test
cmake --build build-mllee --target mllee_bench && ./build-mllee/mllee_bench
```

The gates and the QEMU run are also plain scripts (`tools/check_mllee_ram.sh`,
`tools/check_mllee_train_ram.sh`, `tools/run_mllee_m4_qemu.sh`) and take an
output directory, so they run in CI without the test tree configured.

The test suite covers quantization (bit patterns, nibble order, LUTs),
codec round-trips, every truncation point, CRC and payload corruption, delta
and dictionary paths, every op, the arena/error paths, and an end-to-end
Conv1D → attention → dense → softmax pipeline. It also passes under
AddressSanitizer and UndefinedBehaviorSanitizer.

The training suite (`tests/test_mllee_training_*.cpp`, auto-globbed) pins the
fixed-point vocabulary against hand-computed integers, the pool's alignment and
overflow accounting, the tape order, each optimizer's recurrence, bias
correction and eps rails, softmax's strided and masked paths, and every
`train_simd` primitive against the documented contract it and the SSE2 kernel
share.

The RAM gate prints:

```
layout: 32-bit assertions passed (Core == 640, Workspace == 192, PacklowState == 124)
writable statics : .data=0 .bss=0 (must be 0)
core stack       : 304 B (worst chain: conv1d+codec, limit 384)
flash (code+lut) : 10574 B text + 1536 B rodata = 12110 B (limit 16384)
static RAM       : 640 B (Core object)
module RAM       : 944 B (Core + deepest chain, budget 1024)
mllee RAM gate: PASS
```

and the training gate:

```
pool arena       : 8192 B (budget 8192)
control plane    : 412 B (limit 512)
tape node        : 264 B x 28 nodes fit beside 16 params
forward chain    : 424 B (step 80 + linear_run 264 + deepest child 80)
backward chain   : 264 B (step 80 + deepest node 184)
module stack     : 424 B (worst chain: forward (linear kernel), limit 512)
probe stack      : 912 B (caller hook included, limit 1024)
writable statics : .data=0 .bss=0 (must be 0)
flash (one op)   : 6286 B text + 1282 B rodata = 7568 B (limit 8192)
training RAM     : 9516 B (pool + control + deepest chain, budget 10240)
trainer object   : 80 B
mllee training RAM gate: PASS
```

`tools/mllee_size_check.cpp` prints the host sizes and enforces the exact
32-bit layout; `tools/mllee_stack_check.cpp` is the `-fstack-usage` probe whose
`Core` is caller-owned so the object is not charged to the stack.
`tools/mllee_train_size_check.cpp` and `tools/mllee_train_stack_check.cpp` are
their training-side counterparts.
