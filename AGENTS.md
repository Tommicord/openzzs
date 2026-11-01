# AGENTS.md — OpenZZS contributor & AI-agent code conventions

This file is the mandatory guide for anyone writing or changing code in this repo — human contributors and AI agents alike. It is grounded in the actual repository: CMakeLists.txt, `.clang-format`, the source under `std/`, `mllee/`, `kernel/`, `tests/`, `tools/`, and `README.md`. Where this document and the repo disagree, the repo wins; file a follow-up to reconcile.

## 1. Project purpose & stack

OpenZZS ("ML/Power Outage Detection") is an edge ML framework for power-outage detection on embedded devices. It ingests multi-sensor environmental time series (accelerometer, gyroscope, light, sound, battery, magnetometer), runs preprocessing and neural-network inference/training, and classifies power state. The design target is a Cortex-M4-class microcontroller: no heap, no writable statics, no exceptions, fixed-size arenas, and bounded RAM/stack/flash.

Primary stack:

| Layer | Path | CMake target | Notes |
|---|---|---|---|
| Core C++ utilities | `std/` | `openzzs-std` | `Maybe<T, Error>`, logging, formatting, I/O, signal handling |
| ML inference/training engine | `mllee/` | `openzzs-mllee` (INTERFACE) + `openzzs-mllee-asm` (optional) | Compiled `-fno-exceptions -fno-rtti`; build-time SIMD/asm selection, never run-time CPUID |
| Sensor-driver HAL / bus | `kernel/sensor-driver/` | `openzzs-sensor-driver` | Mock bus/HAL in tests; host + embedded platforms |
| Application entry | `bootstrap/main.cpp` | `openzzs` | Currently a stub returning 0 |
| Benchmark | `tools/mllee_bench.cpp` | `mllee_bench` (host only) | Ops hot-path section |
| RAM/flash/stack gates | `tools/check_mllee_ram.sh`, `check_mllee_train_ram.sh`, `run_mllee_m4_qemu.sh` | Custom targets | Enforce budgets; safe no-ops when toolchains are absent |
| Tests | `tests/` | `test_mllee`, `test_sensor_driver` | Catch2 v3 (tag v3.9.1 via find_package/FetchContent); 82 tests pass |

Supported platforms: Linux, macOS, Windows, Zephyr, Cortex-M4 (`arm-none-eabi`, `-mcpu=cortex-m4 -mthumb`). The existing build tree `cmake-build-debug/` is gitignored — never commit it.

## 2. How to build

CMake 3.20+, Ninja, C++23 (`set(CMAKE_CXX_STANDARD 23)` in `CMakeLists.txt`). `README.md` says C++26 — that is a known discrepancy; **CMake is the source of truth: C++23; align README when reconciling.**

### Host Debug build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=clang++
cmake --build build
ctest --test-dir build --output-on-failure
```

### Host Release build

```bash
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-release
ctest --test-dir build-release --output-on-failure
```

Release additionally applies `-Oz -ffunction-sections -fdata-sections -Wl,--gc-sections -Wl,--strip-all` and drops unwind tables for size.

### Toggle assembly kernels

Assembly/SIMD kernels are selected **at build time, never run-time CPUID dispatch**. Portable C++ fallbacks live in `mllee/core_simd.hpp` and `mllee/core_ops.hpp`.

```bash
# Disable all asm (portable C++ only)
cmake -S . -B build-nomasm -G Ninja -DOPENZZS_MLLEE_ASM=OFF

# x86-64 feature toggles (all default ON except AVX-VNNI / AVX-512)
cmake -S . -B build-avx -G Ninja \
  -DOPENZZS_MLLEE_SSSE3=ON -DOPENZZS_MLLEE_AVX2=ON \
  -DOPENZZS_MLLEE_AVX_VNNI=OFF -DOPENZZS_MLLEE_AVX512=OFF

# Cortex-M4 cross build (requires arm-none-eabi toolchain)
cmake -S . -B build-m4 -G Ninja \
  -DCMAKE_SYSTEM_NAME=Generic -DCMAKE_SYSTEM_PROCESSOR=arm \
  -DCMAKE_CXX_COMPILER=arm-none-eabi-g++ \
  -DOPENZZS_MLLEE_ASM_TARGET=cortex-m4
```

When `OPENZZS_MLLEE_ASM_TARGET` is `auto`, x86_64 hosts pick `x86_64`, arm-none-eabi/cortex picks `cortex-m4`, everything else falls back to `none` (portable path).

### Run a single test

```bash
ctest --test-dir build -R test_mllee --output-on-failure
# or run the binary directly with a Catch2 filter:
./build/test_mllee "[ops]"

## 3. Code conventions

### 3.1 C++ standard and where it's declared

`set(CMAKE_CXX_STANDARD 23)` and `set(CMAKE_CXX_STANDARD_REQUIRED ON)` live in the root `CMakeLists.txt` (line 4–5). `openzzs-mllee` declares `cxx_std_23` via `target_compile_features`. **CMake is the source of truth: C++23.** `README.md` says C++26 — that is a known discrepancy (see §7). Do not use C++26-only features.

### 3.2 clang-format (must run before commit)

`.clang-format` exists at the repo root. Run it on every changed file before committing:

```bash
clang-format -i <changed-file>
# verify clean:
clang-format --dry-run --Werror <changed-file>
```

Key settings (from `.clang-format`):

- `BasedOnStyle: LLVM`
- `ColumnLimit: 120`
- `IndentWidth: 4`, `UseTab: Never` (4 spaces, no tabs)
- `NamespaceIndentation: None`
- `BreakBeforeBraces: Attach`
- `PointerAlignment: Right`
- `AlwaysBreakAfterReturnType: All`
- `SortIncludes: true`

### 3.3 Includes

- Sort includes (`.clang-format` has `SortIncludes: true` and `IncludeBlocks: Regroup`).
- Project headers use quotes with the module prefix: `"std/fmt.hpp"`, `"mllee/mllee.hpp"`, `"std/maybe.hpp"`.
- System/library headers use angle brackets: `<cstdint>`, `<vector>`, `<catch2/catch_test_macros.hpp>`.
- Order within a block: project headers first (quoted), then a blank line, then system headers (angled). See `tests/test_mllee_ops.cpp` for the pattern.

Example:

```cpp
#include "mllee/mllee.hpp"
#include "mllee_test_util.hpp"

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>
```

### 3.4 Namespaces

- All code lives under `zzs::<submodule>`: `zzs::mllee`, `zzs::base`, `zzs::kernel`, `zzs::kernel::fx`.
- Declare as `namespace zzs::mllee {` (nested, C++17 style) and close with `} // namespace zzs::mllee`.
- **No `using namespace`** directives.
- `NamespaceIndentation: None` — namespace contents are not indented an extra level (see `mllee/core.hpp` line 33).

### 3.5 Naming conventions (table)

| Category | Prefix / style | Example | Where |
|---|---|---|---|
| Types (classes, structs, enums) | PascalCase | `Core`, `Layer`, `Error::Kind`, `Op` | `mllee/core.hpp`, `mllee/core_status.hpp` |
| Functions / free functions | snake_case | `status_name`, `parse_header`, `run` | `mllee/core_status.hpp`, `mllee/core.hpp` |
| Member variables | `_M_` prefix | `_M_arena`, `_M_status`, `_M_ws`, `_M_len` | `mllee/core.hpp` |
| File-local static / constexpr | `_S_` prefix | `_S_arena_bytes`, `_S_core_budget_bytes`, `_S_crc16_table` | `mllee/core.hpp`, `mllee/core_status.hpp` |
| Local variables | snake_case | `layer`, `status`, `index`, `source`, `destination` | `mllee/core.hpp` |
| Macros | `ZZS_<MODULE>_<NAME>` | `ZZS_MLLEE_NOINLINE`, `ZZS_MLLEE_STACK_LIMIT` | `mllee/core_status.hpp`, `tools/check_mllee_ram.sh` |

Do not invent a `kConst` constant style — the repo uses `_S_` for file-local static constexpr and plain snake_case for locals.

### 3.6 `noexcept` mandate

Every function in `mllee/` and `std/` is `noexcept` unless it returns a fallible result. Examples from `mllee/core.hpp`:

```cpp
Core () noexcept = default;
[[nodiscard]] Status status () const noexcept;
[[nodiscard]] const int8_t *arena () const noexcept;
[[nodiscard]] static constexpr uint32_t ram_bytes () noexcept;
```

Fallible functions still mark the signature `noexcept` and report errors through the return type (`Status`, `Maybe<T, Error>`, `RunResult`), never by throwing:

```cpp
[[nodiscard]] ZZS_MLLEE_NOINLINE RunResult
run (const uint8_t *model, ...) noexcept
```

### 3.7 `[[nodiscard]]` on all fallible return types

Mark `[[nodiscard]]` on `RunResult`, `Status`, `LogResult`, `IoResult`, `Maybe`, and any function returning `Error` or `Status`. The repo's `Error`, `status_ok`, `status_name`, `Core::status`, `Core::arena`, and `Core::ram_bytes` all carry it.

### 3.8 Error handling

- **No exceptions in `mllee/` or `std/`.** The library is compiled `-fno-exceptions -fno-rtti`.
- Errors are values: `Status` enum / `Error` type (`mllee/core_status.hpp`), `Maybe<T, Error>` (`std/maybe.hpp`), `[[nodiscard]] RunResult` (`zzs::mllee::Core::RunResult = zzs::base::Maybe<Output, Error>`).
- Functions return `Status` or `LogResult`/`IoResult`; callers check with `if (status != Status::OK) return this->_M_fail(status);`.
- **Exceptions are allowed in tests only.** Test targets re-enable them via `-fexceptions -frtti -funwind-tables` (see `tests/CMakeLists.txt`).

### 3.9 `static_assert` for size/alignment budgets

Show examples from `mllee/core.hpp` (lines 276–286):

```cpp
static_assert (sizeof (Core)
                   <= (sizeof (void *) == 4 ? _S_core_budget_bytes
                                            : _S_core_budget_host_bytes),
               "mllee Core object exceeds the RAM budget");
static_assert (sizeof (void *) != 4 || sizeof (Core) <= _S_core_budget_bytes,
               "mllee Core exceeds its share of the module RAM budget on a 32-bit target");
static_assert (sizeof (void *) != 4 || alignof (Core) <= 4,
               "mllee Core must stay 4-byte aligned on a 32-bit target");
static_assert (sizeof (Workspace) <= 208, "mllee workspace exceeds its budget");
static_assert (sizeof (PacklowState) <= 136, "mllee codec state exceeds its budget");
static_assert (_S_arena_bytes == 444, "mllee activation arena size changed");
```

Add a `static_assert` whenever you introduce or resize a budgeted object. The 32-bit layout is also enforced by `tools/mllee_size_check.cpp` via `check_mllee_ram.sh`.

## 4. Mandatory memory constraints

This section is the heart of the document. Every change to `mllee/` or `std/` must respect these limits, and reviewers must reject anything that breaks them.

### 4.1 Hard rules (no exceptions)

1. **No heap allocations in the `mllee`/`std` core path.** No `new`, `delete`, `malloc`, `free`, `calloc`, `realloc`, or `std::allocator`. State lives in fixed-size arenas inside objects.
2. **No writable static storage.** `.data` and `.bss` must be 0 for the module — enforced by `tools/check_mllee_ram.sh`. `const` / `constexpr` tables (e.g. the CRC-16 table in `mllee/core_status.hpp`) live in `.rodata` and are fine.
3. **No global mutable state.** All writable state is inside an object instance (e.g. `Core::_M_ws`, `Core::_M_arena`, `Core::_M_status`).

### 4.2 Quantitative budgets

| Budget | Value | Enforced by | Notes |
|---|---|---|---|
| Module RAM (Core object + deepest call chain) | 1024 bytes | `tools/check_mllee_ram.sh` (`ZZS_MLLEE_RAM_BUDGET`) | 32-bit / Cortex-M4 layout |
| Core object share of RAM | 640 bytes | `static_assert` + `mllee_size_check.cpp` | Includes the 444-byte activation arena |
| Activation arena | 444 bytes | `static_assert (_S_arena_bytes == 444)` | `Core::_M_arena[_S_arena_bytes]{}` |
| Workspace budget | 208 bytes | `static_assert (sizeof (Workspace) <= 208)` | `mllee/core.hpp` |
| Codec state budget | 136 bytes | `static_assert (sizeof (PacklowState) <= 136)` | `mllee/core.hpp` |
| Stack chain limit | 384 bytes | `ZZS_MLLEE_STACK_LIMIT` + `mllee_stack_check.cpp` | Deepest call chain ceiling |
| Flash limit | 16 KB | `tools/mllee_size_check.cpp` / `mllee_train_size_check.cpp` | code + rodata |
| Alignment (32-bit) | 4-byte | `static_assert (alignof (Core) <= 4)` | `mllee/core.hpp` |
| Atomics / log-slot alignment | 64-byte | `alignas(64)` | Cache-line separation |

### 4.3 Arenas and allocation pattern

Fixed-size arenas only:

```cpp
uint8_t _M_arena[_S_arena_bytes]{};
```

`_S_arena_bytes` is `444` (from `mllee/core.hpp`). The arena is zero-initialized (`{}`), stack- or member-embedded, never dynamically sized. Never `new`/`malloc` inside `mllee`/`std`.

### 4.4 Alignment

- 4-byte alignment on 32-bit targets (enforced by `static_assert alignof(Core) <= 4`).
- 64-byte cache-line separation for atomics and log slots: `alignas(64)`.

### 4.5 Quantization, memory-mapped tensors, pooling, checkpointing

The module supports quantization (int8/int4/int2/fp8/fp4/fp3), memory-mapped tensors (model blob is Flash-resident and streamed, never copied), memory pooling, and gradient checkpointing for memory-efficient training. These are implementation techniques, not excuses to exceed the RAM budget — the budget above still applies.

### 4.6 How to verify

```bash
# RAM gate (host, falls back to g++; uses arm-none-eabi-g++ when available)
tools/check_mllee_ram.sh

# With a specific toolchain / stack limit
ZZS_MLLEE_CXX=clang++ ZZS_MLLEE_STACK_LIMIT=288 tools/check_mllee_ram.sh

# Training RAM gate
tools/check_mllee_train_ram.sh

# Cortex-M4 kernels on QEMU mps2-an385 (SKIPs and exits 0 if toolchain/QEMU absent)
tools/run_mllee_m4_qemu.sh
```

These are wired as CMake custom targets (`check_mllee_ram`, `check_mllee_train_ram`, `check_mllee_m4_asm`) so they run as part of the build when the scripts exist.

### 4.7 What breaks the budget and must be rejected in review

- Any `new`/`malloc` in `mllee/` or `std/`.
- Any new writable static or global mutable variable.
- Any increase to `sizeof(Core)`, `sizeof(Workspace)`, or `sizeof(PacklowState)` without a corresponding budget adjustment (and a strong justification).
- Any new `.data`/`.bss` bytes reported by `check_mllee_ram.sh`.
- Any stack-frame growth that pushes the deepest call chain over 384 bytes.
- Any flash growth pushing code+rodata over 16 KB.

## 5. Testing

### 5.1 Framework

Catch2 v3, located via `find_package(Catch2 3 QUIET)` and downloaded via FetchContent (tag `v3.9.1`, shallow) when not installed. See `tests/CMakeLists.txt`.

Two test executables:

- `test_mllee` — `ctest` name `test_mllee`. Sources: `test_feat_fusion.cpp`, `test_mllee_native_dataset.cpp`, `test_mllee_native_inference.cpp`, `test_mllee_native_training.cpp`, `test_mllee_quant.cpp`, `test_mllee_codec.cpp`, `test_mllee_ops.cpp`, `test_mllee_onnx.cpp`, `test_mllee_simd_packed.cpp`, plus globbed `test_mllee_training_*.cpp`.
- `test_sensor_driver` — `ctest` name `test_sensor_driver` (host only, skipped when cross-compiling). Sources under `tests/sensor-driver/`.

82 tests pass. `ctest` runs `test_mllee` and `test_sensor_driver`.

### 5.2 Exceptions/RTTI in tests

The library is built `-fno-exceptions -fno-rtti`, but tests re-enable both:

```cmake
target_compile_options(test_mllee PRIVATE -fexceptions -frtti -funwind-tables)
target_compile_options(test_sensor_driver PRIVATE -fexceptions -frtti -funwind-tables)
```

(MSVC equivalent: `/EHs /GR`.)

### 5.3 Style and hardware independence

- Use `REQUIRE`/`CHECK` style from Catch2.
- **No test should depend on hardware.** For `kernel/sensor-driver`, mock the bus/HAL — see `tests/sensor-driver/test_mock_bus.hpp` and the existing driver tests (`test_lsm6dso_driver.cpp`, etc.).
- Add one test file per module; training tests are globbed via `file(GLOB ... test_mllee_training_*.cpp CONFIGURE_DEPENDS)` so a new file is picked up without editing `tests/CMakeLists.txt`.

### 5.4 Running tests

```bash
ctest --test-dir build --output-on-failure
ctest --test-dir build -R test_mllee --output-on-failure
./build/test_mllee "[ops]"
```

## 6. Contribution / PR checklist (pre-merge)

- [ ] Build Debug **and** Release cleanly (`cmake --build` with no warnings).
- [ ] `ctest` 100% pass (`test_mllee` and `test_sensor_driver`).
- [ ] `clang-format -i` run on every changed file; `clang-format --dry-run --Werror` clean.
- [ ] ASan/UBSan clean on host builds.
- [ ] RAM gate passes: `tools/check_mllee_ram.sh` and `tools/check_mllee_train_ram.sh`.
- [ ] QEMU M4 target passes or SKIPs cleanly: `tools/run_mllee_m4_qemu.sh`.
- [ ] Documentation updated if behavior changed (this file, `README.md`, or module READMEs).
- [ ] No new writable statics, no heap, no exceptions in `mllee/`/`std/`.
- [ ] `static_assert` added for any new or resized budgeted object.
- [ ] Test file added for any new module; sensor-driver tests use mock bus/HAL.
- [ ] `cmake-build-debug/` and other build trees not committed (gitignored).

## 7. Known discrepancies / TODOs (honest)

These exist in the repo today. Do not replicate them in new code; fix them deliberately when reconciling.

1. **C++ standard:** `README.md` says C++26; `CMakeLists.txt` sets C++23. **CMake is the source of truth: C++23.** Align README when reconciling.
2. **License:** `README.md` says GPL 3.0; every source file header says MIT ("Copyright(c) 2026 Tommicord", SPDX). File headers (MIT) are authoritative until a `LICENSE` file is added. Do not change headers without first deciding the license.
3. **`android/` directory:** removed in the working tree (staged deletions). Do not reference it as an active platform target.
4. **`bootstrap/main.cpp`:** currently a stub returning 0. The real `openzzs` executable entry is not yet implemented.
5. **`mllee/status.hpp`:** deleted in the working tree (moved into `mllee/core_status.hpp`); do not `#include "mllee/status.hpp"`.

## 8. License note

- Every source file carries an MIT license block: "Copyright(c) 2026 Tommicord" with the MIT permission text and SPDX.
- `README.md` says GPL 3.0 and references a `LICENSE` file that does not exist — that is a known discrepancy (§7).
- **File headers (MIT) are authoritative until a `LICENSE` file is added and the license is decided.**
- Add a `LICENSE` file matching the file headers. Do not change file headers without first deciding the license and updating `README.md` to match.
