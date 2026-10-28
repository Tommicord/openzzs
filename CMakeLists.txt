cmake_minimum_required(VERSION 3.20)
project(openzzs VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
  add_compile_options(/Oy-)
else ()
  add_compile_options(-fno-omit-frame-pointer)
endif ()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  add_compile_options(
    -Oz
    -ffunction-sections
    -fdata-sections)
  add_link_options(-Wl,--gc-sections -Wl,--strip-all)
  set(OPENZZS_RUNTIME_RELEASE_OPTIONS
    -fno-unwind-tables
    -fno-asynchronous-unwind-tables)
endif ()

set(ZZS_STD_SOURCES
  std/basic_allocator.hpp
  std/basic_iterator.hpp
  std/fmt_sink.hpp
  std/fmt.hpp
  std/io.hpp
  std/log.hpp
  std/maybe.hpp
  std/sighand.hpp
  std/string_view.hpp
  std/sunwind.hpp
)
add_library(openzzs-std STATIC ${ZZS_STD_SOURCES})
target_include_directories(openzzs-std PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(openzzs-std PROPERTIES LINKER_LANGUAGE CXX)

# ---------------------------------------------------------------------------
# mllee optional assembly kernels.
#
# The target and every instruction-set extension are selected at configure
# time (never probed at run time) so the hot path stays free of CPUID branches,
# dispatch tables and hidden module state.  That is what makes the instruction
# count of a layer run predictable on a Cortex-M4 and repeatable on a host.
# The portable C++ fallback in mllee/simd.hpp is used when this is disabled or
# when the selected target has no kernel for a primitive.
#
# File names describe the instruction they depend on:
#   mllee_int8_dot_*      int8 MAC / reduce / requant primitives
#   mllee_int8_reduce_*   element-wise saturating set arithmetic
#   mllee_int8_stats_*    sum / sum-of-squares reduction used by the norm ops
#   mllee_packlow_*       shuffle-based codec and quantized-weight expansion
#
# CRC-16/CCITT has no kernel here on purpose: the fold needs a carry-less
# multiply ladder whose constants are easy to get subtly wrong, the blob it
# guards is a few hundred bytes at load time, and no Cortex-M4 has PCLMULQDQ.
# The portable table form in mllee/status.hpp is branch-free and identical on
# every target, which is the property this module actually needs.
# ---------------------------------------------------------------------------
option(OPENZZS_MLLEE_ASM "Build mllee hand-written assembly kernels" ON)
option(OPENZZS_MLLEE_SSSE3 "Use SSSE3 pshufb kernels on x86-64 (2006+ parts)" ON)
option(OPENZZS_MLLEE_AVX2 "Use AVX2 mllee kernels on x86-64 (Haswell 2013+)" ON)
option(OPENZZS_MLLEE_AVX_VNNI
  "Use AVX-VNNI vpdpbusd kernels on x86-64 (Ice Lake 2019+ parts)" OFF)
option(OPENZZS_MLLEE_AVX512
  "Use AVX-512 VNNI/BW kernels on x86-64 (Skylake-X 2017+ / Ice Lake+)" OFF)
set(OPENZZS_MLLEE_ASM_TARGET "auto" CACHE STRING
  "mllee assembly target: auto, x86_64, cortex-m4, none")

set(_zzs_mllee_asm_target "${OPENZZS_MLLEE_ASM_TARGET}")
if (_zzs_mllee_asm_target STREQUAL "auto")
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _zzs_mllee_arch)
  string(TOLOWER "${CMAKE_CXX_COMPILER}" _zzs_mllee_cxx)
  if (_zzs_mllee_arch MATCHES "x86_64|amd64")
    set(_zzs_mllee_asm_target "x86_64")
  elseif (_zzs_mllee_cxx MATCHES "arm-none-eabi" OR _zzs_mllee_arch MATCHES "cortex|^arm")
    set(_zzs_mllee_asm_target "cortex-m4")
  else ()
    set(_zzs_mllee_asm_target "none")
  endif ()
endif ()

if (OPENZZS_MLLEE_ASM AND NOT _zzs_mllee_asm_target STREQUAL "none")
  enable_language(ASM)
  if (_zzs_mllee_asm_target STREQUAL "x86_64")
    set(_zzs_mllee_asm_sources
      mllee/asm-x86_64/mllee_int8_dot_sse2.S
      mllee/asm-x86_64/mllee_int8_reduce_sse2.S
      mllee/asm-x86_64/mllee_int8_stats_sse2.S
      mllee/asm-x86_64/mllee_onnx_wire_x86_64.S
      mllee/asm-x86_64/mllee_train_q15_sse2.S)
    set(_zzs_mllee_asm_features "sse2")
    if (OPENZZS_MLLEE_SSSE3)
      list(APPEND _zzs_mllee_asm_sources mllee/asm-x86_64/mllee_packlow_shuffle_ssse3.S)
      set_source_files_properties(mllee/asm-x86_64/mllee_packlow_shuffle_ssse3.S
        PROPERTIES COMPILE_OPTIONS "-mssse3")
      list(APPEND _zzs_mllee_asm_features "ssse3")
    endif ()
    if (OPENZZS_MLLEE_AVX2)
      list(APPEND _zzs_mllee_asm_sources
        mllee/asm-x86_64/mllee_int8_dot_avx2.S
        mllee/asm-x86_64/mllee_packed_dot_avx2.S
        mllee/asm-x86_64/mllee_elementwise_avx2.S)
      set_source_files_properties(mllee/asm-x86_64/mllee_int8_dot_avx2.S
        PROPERTIES COMPILE_OPTIONS "-mavx2")
      set_source_files_properties(mllee/asm-x86_64/mllee_packed_dot_avx2.S
        PROPERTIES COMPILE_OPTIONS "-mavx2")
      set_source_files_properties(mllee/asm-x86_64/mllee_elementwise_avx2.S
        PROPERTIES COMPILE_OPTIONS "-mavx2")
      list(APPEND _zzs_mllee_asm_features "avx2")
    endif ()
    # vpdpbusd needs both -mavxvnni and an AVX2 baseline, and it is the only
    # kernel that requires a CPU newer than AVX2, so it stays opt-in: a host
    # without it must still build and run the whole suite.
    if (OPENZZS_MLLEE_AVX_VNNI)
      if (NOT OPENZZS_MLLEE_AVX2)
        message(FATAL_ERROR "OPENZZS_MLLEE_AVX_VNNI requires OPENZZS_MLLEE_AVX2")
      endif ()
      list(APPEND _zzs_mllee_asm_sources
        mllee/asm-x86_64/mllee_int8_dot_avx_vnni.S)
      set_source_files_properties(mllee/asm-x86_64/mllee_int8_dot_avx_vnni.S
        PROPERTIES COMPILE_OPTIONS "-mavxvnni;-mavx2")
      list(APPEND _zzs_mllee_asm_features "avxvnni")
    endif ()
    # AVX-512 VNNI folds 64 codes per vpdpbusd.  It is opt-in for the same
    # reason as AVX-VNNI: the emitted binary faults on older hosts, and the
    # module deliberately has no run-time CPUID dispatch.
    if (OPENZZS_MLLEE_AVX512)
      if (NOT OPENZZS_MLLEE_AVX2)
        message(FATAL_ERROR "OPENZZS_MLLEE_AVX512 requires OPENZZS_MLLEE_AVX2")
      endif ()
      list(APPEND _zzs_mllee_asm_sources
        mllee/asm-x86_64/mllee_int8_dot_avx512.S)
      set_source_files_properties(mllee/asm-x86_64/mllee_int8_dot_avx512.S
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vnni")
      list(APPEND _zzs_mllee_asm_features "avx512vnni")
    endif ()
    add_library(openzzs-mllee-asm STATIC ${_zzs_mllee_asm_sources})
    target_compile_definitions(openzzs-mllee-asm PUBLIC ZZS_MLLEE_ASM_X86_64=1)
    if (OPENZZS_MLLEE_SSSE3)
      target_compile_definitions(openzzs-mllee-asm PUBLIC ZZS_MLLEE_ASM_SSSE3=1)
    endif ()
    if (OPENZZS_MLLEE_AVX2)
      target_compile_definitions(openzzs-mllee-asm PUBLIC ZZS_MLLEE_ASM_AVX2=1)
    endif ()
    if (OPENZZS_MLLEE_AVX_VNNI)
      target_compile_definitions(openzzs-mllee-asm PUBLIC ZZS_MLLEE_ASM_AVX_VNNI=1)
    endif ()
    if (OPENZZS_MLLEE_AVX512)
      target_compile_definitions(openzzs-mllee-asm PUBLIC ZZS_MLLEE_ASM_AVX512=1)
    endif ()
    list(JOIN _zzs_mllee_asm_features "+" _zzs_mllee_asm_features)
  elseif (_zzs_mllee_asm_target STREQUAL "cortex-m4")
    add_library(openzzs-mllee-asm STATIC
      mllee/asm-cortex-m4/mllee_int8_mac_cortex_m4.S
      mllee/asm-cortex-m4/mllee_int8_reduce_cortex_m4.S
      mllee/asm-cortex-m4/mllee_int4_expand_cortex_m4.S
      mllee/asm-cortex-m4/mllee_onnx_wire_cortex_m4.S
      mllee/asm-cortex-m4/mllee_train_q15_lanes_cortex_m4.S
      mllee/asm-cortex-m4/mllee_train_q15_mac_cortex_m4.S)
    target_compile_definitions(openzzs-mllee-asm PUBLIC ZZS_MLLEE_ASM_CORTEX_M4=1)
    target_compile_options(openzzs-mllee-asm PRIVATE -mcpu=cortex-m4 -mthumb)
    set(_zzs_mllee_asm_features "dsp")
  else ()
    message(FATAL_ERROR "Unknown OPENZZS_MLLEE_ASM_TARGET '${_zzs_mllee_asm_target}'")
  endif ()
  set_target_properties(openzzs-mllee-asm PROPERTIES LINKER_LANGUAGE CXX)
  target_include_directories(openzzs-mllee-asm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  message(STATUS "mllee: assembly kernels enabled for ${_zzs_mllee_asm_target} (${_zzs_mllee_asm_features})")
else ()
  message(STATUS "mllee: assembly kernels disabled, using the portable C++ path")
endif ()

add_library(openzzs-mllee INTERFACE)
target_compile_options(openzzs-mllee INTERFACE -fno-exceptions -fno-rtti)
target_include_directories(openzzs-mllee INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(openzzs-mllee INTERFACE cxx_std_23)
target_link_libraries(openzzs-mllee INTERFACE openzzs-std)
if (TARGET openzzs-mllee-asm)
  target_link_libraries(openzzs-mllee INTERFACE openzzs-mllee-asm)
endif ()

if (NOT CMAKE_CROSSCOMPILING)
  add_executable(mllee_bench tools/mllee_bench.cpp)
  target_link_libraries(mllee_bench PRIVATE openzzs-mllee)
  target_compile_definitions(mllee_bench PRIVATE ZZS_MLLEE_HOST)
endif ()

find_program(ZZS_BASH bash)
if (ZZS_BASH AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_mllee_ram.sh)
  add_custom_target(check_mllee_ram
    COMMAND ${ZZS_BASH} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_mllee_ram.sh
            ${CMAKE_CURRENT_BINARY_DIR}/mllee-ram
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking the mllee RAM/stack/Flash budget"
    VERBATIM)
endif ()

if (ZZS_BASH AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_mllee_train_ram.sh)
  add_custom_target(check_mllee_train_ram
    COMMAND ${ZZS_BASH} ${CMAKE_CURRENT_SOURCE_DIR}/tools/check_mllee_train_ram.sh
            ${CMAKE_CURRENT_BINARY_DIR}/mllee-train-ram
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Checking the mllee training RAM/stack/Flash budget"
    VERBATIM)
endif ()

if (ZZS_BASH AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_mllee_m4_qemu.sh)
  # The script prints SKIP and exits 0 when the Arm toolchain or QEMU is absent,
  # so this target is safe to build everywhere.
  add_custom_target(check_mllee_m4_asm
    COMMAND ${ZZS_BASH} ${CMAKE_CURRENT_SOURCE_DIR}/tools/run_mllee_m4_qemu.sh
            ${CMAKE_CURRENT_BINARY_DIR}/mllee-m4
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Running the mllee Cortex-M4 kernels on QEMU mps2-an385"
    VERBATIM)
endif ()

add_executable(openzzs bootstrap/main.cpp)
target_link_libraries(openzzs PRIVATE openzzs-std)

if (OPENZZS_RUNTIME_RELEASE_OPTIONS)
  target_compile_options(openzzs-std PRIVATE ${OPENZZS_RUNTIME_RELEASE_OPTIONS})
  target_compile_options(openzzs PRIVATE ${OPENZZS_RUNTIME_RELEASE_OPTIONS})
  target_compile_options(openzzs-mllee INTERFACE ${OPENZZS_RUNTIME_RELEASE_OPTIONS})
endif ()

option(OPENZZS_BUILD_TESTS "Build the unit test suite" ON)

if (OPENZZS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif ()
