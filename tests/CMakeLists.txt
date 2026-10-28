# Unit tests for the openzzs modules.
#
# Catch2 is used as the test framework. It is located with find_package() first
# and, when unavailable, downloaded via FetchContent so a fresh checkout can run
# the suite without any pre-installed dependency.

enable_language(CXX)

find_package(Catch2 3 QUIET)

if (NOT Catch2_FOUND)
  include(FetchContent)
  FetchContent_Declare(Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.9.1
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(Catch2)
endif ()

# The sensor layer is not part of this checkout yet; keep the suite configurable
# without it instead of failing the whole build.
if (TARGET sensr)
  add_executable(test_sensr
    test_main.cpp
    test_sensr.cpp)

  target_link_libraries(test_sensr PRIVATE sensr Catch2::Catch2)

  # The global Release flags disable exceptions and RTTI for size reasons, but
  # the test framework relies on both. Re-enable them for this target only.
  if (MSVC)
    target_compile_options(test_sensr PRIVATE /EHs /GR)
  else ()
    target_compile_options(test_sensr PRIVATE -fexceptions -frtti -funwind-tables)
  endif ()

  include(Catch)
  catch_discover_tests(test_sensr)
endif ()

# Training modules land one header at a time, and several of them can be in
# flight at once; a glob keeps each test file with its module instead of making
# every contributor edit this list.  CONFIGURE_DEPENDS re-runs the check on build
# so a newly added file is picked up without an explicit configure.
file (GLOB _zzs_mllee_training_tests CONFIGURE_DEPENDS
  ${CMAKE_CURRENT_SOURCE_DIR}/test_mllee_training_*.cpp)

add_executable(test_mllee
  test_mllee_quant.cpp
  test_mllee_codec.cpp
  test_mllee_ops.cpp
  test_mllee_onnx.cpp
  test_mllee_simd_packed.cpp
  ${_zzs_mllee_training_tests})

target_link_libraries(test_mllee PRIVATE openzzs-mllee Catch2::Catch2WithMain)
target_compile_definitions(test_mllee PRIVATE ZZS_MLLEE_HOST)

if (MSVC)
  target_compile_options(test_mllee PRIVATE /EHs /GR)
else ()
  target_compile_options(test_mllee PRIVATE -fexceptions -frtti -funwind-tables)
endif ()

add_test(NAME test_mllee COMMAND test_mllee)
