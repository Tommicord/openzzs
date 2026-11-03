cmake_minimum_required(VERSION 3.22.1)

project("openzzs-android" LANGUAGES CXX C)

# NDK Clang 14 supports gnu++2b (C++23 draft), not c++23 yet
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++2b -fexceptions -frtti -O2 -g -DZZS_MLLEE_HOST")

# OpenZZS source root (project root = ../../.. from this file)
set(OPENZZS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../..")

# Include paths for kernel headers (header-only templates/inline functions)
include_directories(${OPENZZS_ROOT})
include_directories(${OPENZZS_ROOT}/kernel)
include_directories(${OPENZZS_ROOT}/mllee)
include_directories(${OPENZZS_ROOT}/std)
include_directories(${OPENZZS_ROOT}/bootstrap-monitor)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Android NDK libraries
find_library(ANDROID_LOG_LIBRARY log)
find_library(ANDROID_NATIVE_LIBRARY android)

# Host generator for the model blob (compiled with the host toolchain, not the
# NDK). It serializes the model at build time and emits model_blob_generated.cpp
# defining g_model_blob with the real blob bytes.
set(GEN_MODEL_BLOB_SRC "${CMAKE_CURRENT_SOURCE_DIR}/gen_model_blob.cpp")
set(GEN_MODEL_BLOB_EXE "${CMAKE_CURRENT_BINARY_DIR}/gen_model_blob_host")
set(GEN_MODEL_BLOB_OUT "${CMAKE_CURRENT_BINARY_DIR}/model_blob_generated.cpp")

# Locate a host C++ compiler (the NDK toolchain targets Android, not the host).
find_program(HOST_CXX g++)

# Build-time rule: compile the generator for the host, run it to produce
# model_blob_generated.cpp, then compile that into the Android library.
add_custom_command(
    OUTPUT ${GEN_MODEL_BLOB_OUT}
    COMMAND ${HOST_CXX} -std=gnu++2b -O2 -DZZS_MLLEE_HOST
        -I${OPENZZS_ROOT} -I${OPENZZS_ROOT}/mllee-native
        ${GEN_MODEL_BLOB_SRC} -o ${GEN_MODEL_BLOB_EXE}
    COMMAND ${GEN_MODEL_BLOB_EXE} ${GEN_MODEL_BLOB_OUT}
    DEPENDS ${GEN_MODEL_BLOB_SRC}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Generating model blob"
    VERBATIM
)

add_custom_target(gen_model_blob_target DEPENDS ${GEN_MODEL_BLOB_OUT})

# Source files: Android JNI pipeline + Android HAL + generated model blob.
# Note: sensr_jni.cpp (not bootstrap-monitor/main.cpp) is used on Android
# because it implements the correct slow-sensor upsampling path for the
# preprocessing bank.  main.cpp is the host-only standalone monitor.
set(OPENZZS_NATIVE_SRCS
    ${OPENZZS_ROOT}/android/app/src/main/cpp/sensr_jni.cpp
    ${OPENZZS_ROOT}/kernel/sensor-driver/platforms/android/platform_hal.cpp
    ${GEN_MODEL_BLOB_OUT}
)

add_library(openzzs-android SHARED ${OPENZZS_NATIVE_SRCS})
add_dependencies(openzzs-android gen_model_blob_target)

target_link_libraries(openzzs-android
    ${ANDROID_LOG_LIBRARY}
    ${ANDROID_NATIVE_LIBRARY}
    aaudio
)

# Allow the shared library to find dependencies at runtime
set_target_properties(openzzs-android PROPERTIES
    LINKER_LANGUAGE CXX
    SHARED_LIBRARY_PREFIX "lib"
)
