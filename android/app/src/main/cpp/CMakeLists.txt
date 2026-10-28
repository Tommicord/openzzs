cmake_minimum_required(VERSION 3.22.1)
project(openzzs-sensr-jni LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# The repository root: cpp -> main -> src -> app -> android -> <root>
get_filename_component(OPENZZS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../.." ABSOLUTE)

set(OPENZZS_ZZS_std_SOURCES
    ${OPENZZS_ROOT}/std/array.c
    ${OPENZZS_ROOT}/std/atomic.c
    ${OPENZZS_ROOT}/std/heap_allocator.c
    ${OPENZZS_ROOT}/std/log.c
    ${OPENZZS_ROOT}/std/platform.c
)

set(OPENZZS_SENSR_SOURCES
    ${OPENZZS_ROOT}/sensr/sensr.c
    ${OPENZZS_ROOT}/sensr/sensr_mock.c
    ${OPENZZS_ROOT}/sensr/sensr_android.c
)

add_library(openzzs-core STATIC
    ${OPENZZS_ZZS_std_SOURCES}
    ${OPENZZS_SENSR_SOURCES}
)

target_include_directories(openzzs-core PUBLIC ${OPENZZS_ROOT})
target_compile_definitions(openzzs-core PUBLIC OPENZZS_LOG)

# Android-specific system libraries required by the NDK sensor API and logging.
find_library(ANDROID_SENSOR_LIB android)
find_library(ANDROID_LOG_LIB log)

target_link_libraries(openzzs-core PUBLIC ${ANDROID_SENSOR_LIB} ${ANDROID_LOG_LIB})

# JNI bridge shared library loaded by SensrBridge.java.
add_library(openzzs-jni SHARED sensr_jni.c)
target_include_directories(openzzs-jni PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(openzzs-jni PRIVATE openzzs-core)
target_link_libraries(openzzs-jni PRIVATE ${ANDROID_LOG_LIB} ${ANDROID_SENSOR_LIB})

# Export only the JNI entry points from the shared library.
set_target_properties(openzzs-jni PROPERTIES
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
