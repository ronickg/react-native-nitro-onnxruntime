project(nitroonnxruntime)
cmake_minimum_required(VERSION 3.9.0)

set(PACKAGE_NAME nitroonnxruntime)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_CXX_STANDARD 20)

set(build_DIR ${CMAKE_SOURCE_DIR}/build)
file(GLOB libonnxruntime_DIR "${build_DIR}/onnxruntime-jni/jni/${ANDROID_ABI}")
file(GLOB libonnxruntime_include_DIRS "${build_DIR}/onnxruntime-headers/headers")

# Define C++ library and add all sources
add_library(${PACKAGE_NAME} SHARED src/main/cpp/cpp-adapter.cpp ../cpp/Onnxruntime.cpp ../cpp/InferenceSession.cpp)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${libonnxruntime_DIR}/libonnxruntime.so"
)

# Add Nitrogen specs :)
include(${CMAKE_SOURCE_DIR}/../nitrogen/generated/android/nitroonnxruntime+autolinking.cmake)

# Set up local includes
include_directories("src/main/cpp" "../cpp" "${libonnxruntime_include_DIRS}")

find_library(LOG_LIB log)

# Link all libraries together
target_link_libraries(
        ${PACKAGE_NAME}
        ${LOG_LIB}
        android # <-- Android core
        onnxruntime
)
