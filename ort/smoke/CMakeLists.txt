# Native ORT runtime smoke harness.
#
# Builds a standalone C++ program that dynamically loads the ORT shared
# library built by the current job and exercises the ORT C API directly.
# It never links against the onnxruntime Python wheel.
#
# The ORT C API header is included from the pinned ORT source checkout
# (ORT_SOURCE_DIR) so the API struct layout matches the built library's
# ORT_API_VERSION.
#
# Usage:
#   cmake -S ort/smoke -B ort/smoke/build \
#       -DORT_SOURCE_DIR=ort/source \
#       -DCMAKE_BUILD_TYPE=Release
#   cmake --build ort/smoke/build --config Release --parallel

cmake_minimum_required(VERSION 3.28)
project(ort_smoke CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ORT_SOURCE_DIR)
    message(FATAL_ERROR
        "ORT_SOURCE_DIR must point to the checked-out ORT source root "
        "(containing include/onnxruntime/core/session/onnxruntime_c_api.h)")
endif()
get_filename_component(ORT_SOURCE_DIR_ABS "${ORT_SOURCE_DIR}" ABSOLUTE)

add_executable(ort_smoke ort_smoke.cpp)
# The ORT C API header is at <source>/include/onnxruntime/core/session/
# onnxruntime_c_api.h and is included as <onnxruntime/core/session/...>, so the
# include directory must be <source>/include. The source root is also added so
# any ancillary headers referenced relative to it resolve correctly.
target_include_directories(ort_smoke PRIVATE
    "${ORT_SOURCE_DIR_ABS}/include"
    "${ORT_SOURCE_DIR_ABS}")

if(WIN32)
    # LoadLibrary is in kernel32, always linked.
else()
    target_link_libraries(ort_smoke PRIVATE ${CMAKE_DL_LIBS})
endif()

# Disable compiler warnings as errors for this harness — the ORT C API
# header may use features that trigger warnings on some compilers.
if(MSVC)
    target_compile_options(ort_smoke PRIVATE /W3 /wd4267 /wd4244)
else()
    target_compile_options(ort_smoke PRIVATE -Wall -Wno-unused-parameter)
endif()
