cmake_minimum_required(VERSION 3.20)
project(payload_generator_op LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Find packages ─────────────────────────────────────────────────────────────
find_package(holoscan REQUIRED
    PATHS /opt/nvidia/holoscan/lib/cmake/holoscan)

find_package(CUDAToolkit REQUIRED)

# ── pybind11 (header-only, ships in /root/pybind11) ──────────────────────────
set(PYBIND11_INCLUDE_DIR "/root/pybind11/include" CACHE PATH "pybind11 headers")

# ── Python headers ────────────────────────────────────────────────────────────
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# ── Python extension: single self-contained _payload_generator_op.so ─────────
# Compile the operator and its pybind11 bindings into one module so there is
# no separate libpayload_generator_op.so to find at runtime.
add_library(_payload_generator_op MODULE
    payload_generator_op.cpp
    payload_generator_op_pybind.cpp)

target_include_directories(_payload_generator_op PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PYBIND11_INCLUDE_DIR}
    ${Python3_INCLUDE_DIRS}
    /usr/local/include)   # hololink headers live here

target_link_libraries(_payload_generator_op PRIVATE
    holoscan::core
    CUDA::cudart)

# holoscan/hololink were compiled with pybind11 <=2.11 which uses a plain capsule name:
#   __pybind11_internals_v5__
# pybind11 2.12+ appends a platform ABI suffix (e.g. _gcc_libstdcpp_cxxabi1018).
# Override the suffix macros to empty so our capsule name matches exactly.
target_compile_definitions(_payload_generator_op PRIVATE
    "PYBIND11_COMPILER_TYPE=\"\""
    "PYBIND11_STDLIB=\"\""
    "PYBIND11_BUILD_ABI=\"\"")

# pybind11 modules must not have a "lib" prefix and must use .so suffix
set_target_properties(_payload_generator_op PROPERTIES
    PREFIX ""
    SUFFIX ".so")

# ── Install next to the Python script ────────────────────────────────────────
install(TARGETS _payload_generator_op
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
