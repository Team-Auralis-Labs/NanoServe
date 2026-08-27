cmake_minimum_required(VERSION 3.22)

option(NANOSERVE_ENABLE_CUDA "Build CUDA backend" OFF)
option(NANOSERVE_ENABLE_OPENCL "Build OpenCL backend" OFF)
option(NANOSERVE_LLAMA_CPP_KERNELS "Enable llama.cpp kernel subset" ON)

if(NANOSERVE_ENABLE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA SM arch" FORCE)
    set(CMAKE_CUDA_COMPILER "${CMAKE_CURRENT_LIST_DIR}/nvcc_wrapper.sh" CACHE FILEPATH "CUDA compiler" FORCE)
    if(NOT CMAKE_CUDA_HOST_COMPILER)
        set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++ CACHE FILEPATH "CUDA host compiler" FORCE)
    endif()
    set(CMAKE_CUDA_COMPILER_FORCED TRUE CACHE BOOL "" FORCE)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared CACHE STRING "CUDA runtime library type" FORCE)
    project(NanoServeEngine CXX CUDA)
else()
    project(NanoServeEngine CXX)
endif()

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O3 -march=native -mf16c -fPIC)
include_directories(include)

set(RUST_DIR ${CMAKE_SOURCE_DIR}/../rust/nanoq_runtime)
set(ALLOC_DIR ${CMAKE_SOURCE_DIR}/../allocator)

link_directories(${ALLOC_DIR}/target/release)
link_directories(${RUST_DIR}/target/release)

add_custom_target(nanoq_runtime_cargo ALL
    COMMAND cargo build --release --manifest-path ${RUST_DIR}/Cargo.toml
    WORKING_DIRECTORY ${RUST_DIR}
    COMMENT "Building rust/nanoq_runtime"
)

set(ENGINE_SOURCES
    src/engine_ffi.cpp
    src/engine_core.cpp
    src/nanoq_loader.cpp
    src/nanoq_archive.cpp
    src/transformer_gpt2.cpp
    src/transformer_llama.cpp
    src/backend_cpu.cpp
    src/backend_factory.cpp
)

if(NANOSERVE_LLAMA_CPP_KERNELS)
    add_compile_definitions(NANOSERVE_LLAMA_CPP_KERNELS=1)
endif()

if(NANOSERVE_ENABLE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -std=c++20")
    find_package(CUDAToolkit REQUIRED)
    link_directories(${CUDAToolkit_LIBRARY_DIR})
    list(APPEND ENGINE_SOURCES src/backend_cuda.cu)
    add_compile_definitions(NANOSERVE_HAS_CUDA)
else()
    list(APPEND ENGINE_SOURCES src/backend_cuda_stub.cpp)
endif()

if(NANOSERVE_ENABLE_OPENCL)
    find_package(OpenCL REQUIRED)
    list(APPEND ENGINE_SOURCES src/backend_opencl.cpp)
    add_compile_definitions(NANOSERVE_HAS_OPENCL)
else()
    list(APPEND ENGINE_SOURCES src/backend_opencl_stub.cpp)
endif()

add_library(nanoserve_engine SHARED ${ENGINE_SOURCES})
add_dependencies(nanoserve_engine nanoq_runtime_cargo)
target_link_libraries(nanoserve_engine
    buddy_alloc
    pthread
    -Wl,--whole-archive
    ${RUST_DIR}/target/release/libnanoq_runtime.a
    -Wl,--no-whole-archive
    dl
    m
)

if(NANOSERVE_ENABLE_CUDA)
    set_target_properties(nanoserve_engine PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
        CUDA_RESOLVE_DEVICE_SYMBOLS OFF
    )
    target_link_libraries(nanoserve_engine CUDA::cudart)
endif()

if(NANOSERVE_ENABLE_OPENCL)
    target_include_directories(nanoserve_engine PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_link_libraries(nanoserve_engine ${OpenCL_LIBRARIES})
endif()

add_executable(test_transformer_gpt2 tests/test_transformer_gpt2.cpp)
target_link_libraries(test_transformer_gpt2 nanoserve_engine buddy_alloc pthread dl m)
