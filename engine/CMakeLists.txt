cmake_minimum_required(VERSION 3.22)

option(NANOSERVE_ENABLE_CUDA "Build CUDA backend" OFF)
option(NANOSERVE_ENABLE_OPENCL "Build OpenCL backend" OFF)

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
link_directories(${CMAKE_SOURCE_DIR}/../allocator/target/release)

set(ENGINE_SOURCES
    src/engine_ffi.cpp
    src/engine_core.cpp
    src/nanoq_loader.cpp
    src/backend_cpu.cpp
    src/backend_factory.cpp
)

if(NANOSERVE_ENABLE_CUDA)
    # CMake 3.22 in CUDA devel images may not register CUDA20; pass -std=c++20 to nvcc explicitly.
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
target_link_libraries(nanoserve_engine buddy_alloc pthread)

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
