# Emscripten-only build (invoked via deployment/wasm/build.sh + emcmake).
cmake_minimum_required(VERSION 3.22)
project(NanoServeEngineWasm CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR}/include)

add_compile_options(-O3 -fno-exceptions -fno-rtti)
add_compile_definitions(NANOSERVE_WASM=1)

set(WASM_SOURCES
    src/engine_ffi.cpp
    src/engine_core.cpp
    src/nanoq_loader.cpp
    src/backend_cpu.cpp
    src/backend_factory.cpp
    src/backend_cuda_stub.cpp
    src/backend_opencl_stub.cpp
    src/buddy_pool_wasm.cpp
)

add_executable(nanoserve_engine_wasm ${WASM_SOURCES})

set(WASM_EXPORTS
    _engine_init
    _engine_init_with_model_bytes
    _engine_reload_model_bytes
    _engine_infer
    _engine_model_info
    _engine_cleanup
    _malloc
    _free
)

target_link_options(nanoserve_engine_wasm PRIVATE
    "-sWASM=1"
    "-sMODULARIZE=1"
    "-sEXPORT_NAME=createNanoServeModule"
    "-sALLOW_MEMORY_GROWTH=1"
    "-sENVIRONMENT=web"
    "-sEXPORTED_FUNCTIONS=[${WASM_EXPORTS}]"
    "-sEXPORTED_RUNTIME_METHODS=[cwrap,UTF8ToString,getValue,HEAPU8]"
    "-sFILESYSTEM=0"
    "-sNO_EXIT_RUNTIME=1"
)

set_target_properties(nanoserve_engine_wasm PROPERTIES
    OUTPUT_NAME "nanoserve_engine"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/out"
)
