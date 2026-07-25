#pragma once
#include "backend.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>

struct EngineHandle {
    void* weights_pool;
    void* scratch_pool;
    int8_t* weights;
    size_t weights_len;
    EngineBackendKind backend_kind;
    std::unique_ptr<ComputeBackend> backend;
};

EngineHandle* engine_create(EngineBackendKind kind);
int engine_run_infer(EngineHandle* h, const char* prompt, int max_tokens, char* out_buf, int out_buf_len);
void engine_destroy_handle(EngineHandle* h);
