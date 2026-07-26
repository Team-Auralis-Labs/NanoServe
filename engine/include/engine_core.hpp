#pragma once
#include "backend.hpp"
#include "nanoq_loader.hpp"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct EngineHandle {
    void* weights_pool = nullptr;
    void* scratch_pool = nullptr;
    int8_t* weights = nullptr;
    size_t weights_len = 0;
    EngineBackendKind backend_kind = EngineBackendKind::Cpu;
    std::unique_ptr<ComputeBackend> backend;
    NanoqModel model;
    bool has_model = false;
    std::string model_path;
    std::string model_info_json;
};

EngineHandle* engine_create(EngineBackendKind kind);
EngineHandle* engine_create_with_model(EngineBackendKind kind, const char* nanoq_path);
int engine_reload_model(EngineHandle* h, const char* nanoq_path);
const char* engine_get_model_info(EngineHandle* h);
int engine_run_infer(EngineHandle* h, const char* prompt, int max_tokens, char* out_buf, int out_buf_len);
void engine_destroy_handle(EngineHandle* h);
