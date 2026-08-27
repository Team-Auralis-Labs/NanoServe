#include "backend.hpp"
#include "engine_core.hpp"

extern "C" {

void* engine_init() {
    return engine_create(EngineBackendKind::Cpu);
}

void* engine_init_backend(int backend) {
    if (backend < static_cast<int>(EngineBackendKind::Cpu) ||
        backend > static_cast<int>(EngineBackendKind::OpenCl))
        return nullptr;
    return engine_create(static_cast<EngineBackendKind>(backend));
}

void* engine_init_with_model(const char* nanoq_path, int backend) {
    if (backend < static_cast<int>(EngineBackendKind::Cpu) ||
        backend > static_cast<int>(EngineBackendKind::OpenCl))
        return nullptr;
    return engine_create_with_model(static_cast<EngineBackendKind>(backend), nanoq_path);
}

void* engine_init_with_model_bytes(const uint8_t* data, size_t len, int backend) {
    if (backend < static_cast<int>(EngineBackendKind::Cpu) ||
        backend > static_cast<int>(EngineBackendKind::OpenCl))
        return nullptr;
    return engine_create_with_model_bytes(
        static_cast<EngineBackendKind>(backend), data, len);
}

int engine_reload_model(void* handle, const char* nanoq_path) {
    return ::engine_reload_model(static_cast<EngineHandle*>(handle), nanoq_path);
}

int engine_reload_model_bytes(void* handle, const uint8_t* data, size_t len) {
    return ::engine_reload_model_bytes(static_cast<EngineHandle*>(handle), data, len);
}

const char* engine_model_info(void* handle) {
    return engine_get_model_info(static_cast<EngineHandle*>(handle));
}

int engine_infer(void* handle, const char* prompt, int max_tokens, char* out_buf, int out_buf_len) {
    return engine_run_infer(static_cast<EngineHandle*>(handle), prompt, max_tokens, out_buf, out_buf_len);
}

int engine_reset_kv(void* handle) {
    return engine_reset_kv_cache(static_cast<EngineHandle*>(handle));
}

void engine_cleanup(void* handle) {
    engine_destroy_handle(static_cast<EngineHandle*>(handle));
}

void engine_destroy(void* handle) {
    engine_cleanup(handle);
}

int engine_probe_cuda() { return probe_cuda(); }
int engine_probe_opencl() { return probe_opencl(); }

const char* engine_backend_name(int backend) {
    if (backend < static_cast<int>(EngineBackendKind::Cpu) ||
        backend > static_cast<int>(EngineBackendKind::OpenCl))
        return "unknown";
    return backend_name(static_cast<EngineBackendKind>(backend));
}

}
