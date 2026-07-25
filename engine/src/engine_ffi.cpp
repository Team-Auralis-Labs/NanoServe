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

int engine_infer(void* handle, const char* prompt, int max_tokens, char* out_buf, int out_buf_len) {
    return engine_run_infer(static_cast<EngineHandle*>(handle), prompt, max_tokens, out_buf, out_buf_len);
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
