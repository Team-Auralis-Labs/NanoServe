#include "backend.hpp"

extern std::unique_ptr<ComputeBackend> create_cpu_backend();
extern std::unique_ptr<ComputeBackend> create_cuda_backend();
extern std::unique_ptr<ComputeBackend> create_opencl_backend();
extern int probe_cuda_impl();
extern int probe_opencl_impl();

std::unique_ptr<ComputeBackend> create_backend(EngineBackendKind kind) {
    switch (kind) {
        case EngineBackendKind::Cpu: return create_cpu_backend();
        case EngineBackendKind::Cuda: return create_cuda_backend();
        case EngineBackendKind::OpenCl: return create_opencl_backend();
    }
    return nullptr;
}

int probe_cuda() { return probe_cuda_impl(); }
int probe_opencl() { return probe_opencl_impl(); }

const char* backend_name(EngineBackendKind kind) {
    switch (kind) {
        case EngineBackendKind::Cpu: return "cpu";
        case EngineBackendKind::Cuda: return "cuda";
        case EngineBackendKind::OpenCl: return "opencl";
    }
    return "unknown";
}
