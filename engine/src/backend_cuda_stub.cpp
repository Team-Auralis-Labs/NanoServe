#include "backend.hpp"

std::unique_ptr<ComputeBackend> create_cuda_backend() {
    return nullptr;
}

int probe_cuda_impl() {
    return 0;
}
