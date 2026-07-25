#include "backend.hpp"

std::unique_ptr<ComputeBackend> create_opencl_backend() {
    return nullptr;
}

int probe_opencl_impl() {
    return 0;
}
