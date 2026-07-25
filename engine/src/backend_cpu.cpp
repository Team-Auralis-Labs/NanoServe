#include "backend.hpp"
#include "nanoserve_simd.hpp"

class CPUSimdBackend : public ComputeBackend {
public:
    float gemv_int8(std::span<const int8_t> weights, std::span<const float> acts) override {
        return int8_dot_avx2(weights, acts);
    }
    const char* name() const override { return "cpu"; }
};

std::unique_ptr<ComputeBackend> create_cpu_backend() {
    return std::make_unique<CPUSimdBackend>();
}
