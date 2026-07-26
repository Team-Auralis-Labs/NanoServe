#include "backend.hpp"
#include "nanoserve_simd.hpp"

class CPUSimdBackend : public ComputeBackend {
public:
    float gemv_int8(std::span<const int8_t> weights, std::span<const float> acts) override {
        return int8_dot_avx2(weights, acts);
    }

    float gemv_fp16(std::span<const uint16_t> weights, std::span<const float> acts) override {
        return fp16_dot_avx2(weights, acts);
    }

    float gemv_fp4(std::span<const uint8_t> packed, std::span<const float> scales,
                   int block_size, std::span<const float> acts) override {
        return fp4_dot(packed, scales, block_size, acts);
    }

    const char* name() const override { return "cpu"; }
};

std::unique_ptr<ComputeBackend> create_cpu_backend() {
    return std::make_unique<CPUSimdBackend>();
}
