#include "backend.hpp"
#include "nanoserve_simd.hpp"

namespace {

float int8_dot_scaled(std::span<const int8_t> weights, std::span<const float> scales,
                      std::span<const float> acts) {
    if (weights.empty() || acts.empty()) return 0.0f;
    if (scales.empty()) return int8_dot_avx2(weights, acts);
    float acc = 0.0f;
    const size_t n = std::min(weights.size(), acts.size());
    for (size_t i = 0; i < n; ++i) {
        const float scale = scales[std::min(i, scales.size() - 1)];
        acc += static_cast<float>(weights[i]) * scale * acts[i];
    }
    return acc;
}

}  // namespace

class CPUSimdBackend : public ComputeBackend {
public:
    float gemv_int8(std::span<const int8_t> weights, std::span<const float> scales,
                    std::span<const float> acts) override {
        if (pool_view_.nanoq && !pool_view_.nanoq->scales.empty())
            return int8_dot_scaled(weights, pool_view_.nanoq->scales, acts);
        if (!scales.empty())
            return int8_dot_scaled(weights, scales, acts);
        return int8_dot_avx2(weights, acts);
    }

    float gemv_fp16(std::span<const uint16_t> weights, std::span<const float> acts) override {
        return fp16_dot_avx2(weights, acts);
    }

    float gemv_fp4(std::span<const uint8_t> packed, std::span<const float> scales,
                   int block_size, std::span<const float> acts) override {
        return fp4_dot(packed, scales, block_size, acts);
    }

    void gemm_int8(std::span<const int8_t> weights, std::span<const float> scales,
                   std::span<const float> input, std::span<float> output,
                   int rows, int cols, int out_dim) override {
        for (int o = 0; o < out_dim; ++o) {
            float acc = 0.0f;
            const float scale = scales.empty() ? 1.0f : scales[std::min(static_cast<size_t>(o), scales.size() - 1)];
            for (int i = 0; i < cols; ++i)
                acc += static_cast<float>(weights[static_cast<size_t>(o * cols + i)]) * scale *
                       input[static_cast<size_t>(i)];
            output[static_cast<size_t>(o)] = acc;
        }
        (void)rows;
    }

    const char* name() const override { return "cpu"; }
};

std::unique_ptr<ComputeBackend> create_cpu_backend() {
    return std::make_unique<CPUSimdBackend>();
}
