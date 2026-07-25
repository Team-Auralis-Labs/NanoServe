#include "backend.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <stdexcept>

constexpr int BLOCK = 256;

// Tiled INT8 GEMV matmul: score = W·a — host W/a from buddy-allocator pools.
__global__ void int8_gemv_kernel(const int8_t* __restrict__ weights,
                                 const float* __restrict__ acts,
                                 float* __restrict__ out,
                                 int n) {
    __shared__ float partial[BLOCK];
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        acc += static_cast<float>(weights[i]) * acts[i];
    partial[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) sum += partial[i];
        atomicAdd(out, sum);
    }
}

class CUDABackend : public ComputeBackend {
    int8_t* d_weights_ = nullptr;
    float* d_acts_ = nullptr;
    float* d_out_ = nullptr;
    size_t cap_ = 0;

    void ensure_capacity(size_t n) {
        if (n <= cap_) return;
        int8_t* nw = nullptr;
        float* na = nullptr;
        if (cudaMalloc(&nw, n * sizeof(int8_t)) != cudaSuccess) return;
        if (cudaMalloc(&na, n * sizeof(float)) != cudaSuccess) {
            cudaFree(nw);
            return;
        }
        if (d_weights_) cudaFree(d_weights_);
        if (d_acts_) cudaFree(d_acts_);
        d_weights_ = nw;
        d_acts_ = na;
        cap_ = n;
        if (!d_out_ && cudaMalloc(&d_out_, sizeof(float)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc out");
    }

    static void check(cudaError_t err, const char* msg) {
        if (err != cudaSuccess)
            throw std::runtime_error(msg);
    }

public:
    ~CUDABackend() override {
        if (d_weights_) cudaFree(d_weights_);
        if (d_acts_) cudaFree(d_acts_);
        if (d_out_) cudaFree(d_out_);
    }

    float gemv_int8(std::span<const int8_t> weights, std::span<const float> acts) override {
        size_t n = std::min(weights.size(), acts.size());
        ensure_capacity(n);

        const int8_t* h_w = pool_view_.weights ? pool_view_.weights : weights.data();
        const float* h_a = pool_view_.activations ? pool_view_.activations : acts.data();

        check(cudaMemcpy(d_weights_, h_w, n * sizeof(int8_t), cudaMemcpyHostToDevice), "H2D weights");
        check(cudaMemcpy(d_acts_, h_a, n * sizeof(float), cudaMemcpyHostToDevice), "H2D acts");
        float zero = 0.0f;
        check(cudaMemcpy(d_out_, &zero, sizeof(float), cudaMemcpyHostToDevice), "H2D out");

        int threads = BLOCK;
        int blocks = static_cast<int>((n + threads - 1) / threads);
        if (blocks < 1) blocks = 1;
        int8_gemv_kernel<<<blocks, threads>>>(d_weights_, d_acts_, d_out_, static_cast<int>(n));
        check(cudaGetLastError(), "kernel launch");
        check(cudaDeviceSynchronize(), "kernel sync");

        float result = 0.0f;
        check(cudaMemcpy(&result, d_out_, sizeof(float), cudaMemcpyDeviceToHost), "D2H out");
        return result;
    }

    const char* name() const override { return "cuda"; }
};

std::unique_ptr<ComputeBackend> create_cuda_backend() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return nullptr;
    return std::make_unique<CUDABackend>();
}

int probe_cuda_impl() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 0;
    return 1;
}
