#include "backend.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {

float fp16_to_float_host(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float val = static_cast<float>(mant) / 1024.0f;
        return sign ? -val * (1.0f / 16384.0f) : val * (1.0f / 16384.0f);
    }
    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
    uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

float fp16_dot_host(std::span<const uint16_t> weights, std::span<const float> acts) {
    size_t n = std::min(weights.size(), acts.size());
    float acc = 0.0f;
    for (size_t i = 0; i < n; ++i)
        acc += fp16_to_float_host(weights.data()[i]) * acts.data()[i];
    return acc;
}

float fp4_dot_host(std::span<const uint8_t> packed, std::span<const float> scales,
                   int block_size, std::span<const float> acts) {
    size_t n = std::min(acts.size(), packed.size() * 2);
    float acc = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        size_t byte_idx = i / 2;
        uint8_t byte = packed.data()[byte_idx];
        uint8_t nibble = (i & 1) ? (byte >> 4) : (byte & 0x0f);
        int val = static_cast<int>(nibble & 0x0f);
        if (val >= 8) val -= 16;
        size_t block = block_size > 0 ? i / static_cast<size_t>(block_size) : 0;
        float scale = block < scales.size() ? scales.data()[block] : 1.0f;
        acc += (static_cast<float>(val) * scale / 7.0f) * acts.data()[i];
    }
    return acc;
}

}  // namespace

constexpr int BLOCK = 256;

__device__ float device_fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1u;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float val = static_cast<float>(mant) / 1024.0f;
        return sign ? -val * (1.0f / 16384.0f) : val * (1.0f / 16384.0f);
    }
    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
    uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    return __int_as_float(static_cast<int>(bits));
}

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

__global__ void fp16_gemv_kernel(const uint16_t* __restrict__ weights,
                                 const float* __restrict__ acts,
                                 float* __restrict__ out,
                                 int n) {
    __shared__ float partial[BLOCK];
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
        acc += device_fp16_to_float(weights[i]) * acts[i];
    partial[threadIdx.x] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < blockDim.x; ++i) sum += partial[i];
        atomicAdd(out, sum);
    }
}

__global__ void fp4_gemv_kernel(const uint8_t* __restrict__ packed,
                                const float* __restrict__ scales,
                                const float* __restrict__ acts,
                                float* __restrict__ out,
                                int n,
                                int block_size) {
    __shared__ float partial[BLOCK];
    float acc = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        int byte_idx = i / 2;
        uint8_t byte = packed[byte_idx];
        uint8_t nibble = (i & 1) ? (byte >> 4) : (byte & 0x0f);
        int val = static_cast<int>(nibble & 0x0f);
        if (val >= 8) val -= 16;
        int block = block_size > 0 ? i / block_size : 0;
        float scale = scales[block];
        acc += (static_cast<float>(val) * scale / 7.0f) * acts[i];
    }
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
    uint16_t* d_fp16_ = nullptr;
    uint8_t* d_fp4_ = nullptr;
    float* d_scales_ = nullptr;
    float* d_acts_ = nullptr;
    float* d_out_ = nullptr;
    size_t cap_ = 0;
    size_t fp4_cap_ = 0;
    size_t scale_cap_ = 0;

    void ensure_int8_capacity(size_t n) {
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

    void ensure_fp16_capacity(size_t n) {
        if (n <= cap_ && d_fp16_) return;
        uint16_t* nw = nullptr;
        float* na = nullptr;
        if (cudaMalloc(&nw, n * sizeof(uint16_t)) != cudaSuccess) return;
        if (!d_acts_ && cudaMalloc(&na, n * sizeof(float)) != cudaSuccess) {
            cudaFree(nw);
            return;
        }
        if (!d_acts_) {
            d_acts_ = na;
        }
        if (d_fp16_) cudaFree(d_fp16_);
        d_fp16_ = nw;
        cap_ = std::max(cap_, n);
        if (!d_out_ && cudaMalloc(&d_out_, sizeof(float)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc out");
    }

    void ensure_fp4_capacity(size_t packed_len, size_t scale_len, size_t act_len) {
        if (packed_len > fp4_cap_) {
            uint8_t* np = nullptr;
            if (cudaMalloc(&np, packed_len) != cudaSuccess) return;
            if (d_fp4_) cudaFree(d_fp4_);
            d_fp4_ = np;
            fp4_cap_ = packed_len;
        }
        if (scale_len > scale_cap_) {
            float* ns = nullptr;
            if (cudaMalloc(&ns, scale_len * sizeof(float)) != cudaSuccess) return;
            if (d_scales_) cudaFree(d_scales_);
            d_scales_ = ns;
            scale_cap_ = scale_len;
        }
        if (act_len > cap_) {
            float* na = nullptr;
            if (cudaMalloc(&na, act_len * sizeof(float)) != cudaSuccess) return;
            if (d_acts_) cudaFree(d_acts_);
            d_acts_ = na;
            cap_ = act_len;
        }
        if (!d_out_ && cudaMalloc(&d_out_, sizeof(float)) != cudaSuccess)
            throw std::runtime_error("cudaMalloc out");
    }

    static void check(cudaError_t err, const char* msg) {
        if (err != cudaSuccess)
            throw std::runtime_error(msg);
    }

    float gemv_int8(std::span<const int8_t> weights, std::span<const float> scales,
                    std::span<const float> acts) override {
        if (!scales.empty() || (pool_view_.nanoq && !pool_view_.nanoq->scales.empty())) {
            const auto& s = !scales.empty() ? scales : std::span<const float>(pool_view_.nanoq->scales);
            float acc = 0.0f;
            size_t n = std::min(weights.size(), acts.size());
            for (size_t i = 0; i < n; ++i) {
                float sc = s[std::min(i, s.size() - 1)];
                acc += static_cast<float>(weights[i]) * sc * acts[i];
            }
            return acc;
        }
        size_t n = std::min(weights.size(), acts.size());
        ensure_int8_capacity(n);
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

public:
    ~CUDABackend() override {
        if (d_weights_) cudaFree(d_weights_);
        if (d_fp16_) cudaFree(d_fp16_);
        if (d_fp4_) cudaFree(d_fp4_);
        if (d_scales_) cudaFree(d_scales_);
        if (d_acts_) cudaFree(d_acts_);
        if (d_out_) cudaFree(d_out_);
    }

    float gemv_fp16(std::span<const uint16_t> weights, std::span<const float> acts) override {
        size_t n = std::min(weights.size(), acts.size());
        try {
            ensure_fp16_capacity(n);
            if (!d_fp16_ || !d_acts_) return fp16_dot_host(weights, acts);
            check(cudaMemcpy(d_fp16_, weights.data(), n * sizeof(uint16_t), cudaMemcpyHostToDevice), "H2D fp16");
            check(cudaMemcpy(d_acts_, acts.data(), n * sizeof(float), cudaMemcpyHostToDevice), "H2D acts");
            float zero = 0.0f;
            check(cudaMemcpy(d_out_, &zero, sizeof(float), cudaMemcpyHostToDevice), "H2D out");
            int threads = BLOCK;
            int blocks = static_cast<int>((n + threads - 1) / threads);
            if (blocks < 1) blocks = 1;
            fp16_gemv_kernel<<<blocks, threads>>>(d_fp16_, d_acts_, d_out_, static_cast<int>(n));
            check(cudaGetLastError(), "fp16 kernel launch");
            check(cudaDeviceSynchronize(), "fp16 kernel sync");
            float result = 0.0f;
            check(cudaMemcpy(&result, d_out_, sizeof(float), cudaMemcpyDeviceToHost), "D2H out");
            return result;
        } catch (...) {
            return fp16_dot_host(weights, acts);
        }
    }

    float gemv_fp4(std::span<const uint8_t> packed, std::span<const float> scales,
                   int block_size, std::span<const float> acts) override {
        size_t n = std::min(acts.size(), packed.size() * 2);
        try {
            ensure_fp4_capacity(packed.size(), scales.size(), n);
            if (!d_fp4_ || !d_scales_ || !d_acts_) {
                return fp4_dot_host(packed, scales, block_size, acts);
            }
            check(cudaMemcpy(d_fp4_, packed.data(), packed.size(), cudaMemcpyHostToDevice), "H2D fp4");
            check(cudaMemcpy(d_scales_, scales.data(), scales.size() * sizeof(float), cudaMemcpyHostToDevice),
                  "H2D scales");
            check(cudaMemcpy(d_acts_, acts.data(), n * sizeof(float), cudaMemcpyHostToDevice), "H2D acts");
            float zero = 0.0f;
            check(cudaMemcpy(d_out_, &zero, sizeof(float), cudaMemcpyHostToDevice), "H2D out");
            int threads = BLOCK;
            int blocks = static_cast<int>((n + threads - 1) / threads);
            if (blocks < 1) blocks = 1;
            fp4_gemv_kernel<<<blocks, threads>>>(
                d_fp4_, d_scales_, d_acts_, d_out_, static_cast<int>(n), block_size);
            check(cudaGetLastError(), "fp4 kernel launch");
            check(cudaDeviceSynchronize(), "fp4 kernel sync");
            float result = 0.0f;
            check(cudaMemcpy(&result, d_out_, sizeof(float), cudaMemcpyDeviceToHost), "D2H out");
            return result;
        } catch (...) {
            return fp4_dot_host(packed, scales, block_size, acts);
        }
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
