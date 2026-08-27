#pragma once
#include "nanoq_loader.hpp"
#include <cstdint>
#include <memory>

#if !defined(__CUDACC__) && defined(__has_include) && __has_include(<span>)
#include <span>
#else
#include <cstddef>
namespace std {
template <typename T>
class span {
public:
    constexpr span() noexcept : data_(nullptr), size_(0) {}
    constexpr span(const T* data, std::size_t count) noexcept : data_(data), size_(count) {}
    constexpr const T* data() const noexcept { return data_; }
    constexpr std::size_t size() const noexcept { return size_; }

private:
    const T* data_;
    std::size_t size_;
};
}  // namespace std
#endif

enum class EngineBackendKind : int {
    Cpu = 0,
    Cuda = 1,
    OpenCl = 2,
};

struct PoolBufferView {
    int8_t* weights = nullptr;
    float* activations = nullptr;
    size_t length = 0;
    void* weights_pool = nullptr;
    void* scratch_pool = nullptr;
    const NanoqModel* nanoq = nullptr;
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    virtual float gemv_int8(std::span<const int8_t> weights, std::span<const float> scales,
                            std::span<const float> acts) = 0;
    virtual float gemv_fp16(std::span<const uint16_t> weights, std::span<const float> acts) = 0;
    virtual float gemv_fp4(std::span<const uint8_t> packed, std::span<const float> scales,
                           int block_size, std::span<const float> acts) = 0;
    virtual void gemm_int8(std::span<const int8_t> weights, std::span<const float> scales,
                           std::span<const float> input, std::span<float> output,
                           int rows, int cols, int out_dim) {
        (void)weights; (void)scales; (void)input; (void)output;
        (void)rows; (void)cols; (void)out_dim;
    }
    virtual void bind_pool_buffers(const PoolBufferView& view) { pool_view_ = view; }
    virtual const char* name() const = 0;

protected:
    PoolBufferView pool_view_{};
};

std::unique_ptr<ComputeBackend> create_backend(EngineBackendKind kind);
int probe_cuda();
int probe_opencl();
const char* backend_name(EngineBackendKind kind);
