#pragma once
#include <cstdint>
#include <memory>

#if defined(__has_include) && __has_include(<span>)
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

// Host-side weight/activation views backed by the Rust buddy allocator pools.
struct PoolBufferView {
    int8_t* weights = nullptr;
    float* activations = nullptr;
    size_t length = 0;
    void* weights_pool = nullptr;
    void* scratch_pool = nullptr;
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    // INT8 GEMV: score = sum(weights[i] * activations[i]) — matmul row-vector form.
    virtual float gemv_int8(std::span<const int8_t> weights, std::span<const float> acts) = 0;
    virtual void bind_pool_buffers(const PoolBufferView& view) { pool_view_ = view; }
    virtual const char* name() const = 0;

protected:
    PoolBufferView pool_view_{};
};

std::unique_ptr<ComputeBackend> create_backend(EngineBackendKind kind);
int probe_cuda();
int probe_opencl();
const char* backend_name(EngineBackendKind kind);
