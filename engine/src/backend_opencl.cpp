#include "backend.hpp"
#include <CL/cl.h>
#include <algorithm>
#include <cstring>
#include <span>
#include <vector>

static const char* OPENCL_KERNEL = R"(
__kernel void int8_gemv(__global const char* weights, __global const float* acts, __global float* out, int n) {
    float acc = 0.0f;
    int gid = get_global_id(0);
    int stride = get_global_size(0);
    for (int i = gid; i < n; i += stride)
        acc += convert_float(weights[i]) * acts[i];
    atomic_add(out, acc);
}
)";

class OpenCLBackend : public ComputeBackend {
    cl_context ctx_ = nullptr;
    cl_command_queue queue_ = nullptr;
    cl_program prog_ = nullptr;
    cl_kernel kernel_ = nullptr;
    cl_mem buf_w_ = nullptr;
    cl_mem buf_a_ = nullptr;
    cl_mem buf_o_ = nullptr;
    size_t cap_ = 0;

    void ensure_buffers(size_t n) {
        if (n <= cap_) return;
        if (buf_w_) clReleaseMemObject(buf_w_);
        if (buf_a_) clReleaseMemObject(buf_a_);
        cap_ = n;
        buf_w_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY, cap_ * sizeof(int8_t), nullptr, nullptr);
        buf_a_ = clCreateBuffer(ctx_, CL_MEM_READ_ONLY, cap_ * sizeof(float), nullptr, nullptr);
    }

public:
    OpenCLBackend(cl_context ctx, cl_command_queue queue, cl_program prog, cl_kernel kernel)
        : ctx_(ctx), queue_(queue), prog_(prog), kernel_(kernel) {
        buf_o_ = clCreateBuffer(ctx_, CL_MEM_READ_WRITE, sizeof(float), nullptr, nullptr);
    }

    ~OpenCLBackend() override {
        if (buf_w_) clReleaseMemObject(buf_w_);
        if (buf_a_) clReleaseMemObject(buf_a_);
        if (buf_o_) clReleaseMemObject(buf_o_);
        if (kernel_) clReleaseKernel(kernel_);
        if (prog_) clReleaseProgram(prog_);
        if (queue_) clReleaseCommandQueue(queue_);
        if (ctx_) clReleaseContext(ctx_);
    }

    float gemv_int8(std::span<const int8_t> weights, std::span<const float> acts) override {
        size_t n = std::min(weights.size(), acts.size());
        ensure_buffers(n);

        const int8_t* h_w = pool_view_.weights ? pool_view_.weights : weights.data();
        const float* h_a = pool_view_.activations ? pool_view_.activations : acts.data();

        clEnqueueWriteBuffer(queue_, buf_w_, CL_TRUE, 0, n * sizeof(int8_t), h_w, 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue_, buf_a_, CL_TRUE, 0, n * sizeof(float), h_a, 0, nullptr, nullptr);
        float zero = 0.0f;
        clEnqueueWriteBuffer(queue_, buf_o_, CL_TRUE, 0, sizeof(float), &zero, 0, nullptr, nullptr);
        clSetKernelArg(kernel_, 0, sizeof(cl_mem), &buf_w_);
        clSetKernelArg(kernel_, 1, sizeof(cl_mem), &buf_a_);
        clSetKernelArg(kernel_, 2, sizeof(cl_mem), &buf_o_);
        int ni = static_cast<int>(n);
        clSetKernelArg(kernel_, 3, sizeof(int), &ni);
        size_t g = 256;
        clEnqueueNDRangeKernel(queue_, kernel_, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
        clFinish(queue_);
        float result = 0.0f;
        clEnqueueReadBuffer(queue_, buf_o_, CL_TRUE, 0, sizeof(float), &result, 0, nullptr, nullptr);
        return result;
    }

    const char* name() const override { return "opencl"; }
};

static bool init_opencl(cl_context& ctx, cl_command_queue& queue, cl_program& prog, cl_kernel& kernel) {
    cl_uint nplatforms = 0;
    if (clGetPlatformIDs(0, nullptr, &nplatforms) != CL_SUCCESS || nplatforms == 0) return false;
    std::vector<cl_platform_id> platforms(nplatforms);
    clGetPlatformIDs(nplatforms, platforms.data(), nullptr);
    cl_device_id device = nullptr;
    for (auto plat : platforms) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 1, &device, &ndev) == CL_SUCCESS && ndev > 0)
            break;
        device = nullptr;
    }
    if (!device) return false;
    cl_int err = 0;
    ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return false;
    queue = clCreateCommandQueue(ctx, device, 0, &err);
    if (err != CL_SUCCESS) return false;
    const char* src = OPENCL_KERNEL;
    size_t len = std::strlen(src);
    prog = clCreateProgramWithSource(ctx, 1, &src, &len, &err);
    if (err != CL_SUCCESS) return false;
    err = clBuildProgram(prog, 1, &device, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) return false;
    kernel = clCreateKernel(prog, "int8_gemv", &err);
    return err == CL_SUCCESS;
}

std::unique_ptr<ComputeBackend> create_opencl_backend() {
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
    cl_program prog = nullptr;
    cl_kernel kernel = nullptr;
    if (!init_opencl(ctx, queue, prog, kernel)) return nullptr;
    return std::make_unique<OpenCLBackend>(ctx, queue, prog, kernel);
}

int probe_opencl_impl() {
    cl_uint nplatforms = 0;
    if (clGetPlatformIDs(0, nullptr, &nplatforms) != CL_SUCCESS || nplatforms == 0) return 0;
    std::vector<cl_platform_id> platforms(nplatforms);
    clGetPlatformIDs(nplatforms, platforms.data(), nullptr);
    for (auto plat : platforms) {
        cl_uint ndev = 0;
        if (clGetDeviceIDs(plat, CL_DEVICE_TYPE_GPU, 0, nullptr, &ndev) == CL_SUCCESS && ndev > 0)
            return 1;
    }
    return 0;
}
