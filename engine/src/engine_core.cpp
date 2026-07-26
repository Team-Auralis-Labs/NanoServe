#include "engine_core.hpp"
#include <algorithm>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <string_view>

extern "C" {
    void* pool_create(size_t size);
    void* pool_allocate(void* pool, size_t req_size);
    void  pool_free(void* pool, void* ptr, size_t size);
    void  pool_release(void* pool);
}

static const char* VOCAB[] = {
    "the","model","is","fast","and","efficient","quantized","inference",
    "engine","serves","tokens","in","parallel","batches","using","a",
    "buddy","allocator","for","memory","reuse","without","fragmentation"
};
static constexpr int VOCAB_SIZE = sizeof(VOCAB) / sizeof(VOCAB[0]);

static void init_synthetic_weights(EngineHandle* h) {
    h->weights_len = 1024;
    h->weights = static_cast<int8_t*>(pool_allocate(h->weights_pool, h->weights_len));
    if (!h->weights) return;
    std::mt19937 rng(42);
    for (size_t i = 0; i < h->weights_len; ++i)
        h->weights[i] = static_cast<int8_t>((rng() % 200) - 100);
    h->has_model = false;
    h->model_info_json = "{\"dtype\":\"synthetic\",\"rows\":1,\"cols\":1024}";
}

static bool apply_nanoq_model(EngineHandle* h, const char* path, std::string& err) {
    NanoqModel m;
    if (!nanoq_load_file(path, m, err)) return false;
    h->model = std::move(m);
    h->has_model = true;
    h->model_path = path ? path : "";
    h->weights_len = h->model.flat_len();
    if (h->weights_len == 0) {
        err = "empty model weights";
        return false;
    }
    h->model_info_json = nanoq_model_info_json(h->model);
    return true;
}

static float run_gemv(EngineHandle* h, std::span<const float> acts) {
    if (!h || !h->backend) return 0.0f;
    const size_t n = std::min(h->weights_len, acts.size());

    if (h->has_model) {
        switch (h->model.dtype) {
            case NanoqDtype::Int8:
                return h->backend->gemv_int8(
                    std::span<const int8_t>(h->model.int8_weights.data(), n), acts);
            case NanoqDtype::Fp16:
                return h->backend->gemv_fp16(
                    std::span<const uint16_t>(h->model.fp16_weights.data(), n), acts);
            case NanoqDtype::Fp4:
                return h->backend->gemv_fp4(
                    h->model.fp4_packed, h->model.scales, h->model.block_size, acts);
            default:
                break;
        }
    }

    if (h->weights)
        return h->backend->gemv_int8(std::span<const int8_t>(h->weights, n), acts);
    return 0.0f;
}

static EngineHandle* engine_create_impl(EngineBackendKind kind, const char* nanoq_path) {
    auto backend = create_backend(kind);
    if (!backend) return nullptr;

    auto h = std::make_unique<EngineHandle>();
    h->weights_pool = pool_create(64 * 1024 * 1024);
    h->scratch_pool = pool_create(16 * 1024 * 1024);
    if (!h->weights_pool || !h->scratch_pool) {
        if (h->weights_pool) pool_release(h->weights_pool);
        if (h->scratch_pool) pool_release(h->scratch_pool);
        return nullptr;
    }

    h->backend_kind = kind;
    h->backend = std::move(backend);

    std::string err;
    if (nanoq_path && nanoq_path[0]) {
        if (!apply_nanoq_model(h.get(), nanoq_path, err)) {
            pool_release(h->weights_pool);
            pool_release(h->scratch_pool);
            return nullptr;
        }
    } else {
        init_synthetic_weights(h.get());
    }

    PoolBufferView view{};
    view.weights = h->weights;
    view.weights_pool = h->weights_pool;
    view.scratch_pool = h->scratch_pool;
    view.length = h->weights_len;
    view.nanoq = h->has_model ? &h->model : nullptr;
    h->backend->bind_pool_buffers(view);

    return h.release();
}

EngineHandle* engine_create(EngineBackendKind kind) {
    return engine_create_impl(kind, nullptr);
}

EngineHandle* engine_create_with_model(EngineBackendKind kind, const char* nanoq_path) {
    return engine_create_impl(kind, nanoq_path);
}

int engine_reload_model(EngineHandle* h, const char* nanoq_path) {
    if (!h || !nanoq_path || !nanoq_path[0]) return -1;
    std::string err;
    if (!apply_nanoq_model(h, nanoq_path, err)) return -1;

    PoolBufferView view{};
    view.weights = h->weights;
    view.weights_pool = h->weights_pool;
    view.scratch_pool = h->scratch_pool;
    view.length = h->weights_len;
    view.nanoq = &h->model;
    h->backend->bind_pool_buffers(view);
    return 0;
}

const char* engine_get_model_info(EngineHandle* h) {
    if (!h) return "{}";
    return h->model_info_json.c_str();
}

int engine_run_infer(EngineHandle* h, const char* prompt, int max_tokens, char* out_buf, int out_buf_len) {
    if (!h || !h->backend || !prompt || !out_buf || out_buf_len <= 0) {
        if (out_buf && out_buf_len > 0) out_buf[0] = '\0';
        return 0;
    }

    constexpr size_t act_dim = 1024;
    size_t buf_size = act_dim * sizeof(float);
    float* acts = static_cast<float*>(pool_allocate(h->scratch_pool, buf_size));
    if (!acts) { out_buf[0] = '\0'; return 0; }

    size_t plen = std::strlen(prompt);
    for (size_t i = 0; i < act_dim; ++i)
        acts[i] = static_cast<float>((plen + i) % 17) * 0.1f;

    PoolBufferView view{};
    view.weights = h->weights;
    view.activations = acts;
    view.length = h->weights_len;
    view.weights_pool = h->weights_pool;
    view.scratch_pool = h->scratch_pool;
    view.nanoq = h->has_model ? &h->model : nullptr;
    h->backend->bind_pool_buffers(view);

    std::string out;
    out.reserve(static_cast<size_t>(max_tokens) * 16);
    unsigned seed = 2166136261u;
    for (char c : std::string_view{prompt}) seed = (seed ^ static_cast<unsigned>(c)) * 16777619u;

    std::span<const float> act_span(acts, act_dim);
    for (int t = 0; t < max_tokens; ++t) {
        float score = run_gemv(h, act_span);
        seed = seed * 1664525u + 1013904223u + static_cast<unsigned>(score);
        int idx = static_cast<int>(seed % VOCAB_SIZE);
        out += VOCAB[idx];
        if (t + 1 < max_tokens) out += " ";
    }

    pool_free(h->scratch_pool, acts, buf_size);
    int n = std::min(static_cast<int>(out.size()), out_buf_len - 1);
    std::memcpy(out_buf, out.data(), n);
    out_buf[n] = '\0';
    return max_tokens;
}

void engine_destroy_handle(EngineHandle* h) {
    if (!h) return;
    if (h->weights_pool && h->weights && !h->has_model)
        pool_free(h->weights_pool, h->weights, h->weights_len);
    if (h->weights_pool) pool_release(h->weights_pool);
    if (h->scratch_pool) pool_release(h->scratch_pool);
    h->backend.reset();
    delete h;
}
