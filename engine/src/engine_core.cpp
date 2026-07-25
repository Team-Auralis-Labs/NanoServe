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

EngineHandle* engine_create(EngineBackendKind kind) {
    auto backend = create_backend(kind);
    if (!backend) return nullptr;

    auto h = std::make_unique<EngineHandle>();
    h->weights_pool = pool_create(64 * 1024);
    h->scratch_pool = pool_create(16 * 1024 * 1024);
    if (!h->weights_pool || !h->scratch_pool) {
        if (h->weights_pool) pool_release(h->weights_pool);
        if (h->scratch_pool) pool_release(h->scratch_pool);
        return nullptr;
    }

    h->weights_len = 1024;
    h->weights = static_cast<int8_t*>(pool_allocate(h->weights_pool, h->weights_len));
    if (!h->weights) {
        pool_release(h->weights_pool);
        pool_release(h->scratch_pool);
        return nullptr;
    }

    h->backend_kind = kind;
    h->backend = std::move(backend);

    std::mt19937 rng(42);
    for (size_t i = 0; i < h->weights_len; ++i)
        h->weights[i] = static_cast<int8_t>((rng() % 200) - 100);

    PoolBufferView view{};
    view.weights = h->weights;
    view.weights_pool = h->weights_pool;
    view.scratch_pool = h->scratch_pool;
    view.length = h->weights_len;
    h->backend->bind_pool_buffers(view);

    return h.release();
}

int engine_run_infer(EngineHandle* h, const char* prompt, int max_tokens, char* out_buf, int out_buf_len) {
    if (!h || !h->backend || !prompt || !out_buf || out_buf_len <= 0) {
        if (out_buf && out_buf_len > 0) out_buf[0] = '\0';
        return 0;
    }

    size_t buf_size = 1024 * sizeof(float);
    float* acts = static_cast<float*>(pool_allocate(h->scratch_pool, buf_size));
    if (!acts) { out_buf[0] = '\0'; return 0; }

    size_t plen = std::strlen(prompt);
    for (int i = 0; i < 1024; ++i) acts[i] = static_cast<float>((plen + i) % 17) * 0.1f;

    PoolBufferView view{};
    view.weights = h->weights;
    view.activations = acts;
    view.length = h->weights_len;
    view.weights_pool = h->weights_pool;
    view.scratch_pool = h->scratch_pool;
    h->backend->bind_pool_buffers(view);

    std::string out;
    out.reserve(static_cast<size_t>(max_tokens) * 16);
    unsigned seed = 2166136261u;
    for (char c : std::string_view{prompt}) seed = (seed ^ static_cast<unsigned>(c)) * 16777619u;

    std::span<const int8_t> weights_view(h->weights, h->weights_len);
    for (int t = 0; t < max_tokens; ++t) {
        float score = h->backend->gemv_int8(weights_view, std::span<const float>(acts, 1024));
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
    if (h->weights_pool && h->weights)
        pool_free(h->weights_pool, h->weights, h->weights_len);
    if (h->weights_pool) pool_release(h->weights_pool);
    if (h->scratch_pool) pool_release(h->scratch_pool);
    h->backend.reset();
    delete h;
}
