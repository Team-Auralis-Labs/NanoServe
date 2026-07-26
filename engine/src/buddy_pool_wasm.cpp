// Minimal buddy-pool C ABI for Emscripten builds (no Rust link required).
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

struct WasmPool {
    std::vector<uint8_t> arena;
    std::mutex mu;
};

static std::unordered_map<void*, WasmPool*> g_pools;

extern "C" {

void* pool_create(size_t size) {
    auto* p = new WasmPool();
    p->arena.resize(size > 0 ? size : 64 * 1024);
    g_pools[p] = p;
    return p;
}

void* pool_allocate(void* pool, size_t req_size) {
    if (!pool || req_size == 0) return nullptr;
    auto* p = static_cast<WasmPool*>(pool);
    std::lock_guard<std::mutex> lock(p->mu);
    if (req_size > p->arena.size()) return nullptr;
    return p->arena.data();
}

void pool_free(void*, void*, size_t) {}

void pool_release(void* pool) {
    if (!pool) return;
    g_pools.erase(pool);
    delete static_cast<WasmPool*>(pool);
}

void pool_destroy(void* pool) { pool_release(pool); }

}
