#pragma once
#include "nanoq_archive.hpp"
#include "backend.hpp"
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

struct KVCache {
    int n_layers = 0;
    int n_heads = 0;
    int head_dim = 0;
    int max_seq = 0;
    void* pool = nullptr;
    std::vector<float> k_cache;
    std::vector<float> v_cache;
    int seq_len = 0;

    bool init(int layers, int heads, int hidden, int max_seq_len, void* buddy_pool);
    void reset();
    float* k_layer(int layer);
    float* v_layer(int layer);
};

struct SamplerConfig {
    float temperature = 0.8f;
    int top_k = 40;
    float top_p = 0.9f;
    bool greedy = false;
};

class Sampler {
public:
    explicit Sampler(SamplerConfig cfg = {});
    int sample(std::span<const float> logits, unsigned& seed) const;
    void set_config(const SamplerConfig& cfg) { cfg_ = cfg; }

private:
    SamplerConfig cfg_;
    static constexpr int kEosToken = 50256;
};

class TransformerModel {
public:
    explicit TransformerModel(const NanoqArchiveV3* archive, ComputeBackend* backend);

    bool init(std::string& err);
    void reset_kv();
    bool forward_token(const std::vector<uint32_t>& tokens, int pos, std::vector<float>& logits);
    bool generate(const std::vector<uint32_t>& prompt_ids, int max_new_tokens,
                  std::vector<uint32_t>& out_ids, unsigned seed = 42);

    const NanoqConfig& config() const { return archive_->config; }
    KVCache& kv() { return kv_; }

private:
    const NanoqArchiveV3* archive_;
    ComputeBackend* backend_;
    KVCache kv_;
    Sampler sampler_;
    int max_pos_ = 1024;
    std::vector<float> hidden_;
    std::vector<float> scratch_;

    void dequant_matvec(const NanoqTensorEntry& w, std::span<const float> x, std::span<float> y,
                        const NanoqTensorEntry* bias = nullptr);
    void matmul_transposed(const NanoqTensorEntry& w, std::span<const float> x, std::span<float> y,
                           const NanoqTensorEntry* bias = nullptr);
    void layer_norm(std::span<const float> x, const NanoqTensorEntry* w, const NanoqTensorEntry* b,
                    std::span<float> out, int dim);
    void gelu(std::span<float> x);
    void softmax(std::span<float> x);
    void causal_attention(int layer, std::span<float> x, int pos);
    void mlp_block(int layer, std::span<float> x);
    void transformer_block(int layer, std::span<float> x, int pos);
};
