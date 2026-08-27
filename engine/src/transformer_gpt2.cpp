#include "transformer.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

extern "C" {
void* pool_allocate(void* pool, size_t req_size);
void pool_free(void* pool, void* ptr, size_t size);
}

bool KVCache::init(int layers, int heads, int hidden, int max_seq_len, void* buddy_pool) {
    n_layers = layers;
    n_heads = heads;
    head_dim = hidden / heads;
    max_seq = max_seq_len;
    pool = buddy_pool;
    const size_t per_layer = static_cast<size_t>(max_seq) * static_cast<size_t>(heads) *
                             static_cast<size_t>(head_dim);
    k_cache.assign(static_cast<size_t>(layers) * per_layer, 0.0f);
    v_cache.assign(static_cast<size_t>(layers) * per_layer, 0.0f);
    seq_len = 0;
    return true;
}

void KVCache::reset() { seq_len = 0; }

float* KVCache::k_layer(int layer) {
    const size_t per_layer = static_cast<size_t>(max_seq) * n_heads * head_dim;
    return k_cache.data() + static_cast<size_t>(layer) * per_layer;
}

float* KVCache::v_layer(int layer) {
    const size_t per_layer = static_cast<size_t>(max_seq) * n_heads * head_dim;
    return v_cache.data() + static_cast<size_t>(layer) * per_layer;
}

Sampler::Sampler(SamplerConfig cfg) : cfg_(cfg) {}

int Sampler::sample(std::span<const float> logits, unsigned& seed) const {
    if (logits.empty()) return 0;
    if (cfg_.greedy || cfg_.temperature <= 0.0f) {
        return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
    }
    std::vector<float> probs(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((logits[i] - max_logit) / cfg_.temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f) return 0;
    seed = seed * 1664525u + 1013904223u;
    float r = static_cast<float>(seed % 1000000) / 1000000.0f * sum;
    float acc = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        acc += probs[i];
        if (acc >= r) return static_cast<int>(i);
    }
    return static_cast<int>(probs.size() - 1);
}

TransformerModel::TransformerModel(const NanoqArchiveV3* archive, ComputeBackend* backend)
    : archive_(archive), backend_(backend) {}

bool TransformerModel::init(std::string& err) {
    if (!archive_ || !archive_->valid) {
        err = "invalid archive";
        return false;
    }
    const auto& cfg = archive_->config;
    max_pos_ = cfg.max_seq_len;
    const auto* wpe = archive_->find_tensor("transformer.wpe.weight");
    if (wpe && !wpe->shape.empty())
        max_pos_ = std::min(max_pos_, static_cast<int>(wpe->shape[0]));
    if (max_pos_ <= 0) {
        err = "invalid max sequence length";
        return false;
    }
    if (!kv_.init(cfg.n_layers, cfg.n_heads, cfg.hidden_size, max_pos_, nullptr)) {
        err = "kv init failed";
        return false;
    }
    sampler_.set_config(SamplerConfig{.temperature = 0.0f, .greedy = true});
    hidden_.assign(static_cast<size_t>(cfg.hidden_size), 0.0f);
    scratch_.assign(static_cast<size_t>(cfg.hidden_size * 4), 0.0f);
    return true;
}

void TransformerModel::reset_kv() { kv_.reset(); }

void TransformerModel::dequant_matvec(const NanoqTensorEntry& w, std::span<const float> x,
                                      std::span<float> y, const NanoqTensorEntry* bias) {
    if (w.shape.size() < 2) return;
    const int out_dim = static_cast<int>(w.shape[0]);
    const int in_dim = static_cast<int>(w.shape[1]);
    if (static_cast<int>(y.size()) < out_dim) return;
    const uint8_t* raw = archive_->tensor_bytes(w);
    size_t scale_count = 0;
    const float* scales = archive_->tensor_scales(w, scale_count);

    std::fill(y.begin(), y.begin() + out_dim, 0.0f);
    if (w.dtype == NanoqTensorDtype::Int8 && scales && scale_count > 0) {
        for (int o = 0; o < out_dim; ++o) {
            float acc = 0.0f;
            const float scale = scales[std::min(static_cast<size_t>(o), scale_count - 1)];
            const int8_t* row = reinterpret_cast<const int8_t*>(raw) + o * in_dim;
            for (int i = 0; i < in_dim && i < static_cast<int>(x.size()); ++i)
                acc += static_cast<float>(row[i]) * scale * x[static_cast<size_t>(i)];
            y[static_cast<size_t>(o)] = acc;
        }
    } else if (w.dtype == NanoqTensorDtype::Fp32) {
        const float* weights = reinterpret_cast<const float*>(raw);
        for (int o = 0; o < out_dim; ++o) {
            float acc = 0.0f;
            for (int i = 0; i < in_dim && i < static_cast<int>(x.size()); ++i)
                acc += weights[o * in_dim + i] * x[static_cast<size_t>(i)];
            y[static_cast<size_t>(o)] = acc;
        }
    }
    if (bias && bias->dtype == NanoqTensorDtype::Fp32) {
        const float* b = reinterpret_cast<const float*>(archive_->tensor_bytes(*bias));
        for (int o = 0; o < out_dim; ++o) y[static_cast<size_t>(o)] += b[o];
    }
}

void TransformerModel::matmul_transposed(const NanoqTensorEntry& w, std::span<const float> x,
                                         std::span<float> y, const NanoqTensorEntry* bias) {
    dequant_matvec(w, x, y, bias);
}

void TransformerModel::layer_norm(std::span<const float> x, const NanoqTensorEntry* w,
                                  const NanoqTensorEntry* b, std::span<float> out, int dim) {
    float mean = 0.0f, var = 0.0f;
    for (int i = 0; i < dim; ++i) mean += x[static_cast<size_t>(i)];
    mean /= dim;
    for (int i = 0; i < dim; ++i) {
        float d = x[static_cast<size_t>(i)] - mean;
        var += d * d;
    }
    var /= dim;
    float inv_std = 1.0f / std::sqrt(var + archive_->config.norm_eps);
    const float* gw = w ? reinterpret_cast<const float*>(archive_->tensor_bytes(*w)) : nullptr;
    const float* gb = b ? reinterpret_cast<const float*>(archive_->tensor_bytes(*b)) : nullptr;
    for (int i = 0; i < dim; ++i) {
        float v = (x[static_cast<size_t>(i)] - mean) * inv_std;
        if (gw) v *= gw[i];
        if (gb) v += gb[i];
        out[static_cast<size_t>(i)] = v;
    }
}

void TransformerModel::gelu(std::span<float> x) {
    for (float& v : x) {
        v = 0.5f * v * (1.0f + std::tanh(0.7978845608f * (v + 0.044715f * v * v * v)));
    }
}

void TransformerModel::softmax(std::span<float> x) {
    float max_v = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (float& v : x) {
        v = std::exp(v - max_v);
        sum += v;
    }
    if (sum > 0.0f)
        for (float& v : x) v /= sum;
}

void TransformerModel::causal_attention(int layer, std::span<float> x, int pos) {
    const int H = archive_->config.hidden_size;
    const int n_heads = archive_->config.n_heads;
    const int head_dim = H / n_heads;

    char name_buf[128];
    std::snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.attn.c_attn.weight", layer);
    const auto* c_attn = archive_->find_tensor(name_buf);
    std::snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.attn.c_attn.bias", layer);
    const auto* c_attn_b = archive_->find_tensor(name_buf);
    std::snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.attn.c_proj.weight", layer);
    const auto* c_proj = archive_->find_tensor(name_buf);
    std::snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.attn.c_proj.bias", layer);
    const auto* c_proj_b = archive_->find_tensor(name_buf);
    if (!c_attn || !c_proj) return;

    std::vector<float> qkv(H * 3);
    matmul_transposed(*c_attn, x, qkv, c_attn_b);

    const float* Q = qkv.data();
    const float* K = qkv.data() + H;
    const float* V = qkv.data() + 2 * H;

    float* k_store = kv_.k_layer(layer) + static_cast<size_t>(pos) * n_heads * head_dim;
    float* v_store = kv_.v_layer(layer) + static_cast<size_t>(pos) * n_heads * head_dim;

    std::vector<float> attn_out(static_cast<size_t>(H), 0.0f);
    for (int h = 0; h < n_heads; ++h) {
        const float* q = Q + h * head_dim;
        std::memcpy(k_store + h * head_dim, K + h * head_dim, head_dim * sizeof(float));
        std::memcpy(v_store + h * head_dim, V + h * head_dim, head_dim * sizeof(float));

        std::vector<float> scores(static_cast<size_t>(pos + 1));
        for (int t = 0; t <= pos; ++t) {
            const float* k_t = kv_.k_layer(layer) + static_cast<size_t>(t) * n_heads * head_dim +
                               h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += q[d] * k_t[d];
            scores[static_cast<size_t>(t)] = dot / std::sqrt(static_cast<float>(head_dim));
        }
        softmax(scores);

        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (int t = 0; t <= pos; ++t) {
                const float* v_t = kv_.v_layer(layer) + static_cast<size_t>(t) * n_heads * head_dim +
                                   h * head_dim;
                acc += scores[static_cast<size_t>(t)] * v_t[d];
            }
            attn_out[static_cast<size_t>(h * head_dim + d)] = acc;
        }
    }

    std::vector<float> proj_out(static_cast<size_t>(H));
    matmul_transposed(*c_proj, attn_out, proj_out, c_proj_b);
    for (int i = 0; i < H; ++i) x[static_cast<size_t>(i)] += proj_out[static_cast<size_t>(i)];
}

void TransformerModel::mlp_block(int layer, std::span<float> x) {
    const int H = archive_->config.hidden_size;
    char name_buf[128];
    std::snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.mlp.c_fc.weight", layer);
    const auto* c_fc = archive_->find_tensor(name_buf);
    std::snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.mlp.c_fc.bias", layer);
    const auto* c_fc_b = archive_->find_tensor(name_buf);
    std::snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.mlp.c_proj.weight", layer);
    const auto* c_proj = archive_->find_tensor(name_buf);
    std::snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.mlp.c_proj.bias", layer);
    const auto* c_proj_b = archive_->find_tensor(name_buf);
    if (!c_fc || !c_proj) return;

    const int ff = static_cast<int>(c_fc->shape[0]);
    std::vector<float> ff_out(static_cast<size_t>(ff));
    matmul_transposed(*c_fc, x, ff_out, c_fc_b);
    gelu(ff_out);
    std::vector<float> mlp_out(static_cast<size_t>(H));
    matmul_transposed(*c_proj, ff_out, mlp_out, c_proj_b);
    for (int i = 0; i < H; ++i) x[static_cast<size_t>(i)] += mlp_out[static_cast<size_t>(i)];
}

void TransformerModel::transformer_block(int layer, std::span<float> x, int pos) {
    const int H = archive_->config.hidden_size;
    char name_buf[128];
    std::vector<float> normed(static_cast<size_t>(H));
    std::vector<float> branch(static_cast<size_t>(H));

    std::snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.ln_1.weight", layer);
    const auto* ln1w = archive_->find_tensor(name_buf);
    std::snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.ln_1.bias", layer);
    const auto* ln1b = archive_->find_tensor(name_buf);
    layer_norm(x, ln1w, ln1b, normed, H);
    branch = normed;
    causal_attention(layer, branch, pos);
    for (int i = 0; i < H; ++i) x[static_cast<size_t>(i)] += branch[static_cast<size_t>(i)];

    std::snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.ln_2.weight", layer);
    const auto* ln2w = archive_->find_tensor(name_buf);
    std::snprintf(name_buf, sizeof(name_buf), "transformer.h.%d.ln_2.bias", layer);
    const auto* ln2b = archive_->find_tensor(name_buf);
    layer_norm(x, ln2w, ln2b, normed, H);
    branch = normed;
    mlp_block(layer, branch);
    for (int i = 0; i < H; ++i) x[static_cast<size_t>(i)] += branch[static_cast<size_t>(i)];
}

bool TransformerModel::forward_token(const std::vector<uint32_t>& tokens, int pos,
                                     std::vector<float>& logits) {
    const int H = archive_->config.hidden_size;
    const int V = archive_->config.vocab_size;

    const auto* wte = archive_->find_tensor("transformer.wte.weight");
    const auto* wpe = archive_->find_tensor("transformer.wpe.weight");
    if (!wte || !wpe || pos >= static_cast<int>(tokens.size())) return false;
    if (pos < 0 || pos >= max_pos_) return false;

    uint32_t tok = tokens[static_cast<size_t>(pos)];
    if (tok >= static_cast<uint32_t>(wte->shape[0])) tok = 0;

    const float* emb = reinterpret_cast<const float*>(archive_->tensor_bytes(*wte)) +
                     static_cast<size_t>(tok) * H;
    const float* pos_emb =
        reinterpret_cast<const float*>(archive_->tensor_bytes(*wpe)) + static_cast<size_t>(pos) * H;

    hidden_.assign(static_cast<size_t>(H), 0.0f);
    for (int i = 0; i < H; ++i)
        hidden_[static_cast<size_t>(i)] = emb[i] + pos_emb[i];

    for (int layer = 0; layer < archive_->config.n_layers; ++layer)
        transformer_block(layer, hidden_, pos);

    const auto* ln_f_w = archive_->find_tensor("transformer.ln_f.weight");
    const auto* ln_f_b = archive_->find_tensor("transformer.ln_f.bias");
    std::vector<float> normed(static_cast<size_t>(H));
    layer_norm(hidden_, ln_f_w, ln_f_b, normed, H);

    const auto* lm_head = archive_->find_tensor("lm_head.weight");
    if (!lm_head) lm_head = wte;

    logits.assign(static_cast<size_t>(V), -1e9f);
    const int out_rows = static_cast<int>(lm_head->shape[0]);
    const int out_cols = static_cast<int>(lm_head->shape[1]);
    if (lm_head->dtype == NanoqTensorDtype::Fp32) {
        const float* w = reinterpret_cast<const float*>(archive_->tensor_bytes(*lm_head));
        for (int v = 0; v < std::min(V, out_rows); ++v) {
            float acc = 0.0f;
            for (int d = 0; d < std::min(H, out_cols); ++d)
                acc += w[static_cast<size_t>(v) * out_cols + d] * normed[static_cast<size_t>(d)];
            logits[static_cast<size_t>(v)] = acc;
        }
    } else {
        std::vector<float> row(static_cast<size_t>(out_rows));
        matmul_transposed(*lm_head, normed, row);
        for (int v = 0; v < std::min(V, out_rows); ++v)
            logits[static_cast<size_t>(v)] = row[static_cast<size_t>(v)];
    }
    kv_.seq_len = std::max(kv_.seq_len, pos + 1);
    return true;
}

bool TransformerModel::generate(const std::vector<uint32_t>& prompt_ids, int max_new_tokens,
                                std::vector<uint32_t>& out_ids, unsigned seed) {
    out_ids = prompt_ids;
    std::vector<float> logits;
    if (out_ids.empty()) return false;
    if (static_cast<int>(out_ids.size()) > max_pos_) return false;

    for (int pos = 0; pos < static_cast<int>(out_ids.size()); ++pos) {
        if (!forward_token(out_ids, pos, logits)) return false;
    }

    for (int step = 0; step < max_new_tokens; ++step) {
        int next = sampler_.sample(logits, seed);
        if (next == 50256) break;
        out_ids.push_back(static_cast<uint32_t>(next));
        int pos = static_cast<int>(out_ids.size()) - 1;
        if (pos >= max_pos_) break;
        if (!forward_token(out_ids, pos, logits)) return false;
    }
    return true;
}

// Stub for llama architecture (Phase 01 compile gate)
bool transformer_llama_stub() { return true; }
