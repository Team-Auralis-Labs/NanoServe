#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

constexpr uint32_t NANOQ_V3_MAGIC = 0x4E515033;  // "NQP3"
constexpr size_t NANOQ_V3_FOOTER_SIZE = 32;
constexpr size_t NANOQ_V3_ALIGN = 64;

enum class NanoqTensorDtype : int {
    Int8 = 0,
    Fp16 = 1,
    Fp4 = 2,
    Fp32 = 3,
};

enum class NanoqQuantMode : int {
    None = 0,
    PerRow = 1,
    PerBlock = 2,
};

struct NanoqTensorEntry {
    std::string name;
    NanoqTensorDtype dtype = NanoqTensorDtype::Int8;
    std::vector<int64_t> shape;
    uint64_t offset = 0;
    uint64_t size = 0;
    uint64_t scale_offset = 0;
    NanoqQuantMode quant = NanoqQuantMode::None;
    int block_size = 32;
};

struct NanoqConfig {
    std::string arch = "gpt2";
    int vocab_size = 50257;
    int hidden_size = 768;
    int n_layers = 6;
    int n_heads = 12;
    int n_kv_heads = 12;
    int max_seq_len = 2048;
    float norm_eps = 1e-5f;
    float rope_theta = 10000.0f;
    std::string act_fn = "gelu";
};

struct NanoqArchiveV3 {
    std::string path;
    std::vector<uint8_t> owned_data;
    const uint8_t* data = nullptr;
    size_t data_len = 0;
    bool mmap_owned = false;

    std::vector<NanoqTensorEntry> tensors;
    NanoqConfig config;
    std::vector<uint8_t> tokenizer_blob;
    std::unordered_map<std::string, size_t> tensor_index;

    bool legacy_demo = false;
    bool valid = false;

    const NanoqTensorEntry* find_tensor(const std::string& name) const;
    const uint8_t* tensor_bytes(const NanoqTensorEntry& e) const;
    const float* tensor_scales(const NanoqTensorEntry& e, size_t& count) const;
    std::string model_info_json() const;
};

bool nanoq_is_v3_magic(const uint8_t* data, size_t len);
bool nanoq_archive_load_v3(const char* path, NanoqArchiveV3& out, std::string& err);
bool nanoq_archive_load_v3_buffer(const uint8_t* data, size_t len, NanoqArchiveV3& out, std::string& err);
void nanoq_archive_release(NanoqArchiveV3& arch);
