#include "nanoq_archive.hpp"
#include "nanoq_runtime_ffi.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {

int json_int(const std::string& j, const char* key, int def) {
    std::string needle = std::string("\"") + key + "\":";
    auto pos = j.find(needle);
    if (pos == std::string::npos) return def;
    pos += needle.size();
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t')) ++pos;
    return std::atoi(j.c_str() + pos);
}

float json_float(const std::string& j, const char* key, float def) {
    std::string needle = std::string("\"") + key + "\":";
    auto pos = j.find(needle);
    if (pos == std::string::npos) return def;
    pos += needle.size();
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t')) ++pos;
    return static_cast<float>(std::atof(j.c_str() + pos));
}

std::string json_str(const std::string& j, const char* key) {
    std::string needle = std::string("\"") + key + "\":";
    auto pos = j.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t')) ++pos;
    if (pos >= j.size() || j[pos] != '"') return {};
    ++pos;
    auto end = j.find('"', pos);
    if (end == std::string::npos) return {};
    return j.substr(pos, end - pos);
}

NanoqTensorDtype parse_dtype(const std::string& s) {
    if (s == "fp16") return NanoqTensorDtype::Fp16;
    if (s == "fp4") return NanoqTensorDtype::Fp4;
    if (s == "fp32") return NanoqTensorDtype::Fp32;
    return NanoqTensorDtype::Int8;
}

NanoqQuantMode parse_quant(const std::string& s) {
    if (s == "per-row") return NanoqQuantMode::PerRow;
    if (s == "per-block") return NanoqQuantMode::PerBlock;
    return NanoqQuantMode::None;
}

size_t align_up(size_t v, size_t a) {
    return (v + a - 1) / a * a;
}

bool verify_blake3_simple(const uint8_t* data, size_t len) {
    return nanoq_archive_validate(data, len) == 0;
}

bool parse_index_array(const std::string& json, std::vector<NanoqTensorEntry>& out) {
    out.clear();
    size_t pos = 0;
    while (pos < json.size()) {
        auto obj_start = json.find('{', pos);
        if (obj_start == std::string::npos) break;
        auto obj_end = json.find('}', obj_start);
        if (obj_end == std::string::npos) break;
        std::string obj = json.substr(obj_start, obj_end - obj_start + 1);

        NanoqTensorEntry e;
        e.name = json_str(obj, "name");
        e.dtype = parse_dtype(json_str(obj, "dtype"));
        e.quant = parse_quant(json_str(obj, "quant"));
        e.block_size = json_int(obj, "block_size", 32);
        e.offset = static_cast<uint64_t>(json_int(obj, "offset", 0));
        e.size = static_cast<uint64_t>(json_int(obj, "size", 0));
        e.scale_offset = static_cast<uint64_t>(json_int(obj, "scale_offset", 0));

        auto shape_pos = obj.find("\"shape\"");
        if (shape_pos != std::string::npos) {
            auto lb = obj.find('[', shape_pos);
            auto rb = obj.find(']', lb);
            if (lb != std::string::npos && rb != std::string::npos) {
                std::string shape_str = obj.substr(lb + 1, rb - lb - 1);
                std::stringstream ss(shape_str);
                std::string tok;
                while (std::getline(ss, tok, ',')) {
                    e.shape.push_back(std::atoll(tok.c_str()));
                }
            }
        }
        if (!e.name.empty()) out.push_back(std::move(e));
        pos = obj_end + 1;
    }
    return !out.empty();
}

bool parse_config_json(const std::string& json, NanoqConfig& cfg) {
    cfg.arch = json_str(json, "arch");
    if (cfg.arch.empty()) cfg.arch = "gpt2";
    cfg.vocab_size = json_int(json, "vocab_size", 50257);
    cfg.hidden_size = json_int(json, "hidden_size", 768);
    cfg.n_layers = json_int(json, "n_layers", 6);
    cfg.n_heads = json_int(json, "n_heads", 12);
    cfg.n_kv_heads = json_int(json, "n_kv_heads", cfg.n_heads);
    cfg.max_seq_len = json_int(json, "max_seq_len", 2048);
    cfg.norm_eps = json_float(json, "norm_eps", 1e-5f);
    cfg.rope_theta = json_float(json, "rope_theta", 10000.0f);
    cfg.act_fn = json_str(json, "act_fn");
    if (cfg.act_fn.empty()) cfg.act_fn = "gelu";
    return true;
}

bool parse_v3_buffer(const uint8_t* data, size_t len, NanoqArchiveV3& out, std::string& err) {
    if (len < 16) {
        err = "buffer too small";
        return false;
    }
    uint32_t magic = 0;
    std::memcpy(&magic, data, 4);
    if (magic != NANOQ_V3_MAGIC) {
        err = "invalid v3 magic";
        return false;
    }

    uint32_t index_len = 0;
    std::memcpy(&index_len, data + 4, 4);
    if (index_len == 0 || index_len > len) {
        err = "invalid index length";
        return false;
    }

    size_t index_start = 8;
    size_t index_end = index_start + index_len;
    if (index_end + 4 > len) {
        err = "truncated index";
        return false;
    }
    std::string index_json(reinterpret_cast<const char*>(data + index_start), index_len);
    if (!parse_index_array(index_json, out.tensors)) {
        err = "empty tensor index";
        return false;
    }

    uint32_t config_len = 0;
    std::memcpy(&config_len, data + index_end, 4);
    size_t config_start = index_end + 4;
    size_t config_end = config_start + config_len;
    if (config_end + 4 > len) {
        err = "truncated config";
        return false;
    }
    std::string config_json(reinterpret_cast<const char*>(data + config_start), config_len);
    parse_config_json(config_json, out.config);

    uint32_t tokenizer_len = 0;
    std::memcpy(&tokenizer_len, data + config_end, 4);
    size_t tokenizer_start = config_end + 4;
    size_t tokenizer_end = tokenizer_start + tokenizer_len;
    if (tokenizer_end > len) {
        err = "truncated tokenizer";
        return false;
    }
    out.tokenizer_blob.assign(data + tokenizer_start, data + tokenizer_end);

    size_t payload_start = align_up(tokenizer_end, NANOQ_V3_ALIGN);
    if (payload_start + NANOQ_V3_FOOTER_SIZE > len) {
        err = "missing footer";
        return false;
    }

    if (!verify_blake3_simple(data, len)) {
        err = "Blake3 footer validation failed";
        return false;
    }

    out.data = data;
    out.data_len = len;
    out.tensor_index.clear();
    for (size_t i = 0; i < out.tensors.size(); ++i)
        out.tensor_index[out.tensors[i].name] = i;
    out.legacy_demo = false;
    out.valid = true;
    return true;
}

}  // namespace

bool nanoq_is_v3_magic(const uint8_t* data, size_t len) {
    if (!data || len < 4) return false;
    uint32_t magic = 0;
    std::memcpy(&magic, data, 4);
    return magic == NANOQ_V3_MAGIC;
}

const NanoqTensorEntry* NanoqArchiveV3::find_tensor(const std::string& name) const {
    auto it = tensor_index.find(name);
    if (it == tensor_index.end()) return nullptr;
    return &tensors[it->second];
}

const uint8_t* NanoqArchiveV3::tensor_bytes(const NanoqTensorEntry& e) const {
    if (!data || !valid) return nullptr;
    size_t payload_start = 0;
    uint32_t index_len = 0, config_len = 0, tokenizer_len = 0;
    std::memcpy(&index_len, data + 4, 4);
    size_t index_end = 8 + index_len;
    std::memcpy(&config_len, data + index_end, 4);
    size_t config_end = index_end + 4 + config_len;
    std::memcpy(&tokenizer_len, data + config_end, 4);
    payload_start = align_up(config_end + 4 + tokenizer_len, NANOQ_V3_ALIGN);
    return data + payload_start + e.offset;
}

const float* NanoqArchiveV3::tensor_scales(const NanoqTensorEntry& e, size_t& count) const {
    count = 0;
    if (!data || e.scale_offset == 0) return nullptr;
    size_t payload_start = 0;
    uint32_t index_len = 0, config_len = 0, tokenizer_len = 0;
    std::memcpy(&index_len, data + 4, 4);
    size_t index_end = 8 + index_len;
    std::memcpy(&config_len, data + index_end, 4);
    size_t config_end = index_end + 4 + config_len;
    std::memcpy(&tokenizer_len, data + config_end, 4);
    payload_start = align_up(config_end + 4 + tokenizer_len, NANOQ_V3_ALIGN);
    if (e.quant == NanoqQuantMode::PerRow && !e.shape.empty()) {
        count = static_cast<size_t>(e.shape[0]);
    } else {
        count = e.size > 0 ? 1 : 0;
    }
    return reinterpret_cast<const float*>(data + payload_start + e.scale_offset);
}

std::string NanoqArchiveV3::model_info_json() const {
    int max_seq = config.max_seq_len;
    if (const auto* wpe = find_tensor("transformer.wpe.weight"); wpe && !wpe->shape.empty())
        max_seq = std::min(max_seq, static_cast<int>(wpe->shape[0]));
    char buf[640];
    std::snprintf(buf, sizeof(buf),
        "{\"format\":\"nanoq_v3\",\"arch\":\"%s\",\"vocab_size\":%d,\"hidden_size\":%d,"
        "\"n_layers\":%d,\"n_heads\":%d,\"max_seq_len\":%d,\"legacy_demo\":false,\"tensor_count\":%zu}",
        config.arch.c_str(), config.vocab_size, config.hidden_size,
        config.n_layers, config.n_heads, max_seq, tensors.size());
    return buf;
}

bool nanoq_archive_load_v3_buffer(const uint8_t* data, size_t len, NanoqArchiveV3& out, std::string& err) {
    out = NanoqArchiveV3{};
    return parse_v3_buffer(data, len, out, err);
}

bool nanoq_archive_load_v3(const char* path, NanoqArchiveV3& out, std::string& err) {
    if (!path || !path[0]) {
        err = "empty path";
        return false;
    }
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        err = "cannot open file";
        return false;
    }
    auto size = static_cast<size_t>(f.tellg());
    f.seekg(0);
    out.owned_data.resize(size);
    f.read(reinterpret_cast<char*>(out.owned_data.data()), static_cast<std::streamsize>(size));
    if (!f) {
        err = "read failed";
        return false;
    }
    out.path = path;
    out.data = out.owned_data.data();
    out.data_len = size;
    if (!parse_v3_buffer(out.data, out.data_len, out, err)) return false;
    return true;
}

void nanoq_archive_release(NanoqArchiveV3& arch) {
    arch = NanoqArchiveV3{};
}
