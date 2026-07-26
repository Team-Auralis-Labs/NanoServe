#include "nanoq_loader.hpp"
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {

int json_int_field(const std::string& j, const char* key, int def) {
    std::string needle = std::string("\"") + key + "\":";
    auto pos = j.find(needle);
    if (pos == std::string::npos) return def;
    pos += needle.size();
    while (pos < j.size() && (j[pos] == ' ' || j[pos] == '\t')) ++pos;
    return std::atoi(j.c_str() + pos);
}

std::string json_str_field(const std::string& j, const char* key) {
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

NanoqDtype parse_dtype(const std::string& s) {
    if (s == "fp16") return NanoqDtype::Fp16;
    if (s == "fp4") return NanoqDtype::Fp4;
    return NanoqDtype::Int8;
}

}  // namespace

size_t NanoqModel::flat_len() const {
    switch (dtype) {
        case NanoqDtype::Int8:
            return int8_weights.size();
        case NanoqDtype::Fp16:
            return fp16_weights.size();
        case NanoqDtype::Fp4:
            return fp4_packed.size() * 2;
    }
    return 0;
}

const char* NanoqModel::dtype_str() const {
    switch (dtype) {
        case NanoqDtype::Fp16: return "fp16";
        case NanoqDtype::Fp4: return "fp4";
        default: return "int8";
    }
}

std::string nanoq_model_info_json(const NanoqModel& m) {
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "{\"dtype\":\"%s\",\"rows\":%d,\"cols\":%d,\"version\":%d,\"length\":%zu}",
        m.dtype_str(), m.rows, m.cols, m.version, m.flat_len());
    return buf;
}

bool nanoq_load_file(const char* path, NanoqModel& out, std::string& err) {
    if (!path || !path[0]) {
        err = "empty path";
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        err = "cannot open file";
        return false;
    }

    uint32_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 4);
    if (!f || header_len == 0 || header_len > 65536) {
        err = "invalid header length";
        return false;
    }

    std::string header(header_len, '\0');
    f.read(header.data(), header_len);
    if (!f) {
        err = "truncated header";
        return false;
    }

    out = NanoqModel{};
    out.version = json_int_field(header, "version", 1);
    out.rows = json_int_field(header, "rows", 0);
    out.cols = json_int_field(header, "cols", 0);
    out.block_size = json_int_field(header, "block_size", 32);
    out.name = json_str_field(header, "name");
    out.dtype = parse_dtype(json_str_field(header, "dtype"));

    if (out.rows <= 0 || out.cols <= 0) {
        err = "invalid rows/cols";
        return false;
    }

    const size_t elements = static_cast<size_t>(out.rows) * static_cast<size_t>(out.cols);

    switch (out.dtype) {
        case NanoqDtype::Int8: {
            out.int8_weights.resize(elements);
            f.read(reinterpret_cast<char*>(out.int8_weights.data()),
                   static_cast<std::streamsize>(elements));
            const size_t scale_count = static_cast<size_t>(out.rows);
            out.scales.resize(scale_count);
            f.read(reinterpret_cast<char*>(out.scales.data()),
                   static_cast<std::streamsize>(scale_count * sizeof(float)));
            break;
        }
        case NanoqDtype::Fp16: {
            out.fp16_weights.resize(elements);
            f.read(reinterpret_cast<char*>(out.fp16_weights.data()),
                   static_cast<std::streamsize>(elements * sizeof(uint16_t)));
            break;
        }
        case NanoqDtype::Fp4: {
            const size_t packed = (elements + 1) / 2;
            out.fp4_packed.resize(packed);
            f.read(reinterpret_cast<char*>(out.fp4_packed.data()),
                   static_cast<std::streamsize>(packed));
            const size_t num_blocks =
                (elements + static_cast<size_t>(out.block_size) - 1) /
                static_cast<size_t>(out.block_size);
            out.scales.resize(num_blocks);
            f.read(reinterpret_cast<char*>(out.scales.data()),
                   static_cast<std::streamsize>(num_blocks * sizeof(float)));
            break;
        }
    }

    if (!f) {
        err = "truncated payload";
        return false;
    }
    return true;
}
