#include "nanoq_loader.hpp"
#include "nanoq_runtime_ffi.h"
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

bool read_payload(std::istream& in, NanoqModel& out, std::string& err) {
    switch (out.dtype) {
        case NanoqDtype::Int8: {
            const size_t elements = static_cast<size_t>(out.rows) * static_cast<size_t>(out.cols);
            out.int8_weights.resize(elements);
            in.read(reinterpret_cast<char*>(out.int8_weights.data()),
                    static_cast<std::streamsize>(elements));
            const size_t scale_count = static_cast<size_t>(out.rows);
            out.scales.resize(scale_count);
            in.read(reinterpret_cast<char*>(out.scales.data()),
                    static_cast<std::streamsize>(scale_count * sizeof(float)));
            break;
        }
        case NanoqDtype::Fp16: {
            const size_t elements = static_cast<size_t>(out.rows) * static_cast<size_t>(out.cols);
            out.fp16_weights.resize(elements);
            in.read(reinterpret_cast<char*>(out.fp16_weights.data()),
                    static_cast<std::streamsize>(elements * sizeof(uint16_t)));
            break;
        }
        case NanoqDtype::Fp4: {
            const size_t elements = static_cast<size_t>(out.rows) * static_cast<size_t>(out.cols);
            const size_t packed = (elements + 1) / 2;
            out.fp4_packed.resize(packed);
            in.read(reinterpret_cast<char*>(out.fp4_packed.data()),
                    static_cast<std::streamsize>(packed));
            const size_t num_blocks =
                (elements + static_cast<size_t>(out.block_size) - 1) /
                static_cast<size_t>(out.block_size);
            out.scales.resize(num_blocks);
            in.read(reinterpret_cast<char*>(out.scales.data()),
                    static_cast<std::streamsize>(num_blocks * sizeof(float)));
            break;
        }
    }
    if (!in) {
        err = "truncated payload";
        return false;
    }
    return true;
}

bool parse_header_and_payload(std::istream& in, NanoqModel& out, std::string& err) {
    uint32_t header_len = 0;
    in.read(reinterpret_cast<char*>(&header_len), 4);
    if (!in || header_len == 0 || header_len > 65536) {
        err = "invalid header length";
        return false;
    }

    std::string header(header_len, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header_len));
    if (!in) {
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

    return read_payload(in, out, err);
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
        "{\"dtype\":\"%s\",\"rows\":%d,\"cols\":%d,\"version\":%d,\"length\":%zu,\"legacy_demo\":true}",
        m.dtype_str(), m.rows, m.cols, m.version, m.flat_len());
    return buf;
}

std::string NanoqLoadedModel::info_json() const {
    if (format == NanoqFormat::V3Archive) return v3.model_info_json();
    return nanoq_model_info_json(v2);
}

bool nanoq_load_unified_buffer(const uint8_t* data, size_t len, NanoqLoadedModel& out, std::string& err) {
    out = NanoqLoadedModel{};
    if (!data || len < 4) {
        err = "empty buffer";
        return false;
    }
    if (nanoq_is_v3_magic(data, len)) {
        if (nanoq_archive_validate(data, len) != 0) {
            err = "v3 validation failed";
            return false;
        }
        if (!nanoq_archive_load_v3_buffer(data, len, out.v3, err)) return false;
        out.format = NanoqFormat::V3Archive;
        out.legacy_demo = false;
        return true;
    }
    if (!nanoq_load_buffer(data, len, out.v2, err)) return false;
    out.format = NanoqFormat::V2Legacy;
    out.legacy_demo = true;
    return true;
}

bool nanoq_load_unified_file(const char* path, NanoqLoadedModel& out, std::string& err) {
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
    std::vector<uint8_t> buf(size);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(size));
    if (!f) {
        err = "read failed";
        return false;
    }
    if (nanoq_is_v3_magic(buf.data(), buf.size())) {
        if (nanoq_archive_validate_path(path) != 0) {
            err = "v3 validation failed";
            return false;
        }
        if (!nanoq_archive_load_v3(path, out.v3, err)) return false;
        out.format = NanoqFormat::V3Archive;
        out.legacy_demo = false;
        return true;
    }
    if (!nanoq_load_file(path, out.v2, err)) return false;
    out.format = NanoqFormat::V2Legacy;
    out.legacy_demo = true;
    return true;
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
    return parse_header_and_payload(f, out, err);
}

bool nanoq_load_buffer(const uint8_t* data, size_t len, NanoqModel& out, std::string& err) {
    if (!data || len < 4) {
        err = "empty buffer";
        return false;
    }
    std::string blob(reinterpret_cast<const char*>(data), len);
    std::istringstream in(blob, std::ios::binary);
    return parse_header_and_payload(in, out, err);
}
