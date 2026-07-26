#pragma once
#include <cstdint>
#include <string>
#include <vector>

enum class NanoqDtype : int {
    Int8 = 0,
    Fp16 = 1,
    Fp4 = 2,
};

struct NanoqModel {
    NanoqDtype dtype = NanoqDtype::Int8;
    int version = 1;
    int rows = 0;
    int cols = 0;
    int block_size = 32;
    std::string name;

    std::vector<int8_t> int8_weights;
    std::vector<float> scales;
    std::vector<uint16_t> fp16_weights;
    std::vector<uint8_t> fp4_packed;

    size_t flat_len() const;
    const char* dtype_str() const;
};

bool nanoq_load_file(const char* path, NanoqModel& out, std::string& err);
bool nanoq_load_buffer(const uint8_t* data, size_t len, NanoqModel& out, std::string& err);
std::string nanoq_model_info_json(const NanoqModel& m);
