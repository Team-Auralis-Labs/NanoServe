#pragma once
#if !defined(NANOSERVE_WASM) && !defined(__EMSCRIPTEN__)
#include <immintrin.h>
#endif
#include <span>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>
#include <cmath>

#if defined(__AVX2__) && defined(__F16C__)
#define NANOSERVE_HAS_FP16_SIMD 1
#endif

inline float int8_dot_avx2(std::span<const int8_t> weights, std::span<const float> acts) {
    size_t n = std::min(weights.size(), acts.size());
    float acc = 0.0f;
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float wf[8];
        for (int k = 0; k < 8; ++k) wf[k] = static_cast<float>(weights.data()[i + k]);
        __m256 w = _mm256_loadu_ps(wf);
        __m256 a = _mm256_loadu_ps(&acts.data()[i]);
        sum = _mm256_fmadd_ps(w, a, sum);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    for (float v : tmp) acc += v;
    for (; i < n; ++i) acc += weights.data()[i] * acts.data()[i];
#else
    for (size_t i = 0; i < n; ++i) acc += weights.data()[i] * acts.data()[i];
#endif
    return acc;
}

inline float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        float val = static_cast<float>(mant) / 1024.0f;
        return sign ? -val * (1.0f / 16384.0f) : val * (1.0f / 16384.0f);
    }
    if (exp == 31) return mant ? NAN : (sign ? -INFINITY : INFINITY);
    uint32_t bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

inline float fp16_dot_scalar(std::span<const uint16_t> weights, std::span<const float> acts) {
    size_t n = std::min(weights.size(), acts.size());
    float acc = 0.0f;
    for (size_t i = 0; i < n; ++i)
        acc += fp16_to_float(weights.data()[i]) * acts.data()[i];
    return acc;
}

inline float fp16_dot_avx2(std::span<const uint16_t> weights, std::span<const float> acts) {
#if defined(NANOSERVE_HAS_FP16_SIMD)
    size_t n = std::min(weights.size(), acts.size());
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(weights.data() + i));
        __m256 w = _mm256_cvtph_ps(raw);
        __m256 a = _mm256_loadu_ps(acts.data() + i);
        sum = _mm256_fmadd_ps(w, a, sum);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    float acc = 0.0f;
    for (float v : tmp) acc += v;
    for (; i < n; ++i)
        acc += fp16_to_float(weights.data()[i]) * acts.data()[i];
    return acc;
#else
    return fp16_dot_scalar(weights, acts);
#endif
}

inline float fp4_dequant(uint8_t nibble, float scale) {
    int val = static_cast<int>(nibble & 0x0f);
    if (val >= 8) val -= 16;
    return static_cast<float>(val) * scale / 7.0f;
}

inline float fp4_dot_scalar(std::span<const uint8_t> packed, std::span<const float> scales,
                            int block_size, std::span<const float> acts) {
    size_t n = std::min(acts.size(), packed.size() * 2);
    float acc = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        size_t byte_idx = i / 2;
        if (byte_idx >= packed.size()) break;
        uint8_t byte = packed.data()[byte_idx];
        uint8_t nibble = (i & 1) ? (byte >> 4) : (byte & 0x0f);
        size_t block = block_size > 0 ? i / static_cast<size_t>(block_size) : 0;
        float scale = block < scales.size() ? scales.data()[block] : 1.0f;
        acc += fp4_dequant(nibble, scale) * acts.data()[i];
    }
    return acc;
}

namespace detail {

#if defined(__AVX2__)

inline __m256 fp4_nibbles_to_f32(const uint8_t* nibbles, float scale_over_7) {
    alignas(32) int32_t vals[8];
    for (int k = 0; k < 8; ++k) {
        int v = static_cast<int>(nibbles[k] & 0x0f);
        if (v >= 8) v -= 16;
        vals[k] = v;
    }
    __m256 iv = _mm256_cvtepi32_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(vals)));
    return _mm256_mul_ps(iv, _mm256_set1_ps(scale_over_7));
}

inline void fp4_unpack8(const uint8_t* bytes, uint8_t* out16) {
    for (int k = 0; k < 8; ++k) {
        out16[2 * k] = bytes[k] & 0x0f;
        out16[2 * k + 1] = (bytes[k] >> 4) & 0x0f;
    }
}

inline float fp4_dot_block32_simd(std::span<const uint8_t> packed, std::span<const float> scales,
                                  std::span<const float> acts) {
    constexpr int block_size = 32;
    const size_t n = std::min(acts.size(), packed.size() * 2);
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;

    while (i + block_size <= n) {
        const size_t block = i / block_size;
        const float scale_over_7 =
            (block < scales.size() ? scales.data()[block] : 1.0f) / 7.0f;
        const size_t byte_base = i / 2;

        alignas(16) uint8_t nibbles[16];
        fp4_unpack8(packed.data() + byte_base, nibbles);
        __m256 w0 = fp4_nibbles_to_f32(nibbles, scale_over_7);
        __m256 a0 = _mm256_loadu_ps(acts.data() + i);
        sum = _mm256_fmadd_ps(w0, a0, sum);

        __m256 w1 = fp4_nibbles_to_f32(nibbles + 8, scale_over_7);
        __m256 a1 = _mm256_loadu_ps(acts.data() + i + 8);
        sum = _mm256_fmadd_ps(w1, a1, sum);

        fp4_unpack8(packed.data() + byte_base + 8, nibbles);
        __m256 w2 = fp4_nibbles_to_f32(nibbles, scale_over_7);
        __m256 a2 = _mm256_loadu_ps(acts.data() + i + 16);
        sum = _mm256_fmadd_ps(w2, a2, sum);

        __m256 w3 = fp4_nibbles_to_f32(nibbles + 8, scale_over_7);
        __m256 a3 = _mm256_loadu_ps(acts.data() + i + 24);
        sum = _mm256_fmadd_ps(w3, a3, sum);

        i += block_size;
    }

    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    float acc = 0.0f;
    for (float v : tmp) acc += v;

    for (; i < n; ++i) {
        size_t byte_idx = i / 2;
        uint8_t byte = packed.data()[byte_idx];
        uint8_t nibble = (i & 1) ? (byte >> 4) : (byte & 0x0f);
        size_t block = i / block_size;
        float scale = block < scales.size() ? scales.data()[block] : 1.0f;
        acc += fp4_dequant(nibble, scale) * acts.data()[i];
    }
    return acc;
}

#endif  // __AVX2__

}  // namespace detail

inline float fp4_dot(std::span<const uint8_t> packed, std::span<const float> scales,
                     int block_size, std::span<const float> acts) {
#if defined(__AVX2__)
    if (block_size == 32)
        return detail::fp4_dot_block32_simd(packed, scales, acts);
#endif
    return fp4_dot_scalar(packed, scales, block_size, acts);
}
