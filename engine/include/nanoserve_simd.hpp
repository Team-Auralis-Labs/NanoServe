#pragma once
#include <immintrin.h>
#include <span>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>

inline float int8_dot_avx2(std::span<const int8_t> weights, std::span<const float> acts) {
    size_t n = std::min(weights.size(), acts.size());
    float acc = 0.0f;
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        float wf[8];
        for (int k = 0; k < 8; ++k) wf[k] = static_cast<float>(weights[i + k]);
        __m256 w = _mm256_loadu_ps(wf);
        __m256 a = _mm256_loadu_ps(&acts[i]);
        sum = _mm256_fmadd_ps(w, a, sum);
    }
    float tmp[8];
    _mm256_storeu_ps(tmp, sum);
    for (float v : tmp) acc += v;
    for (; i < n; ++i) acc += weights[i] * acts[i];
#else
    for (size_t i = 0; i < n; ++i) acc += weights[i] * acts[i];
#endif
    return acc;
}
