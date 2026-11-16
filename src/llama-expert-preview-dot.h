#pragma once

#include <cstddef>

// Independent preview only: fixed row count lets the compiler reuse each router
// cache line without a dynamic row branch inside the dot loop. Preserve the
// original eight-lane accumulation and scalar lane reduction for every row.
// Callers own/admit scores and validate widths, row strides and finite results.
template <size_t Rows>
inline void llama_expert_preview_dot(const float * router, const char * input, size_t row_stride,
                                     size_t width, size_t experts, float * scores, size_t score_stride) {
    static_assert(Rows == 1 || Rows == 4, "unsupported preview row block");
    const float * x[Rows];
    for (size_t r = 0; r < Rows; ++r) { x[r] = reinterpret_cast<const float *>(input + r * row_stride); }
    for (size_t e = 0; e < experts; ++e) {
        const float * w = router + e * width;
        float sum[Rows][8] {};
        size_t c = 0;
        for (; c + 8 <= width; c += 8) {
            for (size_t r = 0; r < Rows; ++r) {
                for (size_t j = 0; j < 8; ++j) { sum[r][j] += x[r][c + j] * w[c + j]; }
            }
        }
        for (size_t r = 0; r < Rows; ++r) {
            float score = 0;
            for (float v : sum[r]) { score += v; }
            for (size_t tail = c; tail < width; ++tail) { score += x[r][tail] * w[tail]; }
            scores[r * score_stride + e] = score;
        }
    }
}
