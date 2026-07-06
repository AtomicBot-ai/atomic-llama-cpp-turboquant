#pragma once

#include "common.cuh"

// Delta-encode F32 values against F16 keyframes, producing Q4_0 deltas.
// Also updates keyframes for keyframe slots (where slot_pos % kf_interval == 0).
void ggml_cuda_delta_kv_encode(
    const float * src_f32,       // [n_embd] current token values
    const int64_t * idxs,        // [n_tokens] destination slot indices
    half * keyframes,            // [n_embd, kv_size] keyframe buffer (read/write)
    void * deltas,               // [n_embd/32 blocks, kv_size] Q4_0 delta buffer (write)
    const int64_t n_embd,
    const int64_t n_tokens,
    const int64_t kv_size,
    const int64_t nb_src_row,    // stride between token rows in src (bytes)
    const int64_t nb_kf_row,     // stride between rows in keyframe buffer (bytes)
    const int64_t nb_delta_row,  // stride between rows in delta buffer (bytes)
    const int     kf_interval,   // keyframe every N slots (0 = every slot is keyframe)
    cudaStream_t stream);

// Reconstruct F16 values from Q4_0 deltas + F16 keyframes.
void ggml_cuda_delta_kv_reconstruct(
    const void * deltas,         // [n_embd/32 blocks, n_kv] Q4_0 delta buffer
    const half * keyframes,      // [n_embd, n_kv] F16 keyframe buffer
    half * dst,                  // [n_embd, n_kv] reconstructed F16 output
    const int64_t n_embd,
    const int64_t n_kv,
    const int64_t nb_delta_row,  // stride between rows in delta buffer (bytes)
    const int64_t nb_kf_row,     // stride between rows in keyframe buffer (bytes)
    const int64_t nb_dst_row,    // stride between rows in dst buffer (bytes)
    cudaStream_t stream);
