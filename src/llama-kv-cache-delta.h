#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <vector>
#include <cstdint>

// Delta KV Cache Processor
//
// Applies delta compression as a post-processing filter on the KV cache.
// After each graph_compute writes new KV entries, this processor:
//   1. Reads the newly written F16 values
//   2. Computes delta from the previous token's reconstructed values
//   3. Quantizes delta to Q4_0 (lossy step)
//   4. Reconstructs: previous + dequant(delta)
//   5. Writes reconstructed values back to the KV cache
//
// Phase 1: CPU fallback (uses ggml_backend_tensor_get/set)
// Phase 2: GPU-native (uses ggml_backend_cuda_delta_kv_filter)

struct llama_kv_cache_delta {
    bool enabled  = false;
    bool use_gpu  = false;  // Phase 2: GPU-native processing
    int  kf_interval = 32;  // keyframe every N tokens

    ~llama_kv_cache_delta();

    // Initialize for a given configuration
    void init(uint32_t n_embd_k, uint32_t n_embd_v, uint32_t kv_size);

    // Try to initialize GPU buffers. Returns true if GPU path is available.
    bool init_gpu(int device);

    // Apply delta compression filter to newly written entries
    void process(
        ggml_tensor * tensor,
        const int64_t * slot_indices,
        int n_tokens,
        int layer_idx,
        bool is_key);

    // Reset all state
    void clear();

    // Stats
    int64_t n_processed = 0;
    int64_t n_keyframes = 0;
    int64_t n_deltas    = 0;

private:
    // Q4_0 block for CPU-side quantization
    struct block_q4_0_cpu {
        float d;
        uint8_t qs[16];
    };

    static void quantize_q4_0(const float * x, block_q4_0_cpu & block);
    static void dequantize_q4_0(const block_q4_0_cpu & block, float * out);

    uint32_t m_n_embd_k = 0;
    uint32_t m_n_embd_v = 0;
    uint32_t m_kv_size  = 0;

    // CPU fallback state
    struct layer_data {
        std::vector<std::vector<float>> k_prev;
        std::vector<std::vector<float>> v_prev;
        std::vector<bool> k_initialized;
        std::vector<bool> v_initialized;
        int k_tokens_since_kf = 0;
        int v_tokens_since_kf = 0;
    };
    std::vector<layer_data> m_layers;
    void ensure_layer(int layer_idx, uint32_t n_embd, bool is_key);

    // GPU state (Phase 2)
    struct gpu_layer_data {
        void * k_ref = nullptr;  // GPU F16 reference buffer
        void * v_ref = nullptr;
        std::vector<uint8_t> k_initialized;  // use uint8_t instead of bool for .data()
        std::vector<uint8_t> v_initialized;
        int k_tokens_since_kf = 0;
        int v_tokens_since_kf = 0;
    };
    std::vector<gpu_layer_data> m_gpu_layers;
    int m_gpu_device = -1;
    void ensure_gpu_layer(int layer_idx, uint32_t n_embd, bool is_key);

    // Processing methods
    void process_cpu(ggml_tensor * tensor, const int64_t * slot_indices, int n_tokens, int layer_idx, bool is_key);
    void process_gpu(ggml_tensor * tensor, const int64_t * slot_indices, int n_tokens, int layer_idx, bool is_key);
};
