#include "llama-kv-cache-delta.h"
#include "llama-impl.h"

#if defined(GGML_USE_CUDA) || defined(GGML_USE_HIP)
#include "ggml-cuda.h"
#define DELTA_KV_HAS_GPU 1
#else
#define DELTA_KV_HAS_GPU 0
#endif

#include <cmath>
#include <cstring>
#include <algorithm>

// ============================================================================
// Q4_0 encode/decode (CPU reference)
// ============================================================================

void llama_kv_cache_delta::quantize_q4_0(const float * x, block_q4_0_cpu & block) {
    float amax = 0.0f, vmax = 0.0f;
    for (int j = 0; j < 32; j++) {
        if (fabsf(x[j]) > amax) {
            amax = fabsf(x[j]);
            vmax = x[j];
        }
    }
    const float d = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    block.d = d;
    for (int j = 0; j < 16; j++) {
        const float x0 = x[j] * id;
        const float x1 = x[j + 16] * id;
        const uint8_t xi0 = std::min(15, (int)(x0 + 8.5f));
        const uint8_t xi1 = std::min(15, (int)(x1 + 8.5f));
        block.qs[j] = xi0 | (xi1 << 4);
    }
}

void llama_kv_cache_delta::dequantize_q4_0(const block_q4_0_cpu & block, float * out) {
    const float d = block.d;
    for (int j = 0; j < 16; j++) {
        const uint8_t packed = block.qs[j];
        out[j]      = d * ((float)(packed & 0x0F) - 8.0f);
        out[j + 16] = d * ((float)(packed >> 4) - 8.0f);
    }
}

// ============================================================================
// Lifecycle
// ============================================================================

llama_kv_cache_delta::~llama_kv_cache_delta() {
#if DELTA_KV_HAS_GPU
    for (auto & gl : m_gpu_layers) {
        if (gl.k_ref) ggml_backend_cuda_delta_kv_free_ref(gl.k_ref);
        if (gl.v_ref) ggml_backend_cuda_delta_kv_free_ref(gl.v_ref);
    }
#endif
}

void llama_kv_cache_delta::init(uint32_t n_embd_k, uint32_t n_embd_v, uint32_t kv_size) {
    m_n_embd_k = n_embd_k;
    m_n_embd_v = n_embd_v;
    m_kv_size  = kv_size;
    m_layers.clear();
    m_gpu_layers.clear();
    n_processed = 0;
    n_keyframes = 0;
    n_deltas    = 0;
}

bool llama_kv_cache_delta::init_gpu(int device) {
#if DELTA_KV_HAS_GPU
    m_gpu_device = device;
    use_gpu = true;
    LLAMA_LOG_INFO("%s: delta KV using GPU device %d\n", __func__, device);
    return true;
#else
    GGML_UNUSED(device);
    return false;
#endif
}

void llama_kv_cache_delta::clear() {
#if DELTA_KV_HAS_GPU
    for (auto & gl : m_gpu_layers) {
        if (gl.k_ref) ggml_backend_cuda_delta_kv_free_ref(gl.k_ref);
        if (gl.v_ref) ggml_backend_cuda_delta_kv_free_ref(gl.v_ref);
        gl.k_ref = nullptr;
        gl.v_ref = nullptr;
    }
    m_gpu_layers.clear();
#endif
    m_layers.clear();
    n_processed = 0;
    n_keyframes = 0;
    n_deltas    = 0;
}

// ============================================================================
// CPU fallback helpers
// ============================================================================

void llama_kv_cache_delta::ensure_layer(int layer_idx, uint32_t n_embd, bool is_key) {
    if (layer_idx >= (int)m_layers.size()) {
        m_layers.resize(layer_idx + 1);
    }
    auto & ld = m_layers[layer_idx];
    if (is_key && ld.k_prev.empty()) {
        ld.k_prev.resize(m_kv_size);
        for (auto & v : ld.k_prev) v.resize(n_embd, 0.0f);
        ld.k_initialized.resize(m_kv_size, false);
    }
    if (!is_key && ld.v_prev.empty()) {
        ld.v_prev.resize(m_kv_size);
        for (auto & v : ld.v_prev) v.resize(n_embd, 0.0f);
        ld.v_initialized.resize(m_kv_size, false);
    }
}

// ============================================================================
// GPU helpers
// ============================================================================

void llama_kv_cache_delta::ensure_gpu_layer(int layer_idx, uint32_t n_embd, bool is_key) {
#if DELTA_KV_HAS_GPU
    if (layer_idx >= (int)m_gpu_layers.size()) {
        m_gpu_layers.resize(layer_idx + 1);
    }
    auto & gl = m_gpu_layers[layer_idx];
    const size_t ref_bytes = (size_t)n_embd * m_kv_size * sizeof(uint16_t); // F16

    if (is_key && !gl.k_ref) {
        gl.k_ref = ggml_backend_cuda_delta_kv_alloc_ref(m_gpu_device, ref_bytes);
        gl.k_initialized.resize(m_kv_size, 0);
    }
    if (!is_key && !gl.v_ref) {
        gl.v_ref = ggml_backend_cuda_delta_kv_alloc_ref(m_gpu_device, ref_bytes);
        gl.v_initialized.resize(m_kv_size, 0);
    }
#else
    GGML_UNUSED(layer_idx);
    GGML_UNUSED(n_embd);
    GGML_UNUSED(is_key);
#endif
}

// ============================================================================
// Main dispatch
// ============================================================================

void llama_kv_cache_delta::process(
    ggml_tensor * tensor,
    const int64_t * slot_indices,
    int n_tokens,
    int layer_idx,
    bool is_key)
{
    if (!enabled || n_tokens == 0 || !tensor) return;
    if (tensor->type != GGML_TYPE_F16) return;

    if (use_gpu) {
        process_gpu(tensor, slot_indices, n_tokens, layer_idx, is_key);
    } else {
        process_cpu(tensor, slot_indices, n_tokens, layer_idx, is_key);
    }
}

// ============================================================================
// CPU processing (Phase 1)
// ============================================================================

void llama_kv_cache_delta::process_cpu(
    ggml_tensor * tensor,
    const int64_t * slot_indices,
    int n_tokens,
    int layer_idx,
    bool is_key)
{
    const uint32_t n_embd = is_key ? m_n_embd_k : m_n_embd_v;
    if (n_embd == 0) return;

    ensure_layer(layer_idx, n_embd, is_key);

    auto & ld = m_layers[layer_idx];
    auto & prev = is_key ? ld.k_prev : ld.v_prev;
    auto & initialized = is_key ? ld.k_initialized : ld.v_initialized;
    auto & tokens_since_kf = is_key ? ld.k_tokens_since_kf : ld.v_tokens_since_kf;

    const size_t row_bytes = n_embd * ggml_type_size(GGML_TYPE_F16);

    std::vector<uint16_t> f16_buf(n_embd);
    std::vector<float>    f32_buf(n_embd);
    std::vector<float>    reconstructed(n_embd);
    float delta_buf[32], decoded_buf[32];
    block_q4_0_cpu block;

    for (int t = 0; t < n_tokens; t++) {
        const int64_t slot = slot_indices[t];
        if (slot < 0 || slot >= (int64_t)m_kv_size) continue;

        const size_t offset = slot * tensor->nb[1];
        ggml_backend_tensor_get(tensor, f16_buf.data(), offset, row_bytes);

        for (uint32_t j = 0; j < n_embd; j++) {
            f32_buf[j] = ggml_fp16_to_fp32(f16_buf[j]);
        }

        bool is_keyframe = !initialized[slot];
        if (kf_interval > 0) {
            tokens_since_kf++;
            if (tokens_since_kf >= kf_interval) {
                is_keyframe = true;
                tokens_since_kf = 0;
            }
        }

        if (is_keyframe) {
            for (uint32_t j = 0; j < n_embd; j++) {
                reconstructed[j] = f32_buf[j];
            }
            n_keyframes++;
        } else {
            const int blocks = n_embd / 32;
            for (int b = 0; b < blocks; b++) {
                for (int j = 0; j < 32; j++) {
                    delta_buf[j] = f32_buf[b * 32 + j] - prev[slot][b * 32 + j];
                }
                quantize_q4_0(delta_buf, block);
                dequantize_q4_0(block, decoded_buf);
                for (int j = 0; j < 32; j++) {
                    reconstructed[b * 32 + j] = prev[slot][b * 32 + j] + decoded_buf[j];
                }
            }
            n_deltas++;
        }

        for (uint32_t j = 0; j < n_embd; j++) {
            f16_buf[j] = ggml_fp32_to_fp16(reconstructed[j]);
        }
        ggml_backend_tensor_set(tensor, f16_buf.data(), offset, row_bytes);

        prev[slot] = reconstructed;
        initialized[slot] = true;
        n_processed++;
    }
}

// ============================================================================
// GPU processing (Phase 2)
// ============================================================================

void llama_kv_cache_delta::process_gpu(
    ggml_tensor * tensor,
    const int64_t * slot_indices,
    int n_tokens,
    int layer_idx,
    bool is_key)
{
#if DELTA_KV_HAS_GPU
    // TODO: GPU path needs per-device ref buffer allocation for multi-GPU
    // For now, detect multi-GPU case and fall back to CPU
    // The tensor may be on a different device than device 0
    // Fall back to CPU path which works correctly for all configurations
    process_cpu(tensor, slot_indices, n_tokens, layer_idx, is_key);
    return;

    // The code below is the single-GPU implementation, kept for future use
    const uint32_t n_embd = is_key ? m_n_embd_k : m_n_embd_v;
    if (n_embd == 0) return;

    ensure_gpu_layer(layer_idx, n_embd, is_key);

    auto & gl = m_gpu_layers[layer_idx];
    void * ref = is_key ? gl.k_ref : gl.v_ref;
    auto & init_vec = is_key ? gl.k_initialized : gl.v_initialized;
    auto & tokens_since_kf = is_key ? gl.k_tokens_since_kf : gl.v_tokens_since_kf;

    std::vector<char> init_raw(m_kv_size);
    for (uint32_t i = 0; i < m_kv_size; i++) init_raw[i] = init_vec[i] ? 1 : 0;

    int result = ggml_backend_cuda_delta_kv_filter(
        tensor, ref, (bool *)init_raw.data(), slot_indices,
        n_tokens, kf_interval, &tokens_since_kf);

    for (int t = 0; t < n_tokens; t++) {
        int64_t slot = slot_indices[t];
        if (slot >= 0 && slot < (int64_t)m_kv_size) {
            init_vec[slot] = 1;
        }
    }

    if (result > 0) {
        n_processed += result;
    }
#else
    // Fallback to CPU
    process_cpu(tensor, slot_indices, n_tokens, layer_idx, is_key);
#endif
}
