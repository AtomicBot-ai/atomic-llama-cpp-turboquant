#include "delta-kv.cuh"
#include "ggml-common.h"
#include "ggml-cuda.h"

// ============================================================================
// Delta-Quantized KV Cache Kernels
//
// Concept: Instead of quantizing absolute KV values to Q4_0 (lossy),
// store periodic F16 "keyframes" and Q4_0 deltas from those keyframes.
// Since consecutive tokens produce similar hidden states, the deltas are
// small and quantize with much less error than absolute values.
//
// Write: delta = current_f32 - keyframe_f16, quantize delta as Q4_0
// Read:  reconstructed = keyframe_f16 + dequant_q4_0(delta)
// ============================================================================

#define DELTA_KV_BLOCK_SIZE 256

// ----------------------------------------------------------------------------
// Encode kernel: F32 values -> Q4_0 deltas + F16 keyframes
// ----------------------------------------------------------------------------
// Each thread handles one Q4_0 block (32 elements) for one token.
// For keyframe slots: write F16 keyframe, store zero delta.
// For delta slots: read existing keyframe, compute delta, quantize as Q4_0.

__global__ void k_delta_kv_encode(
    const float    * __restrict__ src_f32,
    const int64_t  * __restrict__ idxs,
    half           * __restrict__ keyframes,
    block_q4_0     * __restrict__ deltas,
    const int64_t  n_embd,
    const int64_t  n_tokens,
    const int64_t  kv_size,
    const int64_t  nb_src_row,     // in floats
    const int64_t  nb_kf_row,      // in halfs
    const int64_t  nb_delta_row,   // in block_q4_0 units
    const int      kf_interval)
{
    const int64_t tid = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

    // Each thread handles one Q4_0 block = 32 elements
    const int64_t blocks_per_row = n_embd / QK4_0;
    const int64_t total_blocks = n_tokens * blocks_per_row;

    if (tid >= total_blocks) {
        return;
    }

    const int64_t token_idx = tid / blocks_per_row;
    const int64_t block_idx = tid % blocks_per_row;
    const int64_t elem_offset = block_idx * QK4_0;

    const int64_t dst_slot = idxs[token_idx];

    // Determine if this is a keyframe slot
    const bool is_keyframe = (kf_interval <= 0) || (dst_slot % kf_interval == 0);

    // Pointers to source row and destination keyframe/delta rows
    const float * src_row = src_f32 + token_idx * nb_src_row + elem_offset;
    half * kf_row = keyframes + dst_slot * nb_kf_row + elem_offset;
    block_q4_0 * delta_block = deltas + dst_slot * nb_delta_row + block_idx;

    if (is_keyframe) {
        // Keyframe: store F16 values, store zero delta
        for (int j = 0; j < QK4_0; ++j) {
            kf_row[j] = __float2half(src_row[j]);
        }

        // Zero delta block
        delta_block->d = __float2half(0.0f);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            delta_block->qs[j] = 0x88; // 8 in each nibble = zero point for Q4_0
        }
    } else {
        // Delta: compute difference from keyframe, quantize as Q4_0
        // Also update keyframe to current value for better quality on next token
        float vals[QK4_0];
        float amax = 0.0f;
        float vmax = 0.0f;

        for (int j = 0; j < QK4_0; ++j) {
            const float kf_val = __half2float(kf_row[j]);
            const float delta = src_row[j] - kf_val;
            vals[j] = delta;
            if (fabsf(delta) > amax) {
                amax = fabsf(delta);
                vmax = delta;
            }
        }

        // Quantize delta as Q4_0 and update keyframe to reconstructed value.
        // The keyframe must be updated to keyframe + dequant(delta), NOT the
        // original source, because the decoder reconstructs from that same sum.
        // Using the original source would cause encoder/decoder drift since
        // dequant(quant(delta)) != delta due to Q4_0 quantization error.
        const float d = vmax / -8.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;

        delta_block->d = __float2half(d);

        for (int j = 0; j < QK4_0 / 2; ++j) {
            const float x0 = vals[j] * id;
            const float x1 = vals[j + QK4_0 / 2] * id;

            const uint8_t xi0 = min(15, (int)(x0 + 8.5f));
            const uint8_t xi1 = min(15, (int)(x1 + 8.5f));

            delta_block->qs[j] = xi0 | (xi1 << 4);

            // Update keyframe to reconstructed value (= keyframe + dequant(delta))
            const float dq0 = d * ((float)xi0 - 8.0f);
            const float dq1 = d * ((float)xi1 - 8.0f);
            kf_row[j]              = __float2half(__half2float(kf_row[j])              + dq0);
            kf_row[j + QK4_0 / 2] = __float2half(__half2float(kf_row[j + QK4_0 / 2]) + dq1);
        }
    }
}

// ----------------------------------------------------------------------------
// Reconstruct kernel: Q4_0 deltas + F16 keyframes -> F16 output
// ----------------------------------------------------------------------------
// Each thread handles one Q4_0 block = 32 elements.

__global__ void k_delta_kv_reconstruct(
    const block_q4_0 * __restrict__ deltas,
    const half       * __restrict__ keyframes,
    half             * __restrict__ dst,
    const int64_t    n_embd,
    const int64_t    n_kv,
    const int64_t    nb_delta_row,  // in block_q4_0 units
    const int64_t    nb_kf_row,     // in halfs
    const int64_t    nb_dst_row)    // in halfs
{
    const int64_t tid = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

    const int64_t blocks_per_row = n_embd / QK4_0;
    const int64_t total_blocks = n_kv * blocks_per_row;

    if (tid >= total_blocks) {
        return;
    }

    const int64_t row_idx = tid / blocks_per_row;
    const int64_t block_idx = tid % blocks_per_row;
    const int64_t elem_offset = block_idx * QK4_0;

    const block_q4_0 * delta_block = deltas + row_idx * nb_delta_row + block_idx;
    const half * kf_row = keyframes + row_idx * nb_kf_row + elem_offset;
    half * dst_row = dst + row_idx * nb_dst_row + elem_offset;

    const float d = __half2float(delta_block->d);

    // Reconstruct: keyframe + dequant(delta)
    for (int j = 0; j < QK4_0 / 2; ++j) {
        const uint8_t packed = delta_block->qs[j];

        const int vi0 = (packed & 0x0F);
        const int vi1 = (packed >> 4);

        const float delta0 = d * ((float)vi0 - 8.0f);
        const float delta1 = d * ((float)vi1 - 8.0f);

        const float kf0 = __half2float(kf_row[j]);
        const float kf1 = __half2float(kf_row[j + QK4_0 / 2]);

        dst_row[j]              = __float2half(kf0 + delta0);
        dst_row[j + QK4_0 / 2] = __float2half(kf1 + delta1);
    }
}

// ============================================================================
// Host dispatch functions
// ============================================================================

void ggml_cuda_delta_kv_encode(
    const float * src_f32,
    const int64_t * idxs,
    half * keyframes,
    void * deltas,
    const int64_t n_embd,
    const int64_t n_tokens,
    const int64_t kv_size,
    const int64_t nb_src_row,
    const int64_t nb_kf_row,
    const int64_t nb_delta_row,
    const int     kf_interval,
    cudaStream_t stream)
{
    const int64_t blocks_per_row = n_embd / QK4_0;
    const int64_t total_blocks = n_tokens * blocks_per_row;

    if (total_blocks == 0) return;

    const int num_cuda_blocks = (total_blocks + DELTA_KV_BLOCK_SIZE - 1) / DELTA_KV_BLOCK_SIZE;

    k_delta_kv_encode<<<num_cuda_blocks, DELTA_KV_BLOCK_SIZE, 0, stream>>>(
        src_f32, idxs, keyframes, (block_q4_0 *)deltas,
        n_embd, n_tokens, kv_size,
        nb_src_row / sizeof(float),
        nb_kf_row / sizeof(half),
        nb_delta_row / sizeof(block_q4_0),
        kf_interval);
}

void ggml_cuda_delta_kv_reconstruct(
    const void * deltas,
    const half * keyframes,
    half * dst,
    const int64_t n_embd,
    const int64_t n_kv,
    const int64_t nb_delta_row,
    const int64_t nb_kf_row,
    const int64_t nb_dst_row,
    cudaStream_t stream)
{
    const int64_t blocks_per_row = n_embd / QK4_0;
    const int64_t total_blocks = n_kv * blocks_per_row;

    if (total_blocks == 0) return;

    const int num_cuda_blocks = (total_blocks + DELTA_KV_BLOCK_SIZE - 1) / DELTA_KV_BLOCK_SIZE;

    k_delta_kv_reconstruct<<<num_cuda_blocks, DELTA_KV_BLOCK_SIZE, 0, stream>>>(
        (const block_q4_0 *)deltas, keyframes, dst,
        n_embd, n_kv,
        nb_delta_row / sizeof(block_q4_0),
        nb_kf_row / sizeof(half),
        nb_dst_row / sizeof(half));
}

// ============================================================================
// In-place delta filter kernel (Phase 2)
//
// Operates entirely on GPU memory. For each token:
// 1. Read F16 from KV cache at slot[i]
// 2. Read F16 reference from ref_buf at slot[i]
// 3. Compute delta, quantize Q4_0, dequantize (lossy)
// 4. Reconstruct = ref + dequant(delta)
// 5. Write reconstructed back to KV cache AND ref_buf
//
// For keyframe slots: just copy KV → ref (no lossy step)
// ============================================================================

__global__ void k_delta_kv_filter_inplace(
    half          * __restrict__ kv_data,     // KV cache (read/write)
    half          * __restrict__ ref_data,    // reference buffer (read/write)
    const int64_t * __restrict__ slot_indices,// which slots were written (GPU)
    const bool    * __restrict__ is_keyframe, // per-token keyframe flag (GPU)
    const int64_t n_embd,
    const int64_t n_tokens,
    const int64_t kv_row_stride,   // in halfs
    const int64_t ref_row_stride)  // in halfs
{
    const int64_t tid = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;
    const int64_t blocks_per_row = n_embd / QK4_0;
    const int64_t total_blocks = n_tokens * blocks_per_row;

    if (tid >= total_blocks) return;

    const int64_t token_idx = tid / blocks_per_row;
    const int64_t block_idx = tid % blocks_per_row;
    const int64_t elem_offset = block_idx * QK4_0;
    const int64_t slot = slot_indices[token_idx];

    half * kv_row  = kv_data  + slot * kv_row_stride  + elem_offset;
    half * ref_row = ref_data + slot * ref_row_stride + elem_offset;

    if (is_keyframe[token_idx]) {
        // Keyframe: copy current values to reference (no lossy step)
        for (int j = 0; j < QK4_0; ++j) {
            ref_row[j] = kv_row[j];
        }
        return;
    }

    // Delta compression: compute delta from reference, Q4_0 round-trip
    float vals[QK4_0];
    float amax = 0.0f, vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        float cur = __half2float(kv_row[j]);
        float ref = __half2float(ref_row[j]);
        float delta = cur - ref;
        vals[j] = delta;
        if (fabsf(delta) > amax) {
            amax = fabsf(delta);
            vmax = delta;
        }
    }

    // Q4_0 quantize
    const float d = vmax / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;

    // Q4_0 dequantize and reconstruct in one pass
    for (int j = 0; j < QK4_0 / 2; ++j) {
        const float x0 = vals[j] * id;
        const float x1 = vals[j + QK4_0 / 2] * id;

        const uint8_t xi0 = min(15, (int)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int)(x1 + 8.5f));

        // Dequantize
        const float dq0 = d * ((float)xi0 - 8.0f);
        const float dq1 = d * ((float)xi1 - 8.0f);

        // Reconstruct = ref + dequant(delta)
        const float r0 = __half2float(ref_row[j]) + dq0;
        const float r1 = __half2float(ref_row[j + QK4_0 / 2]) + dq1;

        const half h0 = __float2half(r0);
        const half h1 = __float2half(r1);

        // Write back to both KV cache and reference
        kv_row[j]              = h0;
        kv_row[j + QK4_0 / 2] = h1;
        ref_row[j]              = h0;
        ref_row[j + QK4_0 / 2] = h1;
    }
}

// ============================================================================
// Public API: GPU-native delta KV filter
// ============================================================================

int ggml_backend_cuda_delta_kv_filter(
    ggml_tensor * kv_tensor,
    void * ref_buf,
    bool * ref_initialized,
    const int64_t * slot_indices,
    int n_tokens,
    int kf_interval,
    int * tokens_since_keyframe)
{
    if (!kv_tensor || !ref_buf || n_tokens == 0) return 0;
    if (kv_tensor->type != GGML_TYPE_F16) return -1;

    const int64_t n_embd = kv_tensor->ne[0];
    const int64_t kv_size = kv_tensor->ne[1];
    const int64_t kv_row_stride = kv_tensor->nb[1] / sizeof(half);
    const int64_t ref_row_stride = n_embd; // ref is packed

    // Determine keyframe flags (use char instead of bool for GPU compat)
    std::vector<char> kf_flags_raw(n_tokens);
    for (int t = 0; t < n_tokens; t++) {
        int64_t slot = slot_indices[t];
        bool is_kf = !ref_initialized[slot];
        if (kf_interval > 0 && tokens_since_keyframe) {
            (*tokens_since_keyframe)++;
            if (*tokens_since_keyframe >= kf_interval) {
                is_kf = true;
                *tokens_since_keyframe = 0;
            }
        }
        kf_flags_raw[t] = is_kf ? 1 : 0;
        ref_initialized[slot] = true;
    }

    // Upload slot_indices and keyframe flags to GPU
    int64_t * d_indices;
    bool    * d_kf_flags;
    CUDA_CHECK(cudaMalloc(&d_indices,  n_tokens * sizeof(int64_t)));
    CUDA_CHECK(cudaMalloc(&d_kf_flags, n_tokens * sizeof(bool)));
    CUDA_CHECK(cudaMemcpy(d_indices,  slot_indices, n_tokens * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kf_flags, kf_flags_raw.data(), n_tokens * sizeof(bool), cudaMemcpyHostToDevice));

    // Launch kernel
    const int64_t blocks_per_row = n_embd / QK4_0;
    const int64_t total_blocks = n_tokens * blocks_per_row;
    const int cuda_blocks = (total_blocks + DELTA_KV_BLOCK_SIZE - 1) / DELTA_KV_BLOCK_SIZE;

    cudaStream_t stream = nullptr; // default stream
    k_delta_kv_filter_inplace<<<cuda_blocks, DELTA_KV_BLOCK_SIZE, 0, stream>>>(
        (half *)kv_tensor->data,
        (half *)ref_buf,
        d_indices,
        d_kf_flags,
        n_embd, n_tokens,
        kv_row_stride, ref_row_stride);

    CUDA_CHECK(cudaStreamSynchronize(stream));

    cudaFree(d_indices);
    cudaFree(d_kf_flags);

    return n_tokens;
}

void * ggml_backend_cuda_delta_kv_alloc_ref(int device, size_t n_bytes) {
    void * ptr = nullptr;
    ggml_cuda_set_device(device);
    CUDA_CHECK(cudaMalloc(&ptr, n_bytes));
    CUDA_CHECK(cudaMemset(ptr, 0, n_bytes));
    return ptr;
}

void ggml_backend_cuda_delta_kv_free_ref(void * ptr) {
    if (ptr) {
        cudaFree(ptr);
    }
}
