#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// Delta KV cache: in-place delta compression on GPU KV cache tensors
// Applies lossy delta compression filter: for each slot, computes delta from
// reference, quantizes to Q4_0, dequantizes, and writes back to KV cache.
// ref_buf: GPU buffer holding previous reconstructed values (F16, same layout as tensor)
// ref_initialized: CPU-side flag array [kv_size] tracking which slots have references
// slot_indices: CPU-side array of slot indices that were just written
// kf_interval: keyframe interval (0 = all keyframes)
// Returns: number of tokens processed, or -1 on error
GGML_BACKEND_API int ggml_backend_cuda_delta_kv_filter(
    struct ggml_tensor * kv_tensor,
    void * ref_buf,
    bool * ref_initialized,
    const int64_t * slot_indices,
    int n_tokens,
    int kf_interval,
    int * tokens_since_keyframe);

GGML_BACKEND_API void * ggml_backend_cuda_delta_kv_alloc_ref(int device, size_t n_bytes);
GGML_BACKEND_API void   ggml_backend_cuda_delta_kv_free_ref(void * ptr);

#ifdef  __cplusplus
}
#endif
