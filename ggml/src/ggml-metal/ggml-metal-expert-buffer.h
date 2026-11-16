#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

// Backend-private streaming allocator. The owner outlives all its buffers;
// callers synchronize every GPU user before freeing a buffer or the owner.
typedef struct ggml_metal_expert_pool * ggml_metal_expert_pool_t;
ggml_metal_expert_pool_t ggml_metal_expert_pool_new(ggml_metal_device_t dev);
void ggml_metal_expert_pool_free(ggml_metal_expert_pool_t pool);
ggml_metal_buffer_t ggml_metal_expert_buffer_new(ggml_metal_expert_pool_t pool, size_t size);
void ggml_metal_expert_buffer_free(ggml_metal_buffer_t buffer);

#ifdef __cplusplus
}
#endif
