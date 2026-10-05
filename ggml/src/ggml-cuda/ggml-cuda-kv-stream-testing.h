#pragma once

// Accessors the CUDA KV-streaming tests use to inspect and drive a runtime
// directly. llama itself never calls these, so they stay out of the installed
// ggml-cuda.h and off the backend's public ABI.

#include "ggml-cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_backend_cuda_kv_stream_stats {
    uint64_t resident_hits;
    uint64_t resident_misses;
    uint64_t streamed_pages;
    uint64_t host_to_device_bytes;
    uint64_t resident_attention_spans;
    uint64_t resident_pages_attended;
    uint64_t streamed_attention_spans;
    uint64_t streamed_pages_attended;
    uint64_t mma_prefill_attention_spans;
    uint64_t asynchronous_page_uploads;
    uint64_t host_to_device_copy_commands;
    uint64_t compute_stream_waits;
    uint64_t stage_slot_reuses;
    uint64_t cross_layer_prefetches;
    uint64_t deadline_samples;
    uint64_t deadline_misses;
    uint32_t ring_peak_occupancy;
    uint64_t staged_set_rows;
    uint64_t staged_set_rows_bytes;
};

GGML_BACKEND_API size_t ggml_backend_cuda_kv_stream_stage_bytes(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API uint32_t ggml_backend_cuda_kv_stream_stage_slots(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API uint32_t ggml_backend_cuda_kv_stream_resident_pages_per_layer(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API size_t ggml_backend_cuda_kv_stream_pool_bytes(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API struct ggml_backend_cuda_kv_stream_stats ggml_backend_cuda_kv_stream_get_stats(
    ggml_backend_cuda_kv_stream_runtime_t runtime);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_stage_upload(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    uint32_t slot,
    size_t offset,
    const void * source,
    size_t size);
GGML_BACKEND_API bool ggml_backend_cuda_kv_stream_stage_download(
    ggml_backend_cuda_kv_stream_runtime_t runtime,
    uint32_t slot,
    size_t offset,
    void * destination,
    size_t size);

#ifdef __cplusplus
}
#endif
