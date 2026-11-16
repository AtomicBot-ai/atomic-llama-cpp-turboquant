#pragma once
#include "ggml-backend.h"

// Fork-private, single-copy scheduler storage for admitted Metal/CPU windows.
// The caller covers transient construction and retained host storage separately.
#ifdef __cplusplus
extern "C" {
#endif
GGML_BACKEND_API ggml_backend_sched_t ggml_backend_window_new(
        ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts,
        int n_backends, size_t graph_size, bool op_offload);
GGML_BACKEND_API bool ggml_backend_window_alloc(ggml_backend_sched_t sched,
        struct ggml_cgraph * graph, size_t graph_size);
#ifdef __cplusplus
}
#endif
