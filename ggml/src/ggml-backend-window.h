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
GGML_BACKEND_API ggml_backend_sched_t ggml_backend_window_new_compact(
        ggml_backend_t * backends, ggml_backend_buffer_type_t * bufts,
        int n_backends, size_t graph_size, bool op_offload);
GGML_BACKEND_API bool ggml_backend_window_alloc(ggml_backend_sched_t sched,
        struct ggml_cgraph * graph, size_t graph_size);
// Give a fresh worker the private buffer capacities of a prepared worker.
// Failure leaves the target ready for the ordinary graph reservation path.
GGML_BACKEND_API bool ggml_backend_window_reserve_like(ggml_backend_sched_t target, ggml_backend_sched_t origin);
// Borrow one Metal scratch buffer for tensors whose lifetimes do not cross a
// routing or layer-output boundary. The buffer outlives all window schedulers.
GGML_BACKEND_API bool ggml_backend_window_share_scratch(ggml_backend_sched_t sched, ggml_backend_buffer_t scratch);
// Cursor for the fork's two-backend, layer-major routed graph.
struct ggml_window_cursor {
    int split, node;
    bool copied;
};
enum ggml_window_step { GGML_WINDOW_ROUTE_FRONTIER, GGML_WINDOW_ROUTES, GGML_WINDOW_LAYER_END, GGML_WINDOW_FINISH, GGML_WINDOW_DENSE_LAYER_END, GGML_WINDOW_EXACT_LAYER_END };
struct ggml_window_batch {
    ggml_backend_t metal;
    enum ggml_status (*append)(ggml_backend_t, struct ggml_cgraph *);
    enum ggml_status (*flush)(ggml_backend_t, uint64_t *);
    uint64_t submissions;
};
GGML_BACKEND_API bool ggml_backend_window_batch_init(struct ggml_window_batch * batch, ggml_backend_t metal);
GGML_BACKEND_API void ggml_backend_window_batch_plan(ggml_backend_sched_t sched);
// For EXACT_LAYER_END, route points to the graph's actual layer output.
GGML_BACKEND_API enum ggml_status ggml_backend_window_batch_step(ggml_backend_sched_t sched,
        struct ggml_window_cursor * cursor, struct ggml_window_batch * batch, enum ggml_window_step step,
        struct ggml_tensor ** route);
// The predicate is called with ask=true; execution follows with ask=false
// after synchronizing the node's backend, before any node in its subgraph runs.
GGML_BACKEND_API void ggml_backend_window_set_before(ggml_backend_sched_t sched,
        ggml_backend_sched_eval_callback callback, void * user_data);
// Keep externally observed layer boundaries fixed during backend optimization.
GGML_BACKEND_API void ggml_backend_window_set_fence(ggml_backend_sched_t sched, const char * prefix);
#ifdef __cplusplus
}
#endif
