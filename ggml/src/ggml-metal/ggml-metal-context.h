#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// backend context
//

typedef struct ggml_metal * ggml_metal_t;

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev);
void ggml_metal_free(ggml_metal_t ctx);

const char * ggml_metal_get_name(ggml_metal_t ctx);

void ggml_metal_synchronize(ggml_metal_t ctx);
// Diagnostic only: query after synchronization, before the next graph submission.
bool ggml_metal_last_graph_timing(ggml_metal_t ctx, uint64_t * gpu_ns, uint64_t * buffers);

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);
void ggml_metal_set_host_expert_bridge(ggml_metal_t ctx, bool (*callback)(struct ggml_tensor * node));
bool ggml_metal_host_expert_window_active(ggml_metal_t ctx);
enum ggml_status ggml_metal_host_expert_window_append(ggml_metal_t ctx, struct ggml_cgraph * gf,
        void (*release)(void *), void * user);
enum ggml_status ggml_metal_window_append(ggml_metal_t ctx, struct ggml_cgraph * gf);
enum ggml_status ggml_metal_window_flush(ggml_metal_t ctx, uint64_t * submissions);
// Optional prefix fence. Both graph segments retain normal queue ordering.
enum ggml_status ggml_metal_graph_compute_prefix(ggml_metal_t ctx, struct ggml_cgraph * gf, int prefix_nodes);
bool             ggml_metal_wait_graph_prefix(ggml_metal_t ctx);
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
void ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
