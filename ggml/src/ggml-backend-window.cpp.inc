// Included after ggml_backend_sched's private definition. Storage contract at
// 31953bcf74ba489a8bbb0c936c5fcf24fcf9485c: track scheduler construction, input
// copies, allocator dependencies and split-array writes on upstream updates.
#include "ggml-backend-window.h"
#include <new>

namespace {
struct window_capacity_error {};
// Called once after split optimization has established exact copy/dependency
// counts, before the allocator graph's arrays or dependency views are written.
static void window_check_graph_capacity(ggml_backend_sched_t sched, size_t nodes, size_t views) {
    if (!sched->window_node_capacity) { return; }
    const size_t used = ggml_used_mem(sched->ctx);
    if (nodes > sched->window_node_capacity || used > sched->context_buffer_size ||
            views > (sched->context_buffer_size - used) / ggml_tensor_overhead()) {
        throw window_capacity_error{};
    }
}
}

ggml_backend_sched_t ggml_backend_window_new(ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts, int count, size_t graph_size, bool offload) {
    if (count < 1 || count > 3 || graph_size != 8192) { return nullptr; }
    auto * sched = ggml_backend_sched_new(backends, bufts, count, graph_size, false, offload);
    if (!sched) { return nullptr; }
    // No graph has used this arena. Free the pessimistic upstream capacities
    // before allocating replacements; do not relocate any live tensor metadata.
    free(sched->context_buffer);
    sched->context_buffer = nullptr;
    free(sched->node_backend_ids); sched->node_backend_ids = nullptr;
    free(sched->leaf_backend_ids); sched->leaf_backend_ids = nullptr;
    free(sched->prev_node_backend_ids); sched->prev_node_backend_ids = nullptr;
    free(sched->prev_leaf_backend_ids); sched->prev_leaf_backend_ids = nullptr;
    const size_t nodes = graph_size * 8;
    // Before the guard, at most two backend copies per original tensor exist.
    // The remaining arena is for input/dependency views, checked exactly above.
    sched->context_buffer_size = graph_size * 6 * ggml_tensor_overhead() +
            ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = static_cast<char *>(malloc(sched->context_buffer_size));
    sched->node_backend_ids = static_cast<int *>(calloc(nodes, sizeof(int)));
    sched->leaf_backend_ids = static_cast<int *>(calloc(nodes, sizeof(int)));
    sched->prev_node_backend_ids = static_cast<int *>(calloc(nodes, sizeof(int)));
    sched->prev_leaf_backend_ids = static_cast<int *>(calloc(nodes, sizeof(int)));
    if (!sched->context_buffer || !sched->node_backend_ids || !sched->leaf_backend_ids ||
            !sched->prev_node_backend_ids || !sched->prev_leaf_backend_ids) {
        ggml_backend_sched_free(sched);
        return nullptr;
    }
    sched->window_node_capacity = nodes;
    return sched;
}

bool ggml_backend_window_alloc(ggml_backend_sched_t sched, ggml_cgraph * graph, size_t graph_size) {
    if (!sched || !graph || graph_size != 8192 || sched->n_copies != 1 || sched->n_backends > 3 ||
            size_t(graph->n_nodes) + graph->n_leafs > graph_size ||
            sched->window_node_capacity != graph_size * 8) { return false; }
    try {
        return ggml_backend_sched_alloc_graph(sched, graph);
    } catch (const window_capacity_error &) {
        return false; // caller must discard this partially prepared graph
    }
}
