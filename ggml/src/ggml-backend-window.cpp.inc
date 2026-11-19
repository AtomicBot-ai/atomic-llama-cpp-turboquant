// Included after ggml_backend_sched's private definition. Storage contract at
// 31953bcf74ba489a8bbb0c936c5fcf24fcf9485c: track scheduler construction, input
// copies, allocator dependencies and split-array writes on upstream updates.
#include "ggml-backend-window.h"
#include <new>

extern "C" ggml_gallocr_t ggml_gallocr_window_new(ggml_backend_buffer_type_t * bufts, int n_bufts,
        ggml_backend_buffer_t scratch);
extern "C" bool ggml_gallocr_window_reserve_like(ggml_gallocr_t target, ggml_gallocr_t origin);

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

static bool window_route_pause(const ggml_tensor * node) {
    return node->op == GGML_OP_CUSTOM && strncmp(node->name, "ffn_moe_down-", 13) == 0;
}

static bool window_layer_pause(ggml_backend_sched_t sched, const ggml_tensor * node) {
    const char * prefix = sched->window_fence_prefix ? sched->window_fence_prefix : "l_last-";
    return strncmp(node->name, prefix, strlen(prefix)) == 0;
}

static void window_classify_scratch(ggml_backend_sched_t sched, const std::vector<int> & phases) {
    const auto & graph = sched->graph;
    GGML_ASSERT(phases.size() == size_t(graph.n_nodes));
    std::unordered_map<const ggml_tensor *, int> lifetime;
    lifetime.reserve(graph.n_nodes + graph.n_leafs);
    auto touch = [&](const ggml_tensor * t, int phase) {
        if (!t) { return; }
        while (t->view_src) {
            if (t->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) { phase = -1; }
            t = t->view_src;
        }
        if (t->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) { phase = -1; }
        if (t->data) { return; }
        auto [entry, inserted] = lifetime.emplace(t, phase);
        if (!inserted && entry->second != phase) { entry->second = -1; }
    };
    for (int i = 0; i < graph.n_leafs; ++i) { touch(graph.leafs[i], -1); }
    for (int i = 0; i < graph.n_nodes; ++i) {
        auto * node = graph.nodes[i];
        const bool handoff = window_route_pause(node);
        touch(node, handoff || window_layer_pause(sched, node) ? -1 : phases[i]);
        for (const auto * src : node->src) { touch(src, handoff ? -1 : phases[i]); }
    }
    if (sched->window_batch_plan) {
        for (int i = 0; i < sched->n_splits; ++i) {
            const auto & split = sched->splits[i];
            const bool route = std::any_of(split.graph.nodes, split.graph.nodes + split.graph.n_nodes, window_route_pause);
            for (int j = 0; j < split.n_inputs; ++j) {
                auto * input = split.inputs[j];
                if (route) { touch(input, -1); }
                if (split.backend_id == 0) {
                    const size_t id = ggml_hash_find(&sched->hash_set, input);
                    touch(sched->hv_tensor_copies[id * sched->n_backends], -1); // backend 0, copy 0
                }
            }
        }
    }
    for (int i = 0; i < graph.n_nodes; ++i) {
        const auto * node = graph.nodes[i];
        if (node->view_src || sched->node_backend_ids[i] != 0) { continue; }
        const auto entry = lifetime.find(node);
        if (entry != lifetime.end() && entry->second >= 0) { sched->node_backend_ids[i] = sched->n_backends; }
    }
}

}

bool ggml_backend_window_share_scratch(ggml_backend_sched_t sched, ggml_backend_buffer_t scratch) {
    if (!sched || !sched->window_node_capacity || sched->window_shared_scratch || sched->graph.n_nodes ||
            !scratch || ggml_backend_buffer_get_type(scratch) != sched->bufts[0]) { return false; }
    ggml_backend_buffer_type_t bufts[4];
    for (int i = 0; i < sched->n_backends; ++i) { bufts[i] = sched->bufts[i]; }
    bufts[sched->n_backends] = sched->bufts[0];
    auto * alloc = ggml_gallocr_window_new(bufts, sched->n_backends + 1, scratch);
    if (!alloc) { return false; }
    ggml_gallocr_free(sched->galloc);
    sched->galloc = alloc;
    sched->window_shared_scratch = true;
    sched->window_scratch_buffer = scratch;
    return true;
}

void ggml_backend_window_batch_plan(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched->window_node_capacity && sched->n_copies == 1 && sched->n_backends == 2);
    const char * direct_upload = getenv("LLAMA_EXPERIMENTAL_PREFILL_DIRECT_FFN_UPLOAD");
    if (direct_upload && strcmp(direct_upload, "1") == 0) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(sched->backends[0]));
        sched->window_metal_host_base = reinterpret_cast<decltype(sched->window_metal_host_base)>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_buffer_get_host_base"));
    }
    sched->window_batch_plan = true;
}

bool ggml_backend_window_batch_init(ggml_window_batch * batch, ggml_backend_t metal) {
    auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(metal));
    *batch = {};
    batch->metal = metal;
    batch->append = reinterpret_cast<decltype(batch->append)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_window_append"));
    batch->flush = reinterpret_cast<decltype(batch->flush)>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_window_flush"));
    return batch->append && batch->flush;
}

static ggml_backend_sched_t window_sched_new(ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts, int count, size_t graph_size, bool offload, bool compact) {
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
    const size_t nodes = compact ? graph_size + graph_size / 2 : graph_size * 8;
    // Cross-backend copies and allocation dependency views are both checked
    // before they are created. Graphs outside the compact arena fail admission.
    const size_t arena_tensors = compact ? graph_size / 2 : graph_size * 6;
    sched->context_buffer_size = arena_tensors * ggml_tensor_overhead() +
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

ggml_backend_sched_t ggml_backend_window_new(ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts, int count, size_t graph_size, bool offload) {
    return window_sched_new(backends, bufts, count, graph_size, offload, false);
}

ggml_backend_sched_t ggml_backend_window_new_compact(ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts, int count, size_t graph_size, bool offload) {
    return window_sched_new(backends, bufts, count, graph_size, offload, true);
}

bool ggml_backend_window_alloc(ggml_backend_sched_t sched, ggml_cgraph * graph, size_t graph_size) {
    if (!sched || !graph || graph_size != 8192 || sched->n_copies != 1 || sched->n_backends > 3 ||
            size_t(graph->n_nodes) + graph->n_leafs > graph_size ||
            !sched->window_node_capacity) { return false; }
    try {
        return ggml_backend_sched_alloc_graph(sched, graph);
    } catch (const window_capacity_error &) {
        return false; // caller must discard this partially prepared graph
    }
}

bool ggml_backend_window_reserve_like(ggml_backend_sched_t target, ggml_backend_sched_t origin) {
    return target && origin && target->window_batch_plan && origin->window_batch_plan &&
           target->n_backends == origin->n_backends && target->n_splits == 0 &&
           target->window_scratch_buffer == origin->window_scratch_buffer &&
           ggml_gallocr_window_reserve_like(target->galloc, origin->galloc);
}

void ggml_backend_window_set_before(ggml_backend_sched_t sched,
        ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched->n_copies == 1);
    sched->window_before = callback;
    sched->window_before_data = user_data;
}

void ggml_backend_window_set_fence(ggml_backend_sched_t sched, const char * prefix) {
    GGML_ASSERT(sched->window_node_capacity);
    sched->window_fence_prefix = prefix;
}
