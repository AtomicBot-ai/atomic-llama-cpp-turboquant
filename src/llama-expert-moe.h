#pragma once

#include "llama-expert-lookahead.h"
#include "llama-expert-publication.h"
#include "llama-expert-route-stage.h"
#include "llama-memory-contract.h"
#include "ggml-cpp.h"
#include "../ggml/src/ggml-impl.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

// Fork-owned short-row Metal MoE. This is one graph per expert group, using
// upstream FP32 vector matmuls and SwiGLU, not a different quantization kernel.
// Keep its graph equivalent to the unclamped separate-projection SILU branch
// in llm_graph_context::build_moe_ffn when updating upstream.
class llama_expert_moe {
    static constexpr size_t groups = 16, routes_max = 8 * 32, layers_max = 128;
    static constexpr size_t window_groups = 64;
    static constexpr size_t graph_nodes = routes_max * 8 + 16;
    static constexpr size_t matrix_route_bytes = 2U << 20;
    static constexpr size_t matrix_tensor_bytes = 64U << 20;

    struct layer {
        llama_expert_moe * owner = nullptr;
        const ggml_tensor * up = nullptr;
        const ggml_tensor * gate = nullptr;
        const ggml_tensor * down = nullptr;
        uint64_t publication_generation = 0;
    };
    struct group {
        size_t n = 0;
        std::array<int64_t, groups> experts{};
        std::array<llama_expert_store::lease, groups> up, gate, down;
    };

    std::shared_ptr<llama_memory_budget> budget;
    ggml_backend_t backend;
    llama_expert_store & store;
    std::atomic<bool> & failed;
    char * failure;
    size_t failure_size;
    llama_expert_profile_data * & profile;
    uint64_t & projections, & matmuls, & submissions;
    size_t max_allocation, max_payload;
    // Tickets precede their resources. Every scratch tensor is explicitly bound
    // inside buffer; the inner graph must never allocate through a scheduler.
    llama_memory_budget::ticket metadata_charge, scratch_charge;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    llama_memory_budget::ticket matrix_charge;
    ggml_backend_buffer_ptr matrix_buffer;
    std::array<layer, layers_max> layers{};
    size_t n_layers = 0;
    size_t cursor = 0;
    std::array<int32_t, routes_max> selected{};
    uint64_t calls = 0;
    using graph_timing_t = bool (*)(ggml_backend_t, uint64_t *, uint64_t *);
    graph_timing_t graph_timing = nullptr;
    bool pipeline = false;
    bool queued_demand = false;
    bool reuse_plan = false;
    bool batch_routes = false;
    using addresses_t = bool (*)(ggml_tensor *, ggml_tensor *);
    addresses_t set_addresses = nullptr;
    llama_memory_budget::ticket readers_charge;
    std::unique_ptr<early_io::pool> readers;
    size_t candidate_limit = 0;
    // Two overlapping demand batches, including task descriptors and allocator overhead.
    static constexpr size_t queue_headroom = 64 * 1024;

    size_t demand_min_ready = 4;
    bool flash_next_verify = false;
    bool metal_predictor = false;
    bool graph_predictor = false;
    using compute_prefix_t = ggml_status (*)(ggml_backend_t, ggml_cgraph *, int);
    using wait_prefix_t = bool (*)(ggml_backend_t);
    compute_prefix_t compute_prefix = nullptr;
    wait_prefix_t wait_prefix = nullptr;
    llama_memory_budget::ticket predictor_meta_charge, predictor_scratch_charge;
    ggml_context_ptr predictor_ctx;
    ggml_backend_buffer_ptr predictor_buffer;
    std::unique_ptr<llama_expert_lookahead> lookahead;
    uint64_t pipeline_groups = 0, resident_groups = 0;
    const ggml_tensor * window_ready = nullptr;

    llama_memory_budget::ticket publication_charge;
    std::unique_ptr<llama_expert_publication> publication;
    std::unique_ptr<llama_expert_route_stage> route_stage;
    std::unique_ptr<llama_expert_store::demand_batch> published_demand;
    std::unique_ptr<group> published_resident;
    std::array<int64_t, groups> published_experts{};
    size_t published_count = 0;
    const layer * published_layer = nullptr;
    bool shared_overlap = false, single_row_prediction = false;
    bool verify_overlap = false;
    int publication_layer = 0; // bounded one-layer prototype unless explicitly "all"
    uint64_t graph_generation = 0;

    static void prepare_publication(void * user, const int32_t * ids, size_t count) {
        const auto & p = *static_cast<layer *>(user);
        auto & self = *p.owner;
        std::array<int64_t, GGML_METAL_ROUTE_MAX_IDS> routes;
        for (size_t i = 0; i < count; ++i) {
            if (ids[i] < 0 || ids[i] >= p.up->ne[2]) throw std::runtime_error("invalid published expert route");
            routes[i] = ids[i];
        }
        std::sort(routes.begin(), routes.begin() + count);
        const size_t distinct = std::unique(routes.begin(), routes.begin() + count) - routes.begin();
        self.publication->mark_routes();
        if (self.route_stage) {
            self.route_stage->prepare({p.up->name, p.gate->name, p.down->name}, routes.data(), distinct, *self.publication);
            return;
        }
        const auto memory = self.budget->snapshot(llama_memory_domain::host);
        const size_t fixed = memory.used() - self.store.storage_bytes();
        const size_t available = memory.requested > fixed + queue_headroom ? memory.requested - fixed - queue_headroom : 0;
        const size_t capacity = available > self.max_payload ? (available - self.max_payload) / (3 * self.max_allocation) : 0;
        if (capacity < 2 || available <= queue_headroom + 6 * self.max_allocation + self.max_payload) return;
        auto * ahead = self.lookahead && self.lookahead->active() ? self.lookahead.get() : nullptr;
        if (ahead && ahead->early()) ahead->prepare_early(p.up->name, routes.data(), distinct, self.profile);
        const size_t width = std::min(groups, capacity / 2);
        auto & hot = *self.published_resident;
        for (size_t i = 0; i < distinct && hot.n < width; ++i) {
            const auto id = routes[i];
            if (self.store.contains(p.up->name, id) && self.store.contains(p.gate->name, id) && self.store.contains(p.down->name, id)) {
                hot.experts[hot.n++] = id;
            }
        }
        if (hot.n && !(self.store.acquire_resident_many(p.up->name, hot.experts.data(), hot.n, hot.up.data()) &&
                self.store.acquire_resident_many(p.gate->name, hot.experts.data(), hot.n, hot.gate.data()) &&
                self.store.acquire_resident_many(p.down->name, hot.experts.data(), hot.n, hot.down.data()))) hot = {};
        self.publication->mark_residency();
        // Preserve resident-first execution and pin all resident projections
        // before admitting misses, exactly as on the existing demand pipeline.
        self.published_count = 0;
        for (size_t i = 0; i < distinct && self.published_count < width; ++i) {
            if (std::find(hot.experts.begin(), hot.experts.begin() + hot.n, routes[i]) == hot.experts.begin() + hot.n) {
                self.published_experts[self.published_count++] = routes[i];
            }
        }
        if (self.published_count) {
            self.published_demand = self.store.enqueue_demand({p.up->name, p.gate->name, p.down->name},
                    self.published_experts.data(), self.published_count, ahead, self.readers.get(), self.profile != nullptr);
        }
        self.published_layer = &p;
    }
    static void arm_publication(void * user, uint64_t sequence, uint64_t generation, size_t count) {
        auto & p = *static_cast<layer *>(user);
        if (p.owner->failed.load()) return; // later GPU layers may still finish after a latched I/O error
        auto & service = *p.owner->publication;
        if (p.owner->route_stage && service.pending()) {
            service.abandon();
            if (service.pending()) return;
        }
        if (p.owner->route_stage) p.owner->route_stage->begin();
        service.start(sequence, generation, count, prepare_publication, user, p.owner->profile != nullptr);
    }

    static size_t aligned(size_t size, size_t alignment = 65536) {
        return (size + alignment - 1) / alignment * alignment;
    }

    static void bind_to(ggml_tensor * tensor, ggml_backend_buffer_t buffer, size_t & cursor) {
        const auto buft = ggml_backend_buffer_get_type(buffer);
        const size_t bytes = ggml_backend_buft_get_alloc_size(buft, tensor);
        if (cursor > ggml_backend_buffer_get_size(buffer) ||
                bytes > ggml_backend_buffer_get_size(buffer) - cursor) {
            throw std::runtime_error("combined MoE scratch exceeded its reservation");
        }
        auto * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer));
        if (ggml_backend_tensor_alloc(buffer, tensor, base + cursor) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("cannot bind combined MoE tensor");
        }
        cursor += aligned(bytes, ggml_backend_buft_get_alignment(buft));
    }

    void bind(ggml_tensor * tensor) { bind_to(tensor, buffer.get(), cursor); }

    template<typename WeightAt>
    ggml_tensor * build_address_ffn(ggml_cgraph * graph, ggml_tensor * input, ggml_tensor * ids, size_t n, WeightAt weight_at) {
        const auto address_table = [&](size_t projection) {
            llama_expert_timer timer(profile ? &profile->moe_address_setup_ns : nullptr);
            auto * table = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, n);
            bind(table);
            std::array<uint64_t, window_groups> pointers;
            for (size_t i = 0; i < n; ++i) { pointers[i] = uintptr_t(weight_at(i, projection)); }
            ggml_backend_tensor_set(table, pointers.data(), 0, n * sizeof(uint64_t));
            return table;
        };
        auto * up = ggml_mul_mat_id(ctx.get(), weight_at(0, 0), input, ids);
        auto * gate = ggml_mul_mat_id(ctx.get(), weight_at(0, 1), input, ids);
        auto * act = ggml_swiglu_split(ctx.get(), gate, up);
        auto * down = ggml_mul_mat_id(ctx.get(), weight_at(0, 2), act, ids);
        if (!set_addresses(up, address_table(0)) || !set_addresses(gate, address_table(1)) ||
            !set_addresses(down, address_table(2))) {
            throw std::runtime_error("Metal expert address contract rejected");
        }
        for (auto * node : {up, gate, act, down}) { bind(node); }
        ggml_build_forward_expand(graph, down);
        if (profile) { ++profile->moe_batch_groups; }
        return down;
    }

    void compute(const layer & p, ggml_tensor * dst) {
        if (pipeline && compute_pipeline(p, dst)) { return; }
        llama_allocation_scope explicit_admission;
        const ggml_tensor * input = dst->src[0], * ids = dst->src[1];
        const size_t slots = ids->ne[0], tokens = ids->ne[1], count = slots * tokens;
        if (count > routes_max || input->nb[0] != sizeof(float)) {
            throw std::runtime_error("invalid combined MoE input");
        }
        for (size_t token = 0; token < tokens; ++token) {
            for (size_t slot = 0; slot < slots; ++slot) {
                auto & id = selected[token * slots + slot];
                std::memcpy(&id, static_cast<const char *>(ids->data) + token * ids->nb[1] + slot * ids->nb[0], sizeof(id));
                if (id < 0 || id >= p.up->ne[2]) { throw std::runtime_error("invalid combined MoE route"); }
            }
        }
        const auto memory = budget->snapshot(llama_memory_domain::host);
        const size_t fixed = memory.used() - store.storage_bytes();
        const size_t available = memory.requested > fixed ? memory.requested - fixed : 0;
        const size_t capacity = available > max_payload ? (available - max_payload) / (3 * max_allocation) : 0;
        const size_t width = std::min(groups, capacity);
        if (!width) { throw std::runtime_error("combined MoE working set no longer fits"); }

        llama_expert_timer total(profile ? &profile->projection_ns : nullptr);
        int64_t previous = -1;
        for (;;) {
            std::array<int64_t, groups> experts;
            size_t n = 0;
            for (; n < width; ++n) {
                int64_t next = p.up->ne[2];
                for (size_t row = 0; row < count; ++row) {
                    if (selected[row] > previous) { next = std::min(next, int64_t(selected[row])); }
                }
                if (next == p.up->ne[2]) { break; }
                experts[n] = previous = next;
            }
            if (!n) { break; }
            std::array<llama_expert_store::lease, groups> up, gate, down;
            {
                llama_expert_timer timer(profile ? &profile->acquire_ns : nullptr);
                // Known demand only. Each batch retains admission order, validates
                // complete reads and excludes every referenced slot from reuse.
                store.begin_projection(p.up->name, tokens);
                store.acquire_many(p.up->name, experts.data(), n, up.data());
                store.begin_projection(p.gate->name, tokens);
                store.acquire_many(p.gate->name, experts.data(), n, gate.data());
                store.begin_projection(p.down->name, tokens);
                store.acquire_many(p.down->name, experts.data(), n, down.data());
            }
            llama_expert_timer setup(profile ? &profile->graph_setup_ns : nullptr);
            ggml_reset(ctx.get());
            cursor = 0;
            auto * graph = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
            auto * zero = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
            bind(zero);
            const int32_t id_zero = 0;
            ggml_backend_tensor_set(zero, &id_zero, 0, sizeof(id_zero));
            std::array<ggml_tensor *, routes_max> outputs{};
            std::array<size_t, routes_max> output_rows{};
            size_t nodes = 0;
            for (size_t e = 0; e < n; ++e) {
                for (size_t row = 0; row < count; ++row) {
                    if (selected[row] != experts[e]) { continue; }
                    auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, input->ne[0], 1, 1);
                    bind(x);
                    const size_t token = row / slots;
                    ggml_backend_tensor_set(x, static_cast<const char *>(input->data) + token * input->nb[2], 0,
                                            input->ne[0] * sizeof(float));
                    auto * u = ggml_mul_mat_id(ctx.get(), up[e].tensor(), x, zero);
                    auto * g = ggml_mul_mat_id(ctx.get(), gate[e].tensor(), x, zero);
                    auto * a = ggml_swiglu_split(ctx.get(), g, u);
                    auto * y = ggml_mul_mat_id(ctx.get(), down[e].tensor(), a, zero);
                    for (auto * tensor : {u, g, a, y}) {
                        if (tensor->type != GGML_TYPE_F32 || !ggml_backend_supports_op(backend, tensor)) {
                            throw std::runtime_error("unsupported combined FP32 MoE operation");
                        }
                        bind(tensor);
                    }
                    outputs[nodes] = y;
                    output_rows[nodes++] = row;
                    ggml_build_forward_expand(graph, y);
                }
            }
            // On every exit, drain before any referenced lease/workspace can
            // disappear. No graph pointer crosses this group's lease lifetime.
            setup.next(nullptr);
            llama_expert_timer compute_timer(profile ? &profile->compute_ns : nullptr);
            ggml_status status;
            {
                llama_expert_timer timer(profile ? &profile->submit_ns : nullptr);
                status = ggml_backend_graph_compute_async(backend, graph);
            }
            {
                llama_expert_timer timer(profile ? &profile->wait_ns : nullptr);
                ggml_backend_synchronize(backend);
            }
            compute_timer.next(nullptr);
            if (profile && graph_timing) {
                uint64_t gpu_ns = 0, buffers = 0;
                if (graph_timing(backend, &gpu_ns, &buffers)) {
                    profile->gpu_command_ns += gpu_ns;
                    profile->gpu_command_buffers += buffers;
                    ++profile->gpu_timed_submissions;
                }
            }
            if (status != GGML_STATUS_SUCCESS) { throw std::runtime_error("combined MoE graph failed"); }
            llama_expert_timer readback(profile ? &profile->readback_ns : nullptr);
            for (size_t i = 0; i < nodes; ++i) {
                const size_t row = output_rows[i];
                auto * destination = static_cast<char *>(dst->data) + (row / slots) * dst->nb[2] + (row % slots) * dst->nb[1];
                ggml_backend_tensor_get(outputs[i], destination, 0, dst->ne[0] * sizeof(float));
            }
            matmuls += 3 * nodes;
            ++submissions;
            ggml_reset(ctx.get());
        }
        projections += 3;
        ++calls;
    }

    // Isolated resident-first variant of the FP32 graph above. Keep the
    // operation/precision/scatter contract in sync with the sequential path.
    // One workspace suffices: only weight reads overlap a submitted graph.
    struct predictor_output {
        ggml_tensor * ids = nullptr, * scores = nullptr;
        int prefix = 0;
    };

    predictor_output build_predictor(ggml_cgraph * graph, const llama_expert_lookahead::target & target,
                                    const ggml_tensor * input) {
        const auto * router = target.router_tensor;
        if (!router || size_t(input->ne[0]) != target.width ||
                !ggml_backend_supports_buft(backend, ggml_backend_buffer_get_type(router->buffer))) { return {}; }
        llama_expert_timer timer(profile ? &profile->next_prefetch_metal_ns : nullptr);
        const size_t rows = std::min<int64_t>(2, input->ne[2]);
        const size_t k = std::min(llama_expert_lookahead::top, target.experts);
        auto * pc = predictor_ctx.get();
        ggml_reset(pc);
        auto * x = ggml_new_tensor_2d(pc, GGML_TYPE_F32, target.width, rows);
        auto * logits = ggml_mul_mat(pc, const_cast<ggml_tensor *>(router), x);
        auto * ids = ggml_top_k(pc, logits, k);
        auto * logits_3d = ggml_reshape_3d(pc, logits, 1, target.experts, rows);
        auto * selected_scores = ggml_get_rows(pc, logits_3d, ids);
        for (auto * op : {logits, ids, selected_scores}) {
            if (!ggml_backend_supports_op(backend, op)) { return {}; }
        }
        size_t at = 0;
        for (auto * tensor : {x, logits, ids, selected_scores}) { bind_to(tensor, predictor_buffer.get(), at); }
        if (ggml_backend_view_init(logits_3d) != GGML_STATUS_SUCCESS) { return {}; }
        for (size_t row = 0; row < rows; ++row) {
            ggml_backend_tensor_set(x, static_cast<const char *>(input->data) + row * input->nb[2],
                                    row * x->nb[1], x->nb[1]);
        }
        ggml_build_forward_expand(graph, selected_scores);
        return {ids, selected_scores, ggml_graph_n_nodes(graph)};
    }

    bool read_prediction(const predictor_output & out, llama_expert_lookahead::prediction & predicted) {
        if (!out.ids) { return false; }
        llama_expert_timer timer(profile ? &profile->next_prefetch_metal_ns : nullptr);
        if (!wait_prefix(backend)) { return false; }
        const size_t rows = out.ids->ne[1], k = out.ids->ne[0];
        ggml_backend_tensor_get(out.ids, predicted.ids.data(), 0, rows * k * sizeof(int32_t));
        ggml_backend_tensor_get(out.scores, predicted.logits.data(), 0, rows * k * sizeof(float));
        predicted.rows = rows;
        predicted.ranks = k;
        if (profile) { ++profile->next_prefetch_metal_calls; }
        return true;
    }

    // These nodes live in the target graph. The scheduler owns their storage
    // and transfers both compact top-k outputs at the existing CPU MoE split.
    static bool add_graph_prediction(ggml_context * gc, ggml_backend_sched_t sched, const ggml_tensor * input,
                                    const llama_expert_lookahead::target & target, ggml_tensor ** outputs, bool wide, int64_t routes) {
        const auto * router = target.router_tensor;
        if (!router || size_t(input->ne[0]) != target.width) { return false; }
        auto buft = ggml_backend_buffer_get_type(router->buffer);
        ggml_backend_t gpu = nullptr;
        for (int i = 0; i < ggml_backend_sched_get_n_backends(sched); ++i) {
            auto candidate = ggml_backend_sched_get_backend(sched, i);
            auto device = ggml_backend_get_device(candidate);
            if (device && std::strcmp(ggml_backend_dev_name(device), "MTL0") == 0 &&
                    ggml_backend_supports_buft(candidate, buft)) { gpu = candidate; break; }
        }
        if (!gpu) { return false; }
        const int64_t rows = wide ? input->ne[2] : std::min<int64_t>(2, input->ne[2]);
        const int64_t k = wide ? std::min<int64_t>(routes, target.experts) : std::min(llama_expert_lookahead::top, target.experts);
        auto * x = ggml_view_2d(gc, const_cast<ggml_tensor *>(input), target.width, rows, input->nb[2], 0);
        auto * logits = ggml_mul_mat(gc, const_cast<ggml_tensor *>(router), x);
        auto * ids = ggml_top_k(gc, logits, k);
        auto * scores = ggml_get_rows(gc, ggml_reshape_3d(gc, logits, 1, target.experts, rows), ids);
        for (auto * op : {logits, ids, scores}) {
            if (!ggml_backend_supports_op(gpu, op)) { return false; }
        }
        ggml_set_name(logits, "expert_lookahead_projection");
        ggml_set_name(ids, "expert_lookahead_topk");
        ggml_set_name(scores, "expert_lookahead_scores");
        for (auto * op : {logits, ids, scores}) { ggml_backend_sched_set_tensor_backend(sched, op, gpu); }
        outputs[0] = ids;
        outputs[1] = scores;
        return true;
    }

    bool read_graph_prediction(const ggml_tensor * dst, llama_expert_lookahead::prediction & predicted) {
        const auto * ids = dst->src[2], * scores = dst->src[3];
        if (!ids || !scores || !ggml_backend_buffer_is_host(ids->buffer) ||
                !ggml_backend_buffer_is_host(scores->buffer)) { return false; }
        llama_expert_timer timer(profile ? &profile->next_prefetch_graph_ns : nullptr);
        predicted.rows = ids->ne[1];
        predicted.ranks = ids->ne[0];
        std::memcpy(predicted.ids.data(), ids->data, ggml_nbytes(ids));
        std::memcpy(predicted.logits.data(), scores->data, ggml_nbytes(scores));
        if (profile) { ++profile->next_prefetch_graph_calls; }
        return true;
    }

    // Reuse the operation descriptors and their prebound scratch for a layer
    // evaluation. The backend still encodes a separate graph for each group.
    struct route_lane {
        ggml_tensor * x, * up, * gate, * down;
        size_t input_token;
    };

    route_lane make_lane(int64_t n_embd, ggml_tensor * zero,
                         ggml_tensor * up, ggml_tensor * gate, ggml_tensor * down) {
        auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, n_embd, 1, 1);
        bind(x);
        auto * u = ggml_mul_mat_id(ctx.get(), up, x, zero);
        auto * g = ggml_mul_mat_id(ctx.get(), gate, x, zero);
        auto * a = ggml_swiglu_split(ctx.get(), g, u);
        auto * y = ggml_mul_mat_id(ctx.get(), down, a, zero);
        for (auto * tensor : {u, g, a, y}) {
            if (tensor->type != GGML_TYPE_F32 || !ggml_backend_supports_op(backend, tensor)) {
                throw std::runtime_error("unsupported combined FP32 MoE operation");
            }
            bind(tensor);
        }
        return {x, u, g, y, SIZE_MAX};
    }

    bool compute_pipeline(const layer & p, ggml_tensor * dst) {
        llama_expert_route_stage * staged = nullptr;
        // The scheduler's CPU-side copy has the route shape and bytes, but
        // the publication marker belongs to the original Metal node.
        if (route_stage && dst->src[1]->ne[1] > 1 && publication->pending()) {
            if (route_stage->seal() && publication->stamp() == p.publication_generation) {
                staged = route_stage.get();
            }
            publication->take_if_ready();
            if (profile) {
                ++profile->route_publications;
                if (staged) ++profile->route_stage_ready;
                else ++profile->route_stage_late;
            }
        } else if (!route_stage && publication && publication->pending()) {
            const auto entered = profile ? early_io::now() : 0;
            publication->join();
            if (profile) {
                const auto left = early_io::now();
                profile->route_consumer_wait_ns += left - entered;
                const auto phases = publication->blocked_phases(entered, left);
                profile->route_block_worker_ns += phases[0];
                profile->route_block_signal_ns += phases[1];
                profile->route_block_plan_ns += phases[2];
                profile->route_block_residency_ns += phases[3];
                profile->route_block_enqueue_ns += phases[4];
                profile->route_block_join_ns += phases[5];
            }
            if (publication->stamp() != p.publication_generation ||
                    (published_layer && published_layer != &p)) {
                published_demand.reset();
                throw std::runtime_error("stale route publication");
            }
            if (profile) {
                ++profile->route_publications;
                if (publication->publication_time() && entered > publication->publication_time()) profile->route_publication_lead_ns += entered - publication->publication_time();
                if (publication->preparation_time()) profile->route_preparation_ns += publication->preparation_time() - publication->publication_time();
                if (published_demand) {
                    uint64_t first, hidden, bytes;
                    published_demand->early_read_overlap(entered, first, hidden, bytes);
                    if (first) { ++profile->route_read_windows; profile->route_to_read_ns += first - publication->publication_time(); }
                    profile->route_hidden_io_ns += hidden;
                    profile->route_ready_bytes += bytes;
                }
            }
        }
        struct staged_cleanup {
            llama_expert_route_stage * stage;
            ~staged_cleanup() { if (stage) stage->stop(); }
        } release_staged{staged};
        llama_allocation_scope explicit_admission;
        const ggml_tensor * input = dst->src[0], * ids = dst->src[1];
        const size_t slots = ids->ne[0], tokens = ids->ne[1], count = slots * tokens;
        if (count > routes_max || input->nb[0] != sizeof(float)) {
            throw std::runtime_error("invalid pipelined MoE input");
        }
        const auto memory = budget->snapshot(llama_memory_domain::host);
        const size_t fixed = memory.used() - store.storage_bytes();
        const size_t available = memory.requested > fixed + queue_headroom ? memory.requested - fixed - queue_headroom : 0;
        const size_t capacity = available > max_payload ? (available - max_payload) / (3 * max_allocation) : 0;
        // Never increase the minimum budget merely to enable overlap.
        if (capacity < 2) { published_demand.reset(); if (published_resident) *published_resident = {}; published_layer = nullptr; return false; }
        const size_t width = std::min(groups, capacity / 2);
        std::array<int64_t, routes_max> pending{};
        for (size_t token = 0; token < tokens; ++token) {
            for (size_t slot = 0; slot < slots; ++slot) {
                const size_t row = token * slots + slot;
                std::memcpy(&selected[row], static_cast<const char *>(ids->data) + token * ids->nb[1] + slot * ids->nb[0], sizeof(int32_t));
                if (selected[row] < 0 || selected[row] >= p.up->ne[2]) {
                    throw std::runtime_error("invalid pipelined MoE route");
                }
                pending[row] = selected[row];
            }
        }
        std::sort(pending.begin(), pending.begin() + count);
        const size_t distinct = std::unique(pending.begin(), pending.begin() + count) - pending.begin();
        auto * ahead = lookahead && lookahead->active() ? lookahead.get() : nullptr;
        llama_expert_lookahead::prediction graph_prediction;
        const bool graph_ready = ahead && !ahead->early() && read_graph_prediction(dst, graph_prediction);
        if (ahead && ahead->early() && !published_layer) ahead->prepare_early(p.up->name,pending.data(),distinct,profile);
        bool early_started = false;
        auto load = [&](group & g) {
            llama_expert_timer timer(profile ? &profile->acquire_ns : nullptr);
            store.begin_projection(p.up->name, tokens);
            store.acquire_many(p.up->name, g.experts.data(), g.n, g.up.data(), ahead);
            store.begin_projection(p.gate->name, tokens);
            store.acquire_many(p.gate->name, g.experts.data(), g.n, g.gate.data(), ahead);
            store.begin_projection(p.down->name, tokens);
            store.acquire_many(p.down->name, g.experts.data(), g.n, g.down.data(), ahead);
        };
        auto take = [&](group & g) {
            for (size_t i = 0; i < distinct && g.n < width; ++i) {
                if (pending[i] >= 0) { g.experts[g.n++] = pending[i]; pending[i] = -1; }
            }
        };
        llama_expert_timer total(profile ? &profile->projection_ns : nullptr);
        const bool was_published = published_layer == &p;
        if (was_published) {
            auto remove = [&](const int64_t * ids, size_t n) {
                for (size_t i = 0; i < n; ++i) {
                    auto found = std::find(pending.begin(), pending.begin() + distinct, ids[i]);
                    if (found == pending.begin() + distinct) throw std::runtime_error("published routes differ from CPU MoE input");
                    *found = -1;
                }
            };
            remove(published_resident->experts.data(), published_resident->n);
            remove(published_experts.data(), published_count);
        }
        std::unique_ptr<llama_expert_store::demand_batch> demand = std::move(published_demand);
        published_count = 0; published_layer = nullptr;
        const bool use_queue = was_published || (queued_demand && !metal_predictor && available > queue_headroom + 6 * max_allocation + max_payload);
        std::unique_ptr<llama_expert_store::demand_batch> following;
        auto queue_plan = [&](std::unique_ptr<llama_expert_store::demand_batch> & destination) {
            group plan;
            take(plan);
            if (!plan.n) { return false; }
            llama_expert_timer timer(profile ? &profile->acquire_ns : nullptr);
            destination = store.enqueue_demand({p.up->name, p.gate->name, p.down->name},
                    plan.experts.data(), plan.n, ahead, readers.get(), false, staged);
            return true;
        };
        auto enqueue_ready = [&] {
            if (demand && demand->pending()) { return true; }
            if (following) {
                demand = std::move(following);
                return true;
            }
            return queue_plan(demand);
        };
        auto collect_ready = [&](group & g) {
            llama_expert_timer timer(profile ? &profile->acquire_ns : nullptr);
            g.n = demand->take(g.experts.data(), g.up.data(), g.gate.data(), g.down.data(),
                    flash_next_verify && tokens == 4 ? size_t(8) : demand_min_ready);
        };
        group current;
        if (was_published) {
            current = std::move(*published_resident);
            *published_resident = {};
            if (current.n) ++resident_groups;
        }
        for (size_t i = 0; !was_published && i < distinct && current.n < width; ++i) {
            const auto e = pending[i];
            if (store.contains(p.up->name, e) && store.contains(p.gate->name, e) && store.contains(p.down->name, e)) {
                current.experts[current.n++] = e;
            }
        }
        if (current.n && !was_published) {
            llama_expert_timer timer(profile ? &profile->acquire_ns : nullptr);
            // No source reads on this path, even if residency changed between
            // inspection and pinning. Lifecycle mutations require an idle owner.
            const bool hot = store.acquire_resident_many(p.up->name, current.experts.data(), current.n, current.up.data()) &&
                    store.acquire_resident_many(p.gate->name, current.experts.data(), current.n, current.gate.data()) &&
                    store.acquire_resident_many(p.down->name, current.experts.data(), current.n, current.down.data());
            if (hot) {
                ++resident_groups;
                for (size_t i = 0; i < distinct; ++i) {
                    if (std::find(current.experts.begin(), current.experts.begin() + current.n, pending[i]) != current.experts.begin() + current.n) {
                        pending[i] = -1;
                    }
                }
            } else { current = {}; }
        }
        if (was_published && !current.n && demand) { collect_ready(current); }
        if (!current.n) {
            if (use_queue && enqueue_ready()) { collect_ready(current); }
            else { take(current); load(current); }
        }
        const bool use_batch = use_queue && batch_routes;
        const bool use_plan = use_queue && reuse_plan && !use_batch;
        std::array<route_lane, routes_max> lanes;
        size_t lane_count = 0;
        ggml_cgraph * plan_graph = nullptr;
        ggml_tensor * plan_zero = nullptr;

        while (current.n) {
            llama_expert_timer setup(profile ? &profile->graph_setup_ns : nullptr);
            ggml_cgraph * graph;
            predictor_output output;
            std::array<ggml_tensor *, routes_max> outputs{};
            std::array<size_t, routes_max> output_rows{};
            size_t nodes = 0;
            ggml_tensor * batch_output = nullptr;
            if (use_batch) {
                ggml_reset(ctx.get());
                cursor = 0;
                graph = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
                std::array<int32_t, routes_max> route_ids;
                for (size_t e = 0; e < current.n; ++e) {
                    for (size_t row = 0; row < count; ++row) {
                        if (selected[row] != current.experts[e]) { continue; }
                        output_rows[nodes] = row;
                        route_ids[nodes++] = int32_t(e);
                    }
                }
                auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, nodes, 1);
                bind(ids);
                ggml_backend_tensor_set(ids, route_ids.data(), 0, nodes*sizeof(int32_t));
                auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, input->ne[0], nodes, 1);
                bind(x);
                for (size_t i = 0; i < nodes; ++i) {
                    const size_t bytes = input->ne[0]*sizeof(float);
                    const size_t token = output_rows[i] / slots;
                    ggml_backend_tensor_set(x, static_cast<const char *>(input->data) + token * input->nb[2], i*bytes, bytes);
                }
                const std::array weights{&current.up, &current.gate, &current.down};
                batch_output = build_address_ffn(graph, x, ids, current.n,
                        [&](size_t i, size_t projection) { return (*weights[projection])[i].tensor(); });
            } else if (use_plan) {
                if (plan_graph) {
                    // Rebuild the active node list and use counts for Metal fusion.
                    ggml_graph_clear(plan_graph);
                } else {
                    ggml_reset(ctx.get());
                    cursor = 0;
                    plan_graph = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
                    plan_zero = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
                    bind(plan_zero);
                    const int32_t zero = 0;
                    ggml_backend_tensor_set(plan_zero, &zero, 0, sizeof(zero));
                    if (profile) { ++profile->moe_plan_layers; }
                }
                graph = plan_graph;
                for (size_t e = 0; e < current.n; ++e) {
                    auto * up = current.up[e].tensor();
                    auto * gate = current.gate[e].tensor();
                    auto * down = current.down[e].tensor();
                    for (size_t row = 0; row < count; ++row) {
                        if (selected[row] != current.experts[e]) { continue; }
                        if (nodes == lane_count) {
                            lanes[lane_count++] = make_lane(input->ne[0], plan_zero, up, gate, down);
                            if (profile) { ++profile->moe_plan_lanes_built; }
                        } else {
                            // All slices of a given projection have the same
                            // type, shape and strides. Only the lease changes.
                            if (profile) { ++profile->moe_plan_lanes_reused; }
                        }
                        auto & lane = lanes[nodes];
                        lane.up->src[0] = up;
                        lane.gate->src[0] = gate;
                        lane.down->src[0] = down;
                        const size_t token = row / slots;
                        if (lane.input_token != token) {
                            ggml_backend_tensor_set(lane.x, static_cast<const char *>(input->data) + token * input->nb[2], 0,
                                                    input->ne[0] * sizeof(float));
                            lane.input_token = token;
                        }
                        outputs[nodes] = lane.down;
                        output_rows[nodes++] = row;
                        ggml_build_forward_expand(graph, lane.down);
                    }
                }
            } else {
                ggml_reset(ctx.get());
                cursor = 0;
                graph = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
                if (ahead && metal_predictor && ahead->can_replace() &&
                        std::none_of(pending.begin(), pending.begin() + distinct, [](int64_t id) { return id >= 0; })) {
                    setup.next(nullptr);
                    output = build_predictor(graph, ahead->targets[&p - layers.data()], input);
                    setup.next(profile ? &profile->graph_setup_ns : nullptr);
                }
                auto * zero = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 1);
                bind(zero);
                const int32_t id_zero = 0;
                ggml_backend_tensor_set(zero, &id_zero, 0, sizeof(id_zero));
                for (size_t e = 0; e < current.n; ++e) {
                    for (size_t row = 0; row < count; ++row) {
                        if (selected[row] != current.experts[e]) { continue; }
                        auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, input->ne[0], 1, 1);
                        bind(x);
                        const size_t token = row / slots;
                        ggml_backend_tensor_set(x, static_cast<const char *>(input->data) + token * input->nb[2], 0,
                                                input->ne[0] * sizeof(float));
                        auto * u = ggml_mul_mat_id(ctx.get(), current.up[e].tensor(), x, zero);
                        auto * g = ggml_mul_mat_id(ctx.get(), current.gate[e].tensor(), x, zero);
                        auto * a = ggml_swiglu_split(ctx.get(), g, u);
                        auto * y = ggml_mul_mat_id(ctx.get(), current.down[e].tensor(), a, zero);
                        for (auto * tensor : {u, g, a, y}) {
                            if (tensor->type != GGML_TYPE_F32 || !ggml_backend_supports_op(backend, tensor)) {
                                throw std::runtime_error("unsupported combined FP32 MoE operation");
                            }
                            bind(tensor);
                        }
                        outputs[nodes] = y;
                        output_rows[nodes++] = row;
                        ggml_build_forward_expand(graph, y);
                    }
                }
            }
            setup.next(nullptr);
            group next;
            bool queued_next = false;
            // Declared after both lease owners: every error drains the graph
            // before either group or its tensor metadata can be destroyed.
            struct fence {
                ggml_backend_t backend;
                bool active = true;
                ~fence() { if (active) { ggml_backend_synchronize(backend); } }
                void wait() { ggml_backend_synchronize(backend); active = false; }
            } completion{backend};
            llama_expert_timer compute_timer(profile ? &profile->compute_ns : nullptr);
            ggml_status status;
            {
                llama_expert_timer timer(profile ? &profile->submit_ns : nullptr);
                status = output.prefix ? compute_prefix(backend, graph, output.prefix) :
                    ggml_backend_graph_compute_async(backend, graph);
            }
            if (status == GGML_STATUS_SUCCESS) {
                if (use_queue) {
                    // Submission is asynchronous. Keep draining this wave first, but give
                    // the readers the next known routes while Metal owns the current leases.
                    // With publication disabled, the current leases and the unconsumed
                    // portion of demand belong to one wave. Each wave is <= capacity/2.
                    if (flash_next_verify && tokens == 4 && !was_published && !staged &&
                            demand && demand->pending() && !following) {
                        queue_plan(following);
                    }
                    queued_next = enqueue_ready();
                    if (queued_next) { ++pipeline_groups; }
                } else {
                    take(next);
                    if (next.n) {
                        ++pipeline_groups;
                        llama_expert_timer overlap(profile ? &profile->overlap_acquire_ns : nullptr);
                        const auto before = profile ? profile->direct_read_bytes : 0;
                        load(next);
                        if (profile) {
                            profile->overlap_acquisitions += 3 * next.n;
                            profile->overlap_read_bytes += profile->direct_read_bytes - before;
                        }
                    }
                }
                if (ahead && ahead->early() && !early_started) {
                    ahead->fetch_early(ahead->targets[&p - layers.data()],input,dst->src[2],dst->src[3],profile);
                    early_started = true;
                }
                if (!next.n && !queued_next && ahead && !ahead->early()) {
                    llama_expert_lookahead::prediction predicted;
                    const bool metal_ready = read_prediction(output, predicted);
                    const auto * proposal = graph_ready ? &graph_prediction : (metal_ready ? &predicted : nullptr);
                ahead->fetch(ahead->targets[&p - layers.data()], input, profile, proposal);
                }
            }
            {
                llama_expert_timer timer(profile ? &profile->wait_ns : nullptr);
                completion.wait();
            }
            compute_timer.next(nullptr);
            if (profile && graph_timing) {
                uint64_t gpu_ns = 0, buffers = 0;
                if (graph_timing(backend, &gpu_ns, &buffers)) {
                    profile->gpu_command_ns += gpu_ns;
                    profile->gpu_command_buffers += buffers;
                    ++profile->gpu_timed_submissions;
                }
            }
            if (status != GGML_STATUS_SUCCESS) { throw std::runtime_error("pipelined MoE graph failed"); }
            {
                llama_expert_timer readback(profile ? &profile->readback_ns : nullptr);
                for (size_t i = 0; i < nodes; ++i) {
                    const size_t row = output_rows[i];
                    auto * destination = static_cast<char *>(dst->data) + (row / slots) * dst->nb[2] + (row % slots) * dst->nb[1];
                    const size_t bytes = dst->ne[0]*sizeof(float);
                    ggml_backend_tensor_get(batch_output ? batch_output : outputs[i], destination,
                        batch_output ? i*bytes : 0, bytes);
                }
            }
    
            matmuls += 3 * nodes;
            ++submissions;
            if (use_plan) {
                // The graph has drained. Never keep bindings to entries that
                // can be evicted and recycled after the current leases release.
                for (size_t i = 0; i < nodes; ++i) {
                    lanes[i].up->src[0] = lanes[i].gate->src[0] = lanes[i].down->src[0] = nullptr;
                }
            } else {
                ggml_reset(ctx.get());
            }
            if (queued_next) {
                // Let the previous GPU work and readback run before forming
                // the next group. Include every expert that is ready by then.
                collect_ready(next);
            }
            current = std::move(next);
        }
        projections += 3;
        ++calls;
        return true;
    }

    static void callback(ggml_tensor * dst, int ith, int, void * userdata) noexcept {
        if (ith != 0) { return; }
        const auto & p = *static_cast<layer *>(userdata);
        auto & owner = *p.owner;
        if (!owner.failed.load()) {
            try { owner.compute(p, dst); return; }
            catch (const std::exception & e) { std::snprintf(owner.failure, owner.failure_size, "%s", e.what()); }
            catch (...) { std::snprintf(owner.failure, owner.failure_size, "combined MoE failed"); }
            owner.failed.store(true);
        }
        std::memset(dst->data, 0, ggml_nbytes(dst));
    }

    static void window_callback(ggml_tensor * dst, int ith, int, void * userdata) noexcept {
        if (ith != 0) { return; }
        auto & owner = *static_cast<layer *>(userdata)->owner;
        if (owner.window_ready == dst) {
            owner.window_ready = nullptr;
            return;
        }
        std::snprintf(owner.failure, owner.failure_size, "missing precomputed window MoE output");
        owner.failed.store(true);
        std::memset(dst->data, 0, ggml_nbytes(dst));
    }

    const layer * window_layer(const ggml_tensor * node) const {
        if (!node || node->op != GGML_OP_CUSTOM) { return nullptr; }
        ggml_custom_op_params op;
        std::memcpy(&op, node->op_params, sizeof(op));
        if (op.fun != window_callback) { return nullptr; }
        const auto * descriptor = static_cast<const layer *>(op.userdata);
        return descriptor->owner == this ? descriptor : nullptr;
    }

public:
    llama_expert_moe(std::shared_ptr<llama_memory_budget> budget, ggml_backend_t backend, llama_expert_store & store,
                    std::shared_ptr<llama_expert_source> source, size_t max_width, size_t max_allocation, size_t max_payload,
                    std::atomic<bool> & failed, char * failure, size_t failure_size,
                    llama_expert_profile_data * & profile, uint64_t & projections, uint64_t & matmuls, uint64_t & submissions,
                    bool automatic = false, bool dflash_verify = false, bool profile_overlap = false, bool flash_next = false) :
        budget(std::move(budget)), backend(backend), store(store), failed(failed), failure(failure), failure_size(failure_size),
        profile(profile), projections(projections), matmuls(matmuls), submissions(submissions),
        max_allocation(max_allocation), max_payload(max_payload) {
        const char * enabled = std::getenv("LLAMA_EXPERIMENTAL_METAL_MOE_PIPELINE");
        pipeline = automatic || (enabled && std::strcmp(enabled, "1") == 0);
#ifdef __APPLE__
        const char * demand_mode = std::getenv("LLAMA_EXPERIMENTAL_MOE_DEMAND_QUEUE");
        queued_demand = pipeline && (automatic || (demand_mode && std::strcmp(demand_mode, "1") == 0)) && !source->packed_stride();
        if (queued_demand) {

            const char * batch = std::getenv("ATOMIC_MOE_BATCH");
            set_addresses = reinterpret_cast<addresses_t>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "ggml_backend_metal_expert_addresses_v1"));
            batch_routes = set_addresses && (automatic || (batch && std::strcmp(batch, "1") == 0));
            const char * plan = std::getenv("LLAMA_EXPERIMENTAL_MOE_REUSE_PLAN");
            reuse_plan = !automatic && plan && std::strcmp(plan, "1") == 0;
            const char * min_ready = std::getenv("LLAMA_EXPERIMENTAL_MOE_DEMAND_MIN_READY");
            if (!automatic && min_ready && std::strcmp(min_ready, "8") == 0) { demand_min_ready = 8; }
        }
#endif
        graph_timing = reinterpret_cast<graph_timing_t>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "ggml_backend_metal_last_graph_timing"));
        const size_t metadata = ggml_graph_overhead_custom(graph_nodes, false) +
                (routes_max * 5 + 1) * ggml_tensor_overhead();
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        const size_t scratch = aligned(routes_max * 5 * aligned(max_width * sizeof(float) + 64, alignment) + 65536);
        metadata_charge = this->budget->reserve(llama_memory_domain::host, llama_memory_category::runtime, metadata);
        scratch_charge = this->budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, scratch);
        if (!metadata_charge || !scratch_charge) { throw std::runtime_error("combined MoE workspace does not fit"); }
        ctx.reset(ggml_init({metadata, nullptr, true}));
        if (!ctx) { throw std::runtime_error("cannot allocate combined MoE metadata"); }
        metadata_charge.commit();
        buffer.reset(ggml_backend_alloc_buffer(backend, scratch));
        if (!buffer || ggml_backend_buffer_get_size(buffer.get()) != scratch) {
            throw std::runtime_error("combined MoE buffer differs from its reservation");
        }
        scratch_charge.commit();
        const auto memory = this->budget->snapshot(llama_memory_domain::host);
        const size_t working = admission_working_bytes();
        if (memory.used() > memory.requested || working > memory.requested - memory.used()) {
            throw std::runtime_error("combined MoE minimum working set does not fit");
        }
#ifdef __APPLE__
        flash_next_verify = automatic && flash_next;
        if (automatic && queued_demand) {
            const size_t bytes = early_io::pool::memory_bytes();
            if (memory.requested - memory.used() - working >= bytes) {
                readers_charge = this->budget->reserve(llama_memory_domain::host, llama_memory_category::io, bytes);
                if (readers_charge) {
                    try { readers = std::make_unique<early_io::pool>(); readers_charge.commit(); }
                    catch (const std::exception &) { readers_charge = {}; }
                }
            }
            // No unaccounted GCD workers on the automatic path.
            queued_demand = bool(readers);
        }
        const char * verify_mode = dflash_verify ? std::getenv("ATOMIC_MOE_VERIFY_OVERLAP") : nullptr;
        verify_overlap = dflash_verify && (verify_mode ? std::strcmp(verify_mode, "1") == 0 : profile_overlap);
        const char * publish_mode = verify_overlap ? "verify" : std::getenv("ATOMIC_MOE_PUBLISH");
        if (automatic && readers && publish_mode &&
                (verify_overlap || !std::strcmp(publish_mode, "event") || !std::strcmp(publish_mode, "shared"))) {
            const auto * api = static_cast<const ggml_metal_route_api *>(ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(ggml_backend_get_device(backend)), "ggml_backend_metal_route_publication_v1"));
            const size_t extra = verify_overlap ? llama_expert_route_stage::memory_bytes(max_payload) : sizeof(group);
            const size_t bytes = api ? api->reserved_bytes + sizeof(llama_expert_publication) + extra : 0;
            const auto room = this->budget->snapshot(llama_memory_domain::host);
            if (api && room.used() <= room.requested && working + bytes <= room.requested - room.used()) {
                auto charge = this->budget->reserve(llama_memory_domain::host, llama_memory_category::io, bytes);
                if (charge) {
                    try {
                        auto channel = std::make_unique<llama_expert_publication>(*api, backend, *readers);
                        const char * dedupe = std::getenv("ATOMIC_MOE_VERIFY_DEDUPE");
                        auto stage = verify_overlap ? std::make_unique<llama_expert_route_stage>(
                                source, store, *readers, max_payload, !dedupe || std::strcmp(dedupe, "0") != 0) : nullptr;
                        auto resident = verify_overlap ? nullptr : std::make_unique<group>();
                        charge.commit(); publication_charge = std::move(charge);
                        publication = std::move(channel); published_resident = std::move(resident); route_stage = std::move(stage);
                        shared_overlap = verify_overlap || !std::strcmp(publish_mode, "shared");
                        const char * layer = std::getenv("ATOMIC_MOE_PUBLISH_LAYER");
                        if (verify_overlap || (layer && !std::strcmp(layer, "all"))) publication_layer = -1;
                    } catch (const std::exception &) { /* optional, preserve existing streaming */ }
                }
            }
            std::fprintf(stderr, "expert route publication: experimental=%s admitted=%d bytes=%zu layer=%s%s\n",
                    publish_mode, int(bool(publication)), publication_charge.size(), publication_layer < 0 ? "all" : "0",
                    publication ? "" : " fallback=capability-or-budget");
        }
        const auto admitted = this->budget->snapshot(llama_memory_domain::host);
        const size_t spare = admitted.requested > admitted.used() + working ? admitted.requested - admitted.used() - working : 0;
        const char * qwen_prediction = std::getenv("ATOMIC_MOE_QWEN_PREDICTOR");
        single_row_prediction = automatic && source->architecture() == "qwen35moe" &&
                qwen_prediction && !std::strcmp(qwen_prediction, "1");
        const bool qualified_prediction = source->architecture() == "qwen4exp" || single_row_prediction;
        const char * prefetch = std::getenv("LLAMA_EXPERIMENTAL_MOE_LOOKAHEAD");
        const char * predictor = std::getenv("LLAMA_EXPERIMENTAL_MOE_LOOKAHEAD_METAL");
        const char * graph = std::getenv("LLAMA_EXPERIMENTAL_MOE_LOOKAHEAD_GRAPH");
        const bool request_graph = automatic ? qualified_prediction : graph && std::strcmp(graph, "1") == 0;
        const bool request_metal = !automatic && !request_graph && predictor && std::strcmp(predictor, "1") == 0;
        const char * early_flag = std::getenv("ATOMIC_EARLY_PREFETCH");
        bool request_early = queued_demand && request_graph && (automatic || (early_flag && std::strcmp(early_flag,"1") == 0));
        size_t selected_limit = automatic ? (single_row_prediction ? 4 : 8) : early_lookahead::configured_candidates();
        if (request_early && llama_expert_lookahead::working_bytes(max_payload,true,selected_limit,bool(readers)) > spare) selected_limit = 4;
        if (request_early && llama_expert_lookahead::working_bytes(max_payload,true,selected_limit,bool(readers)) > spare) request_early = false;
        const size_t bytes = llama_expert_lookahead::working_bytes(max_payload,request_early,selected_limit,bool(readers));
        const bool want = automatic ? request_early : prefetch && std::strcmp(prefetch, "1") == 0;
        if (pipeline && want && !source->packed_stride() && bytes <= spare) {
            auto charge = this->budget->reserve(llama_memory_domain::host, llama_memory_category::io, bytes);
            if (charge) {
                try {
                    lookahead = std::make_unique<llama_expert_lookahead>(source, store, std::move(charge), max_payload,
                        request_metal || request_graph, request_early, selected_limit, readers.get());
                    candidate_limit = request_early ? selected_limit : 2;
                } catch (const std::exception &) { /* prediction is optional */ }
            }
        }
        graph_predictor = lookahead && request_graph;
        if (lookahead && request_metal) {
            const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            compute_prefix = reinterpret_cast<compute_prefix_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_compute_graph_prefix"));
            wait_prefix = reinterpret_cast<wait_prefix_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_wait_graph_prefix"));
            const auto available = this->budget->snapshot(llama_memory_domain::host);
            const size_t meta = 8 * ggml_tensor_overhead(), scratch = 128 * 1024;
            if (compute_prefix && wait_prefix && available.used() <= available.requested &&
                    working + meta + scratch <= available.requested - available.used()) {
                auto m = this->budget->reserve(llama_memory_domain::host, llama_memory_category::runtime, meta);
                auto b = this->budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, scratch);
                if (m && b) {
                    ggml_context_ptr pc(ggml_init({meta, nullptr, true}));
                    ggml_backend_buffer_ptr pb(ggml_backend_alloc_buffer(backend, scratch));
                    if (pc && pb && ggml_backend_buffer_get_size(pb.get()) == scratch) {
                        m.commit();
                        b.commit();
                        predictor_meta_charge = std::move(m);
                        predictor_scratch_charge = std::move(b);
                        predictor_ctx = std::move(pc);
                        predictor_buffer = std::move(pb);
                        metal_predictor = true;
                    }
                }
            }
        }
#else
        (void) source;
#endif
    }

    ggml_tensor * build(ggml_context * graph_ctx, const ggml_tensor * up, const ggml_tensor * gate,
                       const ggml_tensor * down, ggml_tensor * input, ggml_tensor * ids,
                       const ggml_tensor * router, const std::array<const ggml_tensor *, 3> & future,
                       ggml_backend_sched_t predictor_sched, bool window = false) {
        const int max_rows = window && matrix_buffer ? 16 : 8;
        if (!up || !gate || !down || up->ne[0] != gate->ne[0] || up->ne[1] != gate->ne[1] ||
                down->ne[0] != up->ne[1] || down->ne[1] != up->ne[0] ||
                up->ne[2] != gate->ne[2] || up->ne[2] != down->ne[2] ||
                input->type != GGML_TYPE_F32 || input->ne[0] != up->ne[0] || input->nb[0] != sizeof(float) ||
                input->ne[1] != 1 || input->ne[3] != 1 || ids->type != GGML_TYPE_I32 ||
                ids->ne[0] < 1 || ids->ne[0] > 32 || ids->ne[1] < (window ? 0 : 1) || ids->ne[1] > max_rows ||
                input->ne[2] != ids->ne[1] || ids->ne[2] != 1 || ids->ne[3] != 1) { return nullptr; }
        for (const auto * weight : {up, gate, down}) {
            ggml_tensor probe{};
            probe.op = GGML_OP_MUL_MAT_ID; probe.type = GGML_TYPE_F32;
            probe.src[0] = const_cast<ggml_tensor *>(weight); probe.src[1] = input; probe.src[2] = ids;
            probe.ne[0] = weight->ne[1]; probe.ne[1] = ids->ne[0]; probe.ne[2] = ids->ne[1]; probe.ne[3] = 1;
            probe.nb[0] = sizeof(float); for (int i=1;i<4;++i) probe.nb[i]=probe.nb[i-1]*probe.ne[i-1];
            if (ids->ne[1] && !ggml_backend_supports_op(backend, &probe)) { return nullptr; }
        }
        layer * descriptor = nullptr;
        for (size_t i = 0; i < n_layers; ++i) {
            if (layers[i].up == up && layers[i].gate == gate && layers[i].down == down) { descriptor = &layers[i]; break; }
        }
        if (!descriptor) {
            if (n_layers == layers.size()) { return nullptr; }
            descriptor = &layers[n_layers++];
            *descriptor = {this, up, gate, down};
        }
        if (window) {
            ggml_tensor * args[] = {input, ids};
            return ggml_custom_4d(graph_ctx, GGML_TYPE_F32, down->ne[1], ids->ne[0], ids->ne[1], 1,
                                  args, 2, window_callback, 1, descriptor);
        }
        if (single_row_prediction && ids->ne[1] != 1) router = nullptr;
        if (lookahead) { lookahead->targets[descriptor - layers.data()] = lookahead->bind(router, future, backend); }
        const bool publish_shape = verify_overlap ? ids->ne[1] > 1 && ggml_nelements(ids) <= GGML_METAL_ROUTE_MAX_IDS :
            ids->ne[1] == 1 && ids->ne[0] <= 32;
        if (publication && predictor_sched && publish_shape &&
                (publication_layer < 0 || !std::strcmp(up->name, "blk.0.ffn_up_exps.weight"))) {
            ggml_backend_t gpu = nullptr;
            for (int i = 0; i < ggml_backend_sched_get_n_backends(predictor_sched); ++i) {
                auto candidate = ggml_backend_sched_get_backend(predictor_sched, i);
                if (ggml_backend_get_device(candidate) == ggml_backend_get_device(backend)) { gpu = candidate; break; }
            }
            if (gpu) {
                auto * copy = ggml_cont(graph_ctx, ids);
                if (!descriptor->publication_generation) descriptor->publication_generation = ++graph_generation;
                if (publication->attach(copy, arm_publication, descriptor, descriptor->publication_generation)) {
                    ggml_backend_sched_set_tensor_backend(predictor_sched, copy, gpu);
                    ggml_set_name(copy, "expert_route_publication");
                    ids = copy;
                }
            }
        }
        ggml_tensor * args[4] = {input, ids};
        int n_args = 2;
        if (graph_predictor && predictor_sched && add_graph_prediction(
                graph_ctx, predictor_sched, input, lookahead->targets[descriptor - layers.data()], args + 2, lookahead->early(), ids->ne[0])) {
            n_args = 4;
        }
        return ggml_custom_4d(graph_ctx, GGML_TYPE_F32, down->ne[1], ids->ne[0], ids->ne[1], 1,
                              args, n_args, callback, 1, descriptor);
    }

    const char * window_weight_name(const ggml_tensor * node) const {
        const auto * descriptor = window_layer(node);
        return descriptor ? descriptor->down->name : nullptr;
    }

    void use_window_output(const ggml_tensor * node) {
        // The CPU backend skips empty nodes. Keep their layer boundary in the
        // graph without expecting the empty callback to consume a result.
        if (ggml_is_empty(node)) { return; }
        if (window_ready) { throw std::logic_error("previous window MoE output was not consumed"); }
        window_ready = node;
    }
    void clear_window_output() { window_ready = nullptr; }

    void compute_window(ggml_tensor * const * nodes, size_t count, const std::array<ggml_tensor *, 3> * bindings) {
        constexpr size_t window_routes = routes_max * 8;
        struct route {
            int32_t id;
            const char * input;
            char * output;
        };
        std::array<route, window_routes> routes;
        const auto * p = count ? window_layer(nodes[0]) : nullptr;
        if (!p || !set_addresses || failed.load()) { throw std::runtime_error("invalid window MoE group"); }
        size_t used = 0;
        for (size_t i = 0; i < count; ++i) {
            auto * node = nodes[i];
            if (window_layer(node) != p) { throw std::runtime_error("mixed layers in window MoE group"); }
            const auto * x = node->src[0];
            const auto * ids = node->src[1];
            for (int64_t row = 0; row < ids->ne[1]; ++row) {
                for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                    int32_t id;
                    std::memcpy(&id, static_cast<const char *>(ids->data) + row * ids->nb[1] + slot * ids->nb[0], sizeof(id));
                    if (id < 0 || id >= p->up->ne[2] || used == routes.size()) {
                        throw std::runtime_error("invalid window MoE route");
                    }
                    routes[used++] = {id, static_cast<const char *>(x->data) + row * x->nb[2],
                            static_cast<char *>(node->data) + row * node->nb[2] + slot * node->nb[1]};
                }
            }
        }
        if (!used) { return; }
        // Each route is independent. Keep its original destination while
        // clustering by expert to use one address entry for repeated rows.
        std::sort(routes.begin(), routes.begin() + used, [](const route & a, const route & b) { return a.id < b.id; });
        const auto memory = budget->snapshot(llama_memory_domain::host);
        const size_t fixed = memory.used() - store.storage_bytes();
        const size_t limit = std::min(memory.requested, memory.applied);
        const size_t available = fixed < limit ? limit - fixed : 0;
        const size_t width = available > max_payload ? std::min(bindings ? window_groups : groups,
                (available - max_payload) / (3 * max_allocation)) : 0;
        if (!width) { throw std::runtime_error("window MoE working set no longer fits"); }
        const size_t in_bytes = p->up->ne[0] * sizeof(float);
        const size_t out_bytes = p->down->ne[1] * sizeof(float);
        llama_expert_timer total(profile ? &profile->projection_ns : nullptr);
        llama_allocation_scope explicit_admission;
        for (size_t first = 0; first < used;) {
            group owned;
            std::array<int64_t, window_groups> experts;
            size_t n_experts = 0;
            size_t end = first;
            std::array<int32_t, routes_max> local_ids;
            for (; end < used && end - first < routes_max; ++end) {
                if (!n_experts || experts[n_experts - 1] != routes[end].id) {
                    if (n_experts == width) { break; }
                    experts[n_experts++] = routes[end].id;
                }
                local_ids[end - first] = int32_t(n_experts - 1);
            }
            if (bindings) {
                for (size_t i = 0; i < n_experts; ++i) {
                    const auto & entry = bindings[experts[i]];
                    if (!entry[0] || !entry[1] || !entry[2]) { throw std::logic_error("missing pinned MoE weights"); }
                }
            } else {
                llama_expert_timer acquire(profile ? &profile->acquire_ns : nullptr);
                store.acquire_many(p->up->name, experts.data(), n_experts, owned.up.data());
                store.acquire_many(p->gate->name, experts.data(), n_experts, owned.gate.data());
                store.acquire_many(p->down->name, experts.data(), n_experts, owned.down.data());
            }
            ggml_reset(ctx.get());
            cursor = 0;
            auto * graph = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
            const size_t rows = end - first;
            auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, rows, 1);
            auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, p->up->ne[0], rows, 1);
            bind(ids);
            bind(input);
            ggml_backend_tensor_set(ids, local_ids.data(), 0, rows * sizeof(int32_t));
            for (size_t r = 0; r < rows; ++r) { ggml_backend_tensor_set(input, routes[first + r].input, r * in_bytes, in_bytes); }
            const std::array held{&owned.up, &owned.gate, &owned.down};
            auto * output = build_address_ffn(graph, input, ids, n_experts, [&](size_t i, size_t projection) {
                return bindings ? bindings[experts[i]][projection] : (*held[projection])[i].tensor();
            });
            llama_expert_timer submit(profile ? &profile->submit_ns : nullptr);
            const auto status = ggml_backend_graph_compute_async(backend, graph);
            submit.next(profile ? &profile->wait_ns : nullptr);
            ggml_backend_synchronize(backend);
            submit.next(nullptr);
            if (status != GGML_STATUS_SUCCESS) { throw std::runtime_error("window MoE Metal graph failed"); }
            if (profile && graph_timing) {
                uint64_t ns = 0, commands = 0;
                if (graph_timing(backend, &ns, &commands)) {
                    profile->gpu_command_ns += ns;
                    profile->gpu_command_buffers += commands;
                    ++profile->gpu_timed_submissions;
                }
            }
            for (size_t r = 0; r < rows; ++r) { ggml_backend_tensor_get(output, routes[first + r].output, r * out_bytes, out_bytes); }
            ggml_reset(ctx.get());
            matmuls += 3 * rows;
            ++submissions;
            first = end;
        }
        projections += 3 * count;
        calls += count;
    }

    bool prepare_matrix_window() {
        const size_t bytes = matrix_route_bytes + matrix_tensor_bytes;
        auto charge = budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, bytes);
        if (!charge) { return false; }
        llama_allocation_scope explicit_admission;
        ggml_backend_buffer_ptr workspace(ggml_backend_alloc_buffer(backend, bytes));
        if (!workspace || ggml_backend_buffer_get_size(workspace.get()) != bytes ||
                !ggml_backend_buffer_get_base(workspace.get())) { return false; }
        charge.commit();
        matrix_charge = std::move(charge);
        matrix_buffer = std::move(workspace);
        return true;
    }

    void release_matrix_window() {
        matrix_buffer.reset();
        matrix_charge.reset();
    }

    // Group exact routes by expert and scatter to their original positions.
    // Bindings stay pinned until the coordinator completes the consumer graphs.
    bool compute_matrix_window(ggml_tensor * const * nodes, size_t count,
            const std::array<ggml_tensor *, 3> * bindings) {
        struct route {
            uint32_t expert, order;
            const char * input;
            char * output;
        };
        const auto * p = count ? window_layer(nodes[0]) : nullptr;
        if (!p || !matrix_buffer || failed.load()) { return false; }
        auto * raw = ggml_backend_buffer_get_base(matrix_buffer.get());
        auto * routes = static_cast<route *>(raw);
        size_t used = 0;
        for (size_t i = 0; i < count; ++i) {
            auto * node = nodes[i];
            if (window_layer(node) != p) { throw std::runtime_error("mixed matrix window layers"); }
            const auto * x = node->src[0];
            const auto * ids = node->src[1];
            for (int64_t row = 0; row < ids->ne[1]; ++row) {
                for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                    int32_t expert;
                    std::memcpy(&expert, static_cast<const char *>(ids->data) + row*ids->nb[1] + slot*ids->nb[0], sizeof(expert));
                    if (expert < 0 || expert >= p->up->ne[2]) {
                        throw std::runtime_error("invalid matrix window route");
                    }
                    if (used >= matrix_route_bytes / sizeof(route)) { return false; }
                    new (routes + used) route{uint32_t(expert), uint32_t(used),
                            static_cast<const char *>(x->data) + row*x->nb[2],
                            static_cast<char *>(node->data) + row*node->nb[2] + slot*node->nb[1]};
                    ++used;
                }
            }
        }
        if (!used) { return true; }
        std::sort(routes, routes + used, [](const route & a, const route & b) {
            return a.expert < b.expert || (a.expert == b.expert && a.order < b.order);
        });
        const size_t in_bytes = p->up->ne[0] * sizeof(float);
        const size_t out_bytes = p->down->ne[1] * sizeof(float);
        const auto buft = ggml_backend_buffer_get_type(matrix_buffer.get());
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        const size_t row_bytes = 2 * in_bytes + 3 * p->up->ne[1] * sizeof(float);
        const size_t padding = 64 * 5 * alignment;
        if (padding >= matrix_tensor_bytes) { return false; }
        const size_t row_limit = std::min<size_t>(1024, (matrix_tensor_bytes - padding) / row_bytes);
        if (!row_limit) { return false; }
        llama_expert_timer projection(profile ? &profile->projection_ns : nullptr);
        llama_allocation_scope explicit_admission;
        struct output_block {
            ggml_tensor * tensor;
            size_t first, size;
        };
        for (size_t first = 0; first < used;) {
            ggml_reset(ctx.get());
            size_t matrix_cursor = matrix_route_bytes;
            auto * graph = ggml_new_graph_custom(ctx.get(), graph_nodes, false);
            std::array<output_block, 64> outputs;
            size_t batch_count = 0, rows = 0;
            for (; first < used && batch_count < outputs.size();) {
                size_t n = 1;
                if (rows == row_limit) { break; }
                const size_t available = std::min<size_t>(256, row_limit - rows);
                while (n < available && first + n < used && routes[first + n].expert == routes[first].expert) { ++n; }
                const auto & w = bindings[routes[first].expert];
                if (!w[0] || !w[1] || !w[2]) { throw std::runtime_error("matrix window binding missing"); }
                auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, p->up->ne[0], n);
                bind_to(input, matrix_buffer.get(), matrix_cursor);
                for (size_t i = 0; i < n; ++i) {
                    ggml_backend_tensor_set(input, routes[first + i].input, i*in_bytes, in_bytes);
                }
                auto * up = ggml_mul_mat(ctx.get(), w[0], input);
                auto * gate = ggml_mul_mat(ctx.get(), w[1], input);
                auto * act = ggml_swiglu_split(ctx.get(), gate, up);
                auto * down = ggml_mul_mat(ctx.get(), w[2], act);
                for (auto * tensor : {up, gate, act, down}) {
                    if (!ggml_backend_supports_op(backend, tensor)) { return false; }
                    bind_to(tensor, matrix_buffer.get(), matrix_cursor);
                }
                ggml_build_forward_expand(graph, down);
                outputs[batch_count++] = {down, first, n};
                first += n;
                rows += n;
            }
            if (!batch_count) { throw std::runtime_error("empty matrix window submission"); }
            llama_expert_timer submit(profile ? &profile->submit_ns : nullptr);
            const auto status = ggml_backend_graph_compute_async(backend, graph);
            submit.next(profile ? &profile->wait_ns : nullptr);
            ggml_backend_synchronize(backend);
            submit.next(nullptr);
            if (status != GGML_STATUS_SUCCESS) { throw std::runtime_error("matrix window Metal failure"); }
            if (profile && graph_timing) {
                uint64_t ns = 0, commands = 0;
                if (graph_timing(backend, &ns, &commands)) {
                    profile->gpu_command_ns += ns;
                    profile->gpu_command_buffers += commands;
                    ++profile->gpu_timed_submissions;
                }
            }
            for (size_t j = 0; j < batch_count; ++j) {
                const auto & block = outputs[j];
                for (size_t i = 0; i < block.size; ++i) {
                    ggml_backend_tensor_get(block.tensor, routes[block.first + i].output, i*out_bytes, out_bytes);
                }
            }
            matmuls += 3 * rows;
            ++submissions;
        }
        projections += 3 * count;
        calls += count;
        return true;
    }

    ~llama_expert_moe() {
        if (route_stage) publication->cancel();
        end_graph();
    }
    bool is_published(const ggml_tensor * ids) const {
        return publication && ids && ids->op == GGML_OP_CONT && ids->op_params[0] == GGML_METAL_ROUTE_MARKER;
    }
    bool shared_publication(const ggml_tensor * up) const {
        if (!shared_overlap) return false;
        for (size_t i = 0; i < n_layers; ++i) if (layers[i].up == up && layers[i].publication_generation) return true;
        return false;
    }
    bool has_graph_lookahead() const { return graph_predictor || bool(publication); }
    bool has_lookahead() const { return bool(lookahead) || bool(publication); }
    void begin_graph(bool decode) { if (lookahead) { lookahead->begin_graph(decode); } }
    void end_graph() {
        if (route_stage) {
            publication->abandon();
            if (!publication->pending()) route_stage->stop();
        } else if (publication) {
            publication->cancel(); published_demand.reset(); *published_resident = {}; published_count = 0; published_layer = nullptr;
        }
        if (lookahead) { lookahead->end_graph(profile); }
    }

    size_t admission_working_bytes() const { return 3 * max_allocation + max_payload + 2 * queue_headroom; }

    size_t optional_bytes() const {
        return metadata_charge.size() + scratch_charge.size() + readers_charge.size() + predictor_meta_charge.size() +
            predictor_scratch_charge.size() + publication_charge.size() + (lookahead ? lookahead->memory_bytes() : 0);
    }
    size_t prefetch_candidates() const { return candidate_limit; }
    bool priority_reads() const { return bool(readers); }
    early_io::pool * window_readers() const { return readers.get(); }
    bool address_batches() const { return batch_routes; }
    uint64_t completed() const { return calls; }
    uint64_t overlapped_groups() const { return pipeline_groups; }
    uint64_t resident_starts() const { return resident_groups; }
};
