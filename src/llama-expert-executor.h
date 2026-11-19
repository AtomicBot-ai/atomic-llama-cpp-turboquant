#pragma once

#include "llama-expert-store.h"

#include <memory>
#include <array>
#include <vector>

struct llama_memory_contract;
struct llama_memory_reclaimer;
struct llama_model;
class llama_prefill_wide_weights;

struct llama_expert_preview_layer {
    const ggml_tensor * gate;
    const ggml_tensor * up;
    const ggml_tensor * next_router;
    std::array<const ggml_tensor *, 3> next_weights; // gate, up, down
};

struct llama_expert_execution_stats {
    llama_expert_store_stats store;
    llama_memory_budget_snapshot memory;
    uint64_t projections = 0;
    uint64_t matmuls = 0;
    size_t minimum_bytes = 0;
    uint64_t submissions = 0;
    uint64_t combined_moe = 0;
    uint64_t moe_pipeline_groups = 0, moe_resident_starts = 0;
    size_t acceleration_bytes = 0, prefetch_candidates = 0;
};

// CPU graph bridge with a separate, synchronous compute backend. Only routed
// projections are replaced; router, activation, weighting and shared FFN stay in
// the original graph. Each inner matmul has one expert and at most 32 rows;
// Metal can submit up to four independent waves while retaining their leases.
// Qualified LFM/Qwen3.5 MoE matrix and Qwen4Exp vector batches can read up to four subsequent demand experts
// during that GPU submission, without extending work past the callback boundary.
class llama_expert_executor {
public:
    llama_expert_executor(std::shared_ptr<llama_expert_source> source,
                          const std::vector<const ggml_tensor *> & weights,
                          ggml_backend_dev_t device, size_t budget_bytes,
                          std::shared_ptr<llama_memory_budget> shared_budget = {},
                          const llama_memory_contract * discrete_contract = nullptr,
                          const std::vector<const ggml_tensor *> & cpu_weights = {});
    ~llama_expert_executor();
    void configure_rows(std::shared_ptr<llama_expert_source> source, const ggml_tensor * table);
    ggml_tensor * build_rows(ggml_context * ctx, ggml_tensor * ids);
    void configure_token_rows(std::shared_ptr<llama_expert_source> source, const ggml_tensor * table);
    ggml_tensor * build_token_rows(ggml_context * ctx, ggml_tensor * ids);
    ggml_tensor * build(ggml_context * ctx, const ggml_tensor * weight, ggml_tensor * input, ggml_tensor * ids);
    // Qualified short-row FP32 Metal profile. Null preserves the ordinary graph.
    // Admit optional acceleration after mandatory context/graph allocation.
    void configure_acceleration(bool dflash_verify = false, bool profile_overlap = false);
    void configure_resident_layers(const llama_model & model);
    const std::array<ggml_tensor *, 3> * resident_layer(int il) const;
    bool can_retain_prefill_tail(size_t routed_layers) const;
    size_t retain_prefill_tail(llama_prefill_wide_weights & weights, int first_layer);
    bool has_prefill_tail() const;
    void release_prefill_tail();
    bool automatic_acceleration() const;
    bool combined_moe_enabled() const;
    bool fused_window_moe_enabled() const;
    ggml_tensor * build_moe(ggml_context * ctx, const ggml_tensor * up, const ggml_tensor * gate,
                            const ggml_tensor * down, ggml_tensor * input, ggml_tensor * ids,
                            const ggml_tensor * router = nullptr,
                            const std::array<const ggml_tensor *, 3> & future = {},
                            ggml_backend_sched_t predictor_sched = nullptr);
    bool moe_route_published(const ggml_tensor * ids) const;
    bool moe_shared_publication(const ggml_tensor * up) const;
    bool moe_graph_lookahead_enabled() const;
    bool moe_lookahead_enabled() const;
    void moe_lookahead_begin(bool decode);
    void moe_lookahead_end();
    const char * error() const;
    llama_expert_execution_stats stats() const;
    bool resize(size_t target); // synchronized; auto mode invalidates previously built expert graphs
    bool resize_discrete(size_t host_target, size_t device_target, size_t scratch_headroom);
    void set_discrete_read_probe(bool (*callback)(void *) noexcept, void * user, bool worker,
            llama_expert_read_probe submitted_read);
    void trim();
    // Empty unless size-aware reclamation is explicitly qualified and enabled.
    llama_memory_reclaimer allocation_reclaimer();
    void set_profile(bool enabled);
    void clear_window_decode_advice(); // idle-only option/policy disable boundary
    bool can_layer_window() const;
    // Private, idle-only route handoff for a qualified prefill window.
    bool can_batch_window() const;
    bool window_weights_fit_without_eviction() const;
    bool can_batch_window_layer(const std::array<const ggml_tensor *, 3> & weights) const;
    const char * window_projection_name(const ggml_tensor * node) const;
    static constexpr size_t cross_window_workers = 8;
    bool prepare_cross_window(size_t rows, size_t routes);
    bool compute_cross_window(ggml_tensor * const * nodes, size_t count);
    void enable_fused_window_moe();
    // Borrowed for one admitted layer group. The enclosing window leases own every pointer.
    using window_bindings = std::array<std::array<ggml_tensor *, 3>, 1024>; // up, gate, down by expert ID
    void compute_fused_window_moe(ggml_tensor * const * nodes, size_t count, const window_bindings * bindings);
    bool prepare_matrix_window_moe();
    bool compute_matrix_window_moe(ggml_tensor * const * nodes, size_t count, const window_bindings & bindings);
    void use_cross_window_output(const ggml_tensor * node);
    size_t window_expert_group_bytes(const std::array<const ggml_tensor *, 3> & weights) const;
    size_t window_expert_capacity(size_t group_bytes = 0, size_t queued_experts = 0) const;
    using window_leases = std::vector<llama_expert_store::lease>;
    void pin_window_experts(const std::array<const ggml_tensor *, 3> & weights,
            const int64_t * ids, size_t count, window_leases & leases, bool adjacent_ranges, window_bindings * bindings = nullptr);
    using window_demand = std::unique_ptr<llama_expert_store::demand_batch>;
    bool can_pipeline_window() const;
    window_demand enqueue_window_experts(const std::array<const ggml_tensor *, 3> & weights,
            const int64_t * ids, size_t count);
    void drain_window_experts(window_demand & demand, window_leases & leases, window_bindings * bindings = nullptr);

    // Synchronized Qwen4Exp/Metal phase switch. Auto MoE stays allocated but
    // inactive inside a window; zero restores its decode capabilities.
    bool set_layer_window(size_t budget, bool vector_batch = false);
    void set_read_overlap(bool enabled); // idle-only; bounded by each projection's budget
    bool set_single_expert_map(bool enabled); // idle-only; Metal diagnostic control
    // Opt-in exact reads for the next independent projection; LRU, Metal only.
    bool set_early_read(bool enabled);
    void bind_early_graph(ggml_cgraph * graph);
    bool set_route_index(bool enabled); // idle-only; explicitly admitted bounded index
    bool get_profile(llama_expert_profile_data & result) const;
    bool set_policy(const std::string & policy);
    bool select_decode_policy(size_t tokens, size_t ubatch); // idle boundary, returns whether ClassLRU is active
    void set_trace(llama_expert_trace_callback callback, void * user);
    void set_route_observer(llama_expert_route_callback callback, void * user);
    // Experimental, idle-only: 0 off, 1 predictor only, 2 matrix-tail prefetch,
    // 3 original first-window prefetch (diagnostic control).
    // Caller retains immutable, host-readable F32 router tensors. No graph edits.
    bool set_preview(int mode, const std::vector<llama_expert_preview_layer> & layers = {});
    // Bind optional work to the graph about to execute, including graph reuse.
    // No graph pointers survive this call; unbound previews fail closed.
    void bind_preview_graph(ggml_cgraph * graph);
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
