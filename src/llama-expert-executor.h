#pragma once

#include "llama-expert-store.h"

#include <memory>
#include <array>
#include <vector>

struct llama_memory_contract;

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
    const char * error() const;
    llama_expert_execution_stats stats() const;
    bool resize(size_t target); // only at a synchronized evaluation boundary
    bool resize_discrete(size_t host_target, size_t device_target, size_t scratch_headroom);
    void set_discrete_read_probe(bool (*callback)(void *) noexcept, void * user, bool worker,
            llama_expert_read_probe submitted_read);
    void trim();
    void set_profile(bool enabled);
    void clear_window_decode_advice(); // idle-only option/policy disable boundary
    bool can_layer_window() const;
    bool set_layer_window(bool enabled); // synchronized, Qwen4Exp/Metal only; preserves demand read advice
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
