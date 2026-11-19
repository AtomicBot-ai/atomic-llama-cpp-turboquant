#pragma once

// Fork-owned experimental C++ controls. Ordinary llama.cpp APIs stay in llama.h.
#include "llama.h"

#include "llama-expert-trace.h"
#include "llama-expert-profile.h"

// Idle-only override and counters for managed Metal prefill windows.
// On a window compute failure, discard the context (partially advanced state).
struct llama_prefill_window_data {
    bool enabled = false, failed = false;
    uint64_t windows = 0, tokens = 0, tails = 0, fallbacks = 0;
};
LLAMA_API bool llama_prefill_window_set(struct llama_context * ctx, bool enabled);
LLAMA_API bool llama_prefill_window_get(const struct llama_context * ctx, llama_prefill_window_data * result);

// Fixed-budget diagnostic on a single discrete CUDA GPU.
LLAMA_API llama_model * llama_load_discrete_model(const char * path, llama_model_params params,
    ggml_backend_dev_t device, size_t host_cap, size_t device_cap, bool stream_experts = false,
    bool stream_cpu_layers_on_cpu = false, bool stream_cuda_boundary_reduction = false);

// Idle-only profiling control/snapshot. Enabling resets cumulative phase times.
LLAMA_API bool llama_expert_profile_set(struct llama_context * ctx, bool enabled);
LLAMA_API bool llama_expert_profile_get(const struct llama_context * ctx, llama_expert_profile_data * result);
LLAMA_API bool llama_atomic_qwen35_profile_set(struct llama_model * target, uint32_t layers);
LLAMA_API bool llama_atomic_dflash2_enable(struct llama_model * target);
constexpr size_t llama_atomic_dflash2_slot_staging_bytes(uint32_t context) {
    return size_t(context == 4096 ? 128 : 32) << 20;
}
LLAMA_API struct llama_model * llama_atomic_dflash2_load(
        struct llama_model * target, const char * path, struct llama_model_params params);
// Idle-only diagnostic control for bounded known-route reads during GPU work.
// Defaults on for LFM/Qwen3.5 MoE Metal matrix waves; falls back when headroom is low.
LLAMA_API bool llama_expert_read_overlap_set(struct llama_context * ctx, bool enabled);
// Idle-only Metal control: prepare known single-expert maps in existing workspace.
// Defaults on for qualified LFM2 MoE and Qwen3.5 MoE matrix waves.
LLAMA_API bool llama_expert_single_map_set(struct llama_context * ctx, bool enabled);
// Idle-only experimental route index; admission failure preserves the current mode.
// Budgeted Qwen3.5 MoE Metal contexts attempt admission after mandatory allocations.
LLAMA_API bool llama_expert_route_index_set(struct llama_context * ctx, bool enabled);
// Idle-only diagnostic, default off. Exact next-projection reads, Metal/LRU only.
// Incompatible with speculative preview and demand-only ATET tracing.
LLAMA_API bool llama_expert_early_read_set(struct llama_context * ctx, bool enabled);

// Install/remove only while idle. Trace users must outlive the context or detach
// before destruction. Captures UMA demand access, not asynchronous leases/resize.
// Explicit override before the first evaluation: "lru" or "class-lru".
// Failure leaves the existing policy intact, including insufficient floor space.
LLAMA_API bool llama_expert_cache_policy_set(struct llama_context * ctx, const char * policy);
// Decode one prompt slice with the uncached token count for the whole prefill.
// The hint applies only to this call and does not alter batch contents.
LLAMA_API int32_t llama_expert_prefill_decode(struct llama_context * ctx, struct llama_batch_ext * batch,
                                           size_t uncached_prompt_tokens);
LLAMA_API bool llama_expert_trace_set(struct llama_context * ctx, llama_expert_trace_callback callback, void * user);
// Experimental Qwen3.5 MoE / Metal preview: 0 off, 1 predictor only,
// 2 matrix-tail prefetch, 3 original first-window prefetch (diagnostic control).
// Configure while idle; unsupported profiles fail explicitly. Default is off.
LLAMA_API bool llama_expert_preview_set(struct llama_context * ctx, int mode);
// Idle-only observer control. Callback storage must outlive the context or detach.
LLAMA_API bool llama_expert_route_observer_set(struct llama_context * ctx, llama_expert_route_callback callback, void * user);

// Experimental streaming counters. Memory fields describe the shared engine
// ledger when enabled, otherwise only expert/workspace accounting. Query idle.
struct llama_expert_streaming_data {
    size_t requested, applied, used, high_water, minimum_bytes;
    uint64_t hits, misses, evictions, logical_read_bytes, projections, matmuls;
    uint64_t buffer_allocations, buffer_reuses, submissions;
    uint64_t packed_reads, packed_reuses;
    uint64_t parallel_batches, parallel_reads;
    uint64_t combined_moe;
    uint64_t moe_pipeline_groups, moe_resident_starts;
    uint64_t reclaim_calls, reclaim_evictions, reclaim_bytes;
};
LLAMA_API bool llama_get_expert_streaming_stats(const struct llama_context * ctx, llama_expert_streaming_data * result);

#include "llama-memory-policy.h"

// Configure/query while idle. System signals are supported on macOS; false
// leaves the existing configuration intact. External samples are thread-safe.
// Automatic changes use the same synchronized boundary as manual resize. Call
// llama_memory_budget_apply periodically while idle (the server does this).
LLAMA_API bool llama_memory_policy_configure(struct llama_context * ctx,
    const llama_memory_policy_config * config, bool system_source);
LLAMA_API bool llama_memory_pressure_submit(struct llama_context * ctx, const llama_memory_pressure_sample * sample);
LLAMA_API bool llama_memory_policy_get(const struct llama_context * ctx, llama_memory_policy_data * result);
LLAMA_API const char * llama_memory_policy_reason_name(llama_memory_policy_reason reason);

enum llama_budget_status { LLAMA_BUDGET_IDLE, LLAMA_BUDGET_PENDING, LLAMA_BUDGET_APPLIED, LLAMA_BUDGET_REJECTED };
struct llama_budget_data {
    size_t requested, applied, minimum, used, high_water, released, runtime_envelope;
    size_t categories[7]; // weights, experts, state, scratch, drafter, io, runtime
    uint64_t request_id, completed_id;
    llama_budget_status status;
};
// Request is thread-safe during evaluation; latest pending request wins. The
// target is applied at the next synchronized graph boundary. 0 means unavailable.
LLAMA_API uint64_t llama_memory_budget_request(struct llama_context * ctx, size_t target);
// Apply while idle, or let evaluation apply automatically. Do not race this or
// the snapshot query with evaluation. Rejection leaves the old cap/session intact.
LLAMA_API bool llama_memory_budget_apply(struct llama_context * ctx);
LLAMA_API bool llama_memory_budget_get(const struct llama_context * ctx, llama_budget_data * result);
// Diagnostic control on an idle context, between decode calls. Headroom is a
// caller-supplied allowance for transient allocations in each domain.
LLAMA_API bool llama_discrete_budget_resize(struct llama_context * ctx, size_t host_target, size_t device_target, size_t scratch_headroom);

// Diagnostic callback at a synchronized microbatch boundary. Return false to
// fail the evaluation. The midpoint callback can cancel once before publication.
LLAMA_API bool llama_discrete_prefill_probe_set(struct llama_context * ctx,
        bool (*mid_read)(void *) noexcept, bool (*boundary)(struct llama_context *, void *) noexcept, void * user,
        bool worker = false, bool (*submitted_read)(void *, int, size_t, void *, size_t) = nullptr);


// Server-owned complete text prefix files. Never use these controls for
// user-supplied slots or without the exact expected token prefix.
LLAMA_API bool llama_atomic_prefix_supported(const struct llama_context * ctx);
LLAMA_API bool llama_atomic_prefix_ready(const struct llama_context * ctx, size_t count);
LLAMA_API bool llama_atomic_prefix_restore(struct llama_context * ctx, const char * path,
        const llama_token * expected, size_t count);

// Idle-only two-domain snapshot. Indices are RAM (0) and VRAM (1).
// Driver allocations and the OS file cache are outside these engine counters.
struct llama_discrete_budget_data {
    size_t requested[2], applied[2], used[2], high_water[2], released[2];
};
LLAMA_API bool llama_discrete_budget_get(const struct llama_context * ctx, llama_discrete_budget_data * result);
