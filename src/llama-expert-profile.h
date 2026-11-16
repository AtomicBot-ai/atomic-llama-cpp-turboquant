#pragma once

#include <chrono>
#include <cstdint>

// Wall-clock diagnostics for the synchronous expert path. Store phases include
// demand/known-route acquisitions (acquire_ns) and speculation (prefetch_ns).
// All phases are included in projection_ns. Query
// only while idle. No timestamps are sampled when profiling is disabled.
struct llama_expert_profile_data {
    uint64_t projection_ns = 0;
    uint64_t acquire_ns = 0;
    uint64_t admission_ns = 0;
    uint64_t eviction_search_ns = 0, eviction_candidates = 0;
    uint64_t allocation_ns = 0;
    // Direct-read tail clearing, nested within allocation_ns.
    uint64_t padding_ns = 0, padding_bytes = 0;
    uint64_t read_ns = 0;
    // Subphases of read_ns; transfer includes buffered I/O and OS page faults.
    uint64_t source_setup_ns = 0, source_validate_ns = 0, source_seek_ns = 0, source_transfer_ns = 0;
    uint64_t direct_read_bytes = 0;
    // Host acquisitions performed between GPU submission and completion wait.
    // Nested in compute/acquire; these are not measured GPU-overlap durations.
    uint64_t overlap_acquire_ns = 0, overlap_acquisitions = 0, overlap_read_bytes = 0;
    // Independent CPU predictor and speculative reads inside an existing GPU
    // submit/wait interval. Nested wall times, not measured overlap durations.
    uint64_t preview_ns = 0, prefetch_ns = 0, preview_generations = 0, preview_rows = 0;
    uint64_t preview_dot_ns = 0, preview_rank_ns = 0, preview_residency_ns = 0, preview_guard_ns = 0;
    // Service/work totals only: GPU command time is not a wall-clock overlap measurement.
    uint64_t preview_window_host_ns = 0, preview_window_gpu_ns = 0, preview_windows = 0;
    uint64_t preview_competing_windows = 0;
    uint64_t preview_skip_hot = 0, preview_skip_budget = 0, preview_skip_short = 0;
    uint64_t preview_bind_ns = 0, preview_skip_graph = 0, preview_skip_shape = 0;
    uint64_t prefetch_experts = 0, prefetch_slices = 0, prefetch_read_bytes = 0;
    // Exact next-projection reads inside the source tail GPU window. Service
    // totals are not a measurement of the CPU/GPU interval intersection.
    uint64_t early_bind_ns = 0, early_bound_pairs = 0, early_windows = 0;
    uint64_t early_acquire_ns = 0, early_window_gpu_ns = 0;
    uint64_t early_acquisitions = 0, early_read_bytes = 0, early_skip_hot = 0;
    uint64_t early_skip_budget = 0, early_skip_short = 0;
    uint64_t upload_ns = 0;
    uint64_t release_ns = 0;
    uint64_t graph_setup_ns = 0;
    uint64_t map_prepare_ns = 0, prepared_map_ops = 0;
    uint64_t route_index_ns = 0, route_indexed_projections = 0, route_index_fallbacks = 0, route_id_reads = 0;
    uint64_t gather_ns = 0;
    uint64_t input_ns = 0;
    uint64_t compute_ns = 0;
    uint64_t submit_ns = 0, wait_ns = 0;
    uint64_t gpu_command_ns = 0, gpu_command_buffers = 0, gpu_timed_submissions = 0;
    uint64_t readback_ns = 0;
    uint64_t scatter_ns = 0;
};

class llama_expert_timer {
    using clock = std::chrono::steady_clock;
    uint64_t * counter;
    clock::time_point start;
public:
    explicit llama_expert_timer(uint64_t * counter) : counter(counter) {
        if (counter) { start = clock::now(); }
    }
    ~llama_expert_timer() { next(nullptr); }
    void next(uint64_t * value) {
        if (counter || value) {
            const auto now = clock::now();
            if (counter) { *counter += std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count(); }
            start = now;
        }
        counter = value;
    }
};
