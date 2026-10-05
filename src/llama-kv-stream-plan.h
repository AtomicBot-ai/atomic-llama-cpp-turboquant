#pragma once

#include <cstdint>
#include <string>

struct llama_kv_stream_partition_params {
    uint32_t total_pool_pages       = 0;
    uint32_t layer_count            = 0;
    uint32_t active_pages_per_layer = 0;
    uint32_t minimum_ring_slots     = 0;

    uint32_t previous_resident_pages_per_layer = 0;
    uint32_t previous_ring_slots               = 0;

    double deadline_miss_ratio       = 0.0;
    double copy_engine_busy_ratio    = 0.0;
    double ring_peak_occupancy_ratio = 0.0;

    // Ring slots relative to the streamed pages consumed by one attention
    // layer. A value above one leaves room to begin the next layer early.
    double target_ring_working_set_ratio = 1.10;

    uint32_t starved_evaluations        = 0;
    uint32_t overprovisioned_evaluations = 0;
    uint32_t grow_hysteresis_evaluations = 3;
    uint32_t shrink_hysteresis_evaluations = 8;
    uint32_t evaluations_since_repartition = UINT32_MAX;
    uint32_t repartition_cooldown_evaluations = 64;

    // The first decode graph has no useful streaming feedback yet, but its
    // active working set is already known. Select the deterministic overlap
    // target immediately instead of spending several tokens in an undersized
    // prefill ring before hysteresis can react.
    bool entering_decode_layout = false;
};

struct llama_kv_stream_partition {
    bool valid = false;
    std::string error;
    bool changed = false;

    uint32_t resident_pages_per_layer = 0;
    uint32_t ring_slots               = 0;
    uint32_t starved_evaluations      = 0;
    uint32_t overprovisioned_evaluations = 0;
};

// Adjusts the boundary inside a fixed page pool. A growth step demotes exactly
// one page from every layer, turning those addresses into immediately reusable
// ring slots. Promotion performs the inverse operation after longer hysteresis.
llama_kv_stream_partition llama_kv_stream_partition_adapt(
    const llama_kv_stream_partition_params & params);

struct llama_kv_stream_feedback_counters {
    uint64_t deadline_samples = 0;
    uint64_t deadline_misses  = 0;
};

struct llama_kv_stream_feedback_delta {
    bool valid = false;
    bool has_evaluation = false;
    std::string error;
    uint64_t deadline_samples = 0;
    uint64_t deadline_misses  = 0;
    double deadline_miss_ratio = 0.0;
};

llama_kv_stream_feedback_delta llama_kv_stream_feedback_delta_make(
    const llama_kv_stream_feedback_counters & current,
    const llama_kv_stream_feedback_counters & previous);
