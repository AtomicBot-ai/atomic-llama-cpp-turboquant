#include "llama-kv-stream-plan.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool checked_mul(uint64_t a, uint64_t b, uint64_t & result) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max()/a) {
        return false;
    }

    result = a*b;
    return true;
}

} // namespace

llama_kv_stream_partition llama_kv_stream_partition_adapt(
        const llama_kv_stream_partition_params & params) {
    llama_kv_stream_partition result;
    result.resident_pages_per_layer = params.previous_resident_pages_per_layer;
    result.ring_slots = params.previous_ring_slots;

    auto fail_partition = [&](const char * message) {
        result.valid = false;
        result.error = message;
        return result;
    };

    if (params.total_pool_pages == 0 || params.layer_count == 0) {
        return fail_partition("KV stream partition geometry must be non-zero");
    }
    if (params.grow_hysteresis_evaluations == 0 ||
        params.shrink_hysteresis_evaluations == 0 ||
        params.repartition_cooldown_evaluations == 0) {
        return fail_partition("KV stream partition hysteresis must be non-zero");
    }
    const auto ratio_valid = [](double value) {
        return value >= 0.0 && value <= 1.0;
    };
    if (!ratio_valid(params.deadline_miss_ratio) ||
        !ratio_valid(params.copy_engine_busy_ratio) ||
        !ratio_valid(params.ring_peak_occupancy_ratio)) {
        return fail_partition("KV stream partition metrics must be ratios");
    }
    if (!std::isfinite(params.target_ring_working_set_ratio) ||
            params.target_ring_working_set_ratio <= 0.0) {
        return fail_partition("KV stream target working-set ratio must be positive");
    }
    if (params.previous_resident_pages_per_layer > params.active_pages_per_layer) {
        return fail_partition("resident KV pages exceed active pages");
    }

    uint64_t resident_pages = 0;
    if (!checked_mul(params.previous_resident_pages_per_layer, params.layer_count, resident_pages) ||
        resident_pages + params.previous_ring_slots != params.total_pool_pages) {
        return fail_partition("previous KV stream partition does not cover the fixed pool");
    }
    if (params.previous_ring_slots < params.minimum_ring_slots) {
        return fail_partition("previous KV stream ring is below its minimum");
    }

    constexpr double MISS_THRESHOLD = 0.01;
    // A repartition invalidates compact resident addresses. Do not pay that
    // transition cost when the copy engine already has too little headroom
    // for a larger lookahead ring to increase sustained throughput.
    constexpr double COPY_SATURATED  = 0.80;
    constexpr double COPY_LIGHT      = 0.50;
    constexpr double RING_LIGHT      = 0.50;
    constexpr uint32_t FEEDBACK_GROWTH_EPOCHS = 1;

    uint32_t target_resident_pages = 0;
    const uint32_t maximum_resident_pages = std::min(
        params.active_pages_per_layer,
        (params.total_pool_pages - params.minimum_ring_slots)/params.layer_count);
    for (uint32_t resident = maximum_resident_pages;; --resident) {
        const uint32_t ring = params.total_pool_pages - resident*params.layer_count;
        const uint32_t streamed = params.active_pages_per_layer - resident;
        if (streamed == 0 || double(ring) >=
                params.target_ring_working_set_ratio*double(streamed)) {
            target_resident_pages = resident;
            break;
        }
        if (resident == 0) {
            break;
        }
    }
    const uint32_t feedback_resident_floor = target_resident_pages >
        FEEDBACK_GROWTH_EPOCHS ? target_resident_pages - FEEDBACK_GROWTH_EPOCHS : 0;
    const bool overgrown_ring =
        params.previous_resident_pages_per_layer < feedback_resident_floor;

    const bool below_overlap_target =
        params.previous_resident_pages_per_layer > target_resident_pages;
    const bool feedback_starved =
        params.previous_resident_pages_per_layer > feedback_resident_floor &&
        params.deadline_miss_ratio > MISS_THRESHOLD &&
        params.copy_engine_busy_ratio < COPY_SATURATED;
    const bool starved = below_overlap_target || feedback_starved;
    const bool overprovisioned = params.deadline_miss_ratio <= MISS_THRESHOLD &&
        params.copy_engine_busy_ratio < COPY_LIGHT &&
        params.ring_peak_occupancy_ratio < RING_LIGHT;

    result.starved_evaluations = starved ? params.starved_evaluations + 1 : 0;
    result.overprovisioned_evaluations = overprovisioned ?
        params.overprovisioned_evaluations + 1 : 0;

    const bool cooldown_complete = params.evaluations_since_repartition >=
        params.repartition_cooldown_evaluations;
    if (params.entering_decode_layout && below_overlap_target) {
        result.resident_pages_per_layer = target_resident_pages;
        result.ring_slots = params.total_pool_pages -
            result.resident_pages_per_layer*params.layer_count;
        result.starved_evaluations = 0;
        result.overprovisioned_evaluations = 0;
        result.changed = true;
    } else if (cooldown_complete && overgrown_ring) {
        result.resident_pages_per_layer = feedback_resident_floor;
        result.ring_slots = params.total_pool_pages -
            result.resident_pages_per_layer*params.layer_count;
        result.starved_evaluations = 0;
        result.overprovisioned_evaluations = 0;
        result.changed = true;
    } else if (cooldown_complete && starved &&
            result.starved_evaluations >= params.grow_hysteresis_evaluations &&
            result.resident_pages_per_layer > 0) {
        result.resident_pages_per_layer = below_overlap_target ?
            target_resident_pages : result.resident_pages_per_layer - 1;
        result.ring_slots = params.total_pool_pages -
            result.resident_pages_per_layer*params.layer_count;
        result.starved_evaluations = 0;
        result.overprovisioned_evaluations = 0;
        result.changed = true;
    } else if (cooldown_complete && overprovisioned &&
            result.overprovisioned_evaluations >= params.shrink_hysteresis_evaluations &&
            result.resident_pages_per_layer < target_resident_pages &&
            result.ring_slots >= params.minimum_ring_slots + params.layer_count) {
        ++result.resident_pages_per_layer;
        result.ring_slots -= params.layer_count;
        result.starved_evaluations = 0;
        result.overprovisioned_evaluations = 0;
        result.changed = true;
    }

    result.valid = true;
    return result;
}

llama_kv_stream_feedback_delta llama_kv_stream_feedback_delta_make(
        const llama_kv_stream_feedback_counters & current,
        const llama_kv_stream_feedback_counters & previous) {
    llama_kv_stream_feedback_delta result;
    if (current.deadline_samples < previous.deadline_samples ||
            current.deadline_misses < previous.deadline_misses) {
        result.error = "KV stream feedback counters moved backwards";
        return result;
    }

    result.deadline_samples = current.deadline_samples - previous.deadline_samples;
    result.deadline_misses = current.deadline_misses - previous.deadline_misses;
    if (result.deadline_misses > result.deadline_samples) {
        result.error = "KV stream deadline misses exceed samples";
        return result;
    }

    result.valid = true;
    result.has_evaluation = result.deadline_samples != 0;
    if (result.has_evaluation) {
        result.deadline_miss_ratio =
            double(result.deadline_misses)/double(result.deadline_samples);
    }
    return result;
}
