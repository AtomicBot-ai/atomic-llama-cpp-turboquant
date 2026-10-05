#include "llama-kv-stream-plan.h"
#include "testing.h"

#include <cstdint>

namespace {

constexpr uint32_t N_TARGET_LAYERS = 16;

} // namespace

int main() {
    testing t;

    t.test("repeated deadline misses grow the ring by one balanced layer epoch", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                 = 160;
        params.layer_count                      = N_TARGET_LAYERS;
        params.active_pages_per_layer           = 12;
        params.minimum_ring_slots               = 16;
        params.previous_resident_pages_per_layer = 9;
        params.previous_ring_slots              = 16;
        params.deadline_miss_ratio               = 0.12;
        params.copy_engine_busy_ratio            = 0.70;
        params.starved_evaluations               = 2;
        params.grow_hysteresis_evaluations       = 3;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition changed", partition.changed);
        t.assert_equal(uint32_t(8), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(32), partition.ring_slots);
        t.assert_equal(uint32_t(0), partition.starved_evaluations);
    });
    t.test("feedback cannot demote beyond one epoch past the overlap target", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 7010;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 716;
        params.minimum_ring_slots                = 18;
        params.previous_resident_pages_per_layer = 416;
        params.previous_ring_slots               = 354;
        params.deadline_miss_ratio                = 0.10;
        params.copy_engine_busy_ratio             = 0.80;
        params.starved_evaluations                = 2;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition remains stable", !partition.changed);
        t.assert_equal(uint32_t(416), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(354), partition.ring_slots);
    });
    t.test("runaway feedback partition heals to the bounded floor", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 7010;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 716;
        params.minimum_ring_slots                = 18;
        params.previous_resident_pages_per_layer = 368;
        params.previous_ring_slots               = 1122;
        params.deadline_miss_ratio                = 0.10;
        params.copy_engine_busy_ratio             = 0.80;
        params.starved_evaluations                = 2;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition repairs the resident boundary", partition.changed);
        t.assert_equal(uint32_t(416), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(354), partition.ring_slots);
        t.assert_equal(uint32_t(0), partition.starved_evaluations);
    });
    t.test("PCIe saturation does not sacrifice more resident pages", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 160;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 12;
        params.minimum_ring_slots                = 16;
        params.previous_resident_pages_per_layer = 9;
        params.previous_ring_slots               = 16;
        params.deadline_miss_ratio                = 0.20;
        params.copy_engine_busy_ratio             = 0.99;
        params.starved_evaluations                = 10;
        params.grow_hysteresis_evaluations        = 3;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition remains stable", !partition.changed);
        t.assert_equal(uint32_t(9), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(16), partition.ring_slots);
    });
    t.test("undersized ring jumps to one-layer overlap target despite copy pressure", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 630;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 257;
        params.minimum_ring_slots                = 22;
        params.previous_resident_pages_per_layer = 38;
        params.previous_ring_slots               = 22;
        params.deadline_miss_ratio                = 0.40;
        params.copy_engine_busy_ratio             = 0.99;
        params.starved_evaluations                = 2;
        params.grow_hysteresis_evaluations        = 3;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition changed", partition.changed);
        t.assert_equal(uint32_t(23), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(262), partition.ring_slots);
    });
    t.test("decode transition selects its overlap target before feedback exists", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 6971;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 673;
        params.minimum_ring_slots                = 11;
        params.previous_resident_pages_per_layer = 435;
        params.previous_ring_slots               = 11;
        params.evaluations_since_repartition     = 0;
        params.entering_decode_layout            = true;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("decode transition changes immediately", partition.changed);
        t.assert_equal(uint32_t(418), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(283), partition.ring_slots);
    });
    t.test("copy pressure stops demotion after overlap target is reached", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 630;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 257;
        params.minimum_ring_slots                = 22;
        params.previous_resident_pages_per_layer = 23;
        params.previous_ring_slots               = 262;
        params.deadline_miss_ratio                = 0.20;
        params.copy_engine_busy_ratio             = 0.90;
        params.starved_evaluations                = 10;
        params.grow_hysteresis_evaluations        = 3;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition remains stable", !partition.changed);
        t.assert_equal(uint32_t(23), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(262), partition.ring_slots);
    });
    t.test("near-saturated copy traffic does not trigger a disruptive feedback epoch", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 6971;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 673;
        params.minimum_ring_slots                = 11;
        params.previous_resident_pages_per_layer = 418;
        params.previous_ring_slots               = 283;
        params.deadline_miss_ratio                = 0.053;
        params.copy_engine_busy_ratio             = 0.848;
        params.ring_peak_occupancy_ratio          = 1.0;
        params.starved_evaluations                = 2;
        params.evaluations_since_repartition      = 128;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("near-saturated partition remains stable", !partition.changed);
        t.assert_equal(uint32_t(418), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(283), partition.ring_slots);
        t.assert_equal(uint32_t(0), partition.starved_evaluations);
    });
    t.test("light utilization does not promote above overlap target", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 630;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 257;
        params.minimum_ring_slots                = 22;
        params.previous_resident_pages_per_layer = 23;
        params.previous_ring_slots               = 262;
        params.deadline_miss_ratio                = 0.0;
        params.copy_engine_busy_ratio             = 0.25;
        params.ring_peak_occupancy_ratio          = 0.20;
        params.overprovisioned_evaluations        = 7;
        params.shrink_hysteresis_evaluations      = 8;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition remains at target", !partition.changed);
        t.assert_equal(uint32_t(23), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(262), partition.ring_slots);
    });
    t.test("constrained pool demotes all resident pages when target is unreachable", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 39;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 65;
        params.minimum_ring_slots                = 23;
        params.previous_resident_pages_per_layer = 1;
        params.previous_ring_slots               = 23;
        params.deadline_miss_ratio                = 0.50;
        params.copy_engine_busy_ratio             = 0.99;
        params.starved_evaluations                = 2;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition changed", partition.changed);
        t.assert_equal(uint32_t(0), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(39), partition.ring_slots);
    });
    t.test("non-positive overlap target is rejected", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 160;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 12;
        params.minimum_ring_slots                = 16;
        params.previous_resident_pages_per_layer = 9;
        params.previous_ring_slots               = 16;
        params.target_ring_working_set_ratio      = 0.0;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is rejected", !partition.valid);
    });
    t.test("partition cooldown accumulates pressure without repeatedly resetting residency", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 160;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 12;
        params.minimum_ring_slots                = 16;
        params.previous_resident_pages_per_layer = 9;
        params.previous_ring_slots               = 16;
        params.deadline_miss_ratio               = 0.12;
        params.copy_engine_busy_ratio            = 0.70;
        params.starved_evaluations               = 2;
        params.evaluations_since_repartition     = 2;

        auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("cooldown preserves the resident boundary", !partition.changed);
        t.assert_equal(uint32_t(3), partition.starved_evaluations);

        params.evaluations_since_repartition = params.repartition_cooldown_evaluations;
        partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition changes after cooldown", partition.changed);
        t.assert_equal(uint32_t(8), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(32), partition.ring_slots);
    });
    t.test("sustained overprovision promotes one balanced resident epoch", [](testing & t) {
        llama_kv_stream_partition_params params;
        params.total_pool_pages                  = 160;
        params.layer_count                       = N_TARGET_LAYERS;
        params.active_pages_per_layer            = 12;
        params.minimum_ring_slots                = 16;
        params.previous_resident_pages_per_layer = 8;
        params.previous_ring_slots               = 32;
        params.deadline_miss_ratio                = 0.0;
        params.copy_engine_busy_ratio             = 0.25;
        params.ring_peak_occupancy_ratio          = 0.20;
        params.overprovisioned_evaluations        = 7;
        params.shrink_hysteresis_evaluations      = 8;

        const auto partition = llama_kv_stream_partition_adapt(params);
        t.assert_true("partition is valid", partition.valid);
        t.assert_true("partition changed", partition.changed);
        t.assert_equal(uint32_t(9), partition.resident_pages_per_layer);
        t.assert_equal(uint32_t(16), partition.ring_slots);
        t.assert_equal(uint32_t(0), partition.overprovisioned_evaluations);
    });
    t.test("cumulative CUDA feedback becomes one bounded evaluation delta", [](testing & t) {
        const auto delta = llama_kv_stream_feedback_delta_make(
            { 145, 17 }, { 120, 12 });
        t.assert_true("feedback delta is valid", delta.valid);
        t.assert_true("feedback contains a new evaluation", delta.has_evaluation);
        t.assert_equal(uint64_t(25), delta.deadline_samples);
        t.assert_equal(uint64_t(5), delta.deadline_misses);
        t.assert_true("deadline ratio is exact",
            std::abs(delta.deadline_miss_ratio - 0.20) < 1e-12);

        const auto unchanged = llama_kv_stream_feedback_delta_make(
            { 145, 17 }, { 145, 17 });
        t.assert_true("unchanged counters are valid", unchanged.valid);
        t.assert_true("unchanged counters do not invent an evaluation",
            !unchanged.has_evaluation);

        const auto reset = llama_kv_stream_feedback_delta_make(
            { 3, 1 }, { 145, 17 });
        t.assert_true("counter reset is rejected", !reset.valid);

        const auto impossible = llama_kv_stream_feedback_delta_make(
            { 150, 30 }, { 145, 17 });
        t.assert_true("more misses than samples are rejected", !impossible.valid);
    });

    return t.summary();
}
