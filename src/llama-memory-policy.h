#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>

enum class llama_memory_mode { fixed, manual, automatic };
enum class llama_memory_pressure { unknown, normal, warning, critical };
enum class llama_memory_policy_reason {
    disabled, manual_override, waiting_signal, stale_signal, cooldown, headroom,
    at_floor, at_preferred, warning_shrink, critical_shrink, recovery_grow,
    preferred_limit, floor_conflict, apply_failed,
};

struct llama_memory_policy_config {
    llama_memory_mode mode = llama_memory_mode::fixed;
    size_t preferred = 0;
    size_t minimum = 0; // combined with the live engine floor, never overrides it
    size_t reserve = size_t(1024) * 1024 * 1024;
    size_t shrink_step = size_t(256) * 1024 * 1024;
    size_t grow_step = size_t(64) * 1024 * 1024;
    uint64_t interval_ms = 1000;
    uint64_t recovery_ms = 30000;
    uint64_t max_recovery_ms = 300000;
    uint64_t sample_ttl_ms = 5000;
};

struct llama_memory_pressure_sample {
    llama_memory_pressure level = llama_memory_pressure::unknown;
    size_t headroom = 0; // advisory bytes; never the Metal recommended working set
    bool headroom_valid = false;
};

struct llama_memory_policy_data {
    llama_memory_policy_config config;
    llama_memory_policy_reason reason = llama_memory_policy_reason::disabled;
    llama_memory_pressure level = llama_memory_pressure::unknown;
    size_t target = 0, minimum = 0, headroom = 0;
    bool headroom_valid = false;
    uint64_t recovery_delay_ms = 0, sample_age_ms = 0, decisions = 0;
    bool system_source = false;
};

// Deterministic controller: caller supplies monotonic time and serializes access.
// It proposes caps only; the runtime applies them at a quiescent graph boundary.
class llama_memory_policy {
public:
    bool configure(const llama_memory_policy_config & config, uint64_t now, size_t floor);
    bool observe(const llama_memory_pressure_sample & sample, uint64_t now);
    void manual(); // explicit manual requests suspend automatic recovery
    size_t decide(uint64_t now, size_t applied, size_t used, size_t floor);
    void failed() { state.reason = llama_memory_policy_reason::apply_failed; }
    const llama_memory_policy_data & status() const { return state; }
private:
    llama_memory_policy_data state;
    llama_memory_pressure_sample sample;
    llama_memory_pressure pending = llama_memory_pressure::unknown;
    uint64_t sample_at = 0, normal_since = 0, action_at = 0, clock = 0;
    bool have_sample = false, normal = false, acted = false, grew = false;
};

LLAMA_API const char * llama_memory_policy_reason_name(llama_memory_policy_reason reason);
uint64_t llama_memory_policy_now_ms();
