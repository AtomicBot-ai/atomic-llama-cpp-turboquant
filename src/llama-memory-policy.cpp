#include "llama-memory-policy.h"

#include <algorithm>
#include <chrono>

bool llama_memory_policy::configure(const llama_memory_policy_config & config, uint64_t now, size_t floor) {
    if ((config.mode != llama_memory_mode::fixed && config.mode != llama_memory_mode::manual &&
         config.mode != llama_memory_mode::automatic) || !config.preferred || config.preferred < floor ||
        config.minimum > config.preferred || !config.shrink_step || !config.grow_step ||
        !config.interval_ms || config.recovery_ms < config.interval_ms ||
        config.max_recovery_ms < config.recovery_ms || config.sample_ttl_ms < config.interval_ms) { return false; }
    *this = {};
    state.config = config;
    state.recovery_delay_ms = config.recovery_ms;
    state.minimum = std::max(config.minimum, floor);
    state.reason = config.mode == llama_memory_mode::automatic ? llama_memory_policy_reason::waiting_signal :
                   config.mode == llama_memory_mode::manual ? llama_memory_policy_reason::manual_override : llama_memory_policy_reason::disabled;
    clock = now;
    return true;
}

bool llama_memory_policy::observe(const llama_memory_pressure_sample & value, uint64_t now) {
    if (now < clock || (have_sample && now < sample_at) ||
        value.level < llama_memory_pressure::unknown || value.level > llama_memory_pressure::critical) { return false; }
    if (have_sample && now - sample_at > state.config.sample_ttl_ms) { normal = false; }
    sample = value;
    sample_at = now;
    have_sample = true;
    // Never lose a brief critical signal when normal arrives before a safe point.
    pending = std::max(pending, value.level);
    return true;
}

void llama_memory_policy::manual() {
    state.config.mode = llama_memory_mode::manual;
    state.reason = llama_memory_policy_reason::manual_override;
    normal = false;
}

size_t llama_memory_policy::decide(uint64_t now, size_t applied, size_t used, size_t floor) {
    if (now < clock || (have_sample && now < sample_at)) { return 0; }
    clock = now;
    state.minimum = std::max(floor, state.config.minimum);
    state.target = applied;
    state.sample_age_ms = have_sample ? now - sample_at : UINT64_MAX;
    state.level = std::max(sample.level, pending);
    state.headroom = sample.headroom;
    state.headroom_valid = have_sample && sample.headroom_valid && state.sample_age_ms <= state.config.sample_ttl_ms;
    const auto mode = state.config.mode;
    auto hold = [&](llama_memory_policy_reason reason) { state.reason = reason; return size_t(0); };
    auto propose = [&](size_t target, llama_memory_policy_reason reason) {
        state.target = target;
        state.reason = reason;
        action_at = now;
        acted = true;
        ++state.decisions;
        return target;
    };
    if (mode == llama_memory_mode::fixed) { return hold(llama_memory_policy_reason::disabled); }
    if (mode == llama_memory_mode::manual) { return hold(llama_memory_policy_reason::manual_override); }
    if (state.minimum > state.config.preferred) { return hold(llama_memory_policy_reason::floor_conflict); }
    if (applied > state.config.preferred) { return propose(state.config.preferred, llama_memory_policy_reason::preferred_limit); }
    const bool pressure = state.level >= llama_memory_pressure::warning;
    if (pressure) {
        normal = false;
        if (grew) {
            state.recovery_delay_ms += std::min(state.recovery_delay_ms,
                state.config.max_recovery_ms - state.recovery_delay_ms);
            grew = false;
        }
        if (applied <= state.minimum) { pending = llama_memory_pressure::unknown; return hold(llama_memory_policy_reason::at_floor); }
        // Critical bypasses throttling; warning is rate-limited, not discarded.
        if (state.level == llama_memory_pressure::warning && acted && now - action_at < state.config.interval_ms) {
            return hold(llama_memory_policy_reason::cooldown);
        }
        pending = llama_memory_pressure::unknown;
        const bool critical = state.level == llama_memory_pressure::critical;
        const size_t target = critical ? state.minimum : applied - std::min(state.config.shrink_step, applied - state.minimum);
        return propose(target, critical ? llama_memory_policy_reason::critical_shrink : llama_memory_policy_reason::warning_shrink);
    }
    pending = llama_memory_pressure::unknown;
    if (!have_sample || sample.level == llama_memory_pressure::unknown) {
        normal = false;
        return hold(llama_memory_policy_reason::waiting_signal);
    }
    if (state.sample_age_ms > state.config.sample_ttl_ms) {
        normal = false;
        return hold(llama_memory_policy_reason::stale_signal);
    }
    if (applied >= state.config.preferred) { return hold(llama_memory_policy_reason::at_preferred); }
    const size_t target = applied + std::min(state.config.grow_step, state.config.preferred - applied);
    const size_t future = target > used ? target - used : 0;
    if (!sample.headroom_valid || sample.headroom < state.config.reserve ||
        future > sample.headroom - state.config.reserve) {
        normal = false;
        return hold(llama_memory_policy_reason::headroom);
    }
    if (!normal) { normal = true; normal_since = now; }
    if (now - normal_since < state.recovery_delay_ms || (acted && now - action_at < state.config.interval_ms)) {
        return hold(llama_memory_policy_reason::cooldown);
    }
    grew = true;
    return propose(target, llama_memory_policy_reason::recovery_grow);
}

uint64_t llama_memory_policy_now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const char * llama_memory_policy_reason_name(llama_memory_policy_reason reason) {
    switch (reason) {
#define CASE(name) case llama_memory_policy_reason::name: return #name
        CASE(disabled); CASE(manual_override); CASE(waiting_signal); CASE(stale_signal);
        CASE(cooldown); CASE(headroom); CASE(at_floor); CASE(at_preferred); CASE(warning_shrink);
        CASE(critical_shrink); CASE(recovery_grow); CASE(preferred_limit); CASE(floor_conflict); CASE(apply_failed);
#undef CASE
    }
    return "unknown";
}
