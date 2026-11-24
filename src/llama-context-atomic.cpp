#include "llama-context.h"
#include "llama-qwen35-metal-profile.h"
#include "llama-io.h"
#include "llama-mmap.h"
#include "llama-context-budget.h"
#include "llama-memory-contract.h"
#include "llama-memory-pressure.h"
#include "llama-model.h"
#include "llama-memory-hybrid-idx.h"

#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <exception>
#include <stdexcept>

bool llama_atomic_qwen35_profile_set(llama_model * target, uint32_t layers) {
    if (!target || !target->memory_contract || target->memory_contract->discrete.device ||
            !target->expert_source || target->arch != LLM_ARCH_QWEN35MOE || target->ftype() != LLAMA_FTYPE_MOSTLY_Q4_K_M ||
            target->hparams.n_layer() != 40 || !layers || layers > 40) { return false; }
    const auto device = target->dev_output();
    if (!device || std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "MTL")) { return false; }
    for (uint32_t i = 0; i < target->hparams.n_layer(); ++i) {
        const auto & layer = target->layers[i];
        if (target->dev_layer(i) != device || !target->streamed_experts.count(layer.ffn_gate_exps) ||
                !target->streamed_experts.count(layer.ffn_up_exps) || !target->streamed_experts.count(layer.ffn_down_exps) ||
                layer.ffn_gate_exps->type != GGML_TYPE_Q4_K || layer.ffn_up_exps->type != GGML_TYPE_Q4_K ||
                (layer.ffn_down_exps->type != GGML_TYPE_Q4_K && layer.ffn_down_exps->type != GGML_TYPE_Q6_K)) { return false; }
    }
    auto & contract = *target->memory_contract;
    const size_t cap = contract.ledger->snapshot(llama_memory_domain::host).requested;
    contract.qwen35_profile_layers = layers;
    contract.qwen35_fast_prefill_default = llama_qwen35_fast_prefill_default(cap);
    const char * grouped = std::getenv("LLAMA_QWEN35_GQA8");
    contract.qwen35_grouped_attention = grouped ? !std::strcmp(grouped, "1") :
            contract.qwen35_fast_prefill_default && (cap >= (size_t(10) << 30) || layers == 1);
    return true;
}

bool llama_atomic_dflash2_enable(llama_model * target) {
    return llama_atomic_dflash2_enable(target, false);
}

bool llama_atomic_dflash2_enable(llama_model * target, bool flash_long_profile) {
    return llama_atomic_dflash2_enable(target, flash_long_profile ? 4096u : 0u);
}

bool llama_atomic_dflash2_enable(llama_model * target, uint32_t flash_context) {
    if (flash_context && flash_context != 4096 && flash_context != 40960 && flash_context != 61440) { return false; }
    if (!target || !target->memory_contract || target->memory_contract->discrete.device || !target->expert_source) {
        return false;
    }
    const size_t cap = target->memory_contract->ledger->snapshot(llama_memory_domain::host).requested;
    const auto & hp = target->hparams;
    const bool flash_long = flash_context && cap >= (size_t(8) << 30) && cap <= (size_t(24) << 30) &&
        target->arch == LLM_ARCH_QWEN4EXP && hp.n_expert == 512;
    const bool flash = (flash_long || cap == (size_t(18) << 30)) && target->arch == LLM_ARCH_QWEN4EXP &&
        hp.n_layer() == 48 && target->streamed_ple;
    if (flash_context > 4096 && !(flash && flash_long)) { return false; }
    const bool qwen35 = (target->memory_contract->qwen35_profile_layers || cap == (size_t(8) << 30)) && target->arch == LLM_ARCH_QWEN35MOE &&
        hp.n_layer() == 40 && hp.n_embd == 2048 && hp.n_expert == 256 && hp.n_expert_used_max() == 8;
    const auto device = target->dev_output();
    if ((!flash && !qwen35) || !device || std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "MTL") != 0) {
        return false;
    }
    // One file action at a time on the inference thread. Reserve the bounded
    // host staging for saving or loading the draft appendix before evaluation.
    auto & contract = *target->memory_contract;
    // The 60k chat profile is only admitted when slot file actions are disabled.
    if (flash_context != 61440) {
        contract.draft_slot_staging = contract.ledger->reserve(llama_memory_domain::host,
                llama_memory_category::io, llama_atomic_dflash2_slot_staging_bytes(flash_long ? flash_context : (qwen35 ? 4096 : 1024)));
        if (!contract.draft_slot_staging) { return false; }
    }
    if (qwen35 && contract.qwen35_profile_layers == 40 &&
            (cap == (size_t(24) << 30) || cap == (size_t(26) << 30))) {
        const char * mode = std::getenv("LLAMA_QWEN35_RESIDENT_VERIFY");
        if (mode) {
            contract.qwen35_grouped_attention = !std::strcmp(mode, "1");
            contract.qwen35_resident_verify = contract.qwen35_grouped_attention || !std::strcmp(mode, "exact");
        } else {
            contract.qwen35_resident_verify = contract.qwen35_fast_prefill_default;
        }
    }
    contract.flash_next_dflash3 = flash_long;
    contract.flash_next_extended_ctx = flash_long && flash_context > 4096 ? flash_context : 0;
    contract.dflash2_target = true;
    return true;
}

bool llama_atomic_is_inkling_dspark(const llama_model * draft) {
    return draft && draft->hparams.inkling2_dspark;
}

static bool inkling_metal_target(llama_model * target) {
    if (!target || target->arch != LLM_ARCH_INKLING2_LOOP || !target->memory_contract ||
            !target->expert_source || target->hparams.n_layer() != 48 || target->hparams.n_embd != 2304) {
        return false;
    }
    const auto device = target->dev_output();
    return device && std::strcmp(ggml_backend_dev_name(device), "MTL0") == 0;
}

bool llama_atomic_inkling_lookup_enable(llama_model * target, uint32_t depth) {
    if (!inkling_metal_target(target) || depth != 7 ||
            (target->ftype() != LLAMA_FTYPE_MOSTLY_IQ2_XXS && target->ftype() != LLAMA_FTYPE_MOSTLY_IQ1_S) ||
            target->memory_contract->ledger->snapshot(llama_memory_domain::host).requested != (size_t(4096) << 20)) {
        return false;
    }
    target->memory_contract->inkling_lookup_depth = depth;
    return true;
}

bool llama_atomic_inkling_dspark_enable(llama_model * target) {
    if (!inkling_metal_target(target)) { return false; }
    target->memory_contract->inkling_dspark_target = true;
    return true;
}

llama_model * llama_atomic_budget_draft_load(llama_model * target, const char * path, llama_model_params params) {
    if (!target || !target->memory_contract ||
            (!target->memory_contract->dflash2_target && !target->memory_contract->inkling_dspark_target)) { return nullptr; }
    auto ledger = target->memory_contract->ledger;
    llama_allocation_scope admission(ledger, llama_memory_category::drafter);
    params.memory_budget_size = 0;
    params.load_mode = LLAMA_LOAD_MODE_NONE;
    params.use_extra_bufts = false;
    std::unique_ptr<llama_model> draft(llama_model_load_from_file(path, params));
    if (!draft || draft->arch != LLM_ARCH_DFLASH || draft->vocab.n_tokens() != target->vocab.n_tokens() ||
            draft->hparams.n_embd != target->hparams.n_embd) { return nullptr; }
    if (target->memory_contract->inkling_dspark_target) {
        constexpr int32_t visits[] = {5, 23, 41, 48};
        if (!llama_atomic_is_inkling_dspark(draft.get()) || draft->dev_output() != target->dev_output() ||
                draft->n_gpu_layers() <= draft->hparams.n_layer_all ||
                (!draft->dspark_markov_w2 && draft->hparams.n_vocab_padded != target->hparams.n_vocab_padded) ||
                !std::equal(draft->target_layer_ids.begin(), draft->target_layer_ids.end(), std::begin(visits), std::end(visits))) {
            return nullptr;
        }
        // The draft indexes its Markov weights with the target's token ids.
        for (llama_token id = 0; id < int32_t(target->vocab.n_tokens()); ++id) {
            if (draft->vocab.get_token_data(id).text != target->vocab.get_token_data(id).text) {
                return nullptr;
            }
        }
    } else if (!draft->hparams.dflash_selector_top_k ||
            draft->target_layer_ids.empty() || std::any_of(draft->target_layer_ids.begin(), draft->target_layer_ids.end(),
                [&](int32_t i) { return i < 0 || uint32_t(i) >= target->hparams.n_layer(); })) {
        return nullptr;
    }
    draft->memory_contract = std::make_shared<llama_memory_contract>(ledger);
    return draft.release();
}

bool llama_atomic_prefix_supported(const llama_context * ctx) {
#if defined(__linux__)
    if (ctx) {
        const auto & model = ctx->get_model();
        return model.arch == LLM_ARCH_QWEN4EXP && model.hparams.n_layer() == 48 &&
                model.memory_contract && model.memory_contract->discrete.device && model.memory_contract->stream_experts;
    }
#else
    (void) ctx;
#endif
    return false;
}

bool llama_atomic_prefix_ready(const llama_context * ctx, size_t count) {
    if (!llama_atomic_prefix_supported(ctx) || !count || count >= llama_n_ctx(ctx)) {
        return false;
    }
    const auto * memory = dynamic_cast<const llama_memory_hybrid_idx *>(ctx->get_memory());
    // The hybrid minimum describes the oldest recoverable recurrent state,
    // not the beginning of the retained attention prefix.
    return memory && memory->get_mem_attn()->seq_pos_min(0) == 0 &&
            memory->get_mem_idx() && memory->get_mem_idx()->seq_pos_min(0) == 0 &&
            memory->seq_pos_max(0) == llama_pos(count - 1);
}

bool llama_atomic_prefix_restore(llama_context * ctx, const char * path, const llama_token * expected, size_t count) {
    if (!llama_atomic_prefix_supported(ctx) || !expected || !count || count >= llama_n_ctx(ctx)) {
        return false;
    }
    try {
        ctx->synchronize();
        std::vector<llama_token> tokens(count);
        size_t restored = 0;
        if (!ctx->state_seq_load_file(0, path, tokens.data(), count, &restored, expected)) {
            return false;
        }
        return llama_atomic_prefix_ready(ctx, count);
    } catch (const std::exception & e) {
        LLAMA_LOG_WARN("prefix checkpoint restore failed: %s\n", e.what());
        return false;
    }
}

bool llama_context::expert_profile_set(bool enabled) {
    if (!expert_executor) { return false; }
    expert_executor->set_profile(enabled);
    return true;
}
bool llama_context::expert_read_overlap_set(bool enabled) {
    if (!expert_executor) { return false; }
    expert_executor->set_read_overlap(enabled);
    return true;
}
bool llama_expert_read_overlap_set(llama_context * ctx, bool enabled) {
    return ctx && ctx->expert_read_overlap_set(enabled);
}
bool llama_context::expert_single_map_set(bool enabled) {
    return expert_executor && expert_executor->set_single_expert_map(enabled);
}
bool llama_expert_single_map_set(llama_context * ctx, bool enabled) {
    return ctx && ctx->expert_single_map_set(enabled);
}
bool llama_context::expert_route_index_set(bool enabled) {
    return expert_executor && expert_executor->set_route_index(enabled);
}
bool llama_expert_route_index_set(llama_context * ctx, bool enabled) {
    return ctx && ctx->expert_route_index_set(enabled);
}
bool llama_context::expert_early_read_set(bool enabled) {
    return expert_executor && expert_executor->set_early_read(enabled);
}
bool llama_expert_early_read_set(llama_context * ctx, bool enabled) {
    return ctx && ctx->expert_early_read_set(enabled);
}
bool llama_context::expert_profile_get(llama_expert_profile_data & result) const {
    return expert_executor && expert_executor->get_profile(result);
}
bool llama_expert_profile_set(llama_context * ctx, bool enabled) {
    return ctx && ctx->expert_profile_set(enabled);
}
bool llama_expert_profile_get(const llama_context * ctx, llama_expert_profile_data * result) {
    return ctx && result && ctx->expert_profile_get(*result);
}

bool llama_context::expert_cache_policy_set(const char * policy) {
    if (!expert_executor || !policy) { return false; }
    try { return expert_executor->set_policy(policy); }
    catch (const std::exception & e) {
        LLAMA_LOG_ERROR("expert cache policy: %s\n", e.what());
        return false;
    }
}

bool llama_expert_cache_policy_set(llama_context * ctx, const char * policy) {
    return ctx && ctx->expert_cache_policy_set(policy);
}

int32_t llama_expert_prefill_decode(llama_context * ctx, llama_batch_ext * batch, size_t uncached_prompt_tokens) {
    return ctx->decode(*batch, uncached_prompt_tokens);
}

bool llama_context::expert_trace_set(llama_expert_trace_callback callback, void * user) {
    if (!expert_executor) { return false; }
    try { expert_executor->set_trace(callback, user); return true; }
    catch (const std::exception & e) { LLAMA_LOG_ERROR("expert trace: %s\n", e.what()); return false; }
}
bool llama_context::expert_preview_set(int mode) {
    if (!expert_executor || !model.memory_contract || model.arch != LLM_ARCH_QWEN35MOE ||
        model.n_gpu_layers() == 0 || model.devices.empty() ||
        std::strcmp(ggml_backend_dev_name(model.devices.front().dev), "MTL0") != 0) { return false; }
    ggml_backend_sched_synchronize(sched.get());
    std::vector<llama_expert_preview_layer> layers;
    if (mode != 0) {
        for (size_t i = 0; i + 1 < model.layers.size(); ++i) {
            const auto & current = model.layers[i];
            const auto & next = model.layers[i + 1];
            if (next.ffn_exp_probs_b) { return false; }
            layers.push_back({current.ffn_gate_exps, current.ffn_up_exps, next.ffn_gate_inp,
                {next.ffn_gate_exps, next.ffn_up_exps, next.ffn_down_exps}});
        }
    }
    return expert_executor->set_preview(mode, layers);
}
bool llama_expert_preview_set(llama_context * ctx, int mode) {
    try { return ctx && ctx->expert_preview_set(mode); }
    catch (const std::exception & e) { LLAMA_LOG_ERROR("expert preview: %s\n", e.what()); return false; }
}

bool llama_context::expert_route_observer_set(llama_expert_route_callback callback, void * user) {
    if (!expert_executor) { return false; }
    expert_executor->set_route_observer(callback, user);
    return true;
}
bool llama_expert_route_observer_set(llama_context * ctx, llama_expert_route_callback callback, void * user) {
    return ctx && ctx->expert_route_observer_set(callback, user);
}

bool llama_expert_trace_set(llama_context * ctx, llama_expert_trace_callback callback, void * user) {
    return ctx && ctx->expert_trace_set(callback, user);
}

bool llama_context::expert_streaming_stats(llama_expert_streaming_data & result) const {
    if (!expert_executor) { return false; }
    const auto s = expert_executor->stats();
    result = {s.memory.requested, s.memory.applied, s.memory.used(), s.memory.high_water, s.minimum_bytes,
            s.store.hits, s.store.misses, s.store.evictions, s.store.logical_read_bytes, s.projections, s.matmuls,
            s.store.buffer_allocations, s.store.buffer_reuses, s.submissions, s.store.packed_reads, s.store.packed_reuses,
            s.store.parallel_batches, s.store.parallel_reads, s.combined_moe,
            s.moe_pipeline_groups, s.moe_resident_starts,
            s.store.reclaim_calls, s.store.reclaim_evictions, s.store.reclaim_bytes};
    return true;
}

bool llama_get_expert_streaming_stats(const llama_context * ctx, llama_expert_streaming_data * result) {
    return ctx && result && ctx->expert_streaming_stats(*result);
}

bool llama_context::configure_memory_policy(const llama_memory_policy_config & config, bool system_source) {
    if (!model.memory_contract || model.memory_contract->discrete.device || !expert_executor) { return false; }
    std::lock_guard<std::mutex> lock(budget_control->budget_mutex);
    if (budget_control->budget_status == LLAMA_BUDGET_PENDING) { return false; }
    auto candidate = budget_control->budget_policy;
    if (!candidate.configure(config, llama_memory_policy_now_ms(), expert_executor->stats().minimum_bytes)) { return false; }
    std::unique_ptr<llama_memory_pressure_monitor> monitor;
    if (system_source && config.mode == llama_memory_mode::automatic) {
        monitor.reset(new llama_memory_pressure_monitor);
        if (!monitor->available()) { return false; }
    }
    budget_control->pressure_monitor = std::move(monitor);
    budget_control->budget_policy = candidate;
    if (budget_control->prefill_window.enabled && config.mode == llama_memory_mode::automatic) {
        expert_executor->clear_window_decode_advice();
    }
    return true;
}

bool llama_context::submit_memory_pressure(const llama_memory_pressure_sample & sample) {
    if (!model.memory_contract) { return false; }
    std::lock_guard<std::mutex> lock(budget_control->budget_mutex);
    if (budget_control->pressure_monitor || budget_control->budget_policy.status().config.mode != llama_memory_mode::automatic) { return false; }
    return budget_control->budget_policy.observe(sample, llama_memory_policy_now_ms());
}

bool llama_context::memory_policy_stats(llama_memory_policy_data & result) const {
    if (!model.memory_contract || model.memory_contract->discrete.device || !expert_executor) { return false; }
    std::lock_guard<std::mutex> lock(budget_control->budget_mutex);
    result = budget_control->budget_policy.status();
    result.minimum = std::max(result.config.minimum, expert_executor->stats().minimum_bytes);
    if (result.config.mode != llama_memory_mode::automatic) { result.target = budget_control->budget_requested; }
    result.system_source = bool(budget_control->pressure_monitor);
    return true;
}

bool llama_memory_policy_configure(llama_context * ctx, const llama_memory_policy_config * config, bool system_source) {
    if (!ctx || !config) { return false; }
    try { return ctx->configure_memory_policy(*config, system_source); }
    catch (const std::exception & e) { LLAMA_LOG_ERROR("memory policy: %s\n", e.what()); return false; }
}
bool llama_memory_pressure_submit(llama_context * ctx, const llama_memory_pressure_sample * sample) {
    return ctx && sample && ctx->submit_memory_pressure(*sample);
}
bool llama_memory_policy_get(const llama_context * ctx, llama_memory_policy_data * result) {
    return ctx && result && ctx->memory_policy_stats(*result);
}

uint64_t llama_context::request_memory_budget(size_t target) {
    if (!model.memory_contract || model.memory_contract->discrete.device) { return 0; }
    std::lock_guard<std::mutex> lock(budget_control->budget_mutex);
    if (budget_control->budget_request_id == UINT64_MAX) { return 0; }
    budget_control->budget_policy.manual();
    budget_control->budget_requested = target;
    budget_control->budget_status = LLAMA_BUDGET_PENDING;
    return ++budget_control->budget_request_id;
}

bool llama_context::resize_discrete_budget(size_t host_target, size_t device_target, size_t scratch_headroom) {
    if (!model.memory_contract || !model.memory_contract->discrete.device || !expert_executor) {
        return false;
    }
    ggml_backend_sched_synchronize(sched.get());
    return expert_executor->resize_discrete(host_target, device_target, scratch_headroom);
}

bool llama_discrete_budget_resize(llama_context * ctx, size_t host_target, size_t device_target, size_t scratch_headroom) {
    return ctx && ctx->resize_discrete_budget(host_target, device_target, scratch_headroom);
}

bool llama_context::set_discrete_prefill_probe(bool (*mid_read)(void *) noexcept,
        bool (*boundary)(llama_context *, void *) noexcept, void * user, bool worker, llama_expert_read_probe submitted_read) {
    if (!model.memory_contract || !model.memory_contract->discrete.device || !expert_executor) {
        return false;
    }
    expert_executor->set_discrete_read_probe(mid_read, user, worker, submitted_read);
    budget_control->discrete_boundary_probe = boundary;
    budget_control->discrete_probe_user = user;
    return true;
}

bool llama_discrete_prefill_probe_set(llama_context * ctx, bool (*mid_read)(void *) noexcept,
        bool (*boundary)(llama_context *, void *) noexcept, void * user, bool worker,
        bool (*submitted_read)(void *, int, size_t, void *, size_t)) {
    return ctx && ctx->set_discrete_prefill_probe(mid_read, boundary, user, worker, submitted_read);
}

bool llama_context::apply_memory_budget() {
    if (!model.memory_contract || model.memory_contract->discrete.device || !expert_executor) { return false; }
    std::lock_guard<std::mutex> lock(budget_control->budget_mutex);
    if (budget_control->budget_status != LLAMA_BUDGET_PENDING && budget_control->budget_policy.status().config.mode == llama_memory_mode::automatic) {
        const auto now = llama_memory_policy_now_ms();
        if (budget_control->pressure_monitor) { budget_control->budget_policy.observe(budget_control->pressure_monitor->poll(now), now); }
        const auto stats = expert_executor->stats();
        const size_t target = budget_control->budget_policy.decide(now, stats.memory.applied, stats.memory.used(), stats.minimum_bytes);
        if (target && budget_control->budget_request_id != UINT64_MAX) {
            budget_control->budget_requested = target;
            ++budget_control->budget_request_id;
            budget_control->budget_status = LLAMA_BUDGET_PENDING;
        }
    }
    if (budget_control->budget_status != LLAMA_BUDGET_PENDING) { return budget_control->budget_status != LLAMA_BUDGET_REJECTED; }
    ggml_backend_sched_synchronize(sched.get());
    if (expert_executor->automatic_acceleration() || expert_executor->has_inkling_core_policy()) {
        reset_expert_graphs();
    }
    const bool applied = expert_executor->resize(budget_control->budget_requested);
    budget_control->budget_completed_id = budget_control->budget_request_id;
    budget_control->budget_status = applied ? LLAMA_BUDGET_APPLIED : LLAMA_BUDGET_REJECTED;
    if (!applied && budget_control->budget_policy.status().config.mode == llama_memory_mode::automatic) { budget_control->budget_policy.failed(); }
    return applied;
}

bool llama_context::memory_budget_stats(llama_budget_data & result) const {
    if (!model.memory_contract || model.memory_contract->discrete.device || !expert_executor) { return false; }
    std::lock_guard<std::mutex> lock(budget_control->budget_mutex);
    const auto stats = expert_executor->stats();
    result = {};
    result.requested = budget_control->budget_requested;
    result.applied = stats.memory.applied;
    result.minimum = stats.minimum_bytes;
    result.used = stats.memory.used();
    result.high_water = stats.memory.high_water;
    result.released = stats.memory.released;
    result.runtime_envelope = llama_memory_contract::runtime_envelope;
    for (size_t i = 0; i < 7; ++i) { result.categories[i] = stats.memory.categories[i].total(); }
    result.request_id = budget_control->budget_request_id;
    result.completed_id = budget_control->budget_completed_id;
    result.status = budget_control->budget_status;
    return true;
}
uint64_t llama_memory_budget_request(llama_context * ctx, size_t target) { return ctx ? ctx->request_memory_budget(target) : 0; }
bool llama_memory_budget_apply(llama_context * ctx) { return ctx && ctx->apply_memory_budget(); }
bool llama_memory_budget_get(const llama_context * ctx, llama_budget_data * result) {
    return ctx && result && ctx->memory_budget_stats(*result);
}


bool llama_discrete_budget_get(const llama_context * ctx, llama_discrete_budget_data * result) {
    if (!ctx || !result) { return false; }
    const auto & contract = ctx->get_model().memory_contract;
    if (!contract || !contract->discrete.device) { return false; }
    *result = {};
    for (size_t i = 0; i < 2; ++i) {
        const auto snapshot = contract->ledger->snapshot(i == 0 ? llama_memory_domain::host : llama_memory_domain::device);
        result->requested[i] = snapshot.requested;
        result->applied[i] = snapshot.applied;
        result->used[i] = snapshot.used();
        result->high_water[i] = snapshot.high_water;
        result->released[i] = snapshot.released;
    }
    return true;
}

namespace {
// Reserve the replacement while the old allocation is still charged. Buffer
// destruction precedes ticket release, including on allocation or IO failure.
struct atomic_state_file_buffer {
    std::shared_ptr<llama_memory_budget> budget;
    llama_memory_budget::ticket charge;
    std::unique_ptr<uint8_t[]> data;
    size_t capacity = 0;

    explicit atomic_state_file_buffer(std::shared_ptr<llama_memory_budget> budget) : budget(std::move(budget)) {}
    uint8_t * get(size_t size) {
        if (size > capacity) {
            auto next_charge = budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, size);
            if (!next_charge) { throw std::runtime_error("Qwen file-state staging exceeds the RAM budget"); }
            auto next = std::make_unique<uint8_t[]>(size);
            next_charge.commit();
            data = std::move(next);
            charge = std::move(next_charge);
            capacity = size;
        }
        return data.get();
    }
};

// File IO mirrors llama-context.cpp at 31953bcf74ba489a8bbb0c936c5fcf24fcf9485c.
// Track upstream tensor ordering, file-error handling and interface changes.
// Only staging ownership/admission differs; the serialized bytes are unchanged.
class atomic_state_file_writer final : public llama_io_write_i {
    llama_file * file;
    atomic_state_file_buffer buffer;
    size_t written = 0;
public:
    atomic_state_file_writer(llama_file * file, std::shared_ptr<llama_memory_budget> budget) : file(file), buffer(std::move(budget)) {}
    void write(const void * src, size_t size) override { file->write_raw(src, size); written += size; }
    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        auto * data = buffer.get(size);
        ggml_backend_tensor_get(tensor, data, offset, size);
        write(data, size);
    }
    size_t n_bytes() override { return written; }
};

class atomic_state_file_reader final : public llama_io_read_i {
    llama_file * file;
    atomic_state_file_buffer buffer;
    size_t read_bytes = 0;
public:
    atomic_state_file_reader(llama_file * file, std::shared_ptr<llama_memory_budget> budget) : file(file), buffer(std::move(budget)) {}
    void read(void * dst, size_t size) override { file->read_raw(dst, size); read_bytes += size; }
    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        auto * data = buffer.get(size);
        read(data, size);
        ggml_backend_tensor_set(tensor, data, offset, size);
    }
    size_t n_bytes() override { return read_bytes; }
};

bool atomic_state_file_budgeted(const llama_model & model) {
    return model.memory_contract && model.memory_contract->discrete.device &&
        model.memory_contract->stream_experts && model.arch == LLM_ARCH_QWEN4EXP;
}
} // namespace

std::unique_ptr<llama_io_read_i> llama_atomic_state_file_reader(const llama_model & model, llama_file * file) {
    if (!atomic_state_file_budgeted(model)) { return nullptr; }
    return std::make_unique<atomic_state_file_reader>(file, model.memory_contract->ledger);
}
std::unique_ptr<llama_io_write_i> llama_atomic_state_file_writer(const llama_model & model, llama_file * file) {
    if (!atomic_state_file_budgeted(model)) { return nullptr; }
    return std::make_unique<atomic_state_file_writer>(file, model.memory_contract->ledger);
}
