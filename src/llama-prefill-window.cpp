// Layer-major prefill, isolated from the ordinary decode loop. The graph/input
// contract follows llama-context.cpp at 31953bcf74ba489a8bbb0c936c5fcf24fcf9485c.
// Track process_ubatch input setup, output extraction and scheduler allocation
// when updating upstream. Only the qualified raw-logits text profile is admitted.
#include "llama-context.h"
#include "llama-context-budget.h"
#include "llama-memory-contract.h"
#include "llama-model.h"
#include "llama-batch.h"
#include "llama-prefill-window-worker.h"
#include "llama-prefill-wide.h"
#include "../ggml/src/ggml-impl.h"

#include <algorithm>
#include <charconv>
#include <string_view>
#include <cstring>
#include <stdexcept>
#ifdef __APPLE__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
// The graph buffers are admitted during window preparation. Until that
// preparation is reversible, retain the qualified floor for this geometry.
constexpr size_t window_budget_floor = size_t(8) * 1024 * 1024 * 1024;
constexpr size_t whole_batch_budget_floor = size_t(18) * 1024 * 1024 * 1024;
constexpr size_t sched_nodes = 8192;
constexpr int standard_window_count = 8;
constexpr int window_ubatch = 8;
constexpr int compact_window_count = 256;
constexpr int shared_window_count = 512;
constexpr std::array<int, 7> admitted_window_counts{shared_window_count, compact_window_count, 128, 64, 32, 16, 8};
constexpr size_t compact_graph_retained = size_t(6) * 1024 * 1024;
constexpr size_t compact_graph_construction = size_t(16) * 1024 * 1024;
// Retained graph pages, compact split/copy metadata and allocator bookkeeping.
// Graph payload buffers are admitted separately.
constexpr size_t compact_window_slot_bytes = compact_graph_retained + size_t(11) * 1024 * 1024 +
        llama_prefill_window_worker::compact_stack_charge;

struct window_graph_mapping {
    void * data = nullptr;
    size_t mapped_bytes = 0;

    window_graph_mapping() = default;
    window_graph_mapping(const window_graph_mapping &) = delete;
    window_graph_mapping & operator=(const window_graph_mapping &) = delete;

    void clear() noexcept {
#ifdef __APPLE__
        if (data) { munmap(data, mapped_bytes); }
#endif
        data = nullptr;
        mapped_bytes = 0;
    }
    ~window_graph_mapping() { clear(); }

    void allocate(size_t bytes) {
        clear();
#ifdef __APPLE__
        const size_t page = size_t(getpagesize());
        const size_t rounded_bytes = (bytes + page - 1) / page * page;
        void * mapping = mmap(nullptr, rounded_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (mapping != MAP_FAILED) {
            data = mapping;
            mapped_bytes = rounded_bytes;
            return;
        }
#else
        (void) bytes;
#endif
        throw std::bad_alloc();
    }

    bool trim(ggml_context * ctx) {
#ifdef __APPLE__
        const size_t page = size_t(getpagesize());
        const size_t retained = (ggml_used_mem(ctx) + page - 1) / page * page;
        if (retained > compact_graph_retained || !ggml_trim_external_context(ctx, retained)) { return false; }
        if (munmap(static_cast<char *>(data) + retained, mapped_bytes - retained)) { return false; }
        mapped_bytes = retained;
        return true;
#else
        (void) ctx;
        return false;
#endif
    }
};

int requested_window_count(const char * mode) {
    if (!mode) { return 0; }
    if (!std::strcmp(mode, "auto")) { return 32; }
    int tokens = 0;
    const std::string_view value(mode);
    const auto result = std::from_chars(value.data(), value.data() + value.size(), tokens);
    if (result.ec != std::errc() || result.ptr != value.data() + value.size()) { return 0; }
    for (int count : admitted_window_counts) {
        if (tokens == count * window_ubatch) { return count; }
    }
    return 0;
}

struct window_route_handoff {
    llama_prefill_window_worker * worker = nullptr;
    llama_expert_executor * executor = nullptr;
    bool park = false;
    bool cross = false;
    std::FILE * trace = nullptr;

    static bool observe(ggml_tensor * node, bool ask, void * data) {
        auto & handoff = *static_cast<window_route_handoff *>(data);
        const char * weight = handoff.executor->window_projection_name(node);
        if (!weight) { return false; }
        const std::string_view name(weight);
        if (name.substr(0, 4) != "blk.") { return false; }
        int layer = -1;
        const auto parsed = std::from_chars(name.data() + 4, name.data() + name.size(), layer);
        const std::string_view suffix(parsed.ptr, name.data() + name.size() - parsed.ptr);
        if (parsed.ec != std::errc() || layer < 0 || layer >= handoff.worker->layer_count ||
                (suffix != ".ffn_gate_exps.weight" && suffix != ".ffn_up_exps.weight" &&
                 suffix != ".ffn_down_exps.weight")) { return false; }
        if (ask) { return handoff.cross || handoff.worker->routes_reached != layer; }
        if (handoff.cross) {
            handoff.worker->observe_routes(layer, node->src[1], false, handoff.trace);
            if (handoff.worker->observe_cross(layer, node)) {
                handoff.executor->use_cross_window_output(node);
            }
        } else {
            handoff.worker->observe_routes(layer, node->src[1], handoff.park, handoff.trace);
        }
        return true;
    }
};


}

bool llama_context::prefill_window_supported() const {
#ifndef __APPLE__
    return false;
#else
    const bool flash = model.arch == LLM_ARCH_QWEN4EXP && model.hparams.n_layer() == 48;
    const bool portable = model.arch == LLM_ARCH_QWEN35MOE || model.arch == LLM_ARCH_LFM2MOE;
    return budget_control && expert_executor && model.memory_contract && !model.memory_contract->discrete.device &&
            (flash || portable) && !backend_ptrs.empty() &&
            std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_ptrs.front()))), "MTL") == 0;
#endif
}

bool llama_context::prefill_wide_qualified() const {
    if (!prefill_window_supported() || !expert_executor->can_batch_window()) { return false; }
    const auto & hp = model.hparams;
    struct profile {
        llm_arch arch;
        uint32_t layers, width, expert_width, experts, active;
        std::initializer_list<ggml_type> gate_up, down;
    };
    // Scope the automatic path to qualified expert geometries and quantizations.
    // The explicit diagnostic override still uses the runtime admission checks.
    static const profile qualified[] = {
        {LLM_ARCH_QWEN4EXP, 48, 2560,  640, 512, 10,
            {GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_S},
            {GGML_TYPE_Q2_0, GGML_TYPE_IQ4_NL}},
        {LLM_ARCH_QWEN35MOE, 40, 2048,  512, 256, 8, {GGML_TYPE_Q4_K}, {GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}},
        {LLM_ARCH_QWEN35MOE, 40, 2048,  512, 256, 8, {GGML_TYPE_IQ4_XS}, {GGML_TYPE_Q4_K}},
        {LLM_ARCH_QWEN35MOE, 48, 3072, 1024, 256, 8, {GGML_TYPE_IQ4_XS}, {GGML_TYPE_Q4_K}},
        {LLM_ARCH_LFM2MOE, 40, 2048, 1536, 64, 4, {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K}, {GGML_TYPE_Q5_K, GGML_TYPE_Q6_K}},
    };
    auto matches = [](const ggml_tensor * tensor, uint32_t cols, uint32_t rows, uint32_t experts,
            std::initializer_list<ggml_type> types) {
        return tensor && tensor->ne[0] == cols && tensor->ne[1] == rows && tensor->ne[2] == experts && tensor->ne[3] == 1 &&
            std::find(types.begin(), types.end(), tensor->type) != types.end();
    };
    return std::any_of(std::begin(qualified), std::end(qualified), [&](const profile & p) {
        if (p.arch != model.arch || p.layers != hp.n_layer() || p.width != hp.n_embd_inp() || p.experts != hp.n_expert) {
            return false;
        }
        for (uint32_t i = 0; i < p.layers; ++i) {
            const auto & layer = model.layers[i];
            if (p.arch == LLM_ARCH_LFM2MOE && i < hp.n_layer_dense_lead &&
                    !layer.ffn_gate_exps && !layer.ffn_up_exps && !layer.ffn_down_exps && !layer.ffn_gate_up_exps) {
                continue;
            }
            if (hp.n_expert_used(i) != p.active || layer.ffn_gate_up_exps ||
                    !matches(layer.ffn_gate_exps, p.width, p.expert_width, p.experts, p.gate_up) ||
                    !matches(layer.ffn_up_exps, p.width, p.expert_width, p.experts, p.gate_up) ||
                    !matches(layer.ffn_down_exps, p.expert_width, p.width, p.experts, p.down)) {
                return false;
            }
        }
        return true;
    });
}

bool llama_context::prefill_window_set(bool enabled) {
    if (!prefill_window_supported() || budget_control->prefill_window.failed) { return false; }
    if (!enabled) { expert_executor->clear_window_decode_advice(); }
    budget_control->prefill_window.enabled = enabled;
    budget_control->prefill_window_legacy = enabled;
    return true;
}
bool llama_context::prefill_window_get(llama_prefill_window_data & result) const {
    if (!budget_control) { return false; }
    result = budget_control->prefill_window;
    return true;
}
bool llama_prefill_window_set(llama_context * ctx, bool enabled) {
    return ctx && ctx->prefill_window_set(enabled);
}
bool llama_prefill_window_get(const llama_context * ctx, llama_prefill_window_data * result) {
    return ctx && result && ctx->prefill_window_get(*result);
}

int llama_context::decode_prefill_window(llama_memory_context_ptr & mctx, int32_t n_outputs_all,
        size_t prompt_tokens, size_t batch_tokens, int64_t & n_outputs_prev, int64_t & n_tokens_prev,
        bool * retry_smaller_wide, bool try_shared_wide, bool partial_wide) {
    auto & counters = budget_control->prefill_window;
    auto fallback = [&] { ++counters.fallbacks; return 0; };
    const bool flash = model.arch == LLM_ARCH_QWEN4EXP;
    const int layer_count = int(model.hparams.n_layer());
    const char * layer_output = flash ? "l_last" : "l_out";
    const bool whole_batch = flash && cparams.n_ctx == 1024;
    const uint32_t rows = mctx->get_ubatch().n_tokens;
    const bool wide_arch = flash || model.arch == LLM_ARCH_QWEN35MOE || model.arch == LLM_ARCH_LFM2MOE;
    const bool smaller_wide = !partial_wide && wide_arch && rows == prefill_smaller_wide_rows;
    const bool native_wide = partial_wide || (wide_arch && (rows == prefill_wide_rows || smaller_wide));
    const bool wide_dflash = native_wide && model.arch == LLM_ARCH_QWEN35MOE &&
        model.memory_contract->dflash2_target && cparams.embeddings_layer_inp_dev && layer_inp_dev;
    // Reuse layer-local Metal temporaries across the same wide workers.
    // Live outputs at layer boundaries and draft feature rows stay private.
    const bool shared_dflash = wide_dflash && !smaller_wide && try_shared_wide;
    // Three smaller workers reduce the number of complete weight traversals
    // compared with one 1024-row worker. Leave admission room for staging.
    const bool shared_wide = flash && smaller_wide && try_shared_wide;
    const int smaller_wide_workers = shared_wide ? 4 : 3;
    // Full-cache reservation for a 512-row Flash-Next graph measured 129.59 MiB.
    // If this layout cannot be admitted, retry the private three-worker layout.
    constexpr size_t wide_shared_scratch_bytes = 130U << 20;
    // The 1024-row Qwen35 graph also holds recurrent rollback temporaries.
    constexpr size_t dflash_shared_scratch_bytes = 320U << 20;
    const bool batch_capable = !whole_batch && expert_executor->can_batch_window() &&
        size_t(model.hparams.n_expert) <= llama_prefill_window_worker::max_experts;
    const char * batch_mode = batch_capable && !native_wide ? std::getenv("LLAMA_EXPERIMENTAL_PREFILL_EXPERT_BATCH") : nullptr;
    if (batch_mode && !std::strcmp(batch_mode, "auto") && expert_executor->window_weights_fit_without_eviction()) {
        LLAMA_LOG_DEBUG("%s: ordinary prefill can hold all expert weights within the budget\n", __func__);
        return fallback();
    }
    const int requested_count = native_wide && batch_capable ? shared_window_count : requested_window_count(batch_mode);
    const bool shared_scratch = requested_count == shared_window_count && !native_wide;
    const size_t graph_nodes = graph_max_nodes();
    const size_t graph_arena = graph_nodes * ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_nodes, false);
    const char * batch_submit_env = shared_scratch ? std::getenv("LLAMA_EXPERIMENTAL_PREFILL_BATCH_SUBMIT") : nullptr;
    const int main_rows = std::max(window_ubatch, int(mctx->get_ubatch().n_tokens));
    const bool grouped = main_rows != window_ubatch;
    const bool wide_submit_requested = flash && batch_submit_env && !std::strcmp(batch_submit_env, "32");
    ggml_window_batch main_batch{};
    const bool batch_submit = (native_wide || wide_submit_requested || (batch_submit_env && !std::strcmp(batch_submit_env, "1"))) &&
        backend_ptrs.size() == 2 && ggml_backend_window_batch_init(&main_batch, backend_ptrs.front());
    if (grouped && !native_wide && (main_rows != 16 || !batch_submit || model.hparams.n_expert_used_max() > 16)) { return fallback(); }
    // Other architectures use the compact layout only with the layer-major
    // Metal coordinator; their existing threaded windows keep their layout.
    const bool compact = (flash || batch_submit) && requested_count >= compact_window_count &&
        graph_arena <= compact_graph_construction;
    if (requested_count >= compact_window_count && !compact) { return fallback(); }
    const int max_batch = std::max(256, requested_count * window_ubatch);
    const bool eligible_batch = whole_batch ? batch_tokens > standard_window_count * window_ubatch :
        cparams.n_ctx == 4096 && (partial_wide || prompt_tokens >= 320);
    if (!expert_executor->can_layer_window() || !eligible_batch || n_outputs_all > 1 || cparams.n_batch > uint32_t(max_batch) ||
            cparams.n_ubatch != window_ubatch || cparams.n_seq_max != 1 || cparams.pipeline_parallel ||
            cparams.embeddings || cparams.embeddings_nextn || cparams.embeddings_nextn_dev ||
            (!wide_dflash && cparams.embeddings_layer_inp_dev) || !cparams.causal_attn || cparams.training ||
            cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || cparams.cb_eval || abort_callback ||
            cparams.ctx_other || cparams.ctx_state_src || backend_ptrs.size() > 3 || graph_nodes > 40000 ||
            (loras && !loras->empty()) || (!wide_dflash && std::any_of(cparams.embeddings_layer_inp.begin(),
                cparams.embeddings_layer_inp.end(), [](bool v) { return v; }))) { return fallback(); }
    if ((!flash && !requested_count) || (native_wide && (!batch_submit || !compact))) { return fallback(); }
    for (int i = 0; i < layer_count; ++i) { if (cvec && cvec->tensor_for(i)) { return fallback(); } }
    const size_t budget_floor = whole_batch ? whole_batch_budget_floor : window_budget_floor;
    const int original_count = whole_batch ? int((batch_tokens + window_ubatch - 1) / window_ubatch) : standard_window_count;
    const char * trace_path = batch_capable && !native_wide ?
        std::getenv("LLAMA_PREFILL_EXPERT_ROUTE_TRACE") : nullptr;
    bool batch_experts = requested_count > 0;
    int routed_layers = 0;
    std::array<std::array<const ggml_tensor *, 3>, LLAMA_MAX_LAYERS> layer_weights {};
    if (batch_experts || trace_path) {
        for (int layer = 0; layer < layer_count; ++layer) {
            const auto & tensors = model.layers[layer];
            layer_weights[layer] = {tensors.ffn_gate_exps, tensors.ffn_up_exps, tensors.ffn_down_exps};
            if (model.arch == LLM_ARCH_LFM2MOE && layer < int(model.hparams.n_layer_dense_lead) &&
                    !tensors.ffn_up_exps && !tensors.ffn_down_exps && !tensors.ffn_gate_exps && !tensors.ffn_gate_up_exps) {
                continue;
            }
            if (tensors.ffn_gate_up_exps || !expert_executor->can_batch_window_layer(layer_weights[layer])) {
                batch_experts = false;
                trace_path = nullptr;
                break;
            }
            ++routed_layers;
        }
    }
    if (!batch_experts && (native_wide || !flash || cparams.n_batch > 256)) { return fallback(); }
    const char * coalesce = batch_experts ? std::getenv("LLAMA_EXPERIMENTAL_PREFILL_COALESCE") : nullptr;
    const bool adjacent_ranges = coalesce && !std::strcmp(coalesce, "1");
    int window_count = original_count;
    size_t window_budget;
    {
        std::lock_guard<std::mutex> lock(budget_control->budget_mutex);
        if (budget_control->budget_status == LLAMA_BUDGET_PENDING ||
                budget_control->budget_policy.status().config.mode != llama_memory_mode::fixed ||
                budget_control->budget_requested < budget_floor) { return fallback(); }
        window_budget = budget_control->budget_requested;
    }
    const auto memory_snapshot = model.memory_contract->ledger->snapshot(llama_memory_domain::host);
    if (memory_snapshot.requested != window_budget || memory_snapshot.applied != window_budget) { return fallback(); }
    synchronize();
    expert_executor->trim();
    // Each slot covers the graph arena, compact scheduler metadata and an <=8 MiB
    // worker stack. Backend buffers are charged separately by allocation_scope.
    auto & ledger = *model.memory_contract->ledger;
    auto construction = ledger.reserve(llama_memory_domain::host,
            llama_memory_category::scratch, size_t(224) * 1024 * 1024);
    if (!construction) { return fallback(); }
    llama_memory_budget::ticket metadata;
    llama_expert_executor::window_leases layer_leases;
    const int batch_workers = int((batch_tokens + main_rows - 1) / main_rows);
    if (batch_experts) {
        const size_t routes_per_batch = native_wide ? 0 : size_t(model.hparams.n_expert_used_max()) * main_rows;
        for (int count : admitted_window_counts) {
            if (count > requested_count || (!compact && count > 8 && batch_tokens <= size_t(count / 2 * window_ubatch))) { continue; }
            int slots = compact ? std::min(count, batch_workers) : count;
            if (smaller_wide) { slots = std::min(slots, smaller_wide_workers); }
            const size_t slot_bytes = compact ? compact_window_slot_bytes : size_t(64) * 1024 * 1024;
            const size_t lease_bytes = 3 * llama_prefill_window_worker::max_experts * sizeof(llama_expert_store::lease);
            auto reserve_window = [&](int size) {
                return ledger.reserve(llama_memory_domain::host, llama_memory_category::scratch,
                        size_t(size) * slot_bytes + lease_bytes);
            };
            auto ticket = reserve_window(slots);
            if (!ticket) { continue; }
            const size_t capacity = expert_executor->window_expert_capacity();
            if (capacity < routes_per_batch) { continue; }
            if (shared_scratch && batch_submit && capacity < size_t(model.hparams.n_expert)) {
                // Keep the number of passes and give spare graph slots back to
                // experts. The retained graphs and the layer share this budget.
                const int passes = (batch_workers + slots - 1) / slots;
                const int balanced = (batch_workers + passes - 1) / passes;
                if (balanced < slots) {
                    ticket.reset();
                    ticket = reserve_window(balanced);
                    if (!ticket) { continue; }
                    slots = balanced;
                }
            }
            metadata = std::move(ticket);
            window_count = slots;
            break;
        }
        batch_experts = bool(metadata);
        if (batch_experts && !native_wide) {
            try {
                layer_leases.reserve(3 * llama_prefill_window_worker::max_experts);
            } catch (const std::bad_alloc &) {
                batch_experts = false;
                metadata.reset();
                window_count = original_count;
            }
        }
    }
    if (!batch_experts && (native_wide || !flash || cparams.n_batch > 256)) { return fallback(); }
    if (!metadata) { metadata = ledger.reserve(llama_memory_domain::host,
            llama_memory_category::scratch, size_t(window_count) * 64 * 1024 * 1024); }
    if (!metadata) { return fallback(); }
    constexpr size_t gib = size_t(1024) * 1024 * 1024;
    constexpr int profile_layers = 40;
    constexpr int lfm_dense_layers = 2;
    const bool residency_profile_shape = native_wide && budget_control->prefill_window_wide_default &&
        layer_count == profile_layers &&
        n_outputs_all == 1 && prompt_tokens == batch_tokens && mctx->get_ubatch().pos[0] == 0;
    const bool lfm_residency_profile = residency_profile_shape && model.arch == LLM_ARCH_LFM2MOE &&
        model.hparams.n_layer_dense_lead == lfm_dense_layers;
    const uint32_t qwen_layers = model.memory_contract->qwen35_profile_layers;
    const bool qwen_residency_qualified = residency_profile_shape && model.arch == LLM_ARCH_QWEN35MOE &&
        window_budget >= 8 * gib && window_budget <= 24 * gib && qwen_layers;
    const char * qwen_resident_opt = qwen_residency_qualified ? std::getenv("LLAMA_QWEN35_PREFILL_RESIDENCY") : nullptr;
    const bool qwen_force = qwen_resident_opt && !std::strcmp(qwen_resident_opt, "1");
    const bool qwen_auto = qwen_residency_qualified && !qwen_resident_opt;
    const bool qwen_serial = !model.memory_contract->dflash2_target;
    const bool qwen_default_direct = qwen_auto && qwen_serial && window_budget == 24 * gib && qwen_layers == profile_layers;
    // Only enable defaults for the measured fixed-budget profiles. Verify the
    // last prefix bank too, since an earlier live resize can have evicted it.
    const bool qwen_default_retain = qwen_auto &&
        ((qwen_serial && window_budget == 22 * gib && qwen_layers == 31) ||
         (wide_dflash && ((window_budget == 20 * gib && qwen_layers == 26) ||
                          (window_budget == 22 * gib && qwen_layers == 31) ||
                          (window_budget == 24 * gib && qwen_layers == 36)))) &&
        expert_executor->resident_layer(qwen_layers - 1);
    const char * retain_opt = lfm_residency_profile && window_budget == 12 * gib ?
        std::getenv("LLAMA_LFM_RETAIN_PREFILL_TAIL") : nullptr;
    const bool retain_lfm = lfm_residency_profile &&
        ((window_budget >= 16 * gib && window_budget <= 18 * gib) || (retain_opt && !std::strcmp(retain_opt, "1")));
    std::array<llama_prefill_wide_weights::layer_weights, profile_layers> resident_bindings{};
    bool resident_prefill = (lfm_residency_profile && window_budget >= 20 * gib && window_budget <= 24 * gib) ||
        qwen_default_direct || (qwen_force && qwen_serial);
    const char * hybrid_mode = qwen_auto ? std::getenv("LLAMA_QWEN35_HYBRID_PREFILL") : nullptr;
    const bool hybrid_prefill = (lfm_residency_profile && partial_wide) ||
            (qwen_auto && (!hybrid_mode || !std::strcmp(hybrid_mode, "1")));
    int direct_layers = 0;
    if (resident_prefill || hybrid_prefill) {
        for (int layer = 0; layer < layer_count; ++layer) {
            if (!layer_weights[layer][0]) { continue; } // leading dense layer
            const auto * weights = expert_executor->resident_layer(layer);
            if (!weights) {
                if (!hybrid_prefill) { break; }
                continue;
            }
            resident_bindings[layer] = *weights;
            ++direct_layers;
        }
        resident_prefill = direct_layers == routed_layers;
    }
    const bool retain_qwen = qwen_default_retain || (qwen_force && !resident_prefill);
    const bool retain_tail = (retain_lfm || retain_qwen) && expert_executor->can_retain_prefill_tail(routed_layers);
    std::unique_ptr<llama_prefill_wide_weights> wide_weights;
    if (native_wide) {
        if (!model.expert_source) { return fallback(); }
        for (const auto & weights : layer_weights) {
            for (const auto * weight : weights) {
                if (weight && model.hdmd.rot.count(weight)) { return fallback(); }
            }
        }
        if (!resident_prefill) {
            try {
                wide_weights = std::make_unique<llama_prefill_wide_weights>(*model.expert_source);
                if (!wide_weights->prepare(backend_ptrs.front(), backend_buft.front(), layer_count, layer_weights.data(),
                        hybrid_prefill && direct_layers ? resident_bindings.data() : nullptr)) {
                    return fallback();
                }
            } catch (const std::exception & e) {
                LLAMA_LOG_DEBUG("%s: wide weights unavailable: %s\n", __func__, e.what());
                return fallback();
            }
        }
    }
    using trace_file = std::unique_ptr<std::FILE, decltype(&std::fclose)>;
    trace_file trace(trace_path ? std::fopen(trace_path, "a") : nullptr, std::fclose);
    if (trace_path && !trace) { LLAMA_LOG_WARN("cannot open prefill expert route trace\n"); }

    std::vector<window_graph_mapping> graph_mappings(compact ? window_count : 0);
    std::vector<std::unique_ptr<llm_graph_result>> results(window_count);
    const bool share_main = shared_scratch || shared_wide || shared_dflash;
    ggml_backend_buffer_ptr shared_buffer;
    if (share_main) {
        const size_t bytes = shared_wide ? wide_shared_scratch_bytes :
            shared_dflash ? dflash_shared_scratch_bytes : size_t(4 * main_rows) << 20;
        shared_buffer.reset(ggml_backend_buft_alloc_buffer(backend_buft.front(), bytes));
        if (!shared_buffer) { return fallback(); }
        ggml_backend_buffer_set_usage(shared_buffer.get(), GGML_BACKEND_BUFFER_USAGE_COMPUTE);
    }
    std::vector<ggml_backend_sched_ptr> schedulers(window_count);
    std::vector<llama_prefill_window_worker> workers(window_count);
    std::vector<int32_t> outputs(window_count), token_counts(window_count);
    // Constructors and admission precede all recurrent/KV state mutations.
    const auto make_scheduler = compact ? ggml_backend_window_new_compact : ggml_backend_window_new;
    for (auto & scheduler : schedulers) {
        scheduler.reset(make_scheduler(backend_ptrs.data(), backend_buft.data(),
                backend_ptrs.size(), sched_nodes, sched_params_cur.op_offload));
        if (!scheduler) { return fallback(); }
        if (share_main && !ggml_backend_window_share_scratch(scheduler.get(), shared_buffer.get())) { return fallback(); }
        if (batch_submit) { ggml_backend_window_batch_plan(scheduler.get()); }
        if (!flash || native_wide) { ggml_backend_window_set_fence(scheduler.get(), flash ? "l_last-" : "l_out-"); }
    }
    metadata.commit();
    construction.reset();
    llama_memory_budget::ticket graph_construction;
    if (compact) {
        graph_construction = ledger.reserve(llama_memory_domain::host, llama_memory_category::scratch,
                compact_graph_construction);
        if (!graph_construction) { return fallback(); }
    }
    const char * vector = batch_experts && !native_wide ? std::getenv("LLAMA_EXPERIMENTAL_PREFILL_EXPERT_VECTOR") : nullptr;
    if (!expert_executor->set_layer_window(window_budget, vector && !std::strcmp(vector, "1"))) { return fallback(); }
    const char * pipeline = batch_experts && !native_wide ? std::getenv("LLAMA_EXPERIMENTAL_PREFILL_EXPERT_PIPELINE") : nullptr;
    const int pipeline_depth = pipeline && expert_executor->can_pipeline_window() ?
        (!std::strcmp(pipeline, "4") ? 4 : !std::strcmp(pipeline, "1") ? 1 : 0) : 0;
    const char * cross_opt = batch_experts && !native_wide ? std::getenv("LLAMA_EXPERIMENTAL_PREFILL_CROSS_BATCH") : nullptr;
    const char * fused_opt = compact && batch_experts && !native_wide ? std::getenv("LLAMA_EXPERIMENTAL_PREFILL_FUSED_MOE") : nullptr;
    if (fused_opt && !std::strcmp(fused_opt, "1")) { expert_executor->enable_fused_window_moe(); }
    const bool fused = expert_executor->fused_window_moe_enabled();
    if (shared_scratch && !fused) {
        expert_executor->set_layer_window(0);
        return fallback();
    }
    const char * fused_group = fused ? std::getenv("LLAMA_EXPERIMENTAL_PREFILL_FUSED_EXPERTS") : nullptr;
    const bool wide_fused = fused_group && !std::strcmp(fused_group, "64");
    const char * matrix_opt = batch_submit && wide_fused ? std::getenv("LLAMA_EXPERIMENTAL_PREFILL_MATRIX_FFN") : nullptr;
    const bool matrix_window = matrix_opt && !std::strcmp(matrix_opt, "1") && expert_executor->prepare_matrix_window_moe();
    if (grouped && !matrix_window && !native_wide) {
        expert_executor->set_layer_window(0);
        return fallback();
    }
    constexpr int chunk = int(llama_expert_executor::cross_window_workers);
    const bool wide_submit = wide_submit_requested && batch_submit && compact && (grouped || !matrix_window);
    const int submit_views = wide_submit ? 32 : chunk;
    size_t matrix_groups = 0;
    int reused_reservations = 0;
    const char * route_reads = wide_fused && pipeline_depth ? std::getenv("LLAMA_EXPERIMENTAL_PREFILL_EARLY_DEMAND") : nullptr;
    const bool early_demand = route_reads && !std::strcmp(route_reads, "1");
    const bool cross = fused || (cross_opt && !std::strcmp(cross_opt, "1") &&
        expert_executor->prepare_cross_window(llama_expert_executor::cross_window_workers * window_ubatch,
                model.hparams.n_expert_used_max()));
    const int cross_phases = fused ? 1 : 3;
    std::vector<window_route_handoff> handoffs;
    std::vector<ggml_window_cursor> cursors(batch_submit ? window_count : 0);
    const int64_t n_vocab = model.vocab.n_tokens();
    int64_t counted_tokens = n_tokens_prev;
    auto original_sched = std::move(sched);
    auto flush_main = [&] {
        if (main_batch.flush(main_batch.metal, &main_batch.submissions) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("prefill Metal batch failed");
        }
    };
    auto step_main = [&](int i, ggml_window_step step) {
        if (ggml_backend_window_batch_step(schedulers[i].get(), &cursors[i], &main_batch, step,
                &workers[i].cross_node) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("prefill Metal batch missed its graph boundary");
        }
    };
    auto cleanup = [&](bool cancel = false) {
        if (batch_submit) { main_batch.flush(main_batch.metal, &main_batch.submissions); }
        if (cancel && (batch_experts || trace)) {
            for (auto & worker : workers) { worker.cancel(); }
        }
        for (auto & worker : workers) { worker.finish(); }
        layer_leases.clear();
        llama_expert_executor::window_leases().swap(layer_leases);
        // Restore before destroying graph metadata; no paused compute survives.
        if (original_sched) { sched = std::move(original_sched); }
        schedulers.clear();
        results.clear();
        graph_mappings.clear();
        workers.clear();
        shared_buffer.reset();
        wide_weights.reset();
        metadata.reset();
        expert_executor->set_layer_window(0);
    };
    auto reset_window_graph = [&](int i) {
        if (compact) {
            results[i].reset();
            graph_mappings[i].allocate(graph_arena);
            results[i] = std::make_unique<llm_graph_result>(graph_nodes, graph_mappings[i].data);
        } else if (!results[i]) {
            results[i] = std::make_unique<llm_graph_result>(graph_nodes);
        } else {
            results[i]->reset();
        }
    };
    auto allocate_window_graph = [&](llm_graph_result & result, const llama_ubatch & u,
            const llama_memory_context_i * graph_memory) {
        auto params = graph_params(&result, u, graph_memory, LLM_GRAPH_TYPE_DEFAULT);
        if (resident_prefill) { params.prefill_expert_weights = resident_bindings.data(); }
        if (wide_weights) { params.prefill_expert_weights = wide_weights->graph_bindings(); }
        const auto original_cb = params.cb;
        params.cb = [original_cb, &result, layer_output](const llama_ubatch & batch, ggml_tensor * tensor, const char * name, int layer) {
            original_cb(batch, tensor, name, layer);
            if (std::strcmp(name, layer_output) == 0) { ggml_build_forward_expand(result.get_gf(), tensor); }
        };
        auto * graph = model.build_graph(params);
        if (fused && graph) {
            int fused_layers = 0;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                const auto * node = ggml_graph_node(graph, i);
                if (!expert_executor->window_projection_name(node)) { continue; }
                if (shared_scratch && std::strncmp(node->name, "ffn_moe_down-", 13) != 0) {
                    return static_cast<ggml_cgraph *>(nullptr);
                }
                ++fused_layers;
            }
            if (fused_layers != routed_layers) {
                LLAMA_LOG_ERROR("%s: fused window graph has %d of %d MoE layers (%u tokens, %lld outputs)\n",
                        __func__, fused_layers, routed_layers, u.n_tokens, (long long) n_outputs);
                return static_cast<ggml_cgraph *>(nullptr);
            }
        }
        size_t tensor_bound = model.n_tensors();
        for (auto * t = ggml_get_first_tensor(result.get_ctx()); t; t = ggml_get_next_tensor(result.get_ctx(), t)) {
            ++tensor_bound;
        }
        if (!graph || tensor_bound > sched_nodes) {
            LLAMA_LOG_DEBUG("%s: prefill graph has %zu tensors, bound %zu\n", __func__, tensor_bound, sched_nodes);
            return static_cast<ggml_cgraph *>(nullptr);
        }
        if (!ggml_backend_window_alloc(sched.get(), graph, sched_nodes)) {
            LLAMA_LOG_ERROR("%s: prefill backend could not allocate graph (%u tokens, %lld outputs)\n",
                    __func__, u.n_tokens, (long long) n_outputs);
            return static_cast<ggml_cgraph *>(nullptr);
        }
        if (native_wide) {
            int expert_ops = 0;
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                auto * node = ggml_graph_node(graph, i);
                if (node->op == GGML_OP_MUL_MAT_ID) {
                    if (ggml_backend_sched_get_tensor_backend(sched.get(), node) != backend_ptrs.front()) {
                        return static_cast<ggml_cgraph *>(nullptr);
                    }
                    ++expert_ops;
                }
            }
            if (expert_ops != 3 * routed_layers) { return static_cast<ggml_cgraph *>(nullptr); }
        }
        return graph;
    };
    if (batch_experts) {
        // Reserve backend buffers against a synthetic full cache before
        // applying any real recurrent/KV update.
        const auto saved_outputs = n_outputs;
        int prepared = 0;
        try {
            const auto full = memory->init_full();
            if (full && !llama_memory_status_is_fail(full->get_status())) {
                llama_batch_allocr allocator(model.hparams.n_pos_per_embd());
                auto dummy = allocator.ubatch_reserve(main_rows, 1);
                std::fill(dummy.output, dummy.output + main_rows, int8_t(0));
                dummy.output[main_rows - 1] = 1;
                n_outputs = 1;
                for (; prepared < window_count; ++prepared) {
                    // All workers use the same synthetic full-cache geometry.
                    // Each still owns its buffers and plans the real graph separately.
                    if (wide_submit && prepared > 0 &&
                            ggml_backend_window_reserve_like(schedulers[prepared].get(), schedulers.front().get())) {
                        ++reused_reservations;
                        continue;
                    }
                    sched = std::move(schedulers[prepared]);
                    reset_window_graph(prepared);
                    const bool allocated = allocate_window_graph(*results[prepared], dummy, full.get()) != nullptr &&
                        (!compact || graph_mappings[prepared].trim(results[prepared]->get_ctx()));
                    ggml_backend_sched_reset(sched.get());
                    schedulers[prepared] = std::move(sched);
                    if (compact) {
                        results[prepared].reset();
                        graph_mappings[prepared].clear();
                    }
                    if (!allocated) { break; }
                }
            }
        } catch (const std::exception &) {
            if (sched) { schedulers[prepared] = std::move(sched); }
        }
        n_outputs = saved_outputs;
        if (retry_smaller_wide && prepared <= 1 && batch_workers > 1) {
            *retry_smaller_wide = true;
            cleanup();
            return fallback();
        }
        if (smaller_wide && prepared < smaller_wide_workers) {
            cleanup();
            return fallback();
        }
        const size_t min_experts = native_wide ? 0 : size_t(model.hparams.n_expert_used_max()) * main_rows;
        int admitted = 0;
        for (int count : admitted_window_counts) {
            if (compact) { count = std::min(count, window_count); }
            if (count > prepared) { continue; }
            for (int i = count; i < window_count; ++i) {
                schedulers[i].reset();
                results[i].reset();
                if (compact) { graph_mappings[i].clear(); }
            }
            if (expert_executor->window_expert_capacity() >= min_experts) {
                admitted = count;
                break;
            }
        }
        if (native_wide && !admitted && prepared > 0) {
            // Some workers fit. Keep their block geometry and reuse them
            // across consecutive groups from the same logical batch.
            admitted = prepared;
            for (int i = admitted; i < window_count; ++i) {
                schedulers[i].reset();
                results[i].reset();
                graph_mappings[i].clear();
            }
        }
        // A grouped request can safely retry using ub8 only before consuming KV.
        if (!admitted || ((shared_dflash || (grouped && !native_wide)) && admitted != batch_workers)) {
            cleanup();
            return fallback();
        }
        window_count = admitted;
        if (compact) {
            LLAMA_LOG_INFO("%s: compact window admitted %d workers for %zu input tokens\n",
                    __func__, window_count, batch_tokens);
        }
    }
    bool wide_overlap = false;
    if (wide_weights) {
        const char * overlap = std::getenv("LLAMA_EXPERIMENTAL_PREFILL_WIDE_OVERLAP");
        if (!overlap || !std::strcmp(overlap, "1")) {
            try {
                wide_overlap = wide_weights->enable_overlap(ledger, shared_wide, shared_dflash);
            } catch (const std::bad_alloc &) {
                // The serial graph and its first bank are already reserved.
            }
            if (!wide_overlap && (smaller_wide || shared_dflash)) {
                cleanup();
                return fallback();
            }
            if (!wide_overlap) {
                LLAMA_LOG_INFO("%s: no wide read-ahead buffer available; using serial staging\n", __func__);
            }
        }
    }
    const size_t wide_staging_capacity = wide_weights ? wide_weights->staging_capacity() : 0;
    uint64_t wide_read_bytes = 0, wide_prefetched_bytes = 0, wide_tail_bytes = 0;
    int64_t wide_read_us = 0, wide_wait_us = 0, wide_compute_us = 0;
    try {
        if ((batch_experts && !native_wide) || trace) { handoffs.resize(window_count); }
        if (backend_cpu) {
            auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
            auto * set_pool = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
            if (set_pool) { set_pool(backend_cpu, threadpool_batch); }
        }
        for (const auto & fn : set_n_threads_fns) { fn.second(fn.first, cparams.n_threads_batch); }
        bool more = true;
        while (more) {
            int count = 0;
            for (int i = 0; i < window_count && more; ++i) {
                const auto & u = mctx->get_ubatch();
                if (u.n_tokens < 1 || u.n_tokens > uint32_t(main_rows) || u.n_seqs != 1 || !mctx->apply()) {
                    throw std::runtime_error("invalid window microbatch");
                }
                n_outputs = 0;
                for (uint32_t t = 0; t < u.n_tokens; ++t) { n_outputs += u.output[t] != 0; }
                outputs[i] = n_outputs;
                token_counts[i] = u.n_tokens;
                sched = std::move(schedulers[i]);
                ggml_backend_sched_reset(sched.get());
                reset_window_graph(i);
                auto * graph = allocate_window_graph(*results[i], u, mctx.get());
                if (!graph) {
                    throw std::runtime_error("window graph exceeds admitted storage");
                }
                if (compact && !graph_mappings[i].trim(results[i]->get_ctx())) {
                    throw std::runtime_error("window graph metadata exceeds retained mapping");
                }
                results[i]->set_inputs(&u);
                ggml_backend_sched_synchronize(sched.get());
                workers[i].reset();
                workers[i].layer_count = layer_count;
                workers[i].cross_phases = cross_phases;
                workers[i].layer_output_prefix = flash ? "l_last-" : "l_out-";
                workers[i].sched = sched.get();
                workers[i].graph = graph;
                workers[i].budget = model.memory_contract->ledger;
                ggml_backend_sched_set_eval_callback(sched.get(), llama_prefill_window_worker::observe, &workers[i]);
                if ((batch_experts && !native_wide) || trace) {
                    handoffs[i] = {&workers[i], expert_executor.get(), batch_experts, cross, trace.get()};
                    ggml_backend_window_set_before(sched.get(), window_route_handoff::observe, &handoffs[i]);
                }
                schedulers[i] = std::move(sched);
                ++count;
                more = mctx->next();
            }
            if (batch_submit) {
                std::fill(cursors.begin(), cursors.begin() + count, ggml_window_cursor{});
            } else {
                for (int i = 0; i < count; ++i) { workers[i].launch(compact); }
            }
            if (trace) { std::fprintf(trace.get(), "window %d\n", count); }
            constexpr size_t read_batch = 16;
            using window_routes = std::bitset<llama_prefill_window_worker::max_experts>;
            using window_reads = std::array<llama_expert_executor::window_demand, window_routes().size() / read_batch>;
            if (wide_weights) { wide_weights->queue(0); }
            for (int layer = 0; layer < layer_count; ++layer) {
                if (native_wide) {
                    if (wide_weights) {
                        const auto wait_start = ggml_time_us();
                        const auto read = wide_weights->wait(layer);
                        wide_read_bytes += read.bytes;
                        wide_prefetched_bytes += read.prefetched_bytes;
                        wide_tail_bytes += read.tail_bytes;
                        wide_read_us += read.elapsed_us;
                        wide_wait_us += ggml_time_us() - wait_start;
                        // The previous layer's flush has retired the bank being
                        // reused. This layer's bank stays read-only through its flush.
                        if (layer + 1 < layer_count) { wide_weights->queue(layer + 1); }
                    }
                    const auto compute_start = ggml_time_us();
                    const std::string output_name = std::string(layer_output) + "-" + std::to_string(layer);
                    for (int i = 0; i < count; ++i) {
                        workers[i].cross_node = ggml_graph_get_tensor(workers[i].graph, output_name.c_str());
                        step_main(i, GGML_WINDOW_EXACT_LAYER_END);
                        workers[i].reached = layer;
                    }
                    flush_main();
                    wide_compute_us += ggml_time_us() - compute_start;
                    continue;
                }
                if (!batch_experts || !layer_weights[layer][0]) {
                    for (int i = 0; i < count; ++i) {
                        if (batch_submit) {
                            step_main(i, GGML_WINDOW_DENSE_LAYER_END);
                            if ((i + 1) % chunk == 0) { flush_main(); }
                        } else {
                            workers[i].advance(layer);
                        }
                    }
                    if (batch_submit) { flush_main(); }
                    continue;
                }
                const size_t expert_group_bytes = expert_executor->window_expert_group_bytes(layer_weights[layer]);
                const size_t queued_experts = pipeline_depth ? size_t(model.hparams.n_expert) : 0;
                window_routes early_routes;
                window_reads early_reads;
                auto read_routes = [&](const window_routes & routes, window_reads & into) {
                    std::array<int64_t, llama_prefill_window_worker::max_experts> ids;
                    size_t n = 0;
                    for (size_t id = 0; id < routes.size(); ++id) {
                        if (routes[id]) { ids[n++] = int64_t(id); }
                    }
                    for (size_t offset = 0; offset < n; offset += read_batch) {
                        into[offset / read_batch] = expert_executor->enqueue_window_experts(
                                layer_weights[layer], ids.data() + offset, std::min(read_batch, n - offset));
                    }
                };
                int submitted_through = 0;
                for (int i = 0; i < count; ++i) {
                    if (batch_submit) {
                        if (i == submitted_through) {
                            // Publish the first routes on the original cadence so
                            // demand reads can start while later views run.
                            const int width = i == 0 && early_demand ? chunk : submit_views;
                            submitted_through = std::min(i + width, count);
                            for (int j = i; j < submitted_through; ++j) {
                                step_main(j, GGML_WINDOW_ROUTE_FRONTIER);
                            }
                            flush_main();
                        }
                        step_main(i, GGML_WINDOW_ROUTES);
                        workers[i].observe_routes(layer, workers[i].cross_node->src[1], false, trace.get());
                    } else if (cross) { workers[i].begin_cross_layer(layer); }
                    else { workers[i].advance_routes(layer); }
                    if (early_demand && i + 1 == chunk && count > chunk) {
                        for (int j = 0; j < chunk; ++j) { early_routes |= workers[j].routes; }
                        if (early_routes.count() <= expert_executor->window_expert_capacity(expert_group_bytes, queued_experts)) {
                            read_routes(early_routes, early_reads);
                        } else {
                            early_routes.reset();
                        }
                    }
                }
                int layer_groups = 0;
                for (int first = 0; first < count;) {
                    ++layer_groups;
                    const size_t capacity = expert_executor->window_expert_capacity(expert_group_bytes, queued_experts);
                    std::bitset<llama_prefill_window_worker::max_experts> experts;
                    int end = first;
                    while (end < count) {
                        const auto candidate = experts | workers[end].routes;
                        if (candidate.count() > capacity) { break; }
                        experts = candidate;
                        ++end;
                    }
                    if (end == first) {
                        // All routes remain known. The ordinary executor can
                        // still stream one microbatch in smaller waves.
                        end = first + 1;
                    } else if (!pipeline_depth && !cross) {
                        std::array<int64_t, llama_prefill_window_worker::max_experts> ids;
                        size_t n_ids = 0;
                        for (size_t id = 0; id < experts.size(); ++id) {
                            if (experts[id]) { ids[n_ids++] = int64_t(id); }
                        }
                        expert_executor->pin_window_experts(layer_weights[layer], ids.data(), n_ids, layer_leases, adjacent_ranges);
                    }
                    if (cross) {
                        window_reads pending;
                        window_routes scheduled;
                        if (first == 0 && early_routes.any()) {
                            if ((early_routes & ~experts).none()) {
                                pending = std::move(early_reads);
                                scheduled = early_routes;
                            } else {
                                // A changed budget can shorten the first group.
                                early_reads = {};
                            }
                        }
                        llama_expert_executor::window_bindings bindings;
                        auto * pinned = wide_fused && experts.any() ? &bindings : nullptr;
                        if (pinned) { pinned->fill({}); }
                        auto stage_routes = [&](const window_routes & needed, bool async) {
                            if (async) {
                                read_routes(needed, pending);
                            } else {
                                std::array<int64_t, llama_prefill_window_worker::max_experts> ids;
                                size_t count = 0;
                                for (size_t id = 0; id < needed.size(); ++id) {
                                    if (needed[id]) { ids[count++] = int64_t(id); }
                                }
                                expert_executor->pin_window_experts(layer_weights[layer], ids.data(), count,
                                        layer_leases, adjacent_ranges, pinned);
                            }
                        };
                        auto stage = [&](int start, bool async) {
                            auto needed = workers[start].routes;
                            for (int i = start + 1; i < std::min(start + chunk, end); ++i) {
                                needed |= workers[i].routes;
                            }
                            needed &= ~scheduled;
                            scheduled |= needed;
                            stage_routes(needed, async);
                        };
                        auto drain = [&] {
                            for (auto & job : pending) {
                                if (!job) { break; }
                                expert_executor->drain_window_experts(job, layer_leases, pinned);
                            }
                        };
                        bool matrix_computed = false;
                        if (matrix_window && pinned) {
                            drain();
                            stage_routes(experts & ~scheduled, pipeline_depth != 0);
                            drain();
                            scheduled = experts;
                            std::array<ggml_tensor *, shared_window_count> nodes;
                            for (int i = first; i < end; ++i) { nodes[i - first] = workers[i].cross_node; }
                            matrix_computed = expert_executor->compute_matrix_window_moe(nodes.data(), end - first, bindings);
                            matrix_groups += matrix_computed;
                        } else if (experts.any()) { stage(first, false); }
                        for (int start = first; start < end; start += chunk) {
                            const int stop = std::min(start + chunk, end);
                            drain();
                            if (scheduled != experts) {
                                if (stop < end && pipeline_depth) { stage(stop, true); }
                                else if (start != first && !pipeline_depth) { stage(start, false); }
                            }
                            std::array<ggml_tensor *, chunk> nodes;
                            for (int phase = 0; phase < cross_phases; ++phase) {
                                for (int i = start; i < stop; ++i) { nodes[i - start] = workers[i].cross_node; }
                                bool computed = true;
                                if (!matrix_computed) {
                                    if (fused) { expert_executor->compute_fused_window_moe(nodes.data(), stop - start, pinned); }
                                    else { computed = expert_executor->compute_cross_window(nodes.data(), stop - start); }
                                }
                                if (batch_submit) {
                                    for (int i = start; i < stop; ++i) {
                                        step_main(i, GGML_WINDOW_LAYER_END);
                                        workers[i].reached = layer;
                                    }
                                    // Completed FFN results belong to their workers.
                                    // The scheduler still flushes at host dependencies;
                                    // finish all submitted work before releasing leases.
                                    if (stop == end || (stop - first) % submit_views == 0) { flush_main(); }
                                } else {
                                    for (int i = start; i < stop; ++i) { workers[i].advance_cross(layer, phase, computed); }
                                }
                            }
                        }
                    } else if (pipeline_depth && experts.any()) {
                        // Exact routes are known for all participants. Admit
                        // each weight once, in first-consumer order.
                        constexpr size_t slices_per_batch = 16;
                        using pending_reads = std::array<llama_expert_executor::window_demand,
                                llama_prefill_window_worker::max_experts / slices_per_batch>;
                        std::array<pending_reads, 4> pending;
                        auto scheduled = workers[first].routes;
                        std::array<int64_t, llama_prefill_window_worker::max_experts> first_ids;
                        size_t first_count = 0;
                        for (size_t id = 0; id < scheduled.size(); ++id) {
                            if (scheduled[id]) { first_ids[first_count++] = int64_t(id); }
                        }
                        expert_executor->pin_window_experts(layer_weights[layer], first_ids.data(), first_count, layer_leases, adjacent_ranges);
                        int queued_through = first + 1;
                        for (int i = first; i < end; ++i) {
                            for (auto & batch : pending[i % pipeline_depth]) {
                                if (!batch) { break; }
                                expert_executor->drain_window_experts(batch, layer_leases);
                            }
                            for (; queued_through < end && queued_through <= i + pipeline_depth; ++queued_through) {
                                auto & target = pending[queued_through % pipeline_depth];
                                const auto next = workers[queued_through].routes & ~scheduled;
                                scheduled |= next;
                                std::array<int64_t, slices_per_batch> ids;
                                size_t n_ids = 0, jobs = 0;
                                for (size_t id = 0; id < next.size(); ++id) {
                                    if (!next[id]) { continue; }
                                    ids[n_ids++] = int64_t(id);
                                    if (n_ids == ids.size()) {
                                        target[jobs++] = expert_executor->enqueue_window_experts(layer_weights[layer], ids.data(), n_ids);
                                        n_ids = 0;
                                    }
                                }
                                if (n_ids) {
                                    target[jobs] = expert_executor->enqueue_window_experts(layer_weights[layer], ids.data(), n_ids);
                                }
                            }
                            workers[i].advance_experts(layer);
                        }
                    } else {
                        for (int i = first; i < end; ++i) {
                            workers[i].advance_experts(layer);
                        }
                    }
                    // The workers are at synchronized layer-end observations.
                    layer_leases.clear();
                    first = end;
                }
                if (trace && shared_scratch) {
                    std::fprintf(stderr, "WINDOW_LAYER_GROUPS %d %d\n", layer, layer_groups);
                }
            }
            if (batch_submit) {
                for (int i = 0; i < count; ++i) {
                    step_main(i, GGML_WINDOW_FINISH);
                    workers[i].status = GGML_STATUS_SUCCESS;
                    if ((i + 1) % chunk == 0) { flush_main(); }
                }
                flush_main();
            } else {
                for (int i = 0; i < count; ++i) { workers[i].finish(); }
            }
            for (int i = 0; i < count; ++i) {
                auto & w = workers[i];
                if (w.failure) { std::rethrow_exception(w.failure); }
                if (w.status != GGML_STATUS_SUCCESS || w.reached != layer_count - 1 || expert_executor->error()) {
                    throw std::runtime_error("window graph compute failed");
                }
                if (outputs[i]) {
                    ggml_backend_tensor_get(results[i]->get_logits(), logits.data + n_outputs_prev * n_vocab,
                            0, size_t(outputs[i]) * n_vocab * sizeof(float));
                }
                n_outputs_prev += outputs[i];
                n_tokens_prev += token_counts[i];
            }
            ++counters.windows;
            counters.tokens += n_tokens_prev - counted_tokens;
            counted_tokens = n_tokens_prev;
            counters.tails += count != window_count || token_counts[count - 1] != main_rows;
            bool pending;
            {
                std::lock_guard<std::mutex> lock(budget_control->budget_mutex);
                pending = budget_control->budget_status == LLAMA_BUDGET_PENDING;
            }
            if (pending && (!native_wide || !more)) {
                // mctx already points at the next unprocessed microbatch. Drop
                // every window resource before applying a smaller memory cap.
                cleanup();
                apply_memory_budget();
                if (more) { ++counters.fallbacks; }
                return more ? 0 : 1;
            }
        }
        // Keep the bank owners alive while cleanup joins the workers, restores the
        // decode scheduler and destroys every graph that borrowed their tensors.
        auto retained_weights = retain_tail ? std::move(wide_weights) : nullptr;
        cleanup();
        size_t retained_bytes = 0;
        if (retained_weights && retain_qwen && wide_dflash) {
            // The draft and its verifier still need workspace after the wide workers retire.
            constexpr size_t verify_headroom = size_t(768) << 20;
            const auto used = ledger.snapshot(llama_memory_domain::host).used();
            if (used > window_budget - verify_headroom) {
                retained_weights.reset();
                if (std::getenv("ATOMIC_WINDOW_PROFILE")) {
                    std::fprintf(stderr, "QWEN35_PREFILL_RETAIN_DECLINED %zu\n", window_budget - used);
                }
            }
        }
        if (retained_weights) {
            reset_expert_graphs();
            retained_bytes = expert_executor->retain_prefill_tail(*retained_weights, layer_count - 2);
            retained_weights.reset();
        }
        if (batch_submit && std::getenv("ATOMIC_WINDOW_PROFILE")) {
            const char * family = lfm_residency_profile ? "LFM" : "QWEN35";
            if (resident_prefill || retained_bytes) {
                if (resident_prefill) { std::fprintf(stderr, "%s_DIRECT_PREFILL %d\n", family, routed_layers); }
                if (retained_bytes) { std::fprintf(stderr, "%s_PREFILL_RETAINED %zu\n", family, retained_bytes); }
            }
            std::fprintf(stderr, "WINDOW_MAIN_SUBMISSIONS %llu\n", (unsigned long long) main_batch.submissions);
            if (wide_submit) {
                std::fprintf(stderr, "WINDOW_MAIN_SUBMIT_VIEWS %d\n", submit_views);
                std::fprintf(stderr, "WINDOW_REUSED_RESERVATIONS %d\n", reused_reservations);
            }
            if (matrix_opt) { std::fprintf(stderr, "WINDOW_MATRIX_GROUPS %zu\n", matrix_groups); }
            if (grouped) { std::fprintf(stderr, "WINDOW_MAIN_ROWS %d\n", main_rows); }
            if (native_wide) {
                if (hybrid_prefill) { std::fprintf(stderr, "%s_HYBRID_PREFILL %d %d\n", family, direct_layers, routed_layers); }
                std::fprintf(stderr, "WINDOW_WIDE %d %d %llu %lld %lld %lld\n", int(wide_overlap), layer_count,
                        (unsigned long long) wide_read_bytes, (long long) wide_read_us,
                        (long long) wide_wait_us, (long long) wide_compute_us);
                std::fprintf(stderr, "WINDOW_WIDE_STAGING %d %zu %llu\n", window_count, wide_staging_capacity,
                        (unsigned long long) wide_prefetched_bytes);
                if (wide_tail_bytes) {
                    std::fprintf(stderr, "WINDOW_WIDE_TAIL_BYTES %llu\n", (unsigned long long) wide_tail_bytes);
                }
            }
        }
        return 1;
    } catch (const std::exception & e) {
        cleanup(true);
        counters.failed = true;
        LLAMA_LOG_ERROR("prefill window failed; discard this context: %s\n", e.what());
        return -3;
    }
}
