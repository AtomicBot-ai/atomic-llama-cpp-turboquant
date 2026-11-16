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

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace {
constexpr size_t window_budget = size_t(8) * 1024 * 1024 * 1024;
constexpr size_t sched_nodes = 8192;
constexpr int window_count = 8;
}

bool llama_context::prefill_window_set(bool enabled) {
#ifndef __APPLE__
    if (enabled) { return false; }
#endif
    if (!budget_control || !expert_executor || !model.memory_contract ||
            model.memory_contract->discrete.device || model.arch != LLM_ARCH_QWEN4EXP ||
            model.hparams.n_layer() != 48 || backend_ptrs.empty() ||
            std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_ptrs.front()))), "MTL") != 0) { return false; }
    if (budget_control->prefill_window.failed) { return false; }
    if (!enabled) { expert_executor->clear_window_decode_advice(); }
    budget_control->prefill_window.enabled = enabled;
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
        size_t prompt_tokens, int64_t & n_outputs_prev, int64_t & n_tokens_prev) {
    auto & counters = budget_control->prefill_window;
    auto fallback = [&] { ++counters.fallbacks; return 0; };
    if (!expert_executor->can_layer_window() || prompt_tokens < 320 || n_outputs_all > 1 || cparams.n_ctx != 4096 || cparams.n_batch > 256 ||
            cparams.n_ubatch != 8 || cparams.n_seq_max != 1 || cparams.pipeline_parallel ||
            cparams.embeddings || cparams.embeddings_nextn || cparams.embeddings_nextn_dev ||
            cparams.embeddings_layer_inp_dev || !cparams.causal_attn || cparams.training ||
            cparams.ctx_type != LLAMA_CONTEXT_TYPE_DEFAULT || cparams.cb_eval || abort_callback ||
            cparams.ctx_other || cparams.ctx_state_src || backend_ptrs.size() > 3 || graph_max_nodes() > 40000 ||
            (loras && !loras->empty()) || std::any_of(cparams.embeddings_layer_inp.begin(),
                cparams.embeddings_layer_inp.end(), [](bool v) { return v; })) { return fallback(); }
    for (int i = 0; i < 48; ++i) { if (cvec && cvec->tensor_for(i)) { return fallback(); } }
    {
        std::lock_guard<std::mutex> lock(budget_control->budget_mutex);
        if (budget_control->budget_status == LLAMA_BUDGET_PENDING ||
                budget_control->budget_policy.status().config.mode != llama_memory_mode::fixed ||
                budget_control->budget_requested != window_budget) { return fallback(); }
    }
    const auto memory = model.memory_contract->ledger->snapshot(llama_memory_domain::host);
    if (memory.requested != window_budget || memory.applied != window_budget) { return fallback(); }
    synchronize();
    expert_executor->trim();
    // Each slot covers the graph arena, compact scheduler metadata and an <=8 MiB
    // worker stack. Backend buffers are charged separately by allocation_scope.
    auto metadata = model.memory_contract->ledger->reserve(llama_memory_domain::host,
            llama_memory_category::scratch, size_t(window_count) * 64 * 1024 * 1024);
    auto construction = model.memory_contract->ledger->reserve(llama_memory_domain::host,
            llama_memory_category::scratch, size_t(224) * 1024 * 1024);
    if (!metadata || !construction) { return fallback(); }
    std::vector<std::unique_ptr<llm_graph_result>> results(window_count);
    std::vector<ggml_backend_sched_ptr> schedulers(window_count);
    std::vector<llama_prefill_window_worker> workers(window_count);
    std::vector<int32_t> outputs(window_count), token_counts(window_count);
    // Constructors and admission precede all recurrent/KV state mutations.
    for (auto & scheduler : schedulers) {
        scheduler.reset(ggml_backend_window_new(backend_ptrs.data(), backend_buft.data(),
                backend_ptrs.size(), sched_nodes, sched_params_cur.op_offload));
        if (!scheduler) { return fallback(); }
    }
    metadata.commit();
    construction.reset();
    if (!expert_executor->set_layer_window(true)) { return fallback(); }
    const size_t graph_nodes = graph_max_nodes();
    const int64_t n_vocab = model.vocab.n_tokens();
    int64_t counted_tokens = n_tokens_prev;
    auto original_sched = std::move(sched);
    auto cleanup = [&] {
        for (auto & worker : workers) { worker.finish(); }
        // Restore before destroying graph metadata; no paused compute survives.
        if (original_sched) { sched = std::move(original_sched); }
        schedulers.clear();
        results.clear();
        workers.clear();
        metadata.reset();
        expert_executor->set_layer_window(false);
    };
    try {
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
                if (u.n_tokens < 1 || u.n_tokens > 8 || u.n_seqs != 1 || !mctx->apply()) {
                    throw std::runtime_error("invalid window microbatch");
                }
                n_outputs = 0;
                for (uint32_t t = 0; t < u.n_tokens; ++t) { n_outputs += u.output[t] != 0; }
                outputs[i] = n_outputs;
                token_counts[i] = u.n_tokens;
                sched = std::move(schedulers[i]);
                ggml_backend_sched_reset(sched.get());
                if (!results[i]) { results[i] = std::make_unique<llm_graph_result>(graph_nodes); }
                else { results[i]->reset(); }
                auto params = graph_params(results[i].get(), u, mctx.get(), LLM_GRAPH_TYPE_DEFAULT);
                const auto original_cb = params.cb;
                params.cb = [original_cb, result = results[i].get()](const llama_ubatch & batch, ggml_tensor * tensor, const char * name, int layer) {
                    original_cb(batch, tensor, name, layer);
                    if (std::strcmp(name, "l_last") == 0) { ggml_build_forward_expand(result->get_gf(), tensor); }
                };
                auto * graph = model.build_graph(params);
                size_t tensor_bound = model.n_tensors();
                for (auto * t = ggml_get_first_tensor(results[i]->get_ctx()); t; t = ggml_get_next_tensor(results[i]->get_ctx(), t)) { ++tensor_bound; }
                if (!graph || tensor_bound > sched_nodes || !ggml_backend_window_alloc(sched.get(), graph, sched_nodes)) {
                    throw std::runtime_error("window graph exceeds admitted storage");
                }
                results[i]->set_inputs(&u);
                ggml_backend_sched_synchronize(sched.get());
                workers[i].reset();
                workers[i].sched = sched.get();
                workers[i].graph = graph;
                workers[i].budget = model.memory_contract->ledger;
                ggml_backend_sched_set_eval_callback(sched.get(), llama_prefill_window_worker::observe, &workers[i]);
                schedulers[i] = std::move(sched);
                ++count;
                more = mctx->next();
            }
            for (int i = 0; i < count; ++i) { workers[i].launch(); }
            for (int layer = 0; layer < 48; ++layer) {
                for (int i = 0; i < count; ++i) { workers[i].advance(layer); }
            }
            for (int i = 0; i < count; ++i) { workers[i].finish(); }
            for (int i = 0; i < count; ++i) {
                auto & w = workers[i];
                if (w.failure) { std::rethrow_exception(w.failure); }
                if (w.status != GGML_STATUS_SUCCESS || w.reached != 47 || expert_executor->error()) {
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
            counters.tails += count != window_count || token_counts[count - 1] != 8;
            bool pending;
            {
                std::lock_guard<std::mutex> lock(budget_control->budget_mutex);
                pending = budget_control->budget_status == LLAMA_BUDGET_PENDING;
            }
            if (pending) {
                // mctx already points at the next unprocessed microbatch. Drop
                // every window resource before applying a smaller memory cap.
                cleanup();
                apply_memory_budget();
                if (more) { ++counters.fallbacks; }
                return more ? 0 : 1;
            }
        }
        cleanup();
        return 1;
    } catch (const std::exception & e) {
        cleanup();
        counters.failed = true;
        LLAMA_LOG_ERROR("prefill window failed; discard this context: %s\n", e.what());
        return -3;
    }
}
