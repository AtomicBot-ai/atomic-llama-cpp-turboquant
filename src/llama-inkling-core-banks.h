#pragma once

#include "llama-expert-source.h"
#include "llama-memory-contract.h"
#include "models/models.h"
#include "ggml-cpp.h"
#if defined(_WIN32)
#include "llama-spark-hot-experts.h"
#endif

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

// The streamed FFN and resident banks use the same native Metal weight layouts.
inline bool llama_inkling_native_expert_pair(const ggml_tensor * gate_up, const ggml_tensor * down) {
    if (!gate_up || !down) { return false; }
    switch (gate_up->type) {
        case GGML_TYPE_Q4_K:    return down->type == GGML_TYPE_Q5_0;
        case GGML_TYPE_IQ3_S:   return down->type == GGML_TYPE_IQ4_NL;
        case GGML_TYPE_IQ2_XXS: return down->type == GGML_TYPE_IQ4_NL || down->type == GGML_TYPE_Q2_0;
        default:                return false;
    }
}

// The loop visits the same physical layers several times. A complete bank can
// use Metal's ordinary routed matmul without a per-visit host callback.
class llama_inkling_core_banks {
public:
    using weights = std::array<ggml_tensor *, 2>; // gate-up, down

    llama_inkling_core_banks(const llama_model_inkling2_loop & model, bool include_outer, bool prefer_full) :
        model(model), include_outer(include_outer), prefer_full(include_outer && prefer_full && model.n_phys == 24) {}

    struct limits {
        size_t core;
        size_t total;
        size_t full = 0;
    };

    limits allowance(size_t cap, size_t floor, bool balanced_prompt) const {
        if (model.memory_contract->discrete.unified) {
            // Preserve scratch for the native routed graph when a streamed layer becomes resident.
            const size_t room = cap > floor ? cap - floor : 0;
            const size_t reserve = std::min(room / 2, size_t(512) << 20);
            return {room - reserve, room - reserve};
        }
        constexpr size_t mib = size_t(1) << 20;
        // Spend only capacity above the measured small-machine profile. Keep
        // at least 2 GiB beyond the mandatory floor for remaining streamed work.
        const size_t base = (8192 + 100) * mib;
        if (balanced_prompt && cap == base) {
            // Two frequently revisited physical banks. Leave at least 1 GiB
            // above the mandatory floor for the prompt matrix and demand reads.
            const size_t allowed = cap > floor && cap - floor > 1024*mib ? std::min(1200*mib, cap - floor - 1024*mib) : 0;
            return {allowed, allowed};
        }
        const size_t total = cap > floor && cap - floor > 2048*mib ? cap - floor - 2048*mib : 0;
        // The full 20 GiB profile needs no streamed prompt matrix once every
        // bank is resident. Retain headroom even for this all-or-nothing case.
        const size_t full = cap == 20580*mib && cap > floor && cap - floor > 512*mib ? cap - floor - 512*mib : 0;
        return {cap > base ? std::min(cap - base, total) : 0, total, full};
    }

    bool all_resident() const { return banks.size() == size_t(model.n_phys); }
    size_t bytes() const {
#if defined(_WIN32)
        if (hot) { return charged + hot->charged; }
#endif
        return charged;
    }
#if defined(_WIN32)
    void set_spark_backend(ggml_backend_t backend) { spark_backend = backend; }
    llama_spark_hot_experts::projection * hot_projection(const std::string & name) {
        return hot ? hot->find(name) : nullptr;
    }
#endif

    bool can_restore_full(limits budget) const {
        return budget.full && full_set_charge > charged && full_set_charge <= bank_limit(budget, true);
    }

    const weights * get(int ip) const {
        if (ip < 0 || ip >= model.n_phys) { return nullptr; }
        const size_t idx = (uint32_t(ip) < model.core_start ? size_t(model.n_phys) : 0) + ip - model.core_start;
        return idx < banks.size() ? &banks[idx].tensors : nullptr;
    }

    void shrink(limits budget) {
#if defined(_WIN32)
        if (hot && bytes() > bank_limit(budget, false)) { hot.reset(); }
#endif
        while (!banks.empty() && charged > bank_limit(budget, banks.size() == size_t(model.n_phys))) {
            charged -= banks.back().charge;
            banks.pop_back();
        }
    }

    void grow(limits budget) {
#if defined(_WIN32)
        hot.reset();
#endif
        try {
            load(budget);
#if defined(_WIN32)
            const char * mode = model.memory_contract->discrete.unified && model.memory_contract->inkling_dspark_target
                    ? std::getenv("LLAMA_EXPERIMENTAL_SPARK_HOT_EXPERTS") : nullptr;
            if (mode && !all_resident() && banks.size() > size_t(model.core_end - model.core_start)) {
                const bool direct = std::strcmp(mode, "direct") == 0 && spark_backend;
                if (std::strcmp(mode, "1") == 0 || direct) {
                    // Trade one outer bank for fixed slots across the remaining layers.
                    const size_t limit = banks.back().charge;
                    charged -= limit;
                    banks.pop_back();
                    hot = llama_spark_hot_experts::create(model, banks.size(), limit, direct ? spark_backend : nullptr);
                    if (!hot) { load(budget); }
                } else if (std::strcmp(mode, "spare") == 0) {
                    // Use only the remaining bank allowance, preserving the streaming reserve.
                    const size_t allowance = bank_limit(budget, false);
                    if (allowance > charged) {
                        const size_t limit = std::min(allowance - charged, size_t(128) << 20);
                        hot = llama_spark_hot_experts::create(model, banks.size(), limit);
                    }
                }
            }
#endif
        } catch (const std::bad_alloc &) {
            // Each completed bank is usable; the remaining layers still stream.
        }
        // A partial set still needs the normal streaming reserve.
        if (prefer_full && budget.full && banks.size() != size_t(model.n_phys)) { shrink(budget); }
        if (prefer_full && banks.size() == size_t(model.n_phys)) { full_set_charge = charged; }
        LLAMA_LOG_INFO("Inkling resident %s: %zu physical layers, %.1f MiB\n",
                include_outer ? "banks" : "core", banks.size(), charged / 1048576.);
    }

private:
    size_t bank_limit(limits budget, bool allow_full) const {
        // Once the regular allowance covers the whole repeated core, outer banks
        // replace streamed work. Keep the same 2 GiB reserve above the floor.
        if (!include_outer || !full_core_charge || budget.core < full_core_charge) { return budget.core; }
        return prefer_full && allow_full && budget.full ? budget.full : budget.total;
    }

    void load(limits capacity) {
#if defined(__APPLE__) || defined(_WIN32)
        const auto dev = model.dev_output();
        const auto buft = ggml_backend_dev_buffer_type(dev);
        using host_base_fn = void * (*)(ggml_backend_buffer_t);
        const auto host_base = reinterpret_cast<host_base_fn>(ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(dev), "ggml_backend_metal_buffer_get_host_base"));
        if (!model.memory_contract->discrete.unified && !host_base) { return; }
        using mark_grouped_fn = void (*)(ggml_tensor *);
        mark_grouped_fn mark_grouped = nullptr;
        const char * grouped = std::getenv("LLAMA_EXPERIMENTAL_INKLING_GROUP_EXPERTS");
        if (grouped && std::strcmp(grouped, "1") == 0) {
            mark_grouped = reinterpret_cast<mark_grouped_fn>(ggml_backend_reg_get_proc_address(
                    ggml_backend_dev_backend_reg(dev), "ggml_backend_metal_mark_inkling_group_bank"));
        }
        if (model.memory_contract->discrete.unified && model.memory_contract->inkling_dspark_target) {
            const char * native = std::getenv("LLAMA_EXPERIMENTAL_SPARK_VERIFY_MMQ");
            if (native && (std::strcmp(native, "1") == 0 || std::strcmp(native, "q8") == 0)) {
                mark_grouped = reinterpret_cast<mark_grouped_fn>(ggml_backend_reg_get_proc_address(
                        ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_spark_mark_verify_mmq"));
            }
        }
        const auto & budget = model.memory_contract->ledger;
        // Prioritize the repeated core, then the suffix and finally the prefix.
        // A shrink pops the least preferred banks first, retaining this order.
        const size_t limit = include_outer ? size_t(model.n_phys) : model.core_end - model.core_start;
        for (size_t i = banks.size(); i < limit; ++i) {
            const size_t allowed = bank_limit(capacity, true);
            if (charged >= allowed) { break; }
            const size_t ip = (model.core_start + i) % model.n_phys;
            const auto & layer = model.layers[ip];
            const weights original = {layer.ffn_gate_up_exps, layer.ffn_down_exps};
            const bool spark_nvfp4 = model.memory_contract->discrete.unified && original[0] && original[1] &&
                    original[0]->type == GGML_TYPE_NVFP4 && original[1]->type == GGML_TYPE_NVFP4;
            if (model.dev_layer(ip) != dev || (!spark_nvfp4 && !llama_inkling_native_expert_pair(original[0], original[1])) ||
                    !model.streamed_experts.count(original[0]) || !model.streamed_experts.count(original[1])) { break; }
            const size_t meta = 2 * ggml_tensor_overhead();
            if (ggml_nbytes(original[0]) + ggml_nbytes(original[1]) + meta > allowed - charged) { break; }
            bank b;
            b.metadata_charge = budget->reserve(llama_memory_domain::host, llama_memory_category::runtime, meta);
            if (!b.metadata_charge) { break; }
            b.ctx.reset(ggml_init({meta, nullptr, true}));
            if (!b.ctx) { break; }
            b.metadata_charge.commit();
            for (size_t k = 0; k < original.size(); ++k) {
                b.tensors[k] = ggml_dup_tensor(b.ctx.get(), original[k]);
                if (mark_grouped) { mark_grouped(b.tensors[k]); }
                ggml_set_name(b.tensors[k], original[k]->name);
            }
            const size_t size = ggml_backend_alloc_ctx_tensors_from_buft_size(b.ctx.get(), buft);
            if (!size || size > ggml_backend_buft_get_max_size(buft) || size + meta > allowed - charged) { break; }
            const auto before = budget->snapshot(llama_memory_domain::host).categories[size_t(llama_memory_category::experts)].total();
            {
                llama_allocation_scope admission(budget, llama_memory_category::experts, {}, model.memory_contract->discrete);
                b.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(b.ctx.get(), buft));
            }
            if (!b.buffer || (host_base && !host_base(b.buffer.get()))) { break; }
            const auto after = budget->snapshot(llama_memory_domain::host).categories[size_t(llama_memory_category::experts)].total();
            if (after < before || after - before < ggml_backend_buffer_get_size(b.buffer.get())) {
                throw std::runtime_error("Inkling core bank allocation was not fully charged");
            }
            b.charge = meta + after - before;
            if (b.charge > allowed - charged) { break; }
            ggml_backend_buffer_set_usage(b.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(b.buffer.get(), 0);
            for (auto * w : b.tensors) {
                const auto & info = model.expert_source->tensor(w->name);
                if (w->buffer != b.buffer.get() || !ggml_is_contiguous(w) || info.expert_bytes() != w->nb[2] ||
                        info.bytes != ggml_nbytes(w)) { return; }
#if defined(__APPLE__)
                model.expert_source->read_expert_range(w->name, 0, w->ne[2], w->data, info.bytes);
#else
                // CUDA addresses need an explicit upload even on physically shared RAM.
                const size_t stride = info.expert_bytes();
                auto charge = budget->reserve(llama_memory_domain::host, llama_memory_category::io, stride);
                if (!charge) { return; }
                std::unique_ptr<uint8_t[]> data(new uint8_t[stride]);
                charge.commit();
                for (int64_t expert = 0; expert < info.ne[2]; ++expert) {
                    model.expert_source->read_expert(w->name, expert, data.get(), stride);
                    ggml_backend_tensor_set(w, data.get(), size_t(expert) * stride, stride);
                }
#endif
            }
            banks.push_back(std::move(b));
            charged += banks.back().charge;
            if (ip + 1 == model.core_end) { full_core_charge = charged; }
        }
#endif
    }

    struct bank {
        llama_memory_budget::ticket metadata_charge;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buffer;
        weights tensors{};
        size_t charge = 0;
    };
    const llama_model_inkling2_loop & model;
    const bool include_outer;
    const bool prefer_full;
    std::vector<bank> banks;
#if defined(_WIN32)
    std::unique_ptr<llama_spark_hot_experts> hot;
    ggml_backend_t spark_backend = nullptr;
#endif
    size_t charged = 0;
    size_t full_core_charge = 0;
    size_t full_set_charge = 0;
};
