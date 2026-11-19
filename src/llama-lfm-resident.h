#pragma once

#include "llama-expert-source.h"
#include "llama-memory-contract.h"
#include "llama-model.h"
#include "llama-prefill-wide.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

// Context-owned, immutable dense banks for the qualified Metal streaming profiles.
// Graphs may borrow a bank until the synchronized auto-acceleration resize boundary.
class llama_lfm_resident {
public:
    using weights = std::array<ggml_tensor *, 3>; // gate, up, down
    explicit llama_lfm_resident(uint32_t first) : first(first) {}

    size_t bytes() const {
        size_t total = charged;
        for (const auto & bank : tail) {
            if (bank.context) { total += bank.buffer_bytes + bank.metadata.size(); }
        }
        return total;
    }
    size_t count() const { return layers.size() + size_t(bool(tail[0].context)) + size_t(bool(tail[1].context)); }
    bool has_tail() const { return tail[0].context || tail[1].context; }
    const weights * get(int il) const {
        if (il >= int(first) && size_t(il - first) < layers.size()) { return &layers[il - first].tensors; }
        if (il < tail_first || size_t(il - tail_first) >= tail.size()) { return nullptr; }
        const auto & bank = tail[il - tail_first];
        return bank.context ? &bank.weights : nullptr;
    }

    size_t adopt_tail(int first_layer, std::array<llama_prefill_wide_weights::retained_bank, 2> banks) {
        GGML_ASSERT(!has_tail());
        tail_first = first_layer;
        size_t transferred = 0;
        for (size_t i = 0; i < banks.size(); ++i) {
            if (!banks[i].context || get(first_layer + int(i))) { continue; }
            transferred += banks[i].buffer_bytes;
            tail[i] = std::move(banks[i]);
        }
        return transferred;
    }

    void release_tail() {
        // Destroy the allocations before releasing their metadata reservations.
        auto retired = std::move(tail);
    }

    // Leave a working set for graph scratch and the remaining streamed experts.
    static size_t allowance(size_t cap, size_t floor) {
        constexpr size_t headroom = size_t(2) << 30;
        return cap > floor && cap - floor > headroom ? cap - floor - headroom : 0;
    }

    void shrink(size_t allowed) {
        while (charged > allowed) {
            charged -= layers.back().charge;
            layers.pop_back();
        }
    }

    void load(const llama_model & model, ggml_backend_dev_t dev, size_t allowed, size_t max_layers) {
        const auto buft = ggml_backend_dev_buffer_type(dev);
        using host_base_fn = void * (*)(ggml_backend_buffer_t);
        const auto host_base = reinterpret_cast<host_base_fn>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_metal_buffer_get_host_base"));
        if (!host_base) { return; }
        const auto ledger = model.memory_contract->ledger;
        for (size_t il = first; il < model.hparams.n_layer() && count() < max_layers; ++il) {
            const auto & layer = model.layers[il];
            const weights original = {layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps};
            if (model.dev_layer(il) != dev || !std::all_of(original.begin(), original.end(),
                    [&](const auto * w) { return model.streamed_experts.count(w) != 0; })) { return; }
            bank b;
            b.metadata.reset(ggml_init({3 * ggml_tensor_overhead(), nullptr, true}));
            if (!b.metadata) { return; }
            for (size_t k = 0; k < original.size(); ++k) {
                b.tensors[k] = ggml_dup_tensor(b.metadata.get(), original[k]);
                ggml_set_name(b.tensors[k], original[k]->name);
            }
            const size_t size = ggml_backend_alloc_ctx_tensors_from_buft_size(b.metadata.get(), buft);
            if (!size || size > allowed - charged) { return; }
            const size_t before = ledger->snapshot(llama_memory_domain::host).used();
            {
                llama_allocation_scope admission(ledger, llama_memory_category::experts);
                b.buffer.reset(ggml_backend_alloc_ctx_tensors_from_buft(b.metadata.get(), buft));
            }
            if (!b.buffer || !host_base(b.buffer.get())) { return; }
            b.charge = ledger->snapshot(llama_memory_domain::host).used() - before;
            if (b.charge < ggml_backend_buffer_get_size(b.buffer.get())) {
                throw std::runtime_error("resident layer buffer was not fully charged");
            }
            if (b.charge > allowed - charged) { return; }
            ggml_backend_buffer_set_usage(b.buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            ggml_backend_buffer_clear(b.buffer.get(), 0);
            for (auto * w : b.tensors) {
                const auto & info = model.expert_source->tensor(w->name);
                if (!ggml_is_contiguous(w) || info.bytes != ggml_nbytes(w) || info.expert_bytes() != w->nb[2] ||
                        w->buffer != b.buffer.get()) { return; }
                for (int64_t e = 0; e < w->ne[2]; ++e) {
                    if (!model.expert_source->read_expert(w->name, e, static_cast<char *>(w->data) + e*w->nb[2], w->nb[2])) {
                        throw std::runtime_error("failed to load complete resident expert layer");
                    }
                }
            }
            layers.push_back(std::move(b));
            charged += layers.back().charge;
        }
    }
private:
    struct bank {
        ggml_context_ptr metadata;
        ggml_backend_buffer_ptr buffer;
        weights tensors{};
        size_t charge = 0;
    };
    uint32_t first;
    size_t charged = 0;
    std::vector<bank> layers;
    int tail_first = 0;
    std::array<llama_prefill_wide_weights::retained_bank, 2> tail;
};
