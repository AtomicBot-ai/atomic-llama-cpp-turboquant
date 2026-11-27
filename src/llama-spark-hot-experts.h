#pragma once

#include "llama-expert-source.h"
#include "llama-memory-contract.h"
#include "models/models.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <cstdio>
#include <memory>
#include <numeric>

// Immutable original GGUF slices, populated from an already-read prompt matrix.
// A fixed slice count per streamed layer gives each projection a bounded region.
struct llama_spark_hot_experts {
    static constexpr size_t experts = 128;
    struct projection {
        const ggml_tensor * tensor = nullptr;
        uint8_t * data = nullptr;
        size_t stride = 0;
        size_t capacity = 0;
        std::array<int16_t, experts> slots;
        bool ready = false;
        ggml_backend_buffer_t mapped = nullptr;
        size_t offset = 0;
        size_t cold_slot = 0;
        static constexpr size_t cold_capacity = 32;
        bool reported = false;

        // All tables end at the same transient tail, reached in units of this
        // projection's stride. Only published indices are consumed by CUDA.
        bool plan_direct(const std::array<bool, experts> & selected, std::array<int32_t, experts> & indices) const {
            if (!mapped || !ready) { return false; }
            size_t misses = 0;
            for (size_t id = 0; id < experts; ++id) {
                if (selected[id] && slots[id] < 0) { ++misses; }
            }
            if (misses > cold_capacity) { return false; }
            size_t next = cold_slot;
            for (size_t id = 0; id < experts; ++id) {
                if (selected[id]) { indices[id] = slots[id] >= 0 ? slots[id] : int32_t(next++); }
            }
            return true;
        }

        size_t populate_tail(llama_expert_source & source, const std::string & name,
                const std::array<bool, experts> & selected) {
            std::array<bool, experts> cold{};
            size_t misses = 0;
            for (size_t id = 0; id < experts; ++id) {
                cold[id] = selected[id] && slots[id] < 0;
                misses += cold[id];
            }
            auto * tail = data + cold_slot * stride;
            if (misses && !source.read_spark_selected_experts(name, cold, tail, misses * stride)) {
                for (size_t id = 0; id < experts; ++id) {
                    if (!cold[id]) { continue; }
                    source.read_expert(name, id, tail, stride);
                    tail += stride;
                }
            }
            if (!reported) {
                std::fprintf(stderr, "Spark direct expert view: %s, %zu resident slots, %zu cold IDs in first window\n",
                        name.c_str(), capacity, misses);
                reported = true;
            }
            return misses * stride;
        }

        void populate(const std::array<unsigned, experts> & frequency, const uint8_t * matrix) {
            std::array<int, experts> order;
            std::iota(order.begin(), order.end(), 0);
            std::partial_sort(order.begin(), order.begin() + capacity, order.end(), [&](int a, int b) {
                return frequency[a] != frequency[b] ? frequency[a] > frequency[b] : a < b;
            });
            slots.fill(-1);
            for (size_t i = 0; i < capacity && frequency[order[i]]; ++i) {
                const int id = order[i];
                std::memcpy(data + i * stride, matrix + id * stride, stride);
                slots[id] = int16_t(i);
            }
            ready = true;
        }

        bool fill(llama_expert_source & source, const std::string & name, const std::array<bool, experts> & selected,
                const std::array<int32_t, experts> & destination, uint8_t * matrix, size_t size, bool compact,
                uint64_t & read_bytes) const {
            std::array<bool, experts> cold{};
            size_t misses = 0;
            for (size_t i = 0; i < experts; ++i) {
                if (!selected[i]) {
                    if (!compact) { std::memset(matrix + i * stride, 0, stride); }
                } else if (slots[i] < 0) {
                    cold[i] = true;
                    ++misses;
                } else {
                    std::memcpy(matrix + size_t(destination[i]) * stride, data + size_t(slots[i]) * stride, stride);
                }
            }
            if (misses && !source.read_spark_selected_experts(name, cold, matrix, size, &destination)) { return false; }
            read_bytes += misses * stride;
            return true;
        }
    };
    llama_memory_budget::ticket charge;
    std::unique_ptr<uint8_t[]> storage;
    ggml_backend_buffer_ptr mapped;
    size_t charged = 0;
    std::array<projection, 48> tables;
    size_t count = 0;

    projection * find(const std::string & name) {
        for (size_t i = 0; i < count; ++i) {
            if (name == tables[i].tensor->name) { return &tables[i]; }
        }
        return nullptr;
    }

    static std::unique_ptr<llama_spark_hot_experts> create(const llama_model_inkling2_loop & model,
            size_t full_banks, size_t limit, ggml_backend_t direct_backend = nullptr) {
        if (limit <= sizeof(llama_spark_hot_experts) || model.n_phys != 24) { return nullptr; }
        size_t row_bytes = 0;
        std::array<const ggml_tensor *, 48> tensors{};
        size_t count = 0;
        for (size_t i = full_banks; i < size_t(model.n_phys); ++i) {
            const auto & layer = model.layers[(model.core_start + i) % model.n_phys];
            for (auto * w : {layer.ffn_gate_up_exps, layer.ffn_down_exps}) {
                if (!w || !model.streamed_experts.count(w) || w->ne[2] != experts || !ggml_is_contiguous(w) ||
                        w->type != GGML_TYPE_NVFP4 || w->nb[2] != model.expert_source->tensor(w->name).expert_bytes() ||
                        w->nb[2] > limit - row_bytes) { return nullptr; }
                tensors[count++] = w;
                row_bytes += w->nb[2];
            }
        }
        if (!row_bytes) { return nullptr; }
        size_t tail_bytes = 0;
        if (direct_backend) {
            size_t largest = 0;
            for (size_t i = 0; i < count; ++i) { largest = std::max(largest, tensors[i]->nb[2]); }
            for (size_t i = 0; i < count; ++i) {
                if ((2 * tensors[i]->nb[2]) % largest) { return nullptr; }
            }
            // Leave one extra stride for the backend's quantized-matrix padding.
            tail_bytes = (projection::cold_capacity + 1) * largest;
            if (tail_bytes >= limit - sizeof(llama_spark_hot_experts)) { return nullptr; }
        }
        size_t capacity = std::min(experts, (limit - sizeof(llama_spark_hot_experts) - tail_bytes) / row_bytes);
        if (direct_backend) { capacity -= capacity % 2; }
        if (!capacity) { return nullptr; }
        const size_t payload = capacity * row_bytes + tail_bytes;
        auto ticket = model.memory_contract->ledger->reserve(llama_memory_domain::host,
                llama_memory_category::experts, sizeof(llama_spark_hot_experts) + (direct_backend ? 0 : payload));
        if (!ticket) { return nullptr; }
        auto cache = std::unique_ptr<llama_spark_hot_experts>(new (std::nothrow) llama_spark_hot_experts);
        if (!cache) { return nullptr; }
        ticket.commit();
        cache->charge = std::move(ticket);
        cache->charged = cache->charge.size();
        cache->count = count;
        uint8_t * base = nullptr;
        if (direct_backend) {
            using allocate_fn = ggml_backend_buffer_t (*)(ggml_backend_t, size_t, void **);
            const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(direct_backend));
            const auto allocate = reinterpret_cast<allocate_fn>(ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_spark_prompt_weights"));
            if (!allocate) { return nullptr; }
            const auto & contract = *model.memory_contract;
            const auto used = [&]() {
                return contract.ledger->snapshot(llama_memory_domain::host).categories[size_t(llama_memory_category::experts)].total();
            };
            const size_t before = used();
            void * host = nullptr;
            llama_allocation_scope scope(contract.ledger, llama_memory_category::experts, {}, contract.discrete);
            cache->mapped.reset(allocate(direct_backend, payload, &host));
            if (!cache->mapped || !host) { return nullptr; }
            const size_t after = used();
            if (after < before || after - before < payload) { throw std::runtime_error("unaccounted Spark hot expert buffer"); }
            cache->charged += after - before;
            if (cache->charged > limit) { return nullptr; }
            base = static_cast<uint8_t *>(host);
            std::memset(base, 0, payload);
        } else {
            cache->storage.reset(new (std::nothrow) uint8_t[payload]);
            if (!cache->storage) { return nullptr; }
            base = cache->storage.get();
        }
        auto * next = base;
        for (size_t i = 0; i < count; ++i) {
            auto & p = cache->tables[i];
            p.tensor = tensors[i];
            p.stride = p.tensor->nb[2];
            p.capacity = capacity;
            p.data = next;
            p.mapped = cache->mapped.get();
            p.offset = size_t(next - base);
            p.cold_slot = (capacity * row_bytes - p.offset) / p.stride;
            next += capacity * p.stride;
        }
        LLAMA_LOG_INFO("Spark hot experts: %zu full banks, %zu slots per streamed projection, %.1f MiB\n",
                full_banks, capacity, double(payload) / (1 << 20));
        return cache;
    }
};
