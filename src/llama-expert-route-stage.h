#pragma once

#include "llama-expert-publication.h"
#include "llama-expert-store.h"
#include <cstring>
#include <memory>

// Read exact routes into bounded, separately charged storage. The publication
// worker never admits or pins store entries. A demanded slice can notify its
// demand batch directly, without taking another reader slot.
class llama_expert_route_stage {
public:
    using completion_fn = void (*)(void *, const void *) noexcept;

private:
    static constexpr size_t experts = 4;
    static constexpr size_t capacity = 3 * experts;
    struct slice {
        llama_expert_route_stage * owner = nullptr;
        llama_expert_source::read_request read{};
        bool finished = false, valid = false, used = false;
        completion_fn notify = nullptr;
        void * user = nullptr;
    };
    std::shared_ptr<llama_expert_source> source;
    llama_expert_store & store;
    early_io::pool & pool;
    const size_t stride;
    const bool deduplicate;
    std::unique_ptr<uint8_t[]> data;
    std::mutex mutex;
    std::array<slice, capacity> slices{};
    std::array<early_io::pool::task, capacity> tasks{};
    size_t count = 0;
    bool accepting = false, available = false;

    static void read_slice(void * user) noexcept {
        auto & slot = *static_cast<slice *>(user);
        bool valid = false;
        try {
            slot.owner->source->read_experts(&slot.read, 1, true, true);
            valid = true;
        } catch (...) { /* demand retries through the validated store path */ }
        completion_fn notify;
        void * context;
        {
            std::lock_guard<std::mutex> lock(slot.owner->mutex);
            slot.finished = true;
            slot.valid = valid;
            notify = slot.notify;
            context = slot.user;
        }
        if (notify) notify(context, valid ? slot.read.destination : nullptr);
    }

    size_t find(const llama_expert_source::read_request & request) const {
        for (size_t i = 0; available && i < count; ++i) {
            const auto & slot = slices[i];
            if (!slot.used && slot.read.tensor == request.tensor && slot.read.expert == request.expert) return i;
        }
        return count;
    }

public:
    static size_t memory_bytes(size_t stride) { return sizeof(llama_expert_route_stage) + capacity * stride; }
    llama_expert_route_stage(std::shared_ptr<llama_expert_source> source, llama_expert_store & store,
            early_io::pool & pool, size_t stride, bool deduplicate) :
        source(std::move(source)), store(store), pool(pool), stride(stride), deduplicate(deduplicate),
        data(new uint8_t[capacity * stride]) {}
    ~llama_expert_route_stage() {
        stop();
        for (size_t i = 0; i < count; ++i) pool.wait(tasks[i]);
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex);
        accepting = false;
        available = false;
        for (size_t i = 0; i < count; ++i) pool.cancel(tasks[i]);
    }

    void begin() {
        std::lock_guard<std::mutex> lock(mutex);
        accepting = true;
        available = false;
    }

    bool seal() {
        std::lock_guard<std::mutex> lock(mutex);
        accepting = false;
        return available;
    }

    void prepare(const std::array<const char *, 3> & names, const int64_t * routes, size_t n,
            const llama_expert_publication & publication) {
        std::array<const llama_expert_tensor_info *, 3> weights{};
        for (size_t i = 0; i < names.size(); ++i) {
            weights[i] = &source->tensor(names[i]);
            if (!weights[i]->can_read_experts() || weights[i]->expert_bytes() > stride) return;
        }
        // Gather outside the handoff lock. The CPU can seal the stage while
        // residency is being inspected, in which case nothing is submitted.
        std::array<llama_expert_source::read_request, capacity> requests{};
        size_t planned = 0;
        size_t misses = 0;
        for (size_t i = 0; i < n && misses < experts && !publication.abandoned(); ++i) {
            const size_t before = planned;
            for (const auto * weight : weights) {
                if (store.contains(weight->name, routes[i])) continue;
                requests[planned] = {weight, routes[i], data.get() + planned * stride};
                ++planned;
            }
            misses += planned != before;
        }
        std::lock_guard<std::mutex> lock(mutex);
        if (!accepting || !planned) return;
        for (size_t i = 0; i < count; ++i) if (!pool.complete(tasks[i])) return;
        count = planned;
        for (size_t i = 0; i < count; ++i) {
            slices[i] = {this, requests[i]};
            tasks[i] = {};
            tasks[i].fn = read_slice;
            tasks[i].context = &slices[i];
        }
        pool.submit(tasks.data(), count, false);
        available = true;
    }

    size_t consume_ready(const llama_expert_source::read_request & request, llama_expert_profile_data * profile) {
        std::unique_lock<std::mutex> lock(mutex);
        const size_t i = find(request);
        if (i == count) return 0;
        auto & slot = slices[i];
        const size_t bytes = request.tensor->expert_bytes();
        if (!slot.finished) {
            if (!deduplicate && !pool.cancel(tasks[i]) && profile) {
                ++profile->route_stage_duplicate_reads;
                profile->route_stage_duplicate_bytes += bytes;
            }
            return 0;
        }
        if (!slot.valid) return 0;
        slot.used = true;
        const auto * data = slot.read.destination;
        lock.unlock();
        // The graph owner keeps the sealed stage alive through demand.
        std::memcpy(request.destination, data, bytes);
        return bytes;
    }

    early_io::pool::task * adopt(const llama_expert_source::read_request & request, completion_fn notify,
            void * context, llama_expert_profile_data * profile) {
        if (!deduplicate) return nullptr;
        std::unique_lock<std::mutex> lock(mutex);
        const size_t i = find(request);
        if (i == count) return nullptr;
        auto & slot = slices[i];
        if (slot.finished && !slot.valid) return nullptr;
        slot.used = true;
        if (profile) ++profile->route_stage_joined;
        auto & task = tasks[i];
        if (slot.finished) {
            const void * bytes = slot.read.destination;
            lock.unlock();
            notify(context, bytes);
        } else {
            slot.notify = notify;
            slot.user = context;
            if (pool.promote(task) && profile) ++profile->route_stage_promoted;
        }
        return &task;
    }
};
