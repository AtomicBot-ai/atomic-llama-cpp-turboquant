#pragma once

#include "llama-expert-read-pool.h"
#include "../ggml/src/ggml-metal/ggml-metal-route-publication.h"
#include <algorithm>
#include <exception>

// One bounded channel, serviced by the existing charged demand pool. No new
// thread or dispatch queue. GPU completion is signaled before host payload reads.
class llama_expert_publication {
public:
    using consume_fn = void (*)(void *, const int32_t *, size_t);
private:
    const ggml_metal_route_api & api;
    early_io::pool & pool;
    void * channel;
    early_io::pool::task task;
    std::atomic<bool> cancelled{false}, armed{false};
    uint64_t sequence = 0, generation = 0;
    size_t count = 0;
    bool measure = false;
    void * context = nullptr;
    consume_fn consume = nullptr;
    std::exception_ptr error;
    uint64_t worker_ns = 0, published_ns = 0, routes_ns = 0, residency_ns = 0, prepared_ns = 0;

    static void work(void * user) noexcept {
        auto & self = *static_cast<llama_expert_publication *>(user);
        try {
            self.worker_ns = self.measure ? early_io::now() : 0;
            std::array<int32_t, GGML_METAL_ROUTE_MAX_IDS> ids{};
            while (!self.cancelled.load(std::memory_order_acquire)) {
                const int status = self.api.read(self.channel, self.sequence, ids.data(), self.count);
                if (status < 0) throw std::runtime_error("stale route publication event");
                if (!status) continue;
                self.published_ns = self.measure ? early_io::now() : 0;
                if (!self.cancelled.load(std::memory_order_acquire)) self.consume(self.context, ids.data(), self.count);
                self.prepared_ns = self.measure ? early_io::now() : 0;
                return;
            }
        } catch (...) { self.error = std::current_exception(); }
    }
public:
    llama_expert_publication(const ggml_metal_route_api & api, ggml_backend_t backend, early_io::pool & pool) :
        api(api), pool(pool), channel(api.create(backend)) {
        if (!channel) throw std::runtime_error("route publication allocation failed");
    }
    ~llama_expert_publication() { cancel(); api.destroy(channel); }
    llama_expert_publication(const llama_expert_publication &) = delete;
    void * handle() const { return channel; }
    bool attach(ggml_tensor * node, void (*arm)(void *, uint64_t, uint64_t, size_t), void * user, uint64_t stamp) {
        return api.attach(node, channel, arm, user, stamp);
    }
    void start(uint64_t seq, uint64_t stamp, size_t n, consume_fn fn, void * user, bool timing = false) noexcept {
        // The scheduler must consume one CPU MoE split before publishing another.
        if (armed.exchange(true)) std::terminate();
        measure = timing;
        sequence = seq; generation = stamp; count = n; consume = fn; context = user;
        error = {}; worker_ns = published_ns = routes_ns = residency_ns = prepared_ns = 0;
        cancelled.store(false, std::memory_order_release);
        task = {}; task.fn = work; task.context = this;
        pool.submit(&task, 1, true);
    }
    bool pending() const { return armed.load(); }
    bool abandoned() const { return cancelled.load(std::memory_order_acquire); }
    uint64_t stamp() const { return generation; }
    void abandon() noexcept {
        if (!armed.load()) return;
        cancelled.store(true, std::memory_order_release);
        pool.cancel(task);
        if (pool.complete(task)) armed.store(false);
    }
    // The staged path can fall back immediately. Only completed tasks transfer
    // ownership; an abandoned task keeps its storage until its worker exits.
    bool take_if_ready() noexcept {
        if (!armed.load()) return false;
        if (!pool.complete(task)) {
            abandon();
            return false;
        }
        armed.store(false);
        return !abandoned() && !error;
    }
    // Synchronizes all worker writes before the graph owner touches the store.
    void join() {
        if (!armed.load()) return;
        pool.wait(task);
        armed.store(false);
        if (error) std::rethrow_exception(error);
    }
    void cancel() noexcept {
        cancelled.store(true, std::memory_order_release);
        if (armed.load()) { pool.cancel(task); pool.wait(task); armed.store(false); }
    }
    uint64_t publication_time() const { return published_ns; }
    uint64_t preparation_time() const { return prepared_ns; }
    void mark_routes() { if (measure) routes_ns = early_io::now(); }
    void mark_residency() { if (measure) residency_ns = early_io::now(); }
    std::array<uint64_t, 6> blocked_phases(uint64_t entered, uint64_t left) const {
        std::array<uint64_t, 6> result{};
        const uint64_t end[] = {worker_ns, published_ns, routes_ns, residency_ns, prepared_ns, left};
        for (size_t i = 0; i < result.size(); ++i) {
            if (end[i] > entered) {
                const uint64_t until = std::min(end[i], left);
                result[i] = until - entered;
                entered = until;
            }
        }
        return result;
    }
};
