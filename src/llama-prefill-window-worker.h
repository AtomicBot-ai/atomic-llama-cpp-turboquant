#pragma once
#include "../ggml/src/ggml-backend-window.h"

// Opt-in only. The compute call stays on its owning thread and pauses at
// synchronized observation callbacks. No scheduler internals or cancellation
// callback semantics are used as a resumable graph API.
#include <array>
#include <bitset>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef __APPLE__
#include <pthread.h>
#endif

struct llama_prefill_window_worker {
    ggml_backend_sched_t sched = nullptr;
    ggml_cgraph * graph = nullptr;
    std::shared_ptr<llama_memory_budget> budget;
    std::mutex mutex;
    std::condition_variable cv;
    std::thread thread;
    static constexpr size_t compact_stack_size = 1024 * 1024;
    // Leave room for Darwin's page rounding and pthread metadata.
    static constexpr size_t compact_stack_charge = compact_stack_size + 64 * 1024;
#ifdef __APPLE__
    pthread_t compact_thread = nullptr;
#endif
    int layer_count = 0;
    const char * layer_output_prefix = "l_last-";
    int permit = -1;
    int reached = -1;
    bool done = false;
    bool cancelled = false;
    int routes_reached = -1;
    int routes_permit = -1;
    static constexpr size_t max_experts = 1024;
    std::bitset<max_experts> routes;
    ggml_tensor * cross_node = nullptr;
    int cross_phases = 3;
    int cross_reached = -1;
    int cross_permit = -1;
    bool cross_precomputed = false;
    ggml_status status = GGML_STATUS_FAILED;
    std::exception_ptr failure;

    void reset() {
        if (thread.joinable()) { throw std::runtime_error("window worker is still active"); }
#ifdef __APPLE__
        if (compact_thread) { throw std::runtime_error("window worker is still active"); }
#endif
        permit = reached = -1;
        done = cancelled = false;
        routes_reached = routes_permit = -1;
        routes.reset();
        cross_node = nullptr;
        cross_reached = cross_permit = -1;
        cross_precomputed = false;
        status = GGML_STATUS_FAILED;
        failure = {};
    }

    static bool observe(ggml_tensor * t, bool ask, void * opaque) {
        auto & self = *static_cast<llama_prefill_window_worker *>(opaque);
        char * suffix = nullptr;
        const bool prefix = std::strncmp(t->name, self.layer_output_prefix, std::strlen(self.layer_output_prefix)) == 0;
        const char * first_digit = t->name + std::strlen(self.layer_output_prefix);
        const long parsed_layer = prefix ? std::strtol(first_digit, &suffix, 10) : -1;
        const bool boundary = prefix && suffix != first_digit && *suffix == 0 &&
            parsed_layer >= 0 && parsed_layer < self.layer_count;
        if (ask) { return boundary; }
        if (!boundary) { return true; }
        const int layer = int(parsed_layer);
        std::unique_lock<std::mutex> lock(self.mutex);
        if (layer != self.reached + 1) { throw std::runtime_error("noncontiguous window layer boundaries"); }
        self.reached = layer;
        self.cv.notify_all();
        self.cv.wait(lock, [&] { return self.permit > layer || self.cancelled; });
        if (self.cancelled) { throw std::runtime_error("prefill window cancelled"); }
        return true;
    }

    void run(size_t stack_limit) {
        try {
#ifdef __APPLE__
            if (pthread_get_stacksize_np(pthread_self()) > stack_limit) {
                throw std::runtime_error("worker stack exceeds admitted bound");
            }
#else
            (void) stack_limit;
            throw std::runtime_error("window workers require Metal on macOS");
#endif
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return permit >= 0 || cancelled; });
                if (cancelled) { throw std::runtime_error("prefill window cancelled"); }
            }
            // Backend buffers created lazily on this thread must be charged.
            llama_allocation_scope allocations(budget, llama_memory_category::scratch);
            status = ggml_backend_sched_graph_compute(sched, graph);
        } catch (...) { failure = std::current_exception(); }
        std::lock_guard<std::mutex> lock(mutex);
        done = true;
        cv.notify_all();
    }

    void launch(bool compact = false) {
#ifdef __APPLE__
        if (compact) {
            pthread_attr_t attrs;
            if (pthread_attr_init(&attrs)) { throw std::runtime_error("cannot initialize window worker attributes"); }
            pthread_t worker = nullptr;
            int error = pthread_attr_setstacksize(&attrs, compact_stack_size);
            if (!error) {
                error = pthread_create(&worker, &attrs, [](void * arg) -> void * {
                    static_cast<llama_prefill_window_worker *>(arg)->run(compact_stack_charge);
                    return nullptr;
                }, this);
            }
            pthread_attr_destroy(&attrs);
            if (error) { throw std::runtime_error("cannot start compact window worker"); }
            compact_thread = worker;
            return;
        }
#else
        (void) compact;
#endif
        thread = std::thread([this] { run(8 * 1024 * 1024); });
    }

    void advance(int layer) {
        std::unique_lock<std::mutex> lock(mutex);
        permit = layer;
        cv.notify_all();
        cv.wait(lock, [&] { return reached >= layer || done; });
        if (failure) { std::rethrow_exception(failure); }
        if (done && reached < layer && layer < layer_count) { throw std::runtime_error("graph ended before its layer boundary"); }
    }

    // Called from the existing CPU projection callback before scratch is in use.
    void observe_routes(int layer, const ggml_tensor * ids, bool park, std::FILE * trace) {
        std::unique_lock<std::mutex> lock(mutex);
        if (cancelled) { throw std::runtime_error("prefill window cancelled"); }
        if (routes_reached == layer) { return; }
        if (layer <= routes_reached || ids->type != GGML_TYPE_I32) {
            throw std::runtime_error("invalid window route boundary");
        }
        routes.reset();
        for (int64_t row = 0; row < ids->ne[1]; ++row) {
            for (int64_t route = 0; route < ids->ne[0]; ++route) {
                int32_t id;
                std::memcpy(&id, static_cast<const char *>(ids->data) + row * ids->nb[1] + route * ids->nb[0], sizeof(id));
                if (id < 0 || size_t(id) >= routes.size()) { throw std::runtime_error("invalid window expert ID"); }
                routes.set(size_t(id));
            }
        }
        if (trace) {
            std::fprintf(trace, "routes %d %lld", layer, static_cast<long long>(ids->ne[1]));
            for (size_t id = 0; id < routes.size(); ++id) {
                if (routes[id]) { std::fprintf(trace, " %zu", id); }
            }
            std::fputc('\n', trace);
        }
        routes_reached = layer;
        cv.notify_all();
        if (park) { cv.wait(lock, [&] { return routes_permit >= layer || cancelled; }); }
        if (cancelled) { throw std::runtime_error("prefill window cancelled"); }
    }

    void advance_routes(int layer) {
        std::unique_lock<std::mutex> lock(mutex);
        permit = layer;
        cv.notify_all();
        cv.wait(lock, [&] { return routes_reached >= layer || reached >= layer || done; });
        if (failure) { std::rethrow_exception(failure); }
        if (routes_reached != layer) { throw std::runtime_error("window graph has no route boundary"); }
    }

    void advance_experts(int layer) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            routes_permit = layer;
        }
        advance(layer);
    }

    bool observe_cross(int layer, ggml_tensor * node) {
        std::unique_lock<std::mutex> lock(mutex);
        const int first = cross_phases * layer;
        const int step = cross_reached < first ? first : cross_reached + 1;
        if (step / cross_phases != layer) { throw std::runtime_error("unexpected cross-window projection order"); }
        cross_node = node;
        cross_reached = step;
        cv.notify_all();
        cv.wait(lock, [&] { return cross_permit >= step || cancelled; });
        if (cancelled) { throw std::runtime_error("prefill window cancelled"); }
        return cross_precomputed;
    }

    void begin_cross_layer(int layer) {
        std::unique_lock<std::mutex> lock(mutex);
        const int first = cross_phases * layer;
        permit = layer;
        cv.notify_all();
        cv.wait(lock, [&] { return cross_reached >= first || reached >= layer || done; });
        if (failure) { std::rethrow_exception(failure); }
        if (cross_reached != first) { throw std::runtime_error("missing first cross-window projection"); }
    }

    void advance_cross(int layer, int phase, bool computed) {
        std::unique_lock<std::mutex> lock(mutex);
        const int step = cross_phases * layer + phase;
        cross_precomputed = computed;
        cross_permit = step;
        cv.notify_all();
        cv.wait(lock, [&] { return cross_reached > step || reached >= layer || done; });
        if (failure) { std::rethrow_exception(failure); }
        if (phase + 1 == cross_phases ? reached != layer : cross_reached != step + 1) {
            throw std::runtime_error("missing cross-window projection boundary");
        }
    }

    void cancel() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        cancelled = true;
        cv.notify_all();
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            permit = routes_permit = cross_permit = 10000;
            cv.notify_all();
        }
        if (thread.joinable()) { thread.join(); }
#ifdef __APPLE__
        if (compact_thread) {
            pthread_join(compact_thread, nullptr);
            compact_thread = nullptr;
        }
#endif
    }
    ~llama_prefill_window_worker() { finish(); }
};
