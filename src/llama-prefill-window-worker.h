#pragma once
#include "../ggml/src/ggml-backend-window.h"

// Opt-in only. The compute call stays on its owning thread and pauses at
// synchronized observation callbacks. No scheduler internals or cancellation
// callback semantics are used as a resumable graph API.
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
    int permit = -1;
    int reached = -1;
    bool done = false;
    ggml_status status = GGML_STATUS_FAILED;
    std::exception_ptr failure;

    void reset() {
        if (thread.joinable()) { throw std::runtime_error("window worker is still active"); }
        permit = reached = -1;
        done = false;
        status = GGML_STATUS_FAILED;
        failure = {};
    }

    static bool observe(ggml_tensor * t, bool ask, void * opaque) {
        auto & self = *static_cast<llama_prefill_window_worker *>(opaque);
        char * suffix = nullptr;
        const bool prefix = std::strncmp(t->name, "l_last-", 7) == 0;
        const long parsed_layer = prefix ? std::strtol(t->name + 7, &suffix, 10) : -1;
        const bool boundary = prefix && suffix != t->name + 7 && *suffix == 0 && parsed_layer >= 0 && parsed_layer < 48;
        if (ask) { return boundary; }
        if (!boundary) { return true; }
        const int layer = int(parsed_layer);
        std::unique_lock<std::mutex> lock(self.mutex);
        if (layer != self.reached + 1) { throw std::runtime_error("noncontiguous window layer boundaries"); }
        self.reached = layer;
        self.cv.notify_all();
        self.cv.wait(lock, [&] { return self.permit > layer; });
        return true;
    }

    void launch() {
        thread = std::thread([this] {
            try {
#ifdef __APPLE__
                if (pthread_get_stacksize_np(pthread_self()) > 8 * 1024 * 1024) {
                    throw std::runtime_error("worker stack exceeds admitted bound");
                }
#else
                throw std::runtime_error("window workers require Metal on macOS");
#endif
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    cv.wait(lock, [&] { return permit >= 0; });
                }
                // Backend buffers created lazily on this thread must be charged.
                llama_allocation_scope allocations(budget, llama_memory_category::scratch);
                status = ggml_backend_sched_graph_compute(sched, graph);
            } catch (...) { failure = std::current_exception(); }
            std::lock_guard<std::mutex> lock(mutex);
            done = true;
            cv.notify_all();
        });
    }

    void advance(int layer) {
        std::unique_lock<std::mutex> lock(mutex);
        permit = layer;
        cv.notify_all();
        cv.wait(lock, [&] { return reached >= layer || done; });
        if (failure) { std::rethrow_exception(failure); }
        if (done && reached < layer && layer < 48) { throw std::runtime_error("graph ended before its layer boundary"); }
    }

    void finish() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            permit = 1000;
            cv.notify_all();
        }
        if (thread.joinable()) { thread.join(); }
    }
    ~llama_prefill_window_worker() { finish(); }
};
