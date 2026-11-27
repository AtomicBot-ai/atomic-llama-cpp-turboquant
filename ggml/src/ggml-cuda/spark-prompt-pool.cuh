// Spark Inkling prefill uses one CUDA stream. Reuse admitted scratch within a
// scheduler split. The CUDA buffers retain their tickets until the split ends.
struct ggml_cuda_spark_prompt_pool final : ggml_cuda_pool_observed {
    using ggml_cuda_pool_observed::ggml_cuda_pool_observed;

    struct slot { void * ptr = nullptr; size_t bytes = 0; };
    slot idle[8] {};
    size_t retained = 0;

    void * alloc(size_t size, size_t * actual) override {
        slot * best = nullptr;
        for (auto & block : idle) {
            if (block.ptr && block.bytes >= size && (!best || block.bytes < best->bytes)) { best = &block; }
        }
        if (best) {
            const auto block = *best;
            *best = {};
            retained -= block.bytes;
            *actual = block.bytes;
            return block.ptr;
        }
        try {
            return ggml_cuda_pool_observed::alloc(size, actual);
        } catch (const std::bad_alloc &) {
            if (!retained) { throw; }
            trim();
            return ggml_cuda_pool_observed::alloc(size, actual);
        }
    }

    void free(void * ptr, size_t size) override {
        if (size <= (size_t(64) << 20) - retained) {
            for (auto & block : idle) {
                if (!block.ptr) {
                    block = {ptr, size};
                    retained += size;
                    return;
                }
            }
        }
        ggml_cuda_pool_observed::free(ptr, size);
    }

    void trim() {
        for (auto & block : idle) {
            if (block.ptr) { ggml_cuda_pool_observed::free(block.ptr, block.bytes); }
            block = {};
        }
        retained = 0;
    }

    ~ggml_cuda_spark_prompt_pool() override { trim(); }
};

static ggml_status ggml_backend_cuda_spark_prompt(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!ggml_backend_has_alloc_observer() || ctx->curr_stream_no != 0 ||
            !ctx->stream_context().concurrent_events.empty()) {
        return ggml_backend_cuda_graph_compute(backend, graph);
    }
    std::unique_ptr<ggml_cuda_pool> temporary(new (std::nothrow) ggml_cuda_spark_prompt_pool(ctx->device));
    if (!temporary) { return ggml_backend_cuda_graph_compute(backend, graph); }
    auto & active = ctx->pools[ctx->device][0];
    active.swap(temporary);
    struct restore_pool {
        std::unique_ptr<ggml_cuda_pool> & active;
        std::unique_ptr<ggml_cuda_pool> & previous;
        ~restore_pool() { active.swap(previous); }
    } restore{active, temporary};
    return ggml_backend_cuda_graph_compute(backend, graph);
}

static void ggml_backend_cuda_set_spark_prompt_pool(ggml_backend_t backend, bool enabled) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (enabled && ctx->observed_pool && backend->iface.graph_compute == ggml_backend_cuda_graph_compute) {
        backend->iface.graph_compute = ggml_backend_cuda_spark_prompt;
    } else if (!enabled && backend->iface.graph_compute == ggml_backend_cuda_spark_prompt) {
        backend->iface.graph_compute = ggml_backend_cuda_graph_compute;
    }
}
