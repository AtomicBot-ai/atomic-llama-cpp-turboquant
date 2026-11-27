// Included by ggml-cuda.cu after the ordinary CUDA backend is defined.
// CUDA graph replay for qualified Inkling decode or short DSpark verification.
struct ggml_cuda_spark_decode_pool final : ggml_cuda_pool_observed {
    explicit ggml_cuda_spark_decode_pool(int device) : ggml_cuda_pool_observed(device) {}

    // The arena belongs to the observed buffer allocator. Its ticket stays live across
    // graph launches. A larger temporary during capture falls back to ordinary dispatch.
    static constexpr size_t arena_bytes = 8u << 20;
    ggml_backend_buffer_t arena = nullptr;
    static constexpr int max_chunks = 8;
    cudaGraphExec_t chunks[max_chunks] = {};
    int n_chunks = 0;
    uint64_t uid = 0;
    uint64_t parent_uid = 0, largest_uid = 0;
    int largest_nodes = 0;
    bool learning_splits = false;
    bool eager_only = false;
    bool capturing = false;
    uint64_t replay_count = 0;
    size_t cursor = 0, live = 0;

    void * alloc(size_t size, size_t * actual) override {
        if (!capturing) { return ggml_cuda_pool_observed::alloc(size, actual); }
        constexpr size_t alignment = 256;
        size_t offset = (cursor + alignment - 1) & ~(alignment - 1);
        if (size > arena_bytes - offset) { throw std::bad_alloc(); }
        cursor = offset + size;
        ++live;
        *actual = size;
        return static_cast<char *>(ggml_backend_buffer_get_base(arena)) + offset;
    }

    void free(void * ptr, size_t size) override {
        if (!capturing) { ggml_cuda_pool_observed::free(ptr, size); return; }
        GGML_ASSERT(live > 0);
        if (--live == 0) { cursor = 0; }
    }

    void discard() {
        for (int i = 0; i < n_chunks; ++i) {
            CUDA_CHECK(cudaGraphExecDestroy(chunks[i]));
        }
        n_chunks = 0;
        if (arena) {
            ggml_backend_buffer_free(arena);
            if (ggml_cuda_graph_debug_enabled()) {
                GGML_LOG_INFO("Spark CUDA graph released uid=%llu scratch=%zu replays=%llu\n",
                        (unsigned long long) uid, arena_bytes, (unsigned long long) replay_count);
            }
            arena = nullptr;
        }
        eager_only = false;
        replay_count = 0;
    }

    ~ggml_cuda_spark_decode_pool() override { discard(); }
};

static ggml_status ggml_backend_cuda_spark_decode(ggml_backend_t backend, ggml_cgraph * graph) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    auto & pool = static_cast<ggml_cuda_spark_decode_pool &>(*ctx->pools[ctx->device][0]);
    ggml_cuda_set_device(ctx->device);
    if (!ggml_backend_has_alloc_observer()) { return GGML_STATUS_ALLOC_FAILED; }

    // At partial residency the scheduler alternates CPU callbacks and CUDA splits.
    // Observe one pass, then reuse just the largest compatible CUDA split. Other
    // splits still run (with their usual copies) and cannot evict its capture.
    if (pool.parent_uid) {
        if (pool.learning_splits && graph->uid && graph->n_nodes > pool.largest_nodes &&
                ggml_cuda_graph_check_compability(graph)) {
            pool.largest_nodes = graph->n_nodes;
            pool.largest_uid = graph->uid;
        }
        if (pool.learning_splits || !graph->uid || graph->uid != pool.largest_uid) {
            return ggml_backend_cuda_graph_compute(backend, graph);
        }
    }

    // The scheduler assigns a fresh UID to each split when it binds/reallocates a graph.
    // Never replay across a new UID, even when the scheduler recycles the tensor addresses.
    if (pool.uid != graph->uid) {
        pool.discard();
        pool.uid = graph->uid;
        pool.eager_only = !ggml_cuda_graph_check_compability(graph);
        // Run once before capture so lazy kernel and cuBLAS initialization can complete.
        if (!ggml_cuda_env_flag("LLAMA_EXPERIMENTAL_SPARK_VERIFY")) {
            return ggml_backend_cuda_graph_compute(backend, graph);
        }
        // The prefill has initialized the CUDA backend. Record the first
        // verification without an eager submission. Capture failure below
        // falls back before any graph node has executed.

    }
    if (!graph->uid || pool.eager_only) { return ggml_backend_cuda_graph_compute(backend, graph); }
    if (pool.n_chunks) {
        for (int i = 0; i < pool.n_chunks; ++i) {
            CUDA_CHECK(cudaGraphLaunch(pool.chunks[i], ctx->stream()));
        }
        if (++pool.replay_count == 1 && ggml_cuda_graph_debug_enabled()) {
            GGML_LOG_INFO("Spark CUDA graph replay uid=%llu\n", (unsigned long long) pool.uid);
        }
        return GGML_STATUS_SUCCESS;
    }

    pool.arena = ggml_backend_buft_alloc_buffer(ggml_backend_cuda_buffer_type(ctx->device), pool.arena_bytes);
    cudaGraph_t recorded = nullptr;
    cudaError_t capture_status = cudaSuccess;
    // The ordinary backend waits on this counter before destroying cuBLAS handles.
    // Keep other contexts from destroying a handle throughout this capture as well.
    struct capture_guard {
        capture_guard() {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }
        ~capture_guard() {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        }
    };
    // Start executing completed chunks while the host records the next one.
    // At a chunk boundary all temporary allocations have left their C++ scope.
    // Reuse the same arena in stream order while retaining its budget ticket.
    int first = 0;
    if (pool.arena) {
        capture_guard guard;
        const bool split = ggml_cuda_env_flag("LLAMA_EXPERIMENTAL_SPARK_VERIFY") && graph->n_nodes > 1024;
        const int stride = split ? (graph->n_nodes + pool.max_chunks - 1) / pool.max_chunks : graph->n_nodes;
        while (first < graph->n_nodes) {
            const int end = std::min(first + stride, graph->n_nodes);
            auto chunk = ggml_graph_view(graph, first, end);
            capture_status = cudaStreamBeginCapture(ctx->stream(), cudaStreamCaptureModeRelaxed);
            if (capture_status != cudaSuccess) { break; }
            bool fits = true;
            pool.capturing = true;
            try {
                ggml_cuda_graph_evaluate_and_capture(ctx, &chunk, false, false, nullptr);
            } catch (const std::bad_alloc &) {
                fits = false;
            }
            pool.capturing = false;
            GGML_ASSERT(pool.live == 0);
            capture_status = cudaStreamEndCapture(ctx->stream(), &recorded);
            if (!fits || capture_status != cudaSuccess || !recorded) { break; }
            auto & exec = pool.chunks[pool.n_chunks];
            capture_status = cudaGraphInstantiate(&exec, recorded, nullptr, nullptr, 0);
            if (capture_status != cudaSuccess) { break; }
            ++pool.n_chunks;
            CUDA_CHECK(cudaGraphDestroy(recorded));
            recorded = nullptr;
            CUDA_CHECK(cudaGraphLaunch(exec, ctx->stream()));
            first = end;
        }
        if (first == graph->n_nodes) {
            if (ggml_cuda_graph_debug_enabled()) {
                GGML_LOG_INFO("Spark CUDA graph captured uid=%llu nodes=%d chunks=%d charged scratch=%zu\n",
                        (unsigned long long) pool.uid, graph->n_nodes, pool.n_chunks, pool.arena_bytes);
            }
            return GGML_STATUS_SUCCESS;
        }
    }
    if (recorded) { CUDA_CHECK(cudaGraphDestroy(recorded)); }
    pool.discard();
    if (capture_status != cudaSuccess) {
        if (capture_status != cudaErrorMemoryAllocation && capture_status != cudaErrorNotSupported &&
                capture_status != cudaErrorStreamCaptureInvalidated && capture_status != cudaErrorStreamCaptureUnsupported) {
            GGML_LOG_ERROR("Spark CUDA graph capture failed: %s\n", cudaGetErrorString(capture_status));
            return GGML_STATUS_FAILED;
        }
        (void) cudaGetLastError();
    }
    pool.eager_only = true;
    // Earlier chunks may already have written KV and recurrent state.
    auto remaining = ggml_graph_view(graph, first, graph->n_nodes);
    return ggml_backend_cuda_graph_compute(backend, &remaining);
}

static void ggml_backend_cuda_set_spark_graph_mode(ggml_backend_t backend, bool enabled, uint64_t parent_uid) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!ctx->observed_pool || ctx->curr_stream_no != 0 || !ctx->stream_context().concurrent_events.empty()) { return; }
    const bool active = backend->iface.graph_compute == ggml_backend_cuda_spark_decode;
    if (!enabled && !active) { return; }
    if (enabled && active) {
        auto & pool = static_cast<ggml_cuda_spark_decode_pool &>(*ctx->pools[ctx->device][0]);
        if (pool.parent_uid == parent_uid) {
            pool.learning_splits = false;
            return;
        }
        ggml_cuda_set_device(ctx->device);
        ggml_backend_cuda_synchronize(backend);
        pool.discard();
        pool.uid = 0;
        pool.parent_uid = parent_uid;
        pool.largest_uid = 0;
        pool.largest_nodes = 0;
        pool.learning_splits = parent_uid != 0;
        return;
    }
    ggml_cuda_set_device(ctx->device);
    if (enabled) {
        if (std::getenv("GGML_CUDA_DISABLE_GRAPHS") || ggml_cuda_pool_debug_enabled()) { return; }
        auto * pool = new (std::nothrow) ggml_cuda_spark_decode_pool(ctx->device);
        if (!pool) { return; }
        pool->parent_uid = parent_uid;
        pool->learning_splits = parent_uid != 0;
        ctx->pools[ctx->device][0].reset(pool);
        backend->iface.graph_compute = ggml_backend_cuda_spark_decode;
    } else {
        ggml_backend_cuda_synchronize(backend);
        ctx->pools[ctx->device][0].reset();
        backend->iface.graph_compute = ggml_backend_cuda_graph_compute;
    }
}

static void ggml_backend_cuda_set_spark_decode_graph(ggml_backend_t backend, bool enabled) {
    ggml_backend_cuda_set_spark_graph_mode(backend, enabled, 0);
}

static void ggml_backend_cuda_set_spark_split_graph(ggml_backend_t backend, const ggml_cgraph * parent) {
    ggml_backend_cuda_set_spark_graph_mode(backend, parent->uid != 0, parent->uid);
}
