// CUDA view of one observed pinned allocation. The backing buffer owns the
// reservation. Tensor payloads are read-only while its GPU stream is active.
struct ggml_cuda_spark_prompt_weight_view {
    ggml_backend_buffer_t backing;
    void * device_ptr;
};

static ggml_backend_buffer_t ggml_backend_cuda_spark_prompt_weights(ggml_backend_t backend, size_t bytes, void ** host) {
    auto * ctx = static_cast<ggml_backend_cuda_context *>(backend->context);
    if (!ctx->observed_pool || !ggml_backend_has_alloc_observer()) { return nullptr; }
    ggml_cuda_set_device(ctx->device);
    auto pinned = ggml_backend_cuda_host_buffer_type();
    ggml_backend_buffer_t backing = ggml_backend_buft_alloc_buffer(pinned, bytes);
    if (!backing) { return nullptr; }
    if (backing->buft != pinned) { ggml_backend_buffer_free(backing); return nullptr; }
    void * cpu = ggml_backend_buffer_get_base(backing);
    void * gpu = nullptr;
    if (cudaHostGetDevicePointer(&gpu, cpu, 0) != cudaSuccess) {
        (void) cudaGetLastError();
        ggml_backend_buffer_free(backing);
        return nullptr;
    }
    auto * view = new (std::nothrow) ggml_cuda_spark_prompt_weight_view{backing, gpu};
    if (!view) { ggml_backend_buffer_free(backing); return nullptr; }
    ggml_backend_buffer_i iface{};
    iface.get_base = [](ggml_backend_buffer_t b) { return static_cast<ggml_cuda_spark_prompt_weight_view *>(b->context)->device_ptr; };
    iface.free_buffer = [](ggml_backend_buffer_t b) {
        auto * v = static_cast<ggml_cuda_spark_prompt_weight_view *>(b->context);
        ggml_backend_buffer_free(v->backing);
        delete v;
    };
    try {
        auto buffer = ggml_backend_buffer_init(ggml_backend_cuda_buffer_type(ctx->device), iface, view, bytes);
        *host = cpu;
        return buffer;
    } catch (...) {
        ggml_backend_buffer_free(backing);
        delete view;
        throw;
    }
}
