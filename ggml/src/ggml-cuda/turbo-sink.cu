#include "turbo-sink.cuh"
#if !defined(GGML_USE_HIP)
#include <cuda_runtime.h>
#endif
#include <cstdlib>
#include <cstdio>
#include <unordered_map>

static int g_n_sink = -1;
static std::unordered_map<const void *, half *> g_sink_bufs;

int turbo_sink_n_tokens() {
    if (g_n_sink < 0) {
        const char * env = getenv("GGML_TURBO_SINK_TOKENS");
        g_n_sink = env ? atoi(env) : 0;
        if (g_n_sink > 0) {
            fprintf(stderr, "turbo: sink token protection enabled, N=%d\n", g_n_sink);
        }
    }
    return g_n_sink;
}

half * turbo_sink_get_or_alloc(const void * tensor_data, int64_t ne0, int n_sink) {
    if (n_sink <= 0) return nullptr;
    auto it = g_sink_bufs.find(tensor_data);
    if (it != g_sink_bufs.end()) return it->second;

    size_t size = ne0 * n_sink * sizeof(half);
    half * buf = nullptr;
    cudaMalloc(&buf, size);
    cudaMemset(buf, 0, size);
    g_sink_bufs[tensor_data] = buf;
    return buf;
}

half * turbo_sink_get(const void * tensor_data) {
    auto it = g_sink_bufs.find(tensor_data);
    return (it != g_sink_bufs.end()) ? it->second : nullptr;
}
