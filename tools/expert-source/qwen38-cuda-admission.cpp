// Allocation-only diagnostic. Never evaluate sparse metadata fixtures as weights.
#include "llama.h"
#include "llama-atomic.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <string>

int main(int argc, char ** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: llama-qwen38-cuda-admission first-shard gpu-layers --allocation-only\n");
        return 1;
    }
    if (std::string(argv[3]) != "--allocation-only") { return 1; }
    llama_backend_init();
    ggml_backend_load_all();
    auto device = ggml_backend_dev_by_name("CUDA0");
    if (!device) { return 1; }
    auto params = llama_model_default_params();
    params.n_gpu_layers = std::atoi(argv[2]);
    auto * model = llama_load_discrete_model(argv[1], params, device, size_t(4) << 30, size_t(2) << 30,
                                            true, true, false);
    if (!model) { std::puts("REJECTED model allocation"); return 2; }
    auto cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = 128;
    cp.n_ubatch = 8;
    cp.n_seq_max = 1;
    cp.n_threads = cp.n_threads_batch = 4;
    auto * ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); std::puts("REJECTED context allocation"); return 2; }
    llama_discrete_budget_data data {};
    const bool measured = llama_discrete_budget_get(ctx, &data);
    if (measured) {
        std::printf("INITIALIZED host_used=%zu device_used=%zu host_peak=%zu device_peak=%zu\n",
            data.used[0], data.used[1], data.high_water[0], data.high_water[1]);
    }
    std::puts("Allocation only: no decode or numerical result; lazy execution workspace is not qualified.");
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return measured ? 0 : 1;
}
