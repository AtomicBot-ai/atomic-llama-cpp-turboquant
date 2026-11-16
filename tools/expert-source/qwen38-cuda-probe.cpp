// Explicit operation qualification on packed real-weight extracts, not a model test.
#include "llama-expert-executor.h"
#include "llama-memory-contract.h"
#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}
static std::vector<float> read(ggml_tensor * tensor) {
    std::vector<float> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, ggml_nbytes(tensor));
    return values;
}
static void exact(ggml_tensor * a, ggml_tensor * b) {
    auto x = read(a), y = read(b);
    require(x.size() == y.size(), "output shape mismatch");
    for (size_t i = 0; i < x.size(); ++i) {
        if (std::memcmp(&x[i], &y[i], sizeof(float))) {
            std::fprintf(stderr, "first mismatch at %zu: %.9g vs %.9g\n", i, x[i], y[i]);
            throw std::runtime_error("output is not bitwise equal");
        }
    }
}
static ggml_cgraph * graph(ggml_context * ctx, ggml_tensor * output) {
    auto * result = ggml_new_graph_custom(ctx, 32, false);
    ggml_build_forward_expand(result, output);
    return result;
}
int main(int argc, char ** argv) {
    try {
        require(argc >= 2 && argc <= 7, "usage: llama-qwen38-cuda-probe fixture.gguf [type|all] [tokens|0] [cuda|cpu|ple|token|reject|all|replay] [SECOND_SHARD REPLAY_STEM]");
        const std::string selected_type = argc > 2 ? argv[2] : "all";
        const int selected_tokens = argc > 3 ? std::stoi(argv[3]) : 0;
        const std::string selected_mode = argc > 4 ? argv[4] : "all";
        const bool replay = selected_mode == "replay";
        require(replay ? argc == 7 && selected_type == "iq2_xxs" && selected_tokens == 8 : argc <= 5, "invalid replay arguments");
        int cases = 0;
        ggml_backend_load_all();
        auto dev = ggml_backend_dev_by_name("CUDA0");
        require(dev != nullptr, "CUDA0 unavailable");
        ggml_backend_ptr cuda(ggml_backend_dev_init(dev, nullptr));
        ggml_backend_ptr cpu(ggml_backend_init_by_name("CPU", nullptr));
        require(cuda && cpu, "backend initialization failed");
        ggml_backend_cpu_set_n_threads(cpu.get(), 1);
        std::vector<std::string> paths {argv[1]};
        if (replay) { paths.push_back(argv[5]); }
        auto source = std::make_shared<llama_expert_source>(paths);
        require(source->architecture() == "qwen4exp", "wrong fixture architecture");
        const size_t ram_cap = size_t(4) << 30, vram_cap = size_t(2) << 30;
        llama_memory_contract contract(ram_cap, vram_cap,
            {ggml_backend_dev_buffer_type(dev), ggml_backend_dev_host_buffer_type(dev)});
        // The replay holds the ordinary reference weights beside streaming.
        // Reserve conservative host/device envelopes before those allocations.
        llama_memory_budget::ticket replay_host, replay_device;
        if (replay) {
            replay_host = contract.ledger->reserve(llama_memory_domain::host, llama_memory_category::scratch, size_t(1) << 30);
            replay_device = contract.ledger->reserve(llama_memory_domain::device, llama_memory_category::scratch, size_t(512) << 20);
            require(replay_host && replay_device, "replay control exceeds budget");
            replay_host.commit(); replay_device.commit();
        }
        ggml_context_ptr meta(ggml_init({1 << 20, nullptr, true}));
        std::vector<const ggml_tensor *> weights;
        ggml_tensor * table = nullptr;
        ggml_tensor * token_table = nullptr;
        for (const auto & info : source->tensors()) {
            if (replay && info.name != "blk.27.ffn_gate_exps.weight") { continue; }
            auto * w = ggml_new_tensor_4d(meta.get(), info.type, info.ne[0], info.ne[1], info.ne[2], info.ne[3]);
            ggml_set_name(w, info.name.c_str());
            if (info.can_read_experts()) { weights.push_back(w); }
            else if (info.name == "token_embd.weight") { token_table = w; }
            else { require(!table, "multiple PLE tables"); table = w; }
        }
        require(replay ? weights.size() == 1 : weights.size() == 7 && table && table->ne[0] == 160 && table->ne[1] == 4,
            "fixture must contain seven expert formats and four width-160 PLE rows");
        if (selected_mode == "reject") {
            {
                llama_expert_executor exec(source, weights, dev, 0, {}, &contract);
                auto * x = ggml_new_tensor_3d(meta.get(), GGML_TYPE_F32, weights[0]->ne[0], 2, 9);
                auto * ids = ggml_new_tensor_2d(meta.get(), GGML_TYPE_I32, 2, 9);
                bool rejected = false;
                try { exec.build(meta.get(), weights[0], x, ids); }
                catch (const std::runtime_error & e) { rejected = std::strstr(e.what(), "at most 8 tokens") != nullptr; }
                require(rejected, "unqualified batch was not rejected");
            }
            require(contract.ledger->snapshot(llama_memory_domain::device).used() == 0, "rejection leaked device storage");
            require(contract.ledger->snapshot(llama_memory_domain::host).used() == llama_memory_contract::runtime_envelope,
                "rejection leaked host storage");
            std::puts("PASS rejects nine rows before compute, no remaining storage");
            return 0;
        }
        for (bool host_experts : {false, true}) {
            if (replay && host_experts) { continue; }
            llama_expert_executor exec(source, weights, dev, 0, {}, &contract,
                host_experts ? weights : std::vector<const ggml_tensor *>{});
            for (const auto * weight : weights) {
                if ((selected_mode == "ple" || selected_mode == "token") || (host_experts && selected_mode == "cuda") ||
                    (!host_experts && selected_mode == "cpu")) { continue; }
                if (selected_type != "all" && selected_type != ggml_type_name(weight->type)) { continue; }
                const auto & info = source->tensor(weight->name);
                std::vector<uint8_t> packed(info.bytes);
                for (int64_t e = 0; e < weight->ne[2]; ++e) {
                    require(source->read_expert(weight->name, e, packed.data() + e * info.expert_bytes(), info.expert_bytes()), "fixture read failed");
                }
                for (int tokens : {1, 8}) {
                    if (selected_tokens && tokens != selected_tokens) { continue; }
                    constexpr int slots = 10;
                    ggml_context_ptr native(ggml_init({1 << 20, nullptr, true}));
                    auto * w = ggml_dup_tensor(native.get(), weight);
                    auto * x = ggml_new_tensor_3d(native.get(), GGML_TYPE_F32, weight->ne[0], replay ? 1 : slots, tokens);
                    auto * ids = ggml_new_tensor_2d(native.get(), GGML_TYPE_I32, slots, tokens);
                    auto * y = ggml_mul_mat_id(native.get(), w, x, ids);
                    auto * g = graph(native.get(), y);
                    // Keep native graph captures within one graph/buffer lifetime.
                    ggml_backend_ptr native_cuda(host_experts ? nullptr : ggml_backend_dev_init(dev, nullptr));
                    auto reference = host_experts ? cpu.get() : native_cuda.get();
                    require(reference != nullptr, "native backend initialization failed");
                    require(ggml_backend_supports_op(reference, y), "native routed operation unsupported");
                    ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(native.get(), reference));
                    require(bool(buf), "native allocation failed");
                    ggml_backend_buffer_set_usage(buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                    std::vector<float> activation(ggml_nelements(x));
                    for (size_t i = 0; i < activation.size(); ++i) { activation[i] = float(int((i * 17 + 3) % 101) - 50) / 64; }
                    std::vector<int32_t> routes(slots * tokens);
                    for (size_t i = 0; i < routes.size(); ++i) { routes[i] = int32_t(((i % slots) * 5 + i / slots) % size_t(weight->ne[2])); }
                    if (replay) {
                        const auto load = [&](const char * suffix, void * dst, size_t size) {
                            std::ifstream file(std::string(argv[6]) + suffix, std::ios::binary);
                            require(bool(file.read(static_cast<char *>(dst), size)) && file.peek() == EOF, "invalid replay input size");
                        };
                        load(".input", activation.data(), activation.size() * sizeof(float));
                        load(".ids", routes.data(), routes.size() * sizeof(int32_t));
                    }
                    ggml_backend_tensor_set(w, packed.data(), 0, packed.size());
                    ggml_backend_tensor_set(x, activation.data(), 0, ggml_nbytes(x));
                    ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
                    std::printf("start native %s %s tokens=%d\n", host_experts ? "CPU" : "CUDA", ggml_type_name(weight->type), tokens); std::fflush(stdout);
                    require(ggml_backend_graph_compute(reference, g) == GGML_STATUS_SUCCESS, "native compute failed");
                    ggml_context_ptr streamed(ggml_init({1 << 20, nullptr, true}));
                    auto * sx = ggml_dup_tensor(streamed.get(), x);
                    auto * si = ggml_dup_tensor(streamed.get(), ids);
                    auto * sy = exec.build(streamed.get(), weight, sx, si);
                    auto * sg = graph(streamed.get(), sy);
                    ggml_backend_buffer_ptr sb(ggml_backend_alloc_ctx_tensors(streamed.get(), cpu.get()));
                    require(bool(sb), "streamed graph allocation failed");
                    ggml_backend_tensor_set(sx, activation.data(), 0, ggml_nbytes(sx));
                    ggml_backend_tensor_set(si, routes.data(), 0, ggml_nbytes(si));
                    std::puts("start streamed"); std::fflush(stdout);
                    require(ggml_backend_graph_compute(cpu.get(), sg) == GGML_STATUS_SUCCESS && !exec.error(), "streamed compute failed");
                    if (replay) {
                        for (auto pair : {std::make_pair(y, ".native"), std::make_pair(sy, ".stream")}) {
                            auto values = read(pair.first);
                            std::ofstream output(std::string(argv[6]) + pair.second, std::ios::binary);
                            output.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
                            require(bool(output), "cannot save projection replay");
                        }
                    }
                    exact(y, sy); ++cases;
                    std::printf("exact %s %s tokens=%d floats=%lld\n", host_experts ? "CPU" : "CUDA", ggml_type_name(weight->type), tokens, (long long) ggml_nelements(y));
                }
            }
            if (!host_experts && (selected_mode == "all" || selected_mode == "ple") && selected_type == "all" && !selected_tokens) {
                exec.configure_rows(source, table);
                ggml_context_ptr ctx(ggml_init({1 << 20, nullptr, true}));
                auto * t = ggml_dup_tensor(ctx.get(), table);
                auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 5);
                auto * y = ggml_get_rows(ctx.get(), t, ids);
                auto * g = graph(ctx.get(), y);
                // PLE is an input-layer tensor: ordinary partial CUDA offload keeps its gather on CPU.
                // The CUDA IQ4_NL GET_ROWS kernel requires width divisible by 256, not this width 160.
                ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu.get()));
                require(bool(buf), "native PLE allocation failed");
                std::vector<uint8_t> data(ggml_nbytes(t));
                const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
                for (int i = 0; i < 4; ++i) { source->read_row(table->name, i, data.data() + i * row_bytes, row_bytes); }
                const int32_t indices[] = {3, 0, 2, 1, 3};
                ggml_backend_tensor_set(t, data.data(), 0, data.size());
                ggml_backend_tensor_set(ids, indices, 0, sizeof(indices));
                require(ggml_backend_graph_compute(cpu.get(), g) == GGML_STATUS_SUCCESS, "native PLE compute failed");
                ggml_context_ptr sc(ggml_init({1 << 20, nullptr, true}));
                auto * si = ggml_new_tensor_1d(sc.get(), GGML_TYPE_I32, 5);
                auto * sy = exec.build_rows(sc.get(), si);
                auto * sg = graph(sc.get(), sy);
                ggml_backend_buffer_ptr sb(ggml_backend_alloc_ctx_tensors(sc.get(), cpu.get()));
                require(bool(sb), "streamed PLE allocation failed");
                ggml_backend_tensor_set(si, indices, 0, sizeof(indices));
                require(ggml_backend_graph_compute(cpu.get(), sg) == GGML_STATUS_SUCCESS && !exec.error(), "streamed PLE compute failed");
                exact(y, sy); ++cases;
                std::printf("exact CPU PLE in CUDA streaming floats=%lld row_workspace=%zu\n", (long long) ggml_nelements(y), row_bytes);
            }
            if (!host_experts && selected_mode == "token" && selected_type == "all" && !selected_tokens) {
                require(token_table && token_table->type == GGML_TYPE_IQ3_S, "missing IQ3_S token fixture");
                exec.configure_token_rows(source, token_table);
                ggml_context_ptr ctx(ggml_init({1 << 20, nullptr, true}));
                auto * t = ggml_dup_tensor(ctx.get(), token_table);
                auto * ids = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 5);
                auto * y = ggml_get_rows(ctx.get(), t, ids);
                auto * g = graph(ctx.get(), y);
                // token embedding is an input-layer tensor: ordinary partial CUDA offload keeps its gather on CPU.
                ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu.get()));
                require(bool(buf), "native token embedding allocation failed");
                std::vector<uint8_t> data(ggml_nbytes(t));
                const size_t row_bytes = ggml_row_size(t->type, t->ne[0]);
                for (int i = 0; i < 4; ++i) { source->read_row(token_table->name, i, data.data() + i * row_bytes, row_bytes); }
                const int32_t indices[] = {3, 0, 2, 1, 3};
                ggml_backend_tensor_set(t, data.data(), 0, data.size());
                ggml_backend_tensor_set(ids, indices, 0, sizeof(indices));
                require(ggml_backend_graph_compute(cpu.get(), g) == GGML_STATUS_SUCCESS, "native token embedding compute failed");
                ggml_context_ptr sc(ggml_init({1 << 20, nullptr, true}));
                auto * si = ggml_new_tensor_1d(sc.get(), GGML_TYPE_I32, 5);
                auto * sy = exec.build_token_rows(sc.get(), si);
                auto * sg = graph(sc.get(), sy);
                ggml_backend_buffer_ptr sb(ggml_backend_alloc_ctx_tensors(sc.get(), cpu.get()));
                require(bool(sb), "streamed token embedding allocation failed");
                ggml_backend_tensor_set(si, indices, 0, sizeof(indices));
                require(ggml_backend_graph_compute(cpu.get(), sg) == GGML_STATUS_SUCCESS && !exec.error(), "streamed token embedding compute failed");
                exact(y, sy); ++cases;
                std::printf("exact CPU token embedding in CUDA streaming floats=%lld row_workspace=%zu\n", (long long) ggml_nelements(y), row_bytes);
            }
        }
        require(cases > 0, "filter selected no cases");
        const auto host = contract.ledger->snapshot(llama_memory_domain::host);
        const auto device = contract.ledger->snapshot(llama_memory_domain::device);
        require(host.high_water <= ram_cap && device.high_water <= vram_cap, "ledger cap exceeded");
        require(host.used() == llama_memory_contract::runtime_envelope + replay_host.size() && device.used() == replay_device.size(), "streaming teardown leaked a reservation");
        std::printf("PASS host_peak=%zu device_peak=%zu host_after=%zu device_after=%zu\n",
            host.high_water, device.high_water, host.used(), device.used());
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
