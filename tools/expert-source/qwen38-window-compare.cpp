// Explicitly invoked numerical/admission diagnostic for the opt-in Metal window.
#include "llama.h"
#include "llama-atomic.h"
#include "llama-model.h"
#include "llama-memory-contract.h"
#include "ggml-cpu.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

static void require(bool ok, const char * message) {
    if (!ok) { throw std::runtime_error(message); }
}

int main(int argc, char ** argv) {
    try {
        require(argc == 5, "usage: window-compare MODEL native|stream PROMPT_FILE OUTPUT_STEM");
        const std::string mode = argv[2], stem = argv[4];
        require(mode == "native" || mode == "stream", "invalid comparison mode");
        const bool streaming = mode == "stream";
        const int prefix = std::getenv("QWEN_WINDOW_TOKENS") ? std::atoi(std::getenv("QWEN_WINDOW_TOKENS")) : 16;
        require(prefix == 16 || prefix == 64 || prefix == 128 || prefix == 153, "unsupported diagnostic prefix size");
        const int steps = std::getenv("QWEN_WINDOW_STEPS") ? std::atoi(std::getenv("QWEN_WINDOW_STEPS")) : 4;
        require(steps >= 4 && steps <= 128, "diagnostic output count must be between 4 and 128");
        const bool profile_decode = std::getenv("QWEN_WINDOW_DECODE_PROFILE");
        require(!profile_decode || streaming, "decode profiling requires streaming");
        llama_backend_init();
        ggml_backend_load_all();
        auto * device = ggml_backend_dev_by_name("MTL0");
        require(device != nullptr, "Metal device unavailable");
        ggml_backend_dev_t devices[] = {device, nullptr};
        auto mp = llama_model_default_params();
        mp.devices = devices;
        mp.n_gpu_layers = 99;
        mp.use_extra_bufts = false;
        llama_model_tensor_buft_override overrides[] = {
            {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
        if (streaming) { mp.memory_budget_size = size_t(8) << 30; }
        else { mp.tensor_buft_overrides = overrides; }
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
                llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model) && model->arch == LLM_ARCH_QWEN4EXP, "Qwen4Exp model load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = 4096; cp.n_batch = prefix; cp.n_ubatch = 8; cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 8;
        if (!streaming) { cp.moe_cache_size = size_t(8) << 30; cp.moe_cache_cpu_share = 0; }
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "context initialization failed");
        const bool window = streaming && std::getenv("QWEN_WINDOW_RUNTIME");
        if (window) { require(llama_prefill_window_set(ctx.get(), true), "window enable failed"); }
        const bool override_policy = window && std::getenv("QWEN_WINDOW_POLICY_OVERRIDE");
        if (override_policy) { require(llama_expert_cache_policy_set(ctx.get(), "lru"), "policy override failed"); }
        llama_memory_budget::ticket admission_hold;
        if (streaming && std::getenv("QWEN_WINDOW_HOLD_MIB")) {
            const auto bytes = size_t(std::stoul(std::getenv("QWEN_WINDOW_HOLD_MIB"))) << 20;
            admission_hold = model->memory_contract->ledger->reserve(llama_memory_domain::host,
                    llama_memory_category::scratch, bytes);
            require(bool(admission_hold), "admission probe reserve failed");
        }
        struct resize_signal {
            std::mutex mutex;
            std::condition_variable cv;
            bool seen = false, acknowledged = false;
        } signal;
        const bool resize = window && std::getenv("QWEN_WINDOW_SHRINK");
        uint64_t resize_id = 0;
        if (resize) {
            require(llama_expert_route_observer_set(ctx.get(), [](const llama_expert_route_event &, void * user) {
                auto & s = *static_cast<resize_signal *>(user);
                std::unique_lock<std::mutex> lock(s.mutex);
                if (!s.seen) {
                    s.seen = true;
                    s.cv.notify_all();
                    s.cv.wait(lock, [&] { return s.acknowledged; });
                }
            }, &signal), "route observer rejected");
        }
        // Bound the serializer's per-tensor staging, including vector growth.
        constexpr size_t capture_bytes = size_t(32) << 20;
        const auto & hp = model->hparams;
        size_t largest = std::max(size_t(hp.n_embd_r()), size_t(hp.n_embd_s())) * sizeof(float);
        largest = std::max(largest, size_t(hp.ple_conv_state()) * sizeof(float));
        for (uint32_t layer = 0; layer < hp.n_layer(); ++layer) {
            largest = std::max(largest, size_t(std::max(hp.n_embd_k_gqa(layer), hp.n_embd_v_gqa(layer))) * cp.n_ctx * sizeof(float));
        }
        require(largest <= capture_bytes / 3, "state serialization exceeds charged staging");
        llama_memory_budget::ticket capture;
        if (streaming) {
            capture = model->memory_contract->ledger->reserve(llama_memory_domain::host,
                    llama_memory_category::scratch, capture_bytes);
            require(bool(capture), "state staging cannot fit budget");
            capture.commit();
        }
        std::ifstream input_file(argv[3]);
        require(bool(input_file), "cannot open prompt");
        std::string prompt(std::istreambuf_iterator<char>(input_file), {});
        const auto * vocab = llama_model_get_vocab(model.get());
        const int count = -llama_tokenize(vocab, prompt.data(), prompt.size(), nullptr, 0, true, true);
        require(count >= prefix, "prompt must contain enough tokens for the selected window");
        std::vector<llama_token> input(count);
        require(llama_tokenize(vocab, prompt.data(), prompt.size(), input.data(), count, true, true) == count, "tokenization failed");
        input.resize(prefix);
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        require(bool(batch), "batch initialization failed");
        const bool input_embeddings = std::getenv("QWEN_WINDOW_EMBD");
        std::vector<float> synthetic_embedding(input_embeddings ? model->hparams.n_embd_inp() : 0);
        for (size_t i = 0; i < synthetic_embedding.size(); ++i) { synthetic_embedding[i] = (int(i % 17) - 8) * 0.001f; }
        std::vector<double> decode_ms;
        decode_ms.reserve(steps);
        auto decode = [&](const std::vector<llama_token> & tokens, int position) {
            const bool embedding_batch = input_embeddings && position == 0;
            const size_t prefill_hint = position == 0 ? 3481 : 0;
            llama_batch_ext_clear(batch.get());
            for (auto token : tokens) {
                const int i = embedding_batch ? llama_batch_ext_add_embd(batch.get(), 0,
                        {synthetic_embedding.data(), 1, synthetic_embedding.size()}) :
                        llama_batch_ext_add_token(batch.get(), 0, token);
                require(i >= 0, "batch append failed");
                llama_pos pos = position++;
                llama_batch_ext_set_pos(batch.get(), i, &pos);
            }
            llama_batch_ext_set_output_logits(batch.get(), tokens.size() - 1, true);
            std::thread requester;
            if (resize && position == int(tokens.size())) {
                requester = std::thread([&] {
                    std::unique_lock<std::mutex> lock(signal.mutex);
                    if (!signal.cv.wait_for(lock, std::chrono::seconds(30), [&] { return signal.seen; })) { return; }
                    // A separate thread requests resize while a real projection
                    // callback is outstanding; the callback never reenters llama.
                    resize_id = llama_memory_budget_request(ctx.get(), size_t(7) << 30);
                    signal.acknowledged = true;
                    signal.cv.notify_all();
                });
            }
            const auto decode_start = std::chrono::steady_clock::now();
            const int status = (window || input_embeddings) ? llama_expert_prefill_decode(ctx.get(), batch.get(), prefill_hint) :
                    llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get());
            if (requester.joinable()) { requester.join(); }
            require(status == 0, "decode failed");
            llama_synchronize(ctx.get());
            decode_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - decode_start).count());
        };
        nlohmann::ordered_json report = {{"mode", mode}, {"input_ids", input}, {"context", 4096},
            {"batch", prefix}, {"microbatch", 8}, {"threads", 8}, {"state_staging_bytes", capture_bytes}};
        if (input_embeddings) {
            report["input_ids"] = nullptr;
            report["input_embeddings"] = "Each row: (int(column % 17) - 8) * 0.001f";
        }
        llama_expert_streaming_data before {}, after {};
        if (streaming) { require(llama_get_expert_streaming_stats(ctx.get(), &before), "missing prefill counters"); }
        const auto begin = std::chrono::steady_clock::now();
        decode(input, 0);
        report["prefill_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
        if (streaming) {
            require(llama_get_expert_streaming_stats(ctx.get(), &after), "missing final prefill counters");
            report["prefill_io"] = {{"hits", after.hits - before.hits}, {"misses", after.misses - before.misses},
                {"logical_read_bytes", after.logical_read_bytes - before.logical_read_bytes}};
        }
        require(llama_state_save_file(ctx.get(), (stem + ".prefill.state").c_str(), nullptr, 0), "prefill state save failed");
        std::ofstream logits(stem + ".f32", std::ios::binary);
        require(bool(logits), "cannot open logits output");
        report["ids"] = nlohmann::ordered_json::array();
        if (profile_decode) { require(llama_expert_profile_set(ctx.get(), true), "profile enable failed"); }
        for (int step = 0; step < steps; ++step) {
            const float * row = llama_get_logits_ith(ctx.get(), -1);
            const size_t size = llama_vocab_n_tokens(vocab);
            logits.write(reinterpret_cast<const char *>(row), size * sizeof(float));
            require(bool(logits), "logit write failed");
            const llama_token token = std::max_element(row, row + size) - row;
            report["ids"].push_back(token);
            if (step + 1 < steps) { decode({token}, prefix + step); }
        }
        require(llama_state_save_file(ctx.get(), (stem + ".state").c_str(), nullptr, 0), "final state save failed");
        if (streaming) {
            llama_budget_data b {};
            require(llama_memory_budget_get(ctx.get(), &b), "budget snapshot failed");
            require(b.high_water <= (size_t(8) << 30), "managed cap exceeded");
            report["budget"] = {{"used", b.used}, {"peak", b.high_water}, {"cap", b.applied}};
        }
        if (window) {
            llama_prefill_window_data w;
            require(llama_prefill_window_get(ctx.get(), &w), "window counters unavailable");
            report["window"] = {{"windows", w.windows}, {"tokens", w.tokens}, {"tails", w.tails},
                {"fallbacks", w.fallbacks}, {"failed", w.failed}};
            require(!w.failed, "window failed");
            if (input_embeddings) { require(w.windows == 0, "embedding input entered the window"); }
            if (admission_hold || override_policy) { require(w.windows == 0 && w.fallbacks > 0, "admission did not fall back"); }
            if (resize) {
                llama_budget_data b;
                require(llama_memory_budget_get(ctx.get(), &b) && resize_id && b.completed_id == resize_id &&
                        b.status == LLAMA_BUDGET_APPLIED && b.applied == (size_t(7) << 30) && b.used <= b.applied,
                        "resize was not safely applied");
                require(w.windows == 1 && w.tokens == 64 && w.fallbacks > 0, "resize did not resume the remaining prefill");
                require(llama_memory_budget_request(ctx.get(), size_t(8) << 30) && llama_memory_budget_apply(ctx.get()),
                        "growth after fallback failed");
                report["resize_completed_id"] = resize_id;
            }
        }
        // Entry zero is prefill; the remaining entries time synchronized decode
        // only, excluding logit/state file writes and greedy token selection.
        report["decode_ms"] = decode_ms;
        if (streaming) {
            llama_expert_streaming_data decoded {};
            require(llama_get_expert_streaming_stats(ctx.get(), &decoded), "missing decode counters");
            report["decode_io"] = {{"hits", decoded.hits - after.hits}, {"misses", decoded.misses - after.misses},
                {"logical_read_bytes", decoded.logical_read_bytes - after.logical_read_bytes},
                {"buffer_allocations", decoded.buffer_allocations - after.buffer_allocations}};
        }
        if (profile_decode) {
            llama_expert_profile_data p {};
            require(llama_expert_profile_get(ctx.get(), &p), "profile snapshot failed");
            report["decode_profile"] = {{"read_ns", p.read_ns}, {"allocation_ns", p.allocation_ns},
                {"admission_ns", p.admission_ns}, {"projection_ns", p.projection_ns}, {"compute_ns", p.compute_ns},
                {"source_setup_ns", p.source_setup_ns}, {"source_validate_ns", p.source_validate_ns},
                {"source_seek_ns", p.source_seek_ns}, {"source_transfer_ns", p.source_transfer_ns}};
        }
        std::ofstream output(stem + ".json"); output << report.dump(2) << '\n';
        require(bool(output), "cannot write report");
        std::printf("%s\n", report.dump().c_str());
    } catch (const std::exception & error) {
        std::fprintf(stderr, "window comparison: %s\n", error.what());
        return 1;
    }
}
