// Fixed, focused native-inference comparison; uses only upstream APIs.
#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <libproc.h>
#include <mach/mach.h>
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;
using clock_type = std::chrono::steady_clock;
static double elapsed(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}
static json memory(const llama_context * ctx) {
    json out;
    if (ctx) {
        out["backend_buffers"] = json::array();
        for (const auto & b : llama_get_memory_breakdown(ctx)) {
            out["backend_buffers"].push_back({{"type", ggml_backend_buft_name(b.first)},
                {"model", b.second.model}, {"context", b.second.context}, {"compute", b.second.compute}});
        }
    }
#if defined(__APPLE__)
    task_vm_info_data_t info {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        throw std::runtime_error("cannot sample process memory");
    }
    out["rss"] = info.resident_size;
    out["footprint"] = info.phys_footprint;
    out["rss_peak"] = info.resident_size_peak;
    out["footprint_peak"] = info.ledger_phys_footprint_peak;
    rusage_info_v4 io {};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t *>(&io)) == 0) {
        out["process_disk_read_bytes"] = io.ri_diskio_bytesread;
    }
#endif
    return out;
}
static void batch_set(llama_batch_ext * batch, const std::vector<llama_token> & tokens, int position) {
    llama_batch_ext_clear(batch);
    for (auto token : tokens) {
        const int idx = llama_batch_ext_add_token(batch, 0, token);
        if (idx < 0) { throw std::runtime_error("batch append failed"); }
        llama_pos pos = position++;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    llama_batch_ext_set_output_logits(batch, int(tokens.size()) - 1, true);
}
static std::string state_digest(llama_context * ctx) {
    std::vector<uint8_t> bytes(llama_state_get_size(ctx));
    const size_t n = llama_state_get_data(ctx, bytes.data(), bytes.size());
    if (!n || n > bytes.size()) { throw std::runtime_error("state capture failed"); }
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; ++i) { hash = (hash ^ bytes[i]) * UINT64_C(1099511628211); }
    char text[32];
    std::snprintf(text, sizeof(text), "%016llx", (unsigned long long) hash);
    return text;
}
int main(int argc, char ** argv) {
    try {
        if (argc != 4) { throw std::runtime_error("usage: native-probe MODEL PROMPT_FILE OUTPUT_STEM"); }
        std::ifstream prompt_file(argv[2]);
        if (!prompt_file) { throw std::runtime_error("cannot open prompt"); }
        const std::string prompt((std::istreambuf_iterator<char>(prompt_file)), {});
        const std::string stem = argv[3];
        llama_backend_init();
        ggml_backend_load_all();
        json report = {{"context", 512}, {"batch", 512}, {"ubatch", 32}, {"threads", 8},
            {"gpu_layers", 99}, {"output_limit", 16}, {"warmup_requests", 1}, {"measured_requests", 2},
            {"sampling", "greedy"}, {"timing", "steady clock, explicit backend synchronization; diagnostics outside phase timers"},
            {"memory_before_load", memory(nullptr)}};
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 99;
        const auto load_start = clock_type::now();
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) { throw std::runtime_error("model load failed"); }
        report["load_seconds"] = elapsed(load_start);
        const auto * vocab = llama_model_get_vocab(model.get());
        const int count = -llama_tokenize(vocab, prompt.c_str(), int(prompt.size()), nullptr, 0, true, true);
        if (count <= 0 || count + 16 > 512) { throw std::runtime_error("invalid prompt length"); }
        std::vector<llama_token> input(count);
        if (llama_tokenize(vocab, prompt.c_str(), int(prompt.size()), input.data(), count, true, true) != count) {
            throw std::runtime_error("tokenization failed");
        }
        report["prompt_ids"] = input;
        auto cp = llama_context_default_params();
        cp.n_ctx = 512; cp.n_batch = 512; cp.n_ubatch = 32; cp.n_seq_max = 1;
        cp.n_threads = 8; cp.n_threads_batch = 8; cp.no_perf = true;
        const auto context_start = clock_type::now();
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) { throw std::runtime_error("context init failed"); }
        report["context_seconds"] = elapsed(context_start);
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
        if (!batch || !sampler) { throw std::runtime_error("inference setup failed"); }
        const size_t n_vocab = llama_vocab_n_tokens(vocab);
        std::vector<float> logits(n_vocab * 16);
        std::vector<llama_token> next(1), committed;
        committed.reserve(16);
        report["memory_after_init"] = memory(ctx.get());
        report["runs"] = json::array();
        for (int run = -1; run < 2; ++run) {
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            llama_sampler_reset(sampler.get());
            committed.clear();
            batch_set(batch.get(), input, 0);
            const auto start = clock_type::now();
            if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) { throw std::runtime_error("prefill failed"); }
            llama_synchronize(ctx.get());
            const double prefill = elapsed(start);
            double decode = 0;
            int decode_calls = 0, rows = 0, stop_id = -1;
            for (int step = 0; step < 16; ++step) {
                std::memcpy(logits.data() + size_t(step) * n_vocab, llama_get_logits_ith(ctx.get(), -1), n_vocab * sizeof(float));
                ++rows;
                const auto token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
                if (llama_vocab_is_eog(vocab, token)) { stop_id = token; break; }
                committed.push_back(token);
                llama_sampler_accept(sampler.get(), token);
                if (step + 1 < 16) {
                    next[0] = token;
                    batch_set(batch.get(), next, count + step);
                    const auto step_start = clock_type::now();
                    if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) { throw std::runtime_error("decode failed"); }
                    llama_synchronize(ctx.get());
                    decode += elapsed(step_start);
                    ++decode_calls;
                }
            }
            const double request = elapsed(start);
            json row = {{"prefill_seconds", prefill}, {"decode_seconds", decode}, {"decode_calls", decode_calls},
                {"request_seconds", request}, {"committed_ids", committed}, {"logits_rows", rows},
                {"stop_id", stop_id}, {"memory_after", memory(ctx.get())}};
            if (run < 0) { report["warmup"] = std::move(row); }
            else {
                std::ofstream out(stem + ".run" + std::to_string(run) + ".f32", std::ios::binary);
                out.write(reinterpret_cast<const char *>(logits.data()), size_t(rows) * n_vocab * sizeof(float));
                if (!out) { throw std::runtime_error("cannot write logits"); }
                report["runs"].push_back(std::move(row));
            }
        }
        report["memory_before_state_capture"] = memory(ctx.get());
        report["state_fnv1a64"] = state_digest(ctx.get());
        std::ofstream out(stem + ".json"); out << report.dump(2) << '\n';
        if (!out) { throw std::runtime_error("cannot write report"); }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "native probe: %s\n", e.what());
        return 1;
    }
}
