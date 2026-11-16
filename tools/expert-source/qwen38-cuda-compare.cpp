// Scoped CUDA numerical control, explicitly invoked and excluded from default builds.
#include "llama.h"
#include "ggml-cpu.h"
#include "llama-atomic.h"
#include "llama-model.h"
#include "llama-memory-contract.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <chrono>
#include <fstream>
#include <memory>
#include <regex>
#include <stdexcept>
#include <string>
#include <vector>
#include <unordered_map>

static void require(bool ok, const char * text) { if (!ok) { throw std::runtime_error(text); } }
struct selected_trace {
    llama_memory_budget::ticket charge;
    std::vector<uint8_t> scratch;
    std::ofstream index, data, layout;
    std::regex pattern;
    bool active = false, failed = false;
    size_t offset = 0;
    std::unordered_map<std::string, ggml_tensor *> reduction_weights;
    bool layout_active = false;

    static bool inspect(ggml_tensor * t, bool ask, void * opaque) {
        auto & self = *static_cast<selected_trace *>(opaque);
        if (!self.active) { return !ask; }
        if (ask) {
            const std::string name = t->name;
            const std::string norm_prefix = "ffn_moe_weights_norm-", weighted_prefix = "ffn_moe_weighted-";
            if (name.compare(0, norm_prefix.size(), norm_prefix) == 0 && t->op == GGML_OP_DIV) {
                self.reduction_weights[name.substr(norm_prefix.size())] = t;
            }
            if (name.compare(0, weighted_prefix.size(), weighted_prefix) == 0 && t->op == GGML_OP_MUL) {
                self.layout_active = std::regex_match("ffn_moe_out-" + name.substr(weighted_prefix.size()), self.pattern);
            }
            if (self.layout_active) {
                const auto describe = [](const ggml_tensor * x) {
                    return nlohmann::ordered_json{{"name", x->name}, {"op", ggml_op_name(x->op)}, {"flags", x->flags},
                        {"tensor", reinterpret_cast<uintptr_t>(x)}, {"data", reinterpret_cast<uintptr_t>(x->data)},
                        {"bytes", x->buffer ? ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(x->buffer), x) : ggml_nbytes(x)},
                        {"buffer", x->buffer ? ggml_backend_buffer_name(x->buffer) : "none"}};
                };
                auto row = describe(t); row["src"] = nlohmann::ordered_json::array();
                for (auto * src : t->src) { if (src) { row["src"].push_back(describe(src)); } }
                self.layout << row.dump() << '\n';
                if (!self.layout) { self.failed = true; }
                if (name.compare(0, std::string("ffn_moe_out-").size(), "ffn_moe_out-") == 0) { self.layout_active = false; }
            }
            if (!std::regex_match(t->name, self.pattern)) { return false; }
            if (ggml_nbytes(t) > self.scratch.size()) { self.failed = true; return false; }
            return true;
        }
        self.record(t, t->name);
        if (std::string(t->name) == "ffn_moe_gate-27") {
            auto * input = t->src[t->op == GGML_OP_CUSTOM ? 0 : 1];
            self.record(input, "ffn_moe_gate-27.input");
        }
        const std::string name = t->name, down_prefix = "ffn_moe_down-";
        if (name.compare(0, down_prefix.size(), down_prefix) == 0) {
            // The weights are computed and still live before the following MUL.
            // Observing them after the reduction is invalid: its output may reuse
            // that allocation. Do not split Top-K itself for this observation.
            const auto it = self.reduction_weights.find(name.substr(down_prefix.size()));
            if (it != self.reduction_weights.end()) { self.record(it->second, name + ".weights"); }
        }
        return !self.failed;
    }

    void record(ggml_tensor * t, const std::string & name) {
        auto & self = *this;
        const size_t bytes = ggml_nbytes(t);
        if (bytes > self.scratch.size()) { self.failed = true; return; }
        ggml_backend_tensor_get(t, self.scratch.data(), 0, bytes);
        uint64_t hash = UINT64_C(14695981039346656037);
        for (size_t i = 0; i < bytes; ++i) { hash = (hash ^ self.scratch[i]) * UINT64_C(1099511628211); }
        self.index << name << '\t' << ggml_op_name(t->op) << '\t' << ggml_type_name(t->type);
        for (auto dim : t->ne) { self.index << '\t' << dim; }
        for (auto stride : t->nb) { self.index << '\t' << stride; }
        self.index << '\t' << bytes << '\t' << hash << '\t' << self.offset << '\t'
                   << (t->buffer ? ggml_backend_buffer_name(t->buffer) : "none") << '\n';
        self.data.write(reinterpret_cast<const char *>(self.scratch.data()), bytes);
        self.offset += bytes;
        self.failed = self.failed || !self.index || !self.data;
    }
};
// Selected manual-resize control. Keep the established transient allowance;
// a rejected request is evidence, not permission to lower the runtime reserve.
struct resize_probe {
    static constexpr size_t mib = size_t(1024) * 1024;
    static constexpr size_t headroom = 64 * mib;
    const llama_memory_contract * contract = nullptr;
    size_t host_shrink = 4092 * mib, device_shrink = 2040 * mib;
    unsigned boundaries = 0;
    std::atomic<unsigned> reads {0};
    std::atomic<size_t> read_staging {0};
    bool worker = false, cancel_first = false;
    bool require_accept = false;
    nlohmann::ordered_json events = nlohmann::ordered_json::array();

    nlohmann::ordered_json snapshot() const {
        auto result = nlohmann::ordered_json::array();
        for (auto domain : {llama_memory_domain::host, llama_memory_domain::device}) {
            const auto s = contract->ledger->snapshot(domain);
            auto categories = nlohmann::ordered_json::array();
            for (const auto & c : s.categories) { categories.push_back(c.total()); }
            result.push_back({{"requested", s.requested}, {"applied", s.applied},
                {"used", s.used()}, {"peak", s.high_water}, {"released", s.released},
                {"categories", categories}});
            require(s.used() <= s.applied && !s.pending(), "resize left an over-budget or pending ledger");
        }
        require(contract->runtime.size() == (size_t(2) << 30), "runtime reserve changed");
        return result;
    }

    bool apply(llama_context * ctx, const char * name, size_t host, size_t device, bool must_reject = false) {
        const auto before = snapshot();
        const bool accepted = llama_discrete_budget_resize(ctx, host, device, headroom);
        const auto after = snapshot();
        std::fprintf(stderr, "resize %s accepted=%d host_used=%zu device_used=%zu device_fixed=%zu released=%zu\n",
            name, int(accepted), after[0]["used"].get<size_t>(), after[1]["used"].get<size_t>(),
            before[1]["used"].get<size_t>() - before[1]["categories"][1].get<size_t>(),
            after[1]["released"].get<size_t>() - before[1]["released"].get<size_t>());
        require(accepted || before == after, "rejected resize changed budget or residency");
        require(!must_reject || !accepted, "impossible resize accepted");
        require(must_reject || !require_accept || accepted, "qualified resize target was rejected");
        require(!accepted || (after[0]["applied"] == host && after[1]["applied"] == device), "resize target not applied");
        events.push_back({{"name", name}, {"host_target", host}, {"device_target", device},
            {"scratch_headroom", headroom}, {"accepted", accepted}, {"before", before}, {"after", after}});
        return accepted;
    }

    void decode_boundary(llama_context * ctx, int step) {
        if (step == 0) {
            apply(ctx, "reject-host", mib, 2048 * mib, true);
            apply(ctx, "reject-device", 4096 * mib, mib, true);
            apply(ctx, "same-cap-admission", 4096 * mib, 2048 * mib);
        } else if (step == 1) {
            apply(ctx, "shrink", host_shrink, device_shrink);
            if (require_accept) {
                const auto & e = events.back();
                require(e["after"][1]["released"].get<size_t>() > e["before"][1]["released"].get<size_t>(),
                    "decode shrink did not reclaim populated device cache");
            }
        } else if (step == 2) {
            apply(ctx, "grow", 4096 * mib, 2048 * mib);
        }
    }

    static bool mid_read(void * user) noexcept {
        auto & self = *static_cast<resize_probe *>(user);
        const auto ram = self.contract->ledger->snapshot(llama_memory_domain::host);
        self.read_staging.store(ram.categories[size_t(llama_memory_category::io)].allocated, std::memory_order_release);
        return self.reads.fetch_add(1, std::memory_order_acq_rel) != 0 || !self.cancel_first;
    }

    bool arm(llama_context * ctx) {
        return llama_discrete_prefill_probe_set(ctx, mid_read, boundary, this, worker);
    }

    static bool boundary(llama_context * ctx, void * user) noexcept {
        auto & self = *static_cast<resize_probe *>(user);
        try {
            if (++self.boundaries > 2) { return true; }
            const auto ram = self.contract->ledger->snapshot(llama_memory_domain::host);
            const auto staging = self.read_staging.load(std::memory_order_acquire);
            require(self.reads.load(std::memory_order_acquire) == self.boundaries &&
                staging > ram.categories[size_t(llama_memory_category::io)].allocated,
                "prefill resize did not cross a completed staged read");
            const bool shrink = self.boundaries == 1;
            self.apply(ctx, shrink ? "prefill-shrink" : "prefill-grow",
                shrink ? self.host_shrink : 4096 * mib, shrink ? self.device_shrink : 2048 * mib);
            self.events.back()["read_staging"] = staging;
            self.events.back()["read_requests"] = self.reads.load(std::memory_order_acquire);
            self.events.back()["worker"] = self.worker;
            self.events.back()["read_cancelled"] = self.cancel_first && shrink;
            if (shrink) { return self.arm(ctx); }
            return llama_discrete_prefill_probe_set(ctx, nullptr, nullptr, nullptr);
        } catch (const std::exception & e) {
            std::fprintf(stderr, "Qwen resize boundary: %s\n", e.what());
            return false;
        }
    }

};
static void batch_set(llama_batch_ext * batch, const std::vector<llama_token> & tokens, int position) {
    llama_batch_ext_clear(batch);
    for (auto token : tokens) {
        int index = llama_batch_ext_add_token(batch, 0, token);
        require(index >= 0, "batch append failed");
        llama_pos pos = position++;
        llama_batch_ext_set_pos(batch, index, &pos);
    }
    llama_batch_ext_set_output_logits(batch, int(tokens.size()) - 1, true);
}
int main(int argc, char ** argv) {
    try {
        require(argc >= 4, "usage: llama-qwen38-cuda-compare MODEL stream|native OUTPUT_STEM [LONG_PROMPT_FILE [PREFIX_TOKENS [TRACE_REGEX]]] [--prefill-only] [--resume-state FILE --resume-position N] [--resize-probe] [--prefill-resize-probe] [--resize-read-worker] [--resize-cancel-read] [--context 2048|4096] [--state-file-staging-probe]");
        const bool long_prefill = argc >= 5;
        const int prefix = argc >= 6 ? std::stoi(argv[5]) : 0;
        const bool tracing = argc >= 7 && argv[6][0] != '\0';
        bool prefill_only = false, resizing = false, prefill_resizing = false, resize_worker = false, resize_cancel = false;
        int context = 0;
        bool state_file_probe = false;
        std::string resume_state;
        int resume_position = 0;
        for (int arg = 7; arg < argc; ++arg) {
            const std::string option = argv[arg];
            if (option == "--prefill-only") { prefill_only = true; }
            else if (option == "--state-file-staging-probe") { state_file_probe = true; }
            else if (option == "--resize-probe") { resizing = true; }
            else if (option == "--prefill-resize-probe") { prefill_resizing = true; }
            else if (option == "--resize-read-worker") { resize_worker = true; }
            else if (option == "--resize-cancel-read") { resize_cancel = true; }
            else if (option == "--context" && arg + 1 < argc) { context = std::stoi(argv[++arg]); }
            else if (option == "--resume-state" && arg + 1 < argc) { resume_state = argv[++arg]; }
            else if (option == "--resume-position" && arg + 1 < argc) { resume_position = std::stoi(argv[++arg]); }
            else { throw std::runtime_error("invalid diagnostic option: " + option); }
        }
        require(resume_state.empty() ? resume_position == 0 : resume_position > 0, "invalid resume position");
        require(argc < 6 || (prefix > 0 && prefix <= 4092), "invalid prefix length");
        const std::string mode = argv[2], stem = argv[3];
        require(mode == "stream" || mode == "native", "invalid comparison mode");
        const bool streaming = mode == "stream";
        require(!(resizing || prefill_resizing) || (streaming && !prefill_only && !tracing), "resize probe requires untraced streamed decode");
        require(!resize_worker || prefill_resizing, "worker requires prefill resize");
        require(!resize_cancel || resize_worker, "read cancellation requires the worker probe");
        require(!state_file_probe || (streaming && !tracing && !resizing && !prefill_resizing), "state-file probe requires isolated untraced streaming");
        require(!context || (long_prefill && (context == 2048 || context == 4096)), "unsupported diagnostic context");
        llama_backend_init();
        ggml_backend_load_all();
        auto device = ggml_backend_dev_by_name("CUDA0");
        require(device != nullptr, "CUDA0 unavailable");
        ggml_backend_dev_t devices[] = {device, nullptr};
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 22;
        mp.devices = devices;
        mp.use_extra_bufts = false;
        llama_model_tensor_buft_override overrides[] = {
            {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_cpu_buffer_type()}, {nullptr, nullptr}};
        if (!streaming) { mp.tensor_buft_overrides = overrides; }
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(streaming ?
            llama_load_discrete_model(argv[1], mp, device, size_t(4) << 30, size_t(2) << 30, true, true, false) :
            llama_model_load_from_file(argv[1], mp), llama_model_free);
        require(bool(model), "model load failed");
        require(model->arch == LLM_ARCH_QWEN4EXP, "selected profile requires Qwen4Exp");
        resize_probe resize;
        resize.contract = model->memory_contract.get();
        resize.worker = resize_worker;
        resize.cancel_first = resize_cancel;
        if (context == 2048) {
            resize.host_shrink = 4064 * resize_probe::mib;
            resize.device_shrink = 2019 * resize_probe::mib;
            resize.require_accept = true;
        }
        selected_trace trace;
        if (tracing) {
            constexpr size_t bytes = size_t(4) << 20;
            trace.pattern = std::regex(argv[6]);
            if (streaming) {
                trace.charge = model->memory_contract->ledger->reserve(llama_memory_domain::host, llama_memory_category::scratch, bytes);
                require(bool(trace.charge), "tensor tracing exceeds the RAM cap");
            }
            trace.scratch.resize(bytes);
            if (trace.charge) { trace.charge.commit(); }
            trace.index.open(stem + ".trace.tsv"); trace.data.open(stem + ".trace.bin", std::ios::binary);
            trace.layout.open(stem + ".layout.jsonl");
            require(bool(trace.index) && bool(trace.data) && bool(trace.layout), "cannot open trace outputs");
        }
        auto cp = llama_context_default_params();
        cp.n_ctx = context ? context : (long_prefill ? 4096 : 128); cp.n_batch = 128; cp.n_ubatch = 8; cp.n_seq_max = 1;
        cp.n_threads = cp.n_threads_batch = 4;
        if (tracing) { cp.cb_eval = selected_trace::inspect; cp.cb_eval_user_data = &trace; }
        // At 2 GiB, singleton layout groups have only 50 slots: upstream
        // falls back to CPU for eight tokens x ten experts. Four GiB admits
        // at least 80 slots per group on this checkpoint, preserving placement.
        constexpr size_t native_cache_bytes = size_t(4) << 30;
        if (!streaming) { cp.moe_cache_size = native_cache_bytes; cp.moe_cache_cpu_share = 0; }
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(bool(ctx), "context initialization failed");
        // Reserve before allocation and before expert cache growth. No full-state
        // diagnostic buffer is hidden outside the streamed memory contract.
        const size_t capture_bytes = size_t(long_prefill ? 32 : 128) << 20;
        if (long_prefill) {
            // The upstream file serializer stages one state tensor range at a
            // time. Bound every range for this single-sequence F32/F16 profile;
            // Allow old and new vector storage to coexist during capacity growth.
            const auto & hp = model->hparams;
            size_t largest = std::max(size_t(hp.n_embd_r()), size_t(hp.n_embd_s())) * sizeof(float);
            largest = std::max(largest, size_t(hp.ple_conv_state()) * sizeof(float));
            for (uint32_t layer = 0; layer < hp.n_layer(); ++layer) {
                largest = std::max(largest, size_t(std::max(hp.n_embd_k_gqa(layer), hp.n_embd_v_gqa(layer))) * cp.n_ctx * sizeof(float));
            }
            require(largest <= capture_bytes / 3, "file state staging exceeds charged bound");
        }
        llama_memory_budget::ticket capture;
        if (streaming) {
            capture = model->memory_contract->ledger->reserve(llama_memory_domain::host, llama_memory_category::scratch, capture_bytes);
            require(bool(capture), "state capture cannot fit inside the RAM cap");
        }
        std::vector<uint8_t> state(long_prefill ? 0 : capture_bytes);
        if (capture) { capture.commit(); }
        std::string prompt = "The capital of France is";
        if (long_prefill) {
            std::ifstream file(argv[4]);
            require(bool(file), "cannot open long prompt");
            prompt.assign(std::istreambuf_iterator<char>(file), {});
        }
        const auto * vocab = llama_model_get_vocab(model.get());
        const int full_count = -llama_tokenize(vocab, prompt.c_str(), int(prompt.size()), nullptr, 0, true, true);
        require(full_count > 0 && (long_prefill ? full_count > 1024 : full_count <= 32), "unexpected prompt length");
        std::vector<llama_token> input(full_count);
        require(llama_tokenize(vocab, prompt.c_str(), int(prompt.size()), input.data(), full_count, true, true) == full_count, "tokenization failed");
        require(prefix <= full_count, "prefix exceeds prompt");
        if (prefix) { input.resize(prefix); }
        const int count = int(input.size());
        require(count + 4 <= int(cp.n_ctx), "selected prompt exceeds context");
        require(!prefill_resizing || count - resume_position > 16, "prefill resize requires three microbatches");
        require(resume_position < count, "resume position exceeds selected prefix");
        if (!resume_state.empty()) {
            size_t saved_tokens = 0;
            require(llama_state_load_file(ctx.get(), resume_state.c_str(), nullptr, 0, &saved_tokens) && saved_tokens == 0,
                "cannot restore diagnostic state with an empty token header");
        }
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        require(bool(batch), "batch allocation failed");
        nlohmann::ordered_json report = {{"mode", mode}, {"prompt_ids", input}, {"gpu_layers", 22},
            {"context_requested", cp.n_ctx}, {"microbatch", 8}, {"threads", 4}, {"capture_capacity", capture_bytes},
            {"prompt_tokens", count}, {"state_format", long_prefill ? "session-file" : "raw"},
            {"trace_regex", tracing ? argv[6] : ""}, {"resume_position", resume_position}, {"prefill_only", prefill_only},
            {"native_host_expert_cache_bytes", streaming ? size_t(0) : native_cache_bytes}};
        std::ofstream logits(stem + ".f32", std::ios::binary);
        require(bool(logits), "cannot open logits output");
        const auto start = std::chrono::steady_clock::now();
        trace.active = tracing;
        if (prefill_resizing) { require(resize.arm(ctx.get()), "cannot arm prefill resize"); }
        for (int pos = resume_position; pos < count; pos += cp.n_batch) {
            const int end = std::min(count, pos + int(cp.n_batch));
            batch_set(batch.get(), std::vector<llama_token>(input.begin() + pos, input.begin() + end), pos);
            require(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "prefill failed");
            llama_synchronize(ctx.get());
            require(!trace.failed, "tensor trace failed or selected tensor exceeds workspace");
            std::printf("prefill=%d/%d\n", end, count); std::fflush(stdout);
        }
        trace.active = false;
        report["prefill_seconds"] = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        report["ids"] = nlohmann::ordered_json::array();
        for (int step = 0; step < 4; ++step) {
            const float * row = llama_get_logits_ith(ctx.get(), -1);
            const size_t size = llama_vocab_n_tokens(vocab);
            logits.write(reinterpret_cast<const char *>(row), size * sizeof(float));
            require(bool(logits), "logit write failed");
            const auto token = llama_token(std::max_element(row, row + size) - row);
            report["ids"].push_back(token);
            std::printf("step=%d token=%d\n", step, token); std::fflush(stdout);
            if (prefill_only || llama_vocab_is_eog(vocab, token) || step == 3) { break; }
            if (resizing) { resize.decode_boundary(ctx.get(), step); }
            batch_set(batch.get(), {token}, count + step);
            require(llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) == 0, "decode failed");
            llama_synchronize(ctx.get());
        }
        if (resizing || prefill_resizing) {
            require(resize.events.size() == (resizing ? 5u : 0u) + (prefill_resizing ? 2u : 0u), "resize sequence did not complete");
            report["resize_events"] = resize.events;
        }
        if (state_file_probe) {
            auto & ledger = *model->memory_contract->ledger;
            const auto host = llama_memory_domain::host;
            const auto scratch = size_t(llama_memory_category::scratch);
            const auto before = ledger.snapshot(host);
            const auto block_free_capacity = [&]() {
                const auto current = ledger.snapshot(host);
                auto held = ledger.reserve(host, llama_memory_category::scratch, current.requested - current.used());
                require(bool(held), "cannot reserve diagnostic free capacity");
                return held;
            };
            const auto failed_path = stem + ".denied.slot";
            const auto valid_path = stem + ".slot";
            {
                auto held = block_free_capacity();
                require(llama_state_seq_save_file(ctx.get(), failed_path.c_str(), 0, nullptr, 0) == 0,
                    "state writer escaped the RAM admission limit");
            }
            require(ledger.snapshot(host).categories[scratch].total() == before.categories[scratch].total(),
                "denied state writer leaked staging");
            const auto bytes = llama_state_seq_save_file(ctx.get(), valid_path.c_str(), 0, nullptr, 0);
            require(bytes > 12, "state writer could not retry after capacity release");
            require(ledger.snapshot(host).categories[scratch].total() == before.categories[scratch].total(),
                "state writer retained staging after completion");
            llama_token dummy = 0;
            size_t restored_tokens = 0;
            {
                auto held = block_free_capacity();
                require(llama_state_seq_load_file(ctx.get(), valid_path.c_str(), 0, &dummy, 0, &restored_tokens) == 0,
                    "state reader escaped the RAM admission limit");
            }
            require(ledger.snapshot(host).categories[scratch].total() == before.categories[scratch].total(),
                "denied state reader leaked staging");
            require(llama_state_seq_load_file(ctx.get(), valid_path.c_str(), 0, &dummy, 0, &restored_tokens) == bytes &&
                    restored_tokens == 0, "state reader could not retry after capacity release");
            require(ledger.snapshot(host).categories[scratch].total() == before.categories[scratch].total(),
                "state reader retained staging after completion");
            report["state_file_staging"] = {{"save_denied_and_retried", true}, {"restore_denied_and_retried", true},
                {"scratch_released", true}, {"file_bytes", bytes},
                {"pressure", "synthetic ledger reservation, not OS memory pressure"}};
        }
        const size_t needed = llama_state_get_size(ctx.get());
        if (long_prefill) {
            require(llama_state_save_file(ctx.get(), (stem + ".state").c_str(), nullptr, 0), "state file capture failed");
        } else {
            require(needed <= state.size(), "state exceeds admitted capture buffer");
            require(llama_state_get_data(ctx.get(), state.data(), state.size()) == needed, "state capture failed");
            std::ofstream state_file(stem + ".state", std::ios::binary);
            state_file.write(reinterpret_cast<const char *>(state.data()), needed);
            require(bool(state_file), "state write failed");
        }
        report["state_bytes"] = needed;
        if (streaming) {
            llama_discrete_budget_data data {};
            require(llama_discrete_budget_get(ctx.get(), &data), "budget snapshot failed");
            require(data.high_water[0] <= (size_t(4) << 30) && data.high_water[1] <= (size_t(2) << 30), "budget exceeded");
            report["host_used"] = data.used[0]; report["device_used"] = data.used[1];
            report["host_peak"] = data.high_water[0]; report["device_peak"] = data.high_water[1];
        }
        std::ofstream out(stem + ".json"); out << report.dump(2) << '\n';
        require(bool(out), "report write failed");
    } catch (const std::exception & e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
