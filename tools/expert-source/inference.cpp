#include "llama.h"
#include "llama-ext.h"
#include "llama-atomic.h"
#include "llama-memory-contract.h"
#include "llama-model.h"
#include "ggml-cpp.h"
#include "nlohmann/json.hpp"
#include "preview-trace.h"

#ifdef ATOMIC_ENGINE_IO_URING_PROBE
#include "uring-read-probe.h"
#endif

#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <sstream>
#include <cmath>

#if defined(__APPLE__)
#include <libproc.h>
#include <mach/mach.h>
#include <unistd.h>
#endif

using json = nlohmann::ordered_json;
using steady_clock = std::chrono::steady_clock;

static json discrete_budget_snapshot(const llama_memory_contract & contract) {
    auto domain = [&](llama_memory_domain id) {
        const auto s = contract.ledger->snapshot(id);
        return json{{"cap", s.applied}, {"used", s.used()}, {"peak", s.high_water}, {"released", s.released},
            {"runtime", s.categories[size_t(llama_memory_category::runtime)].total()},
            {"weights", s.categories[size_t(llama_memory_category::weights)].total()},
            {"experts", s.categories[size_t(llama_memory_category::experts)].total()},
            {"state", s.categories[size_t(llama_memory_category::state)].total()},
            {"scratch", s.categories[size_t(llama_memory_category::scratch)].total()},
            {"io", s.categories[size_t(llama_memory_category::io)].total()}};
    };
    return {{"ram", domain(llama_memory_domain::host)}, {"vram", domain(llama_memory_domain::device)}};
}
static double seconds(steady_clock::time_point start) {
    return std::chrono::duration<double>(steady_clock::now() - start).count();
}

struct cuda_prefill_resize {
    const llama_memory_contract * contract = nullptr;
    ggml_backend_dev_t device = nullptr;
    json events = json::array();
    std::atomic<size_t> read_staging {0};
    std::atomic<unsigned> read_requests {0};
    bool worker = false;
    bool cancel = false;
    size_t boundaries = 0;
#ifdef ATOMIC_ENGINE_IO_URING_PROBE
    std::unique_ptr<uring_read_probe> uring;
    uring_read_result read_result;

    static bool submitted_read(void * user, int fd, size_t offset, void * dst, size_t size) {
        auto & self = *static_cast<cuda_prefill_resize *>(user);
        self.read_result = self.uring->run(fd, offset, dst, size, mid_read, user);
        return !self.read_result.cancel_requested;
    }
#endif

    bool arm(llama_context * ctx) {
#ifdef ATOMIC_ENGINE_IO_URING_PROBE
        if (uring) {
            return llama_discrete_prefill_probe_set(ctx, nullptr, boundary, this, false, submitted_read);
        }
#endif
        return llama_discrete_prefill_probe_set(ctx, mid_read, boundary, this, worker);
    }

    static bool mid_read(void * user) noexcept {
        auto & self = *static_cast<cuda_prefill_resize *>(user);
        const auto ram = self.contract->ledger->snapshot(llama_memory_domain::host);
        self.read_staging.store(ram.categories[size_t(llama_memory_category::io)].allocated, std::memory_order_release);
        return self.read_requests.fetch_add(1, std::memory_order_release) != 0 || !self.cancel;
    }

    static bool boundary(llama_context * ctx, void * user) noexcept {
        auto & self = *static_cast<cuda_prefill_resize *>(user);
        try {
            if (++self.boundaries > 2) { return true; }
            constexpr size_t mib = 1024 * 1024;
            constexpr size_t scratch_headroom = 64 * mib;
            const bool shrink = self.boundaries == 1;
            const auto requests = self.read_requests.load(std::memory_order_acquire);
            const size_t staging = self.read_staging.load(std::memory_order_acquire);
            const auto before = discrete_budget_snapshot(*self.contract);
            const bool rearm = self.worker
#ifdef ATOMIC_ENGINE_IO_URING_PROBE
                || self.uring
#endif
                ;
            // A published request is consumed here, never from inside the
            // store's read callback while its unpublished buffer is live.
            if ((shrink || rearm) &&
                    (requests != self.boundaries || staging <= before["ram"]["io"].get<size_t>())) {
                throw std::runtime_error("CUDA prefill resize did not cross a completed staged read");
            }
            size_t free_before = 0, free_after = 0, total = 0;
            ggml_backend_dev_memory(self.device, &free_before, &total);
            const auto start = steady_clock::now();
            const bool applied = llama_discrete_budget_resize(ctx, (shrink ? 3200 : 4096) * mib,
                    (shrink ? 1024 : 2048) * mib, scratch_headroom);
            const double elapsed = seconds(start);
            ggml_backend_dev_memory(self.device, &free_after, &total);
            self.events.push_back({{"microbatch", self.boundaries}, {"staging_while_reading", staging},
                {"mid_read_requests", requests},
                {"applied", applied}, {"before", before}, {"after", discrete_budget_snapshot(*self.contract)},
                {"seconds", elapsed}, {"device_free_before", free_before}, {"device_free_after", free_after}});
#ifdef ATOMIC_ENGINE_IO_URING_PROBE
            if (self.uring) {
                const auto & result = self.read_result;
                self.events.back()["uring_read"] = {{"result", result.read}, {"expected_bytes", result.expected_bytes},
                    {"cancel_result", result.cancel_requested ? json(result.cancel) : json(nullptr)},
                    {"completed_before_cancel", result.completed_before_cancel}};
            }
#endif
            if (applied && shrink && rearm) {
                return self.arm(ctx);
            }
            return applied;
        } catch (const std::exception & e) {
            std::fprintf(stderr, "CUDA prefill resize: %s\n", e.what());
            return false;
        }
    }
};

static json memory_snapshot(const llama_context * ctx) {
    json result;
    if (ctx) {
        result["backend_buffers"] = json::array();
        for (const auto & item : llama_get_memory_breakdown(ctx)) {
            result["backend_buffers"].push_back({{"type", ggml_backend_buft_name(item.first)},
                {"model", item.second.model}, {"context", item.second.context}, {"compute", item.second.compute}});
        }
        llama_expert_streaming_data streaming {};
        if (llama_get_expert_streaming_stats(ctx, &streaming)) {
            result["expert_streaming"] = {{"requested", streaming.requested}, {"applied", streaming.applied},
                {"used", streaming.used}, {"high_water", streaming.high_water}, {"minimum_bytes", streaming.minimum_bytes},
                {"hits", streaming.hits}, {"misses", streaming.misses}, {"evictions", streaming.evictions},
                {"logical_read_bytes", streaming.logical_read_bytes}, {"projections", streaming.projections}, {"matmuls", streaming.matmuls},
                {"buffer_allocations", streaming.buffer_allocations}, {"buffer_reuses", streaming.buffer_reuses}, {"submissions", streaming.submissions}};
        }
        llama_expert_profile_data profile {};
        if (llama_expert_profile_get(ctx, &profile)) {
            result["expert_profile_ns"] = {{"projection", profile.projection_ns}, {"acquire", profile.acquire_ns},
                {"admission", profile.admission_ns}, {"allocation", profile.allocation_ns}, {"read", profile.read_ns},
                {"upload", profile.upload_ns}, {"release", profile.release_ns}, {"graph_setup", profile.graph_setup_ns},
                {"gather", profile.gather_ns}, {"input", profile.input_ns}, {"compute", profile.compute_ns},
                {"readback", profile.readback_ns}, {"scatter", profile.scatter_ns},
                {"submit", profile.submit_ns}, {"wait", profile.wait_ns},
                {"gpu_command", profile.gpu_command_ns}, {"gpu_command_buffers", profile.gpu_command_buffers},
                {"gpu_timed_submissions", profile.gpu_timed_submissions}};
            result["expert_profile_ns"]["map_prepare"] = profile.map_prepare_ns;
            result["expert_profile_ns"]["eviction_search"] = profile.eviction_search_ns;
            result["expert_profile_ns"]["padding"] = profile.padding_ns;
            result["expert_profile_ns"]["padding_bytes"] = profile.padding_bytes;
            result["expert_profile_ns"]["eviction_candidates"] = profile.eviction_candidates;
            result["expert_profile_ns"]["prepared_map_ops"] = profile.prepared_map_ops;
            result["expert_profile_ns"]["route_index"] = profile.route_index_ns;
            result["expert_profile_ns"]["route_indexed_projections"] = profile.route_indexed_projections;
            result["expert_profile_ns"]["route_index_fallbacks"] = profile.route_index_fallbacks;
            result["expert_profile_ns"]["route_id_reads"] = profile.route_id_reads;
            result["expert_profile_ns"]["early_bind"] = profile.early_bind_ns;
            result["expert_profile_ns"]["early_bound_pairs"] = profile.early_bound_pairs;
            result["expert_profile_ns"]["early_windows"] = profile.early_windows;
            result["expert_profile_ns"]["early_acquire"] = profile.early_acquire_ns;
            result["expert_profile_ns"]["early_window_gpu"] = profile.early_window_gpu_ns;
            result["expert_profile_ns"]["early_acquisitions"] = profile.early_acquisitions;
            result["expert_profile_ns"]["early_read_bytes"] = profile.early_read_bytes;
            result["expert_profile_ns"]["early_skip_hot"] = profile.early_skip_hot;
            result["expert_profile_ns"]["early_skip_budget"] = profile.early_skip_budget;
            result["expert_profile_ns"]["early_skip_short"] = profile.early_skip_short;
            result["expert_profile_ns"]["source_setup"] = profile.source_setup_ns;
            result["expert_profile_ns"]["source_validate"] = profile.source_validate_ns;
            result["expert_profile_ns"]["source_seek"] = profile.source_seek_ns;
            result["expert_profile_ns"]["source_transfer"] = profile.source_transfer_ns;
            result["expert_profile_ns"]["direct_read_bytes"] = profile.direct_read_bytes;
            result["expert_profile_ns"]["preview_dot"] = profile.preview_dot_ns;
            result["expert_profile_ns"]["preview_rank"] = profile.preview_rank_ns;
            result["expert_profile_ns"]["preview_residency"] = profile.preview_residency_ns;
            result["expert_profile_ns"]["preview_guard"] = profile.preview_guard_ns;
            result["expert_profile_ns"]["preview_window_host"] = profile.preview_window_host_ns;
            result["expert_profile_ns"]["preview_window_gpu"] = profile.preview_window_gpu_ns;
            result["expert_profile_ns"]["preview_windows"] = profile.preview_windows;
            result["expert_profile_ns"]["preview_competing_windows"] = profile.preview_competing_windows;
            result["expert_profile_ns"]["preview"] = profile.preview_ns;
            result["expert_profile_ns"]["prefetch"] = profile.prefetch_ns;
            result["expert_profile_ns"]["preview_generations"] = profile.preview_generations;
            result["expert_profile_ns"]["preview_rows"] = profile.preview_rows;
            result["expert_profile_ns"]["preview_skip_short"] = profile.preview_skip_short;
            result["expert_profile_ns"]["preview_bind_ns"] = profile.preview_bind_ns;
            result["expert_profile_ns"]["preview_skip_graph"] = profile.preview_skip_graph;
            result["expert_profile_ns"]["preview_skip_shape"] = profile.preview_skip_shape;
            result["expert_profile_ns"]["preview_skip_hot"] = profile.preview_skip_hot;
            result["expert_profile_ns"]["preview_skip_budget"] = profile.preview_skip_budget;
            result["expert_profile_ns"]["prefetch_experts"] = profile.prefetch_experts;
            result["expert_profile_ns"]["prefetch_slices"] = profile.prefetch_slices;
            result["expert_profile_ns"]["prefetch_read_bytes"] = profile.prefetch_read_bytes;
            result["expert_profile_ns"]["overlap_acquire"] = profile.overlap_acquire_ns;
            result["expert_profile_ns"]["overlap_acquisitions"] = profile.overlap_acquisitions;
            result["expert_profile_ns"]["overlap_read_bytes"] = profile.overlap_read_bytes;
        }
        llama_memory_policy_data policy {};
        if (llama_memory_policy_get(ctx, &policy)) {
            result["memory_policy"] = {{"mode", int(policy.config.mode)}, {"level", int(policy.level)},
                {"reason", llama_memory_policy_reason_name(policy.reason)}, {"target", policy.target},
                {"minimum", policy.minimum}, {"preferred", policy.config.preferred}, {"reserve", policy.config.reserve},
                {"headroom", policy.headroom}, {"headroom_valid", policy.headroom_valid},
                {"recovery_delay_ms", policy.recovery_delay_ms}, {"sample_age_ms", policy.sample_age_ms},
                {"decisions", policy.decisions}, {"system_source", policy.system_source}};
        }
        llama_budget_data budget {};
        if (llama_memory_budget_get(ctx, &budget)) {
            result["engine_budget"] = {{"requested", budget.requested}, {"applied", budget.applied},
                {"minimum", budget.minimum}, {"used", budget.used}, {"high_water", budget.high_water},
                {"released", budget.released}, {"runtime_envelope", budget.runtime_envelope},
                {"request_id", budget.request_id}, {"completed_id", budget.completed_id}, {"status", int(budget.status)},
                {"categories", { {"weights", budget.categories[0]}, {"experts", budget.categories[1]},
                    {"state", budget.categories[2]}, {"scratch", budget.categories[3]}, {"drafter", budget.categories[4]},
                    {"io", budget.categories[5]}, {"runtime", budget.categories[6]} }}};
        }
    }
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        if (std::string(ggml_backend_dev_name(dev)) == "MTL0") {
            size_t free = 0, total = 0;
            ggml_backend_dev_memory(dev, &free, &total);
            if (free > 0) { result["metal_allocated_bytes"] = total - free; }
        }
    }
#if defined(__APPLE__)
    task_vm_info_data_t info {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        result["rss_bytes"] = info.resident_size;
        result["physical_footprint_bytes"] = info.phys_footprint;
        result["rss_peak_bytes"] = info.resident_size_peak;
        result["physical_footprint_peak_bytes"] = info.ledger_phys_footprint_peak;
    }
    rusage_info_v4 io {};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_V4, reinterpret_cast<rusage_info_t *>(&io)) == 0) {
        result["os_charged_process_read_bytes"] = io.ri_diskio_bytesread;
    }
#endif
    return result;
}

// Caller-owned buffered output; no growing in-engine trace allocation.
struct expert_trace_file {
    std::unique_ptr<FILE, decltype(&std::fclose)> file {nullptr, std::fclose};
    bool failed = false;
    static void callback(const llama_expert_trace_event & e, void * user) noexcept {
        auto & self = *static_cast<expert_trace_file *>(user);
        const char kind = e.kind == llama_expert_trace_kind::projection ? 'P' :
                          e.kind == llama_expert_trace_kind::access ? 'A' :
                          e.kind == llama_expert_trace_kind::configuration ? 'C' :
                          e.kind == llama_expert_trace_kind::release ? 'R' : 'X';
        if (std::fprintf(self.file.get(), "%c %zu %lld %lld %zu %zu %zu %d %zu\n", kind,
                e.tensor, (long long) e.expert, (long long) e.tokens, e.payload,
                e.allocation, e.capacity, int(e.hit), e.required) < 0) { self.failed = true; }
    }
};

struct prefill_resize {
    llama_context * ctx = nullptr;
    size_t target = 0;
    uint64_t request_id = 0;
    bool pressure = false, signals_sent = false;
    static bool callback(ggml_tensor * tensor, bool ask, void * user) {
        auto & self = *static_cast<prefill_resize *>(user);
        // A model may start with dense layers; trigger on its first routed FFN.
        if (ask) { return self.ctx && ((self.target && !self.request_id) || (self.pressure && !self.signals_sent)) && std::strncmp(tensor->name, "ffn_moe_down-", 13) == 0; }
        if (self.target && !self.request_id) { self.request_id = llama_memory_budget_request(self.ctx, self.target); }
        if (self.pressure && !self.signals_sent) {
            llama_memory_pressure_sample critical {llama_memory_pressure::critical, 0, true};
            llama_memory_pressure_sample normal {llama_memory_pressure::normal, size_t(4) * 1024 * 1024 * 1024, true};
            llama_memory_pressure_submit(self.ctx, &critical);
            llama_memory_pressure_submit(self.ctx, &normal);
            self.signals_sent = true;
        }
        return true;
    }
};

static std::string digest_bytes(const uint8_t * bytes, size_t n) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (size_t i = 0; i < n; ++i) { hash = (hash ^ bytes[i]) * UINT64_C(1099511628211); }
    char text[32];
    std::snprintf(text, sizeof(text), "%016llx", (unsigned long long) hash);
    return text;
}

struct tensor_trace {
    std::ofstream file;
    std::vector<uint8_t> scratch;
    bool active = false;

    static bool inspect(ggml_tensor * tensor, bool ask, void * user) {
        auto & self = *static_cast<tensor_trace *>(user);
        if (!self.active) { return !ask; }
        const size_t bytes = ggml_nbytes(tensor);
        if (ask) {
            return bytes > 0 && bytes <= self.scratch.size() && ggml_is_contiguous(tensor) &&
                (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) &&
                std::strncmp(tensor->name, "node_", 5) != 0;
        }
        ggml_backend_tensor_get(tensor, self.scratch.data(), 0, bytes);
        self.file << tensor->name << '\t' << ggml_op_name(tensor->op) << '\t' << bytes << '\t'
                  << digest_bytes(self.scratch.data(), bytes) << '\n';
        return bool(self.file);
    }
};

static std::string state_digest(llama_context * ctx, const llama_memory_contract * discrete = nullptr) {
    const auto size = llama_state_get_size(ctx);
    llama_memory_budget::ticket ticket;
    if (discrete) {
        ticket = discrete->ledger->reserve(llama_memory_domain::host, llama_memory_category::state, size);
        if (!ticket) { throw std::runtime_error("state capture exceeds the RAM budget"); }
    }
    std::vector<uint8_t> bytes(size);
    if (ticket) { ticket.commit(); }
    const size_t n = llama_state_get_data(ctx, bytes.data(), bytes.size());
    if (!n || n > bytes.size()) { throw std::runtime_error("state capture failed"); }
    return digest_bytes(bytes.data(), n);
}

static void set_batch(llama_batch_ext * batch, const std::vector<llama_token> & tokens, int position) {
    llama_batch_ext_clear(batch);
    for (auto token : tokens) {
        const int idx = llama_batch_ext_add_token(batch, 0, token);
        llama_pos pos = position++;
        llama_batch_ext_set_pos(batch, idx, &pos);
    }
    llama_batch_ext_set_output_logits(batch, int(tokens.size()) - 1, true);
}

static llama_memory_budget::ticket reserve_probe(const llama_model & model, llama_context * ctx, size_t size) {
    auto reserve_snapshot = [&] {
        return model.memory_contract->ledger->reserve(
            llama_memory_domain::host, llama_memory_category::drafter, size);
    };
    auto ticket = reserve_snapshot();
    if (!ticket && size) {
        llama_budget_data budget {};
        if (!llama_memory_budget_get(ctx, &budget) || size > budget.applied ||
                budget.applied - size < budget.minimum ||
                !llama_memory_budget_request(ctx, budget.applied - size) || !llama_memory_budget_apply(ctx) ||
                !llama_memory_budget_request(ctx, budget.applied) || !llama_memory_budget_apply(ctx)) {
            throw std::runtime_error("rollback snapshot cannot reclaim sufficient expert cache");
        }
        ticket = reserve_snapshot();
    }
    if (!ticket) { throw std::runtime_error("rollback snapshot does not fit the shared budget"); }
    return ticket;
}

static json probe_rollback(const llama_model & model, llama_context * ctx,
        llama_batch_ext * batch, int position, llama_token accepted) {
    const size_t size = llama_state_get_size(ctx);
    auto ticket = reserve_probe(model, ctx, size);
    std::vector<uint8_t> snapshot(size);
    ticket.commit();
    const size_t saved = llama_state_get_data(ctx, snapshot.data(), size);
    if (!saved || saved > size) { throw std::runtime_error("rollback snapshot failed"); }
    const auto before = digest_bytes(snapshot.data(), saved);
    // A deliberately rejected token exercises the target's actual hybrid state.
    const llama_token rejected = accepted == 0 ? 1 : 0;
    set_batch(batch, {rejected}, position);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch) != 0 ||
            llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != position) {
        throw std::runtime_error("rejected token did not advance the target sequence");
    }
    if (llama_state_set_data(ctx, snapshot.data(), saved) != saved) {
        throw std::runtime_error("rollback restore failed");
    }
    const size_t restored = llama_state_get_data(ctx, snapshot.data(), size);
    if (restored != saved || digest_bytes(snapshot.data(), restored) != before) {
        throw std::runtime_error("rollback changed the serialized target state");
    }
    return {{"snapshot_bytes", size}, {"serialized_bytes", saved}, {"state", before},
        {"rejected_id", rejected}, {"accepted_id", accepted}, {"position", position},
        {"memory_while_charged", memory_snapshot(ctx)}};
}

struct draft_admission_probe {
    std::unique_ptr<llama_model, decltype(&llama_model_free)> model {nullptr, llama_model_free};
    std::unique_ptr<llama_context, decltype(&llama_free)> ctx {nullptr, llama_free};
};

static std::string float_digest(const float * data, size_t count) {
    if (!data) { throw std::runtime_error("probe output row missing"); }
    return digest_bytes(reinterpret_cast<const uint8_t *>(data), count * sizeof(float));
}

static void add_draft_input(llama_batch_ext * batch, llama_token token, llama_pos pos,
        const float * hidden, size_t width, bool output) {
    const int idx = llama_batch_ext_add_token(batch, 0, token);
    if (idx < 0 || !llama_batch_ext_set_pos(batch, idx, &pos) ||
            !llama_batch_ext_set_embd_state(batch, idx, {hidden, 1, width}) ||
            !llama_batch_ext_set_output_logits(batch, idx, output)) {
        throw std::runtime_error("cannot bind draft input and target hidden row");
    }
}

// The eval callback adds synchronization boundaries. Use its rows only for
// localization, and keep commit decisions on the uninstrumented graph.
struct mtp_layer_probe {
    static constexpr std::array names = {"attn_norm-0", "linear_attn_qkv_mixed-0",
        "beta-0", "beta_sigmoid-0", "a_softplus-0", "conv_output_raw-0",
        "linear_attn_out-0", "ffn_moe_out-0", "l_out-0", "l_out-1", "result_output"};
    bool active = false;
    json rows = json::array();
    std::array<float, 1 << 16> scratch;

    static bool inspect(ggml_tensor * tensor, bool ask, void * user) {
        auto & self = *static_cast<mtp_layer_probe *>(user);
        if (!self.active || !std::any_of(names.begin(), names.end(), [&](const char * name) {
                return std::strcmp(tensor->name, name) == 0;
            })) {
            return !ask;
        }
        if (ask) { return true; }
        if (tensor->type != GGML_TYPE_F32 || tensor->nb[0] != sizeof(float) ||
                tensor->nb[1] != tensor->ne[0] * sizeof(float)) {
            return true;
        }
        // Beta has a head dimension before the token dimension.
        const bool token_axis_2 = std::strncmp(tensor->name, "beta", 4) == 0;
        const size_t count = tensor->ne[0] * (token_axis_2 ? tensor->ne[1] : 1);
        if (count > self.scratch.size() || (token_axis_2 && tensor->nb[2] != count * sizeof(float))) {
            return true;
        }
        ggml_backend_tensor_get(tensor, self.scratch.data(), 0, count * sizeof(float));
        self.rows.push_back({{"name", tensor->name}, {"count", count},
            {"hash", float_digest(self.scratch.data(), count)},
            {"shape", {tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]}},
            {"op", ggml_op_name(tensor->op)}});
        return true;
    }
};

static json probe_mtp(const llama_model & model, llama_context * target, llama_batch_ext * target_batch,
        llama_context * draft, const std::vector<llama_token> & prompt, llama_token seed, size_t vocab_size,
        bool try_batch_commit, mtp_layer_probe * layers) {
    const size_t width = llama_model_n_embd_out(&model);
    const llama_pos pos = prompt.size();
    auto ledger = model.memory_contract->ledger;
    auto capture = [](llama_context * ctx, std::vector<uint8_t> & data) {
        const size_t n = llama_state_get_data(ctx, data.data(), data.size());
        if (n != data.size() || !n) { throw std::runtime_error("MTP snapshot size mismatch"); }
        return digest_bytes(data.data(), n);
    };
    auto restore = [&](llama_context * ctx, std::vector<uint8_t> & data, const std::string & hash) {
        if (llama_state_set_data(ctx, data.data(), data.size()) != data.size() ||
                capture(ctx, data) != hash || llama_memory_seq_pos_max(llama_get_memory(ctx), 0) != pos - 1) {
            throw std::runtime_error("MTP snapshot restore mismatch");
        }
    };
    const size_t target_size = llama_state_get_size(target);
    auto target_ticket = reserve_probe(model, target, target_size);
    std::vector<uint8_t> target_state(target_size);
    target_ticket.commit();
    const auto target_hash = capture(target, target_state);

    // Capturing every hidden row changes the graph shape. Replay the prompt
    // speculatively and keep the original target state for verification.
    llama_memory_clear(llama_get_memory(target), true);
    llama_set_embeddings_nextn(target, true, false);
    set_batch(target_batch, prompt, 0);
    if (llama_process(target, LLAMA_PROCESS_TYPE_DECODE, target_batch) != 0) {
        throw std::runtime_error("target hidden-state replay failed");
    }

    // Batch state rows are copied by the batch API. Keep headroom for both
    // vector growth and the zero start row during draft prefill.
    auto row_ticket = reserve_probe(model, target, (4 * prompt.size() + 1) * width * sizeof(float));
    std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(draft), llama_batch_ext_free);
    std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> greedy(llama_sampler_init_greedy(), llama_sampler_free);
    std::vector<float> zero(width);
    row_ticket.commit();
    const float * rows = llama_get_embeddings_nextn(target);
    if (!rows) { throw std::runtime_error("target nextn rows missing"); }
    for (size_t i = 0; i < prompt.size(); ++i) {
        add_draft_input(batch.get(), prompt[i], i, i ? rows + (i - 1) * width : zero.data(), width, false);
    }
    const float * hidden = rows + (prompt.size() - 1) * width;
    if (!std::all_of(hidden, hidden + width, [](float x) { return std::isfinite(x); }) ||
            std::all_of(hidden, hidden + width, [](float x) { return x == 0; })) {
        throw std::runtime_error("invalid MTP seed hidden row");
    }
    const auto hidden_hash = float_digest(hidden, width);
    auto draft_decode = [&] {
        llama_allocation_scope admission(ledger, llama_memory_category::drafter);
        if (llama_process(draft, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            throw std::runtime_error("MTP decode failed");
        }
        llama_synchronize(draft);
    };
    draft_decode();
    if (llama_memory_seq_pos_max(llama_get_memory(draft), 0) != pos - 1) {
        throw std::runtime_error("draft prompt alignment failed");
    }
    const size_t draft_size = llama_state_get_size(draft);
    auto draft_ticket = reserve_probe(model, target, draft_size);
    std::vector<uint8_t> draft_state(draft_size);
    draft_ticket.commit();
    const auto draft_hash = capture(draft, draft_state);
    auto propose = [&] {
        llama_batch_ext_clear(batch.get());
        add_draft_input(batch.get(), seed, pos, hidden, width, true);
        draft_decode();
        if (llama_memory_seq_pos_max(llama_get_memory(draft), 0) != pos) {
            throw std::runtime_error("MTP proposal did not advance draft state");
        }
        return llama_sampler_sample(greedy.get(), draft, -1);
    };
    const llama_token proposal = propose();
    const auto draft_logits = float_digest(llama_get_logits_ith(draft, -1), vocab_size);
    restore(draft, draft_state, draft_hash);
    if (propose() != proposal || float_digest(llama_get_logits_ith(draft, -1), vocab_size) != draft_logits) {
        throw std::runtime_error("draft rollback changed the repeated proposal");
    }
    restore(draft, draft_state, draft_hash);
    llama_set_embeddings_nextn(target, false, false);
    restore(target, target_state, target_hash);

    if (try_batch_commit) {
        const auto * vocab = llama_model_get_vocab(&model);
        json result = {{"seed", seed}, {"proposal", proposal}, {"seed_hidden_hash", hidden_hash},
            {"draft_restored_state", draft_hash}, {"batch_committed", false}};
        if (llama_vocab_is_eog(vocab, proposal)) {
            result["decision"] = "proposal_is_eog";
            return result;
        }
        auto logits_ticket = reserve_probe(model, target, 2 * vocab_size * sizeof(float));
        std::vector<float> serial_logits(2 * vocab_size);
        logits_ticket.commit();
        auto serial_step = [&](llama_token token, llama_pos position, float * output) {
            set_batch(target_batch, {token}, position);
            if (llama_process(target, LLAMA_PROCESS_TYPE_DECODE, target_batch) != 0 ||
                    llama_memory_seq_pos_max(llama_get_memory(target), 0) != position) {
                throw std::runtime_error("serial control decode failed");
            }
            const auto * logits = llama_get_logits_ith(target, -1);
            if (!logits) { throw std::runtime_error("serial control logits missing"); }
            std::memcpy(output, logits, vocab_size * sizeof(float));
        };
        if (layers) { layers->active = true; }
        serial_step(seed, pos, serial_logits.data());
        if (layers) {
            layers->active = false;
            result["serial_layers"] = std::move(layers->rows);
            layers->rows = json::array();
        }
        const auto serial_id = llama_sampler_sample(greedy.get(), target, -1);
        result["serial_id"] = serial_id;
        if (proposal != serial_id) {
            restore(target, target_state, target_hash);
            result["decision"] = "proposal_differs_from_serial";
            return result;
        }
        serial_step(proposal, pos + 1, serial_logits.data() + vocab_size);
        const size_t serial_size = llama_state_get_size(target);
        // Two host copies fit with single-token workspace. Keep the serial
        // reference on disk for the larger, uninstrumented verifier.
        const bool compare_in_ram = !layers && llama_n_ubatch(target) == 1;
        auto state_ticket = reserve_probe(model, target, (compare_in_ram ? 2 : 1) * serial_size);
        std::vector<uint8_t> comparison_state(serial_size);
        std::vector<uint8_t> serial_bytes(compare_in_ram ? serial_size : 0);
        state_ticket.commit();
        const auto serial_state = capture(target, compare_in_ram ? serial_bytes : comparison_state);
        std::unique_ptr<std::FILE, decltype(&std::fclose)> serial_file(nullptr, std::fclose);
        if (!layers && !compare_in_ram) {
            serial_file.reset(std::tmpfile());
            if (!serial_file || std::fwrite(comparison_state.data(), 1, serial_size, serial_file.get()) != serial_size ||
                    std::fseek(serial_file.get(), 0, SEEK_SET) != 0) {
                throw std::runtime_error("cannot save serial state for byte comparison");
            }
        }
        restore(target, target_state, target_hash);
        set_batch(target_batch, {seed, proposal}, pos);
        if (layers) { layers->active = true; }
        if (!llama_batch_ext_set_output_logits(target_batch, 0, true) ||
                llama_process(target, LLAMA_PROCESS_TYPE_DECODE, target_batch) != 0 ||
                llama_memory_seq_pos_max(llama_get_memory(target), 0) != pos + 1) {
            throw std::runtime_error("non-EOG batch verification failed");
        }
        if (layers) {
            layers->active = false;
            result["batch_layers"] = std::move(layers->rows);
        }
        const auto verified_id = llama_sampler_sample(greedy.get(), target, 0);
        const auto serial_next = std::max_element(serial_logits.data() + vocab_size,
            serial_logits.data() + 2 * vocab_size) - (serial_logits.data() + vocab_size);
        const auto verified_next = llama_sampler_sample(greedy.get(), target, 1);
        const size_t batch_size = llama_state_get_size(target);
        const auto batch_state = batch_size == serial_size ? capture(target, comparison_state) : std::string{};
        bool exact_logits = true;
        result["logit_rows"] = json::array();
        for (int row = 0; row < 2; ++row) {
            const float * actual = llama_get_logits_ith(target, row);
            if (!actual) { throw std::runtime_error("verification logits missing"); }
            const float * expected = serial_logits.data() + row * vocab_size;
            size_t different = 0;
            double max_abs = 0;
            for (size_t i = 0; i < vocab_size; ++i) {
                different += std::memcmp(actual + i, expected + i, sizeof(float)) != 0;
                max_abs = std::max(max_abs, std::abs(double(actual[i]) - expected[i]));
            }
            exact_logits &= different == 0;
            result["logit_rows"].push_back({{"different_values", different}, {"max_abs_difference", max_abs}});
        }
        bool exact_state = !layers && serial_size == batch_size;
        if (exact_state && compare_in_ram && serial_bytes != comparison_state) {
            exact_state = false;
            result["first_state_difference"] = std::mismatch(serial_bytes.begin(), serial_bytes.end(),
                comparison_state.begin()).first - serial_bytes.begin();
        }
        std::array<uint8_t, 4096> chunk;
        for (size_t offset = 0; exact_state && serial_file && offset < batch_size; offset += chunk.size()) {
            const size_t n = std::min(chunk.size(), batch_size - offset);
            if (std::fread(chunk.data(), 1, n, serial_file.get()) != n) {
                throw std::runtime_error("cannot read serial state for byte comparison");
            }
            const auto mismatch = std::mismatch(chunk.data(), chunk.data() + n, comparison_state.data() + offset);
            if (mismatch.first != chunk.data() + n) {
                exact_state = false;
                result["first_state_difference"] = offset + (mismatch.first - chunk.data());
            }
        }
        const bool exact = verified_id == proposal && verified_next == serial_next && exact_logits && exact_state;
        const bool commit = exact && !layers;
        result.update({{"verification_rows", 2}, {"verified_id", verified_id},
            {"serial_next_id", serial_next}, {"batch_next_id", verified_next},
            {"serial_state_bytes", serial_size}, {"batch_state_bytes", batch_size},
            {"serial_state", serial_state}, {"batch_state", batch_state},
            {"state_bytes_equal", layers ? json(nullptr) : json(exact_state)},
            {"batch_committed", commit}, {"decision", layers ? "layer_trace" : commit ? "exact" : "serial_fallback"},
            {"memory_while_charged", memory_snapshot(target)}});
        if (!commit) { restore(target, target_state, target_hash); }
        return result;
    }

    // Treat the entire two-row evaluation as tentative, including a proposed
    // EOS. Commit through the existing one-token decode after restoring target.
    set_batch(target_batch, {seed, proposal}, pos);
    if (!llama_batch_ext_set_output_logits(target_batch, 0, true) ||
            llama_process(target, LLAMA_PROCESS_TYPE_DECODE, target_batch) != 0 ||
            llama_memory_seq_pos_max(llama_get_memory(target), 0) != pos + 1) {
        throw std::runtime_error("two-row MTP verification failed");
    }
    const auto verified_id = llama_sampler_sample(greedy.get(), target, 0);
    const auto verify_logits = float_digest(llama_get_logits_ith(target, 0), vocab_size);
    const auto suffix_logits = float_digest(llama_get_logits_ith(target, 1), vocab_size);
    restore(target, target_state, target_hash);
    // Replay a deliberately wrong candidate to cover the verifier's reject
    // case independently of whether this model's real proposal matches.
    const llama_token wrong = verified_id == 0 ? 1 : 0;
    set_batch(target_batch, {seed, wrong}, pos);
    if (!llama_batch_ext_set_output_logits(target_batch, 0, true) ||
            llama_process(target, LLAMA_PROCESS_TYPE_DECODE, target_batch) != 0 ||
            llama_sampler_sample(greedy.get(), target, 0) != verified_id ||
            llama_memory_seq_pos_max(llama_get_memory(target), 0) != pos + 1) {
        throw std::runtime_error("negative pair changed the verifier's prefix decision");
    }
    restore(target, target_state, target_hash);
    return {{"seed", seed}, {"proposal", proposal}, {"verified_id", verified_id},
        {"negative_control_proposal", wrong},
        {"accepted_by_verifier", proposal == verified_id}, {"verification_rows", 2},
        {"hidden_width", width}, {"hidden_rows_transferred", prompt.size() + 2},
        {"seed_hidden_hash", hidden_hash}, {"draft_logits_hash", draft_logits},
        {"verify_logits_hash", verify_logits}, {"suffix_logits_hash", suffix_logits},
        {"target_snapshot_bytes", target_size}, {"draft_snapshot_bytes", draft_size},
        {"target_restored_state", target_hash}, {"draft_restored_state", draft_hash},
        {"memory_while_charged", memory_snapshot(target)}};
}

int main(int argc, char ** argv) {
    const auto process_start = steady_clock::now();
    try {
        std::string path, output, logits_path, trace_path, cache_policy = "lru";
        bool explicit_cache_policy = false;
        std::string preview_path, tensor_trace_path;
        std::string draft_admission_path;
        expert_preview_trace preview;
        std::string prompt = "The capital of France is";
        int tokens = 8, context = 512, ubatch = 16, repeat = 1, warmup = 0, gpu_layers = 99;
        size_t store_mib = 0, budget_mib = 0;
        size_t ram_mib = 0, vram_mib = 0;
        std::string cuda_device;
        bool cuda_expert_streaming = false;
        bool cuda_experts_match_layer = false;
        bool cuda_boundary_reduction = false;
        bool cuda_resize_probe = false;
        bool cuda_prefill_resize_probe = false;
        bool cuda_uring_read = false;
        size_t native_moe_cache_mib = 0;
        std::string cuda_worker_read;
        bool capture_state = false, profile_experts = false, rollback_probe = false, mtp_probe = false;
        bool mtp_batch_commit_probe = false;
        bool mtp_trace_layers = false;
        int read_overlap = -1;
        int single_map = -1;
        int route_index = -1;
        int early_read = -1;
        int preview_mode = 0;
        prefill_resize prefill;
        std::string policy_mode;
        llama_memory_policy_config policy_config;
        struct signal_event { int step; llama_memory_pressure_sample sample; };
        std::vector<signal_event> signals;
        int resume_at = 0;
        std::vector<std::pair<int, size_t>> resizes;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            auto value = [&]() -> std::string {
                if (++i == argc) { throw std::invalid_argument("missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--model") { path = value(); }
            else if (arg == "--output") { output = value(); }
            else if (arg == "--logits") { logits_path = value(); }
            else if (arg == "--expert-trace") { trace_path = value(); }
            else if (arg == "--cache-policy") { cache_policy = value(); explicit_cache_policy = true; }
            else if (arg == "--prompt") { prompt = value(); }
            else if (arg == "--prompt-file") {
                std::ifstream file(value());
                if (!file) { throw std::runtime_error("cannot open prompt file"); }
                prompt.assign(std::istreambuf_iterator<char>(file), {});
            }
            else if (arg == "--tokens") { tokens = std::stoi(value()); }
            else if (arg == "--ctx") { context = std::stoi(value()); }
            else if (arg == "--ubatch") { ubatch = std::stoi(value()); }
            else if (arg == "--repeat") { repeat = std::stoi(value()); }
            else if (arg == "--warmup") { warmup = std::stoi(value()); }
            else if (arg == "--gpu-layers") { gpu_layers = std::stoi(value()); }
            else if (arg == "--cuda-device") { cuda_device = value(); }
            else if (arg == "--cuda-expert-streaming") { cuda_expert_streaming = true; }
            else if (arg == "--cuda-experts-match-layer") { cuda_experts_match_layer = true; }
            else if (arg == "--cuda-boundary-reduction") { cuda_boundary_reduction = true; }
            else if (arg == "--cuda-resize-probe") { cuda_resize_probe = true; }
            else if (arg == "--cuda-prefill-resize-probe") { cuda_prefill_resize_probe = true; }
            else if (arg == "--cuda-uring-read") { cuda_uring_read = true; }
            else if (arg == "--native-moe-cache-mib") {
                const auto input = value();
                size_t parsed = 0;
                const int n = std::stoi(input, &parsed);
                if (parsed != input.size() || n <= 0 || size_t(n) > SIZE_MAX / (1024 * 1024)) {
                    throw std::invalid_argument("invalid native MoE cache size");
                }
                native_moe_cache_mib = size_t(n);
            }
            else if (arg == "--cuda-worker-read") {
                cuda_worker_read = value();
                if (cuda_worker_read != "complete" && cuda_worker_read != "cancel") {
                    throw std::invalid_argument("CUDA worker read must be complete or cancel");
                }
            }
            else if (arg == "--ram-mib" || arg == "--vram-mib") {
                const auto input = value();
                size_t parsed = 0;
                const int n = std::stoi(input, &parsed);
                const size_t max_mib = SIZE_MAX / (1024 * 1024);
                if (parsed != input.size() || n <= 0 || size_t(n) > max_mib) {
                    throw std::invalid_argument("invalid discrete budget; expected positive integer MiB");
                }
                (arg == "--ram-mib" ? ram_mib : vram_mib) = size_t(n);
            }
            else if (arg == "--profile-experts") { profile_experts = true; }
            else if (arg == "--preview-trace") { preview_path = value(); }
            else if (arg == "--tensor-trace") { tensor_trace_path = value(); }
            else if (arg == "--expert-preview") {
                const auto mode = value();
                if (mode != "off" && mode != "preview" && mode != "prefetch" && mode != "prefetch-first") { throw std::invalid_argument("expert preview must be off, preview, prefetch or prefetch-first"); }
                preview_mode = mode == "prefetch-first" ? 3 : mode == "prefetch" ? 2 : mode == "preview" ? 1 : 0;
            }
            else if (arg == "--expert-read-overlap") {
                const auto mode = value();
                if (mode != "on" && mode != "off") { throw std::invalid_argument("expert read overlap must be on or off"); }
                read_overlap = mode == "on";
            }
            else if (arg == "--expert-single-map") {
                const auto mode = value();
                if (mode != "on" && mode != "off") { throw std::invalid_argument("expert single map must be on or off"); }
                single_map = mode == "on";
            }
            else if (arg == "--expert-early-read") {
                const auto mode = value();
                if (mode != "on" && mode != "off") { throw std::invalid_argument("expert early read must be on or off"); }
                early_read = mode == "on";
            }
            else if (arg == "--expert-route-index") {
                const auto mode = value();
                if (mode != "on" && mode != "off") { throw std::invalid_argument("expert route index must be on or off"); }
                route_index = mode == "on";
            }
            else if (arg == "--memory-policy") { policy_mode = value(); }
            else if (arg == "--pressure-prefill") { prefill.pressure = true; }
            else if (arg == "--auto-resume-at") { resume_at = std::stoi(value()); }
            else if (arg == "--memory-recovery-ms") {
                policy_config.recovery_ms = std::stoull(value());
                policy_config.interval_ms = std::min(uint64_t(1000), policy_config.recovery_ms);
            }
            else if (arg == "--memory-min-mib") { policy_config.minimum = std::stoull(value()) * 1024 * 1024; }
            else if (arg == "--pressure-at") {
                std::istringstream spec(value());
                std::string step, level, free;
                if (!std::getline(spec, step, ':') || !std::getline(spec, level, ':') || !std::getline(spec, free)) {
                    throw std::invalid_argument("pressure syntax is committed_count:normal|warning|critical|unknown:headroom_MiB");
                }
                llama_memory_pressure_sample sample;
                if (level == "normal") { sample.level = llama_memory_pressure::normal; }
                else if (level == "warning") { sample.level = llama_memory_pressure::warning; }
                else if (level == "critical") { sample.level = llama_memory_pressure::critical; }
                else if (level != "unknown") { throw std::invalid_argument("invalid pressure level"); }
                sample.headroom = std::stoull(free) * 1024 * 1024;
                sample.headroom_valid = level != "unknown";
                signals.push_back({std::stoi(step), sample});
            }
            else if (arg == "--state-digest") { capture_state = true; }
            else if (arg == "--rollback-probe") { rollback_probe = true; }
            else if (arg == "--mtp-probe") { mtp_probe = true; }
            else if (arg == "--mtp-batch-commit-probe") { mtp_probe = mtp_batch_commit_probe = true; }
            else if (arg == "--mtp-trace-layers") { mtp_trace_layers = true; }
            else if (arg == "--draft-admission-probe") { draft_admission_path = value(); }
            else if (arg == "--budget-mib" || arg == "--prefill-budget-mib") {
                const int n = std::stoi(value());
                if (n <= 0) { throw std::invalid_argument("budget must be positive"); }
                if (arg == "--budget-mib") { budget_mib = size_t(n); }
                else { prefill.target = size_t(n) * 1024 * 1024; }
            }
            else if (arg == "--resize-at") {
                const auto spec = value();
                const auto colon = spec.find(':');
                if (colon == std::string::npos) { throw std::invalid_argument("resize syntax is committed_count:MiB"); }
                const int step = std::stoi(spec.substr(0, colon));
                const int mib = std::stoi(spec.substr(colon + 1));
                if (step < 1 || mib < 0) { throw std::invalid_argument("invalid resize schedule"); }
                resizes.emplace_back(step, size_t(mib) * 1024 * 1024);
            }
            else if (arg == "--store-mib") {
                const int n = std::stoi(value());
                if (n <= 0) { throw std::invalid_argument("--store-mib must be positive"); }
                store_mib = size_t(n);
            }
            else { throw std::invalid_argument("unknown argument: " + arg); }
        }
        if (path.empty() || output.empty() || tokens < 1 || context < 1 || ubatch < 1 || repeat < 1 || warmup < 0) {
            throw std::invalid_argument("usage: llama-expert-infer --model MODEL --output REPORT.json [--tokens N --ctx N --ubatch N --repeat N --warmup N --logits FILE]");
        }
        const bool discrete = ram_mib || vram_mib;
        if (native_moe_cache_mib && (budget_mib || store_mib || discrete || !cuda_device.empty() ||
                !gpu_layers || rollback_probe || mtp_probe)) {
            throw std::invalid_argument("native MoE cache control requires ordinary GPU inference without a streaming budget, rollback or MTP probe");
        }
        if (!cuda_worker_read.empty() && !cuda_prefill_resize_probe) {
            throw std::invalid_argument("--cuda-worker-read requires --cuda-prefill-resize-probe");
        }
        if (cuda_uring_read && (!cuda_prefill_resize_probe || !cuda_worker_read.empty())) {
            throw std::invalid_argument("--cuda-uring-read requires --cuda-prefill-resize-probe and excludes --cuda-worker-read");
        }
#ifndef ATOMIC_ENGINE_IO_URING_PROBE
        if (cuda_uring_read) {
            throw std::invalid_argument("--cuda-uring-read requires a build with ATOMIC_ENGINE_IO_URING_PROBE=ON");
        }
#endif
        if (cuda_resize_probe && (!cuda_expert_streaming || ram_mib != 4096 || vram_mib != 2048 ||
                tokens < 3 || repeat != 1 || warmup || !tensor_trace_path.empty())) {
            throw std::invalid_argument("CUDA resize probe requires one streaming request, 4 GiB RAM, 2 GiB VRAM and at least three tokens");
        }
        if (cuda_prefill_resize_probe && (!cuda_expert_streaming || ram_mib != 4096 || vram_mib != 2048 ||
                cuda_resize_probe || repeat != 1 || warmup || !tensor_trace_path.empty())) {
            throw std::invalid_argument("CUDA prefill resize probe requires one streaming request, 4 GiB RAM and 2 GiB VRAM");
        }
        if (cuda_boundary_reduction && !cuda_experts_match_layer) {
            throw std::invalid_argument("--cuda-boundary-reduction requires --cuda-experts-match-layer");
        }
        if (cuda_experts_match_layer && !cuda_expert_streaming) {
            throw std::invalid_argument("--cuda-experts-match-layer requires --cuda-expert-streaming");
        }
        if (!tensor_trace_path.empty() && (warmup || repeat != 1 || mtp_probe || prefill.target || prefill.pressure)) {
            throw std::invalid_argument("tensor tracing requires a single request without another evaluation callback");
        }
        if (cuda_expert_streaming && (!discrete || ubatch > 32)) {
            throw std::invalid_argument("--cuda-expert-streaming requires fixed RAM/VRAM caps and ubatch <= 32");
        }
        if (discrete && (!ram_mib || !vram_mib || cuda_device.empty() || budget_mib || store_mib || !resizes.empty() ||
                !trace_path.empty() || !preview_path.empty() || preview_mode || profile_experts || read_overlap >= 0 ||
                !policy_mode.empty() || prefill.target || prefill.pressure || cache_policy != "lru" || rollback_probe || mtp_probe)) {
            throw std::invalid_argument("discrete diagnostics require both caps and --cuda-device, with no streaming or resize options");
        }
        expert_trace_file trace;
        if (!trace_path.empty()) {
            if (warmup || repeat != 1 || prefill.target || !resizes.empty() || !policy_mode.empty()) {
                throw std::invalid_argument("expert trace requires one fixed-budget request without warmup or resize");
            }
            trace.file.reset(std::fopen(trace_path.c_str(), "wb"));
            if (!trace.file || std::fprintf(trace.file.get(), "ATET3\n") < 0) {
                throw std::runtime_error("cannot open expert trace");
            }
        }
        if ((!signals.empty() || prefill.pressure || resume_at) && policy_mode != "replay") {
            throw std::invalid_argument("injected signals require --memory-policy replay");
        }
        if ((rollback_probe || mtp_probe) && (!budget_mib || tokens < 2 || warmup || repeat != 1 ||
                !trace_path.empty() || !preview_path.empty() || !policy_mode.empty() ||
                prefill.target || prefill.pressure || !resizes.empty() || preview_mode)) {
            throw std::invalid_argument("rollback probe requires one fixed-budget request with decode and no other probes");
        }
        if (!draft_admission_path.empty() && !rollback_probe && !mtp_probe) {
            throw std::invalid_argument("draft admission probe requires --rollback-probe or --mtp-probe");
        }
        if (mtp_probe && (rollback_probe || draft_admission_path.empty() || budget_mib != 12288 ||
                !gpu_layers || (!mtp_batch_commit_probe && ubatch < 2) || (mtp_batch_commit_probe && tokens < 3))) {
            throw std::invalid_argument("MTP probe requires a draft GGUF, Metal and a 12 GiB budget");
        }
        if (mtp_trace_layers && (!mtp_batch_commit_probe || ubatch < 2)) {
            throw std::invalid_argument("MTP layer trace requires a batch commit probe with a multi-row microbatch");
        }
        if (!policy_mode.empty() && policy_mode != "replay" && policy_mode != "system") {
            throw std::invalid_argument("memory policy must be replay or system");
        }
        llama_backend_init();
        ggml_backend_load_all();
        json report = {{"model", path.substr(path.find_last_of("/\\") + 1)}, {"prompt", prompt},
            {"context", context}, {"ubatch", ubatch}, {"gpu_layers", gpu_layers}, {"output_limit", tokens},
            {"warmup_requests", warmup}, {"sampling", "greedy"}, {"chat_template", false},
            {"expert_store_mib", store_mib},
            {"memory_before_load", memory_snapshot(nullptr)}};
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu_layers;
        mp.expert_streaming_size = store_mib * 1024 * 1024;
        mp.memory_budget_size = budget_mib * 1024 * 1024;
        llama_model_tensor_buft_override native_overrides[2] {};
        if (native_moe_cache_mib) {
            auto * cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            if (!cpu) { throw std::runtime_error("native MoE control requires a CPU backend"); }
            // Same expert placement as upstream --cpu-moe; all cache misses
            // compute on the GPU because moe_cache_cpu_share is zero.
            native_overrides[0] = {"\\.ffn_(up|down|gate|gate_up)_(ch|)exps", ggml_backend_dev_buffer_type(cpu)};
            mp.tensor_buft_overrides = native_overrides;
            // The upstream cache uploads host bytes directly. CPU_REPACK
            // changes their layout, so retain the original packed GGUF bytes.
            mp.use_extra_bufts = false;
            report["native_moe_cache_mib"] = native_moe_cache_mib;
            report["native_moe_cache_cpu_share"] = 0.0;
            report["native_moe_cache_repack"] = false;
        }
        report["engine_budget_mib"] = budget_mib;
        const auto load_start = steady_clock::now();
        ggml_backend_dev_t devices[] = {cuda_device.empty() ? nullptr : ggml_backend_dev_by_name(cuda_device.c_str()), nullptr};
        if (!cuda_device.empty() && !devices[0]) { throw std::runtime_error("selected CUDA device unavailable"); }
        if (devices[0]) { mp.devices = devices; }
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(discrete ?
            llama_load_discrete_model(path.c_str(), mp, devices[0], ram_mib * 1024 * 1024, vram_mib * 1024 * 1024,
                cuda_expert_streaming, cuda_experts_match_layer, cuda_boundary_reduction) :
            llama_model_load_from_file(path.c_str(), mp), llama_model_free);
        if (!model) { throw std::runtime_error("model load failed"); }
        if (cuda_boundary_reduction) { report["cuda_boundary_reduction"] = true; }
        if (discrete) { report["discrete_after_load"] = discrete_budget_snapshot(*model->memory_contract); }
        if (!preview_path.empty()) {
            if (!budget_mib || warmup || repeat != 1 || prefill.target || prefill.pressure || !resizes.empty() || !policy_mode.empty()) {
                throw std::runtime_error("preview capture requires one fixed-budget request without resize");
            }
            preview.open(preview_path, *model);
        }
        report["preview_capture"] = !preview_path.empty();
        report["model_load_seconds"] = seconds(load_start);
        const auto * vocab = llama_model_get_vocab(model.get());
        int count = -llama_tokenize(vocab, prompt.c_str(), int(prompt.size()), nullptr, 0, true, true);
        if (count < 1 || count + tokens > context) { throw std::runtime_error("prompt and output do not fit context"); }
        if (cuda_prefill_resize_probe && count <= 2 * ubatch) {
            throw std::runtime_error("CUDA prefill resize probe requires three or more prefill microbatches");
        }
        if (mtp_probe && (model->arch != LLM_ARCH_QWEN35MOE || llama_model_n_layer_nextn(model.get()) != 1 ||
                model->devices.size() != 1 || std::strcmp(ggml_backend_dev_name(model->devices.front().dev), "MTL0"))) {
            throw std::runtime_error("MTP probe requires a Qwen3.5 MoE NextN target on MTL0");
        }
        std::vector<llama_token> input(count);
        if (llama_tokenize(vocab, prompt.c_str(), int(prompt.size()), input.data(), count, true, true) != count) {
            throw std::runtime_error("tokenization failed");
        }
        report["prompt_ids"] = input;
        auto cp = llama_context_default_params();
        cuda_prefill_resize prefill_budget;
        llama_memory_budget::ticket trace_charge;
        tensor_trace graph_trace;
        if (!tensor_trace_path.empty()) {
            constexpr size_t capacity = 1 << 20;
            if (discrete) {
                trace_charge = model->memory_contract->ledger->reserve(llama_memory_domain::host, llama_memory_category::scratch, capacity);
                if (!trace_charge) { throw std::runtime_error("tensor tracing exceeds the RAM budget"); }
            }
            graph_trace.scratch.resize(capacity);
            if (trace_charge) { trace_charge.commit(); }
            graph_trace.file.open(tensor_trace_path);
            if (!graph_trace.file) { throw std::runtime_error("cannot open tensor trace"); }
            cp.cb_eval = tensor_trace::inspect;
            cp.cb_eval_user_data = &graph_trace;
        }
        llama_memory_budget::ticket layer_ticket;
        std::unique_ptr<mtp_layer_probe> layer_probe;
        if (mtp_trace_layers) {
            layer_ticket = model->memory_contract->ledger->reserve(
                llama_memory_domain::host, llama_memory_category::drafter, sizeof(mtp_layer_probe));
            if (!layer_ticket) { throw std::runtime_error("MTP trace allocation denied"); }
            layer_probe.reset(new mtp_layer_probe);
            layer_ticket.commit();
            cp.cb_eval = mtp_layer_probe::inspect;
            cp.cb_eval_user_data = layer_probe.get();
        }
        cp.n_ctx = context; cp.n_batch = context; cp.n_ubatch = ubatch; cp.n_seq_max = 1; cp.no_perf = false;
        if (native_moe_cache_mib) {
            cp.moe_cache_size = native_moe_cache_mib * 1024 * 1024;
            cp.moe_cache_cpu_share = 0.0f;
        }
        if (devices[0]) { cp.n_threads = cp.n_threads_batch = 4; }
        if (prefill.target || prefill.pressure) { cp.cb_eval = prefill_resize::callback; cp.cb_eval_user_data = &prefill; }
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) { throw std::runtime_error("context init failed"); }
        if (discrete) { report["discrete_after_context_init"] = discrete_budget_snapshot(*model->memory_contract); }
        if (preview.file && !llama_expert_route_observer_set(ctx.get(), expert_preview_trace::callback, &preview)) {
            throw std::runtime_error("route observer requires streaming");
        }
        if (explicit_cache_policy && (store_mib || budget_mib || cache_policy != "lru") &&
            !llama_expert_cache_policy_set(ctx.get(), cache_policy.c_str())) {
            throw std::runtime_error("cache policy requires streaming, a valid policy and sufficient memory floor");
        }
        if (preview_mode && (!trace_path.empty() || !preview_path.empty() || cache_policy != "lru" ||
                !llama_expert_preview_set(ctx.get(), preview_mode))) {
            throw std::runtime_error("runtime preview requires qualified Qwen Metal, LRU, budget headroom and no diagnostic trace");
        }
        report["expert_preview"] = preview_mode;
        if (profile_experts && !llama_expert_profile_set(ctx.get(), true)) { throw std::runtime_error("expert profiling requires streaming"); }
        report["profile_experts"] = profile_experts;
        if (read_overlap >= 0 && !llama_expert_read_overlap_set(ctx.get(), read_overlap != 0)) {
            throw std::runtime_error("expert read overlap requires streaming");
        }
        report["expert_read_overlap"] = read_overlap < 0 ? "default" : read_overlap ? "on" : "off";
        if (single_map >= 0 && !llama_expert_single_map_set(ctx.get(), single_map != 0)) {
            throw std::runtime_error("expert single map requires Metal streaming");
        }
        report["expert_single_map"] = single_map < 0 ? "default" : single_map ? "on" : "off";
        if (route_index >= 0 && !llama_expert_route_index_set(ctx.get(), route_index != 0)) {
            throw std::runtime_error("expert route index requires streaming and budget headroom");
        }
        if (early_read >= 0 && !llama_expert_early_read_set(ctx.get(), early_read != 0)) {
            throw std::runtime_error("expert early read requires Metal, LRU, budget headroom and preview disabled");
        }
        report["expert_early_read"] = early_read < 0 ? "default" : early_read ? "on" : "off";
        report["expert_route_index"] = route_index < 0 ? "default" : route_index ? "on" : "off";
        report["cache_policy"] = budget_mib && !explicit_cache_policy ? "auto" : cache_policy;
        if (trace.file && !llama_expert_trace_set(ctx.get(), expert_trace_file::callback, &trace)) {
            throw std::runtime_error("expert trace requires disk streaming");
        }
        report["expert_trace"] = trace_path;
        if (!policy_mode.empty()) {
            policy_config.mode = llama_memory_mode::automatic;
            policy_config.preferred = budget_mib * 1024 * 1024;
            if (!llama_memory_policy_configure(ctx.get(), &policy_config, policy_mode == "system")) {
                throw std::runtime_error("memory policy configuration rejected");
            }
        }
        report["pressure_mode"] = policy_mode;
        report["pressure_recovery_ms"] = policy_config.recovery_ms;
        prefill.ctx = ctx.get();
        draft_admission_probe draft;
        if (!draft_admission_path.empty()) {
            auto ledger = model->memory_contract->ledger;
            const auto charge = [&] {
                return ledger->snapshot(llama_memory_domain::host)
                    .categories[size_t(llama_memory_category::drafter)].total();
            };
            const size_t before = charge();
            size_t weights = 0;
            {
                llama_allocation_scope admission(ledger, llama_memory_category::drafter);
                auto dm = llama_model_default_params();
                dm.n_gpu_layers = gpu_layers;
                dm.load_mtp = true;
                dm.load_mode = LLAMA_LOAD_MODE_NONE;
                dm.use_extra_bufts = false;
                draft.model.reset(llama_model_load_from_file(draft_admission_path.c_str(), dm));
                if (!draft.model || draft.model->arch != model->arch ||
                        llama_model_n_layer_nextn(draft.model.get()) != 1 ||
                        llama_vocab_n_tokens(llama_model_get_vocab(draft.model.get())) != llama_vocab_n_tokens(vocab) ||
                        llama_model_n_embd_out(draft.model.get()) != llama_model_n_embd_out(model.get())) {
                    throw std::runtime_error("MTP draft model load or hidden-width validation failed");
                }
                weights = charge() - before;
                auto dc = cp;
                if (layer_probe) {
                    dc.cb_eval = nullptr;
                    dc.cb_eval_user_data = nullptr;
                }
                dc.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
                dc.ctx_other = ctx.get();
                draft.ctx.reset(llama_init_from_model(draft.model.get(), dc));
                if (!draft.ctx) { throw std::runtime_error("MTP draft context init failed"); }
            }
            if (!weights || charge() - before <= weights) {
                throw std::runtime_error("MTP draft weights and context must both charge the shared ledger");
            }
            report["draft_admission"] = {{"weight_bytes", weights},
                {"context_bytes", charge() - before - weights}};
        }
        report["memory_after_init"] = memory_snapshot(ctx.get());
        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(llama_batch_ext_init(ctx.get()), llama_batch_ext_free);
        std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)> sampler(llama_sampler_init_greedy(), llama_sampler_free);
        std::ofstream logits;
        if (!logits_path.empty()) {
            logits.open(logits_path, std::ios::binary);
            if (!logits) { throw std::runtime_error("cannot open logits file"); }
        }
        report["vocab_size"] = llama_vocab_n_tokens(vocab);
        report["runs"] = json::array();
        for (int run = -warmup; run < repeat; ++run) {
            llama_memory_clear(llama_get_memory(ctx.get()), true);
            llama_perf_context_reset(ctx.get());
            std::vector<llama_token> committed;
            std::string text;
            json row = {{"memory_before", memory_snapshot(ctx.get())}};
            row["resizes"] = json::array();
            const auto start = steady_clock::now();
            set_batch(batch.get(), input, 0);
            if (preview.file) { preview.begin_call(0); }
            if (cuda_prefill_resize_probe) {
                prefill_budget.contract = model->memory_contract.get();
                prefill_budget.device = devices[0];
                prefill_budget.cancel = cuda_worker_read == "cancel" || cuda_uring_read;
                prefill_budget.worker = !cuda_worker_read.empty();
#ifdef ATOMIC_ENGINE_IO_URING_PROBE
                if (cuda_uring_read) {
                    prefill_budget.uring = std::make_unique<uring_read_probe>();
                }
#endif
                if (!prefill_budget.arm(ctx.get())) {
                    throw std::runtime_error("cannot arm CUDA prefill resize probe");
                }
            }
            graph_trace.active = graph_trace.file.is_open();
            if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) { throw std::runtime_error("prefill failed"); }
            if (cuda_prefill_resize_probe) {
                llama_discrete_prefill_probe_set(ctx.get(), nullptr, nullptr, nullptr);
                if (prefill_budget.boundaries < 3 || prefill_budget.events.size() != 2) {
                    throw std::runtime_error("CUDA prefill resize did not continue through the grow");
                }
                row["prefill_budget_boundaries"] = prefill_budget.boundaries;
                row["prefill_budget_events"] = std::move(prefill_budget.events);
                if (!cuda_worker_read.empty()) { row["worker_read_decision"] = cuda_worker_read; }
            }
            if (graph_trace.active) {
                llama_synchronize(ctx.get());
                graph_trace.active = false;
                graph_trace.file.close();
                if (!graph_trace.file) { throw std::runtime_error("cannot write tensor trace"); }
                std::vector<uint8_t>().swap(graph_trace.scratch);
                trace_charge.reset();
            }
            row["prefill_seconds"] = seconds(start);
            row["memory_after_prefill"] = memory_snapshot(ctx.get());
            row["prefill_resize_request_id"] = prefill.request_id;
            bool batch_committed = false;
            for (int step = 0; step < tokens; ++step) {
                const int output_row = batch_committed && step == 1 ? 0 : -1;
                if (run >= 0 && logits.is_open()) {
                    logits.write(reinterpret_cast<const char *>(llama_get_logits_ith(ctx.get(), output_row)),
                                 llama_vocab_n_tokens(vocab) * sizeof(float));
                }
                if (mtp_probe && !mtp_batch_commit_probe && step == 1) {
                    const auto hash = float_digest(llama_get_logits_ith(ctx.get(), -1), llama_vocab_n_tokens(vocab));
                    row["mtp"]["serial_verify_logits_hash"] = hash;
                    row["mtp"]["verify_logits_bitwise_equal"] = row["mtp"]["verify_logits_hash"] == hash;
                }
                const auto token = llama_sampler_sample(sampler.get(), ctx.get(), output_row);
                if (mtp_probe && !mtp_batch_commit_probe && step == 1) {
                    row["mtp"]["verify_matches_serial_token"] = row["mtp"]["verified_id"] == token;
                }
                if (llama_vocab_is_eog(vocab, token)) { row["stop_id"] = token; break; }
                if (rollback_probe && step == 0) {
                    row["rollback"] = probe_rollback(*model, ctx.get(), batch.get(), count, token);
                }
                if (mtp_probe && step == 0) {
                    row["mtp"] = probe_mtp(*model, ctx.get(), batch.get(), draft.ctx.get(), input, token,
                            llama_vocab_n_tokens(vocab), mtp_batch_commit_probe, layer_probe.get());
                    if (mtp_batch_commit_probe) { batch_committed = row["mtp"]["batch_committed"].get<bool>(); }
                }
                committed.push_back(token);
                llama_sampler_accept(sampler.get(), token);
                char piece[256];
                int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, true);
                if (n < 0) { throw std::runtime_error("token piece exceeds diagnostic buffer"); }
                text.append(piece, size_t(n));
                if (cuda_resize_probe && committed.size() <= 2) {
                    llama_synchronize(ctx.get());
                    const auto & contract = *model->memory_contract;
                    constexpr size_t mib = 1024 * 1024;
                    constexpr size_t scratch_headroom = 64 * mib;
                    const bool shrink = committed.size() == 1;
                    const size_t ram_target = (shrink ? 3200 : ram_mib) * mib;
                    const size_t vram_target = (shrink ? 1024 : vram_mib) * mib;
                    const auto before = discrete_budget_snapshot(contract);
                    if (shrink && (llama_discrete_budget_resize(ctx.get(), ram_target, mib, scratch_headroom) ||
                            llama_discrete_budget_resize(ctx.get(), mib, vram_target, scratch_headroom) ||
                            before != discrete_budget_snapshot(contract))) {
                        throw std::runtime_error("impossible CUDA resize changed the budget or residency");
                    }
                    size_t free_before = 0, free_after = 0, device_total = 0;
                    ggml_backend_dev_memory(devices[0], &free_before, &device_total);
                    const auto resize_start = steady_clock::now();
                    if (!llama_discrete_budget_resize(ctx.get(), ram_target, vram_target, scratch_headroom)) {
                        throw std::runtime_error("CUDA resize probe could not apply the requested caps");
                    }
                    const double elapsed = seconds(resize_start);
                    ggml_backend_dev_memory(devices[0], &free_after, &device_total);
                    row["resizes"].push_back({{"after_committed", committed.size()},
                        {"before", before}, {"after", discrete_budget_snapshot(contract)}, {"seconds", elapsed},
                        {"device_free_before", free_before}, {"device_free_after", free_after}});
                }
                if (!policy_mode.empty()) {
                    json event = {{"after_committed", committed.size()}, {"before", memory_snapshot(ctx.get())}};
                    if (resume_at == int(committed.size())) {
                        event["resumed"] = llama_memory_policy_configure(ctx.get(), &policy_config, false);
                    }
                    event["signals"] = json::array();
                    for (const auto & signal : signals) {
                        if (signal.step == int(committed.size())) {
                            event["signals"].push_back({{"level", int(signal.sample.level)},
                                {"accepted", llama_memory_pressure_submit(ctx.get(), &signal.sample)}});
                        }
                    }
                    const auto policy_start = steady_clock::now();
                    event["poll_ok"] = llama_memory_budget_apply(ctx.get());
                    event["poll_seconds"] = seconds(policy_start);
                    event["after"] = memory_snapshot(ctx.get());
                    row["pressure_events"].push_back(std::move(event));
                }
                bool scheduled = false;
                for (const auto & resize : resizes) { scheduled |= resize.first == int(committed.size()); }
                if (scheduled) {
                    json event = {{"after_committed", committed.size()},
                        {"before", memory_snapshot(ctx.get())}};
                    const auto resize_start = steady_clock::now();
                    event["requests"] = json::array();
                    for (const auto & resize : resizes) {
                        if (resize.first != int(committed.size())) { continue; }
                        const auto id = llama_memory_budget_request(ctx.get(), resize.second);
                        event["requests"].push_back({{"id", id}, {"target", resize.second}});
                        event["target"] = resize.second;
                    }
                    event["applied"] = llama_memory_budget_apply(ctx.get());
                    event["seconds"] = seconds(resize_start);
                    event["after"] = memory_snapshot(ctx.get());
                    row["resizes"].push_back(std::move(event));
                }
                if (step + 1 < tokens && !(batch_committed && step < 2)) {
                    set_batch(batch.get(), {token}, count + step);
                    if (preview.file) { preview.begin_call(count + step); }
                    if (llama_process(ctx.get(), LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) { throw std::runtime_error("decode failed"); }
                }
            }
            if (rollback_probe && !row.contains("rollback")) { throw std::runtime_error("rollback probe needs a non-EOG token"); }
            if (cuda_resize_probe && (row["resizes"].size() != 2 || committed.size() < 3)) {
                throw std::runtime_error("CUDA resize probe did not continue decode after both budget changes");
            }
            if (mtp_probe && (!row.contains("mtp") ||
                    (!mtp_batch_commit_probe && !row["mtp"].contains("verify_matches_serial_token")))) {
                throw std::runtime_error("MTP probe requires a non-EOG seed and serial verification");
            }
            row["request_seconds"] = seconds(start);
            row["committed_ids"] = committed;
            row["committed_count"] = committed.size();
            row["text"] = text;
            if (capture_state && mtp_probe) {
                auto ticket = reserve_probe(*model, ctx.get(), llama_state_get_size(ctx.get()));
                row["state_fnv1a64"] = state_digest(ctx.get());
            } else if (capture_state) {
                row["state_fnv1a64"] = state_digest(ctx.get(), discrete ? model->memory_contract.get() : nullptr);
            }
            row["memory_after"] = memory_snapshot(ctx.get());
            if (run >= 0) { report["runs"].push_back(std::move(row)); }
        }
        if (trace.file) {
            llama_expert_trace_set(ctx.get(), nullptr, nullptr);
            if (std::fflush(trace.file.get()) != 0 || trace.failed) { throw std::runtime_error("cannot write expert trace"); }
        }
        if (preview.file) {
            llama_expert_route_observer_set(ctx.get(), nullptr, nullptr);
            preview.finish();
        }
        if (draft.ctx) {
            draft.ctx.reset();
            draft.model.reset();
            report["memory_after_draft_release"] = memory_snapshot(ctx.get());
        }
        report["process_seconds_before_cleanup"] = seconds(process_start);
        if (discrete) {
            llama_synchronize(ctx.get());
            const auto contract = model->memory_contract;
            report["discrete_after_inference"] = discrete_budget_snapshot(*contract);
            batch.reset();
            ctx.reset();
            model.reset();
            report["discrete_after_release"] = discrete_budget_snapshot(*contract);
        }
        std::ofstream file(output);
        file << report.dump(2) << '\n';
        if (!file || (logits.is_open() && !logits)) { throw std::runtime_error("cannot write diagnostic output"); }
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "expert inference: %s\n", e.what());
        return 1;
    }
}
