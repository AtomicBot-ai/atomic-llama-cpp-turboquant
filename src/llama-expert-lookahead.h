#pragma once
#include "llama-expert-prefetch.h"

#include "llama-expert-preview-dot.h"
#include "llama-expert-store.h"
#include "llama-memory-contract.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#endif

// One graph owner. Speculation owns raw GGUF bytes only. Normal demand still
// decides placement, eviction, publication, and which weights reach Metal.
class llama_expert_lookahead {
    static constexpr size_t experts_max = 1024, candidates = 2, slices = 3 * candidates;
    std::shared_ptr<llama_expert_source> source;
    llama_expert_store & store;
    llama_memory_budget::ticket charge;
    std::unique_ptr<uint8_t[]> data;
    std::unique_ptr<early_lookahead> early_state;
    const size_t stride;
    std::array<float, experts_max> scores{};
    std::array<llama_expert_source::read_request, slices> ready{};
    size_t count = 0;
    bool enabled = false;
    bool healthy = true;
    using clock = std::chrono::steady_clock;
    clock::time_point read_begin, read_end;
    bool submitted = false, read_ok = false;
    size_t read_bytes = 0;
#ifdef __APPLE__
    dispatch_group_t group = nullptr;
    struct slice_job {
        llama_expert_lookahead * owner = nullptr;
        size_t index = 0;
        dispatch_group_t done = nullptr;
        clock::time_point begin, end;
        bool ok = false;
        size_t bytes = 0;
    };
    // One group per slice, created only for the Metal predictor variant.
    std::array<slice_job, slices> jobs{};
    int64_t slice_wait_ns = 0;
    bool per_slice = false;

    void release_groups() {
        for (auto & job : jobs) { if (job.done) { dispatch_release(job.done); } }
        if (group) { dispatch_release(group); }
    }

    void wait_slice(size_t i, llama_expert_profile_data * profile, bool demand) {
        if (dispatch_group_wait(jobs[i].done, DISPATCH_TIME_NOW) == 0) { return; }
        const auto start = clock::now();
        dispatch_group_wait(jobs[i].done, DISPATCH_TIME_FOREVER);
        if (profile) {
            const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - start).count();
            slice_wait_ns += ns;
            if (demand) {
                ++profile->next_prefetch_slice_waits;
                profile->next_prefetch_slice_wait_ns += ns;
            }
        }
    }

    void drain_slices(llama_expert_profile_data * profile) {
        auto first = clock::time_point::max(), last = clock::time_point::min();
        size_t bytes = 0, failures = 0;
        for (size_t i = 0; i < count; ++i) {
            wait_slice(i, profile, false);
            first = std::min(first, jobs[i].begin);
            last = std::max(last, jobs[i].end);
            if (jobs[i].ok) { bytes += jobs[i].bytes; }
            else { ++failures; }
        }
        if (profile) {
            // Span across independently completed slice jobs, reported once.
            const auto span = std::chrono::duration_cast<std::chrono::nanoseconds>(last - first).count();
            profile->next_prefetch_read_ns += span;
            profile->next_prefetch_hidden_ns += std::max<int64_t>(0, span - slice_wait_ns);
            profile->next_prefetch_wait_ns += slice_wait_ns;
            profile->next_prefetch_bytes += bytes;
            profile->next_prefetch_failures += failures;
            ++profile->next_prefetch_batches;
        }
        if (failures) { enabled = healthy = false; }
        submitted = false;
    }
#endif

    void join(llama_expert_profile_data * profile) {
        if (early_state) { early_state->finish(profile); return; }
#ifdef __APPLE__
        if (!submitted) { return; }
        if (per_slice) { drain_slices(profile); return; }
        const auto wait_begin = clock::now();
        const bool pending = dispatch_group_wait(group, DISPATCH_TIME_NOW) != 0;
        if (pending) { dispatch_group_wait(group, DISPATCH_TIME_FOREVER); }
        const auto wait_end = clock::now();
        const auto ns = [](auto duration) { return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count(); };
        submitted = false;
        if (profile) {
            const auto service = ns(read_end - read_begin);
            const auto blocked = pending ? std::max<int64_t>(0, ns(std::min(read_end, wait_end) - std::max(read_begin, wait_begin))) : 0;
            profile->next_prefetch_read_ns += service;
            profile->next_prefetch_hidden_ns += service - blocked;
            if (pending) { profile->next_prefetch_wait_ns += ns(wait_end - wait_begin); }
            if (read_ok) {
                ++profile->next_prefetch_batches;
                profile->next_prefetch_bytes += read_bytes;
            } else {
                ++profile->next_prefetch_failures;
            }
        }
        if (!read_ok) { count = 0; enabled = healthy = false; }
#else
        (void) profile;
#endif
    }

public:
    struct target {
        const void * router = nullptr;
        size_t width = 0, experts = 0;
        std::array<const llama_expert_tensor_info *, 3> weights{};
        ggml_type router_type = GGML_TYPE_F32;
        const ggml_tensor * router_tensor = nullptr;
    };
    std::array<target, 128> targets{};
    static constexpr size_t top = 4;
    struct prediction {
        size_t rows = 0, ranks = 0;
        std::array<int32_t, candidates * top> ids{};
        std::array<float, candidates * top> logits{};
    };

private:
    // Predict from the stored router format without shadowing all routers in
    // memory. Expand BF16 lanes to float in registers and accumulate in float.
    static float bf16_dot(const ggml_bf16_t * w, const float * x, size_t width) {
        size_t i = 0;
        float score = 0;
#ifdef __aarch64__
        float32x4_t a = vdupq_n_f32(0), b = vdupq_n_f32(0);
        for (; i + 8 <= width; i += 8) {
            const auto weights = vld1q_u16(reinterpret_cast<const uint16_t *>(w + i));
            const auto low = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(weights)), 16));
            const auto high = vreinterpretq_f32_u32(vshlq_n_u32(vmovl_high_u16(weights), 16));
            a = vfmaq_f32(a, low, vld1q_f32(x + i));
            b = vfmaq_f32(b, high, vld1q_f32(x + i + 4));
        }
        score = vaddvq_f32(vaddq_f32(a, b));
#endif
        for (; i < width; ++i) { score += ggml_bf16_to_fp32(w[i]) * x[i]; }
        return score;
    }

public:

    llama_expert_lookahead(std::shared_ptr<llama_expert_source> source, llama_expert_store & store,
                          llama_memory_budget::ticket charge, size_t stride, bool independent = false, bool early = false, size_t candidate_limit = 8, early_io::pool * readers = nullptr) :
        source(std::move(source)), store(store), charge(std::move(charge)),
        data(early ? nullptr : new uint8_t[slices * stride]), stride(stride) {
        if (early) {
            early_state = std::make_unique<early_lookahead>(this->source, store, stride, candidate_limit, readers);
            this->charge.commit();
            return;
        }
#ifdef __APPLE__
        per_slice = independent;
        if (per_slice) {
            for (size_t i = 0; i < jobs.size(); ++i) {
                jobs[i].owner = this;
                jobs[i].index = i;
                jobs[i].done = dispatch_group_create();
                if (!jobs[i].done) { release_groups(); throw std::bad_alloc(); }
            }
        } else {
            group = dispatch_group_create();
            if (!group) { throw std::bad_alloc(); }
        }
#else
        (void) independent;
#endif
        this->charge.commit();
    }

    ~llama_expert_lookahead() {
        join(nullptr);
#ifdef __APPLE__
        release_groups();
#endif
    }

    static size_t working_bytes(size_t max_payload, bool early = false, size_t limit = 8, bool shared = false) {
        return sizeof(llama_expert_lookahead) + (early ? early_lookahead::memory_bytes(max_payload, limit, shared) : slices * max_payload);
    }
    size_t memory_bytes() const { return charge.size(); }
    bool early() const { return bool(early_state); }
    early_io::pool * priority_pool() const { return early_state ? early_state->pool() : nullptr; }
    void prepare_early(const char * name,const int64_t * actual,size_t n,llama_expert_profile_data * p) {
        if (early_state) early_state->prepare(name,actual,n,p);
    }
    void fetch_early(const target & next,const ggml_tensor * input,const ggml_tensor * ids,const ggml_tensor * scores,llama_expert_profile_data * p) {
        if (early_state) early_state->fetch(next,input,ids,scores,p);
    }

    target bind(const ggml_tensor * router, const std::array<const ggml_tensor *, 3> & weights,
                ggml_backend_t backend) const {
        if (!router || (router->type != GGML_TYPE_F32 && router->type != GGML_TYPE_BF16) || !ggml_is_contiguous(router) ||
                router->ne[0] <= 0 || router->ne[1] <= 0 || size_t(router->ne[1]) > experts_max ||
                router->ne[2] != 1 || router->ne[3] != 1 || !router->buffer || !router->data) { return {}; }
        const auto base = ggml_backend_buffer_get_base(router->buffer);
        if (!ggml_backend_buffer_is_host(router->buffer)) {
            using host_base_t = void * (*)(ggml_backend_buffer_t);
            const auto reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
            const auto host_base = reinterpret_cast<host_base_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_buffer_get_host_base"));
            if (!host_base || host_base(router->buffer) != base) { return {}; }
        }
        const auto start = reinterpret_cast<uintptr_t>(base), at = reinterpret_cast<uintptr_t>(router->data);
        const auto size = ggml_backend_buffer_get_size(router->buffer);
        if (at < start || at - start > size || ggml_nbytes(router) > size - (at - start)) { return {}; }
        target out;
        for (size_t i = 0; i < weights.size(); ++i) {
            const auto * w = weights[i];
            if (!w || w->ne[2] != router->ne[1]) { return {}; }
            const auto & info = source->tensor(w->name);
            if (!info.can_read_experts() || info.expert_bytes() > stride) { return {}; }
            out.weights[i] = &info;
        }
        out.router = router->data;
        out.router_tensor = router;
        out.router_type = router->type;
        out.width = router->ne[0];
        out.experts = router->ne[1];
        return out;
    }

    void begin_graph(bool decode) { count = 0; enabled = decode && healthy; }
    void end_graph(llama_expert_profile_data * profile) { join(profile); count = 0; enabled = false; }

    // Lookup by metadata identity and expert ID. A slice is consumed only by
    // actual demand for a cache miss, never by the predicted route itself.
    size_t consume(const llama_expert_source::read_request & request, llama_expert_profile_data * profile) {
        if (early_state) return early_state->consume(request,profile);
        for (size_t i = 0; i < count; ++i) {
            auto & slot = ready[i];
            if (slot.tensor != request.tensor || slot.expert != request.expert) { continue; }
#ifdef __APPLE__
            if (per_slice) {
                wait_slice(i, profile, true);
                if (!jobs[i].ok) { return 0; }
            } else
#endif
            {
                join(profile);
                if (!count) { return 0; }
            }
            const auto bytes = slot.tensor->expert_bytes();
            std::memcpy(request.destination, slot.destination, bytes);
            slot.tensor = nullptr;
            return bytes;
        }
        return 0;
    }

    // Do not wait for an unrelated old slice just to start a new prediction.
    bool can_replace() const {
#ifdef __APPLE__
        if (per_slice && submitted) {
            for (size_t i = 0; i < count; ++i) {
                if (dispatch_group_wait(jobs[i].done, DISPATCH_TIME_NOW) != 0) { return false; }
            }
        }
#endif
        return true;
    }

    void fetch(const target & next, const ggml_tensor * input, llama_expert_profile_data * profile,
               const prediction * gpu = nullptr) {
        if (!can_replace()) {
            if (profile) { ++profile->next_prefetch_busy_skips; }
            return;
        }
        join(profile);
        count = 0;
        if (!enabled || !next.router || size_t(input->ne[0]) != next.width) { return; }
        llama_expert_timer prediction(profile ? &profile->next_prefetch_predict_ns : nullptr);
        size_t n_ids = 0, n_reads = 0, bytes = 0;
        struct ranked { int64_t id; float weight; };
        std::array<ranked, candidates * top> order{};
        size_t ranked_count = 0;
        // Aggregate a bounded subset of rows and routes. Scores serve only to
        // prioritize speculative reads, never the actual FFN.
        const size_t rows = gpu ? gpu->rows : std::min<size_t>(input->ne[2], candidates);
        const size_t ranks = gpu ? gpu->ranks : std::min(top, next.experts);
        for (size_t row = 0; row < rows; ++row) {
            std::array<ranked, top> local{};
            if (gpu) {
                bool valid = true;
                for (size_t rank = 0; rank < ranks; ++rank) {
                    const auto index = row * ranks + rank;
                    const auto id = gpu->ids[index];
                    const float score = gpu->logits[index];
                    if (id < 0 || size_t(id) >= next.experts || !std::isfinite(score)) { valid = false; break; }
                    local[rank] = {id, score};
                }
                if (!valid) { continue; }
            } else {
                const auto * x = static_cast<const char *>(input->data) + row * input->nb[2];
                if (next.router_type == GGML_TYPE_BF16) {
                    const auto * weights = static_cast<const ggml_bf16_t *>(next.router);
                    for (size_t e = 0; e < next.experts; ++e) {
                        scores[e] = bf16_dot(weights + e * next.width, reinterpret_cast<const float *>(x), next.width);
                    }
                } else {
                    llama_expert_preview_dot<1>(static_cast<const float *>(next.router), x, input->nb[2],
                        next.width, next.experts, scores.data(), experts_max);
                }
                if (!std::all_of(scores.begin(), scores.begin() + next.experts, [](float x) { return std::isfinite(x); })) { continue; }
                for (size_t rank = 0; rank < ranks; ++rank) {
                    const auto best = std::max_element(scores.begin(), scores.begin() + next.experts);
                    local[rank] = {best - scores.begin(), *best};
                    *best = -std::numeric_limits<float>::infinity();
                }
            }
            const float largest = std::max_element(local.begin(), local.begin() + ranks,
                [](const ranked & a, const ranked & b) { return a.weight < b.weight; })->weight;
            float sum = 0;
            for (size_t rank = 0; rank < ranks; ++rank) {
                local[rank].weight = std::exp(local[rank].weight - largest);
                sum += local[rank].weight;
            }
            for (size_t rank = 0; rank < ranks; ++rank) {
                const auto id = local[rank].id;
                auto * end = order.data() + ranked_count;
                auto * found = std::find_if(order.data(), end, [id](const ranked & item) { return item.id == id; });
                if (found == end) { *found = {id, 0}; ++ranked_count; }
                found->weight += local[rank].weight / sum;
            }
        }
        std::sort(order.begin(), order.begin() + ranked_count, [](const ranked & a, const ranked & b) {
            return a.weight == b.weight ? a.id < b.id : a.weight > b.weight;
        });
        for (size_t i = 0; i < ranked_count && n_ids < candidates; ++i) {
            const auto before = n_reads;
            for (const auto * w : next.weights) {
                if (store.contains(w->name, order[i].id)) { continue; }
                ready[n_reads] = {w, order[i].id, data.get() + n_reads * stride};
                ++n_reads;
                bytes += w->expert_bytes();
            }
            if (n_reads != before) { ++n_ids; }
        }
        prediction.next(nullptr);
#ifdef __APPLE__
        if (!n_reads) { return; }
        count = n_reads;
        read_bytes = bytes;
        submitted = true;
        const auto queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
        if (per_slice) {
            slice_wait_ns = 0;
            for (size_t i = 0; i < count; ++i) {
                jobs[i].bytes = ready[i].tensor->expert_bytes();
                dispatch_group_async_f(jobs[i].done, queue, &jobs[i], [](void * opaque) {
                    auto & job = *static_cast<slice_job *>(opaque);
                    job.begin = clock::now();
                    try {
                        job.owner->source->read_experts(&job.owner->ready[job.index], 1, true);
                        job.ok = true;
                    } catch (...) { job.ok = false; }
                    job.end = clock::now();
                });
            }
            return;
        }
        dispatch_group_async_f(group, queue, this, [](void * opaque) {
            auto & self = *static_cast<llama_expert_lookahead *>(opaque);
            self.read_begin = clock::now();
            try {
                self.source->read_experts(self.ready.data(), self.count, true);
                self.read_ok = true;
            } catch (...) {
                // A speculative failure cannot publish any partially read slice.
                self.read_ok = false;
            }
            self.read_end = clock::now();
        });
#endif
    }

    bool active() const { return enabled; }
};
