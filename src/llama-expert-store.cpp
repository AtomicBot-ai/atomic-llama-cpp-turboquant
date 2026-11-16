#include "llama-expert-store.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <condition_variable>
#include <future>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace {

// The staging owner stays on the acquisition thread. Only the file read runs
// on a worker, which pauses at the midpoint until the caller makes a decision.
bool probe_worker_read(llama_expert_source & source, const std::string & name, int64_t expert,
        void * dst, size_t size, llama_expert_profile_data * profile,
        bool (*decide)(void *) noexcept, void * user) {
    struct exchange {
        std::mutex mutex;
        std::condition_variable changed;
        bool midpoint = false, decided = false, complete = false, proceed = false;
    } state;
    auto read = std::async(std::launch::async, [&] {
        try {
            const bool ok = source.read_expert(name, expert, dst, size, profile, [](void * arg) noexcept {
                auto & s = *static_cast<exchange *>(arg);
                std::unique_lock<std::mutex> lock(s.mutex);
                s.midpoint = true;
                s.changed.notify_one();
                s.changed.wait(lock, [&] { return s.decided; });
                return s.proceed;
            }, &state);
            std::lock_guard<std::mutex> lock(state.mutex);
            state.complete = true;
            state.changed.notify_one();
            return ok;
        } catch (...) {
            std::lock_guard<std::mutex> lock(state.mutex);
            state.complete = true;
            state.changed.notify_one();
            throw;
        }
    });
    {
        std::unique_lock<std::mutex> lock(state.mutex);
        state.changed.wait(lock, [&] { return state.midpoint || state.complete; });
        if (state.midpoint) {
            state.proceed = decide(user);
            state.decided = true;
            state.changed.notify_one();
        }
    }
    return read.get();
}

size_t round_allocation(size_t bytes, size_t granularity) {
    if (granularity == 0 || (granularity & (granularity - 1)) != 0 ||
        bytes > std::numeric_limits<size_t>::max() - (granularity - 1)) {
        throw std::invalid_argument("invalid expert allocation size or granularity");
    }
    return (bytes + granularity - 1) & ~(granularity - 1);
}

using expert_key = std::pair<size_t, int64_t>;

} // namespace

struct llama_expert_store::entry {
    expert_key key;
    uint64_t generation = 0;
    uint64_t last_use = 0;
    int frequency_class = 0;
    size_t pins = 0;
    bool retiring = false;
    // Non-owning recency links. C++ control metadata is covered by the
    // engine runtime envelope; payload and arena charges remain separate.
    entry * older = nullptr;
    entry * newer = nullptr;

    // Reverse destruction frees physical resources before releasing their charges.
    llama_memory_budget::ticket arena_charge;
    llama_memory_budget::ticket buffer_charge;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    ggml_tensor * tensor = nullptr;

    void retire() {
        if (!retiring) {
            arena_charge.retire();
            buffer_charge.retire();
            retiring = true;
        }
    }
};

struct llama_expert_store::impl {
    std::shared_ptr<llama_expert_source> source;
    std::shared_ptr<llama_memory_budget> budget;
    ggml_backend_t backend;
    // Declared before entries so the type survives every buffer and retiring lease.
    struct buffer_type_owner {
        ggml_backend_buffer_type_t type = nullptr;
        void (*release)(ggml_backend_buffer_type_t) = nullptr;
        ~buffer_type_owner() { if (type) { release(type); } }
    } owned_buft;
    ggml_backend_buffer_type_t buft;
    bool (*read_probe)(void *) noexcept = nullptr;
    void * read_probe_user = nullptr;
    bool read_probe_worker = false;
    llama_expert_read_probe submitted_read_probe = nullptr;
    using host_base_t = void * (*)(ggml_backend_buffer_t);
    host_base_t host_base = nullptr;
    bool reuse_zero_tail = false;
    llama_memory_domain buffer_domain;
    size_t granularity;
    std::mutex mutex;
    std::map<expert_key, std::shared_ptr<entry>> entries;
    static constexpr size_t class_count = 8;
    // Each class keeps acquisition order using the existing entry links.
    // Plain LRU puts every entry in class zero.
    std::array<entry *, class_count> lru_first {};
    std::array<entry *, class_count> lru_last {};
    uint64_t clock = 0;
    struct policy_tensor { size_t offset = 0; uint64_t clock = 0; };
    struct policy_history { float crf = 0; uint64_t last = 0; };
    llama_memory_budget::ticket policy_charge;
    std::unique_ptr<policy_tensor[]> policy_tensors;
    std::unique_ptr<policy_history[]> history;
    size_t active_tensor = 0;
    uint64_t active_tokens = 0;
    bool frequency_prefill = false;

    size_t policy_bytes() const {
        size_t count = 0;
        for (const auto & info : source->tensors()) {
            if (info.can_read_experts()) {
                if (uint64_t(info.ne[2]) > (SIZE_MAX / sizeof(policy_history)) - count) {
                    throw std::runtime_error("expert policy history is too large");
                }
                count += size_t(info.ne[2]);
            }
        }
        const size_t tensors = source->tensors().size();
        if (tensors > (SIZE_MAX - count * sizeof(policy_history)) / sizeof(policy_tensor)) {
            throw std::runtime_error("expert policy tensor table is too large");
        }
        return count * sizeof(policy_history) + tensors * sizeof(policy_tensor);
    }

    int touch(const expert_key & key) {
        if (!history) { return 0; }
        const auto now = policy_tensors[key.first].clock;
        auto & h = history[policy_tensors[key.first].offset + size_t(key.second)];
        // Adapted from upstream moe_cache_lru (PR #31233): 128-token
        // frequency half-life, eight classes, history retained after eviction.
        h.crf = h.crf * std::exp2(-float(now - h.last) / 128.0f) + 1.0f;
        h.last = now;
        return std::min(int(class_count - 1), int(std::log2(h.crf)));
    }
    llama_expert_store_stats counters;
    llama_expert_profile_data * profile = nullptr;
    llama_expert_trace_callback trace = nullptr;
    void * trace_user = nullptr;

    size_t trace_capacity() const {
        const auto memory = budget->snapshot(llama_memory_domain::host);
        const size_t fixed = memory.used() - counters.resident_bytes;
        return memory.requested > fixed ? memory.requested - fixed : 0;
    }

    impl(std::shared_ptr<llama_expert_source> source, std::shared_ptr<llama_memory_budget> budget,
         ggml_backend_t backend, llama_memory_domain domain, size_t granularity, ggml_backend_buffer_type_t buft) :
        source(std::move(source)), budget(std::move(budget)), backend(backend), buft(buft),
        buffer_domain(domain), granularity(granularity) {
        if (!this->source || !this->budget || !backend) {
            throw std::invalid_argument("expert store requires a source, budget and backend");
        }
        if (!this->buft) {
            this->buft = ggml_backend_get_default_buffer_type(backend);
        }
        if (!ggml_backend_supports_buft(backend, this->buft)) {
            throw std::invalid_argument("expert buffer type is unsupported by the backend");
        }
        const auto device = ggml_backend_get_device(backend);
        if (device) {
            host_base = reinterpret_cast<host_base_t>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(device), "ggml_backend_metal_buffer_get_host_base"));
        }
        reuse_zero_tail = host_base && this->source->architecture() == "qwen4exp";
        if (!buft && reuse_zero_tail) {
            const auto reg = ggml_backend_dev_backend_reg(device);
            using create_t = ggml_backend_buffer_type_t (*)(ggml_backend_dev_t);
            const auto create = reinterpret_cast<create_t>(ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_metal_create_expert_buffer_type"));
            const auto release = reinterpret_cast<void (*)(ggml_backend_buffer_type_t)>(ggml_backend_reg_get_proc_address(
                reg, "ggml_backend_metal_free_expert_buffer_type"));
            if (create && release) {
                owned_buft.release = release;
                owned_buft.type = create(device);
                if (owned_buft.type) { this->buft = owned_buft.type; }
            }
        }
        round_allocation(1, granularity);
    }

    uint64_t next_clock() {
        if (clock == std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error("expert store generation exhausted");
        }
        return ++clock;
    }

    void unlink_lru(entry & e) {
        if (e.older) { e.older->newer = e.newer; }
        else { lru_first[e.frequency_class] = e.newer; }
        if (e.newer) { e.newer->older = e.older; }
        else { lru_last[e.frequency_class] = e.older; }
        e.older = e.newer = nullptr;
    }

    void append_lru(entry & e) {
        auto & tail = lru_last[e.frequency_class];
        e.older = tail;
        e.newer = nullptr;
        if (tail) { tail->newer = &e; }
        else { lru_first[e.frequency_class] = &e; }
        tail = &e;
    }

    entry * oldest_unpinned(size_t cls) {
        auto * candidate = lru_first[cls];
        while (candidate) {
            if (profile) { ++profile->eviction_candidates; }
            if (candidate->pins == 0) { break; }
            candidate = candidate->newer;
        }
        return candidate;
    }

    std::shared_ptr<entry> take_victim() {
        llama_expert_timer timer(profile ? &profile->eviction_search_ns : nullptr);
        entry * chosen = nullptr;
        if (history) {
            double best_score = -1;
            if (frequency_prefill) {
                auto consider = [&](entry * candidate, size_t cls) {
                    if (!candidate) { return; }
                    const auto key = candidate->key;
                    const auto & tensor = policy_tensors[key.first];
                    const auto & h = history[tensor.offset + size_t(key.second)];
                    // Log of inverse current frequency: preserve the existing
                    // 128-token decay without overflowing on long idle ages.
                    // Entries predating optional history are cold candidates.
                    const double score = h.crf > 0 ? double(tensor.clock - h.last) / 128.0 -
                        std::log2(double(h.crf)) : std::numeric_limits<double>::infinity();
                    if (!chosen || score > best_score || (score == best_score &&
                            cls == size_t(chosen->frequency_class) && key > chosen->key)) {
                        best_score = score;
                        chosen = candidate;
                    }
                };
                for (size_t cls = 0; cls < class_count; ++cls) {
                    auto * first = oldest_unpinned(cls);
                    consider(first, cls);
                    // Recently admitted cold weights must also compete with
                    // older frequent weights. Inspect only both queue ends,
                    // skipping active leases, rather than scanning residents.
                    auto * last = lru_last[cls];
                    while (last && last != first) {
                        if (profile) { ++profile->eviction_candidates; }
                        if (!last->pins) { break; }
                        last = last->older;
                    }
                    if (last != first) { consider(last, cls); }
                }
            } else {
                for (size_t cls = 0; cls < class_count; ++cls) {
                    auto * candidate = oldest_unpinned(cls);
                    if (!candidate) { continue; }
                    const auto key = candidate->key;
                    const auto & tensor = policy_tensors[key.first];
                    const auto & h = history[tensor.offset + size_t(key.second)];
                    const double score = std::ldexp(double(tensor.clock - h.last), -int(cls));
                    if (score > best_score) {
                        best_score = score;
                        chosen = candidate;
                    }
                }
            }
        } else {
            chosen = oldest_unpinned(0);
        }
        if (!chosen) { return {}; }
        auto victim = entries.find(chosen->key);
        unlink_lru(*chosen);
        auto result = std::move(victim->second);
        counters.resident_bytes -= result->arena_charge.size() + result->buffer_charge.size();
        entries.erase(victim); // no longer published; zero leases, backend completed
        ++counters.evictions;
        return result;
    }

    bool evict_one() { return bool(take_victim()); }

    void trim() {
        for (auto it = entries.begin(); it != entries.end();) {
            if (it->second->pins == 0) {
                unlink_lru(*it->second);
                counters.resident_bytes -= it->second->arena_charge.size() + it->second->buffer_charge.size();
                it = entries.erase(it);
                ++counters.evictions;
            } else {
                it->second->retire();
                ++it;
            }
        }
    }

    struct reservations {
        llama_memory_budget::ticket arena;
        llama_memory_budget::ticket buffer;
        llama_memory_budget::ticket staging;
        bool complete() const { return arena && buffer && staging; }
    };

    reservations reserve(size_t arena_bytes, size_t buffer_bytes, size_t staging_bytes) {
        reservations r;
        r.arena = budget->reserve(llama_memory_domain::host, llama_memory_category::runtime, arena_bytes);
        if (r.arena) {
            r.buffer = budget->reserve(buffer_domain, llama_memory_category::experts, buffer_bytes);
        }
        if (r.buffer) {
            r.staging = budget->reserve(llama_memory_domain::host, llama_memory_category::io, staging_bytes);
        }
        return r;
    }
};

llama_expert_store::llama_expert_store(std::shared_ptr<llama_expert_source> source,
        std::shared_ptr<llama_memory_budget> budget, ggml_backend_t backend,
        llama_memory_domain buffer_domain, size_t allocation_granularity, ggml_backend_buffer_type_t buft) :
    pimpl(std::make_shared<impl>(std::move(source), std::move(budget), backend,
                                buffer_domain, allocation_granularity, buft)) {}

llama_expert_store::~llama_expert_store() { trim(); }

bool llama_expert_store::contains(const std::string & name, int64_t expert) const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    const auto & info = pimpl->source->tensor(name);
    const size_t index = size_t(&info - pimpl->source->tensors().data());
    if (expert >= info.ne[2]) { return false; }
    for (int64_t i = expert < 0 ? 0 : expert; i < (expert < 0 ? info.ne[2] : expert + 1); ++i) {
        const auto found = pimpl->entries.find({index, i});
        if (found == pimpl->entries.end() || found->second->retiring) { return false; }
    }
    return true;
}

void llama_expert_store::advise_expert(const std::string & name, int64_t expert) {
    if (!contains(name, expert)) { pimpl->source->advise_expert(name, expert); }
}

llama_expert_store::lease llama_expert_store::acquire(const std::string & name, int64_t expert) {
    auto result = acquire_once(name, expert);
    // The one-shot cancellation has released all unpublished owners. Retry
    // the original demand from its GGUF offset through the normal path.
    return result ? std::move(result) : acquire_once(name, expert);
}

llama_expert_store::lease llama_expert_store::acquire_once(const std::string & name, int64_t expert) {
    auto * profile = pimpl->profile;
    llama_expert_timer timer(profile ? &profile->admission_ns : nullptr);
    const auto & info = pimpl->source->tensor(name);
    const size_t data_bytes = info.expert_bytes();
    if (expert < 0 || expert >= info.ne[2]) {
        throw std::invalid_argument("invalid expert ID");
    }
    const expert_key key {size_t(&info - pimpl->source->tensors().data()), expert};
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    const size_t capacity = pimpl->trace ? pimpl->trace_capacity() : 0;
    const int frequency_class = pimpl->touch(key);
    const auto found = pimpl->entries.find(key);
    if (found != pimpl->entries.end()) {
        auto & e = *found->second;
        if (e.retiring) {
            throw std::runtime_error("expert is retiring; release outstanding leases before reacquiring");
        }
        e.last_use = pimpl->next_clock();
        pimpl->unlink_lru(e);
        e.frequency_class = frequency_class;
        pimpl->append_lru(e);
        ++e.pins;
        ++pimpl->counters.hits;
        if (pimpl->trace) {
            const size_t alignment = std::max(pimpl->granularity, ggml_backend_buft_get_alignment(pimpl->buft));
            const size_t required = ggml_tensor_overhead() + round_allocation(
                ggml_backend_buft_get_alloc_size(pimpl->buft, e.tensor), alignment);
            pimpl->trace({llama_expert_trace_kind::access, key.first, expert, 0, data_bytes,
                e.arena_charge.size() + e.buffer_charge.size(), capacity, true, required}, pimpl->trace_user);
        }
        return lease(pimpl, found->second);
    }

    // Query backend layout cost before allocating even the tensor arena.
    ggml_tensor probe {};
    probe.type = info.type;
    probe.ne[0] = info.ne[0]; probe.ne[1] = info.ne[1]; probe.ne[2] = 1; probe.ne[3] = 1;
    probe.nb[0] = ggml_type_size(info.type);
    probe.nb[1] = ggml_row_size(info.type, info.ne[0]);
    probe.nb[2] = data_bytes; probe.nb[3] = data_bytes;
    const size_t alignment = std::max(pimpl->granularity, ggml_backend_buft_get_alignment(pimpl->buft));
    const size_t buffer_bytes = round_allocation(ggml_backend_buft_get_alloc_size(pimpl->buft, &probe), alignment);
    const size_t arena_bytes = ggml_tensor_overhead();
    if (buffer_bytes > ggml_backend_buft_get_max_size(pimpl->buft)) {
        throw std::runtime_error("expert exceeds maximum backend buffer size");
    }

    impl::reservations charges;
    std::shared_ptr<entry> e;
    for (;;) {
        charges = pimpl->reserve(arena_bytes, buffer_bytes, data_bytes);
        if (charges.complete()) {
            break;
        }
        charges = {}; // roll back the whole attempt before eviction or retry
        auto victim = pimpl->take_victim();
        if (!victim) {
            throw std::runtime_error("expert working set does not fit the requested managed budget");
        }
        // Reuse only the policy's chosen, unpinned victim if its allocation fits.
        // Keep the entire physical charge, including any excess capacity. If
        // staging cannot fit, destroy it normally and retry bounded admission.
        if (!victim->retiring && victim->buffer_charge.size() >= buffer_bytes &&
            victim->arena_charge.size() == arena_bytes) {
            charges.staging = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::io, data_bytes);
            if (charges.staging) { e = std::move(victim); break; }
        }
    }

    timer.next(profile ? &profile->allocation_ns : nullptr);
    const bool reused = bool(e);
    const size_t previous_payload = reused && pimpl->reuse_zero_tail ? ggml_nbytes(e->tensor) : 0;
    if (!e) {
        e = std::make_shared<entry>();
        e->arena_charge = std::move(charges.arena);
        e->buffer_charge = std::move(charges.buffer);
        e->ctx.reset(ggml_init({arena_bytes, nullptr, true}));
        if (!e->ctx) { throw std::runtime_error("failed to allocate expert tensor arena"); }
        e->arena_charge.commit();
        e->buffer.reset(ggml_backend_buft_alloc_buffer(pimpl->buft, buffer_bytes));
        if (!e->buffer) { throw std::runtime_error("failed to allocate expert backend buffer"); }
        e->buffer_charge.commit();
        ++pimpl->counters.buffer_allocations;
    } else {
        ggml_reset(e->ctx.get());
    }
    e->key = key;
    e->frequency_class = frequency_class;
    e->tensor = ggml_new_tensor_2d(e->ctx.get(), info.type, info.ne[0], info.ne[1]);
    ggml_set_name(e->tensor, info.name.c_str());
    if (ggml_backend_buffer_get_size(e->buffer.get()) != e->buffer_charge.size()) {
        throw std::runtime_error("backend buffer size differs from admitted allocation");
    }
    if (ggml_backend_tensor_alloc(e->buffer.get(), e->tensor, ggml_backend_buffer_get_base(e->buffer.get())) != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("failed to bind expert tensor to backend buffer");
    }
    ggml_backend_buffer_set_usage(e->buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    // A fresh buffer or a completed, unpinned victim is exclusively owned here.
    // Metal's generic is_host flag is false even for shared storage, so require
    // its explicit capability instead of treating arbitrary device bases as RAM.
    void * direct = ggml_backend_buffer_is_host(e->buffer.get()) ? e->tensor->data :
        (pimpl->host_base ? pimpl->host_base(e->buffer.get()) : nullptr);
    if (direct != e->tensor->data) { direct = nullptr; }
    // Keep the same conservative I/O reservation on both paths. Direct reads
    // need no physical staging allocation, so that ticket remains uncommitted.
    std::unique_ptr<uint8_t[]> staging;
    if (direct) {
        // Published Metal weights are read-only and their entire tail is zero.
        // A reused slot only needs to clear the previous payload's exposed tail;
        // the new read overwrites the rest. Failed/cancelled reads destroy the
        // unpublished owner, so they cannot carry a partially written tail here.
        const size_t padding_end = reused && pimpl->reuse_zero_tail ? previous_payload : e->buffer_charge.size();
        const size_t padding_bytes = padding_end > data_bytes ? padding_end - data_bytes : 0;
        llama_expert_timer padding_timer(profile ? &profile->padding_ns : nullptr);
        if (profile) { profile->padding_bytes += padding_bytes; }
        std::memset(static_cast<uint8_t *>(direct) + data_bytes, 0, padding_bytes);
    } else {
        ggml_backend_buffer_clear(e->buffer.get(), 0);
        staging.reset(new uint8_t[data_bytes]);
        charges.staging.commit();
    }
    timer.next(profile ? &profile->read_ns : nullptr);
    const auto read_probe = std::exchange(pimpl->read_probe, nullptr);
    const auto submitted_read = std::exchange(pimpl->submitted_read_probe, nullptr);
    void * read_dst = direct ? direct : staging.get();
    const bool complete = read_probe && pimpl->read_probe_worker ?
        probe_worker_read(*pimpl->source, info.name, expert, read_dst, data_bytes, profile, read_probe, pimpl->read_probe_user) :
        pimpl->source->read_expert(info.name, expert, read_dst, data_bytes, profile, read_probe, pimpl->read_probe_user, submitted_read);
    if (!complete) { return {}; }
    if (direct && profile) { profile->direct_read_bytes += data_bytes; }
    timer.next(profile ? &profile->upload_ns : nullptr);
    if (!direct) { ggml_backend_tensor_set(e->tensor, staging.get(), 0, data_bytes); }
    // Direct writes touch only this unpublished, exclusively owned buffer.
    // Other pinned experts may still be computing; do not drain that work here.
    if (!direct) { ggml_backend_synchronize(pimpl->backend); }
    timer.next(nullptr);
    e->generation = e->last_use = pimpl->next_clock();
    e->pins = 1;
    pimpl->entries.emplace(key, e); // publish only fully read, backend-ready data
    pimpl->counters.resident_bytes += e->arena_charge.size() + e->buffer_charge.size();
    pimpl->append_lru(*e);
    ++pimpl->counters.misses;
    pimpl->counters.buffer_reuses += reused;
    pimpl->counters.logical_read_bytes += data_bytes;
    if (pimpl->trace) {
        pimpl->trace({llama_expert_trace_kind::access, key.first, expert, 0, data_bytes,
            e->arena_charge.size() + e->buffer_charge.size(), capacity, false, arena_bytes + buffer_bytes}, pimpl->trace_user);
    }
    return lease(pimpl, std::move(e));
}

llama_memory_budget_snapshot llama_expert_store::request_budget(llama_memory_domain domain, size_t target) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->budget->request(domain, target);
    while (pimpl->budget->snapshot(domain).pending() && pimpl->evict_one()) {}
    if (pimpl->budget->snapshot(domain).pending()) {
        // Pinned buffers remain charged until the backend completes and the final
        // lease is dropped. Do not admit more consumers to retiring generations.
        for (auto & item : pimpl->entries) {
            item.second->retire();
        }
    }
    return pimpl->budget->snapshot(domain);
}

void llama_expert_store::trim() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->trim();
    if (pimpl->trace) { pimpl->trace({llama_expert_trace_kind::reset}, pimpl->trace_user); }
}

void llama_expert_store::evict_idle() {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    while (pimpl->evict_one()) {}
}

void llama_expert_store::set_read_probe(bool (*callback)(void *) noexcept, void * user, bool worker,
        llama_expert_read_probe submitted_read) {
    pimpl->read_probe = callback;
    pimpl->read_probe_user = user;
    pimpl->read_probe_worker = worker;
    pimpl->submitted_read_probe = submitted_read;
}


void llama_expert_store::set_profile(llama_expert_profile_data * profile) { pimpl->profile = profile; }

void llama_expert_store::set_trace(llama_expert_trace_callback callback, void * user) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->trace = callback;
    pimpl->trace_user = user;
    if (callback) {
        callback({llama_expert_trace_kind::configuration, 0, 0, 0,
            pimpl->policy_bytes(), pimpl->policy_charge.size()}, user);
    }
}

void llama_expert_store::begin_projection(const std::string & name, int64_t tokens) {
    if (!pimpl->trace && !pimpl->history) { return; }
    const auto & info = pimpl->source->tensor(name);
    const size_t tensor = size_t(&info - pimpl->source->tensors().data());
    if (tokens < 1) { throw std::invalid_argument("invalid expert projection token count"); }
    if (pimpl->history) {
        auto & clock = pimpl->policy_tensors[pimpl->active_tensor].clock;
        if (clock > UINT64_MAX - pimpl->active_tokens) { throw std::runtime_error("expert policy clock exhausted"); }
        clock += pimpl->active_tokens;
        pimpl->active_tensor = tensor;
        pimpl->active_tokens = uint64_t(tokens);
    }
    if (pimpl->trace) {
        pimpl->trace({llama_expert_trace_kind::projection, tensor, 0, tokens}, pimpl->trace_user);
    }
}

bool llama_expert_store::enable_class_lru(size_t working_bytes, bool reclaim) {
    if (pimpl->history) { return true; }
    const size_t bytes = pimpl->policy_bytes();
    const auto memory = pimpl->budget->snapshot(llama_memory_domain::host);
    size_t fixed = memory.used();
    if (reclaim) {
        for (const auto & item : pimpl->entries) {
            const auto & e = *item.second;
            if (!e.pins) { fixed -= e.buffer_charge.size() + e.arena_charge.size(); }
        }
    }
    if (fixed > memory.requested || bytes > memory.requested - fixed ||
        working_bytes > memory.requested - fixed - bytes) { return false; }
    auto charge = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::runtime, bytes);
    while (!charge && reclaim && pimpl->evict_one()) {
        charge = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::runtime, bytes);
    }
    if (!charge) { return false; }
    const auto & tensors = pimpl->source->tensors();
    auto table = std::unique_ptr<impl::policy_tensor[]>(new impl::policy_tensor[tensors.size()]);
    size_t count = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        table[i].offset = count;
        if (tensors[i].can_read_experts()) { count += size_t(tensors[i].ne[2]); }
    }
    auto history = std::unique_ptr<impl::policy_history[]>(new impl::policy_history[count]);
    charge.commit();
    pimpl->policy_charge = std::move(charge);
    pimpl->policy_tensors = std::move(table);
    pimpl->history = std::move(history);
    return true;
}

void llama_expert_store::disable_class_lru() {
    pimpl->frequency_prefill = false;
    if (!pimpl->history) { return; }
    // Merge the already ordered classes into global acquisition order.
    // Keep weights and leases resident and release only policy metadata.
    auto heads = pimpl->lru_first;
    pimpl->lru_first.fill(nullptr);
    pimpl->lru_last.fill(nullptr);
    for (;;) {
        entry ** oldest = nullptr;
        for (auto & head : heads) {
            if (head && (!oldest || head->last_use < (*oldest)->last_use)) { oldest = &head; }
        }
        if (!oldest) { break; }
        auto * e = *oldest;
        *oldest = e->newer;
        e->frequency_class = 0;
        pimpl->append_lru(*e);
    }
    pimpl->active_tensor = pimpl->active_tokens = 0;
    pimpl->history.reset();
    pimpl->policy_tensors.reset();
    pimpl->policy_charge.reset();
}

bool llama_expert_store::set_policy(const std::string & policy, size_t working_bytes) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->clock || pimpl->trace || (policy != "lru" && policy != "class-lru")) { return false; }
    pimpl->frequency_prefill = false;
    if (policy == "class-lru") { return enable_class_lru(working_bytes, false); }
    disable_class_lru();
    return true;
}

bool llama_expert_store::set_prefill_class_lru(bool enabled, size_t working_bytes) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->trace) { return false; }
    if (enabled) {
        const bool admitted = enable_class_lru(working_bytes, true);
        pimpl->frequency_prefill = admitted;
        return admitted;
    }
    disable_class_lru();
    return true;
}

size_t llama_expert_store::class_lru_history_bytes() const {
    return pimpl->policy_charge.size();
}

size_t llama_expert_store::resident_bytes() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->counters.resident_bytes;
}

llama_expert_store_stats llama_expert_store::stats() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto result = pimpl->counters;
    result.entries = pimpl->entries.size();
    for (const auto & item : pimpl->entries) {
        result.pinned_entries += item.second->pins != 0;
    }
    return result;
}

llama_expert_store::lease::lease(std::shared_ptr<impl> owner, std::shared_ptr<entry> value) :
    owner(std::move(owner)), value(std::move(value)) {}

llama_expert_store::lease::~lease() { reset(); }
llama_expert_store::lease::lease(lease && other) noexcept { *this = std::move(other); }
llama_expert_store::lease & llama_expert_store::lease::operator=(lease && other) noexcept {
    if (this != &other) {
        reset();
        owner = std::move(other.owner);
        value = std::move(other.value);
    }
    return *this;
}

ggml_tensor * llama_expert_store::lease::tensor() const { return value ? value->tensor : nullptr; }
uint64_t llama_expert_store::lease::generation() const { return value ? value->generation : 0; }

void llama_expert_store::lease::reset() noexcept {
    if (!value) {
        return;
    }
    auto keep = std::move(owner);
    llama_expert_timer timer(keep->profile ? &keep->profile->release_ns : nullptr);
    std::lock_guard<std::mutex> lock(keep->mutex);
    ggml_backend_synchronize(keep->backend);
    --value->pins;
    if (keep->trace) {
        keep->trace({llama_expert_trace_kind::release, value->key.first, value->key.second}, keep->trace_user);
    }
    if (value->pins == 0 && value->retiring) {
        keep->unlink_lru(*value);
        keep->counters.resident_bytes -= value->arena_charge.size() + value->buffer_charge.size();
        keep->entries.erase(value->key);
        ++keep->counters.evictions;
    }
    value.reset(); // destruction and ledger release only after synchronization
}
