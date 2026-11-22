#include <cstdlib>
#include <cstdio>
#include "llama-expert-store.h"
#include "llama-expert-lookahead.h"
#include "llama-expert-route-stage.h"

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
    bool loading = false;
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
    bool compact_buffers;
    llama_memory_domain buffer_domain;
    size_t granularity;
    std::mutex mutex;
    std::map<expert_key, std::shared_ptr<entry>> entries;
    // Like entry control metadata, this fixed index is covered by the runtime
    // envelope. Every retained buffer AND tensor arena keeps its original ticket.
    struct recycle_pool {
        static constexpr size_t max_buffer_bytes = 16 * 1024 * 1024;
        static constexpr size_t max_pool_bytes = 256 * 1024 * 1024;
        std::array<std::shared_ptr<entry>, 256> slots {};
    };
    std::unique_ptr<recycle_pool> recycled;
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
    bool inkling_loop_age = false;

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
        // Adapted from upstream moe_cache_lru (PR #31233): eight classes with
        // history retained after eviction. The Inkling experiment extends
        // the half-life because the same physical layers run multiple times.
        const float half_life = inkling_loop_age ? 512.0f : 128.0f;
        h.crf = h.crf * std::exp2(-float(now - h.last) / half_life) + 1.0f;
        h.last = now;
        return std::min(int(class_count - 1), int(std::log2(h.crf)));
    }
    llama_expert_store_stats counters;
    llama_expert_profile_data * profile = nullptr;
    llama_expert_trace_callback trace = nullptr;
    void * trace_user = nullptr;

    // Retain one microbatch's routed experts for the other projections of the
    // same layer. The published per-tensor cache and its GPU leases are unchanged.
    struct pack_staging {
        static constexpr size_t slots = 8 * 10;
        llama_memory_budget::ticket charge;
        std::unique_ptr<uint8_t[]> bytes;
        size_t stride;
        const llama_expert_source::packed_group * layer = nullptr;
        std::array<int16_t, 1024> by_expert;
        std::array<int16_t, slots> by_slot;
        size_t next = 0;

        pack_staging(llama_memory_budget::ticket reservation, size_t stride) :
            charge(std::move(reservation)), bytes(new uint8_t[stride * slots]), stride(stride) { charge.commit(); }

        bool contains(const llama_expert_source::packed_group * group, int64_t expert) const {
            return layer == group && by_expert[expert] >= 0;
        }

        uint8_t * load(llama_expert_source & source, const llama_expert_source::packed_group & group,
                        int64_t expert, llama_expert_store_stats & counters) {
            if (layer != &group) {
                layer = &group;
                by_expert.fill(-1);
                by_slot.fill(-1);
                next = 0;
            }
            int slot = by_expert[expert];
            if (slot >= 0) {
                ++counters.packed_reuses;
            } else {
                slot = next;
                next = (next + 1) % slots;
                const int old = by_slot[slot];
                if (old >= 0) { by_expert[old] = -1; }
                by_slot[slot] = -1;
                source.read_packed_expert(group, expert, bytes.get() + slot * stride);
                by_expert[expert] = slot;
                by_slot[slot] = expert;
                ++counters.packed_reads;
                counters.logical_read_bytes += group.stride;
            }
            return bytes.get() + slot * stride;
        }
    };
    std::unique_ptr<pack_staging> packed;

    size_t trace_capacity() const {
        const auto memory = budget->snapshot(llama_memory_domain::host);
        const size_t fixed = memory.used() - counters.resident_bytes - counters.pooled_bytes;
        return memory.requested > fixed ? memory.requested - fixed : 0;
    }

    impl(std::shared_ptr<llama_expert_source> source, std::shared_ptr<llama_memory_budget> budget,
         ggml_backend_t backend, llama_memory_domain domain, size_t granularity, ggml_backend_buffer_type_t buft, bool compact_buffers, bool recycle_buffers) :
        source(std::move(source)), budget(std::move(budget)), backend(backend), buft(buft),
        compact_buffers(compact_buffers), buffer_domain(domain), granularity(granularity) {
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
        reuse_zero_tail = host_base && (compact_buffers || this->source->architecture() == "qwen4exp");
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
        if (compact_buffers && recycle_buffers) { recycled = std::make_unique<recycle_pool>(); }
        if (const size_t stride = host_base ? this->source->packed_stride() : 0) {
            auto reservation = this->budget->reserve(llama_memory_domain::host, llama_memory_category::io,
                    stride * pack_staging::slots + sizeof(pack_staging));
            if (reservation) {
                packed.reset(new pack_staging(std::move(reservation), stride));
            }
        }
    }

    uint64_t next_clock() {
        if (clock == std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error("expert store generation exhausted");
        }
        return ++clock;
    }

    size_t slice_buffer_bytes(const llama_expert_tensor_info & info) const {
        const size_t data_bytes = info.expert_bytes();
        ggml_tensor probe {};
        probe.type = info.type;
        probe.ne[0] = info.ne[0]; probe.ne[1] = info.ne[1]; probe.ne[2] = 1; probe.ne[3] = 1;
        probe.nb[0] = ggml_type_size(info.type);
        probe.nb[1] = ggml_row_size(info.type, info.ne[0]);
        probe.nb[2] = data_bytes; probe.nb[3] = data_bytes;
        const size_t alignment = std::max(granularity, ggml_backend_buft_get_alignment(buft));
        const size_t bytes = round_allocation(ggml_backend_buft_get_alloc_size(buft, &probe), alignment);
        if (bytes > ggml_backend_buft_get_max_size(buft)) {
            throw std::runtime_error("expert exceeds maximum backend buffer size");
        }
        return bytes;
    }

    // Flash Next can recycle Metal buffers within the same aligned size class.
    // Other geometries fall back to the existing LRU admission policy.
    static constexpr size_t slot_alignment = 64 * 1024;
    struct size_index {
        llama_memory_budget::ticket charge;
        std::array<entry *, 16> oldest {};
    };
    std::unique_ptr<size_index> size_lru;

    static bool indexed_size(size_t bytes) {
        return bytes && bytes <= 16 * slot_alignment && bytes % slot_alignment == 0;
    }

    static entry * next_size(entry * start, size_t bytes) {
        while (start && start->buffer_charge.size() != bytes) { start = start->newer; }
        return start;
    }

    void unlink_lru(entry & e) {
        if (size_lru && !history && indexed_size(e.buffer_charge.size())) {
            auto & first = size_lru->oldest[e.buffer_charge.size() / slot_alignment - 1];
            if (first == &e) { first = next_size(e.newer, e.buffer_charge.size()); }
        }
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

    std::shared_ptr<entry> take_victim(size_t preferred_bytes = 0) {
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
                    double score = std::ldexp(double(tensor.clock - h.last), -int(cls));
                    if (inkling_loop_age && active_tokens == 4 && tensor.clock > 1) {
                        // Reused core layers advance their local age several times per
                        // verification. Partially normalize that age across layers.
                        score /= std::sqrt(double(tensor.clock));
                    }
                    if (score > best_score) {
                        best_score = score;
                        chosen = candidate;
                    }
                }
            }
        } else {
            chosen = oldest_unpinned(0);
            if (indexed_size(preferred_bytes)) {
                auto & first = size_lru->oldest[preferred_bytes / slot_alignment - 1];
                if (!first) { first = next_size(lru_first[0], preferred_bytes); }
                auto * candidate = first;
                while (candidate && (candidate->pins || candidate->retiring)) {
                    candidate = next_size(candidate->newer, preferred_bytes);
                }
                if (candidate) { chosen = candidate; }
            }
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

    std::shared_ptr<entry> take_recycled(size_t buffer_bytes, size_t arena_bytes) {
        if (!recycled || trace) { return {}; }
        for (auto & slot : recycled->slots) {
            if (slot && slot->buffer_charge.size() == buffer_bytes && slot->arena_charge.size() == arena_bytes) {
                counters.pooled_bytes -= buffer_bytes + arena_bytes;
                --counters.pooled_buffers;
                return std::move(slot);
            }
        }
        return {};
    }

    void recycle(std::shared_ptr<entry> victim) {
        if (!recycled || trace || victim->retiring || victim->pins ||
                victim->buffer_charge.size() > recycle_pool::max_buffer_bytes ||
                counters.pooled_bytes + victim->buffer_charge.size() + victim->arena_charge.size() > recycle_pool::max_pool_bytes) { return; }
        size_t matching = 0;
        std::shared_ptr<entry> * empty = nullptr;
        for (auto & slot : recycled->slots) {
            if (!slot) { if (!empty) { empty = &slot; } }
            else if (slot->buffer_charge.size() == victim->buffer_charge.size()) { ++matching; }
        }
        // Overflow victims are destroyed normally, releasing room for admission.
        if (empty && matching < 64) {
            counters.pooled_bytes += victim->buffer_charge.size() + victim->arena_charge.size();
            ++counters.pooled_buffers;
            *empty = std::move(victim);
        }
    }

    bool release_recycled() {
        if (!recycled) { return false; }
        for (auto & slot : recycled->slots) {
            if (!slot) { continue; }
            counters.pooled_bytes -= slot->buffer_charge.size() + slot->arena_charge.size();
            --counters.pooled_buffers;
            slot.reset();
            return true;
        }
        return false;
    }

    // Reclamation and resize free unpublished spare buffers before useful weights.
    bool evict_one() { return release_recycled() || bool(take_victim()); }

    void trim() {
        while (release_recycled()) {}
        if (packed) { packed->layer = nullptr; }
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
        llama_memory_domain buffer_domain, size_t allocation_granularity, ggml_backend_buffer_type_t buft, bool compact_buffers, bool recycle_buffers) :
    pimpl(std::make_shared<impl>(std::move(source), std::move(budget), backend,
                                buffer_domain, allocation_granularity, buft, compact_buffers, recycle_buffers)) {}

llama_expert_store::~llama_expert_store() { trim(); }

bool llama_expert_store::contains(const std::string & name, int64_t expert) const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    const auto & info = pimpl->source->tensor(name);
    const size_t index = size_t(&info - pimpl->source->tensors().data());
    if (expert >= info.ne[2]) { return false; }
    for (int64_t i = expert < 0 ? 0 : expert; i < (expert < 0 ? info.ne[2] : expert + 1); ++i) {
        const auto found = pimpl->entries.find({index, i});
        if (found == pimpl->entries.end() || found->second->retiring || found->second->loading) { return false; }
    }
    return true;
}

void llama_expert_store::advise_expert(const std::string & name, int64_t expert) {
    if (pimpl->packed && !pimpl->trace && expert >= 0) {
        const auto & info = pimpl->source->tensor(name);
        const auto index = size_t(&info - pimpl->source->tensors().data());
        const auto slice = pimpl->source->packed_location(index);
        if (slice.group && expert < info.ne[2]) {
            std::lock_guard<std::mutex> lock(pimpl->mutex);
            const auto resident = pimpl->entries.find({index, expert});
            if ((resident == pimpl->entries.end() || resident->second->retiring) &&
                    !pimpl->packed->contains(slice.group, expert)) {
                pimpl->source->advise_packed_expert(*slice.group, expert);
            }
            return;
        }
    }
    if (!contains(name, expert)) { pimpl->source->advise_expert(name, expert); }
}

llama_expert_store::lease llama_expert_store::acquire(const std::string & name, int64_t expert) {
    auto result = acquire_once(name, expert);
    // The one-shot cancellation has released all unpublished owners. Retry
    // the original demand from its GGUF offset through the normal path.
    return result ? std::move(result) : acquire_once(name, expert);
}

llama_expert_store::lease llama_expert_store::acquire_once(const std::string & name, int64_t expert) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return acquire_locked(name, expert, nullptr);
}

bool llama_expert_store::acquire_resident_many(const std::string & name, const int64_t * experts, size_t count, lease * results) {
    if (!count || count > 16 || std::any_of(results, results + count, [](const lease & out) { return bool(out); })) {
        throw std::invalid_argument("invalid resident expert acquisition");
    }
    std::array<lease, 16> held;
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    const auto & info = pimpl->source->tensor(name);
    const size_t index = &info - pimpl->source->tensors().data();
    for (size_t i = 0; i < count; ++i) {
        if (experts[i] < 0 || experts[i] >= info.ne[2]) { throw std::invalid_argument("invalid resident expert ID"); }
        const auto found = pimpl->entries.find({index, experts[i]});
        if (found == pimpl->entries.end() || found->second->retiring || found->second->loading) { return false; }
    }
    for (size_t i = 0; i < count; ++i) { held[i] = acquire_locked(name, experts[i], nullptr); }
    std::move(held.begin(), held.begin() + count, results);
    return true;
}

size_t llama_expert_store::compact_slice_bytes(const std::string & name) const {
    if (!pimpl->compact_buffers) { return 0; }
    return pimpl->slice_buffer_bytes(pimpl->source->tensor(name)) + ggml_tensor_overhead();
}

void llama_expert_store::acquire_many(const std::string & name, const int64_t * experts, size_t count, lease * results,
        llama_expert_lookahead * lookahead, llama_expert_read_layout layout) {
    std::array<llama_expert_source::read_request, 16> pending {}, reads;
    if (!count || count > pending.size() || std::any_of(results, results + count, [](const lease & out) { return bool(out); })) {
        throw std::invalid_argument("invalid batched expert acquisition");
    }
    // Destroy local leases only after unlocking, since release locks the store.
    std::array<lease, 16> admitted;
    size_t n_reads = 0;
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    try {
        // Insert in the normal LRU order while holding the mutex. Nothing can
        // observe the new slots until the mutex is released. The output leases
        // also exclude these slots from eviction during later admissions.
        for (size_t i = 0; i < count; ++i) {
            admitted[i] = acquire_locked(name, experts[i], &pending[i]);
            if (!admitted[i]) { throw std::runtime_error("batched expert acquisition cancelled"); }
            if (pending[i].tensor) {
                const size_t copied = lookahead ? lookahead->consume(pending[i], pimpl->profile) : 0;
                if (!copied) { reads[n_reads++] = pending[i]; }
                else if (pimpl->profile) { pimpl->profile->next_prefetch_used_bytes += copied; }
            }
        }
        if (n_reads) {
            llama_expert_timer timer(pimpl->profile ? &pimpl->profile->read_ns : nullptr);
            pimpl->source->read_experts(reads.data(), n_reads, false, false, layout);
            if (n_reads > 1) {
                ++pimpl->counters.parallel_batches;
                pimpl->counters.parallel_reads += n_reads;
            }
            for (size_t i = 0; i < n_reads; ++i) {
                const size_t bytes = reads[i].tensor->expert_bytes();
                pimpl->counters.logical_read_bytes += bytes;
                if (pimpl->profile) { pimpl->profile->direct_read_bytes += bytes; }
            }
        }
    } catch (...) {
        // No incomplete data may become visible after unlocking. Local
        // retiring leases synchronize and release their charges on unwind.
        for (size_t i = 0; i < count; ++i) {
            if (pending[i].tensor && admitted[i]) { admitted[i].value->retire(); }
        }
        throw;
    }
    std::move(admitted.begin(), admitted.begin() + count, results);
}

struct llama_expert_store::demand_batch::state {
    static constexpr size_t limit = 16;
    struct expert {
        int64_t id = 0;
        std::array<lease, 3> weights;
        std::array<llama_expert_source::read_request, 3> reads{};
        unsigned count = 0, completed = 0;
        unsigned reused_reads = 0; // bit mask of completed reads supplied by staging
        bool ready = false, taken = false;
    };

    std::shared_ptr<impl> owner;
    std::array<expert, limit> experts;
    std::array<size_t, 3 * limit> jobs{};
    bool slice_jobs = false;
    struct scheduled_jobs {
        struct job {state * owner;size_t index;};
        std::array<early_io::pool::task,3*limit> tasks{};
        std::array<job,3*limit> contexts{};
        std::array<early_io::pool::task *,3*limit> adopted{};
        size_t count=0;
        size_t adopted_count=0;
    };
    struct early_times { std::array<uint64_t, 3 * limit> begin{}, end{}; };
    std::unique_ptr<early_times> timing;
    early_io::pool * priority_pool = nullptr;
    llama_memory_budget::ticket scheduled_charge;
    std::unique_ptr<scheduled_jobs> scheduled;
    size_t count = 0, remaining = 0, n_jobs = 0;
    std::mutex mutex;
    std::condition_variable changed;
    size_t next_job = 0, ready = 0;
    std::exception_ptr error;
#ifdef __APPLE__
    dispatch_group_t group = nullptr;
#endif

    explicit state(std::shared_ptr<impl> owner) : owner(std::move(owner)) {}

    void work() noexcept {
        for (;;) {
            expert * item;
            size_t part = 0;
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (error || next_job == n_jobs) { return; }
                const size_t job = jobs[next_job++];
                item = &experts[slice_jobs ? job / 3 : job];
                part = slice_jobs ? job % 3 : 0;
            }
            try {
                // All projections share the same worker queue. A successful
                // triple is validated before it can be taken by the graph owner.
                owner->source->read_experts(item->reads.data() + part, slice_jobs ? 1 : item->count, false, true);
            } catch (...) {
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!error) { error = std::current_exception(); }
                }
                changed.notify_one();
                return;
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                if (!slice_jobs || ++item->completed == item->count) {
                    item->ready = true;
                    ++ready;
                }
            }
            changed.notify_one();
        }
    }

    void priority_job(size_t job, const void * staged_data = nullptr) noexcept {
        expert * item = &experts[job/3];
        const auto & read = item->reads[job%3];
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (error) return;
            if (timing) timing->begin[job] = early_io::now();
        }
        try {
            if (staged_data) std::memcpy(read.destination, staged_data, read.tensor->expert_bytes());
            else owner->source->read_experts(&read, 1, false, true);
        }
        catch (...) {
            {std::lock_guard<std::mutex> lock(mutex);if (!error) error=std::current_exception();}
            changed.notify_one();return;
        }
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (staged_data) item->reused_reads |= 1u << (job%3);
            if (timing) timing->end[job] = early_io::now();
            if (++item->completed==item->count) {item->ready=true;++ready;}
        }
        changed.notify_one();
    }
    ~state() {
        if (scheduled) {
            for (size_t i = 0; i < scheduled->adopted_count; ++i) priority_pool->wait(*scheduled->adopted[i]);
            for (size_t i=0;i<scheduled->count;++i) priority_pool->wait(scheduled->tasks[i]);
        }
#ifdef __APPLE__
        if (group) {
            dispatch_group_wait(group, DISPATCH_TIME_FOREVER);
            dispatch_release(group);
        }
#endif
        // Queued misses that never escaped are still unpublished, even when
        // they completed before a failure elsewhere. Drop them after the join.
        std::lock_guard<std::mutex> lock(owner->mutex);
        for (size_t i = 0; i < count; ++i) {
            for (auto & held : experts[i].weights) {
                if (held && held.value->loading) { held.value->retire(); }
            }
        }
    }
};

llama_expert_store::demand_batch::demand_batch(std::shared_ptr<impl> owner) : impl_(new state(std::move(owner))) {}
llama_expert_store::demand_batch::~demand_batch() = default;
bool llama_expert_store::demand_batch::pending() const { return impl_->remaining != 0; }

void llama_expert_store::demand_batch::early_read_overlap(uint64_t cutoff, uint64_t & first, uint64_t & overlap, uint64_t & bytes) const {
    auto & s = *impl_;
    first = overlap = bytes = 0;
    std::lock_guard<std::mutex> lock(s.mutex);
    if (!s.timing) return;
    std::array<std::pair<uint64_t, uint64_t>, 3 * state::limit> intervals{};
    size_t n = 0;
    for (size_t j = 0; j < s.n_jobs; ++j) {
        const size_t k = s.jobs[j];
        if (!s.timing->begin[k] || s.timing->begin[k] >= cutoff) continue;
        if (!first || s.timing->begin[k] < first) first = s.timing->begin[k];
        intervals[n++] = {s.timing->begin[k], s.timing->end[k] ? std::min(cutoff, s.timing->end[k]) : cutoff};
        if (s.timing->end[k] && s.timing->end[k] <= cutoff) bytes += s.experts[k/3].reads[k%3].tensor->expert_bytes();
    }
    std::sort(intervals.begin(), intervals.begin() + n);
    uint64_t end = 0;
    for (size_t i = 0; i < n; ++i) {
        const auto begin = std::max(end, intervals[i].first);
        if (intervals[i].second > begin) overlap += intervals[i].second - begin;
        end = std::max(end, intervals[i].second);
    }
}

size_t llama_expert_store::demand_batch::take(int64_t * ids, lease * up, lease * gate, lease * down, size_t min_ready) {
    auto & s = *impl_;
    if (!s.remaining) { return 0; }
    std::array<size_t, state::limit> ready{};
    size_t count = 0;
    auto * profile = s.owner->profile;
    {
        std::unique_lock<std::mutex> lock(s.mutex);
        const size_t threshold = std::min(std::max<size_t>(1, min_ready), s.remaining);
        if (!s.error && s.ready < threshold) {
            llama_expert_timer timer(profile ? &profile->demand_queue_wait_ns : nullptr);
            llama_expert_timer read(profile ? &profile->read_ns : nullptr);
            s.changed.wait(lock, [&] { return s.error || s.ready >= threshold; });
        }
        if (s.error) { std::rethrow_exception(s.error); }
        for (size_t i = 0; i < s.count; ++i) {
            auto & item = s.experts[i];
            if (item.ready && !item.taken) {
                ready[count++] = i;
                item.taken = true;
            }
        }
        s.ready -= count;
        s.remaining -= count;
    }
    std::lock_guard<std::mutex> lock(s.owner->mutex);
    std::array<lease *, 3> outputs{up, gate, down};
    for (size_t i = 0; i < count; ++i) {
        auto & item = s.experts[ready[i]];
        ids[i] = item.id;
        for (size_t j = 0; j < item.count; ++j) {
            const size_t bytes = item.reads[j].tensor->expert_bytes();
            if (item.reused_reads & (1u << j)) {
                if (profile) {
                    profile->route_stage_joined_bytes += bytes;
                    profile->route_stage_used_bytes += bytes;
                }
                continue;
            }
            s.owner->counters.logical_read_bytes += bytes;
            if (profile) { profile->direct_read_bytes += bytes; }
        }
        for (size_t j = 0; j < outputs.size(); ++j) {
            if (item.weights[j].value->retiring) { throw std::runtime_error("queued expert is retiring"); }
            item.weights[j].value->loading = false;
            outputs[j][i] = std::move(item.weights[j]);
        }
    }
    if (profile) {
        ++profile->demand_queue_ready_groups;
        profile->demand_queue_partial_groups += s.remaining > 0;
    }
    return count;
}

size_t llama_expert_store::demand_queue_bytes(size_t experts) {
    constexpr size_t batch_size = demand_batch::state::limit;
    const size_t batches = experts / batch_size + (experts % batch_size != 0);
    return batches * (sizeof(demand_batch) + sizeof(demand_batch::state) +
            sizeof(demand_batch::state::scheduled_jobs) + 4096);
}

std::unique_ptr<llama_expert_store::demand_batch> llama_expert_store::enqueue_demand(
        const std::array<const char *, 3> & names, const int64_t * experts, size_t count,
        llama_expert_lookahead * lookahead, early_io::pool * readers, bool early_timing, llama_expert_route_stage * staged) {
#ifdef __APPLE__
    if (!count || count > demand_batch::state::limit || pimpl->trace || pimpl->history || pimpl->source->packed_stride()) {
        throw std::invalid_argument("unsupported queued expert demand");
    }
    auto * shared_readers = readers ? readers : lookahead ? lookahead->priority_pool() : nullptr;
    llama_memory_budget::ticket metadata;
    if (shared_readers) {
        const size_t bytes = demand_queue_bytes(count) + (early_timing ? sizeof(demand_batch::state::early_times) : 0);
        // Admit before allocating task descriptors. Idle cache entries may fund
        // metadata, but live leases held by the GPU remain pinned.
        if (!reclaim_to_fit(llama_memory_domain::host, bytes)) {
            throw std::runtime_error("demand queue metadata does not fit");
        }
        metadata = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::io, bytes);
        if (!metadata) { throw std::runtime_error("demand queue admission failed"); }
    }
    std::unique_ptr<demand_batch> batch(new demand_batch(pimpl));
    auto & s = *batch->impl_;
    s.priority_pool = shared_readers;
    if (early_timing && shared_readers) s.timing = std::make_unique<demand_batch::state::early_times>();
    if (s.priority_pool) {
        s.slice_jobs = true;
        s.scheduled_charge = std::move(metadata);
        s.scheduled = std::make_unique<demand_batch::state::scheduled_jobs>();
        s.scheduled_charge.commit();
    }
    early_io::pool::pause pause_background(s.priority_pool);
    s.count = s.remaining = count;
    {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        // Admit in the same projection/expert order as the ordinary pipeline.
        // Pin direct destinations and mask them from other readers until take().
        for (size_t p = 0; p < names.size(); ++p) {
            const std::string name(names[p]);
            for (size_t i = 0; i < count; ++i) {
                auto & item = s.experts[i];
                item.id = experts[i];
                llama_expert_source::read_request read{};
                auto & lease = item.weights[p];
                lease = acquire_locked(name, experts[i], &read);
                if (!lease) { throw std::runtime_error("queued expert acquisition cancelled"); }
                if (read.tensor) {
                    lease.value->loading = true;
                    if (const size_t copied = staged && s.priority_pool ? staged->consume_ready(read, pimpl->profile) : 0) {
                        if (pimpl->profile) { pimpl->profile->route_stage_used_bytes += copied; }
                        continue;
                    }
                    const size_t copied = lookahead ? lookahead->consume(read, pimpl->profile) : 0;
                    if (!copied) { item.reads[item.count++] = read; }
                    else if (pimpl->profile) { pimpl->profile->next_prefetch_used_bytes += copied; }
                }
            }
        }
    }
    for (size_t i = 0; i < count; ++i) {
        if (s.experts[i].count) {
            if (s.slice_jobs) {
                for (size_t p = 0; p < s.experts[i].count; ++p) s.jobs[s.n_jobs++] = 3*i+p;
            } else { s.jobs[s.n_jobs++] = i; }
        }
        else {
            s.experts[i].ready = true; ++s.ready;
        }
    }
    if (s.n_jobs > 1) {
        std::lock_guard<std::mutex> lock(pimpl->mutex);
        ++pimpl->counters.parallel_batches;
        for (size_t i = 0; i < s.count; ++i) {
            pimpl->counters.parallel_reads += s.experts[i].count;
        }
    }
    if (s.n_jobs && s.priority_pool) {
        auto & pending = *s.scheduled;
        for (size_t i=0;i<s.n_jobs;++i) {
            const size_t index = s.jobs[i];
            auto & job = pending.contexts[i];
            job = {&s, index};
            if (staged) {
                auto * existing = staged->adopt(s.experts[index/3].reads[index%3], [](void * ptr, const void * bytes) noexcept {
                    auto & job = *static_cast<demand_batch::state::scheduled_jobs::job *>(ptr);
                    job.owner->priority_job(job.index, bytes);
                }, &job, pimpl->profile);
                if (existing) {
                    pending.adopted[pending.adopted_count++] = existing;
                    continue;
                }
            }
            auto & task = pending.tasks[pending.count++];
            task.context = &job;
            task.fn = [](void * ptr) noexcept {
                auto & job = *static_cast<demand_batch::state::scheduled_jobs::job *>(ptr);
                job.owner->priority_job(job.index);
            };
        }
        if (pending.count) s.priority_pool->submit(pending.tasks.data(), pending.count, true);
    } else if (s.n_jobs) {
        s.group = dispatch_group_create();
        if (!s.group) { throw std::bad_alloc(); }
        auto queue = dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0);
        for (size_t i = 0; i < std::min<size_t>(4, s.n_jobs); ++i) {
            dispatch_group_async_f(s.group, queue, &s, [](void * opaque) {
                static_cast<demand_batch::state *>(opaque)->work();
            });
        }
    }
    if (pimpl->profile) { ++pimpl->profile->demand_queue_batches; }
    return batch;
#else
    (void) names; (void) experts; (void) count; (void) lookahead; (void) readers; (void) early_timing; (void) staged;
    throw std::invalid_argument("queued expert demand requires Metal");
#endif
}

llama_expert_store::lease llama_expert_store::acquire_locked(const std::string & name, int64_t expert,
        llama_expert_source::read_request * pending) {
    auto * profile = pimpl->profile;
    llama_expert_timer timer(profile ? &profile->admission_ns : nullptr);
    const auto & info = pimpl->source->tensor(name);
    const size_t data_bytes = info.expert_bytes();
    if (expert < 0 || expert >= info.ne[2]) {
        throw std::invalid_argument("invalid expert ID");
    }
    const expert_key key {size_t(&info - pimpl->source->tensors().data()), expert};
    const size_t capacity = pimpl->trace ? pimpl->trace_capacity() : 0;
    const int frequency_class = pimpl->touch(key);
    const auto found = pimpl->entries.find(key);
    if (found != pimpl->entries.end()) {
        auto & e = *found->second;
        if (e.retiring || e.loading) {
            throw std::runtime_error("expert is loading or retiring; release outstanding leases before reacquiring");
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
    const size_t buffer_bytes = pimpl->slice_buffer_bytes(info);
    const size_t arena_bytes = ggml_tensor_overhead();

    impl::reservations charges;
    std::shared_ptr<entry> e;
    bool from_pool = false;
    if (pimpl->recycled && !pimpl->trace) {
        e = pimpl->take_recycled(buffer_bytes, arena_bytes);
        from_pool = bool(e);
        for (;;) {
            if (e) {
                charges.staging = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::io, data_bytes);
                if (charges.staging) { break; }
                // Keep the selected reusable buffer, but release other spares
                // before evicting useful weights to make read headroom.
                if (pimpl->release_recycled()) { continue; }
                if (pimpl->take_victim()) { continue; }
                e.reset();
                from_pool = false;
            }
            charges = pimpl->reserve(arena_bytes, buffer_bytes, data_bytes);
            if (charges.complete()) { break; }
            charges = {};
            auto victim = pimpl->take_victim(pimpl->size_lru ? buffer_bytes : 0);
            if (!victim) {
                // Tight budgets must not fail merely because spares are held.
                if (pimpl->release_recycled()) { continue; }
                throw std::runtime_error("expert working set does not fit the requested managed budget");
            }
            if (!victim->retiring && victim->buffer_charge.size() == buffer_bytes &&
                    victim->arena_charge.size() == arena_bytes) {
                e = std::move(victim);
            } else {
                // Preserve the replacement policy: only already-evicted,
                // unpinned buffers enter the bounded size-class pool.
                pimpl->recycle(std::move(victim));
            }
        }
    } else {
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
            // Compact mode keeps that victim policy but rejects oversized reuse.
            if (!victim->retiring && victim->buffer_charge.size() >= buffer_bytes &&
                (!pimpl->compact_buffers || victim->buffer_charge.size() == buffer_bytes) &&
                victim->arena_charge.size() == arena_bytes) {
                charges.staging = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::io, data_bytes);
                if (charges.staging) { e = std::move(victim); break; }
            }
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
    const auto slice = pimpl->packed && !pimpl->trace && !read_probe && !submitted_read ?
            pimpl->source->packed_location(key.first) : llama_expert_source::packed_slice{};
    const bool deferred = pending && direct && !slice.group && !read_probe && !submitted_read && !pimpl->trace;
    if (deferred) {
        *pending = {&info, expert, direct};
    } else if (slice.group) {
        const auto * group = pimpl->packed->load(*pimpl->source, *slice.group, expert, pimpl->counters);
        std::memcpy(read_dst, group + slice.offset, data_bytes);
    } else {
        const bool complete = read_probe && pimpl->read_probe_worker ?
            probe_worker_read(*pimpl->source, info.name, expert, read_dst, data_bytes, profile, read_probe, pimpl->read_probe_user) :
            pimpl->source->read_expert(info.name, expert, read_dst, data_bytes, profile, read_probe, pimpl->read_probe_user, submitted_read);
        if (!complete) { return {}; }
        if (direct && profile) { profile->direct_read_bytes += data_bytes; }
    }
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
    if (from_pool) { ++pimpl->counters.pool_reuses; }
    if (!slice.group && !deferred) { pimpl->counters.logical_read_bytes += data_bytes; }
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

bool llama_expert_store::reclaim_to_fit(llama_memory_domain domain, size_t bytes) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    // This capability currently reclaims one store's buffer domain. Discrete
    // host metadata/device-payload coordination needs its own qualification.
    if (domain != pimpl->buffer_domain) { return false; }
    ++pimpl->counters.reclaim_calls;
    for (;;) {
        const auto memory = pimpl->budget->snapshot(domain);
        if (bytes > memory.requested) { return false; }
        if (memory.used() <= memory.requested && bytes <= memory.requested - memory.used()) { return true; }
        const auto before = pimpl->counters.resident_bytes + pimpl->counters.pooled_bytes;
        if (!pimpl->evict_one()) { return false; }
        ++pimpl->counters.reclaim_evictions;
        pimpl->counters.reclaim_bytes += before - pimpl->counters.resident_bytes - pimpl->counters.pooled_bytes;
    }
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
    if (callback) { while (pimpl->release_recycled()) {} }
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

void llama_expert_store::enable_size_lru() {
    if (pimpl->size_lru) { return; }
    auto charge = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::runtime, sizeof(impl::size_index));
    if (!charge) { return; }
    std::unique_ptr<impl::size_index> index(new (std::nothrow) impl::size_index{});
    if (!index) { return; }
    charge.commit();
    index->charge = std::move(charge);
    pimpl->size_lru = std::move(index);
}

bool llama_expert_store::enable_class_lru(size_t working_bytes, bool reclaim) {
    if (pimpl->history) { return true; }
    const size_t bytes = pimpl->policy_bytes();
    const auto memory = pimpl->budget->snapshot(llama_memory_domain::host);
    size_t fixed = memory.used();
    if (reclaim) {
        fixed -= pimpl->counters.pooled_bytes;
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
    if (pimpl->size_lru) { pimpl->size_lru->oldest.fill(nullptr); }
    pimpl->history = std::move(history);
    return true;
}

void llama_expert_store::disable_class_lru() {
    pimpl->frequency_prefill = false;
    pimpl->inkling_loop_age = false;
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
    const bool loop_age = policy == "inkling-loop-age" && pimpl->source->architecture() == "inkling2_loop";
    if (pimpl->clock || pimpl->trace || (policy != "lru" && policy != "class-lru" && !loop_age)) { return false; }
    pimpl->frequency_prefill = false;
    if (policy == "class-lru" || loop_age) {
        if (!enable_class_lru(working_bytes, false)) { return false; }
        pimpl->inkling_loop_age = loop_age;
        return true;
    }
    disable_class_lru();
    return true;
}

bool llama_expert_store::set_prefill_class_lru(bool enabled, size_t working_bytes) {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    if (pimpl->trace) { return false; }
    if (enabled) {
        const bool admitted = enable_class_lru(working_bytes, true);
        pimpl->frequency_prefill = admitted;
        pimpl->inkling_loop_age = false;
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

size_t llama_expert_store::storage_bytes() const {
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->counters.resident_bytes + pimpl->counters.pooled_bytes;
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
