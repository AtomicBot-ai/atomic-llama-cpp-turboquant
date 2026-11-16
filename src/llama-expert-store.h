#pragma once

#include "llama-expert-source.h"
#include "llama-memory-budget.h"
#include "llama-expert-trace.h"
#include "llama-expert-profile.h"
#include "ggml-backend.h"

#include <cstdint>
#include <memory>

struct llama_expert_store_stats {
    size_t entries = 0;
    size_t pinned_entries = 0;
    size_t resident_bytes = 0; // backend buffers plus their tensor arenas
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t evictions = 0;
    uint64_t logical_read_bytes = 0;
    uint64_t buffer_allocations = 0;
    uint64_t buffer_reuses = 0;
};

// Synchronous publication into separately releasable backend buffers. Optional
// prefetch uses the same admission, ownership and source validation as demand.
// The backend must outlive the store AND its leases. Backend execution by callers
// must be serialized as required by that backend. Retain a lease for every graph
// that references its tensor, including graphs retained for later execution.
// Direct host writes complete before acquire returns but do not synchronize
// unrelated in-flight GPU work; lease release still synchronizes before reuse.
class llama_expert_store {
    struct impl;
    struct entry;
public:
    class lease {
    public:
        lease() = default;
        ~lease();
        lease(lease && other) noexcept;
        lease & operator=(lease && other) noexcept;
        lease(const lease &) = delete;
        lease & operator=(const lease &) = delete;

        // Borrowed weight tensor: callers must not modify its shape or bytes.
        ggml_tensor * tensor() const;
        uint64_t generation() const;
        explicit operator bool() const { return bool(value); }
        // Synchronizes backend work before dropping the pin. Releasing an idle
        // graph's lease also makes its tensor pointer invalid for future submits.
        void reset() noexcept;

    private:
        friend class llama_expert_store;
        lease(std::shared_ptr<impl> owner, std::shared_ptr<entry> value);
        std::shared_ptr<impl> owner;
        std::shared_ptr<entry> value;
    };

    // allocation_granularity must cover the backend's physical buffer rounding.
    // For Metal use the OS page size, not its 32-byte tensor alignment. The ledger
    // covers buffer requests, tensor arenas and read-staging headroom (reserved
    // even for direct host reads), not hidden pools.
    llama_expert_store(std::shared_ptr<llama_expert_source> source,
                       std::shared_ptr<llama_memory_budget> budget,
                       ggml_backend_t backend, llama_memory_domain buffer_domain,
                       size_t allocation_granularity,
                       ggml_backend_buffer_type_t buft = nullptr);
    ~llama_expert_store();

    lease acquire(const std::string & name, int64_t expert);
    // Read-only residency query; expert < 0 requires the entire tensor resident.
    // Does not touch replacement order, policy history or counters.
    bool contains(const std::string & name, int64_t expert = -1) const;
    // Hint only missing known demand, without admission or replacement updates.
    void advise_expert(const std::string & name, int64_t expert);
    llama_memory_budget_snapshot request_budget(llama_memory_domain domain, size_t target);
    // Constant-time physical size of published entries, including retiring leases.
    // Unpublished allocations and staging remain covered by the shared ledger.
    size_t resident_bytes() const;
    llama_expert_store_stats stats() const;
    void trim(); // retire all entries, free those without leases
    void evict_idle(); // reclaim cache capacity while a graph still owns leases
    void set_read_probe(bool (*callback)(void *) noexcept, void * user, bool worker,
            llama_expert_read_probe submitted_read); // idle only
    void set_profile(llama_expert_profile_data * profile); // idle, caller-owned
    void set_trace(llama_expert_trace_callback callback, void * user);
    void begin_projection(const std::string & name, int64_t tokens);
    // Configure only before the first acquisition. History is admitted separately.
    bool set_policy(const std::string & policy, size_t working_bytes = 0);
    // At an idle batch boundary, switch policy while keeping resident weights.
    // Automatic prefill uses decayed-frequency candidates from both class ends;
    // explicit class-lru retains its original age/class victim rule.
    bool set_prefill_class_lru(bool enabled, size_t working_bytes = 0);
    size_t class_lru_history_bytes() const; // idle only

    llama_expert_store(const llama_expert_store &) = delete;
    llama_expert_store & operator=(const llama_expert_store &) = delete;

private:
    bool enable_class_lru(size_t working_bytes, bool reclaim); // mutex held
    void disable_class_lru(); // mutex held
    lease acquire_once(const std::string & name, int64_t expert);
    std::shared_ptr<impl> pimpl;
};
