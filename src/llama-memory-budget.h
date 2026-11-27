#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <memory>

enum class llama_memory_domain { host, device };
enum class llama_memory_category { weights, experts, state, scratch, drafter, io, runtime, count };

struct llama_memory_charge {
    size_t reserved = 0;
    size_t allocated = 0;
    size_t retiring = 0; // still physically allocated, not reusable capacity
    size_t total() const { return reserved + allocated + retiring; }
};

struct llama_memory_budget_snapshot {
    size_t requested = 0;
    size_t applied = 0;
    size_t high_water = 0;
    size_t released = 0; // completed allocation releases, not cancelled reservations
    std::array<llama_memory_charge, size_t(llama_memory_category::count)> categories {};
    size_t used() const;
    bool pending() const { return requested != applied; }
};

// Accounts only explicitly admitted allocations, not RSS or hidden driver pools.
// A single move-only ticket owns each allocation; aliases do not get tickets.
// With unified memory, device and host requests share the host budget.
class llama_memory_budget {
    struct impl;
public:
    class ticket {
    public:
        ticket() = default;
        ~ticket();
        ticket(ticket && other) noexcept;
        ticket & operator=(ticket && other) noexcept;
        ticket(const ticket &) = delete;
        ticket & operator=(const ticket &) = delete;

        explicit operator bool() const { return bool(owner); }
        size_t size() const { return bytes; }
        void commit(); // after successful allocation, without changing its charge
        void retire(); // no new users; charge remains until actual release
        // The resource must be destroyed BEFORE resetting/destroying its ticket.
        void reset() noexcept;

    private:
        friend class llama_memory_budget;
        ticket(std::shared_ptr<impl> owner, size_t domain, size_t category, size_t bytes);
        std::shared_ptr<impl> owner;
        size_t domain = 0;
        size_t category = 0;
        size_t bytes = 0;
        enum class phase { reserved, allocated, retiring } state = phase::reserved;
    };

    llama_memory_budget(size_t host_limit, size_t device_limit, bool unified);
    // Returns an empty ticket on insufficient capacity. No allocation is allowed
    // until a nonempty ticket is obtained. Zero-byte reservations are invalid.
    ticket reserve(llama_memory_domain domain, llama_memory_category category, size_t bytes);
    // Set only for failed optional Spark startup, after any reclamation retry.
    std::atomic<bool> spark_admission_failed{false};
    // Admissions immediately honor the new target; applied changes only once all
    // outstanding reservations and allocations fit. Growth applies immediately.
    void request(llama_memory_domain domain, size_t target);
    llama_memory_budget_snapshot snapshot(llama_memory_domain domain) const;

private:
    std::shared_ptr<impl> pimpl;
};
