#include "llama-memory-budget.h"

#include <algorithm>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>

size_t llama_memory_budget_snapshot::used() const {
    size_t result = 0;
    for (const auto & c : categories) {
        result += c.total();
    }
    return result;
}

struct llama_memory_budget::impl {
    std::mutex mutex;
    bool unified;
    std::array<llama_memory_budget_snapshot, 2> domains;

    size_t index(llama_memory_domain domain) const {
        if (domain != llama_memory_domain::host && domain != llama_memory_domain::device) {
            throw std::invalid_argument("invalid memory domain");
        }
        return unified ? 0 : size_t(domain);
    }

    void apply(size_t domain) {
        auto & d = domains[domain];
        if (d.used() <= d.requested) {
            d.applied = d.requested;
        }
    }
};

llama_memory_budget::llama_memory_budget(size_t host_limit, size_t device_limit, bool unified) :
    pimpl(std::make_shared<impl>()) {
    pimpl->unified = unified;
    pimpl->domains[0].requested = pimpl->domains[0].applied = host_limit;
    pimpl->domains[1].requested = pimpl->domains[1].applied = device_limit;
}

llama_memory_budget::ticket llama_memory_budget::reserve(
        llama_memory_domain domain, llama_memory_category category, size_t bytes) {
    const size_t d = pimpl->index(domain);
    const size_t c = size_t(category);
    if (c >= size_t(llama_memory_category::count) || bytes == 0) {
        throw std::invalid_argument("invalid memory reservation");
    }
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    auto & budget = pimpl->domains[d];
    const size_t used = budget.used();
    if (used > budget.requested || bytes > budget.requested - used) {
        return {};
    }
    budget.categories[c].reserved += bytes;
    budget.high_water = std::max(budget.high_water, used + bytes);
    return ticket(pimpl, d, c, bytes);
}

void llama_memory_budget::request(llama_memory_domain domain, size_t target) {
    const size_t d = pimpl->index(domain);
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    pimpl->domains[d].requested = target;
    pimpl->apply(d);
}

llama_memory_budget_snapshot llama_memory_budget::snapshot(llama_memory_domain domain) const {
    const size_t d = pimpl->index(domain);
    std::lock_guard<std::mutex> lock(pimpl->mutex);
    return pimpl->domains[d];
}

llama_memory_budget::ticket::ticket(std::shared_ptr<impl> owner, size_t domain, size_t category, size_t bytes) :
    owner(std::move(owner)), domain(domain), category(category), bytes(bytes) {}

llama_memory_budget::ticket::~ticket() { reset(); }

llama_memory_budget::ticket::ticket(ticket && other) noexcept { *this = std::move(other); }

llama_memory_budget::ticket & llama_memory_budget::ticket::operator=(ticket && other) noexcept {
    if (this != &other) {
        reset();
        owner = std::move(other.owner);
        domain = other.domain;
        category = other.category;
        bytes = std::exchange(other.bytes, 0);
        state = other.state;
    }
    return *this;
}

void llama_memory_budget::ticket::commit() {
    if (!owner || state != phase::reserved) {
        throw std::logic_error("only a reservation can be committed");
    }
    std::lock_guard<std::mutex> lock(owner->mutex);
    auto & c = owner->domains[domain].categories[category];
    c.reserved -= bytes;
    c.allocated += bytes;
    state = phase::allocated;
}

void llama_memory_budget::ticket::retire() {
    if (!owner || state != phase::allocated) {
        throw std::logic_error("only an allocation can be retired");
    }
    std::lock_guard<std::mutex> lock(owner->mutex);
    auto & c = owner->domains[domain].categories[category];
    c.allocated -= bytes;
    c.retiring += bytes;
    state = phase::retiring;
}

void llama_memory_budget::ticket::reset() noexcept {
    if (!owner) {
        return;
    }
    // Keep the state alive until the mutex is unlocked, even if this is its last ticket.
    auto keep = std::move(owner);
    std::lock_guard<std::mutex> lock(keep->mutex);
    auto & d = keep->domains[domain];
    auto & c = d.categories[category];
    switch (state) {
        case phase::reserved: c.reserved -= bytes; break;
        case phase::allocated: c.allocated -= bytes; break;
        case phase::retiring: c.retiring -= bytes; break;
    }
    if (state != phase::reserved) {
        // Lifetime counters saturate rather than corrupting admission arithmetic.
        d.released += std::min(bytes, std::numeric_limits<size_t>::max() - d.released);
    }
    keep->apply(domain);
    bytes = 0;
}
