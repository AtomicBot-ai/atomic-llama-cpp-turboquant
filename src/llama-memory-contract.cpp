#include "llama-memory-contract.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <limits>
#include <stdexcept>

llama_allocation_scope::llama_allocation_scope(std::shared_ptr<llama_memory_budget> budget, llama_memory_category category,
        std::function<void()> reclaim, llama_discrete_buffer_types discrete, llama_memory_reclaimer sized_reclaim) :
    budget(std::move(budget)), category(category), reclaim(std::move(reclaim)), discrete(discrete), sized_reclaim(sized_reclaim),
    previous(ggml_backend_set_alloc_observer(this->budget ?
        ggml_backend_alloc_observer{reserve, commit, release, this} : ggml_backend_alloc_observer{})) {}
llama_allocation_scope::~llama_allocation_scope() { ggml_backend_set_alloc_observer(previous); }

void * llama_allocation_scope::reserve(ggml_backend_buffer_type_t buft, size_t size, void * user) noexcept {
    try {
        auto & scope = *static_cast<llama_allocation_scope *>(user);
        const char * name = ggml_backend_buft_name(buft);
        auto domain = llama_memory_domain::host;
        bool supported;
        if (scope.discrete.device) {
            const bool host = buft == ggml_backend_cpu_buffer_type() || buft == scope.discrete.pinned_host;
            if (!host && buft == scope.discrete.device) {
                domain = llama_memory_domain::device;
            }
            supported = host || domain == llama_memory_domain::device;
        } else {
            supported = !std::strcmp(name, "CPU") || !std::strcmp(name, "MTL0") || !std::strcmp(name, "Metal");
        }
        if (!supported) {
            std::fprintf(stderr, "engine budget: buffer type %s is outside the qualified ownership contract\n", name);
            return nullptr;
        }
        constexpr size_t granularity = 65536;
        if (size > std::numeric_limits<size_t>::max() - granularity) { return nullptr; }
        const size_t charge = (size + granularity - 1) / granularity * granularity;
        auto ticket = scope.budget->reserve(domain, scope.category, charge);
        if (!ticket && (scope.sized_reclaim.reclaim || scope.reclaim)) {
            if (scope.sized_reclaim.reclaim) { scope.sized_reclaim.reclaim(scope.sized_reclaim.user, domain, charge); }
            else { scope.reclaim(); }
            ticket = scope.budget->reserve(domain, scope.category, charge);
        }
        if (!ticket) {
            if (scope.discrete.unified) { scope.budget->spark_admission_failed = true; }
            const auto state = scope.budget->snapshot(domain);
            std::fprintf(stderr, "engine budget: denied %zu bytes for %s; used=%zu target=%zu category=%d\n",
                charge, name, state.used(), state.requested, int(scope.category));
            return nullptr;
        }
        return new llama_memory_budget::ticket(std::move(ticket));
    } catch (...) { return nullptr; }
}
bool llama_allocation_scope::commit(void * reservation, size_t actual) noexcept {
    auto & ticket = *static_cast<llama_memory_budget::ticket *>(reservation);
    if (actual > ticket.size()) { return false; }
    ticket.commit();
    return true;
}
void llama_allocation_scope::release(void * reservation) noexcept {
    delete static_cast<llama_memory_budget::ticket *>(reservation);
}

llama_discrete_load_staging::llama_discrete_load_staging(
        std::shared_ptr<llama_memory_budget> budget, llama_discrete_buffer_types types) :
    budget(std::move(budget)), types(types) {
    if (!this->budget || !types.device) {
        throw std::invalid_argument("discrete loader staging requires a ledger and a device buffer type");
    }
}

llama_discrete_load_staging::~llama_discrete_load_staging() {
    ggml_backend_buffer_free(buffer);
}

void llama_discrete_load_staging::upload(ggml_tensor * tensor, const std::function<void(void *, size_t)> & read) {
    if (!tensor || !tensor->buffer || ggml_backend_buffer_get_type(tensor->buffer) != types.device) {
        throw std::invalid_argument("discrete loader staging requires the selected device buffer type");
    }
    if (!buffer) {
        llama_allocation_scope admission(budget, llama_memory_category::io, {}, types);
        buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), chunk_bytes);
        if (!buffer) {
            throw std::runtime_error("discrete loader staging exceeds the RAM budget");
        }
    }
    auto * data = ggml_backend_buffer_get_base(buffer);
    const size_t bytes = ggml_nbytes(tensor);
    for (size_t offset = 0; offset < bytes;) {
        const size_t chunk = std::min(chunk_bytes, bytes - offset);
        read(data, chunk);
        ggml_backend_tensor_set(tensor, data, offset, chunk);
        offset += chunk;
    }
}

llama_memory_contract::llama_memory_contract(size_t cap) : llama_memory_contract(cap, 0, {}) {}
llama_memory_contract::llama_memory_contract(size_t host_cap, size_t device_cap, llama_discrete_buffer_types types) :
    ledger(std::make_shared<llama_memory_budget>(host_cap, device_cap, !types.device)), discrete(types) {
    runtime = ledger->reserve(llama_memory_domain::host, llama_memory_category::runtime, runtime_envelope);
    if (!runtime) { throw std::runtime_error("engine budget is below the 2 GiB runtime envelope, before model weights and state"); }
    // Keep this as a reservation: it is an envelope, not measured payload.
}
std::shared_ptr<void> llama_memory_contract::claim_context() {
    if (context_claimed.exchange(true)) { throw std::runtime_error("engine memory budget currently permits one active context per model"); }
    auto self = shared_from_this();
    return std::shared_ptr<void>(this, [self](void *) { self->context_claimed.store(false); });
}
