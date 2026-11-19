#pragma once

#include "llama-memory-budget.h"
#include "../ggml/src/ggml-alloc-observer.h"

#include <atomic>
#include <memory>
#include <functional>

// Exact buffer types from one qualified discrete backend. Other types fail admission.
struct llama_discrete_buffer_types {
    ggml_backend_buffer_type_t device = nullptr;
    ggml_backend_buffer_type_t pinned_host = nullptr;
};

// Reusable, synchronous loader staging for an explicitly selected discrete
// device. At most one chunk is live, even for a large GGUF tensor.
class llama_discrete_load_staging {
public:
    static constexpr size_t chunk_bytes = 1024 * 1024;
    llama_discrete_load_staging(std::shared_ptr<llama_memory_budget> budget, llama_discrete_buffer_types types);
    ~llama_discrete_load_staging();
    llama_discrete_load_staging(const llama_discrete_load_staging &) = delete;
    llama_discrete_load_staging & operator=(const llama_discrete_load_staging &) = delete;

    void upload(ggml_tensor * tensor, const std::function<void(void *, size_t)> & read);

private:
    std::shared_ptr<llama_memory_budget> budget;
    llama_discrete_buffer_types types;
    ggml_backend_buffer_t buffer = nullptr;
};

// Optional size-aware reclamation receives the full rounded reservation, not
// a stale shortfall. Its owner outlives the scope. A refusal must not fall through
// to destructive legacy reclamation; admission is retried against the ledger.
struct llama_memory_reclaimer {
    void (*reclaim)(void *, llama_memory_domain, size_t) = nullptr;
    void * user = nullptr;
};

// Installs admission only on this thread. Resource tickets outlive the scope.
// An empty scope suppresses interception for already-admitted expert buffers.
class llama_allocation_scope {
public:
    llama_allocation_scope(std::shared_ptr<llama_memory_budget> budget = {},
                           llama_memory_category category = llama_memory_category::runtime,
                           std::function<void()> reclaim = {},
                           llama_discrete_buffer_types discrete = {},
                           llama_memory_reclaimer sized_reclaim = {});
    ~llama_allocation_scope();
    llama_allocation_scope(const llama_allocation_scope &) = delete;
    llama_allocation_scope & operator=(const llama_allocation_scope &) = delete;
private:
    std::shared_ptr<llama_memory_budget> budget;
    llama_memory_category category;
    std::function<void()> reclaim;
    llama_discrete_buffer_types discrete;
    llama_memory_reclaimer sized_reclaim;
    ggml_backend_alloc_observer previous;
    static void * reserve(ggml_backend_buffer_type_t buft, size_t size, void * user) noexcept;
    static bool commit(void * reservation, size_t actual) noexcept;
    static void release(void * reservation) noexcept;
};

// One model + one context. Backend payloads are admitted individually, with a
// fixed host envelope for metadata, graph/control objects and runtime overhead.
struct llama_memory_contract : std::enable_shared_from_this<llama_memory_contract> {
    static constexpr size_t runtime_envelope = size_t(2) * 1024 * 1024 * 1024;
    explicit llama_memory_contract(size_t cap);
    // A dense auxiliary model uses the same ledger and runtime envelope, with
    // an independent context claim and no streamed-weight policy.
    explicit llama_memory_contract(std::shared_ptr<llama_memory_budget> shared) : ledger(std::move(shared)) {}
    llama_memory_contract(size_t host_cap, size_t device_cap, llama_discrete_buffer_types types);
    std::shared_ptr<void> claim_context();
    std::shared_ptr<llama_memory_budget> ledger;
    llama_discrete_buffer_types discrete;
    ggml_backend_dev_t device_list[2] = {};
    bool stream_experts = false;
    bool stream_cpu_layers_on_cpu = false;
    bool stream_cuda_boundary_reduction = false;
    bool dflash2_target = false;
    uint32_t qwen35_profile_layers = 0;
    bool qwen35_fast_prefill_default = false;
    bool qwen35_resident_verify = false;
    bool qwen35_grouped_attention = false;
    llama_memory_budget::ticket runtime;
    llama_memory_budget::ticket draft_slot_staging;
    std::atomic<bool> context_claimed {false};
};

struct llama_model;
struct llama_file;
class llama_io_read_i;
class llama_io_write_i;
// Optional file-state staging for discrete Qwen; null preserves the ordinary IO path.
std::unique_ptr<llama_io_read_i> llama_atomic_state_file_reader(const llama_model & model, llama_file * file);
std::unique_ptr<llama_io_write_i> llama_atomic_state_file_writer(const llama_model & model, llama_file * file);
