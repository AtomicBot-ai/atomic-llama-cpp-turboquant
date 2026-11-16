#pragma once

#include "llama-atomic.h"
#include "llama-memory-pressure.h"

#include <memory>
#include <mutex>

// Control metadata is owned only by contexts with an explicit engine budget.
// Like other control objects, it is covered by the runtime envelope.
struct llama_context_budget {
    llama_prefill_window_data prefill_window;
    std::shared_ptr<void> budget_session;
    mutable std::mutex budget_mutex;
    size_t budget_requested = 0;
    uint64_t budget_request_id = 0, budget_completed_id = 0;
    llama_budget_status budget_status = LLAMA_BUDGET_IDLE;
    bool (*discrete_boundary_probe)(llama_context *, void *) noexcept = nullptr;
    void * discrete_probe_user = nullptr;
    llama_memory_policy budget_policy;
    std::unique_ptr<class llama_memory_pressure_monitor> pressure_monitor;
};
