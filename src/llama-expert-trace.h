#pragma once

struct ggml_tensor;

#include <cstddef>
#include <cstdint>

enum class llama_expert_trace_kind { projection, access, reset, configuration, release };

// Diagnostic observations of the synchronous UMA executor. capacity includes
// cache and miss staging, after subtracting every other ledger owner. Callback
// storage belongs to the caller; callbacks must not reenter the engine.
struct llama_expert_trace_event {
    llama_expert_trace_kind kind;
    size_t tensor = 0;
    int64_t expert = 0;
    int64_t tokens = 0;
    size_t payload = 0;
    size_t allocation = 0; // physical buffer reservation plus tensor arena
    size_t capacity = 0;
    bool hit = false;
    size_t required = 0; // minimum physical reservation; allocation may be reused and larger
};
using llama_expert_trace_callback = void (*)(const llama_expert_trace_event &, void *) noexcept;

// Read-only callback at the existing CPU expert bridge. Pointers are valid only
// during the call. No graph nodes, tensor consumers or router choices are added.
// Do not reenter the engine or retain tensor pointers. An exception rejects the
// projection through the executor's existing sticky failure path.
struct llama_expert_route_event {
    const char * weight;
    const ggml_tensor * input;
    const ggml_tensor * ids;
    uint64_t projection;
};
using llama_expert_route_callback = void (*)(const llama_expert_route_event &, void *);
