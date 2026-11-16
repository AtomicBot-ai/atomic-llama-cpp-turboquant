#pragma once

// Fork-owned allocation admission extension. Buffer ownership stays in ggml-backend.cpp.
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

// Optional thread-local admission for physical buffer allocations. Callbacks
// must not throw. A successful reservation travels with its buffer and is
// released only after the backend frees that buffer, even on another thread.
struct ggml_backend_alloc_observer {
    void * (*reserve)(ggml_backend_buffer_type_t buft, size_t bytes, void * user_data);
    bool (*commit)(void * reservation, size_t actual_bytes);
    void (*release)(void * reservation);
    void * user_data;
};
GGML_API struct ggml_backend_alloc_observer ggml_backend_set_alloc_observer(struct ggml_backend_alloc_observer observer);
GGML_API bool ggml_backend_has_alloc_observer(void);

#ifdef __cplusplus
}
#endif
