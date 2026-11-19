#pragma once

#include "ggml-backend.h"
#include <stdint.h>

// Fork-private v1 contract. No public ggml ABI or caller access to Metal objects.
// A channel has one outstanding publication. The graph and channel must outlive
// main-backend completion; the consumer must drain before graph/channel reuse.
#define GGML_METAL_ROUTE_MARKER 0x41545250
#define GGML_METAL_ROUTE_MAX_IDS 64
struct ggml_metal_route_api {
    size_t reserved_bytes;
    void * (*create)(ggml_backend_t);
    void (*destroy)(void *);
    bool (*attach)(struct ggml_tensor *, void *, void (*)(void *, uint64_t, uint64_t, size_t), void *, uint64_t);
    int (*read)(void *, uint64_t, int32_t *, size_t); // 1 ready, 0 pending (bounded wait), -1 stale/invalid
};

#ifdef __cplusplus
extern "C" {
#endif
void * ggml_metal_route_create(void * device);
void ggml_metal_route_destroy(void * channel);
int ggml_metal_route_read(void * channel, uint64_t sequence, int32_t * ids, size_t count);
#ifdef __cplusplus
}
#endif
