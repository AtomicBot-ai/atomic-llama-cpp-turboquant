#pragma once

#include "ggml-backend.h"

// Private contract: CPU-prepared map in a bound output's existing extra space.
// Valid only for an exclusively owned single-expert 32-row op with immutable IDs.
enum { GGML_METAL_PREPARED_SINGLE_EXPERT = 0x41544531 };

// The tag lives past the ordinary custom callback and its application data.
enum { GGML_METAL_HOST_EXPERT_BRIDGE = 0x41544532 };
enum { GGML_METAL_INKLING_REL_MASK = 0x41544534 };
enum { GGML_METAL_INKLING_CONV_RESIDUAL = 0x41544535 };
enum { GGML_METAL_INKLING_CLIP = 0x41544536 };
enum { GGML_METAL_INKLING_SITU = 0x41544537 };
enum { GGML_METAL_INKLING_CONTROL = 0x41544538 };
static inline bool ggml_metal_is_host_expert(const struct ggml_tensor * node) {
    return node->op == GGML_OP_CUSTOM &&
        node->op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1] == GGML_METAL_HOST_EXPERT_BRIDGE;
}

// Internal proc-address implementations; callers must obey exclusive ownership.
void * ggml_backend_metal_buffer_get_host_base(ggml_backend_buffer_t buffer);
bool ggml_backend_metal_prepare_single_expert(struct ggml_tensor * op);
