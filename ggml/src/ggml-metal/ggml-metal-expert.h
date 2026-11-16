#pragma once

#include "ggml-backend.h"

// Private contract: CPU-prepared map in a bound output's existing extra space.
// Valid only for an exclusively owned single-expert 32-row op with immutable IDs.
enum { GGML_METAL_PREPARED_SINGLE_EXPERT = 0x41544531 };

// Internal proc-address implementations; callers must obey exclusive ownership.
void * ggml_backend_metal_buffer_get_host_base(ggml_backend_buffer_t buffer);
bool ggml_backend_metal_prepare_single_expert(ggml_tensor * op);
