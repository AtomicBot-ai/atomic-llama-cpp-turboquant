#pragma once
#if defined(GGML_USE_HIP)
#include "vendors/hip.h"
#else
#include <cuda_fp16.h>
#endif
#include <cstdint>

// Attention-sink token protection: store rotated fp16 values for the first N
// KV cache positions. These are written during SET_ROWS (before quantization
// loss) and used to patch the dequanted fp16 buffer before flash attention.
// Enable with: GGML_TURBO_SINK_TOKENS=N (e.g. 4, 8, 16)

int turbo_sink_n_tokens();
half * turbo_sink_get_or_alloc(const void * tensor_data, int64_t ne0, int n_sink);
half * turbo_sink_get(const void * tensor_data);
