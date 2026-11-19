#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

// Queried at startup, after the fixed-budget Qwen profile has been qualified.
inline bool llama_qwen35_fast_prefill_default(size_t budget) {
    constexpr size_t gib = size_t(1) << 30;
    const char * mode = std::getenv("LLAMA_QWEN35_FAST_PREFILL");
    return budget >= 8*gib && budget <= 24*gib && (!mode || std::strcmp(mode, "1") == 0);
}
