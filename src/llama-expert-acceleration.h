#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>

// Internal configuration: no public llama_model/context ABI change.
inline bool llama_expert_acceleration_requested() {
    const char * mode = std::getenv("LLAMA_MOE_STREAMING");
    if (!mode || std::strcmp(mode, "auto") == 0) { return true; }
    if (std::strcmp(mode, "off") == 0) { return false; }
    throw std::invalid_argument("LLAMA_MOE_STREAMING must be auto or off");
}
inline bool llama_expert_acceleration_off() {
    const char * mode = std::getenv("LLAMA_MOE_STREAMING");
    return mode && std::strcmp(mode, "off") == 0;
}
