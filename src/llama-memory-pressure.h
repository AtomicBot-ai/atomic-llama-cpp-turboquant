#pragma once

#include "llama-memory-policy.h"
#include <memory>

// A system signal source only. Never calls an executor, ledger, or context.
class llama_memory_pressure_monitor {
public:
    llama_memory_pressure_monitor();
    ~llama_memory_pressure_monitor();
    bool available() const;
    llama_memory_pressure_sample poll(uint64_t now);
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
