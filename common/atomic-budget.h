#pragma once

#include "common.h"

// Fork-only discrete configuration. Ordinary loading retains common_model_load.
inline bool common_atomic_budget_enabled(const common_params & params) {
    return params.engine_budget || params.engine_ram_budget || params.engine_vram_budget;
}
void common_atomic_discrete_validate(const common_params & params);
llama_model * common_atomic_discrete_load(const common_params & params, llama_model_params mparams);
