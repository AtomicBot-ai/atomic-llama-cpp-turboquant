#pragma once

#include "common.h"

// Fork-only discrete configuration. Ordinary loading retains common_model_load.
inline bool common_atomic_budget_enabled(const common_params & params) {
    return params.engine_budget || params.engine_ram_budget || params.engine_vram_budget;
}
void common_atomic_discrete_validate(const common_params & params);
llama_model * common_atomic_discrete_load(const common_params & params, llama_model_params mparams);
void common_atomic_select_qwen35_profile(common_params & params);
bool common_atomic_flash_next_dflash3_profile(const common_params & params);
bool common_atomic_dflash2_enabled(const common_params & params);
bool common_atomic_spark_dflash2_enabled(const common_params & params);
bool common_atomic_inkling_dspark_enabled(const common_params & params);
#ifdef GGML_USE_METAL
void common_atomic_select_inkling_metal_profile(common_params & params);
#endif
