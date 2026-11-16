#include "atomic-budget.h"
#include "log.h"
#include "../src/llama-atomic.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

void common_atomic_discrete_validate(const common_params & p) {
    if (!p.engine_ram_budget && !p.engine_vram_budget) { return; }
    if (!p.engine_ram_budget || !p.engine_vram_budget || p.engine_budget || p.engine_memory_auto) {
        throw std::invalid_argument("CUDA streaming requires both --engine-ram-mib and --engine-vram-mib, without UMA/automatic budgets");
    }
    auto require = [](bool condition, const char * message) {
        if (!condition) { throw std::invalid_argument(message); }
    };
    require(!p.fit_params, "CUDA streaming requires -fit off");
    require(!p.speculative.has_dft() && std::none_of(p.speculative.types.begin(), p.speculative.types.end(),
        [](common_speculative_type type) { return type != COMMON_SPECULATIVE_TYPE_NONE; }),
        "CUDA streaming does not support speculative decoding");
    require(p.mmproj.path.empty() && p.lora_adapters.empty() && p.control_vectors.empty() && !p.embedding && !p.load_mtp,
        "CUDA streaming requires text generation without projectors, adapters or MTP");
    require(!p.moe_cache_size, "CUDA streaming cannot use the resident host MoE cache");
    require(!p.no_op_offload, "CUDA streaming requires op offload to preserve boundary reduction");
    require(p.n_parallel == 1, "CUDA streaming requires one sequence");
    require(p.n_ctx > 0 && p.n_ctx <= 4096 && p.n_batch <= 4096,
        "CUDA streaming requires explicit context and batch sizes no greater than 4096");
    require(p.n_ubatch > 0 && p.n_ubatch <= 32, "CUDA streaming requires an explicit microbatch no greater than 32");
    require(p.n_gpu_layers >= 2 && p.split_mode == LLAMA_SPLIT_MODE_LAYER &&
        std::none_of(p.tensor_buft_overrides.begin(), p.tensor_buft_overrides.end(),
            [](const llama_model_tensor_buft_override & item) { return item.pattern != nullptr; }),
        "CUDA streaming requires partial layer offload without tensor overrides");
    require(!p.check_tensors && !p.no_alloc, "CUDA streaming requires allocation without full tensor validation");
    if (p.devices.size() != 2 || !p.devices[0] || p.devices[1] ||
        ggml_backend_dev_type(p.devices[0]) != GGML_BACKEND_DEVICE_TYPE_GPU ||
        std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(p.devices[0])), "CUDA")) {
        throw std::invalid_argument("CUDA streaming requires one explicit --device CUDA0 (or another single CUDA device)");
    }
}

llama_model * common_atomic_discrete_load(const common_params & p, llama_model_params mparams) try {
    const auto paths = p.model.get_paths();
    if (paths.size() != 1) {
        throw std::invalid_argument("CUDA streaming requires a local first GGUF shard path");
    }
    // The argument parser appends an empty terminator even without overrides.
    mparams.tensor_buft_overrides = nullptr;
    bool boundary_reduction = false;
    {
        // Read metadata only and release it before admitting model allocations.
        // The LFM boundary arithmetic is not the Qwen routed arithmetic.
        gguf_context_ptr metadata(gguf_init_from_file(paths.front().c_str(), {true, nullptr}));
        if (!metadata) { throw std::invalid_argument("cannot read CUDA streaming model metadata"); }
        const auto key = gguf_find_key(metadata.get(), "general.architecture");
        if (key < 0 || gguf_get_kv_type(metadata.get(), key) != GGUF_TYPE_STRING) {
            throw std::invalid_argument("CUDA streaming requires a model architecture");
        }
        const std::string arch = gguf_get_val_str(metadata.get(), key);
        if (arch == "lfm2moe") {
            boundary_reduction = true;
        } else if (arch == "qwen4exp") {
            if (p.n_ubatch > 8) { throw std::invalid_argument("Qwen CUDA streaming requires microbatch <= 8"); }
        } else {
            throw std::invalid_argument("CUDA streaming server profile requires LFM2 MoE or Qwen4Exp");
        }
    }
    return llama_load_discrete_model(paths.front().c_str(), mparams, p.devices.front(),
        p.engine_ram_budget, p.engine_vram_budget, true, true, boundary_reduction);
} catch (const std::exception & error) {
    LOG_ERR("CUDA streaming model load: %s\n", error.what());
    return nullptr;
}
