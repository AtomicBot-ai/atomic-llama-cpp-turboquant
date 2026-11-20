#include "atomic-budget.h"
#include "log.h"
#include "../src/llama-atomic.h"
#include "../src/llama-qwen35-metal-profile.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
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
namespace {
bool metadata_string(const gguf_context * meta, const char * name, const char * expected) {
    const auto key = gguf_find_key(meta, name);
    return key >= 0 && gguf_get_kv_type(meta, key) == GGUF_TYPE_STRING &&
            !std::strcmp(gguf_get_val_str(meta, key), expected);
}

bool metadata_u32(const gguf_context * meta, const char * name, uint32_t expected) {
    const auto key = gguf_find_key(meta, name);
    return key >= 0 && gguf_get_kv_type(meta, key) == GGUF_TYPE_UINT32 && gguf_get_val_u32(meta, key) == expected;
}

// Qwen3.5 and Qwen3.6 can share the same GGUF architecture and MoE geometry.
bool qwen36_name(const gguf_context * meta, const char * field) {
    const auto key = gguf_find_key(meta, field);
    if (key < 0 || gguf_get_kv_type(meta, key) != GGUF_TYPE_STRING) { return false; }
    std::string name(gguf_get_val_str(meta, key));
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return std::tolower(c); });
    return name.find("qwen3.6") != std::string::npos;
}

bool qwen35_grid_target(const std::string & path) {
    gguf_context_ptr meta(gguf_init_from_file(path.c_str(), {true, nullptr}));
    return meta && metadata_string(meta.get(), "general.architecture", "qwen35moe") &&
            (qwen36_name(meta.get(), "general.name") || qwen36_name(meta.get(), "general.basename")) &&
            metadata_u32(meta.get(), "general.file_type", LLAMA_FTYPE_MOSTLY_Q4_K_M) &&
            metadata_u32(meta.get(), "qwen35moe.block_count", 40) && metadata_u32(meta.get(), "qwen35moe.embedding_length", 2048) &&
            metadata_u32(meta.get(), "qwen35moe.expert_count", 256) && metadata_u32(meta.get(), "qwen35moe.expert_used_count", 8) &&
            metadata_u32(meta.get(), "qwen35moe.expert_feed_forward_length", 512);
}

// Zero declines a layout. Compact draft profiles keep both selector codebooks in BF16.
int qwen35_grid_draft_layers(const std::string & path, int bf16_layers, int compact_layers) {
    gguf_context_ptr meta(gguf_init_from_file(path.c_str(), {true, nullptr}));
    if (!meta || !metadata_string(meta.get(), "general.architecture", "dflash") ||
            !metadata_u32(meta.get(), "dflash.embedding_length", 2048) || !metadata_u32(meta.get(), "dflash.block_count", 6) ||
            !metadata_u32(meta.get(), "dflash.selector_rank", 256) || !metadata_u32(meta.get(), "dflash.selector_top_k", 16)) {
        return 0;
    }
    const auto vocab = gguf_find_key(meta.get(), "tokenizer.ggml.tokens");
    const auto layers = gguf_find_key(meta.get(), "dflash.target_layers");
    constexpr int32_t expected[] = {2, 7, 12, 17, 23, 28, 33, 38};
    const bool compatible = vocab >= 0 && gguf_get_kv_type(meta.get(), vocab) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_n(meta.get(), vocab) == 248320 &&
            layers >= 0 && gguf_get_kv_type(meta.get(), layers) == GGUF_TYPE_ARRAY &&
            gguf_get_arr_type(meta.get(), layers) == GGUF_TYPE_INT32 && gguf_get_arr_n(meta.get(), layers) == 8 &&
            !std::memcmp(gguf_get_arr_data(meta.get(), layers), expected, sizeof(expected));
    if (!compatible) { return 0; }
    if (metadata_u32(meta.get(), "general.file_type", LLAMA_FTYPE_MOSTLY_BF16)) { return bf16_layers; }
    if (!compact_layers || !metadata_u32(meta.get(), "general.file_type", LLAMA_FTYPE_MOSTLY_Q4_0)) { return 0; }
    for (const char * name : {"selector_predecessor.weight", "selector_successor.weight"}) {
        const auto id = gguf_find_tensor(meta.get(), name);
        if (id < 0 || gguf_get_tensor_type(meta.get(), id) != GGML_TYPE_BF16) { return 0; }
    }
    for (int64_t id = 0; id < gguf_get_n_tensors(meta.get()); ++id) {
        const std::string name = gguf_get_tensor_name(meta.get(), id);
        if (name == "selector_predecessor.weight" || name == "selector_successor.weight") { continue; }
        const bool fp32 = name.find("norm.weight") != std::string::npos || name.find("conv_base") != std::string::npos;
        if (gguf_get_tensor_type(meta.get(), id) != (fp32 ? GGML_TYPE_F32 : GGML_TYPE_Q4_0)) { return 0; }
    }
    return compact_layers;
}
} // namespace

void common_atomic_select_qwen35_profile(common_params & p) {
    constexpr size_t gib = size_t(1) << 30;
    const bool resident_26 = p.engine_budget == 26*gib;
    if (p.engine_budget < 8*gib || (p.engine_budget > 24*gib && !resident_26) || p.engine_memory_auto || p.fit_params ||
            p.engine_ram_budget || p.engine_vram_budget || p.engine_prefill_window || p.engine_prefill_window_disabled ||
            p.n_ctx != 4096 || p.n_batch != 4096 || p.n_ubatch != 8 || p.n_parallel != 1 ||
            p.cache_type_k != GGML_TYPE_BF16 || p.cache_type_v != GGML_TYPE_BF16 ||
            (p.n_gpu_layers >= 0 && p.n_gpu_layers < 41) || p.no_op_offload || p.no_kv_offload ||
            !p.lora_adapters.empty() || !p.control_vectors.empty() || !p.mmproj.path.empty() || p.embedding ||
            p.moe_cache_size || p.load_mtp || p.speculative.has_synth() ||
            std::any_of(p.tensor_buft_overrides.begin(), p.tensor_buft_overrides.end(),
                [](const auto & item) { return item.pattern != nullptr; })) { return; }
    const char * profile_mode = std::getenv("LLAMA_QWEN35_PROFILE");
    if (profile_mode && !std::strcmp(profile_mode, "0")) { return; }
    if (resident_26) {
        const char * mode = std::getenv("LLAMA_QWEN35_RESIDENT_VERIFY");
        if (!mode || (std::strcmp(mode, "1") && std::strcmp(mode, "exact")) ||
                (p.speculative.draft.n_max != 6 && (p.speculative.draft.n_max_explicit || p.speculative.draft.n_max != 3)) ||
                (p.speculative.draft.n_ubatch && p.speculative.draft.n_ubatch != 128) ||
                p.cpuparams.n_threads != 8 || p.cpuparams_batch.n_threads != 8) { return; }
    }
    const auto device = p.devices.empty() ? ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU) : p.devices.front();
    if (!device || p.devices.size() > 2 || std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "MTL")) { return; }
    auto & spec = p.speculative;
    if (std::any_of(spec.types.begin(), spec.types.end(), [](auto type) {
            return type != COMMON_SPECULATIVE_TYPE_NONE && type != COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
        })) { return; }
    const auto paths = p.model.get_paths();
    if (paths.size() != 1 || !qwen35_grid_target(paths.front())) { return; }

    // Fixed-start 4000/64 Metal sweep, in 2 GiB increments. The ledger still enforces the actual cap.
    struct layer_profile { int serial, draft; };
    static constexpr layer_profile profiles[] = {{2,0}, {5,4}, {8,9}, {18,13}, {22,18}, {22,22}, {31,26}, {31,31}, {40,36}};
    const auto profile = resident_26 ? layer_profile{0,40} : profiles[(p.engine_budget - 8*gib) / (2*gib)];
    const char * legacy = std::getenv("LLAMA_EXPERIMENTAL_BUDGET_DFLASH2");
    int draft_layers = 0;
    const bool fast_default = llama_qwen35_fast_prefill_default(p.engine_budget);
    const bool compact_enabled = legacy ? !std::strcmp(legacy, "1") : fast_default;
    if (!profile.draft && compact_enabled && p.engine_budget == 8*gib && spec.has_dft()) {
        // A second bank forces the long prefill to read the experts twice.
        std::error_code ec;
        const auto & path = spec.draft.mparams.path;
        if (std::filesystem::is_regular_file(path, ec)) {
            draft_layers = qwen35_grid_draft_layers(path, 0, 1);
        }
        if (!draft_layers && legacy) { return; } // Preserve the explicitly requested BF16 experiment.
    }
    // The parser appends explicit --spec-type selections to its initial NONE entry.
    const bool explicit_none = std::count(spec.types.begin(), spec.types.end(), COMMON_SPECULATIVE_TYPE_NONE) > 1;
    const bool explicit_dflash = std::find(spec.types.begin(), spec.types.end(),
            COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) != spec.types.end();
    const bool supplied = spec.has_dft();
    p.engine_qwen35_layers = profile.serial;
    if ((!profile.draft && !draft_layers) || explicit_none || (legacy && !std::strcmp(legacy, "0"))) {
        spec.draft.mparams = {};
        spec.types = {COMMON_SPECULATIVE_TYPE_NONE};
        return;
    }

    const auto candidate = supplied ? std::filesystem::path(spec.draft.mparams.path) :
            std::filesystem::path(paths.front()).parent_path() / "qwen35-dflash2.gguf";
    std::error_code ec;
    const bool present = std::filesystem::is_regular_file(candidate, ec);
    if (!draft_layers && present) {
        draft_layers = qwen35_grid_draft_layers(candidate.string(), profile.draft,
                supplied && p.engine_budget == 24*gib ? 37 : 0);
    }
    if (!draft_layers) {
        if (supplied) { throw std::invalid_argument("Qwen Metal profile requires a compatible local DFlash2 draft"); }
        if (present) {
            LOG_WRN("Qwen Metal profile: ignoring incompatible sidecar '%s'\n", candidate.string().c_str());
        }
        if (explicit_dflash) {
            throw std::invalid_argument("requested Qwen DFlash2 draft is unavailable");
        }
        return;
    }

    spec.draft.mparams.path = candidate.string();
    spec.types = {COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH};
    if (!spec.draft.n_max_explicit && spec.draft.n_max == 3) { spec.draft.n_max = 6; }
    if (!spec.draft.n_ubatch) { spec.draft.n_ubatch = 128; }
    p.engine_qwen35_layers = draft_layers;
    if (draft_layers == 36 && p.engine_budget == 24*gib && spec.draft.n_max == 6 &&
            spec.draft.n_ubatch == 128 && p.cpuparams.n_threads == 8 && p.cpuparams_batch.n_threads == 8) {
        const char * mode = std::getenv("LLAMA_QWEN35_RESIDENT_VERIFY");
        if (mode ? (!std::strcmp(mode, "1") || !std::strcmp(mode, "exact")) : fast_default) {
            p.engine_qwen35_layers = 40;
        }
    }
}

bool common_atomic_flash_next_dflash3_profile(const common_params & params) {
    return params.engine_budget >= (size_t(8) << 30) && params.engine_budget <= (size_t(24) << 30) &&
            (params.n_ctx == 4096 || params.n_ctx == 40960) && params.n_batch == 4096 && params.n_ubatch == 8 && params.n_parallel == 1 &&
            params.cache_type_k == GGML_TYPE_BF16 && params.cache_type_v == GGML_TYPE_BF16 &&
            params.speculative.draft.n_max == 3;
}

bool common_atomic_dflash2_enabled(const common_params & params) {
    const bool short_profile = params.engine_budget == (size_t(18) << 30) && params.n_ctx == 1024;
    const bool long_profile = (params.engine_qwen35_layers || params.engine_budget == (size_t(8) << 30)) && params.n_ctx == 4096 && params.n_batch == 4096;
    const auto draft_ubatch = params.speculative.draft.n_ubatch;
    const bool draft_ubatch_supported = !draft_ubatch ||
        (short_profile ? draft_ubatch == params.n_ubatch : draft_ubatch >= params.n_ubatch && draft_ubatch <= 256);
    if ((!short_profile && !long_profile && !common_atomic_flash_next_dflash3_profile(params)) || params.engine_ram_budget || params.engine_vram_budget ||
            params.engine_memory_auto || params.engine_prefill_window ||
            params.n_ubatch != 8 || params.n_parallel != 1 || !params.speculative.has_dft() ||
            params.speculative.draft.n_max < 1 ||
            params.speculative.draft.n_max >= int32_t(params.n_ubatch) - 1 ||
            !draft_ubatch_supported ||
            params.speculative.has_synth()) {
        return false;
    }
    const auto & types = params.speculative.types;
    if (std::count(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH) != 1 ||
            std::any_of(types.begin(), types.end(), [](common_speculative_type type) {
                return type != COMMON_SPECULATIVE_TYPE_NONE && type != COMMON_SPECULATIVE_TYPE_DRAFT_DFLASH;
            })) {
        return false;
    }
    const char * enabled = std::getenv("LLAMA_EXPERIMENTAL_BUDGET_DFLASH2");
    return params.engine_qwen35_layers || (enabled && std::strcmp(enabled, "1") == 0);
}
