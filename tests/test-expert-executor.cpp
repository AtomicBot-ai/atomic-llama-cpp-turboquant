#include "testing.h"
#include "test-expert-publication.h"
#include "test-expert-prefetch.h"
#include "../src/llama-expert-executor.h"
#include "../src/llama-memory-contract.h"
#include "../src/llama-expert-route-index.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

struct fixture {
    fs::path path;
    ggml_context_ptr weights;
    ggml_tensor * weight;
    std::shared_ptr<llama_expert_source> source;

    static float value(int e, int r, int c) { return (e + 1) * 0.125f + (r - c) * 0.0625f; }
    explicit fixture(int experts = 4, ggml_type type = GGML_TYPE_F32, const char * architecture = "qwen3moe") :
        weights(ggml_init({16384 + size_t(experts) * 1024, nullptr, false})) {
        path = fs::temp_directory_path() / ("test-expert-executor-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
        weight = ggml_new_tensor_3d(weights.get(), type, 64, 4, experts);
        ggml_set_name(weight, "blk.0.ffn_up_exps.weight");
        std::vector<float> data(256 * experts);
        for (int e = 0; e < experts; ++e) {
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 64; ++c) { data[e * 256 + r * 64 + c] = value(e, r, c); }
            }
        }
        if (type == GGML_TYPE_F32) {
            std::memcpy(weight->data, data.data(), ggml_nbytes(weight));
        } else if (ggml_quantize_chunk(type, data.data(), weight->data, 0, 4 * experts, 64, nullptr) != ggml_nbytes(weight)) {
            throw std::runtime_error("fixture quantization failed");
        }
        gguf_context_ptr meta(gguf_init_empty());
        gguf_set_val_str(meta.get(), "general.architecture", architecture);
        gguf_add_tensor(meta.get(), weight);
        if (!gguf_write_to_file(meta.get(), path.string().c_str(), false)) { throw std::runtime_error("fixture write failed"); }
        source = std::make_shared<llama_expert_source>(std::vector<std::string>{path.string()});
    }
    ~fixture() { std::error_code ec; fs::remove(path, ec); }
};

struct request {
    ggml_context_ptr ctx {ggml_init({65536, nullptr, true})};
    ggml_tensor * input;
    ggml_tensor * ids;
    ggml_tensor * output;
    ggml_cgraph * graph;
    ggml_backend_buffer_ptr buffer;
    static constexpr int32_t routes[] = {3, 0, 2, 3, 1, 2};
    static float activation(int token, int slot, int col) { return (1 + token + slot + col % 3) * 0.125f; }

    request(llama_expert_executor & exec, const fixture & f, ggml_backend_t cpu, bool per_slot, int tokens = 3,
            const std::vector<const ggml_tensor *> & future = {}, bool narrow = false, bool strided_ids = false) {
        // Transpose makes the per-slot input non-contiguous along the dot axis.
        const int slots = per_slot ? 2 : 1;
        auto * storage = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, slots, 64, tokens);
        input = ggml_permute(ctx.get(), storage, 1, 0, 2, 3);
        auto * id_storage = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, strided_ids ? tokens : 2, strided_ids ? 2 : tokens);
        ids = strided_ids ? ggml_transpose(ctx.get(), id_storage) : id_storage;
        output = exec.build(ctx.get(), f.weight, input, ids);
        graph = ggml_new_graph_custom(ctx.get(), future.empty() ? 8 : 32, false);
        ggml_build_forward_expand(graph, output);
        auto * next_input = narrow ? ggml_view_3d(ctx.get(), input, 64, slots, 1, input->nb[1], input->nb[2], 0) : input;
        auto * next_ids = narrow ? ggml_view_2d(ctx.get(), ids, 2, 1, ids->nb[1], 0) : ids;
        for (const auto * weight : future) {
            ggml_build_forward_expand(graph, exec.build(ctx.get(), weight, next_input, next_ids));
        }
        buffer.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), cpu));
        if (!buffer) { throw std::runtime_error("request allocation failed"); }
        std::vector<float> data(64 * slots * tokens);
        for (int token = 0; token < tokens; ++token) {
            for (int c = 0; c < 64; ++c) {
                for (int s = 0; s < slots; ++s) { data[(token * 64 + c) * slots + s] = activation(token, s, c); }
            }
        }
        ggml_backend_tensor_set(storage, data.data(), 0, data.size() * sizeof(float));
        std::vector<int32_t> selected(2 * tokens);
        for (size_t i = 0; i < selected.size(); ++i) { selected[strided_ids ? (i % 2) * tokens + i / 2 : i] = routes[i % 6]; }
        ggml_backend_tensor_set(id_storage, selected.data(), 0, selected.size() * sizeof(int32_t));
    }
};
constexpr int32_t request::routes[];

int main(int argc, char ** argv) {
    testing t;
    const char * backend_name = "CPU";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--backend" && i + 1 < argc) { backend_name = argv[++i]; }
        else { t.set_filter(argv[i]); }
    }
    ggml_backend_load_all();
    auto device = ggml_backend_dev_by_name(backend_name);
    ggml_backend_ptr cpu(ggml_backend_init_by_name("CPU", nullptr));
    if (!device || !cpu) { return 1; }
    const size_t cap = 384 * 1024;

#if defined(__APPLE__)
    if (std::string(backend_name) == "MTL0") {
        test_expert_publication(t, device);
        test_expert_publication_service(t);
        test_expert_publication_pipeline(t, device);
        t.test("portable prefetch priority four candidates failure cleanup", [&](testing & t) { test_portable_prefetch(t,4); });
        t.test("portable prefetch priority eight candidates failure cleanup", [&](testing & t) { test_portable_prefetch(t,8); });
        auto combined_test = [&](testing & t, bool pipeline, bool compact = false, bool recycle = false, const char * architecture = nullptr, bool quantized = false) {
            struct env_guard {
                std::string old;
                bool present;
                env_guard(bool automatic) : present(std::getenv("LLAMA_EXPERIMENTAL_METAL_MOE") != nullptr) {
                    if (present) { old = std::getenv("LLAMA_EXPERIMENTAL_METAL_MOE"); }
                    if (automatic) { unsetenv("LLAMA_EXPERIMENTAL_METAL_MOE"); }
                    else { setenv("LLAMA_EXPERIMENTAL_METAL_MOE", "1", 1); }
                }
                ~env_guard() {
                    if (present) { setenv("LLAMA_EXPERIMENTAL_METAL_MOE", old.c_str(), 1); }
                    else { unsetenv("LLAMA_EXPERIMENTAL_METAL_MOE"); }
                }
            } env(architecture != nullptr);
            struct pipeline_env_guard {
                std::string old;
                bool present;
                pipeline_env_guard(bool enabled) : present(std::getenv("LLAMA_EXPERIMENTAL_METAL_MOE_PIPELINE") != nullptr) {
                    if (present) { old = std::getenv("LLAMA_EXPERIMENTAL_METAL_MOE_PIPELINE"); }
                    if (enabled) { setenv("LLAMA_EXPERIMENTAL_METAL_MOE_PIPELINE", "1", 1); }
                    else { unsetenv("LLAMA_EXPERIMENTAL_METAL_MOE_PIPELINE"); }
                }
                ~pipeline_env_guard() {
                    if (present) { setenv("LLAMA_EXPERIMENTAL_METAL_MOE_PIPELINE", old.c_str(), 1); }
                    else { unsetenv("LLAMA_EXPERIMENTAL_METAL_MOE_PIPELINE"); }
                }
            } pipeline_env(pipeline);
            struct compact_env_guard {
                std::string old;
                bool present;
                compact_env_guard(bool enabled) : present(std::getenv("LLAMA_EXPERIMENTAL_EXPERT_COMPACT") != nullptr) {
                    if (present) { old = std::getenv("LLAMA_EXPERIMENTAL_EXPERT_COMPACT"); }
                    if (enabled) { setenv("LLAMA_EXPERIMENTAL_EXPERT_COMPACT", "1", 1); }
                    else { unsetenv("LLAMA_EXPERIMENTAL_EXPERT_COMPACT"); }
                }
                ~compact_env_guard() {
                    if (present) { setenv("LLAMA_EXPERIMENTAL_EXPERT_COMPACT", old.c_str(), 1); }
                    else { unsetenv("LLAMA_EXPERIMENTAL_EXPERT_COMPACT"); }
                }
            } compact_env(compact);
            struct recycle_env_guard {
                std::string old;
                bool present;
                recycle_env_guard(bool enabled) : present(std::getenv("LLAMA_EXPERIMENTAL_EXPERT_RECYCLE") != nullptr) {
                    if (present) { old = std::getenv("LLAMA_EXPERIMENTAL_EXPERT_RECYCLE"); }
                    if (enabled) { setenv("LLAMA_EXPERIMENTAL_EXPERT_RECYCLE", "1", 1); }
                    else { unsetenv("LLAMA_EXPERIMENTAL_EXPERT_RECYCLE"); }
                }
                ~recycle_env_guard() {
                    if (present) { setenv("LLAMA_EXPERIMENTAL_EXPERT_RECYCLE", old.c_str(), 1); }
                    else { unsetenv("LLAMA_EXPERIMENTAL_EXPERT_RECYCLE"); }
                }
            } recycle_env(recycle);
            // 192x192 FP32 slices occupy 192 KiB instead of a 256 KiB slot.
            const int width = quantized ? 256 : compact ? 192 : 64;
            const auto path = fs::temp_directory_path() / ("test-combined-moe-" +
                    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
            struct cleanup { fs::path path; ~cleanup() { std::error_code ec; fs::remove(path, ec); } } cleanup_file{path};
            ggml_context_ptr weights(ggml_init({8 * 1024 * 1024, nullptr, false}));
            std::array<ggml_tensor *, 3> w;
            const char * names[] = {"blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"};
            gguf_context_ptr meta(gguf_init_empty());
            gguf_set_val_str(meta.get(), "general.architecture", architecture ? architecture : "qwen4exp");
            for (size_t k = 0; k < w.size(); ++k) {
                const ggml_type types[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
                w[k] = ggml_new_tensor_3d(weights.get(), quantized ? types[k] : GGML_TYPE_F32, width, width, 4);
                ggml_set_name(w[k], names[k]);
                std::vector<float> data(width * width * 4);
                for (int i = 0; i < width * width * 4; ++i) { data[i] = float((i * 7 + k * 11) % 31) / 256.0f - 0.06f; }
                if (quantized) { ggml_quantize_chunk(w[k]->type, data.data(), w[k]->data, 0, width*4, width, nullptr); }
                else { std::memcpy(w[k]->data,data.data(),ggml_nbytes(w[k])); }
                gguf_add_tensor(meta.get(), w[k]);
            }
            t.assert_true("fixture written", gguf_write_to_file(meta.get(), path.string().c_str(), false));
            auto source = std::make_shared<llama_expert_source>(std::vector<std::string>{path.string()});
            constexpr size_t budget = 64 * 1024 * 1024;
            llama_expert_executor exec(source, {w[0], w[1], w[2]}, device, budget);
            if (architecture) {
                t.assert_true("automatic workspace deferred", !exec.combined_moe_enabled());
                const auto floor = exec.stats().minimum_bytes;
                exec.configure_acceleration();
                t.assert_equal("optional acceleration does not increase floor", exec.stats().minimum_bytes, floor);
                if (std::strcmp(architecture,"qwen4exp")==0) {
                    const size_t payload = width*width*sizeof(float);
                    const size_t allocation = ((payload+65535)/65536)*65536 + ggml_tensor_overhead();
                    const size_t eight = llama_expert_lookahead::working_bytes(payload,true,8,true);
                    const size_t four = llama_expert_lookahead::working_bytes(payload,true,4,true);
                    const size_t no_prediction = floor + exec.stats().acceleration_bytes - eight + 2*allocation + 128*1024;
                    t.assert_equal("auto starts with eight candidates",exec.stats().prefetch_candidates,size_t(8));
                    t.assert_true("auto admits four-candidate budget",exec.resize(no_prediction+four+8192));
                    t.assert_equal("auto shrinks to four candidates",exec.stats().prefetch_candidates,size_t(4));
                    t.assert_true("auto admits demand-only budget",exec.resize(no_prediction));
                    t.assert_equal("auto drops only prediction",exec.stats().prefetch_candidates,size_t(0));
                    t.assert_true("demand-only retains combined capability",exec.combined_moe_enabled());
                    t.assert_true("auto restores large budget",exec.resize(budget));
                    t.assert_equal("auto restores eight candidates",exec.stats().prefetch_candidates,size_t(8));
                }
            }
            t.assert_true("explicit capability enabled", exec.combined_moe_enabled());
            const bool automatic_window = architecture && std::strcmp(architecture, "qwen4exp") == 0;
            t.assert_equal("window qualifies automatic streaming families", exec.can_layer_window(), architecture != nullptr);
            t.assert_true("incompatible preview rejected", !exec.set_early_read(true));
            {
                ggml_context_ptr shapes(ggml_init({16384, nullptr, true}));
                auto * x = ggml_new_tensor_3d(shapes.get(), GGML_TYPE_F32, width, 1, 9);
                auto * ids = ggml_new_tensor_2d(shapes.get(), GGML_TYPE_I32, 2, 9);
                t.assert_true("long rows use ordinary path", exec.build_moe(shapes.get(), w[0], w[1], w[2], x, ids) == nullptr);
            }
            ggml_backend_ptr metal(ggml_backend_dev_init(device, nullptr));
            for (int tokens : {1, 6, 8}) {
                ggml_context_ptr graph_ctx(ggml_init({128 * 1024, nullptr, true}));
                auto * x = ggml_new_tensor_3d(graph_ctx.get(), GGML_TYPE_F32, width, 1, tokens);
                auto * ids = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_I32, 2, tokens);
                auto * out = exec.build_moe(graph_ctx.get(), w[0], w[1], w[2], x, ids);
                t.assert_true("qualified shape combined", out != nullptr);
                if (!out) { return; }
                auto * graph = ggml_new_graph_custom(graph_ctx.get(), 32, false);
                ggml_build_forward_expand(graph, out);
                ggml_backend_buffer_ptr buf(ggml_backend_alloc_ctx_tensors(graph_ctx.get(), cpu.get()));
                std::vector<float> input(width * tokens);
                std::vector<int32_t> selected(2 * tokens);
                for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i % 23) - 11) / 16.0f; }
                for (size_t i = 0; i < selected.size(); ++i) { selected[i] = (i + i / 2) % 4; }
                ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
                ggml_backend_tensor_set(ids, selected.data(), 0, selected.size() * sizeof(int32_t));

                ggml_context_ptr reference(ggml_init({128 * 1024, nullptr, true}));
                auto * rx = ggml_dup_tensor(reference.get(), x);
                auto * ri = ggml_dup_tensor(reference.get(), ids);
                std::array<ggml_tensor *, 3> rw;
                for (size_t k = 0; k < rw.size(); ++k) { rw[k] = ggml_dup_tensor(reference.get(), w[k]); }
                auto * ru = ggml_mul_mat_id(reference.get(), rw[0], rx, ri);
                auto * rg = ggml_mul_mat_id(reference.get(), rw[1], rx, ri);
                auto * ra = ggml_swiglu_split(reference.get(), rg, ru);
                auto * ry = ggml_mul_mat_id(reference.get(), rw[2], ra, ri);
                auto * ref_graph = ggml_new_graph_custom(reference.get(), 32, false);
                ggml_build_forward_expand(ref_graph, ry);
                ggml_backend_buffer_ptr rb(ggml_backend_alloc_ctx_tensors(reference.get(), metal.get()));
                for (size_t k = 0; k < rw.size(); ++k) { ggml_backend_tensor_set(rw[k], w[k]->data, 0, ggml_nbytes(w[k])); }
                ggml_backend_tensor_set(rx, input.data(), 0, input.size() * sizeof(float));
                ggml_backend_tensor_set(ri, selected.data(), 0, selected.size() * sizeof(int32_t));
                t.assert_equal("reference executes", ggml_backend_graph_compute(metal.get(), ref_graph), GGML_STATUS_SUCCESS);
                std::vector<float> expected(width * 2 * tokens), actual(expected.size());
                ggml_backend_tensor_get(ry, expected.data(), 0, expected.size() * sizeof(float));
                auto prime_one = [&]() {
                    const std::vector<int32_t> one(selected.size(), 0);
                    ggml_backend_tensor_set(ids, one.data(), 0, one.size() * sizeof(int32_t));
                    t.assert_equal("resident seed executes", ggml_backend_graph_compute(cpu.get(), graph), GGML_STATUS_SUCCESS);
                    t.assert_true("resident seed healthy", exec.error() == nullptr);
                    ggml_backend_tensor_set(ids, selected.data(), 0, selected.size() * sizeof(int32_t));
                };
                if (pipeline && tokens == 1) { prime_one(); }
                for (int repeat = 0; repeat < 2; ++repeat) {
                    t.assert_equal("combined executes", ggml_backend_graph_compute(cpu.get(), graph), GGML_STATUS_SUCCESS);
                    t.assert_true("executor healthy", exec.error() == nullptr);
                    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
                    t.assert_true("FP32 Metal reference exact", std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)) == 0);
                    t.assert_equal("all leases released", exec.stats().store.pinned_entries, size_t(0));
                    t.assert_true("within shared budget", exec.stats().memory.high_water <= budget);
                    if (compact) {
                        const auto cache = exec.stats().store;
                        t.assert_true("every cached slice uses its compact size class", cache.entries > 0 &&
                            cache.resident_bytes == cache.entries * (192 * 1024 + ggml_tensor_overhead()));
                    }
                }
                if (automatic_window && tokens == 1) {
                    const auto optional = exec.stats().acceleration_bytes;
                    const auto completed = exec.stats().combined_moe;
                    t.assert_true("automatic window admitted", exec.set_layer_window(budget));
                    t.assert_true("combined capability paused", !exec.combined_moe_enabled());
                    t.assert_true("prediction capabilities paused", !exec.moe_lookahead_enabled() && !exec.moe_graph_lookahead_enabled());
                    t.assert_true("window rejects combined graph", exec.build_moe(graph_ctx.get(), w[0], w[1], w[2], x, ids) == nullptr);
                    t.assert_equal("decode workspace retained", exec.stats().acceleration_bytes, optional);
                    {
                        ggml_context_ptr window_ctx(ggml_init({128 * 1024, nullptr, true}));
                        auto * wx = ggml_dup_tensor(window_ctx.get(), x);
                        auto * wi = ggml_dup_tensor(window_ctx.get(), ids);
                        auto * wu = exec.build(window_ctx.get(), w[0], wx, wi);
                        auto * wg = ggml_new_graph_custom(window_ctx.get(), 32, false);
                        ggml_build_forward_expand(wg, wu);
                        ggml_backend_buffer_ptr wb(ggml_backend_alloc_ctx_tensors(window_ctx.get(), cpu.get()));
                        ggml_backend_tensor_set(wx, input.data(), 0, input.size() * sizeof(float));
                        ggml_backend_tensor_set(wi, selected.data(), 0, selected.size() * sizeof(int32_t));
                        t.assert_equal("window projection executes", ggml_backend_graph_compute(cpu.get(), wg), GGML_STATUS_SUCCESS);
                        t.assert_true("window projection healthy", exec.error() == nullptr);
                        std::vector<float> window_actual(expected.size()), window_expected(expected.size());
                        ggml_backend_tensor_get(wu, window_actual.data(), 0, window_actual.size() * sizeof(float));
                        ggml_backend_tensor_get(ru, window_expected.data(), 0, window_expected.size() * sizeof(float));
                        t.assert_true("window projection FP32 exact", std::memcmp(window_actual.data(), window_expected.data(), window_actual.size() * sizeof(float)) == 0);
                    }
                    t.assert_equal("window does no combined work", exec.stats().combined_moe, completed);
                    t.assert_equal("window releases leases", exec.stats().store.pinned_entries, size_t(0));
                    t.assert_true("automatic window ends", exec.set_layer_window(0));
                    t.assert_true("decode capabilities restored", exec.combined_moe_enabled() && exec.moe_lookahead_enabled() && exec.moe_graph_lookahead_enabled());
                    t.assert_equal("cached combined graph survives window", ggml_backend_graph_compute(cpu.get(), graph), GGML_STATUS_SUCCESS);
                    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
                    t.assert_true("resumed decode FP32 exact", std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)) == 0);
                    t.assert_equal("decode workspace not reallocated", exec.stats().acceleration_bytes, optional);
                }
                if (automatic_window && tokens == 6) {
                    const std::array<const ggml_tensor *, 3> layer_weights{w[0], w[1], w[2]};
                    t.assert_true("registered window geometry", exec.can_batch_window_layer(layer_weights));
                    t.assert_true("window vector profile starts", exec.set_layer_window(budget, true));
                    llama_expert_executor::window_leases pins;
                    pins.reserve(12);
                    const int64_t needed[] = {0, 1, 2, 3};
                    exec.pin_window_experts(layer_weights, needed, 4, pins, true);
                    t.assert_equal("layer union pins all projections", exec.stats().store.pinned_entries, size_t(12));
                    ggml_context_ptr window_ctx(ggml_init({128 * 1024, nullptr, true}));
                    auto * wx = ggml_dup_tensor(window_ctx.get(), x);
                    auto * wi = ggml_dup_tensor(window_ctx.get(), ids);
                    auto * wu = exec.build(window_ctx.get(), w[0], wx, wi);
                    auto * wg = ggml_new_graph_custom(window_ctx.get(), 32, false);
                    ggml_build_forward_expand(wg, wu);
                    ggml_backend_buffer_ptr wb(ggml_backend_alloc_ctx_tensors(window_ctx.get(), cpu.get()));
                    ggml_backend_tensor_set(wx, input.data(), 0, input.size() * sizeof(float));
                    ggml_backend_tensor_set(wi, selected.data(), 0, selected.size() * sizeof(int32_t));
                    llama_expert_profile_data profile;
                    exec.set_profile(true);
                    t.assert_equal("window vector executes", ggml_backend_graph_compute(cpu.get(), wg), GGML_STATUS_SUCCESS);
                    t.assert_true("window vector healthy", exec.error() == nullptr);
                    t.assert_true("address batch actually executed", exec.get_profile(profile) && profile.moe_batch_groups > 0);
                    std::vector<float> window_actual(expected.size()), window_expected(expected.size());
                    ggml_backend_tensor_get(wu, window_actual.data(), 0, window_actual.size() * sizeof(float));
                    ggml_backend_tensor_get(ru, window_expected.data(), 0, window_expected.size() * sizeof(float));
                    t.assert_true("window vector is FP32 exact", std::memcmp(window_actual.data(), window_expected.data(),
                            window_actual.size() * sizeof(float)) == 0);
                    pins.clear();
                    t.assert_equal("window releases all layer pins", exec.stats().store.pinned_entries, size_t(0));
                    exec.trim();
                    t.assert_true("window can reuse idle demand readers", exec.can_pipeline_window());
                    auto pending = exec.enqueue_window_experts(layer_weights, needed, 4);
                    exec.drain_window_experts(pending, pins);
                    t.assert_true("exact demand completed and released its batch", !pending && pins.size() == 12);
                    t.assert_equal("queued window keeps all projections pinned", exec.stats().store.pinned_entries, size_t(12));
                    pins.clear();
                    exec.set_profile(false);
                    t.assert_true("window vector profile ends", exec.set_layer_window(0));
                }
                if (architecture && tokens == 6) {
                    t.assert_true("auto shrink to ordinary floor", exec.resize(exec.stats().minimum_bytes));
                    t.assert_true("auto workspace released at floor", !exec.combined_moe_enabled());
                    t.assert_true("auto floor respected", exec.stats().memory.used() <= exec.stats().memory.applied);
                    t.assert_true("auto impossible shrink rejected", !exec.resize(exec.stats().minimum_bytes-1));
                    t.assert_true("auto grow", exec.resize(budget));
                    t.assert_true("auto capability restored", exec.combined_moe_enabled());
                    // Old graph descriptors are intentionally invalid after auto resize.
                } else if (pipeline && tokens == 6) {
                    const auto before = exec.stats().moe_pipeline_groups;
                    t.assert_true("minimum budget resize", exec.resize(exec.stats().minimum_bytes));
                    t.assert_equal("sequential fallback executes", ggml_backend_graph_compute(cpu.get(), graph), GGML_STATUS_SUCCESS);
                    t.assert_true("sequential fallback healthy", exec.error() == nullptr);
                    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
                    t.assert_true("tight FP32 exact", std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)) == 0);
                    t.assert_equal("tight budget does not overlap", exec.stats().moe_pipeline_groups, before);
                    t.assert_true("tight budget respected", exec.stats().memory.used() <= exec.stats().memory.applied);
                    t.assert_equal("tight leases released", exec.stats().store.pinned_entries, size_t(0));
                    t.assert_true("budget restored", exec.resize(budget));
                }
                if (tokens == 8) {
                    exec.trim();
                    const size_t empty = exec.stats().memory.used();
                    if (pipeline) { prime_one(); }
                    const auto before_failure = exec.stats().moe_pipeline_groups;
                    fs::resize_file(path, 0);
                    ggml_backend_graph_compute(cpu.get(), graph);
                    t.assert_true("source failure latched", exec.error() != nullptr);
                    if (pipeline) {
                        t.assert_true("read failed after GPU submission", exec.stats().moe_pipeline_groups > before_failure);
                    }
                    t.assert_equal("failed leases released", exec.stats().store.pinned_entries, size_t(0));
                    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
                    t.assert_true("failed output cleared", std::all_of(actual.begin(), actual.end(), [](float x) { return x == 0; }));
                    exec.trim();
                    t.assert_equal("failed buffers uncharged", exec.stats().memory.used(), empty);
                }
            }
            t.assert_equal("successful MoE calls", exec.stats().combined_moe, uint64_t(pipeline ? (architecture ? 8 : 9) : 6) + uint64_t(automatic_window));
            if (pipeline) {
                t.assert_true("resident groups executed", exec.stats().moe_resident_starts > 0);
                t.assert_true("read pipeline executed", exec.stats().moe_pipeline_groups > 0);
            } else {
                t.assert_equal("disabled pipeline did no work", exec.stats().moe_pipeline_groups, uint64_t(0));
            }
        };
        t.test("combined FP32 MoE matches Metal and releases failed leases", [&](testing & t) { combined_test(t, false); });
        t.test("pipelined FP32 MoE resident cold tight and failed reads", [&](testing & t) { combined_test(t, true); });
        t.test("compact pipelined FP32 MoE cold hot tight and failed reads", [&](testing & t) { combined_test(t, true, true); });
        t.test("recycled compact FP32 MoE cold hot tight and failed reads", [&](testing & t) { combined_test(t, true, true, true); });
        t.test("portable Qwen mixed quantization queue resize failure", [&](testing & t) {
            combined_test(t,true,false,false,"qwen35moe",true);
        });
        t.test("portable LFM mixed quantization queue resize failure", [&](testing & t) {
            combined_test(t,true,false,false,"lfm2moe",true);
        });
        t.test("portable Flash FP32 queue resize failure", [&](testing & t) {
            combined_test(t,true,false,false,"qwen4exp");
        });
        t.test("portable explicit auto overrides legacy flag", [&](testing & t) {
            struct env_restore {
                const char * name; bool present; std::string old;
                env_restore(const char * n,const char * value):name(n),present(std::getenv(n)!=nullptr),old(present?std::getenv(n):""){setenv(n,value,1);}
                ~env_restore(){if(present)setenv(name,old.c_str(),1);else unsetenv(name);}
            } mode("LLAMA_MOE_STREAMING","auto"),legacy("LLAMA_EXPERIMENTAL_METAL_MOE","1");
            fixture f(4,GGML_TYPE_F32,"qwen4exp");
            llama_expert_executor exec(f.source,{f.weight},device,64*1024*1024);
            t.assert_true("explicit auto selected",exec.automatic_acceleration());
            t.assert_true("legacy eager allocation suppressed",!exec.combined_moe_enabled());
        });
        t.test("portable off retains ordinary execution", [&](testing & t) {
            const char * value=std::getenv("LLAMA_MOE_STREAMING");
            const std::string previous=value?value:"";const bool present=value;
            setenv("LLAMA_MOE_STREAMING","off",1);
            struct restore {bool present;std::string value;~restore(){if(present)setenv("LLAMA_MOE_STREAMING",value.c_str(),1);else unsetenv("LLAMA_MOE_STREAMING");}} restore{present,previous};
            fixture f(4,GGML_TYPE_F32,"qwen35moe");
            llama_expert_executor exec(f.source,{f.weight},device,64*1024*1024);
            const auto before=exec.stats().memory.used();exec.configure_acceleration();
            t.assert_true("off has no combined capability",!exec.combined_moe_enabled());
            t.assert_equal("off has no optional allocation",exec.stats().memory.used(),before);
            request req(exec,f,cpu.get(),false,1);
            t.assert_equal("ordinary off graph executes",ggml_backend_graph_compute(cpu.get(),req.graph),GGML_STATUS_SUCCESS);
            t.assert_true("ordinary off graph healthy",!exec.error());
            t.assert_equal("off does no combined work",exec.stats().combined_moe,uint64_t(0));
        });
    }
#endif

    auto routed = [&](testing & t, int tokens, bool concentrated = false, bool grouped = false) {
        for (bool per_slot : {false, true}) {
            fixture f;
            const size_t run_cap = grouped ? 2 * 1024 * 1024 : cap;
            llama_expert_executor exec(f.source, {f.weight}, device, run_cap);
            request req(exec, f, cpu.get(), per_slot, tokens);
            if (concentrated) {
                const std::vector<int32_t> ids(2 * tokens, 0);
                ggml_backend_tensor_set(req.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
            }
            t.assert_equal("graph execution", GGML_STATUS_SUCCESS, ggml_backend_graph_compute(cpu.get(), req.graph));
            t.assert_true("no deferred error", exec.error() == nullptr);
            std::vector<float> actual(8 * tokens);
            ggml_backend_tensor_get(req.output, actual.data(), 0, actual.size() * sizeof(float));
            bool equal = true;
            for (int token = 0; token < tokens; ++token) {
                for (int slot = 0; slot < 2; ++slot) {
                    for (int row = 0; row < 4; ++row) {
                        float expected = 0;
                        for (int col = 0; col < 64; ++col) {
                            expected += fixture::value(concentrated ? 0 : request::routes[(token * 2 + slot) % 6], row, col) *
                                request::activation(token, per_slot ? slot : 0, col);
                        }
                        equal &= std::fabs(actual[(token * 2 + slot) * 4 + row] - expected) <= 1e-5f;
                    }
                }
            }
            t.assert_true("routed gather/scatter matches independent dot products", equal);
            auto stats = exec.stats();
            t.assert_equal("one read per distinct expert", uint64_t(concentrated ? 1 : std::min(4, 2 * tokens)), stats.store.misses);
            uint64_t calls = 2 * tokens;
            if (std::string(backend_name) == "MTL0" && tokens >= 32) {
                int selected[4] {};
                for (int i = 0; i < 2 * tokens; ++i) { ++selected[concentrated ? 0 : request::routes[i % 6]]; }
                calls = 0;
                for (int count : selected) { calls += (count + 31) / 32; }
            }
            t.assert_equal("bounded row waves computed", calls, stats.matmuls);
            uint64_t submissions = calls;
            if (std::string(backend_name) == "MTL0" && tokens > 1 && tokens < 32) {
                int selected[4] {};
                for (int i = 0; i < 2 * tokens; ++i) { ++selected[concentrated ? 0 : request::routes[i % 6]]; }
                submissions = 0;
                for (int count : selected) { submissions += (count + 31) / 32; }
            }
            if (grouped && std::string(backend_name) == "MTL0") { submissions = (submissions + 3) / 4; }
            t.assert_equal("bounded graph submissions", stats.submissions, submissions);
            t.assert_true("expert union cannot remain resident", grouped || concentrated || stats.store.evictions > 0);
            t.assert_true("reservations stay under cap", stats.memory.high_water <= run_cap);
        }
    };
    t.test("route index preserves stable strided routes and bounds", [&](testing & t) {
        llama_expert_route_index index;
        ggml_tensor ids {};
        ids.type = GGML_TYPE_I32;
        ids.ne[0] = ids.ne[1] = 3;
        ids.nb[0] = 8; ids.nb[1] = 32;
        std::array<int32_t, 24> data {};
        const std::array<int32_t, 9> selected {3, 1, 3, 0, 3, 1, 3, 0, 3};
        for (size_t i = 0; i < selected.size(); ++i) { data[(i / 3) * 8 + (i % 3) * 2] = selected[i]; }
        ids.data = data.data();
        index.build(&ids, 4);
        std::vector<uint16_t> order;
        for (size_t i = 0; i < index.count; ++i) {
            for (auto at = index.first[index.active[i]]; at != index.end; at = index.next[at]) { order.push_back(at); }
        }
        t.assert_true("duplicates retain token-slot order", order == std::vector<uint16_t>({3, 7, 1, 5, 0, 2, 4, 6, 8}));
        t.assert_true("known lookahead uses ascending active experts", index.count == 3 && index.after(-1, 4) == 0 &&
            index.after(0, 4) == 1 && index.after(1, 4) == 3 && index.after(3, 4) == 4);
        for (int32_t bad : {-1, 4}) {
            data[0] = bad;
            bool rejected = false;
            try { index.build(&ids, 4); } catch (const std::runtime_error &) { rejected = true; }
            t.assert_true("invalid expert rejected during index construction", rejected);
        }
        std::vector<int32_t> maximum(index.max_routes, 1023);
        ids.data = maximum.data(); ids.ne[0] = 32; ids.ne[1] = 256;
        ids.nb[0] = sizeof(int32_t); ids.nb[1] = 32 * sizeof(int32_t);
        index.build(&ids, 1024);
        size_t traversed = 0;
        bool ordered = true;
        for (auto at = index.first[1023]; at != index.end && traversed <= index.max_routes; at = index.next[at]) {
            if (at >= index.max_routes) { ordered = false; break; }
            ordered &= at == traversed++;
        }
        t.assert_true("maximum shape retains every duplicate in order", ordered && traversed == index.max_routes && index.count == 1);
        ids.ne[1] = 257;
        t.assert_true("oversized rows fall back", !index.fits(&ids, 1024));
        ids.ne[1] = 256; ids.ne[0] = 33;
        t.assert_true("oversized slots fall back", !index.fits(&ids, 1024));
        ids.ne[0] = 32;
        t.assert_true("oversized expert axis falls back", !index.fits(&ids, 1025));
        ids.ne[1] = 0;
        index.build(&ids, 1024);
        t.assert_true("empty reused shape clears prior routes", index.count == 0 && index.after(-1, 1024) == 1024);
    });
    t.test("route index executor preserves outputs order and admission", [&](testing & t) {
        fixture f(8);
        const size_t total = 2 * 1024 * 1024;
        llama_expert_executor exec(f.source, {f.weight}, device, total);
        exec.set_read_overlap(true);
        if (std::string(backend_name) == "MTL0") { exec.set_single_expert_map(true); }
        const size_t original_floor = exec.stats().minimum_bytes;
        t.assert_true("admission rejects insufficient floor headroom", exec.resize(original_floor) && !exec.set_route_index(true) &&
            exec.stats().minimum_bytes == original_floor);
        t.assert_true("grow restores admission headroom", exec.resize(total));
        using events = std::vector<std::pair<int, int64_t>>;
        events observed;
        exec.set_trace([](const llama_expert_trace_event & event, void * user) noexcept {
            if (event.kind == llama_expert_trace_kind::access || event.kind == llama_expert_trace_kind::release) {
                static_cast<events *>(user)->emplace_back(int(event.kind), event.expert);
            }
        }, &observed);
        for (int tokens : {3, 32, 96, 257}) {
            for (bool per_slot : {false, true}) {
                request req(exec, f, cpu.get(), per_slot, tokens, {}, false, true);
                exec.set_route_index(false); exec.trim(); observed.clear();
                exec.set_profile(true);
                const auto before = exec.stats();
                ggml_backend_graph_compute(cpu.get(), req.graph);
                const auto baseline_stats = exec.stats();
                const events baseline_order = observed;
                std::vector<float> expected(ggml_nelements(req.output)), actual(expected.size());
                ggml_backend_tensor_get(req.output, expected.data(), 0, ggml_nbytes(req.output));
                t.assert_true("index payload explicitly admitted", exec.set_route_index(true) &&
                    exec.stats().minimum_bytes == original_floor + sizeof(llama_expert_route_index));
                exec.trim(); observed.clear(); exec.set_profile(true);
                ggml_backend_graph_compute(cpu.get(), req.graph);
                ggml_backend_tensor_get(req.output, actual.data(), 0, ggml_nbytes(req.output));
                const auto indexed_stats = exec.stats();
                llama_expert_profile_data profile; exec.get_profile(profile);
                t.assert_true("same bytes, leases and GPU grouping", !exec.error() &&
                    std::memcmp(expected.data(), actual.data(), ggml_nbytes(req.output)) == 0 && observed == baseline_order &&
                    indexed_stats.submissions - baseline_stats.submissions == baseline_stats.submissions - before.submissions &&
                    indexed_stats.matmuls - baseline_stats.matmuls == baseline_stats.matmuls - before.matmuls &&
                    indexed_stats.store.pinned_entries == 0);
                t.assert_true("one ID read per route or explicit shape fallback", tokens <= 256 ?
                    profile.route_indexed_projections == 1 && profile.route_id_reads == uint64_t(2 * tokens) :
                    profile.route_index_fallbacks == 1 && profile.route_indexed_projections == 0);
            }
        }
        const auto indexed_floor = exec.stats().minimum_bytes;
        t.assert_true("indexed floor is honored", exec.resize(indexed_floor) && !exec.resize(indexed_floor - 1));
        request req(exec, f, cpu.get(), false, 32);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        t.assert_true("floor execution keeps pins and bytes bounded", !exec.error() && exec.stats().memory.used() <= indexed_floor &&
            exec.stats().store.pinned_entries == 0);
        t.assert_true("disable releases index charge", exec.set_route_index(false) && exec.stats().minimum_bytes == original_floor);
        exec.set_trace(nullptr, nullptr);
        t.assert_true("index can be re-enabled after grow", exec.resize(total) && exec.set_route_index(true));
        const int32_t invalid = -1;
        ggml_backend_tensor_set(req.ids, &invalid, 0, sizeof(invalid));
        const auto before = exec.stats();
        ggml_backend_graph_compute(cpu.get(), req.graph);
        std::vector<float> zeros(ggml_nelements(req.output));
        ggml_backend_tensor_get(req.output, zeros.data(), 0, ggml_nbytes(req.output));
        t.assert_true("invalid route poisons before reads or GPU submit", exec.error() && exec.stats().store.misses == before.store.misses &&
            exec.stats().submissions == before.submissions && std::all_of(zeros.begin(), zeros.end(), [](float x) { return x == 0; }));
    });
    t.test("routed batch exceeds resident capacity", [&](testing & t) { routed(t, 3); });
    t.test("discrete CUDA routed projection matches resident weights", [&](testing & t) {
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "CUDA") != 0) {
            t.skip("requires CUDA");
            return;
        }
        constexpr auto ram = llama_memory_domain::host;
        constexpr auto vram = llama_memory_domain::device;
        constexpr size_t page = 65536;
        const size_t runtime = llama_memory_contract::runtime_envelope;
        llama_discrete_buffer_types types {ggml_backend_dev_buffer_type(device), ggml_backend_dev_host_buffer_type(device)};
        fixture f(4, GGML_TYPE_Q4_0);
        ggml_backend_ptr cuda(ggml_backend_dev_init(device, nullptr));
        if (!cuda) { throw std::runtime_error("cannot initialize resident reference backend"); }
        llama_memory_contract contract(runtime + 2 * page, 16 * page, types);
        {
            llama_expert_executor exec(f.source, {f.weight}, device, 0, {}, &contract);
            for (int tokens : {1, 3, 32}) {
                request req(exec, f, cpu.get(), true, tokens);
                ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
                auto * weight = ggml_dup_tensor(ctx.get(), f.weight);
                auto * input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 64, 2, tokens);
                auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 2, tokens);
                auto * output = ggml_mul_mat_id(ctx.get(), weight, input, ids);
                auto * graph = ggml_new_graph_custom(ctx.get(), 8, false);
                ggml_build_forward_expand(graph, output);
                ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), cuda.get()));
                if (!buffer) { throw std::runtime_error("resident reference allocation failed"); }
                std::vector<float> data(128 * tokens);
                std::vector<int32_t> selected(2 * tokens);
                for (int token = 0; token < tokens; ++token) {
                    for (int slot = 0; slot < 2; ++slot) {
                        selected[2 * token + slot] = request::routes[(2 * token + slot) % 6];
                        for (int col = 0; col < 64; ++col) {
                            data[(2 * token + slot) * 64 + col] = request::activation(token, slot, col);
                        }
                    }
                }
                ggml_backend_buffer_set_usage(buffer.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
                ggml_backend_tensor_set(weight, f.weight->data, 0, ggml_nbytes(weight));
                ggml_backend_tensor_set(input, data.data(), 0, ggml_nbytes(input));
                ggml_backend_tensor_set(ids, selected.data(), 0, ggml_nbytes(ids));
                t.assert_equal("resident projection", GGML_STATUS_SUCCESS, ggml_backend_graph_compute(cuda.get(), graph));
                t.assert_equal("streaming bridge", GGML_STATUS_SUCCESS, ggml_backend_graph_compute(cpu.get(), req.graph));
                if (exec.error()) { throw std::runtime_error(exec.error()); }
                std::vector<float> expected(8 * tokens), actual(8 * tokens);
                ggml_backend_tensor_get(output, expected.data(), 0, ggml_nbytes(output));
                ggml_backend_tensor_get(req.output, actual.data(), 0, ggml_nbytes(req.output));
                const auto different = std::mismatch(expected.begin(), expected.end(), actual.begin(), [](float a, float b) {
                    return std::memcmp(&a, &b, sizeof(float)) == 0;
                });
                t.assert_true("exact routed output at " + std::to_string(tokens) + " tokens", different.first == expected.end());
            }
            const auto stats = exec.stats();
            t.assert_true("experts read on demand and reused", stats.store.misses == 4 && stats.store.hits > 0);
            const auto device_memory = contract.ledger->snapshot(vram);
            const auto host_memory = contract.ledger->snapshot(ram);
            t.assert_true("both peaks are bounded", device_memory.high_water <= 16 * page && host_memory.high_water <= runtime + 2 * page);
            t.assert_true("expert bytes in VRAM only", device_memory.categories[size_t(llama_memory_category::experts)].allocated > 0 &&
                host_memory.categories[size_t(llama_memory_category::experts)].total() == 0);
            t.assert_true("device scratch was admitted and released", device_memory.released > 0);
        }
        t.assert_equal("VRAM returned on teardown", size_t(0), contract.ledger->snapshot(vram).used());
        t.assert_equal("RAM returned to envelope", runtime, contract.ledger->snapshot(ram).used());
    });
    t.test("vector micrographs preserve routed row order", [&](testing & t) { routed(t, 3); routed(t, 31); routed(t, 31, true); });
    t.test("matrix waves preserve routed row order", [&](testing & t) { routed(t, 32); routed(t, 96); });
    if (std::string(backend_name) == "MTL0") {
        t.test("prepared single expert map matches generic Metal dispatch", [&](testing & t) {
            using prepare_t = bool (*)(ggml_tensor *);
            auto prepare = reinterpret_cast<prepare_t>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(device), "ggml_backend_metal_prepare_single_expert"));
            t.assert_true("private preparation hook available", prepare != nullptr);
            if (!prepare) { return; }
            ggml_backend_ptr backend(ggml_backend_dev_init(device, nullptr));
            ggml_context_ptr ctx(ggml_init({65536, nullptr, true}));
            auto * w = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, 64, 1);
            auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, 1, 32);
            auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, 32);
            auto * out = ggml_mul_mat_id(ctx.get(), w, x, ids);
            auto * graph = ggml_new_graph_custom(ctx.get(), 8, false);
            ggml_build_forward_expand(graph, out);
            ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
            std::vector<float> weights(256 * 64), input(256 * 32);
            for (size_t i = 0; i < weights.size(); ++i) { weights[i] = float(int(i % 43) - 21) / 37.0f; }
            for (size_t i = 0; i < input.size(); ++i) { input[i] = float(int(i % 71) - 35) / 61.0f; }
            std::array<int32_t, 32> routes {};
            ggml_backend_tensor_set(w, weights.data(), 0, ggml_nbytes(w));
            ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));
            ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
            t.assert_equal("generic matrix executes", GGML_STATUS_SUCCESS, ggml_backend_graph_compute(backend.get(), graph));
            std::vector<float> expected(64 * 32), actual(expected.size());
            ggml_backend_tensor_get(out, expected.data(), 0, ggml_nbytes(out));
            // The shared output tail is readable only after completion. Compare
            // all descriptors and live row entries; the dummy entry is unused.
            std::array<int32_t, 36> generic_map {};
            std::memcpy(generic_map.data(), static_cast<char *>(out->data) + ggml_nbytes(out), sizeof(generic_map));
            const size_t allocated = ggml_backend_buffer_get_size(buffer.get());
            t.assert_true("bound zero-ID operation prepares", prepare(out));
            t.assert_true("CPU map exactly matches GPU prepass", std::memcmp(generic_map.data(),
                static_cast<char *>(out->data) + ggml_nbytes(out), sizeof(generic_map)) == 0);
            t.assert_equal("prepared matrix executes", GGML_STATUS_SUCCESS, ggml_backend_graph_compute(backend.get(), graph));
            ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
            t.assert_true("unchanged matrix kernel is bitwise exact", std::memcmp(expected.data(), actual.data(), ggml_nbytes(out)) == 0);
            t.assert_equal("no additional allocation", allocated, ggml_backend_buffer_get_size(buffer.get()));
            routes[0] = -1;
            ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
            t.assert_true("skipped ID rejects and clears preparation", !prepare(out) && out->op_params[0] == 0);
            t.assert_equal("rejected operation uses generic path", GGML_STATUS_SUCCESS, ggml_backend_graph_compute(backend.get(), graph));
            ggml_backend_tensor_get(out, actual.data(), 0, ggml_nbytes(out));
            t.assert_true("generic skip still zeros its row", std::all_of(actual.begin(), actual.begin() + 64, [](float v) { return v == 0; }));
            routes[0] = 1;
            ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
            t.assert_true("nonzero ID cannot use prepared path", !prepare(out));
            routes[0] = 0;
            ggml_backend_tensor_set(ids, routes.data(), 0, ggml_nbytes(ids));
            void * saved = out->data;
            out->data = static_cast<char *>(ggml_backend_buffer_get_base(buffer.get())) + allocated - sizeof(float);
            t.assert_true("missing output tail is rejected", !prepare(out));
            out->data = saved;
            ids->op = GGML_OP_DUP;
            t.assert_true("GPU-generated IDs are rejected", !prepare(out));
            ids->op = GGML_OP_NONE;
            t.assert_true("restored immutable inputs prepare again", prepare(out));
        });
        t.test("prepared expert waves preserve budget and fallback", [&](testing & t) {
            fixture f;
            const size_t total = 2 * 1024 * 1024;
            llama_expert_executor exec(f.source, {f.weight}, device, total);
            const size_t floor = exec.stats().minimum_bytes;
            for (bool per_slot : {false, true}) {
                request req(exec, f, cpu.get(), per_slot, 96);
                std::vector<float> expected(96 * 2 * 4), actual(expected.size());
                exec.set_single_expert_map(false);
                ggml_backend_graph_compute(cpu.get(), req.graph);
                ggml_backend_tensor_get(req.output, expected.data(), 0, ggml_nbytes(req.output));
                t.assert_true("map preparation enables", exec.set_single_expert_map(true));
                exec.trim();
                exec.set_profile(true);
                ggml_backend_graph_compute(cpu.get(), req.graph);
                ggml_backend_tensor_get(req.output, actual.data(), 0, ggml_nbytes(req.output));
                llama_expert_profile_data profile;
                exec.get_profile(profile);
                t.assert_true("cold grouped output matches generic bytes", !exec.error() && profile.prepared_map_ops > 4 &&
                    std::memcmp(expected.data(), actual.data(), ggml_nbytes(req.output)) == 0);
                t.assert_true("floor unchanged", exec.stats().minimum_bytes == floor && exec.resize(floor));
                exec.set_profile(true);
                ggml_backend_graph_compute(cpu.get(), req.graph);
                exec.get_profile(profile);
                ggml_backend_tensor_get(req.output, actual.data(), 0, ggml_nbytes(req.output));
                t.assert_true("single-wave floor preserves arithmetic and cap", !exec.error() && profile.prepared_map_ops > 0 &&
                    std::memcmp(expected.data(), actual.data(), ggml_nbytes(req.output)) == 0 &&
                    exec.stats().memory.used() <= floor && exec.stats().store.pinned_entries == 0);
                t.assert_true("grow preserves original floor and cap", exec.resize(total) && exec.stats().memory.high_water <= total);
            }
            request vector(exec, f, cpu.get(), false, 3);
            exec.set_profile(true);
            ggml_backend_graph_compute(cpu.get(), vector.graph);
            llama_expert_profile_data profile;
            exec.get_profile(profile);
            t.assert_true("vector fallback does not prepare maps", !exec.error() && profile.prepared_map_ops == 0);
        });
        t.test("grouped waves preserve independent expert outputs", [&](testing & t) {
            routed(t, 1, false, true); routed(t, 3, false, true); routed(t, 31, false, true);
            routed(t, 96, false, true); routed(t, 96, true, true);
        });
        t.test("group acquisition failure releases pending leases", [&](testing & t) {
            fixture f;
            const size_t run_cap = 2 * 1024 * 1024;
            llama_expert_executor exec(f.source, {f.weight}, device, run_cap);
            const size_t fixed = exec.stats().memory.used();
            struct injection { fs::path path; bool done = false, failed = false; } hook {f.path};
            exec.set_trace([](const llama_expert_trace_event & event, void * user) noexcept {
                auto & h = *static_cast<injection *>(user);
                if (event.kind == llama_expert_trace_kind::access && !h.done) {
                    std::error_code ec;
                    fs::resize_file(h.path, 0, ec);
                    h.done = true; h.failed = bool(ec);
                }
            }, &hook);
            request req(exec, f, cpu.get(), false);
            ggml_backend_graph_compute(cpu.get(), req.graph);
            t.assert_true("failure injected after first completed read", hook.done && !hook.failed);
            t.assert_true("context reports deferred source failure", exec.error() != nullptr);
            const auto stats = exec.stats();
            t.assert_true("pending group never submitted", stats.store.misses == 1 && stats.submissions == 0);
            t.assert_equal("all leases released on exception", stats.store.pinned_entries, size_t(0));
            std::vector<float> actual(24, 1);
            ggml_backend_tensor_get(req.output, actual.data(), 0, actual.size() * sizeof(float));
            t.assert_true("failed projection output cleared", std::all_of(actual.begin(), actual.end(), [](float v) { return v == 0; }));
            exec.trim();
            exec.set_trace(nullptr, nullptr); // hook storage is destroyed before exec
            t.assert_equal("failed group payload charges returned", exec.stats().memory.used(), fixed);
            t.assert_true("failure stayed under budget", stats.memory.high_water <= run_cap);
        });
    }
    for (const bool vector : {false, true}) {
        t.test(vector ? "Qwen4Exp vector overlap cold hot and tight budget" :
                       "bounded read overlap preserves cold hot and tight-budget outputs", [&](testing & t) {
            const int experts = vector ? 32 : 8;
            fixture f(experts, GGML_TYPE_F32, vector ? "qwen4exp" : "qwen3moe");
            const int tokens = vector ? 16 : 64;
            const size_t run_cap = (vector ? 4 : 2) * 1024 * 1024;
            llama_expert_executor exec(f.source, {f.weight}, device, run_cap);
            exec.set_read_overlap(true);
            request req(exec, f, cpu.get(), true, tokens);
            std::vector<int32_t> ids(2 * tokens);
            for (size_t i = 0; i < ids.size(); ++i) { ids[i] = i % experts; }
            ggml_backend_tensor_set(req.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
            const bool metal = std::string(backend_name) == "MTL0";
            auto execute = [&]() {
                exec.set_profile(true);
                ggml_backend_graph_compute(cpu.get(), req.graph);
                t.assert_true("no deferred error", exec.error() == nullptr);
                std::vector<float> actual(8 * tokens);
                ggml_backend_tensor_get(req.output, actual.data(), 0, actual.size() * sizeof(float));
                bool exact = true;
                for (int token = 0; token < tokens; ++token) {
                    for (int slot = 0; slot < 2; ++slot) {
                        for (int row = 0; row < 4; ++row) {
                            float expected = 0;
                            for (int col = 0; col < 64; ++col) {
                                expected += fixture::value(ids[token * 2 + slot], row, col) * request::activation(token, slot, col);
                            }
                            exact &= actual[(token * 2 + slot) * 4 + row] == expected;
                        }
                    }
                }
                t.assert_true("strided routed output matches independent dots", exact);
                t.assert_equal("all compute and read leases released", exec.stats().store.pinned_entries, size_t(0));
                llama_expert_profile_data profile;
                exec.get_profile(profile);
                return profile;
            };
            const auto cold = execute();
            t.assert_equal("cold demand read exactly once", exec.stats().store.misses, uint64_t(experts));
            t.assert_equal("bounded future acquisitions", cold.overlap_acquisitions, uint64_t(metal ? experts / 2 : 0));
            t.assert_equal("cold next group read during submission interval", cold.overlap_read_bytes, uint64_t(metal ? experts / 2 * 1024 : 0));
            if (metal) { t.assert_equal("two bounded expert groups submitted", exec.stats().submissions, uint64_t(2)); }
            const auto hot = execute();
            t.assert_equal("hot request performs no source reads", hot.direct_read_bytes, uint64_t(0));
            t.assert_equal("hot acquired leases transfer without duplicate demand", exec.stats().store.hits, uint64_t(experts));
            t.assert_true("shrink to single expert floor", exec.resize(exec.stats().minimum_bytes));
            const auto tight = execute();
            t.assert_equal("no optional reads at floor", tight.overlap_acquisitions, uint64_t(0));
            t.assert_true("used bytes respect floor", exec.stats().memory.used() <= exec.stats().memory.applied);
            if (vector && metal) {
                const size_t reduced_cap = exec.stats().minimum_bytes - 1;
                llama_expert_executor reduced(f.source, {f.weight}, device, reduced_cap);
                t.assert_true("construction falls back to a smaller admitted workspace",
                    reduced.stats().minimum_bytes <= reduced_cap && reduced.stats().minimum_bytes < exec.stats().minimum_bytes);
            }
        });
    }
    if (std::string(backend_name) == "MTL0") {
        t.test("Qwen4Exp selects ClassLRU only for qualified prefill", [&](testing & t) {
            fixture f(8, GGML_TYPE_F32, "qwen4exp");
            const size_t budget = size_t(8) * 1024 * 1024 * 1024;
            llama_expert_executor exec(f.source, {f.weight}, device, budget);
            const auto floor = exec.stats().minimum_bytes;
            exec.select_decode_policy(319, 8);
            t.assert_equal("short batch keeps LRU", exec.stats().minimum_bytes, floor);
            exec.select_decode_policy(320, 32);
            t.assert_equal("matrix microbatch keeps LRU", exec.stats().minimum_bytes, floor);
            exec.select_decode_policy(320, 8);
            const auto class_floor = exec.stats().minimum_bytes;
            t.assert_true("long vector prefill admits history", class_floor > floor);
            t.assert_true("invalid shrink rejected", !exec.resize(1));
            t.assert_equal("rejected resize leaves history admitted", exec.stats().minimum_bytes, class_floor);
            exec.select_decode_policy(1, 8);
            t.assert_equal("decode releases optional history", exec.stats().minimum_bytes, floor);
            exec.select_decode_policy(320, 8);
            t.assert_true("shrink can release ClassLRU history at the old floor", exec.resize(floor));
            exec.select_decode_policy(320, 8);
            t.assert_equal("smaller budget keeps LRU", exec.stats().minimum_bytes, floor);
            t.assert_true("explicit LRU accepted", exec.set_policy("lru") && exec.resize(budget));
            exec.select_decode_policy(320, 8);
            t.assert_equal("explicit LRU overrides the automatic policy", exec.stats().minimum_bytes, floor);
        });

        t.test("read ahead survives repeated groups for one expert", [&](testing & t) {
            fixture f(8);
            llama_expert_executor exec(f.source, {f.weight}, device, 2 * 1024 * 1024);
            exec.set_read_overlap(true);
            exec.set_profile(true);
            request req(exec, f, cpu.get(), true, 256);
            std::vector<int32_t> ids(512);
            for (size_t i = 0; i < ids.size(); ++i) { ids[i] = i < 384 ? 0 : 1 + i % 7; }
            ggml_backend_tensor_set(req.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
            ggml_backend_graph_compute(cpu.get(), req.graph);
            t.assert_true("no deferred error", exec.error() == nullptr);
            std::vector<float> actual(2048);
            ggml_backend_tensor_get(req.output, actual.data(), 0, actual.size() * sizeof(float));
            bool exact = true;
            for (int token = 0; token < 256; ++token) {
                for (int slot = 0; slot < 2; ++slot) {
                    for (int row = 0; row < 4; ++row) {
                        float expected = 0;
                        for (int col = 0; col < 64; ++col) {
                            expected += fixture::value(ids[token * 2 + slot], row, col) * request::activation(token, slot, col);
                        }
                        exact &= actual[(token * 2 + slot) * 4 + row] == expected;
                    }
                }
            }
            t.assert_true("all rows retain their expert and slot", exact);
            llama_expert_profile_data profile;
            exec.get_profile(profile);
            const auto stats = exec.stats();
            t.assert_equal("each subsequent expert acquired once", profile.overlap_acquisitions, uint64_t(7));
            t.assert_equal("no duplicate reads while first expert spans groups", stats.store.misses, uint64_t(8));
            t.assert_equal("original nineteen waves in five submissions", stats.submissions, uint64_t(5));
            t.assert_equal("all leases released", stats.store.pinned_entries, size_t(0));
            t.assert_true("bounded peak", stats.memory.high_water <= 2 * 1024 * 1024);
        });
        for (const bool vector : {false, true}) {
            t.test(vector ? "Qwen4Exp vector overlap read failure drains leases" :
                           "overlap source failure drains submitted graph and leases", [&](testing & t) {
                const int experts = vector ? 32 : 8;
                fixture f(experts, GGML_TYPE_F32, vector ? "qwen4exp" : "qwen3moe");
                const int tokens = vector ? 16 : 64;
                const size_t run_cap = (vector ? 4 : 2) * 1024 * 1024;
                llama_expert_executor exec(f.source, {f.weight}, device, run_cap);
                exec.set_read_overlap(true);
                exec.set_profile(true);
                const auto fixed = exec.stats().memory.used();
                struct fault { fixture * f; size_t trigger; size_t reads = 0; bool failed = false; } hook {&f, size_t(experts / 2)};
                exec.set_trace([](const llama_expert_trace_event & event, void * user) noexcept {
                    auto & h = *static_cast<fault *>(user);
                    if (event.kind == llama_expert_trace_kind::access && ++h.reads == h.trigger) {
                        std::error_code ec;
                        fs::resize_file(h.f->path, 1, ec);
                        h.failed = bool(ec);
                    }
                }, &hook);
                request req(exec, f, cpu.get(), false, tokens);
                std::vector<int32_t> ids(2 * tokens);
                for (size_t i = 0; i < ids.size(); ++i) { ids[i] = i % experts; }
                ggml_backend_tensor_set(req.ids, ids.data(), 0, ids.size() * sizeof(int32_t));
                ggml_backend_graph_compute(cpu.get(), req.graph);
                llama_expert_profile_data profile;
                exec.get_profile(profile);
                t.assert_true("fault follows one completed read group", hook.reads == hook.trigger && !hook.failed);
                t.assert_true("failure occurs after GPU submission", profile.submit_ns > 0 && profile.overlap_acquire_ns > 0);
                t.assert_true("executor rejects incomplete projection", exec.error() != nullptr);
                t.assert_equal("no pending leases after failure", exec.stats().store.pinned_entries, size_t(0));
                std::vector<float> actual(8 * tokens, 1);
                ggml_backend_tensor_get(req.output, actual.data(), 0, actual.size() * sizeof(float));
                t.assert_true("failed result cleared", std::all_of(actual.begin(), actual.end(), [](float x) { return x == 0; }));
                exec.trim();
                exec.set_trace(nullptr, nullptr);
                t.assert_equal("all payload charges returned after completion", exec.stats().memory.used(), fixed);
                t.assert_true("no cap overrun", exec.stats().memory.high_water <= run_cap);
            });
        }
    }
    t.test("exact early reads preserve outputs budget and graph binding", [&](testing & t) {
        if (std::string(backend_name) != "MTL0") { return; }
        fixture f(8), target(8);
        ggml_set_name(target.weight, "blk.0.ffn_gate_exps.weight");
        auto * data = static_cast<float *>(target.weight->data);
        for (size_t i = 0; i < ggml_nbytes(target.weight) / sizeof(float); ++i) { data[i] *= 2; }
        for (int shard = 0; shard < 2; ++shard) {
            auto & part = shard ? target : f;
            gguf_context_ptr meta(gguf_init_empty());
            gguf_set_val_str(meta.get(), "general.architecture", "qwen35moe");
            gguf_set_val_u16(meta.get(), "split.count", 2);
            gguf_set_val_u16(meta.get(), "split.no", shard);
            gguf_set_val_i32(meta.get(), "split.tensors.count", 2);
            gguf_add_tensor(meta.get(), part.weight);
            t.assert_true("write split fixture", gguf_write_to_file(meta.get(), part.path.string().c_str(), false));
        }
        f.source = std::make_shared<llama_expert_source>(std::vector<std::string>{f.path.string(), target.path.string()});
        const size_t total = 4 * 1024 * 1024;
        llama_expert_executor exec(f.source, {f.weight, target.weight}, device, total);
        request req(exec, f, cpu.get(), true, 64, {target.weight}, false, true);
        auto * last = ggml_graph_node(req.graph, ggml_graph_n_nodes(req.graph) - 1);
        std::vector<float> baseline(2 * 64 * 8), actual(baseline.size());
        auto capture = [&](std::vector<float> & values) {
            ggml_backend_tensor_get(req.output, values.data(), 0, values.size() * sizeof(float) / 2);
            ggml_backend_tensor_get(last, values.data() + values.size() / 2, 0, values.size() * sizeof(float) / 2);
        };
        auto run = [&]() {
            t.assert_equal("graph execution", GGML_STATUS_SUCCESS, ggml_backend_graph_compute(cpu.get(), req.graph));
            t.assert_true("no deferred error", exec.error() == nullptr);
            t.assert_equal("no pins survive callback", size_t(0), exec.stats().store.pinned_entries);
            t.assert_true("managed cap respected", exec.stats().memory.used() <= exec.stats().memory.applied);
        };
        run(); capture(baseline);
        const auto baseline_stats = exec.stats();
        exec.trim();
        const auto floor = exec.stats().minimum_bytes;
        t.assert_true("admit early binding", exec.set_early_read(true));
        t.assert_true("binding charged to floor", exec.stats().minimum_bytes > floor);
        t.assert_true("preview cannot combine", !exec.set_preview(2));
        t.assert_true("class policy cannot combine", !exec.set_policy("class-lru"));
        exec.set_profile(true);
        exec.bind_early_graph(req.graph);
        run(); capture(actual);
        llama_expert_profile_data profile;
        exec.get_profile(profile);
        t.assert_true("source and target output bytes unchanged", baseline == actual);
        t.assert_equal("one adjacent independent pair", uint64_t(1), profile.early_bound_pairs);
        t.assert_equal("one tail window", uint64_t(1), profile.early_windows);
        t.assert_equal("four exact first experts", uint64_t(4), profile.early_acquisitions);
        t.assert_equal("exact early slice bytes", uint64_t(4 * 1024), profile.early_read_bytes);
        t.assert_equal("no added logical reads", baseline_stats.store.logical_read_bytes * 2, exec.stats().store.logical_read_bytes);
        t.assert_equal("same arithmetic operations", baseline_stats.matmuls * 2, exec.stats().matmuls);
        t.assert_equal("same submissions", baseline_stats.submissions * 2, exec.stats().submissions);
        exec.set_profile(true); exec.bind_early_graph(req.graph); run();
        exec.get_profile(profile);
        t.assert_equal("warm pair adds no reads", uint64_t(0), profile.early_acquisitions);
        t.assert_equal("warm first experts skipped", uint64_t(4), profile.early_skip_hot);
        // Rebinding must discard stale pairs; narrowed and duplicate occurrences
        // fail closed even when the registered projection identity is unchanged.
        exec.bind_early_graph(nullptr); exec.trim(); exec.set_profile(true); run();
        exec.get_profile(profile);
        t.assert_equal("unbound graph cannot read early", uint64_t(0), profile.early_windows);
        for (bool narrow : {true, false}) {
            request other(exec, f, cpu.get(), false, 32,
                          narrow ? std::vector<const ggml_tensor *>{target.weight} :
                                   std::vector<const ggml_tensor *>{target.weight, f.weight}, narrow);
            exec.set_profile(true); exec.bind_early_graph(other.graph);
            exec.get_profile(profile);
            t.assert_equal("narrowed or ambiguous source rejected", uint64_t(0), profile.early_bound_pairs);
        }
        t.assert_true("shrink to charged floor", exec.resize(exec.stats().minimum_bytes));
        exec.set_profile(true); exec.bind_early_graph(req.graph); run(); capture(actual);
        exec.get_profile(profile);
        t.assert_equal("floor skips early reads", uint64_t(0), profile.early_acquisitions);
        t.assert_equal("floor budget guard observed", uint64_t(1), profile.early_skip_budget);
        t.assert_true("floor output bytes unchanged", baseline == actual);
        t.assert_true("grow", exec.resize(total));
        t.assert_true("admit route index", exec.set_route_index(true));
        exec.trim(); exec.set_profile(true); exec.bind_early_graph(req.graph); run(); capture(actual);
        exec.get_profile(profile);
        t.assert_equal("indexed grow restores exact reads", uint64_t(4), profile.early_acquisitions);
        t.assert_true("indexed output bytes unchanged", baseline == actual);
        // Invalid future source after submission: no lease escapes GPU completion,
        // and neither the source partial result nor target output is accepted.
        exec.trim();
        request source_only(exec, f, cpu.get(), false, 64);
        exec.bind_early_graph(source_only.graph);
        ggml_backend_graph_compute(cpu.get(), source_only.graph);
        t.assert_true("source warmed before failure", !exec.error());
        fs::resize_file(target.path, fs::file_size(target.path) - 1);
        exec.bind_early_graph(req.graph);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        t.assert_true("early read error is sticky", exec.error() != nullptr);
        capture(actual);
        t.assert_true("failed outputs cleared", std::all_of(actual.begin(), actual.end(), [](float x) { return x == 0; }));
        t.assert_equal("failure releases all pins", size_t(0), exec.stats().store.pinned_entries);
        t.assert_true("failure stays within cap", exec.stats().memory.used() <= exec.stats().memory.applied);
    });

    t.test("runtime preview is bounded and budgeted without graph changes", [&](testing & t) {
        fixture f(8);
        fixture optional_shard(8);
        const int prefetch_mode = std::string(backend_name) == "MTL0" ? 2 : 3;
        ggml_context_ptr extra(ggml_init({65536, nullptr, false}));
        std::array<const ggml_tensor *, 3> targets {};
        gguf_context_ptr meta(gguf_init_empty());
        gguf_set_val_str(meta.get(), "general.architecture", "qwen35moe");
        gguf_set_val_u16(meta.get(), "split.count", 2);
        gguf_set_val_u16(meta.get(), "split.no", 1);
        gguf_set_val_i32(meta.get(), "split.tensors.count", 4);
        gguf_context_ptr demand_meta(gguf_init_empty());
        gguf_set_val_str(demand_meta.get(), "general.architecture", "qwen35moe");
        gguf_set_val_u16(demand_meta.get(), "split.count", 2);
        gguf_set_val_u16(demand_meta.get(), "split.no", 0);
        gguf_set_val_i32(demand_meta.get(), "split.tensors.count", 4);
        gguf_add_tensor(demand_meta.get(), f.weight);
        const char * names[] = {"blk.1.ffn_gate_exps.weight", "blk.1.ffn_up_exps.weight", "blk.1.ffn_down_exps.weight"};
        for (size_t i = 0; i < 3; ++i) {
            auto * weight = ggml_new_tensor_3d(extra.get(), GGML_TYPE_F32, 64, 4, 8);
            ggml_set_name(weight, names[i]);
            std::memcpy(weight->data, f.weight->data, ggml_nbytes(weight));
            gguf_add_tensor(meta.get(), weight);
            targets[i] = weight;
        }
        t.assert_true("fixture source writes", gguf_write_to_file(demand_meta.get(), f.path.string().c_str(), false) &&
            gguf_write_to_file(meta.get(), optional_shard.path.string().c_str(), false));
        f.source = std::make_shared<llama_expert_source>(std::vector<std::string>{f.path.string(), optional_shard.path.string()});
        ggml_context_ptr rc(ggml_init({4096, nullptr, true}));
        auto * router = ggml_new_tensor_2d(rc.get(), GGML_TYPE_F32, 64, 8);
        ggml_backend_buffer_ptr rb(ggml_backend_alloc_ctx_tensors(rc.get(), cpu.get()));
        std::array<float, 512> rw {};
        for (size_t i = 0; i < rw.size(); ++i) { rw[i] = float(8 - i / 64) * 0.125f; }
        ggml_backend_tensor_set(router, rw.data(), 0, sizeof(rw));
        std::vector<const ggml_tensor *> weights {f.weight, targets[0], targets[1], targets[2]};
        const std::vector<llama_expert_preview_layer> layers {{f.weight, f.weight, router, targets}};
        const size_t total = 4 * 1024 * 1024;
        llama_expert_executor exec(f.source, weights, device, total);
        request req(exec, f, cpu.get(), false, 32, {targets.begin(), targets.end()});
        exec.bind_preview_graph(req.graph);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        std::vector<float> baseline(32 * 2 * 4), actual(baseline.size());
        ggml_backend_tensor_get(req.output, baseline.data(), 0, baseline.size() * sizeof(float));
        exec.trim();
        const auto floor = exec.stats().minimum_bytes;
        t.assert_true("predictor config admitted", exec.set_preview(1, layers));
        const auto configured_floor = exec.stats().minimum_bytes;
        t.assert_true("predictor workspace raises floor", configured_floor > floor);
        exec.set_profile(true);
        const auto misses = exec.stats().store.misses;
        exec.bind_preview_graph(req.graph);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        llama_expert_profile_data profile;
        exec.get_profile(profile);
        t.assert_true("preview-only computes once without speculative reads", profile.preview_generations == 1 && profile.preview_rows > 0 &&
            profile.prefetch_slices == 0 && exec.stats().store.misses - misses == 16);
        t.assert_true("disable releases predictor charge", exec.set_preview(0) && exec.stats().minimum_bytes == floor);
        exec.trim();
        t.assert_true("bounded prefetch config admitted", exec.set_preview(prefetch_mode, layers));
        exec.set_profile(true);
        exec.bind_preview_graph(req.graph);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        ggml_backend_tensor_get(req.output, actual.data(), 0, actual.size() * sizeof(float));
        exec.get_profile(profile);
        t.assert_true("all output bytes preserved", !exec.error() && std::memcmp(baseline.data(), actual.data(), baseline.size() * sizeof(float)) == 0);
        t.assert_true("tail scheduling never competes with known reads", prefetch_mode != 2 || profile.preview_competing_windows == 0);
        t.assert_true("preview subphase counters are nested", profile.preview_dot_ns > 0 &&
            profile.preview_dot_ns + profile.preview_rank_ns + profile.preview_residency_ns <= profile.preview_ns);
        t.assert_true("global four bundles only", profile.preview_generations == 1 && profile.prefetch_experts == 4 && profile.prefetch_slices == 12);
        t.assert_true("all optional leases end at callback boundary", exec.stats().store.pinned_entries == 0);
        t.assert_true("all payload remains within cap", exec.stats().memory.high_water <= total);
        exec.set_profile(true);
        exec.bind_preview_graph(req.graph);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        exec.get_profile(profile);
        t.assert_true("resident next layer skips predictor and reads", profile.preview_skip_hot == 1 && profile.preview_generations == 0 && profile.prefetch_slices == 0);
        t.assert_true("shrink to predictor-aware floor", exec.resize(configured_floor));
        exec.set_profile(true);
        exec.bind_preview_graph(req.graph);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        exec.get_profile(profile);
        ggml_backend_tensor_get(req.output, actual.data(), 0, actual.size() * sizeof(float));
        t.assert_true("floor skips optional work and preserves computation", !exec.error() && profile.preview_skip_budget == 1 && profile.prefetch_slices == 0 &&
            std::memcmp(baseline.data(), actual.data(), baseline.size() * sizeof(float)) == 0);
        t.assert_true("below floor is rejected", !exec.resize(configured_floor - 1));
        t.assert_true("grow permits bounded prefetch again", exec.resize(total));
        exec.set_profile(true);
        exec.bind_preview_graph(req.graph);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        exec.get_profile(profile);
        t.assert_true("prefetch resumes without stale leases", !exec.error() && profile.prefetch_experts == 4 && exec.stats().store.pinned_entries == 0);
        t.assert_true("disable returns original floor", exec.set_preview(0) && exec.stats().minimum_bytes == floor);
        t.assert_true("short-window probe config admitted", exec.set_preview(2, layers));
        exec.trim();
        exec.set_profile(true);
        request short_req(exec, f, cpu.get(), false, 1, {targets.begin(), targets.end()});
        exec.bind_preview_graph(short_req.graph);
        ggml_backend_graph_compute(cpu.get(), short_req.graph);
        exec.get_profile(profile);
        t.assert_true("short windows do not predict or prefetch", !exec.error() && profile.preview_skip_short == 1 &&
            profile.preview_generations == 0 && profile.prefetch_slices == 0);
        exec.trim();
        exec.set_profile(true);
        request pruned(exec, f, cpu.get(), false, 32);
        exec.bind_preview_graph(pruned.graph);
        ggml_backend_graph_compute(cpu.get(), pruned.graph);
        exec.get_profile(profile);
        t.assert_true("pruned target clears prior graph eligibility", !exec.error() && profile.preview_skip_graph == 1 &&
            profile.preview_generations == 0 && profile.prefetch_slices == 0);
        exec.trim();
        exec.set_profile(true);
        request narrowed(exec, f, cpu.get(), false, 32, {targets.begin(), targets.end()}, true);
        exec.bind_preview_graph(narrowed.graph);
        ggml_backend_graph_compute(cpu.get(), narrowed.graph);
        exec.get_profile(profile);
        t.assert_true("narrowed target does not reuse source row predictions", !exec.error() && profile.preview_skip_shape == 1 &&
            profile.preview_generations == 0 && profile.prefetch_slices == 0);
        exec.trim();
        exec.set_profile(true);
        request duplicate(exec, f, cpu.get(), false, 32, {targets[0], targets[1], targets[2], targets[0]});
        exec.bind_preview_graph(duplicate.graph);
        ggml_backend_graph_compute(cpu.get(), duplicate.graph);
        exec.get_profile(profile);
        t.assert_true("ambiguous target disables optional work", !exec.error() && profile.preview_skip_graph == 1 &&
            profile.preview_generations == 0 && profile.prefetch_slices == 0);
        exec.trim();
        exec.set_profile(true);
        exec.bind_preview_graph(nullptr);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        exec.get_profile(profile);
        t.assert_true("unbound preview fails closed", !exec.error() && profile.preview_skip_graph == 1 &&
            profile.preview_generations == 0 && profile.prefetch_slices == 0);
        t.assert_true("graph probes detach", exec.set_preview(0));
        t.assert_true("reused graph probe admitted", exec.set_preview(prefetch_mode, layers));
        exec.trim();
        exec.set_profile(true);
        exec.bind_preview_graph(req.graph);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        exec.get_profile(profile);
        t.assert_true("reused complete graph restores eligible prefetch", !exec.error() && profile.prefetch_slices == 12 &&
            profile.preview_skip_graph == 0 && profile.preview_skip_shape == 0);
        t.assert_true("short-window probe detaches", exec.set_preview(0));
        exec.trim();
        exec.bind_preview_graph(pruned.graph);
        ggml_backend_graph_compute(cpu.get(), pruned.graph); // warm only mandatory source slices
        t.assert_true("failure probe config admitted", exec.set_preview(prefetch_mode, layers));
        exec.set_profile(true);
        // A later demand expert keeps the warm guard open. Its separate shard
        // is intact, so tail scheduling can reach the final GPU window before
        // the changed optional shard fails validation.
        const int32_t cold_id = 4;
        ggml_backend_tensor_set(req.ids, &cold_id, ggml_nbytes(req.ids) - sizeof(cold_id), sizeof(cold_id));
        fs::resize_file(optional_shard.path, fs::file_size(optional_shard.path) - 1);
        exec.bind_preview_graph(req.graph);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        exec.get_profile(profile);
        ggml_backend_tensor_get(req.output, actual.data(), 0, actual.size() * sizeof(float));
        t.assert_true("speculative read failure occurs after submit", exec.error() != nullptr && profile.submit_ns > 0 && profile.prefetch_ns > 0);
        t.assert_true("failure drains GPU and all leases", exec.stats().store.pinned_entries == 0);
        t.assert_true("failure clears output and keeps cap", std::all_of(actual.begin(), actual.end(), [](float x) { return x == 0; }) &&
            exec.stats().memory.high_water <= total);
    });
    t.test("route observer sees existing tensors and detaches", [&](testing & t) {
        fixture f;
        llama_expert_executor exec(f.source, {f.weight}, device, cap);
        request req(exec, f, cpu.get(), true);
        struct observation {
            const ggml_tensor * input;
            const ggml_tensor * ids;
            int calls = 0;
            bool matches = true;
        } observed {req.input, req.ids};
        ggml_backend_graph_compute(cpu.get(), req.graph);
        float before[24], after[24];
        ggml_backend_tensor_get(req.output, before, 0, sizeof(before));
        exec.set_route_observer([](const llama_expert_route_event & event, void * user) {
            auto & o = *static_cast<observation *>(user);
            ++o.calls;
            o.matches &= event.input == o.input && event.ids == o.ids && event.projection == 1 &&
                std::strcmp(event.weight, "blk.0.ffn_up_exps.weight") == 0;
        }, &observed);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        ggml_backend_tensor_get(req.output, after, 0, sizeof(after));
        t.assert_true("observer binds existing strided input and route IDs", observed.matches && observed.calls == 1);
        t.assert_true("observer leaves all output bytes unchanged", std::memcmp(before, after, sizeof(before)) == 0 && !exec.error());
        exec.set_route_observer(nullptr, nullptr);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        t.assert_equal("detached callback is not invoked", 1, observed.calls);
        t.assert_equal("observer leaves no leases", size_t(0), exec.stats().store.pinned_entries);
        t.assert_true("observer preserves cap", exec.stats().memory.high_water <= cap);
    });
    t.test("route observer failure rejects projection before reads", [&](testing & t) {
        fixture f;
        llama_expert_executor exec(f.source, {f.weight}, device, cap);
        request req(exec, f, cpu.get(), false);
        exec.set_route_observer([](const llama_expert_route_event &, void *) {
            throw std::runtime_error("diagnostic write failed");
        }, nullptr);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        t.assert_true("observer exception becomes explicit executor error", exec.error() && std::strstr(exec.error(), "diagnostic write failed"));
        t.assert_equal("failure precedes expert reads", uint64_t(0), exec.stats().store.misses);
        t.assert_equal("failure holds no leases", size_t(0), exec.stats().store.pinned_entries);
        float actual[24];
        ggml_backend_tensor_get(req.output, actual, 0, sizeof(actual));
        t.assert_true("failed projection is cleared", std::all_of(std::begin(actual), std::end(actual), [](float x) { return x == 0; }));
        exec.set_route_observer(nullptr, nullptr);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        t.assert_true("detaching does not hide failure", exec.error() != nullptr && exec.stats().projections == 0);
    });
    t.test("profiling is opt in and resets counters", [&](testing & t) {
        fixture f;
        llama_expert_executor exec(f.source, {f.weight}, device, cap);
        llama_expert_profile_data profile;
        t.assert_true("disabled by default", !exec.get_profile(profile));
        exec.set_profile(true);
        request req(exec, f, cpu.get(), false);
        t.assert_equal("profiled graph executes", GGML_STATUS_SUCCESS, ggml_backend_graph_compute(cpu.get(), req.graph));
        t.assert_true("profile available", exec.get_profile(profile));
        t.assert_true("nested timing has correct bounds", profile.projection_ns >= profile.acquire_ns &&
            profile.acquire_ns >= profile.read_ns && profile.compute_ns > 0 && profile.allocation_ns > 0 &&
            profile.submit_ns + profile.wait_ns <= profile.compute_ns);
        if (std::string(backend_name) == "MTL0") {
            t.assert_true("completed Metal timing covers submitted graphs", profile.gpu_command_ns > 0 &&
                profile.gpu_timed_submissions == exec.stats().submissions && profile.gpu_command_buffers >= profile.gpu_timed_submissions);
        }
        exec.set_profile(true);
        t.assert_true("reset starts a new interval", exec.get_profile(profile) && profile.projection_ns == 0 && profile.read_ns == 0);
        exec.set_profile(false);
        t.assert_true("can disable again", !exec.get_profile(profile));
    });
    t.test("invalid router ID poisons executor", [&](testing & t) {
        fixture f;
        llama_expert_executor exec(f.source, {f.weight}, device, cap);
        request req(exec, f, cpu.get(), false);
        const int32_t bad = 4;
        ggml_backend_tensor_set(req.ids, &bad, 0, sizeof(bad));
        ggml_backend_graph_compute(cpu.get(), req.graph);
        t.assert_true("deferred error available to context", exec.error() && std::strstr(exec.error(), "invalid expert ID"));
        t.assert_equal("no reads before validation", uint64_t(0), exec.stats().store.misses);
        ggml_backend_tensor_set(req.ids, request::routes, 0, sizeof(request::routes));
        ggml_backend_graph_compute(cpu.get(), req.graph);
        t.assert_true("failure is sticky", exec.error() != nullptr);
        t.assert_equal("no work after failure", uint64_t(0), exec.stats().matmuls);
    });
    t.test("source failure rolls back expert admission", [&](testing & t) {
        fixture f;
        llama_expert_executor exec(f.source, {f.weight}, device, cap);
        request req(exec, f, cpu.get(), false);
        const auto before = exec.stats().memory.used();
        fs::resize_file(f.path, fs::file_size(f.path) - 1);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        t.assert_true("read failure captured", exec.error() != nullptr);
        t.assert_equal("no leaked expert reservations", before, exec.stats().memory.used());
        t.assert_equal("no successful row published", uint64_t(0), exec.stats().matmuls);
    });
    t.test("impossible working set rejected", [&](testing & t) {
        fixture f;
        bool rejected = false;
        try { llama_expert_executor exec(f.source, {f.weight}, device, 1); }
        catch (const std::exception & e) { rejected = std::strstr(e.what(), "minimum working set") != nullptr; }
        t.assert_true("explicit floor error instead of resident fallback", rejected);
    });
    t.test("shared engine budget resize preserves computation", [&](testing & t) {
        fixture f;
        const size_t total = 768 * 1024;
        auto budget = std::make_shared<llama_memory_budget>(total, 0, true);
        auto state = budget->reserve(llama_memory_domain::host, llama_memory_category::state, 128 * 1024);
        state.commit();
        llama_expert_executor exec(f.source, {f.weight}, device, 0, budget);
        request req(exec, f, cpu.get(), false);
        ggml_backend_graph_compute(cpu.get(), req.graph);
        float before[24], after[24];
        ggml_backend_tensor_get(req.output, before, 0, sizeof(before));
        for (int cycle = 0; cycle < 2; ++cycle) {
            const auto full = exec.stats();
            t.assert_true("shrink to computed engine floor", exec.resize(full.minimum_bytes));
            t.assert_true("physical resource charge released", exec.stats().memory.released > full.memory.released);
            t.assert_true("impossible target rejected", !exec.resize(full.minimum_bytes - 1));
            t.assert_equal("rejection preserves applied cap", full.minimum_bytes, exec.stats().memory.applied);
            ggml_backend_graph_compute(cpu.get(), req.graph);
            ggml_backend_tensor_get(req.output, after, 0, sizeof(after));
            t.assert_true("output survives shrink", std::memcmp(before, after, sizeof(before)) == 0 && !exec.error());
            t.assert_true("grow restores capacity", exec.resize(total));
            const auto cold = exec.stats().store.misses;
            ggml_backend_graph_compute(cpu.get(), req.graph);
            t.assert_true("growth warms on demand", exec.stats().store.misses > cold);
            const auto warm = exec.stats().store.hits;
            ggml_backend_graph_compute(cpu.get(), req.graph);
            t.assert_true("warm cache reused", exec.stats().store.hits > warm);
            t.assert_equal("state never reclaimed", size_t(128 * 1024),
                exec.stats().memory.categories[size_t(llama_memory_category::state)].allocated);
        }
    });
    return t.summary();
}
