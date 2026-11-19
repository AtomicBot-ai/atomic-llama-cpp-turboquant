#pragma once
#include "../src/llama-expert-publication.h"
#include "ggml-cpp.h"
#include "../src/llama-expert-executor.h"
#include <filesystem>
#include <algorithm>
#include <thread>

static void test_expert_publication(testing & t, ggml_backend_dev_t device) {
    t.test("Metal route publication ordering reuse cancellation and failure", [&](testing & t) {
        ggml_backend_ptr gpu(ggml_backend_dev_init(device, nullptr));
        const auto * api = static_cast<const ggml_metal_route_api *>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(device), "ggml_backend_metal_route_publication_v1"));
        t.assert_true("private capability present", api != nullptr);
        if (!api) return;
        early_io::pool pool;
        llama_expert_publication publication(*api, gpu.get(), pool);
        ggml_context_ptr ctx(ggml_init({32768, nullptr, true}));
        auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 8);
        auto * copy = ggml_cont(ctx.get(), input);
        struct state {
            llama_expert_publication * pub;
            std::array<int32_t, 8> result{};
            bool fail = false;
            static void take(void * user, const int32_t * ids, size_t n) {
                auto & s = *static_cast<state *>(user);
                if (s.fail) throw std::runtime_error("injected read failure");
                std::copy_n(ids, n, s.result.begin());
            }
            static void arm(void * user, uint64_t seq, uint64_t gen, size_t n) {
                auto & s = *static_cast<state *>(user);
                s.pub->start(seq, gen, n, take, user);
            }
        } state{&publication};
        t.assert_true("route marker attached", publication.attach(copy, state::arm, &state, 17));
        t.assert_true("non-copy marker rejected", !publication.attach(input, state::arm, &state, 17));
        auto * graph = ggml_new_graph_custom(ctx.get(), 8, false);
        ggml_build_forward_expand(graph, copy);
        ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), gpu.get()));
        for (int pass = 0; pass < 8; ++pass) {
            std::array<int32_t, 8> expected{};
            for (size_t i = 0; i < expected.size(); ++i) expected[i] = pass * 11 + int(i);
            ggml_backend_tensor_set(input, expected.data(), 0, sizeof(expected));
            t.assert_equal("publish graph", ggml_backend_graph_compute(gpu.get(), graph), GGML_STATUS_SUCCESS);
            t.assert_true("publication armed", publication.pending());
            publication.join();
            t.assert_true("payload exact after event", expected == state.result);
            t.assert_equal("generation preserved", publication.stamp(), uint64_t(17));
        }
        std::array<int32_t, 8> stale{};
        t.assert_true("retired event cannot read a newer payload", api->read(publication.handle(), 1, stale.data(), stale.size()) == -1);
        state.fail = true;
        ggml_backend_graph_compute(gpu.get(), graph);
        bool caught = false;
        try { publication.join(); } catch (const std::runtime_error &) { caught = true; }
        t.assert_true("worker error propagated", caught && !publication.pending());
        // No command buffer will ever signal this value. Cancellation must drain
        // the bounded waiter, without releasing a GPU-owned buffer or deadlocking.
        publication.start(UINT64_MAX, 18, 8, state::take, &state);
        publication.cancel();
        t.assert_true("unsignaled request cancelled", !publication.pending());
        state.fail = false;
        ggml_backend_graph_compute(gpu.get(), graph);
        publication.join();
        t.assert_true("channel reusable after cancellation", !publication.pending());
    });
}

static void test_expert_publication_pipeline(testing & t, ggml_backend_dev_t device) {
    t.test("published demand mixed quantization partial groups and lifecycle", [&](testing & t) {
        struct env_guard {
            const char * key; std::string old; bool present;
            env_guard(const char * key, const char * value) : key(key), present(std::getenv(key) != nullptr) {
                if (present) old = std::getenv(key);
                if (value) setenv(key, value, 1); else unsetenv(key);
            }
            ~env_guard() { if (present) setenv(key, old.c_str(), 1); else unsetenv(key); }
        } streaming("LLAMA_MOE_STREAMING", "auto"), predictor("ATOMIC_MOE_QWEN_PREDICTOR", nullptr);
        constexpr int width = 256, experts = 8, routes = 7;
        constexpr size_t cap = 64 * 1024 * 1024;
        const auto path = std::filesystem::temp_directory_path() / ("published-demand-" + std::to_string(early_io::now()) + ".gguf");
        struct cleanup { std::filesystem::path p; ~cleanup() { std::error_code ec; std::filesystem::remove(p, ec); } } cleanup{path};
        ggml_context_ptr weights(ggml_init({8 * 1024 * 1024, nullptr, false}));
        gguf_context_ptr meta(gguf_init_empty());
        gguf_set_val_str(meta.get(), "general.architecture", "qwen35moe");
        std::array<ggml_tensor *, 3> w;
        const ggml_type types[] = {GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};
        const char * names[] = {"blk.0.ffn_up_exps.weight", "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"};
        for (size_t k = 0; k < 3; ++k) {
            w[k] = ggml_new_tensor_3d(weights.get(), types[k], width, width, experts);
            ggml_set_name(w[k], names[k]);
            std::vector<float> data(width * width * experts);
            for (size_t j = 0; j < data.size(); ++j) data[j] = float(int((j * 7 + k * 11) % 31) - 15) / 256;
            ggml_quantize_chunk(types[k], data.data(), w[k]->data, 0, width * experts, width, nullptr);
            gguf_add_tensor(meta.get(), w[k]);
        }
        t.assert_true("fixture written", gguf_write_to_file(meta.get(), path.string().c_str(), false));
        auto source = std::make_shared<llama_expert_source>(std::vector<std::string>{path.string()});
        std::array<std::vector<float>, 2> expected;
        for (const char * mode : {"off", "event", "predictor", "shared"}) {
            env_guard feature("ATOMIC_MOE_PUBLISH", mode);
            env_guard prediction("ATOMIC_MOE_QWEN_PREDICTOR", !std::strcmp(mode, "predictor") ? "1" : nullptr);
            llama_expert_executor exec(source, {w[0], w[1], w[2]}, device, cap);
            exec.configure_acceleration(); exec.set_profile(true);
            t.assert_equal("new Qwen predictor has four candidates only when requested", exec.stats().prefetch_candidates,
                    size_t(!std::strcmp(mode, "predictor") ? 4 : 0));
            ggml_backend_ptr gpu(ggml_backend_dev_init(device, nullptr));
            ggml_backend_ptr cpu(ggml_backend_init_by_name("CPU", nullptr));
            ggml_context_ptr router_ctx(ggml_init({16384, nullptr, true}));
            auto * router = ggml_new_tensor_2d(router_ctx.get(), GGML_TYPE_F32, width, experts);
            ggml_backend_buffer_ptr router_buffer(ggml_backend_alloc_ctx_tensors(router_ctx.get(), gpu.get()));
            std::vector<float> router_data(width * experts, 0.01f);
            ggml_backend_tensor_set(router, router_data.data(), 0, router_data.size() * sizeof(float));
            for (int rows : {2, 1}) {
                exec.trim(); // each shape starts cold, then repeats hot
                ggml_backend_t backends[] = {gpu.get(), cpu.get()};
                ggml_backend_sched_ptr sched(ggml_backend_sched_new(backends, nullptr, 2, 128, false, true));
                ggml_context_ptr ctx(ggml_init({128 * 1024, nullptr, true}));
                auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, width, 1, rows);
                auto * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, routes, rows);
                ggml_set_input(x); ggml_set_input(ids);
                ggml_backend_sched_set_tensor_backend(sched.get(), x, gpu.get());
                ggml_backend_sched_set_tensor_backend(sched.get(), ids, gpu.get());
                auto * out = exec.build_moe(ctx.get(), w[0], w[1], w[2], x, ids, router, {w[0], w[1], w[2]}, sched.get());
                t.assert_true("combined graph built", out != nullptr); if (!out) return;
                ggml_backend_sched_set_tensor_backend(sched.get(), out, cpu.get());
                auto * graph = ggml_new_graph_custom(ctx.get(), 128, false);
                const bool published = exec.moe_route_published(out->src[1]);
                t.assert_equal("only one-row opt-in publishes", published, rows == 1 && (!std::strcmp(mode, "event") || !std::strcmp(mode, "shared")));
                t.assert_equal("Qwen prediction restricted to one row", out->src[2] != nullptr,
                        rows == 1 && !std::strcmp(mode, "predictor"));
                if (published) ggml_build_forward_expand(graph, out->src[1]);
                auto * shared = ggml_scale(ctx.get(), x, 0.25f);
                ggml_backend_sched_set_tensor_backend(sched.get(), shared, gpu.get());
                if (published && !std::strcmp(mode, "shared")) ggml_build_forward_expand(graph, shared);
                auto * result = ggml_add(ctx.get(), out, shared);
                ggml_set_output(result); ggml_build_forward_expand(graph, result);
                t.assert_true("graph allocated", ggml_backend_sched_alloc_graph(sched.get(), graph));
                std::vector<float> input(width * rows), actual(width * routes * rows);
                std::vector<int32_t> selected(routes * rows);
                for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 23) - 11) / 16;
                for (size_t i = 0; i < selected.size(); ++i) selected[i] = i % experts;
                ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
                ggml_backend_tensor_set(ids, selected.data(), 0, selected.size() * sizeof(int32_t));
                for (int repeat = 0; repeat < 2; ++repeat) {
                    // Scheduler input storage may be reused after its last consumer.
                    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
                    ggml_backend_tensor_set(ids, selected.data(), 0, selected.size() * sizeof(int32_t));
                    exec.moe_lookahead_begin(true);
                    t.assert_equal("scheduler executes", ggml_backend_sched_graph_compute(sched.get(), graph), GGML_STATUS_SUCCESS);
                    exec.moe_lookahead_end();
                    if (exec.error()) std::fprintf(stderr, "publication mode=%s rows=%d repeat=%d error=%s\n", mode, rows, repeat, exec.error());
                    t.assert_true("executor healthy", exec.error() == nullptr);
                    ggml_backend_tensor_get(result, actual.data(), 0, actual.size() * sizeof(float));
                    if (!std::strcmp(mode, "off") && !repeat) expected[rows - 1] = actual;
                    t.assert_true("all routed outputs and shared sum bit exact", expected[rows - 1].size() == actual.size() &&
                            !std::memcmp(expected[rows - 1].data(), actual.data(), actual.size() * sizeof(float)));
                    t.assert_equal("GPU leases released", exec.stats().store.pinned_entries, size_t(0));
                    t.assert_true("ledger stays within cap", exec.stats().memory.high_water <= cap);
                }
                if (rows == 1 && !std::strcmp(mode, "shared")) {
                    // Invalidate storage after hot execution. A failed published
                    // read must drain without submitting invalid expert weights.
                    exec.trim(); std::filesystem::resize_file(path, 0);
                    exec.moe_lookahead_begin(true);
                    // Do not drain between these evaluations: emulate later
                    // layer publications after the first layer latches an error.
                    for (int extra = 0; extra < 3; ++extra) {
                        ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
                        ggml_backend_tensor_set(ids, selected.data(), 0, selected.size() * sizeof(int32_t));
                        ggml_backend_sched_graph_compute(sched.get(), graph);
                    }
                    exec.moe_lookahead_end();
                    if (exec.error()) std::fprintf(stderr, "published read failure: %s\n", exec.error());
                    t.assert_true("read failure latched", exec.error() != nullptr);
                    t.assert_equal("failed leases drained", exec.stats().store.pinned_entries, size_t(0));
                }
            }
            llama_expert_profile_data profile{}; exec.get_profile(profile);
            t.assert_equal("only one-row graph published", profile.route_publications, uint64_t(!std::strcmp(mode, "shared") ? 3 : !std::strcmp(mode, "event") ? 2 : 0));
            if (!exec.error()) {
                const auto floor = exec.stats().minimum_bytes;
                t.assert_true("resize to minimum", exec.resize(floor));
                t.assert_true("optional publication falls back", !exec.combined_moe_enabled());
                t.assert_true("restore original budget", exec.resize(cap));
                t.assert_true("combined path restored", exec.combined_moe_enabled());
            }
        }
    });
}

// The real Metal test covers payload ordering. This controlled producer covers
// a publication that remains pending long enough to cancel deterministically.
static void test_expert_publication_service(testing & t) {
    t.test("route service delayed publication cancellation and stale error", [&](testing & t) {
        struct event {
            std::atomic<unsigned> polls{0};
            std::atomic<bool> ready{false}, stale{false};
            static void * create(ggml_backend_t) { return new event; }
            static void destroy(void * p) { delete static_cast<event *>(p); }
            static int read(void * p, uint64_t, int32_t * ids, size_t n) {
                auto & e = *static_cast<event *>(p); ++e.polls;
                if (e.stale.load()) return -1;
                if (e.ready.load()) { std::fill_n(ids, n, 7); return 1; }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                return 0;
            }
        };
        const ggml_metal_route_api api{sizeof(event), event::create, event::destroy, nullptr, event::read};
        early_io::pool readers;
        llama_expert_publication service(api, nullptr, readers);
        auto & e = *static_cast<event *>(service.handle());
        struct consumer { unsigned calls = 0; int32_t id = -1; } received;
        auto take = [](void * p, const int32_t * ids, size_t) {
            auto & c = *static_cast<consumer *>(p); ++c.calls; c.id = ids[0];
        };
        auto observed = [&] {
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
            while (e.polls.load() < 3 && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
            t.assert_true("producer remained pending", e.polls.load() >= 3);
        };
        service.start(1, 11, 1, take, &received);
        observed(); e.ready = true; service.join();
        t.assert_true("delayed payload delivered once", received.calls == 1 && received.id == 7);
        e.ready = false; e.polls = 0;
        service.start(2, 11, 1, take, &received);
        observed();
        t.assert_true("try-take falls back while the event is pending", !service.take_if_ready());
        service.cancel();
        t.assert_true("pending consumer cancelled and drained", !service.pending() && received.calls == 1);
        e.stale = true;
        service.start(3, 12, 1, take, &received);
        bool rejected = false;
        try { service.join(); } catch (const std::runtime_error &) { rejected = true; }
        t.assert_true("stale producer fails without consumption", rejected && received.calls == 1 && !service.pending());
        e.stale = false; e.ready = true;
        service.start(4, 13, 1, take, &received); service.join();
        t.assert_true("new generation works after cancellation and error", received.calls == 2 && service.stamp() == 13);
    });
}
