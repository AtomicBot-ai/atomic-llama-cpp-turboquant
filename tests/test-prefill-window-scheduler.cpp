// Only the compact scheduler contract and the adjacent ordinary allocation path.
#include "testing.h"
#include "../ggml/src/ggml-backend-window.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "ggml-cpp.h"
#include <array>
#include <cstring>
#include <vector>

static std::vector<ggml_tensor *> dependency_leaves;

int main(int argc, char ** argv) {
    testing t;
    for (int i = 1; i < argc; ++i) { t.set_filter(argv[i]); }
    ggml_backend_ptr cpu(ggml_backend_cpu_init());
    auto * backend = cpu.get();
    t.test("compact scheduler exact result and graph reuse", [&](testing & t) {
        std::array<float, 32> reference {};
        for (bool compact : {false, true}) {
            ggml_backend_sched_ptr scheduler(compact ? ggml_backend_window_new_compact(&backend, nullptr, 1, 8192, false) :
                    ggml_backend_sched_new(&backend, nullptr, 1, 8192, false, false));
            t.assert_true(bool(scheduler));
            for (int pass = 0; pass < 2; ++pass) {
                ggml_backend_sched_reset(scheduler.get());
                ggml_context_ptr ctx(ggml_init({2 * 1024 * 1024, nullptr, true}));
                auto * graph = ggml_new_graph_custom(ctx.get(), 8192, false);
                auto * input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 32);
                ggml_set_input(input);
                auto * output = ggml_scale(ctx.get(), input, 1.5f);
                ggml_build_forward_expand(graph, output);
                t.assert_true(compact ? ggml_backend_window_alloc(scheduler.get(), graph, 8192) :
                        ggml_backend_sched_alloc_graph(scheduler.get(), graph));
                std::array<float, 32> values;
                for (size_t i = 0; i < values.size(); ++i) { values[i] = float(i) - 15.25f; }
                ggml_backend_tensor_set(input, values.data(), 0, sizeof(values));
                t.assert_true(ggml_backend_sched_graph_compute(scheduler.get(), graph) == GGML_STATUS_SUCCESS);
                ggml_backend_tensor_get(output, values.data(), 0, sizeof(values));
                if (!compact && pass == 0) { reference = values; }
                t.assert_true(std::memcmp(values.data(), reference.data(), sizeof(values)) == 0);
                // Reset while graph metadata is still alive, as in the runtime.
                ggml_backend_sched_reset(scheduler.get());
            }
        }
    });
    t.test("compact scheduler rejects oversized graph before allocation", [&](testing & t) {
        ggml_backend_sched_ptr scheduler(ggml_backend_window_new(&backend, nullptr, 1, 8192, false));
        t.assert_true(bool(scheduler));
        ggml_context_ptr ctx(ggml_init({16 * 1024 * 1024, nullptr, true}));
        auto * graph = ggml_new_graph_custom(ctx.get(), 16384, false);
        auto * value = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
        for (int i = 0; i < 8193; ++i) {
            value = ggml_scale(ctx.get(), value, 0.5f);
            ggml_build_forward_expand(graph, value);
        }
        t.assert_true(!ggml_backend_window_alloc(scheduler.get(), graph, 8192));
        t.assert_true(!ggml_backend_window_new(&backend, nullptr, 1, 16384, false));
    });
    t.test("compact scheduler rejects backend dependency expansion", [&](testing & t) {
        // A backend may retain many input tensors until each output node.
        // Original graph size alone must not admit the expanded allocator graph.
        auto dependent = *backend;
        dependent.iface.graph_optimize = [](ggml_backend_t, ggml_cgraph * graph,
                ggml_backend_graph_optimize_params * params) {
            for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
                for (auto * leaf : dependency_leaves) {
                    params->add_alloc_dep(params->user_data, leaf, ggml_graph_node(graph, i));
                }
            }
        };
        ggml_backend_t backends[] = {&dependent};
        for (bool compact : {false, true}) {
            ggml_backend_sched_ptr scheduler(compact ? ggml_backend_window_new_compact(backends, nullptr, 1, 8192, false) :
                    ggml_backend_sched_new(backends, nullptr, 1, 8192, false, false));
            ggml_context_ptr ctx(ggml_init({16 * 1024 * 1024, nullptr, true}));
            auto * graph = ggml_new_graph_custom(ctx.get(), 8192, false);
            dependency_leaves.clear();
            for (int i = 0; i < 8000; ++i) {
                auto * leaf = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 1);
                ggml_backend_sched_set_tensor_backend(scheduler.get(), leaf, &dependent);
                dependency_leaves.push_back(leaf);
                ggml_build_forward_expand(graph, leaf);
            }
            auto * value = dependency_leaves.front();
            for (int i = 0; i < 64; ++i) {
                value = ggml_scale(ctx.get(), value, 0.5f);
                ggml_build_forward_expand(graph, value);
            }
            const bool admitted = compact ? ggml_backend_window_alloc(scheduler.get(), graph, 8192) :
                    ggml_backend_sched_alloc_graph(scheduler.get(), graph);
            t.assert_true(admitted == !compact);
            ggml_backend_sched_reset(scheduler.get());
        }
        dependency_leaves.clear();
    });
    t.test("window scratch is shared and values survive a layer pause", [&](testing & t) {
        const auto buft = ggml_backend_get_default_buffer_type(backend);
        ggml_backend_buffer_ptr scratch(ggml_backend_buft_alloc_buffer(buft, 64 * 1024));
        t.assert_true(bool(scratch));

        std::array<ggml_context_ptr, 2> contexts;
        std::array<ggml_backend_sched_ptr, 2> schedulers;
        std::array<ggml_cgraph *, 2> graphs;
        std::array<ggml_tensor *, 2> outputs;
        for (size_t i = 0; i < contexts.size(); ++i) {
            contexts[i].reset(ggml_init({2 * 1024 * 1024, nullptr, true}));
            schedulers[i].reset(ggml_backend_window_new_compact(&backend, nullptr, 1, 8192, false));
            t.assert_true(ggml_backend_window_share_scratch(schedulers[i].get(), scratch.get()));
            auto * ctx = contexts[i].get();
            auto * input = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 32);
            ggml_set_input(input);
            auto * temporary = ggml_scale(ctx, input, float(i + 2));
            auto * keep = ggml_scale(ctx, temporary, 3.0f);
            ggml_set_name(keep, "l_last-0");
            outputs[i] = ggml_scale(ctx, keep, 5.0f);
            ggml_set_output(outputs[i]);
            graphs[i] = ggml_new_graph_custom(ctx, 8192, false);
            ggml_build_forward_expand(graphs[i], outputs[i]);
            t.assert_true(ggml_backend_window_alloc(schedulers[i].get(), graphs[i], 8192));
            t.assert_true(temporary->buffer == scratch.get());
            t.assert_true(keep->buffer != scratch.get() && outputs[i]->buffer != scratch.get());
            std::array<float, 32> values;
            values.fill(float(i + 1));
            ggml_backend_tensor_set(input, values.data(), 0, sizeof(values));
        }

        struct pause {
            ggml_backend_sched_t scheduler;
            ggml_cgraph * graph;
            ggml_status status = GGML_STATUS_FAILED;
        } nested{schedulers[1].get(), graphs[1]};
        ggml_backend_sched_set_eval_callback(schedulers[0].get(), [](ggml_tensor * node, bool ask, void * user) {
            if (ask) { return std::strcmp(node->name, "l_last-0") == 0; }
            auto & state = *static_cast<pause *>(user);
            state.status = ggml_backend_sched_graph_compute(state.scheduler, state.graph);
            return state.status == GGML_STATUS_SUCCESS;
        }, &nested);
        t.assert_true(ggml_backend_sched_graph_compute(schedulers[0].get(), graphs[0]) == GGML_STATUS_SUCCESS);
        t.assert_true(nested.status == GGML_STATUS_SUCCESS);
        for (size_t i = 0; i < outputs.size(); ++i) {
            std::array<float, 32> values;
            ggml_backend_tensor_get(outputs[i], values.data(), 0, sizeof(values));
            for (float value : values) { t.assert_true(value == float((i + 1) * (i + 2) * 15)); }
        }
    });
    return t.summary();
}
