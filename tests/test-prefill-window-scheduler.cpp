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
            ggml_backend_sched_ptr scheduler(compact ? ggml_backend_window_new(&backend, nullptr, 1, 8192, false) :
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
            ggml_backend_sched_ptr scheduler(compact ? ggml_backend_window_new(backends, nullptr, 1, 8192, false) :
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
    return t.summary();
}
