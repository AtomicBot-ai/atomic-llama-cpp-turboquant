#include "testing.h"
#include "../src/llama-memory-contract.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <future>
#include <stdexcept>

int main(int argc, char ** argv) {
    testing t;
    const char * cuda_name = nullptr;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--cuda-device") == 0) {
            if (++i == argc) { return 1; }
            cuda_name = argv[i];
        } else {
            t.set_filter(argv[i]);
        }
    }
    ggml_backend_load_all();
    ggml_backend_ptr cpu(ggml_backend_init_by_name("CPU", nullptr));
    if (!cpu) { return 1; }
    const auto domain = llama_memory_domain::host;
    t.test("ordinary allocation preserves backend dispatch", [&](testing & t) {
        struct counters { int single = 0, bulk = 0; bool fail = false; } calls;
        auto fake = *ggml_backend_get_default_buffer_type(cpu.get());
        fake.context = &calls;
        fake.iface.alloc_buffer = [](ggml_backend_buffer_type_t buft, size_t size) -> ggml_backend_buffer_t {
            auto & c = *static_cast<counters *>(buft->context);
            ++c.single;
            if (c.fail) { throw std::runtime_error("backend failure"); }
            return ggml_backend_buffer_init(buft, {}, nullptr, size);
        };
        fake.iface.alloc_buffer_n = [](ggml_backend_buffer_type_t buft, ggml_tensor **, int) {
            ++static_cast<counters *>(buft->context)->bulk;
            return ggml_backend_buffer_init(buft, {}, nullptr, 77);
        };
        ggml_context_ptr ctx(ggml_init({4096, nullptr, true}));
        auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);
        auto check_ordinary = [&] {
            const int before = calls.bulk;
            ggml_backend_buffer_ptr b(ggml_backend_buft_alloc_buffer_n(&fake, &tensor, 1));
            t.assert_true("backend custom bulk allocator retained", b && b->size == 77 && calls.bulk == before + 1);
            t.assert_true("ordinary buffer has no admission owner", !b->allocation_reservation && !b->allocation_release);
        };
        check_ordinary();
        auto budget = std::make_shared<llama_memory_budget>(65536, 0, true);
        {
            llama_allocation_scope scope(budget, llama_memory_category::state);
            const int bulk = calls.bulk;
            ggml_backend_buffer_ptr b(ggml_backend_buft_alloc_buffer_n(&fake, &tensor, 1));
            t.assert_true("enabled admission uses accounted allocation", b && b->allocation_reservation && calls.bulk == bulk);
            t.assert_equal("enabled bytes admitted", budget->snapshot(domain).used(), size_t(65536));
            {
                llama_allocation_scope explicit_admission;
                check_ordinary();
            }
            t.assert_true("enclosing admission restored", !ggml_backend_buft_alloc_buffer(&fake, 1));
        }
        t.assert_equal("all admitted bytes released", budget->snapshot(domain).used(), size_t(0));
        check_ordinary();
        std::promise<void> ready, release;
        auto released = release.get_future();
        std::thread other([&] {
            llama_allocation_scope scope(budget, llama_memory_category::state);
            ready.set_value();
            released.wait();
        });
        ready.get_future().wait();
        check_ordinary();
        t.assert_equal("another thread cannot admit these allocations", budget->snapshot(domain).used(), size_t(0));
        release.set_value(); other.join();
        calls.fail = true;
        bool threw = false;
        try { ggml_backend_buft_alloc_buffer(&fake, 1); } catch (const std::runtime_error &) { threw = true; }
        t.assert_true("ordinary backend exceptions unchanged", threw);
        const int single = calls.single;
        ggml_backend_buffer_ptr zero(ggml_backend_buft_alloc_buffer(&fake, 0));
        t.assert_true("zero-sized allocation still bypasses backend", zero && zero->size == 0 && calls.single == single);
    });
    t.test("physical buffer admission lifetime and overlap", [&](testing & t) {
        auto budget = std::make_shared<llama_memory_budget>(2 * 65536, 0, true);
        ggml_backend_buffer_t first = nullptr, second = nullptr;
        {
            llama_allocation_scope scope(budget, llama_memory_category::weights);
            first = ggml_backend_alloc_buffer(cpu.get(), 100);
            second = ggml_backend_alloc_buffer(cpu.get(), 100);
            t.assert_true("two independently owned buffers", first && second);
            t.assert_true("old plus new allocations charged", !ggml_backend_alloc_buffer(cpu.get(), 1));
            t.assert_equal("physical rounding", size_t(2 * 65536), budget->snapshot(domain).used());
            auto * alias = ggml_backend_buffer_get_base(first);
            (void) alias;
            t.assert_equal("alias does not reserve twice", size_t(2 * 65536), budget->snapshot(domain).used());
        }
        std::thread release([&] { ggml_backend_buffer_free(first); });
        release.join();
        t.assert_equal("release outlives scope and thread", size_t(65536), budget->snapshot(domain).used());
        ggml_backend_buffer_free(second);
        t.assert_equal("all bytes returned", size_t(0), budget->snapshot(domain).used());
    });
    t.test("failed backend allocation cancels reservation", [&](testing & t) {
        auto budget = std::make_shared<llama_memory_budget>(65536, 0, true);
        auto fake = *ggml_backend_get_default_buffer_type(cpu.get());
        fake.iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) -> ggml_backend_buffer_t { return nullptr; };
        llama_allocation_scope scope(budget, llama_memory_category::scratch);
        t.assert_true("allocation fails", !ggml_backend_buft_alloc_buffer(&fake, 100));
        t.assert_equal("reservation rolled back", size_t(0), budget->snapshot(domain).used());
        t.assert_equal("not reported as freed payload", size_t(0), budget->snapshot(domain).released);
    });
    t.test("size-aware reclaim receives rounded request without legacy fallback", [&](testing & t) {
        auto budget = std::make_shared<llama_memory_budget>(65536, 0, true);
        struct state {
            ggml_backend_buffer_t old = nullptr;
            size_t bytes = 0;
            llama_memory_domain domain = llama_memory_domain::device;
            int calls = 0;
            bool deny = false;
        } data;
        int legacy_calls = 0;
        const llama_memory_reclaimer reclaim {[](void * user, llama_memory_domain domain, size_t bytes) {
            auto & s = *static_cast<state *>(user);
            s.domain = domain; s.bytes = bytes; ++s.calls;
            if (!s.deny) { ggml_backend_buffer_free(s.old); s.old = nullptr; }
        }, &data};
        llama_allocation_scope scope(budget, llama_memory_category::state, [&] { ++legacy_calls; }, {}, reclaim);
        data.old = ggml_backend_alloc_buffer(cpu.get(), 100);
        t.assert_true("first buffer admitted", data.old != nullptr);
        t.assert_equal("no callback with sufficient capacity", data.calls, 0);
        {
            ggml_backend_buffer_ptr next(ggml_backend_alloc_buffer(cpu.get(), 100));
            t.assert_true("sized reclaim admits replacement", next && !data.old);
            t.assert_equal("one reclaim call", data.calls, 1);
            t.assert_equal("full rounded charge delivered", data.bytes, size_t(65536));
            t.assert_true("correct memory domain", data.domain == domain);
        }
        data.old = ggml_backend_alloc_buffer(cpu.get(), 100);
        data.deny = true;
        ggml_backend_buffer_ptr denied(ggml_backend_alloc_buffer(cpu.get(), 1));
        t.assert_true("refusal leaves allocation denied", !denied && data.old);
        t.assert_equal("no destructive legacy fallback", legacy_calls, 0);
        t.assert_equal("denial keeps original charge", budget->snapshot(domain).used(), size_t(65536));
        ggml_backend_buffer_free(data.old); data.old = nullptr;
        t.assert_equal("all buffers released", budget->snapshot(domain).used(), size_t(0));
        t.assert_equal("bounded high water", budget->snapshot(domain).high_water, size_t(65536));
    });
    t.test("scope nesting and reclaim", [&](testing & t) {
        auto budget = std::make_shared<llama_memory_budget>(65536, 0, true);
        ggml_backend_buffer_t old = nullptr;
        llama_allocation_scope scope(budget, llama_memory_category::state, [&] { ggml_backend_buffer_free(old); old = nullptr; });
        old = ggml_backend_alloc_buffer(cpu.get(), 100);
        {
            llama_allocation_scope already_admitted;
            ggml_backend_buffer_ptr independent(ggml_backend_alloc_buffer(cpu.get(), 100));
            t.assert_equal("explicit admission not counted twice", size_t(65536), budget->snapshot(domain).used());
        }
        ggml_backend_buffer_ptr next(ggml_backend_alloc_buffer(cpu.get(), 100));
        t.assert_true("reclaim admits new buffer", next && !old);
        t.assert_equal("bounded peak", size_t(65536), budget->snapshot(domain).high_water);
    });
    t.test("runtime floor and single context ownership", [&](testing & t) {
        bool rejected = false;
        try { llama_memory_contract too_small(1024); } catch (...) { rejected = true; }
        t.assert_true("runtime admitted before model load", rejected);
        auto contract = std::make_shared<llama_memory_contract>(llama_memory_contract::runtime_envelope + 1024);
        auto lease = contract->claim_context();
        rejected = false;
        try { contract->claim_context(); } catch (...) { rejected = true; }
        t.assert_true("second context rejected", rejected);
        lease.reset();
        auto replacement = contract->claim_context();
        t.assert_true("context ownership returned", bool(replacement));
        t.assert_equal("runtime remains a reservation", llama_memory_contract::runtime_envelope,
            contract->ledger->snapshot(domain).categories[size_t(llama_memory_category::runtime)].reserved);
    });
    t.test("discrete CUDA buffer accounting", [&](testing & t) {
        if (!cuda_name) {
            t.skip("requires --cuda-device");
            return;
        }
        const auto device = ggml_backend_dev_by_name(cuda_name);
        if (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "CUDA") != 0) {
            throw std::runtime_error("discrete CUDA device unavailable");
        }
        const char * unified = std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
        if (unified && std::atoi(unified)) { throw std::runtime_error("CUDA managed memory must be disabled"); }
        const llama_discrete_buffer_types types {ggml_backend_dev_buffer_type(device),
                                                 ggml_backend_dev_host_buffer_type(device)};
        if (!types.device || !types.pinned_host) { throw std::runtime_error("CUDA buffer types unavailable"); }
        constexpr size_t page = 65536;
        constexpr auto vram = llama_memory_domain::device;
        auto budget = std::make_shared<llama_memory_budget>(2 * page, page, false);
        ggml_backend_buffer_ptr host, pinned, weights, state;
        {
            llama_allocation_scope scope(budget, llama_memory_category::weights, {}, types);
            host.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), 4096));
            pinned.reset(ggml_backend_buft_alloc_buffer(types.pinned_host, 4096));
            weights.reset(ggml_backend_buft_alloc_buffer(types.device, 4096));
            if (!host || !pinned || !weights) { throw std::runtime_error("initial buffer allocation failed"); }
            t.assert_true("pinned memory is backed by the requested type", ggml_backend_buffer_get_type(pinned.get()) == types.pinned_host);
            t.assert_true("VRAM is backed by the device type", ggml_backend_buffer_get_type(weights.get()) == types.device);
            t.assert_true("RAM cap rejects more pinned memory", !ggml_backend_buft_alloc_buffer(types.pinned_host, 4096));
            t.assert_true("VRAM cap rejects more device memory", !ggml_backend_buft_alloc_buffer(types.device, 4096));
            auto unknown = *ggml_backend_cpu_buffer_type();
            t.assert_true("unregistered type rejected even with the CPU name", !ggml_backend_buft_alloc_buffer(&unknown, 4096));
        }
        t.assert_equal("only host buffers charge RAM", 2 * page, budget->snapshot(domain).used());
        t.assert_equal("only device buffers charge VRAM", page, budget->snapshot(vram).used());
        weights.reset();
        {
            llama_allocation_scope scope(budget, llama_memory_category::state, {}, types);
            state.reset(ggml_backend_buft_alloc_buffer(types.device, 4096));
            if (!state) { throw std::runtime_error("device state allocation failed"); }
        }
        t.assert_equal("device state is categorized in VRAM", page,
                budget->snapshot(vram).categories[size_t(llama_memory_category::state)].allocated);
        budget->request(vram, 0);
        t.assert_true("device shrink waits for state", budget->snapshot(vram).pending());
        state.reset();
        t.assert_equal("device shrink applies on release", size_t(0), budget->snapshot(vram).applied);
        t.assert_equal("RAM is unaffected by device shrink", 2 * page, budget->snapshot(domain).used());
        pinned.reset();
        host.reset();
        t.assert_equal("all RAM returned", size_t(0), budget->snapshot(domain).used());
        t.assert_equal("all VRAM returned", size_t(0), budget->snapshot(vram).used());
    });
    t.test("discrete loader staging is bounded and reusable", [&](testing & t) {
        const auto device = cuda_name ? ggml_backend_dev_by_name(cuda_name) : ggml_backend_get_device(cpu.get());
        if (!device || (cuda_name && (ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "CUDA") != 0))) {
            throw std::runtime_error("discrete CUDA device unavailable");
        }
        const char * unified = std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
        if (cuda_name && unified && std::atoi(unified)) { throw std::runtime_error("CUDA managed memory must be disabled"); }
        const auto type = ggml_backend_dev_buffer_type(device);
        const llama_discrete_buffer_types types {type, nullptr};
        constexpr size_t chunk = llama_discrete_load_staging::chunk_bytes;
        constexpr size_t bytes = 2 * chunk + 17;
        auto budget = std::make_shared<llama_memory_budget>(chunk - 1, 0, false);
        ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
        if (!ctx) { throw std::runtime_error("test tensor metadata allocation failed"); }
        auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, bytes);
        ggml_backend_buffer_ptr dst(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), type));
        if (!dst) { throw std::runtime_error("test tensor allocation failed"); }
        auto pattern = [](size_t offset) { return uint8_t(offset ^ (offset >> 8) ^ (offset >> 20)); };
        {
            llama_discrete_load_staging staging(budget, types);
            size_t offset = 0;
            std::vector<size_t> reads;
            auto read = [&](void * dst, size_t n) {
                reads.push_back(n);
                auto * data = static_cast<uint8_t *>(dst);
                for (size_t i = 0; i < n; ++i) { data[i] = pattern(offset + i); }
                offset += n;
            };
            bool denied = false;
            try { staging.upload(tensor, read); } catch (const std::runtime_error &) { denied = true; }
            t.assert_true("insufficient RAM is rejected before reading", denied && reads.empty());
            t.assert_equal("failed staging leaves no charge", size_t(0), budget->snapshot(domain).used());
            budget->request(domain, chunk);
            bool interrupted = false;
            try {
                staging.upload(tensor, [](void *, size_t) { throw std::runtime_error("fixture read failure"); });
            } catch (const std::runtime_error &) { interrupted = true; }
            t.assert_true("read error propagates", interrupted);
            t.assert_equal("reusable staging remains charged after a failed read", chunk, budget->snapshot(domain).used());
            staging.upload(tensor, read);
            t.assert_true("large tensor copied with a bounded tail-aware chunk", reads == std::vector<size_t>{chunk, chunk, 17});
            t.assert_equal("staging charged once as RAM I/O", chunk,
                    budget->snapshot(domain).categories[size_t(llama_memory_category::io)].allocated);
            t.assert_equal("RAM peak is one chunk", chunk, budget->snapshot(domain).high_water);
            t.assert_equal("staging incurs no VRAM charge", size_t(0), budget->snapshot(llama_memory_domain::device).used());
        }
        std::vector<uint8_t> actual(bytes);
        ggml_backend_tensor_get(tensor, actual.data(), 0, bytes);
        bool exact = true;
        for (size_t i = 0; i < bytes; ++i) { exact &= actual[i] == pattern(i); }
        t.assert_true("uploaded tensor matches every original byte", exact);
        t.assert_equal("RAM charge released with the staging buffer", size_t(0), budget->snapshot(domain).used());
        t.assert_equal("exactly one staging buffer released", chunk, budget->snapshot(domain).released);
    });
    t.test("CUDA workspace pool admission and retry", [&](testing & t) {
        if (!cuda_name) {
            t.skip("requires --cuda-device");
            return;
        }
        const auto dev = ggml_backend_dev_by_name(cuda_name);
        if (!dev) { throw std::runtime_error("requested CUDA device unavailable"); }
        using enable_pool = bool (*)(ggml_backend_t);
        auto enable = reinterpret_cast<enable_pool>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_enable_observed_pool"));
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        if (!enable || !backend || !enable(backend.get())) { throw std::runtime_error("observed CUDA pool unavailable"); }

        ggml_context_ptr ctx(ggml_init({3 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true}));
        if (!ctx) { throw std::runtime_error("graph metadata allocation failed"); }
        constexpr int width = 256;
        constexpr int rows = 32;
        auto * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q4_0, width, rows);
        auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, width, 1);
        auto * output = ggml_mul_mat(ctx.get(), weight, input);
        auto * graph = ggml_new_graph_custom(ctx.get(), 8, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_ptr tensors(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!tensors) { throw std::runtime_error("graph tensor allocation failed"); }
        ggml_backend_buffer_set_usage(tensors.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        std::vector<float> source(width * rows, 1.0f);
        std::vector<uint8_t> packed(ggml_nbytes(weight));
        if (ggml_quantize_chunk(GGML_TYPE_Q4_0, source.data(), packed.data(), 0, rows, width, nullptr) != packed.size()) {
            throw std::runtime_error("fixture quantization size mismatch");
        }
        ggml_backend_tensor_set(weight, packed.data(), 0, packed.size());
        // Q8_1 activation quantization has scale 1 for this input.
        std::fill(source.begin(), source.end(), 127.0f);
        ggml_backend_tensor_set(input, source.data(), 0, ggml_nbytes(input));

        constexpr auto vram = llama_memory_domain::device;
        const llama_discrete_buffer_types types {ggml_backend_dev_buffer_type(dev), nullptr};
        auto budget = std::make_shared<llama_memory_budget>(0, 0, false);
        t.assert_equal("observed submission requires an active scope", int(GGML_STATUS_ALLOC_FAILED),
                int(ggml_backend_graph_compute(backend.get(), graph)));
        {
            llama_allocation_scope scope(budget, llama_memory_category::scratch, {}, types);
            t.assert_equal("workspace denial propagates as an allocation failure", int(GGML_STATUS_ALLOC_FAILED),
                    int(ggml_backend_graph_compute(backend.get(), graph)));
        }
        t.assert_equal("denial leaves no workspace charge", size_t(0), budget->snapshot(vram).used());
        t.assert_equal("denied allocation did not raise the peak", size_t(0), budget->snapshot(vram).high_water);
        budget->request(vram, 1024 * 1024);
        {
            llama_allocation_scope scope(budget, llama_memory_category::scratch, {}, types);
            t.assert_equal("same graph runs after budget growth", int(GGML_STATUS_SUCCESS),
                    int(ggml_backend_graph_compute(backend.get(), graph)));
        }
        const auto stats = budget->snapshot(vram);
        t.assert_true("physical workspace contributed to the VRAM peak", stats.high_water > 0);
        t.assert_equal("temporary workspace no longer occupies the ledger", size_t(0), stats.used());
        t.assert_true("physical workspace charge was released", stats.released >= stats.high_water);
        t.assert_equal("CUDA pool does not charge RAM", size_t(0), budget->snapshot(domain).high_water);
        float actual[rows] {};
        ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
        t.assert_true("quantized projection matches the reference",
                std::all_of(actual, actual + rows, [](float value) { return value == width * 127.0f; }));
    });
    t.test("cuBLAS workspace admission and lifetime", [&](testing & t) {
        if (!cuda_name) {
            t.skip("requires --cuda-device");
            return;
        }
        const auto dev = ggml_backend_dev_by_name(cuda_name);
        if (!dev) { throw std::runtime_error("requested CUDA device unavailable"); }
        using enable_pool = bool (*)(ggml_backend_t);
        auto enable = reinterpret_cast<enable_pool>(ggml_backend_reg_get_proc_address(
                ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_enable_observed_pool"));
        ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
        if (!enable || !backend || !enable(backend.get())) { throw std::runtime_error("observed CUDA pool unavailable"); }
        ggml_context_ptr ctx(ggml_init({3 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true}));
        if (!ctx) { throw std::runtime_error("graph metadata allocation failed"); }
        auto * a = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 2);
        auto * b = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 4, 2);
        auto * output = ggml_out_prod(ctx.get(), a, b); // CUDA dispatches this op through cuBLAS.
        auto * graph = ggml_new_graph_custom(ctx.get(), 8, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_ptr tensors(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!tensors) { throw std::runtime_error("graph tensor allocation failed"); }
        const float left[] = {1, 2, 3, 4, 5, 6, 7, 8};
        const float right[] = {8, 6, 4, 2, 1, 3, 5, 7};
        ggml_backend_tensor_set(a, left, 0, sizeof(left));
        ggml_backend_tensor_set(b, right, 0, sizeof(right));

        constexpr auto vram = llama_memory_domain::device;
        constexpr size_t cap = 32 * 1024 * 1024;
        const llama_discrete_buffer_types types {ggml_backend_dev_buffer_type(dev), nullptr};
        auto budget = std::make_shared<llama_memory_budget>(0, 0, false);
        auto compute = [&] {
            llama_allocation_scope scope(budget, llama_memory_category::scratch, {}, types);
            return int(ggml_backend_graph_compute(backend.get(), graph));
        };
        t.assert_equal("workspace admission fails at zero VRAM", int(GGML_STATUS_ALLOC_FAILED), compute());
        t.assert_equal("denied workspace leaves no charges", size_t(0), budget->snapshot(vram).used());
        budget->request(vram, cap);
        t.assert_equal("same backend can retry after VRAM growth", int(GGML_STATUS_SUCCESS), compute());
        const auto used = budget->snapshot(vram);
        t.assert_true("persistent workspace uses the expected CUDA allocation size",
                used.used() == 4 * 1024 * 1024 || used.used() == cap);
        t.assert_equal("workspace is classified as VRAM scratch", used.used(),
                used.categories[size_t(llama_memory_category::scratch)].allocated);
        budget->request(vram, used.used());
        t.assert_equal("a repeat reuses the live workspace at its exact cap", int(GGML_STATUS_SUCCESS), compute());
        t.assert_equal("repeat does not add VRAM", used.used(), budget->snapshot(vram).high_water);
        float actual[16] {};
        ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
        bool exact = true;
        for (int col = 0; col < 4; ++col) {
            for (int row = 0; row < 4; ++row) {
                exact &= actual[col * 4 + row] == left[row] * right[col] + left[row + 4] * right[col + 4];
            }
        }
        t.assert_true("cuBLAS output matches the exact reference", exact);
        budget->request(vram, 0);
        t.assert_true("shrink waits while the backend owns the workspace", budget->snapshot(vram).pending());
        tensors.reset();
        backend.reset();
        const auto after = budget->snapshot(vram);
        t.assert_equal("backend teardown releases the full workspace", used.used(), after.released);
        t.assert_equal("VRAM shrink applies after teardown", size_t(0), after.applied);
        t.assert_equal("no lingering VRAM charge", size_t(0), after.used());
    });
    return t.summary();
}
