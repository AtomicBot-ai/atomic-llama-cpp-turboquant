#include "testing.h"
#include "../src/llama-expert-store.h"
#include "../src/llama-expert-lookahead.h"
#include "../ggml/src/ggml-backend-impl.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <thread>

namespace fs = std::filesystem;
using domain = llama_memory_domain;
using category = llama_memory_category;

template <typename F>
static void rejects(testing & t, const char * label, F fn) {
    bool rejected = false;
    try { fn(); } catch (const std::exception &) { rejected = true; }
    t.assert_true(label, rejected);
}

struct fixture {
    fs::path path;
    static constexpr const char * name = "blk.0.ffn_up_exps.weight";
    static constexpr size_t expert_bytes = 32 * 4 * sizeof(float);

    static float weight(int expert, int row, int col) { return (expert + 1) * 10.0f + row - col * 0.125f; }

    fixture(bool mixed_layout = false, bool larger_alternate = false, const char * architecture = "qwen3moe", int experts = 4, bool gate = false) {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        path = fs::temp_directory_path() / ("test-expert-store-" + std::to_string(now) + ".gguf");
        ggml_context_ptr ctx(ggml_init({size_t(16384 + experts * 2048), nullptr, false}));
        gguf_context_ptr meta(gguf_init_empty());
        gguf_set_val_str(meta.get(), "general.architecture", architecture);
        auto * tensor = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 32, 4, experts);
        ggml_set_name(tensor, name);
        auto * values = static_cast<float *>(tensor->data);
        for (int e = 0; e < experts; ++e) {
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 32; ++c) { values[e * 128 + r * 32 + c] = weight(e, r, c); }
            }
        }
        gguf_add_tensor(meta.get(), tensor);
        if (gate) {
            auto * gate_tensor = ggml_dup_tensor(ctx.get(), tensor);
            ggml_set_name(gate_tensor, "blk.0.ffn_gate_exps.weight");
            std::memcpy(gate_tensor->data, tensor->data, ggml_nbytes(tensor));
            gguf_add_tensor(meta.get(), gate_tensor);
        }
        if (mixed_layout) {
            const int rows = larger_alternate ? 16 : 8;
            auto * alternate = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F16, 32, rows, experts);
            ggml_set_name(alternate, "blk.0.ffn_down_exps.weight");
            auto * values = static_cast<ggml_fp16_t *>(alternate->data);
            for (int i = 0; i < 32 * rows * experts; ++i) { values[i] = ggml_fp32_to_fp16(float(i % 128) / 8); }
            gguf_add_tensor(meta.get(), alternate);
        }
        if (!gguf_write_to_file(meta.get(), path.string().c_str(), false)) {
            throw std::runtime_error("failed to write expert store fixture");
        }
    }
    ~fixture() { std::error_code ec; fs::remove(path, ec); }
    std::shared_ptr<llama_expert_source> source() const {
        return std::make_shared<llama_expert_source>(std::vector<std::string>{path.string()});
    }
};

// Controlled backend completion: prove that a lease and its charge survive while
// synchronization is blocked, without relying on timing or real GPU race luck.
struct completion_gate {
    std::mutex mutex;
    std::condition_variable cv;
    bool entered = false;
    bool done = false;
};
static completion_gate * gate = nullptr;
static void blocked_completion(ggml_backend_t) {
    std::unique_lock<std::mutex> lock(gate->mutex);
    gate->entered = true;
    gate->cv.notify_all();
    gate->cv.wait(lock, [] { return gate->done; });
}

int main(int argc, char ** argv) {
    testing t;
    const char * backend_name = "CPU";
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--backend" && i + 1 < argc) { backend_name = argv[++i]; }
        else { t.set_filter(argv[i]); }
    }
    ggml_backend_load_all();
    ggml_backend_ptr backend(ggml_backend_init_by_name(backend_name, nullptr));
    if (!backend) { std::fprintf(stderr, "backend unavailable: %s\n", backend_name); return 1; }

    t.test("budget ownership and domains", [&](testing & t) {
        llama_memory_budget budget(100, 50, false);
        auto host = budget.reserve(domain::host, category::state, 80);
        auto device = budget.reserve(domain::device, category::experts, 40);
        t.assert_true("domain reservations", host && device);
        t.assert_true("host denial", !budget.reserve(domain::host, category::io, 21));
        t.assert_true("device denial", !budget.reserve(domain::device, category::io, 11));
        host.commit(); device.commit();
        budget.request(domain::host, 20);
        t.assert_true("shrink pending", budget.snapshot(domain::host).pending());
        host.retire();
        auto moved = std::move(host);
        t.assert_true("single owner", !host && moved);
        t.assert_equal("retirement remains charged", size_t(80), budget.snapshot(domain::host).used());
        t.assert_equal("no early release", size_t(0), budget.snapshot(domain::host).released);
        moved.reset();
        auto state = budget.snapshot(domain::host);
        t.assert_equal("applied after release", size_t(20), state.applied);
        t.assert_equal("completed release", size_t(80), state.released);
        t.assert_equal("device unaffected", size_t(40), budget.snapshot(domain::device).used());
        llama_memory_budget uma(100, 0, true);
        auto shared = uma.reserve(domain::device, category::experts, 80);
        shared.commit();
        t.assert_equal("UMA alias shares host accounting", size_t(80), uma.snapshot(domain::host).used());
        t.assert_true("UMA host admission", !uma.reserve(domain::host, category::io, 21));
        auto cancelled = uma.reserve(domain::host, category::io, 20);
        cancelled.reset();
        t.assert_equal("cancellation is not freed allocation", size_t(0), uma.snapshot(domain::host).released);
        rejects(t, "double commit", [&] { shared.commit(); });
    });
    t.test("budget arithmetic and contention", [&](testing & t) {
        llama_memory_budget huge(std::numeric_limits<size_t>::max(), 0, true);
        auto almost_all = huge.reserve(domain::host, category::experts, std::numeric_limits<size_t>::max() - 2);
        t.assert_true("no overflow admission", !huge.reserve(domain::host, category::io, 3));
        auto tail = huge.reserve(domain::device, category::io, 2);
        t.assert_equal("maximum charge", std::numeric_limits<size_t>::max(), huge.snapshot(domain::host).used());
        auto budget = std::make_shared<llama_memory_budget>(64, 0, true);
        std::vector<llama_memory_budget::ticket> tickets(8);
        std::vector<std::thread> workers;
        for (size_t i = 0; i < tickets.size(); ++i) {
            workers.emplace_back([&, i] { tickets[i] = budget->reserve(domain::host, category::io, 16); });
        }
        for (auto & worker : workers) { worker.join(); }
        size_t accepted = 0;
        for (auto & ticket : tickets) { accepted += bool(ticket); }
        t.assert_equal("admission serialized", size_t(4), accepted);
        t.assert_equal("bounded peak", size_t(64), budget->snapshot(domain::host).high_water);
        tickets.clear();
        t.assert_equal("reservation rollback", size_t(0), budget->snapshot(domain::host).used());
    });
    t.test("selective reclaim preserves pinned and unrelated experts", [&](testing & t) {
        fixture f;
        const size_t resident = fixture::expert_bytes + ggml_tensor_overhead();
        const size_t cap = 4 * resident + fixture::expert_bytes;
        auto budget = std::make_shared<llama_memory_budget>(cap, 0, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64);
        auto pinned = store.acquire(fixture::name, 0);
        const auto generation = pinned.generation();
        for (int expert : {1, 2, 3}) { store.acquire(fixture::name, expert); }
        const auto initial = store.stats();
        const size_t free = cap - budget->snapshot(domain::host).used();
        t.assert_true("wrong domain does not reclaim", !store.reclaim_to_fit(domain::device, cap));
        t.assert_true("existing free space retained", store.reclaim_to_fit(domain::host, free));
        t.assert_equal("no needless eviction", store.stats().evictions, initial.evictions);
        t.assert_true("impossible reservation does not flush", !store.reclaim_to_fit(domain::host, cap + 1));
        t.assert_equal("impossible request kept entries", store.stats().entries, size_t(4));
        t.assert_true("shortfall reclaimed", store.reclaim_to_fit(domain::host, free + 1));
        t.assert_equal("one victim suffices", store.stats().reclaim_evictions, uint64_t(1));
        t.assert_true("oldest unpinned victim", !store.contains(fixture::name, 1));
        t.assert_true("unrelated hot entries retained", store.contains(fixture::name, 2) && store.contains(fixture::name, 3));
        t.assert_true("pinned entry not retired", store.contains(fixture::name, 0));
        t.assert_equal("pinned generation unchanged", pinned.generation(), generation);
        auto reservation = budget->reserve(domain::host, category::state, free + 1);
        t.assert_true("new buffer can be admitted", bool(reservation));
        reservation.reset();
        const size_t lower = budget->snapshot(domain::host).used() - 1;
        budget->request(domain::host, lower);
        t.assert_true("lower target initially pending", budget->snapshot(domain::host).pending());
        t.assert_true("reclaim honors requested target", store.reclaim_to_fit(domain::host, 1));
        t.assert_equal("second victim only", store.stats().reclaim_evictions, uint64_t(2));
        t.assert_true("target applied after release", !budget->snapshot(domain::host).pending());
        budget->request(domain::host, cap);
        auto last = store.acquire(fixture::name, 3);
        const auto before = store.stats();
        t.assert_true("fully pinned cache refuses", !store.reclaim_to_fit(domain::host, cap));
        t.assert_equal("pinned cache not evicted", store.stats().entries, before.entries);
        t.assert_true("pinned entries not retired on refusal", store.contains(fixture::name, 0) && store.contains(fixture::name, 3));
        t.assert_equal("selective reclaim performs no reads", store.stats().logical_read_bytes, initial.logical_read_bytes);
        t.assert_equal("freed bytes match two victims", store.stats().reclaim_bytes, uint64_t(2 * resident));
        t.assert_true("ledger within cap", budget->snapshot(domain::host).high_water <= cap);
        last.reset(); pinned.reset();
        store.trim();
        t.assert_equal("lifecycle trim still clears all", store.stats().entries, size_t(0));
        t.assert_equal("all physical charges released", budget->snapshot(domain::host).used(), size_t(0));
    });
    t.test("cache leases eviction and regrowth", [&](testing & t) {
        fixture f;
        const size_t resident = fixture::expert_bytes + ggml_tensor_overhead();
        const size_t cap = 2 * resident + fixture::expert_bytes;
        auto budget = std::make_shared<llama_memory_budget>(cap, 0, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64);
        auto a = store.acquire(fixture::name, 0);
        auto alias = store.acquire(fixture::name, 0);
        t.assert_true("same physical tensor for aliases", a.tensor() == alias.tensor());
        auto b = store.acquire(fixture::name, 1);
        rejects(t, "pinned working set cannot spill beyond cap", [&] { store.acquire(fixture::name, 2); });
        const auto first_generation = a.generation();
        a.reset(); alias.reset();
        auto c = store.acquire(fixture::name, 2);
        t.assert_equal("one eviction", uint64_t(1), store.stats().evictions);
        t.assert_true("generation advances", c.generation() > first_generation);
        t.assert_true("peak within cap", budget->snapshot(domain::host).high_water <= cap);
        b.reset(); c.reset();
        auto shrunk = store.request_budget(domain::host, 0);
        t.assert_equal("physical buffers returned", size_t(0), shrunk.used());
        t.assert_equal("applied zero", size_t(0), shrunk.applied);
        store.request_budget(domain::host, cap);
        auto again = store.acquire(fixture::name, 0);
        t.assert_true("new generation after grow", again.generation() > first_generation);
        std::vector<float> bytes(128);
        ggml_backend_tensor_get(again.tensor(), bytes.data(), 0, fixture::expert_bytes);
        t.assert_equal("original values after reread", fixture::weight(0, 3, 31), bytes.back());
    });
    t.test("LRU order follows acquisition across pinned releases and hits", [&](testing & t) {
        fixture f;
        const size_t resident = fixture::expert_bytes + ggml_tensor_overhead();
        const size_t cap = 3 * resident + fixture::expert_bytes;
        auto budget = std::make_shared<llama_memory_budget>(cap, 0, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64);
        auto oldest = store.acquire(fixture::name, 0);
        auto middle = store.acquire(fixture::name, 1);
        auto newest = store.acquire(fixture::name, 2);
        newest.reset();
        middle.reset();
        store.acquire(fixture::name, 3).reset();
        t.assert_true("oldest pinned entry survives", store.contains(fixture::name, 0));
        t.assert_true("oldest unpinned entry evicted", !store.contains(fixture::name, 1));
        oldest.reset();
        store.acquire(fixture::name, 1).reset();
        t.assert_true("late release does not promote recency", !store.contains(fixture::name, 0));
        store.acquire(fixture::name, 2).reset();
        store.acquire(fixture::name, 0).reset();
        t.assert_true("hit promotes existing entry", store.contains(fixture::name, 2));
        t.assert_true("hit leaves correct oldest victim", !store.contains(fixture::name, 3));
        store.trim();
        t.assert_equal("trim empties queue", store.stats().entries, size_t(0));
        store.acquire(fixture::name, 3).reset();
        t.assert_true("publication after empty queue", store.contains(fixture::name, 3));
        store.request_budget(domain::host, 0);
        t.assert_equal("shrink releases all charges", budget->snapshot(domain::host).used(), size_t(0));
    });
    t.test("recycled allocation preserves pins layout and publication", [&](testing & t) {
        fixture f(true);
        const size_t cap = ggml_tensor_overhead() + 2 * fixture::expert_bytes;
        auto budget = std::make_shared<llama_memory_budget>(cap, 0, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64);
        auto first = store.acquire(fixture::name, 0);
        const auto generation = first.generation();
        const auto * allocation = first.tensor()->data;
        const char * alternate = "blk.0.ffn_down_exps.weight";
        rejects(t, "cannot recycle pinned allocation", [&] { store.acquire(alternate, 1); });
        first.reset();
        auto next = store.acquire(alternate, 1);
        t.assert_true("same physical allocation", next.tensor()->data == allocation);
        t.assert_true("new generation and tensor layout", next.generation() > generation &&
            next.tensor()->type == GGML_TYPE_F16 && next.tensor()->ne[1] == 8);
        std::vector<ggml_fp16_t> values(256);
        ggml_backend_tensor_get(next.tensor(), values.data(), 0, fixture::expert_bytes);
        bool exact = true;
        for (int i = 0; i < 256; ++i) { exact &= values[i] == ggml_fp32_to_fp16(float(i % 128) / 8); }
        t.assert_true("original packed values replace previous expert", exact);
        t.assert_equal("one physical allocation", store.stats().buffer_allocations, uint64_t(1));
        t.assert_equal("one completed reuse", store.stats().buffer_reuses, uint64_t(1));
        next.reset();
        fs::resize_file(f.path, fs::file_size(f.path) - 1);
        rejects(t, "failed reread never publishes recycled buffer", [&] { store.acquire(fixture::name, 2); });
        t.assert_equal("no partial cache entry", store.stats().entries, size_t(0));
        t.assert_equal("failed reuse removes resident accounting", store.resident_bytes(), size_t(0));
        t.assert_equal("failed reuse returns physical charges", budget->snapshot(domain::host).used(), size_t(0));
        t.assert_true("transient staging remains in budget", budget->snapshot(domain::host).high_water <= cap);
    });

    t.test("oversized reuse retains physical accounting and clears padding", [&](testing & t) {
        fixture f(true, true);
        const size_t cap = ggml_tensor_overhead() + 4 * fixture::expert_bytes;
        auto budget = std::make_shared<llama_memory_budget>(cap, 0, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64);
        llama_expert_trace_event access {};
        store.set_trace([](const llama_expert_trace_event & e, void * user) noexcept {
            if (e.kind == llama_expert_trace_kind::access) { *static_cast<llama_expert_trace_event *>(user) = e; }
        }, &access);
        auto first = store.acquire("blk.0.ffn_down_exps.weight", 0);
        const auto * allocation = first.tensor()->data;
        const size_t charged = budget->snapshot(domain::host).used();
        rejects(t, "pinned large allocation cannot be reused", [&] { store.acquire(fixture::name, 1); });
        first.reset();
        auto next = store.acquire(fixture::name, 1);
        t.assert_true("larger physical buffer reused", next.tensor()->data == allocation && store.stats().buffer_reuses == 1);
        t.assert_equal("resident accounting retains oversized allocation", store.resident_bytes(), charged);
        t.assert_equal("stats retain oversized allocation", store.stats().resident_bytes, charged);
        t.assert_equal("trace capacity excludes resident physical bytes", access.capacity, cap);
        t.assert_equal("excess capacity remains charged", budget->snapshot(domain::host).used(), charged);
        t.assert_true("trace separates payload and required/physical charges", access.payload == fixture::expert_bytes &&
            access.allocation == charged && access.required == ggml_tensor_overhead() + fixture::expert_bytes);
        std::vector<float> values(128);
        ggml_backend_tensor_get(next.tensor(), values.data(), 0, fixture::expert_bytes);
        bool exact = true;
        for (int i = 0; i < 128; ++i) { exact &= values[i] == fixture::weight(1, i / 32, i % 32); }
        t.assert_true("smaller expert bytes preserved", exact);
        // Read the old allocation's tail through a temporary larger tensor view.
        auto view = *next.tensor();
        view.ne[1] = 8; view.nb[2] = view.nb[3] = 2 * fixture::expert_bytes;
        std::vector<uint8_t> tail(fixture::expert_bytes, 1);
        ggml_backend_tensor_get(&view, tail.data(), fixture::expert_bytes, tail.size());
        t.assert_true("old payload beyond new shape cleared", std::all_of(tail.begin(), tail.end(), [](uint8_t v) { return v == 0; }));
        next.reset();
        store.request_budget(domain::host, 0);
        t.assert_equal("idle shrink clears resident accounting", store.resident_bytes(), size_t(0));
        t.assert_equal("shrink returns entire physical allocation", budget->snapshot(domain::host).used(), size_t(0));
        t.assert_true("all reservations stayed under cap", budget->snapshot(domain::host).high_water <= cap);
    });

#ifdef __APPLE__
    t.test("demand queue delivers only validated triples and drains failed reads", [&](testing & t) {
        if (std::string(backend_name) != "CPU") { t.skip("CPU buffer admission fixture"); return; }
        fixture f(true, false, "qwen4exp", 8, true);
        auto source = f.source();
        auto budget = std::make_shared<llama_memory_budget>(64 * 1024, 0, true);
        llama_expert_store store(source, budget, backend.get(), domain::host, 64);
        const std::array<const char *, 3> names{fixture::name, "blk.0.ffn_gate_exps.weight", "blk.0.ffn_down_exps.weight"};
        const int64_t ids[] = {0, 2, 3, 5, 6, 7};
        auto batch = store.enqueue_demand(names, ids, std::size(ids));
        t.assert_true("unclaimed misses are hidden", !store.contains(names[0], 0));
        rejects(t, "cannot acquire an unpublished tensor", [&] { store.acquire(names[0], 0); });
        size_t seen = 0, delivered = 0;
        while (batch->pending()) {
            std::array<int64_t, 16> claimed{};
            std::array<llama_expert_store::lease, 16> up, gate, down;
            const size_t size = batch->take(claimed.data(), up.data(), gate.data(), down.data(), 4);
            t.assert_true("bounded ready group", size >= std::min<size_t>(4, std::size(ids) - delivered));
            delivered += size;
            for (size_t j = 0; j < size; ++j) {
                const size_t bit = size_t(1) << claimed[j];
                t.assert_true("expert delivered once", !(seen & bit));
                seen |= bit;
                float u = 0, g = 0;
                ggml_fp16_t d{};
                ggml_backend_tensor_get(up[j].tensor(), &u, 0, sizeof(u));
                ggml_backend_tensor_get(gate[j].tensor(), &g, 0, sizeof(g));
                ggml_backend_tensor_get(down[j].tensor(), &d, 0, sizeof(d));
                t.assert_true("all three slices complete", u == fixture::weight(claimed[j], 0, 0) && g == u && ggml_fp16_to_fp32(d) == 0);
            }
        }
        size_t expected = 0;
        for (auto id : ids) { expected |= size_t(1) << id; }
        t.assert_equal("entire request delivered", seen, expected);
        batch.reset();
        t.assert_true("published expert reusable", store.contains(names[0], 0));
        store.trim();
        fs::resize_file(f.path, 0);
        batch = store.enqueue_demand(names, ids, std::size(ids));
        std::array<int64_t, 16> claimed{};
        std::array<llama_expert_store::lease, 16> up, gate, down;
        rejects(t, "bad source never produces a triple", [&] {
            batch->take(claimed.data(), up.data(), gate.data(), down.data(), 4);
        });
        batch.reset();
        t.assert_equal("failed wave releases all physical charges", budget->snapshot(domain::host).used(), size_t(0));
        t.assert_equal("failed wave leaves no cache entries", store.stats().entries, size_t(0));
        t.assert_true("physical ledger stays bounded", budget->snapshot(domain::host).high_water <= 64 * 1024);
    });
#endif

#ifdef __APPLE__
    auto check_lookahead = [&](testing & t, bool per_slice) {
        if (std::string(backend_name) != "CPU") { t.skip("CPU buffer admission fixture"); return; }
        fixture f(true, false, "qwen4exp", 4, true);
        auto source = f.source();
        const auto & up = source->tensor(fixture::name);
        const auto & gate = source->tensor("blk.0.ffn_gate_exps.weight");
        const auto & down = source->tensor("blk.0.ffn_down_exps.weight");
        const size_t cap = 64 * 1024, bytes = llama_expert_lookahead::working_bytes(fixture::expert_bytes);
        auto budget = std::make_shared<llama_memory_budget>(cap, 0, true);
        llama_expert_store store(source, budget, backend.get(), domain::host, 64);
        llama_expert_profile_data profile{};
        store.set_profile(&profile);
        std::array<ggml_bf16_t, 32 * 4> router{};
        std::array<float, 32> row{};
        row[0] = 1;
        for (size_t e = 0; e < 4; ++e) { router[32 * e] = ggml_fp32_to_bf16(float(e)); }
        const llama_expert_lookahead::target target{router.data(), 32, 4, {&up, &gate, &down}, GGML_TYPE_BF16};
        ggml_tensor input{};
        input.data = row.data();
        input.ne[0] = 32;
        input.ne[2] = 1;
        input.nb[2] = sizeof(row);
        {
            auto ticket = budget->reserve(domain::host, category::io, bytes);
            t.assert_true("scratch fits", bool(ticket));
            llama_expert_lookahead ahead(source, store, std::move(ticket), fixture::expert_bytes, per_slice);
            llama_expert_lookahead::prediction proposal{1, 4, {1, 2, 0, 3}, {4, 3, 2, 1}};
            const auto * predicted = per_slice ? &proposal : nullptr;
            ahead.begin_graph(false);
            ahead.fetch(target, &input, &profile, predicted);
            ahead.end_graph(&profile);
            t.assert_equal("prompt graph performs no speculative read", profile.next_prefetch_bytes, uint64_t(0));

            ahead.begin_graph(true);
            ahead.fetch(target, &input, &profile, predicted);
            t.assert_equal("prefetch publishes no cache entry", store.stats().entries, size_t(0));
            t.assert_equal("all scratch still charged", budget->snapshot(domain::host).used(), bytes);
            const int64_t demand[] = {per_slice ? 1 : 3, 0};
            std::array<llama_expert_store::lease, 2> leases;
            store.acquire_many(fixture::name, demand, 2, leases.data(), &ahead);
            for (size_t i = 0; i < leases.size(); ++i) {
                float first = 0;
                ggml_backend_tensor_get(leases[i].tensor(), &first, 0, sizeof(first));
                t.assert_equal("read the requested expert, regardless of prediction", first, fixture::weight(demand[i], 0, 0));
            }
            ahead.end_graph(&profile);
            t.assert_equal("at most two complete predicted triples read", profile.next_prefetch_bytes,
                uint64_t(2 * (up.expert_bytes() + gate.expert_bytes() + down.expert_bytes())));
            t.assert_equal("matching slice consumed once", profile.next_prefetch_used_bytes, uint64_t(up.expert_bytes()));
            t.assert_equal("mismatch alone reads from source on demand", store.stats().logical_read_bytes, uint64_t(up.expert_bytes()));
            t.assert_true("hidden duration bounded by service time", profile.next_prefetch_hidden_ns <= profile.next_prefetch_read_ns);

            leases = {};
            store.trim();
            const int64_t next = 2;
            store.acquire_many(fixture::name, &next, 1, leases.data(), &ahead);
            t.assert_equal("graph boundary forgets unused speculation", profile.next_prefetch_used_bytes, uint64_t(up.expert_bytes()));
            leases = {};
            store.trim();
            fs::resize_file(f.path, 0);
            ahead.begin_graph(true);
            ahead.fetch(target, &input, &profile, predicted);
            ahead.end_graph(&profile);
            t.assert_equal("failed speculative job drained", profile.next_prefetch_failures, uint64_t(per_slice ? 6 : 1));
            t.assert_equal("failed speculative read publishes nothing", store.stats().entries, size_t(0));
            rejects(t, "actual demand still validates failed source", [&] { store.acquire(fixture::name, 3); });
        }
        t.assert_equal("scratch returned after drain", budget->snapshot(domain::host).used(), size_t(0));
        t.assert_true("full budget respected", budget->snapshot(domain::host).high_water <= cap);
    };
    t.test("bounded expert lookahead copies exact misses and drains unused or failed reads",
           [&](testing & t) { check_lookahead(t, false); });
    t.test("independent expert slices copy predicted hits and isolate failed jobs",
           [&](testing & t) { check_lookahead(t, true); });
#endif

    t.test("size-class recycling preserves eviction order bytes and failed-read ownership", [&](testing & t) {
        if (std::string(backend_name) != "CPU") { t.skip("CPU physical allocation fixture"); return; }
        fixture f(true, true, "qwen4exp", 8);
        const size_t small = fixture::expert_bytes, arena = ggml_tensor_overhead();
        const size_t cap = 3 * (arena + small) + arena + 4 * small;
        auto budget = std::make_shared<llama_memory_budget>(cap, 0, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64, nullptr, true, true);
        auto seed = [&]() {
            store.acquire("blk.0.ffn_down_exps.weight", 0).reset();
            for (int i = 0; i < 3; ++i) { store.acquire(fixture::name, i).reset(); }
            store.acquire(fixture::name, 3).reset();
        };
        seed();
        auto stats = store.stats();
        t.assert_true("original LRU victims removed", !store.contains("blk.0.ffn_down_exps.weight", 0) &&
            !store.contains(fixture::name, 0) && store.contains(fixture::name, 1));
        t.assert_equal("wrong-sized victim retained once", stats.pooled_buffers, size_t(1));
        t.assert_equal("pool includes buffer and arena", stats.pooled_bytes, arena + 2 * small);
        t.assert_equal("all retained memory charged", budget->snapshot(domain::host).used(), stats.resident_bytes + stats.pooled_bytes);
        t.assert_equal("working capacity includes reclaimable spares", store.storage_bytes(), stats.resident_bytes + stats.pooled_bytes);
        const auto allocations = stats.buffer_allocations;
        auto large = store.acquire("blk.0.ffn_down_exps.weight", 1);
        t.assert_equal("size-class hit avoids allocation", store.stats().buffer_allocations, allocations);
        t.assert_equal("size-class hit recorded", store.stats().pool_reuses, uint64_t(1));
        t.assert_equal("pool ownership transferred", store.stats().pooled_buffers, size_t(0));
        std::vector<ggml_fp16_t> values(512);
        ggml_backend_tensor_get(large.tensor(), values.data(), 0, 2 * small);
        bool exact = true;
        for (int i = 0; i < 512; ++i) { exact &= values[i] == ggml_fp32_to_fp16(float(i % 128) / 8); }
        t.assert_true("pooled buffer rebound to requested weights", exact);
        store.request_budget(domain::host, 0);
        t.assert_equal("pinned recycled weight remains charged", budget->snapshot(domain::host).used(), arena + 2 * small);
        t.assert_equal("pinned weight not pooled", store.stats().pooled_buffers, size_t(0));
        large.reset();
        t.assert_equal("retired recycled weight freed", budget->snapshot(domain::host).used(), size_t(0));
        store.request_budget(domain::host, cap);
        seed();
        fs::resize_file(f.path, 0);
        rejects(t, "failed pooled read cannot publish", [&] { store.acquire("blk.0.ffn_down_exps.weight", 1); });
        t.assert_true("failed pooled weight absent", !store.contains("blk.0.ffn_down_exps.weight", 1));
        t.assert_equal("failed pool owner destroyed", store.stats().pooled_bytes, size_t(0));
        t.assert_equal("only unaffected residents stay charged", budget->snapshot(domain::host).used(), 3 * (arena + small));
        t.assert_true("pool plus staging stayed within cap", budget->snapshot(domain::host).high_water <= cap);
    });

    t.test("size-class pool bounds reclamation resize and trace fallback", [&](testing & t) {
        if (std::string(backend_name) != "CPU") { t.skip("CPU physical allocation fixture"); return; }
        fixture f(true, true, "qwen4exp", 72);
        const size_t small = fixture::expert_bytes, arena = ggml_tensor_overhead();
        const size_t cap = 65 * (arena + 2 * small) + 4 * (arena + small) + 2 * small;
        auto budget = std::make_shared<llama_memory_budget>(cap, 0, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64, nullptr, true, true);
        auto seed = [&]() {
            store.trim();
            for (int i = 0; i < 65; ++i) { store.acquire("blk.0.ffn_down_exps.weight", i).reset(); }
            for (int i = 0; i < 5; ++i) { store.acquire(fixture::name, i).reset(); }
            t.assert_equal("at most 64 buffers per size", store.stats().pooled_buffers, size_t(64));
        };
        seed();
        auto before = store.stats();
        const size_t free = cap - budget->snapshot(domain::host).used();
        t.assert_true("selective reclamation succeeds", store.reclaim_to_fit(domain::host, free + 1));
        auto after = store.stats();
        t.assert_equal("reclamation frees one spare", after.pooled_buffers, size_t(63));
        t.assert_equal("useful residents preserved", after.resident_bytes, before.resident_bytes);
        t.assert_equal("reclaim counters include physical spare", after.reclaim_bytes - before.reclaim_bytes, arena + 2 * small);
        store.request_budget(domain::host, after.resident_bytes);
        t.assert_equal("resize releases remaining spare", store.stats().pooled_bytes, size_t(0));
        t.assert_equal("resize retains fitting residents", store.resident_bytes(), after.resident_bytes);
        store.request_budget(domain::host, cap);
        seed();
        store.set_trace([](const llama_expert_trace_event &, void *) noexcept {}, nullptr);
        t.assert_equal("tracing drains unused spares", store.stats().pooled_buffers, size_t(0));
        store.trim();
        for (int i = 0; i < 65; ++i) { store.acquire("blk.0.ffn_down_exps.weight", i).reset(); }
        for (int i = 0; i < 5; ++i) { store.acquire(fixture::name, i).reset(); }
        t.assert_equal("tracing does not populate pool", store.stats().pooled_buffers, size_t(0));
        store.set_trace(nullptr, nullptr);
        seed();
        store.evict_idle();
        t.assert_equal("idle eviction returns pool and resident charges", budget->snapshot(domain::host).used(), size_t(0));
        t.assert_true("all physical charges bounded", budget->snapshot(domain::host).high_water <= cap);
    });

    t.test("compact buffers release oversized victims and retain exact-size reuse", [&](testing & t) {
        if (std::string(backend_name) != "CPU") { t.skip("CPU physical allocation fixture"); return; }
        fixture f(true, true);
        const size_t small = fixture::expert_bytes;
        const size_t arena = ggml_tensor_overhead();
        const size_t cap = arena + 4 * small;
        auto budget = std::make_shared<llama_memory_budget>(cap, 0, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64, nullptr, true);
        auto large = store.acquire("blk.0.ffn_down_exps.weight", 0);
        t.assert_equal("large slice estimate matches physical charge",
                store.compact_slice_bytes("blk.0.ffn_down_exps.weight"), store.resident_bytes());
        rejects(t, "pinned oversized victim remains protected", [&] { store.acquire(fixture::name, 0); });
        large.reset();
        auto next = store.acquire(fixture::name, 0);
        t.assert_equal("only smaller physical buffer charged", store.resident_bytes(), arena + small);
        t.assert_equal("small slice estimate matches physical charge", store.compact_slice_bytes(fixture::name), store.resident_bytes());
        t.assert_equal("oversized allocation replaced", store.stats().buffer_allocations, uint64_t(2));
        t.assert_equal("oversized allocation never reused", store.stats().buffer_reuses, uint64_t(0));
        const auto * allocation = next.tensor()->data;
        next.reset();
        store.request_budget(domain::host, arena + 2 * small);
        auto same = store.acquire(fixture::name, 1);
        t.assert_true("same size can reuse policy victim", same.tensor()->data == allocation && store.stats().buffer_reuses == 1);
        std::vector<float> values(128);
        ggml_backend_tensor_get(same.tensor(), values.data(), 0, small);
        bool exact = true;
        for (int i = 0; i < 128; ++i) { exact &= values[i] == fixture::weight(1, i / 32, i % 32); }
        t.assert_true("rebound weights preserved", exact);
        store.request_budget(domain::host, 0);
        t.assert_equal("retired lease stays charged", budget->snapshot(domain::host).used(), arena + small);
        same.reset();
        t.assert_equal("last release frees compact allocation", budget->snapshot(domain::host).used(), size_t(0));
        store.request_budget(domain::host, arena + 2 * small);
        store.acquire(fixture::name, 2).reset();
        fs::resize_file(f.path, 0);
        rejects(t, "failed compact reuse cannot publish", [&] { store.acquire(fixture::name, 3); });
        t.assert_true("failed buffer absent", !store.contains(fixture::name, 3));
        t.assert_equal("failed buffer uncharged", budget->snapshot(domain::host).used(), size_t(0));
        t.assert_true("physical and staging charges bounded", budget->snapshot(domain::host).high_water <= cap);
    });

    t.test("uniform slots reuse small and large experts within physical cap", [&](testing & t) {
        fixture f(true, true);
        const size_t slot = 2 * fixture::expert_bytes;
        const size_t resident = ggml_tensor_overhead() + slot;
        const size_t cap = resident + slot;
        auto budget = std::make_shared<llama_memory_budget>(cap, 0, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, slot);
        t.assert_equal("noncompact slots retain conservative admission", store.compact_slice_bytes(fixture::name), size_t(0));
        auto small = store.acquire(fixture::name, 0);
        const auto * allocation = small.tensor()->data;
        t.assert_equal("resident accounting includes padding and arena", store.resident_bytes(), resident);
        t.assert_equal("small payload charges entire slot", budget->snapshot(domain::host).used(), resident);
        rejects(t, "pinned slot cannot be rebound", [&] { store.acquire("blk.0.ffn_down_exps.weight", 0); });
        small.reset();
        auto large = store.acquire("blk.0.ffn_down_exps.weight", 0);
        t.assert_true("larger payload reuses small expert slot", large.tensor()->data == allocation);
        std::vector<ggml_fp16_t> values(512);
        ggml_backend_tensor_get(large.tensor(), values.data(), 0, slot);
        bool exact = true;
        for (int i = 0; i < 512; ++i) { exact &= values[i] == ggml_fp32_to_fp16(float(i % 128) / 8); }
        t.assert_true("larger expert bytes preserved", exact);
        large.reset();
        small = store.acquire(fixture::name, 1);
        t.assert_true("same slot reused in both directions", small.tensor()->data == allocation &&
            store.stats().buffer_allocations == 1 && store.stats().buffer_reuses == 2);
        auto view = *small.tensor();
        view.ne[1] = 8; view.nb[2] = view.nb[3] = slot;
        std::vector<uint8_t> tail(fixture::expert_bytes, 1);
        ggml_backend_tensor_get(&view, tail.data(), fixture::expert_bytes, tail.size());
        t.assert_true("old larger payload cleared beyond small shape", std::all_of(tail.begin(), tail.end(), [](uint8_t v) { return v == 0; }));
        store.request_budget(domain::host, 0);
        t.assert_equal("retiring slot remains resident", store.resident_bytes(), resident);
        t.assert_equal("pinned retired slot stays fully charged", budget->snapshot(domain::host).used(), resident);
        small.reset();
        t.assert_equal("last release clears resident accounting", store.resident_bytes(), size_t(0));
        t.assert_equal("last release returns full physical slot", budget->snapshot(domain::host).used(), size_t(0));
        store.request_budget(domain::host, cap);
        store.acquire(fixture::name, 2).reset();
        t.assert_equal("grow restores resident accounting", store.resident_bytes(), resident);
        store.trim();
        t.assert_equal("trim clears resident accounting", store.resident_bytes(), size_t(0));
        store.acquire(fixture::name, 2).reset();
        t.assert_equal("grow admits a fully charged slot", budget->snapshot(domain::host).used(), resident);
        t.assert_true("staging and padding stay under cap", budget->snapshot(domain::host).high_water <= cap);
    });

    t.test("Metal grouped experts preserve independent retirement and ownership", [&](testing & t) {
        if (std::string(backend_name) != "MTL0") { t.skip("Metal-only residency contract"); return; }
        fixture f(false, false, "qwen4exp", 130);
        const size_t slot = 16384;
        const size_t resident = slot + ggml_tensor_overhead();
        const size_t cap = 130 * resident + fixture::expert_bytes;
        auto budget = std::make_shared<llama_memory_budget>(cap, 0, true);
        auto store = std::make_unique<llama_expert_store>(f.source(), budget, backend.get(), domain::host, slot);
        const auto ordinary = ggml_backend_get_default_buffer_type(backend.get());
        std::vector<llama_expert_store::lease> leases;
        for (int i = 0; i < 130; ++i) { leases.push_back(store->acquire(fixture::name, i)); }
        t.assert_equal("qualified experts use grouped residency", std::string("Metal_Expert_Grouped"),
            std::string(ggml_backend_buft_name(ggml_backend_buffer_get_type(leases[0].tensor()->buffer))));
        t.assert_equal("three groups retain independently charged slots", budget->snapshot(domain::host).used(), 130 * resident);
        t.assert_true("ordinary allocator unchanged", ordinary == ggml_backend_get_default_buffer_type(backend.get()));
        auto ordinary_buffer = ggml_backend_buffer_ptr(ggml_backend_buft_alloc_buffer(ordinary, slot));
        t.assert_true("ordinary buffer still uses ordinary type", ordinary_buffer && ggml_backend_buffer_get_type(ordinary_buffer.get()) == ordinary);
        // Retire all groups, then empty the middle group while its neighbors
        // remain pinned. Refill holes after grow, and let leases outlive the store.
        store->request_budget(domain::host, 0);
        for (int i = 64; i < 128; ++i) { leases[i].reset(); }
        t.assert_equal("empty middle group releases only its own slots", budget->snapshot(domain::host).used(), 66 * resident);
        for (int i = 1; i < 64; ++i) { leases[i].reset(); }
        t.assert_equal("one pinned neighbor keeps only one slot charged", budget->snapshot(domain::host).used(), 3 * resident);
        store->request_budget(domain::host, cap);
        for (int i = 1; i < 128; ++i) { leases[i] = store->acquire(fixture::name, i); }
        t.assert_equal("grow fills partial and new groups", store->resident_bytes(), 130 * resident);
        store.reset();
        float value = 0;
        ggml_backend_tensor_get(leases[129].tensor(), &value, 0, sizeof(value));
        t.assert_equal("lease remains valid after store destruction", value, fixture::weight(129, 0, 0));
        leases.clear();
        t.assert_equal("final releases destroy all groups and charges", budget->snapshot(domain::host).used(), size_t(0));
        t.assert_true("full lifecycle stays under cap", budget->snapshot(domain::host).high_water <= cap);
    });

    t.test("Qwen4Exp reused slots preserve the complete zero tail", [&](testing & t) {
        for (const char * architecture : {"qwen4exp", "qwen3moe"}) {
            fixture f(true, true, architecture);
            const size_t slot = 65536;
            const size_t resident = ggml_tensor_overhead() + slot;
            const size_t cap = resident + 2 * fixture::expert_bytes;
            auto budget = std::make_shared<llama_memory_budget>(cap, 0, true);
            llama_expert_profile_data profile;
            llama_expert_store store(f.source(), budget, backend.get(), domain::host, slot);
            store.set_profile(&profile);
            const bool optimized = std::string(backend_name) == "MTL0" && std::string(architecture) == "qwen4exp";
            uint64_t expected_cleared = 0;
            const void * allocation = nullptr;
            for (int step = 0; step < 5; ++step) {
                const bool large = step == 2;
                const size_t payload = fixture::expert_bytes * (large ? 2 : 1);
                auto lease = store.acquire(large ? "blk.0.ffn_down_exps.weight" : fixture::name, step % 4);
                if (!step) { allocation = lease.tensor()->data; }
                t.assert_true("all shapes reuse one physical slot", lease.tensor()->data == allocation);
                auto view = *lease.tensor();
                view.ne[1] = slot / view.nb[1]; view.nb[2] = view.nb[3] = slot;
                std::vector<uint8_t> bytes(slot, 1);
                ggml_backend_tensor_get(&view, bytes.data(), 0, slot);
                t.assert_true("complete padding remains zero after grow shrink and same-size reuse",
                    std::all_of(bytes.begin() + payload, bytes.end(), [](uint8_t v) { return v == 0; }));
                std::vector<uint8_t> original(payload);
                f.source()->read_expert(large ? "blk.0.ffn_down_exps.weight" : fixture::name, step % 4,
                    original.data(), original.size());
                t.assert_true("payload remains the original GGUF bytes", std::equal(original.begin(), original.end(), bytes.begin()));
                expected_cleared += !optimized || !step ? slot - payload : step == 3 ? fixture::expert_bytes : 0;
                t.assert_equal("only qualified reuse skips already-zero bytes", profile.padding_bytes, expected_cleared);
                t.assert_equal("whole slot remains charged", budget->snapshot(domain::host).used(), resident);
            }
            t.assert_equal("one physical allocation across size changes", store.stats().buffer_allocations, uint64_t(1));
            fs::resize_file(f.path, fs::file_size(f.path) - 1);
            rejects(t, "failed reused read cannot publish", [&] { store.acquire(fixture::name, 3); });
            t.assert_equal("failed slot is destroyed", store.stats().entries, size_t(0));
            t.assert_equal("failed slot returns all charges", budget->snapshot(domain::host).used(), size_t(0));
            t.assert_true("padding and staging stay bounded", budget->snapshot(domain::host).high_water <= cap);
        }
    });

    t.test("demand trace records allocation and miss staging", [&](testing & t) {
        fixture f;
        auto budget = std::make_shared<llama_memory_budget>(4096, 4096, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64);
        std::vector<llama_expert_trace_event> events;
        events.reserve(8);
        store.set_trace([](const llama_expert_trace_event & event, void * user) noexcept {
            static_cast<std::vector<llama_expert_trace_event> *>(user)->push_back(event);
        }, &events);
        t.assert_true("configuration reports history cost", events.size() == 1 && events[0].kind == llama_expert_trace_kind::configuration && events[0].payload > 0);
        events.clear();
        store.begin_projection(fixture::name, 7);
        auto first = store.acquire(fixture::name, 2);
        const size_t used = budget->snapshot(domain::host).used();
        first.reset();
        auto second = store.acquire(fixture::name, 2);
        second.reset();
        store.trim();
        store.set_trace(nullptr, nullptr);
        t.assert_equal("projection, two acquisitions/releases and trim", events.size(), size_t(6));
        t.assert_true("projection clock", events[0].kind == llama_expert_trace_kind::projection && events[0].tokens == 7);
        t.assert_true("same key and miss then hit", events[1].tensor == events[3].tensor && events[1].expert == 2 && !events[1].hit && events[3].hit);
        t.assert_equal("payload is transient miss staging", events[1].payload, fixture::expert_bytes);
        t.assert_equal("allocation matches live ledger", events[1].allocation, used);
        t.assert_equal("capacity includes reusable cache", events[3].capacity, size_t(4096));
        t.assert_true("lease releases are observable", events[2].kind == llama_expert_trace_kind::release &&
            events[4].kind == llama_expert_trace_kind::release && events[2].expert == 2);
        t.assert_true("trim is observable", events[5].kind == llama_expert_trace_kind::reset);
    });

    t.test("class LRU scan protection and pinned retirement", [&](testing & t) {
        fixture f;
        auto budget = std::make_shared<llama_memory_budget>(8192, 8192, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64);
        t.assert_true("policy accepted before demand", store.set_policy("class-lru"));
        const size_t history = budget->snapshot(domain::host).used();
        t.assert_true("history admitted", history > 0);
        auto demand = [&](int e) {
            store.begin_projection(fixture::name, 1);
            return store.acquire(fixture::name, e);
        };
        demand(0).reset();
        const size_t entry = store.stats().resident_bytes;
        store.request_budget(domain::host, history + 2 * entry + fixture::expert_bytes);
        for (int i = 0; i < 7; ++i) { demand(0).reset(); }
        demand(1).reset();
        demand(2).reset();
        const auto hits = store.stats().hits;
        auto pinned = demand(0);
        t.assert_equal("frequent expert survives scan", store.stats().hits, hits + 1);
        t.assert_true("mid-session policy switch refused", !store.set_policy("lru"));
        const auto pending = store.request_budget(domain::host, history);
        t.assert_true("live lease remains charged", pending.pending() && store.stats().pinned_entries == 1);
        pinned.reset();
        t.assert_equal("retirement returns buffers only", budget->snapshot(domain::host).used(), history);
        t.assert_true("retirement completes shrink", !budget->snapshot(domain::host).pending());
    });

    t.test("prefill frequency protects repeated weights from cold tail", [&](testing & t) {
        fixture f;
        const size_t entry = fixture::expert_bytes + ggml_tensor_overhead();
        auto budget = std::make_shared<llama_memory_budget>(8192, 0, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64);
        store.acquire(fixture::name, 3).reset();
        t.assert_true("history can start with cold resident weights",
            store.set_prefill_class_lru(true, entry + fixture::expert_bytes));
        const size_t history = budget->snapshot(domain::host).used() - entry;
        const size_t cap = history + 2 * entry + fixture::expert_bytes;
        store.request_budget(domain::host, cap);
        for (int i = 0; i < 2; ++i) {
            store.begin_projection(fixture::name, 128);
            store.acquire(fixture::name, 0).reset();
        }
        // Expert zero has decayed frequency 1.5, still in class zero. The
        // other two demands are cold and have zero age in this projection.
        store.acquire(fixture::name, 1).reset();
        t.assert_true("entry without frequency history is evictable", !store.contains(fixture::name, 3));
        store.acquire(fixture::name, 2).reset();
        t.assert_true("frequent class head survives the cold scan", store.contains(fixture::name, 0));
        t.assert_true("cold class tail is reused", !store.contains(fixture::name, 1) && store.contains(fixture::name, 2));
        t.assert_true("no buffer churn", store.stats().buffer_allocations == 2 && store.stats().buffer_reuses == 2);
        auto pinned = store.acquire(fixture::name, 0);
        store.acquire(fixture::name, 3).reset();
        t.assert_true("active lease cannot be selected", store.contains(fixture::name, 0) && store.stats().pinned_entries == 1);
        t.assert_true("managed peak stays bounded", budget->snapshot(domain::host).high_water <= cap);
        const auto pending = store.request_budget(domain::host, history);
        t.assert_true("pinned shrink waits for completion", pending.pending());
        pinned.reset();
        t.assert_true("lease release completes shrink", !budget->snapshot(domain::host).pending());
        t.assert_equal("retirement leaves only history", budget->snapshot(domain::host).used(), history);
        t.assert_true("prefill mode can end after shrink", store.set_prefill_class_lru(false));
        t.assert_equal("all charges released", budget->snapshot(domain::host).used(), size_t(0));
    });

    t.test("prefill class LRU transitions preserve resident LRU order", [&](testing & t) {
        fixture f;
        const size_t entry = fixture::expert_bytes + ggml_tensor_overhead();
        const size_t working = entry + fixture::expert_bytes;
        auto budget = std::make_shared<llama_memory_budget>(8192, 0, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64);
        store.acquire(fixture::name, 0).reset();
        t.assert_true("class mode can start with resident weights", store.set_prefill_class_lru(true, working));
        const size_t history = budget->snapshot(domain::host).used() - entry;
        t.assert_true("history charged at the batch boundary", history > 0);
        store.request_budget(domain::host, history + 3 * entry + fixture::expert_bytes);
        for (int i = 0; i < 4; ++i) {
            store.begin_projection(fixture::name, 1);
            store.acquire(fixture::name, 0).reset();
        }
        store.acquire(fixture::name, 1).reset();
        store.acquire(fixture::name, 2).reset();
        t.assert_true("short batch restores plain LRU", store.set_prefill_class_lru(false));
        t.assert_equal("switch frees only history", budget->snapshot(domain::host).used(), 3 * entry);
        store.acquire(fixture::name, 3).reset();
        t.assert_true("older high class becomes the next LRU victim", !store.contains(fixture::name, 0));
        t.assert_true("recent lower classes remain resident", store.contains(fixture::name, 1) && store.contains(fixture::name, 2));
        store.request_budget(domain::host, 3 * entry);
        t.assert_true("full cache can admit history for another prefill", store.set_prefill_class_lru(true, working));
        t.assert_equal("optional history reclaims an unpinned slot", store.stats().entries, size_t(2));
        t.assert_true("optional history respects a mandatory working floor", store.set_prefill_class_lru(false));
        store.request_budget(domain::host, working);
        t.assert_true("tight cap falls back to plain LRU", !store.set_prefill_class_lru(true, working));
    });

    t.test("class LRU metadata admission preserves working floor", [&](testing & t) {
        fixture f;
        auto budget = std::make_shared<llama_memory_budget>(1024, 1024, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64);
        t.assert_true("history cannot consume required working set", !store.set_policy("class-lru", 1024));
        t.assert_equal("failed policy leaves ledger unchanged", budget->snapshot(domain::host).used(), size_t(0));
        t.assert_true("LRU still configurable", store.set_policy("lru"));
    });

    t.test("retirement waits for backend completion", [&](testing & t) {
        fixture f;
        auto budget = std::make_shared<llama_memory_budget>(8192, 0, true);
        auto store = std::make_unique<llama_expert_store>(f.source(), budget, backend.get(), domain::host, 64);
        auto lease = store->acquire(fixture::name, 0);
        auto pending = store->request_budget(domain::host, 0);
        t.assert_true("pending while pinned", pending.pending());
        t.assert_true("retiring allocation charged", pending.categories[size_t(category::experts)].retiring > 0);
        store.reset(); // the lease keeps its owner/resources alive
        completion_gate control;
        gate = &control;
        auto original_sync = backend->iface.synchronize;
        backend->iface.synchronize = blocked_completion;
        std::thread worker([&] { lease.reset(); });
        {
            std::unique_lock<std::mutex> lock(control.mutex);
            control.cv.wait(lock, [&] { return control.entered; });
        }
        const auto waiting = budget->snapshot(domain::host);
        t.assert_equal("still allocated during completion", pending.used(), waiting.used());
        t.assert_equal("no early released bytes", pending.released, waiting.released);
        {
            std::lock_guard<std::mutex> lock(control.mutex);
            control.done = true;
        }
        control.cv.notify_all();
        worker.join();
        backend->iface.synchronize = original_sync;
        gate = nullptr;
        auto done = budget->snapshot(domain::host);
        t.assert_equal("released after completion", size_t(0), done.used());
        t.assert_equal("target applied after completion", size_t(0), done.applied);
        t.assert_equal("released allocation charge", pending.released + pending.used(), done.released);
    });
    t.test("direct reads preserve bytes and avoid physical staging", [&](testing & t) {
        fixture f;
        auto budget = std::make_shared<llama_memory_budget>(1024 * 1024, 0, true);
        llama_expert_profile_data profile;
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 65536);
        store.set_profile(&profile);
        auto lease = store.acquire(fixture::name, 2);
        std::vector<float> values(128);
        ggml_backend_tensor_get(lease.tensor(), values.data(), 0, fixture::expert_bytes);
        bool exact = true;
        for (int i = 0; i < 128; ++i) { exact &= values[i] == fixture::weight(2, i / 32, i % 32); }
        t.assert_true("original expert bytes", exact);
        t.assert_equal("direct path selected", profile.direct_read_bytes, uint64_t(fixture::expert_bytes));
        const auto snapshot = budget->snapshot(domain::host);
        t.assert_equal("I/O headroom still admitted", snapshot.high_water, snapshot.used() + fixture::expert_bytes);
        t.assert_equal("no staging allocation was freed", snapshot.released, size_t(0));
        t.assert_true("source phases recorded", profile.source_transfer_ns > 0 && profile.source_validate_ns > 0);
        lease.reset();
        store.trim();
        t.assert_equal("only physical storage reported freed", budget->snapshot(domain::host).released, snapshot.used());
    });
    t.test("unmapped buffers retain staged reads", [&](testing & t) {
        fixture f;
        // CPU storage with host access hidden models a backend without the
        // direct-write capability while keeping tensor_set/get observable.
        auto staged_buft = *ggml_backend_cpu_buffer_type();
        staged_buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t buft, size_t size) {
            auto buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
            if (buffer) { buffer->buft = buft; }
            return buffer;
        };
        auto budget = std::make_shared<llama_memory_budget>(1024 * 1024, 0, true);
        llama_expert_profile_data profile;
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 64, &staged_buft);
        // Hide direct access after the CPU backend's support check. Only the
        // store's I/O path is exercised; no graph executes on this mock type.
        staged_buft.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };
        store.set_profile(&profile);
        auto lease = store.acquire(fixture::name, 1);
        std::vector<float> values(128);
        ggml_backend_tensor_get(lease.tensor(), values.data(), 0, fixture::expert_bytes);
        bool exact = true;
        for (int i = 0; i < 128; ++i) { exact &= values[i] == fixture::weight(1, i / 32, i % 32); }
        t.assert_true("staged bytes preserved", exact);
        t.assert_equal("direct path not used", profile.direct_read_bytes, uint64_t(0));
        t.assert_equal("physical staging returned", budget->snapshot(domain::host).released, fixture::expert_bytes);
    });
#if defined(__APPLE__)
    t.test("adjacent GGUF ranges preserve bytes and never publish partial reads", [&](testing & t) {
        fixture f;
        auto budget = std::make_shared<llama_memory_budget>(1024 * 1024, 0, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 65536);
        const int64_t experts[] = {0, 1, 3};
        std::array<llama_expert_store::lease, 3> weights;
        store.acquire_many(fixture::name, experts, 3, weights.data(), nullptr, true);
        for (size_t i = 0; i < weights.size(); ++i) {
            std::array<float, fixture::expert_bytes / sizeof(float)> values;
            ggml_backend_tensor_get(weights[i].tensor(), values.data(), 0, fixture::expert_bytes);
            bool exact = true;
            for (size_t j = 0; j < values.size(); ++j) {
                exact &= values[j] == fixture::weight(int(experts[i]), int(j / 32), int(j % 32));
            }
            t.assert_true("original slice bytes", exact);
            weights[i].reset();
        }
        t.assert_equal("no gap bytes", store.stats().logical_read_bytes, uint64_t(3 * fixture::expert_bytes));
        t.assert_true("gap remains cold", !store.contains(fixture::name, 2));
        store.trim();
        fs::resize_file(f.path, 0);
        rejects(t, "failed range publishes nothing", [&] {
            store.acquire_many(fixture::name, experts, 3, weights.data(), nullptr, true);
        });
        t.assert_equal("no incomplete entries", store.stats().entries, size_t(0));
        t.assert_equal("no retained charges", budget->snapshot(domain::host).used(), size_t(0));
    });
#endif
    t.test("batched read failure discards unpublished leases", [&](testing & t) {
        fixture f;
        auto budget = std::make_shared<llama_memory_budget>(1024 * 1024, 0, true);
        llama_expert_store store(f.source(), budget, backend.get(), domain::host, 65536);
        const int64_t experts[] = {0, 1};
        llama_expert_store::lease weights[2];

        fs::resize_file(f.path, 0);
        rejects(t, "modified source aborts the batch", [&] { store.acquire_many(fixture::name, experts, 2, weights); });
        t.assert_true("no leases escape", !weights[0] && !weights[1]);
        t.assert_equal("no incomplete entries", size_t(0), store.stats().entries);
        t.assert_equal("no retained charges", size_t(0), budget->snapshot(domain::host).used());
    });

    t.test("allocation and read failure rollback", [&](testing & t) {
        fixture f;
        auto source = f.source();
        auto budget = std::make_shared<llama_memory_budget>(8192, 0, true);
        auto failing_buft = *ggml_backend_cpu_buffer_type();
        failing_buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t, size_t) -> ggml_backend_buffer_t { return nullptr; };
        {
            llama_expert_store fail(source, budget, backend.get(), domain::host, 64, &failing_buft);
            rejects(t, "backend allocation failure", [&] { fail.acquire(fixture::name, 0); });
            t.assert_equal("allocation charges rolled back", size_t(0), budget->snapshot(domain::host).used());
        }
        llama_expert_store store(source, budget, backend.get(), domain::host, 64);
        fs::resize_file(f.path, source->tensor(fixture::name).offset + 1);
        rejects(t, "read failure", [&] { store.acquire(fixture::name, 0); });
        t.assert_equal("no partial publication", size_t(0), store.stats().entries);
        t.assert_equal("all resources returned", size_t(0), budget->snapshot(domain::host).used());
        t.assert_equal("no successful read counted", uint64_t(0), store.stats().logical_read_bytes);
    });
    t.test("staged expert with separate host and device caps", [&](testing & t) {
        fixture f;
        auto staged_buft = *ggml_backend_cpu_buffer_type();
        staged_buft.iface.alloc_buffer = [](ggml_backend_buffer_type_t buft, size_t size) {
            auto buffer = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
            if (buffer) { buffer->buft = buft; }
            return buffer;
        };
        const size_t arena = ggml_tensor_overhead();
        const size_t payload = fixture::expert_bytes;
        auto budget = std::make_shared<llama_memory_budget>(arena + payload - 1, payload, false);
        llama_expert_store store(f.source(), budget, backend.get(), domain::device, 64, &staged_buft);
        staged_buft.iface.is_host = [](ggml_backend_buffer_type_t) { return false; };

        rejects(t, "host staging exceeds its own cap", [&] { store.acquire(fixture::name, 1); });
        t.assert_equal("failed admission returns host charge", size_t(0), budget->snapshot(domain::host).used());
        t.assert_equal("failed admission returns device charge", size_t(0), budget->snapshot(domain::device).used());
        t.assert_equal("failed admission publishes no expert", size_t(0), store.stats().entries);

        budget->request(domain::host, arena + payload);
        auto lease = store.acquire(fixture::name, 1);
        const auto host = budget->snapshot(domain::host);
        const auto device = budget->snapshot(domain::device);
        t.assert_equal("host owns the metadata", arena, host.used());
        t.assert_equal("staging counted at host peak", arena + payload, host.high_water);
        t.assert_equal("staging released after upload", payload, host.released);
        t.assert_equal("device owns the resident weight", payload, device.used());
        float value = 0;
        ggml_backend_tensor_get(lease.tensor(), &value, 0, sizeof(value));
        t.assert_equal("staged upload preserved weight", fixture::weight(1, 0, 0), value);

        t.assert_true("device shrink waits for the lease", store.request_budget(domain::device, 0).pending());
        t.assert_equal("device still charged while leased", payload, budget->snapshot(domain::device).used());
        lease.reset();
        t.assert_equal("device shrink completes", size_t(0), budget->snapshot(domain::device).applied);
        t.assert_equal("device allocation released", payload, budget->snapshot(domain::device).released);
        t.assert_equal("host cap unaffected", arena + payload, budget->snapshot(domain::host).applied);
        t.assert_equal("expert metadata released", size_t(0), budget->snapshot(domain::host).used());
    });
    t.test("backend expert projection", [&](testing & t) {
        fixture f;
        auto budget = std::make_shared<llama_memory_budget>(1024 * 1024, 0, true);
        // 64 KiB is a multiple of the CPU and supported Apple page granularities.
        llama_expert_store store(f.source(), budget, backend.get(), domain::device, 65536);
        auto weight = store.acquire(fixture::name, 2);
        ggml_context_ptr ctx(ggml_init({4 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false), nullptr, true}));
        auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 1);
        auto * output = ggml_mul_mat(ctx.get(), weight.tensor(), input);
        auto * graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        ggml_backend_buffer_ptr scratch(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
        if (!scratch) { throw std::runtime_error("cannot allocate probe scratch"); }
        float inputs[32];
        for (int i = 0; i < 32; ++i) { inputs[i] = (i % 3 - 1) * 0.25f; }
        ggml_backend_tensor_set(input, inputs, 0, sizeof(inputs));
        const auto status = ggml_backend_graph_compute_async(backend.get(), graph);
        t.assert_equal("graph submitted", int(GGML_STATUS_SUCCESS), int(status));
        auto pending = store.request_budget(domain::host, 0);
        t.assert_true("inflight weight remains pinned", pending.pending());
        weight.reset(); // synchronizes the graph before freeing the expert
        float values[4];
        ggml_backend_tensor_get(output, values, 0, sizeof(values));
        for (int row = 0; row < 4; ++row) {
            float expected = 0;
            for (int col = 0; col < 32; ++col) { expected += fixture::weight(2, row, col) * inputs[col]; }
            t.assert_true("projection matches original weights", std::fabs(values[row] - expected) <= 1e-5f);
        }
        t.assert_equal("expert allocations freed", size_t(0), budget->snapshot(domain::host).used());
    });
    t.test("discrete CUDA expert projection and shrink", [&](testing & t) {
        const auto dev = ggml_backend_get_device(backend.get());
        if (std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(dev)), "CUDA") != 0 ||
                ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
            t.skip("requires a discrete CUDA backend");
            return;
        }
        const char * unified = std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY");
        if (unified && std::atoi(unified) != 0) {
            throw std::runtime_error("CUDA unified memory is incompatible with this check");
        }

        fixture f;
        constexpr size_t expert_allocation = 65536;
        const size_t expert_arena = ggml_tensor_overhead();
        const size_t graph_arena = 4 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false);
        const size_t host_cap = graph_arena + expert_arena + fixture::expert_bytes;
        auto budget = std::make_shared<llama_memory_budget>(host_cap, expert_allocation, false);
        auto graph_charge = budget->reserve(domain::host, category::scratch, graph_arena);
        if (!graph_charge) { throw std::runtime_error("graph metadata admission failed"); }
        ggml_context_ptr ctx(ggml_init({graph_arena, nullptr, true}));
        if (!ctx) { throw std::runtime_error("graph metadata allocation failed"); }
        graph_charge.commit();

        llama_expert_profile_data profile {};
        llama_expert_store store(f.source(), budget, backend.get(), domain::device, expert_allocation);
        store.set_profile(&profile);
        llama_memory_budget::ticket scratch_charge;
        ggml_backend_buffer_ptr scratch;
        auto weight = store.acquire(fixture::name, 2);
        t.assert_true("expert buffer is device storage", !ggml_backend_buffer_is_host(weight.tensor()->buffer));
        auto host = budget->snapshot(domain::host);
        t.assert_equal("host peak includes the read staging", host_cap, host.high_water);
        t.assert_equal("read staging released after upload", fixture::expert_bytes, host.released);
        t.assert_equal("host retains graph and expert metadata", graph_arena + expert_arena, host.used());
        t.assert_equal("device owns expert bytes", expert_allocation, budget->snapshot(domain::device).used());
        t.assert_equal("direct host writes were not used", uint64_t(0), profile.direct_read_bytes);

        auto * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 1);
        auto * output = ggml_mul_mat(ctx.get(), weight.tensor(), input);
        auto * graph = ggml_new_graph_custom(ctx.get(), 16, false);
        ggml_build_forward_expand(graph, output);
        const auto buft = ggml_backend_get_default_buffer_type(backend.get());
        const size_t scratch_bytes = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        if (!scratch_bytes) { throw std::runtime_error("graph scratch size unavailable"); }
        budget->request(domain::device, expert_allocation + scratch_bytes);
        scratch_charge = budget->reserve(domain::device, category::scratch, scratch_bytes);
        if (!scratch_charge) { throw std::runtime_error("graph scratch admission failed"); }
        scratch.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        if (!scratch || ggml_backend_buffer_get_size(scratch.get()) != scratch_bytes) {
            throw std::runtime_error("graph scratch allocation differs from admission");
        }
        scratch_charge.commit();
        float inputs[32];
        for (int i = 0; i < 32; ++i) { inputs[i] = (i % 3 - 1) * 0.25f; }
        ggml_backend_tensor_set(input, inputs, 0, sizeof(inputs));
        if (ggml_backend_graph_compute_async(backend.get(), graph) != GGML_STATUS_SUCCESS) {
            weight.reset();
            throw std::runtime_error("CUDA projection submission failed");
        }
        const auto pending = store.request_budget(domain::device, scratch_bytes);
        weight.reset();
        t.assert_true("device shrink waits for the expert lease", pending.pending());
        t.assert_equal("device peak includes expert and graph scratch", expert_allocation + scratch_bytes, pending.high_water);
        const auto after = budget->snapshot(domain::device);
        t.assert_true("device shrink completes after lease release", !after.pending());
        t.assert_equal("only graph scratch remains in VRAM", scratch_bytes, after.used());
        t.assert_equal("expert allocation released", expert_allocation, after.released);
        t.assert_equal("host cap unaffected", host_cap, budget->snapshot(domain::host).applied);
        t.assert_equal("expert metadata released", graph_arena, budget->snapshot(domain::host).used());

        float values[4];
        ggml_backend_tensor_get(output, values, 0, sizeof(values));
        for (int row = 0; row < 4; ++row) {
            float expected = 0;
            for (int col = 0; col < 32; ++col) { expected += fixture::weight(2, row, col) * inputs[col]; }
            t.assert_equal("projection matches original weight exactly", expected, values[row]);
        }
        scratch.reset();
        scratch_charge.reset();
        ctx.reset();
        graph_charge.reset();
        t.assert_equal("all host charges released", size_t(0), budget->snapshot(domain::host).used());
        t.assert_equal("all device charges released", size_t(0), budget->snapshot(domain::device).used());
    });
    return t.summary();
}
