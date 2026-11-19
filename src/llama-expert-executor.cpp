#include "llama-expert-acceleration.h"
#include "llama-lfm-resident.h"
#include "llama-expert-executor.h"
#include "llama-expert-preview-dot.h"
#include "llama-expert-route-index.h"
#include "llama-expert-moe.h"
#include "llama-memory-contract.h"

#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "../ggml/src/ggml-impl.h" // custom op identity, shared with ggml_custom_4d

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

namespace {
constexpr size_t allocation_granularity = 65536;
constexpr size_t max_wave_rows = 32;
constexpr size_t max_wave_groups = 16;
constexpr size_t max_wave_nodes = max_wave_rows * max_wave_groups;
constexpr size_t class_lru_prefill_tokens = 320;
constexpr size_t class_lru_budget = size_t(8) * 1024 * 1024 * 1024;
size_t rounded(size_t size, size_t granularity = allocation_granularity) {
    if (size > std::numeric_limits<size_t>::max() - granularity) {
        throw std::runtime_error("expert workspace size overflow");
    }
    return (size + granularity - 1) / granularity * granularity;
}
llama_memory_budget::ticket admitted(llama_memory_budget & budget, llama_memory_category category, size_t size,
        llama_memory_domain domain = llama_memory_domain::host) {
    auto ticket = budget.reserve(domain, category, size);
    if (!ticket) { throw std::runtime_error("expert workspace does not fit managed budget"); }
    return ticket;
}
} // namespace

struct llama_expert_executor::impl {
    struct projection {
        impl * owner;
        std::string name;
        int64_t input, output, experts;
        bool on_cpu;
    };
    std::shared_ptr<llama_memory_budget> budget;
    ggml_backend_ptr backend;
    llama_expert_profile_data profile_data;
    llama_expert_profile_data * profile = nullptr;
    std::unique_ptr<llama_expert_store> store;
    llama_memory_budget::ticket moe_charge;
    std::unique_ptr<llama_expert_moe> moe;
    ggml_backend_ptr cpu_backend;
    std::unique_ptr<llama_expert_store> cpu_store;
    std::map<const ggml_tensor *, projection> projections;
    std::shared_ptr<llama_expert_source> row_source;
    const ggml_tensor * row_table = nullptr;
    llama_memory_budget::ticket row_charge;
    std::unique_ptr<uint8_t[]> row_buffer;
    // Separate from PLE: retain the qualified Metal row path unchanged.
    std::shared_ptr<llama_expert_source> token_row_source;
    const ggml_tensor * token_row_table = nullptr;
    llama_memory_budget::ticket token_row_charge;
    std::unique_ptr<uint8_t[]> token_row_buffer;
    // Charges precede resources so destruction frees resources first.
    llama_memory_budget::ticket meta_charge, buffer_charge, host_charge;
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr buffer;
    std::unique_ptr<float[]> host;
    llama_memory_budget::ticket route_charge;
    std::unique_ptr<llama_expert_route_index> route_index;
    struct early_state {
        struct occurrence {
            const projection * source = nullptr;
            const projection * target = nullptr;
            int index = -1; // -1 absent, -2 ambiguous
            int64_t rows = 0, slots = 0;
        };
        std::array<occurrence, 768> entries {};
        size_t count = 0;
    };
    llama_memory_budget::ticket early_charge;
    std::unique_ptr<early_state> early;
    size_t input_offset = 0;
    size_t output_offset = 0;
    size_t ids_offset = 0;
    size_t host_output = 0;
    size_t minimum = 0;
    size_t fixed_workspace = 0;
    size_t workspace_groups = 1, max_expert_payload = 0, max_expert_allocation = 0;
    size_t full_expert_storage = 0;
    llama_discrete_buffer_types discrete;
    bool metal = false;
    bool qwen_cuda = false;
    bool read_overlap = false;
    bool vector_read_overlap = false;
    bool parallel_window_reads = false;
    bool single_expert_map = false;
    using prepare_map_t = bool (*)(ggml_tensor *);
    prepare_map_t prepare_map = nullptr;
    llama_expert_route_callback route_observer = nullptr;
    void * route_user = nullptr;
    using graph_timing_t = bool (*)(ggml_backend_t, uint64_t *, uint64_t *);
    graph_timing_t graph_timing = nullptr;
    uint64_t n_projections = 0, n_matmuls = 0, n_submissions = 0;
    struct preview_state {
        struct layer {
            const projection * gate = nullptr;
            const projection * up = nullptr;
            const float * router = nullptr; // immutable alias, already charged to model weights
            size_t width = 0, experts = 0;
            std::array<const projection *, 3> target {};
            struct occurrence {
                int index = -1; // -1 absent, -2 ambiguous
                int64_t rows = 0;
            };
            std::array<occurrence, 5> graph {}; // gate, up, next gate/up/down
        };
        int mode = 0;
        size_t count = 0;
        std::array<layer, 256> layers {};
        std::array<float, 4 * 1024> scores {};
        std::array<std::array<int32_t, 4>, 256> top {};
        const layer * last = nullptr;
        const projection * first = nullptr;
        uint64_t started = 0;
    };
    llama_memory_budget::ticket preview_charge;
    std::unique_ptr<preview_state> preview;
    bool tracing = false, class_policy = false;
    bool selective_reclaim = false;
    bool automatic_acceleration = false;
    bool dflash_verify = false;
    bool profile_overlap = false;
    std::unique_ptr<llama_lfm_resident> lfm_resident;
    std::shared_ptr<llama_expert_source> acceleration_source;
    size_t acceleration_width = 0;
    uint64_t retired_moe_calls = 0, retired_pipeline_groups = 0, retired_resident_groups = 0;
    bool layer_window = false;
    bool fused_window_moe = false;
    using window_addresses_t = bool (*)(ggml_tensor *, ggml_tensor *);
    window_addresses_t window_addresses = nullptr;
    llama_memory_budget::ticket cross_window_charge;
    std::unique_ptr<float[]> cross_window_input, cross_window_output;
    std::unique_ptr<int32_t[]> cross_window_ids;
    size_t cross_window_width = 0, cross_window_routes = 0;
    const ggml_tensor * cross_window_ready = nullptr;
    size_t window_advice_budget = 0;
    bool policy_override = false;
    std::atomic<bool> failed {false};
    char failure[512] {};

    impl(std::shared_ptr<llama_expert_source> source, const std::vector<const ggml_tensor *> & weights,
         ggml_backend_dev_t device, size_t cap, std::shared_ptr<llama_memory_budget> shared_budget,
         const llama_memory_contract * discrete_contract, const std::vector<const ggml_tensor *> & cpu_weights) :
        budget(discrete_contract ? discrete_contract->ledger :
            shared_budget ? std::move(shared_budget) : std::make_shared<llama_memory_budget>(cap, 0, true)),
        discrete(discrete_contract ? discrete_contract->discrete : llama_discrete_buffer_types{}) {
        llama_allocation_scope explicit_admission;
        if (!source || !device || weights.empty()) { throw std::invalid_argument("invalid expert executor configuration"); }
        const std::string device_name = ggml_backend_dev_name(device);
        if (discrete_contract && (!discrete.device || ggml_backend_dev_buffer_type(device) != discrete.device ||
                ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU ||
                std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "CUDA") != 0)) {
            throw std::invalid_argument("invalid discrete CUDA expert executor configuration");
        }
        if (!discrete.device && device_name != "CPU" && device_name != "MTL0") {
            throw std::runtime_error("expert executor currently supports CPU and MTL0 only");
        }
        metal = device_name == "MTL0";
        qwen_cuda = discrete.device && source->architecture() == "qwen4exp";
        // Qwen4Exp is qualified for vector batches; LFM and Qwen3.5 use
        // matrix waves. Both retain the same budget and lease checks.
        vector_read_overlap = metal && source->architecture() == "qwen4exp";
        const bool requested_acceleration = metal && !discrete.device && llama_expert_acceleration_requested();
        // Explicit auto/off takes precedence; an absent unified setting preserves legacy opt-ins.
        automatic_acceleration = requested_acceleration && (std::getenv("LLAMA_MOE_STREAMING") ||
            !std::getenv("LLAMA_EXPERIMENTAL_METAL_MOE")) &&
            !source->packed_stride() && (source->architecture() == "qwen4exp" ||
                source->architecture() == "qwen35moe" || source->architecture() == "lfm2moe");
        if (automatic_acceleration) { acceleration_source = source; }
        const char * reclaim_mode = std::getenv("LLAMA_EXPERIMENTAL_EXPERT_RECLAIM");
        selective_reclaim = automatic_acceleration || (!llama_expert_acceleration_off() && vector_read_overlap && reclaim_mode && std::strcmp(reclaim_mode, "1") == 0);
        if (vector_read_overlap && !source->packed_stride()) {
            const char * parallel = std::getenv("LLAMA_EXPERT_PARALLEL_READS");
            parallel_window_reads = parallel && std::strcmp(parallel, "1") == 0;
        }
        read_overlap = vector_read_overlap ||
            (metal && (source->architecture() == "lfm2moe" || source->architecture() == "qwen35moe"));
        backend.reset(ggml_backend_dev_init(device, nullptr));
        if (!backend) { throw std::runtime_error("cannot initialize expert compute backend"); }
        const auto reg = ggml_backend_dev_backend_reg(device);
        if (discrete.device) {
            using enable_pool_t = bool (*)(ggml_backend_t);
            const auto enable = reinterpret_cast<enable_pool_t>(
                ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_enable_observed_pool"));
            if (!enable || !enable(backend.get())) {
                throw std::runtime_error("cannot enable observed CUDA expert workspace");
            }
        }
        graph_timing = reinterpret_cast<graph_timing_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device), "ggml_backend_metal_last_graph_timing"));
        prepare_map = reinterpret_cast<prepare_map_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device), "ggml_backend_metal_prepare_single_expert"));
        // Repeated on/off runs qualify these Metal profiles. Preparation uses
        // the existing output tail; vector waves and unsupported buffers fall back.
        single_expert_map = metal && prepare_map &&
            (source->architecture() == "lfm2moe" || source->architecture() == "qwen35moe");
        // Avoid nested parallel work while executing inside the main CPU graph.
        using set_threads_t = void (*)(ggml_backend_t, int);
        auto set_threads = reinterpret_cast<set_threads_t>(ggml_backend_reg_get_proc_address(
            ggml_backend_dev_backend_reg(device), "ggml_backend_set_n_threads"));
        if (set_threads) { set_threads(backend.get(), 1); }
        size_t max_input = 0, max_output = 0, max_expert = 0;
        for (const auto * w : weights) {
            switch (w->type) {
                case GGML_TYPE_F32: case GGML_TYPE_F16: case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_K: case GGML_TYPE_Q5_K: case GGML_TYPE_Q6_K: case GGML_TYPE_IQ4_XS: break;
                case GGML_TYPE_Q2_0: case GGML_TYPE_IQ2_XXS: case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S: case GGML_TYPE_IQ3_XXS: case GGML_TYPE_IQ3_S: case GGML_TYPE_IQ4_NL:
                    if (source->architecture() == "qwen4exp" && (metal || discrete.device)) { break; }
                    throw std::runtime_error("this expert format requires experimental Qwen4Exp Metal or CUDA streaming");
                default: throw std::runtime_error("expert quantization is not qualified for streaming: " + std::string(w->name));
            }
            const auto & info = source->tensor(w->name);
            if (!info.can_read_experts() || info.type != w->type || !std::equal(info.ne.begin(), info.ne.end(), w->ne)) {
                throw std::runtime_error("expert source layout differs from model: " + std::string(w->name));
            }
            const bool on_cpu = !cpu_weights.empty() && std::find(cpu_weights.begin(), cpu_weights.end(), w) != cpu_weights.end();
            projections.emplace(w, projection{this, w->name, w->ne[0], w->ne[1], w->ne[2], on_cpu});
            max_input = std::max(max_input, size_t(w->ne[0]));
            max_output = std::max(max_output, size_t(w->ne[1]));
            max_expert = std::max(max_expert, info.expert_bytes());
            if (discrete.device) {
                auto slice = *w;
                slice.ne[2] = 1;
                slice.nb[3] = slice.nb[2];
                max_expert_allocation = std::max(max_expert_allocation,
                    rounded(ggml_backend_buft_get_alloc_size(discrete.device, &slice)));
            }
        }
        max_expert_payload = max_expert;
        const char * compact_mode = std::getenv("LLAMA_EXPERIMENTAL_EXPERT_COMPACT");
        const bool compact_buffers = automatic_acceleration || (!llama_expert_acceleration_off() && vector_read_overlap && compact_mode && std::strcmp(compact_mode, "1") == 0);
        const char * recycle_mode = std::getenv("LLAMA_EXPERIMENTAL_EXPERT_RECYCLE");
        const bool recycle_buffers = compact_buffers && (automatic_acceleration || (max_expert <= 1024 * 1024 &&
            recycle_mode && std::strcmp(recycle_mode, "1") == 0));
        size_t expert_granularity = allocation_granularity;
        // Equal-sized Metal slots avoid reallocating mixed-quantization LRU
        // victims. Bound the qualified Qwen4Exp geometry at 1 MiB; larger
        // experts retain the generic allocation policy. Charge the whole slot.
        // Opt-in compact buffers instead round each slice to 64 KiB and only
        // reuse victims in the same size class, preventing oversized retention.
        if (vector_read_overlap && !compact_buffers && max_expert <= 1024 * 1024) {
            while (expert_granularity < max_expert) { expert_granularity *= 2; }
        }
        if (!discrete.device) { max_expert_allocation = rounded(max_expert, expert_granularity) + ggml_tensor_overhead(); }
        const char * prefill_batch_mode = automatic_acceleration ?
            std::getenv("LLAMA_EXPERIMENTAL_PREFILL_EXPERT_BATCH") : nullptr;
        if (prefill_batch_mode && std::strcmp(prefill_batch_mode, "auto") == 0) {
            for (const auto * weight : weights) {
                const auto & info = source->tensor(weight->name);
                const size_t bytes = rounded(info.expert_bytes(), expert_granularity) + ggml_tensor_overhead();
                if (size_t(info.ne[2]) > (SIZE_MAX - full_expert_storage) / bytes) {
                    full_expert_storage = SIZE_MAX;
                    break;
                }
                full_expert_storage += size_t(info.ne[2]) * bytes;
            }
        }
        const auto initial = budget->snapshot(llama_memory_domain::host);
        const auto buffer_domain = discrete.device ? llama_memory_domain::device : llama_memory_domain::host;
        size_t meta_size = 0, buffer_size = 0, host_elements = 0, host_bytes = 0;
        // Qwen4Exp's qualified vector profile amortizes submission/wait across
        // sixteen experts. Other Metal families retain four; admission below
        // halves the workspace if the requested budget cannot hold it.
        workspace_groups = metal ? (vector_read_overlap ? max_wave_groups : 4) : 1;
        for (;;) {
            const size_t total_rows = max_wave_rows * workspace_groups;
            meta_size = (metal ? 2 * total_rows + 1 : 4) * ggml_tensor_overhead() +
                ggml_graph_overhead_custom(metal ? 4 * total_rows : 8, false);
            ids_offset = rounded(total_rows * max_input * sizeof(float));
            output_offset = ids_offset + rounded(max_wave_rows * sizeof(int32_t));
            // Include per-node Metal indexed-matmul maps and alignment padding.
            buffer_size = output_offset + rounded(total_rows * max_output * sizeof(float)) + allocation_granularity;
            host_output = total_rows * max_input;
            host_elements = total_rows * (max_input + max_output);
            host_bytes = host_elements * sizeof(float);
            minimum = meta_size + buffer_size + host_bytes + max_expert_allocation + max_expert;
            fixed_workspace = meta_size + buffer_size + host_bytes;
            if (discrete.device) {
                const auto vram = budget->snapshot(buffer_domain);
                fixed_workspace = meta_size + host_bytes;
                minimum = fixed_workspace + ggml_tensor_overhead() + max_expert_payload;
                if (initial.used() > initial.requested || vram.used() > vram.requested ||
                        minimum > initial.requested - initial.used() ||
                        buffer_size + max_expert_allocation > vram.requested - vram.used()) {
                    throw std::runtime_error("discrete budget cannot fit expert staging and a device working set");
                }
                break;
            }
            if (initial.used() <= initial.requested && initial.requested - initial.used() >= minimum) { break; }
            if (workspace_groups == 1) {
                throw std::runtime_error("expert streaming budget below minimum working set: requires at least " + std::to_string(minimum) + " bytes");
            }
            workspace_groups /= 2;
        }
        meta_charge = admitted(*budget, llama_memory_category::runtime, meta_size);
        buffer_charge = admitted(*budget, llama_memory_category::scratch, buffer_size, buffer_domain);
        host_charge = admitted(*budget, llama_memory_category::io, host_bytes);
        ctx.reset(ggml_init({meta_size, nullptr, true}));
        if (!ctx) { throw std::runtime_error("cannot allocate expert workspace metadata"); }
        meta_charge.commit();
        buffer.reset(ggml_backend_alloc_buffer(backend.get(), buffer_size));
        if (!buffer) { throw std::runtime_error("cannot allocate expert workspace buffer"); }
        buffer_charge.commit();
        if (ggml_backend_buffer_get_size(buffer.get()) != buffer_size) {
            throw std::runtime_error("expert workspace allocation differs from reservation");
        }
        host.reset(new float[host_elements]);
        host_charge.commit();
        if (!cpu_weights.empty()) {
            if (!discrete.device) { throw std::invalid_argument("mixed expert execution requires a discrete contract"); }
            cpu_backend.reset(ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr));
            if (!cpu_backend) { throw std::runtime_error("cannot initialize host expert backend"); }
            cpu_store = std::make_unique<llama_expert_store>(source, budget, cpu_backend.get(),
                    llama_memory_domain::host, allocation_granularity, ggml_backend_cpu_buffer_type());
        }
        store.reset(new llama_expert_store(source, budget, backend.get(),
                                           buffer_domain, expert_granularity, nullptr, compact_buffers, recycle_buffers));
        const char * combined = std::getenv("LLAMA_EXPERIMENTAL_METAL_MOE");
        acceleration_width = std::max(max_input, max_output);
        if (!std::getenv("LLAMA_MOE_STREAMING") && combined && std::strcmp(combined, "1") == 0 && vector_read_overlap &&
                max_input <= 8192 && max_output <= 8192) {
            moe_charge = admitted(*budget, llama_memory_category::runtime, sizeof(llama_expert_moe));
            moe = std::make_unique<llama_expert_moe>(budget, backend.get(), *store, std::move(source),
                    std::max(max_input, max_output), max_expert_allocation, max_expert_payload,
                    failed, failure, sizeof(failure), profile, n_projections, n_matmuls, n_submissions);
            moe_charge.commit();
            minimum += 2 * max_expert_allocation;
        }
    }

    void configure_acceleration() {
        if (!automatic_acceleration || moe || !acceleration_source) { return; }
        if (acceleration_width > 8192) {
            std::fprintf(stderr,"expert acceleration: fallback (width exceeds qualified Metal geometry)\n"); return;
        }
        try {
            auto charge = budget->reserve(llama_memory_domain::host, llama_memory_category::runtime, sizeof(llama_expert_moe));
            if (!charge) { throw std::runtime_error("metadata does not fit"); }
            auto candidate = std::make_unique<llama_expert_moe>(budget, backend.get(), *store, acceleration_source,
                acceleration_width, max_expert_allocation, max_expert_payload, failed, failure, sizeof(failure),
                profile, n_projections, n_matmuls, n_submissions, true, dflash_verify, profile_overlap);
            charge.commit(); moe_charge = std::move(charge); moe = std::move(candidate);
            std::fprintf(stderr,"expert acceleration: auto architecture=%s compact=1 recycle=1 reclaim=1 FP32=1 address_batch=%d priority_readers=%d prefetch=%zu ready=4 optional_bytes=%zu\n",
                acceleration_source->architecture().c_str(),int(moe->address_batches()),int(moe->priority_reads()),
                moe->prefetch_candidates(),moe->optional_bytes()+moe_charge.size());
        } catch (const std::exception & e) {
            std::fprintf(stderr,"expert acceleration: fallback (%s); ordinary streaming retained\n",e.what());
        }
    }
    size_t optional_bytes() const {
        return automatic_acceleration && moe ? moe_charge.size()+moe->optional_bytes() : 0;
    }
    void drop_acceleration() {
        if (!automatic_acceleration || !moe) { return; }
        moe->end_graph(); ggml_backend_synchronize(backend.get());
        retired_moe_calls += moe->completed(); retired_pipeline_groups += moe->overlapped_groups();
        retired_resident_groups += moe->resident_starts();
        moe.reset(); moe_charge = {};
    }

    static int32_t id_at(const ggml_tensor * ids, int64_t slot, int64_t token) {
        int32_t id;
        std::memcpy(&id, static_cast<const char *>(ids->data) + slot * ids->nb[0] + token * ids->nb[1], sizeof(id));
        return id;
    }

    void compute_cpu(const projection & p, ggml_tensor * dst, const ggml_tensor * input, const ggml_tensor * ids) {
        std::array<char *, max_wave_rows> destinations;
        std::array<int32_t, max_wave_rows> local_ids {};
        for (int64_t expert = 0; expert < p.experts; ++expert) {
            size_t rows = 0;
            for (int64_t token = 0; token < ids->ne[1]; ++token) {
                for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                    if (id_at(ids, slot, token) != expert) { continue; }
                    if (rows == destinations.size()) { throw std::runtime_error("host expert route count exceeds workspace"); }
                    const char * row = static_cast<const char *>(input->data) +
                        (input->ne[1] == 1 ? 0 : slot) * input->nb[1] + token * input->nb[2];
                    for (int64_t col = 0; col < p.input; ++col) {
                        std::memcpy(host.get() + rows * p.input + col, row + col * input->nb[0], sizeof(float));
                    }
                    destinations[rows++] = static_cast<char *>(dst->data) + slot * dst->nb[1] + token * dst->nb[2];
                }
            }
            if (!rows) { continue; }
            auto weight = cpu_store->acquire(p.name, expert);
            ggml_reset(ctx.get());
            auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, p.input, 1, rows);
            auto * route = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, rows);
            auto * y = ggml_mul_mat_id(ctx.get(), weight.tensor(), x, route);
            x->data = host.get();
            route->data = local_ids.data();
            y->data = host.get() + host_output;
            auto * graph = ggml_new_graph_custom(ctx.get(), 8, false);
            ggml_build_forward_expand(graph, y);
            auto plan = ggml_graph_plan(graph, 1, nullptr);
            auto charge = budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, plan.work_size);
            if (plan.work_size && !charge) {
                cpu_store->evict_idle();
                charge = admitted(*budget, llama_memory_category::scratch, plan.work_size);
            }
            std::unique_ptr<uint8_t[]> work;
            if (charge) {
                work.reset(new uint8_t[plan.work_size]);
                charge.commit();
                plan.work_data = work.get();
            }
            if (ggml_graph_compute(graph, &plan) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("host expert projection failed");
            }
            for (size_t row = 0; row < rows; ++row) {
                std::memcpy(destinations[row], host.get() + host_output + row * p.output, p.output * sizeof(float));
            }
            ggml_reset(ctx.get());
            ++n_matmuls;
            ++n_submissions;
        }
        // Keep host paging bounded to one projection so it cannot consume the
        // staging headroom of a later CUDA projection.
        cpu_store->evict_idle();
        ++n_projections;
    }

    const preview_state::layer * preview_layer(const projection & p, const ggml_tensor * input) {
        if (!preview || input->ne[1] != 1 || input->nb[0] != sizeof(float) || input->ne[2] > 256) { return nullptr; }
        for (size_t i = 0; i < preview->count; ++i) {
            const auto & layer = preview->layers[i];
            if (&p != layer.gate && &p != layer.up) { continue; }
            if (preview->last == &layer && preview->first != &p && n_projections == preview->started + 1) { return nullptr; }
            preview->last = &layer; preview->first = &p; preview->started = n_projections;
            const auto & source = layer.graph[&p == layer.gate ? 0 : 1];
            for (size_t target = 2; target < layer.graph.size(); ++target) {
                const auto & next = layer.graph[target];
                if (source.index < 0 || next.index <= source.index) {
                    if (profile) { ++profile->preview_skip_graph; }
                    return nullptr;
                }
                if (source.rows != input->ne[2] || next.rows != source.rows) {
                    if (profile) { ++profile->preview_skip_shape; }
                    return nullptr;
                }
            }
            return &layer;
        }
        return nullptr;
    }

    void run_preview(const preview_state::layer & layer, const ggml_tensor * input, size_t limit,
                     std::array<llama_expert_store::lease, 12> & leases) {
        llama_expert_timer timer(profile ? &profile->preview_ns : nullptr);
        auto resident = [&](const projection * p, int64_t expert = -1) {
            llama_expert_timer check(profile ? &profile->preview_residency_ns : nullptr);
            return store->contains(p->name, expert);
        };
        if (!limit) { if (profile) { ++profile->preview_skip_budget; } return; }
        if (std::all_of(layer.target.begin(), layer.target.end(), [&](const projection * p) { return resident(p); })) {
            if (profile) { ++profile->preview_skip_hot; } return;
        }
        if (profile) { ++profile->preview_generations; }
        std::array<bool, 1024> seen {};
        size_t selected = 0, computed = 0;
        // Lazily compute rows: rank-first global Top4 often finishes before all
        // prefill rows need prediction. No per-row-scaled speculative union.
        for (size_t rank = 0; rank < 4 && selected < limit; ++rank) {
            for (size_t row = 0; row < size_t(input->ne[2]) && selected < limit; ++row) {
                if (row >= computed) {
                    const size_t count = preview->mode != 3 && size_t(input->ne[2]) - row >= 4 ? 4 : 1;
                    const char * rows = static_cast<const char *>(input->data) + row * input->nb[2];
                    llama_expert_timer math(profile ? &profile->preview_dot_ns : nullptr);
                    if (count == 4) {
                        llama_expert_preview_dot<4>(layer.router, rows, input->nb[2], layer.width,
                            layer.experts, preview->scores.data(), 1024);
                    } else {
                        llama_expert_preview_dot<1>(layer.router, rows, input->nb[2], layer.width,
                            layer.experts, preview->scores.data(), 1024);
                    }
                    math.next(profile ? &profile->preview_rank_ns : nullptr);
                    for (size_t r = 0; r < count; ++r) {
                        const auto begin = preview->scores.begin() + r * 1024;
                        if (!std::all_of(begin, begin + layer.experts, [](float score) { return std::isfinite(score); })) { return; }
                        for (size_t k = 0; k < 4; ++k) {
                            const auto best = std::max_element(begin, begin + layer.experts);
                            preview->top[row + r][k] = int32_t(best - begin);
                            *best = -std::numeric_limits<float>::infinity();
                        }
                    }
                    math.next(nullptr);
                    computed += count;
                    if (profile) { profile->preview_rows += count; }
                }
                const int32_t e = preview->top[row][rank];
                if (seen[size_t(e)]) { continue; }
                seen[size_t(e)] = true;
                if (std::all_of(layer.target.begin(), layer.target.end(), [&](const projection * p) { return resident(p, e); })) { continue; }
                if (preview->mode >= 2) {
                    timer.next(nullptr);
                    llama_expert_timer read_timer(profile ? &profile->prefetch_ns : nullptr);
                    const auto before = profile ? profile->direct_read_bytes : 0;
                    for (size_t i = 0; i < 3; ++i) {
                        leases[selected * 3 + i] = store->acquire(layer.target[i]->name, e);
                        if (profile) { ++profile->prefetch_slices; }
                    }
                    if (profile) {
                        ++profile->prefetch_experts;
                        profile->prefetch_read_bytes += profile->direct_read_bytes - before;
                    }
                    read_timer.next(nullptr);
                    timer.next(profile ? &profile->preview_ns : nullptr);
                }
                ++selected;
            }
        }
    }

    void compute(const projection & p, ggml_tensor * dst, bool keep_vector = false) {
        llama_expert_timer projection_timer(profile ? &profile->projection_ns : nullptr);
        llama_allocation_scope explicit_admission;
        const ggml_tensor * input = dst->src[0];
        const ggml_tensor * ids = dst->src[1];
        if (input->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I32 ||
            input->ne[0] != p.input || input->ne[2] != ids->ne[1] ||
            (input->ne[1] != 1 && input->ne[1] != ids->ne[0])) {
            throw std::runtime_error("unsupported routed projection input layout");
        }
        const projection * early_target = nullptr;
        if (early) {
            for (size_t i = 0; i < early->count; ++i) {
                const auto & entry = early->entries[i];
                if (entry.source == &p && entry.rows == ids->ne[1] && entry.slots == ids->ne[0]) {
                    early_target = entry.target;
                    break;
                }
            }
        }
        std::array<int64_t, 4> first_ids;
        first_ids.fill(p.experts);
        int64_t final_expert = -1;
        const uint64_t route_count = uint64_t(ids->ne[0]) * uint64_t(ids->ne[1]);
        auto * routes = route_index && llama_expert_route_index::fits(ids, p.experts) ? route_index.get() : nullptr;
        if (profile) { profile->route_id_reads += route_count; }
        if (routes) {
            llama_expert_timer index_timer(profile ? &profile->route_index_ns : nullptr);
            routes->build(ids, p.experts);
            if (routes->count) { final_expert = routes->active[routes->count - 1]; }
            if (early_target) {
                for (size_t i = 0; i < std::min(size_t(routes->count), first_ids.size()); ++i) { first_ids[i] = routes->active[i]; }
            }
            if (profile) { ++profile->route_indexed_projections; }
        } else {
            if (route_index && profile) { ++profile->route_index_fallbacks; }
            for (int64_t token = 0; token < ids->ne[1]; ++token) {
                for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                    const int32_t id = id_at(ids, slot, token);
                    final_expert = std::max(final_expert, int64_t(id));
                    if (id < 0 || id >= p.experts) { throw std::runtime_error("router produced an invalid expert ID"); }
                    if (early_target) {
                        auto at = std::lower_bound(first_ids.begin(), first_ids.end(), id);
                        if (at != first_ids.end() && *at != id) {
                            std::move_backward(at, first_ids.end() - 1, first_ids.end());
                            *at = id;
                        }
                    }
                }
            }
        }
        store->begin_projection(p.name, ids->ne[1]);
        if (route_observer) { route_observer({p.name.c_str(), input, ids, n_projections}, route_user); }
        if (p.on_cpu) {
            compute_cpu(p, dst, input, ids);
            return;
        }
        // Match Metal's MUL_MAT_ID matrix/vector selection. The matrix kernel
        // rounds dequantized weights and activations to half; replacing it with
        // one-row vector kernels changes prefill numerics. Pad bounded waves to
        // 32 rows so MUL_MAT_ID uses the corresponding matrix kernel, even
        // when an expert owns only one row. Original GGUF bytes remain intact.
        const bool matrix_wave = !keep_vector && metal && p.input >= 64 && p.experts <= 1024 && ids->ne[1] >= 32;
        const bool vector_batch = metal && !matrix_wave && ids->ne[1] > 1;
        const bool address_batch = vector_batch && window_addresses;
        // CUDA selects indexed kernels by the original token count. Preserve
        // that dimension even when only a few rows route to this expert.
        if (discrete.device && ids->ne[1] > int64_t(max_wave_rows)) {
            throw std::runtime_error("discrete expert projection requires a microbatch of at most 32 tokens");
        }
        const size_t wave_rows = discrete.device ? size_t(ids->ne[1]) : matrix_wave || vector_batch ? max_wave_rows : 1;
        // Select a group size that can retain every referenced expert plus one
        // miss staging buffer, even if every victim has the largest slice size.
        size_t available = 0;
        bool file_advice = false;
        const bool decode_advice = window_advice_budget && read_overlap && !layer_window && ids->ne[1] == 1;
        if (!discrete.device) {
            const auto resident = store->storage_bytes();
            const auto memory = budget->snapshot(llama_memory_domain::host);
            const size_t fixed = memory.used() - resident;
            available = memory.applied > fixed ? memory.applied - fixed : 0;
#if defined(__APPLE__)
            const bool at_window_budget = window_advice_budget &&
                    memory.requested == window_advice_budget && memory.applied == window_advice_budget;
            const bool at_class_lru_budget = memory.requested == class_lru_budget && memory.applied == class_lru_budget;
            const bool prefill_advice = vector_batch && ids->ne[1] <= 8 &&
                    ((class_policy && at_class_lru_budget) || (layer_window && at_window_budget));
            file_advice = vector_read_overlap && !policy_override && !tracing &&
                    (prefill_advice || (decode_advice && at_window_budget));
#endif
        }
        const size_t group_limit = vector_read_overlap && !matrix_wave ? workspace_groups :
            std::min(workspace_groups, size_t(4));
        const size_t groups = std::max(size_t(1), std::min(group_limit,
            available > max_expert_payload ? (available - max_expert_payload) / max_expert_allocation : 0));
        const size_t expert_capacity = available > max_expert_payload ?
            (available - max_expert_payload) / max_expert_allocation : 0;
        // Independent vector nodes share the same submitted graph and lease
        // lifetime as matrix waves. Qualify this overlap separately by family.
        const bool overlap_wave = vector_read_overlap ? vector_batch : matrix_wave;
        const size_t read_groups = vector_read_overlap ? max_wave_groups : 4;
        const size_t ahead_limit = read_overlap && overlap_wave && expert_capacity > groups ?
            std::min(read_groups, expert_capacity - groups) : 0;
        const bool parallel_ahead = parallel_window_reads && layer_window && vector_batch && !tracing && ahead_limit > 1;
        const size_t early_limit = expert_capacity > groups ? std::min(size_t(4), expert_capacity - groups) : 0;
        if (early_target && (!matrix_wave || !early_limit)) {
            if (profile) { ++(matrix_wave ? profile->early_skip_budget : profile->early_skip_short); }
            early_target = nullptr;
        }
        std::array<llama_expert_store::lease, 4> early_leases;
        const auto * future = preview_layer(p, input);
        // The original placement is retained only as an explicit diagnostic
        // control (mode 3). Tiny vector kernels cannot hide this read bundle.
        if (future && preview->mode == 2 && !matrix_wave) {
            future = nullptr;
            if (profile) { ++profile->preview_skip_short; }
        }
        if (future) {
            llama_expert_timer guard(profile ? &profile->preview_guard_ns : nullptr);
            // A warm current working set is a conservative stop signal. It is
            // cheaper to miss a possible future opportunity than to keep
            // predicting and reading unused experts on an all-hot request.
            bool warm = true;
            if (routes) {
                for (size_t i = 0; i < routes->count && warm; ++i) { warm = store->contains(p.name, routes->active[i]); }
            } else {
                std::array<bool, 1024> checked {};
                for (int64_t row = 0; row < ids->ne[1] && warm; ++row) {
                    for (int64_t slot = 0; slot < ids->ne[0] && warm; ++slot) {
                        const size_t e = size_t(id_at(ids, slot, row));
                        if (profile) { ++profile->route_id_reads; }
                        if (!checked[e]) { warm = store->contains(p.name, int64_t(e)); checked[e] = true; }
                    }
                }
            }
            if (warm) { future = nullptr; if (profile) { ++profile->preview_skip_hot; } }
        }
        const size_t optional_limit = expert_capacity > groups + ahead_limit ?
            std::min(size_t(4), (expert_capacity - groups - ahead_limit) / 3) : 0;
        // These unused-by-GPU leases expire within this callback after all work
        // synchronizes. Only ordinary evictable residency crosses a graph or
        // resize boundary; no stale prediction/queue survives a microbatch.
        std::array<llama_expert_store::lease, 12> optional;
        std::array<llama_expert_store::lease, max_wave_groups> held;
        std::array<llama_expert_store::lease, max_wave_groups> ahead;
        std::array<int64_t, max_wave_groups> ahead_ids;
        ahead_ids.fill(-1);
        std::array<ggml_tensor *, max_wave_nodes> inputs {}, outputs {};
        std::array<size_t, max_wave_nodes> host_rows {};
        std::array<char *, max_wave_nodes> destinations {};
        std::array<uint64_t, max_wave_groups> weight_addresses;
        std::array<int32_t, max_wave_nodes> address_routes;
        ggml_cgraph * graph = nullptr;
        ggml_tensor * local_ids = nullptr;
        size_t waves = 0, nodes = 0, input_cursor = input_offset, output_cursor = output_offset;
        const size_t node_rows = vector_batch ? 1 : wave_rows;
        const auto buft = ggml_backend_buffer_get_type(buffer.get());
        const size_t alignment = ggml_backend_buft_get_alignment(buft);
        char * base = static_cast<char *>(ggml_backend_buffer_get_base(buffer.get()));
        auto bind = [&](ggml_tensor * tensor, size_t & offset, size_t limit) {
            const size_t bytes = ggml_backend_buft_get_alloc_size(buft, tensor);
            if (offset > limit || bytes > limit - offset) {
                throw std::runtime_error("indexed expert wave exceeds reserved workspace");
            }
            if (ggml_backend_tensor_alloc(buffer.get(), tensor, base + offset) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("cannot bind expert workspace tensors");
            }
            offset += (bytes + alignment - 1) / alignment * alignment;
        };
        const size_t advice_limit = decode_advice ? std::min(size_t(ids->ne[0]), groups) : ahead_limit;
        auto advise = [&](int64_t previous) {
            if (!file_advice || !advice_limit) { return; }
            // At most one known-route group: existing prefill lookahead, or the
            // current decode group after an opted-in layer window. This hints
            // the OS file cache, not our resident cache: no leases, user buffers,
            // policy touches or speculative experts. Demand reads remain checked.
            for (size_t n = 0; n < advice_limit; ++n) {
                int64_t next = p.experts;
                if (routes) { next = routes->after(previous, p.experts); }
                else {
                    if (profile) { profile->route_id_reads += route_count; }
                    for (int64_t token = 0; token < ids->ne[1]; ++token) {
                        for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                            const int64_t id = id_at(ids, slot, token);
                            if (id > previous) { next = std::min(next, id); }
                        }
                    }
                }
                if (next == p.experts) { break; }
                previous = next;
                store->advise_expert(p.name, next);
            }
        };
        advise(-1);
        auto flush = [&](int64_t after_expert) {
            if (!waves) { return; }
            llama_expert_timer timer(profile ? &profile->input_ns : nullptr);
            ggml_tensor * batched_output = nullptr;
            if (address_batch) {
                // Same one-row indexed Metal arithmetic, with one grid over
                // the routes and an address per leased weight. No weight copies.
                auto * x = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, p.input, nodes, 1);
                auto * indices = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, nodes, 1);
                auto * table = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, waves);
                size_t map_cursor = ids_offset;
                bind(x, input_cursor, ids_offset);
                bind(indices, map_cursor, output_offset);
                bind(table, map_cursor, output_offset);
                ggml_backend_tensor_set(indices, address_routes.data(), 0, nodes * sizeof(int32_t));
                ggml_backend_tensor_set(table, weight_addresses.data(), 0, waves * sizeof(uint64_t));
                auto * anchor = reinterpret_cast<ggml_tensor *>(uintptr_t(weight_addresses[0]));
                batched_output = ggml_mul_mat_id(ctx.get(), anchor, x, indices);
                if (!window_addresses(batched_output, table) || !ggml_backend_supports_op(backend.get(), batched_output)) {
                    throw std::runtime_error("Metal rejected the prefill address batch");
                }
                bind(batched_output, output_cursor, ggml_backend_buffer_get_size(buffer.get()));
                for (size_t n = 0; n < nodes; ++n) {
                    // Compaction is forward and wave source offsets never lag
                    // their dense destination.
                    if (n != host_rows[n]) {
                        std::memmove(host.get() + n * p.input, host.get() + host_rows[n] * p.input, p.input * sizeof(float));
                    }
                }
                ggml_backend_tensor_set(x, host.get(), 0, nodes * p.input * sizeof(float));
                ggml_build_forward_expand(graph, batched_output);
                if (profile) { ++profile->moe_batch_groups; }
            } else {
                for (size_t n = 0; n < nodes; ++n) {
                    ggml_backend_tensor_set(inputs[n], host.get() + host_rows[n] * p.input, 0, node_rows * p.input * sizeof(float));
                }
            }
            timer.next(profile ? &profile->compute_ns : nullptr);
            llama_expert_timer submit_timer(profile ? &profile->submit_ns : nullptr);
            ggml_status status;
            if (discrete.device) {
                // The CPU callback can run on a different thread from the outer decode.
                llama_allocation_scope scratch(budget, llama_memory_category::scratch,
                        [this] { store->evict_idle(); }, discrete);
                status = ggml_backend_graph_compute_async(backend.get(), graph);
            } else {
                status = ggml_backend_graph_compute_async(backend.get(), graph);
            }
            submit_timer.next(nullptr);
            // All routes are already known. Read only subsequent demand experts
            // while the submitted graph retains its own leases.
            // No graph/workspace mutation or second backend submission occurs.
            bool preview_window = false, early_window = false;
            try {
                if (status == GGML_STATUS_SUCCESS) {
                    if (early_target && after_expert >= final_expert) {
                        // No remaining source-expert reads compete with this window.
                        // Leases protect only exact target demand and expire here,
                        // after GPU completion; only evictable residency survives.
                        if (std::any_of(ahead.begin(), ahead.end(), [](const auto & lease) { return bool(lease); })) {
                            throw std::runtime_error("early read encountered pending source experts");
                        }
                        early_window = true;
                        if (profile) { ++profile->early_windows; }
                        llama_expert_timer early_timer(profile ? &profile->early_acquire_ns : nullptr);
                        llama_expert_timer acquire_timer(profile ? &profile->acquire_ns : nullptr);
                        for (size_t i = 0; i < early_limit && first_ids[i] < p.experts; ++i) {
                            if (store->contains(early_target->name, first_ids[i])) {
                                if (profile) { ++profile->early_skip_hot; }
                                continue;
                            }
                            const auto before = profile ? profile->direct_read_bytes : 0;
                            early_leases[i] = store->acquire(early_target->name, first_ids[i]);
                            if (profile) {
                                ++profile->early_acquisitions;
                                profile->early_read_bytes += profile->direct_read_bytes - before;
                            }
                        }
                        early_target = nullptr;
                    }
                    if (future && (preview->mode != 2 || after_expert >= final_expert)) {
                        preview_window = true;
                        if (profile) { ++profile->preview_windows; }
                        llama_expert_timer window(profile ? &profile->preview_window_host_ns : nullptr);
                        run_preview(*future, input, optional_limit, optional);
                        future = nullptr;
                    }
                    advise(after_expert);
                    int64_t previous = after_expert;
                    std::array<int64_t, max_wave_groups> queued_ids;
                    std::array<size_t, max_wave_groups> queued_slots;
                    size_t queued = 0;
                    for (size_t n = 0; n < ahead_limit; ++n) {
                        int64_t next = p.experts;
                        if (routes) { next = routes->after(previous, p.experts); }
                        else {
                            if (profile) { profile->route_id_reads += route_count; }
                            for (int64_t token = 0; token < ids->ne[1]; ++token) {
                                for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                                    const int64_t id = id_at(ids, slot, token);
                                    if (id > previous) { next = std::min(next, id); }
                                }
                            }
                        }
                        if (next == p.experts) { break; }
                        if (preview_window && profile && previous == after_expert) { ++profile->preview_competing_windows; }
                        previous = next;
                        if (std::find(ahead_ids.begin(), ahead_ids.end(), next) != ahead_ids.end()) { continue; }
                        auto empty = std::find(ahead_ids.begin(), ahead_ids.end(), int64_t(-1));
                        if (empty == ahead_ids.end()) { break; }
                        if (parallel_ahead) {
                            queued_ids[queued] = next;
                            queued_slots[queued++] = size_t(empty - ahead_ids.begin());
                            *empty = next;
                            continue;
                        }
                        llama_expert_timer acquire_timer(profile ? &profile->acquire_ns : nullptr);
                        llama_expert_timer overlap_timer(profile ? &profile->overlap_acquire_ns : nullptr);
                        const auto before = profile ? profile->direct_read_bytes : 0;
                        ahead[size_t(empty - ahead_ids.begin())] = store->acquire(p.name, next);
                        *empty = next;
                        if (profile) {
                            ++profile->overlap_acquisitions;
                            profile->overlap_read_bytes += profile->direct_read_bytes - before;
                        }
                    }
                    if (queued) {
                        std::array<llama_expert_store::lease, max_wave_groups> ready;
                        llama_expert_timer acquire_timer(profile ? &profile->acquire_ns : nullptr);
                        llama_expert_timer overlap_timer(profile ? &profile->overlap_acquire_ns : nullptr);
                        const auto before = profile ? profile->direct_read_bytes : 0;
                        store->acquire_many(p.name, queued_ids.data(), queued, ready.data());
                        for (size_t i = 0; i < queued; ++i) { ahead[queued_slots[i]] = std::move(ready[i]); }
                        if (profile) {
                            profile->overlap_acquisitions += queued;
                            profile->overlap_read_bytes += profile->direct_read_bytes - before;
                        }
                    }
                }
            } catch (...) {
                // The graph's arenas and every referenced lease must outlive GPU
                // completion even if a subsequent source read fails mid-group.
                ggml_backend_synchronize(backend.get());
                throw;
            }
            submit_timer.next(profile ? &profile->wait_ns : nullptr);
            ggml_backend_synchronize(backend.get());
            submit_timer.next(nullptr);
            if (status != GGML_STATUS_SUCCESS) { throw std::runtime_error("streamed expert computation failed"); }
            if (profile && graph_timing) {
                uint64_t gpu_ns = 0, buffers = 0;
                if (graph_timing(backend.get(), &gpu_ns, &buffers)) {
                    profile->gpu_command_ns += gpu_ns;
                    if (preview_window) { profile->preview_window_gpu_ns += gpu_ns; }
                    if (early_window) { profile->early_window_gpu_ns += gpu_ns; }
                    profile->gpu_command_buffers += buffers;
                    ++profile->gpu_timed_submissions;
                }
            }
            timer.next(profile ? &profile->readback_ns : nullptr);
            if (batched_output) {
                ggml_backend_tensor_get(batched_output, host.get() + host_output, 0, nodes * p.output * sizeof(float));
                timer.next(profile ? &profile->scatter_ns : nullptr);
                for (size_t n = 0; n < nodes; ++n) {
                    std::memcpy(destinations[host_rows[n]], host.get() + host_output + n * p.output, p.output * sizeof(float));
                }
            } else {
                for (size_t n = 0; n < nodes; ++n) {
                    ggml_backend_tensor_get(outputs[n], host.get() + host_output + host_rows[n] * p.output, 0,
                                            node_rows * p.output * sizeof(float));
                }
                timer.next(profile ? &profile->scatter_ns : nullptr);
                for (size_t r = 0; r < waves * max_wave_rows; ++r) {
                    if (destinations[r]) {
                        std::memcpy(destinations[r], host.get() + host_output + r * p.output, p.output * sizeof(float));
                    }
                }
            }
            n_matmuls += nodes;
            ++n_submissions;
            timer.next(nullptr);
            // Discard every graph pointer before completed leases become reusable.
            ggml_reset(ctx.get());
            graph = nullptr;
            for (auto & lease : held) { lease.reset(); }
            waves = nodes = 0;
            input_cursor = input_offset; output_cursor = output_offset;
            destinations.fill(nullptr);
        };
        for (int64_t position = 0; position < (routes ? routes->count : p.experts); ++position) {
            const int64_t expert = routes ? routes->active[size_t(position)] : position;
            // This lease also survives group flushes when one expert spans waves.
            llama_expert_store::lease current;
            size_t rows = 0;
            auto enqueue = [&]() {
                if (!rows) { return; }
                llama_expert_timer timer(profile ? &profile->graph_setup_ns : nullptr);
                if (!graph) {
                    graph = ggml_new_graph_custom(ctx.get(), metal ? 4 * max_wave_rows * workspace_groups : 8, false);
                    if (!address_batch) {
                        local_ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 1, node_rows);
                        size_t cursor = ids_offset;
                        bind(local_ids, cursor, output_offset);
                        const std::array<int32_t, max_wave_rows> zeros {};
                        ggml_backend_tensor_set(local_ids, zeros.data(), 0, node_rows * sizeof(int32_t));
                    }
                }
                const size_t row_base = waves * max_wave_rows;
                if (!vector_batch) {
                    std::fill(host.get() + (row_base + rows) * p.input,
                              host.get() + (row_base + wave_rows) * p.input, 0.0f);
                }
                const size_t count = vector_batch ? rows : 1;
                if (address_batch) { weight_addresses[waves] = uintptr_t(current.tensor()); }
                for (size_t n = 0; n < count; ++n) {
                    if (address_batch) {
                        address_routes[nodes] = int32_t(waves);
                        host_rows[nodes++] = row_base + n;
                        continue;
                    }
                    inputs[nodes] = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, p.input, 1, node_rows);
                    outputs[nodes] = ggml_mul_mat_id(ctx.get(), current.tensor(), inputs[nodes], local_ids);
                    if (!ggml_backend_supports_op(backend.get(), outputs[nodes])) {
                        throw std::runtime_error("backend does not support the streamed expert projection");
                    }
                    bind(inputs[nodes], input_cursor, ids_offset);
                    bind(outputs[nodes], output_cursor, ggml_backend_buffer_get_size(buffer.get()));
                    if (single_expert_map && matrix_wave && prepare_map) {
                        llama_expert_timer prepare_timer(profile ? &profile->map_prepare_ns : nullptr);
                        if (prepare_map(outputs[nodes]) && profile) { ++profile->prepared_map_ops; }
                    }
                    ggml_build_forward_expand(graph, outputs[nodes]);
                    host_rows[nodes++] = row_base + n;
                }
                ++waves;
                rows = 0;
                timer.next(nullptr);
                if (waves == groups) { flush(expert); }
            };
            auto gather = [&](int64_t token, int64_t slot) {
                if (!current) {
                    llama_expert_timer timer(profile ? &profile->acquire_ns : nullptr);
                    const auto ready = std::find(ahead_ids.begin(), ahead_ids.end(), expert);
                    if (ready != ahead_ids.end()) {
                        current = std::move(ahead[size_t(ready - ahead_ids.begin())]);
                        *ready = -1;
                    } else {
                        current = store->acquire(p.name, expert);
                    }
                }
                llama_expert_timer gather_timer(profile ? &profile->gather_ns : nullptr);
                const int64_t input_slot = input->ne[1] == 1 ? 0 : slot;
                const char * row = static_cast<const char *>(input->data) + input_slot * input->nb[1] + token * input->nb[2];
                const size_t index = waves * max_wave_rows + rows;
                if (input->nb[0] == sizeof(float)) {
                    std::memcpy(host.get() + index * p.input, row, p.input * sizeof(float));
                } else {
                    for (int64_t col = 0; col < p.input; ++col) {
                        std::memcpy(host.get() + index * p.input + col, row + col * input->nb[0], sizeof(float));
                    }
                }
                destinations[index] = static_cast<char *>(dst->data) + slot * dst->nb[1] + token * dst->nb[2];
                ++rows;
                gather_timer.next(nullptr);
                if (rows == wave_rows) { enqueue(); }
            };
            if (routes) {
                for (uint16_t at = routes->first[expert]; at != llama_expert_route_index::end; at = routes->next[at]) {
                    gather(at / ids->ne[0], at % ids->ne[0]);
                }
            } else {
                if (profile) { profile->route_id_reads += route_count; }
                for (int64_t token = 0; token < ids->ne[1]; ++token) {
                    for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                        if (id_at(ids, slot, token) == expert) { gather(token, slot); }
                    }
                }
            }
            enqueue();
            if (current && waves) {
                auto slot = std::find_if(held.begin(), held.end(), [](const auto & lease) { return !lease; });
                if (slot == held.end()) { throw std::runtime_error("expert group lease capacity exceeded"); }
                *slot = std::move(current);
            }
        }
        flush(p.experts);
        ++n_projections;
    }

    static void rows_callback(ggml_tensor * dst, int ith, int, void * userdata) noexcept {
        if (ith != 0) { return; }
        auto & owner = *static_cast<impl *>(userdata);
        if (!owner.failed.load()) {
            try {
                const auto * ids = dst->src[0];
                const auto * table = owner.row_table;
                const auto dequantize = ggml_get_type_traits(table->type)->to_float;
                for (int64_t i = 0; i < ids->ne[0]; ++i) {
                    int32_t row;
                    std::memcpy(&row, static_cast<const char *>(ids->data) + i * ids->nb[0], sizeof(row));
                    owner.row_source->read_row(table->name, row, owner.row_buffer.get(), owner.row_charge.size());
                    dequantize(owner.row_buffer.get(),
                        reinterpret_cast<float *>(static_cast<char *>(dst->data) + i * dst->nb[1]), table->ne[0]);
                }
                return;
            } catch (const std::exception & e) {
                std::snprintf(owner.failure, sizeof(owner.failure), "%s", e.what());
            } catch (...) {
                std::snprintf(owner.failure, sizeof(owner.failure), "unknown streamed row failure");
            }
            owner.failed.store(true);
        }
        std::memset(dst->data, 0, ggml_nbytes(dst));
    }

    static void token_rows_callback(ggml_tensor * dst, int ith, int, void * userdata) noexcept {
        if (ith != 0) { return; }
        auto & owner = *static_cast<impl *>(userdata);
        if (!owner.failed.load()) {
            try {
                const auto * ids = dst->src[0];
                const auto * table = owner.token_row_table;
                const auto dequantize = ggml_get_type_traits(table->type)->to_float;
                for (int64_t i = 0; i < ids->ne[0]; ++i) {
                    int32_t row;
                    std::memcpy(&row, static_cast<const char *>(ids->data) + i * ids->nb[0], sizeof(row));
                    owner.token_row_source->read_row(table->name, row, owner.token_row_buffer.get(), owner.token_row_charge.size());
                    dequantize(owner.token_row_buffer.get(),
                        reinterpret_cast<float *>(static_cast<char *>(dst->data) + i * dst->nb[1]), table->ne[0]);
                }
                return;
            } catch (const std::exception & e) {
                std::snprintf(owner.failure, sizeof(owner.failure), "%s", e.what());
            } catch (...) {
                std::snprintf(owner.failure, sizeof(owner.failure), "unknown streamed token row failure");
            }
            owner.failed.store(true);
        }
        std::memset(dst->data, 0, ggml_nbytes(dst));
    }

    static void callback(ggml_tensor * dst, int ith, int, void * userdata) noexcept {
        if (ith != 0) { return; }
        const auto & p = *static_cast<projection *>(userdata);
        auto & owner = *p.owner;
        if (owner.cross_window_ready == dst) {
            owner.cross_window_ready = nullptr;
            return;
        }
        if (!owner.failed.load()) {
            try { owner.compute(p, dst); return; }
            catch (const std::exception & e) { std::snprintf(owner.failure, sizeof(owner.failure), "%s", e.what()); }
            catch (...) { std::snprintf(owner.failure, sizeof(owner.failure), "unknown expert execution failure"); }
            owner.failed.store(true);
        }
        // Never propagate a C++ exception through backend worker threads. The
        // context rejects this entire evaluation and all subsequent evaluations.
        std::memset(dst->data, 0, ggml_nbytes(dst));
    }
};

llama_expert_executor::llama_expert_executor(std::shared_ptr<llama_expert_source> source,
        const std::vector<const ggml_tensor *> & weights, ggml_backend_dev_t device, size_t budget_bytes,
        std::shared_ptr<llama_memory_budget> shared_budget, const llama_memory_contract * discrete_contract,
        const std::vector<const ggml_tensor *> & cpu_weights) :
    pimpl(new impl(std::move(source), weights, device, budget_bytes, std::move(shared_budget), discrete_contract, cpu_weights)) {}
llama_expert_executor::~llama_expert_executor() = default;

void llama_expert_executor::configure_rows(std::shared_ptr<llama_expert_source> source, const ggml_tensor * table) {
    if (pimpl->row_table || !source || !table || (!pimpl->metal && !pimpl->discrete.device) ||
        (pimpl->discrete.device && source->architecture() != "qwen4exp") ||
        table->type != GGML_TYPE_IQ4_NL || table->ne[2] != 1 || table->ne[3] != 1 ||
        !ggml_get_type_traits(table->type)->to_float) {
        throw std::invalid_argument("unsupported streamed row configuration");
    }
    const auto & info = source->tensor(table->name);
    if (info.type != table->type || !std::equal(info.ne.begin(), info.ne.end(), table->ne)) {
        throw std::invalid_argument("streamed row source layout mismatch");
    }
    const size_t bytes = ggml_row_size(table->type, table->ne[0]);
    auto charge = admitted(*pimpl->budget, llama_memory_category::io, bytes);
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[bytes]);
    charge.commit();
    pimpl->row_charge = std::move(charge);
    pimpl->row_buffer = std::move(buffer);
    pimpl->row_source = std::move(source);
    pimpl->row_table = table;
    // This persistent row is included by stats().minimum_bytes through used().
}

ggml_tensor * llama_expert_executor::build_rows(ggml_context * ctx, ggml_tensor * ids) {
    if (!pimpl->row_table || ids->type != GGML_TYPE_I32 || !ggml_is_vector(ids)) {
        throw std::invalid_argument("invalid streamed row indices");
    }
    ggml_tensor * args[] = {ids};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, pimpl->row_table->ne[0], ids->ne[0], 1, 1,
                          args, 1, impl::rows_callback, 1, pimpl.get());
}

void llama_expert_executor::configure_token_rows(std::shared_ptr<llama_expert_source> source, const ggml_tensor * table) {
    if (pimpl->token_row_table || !source || !table || !pimpl->qwen_cuda || source->architecture() != "qwen4exp" ||
        table->type != GGML_TYPE_IQ3_S || table->ne[2] != 1 || table->ne[3] != 1 ||
        !ggml_get_type_traits(table->type)->to_float) {
        throw std::invalid_argument("unsupported streamed token row configuration");
    }
    const auto & info = source->tensor(table->name);
    if (info.type != table->type || !std::equal(info.ne.begin(), info.ne.end(), table->ne)) {
        throw std::invalid_argument("streamed token row source layout mismatch");
    }
    const size_t bytes = ggml_row_size(table->type, table->ne[0]);
    auto charge = admitted(*pimpl->budget, llama_memory_category::io, bytes);
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[bytes]);
    charge.commit();
    pimpl->token_row_charge = std::move(charge);
    pimpl->token_row_buffer = std::move(buffer);
    pimpl->token_row_source = std::move(source);
    pimpl->token_row_table = table;
    // This persistent row is included by stats().minimum_bytes through used().
}

ggml_tensor * llama_expert_executor::build_token_rows(ggml_context * ctx, ggml_tensor * ids) {
    if (!pimpl->token_row_table || ids->type != GGML_TYPE_I32 || !ggml_is_vector(ids)) {
        throw std::invalid_argument("invalid streamed token row indices");
    }
    ggml_tensor * args[] = {ids};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, pimpl->token_row_table->ne[0], ids->ne[0], 1, 1,
                          args, 1, impl::token_rows_callback, 1, pimpl.get());
}

ggml_tensor * llama_expert_executor::build(ggml_context * ctx, const ggml_tensor * weight, ggml_tensor * input, ggml_tensor * ids) {
    // The checked vector profiles match native CUDA. The 32-row IQ4_NL
    // matrix path differs in low bits; do not silently admit that profile.
    if (pimpl->qwen_cuda && ids->ne[1] > 8) {
        throw std::runtime_error("Qwen4Exp CUDA streaming requires a microbatch of at most 8 tokens");
    }
    auto it = pimpl->projections.find(weight);
    if (it == pimpl->projections.end()) { throw std::runtime_error("unregistered streamed expert tensor"); }
    ggml_tensor * args[] = {input, ids};
    return ggml_custom_4d(ctx, GGML_TYPE_F32, weight->ne[1], ids->ne[0], ids->ne[1], 1,
                          args, 2, impl::callback, 1, &it->second);
}

const char * llama_expert_executor::error() const { return pimpl->failed.load() ? pimpl->failure : nullptr; }
void llama_expert_executor::configure_acceleration(bool dflash_verify, bool profile_overlap) {
    pimpl->profile_overlap = profile_overlap;
    pimpl->dflash_verify = dflash_verify;
    pimpl->configure_acceleration();
}
void llama_expert_executor::configure_resident_layers(const llama_model & model) {
    if (!pimpl->moe || !pimpl->automatic_acceleration || !model.memory_contract) { return; }
    auto resident = std::make_unique<llama_lfm_resident>(model.hparams.n_layer_dense_lead);
    const auto state = stats();
    // The 512-row target batch leaves room for the final expert bank. Each
    // allocation still needs an individual ticket from the shared ledger.
    size_t allowance = llama_lfm_resident::allowance(state.memory.applied, state.minimum_bytes);
    if (model.memory_contract->qwen35_resident_verify && state.memory.applied == (size_t(24) << 30)) {
        allowance += size_t(512) << 20;
    }
    resident->load(model, ggml_backend_get_device(pimpl->backend.get()), allowance,
            model.memory_contract->qwen35_profile_layers ? model.memory_contract->qwen35_profile_layers : model.hparams.n_layer());
    std::fprintf(stderr, "%s resident layers: %zu (%zu bytes)\n", model.arch == LLM_ARCH_QWEN35MOE ? "Qwen" : "LFM", resident->count(), resident->bytes());
    pimpl->lfm_resident = std::move(resident);
}

const std::array<ggml_tensor *, 3> * llama_expert_executor::resident_layer(int il) const {
    return pimpl->lfm_resident && !pimpl->layer_window ? pimpl->lfm_resident->get(il) : nullptr;
}

bool llama_expert_executor::can_retain_prefill_tail(size_t routed_layers) const {
    return pimpl->lfm_resident && pimpl->lfm_resident->count() < routed_layers &&
            !pimpl->lfm_resident->has_tail();
}

size_t llama_expert_executor::retain_prefill_tail(llama_prefill_wide_weights & weights, int first_layer) {
    GGML_ASSERT(pimpl->lfm_resident && !pimpl->layer_window);
    auto & resident = *pimpl->lfm_resident;
    return resident.adopt_tail(first_layer, weights.take_tail_banks(*pimpl->budget));
}

bool llama_expert_executor::has_prefill_tail() const {
    return pimpl->lfm_resident && pimpl->lfm_resident->has_tail();
}

void llama_expert_executor::release_prefill_tail() {
    pimpl->lfm_resident->release_tail();
}

bool llama_expert_executor::automatic_acceleration() const { return pimpl->automatic_acceleration; }
bool llama_expert_executor::combined_moe_enabled() const { return pimpl->moe && !pimpl->layer_window; }
bool llama_expert_executor::fused_window_moe_enabled() const { return pimpl->fused_window_moe; }

ggml_tensor * llama_expert_executor::build_moe(ggml_context * ctx, const ggml_tensor * up,
        const ggml_tensor * gate, const ggml_tensor * down, ggml_tensor * input, ggml_tensor * ids,
        const ggml_tensor * router, const std::array<const ggml_tensor *, 3> & future, ggml_backend_sched_t predictor_sched) {
    if (!pimpl->moe || pimpl->tracing || pimpl->class_policy || (pimpl->layer_window && !pimpl->fused_window_moe) || pimpl->route_observer ||
            pimpl->preview || pimpl->early || !pimpl->projections.count(up) ||
            !pimpl->projections.count(gate) || !pimpl->projections.count(down)) { return nullptr; }
    if (router && std::any_of(future.begin(), future.end(), [&](const ggml_tensor * w) {
            return !pimpl->projections.count(w);
        })) { router = nullptr; }
    return pimpl->moe->build(ctx, up, gate, down, input, ids, router, future, predictor_sched, pimpl->fused_window_moe);
}
bool llama_expert_executor::moe_route_published(const ggml_tensor * ids) const { return combined_moe_enabled() && pimpl->moe->is_published(ids); }
bool llama_expert_executor::moe_shared_publication(const ggml_tensor * up) const { return combined_moe_enabled() && pimpl->moe->shared_publication(up); }
bool llama_expert_executor::moe_graph_lookahead_enabled() const { return combined_moe_enabled() && pimpl->moe->has_graph_lookahead(); }
bool llama_expert_executor::moe_lookahead_enabled() const { return combined_moe_enabled() && pimpl->moe->has_lookahead(); }
void llama_expert_executor::moe_lookahead_begin(bool decode) { pimpl->moe->begin_graph(decode); }
void llama_expert_executor::moe_lookahead_end() { pimpl->moe->end_graph(); }

llama_expert_execution_stats llama_expert_executor::stats() const {
    auto store = pimpl->store->stats();
    if (pimpl->cpu_store) {
        const auto host = pimpl->cpu_store->stats();
        store.hits += host.hits;
        store.misses += host.misses;
        store.evictions += host.evictions;
        store.logical_read_bytes += host.logical_read_bytes;
        store.buffer_allocations += host.buffer_allocations;
        store.buffer_reuses += host.buffer_reuses;
    }
    const auto memory = pimpl->budget->snapshot(llama_memory_domain::host);
    return {store, memory, pimpl->n_projections, pimpl->n_matmuls,
            memory.used() - (pimpl->discrete.device ? store.entries * ggml_tensor_overhead() : store.resident_bytes + store.pooled_bytes) +
                pimpl->minimum - pimpl->fixed_workspace - pimpl->optional_bytes() -
                (pimpl->lfm_resident ? pimpl->lfm_resident->bytes() : 0), pimpl->n_submissions,
                pimpl->retired_moe_calls + (pimpl->moe ? pimpl->moe->completed() : 0),
            pimpl->retired_pipeline_groups + (pimpl->moe ? pimpl->moe->overlapped_groups() : 0),
            pimpl->retired_resident_groups + (pimpl->moe ? pimpl->moe->resident_starts() : 0),
            pimpl->optional_bytes(), pimpl->moe ? pimpl->moe->prefetch_candidates() : 0};
}
bool llama_expert_executor::resize(size_t target) {
    if (pimpl->discrete.device || error()) { return false; }
    const bool drop_history = target != class_lru_budget && pimpl->class_policy && !pimpl->policy_override;
    const size_t optional_bytes = drop_history ? pimpl->store->class_lru_history_bytes() : 0;
    const auto state = stats();
    if (target < state.minimum_bytes - optional_bytes) { return false; }
    // The context has synchronized and invalidated graphs before resizing.
    // Only permanent banks participate in the new budget's prefix allowance.
    if (target != state.memory.applied && has_prefill_tail()) { release_prefill_tail(); }
    if (drop_history) { select_decode_policy(0, 0); }
    const bool shrinking_lfm = target < state.memory.applied && pimpl->automatic_acceleration && pimpl->moe &&
            (pimpl->lfm_resident || pimpl->acceleration_source->architecture() == "lfm2moe");
    const size_t combined_room = shrinking_lfm ? state.acceleration_bytes + pimpl->moe->admission_working_bytes() : 0;
    if (pimpl->lfm_resident && target < state.memory.applied) {
        pimpl->lfm_resident->shrink(llama_lfm_resident::allowance(target, state.minimum_bytes));
        std::fprintf(stderr, "%s resident layers: %zu (%zu bytes)\n",
                pimpl->acceleration_source->architecture() == "lfm2moe" ? "LFM" : "Qwen", pimpl->lfm_resident->count(), pimpl->lfm_resident->bytes());
    }
    pimpl->drop_acceleration();
    pimpl->store->request_budget(llama_memory_domain::host, target);
    const bool applied = !pimpl->budget->snapshot(llama_memory_domain::host).pending();
    if (applied) {
        if (combined_room && target >= state.minimum_bytes && combined_room <= target - state.minimum_bytes) {
            pimpl->store->reclaim_to_fit(llama_memory_domain::host, combined_room);
        }
        pimpl->configure_acceleration();
    }
    return applied;
}
bool llama_expert_executor::resize_discrete(size_t host_target, size_t device_target, size_t scratch_headroom) {
    if (!pimpl->discrete.device || error()) { return false; }
    const auto cache = pimpl->store->stats();
    const auto cpu = pimpl->cpu_store ? pimpl->cpu_store->stats() : llama_expert_store_stats{};
    if (cache.pinned_entries || cpu.entries) { return false; }

    const size_t cache_metadata = cache.entries * ggml_tensor_overhead();
    auto & budget = *pimpl->budget;
    const auto ram = llama_memory_domain::host;
    const auto vram = llama_memory_domain::device;
    const size_t host_fixed = budget.snapshot(ram).used() - cache_metadata;
    const size_t device_fixed = budget.snapshot(vram).used() - (cache.resident_bytes - cache_metadata);
    const auto fits = [scratch_headroom](size_t cap, size_t fixed, size_t working) {
        return cap >= fixed && cap - fixed >= working && cap - fixed - working >= scratch_headroom;
    };
    if (!fits(host_target, host_fixed, pimpl->minimum - pimpl->fixed_workspace) ||
        !fits(device_target, device_fixed, pimpl->max_expert_allocation)) {
        return false;
    }

    // No leases survive a synchronized projection. Reclaim device entries
    // first so their host metadata can also help satisfy the RAM cap.
    pimpl->store->request_budget(vram, device_target);
    pimpl->store->request_budget(ram, host_target);
    return !budget.snapshot(ram).pending() && !budget.snapshot(vram).pending();
}
void llama_expert_executor::set_discrete_read_probe(bool (*callback)(void *) noexcept, void * user, bool worker,
        llama_expert_read_probe submitted_read) {
    pimpl->store->set_read_probe(callback, user, worker, submitted_read);
}
llama_memory_reclaimer llama_expert_executor::allocation_reclaimer() {
    if (!pimpl->selective_reclaim || pimpl->tracing) { return {}; }
    return {[](void * user, llama_memory_domain domain, size_t bytes) {
        auto & self = *static_cast<impl *>(user);
        self.store->reclaim_to_fit(domain, bytes);
    }, pimpl.get()};
}

void llama_expert_executor::trim() {
    pimpl->store->trim();
    if (pimpl->cpu_store) { pimpl->cpu_store->trim(); }
}

void llama_expert_executor::set_trace(llama_expert_trace_callback callback, void * user) {
    if (callback && pimpl->moe) { throw std::invalid_argument("combined MoE does not support demand trace mode"); }
    if (callback && pimpl->discrete.device) { throw std::invalid_argument("expert trace requires unified memory"); }
    if (callback && (pimpl->preview || pimpl->early)) { throw std::invalid_argument("early reads and preview cannot use demand-only ATET tracing"); }
    if (callback && pimpl->class_policy && !pimpl->policy_override) { select_decode_policy(0, 0); }
    pimpl->tracing = callback != nullptr;
    pimpl->store->set_trace(callback, user);
}

bool llama_expert_executor::set_policy(const std::string & policy) {
    if (pimpl->discrete.device) { return policy == "lru"; }
    if ((pimpl->preview || pimpl->early || pimpl->moe) && policy != "lru") { return false; }
    if (!pimpl->store->set_policy(policy, pimpl->minimum - pimpl->fixed_workspace)) { return false; }
    pimpl->class_policy = policy != "lru";
    pimpl->policy_override = true;
    return true;
}

bool llama_expert_executor::select_decode_policy(size_t tokens, size_t ubatch) {
    if (pimpl->policy_override || !pimpl->vector_read_overlap || pimpl->tracing) { return pimpl->class_policy; }
    // Qualify the measured Qwen4Exp Metal profile at the logical batch boundary.
    bool use_class = tokens >= class_lru_prefill_tokens && ubatch == 8 &&
        pimpl->read_overlap && !pimpl->preview && !pimpl->early && !error();
    if (use_class) {
        const auto budget = pimpl->budget->snapshot(llama_memory_domain::host);
        use_class = budget.requested == class_lru_budget && budget.applied == class_lru_budget;
    }
    if (use_class == pimpl->class_policy) { return pimpl->class_policy; }
    try {
        if (pimpl->store->set_prefill_class_lru(use_class, pimpl->minimum - pimpl->fixed_workspace)) {
            pimpl->class_policy = use_class;
        }
    } catch (const std::bad_alloc &) {
        // History is optional. Its reservation rolls back if allocation fails.
    }
    return pimpl->class_policy;
}

void llama_expert_executor::set_profile(bool enabled) {
    pimpl->profile_data = {};
    pimpl->profile = enabled ? &pimpl->profile_data : nullptr;
    pimpl->store->set_profile(pimpl->profile);
}
bool llama_expert_executor::can_layer_window() const {
    return (pimpl->vector_read_overlap || pimpl->automatic_acceleration) && !pimpl->policy_override && !pimpl->tracing &&
            !pimpl->preview && !pimpl->early && (!pimpl->moe || pimpl->automatic_acceleration) && !error();
}
bool llama_expert_executor::can_batch_window() const {
    return can_layer_window() && pimpl->automatic_acceleration && pimpl->acceleration_source &&
           !pimpl->cpu_store && !pimpl->route_observer;
}

bool llama_expert_executor::window_weights_fit_without_eviction() const {
    const auto snapshot = pimpl->budget->snapshot(llama_memory_domain::host);
    const size_t limit = std::min(snapshot.applied, snapshot.requested);
    const size_t fixed = snapshot.used() - pimpl->store->storage_bytes();
    return pimpl->full_expert_storage && fixed < limit && pimpl->max_expert_payload < limit - fixed &&
        pimpl->full_expert_storage <= limit - fixed - pimpl->max_expert_payload;
}

bool llama_expert_executor::can_batch_window_layer(const std::array<const ggml_tensor *, 3> & weights) const {
    return weights[0] && std::all_of(weights.begin(), weights.end(), [&](const ggml_tensor * weight) {
        const auto projection = pimpl->projections.find(weight);
        return projection != pimpl->projections.end() && !projection->second.on_cpu &&
               projection->second.experts == weights[0]->ne[2];
    });
}

const char * llama_expert_executor::window_projection_name(const ggml_tensor * node) const {
    if (pimpl->fused_window_moe) { return pimpl->moe->window_weight_name(node); }
    if (node->op != GGML_OP_CUSTOM) { return nullptr; }
    ggml_custom_op_params params;
    std::memcpy(&params, node->op_params, sizeof(params));
    if (params.fun != impl::callback) { return nullptr; }
    const auto & projection = *static_cast<const impl::projection *>(params.userdata);
    return projection.owner == pimpl.get() ? projection.name.c_str() : nullptr;
}

bool llama_expert_executor::prepare_cross_window(size_t rows, size_t routes) {
    if (!pimpl->layer_window || !pimpl->window_addresses || !rows || !routes ||
            rows > cross_window_workers * 8 || routes > 32 || pimpl->cross_window_charge) { return false; }
    size_t width = 0;
    for (const auto & entry : pimpl->projections) {
        const auto & projection = entry.second;
        width = std::max(width, size_t(std::max(projection.input, projection.output)));
    }
    if (!width || width > 8192) { return false; }
    const size_t lanes = rows * routes;
    auto charge = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::scratch,
            lanes * (2 * width * sizeof(float) + sizeof(int32_t)));
    if (!charge) { return false; }
    try {
        auto input = std::unique_ptr<float[]>(new float[lanes * width]);
        auto output = std::unique_ptr<float[]>(new float[lanes * width]);
        auto ids = std::unique_ptr<int32_t[]>(new int32_t[lanes]);
        charge.commit();
        pimpl->cross_window_charge = std::move(charge);
        pimpl->cross_window_input = std::move(input);
        pimpl->cross_window_output = std::move(output);
        pimpl->cross_window_ids = std::move(ids);
        pimpl->cross_window_width = width;
        pimpl->cross_window_routes = lanes;
    } catch (const std::bad_alloc &) { return false; }
    return true;
}

bool llama_expert_executor::compute_cross_window(ggml_tensor * const * nodes, size_t count) {
    if (count < 2 || count > cross_window_workers || !pimpl->cross_window_charge || error()) { return false; }
    const auto * first = nodes[0];
    const char * weight = window_projection_name(first);
    if (!weight) { return false; }
    const auto * x = first->src[0];
    const auto * ids = first->src[1];
    const int64_t k = ids->ne[0], input_slots = x->ne[1];
    if (k < 1 || (input_slots != 1 && input_slots != k)) { return false; }
    int64_t rows = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto * node = nodes[i];
        const char * name = window_projection_name(node);
        if (!name || name != weight || node->src[0]->ne[2] < 2 ||
                node->src[0]->ne[1] != input_slots || node->src[1]->ne[0] != k ||
                node->src[0]->type != GGML_TYPE_F32 || node->src[1]->type != GGML_TYPE_I32 ||
                node->src[0]->nb[0] != sizeof(float) || node->src[1]->nb[0] != sizeof(int32_t) ||
                node->nb[0] != sizeof(float) || node->src[0]->ne[2] != node->src[1]->ne[1]) {
            return false;
        }
        rows += node->src[0]->ne[2];
    }
    if (size_t(rows * k) > pimpl->cross_window_routes ||
            size_t(std::max(x->ne[0], first->ne[0])) > pimpl->cross_window_width) { return false; }

    ggml_custom_op_params params;
    std::memcpy(&params, first->op_params, sizeof(params));
    const auto & projection = *static_cast<const impl::projection *>(params.userdata);
    auto pack = [](ggml_tensor copy, void * data, int64_t a, int64_t b, int64_t c) {
        copy.data = data;
        copy.buffer = nullptr;
        copy.ne[0] = a; copy.ne[1] = b; copy.ne[2] = c; copy.ne[3] = 1;
        copy.nb[1] = copy.nb[0] * a; copy.nb[2] = copy.nb[1] * b; copy.nb[3] = copy.nb[2] * c;
        return copy;
    };
    auto input = pack(*x, pimpl->cross_window_input.get(), x->ne[0], input_slots, rows);
    auto routed = pack(*ids, pimpl->cross_window_ids.get(), k, rows, 1);
    auto output = pack(*first, pimpl->cross_window_output.get(), first->ne[0], k, rows);
    output.src[0] = &input; output.src[1] = &routed;
    size_t offset = 0;
    for (size_t i = 0; i < count; ++i) {
        const auto * source = nodes[i]->src[0];
        const auto * routes = nodes[i]->src[1];
        for (int64_t row = 0; row < source->ne[2]; ++row, ++offset) {
            std::memcpy(pimpl->cross_window_ids.get() + offset * k,
                    static_cast<const char *>(routes->data) + row * routes->nb[1], k * sizeof(int32_t));
            for (int64_t slot = 0; slot < input_slots; ++slot) {
                std::memcpy(static_cast<char *>(input.data) + offset * input.nb[2] + slot * input.nb[1],
                        static_cast<const char *>(source->data) + row * source->nb[2] + slot * source->nb[1], input.nb[1]);
            }
        }
    }
    // Keep the same one-row arithmetic even after concatenation crosses the
    // ordinary matrix threshold. Each result returns to its original route.
    pimpl->compute(projection, &output, true);
    offset = 0;
    for (size_t i = 0; i < count; ++i) {
        auto * node = nodes[i];
        for (int64_t row = 0; row < node->ne[2]; ++row, ++offset) {
            for (int64_t slot = 0; slot < k; ++slot) {
                std::memcpy(static_cast<char *>(node->data) + row * node->nb[2] + slot * node->nb[1],
                        static_cast<const char *>(output.data) + offset * output.nb[2] + slot * output.nb[1], output.nb[1]);
            }
        }
    }
    return true;
}

void llama_expert_executor::use_cross_window_output(const ggml_tensor * node) {
    if (pimpl->fused_window_moe) { pimpl->moe->use_window_output(node); return; }
    if (pimpl->cross_window_ready) { throw std::logic_error("previous prefill projection was not consumed"); }
    pimpl->cross_window_ready = node;
}

void llama_expert_executor::enable_fused_window_moe() {
    pimpl->fused_window_moe = pimpl->layer_window && pimpl->window_addresses && pimpl->moe && pimpl->moe->address_batches();
}

void llama_expert_executor::compute_fused_window_moe(ggml_tensor * const * nodes, size_t count, const window_bindings * bindings) {
    if (!pimpl->fused_window_moe || count > cross_window_workers) { throw std::logic_error("invalid fused window state"); }
    pimpl->moe->compute_window(nodes, count, bindings ? bindings->data() : nullptr);
}

bool llama_expert_executor::prepare_matrix_window_moe() {
    return pimpl->fused_window_moe && pimpl->moe->prepare_matrix_window();
}

bool llama_expert_executor::compute_matrix_window_moe(ggml_tensor * const * nodes, size_t count, const window_bindings & bindings) {
    return pimpl->moe->compute_matrix_window(nodes, count, bindings.data());
}

size_t llama_expert_executor::window_expert_group_bytes(const std::array<const ggml_tensor *, 3> & weights) const {
    size_t bytes = 0;
    for (const auto * weight : weights) {
        const size_t slice = pimpl->store->compact_slice_bytes(pimpl->projections.at(weight).name);
        if (!slice || slice > SIZE_MAX - bytes) { return 0; }
        bytes += slice;
    }
    return bytes;
}

size_t llama_expert_executor::window_expert_capacity(size_t group_bytes, size_t queued_experts) const {
    const auto snapshot = pimpl->budget->snapshot(llama_memory_domain::host);
    const size_t fixed = snapshot.used() - pimpl->store->storage_bytes();
    // Keep one projection and its read workspace available to the original
    // executor. The full layer union must fit even when every slice is cold.
    size_t headroom = pimpl->max_expert_payload + pimpl->max_expert_allocation;
    if (group_bytes) {
        headroom += llama_expert_store::demand_queue_bytes(queued_experts);
    } else {
        group_bytes = 3 * pimpl->max_expert_allocation;
    }
    const size_t limit = std::min(snapshot.requested, snapshot.applied);
    return limit > fixed + headroom ?
        (limit - fixed - headroom) / group_bytes : 0;
}

void llama_expert_executor::pin_window_experts(const std::array<const ggml_tensor *, 3> & weights,
        const int64_t * ids, size_t count, window_leases & leases, bool adjacent_ranges, window_bindings * bindings) {
    if (!pimpl->layer_window || 3 * count > leases.capacity() - leases.size()) {
        throw std::logic_error("invalid prefill expert group");
    }
    // The store reserves these buffers itself, as in the CPU projection bridge.
    llama_allocation_scope explicit_admission;
    // File layout is three separate tensors in ordinary GGUF. The existing
    // store admits, validates and publishes each bounded set of exact slices.
    for (size_t projection = 0; projection < weights.size(); ++projection) {
        const auto & name = pimpl->projections.at(weights[projection]).name;
        for (size_t first = 0; first < count;) {
            const size_t size = std::min(size_t(16), count - first);
            const size_t output = leases.size();
            leases.resize(output + size);
            pimpl->store->acquire_many(name, ids + first, size, leases.data() + output, nullptr, adjacent_ranges);
            if (bindings) {
                const size_t binding = projection < 2 ? 1 - projection : projection;
                for (size_t i = 0; i < size; ++i) {
                    bindings->at(ids[first + i])[binding] = leases[output + i].tensor();
                }
            }
            first += size;
        }
    }
}

bool llama_expert_executor::can_pipeline_window() const {
    return pimpl->layer_window && pimpl->moe && pimpl->moe->window_readers();
}

llama_expert_executor::window_demand llama_expert_executor::enqueue_window_experts(
        const std::array<const ggml_tensor *, 3> & weights, const int64_t * ids, size_t count) {
    if (!can_pipeline_window()) { throw std::logic_error("window demand pool unavailable"); }
    llama_allocation_scope explicit_admission;
    // The store expects up/gate/down. It keeps every slice unpublished until
    // the entire triple validates. Decode prediction is idle inside the window.
    const std::array<const char *, 3> names{
        pimpl->projections.at(weights[1]).name.c_str(),
        pimpl->projections.at(weights[0]).name.c_str(),
        pimpl->projections.at(weights[2]).name.c_str(),
    };
    return pimpl->store->enqueue_demand(names, ids, count, nullptr, pimpl->moe->window_readers());
}

void llama_expert_executor::drain_window_experts(window_demand & demand, window_leases & leases, window_bindings * bindings) {
    std::array<int64_t, 16> ids;
    std::array<llama_expert_store::lease, 16> gate, up, down;
    while (demand->pending()) {
        const size_t count = demand->take(ids.data(), up.data(), gate.data(), down.data(), 1);
        if (3 * count > leases.capacity() - leases.size()) {
            throw std::logic_error("queued window leases exceed the layer union");
        }
        for (size_t i = 0; i < count; ++i) {
            if (bindings) { bindings->at(ids[i]) = {up[i].tensor(), gate[i].tensor(), down[i].tensor()}; }
            leases.push_back(std::move(up[i]));
            leases.push_back(std::move(gate[i]));
            leases.push_back(std::move(down[i]));
        }
    }
    demand.reset();
}

bool llama_expert_executor::set_layer_window(size_t budget, bool vector_batch) {
    if (budget) {
        if (!can_layer_window()) { return false; }
        // Window graphs use the existing per-projection executor. Keep the
        // automatic MoE workspace/descriptors alive for cached decode graphs,
        // but drain its predictions before the window can use the shared store.
        if (pimpl->moe) {
            pimpl->moe->end_graph();
            ggml_backend_synchronize(pimpl->backend.get());
        }
        select_decode_policy(0, 0);
        // A resize suppresses advice while the ledger is at a different cap.
        pimpl->window_advice_budget = budget;
    }
    pimpl->layer_window = budget != 0;
    pimpl->window_addresses = nullptr;
    if (!budget) {
        if (pimpl->fused_window_moe) {
            pimpl->moe->clear_window_output();
            pimpl->moe->release_matrix_window();
        }
        pimpl->fused_window_moe = false;
        pimpl->cross_window_ready = nullptr;
        pimpl->cross_window_ids.reset();
        pimpl->cross_window_output.reset();
        pimpl->cross_window_input.reset();
        pimpl->cross_window_charge.reset();
    }
    if (budget && vector_batch && can_batch_window()) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(pimpl->backend.get()));
        pimpl->window_addresses = reinterpret_cast<impl::window_addresses_t>(
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_metal_expert_addresses_v1"));
    }
    return true;
}

void llama_expert_executor::clear_window_decode_advice() { pimpl->window_advice_budget = 0; }

void llama_expert_executor::set_read_overlap(bool enabled) { pimpl->read_overlap = enabled; }
bool llama_expert_executor::set_single_expert_map(bool enabled) {
    if (enabled && (!pimpl->metal || !pimpl->prepare_map)) { return false; }
    pimpl->single_expert_map = enabled;
    return true;
}
bool llama_expert_executor::set_route_index(bool enabled) {
    if (error()) { return false; }
    if (!enabled) { pimpl->route_index.reset(); pimpl->route_charge.reset(); return true; }
    if (pimpl->route_index) { return true; }
    const auto memory = pimpl->budget->snapshot(llama_memory_domain::host);
    const auto floor = stats().minimum_bytes;
    if (memory.requested < floor || sizeof(llama_expert_route_index) > memory.requested - floor) { return false; }
    auto charge = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, sizeof(llama_expert_route_index));
    if (!charge) {
        pimpl->store->trim();
        charge = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, sizeof(llama_expert_route_index));
    }
    if (!charge) { return false; }
    std::unique_ptr<llama_expert_route_index> index;
    try { index = std::make_unique<llama_expert_route_index>(); }
    catch (const std::bad_alloc &) { return false; }
    charge.commit();
    pimpl->route_charge = std::move(charge);
    pimpl->route_index = std::move(index);
    return true;
}
bool llama_expert_executor::set_early_read(bool enabled) {
    if (error()) { return false; }
    if (!enabled) { pimpl->early.reset(); pimpl->early_charge.reset(); return true; }
    if (pimpl->early) { return true; }
    if (!pimpl->metal || pimpl->preview || pimpl->moe || pimpl->tracing || pimpl->class_policy || pimpl->projections.size() > 768) { return false; }
    const auto memory = pimpl->budget->snapshot(llama_memory_domain::host);
    const auto floor = stats().minimum_bytes;
    if (memory.requested < floor || sizeof(impl::early_state) > memory.requested - floor) { return false; }
    auto charge = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, sizeof(impl::early_state));
    if (!charge) {
        pimpl->store->trim();
        charge = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, sizeof(impl::early_state));
    }
    if (!charge) { return false; }
    std::unique_ptr<impl::early_state> state;
    try { state = std::make_unique<impl::early_state>(); }
    catch (const std::bad_alloc &) { return false; }
    charge.commit();
    pimpl->early_charge = std::move(charge);
    pimpl->early = std::move(state);
    return true;
}

void llama_expert_executor::bind_early_graph(ggml_cgraph * graph) {
    auto * state = pimpl->early.get();
    if (!state) { return; }
    llama_expert_timer timer(pimpl->profile ? &pimpl->profile->early_bind_ns : nullptr);
    state->count = 0;
    if (!graph) { return; }
    for (const auto & item : pimpl->projections) {
        auto & entry = state->entries[state->count++];
        entry = {};
        entry.source = &item.second;
    }
    auto identity = [&](const ggml_tensor * node) -> impl::early_state::occurrence * {
        if (node->op != GGML_OP_CUSTOM) { return nullptr; }
        ggml_custom_op_params params;
        std::memcpy(&params, node->op_params, sizeof(params));
        if (params.fun != impl::callback) { return nullptr; }
        for (size_t i = 0; i < state->count; ++i) {
            if (params.userdata == state->entries[i].source) { return &state->entries[i]; }
        }
        return nullptr;
    };
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        if (auto * entry = identity(ggml_graph_node(graph, i))) { entry->index = entry->index == -1 ? i : -2; }
    }
    for (size_t i = 0; i < state->count; ++i) {
        auto & entry = state->entries[i];
        if (entry.index < 0 || entry.index + 1 >= ggml_graph_n_nodes(graph)) { continue; }
        const auto * source = ggml_graph_node(graph, entry.index);
        const auto * target = ggml_graph_node(graph, entry.index + 1);
        const auto * next = identity(target);
        // Adjacency excludes intervening writers. Shared inputs and actual IDs
        // prove independence and exact demand, with no predictor or graph edit.
        // Never retain scheduler-owned tensor pointers across bindings.
        if (!next || next->index != entry.index + 1 || entry.source == next->source ||
            source->src[0] != target->src[0] || source->src[1] != target->src[1] ||
            source->ne[1] != target->ne[1] || source->ne[2] != target->ne[2] ||
            entry.source->experts != next->source->experts || entry.source->input != next->source->input) { continue; }
        entry.target = next->source;
        entry.rows = source->ne[2]; entry.slots = source->ne[1];
        if (pimpl->profile) { ++pimpl->profile->early_bound_pairs; }
    }
}

void llama_expert_executor::set_route_observer(llama_expert_route_callback callback, void * user) {
    if (callback && pimpl->moe) { throw std::invalid_argument("combined MoE does not support projection observers"); }
    pimpl->route_observer = callback; pimpl->route_user = user;
}
bool llama_expert_executor::get_profile(llama_expert_profile_data & result) const {
    if (!pimpl->profile) { return false; }
    result = pimpl->profile_data;
    return true;
}

void llama_expert_executor::bind_preview_graph(ggml_cgraph * graph) {
    auto * state = pimpl->preview.get();
    if (!state) { return; }
    llama_expert_timer timer(pimpl->profile ? &pimpl->profile->preview_bind_ns : nullptr);
    state->last = nullptr;
    state->first = nullptr;
    for (size_t l = 0; l < state->count; ++l) { state->layers[l].graph = {}; }
    if (!graph) { return; }
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto * node = ggml_graph_node(graph, i);
        if (node->op != GGML_OP_CUSTOM) { continue; }
        ggml_custom_op_params params;
        std::memcpy(&params, node->op_params, sizeof(params));
        if (params.fun != impl::callback) { continue; }
        // Compare opaque identities, never dereference a foreign executor's
        // userdata. Only scalar geometry/order is retained across scheduling.
        for (size_t l = 0; l < state->count; ++l) {
            auto & layer = state->layers[l];
            const std::array<const impl::projection *, 5> weights {
                layer.gate, layer.up, layer.target[0], layer.target[1], layer.target[2]};
            for (size_t w = 0; w < weights.size(); ++w) {
                if (params.userdata != weights[w]) { continue; }
                auto & occurrence = layer.graph[w];
                occurrence.index = occurrence.index == -1 ? i : -2;
                occurrence.rows = node->ne[2];
            }
        }
    }
}

bool llama_expert_executor::set_preview(int mode, const std::vector<llama_expert_preview_layer> & layers) {
    if (mode < 0 || mode > 3 || error()) { return false; }
    if (mode == 0) {
        pimpl->preview.reset(); pimpl->preview_charge.reset(); return true;
    }
    if (pimpl->early || pimpl->moe || pimpl->tracing || pimpl->class_policy || layers.empty() || layers.size() > 256) { return false; }
    // Configure before evaluation, or after an explicit disable. A failed
    // configuration never changes the active policy or its memory floor.
    if (pimpl->preview) { return false; }
    using host_base_t = void * (*)(ggml_backend_buffer_t);
    const auto device = ggml_backend_get_device(pimpl->backend.get());
    const auto host_base = reinterpret_cast<host_base_t>(ggml_backend_reg_get_proc_address(
        ggml_backend_dev_backend_reg(device), "ggml_backend_metal_buffer_get_host_base"));
    for (const auto & l : layers) {
        const auto * r = l.next_router;
        if (!r || r->type != GGML_TYPE_F32 || !ggml_is_contiguous(r) || r->ne[0] < 1 || r->ne[0] > 16384 ||
            r->ne[1] < 4 || r->ne[1] > 1024 || r->ne[2] != 1 || r->ne[3] != 1 || !r->buffer || !r->data) { return false; }
        void * base = ggml_backend_buffer_get_base(r->buffer);
        if (!ggml_backend_buffer_is_host(r->buffer) && (!host_base || host_base(r->buffer) != base)) { return false; }
        const auto at = reinterpret_cast<uintptr_t>(r->data), start = reinterpret_cast<uintptr_t>(base);
        const size_t size = ggml_backend_buffer_get_size(r->buffer);
        if (at < start || at - start > size || ggml_nbytes(r) > size - (at - start)) { return false; }
        for (const auto * w : {l.gate, l.up}) {
            const auto found = pimpl->projections.find(w);
            if (found == pimpl->projections.end() || found->second.input != r->ne[0] || found->second.experts > 1024) { return false; }
        }
        for (const auto * w : l.next_weights) {
            const auto found = pimpl->projections.find(w);
            if (found == pimpl->projections.end() || found->second.experts != r->ne[1]) { return false; }
        }
    }
    const auto memory = pimpl->budget->snapshot(llama_memory_domain::host);
    if (memory.requested < stats().minimum_bytes || sizeof(impl::preview_state) > memory.requested - stats().minimum_bytes) { return false; }
    // Cache is reclaimable; predictor payload has its own explicit owner.
    auto charge = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, sizeof(impl::preview_state));
    if (!charge) {
        pimpl->store->trim();
        charge = pimpl->budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, sizeof(impl::preview_state));
    }
    if (!charge) { return false; }
    auto state = std::make_unique<impl::preview_state>();
    charge.commit();
    state->mode = mode;
    for (const auto & l : layers) {
        auto & dst = state->layers[state->count++];
        dst.gate = &pimpl->projections.at(l.gate); dst.up = &pimpl->projections.at(l.up);
        dst.router = static_cast<const float *>(l.next_router->data);
        dst.width = size_t(l.next_router->ne[0]); dst.experts = size_t(l.next_router->ne[1]);
        for (size_t i = 0; i < 3; ++i) { dst.target[i] = &pimpl->projections.at(l.next_weights[i]); }
    }
    pimpl->preview_charge = std::move(charge);
    pimpl->preview = std::move(state);
    return true;
}
