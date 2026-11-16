#include "llama-expert-store.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__APPLE__)
#include <mach/mach.h>
#endif

static uint64_t process_footprint() {
#if defined(__APPLE__)
    task_vm_info_data_t info {};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        return info.phys_footprint;
    }
#endif
    return 0; // unavailable, not a measured zero
}

static bool routed_weight(const std::string & name) {
    for (const char * component : {".ffn_gate_exps.weight", ".ffn_up_exps.weight",
                                   ".ffn_down_exps.weight", ".ffn_gate_up_exps.weight"}) {
        const std::string suffix(component);
        if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0) {
            return true;
        }
    }
    return false;
}

int main(int argc, char ** argv) {
    try {
        bool sample = false;
        size_t store_bytes = 0;
        std::string backend_name = "CPU";
        std::vector<std::string> paths;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--sample") {
                sample = true;
            } else if (arg == "--store-mib") {
                if (++i == argc) { throw std::invalid_argument("--store-mib requires a positive integer"); }
                const std::string value(argv[i]);
                size_t mib = 0;
                for (char c : value) {
                    if (c < '0' || c > '9' || mib > (std::numeric_limits<size_t>::max() - size_t(c - '0')) / 10) {
                        throw std::invalid_argument("invalid --store-mib value");
                    }
                    mib = mib * 10 + size_t(c - '0');
                }
                if (mib == 0 || mib > std::numeric_limits<size_t>::max() / (1024 * 1024)) {
                    throw std::invalid_argument("--store-mib is out of range");
                }
                store_bytes = mib * 1024 * 1024;
                sample = true;
            } else if (arg == "--backend") {
                if (++i == argc) { throw std::invalid_argument("--backend requires a device name"); }
                backend_name = argv[i];
            } else if (arg == "--help" || arg == "-h") {
                std::printf("usage: %s [--sample | --store-mib N] [--backend CPU|MTL0] SHARD.gguf [SHARD.gguf ...]\n"
                            "Index explicit shards in order without loading weights.\n"
                            "--sample reads the first and last expert of each routed weight tensor.\n"
                            "--store-mib samples through an admitted store, then shrinks it to zero.\n"
                            "The store limit covers buffers, tensor arenas, staging and sample scratch.\n"
                            "This checks storage only, not streaming inference or a total memory cap.\n", argv[0]);
                return 0;
            } else {
                paths.push_back(arg);
            }
        }
        if (backend_name != "CPU" && store_bytes == 0) {
            throw std::invalid_argument("--backend requires --store-mib");
        }
        auto source = std::make_shared<llama_expert_source>(paths);
        auto budget = std::make_shared<llama_memory_budget>(store_bytes, 0, true);
        ggml_backend_ptr backend;
        std::unique_ptr<llama_expert_store> store;
        if (store_bytes != 0) {
            if (backend_name != "CPU" && backend_name != "MTL0") {
                throw std::invalid_argument("the store probe currently supports CPU or MTL0");
            }
            ggml_backend_load_all();
            backend.reset(ggml_backend_init_by_name(backend_name.c_str(), nullptr));
            if (!backend) { throw std::runtime_error("requested backend is unavailable"); }
            store.reset(new llama_expert_store(source, budget, backend.get(), llama_memory_domain::device, 65536));
        }
        std::printf("architecture=%s shards=%zu tensors=%zu io=buffered\n",
                    source->architecture().c_str(), paths.size(), source->tensors().size());
        size_t projections = 0;
        size_t unsupported = 0;
        size_t max_bytes = 0;
        for (const auto & t : source->tensors()) {
            if (!routed_weight(t.name)) {
                continue;
            }
            ++projections;
            if (!t.can_read_experts()) {
                ++unsupported;
                std::printf("tensor=%s storage=unsupported\n", t.name.c_str());
                continue;
            }
            max_bytes = std::max(max_bytes, t.expert_bytes());
            std::printf("tensor=%s shard=%zu type=%s shape=%lld,%lld,%lld offset=%zu expert_bytes=%zu\n",
                        t.name.c_str(), t.shard, ggml_type_name(t.type),
                        (long long) t.ne[0], (long long) t.ne[1], (long long) t.ne[2], t.offset, t.expert_bytes());
        }
        if (projections == 0 || unsupported != 0) {
            throw std::runtime_error("no supported routed weights, or unsupported expert layouts found");
        }
        uint64_t read_bytes = 0;
        uint64_t digest = 14695981039346656037ULL;
        if (sample) {
            // A diagnostic bound, not a model capability or an engine memory cap.
            constexpr size_t sample_limit = 64 * 1024 * 1024;
            if (max_bytes > sample_limit) {
                throw std::runtime_error("sample buffer would exceed the 64 MiB diagnostic limit");
            }
            llama_memory_budget::ticket sample_charge;
            if (store) {
                sample_charge = budget->reserve(llama_memory_domain::host, llama_memory_category::scratch, max_bytes);
                if (!sample_charge) { throw std::runtime_error("sample scratch does not fit store budget"); }
            }
            std::unique_ptr<uint8_t[]> buffer(new uint8_t[max_bytes]);
            if (sample_charge) { sample_charge.commit(); }
            for (const auto & t : source->tensors()) {
                if (!routed_weight(t.name)) {
                    continue;
                }
                for (int64_t expert : {int64_t(0), t.ne[2] - 1}) {
                    if (store) {
                        auto lease = store->acquire(t.name, expert);
                        ggml_backend_tensor_get(lease.tensor(), buffer.get(), 0, t.expert_bytes());
                    } else {
                        source->read_expert(t.name, expert, buffer.get(), t.expert_bytes());
                    }
                    for (size_t j = 0; j < t.expert_bytes(); ++j) {
                        digest = (digest ^ buffer[j]) * 1099511628211ULL;
                    }
                    read_bytes += t.expert_bytes();
                }
            }
        }
        if (store) {
            const auto stats = store->stats();
            const auto accounting_before = budget->snapshot(llama_memory_domain::host);
            const auto footprint_before = process_footprint();
            size_t free_before = 0, total_before = 0, free_after = 0, total_after = 0;
            if (backend_name == "MTL0") {
                ggml_backend_dev_memory(ggml_backend_get_device(backend.get()), &free_before, &total_before);
            }
            const auto released = store->request_budget(llama_memory_domain::host, 0);
            const auto footprint_after = process_footprint();
            if (backend_name == "MTL0") {
                ggml_backend_dev_memory(ggml_backend_get_device(backend.get()), &free_after, &total_after);
                // This Metal backend exposes recommendedMaxWorkingSetSize minus
                // currentAllocatedSize as free memory. Saturated values are unusable.
                if (free_before > 0 && free_after > 0) {
                    std::printf("metal_allocated_before_shrink=%zu metal_allocated_after_shrink=%zu\n",
                                total_before - free_before, total_after - free_after);
                } else {
                    std::printf("metal_allocation_measurement=unavailable\n");
                }
            }
            std::printf("process_footprint_available=%s before_shrink=%llu after_shrink=%llu\n",
                        footprint_before && footprint_after ? "true" : "false",
                        (unsigned long long) footprint_before, (unsigned long long) footprint_after);
            if (released.pending() || released.used() != 0) {
                throw std::runtime_error("store failed to release all admitted allocations");
            }
            std::printf("store_backend=%s managed_cap=%zu high_water=%zu misses=%llu evictions=%llu "
                        "shrink_released_bytes=%zu lifetime_released_bytes=%zu applied_after_shrink=%zu remaining=%zu\n",
                        backend_name.c_str(), store_bytes, released.high_water,
                        (unsigned long long) stats.misses, (unsigned long long) stats.evictions,
                        released.released - accounting_before.released, released.released, released.applied, released.used());
        }
        std::printf("routed_tensors=%zu max_expert_bytes=%zu sample_logical_bytes=%llu sample_fnv1a64=%016llx\n",
                    projections, max_bytes, (unsigned long long) read_bytes, (unsigned long long) digest);
        std::printf("streaming_inference=unverified total_memory_cap=unimplemented live_resize=unimplemented\n");
        std::fflush(stdout);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "expert source: %s\n", e.what());
        return 1;
    }
}
