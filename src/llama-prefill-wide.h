#pragma once

#include "llama-expert-source.h"
#include "llama-memory-budget.h"
#include "ggml-cpp.h"

#include <algorithm>
#include <array>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

// The graph binds each layer to a bank before scheduling. The coordinator
// publishes weights only after reading the whole layer and waits for Metal
// at the exact layer output before a bank can be reused. When the second bank
// does not fit, a bounded host buffer admits a prefix of the next layer.
class llama_prefill_wide_weights {
public:
    using layer_weights = std::array<ggml_tensor *, 3>;

    struct retained_bank {
        llama_memory_budget::ticket metadata;
        std::array<ggml_backend_buffer_ptr, 3> buffers;
        ggml_context_ptr context;
        layer_weights weights{};
        size_t buffer_bytes = 0;
    };

    struct reading {
        uint64_t bytes = 0;
        int64_t elapsed_us = 0;
        uint64_t prefetched_bytes = 0;
        uint64_t tail_bytes = 0;
    };

    explicit llama_prefill_wide_weights(llama_expert_source & source) : source(source) {
#if defined(__APPLE__)
        if (source.architecture() == "qwen4exp") {
            const char * option = std::getenv("LLAMA_FLASH_WIDE_BULK_READ");
            bulk_read = option && std::strcmp(option, "1") == 0;
        }
#endif
    }

    ~llama_prefill_wide_weights() { stop_reader(); }

    bool prepare(ggml_backend_t backend, ggml_backend_buffer_type_t buffer_type, int layers,
            const std::array<const ggml_tensor *, 3> * weights, const layer_weights * resident = nullptr) {
        buft = buffer_type;
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
        shared_base = reinterpret_cast<shared_base_fn>(ggml_backend_reg_get_proc_address(reg,
                "ggml_backend_metal_buffer_get_host_base"));
        if (!shared_base) { return false; }

        sources.resize(layers);
        for (int layer = 0; layer < layers; ++layer) {
            if (resident && resident[layer][0]) { continue; }
            if (std::none_of(weights[layer].begin(), weights[layer].end(), [](const auto * tensor) { return tensor; })) {
                continue; // leading dense layer
            }
            for (size_t part = 0; part < bank_sizes.size(); ++part) {
                const auto * original = weights[layer][part];
                if (!original) { return false; }
                const auto & info = source.tensor(original->name);
                if (!info.can_read_experts() || info.type != original->type ||
                        !std::equal(info.ne.begin(), info.ne.end(), original->ne) ||
                        info.bytes != ggml_nbytes(original)) { return false; }
                sources[layer][part] = &info;
                bank_sizes[part] = std::max(bank_sizes[part], info.bytes);
            }
        }
        if (!allocate_bank(banks[0])) { return false; }
        bindings = banks[0].aliases;
        if (resident) {
            for (int layer = 0; layer < layers; ++layer) {
                if (resident[layer][0]) { bindings[layer] = resident[layer]; }
            }
        }
        return true;
    }

    // Called after the serial graph reservation, before applying prompt state.
    // Backend allocations are charged by the enclosing decode allocation scope.
    bool enable_overlap(llama_memory_budget & ledger, bool allow_bank_tails, bool require_full_bank = false) {
        // Allow for the reader's default Darwin stack and pthread metadata.
        auto charge = ledger.reserve(llama_memory_domain::host, llama_memory_category::scratch, 1024 * 1024 + 64 * 1024);
        if (!charge) { return false; }
        bank second;
        if (!allocate_bank(second)) {
            if (require_full_bank) { return false; }
            second = {};
            for (size_t bytes : {256U << 20, 128U << 20, 64U << 20}) {
                staging.reset(ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), bytes));
                if (staging) {
                    staging_size = bytes;
                    break;
                }
            }
            use_bank_tails = allow_bank_tails;
            if (!staging && !use_bank_tails) { return false; }
        }
        try {
            reader = std::thread(&llama_prefill_wide_weights::read_loop, this);
        } catch (const std::system_error &) {
            staging.reset();
            staging_size = 0;
            use_bank_tails = false;
            return false;
        }
        if (!staging && !use_bank_tails) {
            banks[1] = std::move(second);
            for (size_t layer = 1; layer < bindings.size(); layer += 2) {
                if (sources[layer][0]) { bindings[layer] = banks[1].aliases[layer]; }
            }
        }
        charge.commit();
        reader_charge = std::move(charge);
        return true;
    }

    const layer_weights * graph_bindings() const { return bindings.data(); }
    size_t staging_capacity() const { return staging_size; }

    // Only after the final layer flush and destruction of the window graphs.
    // Each complete Metal bank still contains the last layer that used it.
    std::array<retained_bank, 2> take_tail_banks(llama_memory_budget & ledger) {
        stop_reader();
        const size_t last = sources.size() - 1;
        const bool double_buffered = bool(banks[1].context);
        std::array<retained_bank, 2> result;
        for (size_t i = double_buffered ? 0 : 1; i < result.size(); ++i) {
            const size_t layer = last - 1 + i;
            if (!sources[layer][0]) { continue; }
            auto & source_bank = banks[double_buffered ? layer % 2 : 0];
            auto charge = ledger.reserve(llama_memory_domain::host, llama_memory_category::experts,
                    ggml_get_mem_size(source_bank.context.get()));
            if (!charge) { continue; }
            charge.commit();
            auto & retained = result[i];
            retained.metadata = std::move(charge);
            retained.weights = source_bank.aliases[layer];
            retained.context = std::move(source_bank.context);
            retained.buffers = std::move(source_bank.buffers);
            for (const auto & buffer : retained.buffers) {
                retained.buffer_bytes += ggml_backend_buffer_get_size(buffer.get());
            }
        }
        return result;
    }

    void queue(int layer) {
        if (!reader.joinable() || ((staging || use_bank_tails) && layer == 0)) { return; }
        {
            std::lock_guard<std::mutex> lock(mutex);
            GGML_ASSERT(requested_layer == -1 && !ready);
            requested_layer = layer;
        }
        cv.notify_one();
    }

    reading wait(int layer) {
        if (!reader.joinable() || ((staging || use_bank_tails) && layer == 0)) { return read_layer(layer, false); }
        reading result;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [&] { return ready; });
            GGML_ASSERT(completed_layer == layer);
            ready = false;
            if (failure) { std::rethrow_exception(failure); }
            result = completed;
        }
        result.prefetched_bytes = result.bytes;
        if (staging || use_bank_tails) {
            // The other bank does not fit. Publish the staged prefix only after
            // Metal has completed the previous layer's use of the sole bank.
            const auto remaining = read_layer(layer, false);
            result.bytes += remaining.bytes;
            result.elapsed_us += remaining.elapsed_us;
        }
        return result;
    }

private:
    using shared_base_fn = void * (*)(ggml_backend_buffer_t);

    void stop_reader() {
        if (!reader.joinable()) { return; }
        {
            std::lock_guard<std::mutex> lock(mutex);
            stopping = true;
        }
        cv.notify_one();
        reader.join();
        reader_charge.reset();
    }

    struct bank {
        std::array<ggml_backend_buffer_ptr, 3> buffers;
        ggml_context_ptr context;
        std::vector<layer_weights> aliases;
    };

    bool allocate_bank(bank & out) {
        std::array<void *, 3> addresses;
        for (size_t part = 0; part < bank_sizes.size(); ++part) {
            out.buffers[part].reset(ggml_backend_buft_alloc_buffer(buft, bank_sizes[part]));
            if (!out.buffers[part]) { return false; }
            addresses[part] = shared_base(out.buffers[part].get());
            if (!addresses[part]) { return false; }
            ggml_backend_buffer_set_usage(out.buffers[part].get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        out.context.reset(ggml_init({ggml_tensor_overhead() * 3 * sources.size(), nullptr, true}));
        if (!out.context) { return false; }
        out.aliases.resize(sources.size());
        for (size_t layer = 0; layer < sources.size(); ++layer) {
            for (size_t part = 0; part < bank_sizes.size(); ++part) {
                if (!sources[layer][part]) { continue; }
                const auto & info = *sources[layer][part];
                auto * alias = ggml_new_tensor(out.context.get(), info.type, GGML_MAX_DIMS, info.ne.data());
                ggml_set_name(alias, info.name.c_str());
                if (ggml_backend_tensor_alloc(out.buffers[part].get(), alias, addresses[part]) != GGML_STATUS_SUCCESS) {
                    return false;
                }
                out.aliases[layer][part] = alias;
            }
        }
        return true;
    }

    void read_range(const llama_expert_tensor_info & info, int64_t first, int64_t end, char * destination) {
        const size_t stride = info.expert_bytes();
#if defined(__APPLE__)
        if (bulk_read && first < end) {
            source.read_expert_range(info.name, first, end, destination, size_t(end - first) * stride);
            return;
        }
#endif
        std::array<llama_expert_source::read_request, 16> reads;
        const int64_t origin = first;
        for (; first < end; first += reads.size()) {
            const size_t count = std::min<int64_t>(reads.size(), end - first);
            for (size_t i = 0; i < count; ++i) {
                const int64_t expert = first + i;
                reads[i] = {&info, expert, destination + (expert - origin) * stride};
            }
            source.read_experts(reads.data(), count, false, false, true);
        }
    }

    reading read_layer(int layer, bool prefetch) {
        const int64_t start = ggml_time_us();
        reading result;
        auto * prefix = staging ? static_cast<char *>(ggml_backend_buffer_get_base(staging.get())) : nullptr;
        const bool partial = (prefix || use_bank_tails) && layer > 0;
        size_t offset = 0;
        for (size_t part = 0; part < bank_sizes.size(); ++part) {
            if (!sources[layer][part]) { continue; }
            const auto & info = *sources[layer][part];
            auto * destination = static_cast<char *>(bindings[layer][part]->data);
            const size_t stride = info.expert_bytes();
            const int64_t experts = info.ne[2];
            int64_t staged = 0;
            if (partial) {
                size_t tail_offset = bank_sizes[part];
                if (use_bank_tails) {
                    // Stay beyond the live tensor, with a page of separation.
                    // Each binding refers to the same bank in partial mode.
                    constexpr size_t page = 16U << 10;
                    const auto * previous = sources[layer - 1][part];
                    if (previous) {
                        tail_offset = std::min(bank_sizes[part], (previous->bytes + 2*page - 1) / page * page);
                    } else {
                        tail_offset = 0;
                    }
                }
                const int64_t in_tail = std::min<int64_t>(experts, (bank_sizes[part] - tail_offset) / stride);
                const int64_t in_prefix = std::min<int64_t>(experts - in_tail, (staging_size - offset) / stride);
                staged = in_tail + in_prefix;
                if (prefetch) {
                    if (in_tail) { read_range(info, 0, in_tail, destination + tail_offset); }
                    if (in_prefix) { read_range(info, in_tail, staged, prefix + offset); }
                    offset += in_prefix * stride;
                    result.bytes += staged * stride;
                    result.tail_bytes += in_tail * stride;
                    continue;
                }
                if (in_tail) { std::memmove(destination, destination + tail_offset, in_tail * stride); }
                if (in_prefix) { std::memcpy(destination + in_tail * stride, prefix + offset, in_prefix * stride); }
                offset += in_prefix * stride;
            }
            read_range(info, staged, experts, destination + staged * stride);
            result.bytes += (experts - staged) * stride;
        }
        result.elapsed_us = ggml_time_us() - start;
        return result;
    }

    void read_loop() {
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            cv.wait(lock, [&] { return stopping || requested_layer >= 0; });
            if (stopping) { return; }
            const int layer = requested_layer;
            requested_layer = -1;
            lock.unlock();
            reading result;
            std::exception_ptr error;
            try {
                result = read_layer(layer, true);
            } catch (...) {
                error = std::current_exception();
            }
            lock.lock();
            completed = result;
            completed_layer = layer;
            failure = error;
            ready = true;
            cv.notify_one();
        }
    }

    llama_expert_source & source;
#if defined(__APPLE__)
    bool bulk_read = false;
#endif
    ggml_backend_buffer_type_t buft = nullptr;
    shared_base_fn shared_base = nullptr;
    std::array<size_t, 3> bank_sizes {};
    std::vector<std::array<const llama_expert_tensor_info *, 3>> sources;
    std::array<bank, 2> banks;
    std::vector<layer_weights> bindings;
    ggml_backend_buffer_ptr staging;
    size_t staging_size = 0;
    bool use_bank_tails = false;

    llama_memory_budget::ticket reader_charge;
    std::mutex mutex;
    std::condition_variable cv;
    std::thread reader;
    int requested_layer = -1;
    int completed_layer = -1;
    bool ready = false;
    bool stopping = false;
    reading completed;
    std::exception_ptr failure;
};
