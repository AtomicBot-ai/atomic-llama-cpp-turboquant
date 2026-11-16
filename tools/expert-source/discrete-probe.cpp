#include "ggml-cpp.h"
#include "llama-memory-budget.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

using domain = llama_memory_domain;
using category = llama_memory_category;

static void require(bool condition, const char * message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

static int probe(const char * name, bool cpu_control) {
    ggml_backend_load_all();
    const auto dev = ggml_backend_dev_by_name(name);
    if (!dev) {
        std::fprintf(stderr, "device unavailable: %s\n", name);
        return 2;
    }

    const auto registry = ggml_backend_dev_backend_reg(dev);
    ggml_backend_dev_props props {};
    ggml_backend_dev_get_props(dev, &props);
    std::printf("device=%s backend=%s async=%d events=%d pinned_host_capability=%d\n",
            props.name, ggml_backend_reg_name(registry), props.caps.async, props.caps.events, props.caps.host_buffer);
    if (!cpu_control) {
        require(std::strcmp(ggml_backend_reg_name(registry), "CUDA") == 0 &&
                props.type == GGML_BACKEND_DEVICE_TYPE_GPU, "probe requires a discrete CUDA device");
        require(props.caps.async && props.caps.events && props.caps.host_buffer,
                "CUDA async copies, events and pinned host buffers are required");
    }

    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    require(bool(backend), "backend initialization failed");
    const auto device_type = ggml_backend_dev_buffer_type(dev);
    const auto host_type = cpu_control ? device_type : ggml_backend_dev_host_buffer_type(dev);
    require(device_type && host_type, "backend buffer types unavailable");

    constexpr size_t payload = 1024 * 1024;
    constexpr size_t host_bytes = 2 * payload;
    ggml_context_ptr ctx(ggml_init({ggml_tensor_overhead(), nullptr, true}));
    require(bool(ctx), "tensor metadata allocation failed");
    auto * tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, payload);
    const size_t device_bytes = ggml_backend_buft_get_alloc_size(device_type, tensor);
    size_t free_before = 0, free_allocated = 0, free_released = 0, total = 0;
    ggml_backend_dev_memory(dev, &free_before, &total);

    llama_memory_budget budget(host_bytes, device_bytes, false);
    auto host_charge = budget.reserve(domain::host, category::io, host_bytes);
    auto device_charge = budget.reserve(domain::device, category::experts, device_bytes);
    require(host_charge && device_charge, "initial budget reservation failed");
    ggml_backend_buffer_ptr host(ggml_backend_buft_alloc_buffer(host_type, host_bytes));
    ggml_backend_buffer_ptr device(ggml_backend_buft_alloc_buffer(device_type, device_bytes));
    require(host && device, "copy buffer allocation failed");
    require(ggml_backend_buffer_get_type(host.get()) == host_type && ggml_backend_buffer_is_host(host.get()),
            "pinned host allocation fell back to another buffer type");
    require(ggml_backend_buffer_get_type(device.get()) == device_type &&
            (cpu_control || !ggml_backend_buffer_is_host(device.get())), "device allocation used an unexpected buffer type");
    require(ggml_backend_buffer_get_size(host.get()) == host_bytes &&
            ggml_backend_buffer_get_size(device.get()) == device_bytes, "buffer size differed from reserved bytes");
    host_charge.commit();
    device_charge.commit();
    require(ggml_backend_tensor_alloc(device.get(), tensor, ggml_backend_buffer_get_base(device.get())) ==
            GGML_STATUS_SUCCESS, "device tensor binding failed");
    ggml_backend_dev_memory(dev, &free_allocated, &total);

    auto * source = static_cast<unsigned char *>(ggml_backend_buffer_get_base(host.get()));
    require(source != nullptr, "host buffer has no address");
    auto * destination = source + payload;
    for (size_t i = 0; i < payload; ++i) {
        source[i] = static_cast<unsigned char>(i ^ (i >> 8) ^ (i >> 16));
        destination[i] = source[i] ^ 0xff;
    }

    ggml_backend_event_ptr completion(cpu_control ? nullptr : ggml_backend_event_new(dev));
    require(cpu_control || bool(completion), "device event creation failed");
    const auto start = std::chrono::steady_clock::now();
    ggml_backend_tensor_set_async(backend.get(), tensor, source, 0, payload);
    ggml_backend_tensor_get_async(backend.get(), tensor, destination, 0, payload);
    if (completion) {
        ggml_backend_event_record(completion.get(), backend.get());
    }
    // A pending shrink must not release either buffer while the transfer owns it.
    budget.request(domain::host, 0);
    budget.request(domain::device, 0);
    const auto host_pending = budget.snapshot(domain::host);
    const auto device_pending = budget.snapshot(domain::device);
    if (completion) {
        ggml_backend_event_synchronize(completion.get());
    } else {
        ggml_backend_synchronize(backend.get());
    }
    const double milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    require(host_pending.pending() && host_pending.used() == host_bytes &&
            device_pending.pending() && device_pending.used() == device_bytes, "shrink applied before buffer release");
    require(std::memcmp(source, destination, payload) == 0, "copy round trip changed payload");

    completion.reset();
    device.reset();
    device_charge.reset();
    host.reset();
    host_charge.reset();
    const auto host_done = budget.snapshot(domain::host);
    const auto device_done = budget.snapshot(domain::device);
    require(!host_done.pending() && host_done.applied == 0 && host_done.used() == 0 && host_done.released == host_bytes &&
            !device_done.pending() && device_done.applied == 0 && device_done.used() == 0 &&
            device_done.released == device_bytes, "charges survived buffer release");
    ggml_backend_dev_memory(dev, &free_released, &total);
    std::printf("mode=%s payload_bytes=%zu round_trip_ms=%.3f host_peak=%zu device_peak=%zu\n",
            cpu_control ? "cpu_control" : "cuda_event", payload, milliseconds, host_done.high_water, device_done.high_water);
    std::printf("free_bytes_before=%zu free_bytes_allocated=%zu free_bytes_released=%zu total_bytes=%zu\n",
            free_before, free_allocated, free_released, total);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc > 2 || (argc == 2 && argv[1][0] == '-' && std::strcmp(argv[1], "--cpu-control") != 0)) {
        std::fprintf(stderr, "usage: llama-discrete-probe [CUDA_DEVICE | --cpu-control]\n");
        return EXIT_FAILURE;
    }
    const bool cpu_control = argc == 2 && std::strcmp(argv[1], "--cpu-control") == 0;
    // CUDA managed allocations cannot validate the device-only ownership split.
    if (!cpu_control && std::getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY")) {
        std::fprintf(stderr, "unset GGML_CUDA_ENABLE_UNIFIED_MEMORY for the discrete copy probe\n");
        return EXIT_FAILURE;
    }
    try {
        return probe(cpu_control ? "CPU" : argc == 2 ? argv[1] : "CUDA0", cpu_control);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "discrete memory probe: %s\n", e.what());
        return EXIT_FAILURE;
    }
}
