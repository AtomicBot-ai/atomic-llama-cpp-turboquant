#include "llama-memory-budget.h"

#include <cuda.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using domain = llama_memory_domain;
using category = llama_memory_category;

constexpr size_t readback_bytes = 1024 * 1024;

static void check(CUresult result, const char * operation) {
    if (result != CUDA_SUCCESS) {
        const char * name = nullptr;
        cuGetErrorName(result, &name);
        throw std::runtime_error(std::string(operation) + ": " + (name ? name : "unknown CUDA error"));
    }
}

struct primary_context {
    CUdevice device;
    CUcontext context = nullptr;

    ~primary_context() {
        if (context) {
            cuCtxSetCurrent(nullptr);
            cuDevicePrimaryCtxRelease(device);
        }
    }
};

struct address_reservation {
    CUdeviceptr address = 0;
    size_t bytes;

    ~address_reservation() {
        if (address) {
            cuMemAddressFree(address, bytes);
        }
    }
};

struct device_allocation {
    CUdeviceptr address = 0;
    size_t mapped_bytes = 0;
    CUmemGenericAllocationHandle handle {};
    bool owns_handle = false;
    bool ordinary = false;

    ~device_allocation() {
        if (mapped_bytes) {
            cuMemUnmap(address, mapped_bytes);
        }
        if (owns_handle) {
            cuMemRelease(handle);
        }
        if (ordinary && address) {
            cuMemFree(address);
        }
    }
};

static void run_cycle(llama_memory_budget & budget, const address_reservation & range,
        const CUmemAllocationProp & props, size_t bytes, unsigned char pattern, bool use_vmm,
        std::vector<unsigned char> & readback) {
    budget.request(domain::device, bytes);
    auto charge = budget.reserve(domain::device, category::experts, bytes);
    if (!charge || budget.reserve(domain::device, category::experts, 1)) {
        throw std::runtime_error("device admission did not enforce the cap");
    }

    device_allocation allocation;
    if (use_vmm) {
        check(cuMemCreate(&allocation.handle, bytes, &props, 0), "cuMemCreate");
        allocation.owns_handle = true;
        check(cuMemMap(range.address, bytes, 0, allocation.handle, 0), "cuMemMap");
        allocation.address = range.address;
        allocation.mapped_bytes = bytes;
        check(cuMemRelease(allocation.handle), "cuMemRelease");
        allocation.owns_handle = false;
        CUmemAccessDesc access {};
        access.location = props.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        check(cuMemSetAccess(allocation.address, bytes, &access, 1), "cuMemSetAccess");
    } else {
        check(cuMemAlloc(&allocation.address, bytes), "cuMemAlloc");
        allocation.ordinary = true;
    }
    charge.commit();

    check(cuMemsetD8Async(allocation.address, pattern, bytes, nullptr), "cuMemsetD8Async");
    budget.request(domain::device, 0);
    charge.retire();
    const auto pending = budget.snapshot(domain::device);
    check(cuCtxSynchronize(), "cuCtxSynchronize");
    if (!pending.pending() || pending.applied != bytes || pending.used() != bytes ||
            budget.snapshot(domain::device).used() != bytes) {
        throw std::runtime_error("device shrink lost its live physical charge");
    }
    for (size_t offset = 0; offset < bytes; offset += readback.size()) {
        const size_t chunk = std::min(bytes - offset, readback.size());
        check(cuMemcpyDtoH(readback.data(), allocation.address + offset, chunk), "cuMemcpyDtoH");
        if (!std::all_of(readback.data(), readback.data() + chunk, [pattern](unsigned char byte) { return byte == pattern; })) {
            throw std::runtime_error("device payload verification failed");
        }
    }

    if (use_vmm) {
        check(cuMemUnmap(allocation.address, allocation.mapped_bytes), "cuMemUnmap");
        allocation.mapped_bytes = 0;
    } else {
        check(cuMemFree(allocation.address), "cuMemFree");
        allocation.address = 0;
    }
    charge.reset();
    const auto released = budget.snapshot(domain::device);
    if (released.pending() || released.applied || released.used()) {
        throw std::runtime_error("device shrink failed to apply after release");
    }
}

static int probe(int ordinal, bool force_ordinary) {
    const CUresult init = cuInit(0);
    if (init == CUDA_ERROR_NO_DEVICE) {
        std::fprintf(stderr, "CUDA device unavailable\n");
        return 2;
    }
    check(init, "cuInit");
    int count = 0;
    check(cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (ordinal >= count) {
        std::fprintf(stderr, "CUDA device ordinal unavailable: %d\n", ordinal);
        return 2;
    }
    CUdevice device;
    check(cuDeviceGet(&device, ordinal), "cuDeviceGet");
    int integrated = 0;
    check(cuDeviceGetAttribute(&integrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, device), "integrated device query");
    if (integrated) {
        std::fprintf(stderr, "CUDA device is integrated: %d\n", ordinal);
        return 2;
    }
    int supports_vmm = 0;
    check(cuDeviceGetAttribute(&supports_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device),
            "VMM capability query");
    primary_context context {device};
    check(cuDevicePrimaryCtxRetain(&context.context, device), "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(context.context), "cuCtxSetCurrent");

    const bool use_vmm = supports_vmm && !force_ordinary;
    CUmemAllocationProp props {};
    props.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    props.location.id = device;
    constexpr size_t ordinary_bytes = 1024 * 1024;
    constexpr size_t limit = 64 * ordinary_bytes;
    size_t granularity = 0;
    size_t bytes = ordinary_bytes;
    if (use_vmm) {
        check(cuMemGetAllocationGranularity(&granularity, &props, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                "cuMemGetAllocationGranularity");
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0 || !granularity || (granularity & (granularity - 1)) || granularity > limit) {
            throw std::runtime_error("unsupported host page size or VMM alignment");
        }
        const size_t host_page = size_t(page);
        if (host_page > limit || (host_page & (host_page - 1))) {
            throw std::runtime_error("unsupported host page alignment");
        }
        bytes = std::max({ordinary_bytes, host_page, granularity});
    }

    address_reservation range {0, use_vmm ? 2 * bytes : 0};
    if (use_vmm) {
        check(cuMemAddressReserve(&range.address, range.bytes, granularity, 0, 0), "cuMemAddressReserve");
    }
    llama_memory_budget budget(readback_bytes, bytes, false);
    {
        auto host_charge = budget.reserve(domain::host, category::io, readback_bytes);
        if (!host_charge) {
            throw std::runtime_error("host readback budget reservation failed");
        }
        std::vector<unsigned char> readback(readback_bytes);
        host_charge.commit();
        run_cycle(budget, range, props, bytes, 0x3c, use_vmm, readback);
        run_cycle(budget, range, props, bytes, 0xa7, use_vmm, readback);
    }
    if (use_vmm) {
        check(cuMemAddressFree(range.address, range.bytes), "cuMemAddressFree");
        range.address = 0;
    }
    const auto final = budget.snapshot(domain::device);
    const auto host = budget.snapshot(domain::host);
    if (final.high_water != bytes || final.released != 2 * bytes ||
            host.used() != 0 || host.released != readback_bytes) {
        throw std::runtime_error("unexpected ledger peak or released bytes");
    }
    std::printf("device=%d vmm_supported=%d mode=%s granularity_bytes=%zu address_bytes=%zu "
                "cycles=2 device_ledger_peak_bytes=%zu device_released_bytes=%zu host_ledger_peak_bytes=%zu\n",
            ordinal, supports_vmm, use_vmm ? "vmm" : "cuMemAlloc", granularity,
            range.bytes, final.high_water, final.released, host.high_water);
    return 0;
}

int main(int argc, char ** argv) {
    int ordinal = 0;
    bool force_ordinary = false;
    bool has_ordinal = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--no-vmm") == 0 && !force_ordinary) {
            force_ordinary = true;
        } else {
            char * end = nullptr;
            const long parsed = std::strtol(argv[i], &end, 10);
            if (has_ordinal || end == argv[i] || *end || parsed < 0 || parsed >= std::numeric_limits<int>::max()) {
                std::fprintf(stderr, "usage: llama-cuda-vmm-probe [DEVICE_ORDINAL] [--no-vmm]\n");
                return 1;
            }
            has_ordinal = true;
            ordinal = int(parsed);
        }
    }
    try {
        return probe(ordinal, force_ordinary);
    } catch (const std::exception & error) {
        std::fprintf(stderr, "CUDA VMM probe: %s\n", error.what());
        return 1;
    }
}
