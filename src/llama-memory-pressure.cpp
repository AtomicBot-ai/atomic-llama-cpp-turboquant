#include "llama-memory-pressure.h"

#include <algorithm>
#include <atomic>

#if defined(__APPLE__)
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#endif

struct llama_memory_pressure_monitor::impl {
    std::atomic<llama_memory_pressure> level {llama_memory_pressure::unknown};
    std::atomic<llama_memory_pressure> peak {llama_memory_pressure::unknown};
    llama_memory_pressure_sample cached;
    uint64_t sampled_at = 0;
#if defined(__APPLE__)
    dispatch_queue_t queue = nullptr;
    dispatch_source_t source = nullptr;
    dispatch_semaphore_t stopped = nullptr;
    impl() {
        queue = dispatch_queue_create("atomic-engine.memory-pressure", DISPATCH_QUEUE_SERIAL);
        stopped = dispatch_semaphore_create(0);
        if (queue && stopped) { source = dispatch_source_create(DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
            DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL, queue); }
        if (source) {
            dispatch_set_context(source, this);
            dispatch_source_set_event_handler_f(source, [](void * user) {
                auto & self = *static_cast<impl *>(user);
                const auto flags = dispatch_source_get_data(self.source);
                const auto value = flags & DISPATCH_MEMORYPRESSURE_CRITICAL ? llama_memory_pressure::critical :
                                   flags & DISPATCH_MEMORYPRESSURE_WARN ? llama_memory_pressure::warning :
                                   flags & DISPATCH_MEMORYPRESSURE_NORMAL ? llama_memory_pressure::normal : llama_memory_pressure::unknown;
                self.level.store(value);
                auto old = self.peak.load();
                while (old < value && !self.peak.compare_exchange_weak(old, value)) {}
            });
            dispatch_source_set_cancel_handler_f(source, [](void * user) {
                dispatch_semaphore_signal(static_cast<impl *>(user)->stopped);
            });
            dispatch_resume(source);
        }
    }
    ~impl() {
        if (source) {
            dispatch_source_cancel(source);
            // Cancellation runs after all handlers: no callback can outlive us.
            dispatch_semaphore_wait(stopped, DISPATCH_TIME_FOREVER);
            dispatch_sync_f(queue, nullptr, [](void *) {});
            dispatch_release(source);
        }
        if (stopped) { dispatch_release(stopped); }
        if (queue) { dispatch_release(queue); }
    }
    bool available() const { return source != nullptr; }
#else
    bool available() const { return false; }
#endif
};

llama_memory_pressure_monitor::llama_memory_pressure_monitor() : pimpl(new impl) {}
llama_memory_pressure_monitor::~llama_memory_pressure_monitor() = default;
bool llama_memory_pressure_monitor::available() const { return pimpl->available(); }

llama_memory_pressure_sample llama_memory_pressure_monitor::poll(uint64_t now) {
#if defined(__APPLE__)
    if (!pimpl->sampled_at || now - pimpl->sampled_at >= 1000) {
        vm_statistics64_data_t vm {};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        vm_size_t page_size = 0;
        const auto host = mach_host_self();
        pimpl->cached.headroom_valid = host_page_size(host, &page_size) == KERN_SUCCESS &&
            host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) == KERN_SUCCESS;
        mach_port_deallocate(mach_task_self(), host);
        pimpl->cached.headroom = pimpl->cached.headroom_valid ? size_t(vm.free_count) * page_size : 0;
        pimpl->sampled_at = now;
    }
#endif
    auto result = pimpl->cached;
    result.level = std::max(pimpl->level.load(), pimpl->peak.exchange(llama_memory_pressure::unknown));
    return result;
}
