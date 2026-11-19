// Included inside the Metal device implementation to keep its private buffer
// layout in the backend. Allocation/free adapted from upstream 31953bcf74ba489a8bbb0c936c5fcf24fcf9485c.
// Track shared-buffer alignment, ownership, residency and get_id changes there.
#include "ggml-metal-expert-buffer.h"

struct ggml_metal_expert_group {
    id rset;
    size_t count;
    struct ggml_metal_expert_group * prev;
    struct ggml_metal_expert_group * next;
};

struct ggml_metal_expert_pool {
    ggml_metal_device_t dev;
    NSLock * lock;
    struct ggml_metal_expert_group * available;
    size_t live;
};

struct ggml_metal_expert_buffer {
    struct ggml_metal_buffer base; // retain the ordinary tensor lookup layout
    ggml_metal_expert_pool_t pool;
    struct ggml_metal_expert_group * group;
};

enum { GGML_METAL_EXPERT_GROUP_SIZE = 64 };

static void ggml_metal_expert_group_unlink(ggml_metal_expert_pool_t pool, struct ggml_metal_expert_group * group) {
    if (group->prev) { group->prev->next = group->next; }
    else { pool->available = group->next; }
    if (group->next) { group->next->prev = group->prev; }
    group->prev = group->next = NULL;
}

static void ggml_metal_expert_group_link(ggml_metal_expert_pool_t pool, struct ggml_metal_expert_group * group) {
    group->next = pool->available;
    if (group->next) { group->next->prev = group; }
    pool->available = group;
}

ggml_metal_expert_pool_t ggml_metal_expert_pool_new(ggml_metal_device_t dev) {
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        const struct ggml_metal_device_props * props = ggml_metal_device_get_props(dev);
        if (!props->use_residency_sets || !props->use_shared_buffers || !dev->rsets) { return NULL; }
        ggml_metal_expert_pool_t pool = calloc(1, sizeof(*pool));
        if (!pool) { return NULL; }
        pool->dev = dev;
        pool->lock = [[NSLock alloc] init];
        return pool;
    }
#endif
    GGML_UNUSED(dev);
    return NULL;
}

void ggml_metal_expert_pool_free(ggml_metal_expert_pool_t pool) {
    if (!pool) { return; }
    GGML_ASSERT(pool->live == 0 && pool->available == NULL);
    [pool->lock release];
    free(pool);
}

ggml_metal_buffer_t ggml_metal_expert_buffer_new(ggml_metal_expert_pool_t pool, size_t size) {
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        const size_t page = sysconf(_SC_PAGESIZE);
        if (!size || size > SIZE_MAX - (page - 1)) { return NULL; }
        const size_t aligned = (size + page - 1) / page * page;
        struct ggml_metal_expert_buffer * expert = calloc(1, sizeof(*expert));
        if (!expert) { return NULL; }
        ggml_metal_buffer_t buf = &expert->base;
        buf->dev = pool->dev;
        buf->all_size = aligned;
        buf->is_shared = buf->owned = true;
        buf->n_buffers = 1;
        buf->all_data = ggml_metal_host_malloc(aligned);
        if (!buf->all_data) { free(expert); return NULL; }
        buf->buffers[0].metal = [pool->dev->mtl_device newBufferWithBytesNoCopy:buf->all_data
            length:aligned options:MTLResourceStorageModeShared deallocator:nil];
        if (!buf->buffers[0].metal) {
            vm_deallocate(mach_task_self(), (vm_address_t) buf->all_data, aligned);
            free(expert);
            return NULL;
        }
        buf->buffers[0].data = buf->all_data;
        buf->buffers[0].size = size;
        expert->pool = pool;

        [pool->lock lock];
        struct ggml_metal_expert_group * group = pool->available;
        if (!group) {
            group = calloc(1, sizeof(*group));
            if (group) {
                MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
                desc.label = @"atomic_engine_experts";
                desc.initialCapacity = GGML_METAL_EXPERT_GROUP_SIZE;
                NSError * error = nil;
                group->rset = [pool->dev->mtl_device newResidencySetWithDescriptor:desc error:&error];
                [desc release];
                if (!group->rset) { free(group); group = NULL; }
                else {
                    ggml_metal_expert_group_link(pool, group);
                    ggml_metal_device_rsets_add(pool->dev, group->rset);
                }
            }
        }
        if (!group) {
            [pool->lock unlock];
            [buf->buffers[0].metal release];
            vm_deallocate(mach_task_self(), (vm_address_t) buf->all_data, aligned);
            free(expert);
            return NULL;
        }
        // MTLResidencySet methods are not thread-safe. Share the heartbeat's
        // lock, including commit/request, instead of racing its requestResidency.
        [pool->dev->rsets->lock lock];
        [group->rset addAllocation:buf->buffers[0].metal];
        [group->rset commit];
        [group->rset requestResidency];
        [pool->dev->rsets->lock unlock];
        expert->group = group;
        if (++group->count == GGML_METAL_EXPERT_GROUP_SIZE) { ggml_metal_expert_group_unlink(pool, group); }
        ++pool->live;
        [pool->lock unlock];
        return buf;
    }
#endif
    GGML_UNUSED(pool);
    GGML_UNUSED(size);
    return NULL;
}

void ggml_metal_expert_buffer_free(ggml_metal_buffer_t buf) {
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        @autoreleasepool {
            struct ggml_metal_expert_buffer * expert = (struct ggml_metal_expert_buffer *) buf;
            ggml_metal_expert_pool_t pool = expert->pool;
            struct ggml_metal_expert_group * group = expert->group;
            [pool->lock lock];
            [pool->dev->rsets->lock lock];
            // The caller's synchronized lease has ended; other allocations in this
            // group remain resident, including ones referenced by in-flight graphs.
            [group->rset removeAllocation:buf->buffers[0].metal];
            [group->rset commit];
            if (group->count-- == GGML_METAL_EXPERT_GROUP_SIZE) { ggml_metal_expert_group_link(pool, group); }
            if (!group->count) {
                ggml_metal_expert_group_unlink(pool, group);
                [pool->dev->rsets->data removeObject:group->rset];
                [group->rset endResidency];
                [group->rset release];
                free(group);
            }
            --pool->live;
            [pool->dev->rsets->lock unlock];
            [pool->lock unlock];
            [buf->buffers[0].metal release];
            vm_deallocate(mach_task_self(), (vm_address_t) buf->all_data, buf->all_size);
            free(expert);
            return;
        }
    }
#endif
    GGML_UNUSED(buf);
    GGML_ABORT("unsupported expert residency buffer");
}
