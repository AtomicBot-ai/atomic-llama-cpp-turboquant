// Fork-owned early route publication. End the encoder before publishing;
// preserve the existing expert kernels, reader pool, and lease ownership.
#include "ggml-metal-route-publication.h"

struct atomic_route_channel {
    id<MTLSharedEvent> event;
    id<MTLBuffer> payload;
    atomic_uint_fast64_t sequence;
};

void * ggml_metal_route_create(void * device) {
    ggml_metal_device_t dev = device;
    struct atomic_route_channel * c = calloc(1, sizeof(*c));
    if (!c) return NULL;
    c->event = [dev->mtl_device newSharedEvent];
    c->payload = [dev->mtl_device newBufferWithLength:16384 options:MTLResourceStorageModeShared];
    if (!c->event || !c->payload) { ggml_metal_route_destroy(c); return NULL; }
    return c;
}

void ggml_metal_route_destroy(void * channel) {
    struct atomic_route_channel * c = channel;
    if (c) { [c->event release]; [c->payload release]; free(c); }
}

int ggml_metal_route_read(void * channel, uint64_t sequence, int32_t * ids, size_t count) {
    struct atomic_route_channel * c = channel;
    if (!c || !count || count > GGML_METAL_ROUTE_MAX_IDS ||
            atomic_load_explicit(&c->sequence, memory_order_acquire) != sequence) return -1;
    if (![c->event waitUntilSignaledValue:sequence timeoutMS:1]) return 0;
    if (atomic_load_explicit(&c->sequence, memory_order_acquire) != sequence) return -1;
    memcpy(ids, c->payload.contents, count * sizeof(int32_t));
    return 1;
}

// Called after the marked CONT has been encoded. End the compute encoder before
// copying/publishing, and order subsequent shared work after the publication.
void ggml_metal_encoder_publish_routes(ggml_metal_encoder_t enc, struct ggml_metal_buffer_id source,
        void * channel, void (*arm)(void *, uint64_t, uint64_t, size_t), void * user,
        uint64_t generation, size_t count) {
    struct atomic_route_channel * c = channel;
    [enc->obj endEncoding];
    [enc->obj release];
    id<MTLBlitCommandEncoder> blit = [enc->cmd_buf blitCommandEncoder];
    [blit copyFromBuffer:source.metal sourceOffset:source.offs toBuffer:c->payload
        destinationOffset:0 size:count * sizeof(int32_t)];
    [blit endEncoding];
    const uint64_t sequence = atomic_fetch_add_explicit(&c->sequence, 1, memory_order_release) + 1;
    [enc->cmd_buf encodeSignalEvent:c->event value:sequence];
    [enc->cmd_buf encodeWaitForEvent:c->event value:sequence];
    enc->obj = enc->concurrent ? [enc->cmd_buf computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]
                              : [enc->cmd_buf computeCommandEncoder];
    [enc->obj retain];
    arm(user, sequence, generation, count);
}
