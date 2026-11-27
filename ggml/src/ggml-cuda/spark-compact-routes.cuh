#pragma once

// The caller bounds the route list at 64 and launches one thread per slot.
static __global__ void spark_verify_routes(
        const int32_t * ids, int32_t * ids_src1, int32_t * ids_dst, ggml_cuda_mm_id_tile_spec * tiles,
        int routes, int experts_used, int channels, int width, int stride_ids, int stride_src, bool inverse) {
    __shared__ int experts[64];
    const int slot = threadIdx.x;
    const int token = slot / experts_used;
    const int lane = slot % experts_used;
    if (slot < routes) {
        experts[slot] = ids[token * stride_ids + lane];
    }
    for (int i = slot; i < width - 1; i += blockDim.x) {
        ids_dst[routes + i] = 0;
    }
    __syncthreads();
    if (slot >= routes) {
        return;
    }

    const int expert = experts[slot];
    int lower = 0;
    int prior = 0;
    int count = 0;
    for (int i = 0; i < routes; ++i) {
        lower += experts[i] < expert;
        prior += experts[i] == expert && i < slot;
        count += experts[i] == expert;
    }
    const int row = lower + prior;
    ids_dst[row] = slot;
    if (inverse) {
        ids_src1[slot] = row;
    } else {
        ids_src1[row] = token * stride_src + lane % channels;
    }
    tiles[slot] = prior % width == 0
        ? ggml_cuda_mm_id_tile_spec{expert, row, row + min(width, count - prior), 0}
        : ggml_cuda_mm_id_tile_spec{0, 0, 0, 0};
}
