#include "turbo-wht.cuh"
#include "tq3-quant.cuh"

// Each thread independently transforms one 128-element group.
// We flatten the 3D grid (n_groups × ne01 × ne02) into a 1D index and
// launch with many threads per block for high occupancy.
static __global__ void k_turbo_wht(
        const char * __restrict__ src_base,
        char       * __restrict__ dst_base,
        const int64_t ne00,
        const int64_t ne01,
        const int64_t ne02,
        const int64_t nb00,
        const int64_t nb01,
        const int64_t nb02,
        const int64_t nb03,
        const int64_t total_groups,
        const int64_t groups_per_row,
        int           direction) {
    const int64_t gid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= total_groups) return;

    // Decompose flat index → (group_in_row, i01, i02)
    const int64_t g   = gid % groups_per_row;
    const int64_t rem = gid / groups_per_row;
    const int64_t i01 = rem % ne01;
    const int64_t i02 = rem / ne01;

    const float * row = (const float *)(src_base + i01 * nb01 + i02 * nb02) + g * QK_TQ3_0_GROUP;
    float * out_row   = (float *)(dst_base + i01 * nb01 + i02 * nb02) + g * QK_TQ3_0_GROUP;

    float x[128];
    for (int i = 0; i < 128; i++) x[i] = row[i];

    if (direction == 0) {
        tq3_rotate_forward(x);
    } else {
        tq3_rotate_inverse(x);
    }

    for (int i = 0; i < 128; i++) out_row[i] = x[i];
}

void ggml_cuda_op_turbo_wht(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    int direction;
    memcpy(&direction, dst->op_params, sizeof(int));

    const int64_t ne00 = src0->ne[0];
    const int64_t ne01 = src0->ne[1];
    const int64_t ne02 = src0->ne[2];
    GGML_ASSERT(ne00 % QK_TQ3_0_GROUP == 0);

    const int64_t groups_per_row = ne00 / QK_TQ3_0_GROUP;
    const int64_t total_groups   = groups_per_row * ne01 * ne02;

    // Pack many independent groups into each block for high GPU occupancy.
    // Each thread processes one 128-element group using 128 registers.
    constexpr int THREADS_PER_BLOCK = 128;
    const int n_blocks = (int)((total_groups + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    k_turbo_wht<<<n_blocks, THREADS_PER_BLOCK, 0, ctx.stream()>>>(
        (const char *)src0->data, (char *)dst->data,
        ne00, ne01, ne02,
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        total_groups, groups_per_row,
        direction);
}
