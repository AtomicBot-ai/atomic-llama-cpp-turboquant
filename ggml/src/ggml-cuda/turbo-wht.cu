#include "turbo-wht.cuh"
#include "tq3-quant.cuh"

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
        int           direction) {
    const int64_t i01 = blockIdx.y;
    const int64_t i02 = blockIdx.z;
    const int64_t g   = blockIdx.x;
    if (i01 >= ne01 || i02 >= ne02 || g * QK_TQ3_0_GROUP >= ne00) return;

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

    const int64_t n_groups = ne00 / QK_TQ3_0_GROUP;

    dim3 grid(n_groups, ne01, ne02);
    dim3 threads(1, 1, 1);

    k_turbo_wht<<<grid, threads, 0, ctx.stream()>>>(
        (const char *)src0->data, (char *)dst->data,
        ne00, ne01, ne02,
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        direction);
}
