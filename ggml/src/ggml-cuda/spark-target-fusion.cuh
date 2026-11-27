// Capture-time fusions for the resident width-one Inkling target graph.
static thread_local bool spark_target_capture = false;

static __global__ void spark_target_apply_clip(const float * sum,
                                               const float * input,
                                               float *       output,
                                               int           width,
                                               float         threshold) {
    const int row = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        const float divisor       = sqrtf(sum[row]) + 1e-6f;
        const float factor        = fminf(fmaxf(threshold / divisor, 0.0f), 1.0f);
        output[row * width + col] = input[row * width + col] * factor;
    }
}

// mmvf.cu at 6115c7090 uses pairs (2*lane, 2*lane+1) for the width-one FP32 rank-64 projection.
static __global__ void spark_target_project_kernel(const float *  cur,
                                                   const float *  msg,
                                                   const float2 * gate,
                                                   const float2 * gain,
                                                   const float2 * forget,
                                                   float *        dst) {
    const int    lane = threadIdx.x % 32, row = blockIdx.x * 8 + threadIdx.x / 32;
    const float2 x = gate[lane], g = gain[row * 32 + lane], f = forget[row * 32 + lane];
    float        a = 0, b = 0;
    ggml_cuda_mad(a, g.x, x.x);
    ggml_cuda_mad(a, g.y, x.y);
    ggml_cuda_mad(b, f.x, x.x);
    ggml_cuda_mad(b, f.y, x.y);
    a = warp_reduce_sum<32>(a);
    b = warp_reduce_sum<32>(b);
    if (!lane) {
        dst[row] = __fadd_rn(__fsub_rn(cur[row], __fmul_rn(tanhf(b), cur[row])), __fmul_rn(tanhf(a), msg[row]));
    }
}

static bool spark_target_try_project(ggml_backend_cuda_context * ctx, ggml_cgraph * graph, int i) {
    if (i + 2 >= graph->n_nodes) {
        return false;
    }
    auto ** op = graph->nodes + i;
    if (op[2]->op != GGML_OP_CUSTOM || !spark_verify_owned(op[2]) ||
        op[2]->op_params[spark_op_tag] != spark_target_control ||
        !ggml_can_fuse_subgraph(graph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_CUSTOM }, { i + 2 })) {
        return false;
    }
    auto *g = op[2]->src[2], *f = op[2]->src[3], *cur = op[2]->src[0], *msg = op[2]->src[1];
    if (!((g == op[0] && f == op[1]) || (g == op[1] && f == op[0])) || g->src[1] != f->src[1]) {
        return false;
    }
    auto *gate = g->src[1], *wg = g->src[0], *wf = f->src[0], *out = op[2];
    for (auto * t : { gate, wg, wf, cur, msg, out }) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || reinterpret_cast<uintptr_t>(t->data) % 8 != 0) {
            return false;
        }
        const uintptr_t x = reinterpret_cast<uintptr_t>(t->data), y = reinterpret_cast<uintptr_t>(out->data);
        if (t != out && !((t == cur || t == msg) && x == y) && x < y + ggml_nbytes(out) && y < x + ggml_nbytes(t)) {
            return false;
        }
    }
    if (ggml_nelements(cur) != 2304 || !ggml_are_same_shape(cur, msg) || !ggml_are_same_shape(cur, out) ||
        ggml_nelements(gate) != 64 || gate->ne[0] != 64 || wg->ne[0] != 64 || wg->ne[1] != 2304 ||
        !ggml_is_matrix(wg) || !ggml_are_same_shape(wg, wf)) {
        return false;
    }
    spark_target_project_kernel<<<288, 256, 0, ctx->stream()>>>(
        static_cast<const float *>(cur->data), static_cast<const float *>(msg->data),
        static_cast<const float2 *>(gate->data), static_cast<const float2 *>(wg->data),
        static_cast<const float2 *>(wf->data), static_cast<float *>(out->data));

    return true;
}

static bool spark_target_try_clip(ggml_backend_cuda_context * ctx, ggml_cgraph * graph, int start, bool include_sum) {
    const int i = start + (include_sum ? 2 : 0);
    if (include_sum && (!ggml_can_fuse_subgraph(graph, start,
                                                {
                                                    GGML_OP_SQR, GGML_OP_SUM_ROWS, GGML_OP_SQRT, GGML_OP_SCALE,
                                                    GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL
    },
                                                { start + 7 }) ||
                        !ggml_check_edges(graph, start, { { 1, 0, 0 }, { 2, 0, 1 } }))) {
        return false;
    }
    if (!ggml_can_fuse_subgraph(
            graph, i,
            {
                GGML_OP_SQRT, GGML_OP_SCALE, GGML_OP_FILL, GGML_OP_DIV, GGML_OP_CLAMP, GGML_OP_MUL
    },
            { i + 5 }) ||
        !ggml_check_edges(graph, i, { { 1, 0, 0 }, { 2, 0, 1 }, { 3, 0, 2 }, { 3, 1, 1 }, { 4, 0, 3 }, { 5, 1, 4 } })) {
        return false;
    }
    auto **      ops = graph->nodes + i;
    const auto * sum = ops[0]->src[0];
    const auto * cur = ops[5]->src[0];
    auto *       out = ops[5];
    if (sum->op != GGML_OP_SUM_ROWS || sum->src[0]->op != GGML_OP_SQR || sum->src[0]->src[0] != cur ||
        sum->ne[0] != 1 || ggml_nrows(sum) != ggml_nrows(cur) || !ggml_are_same_shape(cur, out) ||
        (cur->ne[0] != 128 && cur->ne[0] != 16) || cur->ne[2] != 1 || cur->ne[3] != 1 ||
        ggml_get_op_params_f32(ops[1], 0) != 1.0f || ggml_get_op_params_f32(ops[1], 1) != 1e-6f ||
        ggml_get_op_params_f32(ops[4], 0) != 0.0f || ggml_get_op_params_f32(ops[4], 1) != 1.0f) {
        return false;
    }
    for (const ggml_tensor * t : { sum, cur, static_cast<const ggml_tensor *>(out) }) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
            return false;
        }
    }
    auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const uintptr_t x = reinterpret_cast<uintptr_t>(a->data);
        const uintptr_t y = reinterpret_cast<uintptr_t>(b->data);
        return x < y + ggml_nbytes(b) && y < x + ggml_nbytes(a);
    };
    // One-to-one in-place scaling is safe. Broadcast sums must survive every thread's read.
    if ((!include_sum && overlaps(out, sum)) || (out->data != cur->data && overlaps(out, cur))) {
        return false;
    }
    if (include_sum) {
        const int nrows   = ggml_nrows(cur);
        const int threads = nrows / ggml_cuda_info().devices[ctx->device].nsm < 2 ? 512 : 32;
        spark_clip_kernel<<<nrows, threads, 0, ctx->stream()>>>(static_cast<const float *>(cur->data), nullptr,
                                                                static_cast<float *>(out->data), cur->ne[0], nrows,
                                                                ggml_get_op_params_f32(ops[2], 0));
        return true;
    }
    spark_target_apply_clip<<<dim3((cur->ne[0] + 127) / 128, ggml_nrows(cur)), 128, 0, ctx->stream()>>>(
        static_cast<const float *>(sum->data), static_cast<const float *>(cur->data), static_cast<float *>(out->data),
        cur->ne[0], ggml_get_op_params_f32(ops[2], 0));
    return true;
}
