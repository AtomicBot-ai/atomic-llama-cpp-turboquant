// Private single-sequence Inkling verify operations. The graph builder pins
// these nodes to the same backend that supplied their constructors.
constexpr int spark_op_tag = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
enum spark_verify_op { spark_conv = 0x53564331, spark_mask, spark_control, spark_depth, spark_clip, spark_control_project };

static void spark_verify_device_only(ggml_tensor *, int, int, void *) {
    GGML_ABORT("Spark verification op requires the CUDA backend");
}

static bool spark_verify_owned(const ggml_tensor * t) {
    ggml_custom_op_params params;
    memcpy(&params, t->op_params, sizeof(params));
    return params.fun == spark_verify_device_only &&
            t->op_params[spark_op_tag] >= spark_conv && t->op_params[spark_op_tag] <= spark_control_project;
}

// Match the reduction geometry of sum_rows_f32_cuda for the two Inkling head widths.
static __global__ void spark_clip_kernel(const float * input, const float * scale, float * output,
        int width, int heads, float threshold) {
    const int row = blockIdx.x;
    const int offset = row * width;
    const int scale_offset = (row % heads) * width;
    float values[4];
    float sum = 0.0f;
    for (int col = threadIdx.x, i = 0; col < width; col += blockDim.x, ++i) {
        const float value = input[offset + col] * (scale ? scale[scale_offset + col] : 1.0f);
        values[i] = value;
        sum = __fadd_rn(sum, __fmul_rn(value, value));
    }
    __shared__ float warp_sums[32];
    sum = block_reduce<block_reduce_method::SUM>(sum, warp_sums);
    sum = ggml_cuda_shfl(sum, 0);
    const float factor = fminf(fmaxf(threshold / (sqrtf(sum) + 1e-6f), 0.0f), 1.0f);
    for (int col = threadIdx.x, i = 0; col < width; col += blockDim.x, ++i) {
        output[offset + col] = values[i] * factor;
    }
}

static __global__ void spark_conv_kernel(const float * cur, const float * weights, const float * window,
        const float * taps, float * dst, int channels, int rows, int tap_width, float scale) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y;
    if (c >= channels) { return; }
    const float * x = window + c*(rows + 3) + row;
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) { sum = __fmaf_rn(x[i], weights[4*c+i], sum); }
    if (taps) {
        const int head = c / (channels / (tap_width/4));
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            sum = __fadd_rn(sum, __fmul_rn(x[3-i], taps[row*tap_width + head*4+i]));
        }
    }
    dst[row*channels+c] = (cur[row*channels+c] + sum)*scale;
}

template <typename T>
static __global__ void spark_mask_kernel(const float * table, const float * sink, const int * distance,
        const T * causal, half * dst, int cells, int rows, int heads, int extent) {
    const int cell = blockIdx.x*blockDim.x + threadIdx.x;
    if (cell >= cells) { return; }
    const int row = blockIdx.y, head = blockIdx.z;
    const int i = row*cells + cell;
    const int q = row*heads + head;
    dst[head*cells*rows + i] = __float2half_rn((table[q*extent + distance[i]] - sink[q]) + float(causal[i]));
}

static __global__ void spark_control_kernel(const float * cur, const float * message,
        const float * gain, const float * forget, float * dst, int n) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n) {
        const float forgotten = __fmul_rn(tanhf(forget[i]), cur[i]);
        const float gained = __fmul_rn(tanhf(gain[i]), message[i]);
        dst[i] = __fadd_rn(__fsub_rn(cur[i], forgotten), gained);
    }
}

// A warp projects one channel of both rank-64 controller matrices. Reuse the
// four loaded weight values across token positions before merging the residual.
static __global__ void spark_control_project_kernel(const float * cur, const float * message,
        const float * gate, const float * gain, const float * forget, float * dst, int rows) {
    const int channel = blockIdx.x * 8 + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int index = channel * 64 + lane;
    const float g0 = gain[index], g1 = gain[index + 32];
    const float f0 = forget[index], f1 = forget[index + 32];
    for (int row = 0; row < rows; ++row) {
        const float a = gate[row * 64 + lane], b = gate[row * 64 + lane + 32];
        const float g = warp_reduce_sum(__fmaf_rn(b, g1, a * g0));
        const float f = warp_reduce_sum(__fmaf_rn(b, f1, a * f0));
        if (lane == 0) {
            const int pos = row * 2304 + channel;
            dst[pos] = __fadd_rn(__fsub_rn(cur[pos], __fmul_rn(tanhf(f), cur[pos])),
                                 __fmul_rn(tanhf(g), message[pos]));
        }
    }
}

// Each block mixes one query and token across up to eight residual streams.
// Normalize each depth vector, score it, then use softmax across depth only.
struct spark_depth_states {
    const float * data[8];
};
static __global__ void spark_depth_kernel(spark_depth_states states, const float * queries, float * output,
        int depth, int width, int n_queries) {
    const int query = blockIdx.x, row = blockIdx.y, tid = threadIdx.x;
    __shared__ float warp_sums[32];
    __shared__ float factors[8], scores[8];
    for (int d = 0; d < depth; ++d) {
        const auto * x = states.data[d] + row*width;
        float sum = 0;
        for (int col = tid; col < width; col += blockDim.x) { sum = fmaf(x[col], x[col], sum); }
        sum = block_reduce<block_reduce_method::SUM>(sum, warp_sums);
        if (!tid) { factors[d] = rsqrtf(sum/width + 1e-6f); }
        __syncthreads();
        float dot = 0;
        for (int col = tid; col < width; col += blockDim.x) {
            dot = fmaf(x[col]*factors[d], queries[query*width+col], dot);
        }
        dot = block_reduce<block_reduce_method::SUM>(dot, warp_sums);
        if (!tid) { scores[d] = dot*rsqrtf(float(width)); }
        __syncthreads();
    }
    if (!tid) {
        float max_s = scores[0];
        for (int d=1; d<depth; ++d) { max_s = fmaxf(max_s,scores[d]); }
        float total = 0;
        for (int d=0; d<depth; ++d) { scores[d] = expf(scores[d]-max_s); total += scores[d]; }
        for (int d=0; d<depth; ++d) { scores[d] /= total; }
    }
    __syncthreads();
    for (int col = tid; col < width; col += blockDim.x) {
        float sum = 0;
        for (int d=0; d<depth; ++d) {
            sum = fmaf(states.data[d][row*width+col]*factors[d], scores[d], sum);
        }
        output[(row*n_queries+query)*width+col] = sum;
    }
}

static bool spark_verify_dispatch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (!spark_verify_owned(dst)) { return false; }
    const auto f = [&](int i) { return static_cast<const float *>(dst->src[i]->data); };
    const int width = dst->ne[0], rows = dst->ne[1];
    dim3 grid((width+255)/256, rows);
    switch (dst->op_params[spark_op_tag]) {
        case spark_clip: {
            const int nrows = ggml_nrows(dst);
            const int threads = nrows / ggml_cuda_info().devices[ctx.device].nsm < 2 ? 512 : 32;
            spark_clip_kernel<<<nrows, threads, 0, ctx.stream()>>>(f(0), dst->src[1] ? f(1) : nullptr,
                    static_cast<float *>(dst->data), width, rows, ggml_get_op_params_f32(dst, spark_op_tag-1));
        } break;
        case spark_depth: {
            spark_depth_states states{};
            int depth = 0;
            while (depth < 8 && dst->src[depth+1]) {
                states.data[depth] = f(depth+1);
                ++depth;
            }
            spark_depth_kernel<<<dim3(dst->ne[1],dst->ne[2]),256,0,ctx.stream()>>>(states, f(0),
                    static_cast<float *>(dst->data), depth, width, dst->ne[1]);
        } break;
        case spark_conv: {
            const auto * dyn = dst->src[3];
            spark_conv_kernel<<<grid,256,0,ctx.stream()>>>(f(0), f(1), f(2), dyn ? f(3) : nullptr,
                    static_cast<float *>(dst->data), width, rows, dyn ? dyn->ne[0] : 0,
                    ggml_get_op_params_f32(dst, spark_op_tag-1));
        } break;
        case spark_mask: {
            grid.z = dst->ne[2];
            auto * pos = static_cast<const int *>(dst->src[2]->data);
            auto * out = static_cast<half *>(dst->data);
            const int extent = dst->src[0]->ne[0];
            if (dst->src[3]->type == GGML_TYPE_F16) {
                spark_mask_kernel<<<grid,256,0,ctx.stream()>>>(f(0), f(1), pos,
                        static_cast<const half *>(dst->src[3]->data), out, width, rows, grid.z, extent);
            } else {
                spark_mask_kernel<<<grid,256,0,ctx.stream()>>>(f(0), f(1), pos, f(3), out, width, rows, grid.z, extent);
            }
        } break;
        case spark_control_project: {
            spark_control_project_kernel<<<2304/8, 256, 0, ctx.stream()>>>(f(0), f(1), f(2), f(3), f(4),
                    static_cast<float *>(dst->data), rows);
        } break;
        case spark_control: {
            const int n = width*rows;
            spark_control_kernel<<<(n+255)/256,256,0,ctx.stream()>>>(f(0), f(1), f(2), f(3), static_cast<float *>(dst->data), n);
        } break;
    }
    return true;
}

// The rel-mask, controller and four-tap convolution constructors follow the
// fork's Metal v1 contracts at 95b0cb23e. Track layout and sink changes there.
static ggml_tensor * ggml_backend_cuda_spark_inkling_clip_v1(ggml_context * ctx, ggml_tensor * cur,
        ggml_tensor * scale, float threshold) {
    if (cur->type != GGML_TYPE_F32 || !ggml_is_contiguous(cur) ||
            (cur->ne[0] != 16 && cur->ne[0] != 128) || cur->ne[3] != 1) { return nullptr; }
    if (scale && (scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) ||
            scale->ne[0] != cur->ne[0] || scale->ne[1] != cur->ne[1] || !ggml_is_matrix(scale))) {
        return nullptr;
    }
    ggml_tensor * args[] = {cur, scale};
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], cur->ne[1], cur->ne[2], 1,
            args, scale ? 2 : 1, spark_verify_device_only, 1, nullptr);
    ggml_set_op_params_f32(node, spark_op_tag-1, threshold);
    node->op_params[spark_op_tag] = spark_clip;
    return node;
}

static ggml_tensor * ggml_backend_cuda_spark_inkling_rel_mask_v1(ggml_context * ctx, ggml_tensor * table,
        ggml_tensor * sinks, ggml_tensor * distances, ggml_tensor * causal) {
    if (table->type != GGML_TYPE_F32 || sinks->type != GGML_TYPE_F32 || distances->type != GGML_TYPE_I32 ||
            (causal->type != GGML_TYPE_F16 && causal->type != GGML_TYPE_F32) || !ggml_are_same_shape(distances, causal) ||
            !ggml_is_contiguous(table) || !ggml_is_contiguous(sinks) ||
            !ggml_is_contiguous(distances) || !ggml_is_contiguous(causal) ||
            causal->ne[2] != 1 || causal->ne[3] != 1 || table->ne[3] != 1 ||
            table->ne[2] != causal->ne[1] || sinks->ne[0] != table->ne[1] ||
            sinks->ne[1] != table->ne[2] || ggml_nrows(sinks) != sinks->ne[1]) {
        return nullptr;
    }
    ggml_tensor * args[] = {table, sinks, distances, causal};
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F16, causal->ne[0], causal->ne[1], table->ne[1], 1,
            args, 4, spark_verify_device_only, 1, nullptr);
    node->op_params[spark_op_tag] = spark_mask;
    return node;
}

static ggml_tensor * ggml_backend_cuda_spark_inkling_control_v1(ggml_context * ctx, ggml_tensor * cur,
        ggml_tensor * message, ggml_tensor * gain, ggml_tensor * forget) {
    ggml_tensor * args[] = {cur, message, gain, forget};
    if (cur->ne[0] != 2304 || (cur->ne[1] < 1 || cur->ne[1] > 8) || !ggml_is_matrix(cur)) { return nullptr; }
    for (auto * arg : args) {
        if (arg->type != GGML_TYPE_F32 || !ggml_are_same_shape(arg, cur) || !ggml_is_contiguous(arg)) {
            return nullptr;
        }
    }
    static const bool project = [] {
        const char * mode = std::getenv("LLAMA_EXPERIMENTAL_SPARK_CONTROL_PROJECT");
        return mode && std::strcmp(mode, "1") == 0;
    }();
    if (project && gain->op == GGML_OP_MUL_MAT && forget->op == GGML_OP_MUL_MAT &&
            gain->src[1] == forget->src[1]) {
        auto * gate = gain->src[1];
        auto * wg = gain->src[0];
        auto * wf = forget->src[0];
        if (gate->type == GGML_TYPE_F32 && wg->type == GGML_TYPE_F32 && wf->type == GGML_TYPE_F32 &&
                ggml_is_matrix(gate) && gate->ne[0] == 64 && gate->ne[1] == cur->ne[1] &&
                ggml_is_matrix(wg) && wg->ne[0] == 64 && wg->ne[1] == 2304 && ggml_are_same_shape(wg, wf) &&
                ggml_is_contiguous(gate) && ggml_is_contiguous(wg) && ggml_is_contiguous(wf)) {
            ggml_tensor * inputs[] = {cur, message, gate, wg, wf};
            auto * result = ggml_custom_4d(ctx, GGML_TYPE_F32, 2304, cur->ne[1], 1, 1, inputs, 5,
                    spark_verify_device_only, 1, nullptr);
            result->op_params[spark_op_tag] = spark_control_project;
            return result;
        }
    }
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], cur->ne[1], 1, 1, args, 4,
            spark_verify_device_only, 1, nullptr);
    node->op_params[spark_op_tag] = spark_control;
    return node;
}

static ggml_tensor * ggml_backend_cuda_spark_inkling_conv_residual_v1(ggml_context * ctx, ggml_tensor * cur,
        ggml_tensor * weights, ggml_tensor * window, ggml_tensor * dynamic, float scale) {
    if (cur->type != GGML_TYPE_F32 || weights->type != GGML_TYPE_F32 || window->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(cur) || !ggml_is_contiguous(weights) || !ggml_is_contiguous(window) ||
            window->ne[2] != 1 || window->ne[3] != 1 || ggml_nrows(cur) != cur->ne[1] ||
            weights->ne[0] != 4 || ggml_nrows(weights) != cur->ne[0] ||
            window->ne[1] != cur->ne[0] || window->ne[0] != cur->ne[1] + 3) {
        return nullptr;
    }
    if (dynamic && (dynamic->type != GGML_TYPE_F32 || !ggml_is_contiguous(dynamic) ||
            ggml_nrows(dynamic) != dynamic->ne[1] || dynamic->ne[1] != cur->ne[1] ||
            dynamic->ne[0] <= 0 || dynamic->ne[0] % 4 != 0 || cur->ne[0] % (dynamic->ne[0] / 4) != 0)) {
        return nullptr;
    }
    ggml_tensor * args[] = {cur, weights, window, dynamic};
    auto * node = ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], cur->ne[1], 1, 1,
            args, dynamic ? 4 : 3,
            spark_verify_device_only, 1, nullptr);
    ggml_set_op_params_f32(node, spark_op_tag - 1, scale);
    node->op_params[spark_op_tag] = spark_conv;
    return node;
}

static ggml_tensor * ggml_backend_cuda_spark_depth(ggml_context * ctx, ggml_tensor * query,
        ggml_tensor * cur, ggml_tensor * const * blocks, size_t count) {
    if (count >= 8 || cur->ne[0] != 2304 || cur->ne[1] > 8 || !ggml_is_matrix(cur) ||
            cur->type != GGML_TYPE_F32 || !ggml_is_contiguous(cur) ||
            query->type != GGML_TYPE_F32 || !ggml_is_contiguous(query) || !ggml_is_matrix(query) ||
            query->ne[0] != cur->ne[0] || query->ne[1] > 4) { return nullptr; }
    ggml_tensor * args[GGML_MAX_SRC] = {query};
    for (size_t i = 0; i < count; ++i) {
        auto * b = blocks[count-i-1];
        if (b->type != cur->type || !ggml_is_contiguous(b) || !ggml_are_same_shape(b,cur)) { return nullptr; }
        args[1+i] = b;
    }
    args[count+1] = cur;
    auto * dst = ggml_custom_4d(ctx, GGML_TYPE_F32, cur->ne[0], query->ne[1], cur->ne[1], 1,
            args, int(count)+2, spark_verify_device_only, 1, nullptr);
    dst->op_params[spark_op_tag] = spark_depth;
    return dst;
}
