// The retained bank is the only new input contract. The route map fits in the
// existing MUL_MAT_ID output workspace and expires at the end of this operation.
static bool ggml_metal_encode_inkling_groups(ggml_metal_op_t ctx, ggml_tensor * op) {
    const auto * w = op->src[0];
    const auto * x = op->src[1];
    const auto * ids = op->src[2];
    const int n = int(ids->ne[1]);
    if (op->src[3] || ids->ne[0] != 8 || n < 2 || n > 5 || x->type != GGML_TYPE_F32 ||
            op->type != GGML_TYPE_F32 || !ggml_is_contiguous(op) || !ggml_is_matrix(ids) ||
            ids->nb[0] != sizeof(int32_t) || x->ne[0] != w->ne[0] || x->nb[0] != sizeof(float) ||
            (x->ne[1] != 1 && x->ne[1] != 8) || x->ne[2] != n || x->ne[3] != 1 ||
            x->nb[1] % 16 || x->nb[2] % 16 || w->ne[0] % 16 || w->ne[1] % 8 ||
            w->ne[2] != 128 || w->ne[3] != 1 || !ggml_is_contiguous(w)) {
        return false;
    }
    const char * kernel;
    switch (w->type) {
        case GGML_TYPE_IQ2_XXS: kernel = "inkling_group_iq2_xxs"; break;
        case GGML_TYPE_Q2_0:    kernel = "inkling_group_q2_0";    break;
        case GGML_TYPE_IQ4_NL:  kernel = "inkling_group_iq4_nl";  break;
        case GGML_TYPE_IQ3_S:   kernel = "inkling_group_iq3_s";   break;
        // Q4_K gate/up uses the native path to preserve greedy verification on handover.
        case GGML_TYPE_Q5_0:    kernel = "inkling_group_q5_0";    break;
        default: return false;
    }
    const auto pipeline = [&](const char * name) {
        auto p = ggml_metal_library_get_pipeline(ctx->lib, name);
        if (!p.pipeline) {
            p = ggml_metal_library_compile_pipeline(ctx->lib, name, name, nullptr);
            if (p.pipeline) { GGML_LOG_INFO("Inkling grouped expert pipeline: %s\n", name); }
        }
        return p;
    };
    auto plan = pipeline("inkling_group_routes");
    auto project = pipeline(kernel);
    if (!plan.pipeline || !project.pipeline) { return false; }

    const int routes = 8 * n;
    auto output = ggml_metal_get_buffer_id(op);
    auto map = output;
    map.offs += ggml_nbytes(op);
    GGML_ASSERT(2 * routes * sizeof(int32_t) <= ggml_metal_op_mul_mat_id_extra_tpe(op));
    uint32_t map_args[] = {uint32_t(routes), uint32_t(ids->nb[1] / sizeof(int32_t))};
    auto enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, plan);
    ggml_metal_encoder_set_bytes(enc, map_args, sizeof(map_args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(ids), 1);
    ggml_metal_encoder_set_buffer(enc, map, 2);
    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 64, 1, 1);
    ggml_metal_op_concurrency_reset(ctx);

    uint64_t args[] = {uint64_t(w->ne[0]), uint64_t(w->ne[1]), w->nb[1], w->nb[2],
                       uint64_t(x->ne[1]), x->nb[1], x->nb[2]};
    ggml_metal_encoder_set_pipeline(enc, project);
    ggml_metal_encoder_set_bytes(enc, args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(w), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x), 2);
    ggml_metal_encoder_set_buffer(enc, map, 3);
    ggml_metal_encoder_set_buffer(enc, output, 4);
    ggml_metal_encoder_dispatch_threadgroups(enc, w->ne[1]/8, routes, 1, 32, 8, 1);
    return true;
}
