#include "models.h"

#include <cmath>
#include <cstring>

// Kimi-K3 MoonViT-3d, image path.
// Follows clip_graph_kimik25, but with RMSNorm, no biases, qkv width != n_embd, and a post-norm patchmergerv2 projector.
// Images only: at t == 1 the temporal pool and the temporal position term vanish.

ggml_tensor * clip_graph_kimik3::resize_position_embeddings_3d(uint32_t interpolation_mode) {
    ggml_tensor * pos_embd = model.position_embeddings;
    const int height       = img.ny() / patch_size;
    const int width        = img.nx() / patch_size;

    GGML_ASSERT(pos_embd);

    const int64_t stored_c = pos_embd->ne[0];
    const int64_t orig_w   = pos_embd->ne[1];
    const int64_t orig_h   = pos_embd->ne[2];

    GGML_ASSERT(stored_c == n_embd);

    if (height == (int) orig_h && width == (int) orig_w) {
        return ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);
    }

    pos_embd = ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
    pos_embd = ggml_interpolate(ctx0, pos_embd, height, width, n_embd, 1, interpolation_mode);
    pos_embd = ggml_permute(ctx0, pos_embd, 2, 1, 0, 3);
    pos_embd = ggml_cont_2d(ctx0, pos_embd, n_embd, width * height);
    return pos_embd;
}

ggml_cgraph * clip_graph_kimik3::build() {
    ggml_tensor * pos_h = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_h, "pos_h");
    ggml_set_input(pos_h);

    ggml_tensor * pos_w = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_patches);
    ggml_set_name(pos_w, "pos_w");
    ggml_set_input(pos_w);

    ggml_tensor * learned_pos_embd = resize_position_embeddings_3d(GGML_SCALE_MODE_BILINEAR);

    // Q/K are de-interleaved during conversion.
    auto add_pos = [&](ggml_tensor * cur, const clip_layer &) {
        return build_rope_2d(ctx0, cur, pos_w, pos_h, hparams.rope_theta, false);
    };

    ggml_tensor * inp = build_inp();
    inp = ggml_add(ctx0, inp, learned_pos_embd);

    ggml_tensor * cur = build_vit(
                            inp, n_patches,
                            NORM_TYPE_RMS,
                            hparams.ffn_op,
                            nullptr,
                            add_pos);
    cb(cur, "vit_out", -1);

    {
        const int scale_factor = model.hparams.n_merge;
        cur = build_patch_merge_permute(cur, scale_factor);

        cur = build_ffn(cur,
            model.mm_1_w, nullptr,
            nullptr,      nullptr,
            model.mm_2_w, nullptr,
            FFN_GELU,
            -1);
        cb(cur, "proj_mlp_out", -1);

        cur = build_norm(cur, model.mm_post_norm_w, nullptr, NORM_TYPE_RMS, hparams.eps, -1);
        cb(cur, "proj_out", -1);
    }

    ggml_build_forward_expand(gf, cur);

    return gf;
}
