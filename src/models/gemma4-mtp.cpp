#include "models.h"

void llama_model_gemma4_mtp::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_MTP_BACKBONE_EMBEDDING_LENGTH, hparams.mtp_backbone_n_embd);
    ml.get_key(LLM_KV_MTP_ORDERED_EMBEDDINGS,        hparams.mtp_use_ordered_embeddings, false);
    ml.get_key(LLM_KV_MTP_CENTROID_COUNT,            hparams.mtp_num_centroids,         false);
    ml.get_key(LLM_KV_MTP_CENTROID_TOP_K,            hparams.mtp_centroid_top_k,        false);
}

void llama_model_gemma4_mtp::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t n_backbone = hparams.mtp_backbone_n_embd;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    mtp_pre_proj       = create_tensor(tn(LLM_TENSOR_MTP_PRE_PROJ,      "weight"), {n_embd + n_backbone, n_embd}, 0);
    mtp_post_proj      = create_tensor(tn(LLM_TENSOR_MTP_POST_PROJ,     "weight"), {n_embd, n_backbone}, 0);
    mtp_token_ordering = create_tensor(tn(LLM_TENSOR_MTP_TOKEN_ORDERING, "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    mtp_centroids      = create_tensor(tn(LLM_TENSOR_MTP_CENTROIDS,     "weight"), {n_embd, hparams.mtp_num_centroids}, TENSOR_NOT_REQUIRED);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm       = create_tensor(tn(LLM_TENSOR_ATTN_NORM,       "weight", i), {n_embd}, 0);
        layer.wq              = create_tensor(tn(LLM_TENSOR_ATTN_Q,          "weight", i), {n_embd, n_embd_head_k * n_head}, 0);
        layer.attn_q_norm     = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,     "weight", i), {n_embd_head_k * n_head}, 0);
        layer.wo              = create_tensor(tn(LLM_TENSOR_ATTN_OUT,        "weight", i), {n_embd_head_k * n_head, n_embd}, 0);
        layer.attn_post_norm  = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);

        layer.ffn_norm        = create_tensor(tn(LLM_TENSOR_FFN_NORM,        "weight", i), {n_embd}, 0);
        layer.ffn_gate        = create_tensor(tn(LLM_TENSOR_FFN_GATE,        "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down        = create_tensor(tn(LLM_TENSOR_FFN_DOWN,        "weight", i), {n_ff, n_embd}, 0);
        layer.ffn_up          = create_tensor(tn(LLM_TENSOR_FFN_UP,          "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_post_norm   = create_tensor(tn(LLM_TENSOR_FFN_POST_NORM,  "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);

        layer.layer_out_norm  = create_tensor(tn(LLM_TENSOR_LAYER_OUT_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);
        layer.layer_out_scale = create_tensor(tn(LLM_TENSOR_LAYER_OUT_SCALE, "weight", i), {1}, TENSOR_NOT_REQUIRED);
    }
}

std::unique_ptr<llm_graph_context> llama_model_gemma4_mtp::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_gemma4_mtp::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_backbone = hparams.mtp_backbone_n_embd;
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_backbone > 0);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // hidden state from target model
    ggml_tensor * hidden_state = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_backbone, n_tokens);
    ggml_set_name(hidden_state, "inp_mtp_states");
    ggml_set_input(hidden_state);

    // inp_tokens for this MTP model
    ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(inp_tokens, "inp_tokens");
    ggml_set_input(inp_tokens);

    ggml_tensor * inp_pos = build_inp_pos();
    auto * inp_attn = build_attn_inp_kv();

    // token embedding from target model's vocab
    ggml_tensor * token_embd = ggml_get_rows(ctx0, model.tok_embd, inp_tokens);
    cb(token_embd, "inp_embd_target", -1);
    token_embd = ggml_scale(ctx0, token_embd, std::sqrt(float(n_backbone)));
    cb(token_embd, "inp_embd_scaled", -1);

    // concat token embedding with hidden state
    cur = ggml_concat(ctx0, token_embd, hidden_state, 0);
    cb(cur, "inp_mtp_combined", -1);

    // pre-projection
    cur = build_lora_mm(model.mtp_pre_proj, cur);
    cb(cur, "mtp_pre_proj", -1);

    inpL = cur;

    const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            cb(Qcur, "Qcur", il);

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur_rope", il);

            // MTP uses frozen KV from target model - for now use standard attention
            // In full implementation, this would read from target_kv
            cur = build_attn(inp_attn,
                    model.layers[il].wo, NULL, model.layers[il].wo_s,
                    Qcur, nullptr, nullptr, nullptr, nullptr, nullptr,
                    kq_scale, il);
        }

        cur = build_norm(cur, model.layers[il].attn_post_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_post_norm", il);
        cur = ggml_add(ctx0, cur, inpSA);
        cb(cur, "attn_out", il);

        cur = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        ggml_tensor * ffn = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_GELU, LLM_FFN_PAR, il);
        cb(ffn, "ffn_out", il);

        cur = ffn;
        if (model.layers[il].layer_out_scale) {
            cur = ggml_mul(ctx0, cur, model.layers[il].layer_out_scale);
            cb(cur, "out_scaled", il);
        }
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "l_out_normed", -1);

    // MTP output projection
    ggml_tensor * mtp_embd = build_lora_mm(model.mtp_post_proj, cur);
    cb(mtp_embd, "result_mtp_embd", -1);
    ggml_set_output(mtp_embd);
    res->t_embd = mtp_embd;

    // logits using tied output weight
    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
