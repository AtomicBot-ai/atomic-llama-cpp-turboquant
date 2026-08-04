#include "models.h"
#include "llama-memory-recurrent.h"

// Ling 3.0 (BailingMoeV3): hybrid linear architecture, 5 KDA layers per 1 gated MLA layer,
// on top of a bailing MoE (sigmoid router with expert bias and group-limited routing).
//
// Differences to Kimi-Linear, which shares the KDA + MLA layout:
//   - KDA forget/output gates are full-rank projections (config: no_kda_lora = true)
//   - the KDA log-decay is clamped from below (config: kda_safe_gate, kda_lower_bound)
//   - MLA layers do use RoPE (partial, interleaved -> ggml NORM) and carry a head-wise
//     sigmoid output gate applied before the output projection

// safe_gate clamp of the KDA log-decay, config: kda_lower_bound
static const float BAILINGMOE3_KDA_LOWER_BOUND = -5.0f;

void llama_model_bailingmoe3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);

    // n_head_kv == 0 marks the KDA (recurrent) layers, the rest are full attention
    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;
    }

    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,        hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, hparams.n_ff_shexp, false);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,               hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,         hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,              hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,               hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,                hparams.expert_gating_func);
    ml.get_key(LLM_KV_EXPERT_GROUP_COUNT,                hparams.n_expert_groups, false);
    ml.get_key(LLM_KV_EXPERT_GROUP_USED_COUNT,           hparams.n_group_used, false);

    switch (hparams.n_layer()) {
        case 42: type = LLM_TYPE_124B_A5B; break; // Ling-3.0-flash
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_bailingmoe3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    const int64_t head_dim_kda = hparams.n_embd_head_kda;
    const int64_t d_inner      = head_dim_kda * n_head;
    const int64_t ssm_d_conv   = hparams.ssm_d_conv;

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        if (hparams.is_recr(i)) {
            // === KDA (linear attention) layer ===
            // conv1d weights are 4D in GGUF, quantization may drop the trailing 1
            layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_q_conv) {
                layer.ssm_q_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_Q, "weight", i), {ssm_d_conv, 1, d_inner}, 0);
            }
            layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_k_conv) {
                layer.ssm_k_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_K, "weight", i), {ssm_d_conv, 1, d_inner}, 0);
            }
            layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, d_inner, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_v_conv) {
                layer.ssm_v_conv = create_tensor(tn(LLM_TENSOR_SSM_CONV1D_V, "weight", i), {ssm_d_conv, 1, d_inner}, 0);
            }

            create_tensor_qkv(layer, i, n_embd, d_inner, d_inner, d_inner, 0);

            // full-rank forget/output gate projections (no_kda_lora)
            layer.ssm_f = create_tensor(tn(LLM_TENSOR_SSM_F, "weight", i), {n_embd, d_inner}, 0);
            layer.ssm_g = create_tensor(tn(LLM_TENSOR_SSM_G, "weight", i), {n_embd, d_inner}, 0);

            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, 0);

            // note: -exp(A_log) is already applied by the conversion script
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head, 1, 1}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_a) {
                layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head}, 0);
            }

            layer.ssm_dt_b   = create_tensor(tn(LLM_TENSOR_SSM_DT,   "bias",   i), {d_inner}, 0);
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {head_dim_kda}, 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {d_inner, n_embd}, 0);
        } else {
            // === gated MLA layer ===
            const int64_t kv_lora_rank      = hparams.n_lora_kv;
            const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
            const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
            const int64_t qk_rope_head_dim  = hparams.n_rot();

            // Ling 3.0 has no query compression (q_lora_rank = null)
            layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_head * n_embd_head_k_mla}, 0);

            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);
            layer.wkv_a_mqa      = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA,  "weight", i), {n_embd, kv_lora_rank + qk_rope_head_dim}, 0);

            layer.wkv_b = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i),
                {kv_lora_rank, n_head * (n_embd_head_k_mla - qk_rope_head_dim + n_embd_head_v_mla)}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!layer.wkv_b) { // MLA KV cache enabled
                layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_k_mla - qk_rope_head_dim, kv_lora_rank, n_head}, 0);
                layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);
            }

            // head-wise output gate (gated_attention_proj_granularity_type = "head_wise")
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head}, 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            const int64_t n_ff_exp   = hparams.n_ff_exp;
            const int64_t n_ff_shexp = hparams.n_ff_shexp > 0 ? hparams.n_ff_shexp : n_ff_exp * hparams.n_expert_shared;

            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_bailingmoe3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// projection + causal conv1d + silu for one of Q, K, V (qkv: 0 = Q, 1 = K, 2 = V)
static ggml_tensor * bailingmoe3_causal_conv1d(
        ggml_cgraph * gf, ggml_context * ctx0,
        ggml_tensor * conv_states_all, ggml_tensor * conv_state_all,
        int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w,
        int64_t d_conv, int64_t head_dim, int64_t n_head,
        int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner         = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total  = 3 * conv_state_size; // Q + K + V

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1) * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);
    ggml_tensor * x_3d   = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);

    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    ggml_tensor * last_conv_x = ggml_view_3d(ctx0, conv_x, d_conv - 1, d_inner, n_seqs,
        conv_x->nb[1], conv_x->nb[2], n_seq_tokens * conv_x->nb[0]);
    ggml_build_forward_expand(gf,
        ggml_cpy(ctx0, last_conv_x,
            ggml_view_3d(ctx0, conv_states_all,
                d_conv - 1, d_inner, n_seqs,
                (d_conv - 1) * ggml_element_size(conv_states_all),
                n_embd_r_total * ggml_element_size(conv_states_all),
                (kv_head * n_embd_r_total + qkv * conv_state_size) * ggml_element_size(conv_states_all))));

    ggml_tensor * conv_weight = ggml_reshape_2d(ctx0, conv_w, d_conv, d_inner);

    ggml_tensor * Xcur = ggml_ssm_conv(ctx0, conv_x, conv_weight);
    Xcur = ggml_reshape_2d(ctx0, Xcur, d_inner, n_tokens);
    Xcur = ggml_silu(ctx0, Xcur);

    return ggml_reshape_4d(ctx0, Xcur, head_dim, n_head, n_seq_tokens, n_seqs);
}

llama_model_bailingmoe3::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "inp_embd", -1);

    // MLA layers use RoPE, so positions are needed
    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_kv      = !hparams.is_mla() ? build_inp_mem_hybrid()   : nullptr;
    auto * inp_k       =  hparams.is_mla() ? build_inp_mem_hybrid_k() : nullptr;
    auto * inp_rs      =  hparams.is_mla() ? inp_k->get_recr() : inp_kv->get_recr();
    auto * inp_attn_kv = !hparams.is_mla() ? inp_kv->get_attn() : nullptr;
    auto * inp_attn_k  =  hparams.is_mla() ? inp_k->get_attn()  : nullptr;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head      = hparams.n_head();
    const int64_t head_dim    = hparams.n_embd_head_kda;
    const int64_t d_conv      = hparams.ssm_d_conv;
    const int64_t d_inner     = n_head * head_dim;
    const int64_t n_seqs      = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    const int64_t n_embd_head_k_mla  = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla  = hparams.n_embd_head_v_mla();
    const int64_t kv_lora_rank       = hparams.n_lora_kv;
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;

    const float kq_scale_mla = 1.0f / sqrtf(float(n_embd_head_k_mla));

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        ggml_tensor * inpSA = inpL;

        cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // the MLA output gate is computed from this same normed hidden state
        ggml_tensor * attn_inp = cur;

        if (hparams.is_recr(il)) {
            // === KDA ===
            const auto * mctx_cur = inp_rs->mctx;
            const auto kv_head = mctx_cur->get_head();

            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            ggml_tensor * conv_state_all  = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);

            ggml_tensor * Qcur = bailingmoe3_causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Kcur = bailingmoe3_causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Vcur = bailingmoe3_causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);

            // g1 = -exp(A_log) * softplus(f_proj(x) + dt_bias), clamped from below (safe_gate)
            ggml_tensor * g1 = ggml_mul_mat(ctx0, layer.ssm_f, cur);
            g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);
            g1 = ggml_softplus(ctx0, g1);
            g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head, n_tokens);

            ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
            g1 = ggml_mul(ctx0, g1, A);
            g1 = ggml_clamp(ctx0, g1, BAILINGMOE3_KDA_LOWER_BOUND, INFINITY);
            cb(g1, "kda_g1", il);

            g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs);

            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
            beta = ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs);
            beta = ggml_sigmoid(ctx0, beta);
            cb(beta, "kda_beta", il);

            cur = ggml_reshape_3d(ctx0, cur, cur->ne[0], n_seq_tokens, n_seqs);

            ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

            const float eps_norm = hparams.f_norm_rms_eps;

            Qcur = ggml_l2_norm(ctx0, Qcur, eps_norm);
            Kcur = ggml_l2_norm(ctx0, Kcur, eps_norm);

            auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);

            ggml_tensor * output    = ggml_cont(ctx0, attn_out.first);
            ggml_tensor * new_state = attn_out.second;

            ggml_build_forward_expand(gf,
                ggml_cpy(ctx0, new_state,
                    ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                                 kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

            // output gate: RMSNorm(o) * sigmoid(g_proj(x))
            ggml_tensor * cur_2d = ggml_reshape_2d(ctx0, cur, cur->ne[0], n_seq_tokens * n_seqs);
            ggml_tensor * g2 = ggml_mul_mat(ctx0, layer.ssm_g, cur_2d);
            g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head, n_seq_tokens * n_seqs);

            ggml_tensor * attn_out_final = ggml_reshape_3d(ctx0, output, head_dim, n_head, n_seq_tokens * n_seqs);
            ggml_tensor * normed = build_norm(attn_out_final, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
            ggml_tensor * gated  = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, g2));

            gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
            cur = ggml_mul_mat(ctx0, layer.wo, gated);
            cb(cur, "kda_out", il);
        } else {
            // === gated MLA ===
            ggml_tensor * q = ggml_mul_mat(ctx0, layer.wq, cur);
            cb(q, "q", il);

            ggml_tensor * q_nope = ggml_view_3d(ctx0, q, n_embd_head_qk_nope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head_k_mla),
                ggml_row_size(q->type, n_embd_head_k_mla) * n_head, 0);
            ggml_tensor * q_pe = ggml_view_3d(ctx0, q, n_embd_head_qk_rope, n_head, n_tokens,
                ggml_row_size(q->type, n_embd_head_k_mla),
                ggml_row_size(q->type, n_embd_head_k_mla) * n_head,
                ggml_row_size(q->type, n_embd_head_qk_nope));

            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);

            ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));

            // note: HF applies rope on de-interleaved pairs, which is ggml's NORM rope
            q_pe = ggml_rope_ext(ctx0, q_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                 freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(q_pe, "q_pe", il);

            k_pe = ggml_rope_ext(ctx0, k_pe, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                 freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
            cb(k_pe, "k_pe", il);

            kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);
            cb(kv_cmpr, "kv_cmpr", il);

            if (layer.wk_b && layer.wv_b) { // MLA KV cache enabled
                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);

                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

                // note: rope must go first for in-place context shifting in build_rope_shift()
                ggml_tensor * Qcur = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);

                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                ggml_tensor * Vcur = kv_cmpr;

                // no output projection here - the gate is applied first
                cur = build_attn(inp_attn_k, nullptr, NULL, nullptr,
                                 Qcur, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale_mla, il);
            } else { // MLA KV cache disabled, fall back to MHA
                ggml_tensor * Qcur = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q), n_embd_head_k_mla, n_head, n_tokens);

                ggml_tensor * kv = ggml_mul_mat(ctx0, layer.wkv_b, kv_cmpr);
                const int64_t kv_per_head = n_embd_head_qk_nope + n_embd_head_v_mla;

                ggml_tensor * k_nope = ggml_view_3d(ctx0, kv, n_embd_head_qk_nope, n_head, n_tokens,
                    ggml_row_size(kv->type, kv_per_head),
                    ggml_row_size(kv->type, kv_per_head * n_head), 0);
                ggml_tensor * Vcur = ggml_view_3d(ctx0, kv, n_embd_head_v_mla, n_head, n_tokens,
                    ggml_row_size(kv->type, kv_per_head),
                    ggml_row_size(kv->type, kv_per_head * n_head),
                    ggml_row_size(kv->type, n_embd_head_qk_nope));
                Vcur = ggml_cont(ctx0, Vcur);

                ggml_tensor * k_pe_target   = ggml_new_tensor_3d(ctx0, k_pe->type, n_embd_head_qk_rope, n_head, n_tokens);
                ggml_tensor * k_pe_repeated = ggml_repeat(ctx0, k_pe, k_pe_target);
                ggml_tensor * Kcur = ggml_concat(ctx0, k_nope, k_pe_repeated, 0);

                cur = build_attn(inp_attn_kv, nullptr, NULL, nullptr,
                                 Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale_mla, il);
            }
            cb(cur, "attn_out_pre_gate", il);

            // head-wise sigmoid gate, computed from the same normed hidden state as q/kv
            ggml_tensor * gate = ggml_mul_mat(ctx0, layer.wqkv_gate, attn_inp);
            gate = ggml_sigmoid(ctx0, gate);
            cb(gate, "attn_gate", il);

            cur  = ggml_reshape_3d(ctx0, cur,  n_embd_head_v_mla, n_head, n_tokens);
            gate = ggml_reshape_3d(ctx0, gate, 1,                 n_head, n_tokens);
            cur  = ggml_mul(ctx0, cur, gate);
            cur  = ggml_reshape_2d(ctx0, cur, n_embd_head_v_mla * n_head, n_tokens);
            cb(cur, "attn_gated", il);

            cur = ggml_mul_mat(ctx0, layer.wo, cur);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0, cur,   inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            cur = build_ffn(cur,
                layer.ffn_up,   NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                hparams.n_expert,
                hparams.n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   NULL, NULL,
                    layer.ffn_gate_shexp, NULL, NULL,
                    layer.ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
