#include "models.h"
#include "llama-memory-recurrent.h"

// Kimi K3: hybrid KDA + gated-MLA (NoPE) with Attention Residuals (AttnRes),
// Stable LatentMoE and SiTU-GLU activation.
//
// Reference: moonshotai/Kimi-K3 modeling_kimi_linear.py (HF), sglang kimi_k3.py
//
// Deltas vs Kimi Linear (LLM_ARCH_KIMI_LINEAR):
//   - KDA safe gate:   g_log = gate_lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
//     (ssm_a stores exp(A_log) > 0, sliced to n_head at conversion)
//   - KDA output gate is full-rank: g2 = g_proj(x)  (wqkv_gate), not low-rank g_a/g_b
//   - MLA has an output gate: attn = attn * sigmoid(g_proj(x)) before o_proj
//   - MoE experts run in a latent space:  down -> experts -> weighted sum -> RMSNorm -> up
//     with the router operating on the full hidden state
//   - AttnRes: the residual stream restarts every attn_res_block_size layers; snapshots are
//     banked and re-mixed via a learned softmax mixture before attention, before the FFN
//     and at the model output

void llama_model_kimi_k3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,    hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,  hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,      hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,       hparams.n_lora_q);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,             hparams.ssm_d_conv);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                hparams.n_embd_head_kda);
    ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND,        hparams.kda_gate_lower_bound);

    ml.get_key(LLM_KV_SITU_BETA,                   hparams.situ_beta);
    ml.get_key(LLM_KV_SITU_LINEAR_BETA,            hparams.situ_linear_beta);
    ml.get_key(LLM_KV_ATTN_RES_BLOCK_SIZE,         hparams.attn_res_block_size);

    // Mark KDA layers as recurrent using the n_head_kv pattern (like Kimi Linear)
    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;
    }

    // Stable LatentMoE
    ml.get_key(LLM_KV_MOE_LATENT_SIZE,             hparams.moe_latent_size);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);

    // K3 always renormalizes the top-k sigmoid weights (moe_renormalize = true)
    hparams.expert_weights_norm = true;

    GGML_ASSERT(hparams.attn_res_block_size > 0 && "Kimi-K3 requires attn_res_block_size");
    GGML_ASSERT(hparams.moe_latent_size    > 0 && "Kimi-K3 requires moe_latent_size");

    switch (hparams.n_layer()) {
        case 93: type = LLM_TYPE_2_8T_A104B; break; // Kimi-K3
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_kimi_k3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    const int64_t moe_latent = hparams.moe_latent_size;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    // AttnRes output mixture
    output_res_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_NORM, "weight"), {n_embd}, 0);
    output_res_proj = create_tensor(tn(LLM_TENSOR_OUTPUT_RES_PROJ, "weight"), {n_embd}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        // AttnRes per-layer mixtures
        layer.attn_res_norm = create_tensor(tn(LLM_TENSOR_ATTN_RES_NORM, "weight", i), {n_embd}, 0);
        layer.attn_res_proj = create_tensor(tn(LLM_TENSOR_ATTN_RES_PROJ, "weight", i), {n_embd}, 0);
        layer.ffn_res_norm  = create_tensor(tn(LLM_TENSOR_FFN_RES_NORM,  "weight", i), {n_embd}, 0);
        layer.ffn_res_proj  = create_tensor(tn(LLM_TENSOR_FFN_RES_PROJ,  "weight", i), {n_embd}, 0);

        const int64_t n_embd_head_k_kda = hparams.n_embd_head_kda;
        const int64_t n_embd_head_v_kda = hparams.n_embd_head_kda;
        const int64_t ssm_d_conv = hparams.ssm_d_conv;
        const int64_t d_inner = n_embd_head_k_kda * n_head;

        if (hparams.is_recr(i)) {
            // === KDA layer ===
            // Conv1d weights: 4D [d_conv, 1, d_inner, 1] with 3D fallback
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

            // q, k, v projections
            create_tensor_qkv(layer, i, n_embd, d_inner, d_inner, d_inner, 0);

            // forget gate projections (low-rank)
            layer.ssm_f_a = create_tensor(tn(LLM_TENSOR_SSM_F_A, "weight", i), {n_embd, n_embd_head_k_kda}, 0);
            layer.ssm_f_b = create_tensor(tn(LLM_TENSOR_SSM_F_B, "weight", i), {n_embd_head_k_kda, d_inner}, 0);

            // b_proj (beta mixing coefficient)
            layer.ssm_beta = create_tensor(tn(LLM_TENSOR_SSM_BETA, "weight", i), {n_embd, n_head}, 0);

            // exp(A_log), per head (sliced from [head_dim] to [n_head] at conversion)
            layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {n_head}, TENSOR_NOT_REQUIRED);
            if (!layer.ssm_a) {
                layer.ssm_a = create_tensor(tn(LLM_TENSOR_SSM_A, i), {1, n_head}, 0);
            }

            // dt_bias
            layer.ssm_dt_b = create_tensor(tn(LLM_TENSOR_SSM_DT, "bias", i), {d_inner}, 0);

            // full-rank output gate (use_full_rank_gate = true)
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, d_inner}, 0);

            // o_norm
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM, "weight", i), {n_embd_head_k_kda}, 0);

            // o_proj
            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_v_kda * n_head, n_embd}, 0);
        } else {
            // === Gated MLA layer (NoPE) ===
            const int64_t q_lora_rank  = hparams.n_lora_q;
            const int64_t kv_lora_rank = hparams.n_lora_kv;
            const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
            const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
            const int64_t qk_rope_head_dim = hparams.n_rot();

            layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM, "weight", i), {q_lora_rank}, 0);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, 0);

            layer.wq_a = create_tensor(tn(LLM_TENSOR_ATTN_Q_A, "weight", i), {n_embd, q_lora_rank}, 0);
            layer.wq_b = create_tensor(tn(LLM_TENSOR_ATTN_Q_B, "weight", i), {q_lora_rank, n_head * n_embd_head_k_mla}, 0);

            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + qk_rope_head_dim}, 0);
            // Support legacy GGUFs that don't split wkv_b (MLA KV cache disabled)
            layer.wkv_b = create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i),
                {kv_lora_rank, n_head * (n_embd_head_k_mla - qk_rope_head_dim + n_embd_head_v_mla)}, TENSOR_NOT_REQUIRED | TENSOR_SKIP_IF_VIRTUAL);
            if (!layer.wkv_b) { // MLA KV cache enabled
                layer.wk_b = create_tensor(tn(LLM_TENSOR_ATTN_K_B, "weight", i), {n_embd_head_k_mla - qk_rope_head_dim, kv_lora_rank, n_head}, 0);
                layer.wv_b = create_tensor(tn(LLM_TENSOR_ATTN_V_B, "weight", i), {kv_lora_rank, n_embd_head_v_mla, n_head}, 0);
            }

            // output gate: attn = attn * sigmoid(g_proj(x)) before o_proj
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head * n_embd_head_v_mla}, 0);

            layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_head * n_embd_head_v_mla, n_embd}, 0);
        }

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        const int64_t n_ff_exp = hparams.n_ff_exp;

        if (i < (int) hparams.n_layer_dense_lead) {
            // Dense FFN layer (SiTU-GLU)
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
        } else {
            // Stable LatentMoE: router on full hidden, experts in the latent space
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);

            layer.ffn_latent_down = create_tensor(tn(LLM_TENSOR_FFN_LATENT_DOWN, "weight", i), {n_embd, moe_latent}, 0);
            layer.ffn_latent_norm = create_tensor(tn(LLM_TENSOR_FFN_LATENT_NORM, "weight", i), {moe_latent}, 0);
            layer.ffn_latent_up   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_UP,   "weight", i), {moe_latent, n_embd}, 0);

            layer.ffn_gate_exps = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS, "weight", i), {moe_latent, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, moe_latent, n_expert}, 0);
            layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {moe_latent, n_ff_exp, n_expert}, 0);

            // Shared experts operate on the full hidden state
            const int64_t n_ff_shexp = n_ff_exp * (hparams.n_expert_shared > 0 ? hparams.n_expert_shared : 1);
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_shexp}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {n_ff_shexp, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_shexp}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_kimi_k3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// Causal Conv1d for Q/K/V (identical to Kimi Linear)
// qkv: 0 = Q, 1 = K, 2 = V
static ggml_tensor * causal_conv1d(ggml_cgraph * gf, ggml_context * ctx0, ggml_tensor * conv_states_all, ggml_tensor * conv_state_all, int64_t qkv, ggml_tensor * x, ggml_tensor * proj_w, ggml_tensor * conv_w, int64_t d_conv, int64_t head_dim, int64_t n_head, int64_t n_seq_tokens, int64_t n_seqs, int64_t n_tokens, int64_t kv_head) {
    const int64_t d_inner = head_dim * n_head;
    const int64_t conv_state_size = (d_conv - 1) * d_inner;
    const int64_t n_embd_r_total = 3 * conv_state_size;  // Q + K + V

    ggml_tensor * conv_state_x = ggml_view_3d(ctx0, conv_state_all, d_conv - 1, d_inner, n_seqs,
        (d_conv - 1) * ggml_element_size(conv_state_all),
        n_embd_r_total * ggml_element_size(conv_state_all),
        qkv * conv_state_size * ggml_element_size(conv_state_all));

    ggml_tensor * x_proj = ggml_mul_mat(ctx0, proj_w, x);

    ggml_tensor * x_3d = ggml_reshape_3d(ctx0, x_proj, d_inner, n_seq_tokens, n_seqs);

    ggml_tensor * conv_x = ggml_concat(ctx0, conv_state_x, ggml_transpose(ctx0, x_3d), 0);

    // Save last (d_conv-1) columns back to the conv state
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

// AttnRes mixture: softmax over [bank rows..., prefix] scored by a scalar projection
// of the RMS-normed rows; the mixture combines the raw (unnormalized) rows.
//   k       = rms_norm(v)
//   scores  = sum(k * (norm_w * proj_w), dim=embd)
//   probs   = softmax(scores over rows)
//   out     = sum_r probs_r * v_r
ggml_tensor * llama_model_kimi_k3::graph::build_attn_res_mix(
        ggml_tensor * prefix,
        const std::vector<ggml_tensor *> & bank,
        ggml_tensor * norm_w,
        ggml_tensor * proj_w,
        int il) {
    const int64_t n_embd = prefix->ne[0];
    const int64_t n_toks = prefix->ne[1];
    const int64_t n_rows = (int64_t) bank.size() + 1;

    // v: [n_embd, n_rows, n_toks]
    ggml_tensor * v = nullptr;
    for (ggml_tensor * b : bank) {
        ggml_tensor * r = ggml_reshape_3d(ctx0, b, n_embd, 1, n_toks);
        v = v ? ggml_concat(ctx0, v, r, 1) : r;
    }
    {
        ggml_tensor * r = ggml_reshape_3d(ctx0, prefix, n_embd, 1, n_toks);
        v = v ? ggml_concat(ctx0, v, r, 1) : r;
    }

    ggml_tensor * k = ggml_rms_norm(ctx0, v, hparams.f_norm_rms_eps);
    cb(k, "attn_res_k", il);

    // score weight: norm.weight * proj.weight, [n_embd]
    ggml_tensor * sw = ggml_mul(ctx0, norm_w, proj_w);

    // scores: [1, n_rows, n_toks]
    ggml_tensor * scores = ggml_mul_mat(ctx0, ggml_reshape_2d(ctx0, sw, n_embd, 1), k);
    scores = ggml_reshape_2d(ctx0, scores, n_rows, n_toks);
    cb(scores, "attn_res_scores", il);

    ggml_tensor * probs = ggml_soft_max(ctx0, scores);
    cb(probs, "attn_res_probs", il);

    // weighted sum of raw rows
    ggml_tensor * w = ggml_mul(ctx0, v, ggml_reshape_3d(ctx0, probs, 1, n_rows, n_toks));
    w = ggml_cont(ctx0, ggml_permute(ctx0, w, 1, 0, 2, 3)); // [n_rows, n_embd, n_toks]
    ggml_tensor * out = ggml_sum_rows(ctx0, w);             // [1, n_embd, n_toks]
    out = ggml_reshape_2d(ctx0, out, n_embd, n_toks);
    cb(out, "attn_res_mix", il);

    return out;
}

llama_model_kimi_k3::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_build_delta_net_base(params), model(model) {
    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);
    cb(inpL, "model.embed_tokens", -1);

    // K3 uses no positional embeddings anywhere (KDA recurrence + NoPE MLA)

    auto * inp_kv = !hparams.is_mla() ? build_inp_mem_hybrid() : nullptr;
    auto * inp_k = hparams.is_mla() ? build_inp_mem_hybrid_k() : nullptr;
    auto * inp_rs = hparams.is_mla() ? inp_k->get_recr() : inp_kv->get_recr();
    auto * inp_attn_kv = !hparams.is_mla() ? inp_kv->get_attn() : nullptr;
    auto * inp_attn_k = hparams.is_mla() ? inp_k->get_attn() : nullptr;

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    const int64_t n_head = hparams.n_head();
    const int64_t head_dim = hparams.n_embd_head_kda;
    const int64_t d_conv = hparams.ssm_d_conv;
    const int64_t d_inner = n_head * head_dim;
    const int64_t n_seqs = ubatch.n_seqs;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;

    GGML_ASSERT(n_seqs != 0);
    GGML_ASSERT(ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == n_seq_tokens * n_seqs);

    // MLA params
    const int64_t n_embd_head_k_mla = hparams.n_embd_head_k_mla();
    const int64_t n_embd_head_v_mla = hparams.n_embd_head_v_mla();
    const int64_t kv_lora_rank = hparams.n_lora_kv;
    const int64_t n_embd_head_qk_rope = hparams.n_rot();
    const int64_t n_embd_head_qk_nope = n_embd_head_k_mla - n_embd_head_qk_rope;
    const float kq_scale_mla = 1.0f / sqrtf((float)n_embd_head_k_mla);

    const uint32_t res_block = hparams.attn_res_block_size;
    GGML_ASSERT(res_block > 0);

    // AttnRes state: snapshot bank + current prefix sum of the residual stream
    std::vector<ggml_tensor *> res_bank;
    ggml_tensor * prefix = inpL;

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];

        // pre-attention mixture (bank is empty only before the first snapshot)
        ggml_tensor * h = res_bank.empty()
            ? prefix
            : build_attn_res_mix(prefix, res_bank, layer.attn_res_norm, layer.attn_res_proj, il);

        // snapshot + restart of the residual stream
        const bool snapshot = (il % (int) res_block) == 0;
        if (snapshot) {
            res_bank.push_back(prefix);
        }

        cur = build_norm(h, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        ggml_build_forward_expand(gf, cur);

        if (hparams.is_recr(il)) {
            // === KDA layer (Kimi Delta Attention) ===
            const auto * mctx_cur = inp_rs->mctx;
            const auto kv_head = mctx_cur->get_head();

            ggml_tensor * conv_states_all = mctx_cur->get_r_l(il);
            cb(conv_states_all, "conv_states_all", il);
            ggml_tensor * conv_state_all = build_rs(inp_rs, conv_states_all, hparams.n_embd_r(), n_seqs);
            ggml_tensor * Qcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 0, cur, layer.wq, layer.ssm_q_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Kcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 1, cur, layer.wk, layer.ssm_k_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);
            ggml_tensor * Vcur = causal_conv1d(gf, ctx0, conv_states_all, conv_state_all, 2, cur, layer.wv, layer.ssm_v_conv, d_conv, head_dim, n_head, n_seq_tokens, n_seqs, n_tokens, kv_head);

            // K3 safe gate: g1 = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
            // ssm_a stores exp(A_log) (positive), per head
            ggml_tensor * f_a = ggml_mul_mat(ctx0, layer.ssm_f_a, cur);
            ggml_tensor * g1 = ggml_mul_mat(ctx0, layer.ssm_f_b, f_a);
            cb(g1, "kda_g1_raw", il);
            g1 = ggml_add(ctx0, g1, layer.ssm_dt_b);
            g1 = ggml_reshape_3d(ctx0, g1, head_dim, n_head, n_tokens);

            ggml_tensor * A = ggml_reshape_3d(ctx0, layer.ssm_a, 1, n_head, 1);
            g1 = ggml_mul(ctx0, g1, A);
            g1 = ggml_sigmoid(ctx0, g1);
            g1 = ggml_scale(ctx0, g1, hparams.kda_gate_lower_bound);
            cb(g1, "kda_g1", il);

            g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs);

            // beta (mixing coefficient)
            ggml_tensor * beta = ggml_mul_mat(ctx0, layer.ssm_beta, cur);
            beta = ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs);
            beta = ggml_sigmoid(ctx0, beta);
            cb(beta, "kda_beta", il);

            // KDA recurrence
            ggml_tensor * ssm_states_all = mctx_cur->get_s_l(il);
            ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
            state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);

            const float eps_norm = hparams.f_norm_rms_eps;

            Qcur = ggml_l2_norm(ctx0, Qcur, eps_norm);
            Kcur = ggml_l2_norm(ctx0, Kcur, eps_norm);

            auto attn_out = build_delta_net(Qcur, Kcur, Vcur, g1, beta, state, il);

            ggml_tensor * output = ggml_cont(ctx0, attn_out.first);
            ggml_tensor * new_state = attn_out.second;
            cb(output, "attn_output", il);

            ggml_build_forward_expand(gf,
                                     ggml_cpy(ctx0, new_state,
                                              ggml_view_1d(ctx0, ssm_states_all, hparams.n_embd_s() * n_seqs,
                                                           kv_head * hparams.n_embd_s() * ggml_element_size(ssm_states_all))));

            // full-rank output gate g2 = g_proj(x)
            ggml_tensor * g2 = ggml_mul_mat(ctx0, layer.wqkv_gate, cur);
            cb(g2, "kda_g2", il);
            g2 = ggml_reshape_3d(ctx0, g2, head_dim, n_head, n_tokens);

            // o_norm with sigmoid gating: out = RMSNorm(o) * sigmoid(g2)
            ggml_tensor * attn_out_final = ggml_reshape_3d(ctx0, output, head_dim, n_head, n_tokens);
            ggml_tensor * normed = build_norm(attn_out_final, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il);
            cb(normed, "kda_normed", il);
            ggml_tensor * gated = ggml_mul(ctx0, normed, ggml_sigmoid(ctx0, g2));

            gated = ggml_cont_2d(ctx0, gated, d_inner, n_tokens);
            cur = ggml_mul_mat(ctx0, layer.wo, gated);
            cb(cur, "kda_out", il);
        } else {
            // === Gated MLA layer (NoPE) ===
            // Q: q_b(q_a_norm(q_a(x)))
            ggml_tensor * Qcur = ggml_mul_mat(ctx0, layer.wq_a, cur);
            Qcur = build_norm(Qcur, layer.attn_q_a_norm, nullptr, LLM_NORM_RMS, il);
            Qcur = ggml_mul_mat(ctx0, layer.wq_b, Qcur);
            cb(Qcur, "mla_q", il);

            // KV compression
            ggml_tensor * kv_cmpr_pe = ggml_mul_mat(ctx0, layer.wkv_a_mqa, cur);

            ggml_tensor * kv_cmpr = ggml_view_2d(ctx0, kv_cmpr_pe, kv_lora_rank, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope), 0);
            ggml_tensor * k_pe = ggml_view_3d(ctx0, kv_cmpr_pe, n_embd_head_qk_rope, 1, n_tokens,
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank + n_embd_head_qk_rope),
                ggml_row_size(kv_cmpr_pe->type, kv_lora_rank));
            // NoPE: k_pe is a positional-encoding-free shared key dimension, no RoPE applied
            kv_cmpr = build_norm(kv_cmpr, layer.attn_kv_a_norm, nullptr, LLM_NORM_RMS, il);

            ggml_tensor * attn_pregate = nullptr;

            if (layer.wk_b && layer.wv_b) { // MLA KV cache enabled (absorption)
                ggml_tensor * q_nope =
                    ggml_view_3d(ctx0, Qcur, n_embd_head_qk_nope, n_head, n_tokens, ggml_row_size(Qcur->type, n_embd_head_k_mla),
                                 ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head, 0);
                ggml_tensor * q_pe = ggml_view_3d(
                    ctx0, Qcur, n_embd_head_qk_rope, n_head, n_tokens, ggml_row_size(Qcur->type, n_embd_head_k_mla),
                    ggml_row_size(Qcur->type, n_embd_head_k_mla) * n_head, ggml_row_size(Qcur->type, n_embd_head_qk_nope));

                q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
                ggml_tensor * q_nope_absorbed = ggml_mul_mat(ctx0, layer.wk_b, q_nope);
                q_nope_absorbed = ggml_permute(ctx0, q_nope_absorbed, 0, 2, 1, 3);

                ggml_tensor * Qmla = ggml_concat(ctx0, q_nope_absorbed, q_pe, 0);
                cb(Qmla, "mla_q_absorbed", il);

                kv_cmpr = ggml_reshape_3d(ctx0, kv_cmpr, kv_lora_rank, 1, n_tokens);
                ggml_tensor * Kcur = ggml_concat(ctx0, kv_cmpr, k_pe, 0);
                ggml_tensor * Vcur = kv_cmpr;

                attn_pregate = build_attn(inp_attn_k, nullptr, nullptr, nullptr, Qmla, Kcur, Vcur, nullptr, nullptr, layer.wv_b, kq_scale_mla, il);
            } else { // MLA KV cache disabled: fall back to MHA
                ggml_tensor * Qmla = ggml_reshape_3d(ctx0, Qcur, n_embd_head_k_mla, n_head, n_tokens);
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

                ggml_tensor * k_pe_target = ggml_new_tensor_3d(ctx0, k_pe->type, n_embd_head_qk_rope, n_head, n_tokens);
                ggml_tensor * k_pe_repeated = ggml_repeat(ctx0, k_pe, k_pe_target);
                ggml_tensor * Kcur = ggml_concat(ctx0, k_pe_repeated, k_nope, 0);

                attn_pregate = build_attn(inp_attn_kv, nullptr, nullptr, nullptr, Qmla, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale_mla, il);
            }
            cb(attn_pregate, "mla_pregate", il);

            // output gate: attn = attn * sigmoid(g_proj(x)), then o_proj
            ggml_tensor * gate = ggml_mul_mat(ctx0, layer.wqkv_gate, cur);
            gate = ggml_sigmoid(ctx0, gate);
            cb(gate, "mla_gate", il);

            cur = ggml_mul(ctx0, attn_pregate, gate);
            cur = build_lora_mm(layer.wo, cur, layer.wo_s);
            cb(cur, "mla_out", il);
        }

        // residual stream update (restarts at snapshot layers)
        prefix = snapshot ? cur : ggml_add(ctx0, prefix, cur);
        cb(prefix, "attn_prefix", il);

        // pre-FFN mixture (unconditional: bank is never empty here)
        ggml_tensor * h2 = build_attn_res_mix(prefix, res_bank, layer.ffn_res_norm, layer.ffn_res_proj, il);

        cur = build_norm(h2, layer.ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // Dense FFN (SiTU-GLU)
            cur = build_ffn(cur,
                layer.ffn_up, NULL, NULL,
                layer.ffn_gate, NULL, NULL,
                layer.ffn_down, NULL, NULL,
                NULL, LLM_FFN_SITU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // Stable LatentMoE
            // router operates on the full hidden state
            ggml_tensor * router_logits = build_lora_mm(layer.ffn_gate_inp, cur);
            cb(router_logits, "ffn_moe_logits", il);

            // experts run in the latent space
            ggml_tensor * latent = ggml_mul_mat(ctx0, layer.ffn_latent_down, cur);
            cb(latent, "ffn_moe_latent", il);

            ggml_tensor * moe_out = build_moe_ffn(latent,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                hparams.n_expert,
                hparams.n_expert_used,
                LLM_FFN_SITU,
                hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                router_logits);
            cb(moe_out, "ffn_moe_out", il);

            // latent norm applies AFTER the weighted expert sum, then project back up
            moe_out = build_norm(moe_out, layer.ffn_latent_norm, NULL, LLM_NORM_RMS, il);
            moe_out = ggml_mul_mat(ctx0, layer.ffn_latent_up, moe_out);
            cb(moe_out, "ffn_moe_out_up", il);

            // shared experts on the full hidden state (SiTU-GLU)
            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp, NULL, NULL,
                    layer.ffn_gate_shexp, NULL, NULL,
                    layer.ffn_down_shexp, NULL, NULL,
                    NULL, LLM_FFN_SITU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        // residual stream update
        prefix = ggml_add(ctx0, prefix, cur);
        prefix = build_cvec(prefix, il);
        cb(prefix, "l_out", il);
    }

    // output mixture over the final prefix sum and the snapshot bank
    cur = build_attn_res_mix(prefix, res_bank, model.output_res_norm, model.output_res_proj, -1);

    // select only the output tokens (AttnRes needs the full token set until here)
    if (inp_out_ids) {
        cur = ggml_get_rows(ctx0, cur, inp_out_ids);
    }

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = ggml_mul_mat(ctx0, model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
