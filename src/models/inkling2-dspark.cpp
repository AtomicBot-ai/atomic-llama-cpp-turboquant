// Inkling2 DSpark graph from llama.cpp PR #32218, revision 8ea4363119c0d8d205e0cce264d004c017cbd402.
// Keep the target visit ordering, shared output weights and full noncausal draft block in sync with upstream.
#include "models.h"

#include "llama-impl.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

void llama_model_dflash::load_hparams_inkling2(llama_model_loader & ml) {
    ml.get_key(LLM_KV_VOCAB_SIZE, hparams.n_vocab_padded);
    ml.get_key(LLM_KV_VOCAB_SIZE_UNPADDED, hparams.n_vocab_unpadded);
    ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, hparams.n_swa);
    ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN, hparams.is_swa_impl, hparams.n_layer());

    if (hparams.dflash_block_size != 8 || !hparams.n_swa || hparams.n_swa >= INT32_MAX) {
        throw std::runtime_error("invalid Inkling2 DSpark block size or sliding window");
    }
    if (target_layer_ids.empty() || target_layer_ids.front() < 0 || target_layer_ids.back() > LLAMA_MAX_LAYERS ||
            std::adjacent_find(target_layer_ids.begin(), target_layer_ids.end(), std::greater_equal<int32_t>()) != target_layer_ids.end()) {
        throw std::runtime_error("Inkling2 DSpark target layers must be in range and strictly increasing");
    }
    if (target_layer_ids.size() * (uint64_t) hparams.n_embd > UINT32_MAX) {
        throw std::runtime_error("Inkling2 DSpark target features exceed the input width");
    }
    if (!std::isfinite(hparams.f_norm_rms_eps) || hparams.f_norm_rms_eps <= 0 ||
            !std::isfinite(hparams.f_final_logit_softcapping) || hparams.f_final_logit_softcapping <= 0) {
        throw std::runtime_error("invalid Inkling2 DSpark RMS epsilon or logit softcap");
    }
    for (uint32_t il = 0; il < hparams.n_layer(); ++il) {
        if (!hparams.n_head_kv(il) || hparams.n_head(il) % hparams.n_head_kv(il) ||
                hparams.n_embd_head_k(il) != hparams.n_embd_head_v(il) || !hparams.n_ff(il) ||
                hparams.n_rot(il) != hparams.n_embd_head_k(il) || hparams.n_rot(il) % 2) {
            throw std::runtime_error("invalid Inkling2 DSpark attention dimensions");
        }
    }
    // The anchor row includes the anchor and the window before it.
    hparams.n_swa += 1;
    hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;
    hparams.rope_freq_base_train_swa = hparams.rope_freq_base_train;
    hparams.rope_freq_scale_train_swa = hparams.rope_freq_scale_train;
}

void llama_model_dflash::load_tensors_inkling2() {
    LLAMA_LOAD_LOCALS;

    const int64_t rank = ml->require_tensor_meta("markov_w1.weight")->ne[0];
    if (rank <= 0 || rank > n_embd) {
        throw std::runtime_error("invalid Inkling2 DSpark recurrent rank");
    }
    // The greedy tie-break stores token ranks in F32.
    if (n_vocab <= 0 || n_vocab != hparams.n_vocab_unpadded || n_vocab > hparams.n_vocab_padded || n_vocab > (1 << 24)) {
        throw std::runtime_error("invalid Inkling2 DSpark vocabulary size");
    }
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    dspark_mask_embd = create_tensor(tn(LLM_TENSOR_DSPARK_MASK_EMBD, "weight"), {n_embd}, 0);
    dspark_fuse = create_tensor(tn(LLM_TENSOR_DSPARK_FUSE, "weight"), {(int64_t) target_layer_ids.size(), n_layer + 1}, 0);
    dspark_norm = create_tensor(tn(LLM_TENSOR_DSPARK_NORM, "weight"), {n_embd}, 0);
    dspark_markov_w1 = create_tensor(tn(LLM_TENSOR_DSPARK_MARKOV_W1, "weight"), {rank, hparams.n_vocab_padded}, 0);
    dspark_joint_proj = create_tensor(tn(LLM_TENSOR_DSPARK_JOINT_PROJ, "weight"), {2*rank + 2*n_embd, 3*rank}, 0);
    dspark_joint_proj_b = create_tensor(tn(LLM_TENSOR_DSPARK_JOINT_PROJ, "bias"), {3*rank}, 0);
    dspark_hidden_out = create_tensor(tn(LLM_TENSOR_DSPARK_HIDDEN_OUT, "weight"), {rank, n_embd}, 0);
    dspark_hidden_out_b = create_tensor(tn(LLM_TENSOR_DSPARK_HIDDEN_OUT, "bias"), {n_embd}, 0);
    const auto folded_name = tn(LLM_TENSOR_DSPARK_FOLDED_OUTPUT, "weight");
    if (const auto * folded = ml->get_tensor_meta(folded_name.str().c_str())) {
        if (folded->type != GGML_TYPE_F16) {
            throw std::runtime_error("Inkling2 DSpark folded output must use F16");
        }
        dspark_folded_output = create_tensor(folded_name, {GGML_PAD(rank + 1, 32), hparams.n_vocab_padded}, 0);
    }
    for (int i = 0; i < n_layer; ++i) {
        auto & l = layers[i];
        const int64_t hd = hparams.n_embd_head_k(i);
        const int64_t q = hd*hparams.n_head(i);
        const int64_t kv = hd*hparams.n_head_kv(i);
        l.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        l.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        l.dflare_norm = create_tensor(tn(LLM_TENSOR_DFLARE_NORM, "weight", i), {n_embd}, 0);
        l.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, q}, 0);
        l.wk = create_tensor(tn(LLM_TENSOR_ATTN_K, "weight", i), {n_embd, kv}, 0);
        l.wv = create_tensor(tn(LLM_TENSOR_ATTN_V, "weight", i), {n_embd, kv}, 0);
        l.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {q, n_embd}, 0);
        l.attn_k_ctx = create_tensor(tn(LLM_TENSOR_ATTN_K_CTX, "weight", i), {n_embd, kv}, 0);
        l.attn_v_ctx = create_tensor(tn(LLM_TENSOR_ATTN_V_CTX, "weight", i), {n_embd, kv}, 0);
        l.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {hd}, 0);
        l.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {hd}, 0);
        l.attn_k_ctx_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_CTX_NORM, "weight", i), {hd}, 0);
        l.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, hparams.n_ff(i)}, 0);
        l.ffn_up = create_tensor(tn(LLM_TENSOR_FFN_UP, "weight", i), {n_embd, hparams.n_ff(i)}, 0);
        l.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {hparams.n_ff(i), n_embd}, 0);
    }
}

static ggml_tensor * build_inkling2_head(llm_graph_context & g, const llama_model & model, const llama_model & target,
        ggml_tensor * prev, ggml_tensor * & state, ggml_tensor * hidden, ggml_tensor * base, ggml_tensor * context) {
    auto * ctx0 = g.ctx0;
    const int64_t rank = model.dspark_markov_w1->ne[0];
    auto * joint = ggml_concat(ctx0, state, ggml_get_rows(ctx0, model.dspark_markov_w1, prev), 0);
    joint = ggml_concat(ctx0, joint, ggml_concat(ctx0, hidden, context, 0), 0);
    joint = ggml_add(ctx0, g.build_lora_mm(model.dspark_joint_proj, joint), model.dspark_joint_proj_b);
    const auto chunk = [&](int64_t j) {
        return ggml_cont(ctx0, ggml_view_2d(ctx0, joint, rank, joint->ne[1], joint->nb[1], j*rank*joint->nb[0]));
    };
    auto * gate = ggml_sigmoid(ctx0, chunk(0));
    auto * candidate = ggml_tanh(ctx0, chunk(1));
    state = ggml_add(ctx0, candidate, ggml_mul(ctx0, gate, ggml_sub(ctx0, state, candidate)));
    auto * latent = ggml_tanh(ctx0, chunk(2));
    ggml_tensor * bias;
    if (model.dspark_folded_output && g.loras->empty() && !target.output_s) {
        // This optional sidecar tensor folds the target output projection
        // through the rank-sized DSpark correction, including its bias.
        auto * one = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, latent->ne[1]), 1.0f);
        auto * augmented = ggml_concat(ctx0, latent, one, 0);
        augmented = ggml_pad(ctx0, augmented, model.dspark_folded_output->ne[0] - augmented->ne[0], 0, 0, 0);
        bias = ggml_mul_mat(ctx0, model.dspark_folded_output, augmented);
    } else {
        auto * correction = ggml_add(ctx0, g.build_lora_mm(model.dspark_hidden_out, latent), model.dspark_hidden_out_b);
        bias = g.build_lora_mm(target.output, correction, target.output_s);
    }
    ggml_prec_set_acc(bias, GGML_PREC_F32);
    if (bias->ne[0] != base->ne[0]) {
        bias = ggml_cont(ctx0, ggml_view_2d(ctx0, bias, base->ne[0], bias->ne[1], bias->nb[1], 0));
    }
    auto * logits = ggml_add(ctx0, base, bias);
    const float cap = model.hparams.f_final_logit_softcapping;
    return ggml_scale(ctx0, ggml_tanh(ctx0, ggml_scale(ctx0, logits, 1.0f/cap)), cap);
}

namespace {
// A prefix request can stop the recurrent head early. Another mask of the same size
// can need later recurrent steps, so it must not reuse the prefix graph.
class inkling_draft_input final : public llm_graph_input_embd_h {
public:
    inkling_draft_input(int64_t width, const llm_graph_params & params)
        : llm_graph_input_embd_h(width, width), prefix_rows(output_prefix(params)) {}

    bool can_reuse(const llm_graph_params & params) override {
        return prefix_rows == output_prefix(params) && llm_graph_input_embd_h::can_reuse(params);
    }

    const uint32_t prefix_rows;

private:
    static uint32_t output_prefix(const llm_graph_params & params) {
        const auto & batch = params.ubatch;
        if (batch.n_seqs_unq == 1 && params.n_outputs < batch.n_tokens && batch.output &&
                std::all_of(batch.output, batch.output + params.n_outputs, [](int8_t flag) { return flag != 0; })) {
            return params.n_outputs;
        }
        return 0;
    }
};
} // namespace

// dspark_fuse has one row of target-layer mixing weights per draft layer and a final row for the recurrent head.
llama_model_dflash::graph_inkling2::graph_inkling2(const llm_graph_params & params) : llm_graph_context(params) {
    auto * positions = build_inp_pos();
    auto * attn = build_attn_inp_kv_iswa();
    const int64_t rank = model.dspark_markov_w1->ne[0];
    const int64_t ntarget = model.target_layer_ids.size();
    const auto mix = [&](ggml_tensor * features, int row, ggml_tensor * norm) {
        auto * weights = ggml_soft_max(ctx0, ggml_view_1d(ctx0, model.dspark_fuse, ntarget, row*model.dspark_fuse->nb[1]));
        auto * x = ggml_reshape_3d(ctx0, features, n_embd, ntarget, n_tokens);
        x = ggml_cont(ctx0, ggml_permute(ctx0, x, 1, 0, 2, 3));
        x = ggml_sum_rows(ctx0, ggml_mul(ctx0, x, weights));
        return build_norm(ggml_reshape_2d(ctx0, x, n_embd, n_tokens), norm, nullptr, LLM_NORM_RMS, row == n_layer ? -1 : row);
    };

    if (ubatch.embd || ubatch.embd_row) {
        auto inp = std::make_unique<llm_graph_input_embd>(hparams.n_embd_inp_enc());
        auto * features = build_inp_target_features(*this, *inp, hparams.n_embd_inp_enc());
        res->add_input(std::move(inp));
        for (int il = 0; il < n_layer; ++il) {
            const auto & l = model.layers[il];
            const int64_t hd = hparams.n_embd_head_k(il);
            const int64_t nk = hparams.n_head_kv(il);
            auto * context = mix(features, il, l.dflare_norm);
            cb(context, "dspark_context", il);
            auto * k = ggml_reshape_3d(ctx0, build_lora_mm(l.attn_k_ctx, context), hd, nk, n_tokens);
            auto * v = ggml_reshape_3d(ctx0, build_lora_mm(l.attn_v_ctx, context), hd, nk, n_tokens);
            k = build_norm(k, l.attn_k_ctx_norm, nullptr, LLM_NORM_RMS, il);
            k = build_rope(k, positions, il);
            const bool swa = hparams.is_swa(il);
            const auto * kv = swa ? attn->mctx->get_swa() : attn->mctx->get_base();
            auto * k_rot = swa ? attn->self_k_rot_swa[il] : attn->self_k_rot[il];
            auto * v_rot = swa ? attn->self_v_rot_swa[il] : attn->self_v_rot[il];
            if (k_rot) {
                k = llama_mul_mat_hadamard(ctx0, k, k_rot);
            }
            if (v_rot) {
                v = llama_mul_mat_hadamard(ctx0, v, v_rot);
            }
            ggml_build_forward_expand(gf, kv->cpy_k(ctx0, k, swa ? attn->get_k_idxs_swa() : attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, kv->cpy_v(ctx0, v, swa ? attn->get_v_idxs_swa() : attn->get_v_idxs(), il));
        }
        res->t_embd = res->t_h_nextn = mix(features, n_layer, model.dspark_norm);
        cb(res->t_h_nextn, "dspark_f_rnn", -1);
        ggml_build_forward_expand(gf, res->t_h_nextn);
        return;
    }

    GGML_ASSERT(cparams.ctx_other);
    const auto * target = llama_get_model(cparams.ctx_other);
    GGML_ASSERT(target->tok_embd && target->tok_norm && target->output);
    auto inp = std::make_unique<inkling_draft_input>(n_embd, params);
    const auto prefix_rows = inp->prefix_rows;
    auto * tokens = inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    auto * f_rnn = inp->h = build_inp_embd_state(inp.get(), n_embd);
    res->add_input(std::move(inp));

    const int64_t nb = ubatch.n_seqs_unq;
    const int64_t block = n_tokens/nb;
    auto * ids = ggml_cont_1d(ctx0, ggml_view_2d(ctx0, tokens, 1, nb, block*tokens->nb[0], 0), nb);
    auto * anchors = ggml_get_rows(ctx0, target->tok_embd, ids);
    anchors = ggml_mul(ctx0, ggml_rms_norm(ctx0, anchors, target->hparams.f_norm_rms_eps), target->tok_norm);
    auto * cur = ggml_reshape_3d(ctx0, anchors, n_embd, 1, nb);
    if (block > 1) {
        auto * masks = ggml_repeat_4d(ctx0, ggml_cast(ctx0, model.dspark_mask_embd, GGML_TYPE_F32), n_embd, block - 1, nb, 1);
        cur = ggml_concat(ctx0, cur, masks, 1);
    }
    cur = ggml_reshape_2d(ctx0, cur, n_embd, n_tokens);
    attn->self_attn_mask_swa.block_uniform = true;
    for (int il = 0; il < n_layer; ++il) {
        const auto & l = model.layers[il];
        const int64_t hd = hparams.n_embd_head_k(il);
        auto * residual = cur;
        cur = build_norm(cur, l.attn_norm, nullptr, LLM_NORM_RMS, il);
        auto * q = ggml_reshape_3d(ctx0, build_lora_mm(l.wq, cur), hd, hparams.n_head(il), n_tokens);
        auto * k = ggml_reshape_3d(ctx0, build_lora_mm(l.wk, cur), hd, hparams.n_head_kv(il), n_tokens);
        auto * v = ggml_reshape_3d(ctx0, build_lora_mm(l.wv, cur), hd, hparams.n_head_kv(il), n_tokens);
        q = build_rope(build_norm(q, l.attn_q_norm, nullptr, LLM_NORM_RMS, il), positions, il);
        k = build_rope(build_norm(k, l.attn_k_norm, nullptr, LLM_NORM_RMS, il), positions, il);
        cur = build_attn(attn, l.wo, nullptr, nullptr, q, k, v, nullptr, nullptr, nullptr, 1.0f/sqrtf(hd), il);
        residual = ggml_add(ctx0, residual, cur);
        cur = build_norm(residual, l.ffn_norm, nullptr, LLM_NORM_RMS, il);
        cur = build_ffn(cur, l.ffn_up, nullptr, nullptr, l.ffn_gate, nullptr, nullptr,
                l.ffn_down, nullptr, nullptr, nullptr, LLM_FFN_SILU, LLM_FFN_PAR, il);
        cur = ggml_add(ctx0, residual, cur);
        cb(cur, "dspark_layer", il);
    }
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    res->t_embd = cur;
    auto * base = build_lora_mm(target->output, cur, target->output_s);
    ggml_prec_set_acc(base, GGML_PREC_F32);
    if (block > hparams.dflash_block_size) {
        res->t_logits = ggml_cont(ctx0, ggml_view_2d(ctx0, base, hparams.n_vocab_unpadded, n_tokens, base->nb[1], 0));
        return;
    }
    auto * state = ggml_fill(ctx0, ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, rank, nb), 0.0f);
    auto * logits = ggml_cont(ctx0, ggml_view_2d(ctx0, base, base->ne[0], nb, block*base->nb[1], 0));
    const auto column = [&](ggml_tensor * t, int64_t i) {
        return ggml_cont(ctx0, ggml_view_2d(ctx0, t, t->ne[0], nb, block*t->nb[1], i*t->nb[1]));
    };
    auto * context = column(f_rnn, 0);
    auto * tie_rank = ggml_arange(ctx0, 0, hparams.n_vocab_unpadded, 1);
    tie_rank = ggml_scale_bias(ctx0, tie_rank, -1.0f, hparams.n_vocab_unpadded);
    auto * prev = ids;
    const int64_t head_rows = prefix_rows ? prefix_rows : block;
    for (int64_t i = 1; i < head_rows; ++i) {
        auto * col = build_inkling2_head(*this, model, *target, prev, state, column(cur, i), column(base, i), context);
        logits = ggml_concat(ctx0, logits, col, 1);
        // Select the first maximum, independent of backend argmax tie order.
        auto * valid = ggml_cont(ctx0, ggml_view_2d(ctx0, col, hparams.n_vocab_unpadded, nb, col->nb[1], 0));
        auto * maximum = ggml_pool_1d(ctx0, valid, GGML_OP_POOL_MAX, valid->ne[0], 1, 0);
        auto * is_max = ggml_scale_bias(ctx0, ggml_step(ctx0, ggml_sub(ctx0, ggml_repeat(ctx0, maximum, valid), valid)), -1.0f, 1.0f);
        prev = ggml_argmax(ctx0, ggml_mul(ctx0, is_max, tie_rank));
    }
    logits = ggml_reshape_3d(ctx0, logits, logits->ne[0], nb, head_rows);
    logits = ggml_reshape_2d(ctx0, ggml_cont(ctx0, ggml_permute(ctx0, logits, 0, 2, 1, 3)), logits->ne[0], nb*head_rows);
    res->t_logits = ggml_cont(ctx0, ggml_view_2d(ctx0, logits, hparams.n_vocab_unpadded, nb*head_rows, logits->nb[1], 0));
}
