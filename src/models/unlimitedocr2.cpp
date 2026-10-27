#include "models.h"
#include "llama-kv-cache-msa.h"

#include <algorithm>
#include <cmath>
#include <vector>

// Unlimited-OCR-2: the DeepSeek-OCR decoder (MHA + DeepSeek-V2 MoE) with
//  - page-2D RoPE: positions are (offset inside the page bucket, page parity) instead of the sequence index
//  - DASH attention: a token-level local window plus up to top_k remote image blocks of the query's page,
//    picked per layer by a router (head_cls block summaries, entmax-1.5 prior -> additive log bias)
// The routed attention is applied as a mask over the full KV cache: same result as the sparse kernel, without the speedup.
// Positions and pages are derived from the token ids stored in the KV cells, so only one sequence is supported.

void llama_model_unlimitedocr2::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,  hparams.n_ff_exp);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func, false);

    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
    }

    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,   hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,   hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,        hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,   hparams.indexer_block_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS, hparams.indexer_local_blocks);

    ml.get_key(LLM_KV_DASH_TOP_M,          hparams.dash_top_m);
    ml.get_key(LLM_KV_DASH_SCALE,          hparams.dash_scale);
    ml.get_key(LLM_KV_DASH_PRIOR_FLOOR,    hparams.dash_prior_floor);
    ml.get_key(LLM_KV_DASH_SIGMA,          hparams.dash_sigma);
    ml.get_key(LLM_KV_PAGE_TOKEN_ID,       hparams.page_token_id);
    ml.get_key(LLM_KV_ROPE_PAGE_FREQ_BASE, hparams.rope_page_freq_base);
    ml.get_key(LLM_KV_ROPE_PAGE_PAIRS,     hparams.rope_page_pairs);
    ml.get_key(LLM_KV_ROPE_PAGE_MOD,       hparams.rope_page_mod);

    GGML_ASSERT(hparams.indexer_n_head == 1);
    GGML_ASSERT(hparams.indexer_head_size == hparams.n_embd_head_k() && "router keys share the main rope");
    GGML_ASSERT(hparams.indexer_block_size > 0 && hparams.indexer_top_k > 0);
    GGML_ASSERT(hparams.dash_top_m >= hparams.indexer_top_k);
    GGML_ASSERT(hparams.rope_page_pairs > 0 && hparams.rope_page_pairs < hparams.n_embd_head_k()/2);

    switch (hparams.n_layer()) {
        case 12: type = LLM_TYPE_3B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_unlimitedocr2::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_ff_exp        = hparams.n_ff_exp;
    const int64_t n_idx           = hparams.indexer_head_size;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    if (n_expert == 0 || n_expert_used == 0) {
        throw std::runtime_error("n_expert and n_expert_used must be > 0");
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        create_tensor_qkv(layer, i, n_embd, n_embd, n_embd, n_embd, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd, n_embd}, 0);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, 0);

        // DASH router (gemma-style norm weights, +1 baked in by the converter)
        layer.index_q_proj   = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ,   "weight", i), {n_embd, n_idx}, 0);
        layer.index_k_proj   = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ,   "weight", i), {n_embd, n_idx}, 0);
        layer.index_q_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM,   "weight", i), {n_idx}, 0);
        layer.index_k_norm   = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM,   "weight", i), {n_idx}, 0);
        layer.index_head_cls = create_tensor(tn(LLM_TENSOR_INDEXER_HEAD_CLS,           i), {n_idx}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
        } else {
            layer.ffn_gate_inp = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP, "weight", i), {n_embd, n_expert}, 0);

            layer.ffn_down_exps = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS, "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            create_tensor_gate_up_exps(layer, i, n_embd, n_ff_exp, n_expert, 0);

            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_unlimitedocr2::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

// per-ubatch inputs of the page rope and the DASH candidate domain, all derived on the host
class llm_graph_input_uocr2 : public llm_graph_input_i {
public:
    llm_graph_input_uocr2(const llama_kv_cache_msa_context * mctx, const llama_hparams & hparams) :
        mctx(mctx), hparams(hparams) {}

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * rope_tok  = nullptr; // I32 [n_tokens]  offset inside the page bucket
    ggml_tensor * rope_page = nullptr; // I32 [n_tokens]  page bucket (mod rope_page_mod)
    ggml_tensor * ff_tok    = nullptr; // F32 [n_rot/2]   freq factors selecting the token-axis pairs
    ggml_tensor * ff_page   = nullptr; // F32 [n_rot/2]   freq factors selecting the page-axis pairs
    ggml_tensor * q_pos     = nullptr; // F32 [n_tokens]  sequence position of the queries
    ggml_tensor * pos_slot  = nullptr; // I32 [n_ps]      position -> cell
    ggml_tensor * pos_valid = nullptr; // F32 [n_ps]      0 / -1e30 for positions without a cell
    ggml_tensor * cand      = nullptr; // F32 [nblk, n_tokens] 1 = remote block the query may route to
    ggml_tensor * cell_blk  = nullptr; // I32 [n_kv]      cell -> position block
    ggml_tensor * cell_pos  = nullptr; // I32 [n_kv]      cell -> position

    const llama_kv_cache_msa_context * mctx;
    const llama_hparams & hparams;
};

void llm_graph_input_uocr2::set_input(const llama_ubatch * ubatch) {
    const int64_t n_tokens = ubatch->n_tokens;
    const int64_t n_ps     = pos_slot->ne[0];
    const int     blk      = (int) hparams.indexer_block_size;
    const int64_t nblk     = n_ps / blk;
    const int     n_half   = (int) ff_tok->ne[0];
    const int     n_page_p = (int) hparams.rope_page_pairs;
    const int     n_tok_p  = n_half - n_page_p;

    // freq factors: ggml uses theta_i = p * base^(-2i/n_rot) / ff_i
    {
        const float base = hparams.rope_freq_base_train;
        std::vector<float> ft(n_half), fp(n_half);
        for (int i = 0; i < n_half; ++i) {
            const float ggml_freq = powf(base, -2.0f*i/(2*n_half));
            const bool  is_tok    = i < n_tok_p;
            const float want = is_tok
                ? powf(base, -(float) i/n_tok_p)                                       // "resample" token axis
                : powf(hparams.rope_page_freq_base, -(float) (i - n_tok_p)/n_page_p); // page axis
            ft[i] = is_tok ? ggml_freq/want : 1e30f;
            fp[i] = is_tok ? 1e30f : ggml_freq/want;
        }
        ggml_backend_tensor_set(ff_tok,  ft.data(), 0, n_half*sizeof(float));
        ggml_backend_tensor_set(ff_page, fp.data(), 0, n_half*sizeof(float));
    }

    GGML_ASSERT(n_tokens > 0);
    const llama_seq_id seq_id = ubatch->seq_id[0][0];

    std::vector<llama_token> toks(n_ps);
    mctx->get_seq_tokens(seq_id, toks);

    // walk the sequence: the prompt ("<PAGE><image>" per page + instruction) ends at the first plain text
    // token after an image, which mirrors the HF prompt split at the last image token
    const llama_token page_id = (llama_token) hparams.page_token_id;

    std::vector<int32_t> tok_pos(n_ps, 0), page_pos(n_ps, 0), q_page(n_ps, -1), img_page(n_ps, -1);
    std::vector<int32_t> cnt;      // running count per page bucket
    std::vector<int32_t> pf_cnt;   // prompt count per page bucket, frozen at the response start
    int32_t cum = 0, n_pages = 0, cur_page = 0, cur_tok = 0;
    bool seen_img = false, in_resp = false;

    for (int64_t p = 0; p < n_ps; ++p) {
        const llama_token t = toks[p];
        if (t == -2) {
            continue;
        }
        const bool is_img = t == LLAMA_TOKEN_NULL;
        const bool is_mk  = t == page_id;

        if (is_mk) {
            cum++;
        }
        if (!in_resp && seen_img && !is_img && !is_mk) {
            in_resp  = true;
            n_pages  = cum;
            pf_cnt   = cnt;
            pf_cnt.resize(n_pages + 1, 0);
            cur_page = 0;
            cur_tok  = pf_cnt[0] - 1;
        }
        seen_img |= is_img;

        int32_t page;
        if (!in_resp) {
            page = cum;
            if ((int32_t) cnt.size() <= page) {
                cnt.resize(page + 1, 0);
            }
            tok_pos[p] = cnt[page]++;
            q_page[p]  = cum - 1;
        } else {
            // decode state machine of the reference: a marker opens the next page (clamped), else advance
            if (is_mk) {
                cur_page = std::min(cur_page + 1, n_pages);
                cur_tok  = pf_cnt[cur_page];
            } else {
                cur_tok++;
            }
            page       = cur_page;
            tok_pos[p] = cur_tok;
            q_page[p]  = cum - 1 - n_pages;
        }
        page_pos[p] = hparams.rope_page_mod > 0 ? page % (int32_t) hparams.rope_page_mod : page;
        img_page[p] = is_img ? cum - 1 : -1;
    }

    // rope + query positions of this ubatch
    {
        std::vector<int32_t> rt(n_tokens), rp(n_tokens);
        std::vector<float>   qp(n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const llama_pos p = ubatch->pos[i];
            GGML_ASSERT(p >= 0 && p < n_ps);
            rt[i] = tok_pos[p];
            rp[i] = page_pos[p];
            qp[i] = (float) p;
        }
        ggml_backend_tensor_set(rope_tok,  rt.data(), 0, n_tokens*sizeof(int32_t));
        ggml_backend_tensor_set(rope_page, rp.data(), 0, n_tokens*sizeof(int32_t));
        ggml_backend_tensor_set(q_pos,     qp.data(), 0, n_tokens*sizeof(float));
    }

    {
        std::vector<float> pv(n_ps);
        for (int64_t p = 0; p < n_ps; ++p) {
            pv[p] = toks[p] == -2 ? -1e30f : 0.0f;
        }
        ggml_backend_tensor_set(pos_valid, pv.data(), 0, n_ps*sizeof(float));
    }

    // candidate remote blocks: fully in the past beyond the local blocks, restricted to image blocks of the
    // query's page when the query has one and any such block is remote (the reference's per-query fallback)
    {
        std::vector<int32_t> bmin(nblk, -1), bmax(nblk, -1);
        std::vector<bool>    bvalid(nblk, false);
        for (int64_t p = 0; p < n_ps; ++p) {
            const int64_t b = p / blk;
            bvalid[b] = bvalid[b] || toks[p] != -2;
            if (img_page[p] >= 0) {
                bmin[b] = bmin[b] < 0 ? img_page[p] : std::min(bmin[b], img_page[p]);
                bmax[b] = std::max(bmax[b], img_page[p]);
            }
        }

        const int64_t n_local = hparams.indexer_local_blocks;
        std::vector<float> cd(nblk*n_tokens, 0.0f);
        for (int64_t i = 0; i < n_tokens; ++i) {
            const llama_pos p    = ubatch->pos[i];
            const int64_t   last = p/blk - n_local; // last remote block
            const int32_t   qp   = q_page[p];
            float * row = cd.data() + i*nblk;

            bool any_same = false;
            for (int64_t b = 0; b <= last && b < nblk; ++b) {
                const bool same = qp >= 0 && bmin[b] >= 0 && bmin[b] <= qp && qp <= bmax[b];
                any_same |= bvalid[b] && same;
            }
            for (int64_t b = 0; b <= last && b < nblk; ++b) {
                const bool same = qp >= 0 && bmin[b] >= 0 && bmin[b] <= qp && qp <= bmax[b];
                row[b] = bvalid[b] && (!any_same || same) ? 1.0f : 0.0f;
            }
        }
        ggml_backend_tensor_set(cand, cd.data(), 0, cd.size()*sizeof(float));
    }

    mctx->set_input_pos_slot(pos_slot, ubatch);
    mctx->set_input_cell_pos(cell_blk, ubatch, blk);
    mctx->set_input_cell_pos(cell_pos, ubatch, 1);
}

llama_model_unlimitedocr2::graph::graph(const llama_model & model, const llm_graph_params & params) :
    llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_k();
    const int64_t n_idx       = hparams.indexer_head_size;
    const int     blk         = (int) hparams.indexer_block_size;
    const float   kq_scale    = 1.0f/sqrtf(float(n_embd_head));
    const int     rope_type   = GGML_ROPE_TYPE_NEOX;
    const float   NEG         = -1e30f;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_v());
    if (cparams.n_seq_max > 1) {
        throw std::runtime_error("unlimited-ocr2: only a single sequence is supported (use -np 1)");
    }

    const auto * mctx_msa  = static_cast<const llama_kv_cache_msa_context *>(mctx);
    const auto * mctx_base = mctx_msa->get_base();
    const auto * mctx_idx  = mctx_msa->get_idx();

    ggml_tensor * inpL = build_inp_embd(model.tok_embd);

    auto * inp_attn = build_attn_inp_kv_msa(true);

    ggml_tensor * kq_mask = inp_attn->get_kq_mask();
    const int64_t n_kv = kq_mask->ne[0];
    GGML_ASSERT(kq_mask->ne[1] == n_tokens && kq_mask->ne[3] == 1);

    const int64_t n_ps = GGML_PAD((int64_t) mctx_msa->get_n_pos(), blk);
    const int64_t nblk = n_ps/blk;
    const int     M    = std::min<int>((int) hparams.dash_top_m,    (int) nblk);
    const int     K    = std::min<int>((int) hparams.indexer_top_k, M);

    auto * u = res->add_input(std::make_unique<llm_graph_input_uocr2>(mctx_msa, hparams));
    auto * inp = static_cast<llm_graph_input_uocr2 *>(u);
    {
        auto mk = [&](ggml_type t, int64_t ne0, int64_t ne1) {
            ggml_tensor * x = ne1 == 0 ? ggml_new_tensor_1d(ctx0, t, ne0) : ggml_new_tensor_2d(ctx0, t, ne0, ne1);
            ggml_set_input(x);
            return x;
        };
        inp->rope_tok  = mk(GGML_TYPE_I32, n_tokens, 0);
        inp->rope_page = mk(GGML_TYPE_I32, n_tokens, 0);
        inp->ff_tok    = mk(GGML_TYPE_F32, n_embd_head/2, 0);
        inp->ff_page   = mk(GGML_TYPE_F32, n_embd_head/2, 0);
        inp->q_pos     = mk(GGML_TYPE_F32, 1, n_tokens);
        inp->pos_slot  = mk(GGML_TYPE_I32, n_ps, 1);
        inp->pos_valid = mk(GGML_TYPE_F32, n_ps, 1);
        inp->cand      = mk(GGML_TYPE_F32, nblk, n_tokens);
        inp->cell_blk  = mk(GGML_TYPE_I32, n_kv, 1);
        inp->cell_pos  = mk(GGML_TYPE_I32, n_kv, 1);
    }

    // page-2D rope as two NEOX passes over disjoint frequency pairs
    auto rope2d = [&](ggml_tensor * x) {
        x = ggml_rope_ext(ctx0, x, inp->rope_tok,  inp->ff_tok,  n_embd_head, rope_type, 0,
                          hparams.rope_freq_base_train, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        x = ggml_rope_ext(ctx0, x, inp->rope_page, inp->ff_page, n_embd_head, rope_type, 0,
                          hparams.rope_freq_base_train, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        return x;
    };

    // graph-wide routing helpers
    // local window (q - W, q] at token level, shared by all layers
    ggml_tensor * notlocal;
    {
        const float W = (float) (hparams.indexer_local_blocks * blk);
        ggml_tensor * kp = ggml_cast(ctx0, inp->cell_pos, GGML_TYPE_F32);                // [n_kv, 1]
        ggml_tensor * d  = ggml_sub(ctx0, ggml_repeat_4d(ctx0, inp->q_pos, n_kv, n_tokens, 1, 1), kp);
        ggml_tensor * local = ggml_mul(ctx0,
                ggml_step(ctx0, ggml_scale_bias(ctx0, d,  1.0f, 0.5f)),
                ggml_step(ctx0, ggml_scale_bias(ctx0, d, -1.0f, W - 0.5f)));
        notlocal = ggml_scale_bias(ctx0, local, -1.0f, 1.0f);                            // [n_kv, n_tokens]
    }
    ggml_tensor * kq_mask_f = ggml_cast(ctx0, ggml_reshape_2d(ctx0, kq_mask, n_kv, n_tokens), GGML_TYPE_F32);
    // non candidates enter entmax with a finite -1e4 like the reference
    ggml_tensor * cand_neg = ggml_scale_bias(ctx0, inp->cand, 1e4f, -1e4f);
    ggml_tensor * rho      = ggml_arange(ctx0, 1.0f, (float) M + 1.0f, 1.0f);               // [M]
    ggml_tensor * rho_rep  = ggml_repeat_4d(ctx0, rho, M, n_tokens, 1, 1);
    ggml_tensor * cand3    = ggml_reshape_3d(ctx0, inp->cand, 1, nblk, n_tokens);

    // entmax input = dash_scale/sqrt(d) * (q . k_bar)/sqrt(d)
    const float idx_scale = hparams.dash_scale/sqrtf((float) n_idx);

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        const auto & layer = model.layers[il];
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = build_norm(inpL, layer.attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, n_embd_head, n_head, n_head, il);
            Qcur = rope2d(Qcur);
            Kcur = rope2d(Kcur);
            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);

            // router q/k: project, gemma rms norm, page rope; keys go to the indexer cache
            ggml_tensor * iq = ggml_reshape_3d(ctx0, build_lora_mm(layer.index_q_proj, cur), n_idx, 1, n_tokens);
            ggml_tensor * ik = ggml_reshape_3d(ctx0, build_lora_mm(layer.index_k_proj, cur), n_idx, 1, n_tokens);
            iq = rope2d(build_norm(iq, layer.index_q_norm, NULL, LLM_NORM_RMS, il));
            ik = rope2d(build_norm(ik, layer.index_k_norm, NULL, LLM_NORM_RMS, il));
            cb(iq, "dash_iq", il);

            ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, ik, inp_attn->get_k_idxs_idx(), il));
            ggml_tensor * ik_kv = mctx_idx->get_k(ctx0, il);
            ik_kv = ggml_view_2d(ctx0, ik_kv, n_idx, n_kv, ik_kv->nb[2], 0);

            // block summaries: head_cls attention-pooled router keys of each position block
            ggml_tensor * ikp = ggml_get_rows(ctx0, ik_kv, inp->pos_slot);                     // [n_idx, n_ps]
            ggml_tensor * aw  = ggml_mul_mat(ctx0, ikp, layer.index_head_cls);                // [n_ps, 1]
            aw = ggml_add(ctx0, ggml_scale(ctx0, aw, 1.0f/sqrtf((float) n_idx)), inp->pos_valid);
            ggml_tensor * w = ggml_soft_max(ctx0, ggml_reshape_2d(ctx0, aw, blk, nblk));       // [blk, nblk]
            ggml_tensor * ikt = ggml_cont(ctx0, ggml_permute(ctx0,
                    ggml_reshape_3d(ctx0, ikp, n_idx, blk, nblk), 1, 0, 2, 3));                // [blk, n_idx, nblk]
            ggml_tensor * kbar = ggml_mul_mat(ctx0, ikt, ggml_reshape_3d(ctx0, w, blk, 1, nblk)); // [n_idx, 1, nblk]
            kbar = ggml_reshape_2d(ctx0, kbar, n_idx, nblk);

            // block scores -> entmax inputs (non candidates at -1e4)
            ggml_tensor * z = ggml_mul_mat(ctx0, kbar, ggml_reshape_2d(ctx0, iq, n_idx, n_tokens)); // [nblk, n_tokens]
            ggml_mul_mat_set_prec(z, GGML_PREC_F32);
            z = ggml_clamp(ctx0, ggml_scale(ctx0, z, 1.0f/sqrtf((float) n_idx)), -1e4f, 1e4f);
            ggml_tensor * x = ggml_add(ctx0, ggml_mul(ctx0, ggml_scale(ctx0, z, idx_scale), inp->cand), cand_neg);
            cb(x, "dash_x", il);

            // top-M window (sorted), entmax-1.5 prior, then the first K slots form the read set
            ggml_tensor * topi = ggml_argsort_top_k(ctx0, x, M);                               // [M, n_tokens]
            ggml_tensor * xv   = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, x, 1, nblk, n_tokens), topi), M, n_tokens);
            ggml_tensor * vv   = ggml_reshape_2d(ctx0, ggml_get_rows(ctx0, cand3, topi), M, n_tokens);

            ggml_tensor * X  = ggml_scale(ctx0, ggml_sub(ctx0, xv, ggml_cont(ctx0, ggml_view_2d(ctx0, xv, 1, n_tokens, xv->nb[1], 0))), 0.5f);
            ggml_tensor * mn = ggml_div(ctx0, ggml_cumsum(ctx0, X), rho);
            ggml_tensor * ms = ggml_div(ctx0, ggml_cumsum(ctx0, ggml_sqr(ctx0, X)), rho);
            ggml_tensor * ss = ggml_mul(ctx0, ggml_sub(ctx0, ms, ggml_sqr(ctx0, mn)), rho);
            ggml_tensor * dl = ggml_relu(ctx0, ggml_div(ctx0, ggml_scale_bias(ctx0, ss, -1.0f, 1.0f), rho));
            ggml_tensor * tau = ggml_sub(ctx0, mn, ggml_sqrt(ctx0, dl));
            ggml_tensor * sup = ggml_sum_rows(ctx0, ggml_step(ctx0, ggml_sub(ctx0, X, tau)));   // [1, n_tokens]
            sup = ggml_clamp(ctx0, sup, 1.0f, (float) M);
            ggml_tensor * sel = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_abs(ctx0, ggml_sub(ctx0, rho_rep, sup)), -1.0f, 0.5f));
            ggml_tensor * tau_s = ggml_sum_rows(ctx0, ggml_mul(ctx0, sel, tau));                // [1, n_tokens]
            ggml_tensor * prior = ggml_sqr(ctx0, ggml_relu(ctx0, ggml_sub(ctx0, X, tau_s)));   // [M, n_tokens]

            ggml_tensor * pK = ggml_cont(ctx0, ggml_view_2d(ctx0, prior, K, n_tokens, prior->nb[1], 0));
            ggml_tensor * vK = ggml_cont(ctx0, ggml_view_2d(ctx0, vv,    K, n_tokens, vv->nb[1],    0));
            ggml_tensor * iK = ggml_cont(ctx0, ggml_view_2d(ctx0, topi,  K, n_tokens, topi->nb[1],  0));

            ggml_tensor * logp = ggml_log(ctx0, ggml_clamp(ctx0, pK, hparams.dash_prior_floor, 1e30f));
            ggml_tensor * cnt  = ggml_clamp(ctx0, ggml_sum_rows(ctx0, vK), 1.0f, 1e30f);
            ggml_tensor * mu   = ggml_div(ctx0, ggml_sum_rows(ctx0, ggml_mul(ctx0, logp, vK)), cnt);
            ggml_tensor * bias = ggml_scale(ctx0, ggml_sub(ctx0, logp, mu), 1.0f/hparams.dash_sigma);
            // invalid slots of the read set stay masked
            bias = ggml_add(ctx0, ggml_mul(ctx0, bias, vK), ggml_scale_bias(ctx0, vK, -NEG, NEG));
            cb(bias, "dash_bias", il);

            // block mask: routed bias for the read set, NEG elsewhere; expand blocks -> cells
            ggml_tensor * bm = ggml_scale_bias(ctx0, inp->cand, 0.0f, NEG);
            bm = ggml_set_rows(ctx0, ggml_reshape_3d(ctx0, bm, 1, nblk, n_tokens),
                    ggml_reshape_3d(ctx0, bias, 1, K, n_tokens), iK);
            bm = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_reshape_2d(ctx0, bm, nblk, n_tokens))); // [n_tokens, nblk]
            ggml_tensor * mask = ggml_cont(ctx0, ggml_transpose(ctx0, ggml_get_rows(ctx0, bm, inp->cell_blk))); // [n_kv, n_tokens]
            // the token-level local window wins over the block mask; causal/cell masking comes from the kv mask
            mask = ggml_add(ctx0, ggml_mul(ctx0, mask, notlocal), kq_mask_f);
            if (kq_mask->type != GGML_TYPE_F32) {
                mask = ggml_cast(ctx0, mask, kq_mask->type);
            }
            mask = ggml_reshape_4d(ctx0, mask, n_kv, n_tokens, 1, 1);
            cb(mask, "dash_mask", il);

            // main attention over the full cache with the routed mask
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Vcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, mctx_base->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
            ggml_build_forward_expand(gf, mctx_base->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
            ggml_tensor * k = mctx_base->get_k(ctx0, il);
            ggml_tensor * v = mctx_base->get_v(ctx0, il);

            cur = build_attn_mha(Qcur, k, v, nullptr, mask, nullptr, nullptr, 0, kq_scale, il);
            cb(cur, "kqv_out", il);
            cur = build_lora_mm(layer.wo, cur);
            cb(cur, "attn_out", il);
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
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
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    nullptr,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr,
                    layer.ffn_gate_up_exps);
            cb(moe_out, "ffn_moe_out", il);

            ggml_tensor * ffn_shexp = build_ffn(cur,
                    layer.ffn_up_shexp,   NULL, NULL,
                    layer.ffn_gate_shexp, NULL, NULL,
                    layer.ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SILU, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    ggml_tensor * cur = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
