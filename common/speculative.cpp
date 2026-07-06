#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "../src/llama-ext.h" // staging API: llama_set_embeddings_nextn / llama_get_embeddings_nextn_ith (used by MTP)
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "sampling.h"
#include "suffix-tree.h"
#include "../src/dflash-profile.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstring>
#include <iomanip>
#include <limits>
#include <map>
#include <unordered_map>
#include <cmath>
#include <cinttypes>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

const std::map<std::string, common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft-simple",  COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"draft-eagle3",  COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3},
    {"draft-mtp",     COMMON_SPECULATIVE_TYPE_DRAFT_MTP},
    {"ngram-simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram-map-k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram-map-k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram-mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram-cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"suffix",        COMMON_SPECULATIVE_TYPE_SUFFIX},
    {"copyspec",      COMMON_SPECULATIVE_TYPE_COPYSPEC},
    {"recycle",       COMMON_SPECULATIVE_TYPE_RECYCLE},
    {"dflash",        COMMON_SPECULATIVE_TYPE_DFLASH},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE},
    {"mtp",           COMMON_SPECULATIVE_TYPE_DRAFT_MTP}
};

static std::string common_speculative_get_devices_str(const std::vector<ggml_backend_dev_t> & devices) {
    std::string result;
    for (size_t i = 0; i < devices.size(); i++) {
        if (devices[i] == nullptr) {
            continue;
        }
        if (!result.empty()) result += ", ";
        result += ggml_backend_dev_name(devices[i]);
    }
    return result.empty() ? "default" : result;
}

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_dflash_debug_logs_enabled() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_DEBUG");
        return env && std::atoi(env) != 0;
    }();
    return enabled;
}

static bool common_dflash_kv_cache_disabled() {
    static const bool disabled = [] {
        const char * env = std::getenv("GGML_DFLASH_DISABLE_KV_CACHE");
        if (env && std::atoi(env) != 0) {
            return true;
        }

        const char * mode = std::getenv("GGML_DFLASH_KV_CACHE_MODE");
        return mode && (std::strcmp(mode, "off") == 0 ||
                        std::strcmp(mode, "none") == 0 ||
                        std::strcmp(mode, "disabled") == 0);
    }();
    return disabled;
}

static bool common_dflash_gpu_ring_env_enabled() {
    const char * env = std::getenv("GGML_DFLASH_GPU_RING");
    return env == nullptr || std::atoi(env) != 0;
}

static bool common_dflash_force_cpu_cross_ring() {
    static const bool forced = [] {
        const char * env = std::getenv("GGML_DFLASH_FORCE_CPU_CROSS");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return forced;
}

static bool common_dflash_log_contract_verbose() {
    static const bool enabled = [] {
        const char * env = std::getenv("GGML_DFLASH_VERBOSE_CONTRACT");
        return env && env[0] != '\0' && std::strcmp(env, "0") != 0;
    }();
    return enabled;
}

static bool common_dflash_target_capture_ready_or_skip(llama_context * ctx_tgt) {
    if (llama_dflash_hidden_capture_available(ctx_tgt)) {
        return true;
    }

    LOG_WRN("dflash: target hidden capture unavailable: %s\n",
        llama_dflash_hidden_capture_unavailable_reason(ctx_tgt));
    return false;
}

static bool common_dflash_gpu_ring_allowed(llama_context * ctx_tgt, llama_context * ctx_dft) {
    if (!common_dflash_gpu_ring_env_enabled()) {
        LOG_INF("dflash: GPU cross ring disabled by GGML_DFLASH_GPU_RING=0; using CPU hidden capture\n");
        return false;
    }

    const int32_t n_tgt_devices = ctx_tgt ? llama_model_n_devices(llama_get_model(ctx_tgt)) : 1;
    const int32_t n_dft_devices = ctx_dft ? llama_model_n_devices(llama_get_model(ctx_dft)) : 1;
    if (n_tgt_devices > 1 || n_dft_devices > 1) {
        if (!llama_dflash_allow_multi_gpu_tape()) {
            LOG_INF("dflash: multi-GPU placement detected (target=%d devices, drafter=%d devices); disabling GPU cross ring and graph hidden capture by multi-GPU tape kill switch\n",
                    n_tgt_devices, n_dft_devices);
            return false;
        }
        LOG_INF("dflash: multi-GPU placement detected (target=%d devices, drafter=%d devices); enabling GPU cross ring and graph hidden capture\n",
                n_tgt_devices, n_dft_devices);
    }

    return true;
}

static void common_dflash_clear_drafter_seq(llama_context * ctx_dft, llama_seq_id seq_id, const char * reason) {
    auto * mem_dft = ctx_dft ? llama_get_memory(ctx_dft) : nullptr;
    if (!mem_dft) {
        return;
    }

    if (common_dflash_debug_logs_enabled()) {
        LOG_DBG("DFLASH_DBG clear drafter seq: seq=%d reason=%s pos_max=%d\n",
                seq_id, reason && reason[0] ? reason : "-",
                (int) llama_memory_seq_pos_max(mem_dft, seq_id));
    }

    llama_memory_seq_rm(mem_dft, seq_id, -1, -1);
}

static void common_dflash_reset_drafter_seq_and_kv_cache(llama_context * ctx_dft, llama_seq_id seq_id, const char * reason) {
    common_dflash_clear_drafter_seq(ctx_dft, seq_id, reason);
    if (ctx_dft) {
        llama_dflash_kv_cache_set_active_seq(ctx_dft, seq_id);
        llama_dflash_kv_cache_reset(ctx_dft);
    }
}

static void common_dflash_align_drafter_seq_or_clear(
        llama_context * ctx_dft,
        llama_seq_id   seq_id,
        llama_pos      next_pos,
        const char   * where) {
    auto * mem_dft = ctx_dft ? llama_get_memory(ctx_dft) : nullptr;
    if (!mem_dft) {
        return;
    }

    const llama_pos pos_max = llama_memory_seq_pos_max(mem_dft, seq_id);
    if (pos_max < 0 || pos_max + 1 == next_pos) {
        return;
    }

    LOG_WRN("dflash: clearing stale drafter KV for seq=%d before %s: pos_max=%d next_pos=%d\n",
            seq_id, where && where[0] ? where : "draft decode", (int) pos_max, (int) next_pos);
    common_dflash_reset_drafter_seq_and_kv_cache(ctx_dft, seq_id, where);
}

static bool common_dflash_argmax_token_valid(int32_t token_id, int n_vocab) {
    return token_id >= 0 && token_id < n_vocab;
}

static int common_dflash_invalid_reduced_logits_next_streak(
        int  current_streak,
        bool valid_draft) {
    if (valid_draft) {
        return 0;
    }

    return current_streak < INT_MAX ? current_streak + 1 : current_streak;
}

static bool common_dflash_invalid_reduced_logits_fail_closed(
        int invalid_streak,
        int threshold) {
    return threshold > 0 && invalid_streak >= threshold;
}

static constexpr int DFLASH_INVALID_REDUCED_LOGITS_FAIL_CLOSED_STREAK = 4;

static bool common_dflash_argmax_shape_valid(
        const char * where,
        int rows_available,
        int rows_required,
        int top_k) {
    if (top_k < 1 || rows_available < rows_required) {
        LOG_ERR("dflash: invalid reduced-logits shape in %s (rows=%d required=%d top_k=%d)\n",
                where, rows_available, rows_required, top_k);
        return false;
    }

    return true;
}

common_dflash_ring_write common_dflash_ring_write_plan(int ring_size, int ring_pos, int n_tokens) {
    if (ring_size <= 0 || n_tokens <= 0) {
        return { 0, 0, 0 };
    }

    int normalized_pos = ring_pos % ring_size;
    if (normalized_pos < 0) {
        normalized_pos += ring_size;
    }

    if (n_tokens <= ring_size) {
        return { normalized_pos, n_tokens, 0 };
    }

    const int skip = n_tokens - ring_size;
    normalized_pos = (normalized_pos + skip) % ring_size;
    return { normalized_pos, ring_size, skip };
}

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

using common_speculative_draft_params_vec = std::vector<common_speculative_draft_params>;

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_impl
struct common_speculative_impl {
    const common_speculative_type type;

    uint32_t n_seq;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_impl(common_speculative_type type, uint32_t n_seq) : type(type), n_seq(n_seq) {}

    virtual ~common_speculative_impl() = default;

    virtual void begin(llama_seq_id seq_id, const llama_tokens & prompt) = 0;

    virtual bool process(const llama_batch & batch) = 0;

    virtual void draft(common_speculative_draft_params_vec & dparams) = 0;

    virtual void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) = 0;

    // true if this implementation requires the target context to extract post-norm embeddings
    virtual bool need_embd() const = 0;

    // true if this implementation requires the target context to extract nextn embeddings
    virtual bool need_embd_nextn() const { return false; }

    virtual void set_seq_id(llama_seq_id /*seq_id*/) {}

    virtual void draft_tree(
            const common_params_speculative & /*params*/,
            const llama_tokens              & /*prompt_tgt*/,
            llama_token                       /*id_last*/,
            int                               /*tree_budget*/,
            common_speculative_tree         & /*tree*/) {}

    virtual void update_logits(llama_context * /*ctx*/, const llama_tokens & /*batch_tokens*/, int /*n_accepted*/) {}
    virtual void update_logits_deferred_dflash_kv(llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) {
        update_logits(ctx, batch_tokens, n_accepted);
    }

    virtual void update_logits_by_indices(llama_context * /*ctx*/, const std::vector<int> & /*capture_indices*/) {}

    virtual int flush_prefill(int /*src_offset*/ = 0, int /*n_tokens*/ = 0) { return 0; }

    virtual int prepare_batch_draft(llama_context * /*ctx_dft_ext*/) { return -1; }

    virtual void set_prefill_capture_enabled(bool /*enabled*/) {}

    virtual void discard_dflash_state(const char * /*reason*/) {}

    virtual void note_prefill_suffix_scheduled() {}

    virtual size_t ring_state_size() const { return 0; }

    virtual bool ring_state_save(uint8_t * /*buf*/, size_t /*size*/) const { return false; }

    virtual bool ring_state_load(const uint8_t * /*buf*/, size_t /*size*/) { return false; }

    virtual common_dflash_ring_stats dflash_ring_stats() const { return {}; }
};

struct common_speculative_impl_draft_simple : public common_speculative_impl {
    common_params_speculative_draft params;

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    common_speculative_impl_draft_simple(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, n_seq)
        , params(params.draft)
    {
        auto * ctx_dft = this->params.ctx_dft;
        auto * ctx_tgt = this->params.ctx_tgt;

        LOG_INF("%s: adding speculative implementation 'draft-simple'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%f\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min);
        LOG_INF("%s: - gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", __func__,
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        if (!ctx_tgt || !ctx_dft) {
            throw std::runtime_error("draft-simple requires ctx_tgt and ctx_dft");
        }

        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }

        smpls.resize(n_seq);
        for (auto & smpl : smpls) {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl.reset(common_sampler_init(llama_get_model(ctx_dft), params));
        }

        const bool vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("%s: vocab_cmpt = %d\n", __func__, vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_ERR("%s: the target and draft vocabs are not compatible\n", __func__);

            throw std::runtime_error("draft model vocab type must match target model to use speculation");
        }

        if (n_seq != llama_n_seq_max(ctx_dft)) {
            LOG_ERR("%s: n_seq mismatch: %d != %d\n", __func__, n_seq, llama_n_seq_max(ctx_dft));

            throw std::runtime_error("the draft model number of sequences is incompatible with the speculative n_seq");
        }
    }

    ~common_speculative_impl_draft_simple() override {
        llama_batch_free(batch);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & batch) override {
        auto * ctx_dft = params.ctx_dft;

        std::vector<llama_pos> pos_min_by_seq(n_seq, 0);
        std::vector<bool> seen_seq(n_seq, false);
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (!batch.seq_id || !batch.seq_id[i]) {
                continue;
            }
            for (int32_t j = 0; j < batch.n_seq_id[i]; ++j) {
                const llama_seq_id seq_id = batch.seq_id[i][j];
                if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
                    continue;
                }
                if (!seen_seq[seq_id] || batch.pos[i] < pos_min_by_seq[seq_id]) {
                    pos_min_by_seq[seq_id] = batch.pos[i];
                    seen_seq[seq_id] = true;
                }
            }
        }
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (seen_seq[seq_id]) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, pos_min_by_seq[seq_id], -1);
            }
        }

        const int ret = llama_decode(ctx_dft, batch);

        if (ret != 0) {
            LOG_ERR("%s: failed to decode draft batch, ret = %d\n", __func__, ret);

            return false;
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, dp.n_past, -1);

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                    LOG_DBG(" - seq_id %d, draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                            seq_id, k, i, cur_p->data[k].id, cur_p->data[k].p,
                            common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
                }

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if ((params.n_max <= (int) result.size()) ||
                    (dp.n_max > 0 && dp.n_max <= (int) result.size())) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (auto & dp : dparams) {
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_eagle3 : public common_speculative_impl {
    //common_params_speculative_eagle3 params;

    common_speculative_impl_draft_eagle3(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, n_seq)
    {
        LOG_INF("%s: adding speculative implementation 'draft-eagle3'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%f\n", __func__, params.draft.n_max, params.draft.n_min, params.draft.p_min);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & /*dparams*/) override {
        // TODO: implement
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_draft_mtp : public common_speculative_impl {
    common_params_speculative_draft params; // reuses the draft-model params slot (ctx_tgt/ctx_dft)

    llama_batch batch;

    std::vector<common_sampler_ptr> smpls;

    // backend sampler chain per seq, attached to ctx_dft
    std::vector<llama_sampler *> backend_chains;

    int32_t n_embd = 0;

    // Per-sequence cross-batch carryover: pair (h_p, x_{p+1}) at MTP pos p+1.
    // The last h-row of one process() call needs the first token of the NEXT
    // call to pair with, so it's stashed here until that next call fires.
    std::vector<std::vector<float>> pending_h;   // [n_seq][n_embd]

    std::vector<int32_t> i_batch_beg;
    std::vector<int32_t> i_batch_end;

    // Hidden rows from the most recent target verification batch, grouped by seq.
    // Row 0 corresponds to the sampled token, row N to the Nth accepted draft token.
    std::vector<std::vector<float>> verify_h;
    std::vector<int32_t> verify_h_rows;

    // Per-seq draft length from the last draft() call, used in accept() to
    // roll back ctx_dft's recurrent state past the AR draft's redundant
    // pre-advancement before process() mirrored the verify batch.
    std::vector<uint16_t> last_n_drafted;

    common_speculative_impl_draft_mtp(const common_params_speculative & params, uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, n_seq)
        , params(params.draft)
    {
        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;
        GGML_ASSERT(ctx_tgt && ctx_dft && "MTP requires ctx_tgt and ctx_dft to be set");

        n_embd = llama_model_n_embd(llama_get_model(ctx_dft));

        LOG_INF("%s: adding speculative implementation 'draft-mtp'\n", __func__);
        LOG_INF("%s: - n_max=%d, n_min=%d, p_min=%.2f, n_embd=%d, backend_sampling=%d\n", __func__, this->params.n_max, this->params.n_min, this->params.p_min, n_embd, (int) this->params.backend_sampling);
        LOG_INF("%s: - gpu_layers=%d, cache_k=%s, cache_v=%s, ctx_tgt=%s, ctx_dft=%s, devices=[%s]\n", __func__,
                this->params.n_gpu_layers,
                ggml_type_name(this->params.cache_type_k),
                ggml_type_name(this->params.cache_type_v),
                ctx_tgt ? "yes" : "no",
                ctx_dft ? "yes" : "no",
                common_speculative_get_devices_str(this->params.devices).c_str());

        const int32_t n_b = (int32_t) llama_n_batch(ctx_dft);
        batch = llama_batch_init(/*n_tokens=*/ n_b, /*embd=*/ n_embd, /*n_seq_max=*/ 1);
        // llama_batch_init allocates only one of token/embd; MTP needs both.
        // TODO: fix, how to call without malloc
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        smpls.resize(n_seq);
        for (auto & s : smpls) {
            common_params_sampling sparams;
            sparams.no_perf  = false;
            sparams.top_k    = 10;
            sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };
            s.reset(common_sampler_init(llama_get_model(ctx_dft), sparams));
        }

        // offload draft sampling to the backend
        backend_chains.assign(n_seq, nullptr);
        if (this->params.backend_sampling) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
                llama_sampler_chain_add(chain, llama_sampler_init_top_k(10));

                if (!llama_set_sampler(ctx_dft, seq_id, chain)) {
                    LOG_WRN("%s: backend offload failed for seq_id=%d; using CPU sampler\n", __func__, (int) seq_id);
                    llama_sampler_free(chain);
                    chain = nullptr;
                }
                backend_chains[seq_id] = chain;
            }
        }

        llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked*/ true);

        pending_h.assign(n_seq, std::vector<float>(n_embd, 0.0f));

        i_batch_beg.assign(n_seq, -1);
        i_batch_end.assign(n_seq, -1);

        verify_h.assign(n_seq, {});
        verify_h_rows.assign(n_seq, 0);

        last_n_drafted.assign(n_seq, 0);
    }

    ~common_speculative_impl_draft_mtp() override {
        auto * ctx_dft = this->params.ctx_dft;
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) backend_chains.size(); ++seq_id) {
            if (backend_chains[seq_id] == nullptr) {
                continue;
            }
            if (ctx_dft) {
                llama_set_sampler(ctx_dft, seq_id, nullptr);
            }
            llama_sampler_free(backend_chains[seq_id]);
        }
        backend_chains.clear();

        if (batch.token != nullptr) {
            free(batch.token);
            batch.token = nullptr;
        }
        llama_batch_free(batch);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        const int32_t N = (int32_t) prompt.size();
        if (N <= 0) {
            return;
        }
        auto * ctx_dft = this->params.ctx_dft;
        const llama_pos pos_max = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), seq_id);
        if (pos_max < N - 1) {
            LOG_WRN("%s: ctx_dft pos_max=%d < N-1=%d - "
                    "process() hook may not have run on every prefill ubatch "
                    "(need_embd / logits=1 on every prompt position?). "
                    "Drafts may degrade.\n",
                    __func__, (int) pos_max, N - 1);
        }
    }

    bool process(const llama_batch & batch_in) override {
        if (batch_in.n_tokens <= 0) {
            return true;
        }

        // TODO: how to make it work with vision tokens?
        if (batch_in.token == nullptr || batch_in.embd != nullptr) {
            return true;
        }

        const int32_t n_tokens = batch_in.n_tokens;

        // remember the frist and last batch index for each sequence
        std::fill(i_batch_beg.begin(), i_batch_beg.end(), -1);
        std::fill(i_batch_end.begin(), i_batch_end.end(), -1);

        for (int k = 0; k < n_tokens; ++k) {
            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                GGML_ASSERT(batch_in.n_seq_id[k] == 1);

                if (batch_in.seq_id[k][0] == seq_id) {
                    i_batch_end[seq_id] = k;
                    if (i_batch_beg[seq_id] < 0) {
                        i_batch_beg[seq_id] = k;
                    }
                }
            }
        }

        auto * ctx_tgt = this->params.ctx_tgt;
        auto * ctx_dft = this->params.ctx_dft;

        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        common_batch_clear(batch);

        for (int k = 0; k < n_tokens; ++k) {
            common_batch_add(batch, batch_in.token[k], batch_in.pos[k], { batch_in.seq_id[k][0] }, 0);
        }

        // shift the tgt embeddings to the right by one position
        // assumes that the tokens in the batch are sequential for each sequence
        // i.e. we cannot have seq_id like this: [0, 0, 0, 1, 1, 0, 1, 1]
        //                                                       ^--- this is a problem
        // TODO:this is generally true, but would be nice to assert it
        {
            const float * h_tgt = llama_get_embeddings_nextn(ctx_tgt);
            if (h_tgt == nullptr) {
                LOG_ERR("%s: target nextn embeddings are not available\n", __func__);
                return false;
            }
            std::memcpy(batch.embd + (size_t) 1 * n_embd, h_tgt, row_bytes * (n_tokens-1));

        }

        // fill the pending embeddings from a previous run
        auto set_h = [&](int idx, const float * h_row) {
            std::memcpy(batch.embd + (size_t) idx * n_embd, h_row, row_bytes);
        };

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_beg[seq_id] < 0) {
                continue;
            }

            set_h(i_batch_beg[seq_id], pending_h[seq_id].data());
        }

        const int32_t rc = llama_decode(ctx_dft, batch);
        if (rc != 0) {
            LOG_ERR("%s: llama_decode(ctx_dft) failed rc=%d (pos=%d)\n", __func__, (int) rc, (int) batch_in.pos[0]);
            return false;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            if (i_batch_end[seq_id] < 0) {
                continue;
            }

            const int32_t n_rows = i_batch_end[seq_id] - i_batch_beg[seq_id] + 1;
            verify_h_rows[seq_id] = n_rows;
            verify_h[seq_id].resize((size_t) n_rows * n_embd);

            for (int32_t i = 0; i < n_rows; ++i) {
                const float * h = llama_get_embeddings_nextn_ith(ctx_tgt, i_batch_beg[seq_id] + i);
                if (h == nullptr) {
                    LOG_ERR("%s: target nextn embedding row %d is not available\n",
                            __func__, (int) (i_batch_beg[seq_id] + i));
                    return false;
                }
                std::memcpy(verify_h[seq_id].data() + (size_t) i * n_embd, h, row_bytes);
            }

            std::memcpy(pending_h[seq_id].data(),
                    verify_h[seq_id].data() + (size_t) (n_rows - 1) * n_embd, row_bytes);
        }

        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        auto & ctx_dft = params.ctx_dft;

        common_batch_clear(batch);

        // keep track of which sequences are still drafting
        int n_drafting = 0;
        std::vector<bool> drafting(n_seq);

        const float * h_row = nullptr;
        const size_t row_bytes = (size_t) n_embd * sizeof(float);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];

            if (!dp.drafting) {
                continue;
            }

            n_drafting++;
            drafting[seq_id] = true;
            common_sampler_reset(smpls[seq_id].get());

            common_batch_add(batch, dp.id_last, dp.n_past, { seq_id }, true);

            h_row = pending_h[seq_id].data();
            std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
        }

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0) {
            LOG_WRN("%s: llama_decode returned %d\n", __func__, ret);
            return;
        }

        int i = 0;

        while (n_drafting > 0) {
            int i_batch = 0;

            common_batch_clear(batch);

            for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
                if (!drafting[seq_id]) {
                    continue;
                }

                auto * smpl = smpls[seq_id].get();

                common_sampler_sample(smpl, ctx_dft, i_batch, true);
                h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_batch);
                if (h_row == nullptr) {
                    LOG_WRN("%s: draft nextn embedding row %d is not available; stopping MTP draft for seq_id=%d\n",
                            __func__, i_batch, (int) seq_id);
                    drafting[seq_id] = false;
                    n_drafting--;
                    ++i_batch;
                    continue;
                }
                ++i_batch;

                const auto * cur_p = common_sampler_get_candidates(smpl, true);

                // add drafted token for each sequence
                const llama_token id = cur_p->data[0].id;

                // only collect very high-confidence draft tokens
                if (cur_p->data[0].p < params.p_min) {
                    drafting[seq_id] = false;
                    n_drafting--;

                    continue;
                }

                common_sampler_accept(smpl, id, true);

                auto & dp = dparams.at(seq_id);
                auto & result = *dp.result;

                result.push_back(id);

                if (params.n_max <= (int) result.size()) {
                    drafting[seq_id] = false;
                    n_drafting--;
                    continue;
                }

                common_batch_add(batch, id, dp.n_past + i + 1, { seq_id }, true);
                std::memcpy(batch.embd + n_embd*(batch.n_tokens - 1), h_row, row_bytes);
            }

            if (batch.n_tokens == 0) {
                break;
            }

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d\n", __func__, i, ret);
                break;
            }

            ++i;
        }

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            if (dp.result->size() < (size_t) params.n_min) {
                dp.result->clear();
            }

            last_n_drafted[seq_id] = (uint16_t) dp.result->size();
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool /*is_other*/) override {
        if (seq_id < 0 || seq_id >= (llama_seq_id) n_seq) {
            return;
        }

        const int32_t n_rows = verify_h_rows[seq_id];
        if (n_rows <= 0) {
            return;
        }

        const int32_t i_h = std::min<int32_t>(n_accepted, n_rows - 1);
        const size_t row_bytes = (size_t) n_embd * sizeof(float);
        std::memcpy(pending_h[seq_id].data(), verify_h[seq_id].data() + (size_t) i_h * n_embd, row_bytes);
    }

    bool need_embd() const override {
        return false;
    }

    bool need_embd_nextn() const override {
        return true;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_impl_ngram_simple : public common_speculative_impl {
    common_params_speculative_ngram_map params;

    // shared across all sequences
    common_ngram_simple_config config;

    common_speculative_impl_ngram_simple(
            const common_params_speculative & params, uint32_t n_seq,
            common_ngram_simple_config config)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, n_seq)
        , params(params.ngram_simple)
        , config(config)
    {
        LOG_INF("%s: adding speculative implementation 'ngram-simple'\n", __func__);
        LOG_INF("%s: - size_n=%d, size_m=%d, min_hits=%d\n", __func__,
                this->params.size_n, this->params.size_m, this->params.min_hits);
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            *dp.result = common_ngram_simple_draft(config, *dp.prompt, dp.id_last);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_map_k : public common_speculative_impl {
    // n_seq configs
    std::vector<common_ngram_map> config;

    common_speculative_impl_ngram_map_k(
            const common_ngram_map & config,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, n_seq)
    {
        for (uint32_t i = 0; i < n_seq; i++) {
            this->config.push_back(config);
        }

        LOG_INF("%s: adding speculative implementation '%s'\n", __func__, common_speculative_type_to_str(this->type).c_str());
        LOG_INF("%s: - size_key=%d, size_value=%d, key_only=%d, min_hits=%d\n", __func__,
                config.size_key, config.size_value, config.key_only, config.min_hits);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        GGML_ASSERT(seq_id < (llama_seq_id) n_seq);

        common_ngram_map_begin(config[seq_id], prompt);
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            common_ngram_map_draft(config[seq_id], *dp.prompt, dp.id_last, *dp.result);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        GGML_ASSERT((seq_id < (llama_seq_id) config.size()));

        if (is_other) {
            return;
        }

        common_ngram_map_accept(config[seq_id], n_accepted);
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_mod : public common_speculative_impl {
    common_params_speculative_ngram_mod params;

    // shared across all sequences
    common_ngram_mod mod;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    struct seq_info {
        // the last position in the prompt that was added to the ngram container
        size_t i_last = 0;

        // length of the last drafted n-gram (number of tokens returned by draft)
        size_t n_draft_last = 0;

        // consecutive accept rounds with low acceptance fraction (< 0.5)
        int n_low = 0;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_mod(
            const common_params_speculative & params,
            uint32_t n_seq)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, n_seq)
        , params(params.ngram_mod)
        , mod(params.ngram_mod.n_match, 4*1024*1024)
        , verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));

        LOG_INF("%s: adding speculative implementation 'ngram-mod'\n", __func__);
        LOG_INF("%s: - n_match=%d, n_max=%d, n_min=%d\n", __func__,
                this->params.n_match, this->params.n_max, this->params.n_min);
        LOG_INF("%s: - mod size=%zu (%.3f MB)\n", __func__,
                mod.size(), (float)(mod.size_bytes())/1024/1024);

        if (this->params.n_match < 16) {
            LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                    "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, this->params.n_match);
        }

        sinfos.resize(n_seq);
    }

    void begin(llama_seq_id seq_id, const llama_tokens & prompt) override {
        auto & sinfo = sinfos[seq_id];

        sinfo.i_last = 0;
        sinfo.n_draft_last = 0;

        const size_t n = mod.get_n();
        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        sinfo.i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        sinfo.n_draft_last = 0;

        const size_t cur_len = prompt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (sinfo.i_last + 32 < cur_len) {
            for (size_t i = sinfo.i_last; i < cur_len - n; ++i) {
                mod.add(prompt.data() + i);
            }

            sinfo.i_last = cur_len - n;
        }

        result.resize(n + params.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt.at(cur_len - n + 1 + i);
        }
        result[n - 1] = dparams.id_last;

        for (int i = 0; i < params.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < params.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n-gram for later acceptance analysis
        sinfo.n_draft_last = result.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id seq_id, uint16_t n_accepted, bool is_other) override {
        if (is_other) {
            return;
        }

        auto & sinfo = sinfos[seq_id];

        // compute acceptance fraction if we have a recorded draft length
        if (sinfo.n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)sinfo.n_draft_last;
            if (f_acc < 0.25) {
                sinfo.n_low++;
                if (sinfo.n_low >= 5) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) - resetting ngram_mod\n", __func__, sinfo.n_low);
                    }

                    mod.reset();
                    sinfo.n_low = 0;
                    sinfo.i_last = 0;
                }
            } else {
                sinfo.n_low = 0;
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

struct common_speculative_impl_ngram_cache : public common_speculative_impl {
    common_params_speculative_ngram_cache params;

    uint16_t n_draft;

    bool save_dynamic;
    bool save_static;

    struct seq_info {
        size_t cache_size = 0; // number of tokens in n-gram cache

        common_ngram_cache ngram_cache_context;
        common_ngram_cache ngram_cache_dynamic;
        common_ngram_cache ngram_cache_static;
    };

    std::vector<seq_info> sinfos;

    common_speculative_impl_ngram_cache(
            const common_params_speculative & params,
            uint32_t n_seq,
            uint16_t n_draft,
            const std::string & path_static,
            const std::string & path_dynamic,
            bool save_dynamic,
            bool save_static)
        : common_speculative_impl(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, n_seq)
        , params(params.ngram_cache)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        LOG_INF("%s: adding speculative implementation 'ngram-cache'\n", __func__);
        LOG_INF("%s: - n_draft=%d, cache_static=%s, cache_dynamic=%s\n", __func__,
                n_draft,
                path_static.empty() ? "none" : path_static.c_str(),
                path_dynamic.empty() ? "none" : path_dynamic.c_str());

        sinfos.resize(n_seq);

        if (!path_static.empty()) {
            try {
                auto ngram_cache_static = common_ngram_cache_load(path_static);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_static = ngram_cache_static;
                }
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                auto ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);

                for (auto & sinfo : sinfos) {
                    sinfo.ngram_cache_dynamic = ngram_cache_dynamic;
                }
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & /*prompt*/) override {
        // noop
    }

    void draft_one(
            llama_seq_id seq_id,
            common_speculative_draft_params & dparams) {
        auto & sinfo = sinfos[seq_id];
        auto & result = *dparams.result;

        const auto & prompt = *dparams.prompt;

        if (sinfo.cache_size < prompt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt.size() + 1 - sinfo.cache_size);
            for (size_t j = sinfo.cache_size; j < prompt.size(); ++j) {
                tokens_new.push_back(prompt[j]);
            }
            tokens_new.push_back(dparams.id_last); // add the last token

            // Update context ngram cache with new dparams.prompt:
            common_ngram_cache_update(
                    sinfo.ngram_cache_context,
                    LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            sinfo.cache_size = prompt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt.size() + 1);
        for (size_t j = 0; j < prompt.size(); ++j) {
            inp.push_back(prompt[j]);
        }
        inp.push_back(dparams.id_last);

        result.push_back(dparams.id_last);

        common_ngram_cache_draft(
                inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                sinfo.ngram_cache_context,
                sinfo.ngram_cache_dynamic,
                sinfo.ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        // TODO: implement
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        assert(dparams.size() == n_seq);

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            draft_one(seq_id, dp);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
        // noop
    }

    bool need_embd() const override {
        return false;
    }
};

// ============================================================================
// Fork-specific speculative implementations
// ============================================================================

// (fork classes will be inserted here via sed)
// Fork-specific speculative decoding classes, ported to common_speculative_impl base.
// Insert after the last upstream impl (ngram_cache) and before `struct common_speculative`.

// Checkpoint struct used by server for DFlash ring persistence
struct common_speculative_checkpoint {
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;

    int64_t   n_tokens = 0;

    std::vector<uint8_t> data;

    size_t size() const {
        return data.size();
    }
};

// ---- Suffix tree speculative decoding ----

struct common_speculative_impl_suffix : public common_speculative_impl {
    SuffixTree tree;
    static constexpr int SEQ_ID = 1;

    int32_t max_depth;
    int32_t n_draft_max;
    float   spec_factor;
    float   spec_offset;
    float   min_prob;

    size_t tree_size = 0;  // number of tokens fed to the tree (prompt_tgt.size() + 1)

    common_speculative_impl_suffix(
            common_speculative_type type,
            uint32_t n_seq,
            int32_t max_depth,
            int32_t n_draft_max,
            float   spec_factor,
            float   spec_offset,
            float   min_prob)
        : common_speculative_impl(type, n_seq)
        , tree(max_depth)
        , max_depth(max_depth)
        , n_draft_max(n_draft_max)
        , spec_factor(spec_factor)
        , spec_offset(spec_offset)
        , min_prob(min_prob)
    {}

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & prompt) override {
        tree = SuffixTree(max_depth);
        tree_size = 0;
        if (!prompt.empty()) {
            tree.extend(SEQ_ID, prompt.data(), prompt.size());
            tree_size = prompt.size();
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            const llama_tokens & prompt_tgt = *dp.prompt;
            const llama_token id_last = dp.id_last;
            llama_tokens & result = *dp.result;
            const int32_t n_max_eff = dp.n_max > 0 ? dp.n_max : n_draft_max;

            // feed new tokens to suffix tree (same pattern as ngram_cache)
            if (tree_size < prompt_tgt.size() + 1) {
                for (size_t j = tree_size; j < prompt_tgt.size(); ++j) {
                    tree.append(SEQ_ID, prompt_tgt[j]);
                }
                tree.append(SEQ_ID, id_last);
                tree_size = prompt_tgt.size() + 1;
            }

            // build full context for pattern matching
            std::vector<int32_t> context;
            context.reserve(prompt_tgt.size() + 1);
            for (size_t i = 0; i < prompt_tgt.size(); i++) {
                context.push_back(prompt_tgt[i]);
            }
            context.push_back(id_last);

            if (context.size() < 2) { continue; }

            SuffixDraft draft = tree.speculate(
                context.data(), context.size(),
                n_max_eff, spec_factor, spec_offset, min_prob, false);

            for (size_t i = 0; i < draft.token_ids.size(); i++) {
                result.push_back(draft.token_ids[i]);
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    }

    bool need_embd() const override {
        return false;
    }
};

// ---- CopySpec: draft by copying matching subsequences from the prompt context ----
// Builds a rolling-hash index of all gamma-length windows in the prompt.
// On each draft call, hashes the last gamma tokens of output and looks up matches.

struct common_speculative_impl_copyspec : public common_speculative_impl {
    static constexpr uint64_t FNV_OFFSET = 14695981039346656037ULL;
    static constexpr uint64_t FNV_PRIME  = 1099511628211ULL;

    int32_t gamma; // window size for matching

    // hash of gamma-length window -> position after the window in the prompt
    std::unordered_multimap<uint64_t, int32_t> index;
    llama_tokens prompt_tokens;

    int32_t original_prompt_size = 0;
    bool has_model_drafter = false; // true when paired with DFlash/draft (apply primary threshold)

    common_speculative_impl_copyspec(common_speculative_type type, uint32_t n_seq, int32_t gamma)
        : common_speculative_impl(type, n_seq)
        , gamma(gamma)
    {}

    static uint64_t hash_window(const llama_token * tokens, int32_t len) {
        uint64_t h = FNV_OFFSET;
        for (int32_t i = 0; i < len; i++) {
            h ^= (uint64_t)(uint32_t)tokens[i];
            h *= FNV_PRIME;
        }
        return h;
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & prompt) override {
        index.clear();
        prompt_tokens = prompt;
        original_prompt_size = (int32_t)prompt.size();
        if ((int32_t)prompt.size() <= gamma) {
            return;
        }
        for (int32_t i = 0; i <= (int32_t)prompt.size() - gamma; i++) {
            uint64_t h = hash_window(prompt.data() + i, gamma);
            index.emplace(h, i + gamma);
        }
    }

    bool process(const llama_batch & /*batch*/) override {
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            const llama_tokens & prompt_tgt = *dp.prompt;
            const llama_token id_last = dp.id_last;
            llama_tokens & result = *dp.result;
            const int32_t n_max_eff = dp.n_max > 0 ? dp.n_max : gamma;

            // build the full context (prompt_tgt + id_last)
            const int32_t ctx_len = (int32_t)prompt_tgt.size() + 1;
            if (ctx_len < gamma) {
                continue;
            }

            // hash the last gamma tokens of context
            std::vector<llama_token> window(gamma);
            const int32_t start = ctx_len - gamma;
            for (int32_t i = 0; i < gamma; i++) {
                const int32_t pos = start + i;
                window[i] = (pos < (int32_t)prompt_tgt.size()) ? prompt_tgt[pos] : id_last;
            }
            uint64_t h = hash_window(window.data(), gamma);

            // find longest match in prompt
            int32_t best_pos = -1;
            int32_t best_avail = 0; // uncapped available tokens after match
            auto range = index.equal_range(h);
            for (auto it = range.first; it != range.second; ++it) {
                int32_t pos = it->second;
                // verify hash match (collision check)
                if (pos < gamma || pos > (int32_t)prompt_tokens.size()) {
                    continue;
                }
                bool match = true;
                for (int32_t j = 0; j < gamma; j++) {
                    if (prompt_tokens[pos - gamma + j] != window[j]) {
                        match = false;
                        break;
                    }
                }
                if (!match) {
                    continue;
                }
                int32_t avail = (int32_t)prompt_tokens.size() - pos;
                if (avail > best_avail) {
                    best_avail = avail;
                    best_pos = pos;
                }
            }

            if (best_pos < 0) {
                continue;
            }

            // when paired with a model-based drafter, only fire as primary if the match
            // has enough original prompt tokens to justify skipping the model drafter
            if (has_model_drafter) {
                const int32_t avail_in_orig = std::max(0, original_prompt_size - best_pos);
                if (avail_in_orig < 2 * n_max_eff) {
                    continue;
                }
            }

            const int32_t draft_len = std::min(n_max_eff, best_avail);
            for (int32_t i = 0; i < draft_len; i++) {
                result.push_back(prompt_tokens[best_pos + i]);
            }
        }
    }

    // Extend an existing draft by looking for suffix matches at the end of (prompt + draft)
    void extend(
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result,
            int32_t n_max_ext) {
        if (result.empty() || index.empty()) {
            return;
        }

        // build full context: prompt_tgt + id_last + result
        // hash the last gamma tokens of this extended context
        const int32_t ctx_len = (int32_t)prompt_tgt.size() + 1 + (int32_t)result.size();
        if (ctx_len < gamma) {
            return;
        }

        std::vector<llama_token> window(gamma);
        const int32_t start = ctx_len - gamma;
        for (int32_t i = 0; i < gamma; i++) {
            const int32_t pos = start + i;
            if (pos < (int32_t)prompt_tgt.size()) {
                window[i] = prompt_tgt[pos];
            } else if (pos == (int32_t)prompt_tgt.size()) {
                window[i] = id_last;
            } else {
                window[i] = result[pos - (int32_t)prompt_tgt.size() - 1];
            }
        }
        uint64_t h = hash_window(window.data(), gamma);

        // find longest match
        int32_t best_pos = -1;
        int32_t best_len = 0;
        const int32_t max_ext = n_max_ext - (int32_t)result.size();
        if (max_ext <= 0) {
            return;
        }

        auto range = index.equal_range(h);
        for (auto it = range.first; it != range.second; ++it) {
            int32_t pos = it->second;
            if (pos < gamma || pos > (int32_t)prompt_tokens.size()) {
                continue;
            }
            bool match = true;
            for (int32_t j = 0; j < gamma; j++) {
                if (prompt_tokens[pos - gamma + j] != window[j]) {
                    match = false;
                    break;
                }
            }
            if (!match) {
                continue;
            }
            int32_t avail = std::min(max_ext, (int32_t)prompt_tokens.size() - pos);
            if (avail > best_len) {
                best_len = avail;
                best_pos = pos;
            }
        }

        if (best_pos < 0) {
            return;
        }

        for (int32_t i = 0; i < best_len; i++) {
            result.push_back(prompt_tokens[best_pos + i]);
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    }

    // incrementally extend index with accepted tokens
    void update_logits(llama_context * /*ctx*/, const llama_tokens & batch_tokens, int n_accepted) override {
        // batch_tokens = [id_last, draft0, draft1, ...], n_accepted of which were accepted
        for (int i = 0; i < n_accepted && i < (int)batch_tokens.size(); i++) {
            prompt_tokens.push_back(batch_tokens[i]);
            // add new gamma-length window ending at this position
            if ((int32_t)prompt_tokens.size() >= gamma) {
                int32_t start = (int32_t)prompt_tokens.size() - gamma;
                uint64_t h = hash_window(prompt_tokens.data() + start, gamma);
                index.emplace(h, (int32_t)prompt_tokens.size());
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

// ---- Token Recycling: adjacency matrix tracking top-k successors per token ----
// Seeded from observed bigrams, then updated from model logits after each
// verification decode. Logit-based entries have much higher scores and
// dominate the adjacency matrix after the first few iterations.

struct common_speculative_impl_recycle : public common_speculative_impl {
    int32_t k; // top-k successors per token

    // adjacency: token -> vector of (score, successor) pairs, sorted by score descending
    // scores: bigram observations use small integer counts (1, 2, ...),
    //         logit-derived entries use logit values (typically 10-30+ for top tokens)
    std::unordered_map<llama_token, std::vector<std::pair<float, llama_token>>> adj;

    size_t n_fed = 0;
    int32_t n_vocab = 0;

    common_speculative_impl_recycle(common_speculative_type type, uint32_t n_seq, int32_t k)
        : common_speculative_impl(type, n_seq)
        , k(k)
    {}

    void set_successors(llama_token tok, const float * logits, int32_t vocab_size) {
        // partial sort to find top-k logits
        std::vector<std::pair<float, llama_token>> top(k, std::make_pair(-INFINITY, (llama_token)-1));
        for (int32_t i = 0; i < vocab_size; i++) {
            if (logits[i] > top[k-1].first) {
                top[k-1] = std::make_pair(logits[i], (llama_token)i);
                // bubble up
                for (int32_t j = k-2; j >= 0; j--) {
                    if (top[j+1].first > top[j].first) {
                        std::swap(top[j], top[j+1]);
                    } else {
                        break;
                    }
                }
            }
        }
        // remove unfilled slots
        while (!top.empty() && top.back().second < 0) {
            top.pop_back();
        }
        adj[tok] = std::move(top);
    }

    void add_bigram(llama_token a, llama_token b) {
        auto & succs = adj[a];
        for (size_t i = 0; i < succs.size(); i++) {
            if (succs[i].second == b) {
                succs[i].first += 1.0f;
                while (i > 0 && succs[i].first > succs[i-1].first) {
                    std::swap(succs[i], succs[i-1]);
                    i--;
                }
                return;
            }
        }
        if ((int32_t)succs.size() < k) {
            succs.push_back(std::make_pair(1.0f, b));
        }
    }

    void begin(llama_seq_id /*seq_id*/, const llama_tokens & prompt) override {
        adj.clear();
        n_fed = 0;
        for (size_t i = 0; i + 1 < prompt.size(); i++) {
            add_bigram(prompt[i], prompt[i + 1]);
        }
        n_fed = prompt.size();
    }

    bool process(const llama_batch & /*batch*/) override {
        return true;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) n_seq; ++seq_id) {
            auto & dp = dparams[seq_id];
            if (!dp.drafting) {
                continue;
            }

            const llama_tokens & prompt_tgt = *dp.prompt;
            const llama_token id_last = dp.id_last;
            llama_tokens & result = *dp.result;
            const int32_t n_max_eff = dp.n_max > 0 ? dp.n_max : k;

            // feed new bigrams from generated tokens
            if (n_fed < prompt_tgt.size() + 1) {
                size_t start = (n_fed > 0) ? n_fed - 1 : 0;
                for (size_t i = start; i < prompt_tgt.size(); i++) {
                    llama_token next = (i + 1 < prompt_tgt.size()) ? prompt_tgt[i + 1] : id_last;
                    add_bigram(prompt_tgt[i], next);
                }
                n_fed = prompt_tgt.size() + 1;
            }

            // greedy walk through adjacency matrix
            llama_token cur = id_last;
            for (int32_t i = 0; i < n_max_eff; i++) {
                auto it = adj.find(cur);
                if (it == adj.end() || it->second.empty()) {
                    break;
                }
                cur = it->second[0].second;
                result.push_back(cur);
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t /*n_accepted*/, bool /*is_other*/) override {
    }

    void update_logits(llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) override {
        if (n_vocab == 0) {
            const llama_model * model = llama_get_model(ctx);
            n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
        }
        // update adjacency from logits for each position that had logits computed
        // batch_tokens[i] is the token at position i; logits at i predict its successor
        const int n_positions = std::min(n_accepted, (int)batch_tokens.size());
        for (int i = 0; i < n_positions; i++) {
            const float * logits = llama_get_logits_ith(ctx, i);
            if (logits) {
                set_successors(batch_tokens[i], logits, n_vocab);
            }
        }
    }

    bool need_embd() const override {
        return false;
    }
};

static std::string common_dflash_layer_ids_to_string(const std::vector<int32_t> & ids) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            ss << ",";
        }
        ss << ids[i];
    }
    ss << "]";
    return ss.str();
}

static const char * common_dflash_layer_attention_kind(const llama_model * model, int32_t il) {
    const int32_t is_swa = llama_model_is_swa_layer(model, il);
    if (is_swa < 0) {
        return "UNK";
    }
    return is_swa ? "SWA" : "FULL";
}

static std::string common_dflash_layer_ids_with_attention_to_string(
        const llama_model * model,
        const std::vector<int32_t> & ids) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            ss << ",";
        }
        ss << ids[i] << ":" << common_dflash_layer_attention_kind(model, ids[i]);
    }
    ss << "]";
    return ss.str();
}

static std::string common_dflash_swa_pattern_to_string(const llama_model * model, int32_t n_layer, int32_t max_layers = 32) {
    std::ostringstream ss;
    ss << "[";
    const int32_t n_show = std::min(n_layer, max_layers);
    for (int32_t il = 0; il < n_show; ++il) {
        if (il > 0) {
            ss << ",";
        }
        ss << common_dflash_layer_attention_kind(model, il);
    }
    if (n_layer > n_show) {
        ss << ",...";
    }
    ss << "]";
    return ss.str();
}

static void common_dflash_capture_attention_counts(
        const llama_model * model,
        const std::vector<int32_t> & ids,
        int & n_swa,
        int & n_full,
        int & n_unknown) {
    n_swa = 0;
    n_full = 0;
    n_unknown = 0;
    for (const int32_t il : ids) {
        const int32_t is_swa = llama_model_is_swa_layer(model, il);
        if (is_swa < 0) {
            n_unknown++;
        } else if (is_swa) {
            n_swa++;
        } else {
            n_full++;
        }
    }
}

static std::string common_dflash_capture_neighborhood_to_string(
        const llama_model * model,
        const std::vector<int32_t> & ids,
        int32_t n_layer,
        int32_t radius = 2) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i > 0) {
            ss << ",";
        }
        const int32_t center = ids[i];
        ss << center << ":{";
        bool first = true;
        for (int32_t il = std::max<int32_t>(0, center - radius);
                il <= std::min<int32_t>(n_layer - 1, center + radius); ++il) {
            if (!first) {
                ss << ",";
            }
            first = false;
            ss << il << ":" << common_dflash_layer_attention_kind(model, il);
        }
        ss << "}";
    }
    ss << "]";
    return ss.str();
}

static bool common_dflash_layer_ids_unique(const std::vector<int32_t> & ids) {
    std::set<int32_t> seen;
    for (const int32_t id : ids) {
        if (!seen.insert(id).second) {
            return false;
        }
    }
    return true;
}

// ---- DFlash block-diffusion speculative decoding ----
// Uses an external drafter model conditioned on target hidden states via KV injection

enum class dflash_capture_source {
    cpu_hidden,
    verify_gpu_hidden,
    prefill_gpu_hidden,
};

struct common_speculative_impl_dflash : public common_speculative_impl {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;
    llama_model   * model_dft;
    bool            owns_ctx_dft; // when false, ctx_dft is externally owned (shared across slots)
    llama_seq_id    seq_id = 0;   // which server slot this state owns

    float p_min; // minimum probability threshold from config

    int block_size;
    llama_token mask_token_id;
    int n_target_layers;
    int n_embd;
    int n_target_features;

    // Ring buffer for target hidden states — fixed memory regardless of context length
    // Stores last RING_SIZE tokens per layer in circular fashion
    static constexpr int RING_SIZE = 4096;

    // ring_buf[layer][slot * n_embd ... (slot+1) * n_embd - 1], slot = pos % RING_SIZE
    std::vector<std::vector<float>> ring_buf; // [n_target_layers][RING_SIZE * n_embd]
    int ring_write_pos = 0;    // next write slot (0..RING_SIZE-1)
    int ring_filled = 0;       // how many valid slots (0..RING_SIZE)
    int committed_len = 0;     // total tokens committed (unbounded counter)
    bool cpu_ring_valid = true;
    bool prefill_flushed = false; // true if flush_prefill() produced tokens during this request
    bool prefill_flush_called = false;
    int  prefill_flush_requested = 0;
    int  prefill_flush_written = 0;
    bool prefill_suffix_seen = false;

    // Interleaved cross-attention buffer — rebuilt from ring on each draft call
    // Only holds ctx_window tokens worth of data
    std::vector<float> cross_buf;
    std::vector<float> gpu_restore_staging;

    int cross_ctx;

    // GPU cross-attention ring (nullptr = CPU fallback)
    void * gpu_ring_handle = nullptr;
    uint32_t profile_flags = dflash_profile_flags();
    bool kv_cache_init_attempted = false;
    bool kv_cache_enabled = false;
    bool ring_write_discarded = false;
    int deferred_drafter_kv_tokens = 0;

    int n_draft_last = 0;
    std::vector<int32_t> capture_layers;
    bool target_capture_enabled = true;
    bool gpu_capture_available = false;
    int  invalid_reduced_logits_streak = 0;
    bool invalid_reduced_logits_fail_closed = false;

    bool profile_enabled(uint32_t flags) const {
        return dflash_profile_has(profile_flags, flags);
    }

    void reset_invalid_reduced_logits_state() {
        invalid_reduced_logits_streak = 0;
        invalid_reduced_logits_fail_closed = false;
    }

    void note_valid_reduced_logits_draft() {
        reset_invalid_reduced_logits_state();
    }

    void note_invalid_reduced_logits(
            const char * where,
            int32_t token_raw,
            int row,
            int rows,
            int top_k,
            int committed,
            int cross_len,
            int spec_idx,
            int offset,
            float score) {
        invalid_reduced_logits_streak = common_dflash_invalid_reduced_logits_next_streak(
                invalid_reduced_logits_streak, false);

        LOG_ERR("dflash: invalid reduced-logits token %d in %s at row=%d/%d "
                "(top_k=%d committed=%d cross_len=%d ring_filled=%d ring_write_pos=%d "
                "gpu_ring=%d cpu_ring_valid=%d invalid_reduced_logits_streak=%d score=%g spec=%d offset=%d)\n",
                token_raw, where && where[0] ? where : "draft", row, rows, top_k,
                committed, cross_len, ring_filled, ring_write_pos,
                gpu_ring_handle ? 1 : 0, cpu_ring_valid ? 1 : 0,
                invalid_reduced_logits_streak, (double) score, spec_idx, offset);

        if (common_dflash_invalid_reduced_logits_fail_closed(
                    invalid_reduced_logits_streak,
                    DFLASH_INVALID_REDUCED_LOGITS_FAIL_CLOSED_STREAK)) {
            invalid_reduced_logits_fail_closed = true;
            LOG_ERR("dflash: fail-closed after %d consecutive invalid reduced-logits drafts; "
                    "skipping DFlash drafting until the cross-ring state is reset\n",
                    invalid_reduced_logits_streak);
        }
    }

    void discard_cross_ring(const char * reason) {
        if (reason && reason[0]) {
            LOG_WRN("dflash: discarding cross-ring state: %s\n", reason);
        }

        ring_write_pos = 0;
        ring_filled = 0;
        committed_len = 0;
        cpu_ring_valid = true;
        prefill_flushed = false;
        prefill_flush_called = false;
        prefill_flush_requested = 0;
        prefill_flush_written = 0;
        prefill_suffix_seen = false;
        deferred_drafter_kv_tokens = 0;
        ring_write_discarded = true;
        reset_invalid_reduced_logits_state();
        cross_buf.clear();
        common_dflash_reset_drafter_seq_and_kv_cache(ctx_dft, seq_id, reason);
    }

    bool validate_target_hiddens(const char * where) {
        const int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots != n_target_layers) {
            LOG_WRN("dflash: %s hidden slot count mismatch: got=%d expected=%d\n",
                    where, (int) n_slots, n_target_layers);
            return false;
        }

        for (int i = 0; i < n_slots; ++i) {
            const int64_t h_embd = llama_get_layer_hidden_n_embd(ctx_tgt, i);
            const int64_t h_tok  = llama_get_layer_hidden_n_tokens(ctx_tgt, i);
            if (h_embd != n_embd || h_tok < 0) {
                LOG_WRN("dflash: %s hidden[%d] shape mismatch: embd=%lld expected=%d tokens=%lld\n",
                        where, i, (long long) h_embd, n_embd, (long long) h_tok);
                return false;
            }
        }

        return true;
    }

    int drafter_prefix_window() const {
        const int n_ctx_dft = llama_n_ctx_seq(ctx_dft);
        return std::max(0, n_ctx_dft - block_size);
    }

    void trim_drafter_prefix_window() {
        auto * mem_dft = llama_get_memory(ctx_dft);
        if (!mem_dft) {
            return;
        }

        llama_memory_seq_rm(mem_dft, seq_id, committed_len, -1);

        const int keep = drafter_prefix_window();
        if (keep <= 0 || committed_len <= keep) {
            return;
        }

        const llama_pos trim_before = committed_len - keep;
        const bool trimmed = llama_memory_seq_rm(mem_dft, seq_id, 0, trim_before);
        if (!trimmed) {
            static bool warned = false;
            if (!warned) {
                LOG_WRN("dflash: failed to slide drafter accepted-prefix window to current context (seq=%d keep=%d committed=%d)\n",
                        seq_id, keep, committed_len);
                warned = true;
            }
            return;
        }

        if (common_dflash_debug_logs_enabled()) {
            LOG_DBG("DFLASH_DBG trim_drafter_prefix_window: seq=%d keep=%d trim_before=%d committed_len=%d\n",
                    seq_id, keep, (int) trim_before, committed_len);
        }
    }

    void update_drafter_kv_cache(int n_written) {
        if (!gpu_ring_handle || n_written <= 0) {
            return;
        }

        trim_drafter_prefix_window();

        const int n_update = std::min(n_written, cross_ctx);
        const int n_full_kv_update = std::min(n_update, drafter_prefix_window());
        const int gpu_write_pos = ring_write_pos % cross_ctx;
        const int gpu_filled = std::min(ring_filled, cross_ctx);

        if (n_full_kv_update > 0) {
            const llama_pos start_pos = committed_len - n_full_kv_update;
            const bool full_kv_ok = llama_dflash_target_kv_cache_update_from_ring(
                ctx_dft, gpu_ring_handle,
                gpu_write_pos, gpu_filled,
                n_target_layers, n_embd, n_full_kv_update,
                seq_id, start_pos);
            if (!full_kv_ok) {
                static bool warned_full_kv = false;
                if (!warned_full_kv) {
                    LOG_WRN("dflash: accepted target-hidden full-KV commit failed; clearing drafter KV suffix for seq=%d start=%d\n",
                            seq_id, (int) start_pos);
                    warned_full_kv = true;
                }
                llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, start_pos, -1);
            }
        }

        if (common_dflash_kv_cache_disabled()) {
            return;
        }

        if (!kv_cache_init_attempted) {
            kv_cache_init_attempted = true;
            llama_dflash_kv_cache_set_active_seq(ctx_dft, seq_id);
            kv_cache_enabled = llama_dflash_kv_cache_init(ctx_dft, cross_ctx);
            if (kv_cache_enabled) {
                LOG_INF("dflash: drafter K/V projection cache enabled (%d-token window)\n", cross_ctx);
            } else {
                LOG_WRN("dflash: drafter K/V projection cache unavailable; using full-window K/V projection\n");
            }
        }
        if (!kv_cache_enabled) {
            return;
        }

        const bool profile_copy = profile_enabled(DFLASH_PROFILE_COPY);
        const int64_t t_start = profile_copy ? ggml_time_us() : 0;
        const bool ok = llama_dflash_kv_cache_update_from_ring_seq(ctx_dft, gpu_ring_handle,
                gpu_write_pos, gpu_filled, n_target_layers, n_embd, n_update, seq_id);
        if (profile_copy) {
            LOG_INF("dflash profile: kv_cache_update requested=%d update=%d ok=%d time=%.3f ms ring_pos=%d filled=%d committed=%d\n",
                    n_written, n_update, ok ? 1 : 0, (ggml_time_us() - t_start) / 1e3,
                    gpu_write_pos, gpu_filled, committed_len);
        }

        if (!ok) {
            static bool warned = false;
            if (!warned) {
                LOG_WRN("dflash: drafter K/V projection cache update failed; falling back to full-window projection\n");
                warned = true;
            }
            llama_dflash_kv_cache_reset(ctx_dft);
            kv_cache_enabled = false;
        }
    }

    void defer_drafter_kv_cache_update(int n_written) {
        if (n_written <= 0) {
            return;
        }

        deferred_drafter_kv_tokens = std::min(
                deferred_drafter_kv_tokens + n_written,
                std::max(RING_SIZE, cross_ctx));

        if (profile_enabled(DFLASH_PROFILE_COPY)) {
            LOG_INF("dflash profile: deferred drafter KV update tokens=%d pending=%d committed=%d ring_filled=%d\n",
                    n_written, deferred_drafter_kv_tokens, committed_len, ring_filled);
        }
    }

    void flush_deferred_drafter_kv_cache(const char * where) {
        if (deferred_drafter_kv_tokens <= 0) {
            return;
        }

        const int pending = deferred_drafter_kv_tokens;
        deferred_drafter_kv_tokens = 0;

        const int n_update = std::min(pending, std::max(0, ring_filled));
        if (n_update <= 0) {
            return;
        }

        if (profile_enabled(DFLASH_PROFILE_COPY | DFLASH_PROFILE_SUMMARY)) {
            LOG_INF("dflash profile: flushing deferred drafter KV before %s pending=%d update=%d committed=%d ring_filled=%d\n",
                    where && where[0] ? where : "draft", pending, n_update, committed_len, ring_filled);
        }

        update_drafter_kv_cache(n_update);
    }

    // build interleaved cross-attention data from ring buffer (GPU or CPU path)
    int build_cross_data(llama_context * ctx) {
        if (common_dflash_debug_logs_enabled()) {
            LOG_DBG("DFLASH_DBG build_cross_data: ring_write_pos=%d ring_filled=%d committed_len=%d cross_ctx=%d gpu=%d\n",
                ring_write_pos, ring_filled, committed_len, cross_ctx, gpu_ring_handle ? 1 : 0);
        }
        if (!common_dflash_target_capture_ready_or_skip(ctx_tgt)) {
            return -1;
        }
        if (gpu_ring_handle) {
            int gpu_write_pos = ring_write_pos % cross_ctx;
            int gpu_filled = std::min(ring_filled, cross_ctx);
            llama_dflash_cross_ring_gpu_set_cross(ctx, gpu_ring_handle, seq_id,
                gpu_write_pos, gpu_filled, n_target_layers, n_embd, cross_ctx);
            return gpu_filled;
        }
        if (!cpu_ring_valid) {
            LOG_WRN("dflash: CPU cross ring is stale and GPU ring is unavailable; refusing to build DFlash cross data\n");
            return -1;
        }
        int cross_len = std::min(ring_filled, cross_ctx > 0 ? cross_ctx : ring_filled);
        cross_buf.resize((size_t)n_target_features * cross_len);
        int read_start = (ring_write_pos - cross_len + RING_SIZE) % RING_SIZE;
        for (int t = 0; t < cross_len; ++t) {
            int slot = (read_start + t) % RING_SIZE;
            for (int layer = 0; layer < n_target_layers; ++layer) {
                memcpy(&cross_buf[(size_t)(layer * n_embd) + (size_t)t * n_target_features],
                       ring_buf[layer].data() + (size_t)slot * n_embd,
                       n_embd * sizeof(float));
            }
        }
        llama_set_cross_data_seq(ctx, seq_id, cross_buf.data(), n_target_features, cross_len);
        return cross_len;
    }

    llama_batch batch_dft;

    common_speculative_impl_dflash(
            common_speculative_type type,
            uint32_t n_seq,
            llama_context * ctx_tgt_,
            llama_context * ctx_dft_,
            llama_model   * model_dft_,
            bool            owns_ctx_dft_ = true,
            float           p_min_ = 0.0f,
            int             cross_ctx_ = LLAMA_DFLASH_PER_SLOT_CTX)
        : common_speculative_impl(type, n_seq)
        , ctx_tgt(ctx_tgt_)
        , ctx_dft(ctx_dft_)
        , model_dft(model_dft_)
        , owns_ctx_dft(owns_ctx_dft_)
        , p_min(p_min_)
        , cross_ctx(cross_ctx_ > 0 ? cross_ctx_ : LLAMA_DFLASH_PER_SLOT_CTX)
    {
        block_size        = llama_model_dflash_block_size(model_dft_);
        mask_token_id     = (llama_token) llama_model_dflash_mask_token_id(model_dft_);
        n_target_layers   = llama_model_dflash_n_target_layers(model_dft_);
        n_embd            = llama_model_n_embd(model_dft_);
        n_target_features = llama_model_dflash_n_target_features(model_dft_);

        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        const int target_n_layer = model_tgt ? llama_model_n_layer(model_tgt) : 0;
        const int target_n_embd  = model_tgt ? llama_model_n_embd(model_tgt)  : 0;

        auto fail_contract = [&](const std::string & why) {
            throw std::runtime_error(string_format(
                "dflash: invalid drafter/target contract: %s "
                "(block_size=%d mask_token=%d n_target_layers=%d draft_n_embd=%d "
                "target_n_layer=%d target_n_embd=%d n_target_features=%d)",
                why.c_str(),
                block_size,
                (int) mask_token_id,
                n_target_layers,
                n_embd,
                target_n_layer,
                target_n_embd,
                n_target_features));
        };

        if (!model_tgt) {
            fail_contract("target model is null");
        }
        if (block_size <= 1) {
            fail_contract("block_size must be greater than 1");
        }
        if (mask_token_id < 0) {
            fail_contract("mask_token_id must be non-negative");
        }
        {
            const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
            const llama_token target_mask = vocab_tgt ? llama_vocab_mask(vocab_tgt) : LLAMA_TOKEN_NULL;
            if (target_mask != LLAMA_TOKEN_NULL && mask_token_id != target_mask) {
                fail_contract(string_format(
                    "mask_token_id must match target vocab mask token: drafter=%d target=%d",
                    (int) mask_token_id,
                    (int) target_mask));
            }
        }
        if (n_target_layers <= 0) {
            fail_contract("n_target_layers must be positive");
        }
        if (n_embd <= 0 || target_n_embd <= 0) {
            fail_contract("embedding sizes must be positive");
        }
        if (n_embd != target_n_embd) {
            fail_contract("drafter n_embd must match target hidden size");
        }
        if (n_target_features != n_embd * n_target_layers) {
            fail_contract("n_target_features must equal n_embd * n_target_layers");
        }

        capture_layers.assign(n_target_layers, 0);
        const int n_read = llama_model_dflash_target_layer_ids(model_dft_, capture_layers.data(), n_target_layers);
        if (n_read != n_target_layers) {
            fail_contract(string_format("target_layer_ids read count mismatch: read %d expected %d", n_read, n_target_layers));
        }
        if (!common_dflash_layer_ids_unique(capture_layers)) {
            fail_contract("target_layer_ids contain duplicates");
        }
        for (const int32_t il : capture_layers) {
            if (il < 0 || il >= target_n_layer) {
                fail_contract(string_format("target_layer_id %d is outside target layer range [0,%d)", il, target_n_layer));
            }
        }

        LOG_INF("dflash: contract ok: block_size=%d mask_token=%d target_layer_ids=%s "
                "n_target_layers=%d n_embd=%d n_target_features=%d target_layers=%d cross_ctx=%d\n",
                block_size,
                (int) mask_token_id,
                common_dflash_layer_ids_to_string(capture_layers).c_str(),
                n_target_layers,
                n_embd,
                n_target_features,
                target_n_layer,
                cross_ctx);

        if (profile_enabled(DFLASH_PROFILE_SUMMARY) || common_dflash_log_contract_verbose()) {
            LOG_INF("dflash: contract detail: target_arch=%s draft_arch=%s "
                    "target_n_swa=%d draft_n_swa=%d capture_layers=%s draft_swa_pattern=%s "
                    "target_rope_full=%.1f target_rope_swa=%.1f target_rope_scale_full=%g target_rope_scale_swa=%g "
                    "draft_rope_full=%.1f draft_rope_swa=%.1f draft_rope_scale_full=%g draft_rope_scale_swa=%g\n",
                    llama_model_arch_name(model_tgt),
                    llama_model_arch_name(model_dft_),
                    llama_model_n_swa(model_tgt),
                    llama_model_n_swa(model_dft_),
                    common_dflash_layer_ids_with_attention_to_string(model_tgt, capture_layers).c_str(),
                    common_dflash_swa_pattern_to_string(model_dft_, llama_model_n_layer(model_dft_)).c_str(),
                    (double) llama_model_rope_freq_base_train(model_tgt),
                    (double) llama_model_rope_freq_base_train_swa(model_tgt),
                    (double) llama_model_rope_freq_scale_train(model_tgt),
                    (double) llama_model_rope_freq_scale_train_swa(model_tgt),
                    (double) llama_model_rope_freq_base_train(model_dft_),
                    (double) llama_model_rope_freq_base_train_swa(model_dft_),
                    (double) llama_model_rope_freq_scale_train(model_dft_),
                    (double) llama_model_rope_freq_scale_train_swa(model_dft_));
        }

        {
            int n_capture_swa = 0;
            int n_capture_full = 0;
            int n_capture_unknown = 0;
            const int target_n_swa = llama_model_n_swa(model_tgt);
            const int draft_n_swa  = llama_model_n_swa(model_dft_);
            common_dflash_capture_attention_counts(
                    model_tgt, capture_layers, n_capture_swa, n_capture_full, n_capture_unknown);

            if (n_capture_swa > 0 && n_capture_full > 0) {
                LOG_WRN("dflash: target capture layers mix SWA and FULL attention "
                        "(swa=%d full=%d unknown=%d layers=%s neighborhoods=%s); "
                        "long-context acceptance may be contract-limited\n",
                        n_capture_swa, n_capture_full, n_capture_unknown,
                        common_dflash_layer_ids_with_attention_to_string(model_tgt, capture_layers).c_str(),
                        common_dflash_capture_neighborhood_to_string(
                            model_tgt, capture_layers, target_n_layer).c_str());
            }
            if (target_n_swa > 0 && draft_n_swa > 0 && target_n_swa != draft_n_swa) {
                LOG_WRN("dflash: target/draft SWA window mismatch target_n_swa=%d draft_n_swa=%d; "
                        "treat this DFlash pairing as suspect for long chat until acceptance is verified\n",
                        target_n_swa, draft_n_swa);
            }
        }

        {
            const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
            const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft_);
            const int vocab_tgt_n = vocab_tgt ? llama_vocab_n_tokens(vocab_tgt) : 0;
            const int vocab_dft_n = vocab_dft ? llama_vocab_n_tokens(vocab_dft) : 0;

            int32_t capture_min = capture_layers.empty() ? -1 : capture_layers[0];
            int32_t capture_max = capture_layers.empty() ? -1 : capture_layers[0];
            for (int32_t il : capture_layers) {
                capture_min = std::min(capture_min, il);
                capture_max = std::max(capture_max, il);
            }

            LOG_INF("dflash: target/drafter info: target_ctx_train=%d target_vocab=%d drafter_vocab=%d vocab_match=%d capture_min=%d capture_max=%d\n",
                    llama_model_n_ctx_train(model_tgt),
                    vocab_tgt_n,
                    vocab_dft_n,
                    vocab_tgt_n == vocab_dft_n ? 1 : 0,
                    capture_min,
                    capture_max);

            if (!common_speculative_are_compatible(model_tgt, model_dft_)) {
                throw std::runtime_error(string_format(
                    "dflash: target and drafter vocab are incompatible; DFlash cannot retokenize draft outputs "
                    "(target_vocab=%d drafter_vocab=%d)",
                    vocab_tgt_n, vocab_dft_n));
            }
        }

        ring_buf.resize(n_target_layers);
        for (int i = 0; i < n_target_layers; ++i) {
            ring_buf[i].resize((size_t)RING_SIZE * n_embd, 0.0f);
        }

        llama_set_dflash_capture(ctx_tgt, capture_layers.data(), n_target_layers);
        target_capture_enabled = true;

        batch_dft = llama_batch_init(block_size, 0, 1);

        const bool gpu_ring_allowed = common_dflash_gpu_ring_allowed(ctx_tgt, ctx_dft);
        const bool gpu_ring_forced_off = common_dflash_force_cpu_cross_ring();
        const bool gpu_ring_requested = gpu_ring_allowed && !gpu_ring_forced_off;

        llama_set_dflash_gpu_capture(ctx_tgt, gpu_ring_requested);

        LOG_INF("dflash: GPU hidden capture policy: allowed=%d forced_cpu=%d requested=%d target_devices=%d drafter_devices=%d\n",
                gpu_ring_allowed ? 1 : 0,
                gpu_ring_forced_off ? 1 : 0,
                gpu_ring_requested ? 1 : 0,
                llama_model_n_devices(llama_get_model(ctx_tgt)),
                llama_model_n_devices(model_dft));

        if (gpu_ring_requested) {
            gpu_ring_handle = llama_dflash_cross_ring_gpu_init(ctx_dft, n_target_layers, n_embd, cross_ctx);
        }
        if (gpu_ring_forced_off) {
            LOG_INF("dflash: GPU cross ring forced off by GGML_DFLASH_FORCE_CPU_CROSS; using CPU hidden capture\n");
        }
        if (gpu_ring_handle) {
            LOG_INF("dflash: GPU cross ring enabled (%d layers x %d slots x %d embd)\n",
                    n_target_layers, cross_ctx, n_embd);
            gpu_capture_available = true;
        } else if (gpu_ring_requested) {
            llama_set_dflash_gpu_capture(ctx_tgt, false);
            LOG_WRN("dflash: GPU cross ring unavailable; using CPU hidden capture\n");
        }
    }

    ~common_speculative_impl_dflash() override {
        llama_dflash_cross_ring_gpu_free(gpu_ring_handle);
        llama_batch_free(batch_dft);
        if (owns_ctx_dft) {
            llama_free(ctx_dft);
        }
    }

    void set_seq_id(llama_seq_id seq_id_) override {
        seq_id = seq_id_;
    }

    void set_prefill_capture_enabled(bool enabled) override {
        const bool changed = target_capture_enabled != enabled;
        target_capture_enabled = enabled;

        if (enabled) {
            llama_set_dflash_capture_active(ctx_tgt, true);
            if (changed && profile_enabled(DFLASH_PROFILE_PREFILL)) {
                LOG_INF("dflash prefill capture: enabled hidden capture gpu=%d layers=%d\n",
                        gpu_capture_available ? 1 : 0, (int) capture_layers.size());
            }
        } else {
            llama_set_dflash_capture_active(ctx_tgt, false);
            if (changed && profile_enabled(DFLASH_PROFILE_PREFILL)) {
                LOG_INF("dflash prefill capture: disabled hidden capture for non-suffix prompt chunk\n");
            }
        }
    }

    void discard_dflash_state(const char * reason) override {
        discard_cross_ring(reason);
    }

    void note_prefill_suffix_scheduled() override {
        prefill_suffix_seen = true;
    }

    // prepare cross-attention data for batched draft decode.
    // Returns cross_len (position offset for tokens), or -1 if no committed tokens.
    int prepare_batch_draft(llama_context * ctx_dft_ext) override {
        if (committed_len == 0) {
            return -1;
        }
        if (invalid_reduced_logits_fail_closed) {
            return -1;
        }

        flush_deferred_drafter_kv_cache("batched draft");
        return build_cross_data(ctx_dft_ext);
    }

    // called after initial prefill — extract hidden states from target
    void begin(llama_seq_id /*seq_id*/, const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
        if (prefill_suffix_seen && !prefill_flush_called) {
            LOG_ERR("dflash: prefill suffix was scheduled but flush_prefill() was never called (ring_filled=%d committed_len=%d)\n",
                ring_filled, committed_len);
        }
        if (prefill_suffix_seen && prefill_flush_called && prefill_flush_written <= 0) {
            LOG_ERR("dflash: prefill suffix flush was called but wrote 0/%d tokens (ring_filled=%d committed_len=%d)\n",
                prefill_flush_requested, ring_filled, committed_len);
        }
        if (prefill_suffix_seen && prefill_flush_written > 0 && ring_filled <= 0) {
            LOG_ERR("dflash: prefill suffix flush wrote %d tokens but ring is empty (ring_filled=%d committed_len=%d)\n",
                prefill_flush_written, ring_filled, committed_len);
        }
        if (prefill_flushed) {
            prefill_flushed = false;
            prefill_flush_called = false;
            prefill_flush_requested = 0;
            prefill_flush_written = 0;
            prefill_suffix_seen = false;
            reset_invalid_reduced_logits_state();
            llama_dflash_prefill_capture_end(ctx_tgt);
            return;
        }
        if (!validate_target_hiddens("begin")) {
            return;
        }
        capture_target_hiddens();
    }

    bool process(const llama_batch & /*batch*/) override {
        return true;
    }

    int flush_prefill(int src_offset = 0, int n_tokens = 0) override {
        llama_dflash_set_active_slot(ctx_tgt, seq_id);

        prefill_flush_called = true;
        prefill_flush_requested += n_tokens;

        // In GPU prefill-staging mode the eval callback is intentionally not
        // installed, so CPU hidden buffers may be empty while prefill_gpu has
        // the suffix that must be flushed into the cross ring.
        const bool use_prefill_gpu = llama_dflash_prefill_gpu_active(ctx_tgt);

        if (!use_prefill_gpu && !validate_target_hiddens("flush_prefill")) {
            return 0;
        }

        const int32_t n_slots = use_prefill_gpu
            ? n_target_layers
            : llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0) {
            return 0;
        }

        dflash_capture_source source;
        int64_t captured = 0;
        int32_t n_src_layers = 0;
        int offset = 0;

        if (use_prefill_gpu) {
            source = dflash_capture_source::prefill_gpu_hidden;
            n_src_layers = n_target_layers;

            int32_t planned = 0;
            int32_t written = 0;
            if (llama_dflash_prefill_capture_info(ctx_tgt, seq_id, &planned, &written)) {
                captured = written;
            } else {
                captured = llama_dflash_prefill_gpu_n_tokens(ctx_tgt, seq_id);
            }

            // GPU staging is window-relative.
            offset = 0;

            if (n_tokens > 0 && captured < n_tokens) {
                LOG_ERR("dflash prefill flush incomplete GPU capture: captured=%lld requested=%d seq=%d; refusing partial ring write\n",
                        (long long) captured, n_tokens, seq_id);
                return 0;
            }
        } else {
            n_src_layers = llama_get_n_layer_hiddens(ctx_tgt);
            if (n_src_layers == 0) {
                if (profile_enabled(DFLASH_PROFILE_PREFILL)) {
                    LOG_INF("dflash prefill flush skipped: source=cpu reason=no-layer-slots n_src_layers=0\n");
                }
                return 0;
            }

            float * cpu_hidden0 = llama_get_layer_hidden(ctx_tgt, 0);
            captured = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);
            if (captured <= 0) {
                if (profile_enabled(DFLASH_PROFILE_PREFILL)) {
                    LOG_INF("dflash prefill flush skipped: source=cpu/verify_gpu reason=no-captured-tokens captured=0 n_tokens_arg=%d src_offset=%d\n",
                            n_tokens, src_offset);
                }
                return 0;
            }

            if (common_dflash_should_refuse_large_prefill_fallback_for_test(
                        n_tokens, (int) captured, use_prefill_gpu, gpu_ring_handle != nullptr)) {
                LOG_ERR("dflash prefill flush expected GPU staging for large suffix span but only partial fallback capture is available: captured=%lld requested=%d seq=%d; refusing partial ring write\n",
                        (long long) captured, n_tokens, seq_id);
                return 0;
            }

            if (cpu_hidden0) {
                source = dflash_capture_source::cpu_hidden;
            } else if (gpu_ring_handle) {
                source = dflash_capture_source::verify_gpu_hidden;
            } else {
                LOG_ERR("dflash prefill flush has GPU hidden capture but no GPU ring and no CPU hidden data: requested=%d seq=%d\n",
                        n_tokens, seq_id);
                return 0;
            }

            offset = n_tokens > 0 ? src_offset : 0;
        }

        if (captured <= 0) {
            if (profile_enabled(DFLASH_PROFILE_PREFILL)) {
                LOG_INF("dflash prefill flush skipped: source=%s reason=no-captured-tokens captured=0 n_tokens_arg=%d src_offset=%d\n",
                        source == dflash_capture_source::prefill_gpu_hidden ? "prefill_gpu" : "cpu",
                        n_tokens, src_offset);
            }
            return 0;
        }

        int to_write = n_tokens > 0 ? n_tokens : (int) captured;
        if (offset < 0) {
            offset = 0;
        }
        if (offset + to_write > (int) captured) {
            to_write = std::max(0, (int) captured - offset);
        }
        if (to_write <= 0) {
            if (profile_enabled(DFLASH_PROFILE_PREFILL)) {
                LOG_INF("dflash prefill flush skipped: source=%s reason=empty-clamped-span captured=%lld offset=%d to_write=%d\n",
                        source == dflash_capture_source::prefill_gpu_hidden ? "prefill_gpu" : "cpu",
                        (long long) captured, offset, to_write);
            }
            return 0;
        }

        if (profile_enabled(DFLASH_PROFILE_PREFILL)) {
            const char * capture_source =
                source == dflash_capture_source::prefill_gpu_hidden ? "prefill_gpu" :
                source == dflash_capture_source::verify_gpu_hidden  ? "verify_gpu"  :
                                                                       "callback_cpu";
            const bool writes_cpu_ring = source == dflash_capture_source::cpu_hidden;
            const char * ring_dst =
                gpu_ring_handle
                    ? (writes_cpu_ring ? "cpu_ring+gpu_ring" : "gpu_ring")
                    : (writes_cpu_ring ? "cpu_ring" : "none");

            LOG_INF("dflash prefill flush: capture_source=%s ring_dst=%s captured=%lld offset=%d n_tokens=%d to_write=%d n_src_layers=%d prefill_flushed=%d ring_write_pos=%d ring_filled=%d committed_len=%d\n",
                    capture_source, ring_dst, (long long) captured, offset, n_tokens,
                    to_write, n_src_layers, (int) prefill_flushed, ring_write_pos,
                    ring_filled, committed_len);
        }

        if (!prefill_flushed) {
            ring_write_pos = 0;
            ring_filled = 0;
            committed_len = 0;
            deferred_drafter_kv_tokens = 0;
            common_dflash_reset_drafter_seq_and_kv_cache(ctx_dft, seq_id, "first prefill flush");
        }

        const bool force_cpu_ring_for_flush = source == dflash_capture_source::cpu_hidden;
        const int actual_written = ring_write(to_write, offset, force_cpu_ring_for_flush, source);
        if (actual_written != to_write) {
            LOG_ERR("dflash prefill flush wrote incomplete ring span: requested=%d written=%d seq=%d; discarding DFlash state\n",
                    to_write, actual_written, seq_id);
            discard_cross_ring("incomplete prefill flush");
            return 0;
        }

        committed_len += actual_written;
        update_drafter_kv_cache(actual_written);
        prefill_flushed = true;
        prefill_flush_written += actual_written;
        return actual_written;
    }

    // Ring state serialization for checkpoint persistence.
    // Format: [ring_write_pos:i32][ring_filled:i32][committed_len:i32]
    //         [n_target_layers:i32][n_embd:i32][n_entries:i32]
    //         [layer_0 data: n_entries * n_embd * f32] ...
    // If n_target_layers is negative, data is a compact chronological GPU
    // snapshot normalized to ring slots [0,n_entries).
    size_t ring_state_size() const override {
        if (!cpu_ring_valid && !gpu_ring_handle) {
            return 0;
        }

        int n_entries = cpu_ring_valid
            ? std::min(ring_filled, RING_SIZE)
            : std::min(ring_filled, cross_ctx);
        return 6 * sizeof(int32_t) +
               (size_t)n_entries * n_embd * sizeof(float) * n_target_layers;
    }

    bool ring_state_save(uint8_t * buf, size_t size) const override {
        if (!cpu_ring_valid && !gpu_ring_handle) {
            return false;
        }

        const bool compact_gpu_snapshot = !cpu_ring_valid && gpu_ring_handle;
        int n_entries = compact_gpu_snapshot
            ? std::min(ring_filled, cross_ctx)
            : std::min(ring_filled, RING_SIZE);
        size_t expected = 6 * sizeof(int32_t) +
                          (size_t)n_entries * n_embd * sizeof(float) * n_target_layers;
        if (size < expected) {
            return false;
        }

        int32_t * hdr = (int32_t *) buf;
        hdr[0] = compact_gpu_snapshot ? (n_entries % RING_SIZE) : ring_write_pos;
        hdr[1] = compact_gpu_snapshot ? n_entries : ring_filled;
        hdr[2] = committed_len;
        hdr[3] = compact_gpu_snapshot ? -n_target_layers : n_target_layers;
        hdr[4] = n_embd;
        hdr[5] = n_entries;

        uint8_t * dst = buf + 6 * sizeof(int32_t);
        size_t layer_bytes = (size_t) n_entries * n_embd * sizeof(float);

        if (compact_gpu_snapshot) {
            const int gpu_write_pos = cross_ctx > 0 ? ring_write_pos % cross_ctx : 0;
            const bool ok = llama_dflash_cross_ring_gpu_snapshot(gpu_ring_handle,
                    gpu_write_pos, ring_filled, cross_ctx,
                    (float *) dst, n_entries, n_target_layers, n_embd);
            if (!ok) {
                LOG_WRN("dflash: failed to snapshot GPU cross ring for checkpoint\n");
            } else if (profile_enabled(DFLASH_PROFILE_PREFILL | DFLASH_PROFILE_SUMMARY)) {
                LOG_INF("dflash checkpoint: GPU ring snapshot entries=%d committed=%d cross_ctx=%d\n",
                        n_entries, committed_len, cross_ctx);
            }
            return ok;
        }

        for (int l = 0; l < n_target_layers; ++l) {
            memcpy(dst, ring_buf[l].data(), layer_bytes);
            dst += layer_bytes;
        }

        return true;
    }

    bool ring_state_load(const uint8_t * buf, size_t size) override {
        if (size < 6 * sizeof(int32_t)) {
            return false;
        }

        const int32_t * hdr = (const int32_t *) buf;
        int saved_write_pos = hdr[0];
        int saved_filled    = hdr[1];
        int saved_committed = hdr[2];
        int saved_layers_raw = hdr[3];
        int saved_embd      = hdr[4];
        int saved_entries   = hdr[5];
        const bool compact_gpu_snapshot = saved_layers_raw < 0;
        int saved_layers = compact_gpu_snapshot ? -saved_layers_raw : saved_layers_raw;

        if (saved_layers != n_target_layers || saved_embd != n_embd) {
            LOG_WRN("dflash: ring state mismatch: layers %d/%d, embd %d/%d\n",
                    saved_layers, n_target_layers, saved_embd, n_embd);
            return false;
        }

        if (saved_write_pos < 0 || saved_write_pos >= RING_SIZE ||
                saved_filled < 0 || saved_filled > RING_SIZE ||
                saved_entries < 0 || saved_entries != saved_filled ||
                saved_committed < saved_filled) {
            LOG_WRN("dflash: ring state corrupt: write_pos=%d, filled=%d, committed=%d, entries=%d compact=%d\n",
                    saved_write_pos, saved_filled, saved_committed, saved_entries,
                    compact_gpu_snapshot ? 1 : 0);
            return false;
        }

        size_t layer_bytes = (size_t) saved_entries * n_embd * sizeof(float);
        if (size < 6 * sizeof(int32_t) + layer_bytes * n_target_layers) {
            return false;
        }

        ring_write_pos = saved_write_pos;
        ring_filled = saved_filled;
        committed_len = saved_committed;
        deferred_drafter_kv_tokens = 0;
        common_dflash_reset_drafter_seq_and_kv_cache(ctx_dft, seq_id, "ring state load");

        if (profile_enabled(DFLASH_PROFILE_PREFILL | DFLASH_PROFILE_COPY)) {
            LOG_INF("dflash checkpoint: ring restore write_pos=%d filled=%d committed=%d entries=%d compact=%d gpu=%d\n",
                    saved_write_pos, saved_filled, saved_committed, saved_entries,
                    compact_gpu_snapshot ? 1 : 0, gpu_ring_handle ? 1 : 0);
        }

        const uint8_t * src = buf + 6 * sizeof(int32_t);
        for (int l = 0; l < n_target_layers; ++l) {
            memcpy(ring_buf[l].data(), src, layer_bytes);
            src += layer_bytes;
        }

        if (gpu_ring_handle) {
            int gpu_entries = std::min(ring_filled, cross_ctx);
            const size_t layer_floats = (size_t) gpu_entries * n_embd;
            gpu_restore_staging.resize(layer_floats * n_target_layers);
            for (int l = 0; l < n_target_layers; ++l) {
                float * tmp = gpu_restore_staging.data() + layer_floats * l;
                for (int t = 0; t < gpu_entries; ++t) {
                    int cpu_slot = (ring_write_pos - gpu_entries + t + RING_SIZE) % RING_SIZE;
                    memcpy(tmp + (size_t) t * n_embd,
                           ring_buf[l].data() + (size_t) cpu_slot * n_embd,
                           n_embd * sizeof(float));
                }
                int gpu_pos = ((ring_write_pos - gpu_entries) % cross_ctx + cross_ctx) % cross_ctx;
                llama_dflash_cross_ring_gpu_write(gpu_ring_handle, l, gpu_pos,
                    tmp, gpu_entries, n_embd);
            }
            llama_dflash_cross_ring_gpu_synchronize(gpu_ring_handle);
            update_drafter_kv_cache(gpu_entries);
        }

        prefill_flushed = true;
        cpu_ring_valid = true;

        return true;
    }

    common_dflash_ring_stats dflash_ring_stats() const override {
        common_dflash_ring_stats stats;
        stats.ring_write_pos = ring_write_pos;
        stats.ring_filled    = ring_filled;
        stats.committed_len  = committed_len;
        stats.cross_ctx      = cross_ctx;
        stats.cross_len      = std::min(ring_filled, cross_ctx > 0 ? cross_ctx : ring_filled);
        return stats;
    }

    void draft(common_speculative_draft_params_vec & dparams) override {
        for (llama_seq_id sid = 0; sid < (llama_seq_id) n_seq; ++sid) {
            auto & dp = dparams[sid];
            if (!dp.drafting) {
                continue;
            }

            llama_tokens & result = *dp.result;
            const llama_token id_last = dp.id_last;
            const int32_t n_max_eff = dp.n_max > 0 ? dp.n_max : (block_size - 1);
            const int n_draft = std::min(block_size - 1, n_max_eff);
            if (committed_len == 0) {
                continue;
            }
            if (invalid_reduced_logits_fail_closed) {
                continue;
            }

            const int64_t t0 = ggml_time_us();

            flush_deferred_drafter_kv_cache("flat draft");
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, committed_len, -1);
            common_dflash_align_drafter_seq_or_clear(ctx_dft, seq_id, committed_len, "flat draft");

            int cross_len = build_cross_data(ctx_dft);
            if (cross_len <= 0) {
                continue;
            }

            const int64_t t1 = ggml_time_us();

            // build drafter batch: [id_last, mask, mask, ..., mask]
            // positions stay on the target model's absolute timeline so RoPE
            // tracks the accepted suffix instead of restarting at the window.
            // batch size adapts to n_draft+1 (saves compute when n_max < block_size-1)
            const int batch_len = n_draft + 1;
            const int draft_pos_base = committed_len;
            common_batch_clear(batch_dft);
            common_batch_add(batch_dft, id_last, draft_pos_base, { seq_id }, true);
            for (int i = 1; i < batch_len; ++i) {
                common_batch_add(batch_dft, mask_token_id, draft_pos_base + i, { seq_id }, true);
            }

            const int64_t t2 = ggml_time_us();

            // run drafter forward pass
            int ret = llama_decode(ctx_dft, batch_dft);
            if (ret != 0) {
                LOG_ERR("dflash: drafter decode failed with %d\n", ret);
                continue;
            }

            const int64_t t3 = ggml_time_us();

            // read argmax tokens for positions 1..batch_len-1 (skip position 0 = staged_first)
            {
                int32_t * argmax = llama_get_logits_argmax(ctx_dft);
                float * argmax_probs = llama_get_logits_argmax_probs(ctx_dft);
                const int K_flat = llama_get_logits_argmax_k(ctx_dft);
                const int argmax_rows = llama_get_logits_argmax_n(ctx_dft);
                if (argmax) {
                    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
                    if (!common_dflash_argmax_shape_valid(__func__, argmax_rows, batch_len, K_flat)) {
                        if (dp.draft_log_probs) {
                            dp.draft_log_probs->clear();
                        }
                        result.clear();
                        continue;
                    }

                    // GPU argmax path - only top-k ids/probs are transferred.
                    for (int i = 1; i < batch_len && (int) result.size() < n_draft; ++i) {
                        const auto params = dp;
                        if (argmax_probs && p_min > 0.0f && (int) result.size() >= params.n_min) {
                            float log_prob = argmax_probs[i * K_flat];
                            float log_p_min = logf(p_min);
                            if (log_prob < log_p_min) {
                                LOG_DBG("dflash: early stop at position %d/%d (prob %.3f < p_min %.3f)\n",
                                        i, batch_len, expf(log_prob), p_min);
                                break;
                            }
                        }
                        const int32_t token_raw = argmax[i * K_flat];
                        if (!common_dflash_argmax_token_valid(token_raw, n_vocab)) {
                            const float score = argmax_probs ? argmax_probs[i * K_flat] : std::numeric_limits<float>::quiet_NaN();
                            note_invalid_reduced_logits(__func__, token_raw, i, batch_len, K_flat,
                                    committed_len, cross_len, -1, 0, score);
                            if (dp.draft_log_probs) {
                                dp.draft_log_probs->clear();
                            }
                            result.clear();
                            break;
                        }

                        result.push_back((llama_token) token_raw);
                        if (dp.draft_log_probs && argmax_probs) {
                            dp.draft_log_probs->push_back(argmax_probs[i * K_flat]);
                        }
                    }
                } else {
                    // fallback: CPU argmax over full vocab
                    const int n_vocab_dft = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
                    for (int i = 1; i < batch_len && (int) result.size() < n_draft; ++i) {
                        float * logits = llama_get_logits_ith(ctx_dft, i);
                        if (!logits) {
                            break;
                        }
                        llama_token best = (llama_token)(std::max_element(logits, logits + n_vocab_dft) - logits);
                        result.push_back(best);
                        if (dp.draft_log_probs) {
                            dp.draft_log_probs->push_back(0.0f);
                        }
                    }
                }
            }

            const int64_t t4 = ggml_time_us();

            n_draft_last = (int) result.size();
            if (!result.empty()) {
                note_valid_reduced_logits_draft();
            }

            if (profile_enabled(DFLASH_PROFILE_SUMMARY)) {
                const llama_perf_context_data perf_dft = llama_perf_context(ctx_dft);
                LOG_INF("dflash profile: draft ctx=%d cross_len=%d n_draft=%d produced=%d "
                        "cross=%.3f ms batch=%.3f ms decode=%.3f ms argmax=%.3f ms total=%.3f ms gpu_ring=%d graph_reuse=%d\n",
                        committed_len, cross_len, n_draft, (int) result.size(),
                        (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t4 - t3) / 1e3,
                        (t4 - t0) / 1e3, gpu_ring_handle ? 1 : 0, perf_dft.n_reused);
            } else {
                LOG_DBG("dflash draft breakdown (ctx=%d): concat=%.1fms cross=%.1fms decode=%.1fms argmax=%.1fms total=%.1fms\n",
                        committed_len,
                        (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t4 - t3) / 1e3, (t4 - t0) / 1e3);
            }
        }
    }

    void accept(llama_seq_id /*seq_id*/, uint16_t n_accepted, bool /*is_other*/) override {
        GGML_UNUSED(n_accepted);
    }

    void draft_tree(
            const common_params_speculative & params,
            const llama_tokens              & prompt_tgt,
            llama_token                       id_last,
            int                               tree_budget,
            common_speculative_tree         & tree) override {
        const int n_draft = std::min((int) params.n_max, block_size - 1);
        if (n_draft <= 0 || committed_len == 0) {
            return;
        }
        if (invalid_reduced_logits_fail_closed) {
            return;
        }

        const int64_t t0 = ggml_time_us();

        flush_deferred_drafter_kv_cache("tree draft");
        llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, committed_len, -1);
        common_dflash_align_drafter_seq_or_clear(ctx_dft, seq_id, committed_len, "tree draft");

        int cross_len = build_cross_data(ctx_dft);
        if (cross_len <= 0) {
            return;
        }

        common_batch_clear(batch_dft);
        const int draft_pos_base = committed_len;
        common_batch_add(batch_dft, id_last, draft_pos_base, { seq_id }, true);
        for (int i = 1; i < block_size; ++i) {
            common_batch_add(batch_dft, mask_token_id, draft_pos_base + i, { seq_id }, true);
        }

        int ret = llama_decode(ctx_dft, batch_dft);
        if (ret != 0) {
            LOG_ERR("dflash: drafter decode failed with %d\n", ret);
            return;
        }

        const int64_t t_gpu = ggml_time_us();

        const int draft_horizon = std::min(n_draft, block_size - 1);
        const int K = llama_get_logits_argmax_k(ctx_dft);
        const bool can_branch = K > 1;

        const int n_max_original = params.n_max_base > 0 ? params.n_max_base : params.n_max;
        const int original_horizon = std::min(n_max_original, block_size - 1);

        int branch_budget = 0;
        if (can_branch && tree_budget > original_horizon) {
            const int base_branch_budget = tree_budget - original_horizon;
            const float ratio = (float) base_branch_budget / original_horizon;
            branch_budget = std::min(
                (int) std::round(draft_horizon * ratio),
                std::max(0, tree_budget - draft_horizon));
            branch_budget = std::max(0, branch_budget);
        }

        const int main_path_len = std::min(draft_horizon, std::max(1, tree_budget - branch_budget));
        const int depth_limit = draft_horizon;

        // Use GPU argmax/topk for tree building
        int32_t * argmax = llama_get_logits_argmax(ctx_dft);
        if (!argmax) {
            LOG_ERR("draft_tree: no GPU argmax available\n");
            return;
        }
        float * argmax_probs = llama_get_logits_argmax_probs(ctx_dft);
        const int argmax_rows = llama_get_logits_argmax_n(ctx_dft);
        const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
        if (!common_dflash_argmax_shape_valid(__func__, argmax_rows, depth_limit + 1, K)) {
            return;
        }

        // Build tree using best-first heap expansion with chain-seed backbone
        tree.tokens.clear();
        tree.parents.clear();
        tree.depths.clear();
        tree.child_maps.clear();
        tree.visibility.clear();
        tree.log_probs.clear();

        tree.parents.push_back(-1); // root parent
        tree.child_maps.push_back({}); // root child_map
        tree.n_nodes = 0;
        tree.main_path_len = 0;

        // Chain-seed: pre-insert greedy backbone (top-1 at each depth)
        {
            int parent = 0;
            for (int d = 1; d <= depth_limit && d <= main_path_len && tree.n_nodes < tree_budget; ++d) {
                const int32_t token_raw = argmax[d * K];
                if (!common_dflash_argmax_token_valid(token_raw, n_vocab)) {
                    const float score = argmax_probs ? argmax_probs[d * K] : std::numeric_limits<float>::quiet_NaN();
                    note_invalid_reduced_logits(__func__, token_raw, d, depth_limit, K,
                            committed_len, cross_len, -1, 0, score);
                    break;
                }

                llama_token token_id = (llama_token) token_raw;
                float log_prob = argmax_probs ? argmax_probs[d * K + 0] : -INFINITY;
                int current_idx = tree.n_nodes + 1;
                tree.tokens.push_back(token_id);
                tree.parents.push_back(parent);
                tree.depths.push_back(d);
                tree.log_probs.push_back(log_prob);
                tree.child_maps.push_back({});
                tree.child_maps[parent][token_id] = current_idx;
                tree.n_nodes++;

                parent = current_idx;
            }
            tree.main_path_len = tree.n_nodes;
        }

        // Best-first expansion using log-prob heap (DDTree Algorithm 1)
        if (K > 1 && branch_budget > 0 && argmax_probs) {
            // Heap entry: (cumulative_log_prob, parent_tree_idx, depth, rank)
            struct heap_entry {
                float  log_w;
                int    parent_idx;
                int    depth;  // 1-based position in draft sequence
                int    rank;   // rank within top-K at this depth
                bool operator<(const heap_entry & o) const { return log_w < o.log_w; }
            };

            std::priority_queue<heap_entry> heap;

            // Seed heap: siblings of the main chain at each depth
            float cum_log_prob = 0.0f;
            for (int d = 1; d <= main_path_len; ++d) {
                float sibling_lp = cum_log_prob + argmax_probs[d * K + 1];
                int sibling_parent = (d == 1) ? 0 : d - 1;
                heap.push({sibling_lp, sibling_parent, d, 1});
                cum_log_prob += argmax_probs[d * K + 0];
            }

            while (!heap.empty() && (tree.n_nodes - tree.main_path_len) < branch_budget) {
                auto top = heap.top();
                heap.pop();

                const int32_t token_raw = argmax[top.depth * K + top.rank];
                if (!common_dflash_argmax_token_valid(token_raw, n_vocab)) continue;
                llama_token token_id = (llama_token) token_raw;
                if (tree.child_maps[top.parent_idx].count(token_id)) continue;

                int current_idx = tree.n_nodes + 1;
                tree.tokens.push_back(token_id);
                tree.parents.push_back(top.parent_idx);
                tree.depths.push_back(top.depth);
                tree.log_probs.push_back(argmax_probs[top.depth * K + top.rank]);
                tree.child_maps.push_back({});
                tree.child_maps[top.parent_idx][token_id] = current_idx;
                tree.n_nodes++;

                // Push sibling: same depth, next rank
                if (top.rank + 1 < K) {
                    float sib_lp = top.log_w - argmax_probs[top.depth * K + top.rank]
                                             + argmax_probs[top.depth * K + top.rank + 1];
                    heap.push({sib_lp, top.parent_idx, top.depth, top.rank + 1});
                }

                // Push child: extend this branch one depth deeper (rank 0)
                if (top.depth < depth_limit) {
                    int child_depth = top.depth + 1;
                    float child_lp = top.log_w + argmax_probs[child_depth * K + 0];
                    heap.push({child_lp, current_idx, child_depth, 0});
                }
            }
        } else if (K > 1 && branch_budget > 0) {
            // Fallback without log-probs: uniform sibling addition
            for (int d = 1; d <= main_path_len && (tree.n_nodes - tree.main_path_len) < branch_budget; ++d) {
                int parent_idx = (d == 1) ? 0 : d - 1;
                for (int ki = 1; ki < K && (tree.n_nodes - tree.main_path_len) < branch_budget; ++ki) {
                    const int32_t token_raw = argmax[d * K + ki];
                    if (!common_dflash_argmax_token_valid(token_raw, n_vocab)) continue;
                    llama_token alt_token = (llama_token) token_raw;
                    if (tree.child_maps[parent_idx].count(alt_token)) continue;

                    int current_idx = tree.n_nodes + 1;
                    tree.tokens.push_back(alt_token);
                    tree.parents.push_back(parent_idx);
                    tree.depths.push_back(d);
                    tree.log_probs.push_back(-INFINITY);
                    tree.child_maps.push_back({});
                    tree.child_maps[parent_idx][alt_token] = current_idx;
                    tree.n_nodes++;
                }
            }
        }

        // build visibility matrix [(n_nodes+1) × (n_nodes+1)]
        if (tree.n_nodes > 0) {
            note_valid_reduced_logits_draft();
        }

        int n = tree.n_nodes + 1;
        tree.visibility.assign(n * n, false);
        tree.visibility[0] = true; // root sees itself
        for (int i = 1; i < n; ++i) {
            int parent = tree.parents[i];
            // inherit parent's visibility row
            for (int j = 0; j < i; ++j) {
                tree.visibility[i * n + j] = tree.visibility[parent * n + j];
            }
            tree.visibility[i * n + i] = true; // see itself
        }

        const int64_t t1 = ggml_time_us();
        const int64_t gpu_ms = (t_gpu - t0) / 1000;
        const int64_t cpu_ms = (t1 - t_gpu) / 1000;
        LOG_INF("ddtree: built tree with %d nodes (%d main + %d branch, budget %d) in %.1fms (gpu=%.0fms cpu=%.0fms)\n",
                tree.n_nodes, tree.main_path_len, tree.n_nodes - tree.main_path_len,
                tree_budget, (t1 - t0) / 1e3, (double) gpu_ms, (double) cpu_ms);

        GGML_UNUSED(prompt_tgt);
    }

    // called after target verification decode — capture and append new hidden states
    void update_logits(llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) override {
        GGML_UNUSED(ctx);
        GGML_UNUSED(batch_tokens);
        // In this path n_accepted means committed hidden-state rows, not output-token count.
        // [id_last, draft0, ..., draftN-1] => root + accepted draft tokens.
        // Boundary stops pass root + accepted draft tokens even when no bonus token was sampled.
        append_target_hiddens(n_accepted, false);
    }

    void update_logits_deferred_dflash_kv(llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) override {
        GGML_UNUSED(ctx);
        GGML_UNUSED(batch_tokens);
        append_target_hiddens(n_accepted, true);
    }

    // tree variant: write specific capture-buffer indices to the ring
    void update_logits_by_indices(llama_context * ctx, const std::vector<int> & capture_indices) override {
        GGML_UNUSED(ctx);
        llama_dflash_set_active_slot(ctx_tgt, seq_id);

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0 || capture_indices.empty()) {
            return;
        }

        float * cpu_hidden0 = llama_get_layer_hidden(ctx_tgt, 0);
        if (!cpu_hidden0 && gpu_ring_handle) {
            LOG_ERR("dflash tree update requires indexed GPU hidden write, but only GPU hidden capture is available; disabling DFlash ring for this slot\n");
            ring_write_pos = 0;
            ring_filled = 0;
            committed_len = 0;
            cpu_ring_valid = false;
            common_dflash_reset_drafter_seq_and_kv_cache(ctx_dft, seq_id, "tree update GPU hidden unavailable");
            return;
        }

        const int actual_written = ring_write_by_indices(capture_indices);
        committed_len += actual_written;
        update_drafter_kv_cache(actual_written);
    }

    bool need_embd() const override {
        return false;
    }

private:
    void log_ring_profile(const char * name, int n_tokens, int actual_written,
            int64_t cpu_copy_us, int64_t gpu_enqueue_us, int64_t gpu_sync_us) const {
        if (!profile_enabled(DFLASH_PROFILE_COPY)) {
            return;
        }
        LOG_INF("dflash profile: %s requested=%d written=%d cpu_copy=%.3f ms gpu_enqueue=%.3f ms gpu_sync=%.3f ms ring_filled_before=%d committed_before=%d gpu=%d\n",
                name, n_tokens, actual_written,
                cpu_copy_us / 1e3, gpu_enqueue_us / 1e3, gpu_sync_us / 1e3,
                ring_filled, committed_len, gpu_ring_handle ? 1 : 0);
    }

    // write n_tokens from the capture buffer into the ring, starting at
    // src_offset in the capture buffer. wraps circularly in the ring.
    int ring_write(int n_tokens, int src_offset = 0, bool force_cpu_ring = false,
                   dflash_capture_source source = dflash_capture_source::cpu_hidden) {
        if (n_tokens <= 0) return 0;
        ring_write_discarded = false;

        const bool use_prefill_gpu = source == dflash_capture_source::prefill_gpu_hidden;
        const bool source_has_cpu_hidden = source == dflash_capture_source::cpu_hidden;
        const bool cpu_ring_should_track =
            source_has_cpu_hidden && (force_cpu_ring || !gpu_ring_handle);

        if (!cpu_ring_should_track) {
            cpu_ring_valid = false;
        }

        const int32_t n_src_layers = use_prefill_gpu
            ? n_target_layers
            : llama_get_n_layer_hiddens(ctx_tgt);
        int actual_written = n_tokens;
        bool first_layer = true;
        bool any_layer = false;
        for (int layer = 0; layer < n_target_layers && layer < n_src_layers; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);

            if (use_prefill_gpu && !data) {
                any_layer = true;
                continue;
            }

            if ((!data && !gpu_ring_handle) || ntok <= 0) continue;

            int to_write = std::min(n_tokens, std::max(0, (int) ntok - src_offset));
            if (to_write < n_tokens && common_dflash_debug_logs_enabled()) {
                LOG_WRN("DFLASH_DBG ring_write MISMATCH: requested=%d actual=%d ntok=%lld src_offset=%d ring_write_pos=%d ring_filled=%d committed_len=%d\n",
                        n_tokens, to_write, (long long) ntok, src_offset, ring_write_pos, ring_filled, committed_len);
            }
            if (first_layer) {
                actual_written = to_write;
                first_layer = false;
            } else {
                actual_written = std::min(actual_written, to_write);
            }
            any_layer = true;
        }
        if (!any_layer || actual_written <= 0) return 0;

        bool gpu_upload_queued = false;
        bool gpu_d2d_failed = false;
        int64_t cpu_copy_us = 0;
        int64_t gpu_enqueue_us = 0;
        bool cpu_ring_written_all = cpu_ring_should_track;
        for (int layer = 0; layer < n_target_layers && layer < n_src_layers; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t embd = llama_get_layer_hidden_n_embd(ctx_tgt, layer);

            if (use_prefill_gpu && !data) {
                embd = n_embd;
            } else {
                int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);
                if ((!data && !gpu_ring_handle) || ntok <= 0) continue;
            }

            if (cpu_ring_should_track && data) {
                const bool profile_copy = profile_enabled(DFLASH_PROFILE_COPY);
                const int64_t t_start = profile_copy ? ggml_time_us() : 0;
                for (int t = 0; t < actual_written; ++t) {
                    int slot = (ring_write_pos + t) % RING_SIZE;
                    memcpy(ring_buf[layer].data() + (size_t) slot * embd,
                           data + (size_t) (src_offset + t) * embd,
                           embd * sizeof(float));
                }
                if (profile_copy) {
                    cpu_copy_us += ggml_time_us() - t_start;
                }
            } else if (cpu_ring_should_track) {
                cpu_ring_written_all = false;
            }

            if (gpu_ring_handle) {
                const auto plan = common_dflash_ring_write_plan(cross_ctx, ring_write_pos, actual_written);
                if (plan.n_tokens > 0) {
                    const bool profile_copy = profile_enabled(DFLASH_PROFILE_COPY);
                    const int64_t t_start = profile_copy ? ggml_time_us() : 0;
                    bool used_d2d = false;
                    if (!data) {
                        if (use_prefill_gpu) {
                            used_d2d = llama_dflash_prefill_gpu_write_hidden(
                                gpu_ring_handle, ctx_tgt, seq_id, layer, plan.ring_pos,
                                src_offset + plan.src_token_offset, plan.n_tokens, embd);
                            if (!used_d2d) {
                                used_d2d = llama_dflash_cross_ring_gpu_write_hidden(
                                    gpu_ring_handle, ctx_tgt, layer, plan.ring_pos,
                                    src_offset + plan.src_token_offset, plan.n_tokens, embd);
                            }
                        } else {
                            used_d2d = llama_dflash_cross_ring_gpu_write_hidden(
                                gpu_ring_handle, ctx_tgt, layer, plan.ring_pos,
                                src_offset + plan.src_token_offset, plan.n_tokens, embd);
                            if (!used_d2d) {
                                used_d2d = llama_dflash_prefill_gpu_write_hidden(
                                    gpu_ring_handle, ctx_tgt, seq_id, layer, plan.ring_pos,
                                    src_offset + plan.src_token_offset, plan.n_tokens, embd);
                            }
                        }
                        gpu_d2d_failed = gpu_d2d_failed || !used_d2d;
                    }
                    if (!used_d2d && data) {
                        llama_dflash_cross_ring_gpu_write(gpu_ring_handle, layer, plan.ring_pos,
                            data + (size_t) (src_offset + plan.src_token_offset) * embd,
                            plan.n_tokens, embd);
                    }
                    if (profile_copy) {
                        gpu_enqueue_us += ggml_time_us() - t_start;
                    }
                    gpu_upload_queued = true;
                }
            }
        }
        if (gpu_d2d_failed) {
            discard_cross_ring("GPU hidden D2D ring write failed");
            return 0;
        }
        int64_t gpu_sync_us = 0;
        if (gpu_upload_queued) {
            const bool profile_copy = profile_enabled(DFLASH_PROFILE_COPY);
            const int64_t t_start = profile_copy ? ggml_time_us() : 0;
            llama_dflash_cross_ring_gpu_synchronize(gpu_ring_handle);
            if (profile_copy) {
                gpu_sync_us += ggml_time_us() - t_start;
            }
        }
        log_ring_profile("ring_write", n_tokens, actual_written, cpu_copy_us, gpu_enqueue_us, gpu_sync_us);
        cpu_ring_valid = cpu_ring_should_track ? cpu_ring_written_all : false;
        ring_write_pos = (ring_write_pos + actual_written) % RING_SIZE;
        ring_filled = std::min(ring_filled + actual_written, RING_SIZE);
        return actual_written;
    }

    // tree variant: write specific capture-buffer indices to the ring.
    int ring_write_by_indices(const std::vector<int> & indices) {
        const int n_tokens = (int) indices.size();
        if (n_tokens <= 0) return 0;

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        int actual_written = n_tokens;
        bool first_layer = true;
        bool any_layer = false;
        for (int layer = 0; layer < n_target_layers && layer < n_slots; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);
            if (!data || ntok <= 0) continue;

            int wrote_this_layer = 0;
            for (int t = 0; t < n_tokens; ++t) {
                int src_idx = indices[t];
                if (src_idx < 0 || src_idx >= (int) ntok) {
                    break;
                }
                wrote_this_layer++;
            }
            if (wrote_this_layer < n_tokens && common_dflash_debug_logs_enabled()) {
                LOG_INF("DFLASH_DBG ring_write_by_indices MISMATCH: requested=%d prefix=%d ntok=%lld ring_write_pos=%d ring_filled=%d committed_len=%d\n",
                        n_tokens, wrote_this_layer, (long long) ntok, ring_write_pos, ring_filled, committed_len);
            }
            if (first_layer) {
                actual_written = wrote_this_layer;
                first_layer = false;
            } else {
                actual_written = std::min(actual_written, wrote_this_layer);
            }
            any_layer = true;
        }

        if (!any_layer || actual_written <= 0) return 0;

        bool gpu_upload_queued = false;
        int64_t cpu_copy_us = 0;
        int64_t gpu_enqueue_us = 0;
        for (int layer = 0; layer < n_target_layers && layer < n_slots; ++layer) {
            float * data = llama_get_layer_hidden(ctx_tgt, layer);
            int64_t embd = llama_get_layer_hidden_n_embd(ctx_tgt, layer);
            int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, layer);
            if (!data || ntok <= 0) continue;

            for (int t = 0; t < actual_written; ++t) {
                int src_idx = indices[t];
                int ring_slot = (ring_write_pos + t) % RING_SIZE;
                const bool profile_copy = profile_enabled(DFLASH_PROFILE_COPY);
                const int64_t t_cpu_start = profile_copy ? ggml_time_us() : 0;
                memcpy(ring_buf[layer].data() + (size_t) ring_slot * embd,
                       data + (size_t) src_idx * embd,
                       embd * sizeof(float));
                if (profile_copy) {
                    cpu_copy_us += ggml_time_us() - t_cpu_start;
                }
                if (gpu_ring_handle) {
                    int gpu_pos = (ring_write_pos + t) % cross_ctx;
                    const int64_t t_gpu_start = profile_copy ? ggml_time_us() : 0;
                    llama_dflash_cross_ring_gpu_write(gpu_ring_handle, layer, gpu_pos,
                        data + (size_t) src_idx * embd, 1, embd);
                    if (profile_copy) {
                        gpu_enqueue_us += ggml_time_us() - t_gpu_start;
                    }
                    gpu_upload_queued = true;
                }
            }
        }
        int64_t gpu_sync_us = 0;
        if (gpu_upload_queued) {
            const bool profile_copy = profile_enabled(DFLASH_PROFILE_COPY);
            const int64_t t_start = profile_copy ? ggml_time_us() : 0;
            llama_dflash_cross_ring_gpu_synchronize(gpu_ring_handle);
            if (profile_copy) {
                gpu_sync_us += ggml_time_us() - t_start;
            }
        }
        log_ring_profile("ring_write_by_indices", n_tokens, actual_written, cpu_copy_us, gpu_enqueue_us, gpu_sync_us);
        cpu_ring_valid = true;
        ring_write_pos = (ring_write_pos + actual_written) % RING_SIZE;
        ring_filled = std::min(ring_filled + actual_written, RING_SIZE);
        return actual_written;
    }

    // called after initial prefill — grab all hidden states
    void capture_target_hiddens() {
        if (!common_dflash_target_capture_ready_or_skip(ctx_tgt)) {
            ring_write_pos = 0;
            ring_filled = 0;
            committed_len = 0;
            reset_invalid_reduced_logits_state();
            return;
        }

        llama_dflash_set_active_slot(ctx_tgt, seq_id);

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0) return;

        int64_t n_tokens = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);
        if (n_tokens <= 0) return;

        LOG_DBG("DFLASH_DBG capture_target_hiddens: n_tokens=%lld ring_write_pos=%d ring_filled=%d committed_len=%d\n",
            (long long) n_tokens, ring_write_pos, ring_filled, committed_len);

        int start_offset = std::max(0, (int) n_tokens - RING_SIZE);
        int to_store = (int) n_tokens - start_offset;

        ring_write_pos = 0;
        ring_filled = 0;
        deferred_drafter_kv_tokens = 0;
        reset_invalid_reduced_logits_state();
        common_dflash_reset_drafter_seq_and_kv_cache(ctx_dft, seq_id, "capture target hiddens");
        const int actual_written = ring_write(to_store, start_offset, true);
        if (ring_write_discarded) {
            return;
        }
        committed_len = start_offset + actual_written;
        update_drafter_kv_cache(actual_written);
    }

    // called after each verification decode — append only the accepted tokens' hidden states
    void append_target_hiddens(int n_accepted, bool defer_drafter_kv_cache = false) {
        if (!common_dflash_target_capture_ready_or_skip(ctx_tgt)) {
            return;
        }

        llama_dflash_set_active_slot(ctx_tgt, seq_id);

        int32_t n_slots = llama_get_n_layer_hiddens(ctx_tgt);
        if (n_slots == 0 || n_accepted <= 0) {
            return;
        }

        if (common_dflash_debug_logs_enabled()) {
            int64_t ntok = llama_get_layer_hidden_n_tokens(ctx_tgt, 0);
            int64_t embd = llama_get_layer_hidden_n_embd(ctx_tgt, 0);
            const float * data = llama_get_layer_hidden(ctx_tgt, 0);
            float first5[5] = {}, last5[5] = {};
            if (data && ntok > 0 && embd >= 5) {
                memcpy(first5, data, 5 * sizeof(float));
                memcpy(last5, data + (ntok - 1) * embd + embd - 5, 5 * sizeof(float));
            }
            LOG_DBG("DFLASH_DBG append_target_hiddens: n_accepted=%d ntok=%lld embd=%lld ring_write_pos=%d ring_filled=%d committed_len=%d layer0_first=[%.4f,%.4f,%.4f,%.4f,%.4f] last=[%.4f,%.4f,%.4f,%.4f,%.4f]\n",
                n_accepted, (long long) ntok, (long long) embd, ring_write_pos, ring_filled, committed_len,
                first5[0], first5[1], first5[2], first5[3], first5[4],
                last5[0], last5[1], last5[2], last5[3], last5[4]);
        }

        const int actual_written = ring_write(n_accepted);
        if (ring_write_discarded) {
            return;
        }
        if (actual_written != n_accepted) {
            LOG_ERR("dflash: incomplete target hidden capture: requested=%d written=%d; refusing to advance DFlash ring\n",
                    n_accepted, actual_written);
            discard_cross_ring("incomplete target hidden capture");
            return;
        }
        committed_len += actual_written;
        if (defer_drafter_kv_cache) {
            defer_drafter_kv_cache_update(actual_written);
        } else {
            int n_update = actual_written;
            if (deferred_drafter_kv_tokens > 0) {
                n_update += deferred_drafter_kv_tokens;
                deferred_drafter_kv_tokens = 0;
            }
            update_drafter_kv_cache(std::min(n_update, ring_filled));
        }
    }
};


struct common_speculative {
    common_speculative_draft_params_vec dparams;

    // list of implementations to use and their states
    std::vector<std::unique_ptr<common_speculative_impl>> impls;

    // which implementaion was used for a given seq_id
    std::vector<common_speculative_impl *> impl_last;

    // fork: current implementation (for single-seq mode, used by server per-slot)
    common_speculative_impl * curr_impl = nullptr;
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_impl_ngram_cache create_state_ngram_cache(
        const common_speculative_config & config,
        uint32_t n_seq,
        const std::string & path_static,
        const std::string & path_dynamic) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_impl_ngram_cache state(config.params, n_seq, n_draft, path_static, path_dynamic, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str(const std::vector<common_speculative_type> & types) {
    std::string result;

    for (size_t i = 0; i < types.size(); i++) {
        if (i > 0) {
            result += ",";
        }
        result += common_speculative_type_to_str(types[i]);
    }
    return result;
}

const char * common_speculative_all_types_str() {
    static std::string all_types_str = []() {
        std::vector<common_speculative_type> types;
        types.reserve(COMMON_SPECULATIVE_TYPE_COUNT);
        for (int i = 0; i < COMMON_SPECULATIVE_TYPE_COUNT; i++) {
            types.push_back((common_speculative_type) i);
        }
        return common_speculative_type_name_str(types);
    }();
    return all_types_str.c_str();
}

std::string common_speculative_type_to_str(common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:  return "draft-simple";
        case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:  return "draft-eagle3";
        case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:     return "draft-mtp";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram-simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram-map-k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram-map-k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram-mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram-cache";
        case COMMON_SPECULATIVE_TYPE_SUFFIX:        return "suffix";
        case COMMON_SPECULATIVE_TYPE_COPYSPEC:      return "copyspec";
        case COMMON_SPECULATIVE_TYPE_RECYCLE:       return "recycle";
        case COMMON_SPECULATIVE_TYPE_DFLASH:        return "dflash";
        default:                                    return "unknown";
    }
}

std::vector<common_speculative_type> common_speculative_types_from_names(const std::vector<std::string> & names) {
    std::vector<common_speculative_type> types;
    types.reserve(names.size());

    for (const auto & name : names) {
        auto type = common_speculative_type_from_name_map.find(name);
        if (type != common_speculative_type_from_name_map.end()) {
            if (type->second == COMMON_SPECULATIVE_TYPE_NONE) {
                return std::vector<common_speculative_type> { COMMON_SPECULATIVE_TYPE_NONE };
            }
            types.push_back(type->second);
            continue;
        }
        throw std::invalid_argument("unknown speculative type: " + name);
    }

    return types;
}

common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

static uint32_t common_get_enabled_speculative_configs(const std::vector<common_speculative_type> & configs) {
    uint32_t result = 0;
    for (size_t i = 0; i < configs.size(); i++) {
        result |= (1u << configs[i]);
    }
    return result;
}

int32_t common_speculative_n_max(const common_params_speculative * spec) {
    int32_t n_max = 0;

    for (const auto type : spec->types) {
        switch (type) {
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE:
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3:
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP:
                n_max = std::max(n_max, std::max(0, spec->draft.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:
                n_max = std::max(n_max, (int32_t) spec->ngram_simple.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V:
                n_max = std::max(n_max, (int32_t) spec->ngram_map_k4v.size_m);
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:
                n_max = std::max(n_max, std::max(0, spec->ngram_mod.n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:
                n_max = std::max(n_max, (int32_t) 8);
                break;
            case COMMON_SPECULATIVE_TYPE_SUFFIX:
            case COMMON_SPECULATIVE_TYPE_COPYSPEC:
            case COMMON_SPECULATIVE_TYPE_RECYCLE:
            case COMMON_SPECULATIVE_TYPE_DFLASH:
                n_max = std::max(n_max, std::max(0, spec->n_max));
                break;
            case COMMON_SPECULATIVE_TYPE_NONE:
            case COMMON_SPECULATIVE_TYPE_COUNT:
                break;
        }
    }

    return n_max;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(common_params_speculative & params, uint32_t n_seq) {
    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        uint32_t enabled_configs = common_get_enabled_speculative_configs(params.types);

        bool has_draft_simple = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE));
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3
        bool has_mtp = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_DRAFT_MTP)) && params.draft.ctx_dft != nullptr;

        bool has_ngram_cache   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_CACHE));
        bool has_ngram_simple  = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE));
        bool has_ngram_map_k   = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K));
        bool has_ngram_map_k4v = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V));
        bool has_ngram_mod     = (enabled_configs & (1u << COMMON_SPECULATIVE_TYPE_NGRAM_MOD));

        // when adding a new type - update here the logic above
        static_assert(COMMON_SPECULATIVE_TYPE_COUNT == 13);

        // this list here defines the priority of the speculators
        // the one with highest priority are listed first
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_draft_simple) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3, params));
        }
        if (has_mtp) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT_MTP, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_simple>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_eagle3>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DRAFT_MTP: {
                impls.push_back(std::make_unique<common_speculative_impl_draft_mtp>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_impl_ngram_simple>(
                    /* .params = */ config.params,
                    /* .n_seq  = */ n_seq,
                    /* .state  = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_map_k>(
                            get_common_ngram_map(config.type, config.params.ngram_map_k4v), n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                impls.push_back(
                        std::make_unique<common_speculative_impl_ngram_mod>(config.params, n_seq));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(
                        config, n_seq,
                        params.ngram_cache.lookup_cache_static,
                        params.ngram_cache.lookup_cache_dynamic);
                impls.push_back(std::make_unique<common_speculative_impl_ngram_cache>(state));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s: no implementations specified for speculative decoding\n", __func__);
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr)
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

common_speculative_draft_params & common_speculative_get_draft_params(
        common_speculative * spec,
        llama_seq_id seq_id) {
    GGML_ASSERT(spec);
    GGML_ASSERT(seq_id < (llama_seq_id) spec->dparams.size());

    return spec->dparams[seq_id];
}

void common_speculative_begin(common_speculative * spec, llama_seq_id seq_id, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(seq_id, prompt);
        impl->n_call_begin++;
    }
}

bool common_speculative_process(common_speculative * spec, const llama_batch & batch) {
    bool result = true;

    if (spec == nullptr) {
        return result;
    }

    for (auto & impl : spec->impls) {
        result = result && impl->process(batch);
    }

    return result;
}

bool common_speculative_need_embd(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd()) {
            return true;
        }
    }

    return false;
}

bool common_speculative_need_embd_nextn(common_speculative * spec) {
    if (spec == nullptr) {
        return false;
    }

    for (auto & impl : spec->impls) {
        if (impl->need_embd_nextn()) {
            return true;
        }
    }

    return false;
}

void common_speculative_draft(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    auto & dparams = spec->dparams;

    {
        int n_drafting = 0;

        for (auto & dp : dparams) {
            GGML_ASSERT(!dp.drafting || dp.result->empty());

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            return;
        }
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(dparams);
            impl->n_call_draft++;
        }

        int n_drafting = 0;

        for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
            auto & dp = dparams[seq_id];

            auto & result = *dp.result;

            // a new draft has been sampled
            if (dp.drafting && !result.empty()) {
                dp.drafting = false;

                if (dp.n_max > 0) {
                    if (!result.empty() && (int) result.size() > dp.n_max) {
                        LOG_DBG("%s: truncating draft to %d tokens\n", __func__, dp.n_max);
                        result.resize(dp.n_max);
                    }
                }

                if (!result.empty()) {
                    LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                            common_speculative_type_to_str(impl.get()->type).c_str(), dp.prompt->size(),
                            impl.get()->n_call_draft, result.size());

                    // remember which implementation was used
                    spec->impl_last[seq_id] = impl.get();

                    impl->n_gen_drafts++;
                    impl->n_gen_tokens += result.size();
                }
            }

            if (dp.drafting) {
                n_drafting++;
            }
        }

        if (n_drafting == 0) {
            break;
        }
    }

    // these sequences failed to generate a draft
    for (llama_seq_id seq_id = 0; seq_id < (llama_seq_id) dparams.size(); ++seq_id) {
        auto & dp = dparams[seq_id];

        if (dp.drafting) {
            dp.drafting = false;
        }
    }
}

void common_speculative_accept(common_speculative * spec, llama_seq_id seq_id, uint16_t n_accepted) {
    common_speculative_impl * impl = spec->impl_last[seq_id];

    if (!impl) {
        return;
    }

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(seq_id, n_accepted, false);
        impl->n_call_accept++;
    }

    // accept with the rest of the implementations, using is_other == true
    for (auto & impl_other : spec->impls) {
        if (impl_other.get() != impl) {
            impl_other->accept(seq_id, n_accepted, true);
        }
    }
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %16s: #calls(b,g,a) = %4zu %6zu %6zu, #gen drafts = %6zu, #acc drafts = %5zu, #gen tokens = %6zu, #acc tokens = %5zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}

bool common_dflash_prefill_capture_complete_for_test(int captured, int requested) {
    return requested <= 0 || captured >= requested;
}

bool common_dflash_cpu_ring_valid_after_write_for_test(
        bool was_valid,
        bool force_cpu_ring,
        bool has_gpu_ring,
        bool cpu_ring_written_all) {
    bool cpu_ring_valid = was_valid;
    const bool cpu_ring_should_track = force_cpu_ring || !has_gpu_ring;

    if (!cpu_ring_should_track) {
        cpu_ring_valid = false;
    }

    if (cpu_ring_should_track) {
        cpu_ring_valid = cpu_ring_written_all;
    }

    return cpu_ring_valid;
}

bool common_dflash_should_refuse_large_prefill_fallback_for_test(
        int requested,
        int captured,
        bool use_prefill_gpu,
        bool has_gpu_ring) {
    return !use_prefill_gpu &&
        requested > LLAMA_DFLASH_MAX_VERIFY_TOKENS &&
        has_gpu_ring &&
        captured < requested;
}

bool common_dflash_cpu_ring_valid_after_source_write_for_test(
        bool was_valid,
        int source,
        bool force_cpu_ring,
        bool has_gpu_ring,
        bool cpu_data_all_layers) {
    const bool source_has_cpu_hidden = source == 0;
    const bool cpu_ring_should_track = source_has_cpu_hidden && (force_cpu_ring || !has_gpu_ring);
    bool cpu_ring_valid = was_valid;

    if (!cpu_ring_should_track) {
        cpu_ring_valid = false;
    }

    if (cpu_ring_should_track) {
        cpu_ring_valid = cpu_data_all_layers;
    } else {
        cpu_ring_valid = false;
    }

    return cpu_ring_valid;
}

bool common_dflash_tree_update_requires_cpu_hidden_for_test(
        bool has_cpu_hidden,
        bool has_gpu_ring) {
    return !has_cpu_hidden && has_gpu_ring;
}

int common_dflash_invalid_reduced_logits_next_streak_for_test(
        int  current_streak,
        bool valid_draft) {
    return common_dflash_invalid_reduced_logits_next_streak(current_streak, valid_draft);
}

bool common_dflash_invalid_reduced_logits_fail_closed_for_test(
        int invalid_streak,
        int threshold) {
    return common_dflash_invalid_reduced_logits_fail_closed(invalid_streak, threshold);
}

// ============================================================================
// Fork-specific init and dispatch functions
// ============================================================================

llama_context * common_speculative_create_ctx_dft(const common_params_speculative & params, int dflash_n_slots) {
    if (!params.model_dft) {
        return nullptr;
    }
    llama_context_params cparams_dft = params.cparams_dft;
    cparams_dft.dflash_n_slots = dflash_n_slots;
    cparams_dft.dflash_cross_ctx = params.dflash_cross_ctx;
    llama_context * ctx_dft = llama_init_from_model(params.model_dft, cparams_dft);
    if (ctx_dft == nullptr) {
        LOG_ERR("%s", "failed to create draft context\n");
        return nullptr;
    }
    if (params.draft_topk > 1) {
        llama_set_dflash_topk(ctx_dft, params.draft_topk);
        LOG_INF("dflash: top-K=%d enabled for tree branching\n", params.draft_topk);
    }
    if (params.sample_temp > 0.0f) {
        llama_set_dflash_sample_temp(ctx_dft, params.sample_temp);
    }

    // warmup the draft context
    {
        const llama_vocab * vocab_dft = llama_model_get_vocab(llama_get_model(ctx_dft));

        llama_token bos = llama_vocab_bos(vocab_dft);
        llama_token eos = llama_vocab_eos(vocab_dft);

        llama_token tmp[2];
        int n_tmp = 0;
        if (bos != LLAMA_TOKEN_NULL) { tmp[n_tmp++] = bos; }
        if (eos != LLAMA_TOKEN_NULL) { tmp[n_tmp++] = eos; }
        if (n_tmp == 0) { tmp[n_tmp++] = 0; }

        int ret = llama_decode(ctx_dft, llama_batch_get_one(tmp, n_tmp));
        if (ret != 0) {
            LOG_WRN("%s: draft warmup decode failed: %d (non-fatal)\n", __func__, ret);
        }

        llama_memory_t mem_dft = llama_get_memory(ctx_dft);
        if (mem_dft) {
            llama_memory_clear(mem_dft, true);
        }
        llama_synchronize(ctx_dft);
        llama_perf_context_reset(ctx_dft);

        LOG_INF("%s: draft model warmup complete\n", __func__);
    }

    return ctx_dft;
}

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt,
        llama_context             * ctx_dft_shared) {
    const bool owns_ctx_dft = (ctx_dft_shared == nullptr);
    llama_context * ctx_dft = ctx_dft_shared;
    if (ctx_dft == nullptr && params.model_dft) {
        ctx_dft = common_speculative_create_ctx_dft(params);
    }

    std::vector<common_speculative_config> configs = {};
    {
        bool has_suffix   = params.has_type(COMMON_SPECULATIVE_TYPE_SUFFIX);
        bool has_copyspec = params.has_type(COMMON_SPECULATIVE_TYPE_COPYSPEC);
        bool has_recycle  = params.has_type(COMMON_SPECULATIVE_TYPE_RECYCLE);
        bool has_dflash   = params.has_type(COMMON_SPECULATIVE_TYPE_DFLASH);

        if (has_copyspec) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_COPYSPEC, params));
        }
        if (has_recycle) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_RECYCLE, params));
        }
        if (has_suffix) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_SUFFIX, params));
        }
        if (has_dflash) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DFLASH, params));
        }
    }

    const uint32_t n_seq = 1;
    std::vector<std::unique_ptr<common_speculative_impl>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_INF("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_DFLASH: {
                GGML_ASSERT(ctx_dft != nullptr);
                impls.push_back(std::make_unique<common_speculative_impl_dflash>(
                    config.type, n_seq, ctx_tgt, ctx_dft, params.model_dft,
                    owns_ctx_dft, params.p_min, params.dflash_cross_ctx));
                if (owns_ctx_dft) {
                    ctx_dft = nullptr;
                }
                break;
            }
            case COMMON_SPECULATIVE_TYPE_SUFFIX: {
                impls.push_back(std::make_unique<common_speculative_impl_suffix>(
                    config.type, n_seq,
                    config.params.suffix_max_depth,
                    config.params.n_max,
                    config.params.suffix_spec_factor,
                    config.params.suffix_spec_offset,
                    config.params.suffix_min_prob));
                LOG_INF("%s: suffix tree speculative decoding (max_depth=%d, factor=%.1f, min_prob=%.2f)\n",
                    __func__, config.params.suffix_max_depth,
                    config.params.suffix_spec_factor, config.params.suffix_min_prob);
                break;
            }
            case COMMON_SPECULATIVE_TYPE_COPYSPEC: {
                impls.push_back(std::make_unique<common_speculative_impl_copyspec>(
                    config.type, n_seq, config.params.copyspec_gamma));
                LOG_INF("%s: copyspec speculative decoding (gamma=%d)\n",
                    __func__, config.params.copyspec_gamma);
                break;
            }
            case COMMON_SPECULATIVE_TYPE_RECYCLE: {
                impls.push_back(std::make_unique<common_speculative_impl_recycle>(
                    config.type, n_seq, config.params.recycle_k));
                LOG_INF("%s: token recycling speculative decoding (k=%d)\n",
                    __func__, config.params.recycle_k);
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    // if a model-based drafter exists, tell CopySpec to only fire as primary for long matches
    bool has_model_impl = false;
    for (auto & impl : impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH ||
            impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE ||
            impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_EAGLE3) {
            has_model_impl = true;
            break;
        }
    }
    if (has_model_impl) {
        for (auto & impl : impls) {
            if (impl->type == COMMON_SPECULATIVE_TYPE_COPYSPEC) {
                static_cast<common_speculative_impl_copyspec *>(impl.get())->has_model_drafter = true;
            }
        }
    }

    auto * result = new common_speculative {
        /* .dparams   = */ common_speculative_draft_params_vec(n_seq),
        /* .impls     = */ std::move(impls),
        /* .impl_last = */ std::vector<common_speculative_impl *>(n_seq, nullptr),
        /* .curr_impl = */ nullptr,
    };

    return result;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(0, prompt);
        impl->n_call_begin++;
    }
}

void common_speculative_set_seq_id(common_speculative * spec, llama_seq_id seq_id) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->set_seq_id(seq_id);
    }
}

llama_tokens common_speculative_draft(
        common_speculative              * spec,
        const common_params_speculative & params,
        const llama_tokens              & prompt_tgt,
        llama_token                       id_last,
        std::vector<float>              * draft_log_probs,
        llama_pos                         n_past_override) {
    llama_tokens result;

    if (spec == nullptr) {
        return result;
    }

    spec->curr_impl = nullptr;

    // set up dparams for seq 0
    auto & dp = spec->dparams[0];
    dp.drafting = true;
    dp.n_max    = params.n_max;
    dp.n_min    = params.n_min;
    // M-RoPE: actual positions may exceed text token count due to image spatial dims
    dp.n_past   = n_past_override >= 0 ? n_past_override : (llama_pos)prompt_tgt.size();
    dp.id_last  = id_last;
    dp.prompt   = &prompt_tgt;
    dp.result   = &result;
    dp.draft_log_probs = draft_log_probs;
    if (draft_log_probs) {
        draft_log_probs->clear();
    }

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(spec->dparams);
            impl->n_call_draft++;
        }

        if (!result.empty() && (int) result.size() < params.n_min) {
            LOG_DBG("%s: ignoring small draft: %d < %d\n", __func__, (int) result.size(), params.n_min);
            result.clear();
            if (draft_log_probs) {
                draft_log_probs->clear();
            }
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->n_call_draft, result.size());

            spec->curr_impl = impl.get();
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();

            break;
        }
    }

    dp.drafting = false;
    dp.draft_log_probs = nullptr;

    // try extension impls (e.g. CopySpec appending suffix matches after DFlash draft)
    if (!result.empty()) {
        for (auto & impl : spec->impls) {
            if (impl->type == COMMON_SPECULATIVE_TYPE_COPYSPEC) {
                const size_t pre = result.size();
                auto * cs = static_cast<common_speculative_impl_copyspec *>(impl.get());
                cs->extend(prompt_tgt, id_last, result, params.n_max);
                if (result.size() > pre) {
                    LOG_DBG("%s: extended draft by %zu tokens (%s)\n", __func__,
                        result.size() - pre, common_speculative_type_to_str(impl->type).c_str());
                }
            }
        }
    }

    return result;
}

void common_speculative_draft_batch(
        std::vector<common_speculative *> & specs,
        llama_context                     * ctx_dft,
        const common_params_speculative   & params,
        const std::vector<llama_token>    & id_last_per_spec,
        std::vector<llama_tokens>         & result_per_spec,
        std::vector<std::vector<float>>   * log_probs_per_spec) {
    const int n_specs = (int) specs.size();
    result_per_spec.clear();
    result_per_spec.resize(n_specs);
    if (log_probs_per_spec) {
        log_probs_per_spec->clear();
        log_probs_per_spec->resize(n_specs);
    }

    if (n_specs == 0 || !ctx_dft) {
        return;
    }

    const llama_model * model_dft  = llama_get_model(ctx_dft);
    const int block_size           = llama_model_dflash_block_size(model_dft);
    const int n_draft              = std::min(block_size - 1, params.n_max);
    const int batch_len            = n_draft + 1;
    const llama_token mask_tok     = (llama_token) llama_model_dflash_mask_token_id(model_dft);

    const int64_t t0 = ggml_time_us();

    struct ready_slot {
        common_speculative_impl * impl;
        int           cross_len;
        int           draft_pos_base;
        llama_seq_id  seq_id;
        int           spec_idx;
    };
    std::vector<ready_slot> ready;
    ready.reserve(n_specs);

    // The DFlash projection cache is shared by the drafter context. Put the
    // context into multi-slot mode before prepare_batch_draft() builds per-slot
    // cross data so an active batched draft cannot expose single-slot cache
    // state from a previous request.
    llama_set_dflash_n_slots(ctx_dft, std::max(1, std::min(n_specs, (int) LLAMA_DFLASH_MAX_SLOTS)));

    for (int s = 0; s < n_specs; s++) {
        for (auto & impl : specs[s]->impls) {
            if (impl->type != COMMON_SPECULATIVE_TYPE_DFLASH) {
                continue;
            }

            const int cross_len = impl->prepare_batch_draft(ctx_dft);
            if (cross_len < 0) {
                break;
            }

            auto * dfl = static_cast<common_speculative_impl_dflash *>(impl.get());
            ready.push_back({ impl.get(), cross_len, dfl->committed_len, dfl->seq_id, s });
            break;
        }
    }

    if (ready.empty()) {
        return;
    }

    const int n_ready = (int) ready.size();

    llama_set_dflash_n_slots(ctx_dft, n_ready);

    std::vector<llama_seq_id> ready_seq_ids;
    ready_seq_ids.reserve(n_ready);
    for (const auto & rs : ready) {
        ready_seq_ids.push_back(rs.seq_id);
        llama_memory_seq_rm(llama_get_memory(ctx_dft), rs.seq_id, rs.draft_pos_base, -1);
        common_dflash_align_drafter_seq_or_clear(ctx_dft, rs.seq_id, rs.draft_pos_base, "batched draft");
    }

    if (n_ready > 1) {
        const int ctx_window = params.dflash_cross_ctx > 0 ? params.dflash_cross_ctx : (int) LLAMA_DFLASH_PER_SLOT_CTX;
        if (!llama_dflash_kv_cache_prepare_batch(ctx_dft, ready_seq_ids.data(), n_ready, ctx_window)) {
            static bool warned_batch_kv = false;
            if (!warned_batch_kv) {
                LOG_WRN("dflash batch: batched K/V projection cache unavailable; falling back to per-slot cached drafting\n");
                warned_batch_kv = true;
            }
            llama_set_dflash_n_slots(ctx_dft, 1);
            return;
        }
        static bool logged_batch_kv = false;
        if (!logged_batch_kv) {
            LOG_INF("dflash: shared batched K/V projection cache enabled (%d slots, %d-token window)\n",
                    n_ready, ctx_window);
            logged_batch_kv = true;
        }
    }

    const int64_t t1 = ggml_time_us();

    llama_batch batch = llama_batch_init(n_ready * batch_len, 0, 1);

    for (const auto & rs : ready) {
        common_batch_add(batch, id_last_per_spec[rs.spec_idx], rs.draft_pos_base, { rs.seq_id }, true);
        for (int i = 1; i < batch_len; i++) {
            common_batch_add(batch, mask_tok, rs.draft_pos_base + i, { rs.seq_id }, true);
        }
    }

    const int ret = llama_decode(ctx_dft, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        LOG_ERR("dflash batch: decode failed with %d\n", ret);
        return;
    }

    const int64_t t2 = ggml_time_us();

    int32_t * argmax       = llama_get_logits_argmax(ctx_dft);
    float   * argmax_probs = llama_get_logits_argmax_probs(ctx_dft);
    const int K_flat       = llama_get_logits_argmax_k(ctx_dft);
    const int argmax_rows  = llama_get_logits_argmax_n(ctx_dft);

    for (int r = 0; r < n_ready; r++) {
        auto & rs     = ready[r];
        auto & result = result_per_spec[rs.spec_idx];
        std::vector<float> * log_probs = log_probs_per_spec ? &(*log_probs_per_spec)[rs.spec_idx] : nullptr;
        const int offset = r * batch_len;

        if (argmax) {
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
            if (!common_dflash_argmax_shape_valid(__func__, argmax_rows, n_ready * batch_len, K_flat)) {
                if (log_probs) {
                    log_probs->clear();
                }
                result.clear();
                return;
            }

            for (int i = 1; i < batch_len && (int) result.size() < n_draft; i++) {
                if (argmax_probs && params.p_min > 0.0f && (int) result.size() >= params.n_min) {
                    float log_prob = argmax_probs[(offset + i) * K_flat];
                    if (log_prob < logf(params.p_min)) {
                        break;
                    }
                }
                const int32_t token_raw = argmax[(offset + i) * K_flat];
                if (!common_dflash_argmax_token_valid(token_raw, n_vocab)) {
                    const float score = argmax_probs ? argmax_probs[(offset + i) * K_flat] : std::numeric_limits<float>::quiet_NaN();
                    auto * dfl = static_cast<common_speculative_impl_dflash *>(rs.impl);
                    dfl->note_invalid_reduced_logits(__func__, token_raw, i, batch_len, K_flat,
                            rs.draft_pos_base, rs.cross_len, rs.spec_idx, offset, score);
                    if (log_probs) {
                        log_probs->clear();
                    }
                    result.clear();
                    break;
                }

                result.push_back((llama_token) token_raw);
                if (log_probs && argmax_probs) {
                    log_probs->push_back(argmax_probs[(offset + i) * K_flat]);
                }
            }
        } else {
            const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model_dft));
            for (int i = 1; i < batch_len && (int) result.size() < n_draft; i++) {
                float * logits = llama_get_logits_ith(ctx_dft, offset + i);
                if (!logits) {
                    break;
                }
                llama_token best = (llama_token)(std::max_element(logits, logits + n_vocab) - logits);
                result.push_back(best);
                if (log_probs) {
                    log_probs->push_back(0.0f);
                }
            }
        }

        rs.impl->n_call_draft++;
        auto * dfl = static_cast<common_speculative_impl_dflash *>(rs.impl);
        dfl->n_draft_last = (int) result.size();
        if (!result.empty() && (int) result.size() < params.n_min) {
            result.clear();
            if (log_probs) {
                log_probs->clear();
            }
        }
        if (!result.empty()) {
            dfl->note_valid_reduced_logits_draft();
            rs.impl->n_gen_drafts++;
            rs.impl->n_gen_tokens += result.size();
            specs[rs.spec_idx]->curr_impl = rs.impl;
        }
    }

    const int64_t t3 = ggml_time_us();

    LOG_DBG("dflash batch draft (%d specs): prepare=%.1fms decode=%.1fms argmax=%.1fms total=%.1fms\n",
            n_ready, (t1 - t0) / 1e3, (t2 - t1) / 1e3, (t3 - t2) / 1e3, (t3 - t0) / 1e3);
}

common_speculative_tree common_speculative_draft_tree(
        common_speculative              * spec,
        const common_params_speculative & params,
        const llama_tokens              & prompt_tgt,
        llama_token                       id_last,
        int                               tree_budget) {
    common_speculative_tree tree;

    if (spec == nullptr) {
        return tree;
    }

    spec->curr_impl = nullptr;

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft_tree(params, prompt_tgt, id_last, tree_budget, tree);
            impl->n_call_draft++;
        }

        if (tree.n_nodes > 0) {
            spec->curr_impl = impl.get();
            impl->n_gen_drafts++;
            impl->n_gen_tokens += tree.n_nodes;
            break;
        }
    }

    return tree;
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (spec == nullptr) {
        return;
    }

    common_speculative_impl * impl = spec->curr_impl;
    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }
        impl->accept(0, n_accepted, false);
        impl->n_call_accept++;
    }
}

void common_speculative_update_logits(common_speculative * spec, llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->update_logits(ctx, batch_tokens, n_accepted);
        if (impl->type == COMMON_SPECULATIVE_TYPE_COPYSPEC) {
            static_cast<common_speculative_impl_copyspec *>(impl.get())->update_logits(ctx, batch_tokens, n_accepted);
        } else if (impl->type == COMMON_SPECULATIVE_TYPE_RECYCLE) {
            static_cast<common_speculative_impl_recycle *>(impl.get())->update_logits(ctx, batch_tokens, n_accepted);
        }
    }
}

void common_speculative_update_logits_deferred_dflash_kv(common_speculative * spec, llama_context * ctx, const llama_tokens & batch_tokens, int n_accepted) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->update_logits_deferred_dflash_kv(ctx, batch_tokens, n_accepted);
        if (impl->type == COMMON_SPECULATIVE_TYPE_COPYSPEC) {
            static_cast<common_speculative_impl_copyspec *>(impl.get())->update_logits(ctx, batch_tokens, n_accepted);
        } else if (impl->type == COMMON_SPECULATIVE_TYPE_RECYCLE) {
            static_cast<common_speculative_impl_recycle *>(impl.get())->update_logits(ctx, batch_tokens, n_accepted);
        }
    }
}

void common_speculative_update_logits_by_indices(common_speculative * spec, llama_context * ctx, const std::vector<int> & capture_indices) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->update_logits_by_indices(ctx, capture_indices);
    }
}

void common_speculative_rollback_dft(common_speculative * spec, llama_seq_id seq_id, llama_pos n_past, uint16_t n_accepted) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DRAFT_MTP) {
            auto * mtp = static_cast<common_speculative_impl_draft_mtp *>(impl.get());
            auto * ctx_dft = mtp->params.ctx_dft;
            llama_memory_seq_rm(llama_get_memory(ctx_dft), seq_id, n_past, -1);
            mtp->accept(seq_id, n_accepted, false);
        }
    }
}

int common_speculative_flush_prefill(common_speculative * spec, int src_offset, int n_tokens) {
    if (spec == nullptr) {
        return 0;
    }
    int total_written = 0;
    for (auto & impl : spec->impls) {
        total_written += impl->flush_prefill(src_offset, n_tokens);
    }
    return total_written;
}

void common_speculative_set_prefill_capture_enabled(common_speculative * spec, bool enabled) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->set_prefill_capture_enabled(enabled);
    }
}

void common_speculative_discard_dflash_state(common_speculative * spec, const char * reason) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->discard_dflash_state(reason);
    }
}

void common_speculative_note_prefill_suffix_scheduled(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->note_prefill_suffix_scheduled();
    }
}

size_t common_speculative_ring_state_size(const common_speculative * spec) {
    if (spec == nullptr) return 0;
    size_t total = 0;
    for (auto & impl : spec->impls) {
        if (impl->type == COMMON_SPECULATIVE_TYPE_DFLASH) {
            total += static_cast<const common_speculative_impl_dflash *>(impl.get())->ring_state_size();
        }
    }
    return total;
}

bool common_speculative_ring_state_save(const common_speculative * spec, uint8_t * buf, size_t size) {
    if (spec == nullptr) return false;
    bool saved_any = false;
    for (auto & impl : spec->impls) {
        size_t impl_size = impl->ring_state_size();
        if (impl_size > 0 && impl_size <= size) {
            if (!impl->ring_state_save(buf, impl_size)) {
                return false;
            }
            saved_any = true;
            buf += impl_size;
            size -= impl_size;
        }
    }
    return saved_any;
}

bool common_speculative_ring_state_load(common_speculative * spec, const uint8_t * buf, size_t size) {
    if (spec == nullptr) return false;
    for (auto & impl : spec->impls) {
        if (impl->ring_state_load(buf, size)) {
            return true;
        }
    }
    return false;
}

common_dflash_ring_stats common_speculative_dflash_ring_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return {};
    }
    for (const auto & impl : spec->impls) {
        const common_dflash_ring_stats stats = impl->dflash_ring_stats();
        if (stats.cross_ctx > 0 || stats.committed_len > 0 || stats.ring_filled > 0) {
            return stats;
        }
    }
    return {};
}

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }
    GGML_UNUSED(params);
    return params.n_max;
}

int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }
    return params.n_min;
}
