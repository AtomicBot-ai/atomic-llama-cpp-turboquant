// laya engine: tools/laya (own GGUF loader + ggml graph, CPU only) behind decision_engine.
//
// Plans:
//   sequential (default) - one graph per item; no state between calls, so results
//                          are bitwise stable at a fixed thread count and items are independent
//   packed               - items in one graph with a block-diagonal mask (opt-in); never used
//                          for independent items (router scores), and packed in groups of at
//                          most LAYA_PACKED_MAX_TOKENS tokens
//
// The state is tokenized once per request: items that share a state text (all
// questions of a systemone request, the criterion and task of router candidates) reuse its tokens.
// Router pieces are tokenized separately only when the init probe shows the split is exact.

#include "decision.h"
#include "decision-spec.h"
#include "laya.h"
#include "laya-decide.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

static_assert(LAYA_MAX_MARKERS == DECISION_LAYA_MAX_OPTIONS, "decision-spec.h must know the laya option limit");

// packed graphs hold an n_tok x n_tok x n_head KQ matrix and two n_tok x n_tok
// masks: 2048 tokens are ~200 MB + 34 MB for mmBERT-base (12 heads), about four
// times one sequential pass at max_len 1024
#define LAYA_PACKED_MAX_TOKENS 2048

namespace {

struct engine_laya : decision_engine {
    laya_model *   model = nullptr;
    laya_context * ctx   = nullptr;
    laya_escape    esc;
    std::string    plan;
    int32_t        n_threads = 1;
    bool           split_exact = true; // tokenize(A + B) == tokenize(A) + tokenize(B) at the router split points (probed at init)

    // escaped state text -> tokens, for one evaluate call
    using token_cache = std::unordered_map<std::string, std::vector<int32_t>>;

    ~engine_laya() override {
        laya_free(ctx);
        if (model) {
            laya_model_free(model);
        }
    }

    decision_caps caps() const override {
        const laya_hparams & hp = laya_model_hparams(model);
        decision_caps c;
        c.layout           = "laya";
        c.format           = "laya-v1";
        c.plan             = plan;
        c.plan_independent = "sequential";
        c.plans            = { "sequential", "packed" };
        c.question_types   = { "noul", "choice", "score" };
        c.max_options      = LAYA_MAX_MARKERS;
        c.max_tokens       = hp.max_len > 0 ? hp.max_len : 1024;
        c.n_threads        = n_threads;
        c.device           = "cpu";
        c.state_split      = split_exact;
        return c;
    }

    int32_t n_tokens(const std::string & text, bool escape_control) const override {
        return (int32_t) laya_tokenize(model, laya_escape_text(esc, text, escape_control)).size();
    }

    std::vector<int32_t> tokenize(const std::string & text, bool escape_control, token_cache * cache) const {
        std::string escaped = laya_escape_text(esc, text, escape_control);
        if (!cache) {
            return laya_tokenize(model, escaped);
        }
        auto it = cache->find(escaped);
        if (it == cache->end()) {
            std::vector<int32_t> ids = laya_tokenize(model, escaped);
            it = cache->emplace(std::move(escaped), std::move(ids)).first;
        }
        return it->second;
    }

    // state tokens; a split state (router) is the concatenation of its pieces' tokens
    std::vector<int32_t> state_tokens(const decision_item & item, token_cache * cache, int32_t & n_head) const {
        const std::string text = laya_serialize_state(item.state);
        n_head = -1;
        if (!item.state.is_string() || item.state_splits.empty() || !split_exact) {
            return tokenize(text, item.escape_control, cache);
        }
        std::vector<int32_t> ids;
        size_t start = 0;
        for (size_t k = 0; k <= item.state_splits.size(); ++k) {
            const size_t end = k < item.state_splits.size() ? std::min(item.state_splits[k], text.size()) : text.size();
            if (k == item.state_splits.size()) {
                n_head = (int32_t) ids.size();
            }
            if (end > start) {
                const std::vector<int32_t> piece = tokenize(text.substr(start, end - start), item.escape_control, cache);
                ids.insert(ids.end(), piece.begin(), piece.end());
                start = end;
            }
        }
        return ids;
    }

    laya_seq build(const decision_item & item, token_cache * cache, int32_t & n_head) const {
        const std::string param = item.param.empty() ? "questions." + item.q.id + ".criteria" : item.param;
        std::vector<std::string> options = item.q.options;
        if (options.empty()) {
            // a question normalized by the TypeSafe rules
            const json q = { {"type", decision_qtype_name(item.q.type)}, {"criteria", item.q.criteria} };
            try {
                options = laya_render_options(q);
            } catch (const std::exception & e) {
                throw decision_error(DECISION_REASON_UNSUPPORTED_CRITERIA_VALUE, e.what(), param);
            }
        }
        if (options.size() != item.q.keys.size()) {
            throw decision_error(DECISION_REASON_INTERNAL, "option count mismatch for question '" + item.q.id + "'");
        }
        if ((int32_t) options.size() > LAYA_MAX_MARKERS) {
            throw decision_error(DECISION_REASON_TOO_MANY_OPTIONS,
                                 "at most " + std::to_string(LAYA_MAX_MARKERS) + " options are supported", param);
        }
        laya_seq seq = laya_build_sequence(model, (int32_t) item.q.type, item.q.instructions, options,
                                           state_tokens(item, cache, n_head), item.state.is_array());
        if ((int32_t) seq.markers.size() < seq.n_options) {
            throw decision_error(DECISION_REASON_OPTIONS_TRUNCATED,
                                 "options of question '" + item.q.id + "' do not fit in the model input", param);
        }
        return seq;
    }

    static decision_output describe(const laya_seq & seq, int32_t n_head) {
        decision_output out;
        out.n_tokens     = (int32_t) seq.ids.size();
        out.n_state      = seq.n_state;
        out.n_state_cut  = seq.n_state_cut;
        out.n_state_head = n_head;
        out.options_cut  = seq.options_cut;
        out.head_cut     = seq.head_cut;
        out.tokens       = seq.ids;
        return out;
    }

    decision_output render(const decision_item & item) const override {
        int32_t n_head = -1;
        const laya_seq seq = build(item, nullptr, n_head);
        return describe(seq, n_head);
    }

    void encode(const std::vector<const laya_seq *> & seqs, laya_result & res) {
        laya_batch_data data;
        laya_batch_pack(seqs, data);
        if (laya_encode(ctx, data.batch, res) != 0) {
            throw decision_error(DECISION_REASON_INTERNAL, "laya forward pass failed");
        }
    }

    static void read_logits(const laya_seq & seq, const laya_result & res, int32_t idx, decision_output & out) {
        out.logits.resize(seq.markers.size());
        for (size_t j = 0; j < seq.markers.size(); ++j) {
            out.logits[j] = res.logits[(size_t) idx * res.n_markers_max + j];
        }
        out.n_evaluated = out.n_tokens;
    }

    static bool cancelled(const std::atomic<bool> * cancel) {
        return cancel && cancel->load();
    }

    bool evaluate(
            const std::vector<decision_item> & items,
            bool independent,
            const std::atomic<bool> * cancel,
            const decision_render_check & check,
            std::vector<decision_output> & out) override {
        std::vector<laya_seq> seqs;
        seqs.reserve(items.size());
        out.clear();
        {
            token_cache cache;
            for (size_t i = 0; i < items.size(); ++i) {
                if (cancelled(cancel)) {
                    return false;
                }
                int32_t n_head = -1;
                seqs.push_back(build(items[i], &cache, n_head));
                out.push_back(describe(seqs.back(), n_head));
                if (check) {
                    check(i, out.back());
                }
            }
        }

        laya_result res;
        if (plan == "packed" && !independent) {
            // greedy groups in item order; an item longer than the budget runs alone
            size_t i = 0;
            while (i < seqs.size()) {
                if (cancelled(cancel)) {
                    return false;
                }
                const size_t first = i;
                size_t n_tok = 0;
                std::vector<const laya_seq *> group;
                while (i < seqs.size() && (group.empty() || n_tok + seqs[i].ids.size() <= LAYA_PACKED_MAX_TOKENS)) {
                    n_tok += seqs[i].ids.size();
                    group.push_back(&seqs[i++]);
                }
                encode(group, res);
                for (size_t k = 0; k < group.size(); ++k) {
                    read_logits(seqs[first + k], res, (int32_t) k, out[first + k]);
                }
            }
            return true;
        }

        for (size_t i = 0; i < seqs.size(); ++i) {
            if (cancelled(cancel)) {
                return false;
            }
            encode({ &seqs[i] }, res);
            read_logits(seqs[i], res, 0, out[i]);
        }
        return true;
    }
};

// Router states split after the card and after "task:" (decision_router_items), and
// each piece is tokenized on its own. That is exact only when both boundaries are
// token boundaries for this vocabulary: "\n\n" an added token, " " a Metaspace word
// start. Probe it on card endings and task starts of every kind; any difference turns
// splitting off, so router states are then tokenized whole (same tokens, more work).
static bool laya_probe_split(const laya_model * model, const laya_escape & esc) {
    const char * card_ends[] = {
        "executor: A\nkind: local\nchecks: none",
        "- field extraction: passed 188 of 200; criterion: exact match; source: atomic-evals/extract 2026-09",
        "- long-document QA: not measured; source: s v1",
        "description: 4B model (Q4_K_M)", "kind: api.", "kind: \xe6\x9c\xac\xe5\x9c\xb0", "kind: caf\xc3\xa9",
        "kind: x<eos>", "kind: 42", "kind: a-", "kind: a:", "kind: a\xe2\x96\x81",
    };
    const char * criteria[] = { "All three fields exact; nothing invented.", "x", "\xe6\xad\xa3\xe7\xa1\xae" };
    const char * tasks[] = {
        "Extract invoice number, date and total", "x", " leading space", "\nnewline first", "\n\nnewlines",
        "\ttab", "42 items", "\xe6\x8f\x90\xe5\x8f\x96\xe5\x8f\x91\xe7\xa5\xa8", "<mask> masked", "<eos>", "!?", "\xe2\x96\x81meta",
        "caf\xc3\xa9 au lait", "https://example.com/a?b=c", "",
    };
    for (bool escape_control : { false, true }) {
        const auto tok = [&](const std::string & t) { return laya_tokenize(model, laya_escape_text(esc, t, escape_control)); };
        const auto exact = [&](const std::string & a, const std::string & b) {
            std::vector<int32_t> ab = tok(a);
            const std::vector<int32_t> bb = tok(b);
            ab.insert(ab.end(), bb.begin(), bb.end());
            return tok(a + b) == ab;
        };
        for (const char * card : card_ends) {
            for (const char * crit : criteria) {
                const std::string head = std::string("\n\nsuccess criterion: ") + crit + "\n\ntask:";
                if (!exact(card, head)) {
                    return false;
                }
                for (const char * task : tasks) {
                    if (!exact(card + head, std::string(" ") + task)) {
                        return false;
                    }
                }
            }
        }
    }
    return true;
}

} // namespace

std::unique_ptr<decision_engine> decision_engine_laya_init(const decision_spec & spec, const decision_engine_params & params) {
    std::string plan = params.plan;
    if (plan.empty() && spec.plan.contains("name") && spec.plan.at("name").is_string()) {
        plan = spec.plan.at("name").get<std::string>();
    }
    if (plan.empty()) {
        plan = "sequential";
    }
    if (plan != "sequential" && plan != "packed") {
        throw std::runtime_error("laya: unknown decision plan '" + plan + "' (sequential, packed)");
    }

    std::unique_ptr<engine_laya> eng(new engine_laya());
    eng->plan      = plan;
    eng->n_threads = std::max(1, params.n_threads);
    eng->model     = laya_model_load_from_file(params.model_path.c_str());
    eng->esc       = laya_escape_init(eng->model);
    eng->split_exact = laya_probe_split(eng->model, eng->esc);
    eng->ctx       = laya_init(eng->model, eng->n_threads);
    return eng;
}
