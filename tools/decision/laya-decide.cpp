#include "laya-decide.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <unordered_map>

const char * const LAYA_QTYPE_NAMES[3] = { "choice", "score", "noul" };

int32_t laya_qtype_from_name(const std::string & name) {
    for (int32_t i = 0; i < 3; ++i) {
        if (name == LAYA_QTYPE_NAMES[i]) {
            return i;
        }
    }
    return -1;
}

std::string laya_serialize_state(const json & state) {
    if (state.is_string()) {
        return state.get<std::string>();
    }
    return decision_py_dumps(state);
}

std::string laya_render_criterion(const json & value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    return decision_py_dumps(value);
}

// "raw_options": true requests the legacy plain-option protocol: render the
// bare option strings (descriptions only; literal false/true for noul) instead
// of the "key: desc" / "level i: desc" / "false: ..." form. This matches the
// reference used by the Julia family of models.
std::vector<std::string> laya_render_options(const json & q) {
    std::vector<std::string> opts;
    const std::string t = q.at("type").get<std::string>();
    const bool raw = q.contains("raw_options") && q.at("raw_options").get<bool>();
    json crit = q.contains("criteria") ? q.at("criteria") : json::object();
    if (t == "choice") {
        if (crit.is_array()) {
            // list criteria name the options: dict.fromkeys(str(c) for c in criteria)
            json keys = json::object();
            for (const auto & c : crit) {
                std::string k;
                if (!decision_py_str(c, k)) {
                    throw std::runtime_error("choice criteria list entries must be scalars");
                }
                if (!keys.contains(k)) {
                    keys[k] = nullptr;
                }
            }
            crit = std::move(keys);
        }
        for (auto it = crit.begin(); it != crit.end(); ++it) {
            const std::string k = it.key();
            const std::string v = laya_render_criterion(it.value());
            if (raw) {
                opts.push_back(v.empty() || it.value().is_null() ? k : v);
            } else {
                opts.push_back(v.empty() || it.value().is_null() ? k : k + ": " + v);
            }
        }
    } else if (t == "score") {
        int32_t i = 0;
        for (auto it = crit.begin(); it != crit.end(); ++it, ++i) {
            opts.push_back(raw ? laya_render_criterion(it.value())
                               : "level " + std::to_string(i) + ": " + laya_render_criterion(it.value()));
        }
    } else { // noul
        if (raw) {
            opts.push_back("false");
            opts.push_back("true");
        } else {
            const json fc = crit.contains("false") ? crit.at("false") : json();
            const json tc = crit.contains("true")  ? crit.at("true")  : json();
            const std::string fcs = fc.is_null() ? "" : laya_render_criterion(fc);
            const std::string tcs = tc.is_null() ? "" : laya_render_criterion(tc);
            opts.push_back("false: " + (fcs.empty() ? "no, the statement does not hold" : fcs));
            opts.push_back("true: "  + (tcs.empty() ? "yes, the statement holds"          : tcs));
        }
    }
    return opts;
}

// ---- reference question semantics (agent.py _check_question / _to_internal, common.py render_options) ----

[[noreturn]] static void question_fail(laya_question_fault fault, const std::string & qid, const std::string & field, const std::string & msg) {
    throw laya_question_error(fault, field, "question '" + qid + "': " + msg);
}

// identity of a choice label under Python == / hash: equal labels give equal strings
static std::string py_label_identity(const json & v) {
    if (v.is_string()) {
        return "s" + v.get<std::string>();
    }
    if (v.is_number_float()) {
        const double d = v.get<double>();
        if (d != std::floor(d) || std::fabs(d) >= 18446744073709551616.0) {
            return "f" + decision_py_float(d); // shortest round-trip text: distinct doubles differ
        }
        const uint64_t mag = (uint64_t) std::fabs(d);
        return std::string(d < 0 && mag != 0 ? "n-" : "n") + std::to_string(mag);
    }
    if (v.is_boolean()) {
        return v.get<bool>() ? "n1" : "n0";
    }
    if (v.is_number_unsigned()) {
        return "n" + std::to_string(v.get<uint64_t>());
    }
    const int64_t i = v.get<int64_t>();
    const uint64_t mag = i < 0 ? (uint64_t) 0 - (uint64_t) i : (uint64_t) i;
    return std::string(i < 0 ? "n-" : "n") + std::to_string(mag);
}

// the keys of a noul criteria object: str(k).lower() in {"true", "false"}. Only the
// Kelvin sign lowercases a non-ASCII character to an ASCII letter, and "k" is in
// neither word, so ASCII lowercasing gives the same verdict as Python's.
static std::string ascii_lower(std::string s) {
    for (char & c : s) {
        if (c >= 'A' && c <= 'Z') {
            c = (char) (c - 'A' + 'a');
        }
    }
    return s;
}

// a description the reference leaves out: None or ""
static bool crit_absent(const json & v) {
    return v.is_null() || (v.is_string() && v.get_ref<const std::string &>().empty());
}

laya_question laya_question_parse(const std::string & qid, const json & q, bool raw_options) {
    if (!q.is_object()) {
        question_fail(LAYA_QF_NOT_OBJECT, qid, "", "definition must be an object");
    }
    laya_question out;
    const json type = q.contains("type") ? q.at("type") : json();
    out.qtype = type.is_string() ? laya_qtype_from_name(type.get<std::string>()) : -1;
    if (out.qtype < 0) {
        question_fail(LAYA_QF_TYPE, qid, "type", "unknown type " + decision_py_dumps(type) + "; use one of choice, noul, score");
    }
    if (!q.contains("instructions")) {
        question_fail(LAYA_QF_INSTRUCTIONS, qid, "instructions", "no 'instructions'; add the text the model should answer");
    }
    const json & ins = q.at("instructions");
    out.instructions = ins.is_string() ? ins.get<std::string>() : decision_py_dumps(ins);

    const std::string t   = LAYA_QTYPE_NAMES[out.qtype];
    const json        crit = q.contains("criteria") ? q.at("criteria") : json();
    const bool        raw  = raw_options && q.contains("raw_options") && q.at("raw_options").is_boolean() && q.at("raw_options").get<bool>();

    if (t == "choice") {
        if (!crit.is_object() && !crit.is_array()) {
            question_fail(LAYA_QF_NO_OPTIONS, qid, "criteria", "a choice question takes 'criteria' as an object of label -> description, or a list of labels");
        }
        if (crit.empty()) {
            question_fail(LAYA_QF_NO_OPTIONS, qid, "criteria", "a choice question needs at least one criterion");
        }
        std::vector<std::pair<json, json>> entries; // (label, description)
        if (crit.is_array()) {
            std::unordered_map<std::string, size_t> seen;
            for (size_t i = 0; i < crit.size(); ++i) {
                const json & label = crit[i];
                if (label.is_object() || label.is_array()) {
                    question_fail(LAYA_QF_CRITERIA, qid, "criteria", "choice label " + std::to_string(i) +
                                  " is nested; a label is rendered as option text and used as the answer key, so it must be a string, number or bool");
                }
                if (label.is_null()) {
                    question_fail(LAYA_QF_CRITERIA, qid, "criteria", "choice label " + std::to_string(i) +
                                  " is null; a label is rendered as option text and used as the answer key, so it must be a string, number or bool");
                }
                const auto ins_res = seen.emplace(py_label_identity(label), i);
                if (!ins_res.second) {
                    question_fail(LAYA_QF_CRITERIA, qid, "criteria", "choice label " + std::to_string(i) + " (" + decision_py_dumps(label) +
                                  ") repeats label " + std::to_string(ins_res.first->second) +
                                  "; the labels are the answer keys, so every option needs its own (1, 1.0 and true are one key)");
                }
                entries.emplace_back(label, json());
            }
        } else {
            for (auto it = crit.begin(); it != crit.end(); ++it) {
                entries.emplace_back(json(it.key()), it.value());
            }
        }
        for (const auto & e : entries) {
            std::string text;
            decision_py_str(e.first, text);
            std::string key;
            decision_py_json_key(e.first, key);
            if (raw) {
                out.options.push_back(crit_absent(e.second) ? text : laya_render_criterion(e.second));
            } else {
                out.options.push_back(crit_absent(e.second) ? text : text + ": " + laya_render_criterion(e.second));
            }
            out.keys.push_back(key);
            out.labels.push_back(e.first);
        }
        out.criteria = crit;
    } else if (t == "score") {
        if (!crit.is_array()) {
            question_fail(crit.is_object() ? LAYA_QF_CRITERIA : LAYA_QF_NO_OPTIONS, qid, "criteria",
                          "a score question takes 'criteria' as a list of level descriptions, index 0 first");
        }
        if (crit.empty()) {
            question_fail(LAYA_QF_NO_OPTIONS, qid, "criteria", "a score question needs at least one level");
        }
        for (size_t i = 0; i < crit.size(); ++i) {
            if (crit[i].is_null()) {
                question_fail(LAYA_QF_CRITERIA, qid, "criteria", "score level " + std::to_string(i) + " is null; give every level a description, index 0 first");
            }
            out.options.push_back(raw ? laya_render_criterion(crit[i]) : "level " + std::to_string(i) + ": " + laya_render_criterion(crit[i]));
            out.keys.push_back(std::to_string(i));
        }
        out.criteria = crit;
    } else { // noul
        if (!crit.is_null() && !crit.is_object()) {
            question_fail(LAYA_QF_NOUL_CRITERIA, qid, "criteria", "a noul question takes 'criteria' as an object with optional 'true'/'false' descriptions, or omits it");
        }
        json lowered = crit.is_null() ? json() : json::object();
        for (auto it = crit.begin(); crit.is_object() && it != crit.end(); ++it) {
            const std::string k = ascii_lower(it.key());
            if (k != "true" && k != "false") {
                question_fail(LAYA_QF_NOUL_CRITERIA, qid, "criteria", "a noul question takes 'criteria' keyed only 'true'/'false' (either or both, and omitted is fine), got '" + it.key() + "'");
            }
            lowered[k] = it.value(); // a later duplicate wins, at the first position (dict comprehension)
        }
        std::string false_label = "false";
        std::string true_label  = "true";
        if (q.contains("labels") && !q.at("labels").is_null()) {
            const json & lb = q.at("labels");
            const std::string msg = "noul labels must map exactly 'false' and 'true' to distinct non-empty strings";
            if (!lb.is_object() || lb.size() != 2 || !lb.contains("false") || !lb.contains("true") ||
                !lb.at("false").is_string() || !lb.at("true").is_string()) {
                question_fail(LAYA_QF_LABELS, qid, "labels", msg);
            }
            false_label = decision_py_strip(lb.at("false").get<std::string>());
            true_label  = decision_py_strip(lb.at("true").get<std::string>());
            if (false_label.empty() || true_label.empty() || false_label == true_label) {
                question_fail(LAYA_QF_LABELS, qid, "labels", msg);
            }
        }
        if (raw) {
            out.options = { "false", "true" };
        } else {
            const json fc = lowered.is_object() && lowered.contains("false") ? lowered.at("false") : json();
            const json tc = lowered.is_object() && lowered.contains("true")  ? lowered.at("true")  : json();
            out.options = {
                false_label + ": " + (crit_absent(fc) ? std::string("no, the statement does not hold") : laya_render_criterion(fc)),
                true_label  + ": " + (crit_absent(tc) ? std::string("yes, the statement holds")        : laya_render_criterion(tc)),
            };
        }
        out.keys     = { "false", "true" };
        out.criteria = lowered;
    }
    if (q.contains("labels") && t != "noul") {
        question_fail(LAYA_QF_LABELS_NOT_NOUL, qid, "labels", "'labels' is only supported for noul questions");
    }
    return out;
}

static std::string replace_all(std::string s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return s;
    }
    size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.length(), to);
        pos += to.length();
    }
    return s;
}

// <unusedN> (mmBERT) or [unusedN] (ModernBERT)
static bool is_unused_token(const std::string & s) {
    if (s.size() < 9 || s.compare(1, 6, "unused") != 0 ||
        !((s[0] == '<' && s.back() == '>') || (s[0] == '[' && s.back() == ']'))) {
        return false;
    }
    for (size_t i = 7; i + 1 < s.size(); ++i) {
        if (s[i] < '0' || s[i] > '9') {
            return false;
        }
    }
    return true;
}

laya_escape laya_escape_init(const laya_model * model) {
    std::vector<std::string> text;
    std::vector<bool>        control;
    std::vector<std::string> strs;
    laya_vocab_added_tokens(model, text, control);
    for (size_t i = 0; i < text.size(); ++i) {
        if (!text[i].empty() && (control[i] || is_unused_token(text[i]))) {
            strs.push_back(text[i]);
        }
    }
    return laya_escape_make(laya_vocab_mask_token(model), std::move(strs));
}

laya_escape laya_escape_make(const std::string & mask, std::vector<std::string> control) {
    laya_escape esc;
    esc.mask    = mask;
    esc.control = std::move(control);
    esc.control.erase(std::remove(esc.control.begin(), esc.control.end(), std::string()), esc.control.end());
    if (!esc.mask.empty()) {
        esc.control.push_back(esc.mask); // escape-control includes mask-to-space
    }
    std::sort(esc.control.begin(), esc.control.end());
    esc.control.erase(std::unique(esc.control.begin(), esc.control.end()), esc.control.end());

    esc.trie.emplace_back();
    for (const auto & c : esc.control) {
        int32_t cur = 0;
        for (unsigned char ch : c) {
            int32_t child = -1;
            for (const auto & e : esc.trie[cur].next) {
                if (e.first == ch) {
                    child = e.second;
                    break;
                }
            }
            if (child < 0) {
                child = (int32_t) esc.trie.size();
                esc.trie[cur].next.emplace_back(ch, child);
                esc.trie.emplace_back();
            }
            cur = child;
        }
        esc.trie[cur].end = true;
    }
    return esc;
}

// length of the longest control string at text[pos], 0 if none
static size_t escape_match(const laya_escape & esc, const std::string & text, size_t pos) {
    size_t  len = 0;
    int32_t cur = 0;
    for (size_t j = pos; j < text.size(); ++j) {
        int32_t child = -1;
        for (const auto & e : esc.trie[cur].next) {
            if (e.first == (uint8_t) text[j]) {
                child = e.second;
                break;
            }
        }
        if (child < 0) {
            break;
        }
        cur = child;
        if (esc.trie[cur].end) {
            len = j + 1 - pos;
        }
    }
    return len;
}

std::string laya_escape_text(const laya_escape & esc, const std::string & text, bool escape_control) {
    if (!escape_control || esc.trie.empty()) {
        return replace_all(text, esc.mask, " ");
    }
    std::string out;
    out.reserve(text.size());
    size_t i = 0;
    while (i < text.size()) {
        const size_t len = escape_match(esc, text, i);
        if (len > 0) {
            out += ' ';
            i += len;
        } else {
            out += text[i++];
        }
    }
    return out;
}

laya_seq laya_build_sequence(
        const laya_model * model,
        int32_t qtype,
        const std::string & instructions,
        const std::vector<std::string> & options,
        const std::string & state_text,
        bool state_is_list) {
    const std::string mask_str = laya_vocab_mask_token(model);
    return laya_build_sequence(model, qtype, instructions, options,
                               laya_tokenize(model, replace_all(state_text, mask_str, " ")), state_is_list);
}

laya_seq laya_build_sequence(
        const laya_model * model,
        int32_t qtype,
        const std::string & instructions,
        const std::vector<std::string> & options,
        const std::vector<int32_t> & state_ids,
        bool state_is_list) {
    const laya_hparams & hp = laya_model_hparams(model);
    const int32_t max_len      = hp.max_len > 0 ? hp.max_len : 1024;
    const int32_t head_max_len = hp.head_max_len > 0 ? hp.head_max_len : 256;

    const int32_t cls_token_id  = laya_vocab_bos(model);
    const int32_t sep_token_id  = laya_vocab_sep(model);
    const int32_t mask_token_id = laya_vocab_mask(model);

    const std::string t        = LAYA_QTYPE_NAMES[qtype];
    const std::string mask_str = laya_vocab_mask_token(model);
    const std::string ins      = replace_all(instructions, mask_str, " ");

    laya_seq seq;
    seq.qtype = qtype;

    // head_ids = tokenizer("%s question: %s" % (t, ins), add_special_tokens=False)
    std::vector<int32_t> head_ids = laya_tokenize(model, t + " question: " + ins);

    // opt_ids = [ [mask] + tokenizer(" " + opt)[:48] ... ]
    std::vector<std::vector<int32_t>> opt_ids;
    for (const auto & opt : options) {
        std::vector<int32_t> o = { mask_token_id };
        std::vector<int32_t> tok = laya_tokenize(model, " " + replace_all(opt, mask_str, " "));
        if (tok.size() > 48) {
            tok.resize(48);
            seq.options_cut = true;
        }
        o.insert(o.end(), tok.begin(), tok.end());
        opt_ids.push_back(o);
    }

    int32_t opt_budget = head_max_len;
    for (const auto & o : opt_ids) {
        opt_budget -= (int32_t) o.size();
    }
    if (opt_budget < 16) {
        const int32_t per = std::max(4, (head_max_len - 16) / std::max(1, (int32_t) opt_ids.size()));
        seq.tokens_per_option = per;
        for (auto & o : opt_ids) {
            if ((int32_t) o.size() > per) {
                o.resize(per);
                seq.options_cut = true;
            }
        }
        opt_budget = head_max_len;
        for (const auto & o : opt_ids) {
            opt_budget -= (int32_t) o.size();
        }
    }
    if ((int32_t) head_ids.size() > std::max(8, opt_budget)) {
        head_ids.resize(std::max(8, opt_budget));
        seq.head_cut = true;
    }

    std::vector<int32_t> & ids = seq.ids;
    ids.push_back(cls_token_id);
    ids.insert(ids.end(), head_ids.begin(), head_ids.end());
    ids.push_back(sep_token_id);

    std::vector<int32_t> markers;
    for (const auto & o : opt_ids) {
        markers.push_back((int32_t) ids.size());
        ids.insert(ids.end(), o.begin(), o.end());
    }
    ids.push_back(sep_token_id);

    const int32_t room  = std::max(0, max_len - (int32_t) ids.size() - 1);
    const int32_t n_st  = (int32_t) state_ids.size();
    const int32_t n_keep = std::min(n_st, room);
    seq.n_state     = n_st;
    seq.n_state_cut = n_st - n_keep;
    if (state_is_list) {
        ids.insert(ids.end(), state_ids.end() - n_keep, state_ids.end());
    } else {
        ids.insert(ids.end(), state_ids.begin(), state_ids.begin() + n_keep);
    }
    ids.push_back(sep_token_id);

    if ((int32_t) ids.size() > max_len) {
        ids.resize(max_len);
    }

    seq.n_options = (int32_t) opt_ids.size();
    seq.n_options_distinct = (int32_t) std::set<std::vector<int32_t>>(opt_ids.begin(), opt_ids.end()).size();
    for (int32_t m : markers) {
        if (m < max_len) {
            seq.markers.push_back(m);
        }
    }
    return seq;
}

void laya_batch_pack(const std::vector<const laya_seq *> & seqs, laya_batch_data & out) {
    const int32_t n_seqs = (int32_t) seqs.size();
    int32_t n_tokens = 0;
    for (const auto * s : seqs) {
        n_tokens += (int32_t) s->ids.size();
    }

    out.tokens.assign(n_tokens, 0);
    out.positions.assign(n_tokens, 0);
    out.seq_id.assign(n_tokens, 0);
    out.qtype.assign(n_tokens, 0);
    out.marker_pos.assign((size_t) LAYA_MAX_MARKERS * n_seqs, 0);
    out.marker_mask.assign((size_t) LAYA_MAX_MARKERS * n_seqs, 0);
    out.seq_start.assign(n_seqs, 0);

    int32_t t = 0;
    for (int32_t s = 0; s < n_seqs; ++s) {
        const laya_seq & seq = *seqs[s];
        out.seq_start[s] = t;
        for (size_t i = 0; i < seq.ids.size(); ++i) {
            out.tokens[t]    = seq.ids[i];
            out.positions[t] = (int32_t) i;
            out.seq_id[t]    = s;
            out.qtype[t]     = seq.qtype;
            ++t;
        }
        const int32_t k = (int32_t) seq.markers.size();
        for (int32_t j = 0; j < k && j < LAYA_MAX_MARKERS; ++j) {
            // row-major [n_markers_max, n_seqs]: index = m + n_markers_max * s
            // marker positions are sequence-local; the flattened encoder output
            // concatenates the sequences, so offset by this sequence's start.
            out.marker_pos[j + LAYA_MAX_MARKERS * s]  = out.seq_start[s] + seq.markers[j];
            out.marker_mask[j + LAYA_MAX_MARKERS * s] = 1;
        }
    }

    laya_batch & batch = out.batch;
    batch.n_tokens    = n_tokens;
    batch.n_seqs      = n_seqs;
    batch.tokens      = out.tokens.data();
    batch.positions   = out.positions.data();
    batch.seq_id      = out.seq_id.data();
    batch.qtype       = out.qtype.data();
    batch.marker_pos  = out.marker_pos.data();
    batch.marker_mask = out.marker_mask.data();
    batch.seq_start   = out.seq_start.data();
}

// ---- temperature / confidence (ports of laya/common.py + agent.py) ----

static float clamp_temperature(float t) {
    if (std::isnan(t) || std::isinf(t)) {
        return 1.0f;
    }
    return std::min(5.0f, std::max(0.5f, t));
}

// reference: temperature_by_options[temp_bucket(qtype, k)], else temperature[qtype]; each clamped
static float temperature_for_qtype(const laya_hparams & hp, int32_t qtype, int32_t k) {
    if (qtype >= 0 && qtype < 3 && !hp.temperature_buckets.empty()) {
        const char * size = k <= 2 ? "2" : k <= 5 ? "3-5" : k <= 10 ? "6-10" : "11+";
        const std::string bucket = std::string(LAYA_QTYPE_NAMES[qtype]) + ":" + size;
        for (size_t i = 0; i < hp.temperature_buckets.size(); ++i) {
            if (hp.temperature_buckets[i] == bucket) {
                return clamp_temperature(hp.temperature_bucket_values[i]);
            }
        }
    }
    if (hp.temperature.empty()) {
        return 1.0f;
    }
    return clamp_temperature(hp.temperature[qtype % (int32_t) hp.temperature.size()]);
}

static float confidence_from_probs(const std::vector<float> & p) {
    const int k = (int) p.size();
    if (k < 2) {
        return 1.0f;
    }
    float ent = 0.0f;
    for (int i = 0; i < k; ++i) {
        if (p[i] > 0.0f) {
            ent -= p[i] * std::log(p[i]);
        }
    }
    return std::min(1.0f, std::max(0.0f, 1.0f - ent / std::log((float) k)));
}

// Python round(x, 4): correctly rounded on the exact binary value, like printf
static double py_round4(double x) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.4f", x);
    return strtod(buf, nullptr);
}

void laya_postprocess(
        const laya_question & q,
        const laya_seq & seq,
        const laya_result & result,
        int32_t qid,
        const laya_hparams & hp,
        json & ans,
        json & pq) {
    const int32_t n_markers_max = result.n_markers_max;
    const int32_t n_act         = result.n_act;

    const std::string t = LAYA_QTYPE_NAMES[q.qtype];
    const int32_t k  = (int32_t) seq.markers.size();
    const int32_t qt = seq.qtype;

    const float t_scale = temperature_for_qtype(hp, qt, k);
    std::vector<float> logits_k(k);
    for (int32_t j = 0; j < k; ++j) {
        logits_k[j] = result.logits[qid*n_markers_max + j] / t_scale;
    }

    // softmax
    std::vector<float> p(k);
    float max_l = logits_k[0];
    for (int32_t j = 1; j < k; ++j) {
        max_l = std::max(max_l, logits_k[j]);
    }
    float sum = 0.0f;
    for (int32_t j = 0; j < k; ++j) {
        p[j] = std::exp(logits_k[j] - max_l);
        sum += p[j];
    }
    for (int32_t j = 0; j < k; ++j) {
        p[j] /= sum;
    }

    int32_t argmax = 0;
    for (int32_t j = 1; j < k; ++j) {
        if (p[j] > p[argmax]) {
            argmax = j;
        }
    }
    const float answer_conf = std::min(1.0f, std::max(0.0f, p[argmax]));

    // action head softmax
    std::vector<float> act(n_act);
    float amax = result.act_logits[qid*n_act];
    for (int32_t j = 0; j < n_act; ++j) {
        act[j] = result.act_logits[qid*n_act + j];
        amax = std::max(amax, act[j]);
    }
    float asum = 0.0f;
    for (int32_t j = 0; j < n_act; ++j) {
        act[j] = std::exp(act[j] - amax);
        asum += act[j];
    }
    for (int32_t j = 0; j < n_act; ++j) {
        act[j] /= asum;
    }
    const float act_probability = act[0];
    const json action = { {"act_probability", py_round4(act_probability)} };

    // key order as in the reference Agent._decode_answers
    ans = json::object();
    ans["type"] = t;
    if (t == "choice") {
        json probs = json::object();
        for (int32_t j = 0; j < k; ++j) {
            probs[q.keys[j]] = py_round4(p[j]); // colliding answer keys (e.g. "1" and 1) keep the last value
        }
        ans["choice"]        = q.labels[argmax];
        ans["probabilities"] = probs;
        ans["confidence"]    = py_round4(confidence_from_probs(p));
    } else if (t == "score") {
        double exp_score = 0.0;
        json legend = json::object();
        json probs  = json::object();
        for (int32_t j = 0; j < k; ++j) {
            exp_score += (double) j * (double) p[j];
            legend[std::to_string(j)] = q.criteria[j];
            probs[std::to_string(j)]  = py_round4(p[j]);
        }
        ans["score"]         = py_round4(exp_score);
        ans["legend"]        = legend;
        ans["probabilities"] = probs;
        ans["confidence"]    = py_round4(confidence_from_probs(p));
    } else { // noul
        ans["noul"]       = py_round4(p[1]);
        ans["confidence"] = py_round4(std::max(p[1], 1.0f - p[1]));
    }
    ans["answer_confidence"] = py_round4(answer_conf);
    ans["action"]            = action;

    // per-question raw outputs (for comparison with golden fixtures)
    pq = json::object();
    pq["qtype"] = t;
    json opt_arr = json::array();
    for (const auto & o : q.options) {
        opt_arr.push_back(o);
    }
    pq["options"] = opt_arr;
    json ids_arr = json::array();
    for (auto tok : seq.ids) {
        ids_arr.push_back((int64_t) tok);
    }
    pq["input_ids"] = ids_arr;
    json mk_arr = json::array();
    for (auto m : seq.markers) {
        mk_arr.push_back((int64_t) m);
    }
    pq["marker_pos"] = mk_arr;
    json rl_arr = json::array();
    for (int32_t j = 0; j < k; ++j) {
        rl_arr.push_back((double) result.logits[qid*n_markers_max + j]);
    }
    pq["raw_logits"] = rl_arr;
    json al_arr = json::array();
    for (int32_t j = 0; j < n_act; ++j) {
        al_arr.push_back((double) act[j]);
    }
    pq["act_logits"] = al_arr;
    json ar_arr = json::array();
    for (int32_t j = 0; j < n_act; ++j) {
        ar_arr.push_back((double) result.act_logits[qid*n_act + j]);
    }
    pq["act_raw_logits"] = ar_arr;
    pq["act_probability"] = (double) act_probability;
}
