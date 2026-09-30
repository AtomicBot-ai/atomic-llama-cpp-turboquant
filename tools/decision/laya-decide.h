#pragma once

// Input building and answer post-processing for the laya model, shared by
// llama-laya-cli (tools/laya) and the decision engine (engine-laya.cpp).
// Ports of laya/common.py (build_sequence, serialize_state, render_options) and of the
// question checks of laya/agent.py (Agent._check_question, Agent._to_internal).

#include "laya.h"
#include "decision-json.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// question types in type-embedding order
extern const char * const LAYA_QTYPE_NAMES[3];

// index into LAYA_QTYPE_NAMES, -1 if unknown
int32_t laya_qtype_from_name(const std::string & name);

// strings as-is, everything else as json.dumps(ensure_ascii=False)
std::string laya_serialize_state(const json & state);
std::string laya_render_criterion(const json & value);

// option texts in label order for an already normalized question {type, criteria, raw_options?}
// (choice criteria as an object, or a list collapsed with dict.fromkeys(str(c))).
// Used for questions normalized by the TypeSafe rules; laya_question_parse renders the
// reference semantics. Throws std::runtime_error on criteria that cannot be rendered.
std::vector<std::string> laya_render_options(const json & q);

// Question semantics of the Laya reference (laya 0.3.21 agent.py Agent._check_question,
// Agent._to_internal, common.py render_options), shared by llama-laya-cli and the
// server's laya layout:
//   - "type" is choice / score / noul; "instructions" must be present and may be any
//     JSON value: a non-string is json.dumps(ensure_ascii=False)'d
//   - choice: "criteria" is an object {label: description} or a non-empty list of
//     labels. A list label must be a string, number or bool (not null, not nested) and
//     must not repeat another by Python equality (1, 1.0 and True are one label).
//     Option text: str(label), plus ": " + description unless the description is null or "".
//   - score: a non-empty list of levels, none null; option text "level i: <level>"
//   - noul: "criteria" is null/absent or an object whose keys lowercase to "true"/"false"
//     (a later duplicate after lowercasing wins); "labels" (noul only) is null or exactly
//     {"false": str, "true": str}, stripped (Python str.strip), non-empty and distinct,
//     and replaces the "false" / "true" words in the option texts
// A structured description or level is compact JSON (render_criterion).
enum laya_question_fault {
    LAYA_QF_NOT_OBJECT,       // the definition is not an object
    LAYA_QF_TYPE,             // unknown type
    LAYA_QF_INSTRUCTIONS,     // no "instructions" key
    LAYA_QF_NO_OPTIONS,       // choice/score criteria missing, of the wrong kind, or empty
    LAYA_QF_CRITERIA,         // a label or level the reference rejects, or score criteria given as an object
    LAYA_QF_NOUL_CRITERIA,    // noul criteria not an object, or keyed other than true/false
    LAYA_QF_LABELS,           // invalid noul labels
    LAYA_QF_LABELS_NOT_NOUL,  // labels on a choice or score question
};

struct laya_question_error : std::runtime_error {
    laya_question_fault fault;
    std::string         field;  // "type", "instructions", "criteria", "labels", or "" for the question itself

    laya_question_error(laya_question_fault fault, const std::string & field, const std::string & msg)
        : std::runtime_error(msg), fault(fault), field(field) {}
};

struct laya_question {
    int32_t                  qtype = 0;
    std::string              instructions;            // model text of the instructions
    json                     criteria;                // choice: as given; score: as given; noul: keys lowercased, or null
    std::vector<std::string> options;                 // option texts in label order
    std::vector<std::string> keys;                    // answer keys: json.dumps of each choice label as a dict key; score "0".."K-1"; noul "false", "true"
    json                     labels = json::array();  // choice: the labels as given (the "choice" value of an answer)
};

// Throws laya_question_error. raw_options: honour the llama-laya-cli extension
// "raw_options": true (bare option texts, for the Julia family), which the reference does not have.
laya_question laya_question_parse(const std::string & qid, const json & q, bool raw_options = false);

struct laya_seq {
    std::vector<int32_t> ids;
    std::vector<int32_t> markers;       // marker positions inside max_len
    int32_t qtype        = 0;
    int32_t n_options    = 0;           // options rendered; markers.size() < n_options means markers were cut
    int32_t n_options_distinct = 0;     // options whose capped token spans differ (reference usage "options")
    int32_t tokens_per_option  = -1;    // per-option cap of the head budget, -1 when none applied
    int32_t n_state      = 0;           // state tokens before truncation
    int32_t n_state_cut  = 0;           // state tokens dropped to fit max_len
    bool    options_cut  = false;       // an option lost tokens (48 per option, head budget)
    bool    head_cut     = false;       // the instructions lost tokens
};

// Special-token text in user input. The reference turns the mask literal into a
// space (special_tokens "mask-to-space"). "escape-control" (router inputs) also
// turns every CONTROL-token string (<pad>, <eos>, <bos>, <unusedN>, ...) and every
// <unusedN> added token into a space. These strings cannot overlap each other and
// a space cannot form a new one, so one pass equals str.replace in any order.
struct laya_escape {
    struct node {
        std::vector<std::pair<uint8_t, int32_t>> next;  // byte -> node
        bool                                     end = false;
    };
    std::string              mask;      // the mask literal, e.g. "<mask>"
    std::vector<std::string> control;   // escape-control strings (includes the mask)
    std::vector<node>        trie;      // byte trie over control, for longest match
};

laya_escape laya_escape_init(const laya_model * model);

std::string laya_escape_text(const laya_escape & esc, const std::string & text, bool escape_control);

// Format: [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] <state> [SEP]
// The mask literal in instructions and options becomes a space. state_ids are the
// tokens of the escaped state text. A list state is cut from the left, others from the right.
laya_seq laya_build_sequence(
        const laya_model * model,
        int32_t qtype,
        const std::string & instructions,
        const std::vector<std::string> & options,
        const std::vector<int32_t> & state_ids,
        bool state_is_list);

// same, tokenizing state_text with the mask literal turned into a space
laya_seq laya_build_sequence(
        const laya_model * model,
        int32_t qtype,
        const std::string & instructions,
        const std::vector<std::string> & options,
        const std::string & state_text,
        bool state_is_list);

// inputs of one forward pass over several sequences (block-diagonal attention)
struct laya_batch_data {
    std::vector<int32_t> tokens;
    std::vector<int32_t> positions;
    std::vector<int32_t> seq_id;
    std::vector<int32_t> qtype;
    std::vector<int32_t> marker_pos;
    std::vector<int32_t> marker_mask;
    std::vector<int32_t> seq_start;
    laya_batch batch;
};

// seqs must have at most LAYA_MAX_MARKERS markers each
void laya_batch_pack(const std::vector<const laya_seq *> & seqs, laya_batch_data & out);

// llama-laya-cli answer for question idx of res, shaped like the reference Agent answer:
// per-qtype base temperature, float softmax, noul confidence max(p, 1 - p), choice/score
// confidence 1 - H/log K, answer_confidence max(p), the action head, values rounded
// like Python round(x, 4). per_question gets the raw outputs compared against tests/laya/golden.
void laya_postprocess(
        const laya_question & q,
        const laya_seq & seq,
        const laya_result & res,
        int32_t idx,
        const std::vector<float> & temperature,
        json & answer,
        json & per_question);
