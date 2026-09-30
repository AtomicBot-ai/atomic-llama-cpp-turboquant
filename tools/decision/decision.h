#pragma once

// Decision API v1: types shared by the request layer, calibration and engines.
// Engines only turn (question, state) items into raw per-option logits; the
// request checks, calibration, confidence and answer shapes are common code.

#include "decision-json.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define DECISION_API_VERSION 1

// reason codes of the error envelope {"error": {"code", "type", "reason", "message", "param"}}
enum decision_reason {
    // 400
    DECISION_REASON_MALFORMED_JSON,
    DECISION_REASON_BODY_NOT_OBJECT,
    DECISION_REASON_INVALID_REQUEST,          // a top-level field has the wrong shape
    DECISION_REASON_UNKNOWN_QUESTION_TYPE,
    DECISION_REASON_EMPTY_INSTRUCTIONS,
    DECISION_REASON_TOO_FEW_OPTIONS,
    DECISION_REASON_TOO_MANY_OPTIONS,
    DECISION_REASON_INVALID_NOUL_CRITERIA,
    DECISION_REASON_UNSUPPORTED_NUMBER,
    DECISION_REASON_UNSUPPORTED_CRITERIA_VALUE,
    DECISION_REASON_TOO_MANY_QUESTIONS,
    DECISION_REASON_INVALID_CARD,
    DECISION_REASON_DUPLICATE_CANDIDATE_ID,
    DECISION_REASON_INVALID_CANDIDATE_ID,
    DECISION_REASON_TOO_MANY_CANDIDATES,
    // 413
    DECISION_REASON_BODY_TOO_LARGE,
    // 422
    DECISION_REASON_PROMPT_TOO_LONG,
    DECISION_REASON_CARD_TOO_LONG,
    DECISION_REASON_CRITERION_TOO_LONG,
    DECISION_REASON_OPTIONS_TRUNCATED,
    DECISION_REASON_STATE_TRUNCATED,
    // 429
    DECISION_REASON_OVERLOADED,
    // 501
    DECISION_REASON_ROUTER_NOT_CALIBRATED,
    // 500
    DECISION_REASON_INTERNAL,

    DECISION_REASON_COUNT,
};

const char * decision_reason_name(decision_reason reason);
int          decision_reason_status(decision_reason reason);
const char * decision_reason_type(decision_reason reason);

struct decision_error : std::runtime_error {
    decision_reason reason;
    std::string     param;

    decision_error(decision_reason reason, const std::string & msg, const std::string & param = "")
        : std::runtime_error(msg), reason(reason), param(param) {}
};

// {"error": {"code", "type", "reason", "message", "param"?}}
json decision_error_body(const decision_error & err);

// same ids as the laya type embedding
enum decision_qtype {
    DECISION_QTYPE_CHOICE = 0,
    DECISION_QTYPE_SCORE  = 1,
    DECISION_QTYPE_NOUL   = 2,
};

const char * decision_qtype_name(decision_qtype type);
bool         decision_qtype_from_name(const std::string & name, decision_qtype & type);

struct decision_question {
    std::string              id;            // key in "questions", or the router candidate id
    decision_qtype           type = DECISION_QTYPE_NOUL;
    std::string              instructions;
    json                     criteria;      // noul: object or null, choice: object (laya layout: object or list as given), score: array
    std::vector<std::string> keys;          // answer keys in option order; noul: "false", "true"
    json                     labels = json::array(); // laya layout, choice: the labels as given (answer "choice"); empty: keys
    std::vector<std::string> options;       // laya layout: option texts (reference render_options); empty: the engine renders criteria
};

// one sequence: a question over a state
struct decision_item {
    decision_question q;
    json              state;
    std::string       param;                // error param for this item's options; empty: questions.<id>.criteria
    std::vector<size_t> state_splits;       // router: byte offsets where the string state splits into pieces that
                                            // tokenize on their own (after the card, after "task:"); engines
                                            // tokenize each distinct piece once per request
    bool              escape_control = false; // special_tokens "escape-control": control-token text in the state becomes a space
};

struct decision_output {
    std::vector<double>  logits;            // raw logits in q.keys order
    int32_t              n_tokens    = 0;   // full prompt length
    int32_t              n_evaluated = 0;   // tokens computed for this item
    int32_t              n_state_cut = 0;   // state tokens dropped to fit the model
    int32_t              n_state     = 0;   // state tokens before the cut
    int32_t              n_state_head = -1; // state tokens before the last split (router: card + criterion), -1 if not split
    bool                 options_cut = false;
    bool                 head_cut    = false;
    std::vector<int32_t> tokens;            // prompt token ids
};

struct decision_caps {
    std::string              layout;        // "laya", "semif-letters"
    std::string              format;        // "laya-v1", "semif-v1"
    std::string              plan;
    std::string              plan_independent; // plan of evaluate(independent = true): items never influence each other
    std::vector<std::string> plans;
    std::vector<std::string> question_types;
    int32_t                  max_options = 0;
    int32_t                  max_tokens  = 0;  // sequence limit of the model
    int32_t                  n_threads   = 0;
    std::string              device;
    bool                     state_split = false; // router state pieces are tokenized once per request (exact for this vocabulary)
};

// called for every rendered item before any compute; may throw decision_error
using decision_render_check = std::function<void(size_t idx, const decision_output & rendered)>;

struct decision_engine {
    virtual ~decision_engine() = default;

    virtual decision_caps caps() const = 0;

    // tokens of a text as the engine tokenizes a state (with the same special-token handling)
    virtual int32_t n_tokens(const std::string & text, bool escape_control) const = 0;

    // prompt of one item without compute (tokens and cut info)
    virtual decision_output render(const decision_item & item) const = 0;

    // render all items, run check on each, then compute them in item order.
    // independent: use caps().plan_independent so that no item changes another's
    // logits (router scores). Throws decision_error on items that cannot be rendered.
    // Returns false when cancel became true (out is then incomplete).
    virtual bool evaluate(
            const std::vector<decision_item> & items,
            bool independent,
            const std::atomic<bool> * cancel,
            const decision_render_check & check,
            std::vector<decision_output> & out) = 0;
};

struct decision_spec;

struct decision_engine_params {
    std::string model_path;
    std::string plan;           // empty: spec plan, then engine default
    int32_t     n_threads = 4;
};

// engine for spec.layout; throws std::runtime_error when the model cannot be used
std::unique_ptr<decision_engine> decision_engine_init(const decision_spec & spec, const decision_engine_params & params);

std::unique_ptr<decision_engine> decision_engine_laya_init(const decision_spec & spec, const decision_engine_params & params);
