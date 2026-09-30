#pragma once

// POST /v1/systemone request checks and normalization.
// Layout "laya": question semantics of the Laya reference (laya_question_parse in
// laya-decide.h): a missing state is refused, instructions may be any JSON value,
// choice/score need one option, list labels must not repeat, noul labels are
// accepted. Other layouts: TypeSafe superset following atomic-arbiter
// arbiter/prompt.py normalized_question: list choice criteria become
// dict.fromkeys(str(c)), duplicates collapse, the option count is checked after that.

#include "decision.h"

#include <string>
#include <vector>

#define DECISION_MAX_BODY_BYTES (1024 * 1024)

struct decision_limits {
    int32_t     max_items   = 16;   // questions per systemone request, candidates per router request
    int32_t     max_options = 20;   // options per question (laya 20, semif 16)
    std::string layout;             // "laya": Laya reference question semantics; anything else: TypeSafe
};

struct decision_request {
    std::string                    model;
    json                           state;
    std::vector<decision_question> questions;
    bool                           truncation_error = false; // "truncation": "error"
};

// body text -> JSON object; throws decision_error (BODY_TOO_LARGE, MALFORMED_JSON, UNSUPPORTED_NUMBER, BODY_NOT_OBJECT)
json decision_parse_body(const std::string & text, size_t max_bytes = DECISION_MAX_BODY_BYTES);

// "truncation": "allow" (default) or "error"
bool decision_parse_truncation(const json & body);

// one question; param is the JSON path used in errors
decision_question decision_normalize_question(const std::string & id, const json & q, const decision_limits & lim, const std::string & param);

decision_request decision_parse_systemone(const json & body, const decision_limits & lim);

// one item per question, all over the request state
std::vector<decision_item> decision_request_items(const decision_request & req);

// render check for "truncation": "error": OPTIONS_TRUNCATED, PROMPT_TOO_LONG (instructions) or STATE_TRUNCATED on any cut
decision_render_check decision_request_render_check(const decision_request & req);

// Python str.strip() is empty
bool decision_is_blank(const std::string & s);
