#pragma once

// POST /v1/router/score: executor cards (atomic.executor-card/1), the card-v1
// text rendering and the expansion into one noul question per candidate.
//
// card-v1 (integers only, so Python and C++ render the same bytes):
//   executor: <name>
//   kind: <kind>
//   description: <description>                          (when present)
//   checks:                                             ("checks: none" when empty)
//   - <skill>: passed <passed> of <total>; criterion: <criterion>; source: <source> <version>
//   - <skill>: not measured; source: <source> <version>
// Card strings are made single-line: control characters become spaces, space
// runs collapse, ends are trimmed. Optional parts are left out when absent.
// With special_tokens "escape-control" the engine also turns control-token text
// (<eos>, <unusedN>, ...) in card, criterion and task into a space.
//
// Candidate state (card first, so a cut only hits the task tail):
//   <card-v1>\n\nsuccess criterion: <criterion>\n\ntask: <task>

#include "decision.h"
#include "decision-request.h"
#include "decision-spec.h"

#include <string>
#include <vector>

#define DECISION_MAX_CANDIDATES       16
#define DECISION_MAX_CHECKS           32
#define DECISION_MAX_CARD_FIELD_BYTES 4096 // per card string; bounds tokenizer work before CARD_TOO_LONG

struct decision_candidate {
    std::string id;
    json        card;
    std::string card_text;          // card-v1
};

struct decision_router_request {
    std::string                     model;
    std::string                     task;
    std::string                     criterion;
    std::vector<decision_candidate> candidates;
    bool                            truncation_error = false;
    std::vector<std::string>        warnings;      // unknown card fields, ...
};

// throws decision_error (INVALID_CARD, also for strings over DECISION_MAX_CARD_FIELD_BYTES); unknown fields add warnings
void decision_card_validate(const json & card, const std::string & param, std::vector<std::string> & warnings);

std::string decision_card_render(const json & card);

// id matches [A-Za-z0-9._:/@+-]{1,128}
bool decision_candidate_id_valid(const std::string & id);

// max_candidates = min(--decision-max-items, spec limit)
decision_router_request decision_parse_router(const json & body, int32_t max_candidates);

// "<card>\n\nsuccess criterion: <criterion>\n\ntask:" - the part that must never be cut
std::string decision_router_state_prefix(const std::string & card_text, const std::string & criterion);

std::string decision_router_state(const std::string & card_text, const std::string & criterion, const std::string & task);

// one noul item per candidate with the spec router question; the item id is the candidate id.
// The card and the "\n\n" after it, and "task:" and the " " after it, are token boundaries,
// so engines tokenize the shared criterion and task once per request.
std::vector<decision_item> decision_router_items(const decision_router_request & req, const decision_router_spec & router, const decision_limits & lim);

// raw router logit z = s_true - s_false
double decision_router_logit(const decision_output & out);

// CARD_TOO_LONG when a card has more than router.max_card_tokens tokens (counted as in the state)
void decision_router_check_cards(const decision_router_request & req, const decision_router_spec & router, const decision_engine & engine);

// render check: CRITERION_TOO_LONG when the state cut reaches into the card or the
// criterion; STATE_TRUNCATED on any cut with "truncation": "error"
decision_render_check decision_router_render_check(const decision_router_request & req, const decision_router_spec & router, const decision_engine & engine);
