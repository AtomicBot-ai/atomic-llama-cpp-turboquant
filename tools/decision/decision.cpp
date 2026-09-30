#include "decision.h"
#include "decision-spec.h"

static const struct {
    const char * name;
    int          status;
} DECISION_REASONS[DECISION_REASON_COUNT] = {
    { "MALFORMED_JSON",             400 },
    { "BODY_NOT_OBJECT",            400 },
    { "INVALID_REQUEST",            400 },
    { "UNKNOWN_QUESTION_TYPE",      400 },
    { "EMPTY_INSTRUCTIONS",         400 },
    { "TOO_FEW_OPTIONS",            400 },
    { "TOO_MANY_OPTIONS",           400 },
    { "INVALID_NOUL_CRITERIA",      400 },
    { "UNSUPPORTED_NUMBER",         400 },
    { "UNSUPPORTED_CRITERIA_VALUE", 400 },
    { "TOO_MANY_QUESTIONS",         400 },
    { "INVALID_CARD",               400 },
    { "DUPLICATE_CANDIDATE_ID",     400 },
    { "INVALID_CANDIDATE_ID",       400 },
    { "TOO_MANY_CANDIDATES",        400 },
    { "BODY_TOO_LARGE",             413 },
    { "PROMPT_TOO_LONG",            422 },
    { "CARD_TOO_LONG",              422 },
    { "CRITERION_TOO_LONG",         422 },
    { "OPTIONS_TRUNCATED",          422 },
    { "STATE_TRUNCATED",            422 },
    { "OVERLOADED",                 429 },
    { "ROUTER_NOT_CALIBRATED",      501 },
    { "INTERNAL",                   500 },
};

const char * decision_reason_name(decision_reason reason) {
    return reason < DECISION_REASON_COUNT ? DECISION_REASONS[reason].name : "INTERNAL";
}

int decision_reason_status(decision_reason reason) {
    return reason < DECISION_REASON_COUNT ? DECISION_REASONS[reason].status : 500;
}

// same type names as format_error_response in tools/server
const char * decision_reason_type(decision_reason reason) {
    switch (decision_reason_status(reason)) {
        case 400:
        case 413:
        case 422: return "invalid_request_error";
        case 429: return "unavailable_error";
        case 501: return "not_supported_error";
        default:  return "server_error";
    }
}

json decision_error_body(const decision_error & err) {
    json e = {
        {"code",    decision_reason_status(err.reason)},
        {"type",    decision_reason_type(err.reason)},
        {"reason",  decision_reason_name(err.reason)},
        {"message", err.what()},
    };
    if (!err.param.empty()) {
        e["param"] = err.param;
    }
    return json{{"error", e}};
}

static const char * const DECISION_QTYPE_NAMES[] = { "choice", "score", "noul" };

const char * decision_qtype_name(decision_qtype type) {
    return DECISION_QTYPE_NAMES[type];
}

bool decision_qtype_from_name(const std::string & name, decision_qtype & type) {
    for (int i = 0; i < 3; ++i) {
        if (name == DECISION_QTYPE_NAMES[i]) {
            type = (decision_qtype) i;
            return true;
        }
    }
    return false;
}

std::unique_ptr<decision_engine> decision_engine_init(const decision_spec & spec, const decision_engine_params & params) {
    if (spec.layout == "laya") {
        return decision_engine_laya_init(spec, params);
    }
    throw std::runtime_error("decision layout '" + spec.layout + "' is not supported yet (letters engine: Arbiter/JevK5 comes later)");
}
