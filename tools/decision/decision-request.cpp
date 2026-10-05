#include "decision-request.h"
#include "laya-decide.h"

#include <set>

json decision_parse_body(const std::string & text, size_t max_bytes) {
    if (text.size() > max_bytes) {
        throw decision_error(DECISION_REASON_BODY_TOO_LARGE, "request body is larger than " + std::to_string(max_bytes) + " bytes");
    }
    json body;
    std::string err;
    switch (decision_json_parse(text, body, err)) {
        case DECISION_JSON_OK: break;
        case DECISION_JSON_UNSUPPORTED_NUMBER: throw decision_error(DECISION_REASON_UNSUPPORTED_NUMBER, err);
        default:                               throw decision_error(DECISION_REASON_MALFORMED_JSON, err);
    }
    if (!body.is_object()) {
        throw decision_error(DECISION_REASON_BODY_NOT_OBJECT, "request body must be a JSON object");
    }
    return body;
}

bool decision_parse_truncation(const json & body) {
    if (!body.contains("truncation") || body.at("truncation").is_null()) {
        return false;
    }
    const json & t = body.at("truncation");
    if (t == "allow") {
        return false;
    }
    if (t == "error") {
        return true;
    }
    throw decision_error(DECISION_REASON_INVALID_REQUEST, "truncation must be \"allow\" or \"error\"", "truncation");
}

bool decision_is_blank(const std::string & s) {
    return decision_py_strip(s).empty();
}

static decision_reason laya_fault_reason(laya_question_fault fault) {
    switch (fault) {
        case LAYA_QF_NOT_OBJECT:
        case LAYA_QF_TYPE:            return DECISION_REASON_UNKNOWN_QUESTION_TYPE;
        case LAYA_QF_INSTRUCTIONS:    return DECISION_REASON_EMPTY_INSTRUCTIONS;
        case LAYA_QF_NO_OPTIONS:      return DECISION_REASON_TOO_FEW_OPTIONS;
        case LAYA_QF_NOUL_CRITERIA:
        case LAYA_QF_LABELS:          return DECISION_REASON_INVALID_NOUL_CRITERIA;
        case LAYA_QF_CRITERIA:
        case LAYA_QF_LABELS_NOT_NOUL:
        default:                      return DECISION_REASON_UNSUPPORTED_CRITERIA_VALUE;
    }
}

// layout laya: the reference semantics, plus what a JSON response and the engine need
static decision_question normalize_laya(const std::string & id, const json & q, const decision_limits & lim, const std::string & param) {
    laya_question lq;
    try {
        lq = laya_question_parse(id, q);
    } catch (const laya_question_error & e) {
        throw decision_error(laya_fault_reason(e.fault), e.what(), e.field.empty() ? param : param + "." + e.field);
    }
    decision_question out;
    out.id           = id;
    out.type         = (decision_qtype) lq.qtype;
    out.instructions = lq.instructions;
    out.criteria     = lq.criteria;
    out.keys         = lq.keys;
    out.labels       = lq.labels;
    out.options      = lq.options;
    const std::string cparam = param + ".criteria";
    if ((int32_t) out.keys.size() > lim.max_options) {
        throw decision_error(DECISION_REASON_TOO_MANY_OPTIONS,
                             "at most " + std::to_string(lim.max_options) + " options are supported, got " + std::to_string(out.keys.size()), cparam);
    }
    // the reference answers with a Python dict that json.dumps may write with a repeated
    // key ("1" and 1, "true" and true); a JSON object here cannot hold both
    std::set<std::string> seen;
    for (const auto & k : out.keys) {
        if (!seen.insert(k).second) {
            throw decision_error(DECISION_REASON_UNSUPPORTED_CRITERIA_VALUE,
                                 "question '" + id + "': two choice labels have the same answer key \"" + k + "\"", cparam);
        }
    }
    return out;
}

// layout clef: question semantics of joint_schema_model.py (model repo). instructions are optional (the
// question id when missing, null or "") and any JSON value is rendered as compact JSON; one choice or score
// option is enough; noul criteria replace the built-in true/false descriptions. Choice criteria may also be
// a list (TypeSafe: dict.fromkeys(str(c))).
static decision_question normalize_clef(const std::string & id, const json & q, const decision_limits & lim, const std::string & param) {
    if (!q.is_object()) {
        throw decision_error(DECISION_REASON_UNKNOWN_QUESTION_TYPE, "question must be an object", param);
    }
    decision_question out;
    out.id = id;
    const json type = q.contains("type") ? q.at("type") : json();
    if (!type.is_string() || !decision_qtype_from_name(type.get<std::string>(), out.type)) {
        throw decision_error(DECISION_REASON_UNKNOWN_QUESTION_TYPE, "unknown question type " + decision_py_dumps(type), param + ".type");
    }
    const json instructions = q.contains("instructions") ? q.at("instructions") : json();
    if (instructions.is_null() || instructions == "") {
        out.instructions = id;
    } else if (instructions.is_string()) {
        out.instructions = instructions.get<std::string>();
    } else {
        out.instructions = decision_py_dumps_compact_sorted(instructions);
    }
    if (out.instructions.empty()) {
        throw decision_error(DECISION_REASON_EMPTY_INSTRUCTIONS, "question has no instructions and an empty id", param + ".instructions");
    }

    const json criteria = q.contains("criteria") ? q.at("criteria") : json();
    const std::string cparam = param + ".criteria";
    switch (out.type) {
        case DECISION_QTYPE_NOUL:
            if (!criteria.is_null() && !criteria.is_object()) {
                throw decision_error(DECISION_REASON_INVALID_NOUL_CRITERIA, "noul criteria must be an object with true/false descriptions", cparam);
            }
            out.criteria = criteria;
            out.keys = { "false", "true" };
            break;
        case DECISION_QTYPE_CHOICE:
            if (criteria.is_array()) {
                json keys = json::object();
                for (const auto & c : criteria) {
                    std::string k;
                    if (!decision_py_str(c, k)) {
                        throw decision_error(DECISION_REASON_UNSUPPORTED_CRITERIA_VALUE, "choice criteria list entries must be strings, numbers, booleans or null", cparam);
                    }
                    if (!keys.contains(k)) {
                        keys[k] = nullptr;
                    }
                }
                out.criteria = std::move(keys);
            } else if (criteria.is_object()) {
                out.criteria = criteria;
            }
            if (!out.criteria.is_object() || out.criteria.empty()) {
                throw decision_error(DECISION_REASON_TOO_FEW_OPTIONS, "choice criteria must name at least one option", cparam);
            }
            for (auto it = out.criteria.begin(); it != out.criteria.end(); ++it) {
                out.keys.push_back(it.key());
            }
            break;
        case DECISION_QTYPE_SCORE:
            if (!criteria.is_array() || criteria.empty()) {
                throw decision_error(criteria.is_object() ? DECISION_REASON_UNSUPPORTED_CRITERIA_VALUE : DECISION_REASON_TOO_FEW_OPTIONS,
                                     "score criteria must list at least one level", cparam);
            }
            out.criteria = criteria;
            for (size_t i = 0; i < criteria.size(); ++i) {
                out.keys.push_back(std::to_string(i));
            }
            break;
    }
    if ((int32_t) out.keys.size() > lim.max_options) {
        throw decision_error(DECISION_REASON_TOO_MANY_OPTIONS,
                             "at most " + std::to_string(lim.max_options) + " options are supported, got " + std::to_string(out.keys.size()), cparam);
    }
    return out;
}

decision_question decision_normalize_question(const std::string & id, const json & q, const decision_limits & lim, const std::string & param) {
    if (lim.layout == "laya") {
        return normalize_laya(id, q, lim, param);
    }
    if (lim.layout == "clef") {
        return normalize_clef(id, q, lim, param);
    }
    if (!q.is_object()) {
        throw decision_error(DECISION_REASON_UNKNOWN_QUESTION_TYPE, "question must be an object", param);
    }
    decision_question out;
    out.id = id;
    const json type = q.contains("type") ? q.at("type") : json();
    if (!type.is_string() || !decision_qtype_from_name(type.get<std::string>(), out.type)) {
        throw decision_error(DECISION_REASON_UNKNOWN_QUESTION_TYPE, "unknown question type " + decision_py_dumps(type), param + ".type");
    }
    if (!q.contains("instructions") || !q.at("instructions").is_string() || decision_is_blank(q.at("instructions").get<std::string>())) {
        throw decision_error(DECISION_REASON_EMPTY_INSTRUCTIONS, "question is missing instructions", param + ".instructions");
    }
    out.instructions = q.at("instructions").get<std::string>();

    const json criteria = q.contains("criteria") ? q.at("criteria") : json();
    const std::string cparam = param + ".criteria";
    switch (out.type) {
        case DECISION_QTYPE_NOUL:
            if (!criteria.is_null() && !criteria.is_object()) {
                throw decision_error(DECISION_REASON_INVALID_NOUL_CRITERIA, "noul criteria must be an object with true/false descriptions", cparam);
            }
            out.criteria = criteria;
            out.keys = { "false", "true" };
            break;
        case DECISION_QTYPE_CHOICE:
            if (criteria.is_array()) {
                json keys = json::object();
                for (const auto & c : criteria) {
                    std::string k;
                    if (!decision_py_str(c, k)) {
                        throw decision_error(DECISION_REASON_UNSUPPORTED_CRITERIA_VALUE, "choice criteria list entries must be strings, numbers, booleans or null", cparam);
                    }
                    if (!keys.contains(k)) {
                        keys[k] = nullptr;
                    }
                }
                out.criteria = std::move(keys);
            } else if (criteria.is_object()) {
                out.criteria = criteria;
            } else {
                throw decision_error(DECISION_REASON_TOO_FEW_OPTIONS, "choice criteria must name at least two options", cparam);
            }
            if (out.criteria.size() < 2) {
                throw decision_error(DECISION_REASON_TOO_FEW_OPTIONS, "choice criteria must name at least two options", cparam);
            }
            for (auto it = out.criteria.begin(); it != out.criteria.end(); ++it) {
                out.keys.push_back(it.key());
            }
            break;
        case DECISION_QTYPE_SCORE:
            if (!criteria.is_array()) {
                throw decision_error(criteria.is_object() ? DECISION_REASON_UNSUPPORTED_CRITERIA_VALUE : DECISION_REASON_TOO_FEW_OPTIONS,
                                     "score criteria must list at least two levels", cparam);
            }
            if (criteria.size() < 2) {
                throw decision_error(DECISION_REASON_TOO_FEW_OPTIONS, "score criteria must list at least two levels", cparam);
            }
            out.criteria = criteria;
            for (size_t i = 0; i < criteria.size(); ++i) {
                out.keys.push_back(std::to_string(i));
            }
            break;
    }
    if ((int32_t) out.keys.size() > lim.max_options) {
        throw decision_error(DECISION_REASON_TOO_MANY_OPTIONS,
                             "at most " + std::to_string(lim.max_options) + " options are supported, got " + std::to_string(out.keys.size()), cparam);
    }
    return out;
}

decision_request decision_parse_systemone(const json & body, const decision_limits & lim) {
    if (!body.is_object()) {
        throw decision_error(DECISION_REASON_BODY_NOT_OBJECT, "request body must be a JSON object");
    }
    decision_request req;
    if (body.contains("model") && body.at("model").is_string()) {
        req.model = body.at("model").get<std::string>();
    }
    req.state = body.contains("state") ? body.at("state") : json();
    req.truncation_error = decision_parse_truncation(body);

    const bool laya = lim.layout == "laya";
    if (laya && req.state.is_null()) {
        // reference serve.py: serialize_state(None) would be the text "null"
        throw decision_error(DECISION_REASON_INVALID_REQUEST, "state is required", "state");
    }
    if (lim.layout == "clef" && !body.contains("state")) {
        // reference systemone(): a null state is the text "null", a missing one is an error
        throw decision_error(DECISION_REASON_INVALID_REQUEST, "state is required", "state");
    }
    // laya: an empty questions object gets empty answers (reference predict_batch)
    if (!body.contains("questions") || !body.at("questions").is_object() || (!laya && body.at("questions").empty())) {
        throw decision_error(DECISION_REASON_INVALID_REQUEST, laya ? "questions must be an object" : "questions must be a non-empty object", "questions");
    }
    const json & qs = body.at("questions");
    if ((int32_t) qs.size() > lim.max_items) {
        throw decision_error(DECISION_REASON_TOO_MANY_QUESTIONS,
                             "at most " + std::to_string(lim.max_items) + " questions per request, got " + std::to_string(qs.size()), "questions");
    }
    for (auto it = qs.begin(); it != qs.end(); ++it) {
        req.questions.push_back(decision_normalize_question(it.key(), it.value(), lim, "questions." + it.key()));
    }
    return req;
}

std::vector<decision_item> decision_request_items(const decision_request & req) {
    std::vector<decision_item> items;
    items.reserve(req.questions.size());
    for (const auto & q : req.questions) {
        decision_item item;
        item.q     = q;
        item.state = req.state;
        items.push_back(std::move(item));
    }
    return items;
}

decision_render_check decision_request_render_check(const decision_request & req) {
    const bool strict = req.truncation_error;
    return [strict, &req](size_t idx, const decision_output & r) {
        if (!strict) {
            return;
        }
        const std::string param = "questions." + req.questions[idx].id;
        if (r.options_cut) {
            throw decision_error(DECISION_REASON_OPTIONS_TRUNCATED, "option texts were cut to fit the model input", param + ".criteria");
        }
        if (r.head_cut) {
            throw decision_error(DECISION_REASON_PROMPT_TOO_LONG, "instructions were cut to fit the model input", param + ".instructions");
        }
        if (r.n_state_cut > 0) {
            throw decision_error(DECISION_REASON_STATE_TRUNCATED, "state was cut by " + std::to_string(r.n_state_cut) + " tokens", "state");
        }
    };
}
