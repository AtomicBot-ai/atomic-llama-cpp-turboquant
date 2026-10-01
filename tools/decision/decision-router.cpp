#include "decision-router.h"

#include <cstdint>
#include <set>

static bool is_json_int(const json & v) {
    return v.is_number_integer() || v.is_number_unsigned();
}

static std::string canon_line(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    bool space = false;
    for (unsigned char c : s) {
        if (c < 0x20 || c == 0x7f || c == ' ') {
            space = !out.empty();
            continue;
        }
        if (space) {
            out += ' ';
            space = false;
        }
        out += (char) c;
    }
    return out;
}

[[noreturn]] static void card_fail(const std::string & param, const std::string & msg) {
    throw decision_error(DECISION_REASON_INVALID_CARD, msg, param);
}

static void require_string(const json & obj, const char * key, const std::string & param, bool required) {
    if (!obj.contains(key)) {
        if (required) {
            card_fail(param + "." + key, std::string(key) + " is required");
        }
        return;
    }
    const json & v = obj.at(key);
    if (!v.is_string() || (required && canon_line(v.get<std::string>()).empty())) {
        card_fail(param + "." + key, std::string(key) + (required ? " must be a non-empty string" : " must be a string"));
    }
    if (v.get<std::string>().size() > DECISION_MAX_CARD_FIELD_BYTES) {
        card_fail(param + "." + key, std::string(key) + " is longer than " + std::to_string(DECISION_MAX_CARD_FIELD_BYTES) + " bytes");
    }
}

void decision_card_validate(const json & card, const std::string & param, std::vector<std::string> & warnings) {
    if (!card.is_object()) {
        card_fail(param, "card must be an object");
    }
    if (!card.contains("schema") || card.at("schema") != DECISION_CARD_SCHEMA) {
        card_fail(param + ".schema", "card schema must be \"" DECISION_CARD_SCHEMA "\"");
    }
    require_string(card, "name", param, true);
    require_string(card, "kind", param, true);
    require_string(card, "description", param, false);

    static const std::set<std::string> card_keys  = { "schema", "name", "kind", "description", "checks" };
    static const std::set<std::string> check_keys = { "skill", "status", "passed", "total", "criterion", "source", "version" };
    for (auto it = card.begin(); it != card.end(); ++it) {
        if (!card_keys.count(it.key())) {
            warnings.push_back(param + "." + it.key() + ": unknown field ignored");
        }
    }

    if (!card.contains("checks")) {
        return;
    }
    const json & checks = card.at("checks");
    if (!checks.is_array()) {
        card_fail(param + ".checks", "checks must be an array");
    }
    if (checks.size() > DECISION_MAX_CHECKS) {
        card_fail(param + ".checks", "at most " + std::to_string(DECISION_MAX_CHECKS) + " checks per card");
    }
    for (size_t i = 0; i < checks.size(); ++i) {
        const json & c = checks[i];
        const std::string cp = param + ".checks[" + std::to_string(i) + "]";
        if (!c.is_object()) {
            card_fail(cp, "check must be an object");
        }
        require_string(c, "skill", cp, true);
        require_string(c, "criterion", cp, false);
        require_string(c, "source", cp, false);
        require_string(c, "version", cp, false);
        const json status = c.contains("status") ? c.at("status") : json();
        if (status == "measured") {
            if (!c.contains("passed") || !c.contains("total") || !is_json_int(c.at("passed")) || !is_json_int(c.at("total"))) {
                card_fail(cp, "a measured check needs integer passed and total");
            }
            if ((c.at("passed").is_number_unsigned() && c.at("passed").get<uint64_t>() > INT64_MAX) ||
                (c.at("total").is_number_unsigned()  && c.at("total").get<uint64_t>()  > INT64_MAX)) {
                card_fail(cp, "passed and total must fit in int64");
            }
            const int64_t passed = c.at("passed").get<int64_t>();
            const int64_t total  = c.at("total").get<int64_t>();
            if (total < 1 || passed < 0 || passed > total) {
                card_fail(cp, "a measured check needs 0 <= passed <= total and total >= 1");
            }
        } else if (status == "missing") {
            if (c.contains("passed") || c.contains("total")) {
                card_fail(cp, "a missing check has no passed/total");
            }
        } else {
            card_fail(cp + ".status", "status must be \"measured\" or \"missing\"");
        }
        for (auto it = c.begin(); it != c.end(); ++it) {
            if (!check_keys.count(it.key())) {
                warnings.push_back(cp + "." + it.key() + ": unknown field ignored");
            }
        }
    }
}

static std::string opt_field(const json & obj, const char * key) {
    return obj.contains(key) ? canon_line(obj.at(key).get<std::string>()) : "";
}

std::string decision_card_render(const json & card) {
    std::string out;
    out += "executor: " + opt_field(card, "name") + "\n";
    out += "kind: " + opt_field(card, "kind") + "\n";
    const std::string desc = opt_field(card, "description");
    if (!desc.empty()) {
        out += "description: " + desc + "\n";
    }
    const json checks = card.contains("checks") ? card.at("checks") : json::array();
    if (checks.empty()) {
        out += "checks: none";
        return out;
    }
    out += "checks:";
    for (const auto & c : checks) {
        out += "\n- " + opt_field(c, "skill") + ": ";
        if (c.at("status") == "measured") {
            out += "passed " + std::to_string(c.at("passed").get<int64_t>()) + " of " + std::to_string(c.at("total").get<int64_t>());
            const std::string crit = opt_field(c, "criterion");
            if (!crit.empty()) {
                out += "; criterion: " + crit;
            }
        } else {
            out += "not measured";
        }
        const std::string src = opt_field(c, "source");
        const std::string ver = opt_field(c, "version");
        if (!src.empty() || !ver.empty()) {
            out += "; source: " + src;
            if (!ver.empty()) {
                out += (src.empty() ? "" : " ") + ver;
            }
        }
    }
    return out;
}

bool decision_candidate_id_valid(const std::string & id) {
    if (id.empty() || id.size() > 128) {
        return false;
    }
    for (unsigned char c : id) {
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                        c == '.' || c == '_' || c == ':' || c == '/' || c == '@' || c == '+' || c == '-';
        if (!ok) {
            return false;
        }
    }
    return true;
}

static std::string required_text(const json & body, const char * key) {
    if (!body.contains(key) || !body.at(key).is_string() || decision_is_blank(body.at(key).get<std::string>())) {
        throw decision_error(DECISION_REASON_INVALID_REQUEST, std::string(key) + " must be a non-empty string", key);
    }
    return body.at(key).get<std::string>();
}

decision_router_request decision_parse_router(const json & body, int32_t max_candidates) {
    if (!body.is_object()) {
        throw decision_error(DECISION_REASON_BODY_NOT_OBJECT, "request body must be a JSON object");
    }
    decision_router_request req;
    if (body.contains("model") && body.at("model").is_string()) {
        req.model = body.at("model").get<std::string>();
    }
    req.task             = required_text(body, "task");
    req.criterion        = required_text(body, "criterion");
    req.truncation_error = decision_parse_truncation(body);

    if (!body.contains("candidates") || !body.at("candidates").is_array() || body.at("candidates").empty()) {
        throw decision_error(DECISION_REASON_INVALID_REQUEST, "candidates must be a non-empty array", "candidates");
    }
    const json & cands = body.at("candidates");
    if ((int32_t) cands.size() > max_candidates) {
        throw decision_error(DECISION_REASON_TOO_MANY_CANDIDATES,
                             "at most " + std::to_string(max_candidates) + " candidates per request, got " + std::to_string(cands.size()), "candidates");
    }

    std::set<std::string> seen;
    for (size_t i = 0; i < cands.size(); ++i) {
        const json & c = cands[i];
        const std::string param = "candidates[" + std::to_string(i) + "]";
        if (!c.is_object()) {
            throw decision_error(DECISION_REASON_INVALID_CARD, "candidate must be an object", param);
        }
        if (!c.contains("id") || !c.at("id").is_string() || !decision_candidate_id_valid(c.at("id").get<std::string>())) {
            throw decision_error(DECISION_REASON_INVALID_CANDIDATE_ID, "candidate id must match [A-Za-z0-9._:/@+-]{1,128}", param + ".id");
        }
        decision_candidate cand;
        cand.id = c.at("id").get<std::string>();
        if (!seen.insert(cand.id).second) {
            throw decision_error(DECISION_REASON_DUPLICATE_CANDIDATE_ID, "duplicate candidate id '" + cand.id + "'", param + ".id");
        }
        if (!c.contains("card")) {
            throw decision_error(DECISION_REASON_INVALID_CARD, "candidate has no card", param + ".card");
        }
        decision_card_validate(c.at("card"), param + ".card", req.warnings);
        for (auto it = c.begin(); it != c.end(); ++it) {
            if (it.key() != "id" && it.key() != "card") {
                req.warnings.push_back(param + "." + it.key() + ": unknown field ignored");
            }
        }
        cand.card      = c.at("card");
        cand.card_text = decision_card_render(cand.card);
        req.candidates.push_back(std::move(cand));
    }
    return req;
}

std::string decision_router_state_prefix(const std::string & card_text, const std::string & criterion) {
    return card_text + "\n\nsuccess criterion: " + criterion + "\n\ntask:";
}

std::string decision_router_state(const std::string & card_text, const std::string & criterion, const std::string & task) {
    return decision_router_state_prefix(card_text, criterion) + " " + task;
}

std::vector<decision_item> decision_router_items(const decision_router_request & req, const decision_router_spec & router, const decision_limits & lim) {
    const decision_question base = decision_normalize_question("", router.question, lim, "router.question");
    std::vector<decision_item> items;
    items.reserve(req.candidates.size());
    for (size_t i = 0; i < req.candidates.size(); ++i) {
        const decision_candidate & c = req.candidates[i];
        decision_item item;
        item.q              = base;
        item.q.id           = c.id;
        item.param          = "candidates[" + std::to_string(i) + "]";
        item.state_splits   = { c.card_text.size(), decision_router_state_prefix(c.card_text, req.criterion).size() };
        item.state          = decision_router_state(c.card_text, req.criterion, req.task);
        item.escape_control = router.escape_control;
        items.push_back(std::move(item));
    }
    return items;
}

double decision_router_logit(const decision_output & out) {
    return out.logits[1] - out.logits[0];
}

void decision_router_check_cards(const decision_router_request & req, const decision_router_spec & router, const decision_engine & engine) {
    if (router.max_card_tokens <= 0) {
        return;
    }
    for (size_t i = 0; i < req.candidates.size(); ++i) {
        const int32_t n = engine.n_tokens(req.candidates[i].card_text, router.escape_control);
        if (n > router.max_card_tokens) {
            throw decision_error(DECISION_REASON_CARD_TOO_LONG,
                                 "card has " + std::to_string(n) + " tokens, the limit is " + std::to_string(router.max_card_tokens),
                                 "candidates[" + std::to_string(i) + "].card");
        }
    }
}

decision_render_check decision_router_render_check(const decision_router_request & req, const decision_router_spec & router, const decision_engine & engine) {
    const bool escape_control = router.escape_control;
    return [&req, &engine, escape_control](size_t idx, const decision_output & r) {
        if (r.n_state_cut == 0) {
            return;
        }
        const decision_candidate & c = req.candidates[idx];
        // the prefix ends at a word boundary, so its tokens are a prefix of the state tokens
        const int32_t n_prefix = r.n_state_head >= 0 ? r.n_state_head
                               : engine.n_tokens(decision_router_state_prefix(c.card_text, req.criterion), escape_control);
        if (r.n_state - r.n_state_cut < n_prefix) {
            throw decision_error(DECISION_REASON_CRITERION_TOO_LONG,
                                 "card and criterion do not fit in the model input", "candidates[" + std::to_string(idx) + "]");
        }
        if (req.truncation_error) {
            throw decision_error(DECISION_REASON_STATE_TRUNCATED,
                                 "task was cut by " + std::to_string(r.n_state_cut) + " tokens", "task");
        }
    };
}
