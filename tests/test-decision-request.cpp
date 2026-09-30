// /v1/systemone request checks: reason codes, normalization, limits

#include "decision.h"
#include "decision-calib.h"
#include "decision-request.h"

#include <cstdio>
#include <string>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static decision_limits lim(int32_t max_items = 16, int32_t max_options = 20) {
    decision_limits l;
    l.max_items   = max_items;
    l.max_options = max_options;
    return l;
}

// reason of the decision_error thrown by fn, or "OK"
template <typename F>
static std::string reason_of(F fn) {
    try {
        fn();
    } catch (const decision_error & e) {
        return decision_reason_name(e.reason);
    }
    return "OK";
}

static std::string systemone_reason(const std::string & body_text, const decision_limits & l = lim()) {
    return reason_of([&]() { decision_parse_systemone(decision_parse_body(body_text), l); });
}

static std::string question_reason(const std::string & q_text, const decision_limits & l = lim()) {
    return systemone_reason("{\"state\": \"x\", \"questions\": {\"decision\": " + q_text + "}}", l);
}

#define CHECK_REASON(got, want) do { const std::string g_ = (got); if (g_ != (want)) { fprintf(stderr, "%s:%d: reason %s, want %s\n", __FILE__, __LINE__, g_.c_str(), want); n_fail++; } } while (0)

static void test_body() {
    CHECK_REASON(reason_of([]() { decision_parse_body(std::string(DECISION_MAX_BODY_BYTES + 1, ' ')); }), "BODY_TOO_LARGE");
    CHECK_REASON(reason_of([]() { decision_parse_body("{\"a\":"); }), "MALFORMED_JSON");
    CHECK_REASON(reason_of([]() { decision_parse_body("{\"a\": NaN}"); }), "UNSUPPORTED_NUMBER");
    CHECK_REASON(reason_of([]() { decision_parse_body("{\"a\": 99999999999999999999}"); }), "UNSUPPORTED_NUMBER");
    CHECK_REASON(reason_of([]() { decision_parse_body("[1, 2]"); }), "BODY_NOT_OBJECT");
    CHECK_REASON(reason_of([]() { decision_parse_body("\"x\""); }), "BODY_NOT_OBJECT");
    CHECK(decision_reason_status(DECISION_REASON_BODY_TOO_LARGE) == 413);
    CHECK(decision_reason_status(DECISION_REASON_STATE_TRUNCATED) == 422);
    CHECK(decision_reason_status(DECISION_REASON_OVERLOADED) == 429);
    CHECK(decision_reason_status(DECISION_REASON_ROUTER_NOT_CALIBRATED) == 501);
}

// the four malformed requests of atomic-arbiter tools/contract_test.py (G6) must be 400
static void test_g6() {
    std::string o17 = "{";
    for (int i = 0; i < 17; ++i) {
        o17 += (i ? ", " : "") + std::string("\"o") + std::to_string(i) + "\": \"d\"";
    }
    o17 += "}";
    const std::string r1 = question_reason("{\"type\": \"vote\", \"instructions\": \"x\"}");
    const std::string r2 = question_reason("{\"type\": \"noul\"}");
    const std::string r3 = question_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": " + o17 + "}", lim(16, 16));
    const std::string r4 = question_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": {\"only\": \"d\"}}");
    CHECK_REASON(r1, "UNKNOWN_QUESTION_TYPE");
    CHECK_REASON(r2, "EMPTY_INSTRUCTIONS");
    CHECK_REASON(r3, "TOO_MANY_OPTIONS");
    CHECK_REASON(r4, "TOO_FEW_OPTIONS");
    const decision_reason rs[] = { DECISION_REASON_UNKNOWN_QUESTION_TYPE, DECISION_REASON_EMPTY_INSTRUCTIONS, DECISION_REASON_TOO_MANY_OPTIONS, DECISION_REASON_TOO_FEW_OPTIONS };
    for (auto r : rs) {
        CHECK(decision_reason_status(r) == 400);
    }
    // laya allows 20 options: 17 pass, 21 do not
    CHECK_REASON(question_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": " + o17 + "}", lim(16, 20)), "OK");
}

static void test_questions() {
    CHECK_REASON(question_reason("[1]"), "UNKNOWN_QUESTION_TYPE");
    CHECK_REASON(question_reason("{\"type\": 1, \"instructions\": \"x\"}"), "UNKNOWN_QUESTION_TYPE");
    CHECK_REASON(question_reason("{\"type\": \"noul\", \"instructions\": \" \\t\\n\\u3000\\u00a0\"}"), "EMPTY_INSTRUCTIONS");
    CHECK_REASON(question_reason("{\"type\": \"noul\", \"instructions\": 5}"), "EMPTY_INSTRUCTIONS");
    CHECK_REASON(question_reason("{\"type\": \"noul\", \"instructions\": \"x\", \"criteria\": \"yes\"}"), "INVALID_NOUL_CRITERIA");
    CHECK_REASON(question_reason("{\"type\": \"noul\", \"instructions\": \"x\", \"criteria\": [\"a\"]}"), "INVALID_NOUL_CRITERIA");
    CHECK_REASON(question_reason("{\"type\": \"noul\", \"instructions\": \"x\", \"criteria\": null}"), "OK");
    CHECK_REASON(question_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": [\"a\", \"a\"]}"), "TOO_FEW_OPTIONS");
    CHECK_REASON(question_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": \"a\"}"), "TOO_FEW_OPTIONS");
    CHECK_REASON(question_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": [{\"k\": 1}, \"b\"]}"), "UNSUPPORTED_CRITERIA_VALUE");
    CHECK_REASON(question_reason("{\"type\": \"score\", \"instructions\": \"x\", \"criteria\": {\"a\": 1, \"b\": 2}}"), "UNSUPPORTED_CRITERIA_VALUE");
    CHECK_REASON(question_reason("{\"type\": \"score\", \"instructions\": \"x\", \"criteria\": [\"only\"]}"), "TOO_FEW_OPTIONS");
    CHECK_REASON(question_reason("{\"type\": \"score\", \"instructions\": \"x\"}"), "TOO_FEW_OPTIONS");

    CHECK_REASON(systemone_reason("{\"state\": \"x\"}"), "INVALID_REQUEST");
    CHECK_REASON(systemone_reason("{\"questions\": {}}"), "INVALID_REQUEST");
    CHECK_REASON(systemone_reason("{\"questions\": []}"), "INVALID_REQUEST");
    CHECK_REASON(systemone_reason("{\"truncation\": \"cut\", \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"x\"}}}"), "INVALID_REQUEST");
    CHECK_REASON(systemone_reason("{\"questions\": {\"a\": {\"type\": \"noul\", \"instructions\": \"x\"}, \"b\": {\"type\": \"noul\", \"instructions\": \"y\"}}}", lim(1)),
                 "TOO_MANY_QUESTIONS");
}

static void test_normalize() {
    const decision_request req = decision_parse_systemone(decision_parse_body(
        "{\"model\": \"m\", \"state\": {\"k\": 1.0}, \"truncation\": \"error\", \"questions\": {"
        "\"z\": {\"type\": \"choice\", \"instructions\": \"pick\", \"criteria\": [\"b\", 1, 1.0, true, null, \"b\", \"1\"]},"
        "\"a\": {\"type\": \"score\", \"instructions\": \"rate\", \"criteria\": [\"bad\", 2, \"good\"]},"
        "\"m\": {\"type\": \"noul\", \"instructions\": \"yes?\"},"
        "\"c\": {\"type\": \"choice\", \"instructions\": \"route\", \"criteria\": {\"x\": \"desc\", \"y\": null}}}}"), lim());
    CHECK(req.model == "m");
    CHECK(req.truncation_error);
    CHECK(decision_py_dumps(req.state) == "{\"k\": 1.0}");
    CHECK(req.questions.size() == 4);
    // request order kept
    CHECK(req.questions[0].id == "z" && req.questions[1].id == "a" && req.questions[2].id == "m" && req.questions[3].id == "c");

    const decision_question & z = req.questions[0];
    CHECK(z.type == DECISION_QTYPE_CHOICE);
    CHECK((z.keys == std::vector<std::string>{ "b", "1", "1.0", "True", "None" }));
    CHECK(decision_py_dumps(z.criteria) == "{\"b\": null, \"1\": null, \"1.0\": null, \"True\": null, \"None\": null}");

    const decision_question & a = req.questions[1];
    CHECK(a.type == DECISION_QTYPE_SCORE);
    CHECK((a.keys == std::vector<std::string>{ "0", "1", "2" }));
    CHECK(decision_py_dumps(a.criteria) == "[\"bad\", 2, \"good\"]");

    const decision_question & m = req.questions[2];
    CHECK(m.type == DECISION_QTYPE_NOUL);
    CHECK((m.keys == std::vector<std::string>{ "false", "true" }));
    CHECK(m.criteria.is_null());

    const decision_question & c = req.questions[3];
    CHECK((c.keys == std::vector<std::string>{ "x", "y" }));

    const auto items = decision_request_items(req);
    CHECK(items.size() == 4 && items[3].q.id == "c" && items[3].state == req.state);

    // "truncation": "error" turns cuts into 422
    const auto check = decision_request_render_check(req);
    decision_output r;
    CHECK_REASON(reason_of([&]() { check(0, r); }), "OK");
    r.n_state_cut = 3;
    CHECK_REASON(reason_of([&]() { check(0, r); }), "STATE_TRUNCATED");
    r.options_cut = true;
    CHECK_REASON(reason_of([&]() { check(0, r); }), "OPTIONS_TRUNCATED");

    decision_request loose = req;
    loose.truncation_error = false;
    const auto check_loose = decision_request_render_check(loose);
    CHECK_REASON(reason_of([&]() { check_loose(0, r); }), "OK");
}

// layout laya: the Laya reference question semantics (laya_question_parse)
static decision_limits lim_laya(int32_t max_items = 16, int32_t max_options = 20) {
    decision_limits l = lim(max_items, max_options);
    l.layout = "laya";
    return l;
}

static std::string laya_reason(const std::string & q_text, int32_t max_options = 20) {
    return question_reason(q_text, lim_laya(16, max_options));
}

// param of the decision_error thrown for a laya question, or "OK"
static std::string laya_param(const std::string & q_text) {
    try {
        decision_parse_systemone(decision_parse_body("{\"state\": \"x\", \"questions\": {\"q\": " + q_text + "}}"), lim_laya());
    } catch (const decision_error & e) {
        return e.param;
    }
    return "OK";
}

static decision_question laya_q(const std::string & q_text) {
    return decision_parse_systemone(decision_parse_body("{\"state\": \"x\", \"questions\": {\"q\": " + q_text + "}}"), lim_laya()).questions[0];
}

static void test_laya() {
    // G6 under the reference: one choice option is a valid question there
    std::string o17 = "{";
    for (int i = 0; i < 17; ++i) {
        o17 += (i ? ", " : "") + std::string("\"o") + std::to_string(i) + "\": \"d\"";
    }
    o17 += "}";
    CHECK_REASON(laya_reason("{\"type\": \"vote\", \"instructions\": \"x\"}"), "UNKNOWN_QUESTION_TYPE");
    CHECK_REASON(laya_reason("{\"type\": \"noul\"}"), "EMPTY_INSTRUCTIONS");
    CHECK_REASON(laya_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": " + o17 + "}", 16), "TOO_MANY_OPTIONS");
    CHECK_REASON(laya_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": {\"only\": \"d\"}}"), "OK");
    CHECK_REASON(laya_reason("[1]"), "UNKNOWN_QUESTION_TYPE");

    // instructions: present is enough; a non-string is json.dumps'd
    CHECK_REASON(laya_reason("{\"type\": \"noul\", \"instructions\": \" \\t\\u2028\\u0085\"}"), "OK");
    CHECK(laya_q("{\"type\": \"noul\", \"instructions\": 5}").instructions == "5");
    CHECK(laya_q("{\"type\": \"noul\", \"instructions\": null}").instructions == "null");
    CHECK(laya_q("{\"type\": \"noul\", \"instructions\": {\"q\": \"Hostile?\", \"w\": [1.0, true]}}").instructions == "{\"q\": \"Hostile?\", \"w\": [1.0, true]}");

    // state is required; an empty questions object is a valid (empty) request
    CHECK_REASON(systemone_reason("{\"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"x\"}}}", lim_laya()), "INVALID_REQUEST");
    CHECK_REASON(systemone_reason("{\"state\": null, \"questions\": {\"q\": {\"type\": \"noul\", \"instructions\": \"x\"}}}", lim_laya()), "INVALID_REQUEST");
    CHECK_REASON(systemone_reason("{\"state\": \"x\", \"questions\": {}}", lim_laya()), "OK");
    CHECK(decision_parse_systemone(decision_parse_body("{\"state\": \"x\", \"questions\": {}}"), lim_laya()).questions.empty());
    CHECK_REASON(systemone_reason("{\"state\": \"x\", \"questions\": []}", lim_laya()), "INVALID_REQUEST");
    CHECK_REASON(systemone_reason("{\"state\": \"x\"}", lim_laya()), "INVALID_REQUEST");

    // choice list: labels as given, str() option text, json.dumps answer keys; no repeats by Python ==
    const decision_question c = laya_q("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": [1, 2.5, false, \"x\", 1e300]}");
    CHECK((c.keys == std::vector<std::string>{ "1", "2.5", "false", "x", "1e+300" }));
    CHECK((c.options == std::vector<std::string>{ "1", "2.5", "False", "x", "1e+300" }));
    CHECK(decision_py_dumps(c.labels) == "[1, 2.5, false, \"x\", 1e+300]");
    CHECK(decision_py_dumps(c.criteria) == "[1, 2.5, false, \"x\", 1e+300]");
    const char * rejected[] = { "[\"a\", \"a\"]", "[1, 1.0]", "[true, 1]", "[0, false]", "[0, -0.0]", "[\"a\", null]", "[\"a\", [\"b\"]]",
                                "[\"a\", {\"b\": 1}]", "[\"1\", 1]", "[\"true\", true]" };
    for (const char * r : rejected) {
        CHECK_REASON(laya_reason(std::string("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": ") + r + "}"), "UNSUPPORTED_CRITERIA_VALUE");
    }
    CHECK_REASON(laya_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": [9007199254740993, 9007199254740992.0]}"), "OK");
    CHECK_REASON(laya_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": [\"only\"]}"), "OK");
    CHECK_REASON(laya_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": []}"), "TOO_FEW_OPTIONS");
    CHECK_REASON(laya_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": {}}"), "TOO_FEW_OPTIONS");
    CHECK_REASON(laya_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": \"a\"}"), "TOO_FEW_OPTIONS");
    CHECK_REASON(laya_reason("{\"type\": \"choice\", \"instructions\": \"x\"}"), "TOO_FEW_OPTIONS");
    const decision_question cd = laya_q("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": {\"a\": 0, \"b\": \"\", \"c\": null, \"d\": {\"k\": [1, 2]}}}");
    CHECK((cd.options == std::vector<std::string>{ "a: 0", "b", "c", "d: {\"k\": [1, 2]}" }));
    CHECK((cd.keys == std::vector<std::string>{ "a", "b", "c", "d" }));

    // score: one level is enough; a null level is not
    CHECK_REASON(laya_reason("{\"type\": \"score\", \"instructions\": \"x\", \"criteria\": [\"only\"]}"), "OK");
    CHECK_REASON(laya_reason("{\"type\": \"score\", \"instructions\": \"x\", \"criteria\": [\"a\", null]}"), "UNSUPPORTED_CRITERIA_VALUE");
    CHECK_REASON(laya_reason("{\"type\": \"score\", \"instructions\": \"x\", \"criteria\": {\"a\": 1}}"), "UNSUPPORTED_CRITERIA_VALUE");
    CHECK_REASON(laya_reason("{\"type\": \"score\", \"instructions\": \"x\", \"criteria\": []}"), "TOO_FEW_OPTIONS");
    CHECK((laya_q("{\"type\": \"score\", \"instructions\": \"x\", \"criteria\": [\"lo\", 2]}").options == std::vector<std::string>{ "level 0: lo", "level 1: 2" }));

    // noul: keys lowercased (a later duplicate wins), only true/false; labels
    const decision_question n = laya_q("{\"type\": \"noul\", \"instructions\": \"x\", \"criteria\": {\"TRUE\": \"first\", \"False\": \"\", \"true\": \"today\"}}");
    CHECK(decision_py_dumps(n.criteria) == "{\"true\": \"today\", \"false\": \"\"}");
    CHECK((n.options == std::vector<std::string>{ "false: no, the statement does not hold", "true: today" }));
    CHECK_REASON(laya_reason("{\"type\": \"noul\", \"instructions\": \"x\", \"criteria\": {\"true\": \"a\", \"maybe\": \"b\"}}"), "INVALID_NOUL_CRITERIA");
    CHECK(laya_param("{\"type\": \"noul\", \"instructions\": \"x\", \"criteria\": {\"maybe\": \"b\"}}") == "questions.q.criteria");
    const decision_question nl = laya_q("{\"type\": \"noul\", \"instructions\": \"x\", \"labels\": {\"false\": \" calm\\u2028\", \"true\": \"angry\"}}");
    CHECK((nl.options == std::vector<std::string>{ "calm: no, the statement does not hold", "angry: yes, the statement holds" }));
    CHECK((nl.keys == std::vector<std::string>{ "false", "true" }));
    CHECK_REASON(laya_reason("{\"type\": \"noul\", \"instructions\": \"x\", \"labels\": null}"), "OK");
    const char * bad_labels[] = { "{\"False\": \"n\", \"true\": \"y\"}", "{\"false\": \"y\", \"true\": \" y\"}", "{\"false\": \" \", \"true\": \"y\"}",
                                  "{\"false\": 0, \"true\": 1}", "{\"false\": \"n\", \"true\": \"y\", \"maybe\": \"m\"}", "[\"n\", \"y\"]" };
    for (const char * b : bad_labels) {
        CHECK_REASON(laya_reason(std::string("{\"type\": \"noul\", \"instructions\": \"x\", \"labels\": ") + b + "}"), "INVALID_NOUL_CRITERIA");
    }
    CHECK(laya_param("{\"type\": \"noul\", \"instructions\": \"x\", \"labels\": []}") == "questions.q.labels");
    CHECK_REASON(laya_reason("{\"type\": \"choice\", \"instructions\": \"x\", \"criteria\": [\"a\", \"b\"], \"labels\": null}"), "UNSUPPORTED_CRITERIA_VALUE");
    CHECK(laya_param("{\"type\": \"score\", \"instructions\": \"x\", \"criteria\": [\"a\"], \"labels\": {\"false\": \"n\", \"true\": \"y\"}}") == "questions.q.labels");

    // the answer names the label as given
    const json ans = decision_answer(c, { 0.1, 0.6, 0.1, 0.1, 0.1 }, DECISION_CONFIDENCE_LAYA);
    CHECK(decision_py_dumps(ans.at("choice")) == "2.5");
    CHECK(decision_py_dumps(ans.at("probabilities")) == "{\"1\": 0.1, \"2.5\": 0.6, \"false\": 0.1, \"x\": 0.1, \"1e+300\": 0.1}");
    CHECK(decision_answer(c, { 0.1, 0.1, 0.6, 0.1, 0.1 }, DECISION_CONFIDENCE_LAYA).at("choice") == json(false));
}

static void test_envelope() {
    const decision_error e(DECISION_REASON_TOO_MANY_OPTIONS, "too many", "questions.q.criteria");
    CHECK(decision_py_dumps(decision_error_body(e)) ==
          "{\"error\": {\"code\": 400, \"type\": \"invalid_request_error\", \"reason\": \"TOO_MANY_OPTIONS\", \"message\": \"too many\", \"param\": \"questions.q.criteria\"}}");
    const decision_error e2(DECISION_REASON_OVERLOADED, "busy");
    CHECK(decision_py_dumps(decision_error_body(e2)) ==
          "{\"error\": {\"code\": 429, \"type\": \"unavailable_error\", \"reason\": \"OVERLOADED\", \"message\": \"busy\"}}");
    for (int r = 0; r < DECISION_REASON_COUNT; ++r) {
        CHECK(std::string(decision_reason_name((decision_reason) r)).size() > 0);
    }
}

int main() {
    test_body();
    test_g6();
    test_questions();
    test_normalize();
    test_laya();
    test_envelope();
    CHECK(decision_is_blank(""));
    CHECK(decision_is_blank(" \xe3\x80\x80\xc2\x85"));
    CHECK(!decision_is_blank(" a "));
    CHECK(!decision_is_blank("\xe4\xb8\xad"));
    CHECK(decision_py_strip("\xe2\x80\xa8 a b\xc2\x85\t") == "a b");
    CHECK(decision_py_strip("\xe4\xb8\xad") == "\xe4\xb8\xad");
    if (n_fail) {
        fprintf(stderr, "test-decision-request: %d failures\n", n_fail);
        return 1;
    }
    printf("test-decision-request: OK\n");
    return 0;
}
