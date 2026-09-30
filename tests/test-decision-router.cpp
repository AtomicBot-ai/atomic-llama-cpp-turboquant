// /v1/router/score: executor card checks, card-v1 rendering, expansion into noul items

#include "decision.h"
#include "decision-router.h"
#include "decision-spec.h"

#include <cstdio>
#include <cstring>
#include <string>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)
#define CHECK_REASON(got, want) do { const std::string g_ = (got); if (g_ != (want)) { fprintf(stderr, "%s:%d: reason %s, want %s\n", __FILE__, __LINE__, g_.c_str(), want); n_fail++; } } while (0)

template <typename F>
static std::string reason_of(F fn) {
    try {
        fn();
    } catch (const decision_error & e) {
        return decision_reason_name(e.reason);
    }
    return "OK";
}

static const char * CARD = R"json({"schema":"atomic.executor-card/1","name":"Qwen3.5-4B local","kind":"local",
   "description":"4B general model on this laptop (Q4_K_M)",
   "checks":[{"skill":"field extraction","status":"measured","passed":188,"total":200,"criterion":"exact match","source":"atomic-evals/extract","version":"2026-09"},
             {"skill":"long-document QA","status":"missing","source":"atomic-evals/longqa","version":"2026-09"}]})json";

static const char * CARD_TEXT =
    "executor: Qwen3.5-4B local\n"
    "kind: local\n"
    "description: 4B general model on this laptop (Q4_K_M)\n"
    "checks:\n"
    "- field extraction: passed 188 of 200; criterion: exact match; source: atomic-evals/extract 2026-09\n"
    "- long-document QA: not measured; source: atomic-evals/longqa 2026-09";

static json card_with(const char * patch) {
    json c = json::parse(CARD);
    const json p = json::parse(patch);
    for (auto it = p.begin(); it != p.end(); ++it) {
        if (it->is_null()) {
            c.erase(it.key());
        } else {
            c[it.key()] = it.value();
        }
    }
    return c;
}

static json check_with(const char * patch) {
    json c = json::parse(CARD).at("checks")[0];
    const json p = json::parse(patch);
    for (auto it = p.begin(); it != p.end(); ++it) {
        if (it->is_null()) {
            c.erase(it.key());
        } else {
            c[it.key()] = it.value();
        }
    }
    return c;
}

static std::string card_reason(const json & card) {
    std::vector<std::string> w;
    return reason_of([&]() { decision_card_validate(card, "card", w); });
}

static std::string card_reason_check(const json & check) {
    json card = json::parse(CARD);
    card["checks"] = json::array({ check });
    return card_reason(card);
}

static void test_cards() {
    std::vector<std::string> warnings;
    const json card = json::parse(CARD);
    CHECK_REASON(reason_of([&]() { decision_card_validate(card, "card", warnings); }), "OK");
    CHECK(warnings.empty());
    CHECK(decision_card_render(card) == CARD_TEXT);

    CHECK_REASON(card_reason(json::array()), "INVALID_CARD");
    CHECK_REASON(card_reason(card_with(R"({"schema": "atomic.executor-card/2"})")), "INVALID_CARD");
    CHECK_REASON(card_reason(card_with(R"({"schema": null})")), "INVALID_CARD");
    CHECK_REASON(card_reason(card_with(R"({"name": ""})")), "INVALID_CARD");
    CHECK_REASON(card_reason(card_with(R"({"name": " \n "})")), "INVALID_CARD");
    CHECK_REASON(card_reason(card_with(R"({"kind": null})")), "INVALID_CARD");
    CHECK_REASON(card_reason(card_with(R"({"description": 5})")), "INVALID_CARD");
    CHECK_REASON(card_reason(card_with(R"({"checks": {}})")), "INVALID_CARD");
    {
        json big = json::parse(CARD);
        big["description"] = std::string(DECISION_MAX_CARD_FIELD_BYTES, 'x');
        CHECK_REASON(card_reason(big), "OK");
        big["description"] = std::string(DECISION_MAX_CARD_FIELD_BYTES + 1, 'x');
        CHECK_REASON(card_reason(big), "INVALID_CARD");
        CHECK_REASON(card_reason_check(check_with(("{\"source\": \"" + std::string(DECISION_MAX_CARD_FIELD_BYTES + 1, 's') + "\"}").c_str())), "INVALID_CARD");
    }

    CHECK_REASON(card_reason_check(check_with(R"({"passed": 201})")), "INVALID_CARD");
    CHECK_REASON(card_reason_check(check_with(R"({"passed": -1})")), "INVALID_CARD");
    CHECK_REASON(card_reason_check(check_with(R"({"total": 0, "passed": 0})")), "INVALID_CARD");
    CHECK_REASON(card_reason_check(check_with(R"({"passed": 188.0})")), "INVALID_CARD");
    CHECK_REASON(card_reason_check(check_with(R"({"passed": true})")), "INVALID_CARD");
    CHECK_REASON(card_reason_check(check_with(R"({"total": null})")), "INVALID_CARD");
    CHECK_REASON(card_reason_check(check_with(R"({"status": "estimated"})")), "INVALID_CARD");
    CHECK_REASON(card_reason_check(check_with(R"({"status": "missing"})")), "INVALID_CARD");
    CHECK_REASON(card_reason_check(check_with(R"({"status": "missing", "passed": null, "total": null})")), "OK");
    CHECK_REASON(card_reason_check(check_with(R"({"skill": null})")), "INVALID_CARD");
    CHECK_REASON(card_reason_check(check_with(R"({"source": 1})")), "INVALID_CARD");
    CHECK_REASON(card_reason_check(check_with(R"({"passed": 0, "total": 1})")), "OK");

    json many = json::parse(CARD);
    many["checks"] = json::array();
    for (int i = 0; i < DECISION_MAX_CHECKS + 1; ++i) {
        many["checks"].push_back(check_with("{}"));
    }
    CHECK_REASON(card_reason(many), "INVALID_CARD");

    // unknown fields are ignored with a warning and do not change the text
    warnings.clear();
    json extra = card_with(R"({"price": 1.5})");
    extra["checks"][0]["latency_ms"] = 12;
    decision_card_validate(extra, "card", warnings);
    CHECK(warnings.size() == 2);
    CHECK(decision_card_render(extra) == CARD_TEXT);

    // single-line strings, optional parts, no checks
    const json bare = json::parse(R"({"schema":"atomic.executor-card/1","name":"  GPT\tcloud\n model ","kind":"cloud"})");
    CHECK(decision_card_render(bare) == "executor: GPT cloud model\nkind: cloud\nchecks: none");
    json partial = json::parse(R"({"schema":"atomic.executor-card/1","name":"n","kind":"k","checks":[
        {"skill":"s","status":"measured","passed":3,"total":4},{"skill":"t","status":"missing","version":"v1"}]})");
    CHECK(decision_card_render(partial) == "executor: n\nkind: k\nchecks:\n- s: passed 3 of 4\n- t: not measured; source: v1");
}

static std::string router_body(const std::string & candidates) {
    return R"({"model":"opt","task":"Extract invoice number, date and total.","criterion":"All three fields exact; nothing invented.","candidates":)" +
           candidates + "}";
}

static std::string router_reason(const std::string & text, int32_t max = 16) {
    return reason_of([&]() { decision_parse_router(json::parse(text), max); });
}

static void test_request() {
    const std::string c1 = std::string(R"({"id":"local/qwen3.5-4b-q4_k_m","card":)") + CARD + "}";
    const std::string c2 = std::string(R"({"id":"cloud@provider:model+v2","card":)") + CARD + "}";
    CHECK_REASON(router_reason(router_body("[" + c1 + "," + c2 + "]")), "OK");
    CHECK_REASON(router_reason(router_body("[" + c1 + "," + c1 + "]")), "DUPLICATE_CANDIDATE_ID");
    CHECK_REASON(router_reason(router_body("[" + c1 + "," + c2 + "]"), 1), "TOO_MANY_CANDIDATES");
    CHECK_REASON(router_reason(router_body("[]")), "INVALID_REQUEST");
    CHECK_REASON(router_reason(router_body("{}")), "INVALID_REQUEST");
    CHECK_REASON(router_reason(router_body(std::string(R"([{"id":"has space","card":)") + CARD + "}]")), "INVALID_CANDIDATE_ID");
    CHECK_REASON(router_reason(router_body(std::string(R"([{"id":"","card":)") + CARD + "}]")), "INVALID_CANDIDATE_ID");
    CHECK_REASON(router_reason(router_body(std::string(R"([{"id":7,"card":)") + CARD + "}]")), "INVALID_CANDIDATE_ID");
    CHECK_REASON(router_reason(router_body(R"([{"id":"a"}])")), "INVALID_CARD");
    CHECK_REASON(router_reason(router_body(R"([{"id":"a","card":{"schema":"atomic.executor-card/1"}}])")), "INVALID_CARD");
    CHECK_REASON(router_reason(R"({"task":"","criterion":"c","candidates":[]})"), "INVALID_REQUEST");
    CHECK_REASON(router_reason(R"({"task":"t","candidates":[]})"), "INVALID_REQUEST");
    CHECK_REASON(router_reason("[]"), "BODY_NOT_OBJECT");

    CHECK(decision_candidate_id_valid(std::string(128, 'a')));
    CHECK(!decision_candidate_id_valid(std::string(129, 'a')));
    CHECK(!decision_candidate_id_valid("a\xc3\xa9"));

    const auto req = decision_parse_router(json::parse(router_body("[" + c2 + "," + c1 + "]")), 16);
    CHECK(req.candidates.size() == 2 && req.candidates[0].id == "cloud@provider:model+v2");
    CHECK(req.candidates[1].card_text == CARD_TEXT);
    CHECK(!req.truncation_error && req.warnings.empty());

    decision_router_spec router;
    router.present  = true;
    router.question = json::parse(R"({"type":"noul","instructions":"Will the executor meet the success criterion on this task?",
                                      "criteria":{"true":"meets the criterion","false":"does not meet the criterion"}})");
    decision_limits lim;
    const auto items = decision_router_items(req, router, lim);
    CHECK(items.size() == 2);
    CHECK(items[0].q.id == "cloud@provider:model+v2" && items[1].q.id == "local/qwen3.5-4b-q4_k_m");
    CHECK(items[0].q.type == DECISION_QTYPE_NOUL && items[0].q.instructions == router.question.at("instructions"));
    CHECK((items[0].q.keys == std::vector<std::string>{ "false", "true" }));
    // card first, then the criterion, then the task
    const std::string want = std::string(CARD_TEXT) +
        "\n\nsuccess criterion: All three fields exact; nothing invented.\n\ntask: Extract invoice number, date and total.";
    CHECK(items[1].state == want);
    CHECK(want.rfind(decision_router_state_prefix(CARD_TEXT, req.criterion), 0) == 0);
    // errors name the candidate; the head is the prefix; escaping follows the spec
    CHECK(items[0].param == "candidates[0]" && items[1].param == "candidates[1]");
    CHECK((items[1].state_splits == std::vector<size_t>{ strlen(CARD_TEXT), decision_router_state_prefix(CARD_TEXT, req.criterion).size() }));
    CHECK(!items[1].escape_control);
    router.escape_control = true;
    CHECK(decision_router_items(req, router, lim)[0].escape_control);
    router.escape_control = false;
    // layout laya: the question is rendered by the reference rules
    decision_limits lim_laya;
    lim_laya.layout = "laya";
    CHECK((decision_router_items(req, router, lim_laya)[0].q.options ==
           std::vector<std::string>{ "false: does not meet the criterion", "true: meets the criterion" }));
    CHECK(items[0].q.options.empty());

    decision_output out;
    out.logits = { -0.5, 2.25 };
    CHECK(decision_router_logit(out) == 2.75);
}

// tokens = whitespace-separated words
struct fake_engine : decision_engine {
    decision_caps caps() const override { return decision_caps(); }
    int32_t n_tokens(const std::string & text, bool) const override {
        int32_t n = 0;
        bool in_word = false;
        for (char c : text) {
            const bool ws = c == ' ' || c == '\n' || c == '\t';
            n += !ws && !in_word;
            in_word = !ws;
        }
        return n;
    }
    decision_output render(const decision_item &) const override { return decision_output(); }
    bool evaluate(const std::vector<decision_item> &, bool, const std::atomic<bool> *, const decision_render_check &, std::vector<decision_output> &) override {
        return true;
    }
};

static void test_limits() {
    const std::string c1 = std::string(R"({"id":"a","card":)") + CARD + "}";
    auto req = decision_parse_router(json::parse(router_body("[" + c1 + "]")), 16);
    fake_engine eng;
    decision_router_spec router;
    router.max_card_tokens = 1000;
    CHECK_REASON(reason_of([&]() { decision_router_check_cards(req, router, eng); }), "OK");
    router.max_card_tokens = 10;
    CHECK_REASON(reason_of([&]() { decision_router_check_cards(req, router, eng); }), "CARD_TOO_LONG");
    router.max_card_tokens = 0;
    CHECK_REASON(reason_of([&]() { decision_router_check_cards(req, router, eng); }), "OK");

    const int32_t n_prefix = eng.n_tokens(decision_router_state_prefix(req.candidates[0].card_text, req.criterion), false);
    const int32_t n_state  = eng.n_tokens(decision_router_state(req.candidates[0].card_text, req.criterion, req.task), false);
    CHECK(n_state == n_prefix + 6);

    const auto check = decision_router_render_check(req, router, eng);
    decision_output r;
    r.n_state = n_state;
    CHECK_REASON(reason_of([&]() { check(0, r); }), "OK");
    r.n_state_cut = 6;    // only the task is cut
    CHECK_REASON(reason_of([&]() { check(0, r); }), "OK");
    r.n_state_cut = 7;    // the cut reaches the criterion
    CHECK_REASON(reason_of([&]() { check(0, r); }), "CRITERION_TOO_LONG");

    req.truncation_error = true;
    const auto strict = decision_router_render_check(req, router, eng);
    r.n_state_cut = 1;
    CHECK_REASON(reason_of([&]() { strict(0, r); }), "STATE_TRUNCATED");
    r.n_state_cut = 0;
    CHECK_REASON(reason_of([&]() { strict(0, r); }), "OK");
}

int main() {
    test_cards();
    test_request();
    test_limits();
    if (n_fail) {
        fprintf(stderr, "test-decision-router: %d failures\n", n_fail);
        return 1;
    }
    printf("test-decision-router: OK\n");
    return 0;
}
