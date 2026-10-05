// clef prompt: request checks, rendering and the prompt of tools/decision/clef-prompt.h against
// encode_record of joint_schema_model.py (tests/clef/golden/prompts.jsonl, tests/clef/gen_golden.py).
// Needs the Qwen3.5 vocabulary only (models/ggml-vocab-qwen35.gguf).

#include "clef-prompt.h"
#include "decision.h"
#include "decision-request.h"
#include "decision-spec.h"

#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <string>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static decision_limits clef_limits() {
    decision_limits l;
    l.max_items   = 64;
    l.max_options = DECISION_CLEF_MAX_OPTIONS;
    l.layout      = "clef";
    return l;
}

template <typename F>
static std::string reason_of(F fn) {
    try {
        fn();
    } catch (const decision_error & e) {
        return decision_reason_name(e.reason);
    }
    return "OK";
}

static decision_request parse(const std::string & body) {
    return decision_parse_systemone(decision_parse_body(body), clef_limits());
}

static void test_render() {
    CHECK(clef_render(json("a b")) == "a b");
    CHECK(clef_render(json::parse(R"({"b": [1.0, 1e-05, 1e16, -0.0], "a": {"y": null, "x": true}, "\u00c9": "\u00e9"})")) ==
          "{\"a\":{\"x\":true,\"y\":null},\"b\":[1.0,1e-05,1e+16,-0.0],\"\xc3\x89\":\"\xc3\xa9\"}");
    CHECK(clef_render(json::parse(R"("<b>&'\"")")) == "<b>&'\"");
    CHECK(clef_render(json::parse(R"(["\u0001", "\u2028", "\n"])")) == "[\"\\u0001\",\"\u2028\",\"\\n\"]");
}

static void test_request() {
    // instructions: the question id when missing, null or ""; other values as compact JSON
    const decision_request r = parse(R"({"state": "s", "questions": {
        "q1": {"type": "noul"}, "q2": {"type": "noul", "instructions": null}, "q3": {"type": "noul", "instructions": ""},
        "q4": {"type": "noul", "instructions": {"b": 1, "a": [2.5]}}, "q5": {"type": "noul", "instructions": 7}}})");
    CHECK(r.questions.size() == 5);
    CHECK(r.questions[0].instructions == "q1");
    CHECK(r.questions[1].instructions == "q2");
    CHECK(r.questions[2].instructions == "q3");
    CHECK(r.questions[3].instructions == R"({"a":[2.5],"b":1})");
    CHECK(r.questions[4].instructions == "7");

    // a missing state is refused, a null state is the text "null"
    CHECK(reason_of([]() { parse(R"({"questions": {"q": {"type": "noul"}}})"); }) == "INVALID_REQUEST");
    CHECK(reason_of([]() { parse(R"({"state": null, "questions": {"q": {"type": "noul"}}})"); }) == "OK");
    CHECK(reason_of([]() { parse(R"({"state": "s", "questions": {}})"); }) == "INVALID_REQUEST");
    CHECK(reason_of([]() { parse(R"({"state": "s", "questions": {"q": {"type": "maybe"}}})"); }) == "UNKNOWN_QUESTION_TYPE");
    CHECK(reason_of([]() { parse(R"({"state": "s", "questions": {"": {"type": "noul"}}})"); }) == "EMPTY_INSTRUCTIONS");
    CHECK(reason_of([]() { parse(R"({"state": "s", "questions": {"q": {"type": "noul", "criteria": ["x"]}}})"); }) == "INVALID_NOUL_CRITERIA");
    CHECK(reason_of([]() { parse(R"({"state": "s", "questions": {"q": {"type": "choice", "criteria": {}}}})"); }) == "TOO_FEW_OPTIONS");
    CHECK(reason_of([]() { parse(R"({"state": "s", "questions": {"q": {"type": "score", "criteria": {"a": 1}}}})"); }) == "UNSUPPORTED_CRITERIA_VALUE");

    // one option is enough (the reference only needs non-empty criteria); a list is dict.fromkeys(str(c))
    const decision_request c = parse(R"({"state": "s", "questions": {
        "one": {"type": "choice", "criteria": {"x": "only"}}, "lst": {"type": "choice", "criteria": ["b", 1, true, "b"]},
        "lvl": {"type": "score", "criteria": ["fine"]}}})");
    CHECK(c.questions[0].keys == std::vector<std::string>({"x"}));
    CHECK(c.questions[1].keys == std::vector<std::string>({"b", "1", "True"}));
    CHECK(c.questions[2].keys == std::vector<std::string>({"0"}));

    // option order of the prompt: noul true first, choice sorted, score by level; null descriptions stay null
    const decision_request o = parse(R"({"state": "s", "questions": {
        "n": {"type": "noul", "criteria": {"false": null, "maybe": "x"}},
        "c": {"type": "choice", "criteria": {"b": 1, "A": null, "a": "x", "\u00e9": [1], "10": 2, "9": 3}}}})");
    const auto n = clef_question_options(o.questions[0]);
    CHECK(n.size() == 2 && n[0].first == "true" && n[1].first == "false");
    CHECK(n[0].second == "The proposition is true or the answer is yes.");
    CHECK(n[1].second.is_null());
    const auto ch = clef_question_options(o.questions[1]);
    std::vector<std::string> ids;
    for (const auto & p : ch) {
        ids.push_back(p.first);
    }
    CHECK(ids == std::vector<std::string>({"10", "9", "A", "a", "b", "\xc3\xa9"}));
}

static bool test_golden(const std::string & vocab_path, const std::string & golden_path) {
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(vocab_path.c_str(), mparams);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", vocab_path.c_str());
        return false;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const clef_tokenize_fn tokenize = [vocab](const std::string & text) {
        const int32_t n = -llama_tokenize(vocab, text.data(), (int32_t) text.size(), nullptr, 0, false, true);
        std::vector<int32_t> ids(std::max(n, 0));
        if (n > 0) {
            llama_tokenize(vocab, text.data(), (int32_t) text.size(), ids.data(), n, false, true);
        }
        return ids;
    };

    std::ifstream f(golden_path);
    if (!f) {
        fprintf(stderr, "cannot read %s\n", golden_path.c_str());
        llama_model_free(model);
        return false;
    }
    int n_cases = 0;
    std::string line;
    while (std::getline(f, line)) {
        const json g = json::parse(line);
        const std::string id = g.at("id");
        const decision_request req = parse(g.at("request").dump());
        std::vector<const decision_question *> qs;
        for (const auto & q : req.questions) {
            qs.push_back(&q);
        }
        n_cases++;

        clef_prompt p;
        const std::string reason = reason_of([&]() { p = clef_build_prompt(qs, req.state, g.at("max_tokens").get<int32_t>(), tokenize); });
        if (g.contains("error")) {
            if (reason != g.at("error").get<std::string>()) {
                fprintf(stderr, "%s: reason %s, want %s\n", id.c_str(), reason.c_str(), g.at("error").get<std::string>().c_str());
                n_fail++;
            }
            continue;
        }
        if (reason != "OK") {
            fprintf(stderr, "%s: unexpected %s\n", id.c_str(), reason.c_str());
            n_fail++;
            continue;
        }

        const std::vector<int32_t> want = g.at("tokens").get<std::vector<int32_t>>();
        if (p.tokens != want) {
            size_t i = 0;
            while (i < p.tokens.size() && i < want.size() && p.tokens[i] == want[i]) {
                i++;
            }
            fprintf(stderr, "%s: tokens differ at %zu (%zu vs %zu tokens)\n", id.c_str(), i, p.tokens.size(), want.size());
            n_fail++;
            continue;
        }

        // the decision order from the reference spans
        std::vector<int32_t> order(want.size(), CLEF_ORDER_NONE);
        for (const auto & s : g.at("question_spans")) {
            for (int32_t k = s[0]; k < s[1].get<int32_t>(); ++k) {
                order[k] = CLEF_ORDER_QUESTION_NOUL + s[2].get<int32_t>();
            }
        }
        for (const auto & s : g.at("option_spans")) {
            for (int32_t k = s[0]; k < s[1].get<int32_t>(); ++k) {
                order[k] = CLEF_ORDER_OPTION;
            }
        }
        if (p.order != order) {
            fprintf(stderr, "%s: decision order differs\n", id.c_str());
            n_fail++;
        }
        if (p.n_options != (int32_t) g.at("option_spans").size() || p.n_state != g.at("n_state").get<int32_t>() ||
            p.n_state_cut != g.at("n_state_cut").get<int32_t>()) {
            fprintf(stderr, "%s: n_options %d, n_state %d, n_state_cut %d\n", id.c_str(), p.n_options, p.n_state, p.n_state_cut);
            n_fail++;
        }

        // each criteria key maps to its option span in the prompt
        int32_t base = 0;
        for (size_t i = 0; i < req.questions.size(); ++i) {
            const auto & opts = g.at("options").at(req.questions[i].id);
            for (size_t j = 0; j < req.questions[i].keys.size(); ++j) {
                int32_t want_idx = -1;
                for (size_t k = 0; k < opts.size(); ++k) {
                    if (opts[k] == req.questions[i].keys[j]) {
                        want_idx = base + (int32_t) k;
                    }
                }
                if (p.option_index[i][j] != want_idx) {
                    fprintf(stderr, "%s: question %s option %s at %d, want %d\n", id.c_str(), req.questions[i].id.c_str(),
                            req.questions[i].keys[j].c_str(), p.option_index[i][j], want_idx);
                    n_fail++;
                }
            }
            base += (int32_t) opts.size();
        }
    }
    llama_model_free(model);
    printf("golden: %d cases\n", n_cases);
    return n_cases > 0;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <ggml-vocab-qwen35.gguf> <prompts.jsonl>\n", argv[0]);
        return 1;
    }
    llama_backend_init();
    test_render();
    test_request();
    if (!test_golden(argv[1], argv[2])) {
        n_fail++;
    }
    llama_backend_free();
    if (n_fail) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}
