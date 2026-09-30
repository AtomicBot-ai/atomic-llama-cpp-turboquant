// calibration (temperature buckets, clamp, softmax, confidence, Platt), answers and decision.spec loading
// usage: test-decision-calib <tests/decision>   (spec_cases.json is shared with gguf_decision_spec.py)

#include "decision.h"
#include "decision-calib.h"
#include "decision-spec.h"

#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <process.h>
#define test_getpid _getpid
#else
#include <unistd.h>
#define test_getpid getpid
#endif

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)
#define CHECK_NEAR(a, b, eps) do { const double a_ = (a), b_ = (b); if (!(std::fabs(a_ - b_) <= (eps))) { fprintf(stderr, "%s:%d: %.17g != %.17g\n", __FILE__, __LINE__, a_, b_); n_fail++; } } while (0)

static const char * SPEC_ROUTER = R"({"spec_version":1,"model_id":"atomic/router-laya","model_version":"1.0.0","layout":"laya",
 "input_contract":"laya-router-v1","special_tokens":"escape-control","limits":{"max_options":20,"max_candidates":16},
 "calibration":{"method":"temperature","required":true,"clamp":[0.5,5.0],
   "temperature":{"noul":{"2":1.37},"choice":{"2":1.2,"3-5":1.45,"6-10":1.6,"11+":1.8},"score":{"*":1.3}}},
 "confidence":"laya","plan":{"name":"sequential","recipe":"Q8_0-mixed","n_threads":4},
 "router":{"card_schema":"atomic.executor-card/1","card_renderer":"card-v1","max_card_tokens":384,
   "question":{"type":"noul","instructions":"Will the executor meet the success criterion on this task?",
               "criteria":{"true":"meets the criterion","false":"does not meet the criterion"}},
   "calibration":{"method":"platt","a":1.0,"b":0.0}},
 "parity":{"reference":"pytorch-fp32","laya_commit":"x","report_sha256":"y"}})";

static const char * SPEC_ARBITER = R"({"spec_version":1,"model_id":"AtomicChat/Arbiter-4B","layout":"semif-letters","input_contract":"semif-v1",
 "special_tokens":"escape","letters":{"strings":["A","B"],"token_ids":[32,33]},"limits":{"max_options":16,"max_prompt_tokens":16384},
 "calibration":{"method":"temperature","required":true,"clamp":[0.05,20],"temperature":{"all":1.6121}},
 "confidence":"max_p","plan":{"name":"server-split","n_batch":2048,"n_ubatch":512,"kv_type":"f16","flash_attn":"auto"},"router":null})";

static bool spec_from(const std::string & text, decision_spec & spec, std::string & err, const char * layout = "laya") {
    json j;
    if (decision_json_parse(text, j, err) != DECISION_JSON_OK) {
        return false;
    }
    spec = decision_spec();
    spec.layout = layout;
    if (std::string(layout) == "semif-letters") {
        spec.calibration.clamp_min = 0.05;
        spec.calibration.clamp_max = 20.0;
    }
    return decision_spec_from_json(j, spec, err);
}

static void test_spec_parse() {
    decision_spec spec;
    std::string err;
    CHECK(spec_from(SPEC_ROUTER, spec, err));
    CHECK(spec.model_id == "atomic/router-laya" && spec.model_version == "1.0.0" && spec.layout == "laya");
    CHECK(spec.format == "laya-v1" && spec.input_contract == "laya-router-v1" && spec.special_tokens == "escape-control");
    CHECK(spec.max_options == 20 && spec.max_candidates == 16);
    CHECK(spec.calibration.calibrated && spec.calibration.required && spec.calibration.method == "temperature");
    CHECK(spec.confidence == "laya" && spec.plan.at("name") == "sequential");
    CHECK(spec.router.present && spec.router.calibrated && spec.router.max_card_tokens == 384);
    CHECK(spec.router.platt_a == 1.0 && spec.router.platt_b == 0.0);

    CHECK(spec_from(SPEC_ARBITER, spec, err));
    CHECK(spec.layout == "semif-letters" && spec.format == "semif-v1" && spec.confidence == "max_p");
    CHECK(!spec.router.present && spec.max_prompt_tokens == 16384);
    CHECK_NEAR(decision_temperature(spec.calibration, DECISION_QTYPE_SCORE, 4), 1.6121, 0);

    // bare Arbiter/JevK5 calibration.json shapes
    CHECK(spec_from("{\"temperature\": 1.41}", spec, err, "semif-letters"));
    CHECK(spec.calibration.required && spec.calibration.calibrated);
    CHECK_NEAR(decision_temperature(spec.calibration, DECISION_QTYPE_NOUL, 2), 1.41, 0);
    CHECK(spec_from("{\"temperature\": {\"noul\": 1.2, \"choice\": 1.5, \"score\": 1.1}, \"fitted_on\": \"transfer\"}", spec, err, "semif-letters"));
    CHECK_NEAR(decision_temperature(spec.calibration, DECISION_QTYPE_CHOICE, 9), 1.5, 0);
    CHECK(!spec_from("{\"temperature\": 25}", spec, err, "semif-letters"));
    CHECK(!spec_from("{\"temperature\": \"hot\"}", spec, err, "semif-letters"));
    // a bare sidecar keeps the layout clamp (laya [0.5, 5]) and must cover every question type
    CHECK(spec_from("{\"temperature\": 10}", spec, err, "laya"));
    CHECK_NEAR(decision_temperature(spec.calibration, DECISION_QTYPE_CHOICE, 3), 5.0, 0);
    CHECK(spec_from("{\"temperature\": 10}", spec, err, "semif-letters"));
    CHECK_NEAR(decision_temperature(spec.calibration, DECISION_QTYPE_CHOICE, 3), 10.0, 0);
    // a bare sidecar does not inherit the version of the calibration it replaces
    {
        json j = json::parse("{\"temperature\": 1.3}");
        decision_spec s2;
        s2.layout = "laya";
        s2.calibration.version = "gguf:laya.temperature";
        CHECK(decision_spec_from_json(j, s2, err) && s2.calibration.version.empty());
        j = json::parse("{\"temperature\": 1.3, \"version\": \"cal-7\"}");
        s2.calibration.version = "gguf:laya.temperature";
        CHECK(decision_spec_from_json(j, s2, err) && s2.calibration.version == "cal-7");
    }
    CHECK(!spec_from("{\"temperature\": {\"choice\": 1.5}}", spec, err, "semif-letters"));
    CHECK(err.find("noul with 2 options") != std::string::npos);

    // a required calibration covers every (type, K); a missing bucket is a load error, never T = 1
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"calibration\": {\"method\": \"temperature\", \"required\": true, \"temperature\": {\"choice\": {\"2\": 1.2}}}}", spec, err));
    CHECK(err.find("choice with 3 options") != std::string::npos || err.find("noul with 2 options") != std::string::npos);
    CHECK(spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"calibration\": {\"method\": \"temperature\", \"temperature\": {\"choice\": {\"2\": 1.2}}}}", spec, err));
    decision_calibration cov;
    cov.method = "temperature";
    cov.temperature = json::parse(R"({"noul": 1.1, "score": {"*": 1.3}, "choice": {"2-5": 1.2, "7+": 1.4}})");
    std::string missing;
    CHECK(!decision_calibration_covers(cov, 20, missing) && missing == "choice with 6 options");
    CHECK(decision_calibration_covers(cov, 5, missing));

    // router block: the built-in question when absent, the built-in criteria when a question has none
    CHECK(spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"router\": {}}", spec, err));
    CHECK(spec.router.present && spec.router.question == decision_router_default_question() && spec.router.max_card_tokens == -1);
    CHECK(spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"router\": {\"max_card_tokens\": 0, \"question\": {\"type\": \"noul\", \"instructions\": \"Fits?\"}}}", spec, err));
    CHECK(spec.router.max_card_tokens == 0 && spec.router.question.at("instructions") == "Fits?");
    CHECK(spec.router.question.at("criteria") == decision_router_default_question().at("criteria"));
    CHECK(spec_from("{\"spec_version\": 1, \"layout\": \"laya\"}", spec, err));
    CHECK(!spec.router.present && spec.router.question == decision_router_default_question());
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"router\": {\"question\": {\"type\": \"noul\", \"instructions\": \"x\", \"criteria\": [\"a\"]}}}", spec, err));

    // rejected specs
    CHECK(!spec_from("{\"spec_version\": 2, \"layout\": \"laya\"}", spec, err));
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"onnx\"}", spec, err));
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"calibration\": {\"method\": \"temperature\", \"required\": true}}", spec, err));
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"calibration\": {\"method\": \"temperature\", \"temperature\": {\"choice\": {\"two\": 1.2}}}}", spec, err));
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"calibration\": {\"method\": \"temperature\", \"temperature\": {\"vote\": 1.2}}}", spec, err));
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"calibration\": {\"method\": \"temperature\", \"temperature\": {\"all\": -1}}}", spec, err));
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"confidence\": \"entropy\"}", spec, err));
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"router\": {\"question\": {\"type\": \"choice\", \"instructions\": \"x\"}}}", spec, err));
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"router\": {\"question\": {\"type\": \"noul\", \"instructions\": \"x\"}, \"calibration\": {\"method\": \"isotonic\"}}}", spec, err));

    // special-token and input contracts: only what the engine implements
    CHECK(spec_from(SPEC_ROUTER, spec, err) && spec.router.escape_control);
    CHECK(spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"special_tokens\": \"escape-control\"}", spec, err));
    CHECK(spec.router.escape_control && spec.input_contract == "laya-v1");
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"special_tokens\": \"escape\"}", spec, err));
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"special_tokens\": \"parse\"}", spec, err));
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"input_contract\": \"semif-v1\"}", spec, err));
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"laya\", \"input_contract\": \"laya-router-v1\"}", spec, err));
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"semif-letters\", \"special_tokens\": \"escape-control\"}", spec, err, "semif-letters"));
    CHECK(!spec_from("{\"spec_version\": 1, \"layout\": \"semif-letters\", \"input_contract\": \"laya-v1\"}", spec, err, "semif-letters"));

    // uncalibrated laya spec
    CHECK(spec_from("{\"spec_version\": 1, \"layout\": \"laya\"}", spec, err));
    CHECK(!spec.calibration.calibrated && spec.calibration.method == "none");
    CHECK(spec.special_tokens == "mask-to-space" && !spec.router.escape_control);
    CHECK_NEAR(decision_temperature(spec.calibration, DECISION_QTYPE_CHOICE, 3), 1.0, 0);
}

static void test_temperature() {
    decision_spec spec;
    std::string err;
    CHECK(spec_from(SPEC_ROUTER, spec, err));
    const decision_calibration & cal = spec.calibration;
    CHECK_NEAR(decision_temperature(cal, DECISION_QTYPE_NOUL,   2),  1.37, 0);
    CHECK_NEAR(decision_temperature(cal, DECISION_QTYPE_CHOICE, 2),  1.2,  0);
    CHECK_NEAR(decision_temperature(cal, DECISION_QTYPE_CHOICE, 3),  1.45, 0);
    CHECK_NEAR(decision_temperature(cal, DECISION_QTYPE_CHOICE, 5),  1.45, 0);
    CHECK_NEAR(decision_temperature(cal, DECISION_QTYPE_CHOICE, 6),  1.6,  0);
    CHECK_NEAR(decision_temperature(cal, DECISION_QTYPE_CHOICE, 10), 1.6,  0);
    CHECK_NEAR(decision_temperature(cal, DECISION_QTYPE_CHOICE, 11), 1.8,  0);
    CHECK_NEAR(decision_temperature(cal, DECISION_QTYPE_CHOICE, 20), 1.8,  0);
    CHECK_NEAR(decision_temperature(cal, DECISION_QTYPE_SCORE,  7),  1.3,  0);

    // exact bucket beats range, range beats open, open beats "*"; clamp; "all" fallback
    decision_calibration c;
    c.method = "temperature";
    c.clamp_min = 0.5;
    c.clamp_max = 5.0;
    c.temperature = json::parse(R"({"choice": {"*": 1.1, "3+": 1.2, "2-4": 1.3, "3": 1.4}, "all": 9.0, "noul": 0.1})");
    CHECK_NEAR(decision_temperature(c, DECISION_QTYPE_CHOICE, 3), 1.4, 0);
    CHECK_NEAR(decision_temperature(c, DECISION_QTYPE_CHOICE, 4), 1.3, 0);
    CHECK_NEAR(decision_temperature(c, DECISION_QTYPE_CHOICE, 7), 1.2, 0);
    CHECK_NEAR(decision_temperature(c, DECISION_QTYPE_CHOICE, 2), 1.3, 0);
    CHECK_NEAR(decision_temperature(c, DECISION_QTYPE_SCORE,  4), 5.0, 0);
    CHECK_NEAR(decision_temperature(c, DECISION_QTYPE_NOUL,   2), 0.5, 0);
    c.temperature = json::parse(R"({"choice": {"11+": 2.0}})");
    CHECK_NEAR(decision_temperature(c, DECISION_QTYPE_CHOICE, 3), 1.0, 0);
    // the T = 1 fallback of a calibration that is not required is not clamped; found values are
    c.clamp_min = 3.0;
    CHECK_NEAR(decision_temperature(c, DECISION_QTYPE_CHOICE, 3),  1.0, 0);
    CHECK_NEAR(decision_temperature(c, DECISION_QTYPE_NOUL,   2),  1.0, 0);
    CHECK_NEAR(decision_temperature(c, DECISION_QTYPE_CHOICE, 12), 3.0, 0);
    c.clamp_min = 0.5;
    c.method = "none";
    CHECK_NEAR(decision_temperature(c, DECISION_QTYPE_CHOICE, 20), 1.0, 0);
}

static void test_softmax_confidence() {
    auto p = decision_softmax({ 1.0, 2.0, 3.0 }, 1.0);
    CHECK_NEAR(p[0] + p[1] + p[2], 1.0, 1e-15);
    CHECK_NEAR(p[2], std::exp(2.0) / (1.0 + std::exp(1.0) + std::exp(2.0)), 1e-15);
    p = decision_softmax({ 1.0, 2.0, 3.0 }, 2.0);
    CHECK_NEAR(p[2], std::exp(1.5) / (std::exp(0.5) + std::exp(1.0) + std::exp(1.5)), 1e-15);
    p = decision_softmax({ 1000.0, 0.0 }, 1.0);
    CHECK(std::isfinite(p[0]) && p[0] == 1.0 && p[1] >= 0.0);

    const std::vector<double> u3 = { 1.0 / 3, 1.0 / 3, 1.0 / 3 };
    CHECK_NEAR(decision_confidence(DECISION_CONFIDENCE_LAYA,     DECISION_QTYPE_CHOICE, u3), 0.0, 1e-12);
    CHECK_NEAR(decision_confidence(DECISION_CONFIDENCE_TYPESAFE, DECISION_QTYPE_CHOICE, u3), 0.0, 1e-12);
    CHECK_NEAR(decision_confidence(DECISION_CONFIDENCE_MAX_P,    DECISION_QTYPE_CHOICE, u3), 1.0 / 3, 0);
    const std::vector<double> n = { 0.2, 0.8 };
    CHECK_NEAR(decision_confidence(DECISION_CONFIDENCE_LAYA,     DECISION_QTYPE_NOUL, n), 0.8, 0);
    CHECK_NEAR(decision_confidence(DECISION_CONFIDENCE_TYPESAFE, DECISION_QTYPE_NOUL, n),
               1.0 + (0.2 * std::log(0.2) + 0.8 * std::log(0.8)) / std::log(2.0), 1e-15);
    CHECK_NEAR(decision_confidence(DECISION_CONFIDENCE_MAX_P,    DECISION_QTYPE_NOUL, n), 0.8, 0);
    CHECK_NEAR(decision_confidence(DECISION_CONFIDENCE_LAYA,     DECISION_QTYPE_CHOICE, { 1.0, 0.0 }), 1.0, 0);

    decision_confidence_mode mode;
    CHECK(decision_confidence_from_name("max_p", mode) && mode == DECISION_CONFIDENCE_MAX_P);
    CHECK(!decision_confidence_from_name("entropy", mode));

    CHECK_NEAR(decision_platt(1.0, 0.0, 0.0), 0.5, 0);
    CHECK_NEAR(decision_platt(2.0, -1.0, 1.0), 1.0 / (1.0 + std::exp(-1.0)), 1e-15);
    CHECK(decision_platt(1.0, 0.0, -1000.0) >= 0.0 && decision_platt(1.0, 0.0, 1000.0) == 1.0);
}

static void test_answer() {
    decision_question q;
    q.id = "q";
    q.type = DECISION_QTYPE_CHOICE;
    q.criteria = json::parse(R"({"billing": "Payment problems.", "other": null, "tech": "x"})");
    q.keys = { "billing", "other", "tech" };
    // ties go to the first key in criteria order
    json a = decision_answer(q, { 0.4, 0.4, 0.2 }, DECISION_CONFIDENCE_MAX_P);
    CHECK(decision_py_dumps(a) == "{\"type\": \"choice\", \"choice\": \"billing\", \"probabilities\": {\"billing\": 0.4, \"other\": 0.4, \"tech\": 0.2}, \"confidence\": 0.4}");

    q.type = DECISION_QTYPE_SCORE;
    q.criteria = json::parse(R"(["poor", 2, {"k": "v"}])");
    q.keys = { "0", "1", "2" };
    a = decision_answer(q, { 0.25, 0.25, 0.5 }, DECISION_CONFIDENCE_MAX_P);
    CHECK(decision_py_dumps(a) == "{\"type\": \"score\", \"score\": 1.25, \"probabilities\": {\"0\": 0.25, \"1\": 0.25, \"2\": 0.5}, "
                                  "\"legend\": {\"0\": \"poor\", \"1\": 2, \"2\": {\"k\": \"v\"}}, \"confidence\": 0.5}");

    q.type = DECISION_QTYPE_NOUL;
    q.criteria = nullptr;
    q.keys = { "false", "true" };
    a = decision_answer(q, { 0.25, 0.75 }, DECISION_CONFIDENCE_LAYA);
    CHECK(decision_py_dumps(a) == "{\"type\": \"noul\", \"noul\": 0.75, \"confidence\": 0.75}");
}

static void test_sha256() {
    CHECK(decision_sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(decision_sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(decision_sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(decision_sha256_hex(std::string(1000000, 'a')) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

// ---- decision_spec_load on metadata-only GGUF files ----

// unique per process: two build trees may run ctest at the same time
static std::string tmp_path(const char * name) {
    return (std::filesystem::temp_directory_path() / (std::string(name) + "-" + std::to_string(test_getpid()))).string();
}

static void write_gguf(const std::string & path, const char * arch, const std::vector<float> & temps, const char * spec,
                       const char * mirror_layout = nullptr, const char * mirror_id = nullptr) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", arch);
    if (!temps.empty()) {
        gguf_set_arr_data(g, "laya.temperature", GGUF_TYPE_FLOAT32, temps.data(), temps.size());
    }
    if (spec) {
        gguf_set_val_str(g, "decision.spec", spec);
        gguf_set_val_u32(g, "decision.spec_version", 1);
    }
    if (mirror_layout) {
        gguf_set_val_str(g, "decision.layout", mirror_layout);
    }
    if (mirror_id) {
        gguf_set_val_str(g, "decision.model_id", mirror_id);
    }
    CHECK(gguf_write_to_file(g, path.c_str(), false));
    gguf_free(g);
}

static void test_spec_load() {
    const std::string path = tmp_path("test-decision-calib") + ".gguf";
    decision_spec spec;
    std::string err;

    // unstamped laya GGUF with T = [1, 1, 1] (laya-multilingual): calibrated:false
    write_gguf(path, "laya", { 1.0f, 1.0f, 1.0f }, nullptr);
    CHECK(decision_spec_load(path, "", spec, err));
    CHECK(spec.source == "default" && spec.layout == "laya" && spec.model_id == std::filesystem::path(path).stem().string());
    CHECK(!spec.calibration.calibrated && spec.sha256 == decision_sha256_hex(spec.text));
    CHECK(spec.raw.at("router").is_null());

    // laya.temperature carries per-qtype base temperatures (choice, score, noul)
    write_gguf(path, "laya", { 1.25f, 1.5f, 0.75f }, nullptr);
    CHECK(decision_spec_load(path, "", spec, err));
    CHECK(spec.calibration.calibrated && !spec.calibration.required);
    CHECK_NEAR(decision_temperature(spec.calibration, DECISION_QTYPE_CHOICE, 3), 1.25, 0);
    CHECK_NEAR(decision_temperature(spec.calibration, DECISION_QTYPE_SCORE,  5), 1.5,  0);
    CHECK_NEAR(decision_temperature(spec.calibration, DECISION_QTYPE_NOUL,   2), 0.75, 0);

    // embedded spec + matching mirror keys
    write_gguf(path, "laya", {}, SPEC_ROUTER, "laya", "atomic/router-laya");
    CHECK(decision_spec_load(path, "", spec, err));
    CHECK(spec.source == "gguf" && spec.sha256 == decision_sha256_hex(SPEC_ROUTER) && spec.router.present);

    // mismatching mirror key
    write_gguf(path, "laya", {}, SPEC_ROUTER, "laya", "someone/else");
    CHECK(!decision_spec_load(path, "", spec, err));
    CHECK(err.find("decision.model_id") != std::string::npos);

    // layout that does not fit the architecture
    write_gguf(path, "laya", {}, SPEC_ARBITER);
    CHECK(!decision_spec_load(path, "", spec, err));

    // letters GGUF without a spec needs a sidecar; a bare calibration sidecar is a required calibration
    write_gguf(path, "qwen35", {}, nullptr);
    CHECK(!decision_spec_load(path, "", spec, err));
    const std::string side = tmp_path("test-decision-calib") + ".json";
    {
        std::ofstream f(side, std::ios::binary);
        f << "{\"temperature\": 1.6121}";
    }
    CHECK(decision_spec_load(path, side, spec, err));
    CHECK(spec.source == "file" && spec.layout == "semif-letters" && spec.calibration.required);
    CHECK(spec.sha256 == decision_sha256_hex("{\"temperature\": 1.6121}"));

    // sidecar replaces the embedded spec as a whole
    write_gguf(path, "laya", {}, SPEC_ROUTER, "laya", "atomic/router-laya");
    {
        std::ofstream f(side, std::ios::binary);
        f << "{\"spec_version\": 1, \"layout\": \"laya\", \"model_id\": \"other\"}";
    }
    CHECK(decision_spec_load(path, side, spec, err));
    CHECK(spec.source == "file" && spec.model_id == "other" && !spec.router.present);

    // a stamped laya spec without a calibration block keeps laya.temperature; an explicit block replaces it
    write_gguf(path, "laya", { 1.25f, 1.5f, 0.75f }, "{\"spec_version\": 1, \"layout\": \"laya\"}");
    CHECK(decision_spec_load(path, "", spec, err));
    CHECK(spec.source == "gguf" && spec.calibration.calibrated && spec.calibration.version == "gguf:laya.temperature");
    CHECK_NEAR(decision_temperature(spec.calibration, DECISION_QTYPE_SCORE, 5), 1.5, 0);
    write_gguf(path, "laya", { 1.25f, 1.5f, 0.75f }, "{\"spec_version\": 1, \"layout\": \"laya\", \"calibration\": {\"method\": \"none\"}}");
    CHECK(decision_spec_load(path, "", spec, err));
    CHECK(!spec.calibration.calibrated);
    write_gguf(path, "laya", { 1.0f, 1.0f, 1.0f }, "{\"spec_version\": 1, \"layout\": \"laya\"}");
    CHECK(decision_spec_load(path, "", spec, err));
    CHECK(!spec.calibration.calibrated);

    CHECK(!decision_spec_load(tmp_path("does-not-exist") + ".gguf", "", spec, err));

    std::filesystem::remove(path);
    std::filesystem::remove(side);
}

// the same cases run through gguf_decision_spec.py in tools/server/tests/unit/test_decision.py
static void test_spec_cases(const std::string & dir) {
    std::ifstream f(dir + "/spec_cases.json", std::ios::binary);
    CHECK(f.good());
    std::stringstream ss;
    ss << f.rdbuf();
    json cases;
    std::string err;
    CHECK(decision_json_parse(ss.str(), cases, err) == DECISION_JSON_OK && cases.is_array());
    int n = 0;
    for (const auto & c : cases) {
        decision_spec spec;
        spec.layout = "laya";
        const bool ok = decision_spec_from_json(c.at("spec"), spec, err);
        if (ok != c.at("ok").get<bool>()) {
            fprintf(stderr, "spec case '%s': loader says %s (%s)\n", c.at("name").get<std::string>().c_str(), ok ? "ok" : "bad", ok ? "" : err.c_str());
            n_fail++;
        }
        n++;
    }
    printf("spec cases: %d\n", n);
    CHECK(n >= 50);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <tests/decision>\n", argv[0]);
        return 1;
    }
    test_spec_cases(argv[1]);
    test_spec_parse();
    test_temperature();
    test_softmax_confidence();
    test_answer();
    test_sha256();
    test_spec_load();
    if (n_fail) {
        fprintf(stderr, "test-decision-calib: %d failures\n", n_fail);
        return 1;
    }
    printf("test-decision-calib: OK\n");
    return 0;
}
