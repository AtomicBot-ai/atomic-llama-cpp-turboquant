// Router rendering of the engine against the Python reference (tools/decision/reference.py).
//
// usage: test-decision-reference <tests/decision/golden/router_cases.jsonl> [--dump NAME]
//        test-decision-reference --control MODEL.gguf     (escape set of a laya GGUF as JSON)
//
// Each golden case holds an exact request body and what reference.py made of it (written by
// tests/decision/gen_router_golden.py): the 400 reason and param, or a SHA-256 over warnings,
// the question's type and instructions, options, and per candidate the state, its split offsets
// and the escaped model text. This test runs the same body through decision_parse_body /
// decision_parse_router / decision_router_items and laya_escape_text, as llama-server --decision
// does, and hashes the result the same way.
// --dump prints the engine's rendering of one case, to compare with `reference.py router`.

#include "decision.h"
#include "decision-request.h"
#include "decision-router.h"
#include "decision-spec.h"
#include "laya-decide.h"
#include "laya.h"

#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

static std::string netstring(const std::string & s) {
    return std::to_string(s.size()) + ":" + s;
}

struct rendered {
    bool                     ok = false;
    std::string              reason;
    std::string              param;
    std::vector<std::string> fields;  // digest input, in reference digest() order
    size_t                   n_items = 0;
};

static rendered render_case(const json & c, const std::map<std::string, laya_escape> & escapes) {
    rendered r;
    const std::string st = c.at("special_tokens").get<std::string>();
    json spec_j = {
        {"spec_version",   1},
        {"model_id",       "golden"},
        {"layout",         "laya"},
        {"special_tokens", st},
        {"input_contract", st == "escape-control" ? "laya-router-v1" : "laya-v1"},
        {"calibration",    nullptr},
        {"router",         c.at("router").is_null() ? json::object() : c.at("router")},
    };
    decision_spec spec;
    std::string err;
    if (!decision_spec_from_json(spec_j, spec, err)) {
        r.reason = "SPEC: " + err;
        return r;
    }
    decision_limits lim;
    lim.layout      = "laya";
    lim.max_options = DECISION_LAYA_MAX_OPTIONS;
    const laya_escape & esc = escapes.at(c.at("control").get<std::string>());
    try {
        const json body = decision_parse_body(c.at("body_text").get<std::string>());
        const decision_router_request req = decision_parse_router(body, DECISION_MAX_CANDIDATES);
        const std::vector<decision_item> items = decision_router_items(req, spec.router, lim);
        r.fields.push_back(std::to_string(req.warnings.size()));
        r.fields.insert(r.fields.end(), req.warnings.begin(), req.warnings.end());
        r.fields.push_back(req.truncation_error ? "1" : "0");
        r.fields.push_back(decision_qtype_name(items.at(0).q.type));
        r.fields.push_back(items.at(0).q.instructions);
        r.fields.insert(r.fields.end(), items.at(0).q.options.begin(), items.at(0).q.options.end());
        for (size_t i = 0; i < items.size(); ++i) {
            const decision_item & it = items[i];
            const std::string state = it.state.get<std::string>();
            if (it.escape_control != (st == "escape-control") || it.state_splits.size() != 2 ||
                state.compare(0, it.state_splits[0], req.candidates[i].card_text) != 0 || it.state_splits[0] != req.candidates[i].card_text.size()) {
                r.reason = "INCONSISTENT item " + std::to_string(i);
                return r;
            }
            r.fields.push_back(it.q.id);
            r.fields.push_back(state);
            r.fields.push_back(std::to_string(it.state_splits[0]) + "," + std::to_string(it.state_splits[1]));
            r.fields.push_back(laya_escape_text(esc, state, it.escape_control));
        }
        r.n_items = items.size();
        r.ok = true;
    } catch (const decision_error & e) {
        r.reason = decision_reason_name(e.reason);
        r.param  = e.param;
    }
    return r;
}

static std::string digest(const rendered & r) {
    std::string buf;
    for (const auto & f : r.fields) {
        buf += netstring(f);
    }
    return decision_sha256_hex(buf);
}

static int print_control(const char * path) {
    ggml_backend_load_all();
    laya_model_params mp;
    laya_model * model = laya_model_load_from_file_ext(path, mp);
    if (!model) {
        fprintf(stderr, "cannot load %s\n", path);
        return 1;
    }
    const laya_escape esc = laya_escape_init(model);
    laya_model_free(model);
    printf("%s\n", decision_py_dumps(json{{"mask", esc.mask}, {"control", esc.control}}).c_str());
    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc >= 3 && std::string(argv[1]) == "--control") {
        return print_control(argv[2]);
    }
    if (argc < 2) {
        fprintf(stderr, "usage: %s <router_cases.jsonl> [--dump NAME] | --control MODEL.gguf\n", argv[0]);
        return 1;
    }
    const std::string dump = argc >= 4 && std::string(argv[2]) == "--dump" ? argv[3] : "";

    std::ifstream f(argv[1], std::ios::binary);
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[1]);
        return 1;
    }
    std::string line;
    std::getline(f, line);
    const json header = json::parse(line);
    if (header.value("format", "") != "router-cases-v1") {
        fprintf(stderr, "%s: unknown format\n", argv[1]);
        return 1;
    }
    std::map<std::string, laya_escape> escapes;
    for (auto it = header.at("control_sets").begin(); it != header.at("control_sets").end(); ++it) {
        escapes[it.key()] = laya_escape_make(it->at("mask").get<std::string>(), it->at("control").get<std::vector<std::string>>());
    }

    int n = 0;
    int n_fail = 0;
    int n_ok = 0;
    size_t n_items = 0;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        }
        const json c = json::parse(line);
        const std::string name = c.at("name").get<std::string>();
        const json & want = c.at("expect");
        const rendered r = render_case(c, escapes);
        n++;
        if (name == dump) {
            json out = {{"ok", r.ok}, {"reason", r.reason}, {"param", r.param}, {"fields", r.fields}};
            printf("%s\n", out.dump(1).c_str());
        }
        std::string why;
        if (want.at("ok").get<bool>()) {
            if (!r.ok) {
                why = "engine rejects it: " + r.reason + " " + r.param;
            } else if (r.n_items != want.at("n_items").get<size_t>()) {
                why = "item count " + std::to_string(r.n_items);
            } else if (digest(r) != want.at("digest").get<std::string>()) {
                why = "rendering differs (rerun with --dump " + name + ")";
            }
            n_ok += r.ok;
            n_items += r.n_items;
        } else if (r.ok) {
            why = "engine accepts it, reference says " + want.at("reason").get<std::string>();
        } else if (r.reason != want.at("reason").get<std::string>() || r.param != want.at("param").get<std::string>()) {
            why = "engine " + r.reason + " '" + r.param + "', reference " + want.at("reason").get<std::string>() + " '" +
                  want.at("param").get<std::string>() + "'";
        }
        if (!why.empty()) {
            fprintf(stderr, "%s: %s\n", name.c_str(), why.c_str());
            n_fail++;
        }
    }
    if (n < 500 || n_items < 500) {
        fprintf(stderr, "test-decision-reference: only %d cases / %zu cards\n", n, n_items);
        return 1;
    }
    if (n_fail) {
        fprintf(stderr, "test-decision-reference: %d of %d cases differ from reference.py\n", n_fail, n);
        return 1;
    }
    printf("test-decision-reference: OK (%d cases, %d accepted, %zu candidate states)\n", n, n_ok, n_items);
    return 0;
}
