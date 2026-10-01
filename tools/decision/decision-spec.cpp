#include "decision-spec.h"
#include "decision-calib.h"
#include "decision-request.h"
#include "decision.h"

#include "gguf.h"

extern "C" {
#include "sha256/sha256.h" // vendored in examples/gguf-hash/deps
}

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <sstream>
#include <type_traits>
#include <vector>

std::string decision_sha256_hex(const std::string & data) {
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_hash(digest, (const unsigned char *) data.data(), data.size());
    char out[2 * SHA256_DIGEST_SIZE + 1];
    for (int i = 0; i < SHA256_DIGEST_SIZE; ++i) {
        snprintf(out + 2 * i, 3, "%02x", digest[i]);
    }
    return std::string(out, 2 * SHA256_DIGEST_SIZE);
}

// ---- validation helpers ----

static bool is_uint_str(const std::string & s) {
    if (s.empty()) {
        return false;
    }
    for (char c : s) {
        if (c < '0' || c > '9') {
            return false;
        }
    }
    return true;
}

static bool valid_bucket(const std::string & b) {
    if (b == "*" || is_uint_str(b)) {
        return true;
    }
    if (b.size() > 1 && b.back() == '+') {
        return is_uint_str(b.substr(0, b.size() - 1));
    }
    const size_t dash = b.find('-');
    return dash != std::string::npos && is_uint_str(b.substr(0, dash)) && is_uint_str(b.substr(dash + 1));
}

static bool positive_number(const json & v) {
    return v.is_number() && std::isfinite(v.get<double>()) && v.get<double>() > 0.0;
}

static bool check_temperature(const json & t, std::string & err) {
    if (!t.is_object()) {
        err = "calibration.temperature must be an object";
        return false;
    }
    for (auto it = t.begin(); it != t.end(); ++it) {
        const std::string & k = it.key();
        if (k != "all" && k != "noul" && k != "choice" && k != "score") {
            err = "calibration.temperature: unknown question type '" + k + "'";
            return false;
        }
        if (it->is_object()) {
            for (auto b = it->begin(); b != it->end(); ++b) {
                if (!valid_bucket(b.key())) {
                    err = "calibration.temperature." + k + ": bad option-count bucket '" + b.key() + "'";
                    return false;
                }
                if (!positive_number(b.value())) {
                    err = "calibration.temperature." + k + "." + b.key() + " must be a positive number";
                    return false;
                }
            }
        } else if (!positive_number(*it)) {
            err = "calibration.temperature." + k + " must be a positive number or a bucket object";
            return false;
        }
    }
    return true;
}

template <typename T>
static bool get_opt(const json & j, const char * key, T & out, std::string & err, const char * what) {
    if (!j.contains(key) || j.at(key).is_null()) {
        return true;
    }
    try {
        const json & v = j.at(key);
        if constexpr (std::is_same_v<T, std::string>) {
            if (!v.is_string()) throw std::runtime_error("");
        } else if constexpr (std::is_same_v<T, bool>) {
            if (!v.is_boolean()) throw std::runtime_error("");
        } else if constexpr (std::is_integral_v<T>) {
            if (!v.is_number_integer() || v.get<int64_t>() < 0 || v.get<int64_t>() > INT32_MAX) throw std::runtime_error("");
        } else {
            if (!v.is_number() || !std::isfinite(v.get<double>())) throw std::runtime_error("");
        }
        out = v.get<T>();
        return true;
    } catch (const std::exception &) {
        err = std::string(key) + " must be " + what;
        return false;
    }
}

// layout defaults; fields the spec does not set keep these
static void set_layout_defaults(decision_spec & spec, const std::string & layout) {
    spec.layout = layout;
    spec.calibration = decision_calibration();
    if (layout == "laya") {
        spec.format         = "laya-v1";
        spec.input_contract = "laya-v1";
        spec.special_tokens = "mask-to-space";
        spec.confidence     = "laya";
        spec.plan           = json{{"name", "sequential"}};
        spec.calibration.clamp_min = 0.5;
        spec.calibration.clamp_max = 5.0;
    } else {
        spec.format         = "semif-v1";
        spec.input_contract = "semif-v1";
        spec.special_tokens = "escape";
        spec.confidence     = "max_p";
        spec.plan           = json{{"name", "server-split"}};
        spec.calibration.clamp_min = 0.05;
        spec.calibration.clamp_max = 20.0;
    }
    spec.router = decision_router_spec();
}

static bool parse_calibration(const json & c, decision_calibration & cal, std::string & err) {
    if (!c.is_object()) {
        err = "calibration must be an object";
        return false;
    }
    if (!get_opt(c, "method", cal.method, err, "a string") ||
        !get_opt(c, "required", cal.required, err, "a boolean") ||
        !get_opt(c, "version", cal.version, err, "a string")) {
        return false;
    }
    if (cal.method != "temperature" && cal.method != "none") {
        err = "calibration.method must be \"temperature\" or \"none\"";
        return false;
    }
    if (c.contains("clamp")) {
        const json & cl = c.at("clamp");
        if (!cl.is_array() || cl.size() != 2 || !positive_number(cl[0]) || !positive_number(cl[1]) ||
            cl[0].get<double>() > cl[1].get<double>()) {
            err = "calibration.clamp must be [min, max] with 0 < min <= max";
            return false;
        }
        cal.clamp_min = cl[0].get<double>();
        cal.clamp_max = cl[1].get<double>();
    }
    if (c.contains("temperature") && !c.at("temperature").is_null()) {
        if (!check_temperature(c.at("temperature"), err)) {
            return false;
        }
        cal.temperature = c.at("temperature");
    }
    cal.calibrated = cal.method == "temperature" && !cal.temperature.empty();
    if (cal.required && !cal.calibrated) {
        err = "calibration is required but has no temperatures";
        return false;
    }
    return true;
}

json decision_router_default_question() {
    return json{
        {"type",         "noul"},
        {"instructions", "Will the executor meet the success criterion on this task?"},
        {"criteria",     {{"true", "meets the criterion"}, {"false", "does not meet the criterion"}}},
    };
}

static bool parse_router(const json & r, decision_router_spec & router, std::string & err) {
    if (!r.is_object()) {
        err = "router must be an object or null";
        return false;
    }
    router.present = true;
    if (!get_opt(r, "card_schema", router.card_schema, err, "a string") ||
        !get_opt(r, "card_renderer", router.card_renderer, err, "a string") ||
        !get_opt(r, "max_card_tokens", router.max_card_tokens, err, "a non-negative integer")) {
        err = "router." + err;
        return false;
    }
    if (router.card_schema != DECISION_CARD_SCHEMA) {
        err = "router.card_schema must be \"" DECISION_CARD_SCHEMA "\"";
        return false;
    }
    if (router.card_renderer != "card-v1") {
        err = "router.card_renderer must be \"card-v1\"";
        return false;
    }
    // no question: the built-in one; a question without criteria gets the built-in criteria
    router.question = decision_router_default_question();
    if (r.contains("question") && !r.at("question").is_null()) {
        const json & q = r.at("question");
        if (!q.is_object() || !q.contains("type") || q.at("type") != "noul" ||
            !q.contains("instructions") || !q.at("instructions").is_string() || decision_is_blank(q.at("instructions").get<std::string>())) {
            err = "router.question must be a noul question with instructions";
            return false;
        }
        if (q.contains("criteria") && !q.at("criteria").is_null() && !q.at("criteria").is_object()) {
            err = "router.question.criteria must be an object with true/false descriptions";
            return false;
        }
        // the laya request rules for a noul question (laya_question_parse), checked at load
        if (q.contains("criteria") && q.at("criteria").is_object()) {
            for (auto it = q.at("criteria").begin(); it != q.at("criteria").end(); ++it) {
                std::string k = it.key();
                for (char & ch : k) {
                    ch = (ch >= 'A' && ch <= 'Z') ? (char) (ch - 'A' + 'a') : ch;
                }
                if (k != "true" && k != "false") {
                    err = "router.question.criteria must be keyed only true/false";
                    return false;
                }
            }
        }
        if (q.contains("labels") && !q.at("labels").is_null()) {
            const json & lb = q.at("labels");
            const bool shape = lb.is_object() && lb.size() == 2 && lb.contains("false") && lb.contains("true") &&
                               lb.at("false").is_string() && lb.at("true").is_string();
            const std::string fl = shape ? decision_py_strip(lb.at("false").get<std::string>()) : "";
            const std::string tl = shape ? decision_py_strip(lb.at("true").get<std::string>()) : "";
            if (fl.empty() || tl.empty() || fl == tl) {
                err = "router.question.labels must map exactly 'false' and 'true' to distinct non-empty strings";
                return false;
            }
        }
        const json criteria = q.contains("criteria") && !q.at("criteria").is_null() ? q.at("criteria") : router.question.at("criteria");
        router.question = q;
        router.question["criteria"] = criteria;
    }
    if (r.contains("calibration") && !r.at("calibration").is_null()) {
        const json & c = r.at("calibration");
        if (!c.is_object() || !c.contains("method") || c.at("method") != "platt") {
            err = "router.calibration.method must be \"platt\"";
            return false;
        }
        if (!c.contains("a") || !c.contains("b") || !c.at("a").is_number() || !c.at("b").is_number() ||
            !std::isfinite(c.at("a").get<double>()) || !std::isfinite(c.at("b").get<double>())) {
            err = "router.calibration needs finite numbers a and b";
            return false;
        }
        router.platt_a    = c.at("a").get<double>();
        router.platt_b    = c.at("b").get<double>();
        router.calibrated = true;
    }
    return true;
}

// a required calibration never falls back to T = 1
static bool check_required(const decision_spec & spec, std::string & err) {
    if (!spec.calibration.required) {
        return true;
    }
    const int32_t layout_max = spec.layout == "laya" ? DECISION_LAYA_MAX_OPTIONS : DECISION_SEMIF_MAX_OPTIONS;
    const int32_t max_options = spec.max_options > 0 ? std::min(spec.max_options, layout_max) : layout_max;
    std::string missing;
    if (!decision_calibration_covers(spec.calibration, max_options, missing)) {
        err = "calibration is required but has no temperature for " + missing;
        return false;
    }
    return true;
}

bool decision_spec_from_json(const json & j, decision_spec & spec, std::string & err) {
    if (!j.is_object()) {
        err = "decision spec must be a JSON object";
        return false;
    }

    // bare Arbiter/JevK5 calibration.json: {"temperature": x} or {"temperature": {...}}
    if (!j.contains("spec_version") && j.contains("temperature")) {
        const json & t = j.at("temperature");
        decision_calibration cal = spec.calibration; // keeps the layout clamp
        cal.version.clear();                         // not the version of the calibration it replaces (gguf:laya.temperature)
        cal.method     = "temperature";
        cal.required   = true;
        cal.temperature = t.is_object() ? t : json{{"all", t}};
        if (!check_temperature(cal.temperature, err)) {
            return false;
        }
        // same range check as arbiter/calibration.py
        for (auto it = cal.temperature.begin(); it != cal.temperature.end(); ++it) {
            if (it->is_number() && (it->get<double>() < 0.05 || it->get<double>() > 20.0)) {
                err = "temperature " + it.key() + " is outside [0.05, 20]";
                return false;
            }
        }
        if (j.contains("version") && j.at("version").is_string()) {
            cal.version = j.at("version").get<std::string>();
        }
        cal.calibrated   = true;
        spec.calibration = cal;
        return check_required(spec, err);
    }

    if (!j.contains("spec_version") || !j.at("spec_version").is_number_integer() ||
        j.at("spec_version").get<int64_t>() != DECISION_SPEC_VERSION) {
        err = "spec_version must be " + std::to_string(DECISION_SPEC_VERSION);
        return false;
    }
    if (!j.contains("layout") || !j.at("layout").is_string() ||
        (j.at("layout") != "laya" && j.at("layout") != "semif-letters")) {
        err = "layout must be \"laya\" or \"semif-letters\"";
        return false;
    }

    const std::string model_id = spec.model_id;
    set_layout_defaults(spec, j.at("layout").get<std::string>());
    spec.model_id = model_id;
    spec.model_version.clear();

    if (!get_opt(j, "model_id", spec.model_id, err, "a string") ||
        !get_opt(j, "model_version", spec.model_version, err, "a string") ||
        !get_opt(j, "input_contract", spec.input_contract, err, "a string") ||
        !get_opt(j, "special_tokens", spec.special_tokens, err, "a string") ||
        !get_opt(j, "confidence", spec.confidence, err, "a string")) {
        return false;
    }
    if (spec.confidence != "laya" && spec.confidence != "typesafe" && spec.confidence != "max_p") {
        err = "confidence must be \"laya\", \"typesafe\" or \"max_p\"";
        return false;
    }
    // contracts the engine implements; anything else would be claimed but not followed
    if (spec.layout == "laya") {
        if (spec.input_contract != "laya-v1" && spec.input_contract != "laya-router-v1") {
            err = "input_contract must be \"laya-v1\" or \"laya-router-v1\" for layout laya";
            return false;
        }
        if (spec.special_tokens != "mask-to-space" && spec.special_tokens != "escape-control") {
            err = "special_tokens must be \"mask-to-space\" or \"escape-control\" for layout laya";
            return false;
        }
        if (spec.input_contract == "laya-router-v1" && spec.special_tokens != "escape-control") {
            err = "input_contract \"laya-router-v1\" needs special_tokens \"escape-control\"";
            return false;
        }
    } else {
        if (spec.input_contract != "semif-v1") {
            err = "input_contract must be \"semif-v1\" for layout semif-letters";
            return false;
        }
        if (spec.special_tokens != "escape") {
            err = "special_tokens must be \"escape\" for layout semif-letters";
            return false;
        }
    }
    if (j.contains("limits") && !j.at("limits").is_null()) {
        const json & l = j.at("limits");
        if (!l.is_object()) {
            err = "limits must be an object";
            return false;
        }
        if (!get_opt(l, "max_options", spec.max_options, err, "a non-negative integer") ||
            !get_opt(l, "max_candidates", spec.max_candidates, err, "a non-negative integer") ||
            !get_opt(l, "max_prompt_tokens", spec.max_prompt_tokens, err, "a non-negative integer")) {
            err = "limits." + err;
            return false;
        }
    }
    if (j.contains("calibration") && !j.at("calibration").is_null()) {
        if (!parse_calibration(j.at("calibration"), spec.calibration, err)) {
            return false;
        }
    }
    if (j.contains("plan") && !j.at("plan").is_null()) {
        const json & p = j.at("plan");
        if (!p.is_object() || (p.contains("name") && !p.at("name").is_string())) {
            err = "plan must be an object with a string name";
            return false;
        }
        // the engine reads it (engine-laya.cpp): a spec the loader accepts must also load
        if (p.contains("kernels")) {
            const json & k = p.at("kernels");
            static const char * names[] = { "auto", "default", "repack", "blas", "repack+blas" };
            if (!k.is_string() || std::none_of(std::begin(names), std::end(names), [&](const char * n) { return k.get<std::string>() == n; })) {
                err = "plan.kernels must be auto, default, repack, blas or repack+blas";
                return false;
            }
        }
        spec.plan = p;
    }
    if (j.contains("router") && !j.at("router").is_null()) {
        if (!parse_router(j.at("router"), spec.router, err)) {
            return false;
        }
    }
    spec.router.escape_control = spec.special_tokens == "escape-control";
    return check_required(spec, err);
}

// effective spec of an unstamped GGUF, as a spec object
static json defaults_to_json(const decision_spec & spec) {
    json cal = {
        {"method",   spec.calibration.method},
        {"required", spec.calibration.required},
        {"clamp",    json::array({spec.calibration.clamp_min, spec.calibration.clamp_max})},
    };
    if (spec.calibration.method == "temperature") {
        cal["temperature"] = spec.calibration.temperature;
        cal["version"]     = spec.calibration.version;
    }
    return json{
        {"spec_version",   spec.spec_version},
        {"model_id",       spec.model_id},
        {"layout",         spec.layout},
        {"input_contract", spec.input_contract},
        {"special_tokens", spec.special_tokens},
        {"calibration",    cal},
        {"confidence",     spec.confidence},
        {"plan",           spec.plan},
        {"router",         nullptr},
    };
}

static bool read_file(const std::string & path, std::string & out) {
    std::ifstream f = decision_ifstream(path);
    if (!f) {
        return false;
    }
    std::ostringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

std::string decision_model_name(const std::string & path) {
    const size_t slash = path.find_last_of("/\\");
    std::string name = slash == std::string::npos ? path : path.substr(slash + 1);
    if (name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0) {
        name.resize(name.size() - 5);
    }
    return name;
}

bool decision_spec_load(const std::string & model_path, const std::string & sidecar_path, decision_spec & spec, std::string & err) {
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(gguf_init_from_file(model_path.c_str(), params), gguf_free);
    if (!ctx) {
        err = "failed to read GGUF metadata from " + model_path;
        return false;
    }
    const gguf_context * g = ctx.get();

    auto get_str = [&](const char * key, std::string & out) {
        const int64_t id = gguf_find_key(g, key);
        if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) {
            return false;
        }
        out = gguf_get_val_str(g, id);
        return true;
    };

    spec = decision_spec();
    if (!get_str("general.architecture", spec.architecture)) {
        err = "GGUF has no general.architecture";
        return false;
    }
    spec.model_id = decision_model_name(model_path);

    // defaults for an unstamped GGUF
    if (spec.architecture == "laya") {
        set_layout_defaults(spec, "laya");
        const int64_t id = gguf_find_key(g, "laya.temperature");
        if (id >= 0 && gguf_get_kv_type(g, id) == GGUF_TYPE_ARRAY && gguf_get_arr_type(g, id) == GGUF_TYPE_FLOAT32) {
            const size_t n = gguf_get_arr_n(g, id);
            const float * t = (const float *) gguf_get_arr_data(g, id);
            bool all_one = true;
            for (size_t i = 0; i < n; ++i) {
                all_one = all_one && t[i] == 1.0f;
            }
            // temperature_by_options: "<qtype>:<bucket>" -> T; the reference bucket "2" is k <= 2
            std::vector<std::pair<std::string, double>> buckets;
            const int64_t kb = gguf_find_key(g, "laya.temperature_by_options.buckets");
            const int64_t kv = gguf_find_key(g, "laya.temperature_by_options.values");
            if (kb >= 0 && kv >= 0 && gguf_get_kv_type(g, kb) == GGUF_TYPE_ARRAY && gguf_get_arr_type(g, kb) == GGUF_TYPE_STRING &&
                gguf_get_kv_type(g, kv) == GGUF_TYPE_ARRAY && gguf_get_arr_type(g, kv) == GGUF_TYPE_FLOAT32 &&
                gguf_get_arr_n(g, kb) == gguf_get_arr_n(g, kv)) {
                const float * v = (const float *) gguf_get_arr_data(g, kv);
                for (size_t i = 0; i < gguf_get_arr_n(g, kb); ++i) {
                    buckets.emplace_back(gguf_get_arr_str(g, kb, i), (double) v[i]);
                }
            }
            if (n > 0 && (!all_one || !buckets.empty())) {
                // per-qtype base temperature in type-embedding order (choice, score, noul)
                static const char * names[3] = { "choice", "score", "noul" };
                json temps = json::object();
                for (size_t i = 0; i < 3; ++i) {
                    temps[names[i]] = json{{"*", (double) t[i < n ? i : n - 1]}};
                }
                for (const auto & b : buckets) {
                    const size_t colon = b.first.find(':');
                    const std::string type = b.first.substr(0, colon);
                    const std::string size = colon == std::string::npos ? "" : b.first.substr(colon + 1);
                    if (temps.contains(type) && valid_bucket(size) && positive_number(b.second)) {
                        temps[type][size == "2" ? "1-2" : size] = b.second;
                    }
                }
                spec.calibration.method      = "temperature";
                spec.calibration.temperature = temps;
                spec.calibration.calibrated  = true;
                spec.calibration.version     = "gguf:laya.temperature";
            }
        }
    } else {
        set_layout_defaults(spec, "semif-letters");
    }

    std::string text;
    bool have_spec = false;
    if (!sidecar_path.empty()) {
        if (!read_file(sidecar_path, text)) {
            err = "cannot read decision spec file " + sidecar_path;
            return false;
        }
        spec.source = "file";
        have_spec = true;
    } else if (get_str("decision.spec", text)) {
        spec.source = "gguf";
        have_spec = true;
    }

    if (have_spec) {
        json j;
        std::string perr;
        if (decision_json_parse(text, j, perr) != DECISION_JSON_OK) {
            err = "decision spec (" + spec.source + ") is not valid JSON: " + perr;
            return false;
        }
        const decision_calibration gguf_cal = spec.calibration;
        if (!decision_spec_from_json(j, spec, err)) {
            err = "decision spec (" + spec.source + "): " + err;
            return false;
        }
        // a laya spec without a calibration block keeps the laya.temperature calibration
        if (spec.layout == "laya" && j.contains("spec_version") && !j.contains("calibration") && gguf_cal.calibrated) {
            spec.calibration = gguf_cal;
        }
        spec.raw  = j;
        spec.text = text;
    } else {
        if (spec.architecture != "laya") {
            err = "GGUF (" + spec.architecture + ") has no decision.spec; pass --decision-spec FILE";
            return false;
        }
        spec.source = "default";
        spec.raw    = defaults_to_json(spec);
        spec.text   = decision_py_dumps(spec.raw);
    }

    // the mirror keys are for readers that skip the JSON; they must agree with it
    if (spec.source == "gguf") {
        const int64_t id = gguf_find_key(g, "decision.spec_version");
        if (id >= 0 && (gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32 || (int64_t) gguf_get_val_u32(g, id) != spec.spec_version)) {
            err = "decision.spec_version does not match decision.spec";
            return false;
        }
        const std::pair<const char *, const std::string *> mirrors[] = {
            { "decision.layout",        &spec.layout },
            { "decision.model_id",      &spec.model_id },
            { "decision.model_version", &spec.model_version },
        };
        for (const auto & m : mirrors) {
            std::string v;
            if (gguf_find_key(g, m.first) >= 0 && (!get_str(m.first, v) || v != *m.second)) {
                err = std::string(m.first) + " does not match decision.spec";
                return false;
            }
        }
    }

    if ((spec.architecture == "laya") != (spec.layout == "laya")) {
        err = "decision spec layout '" + spec.layout + "' does not fit a '" + spec.architecture + "' GGUF";
        return false;
    }

    spec.sha256 = decision_sha256_hex(spec.text);
    return true;
}
