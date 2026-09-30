#pragma once

// decision.spec: the model contract (layout, limits, calibration, plan, router).
// Source of truth is the "decision.spec" GGUF string (UTF-8 JSON); the mirror
// keys decision.spec_version / layout / model_id / model_version must agree
// with it. --decision-spec FILE replaces the embedded spec as a whole.

#include "decision-json.h"

#include <cstdint>
#include <string>

#define DECISION_SPEC_VERSION 1
#define DECISION_CARD_SCHEMA  "atomic.executor-card/1"

// options per question of each layout (laya: LAYA_MAX_MARKERS)
#define DECISION_LAYA_MAX_OPTIONS  20
#define DECISION_SEMIF_MAX_OPTIONS 16

struct decision_calibration {
    std::string method     = "none";  // "temperature" or "none"
    bool        required   = false;   // a missing named calibration is a load error, never T = 1
    bool        calibrated = false;
    double      clamp_min  = 0.5;
    double      clamp_max  = 5.0;
    json        temperature = json::object(); // {"<type>|all": T | {"<bucket>": T}}, buckets "*", "N", "A-B", "N+"
    std::string version;
};

// built-in router question; a spec question without criteria gets these criteria
json decision_router_default_question();

struct decision_router_spec {
    bool        present         = false;
    std::string card_schema     = DECISION_CARD_SCHEMA;
    std::string card_renderer   = "card-v1";
    int32_t     max_card_tokens = -1;     // -1: layout default (3/8 of the model sequence length), 0: no limit
    json        question = decision_router_default_question(); // {"type": "noul", "instructions", "criteria"}
    bool        escape_control  = false;  // special_tokens "escape-control": control-token text in card, criterion and task becomes a space
    bool        calibrated      = false;  // platt parameters present
    double      platt_a         = 1.0;
    double      platt_b         = 0.0;
};

struct decision_spec {
    int32_t     spec_version = DECISION_SPEC_VERSION;
    std::string architecture;             // general.architecture of the GGUF
    std::string model_id;
    std::string model_version;
    std::string layout;                   // "laya", "semif-letters"
    std::string format;                   // "laya-v1", "semif-v1"
    std::string input_contract;           // laya: "laya-v1" | "laya-router-v1"; semif-letters: "semif-v1"
    std::string special_tokens;           // laya: "mask-to-space" | "escape-control"; semif-letters: "escape"
    int32_t     max_options       = 0;    // 0: engine limit
    int32_t     max_candidates    = 16;
    int32_t     max_prompt_tokens = 0;
    decision_calibration calibration;
    std::string confidence;               // "laya", "typesafe", "max_p"
    json        plan = json::object();    // {"name", ...} as given
    decision_router_spec router;

    json        raw;                      // the spec as parsed (or as generated for defaults)
    std::string text;                     // exact bytes hashed into sha256
    std::string sha256;
    std::string source;                   // "gguf", "file", "default"
};

// Load the spec of a model: sidecar file if given, else the GGUF decision.spec,
// else defaults for the architecture (laya: calibration from laya.temperature
// when it is not all 1.0, else calibrated:false). A laya spec without a
// "calibration" key keeps that laya.temperature calibration. Returns false with err set.
bool decision_spec_load(const std::string & model_path, const std::string & sidecar_path, decision_spec & spec, std::string & err);

// Validate a spec object and fill spec from it (raw/text/sha256/source untouched).
// A bare {"temperature": x} / {"temperature": {...}} (Arbiter/JevK5 calibration.json)
// is accepted as a required temperature calibration over the layout defaults and
// keeps the layout clamp. A required calibration must give a temperature for every
// question type and option count of the layout, else the spec is rejected.
bool decision_spec_from_json(const json & j, decision_spec & spec, std::string & err);

// default model name of a GGUF path: the file name without directories and without a
// trailing ".gguf" (the default spec model_id and the server's default alias)
std::string decision_model_name(const std::string & model_path);

// lowercase hex SHA-256
std::string decision_sha256_hex(const std::string & data);
