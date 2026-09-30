#include "decision.h"
#include "decision-spec.h"
#include "laya.h"

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <shellapi.h>
#    include <filesystem>
#endif

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

int32_t decision_cpu_perf_cores() {
    return laya_cpu_perf_cores();
}

void decision_cpu_env_defaults() {
    laya_cpu_env_defaults();
}

std::ifstream decision_ifstream(const std::string & path) {
#if defined(_WIN32)
    return std::ifstream(std::filesystem::path(laya_utf8_to_wide(path)), std::ios::binary);
#else
    return std::ifstream(path, std::ios::binary);
#endif
}

std::ofstream decision_ofstream(const std::string & path) {
#if defined(_WIN32)
    return std::ofstream(std::filesystem::path(laya_utf8_to_wide(path)), std::ios::binary);
#else
    return std::ofstream(path, std::ios::binary);
#endif
}

std::vector<std::string> decision_utf8_args(int argc, char ** argv) {
    std::vector<std::string> args(argv, argv + argc);
#if defined(_WIN32)
    int wargc = 0;
    LPWSTR * wargv = CommandLineToArgvW(GetCommandLineW(), &wargc);
    if (!wargv) {
        return args;
    }
    std::vector<std::string> wide_args;
    for (int i = 0; i < wargc; ++i) {
        const int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr, nullptr);
        std::string s((size_t) (n > 0 ? n : 1), '\0');
        if (n > 0) {
            WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], n, nullptr, nullptr);
        }
        s.resize(s.size() - 1);
        wide_args.push_back(std::move(s));
    }
    LocalFree(wargv);
    // only when it is the same command line (a launcher can pass argv that differs)
    if ((int) wide_args.size() == argc) {
        args = std::move(wide_args);
    }
#endif
    return args;
}
