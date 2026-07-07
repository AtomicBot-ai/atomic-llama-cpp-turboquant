#pragma once

#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <string>

enum dflash_profile_flag : uint32_t {
    DFLASH_PROFILE_SUMMARY = 1u << 0,
    DFLASH_PROFILE_REPLAY  = 1u << 1,
    DFLASH_PROFILE_COPY    = 1u << 2,
    DFLASH_PROFILE_PREFILL = 1u << 3,
    DFLASH_PROFILE_VERIFY  = 1u << 4,
    DFLASH_PROFILE_TRACE   = 1u << 5,
};

static constexpr uint32_t DFLASH_PROFILE_DEFAULT =
    DFLASH_PROFILE_SUMMARY |
    DFLASH_PROFILE_REPLAY  |
    DFLASH_PROFILE_COPY    |
    DFLASH_PROFILE_VERIFY;

static constexpr uint32_t DFLASH_PROFILE_ALL =
    DFLASH_PROFILE_SUMMARY |
    DFLASH_PROFILE_REPLAY  |
    DFLASH_PROFILE_COPY    |
    DFLASH_PROFILE_PREFILL |
    DFLASH_PROFILE_VERIFY  |
    DFLASH_PROFILE_TRACE;

static inline bool dflash_profile_has(uint32_t active, uint32_t flags) {
    return (active & flags) != 0;
}

static inline bool dflash_profile_is_separator(char c) {
    return c == ',' || c == ';' || c == ':' || c == '|' || std::isspace((unsigned char) c);
}

static inline std::string dflash_profile_token(const char * begin, const char * end) {
    while (begin < end && std::isspace((unsigned char) *begin)) {
        ++begin;
    }
    while (end > begin && std::isspace((unsigned char) *(end - 1))) {
        --end;
    }

    std::string token;
    token.reserve((size_t) (end - begin));
    for (const char * p = begin; p < end; ++p) {
        token.push_back((char) std::tolower((unsigned char) *p));
    }
    return token;
}

static inline uint32_t dflash_profile_parse_token(const std::string & token) {
    if (token.empty() || token == "0" || token == "off" || token == "false" || token == "none") {
        return 0;
    }
    if (token == "1" || token == "on" || token == "true" || token == "default") {
        return DFLASH_PROFILE_DEFAULT;
    }
    if (token == "all" || token == "*") {
        return DFLASH_PROFILE_ALL;
    }
    if (token == "summary" || token == "cycle") {
        return DFLASH_PROFILE_SUMMARY;
    }
    if (token == "replay" || token == "tape") {
        return DFLASH_PROFILE_REPLAY;
    }
    if (token == "copy" || token == "copies") {
        return DFLASH_PROFILE_COPY;
    }
    if (token == "prefill" || token == "prompt") {
        return DFLASH_PROFILE_PREFILL;
    }
    if (token == "verify" || token == "verifier" || token == "logits") {
        return DFLASH_PROFILE_VERIFY;
    }
    if (token == "trace" || token == "verbose") {
        return DFLASH_PROFILE_TRACE;
    }

    return 0;
}

static inline uint32_t dflash_profile_parse_env(const char * env) {
    if (!env || env[0] == '\0') {
        return 0;
    }

    uint32_t flags = 0;
    const char * token_begin = env;
    for (const char * p = env; ; ++p) {
        if (*p != '\0' && !dflash_profile_is_separator(*p)) {
            continue;
        }

        flags |= dflash_profile_parse_token(dflash_profile_token(token_begin, p));
        if (*p == '\0') {
            break;
        }
        token_begin = p + 1;
    }

    return flags;
}

static inline uint32_t dflash_profile_flags() {
    static const uint32_t flags = dflash_profile_parse_env(std::getenv("GGML_DFLASH_PROFILE"));
    return flags;
}

static inline bool dflash_profile_enabled(uint32_t flags) {
    return dflash_profile_has(dflash_profile_flags(), flags);
}
