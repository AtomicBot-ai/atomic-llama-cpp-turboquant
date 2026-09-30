// py-json: strict parse + json.dumps(ensure_ascii=False) compatible dump
// usage: test-decision-json <tests/decision/golden>

#include "decision-json.h"
#include "decision-spec.h"

#include <cinttypes>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>

static int n_fail = 0;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: CHECK(%s) failed\n", __FILE__, __LINE__, #cond); n_fail++; } } while (0)

static void check_dump(const char * text, const char * expected) {
    json v;
    std::string err;
    const auto st = decision_json_parse(text, v, err);
    const std::string got = st == DECISION_JSON_OK ? decision_py_dumps(v) : "<error: " + err + ">";
    if (got != expected) {
        fprintf(stderr, "dump(%s): got %s, want %s\n", text, got.c_str(), expected);
        n_fail++;
    }
}

static void check_status(const std::string & text, decision_json_status want) {
    json v;
    std::string err;
    const auto st = decision_json_parse(text, v, err);
    if (st != want) {
        fprintf(stderr, "parse(%.60s): status %d, want %d (%s)\n", text.c_str(), (int) st, (int) want, err.c_str());
        n_fail++;
    }
}

static void test_golden_floats(const std::string & dir) {
    std::ifstream f(dir + "/py_float.txt");
    CHECK(f.good());
    std::string line;
    int n = 0, n_bad = 0, n_pow2 = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back(); // CRLF checkout
        }
        const double d = strtod(line.c_str(), nullptr);
        const std::string got = decision_py_float(d);
        if (got != line) {
            if (n_bad++ < 10) {
                fprintf(stderr, "py_float: got %s, want %s\n", got.c_str(), line.c_str());
            }
        }
        int e = 0;
        if (d > 0 && std::frexp(d, &e) == 0.5) {
            n_pow2++;
        }
        n++;
    }
    printf("py_float golden: %d values (%d powers of two), %d mismatches\n", n, n_pow2, n_bad);
    CHECK(n > 3000);
    CHECK(n_pow2 >= 2098);
    CHECK(n_bad == 0);
}

static void test_random_roundtrip() {
    std::mt19937_64 rng(42);
    int n_bad = 0;
    for (int i = 0; i < 30000; ++i) {
        uint64_t bits = rng();
        double d;
        memcpy(&d, &bits, sizeof(d));
        if (!std::isfinite(d)) {
            continue;
        }
        const std::string s = decision_py_float(d);
        const double back = strtod(s.c_str(), nullptr);
        if (memcmp(&back, &d, sizeof(d)) != 0 && !(d == 0 && back == 0)) {
            if (n_bad++ < 5) {
                fprintf(stderr, "roundtrip: %s\n", s.c_str());
            }
        }
    }
    CHECK(n_bad == 0);
}

// ---- splitmix64 streams, same generator as tests/decision/gen_golden.py ----

struct splitmix64 {
    uint64_t s;
    explicit splitmix64(uint64_t seed) : s(seed) {}
    uint64_t next() {
        s += 0x9E3779B97F4A7C15ull;
        uint64_t z = s;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    uint64_t below(uint64_t n) { return next() % n; }
};

static const uint32_t SM_ALPHABET[] = {
    'a', 'b', 'c', 'X', 'Y', 'Z', ' ', '0', '1', '9', '_', '-', '"', '\\', '/', '\n', '\t', '\r', '\b', '\f',
    0x00, 0x01, 0x1f, 0x7f, 0xe9, 0x4e2d, 0x6587, 0x438, 0x2028, 0x2029, 0x1F600, 0x301, 0xfeff, 0xffff, 0x10ffff,
};

static double sm_double(splitmix64 & r) {
    while (true) {
        double d;
        if (r.below(2) == 0) {
            const uint64_t bits = r.next();
            memcpy(&d, &bits, sizeof(d));
        } else {
            const uint64_t digits = 1 + r.below(17);
            uint64_t p10 = 1;
            for (uint64_t i = 0; i < digits; ++i) {
                p10 *= 10;
            }
            const uint64_t m = r.next() % p10;
            const int e = (int) r.below(660) - 340;
            const bool neg = r.below(2) == 1;
            char buf[64];
            snprintf(buf, sizeof(buf), "%s%" PRIu64 "e%d", neg ? "-" : "", m, e);
            d = strtod(buf, nullptr);
        }
        if (std::isfinite(d)) {
            return d;
        }
    }
}

static void utf8_append(std::string & out, uint32_t cp) {
    if (cp < 0x80) {
        out += (char) cp;
    } else if (cp < 0x800) {
        out += (char) (0xc0 | (cp >> 6));
        out += (char) (0x80 | (cp & 0x3f));
    } else if (cp < 0x10000) {
        out += (char) (0xe0 | (cp >> 12));
        out += (char) (0x80 | ((cp >> 6) & 0x3f));
        out += (char) (0x80 | (cp & 0x3f));
    } else {
        out += (char) (0xf0 | (cp >> 18));
        out += (char) (0x80 | ((cp >> 12) & 0x3f));
        out += (char) (0x80 | ((cp >> 6) & 0x3f));
        out += (char) (0x80 | (cp & 0x3f));
    }
}

static std::string sm_str(splitmix64 & r) {
    const uint64_t n = r.below(13);
    std::string s;
    for (uint64_t i = 0; i < n; ++i) {
        utf8_append(s, SM_ALPHABET[r.below(sizeof(SM_ALPHABET) / sizeof(SM_ALPHABET[0]))]);
    }
    return s;
}

static json sm_scalar(splitmix64 & r) {
    const uint64_t k = r.below(10);
    if (k == 0) {
        return nullptr;
    }
    if (k == 1) {
        return r.below(2) == 1;
    }
    if (k == 2) {
        switch (r.below(8)) {
            case 0: return 0;
            case 1: return -1;
            case 2: return (int64_t) 1 << 53;
            case 3: return INT64_MAX;
            case 4: return INT64_MIN;
            case 5: return (uint64_t) 1 << 63;
            case 6: return UINT64_MAX;
            default: return (int64_t) r.below(2000001) - 1000000;
        }
    }
    if (k == 3 || k == 4) {
        return sm_double(r);
    }
    return sm_str(r);
}

static json sm_value(splitmix64 & r, int depth) {
    if (depth >= 4 || r.below(10) < 4) {
        return sm_scalar(r);
    }
    if (r.below(2) == 0) {
        const uint64_t n = r.below(5);
        json a = json::array();
        for (uint64_t i = 0; i < n; ++i) {
            a.push_back(sm_value(r, depth + 1));
        }
        return a;
    }
    const uint64_t n = r.below(5);
    json o = json::object();
    for (uint64_t i = 0; i < n; ++i) {
        const std::string k = sm_str(r);
        o[k] = sm_value(r, depth + 1); // a repeated key keeps its first place, like a Python dict
    }
    return o;
}

static json sm_object(splitmix64 & r) {
    const uint64_t n = 1 + r.below(5);
    json o = json::object();
    for (uint64_t i = 0; i < n; ++i) {
        const std::string k = sm_str(r);
        o[k] = sm_value(r, 1);
    }
    return o;
}

// repr() of 30k doubles and json.dumps() of 1k objects, compared by SHA-256 with CPython
static void test_golden_hashes(const std::string & dir) {
    std::ifstream f(dir + "/py_hashes.txt", std::ios::binary);
    CHECK(f.good());
    std::string line;
    int n_streams = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ss(line);
        std::string name, want;
        uint64_t seed = 0;
        int count = 0;
        ss >> name >> seed >> count >> want;
        splitmix64 r(seed);
        std::string text;
        int n_roundtrip_bad = 0;
        for (int i = 0; i < count; ++i) {
            if (name == "floats") {
                text += decision_py_float(sm_double(r)) + "\n";
            } else {
                const std::string dumped = decision_py_dumps(sm_object(r));
                json back;
                std::string err;
                if (decision_json_parse(dumped, back, err) != DECISION_JSON_OK || decision_py_dumps(back) != dumped) {
                    n_roundtrip_bad++;
                }
                text += dumped + "\n";
            }
        }
        const std::string got = decision_sha256_hex(text);
        printf("py_hashes %s: %d values, sha256 %s\n", name.c_str(), count, got == want ? "OK" : "MISMATCH");
        CHECK(name == "floats" || name == "objects");
        CHECK(got == want);
        CHECK(n_roundtrip_bad == 0);
        n_streams++;
    }
    CHECK(n_streams == 2);
}

static void test_golden_json(const std::string & dir) {
    std::ifstream f(dir + "/py_json.jsonl", std::ios::binary);
    CHECK(f.good());
    std::string line;
    int n = 0, n_bad = 0;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back(); // CRLF checkout
        }
        json v;
        std::string err;
        const auto st = decision_json_parse(line, v, err);
        const std::string got = st == DECISION_JSON_OK ? decision_py_dumps(v) : "<error: " + err + ">";
        if (got != line && n_bad++ < 3) {
            fprintf(stderr, "py_json line %d:\n got  %s\n want %s\n", n + 1, got.c_str(), line.c_str());
        }
        n++;
    }
    printf("py_json golden: %d objects, %d mismatches\n", n, n_bad);
    CHECK(n >= 100);
    CHECK(n_bad == 0);
}

static void test_policy() {
    check_dump("{\"a\": 1, \"b\": 2, \"a\": 3}", "{\"a\": 3, \"b\": 2}");
    check_dump("[1.0, 1E2, -0, -0.0, 1e-999, 1e16, 1e15, 0.0001, 0.00001]",
               "[1.0, 100.0, 0, -0.0, 0.0, 1e+16, 1000000000000000.0, 0.0001, 1e-05]");
    check_dump("[18446744073709551615, -9223372036854775808, 9223372036854775808]",
               "[18446744073709551615, -9223372036854775808, 9223372036854775808]");
    check_dump("\"\\ud83d\\ude00 \\u00e9 \\u2028 \\u007f \\u0000\"", "\"\xf0\x9f\x98\x80 \xc3\xa9 \xe2\x80\xa8 \x7f \\u0000\"");
    check_dump("{\"k\":[true,false,null,{}],\"s\":\"a\\/b\"}", "{\"k\": [true, false, null, {}], \"s\": \"a/b\"}");

    check_status("NaN", DECISION_JSON_UNSUPPORTED_NUMBER);
    check_status("[1, Infinity]", DECISION_JSON_UNSUPPORTED_NUMBER);
    check_status("{\"x\": -Infinity}", DECISION_JSON_UNSUPPORTED_NUMBER);
    check_status("1e999", DECISION_JSON_UNSUPPORTED_NUMBER);
    check_status("-1e999", DECISION_JSON_UNSUPPORTED_NUMBER);
    check_status("18446744073709551616", DECISION_JSON_UNSUPPORTED_NUMBER);
    check_status("-9223372036854775809", DECISION_JSON_UNSUPPORTED_NUMBER);
    check_status("{\"a\": 1,}", DECISION_JSON_MALFORMED);
    check_status("{} x", DECISION_JSON_MALFORMED);
    check_status("\"\\ud800\"", DECISION_JSON_MALFORMED);
    check_status("\"\xff\"", DECISION_JSON_MALFORMED);
    check_status("", DECISION_JSON_MALFORMED);
    check_status("nul", DECISION_JSON_MALFORMED);

    std::string deep_ok  = std::string(DECISION_JSON_MAX_DEPTH, '[') + std::string(DECISION_JSON_MAX_DEPTH, ']');
    std::string deep_bad = std::string(DECISION_JSON_MAX_DEPTH + 1, '[') + std::string(DECISION_JSON_MAX_DEPTH + 1, ']');
    check_status(deep_ok, DECISION_JSON_OK);
    check_status(deep_bad, DECISION_JSON_MALFORMED);
    check_status(std::string(100000, '['), DECISION_JSON_MALFORMED);

    std::string s;
    CHECK(decision_py_str(json(true), s) && s == "True");
    CHECK(decision_py_str(json(nullptr), s) && s == "None");
    CHECK(decision_py_str(json(1.0), s) && s == "1.0");
    CHECK(decision_py_str(json(-7), s) && s == "-7");
    CHECK(decision_py_str(json("x y"), s) && s == "x y");
    CHECK(!decision_py_str(json::object(), s));
    CHECK(!decision_py_str(json::array(), s));

    // dict keys as json.dumps writes them, and Python == over JSON scalars
    CHECK(decision_py_json_key(json(true), s) && s == "true");
    CHECK(decision_py_json_key(json(nullptr), s) && s == "null");
    CHECK(decision_py_json_key(json(1.0), s) && s == "1.0");
    CHECK(decision_py_json_key(json("True"), s) && s == "True");
    CHECK(!decision_py_json_key(json::array(), s));
    CHECK(decision_py_scalar_eq(json(1), json(1.0)) && decision_py_scalar_eq(json(true), json(1)) && decision_py_scalar_eq(json(1u), json(true)));
    CHECK(decision_py_scalar_eq(json(0), json(-0.0)) && decision_py_scalar_eq(json(false), json(0.0)));
    CHECK(decision_py_scalar_eq(json(1.5), json(1.5)) && decision_py_scalar_eq(json("a"), json("a")) && decision_py_scalar_eq(json(), json()));
    CHECK(!decision_py_scalar_eq(json("1"), json(1)) && !decision_py_scalar_eq(json(1), json(2)) && !decision_py_scalar_eq(json(), json(0)));
    CHECK(!decision_py_scalar_eq(json(-1), json(UINT64_MAX)) && !decision_py_scalar_eq(json(UINT64_MAX), json(18446744073709551616.0)));
    CHECK(decision_py_scalar_eq(json(INT64_MIN), json(-9223372036854775808.0)));
    CHECK(!decision_py_scalar_eq(json(9007199254740993), json(9007199254740992.0)) && decision_py_scalar_eq(json(9007199254740992), json(9007199254740992.0)));
    CHECK(decision_py_isspace(0x2028) && decision_py_isspace(0x85) && !decision_py_isspace(0x200b));
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    if (argc < 2) {
        fprintf(stderr, "usage: %s <tests/decision/golden>\n", argv[0]);
        return 1;
    }
    const std::string dir = argv[1];
    test_golden_floats(dir);
    test_random_roundtrip();
    test_golden_hashes(dir);
    test_golden_json(dir);
    test_policy();
    if (n_fail) {
        fprintf(stderr, "test-decision-json: %d failures\n", n_fail);
        return 1;
    }
    printf("test-decision-json: OK\n");
    return 0;
}
