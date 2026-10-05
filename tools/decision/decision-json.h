#pragma once

// Strict JSON input and Python-compatible JSON output for the decision API.
//
// Prompts embed request values as text, so the text must match what the Python
// references produce with json.dumps(v, ensure_ascii=False): ", " / ": "
// separators, shortest round-trip floats in repr() layout, raw UTF-8.

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>

using json = nlohmann::ordered_json;

// max nesting depth of an accepted document
#define DECISION_JSON_MAX_DEPTH 128

enum decision_json_status {
    DECISION_JSON_OK,
    DECISION_JSON_MALFORMED,          // not RFC 8259 JSON, bad UTF-8, too deep
    DECISION_JSON_UNSUPPORTED_NUMBER, // NaN/Infinity, integer outside int64/uint64, float overflow
};

// Parse with the number policy above. Duplicate keys keep the first position and
// the last value (like a Python dict). On failure err holds a short message.
decision_json_status decision_json_parse(const std::string & text, json & out, std::string & err);

// json.dumps(v, ensure_ascii=False)
std::string decision_py_dumps(const json & v);

// json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
std::string decision_py_dumps_compact_sorted(const json & v);

// repr(float): shortest round-trip digits, fixed for 1e-4 <= |x| < 1e16, ".0" on integral values
std::string decision_py_float(double d);

// str(v) for a JSON scalar (str, int, float, True/False, None); false for objects and arrays
bool decision_py_str(const json & v, std::string & out);

// the key json.dumps writes for a dict key v: str as is, True/False -> "true"/"false",
// None -> "null", int -> str(v), float -> repr(v); false for objects and arrays
bool decision_py_json_key(const json & v, std::string & out);

// Python a == b for JSON scalars: strings by value, numbers and booleans by exact
// numeric value (1 == 1.0 == True, 0 == -0.0 == False), None only to None, never str to number
bool decision_py_scalar_eq(const json & a, const json & b);

// Python str.isspace() for one code point
bool decision_py_isspace(uint32_t cp);

// Python str.strip() of valid UTF-8 text
std::string decision_py_strip(const std::string & s);
