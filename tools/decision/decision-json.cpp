#include "decision-json.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

// SAX handler that builds an ordered_json DOM and applies the number policy.
// nlohmann reports integers beyond uint64/int64 as floats, with the raw text.
struct strict_sax : nlohmann::json_sax<json> {
    json & root;
    std::vector<json *> stack;
    json * key_slot = nullptr;
    bool bad_number = false;
    bool too_deep   = false;
    std::string err;
    size_t err_pos = 0;

    explicit strict_sax(json & r) : root(r) {}

    json * put(json && v) {
        if (stack.empty()) {
            root = std::move(v);
            return &root;
        }
        json * top = stack.back();
        if (top->is_array()) {
            top->push_back(std::move(v));
            return &top->back();
        }
        *key_slot = std::move(v);
        return key_slot;
    }

    bool null() override                { put(nullptr); return true; }
    bool boolean(bool v) override       { put(v); return true; }
    bool number_integer(number_integer_t v) override   { put(v); return true; }
    bool number_unsigned(number_unsigned_t v) override { put(v); return true; }

    bool number_float(number_float_t v, const string_t & s) override {
        if (!std::isfinite(v)) {
            err = "number out of range: " + s;
            bad_number = true;
            return false;
        }
        if (s.find_first_of(".eE") == std::string::npos) {
            err = "integer outside the int64/uint64 range: " + s;
            bad_number = true;
            return false;
        }
        put(v);
        return true;
    }

    bool string(string_t & v) override { put(std::move(v)); return true; }
    bool binary(binary_t &) override   { return false; }

    bool open(json && v) {
        if (stack.size() >= DECISION_JSON_MAX_DEPTH) {
            err = "nesting deeper than " + std::to_string(DECISION_JSON_MAX_DEPTH);
            too_deep = true;
            return false;
        }
        stack.push_back(put(std::move(v)));
        return true;
    }

    bool start_object(std::size_t) override { return open(json::object()); }
    bool start_array(std::size_t) override  { return open(json::array()); }
    bool end_object() override { stack.pop_back(); return true; }
    bool end_array() override  { stack.pop_back(); return true; }

    bool key(string_t & k) override {
        // operator[] keeps the position of an existing key, so the last value wins in place
        key_slot = &(*stack.back())[k];
        return true;
    }

    bool parse_error(std::size_t pos, const std::string &, const nlohmann::detail::exception & ex) override {
        err = ex.what();
        err_pos = pos;
        bad_number = ex.id == 406; // number overflow
        return false;
    }
};

bool starts_with_at(const std::string & s, size_t pos, const char * lit) {
    return s.compare(pos, strlen(lit), lit) == 0;
}

void py_escape(const std::string & s, std::string & out) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    out += '"';
}

// compact: separators=(",", ":"); sort_keys: by code point, which is the byte order of UTF-8
void py_dump(const json & v, std::string & out, bool compact = false, bool sort_keys = false) {
    const char * item_sep = compact ? "," : ", ";
    const char * key_sep  = compact ? ":" : ": ";
    switch (v.type()) {
        case json::value_t::object: {
            std::vector<json::const_iterator> items;
            for (auto it = v.begin(); it != v.end(); ++it) {
                items.push_back(it);
            }
            if (sort_keys) {
                std::sort(items.begin(), items.end(), [](const json::const_iterator & a, const json::const_iterator & b) {
                    return a.key() < b.key();
                });
            }
            out += '{';
            bool first = true;
            for (const auto & it : items) {
                if (!first) {
                    out += item_sep;
                }
                first = false;
                py_escape(it.key(), out);
                out += key_sep;
                py_dump(it.value(), out, compact, sort_keys);
            }
            out += '}';
            break;
        }
        case json::value_t::array: {
            out += '[';
            bool first = true;
            for (const auto & e : v) {
                if (!first) {
                    out += item_sep;
                }
                first = false;
                py_dump(e, out, compact, sort_keys);
            }
            out += ']';
            break;
        }
        case json::value_t::string:          py_escape(v.get_ref<const std::string &>(), out); break;
        case json::value_t::boolean:         out += v.get<bool>() ? "true" : "false"; break;
        case json::value_t::number_integer:  out += std::to_string(v.get<int64_t>()); break;
        case json::value_t::number_unsigned: out += std::to_string(v.get<uint64_t>()); break;
        case json::value_t::number_float:    out += decision_py_float(v.get<double>()); break;
        default:                             out += "null"; break;
    }
}

} // namespace

decision_json_status decision_json_parse(const std::string & text, json & out, std::string & err) {
    out = json();
    strict_sax sax(out);
    bool ok = false;
    try {
        ok = json::sax_parse(text, &sax);
    } catch (const std::exception & e) {
        sax.err = e.what();
        ok = false;
    }
    if (ok) {
        return DECISION_JSON_OK;
    }
    out = json();
    err = sax.err.empty() ? "invalid JSON" : sax.err;
    if (sax.bad_number) {
        return DECISION_JSON_UNSUPPORTED_NUMBER;
    }
    if (!sax.too_deep && sax.err_pos > 0) {
        // Python accepts these literals; say why they fail here
        const size_t p = sax.err_pos - 1;
        if (starts_with_at(text, p, "NaN") || starts_with_at(text, p, "Infinity") || starts_with_at(text, p, "-Infinity") ||
            (p > 0 && text[p - 1] == '-' && starts_with_at(text, p, "Infinity"))) {
            err = "NaN and Infinity are not JSON numbers";
            return DECISION_JSON_UNSUPPORTED_NUMBER;
        }
    }
    return DECISION_JSON_MALFORMED;
}

std::string decision_py_dumps(const json & v) {
    std::string out;
    py_dump(v, out);
    return out;
}

std::string decision_py_dumps_compact_sorted(const json & v) {
    std::string out;
    py_dump(v, out, true, true);
    return out;
}

std::string decision_py_float(double d) {
    if (std::isnan(d)) {
        return "NaN";
    }
    if (std::isinf(d)) {
        return d > 0 ? "Infinity" : "-Infinity";
    }

    std::string out = std::signbit(d) ? "-" : "";
    d = std::fabs(d);
    if (d == 0.0) {
        return out + "0.0";
    }

    // shortest round-trip digits, "D.DDDe+XX"; to_chars picks the closest of the
    // shortest candidates, like CPython's dtoa (a strtod search does not at powers of two)
    char buf[64];
    const auto res = std::to_chars(buf, buf + sizeof(buf) - 1, d, std::chars_format::scientific);
    *res.ptr = '\0';

    const char * p = buf;
    std::string digits(1, *p++);
    if (*p == '.') {
        for (++p; *p != 'e'; ++p) {
            digits += *p;
        }
    }
    const int exp = atoi(p + 1);

    const int decpos = exp + 1; // digits before the decimal point
    if (exp >= -4 && exp < 16) {
        if (decpos <= 0) {
            out += "0.";
            out.append(-decpos, '0');
            out += digits;
        } else if ((int) digits.size() <= decpos) {
            out += digits;
            out.append(decpos - digits.size(), '0');
            out += ".0";
        } else {
            out += digits.substr(0, decpos);
            out += '.';
            out += digits.substr(decpos);
        }
    } else {
        out += digits[0];
        if (digits.size() > 1) {
            out += '.';
            out += digits.substr(1);
        }
        char e[16];
        snprintf(e, sizeof(e), "e%+03d", exp);
        out += e;
    }
    return out;
}

bool decision_py_str(const json & v, std::string & out) {
    switch (v.type()) {
        case json::value_t::string:  out = v.get<std::string>(); return true;
        case json::value_t::boolean: out = v.get<bool>() ? "True" : "False"; return true;
        case json::value_t::null:    out = "None"; return true;
        case json::value_t::number_integer:
        case json::value_t::number_unsigned:
        case json::value_t::number_float:
            out = decision_py_dumps(v);
            // str(inf) differs from json.dumps, but the parser never yields non-finite values
            return true;
        default:
            return false;
    }
}

bool decision_py_json_key(const json & v, std::string & out) {
    switch (v.type()) {
        case json::value_t::boolean: out = v.get<bool>() ? "true" : "false"; return true;
        case json::value_t::null:    out = "null"; return true;
        default:                     return decision_py_str(v, out);
    }
}

namespace {

// exact value of a JSON number or bool: an integer (neg, mag) or a non-integral double
struct py_num {
    bool     integral = true;
    bool     neg      = false;
    uint64_t mag      = 0;
    double   d        = 0.0;
};

py_num py_num_of(const json & v) {
    py_num n;
    switch (v.type()) {
        case json::value_t::boolean:
            n.mag = v.get<bool>() ? 1 : 0;
            break;
        case json::value_t::number_unsigned:
            n.mag = v.get<uint64_t>();
            break;
        case json::value_t::number_integer: {
            const int64_t i = v.get<int64_t>();
            n.neg = i < 0;
            n.mag = n.neg ? (uint64_t) 0 - (uint64_t) i : (uint64_t) i;
            break;
        }
        default: {
            const double d = v.get<double>();
            // 2^64 as a double; integral doubles below it convert exactly
            if (d == std::floor(d) && std::fabs(d) < 18446744073709551616.0) {
                n.neg = d < 0;
                n.mag = (uint64_t) std::fabs(d);
            } else {
                n.integral = false;
                n.d        = d;
            }
            break;
        }
    }
    if (n.mag == 0) {
        n.neg = false; // -0.0 == 0
    }
    return n;
}

bool py_is_num(const json & v) {
    return v.is_number() || v.is_boolean();
}

} // namespace

bool decision_py_scalar_eq(const json & a, const json & b) {
    if (a.is_string() || b.is_string()) {
        return a.is_string() && b.is_string() && a.get_ref<const std::string &>() == b.get_ref<const std::string &>();
    }
    if (a.is_null() || b.is_null()) {
        return a.is_null() && b.is_null();
    }
    if (!py_is_num(a) || !py_is_num(b)) {
        return false;
    }
    const py_num x = py_num_of(a);
    const py_num y = py_num_of(b);
    if (x.integral != y.integral) {
        return false;
    }
    if (!x.integral) {
        return x.d == y.d;
    }
    return x.neg == y.neg && x.mag == y.mag;
}

bool decision_py_isspace(uint32_t c) {
    return (c >= 0x09 && c <= 0x0d) || (c >= 0x1c && c <= 0x20) || c == 0x85 || c == 0xa0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200a) || c == 0x2028 || c == 0x2029 || c == 0x202f || c == 0x205f || c == 0x3000;
}

// code point at s[i] of valid UTF-8, and its byte length
static uint32_t utf8_at(const std::string & s, size_t i, size_t & len) {
    const unsigned char c = s[i];
    uint32_t cp;
    if (c < 0x80)      { cp = c;        len = 1; }
    else if (c < 0xe0) { cp = c & 0x1f; len = 2; }
    else if (c < 0xf0) { cp = c & 0x0f; len = 3; }
    else               { cp = c & 0x07; len = 4; }
    for (size_t k = 1; k < len; ++k) {
        cp = i + k < s.size() ? (cp << 6) | (s[i + k] & 0x3f) : cp;
    }
    return cp;
}

std::string decision_py_strip(const std::string & s) {
    size_t begin = 0;
    size_t end   = 0; // one past the last non-space code point
    bool   found = false;
    for (size_t i = 0; i < s.size();) {
        size_t len;
        const uint32_t cp = utf8_at(s, i, len);
        if (!decision_py_isspace(cp)) {
            if (!found) {
                begin = i;
                found = true;
            }
            end = std::min(s.size(), i + len);
        }
        i += len;
    }
    return found ? s.substr(begin, end - begin) : std::string();
}
