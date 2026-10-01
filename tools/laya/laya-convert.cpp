// laya HF checkpoint -> GGUF (see laya-convert.h). Each step names the Python code it ports.

#include "laya-convert.h"

#include "ggml.h"
#include "gguf.h"
#include "laya.h" // laya_utf8_to_wide

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <io.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#else
#    include <dirent.h>
#    include <fcntl.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif

using json         = nlohmann::json;
using ordered_json = nlohmann::ordered_json;

namespace {

struct lc_error : std::runtime_error {
    explicit lc_error(const std::string & msg) : std::runtime_error(msg) {}
};

[[noreturn]] void lc_fail(const std::string & msg) {
    throw lc_error(msg);
}

// ---------------------------------------------------------------------------------------------
// files (UTF-8 paths; UTF-16 APIs on Windows)
// ---------------------------------------------------------------------------------------------

#ifdef _WIN32
// laya_utf8_to_wide (laya.h) is empty for invalid UTF-8; here that is an error, not an empty name
std::wstring lc_widen(const std::string & s) {
    if (s.empty()) {
        return std::wstring();
    }
    std::wstring w = laya_utf8_to_wide(s);
    if (w.empty()) {
        lc_fail("path is not valid UTF-8");
    }
    return w;
}

std::string lc_narrow(const wchar_t * w) {
    const int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) {
        lc_fail("file name is not valid UTF-16");
    }
    std::string s((size_t) n, '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, w, -1, &s[0], n, nullptr, nullptr);
    s.resize((size_t) n - 1);
    return s;
}
#endif

struct lc_stat_info {
    bool     exists  = false;
    bool     is_dir  = false;
    bool     is_file = false;
    uint64_t size    = 0;
};

// follows symlinks, like Path.is_file() / is_dir() (HF snapshot files are symlinks into blobs)
lc_stat_info lc_stat(const std::string & path) {
    lc_stat_info r;
#ifdef _WIN32
    struct _stat64 st;
    if (_wstat64(lc_widen(path).c_str(), &st) != 0) {
        return r;
    }
    r.exists  = true;
    r.is_dir  = (st.st_mode & _S_IFMT) == _S_IFDIR;
    r.is_file = (st.st_mode & _S_IFMT) == _S_IFREG;
    r.size    = (uint64_t) st.st_size;
#else
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return r;
    }
    r.exists  = true;
    r.is_dir  = S_ISDIR(st.st_mode);
    r.is_file = S_ISREG(st.st_mode);
    r.size    = (uint64_t) st.st_size;
#endif
    return r;
}

std::vector<std::string> lc_list_dir(const std::string & dir) {
    std::vector<std::string> names;
#ifdef _WIN32
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((lc_widen(dir) + L"\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        lc_fail("cannot list directory '" + dir + "'");
    }
    do {
        const std::string name = lc_narrow(fd.cFileName);
        if (name != "." && name != "..") {
            names.push_back(name);
        }
    } while (FindNextFileW(h, &fd));
    FindClose(h);
#else
    DIR * d = opendir(dir.c_str());
    if (!d) {
        lc_fail("cannot list directory '" + dir + "'");
    }
    while (struct dirent * e = readdir(d)) {
        const std::string name = e->d_name;
        if (name != "." && name != "..") {
            names.push_back(name);
        }
    }
    closedir(d);
#endif
    return names;
}

bool lc_rename_replace(const std::string & from, const std::string & to) {
#ifdef _WIN32
    return MoveFileExW(lc_widen(from).c_str(), lc_widen(to).c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return rename(from.c_str(), to.c_str()) == 0;
#endif
}

bool lc_seek(FILE * f, uint64_t off) {
    if (off > (uint64_t) std::numeric_limits<int64_t>::max()) {
        return false;
    }
#ifdef _WIN32
    return _fseeki64(f, (__int64) off, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t) off, SEEK_SET) == 0;
#endif
}

// 16 hex digits for temporary file names (unique per process and call, not a secret)
std::string lc_random_hex() {
    static uint64_t counter = 0;
    std::random_device rd;
    const uint64_t t = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    const uint64_t x = (((uint64_t) rd() << 32) ^ (uint64_t) rd()) ^ (t * 0x9E3779B97F4A7C15ull) ^ (++counter * 0xBF58476D1CE4E5B9ull);
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) x);
    return buf;
}

// a new file for writing: fails when the name exists (also as a symlink), so two writers never share it
FILE * lc_create_excl(const std::string & path) {
#ifdef _WIN32
    return _wfopen(lc_widen(path).c_str(), L"wbx");
#else
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0666);
    if (fd < 0) {
        return nullptr;
    }
    FILE * f = fdopen(fd, "wb");
    if (!f) {
        close(fd);
    }
    return f;
#endif
}

// data on the disk before the file is renamed into place: a power loss must not leave a complete
// looking GGUF with zeroed data (a later cache hit only checks the layout)
bool lc_sync_file(FILE * f) {
    if (fflush(f) != 0) {
        return false;
    }
#ifdef _WIN32
    return FlushFileBuffers((HANDLE) _get_osfhandle(_fileno(f))) != 0;
#elif defined(__APPLE__)
    // fsync() on macOS does not flush the drive cache
    return fcntl(fileno(f), F_FULLFSYNC) == 0 || fsync(fileno(f)) == 0;
#else
    return fsync(fileno(f)) == 0;
#endif
}

struct lc_file_closer {
    void operator()(FILE * f) const {
        if (f) {
            fclose(f);
        }
    }
};
using lc_file = std::unique_ptr<FILE, lc_file_closer>;

std::string lc_join(const std::string & dir, const std::string & name) {
    if (dir.empty()) {
        return name;
    }
    const char last = dir.back();
    if (last == '/'
#ifdef _WIN32
        || last == '\\'
#endif
    ) {
        return dir + name;
    }
    return dir + "/" + name;
}

// Path(dir).name: last component, "" for "." or "/"
std::string lc_path_name(const std::string & path) {
    std::vector<std::string> parts;
    std::string cur;
    for (char c : path) {
        bool sep = c == '/';
#ifdef _WIN32
        sep = sep || c == '\\';
#endif
        if (sep) {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    parts.push_back(cur);
    std::string name;
    for (const std::string & p : parts) {
        if (!p.empty() && p != ".") {
            name = p;
        }
    }
#ifdef _WIN32
    if (name.size() == 2 && name[1] == ':') {
        name.clear(); // drive
    }
#endif
    return name;
}

std::string lc_read_file(const std::string & path, uint64_t max_size) {
    const lc_stat_info st = lc_stat(path);
    if (!st.is_file) {
        lc_fail("missing file '" + path + "'");
    }
    if (st.size > max_size) {
        lc_fail("'" + path + "' is too large (" + std::to_string(st.size) + " bytes, limit " + std::to_string(max_size) + ")");
    }
    lc_file f(laya_convert_fopen(path, "rb"));
    if (!f) {
        lc_fail("cannot open '" + path + "'");
    }
    std::string data((size_t) st.size, '\0');
    if (!data.empty() && fread(&data[0], 1, data.size(), f.get()) != data.size()) {
        lc_fail("cannot read '" + path + "'");
    }
    if (fgetc(f.get()) != EOF) {
        lc_fail("'" + path + "' changed while reading");
    }
    return data;
}

bool lc_valid_utf8(const std::string & s) {
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        const unsigned char c = (unsigned char) s[i];
        size_t len;
        uint32_t cp;
        if (c < 0x80) {
            i++;
            continue;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2; cp = c & 0x1F;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3; cp = c & 0x0F;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4; cp = c & 0x07;
        } else {
            return false;
        }
        if (i + len > n) {
            return false;
        }
        for (size_t k = 1; k < len; k++) {
            const unsigned char cc = (unsigned char) s[i + k];
            if ((cc & 0xC0) != 0x80) {
                return false;
            }
            cp = (cp << 6) | (cc & 0x3F);
        }
        if ((len == 2 && cp < 0x80) || (len == 3 && cp < 0x800) || (len == 4 && (cp < 0x10000 || cp > 0x10FFFF)) || (cp >= 0xD800 && cp <= 0xDFFF)) {
            return false;
        }
        i += len;
    }
    return true;
}

const uint64_t LC_MAX_CONFIG_BYTES    = 64ull << 20;  // JSON configs, README.md
const uint64_t LC_MAX_TOKENIZER_BYTES = 256ull << 20; // tokenizer.json (mmBERT: 34 MB; parsed JSON takes ~10-20x the file in memory)
const uint64_t LC_MAX_ST_HEADER_BYTES = 100ull << 20; // safetensors header, the limit of the safetensors library
const int64_t  LC_MAX_VOCAB           = 1ll << 24;    // vocab_size (largest real vocabs are ~1M); bounds the [PAD<i>] padding
const uint64_t LC_MAX_BLOCKS          = 65536;        // num_hidden_layers; the largest value Python converts in reasonable time
// JSON nesting: nlohmann parses without recursion, but copying, dump() and destruction of a parsed
// value recurse, so a deeply nested input would overflow the stack. 127 is what serde_json (the
// tokenizers library, which Python loads tokenizer.json with) accepts; real checkpoint JSON nests < 10.
const int      LC_MAX_JSON_DEPTH      = 127;

// deepest [ / { nesting of a JSON text, counted outside strings (the text is not validated here)
int lc_json_depth(const std::string & text) {
    int  depth = 0, max_depth = 0;
    bool in_str = false;
    for (size_t i = 0; i < text.size(); i++) {
        const char ch = text[i];
        if (in_str) {
            if (ch == '\\') {
                i++;
            } else if (ch == '"') {
                in_str = false;
            }
        } else if (ch == '"') {
            in_str = true;
        } else if (ch == '[' || ch == '{') {
            max_depth = std::max(max_depth, ++depth);
        } else if (ch == ']' || ch == '}') {
            depth = std::max(0, depth - 1);
        }
    }
    return max_depth;
}

void lc_check_json_depth(const std::string & text, const std::string & what) {
    const int d = lc_json_depth(text);
    if (d > LC_MAX_JSON_DEPTH) {
        lc_fail(what + ": JSON nested " + std::to_string(d) + " levels deep (limit " + std::to_string(LC_MAX_JSON_DEPTH) + ")");
    }
}

// JSON that Python's json module reads but nlohmann rejects: NaN / Infinity / -Infinity outside
// strings and lone UTF-16 surrogate escapes. Only called on a text nlohmann failed to parse.
bool lc_json_python_only(const std::string & text) {
    bool in_str = false;
    for (size_t i = 0; i < text.size(); i++) {
        const char ch = text[i];
        if (in_str) {
            if (ch == '\\' && i + 1 < text.size()) {
                if (text[i + 1] == 'u' && i + 5 < text.size() && (text[i + 2] == 'd' || text[i + 2] == 'D') &&
                    strchr("89abcdefABCDEF", text[i + 3]) != nullptr) {
                    return true; // \uD800-\uDFFF (a valid pair parses, so this one is lone or broken)
                }
                i++;
            } else if (ch == '"') {
                in_str = false;
            }
        } else if (ch == '"') {
            in_str = true;
        } else if (text.compare(i, 3, "NaN") == 0 || text.compare(i, 8, "Infinity") == 0) {
            return true;
        }
    }
    return false;
}

// json.load(open(path, encoding="utf-8")): valid UTF-8, no BOM, no NaN / Infinity (nlohmann rejects them, Python accepts)
template <typename J>
J lc_parse_json(const std::string & text, const std::string & what) {
    if (!lc_valid_utf8(text)) {
        lc_fail(what + ": not valid UTF-8");
    }
    if (text.size() >= 3 && (unsigned char) text[0] == 0xEF && (unsigned char) text[1] == 0xBB && (unsigned char) text[2] == 0xBF) {
        lc_fail(what + ": starts with a UTF-8 BOM (Python json.load rejects it)");
    }
    lc_check_json_depth(text, what);
    J j;
    try {
        j = J::parse(text);
    } catch (const std::exception & e) {
        lc_fail(what + ": invalid JSON: " + e.what());
    }
    return j;
}

template <typename J>
J lc_load_json(const std::string & path, uint64_t max_size = LC_MAX_CONFIG_BYTES) {
    return lc_parse_json<J>(lc_read_file(path, max_size), path);
}

// ---------------------------------------------------------------------------------------------
// JSON value helpers with Python semantics
// ---------------------------------------------------------------------------------------------

// Python truthiness of a json value
template <typename J>
bool lc_truthy(const J & v) {
    switch (v.type()) {
        case J::value_t::null:            return false;
        case J::value_t::boolean:         return v.template get<bool>();
        case J::value_t::number_integer:  return v.template get<int64_t>() != 0;
        case J::value_t::number_unsigned: return v.template get<uint64_t>() != 0;
        case J::value_t::number_float:    return v.template get<double>() != 0.0;
        case J::value_t::string:          return !v.template get_ref<const std::string &>().empty();
        case J::value_t::array:
        case J::value_t::object:          return !v.empty();
        default:                          return true;
    }
}

// d.get(key): nullptr when absent (a present null is a json null)
template <typename J>
const J * lc_get(const J & obj, const std::string & key) {
    if (!obj.is_object()) {
        return nullptr;
    }
    auto it = obj.find(key);
    return it == obj.end() ? nullptr : &*it;
}

// d.get(key) is not None
template <typename J>
const J * lc_get_nn(const J & obj, const std::string & key) {
    const J * v = lc_get(obj, key);
    return v && !v->is_null() ? v : nullptr;
}

template <typename J>
bool lc_is_int(const J & v) {
    return v.is_number_integer() || v.is_number_unsigned();
}

// serde u32 as the tokenizers library reads it: a JSON integer in [0, 2^32), no float, no bool
template <typename J>
bool lc_is_serde_u32(const J & v) {
    return v.is_number_unsigned() && v.template get<uint64_t>() <= UINT32_MAX;
}

// struct.pack("<I", v): Python int in [0, 2^32) (bools are refused)
template <typename J>
uint32_t lc_u32(const J & v, const std::string & what) {
    if (v.is_number_unsigned() && v.template get<uint64_t>() <= UINT32_MAX) {
        return (uint32_t) v.template get<uint64_t>();
    }
    if (v.is_number_integer() && v.template get<int64_t>() >= 0 && v.template get<int64_t>() <= (int64_t) UINT32_MAX) {
        return (uint32_t) v.template get<int64_t>();
    }
    lc_fail(what + ": expected an integer in [0, 4294967295], got " + v.dump());
}

// struct.pack("<i", v)
template <typename J>
int32_t lc_i32(const J & v, const std::string & what) {
    if (v.is_number_unsigned() && v.template get<uint64_t>() <= (uint64_t) INT32_MAX) {
        return (int32_t) v.template get<uint64_t>();
    }
    if (v.is_number_integer() && !v.is_number_unsigned() && v.template get<int64_t>() >= INT32_MIN && v.template get<int64_t>() <= INT32_MAX) {
        return (int32_t) v.template get<int64_t>();
    }
    lc_fail(what + ": expected an integer in the int32 range, got " + v.dump());
}

float lc_f32_from_double(double d, const std::string & what) {
    const float f = (float) d;
    if (std::isinf(f) && !std::isinf(d)) {
        lc_fail(what + ": float too large to pack with f format");
    }
    return f;
}

// struct.pack("<f", v): Python int or float
template <typename J>
float lc_f32(const J & v, const std::string & what) {
    if (v.is_number_float()) {
        return lc_f32_from_double(v.template get<double>(), what);
    }
    if (v.is_number_unsigned()) {
        return lc_f32_from_double((double) v.template get<uint64_t>(), what);
    }
    if (v.is_number_integer()) {
        return lc_f32_from_double((double) v.template get<int64_t>(), what);
    }
    lc_fail(what + ": expected a number, got " + v.dump());
}

template <typename J>
std::string lc_str(const J & v, const std::string & what) {
    if (!v.is_string()) {
        lc_fail(what + ": expected a string, got " + v.dump());
    }
    return v.template get<std::string>();
}

bool lc_ends_with(const std::string & s, const std::string & suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool lc_starts_with(const std::string & s, const std::string & prefix) {
    return s.compare(0, prefix.size(), prefix) == 0;
}

// str.replace(from, to): one forward pass (linear; replacing in place would move the tail at every hit)
std::string lc_replace_all(const std::string & s, const std::string & from, const std::string & to) {
    if (from.empty()) {
        return s;
    }
    std::string out;
    out.reserve(s.size());
    size_t pos = 0;
    for (size_t hit; (hit = s.find(from, pos)) != std::string::npos; pos = hit + from.size()) {
        out.append(s, pos, hit - pos);
        out += to;
    }
    out.append(s, pos, std::string::npos);
    return out;
}

// ---------------------------------------------------------------------------------------------
// ordered KV store with gguf-py GGUFWriter semantics
// ---------------------------------------------------------------------------------------------

// gguf_type values
enum lc_vtype : uint32_t {
    LC_U32 = 4, LC_I32 = 5, LC_F32 = 6, LC_BOOL = 7, LC_STR = 8, LC_ARR = 9,
};

struct lc_kv {
    std::string key;
    lc_vtype    type     = LC_U32;
    lc_vtype    arr_type = LC_U32;
    uint32_t    u32 = 0;
    int32_t     i32 = 0;
    float       f32 = 0.0f;
    bool        b   = false;
    std::string s;
    std::vector<std::string> as;
    std::vector<int32_t>     ai;
    std::vector<float>       af;
    std::vector<uint8_t>     ab;

    size_t arr_n() const {
        switch (arr_type) {
            case LC_STR:  return as.size();
            case LC_I32:  return ai.size();
            case LC_F32:  return af.size();
            case LC_BOOL: return ab.size();
            default:      return 0;
        }
    }
};

struct lc_kvs {
    std::vector<lc_kv> v;

    // a key written twice keeps its first position (gguf-py kv_data is a dict)
    void put(lc_kv kv) {
        for (lc_kv & e : v) {
            if (e.key == kv.key) {
                e = std::move(kv);
                return;
            }
        }
        v.push_back(std::move(kv));
    }
    void u32(const std::string & key, uint32_t x) { lc_kv kv; kv.key = key; kv.type = LC_U32; kv.u32 = x; put(std::move(kv)); }
    void i32(const std::string & key, int32_t x)  { lc_kv kv; kv.key = key; kv.type = LC_I32; kv.i32 = x; put(std::move(kv)); }
    void f32(const std::string & key, float x)    { lc_kv kv; kv.key = key; kv.type = LC_F32; kv.f32 = x; put(std::move(kv)); }
    void boolean(const std::string & key, bool x) { lc_kv kv; kv.key = key; kv.type = LC_BOOL; kv.b = x; put(std::move(kv)); }
    // add_string skips empty values
    void str(const std::string & key, const std::string & x) {
        if (x.empty()) {
            return;
        }
        lc_kv kv; kv.key = key; kv.type = LC_STR; kv.s = x; put(std::move(kv));
    }
    // add_array skips empty arrays
    void arr_str(const std::string & key, std::vector<std::string> x) {
        if (x.empty()) {
            return;
        }
        lc_kv kv; kv.key = key; kv.type = LC_ARR; kv.arr_type = LC_STR; kv.as = std::move(x); put(std::move(kv));
    }
    void arr_i32(const std::string & key, std::vector<int32_t> x) {
        if (x.empty()) {
            return;
        }
        lc_kv kv; kv.key = key; kv.type = LC_ARR; kv.arr_type = LC_I32; kv.ai = std::move(x); put(std::move(kv));
    }
    void arr_f32(const std::string & key, std::vector<float> x) {
        if (x.empty()) {
            return;
        }
        lc_kv kv; kv.key = key; kv.type = LC_ARR; kv.arr_type = LC_F32; kv.af = std::move(x); put(std::move(kv));
    }
    void arr_bool(const std::string & key, std::vector<uint8_t> x) {
        if (x.empty()) {
            return;
        }
        lc_kv kv; kv.key = key; kv.type = LC_ARR; kv.arr_type = LC_BOOL; kv.ab = std::move(x); put(std::move(kv));
    }
};

// add_array(key, list) with the element type of the first item (GGUFValueType.get_type)
void lc_put_json_array(lc_kvs & kvs, const std::string & key, const ordered_json & arr, const std::string & what) {
    if (!arr.is_array()) {
        lc_fail(what + ": expected a list, got " + arr.dump());
    }
    if (arr.empty()) {
        return;
    }
    const ordered_json & first = arr[0];
    auto same_kind = [&](const ordered_json & e) {
        if (first.is_number_float())   return e.is_number_float();
        if (first.is_boolean())        return e.is_boolean();
        if (lc_is_int(first))          return lc_is_int(e) && !e.is_boolean();
        if (first.is_string())         return e.is_string();
        return false;
    };
    for (const ordered_json & e : arr) {
        if (!same_kind(e)) {
            lc_fail(what + ": all items of a GGUF array must have the same type (" + arr.dump() + ")");
        }
    }
    if (first.is_number_float()) {
        std::vector<float> x;
        for (const ordered_json & e : arr) {
            x.push_back(lc_f32(e, what));
        }
        kvs.arr_f32(key, std::move(x));
    } else if (first.is_boolean()) {
        std::vector<uint8_t> x;
        for (const ordered_json & e : arr) {
            x.push_back(e.get<bool>() ? 1 : 0);
        }
        kvs.arr_bool(key, std::move(x));
    } else if (lc_is_int(first)) {
        std::vector<int32_t> x;
        for (const ordered_json & e : arr) {
            x.push_back(lc_i32(e, what));
        }
        kvs.arr_i32(key, std::move(x));
    } else {
        std::vector<std::string> x;
        for (const ordered_json & e : arr) {
            x.push_back(e.get<std::string>());
        }
        kvs.arr_str(key, std::move(x));
    }
}

// ---------------------------------------------------------------------------------------------
// safetensors (gguf-py utility.SafetensorsLocal), strict bounds checks
// ---------------------------------------------------------------------------------------------

enum lc_dtype { LC_DT_F32, LC_DT_F16, LC_DT_BF16, LC_DT_F64 };

size_t lc_dtype_size(lc_dtype t) {
    switch (t) {
        case LC_DT_F32:  return 4;
        case LC_DT_F16:  return 2;
        case LC_DT_BF16: return 2;
        case LC_DT_F64:  return 8;
    }
    return 0;
}

const char * lc_dtype_name(lc_dtype t) {
    switch (t) {
        case LC_DT_F32:  return "F32";
        case LC_DT_F16:  return "F16";
        case LC_DT_BF16: return "BF16";
        case LC_DT_F64:  return "F64";
    }
    return "?";
}

struct lc_st_tensor {
    std::string          name;
    lc_dtype             dtype = LC_DT_F32;
    std::vector<int64_t> shape;   // safetensors (row-major) order
    uint64_t             n_elem = 0;
    uint64_t             offset = 0; // absolute file offset
    uint64_t             nbytes = 0;
    size_t               file   = 0;
};

const uint64_t LC_MAX_DIM   = 1ull << 40;
const uint64_t LC_MAX_ELEMS = 1ull << 44;

void lc_st_read(const std::string & path, size_t file_idx, std::vector<lc_st_tensor> & out) {
    const lc_stat_info st = lc_stat(path);
    if (!st.is_file) {
        lc_fail("missing file '" + path + "'");
    }
    lc_file f(laya_convert_fopen(path, "rb"));
    if (!f) {
        lc_fail("cannot open '" + path + "'");
    }
    const uint64_t file_size = st.size;
    if (file_size < 8) {
        lc_fail(path + ": truncated safetensors file (" + std::to_string(file_size) + " bytes)");
    }
    unsigned char hdr[8];
    if (fread(hdr, 1, 8, f.get()) != 8) {
        lc_fail(path + ": cannot read the header length");
    }
    uint64_t n = 0;
    for (int i = 7; i >= 0; i--) {
        n = (n << 8) | hdr[i];
    }
    if (n > file_size - 8) {
        lc_fail(path + ": header length " + std::to_string(n) + " exceeds the file size " + std::to_string(file_size));
    }
    if (n > LC_MAX_ST_HEADER_BYTES) {
        lc_fail(path + ": header length " + std::to_string(n) + " is larger than the 100 MiB limit");
    }
    std::string text((size_t) n, '\0');
    if (n > 0 && fread(&text[0], 1, text.size(), f.get()) != text.size()) {
        lc_fail(path + ": cannot read the header");
    }
    const json header = lc_parse_json<json>(text, path + " header");
    if (!header.is_object()) {
        lc_fail(path + ": the header is not a JSON object");
    }
    const uint64_t data_start = 8 + n;
    const uint64_t data_size  = file_size - data_start;

    for (auto it = header.begin(); it != header.end(); ++it) {
        const std::string & name = it.key();
        if (name == "__metadata__") {
            continue;
        }
        const json & m = it.value();
        const std::string what = path + ": tensor '" + name + "'";
        if (!m.is_object()) {
            lc_fail(what + ": entry is not an object");
        }
        const json * jdt = lc_get(m, "dtype");
        const json * jsh = lc_get(m, "shape");
        const json * jof = lc_get(m, "data_offsets");
        if (!jdt || !jdt->is_string() || !jsh || !jsh->is_array() || !jof || !jof->is_array()) {
            lc_fail(what + ": needs dtype (string), shape (list) and data_offsets (list)");
        }
        lc_st_tensor t;
        t.name = name;
        t.file = file_idx;
        const std::string & dt = jdt->get_ref<const std::string &>();
        if      (dt == "F32")  t.dtype = LC_DT_F32;
        else if (dt == "F16")  t.dtype = LC_DT_F16;
        else if (dt == "BF16") t.dtype = LC_DT_BF16;
        else if (dt == "F64")  t.dtype = LC_DT_F64;
        else if (dt == "I64" || dt == "I32" || dt == "I16" || dt == "I8" || dt == "U8" || dt == "U16" || dt == "U32" || dt == "U64" ||
                 dt == "BOOL" || dt == "F8_E4M3" || dt == "F8_E5M2") {
            lc_fail(what + ": dtype " + dt + " is not supported (F32, F16, BF16, F64 are)");
        } else {
            lc_fail(what + ": unknown dtype '" + dt + "'");
        }
        if (jsh->empty() || jsh->size() > 4) {
            lc_fail(what + ": " + std::to_string(jsh->size()) + " dimensions (1 to 4 are supported)");
        }
        uint64_t n_elem = 1;
        for (const json & d : *jsh) {
            if (!lc_is_int(d) || d.is_boolean() || (d.is_number_integer() && !d.is_number_unsigned() && d.get<int64_t>() < 0)) {
                lc_fail(what + ": bad shape " + jsh->dump());
            }
            const uint64_t dim = d.get<uint64_t>();
            if (dim == 0 || dim > LC_MAX_DIM) {
                lc_fail(what + ": bad dimension " + std::to_string(dim) + " in shape " + jsh->dump());
            }
            if (n_elem > LC_MAX_ELEMS / dim) {
                lc_fail(what + ": too many elements in shape " + jsh->dump());
            }
            n_elem *= dim;
            t.shape.push_back((int64_t) dim);
        }
        if (jof->size() != 2 || !lc_is_int((*jof)[0]) || !lc_is_int((*jof)[1]) ||
            !(*jof)[0].is_number_unsigned() || !(*jof)[1].is_number_unsigned()) {
            lc_fail(what + ": bad data_offsets " + jof->dump());
        }
        const uint64_t begin = (*jof)[0].get<uint64_t>();
        const uint64_t end   = (*jof)[1].get<uint64_t>();
        if (begin > end || end > data_size) {
            lc_fail(what + ": data_offsets [" + std::to_string(begin) + ", " + std::to_string(end) + "] are outside the data section (" + std::to_string(data_size) + " bytes)");
        }
        const uint64_t nbytes = n_elem * lc_dtype_size(t.dtype);
        if (end - begin != nbytes) {
            lc_fail(what + ": " + dt + " shape " + jsh->dump() + " needs " + std::to_string(nbytes) + " bytes, data_offsets give " + std::to_string(end - begin));
        }
        t.n_elem = n_elem;
        t.nbytes = nbytes;
        t.offset = data_start + begin;
        out.push_back(std::move(t));
    }
}

// ---------------------------------------------------------------------------------------------
// fp16 (numpy on AArch64 / x86-64 F16C) and bf16 conversions
// ---------------------------------------------------------------------------------------------

uint32_t lc_f32_bits(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    return x;
}

float lc_bits_f32(uint32_t x) {
    float f;
    memcpy(&f, &x, 4);
    return f;
}

// ---------------------------------------------------------------------------------------------
// YAML subset for the README.md front matter (gguf-py Metadata.load_model_card + PyYAML safe_load)
// ---------------------------------------------------------------------------------------------

struct lc_yval {
    enum kind_t { NUL, BOOL, INT, FLOAT, TIMESTAMP, STR, LIST, OTHER } kind = NUL;
    std::string          s;    // STR: the value, OTHER: why it is not supported
    std::vector<lc_yval> list;
};

// the implicit resolvers of PyYAML (yaml/resolver.py, YAML 1.1) for plain scalars
bool lc_is_digit(char c) { return c >= '0' && c <= '9'; }

bool lc_yaml_is_bool(const std::string & s) {
    static const char * v[] = { "yes", "Yes", "YES", "no", "No", "NO", "true", "True", "TRUE", "false", "False", "FALSE", "on", "On", "ON", "off", "Off", "OFF" };
    for (const char * x : v) {
        if (s == x) {
            return true;
        }
    }
    return false;
}

bool lc_yaml_is_null(const std::string & s) {
    return s.empty() || s == "~" || s == "null" || s == "Null" || s == "NULL";
}

bool lc_in(const char * set, char c) {
    return c != '\0' && strchr(set, c) != nullptr;
}

// end of the run of chars from `set` that starts at i
size_t lc_span(const std::string & s, size_t i, const char * set) {
    while (i < s.size() && lc_in(set, s[i])) {
        i++;
    }
    return i;
}

// (:[0-5]?[0-9])+ from i, returns end or npos
size_t lc_yaml_sexagesimal(const std::string & s, size_t i) {
    size_t n = 0;
    while (i < s.size() && s[i] == ':') {
        size_t j = i + 1;
        if (j + 1 < s.size() && s[j] >= '0' && s[j] <= '5' && lc_is_digit(s[j + 1])) {
            j += 2;
        } else if (j < s.size() && lc_is_digit(s[j])) {
            j += 1;
        } else {
            return std::string::npos;
        }
        i = j;
        n++;
    }
    return n > 0 ? i : std::string::npos;
}

bool lc_yaml_is_int(const std::string & s) {
    size_t i = 0;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) {
        i++;
    }
    const std::string r = s.substr(i);
    if (r.empty()) {
        return false;
    }
    if (r.size() > 2 && r[0] == '0' && r[1] == 'b') {
        return lc_span(r, 2, "01_") == r.size();
    }
    if (r.size() > 2 && r[0] == '0' && r[1] == 'x') {
        return lc_span(r, 2, "0123456789abcdefABCDEF_") == r.size();
    }
    if (r == "0") {
        return true;
    }
    if (r[0] == '0') {
        return r.size() > 1 && lc_span(r, 1, "01234567_") == r.size();
    }
    if (r[0] >= '1' && r[0] <= '9') {
        const size_t j = lc_span(r, 1, "0123456789_");
        if (j == r.size()) {
            return true;
        }
        return lc_yaml_sexagesimal(r, j) == r.size();
    }
    return false;
}

// [eE][-+][0-9]+ from i (optional), returns end or npos when malformed
size_t lc_yaml_exp(const std::string & s, size_t i) {
    if (i < s.size() && (s[i] == 'e' || s[i] == 'E')) {
        if (i + 2 < s.size() && (s[i + 1] == '-' || s[i + 1] == '+') && lc_is_digit(s[i + 2])) {
            return lc_span(s, i + 2, "0123456789");
        }
        return std::string::npos;
    }
    return i;
}

bool lc_yaml_is_float(const std::string & s) {
    if (s == ".nan" || s == ".NaN" || s == ".NAN") {
        return true;
    }
    size_t i = 0;
    const bool sign = i < s.size() && (s[i] == '-' || s[i] == '+');
    if (sign) {
        i++;
    }
    const std::string r = s.substr(i);
    if (r == ".inf" || r == ".Inf" || r == ".INF") {
        return true;
    }
    if (r.empty()) {
        return false;
    }
    if (lc_is_digit(r[0])) {
        size_t j = lc_span(r, 1, "0123456789_");
        if (j < r.size() && r[j] == '.') {
            j = lc_span(r, j + 1, "0123456789_");
            return lc_yaml_exp(r, j) == r.size();
        }
        if (j < r.size() && r[j] == ':') {
            j = lc_yaml_sexagesimal(r, j);
            if (j == std::string::npos || j >= r.size() || r[j] != '.') {
                return false;
            }
            return lc_span(r, j + 1, "0123456789_") == r.size();
        }
        return false;
    }
    if (!sign && r[0] == '.' && r.size() > 1 && lc_is_digit(r[1])) {
        const size_t j = lc_span(r, 2, "0123456789_");
        return lc_yaml_exp(r, j) == r.size();
    }
    return false;
}

bool lc_yaml_is_timestamp(const std::string & s) {
    auto digits = [&](size_t & i, size_t lo, size_t hi) {
        size_t n = 0;
        while (i < s.size() && lc_is_digit(s[i]) && n < hi) {
            i++; n++;
        }
        return n >= lo;
    };
    size_t i = 0;
    if (!digits(i, 4, 4) || i >= s.size() || s[i] != '-') return false;
    i++;
    const size_t m0 = i;
    if (!digits(i, 1, 2) || i >= s.size() || s[i] != '-') return false;
    const bool m2 = i - m0 == 2;
    i++;
    const size_t d0 = i;
    if (!digits(i, 1, 2)) return false;
    const bool d2 = i - d0 == 2;
    if (i == s.size()) {
        return m2 && d2;
    }
    if (s[i] == 'T' || s[i] == 't') {
        i++;
    } else if (s[i] == ' ' || s[i] == '\t') {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) i++;
    } else {
        return false;
    }
    if (!digits(i, 1, 2) || i >= s.size() || s[i] != ':') return false;
    i++;
    if (!digits(i, 2, 2) || i >= s.size() || s[i] != ':') return false;
    i++;
    if (!digits(i, 2, 2)) return false;
    if (i < s.size() && s[i] == '.') {
        i = lc_span(s, i + 1, "0123456789");
    }
    const size_t before_tz = i;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) i++;
    if (i == s.size()) {
        return before_tz == s.size();
    }
    if (s[i] == 'Z') {
        return i + 1 == s.size();
    }
    if (s[i] == '-' || s[i] == '+') {
        i++;
        if (!digits(i, 1, 2)) return false;
        if (i < s.size() && s[i] == ':') {
            i++;
            if (!digits(i, 2, 2)) return false;
        }
        return i == s.size();
    }
    return false;
}

lc_yval lc_yaml_plain(const std::string & s) {
    lc_yval v;
    if (lc_yaml_is_null(s)) {
        v.kind = lc_yval::NUL;
    } else if (lc_yaml_is_bool(s)) {
        v.kind = lc_yval::BOOL;
    } else if (lc_yaml_is_int(s)) {
        v.kind = lc_yval::INT;
    } else if (lc_yaml_is_float(s)) {
        v.kind = lc_yval::FLOAT;
    } else if (lc_yaml_is_timestamp(s)) {
        v.kind = lc_yval::TIMESTAMP;
    } else if (s == "=" || s == "<<") {
        v.kind = lc_yval::OTHER;
        v.s = "YAML value / merge key";
    } else {
        v.kind = lc_yval::STR;
        v.s = s;
    }
    return v;
}

lc_yval lc_yaml_other(const std::string & why) {
    lc_yval v;
    v.kind = lc_yval::OTHER;
    v.s = why;
    return v;
}

size_t lc_indent(const std::string & line) {
    size_t i = 0;
    while (i < line.size() && line[i] == ' ') {
        i++;
    }
    return i;
}

bool lc_yaml_blank(const std::string & line) {
    const size_t i = lc_indent(line);
    return i == line.size() || line[i] == '#';
}

void lc_rtrim(std::string & s) {
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
}

void lc_append_utf8(std::string & out, uint32_t cp) {
    if (cp < 0x80) {
        out += (char) cp;
    } else if (cp < 0x800) {
        out += (char) (0xC0 | (cp >> 6));
        out += (char) (0x80 | (cp & 0x3F));
    } else if (cp < 0x10000) {
        out += (char) (0xE0 | (cp >> 12));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    } else {
        out += (char) (0xF0 | (cp >> 18));
        out += (char) (0x80 | ((cp >> 12) & 0x3F));
        out += (char) (0x80 | ((cp >> 6) & 0x3F));
        out += (char) (0x80 | (cp & 0x3F));
    }
}

// quoted scalar starting at s[i] (quote char), single line; sets i past the closing quote
bool lc_yaml_quoted(const std::string & s, size_t & i, std::string & out, std::string & err) {
    const char q = s[i++];
    out.clear();
    while (i < s.size()) {
        const char c = s[i];
        if (q == '\'') {
            if (c == '\'') {
                if (i + 1 < s.size() && s[i + 1] == '\'') {
                    out += '\'';
                    i += 2;
                    continue;
                }
                i++;
                return true;
            }
            out += c;
            i++;
            continue;
        }
        if (c == '"') {
            i++;
            return true;
        }
        if (c != '\\') {
            out += c;
            i++;
            continue;
        }
        if (i + 1 >= s.size()) {
            err = "multi-line double-quoted scalar";
            return false;
        }
        const char e = s[i + 1];
        i += 2;
        int nhex = 0;
        switch (e) {
            case '0':  out += '\0'; break;
            case 'a':  out += '\a'; break;
            case 'b':  out += '\b'; break;
            case 't':  case '\t': out += '\t'; break;
            case 'n':  out += '\n'; break;
            case 'v':  out += '\v'; break;
            case 'f':  out += '\f'; break;
            case 'r':  out += '\r'; break;
            case 'e':  out += '\x1b'; break;
            case ' ':  out += ' '; break;
            case '"':  out += '"'; break;
            case '/':  out += '/'; break;
            case '\\': out += '\\'; break;
            case 'N':  lc_append_utf8(out, 0x85); break;
            case '_':  lc_append_utf8(out, 0xA0); break;
            case 'L':  lc_append_utf8(out, 0x2028); break;
            case 'P':  lc_append_utf8(out, 0x2029); break;
            case 'x':  nhex = 2; break;
            case 'u':  nhex = 4; break;
            case 'U':  nhex = 8; break;
            default:
                err = std::string("unknown escape \\") + e;
                return false;
        }
        if (nhex > 0) {
            if (i + (size_t) nhex > s.size()) {
                err = "short escape";
                return false;
            }
            uint32_t cp = 0;
            for (int k = 0; k < nhex; k++) {
                const char h = s[i + (size_t) k];
                uint32_t d;
                if (h >= '0' && h <= '9') d = (uint32_t) (h - '0');
                else if (h >= 'a' && h <= 'f') d = (uint32_t) (h - 'a' + 10);
                else if (h >= 'A' && h <= 'F') d = (uint32_t) (h - 'A' + 10);
                else { err = "bad escape"; return false; }
                cp = (cp << 4) | d;
            }
            if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) {
                err = "escape outside Unicode scalar values";
                return false;
            }
            lc_append_utf8(out, cp);
            i += (size_t) nhex;
        }
    }
    err = q == '"' ? "multi-line double-quoted scalar" : "multi-line single-quoted scalar";
    return false;
}

// rest of a line after a value: spaces and an optional comment
bool lc_yaml_rest_empty(const std::string & s, size_t i) {
    while (i < s.size() && s[i] == ' ') {
        i++;
    }
    return i == s.size() || (s[i] == '#' && i > 0 && s[i - 1] == ' ');
}

// flow sequence on one line: "[a, 'b', "c"]"
lc_yval lc_yaml_flow_seq(const std::string & s, size_t i) {
    lc_yval out;
    out.kind = lc_yval::LIST;
    i++; // '['
    bool expect_item = true;
    while (true) {
        while (i < s.size() && s[i] == ' ') {
            i++;
        }
        if (i >= s.size()) {
            return lc_yaml_other("multi-line flow sequence");
        }
        const char c = s[i];
        if (c == ']') {
            i++;
            break;
        }
        if (c == ',') {
            if (expect_item) {
                lc_fail("README.md: YAML: empty item in flow sequence");
            }
            expect_item = true;
            i++;
            continue;
        }
        if (!expect_item) {
            lc_fail("README.md: YAML: expected ',' or ']' in flow sequence");
        }
        if (c == '[' || c == '{') {
            return lc_yaml_other("nested flow collection");
        }
        if (c == '\'' || c == '"') {
            std::string v, err;
            if (!lc_yaml_quoted(s, i, v, err)) {
                return lc_yaml_other(err);
            }
            lc_yval item;
            item.kind = lc_yval::STR;
            item.s = v;
            out.list.push_back(item);
        } else {
            if (c == '#' || c == '&' || c == '*' || c == '!' || c == '|' || c == '>' || c == '%' || c == '@' || c == '`') {
                return lc_yaml_other(std::string("flow item starting with '") + c + "'");
            }
            size_t j = i;
            while (j < s.size() && s[j] != ',' && s[j] != ']' && s[j] != '[' && s[j] != '{' && s[j] != '}' &&
                   !(s[j] == ':' && (j + 1 == s.size() || s[j + 1] == ' ' || s[j + 1] == ',' || s[j + 1] == ']')) &&
                   !(s[j] == '#' && j > i && s[j - 1] == ' ')) {
                j++;
            }
            if (j < s.size() && (s[j] == ':' || s[j] == '#' || s[j] == '[' || s[j] == '{' || s[j] == '}')) {
                return lc_yaml_other("mapping or comment inside a flow sequence");
            }
            std::string v = s.substr(i, j - i);
            lc_rtrim(v);
            out.list.push_back(lc_yaml_plain(v));
            i = j;
        }
        expect_item = false;
    }
    if (!lc_yaml_rest_empty(s, i)) {
        lc_fail("README.md: YAML: text after a flow sequence");
    }
    return out;
}

// a value that starts at s[i] on a single line (after "key:" or "- ")
lc_yval lc_yaml_inline_value(const std::string & s, size_t i, bool seq_item) {
    const char c = s[i];
    if (c == '[') {
        return lc_yaml_flow_seq(s, i);
    }
    if (c == '{') {
        return lc_yaml_other("flow mapping");
    }
    if (c == '|' || c == '>') {
        return lc_yaml_other("block scalar");
    }
    if (c == '&' || c == '*' || c == '!') {
        return lc_yaml_other("anchor, alias or tag");
    }
    if (c == '%' || c == '@' || c == '`') {
        lc_fail(std::string("README.md: YAML: a plain scalar cannot start with '") + c + "'");
    }
    if ((c == '-' || c == '?' || c == ':') && (i + 1 == s.size() || s[i + 1] == ' ')) {
        return lc_yaml_other("nested block collection on one line");
    }
    if (c == '\'' || c == '"') {
        std::string v, err;
        size_t j = i;
        if (!lc_yaml_quoted(s, j, v, err)) {
            return lc_yaml_other(err);
        }
        if (!lc_yaml_rest_empty(s, j)) {
            lc_fail("README.md: YAML: text after a quoted scalar");
        }
        lc_yval r;
        r.kind = lc_yval::STR;
        r.s = v;
        return r;
    }
    size_t j = i;
    while (j < s.size()) {
        if (s[j] == '#' && s[j - 1] == ' ') {
            break;
        }
        if (s[j] == ':' && (j + 1 == s.size() || s[j + 1] == ' ')) {
            if (seq_item) {
                return lc_yaml_other("mapping in a block sequence item");
            }
            lc_fail("README.md: YAML: mapping values are not allowed here (" + s + ")");
        }
        j++;
    }
    std::string v = s.substr(i, j - i);
    lc_rtrim(v);
    return lc_yaml_plain(v);
}

using lc_ymap = std::vector<std::pair<std::string, lc_yval>>;

// top-level block mapping with scalar, flow-list and block-list values
lc_ymap lc_yaml_parse(const std::string & text, bool & is_mapping) {
    std::vector<std::string> lines;
    {
        size_t p = 0;
        while (p <= text.size()) {
            const size_t q = text.find('\n', p);
            if (q == std::string::npos) {
                if (p < text.size()) {
                    lines.push_back(text.substr(p));
                }
                break;
            }
            lines.push_back(text.substr(p, q - p));
            p = q + 1;
        }
    }
    lc_ymap out;
    std::unordered_map<std::string, size_t> index; // key -> position in out: a README with millions of keys stays linear
    is_mapping = true;
    auto put = [&](const std::string & key, lc_yval v) {
        auto it = index.find(key);
        if (it != index.end()) {
            out[it->second].second = std::move(v);
            return;
        }
        index.emplace(key, out.size());
        out.emplace_back(key, std::move(v));
    };

    size_t li = 0;
    bool first = true;
    while (li < lines.size()) {
        const std::string & line = lines[li];
        if (lc_yaml_blank(line)) {
            li++;
            continue;
        }
        if (line == "..." ) {
            break;
        }
        if (line.size() >= 3 && line.compare(0, 3, "---") == 0 && (line.size() == 3 || line[3] == ' ')) {
            lc_fail("README.md: YAML: more than one document in the front matter");
        }
        if (lc_indent(line) != 0) {
            lc_fail("README.md: YAML: unexpected indentation at line " + std::to_string(li + 1) + " of the front matter");
        }
        if (line[0] == '%') {
            lc_fail("README.md: YAML directives are not supported");
        }
        if (line[0] == '-' && (line.size() == 1 || line[1] == ' ')) {
            if (first) {
                is_mapping = false; // Python logs an error and uses no model card
                return {};
            }
            lc_fail("README.md: YAML: a sequence entry at the top level of a mapping");
        }
        first = false;

        // key
        std::string key;
        bool key_is_str = true;
        size_t i = 0;
        if (line[0] == '\'' || line[0] == '"') {
            std::string err;
            if (!lc_yaml_quoted(line, i, key, err)) {
                lc_fail("README.md: YAML: " + err);
            }
            while (i < line.size() && line[i] == ' ') i++;
            if (i >= line.size() || line[i] != ':' || !(i + 1 == line.size() || line[i + 1] == ' ')) {
                lc_fail("README.md: YAML: expected ':' after a quoted key at line " + std::to_string(li + 1));
            }
        } else {
            if (line[0] == '?' || line[0] == '[' || line[0] == '{' || line[0] == '&' || line[0] == '*' || line[0] == '!' || line[0] == '|' || line[0] == '>') {
                lc_fail("README.md: YAML: unsupported key syntax at line " + std::to_string(li + 1));
            }
            while (i < line.size() && !(line[i] == ':' && (i + 1 == line.size() || line[i + 1] == ' '))) {
                if (line[i] == '#' && i > 0 && line[i - 1] == ' ') {
                    break;
                }
                i++;
            }
            if (i >= line.size() || line[i] != ':') {
                lc_fail("README.md: YAML: expected 'key: value' at line " + std::to_string(li + 1));
            }
            key = line.substr(0, i);
            lc_rtrim(key);
            key_is_str = lc_yaml_plain(key).kind == lc_yval::STR;
        }
        if (!key_is_str) {
            key = std::string("\x01") + key; // a bool / int / null key never equals a str key
        }
        i++; // ':'
        while (i < line.size() && line[i] == ' ') {
            i++;
        }
        li++;

        const bool empty_value = i >= line.size() || (line[i] == '#');
        if (!empty_value) {
            lc_yval v = lc_yaml_inline_value(line, i, false);
            // more-indented lines continue the value (multi-line plain scalar, block scalar body)
            bool continued = false;
            while (li < lines.size() && (lc_yaml_blank(lines[li]) || lc_indent(lines[li]) > 0)) {
                continued = continued || !lc_yaml_blank(lines[li]);
                li++;
            }
            if (continued && v.kind != lc_yval::OTHER) {
                v = lc_yaml_other("multi-line value");
            }
            put(key, std::move(v));
            continue;
        }

        // empty inline value: a block sequence, a nested mapping, or null
        size_t lj = li;
        while (lj < lines.size() && lc_yaml_blank(lines[lj])) {
            lj++;
        }
        if (lj >= lines.size()) {
            put(key, lc_yval());
            li = lj;
            continue;
        }
        const std::string & nx = lines[lj];
        const size_t ind = lc_indent(nx);
        const bool is_item = ind < nx.size() && nx[ind] == '-' && (ind + 1 == nx.size() || nx[ind + 1] == ' ');
        if (!is_item) {
            if (ind == 0) {
                put(key, lc_yval());
                continue;
            }
            // nested mapping / multi-line scalar: skip its lines
            li = lj;
            while (li < lines.size() && (lc_yaml_blank(lines[li]) || lc_indent(lines[li]) > 0)) {
                li++;
            }
            put(key, lc_yaml_other("nested mapping or multi-line value"));
            continue;
        }
        lc_yval seq;
        seq.kind = lc_yval::LIST;
        li = lj;
        while (li < lines.size()) {
            const std::string & l = lines[li];
            if (lc_yaml_blank(l)) {
                li++;
                continue;
            }
            const size_t k = lc_indent(l);
            if (k < ind) {
                break;
            }
            if (k == ind) {
                if (!(l[k] == '-' && (k + 1 == l.size() || l[k + 1] == ' '))) {
                    if (ind == 0) {
                        break; // next top-level key
                    }
                    lc_fail("README.md: YAML: bad indentation in a block sequence at line " + std::to_string(li + 1));
                }
                size_t p = k + 1;
                while (p < l.size() && l[p] == ' ') p++;
                lc_yval item = (p >= l.size() || l[p] == '#') ? lc_yval() : lc_yaml_inline_value(l, p, true);
                li++;
                // item continuation (nested mapping entry, multi-line scalar)
                bool continued = false;
                while (li < lines.size() && (lc_yaml_blank(lines[li]) || lc_indent(lines[li]) > ind)) {
                    continued = continued || !lc_yaml_blank(lines[li]);
                    li++;
                }
                if (continued) {
                    item = lc_yaml_other("nested value in a block sequence item");
                }
                seq.list.push_back(std::move(item));
                continue;
            }
            lc_fail("README.md: YAML: bad indentation in a block sequence at line " + std::to_string(li + 1));
        }
        put(key, std::move(seq));
    }
    return out;
}

// Python str.splitlines() separators on already-decoded text
std::vector<std::string> lc_splitlines(const std::string & s) {
    std::vector<std::string> out;
    std::string cur;
    size_t i = 0;
    auto flush = [&]() { out.push_back(cur); cur.clear(); };
    while (i < s.size()) {
        const unsigned char c = (unsigned char) s[i];
        if (c == '\r') {
            flush();
            i += (i + 1 < s.size() && s[i + 1] == '\n') ? 2 : 1;
            continue;
        }
        if (c == '\n' || c == 0x0B || c == 0x0C || c == 0x1C || c == 0x1D || c == 0x1E) {
            flush();
            i++;
            continue;
        }
        if (c == 0xC2 && i + 1 < s.size() && (unsigned char) s[i + 1] == 0x85) {
            flush();
            i += 2;
            continue;
        }
        if (c == 0xE2 && i + 2 < s.size() && (unsigned char) s[i + 1] == 0x80 && ((unsigned char) s[i + 2] == 0xA8 || (unsigned char) s[i + 2] == 0xA9)) {
            flush();
            i += 3;
            continue;
        }
        cur += (char) c;
        i++;
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    return out;
}

// Metadata.load_model_card: the YAML between the first two "---" lines; {} without front matter
uint32_t lc_next_cp(const std::string & s, size_t & i);

lc_ymap lc_load_model_card(const std::string & dir) {
    const std::string path = lc_join(dir, "README.md");
    if (!lc_stat(path).is_file) {
        return {};
    }
    std::string content = lc_read_file(path, LC_MAX_CONFIG_BYTES);
    if (!lc_valid_utf8(content)) {
        lc_fail(path + ": not valid UTF-8");
    }
    // universal newlines of open() in text mode
    content = lc_replace_all(content, "\r\n", "\n");
    content = lc_replace_all(content, "\r", "\n");
    const std::vector<std::string> lines = lc_splitlines(content);
    if (lines.empty() || lines[0] != "---") {
        return {};
    }
    std::string yaml;
    for (size_t i = 1; i < lines.size(); i++) {
        if (lines[i] == "---") {
            break;
        }
        yaml += lines[i];
        yaml += "\n";
    }
    if (yaml.empty()) {
        yaml = "\n";
    }
    yaml = lc_replace_all(yaml, "- no\n", "- \"no\"\n");
    yaml = lc_replace_all(yaml, "\t", "  ");
    // PyYAML's Reader rejects the whole stream on a non-printable character (reader.py NON_PRINTABLE),
    // also inside keys the subset below would skip
    for (size_t i = 0; i < yaml.size();) {
        const uint32_t cp = lc_next_cp(yaml, i);
        const bool printable = cp == 0x09 || cp == 0x0A || cp == 0x0D || (cp >= 0x20 && cp <= 0x7E) || cp == 0x85 ||
                               (cp >= 0xA0 && cp <= 0xD7FF) || (cp >= 0xE000 && cp <= 0xFFFD) || (cp >= 0x10000 && cp <= 0x10FFFF);
        if (!printable) {
            char buf[16];
            snprintf(buf, sizeof(buf), "#x%04x", (unsigned) cp);
            lc_fail(path + ": YAML: unacceptable character " + buf + ": special characters are not allowed");
        }
    }
    bool is_mapping = true;
    lc_ymap card = lc_yaml_parse(yaml, is_mapping);
    return is_mapping ? card : lc_ymap{};
}

const lc_yval * lc_card_get(const lc_ymap & card, const std::string & key) {
    for (const auto & kv : card) {
        if (kv.first == key) {
            return &kv.second;
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------------------------
// gguf-py Metadata heuristics for a model id (directory name), ASCII rules of the Python regexes
// ---------------------------------------------------------------------------------------------

char lc_lower(char c) { return (c >= 'A' && c <= 'Z') ? (char) (c - 'A' + 'a') : c; }
char lc_upper(char c) { return (c >= 'a' && c <= 'z') ? (char) (c - 'a' + 'A') : c; }
bool lc_is_alpha(char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || ((unsigned char) c >= 0x80); }
bool lc_is_word(char c) { return lc_is_alpha(c) || lc_is_digit(c) || c == '_'; }

std::string lc_lower_str(std::string s) {
    for (char & c : s) c = lc_lower(c);
    return s;
}

std::string lc_upper_str(std::string s) {
    for (char & c : s) c = lc_upper(c);
    return s;
}

// fullmatch (v|iter)?\d+([.]\d+)*  (IGNORECASE)
bool lc_re_version(const std::string & p) {
    const std::string s = lc_lower_str(p);
    size_t i = 0;
    if (lc_starts_with(s, "iter")) {
        // (v|iter)? is optional: "iter" must then be followed by digits
        i = 4;
    } else if (!s.empty() && s[0] == 'v') {
        i = 1;
    }
    auto rest = [&](size_t j) {
        const size_t d = lc_span(s, j, "0123456789");
        if (d == j) return false;
        j = d;
        while (j < s.size()) {
            if (s[j] != '.') return false;
            const size_t e = lc_span(s, j + 1, "0123456789");
            if (e == j + 1) return false;
            j = e;
        }
        return true;
    };
    return rest(i) || (i > 0 && rest(0));
}

// fullmatch i?q\d(_\w)*|b?fp?(16|32)  (IGNORECASE)
bool lc_re_quant_type(const std::string & p) {
    const std::string s = lc_lower_str(p);
    auto alt1 = [&](size_t i) {
        if (i >= s.size() || s[i] != 'q') return false;
        i++;
        if (i >= s.size() || !lc_is_digit(s[i])) return false;
        i++;
        while (i < s.size()) {
            if (s[i] != '_' || i + 1 >= s.size() || !lc_is_word(s[i + 1])) return false;
            i += 2;
        }
        return true;
    };
    if (alt1(0) || (!s.empty() && s[0] == 'i' && alt1(1))) {
        return true;
    }
    size_t i = 0;
    if (i < s.size() && s[i] == 'b') i++;
    if (i >= s.size() || s[i] != 'f') return false;
    i++;
    if (i < s.size() && s[i] == 'p') i++;
    const std::string r = s.substr(i);
    return r == "16" || r == "32";
}

// fullmatch (([A]|\d+[x])?\d+([._]\d+)?[KMBT][\d]?|small|mini|medium|large|x?xl)  (IGNORECASE)
bool lc_re_size(const std::string & p) {
    const std::string s = lc_lower_str(p);
    if (s == "small" || s == "mini" || s == "medium" || s == "large" || s == "xl" || s == "xxl") {
        return true;
    }
    // \d+([._]\d+)?[kmbt]\d? from j
    auto core = [&](size_t j) {
        const size_t d = lc_span(s, j, "0123456789");
        if (d == j) return false;
        j = d;
        if (j < s.size() && (s[j] == '.' || s[j] == '_')) {
            const size_t e = lc_span(s, j + 1, "0123456789");
            if (e > j + 1) {
                // try with the fraction, then without (backtracking of the regex)
                size_t k = e;
                if (k < s.size() && lc_in("kmbt", s[k])) {
                    k++;
                    if (k == s.size() || (k + 1 == s.size() && lc_is_digit(s[k]))) return true;
                }
            }
        }
        // no fraction
        size_t k = j;
        if (k < s.size() && lc_in("kmbt", s[k])) {
            k++;
            return k == s.size() || (k + 1 == s.size() && lc_is_digit(s[k]));
        }
        return false;
    };
    if (core(0)) return true;
    if (!s.empty() && s[0] == 'a' && core(1)) return true;
    // \d+x prefix
    const size_t d = lc_span(s, 0, "0123456789");
    if (d > 0 && d < s.size() && s[d] == 'x' && core(d + 1)) return true;
    return false;
}

bool lc_is_lower_word(const std::string & w, bool & ascii_only) {
    bool cased = false;
    for (char c : w) {
        if ((unsigned char) c >= 0x80) ascii_only = false;
        if (c >= 'A' && c <= 'Z') return false;
        if (c >= 'a' && c <= 'z') cased = true;
    }
    return cased;
}

// str.title() on ASCII
std::string lc_title(const std::string & w) {
    std::string out;
    bool prev_cased = false;
    for (char c : w) {
        const bool letter = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
        if (letter) {
            out += prev_cased ? lc_lower(c) : lc_upper(c);
        } else {
            out += c;
        }
        prev_cased = letter;
    }
    return out;
}

bool lc_py_space(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f' || c == 0x1C || c == 0x1D || c == 0x1E || c == 0x1F;
}

// Metadata.id_to_title
std::string lc_id_to_title(const std::string & id, bool & ascii_only) {
    std::string s = lc_replace_all(id, "-", " ");
    std::vector<std::string> words;
    std::string cur;
    for (char c : s) {
        if (lc_py_space(c)) {
            if (!cur.empty()) words.push_back(cur);
            cur.clear();
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) words.push_back(cur);
    std::string out;
    for (size_t i = 0; i < words.size(); i++) {
        const std::string & w = words[i];
        // re.match(r'^(v\d+(?:\.\d+)*|\d.*)$', w)
        bool version_like = !w.empty() && lc_is_digit(w[0]);
        if (!version_like && w.size() > 1 && w[0] == 'v') {
            size_t j = lc_span(w, 1, "0123456789");
            bool ok = j > 1;
            while (ok && j < w.size()) {
                if (w[j] != '.') { ok = false; break; }
                const size_t e = lc_span(w, j + 1, "0123456789");
                if (e == j + 1) { ok = false; break; }
                j = e;
            }
            version_like = ok;
        }
        if (i > 0) out += ' ';
        out += (lc_is_lower_word(w, ascii_only) && !version_like) ? lc_title(w) : w;
    }
    return out;
}

struct lc_id_components {
    std::optional<std::string> full, org, basename, finetune, version, size_label;
};

// Metadata.get_model_id_components
lc_id_components lc_model_id_components(const std::string & model_id, int64_t total_params, bool & ascii_only) {
    lc_id_components r;
    for (char c : model_id) {
        if ((unsigned char) c >= 0x80) ascii_only = false;
    }
    if (model_id.find(' ') != std::string::npos) {
        r.full = model_id;
        return r;
    }
    std::string full = model_id;
    const size_t slash = model_id.find('/');
    if (slash != std::string::npos) {
        std::string org = model_id.substr(0, slash);
        full = model_id.substr(slash + 1);
        if (!org.empty() && org[0] != '.') {
            r.org = org;
        }
    }
    r.full = full;

    std::vector<std::string> parts;
    {
        std::string cur;
        for (char c : full) {
            if (c == '-') {
                if (!cur.empty()) parts.push_back(cur);
                cur.clear();
            } else {
                cur += c;
            }
        }
        if (!cur.empty()) parts.push_back(cur);
    }
    std::vector<std::set<std::string>> types(parts.size());
    for (size_t i = 0; i < parts.size(); i++) {
        std::string part = parts[i];
        if (lc_re_version(part)) {
            types[i].insert("version");
        } else if (lc_re_quant_type(part)) {
            types[i].insert("type");
            parts[i] = lc_upper_str(part);
        } else if (i > 0 && lc_re_size(part)) {
            part = lc_replace_all(part, "_", ".");
            if (lc_is_digit(part.back())) {
                part = part.substr(0, part.size() - 2) + "." + part[part.size() - 1] + part[part.size() - 2];
            }
            if (part.size() > 1 && lc_is_digit(part[part.size() - 2]) && lc_in("kmbt", part.back())) {
                part.back() = lc_upper(part.back());
            }
            if (total_params != 0) {
                // float(part[:-1]) * pow(1000, " KMBT".find(part[-1]))
                const std::string num = part.substr(0, part.size() - 1);
                bool ok = !num.empty();
                size_t dots = 0;
                for (char c : num) {
                    if (c == '.') dots++;
                    else if (!lc_is_digit(c)) ok = false;
                }
                ok = ok && dots <= 1 && num != ".";
                if (ok) {
                    const double v = strtod(num.c_str(), nullptr);
                    const char * scale = " KMBT";
                    const int e = lc_in(scale, part.back()) ? (int) (strchr(scale, part.back()) - scale) : -1;
                    const double label_params = v * std::pow(1000.0, (double) e);
                    const bool small = (total_params < 0 && label_params < (double) ((-total_params) / 8)) ||
                                       (total_params > 0 && std::fabs(label_params - (double) total_params) > (double) (7 * total_params / 8));
                    if (small) {
                        types[i].insert("finetune");
                        part.back() = lc_lower(part.back());
                    }
                }
            }
            if (types[i].empty()) {
                types[i].insert("size_label");
            }
            parts[i] = part;
        } else if (i > 0) {
            const std::string l = lc_lower_str(part);
            if (l == "chat" || l == "instruct" || l == "vision" || l == "lora") {
                if (total_params < 0 && l == "lora") {
                    types[i].insert("type");
                } else {
                    types[i].insert("finetune");
                }
            }
        }
    }
    // word-based size labels go when a number-based one exists
    bool numeric_size = false;
    for (size_t i = 0; i < parts.size(); i++) {
        if (types[i].count("size_label")) {
            for (char c : parts[i]) {
                if (lc_is_digit(c)) numeric_size = true;
            }
        }
    }
    if (numeric_size) {
        for (size_t i = 0; i < parts.size(); i++) {
            if (types[i].count("size_label")) {
                bool all_alpha = true;
                for (char c : parts[i]) {
                    if (!lc_is_alpha(c)) all_alpha = false;
                }
                if (all_alpha) types[i].erase("size_label");
            }
        }
    }
    bool at_start = true;
    for (size_t i = 0; i < parts.size(); i++) {
        if (at_start && ((types[i].empty() && lc_is_alpha(parts[i][0])) || types[i].count("version"))) {
            types[i].insert("basename");
        } else {
            at_start = false;
            if (types[i].empty()) {
                types[i].insert("finetune");
            }
        }
    }
    for (size_t k = parts.size(); k-- > 0;) {
        if (types[k].count("basename") && types[k].size() > 1) {
            types[k].erase("basename");
        } else {
            break;
        }
    }
    auto join = [&](const char * t, bool dedup, bool skip_basename) -> std::optional<std::string> {
        std::string s;
        std::vector<std::string> seen;
        for (size_t i = 0; i < parts.size(); i++) {
            if (!types[i].count(t)) continue;
            if (skip_basename && types[i].count("basename")) continue;
            if (dedup) {
                if (std::find(seen.begin(), seen.end(), parts[i]) != seen.end()) continue;
                seen.push_back(parts[i]);
            }
            if (!s.empty()) s += "-";
            s += parts[i];
        }
        if (s.empty()) return std::nullopt;
        return s;
    };
    r.basename   = join("basename", false, false);
    r.size_label = join("size_label", true, false);
    r.finetune   = join("finetune", false, false);
    r.version    = join("version", false, true);
    if (!r.size_label && !r.finetune && !r.version) {
        r.basename.reset();
    }
    return r;
}

// gguf-py utility.model_weight_count_rounded_notation(min_digits=2)
std::string lc_size_label(int64_t n) {
    const double x = (double) n;
    double scaled;
    const char * suffix;
    if (x > 1e12) {
        scaled = x * 1e-12; suffix = "T";
    } else if (x > 1e9) {
        scaled = x * 1e-9; suffix = "B";
    } else if (x > 1e6) {
        scaled = x * 1e-6; suffix = "M";
    } else {
        scaled = x * 1e-3; suffix = "K";
    }
    // len(str(round(scaled)).lstrip('0')): round() is half-to-even
    const double r = std::nearbyint(scaled);
    std::string digits = std::to_string((long long) r);
    size_t z = 0;
    while (z < digits.size() && digits[z] == '0') z++;
    const int len = (int) (digits.size() - z);
    const int fix = std::max(2 - len, 0);
    char buf[64];
    snprintf(buf, sizeof(buf), "%.*f%s", fix, scaled, suffix);
    return buf;
}

// ---------------------------------------------------------------------------------------------
// tensor names (TensorNameMap for MODEL_ARCH.LAYA, laya checkpoint names only)
// ---------------------------------------------------------------------------------------------

const char * const LC_MAP_PLAIN[][2] = {
    { "token_embd",                "token_embd"      },
    { "token_embd_norm",           "token_embd_norm" },
    { "output_norm",               "output_norm"     },
    { "type_emb",                  "type_emb"        },
    { "scorer.0",                  "scorer.0"        },
    { "scorer.1",                  "scorer.1"        },
    { "scorer.3",                  "scorer.3"        },
    { "act_head.0",                "act_head.0"      },
    { "act_head.2",                "act_head.2"      },
    { "embeddings.tok_embeddings", "token_embd"      },
    { "embeddings.norm",           "token_embd_norm" },
    { "final_norm",                "output_norm"     },
};

const char * const LC_MAP_BLOCK[][2] = {
    { "blk.{bid}.attn_norm",                  "blk.{bid}.attn_norm"    },
    { "blk.{bid}.attn_qkv",                   "blk.{bid}.attn_qkv"     },
    { "blk.{bid}.attn_output",                "blk.{bid}.attn_output"  },
    { "blk.{bid}.ffn_up",                     "blk.{bid}.ffn_up"       },
    { "blk.{bid}.ffn_down",                   "blk.{bid}.ffn_down"     },
    { "blk.{bid}.ffn_norm",                   "blk.{bid}.ffn_norm"     },
    { "head.{bid}.attn_qkv",                  "head.{bid}.attn_qkv"    },
    { "head.{bid}.attn_output",               "head.{bid}.attn_output" },
    { "head.{bid}.attn_norm",                 "head.{bid}.attn_norm"   },
    { "head.{bid}.ffn_norm",                  "head.{bid}.ffn_norm"    },
    { "head.{bid}.ffn_up",                    "head.{bid}.ffn_up"      },
    { "head.{bid}.ffn_down",                  "head.{bid}.ffn_down"    },
    { "layers.{bid}.attn_norm",               "blk.{bid}.attn_norm"    },
    { "layers.{bid}.attn.Wqkv",               "blk.{bid}.attn_qkv"     },
    { "layers.{bid}.attn.Wo",                 "blk.{bid}.attn_output"  },
    { "layers.{bid}.mlp.Wi",                  "blk.{bid}.ffn_up"       },
    { "layers.{bid}.mlp.Wo",                  "blk.{bid}.ffn_down"     },
    { "layers.{bid}.mlp_norm",                "blk.{bid}.ffn_norm"     },
    { "head.layers.{bid}.self_attn.in_proj",  "head.{bid}.attn_qkv"    },
    { "head.layers.{bid}.self_attn.out_proj", "head.{bid}.attn_output" },
    { "head.layers.{bid}.norm1",              "head.{bid}.attn_norm"   },
    { "head.layers.{bid}.norm2",              "head.{bid}.ffn_norm"    },
    { "head.layers.{bid}.linear1",            "head.{bid}.ffn_up"      },
    { "head.layers.{bid}.linear2",            "head.{bid}.ffn_down"    },
};

// mapping[key] of TensorNameMap(arch, n_blocks)
bool lc_map_exact(const std::string & key, uint64_t n_blocks, std::string & out) {
    for (const auto & e : LC_MAP_PLAIN) {
        if (key == e[0]) {
            out = e[1];
            return true;
        }
    }
    for (const auto & e : LC_MAP_BLOCK) {
        const std::string tmpl = e[0];
        const size_t p = tmpl.find("{bid}");
        const std::string pre = tmpl.substr(0, p);
        const std::string suf = tmpl.substr(p + 5);
        if (key.size() <= pre.size() + suf.size() || !lc_starts_with(key, pre) || !lc_ends_with(key, suf)) {
            continue;
        }
        const std::string mid = key.substr(pre.size(), key.size() - pre.size() - suf.size());
        if (mid.size() > 12 || lc_span(mid, 0, "0123456789") != mid.size() || (mid.size() > 1 && mid[0] == '0')) {
            continue;
        }
        const uint64_t bid = std::stoull(mid);
        if (bid >= n_blocks) {
            continue;
        }
        out = lc_replace_all(e[1], "{bid}", mid);
        return true;
    }
    return false;
}

// TensorNameMap.get_name(key, try_suffixes=(".weight", ".bias"))
bool lc_map_name(const std::string & key, uint64_t n_blocks, std::string & out) {
    if (lc_map_exact(key, n_blocks, out)) {
        return true;
    }
    for (const char * suffix : { ".weight", ".bias" }) {
        if (lc_ends_with(key, suffix)) {
            std::string base;
            if (lc_map_exact(key.substr(0, key.size() - strlen(suffix)), n_blocks, base)) {
                out = base + suffix;
                return true;
            }
        }
    }
    return false;
}

// ---------------------------------------------------------------------------------------------
// the conversion
// ---------------------------------------------------------------------------------------------

struct lc_tensor {
    std::string          name;       // GGUF name
    std::string          src_name;   // safetensors name
    lc_dtype             src_dtype = LC_DT_F32;
    uint64_t             src_offset = 0;
    size_t               src_file   = 0;
    std::vector<int64_t> ne;         // ggml order
    uint64_t             n_elem = 0;
    ggml_type            type = GGML_TYPE_F32;
    uint64_t             nbytes = 0;
    uint64_t             offset = 0; // in the data section
};

struct lc_ctx {
    std::string dir;
    std::string tok_dir;
    const laya_convert_params * params = nullptr;
    std::vector<std::string> st_files;
    std::vector<lc_tensor>   tensors;
    lc_kvs kvs;

    void log(const std::string & msg) const {
        if (params->log) {
            params->log(msg);
        }
    }
};

const char * LC_ARCH = "laya";

std::string lc_arch_key(const char * k) {
    return std::string(LC_ARCH) + "." + k;
}

// ModelBase.index_tensors + LayaModel.filter_tensors + prepare_tensors (types, names, order)
void lc_collect_tensors(lc_ctx & c, uint64_t n_blocks, laya_convert_outtype outtype) {
    for (const char * f : { "hf_quant_config.json", "config.json" }) {
        if (lc_stat(lc_join(c.dir, f)).exists) {
            lc_fail(std::string(f) + " in the checkpoint root: the Python converter would not use the laya loader for it; not supported");
        }
    }
    // get_model_part_names(dir, "model", ".safetensors") first: without such a file Python takes the
    // pytorch_model*.bin path (not supported here) even when model.safetensors.index.json names shards
    std::vector<std::string> parts;
    for (const std::string & n : lc_list_dir(c.dir)) {
        if (lc_starts_with(n, "model") && lc_ends_with(n, ".safetensors")) {
            parts.push_back(n);
        }
    }
    std::sort(parts.begin(), parts.end());
    if (parts.empty()) {
        lc_fail("no model*.safetensors in '" + c.dir + "' (pytorch_model*.bin is not supported)");
    }
    std::map<std::string, std::string> weight_map;
    bool has_index = false;
    const std::string index_path = lc_join(c.dir, "model.safetensors.index.json");
    if (lc_stat(index_path).is_file) {
        has_index = true;
        const json index = lc_load_json<json>(index_path);
        const json * wm = lc_get(index, "weight_map");
        if (!wm || !wm->is_object()) {
            lc_fail("Can't load 'weight_map' from 'model.safetensors.index.json'");
        }
        std::set<std::string> uniq;
        for (auto it = wm->begin(); it != wm->end(); ++it) {
            const std::string part = lc_str(it.value(), "model.safetensors.index.json: weight_map['" + it.key() + "']");
            if (part.empty() || part == "." || part == ".." || part.find('/') != std::string::npos || part.find('\\') != std::string::npos) {
                lc_fail("model.safetensors.index.json: shard name '" + part + "' is not a plain file name");
            }
            weight_map[it.key()] = part;
            uniq.insert(part);
        }
        parts.assign(uniq.begin(), uniq.end());
        if (parts.empty()) {
            lc_fail("model.safetensors.index.json: the weight_map names no shards");
        }
    }

    struct item { std::string name; lc_st_tensor st; };
    std::vector<item> items;
    std::set<std::string> names_from_parts;
    std::set<std::string> filtered_names;
    for (size_t pi = 0; pi < parts.size(); pi++) {
        c.log("indexing model part '" + parts[pi] + "'");
        const std::string path = lc_join(c.dir, parts[pi]);
        c.st_files.push_back(path);
        std::vector<lc_st_tensor> ts;
        lc_st_read(path, pi, ts);
        std::sort(ts.begin(), ts.end(), [](const lc_st_tensor & a, const lc_st_tensor & b) { return a.name < b.name; });
        for (lc_st_tensor & t : ts) {
            names_from_parts.insert(t.name);
            if (!weight_map.empty()) {
                auto it = weight_map.find(t.name);
                if (it != weight_map.end() && it->second != parts[pi]) {
                    lc_fail("tensor '" + t.name + "' found in '" + parts[pi] + "' but the index assigns it to '" + it->second + "'; refusing to load a wrong-shard copy");
                }
            }
            // LayaModel.filter_tensors
            std::string name = t.name;
            if (name == "temperature") {
                continue;
            }
            if (lc_starts_with(name, "encoder.")) {
                name = name.substr(8);
            }
            if (lc_ends_with(name, "self_attn.in_proj_weight")) {
                name = name.substr(0, name.size() - strlen("in_proj_weight")) + "in_proj.weight";
            } else if (lc_ends_with(name, "self_attn.in_proj_bias")) {
                name = name.substr(0, name.size() - strlen("in_proj_bias")) + "in_proj.bias";
            }
            if (!filtered_names.insert(name).second) {
                lc_fail("duplicate tensor '" + name + "' found in multiple model parts; refusing to silently overwrite");
            }
            items.push_back({ name, std::move(t) });
        }
    }
    if (has_index) {
        std::vector<std::string> missing, extra;
        for (const auto & kv : weight_map) {
            if (!names_from_parts.count(kv.first)) missing.push_back(kv.first);
        }
        for (const std::string & n : names_from_parts) {
            if (!weight_map.count(n)) extra.push_back(n);
        }
        if (!missing.empty() || !extra.empty()) {
            lc_fail("mismatch between weight map and model parts: " + std::to_string(missing.size()) + " missing (first: " +
                    (missing.empty() ? std::string("-") : missing[0]) + "), " + std::to_string(extra.size()) + " extra (first: " +
                    (extra.empty() ? std::string("-") : extra[0]) + ")");
        }
    }

    std::set<std::string> gguf_names;
    for (item & it : items) {
        if (lc_ends_with(it.name, ".attention.masked_bias") || lc_ends_with(it.name, ".attention.bias") || lc_ends_with(it.name, ".rotary_emb.inv_freq")) {
            continue;
        }
        lc_tensor t;
        if (!lc_map_name(it.name, n_blocks, t.name)) {
            lc_fail("Can not map tensor '" + it.name + "' (the C++ converter knows the laya checkpoint names only)");
        }
        if (!gguf_names.insert(t.name).second) {
            lc_fail("Duplicated tensor name '" + t.name + "'");
        }
        t.src_name   = it.st.name;
        t.src_dtype  = it.st.dtype;
        t.src_offset = it.st.offset;
        t.src_file   = it.st.file;
        t.n_elem     = it.st.n_elem;
        t.ne.assign(it.st.shape.rbegin(), it.st.shape.rend());
        const size_t n_dims = t.ne.size();

        // prepare_tensors: 1-D and *_norm.weight -> F32, non-weights -> F32, else the outtype
        const std::string tail = t.name.size() >= 7 ? t.name.substr(t.name.size() - 7) : t.name;
        if (n_dims <= 1 || lc_ends_with(t.name, "_norm.weight") || (tail != ".weight" && tail != ".lora_a" && tail != ".lora_b")) {
            t.type = GGML_TYPE_F32;
        } else if (outtype == LAYA_CONVERT_F32) {
            t.type = GGML_TYPE_F32;
        } else if (outtype == LAYA_CONVERT_F16) {
            t.type = GGML_TYPE_F16;
        } else {
            // gguf.quants.quantize raises QuantError for rows that are not whole Q8_0 blocks
            t.type = (t.ne[0] % 32 == 0) ? GGML_TYPE_Q8_0 : GGML_TYPE_F16;
            if (t.type == GGML_TYPE_F16) {
                c.log("Can't quantize tensor '" + t.name + "' to Q8_0, falling back to F16");
            }
        }
        if (t.type == GGML_TYPE_Q8_0) {
            t.nbytes = t.n_elem / 32 * 34;
        } else {
            t.nbytes = t.n_elem * (t.type == GGML_TYPE_F32 ? 4 : 2);
        }
        c.tensors.push_back(std::move(t));
    }
    if (c.tensors.empty()) {
        lc_fail("no tensors were written: the model shards could not be found");
    }
    uint64_t off = 0;
    for (lc_tensor & t : c.tensors) {
        t.offset = off;
        off += (t.nbytes + 31) / 32 * 32;
    }
}

// Python find_hparam: the first key present (its value may be null); nullptr when none is
const json * lc_find_hparam(const json & hp, std::initializer_list<const char *> keys) {
    for (const char * k : keys) {
        if (const json * v = lc_get(hp, k)) {
            return v;
        }
    }
    return nullptr;
}

const json * lc_find_hparam_nn(const json & hp, std::initializer_list<const char *> keys) {
    const json * v = lc_find_hparam(hp, keys);
    return v && !v->is_null() ? v : nullptr;
}

// ---------------------------------------------------------------------------------------------
// tokenizer (LayaModel.set_vocab, TextModel.get_vocab_base, gguf-py SpecialVocab)
// ---------------------------------------------------------------------------------------------

struct lc_added_token {
    int64_t     id = 0;
    std::string content;
    bool special = false, normalized = false, lstrip = false, rstrip = false, single_word = false;
};

const char * const LC_METASPACE = "\xE2\x96\x81"; // U+2581

// TextModel.does_token_look_special
bool lc_looks_special(const std::string & t) {
    if (t == "<pad>" || t == "<mask>" || t == "<2mass>" || t == "[@BOS@]") return true;
    if (lc_starts_with(t, "<|") && lc_ends_with(t, "|>")) return true;
    if (lc_starts_with(t, "<\xEF\xBD\x9C") && lc_ends_with(t, "\xEF\xBD\x9C>")) return true; // U+FF5C fullwidth bars (deepseek-coder)
    if (lc_starts_with(t, "<unused") && lc_ends_with(t, ">")) return true;
    return false;
}

// GPT-2 bytes_to_unicode as a reverse map: code point -> byte
bool lc_bytelevel_byte(uint32_t cp, uint8_t & b) {
    static const std::vector<int> table = [] {
        std::vector<int> t(324, -1);
        int n = 0;
        for (int v = 0; v < 256; v++) {
            const bool printable = (v >= 0x21 && v <= 0x7E) || (v >= 0xA1 && v <= 0xAC) || (v >= 0xAE && v <= 0xFF);
            if (printable) {
                t[(size_t) v] = v;
            } else {
                t[(size_t) (256 + n)] = v;
                n++;
            }
        }
        return t;
    }();
    if (cp >= table.size() || table[cp] < 0) {
        return false;
    }
    b = (uint8_t) table[cp];
    return true;
}

// decode one UTF-8 code point at s[i]; the input is valid UTF-8 (nlohmann checked it)
uint32_t lc_next_cp(const std::string & s, size_t & i) {
    const unsigned char c = (unsigned char) s[i];
    uint32_t cp;
    size_t len;
    if (c < 0x80) { cp = c; len = 1; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
    else { cp = c & 0x07; len = 4; }
    for (size_t k = 1; k < len && i + k < s.size(); k++) {
        cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
    }
    i += len;
    return cp;
}

bool lc_is_byte_fallback_token(const std::string & t) {
    // <0xHH>
    if (t.size() != 6 || t[0] != '<' || t[1] != '0' || t[2] != 'x' || t[5] != '>') return false;
    return isxdigit((unsigned char) t[3]) && isxdigit((unsigned char) t[4]);
}

// tokenizer.decode(tokenizer.encode(token, add_special_tokens=False)) for a non-normalized added token:
// the decoder chain on [token] (tokenizers 0.23, transformers 5.17 skip clean_up for BPE)
std::string lc_added_roundtrip(const json & decoder, const std::string & token) {
    if (decoder.is_null()) {
        return token;
    }
    std::vector<const json *> chain;
    const json * type = lc_get(decoder, "type");
    if (type && type->is_string() && type->get<std::string>() == "Sequence") {
        const json * ds = lc_get(decoder, "decoders");
        if (!ds || !ds->is_array()) {
            lc_fail("tokenizer.json: decoder Sequence without decoders");
        }
        for (const json & d : *ds) chain.push_back(&d);
    } else {
        chain.push_back(&decoder);
    }
    std::string t = token;
    const std::string what = "added token " + json(token).dump() + " (normalized=false): the AutoTokenizer round trip through decoder ";
    for (const json * d : chain) {
        const json * dt = lc_get(*d, "type");
        const std::string ty = dt && dt->is_string() ? dt->get<std::string>() : std::string("?");
        if (ty == "Replace") {
            const json * pat = lc_get(*d, "pattern");
            const json * con = lc_get(*d, "content");
            const json * ps  = pat ? lc_get(*pat, "String") : nullptr;
            if (!ps || !ps->is_string() || !con || !con->is_string()) {
                lc_fail(what + "Replace with a non-string pattern is not supported");
            }
            t = lc_replace_all(t, ps->get<std::string>(), con->get<std::string>());
        } else if (ty == "ByteFallback") {
            if (lc_is_byte_fallback_token(t)) {
                lc_fail(what + "ByteFallback is not supported");
            }
        } else if (ty == "Fuse") {
            // one token: nothing to fuse
        } else if (ty == "Metaspace") {
            const json * rep = lc_get(*d, "replacement");
            const std::string r = rep && rep->is_string() ? rep->get<std::string>() : std::string(LC_METASPACE);
            if (t.find(r) != std::string::npos) {
                lc_fail(what + "Metaspace is not supported");
            }
        } else if (ty == "ByteLevel") {
            std::string bytes;
            bool all = true;
            for (size_t i = 0; i < t.size();) {
                uint8_t b;
                if (!lc_bytelevel_byte(lc_next_cp(t, i), b)) {
                    all = false;
                    break;
                }
                bytes += (char) b;
            }
            if (all && bytes != t) {
                if (!lc_valid_utf8(bytes)) {
                    lc_fail(what + "ByteLevel gives invalid UTF-8; not supported");
                }
                t = bytes;
            }
        } else {
            lc_fail(what + ty + " is not supported");
        }
    }
    return t;
}

// LayaModel._tokenizer_kind
std::string lc_tokenizer_kind(const json & tj) {
    // references, not copies: model holds the whole vocab and merges
    static const json empty = json::object();
    static const json none;
    const json * jpre = lc_get(tj, "pre_tokenizer");
    const json * jmod = lc_get(tj, "model");
    const json & pre   = jpre && lc_truthy(*jpre) ? *jpre : empty;
    const json & model = jmod && lc_truthy(*jmod) ? *jmod : empty;
    auto mget = [&](const char * k) -> const json & { const json * v = lc_get(model, k); return v ? *v : none; };
    auto pget = [&](const char * k) -> const json & { const json * v = lc_get(pre, k); return v ? *v : none; };
    if (mget("type") != json("BPE") || lc_truthy(mget("dropout")) || lc_truthy(mget("continuing_subword_prefix")) || lc_truthy(mget("end_of_word_suffix"))) {
        lc_fail("laya: unsupported tokenizer model (type " + mget("type").dump() + ")");
    }
    if (pget("type") == json("Metaspace")) {
        return "metaspace-bpe";
    }
    if (pget("type") == json("ByteLevel")) {
        const json * ur = lc_get(pre, "use_regex");
        if (lc_truthy(pget("add_prefix_space")) || (ur && !lc_truthy(*ur))) {
            lc_fail("laya: unsupported ByteLevel pre-tokenizer " + pre.dump());
        }
        if (lc_truthy(mget("byte_fallback")) || lc_truthy(mget("ignore_merges"))) {
            lc_fail("laya: unsupported byte-level BPE options");
        }
        return "bytelevel-bpe";
    }
    lc_fail("laya: unsupported pre-tokenizer " + pre.dump());
}

// ordered dict helpers (Python dict insertion order, update keeps the position)
template <typename T>
void lc_odict_set(std::vector<std::pair<std::string, T>> & d, const std::string & k, T v) {
    for (auto & e : d) {
        if (e.first == k) {
            e.second = v;
            return;
        }
    }
    d.emplace_back(k, v);
}

template <typename T>
const T * lc_odict_get(const std::vector<std::pair<std::string, T>> & d, const std::string & k) {
    for (auto & e : d) {
        if (e.first == k) {
            return &e.second;
        }
    }
    return nullptr;
}

std::string lc_special_content(const json & v) {
    if (v.is_string()) {
        return v.get<std::string>();
    }
    if (v.is_object()) {
        const json * c = lc_get(v, "content");
        if (c && c->is_string()) {
            return c->get<std::string>();
        }
    }
    return std::string();
}

// The tokenizers library (which Python loads tokenizer.json with) deserializes TemplateProcessing
// strictly: single / pair are lists of {"SpecialToken": {id: str, type_id: u32}} or
// {"Sequence": {id: "A" | "B", type_id: u32}}, special_tokens maps to {id: str, ids: [u32], tokens: [str]}.
// Anything else fails in Python, so it is refused here instead of being read with defaults.
// (tokenizers also takes a special token as a 3-element list; that form is refused here.)
void lc_check_template_processing(const json & p, const std::string & where) {
    auto bad = [&](const std::string & what) {
        lc_fail(where + ": post_processor TemplateProcessing: " + what + " (the tokenizers library rejects it)");
    };
    for (const char * k : { "single", "pair" }) {
        const json * v = lc_get(p, k);
        if (!v || !v->is_array()) {
            bad(std::string(k) + " is not a list of pieces");
        }
        for (const json & piece : *v) {
            if (!piece.is_object() || piece.size() != 1) {
                bad(std::string("bad piece in ") + k + ": " + piece.dump());
            }
            const std::string & kind = piece.begin().key();
            const json & inner = piece.begin().value();
            const json * id = lc_get(inner, "id");
            const json * type_id = lc_get(inner, "type_id");
            const bool id_ok = id && id->is_string() &&
                (kind == "SpecialToken" || (kind == "Sequence" && (*id == json("A") || *id == json("B"))));
            if ((kind != "SpecialToken" && kind != "Sequence") || !id_ok || !type_id || !lc_is_serde_u32(*type_id)) {
                bad(std::string("bad piece in ") + k + ": " + piece.dump());
            }
        }
    }
    const json * st = lc_get(p, "special_tokens");
    if (!st || !st->is_object()) {
        bad("special_tokens is not an object");
    }
    for (auto it = st->begin(); it != st->end(); ++it) {
        const json & e = it.value();
        const json * id = lc_get(e, "id");
        const json * ids = lc_get(e, "ids");
        const json * toks = lc_get(e, "tokens");
        bool ok = id && id->is_string() && ids && ids->is_array() && toks && toks->is_array();
        for (size_t i = 0; ok && i < ids->size(); i++) ok = lc_is_serde_u32((*ids)[i]);
        for (size_t i = 0; ok && i < toks->size(); i++) ok = (*toks)[i].is_string();
        if (!ok) {
            bad("bad special_tokens entry " + json(it.key()).dump());
        }
    }
}

void lc_convert_vocab(lc_ctx & c, const json & hp) {
    const std::string tj_path = lc_join(c.tok_dir, "tokenizer.json");
    const std::string tc_path = lc_join(c.tok_dir, "tokenizer_config.json");
    for (const char * f : { "special_tokens_map.json", "added_tokens.json" }) {
        if (lc_stat(lc_join(c.tok_dir, f)).exists) {
            lc_fail(lc_join(c.tok_dir, f) + ": not supported (the Python path would read it through AutoTokenizer)");
        }
    }
    c.log("loading " + tj_path);
    const json tj = lc_load_json<json>(tj_path, LC_MAX_TOKENIZER_BYTES);
    json tcfg = lc_load_json<json>(tc_path);
    if (!tj.is_object()) {
        lc_fail(tj_path + ": not a JSON object");
    }
    if (!tcfg.is_object()) {
        lc_fail(tc_path + ": not a JSON object");
    }
    // Python reads the vocab through AutoTokenizer.from_pretrained(tokenizer dir), which picks the class
    // from tokenizer_config.tokenizer_class, else from tokenizer/config.json (tokenizer_class, model_type).
    // This port is the generic fast tokenizer (transformers 5.17: TokenizersBackend); a model-specific
    // class can rebuild the decoder or re-flag added tokens, so it is refused.
    {
        const json * cls = lc_get_nn(tcfg, "tokenizer_class");
        if (cls) {
            if (!cls->is_string() || (cls->get<std::string>() != "PreTrainedTokenizerFast" && cls->get<std::string>() != "TokenizersBackend")) {
                lc_fail(tc_path + ": tokenizer_class " + cls->dump() + " is not supported (PreTrainedTokenizerFast and TokenizersBackend are)");
            }
        } else {
            const std::string tconf = lc_join(c.tok_dir, "config.json");
            if (lc_stat(tconf).is_file) {
                const json cj = lc_load_json<json>(tconf);
                for (const char * k : { "model_type", "tokenizer_class" }) {
                    if (lc_get_nn(cj, k)) {
                        lc_fail(tconf + ": '" + k + "' without tokenizer_config.tokenizer_class: AutoTokenizer would pick a model-specific tokenizer; not supported");
                    }
                }
            }
        }
        if (lc_get_nn(tcfg, "auto_map")) {
            lc_fail(tc_path + ": auto_map (custom tokenizer code) is not supported");
        }
    }
    const std::string kind = lc_tokenizer_kind(tj);
    const json & model = tj.at("model");

    // model.vocab and added_tokens
    const json * jvocab = lc_get(model, "vocab");
    if (!jvocab || !jvocab->is_object()) {
        lc_fail(tj_path + ": model.vocab is not an object");
    }
    std::unordered_map<std::string, int64_t> vocab;
    std::unordered_map<int64_t, std::string> id_to_text;
    int64_t max_id = -1;
    for (auto it = jvocab->begin(); it != jvocab->end(); ++it) {
        if (!lc_is_int(it.value()) || it.value().is_boolean() || (it.value().is_number_integer() && !it.value().is_number_unsigned() && it.value().get<int64_t>() < 0) ||
            it.value().get<uint64_t>() > (uint64_t) INT32_MAX) {
            lc_fail(tj_path + ": model.vocab[" + json(it.key()).dump() + "] is not a token id");
        }
        const int64_t id = it.value().get<int64_t>();
        vocab[it.key()] = id;
        auto ins = id_to_text.emplace(id, it.key());
        if (!ins.second) {
            lc_fail(tj_path + ": token id " + std::to_string(id) + " is used by two vocab entries");
        }
        max_id = std::max(max_id, id);
    }
    std::vector<lc_added_token> added;
    std::unordered_map<std::string, int64_t> added_by_content;
    std::unordered_map<int64_t, size_t> added_by_id;
    if (const json * jat = lc_get(tj, "added_tokens")) {
        if (!jat->is_array()) {
            lc_fail(tj_path + ": added_tokens is not a list");
        }
        for (const json & a : *jat) {
            const json * id = lc_get(a, "id");
            const json * content = lc_get(a, "content");
            if (!id || !lc_is_int(*id) || id->is_boolean() || !id->is_number_unsigned() || id->get<uint64_t>() > (uint64_t) INT32_MAX || !content || !content->is_string()) {
                lc_fail(tj_path + ": bad added token " + a.dump());
            }
            // Python loads tokenizer.json with the tokenizers library, whose AddedToken has no defaults:
            // a missing or non-bool flag fails there, so it must not read as false here
            for (const char * k : { "single_word", "lstrip", "rstrip", "normalized", "special" }) {
                const json * v = lc_get(a, k);
                if (!v || !v->is_boolean()) {
                    lc_fail(tj_path + ": added token " + json(content->get<std::string>()).dump() + ": '" + k + "' is missing or not a bool (the tokenizers library rejects it)");
                }
            }
            lc_added_token t;
            t.id = id->get<int64_t>();
            t.content = content->get<std::string>();
            auto flag = [&](const char * k) { const json * v = lc_get(a, k); return v && lc_truthy(*v); };
            t.special = flag("special");
            t.normalized = flag("normalized");
            t.lstrip = flag("lstrip");
            t.rstrip = flag("rstrip");
            t.single_word = flag("single_word");
            if (added_by_id.count(t.id)) {
                lc_fail(tj_path + ": two added tokens have id " + std::to_string(t.id));
            }
            if (added_by_content.count(t.content)) {
                lc_fail(tj_path + ": two added tokens have the content " + json(t.content).dump());
            }
            // get_vocab(with_added_tokens=True): an added token overrides model.vocab by content; one id with two
            // strings makes the Python result depend on hash order
            auto vit = id_to_text.find(t.id);
            if (vit != id_to_text.end() && vit->second != t.content) {
                lc_fail(tj_path + ": added token " + json(t.content).dump() + " reuses id " + std::to_string(t.id) + " of vocab entry " + json(vit->second).dump());
            }
            auto cit = vocab.find(t.content);
            if (cit != vocab.end() && cit->second != t.id) {
                lc_fail(tj_path + ": added token " + json(t.content).dump() + " has id " + std::to_string(t.id) + " but model.vocab says " + std::to_string(cit->second));
            }
            added_by_id[t.id] = added.size();
            added_by_content[t.content] = t.id;
            max_id = std::max(max_id, t.id);
            added.push_back(t);
        }
    }

    // tokenizer_config tokens that AutoTokenizer would add or re-flag
    if (const json * atd = lc_get(tcfg, "added_tokens_decoder")) {
        if (!atd->is_object()) {
            lc_fail(tc_path + ": added_tokens_decoder is not an object");
        }
        for (auto it = atd->begin(); it != atd->end(); ++it) {
            char * end = nullptr;
            const long long id = strtoll(it.key().c_str(), &end, 10);
            auto ai = added_by_id.find((int64_t) id);
            const json & v = it.value();
            auto flag = [&](const char * k) { const json * x = lc_get(v, k); return x && lc_truthy(*x); };
            if (end == it.key().c_str() || *end != '\0' || ai == added_by_id.end() || !v.is_object() ||
                lc_special_content(v) != added[ai->second].content || flag("special") != added[ai->second].special ||
                flag("normalized") != added[ai->second].normalized || flag("lstrip") != added[ai->second].lstrip ||
                flag("rstrip") != added[ai->second].rstrip || flag("single_word") != added[ai->second].single_word) {
                lc_fail(tc_path + ": added_tokens_decoder[" + it.key() + "] differs from tokenizer.json; not supported");
            }
        }
    }
    {
        std::vector<std::string> specials;
        for (const char * k : { "bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token" }) {
            if (const json * v = lc_get_nn(tcfg, k)) {
                const std::string s = lc_special_content(*v);
                if (!s.empty()) specials.push_back(s);
            }
        }
        if (const json * v = lc_get_nn(tcfg, "additional_special_tokens")) {
            if (v->is_array()) {
                for (const json & e : *v) specials.push_back(lc_special_content(e));
            }
        }
        if (const json * v = lc_get_nn(tcfg, "extra_special_tokens")) {
            if (v->is_object() || v->is_array()) {
                for (const json & e : *v) specials.push_back(lc_special_content(e));
            }
        }
        for (const std::string & s : specials) {
            if (!s.empty() && !added_by_content.count(s)) {
                lc_fail(tc_path + ": special token " + json(s).dump() + " is not an added token of tokenizer.json; AutoTokenizer would add it; not supported");
            }
        }
        // transformers 5.17 marks the named special tokens and the values of a dict extra_special_tokens as special
        // (other flags unchanged); additional_special_tokens and a list extra_special_tokens change nothing
        std::vector<std::string> reflag;
        for (const char * k : { "bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token" }) {
            if (const json * v = lc_get_nn(tcfg, k)) {
                reflag.push_back(lc_special_content(*v));
            }
        }
        if (const json * v = lc_get_nn(tcfg, "extra_special_tokens")) {
            if (v->is_object()) {
                for (const json & e : *v) reflag.push_back(lc_special_content(e));
            }
        }
        for (const std::string & s : reflag) {
            auto it = added_by_content.find(s);
            if (!s.empty() && it != added_by_content.end()) {
                added[added_by_id.at(it->second)].special = true;
            }
        }
        if (const json * v = lc_get(tcfg, "clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output")) {
            if (lc_truthy(*v)) {
                lc_fail(tc_path + ": forced clean_up_tokenization for BPE is not supported");
            }
        }
    }

    // LayaModel.set_vocab: [CLS] / [SEP] / [MASK] ids
    auto token_id = [&](const std::string & content) -> int64_t {
        auto a = added_by_content.find(content);
        if (a != added_by_content.end()) return a->second;
        auto v = vocab.find(content);
        if (v != vocab.end()) return v->second;
        lc_fail("laya: token " + json(content).dump() + " is not in the vocabulary");
    };
    auto special = [&](const char * name) -> int64_t {
        const json * tok = lc_get(tcfg, name);
        std::string s;
        if (tok && tok->is_string()) {
            s = tok->get<std::string>();
        } else if (tok && tok->is_object() && lc_get(*tok, "content") && lc_get(*tok, "content")->is_string()) {
            s = lc_get(*tok, "content")->get<std::string>();
        } else {
            lc_fail(std::string("laya: tokenizer_config.json has no ") + name);
        }
        return token_id(s);
    };
    const int64_t cls_id  = special("cls_token");
    const int64_t sep_id  = special("sep_token");
    const int64_t mask_id = special("mask_token");

    c.kvs.u32(lc_arch_key("marker_token_id"), (uint32_t) mask_id);

    // TextModel.get_vocab_base
    int64_t vocab_size;
    if (const json * vs = lc_get(hp, "vocab_size")) {
        vocab_size = (int64_t) lc_u32(*vs, "vocab_size");
    } else {
        std::set<std::string> all;
        for (const auto & kv : vocab) all.insert(kv.first);
        for (const auto & a : added) all.insert(a.content);
        vocab_size = (int64_t) all.size();
    }
    if (max_id >= vocab_size) {
        lc_fail("the tokenizer has token id " + std::to_string(max_id) + " but vocab_size is " + std::to_string(vocab_size));
    }
    // vocab_size comes from an untrusted config and every id without a token becomes a "[PAD<i>]"
    // string: bound it before the loop below allocates it. The padding exists to fill the rows of
    // the token embedding, and llama.cpp needs n_vocab == token_embd rows to load the file, so a
    // larger value never makes a usable GGUF (Python would write it, or run out of memory).
    if (vocab_size > LC_MAX_VOCAB) {
        lc_fail("vocab_size " + std::to_string(vocab_size) + " is absurd (limit " + std::to_string(LC_MAX_VOCAB) + ")");
    }
    for (const lc_tensor & t : c.tensors) {
        if (t.name == "token_embd.weight" && t.ne.size() == 2 && vocab_size > t.ne[1]) {
            lc_fail("vocab_size " + std::to_string(vocab_size) + " is larger than the " + std::to_string(t.ne[1]) +
                    " rows of token_embd.weight: the padded vocab would not load; not supported");
        }
    }
    static const json no_decoder;
    const json * jdec = lc_get(tj, "decoder");
    const json & decoder = jdec ? *jdec : no_decoder;
    std::vector<std::string> tokens;
    std::vector<int32_t> toktypes;
    tokens.reserve((size_t) vocab_size);
    toktypes.reserve((size_t) vocab_size);
    for (int64_t i = 0; i < vocab_size; i++) {
        auto ai = added_by_id.find(i);
        std::string token;
        if (ai != added_by_id.end()) {
            token = added[ai->second].content;
        } else {
            auto vi = id_to_text.find(i);
            if (vi == id_to_text.end()) {
                tokens.push_back("[PAD" + std::to_string(i) + "]");
                toktypes.push_back(5); // UNUSED
                continue;
            }
            token = vi->second;
        }
        if (added_by_content.count(token)) {
            const lc_added_token & a = added[added_by_id.at(i)];
            if (!a.normalized) {
                token = lc_added_roundtrip(decoder, token);
            }
            if (a.special || lc_looks_special(token)) {
                toktypes.push_back(3); // CONTROL
            } else {
                token = lc_replace_all(token, LC_METASPACE, " ");
                toktypes.push_back(4); // USER_DEFINED
            }
        } else {
            toktypes.push_back(1); // NORMAL
        }
        tokens.push_back(token);
    }

    // _set_vocab_gpt2
    c.kvs.str("tokenizer.ggml.model", "gpt2");
    c.kvs.str("tokenizer.ggml.pre", "modern-bert");
    c.kvs.arr_str("tokenizer.ggml.tokens", std::move(tokens));
    c.kvs.arr_i32("tokenizer.ggml.token_type", std::move(toktypes));

    // SpecialVocab(tok_dir, load_merges=True)
    std::vector<std::string> merges;
    std::vector<std::pair<std::string, int64_t>> ids;
    std::vector<std::pair<std::string, bool>> add;
    std::optional<std::string> chat_template;
    std::optional<bool> norm_lower, norm_strip;
    std::vector<std::string> special_types = { "bos", "eos", "unk", "sep", "pad", "cls", "mask" };

    if (const json * jm = lc_get(model, "merges")) {
        if (jm->is_array() && !jm->empty()) {
            if ((*jm)[0].is_string()) {
                for (const json & m : *jm) {
                    merges.push_back(lc_str(m, tj_path + ": merges"));
                }
            } else if ((*jm)[0].is_array() && (*jm)[0].size() == 2 && (*jm)[0][0].is_string()) {
                for (const json & m : *jm) {
                    if (!m.is_array() || m.size() != 2 || !m[0].is_string() || !m[1].is_string()) {
                        lc_fail(tj_path + ": bad merge " + m.dump());
                    }
                    // spaces inside a pair are written as U+0120
                    std::string s = lc_replace_all(m[0].get<std::string>(), " ", "\xC4\xA0") + " " + lc_replace_all(m[1].get<std::string>(), " ", "\xC4\xA0");
                    merges.push_back(s);
                }
            } else {
                lc_fail("Unknown tokenizer merges format");
            }
        }
    }
    // _parse_normalizer: depth first, in order, with an explicit stack (no recursion on input nesting)
    auto parse_normalizer = [&](const json & root) {
        std::vector<const json *> stack = { &root };
        while (!stack.empty()) {
            const json & n = *stack.back();
            stack.pop_back();
            const json * t = lc_get(n, "type");
            const std::string ty = t && t->is_string() ? t->get<std::string>() : std::string();
            if (ty == "Lowercase") {
                norm_lower = true;
            } else if (ty == "StripAccents") {
                norm_strip = true;
            } else if (ty == "BertNormalizer") {
                if (const json * v = lc_get(n, "lowercase")) {
                    if (v->is_null()) norm_lower.reset();
                    else if (v->is_boolean()) norm_lower = v->get<bool>();
                    else lc_fail(tj_path + ": BertNormalizer.lowercase is not a bool");
                }
                if (const json * v = lc_get(n, "strip_accents")) {
                    if (v->is_null()) norm_strip.reset();
                    else if (v->is_boolean()) norm_strip = v->get<bool>();
                    else lc_fail(tj_path + ": BertNormalizer.strip_accents is not a bool");
                }
            } else if (ty == "Sequence") {
                if (const json * ns = lc_get(n, "normalizers")) {
                    if (ns->is_array()) {
                        for (size_t k = ns->size(); k-- > 0;) {
                            stack.push_back(&(*ns)[k]); // reversed: the first child is processed next
                        }
                    }
                }
            }
        }
    };
    if (const json * n = lc_get(tj, "normalizer")) {
        if (lc_truthy(*n)) {
            if (!n->is_object()) {
                lc_fail(tj_path + ": normalizer is not an object");
            }
            parse_normalizer(*n);
        }
    }
    auto tget = [&](const char * k) { const json * v = lc_get(tcfg, k); return v ? *v : json(); };
    json special_bos = tget("bos_token");
    json special_cls = tget("cls_token");
    json special_eos = tget("eos_token");
    json special_sep = tget("sep_token");
    const bool tcfg_truthy = !tcfg.empty();
    if (!lc_truthy(special_bos) && lc_truthy(special_cls) && tcfg_truthy) {
        tcfg["bos_token"] = special_bos = special_cls;
    }
    if (!lc_truthy(special_eos) && lc_truthy(special_sep) && tcfg_truthy) {
        tcfg["eos_token"] = special_eos = special_sep;
    }
    auto sp_id = [](const json & tmpl_entry) -> json {
        const json * st = lc_get(tmpl_entry, "SpecialToken");
        if (!st) return json();
        const json * id = lc_get(*st, "id");
        return id ? *id : json();
    };
    auto seq_id = [](const json & tmpl_entry) -> json {
        const json * st = lc_get(tmpl_entry, "Sequence");
        if (!st) return json();
        const json * id = lc_get(*st, "id");
        return id ? *id : json();
    };
    auto in2 = [](const json & x, const json & a, const json & b) { return x == a || x == b; };
    if (const json * pp = lc_get(tj, "post_processor")) {
        if (lc_truthy(*pp)) {
            if (!pp->is_object()) {
                lc_fail(tj_path + ": post_processor is not an object");
            }
            std::vector<const json *> procs;
            if (const json * ps = lc_get(*pp, "processors")) {
                if (!ps->is_array()) lc_fail(tj_path + ": post_processor.processors is not a list");
                for (const json & p : *ps) procs.push_back(&p);
            } else {
                procs.push_back(pp);
            }
            for (const json * procp : procs) {
                const json & proc = *procp;
                const json * pt = lc_get(proc, "type");
                const std::string ptype = pt && pt->is_string() ? pt->get<std::string>() : std::string();
                if (ptype == "RobertaProcessing") {
                    lc_odict_set(add, "bos", true);
                    lc_odict_set(add, "eos", true);
                    lc_odict_set(add, "sep", true);
                    auto first_of = [&](const char * k, const json & dflt) -> json {
                        const json * v = lc_get(proc, k);
                        if (!v) return dflt;
                        if (!v->is_array() || v->empty()) lc_fail(tj_path + ": RobertaProcessing." + k + " is not a list");
                        return (*v)[0];
                    };
                    if (!lc_truthy(special_cls) && tcfg_truthy) {
                        special_cls = first_of("cls", special_bos);
                        tcfg["cls_token"] = special_cls;
                    }
                    if (!lc_truthy(special_sep) && tcfg_truthy) {
                        special_sep = first_of("sep", special_eos);
                        tcfg["sep_token"] = special_sep;
                    }
                    continue;
                }
                if (ptype != "TemplateProcessing") {
                    continue;
                }
                lc_check_template_processing(proc, tj_path);
                const json * js = lc_get(proc, "single");
                const json * jp = lc_get(proc, "pair");
                const json single = js && js->is_array() ? *js : json::array();
                json pair         = jp && jp->is_array() ? *jp : json::array();
                json special_first, special_last;
                if (single.size() > 1) {
                    special_first = sp_id(single[0]);
                    if (lc_truthy(special_first)) {
                        if (!tcfg_truthy) {
                            special_bos = special_first;
                        } else if (!in2(special_first, special_bos, special_cls)) {
                            if (!lc_truthy(special_bos)) {
                                tcfg["bos_token"] = special_bos = special_first;
                            }
                            if (!lc_truthy(special_cls)) {
                                tcfg["cls_token"] = special_cls = special_first;
                            }
                        }
                        lc_odict_set(add, "bos", in2(special_first, special_bos, special_cls));
                    }
                    special_last = sp_id(single[single.size() - 1]);
                    if (lc_truthy(special_last)) {
                        if (!tcfg_truthy) {
                            special_eos = special_last;
                        } else if (special_last != special_eos) {
                            if (std::find(special_types.begin(), special_types.end(), "eot") == special_types.end()) {
                                special_types.push_back("eot");
                                tcfg["eot_token"] = special_eos;
                            } else if (std::find(special_types.begin(), special_types.end(), "eom") == special_types.end()) {
                                special_types.push_back("eom");
                                tcfg["eom_token"] = special_eos;
                            }
                            tcfg["eos_token"] = special_eos = special_last;
                        }
                        lc_odict_set(add, "eos", special_last == special_eos);
                    }
                }
                if (!pair.empty()) {
                    const size_t seq_start = (lc_truthy(special_first) && sp_id(pair[0]) == special_first) ? 1 : 0;
                    const bool   stop_last = lc_truthy(special_last) && sp_id(pair[pair.size() - 1]) == special_last;
                    json sub = json::array();
                    const size_t end = stop_last ? pair.size() - 1 : pair.size();
                    for (size_t k = seq_start; k < end; k++) sub.push_back(pair[k]);
                    if (!sub.empty()) {
                        const json tmpl_a = seq_id(sub[0]);
                        const json tmpl_b = seq_id(sub[sub.size() - 1]);
                        if (tmpl_a == json("A") && tmpl_b == json("B") && sub.size() > 2) {
                            json mid = json::array();
                            for (size_t k = 1; k + 1 < sub.size(); k++) mid.push_back(sub[k]);
                            bool add_sep = false;
                            const json e0 = sp_id(mid[0]);
                            if (lc_truthy(e0)) {
                                if (in2(e0, special_sep, special_eos) && !lc_truthy(special_last)) {
                                    add_sep = true;
                                }
                            }
                            if (mid.size() == 2) {
                                const json e1 = sp_id(mid[1]);
                                if (lc_truthy(e1) && in2(e1, special_sep, special_eos)) {
                                    add_sep = true;
                                }
                            }
                            lc_odict_set(add, "sep", add_sep);
                            if (add_sep && !lc_truthy(special_sep) && tcfg_truthy) {
                                tcfg["sep_token"] = special_eos;
                            }
                        }
                    }
                }
            }
        }
    }
    auto set_special = [&](const std::string & typ, const json & tid, const std::string & what) {
        if (!lc_is_int(tid) || tid.is_boolean()) {
            if (tid.is_boolean()) lc_fail(what + ": bool token id");
            return;
        }
        if (tid.is_number_integer() && !tid.is_number_unsigned() && tid.get<int64_t>() < 0) {
            lc_fail("invalid value for special token type " + typ + ": " + tid.dump());
        }
        if (lc_odict_get(ids, typ)) {
            return;
        }
        if (tid.get<uint64_t>() > UINT32_MAX) {
            lc_fail(what + ": token id " + tid.dump() + " does not fit u32");
        }
        lc_odict_set(ids, typ, (int64_t) tid.get<uint64_t>());
    };
    if (tcfg_truthy) {
        // chat template: tokenizer_config.chat_template, else chat_template.jinja / .json
        json alt;
        const std::string jinja = lc_join(c.tok_dir, "chat_template.jinja");
        const std::string jctj  = lc_join(c.tok_dir, "chat_template.json");
        if (lc_stat(jinja).is_file) {
            if (lc_stat(lc_join(c.tok_dir, "additional_chat_templates")).is_dir) {
                lc_fail("additional_chat_templates/: list-form chat templates are not supported");
            }
            std::string t = lc_read_file(jinja, LC_MAX_CONFIG_BYTES);
            if (!lc_valid_utf8(t)) lc_fail(jinja + ": not valid UTF-8");
            t = lc_replace_all(lc_replace_all(t, "\r\n", "\n"), "\r", "\n");
            alt = t;
        } else if (lc_stat(jctj).is_file) {
            const json j = lc_load_json<json>(jctj);
            if (!j.is_object()) lc_fail(jctj + ": not a JSON object");
            const json * ct = lc_get(j, "chat_template");
            alt = ct ? *ct : json();
        }
        const json * ct = lc_get(tcfg, "chat_template");
        const json tmpl = ct ? *ct : alt;
        if (tmpl.is_string()) {
            chat_template = tmpl.get<std::string>();
        } else if (tmpl.is_array()) {
            lc_fail("list-form chat templates are not supported (their GGUF key order depends on Python set order)");
        }
        const json * jat = lc_get(tj, "added_tokens");
        for (const std::string & typ : special_types) {
            if (const json * ae = lc_get(tcfg, "add_" + typ + "_token")) {
                if (ae->is_boolean()) {
                    lc_odict_set(add, typ, ae->get<bool>());
                }
            }
            const json * entry = lc_get(tcfg, typ + "_token");
            if (!entry) continue;
            std::string content;
            if (entry->is_string()) {
                content = entry->get<std::string>();
            } else if (entry->is_object()) {
                const json * ec = lc_get(*entry, "content");
                if (!ec || !ec->is_string()) continue;
                content = ec->get<std::string>();
            } else {
                continue;
            }
            json found;
            if (jat) {
                for (const json & a : *jat) {
                    const json * ac = lc_get(a, "content");
                    if (ac && ac->is_string() && ac->get<std::string>() == content) {
                        const json * aid = lc_get(a, "id");
                        found = aid ? *aid : json();
                        break;
                    }
                }
            }
            set_special(typ, found, tc_path + ": " + typ + "_token");
        }
    }
    // _try_load_from_config_json
    const std::string tconf = lc_join(c.tok_dir, "config.json");
    if (lc_stat(tconf).is_file) {
        const json cj = lc_load_json<json>(tconf);
        if (!cj.is_object()) lc_fail(tconf + ": not a JSON object");
        for (const std::string & typ : special_types) {
            const json * v = lc_get(cj, typ + "_token_id");
            json tid = v ? *v : json();
            if (tid.is_null()) {
                if (const json * tc = lc_get(cj, "text_config")) {
                    const json * v2 = lc_get(*tc, typ + "_token_id");
                    tid = v2 ? *v2 : json();
                }
            }
            set_special(typ, tid, tconf + ": " + typ + "_token_id");
        }
    }
    if (merges.empty() && lc_stat(lc_join(c.tok_dir, "merges.txt")).is_file) {
        lc_fail("merges.txt: not supported (tokenizer.json has no merges)");
    }

    // LayaModel: add bos/eos/sep, bos = [CLS], eos = sep = [SEP], mask, unk, pad
    lc_odict_set(add, "bos", true);
    lc_odict_set(add, "eos", true);
    lc_odict_set(add, "sep", true);
    ids.erase(std::remove_if(ids.begin(), ids.end(), [](const std::pair<std::string, int64_t> & e) { return e.first == "cls"; }), ids.end());
    lc_odict_set(ids, "bos", cls_id);
    lc_odict_set(ids, "eos", sep_id);
    lc_odict_set(ids, "sep", sep_id);
    lc_odict_set(ids, "mask", mask_id);
    if (lc_get(tcfg, "unk_token")) {
        lc_odict_set(ids, "unk", special("unk_token"));
    }
    if (lc_get(tcfg, "pad_token")) {
        lc_odict_set(ids, "pad", special("pad_token"));
    }

    // SpecialVocab.add_to_gguf
    c.kvs.arr_str("tokenizer.ggml.merges", std::move(merges));
    static const std::map<std::string, std::string> id_keys = {
        { "bos", "tokenizer.ggml.bos_token_id" }, { "eos", "tokenizer.ggml.eos_token_id" }, { "unk", "tokenizer.ggml.unknown_token_id" },
        { "sep", "tokenizer.ggml.seperator_token_id" }, { "pad", "tokenizer.ggml.padding_token_id" }, { "mask", "tokenizer.ggml.mask_token_id" },
        { "eot", "tokenizer.ggml.eot_token_id" }, { "eom", "tokenizer.ggml.eom_token_id" },
    };
    for (const auto & e : ids) {
        auto k = id_keys.find(e.first);
        if (k != id_keys.end()) {
            c.kvs.u32(k->second, (uint32_t) e.second);
        }
    }
    static const std::map<std::string, std::string> add_keys = {
        { "bos", "tokenizer.ggml.add_bos_token" }, { "eos", "tokenizer.ggml.add_eos_token" }, { "sep", "tokenizer.ggml.add_sep_token" },
    };
    for (const auto & e : add) {
        auto k = add_keys.find(e.first);
        if (k != add_keys.end()) {
            c.kvs.boolean(k->second, e.second);
        }
    }
    if (chat_template) {
        c.kvs.str("tokenizer.chat_template", *chat_template);
    }
    if (norm_lower) {
        c.kvs.boolean("tokenizer.ggml.normalizer.lowercase", *norm_lower);
    }
    if (norm_strip) {
        c.kvs.boolean("tokenizer.ggml.normalizer.strip_accents", *norm_strip);
    }

    // decision.laya.* for tools/laya
    c.kvs.str("decision.laya.tokenizer", kind);
    if (kind == "bytelevel-bpe") {
        const json * norm = lc_get(tj, "normalizer");
        if (!norm || norm->is_null()) {
            c.kvs.str("decision.laya.normalizer", "none");
        } else if (norm->is_object() && lc_get(*norm, "type") && *lc_get(*norm, "type") == json("NFC")) {
            c.kvs.str("decision.laya.normalizer", "nfc");
        } else {
            lc_fail("laya: unsupported normalizer " + norm->dump());
        }
        std::vector<int32_t> flags;
        for (const lc_added_token & a : added) {
            if (a.rstrip || a.single_word) {
                lc_fail("laya: added token " + json(a.content).dump() + " uses rstrip / single_word");
            }
            flags.push_back((int32_t) a.id);
            flags.push_back((a.lstrip ? 1 : 0) | (a.normalized ? 2 : 0));
        }
        c.kvs.arr_i32("decision.laya.added_tokens", std::move(flags));
    }
    c.log("laya: tokenizer " + kind + ", [CLS] " + std::to_string(cls_id) + ", [SEP] " + std::to_string(sep_id) + ", [MASK] " + std::to_string(mask_id));
}

// ---------------------------------------------------------------------------------------------
// hparams (conversion/laya.py _load_laya_hparams) and set_gguf_parameters of the class chain
// ---------------------------------------------------------------------------------------------

struct lc_hparams {
    json         hp;        // encoder config + laya keys
    ordered_json rl;        // rl_agent_config.json, key order kept (temperature_by_options)
    uint64_t     block_count = 0;
};

lc_hparams lc_load_hparams(const lc_ctx & c) {
    lc_hparams h;
    const std::string rl_path = lc_join(c.dir, "rl_agent_config.json");
    if (!lc_stat(rl_path).is_file) {
        lc_fail("no rl_agent_config.json in '" + c.dir + "': not a laya checkpoint");
    }
    h.rl = lc_load_json<ordered_json>(rl_path);
    if (!h.rl.is_object()) {
        lc_fail(rl_path + ": not a JSON object");
    }
    const std::string enc_path = lc_join(c.dir, "encoder/config.json");
    h.hp = json::object();
    if (lc_stat(enc_path).is_file) {
        h.hp = lc_load_json<json>(enc_path);
        if (!h.hp.is_object()) {
            lc_fail(enc_path + ": not a JSON object");
        }
    }
    h.hp["architectures"] = json::array({ "LayaModel" });
    h.hp["model_type"]    = "laya";
    for (const char * k : { "text_config", "llm_config", "lm_config", "thinker_config", "language_config", "lfm" }) {
        if (lc_get(h.hp, k)) {
            lc_fail(std::string("encoder/config.json: '") + k + "' is not supported");
        }
    }
    if (const json * v = lc_get(h.hp, "id2label")) {
        if (lc_truthy(*v)) {
            lc_fail("encoder/config.json: id2label (classifier labels) is not supported");
        }
    }
    if (const json * v = lc_get(h.hp, "quantization_config")) {
        if (lc_truthy(*v)) {
            lc_fail("encoder/config.json: quantization_config is not supported");
        }
    }
    const json * bc = lc_find_hparam(h.hp, { "n_layers", "num_hidden_layers", "n_layer", "num_layers" });
    if (!bc) {
        lc_fail("could not find any of: n_layers, num_hidden_layers, n_layer, num_layers");
    }
    h.block_count = lc_u32(*bc, "block_count");
    // Python builds TensorNameMap over range(block_count): 15 s at 65536 blocks, and it never
    // finishes (or runs out of memory) for values like 2^31. Such a config is not a model; refuse it
    // instead of writing a file Python cannot produce.
    if (h.block_count > LC_MAX_BLOCKS) {
        lc_fail("encoder/config.json: block_count " + std::to_string(h.block_count) + " is absurd (limit " + std::to_string(LC_MAX_BLOCKS) + ")");
    }
    return h;
}

void lc_set_gguf_parameters(lc_ctx & c, lc_hparams & h, laya_convert_outtype outtype) {
    const json & hp = h.hp;
    lc_kvs & kv = c.kvs;

    // TextModel.__init__: rope_parameters
    json rp = json::object();
    {
        const json * v = lc_get(hp, "rope_parameters");
        if (!v) v = lc_get(hp, "rope_scaling");
        if (v && lc_truthy(*v)) {
            if (!v->is_object()) lc_fail("rope_parameters is not an object");
            rp = *v;
        }
    }
    if (!lc_get(rp, "full_attention") && !lc_get(rp, "sliding_attention")) {
        const json * rope_theta = lc_find_hparam_nn(hp, { "global_rope_theta", "rope_global_theta", "rope_theta_global", "rope_theta", "rotary_emb_base" });
        const json * local      = lc_find_hparam_nn(hp, { "local_rope_theta", "rope_local_theta", "rope_theta_local", "swa_rope_theta", "rope_local_base_freq" });
        if (local) {
            rp["sliding_attention"] = json::object({ { "rope_theta", *local } });
        }
        if (!lc_get(rp, "rope_theta") && rope_theta) {
            rp["rope_theta"] = *rope_theta;
        }
        if (!lc_get(rp, "rope_type")) {
            if (const json * t = lc_get_nn(rp, "type")) {
                rp["rope_type"] = *t;
            }
        }
    }

    // TextModel.set_gguf_parameters
    kv.u32(lc_arch_key("block_count"), (uint32_t) h.block_count);
    if (const json * v = lc_find_hparam_nn(hp, { "max_position_embeddings", "n_ctx", "n_positions", "max_length", "max_sequence_length", "model_max_length" })) {
        kv.u32(lc_arch_key("context_length"), lc_u32(*v, "context length"));
    }
    if (const json * v = lc_find_hparam_nn(hp, { "hidden_size", "n_embd", "dim" })) {
        kv.u32(lc_arch_key("embedding_length"), lc_u32(*v, "embedding length"));
    }
    if (const json * v = lc_find_hparam_nn(hp, { "prefix_dense_intermediate_size", "dense_intermediate_size", "intermediate_size", "n_inner", "hidden_dim" })) {
        kv.u32(lc_arch_key("feed_forward_length"), lc_u32(*v, "feed forward length"));
    }
    if (const json * v = lc_find_hparam_nn(hp, { "num_attention_heads", "n_head", "n_heads" })) {
        kv.u32(lc_arch_key("attention.head_count"), lc_u32(*v, "head count"));
    }
    if (const json * v = lc_find_hparam_nn(hp, { "num_key_value_heads", "n_kv_heads" })) {
        kv.u32(lc_arch_key("attention.head_count_kv"), lc_u32(*v, "head count kv"));
    }
    if (const json * v = lc_get(hp, "is_causal")) {
        if (v->is_boolean() && !v->get<bool>()) {
            kv.boolean(lc_arch_key("attention.causal"), false);
        }
    }
    const json * rpp = lc_get(rp, "full_attention");
    const json rope_params = rpp ? *rpp : rp;
    if (!rope_params.is_object()) {
        lc_fail("rope_parameters.full_attention is not an object");
    }
    if (const json * rt = lc_get_nn(rope_params, "rope_type")) {
        if (!rt->is_string()) {
            lc_fail("rope_type is not a string");
        }
        const std::string t = rt->get<std::string>();
        const bool factor = lc_get_nn(rope_params, "factor") != nullptr;
        if (((t == "linear" || t == "yarn") && factor) || t == "su" || t == "longrope") {
            lc_fail("rope scaling '" + t + "' is not supported");
        }
        if (t != "dynamic" && lc_lower_str(t) != "llama3") {
            c.log("Unknown RoPE type: " + t);
        }
    }
    if (lc_get(rp, "mrope_section")) {
        lc_fail("mrope_section is not supported");
    }
    if (const json * v = lc_get_nn(rope_params, "rope_theta")) {
        kv.f32(lc_arch_key("rope.freq_base"), lc_f32(*v, "rope_theta"));
    }
    if (const json * sa = lc_get(rp, "sliding_attention")) {
        if (!sa->is_object()) {
            lc_fail("rope_parameters.sliding_attention is not an object");
        }
        if (const json * v = lc_get_nn(*sa, "rope_theta")) {
            kv.f32(lc_arch_key("rope.freq_base_swa"), lc_f32(*v, "sliding_attention.rope_theta"));
        }
    }
    if (const json * v = lc_find_hparam_nn(hp, { "rms_norm_eps", "norm_eps" })) {
        kv.f32(lc_arch_key("attention.layer_norm_rms_epsilon"), lc_f32(*v, "rms norm epsilon"));
    }
    if (const json * v = lc_find_hparam_nn(hp, { "layer_norm_eps", "layer_norm_epsilon", "norm_epsilon" })) {
        kv.f32(lc_arch_key("attention.layer_norm_epsilon"), lc_f32(*v, "layer norm epsilon"));
    }
    if (lc_find_hparam_nn(hp, { "num_local_experts", "num_experts", "n_routed_experts" }) ||
        lc_find_hparam_nn(hp, { "num_experts_per_tok", "num_experts_per_token", "top_k_experts" }) ||
        lc_get_nn(hp, "n_group") || lc_get_nn(hp, "topk_group") ||
        lc_find_hparam_nn(hp, { "score_function", "scoring_func", "score_func", "moe_router_activation", "moe_router_activation_func", "expert_selection_fn" })) {
        lc_fail("mixture-of-experts settings in encoder/config.json are not supported");
    }
    if (const json * v = lc_get_nn(hp, "head_dim")) {
        kv.u32(lc_arch_key("attention.key_length"), lc_u32(*v, "head_dim"));
        kv.u32(lc_arch_key("attention.value_length"), lc_u32(*v, "head_dim"));
    }
    kv.u32("general.file_type", (uint32_t) outtype);

    // BertModel.set_gguf_parameters
    kv.boolean(lc_arch_key("attention.causal"), false);
    // modules.json (sentence-transformers pooling): laya checkpoints have none. Python would follow its
    // pooling path to another config.json anywhere on disk; refused instead of ported.
    if (lc_stat(lc_join(c.dir, "modules.json")).exists) {
        lc_fail("modules.json (sentence-transformers pooling) is not supported in a laya checkpoint");
    }

    // ModernBertModel.set_gguf_parameters
    const json * sw = lc_get(hp, "local_attention");
    if (!sw) lc_fail("encoder/config.json: missing local_attention");
    kv.u32(lc_arch_key("attention.sliding_window"), lc_u32(*sw, "local_attention"));
    if (const json * v = lc_get_nn(hp, "global_attn_every_n_layers")) {
        kv.u32(lc_arch_key("attention.sliding_window_pattern"), lc_u32(*v, "global_attn_every_n_layers"));
    }
    kv.str(lc_arch_key("rope.scaling.type"), "none");
    const json * vs = lc_get(hp, "vocab_size");
    if (!vs) lc_fail("encoder/config.json: missing vocab_size");
    kv.u32(lc_arch_key("vocab_size"), lc_u32(*vs, "vocab_size"));
    if (const json * v = lc_get(hp, "hidden_activation")) {
        if (lc_truthy(*v)) {
            kv.str(lc_arch_key("hidden_activation"), lc_str(*v, "hidden_activation"));
        }
    }

    // LayaModel.set_gguf_parameters
    const ordered_json & rl = h.rl;
    auto rl_u32 = [&](const char * k, uint32_t dflt) -> uint32_t {
        auto it = rl.find(k);
        return it == rl.end() ? dflt : lc_u32(*it, std::string("rl_agent_config.json: ") + k);
    };
    kv.u32(lc_arch_key("head_layers"), rl_u32("head_layers", 2));
    kv.u32(lc_arch_key("n_qtype"), 3);
    kv.u32(lc_arch_key("max_len"), rl_u32("max_len", 512));
    kv.u32(lc_arch_key("head_max_len"), rl_u32("head_max_len", 192));
    {
        auto it = rl.find("temperature");
        const ordered_json temp = it == rl.end() ? ordered_json::array({ 1.0, 1.0, 1.0 }) : *it;
        lc_put_json_array(kv, lc_arch_key("temperature"), temp, "rl_agent_config.json: temperature");
    }
    {
        auto it = rl.find("act_costs");
        uint32_t n = 0;
        if (it != rl.end()) {
            if (!it->is_object() && !it->is_array()) {
                lc_fail("rl_agent_config.json: act_costs is not an object");
            }
            n = (uint32_t) it->size();
        }
        kv.u32(lc_arch_key("act_classes"), n + 1);
    }
    {
        auto it = rl.find("temperature_by_options");
        if (it != rl.end() && lc_truthy(*it)) {
            if (!it->is_object()) {
                lc_fail("rl_agent_config.json: temperature_by_options is not an object");
            }
            std::vector<std::string> buckets;
            std::vector<float> values;
            for (auto e = it->begin(); e != it->end(); ++e) {
                buckets.push_back(e.key());
                if (e.value().is_boolean()) {
                    lc_fail("rl_agent_config.json: temperature_by_options['" + e.key() + "'] is not a number");
                }
                values.push_back(lc_f32(e.value(), "rl_agent_config.json: temperature_by_options['" + e.key() + "']"));
            }
            kv.arr_str(lc_arch_key("temperature_by_options.buckets"), std::move(buckets));
            kv.arr_f32(lc_arch_key("temperature_by_options.values"), std::move(values));
        }
    }
}

// ---------------------------------------------------------------------------------------------
// general.* (Metadata.load + set_gguf_meta_model)
// ---------------------------------------------------------------------------------------------

struct lc_base_entry {
    std::vector<std::pair<std::string, std::string>> fields; // name, organization, version, repo_url
};

void lc_general_metadata(lc_ctx & c, int64_t total_params) {
    const lc_ymap card = lc_load_model_card(c.dir);

    // attributes set from the card hold the YAML value; checked when written
    std::map<std::string, lc_yval> attr;
    auto is_none = [&](const std::string & a) { auto it = attr.find(a); return it == attr.end() || it->second.kind == lc_yval::NUL; };
    auto use = [&](const std::string & a, const std::string & key) {
        const lc_yval * v = lc_card_get(card, key);
        if (v && is_none(a)) {
            attr[a] = *v;
        }
    };
    std::optional<std::vector<lc_yval>> tags, languages;
    auto use_array = [&](std::optional<std::vector<lc_yval>> & dst, const std::string & key) {
        const lc_yval * v = lc_card_get(card, key);
        if (!v || v->kind == lc_yval::NUL) {
            return;
        }
        if (!dst) dst.emplace();
        if (v->kind == lc_yval::STR) {
            dst->push_back(*v);
        } else if (v->kind == lc_yval::LIST) {
            dst->insert(dst->end(), v->list.begin(), v->list.end());
        } else if (v->kind == lc_yval::OTHER) {
            lc_fail("README.md: unsupported YAML for '" + key + "' (" + v->s + ")");
        }
    };
    use("name", "name"); use("author", "author"); use("version", "version"); use("organization", "organization");
    use("description", "description"); use("finetune", "finetune"); use("basename", "basename"); use("size_label", "size_label");
    use("source_url", "url"); use("source_doi", "doi"); use("source_uuid", "uuid"); use("source_repo_url", "repo_url");
    use("name", "model_name"); use("author", "model_author"); use("version", "model_version"); use("organization", "model_organization");
    use("description", "model_description"); use("finetune", "model_finetune"); use("basename", "model_basename"); use("size_label", "model_size_label");
    use("source_url", "model_url"); use("source_doi", "model_doi"); use("source_uuid", "model_uuid"); use("source_repo_url", "model_repo_url");
    use("name", "model_name"); use("author", "model_creator"); use("basename", "model_type");

    bool ascii_only = true;
    // base models / datasets: HF ids and URLs
    auto sources = [&](std::initializer_list<const char *> keys, std::optional<std::vector<lc_base_entry>> & dst) {
        const lc_yval * v = nullptr;
        bool present = false;
        for (const char * k : keys) {
            if (lc_card_get(card, k)) present = true;
        }
        if (!present) return;
        for (const char * k : keys) {
            if ((v = lc_card_get(card, k))) break;
        }
        std::vector<lc_yval> ids;
        if (v) {
            if (v->kind == lc_yval::STR) ids.push_back(*v);
            else if (v->kind == lc_yval::LIST) ids = v->list;
            else if (v->kind == lc_yval::OTHER) lc_fail(std::string("README.md: unsupported YAML for '") + *keys.begin() + "' (" + v->s + ")");
        }
        if (!dst) dst.emplace();
        for (const lc_yval & id : ids) {
            lc_base_entry e;
            if (id.kind == lc_yval::OTHER) {
                lc_fail(std::string("README.md: unsupported entry in '") + *keys.begin() + "' (" + id.s + ")");
            }
            if (id.kind == lc_yval::STR) {
                const std::string & s = id.s;
                auto add_components = [&](const std::string & model_id, bool with_repo) {
                    const lc_id_components comp = lc_model_id_components(model_id, total_params, ascii_only);
                    if (comp.full) e.fields.emplace_back("name", lc_id_to_title(*comp.full, ascii_only));
                    if (comp.org) e.fields.emplace_back("organization", lc_id_to_title(*comp.org, ascii_only));
                    if (comp.version) e.fields.emplace_back("version", *comp.version);
                    if (with_repo && comp.org && comp.full) e.fields.emplace_back("repo_url", "https://huggingface.co/" + *comp.org + "/" + *comp.full);
                };
                if (lc_starts_with(s, "http://") || lc_starts_with(s, "https://") || lc_starts_with(s, "ssh://")) {
                    e.fields.emplace_back("repo_url", s);
                    if (s.find("huggingface.co") != std::string::npos) {
                        // re.match(r"https?://huggingface.co/([^/]+/[^/]+)$", s)
                        const std::string pre = lc_starts_with(s, "https://") ? "https://huggingface.co/" : "http://huggingface.co/";
                        if (lc_starts_with(s, pre)) {
                            const std::string rest = s.substr(pre.size());
                            const size_t sl = rest.find('/');
                            if (sl != std::string::npos && sl > 0 && sl + 1 < rest.size() && rest.find('/', sl + 1) == std::string::npos && rest.find('\n') == std::string::npos) {
                                add_components(rest, false);
                            }
                        }
                    }
                } else {
                    add_components(s, true);
                }
            } else if (id.kind == lc_yval::LIST) {
                lc_fail(std::string("README.md: unsupported entry in '") + *keys.begin() + "'");
            }
            // other scalars: Python logs an error and appends an empty entry
            dst->push_back(std::move(e));
        }
    };
    std::optional<std::vector<lc_base_entry>> base_models, datasets;
    sources({ "base_model", "base_models", "base_model_sources" }, base_models);
    sources({ "datasets", "dataset", "dataset_sources" }, datasets);

    use("license", "license"); use("license_name", "license_name"); use("license_link", "license_link");
    use_array(tags, "tags");
    use_array(tags, "pipeline_tag");
    use_array(languages, "languages");
    use_array(languages, "language");

    // directory name fallback
    std::map<std::string, std::string> fallback;
    {
        const std::string dir_name = lc_path_name(c.dir);
        if (!lc_valid_utf8(dir_name)) {
            lc_fail("the checkpoint directory name is not valid UTF-8");
        }
        const lc_id_components comp = lc_model_id_components(dir_name, total_params, ascii_only);
        if (comp.full)       fallback["name"] = lc_id_to_title(*comp.full, ascii_only);
        if (comp.org)        fallback["organization"] = lc_id_to_title(*comp.org, ascii_only);
        if (comp.basename)   fallback["basename"] = *comp.basename;
        if (comp.finetune)   fallback["finetune"] = *comp.finetune;
        if (comp.version)    fallback["version"] = *comp.version;
        if (comp.size_label) fallback["size_label"] = *comp.size_label;
    }
    for (const auto & f : fallback) {
        if (is_none(f.first)) {
            lc_yval v;
            v.kind = lc_yval::STR;
            v.s = f.second;
            attr[f.first] = v;
        }
    }
    if (!ascii_only) {
        c.log("warning: non-ASCII text in the directory name or model card ids: general.* derived from it may differ from Python (Unicode case rules); pass --model-name");
    }
    if (c.params->has_model_name) {
        lc_yval v;
        v.kind = lc_yval::STR;
        v.s = c.params->model_name;
        attr["name"] = v;
    }
    if (is_none("name")) {
        lc_yval v;
        v.kind = lc_yval::STR;
        v.s = lc_path_name(c.dir);
        attr["name"] = v;
    }
    if (is_none("size_label") && total_params > 0) {
        lc_yval v;
        v.kind = lc_yval::STR;
        v.s = lc_size_label(total_params);
        attr["size_label"] = v;
    }

    // generation_config.json sampling keys
    json gen = json::object();
    {
        const std::string gp = lc_join(c.dir, "generation_config.json");
        if (lc_stat(gp).is_file) {
            const std::string text = lc_read_file(gp, LC_MAX_CONFIG_BYTES);
            if (!lc_valid_utf8(text)) {
                lc_fail(gp + ": not valid UTF-8");
            }
            // Metadata.load_generation_config ignores json.JSONDecodeError only: a syntax error or a BOM.
            // What Python reads but nlohmann does not (NaN, Infinity, lone surrogate escapes) would give
            // other general.sampling.* bytes, and nesting Python cannot parse (RecursionError) fails there:
            // both are refused instead of being ignored.
            lc_check_json_depth(text, gp);
            const bool bom = text.size() >= 3 && (unsigned char) text[0] == 0xEF && (unsigned char) text[1] == 0xBB && (unsigned char) text[2] == 0xBF;
            if (bom) {
                gen = json::object();
            } else {
                try {
                    gen = json::parse(text);
                } catch (const std::exception &) {
                    if (lc_json_python_only(text)) {
                        lc_fail(gp + ": NaN / Infinity / a lone surrogate escape (Python's json reads them); not supported");
                    }
                    gen = json::object(); // Python ignores an invalid generation_config.json
                }
            }
            if (!gen.is_object()) {
                if (lc_truthy(gen)) lc_fail(gp + ": not a JSON object");
                gen = json::object();
            }
        }
    }

    // set_gguf_meta_model order
    lc_kvs & kv = c.kvs;
    struct samp { const char * src; const char * key; char kind; };
    static const samp sampling[] = {
        { "sequence", "general.sampling.sequence", 's' }, { "top_k", "general.sampling.top_k", 'i' }, { "top_p", "general.sampling.top_p", 'f' },
        { "min_p", "general.sampling.min_p", 'f' }, { "xtc_probability", "general.sampling.xtc_probability", 'f' },
        { "xtc_threshold", "general.sampling.xtc_threshold", 'f' }, { "temperature", "general.sampling.temp", 'f' },
        { "penalty_last_n", "general.sampling.penalty_last_n", 'i' }, { "penalty_repeat", "general.sampling.penalty_repeat", 'f' },
        { "mirostat", "general.sampling.mirostat", 'i' }, { "mirostat_tau", "general.sampling.mirostat_tau", 'f' },
        { "mirostat_eta", "general.sampling.mirostat_eta", 'f' },
    };
    for (const samp & s : sampling) {
        const json * v = lc_get_nn(gen, s.src);
        if (!v) continue;
        const std::string what = std::string("generation_config.json: ") + s.src;
        if (v->is_boolean()) lc_fail(what + ": bool is not supported");
        if (s.kind == 's') kv.str(s.key, lc_str(*v, what));
        else if (s.kind == 'i') kv.i32(s.key, lc_i32(*v, what));
        else kv.f32(s.key, lc_f32(*v, what));
    }
    auto put_str = [&](const std::string & a, const std::string & key) {
        auto it = attr.find(a);
        if (it == attr.end() || it->second.kind == lc_yval::NUL) return;
        if (it->second.kind == lc_yval::OTHER) {
            lc_fail("README.md: unsupported YAML for '" + a + "' (" + it->second.s + ")");
        }
        if (it->second.kind != lc_yval::STR) {
            lc_fail("README.md: '" + a + "' is not a string");
        }
        kv.str(key, it->second.s);
    };
    put_str("name", "general.name");
    put_str("author", "general.author");
    put_str("version", "general.version");
    put_str("organization", "general.organization");
    put_str("finetune", "general.finetune");
    put_str("basename", "general.basename");
    put_str("description", "general.description");
    put_str("size_label", "general.size_label");
    {
        auto it = attr.find("license");
        if (it != attr.end() && it->second.kind != lc_yval::NUL) {
            if (it->second.kind == lc_yval::LIST) {
                std::string s;
                for (size_t i = 0; i < it->second.list.size(); i++) {
                    if (it->second.list[i].kind != lc_yval::STR) lc_fail("README.md: license list with a non-string item");
                    if (i > 0) s += ",";
                    s += it->second.list[i].s;
                }
                kv.str("general.license", s);
            } else {
                put_str("license", "general.license");
            }
        }
    }
    put_str("license_name", "general.license.name");
    put_str("license_link", "general.license.link");
    put_str("source_url", "general.source.url");
    put_str("source_doi", "general.source.doi");
    put_str("source_uuid", "general.source.uuid");
    put_str("source_repo_url", "general.source.repo_url");
    auto put_sources = [&](const std::optional<std::vector<lc_base_entry>> & src, const char * prefix) {
        if (!src) return;
        kv.u32(std::string("general.") + prefix + ".count", (uint32_t) src->size());
        static const char * order[] = { "name", "author", "version", "organization", "description", "url", "doi", "uuid", "repo_url" };
        for (size_t i = 0; i < src->size(); i++) {
            for (const char * f : order) {
                for (const auto & fv : (*src)[i].fields) {
                    if (fv.first == f) {
                        kv.str(std::string("general.") + prefix + "." + std::to_string(i) + "." + f, fv.second);
                    }
                }
            }
        }
    };
    put_sources(base_models, "base_model");
    put_sources(datasets, "dataset");
    auto put_list = [&](const std::optional<std::vector<lc_yval>> & l, const char * key) {
        if (!l) return;
        std::vector<std::string> out;
        for (const lc_yval & v : *l) {
            if (v.kind != lc_yval::STR) lc_fail(std::string("README.md: ") + key + " has a non-string item");
            out.push_back(v.s);
        }
        kv.arr_str(key, std::move(out));
    };
    put_list(tags, "general.tags");
    put_list(languages, "general.languages");
}

// ---------------------------------------------------------------------------------------------
// GGUF writer
// ---------------------------------------------------------------------------------------------

struct lc_writer {
    FILE *   f = nullptr;
    uint64_t n = 0;

    void bytes(const void * p, size_t len) {
        if (len > 0 && fwrite(p, 1, len, f) != len) {
            lc_fail("write failed (disk full?)");
        }
        n += len;
    }
    void u8(uint8_t x) { bytes(&x, 1); }
    void u32(uint32_t x) { unsigned char b[4]; for (int i = 0; i < 4; i++) b[i] = (unsigned char) (x >> (8 * i)); bytes(b, 4); }
    void u64(uint64_t x) { unsigned char b[8]; for (int i = 0; i < 8; i++) b[i] = (unsigned char) (x >> (8 * i)); bytes(b, 8); }
    void i32(int32_t x) { u32((uint32_t) x); }
    void f32(float x) { u32(lc_f32_bits(x)); }
    void str(const std::string & s) { u64(s.size()); bytes(s.data(), s.size()); }
    void pad(uint64_t align) {
        static const char zeros[64] = { 0 };
        while (n % align != 0) {
            const size_t k = (size_t) std::min<uint64_t>(align - n % align, sizeof(zeros));
            bytes(zeros, k);
        }
    }
};

void lc_write_kv(lc_writer & w, const lc_kv & kv) {
    w.str(kv.key);
    w.u32(kv.type);
    switch (kv.type) {
        case LC_U32:  w.u32(kv.u32); break;
        case LC_I32:  w.i32(kv.i32); break;
        case LC_F32:  w.f32(kv.f32); break;
        case LC_BOOL: w.u8(kv.b ? 1 : 0); break;
        case LC_STR:  w.str(kv.s); break;
        case LC_ARR:
            w.u32(kv.arr_type);
            w.u64(kv.arr_n());
            switch (kv.arr_type) {
                case LC_STR:  for (const std::string & s : kv.as) w.str(s); break;
                case LC_I32:  for (int32_t x : kv.ai) w.i32(x); break;
                case LC_F32:  for (float x : kv.af) w.f32(x); break;
                case LC_BOOL: for (uint8_t x : kv.ab) w.u8(x); break;
                default: lc_fail("internal: bad array type");
            }
            break;
    }
}

// one tensor's data, converted in row chunks
void lc_write_tensor_data(lc_writer & w, FILE * src, const lc_tensor & t) {
    const int64_t ne0   = t.ne[0];
    const int64_t nrows = (int64_t) (t.n_elem / (uint64_t) ne0);
    const size_t  ssz   = lc_dtype_size(t.src_dtype);
    // one chunk holds at least one row as raw (up to 8 bytes / value), f32 and q8_0 (34 bytes / 32 values):
    // it must fit size_t (32-bit builds)
    if ((uint64_t) ne0 > (uint64_t) SIZE_MAX / 16) {
        lc_fail("tensor '" + t.src_name + "': rows of " + std::to_string(ne0) + " values are too long for this build");
    }
    const int64_t chunk_rows = std::max<int64_t>(1, (int64_t) (4 << 20) / ne0);
    std::vector<uint8_t>  raw;
    std::vector<float>    f32;
    std::vector<uint16_t> f16;
    std::vector<uint8_t>  q8;
    if (!lc_seek(src, t.src_offset)) {
        lc_fail("seek failed in the safetensors file for '" + t.src_name + "'");
    }
    for (int64_t r0 = 0; r0 < nrows; r0 += chunk_rows) {
        const int64_t nr = std::min(chunk_rows, nrows - r0);
        const size_t  n  = (size_t) (nr * ne0);
        raw.resize(n * ssz);
        if (fread(raw.data(), 1, raw.size(), src) != raw.size()) {
            lc_fail("short read in the safetensors file for '" + t.src_name + "' (truncated file?)");
        }
        if (t.type == GGML_TYPE_F16 && t.src_dtype == LC_DT_F16) {
            w.bytes(raw.data(), raw.size());
            continue;
        }
        f32.resize(n);
        switch (t.src_dtype) {
            case LC_DT_F32:
                memcpy(f32.data(), raw.data(), n * 4);
                break;
            case LC_DT_F16:
                for (size_t i = 0; i < n; i++) {
                    uint16_t h;
                    memcpy(&h, raw.data() + 2 * i, 2);
                    f32[i] = laya_convert_fp16_to_fp32(h);
                }
                break;
            case LC_DT_BF16:
                for (size_t i = 0; i < n; i++) {
                    uint16_t h;
                    memcpy(&h, raw.data() + 2 * i, 2);
                    f32[i] = lc_bits_f32((uint32_t) h << 16);
                }
                break;
            case LC_DT_F64:
                for (size_t i = 0; i < n; i++) {
                    double d;
                    memcpy(&d, raw.data() + 8 * i, 8);
                    f32[i] = (float) d;
                }
                break;
        }
        if (t.type == GGML_TYPE_F32) {
            w.bytes(f32.data(), n * 4);
        } else if (t.type == GGML_TYPE_F16) {
            f16.resize(n);
            for (size_t i = 0; i < n; i++) {
                f16[i] = laya_convert_fp32_to_fp16(f32[i]);
            }
            w.bytes(f16.data(), n * 2);
        } else {
            for (size_t i = 0; i < n; i++) {
                if (!std::isfinite(f32[i])) {
                    lc_fail("tensor '" + t.src_name + "' has a non-finite value; q8_0 of it would depend on the platform");
                }
            }
            q8.resize((size_t) ggml_row_size(GGML_TYPE_Q8_0, ne0) * (size_t) nr);
            const size_t written = ggml_quantize_chunk(GGML_TYPE_Q8_0, f32.data(), q8.data(), 0, nr, ne0, nullptr);
            if (written != q8.size()) {
                lc_fail("internal: q8_0 size mismatch");
            }
            w.bytes(q8.data(), q8.size());
        }
    }
    w.pad(32);
}

// read the file back with the gguf API and check the layout
void lc_check_written(const std::string & path, const lc_ctx & c, uint64_t data_offset, uint64_t file_size) {
    lc_file f(laya_convert_fopen(path, "rb"));
    if (!f) {
        lc_fail("cannot reopen '" + path + "'");
    }
    gguf_init_params ip = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file_ptr(f.get(), ip);
    if (!g) {
        lc_fail("internal: gguf_init_from_file_ptr rejects the written file");
    }
    std::unique_ptr<gguf_context, decltype(&gguf_free)> guard(g, gguf_free);
    if (gguf_get_n_kv(g) != (int64_t) c.kvs.v.size() || gguf_get_n_tensors(g) != (int64_t) c.tensors.size() ||
        gguf_get_data_offset(g) != data_offset || gguf_get_alignment(g) != 32) {
        lc_fail("internal: the written GGUF header does not read back");
    }
    for (size_t i = 0; i < c.kvs.v.size(); i++) {
        if (c.kvs.v[i].key != gguf_get_key(g, (int64_t) i) || (uint32_t) gguf_get_kv_type(g, (int64_t) i) != c.kvs.v[i].type) {
            lc_fail("internal: KV " + c.kvs.v[i].key + " does not read back");
        }
    }
    for (size_t i = 0; i < c.tensors.size(); i++) {
        const lc_tensor & t = c.tensors[i];
        const int64_t * ne = gguf_get_tensor_ne(g, (int64_t) i);
        bool ok = t.name == gguf_get_tensor_name(g, (int64_t) i) && gguf_get_tensor_type(g, (int64_t) i) == t.type &&
                  gguf_get_tensor_offset(g, (int64_t) i) == t.offset && gguf_get_tensor_size(g, (int64_t) i) == t.nbytes;
        for (size_t d = 0; d < 4; d++) {
            ok = ok && ne[d] == (d < t.ne.size() ? t.ne[d] : 1);
        }
        if (!ok) {
            lc_fail("internal: tensor " + t.name + " does not read back");
        }
    }
    const lc_tensor & last = c.tensors.back();
    if (data_offset + last.offset + (last.nbytes + 31) / 32 * 32 != file_size) {
        lc_fail("internal: file size mismatch");
    }
}

void lc_convert(const std::string & dir, const std::string & out, const laya_convert_params & params) {
    lc_ctx c;
    c.dir = dir;
    c.params = &params;
    const lc_stat_info ds = lc_stat(dir);
    if (!ds.is_dir) {
        lc_fail("'" + dir + "' is not a directory");
    }
    if (params.outtype != LAYA_CONVERT_F32 && params.outtype != LAYA_CONVERT_F16 && params.outtype != LAYA_CONVERT_Q8_0) {
        lc_fail("unsupported outtype");
    }
    if (out.empty()) {
        lc_fail("no output path");
    }
    const std::string tok = lc_join(dir, "tokenizer");
    c.tok_dir = lc_stat(tok).is_dir ? tok : dir;
    for (const char * f : { "tokenizer.json", "tokenizer_config.json" }) {
        if (!lc_stat(lc_join(c.tok_dir, f)).is_file) {
            lc_fail("missing tokenizer file '" + lc_join(c.tok_dir, f) + "'");
        }
    }

    lc_hparams h = lc_load_hparams(c);
    lc_collect_tensors(c, h.block_count, params.outtype);
    int64_t total_params = 0;
    for (const lc_tensor & t : c.tensors) {
        total_params += (int64_t) t.n_elem;
    }

    // ModelBase.prepare_metadata, then TextModel.prepare_metadata (set_vocab)
    c.kvs.str("general.architecture", LC_ARCH);
    c.kvs.str("general.type", "model");
    lc_general_metadata(c, total_params);
    lc_set_gguf_parameters(c, h, params.outtype);
    c.kvs.u32("general.quantization_version", 2);
    lc_convert_vocab(c, h.hp);

    // write to a new, uniquely named out.<hex>.tmp, check, sync, rename
    const std::string tmp = out + "." + lc_random_hex() + ".tmp";
    lc_file f(lc_create_excl(tmp));
    if (!f) {
        lc_fail("cannot create '" + tmp + "'");
    }
    std::vector<char> iobuf(4 << 20);
    setvbuf(f.get(), iobuf.data(), _IOFBF, iobuf.size());
    lc_writer w;
    w.f = f.get();
    try {
        w.bytes("GGUF", 4);
        w.u32(3);
        w.u64(c.tensors.size());
        w.u64(c.kvs.v.size());
        for (const lc_kv & kv : c.kvs.v) {
            lc_write_kv(w, kv);
        }
        for (const lc_tensor & t : c.tensors) {
            w.str(t.name);
            w.u32((uint32_t) t.ne.size());
            for (int64_t d : t.ne) {
                w.u64((uint64_t) d);
            }
            w.i32((int32_t) t.type);
            w.u64(t.offset);
        }
        w.pad(32);
        const uint64_t data_offset = w.n;
        c.log("writing " + std::to_string(c.tensors.size()) + " tensors, " + std::to_string(c.kvs.v.size()) + " KV, data at " + std::to_string(data_offset));
        std::vector<lc_file> srcs;
        for (const std::string & p : c.st_files) {
            srcs.emplace_back(laya_convert_fopen(p, "rb"));
            if (!srcs.back()) {
                lc_fail("cannot open '" + p + "'");
            }
        }
        for (const lc_tensor & t : c.tensors) {
            if (w.n != data_offset + t.offset) {
                lc_fail("internal: tensor offset mismatch");
            }
            std::string shape;
            for (size_t d = 0; d < t.ne.size(); d++) {
                shape += (d ? ", " : "") + std::to_string(t.ne[d]);
            }
            c.log(t.name + ", " + lc_dtype_name(t.src_dtype) + " --> " + ggml_type_name(t.type) + ", shape = {" + shape + "}");
            lc_write_tensor_data(w, srcs[t.src_file].get(), t);
        }
        if (!lc_sync_file(f.get())) {
            lc_fail("write failed (disk full?)");
        }
        const uint64_t size = w.n;
        f.reset();
        lc_check_written(tmp, c, data_offset, size);
        if (!lc_rename_replace(tmp, out)) {
            lc_fail("cannot rename '" + tmp + "' to '" + out + "'");
        }
        c.log("wrote '" + out + "' (" + std::to_string(size) + " bytes)");
    } catch (...) {
        f.reset();
        laya_convert_remove(tmp);
        throw;
    }
}

// ---------------------------------------------------------------------------------------------
// compare
// ---------------------------------------------------------------------------------------------

size_t lc_gguf_type_size(gguf_type t) {
    switch (t) {
        case GGUF_TYPE_UINT8:  case GGUF_TYPE_INT8:  case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16: case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32: case GGUF_TYPE_INT32: case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64: case GGUF_TYPE_INT64: case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

std::string lc_kv_value_str(const gguf_context * g, int64_t i, size_t max_items = 8) {
    const gguf_type t = gguf_get_kv_type(g, i);
    char buf[128];
    auto scalar = [&](gguf_type type, const void * p) -> std::string {
        switch (type) {
            case GGUF_TYPE_UINT8:   return std::to_string(*(const uint8_t *) p);
            case GGUF_TYPE_INT8:    return std::to_string(*(const int8_t *) p);
            case GGUF_TYPE_UINT16:  return std::to_string(*(const uint16_t *) p);
            case GGUF_TYPE_INT16:   return std::to_string(*(const int16_t *) p);
            case GGUF_TYPE_UINT32:  return std::to_string(*(const uint32_t *) p);
            case GGUF_TYPE_INT32:   return std::to_string(*(const int32_t *) p);
            case GGUF_TYPE_UINT64:  return std::to_string(*(const uint64_t *) p);
            case GGUF_TYPE_INT64:   return std::to_string(*(const int64_t *) p);
            case GGUF_TYPE_FLOAT32: snprintf(buf, sizeof(buf), "%.9g", *(const float *) p); return buf;
            case GGUF_TYPE_FLOAT64: snprintf(buf, sizeof(buf), "%.17g", *(const double *) p); return buf;
            case GGUF_TYPE_BOOL:    return *(const int8_t *) p ? "true" : "false";
            default:                return "?";
        }
    };
    if (t == GGUF_TYPE_STRING) {
        return json(std::string(gguf_get_val_str(g, i))).dump();
    }
    if (t != GGUF_TYPE_ARRAY) {
        return scalar(t, gguf_get_val_data(g, i));
    }
    const gguf_type at = gguf_get_arr_type(g, i);
    const size_t n = gguf_get_arr_n(g, i);
    std::string s = std::string(gguf_type_name(at)) + "[" + std::to_string(n) + "] [";
    for (size_t k = 0; k < n && k < max_items; k++) {
        if (k) s += ", ";
        if (at == GGUF_TYPE_STRING) {
            s += json(std::string(gguf_get_arr_str(g, i, k))).dump();
        } else {
            s += scalar(at, (const char *) gguf_get_arr_data(g, i) + k * lc_gguf_type_size(at));
        }
    }
    if (n > max_items) s += ", ...";
    return s + "]";
}

// "" when equal, else where they differ
std::string lc_kv_diff(const gguf_context * a, const gguf_context * b, int64_t i) {
    const std::string key = gguf_get_key(a, i);
    if (key != gguf_get_key(b, i)) {
        return "KV #" + std::to_string(i) + ": key " + key + " vs " + gguf_get_key(b, i);
    }
    const gguf_type ta = gguf_get_kv_type(a, i), tb = gguf_get_kv_type(b, i);
    if (ta != tb) {
        return "KV " + key + ": type " + gguf_type_name(ta) + " vs " + gguf_type_name(tb);
    }
    if (ta == GGUF_TYPE_STRING) {
        if (strcmp(gguf_get_val_str(a, i), gguf_get_val_str(b, i)) != 0) {
            return "KV " + key + ": " + lc_kv_value_str(a, i) + " vs " + lc_kv_value_str(b, i);
        }
        return "";
    }
    if (ta != GGUF_TYPE_ARRAY) {
        if (memcmp(gguf_get_val_data(a, i), gguf_get_val_data(b, i), lc_gguf_type_size(ta)) != 0) {
            return "KV " + key + ": " + lc_kv_value_str(a, i) + " vs " + lc_kv_value_str(b, i);
        }
        return "";
    }
    const gguf_type aa = gguf_get_arr_type(a, i), ab = gguf_get_arr_type(b, i);
    const size_t na = gguf_get_arr_n(a, i), nb = gguf_get_arr_n(b, i);
    if (aa != ab || na != nb) {
        return "KV " + key + ": " + lc_kv_value_str(a, i) + " vs " + lc_kv_value_str(b, i);
    }
    for (size_t k = 0; k < na; k++) {
        bool same;
        std::string va, vb;
        if (aa == GGUF_TYPE_STRING) {
            va = gguf_get_arr_str(a, i, k);
            vb = gguf_get_arr_str(b, i, k);
            same = va == vb;
            va = json(va).dump();
            vb = json(vb).dump();
        } else {
            const size_t es = lc_gguf_type_size(aa);
            const char * pa = (const char *) gguf_get_arr_data(a, i) + k * es;
            const char * pb = (const char *) gguf_get_arr_data(b, i) + k * es;
            same = memcmp(pa, pb, es) == 0;
            if (!same) {
                va = "0x";
                vb = "0x";
                char h[4];
                for (size_t e = es; e-- > 0;) {
                    snprintf(h, sizeof(h), "%02x", (unsigned char) pa[e]); va += h;
                    snprintf(h, sizeof(h), "%02x", (unsigned char) pb[e]); vb += h;
                }
            }
        }
        if (!same) {
            return "KV " + key + "[" + std::to_string(k) + "]: " + va + " vs " + vb;
        }
    }
    return "";
}

// failed: set when the files could not be compared (open, parse, seek), as opposed to a difference
std::string lc_compare(const std::string & pa, const std::string & pb, bool & failed) {
    failed = true;
    lc_file fa(laya_convert_fopen(pa, "rb"));
    lc_file fb(laya_convert_fopen(pb, "rb"));
    if (!fa) return "cannot open '" + pa + "'";
    if (!fb) return "cannot open '" + pb + "'";
    gguf_init_params ip = { /*no_alloc =*/ true, /*ctx =*/ nullptr };
    std::unique_ptr<gguf_context, decltype(&gguf_free)> ga(gguf_init_from_file_ptr(fa.get(), ip), gguf_free);
    std::unique_ptr<gguf_context, decltype(&gguf_free)> gb(gguf_init_from_file_ptr(fb.get(), ip), gguf_free);
    if (!ga) return "'" + pa + "' is not a valid GGUF";
    if (!gb) return "'" + pb + "' is not a valid GGUF";
    failed = false;
    const gguf_context * a = ga.get();
    const gguf_context * b = gb.get();
    if (gguf_get_version(a) != gguf_get_version(b)) {
        return "GGUF version " + std::to_string(gguf_get_version(a)) + " vs " + std::to_string(gguf_get_version(b));
    }
    const int64_t nka = gguf_get_n_kv(a), nkb = gguf_get_n_kv(b);
    for (int64_t i = 0; i < std::min(nka, nkb); i++) {
        const std::string d = lc_kv_diff(a, b, i);
        if (!d.empty()) return d;
    }
    if (nka != nkb) {
        const bool a_more = nka > nkb;
        return "KV count " + std::to_string(nka) + " vs " + std::to_string(nkb) + "; first extra key: " +
               gguf_get_key(a_more ? a : b, std::min(nka, nkb)) + " in " + (a_more ? pa : pb);
    }
    const int64_t nta = gguf_get_n_tensors(a), ntb = gguf_get_n_tensors(b);
    for (int64_t i = 0; i < std::min(nta, ntb); i++) {
        const std::string na = gguf_get_tensor_name(a, i), nb = gguf_get_tensor_name(b, i);
        std::string why;
        if (na != nb) why = "name " + na + " vs " + nb;
        else if (gguf_get_tensor_type(a, i) != gguf_get_tensor_type(b, i)) why = std::string("type ") + ggml_type_name(gguf_get_tensor_type(a, i)) + " vs " + ggml_type_name(gguf_get_tensor_type(b, i));
        else if (memcmp(gguf_get_tensor_ne(a, i), gguf_get_tensor_ne(b, i), 4 * sizeof(int64_t)) != 0) {
            auto ne = [](const int64_t * x) { return "{" + std::to_string(x[0]) + ", " + std::to_string(x[1]) + ", " + std::to_string(x[2]) + ", " + std::to_string(x[3]) + "}"; };
            why = "ne " + ne(gguf_get_tensor_ne(a, i)) + " vs " + ne(gguf_get_tensor_ne(b, i));
        } else if (gguf_get_tensor_offset(a, i) != gguf_get_tensor_offset(b, i)) {
            why = "offset " + std::to_string(gguf_get_tensor_offset(a, i)) + " vs " + std::to_string(gguf_get_tensor_offset(b, i));
        }
        if (!why.empty()) return "tensor #" + std::to_string(i) + " (" + na + "): " + why;
    }
    if (nta != ntb) {
        return "tensor count " + std::to_string(nta) + " vs " + std::to_string(ntb);
    }
    // tensor data
    std::vector<char> ba(1 << 20), bb(1 << 20);
    for (int64_t i = 0; i < nta; i++) {
        const uint64_t oa = gguf_get_data_offset(a) + gguf_get_tensor_offset(a, i);
        const uint64_t ob = gguf_get_data_offset(b) + gguf_get_tensor_offset(b, i);
        const uint64_t size = gguf_get_tensor_size(a, i);
        if (!lc_seek(fa.get(), oa) || !lc_seek(fb.get(), ob)) { failed = true; return "seek failed"; }
        for (uint64_t done = 0; done < size;) {
            const size_t k = (size_t) std::min<uint64_t>(ba.size(), size - done);
            const size_t ra = fread(ba.data(), 1, k, fa.get());
            const size_t rb = fread(bb.data(), 1, k, fb.get());
            if (ra != k || rb != k) {
                return std::string("tensor ") + gguf_get_tensor_name(a, i) + ": data is truncated in " + (ra != k ? pa : pb);
            }
            if (memcmp(ba.data(), bb.data(), k) != 0) {
                size_t j = 0;
                while (ba[j] == bb[j]) j++;
                return std::string("tensor ") + gguf_get_tensor_name(a, i) + ": data differs at byte " + std::to_string(done + j) + " of " + std::to_string(size);
            }
            done += k;
        }
    }
    // raw bytes (tensor-info n_dims, padding, trailing data)
    if (!lc_seek(fa.get(), 0) || !lc_seek(fb.get(), 0)) { failed = true; return "seek failed"; }
    for (uint64_t off = 0;;) {
        const size_t ra = fread(ba.data(), 1, ba.size(), fa.get());
        const size_t rb = fread(bb.data(), 1, bb.size(), fb.get());
        const size_t k = std::min(ra, rb);
        if (memcmp(ba.data(), bb.data(), k) != 0) {
            size_t j = 0;
            while (ba[j] == bb[j]) j++;
            const uint64_t pos = off + j;
            const char * where = pos < gguf_get_data_offset(a) ? "header / tensor info / padding" : "data section";
            return "byte " + std::to_string(pos) + " differs (" + where + ")";
        }
        if (ra != rb) {
            return "file sizes differ (" + std::to_string(off + ra) + " vs " + std::to_string(off + rb) + " bytes read)";
        }
        if (ra == 0) break;
        off += ra;
    }
    return "";
}

} // namespace

// ---------------------------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------------------------

FILE * laya_convert_fopen(const std::string & utf8_path, const char * mode) {
    return ggml_fopen(utf8_path.c_str(), mode); // UTF-8 name, _wfopen on Windows
}

bool laya_convert_remove(const std::string & utf8_path) {
#ifdef _WIN32
    try {
        return _wremove(lc_widen(utf8_path).c_str()) == 0;
    } catch (const std::exception &) {
        return false;
    }
#else
    return remove(utf8_path.c_str()) == 0;
#endif
}

uint16_t laya_convert_fp32_to_fp16(float f) {
    const uint32_t x    = lc_f32_bits(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t a    = x & 0x7FFFFFFFu;
    if (a >= 0x7F800000u) {
        if (a == 0x7F800000u) {
            return (uint16_t) (sign | 0x7C00u);
        }
        return (uint16_t) (sign | 0x7E00u | ((a >> 13) & 0x3FFu));
    }
    const uint32_t e = a >> 23;
    if (e >= 143) {
        return (uint16_t) (sign | 0x7C00u);
    }
    if (e >= 113) {
        uint32_t h = ((e - 112) << 10) | ((a & 0x7FFFFFu) >> 13);
        const uint32_t rem = a & 0x1FFFu;
        if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
            h++; // may carry into the exponent, up to inf
        }
        return (uint16_t) (sign | h);
    }
    // f16 subnormal or zero: h = m * 2^(e - 126), m with the implicit bit
    const uint32_t s = 126 - e;
    if (s > 24) {
        return (uint16_t) sign;
    }
    const uint32_t m    = (a & 0x7FFFFFu) | 0x800000u;
    uint32_t       h    = m >> s;
    const uint32_t rem  = m & ((1u << s) - 1);
    const uint32_t half = 1u << (s - 1);
    if (rem > half || (rem == half && (h & 1u))) {
        h++;
    }
    return (uint16_t) (sign | h);
}

float laya_convert_fp16_to_fp32(uint16_t h) {
    const uint32_t sign = (uint32_t) (h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    if (e == 0) {
        if (m == 0) {
            return lc_bits_f32(sign);
        }
        e = 113;
        while (!(m & 0x400u)) {
            m <<= 1;
            e--;
        }
        m &= 0x3FFu;
        return lc_bits_f32(sign | (e << 23) | (m << 13));
    }
    if (e == 31) {
        if (m == 0) {
            return lc_bits_f32(sign | 0x7F800000u);
        }
        return lc_bits_f32(sign | 0x7FC00000u | (m << 13));
    }
    return lc_bits_f32(sign | ((e + 112) << 23) | (m << 13));
}

bool laya_convert_parse_outtype(const std::string & s, laya_convert_outtype & out) {
    if (s == "f32")  { out = LAYA_CONVERT_F32;  return true; }
    if (s == "f16")  { out = LAYA_CONVERT_F16;  return true; }
    if (s == "q8_0") { out = LAYA_CONVERT_Q8_0; return true; }
    return false;
}

const char * laya_convert_outtype_name(laya_convert_outtype t) {
    switch (t) {
        case LAYA_CONVERT_F32:  return "f32";
        case LAYA_CONVERT_F16:  return "f16";
        case LAYA_CONVERT_Q8_0: return "q8_0";
    }
    return "?";
}

std::string laya_convert_path_name(const std::string & path) {
    return lc_path_name(path);
}

bool laya_convert_rename(const std::string & from, const std::string & to) {
    try {
        return lc_rename_replace(from, to);
    } catch (const std::exception &) {
        return false;
    }
}

std::string laya_convert(const std::string & dir, const std::string & out, const laya_convert_params & params) {
    try {
        lc_convert(dir, out, params);
        return "";
    } catch (const std::bad_alloc &) {
        return "out of memory";
    } catch (const std::exception & e) {
        const std::string msg = e.what();
        return msg.empty() ? std::string("unknown error") : msg;
    }
}

std::string laya_convert_compare(const std::string & a, const std::string & b, bool * failed) {
    bool f = true;
    std::string r;
    try {
        r = lc_compare(a, b, f);
    } catch (const std::exception & e) {
        f = true;
        r = std::string("compare failed: ") + e.what();
    }
    if (failed) {
        *failed = f && !r.empty();
    }
    return r;
}

std::string laya_convert_dir_metadata(const std::string & dir_name, int64_t total_params) {
    bool ascii_only = true;
    const lc_id_components c = lc_model_id_components(dir_name, total_params, ascii_only);
    auto s = [](const std::optional<std::string> & x) { return x ? *x : std::string("-"); };
    const std::string name = c.full ? lc_id_to_title(*c.full, ascii_only) : std::string("-");
    return name + "|" + s(c.basename) + "|" + s(c.finetune) + "|" + s(c.version) + "|" + s(c.size_label);
}
