#include "decision-checkpoint.h"
#include "decision-spec.h"
#include "laya.h" // laya_utf8_to_wide

#include "gguf.h"

extern "C" {
#include "sha256/sha256.h" // vendored in examples/gguf-hash/deps
}

#include <algorithm>
#include <cctype>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <direct.h>
#    include <stdlib.h>
#else
#    include <dirent.h>
#    include <errno.h>
#    include <fcntl.h>
#    include <limits.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <unistd.h>
#endif

namespace {

const char *   CK_FORMAT         = "laya-gguf-cache 2"; // bump when the manifest or the layout changes
const uint64_t CK_HASH_MAX_BYTES = 8ull << 20;          // files up to this size are keyed by content
const size_t   CK_MAX_FILES      = 20000;               // entries in the keyed directories (real checkpoints: ~10)
const size_t   CK_KEY_HEX        = 32;                  // 128 bits of the sha256 name the key directory
const int64_t  CK_STALE_SECONDS  = 600;                 // temporary files older than this are left by a killed conversion

//
// files (UTF-8 paths; UTF-16 APIs on Windows)
//

#ifdef _WIN32
// laya_utf8_to_wide (laya.h) is empty for invalid UTF-8; false then, so an empty name is never used
bool ck_widen(const std::string & s, std::wstring & w) {
    w = laya_utf8_to_wide(s);
    return s.empty() || !w.empty();
}

bool ck_narrow(const wchar_t * w, std::string & s) {
    s.clear();
    const int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, w, -1, nullptr, 0, nullptr, nullptr);
    if (n <= 0) {
        return false;
    }
    s.assign((size_t) n, '\0');
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, w, -1, &s[0], n, nullptr, nullptr);
    s.resize((size_t) n - 1);
    return true;
}
#endif

struct ck_stat {
    bool     exists  = false;
    bool     is_dir  = false;
    bool     is_file = false;
    uint64_t size    = 0;
    int64_t  mtime   = 0; // POSIX: ns since the epoch; Windows: 100 ns ticks since 1601
};

// follows symlinks (HF snapshot files are symlinks into blobs)
ck_stat ck_stat_path(const std::string & path) {
    ck_stat r;
#ifdef _WIN32
    std::wstring w;
    if (!ck_widen(path, w)) {
        return r;
    }
    HANDLE h = CreateFileW(w.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                           OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
    if (h == INVALID_HANDLE_VALUE) {
        return r;
    }
    BY_HANDLE_FILE_INFORMATION info;
    const BOOL ok = GetFileInformationByHandle(h, &info);
    CloseHandle(h);
    if (!ok) {
        return r;
    }
    r.exists  = true;
    r.is_dir  = (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
    r.is_file = !r.is_dir && (info.dwFileAttributes & FILE_ATTRIBUTE_DEVICE) == 0;
    r.size    = ((uint64_t) info.nFileSizeHigh << 32) | info.nFileSizeLow;
    r.mtime   = (int64_t) ((((uint64_t) info.ftLastWriteTime.dwHighDateTime << 32) | info.ftLastWriteTime.dwLowDateTime) & 0x7FFFFFFFFFFFFFFFull);
#else
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        return r;
    }
    r.exists  = true;
    r.is_dir  = S_ISDIR(st.st_mode);
    r.is_file = S_ISREG(st.st_mode);
    r.size    = st.st_size > 0 ? (uint64_t) st.st_size : 0;
#    if defined(__APPLE__)
    r.mtime   = (int64_t) st.st_mtimespec.tv_sec * 1000000000 + (int64_t) st.st_mtimespec.tv_nsec;
#    else
    r.mtime   = (int64_t) st.st_mtim.tv_sec * 1000000000 + (int64_t) st.st_mtim.tv_nsec;
#    endif
#endif
    return r;
}

// symlink target as stored, "" when path is not a symlink (Windows: always "")
std::string ck_link_target(const std::string & path) {
#ifdef _WIN32
    (void) path;
    return "";
#else
    struct stat st;
    if (lstat(path.c_str(), &st) != 0 || !S_ISLNK(st.st_mode)) {
        return "";
    }
    char buf[4096];
    const ssize_t n = readlink(path.c_str(), buf, sizeof(buf));
    if (n <= 0 || (size_t) n >= sizeof(buf)) {
        return "?";
    }
    return std::string(buf, (size_t) n);
#endif
}

bool ck_list(const std::string & dir, std::vector<std::string> & names) {
    names.clear();
#ifdef _WIN32
    std::wstring w;
    if (!ck_widen(dir, w)) {
        return false;
    }
    WIN32_FIND_DATAW fd;
    HANDLE h = FindFirstFileW((w + L"\\*").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) {
        return false;
    }
    bool ok = true;
    do {
        std::string name;
        if (!ck_narrow(fd.cFileName, name)) {
            ok = false;
            break;
        }
        if (name != "." && name != "..") {
            names.push_back(name);
        }
    } while (FindNextFileW(h, &fd));
    FindClose(h);
    return ok;
#else
    DIR * d = opendir(dir.c_str());
    if (!d) {
        return false;
    }
    while (struct dirent * e = readdir(d)) {
        const std::string name = e->d_name;
        if (name != "." && name != "..") {
            names.push_back(name);
        }
    }
    closedir(d);
    return true;
#endif
}

bool ck_is_sep(char c) {
#ifdef _WIN32
    return c == '/' || c == '\\';
#else
    return c == '/';
#endif
}

// native separator: paths built here are also shown to the user (/props, logs)
std::string ck_join(const std::string & dir, const std::string & name) {
    if (dir.empty()) {
        return name;
    }
#ifdef _WIN32
    return ck_is_sep(dir.back()) ? dir + name : dir + "\\" + name;
#else
    return ck_is_sep(dir.back()) ? dir + name : dir + "/" + name;
#endif
}

bool ck_mkdir_one(const std::string & path) {
#ifdef _WIN32
    std::wstring w;
    if (!ck_widen(path, w)) {
        return false;
    }
    if (_wmkdir(w.c_str()) == 0) {
        return true;
    }
#else
    if (mkdir(path.c_str(), 0755) == 0) {
        return true;
    }
#endif
    return ck_stat_path(path).is_dir; // exists already (or made by a concurrent process)
}

// mkdir -p
bool ck_mkdirs(const std::string & path) {
    if (path.empty()) {
        return false;
    }
    if (ck_stat_path(path).is_dir) {
        return true;
    }
    for (size_t i = 1; i <= path.size(); ++i) {
        if (i == path.size() || ck_is_sep(path[i])) {
            const std::string prefix = path.substr(0, i);
#ifdef _WIN32
            if (prefix.size() == 2 && prefix[1] == ':') {
                continue; // drive
            }
#endif
            if (ck_is_sep(prefix.back()) || ck_stat_path(prefix).is_dir) {
                continue;
            }
            if (!ck_mkdir_one(prefix)) {
                return false;
            }
        }
    }
    return ck_stat_path(path).is_dir;
}

// removes dir when it is empty (rmdir fails on a non-empty directory, which is fine)
void ck_rmdir_if_empty(const std::string & dir) {
#ifdef _WIN32
    std::wstring w;
    if (ck_widen(dir, w)) {
        _wrmdir(w.c_str());
    }
#else
    rmdir(dir.c_str());
#endif
}

// absolute path with symlinks resolved (Windows: absolute only); "" on failure
std::string ck_realpath(const std::string & path) {
#ifdef _WIN32
    std::wstring w;
    if (!ck_widen(path, w)) {
        return "";
    }
    wchar_t * full = _wfullpath(nullptr, w.c_str(), 0);
    if (!full) {
        return "";
    }
    std::string out;
    const bool ok = ck_narrow(full, out);
    free(full);
    return ok ? out : "";
#else
    char * full = realpath(path.c_str(), nullptr);
    if (!full) {
        return "";
    }
    std::string out = full;
    free(full);
    return out;
#endif
}

// true when `inner` is `outer` or below it (both absolute, resolved)
bool ck_is_within(const std::string & inner, const std::string & outer) {
    if (outer.empty() || inner.size() < outer.size()) {
        return false;
    }
#ifdef _WIN32
    auto eq = [](char a, char b) {
        const char la = (a == '\\') ? '/' : (char) tolower((unsigned char) a);
        const char lb = (b == '\\') ? '/' : (char) tolower((unsigned char) b);
        return la == lb;
    };
    for (size_t i = 0; i < outer.size(); ++i) {
        if (!eq(inner[i], outer[i])) {
            return false;
        }
    }
#else
    if (inner.compare(0, outer.size(), outer) != 0) {
        return false;
    }
#endif
    return inner.size() == outer.size() || ck_is_sep(inner[outer.size()]) || ck_is_sep(outer.back());
}

std::string ck_hex(const unsigned char * d, size_t n) {
    static const char * digits = "0123456789abcdef";
    std::string out(2 * n, '0');
    for (size_t i = 0; i < n; ++i) {
        out[2 * i]     = digits[d[i] >> 4];
        out[2 * i + 1] = digits[d[i] & 15];
    }
    return out;
}

// sha256 of a file of `size` bytes; false when it cannot be read or its size changed
bool ck_hash_file(const std::string & path, uint64_t size, std::string & hex) {
    std::unique_ptr<FILE, int (*)(FILE *)> f(laya_convert_fopen(path, "rb"), fclose);
    if (!f) {
        return false;
    }
    sha256_t ctx;
    sha256_init(&ctx);
    std::vector<unsigned char> buf(1 << 20);
    uint64_t total = 0;
    size_t n;
    while ((n = fread(buf.data(), 1, buf.size(), f.get())) > 0) {
        sha256_update(&ctx, buf.data(), n);
        total += n;
        if (total > size) {
            return false;
        }
    }
    if (ferror(f.get()) || total != size) {
        return false;
    }
    unsigned char digest[SHA256_DIGEST_SIZE];
    sha256_final(&ctx, digest);
    hex = ck_hex(digest, SHA256_DIGEST_SIZE);
    return true;
}

struct ck_file {
    std::string rel;   // '/'-separated
    std::string path;
    ck_stat     st;
};

// What the converter can read (tools/laya/laya-convert.h): files directly in the checkpoint root
// (model*.safetensors, the index, rl_agent_config.json, README.md, generation_config.json, and the
// files whose presence makes it refuse, such as config.json or modules.json), in encoder/ and in
// tokenizer/. Those are keyed; any other subdirectory only by its name (its presence), because the
// converter never reads below it (the laya repo keeps 1.5 GB of other checkpoints and eval data there).
// Dot-entries are skipped (.git, .cache/huggingface): the converter never reads them.
bool ck_walk_dir(const std::string & root, const std::string & rel, bool descend, std::vector<ck_file> & out,
                 std::vector<std::string> & dirs, std::string & err) {
    const std::string dir = rel.empty() ? root : ck_join(root, rel);
    std::vector<std::string> names;
    if (!ck_list(dir, names)) {
        err = "cannot list directory '" + dir + "'";
        return false;
    }
    std::sort(names.begin(), names.end());
    for (const std::string & name : names) {
        if (name.empty() || name[0] == '.') {
            continue;
        }
        if (out.size() + dirs.size() >= CK_MAX_FILES) {
            err = "more than " + std::to_string(CK_MAX_FILES) + " entries in '" + dir + "'";
            return false;
        }
        const std::string r    = rel.empty() ? name : rel + "/" + name;
        const std::string path = ck_join(root, rel.empty() ? name : ck_join(rel, name));
        const ck_stat st = ck_stat_path(path);
        if (st.is_dir) {
            if (descend && (name == "encoder" || name == "tokenizer")) {
                if (!ck_walk_dir(root, r, false, out, dirs, err)) {
                    return false;
                }
            } else {
                dirs.push_back(r);
            }
        } else if (st.is_file) {
            out.push_back({ r, path, st });
        }
        // dangling symlinks and special files: skipped (the converter cannot read them either)
    }
    return true;
}

// realpath of path, or, when it does not exist yet, the realpath of its deepest existing ancestor
// joined with the rest; "" on failure or when the missing rest has "." / ".." components
std::string ck_resolve_prefix(const std::string & path) {
    std::string head = path;
    std::string tail;
    while (true) {
        const std::string r = ck_realpath(head.empty() ? std::string(".") : head);
        if (!r.empty()) {
            return tail.empty() ? r : ck_join(r, tail);
        }
        size_t end = head.size();
        while (end > 0 && ck_is_sep(head[end - 1])) {
            end--;
        }
        if (end == 0) {
            return ""; // the root itself does not resolve
        }
        size_t start = end;
        while (start > 0 && !ck_is_sep(head[start - 1])) {
            start--;
        }
        const std::string comp = head.substr(start, end - start);
        if (comp == "." || comp == "..") {
            return "";
        }
#ifdef _WIN32
        if (start == 0 && comp.size() == 2 && comp[1] == ':') {
            return ""; // a drive that does not resolve
        }
#endif
        tail = tail.empty() ? comp : ck_join(comp, tail);
        head = head.substr(0, start);
    }
}

// now, in the unit of ck_stat::mtime
int64_t ck_now_mtime() {
#ifdef _WIN32
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    return (int64_t) ((((uint64_t) ft.dwHighDateTime << 32) | ft.dwLowDateTime) & 0x7FFFFFFFFFFFFFFFull);
#else
    return (int64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
#endif
}

int64_t ck_mtime_per_second() {
#ifdef _WIN32
    return 10000000; // 100 ns ticks
#else
    return 1000000000;
#endif
}

// removes <stem>.<hex>.part and <stem>.<hex>.part.<hex>.tmp files older than CK_STALE_SECONDS in dir:
// what a killed or crashed conversion leaves behind (a running one writes its file continuously)
void ck_remove_stale_parts(const std::string & dir, const std::string & stem, const decision_checkpoint_params & params) {
    std::vector<std::string> names;
    if (!ck_list(dir, names)) {
        return;
    }
    const int64_t now = ck_now_mtime();
    for (const std::string & n : names) {
        const bool ours = n.size() > stem.size() + 1 && n.compare(0, stem.size() + 1, stem + ".") == 0 &&
                          (n.size() >= 5 && (n.compare(n.size() - 5, 5, ".part") == 0 || n.compare(n.size() - 4, 4, ".tmp") == 0));
        if (!ours) {
            continue;
        }
        const std::string path = ck_join(dir, n);
        const ck_stat st = ck_stat_path(path);
        if (st.is_file && now - st.mtime > CK_STALE_SECONDS * ck_mtime_per_second()) {
            if (laya_convert_remove(path)) {
                if (params.log) {
                    params.log("removed a stale temporary file of an interrupted conversion: " + path);
                }
            }
        }
    }
}

// the rename into the key directory survives a power loss (POSIX; NTFS journals the rename itself)
void ck_sync_dir(const std::string & dir) {
#ifdef _WIN32
    (void) dir;
#else
    const int fd = open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (fd >= 0) {
        fsync(fd);
        close(fd);
    }
#endif
}

// a cache path someone else could have put there: owned by another user, or writable by others
// (a shared LLAMA_CACHE). POSIX only; "" when it is ours.
std::string ck_foreign(const std::string & path) {
#ifdef _WIN32
    (void) path;
    return "";
#else
    struct stat st;
    if (lstat(path.c_str(), &st) != 0) {
        return "";
    }
    if (st.st_uid != geteuid()) {
        return "'" + path + "' is owned by another user";
    }
    if ((st.st_mode & S_IWOTH) != 0 && !S_ISLNK(st.st_mode)) {
        return "'" + path + "' is writable by other users";
    }
    if (S_ISLNK(st.st_mode)) {
        return "'" + path + "' is a symlink";
    }
    return "";
#endif
}

bool ck_portable_name(const std::string & n) {
    if (n.empty() || n.size() > 200 || n == "." || n == ".." || n.back() == '.' || n.back() == ' ') {
        return false;
    }
    for (unsigned char c : n) {
        if (c < 0x20 || c == 0x7F || strchr("<>:\"/\\|?*", (int) c) != nullptr) {
            return false;
        }
    }
    // Windows device names, with or without an extension
    std::string stem = n.substr(0, n.find('.'));
    std::transform(stem.begin(), stem.end(), stem.begin(), [](unsigned char c) { return (char) tolower(c); });
    static const char * reserved[] = { "con", "prn", "aux", "nul",
        "com1", "com2", "com3", "com4", "com5", "com6", "com7", "com8", "com9",
        "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9" };
    for (const char * r : reserved) {
        if (stem == r) {
            return false;
        }
    }
    return true;
}

// a readable laya GGUF whose tensor data fits in the file
bool ck_gguf_usable(const std::string & path, uint64_t file_size, std::string & why) {
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    std::unique_ptr<gguf_context, decltype(&gguf_free)> g(gguf_init_from_file(path.c_str(), gp), gguf_free);
    if (!g) {
        why = "not a readable GGUF";
        return false;
    }
    const int64_t id = gguf_find_key(g.get(), "general.architecture");
    if (id < 0 || gguf_get_kv_type(g.get(), id) != GGUF_TYPE_STRING || strcmp(gguf_get_val_str(g.get(), id), "laya") != 0) {
        why = "not a laya GGUF";
        return false;
    }
    const uint64_t base = gguf_get_data_offset(g.get());
    uint64_t end = base;
    for (int64_t i = 0; i < gguf_get_n_tensors(g.get()); ++i) {
        const uint64_t off = gguf_get_tensor_offset(g.get(), i);
        const uint64_t sz  = gguf_get_tensor_size(g.get(), i);
        if (off > UINT64_MAX - base || sz > UINT64_MAX - base - off) {
            why = "tensor offsets overflow";
            return false;
        }
        end = std::max(end, base + off + sz);
    }
    if (end > file_size) {
        why = "truncated (" + std::to_string(file_size) + " of " + std::to_string(end) + " bytes)";
        return false;
    }
    return true;
}

std::string ck_random_hex() {
    std::random_device rd;
    const uint64_t t = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    const uint64_t x = (((uint64_t) rd() << 32) ^ (uint64_t) rd()) ^ (t * 0x9E3779B97F4A7C15ull);
    char buf[17];
    snprintf(buf, sizeof(buf), "%016llx", (unsigned long long) x);
    return buf;
}

void ck_log(const decision_checkpoint_params & params, const std::string & msg) {
    if (params.log) {
        params.log(msg);
    }
}

#ifdef _WIN32
std::string ck_getenv(const wchar_t * name) {
    const wchar_t * v = _wgetenv(name);
    std::string out;
    if (!v || !*v || !ck_narrow(v, out)) {
        return "";
    }
    return out;
}
#else
std::string ck_getenv(const char * name) {
    const char * v = std::getenv(name);
    return v && *v ? std::string(v) : std::string();
}
#endif

} // namespace

bool decision_path_is_dir(const std::string & path) {
    return !path.empty() && ck_stat_path(path).is_dir;
}

bool decision_is_laya_checkpoint_dir(const std::string & dir) {
    return decision_path_is_dir(dir) && ck_stat_path(ck_join(dir, "rl_agent_config.json")).is_file;
}

std::string decision_checkpoint_name(const std::string & dir) {
    const std::string n = laya_convert_path_name(dir);
    return ck_portable_name(n) ? n : std::string("checkpoint");
}

std::string decision_checkpoint_default_cache() {
    // the layout of common's fs_get_cache_directory (what -mu URL and docker downloads use; -hf uses the
    // Hugging Face hub cache instead): $LLAMA_CACHE replaces <user cache>/llama.cpp
#ifdef _WIN32
    std::string base = ck_getenv(L"LLAMA_CACHE");
    if (base.empty()) {
        const std::string local = ck_getenv(L"LOCALAPPDATA");
        if (local.empty()) {
            return "";
        }
        base = ck_join(local, "llama.cpp");
    }
#else
    std::string base = ck_getenv("LLAMA_CACHE");
    if (base.empty()) {
        const std::string home = ck_getenv("HOME");
#    if defined(__APPLE__)
        if (home.empty()) {
            return "";
        }
        base = ck_join(home, "Library/Caches/llama.cpp");
#    else
        std::string xdg = ck_getenv("XDG_CACHE_HOME");
        if (xdg.empty()) {
            if (home.empty()) {
                return "";
            }
            xdg = ck_join(home, ".cache");
        }
        base = ck_join(xdg, "llama.cpp");
#    endif
    }
#endif
    return ck_join(ck_join(base, "laya"), "gguf-cache");
}

std::string decision_checkpoint_key(const std::string & dir, const decision_checkpoint_params & params, std::string & err) {
    std::vector<ck_file> files;
    std::vector<std::string> dirs;
    if (!ck_walk_dir(dir, "", true, files, dirs, err)) {
        return "";
    }
    // length-prefixed fields: no name can be confused with a separator
    auto field = [](const std::string & s) { return std::to_string(s.size()) + ":" + s; };
    std::string m;
    m += std::string(CK_FORMAT) + "\n";
    m += "converter " + std::to_string(LAYA_CONVERT_VERSION) + "\n";
    m += std::string("outtype ") + laya_convert_outtype_name(params.outtype) + "\n";
    m += "name " + field(laya_convert_path_name(dir)) + "\n";
    for (const ck_file & f : files) {
        m += "file " + field(f.rel) + " size " + std::to_string(f.st.size);
        if (f.st.size <= CK_HASH_MAX_BYTES) {
            std::string hex;
            if (!ck_hash_file(f.path, f.st.size, hex)) {
                err = "cannot read '" + f.path + "' (or it changed while reading)";
                return "";
            }
            m += " sha256 " + hex;
        } else {
            m += " mtime " + std::to_string(f.st.mtime) + " link " + field(ck_link_target(f.path));
        }
        m += "\n";
    }
    for (const std::string & d : dirs) {
        m += "dir " + field(d) + "\n";
    }
    return decision_sha256_hex(m).substr(0, CK_KEY_HEX);
}

bool decision_model_source_resolve(const std::string & path, const decision_checkpoint_params & params,
                                   decision_model_source & out, std::string & err) {
    out       = decision_model_source();
    out.input = path;
    if (!decision_path_is_dir(path)) {
        out.kind      = "gguf";
        out.gguf_path = path;
        return true;
    }
    out.kind    = "checkpoint-dir";
    out.outtype = laya_convert_outtype_name(params.outtype);

    if (!decision_is_laya_checkpoint_dir(path)) {
        err = "'" + path + "' is a directory but not a laya checkpoint (no rl_agent_config.json; a checkpoint has "
              "rl_agent_config.json, encoder/config.json, tokenizer/ and model*.safetensors)";
        return false;
    }
    if (params.outtype != LAYA_CONVERT_F16 && params.outtype != LAYA_CONVERT_F32) {
        // convert_hf_to_gguf.py's q8_0 also quantizes token_embd, type_emb, scorer.* and act_head.*, which the
        // served Q8_0 recipe (tests/laya/quantize.sh) keeps at F16; its accuracy is not measured
        err = std::string("-m DIR converts to f16 or f32, not ") + out.outtype + " (for Q8_0 quantize the f16 GGUF with tests/laya/quantize.sh)";
        return false;
    }

    const std::string cache = params.cache_dir.empty() ? decision_checkpoint_default_cache() : params.cache_dir;
    if (cache.empty()) {
        err = "no user cache directory for converted checkpoints; pass --decision-convert-cache DIR";
        return false;
    }
    // the cache must not live in the checkpoint: it would change the checkpoint (and the key) it caches.
    // Checked before anything is created (on the deepest existing ancestor), and again after (symlinks).
    const std::string real_dir = ck_realpath(path);
    if (real_dir.empty()) {
        err = "cannot resolve '" + path + "'";
        return false;
    }
    const std::string pre_cache = ck_resolve_prefix(cache);
    if (pre_cache.empty()) {
        err = "cannot resolve the conversion cache directory '" + cache + "'";
        return false;
    }
    if (ck_is_within(pre_cache, real_dir)) {
        err = "the conversion cache '" + cache + "' is inside the checkpoint directory '" + path + "'; pass another --decision-convert-cache";
        return false;
    }
    if (!ck_mkdirs(cache)) {
        err = "cannot create the conversion cache directory '" + cache + "'";
        return false;
    }
    const std::string real_cache = ck_realpath(cache);
    if (real_cache.empty() || ck_is_within(real_cache, real_dir)) {
        err = real_cache.empty() ? "cannot resolve '" + cache + "'"
                                 : "the conversion cache '" + cache + "' is inside the checkpoint directory '" + path + "'; pass another --decision-convert-cache";
        return false;
    }
    out.cache_dir = cache;

    std::string kerr;
    out.key = decision_checkpoint_key(path, params, kerr);
    if (out.key.empty()) {
        err = "cannot read checkpoint directory '" + path + "': " + kerr;
        return false;
    }
    const std::string key_dir = ck_join(cache, out.key);
    const std::string stem    = decision_checkpoint_name(path) + ".gguf";
    const std::string final   = ck_join(key_dir, stem);
    out.gguf_path = final;

    if (ck_stat_path(key_dir).exists) {
        const std::string foreign = ck_foreign(key_dir);
        if (!foreign.empty()) {
            err = "refusing the conversion cache entry: " + foreign + "; pass another --decision-convert-cache";
            return false;
        }
    }
    const ck_stat fst = ck_stat_path(final);
    if (fst.is_file) {
        const std::string foreign = ck_foreign(final);
        if (!foreign.empty()) {
            err = "refusing the conversion cache entry: " + foreign + "; pass another --decision-convert-cache";
            return false;
        }
        std::string why;
        if (ck_gguf_usable(final, fst.size, why)) {
            out.cache_hit = true;
            ck_log(params, "cache hit: " + final + " (checkpoint '" + path + "', " + out.outtype + ", key " + out.key + ")");
            return true;
        }
        ck_log(params, "cache entry '" + final + "' is not usable (" + why + "), converting again");
    }

    if (!ck_mkdirs(key_dir)) {
        err = "cannot create the conversion cache directory '" + key_dir + "'";
        return false;
    }
    const std::string foreign = ck_foreign(key_dir);
    if (!foreign.empty()) {
        err = "refusing the conversion cache entry: " + foreign + "; pass another --decision-convert-cache";
        return false;
    }
    ck_remove_stale_parts(key_dir, stem, params);
    // unique name: concurrent starts on the same checkpoint do not share a temporary file
    const std::string tmp = final + "." + ck_random_hex() + ".part";
    laya_convert_params cp;
    cp.outtype = params.outtype;
    cp.log = [&params](const std::string & msg) {
        if (msg.rfind("warning", 0) == 0) {
            ck_log(params, msg);
        }
    };
    ck_log(params, "converting checkpoint '" + path + "' to " + out.outtype + " GGUF (key " + out.key + ")");
    const auto t0 = std::chrono::steady_clock::now();
    const std::string cerr = laya_convert(path, tmp, cp);
    out.convert_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!cerr.empty()) {
        laya_convert_remove(tmp);
        ck_rmdir_if_empty(key_dir);
        err = "cannot convert checkpoint directory '" + path + "': " + cerr;
        return false;
    }

    // a checkpoint that changed while it was converted (a download in progress) is not cached under the old key
    std::string kerr2;
    const std::string key2 = decision_checkpoint_key(path, params, kerr2);
    if (key2 != out.key) {
        laya_convert_remove(tmp);
        ck_rmdir_if_empty(key_dir);
        err = "checkpoint directory '" + path + "' changed while it was converted; try again";
        return false;
    }

    if (!laya_convert_rename(tmp, final)) {
        // another process may have put the same bytes there first (Windows cannot replace a mapped file)
        const ck_stat st2 = ck_stat_path(final);
        std::string why;
        laya_convert_remove(tmp);
        if (!(st2.is_file && ck_gguf_usable(final, st2.size, why))) {
            ck_rmdir_if_empty(key_dir);
            err = "cannot move the converted GGUF into place at '" + final + "'";
            return false;
        }
    }
    ck_sync_dir(key_dir);
    char secs[32];
    snprintf(secs, sizeof(secs), "%.2f", out.convert_ms / 1000.0);
    ck_log(params, "converted: " + final + " in " + secs + " s (checkpoint '" + path + "', " + out.outtype + ", key " + out.key + ")");
    return true;
}
