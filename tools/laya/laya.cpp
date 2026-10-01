#include "laya.h"
#include "laya-unicode.h"

#include "ggml.h"
#include "ggml-cpp.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <filesystem>
#else
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <sys/stat.h>
#    include <unistd.h>
#endif

#if defined(__APPLE__)
#    include <pthread.h>
#    include <pthread/qos.h>
#    include <sys/sysctl.h>
#endif

// ======================================================================
// Reference (PyTorch): laya/common.py -> DecisionModel
//   h        = encoder(input_ids, attention_mask).last_hidden_state
//   h        = h + type_emb(qtype)[:, None, :]
//   for layer in head.layers: h = layer(h, src_key_padding_mask=pad)
//   m        = gather(h, marker_pos)
//   logits   = scorer(m).squeeze(-1); logits.masked_fill(~marker_mask, -1e4)
//   p        = softmax(logits.detach(), -1)
//   k        = marker_mask.sum(-1).clamp(min=2)
//   ent      = -(p * log(p)).sum(-1) / log(k)
//   top2     = p.topk(2, -1).values
//   feats    = [top2[:,0], top2[:,0]-top2[:,1], ent, k/255]
//   pooled   = h[:, 0]
//   act_logits = act_head(cat([pooled, feats], -1))
//
// The encoder is a ModernBERT (mmBERT) body: bidirectional attention with a
// per-layer dense/sliding window pattern, GeGLU FFN, RoPE (NEOX), norm_first
// residual blocks. The decision head is 2 norm_first TransformerEncoderLayers
// (full self-attention + ReLU FFN, like PyTorch nn.TransformerEncoderLayer),
// a scorer and an action head.
// ======================================================================

struct laya_encoder_layer {
    ggml_tensor * attn_norm = nullptr; // layer 0 is identity (may be null)
    ggml_tensor * wqkv      = nullptr;
    ggml_tensor * wo        = nullptr;
    ggml_tensor * ffn_up    = nullptr;
    ggml_tensor * ffn_down  = nullptr;
    ggml_tensor * ffn_norm  = nullptr;
};

struct laya_head_layer {
    ggml_tensor * attn_norm   = nullptr;
    ggml_tensor * attn_norm_b = nullptr;
    ggml_tensor * wqkv        = nullptr;
    ggml_tensor * wqkv_b      = nullptr;
    ggml_tensor * wo          = nullptr;
    ggml_tensor * wo_b        = nullptr;
    ggml_tensor * ffn_norm    = nullptr;
    ggml_tensor * ffn_norm_b  = nullptr;
    ggml_tensor * ffn_up      = nullptr;
    ggml_tensor * ffn_up_b    = nullptr;
    ggml_tensor * ffn_down    = nullptr;
    ggml_tensor * ffn_down_b  = nullptr;
};

// BPE merges as an open-addressing table: long words need millions of lookups,
// and 580k merges fit in 16 MB (std::unordered_map nodes cost several times that)
struct laya_merge_map {
    struct slot {
        uint64_t key  = UINT64_MAX;  // left id << 32 | right id; UINT64_MAX = empty
        uint32_t rank = 0;
        int32_t  id   = 0;           // id of the merged token
    };
    std::vector<slot> slots;
    uint64_t          mask = 0;

    static uint64_t key_of(int32_t a, int32_t b) {
        return ((uint64_t) (uint32_t) a << 32) | (uint32_t) b;
    }
    static uint64_t hash(uint64_t k) {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33;
        return k;
    }
    void init(size_t n) {
        size_t cap = 16;
        while (cap * 7 < n * 10) {
            cap <<= 1;
        }
        slots.assign(cap, slot());
        mask = cap - 1;
    }
    // a later duplicate wins, as when HF collects the merges into a map
    void set(uint64_t k, uint32_t rank, int32_t id) {
        uint64_t i = hash(k) & mask;
        while (slots[i].key != UINT64_MAX && slots[i].key != k) {
            i = (i + 1) & mask;
        }
        slots[i] = { k, rank, id };
    }
    const slot * find(int32_t a, int32_t b) const {
        if (slots.empty()) {
            return nullptr;
        }
        const uint64_t k = key_of(a, b);
        for (uint64_t i = hash(k) & mask; slots[i].key != UINT64_MAX; i = (i + 1) & mask) {
            if (slots[i].key == k) {
                return &slots[i];
            }
        }
        return nullptr;
    }
};

// read-only mapping of one byte range of a file (POSIX mmap / Win32 MapViewOfFile)
struct laya_mmap {
    void * base   = nullptr;  // start of the mapping (granularity aligned)
    size_t size   = 0;        // bytes mapped from base
    size_t skip   = 0;        // requested offset - mapped offset
    bool   locked = false;
#if defined(_WIN32)
    HANDLE hfile = INVALID_HANDLE_VALUE;
    HANDLE hmap  = nullptr;
#endif

    laya_mmap() = default;
    laya_mmap(const laya_mmap &) = delete;
    laya_mmap & operator=(const laya_mmap &) = delete;

    void * data() const { return base ? (char *) base + skip : nullptr; }

    // map [offset, offset + len); false (and nothing mapped) on any failure
    bool map(const char * fname, size_t offset, size_t len) {
#if defined(_WIN32)
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        const size_t gran = si.dwAllocationGranularity;
        const size_t off0 = offset / gran * gran;
        const std::wstring wname = laya_utf8_to_wide(fname);
        if (wname.empty()) {
            return false;
        }
        hfile = CreateFileW(wname.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (hfile == INVALID_HANDLE_VALUE) {
            return false;
        }
        LARGE_INTEGER fsize;
        if (!GetFileSizeEx(hfile, &fsize) || (uint64_t) fsize.QuadPart < (uint64_t) (offset + len)) {
            unmap();
            return false;
        }
        hmap = CreateFileMappingW(hfile, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!hmap) {
            unmap();
            return false;
        }
        base = MapViewOfFile(hmap, FILE_MAP_READ, (DWORD) ((uint64_t) off0 >> 32), (DWORD) (off0 & 0xffffffffu), offset + len - off0);
        if (!base) {
            unmap();
            return false;
        }
        size = offset + len - off0;
        skip = offset - off0;
        return true;
#else
        const long   page = sysconf(_SC_PAGESIZE);
        const size_t gran = page > 0 ? (size_t) page : 4096;
        const size_t off0 = offset / gran * gran;
        const int fd = open(fname, O_RDONLY);
        if (fd < 0) {
            return false;
        }
        // a mapping past the end of the file faults (SIGBUS) on access
        struct stat st;
        if (fstat(fd, &st) != 0 || (uint64_t) st.st_size < (uint64_t) (offset + len)) {
            close(fd);
            return false;
        }
        void * addr = mmap(nullptr, offset + len - off0, PROT_READ, MAP_SHARED, fd, (off_t) off0);
        close(fd);
        if (addr == MAP_FAILED) {
            return false;
        }
        base = addr;
        size = offset + len - off0;
        skip = offset - off0;
        // get_rows reads scattered rows: no read-ahead around each fault
        posix_madvise(base, size, POSIX_MADV_RANDOM);
        return true;
#endif
    }

    void unmap() {
#if defined(_WIN32)
        if (base) {
            if (locked) {
                VirtualUnlock(base, size);
            }
            UnmapViewOfFile(base);
        }
        if (hmap) {
            CloseHandle(hmap);
        }
        if (hfile != INVALID_HANDLE_VALUE) {
            CloseHandle(hfile);
        }
        hmap  = nullptr;
        hfile = INVALID_HANDLE_VALUE;
#else
        if (base) {
            if (locked) {
                munlock(base, size);
            }
            munmap(base, size);
        }
#endif
        base   = nullptr;
        size   = 0;
        skip   = 0;
        locked = false;
    }

    ~laya_mmap() { unmap(); }
};

// lock [addr, addr + size) in RAM; false on failure (limits, permissions)
static bool laya_mlock(void * addr, size_t size) {
    if (!addr || size == 0) {
        return true;
    }
#if defined(_WIN32)
    if (VirtualLock(addr, size)) {
        return true;
    }
    // the working set minimum bounds what can be locked: grow it by this range and retry
    SIZE_T wmin = 0, wmax = 0;
    if (!GetProcessWorkingSetSize(GetCurrentProcess(), &wmin, &wmax)) {
        return false;
    }
    wmin += size;
    wmax = std::max(wmax, wmin);
    if (!SetProcessWorkingSetSize(GetCurrentProcess(), wmin, wmax)) {
        return false;
    }
    return VirtualLock(addr, size) != 0;
#else
    return mlock(addr, size) == 0;
#endif
}

struct laya_model {
    laya_hparams hparams;

    ggml_context_ptr ctx_meta;   // tensor definitions from the gguf header (freed after the load)
    gguf_context_ptr ctx_gguf;   // (freed after the load)

    // data contexts holding the real tensors, one per buffer: loaded (default CPU buffer),
    // repacked (one per extra buffer type), mapped (token_embd in the file mapping), device (the
    // buffer type of the compute device)
    laya_mmap                            mapping;   // token_embd, when mapped (outlives bufs)
    std::vector<ggml_context_ptr>        ctxs;
    std::vector<ggml_backend_buffer_ptr> bufs;

    ggml_backend_t         backend = nullptr;
    ggml_backend_dev_t     dev_gpu = nullptr;   // compute device of laya_model_params.device; nullptr = CPU

    // what the load did, for the log and laya_context_kernels
    size_t  n_bytes_device   = 0;
    size_t  n_bytes_loaded   = 0;
    size_t  n_bytes_mapped   = 0;
    size_t  n_bytes_repacked = 0;
    int32_t n_repacked       = 0;
    int32_t n_host_weights   = 0;   // weights of a device model kept in host memory (laya_placement.host_weights)

    // self-contained tokenizer data (HF tokenizers semantics)
    bool bytelevel = false;                          // decision.laya.tokenizer: bytelevel-bpe, else metaspace-bpe
    bool nfc       = false;                          // bytelevel: decision.laya.normalizer "nfc"
    std::vector<int32_t>                  cpt_bmp;   // metaspace: single-codepoint tokens below U+10000, -1 if none
    std::unordered_map<uint32_t, int32_t> cpt_high;  // metaspace: single-codepoint tokens from U+10000
    laya_merge_map                        merges;    // (left id, right id) -> rank, merged id
    int32_t byte_id[256];                            // metaspace: <0xXX> byte-fallback tokens, -1 if missing
    int32_t byte_sym[256];                           // bytelevel: token of the byte-to-unicode char of each byte

    // added tokens (HF fast-tokenizer AddedVocabulary), each match breaking the word boundary.
    // metaspace: all matched on the raw text; bytelevel: see added_trie_norm.
    // Identified from tokenizer.ggml.token_type (CONTROL / USER_DEFINED).
    struct added_token {
        std::string s;
        int32_t     id         = 0;
        bool        lstrip     = false; // consume preceding whitespace on match
        bool        control    = false; // CONTROL type (special in HF)
        bool        normalized = false; // bytelevel: matched on the normalized text (HF normalized=True)
    };
    std::vector<added_token> added_tokens;

    // byte tries over added_tokens for leftmost-longest matching. metaspace: every added token in
    // added_trie; bytelevel: added_trie has the tokens matched on the raw text, added_trie_norm the rest
    struct trie_node {
        std::vector<std::pair<uint8_t, int32_t>> next; // byte -> node
        int32_t token = -1;                            // index into added_tokens
    };
    std::vector<trie_node> added_trie;
    std::vector<trie_node> added_trie_norm;
    int32_t bos_id  = 2;
    int32_t eos_id  = 1;
    int32_t sep_id  = 1;
    int32_t mask_id = 4;
    int32_t unk_id  = 3;
    int32_t pad_id  = 0;

    ggml_tensor * tok_embd   = nullptr;
    ggml_tensor * tok_norm   = nullptr;
    ggml_tensor * output_norm = nullptr;

    std::vector<laya_encoder_layer> layers;
    ggml_tensor * type_emb = nullptr;
    std::vector<laya_head_layer> head_layers;

    ggml_tensor * scorer_0   = nullptr; ggml_tensor * scorer_0_b = nullptr;
    ggml_tensor * scorer_1   = nullptr; ggml_tensor * scorer_1_b = nullptr;
    ggml_tensor * scorer_3   = nullptr; ggml_tensor * scorer_3_b = nullptr;
    ggml_tensor * act_head_0 = nullptr; ggml_tensor * act_head_0_b = nullptr;
    ggml_tensor * act_head_2 = nullptr; ggml_tensor * act_head_2_b = nullptr;
};

typedef ggml_threadpool_t (*laya_threadpool_new_t)(ggml_threadpool_params * params);
typedef void (*laya_threadpool_free_t)(ggml_threadpool_t threadpool);
typedef void (*laya_set_threadpool_t)(ggml_backend_t backend, ggml_threadpool_t threadpool);

struct laya_context {
    const laya_model * model = nullptr;

    ggml_backend_t         backend      = nullptr;
    ggml_backend_t         backend_cpu  = nullptr;
    ggml_backend_t         backend_blas = nullptr;
    ggml_backend_t         backend_gpu  = nullptr;   // the model's compute device, ahead of the CPU in the scheduler
    ggml_backend_sched_ptr sched;

    laya_placement placement;   // of the load-time probe graph (laya_placement_probe)

    // persistent threadpool: without OpenMP ggml otherwise creates and joins the worker
    // threads on every graph compute
    ggml_threadpool_t      threadpool      = nullptr;
    laya_threadpool_free_t threadpool_free = nullptr;

    int  n_threads      = 1;
    int  n_threads_blas = 0;
    bool qos            = true;

    laya_precision precision = LAYA_PRECISION_DEFAULT;

    // layer trace (laya_context_params.trace_dir): trace.jsonl + one .f32 file per dumped node
    std::string trace_dir;
    FILE *      trace_index = nullptr;
    int64_t     trace_call  = -1;  // laya_encode calls of this context, from 0
    int32_t     trace_seq   = 0;   // dumps within the current call
};

// fills ctx->placement from the warm-up graph (defined after laya_graph_build)
static void laya_placement_probe(laya_context * ctx);

static void laya_log(const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "laya: ");
    vfprintf(stderr, fmt, args);
    va_end(args);
}

// ---- layer trace ----------------------------------------------------------
// laya_context_params.trace_dir (an existing directory; the tools take it from LAYA_TRACE_DIR) makes
// every laya_encode of a context dump the nodes at the ends
// of op chains to raw float32 files (tools/laya/trace_diff.py compares two such directories). Only
// these names are asked for: the scheduler splits the graph at every node the eval callback asks
// for, so a name inside a chain would break the fused kernels of Metal / CUDA (norm + mul + add,
// multi-add) and trace a different computation. Off (no callback) the scheduler takes its usual path.
static bool laya_trace_wanted(const ggml_tensor * t) {
    const char * n = t->name;
    return strncmp(n, "l_out-", 6) == 0 || strncmp(n, "head_out-", 9) == 0 || strcmp(n, "enc_out") == 0 ||
           strcmp(n, "type_emb_out") == 0 || strcmp(n, "markers") == 0 || strcmp(n, "logits") == 0 ||
           strcmp(n, "logits_masked") == 0 || strcmp(n, "act_logits") == 0;
}

static bool laya_trace_eval(ggml_tensor * t, bool ask, void * user_data) {
    laya_context * ctx = (laya_context *) user_data;
    if (ask) {
        return laya_trace_wanted(t);
    }
    if (!laya_trace_wanted(t) || !ctx->trace_index) {
        return true;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
        laya_log("trace: skipping %s (not a contiguous F32 tensor)\n", t->name);
        return true;
    }
    std::vector<float> data((size_t) ggml_nelements(t));
    ggml_backend_tensor_get(t, data.data(), 0, ggml_nbytes(t));
    char file[160];
    snprintf(file, sizeof(file), "%06lld-%03d-%s.f32", (long long) ctx->trace_call, ctx->trace_seq, t->name);
    const std::string path = ctx->trace_dir + "/" + file;
    FILE * f = ggml_fopen(path.c_str(), "wb"); // UTF-8 name, also on Windows
    if (!f || fwrite(data.data(), sizeof(float), data.size(), f) != data.size()) {
        laya_log("trace: cannot write %s, tracing stops\n", path.c_str());
        if (f) {
            fclose(f);
        }
        fclose(ctx->trace_index);
        ctx->trace_index = nullptr;
        return true;
    }
    fclose(f);
    const char * buft = t->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(t->buffer)) : "";
    fprintf(ctx->trace_index,
            "{\"call\": %lld, \"node\": %d, \"name\": \"%s\", \"op\": \"%s\", \"buffer\": \"%s\", \"type\": \"f32\", "
            "\"ne\": [%lld, %lld, %lld, %lld], \"file\": \"%s\"}\n",
            (long long) ctx->trace_call, ctx->trace_seq, t->name, ggml_op_name(t->op), buft,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3], file);
    fflush(ctx->trace_index);
    ctx->trace_seq++;
    return true;
}

// ---- GGUF helpers ------------------------------------------------------

static int64_t gguf_find_key_or(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        throw std::runtime_error(std::string("missing GGUF key: ") + key);
    }
    return id;
}

static uint32_t gguf_get_u32(const gguf_context * ctx, const char * key, uint32_t def = 0) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    return gguf_get_val_u32(ctx, id);
}

static int32_t gguf_get_i32(const gguf_context * ctx, const char * key, int32_t def = 0) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_INT32: return gguf_get_val_i32(ctx, id);
        case GGUF_TYPE_UINT32: return (int32_t) gguf_get_val_u32(ctx, id);
        default: throw std::runtime_error(std::string("unexpected type for GGUF key: ") + key);
    }
}

static float gguf_get_f32(const gguf_context * ctx, const char * key, float def = 0.0f) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    return gguf_get_val_f32(ctx, id);
}

static bool gguf_get_bool(const gguf_context * ctx, const char * key, bool def = false) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    return gguf_get_val_bool(ctx, id);
}

static std::vector<float> gguf_get_arr_f32(const gguf_context * ctx, const char * key) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return {};
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY ||
        gguf_get_arr_type(ctx, id) != GGUF_TYPE_FLOAT32) {
        return {};
    }
    const size_t n = gguf_get_arr_n(ctx, id);
    const float * data = (const float *) gguf_get_arr_data(ctx, id);
    return std::vector<float>(data, data + n);
}

static std::vector<int32_t> gguf_get_arr_i32(const gguf_context * ctx, const char * key, bool & found) {
    const int64_t id = gguf_find_key(ctx, key);
    found = id >= 0;
    if (!found) {
        return {};
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx, id) != GGUF_TYPE_INT32) {
        throw std::runtime_error(std::string("unexpected type for GGUF key: ") + key);
    }
    const int32_t * data = (const int32_t *) gguf_get_arr_data(ctx, id);
    return std::vector<int32_t>(data, data + gguf_get_arr_n(ctx, id));
}

static std::string gguf_get_str(const gguf_context * ctx, const char * key, const std::string & def) {
    const int64_t id = gguf_find_key(ctx, key);
    if (id < 0) {
        return def;
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_STRING) {
        throw std::runtime_error(std::string("unexpected type for GGUF key: ") + key);
    }
    return gguf_get_val_str(ctx, id);
}

// ---- model loading ------------------------------------------------------

// decode one UTF-8 codepoint; ok = false (len 1) on an invalid sequence
static uint32_t laya_cpt_from_utf8(const char * s, size_t n, size_t & len, bool & ok) {
    const uint8_t * u = (const uint8_t *) s;
    ok  = false;
    len = n == 0 ? 0 : 1;
    if (n == 0) {
        return 0;
    }
    if (u[0] < 0x80) {
        ok = true;
        return u[0];
    }
    size_t   need = 0;
    uint32_t cpt  = 0;
    if      ((u[0] & 0xE0) == 0xC0) { need = 2; cpt = u[0] & 0x1F; }
    else if ((u[0] & 0xF0) == 0xE0) { need = 3; cpt = u[0] & 0x0F; }
    else if ((u[0] & 0xF8) == 0xF0) { need = 4; cpt = u[0] & 0x07; }
    else {
        return 0xFFFD;
    }
    if (n < need) {
        return 0xFFFD;
    }
    for (size_t k = 1; k < need; ++k) {
        if ((u[k] & 0xC0) != 0x80) {
            return 0xFFFD;
        }
        cpt = (cpt << 6) | (u[k] & 0x3F);
    }
    len = need;
    ok  = true;
    return cpt;
}

laya_model * laya_model_load_from_file(const char * fname) {
    return laya_model_load_from_file_ext(fname, laya_model_params());
}

// A device model on a backend or with a weight type that no parity run covers (DECISION.md "Backend parity
// tiers": Metal, CUDA and Vulkan; F32, F16 and Q8_0 weights) still loads, with a warning: quantized types
// other than Q8_0 keep the device's own kernels (CUDA MMQ quantizes the activations, the failure mode Q8_0
// had before the laya graph dequantized it).
static void laya_warn_unverified(const laya_model * model) {
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(model->dev_gpu);
    const std::string backend = reg ? ggml_backend_reg_name(reg) : "?";
    if (backend != "MTL" && backend != "CUDA" && backend != "Vulkan") {
        laya_log("%s: warning: the %s backend has no laya parity run (gated: Metal, CUDA, Vulkan); run the f16-class gate "
                 "(DECISION.md, Backend parity tiers) before trusting its answers\n", __func__, backend.c_str());
    }
    std::vector<const ggml_tensor *> mm;
    for (const auto & l : model->layers) {
        mm.insert(mm.end(), { l.wqkv, l.wo, l.ffn_up, l.ffn_down });
    }
    for (const auto & l : model->head_layers) {
        mm.insert(mm.end(), { l.wqkv, l.wo, l.ffn_up, l.ffn_down });
    }
    mm.insert(mm.end(), { model->scorer_1, model->scorer_3, model->act_head_0, model->act_head_2 });
    std::set<std::string> ungated;
    for (const ggml_tensor * t : mm) {
        if (t && t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_Q8_0) {
            ungated.insert(ggml_type_name(t->type));
        }
    }
    for (const std::string & t : ungated) {
        laya_log("%s: warning: %s weights have no parity gate on a device (gated: f32, f16, q8_0); run the f16-class gate "
                 "for this GGUF before trusting its answers\n", __func__, t.c_str());
    }
}

laya_model * laya_model_load_from_file_ext(const char * fname, const laya_model_params & mparams) {
    std::unique_ptr<laya_model> model(new laya_model());

    struct ggml_context * meta = nullptr;
    struct gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ &meta,
    };

    model->ctx_gguf.reset(gguf_init_from_file(fname, params));
    if (!model->ctx_gguf.get()) {
        throw std::runtime_error(std::string("failed to load laya model from ") + fname + " (does the file exist?)");
    }
    model->ctx_meta.reset(meta);

    const gguf_context * ctx_gguf = model->ctx_gguf.get();

    // architecture check
    {
        const int64_t id = gguf_find_key(ctx_gguf, "general.architecture");
        if (id < 0 || std::string(gguf_get_val_str(ctx_gguf, id)) != "laya") {
            throw std::runtime_error("not a laya GGUF (general.architecture != \"laya\")");
        }
    }

    // keys without a safe default: a missing sliding window would make every layer dense, a missing
    // SWA base would take the global one (the English checkpoints use 160000 / 10000). The converter
    // writes all four (conversion/laya.py), so their absence means a foreign or broken GGUF.
    for (const char * key : { "laya.attention.sliding_window", "laya.attention.sliding_window_pattern",
                              "laya.rope.freq_base", "laya.rope.freq_base_swa" }) {
        if (gguf_find_key(ctx_gguf, key) < 0) {
            throw std::runtime_error(std::string("missing GGUF key: ") + key + " (reconvert with conversion/laya.py)");
        }
    }
    // the graph implements GeGLU with the erf GELU only
    {
        const std::string act = gguf_get_str(ctx_gguf, "laya.hidden_activation", "gelu");
        if (act != "gelu") {
            throw std::runtime_error("laya.hidden_activation '" + act + "' is not supported (gelu only)");
        }
    }

    auto & hp = model->hparams;
    hp.n_embd          = (int32_t) gguf_get_u32(ctx_gguf, "laya.embedding_length");
    hp.n_layer         = (int32_t) gguf_get_u32(ctx_gguf, "laya.block_count");
    hp.n_head          = (int32_t) gguf_get_u32(ctx_gguf, "laya.attention.head_count");
    hp.n_ff            = (int32_t) gguf_get_u32(ctx_gguf, "laya.feed_forward_length");
    hp.n_vocab         = (int32_t) gguf_get_u32(ctx_gguf, "laya.vocab_size");
    hp.n_qtype         = (int32_t) gguf_get_u32(ctx_gguf, "laya.n_qtype", 3);
    hp.marker_token_id = (int32_t) gguf_get_u32(ctx_gguf, "laya.marker_token_id");
    hp.max_len         = (int32_t) gguf_get_u32(ctx_gguf, "laya.max_len", 1024);
    hp.head_max_len    = (int32_t) gguf_get_u32(ctx_gguf, "laya.head_max_len", 256);
    hp.n_act           = (int32_t) gguf_get_u32(ctx_gguf, "laya.act_classes", 2);
    hp.n_head_layers   = (int32_t) gguf_get_u32(ctx_gguf, "laya.head_layers", 2);
    hp.n_swa           = (int32_t) gguf_get_u32(ctx_gguf, "laya.attention.sliding_window", 0);
    hp.swa_pattern     = (int32_t) gguf_get_u32(ctx_gguf, "laya.attention.sliding_window_pattern", 0);
    // the encoder uses RMSNorm (rms epsilon), the decision head / scorer use
    // LayerNorm (layer-norm epsilon); they can differ across checkpoints.
    hp.norm_eps        = gguf_get_f32(ctx_gguf, "laya.attention.layer_norm_rms_epsilon", 1e-5f);
    hp.norm_eps_layer  = gguf_get_f32(ctx_gguf, "laya.attention.layer_norm_epsilon", hp.norm_eps);
    hp.rope_freq_base    = gguf_get_f32(ctx_gguf, "laya.rope.freq_base", 10000.0f);
    hp.rope_freq_base_swa = gguf_get_f32(ctx_gguf, "laya.rope.freq_base_swa", hp.rope_freq_base);

    if (hp.n_embd == 0 || hp.n_layer == 0 || hp.n_head == 0) {
        throw std::runtime_error("invalid laya hparams (missing encoder dims)");
    }
    if (hp.n_embd % hp.n_head != 0) {
        throw std::runtime_error("n_embd not divisible by n_head");
    }
    if (hp.n_act < 1) {
        // act_probability reads class 0 of the act head
        throw std::runtime_error("invalid laya hparams (laya.act_classes must be >= 1)");
    }
    hp.n_embd_head = hp.n_embd / hp.n_head;

    // temperature array, padded to n_qtype with the last value (or 1.0)
    hp.temperature = gguf_get_arr_f32(ctx_gguf, "laya.temperature");
    if (hp.temperature.empty()) {
        hp.temperature.assign(std::max(1, hp.n_qtype), 1.0f);
    }
    while ((int32_t) hp.temperature.size() < hp.n_qtype) {
        hp.temperature.push_back(hp.temperature.back());
    }
    {
        const int64_t id = gguf_find_key(ctx_gguf, "laya.temperature_by_options.buckets");
        const std::vector<float> values = gguf_get_arr_f32(ctx_gguf, "laya.temperature_by_options.values");
        if (id >= 0) {
            if (gguf_get_kv_type(ctx_gguf, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(ctx_gguf, id) != GGUF_TYPE_STRING ||
                gguf_get_arr_n(ctx_gguf, id) != values.size()) {
                throw std::runtime_error("laya.temperature_by_options: buckets and values do not match");
            }
            for (size_t i = 0; i < values.size(); ++i) {
                hp.temperature_buckets.push_back(gguf_get_arr_str(ctx_gguf, id, i));
            }
            hp.temperature_bucket_values = values;
        }
    }

    // ---- tokenizer data ----
    int64_t n_vocab_tok = 0;  // tokenizer.ggml.tokens entries: every id the tokenizer can produce is below
    {
        const std::string kind = gguf_get_str(ctx_gguf, "decision.laya.tokenizer", "metaspace-bpe");
        if (kind != "metaspace-bpe" && kind != "bytelevel-bpe") {
            throw std::runtime_error("unknown decision.laya.tokenizer '" + kind + "' (metaspace-bpe, bytelevel-bpe)");
        }
        model->bytelevel = kind == "bytelevel-bpe";
        const std::string normalizer = gguf_get_str(ctx_gguf, "decision.laya.normalizer", "none");
        if (normalizer != "none" && !(normalizer == "nfc" && model->bytelevel)) {
            throw std::runtime_error("unsupported decision.laya.normalizer '" + normalizer + "' for " + kind);
        }
        model->nfc = normalizer == "nfc";
        // HF AddedToken flags: [id, flags, id, flags, ...], flags bit 0 lstrip, bit 1 normalized
        bool have_flags = false;
        const std::vector<int32_t> flag_pairs = gguf_get_arr_i32(ctx_gguf, "decision.laya.added_tokens", have_flags);
        if (flag_pairs.size() % 2 != 0) {
            throw std::runtime_error("decision.laya.added_tokens must hold (id, flags) pairs");
        }
        std::unordered_map<int32_t, int32_t> added_flags;
        for (size_t i = 0; i < flag_pairs.size(); i += 2) {
            added_flags[flag_pairs[i]] = flag_pairs[i + 1];
        }

        auto get_special = [&](const char * key, int32_t def) {
            const int64_t id = gguf_find_key(ctx_gguf, key);
            return id < 0 ? def : (int32_t) gguf_get_val_u32(ctx_gguf, id);
        };

        model->bos_id  = get_special("tokenizer.ggml.bos_token_id", 2);
        model->eos_id  = get_special("tokenizer.ggml.eos_token_id", 1);
        model->sep_id  = get_special("tokenizer.ggml.seperator_token_id", model->eos_id);
        model->mask_id = get_special("tokenizer.ggml.mask_token_id", 4);
        model->unk_id  = get_special("tokenizer.ggml.unknown_token_id", 3);
        model->pad_id  = get_special("tokenizer.ggml.padding_token_id", 0);

        const int64_t tid = gguf_find_key(ctx_gguf, "tokenizer.ggml.tokens");
        if (tid < 0) {
            throw std::runtime_error("missing tokenizer.ggml.tokens in GGUF");
        }
        const int64_t n_tokens = gguf_get_arr_n(ctx_gguf, tid);
        n_vocab_tok = n_tokens;

        // the fallbacks above are the mmBERT ids; a byte-level vocab (OLMo: [CLS] 50281) must name its own
        if (model->bytelevel) {
            for (const char * key : { "tokenizer.ggml.bos_token_id", "tokenizer.ggml.seperator_token_id", "tokenizer.ggml.mask_token_id" }) {
                if (gguf_find_key(ctx_gguf, key) < 0) {
                    throw std::runtime_error(std::string("bytelevel-bpe GGUF without ") + key + " (reconvert with conversion/laya.py)");
                }
            }
        }
        for (const int32_t id : { model->bos_id, model->eos_id, model->sep_id, model->mask_id }) {
            if (id < 0 || id >= n_tokens) {
                throw std::runtime_error("special token id " + std::to_string(id) + " is outside the vocabulary");
            }
        }
        // build_sequence writes mask_id where the head looks for marker_token_id
        if (model->mask_id != hp.marker_token_id) {
            throw std::runtime_error("tokenizer.ggml.mask_token_id " + std::to_string(model->mask_id) +
                                     " != laya.marker_token_id " + std::to_string(hp.marker_token_id));
        }

        const int64_t tid_type = gguf_find_key(ctx_gguf, "tokenizer.ggml.token_type");
        const uint32_t * tt = tid_type >= 0 ? (const uint32_t *) gguf_get_arr_data(ctx_gguf, tid_type) : nullptr;
        const int64_t n_tt  = tt ? (int64_t) gguf_get_arr_n(ctx_gguf, tid_type) : 0;

        // HF vocab strings. The converter pre-normalizes user-defined tokens
        // (U+2581 -> ' '), so restore U+2581 to match HF on the raw text.
        std::vector<std::string> vocab((size_t) n_tokens);
        std::unordered_map<std::string, int32_t> token_to_id;
        token_to_id.reserve((size_t) n_tokens);
        for (int64_t i = 0; i < n_tokens; ++i) {
            std::string tok = gguf_get_arr_str(ctx_gguf, tid, i);
            const uint32_t type = i < n_tt ? tt[i] : 1;
            if (type == 4 /* USER_DEFINED */ && !model->bytelevel) {
                std::string restored;
                restored.reserve(tok.size() + 3);
                for (char c : tok) {
                    if (c == ' ') restored += "\xe2\x96\x81"; else restored += c;
                }
                tok = std::move(restored);
            }
            token_to_id.emplace(tok, (int32_t) i);
            vocab[(size_t) i] = std::move(tok);
        }

        // bytelevel: the GPT-2 byte-to-unicode chars are the initial BPE symbols
        if (model->bytelevel) {
            uint32_t n_shift = 0;
            for (int b = 0; b < 256; ++b) {
                const bool printable = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
                const uint32_t cpt = printable ? (uint32_t) b : 256 + n_shift++;
                // cpt < 0x144: one or two UTF-8 bytes
                const std::string ch = cpt < 0x80 ? std::string(1, (char) cpt)
                                                  : std::string{ (char) (0xC0 | (cpt >> 6)), (char) (0x80 | (cpt & 0x3F)) };
                // OLMo has no token for bytes that UTF-8 never uses (0xC0, 0xC1, 0xF5..): HF drops them (unk_token null)
                const auto it = token_to_id.find(ch);
                model->byte_sym[b] = it == token_to_id.end() ? -1 : it->second;
            }
        }

        // metaspace: single-codepoint tokens and byte-fallback tokens (initial BPE symbols)
        if (!model->bytelevel) {
            model->cpt_bmp.assign(0x10000, -1);
            for (int64_t i = n_tokens - 1; i >= 0; --i) {
                const std::string & s = vocab[(size_t) i];
                size_t len = 0;
                bool   ok  = false;
                const uint32_t cpt = laya_cpt_from_utf8(s.c_str(), s.size(), len, ok);
                if (ok && len == s.size()) {
                    // lowest id wins on duplicates (iterating downwards)
                    if (cpt < 0x10000) {
                        model->cpt_bmp[cpt] = (int32_t) i;
                    } else {
                        model->cpt_high[cpt] = (int32_t) i;
                    }
                }
            }
            for (int b = 0; b < 256; ++b) {
                char buf[8];
                snprintf(buf, sizeof(buf), "<0x%02X>", b);
                const auto it = token_to_id.find(buf);
                model->byte_id[b] = it == token_to_id.end() ? -1 : it->second;
            }
        }

        // merges "a b" -> (id(a), id(b)) : (rank, id(a + b))
        const int64_t mid = gguf_find_key(ctx_gguf, "tokenizer.ggml.merges");
        if (mid >= 0) {
            const int64_t n_merges = gguf_get_arr_n(ctx_gguf, mid);
            model->merges.init((size_t) n_merges);
            for (int64_t i = 0; i < n_merges; ++i) {
                const std::string word = gguf_get_arr_str(ctx_gguf, mid, i);
                const size_t pos = word.find(' ', 1);
                if (pos == std::string::npos) {
                    continue;
                }
                const std::string a = word.substr(0, pos);
                const std::string b = word.substr(pos + 1);
                const auto ia = token_to_id.find(a);
                const auto ib = token_to_id.find(b);
                const auto im = token_to_id.find(a + b);
                if (ia == token_to_id.end() || ib == token_to_id.end() || im == token_to_id.end()) {
                    continue;
                }
                model->merges.set(laya_merge_map::key_of(ia->second, ib->second), (uint32_t) i, im->second);
            }
        }

        // added-token table: tokens typed CONTROL (3) or USER_DEFINED (4) are the
        // HF AddedVocabulary entries (newline runs, tab runs, HTML tags, <unusedN>,
        // <mask>, ...). They are matched on the raw text before Metaspace.
        // Empty strings are skipped: they would match without consuming input.
        // bytelevel: the tokens with HF normalized=True are matched after the normalizer, on the
        // normalized text; the others (the special tokens) on the raw text. lstrip and normalized come
        // from decision.laya.added_tokens, else the HF defaults (mask lstrip, USER_DEFINED normalized)
        auto insert = [](std::vector<laya_model::trie_node> & trie, const std::string & key, int32_t idx) {
            int32_t node = 0;
            for (unsigned char c : key) {
                int32_t child = -1;
                for (const auto & e : trie[node].next) {
                    if (e.first == c) {
                        child = e.second;
                        break;
                    }
                }
                if (child < 0) {
                    child = (int32_t) trie.size();
                    trie[node].next.emplace_back(c, child);
                    trie.emplace_back();
                }
                node = child;
            }
            if (trie[node].token < 0) {
                trie[node].token = idx;
            }
        };
        model->added_trie.emplace_back();
        model->added_trie_norm.emplace_back();
        for (int64_t i = 0; i < n_tt && i < n_tokens; ++i) {
            const uint32_t type = tt[i];
            if ((type != 3 /* CONTROL */ && type != 4 /* USER_DEFINED */) || vocab[(size_t) i].empty()) {
                continue;
            }
            laya_model::added_token at;
            at.s       = vocab[(size_t) i];
            at.id      = (int32_t) i;
            at.control = type == 3;
            if (model->bytelevel) {
                const auto it = added_flags.find(at.id);
                at.lstrip     = have_flags ? it != added_flags.end() && (it->second & 1) : at.id == model->mask_id;
                at.normalized = have_flags ? it != added_flags.end() && (it->second & 2) : type == 4;
            } else {
                at.lstrip = at.s == "<mask>"; // the only added token with lstrip in this family
            }

            std::string key = at.s;
            std::string key_nfc;
            if (at.normalized && model->nfc && laya_nfc(key, key_nfc)) {
                key = key_nfc;
            }
            insert(at.normalized ? model->added_trie_norm : model->added_trie, key, (int32_t) model->added_tokens.size());
            model->added_tokens.push_back(std::move(at));
        }
    }

    // ---- backend, buffer types and tensor placement ----
    model->backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!model->backend) {
        throw std::runtime_error("failed to initialize CPU backend");
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(model->backend);
    ggml_backend_buffer_type_t buft_default = ggml_backend_get_default_buffer_type(model->backend);

    // compute device: the weights go into its buffer type (token_embd stays on the host, see laya.h)
    if (mparams.device && ggml_backend_dev_type(mparams.device) != GGML_BACKEND_DEVICE_TYPE_CPU) {
        if (mparams.use_extra_bufts) {
            throw std::runtime_error("CPU repack buffers cannot be combined with the device " + laya_device_name(mparams.device));
        }
        model->dev_gpu = mparams.device;
    }
    ggml_backend_buffer_type_t buft_device = model->dev_gpu ? ggml_backend_dev_buffer_type(model->dev_gpu) : nullptr;
    if (model->dev_gpu && !buft_device) {
        throw std::runtime_error("the device " + laya_device_name(model->dev_gpu) + " has no buffer type");
    }

    std::map<std::string, size_t> tensor_offset;
    for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf); ++i) {
        tensor_offset[gguf_get_tensor_name(ctx_gguf, i)] =
            gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, i);
    }

    // one data context per destination; every context can hold every tensor
    const size_t ctx_size = static_cast<size_t>(gguf_get_n_tensors(ctx_gguf) + 1) * ggml_tensor_overhead();
    auto new_ctx = [&]() -> ggml_context * {
        struct ggml_init_params p = {
            /*.mem_size =*/ ctx_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc =*/ true,
        };
        ggml_context * c = ggml_init(p);
        if (!c) {
            throw std::runtime_error("failed to init ggml data context");
        }
        model->ctxs.emplace_back(c);
        return c;
    };
    ggml_context * ctx_loaded = new_ctx();
    ggml_context * ctx_mapped = nullptr;
    ggml_context * ctx_device = model->dev_gpu ? new_ctx() : nullptr;
    std::vector<std::pair<ggml_backend_buffer_type_t, ggml_context *>> ctx_extra; // repack destinations

    // token_embd in place: map just its byte range
    {
        const ggml_tensor * embd = ggml_get_tensor(meta, "token_embd.weight");
        ggml_backend_dev_props props;
        ggml_backend_dev_get_props(dev, &props);
        if (mparams.use_mmap && embd && props.caps.buffer_from_host_ptr) {
            const size_t off = tensor_offset.at("token_embd.weight");
            if (model->mapping.map(fname, off, ggml_nbytes(embd)) && (uintptr_t) model->mapping.data() % 32 == 0) {
                ctx_mapped = new_ctx();
            } else {
                model->mapping.unmap();
                laya_log("%s: cannot map token_embd, loading it instead\n", __func__);
            }
        }
    }

    std::vector<ggml_backend_buffer_type_t> extra_bufts;
    if (mparams.use_extra_bufts) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        auto get_extra = reg ? (ggml_backend_dev_get_extra_bufts_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_dev_get_extra_bufts") : nullptr;
        ggml_backend_buffer_type_t * p = get_extra ? get_extra(dev) : nullptr;
        while (p && *p) {
            extra_bufts.push_back(*p++);
        }
    }

    enum laya_weight_use {
        LAYA_USE_OTHER,   // added to / multiplied into F32 activations (norm weights, biases)
        LAYA_USE_MATMUL,  // src0 of a matmul with F32 activations
        LAYA_USE_MATMUL_NO_REPACK,  // the same, never in a CPU repack buffer (scorer, act head)
        LAYA_USE_ROWS,    // token_embd: get_rows, host memory
        LAYA_USE_GATHER,  // type_emb: get_rows on the compute device
    };

    // would device d run the op that uses w, with w in buft? (the check llama.cpp does for the matmuls)
    auto op_supported = [&](ggml_backend_dev_t d, ggml_tensor * w, ggml_backend_buffer_type_t buft, laya_weight_use use) {
        struct ggml_init_params p = {
            /*.mem_size =*/ 4 * ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc =*/ true,
        };
        ggml_context_ptr c(ggml_init(p));
        ggml_tensor * op = nullptr;
        if (use == LAYA_USE_MATMUL || use == LAYA_USE_MATMUL_NO_REPACK) {
            op = ggml_mul_mat(c.get(), w, ggml_new_tensor_4d(c.get(), GGML_TYPE_F32, w->ne[0], 512, w->ne[2], w->ne[3]));
        } else if (use == LAYA_USE_GATHER) {
            op = ggml_get_rows(c.get(), w, ggml_new_tensor_1d(c.get(), GGML_TYPE_I32, 512));
        } else {
            op = ggml_add(c.get(), ggml_new_tensor_4d(c.get(), GGML_TYPE_F32, w->ne[0], 512 * w->ne[1], w->ne[2], w->ne[3]), w);
        }
        GGML_ASSERT(w->buffer == nullptr);
        w->buffer = ggml_backend_buft_alloc_buffer(buft, 0);
        const bool ok = ggml_backend_dev_supports_op(d, op);
        ggml_backend_buffer_free(w->buffer);
        w->buffer = nullptr;
        return ok;
    };
    auto matmul_supported = [&](ggml_tensor * w, ggml_backend_buffer_type_t buft) {
        return op_supported(dev, w, buft, LAYA_USE_MATMUL);
    };
    int32_t n_host_fallback = 0;  // weights a device model keeps in host memory (the device cannot run their op)

    std::vector<ggml_tensor *> tensors_to_load;
    std::vector<ggml_tensor *> tensors_data;

    // weights stay in their native GGUF type (F16 for the reference F16 GGUF,
    // quantized for the k-quant models); ggml mul_mat dequantizes internally.
    auto get_tensor = [&](const std::string & name, bool required = true, laya_weight_use use = LAYA_USE_OTHER) -> ggml_tensor * {
        ggml_tensor * cur = ggml_get_tensor(meta, name.c_str());
        if (!cur) {
            if (required) {
                throw std::runtime_error("missing tensor: " + name);
            }
            return nullptr;
        }
        ggml_context * dst = ctx_loaded;
        if (use == LAYA_USE_ROWS) {
            if (ctx_mapped) {
                dst = ctx_mapped;
            }
        } else if (ctx_device) {
            const bool forced = std::find(mparams.host_weights.begin(), mparams.host_weights.end(), name) != mparams.host_weights.end();
            if (!forced && op_supported(model->dev_gpu, cur, buft_device, use)) {
                dst = ctx_device;
            } else {
                n_host_fallback++;
                laya_log("%s: %s stays in host memory: %s%s cannot run its op on %s\n", __func__, name.c_str(),
                         forced ? "(test hook) " : "", laya_device_name(model->dev_gpu).c_str(), ggml_type_name(cur->type));
            }
        } else if (use == LAYA_USE_MATMUL) {
            for (ggml_backend_buffer_type_t buft : extra_bufts) {
                if (!matmul_supported(cur, buft)) {
                    continue;
                }
                auto it = std::find_if(ctx_extra.begin(), ctx_extra.end(), [&](const auto & e) { return e.first == buft; });
                dst = it != ctx_extra.end() ? it->second : ctx_extra.emplace_back(buft, new_ctx()).second;
                break;
            }
        }
        ggml_tensor * data_tensor = ggml_dup_tensor(dst, cur);
        ggml_set_name(data_tensor, cur->name);
        tensors_to_load.push_back(cur);
        tensors_data.push_back(data_tensor);
        return data_tensor;
    };

    // encoder
    model->tok_embd   = get_tensor("token_embd.weight", true, LAYA_USE_ROWS);
    model->tok_norm   = get_tensor("token_embd_norm.weight", false);
    model->output_norm = get_tensor("output_norm.weight", false);

    model->layers.resize(hp.n_layer);
    for (int32_t il = 0; il < hp.n_layer; ++il) {
        auto & layer = model->layers[il];
        char buf[64];
        snprintf(buf, sizeof(buf), "blk.%d.", il);
        const std::string p = buf;

        // layer 0 uses an identity attn pre-norm (ModernBERT); the GGUF ships a
        // blk.0.attn_norm.weight but it must NOT be applied (matches the
        // reference, where layer 0 has no attn_norm parameter at all).
        layer.attn_norm = il == 0 ? nullptr : get_tensor(p + "attn_norm.weight");
        layer.wqkv      = get_tensor(p + "attn_qkv.weight",    true, LAYA_USE_MATMUL);
        layer.wo        = get_tensor(p + "attn_output.weight", true, LAYA_USE_MATMUL);
        layer.ffn_up    = get_tensor(p + "ffn_up.weight",      true, LAYA_USE_MATMUL);
        layer.ffn_down  = get_tensor(p + "ffn_down.weight",    true, LAYA_USE_MATMUL);
        layer.ffn_norm  = get_tensor(p + "ffn_norm.weight");
    }

    // decision head
    model->type_emb = get_tensor("type_emb.weight", true, LAYA_USE_GATHER);

    model->head_layers.resize(hp.n_head_layers);
    for (int32_t il = 0; il < hp.n_head_layers; ++il) {
        auto & layer = model->head_layers[il];
        char buf[64];
        snprintf(buf, sizeof(buf), "head.%d.", il);
        const std::string p = buf;

        layer.attn_norm   = get_tensor(p + "attn_norm.weight");
        layer.attn_norm_b = get_tensor(p + "attn_norm.bias");
        layer.wqkv        = get_tensor(p + "attn_qkv.weight",    true, LAYA_USE_MATMUL);
        layer.wqkv_b      = get_tensor(p + "attn_qkv.bias");
        layer.wo          = get_tensor(p + "attn_output.weight", true, LAYA_USE_MATMUL);
        layer.wo_b        = get_tensor(p + "attn_output.bias");
        layer.ffn_norm    = get_tensor(p + "ffn_norm.weight");
        layer.ffn_norm_b  = get_tensor(p + "ffn_norm.bias");
        layer.ffn_up      = get_tensor(p + "ffn_up.weight",      true, LAYA_USE_MATMUL);
        layer.ffn_up_b    = get_tensor(p + "ffn_up.bias");
        layer.ffn_down    = get_tensor(p + "ffn_down.weight",    true, LAYA_USE_MATMUL);
        layer.ffn_down_b  = get_tensor(p + "ffn_down.bias");
    }

    // scorer: LayerNorm(d) -> Linear(d, d) -> GELU -> Linear(d, 1)
    // (the scorer and act head matrices are F16 in every tier: no repacked kernel)
    model->scorer_0   = get_tensor("scorer.0.weight");
    model->scorer_0_b = get_tensor("scorer.0.bias");
    model->scorer_1   = get_tensor("scorer.1.weight", true, LAYA_USE_MATMUL_NO_REPACK);
    model->scorer_1_b = get_tensor("scorer.1.bias");
    model->scorer_3   = get_tensor("scorer.3.weight", true, LAYA_USE_MATMUL_NO_REPACK);
    model->scorer_3_b = get_tensor("scorer.3.bias");

    // act head: Linear(d+4, 256) -> GELU -> Linear(256, n_act)
    model->act_head_0   = get_tensor("act_head.0.weight", true, LAYA_USE_MATMUL_NO_REPACK);
    model->act_head_0_b = get_tensor("act_head.0.bias");
    model->act_head_2   = get_tensor("act_head.2.weight", true, LAYA_USE_MATMUL_NO_REPACK);
    model->act_head_2_b = get_tensor("act_head.2.bias");

    // ---- shapes against the hparams ----
    // a GGUF whose tensors do not fit would compute garbage, or read out of bounds on a device
    // without bounds checks (get_rows of token ids past the token_embd rows)
    {
        auto check = [](const ggml_tensor * t, int64_t ne0, int64_t ne1) {
            if (t && (t->ne[0] != ne0 || t->ne[1] != ne1 || t->ne[2] != 1 || t->ne[3] != 1)) {
                throw std::runtime_error(std::string("tensor ") + t->name + " has shape [" + std::to_string(t->ne[0]) + ", " +
                                         std::to_string(t->ne[1]) + ", " + std::to_string(t->ne[2]) + ", " + std::to_string(t->ne[3]) +
                                         "], expected [" + std::to_string(ne0) + ", " + std::to_string(ne1) + "] from the hparams");
            }
        };
        const int64_t d = hp.n_embd;
        check(model->tok_embd, d, hp.n_vocab > 0 ? hp.n_vocab : model->tok_embd->ne[1]);
        check(model->tok_norm, d, 1);
        check(model->output_norm, d, 1);
        for (const auto & layer : model->layers) {
            check(layer.attn_norm, d, 1);
            check(layer.wqkv, d, 3 * d);
            check(layer.wo, d, d);
            check(layer.ffn_up, d, 2 * (int64_t) hp.n_ff);
            check(layer.ffn_down, hp.n_ff, d);
            check(layer.ffn_norm, d, 1);
        }
        check(model->type_emb, d, model->type_emb->ne[1]);
        for (const auto & layer : model->head_layers) {
            const int64_t n_ff_head = layer.ffn_up->ne[1];
            check(layer.attn_norm, d, 1);
            check(layer.attn_norm_b, d, 1);
            check(layer.wqkv, d, 3 * d);
            check(layer.wqkv_b, 3 * d, 1);
            check(layer.wo, d, d);
            check(layer.wo_b, d, 1);
            check(layer.ffn_norm, d, 1);
            check(layer.ffn_norm_b, d, 1);
            check(layer.ffn_up_b, n_ff_head, 1);
            check(layer.ffn_down, n_ff_head, d);
            check(layer.ffn_down_b, d, 1);
        }
        check(model->scorer_0, d, 1);
        check(model->scorer_0_b, d, 1);
        check(model->scorer_1, d, d);
        check(model->scorer_1_b, d, 1);
        check(model->scorer_3, d, 1);
        check(model->scorer_3_b, 1, 1);
        const int64_t n_act_hidden = model->act_head_0->ne[1];
        check(model->act_head_0, d + 4, n_act_hidden);
        check(model->act_head_0_b, n_act_hidden, 1);
        check(model->act_head_2, n_act_hidden, hp.n_act);
        check(model->act_head_2_b, hp.n_act, 1);

        if (n_vocab_tok > model->tok_embd->ne[1]) {
            throw std::runtime_error("the tokenizer has " + std::to_string(n_vocab_tok) + " tokens, token_embd.weight only " +
                                     std::to_string(model->tok_embd->ne[1]) + " rows");
        }
        if (hp.n_qtype < 1 || hp.n_qtype > model->type_emb->ne[1]) {
            throw std::runtime_error("laya.n_qtype " + std::to_string(hp.n_qtype) + " does not fit type_emb.weight (" +
                                     std::to_string(model->type_emb->ne[1]) + " rows)");
        }
    }

    // ---- buffers ----
    auto alloc_ctx = [&](ggml_context * c, ggml_backend_buffer_type_t buft) {
        if (!ggml_get_first_tensor(c)) {
            return;
        }
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
        if (!b) {
            throw std::runtime_error(std::string("failed to allocate the ") + ggml_backend_buft_name(buft) + " weight buffer");
        }
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        model->bufs.emplace_back(b);
    };
    if (ctx_mapped) {
        ggml_backend_buffer_t b = ggml_backend_dev_buffer_from_host_ptr(dev, model->mapping.data(), ggml_nbytes(model->tok_embd), ggml_nbytes(model->tok_embd));
        if (!b) {
            throw std::runtime_error("failed to wrap the token_embd mapping in a buffer");
        }
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        model->bufs.emplace_back(b);
        if (ggml_backend_tensor_alloc(b, model->tok_embd, model->mapping.data()) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("failed to place token_embd in the mapping");
        }
        model->n_bytes_mapped = ggml_nbytes(model->tok_embd);
    }
    alloc_ctx(ctx_loaded, buft_default);
    if (ctx_device && ggml_get_first_tensor(ctx_device)) {
        ggml_backend_buffer_t b = ggml_backend_alloc_ctx_tensors_from_buft(ctx_device, buft_device);
        if (!b) {
            // no silent host fallback: the scheduler would copy every large-batch matmul's weight to the device
            // on each pass (op_offload) and run the rest on the CPU. The caller decides (--decision-device auto
            // falls back to the CPU model, gpu fails).
            size_t need = 0;
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_device); t; t = ggml_get_next_tensor(ctx_device, t)) {
                need += ggml_nbytes(t);
            }
            throw std::runtime_error("cannot allocate " + std::to_string(need / 1048576) + " MiB of weights in the " +
                                     ggml_backend_buft_name(buft_device) + " buffer");
        }
        ggml_backend_buffer_set_usage(b, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        model->bufs.emplace_back(b);
    }
    for (const auto & e : ctx_extra) {
        alloc_ctx(e.second, e.first);
    }

    // ---- read the tensor data ----
    {
#if defined(_WIN32)
        std::ifstream fin(std::filesystem::path(laya_utf8_to_wide(fname)), std::ios::binary);
#else
        std::ifstream fin(fname, std::ios::binary);
#endif
        if (!fin) {
            throw std::runtime_error("failed to open " + std::string(fname));
        }

        std::vector<uint8_t> read_buf;
        for (size_t i = 0; i < tensors_to_load.size(); ++i) {
            ggml_tensor * cur = tensors_data[i];
            GGML_ASSERT(cur && "tensor not found in the data contexts");
            if (cur == model->tok_embd && ctx_mapped) {
                continue;
            }
            auto it_off = tensor_offset.find(cur->name);
            GGML_ASSERT(it_off != tensor_offset.end() && "no offset for tensor");
            fin.seekg(it_off->second, std::ios::beg);
            if (!fin) {
                throw std::runtime_error("failed to seek for tensor " + std::string(cur->name));
            }
            const size_t num_bytes = ggml_nbytes(cur);
            if (ggml_backend_buft_is_host(ggml_backend_buffer_get_type(cur->buffer))) {
                fin.read((char *) cur->data, num_bytes);
            } else {
                // repack buffers convert in set_tensor, which takes the whole tensor at once
                read_buf.resize(num_bytes);
                fin.read((char *) read_buf.data(), num_bytes);
                ggml_backend_tensor_set(cur, read_buf.data(), 0, num_bytes);
            }
            if (!fin) {
                throw std::runtime_error("failed to read tensor " + std::string(cur->name));
            }
            if (ggml_backend_buffer_get_type(cur->buffer) == buft_default) {
                model->n_bytes_loaded += num_bytes;
            } else if (buft_device && ggml_backend_buffer_get_type(cur->buffer) == buft_device) {
                model->n_bytes_device += num_bytes;
            } else {
                model->n_bytes_repacked += num_bytes;
                model->n_repacked++;
            }
        }
    }

    if (mparams.use_mlock) {
        bool ok = true;
        if (model->mapping.base) {
            model->mapping.locked = laya_mlock(model->mapping.base, model->mapping.size);
            ok = model->mapping.locked;
        }
        for (const auto & b : model->bufs) {
            if (ggml_backend_buffer_is_host(b.get()) && ggml_backend_buffer_get_base(b.get()) != model->mapping.data()) {
                ok = laya_mlock(ggml_backend_buffer_get_base(b.get()), ggml_backend_buffer_get_size(b.get())) && ok;
            }
        }
        if (!ok) {
            laya_log("%s: warning: failed to lock the model in RAM (RLIMIT_MEMLOCK / working set too small?)\n", __func__);
        }
    }

    model->n_host_weights = n_host_fallback;
    if (model->dev_gpu) {
        laya_log("%s: device %s (%s): %.1f MiB of weights in device memory, %.1f MiB in host memory, %.1f MiB mapped, %d weights kept on the host\n",
                 __func__, laya_device_name(model->dev_gpu).c_str(), ggml_backend_dev_description(model->dev_gpu),
                 model->n_bytes_device / 1048576.0, model->n_bytes_loaded / 1048576.0, model->n_bytes_mapped / 1048576.0, n_host_fallback);
        laya_warn_unverified(model.get());
    }

    // the header, the vocab strings and the merges are no longer needed
    model->ctx_gguf.reset();
    model->ctx_meta.reset();
    return model.release();
}

void laya_model_free(laya_model * model) {
    if (!model) {
        return;
    }
    if (model->backend) {
        ggml_backend_free(model->backend);
    }
    delete model;
}

const laya_hparams & laya_model_hparams(const laya_model * model) {
    return model->hparams;
}

laya_model_memory laya_model_memory_info(const laya_model * model) {
    laya_model_memory m;
    m.device     = model->n_bytes_device;
    m.loaded     = model->n_bytes_loaded;
    m.mapped     = model->n_bytes_mapped;
    m.repacked   = model->n_bytes_repacked;
    m.n_repacked = model->n_repacked;
    return m;
}

std::vector<ggml_backend_dev_t> laya_gpu_devices() {
    // llama.cpp order: the GPUs, then the iGPUs
    std::vector<ggml_backend_dev_t> gpus, igpus;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        switch (ggml_backend_dev_type(d)) {
            case GGML_BACKEND_DEVICE_TYPE_GPU:  gpus.push_back(d);  break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU: igpus.push_back(d); break;
            default: break;
        }
    }
    gpus.insert(gpus.end(), igpus.begin(), igpus.end());
    return gpus;
}

ggml_backend_dev_t laya_device_select(const std::string & mode, int32_t gpu_index) {
    if (mode == "cpu") {
        return nullptr;
    }
    if (mode != "gpu" && mode != "auto") {
        throw std::runtime_error("unknown device '" + mode + "' (cpu, gpu, auto)");
    }
    const std::vector<ggml_backend_dev_t> gpus = laya_gpu_devices();
    if (gpu_index >= 0 && (size_t) gpu_index < gpus.size()) {
        return gpus[gpu_index];
    }
    if (mode == "auto") {
        if (gpus.empty()) {
            laya_log("%s: auto: no GPU / iGPU device in this build and machine, using the CPU\n", __func__);
        } else {
            laya_log("%s: auto: no GPU device %d (have %zu), using the CPU\n", __func__, gpu_index, gpus.size());
        }
        return nullptr;
    }
    std::string have;
    for (size_t i = 0; i < gpus.size(); ++i) {
        have += (i ? ", " : "") + std::to_string(i) + " = " + ggml_backend_dev_name(gpus[i]);
    }
    throw std::runtime_error("no GPU device " + std::to_string(gpu_index) + " (this build and machine have " +
                             (gpus.empty() ? std::string("none") : have) + ")");
}

std::string laya_device_name(ggml_backend_dev_t device) {
    return device && ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_CPU ? ggml_backend_dev_name(device) : "cpu";
}

ggml_backend_dev_t laya_model_device(const laya_model * model) {
    return model->dev_gpu;
}

int32_t laya_vocab_bos (const laya_model * model) { return model->bos_id; }
int32_t laya_vocab_sep (const laya_model * model) { return model->sep_id; }
int32_t laya_vocab_mask(const laya_model * model) { return model->mask_id; }

std::string laya_vocab_mask_token(const laya_model * model) {
    for (const auto & at : model->added_tokens) {
        if (at.id == model->mask_id) {
            return at.s;
        }
    }
    return "<mask>";
}

// ---- self-contained tokenizer (HF fast tokenizer port) ----
// bytelevel-bpe (English checkpoints) is laya_tokenize_bytelevel below. metaspace-bpe:
// the laya-multilingual tokenizer is a Metaspace pre-tokenizer
// (replacement U+2581, prepend_scheme="always") on top of a BPE with byte
// fallback and the mmBERT vocabulary. llama.cpp's built-in GPT-2 pre-tokenizer
// cannot reproduce it for this vocab, so this is implemented here against the
// GGUF tokenizer data, following HF tokenizers step by step:
//   1. AddedVocabulary: leftmost-longest match of the added tokens on the raw
//      text (a byte trie); <mask> has lstrip and takes the whitespace before it
//   2. each gap: Replace(' ' -> U+2581), prepend U+2581 unless present, split
//      before every U+2581
//   3. each word: BPE merges from a heap (lowest rank, then leftmost), with
//      the same stale-entry test as tokenizers' Word::merge_all
// Cost is O(n log n) in the text length, so long space-free words (CJK,
// base64, URLs) stay cheap.

static const char LAYA_SPACE[] = "\xe2\x96\x81"; // U+2581

// Rust char::is_whitespace (Unicode White_Space), used by HF lstrip
static bool laya_is_whitespace(uint32_t c) {
    return (c >= 0x09 && c <= 0x0D) || c == 0x20 || c == 0x85 || c == 0xA0 || c == 0x1680 ||
           (c >= 0x2000 && c <= 0x200A) || c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

// buffers reused across the words of one laya_tokenize call
struct laya_bpe_scratch {
    struct sym {
        int32_t id;
        int32_t prev;
        int32_t next;
        bool    alive;
    };
    struct merge {
        uint64_t key;   // rank << 32 | pos: top = lowest rank, then lowest position
        int32_t  id;
    };
    std::vector<sym>   syms;
    std::vector<merge> heap;
    std::string        norm;
    std::vector<std::pair<size_t, size_t>> pieces; // bytelevel: pre-tokenized pieces
};

static void laya_bpe_merge(const laya_model * model, laya_bpe_scratch & sc, std::vector<int32_t> & out);

// BPE over one word (starts with U+2581, contains no other U+2581)
static void laya_bpe_word(const laya_model * model, const char * word, size_t n, laya_bpe_scratch & sc, std::vector<int32_t> & out) {
    using sym   = laya_bpe_scratch::sym;
    std::vector<sym> & syms = sc.syms;
    syms.clear();

    auto add = [&](int32_t id) {
        const int32_t idx = (int32_t) syms.size();
        syms.push_back({ id, idx - 1, idx + 1, true });
    };

    // initial symbols: one per codepoint, else its <0xXX> bytes, else <unk>
    // (consecutive unknowns fused, fuse_unk = true)
    bool pending_unk = false;
    size_t i = 0;
    while (i < n) {
        size_t len = 0;
        bool   ok  = false;
        const uint32_t cpt = laya_cpt_from_utf8(word + i, n - i, len, ok);
        int32_t id = -1;
        if (ok && cpt < 0x10000) {
            id = model->cpt_bmp[cpt];
        } else if (ok) {
            const auto it = model->cpt_high.find(cpt);
            id = it == model->cpt_high.end() ? -1 : it->second;
        }
        if (id >= 0) {
            if (pending_unk) {
                add(model->unk_id);
                pending_unk = false;
            }
            add(id);
        } else {
            bool all = true;
            for (size_t k = 0; k < len; ++k) {
                all = all && model->byte_id[(uint8_t) word[i + k]] >= 0;
            }
            if (all) {
                for (size_t k = 0; k < len; ++k) {
                    add(model->byte_id[(uint8_t) word[i + k]]);
                }
            } else {
                pending_unk = true;
            }
        }
        i += len;
    }
    if (pending_unk) {
        add(model->unk_id);
    }
    laya_bpe_merge(model, sc, out);
}

// BPE merges over sc.syms (linked in order), appending the result to out
static void laya_bpe_merge(const laya_model * model, laya_bpe_scratch & sc, std::vector<int32_t> & out) {
    using merge = laya_bpe_scratch::merge;
    using sym   = laya_bpe_scratch::sym;
    std::vector<sym> & syms = sc.syms;
    if (syms.empty()) {
        return;
    }
    syms.back().next = -1;

    auto cmp = [](const merge & a, const merge & b) {
        return a.key > b.key;
    };
    auto entry = [](uint32_t rank, int32_t pos, int32_t id) {
        return merge{ ((uint64_t) rank << 32) | (uint32_t) pos, id };
    };
    const laya_merge_map & merges = model->merges;

    std::vector<merge> & heap = sc.heap;
    heap.clear();
    for (size_t k = 0; k + 1 < syms.size(); ++k) {
        const auto * m = merges.find(syms[k].id, syms[k + 1].id);
        if (m) {
            heap.push_back(entry(m->rank, (int32_t) k, m->id));
        }
    }
    std::make_heap(heap.begin(), heap.end(), cmp);

    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), cmp);
        const merge   top = heap.back();
        const int32_t pos = (int32_t) (uint32_t) top.key;
        heap.pop_back();

        sym & left = syms[pos];
        if (!left.alive || left.next < 0) {
            continue;
        }
        sym & right = syms[left.next];
        // stale entry: the pair at pos no longer merges into the same token
        const auto * m = merges.find(left.id, right.id);
        if (!m || m->id != top.id) {
            continue;
        }

        left.id      = top.id;
        right.alive  = false;
        left.next    = right.next;
        if (left.next >= 0) {
            syms[left.next].prev = pos;
        }

        if (left.prev >= 0) {
            const auto * mp = merges.find(syms[left.prev].id, left.id);
            if (mp) {
                heap.push_back(entry(mp->rank, left.prev, mp->id));
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
        }
        if (left.next >= 0) {
            const auto * mn = merges.find(left.id, syms[left.next].id);
            if (mn) {
                heap.push_back(entry(mn->rank, pos, mn->id));
                std::push_heap(heap.begin(), heap.end(), cmp);
            }
        }
    }

    for (int32_t k = 0; k >= 0; k = syms[k].next) {
        out.push_back(syms[k].id);
    }
}

// Tokenize one Metaspace segment (the text between two added-token matches):
// normalize ' ' -> U+2581, prepend U+2581 if not already present, split before
// every marker and run BPE per word. prepend_scheme is "always", so every
// segment is treated as an independent word boundary.
static void laya_tokenize_segment(const laya_model * model, const char * text, size_t n, laya_bpe_scratch & sc, std::vector<int32_t> & out) {
    if (n == 0) {
        return;
    }

    std::string & norm = sc.norm;
    norm.clear();
    if (!(n >= 3 && memcmp(text, LAYA_SPACE, 3) == 0) && text[0] != ' ') {
        norm += LAYA_SPACE;
    }
    for (size_t i = 0; i < n; ++i) {
        if (text[i] == ' ') {
            norm += LAYA_SPACE;
        } else {
            norm += text[i];
        }
    }

    size_t pos = 0;
    while (pos < norm.size()) {
        // at pos we are at a word boundary (the leading marker)
        size_t next = norm.find(LAYA_SPACE, pos + 3);
        if (next == std::string::npos) {
            next = norm.size();
        }
        laya_bpe_word(model, norm.data() + pos, next - pos, sc, out);
        pos = next;
    }
}

// Longest added token of trie starting at text[pos]: index into added_tokens, or -1
static int32_t laya_match_added(const std::vector<laya_model::trie_node> & trie, const std::string & text, size_t pos, size_t & len) {
    int32_t best = -1;
    int32_t node = 0;
    for (size_t j = pos; j < text.size(); ++j) {
        int32_t child = -1;
        for (const auto & e : trie[node].next) {
            if (e.first == (uint8_t) text[j]) {
                child = e.second;
                break;
            }
        }
        if (child < 0) {
            break;
        }
        node = child;
        if (trie[node].token >= 0) {
            best = trie[node].token;
            len  = j + 1 - pos;
        }
    }
    return best;
}

// start of the whitespace run that ends at pos, not before lo
static size_t laya_whitespace_start(const std::string & text, size_t lo, size_t pos) {
    while (pos > lo) {
        size_t b = pos - 1;
        while (b > lo && ((uint8_t) text[b] & 0xC0) == 0x80) {
            --b;
        }
        size_t len = 0;
        bool   ok  = false;
        const uint32_t cpt = laya_cpt_from_utf8(text.c_str() + b, pos - b, len, ok);
        if (!ok || len != pos - b || !laya_is_whitespace(cpt)) {
            break;
        }
        pos = b;
    }
    return pos;
}

// Split text at the added tokens of trie (leftmost-longest, lstrip); on_gap(a, b) gets each text
// range between two matches, the ids of the matches go to out
template <typename F>
static void laya_split_added(const laya_model * model, const std::vector<laya_model::trie_node> & trie, const std::string & text, std::vector<int32_t> & out, F on_gap) {
    size_t seg_start = 0;
    size_t i = 0;
    while (i < text.size()) {
        size_t len = 0;
        const int32_t idx = trie.size() <= 1 ? -1 : laya_match_added(trie, text, i, len);
        if (idx < 0) {
            ++i;
            continue;
        }
        const auto & at = model->added_tokens[(size_t) idx];
        const size_t start = at.lstrip ? laya_whitespace_start(text, seg_start, i) : i;
        on_gap(seg_start, start);
        out.push_back(at.id);
        i += len;
        seg_start = i;
    }
    on_gap(seg_start, text.size());
}

// bytelevel: pre-tokenize text[begin, end) with the GPT-2 regex, then BPE over the byte-to-unicode chars
static void laya_tokenize_bytelevel_words(const laya_model * model, const std::string & text, size_t begin, size_t end, laya_bpe_scratch & sc, std::vector<int32_t> & out) {
    if (begin >= end) {
        return;
    }
    sc.pieces.clear();
    laya_gpt2_split(text, begin, end, sc.pieces);
    for (const auto & p : sc.pieces) {
        sc.syms.clear();
        for (size_t k = p.first; k < p.second; ++k) {
            const int32_t id = model->byte_sym[(uint8_t) text[k]];
            if (id >= 0) {
                const int32_t idx = (int32_t) sc.syms.size();
                sc.syms.push_back({ id, idx - 1, idx + 1, true });
            }
        }
        laya_bpe_merge(model, sc, out);
    }
}

// HF AddedVocabulary + ByteLevel BPE: split the raw text at the raw added tokens; normalize each gap,
// split it at the normalized added tokens, and pre-tokenize + BPE what is left
static std::vector<int32_t> laya_tokenize_bytelevel(const laya_model * model, const std::string & text) {
    std::vector<int32_t> out;
    laya_bpe_scratch sc;
    std::string gap;
    std::string norm;
    laya_split_added(model, model->added_trie, text, out, [&](size_t a, size_t b) {
        if (a >= b) {
            return;
        }
        gap.assign(text, a, b - a);
        const std::string & t = model->nfc && laya_nfc(gap, norm) ? norm : gap;
        laya_split_added(model, model->added_trie_norm, t, out, [&](size_t a2, size_t b2) {
            laya_tokenize_bytelevel_words(model, t, a2, b2, sc, out);
        });
    });
    return out;
}

std::vector<int32_t> laya_tokenize(const laya_model * model, const std::string & text) {
    std::vector<int32_t> out;
    if (text.empty()) {
        return out;
    }
    if (model->bytelevel) {
        return laya_tokenize_bytelevel(model, text);
    }

    // split the raw text at added tokens; every gap goes through Metaspace + BPE
    laya_bpe_scratch sc;
    size_t seg_start = 0;
    size_t i = 0;
    while (i < text.size()) {
        size_t len = 0;
        const int32_t idx = model->added_trie.empty() ? -1 : laya_match_added(model->added_trie, text, i, len);
        if (idx < 0) {
            ++i;
            continue;
        }
        const auto & at = model->added_tokens[(size_t) idx];
        // lstrip: the whitespace before the token belongs to it (HF lstrip=True)
        const size_t start = at.lstrip ? laya_whitespace_start(text, seg_start, i) : i;
        laya_tokenize_segment(model, text.data() + seg_start, start - seg_start, sc, out);
        out.push_back(at.id);
        i += len;
        seg_start = i;
    }
    laya_tokenize_segment(model, text.data() + seg_start, text.size() - seg_start, sc, out);
    return out;
}

void laya_vocab_added_tokens(const laya_model * model, std::vector<std::string> & text, std::vector<bool> & control) {
    text.clear();
    control.clear();
    for (const auto & at : model->added_tokens) {
        text.push_back(at.s);
        control.push_back(at.control);
    }
}

laya_context * laya_init(const laya_model * model, int n_threads) {
    laya_context_params params;
    params.n_threads = n_threads;
    return laya_init_ext(model, params);
}

laya_context * laya_init_ext(const laya_model * model, const laya_context_params & params) {
    // laya_free on a throw: it frees whatever was created so far (backends, threadpool)
    struct ctx_deleter {
        void operator()(laya_context * c) const { laya_free(c); }
    };
    std::unique_ptr<laya_context, ctx_deleter> ctx(new laya_context());
    ctx->model     = model;
    ctx->n_threads = std::max(1, params.n_threads);
    ctx->qos       = params.qos;
    ctx->precision = params.precision;

    ctx->backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!ctx->backend_cpu) {
        throw std::runtime_error("failed to initialize CPU backend");
    }
    ctx->backend = ctx->backend_cpu;

    // resolve through the registry so that GGML_BACKEND_DL builds link
    ggml_backend_dev_t dev = ggml_backend_get_device(ctx->backend_cpu);
    ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
    if (reg) {
        auto set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        if (set_n_threads_fn) {
            set_n_threads_fn(ctx->backend_cpu, ctx->n_threads);
        }

        auto tp_new  = (laya_threadpool_new_t)  ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_new");
        auto tp_free = (laya_threadpool_free_t) ggml_backend_reg_get_proc_address(reg, "ggml_threadpool_free");
        auto tp_set  = (laya_set_threadpool_t)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (tp_new && tp_free && tp_set) {
            ggml_threadpool_params tpp = ggml_threadpool_params_default(ctx->n_threads);
            tpp.poll = (uint32_t) std::min(100, std::max(0, params.poll));
            ctx->threadpool = tp_new(&tpp);
            if (ctx->threadpool) {
                ctx->threadpool_free = tp_free;
                tp_set(ctx->backend_cpu, ctx->threadpool);
            }
        }
        if (!ctx->threadpool) {
            laya_log("%s: no persistent threadpool, ggml creates the threads on every compute\n", __func__);
        }
    }

    std::vector<ggml_backend_t> backends;
    if (model->dev_gpu) {
        if (params.use_blas) {
            throw std::runtime_error("BLAS kernels are CPU-only, the model was loaded for the device " + laya_device_name(model->dev_gpu));
        }
        ctx->backend_gpu = ggml_backend_dev_init(model->dev_gpu, nullptr);
        if (!ctx->backend_gpu) {
            throw std::runtime_error("failed to initialize the backend of the device " + laya_device_name(model->dev_gpu));
        }
        ctx->backend = ctx->backend_gpu;
        backends.push_back(ctx->backend_gpu);
    } else if (params.use_blas) {
        ggml_backend_dev_t blas = ggml_backend_dev_by_name("BLAS");
        ctx->backend_blas = blas ? ggml_backend_dev_init(blas, nullptr) : nullptr;
        if (!ctx->backend_blas) {
            // an explicit request: computing with other kernels would change the logits silently
            throw std::runtime_error("BLAS kernels requested, but this build has no BLAS backend");
        }
        // capped: ggml-blas starts n - 1 new threads for every weight conversion
        ctx->n_threads_blas = params.n_threads_blas > 0 ? params.n_threads_blas
                                                        : std::min(ctx->n_threads, LAYA_BLAS_THREADS_AUTO);
        ggml_backend_reg_t breg = ggml_backend_dev_backend_reg(blas);
        auto set_n_threads_fn = breg ? (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(breg, "ggml_backend_set_n_threads") : nullptr;
        if (set_n_threads_fn) {
            set_n_threads_fn(ctx->backend_blas, ctx->n_threads_blas);
        }
        backends.push_back(ctx->backend_blas);
    }
    // the scheduler wants the CPU backend last
    backends.push_back(ctx->backend_cpu);

    std::vector<ggml_backend_buffer_type_t> bufts;
    for (ggml_backend_t b : backends) {
        bufts.push_back(ggml_backend_get_default_buffer_type(b));
    }
    ctx->sched.reset(ggml_backend_sched_new(backends.data(), bufts.data(), (int) backends.size(), 8192, false, true));
    if (!ctx->sched.get()) {
        throw std::runtime_error("failed to initialize backend scheduler");
    }

    if (!params.trace_dir.empty()) {
        ctx->trace_dir   = params.trace_dir;
        ctx->trace_index = ggml_fopen((ctx->trace_dir + "/trace.jsonl").c_str(), "wb");
        if (ctx->trace_index) {
            ggml_backend_sched_set_eval_callback(ctx->sched.get(), laya_trace_eval, ctx.get());
            laya_log("%s: layer trace into %s\n", __func__, ctx->trace_dir.c_str());
        } else {
            laya_log("%s: trace dir %s: cannot create trace.jsonl there, no trace\n", __func__, ctx->trace_dir.c_str());
        }
    }

    laya_placement_probe(ctx.get());
    laya_log("%s: placement %s\n", __func__, laya_placement_str(ctx->placement).c_str());
    if (params.strict_placement && ctx->placement.cpu_fallback > 0) {
        std::string ops;
        for (const std::string & o : ctx->placement.fallback_ops) {
            ops += (ops.empty() ? "" : ", ") + o;
        }
        throw std::runtime_error("strict placement: " + std::to_string(ctx->placement.cpu_fallback) +
                                 " graph nodes would run on the CPU instead of " + ctx->placement.device + ": " + ops);
    }
    if (params.strict_placement && ctx->placement.host_weights > 0) {
        throw std::runtime_error("strict placement: " + std::to_string(ctx->placement.host_weights) +
                                 " weights stay in host memory instead of " + ctx->placement.device);
    }

    return ctx.release();
}

void laya_free(laya_context * ctx) {
    if (!ctx) {
        return;
    }
    ctx->sched.reset();
    if (ctx->trace_index) {
        fclose(ctx->trace_index);
    }
    if (ctx->backend_gpu) {
        ggml_backend_free(ctx->backend_gpu);
    }
    if (ctx->backend_blas) {
        ggml_backend_free(ctx->backend_blas);
    }
    if (ctx->backend_cpu) {
        ggml_backend_free(ctx->backend_cpu);
    }
    // after the backend that uses it
    if (ctx->threadpool) {
        ctx->threadpool_free(ctx->threadpool);
    }
    delete ctx;
}

std::string laya_context_kernels(const laya_context * ctx) {
    if (ctx->backend_gpu) {
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(ctx->model->dev_gpu);
        std::string s = reg ? ggml_backend_reg_name(reg) : laya_device_name(ctx->model->dev_gpu);
        std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char) std::tolower(c); });
        return s == "mtl" ? "metal" : s;  // the Metal registry is called "MTL"
    }
    std::string s = "cpu";
    if (ctx->model->n_repacked > 0) {
        s += "+repack";
    }
    if (ctx->backend_blas) {
        s += "+blas";
    }
    return s;
}

laya_placement laya_context_placement(const laya_context * ctx) {
    return ctx->placement;
}

std::string laya_placement_str(const laya_placement & p) {
    std::string s = p.device + ": " + std::to_string(p.n_nodes) + " nodes (";
    for (size_t i = 0; i < p.backends.size(); ++i) {
        s += (i ? ", " : "") + p.backends[i].first + " " + std::to_string(p.backends[i].second);
    }
    s += "), " + std::to_string(p.n_splits) + (p.n_splits == 1 ? " split" : " splits") + ", cpu fallback " + std::to_string(p.cpu_fallback);
    if (p.host_weights > 0) {
        s += ", host weights " + std::to_string(p.host_weights);
    }
    for (size_t i = 0; i < p.fallback_ops.size(); ++i) {
        s += (i ? ", " : " (") + p.fallback_ops[i] + (i + 1 == p.fallback_ops.size() ? ")" : "");
    }
    return s;
}

int32_t laya_context_n_threads_blas(const laya_context * ctx) {
    return ctx->backend_blas ? ctx->n_threads_blas : 0;
}

laya_precision laya_context_precision(const laya_context * ctx) {
    return ctx->precision;
}

const char * laya_precision_name(laya_precision precision) {
    return precision == LAYA_PRECISION_STRICT ? "strict" : "default";
}

bool laya_precision_from_name(const std::string & name, laya_precision & precision) {
    if (name == "default") {
        precision = LAYA_PRECISION_DEFAULT;
        return true;
    }
    if (name == "strict") {
        precision = LAYA_PRECISION_STRICT;
        return true;
    }
    return false;
}

// the warm-up input: n tokens (8 .. max_len) of one sequence, two markers
struct laya_probe_input {
    std::vector<int32_t> tokens, pos, seq, qtype, marker_pos, marker_mask;
    int32_t              seq_start = 0;
    laya_batch           batch;
};

static void laya_probe_input_init(const laya_model * model, int32_t n_tokens, laya_probe_input & in) {
    const int32_t n = std::max(8, std::min(n_tokens, model->hparams.max_len > 0 ? model->hparams.max_len : 1024));
    in.tokens.assign(n, model->sep_id);
    in.pos.resize(n);
    in.seq.assign(n, 0);
    in.qtype.assign(n, 0);
    in.marker_pos.assign(LAYA_MAX_MARKERS, 0);
    in.marker_mask.assign(LAYA_MAX_MARKERS, 0);
    in.tokens[0] = model->bos_id;
    for (int32_t i = 0; i < n; ++i) {
        in.pos[i] = i;
    }
    for (int32_t m = 0; m < 2; ++m) {
        in.tokens[n - 3 + m] = model->mask_id;
        in.marker_pos[m]     = n - 3 + m;
        in.marker_mask[m]    = 1;
    }
    in.seq_start = 0;

    in.batch.n_tokens    = n;
    in.batch.n_seqs      = 1;
    in.batch.tokens      = in.tokens.data();
    in.batch.positions   = in.pos.data();
    in.batch.seq_id      = in.seq.data();
    in.batch.qtype       = in.qtype.data();
    in.batch.marker_pos  = in.marker_pos.data();
    in.batch.marker_mask = in.marker_mask.data();
    in.batch.seq_start   = &in.seq_start;
}

int laya_warmup(laya_context * ctx, int32_t n_tokens) {
    // >= 32 tokens: the BLAS backend (min batch 32) takes its matmuls too
    laya_probe_input in;
    laya_probe_input_init(ctx->model, n_tokens, in);
    laya_result res;
    return laya_encode(ctx, in.batch, res);
}

std::string laya_blas_description() {
    ggml_backend_dev_t blas = ggml_backend_dev_by_name("BLAS");
    return blas ? ggml_backend_dev_description(blas) : "";
}

#if defined(_WIN32)
std::wstring laya_utf8_to_wide(const std::string & utf8) {
    if (utf8.empty() || utf8.size() > (size_t) INT_MAX) {
        return std::wstring();
    }
    const int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), (int) utf8.size(), nullptr, 0);
    if (n <= 0) {
        return std::wstring();
    }
    std::wstring out((size_t) n, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8.data(), (int) utf8.size(), &out[0], n);
    return out;
}

// PROCESSOR_RELATIONSHIP::EfficiencyClass exists in the Windows 10 SDK headers; older headers
// (older MinGW-w64) declare the same byte as Reserved[0] (Flags, EfficiencyClass, Reserved[20]
// vs Flags, Reserved[21]). Picked at compile time, so either header set builds.
template <typename T>
static auto laya_efficiency_class(const T & p, int) -> decltype((int32_t) p.EfficiencyClass) {
    return (int32_t) p.EfficiencyClass;
}
template <typename T>
static int32_t laya_efficiency_class(const T & p, long) {
    return (int32_t) p.Reserved[0];
}
#endif

int32_t laya_cpu_perf_cores() {
#if defined(__APPLE__)
    int32_t n = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.perflevel0.physicalcpu", &n, &len, nullptr, 0) == 0 && n > 0) {
        return n;
    }
    len = sizeof(n);
    if (sysctlbyname("hw.physicalcpu", &n, &len, nullptr, 0) == 0 && n > 0) {
        return n;
    }
    return 0;
#elif defined(_WIN32) && (_WIN32_WINNT >= 0x0601)
    // hybrid CPUs (Alder Lake and later, Snapdragon X): count the cores of the highest
    // EfficiencyClass only; on a uniform CPU every core has class 0 and all are counted
    DWORD size = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &size);
    if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || size == 0) {
        return 0;
    }
    std::vector<char> buf(size);
    auto * info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data());
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &size)) {
        return 0;
    }
    int32_t best_class = -1;
    int32_t n_best     = 0;
    for (DWORD off = 0; off < size;) {
        const auto * p = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *>(buf.data() + off);
        if (p->Relationship == RelationProcessorCore) {
            const int32_t cls = laya_efficiency_class(p->Processor, 0);
            if (cls > best_class) {
                best_class = cls;
                n_best     = 0;
            }
            if (cls == best_class) {
                n_best++;
            }
        }
        off += p->Size;
    }
    return n_best;
#else
    return 0;
#endif
}

// Only libomp and libgomp read these. The MSVC runtime (vcomp, /openmp, OpenMP 2.0) documents
// OMP_DYNAMIC, OMP_NESTED, OMP_NUM_THREADS and OMP_SCHEDULE only, so in an MSVC build with
// GGML_OPENMP=ON this is a no-op and idle workers keep vcomp's own wait policy.
void laya_cpu_env_defaults() {
    const char * vars[][2] = {
        { "KMP_BLOCKTIME",   "0"       }, // LLVM libomp: workers sleep right after a parallel region
        { "OMP_WAIT_POLICY", "passive" }, // libomp / libgomp: no spin-waiting between regions
    };
    for (const auto & v : vars) {
        if (getenv(v[0]) != nullptr) {
            continue;
        }
#if defined(_WIN32)
        _putenv_s(v[0], v[1]);
#else
        setenv(v[0], v[1], 0);
#endif
    }
}

// ---- graph helpers ------------------------------------------------------

static ggml_tensor * laya_norm(ggml_context * ctx0, ggml_tensor * cur,
                               ggml_tensor * w, ggml_tensor * b, float eps) {
    cur = ggml_norm(ctx0, cur, eps);
    if (w) {
        cur = ggml_mul(ctx0, cur, w);
    }
    if (b) {
        cur = ggml_add(ctx0, cur, b);
    }
    return cur;
}

// Every matmul of the graph goes through laya_mm (14 sites: KQ and PV in laya_attn; QKV, Wo, FFN up
// and down of each encoder and head layer; scorer.1, scorer.3, act_head.0, act_head.2).
// LAYA_MM_ACT: both operands are F32 activations (KQ: src0 = K, PV: src0 = V^T). CUDA runs an F32
// src0 without tensor cores only at GGML_PREC_F32_PEDANTIC (with GGML_PREC_F32 cuBLAS still uses
// TF32), so these always ask for PEDANTIC. LAYA_MM_WEIGHT: src0 is a weight matrix; GGML_PREC_F32
// keeps F16 / quantized weights at F32 accumulation, strict asks for PEDANTIC here too (an F32 GGUF
// then runs without TF32). The CPU and BLAS backends ignore prec: both modes are bitwise equal there.
enum laya_mm_kind {
    LAYA_MM_ACT,
    LAYA_MM_WEIGHT,
};

static ggml_tensor * laya_mm(ggml_context * ctx0, laya_precision precision, laya_mm_kind kind, ggml_tensor * a, ggml_tensor * b) {
    ggml_tensor * cur = ggml_mul_mat(ctx0, a, b);
    const bool pedantic = kind == LAYA_MM_ACT || precision == LAYA_PRECISION_STRICT;
    ggml_mul_mat_set_prec(cur, pedantic ? GGML_PREC_F32_PEDANTIC : GGML_PREC_F32);
    return cur;
}

// Precision fixes for the stock CUDA backend (Phase 4b step 6, localized with the layer trace). Both
// only add stock ops to the graph, and only when the compute device is CUDA: the CPU, Metal and
// Vulkan graphs are unchanged.
// - Weights (laya_cuda_upcast): ggml-cuda honours GGML_PREC_F32_PEDANTIC for an F32 src0 only. An F16
//   weight under PEDANTIC keeps cuBLAS in F16 compute, worse than GGML_PREC_F32, and a Q8_0 weight
//   goes through MMQ / MMVQ, which quantize the activations to 8-bit blocks: the massive-activation
//   channel of this model family (|x| ~ 1e4 from layer 10 on) then takes the precision of the
//   other 31 values of its block, and CUDA and the CPU (which quantizes too) fail in different
//   items. Such a weight is dequantized to F32 in the graph (exact for F16 and Q8_0), so the matmul
//   is F32 x F32: strict does it for F16 and Q8_0 weights, default for Q8_0 (F16 stays at
//   GGML_PREC_F32, which passes f16-class). The f16-class baseline of a Q8_0 run on CUDA is then the
//   CPU run with F32 activations (blas / auto). Other quantized types keep the CUDA kernels. The
//   dequantization is ggml_get_rows over all rows: ggml_cast would be a CPY, and the CUDA CPY kernel
//   for Q8_0 -> F32 runs one thread per block (it made a Q8_0 corpus run about 3x slower).
// - RoPE (laya_rope_host, strict only): ggml-cuda is built with -use_fast_math, so its rope kernel
//   takes sin / cos / pow from the fast intrinsics, whose error grows with the angle; in F32 the
//   layer-0 difference to the CPU grew about 5x from position 0 to 1000. Strict computes the
//   rotary tables on the host (laya_rope_tables) and applies them with mul / sub / add / concat.
// ggml-cuda registers as "CUDA", or "ROCm" / "MUSA" when built for HIP / MUSA: the same kernels (MMQ on
// quantized weights, PEDANTIC for an F32 src0 only), so the same graph changes; only CUDA has parity runs
// (laya_warn_unverified says so at load).
static bool laya_dev_is_cuda(ggml_backend_dev_t dev) {
    if (!dev || ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU) {
        return false;
    }
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
    const char * name = reg ? ggml_backend_reg_name(reg) : "";
    return strcmp(name, "CUDA") == 0 || strcmp(name, "ROCm") == 0 || strcmp(name, "MUSA") == 0;
}

static bool laya_cuda_upcast(laya_precision precision, ggml_type type) {
    return type == GGML_TYPE_Q8_0 || (type == GGML_TYPE_F16 && precision == LAYA_PRECISION_STRICT);
}

bool laya_weight_dequantized(ggml_backend_dev_t device, laya_precision precision, ggml_type type) {
    return laya_dev_is_cuda(device) && laya_cuda_upcast(precision, type);
}

// rotary tables as the PyTorch reference computes them in float32 (transformers ModernBERT:
// inv_freq = 1 / base^(arange(0, d, 2) / d), angle = pos * inv_freq, then cos / sin), one row of
// n_dims/2 values per token. The idea of host tables is from laya.cpp (MIT, Lars Karlslund).
static void laya_rope_tables(const laya_batch & batch, float freq_base, int n_dims,
                             std::vector<float> & cos_t, std::vector<float> & sin_t) {
    const int half = n_dims / 2;
    std::vector<float> inv_freq(half);
    for (int i = 0; i < half; ++i) {
        const float e = (float) (2*i) / (float) n_dims;
        inv_freq[i] = 1.0f / (float) std::pow((double) freq_base, (double) e);
    }
    cos_t.resize((size_t) half * batch.n_tokens);
    sin_t.resize((size_t) half * batch.n_tokens);
    for (int32_t t = 0; t < batch.n_tokens; ++t) {
        const float p = (float) batch.positions[t];
        for (int i = 0; i < half; ++i) {
            const float a = p * inv_freq[i];
            cos_t[(size_t) t*half + i] = (float) std::cos((double) a);
            sin_t[(size_t) t*half + i] = (float) std::sin((double) a);
        }
    }
}

// GPT-NeoX rotation of x [n_dims, n_head, n_tokens] with tables [n_dims/2, 1, n_tokens], in the
// order of the reference: out0 = x0*cos - x1*sin, out1 = x1*cos + x0*sin
static ggml_tensor * laya_rope_host(ggml_context * ctx0, ggml_tensor * x, ggml_tensor * cos_t, ggml_tensor * sin_t) {
    const int64_t half = x->ne[0] / 2;
    ggml_tensor * x0 = ggml_view_3d(ctx0, x, half, x->ne[1], x->ne[2], x->nb[1], x->nb[2], 0);
    ggml_tensor * x1 = ggml_view_3d(ctx0, x, half, x->ne[1], x->ne[2], x->nb[1], x->nb[2], half*ggml_element_size(x));
    ggml_tensor * o0 = ggml_sub(ctx0, ggml_mul(ctx0, x0, cos_t), ggml_mul(ctx0, x1, sin_t));
    ggml_tensor * o1 = ggml_add(ctx0, ggml_mul(ctx0, x1, cos_t), ggml_mul(ctx0, x0, sin_t));
    return ggml_concat(ctx0, o0, o1, 0);
}

// bidirectional multi-head attention without KV cache.
// q/k/v: [n_embd_head, n_head, n_tokens]; mask: [n_tokens, n_tokens, 1, 1]
// returns [n_embd, n_tokens]
static ggml_tensor * laya_attn(ggml_context * ctx0, laya_precision precision,
                               ggml_tensor * Qcur, ggml_tensor * Kcur, ggml_tensor * Vcur,
                               ggml_tensor * kq_mask, float kq_scale) {
    ggml_tensor * q = ggml_permute(ctx0, Qcur, 0, 2, 1, 3);   // [n_embd_head, n_tokens, n_head, 1]
    ggml_tensor * k = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);
    ggml_tensor * v = ggml_permute(ctx0, Vcur, 0, 2, 1, 3);

    ggml_tensor * kq = laya_mm(ctx0, precision, LAYA_MM_ACT, k, q); // [n_tokens, n_tokens, n_head, 1]
    kq = ggml_soft_max_ext(ctx0, kq, kq_mask, kq_scale, 0.0f);

    v = ggml_cont(ctx0, ggml_transpose(ctx0, v));             // [n_tokens, n_embd_head, n_head, 1]

    ggml_tensor * kqv = laya_mm(ctx0, precision, LAYA_MM_ACT, v, kq); // [n_embd_head, n_tokens, n_head, 1]
    ggml_tensor * cur = ggml_permute(ctx0, kqv, 0, 2, 1, 3);  // [n_embd_head, n_head, n_tokens, 1]
    cur = ggml_cont_2d(ctx0, cur, cur->ne[0]*cur->ne[1], cur->ne[2]*cur->ne[3]); // [n_embd, n_tokens]

    return cur;
}

// fused QKV projection + split into per-head views.
// qkv: [3*n_embd, n_tokens] (already projected); b applied by the caller
// returns contiguous per-head tensors [n_embd_head, n_head, n_tokens]
static std::array<ggml_tensor *, 3> laya_qkv_views(ggml_context * ctx0,
        ggml_tensor * qkv, int64_t n_embd_head, int64_t n_head, int64_t n_tokens) {
    const int64_t n_embd = n_embd_head * n_head;

    // extract contiguous Q / K / V blocks, then reshape into per-head views.
    // (a strided view of the fused projection feeds RoPE/attention with
    //  non-contiguous strides; materializing the blocks keeps every
    //  downstream op contiguous and matches the reference numerically)
    auto block = [&](int64_t offset) -> ggml_tensor * {
        ggml_tensor * t = ggml_view_2d(ctx0, qkv, n_embd, n_tokens, qkv->nb[1], offset * ggml_row_size(qkv->type, n_embd));
        t = ggml_cont(ctx0, t);
        t = ggml_reshape_3d(ctx0, t, n_embd_head, n_head, n_tokens);
        return t;
    };

    ggml_tensor * Qcur = block(0);
    ggml_tensor * Kcur = block(1);
    ggml_tensor * Vcur = block(2);

    return { Qcur, Kcur, Vcur };
}

// RoPE (GPT-NeoX) on [n_embd_head, n_head, n_tokens] tensors.
// ggml_rope_ext returns a new tensor, so the inputs are replaced in-place.
static void laya_rope(ggml_context * ctx0,
                      ggml_tensor ** Qcur, ggml_tensor ** Kcur,
                      ggml_tensor * inp_pos,
                      int n_dims, float freq_base, int n_ctx_orig) {
    *Qcur = ggml_rope_ext(ctx0, *Qcur, inp_pos, nullptr, n_dims, GGML_ROPE_TYPE_NEOX, n_ctx_orig,
                          freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
    *Kcur = ggml_rope_ext(ctx0, *Kcur, inp_pos, nullptr, n_dims, GGML_ROPE_TYPE_NEOX, n_ctx_orig,
                          freq_base, 1.0f, 0.0f, 1.0f, 32.0f, 1.0f);
}

// is layer `il` a sliding-window layer? pattern: dense_first (layer 0 dense,
// then every `pattern`-th layer dense)
static bool laya_is_swa(int32_t il, int32_t pattern) {
    if (pattern <= 0) {
        return false;
    }
    return il % pattern != 0;
}

// build the dense / swa attention masks, matching the PyTorch reference:
//   dense: 0.0 within the same sequence, -inf across sequences
//   swa:   additionally mask |p_query - p_key| > n_swa/2 + 1
static void laya_build_masks(const laya_batch & batch, int32_t n_swa,
                             std::vector<float> & mask, std::vector<float> & mask_swa) {
    const int64_t n = batch.n_tokens;
    mask.assign((size_t) n * n, -INFINITY);
    mask_swa.assign((size_t) n * n, -INFINITY);

    const int32_t half = n_swa / 2; // reference: config.sliding_window = local_attention//2 (HF mask: |p_q-p_k| <= half)

    for (int64_t i1 = 0; i1 < n; ++i1) {
        const int32_t s1 = batch.seq_id[i1];
        const int32_t p1 = batch.positions[i1];
        for (int64_t i0 = 0; i0 < n; ++i0) {
            if (batch.seq_id[i0] != s1) {
                continue;
            }
            const int32_t p0 = batch.positions[i0];
            mask[i1*n + i0] = 0.0f;
            if (half <= 0 || std::abs(p1 - p0) <= half) {
                mask_swa[i1*n + i0] = 0.0f;
            }
        }
    }
}

// ---- graph building -----------------------------------------------------

struct laya_graph {
    ggml_context_ptr ctx;
    ggml_cgraph * gf = nullptr;

    ggml_tensor * inp_tokens = nullptr;
    ggml_tensor * inp_pos    = nullptr;
    ggml_tensor * inp_qtype  = nullptr;
    ggml_tensor * kq_mask    = nullptr;
    ggml_tensor * kq_mask_swa = nullptr;
    ggml_tensor * marker_pos  = nullptr;
    ggml_tensor * marker_mask = nullptr;
    ggml_tensor * seq_start   = nullptr;
    ggml_tensor * rope_cos[2] = { nullptr, nullptr };  // host rotary tables (laya_rope_host): global, sliding-window layers
    ggml_tensor * rope_sin[2] = { nullptr, nullptr };
    std::vector<ggml_tensor *> row_ids;                // 0..n-1 for the weight dequantization (laya_cuda_upcast)

    ggml_tensor * logits     = nullptr;
    ggml_tensor * act_logits = nullptr;

};

static laya_graph laya_graph_build(const laya_model * model, const laya_batch & batch, laya_precision precision) {
    const auto & hp = model->hparams;
    const int64_t n_tokens = batch.n_tokens;
    const int64_t n_seqs   = batch.n_seqs;

    struct ggml_init_params params = {
        /*.mem_size =*/ ggml_tensor_overhead() * 16384,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };

    laya_graph g;
    g.ctx.reset(ggml_init(params));
    ggml_context * ctx0 = g.ctx.get();
    g.gf = ggml_new_graph_custom(ctx0, 8192, false);

    const bool cuda      = laya_dev_is_cuda(model->dev_gpu);
    const bool rope_host = cuda && precision == LAYA_PRECISION_STRICT;

    // weight matmul (src0 = w)
    std::map<int64_t, ggml_tensor *> row_ids;  // by row count
    const auto mm = [&](ggml_tensor * w, ggml_tensor * x) {
        if (cuda && laya_cuda_upcast(precision, w->type)) {
            ggml_tensor *& ids = row_ids[w->ne[1]];
            if (!ids) {
                ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, w->ne[1]);
                ggml_format_name(ids, "row_ids-%lld", (long long) w->ne[1]);
                ggml_set_input(ids);
                g.row_ids.push_back(ids);
            }
            w = ggml_get_rows(ctx0, w, ids);  // [ne0, ne1] F32
        }
        return laya_mm(ctx0, precision, LAYA_MM_WEIGHT, w, x);
    };

    const float kq_scale = 1.0f / sqrtf((float) hp.n_embd_head);
    const int   n_ctx_orig = 8192;

    // ---- inputs ----
    g.inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(g.inp_tokens, "inp_tokens");
    ggml_set_input(g.inp_tokens);

    g.inp_pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(g.inp_pos, "inp_pos");
    ggml_set_input(g.inp_pos);

    g.inp_qtype = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_name(g.inp_qtype, "inp_qtype");
    ggml_set_input(g.inp_qtype);

    g.kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(g.kq_mask, "kq_mask");
    ggml_set_input(g.kq_mask);

    g.kq_mask_swa = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_tokens, n_tokens);
    ggml_set_name(g.kq_mask_swa, "kq_mask_swa");
    ggml_set_input(g.kq_mask_swa);

    g.marker_pos = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, LAYA_MAX_MARKERS, n_seqs);
    ggml_set_name(g.marker_pos, "marker_pos");
    ggml_set_input(g.marker_pos);

    g.marker_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, LAYA_MAX_MARKERS, n_seqs);
    ggml_set_name(g.marker_mask, "marker_mask");
    ggml_set_input(g.marker_mask);

    g.seq_start = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_seqs);
    ggml_set_name(g.seq_start, "seq_start");
    ggml_set_input(g.seq_start);

    if (rope_host) {
        for (int k = 0; k < 2; ++k) {
            g.rope_cos[k] = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hp.n_embd_head / 2, 1, n_tokens);
            g.rope_sin[k] = ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, hp.n_embd_head / 2, 1, n_tokens);
            ggml_format_name(g.rope_cos[k], "rope_cos-%d", k);
            ggml_format_name(g.rope_sin[k], "rope_sin-%d", k);
            ggml_set_input(g.rope_cos[k]);
            ggml_set_input(g.rope_sin[k]);
        }
    }

    ggml_tensor * cur;
    ggml_tensor * inpL;

    // ---- input embeddings + embedding norm ----
    // the token embedding rows are stored in F16; the reference runs the whole
    // encoder in F32, so upcast the lookup to F32 before the embedding norm to
    // avoid F16 activation error (the near-one-hot attention amplifies it).
    inpL = ggml_cast(ctx0, ggml_get_rows(ctx0, model->tok_embd, g.inp_tokens), GGML_TYPE_F32); // [n_embd, n_tokens]
    if (model->tok_norm) {
        inpL = laya_norm(ctx0, inpL, model->tok_norm, nullptr, hp.norm_eps);
    }

    // ---- encoder (ModernBERT) ----
    for (int32_t il = 0; il < hp.n_layer; ++il) {
        const auto & layer = model->layers[il];
        const bool is_swa = hp.n_swa > 0 && laya_is_swa(il, hp.swa_pattern);
        const float freq_base = is_swa ? hp.rope_freq_base_swa : hp.rope_freq_base;

        cur = inpL;
        if (layer.attn_norm) {
            cur = laya_norm(ctx0, inpL, layer.attn_norm, nullptr, hp.norm_eps);
        }

        ggml_tensor * qkv_full = mm(layer.wqkv, cur); // [3*n_embd, n_tokens]
        auto qkv = laya_qkv_views(ctx0, qkv_full, hp.n_embd_head, hp.n_head, n_tokens);
        if (rope_host) {
            const int k = is_swa ? 1 : 0;
            qkv[0] = laya_rope_host(ctx0, qkv[0], g.rope_cos[k], g.rope_sin[k]);
            qkv[1] = laya_rope_host(ctx0, qkv[1], g.rope_cos[k], g.rope_sin[k]);
        } else {
            laya_rope(ctx0, &qkv[0], &qkv[1], g.inp_pos, hp.n_embd_head, freq_base, n_ctx_orig);
        }

        cur = laya_attn(ctx0, precision, qkv[0], qkv[1], qkv[2],
                        is_swa ? g.kq_mask_swa : g.kq_mask, kq_scale);
        cur = mm(layer.wo, cur);           // attn output projection (Wo)

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpL);

        cur = laya_norm(ctx0, ffn_inp, layer.ffn_norm, nullptr, hp.norm_eps);
        cur = mm(layer.ffn_up, cur);      // [2*n_ff, n_tokens]
        cur = ggml_geglu_erf(ctx0, cur);                  // GeGLU with erf GELU, matching PyTorch
        cur = mm(layer.ffn_down, cur);    // [n_embd, n_tokens]

        inpL = ggml_add(ctx0, cur, ffn_inp);
        ggml_format_name(inpL, "l_out-%d", il);  // traced (trace_dir): the residual output of the layer
    }

    if (model->output_norm) {
        inpL = laya_norm(ctx0, inpL, model->output_norm, nullptr, hp.norm_eps);
        ggml_set_name(inpL, "enc_out");
    }

    // ---- type embedding ----
    // h = h + type_emb[qtype] broadcast over every token
    inpL = ggml_add(ctx0, inpL, ggml_get_rows(ctx0, model->type_emb, g.inp_qtype));
    ggml_set_name(inpL, "type_emb_out");

    // ---- decision head transformer layers (norm_first, full self-attn) ----
    for (int32_t il = 0; il < hp.n_head_layers; ++il) {
        const auto & layer = model->head_layers[il];

        // attn pre-norm
        cur = laya_norm(ctx0, inpL, layer.attn_norm, layer.attn_norm_b, hp.norm_eps_layer);

        ggml_tensor * qkv_full = mm(layer.wqkv, cur);
        if (layer.wqkv_b) {
            qkv_full = ggml_add(ctx0, qkv_full, layer.wqkv_b);
        }
        auto qkv = laya_qkv_views(ctx0, qkv_full, hp.n_embd_head, hp.n_head, n_tokens);
        // full self-attention, no RoPE, pad mask only
        cur = laya_attn(ctx0, precision, qkv[0], qkv[1], qkv[2], g.kq_mask, kq_scale);

        cur = mm(layer.wo, cur);
        if (layer.wo_b) {
            cur = ggml_add(ctx0, cur, layer.wo_b);
        }

        ggml_tensor * res_inp = ggml_add(ctx0, cur, inpL);

        // ffn pre-norm + FFN (ReLU, like nn.TransformerEncoderLayer default)
        cur = laya_norm(ctx0, res_inp, layer.ffn_norm, layer.ffn_norm_b, hp.norm_eps_layer);
        cur = mm(layer.ffn_up, cur);      // [4*d, n_tokens]
        if (layer.ffn_up_b) {
            cur = ggml_add(ctx0, cur, layer.ffn_up_b);
        }
        cur = ggml_relu(ctx0, cur);
        cur = mm(layer.ffn_down, cur);    // [d, n_tokens]
        if (layer.ffn_down_b) {
            cur = ggml_add(ctx0, cur, layer.ffn_down_b);
        }

        inpL = ggml_add(ctx0, cur, res_inp);
        ggml_format_name(inpL, "head_out-%d", il);
    }

    // ---- gather hidden states at marker positions ----
    // ggml_get_rows only indexes the ne1 (row) dimension, so flatten the
    // [n_markers_max, n_seqs] position tensor to a single row-vector of
    // absolute token indices, then restore the 3D layout afterwards.
    // marker_pos flat index = m + n_markers_max * s, matching a row-major
    // [n_markers_max, n_seqs] input.
    ggml_tensor * marker_pos_flat = ggml_reshape_2d(ctx0, g.marker_pos, LAYA_MAX_MARKERS * n_seqs, 1);
    ggml_tensor * markers = ggml_get_rows(ctx0, inpL, marker_pos_flat);  // [n_embd, n_markers_max * n_seqs]
    markers = ggml_reshape_3d(ctx0, markers, hp.n_embd, LAYA_MAX_MARKERS, n_seqs);
    ggml_set_name(markers, "markers");

    // ---- scorer: LayerNorm -> Linear -> GELU -> Linear(1) ----
    cur = laya_norm(ctx0, markers, model->scorer_0, model->scorer_0_b, hp.norm_eps_layer);
    cur = mm(model->scorer_1, cur);
    if (model->scorer_1_b) {
        cur = ggml_add(ctx0, cur, model->scorer_1_b);
    }
    cur = ggml_gelu_erf(ctx0, cur);
    cur = mm(model->scorer_3, cur);       // [1, n_markers, n_seqs]
    if (model->scorer_3_b) {
        cur = ggml_add(ctx0, cur, model->scorer_3_b);
    }

    ggml_tensor * logits = ggml_reshape_2d(ctx0, cur, LAYA_MAX_MARKERS, n_seqs);
    ggml_set_name(logits, "logits");

    // mask invalid markers with -1e4: logits = where(mask, logits, -1e4)
    {
        ggml_tensor * mask_f = ggml_cast(ctx0, g.marker_mask, GGML_TYPE_F32);
        ggml_tensor * ones   = ggml_scale_bias(ctx0, mask_f, 0.0f, 1.0f);
        logits = ggml_add(ctx0,
                ggml_mul(ctx0, logits, mask_f),
                ggml_scale(ctx0, ggml_sub(ctx0, ones, mask_f), -1e4f));
        ggml_set_name(logits, "logits_masked");
    }

    // probabilities over markers (ne0)
    ggml_tensor * probs = ggml_soft_max(ctx0, logits);

    // ---- act head features ----
    // k = marker_mask.sum(-1).clamp(min=2) -> [1, n_seqs]
    ggml_tensor * k_f = ggml_cast(ctx0, g.marker_mask, GGML_TYPE_F32);
    ggml_tensor * k = ggml_clamp(ctx0, ggml_sum_rows(ctx0, k_f), 2.0f, FLT_MAX);

    // ent = -(p * log(p.clamp_min(1e-9))).sum(-1) / log(k)
    ggml_tensor * p_clamped = ggml_clamp(ctx0, probs, 1e-9f, FLT_MAX);
    ggml_tensor * ent = ggml_div(ctx0,
            ggml_neg(ctx0, ggml_sum_rows(ctx0, ggml_mul(ctx0, probs, ggml_log(ctx0, p_clamped)))),
            ggml_log(ctx0, k));

    // top-2 probability values per question (argsort desc along the marker dim)
    // argsort_top_k returns a strided view of the full argsort, make it contiguous first
    ggml_tensor * topk_idx = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, probs, 2)); // I32 [2, n_seqs]
    ggml_tensor * top1_idx = ggml_cont(ctx0, ggml_view_2d(ctx0, topk_idx, 1, n_seqs, topk_idx->nb[1], 0));
    ggml_tensor * top2_idx = ggml_cont(ctx0, ggml_view_2d(ctx0, topk_idx, 1, n_seqs, topk_idx->nb[1], ggml_element_size(topk_idx)));

    ggml_tensor * probs4 = ggml_reshape_4d(ctx0, probs, 1, LAYA_MAX_MARKERS, n_seqs, 1);
    ggml_tensor * b1 = ggml_reshape_4d(ctx0, top1_idx, 1, n_seqs, 1, 1);
    ggml_tensor * b2 = ggml_reshape_4d(ctx0, top2_idx, 1, n_seqs, 1, 1);

    ggml_tensor * top1 = ggml_view_1d(ctx0, ggml_get_rows(ctx0, probs4, b1), n_seqs, 0);
    ggml_tensor * top2 = ggml_view_1d(ctx0, ggml_get_rows(ctx0, probs4, b2), n_seqs, 0);

    // feats = [top1, top1-top2, ent, k/255] -> [4, n_seqs]
    ggml_tensor * f0 = ggml_reshape_2d(ctx0, top1, 1, n_seqs);
    ggml_tensor * f1 = ggml_reshape_2d(ctx0, ggml_sub(ctx0, top1, top2), 1, n_seqs);
    ggml_tensor * f2 = ggml_reshape_2d(ctx0, ent, 1, n_seqs);
    ggml_tensor * f3 = ggml_reshape_2d(ctx0, ggml_scale(ctx0, k, 1.0f/255.0f), 1, n_seqs);

    ggml_tensor * feats = ggml_concat(ctx0,
            ggml_concat(ctx0, ggml_concat(ctx0, f0, f1, 0), f2, 0), f3, 0);

    // pooled = h[:, seq_start] -> [n_embd, n_seqs]
    ggml_tensor * pooled = ggml_get_rows(ctx0, inpL, g.seq_start);

    // act input: concat(pooled, feats) -> [n_embd+4, n_seqs]
    ggml_tensor * act_in = ggml_concat(ctx0, pooled, feats, 0);

    // act head: Linear(d+4, 256) -> GELU -> Linear(256, n_act)
    cur = mm(model->act_head_0, act_in);
    if (model->act_head_0_b) {
        cur = ggml_add(ctx0, cur, model->act_head_0_b);
    }
    cur = ggml_gelu_erf(ctx0, cur);
    cur = mm(model->act_head_2, cur);
    if (model->act_head_2_b) {
        cur = ggml_add(ctx0, cur, model->act_head_2_b);
    }

    g.act_logits = cur;
    ggml_set_name(g.act_logits, "act_logits");

    g.logits = logits;
    // the two results are graph outputs: the allocator never frees or reuses the memory of an
    // output node (ggml-alloc.c, ggml_gallocr_free_node), so laya_encode reads them from the compute
    // buffer after the compute, without an output buffer and copy nodes of its own
    ggml_set_output(g.logits);
    ggml_set_output(g.act_logits);
    ggml_build_forward_expand(g.gf, g.logits);
    ggml_build_forward_expand(g.gf, g.act_logits);

    return g;
}

// nodes that compute nothing (a view of their source)
static bool laya_is_view_op(const ggml_tensor * t) {
    return t->op == GGML_OP_NONE || t->op == GGML_OP_VIEW || t->op == GGML_OP_RESHAPE ||
           t->op == GGML_OP_PERMUTE || t->op == GGML_OP_TRANSPOSE;
}

// a graph input, or a view of one
static bool laya_is_input(const ggml_tensor * t) {
    while (t->view_src) {
        t = t->view_src;
    }
    return (t->flags & GGML_TENSOR_FLAG_INPUT) != 0;
}

// may this node of a device context run on the CPU? (laya_placement)
static bool laya_cpu_allowed(const laya_model * model, const ggml_tensor * t) {
    const auto is_embd_rows = [&](const ggml_tensor * n) {
        return n && n->op == GGML_OP_GET_ROWS && n->src[0] == model->tok_embd;
    };
    if (is_embd_rows(t)) {
        return true;  // token_embd is in host memory by design (laya_model_params.device)
    }
    // ggml_cast is a CPY: the input casts (marker_mask to F32) and the upcast of the embedding rows
    return t->op == GGML_OP_CPY && t->src[0] && (laya_is_input(t->src[0]) || is_embd_rows(t->src[0]));
}

static void laya_placement_probe(laya_context * ctx) {
    const laya_model * model = ctx->model;
    ggml_backend_sched_t sched = ctx->sched.get();

    laya_probe_input in;
    laya_probe_input_init(model, 64, in);
    ggml_backend_sched_reset(sched);
    laya_graph g = laya_graph_build(model, in.batch, ctx->precision);
    if (!ggml_backend_sched_alloc_graph(sched, g.gf)) {
        throw std::runtime_error("failed to allocate the compute graph");
    }

    laya_placement p;
    p.device   = laya_device_name(model->dev_gpu);
    p.n_splits = ggml_backend_sched_get_n_splits(sched);
    p.host_weights = model->n_host_weights;
    const int n_backends = ggml_backend_sched_get_n_backends(sched);
    std::vector<int32_t> per_backend(n_backends, 0);
    int32_t n_unassigned = 0;
    std::map<std::string, int32_t> fallback;
    for (int i = 0; i < ggml_graph_n_nodes(g.gf); ++i) {
        ggml_tensor * t = ggml_graph_node(g.gf, i);
        if (laya_is_view_op(t)) {
            continue;
        }
        p.n_nodes++;
        ggml_backend_t b = ggml_backend_sched_get_tensor_backend(sched, t);
        int ib = 0;
        while (ib < n_backends && ggml_backend_sched_get_backend(sched, ib) != b) {
            ib++;
        }
        if (ib == n_backends) {
            n_unassigned++;
            continue;
        }
        per_backend[ib]++;
        if (ctx->backend_gpu && b == ctx->backend_cpu && !laya_cpu_allowed(model, t)) {
            p.cpu_fallback++;
            fallback[ggml_op_desc(t)]++;
        }
    }
    for (int ib = 0; ib < n_backends; ++ib) {
        p.backends.emplace_back(ggml_backend_name(ggml_backend_sched_get_backend(sched, ib)), per_backend[ib]);
    }
    if (n_unassigned > 0) {
        p.backends.emplace_back("none", n_unassigned);
    }
    for (const auto & f : fallback) {
        p.fallback_ops.push_back(f.first + " x" + std::to_string(f.second));
    }
    ggml_backend_sched_reset(sched);
    ctx->placement = std::move(p);
}

// the indices of a batch within the tensors they index (laya_encode)
static bool laya_batch_check(const laya_model * model, const laya_batch & batch) {
    const int64_t n_rows  = model->tok_embd->ne[1];
    const int64_t n_types = model->type_emb->ne[1];
    const int32_t n       = batch.n_tokens;
    for (int32_t i = 0; i < n; ++i) {
        if (batch.tokens[i] < 0 || batch.tokens[i] >= n_rows) {
            laya_log("laya_encode: token %d at %d is outside the %lld token_embd rows\n", batch.tokens[i], i, (long long) n_rows);
            return false;
        }
        if (batch.qtype[i] < 0 || batch.qtype[i] >= n_types) {
            laya_log("laya_encode: qtype %d at %d is outside the %lld type_emb rows\n", batch.qtype[i], i, (long long) n_types);
            return false;
        }
    }
    for (int32_t s = 0; s < batch.n_seqs; ++s) {
        if (batch.seq_start[s] < 0 || batch.seq_start[s] >= n) {
            laya_log("laya_encode: seq_start %d of sequence %d is outside [0, %d)\n", batch.seq_start[s], s, n);
            return false;
        }
        // every slot is gathered, masked ones too
        for (int32_t m = 0; m < LAYA_MAX_MARKERS; ++m) {
            const int32_t v = batch.marker_pos[(size_t) s * LAYA_MAX_MARKERS + m];
            if (v < 0 || v >= n) {
                laya_log("laya_encode: marker_pos %d (sequence %d, slot %d) is outside [0, %d)\n", v, s, m, n);
                return false;
            }
        }
    }
    return true;
}

int laya_encode(laya_context * ctx, const laya_batch & batch, laya_result & result) {
    const laya_model * model = ctx->model;

    if (batch.n_tokens <= 0 || batch.n_seqs <= 0) {
        return 1;
    }
    if (!laya_batch_check(model, batch)) {
        return 1;
    }

#if defined(__APPLE__)
    // the calling thread is thread 0 of the graph compute: keep it off the efficiency
    // cores when the machine is busy (a server's worker thread starts at QOS_CLASS_DEFAULT)
    static thread_local bool qos_done = false;
    if (ctx->qos && !qos_done) {
        qos_done = true;
        if ((int) qos_class_self() < (int) QOS_CLASS_USER_INITIATED) {
            pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
        }
    }
#endif

    ggml_backend_sched_reset(ctx->sched.get());
    laya_graph g = laya_graph_build(model, batch, ctx->precision);
    if (ctx->trace_index) {
        ctx->trace_call++;
        ctx->trace_seq = 0;
    }

    if (!ggml_backend_sched_alloc_graph(ctx->sched.get(), g.gf)) {
        laya_log("%s: failed to allocate compute graph\n", __func__);
        return 1;
    }

    // debug: verify input buffers
    {
        // with host rotary tables inp_pos feeds no op (rope_cos-0 takes its place)
        ggml_tensor * inps[] = { g.inp_tokens, g.rope_cos[0] ? g.rope_cos[0] : g.inp_pos, g.inp_qtype,
                                 g.kq_mask, g.kq_mask_swa, g.marker_pos, g.marker_mask, g.seq_start };
        for (auto * t : inps) {
            if (!t->buffer) {
                laya_log("%s: input tensor '%s' has no buffer\n", __func__, t->name);
                return 1;
            }
        }
    }

    // set inputs
    auto set_input = [&](ggml_tensor * t, const void * data) {
        ggml_backend_tensor_set(t, data, 0, ggml_nbytes(t));
    };

    set_input(g.inp_tokens, batch.tokens);
    if (!g.rope_cos[0]) {
        set_input(g.inp_pos, batch.positions);
    }
    set_input(g.inp_qtype,  batch.qtype);
    set_input(g.marker_pos,  batch.marker_pos);
    set_input(g.marker_mask, batch.marker_mask);
    set_input(g.seq_start,   batch.seq_start);

    std::vector<float> mask;
    std::vector<float> mask_swa;
    laya_build_masks(batch, model->hparams.n_swa, mask, mask_swa);
    set_input(g.kq_mask, mask.data());
    set_input(g.kq_mask_swa, mask_swa.data());

    for (ggml_tensor * ids : g.row_ids) {
        std::vector<int32_t> rows((size_t) ids->ne[0]);
        for (size_t i = 0; i < rows.size(); ++i) {
            rows[i] = (int32_t) i;
        }
        set_input(ids, rows.data());
    }

    if (g.rope_cos[0]) {
        const auto & hp = model->hparams;
        std::vector<float> cos_t, sin_t;
        for (int k = 0; k < 2; ++k) {
            if (!g.rope_cos[k]->buffer) {
                continue;  // a model without sliding-window layers uses only the global tables
            }
            laya_rope_tables(batch, k == 0 ? hp.rope_freq_base : hp.rope_freq_base_swa, hp.n_embd_head, cos_t, sin_t);
            set_input(g.rope_cos[k], cos_t.data());
            set_input(g.rope_sin[k], sin_t.data());
        }
    }

    const auto status = ggml_backend_sched_graph_compute(ctx->sched.get(), g.gf);
    if (status != GGML_STATUS_SUCCESS) {
        laya_log("%s: graph compute failed with status %d\n", __func__, (int) status);
        return 1;
    }


    result.n_markers_max = LAYA_MAX_MARKERS;
    result.n_seqs        = batch.n_seqs;
    result.n_act         = model->hparams.n_act;

    result.logits.resize((size_t) LAYA_MAX_MARKERS * batch.n_seqs);
    result.act_logits.resize((size_t) model->hparams.n_act * batch.n_seqs);

    GGML_ASSERT(g.logits->type == GGML_TYPE_F32 && ggml_nelements(g.logits) == (int64_t) result.logits.size());
    GGML_ASSERT(g.act_logits->type == GGML_TYPE_F32 && ggml_nelements(g.act_logits) == (int64_t) result.act_logits.size());
    GGML_ASSERT(ggml_is_contiguous(g.logits) && ggml_is_contiguous(g.act_logits));
    ggml_backend_tensor_get(g.logits, result.logits.data(), 0, ggml_nbytes(g.logits));
    ggml_backend_tensor_get(g.act_logits, result.act_logits.data(), 0, ggml_nbytes(g.act_logits));

    return 0;
}
