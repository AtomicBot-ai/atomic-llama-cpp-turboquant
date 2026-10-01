// tests for tools/laya/laya-convert (laya HF checkpoint -> GGUF without Python)
//
// usage: test-laya-convert <tests/laya/convert> <scratch-dir>
//
// 1. converts the tiny fixtures (tests/laya/make_tiny_hf_laya.py) and checks each GGUF against the sha256
//    of the Python converter's output (golden.sha256), also through non-ASCII paths
// 2. broken inputs: truncated / overflowing / inconsistent safetensors, missing files, unknown names,
//    non-finite q8_0 input, bad YAML, deep JSON, model-specific tokenizer classes, modules.json, a shard
//    index without model*.safetensors, generation_config.json that only Python reads; each must fail
//    with a clear error and leave no output behind. Big README.md inputs must stay fast (linear).
// 3. fp16 conversions and the gguf-py name heuristics on fixed cases (values from numpy / gguf-py)

#include "laya-convert.h"

#include <nlohmann/json.hpp>
extern "C" {
#include "sha256/sha256.h"
}

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

static int n_fail = 0;
static int n_pass = 0;

#define CHECK(cond, ...) do { \
    if (cond) { n_pass++; } else { n_fail++; fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } \
} while (0)

static fs::path u8(const std::string & s) {
#if defined(__cpp_lib_char8_t)
    return fs::path(std::u8string(s.begin(), s.end()));
#else
    return fs::u8path(s);
#endif
}

static std::string read_all(const std::string & path) {
    FILE * f = laya_convert_fopen(path, "rb");
    if (!f) {
        return std::string();
    }
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        s.append(buf, n);
    }
    fclose(f);
    return s;
}

static bool write_all(const std::string & path, const std::string & data) {
    FILE * f = laya_convert_fopen(path, "wb");
    if (!f) {
        return false;
    }
    const bool ok = fwrite(data.data(), 1, data.size(), f) == data.size();
    return fclose(f) == 0 && ok;
}

static std::string sha256_hex(const std::string & data) {
    unsigned char d[32];
    sha256_hash(d, (const unsigned char *) data.data(), data.size());
    std::string s;
    char h[3];
    for (unsigned char c : d) {
        snprintf(h, sizeof(h), "%02x", c);
        s += h;
    }
    return s;
}

static bool exists(const std::string & p) {
    std::error_code ec;
    return fs::exists(u8(p), ec);
}

static void copy_dir(const std::string & from, const std::string & to) {
    std::error_code ec;
    fs::remove_all(u8(to), ec);
    fs::create_directories(u8(to).parent_path(), ec);
    fs::copy(u8(from), u8(to), fs::copy_options::recursive, ec);
    if (ec) {
        fprintf(stderr, "copy %s -> %s failed: %s\n", from.c_str(), to.c_str(), ec.message().c_str());
        exit(1);
    }
}

// rewrite the JSON header of a safetensors file (data bytes kept)
static void edit_header(const std::string & path, const std::function<void(json &)> & fn) {
    std::string data = read_all(path);
    uint64_t n = 0;
    for (int i = 7; i >= 0; i--) {
        n = (n << 8) | (unsigned char) data[(size_t) i];
    }
    json h = json::parse(data.substr(8, (size_t) n));
    fn(h);
    std::string hs = h.dump();
    while (hs.size() % 8) {
        hs += ' ';
    }
    std::string out(8, '\0');
    for (int i = 0; i < 8; i++) {
        out[(size_t) i] = (char) ((hs.size() >> (8 * i)) & 0xFF);
    }
    write_all(path, out + hs + data.substr(8 + (size_t) n));
}

static std::string convert(const std::string & dir, const std::string & out, laya_convert_outtype t, const char * name = nullptr) {
    laya_convert_params p;
    p.outtype = t;
    if (name) {
        p.has_model_name = true;
        p.model_name = name;
    }
    return laya_convert(dir, out, p);
}

// a temporary file of out (out.<hex>.tmp) left in its directory
static std::string u8str(const fs::path & p) {
    const auto s = p.u8string(); // std::string in C++17, std::u8string in C++20
    return std::string(s.begin(), s.end());
}

static bool left_behind(const std::string & out) {
    const fs::path p = u8(out);
    const std::string prefix = u8str(p.filename()) + ".";
    std::error_code ec;
    for (const auto & e : fs::directory_iterator(p.parent_path(), ec)) {
        const std::string n = u8str(e.path().filename());
        if (n.compare(0, prefix.size(), prefix) == 0) {
            return true;
        }
    }
    return false;
}

static void expect_fail(const std::string & what, const std::string & dir, const std::string & out, const std::string & needle,
                        laya_convert_outtype t = LAYA_CONVERT_F16) {
    const std::string err = convert(dir, out, t);
    CHECK(!err.empty(), "%s: conversion succeeded, expected an error with '%s'", what.c_str(), needle.c_str());
    CHECK(err.find(needle) != std::string::npos, "%s: error '%s' does not contain '%s'", what.c_str(), err.c_str(), needle.c_str());
    CHECK(!exists(out) && !left_behind(out), "%s: output left behind", what.c_str());
    if (!err.empty()) {
        printf("  ok  %-34s -> %s\n", what.c_str(), err.c_str());
    }
}

static uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bits_f32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <tests/laya/convert> <scratch-dir>\n", argv[0]);
        return 1;
    }
    const std::string fixtures = argv[1];
    const std::string scratch  = argv[2];
    std::error_code ec;
    fs::remove_all(u8(scratch), ec);
    fs::create_directories(u8(scratch), ec);

    // ---- 1. golden outputs ----
    std::istringstream golden(read_all(fixtures + "/golden.sha256"));
    std::string line;
    int n_cases = 0;
    while (std::getline(golden, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::istringstream ls(line);
        std::string file, fixture, outtype, name, sha;
        uint64_t size = 0;
        ls >> file >> fixture >> outtype >> name >> size >> sha;
        laya_convert_outtype t;
        CHECK(laya_convert_parse_outtype(outtype, t), "bad outtype in golden.sha256: %s", outtype.c_str());
        const std::string out = scratch + "/" + file;
        const std::string err = convert(fixtures + "/" + fixture, out, t, name == "-" ? nullptr : name.c_str());
        CHECK(err.empty(), "%s: %s", file.c_str(), err.c_str());
        const std::string data = read_all(out);
        const std::string got = sha256_hex(data);
        CHECK(data.size() == size && got == sha, "%s: %zu bytes sha256 %s, golden %llu bytes %s", file.c_str(), data.size(), got.c_str(),
              (unsigned long long) size, sha.c_str());
        printf("  %s %-40s %s\n", got == sha ? "ok " : "BAD", file.c_str(), got.c_str());
        n_cases++;
    }
    CHECK(n_cases >= 7, "golden.sha256 has %d cases", n_cases);

    const std::string ms = fixtures + "/laya-tiny-ms-v0.1-8M";
    const std::string ms_f16 = scratch + "/laya-tiny-ms-v0.1-8M-f16.gguf";
    const std::string ms_q8 = scratch + "/laya-tiny-ms-v0.1-8M-q8_0.gguf";

    // non-ASCII output path and non-ASCII parent directory of the checkpoint (the directory name itself is kept)
    {
        const std::string out = scratch + "/\xD0\xB2\xD1\x8B\xD1\x85\xD0\xBE\xD0\xB4-\xC3\xBC-\xE2\x82\xAC.gguf"; // Cyrillic, Latin-1 and euro sign
        std::string err = convert(ms, out, LAYA_CONVERT_F16);
        CHECK(err.empty(), "non-ASCII output: %s", err.c_str());
        CHECK(read_all(out) == read_all(ms_f16), "non-ASCII output path gives different bytes");
        const std::string dir = scratch + "/\xD1\x82\xD0\xB5\xD1\x81\xD1\x82 \xE6\xB5\x8B\xE8\xAF\x95/laya-tiny-ms-v0.1-8M"; // Cyrillic and CJK, with a space
        copy_dir(ms, dir);
        const std::string out2 = scratch + "/nonascii-dir.gguf";
        err = convert(dir, out2, LAYA_CONVERT_F16);
        CHECK(err.empty(), "non-ASCII input dir: %s", err.c_str());
        CHECK(read_all(out2) == read_all(ms_f16), "non-ASCII input directory gives different bytes");
        printf("  ok  non-ASCII output path and input directory\n");
    }

    // compare: identical, and a one-byte change is found
    {
        CHECK(laya_convert_compare(ms_f16, ms_f16).empty(), "compare of a file with itself");
        std::string data = read_all(ms_f16);
        const std::string mod = scratch + "/modified.gguf";
        data[data.size() - 40] ^= 1;
        write_all(mod, data);
        const std::string diff = laya_convert_compare(ms_f16, mod);
        CHECK(diff.find("data differs") != std::string::npos, "compare: '%s'", diff.c_str());
        CHECK(!laya_convert_compare(ms_f16, ms_q8).empty(), "compare f16 vs q8_0");
        printf("  ok  compare: %s\n", diff.c_str());
    }

    // ---- 2. broken inputs ----
    const std::string bad = scratch + "/bad/laya-tiny-ms-v0.1-8M";
    const std::string out = scratch + "/bad-out.gguf";
    const std::string st = bad + "/model.safetensors";
    auto fresh = [&]() { copy_dir(ms, bad); };

    fresh();
    { std::string d = read_all(st); write_all(st, d.substr(0, 5)); }
    expect_fail("safetensors shorter than 8 bytes", bad, out, "truncated safetensors");

    fresh();
    { std::string d = read_all(st); write_all(st, d.substr(0, d.size() / 2)); }
    expect_fail("truncated tensor data", bad, out, "outside the data section");

    fresh();
    { std::string d = read_all(st); for (int i = 0; i < 8; i++) d[(size_t) i] = (char) 0xFF; write_all(st, d); }
    expect_fail("header length overflow", bad, out, "exceeds the file size");

    fresh();
    { std::string d = read_all(st); d[8] = '['; write_all(st, d); }
    expect_fail("header is not JSON", bad, out, "invalid JSON");

    fresh();
    edit_header(st, [](json & h) { h["temperature"]["shape"] = json::array({ 4 }); });
    expect_fail("dtype / shape / bytes mismatch", bad, out, "needs 16 bytes, data_offsets give 12");

    fresh();
    edit_header(st, [](json & h) { h["temperature"]["dtype"] = "I32"; });
    expect_fail("unsupported dtype", bad, out, "dtype I32 is not supported");

    fresh();
    edit_header(st, [](json & h) { h["temperature"]["dtype"] = "Q9"; });
    expect_fail("unknown dtype", bad, out, "unknown dtype");

    fresh();
    edit_header(st, [](json & h) { h["temperature"]["data_offsets"] = json::array({ 0, (uint64_t) -1 }); });
    expect_fail("data_offsets overflow", bad, out, "outside the data section");

    fresh();
    edit_header(st, [](json & h) { h["temperature"]["shape"] = json::array({ 3, 0 }); });
    expect_fail("zero dimension", bad, out, "bad dimension 0");

    fresh();
    edit_header(st, [](json & h) { h["temperature"]["shape"] = json::array({ 1ull << 40, 1ull << 40 }); });
    expect_fail("element count overflow", bad, out, "too many elements");

    fresh();
    edit_header(st, [](json & h) { h["temperature"]["data_offsets"] = json::array({ 12, 0 }); });
    expect_fail("data_offsets begin > end", bad, out, "outside the data section");

    fresh();
    edit_header(st, [](json & h) { h["scorer.9.bias"] = h["scorer.0.bias"]; h.erase("scorer.0.bias"); });
    expect_fail("unknown tensor name", bad, out, "Can not map tensor 'scorer.9.bias'");

    fresh();
    edit_header(st, [](json & h) { h["type_emb.weight"] = h["encoder.embeddings.norm.weight"]; h["encoder.type_emb.weight"] = h["scorer.0.bias"]; });
    expect_fail("duplicate after encoder. strip", bad, out, "duplicate tensor 'type_emb.weight'");

    fresh();
    fs::remove(u8(bad + "/tokenizer/tokenizer.json"), ec);
    expect_fail("missing tokenizer.json", bad, out, "missing tokenizer file");

    fresh();
    fs::remove(u8(bad + "/rl_agent_config.json"), ec);
    expect_fail("missing rl_agent_config.json", bad, out, "not a laya checkpoint");

    fresh();
    write_all(bad + "/config.json", "{}");
    expect_fail("root config.json", bad, out, "config.json in the checkpoint root");

    fresh();
    fs::remove(u8(st), ec);
    expect_fail("no safetensors", bad, out, "no model*.safetensors");

    fresh();
    write_all(bad + "/README.md", "---\ndescription: laya: a model\n---\n");
    expect_fail("YAML error", bad, out, "mapping values are not allowed");

    fresh();
    write_all(bad + "/README.md", "---\nlicense:\n  id: mit\n---\n");
    expect_fail("YAML beyond the subset", bad, out, "unsupported YAML for 'license'");

    fresh();
    write_all(bad + "/README.md", "---\nversion: 1.0\n---\n");
    expect_fail("non-string README value", bad, out, "'version' is not a string");

    fresh();
    {
        std::string d = read_all(bad + "/tokenizer/tokenizer_config.json");
        json tc = json::parse(d);
        tc["mask_token"] = "<not-there>";
        write_all(bad + "/tokenizer/tokenizer_config.json", tc.dump());
    }
    expect_fail("special token not in vocab", bad, out, "is not an added token");

    // vocab_size drives the [PAD<i>] padding: an absurd value must fail fast, not allocate
    for (const uint64_t vs : { (uint64_t) 4294967295u, (uint64_t) 65 }) {
        fresh();
        const std::string cp = bad + "/encoder/config.json";
        json cfg = json::parse(read_all(cp));
        cfg["vocab_size"] = vs;
        write_all(cp, cfg.dump());
        expect_fail("vocab_size " + std::to_string(vs), bad, out, vs == 65 ? "rows of token_embd.weight" : "is absurd");
    }

    // what the tokenizers library (Python's loader) rejects must not convert here (found by fuzzing)
    {
        const std::string tp = bad + "/tokenizer/tokenizer.json";
        const std::vector<std::pair<std::string, std::function<void(json &)>>> cases = {
            { "added token without normalized", [](json & t) { t["added_tokens"][0].erase("normalized"); } },
            { "added token special = 1",        [](json & t) { t["added_tokens"][0]["special"] = 1; } },
            { "template without pair",          [](json & t) { t["post_processor"].erase("pair"); } },
            { "template ids > u32",             [](json & t) { t["post_processor"]["special_tokens"].begin().value()["ids"][0] = (uint64_t) -1; } },
            { "template special_tokens no id",  [](json & t) { t["post_processor"]["special_tokens"].begin().value().erase("id"); } },
            { "template piece type_id float",   [](json & t) { t["post_processor"]["single"][0]["SpecialToken"]["type_id"] = 0.5; } },
        };
        for (const auto & tc : cases) {
            fresh();
            json t = json::parse(read_all(tp));
            tc.second(t);
            write_all(tp, t.dump());
            expect_fail(tc.first, bad, out, "the tokenizers library rejects it");
        }
    }
    fresh();
    write_all(bad + "/README.md", "---\nlicense: mit\nmodel-index:\n  -\x12name: x\n---\n");
    expect_fail("YAML control character", bad, out, "unacceptable character #x0012");

    fresh();
    {
        const std::string cp = bad + "/encoder/config.json";
        json cfg = json::parse(read_all(cp));
        cfg["num_hidden_layers"] = 2147483647;
        write_all(cp, cfg.dump());
        expect_fail("num_hidden_layers 2^31-1", bad, out, "block_count 2147483647 is absurd");
    }

    fresh();
    {
        // inf in a q8_0 tensor (scorer.1.weight, F16, 32 columns)
        std::string d = read_all(st);
        uint64_t n = 0;
        for (int i = 7; i >= 0; i--) n = (n << 8) | (unsigned char) d[(size_t) i];
        json h = json::parse(d.substr(8, (size_t) n));
        const uint64_t off = 8 + n + h["scorer.1.weight"]["data_offsets"][0].get<uint64_t>();
        d[(size_t) off] = 0x00;
        d[(size_t) off + 1] = 0x7C;
        write_all(st, d);
        expect_fail("non-finite q8_0 input", bad, out, "non-finite value", LAYA_CONVERT_Q8_0);
        CHECK(convert(bad, out, LAYA_CONVERT_F16).empty(), "f16 of a tensor with inf must work");
        laya_convert_remove(out);
    }

    // AutoTokenizer picks a model-specific class from tokenizer_class, else from tokenizer/config.json
    {
        const std::string tcp = bad + "/tokenizer/tokenizer_config.json";
        auto edit_tc = [&](const std::function<void(json &)> & fn) {
            json tc = json::parse(read_all(tcp));
            fn(tc);
            write_all(tcp, tc.dump());
        };
        fresh();
        edit_tc([](json & tc) { tc["tokenizer_class"] = "BertTokenizer"; });
        expect_fail("tokenizer_class BertTokenizer", bad, out, "tokenizer_class \"BertTokenizer\" is not supported");
        fresh();
        edit_tc([](json & tc) { tc["tokenizer_class"] = 5; });
        expect_fail("tokenizer_class not a string", bad, out, "tokenizer_class 5 is not supported");
        fresh();
        edit_tc([](json & tc) { tc.erase("tokenizer_class"); });
        write_all(bad + "/tokenizer/config.json", "{\"model_type\": \"bert\"}");
        expect_fail("tokenizer/config.json model_type", bad, out, "'model_type' without tokenizer_config.tokenizer_class");
        fresh();
        edit_tc([](json & tc) { tc.erase("tokenizer_class"); });
        write_all(bad + "/tokenizer/config.json", "{\"tokenizer_class\": \"BertTokenizer\"}");
        expect_fail("tokenizer/config.json tokenizer_class", bad, out, "'tokenizer_class' without tokenizer_config.tokenizer_class");
        fresh();
        edit_tc([](json & tc) { tc["auto_map"] = { { "AutoTokenizer", json::array({ "x.X", nullptr }) } }; });
        expect_fail("tokenizer_config auto_map", bad, out, "auto_map");
        // TokenizersBackend and an absent tokenizer_class are the class this port implements
        for (const char * cls : { "TokenizersBackend", "" }) {
            fresh();
            edit_tc([&](json & tc) { if (*cls) tc["tokenizer_class"] = cls; else tc.erase("tokenizer_class"); });
            const std::string o = scratch + "/cls.gguf";
            CHECK(convert(bad, o, LAYA_CONVERT_F16).empty() && read_all(o) == read_all(ms_f16), "tokenizer_class '%s' must convert to the same bytes", cls);
            laya_convert_remove(o);
        }
        printf("  ok  tokenizer_class checks\n");
    }

    fresh();
    write_all(bad + "/modules.json", "[{\"type\": \"sentence_transformers.models.Pooling\", \"path\": \"../../x\"}]");
    expect_fail("modules.json", bad, out, "modules.json");

    // shards named only by the index: Python needs some model*.safetensors first
    {
        const std::string bl  = fixtures + "/laya-bl-tiny-instruct-30K";
        const std::string bad2 = scratch + "/bad2/laya-bl-tiny-instruct-30K";
        copy_dir(bl, bad2);
        std::string idx = read_all(bad2 + "/model.safetensors.index.json");
        for (const char * part : { "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" }) {
            const std::string np = std::string("weights") + (part + 5);
            fs::rename(u8(bad2 + "/" + part), u8(bad2 + "/" + np), ec);
            for (size_t k; (k = idx.find(part)) != std::string::npos;) {
                idx.replace(k, strlen(part), np);
            }
        }
        write_all(bad2 + "/model.safetensors.index.json", idx);
        expect_fail("index without model*.safetensors", bad2, out, "no model*.safetensors");
    }

    // deep JSON: refused before anything copies or dumps it (a copy or dump() recurses per level)
    {
        const std::string tp = bad + "/tokenizer/tokenizer.json";
        fresh();
        {
            json t = json::parse(read_all(tp));
            json n = { { "type", "Lowercase" } };
            for (int k = 0; k < 200; k++) {
                n = { { "type", "Sequence" }, { "normalizers", json::array({ n }) } };
            }
            t["normalizer"] = n;
            write_all(tp, t.dump());
        }
        expect_fail("normalizer nested 400 levels", bad, out, "levels deep (limit 127)");
        fresh();
        {
            std::string t = read_all(tp);
            const size_t brace = t.rfind('}');
            std::string deep = ", \"x\": " + std::string(1000000, '[') + std::string(1000000, ']');
            t.insert(brace, deep);
            write_all(tp, t);
        }
        expect_fail("tokenizer.json nested 10^6 levels", bad, out, "levels deep (limit 127)");
        fresh();
        edit_header(st, [](json & h) {
            json sh = json::array({ 3 });
            for (int k = 0; k < 300; k++) sh = json::array({ sh });
            h["temperature"]["shape"] = sh;
        });
        expect_fail("safetensors shape nested 301 levels", bad, out, "levels deep (limit 127)");
    }

    // generation_config.json: Python ignores a syntax error or a BOM, but reads NaN / Infinity / lone surrogates
    {
        const std::string gp = bad + "/generation_config.json";
        const std::pair<const char *, const char *> refused[] = {
            { "{\"temperature\": NaN}", "NaN / Infinity" },
            { "{\"top_p\": -Infinity}", "NaN / Infinity" },
            { "{\"sequence\": \"\\ud800\"}", "lone surrogate" },
        };
        for (const auto & r : refused) {
            fresh();
            write_all(gp, r.first);
            expect_fail(std::string("generation_config ") + r.first, bad, out, r.second);
        }
        const char * ignored[] = { "{\"temperature\": 0.5", "\xEF\xBB\xBF{\"temperature\": 0.5}", "not json at all" };
        for (const char * g : ignored) {
            fresh();
            write_all(gp, g);
            const std::string o = scratch + "/gen.gguf";
            CHECK(convert(bad, o, LAYA_CONVERT_F16).empty() && read_all(o) == read_all(ms_f16), "generation_config '%s' must be ignored", g);
            laya_convert_remove(o);
        }
        fresh();
        write_all(gp, "{\"x\": " + std::string(1000, '[') + std::string(1000, ']') + "}");
        expect_fail("generation_config nested 1001 levels", bad, out, "levels deep");
        printf("  ok  generation_config.json parse rules\n");
    }

    // big README.md: universal newlines and the YAML keys must stay linear (they were quadratic)
    {
        const std::string rp = bad + "/README.md";
        const std::string readme = read_all(ms + "/README.md");
        fresh();
        std::string crlf;
        for (int k = 0; k < 4000000; k++) crlf += "\r\n";
        write_all(rp, readme + crlf);
        auto t0 = std::chrono::steady_clock::now();
        const std::string o = scratch + "/big-readme.gguf";
        std::string err = convert(bad, o, LAYA_CONVERT_F16);
        double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        CHECK(err.empty() && read_all(o) == read_all(ms_f16), "README with 4M CRLF: %s", err.c_str());
        CHECK(secs < 20.0, "README with 4M CRLF took %.1f s", secs);
        printf("  ok  README.md with 4M CRLF line ends: %.2f s\n", secs);
        laya_convert_remove(o);

        fresh();
        std::string keys;
        for (int k = 0; k < 300000; k++) {
            char b[32];
            snprintf(b, sizeof(b), "k%07d: x\n", k);
            keys += b;
        }
        write_all(rp, "---\n" + keys + readme.substr(4));
        t0 = std::chrono::steady_clock::now();
        err = convert(bad, o, LAYA_CONVERT_F16);
        secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        CHECK(err.empty() && read_all(o) == read_all(ms_f16), "README with 300k YAML keys: %s", err.c_str());
        CHECK(secs < 20.0, "README with 300k YAML keys took %.1f s", secs);
        printf("  ok  README.md with 300k YAML keys: %.2f s\n", secs);
        laya_convert_remove(o);
    }

    // compare: an unreadable file is an error, not a difference
    {
        bool failed = false;
        std::string r = laya_convert_compare(scratch + "/no-such.gguf", ms_f16, &failed);
        CHECK(failed && r.find("cannot open") != std::string::npos, "compare of a missing file: failed=%d '%s'", (int) failed, r.c_str());
        write_all(scratch + "/not.gguf", "not a gguf");
        r = laya_convert_compare(ms_f16, scratch + "/not.gguf", &failed);
        CHECK(failed, "compare with a non-GGUF: failed=%d '%s'", (int) failed, r.c_str());
        r = laya_convert_compare(ms_f16, ms_q8, &failed);
        CHECK(!failed && !r.empty(), "compare f16 vs q8_0: failed=%d '%s'", (int) failed, r.c_str());
        printf("  ok  compare errors vs differences\n");
    }

    // the temporary file is created exclusively: an existing name (or symlink) is never written through
    {
        const std::string o = scratch + "/excl.gguf";
        CHECK(convert(ms, o, LAYA_CONVERT_F16).empty() && read_all(o) == read_all(ms_f16), "conversion over an existing output");
        CHECK(!left_behind(o), "temporary file left after a conversion");
        laya_convert_remove(o);
    }

    expect_fail("output directory missing", ms, scratch + "/no/such/dir/out.gguf", "cannot create");

    // ---- 3. fp16 and name heuristics ----
    {
        // numpy astype(float16) on AArch64 / x86-64 F16C
        const uint32_t f2h[][2] = {
            { 0x7F800001, 0x7E00 }, { 0x7FC00000, 0x7E00 }, { 0xFFC00001, 0xFE00 }, { 0x7F802000, 0x7E01 },
            { 0x7FBFFFFF, 0x7FFF }, { 0x7F800000, 0x7C00 }, { 0x477FF000, 0x7C00 }, { 0x477FEFFF, 0x7BFF },
            { 0x33000000, 0x0000 }, { 0x33000001, 0x0001 }, { 0x387FC000, 0x03FF }, { 0x00000001, 0x0000 },
            { 0x80000000, 0x8000 }, { 0x3F801000, 0x3C00 }, { 0x3F803000, 0x3C02 }, { 0xB3400000, 0x8001 },
        };
        for (const auto & e : f2h) {
            const uint16_t h = laya_convert_fp32_to_fp16(bits_f32(e[0]));
            CHECK(h == e[1], "fp32_to_fp16(0x%08x) = 0x%04x, numpy 0x%04x", e[0], h, e[1]);
        }
        const uint32_t h2f[][2] = {
            { 0x7C01, 0x7FC02000 }, { 0xFE3F, 0xFFC7E000 }, { 0x0001, 0x33800000 }, { 0x03FF, 0x387FC000 },
            { 0x7C00, 0x7F800000 }, { 0x8000, 0x80000000 }, { 0x3C00, 0x3F800000 }, { 0x7BFF, 0x477FE000 },
        };
        for (const auto & e : h2f) {
            const uint32_t f = f32_bits(laya_convert_fp16_to_fp32((uint16_t) e[0]));
            CHECK(f == e[1], "fp16_to_fp32(0x%04x) = 0x%08x, numpy 0x%08x", e[0], f, e[1]);
        }
        printf("  ok  fp16 conversions\n");
    }
    {
        // gguf-py Metadata.get_model_id_components + id_to_title: name|basename|finetune|version|size_label
        struct { const char * id; int64_t n; const char * want; } cases[] = {
            { "laya", 421293827, "Laya|-|-|-|-" },
            { "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851", 421293827, "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851|-|55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851|-|-" },
            { "e4e9ddf21a7b1903b7acffd8814ad4307bf63a67", 421293827, "E4E9Ddf21A7B1903B7Acffd8814Ad4307Bf63A67|-|-|-|-" },
            { "Mistral-7B-Instruct-v0.2", 421293827, "Mistral 7B Instruct v0.2|Mistral|7b-Instruct|v0.2|-" },
            { "bloom-7b1", 421293827, "Bloom 7b1|bloom|7.1b|-|-" },
            { "Qwen2.5-0.5B-Instruct-GGUF-q4_k_m", 421293827, "Qwen2.5 0.5B Instruct GGUF Q4_K_M|Qwen2.5|Instruct-GGUF|-|0.5B" },
            { "Meta-Llama-3.1-8B", 421293827, "Meta Llama 3.1 8B|Meta-Llama-3.1|8b|-|-" },
            { "mixtral-8x7b-v0.1-f16", 421293827, "Mixtral 8x7b v0.1 F16|mixtral|-|v0.1|8x7B" },
            { "gpt2-small", 421293827, "Gpt2 Small|gpt2|-|-|small" },
            { "my model", 421293827, "My Model|-|-|-|-" },
            { "phi-3-mini-128k-instruct", 421293827, "Phi 3 Mini 128k Instruct|phi-3|128k-instruct|-|mini" },
            { "SmolLM2-135M", 421293827, "SmolLM2 135M|SmolLM2|-|-|135M" },
            { "Llama-2-7b-chat-hf", 421293827, "Llama 2 7b Chat Hf|Llama-2|7b-chat-hf|-|-" },
            { "iter100-model", 421293827, "Iter100 Model|-|-|-|-" },
            { "v2-base", 421293827, "v2 Base|-|-|-|-" },
            { "model-A3B-400M", 421293827, "Model A3B 400M|model|-|-|A3B-400M" },
            { "x--y--1.5K", 421293827, "X Y 1.5K|x-y|1.5k|-|-" },
            { "deepseek-coder-6.7b-instruct", 421293827, "Deepseek Coder 6.7b Instruct|deepseek-coder|6.7b-instruct|-|-" },
            { "OLMo-2-1124-7B", 421293827, "OLMo 2 1124 7B|OLMo-2-1124|7b|-|-" },
            { "bert_base_uncased-512", 421293827, "Bert_Base_Uncased 512|bert_base_uncased|-|512|-" },
            { "Qwen2.5-0.5B-Instruct-GGUF-q4_k_m", 32000, "Qwen2.5 0.5B Instruct GGUF Q4_K_M|Qwen2.5|0.5b-Instruct-GGUF|-|-" },
            { "SmolLM2-135M", 32000, "SmolLM2 135M|SmolLM2|135m|-|-" },
            { "laya-bl-tiny-instruct-30K", 32000, "Laya Bl Tiny Instruct 30K|laya-bl-tiny|instruct|-|30K" },
            { "model-A3B-400M", 32000, "Model A3B 400M|model|400m|-|A3B" },
        };
        int n_ok = 0;
        for (const auto & c : cases) {
            const std::string got = laya_convert_dir_metadata(c.id, c.n);
            CHECK(got == c.want, "dir metadata '%s' (%lld): '%s', gguf-py '%s'", c.id, (long long) c.n, got.c_str(), c.want);
            n_ok += got == c.want;
        }
        printf("  ok  name heuristics %d/%zu\n", n_ok, sizeof(cases) / sizeof(cases[0]));
    }

    fs::remove_all(u8(scratch), ec);
    printf("%s: %d checks passed, %d failed\n", n_fail ? "FAILED" : "PASSED", n_pass, n_fail);
    return n_fail ? 1 : 0;
}
