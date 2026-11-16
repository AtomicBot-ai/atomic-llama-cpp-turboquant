#include "testing.h"
#include "../src/llama-expert-source.h"
#include "ggml-cpp.h"

#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace fs = std::filesystem;

struct fixture {
    fs::path dir;

    fixture() {
        const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        dir = fs::temp_directory_path() / ("test-expert-source-" + std::to_string(now));
        if (!fs::create_directory(dir)) {
            throw std::runtime_error("cannot create fixture directory");
        }
    }

    ~fixture() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    std::string path(int shard) const { return (dir / (std::to_string(shard) + ".gguf")).string(); }

    void write(int shard, ggml_type type, const char * name, int split_count = 2,
               int tensor_count = 2, const char * arch = "qwen3moe", bool rank4 = false, bool dense = false) {
        ggml_context_ptr ctx(ggml_init({1024 * 1024, nullptr, false}));
        if (!ctx) {
            throw std::runtime_error("fixture allocation failed");
        }
        ggml_tensor * tensor = ggml_new_tensor_4d(ctx.get(), type, 256, 3, dense ? 1 : 4, rank4 ? 2 : 1);
        ggml_set_name(tensor, name);
        auto * bytes = static_cast<uint8_t *>(tensor->data);
        for (size_t i = 0; i < ggml_nbytes(tensor); ++i) {
            bytes[i] = uint8_t(i * 17 + (i >> 8) + shard * 97);
        }
        gguf_context_ptr meta(gguf_init_empty());
        if (arch != nullptr) {
            gguf_set_val_str(meta.get(), "general.architecture", arch);
        }
        // Vary header size between shards to expose accidental reuse of the
        // first shard's data offset when reading the second shard.
        gguf_set_val_str(meta.get(), "fixture.padding", std::string(shard * 128, 'x').c_str());
        if (split_count > 0) {
            gguf_set_val_u16(meta.get(), "split.count", split_count);
            gguf_set_val_u16(meta.get(), "split.no", shard);
            gguf_set_val_i32(meta.get(), "split.tensors.count", tensor_count);
        }
        gguf_add_tensor(meta.get(), tensor);
        if (!gguf_write_to_file(meta.get(), path(shard).c_str(), false)) {
            throw std::runtime_error("cannot write fixture");
        }
    }

    std::vector<std::string> paths() const { return {path(0), path(1)}; }
};

template <typename F>
static void rejects(testing & t, const std::string & label, F fn) {
    bool rejected = false;
    try {
        fn();
    } catch (const std::exception &) {
        rejected = true;
    }
    t.assert_true(label, rejected);
}

static void compare_loader(testing & t, llama_expert_source & source, const fixture & f) {
    for (int shard = 0; shard < 2; ++shard) {
        ggml_context * raw = nullptr;
        gguf_context_ptr meta(gguf_init_from_file(f.path(shard).c_str(), {false, &raw}));
        ggml_context_ptr loaded(raw);
        if (!meta || !loaded) {
            throw std::runtime_error("reference GGUF load failed");
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(meta.get()); ++i) {
            ggml_tensor * tensor = ggml_get_tensor(loaded.get(), gguf_get_tensor_name(meta.get(), i));
            if (!tensor) {
                throw std::runtime_error("missing reference tensor");
            }
            const auto & info = source.tensor(tensor->name);
            t.assert_equal("source shard", size_t(shard), info.shard);
            t.assert_equal("original quantization", int(tensor->type), int(info.type));
            t.assert_equal("expert stride", tensor->nb[2], info.expert_bytes());
            for (int64_t expert : {int64_t(0), int64_t(1), tensor->ne[2] - 1}) {
                std::vector<uint8_t> data(info.expert_bytes() + 2, 0xA5);
                source.read_expert(info.name, expert, data.data() + 1, info.expert_bytes());
                t.assert_true("exact original expert bytes", std::memcmp(data.data() + 1,
                    static_cast<uint8_t *>(tensor->data) + expert * tensor->nb[2], info.expert_bytes()) == 0);
                t.assert_true("destination guards", data.front() == 0xA5 && data.back() == 0xA5);
            }
        }
    }
}

int main(int argc, char ** argv) {
    testing t;
    if (argc == 2) {
        t.set_filter(argv[1]);
    }
    t.test("advisory reads preserve checked demand", [&](testing & t) {
        fixture f;
        f.write(0, GGML_TYPE_IQ4_XS, "gate");
        f.write(1, GGML_TYPE_Q4_K, "down", 2, 2, nullptr);
        llama_expert_source source(f.paths());
        t.assert_true("negative hint declined", !source.advise_expert("gate", -1));
        t.assert_true("past-end hint declined", !source.advise_expert("gate", 4));
        t.assert_true("unknown hint declined", !source.advise_expert("missing", 0));
        for (int64_t expert : {3, 0, 2, 1}) {
            // Unsupported filesystems may decline; the synchronous fallback
            // must return the same original bytes in either case.
            source.advise_expert("down", expert);
            source.advise_expert("gate", expert);
        }
        compare_loader(t, source, f);
        const size_t bytes = source.tensor("down").expert_bytes();
        std::vector<uint8_t> data(bytes);
        fs::resize_file(f.path(1), 0);
        source.advise_expert("down", 0);
        rejects(t, "hint cannot bypass file validation", [&] {
            source.read_expert("down", 0, data.data(), data.size());
        });
        t.assert_true("poisoned source declines advice", !source.advise_expert("gate", 0));
    });
    t.test("packed bytes across shards", [&](testing & t) {
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_XS, GGML_TYPE_Q5_K}) {
            fixture f;
            f.write(0, type, "blk.0.ffn_gate_exps.weight");
            f.write(1, GGML_TYPE_F16, "blk.1.ffn_gate_up_exps.weight", 2, 2, nullptr);
            llama_expert_source source(f.paths());
            t.assert_equal("architecture", std::string("qwen3moe"), source.architecture());
            compare_loader(t, source, f);
        }
    });
    t.test("argument validation and recovery", [&](testing & t) {
        fixture f;
        const std::string name = "blk.0.ffn_up_exps.weight";
        f.write(0, GGML_TYPE_Q4_K, name.c_str(), 0);
        llama_expert_source source({f.path(0)});
        std::vector<uint8_t> data(source.tensor(name).expert_bytes(), 0xA5);
        rejects(t, "negative expert", [&] { source.read_expert(name, -1, data.data(), data.size()); });
        rejects(t, "past last expert", [&] { source.read_expert(name, 4, data.data(), data.size()); });
        rejects(t, "small destination", [&] { source.read_expert(name, 0, data.data(), data.size() - 1); });
        rejects(t, "large destination", [&] { source.read_expert(name, 0, data.data(), data.size() + 1); });
        rejects(t, "null destination", [&] { source.read_expert(name, 0, nullptr, data.size()); });
        rejects(t, "missing tensor", [&] { source.read_expert("missing", 0, data.data(), data.size()); });
        t.assert_true("invalid requests leave destination untouched",
                      std::all_of(data.begin(), data.end(), [](uint8_t b) { return b == 0xA5; }));
        source.read_expert(name, 0, data.data(), data.size());
        t.assert_true("valid read after argument errors", data[0] == 0 && data[1] == 17);
    });
    t.test("packed PLE rows across shards", [&](testing & t) {
        fixture f;
        f.write(0, GGML_TYPE_F32, "experts");
        const char * name = "per_layer_token_embd.weight";
        f.write(1, GGML_TYPE_IQ4_NL, name, 2, 2, nullptr, false, true);
        llama_expert_source source(f.paths());
        ggml_context * raw = nullptr;
        gguf_context_ptr meta(gguf_init_from_file(f.path(1).c_str(), {false, &raw}));
        ggml_context_ptr loaded(raw);
        if (!meta || !loaded) { throw std::runtime_error("row reference load failed"); }
        const auto * ref = ggml_get_tensor(loaded.get(), name);
        const size_t bytes = ggml_row_size(ref->type, ref->ne[0]);
        std::vector<uint8_t> data(bytes + 2, 0xA5);
        rejects(t, "negative row", [&] { source.read_row(name, -1, data.data() + 1, bytes); });
        rejects(t, "past last row", [&] { source.read_row(name, ref->ne[1], data.data() + 1, bytes); });
        rejects(t, "short row buffer", [&] { source.read_row(name, 0, data.data() + 1, bytes - 1); });
        rejects(t, "null row buffer", [&] { source.read_row(name, 0, nullptr, bytes); });
        for (int64_t row : {int64_t(2), int64_t(0), int64_t(1)}) {
            source.read_row(name, row, data.data() + 1, bytes);
            t.assert_true("original row bytes", std::memcmp(data.data() + 1,
                static_cast<const uint8_t *>(ref->data) + row * ref->nb[1], bytes) == 0);
            t.assert_true("row destination guards", data.front() == 0xA5 && data.back() == 0xA5);
        }
        fs::resize_file(f.path(1), fs::file_size(f.path(1)) - 1);
        rejects(t, "changed row source", [&] { source.read_row(name, 0, data.data() + 1, bytes); });
        rejects(t, "failed source stays invalid", [&] { source.read_row(name, 0, data.data() + 1, bytes); });
    });
    t.test("unsupported layouts", [&](testing & t) {
        for (bool dense : {false, true}) {
            fixture f;
            f.write(0, GGML_TYPE_F32, "weights", 0, 1, "qwen3moe", !dense, dense);
            llama_expert_source source({f.path(0)});
            t.assert_true("layout rejected", !source.tensor("weights").can_read_experts());
            uint8_t byte = 0;
            rejects(t, "unsupported read", [&] { source.read_expert("weights", 0, &byte, 1); });
        }
    });
    t.test("shard set validation", [&](testing & t) {
        fixture f;
        f.write(0, GGML_TYPE_F32, "first");
        f.write(1, GGML_TYPE_Q4_0, "second");
        rejects(t, "empty source", [&] { llama_expert_source source({}); });
        rejects(t, "missing shard", [&] { llama_expert_source source({f.path(0)}); });
        rejects(t, "unordered shards", [&] { llama_expert_source source({f.path(1), f.path(0)}); });
        rejects(t, "missing file", [&] { llama_expert_source source({f.path(0), f.path(9)}); });
        f.write(1, GGML_TYPE_F32, "first");
        rejects(t, "duplicate tensor", [&] { llama_expert_source source(f.paths()); });
        f.write(1, GGML_TYPE_F32, "second", 2, 2, "other");
        rejects(t, "architecture mismatch", [&] { llama_expert_source source(f.paths()); });
        f.write(1, GGML_TYPE_F32, "second", 2, 3);
        rejects(t, "inconsistent declared count", [&] { llama_expert_source source(f.paths()); });
        f.write(0, GGML_TYPE_F32, "first", 2, 3);
        rejects(t, "missing declared tensor", [&] { llama_expert_source source(f.paths()); });
    });
    t.test("truncation and source invalidation", [&](testing & t) {
        fixture f;
        f.write(0, GGML_TYPE_F32, "weights", 0);
        llama_expert_source source({f.path(0)});
        std::vector<uint8_t> data(source.tensor("weights").expert_bytes(), 0xA5);
        fs::resize_file(f.path(0), source.tensor("weights").offset + 1);
        rejects(t, "truncated indexed file", [&] { source.read_expert("weights", 0, data.data(), data.size()); });
        t.assert_true("changed source rejected before writing destination", data.front() == 0xA5);
        rejects(t, "truncated file at open", [&] { llama_expert_source bad({f.path(0)}); });
        f.write(0, GGML_TYPE_F32, "weights", 0);
        rejects(t, "restoring file cannot revive invalid source", [&] {
            source.read_expert("weights", 0, data.data(), data.size());
        });
    });
    t.test("same-size mutation", [&](testing & t) {
        fixture f;
        f.write(0, GGML_TYPE_F32, "weights", 0);
        llama_expert_source source({f.path(0)});
        std::vector<uint8_t> data(source.tensor("weights").expert_bytes());
        {
            std::fstream file(f.path(0), std::ios::in | std::ios::out | std::ios::binary);
            file.seekp(source.tensor("weights").offset);
            file.put('\x7f');
        }
        // Force a detectable timestamp even on filesystems with coarse resolution.
        fs::last_write_time(f.path(0), fs::last_write_time(f.path(0)) + std::chrono::seconds(2));
        rejects(t, "modified source", [&] { source.read_expert("weights", 0, data.data(), data.size()); });
    });
    t.test("concurrent reads retain source identity", [&](testing & t) {
        fixture f;
        f.write(0, GGML_TYPE_F32, "weights", 0);
        llama_expert_source source({f.path(0)});
        const size_t size = source.tensor("weights").expert_bytes();
        std::atomic<bool> ok {true};
        std::vector<std::thread> workers;
        for (int64_t expert = 0; expert < 4; ++expert) {
            workers.emplace_back([&, expert] {
                try {
                    std::vector<uint8_t> data(size);
                    for (int repeat = 0; repeat < 8; ++repeat) {
                        source.read_expert("weights", expert, data.data(), size);
                        for (size_t i = 0; i < size; ++i) {
                            const size_t offset = expert * size + i;
                            if (data[i] != uint8_t(offset * 17 + (offset >> 8))) {
                                ok = false;
                            }
                        }
                    }
                } catch (...) {
                    ok = false;
                }
            });
        }
        for (auto & worker : workers) {
            worker.join();
        }
        t.assert_true("no shared-cursor corruption", ok.load());
    });
    return t.summary();
}
