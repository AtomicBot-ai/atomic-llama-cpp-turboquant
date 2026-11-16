#pragma once

#include "llama-model.h"
#include "llama-expert-source.h"
#include "llama-expert-trace.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"

#include <array>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Caller-owned diagnostic files, like the runner's logits output. A fixed 64 KiB
// stack buffer copies existing tensors; there is no activation queue or model
// clone. This storage is covered by the existing runtime control/stack envelope.
struct expert_preview_trace {
    using json = nlohmann::ordered_json;
    std::unique_ptr<FILE, decltype(&std::fclose)> file {nullptr, std::fclose};
    std::unique_ptr<FILE, decltype(&std::fclose)> data {nullptr, std::fclose};
    int first_layer = -1, last_layer = -1, previous_layer = -1;
    int call = -1, microbatch = -1, first_position = 0, width = 0;
    size_t offset = 0;

    void write(const json & record) {
        const auto line = record.dump();
        if (std::fprintf(file.get(), "%s\n", line.c_str()) < 0) { throw std::runtime_error("preview trace write failed"); }
    }
    void bytes(const void * src, size_t n) {
        if (std::fwrite(src, 1, n, data.get()) != n) { throw std::runtime_error("preview tensor write failed"); }
        offset += n;
    }
    json tensor(const ggml_tensor * t) {
        if (!t) { return nullptr; }
        if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16) { throw std::runtime_error("unsupported preview router type"); }
        if (!ggml_is_contiguous(t)) { throw std::runtime_error("noncontiguous preview router"); }
        json meta = {{"offset", offset}, {"bytes", ggml_nbytes(t)}, {"type", t->type == GGML_TYPE_F32 ? "f32" : "f16"},
                     {"input", t->ne[0]}, {"output", t->ne[1]}};
        std::array<uint8_t, 65536> chunk {};
        for (size_t at = 0; at < ggml_nbytes(t); at += chunk.size()) {
            const size_t n = std::min(chunk.size(), ggml_nbytes(t) - at);
            ggml_backend_tensor_get(t, chunk.data(), at, n);
            bytes(chunk.data(), n);
        }
        return meta;
    }
    void open(const std::string & path, const llama_model & model) {
        if (!model.expert_source || !model.memory_contract ||
            (model.arch != LLM_ARCH_QWEN35MOE && model.arch != LLM_ARCH_LFM2MOE)) {
            throw std::runtime_error("preview capture requires a budgeted Qwen3.5 MoE or LFM2 MoE model");
        }
        file.reset(std::fopen(path.c_str(), "wb"));
        data.reset(std::fopen((path + ".bin").c_str(), "wb"));
        if (!file || !data) { throw std::runtime_error("cannot open preview capture"); }
        json tensors = json::array(), routers = json::array();
        const auto & source = model.expert_source->tensors();
        for (size_t i = 0; i < source.size(); ++i) {
            const auto & t = source[i];
            int layer = -1, end = 0;
            if (std::sscanf(t.name.c_str(), "blk.%d.%n", &layer, &end) != 1 || end == 0) { continue; }
            const auto suffix = t.name.substr(size_t(end));
            if (suffix != "ffn_up_exps.weight" && suffix != "ffn_gate_exps.weight" && suffix != "ffn_down_exps.weight") { continue; }
            if (first_layer < 0 || layer < first_layer) { first_layer = layer; }
            last_layer = std::max(last_layer, layer);
            tensors.push_back({{"index", i}, {"layer", layer}, {"name", suffix}, {"payload", t.expert_bytes()}, {"experts", t.ne[2]}});
        }
        if (first_layer < 0 || last_layer <= first_layer) { throw std::runtime_error("no adjacent routed layers for preview"); }
        for (int layer = first_layer; layer <= last_layer; ++layer) {
            const auto & l = model.layers[layer];
            routers.push_back({{"layer", layer}, {"weight", tensor(l.ffn_gate_inp)}, {"bias", tensor(l.ffn_exp_probs_b)}});
        }
        write({{"format", "ATP2"}, {"architecture", model.expert_source->architecture()},
               {"first_layer", first_layer}, {"last_layer", last_layer}, {"sequence", 0},
               {"output_policy", "last_row_only"}, {"tensors", tensors}, {"routers", routers}});
    }
    void begin_call(int position) {
        ++call; microbatch = -1; previous_layer = -1; first_position = position; width = 0;
    }
    static void callback(const llama_expert_route_event & event, void * user) {
        auto & self = *static_cast<expert_preview_trace *>(user);
        int layer = -1, end = 0;
        if (std::sscanf(event.weight, "blk.%d.%n", &layer, &end) != 1 || end == 0) { return; }
        const std::string suffix(event.weight + end);
        if ((suffix != "ffn_up_exps.weight" && suffix != "ffn_gate_exps.weight") || layer == self.previous_layer) { return; }
        const auto * input = event.input;
        const auto * ids = event.ids;
        if (input->type != GGML_TYPE_F32 || input->ne[0] < 1 || input->ne[0] > 16384 || input->ne[1] != 1 ||
            input->ne[2] != ids->ne[1] || ids->type != GGML_TYPE_I32 || ids->ne[0] < 1 || ids->ne[0] > 32 ||
            ids->ne[1] < 1 || ids->ne[1] > 256) { throw std::runtime_error("unsupported route capture geometry"); }
        if (self.previous_layer < 0 || layer < self.previous_layer) {
            if (layer != self.first_layer) { throw std::runtime_error("capture missed first routed layer"); }
            self.first_position += self.width; self.width = int(ids->ne[1]); ++self.microbatch;
        }
        self.previous_layer = layer;
        const int rows = int(ids->ne[1]);
        const bool last_only = layer == self.last_layer && rows == 1 && self.width > 1;
        if (rows != self.width && !last_only) { throw std::runtime_error("unmatched route rows"); }
        json selected = json::array(), positions = json::array();
        const size_t start = self.offset;
        std::array<float, 16384> row {};
        for (int i = 0; i < rows; ++i) {
            for (int64_t col = 0; col < input->ne[0]; ++col) {
                std::memcpy(&row[size_t(col)], static_cast<const char *>(input->data) + i * input->nb[2] + col * input->nb[0], sizeof(float));
            }
            self.bytes(row.data(), size_t(input->ne[0]) * sizeof(float));
            std::vector<int32_t> chosen(size_t(ids->ne[0]));
            for (int64_t slot = 0; slot < ids->ne[0]; ++slot) {
                std::memcpy(&chosen[size_t(slot)], static_cast<const char *>(ids->data) + i * ids->nb[1] + slot * ids->nb[0], sizeof(int32_t));
            }
            selected.push_back(chosen);
            positions.push_back(self.first_position + (last_only ? self.width - 1 : i));
        }
        self.write({{"call", self.call}, {"microbatch", self.microbatch}, {"sequence", 0}, {"layer", layer},
                    {"projection", event.projection}, {"positions", positions}, {"ids", selected},
                    {"offset", start}, {"bytes", self.offset - start}, {"input", input->ne[0]}});
    }
    void finish() {
        if (std::fflush(file.get()) != 0 || std::fflush(data.get()) != 0) { throw std::runtime_error("cannot flush preview capture"); }
    }
};
