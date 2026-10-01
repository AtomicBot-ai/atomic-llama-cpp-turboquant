// Cross-backend replay test of the laya engine (Phase 4b): a CPU context against the first GPU / iGPU
// device (laya_gpu_devices()[0]) on the tiny random laya GGUF (tests/decision/make_tiny_laya.py).
//
// Each backend runs three packed passes, A, B, A. A and B have the same total token count and the
// same two sequences in the opposite order, so their seq_start (and the marker rows, masks and
// positions of every token) differ. Checked:
//   - replay: the second A is bitwise equal to the first on each backend (scorer and act logits).
//     The graph is rebuilt on every laya_encode and nothing is kept between calls, so this is a
//     determinism check of the backend and of ggml_backend_sched across changing inputs, weaker than
//     the graph-reuse replay of laya.cpp tests/backend_replay.cpp that it follows.
//   - cross-backend: every sequence of A and B on the device against the CPU, with the f16-class rule
//     of tests/laya/parity/tiers.json for the weight type of the GGUF (F16 for the tiny model): an
//     argmax flip only below the baseline top-2 gap flip_max_gap and on at most max_flip_fraction of
//     the sequences, max and mean |dp| of softmax(raw scorer logits) within max_abs_dp / mean_abs_dp,
//     act-head relative dlogit within act_max_rel_dlogit. The CPU context uses the kernels the tier
//     names as the baseline for Metal and for CUDA / Vulkan on F16 weights: BLAS when the build has
//     Accelerate (what the decision server's kernels "auto" picks), else the ggml CPU kernels.
//   - informational: whether a sequence gets the same bits in A and in B on each backend.
//   - strict placement: a device model with one encoder weight forced into host memory (the
//     laya_model_params.host_weights test hook) reports host_weights 1, and laya_init_ext with
//     strict_placement refuses it.
// Exit code 77 (ctest: skipped) when the build has no GPU / iGPU device.
//
//   test-laya-backend-parity <tiny-laya.gguf> <tiers.json>
//
// Test shape adapted from laya.cpp tests/backend_replay.cpp (https://github.com/lkarlslund/laya.cpp),
// MIT License, Copyright (c) 2026 Lars Karlslund.
// The full license text (permission notice included) is in licenses/LICENSE-laya.cpp.

#include "laya.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

namespace {

struct seq_spec {
    std::vector<int32_t> ids;
    std::vector<int32_t> markers;  // sequence-local positions of the marker tokens
    int32_t              qtype = 0;
};

struct batch_data {
    std::vector<int32_t> tokens, positions, seq_id, qtype, marker_pos, marker_mask, seq_start;
    laya_batch           batch;
};

// a sequence shaped like build_sequence: [CLS] instr [SEP] ([MASK] option)* [SEP] state [SEP]
seq_spec make_seq(const laya_model * model, int32_t n_tokens, int32_t n_options, int32_t qtype, uint32_t seed) {
    const laya_hparams & hp = laya_model_hparams(model);
    const int32_t cls = laya_vocab_bos(model), sep = laya_vocab_sep(model), mask = laya_vocab_mask(model);
    uint32_t state = seed;
    auto next_token = [&]() {
        for (;;) {
            state = state * 747796405u + 2891336453u;
            const int32_t t = (int32_t) ((state >> 8) % (uint32_t) hp.n_vocab);
            if (t != cls && t != sep && t != mask && t > 4) {
                return t;
            }
        }
    };
    seq_spec s;
    s.qtype = qtype;
    s.ids.push_back(cls);
    for (int i = 0; i < 6; ++i) s.ids.push_back(next_token());
    s.ids.push_back(sep);
    for (int o = 0; o < n_options; ++o) {
        s.markers.push_back((int32_t) s.ids.size());
        s.ids.push_back(mask);
        for (int i = 0; i < 3; ++i) s.ids.push_back(next_token());
    }
    s.ids.push_back(sep);
    while ((int32_t) s.ids.size() < n_tokens - 1) s.ids.push_back(next_token());
    s.ids.push_back(sep);
    if ((int32_t) s.ids.size() != n_tokens) {
        throw std::runtime_error("sequence too short for its options");
    }
    return s;
}

void pack(const std::vector<const seq_spec *> & seqs, batch_data & d) {
    const int32_t n_seqs = (int32_t) seqs.size();
    d.tokens.clear(); d.positions.clear(); d.seq_id.clear(); d.qtype.clear();
    d.marker_pos.assign((size_t) LAYA_MAX_MARKERS * n_seqs, 0);
    d.marker_mask.assign((size_t) LAYA_MAX_MARKERS * n_seqs, 0);
    d.seq_start.assign(n_seqs, 0);
    for (int32_t s = 0; s < n_seqs; ++s) {
        d.seq_start[s] = (int32_t) d.tokens.size();
        for (size_t i = 0; i < seqs[s]->ids.size(); ++i) {
            d.tokens.push_back(seqs[s]->ids[i]);
            d.positions.push_back((int32_t) i);
            d.seq_id.push_back(s);
            d.qtype.push_back(seqs[s]->qtype);
        }
        for (size_t j = 0; j < seqs[s]->markers.size(); ++j) {
            d.marker_pos[j + (size_t) LAYA_MAX_MARKERS * s]  = d.seq_start[s] + seqs[s]->markers[j];
            d.marker_mask[j + (size_t) LAYA_MAX_MARKERS * s] = 1;
        }
    }
    d.batch.n_tokens    = (int32_t) d.tokens.size();
    d.batch.n_seqs      = n_seqs;
    d.batch.tokens      = d.tokens.data();
    d.batch.positions   = d.positions.data();
    d.batch.seq_id      = d.seq_id.data();
    d.batch.qtype       = d.qtype.data();
    d.batch.marker_pos  = d.marker_pos.data();
    d.batch.marker_mask = d.marker_mask.data();
    d.batch.seq_start   = d.seq_start.data();
}

// the outputs of one sequence of a pass
struct seq_out {
    std::vector<float> logits;  // the valid markers only
    std::vector<float> act;
};

seq_out take(const laya_result & r, int32_t s, size_t n_markers) {
    seq_out o;
    const float * l = r.logits.data() + (size_t) s * r.n_markers_max;
    o.logits.assign(l, l + n_markers);
    const float * a = r.act_logits.data() + (size_t) s * r.n_act;
    o.act.assign(a, a + r.n_act);
    return o;
}

bool same_bits(const seq_out & a, const seq_out & b) {
    return a.logits.size() == b.logits.size() && a.act.size() == b.act.size() &&
           std::memcmp(a.logits.data(), b.logits.data(), a.logits.size() * sizeof(float)) == 0 &&
           std::memcmp(a.act.data(), b.act.data(), a.act.size() * sizeof(float)) == 0;
}

bool same_bits(const laya_result & a, const laya_result & b) {
    return a.logits.size() == b.logits.size() && a.act_logits.size() == b.act_logits.size() &&
           std::memcmp(a.logits.data(), b.logits.data(), a.logits.size() * sizeof(float)) == 0 &&
           std::memcmp(a.act_logits.data(), b.act_logits.data(), a.act_logits.size() * sizeof(float)) == 0;
}

std::vector<double> softmax(const std::vector<float> & z) {
    double mx = -INFINITY;
    for (float v : z) mx = std::max(mx, (double) v);
    std::vector<double> p(z.size());
    double sum = 0.0;
    for (size_t i = 0; i < z.size(); ++i) { p[i] = std::exp((double) z[i] - mx); sum += p[i]; }
    for (double & v : p) v /= sum;
    return p;
}

size_t argmax(const std::vector<double> & p) {
    return (size_t) (std::max_element(p.begin(), p.end()) - p.begin());
}

const char * weights_of(const std::string & path) {
    // general.file_type is not needed: the tiny model's matmul weights are F16 (make_tiny_laya.py);
    // a Q8_0 tiny model (--q8) is named so by the caller
    return path.find("q8") != std::string::npos ? "q8_0" : "f16";
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <tiny-laya.gguf> <tiers.json>\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];

    ggml_backend_load_all();
    const std::vector<ggml_backend_dev_t> gpus = laya_gpu_devices();
    if (gpus.empty()) {
        printf("no GPU / iGPU device in this build: skipped\n");
        return 77;
    }

    json tier;
    try {
        std::ifstream f(argv[2]);
        const json tiers = json::parse(f);
        tier = tiers.at("tiers").at("f16-class").at("weights").at(weights_of(model_path));
    } catch (const std::exception & e) {
        fprintf(stderr, "cannot read the f16-class tier from %s: %s\n", argv[2], e.what());
        return 1;
    }
    const double flip_max_gap       = tier.at("flip_max_gap").get<double>();
    const double max_flip_fraction  = tier.at("max_flip_fraction").get<double>();
    const double max_abs_dp         = tier.at("max_abs_dp").get<double>();
    const double mean_abs_dp        = tier.at("mean_abs_dp").get<double>();
    const double act_max_rel_dlogit = tier.at("act_max_rel_dlogit").get<double>();

    laya_model * m_cpu = nullptr;
    laya_model * m_dev = nullptr;
    laya_context * c_cpu = nullptr;
    laya_context * c_dev = nullptr;
    int rc = 0;
    try {
        laya_model_params mp;
        m_cpu = laya_model_load_from_file_ext(model_path.c_str(), mp);
        mp.device = gpus[0];
        m_dev = laya_model_load_from_file_ext(model_path.c_str(), mp);
        if (!m_cpu || !m_dev) {
            throw std::runtime_error("failed to load " + model_path);
        }
        laya_context_params cp;
        cp.n_threads = 4;
        cp.use_blas  = laya_blas_description() == "Accelerate";
        c_cpu = laya_init_ext(m_cpu, cp);
        cp.use_blas = false;
        c_dev = laya_init_ext(m_dev, cp);
        printf("baseline %s, candidate %s (%s): %s\n", laya_context_kernels(c_cpu).c_str(), laya_device_name(gpus[0]).c_str(),
               ggml_backend_dev_description(gpus[0]), laya_placement_str(laya_context_placement(c_dev)).c_str());

        // A = [s1, s2], B = [s2, s1]: same total tokens, different seq_start
        const seq_spec s1 = make_seq(m_cpu, 40, 3, 0, 1);
        const seq_spec s2 = make_seq(m_cpu, 56, 5, 1, 2);
        batch_data a, b;
        pack({ &s1, &s2 }, a);
        pack({ &s2, &s1 }, b);

        struct runs { laya_result a1, b, a2; };
        auto run3 = [&](laya_context * ctx, runs & r) {
            if (laya_encode(ctx, a.batch, r.a1) != 0 || laya_encode(ctx, b.batch, r.b) != 0 ||
                laya_encode(ctx, a.batch, r.a2) != 0) {
                throw std::runtime_error("laya_encode failed");
            }
        };
        runs cpu, dev;
        run3(c_cpu, cpu);
        run3(c_dev, dev);

        for (auto * which : { &cpu, &dev }) {
            const std::string name_s = which == &cpu ? std::string("cpu") : laya_device_name(gpus[0]);
            const char * name = name_s.c_str();
            const bool replay = same_bits(which->a1, which->a2);
            printf("replay A,B,A on %s: %s\n", name, replay ? "bitwise" : "DIFFERENT");
            rc |= replay ? 0 : 1;
            // informational: the same sequence at another offset of the packed batch
            const bool moved = same_bits(take(which->a1, 0, s1.markers.size()), take(which->b, 1, s1.markers.size())) &&
                               same_bits(take(which->a1, 1, s2.markers.size()), take(which->b, 0, s2.markers.size()));
            printf("  same sequence in A and B on %s: %s (informational)\n", name, moved ? "bitwise" : "not bitwise");
        }

        // cross-backend, f16-class: 4 sequences (s1, s2 in A; s2, s1 in B)
        struct pair { const laya_result * base; const laya_result * cand; int32_t idx; size_t n; };
        const pair pairs[] = {
            { &cpu.a1, &dev.a1, 0, s1.markers.size() }, { &cpu.a1, &dev.a1, 1, s2.markers.size() },
            { &cpu.b,  &dev.b,  0, s2.markers.size() }, { &cpu.b,  &dev.b,  1, s1.markers.size() },
        };
        int n_flips = 0, n_bad_flips = 0, n_dp = 0;
        double max_dp = 0.0, sum_dp = 0.0, max_rel_act = 0.0, max_dlogit = 0.0;
        for (const pair & pr : pairs) {
            const seq_out x = take(*pr.base, pr.idx, pr.n);
            const seq_out y = take(*pr.cand, pr.idx, pr.n);
            const std::vector<double> px = softmax(x.logits), py = softmax(y.logits);
            for (size_t i = 0; i < px.size(); ++i) {
                const double dp = std::fabs(px[i] - py[i]);
                if (!std::isfinite(px[i]) || !std::isfinite(py[i])) {
                    max_dp = INFINITY;  // std::max would drop a NaN
                }
                max_dp = std::max(max_dp, dp);
                sum_dp += dp;
                ++n_dp;
                max_dlogit = std::max(max_dlogit, (double) std::fabs(x.logits[i] - y.logits[i]));
            }
            if (argmax(px) != argmax(py)) {
                std::vector<double> sorted = px;
                std::sort(sorted.begin(), sorted.end(), std::greater<double>());
                const double gap = sorted[0] - sorted[1];
                ++n_flips;
                n_bad_flips += gap < flip_max_gap ? 0 : 1;
                printf("  flip (baseline top-2 gap %.6g)\n", gap);
            }
            double num = 0.0, den = 1.0;
            bool act_finite = x.act.size() == y.act.size();
            for (size_t i = 0; i < x.act.size() && act_finite; ++i) {
                act_finite = std::isfinite(x.act[i]) && std::isfinite(y.act[i]);
                num = std::max(num, (double) std::fabs(x.act[i] - y.act[i]));
                den = std::max({ den, (double) std::fabs(x.act[i]), (double) std::fabs(y.act[i]) });
            }
            // a NaN act logit would vanish in std::max: not finite fails outright
            max_rel_act = act_finite ? std::max(max_rel_act, num / den) : INFINITY;
        }
        const double flip_fraction = (double) n_flips / (double) (sizeof(pairs) / sizeof(pairs[0]));
        const double mean_dp = sum_dp / std::max(1, n_dp);
        const bool ok = n_bad_flips == 0 && flip_fraction <= max_flip_fraction && max_dp <= max_abs_dp &&
                        mean_dp <= mean_abs_dp && max_rel_act <= act_max_rel_dlogit;
        printf("cross-backend (f16-class, %s weights): flips %d (fraction %.3g, limit %.3g; above the gap %d), "
               "max|dp| %.3g (limit %.3g), mean|dp| %.3g (limit %.3g), act rel dlogit %.3g (limit %.3g), "
               "max|dlogit| %.3g (diagnostic): %s\n",
               weights_of(model_path), n_flips, flip_fraction, max_flip_fraction, n_bad_flips, max_dp, max_abs_dp,
               mean_dp, mean_abs_dp, max_rel_act, act_max_rel_dlogit, max_dlogit, ok ? "pass" : "FAIL");
        rc |= ok ? 0 : 1;

        // strict placement: a weight forced into host memory is reported and refused
        {
            laya_model_params mph;
            mph.device = gpus[0];
            mph.host_weights = { "blk.0.ffn_up.weight" };
            laya_model * m_host = laya_model_load_from_file_ext(model_path.c_str(), mph);
            laya_context_params cph;
            cph.n_threads = 4;
            laya_context * c_host = laya_init_ext(m_host, cph);
            const laya_placement pl = laya_context_placement(c_host);
            laya_free(c_host);
            bool refused = false;
            cph.strict_placement = true;
            try {
                laya_free(laya_init_ext(m_host, cph));
            } catch (const std::exception & e) {
                refused = true;
                printf("  strict placement refused: %s\n", e.what());
            }
            laya_model_free(m_host);
            const bool ok_pl = pl.host_weights == 1 && refused;
            printf("strict placement with one host weight: host_weights %d, %s: %s\n", pl.host_weights,
                   refused ? "refused" : "NOT refused", ok_pl ? "pass" : "FAIL");
            rc |= ok_pl ? 0 : 1;
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        rc = 1;
    }
    laya_free(c_dev);
    laya_free(c_cpu);
    laya_model_free(m_dev);
    laya_model_free(m_cpu);
    printf("%s\n", rc == 0 ? "OK" : "FAIL");
    return rc;
}
