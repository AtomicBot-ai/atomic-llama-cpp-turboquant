// Matmul precision probe of the ggml backend devices, for the laya parity tiers
// (DECISION.md, "Backend parity tiers").
//
// Every device multiplies small matrices whose exact products are known: F32 x F32 at k = 65 and
// F16 x F16 at k = 65, 1024 and 2624, each at n = 33 columns (the matrix-matrix path of a
// multi-token pass) and n = 5 (the small-batch path of the marker-row head matmuls), with the two
// precision requests the laya graph uses: GGML_PREC_F32 (laya "default" on the weight matmuls) and
// GGML_PREC_F32_PEDANTIC (laya "strict"). The oracle sums, in double, the values the device really
// received (the F16 inputs are rounded on the host first).
//
// The values are those of the laya.cpp probe. The F32 operands have low bits that half and TF32
// rounding drop (1 + j/8192), so a kernel that rounds F32 operands to 10 mantissa bits misses the
// oracle by ~1e-4; the F16 products and their sums are exact in fp32 at k = 1024 and 2624 (multiples
// of 2^-17 below 2^6), so the F16 cases see only the accumulator (a half accumulator misses by 1e-2
// and more). A case passes when every output is within 1e-5 of the oracle (the laya.cpp tolerance;
// the outputs are of size ~1 to ~30). The relative error (to sum |a * b|) is printed as well.
//
// Tier of a device for a precision request:
//   strict-f32       every case passes (fp32 operands and fp32 accumulation)
//   f16-class        the F16 cases pass (fp32 accumulation), the F32 case does not (half / TF32 operands)
//   below-f16-class  an F16 case fails (half accumulation): no laya tier admits it
//
// A case the laya graph does not issue on a device is run the way the graph does it: where laya
// dequantizes an F16 weight to F32 first (laya_weight_dequantized: CUDA in strict, where ggml-cuda
// would keep a PEDANTIC F16 matmul in F16 compute), the same half-rounded values go in as F32 x F32
// (marked "as f32").
//
// GPU and iGPU devices are gated: the test fails when one of them is below-f16-class for either
// request, or cannot compute a case. The CPU and accelerator (BLAS) devices are printed as a
// reference only (their tier is set by the kernels, DECISION.md). Exit code 77 (ctest: skipped)
// when the build has no GPU / iGPU device.
//
//   test-laya-backend-precision [--json out.json]
//
// Adapted from the matrix precision check of laya.cpp tests/vulkan_precision.cpp
// (https://github.com/lkarlslund/laya.cpp), MIT License, Copyright (c) 2026 Lars Karlslund.
// The full license text (permission notice included) is in licenses/LICENSE-laya.cpp.

#include "ggml.h"
#include "ggml-backend.h"
#include "laya.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct probe_case {
    ggml_type type;
    int       k;
    int       n;
};

struct probe_result {
    bool   supported = false;
    bool   computed  = false;
    bool   pass      = false;
    double max_abs   = 0.0;  // max |device - oracle|
    double max_rel   = 0.0;  // max |device - oracle| / sum |a * b|
};

const double ABS_TOL = 1e-5;
const int    M       = 37;

// as_f32: the F16 values go in as F32 tensors (the graph dequantizes such a weight first)
probe_result run_case(ggml_backend_t backend, ggml_backend_dev_t dev, const probe_case & pc, ggml_prec prec, bool as_f32) {
    probe_result r;
    const int k = pc.k;
    const int n = pc.n;
    const ggml_type tt = as_f32 ? GGML_TYPE_F32 : pc.type;

    ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, tt, k, M);
    ggml_tensor * b = ggml_new_tensor_2d(ctx, tt, k, n);
    ggml_set_input(a);
    ggml_set_input(b);
    ggml_tensor * product = ggml_mul_mat(ctx, a, b);
    ggml_mul_mat_set_prec(product, prec);
    ggml_set_output(product);

    r.supported = ggml_backend_dev_supports_op(dev, product);
    if (!r.supported) {
        ggml_free(ctx);
        return r;
    }

    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, product);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buf) {
        ggml_free(ctx);
        return r;
    }

    std::vector<float> av((size_t) k * M), bv((size_t) k * n), actual((size_t) M * n);
    const float da = k == 65 ? 8192.0f : 512.0f;
    const float db = k == 65 ? 4096.0f : 256.0f;
    for (size_t i = 0; i < av.size(); ++i) {
        av[i] = 1.0f + float(i % 7 + 1) / da;
    }
    for (size_t i = 0; i < bv.size(); ++i) {
        bv[i] = (i % 2 ? -0.5f : 0.5f) + float(i % 5) / db;
    }
    auto upload = [&](ggml_tensor * t, std::vector<float> & values) {
        if (pc.type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
            return;
        }
        std::vector<ggml_fp16_t> half(values.size());
        for (size_t i = 0; i < values.size(); ++i) {
            half[i]   = ggml_fp32_to_fp16(values[i]);
            values[i] = ggml_fp16_to_fp32(half[i]);  // the oracle uses what the device got
        }
        if (t->type == GGML_TYPE_F32) {
            ggml_backend_tensor_set(t, values.data(), 0, ggml_nbytes(t));
        } else {
            ggml_backend_tensor_set(t, half.data(), 0, ggml_nbytes(t));
        }
    };
    upload(a, av);
    upload(b, bv);

    if (ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS) {
        r.computed = true;
        ggml_backend_tensor_get(product, actual.data(), 0, ggml_nbytes(product));
        r.pass = true;
        for (int col = 0; col < n; ++col) {
            for (int row = 0; row < M; ++row) {
                double expected = 0.0, scale = 0.0;
                for (int i = 0; i < k; ++i) {
                    const double p = double(av[(size_t) row * k + i]) * double(bv[(size_t) col * k + i]);
                    expected += p;
                    scale    += std::fabs(p);
                }
                const double got = actual[(size_t) col * M + row];
                const double err = std::fabs(got - expected);
                if (!std::isfinite(got) || err > ABS_TOL) {
                    r.pass = false;
                }
                r.max_abs = std::isfinite(err) ? std::max(r.max_abs, err) : INFINITY;
                r.max_rel = std::isfinite(err) ? std::max(r.max_rel, err / scale) : INFINITY;
            }
        }
    }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return r;
}

const char * type_str(enum ggml_backend_dev_type t) {
    switch (t) {
        case GGML_BACKEND_DEVICE_TYPE_CPU:   return "cpu";
        case GGML_BACKEND_DEVICE_TYPE_GPU:   return "gpu";
        case GGML_BACKEND_DEVICE_TYPE_IGPU:  return "igpu";
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return "accel";
        default:                             return "other";
    }
}

std::string json_str(const std::string & s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            out += '\\';
            out += c;
        } else if ((unsigned char) c < 0x20) {
            char tmp[8];
            snprintf(tmp, sizeof(tmp), "\\u%04x", (unsigned char) c);
            out += tmp;
        } else {
            out += c;
        }
    }
    return out + "\"";
}

std::string json_num(double v) {
    if (!std::isfinite(v)) {
        return "null";
    }
    char tmp[32];
    snprintf(tmp, sizeof(tmp), "%.6g", v);
    return tmp;
}

}  // namespace

int main(int argc, char ** argv) {
    std::string json_path;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            json_path = argv[++i];
        } else {
            fprintf(stderr, "usage: %s [--json out.json]\n", argv[0]);
            return 1;
        }
    }

    ggml_backend_load_all();

    const std::vector<probe_case> cases = {
        { GGML_TYPE_F32,   65, 33 }, { GGML_TYPE_F32,   65, 5 },
        { GGML_TYPE_F16,   65, 33 }, { GGML_TYPE_F16,   65, 5 },
        { GGML_TYPE_F16, 1024, 33 }, { GGML_TYPE_F16, 1024, 5 },
        { GGML_TYPE_F16, 2624, 33 }, { GGML_TYPE_F16, 2624, 5 },
    };
    const struct { ggml_prec prec; laya_precision laya; const char * name; } precs[] = {
        { GGML_PREC_F32,          LAYA_PRECISION_DEFAULT, "default" },  // laya default on the weight matmuls
        { GGML_PREC_F32_PEDANTIC, LAYA_PRECISION_STRICT,  "strict"  },
    };

    int  n_gpu  = 0;
    bool failed = false;
    std::string js = "{\n \"schema\": \"laya-backend-precision/1\",\n \"abs_tol\": " + json_num(ABS_TOL) + ",\n \"devices\": [";
    bool first_dev = true;

    for (size_t di = 0; di < ggml_backend_dev_count(); ++di) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(di);
        const enum ggml_backend_dev_type dt = ggml_backend_dev_type(dev);
        const bool gated = dt == GGML_BACKEND_DEVICE_TYPE_GPU || dt == GGML_BACKEND_DEVICE_TYPE_IGPU;
        n_gpu += gated ? 1 : 0;

        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "%s: failed to init the backend\n", ggml_backend_dev_name(dev));
            failed = failed || gated;
            continue;
        }
        printf("%s (%s, %s)%s\n", ggml_backend_dev_name(dev), type_str(dt), ggml_backend_dev_description(dev),
               gated ? "" : " - reference only, not gated");

        js += std::string(first_dev ? "" : ",") + "\n  {\"name\": " + json_str(ggml_backend_dev_name(dev)) +
              ", \"type\": " + json_str(type_str(dt)) + ", \"description\": " + json_str(ggml_backend_dev_description(dev)) +
              ", \"gated\": " + (gated ? "true" : "false") + ", \"precision\": {";
        first_dev = false;

        for (size_t pi = 0; pi < sizeof(precs) / sizeof(precs[0]); ++pi) {
            bool f32_ok = true, f16_ok = true, broken = false;
            std::string jcases;
            for (size_t ci = 0; ci < cases.size(); ++ci) {
                const probe_case & pc = cases[ci];
                const bool as_f32 = pc.type != GGML_TYPE_F32 && laya_weight_dequantized(dev, precs[pi].laya, pc.type);
                const probe_result r = run_case(backend, dev, pc, precs[pi].prec, as_f32);
                const char * verdict = !r.supported ? "unsupported" : !r.computed ? "error" : r.pass ? "pass" : "FAIL";
                if (r.supported) {
                    if (!r.computed) {
                        broken = true;
                    } else if (!r.pass) {
                        (pc.type == GGML_TYPE_F32 ? f32_ok : f16_ok) = false;
                    }
                } else if (gated) {
                    broken = true;  // a device the laya graph runs on must take every mul_mat
                }
                printf("  %-7s %s x %s k=%-4d n=%-2d  max|err| %.3g  max rel %.3g  %s%s\n", precs[pi].name,
                       ggml_type_name(pc.type), ggml_type_name(pc.type), pc.k, pc.n, r.max_abs, r.max_rel, verdict,
                       as_f32 ? " (as f32)" : "");
                jcases += std::string(ci ? "," : "") + "\n     {\"type\": " + json_str(ggml_type_name(pc.type)) +
                          ", \"k\": " + std::to_string(pc.k) + ", \"n\": " + std::to_string(pc.n) +
                          ", \"as_f32\": " + (as_f32 ? "true" : "false") +
                          ", \"verdict\": " + json_str(verdict) + ", \"max_abs_err\": " + json_num(r.max_abs) +
                          ", \"max_rel_err\": " + json_num(r.max_rel) + "}";
            }
            const char * tier = broken ? "error" : f16_ok ? (f32_ok ? "strict-f32" : "f16-class") : "below-f16-class";
            printf("  %-7s tier: %s\n", precs[pi].name, tier);
            if (gated && (broken || !f16_ok)) {
                failed = true;
            }
            js += std::string(pi ? "," : "") + "\n   " + json_str(precs[pi].name) + ": {\"tier\": " + json_str(tier) +
                  ", \"cases\": [" + jcases + "]}";
        }
        js += "}}";
        ggml_backend_free(backend);
    }
    js += "\n ],\n \"n_gated\": " + std::to_string(n_gpu) + ",\n \"verdict\": " +
          json_str(n_gpu == 0 ? "skip" : failed ? "fail" : "pass") + "\n}\n";

    if (!json_path.empty()) {
        FILE * f = fopen(json_path.c_str(), "wb");
        if (!f) {
            fprintf(stderr, "cannot write %s\n", json_path.c_str());
            return 1;
        }
        fputs(js.c_str(), f);
        fclose(f);
    }

    if (n_gpu == 0) {
        printf("no GPU / iGPU device in this build: skipped\n");
        return 77;
    }
    printf("%s\n", failed ? "FAIL: a GPU device is below f16-class or cannot compute a case" : "OK: every GPU device is in a laya tier");
    return failed ? 1 : 0;
}
